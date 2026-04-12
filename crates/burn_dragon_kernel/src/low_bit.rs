use std::any::{Any, TypeId};
use std::collections::HashMap as StdHashMap;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Int, Shape, Tensor, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::NodeId;
use burn_autodiff::checkpoint::{
    base::Checkpointer,
    retro_forward::RetroForward,
    state::BackwardStates,
    strategy::{BalancedCheckpointing, CheckpointStrategy, NoCheckpointing},
};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_cubecl::BoolElement;
use burn_cubecl::CubeRuntime;
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::server::KernelArguments;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::{Client, FusionTensor};
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

use crate::fusion_compat::{register_fusion_float_tensor, register_fusion_int_tensor};

#[derive(Debug, Clone)]
pub struct PackedRhoInt8BlockDeviceTensors<B: BackendTrait> {
    pub packed: Tensor<B, 1, Int>,
    pub scales: Tensor<B, 1>,
}

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend<C = NoCheckpointing> = Autodiff<WgpuCubeBackend, C>;
type WgpuCubeAutodiffTensor<C = NoCheckpointing> =
    <WgpuCubeAutodiffBackend<C> as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend<C = NoCheckpointing> = Autodiff<CudaCubeBackend, C>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor<C = NoCheckpointing> =
    <CudaCubeAutodiffBackend<C> as BackendTrait>::FloatTensorPrimitive;

const WGPU_WORKGROUP_SIZE_X: u32 = 64;
const CUDA_WORKGROUP_SIZE_X: u32 = 128;
#[cfg(feature = "cuda")]
const CUDA_RAW_WORKGROUP_SIZE_X: u32 = 256;

#[derive(Clone)]
enum WgpuFloatOutputOrigin {
    Plain,
    FusionU32(Client<FusionCubeRuntime<WgpuRuntime>>),
    FusionU8(Client<FusionCubeRuntime<WgpuRuntime>>),
}

#[derive(Clone)]
enum WgpuIntOutputOrigin {
    Plain,
    FusionU32(Client<FusionCubeRuntime<WgpuRuntime>>),
    FusionU8(Client<FusionCubeRuntime<WgpuRuntime>>),
}

mod cuda;
mod training;
mod wgpu;

pub use self::cuda::{
    pack_decoder_input_codes_i8x4, pack_decoder_weight_codes_i8x4, pack_lowrank_input_codes_i8x4,
    pack_lowrank_weight_codes_i8x4, supports_packed_low_bit_device_backend,
    supports_packed_rho_int8_block_device_backend, try_raw_cuda_packed_decoder_tail,
    try_raw_cuda_packed_decoder_tail_device_scale, try_raw_cuda_packed_decoder_tail_grad_input,
    try_raw_cuda_packed_decoder_tail_grad_weight, try_raw_cuda_packed_decoder_tail_prepacked_input,
    try_raw_cuda_packed_decoder_tail_prepacked_input_device_scale,
    try_raw_cuda_packed_lowrank_grad_input, try_raw_cuda_packed_lowrank_grad_weight,
    try_raw_cuda_packed_lowrank_projection, try_raw_cuda_packed_lowrank_projection_device_scale,
    try_raw_cuda_packed_lowrank_projection_prepacked_input,
    try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale,
    try_raw_cuda_quantize_pack_activation_i8x4,
};
pub use self::training::{
    try_fused_packed_decoder_tail_training_autodiff, try_fused_packed_lowrank_training_autodiff,
    try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale,
};
pub use self::wgpu::{
    cached_wgpu_packed_dot_decoder_tail_support, cached_wgpu_packed_dot_lowrank_support,
    diagnose_wgpu_packed_dot_decoder_tail, diagnose_wgpu_packed_dot_lowrank_projection,
    diagnose_wgpu_quantize_pack_activation_i8x4, try_cube_fused_packed_decoder_tail_wgpu,
    try_cube_fused_packed_lowrank_projection_wgpu, try_wgpu_packed_dot_decoder_tail,
    try_wgpu_packed_dot_decoder_tail_device_scale,
    try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale,
    try_wgpu_packed_dot_lowrank_projection, try_wgpu_packed_dot_lowrank_projection_device_scale,
    try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale,
    try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale,
    try_wgpu_quantize_activation_codes_i32, try_wgpu_quantize_pack_activation_i8x4,
};

pub fn try_fused_packed_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let debug_wgpu = std::env::var_os("BDH_DEBUG_WGPU_LOWRANK_DIRECT").is_some();
    match diagnose_wgpu_packed_dot_lowrank_projection(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    ) {
        Ok(output) => return Some(output),
        Err(err) => {
            if debug_wgpu {
                eprintln!("[low-bit][wgpu][lowrank-packed-dot] {err}");
            }
        }
    }
    let direct_wgpu = try_direct_packed_lowrank_projection::<B, WgpuRuntime>(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    );
    if direct_wgpu.is_some() {
        return direct_wgpu;
    }
    if debug_wgpu {
        eprintln!("[low-bit][wgpu][lowrank-direct] unavailable");
    }
    #[cfg(feature = "cuda")]
    {
        return try_direct_packed_lowrank_projection::<B, CudaRuntime>(
            input_codes,
            weight_codes,
            activation_scale,
            weight_scale,
            latent_out,
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        None
    }
}

pub fn try_fused_packed_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    diagnose_wgpu_packed_dot_decoder_tail(y_codes, weight_codes, activation_scale, weight_scale)
        .ok()
        .or_else(|| {
            try_direct_packed_decoder_tail::<B, WgpuRuntime>(
                y_codes,
                weight_codes,
                activation_scale,
                weight_scale,
            )
        })
        .or_else(|| {
            #[cfg(feature = "cuda")]
            {
                try_direct_packed_decoder_tail::<B, CudaRuntime>(
                    y_codes,
                    weight_codes,
                    activation_scale,
                    weight_scale,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        })
}

pub fn try_fused_packed_lowrank_grad_input<B: BackendTrait>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 3, Int>,
    weight_scale: f32,
    input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_lowrank_grad_input::<B, WgpuRuntime>(
        grad_output,
        weight_codes,
        weight_scale,
        input_heads,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_packed_lowrank_grad_input::<B, CudaRuntime>(
                grad_output,
                weight_codes,
                weight_scale,
                input_heads,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub fn try_fused_packed_decoder_tail_grad_input<B: BackendTrait>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 2, Int>,
    weight_scale: f32,
    heads: usize,
    latent: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_decoder_tail_grad_input::<B, WgpuRuntime>(
        grad_output,
        weight_codes,
        weight_scale,
        heads,
        latent,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_packed_decoder_tail_grad_input::<B, CudaRuntime>(
                grad_output,
                weight_codes,
                weight_scale,
                heads,
                latent,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub fn try_fused_packed_lowrank_grad_weight<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_lowrank_grad_weight::<B, WgpuRuntime>(
        input_codes,
        grad_output,
        activation_scale,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_packed_lowrank_grad_weight::<B, CudaRuntime>(
                input_codes,
                grad_output,
                activation_scale,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub fn try_fused_packed_decoder_tail_grad_weight<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_decoder_tail_grad_weight::<B, WgpuRuntime>(
        y_codes,
        grad_output,
        activation_scale,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_raw_cuda_packed_decoder_tail_grad_weight(y_codes, grad_output, activation_scale)
                .or_else(|| {
                    try_direct_packed_decoder_tail_grad_weight::<B, CudaRuntime>(
                        y_codes,
                        grad_output,
                        activation_scale,
                    )
                })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub fn packed_lowrank_projection_device_reference<B: BackendTrait>(
    input: Tensor<B, 4>,
    weight_codes: Tensor<B, 3, Int>,
    weight_scale: f32,
    latent_out: usize,
) -> Tensor<B, 4> {
    let [batch, input_heads, time, embd] = input.shape().dims::<4>();
    let [artifact_heads, artifact_embd, artifact_latent] = weight_codes.shape().dims::<3>();
    assert!(
        input_heads == 1 || input_heads == artifact_heads,
        "packed low-rank device projection head mismatch: artifact={} input={}",
        artifact_heads,
        input_heads
    );
    assert_eq!(
        artifact_embd, embd,
        "packed low-rank device projection embd mismatch: artifact={} input={}",
        artifact_embd, embd
    );
    assert!(
        latent_out <= artifact_latent,
        "packed low-rank device projection latent mismatch: requested {} > artifact {}",
        latent_out,
        artifact_latent
    );

    let input = if input_heads == 1 && artifact_heads > 1 {
        input.repeat_dim(1, artifact_heads)
    } else {
        input
    };
    let weight = weight_codes
        .slice([0..artifact_heads, 0..artifact_embd, 0..latent_out])
        .float()
        .mul_scalar(weight_scale);
    let weight = weight.unsqueeze::<4>().repeat_dim(0, batch);
    let output = input.matmul(weight);
    output.reshape([batch, artifact_heads, time, latent_out])
}

pub fn packed_decoder_tail_device_reference<B: BackendTrait>(
    y_neuron: Tensor<B, 4>,
    weight_codes: Tensor<B, 2, Int>,
    weight_scale: f32,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let [artifact_latent_total, dim] = weight_codes.shape().dims::<2>();
    assert_eq!(
        artifact_latent_total % heads,
        0,
        "packed decoder tail device projection latent_total must divide across heads"
    );
    let artifact_latent_per_head = artifact_latent_total / heads;
    assert!(
        latent <= artifact_latent_per_head,
        "packed decoder tail device projection latent mismatch: requested {} > artifact {}",
        latent,
        artifact_latent_per_head
    );

    let weight = weight_codes
        .reshape([heads, artifact_latent_per_head, dim])
        .slice([0..heads, 0..latent, 0..dim])
        .float()
        .mul_scalar(weight_scale);
    let mixed_by_head = y_neuron
        .swap_dims(0, 1)
        .reshape([heads, batch * time, latent]);
    mixed_by_head
        .matmul(weight)
        .sum_dim(0)
        .reshape([batch, 1, time, dim])
}

pub fn packed_lowrank_grad_input_device_reference<B: BackendTrait>(
    grad_output: Tensor<B, 4>,
    weight_codes: Tensor<B, 3, Int>,
    weight_scale: f32,
    input_heads: usize,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [weight_heads, embd, weight_latent] = weight_codes.shape().dims::<3>();
    assert_eq!(heads, weight_heads);
    assert_eq!(latent, weight_latent);
    let input_heads = input_heads.max(1);
    if input_heads == 1 {
        let grad_flat = grad_output
            .clone()
            .swap_dims(1, 2)
            .reshape([batch * time, heads * latent]);
        let weight_flat = weight_codes
            .float()
            .mul_scalar(weight_scale)
            .swap_dims(0, 1)
            .reshape([embd, heads * latent]);
        grad_flat
            .matmul(weight_flat.swap_dims(0, 1))
            .reshape([batch, time, embd])
            .reshape([batch, 1, time, embd])
    } else {
        let grad_by_head =
            grad_output
                .clone()
                .swap_dims(0, 1)
                .reshape([heads, batch * time, latent]);
        let weight_by_head = weight_codes.float().mul_scalar(weight_scale);
        grad_by_head
            .matmul(weight_by_head.swap_dims(1, 2))
            .reshape([heads, batch, time, embd])
            .swap_dims(0, 1)
    }
}

#[cfg(feature = "cuda")]
pub fn packed_lowrank_grad_input_from_float_weight_cuda(
    grad_output: BurnTensor<CudaCubeBackend, 4>,
    weight: BurnTensor<CudaCubeBackend, 4>,
    input_heads: usize,
) -> BurnTensor<CudaCubeBackend, 4> {
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [_, weight_heads, embd, weight_latent] = weight.shape().dims::<4>();
    assert_eq!(heads, weight_heads);
    assert_eq!(latent, weight_latent);
    let weight_by_head = weight
        .slice([0..1, 0..heads, 0..embd, 0..latent])
        .reshape([heads, embd, latent]);
    if input_heads == 1 {
        let grad_flat = grad_output
            .swap_dims(1, 2)
            .reshape([batch * time, heads * latent]);
        let weight_flat = weight_by_head
            .swap_dims(0, 1)
            .reshape([embd, heads * latent]);
        grad_flat
            .matmul(weight_flat.swap_dims(0, 1))
            .reshape([batch, time, embd])
            .reshape([batch, 1, time, embd])
    } else {
        let grad_by_head =
            grad_output
                .clone()
                .swap_dims(0, 1)
                .reshape([heads, batch * time, latent]);
        grad_by_head
            .matmul(weight_by_head.swap_dims(1, 2))
            .reshape([heads, batch, time, embd])
            .swap_dims(0, 1)
    }
}

#[cfg(feature = "cuda")]
pub fn packed_lowrank_grad_input_from_transposed_float_weight_cuda(
    grad_output: BurnTensor<CudaCubeBackend, 4>,
    weight_t: BurnTensor<CudaCubeBackend, 3>,
    input_heads: usize,
) -> BurnTensor<CudaCubeBackend, 4> {
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [weight_heads, weight_latent, embd] = weight_t.shape().dims::<3>();
    assert_eq!(heads, weight_heads);
    assert_eq!(latent, weight_latent);
    if input_heads == 1 {
        grad_output
            .swap_dims(1, 2)
            .reshape([batch * time, heads * latent])
            .matmul(weight_t.reshape([heads * latent, embd]))
            .reshape([batch, time, embd])
            .reshape([batch, 1, time, embd])
    } else {
        grad_output
            .clone()
            .swap_dims(0, 1)
            .reshape([heads, batch * time, latent])
            .matmul(weight_t)
            .reshape([heads, batch, time, embd])
            .swap_dims(0, 1)
    }
}

pub fn packed_lowrank_grad_weight_device_reference<B: BackendTrait>(
    input_codes: Tensor<B, 4, Int>,
    grad_output: Tensor<B, 4>,
    activation_scale: f32,
) -> Tensor<B, 4> {
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [_, heads, _, latent] = grad_output.shape().dims::<4>();
    if input_heads == 1 {
        let input_flat = input_codes
            .float()
            .mul_scalar(activation_scale)
            .reshape([batch, time, embd])
            .reshape([batch * time, embd]);
        let grad_flat = grad_output
            .swap_dims(1, 2)
            .reshape([batch * time, heads * latent]);
        input_flat
            .swap_dims(0, 1)
            .matmul(grad_flat)
            .reshape([embd, heads, latent])
            .swap_dims(0, 1)
            .reshape([1, heads, embd, latent])
    } else {
        let input_by_head = input_codes
            .float()
            .mul_scalar(activation_scale)
            .swap_dims(0, 1)
            .reshape([heads, batch * time, embd]);
        let grad_by_head = grad_output
            .swap_dims(0, 1)
            .reshape([heads, batch * time, latent]);
        input_by_head
            .swap_dims(1, 2)
            .matmul(grad_by_head)
            .reshape([1, heads, embd, latent])
    }
}

pub fn packed_decoder_tail_grad_input_device_reference<B: BackendTrait>(
    grad_output: Tensor<B, 4>,
    weight_codes: Tensor<B, 2, Int>,
    weight_scale: f32,
    heads: usize,
    latent: usize,
) -> Tensor<B, 4> {
    let [batch, _, time, dim] = grad_output.shape().dims::<4>();
    let [latent_total, artifact_dim] = weight_codes.shape().dims::<2>();
    assert_eq!(dim, artifact_dim);
    assert_eq!(latent_total, heads * latent);
    let decoder_flat = weight_codes
        .float()
        .mul_scalar(weight_scale)
        .reshape([heads * latent, dim]);
    grad_output
        .reshape([batch * time, dim])
        .matmul(decoder_flat.swap_dims(0, 1))
        .reshape([batch, time, heads, latent])
        .swap_dims(1, 2)
}

#[cfg(feature = "cuda")]
pub fn packed_decoder_tail_grad_input_from_float_decoder_cuda(
    grad_output: BurnTensor<CudaCubeBackend, 4>,
    decoder: BurnTensor<CudaCubeBackend, 2>,
    heads: usize,
    latent: usize,
) -> BurnTensor<CudaCubeBackend, 4> {
    let [batch, _, time, dim] = grad_output.shape().dims::<4>();
    let decoder_flat = decoder.reshape([heads * latent, dim]);
    grad_output
        .reshape([batch * time, dim])
        .matmul(decoder_flat.swap_dims(0, 1))
        .reshape([batch, time, heads, latent])
        .swap_dims(1, 2)
}

pub fn packed_decoder_tail_grad_weight_device_reference<B: BackendTrait>(
    y_codes: Tensor<B, 4, Int>,
    grad_output: Tensor<B, 4>,
    activation_scale: f32,
) -> Tensor<B, 2> {
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let dim = grad_output.shape().dims::<4>()[3];
    let y_by_head = y_codes
        .float()
        .mul_scalar(activation_scale)
        .swap_dims(0, 1)
        .reshape([heads, batch * time, latent]);
    let grad_flat = grad_output.reshape([batch * time, dim]);
    let mut weights = Vec::with_capacity(heads);
    for head_idx in 0..heads {
        let y = y_by_head
            .clone()
            .slice([head_idx..head_idx + 1, 0..batch * time, 0..latent])
            .reshape([batch * time, latent]);
        weights.push(y.swap_dims(0, 1).matmul(grad_flat.clone()));
    }
    Tensor::cat(weights, 0).reshape([heads * latent, dim])
}

pub fn pack_rho_int8_block_device_reference<B: BackendTrait>(
    rho: Tensor<B, 4>,
    block_size: usize,
) -> PackedRhoInt8BlockDeviceTensors<B> {
    assert!(block_size > 0, "rho int8 block size must be positive");
    let device = rho.device();
    let values = rho
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho values");
    let mut packed = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len().div_ceil(block_size));

    for block in values.chunks(block_size) {
        let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let scale = (max_abs / 127.0).max(1.0e-8);
        scales.push(scale);
        for value in block {
            packed.push((value / scale).round().clamp(-127.0, 127.0) as i64);
        }
    }

    PackedRhoInt8BlockDeviceTensors {
        packed: Tensor::<B, 1, Int>::from_data(TensorData::new(packed, [values.len()]), &device),
        scales: Tensor::<B, 1>::from_data(TensorData::new(scales.clone(), [scales.len()]), &device),
    }
}

pub fn unpack_rho_int8_block_device_reference<B: BackendTrait>(
    packed: Tensor<B, 1, Int>,
    scales: Tensor<B, 1>,
    logical_shape: [usize; 4],
    block_size: usize,
) -> Tensor<B, 4> {
    assert!(block_size > 0, "rho int8 block size must be positive");
    let device = packed.device();
    let packed_values = packed
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("packed rho values");
    let scale_values = scales
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho scales");
    let mut values = Vec::with_capacity(packed_values.len());

    for (block_index, block) in packed_values.chunks(block_size).enumerate() {
        let scale = scale_values.get(block_index).copied().unwrap_or(1.0e-8);
        for value in block {
            values.push(*value as f32 * scale);
        }
    }

    Tensor::<B, 4>::from_data(TensorData::new(values, logical_shape), &device)
}

fn try_direct_packed_lowrank_projection<B, R>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, weight_embd, artifact_latent] = weight_codes.shape().dims::<3>();
    if weight_embd != embd
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return None;
    }
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let debug_wgpu = std::env::var_os("BDH_DEBUG_WGPU_LOWRANK_DIRECT").is_some();
        let input: CubeTensor<WgpuRuntime> = match resolve_wgpu_int_tensor_read::<B, 4>(input_codes)
        {
            Some(input) => input,
            None => {
                if debug_wgpu {
                    eprintln!(
                        "[low-bit][wgpu][lowrank-direct] input cast failed for backend {}",
                        core::any::type_name::<B>()
                    );
                }
                return None;
            }
        };
        let weight: CubeTensor<WgpuRuntime> =
            match resolve_wgpu_int_tensor_read::<B, 3>(weight_codes) {
                Some(weight) => weight,
                None => {
                    if debug_wgpu {
                        eprintln!(
                            "[low-bit][wgpu][lowrank-direct] weight cast failed for backend {}",
                            core::any::type_name::<B>()
                        );
                    }
                    return None;
                }
            };
        if input.dtype != DType::I32 || weight.dtype != DType::I32 {
            if debug_wgpu {
                eprintln!(
                    "[low-bit][wgpu][lowrank-direct] dtype mismatch input={:?} weight={:?}",
                    input.dtype, weight.dtype
                );
            }
            return None;
        }
        let params = BurnTensor::<B, 1>::from_floats(
            [
                batch as f32,
                input_heads as f32,
                heads as f32,
                time as f32,
                embd as f32,
                latent_out as f32,
                activation_scale,
                weight_scale,
            ],
            &input_codes.device(),
        );
        let (params, output_origin) =
            match resolve_wgpu_float_tensor_with_output_origin::<B, 1>(&params) {
                Some(resolved) => resolved,
                None => {
                    if debug_wgpu {
                        eprintln!(
                            "[low-bit][wgpu][lowrank-direct] params resolve failed for backend {}",
                            core::any::type_name::<B>()
                        );
                    }
                    return None;
                }
            };
        let output = packed_lowrank_projection_cube_runtime::<WgpuRuntime>(
            input, weight, params, batch, heads, time, latent_out,
        );
        let wrapped = wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin);
        if wrapped.is_none() && debug_wgpu {
            eprintln!(
                "[low-bit][wgpu][lowrank-direct] output wrap failed for backend {}",
                core::any::type_name::<B>()
            );
        }
        return wrapped;
    }
    let input: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let weight: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            activation_scale,
            weight_scale,
        ],
        &input_codes.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output = packed_lowrank_projection_cube_runtime::<R>(
        input, weight, params, batch, heads, time, latent_out,
    );
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_direct_packed_decoder_tail<B, R>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [artifact_latent_total, dim] = weight_codes.shape().dims::<2>();
    if artifact_latent_total % heads != 0 {
        return None;
    }
    let artifact_latent_per_head = artifact_latent_total / heads;
    if latent > artifact_latent_per_head {
        return None;
    }
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let y = resolve_wgpu_int_tensor_read::<B, 4>(y_codes)?;
        let weight = resolve_wgpu_int_tensor_read::<B, 2>(weight_codes)?;
        if y.dtype != DType::I32 || weight.dtype != DType::I32 {
            return None;
        }
        let params = BurnTensor::<B, 1>::from_floats(
            [
                batch as f32,
                heads as f32,
                time as f32,
                latent as f32,
                artifact_latent_per_head as f32,
                dim as f32,
                activation_scale,
                weight_scale,
            ],
            &y_codes.device(),
        );
        let (params, output_origin) =
            resolve_wgpu_float_tensor_with_output_origin::<B, 1>(&params)?;
        let output =
            packed_decoder_tail_cube_runtime::<WgpuRuntime>(y, weight, params, batch, time, dim);
        return wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin);
    }
    let y: CubeTensor<R> = try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let weight: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            artifact_latent_per_head as f32,
            dim as f32,
            activation_scale,
            weight_scale,
        ],
        &y_codes.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output = packed_decoder_tail_cube_runtime::<R>(y, weight, params, batch, time, dim);
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_direct_packed_lowrank_grad_input<B, R>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 3, Int>,
    weight_scale: f32,
    input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [weight_heads, embd, weight_latent] = weight_codes.shape().dims::<3>();
    if heads != weight_heads
        || latent != weight_latent
        || !(input_heads == 1 || input_heads == heads)
    {
        return None;
    }
    let grad: CubeTensor<R> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent as f32,
            weight_scale,
        ],
        &grad_output.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output = packed_lowrank_grad_input_cube_runtime::<R>(
        grad,
        weight,
        params,
        batch,
        input_heads,
        time,
        embd,
    );
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_direct_packed_lowrank_grad_weight<B, R>(
    input_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [_, heads, _, latent] = grad_output.shape().dims::<4>();
    if !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    let input: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let grad: CubeTensor<R> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent as f32,
            activation_scale,
        ],
        &input_codes.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output =
        packed_lowrank_grad_weight_cube_runtime::<R>(input, grad, params, heads, embd, latent);
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_direct_packed_decoder_tail_grad_input<B, R>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 2, Int>,
    weight_scale: f32,
    heads: usize,
    latent: usize,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, _, time, dim] = grad_output.shape().dims::<4>();
    let [latent_total, weight_dim] = weight_codes.shape().dims::<2>();
    if dim != weight_dim || latent_total != heads * latent {
        return None;
    }
    let grad: CubeTensor<R> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<R> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            dim as f32,
            weight_scale,
        ],
        &grad_output.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output = packed_decoder_tail_grad_input_cube_runtime::<R>(
        grad, weight, params, batch, heads, time, latent,
    );
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_direct_packed_decoder_tail_grad_weight<B, R>(
    y_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let dim = grad_output.shape().dims::<4>()[3];
    let y: CubeTensor<R> = try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let grad: CubeTensor<R> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if y.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let params = BurnTensor::<B, 1>::from_floats(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            dim as f32,
            activation_scale,
        ],
        &y_codes.device(),
    );
    let params: CubeTensor<R> = try_cast_float_primitive::<B, _>(params.into_primitive().tensor())?;
    let output =
        packed_decoder_tail_grad_weight_cube_runtime::<R>(y, grad, params, heads, latent, dim);
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn packed_lowrank_projection_cube_runtime<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    params: CubeTensor<R>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let params = into_contiguous(params);
    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let cube_dim_x = lowrank_grad_input_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(latent as u32, cube_dim_x),
        time as u32,
        (batch * heads) as u32,
    );
    let _ = packed_lowrank_projection_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        input.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

fn packed_decoder_tail_cube_runtime<R: CubeRuntime>(
    y: CubeTensor<R>,
    weight: CubeTensor<R>,
    params: CubeTensor<R>,
    _batch: usize,
    _time: usize,
    _dim: usize,
) -> CubeTensor<R> {
    let y = into_contiguous(y);
    let weight = into_contiguous(weight);
    let params = into_contiguous(params);
    let batch = y.meta.shape.dims::<4>()[0];
    let time = y.meta.shape.dims::<4>()[2];
    let dim = weight.meta.shape.dims::<2>()[1];
    let client = y.client.clone();
    let device = y.device.clone();
    let output = empty_device::<R, f32>(client.clone(), device, Shape::new([batch, 1, time, dim]));
    let cube_dim_x = lowrank_grad_weight_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(dim as u32, cube_dim_x),
        time as u32,
        batch as u32,
    );
    let _ = packed_decoder_tail_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        y.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

fn packed_lowrank_grad_input_cube_runtime<R: CubeRuntime>(
    grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    params: CubeTensor<R>,
    batch: usize,
    input_heads: usize,
    time: usize,
    embd: usize,
) -> CubeTensor<R> {
    let grad = into_contiguous(grad);
    let weight = into_contiguous(weight);
    let params = into_contiguous(params);
    let client = grad.client.clone();
    let device = grad.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, input_heads, time, embd]),
    );
    let cube_dim_x = decoder_tail_grad_input_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, cube_dim_x),
        time as u32,
        (batch * input_heads.max(1)) as u32,
    );
    let _ = packed_lowrank_grad_input_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

fn packed_lowrank_grad_weight_cube_runtime<R: CubeRuntime>(
    input: CubeTensor<R>,
    grad: CubeTensor<R>,
    params: CubeTensor<R>,
    heads: usize,
    embd: usize,
    latent: usize,
) -> CubeTensor<R> {
    let input = into_contiguous(input);
    let grad = into_contiguous(grad);
    let params = into_contiguous(params);
    let client = input.client.clone();
    let device = input.device.clone();
    let output =
        empty_device::<R, f32>(client.clone(), device, Shape::new([1, heads, embd, latent]));
    let cube_dim_x = decoder_tail_grad_weight_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(latent as u32, cube_dim_x),
        embd as u32,
        heads as u32,
    );
    let _ = packed_lowrank_grad_weight_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        input.clone().into_tensor_arg(),
        grad.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

fn packed_decoder_tail_grad_input_cube_runtime<R: CubeRuntime>(
    grad: CubeTensor<R>,
    weight: CubeTensor<R>,
    params: CubeTensor<R>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
) -> CubeTensor<R> {
    let grad = into_contiguous(grad);
    let weight = into_contiguous(weight);
    let params = into_contiguous(params);
    let client = grad.client.clone();
    let device = grad.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let cube_dim_x = cube_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(latent as u32, cube_dim_x),
        time as u32,
        (batch * heads) as u32,
    );
    let _ = packed_decoder_tail_grad_input_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

fn packed_decoder_tail_grad_weight_cube_runtime<R: CubeRuntime>(
    y: CubeTensor<R>,
    grad: CubeTensor<R>,
    params: CubeTensor<R>,
    heads: usize,
    latent: usize,
    dim: usize,
) -> CubeTensor<R> {
    let y = into_contiguous(y);
    let grad = into_contiguous(grad);
    let params = into_contiguous(params);
    let client = y.client.clone();
    let device = y.device.clone();
    let output = empty_device::<R, f32>(client.clone(), device, Shape::new([heads * latent, dim]));
    let cube_dim_x = cube_workgroup_size_x::<R>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32(dim as u32, cube_dim_x),
        (heads * latent) as u32,
        1,
    );
    let _ = packed_decoder_tail_grad_weight_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        y.clone().into_tensor_arg(),
        grad.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    output
}

#[cube(launch)]
fn quantize_pack_i8x4_cube_kernel(
    input: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    activation_scale: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<i32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let outer = u32::cast_from(params[0]) as usize;
    let inner = u32::cast_from(params[1]) as usize;
    let pack_len = u32::cast_from(params[2]) as usize;
    let qmax = i32::cast_from(params[3]);
    let positive_only = u32::cast_from(params[4]) != 0u32;

    let packed_idx = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if packed_idx >= outer * pack_len {
        terminate!();
    }

    let outer_idx = packed_idx / pack_len;
    let pack_offset = packed_idx % pack_len;
    let base = outer_idx * inner;
    let value_offset = pack_offset * 4usize;
    let total_values = outer * inner;
    let scale = activation_scale[0];
    let zero = f32::cast_from(0u32);
    let one = f32::cast_from(1u32);
    let half = one / f32::cast_from(2u32);
    let inv_scale = if scale > zero { one / scale } else { zero };
    let idx0 = value_offset;
    let idx1 = value_offset + 1usize;
    let idx2 = value_offset + 2usize;
    let idx3 = value_offset + 3usize;

    let mut v0 = 0i32;
    let abs0 = base + idx0;
    if abs0 < total_values {
        let mut raw = input[abs0];
        if positive_only && raw < zero {
            raw = zero;
        }
        let shifted = if raw >= zero {
            raw * inv_scale + half
        } else {
            raw * inv_scale - half
        };
        v0 = i32::cast_from(shifted);
        if positive_only {
            if v0 < 0i32 {
                v0 = 0i32;
            }
            if v0 > qmax {
                v0 = qmax;
            }
        } else {
            if v0 < -qmax {
                v0 = -qmax;
            }
            if v0 > qmax {
                v0 = qmax;
            }
        }
    }

    let mut v1 = 0i32;
    let abs1 = base + idx1;
    if abs1 < total_values {
        let mut raw = input[abs1];
        if positive_only && raw < zero {
            raw = zero;
        }
        let shifted = if raw >= zero {
            raw * inv_scale + half
        } else {
            raw * inv_scale - half
        };
        v1 = i32::cast_from(shifted);
        if positive_only {
            if v1 < 0i32 {
                v1 = 0i32;
            }
            if v1 > qmax {
                v1 = qmax;
            }
        } else {
            if v1 < -qmax {
                v1 = -qmax;
            }
            if v1 > qmax {
                v1 = qmax;
            }
        }
    }

    let mut v2 = 0i32;
    let abs2 = base + idx2;
    if abs2 < total_values {
        let mut raw = input[abs2];
        if positive_only && raw < zero {
            raw = zero;
        }
        let shifted = if raw >= zero {
            raw * inv_scale + half
        } else {
            raw * inv_scale - half
        };
        v2 = i32::cast_from(shifted);
        if positive_only {
            if v2 < 0i32 {
                v2 = 0i32;
            }
            if v2 > qmax {
                v2 = qmax;
            }
        } else {
            if v2 < -qmax {
                v2 = -qmax;
            }
            if v2 > qmax {
                v2 = qmax;
            }
        }
    }

    let mut v3 = 0i32;
    let abs3 = base + idx3;
    if abs3 < total_values {
        let mut raw = input[abs3];
        if positive_only && raw < zero {
            raw = zero;
        }
        let shifted = if raw >= zero {
            raw * inv_scale + half
        } else {
            raw * inv_scale - half
        };
        v3 = i32::cast_from(shifted);
        if positive_only {
            if v3 < 0i32 {
                v3 = 0i32;
            }
            if v3 > qmax {
                v3 = qmax;
            }
        } else {
            if v3 < -qmax {
                v3 = -qmax;
            }
            if v3 > qmax {
                v3 = qmax;
            }
        }
    }

    let b0 = if v0 < 0i32 { v0 + 256i32 } else { v0 };
    let b1 = if v1 < 0i32 { v1 + 256i32 } else { v1 };
    let b2 = if v2 < 0i32 { v2 + 256i32 } else { v2 };
    let packed = b0 + b1 * 256i32 + b2 * 65536i32 + v3 * 16777216i32;
    output[packed_idx] = packed;
}

#[cube(launch)]
fn quantize_codes_i32_cube_kernel(
    input: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    activation_scale: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<i32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let total = u32::cast_from(params[0]) as usize;
    let qmax = i32::cast_from(params[1]);
    let positive_only = u32::cast_from(params[2]) != 0u32;

    let group_idx = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let base_idx = group_idx * 4usize;
    if base_idx >= total {
        terminate!();
    }

    let scale = activation_scale[0];
    let zero = f32::cast_from(0u32);
    let one = f32::cast_from(1u32);
    let half = one / f32::cast_from(2u32);
    let inv_scale = if scale > zero { one / scale } else { zero };

    let mut lane = 0usize;
    while lane < 4usize {
        let idx = base_idx + lane;
        if idx < total {
            let mut raw = input[idx];
            if positive_only && raw < zero {
                raw = zero;
            }
            let shifted = if raw >= zero {
                raw * inv_scale + half
            } else {
                raw * inv_scale - half
            };
            let mut code = i32::cast_from(shifted);
            if positive_only {
                if code < 0i32 {
                    code = 0i32;
                }
                if code > qmax {
                    code = qmax;
                }
            } else {
                if code < -qmax {
                    code = -qmax;
                }
                if code > qmax {
                    code = qmax;
                }
            }
            output[idx] = code;
        }
        lane += 1usize;
    }
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn env_workgroup_size(var: &str) -> Option<u32> {
    std::env::var(var)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|value| *value > 0)
}

#[cfg(feature = "cuda")]
fn raw_cuda_workgroup_size_x() -> u32 {
    env_workgroup_size("LOW_BIT_CUDA_RAW_WORKGROUP_SIZE_X").unwrap_or(CUDA_RAW_WORKGROUP_SIZE_X)
}

fn cube_workgroup_size_x<R: CubeRuntime>() -> u32 {
    if core::any::type_name::<R>().contains("CudaRuntime") {
        env_workgroup_size("LOW_BIT_CUDA_WORKGROUP_SIZE_X").unwrap_or(CUDA_WORKGROUP_SIZE_X)
    } else {
        env_workgroup_size("LOW_BIT_WGPU_WORKGROUP_SIZE_X").unwrap_or(WGPU_WORKGROUP_SIZE_X)
    }
}

fn lowrank_grad_input_workgroup_size_x<R: CubeRuntime>() -> u32 {
    if core::any::type_name::<R>().contains("CudaRuntime") {
        env_workgroup_size("LOW_BIT_CUDA_GRAD_INPUT_WORKGROUP_SIZE_X")
            .unwrap_or(cube_workgroup_size_x::<R>())
    } else {
        cube_workgroup_size_x::<R>()
    }
}

fn lowrank_grad_weight_workgroup_size_x<R: CubeRuntime>() -> u32 {
    if core::any::type_name::<R>().contains("CudaRuntime") {
        env_workgroup_size("LOW_BIT_CUDA_GRAD_WEIGHT_WORKGROUP_SIZE_X")
            .unwrap_or(cube_workgroup_size_x::<R>())
    } else {
        cube_workgroup_size_x::<R>()
    }
}

fn decoder_tail_grad_input_workgroup_size_x<R: CubeRuntime>() -> u32 {
    if core::any::type_name::<R>().contains("CudaRuntime") {
        env_workgroup_size("LOW_BIT_CUDA_DECODER_GRAD_INPUT_WORKGROUP_SIZE_X")
            .unwrap_or(cube_workgroup_size_x::<R>())
    } else {
        cube_workgroup_size_x::<R>()
    }
}

fn decoder_tail_grad_weight_workgroup_size_x<R: CubeRuntime>() -> u32 {
    if core::any::type_name::<R>().contains("CudaRuntime") {
        env_workgroup_size("LOW_BIT_CUDA_DECODER_GRAD_WEIGHT_WORKGROUP_SIZE_X")
            .unwrap_or(cube_workgroup_size_x::<R>())
    } else {
        cube_workgroup_size_x::<R>()
    }
}

fn resolve_wgpu_fusion_float_tensor<B, BT, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<(
    CubeTensor<WgpuRuntime>,
    Client<FusionCubeRuntime<WgpuRuntime>>,
)>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
        try_cast_float_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some((cube, client))
}

fn resolve_wgpu_fusion_int_tensor<B, BT, const D: usize>(
    tensor: &BurnTensor<B, D, Int>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
        try_cast_int_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_int::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::I32 {
        return None;
    }
    Some(cube)
}

fn resolve_wgpu_float_tensor_with_output_origin<B, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<(CubeTensor<WgpuRuntime>, WgpuFloatOutputOrigin)>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
{
    if let Some(cube) = try_cast_float_primitive::<B, CubeTensor<WgpuRuntime>>(
        tensor.clone().into_primitive().tensor(),
    ) {
        if cube.dtype == DType::F32 {
            return Some((cube, WgpuFloatOutputOrigin::Plain));
        }
    }
    if let Some((cube, client)) = resolve_wgpu_fusion_float_tensor::<B, u32, D>(tensor) {
        return Some((cube, WgpuFloatOutputOrigin::FusionU32(client)));
    }
    if let Some((cube, client)) = resolve_wgpu_fusion_float_tensor::<B, u8, D>(tensor) {
        return Some((cube, WgpuFloatOutputOrigin::FusionU8(client)));
    }
    None
}

fn resolve_wgpu_float_tensor_read<B, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
{
    resolve_wgpu_float_tensor_with_output_origin::<B, D>(tensor).map(|(cube, _)| cube)
}

fn int_output_origin_from_float_origin(origin: &WgpuFloatOutputOrigin) -> WgpuIntOutputOrigin {
    match origin {
        WgpuFloatOutputOrigin::Plain => WgpuIntOutputOrigin::Plain,
        WgpuFloatOutputOrigin::FusionU32(client) => WgpuIntOutputOrigin::FusionU32(client.clone()),
        WgpuFloatOutputOrigin::FusionU8(client) => WgpuIntOutputOrigin::FusionU8(client.clone()),
    }
}

fn resolve_wgpu_float_output_origin_from_device<B: BackendTrait>(
    device: &B::Device,
) -> Option<WgpuFloatOutputOrigin>
where
    B::FloatTensorPrimitive: 'static,
{
    let marker = BurnTensor::<B, 1>::from_floats([0.0], device);
    resolve_wgpu_float_tensor_with_output_origin::<B, 1>(&marker).map(|(_, origin)| origin)
}

fn resolve_wgpu_int_tensor_read<B, const D: usize>(
    tensor: &BurnTensor<B, D, Int>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
{
    if let Some(cube) =
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(tensor.clone().into_primitive())
    {
        if cube.dtype == DType::I32 {
            return Some(cube);
        }
    }
    resolve_wgpu_fusion_int_tensor::<B, u32, D>(tensor)
        .or_else(|| resolve_wgpu_fusion_int_tensor::<B, u8, D>(tensor))
}

fn wrap_wgpu_float_output_for_backend<B: BackendTrait, const D: usize>(
    output: CubeTensor<WgpuRuntime>,
    origin: WgpuFloatOutputOrigin,
) -> Option<BurnTensor<B, D>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim = match origin {
        WgpuFloatOutputOrigin::Plain => try_cast_float_backend::<B, _>(output)?,
        WgpuFloatOutputOrigin::FusionU32(client) => {
            let fusion = register_fusion_float_tensor(&client, output);
            try_cast_float_backend::<B, _>(fusion)?
        }
        WgpuFloatOutputOrigin::FusionU8(client) => {
            let fusion = register_fusion_float_tensor(&client, output);
            try_cast_float_backend::<B, _>(fusion)?
        }
    };
    Some(BurnTensor::<B, D>::from_primitive(TensorPrimitive::Float(
        prim,
    )))
}

fn wrap_wgpu_int_output_for_backend<B: BackendTrait, const D: usize>(
    output: CubeTensor<WgpuRuntime>,
    origin: WgpuIntOutputOrigin,
) -> Option<BurnTensor<B, D, Int>>
where
    B::IntTensorPrimitive: 'static,
{
    let prim = match origin {
        WgpuIntOutputOrigin::Plain => try_cast_int_backend::<B, _>(output)?,
        WgpuIntOutputOrigin::FusionU32(client) => {
            let fusion = register_fusion_int_tensor(&client, output);
            try_cast_int_backend::<B, _>(fusion)?
        }
        WgpuIntOutputOrigin::FusionU8(client) => {
            let fusion = register_fusion_int_tensor(&client, output);
            try_cast_int_backend::<B, _>(fusion)?
        }
    };
    Some(BurnTensor::<B, D, Int>::from_primitive(prim))
}

fn try_cast_int_primitive<B: BackendTrait, T: 'static>(value: B::IntTensorPrimitive) -> Option<T>
where
    B::IntTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_float_primitive<B: BackendTrait, T: 'static>(
    value: B::FloatTensorPrimitive,
) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_float_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

fn try_cast_int_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::IntTensorPrimitive>
where
    B::IntTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::IntTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

#[cube(launch)]
fn packed_lowrank_projection_cube_kernel(
    input: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let input_heads = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;
    let latent = u32::cast_from(params[5]) as usize;
    let activation_scale = params[6];
    let weight_scale = params[7];

    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let bh = CUBE_POS_Z as usize;
    if l >= latent || t >= time || bh >= batch * heads {
        terminate!();
    }

    let h = bh % heads;
    let b = bh / heads;
    let mut input_head = h;
    if input_heads == 1usize {
        input_head = 0usize;
    }

    let mut acc = i32::cast_from(0u32);
    let input_base = ((b * input_heads + input_head) * time + t) * embd;
    let weight_base = h * embd * latent + l;
    let mut e = 0usize;
    while e + 4usize <= embd {
        let weight_index = weight_base + e * latent;
        acc += input[input_base + e] * weight[weight_index];
        acc += input[input_base + e + 1usize] * weight[weight_index + latent];
        acc += input[input_base + e + 2usize] * weight[weight_index + latent * 2usize];
        acc += input[input_base + e + 3usize] * weight[weight_index + latent * 3usize];
        e += 4usize;
    }
    while e < embd {
        let weight_index = weight_base + e * latent;
        acc += input[input_base + e] * weight[weight_index];
        e += 1usize;
    }

    let output_index = ((b * heads + h) * time + t) * latent + l;
    output[output_index] = f32::cast_from(acc) * activation_scale * weight_scale;
}

#[cube(launch)]
fn packed_decoder_tail_cube_kernel(
    y: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let artifact_latent_per_head = u32::cast_from(params[4]) as usize;
    let dim = u32::cast_from(params[5]) as usize;
    let activation_scale = params[6];
    let weight_scale = params[7];

    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let b = CUBE_POS_Z as usize;
    if d >= dim || t >= time || b >= batch {
        terminate!();
    }

    let mut acc = i32::cast_from(0u32);
    let mut h = 0usize;
    while h < heads {
        let input_base = ((b * heads + h) * time + t) * latent;
        let weight_base = (h * artifact_latent_per_head) * dim;
        let mut l = 0usize;
        while l + 4usize <= latent {
            let weight_index = weight_base + l * dim + d;
            acc += y[input_base + l] * weight[weight_index];
            acc += y[input_base + l + 1usize] * weight[weight_index + dim];
            acc += y[input_base + l + 2usize] * weight[weight_index + dim * 2usize];
            acc += y[input_base + l + 3usize] * weight[weight_index + dim * 3usize];
            l += 4usize;
        }
        while l < latent {
            acc += y[input_base + l] * weight[weight_base + l * dim + d];
            l += 1usize;
        }
        h += 1usize;
    }

    output[(b * time + t) * dim + d] = f32::cast_from(acc) * activation_scale * weight_scale;
}

#[cube(launch)]
fn packed_lowrank_grad_input_cube_kernel(
    grad: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let input_heads = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;
    let latent = u32::cast_from(params[5]) as usize;
    let weight_scale = params[6];

    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let bih = CUBE_POS_Z as usize;
    if e >= embd || t >= time || bih >= batch * input_heads {
        terminate!();
    }

    let input_head = bih % input_heads;
    let b = bih / input_heads;
    let mut acc = f32::cast_from(0u32);
    if input_heads == 1usize {
        let mut h = 0usize;
        while h < heads {
            let mut l = 0usize;
            while l + 4usize <= latent {
                let grad_index = ((b * heads + h) * time + t) * latent + l;
                let weight_index = (h * embd + e) * latent + l;
                acc += grad[grad_index] * f32::cast_from(weight[weight_index]);
                acc += grad[grad_index + 1usize] * f32::cast_from(weight[weight_index + 1usize]);
                acc += grad[grad_index + 2usize] * f32::cast_from(weight[weight_index + 2usize]);
                acc += grad[grad_index + 3usize] * f32::cast_from(weight[weight_index + 3usize]);
                l += 4usize;
            }
            while l < latent {
                let grad_index = ((b * heads + h) * time + t) * latent + l;
                let weight_index = (h * embd + e) * latent + l;
                acc += grad[grad_index] * f32::cast_from(weight[weight_index]);
                l += 1usize;
            }
            h += 1usize;
        }
    } else {
        let h = input_head;
        let mut l = 0usize;
        while l + 4usize <= latent {
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            let weight_index = (h * embd + e) * latent + l;
            acc += grad[grad_index] * f32::cast_from(weight[weight_index]);
            acc += grad[grad_index + 1usize] * f32::cast_from(weight[weight_index + 1usize]);
            acc += grad[grad_index + 2usize] * f32::cast_from(weight[weight_index + 2usize]);
            acc += grad[grad_index + 3usize] * f32::cast_from(weight[weight_index + 3usize]);
            l += 4usize;
        }
        while l < latent {
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            let weight_index = (h * embd + e) * latent + l;
            acc += grad[grad_index] * f32::cast_from(weight[weight_index]);
            l += 1usize;
        }
    }

    let output_index = ((b * input_heads + input_head) * time + t) * embd + e;
    output[output_index] = acc * weight_scale;
}

#[cube(launch)]
fn packed_lowrank_grad_weight_cube_kernel(
    input: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    grad: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let input_heads = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;
    let latent = u32::cast_from(params[5]) as usize;
    let activation_scale = params[6];

    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let e = CUBE_POS_Y as usize;
    let h = CUBE_POS_Z as usize;
    if l >= latent || e >= embd || h >= heads {
        terminate!();
    }

    let mut input_head = h;
    if input_heads == 1usize {
        input_head = 0usize;
    }
    let mut acc = f32::cast_from(0u32);
    let mut b = 0usize;
    while b < batch {
        let mut t = 0usize;
        while t < time {
            let input_index = ((b * input_heads + input_head) * time + t) * embd + e;
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            acc += f32::cast_from(input[input_index]) * grad[grad_index];
            t += 1usize;
        }
        b += 1usize;
    }

    let output_index = (h * embd + e) * latent + l;
    output[output_index] = acc * activation_scale;
}

#[cube(launch)]
fn packed_decoder_tail_grad_input_cube_kernel(
    grad: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dim = u32::cast_from(params[4]) as usize;
    let weight_scale = params[5];

    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let t = CUBE_POS_Y as usize;
    let bh = CUBE_POS_Z as usize;
    if l >= latent || t >= time || bh >= batch * heads {
        terminate!();
    }
    let h = bh % heads;
    let b = bh / heads;
    let weight_row_base = (h * latent + l) * dim;
    let grad_base = (b * time + t) * dim;
    let mut acc = f32::cast_from(0u32);
    let mut d = 0usize;
    while d + 4usize <= dim {
        acc += grad[grad_base + d] * f32::cast_from(weight[weight_row_base + d]);
        acc += grad[grad_base + d + 1usize] * f32::cast_from(weight[weight_row_base + d + 1usize]);
        acc += grad[grad_base + d + 2usize] * f32::cast_from(weight[weight_row_base + d + 2usize]);
        acc += grad[grad_base + d + 3usize] * f32::cast_from(weight[weight_row_base + d + 3usize]);
        d += 4usize;
    }
    while d < dim {
        acc += grad[grad_base + d] * f32::cast_from(weight[weight_row_base + d]);
        d += 1usize;
    }
    let output_index = ((b * heads + h) * time + t) * latent + l;
    output[output_index] = acc * weight_scale;
}

#[cube(launch)]
fn packed_decoder_tail_grad_weight_cube_kernel(
    y: &burn_cubecl::cubecl::prelude::Tensor<i32>,
    grad: &burn_cubecl::cubecl::prelude::Tensor<f32>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<f32>,
    params: &burn_cubecl::cubecl::prelude::Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let dim = u32::cast_from(params[4]) as usize;
    let activation_scale = params[5];

    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let hl = CUBE_POS_Y as usize;
    if d >= dim || hl >= heads * latent {
        terminate!();
    }
    let h = hl / latent;
    let l = hl % latent;
    let mut acc = f32::cast_from(0u32);
    let mut b = 0usize;
    while b < batch {
        let mut t = 0usize;
        while t < time {
            let y_index = ((b * heads + h) * time + t) * latent + l;
            let grad_index = (b * time + t) * dim + d;
            acc += f32::cast_from(y[y_index]) * grad[grad_index];
            t += 1usize;
        }
        b += 1usize;
    }

    output[hl * dim + d] = acc * activation_scale;
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};
    use burn_ndarray::NdArray;
    use burn_wgpu::{RuntimeOptions, graphics};

    type NdBackend = NdArray<f32>;
    type WgpuBackend = WgpuCubeBackend;

    fn init_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn quantize_signed_codes<B: BackendTrait, const D: usize>(
        tensor: Tensor<B, D>,
    ) -> (Tensor<B, D, Int>, f32) {
        let logical_shape = tensor.shape().dims::<D>();
        let device = tensor.device();
        let values = tensor
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("values");
        let mean_abs = if values.is_empty() {
            0.0
        } else {
            values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
        };
        let scale = (mean_abs * 2.0 / 127.0).max(1.0e-8);
        let codes = values
            .into_iter()
            .map(|value| (value / scale).round().clamp(-127.0, 127.0) as i64)
            .collect::<Vec<_>>();
        (
            Tensor::<B, D, Int>::from_data(TensorData::new(codes, logical_shape), &device),
            scale,
        )
    }

    fn assert_close<const D: usize, B: BackendTrait>(
        lhs: BurnTensor<B, D>,
        rhs: BurnTensor<B, D>,
        atol: f32,
        rtol: f32,
    ) {
        let lhs = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs");
        let rhs = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs");
        assert_eq!(lhs.len(), rhs.len());
        for (index, (lhs, rhs)) in lhs.into_iter().zip(rhs.into_iter()).enumerate() {
            let diff = (lhs - rhs).abs();
            let limit = atol + rtol * rhs.abs();
            assert!(
                diff <= limit,
                "mismatch at {index}: lhs={lhs} rhs={rhs} diff={diff} limit={limit}"
            );
        }
    }

    #[test]
    fn packed_lowrank_projection_device_reference_runs() {
        let device = Default::default();
        let input = Tensor::<NdBackend, 4>::from_data(
            TensorData::new(vec![0.5, -1.0, 0.25, 0.75], [1, 1, 2, 2]),
            &device,
        );
        let weight_codes = Tensor::<NdBackend, 3, Int>::from_data(
            TensorData::new(vec![1i64, -1, 0, 1], [1, 2, 2]),
            &device,
        );
        let output = packed_lowrank_projection_device_reference(input, weight_codes, 0.5, 2);
        assert_eq!(output.shape().dims::<4>(), [1, 1, 2, 2]);
        assert!(
            output
                .into_data()
                .to_vec::<f32>()
                .expect("f32 output")
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn packed_decoder_tail_device_reference_runs() {
        let device = Default::default();
        let y_neuron = Tensor::<NdBackend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.5, 0.25], [1, 2, 1, 2]),
            &device,
        );
        let weight_codes = Tensor::<NdBackend, 2, Int>::from_data(
            TensorData::new(vec![1i64, 0, -1, 1, 0, 1, 1, -1], [4, 2]),
            &device,
        );
        let output = packed_decoder_tail_device_reference(y_neuron, weight_codes, 0.25);
        assert_eq!(output.shape().dims::<4>(), [1, 1, 1, 2]);
        assert!(
            output
                .into_data()
                .to_vec::<f32>()
                .expect("f32 output")
                .iter()
                .all(|value| value.is_finite())
        );
    }

    #[test]
    fn rho_int8_block_device_reference_round_trip_runs() {
        let device = Default::default();
        let rho = Tensor::<NdBackend, 4>::from_data(
            TensorData::new(
                (0..48)
                    .map(|index| ((index as f32 * 0.11).sin() * 2.0) + (index % 5) as f32 * 0.1)
                    .collect::<Vec<_>>(),
                [1, 2, 4, 6],
            ),
            &device,
        );
        let packed = pack_rho_int8_block_device_reference(rho.clone(), 8);
        let restored =
            unpack_rho_int8_block_device_reference(packed.packed, packed.scales, [1, 2, 4, 6], 8);
        let original = rho.into_data().to_vec::<f32>().expect("rho data");
        let restored = restored.into_data().to_vec::<f32>().expect("restored data");
        assert_eq!(original.len(), restored.len());
        let max_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(max_abs_error <= 0.05);
    }

    #[test]
    fn fused_lowrank_projection_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        let input = Tensor::<WgpuBackend, 4>::random([2, 1, 7, 16], Distribution::Default, &device);
        let weight = Tensor::<WgpuBackend, 3>::random([4, 16, 12], Distribution::Default, &device);
        let (input_codes, input_scale) = quantize_signed_codes(input.clone());
        let (weight_codes, weight_scale) = quantize_signed_codes(weight.clone());
        let fused = try_fused_packed_lowrank_projection(
            &input_codes,
            &weight_codes,
            input_scale,
            weight_scale,
            12,
        )
        .expect("fused lowrank");
        let reference = packed_lowrank_projection_device_reference(
            input_codes.float().mul_scalar(input_scale),
            weight_codes.clone(),
            weight_scale,
            12,
        );
        assert_close(fused, reference, 1.0e-4, 1.0e-4);
    }

    #[test]
    fn fused_decoder_tail_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        let y = Tensor::<WgpuBackend, 4>::random([2, 4, 5, 8], Distribution::Default, &device);
        let decoder = Tensor::<WgpuBackend, 2>::random([32, 16], Distribution::Default, &device);
        let (y_codes, y_scale) = quantize_signed_codes(y.clone());
        let (decoder_codes, decoder_scale) = quantize_signed_codes(decoder.clone());
        let fused = try_fused_packed_decoder_tail(&y_codes, &decoder_codes, y_scale, decoder_scale)
            .expect("fused decoder tail");
        let reference = packed_decoder_tail_device_reference(
            y_codes.float().mul_scalar(y_scale),
            decoder_codes.clone(),
            decoder_scale,
        );
        assert_close(fused, reference, 1.0e-4, 1.0e-4);
    }

    #[test]
    fn fused_lowrank_backward_helpers_match_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        let input = Tensor::<WgpuBackend, 4>::random([2, 1, 7, 16], Distribution::Default, &device);
        let grad = Tensor::<WgpuBackend, 4>::random([2, 4, 7, 12], Distribution::Default, &device);
        let weight = Tensor::<WgpuBackend, 3>::random([4, 16, 12], Distribution::Default, &device);
        let (input_codes, input_scale) = quantize_signed_codes(input);
        let (weight_codes, weight_scale) = quantize_signed_codes(weight);
        let fused_input =
            try_fused_packed_lowrank_grad_input(&grad, &weight_codes, weight_scale, 1)
                .expect("fused grad input");
        let reference_input = packed_lowrank_grad_input_device_reference(
            grad.clone(),
            weight_codes.clone(),
            weight_scale,
            1,
        );
        assert_close(fused_input, reference_input, 1.0e-4, 1.0e-4);

        let fused_weight = try_fused_packed_lowrank_grad_weight(&input_codes, &grad, input_scale)
            .expect("fused grad weight");
        let reference_weight =
            packed_lowrank_grad_weight_device_reference(input_codes, grad, input_scale);
        assert_close(fused_weight, reference_weight, 1.0e-4, 1.0e-4);
    }

    #[test]
    fn fused_decoder_tail_backward_helpers_match_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        let y = Tensor::<WgpuBackend, 4>::random([2, 4, 5, 8], Distribution::Default, &device);
        let grad = Tensor::<WgpuBackend, 4>::random([2, 1, 5, 16], Distribution::Default, &device);
        let decoder = Tensor::<WgpuBackend, 2>::random([32, 16], Distribution::Default, &device);
        let (y_codes, y_scale) = quantize_signed_codes(y);
        let (decoder_codes, decoder_scale) = quantize_signed_codes(decoder);
        let fused_input =
            try_fused_packed_decoder_tail_grad_input(&grad, &decoder_codes, decoder_scale, 4, 8)
                .expect("fused tail grad input");
        let reference_input = packed_decoder_tail_grad_input_device_reference(
            grad.clone(),
            decoder_codes.clone(),
            decoder_scale,
            4,
            8,
        );
        assert_close(fused_input, reference_input, 1.0e-4, 1.0e-4);

        let fused_weight = try_fused_packed_decoder_tail_grad_weight(&y_codes, &grad, y_scale)
            .expect("fused tail grad weight");
        let reference_weight =
            packed_decoder_tail_grad_weight_device_reference(y_codes, grad, y_scale);
        assert_close(fused_weight, reference_weight, 1.0e-4, 1.0e-4);
    }
}
