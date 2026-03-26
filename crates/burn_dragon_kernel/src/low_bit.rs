use std::any::Any;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
use std::marker::PhantomData;
#[cfg(feature = "cuda")]
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
use burn_cubecl::CubeRuntime;
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::server::Bindings;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
#[cfg(feature = "cuda")]
use cudarc::nvrtc::{CompileOptions, compile_ptx_with_opts};

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
const PACKED_DOT_WGPU_WORKGROUP_SIZE_X: u32 = 64;
const LOW_BIT_PACKED_LOWRANK_WGSL_SHADER: &str = include_str!("low_bit_packed_lowrank.wgsl");
const LOW_BIT_PACKED_DECODER_TAIL_WGSL_SHADER: &str =
    include_str!("low_bit_packed_decoder_tail.wgsl");
#[cfg(feature = "cuda")]
const LOW_BIT_CUDA_RAW_DOT_FROM_CODES_SRC: &str = r#"
extern "C" __global__ void packed_lowrank_dp4a(
    const int* input_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int pack_len,
    int latent_out,
    float input_scale,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * pack_len;
    int weight_base = (head_idx * pack_len) * latent_out + latent_idx;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        acc = __dp4a(input_packed[input_base + p], weight_packed[weight_base + p * latent_out], acc);
    }
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_from_codes(
    const int* input_codes,
    const int* weight_packed,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int embd,
    int pack_len,
    int latent_out,
    float input_scale,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * embd;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        int e = p * 4;
        int v0 = (e < embd) ? input_codes[input_base + e] : 0;
        int v1 = (e + 1 < embd) ? input_codes[input_base + e + 1] : 0;
        int v2 = (e + 2 < embd) ? input_codes[input_base + e + 2] : 0;
        int v3 = (e + 3 < embd) ? input_codes[input_base + e + 3] : 0;
        int packed_input =
            (v0 & 0xff) |
            ((v1 & 0xff) << 8) |
            ((v2 & 0xff) << 16) |
            ((v3 & 0xff) << 24);
        int packed_weight = weight_packed[(head_idx * pack_len + p) * latent_out + latent_idx];
        acc = __dp4a(packed_input, packed_weight, acc);
    }
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_from_codes_scale_ptr(
    const int* input_codes,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int embd,
    int pack_len,
    int latent_out,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * embd;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        int e = p * 4;
        int v0 = (e < embd) ? input_codes[input_base + e] : 0;
        int v1 = (e + 1 < embd) ? input_codes[input_base + e + 1] : 0;
        int v2 = (e + 2 < embd) ? input_codes[input_base + e + 2] : 0;
        int v3 = (e + 3 < embd) ? input_codes[input_base + e + 3] : 0;
        int packed_input =
            (v0 & 0xff) |
            ((v1 & 0xff) << 8) |
            ((v2 & 0xff) << 16) |
            ((v3 & 0xff) << 24);
        int packed_weight = weight_packed[(head_idx * pack_len + p) * latent_out + latent_idx];
        acc = __dp4a(packed_input, packed_weight, acc);
    }
    float input_scale = input_scale_ptr[0];
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a(
    const int* y_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int heads,
    int tokens,
    int pack_len,
    int dim,
    float input_scale,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * pack_len;
        int weight_base = (head_idx * pack_len) * dim + dim_idx;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            acc = __dp4a(y_packed[input_base + p], weight_packed[weight_base + p * dim], acc);
        }
    }
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a_from_codes(
    const int* y_codes,
    const int* weight_packed,
    float* output,
    int batch,
    int heads,
    int tokens,
    int latent,
    int pack_len,
    int dim,
    float input_scale,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * latent;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            int l = p * 4;
            int v0 = (l < latent) ? y_codes[input_base + l] : 0;
            int v1 = (l + 1 < latent) ? y_codes[input_base + l + 1] : 0;
            int v2 = (l + 2 < latent) ? y_codes[input_base + l + 2] : 0;
            int v3 = (l + 3 < latent) ? y_codes[input_base + l + 3] : 0;
            int packed_input =
                (v0 & 0xff) |
                ((v1 & 0xff) << 8) |
                ((v2 & 0xff) << 16) |
                ((v3 & 0xff) << 24);
            int packed_weight = weight_packed[(head_idx * pack_len + p) * dim + dim_idx];
            acc = __dp4a(packed_input, packed_weight, acc);
        }
    }
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_grad_input_raw(
    const float* grad,
    const int* weight_codes,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int time,
    int embd,
    int latent,
    float weight_scale
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int bih = blockIdx.z;
    if (e >= embd || t >= time || bih >= batch * input_heads) {
        return;
    }
    int input_head = bih % input_heads;
    int b = bih / input_heads;
    float acc = 0.0f;
    if (input_heads == 1) {
        for (int h = 0; h < heads; ++h) {
            int grad_base = ((b * heads + h) * time + t) * latent;
            int weight_base = (h * embd + e) * latent;
            #pragma unroll 4
            for (int l = 0; l < latent; ++l) {
                acc += grad[grad_base + l] * (float)weight_codes[weight_base + l];
            }
        }
    } else {
        int h = input_head;
        int grad_base = ((b * heads + h) * time + t) * latent;
        int weight_base = (h * embd + e) * latent;
        #pragma unroll 4
        for (int l = 0; l < latent; ++l) {
            acc += grad[grad_base + l] * (float)weight_codes[weight_base + l];
        }
    }
    output[((b * input_heads + input_head) * time + t) * embd + e] = acc * weight_scale;
}

extern "C" __global__ void packed_lowrank_grad_weight_raw(
    const int* input_codes,
    const float* grad,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int time,
    int embd,
    int latent,
    float activation_scale
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int e = blockIdx.y;
    int h = blockIdx.z;
    if (l >= latent || e >= embd || h >= heads) {
        return;
    }
    int input_head = input_heads == 1 ? 0 : h;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < time; ++t) {
            int input_index = ((b * input_heads + input_head) * time + t) * embd + e;
            int grad_index = ((b * heads + h) * time + t) * latent + l;
            acc += (float)input_codes[input_index] * grad[grad_index];
        }
    }
    output[(h * embd + e) * latent + l] = acc * activation_scale;
}

extern "C" __global__ void packed_decoder_tail_grad_input_raw(
    const float* grad,
    const int* weight_codes,
    float* output,
    int batch,
    int heads,
    int time,
    int latent,
    int dim,
    float weight_scale
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int bh = blockIdx.z;
    if (l >= latent || t >= time || bh >= batch * heads) {
        return;
    }
    int h = bh % heads;
    int b = bh / heads;
    int weight_row_base = (h * latent + l) * dim;
    int grad_base = (b * time + t) * dim;
    float acc = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < dim; ++d) {
        acc += grad[grad_base + d] * (float)weight_codes[weight_row_base + d];
    }
    output[((b * heads + h) * time + t) * latent + l] = acc * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_grad_weight_raw(
    const int* y_codes,
    const float* grad,
    float* output,
    int batch,
    int heads,
    int time,
    int latent,
    int dim,
    float activation_scale
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int hl = blockIdx.y;
    if (d >= dim || hl >= heads * latent) {
        return;
    }
    int h = hl / latent;
    int l = hl % latent;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < time; ++t) {
            int y_index = ((b * heads + h) * time + t) * latent + l;
            int grad_index = (b * time + t) * dim + d;
            acc += (float)y_codes[y_index] * grad[grad_index];
        }
    }
    output[hl * dim + d] = acc * activation_scale;
}
"#;

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct RawCudaPackedDotKernels {
    #[allow(dead_code)]
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    lowrank: CudaFunction,
    lowrank_from_codes: CudaFunction,
    lowrank_from_codes_scale_ptr: CudaFunction,
    decoder: CudaFunction,
    decoder_from_codes: CudaFunction,
    lowrank_grad_input: CudaFunction,
    lowrank_grad_weight: CudaFunction,
    decoder_grad_input: CudaFunction,
    decoder_grad_weight: CudaFunction,
}

#[cfg(feature = "cuda")]
static RAW_CUDA_PACKED_DOT_KERNELS: OnceLock<Mutex<HashMap<usize, RawCudaPackedDotKernels>>> =
    OnceLock::new();

struct PackedDotLowrankProjectionKernel;

impl KernelSource for PackedDotLowrankProjectionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_LOWRANK_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotDecoderTailKernel;

impl KernelSource for PackedDotDecoderTailKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_DECODER_TAIL_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

pub fn supports_packed_low_bit_device_backend<B: BackendTrait>() -> bool {
    let _ = core::any::type_name::<B>();
    true
}

pub fn supports_packed_rho_int8_block_device_backend<B: BackendTrait>() -> bool {
    supports_packed_low_bit_device_backend::<B>()
}

fn pack_i8x4_host(v0: i8, v1: i8, v2: i8, v3: i8) -> i32 {
    let to_byte = |value: i8| (i32::from(value).clamp(-127, 127) & 0xff) as u32;
    (to_byte(v0) | (to_byte(v1) << 8) | (to_byte(v2) << 16) | (to_byte(v3) << 24)) as i32
}

pub fn pack_lowrank_weight_codes_i8x4(
    codes: &[i8],
    heads: usize,
    embd: usize,
    latent: usize,
) -> Vec<i32> {
    let pack_len = embd.div_ceil(4);
    let mut packed = vec![0i32; heads * pack_len * latent];
    for h in 0..heads {
        for p in 0..pack_len {
            for l in 0..latent {
                let e = p * 4;
                let v0 = if e < embd {
                    codes[(h * embd + e) * latent + l]
                } else {
                    0
                };
                let v1 = if e + 1 < embd {
                    codes[(h * embd + e + 1) * latent + l]
                } else {
                    0
                };
                let v2 = if e + 2 < embd {
                    codes[(h * embd + e + 2) * latent + l]
                } else {
                    0
                };
                let v3 = if e + 3 < embd {
                    codes[(h * embd + e + 3) * latent + l]
                } else {
                    0
                };
                packed[(h * pack_len + p) * latent + l] = pack_i8x4_host(v0, v1, v2, v3);
            }
        }
    }
    packed
}

pub fn pack_decoder_weight_codes_i8x4(
    codes: &[i8],
    heads: usize,
    latent_per_head: usize,
    dim: usize,
) -> Vec<i32> {
    let pack_len = latent_per_head.div_ceil(4);
    let mut packed = vec![0i32; heads * pack_len * dim];
    for h in 0..heads {
        for p in 0..pack_len {
            for d in 0..dim {
                let l = p * 4;
                let row0 = h * latent_per_head + l;
                let row1 = h * latent_per_head + l + 1;
                let row2 = h * latent_per_head + l + 2;
                let row3 = h * latent_per_head + l + 3;
                let v0 = if l < latent_per_head {
                    codes[row0 * dim + d]
                } else {
                    0
                };
                let v1 = if l + 1 < latent_per_head {
                    codes[row1 * dim + d]
                } else {
                    0
                };
                let v2 = if l + 2 < latent_per_head {
                    codes[row2 * dim + d]
                } else {
                    0
                };
                let v3 = if l + 3 < latent_per_head {
                    codes[row3 * dim + d]
                } else {
                    0
                };
                packed[(h * pack_len + p) * dim + d] = pack_i8x4_host(v0, v1, v2, v3);
            }
        }
    }
    packed
}

pub fn pack_lowrank_input_codes_i8x4(
    codes: &[i8],
    batch: usize,
    input_heads: usize,
    tokens: usize,
    embd: usize,
) -> Vec<i32> {
    let pack_len = embd.div_ceil(4);
    let mut packed = vec![0i32; batch * input_heads * tokens * pack_len];
    for b in 0..batch {
        for h in 0..input_heads {
            for t in 0..tokens {
                let base = ((b * input_heads + h) * tokens + t) * embd;
                let out_base = ((b * input_heads + h) * tokens + t) * pack_len;
                for p in 0..pack_len {
                    let e = p * 4;
                    let v0 = *codes.get(base + e).unwrap_or(&0);
                    let v1 = *codes.get(base + e + 1).unwrap_or(&0);
                    let v2 = *codes.get(base + e + 2).unwrap_or(&0);
                    let v3 = *codes.get(base + e + 3).unwrap_or(&0);
                    packed[out_base + p] = pack_i8x4_host(v0, v1, v2, v3);
                }
            }
        }
    }
    packed
}

pub fn pack_decoder_input_codes_i8x4(
    codes: &[i8],
    batch: usize,
    heads: usize,
    tokens: usize,
    latent: usize,
) -> Vec<i32> {
    let pack_len = latent.div_ceil(4);
    let mut packed = vec![0i32; batch * heads * tokens * pack_len];
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..tokens {
                let base = ((b * heads + h) * tokens + t) * latent;
                let out_base = ((b * heads + h) * tokens + t) * pack_len;
                for p in 0..pack_len {
                    let l = p * 4;
                    let v0 = *codes.get(base + l).unwrap_or(&0);
                    let v1 = *codes.get(base + l + 1).unwrap_or(&0);
                    let v2 = *codes.get(base + l + 2).unwrap_or(&0);
                    let v3 = *codes.get(base + l + 3).unwrap_or(&0);
                    packed[out_base + p] = pack_i8x4_host(v0, v1, v2, v3);
                }
            }
        }
    }
    packed
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch_for_device(device_index: usize) -> Option<String> {
    if let Ok(value) = std::env::var("LOW_BIT_CUDA_NVRTC_ARCH") {
        if !value.trim().is_empty() {
            return Some(value);
        }
    }
    cudarc::driver::result::init().ok()?;
    let device_ptr = cudarc::driver::result::device::get(device_index as i32).ok()?;
    let (major, minor) = unsafe {
        (
            cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .ok()?,
            cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .ok()?,
        )
    };
    Some(format!("compute_{}{}", major, minor))
}

#[cfg(feature = "cuda")]
fn raw_cuda_packed_dot_kernels(device_index: usize) -> Option<RawCudaPackedDotKernels> {
    let cache = RAW_CUDA_PACKED_DOT_KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().ok()?;
    if let Some(existing) = cache.get(&device_index) {
        return Some(existing.clone());
    }
    let arch = Box::leak(detect_cuda_arch_for_device(device_index)?.into_boxed_str());
    let ctx = CudaContext::new(device_index).ok()?;
    let ptx = compile_ptx_with_opts(
        LOW_BIT_CUDA_RAW_DOT_FROM_CODES_SRC,
        CompileOptions {
            arch: Some(arch),
            fmad: Some(true),
            ..Default::default()
        },
    )
    .ok()?;
    let module = ctx.load_module(ptx).ok()?;
    let bundle = RawCudaPackedDotKernels {
        ctx: ctx.clone(),
        stream: ctx.default_stream(),
        lowrank: module.load_function("packed_lowrank_dp4a").ok()?,
        lowrank_from_codes: module
            .load_function("packed_lowrank_dp4a_from_codes")
            .ok()?,
        lowrank_from_codes_scale_ptr: module
            .load_function("packed_lowrank_dp4a_from_codes_scale_ptr")
            .ok()?,
        decoder: module.load_function("packed_decoder_tail_dp4a").ok()?,
        decoder_from_codes: module
            .load_function("packed_decoder_tail_dp4a_from_codes")
            .ok()?,
        lowrank_grad_input: module.load_function("packed_lowrank_grad_input_raw").ok()?,
        lowrank_grad_weight: module
            .load_function("packed_lowrank_grad_weight_raw")
            .ok()?,
        decoder_grad_input: module
            .load_function("packed_decoder_tail_grad_input_raw")
            .ok()?,
        decoder_grad_weight: module
            .load_function("packed_decoder_tail_grad_weight_raw")
            .ok()?,
    };
    cache.insert(device_index, bundle.clone());
    Some(bundle)
}

pub fn try_wgpu_packed_dot_lowrank_projection<B: BackendTrait>(
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
    diagnose_wgpu_packed_dot_lowrank_projection(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
    .ok()
}

pub fn try_cube_fused_packed_lowrank_projection_wgpu<B: BackendTrait>(
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
    try_direct_packed_lowrank_projection::<B, WgpuRuntime>(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
}

pub fn try_wgpu_packed_dot_decoder_tail<B: BackendTrait>(
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
}

pub fn diagnose_wgpu_packed_dot_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_lowrank_projection_wgpu_impl(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
}

pub fn diagnose_wgpu_packed_dot_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_decoder_tail_wgpu_impl(y_codes, weight_codes, activation_scale, weight_scale)
}

pub fn try_cube_fused_packed_decoder_tail_wgpu<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_decoder_tail::<B, WgpuRuntime>(
        y_codes,
        weight_codes,
        activation_scale,
        weight_scale,
    )
}

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
    try_packed_dot_lowrank_projection_wgpu_impl(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
    .ok()
    .or_else(|| {
        try_direct_packed_lowrank_projection::<B, WgpuRuntime>(
            input_codes,
            weight_codes,
            activation_scale,
            weight_scale,
            latent_out,
        )
    })
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_packed_lowrank_projection::<B, CudaRuntime>(
                input_codes,
                weight_codes,
                activation_scale,
                weight_scale,
                latent_out,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
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
    try_packed_dot_decoder_tail_wgpu_impl(y_codes, weight_codes, activation_scale, weight_scale)
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

#[derive(Debug, Clone, Copy)]
struct PackedLowrankTrainingShape {
    input_heads: usize,
}

#[derive(Debug, Clone, Copy)]
struct PackedDecoderTailTrainingShape {
    heads: usize,
    latent: usize,
}

#[derive(Debug, Clone)]
enum PackedActivationCodesState<T> {
    Device(T),
    HostI8 { values: Vec<i8>, shape: [usize; 4] },
}

#[derive(Debug, Clone)]
struct PackedLowrankTrainingStateWgpu {
    input_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    relu_threshold: Option<f32>,
    shape: PackedLowrankTrainingShape,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum PackedLowrankGradInputStateCuda {
    WeightCodes {
        weight_codes: CubeTensor<CudaRuntime>,
        weight_scale: f32,
    },
    WeightTransposedFloat {
        weight_transposed_float: CubeTensor<CudaRuntime>,
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct PackedLowrankTrainingStateCuda {
    input_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    grad_input_state: PackedLowrankGradInputStateCuda,
    activation_scale: f32,
    weight_scale: f32,
    relu_threshold: Option<f32>,
    shape: PackedLowrankTrainingShape,
}

#[derive(Debug, Clone)]
struct PackedDecoderTailTrainingStateWgpu {
    y_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    shape: PackedDecoderTailTrainingShape,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
enum PackedDecoderTailGradInputStateCuda {
    WeightCodes {
        weight_codes: CubeTensor<CudaRuntime>,
        weight_scale: f32,
    },
    DecoderFloat {
        decoder_float: CubeTensor<CudaRuntime>,
    },
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct PackedDecoderTailTrainingStateCuda {
    y_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    grad_input_state: PackedDecoderTailGradInputStateCuda,
    activation_scale: f32,
    shape: PackedDecoderTailTrainingShape,
}

#[derive(Debug, Clone)]
struct RetroPackedLowrankWgpu {
    input_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    relu_threshold: Option<f32>,
}

impl RetroForward for RetroPackedLowrankWgpu {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let input_codes = restore_activation_codes_state_wgpu(&self.input_codes, &device);
        let input_codes_tensor = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(input_codes.into_primitive())
                .expect("wgpu retro lowrank input codes"),
        );
        let weight_codes_tensor = BurnTensor::<WgpuCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(self.weight_codes.clone())
                .expect("wgpu retro lowrank weight codes"),
        );
        let output = try_fused_packed_lowrank_projection(
            &input_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
            self.latent_out,
        )
        .unwrap_or_else(|| {
            packed_lowrank_projection_device_reference(
                input_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
                self.latent_out,
            )
        });
        let output = if let Some(threshold) = self.relu_threshold {
            activation::relu(output.sub_scalar(threshold))
        } else {
            output
        };
        states.save(
            out_node,
            try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("wgpu retro lowrank output primitive"),
        );
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RetroPackedLowrankCuda {
    input_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    relu_threshold: Option<f32>,
}

#[cfg(feature = "cuda")]
impl RetroForward for RetroPackedLowrankCuda {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let input_codes = restore_activation_codes_state_cuda(&self.input_codes, &device);
        let input_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(input_codes.into_primitive())
                .expect("cuda retro lowrank input codes"),
        );
        let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(self.weight_codes.clone())
                .expect("cuda retro lowrank weight codes"),
        );
        let output = try_fused_packed_lowrank_projection(
            &input_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
            self.latent_out,
        )
        .unwrap_or_else(|| {
            packed_lowrank_projection_device_reference(
                input_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
                self.latent_out,
            )
        });
        let output = if let Some(threshold) = self.relu_threshold {
            activation::relu(output.sub_scalar(threshold))
        } else {
            output
        };
        states.save(
            out_node,
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("cuda retro lowrank output primitive"),
        );
    }
}

#[derive(Debug, Clone)]
struct RetroPackedDecoderTailWgpu {
    y_codes: PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
}

impl RetroForward for RetroPackedDecoderTailWgpu {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let y_codes = restore_activation_codes_state_wgpu(&self.y_codes, &device);
        let y_codes_tensor = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(y_codes.into_primitive())
                .expect("wgpu retro decoder-tail activation codes"),
        );
        let weight_codes_tensor = BurnTensor::<WgpuCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(self.weight_codes.clone())
                .expect("wgpu retro decoder-tail weight codes"),
        );
        let output = try_fused_packed_decoder_tail(
            &y_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
        )
        .unwrap_or_else(|| {
            packed_decoder_tail_device_reference(
                y_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
            )
        });
        states.save(
            out_node,
            try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("wgpu retro decoder-tail output primitive"),
        );
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Clone)]
struct RetroPackedDecoderTailCuda {
    y_codes: PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
}

#[cfg(feature = "cuda")]
impl RetroForward for RetroPackedDecoderTailCuda {
    fn forward(&self, states: &mut BackwardStates, out_node: NodeId) {
        let device = self.weight_codes.device.clone();
        let y_codes = restore_activation_codes_state_cuda(&self.y_codes, &device);
        let y_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(y_codes.into_primitive())
                .expect("cuda retro decoder-tail activation codes"),
        );
        let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(self.weight_codes.clone())
                .expect("cuda retro decoder-tail weight codes"),
        );
        let output = try_fused_packed_decoder_tail(
            &y_codes_tensor,
            &weight_codes_tensor,
            self.activation_scale,
            self.weight_scale,
        )
        .unwrap_or_else(|| {
            packed_decoder_tail_device_reference(
                y_codes_tensor.float().mul_scalar(self.activation_scale),
                weight_codes_tensor,
                self.weight_scale,
            )
        });
        states.save(
            out_node,
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                output.into_primitive().tensor(),
            )
            .expect("cuda retro decoder-tail output primitive"),
        );
    }
}

#[derive(Debug)]
struct FusedPackedLowrankBackward<B>(PhantomData<B>);

impl Backward<WgpuCubeBackend, 2> for FusedPackedLowrankBackward<WgpuCubeBackend> {
    type State = PackedLowrankTrainingStateWgpu;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<WgpuCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let weight_codes = BurnTensor::<WgpuCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(state.weight_codes.clone())
                .expect("wgpu lowrank backward weight codes"),
        );
        let input_codes =
            restore_activation_codes_state_wgpu(&state.input_codes, &state.weight_codes.device);
        let grad_projected = if let Some(threshold) = state.relu_threshold {
            let projected = packed_lowrank_projection_device_reference(
                input_codes
                    .clone()
                    .float()
                    .mul_scalar(state.activation_scale),
                weight_codes.clone(),
                state.weight_scale,
                weight_codes.shape().dims::<3>()[2],
            );
            let activation_mask = projected.sub_scalar(threshold).greater_elem(0.0).float();
            grad_output.clone() * activation_mask
        } else {
            grad_output.clone()
        };

        if let Some(parent) = &parents[0] {
            let grad_input = try_fused_packed_lowrank_grad_input(
                &grad_projected,
                &weight_codes,
                state.weight_scale,
                state.shape.input_heads,
            )
            .unwrap_or_else(|| {
                packed_lowrank_grad_input_device_reference(
                    grad_projected.clone(),
                    weight_codes.clone(),
                    state.weight_scale,
                    state.shape.input_heads,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let grad_weight = try_fused_packed_lowrank_grad_weight(
                &input_codes,
                &grad_projected,
                state.activation_scale,
            )
            .unwrap_or_else(|| {
                packed_lowrank_grad_weight_device_reference(
                    input_codes.clone(),
                    grad_projected.clone(),
                    state.activation_scale,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 2> for FusedPackedLowrankBackward<CudaCubeBackend> {
    type State = PackedLowrankTrainingStateCuda;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<CudaCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let grad_device = grad_output.device();
        let input_codes = restore_activation_codes_state_cuda(&state.input_codes, &grad_device);
        let weight_codes = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
            try_cast_int_backend::<CudaCubeBackend, _>(state.weight_codes.clone())
                .expect("cuda lowrank backward weight codes"),
        );
        let grad_projected = if let Some(threshold) = state.relu_threshold {
            let projected = packed_lowrank_projection_device_reference(
                input_codes
                    .clone()
                    .float()
                    .mul_scalar(state.activation_scale),
                weight_codes.clone(),
                state.weight_scale,
                weight_codes.shape().dims::<3>()[2],
            );
            let activation_mask = projected.sub_scalar(threshold).greater_elem(0.0).float();
            grad_output.clone() * activation_mask
        } else {
            grad_output.clone()
        };

        if let Some(parent) = &parents[0] {
            let grad_input = match &state.grad_input_state {
                PackedLowrankGradInputStateCuda::WeightCodes {
                    weight_codes,
                    weight_scale,
                } => {
                    let weight_codes = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
                        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
                            .expect("cuda lowrank backward weight codes"),
                    );
                    packed_lowrank_grad_input_device_reference(
                        grad_projected.clone(),
                        weight_codes,
                        *weight_scale,
                        state.shape.input_heads,
                    )
                }
                PackedLowrankGradInputStateCuda::WeightTransposedFloat {
                    weight_transposed_float,
                } => {
                    let weight_transposed_float = BurnTensor::<CudaCubeBackend, 3>::from_primitive(
                        TensorPrimitive::Float(weight_transposed_float.clone()),
                    );
                    packed_lowrank_grad_input_from_transposed_float_weight_cuda(
                        grad_projected.clone(),
                        weight_transposed_float,
                        state.shape.input_heads,
                    )
                }
            };
            grads.register::<CudaCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            // Prefer the exact reference weight-gradient path for training fidelity until the
            // raw/fused CUDA grad-weight kernels prove parity at the model level.
            let grad_weight = packed_lowrank_grad_weight_device_reference(
                input_codes.clone(),
                grad_projected.clone(),
                state.activation_scale,
            );
            grads.register::<CudaCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[derive(Debug)]
struct FusedPackedDecoderTailBackward<B>(PhantomData<B>);

impl Backward<WgpuCubeBackend, 2> for FusedPackedDecoderTailBackward<WgpuCubeBackend> {
    type State = PackedDecoderTailTrainingStateWgpu;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<WgpuCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let weight_codes = BurnTensor::<WgpuCubeBackend, 2, Int>::from_primitive(
            try_cast_int_backend::<WgpuCubeBackend, _>(state.weight_codes.clone())
                .expect("wgpu decoder-tail backward weight codes"),
        );

        if let Some(parent) = &parents[0] {
            let grad_input = try_fused_packed_decoder_tail_grad_input(
                &grad_output,
                &weight_codes,
                state.weight_scale,
                state.shape.heads,
                state.shape.latent,
            )
            .unwrap_or_else(|| {
                packed_decoder_tail_grad_input_device_reference(
                    grad_output.clone(),
                    weight_codes.clone(),
                    state.weight_scale,
                    state.shape.heads,
                    state.shape.latent,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let y_codes =
                restore_activation_codes_state_wgpu(&state.y_codes, &state.weight_codes.device);
            let grad_weight = try_fused_packed_decoder_tail_grad_weight(
                &y_codes,
                &grad_output,
                state.activation_scale,
            )
            .unwrap_or_else(|| {
                packed_decoder_tail_grad_weight_device_reference(
                    y_codes.clone(),
                    grad_output.clone(),
                    state.activation_scale,
                )
            });
            grads.register::<WgpuCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 2> for FusedPackedDecoderTailBackward<CudaCubeBackend> {
    type State = PackedDecoderTailTrainingStateCuda;

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        let grad_output = grads.consume::<CudaCubeBackend>(&ops.node);
        let state = ops.state;
        let parents = ops.parents;

        let grad_output =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(grad_output));
        let grad_device = grad_output.device();

        if let Some(parent) = &parents[0] {
            let grad_input = match &state.grad_input_state {
                PackedDecoderTailGradInputStateCuda::WeightCodes {
                    weight_codes,
                    weight_scale,
                } => {
                    let weight_codes = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
                        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
                            .expect("cuda decoder-tail backward weight codes"),
                    );
                    packed_decoder_tail_grad_input_device_reference(
                        grad_output.clone(),
                        weight_codes,
                        *weight_scale,
                        state.shape.heads,
                        state.shape.latent,
                    )
                }
                PackedDecoderTailGradInputStateCuda::DecoderFloat { decoder_float } => {
                    let decoder_float = BurnTensor::<CudaCubeBackend, 2>::from_primitive(
                        TensorPrimitive::Float(decoder_float.clone()),
                    );
                    packed_decoder_tail_grad_input_from_float_decoder_cuda(
                        grad_output.clone(),
                        decoder_float,
                        state.shape.heads,
                        state.shape.latent,
                    )
                }
            };
            grads.register::<CudaCubeBackend>(parent.id, grad_input.into_primitive().tensor());
        }

        if let Some(parent) = &parents[1] {
            let y_codes = restore_activation_codes_state_cuda(&state.y_codes, &grad_device);
            // Prefer the exact reference weight-gradient path for training fidelity until the
            // raw/fused CUDA grad-weight kernels prove parity at the model level.
            let grad_weight = packed_decoder_tail_grad_weight_device_reference(
                y_codes.clone(),
                grad_output.clone(),
                state.activation_scale,
            );
            grads.register::<CudaCubeBackend>(parent.id, grad_weight.into_primitive().tensor());
        }
    }
}

fn create_lowrank_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
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
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed low-rank params tensor")
}

#[cfg(feature = "cuda")]
fn create_lowrank_params_cuda(
    device: &<CudaCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> CubeTensor<CudaRuntime> {
    let params = Tensor::<CudaCubeBackend, 1>::from_data(
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
        device,
    );
    try_cast_float_primitive::<CudaCubeBackend, _>(params.into_primitive().tensor())
        .expect("cuda packed low-rank params tensor")
}

fn create_decoder_tail_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
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
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed decoder tail params tensor")
}

#[cfg(feature = "cuda")]
fn create_decoder_tail_params_cuda(
    device: &<CudaCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<CudaRuntime> {
    let params = Tensor::<CudaCubeBackend, 1>::from_data(
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
        device,
    );
    try_cast_float_primitive::<CudaCubeBackend, _>(params.into_primitive().tensor())
        .expect("cuda packed decoder tail params tensor")
}

fn pack_activation_codes_state_wgpu(
    codes: CubeTensor<WgpuRuntime>,
    pack_to_host: bool,
) -> PackedActivationCodesState<CubeTensor<WgpuRuntime>> {
    if !pack_to_host {
        return PackedActivationCodesState::Device(codes);
    }
    let shape = codes.meta.shape.dims::<4>();
    let values = BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<WgpuCubeBackend, _>(codes)
            .expect("wgpu packed activation codes primitive"),
    )
    .into_data()
    .convert::<i32>()
    .into_vec::<i32>()
    .expect("wgpu packed activation codes values")
    .into_iter()
    .map(|value| value.clamp(-127, 127) as i8)
    .collect();
    PackedActivationCodesState::HostI8 { values, shape }
}

fn restore_activation_codes_state_wgpu(
    state: &PackedActivationCodesState<CubeTensor<WgpuRuntime>>,
    device: &<WgpuCubeBackend as BackendTrait>::Device,
) -> BurnTensor<WgpuCubeBackend, 4, Int> {
    match state {
        PackedActivationCodesState::Device(codes) => {
            BurnTensor::<WgpuCubeBackend, 4, Int>::from_primitive(
                try_cast_int_backend::<WgpuCubeBackend, _>(codes.clone())
                    .expect("wgpu activation codes primitive"),
            )
        }
        PackedActivationCodesState::HostI8 { values, shape } => {
            Tensor::<WgpuCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    values.iter().map(|value| *value as i32).collect::<Vec<_>>(),
                    *shape,
                ),
                device,
            )
        }
    }
}

#[cfg(feature = "cuda")]
fn pack_activation_codes_state_cuda(
    codes: CubeTensor<CudaRuntime>,
    pack_to_host: bool,
) -> PackedActivationCodesState<CubeTensor<CudaRuntime>> {
    if !pack_to_host {
        return PackedActivationCodesState::Device(codes);
    }
    let shape = codes.meta.shape.dims::<4>();
    let values = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(codes)
            .expect("cuda packed activation codes primitive"),
    )
    .into_data()
    .convert::<i32>()
    .into_vec::<i32>()
    .expect("cuda packed activation codes values")
    .into_iter()
    .map(|value| value.clamp(-127, 127) as i8)
    .collect();
    PackedActivationCodesState::HostI8 { values, shape }
}

#[cfg(feature = "cuda")]
fn restore_activation_codes_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> BurnTensor<CudaCubeBackend, 4, Int> {
    match state {
        PackedActivationCodesState::Device(codes) => {
            BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
                try_cast_int_backend::<CudaCubeBackend, _>(codes.clone())
                    .expect("cuda activation codes primitive"),
            )
        }
        PackedActivationCodesState::HostI8 { values, shape } => {
            Tensor::<CudaCubeBackend, 4, Int>::from_data(
                TensorData::new(
                    values.iter().map(|value| *value as i32).collect::<Vec<_>>(),
                    *shape,
                ),
                device,
            )
        }
    }
}

#[cfg(feature = "cuda")]
fn packed_lowrank_input_tensor_from_activation_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> Option<BurnTensor<CudaCubeBackend, 4, Int>> {
    let PackedActivationCodesState::HostI8 { values, shape } = state else {
        return None;
    };
    let packed = pack_lowrank_input_codes_i8x4(values, shape[0], shape[1], shape[2], shape[3]);
    Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
        TensorData::new(
            packed.into_iter().map(i64::from).collect::<Vec<_>>(),
            [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
        ),
        device,
    ))
}

#[cfg(feature = "cuda")]
fn packed_decoder_input_tensor_from_activation_state_cuda(
    state: &PackedActivationCodesState<CubeTensor<CudaRuntime>>,
    device: &<CudaCubeBackend as BackendTrait>::Device,
) -> Option<BurnTensor<CudaCubeBackend, 4, Int>> {
    let PackedActivationCodesState::HostI8 { values, shape } = state else {
        return None;
    };
    let packed = pack_decoder_input_codes_i8x4(values, shape[0], shape[1], shape[2], shape[3]);
    Some(Tensor::<CudaCubeBackend, 4, Int>::from_data(
        TensorData::new(
            packed.into_iter().map(i64::from).collect::<Vec<_>>(),
            [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
        ),
        device,
    ))
}

fn fused_packed_lowrank_training_autodiff_wgpu<C: CheckpointStrategy>(
    input: WgpuCubeAutodiffTensor<C>,
    weight: WgpuCubeAutodiffTensor<C>,
    input_codes: CubeTensor<WgpuRuntime>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> WgpuCubeAutodiffTensor<C> {
    let input_inner = <WgpuCubeAutodiffBackend<C> as AutodiffBackend>::inner(input.clone());
    let [batch, input_heads, time, _] = input_inner.meta.shape.dims::<4>();
    let embd = input_inner.meta.shape.dims::<4>()[3];
    let heads = weight_codes.meta.shape.dims::<3>()[0];
    let artifact_latent = weight_codes.meta.shape.dims::<3>()[2];
    let output = packed_lowrank_projection_packed_dot_wgsl_runtime(
        input_codes.clone(),
        weight_codes.clone(),
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    )
    .unwrap_or_else(|_| {
        let params = create_lowrank_params_wgpu(
            &input_codes.device,
            batch,
            input_heads,
            heads,
            time,
            embd,
            activation_scale,
            weight_scale,
            latent_out,
        );
        packed_lowrank_projection_cube_runtime::<WgpuRuntime>(
            input_codes.clone(),
            weight_codes.clone(),
            params,
            batch,
            heads,
            time,
            latent_out,
        )
    });
    let output = if let Some(threshold) = relu_threshold {
        let activated = activation::relu(
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(output))
                .sub_scalar(threshold),
        );
        try_cast_float_primitive::<WgpuCubeBackend, CubeTensor<WgpuRuntime>>(
            activated.into_primitive().tensor(),
        )
        .expect("wgpu packed lowrank relu output primitive")
    } else {
        output
    };
    let shape = PackedLowrankTrainingShape { input_heads };
    let input_codes_state =
        pack_activation_codes_state_wgpu(input_codes, pack_activation_state_to_host);
    match FusedPackedLowrankBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<C>([input.node.clone(), weight.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedLowrankWgpu {
            input_codes: input_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
            latent_out,
            relu_threshold,
        })
        .parents([&input, &weight])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&input);
            prep.checkpoint(&weight);
            prep.finish(
                PackedLowrankTrainingStateWgpu {
                    input_codes: input_codes_state,
                    weight_codes,
                    activation_scale,
                    weight_scale,
                    relu_threshold,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_packed_lowrank_training_autodiff_cuda<C: CheckpointStrategy>(
    input: CudaCubeAutodiffTensor<C>,
    weight: CudaCubeAutodiffTensor<C>,
    input_codes: CubeTensor<CudaRuntime>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> CudaCubeAutodiffTensor<C> {
    let input_inner = <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(input.clone());
    let [batch, input_heads, time, _] = input_inner.meta.shape.dims::<4>();
    let heads = weight_codes.meta.shape.dims::<3>()[0];
    let [_, _, embd, latent] = input_inner.meta.shape.dims::<4>();
    let input_device = input_codes.device.clone();
    let input_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(input_codes.clone())
            .expect("cuda lowrank training input codes tensor"),
    );
    let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 3, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
            .expect("cuda lowrank training weight codes tensor"),
    );
    let grad_input_state = if pack_activation_state_to_host {
        PackedLowrankGradInputStateCuda::WeightCodes {
            weight_codes: weight_codes.clone(),
            weight_scale,
        }
    } else {
        let weight_float =
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(weight.clone()),
            ));
        let weight_transposed_float = try_cast_float_primitive::<CudaCubeBackend, _>(
            weight_float
                .slice([0..1, 0..heads, 0..embd, 0..latent])
                .reshape([heads, embd, latent])
                .swap_dims(1, 2)
                .into_primitive()
                .tensor(),
        )
        .expect("cuda lowrank backward transposed float weight");
        PackedLowrankGradInputStateCuda::WeightTransposedFloat {
            weight_transposed_float,
        }
    };
    let input_codes_state =
        pack_activation_codes_state_cuda(input_codes, pack_activation_state_to_host);
    let output =
        packed_lowrank_input_tensor_from_activation_state_cuda(&input_codes_state, &input_device)
            .and_then(|packed_input| {
                try_raw_cuda_packed_lowrank_projection_prepacked_input(
                    &packed_input,
                    &weight_codes_tensor,
                    activation_scale,
                    weight_scale,
                    latent_out,
                )
            })
            .or_else(|| {
                try_raw_cuda_packed_lowrank_projection(
                    &input_codes_tensor,
                    &weight_codes_tensor,
                    activation_scale,
                    weight_scale,
                    latent_out,
                )
            })
            .and_then(|tensor| {
                try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    tensor.into_primitive().tensor(),
                )
            })
            .unwrap_or_else(|| {
                let restored_input_codes =
                    restore_activation_codes_state_cuda(&input_codes_state, &input_device);
                let restored_input_codes = try_cast_int_primitive::<
                    CudaCubeBackend,
                    CubeTensor<CudaRuntime>,
                >(restored_input_codes.into_primitive())
                .expect("cuda lowrank restored activation codes primitive");
                let params = create_lowrank_params_cuda(
                    &input_device,
                    batch,
                    input_heads,
                    heads,
                    time,
                    input_inner.meta.shape.dims::<4>()[3],
                    activation_scale,
                    weight_scale,
                    latent_out,
                );
                packed_lowrank_projection_cube_runtime::<CudaRuntime>(
                    restored_input_codes,
                    weight_codes.clone(),
                    params,
                    batch,
                    heads,
                    time,
                    latent_out,
                )
            });
    let output = if let Some(threshold) = relu_threshold {
        let activated = activation::relu(
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(output))
                .sub_scalar(threshold),
        );
        try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
            activated.into_primitive().tensor(),
        )
        .expect("cuda packed lowrank relu output primitive")
    } else {
        output
    };
    let shape = PackedLowrankTrainingShape { input_heads };
    match FusedPackedLowrankBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<C>([input.node.clone(), weight.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedLowrankCuda {
            input_codes: input_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
            latent_out,
            relu_threshold,
        })
        .parents([&input, &weight])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&input);
            prep.checkpoint(&weight);
            prep.finish(
                PackedLowrankTrainingStateCuda {
                    input_codes: input_codes_state,
                    weight_codes,
                    grad_input_state,
                    activation_scale,
                    weight_scale,
                    relu_threshold,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

fn fused_packed_decoder_tail_training_autodiff_wgpu<C: CheckpointStrategy>(
    y_neuron: WgpuCubeAutodiffTensor<C>,
    decoder: WgpuCubeAutodiffTensor<C>,
    y_codes: CubeTensor<WgpuRuntime>,
    weight_codes: CubeTensor<WgpuRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> WgpuCubeAutodiffTensor<C> {
    let y_inner = <WgpuCubeAutodiffBackend<C> as AutodiffBackend>::inner(y_neuron.clone());
    let [batch, heads, time, latent] = y_inner.meta.shape.dims::<4>();
    let decoder_tensor = BurnTensor::<WgpuCubeAutodiffBackend<C>, 2>::from_primitive(
        TensorPrimitive::Float(decoder.clone()),
    );
    let dim = decoder_tensor.shape().dims::<2>()[1];
    let artifact_latent_per_head = weight_codes.meta.shape.dims::<2>()[0] / heads;
    let output = packed_decoder_tail_packed_dot_wgsl_runtime(
        y_codes.clone(),
        weight_codes.clone(),
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    )
    .unwrap_or_else(|_| {
        let params = create_decoder_tail_params_wgpu(
            &y_codes.device,
            batch,
            heads,
            time,
            latent,
            artifact_latent_per_head,
            dim,
            activation_scale,
            weight_scale,
        );
        packed_decoder_tail_cube_runtime::<WgpuRuntime>(
            y_codes.clone(),
            weight_codes.clone(),
            params,
            batch,
            time,
            dim,
        )
    });
    let shape = PackedDecoderTailTrainingShape { heads, latent };
    let y_codes_state = pack_activation_codes_state_wgpu(y_codes, pack_activation_state_to_host);
    match FusedPackedDecoderTailBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<C>([y_neuron.node.clone(), decoder.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedDecoderTailWgpu {
            y_codes: y_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
        })
        .parents([&y_neuron, &decoder])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&y_neuron);
            prep.checkpoint(&decoder);
            prep.finish(
                PackedDecoderTailTrainingStateWgpu {
                    y_codes: y_codes_state,
                    weight_codes,
                    activation_scale,
                    weight_scale,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

#[cfg(feature = "cuda")]
fn fused_packed_decoder_tail_training_autodiff_cuda<C: CheckpointStrategy>(
    y_neuron: CudaCubeAutodiffTensor<C>,
    decoder: CudaCubeAutodiffTensor<C>,
    y_codes: CubeTensor<CudaRuntime>,
    weight_codes: CubeTensor<CudaRuntime>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> CudaCubeAutodiffTensor<C> {
    let y_inner = <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(y_neuron.clone());
    let [batch, heads, time, latent] = y_inner.meta.shape.dims::<4>();
    let y_device = y_codes.device.clone();
    let decoder_tensor = BurnTensor::<CudaCubeAutodiffBackend<C>, 2>::from_primitive(
        TensorPrimitive::Float(decoder.clone()),
    );
    let y_codes_tensor = BurnTensor::<CudaCubeBackend, 4, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(y_codes.clone())
            .expect("cuda decoder-tail training activation codes tensor"),
    );
    let weight_codes_tensor = BurnTensor::<CudaCubeBackend, 2, Int>::from_primitive(
        try_cast_int_backend::<CudaCubeBackend, _>(weight_codes.clone())
            .expect("cuda decoder-tail training weight codes tensor"),
    );
    let dim = decoder_tensor.shape().dims::<2>()[1];
    let grad_input_state = if pack_activation_state_to_host {
        PackedDecoderTailGradInputStateCuda::WeightCodes {
            weight_codes: weight_codes.clone(),
            weight_scale,
        }
    } else {
        PackedDecoderTailGradInputStateCuda::DecoderFloat {
            decoder_float: <CudaCubeAutodiffBackend<C> as AutodiffBackend>::inner(decoder.clone()),
        }
    };
    let artifact_latent_per_head = weight_codes.meta.shape.dims::<2>()[0] / heads;
    let y_codes_state = pack_activation_codes_state_cuda(y_codes, pack_activation_state_to_host);
    let output = packed_decoder_input_tensor_from_activation_state_cuda(&y_codes_state, &y_device)
        .and_then(|packed_input| {
            try_raw_cuda_packed_decoder_tail_prepacked_input(
                &packed_input,
                &weight_codes_tensor,
                activation_scale,
                weight_scale,
            )
        })
        .or_else(|| {
            try_raw_cuda_packed_decoder_tail(
                &y_codes_tensor,
                &weight_codes_tensor,
                activation_scale,
                weight_scale,
            )
        })
        .and_then(|tensor| {
            try_cast_float_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                tensor.into_primitive().tensor(),
            )
        })
        .unwrap_or_else(|| {
            let restored_y_codes = restore_activation_codes_state_cuda(&y_codes_state, &y_device);
            let restored_y_codes =
                try_cast_int_primitive::<CudaCubeBackend, CubeTensor<CudaRuntime>>(
                    restored_y_codes.into_primitive(),
                )
                .expect("cuda decoder restored activation codes primitive");
            let params = create_decoder_tail_params_cuda(
                &y_device,
                batch,
                heads,
                time,
                latent,
                artifact_latent_per_head,
                dim,
                activation_scale,
                weight_scale,
            );
            packed_decoder_tail_cube_runtime::<CudaRuntime>(
                restored_y_codes,
                weight_codes.clone(),
                params,
                batch,
                time,
                dim,
            )
        });
    let shape = PackedDecoderTailTrainingShape { heads, latent };
    match FusedPackedDecoderTailBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<C>([y_neuron.node.clone(), decoder.node.clone()])
        .memory_bound()
        .retro_forward(RetroPackedDecoderTailCuda {
            y_codes: y_codes_state.clone(),
            weight_codes: weight_codes.clone(),
            activation_scale,
            weight_scale,
        })
        .parents([&y_neuron, &decoder])
        .stateful()
    {
        OpsKind::Tracked(mut prep) => {
            prep.checkpoint(&y_neuron);
            prep.checkpoint(&decoder);
            prep.finish(
                PackedDecoderTailTrainingStateCuda {
                    y_codes: y_codes_state,
                    grad_input_state,
                    activation_scale,
                    shape,
                },
                output,
            )
        }
        OpsKind::UnTracked(prep) => prep.finish(output),
    }
}

pub fn try_fused_packed_lowrank_training_autodiff<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
    pack_activation_state_to_host: bool,
    relu_threshold: Option<f32>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let use_balanced_checkpointing = core::any::type_name::<B>().contains("BalancedCheckpointing");
    if let (Some(input_ad), Some(weight_ad), Some(input_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            input.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            weight.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(input_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_lowrank_training_autodiff_wgpu::<BalancedCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        } else {
            fused_packed_lowrank_training_autodiff_wgpu::<NoCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    if let (Some(input_ad), Some(weight_ad), Some(input_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            input.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            weight.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(input_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_lowrank_training_autodiff_cuda::<BalancedCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        } else {
            fused_packed_lowrank_training_autodiff_cuda::<NoCheckpointing>(
                input_ad,
                weight_ad,
                input_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    None
}

pub fn try_fused_packed_decoder_tail_training_autodiff<B: BackendTrait>(
    y_neuron: &BurnTensor<B, 4>,
    decoder: &BurnTensor<B, 2>,
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
    pack_activation_state_to_host: bool,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let use_balanced_checkpointing = core::any::type_name::<B>().contains("BalancedCheckpointing");
    if let (Some(y_ad), Some(decoder_ad), Some(y_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            y_neuron.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, WgpuCubeAutodiffTensor>(
            decoder.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(y_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<WgpuRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_decoder_tail_training_autodiff_wgpu::<BalancedCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        } else {
            fused_packed_decoder_tail_training_autodiff_wgpu::<NoCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    #[cfg(feature = "cuda")]
    if let (Some(y_ad), Some(decoder_ad), Some(y_codes_inner), Some(weight_codes_inner)) = (
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            y_neuron.clone().into_primitive().tensor(),
        ),
        try_cast_float_primitive::<B, CudaCubeAutodiffTensor>(
            decoder.clone().into_primitive().tensor(),
        ),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(y_codes.clone().into_primitive()),
        try_cast_int_primitive::<B, CubeTensor<CudaRuntime>>(weight_codes.clone().into_primitive()),
    ) {
        let output = if use_balanced_checkpointing {
            fused_packed_decoder_tail_training_autodiff_cuda::<BalancedCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        } else {
            fused_packed_decoder_tail_training_autodiff_cuda::<NoCheckpointing>(
                y_ad,
                decoder_ad,
                y_codes_inner,
                weight_codes_inner,
                activation_scale,
                weight_scale,
                pack_activation_state_to_host,
            )
        };
        return try_cast_float_backend::<B, _>(output)
            .map(|prim| BurnTensor::from_primitive(TensorPrimitive::Float(prim)));
    }
    None
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

fn create_lowrank_packed_dot_meta_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            artifact_latent as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot low-rank meta tensor")
}

fn packed_lowrank_projection_packed_dot_wgsl_runtime(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let meta = create_lowrank_packed_dot_meta_wgpu(
        &input.device,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_out]),
    );
    let kernel = SourceKernel::new(
        PackedDotLowrankProjectionKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(latent_out as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );
    let bindings = Bindings::new().with_buffers(vec![
        input.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .map_err(|err| format!("wgpu packed-dot lowrank launch failed: {err:?}"))?;
    Ok(output)
}

fn create_decoder_tail_packed_dot_meta_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
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
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot decoder-tail meta tensor")
}

fn packed_decoder_tail_packed_dot_wgsl_runtime(
    y: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let y = into_contiguous(y);
    let weight = into_contiguous(weight);
    let meta = create_decoder_tail_packed_dot_meta_wgpu(
        &y.device,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = y.client.clone();
    let device = y.device.clone();
    let output =
        empty_device::<WgpuRuntime, f32>(client.clone(), device, Shape::new([batch, 1, time, dim]));
    let kernel = SourceKernel::new(
        PackedDotDecoderTailKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(dim as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        batch as u32,
    );
    let bindings = Bindings::new().with_buffers(vec![
        y.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .map_err(|err| format!("wgpu packed-dot decoder-tail launch failed: {err:?}"))?;
    Ok(output)
}

fn try_packed_dot_lowrank_projection_wgpu_impl<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, weight_embd, artifact_latent] = weight_codes.shape().dims::<3>();
    if weight_embd != embd
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return Err(format!(
            "wgpu packed-dot lowrank shape mismatch: input_heads={input_heads} heads={heads} embd={embd} weight_embd={weight_embd} latent_out={latent_out} artifact_latent={artifact_latent}"
        ));
    }

    let input: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive()).ok_or_else(|| {
            format!(
                "wgpu packed-dot lowrank cast failed for input backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let weight: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive()).ok_or_else(|| {
            format!(
                "wgpu packed-dot lowrank cast failed for weight backend {}",
                core::any::type_name::<B>()
            )
        })?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return Err(format!(
            "wgpu packed-dot lowrank dtype mismatch: input={:?} weight={:?}",
            input.dtype, weight.dtype
        ));
    }

    let output = packed_lowrank_projection_packed_dot_wgsl_runtime(
        input,
        weight,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    )?;
    let output_prim = try_cast_float_backend::<B, _>(output).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })?;
    Ok(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

fn try_packed_dot_decoder_tail_wgpu_impl<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [artifact_latent_total, dim] = weight_codes.shape().dims::<2>();
    if artifact_latent_total % heads != 0 {
        return Err(format!(
            "wgpu packed-dot decoder-tail shape mismatch: artifact_latent_total={artifact_latent_total} heads={heads}"
        ));
    }
    let artifact_latent_per_head = artifact_latent_total / heads;
    if latent > artifact_latent_per_head {
        return Err(format!(
            "wgpu packed-dot decoder-tail latent mismatch: latent={latent} artifact_latent_per_head={artifact_latent_per_head}"
        ));
    }

    let y: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive()).ok_or_else(|| {
            format!(
                "wgpu packed-dot decoder-tail cast failed for input backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let weight: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive()).ok_or_else(|| {
            format!(
                "wgpu packed-dot decoder-tail cast failed for weight backend {}",
                core::any::type_name::<B>()
            )
        })?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return Err(format!(
            "wgpu packed-dot decoder-tail dtype mismatch: y={:?} weight={:?}",
            y.dtype, weight.dtype
        ));
    }

    let output = packed_decoder_tail_packed_dot_wgsl_runtime(
        y,
        weight,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    )?;
    let output_prim = try_cast_float_backend::<B, _>(output).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })?;
    Ok(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input<B: BackendTrait>(
    input_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, pack_len] = input_packed.shape().dims::<4>();
    let [heads, weight_pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if weight_pack_len != pack_len || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input<B: BackendTrait>(
    _input_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if pack_len != embd.div_ceil(4) || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_from_codes);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection_device_scale<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if pack_len != embd.div_ceil(4) || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels
        .stream
        .launch_builder(&kernels.lowrank_from_codes_scale_ptr);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection_device_scale<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: &BurnTensor<B, 1>,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input<B: BackendTrait>(
    y_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, pack_len] = y_packed.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let weight_pack_len = packed_latent_total / heads;
    if weight_pack_len != pack_len {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input<B: BackendTrait>(
    _y_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_grad_input<B: BackendTrait>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 3, Int>,
    weight_scale: f32,
    input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [weight_heads, embd, weight_latent] = weight_codes.shape().dims::<3>();
    if heads != weight_heads || latent != weight_latent {
        return None;
    }
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(grad.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        grad.client.clone(),
        grad.device.clone(),
        Shape::new([batch, input_heads, time, embd]),
    );
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            embd.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * input_heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let latent_i32 = latent as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_grad_input);
    builder.arg(&grad_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&latent_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_grad_input<B: BackendTrait>(
    _grad_output: &BurnTensor<B, 4>,
    _weight_codes: &BurnTensor<B, 3, Int>,
    _weight_scale: f32,
    _input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_grad_weight<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [grad_batch, heads, grad_time, latent] = grad_output.shape().dims::<4>();
    if batch != grad_batch || time != grad_time {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([1, heads, embd, latent]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent.div_ceil(block_size_x as usize) as u32,
            embd as u32,
            heads as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let latent_i32 = latent as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_grad_weight);
    builder.arg(&input_ptr);
    builder.arg(&grad_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&latent_i32);
    builder.arg(&activation_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_grad_weight<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _grad_output: &BurnTensor<B, 4>,
    _activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_grad_input<B: BackendTrait>(
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
    let [batch, grad_heads, time, dim] = grad_output.shape().dims::<4>();
    let [weight_rows, weight_dim] = weight_codes.shape().dims::<2>();
    if grad_heads != 1 || weight_rows != heads * latent || weight_dim != dim {
        return None;
    }
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(grad.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        grad.client.clone(),
        grad.device.clone(),
        Shape::new([batch, heads, time, latent]),
    );
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_grad_input);
    builder.arg(&grad_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&dim_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_grad_input<B: BackendTrait>(
    _grad_output: &BurnTensor<B, 4>,
    _weight_codes: &BurnTensor<B, 2, Int>,
    _weight_scale: f32,
    _heads: usize,
    _latent: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_grad_weight<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [grad_batch, grad_heads, grad_time, dim] = grad_output.shape().dims::<4>();
    if grad_batch != batch || grad_heads != 1 || grad_time != time {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if y.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([heads * latent, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            (heads * latent) as u32,
            1,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_grad_weight);
    builder.arg(&y_ptr);
    builder.arg(&grad_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_grad_weight<B: BackendTrait>(
    _y_codes: &BurnTensor<B, 4, Int>,
    _grad_output: &BurnTensor<B, 4>,
    _activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let pack_len = packed_latent_total / heads;
    if pack_len != latent.div_ceil(4) {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_from_codes);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail<B: BackendTrait>(
    _y_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
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
        input.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
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
        y.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
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
        grad.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
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
        input.as_tensor_arg(1),
        grad.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
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
        grad.as_tensor_arg(1),
        weight.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
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
        y.as_tensor_arg(1),
        grad.as_tensor_arg(1),
        output.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    output
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
    input: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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

    let mut acc = Line::cast_from(0u32);
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
    output[output_index] = Line::cast_from(acc) * activation_scale * weight_scale;
}

#[cube(launch)]
fn packed_decoder_tail_cube_kernel(
    y: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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

    let mut acc = Line::cast_from(0u32);
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

    output[(b * time + t) * dim + d] = Line::cast_from(acc) * activation_scale * weight_scale;
}

#[cube(launch)]
fn packed_lowrank_grad_input_cube_kernel(
    grad: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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
    let mut acc = Line::cast_from(0u32);
    if input_heads == 1usize {
        let mut h = 0usize;
        while h < heads {
            let mut l = 0usize;
            while l + 4usize <= latent {
                let grad_index = ((b * heads + h) * time + t) * latent + l;
                let weight_index = (h * embd + e) * latent + l;
                acc += grad[grad_index] * Line::cast_from(weight[weight_index]);
                acc += grad[grad_index + 1usize] * Line::cast_from(weight[weight_index + 1usize]);
                acc += grad[grad_index + 2usize] * Line::cast_from(weight[weight_index + 2usize]);
                acc += grad[grad_index + 3usize] * Line::cast_from(weight[weight_index + 3usize]);
                l += 4usize;
            }
            while l < latent {
                let grad_index = ((b * heads + h) * time + t) * latent + l;
                let weight_index = (h * embd + e) * latent + l;
                acc += grad[grad_index] * Line::cast_from(weight[weight_index]);
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
            acc += grad[grad_index] * Line::cast_from(weight[weight_index]);
            acc += grad[grad_index + 1usize] * Line::cast_from(weight[weight_index + 1usize]);
            acc += grad[grad_index + 2usize] * Line::cast_from(weight[weight_index + 2usize]);
            acc += grad[grad_index + 3usize] * Line::cast_from(weight[weight_index + 3usize]);
            l += 4usize;
        }
        while l < latent {
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            let weight_index = (h * embd + e) * latent + l;
            acc += grad[grad_index] * Line::cast_from(weight[weight_index]);
            l += 1usize;
        }
    }

    let output_index = ((b * input_heads + input_head) * time + t) * embd + e;
    output[output_index] = acc * weight_scale;
}

#[cube(launch)]
fn packed_lowrank_grad_weight_cube_kernel(
    input: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    grad: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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
    let mut acc = Line::cast_from(0u32);
    let mut b = 0usize;
    while b < batch {
        let mut t = 0usize;
        while t < time {
            let input_index = ((b * input_heads + input_head) * time + t) * embd + e;
            let grad_index = ((b * heads + h) * time + t) * latent + l;
            acc += Line::cast_from(input[input_index]) * grad[grad_index];
            t += 1usize;
        }
        b += 1usize;
    }

    let output_index = (h * embd + e) * latent + l;
    output[output_index] = acc * activation_scale;
}

#[cube(launch)]
fn packed_decoder_tail_grad_input_cube_kernel(
    grad: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    weight: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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
    let mut acc = Line::cast_from(0u32);
    let mut d = 0usize;
    while d + 4usize <= dim {
        acc += grad[grad_base + d] * Line::cast_from(weight[weight_row_base + d]);
        acc += grad[grad_base + d + 1usize] * Line::cast_from(weight[weight_row_base + d + 1usize]);
        acc += grad[grad_base + d + 2usize] * Line::cast_from(weight[weight_row_base + d + 2usize]);
        acc += grad[grad_base + d + 3usize] * Line::cast_from(weight[weight_row_base + d + 3usize]);
        d += 4usize;
    }
    while d < dim {
        acc += grad[grad_base + d] * Line::cast_from(weight[weight_row_base + d]);
        d += 1usize;
    }
    let output_index = ((b * heads + h) * time + t) * latent + l;
    output[output_index] = acc * weight_scale;
}

#[cube(launch)]
fn packed_decoder_tail_grad_weight_cube_kernel(
    y: &burn_cubecl::cubecl::prelude::Tensor<Line<i32>>,
    grad: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    output: &mut burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
    params: &burn_cubecl::cubecl::prelude::Tensor<Line<f32>>,
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
    let mut acc = Line::cast_from(0u32);
    let mut b = 0usize;
    while b < batch {
        let mut t = 0usize;
        while t < time {
            let y_index = ((b * heads + h) * time + t) * latent + l;
            let grad_index = (b * time + t) * dim + d;
            acc += Line::cast_from(y[y_index]) * grad[grad_index];
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
