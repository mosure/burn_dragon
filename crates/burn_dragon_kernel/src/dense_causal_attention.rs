use std::any::{Any, TypeId};
use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Int, Shape, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::NoCheckpointing};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::{prelude::*, server::KernelArguments};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::{Fusion, FusionTensor};
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;

mod backward_runtime;
mod forward_runtime;

use self::backward_runtime::dense_causal_attention_autodiff_custom;
use self::forward_runtime::{
    try_direct_path_autodiff_cube_runtime, try_direct_path_runtime,
    try_fusion_path_autodiff_runtime, try_fusion_path_runtime,
};

const WORKGROUP_SIZE_X: u32 = 64;
const MAX_FUSED_TIME: usize = 1024;
const META_LEN: usize = 6;
const DENSE_CAUSAL_ATTENTION_SHADER: &str = include_str!("dense_causal_attention.wgsl");

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
type WgpuFusionAutodiffBackend<BT> = Autodiff<WgpuFusionBackend<BT>>;
type WgpuFusionAutodiffTensor<BT> =
    <WgpuFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaFusionBackend<BT> = Fusion<CubeBackend<CudaRuntime, f32, i32, BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffBackend<BT> = Autodiff<CudaFusionBackend<BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffTensor<BT> =
    <CudaFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;

#[derive(Debug, Clone)]
struct DenseCausalAttentionBackwardState<T> {
    query: T,
    value: T,
    decay: T,
}

#[derive(Debug)]
struct FusedDenseCausalAttentionBackward<B>(PhantomData<B>);

#[derive(Debug, Clone)]
pub struct CompiledDenseCausalAttentionPlan<B: BackendTrait> {
    meta: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    value_dim: usize,
}

impl<B: BackendTrait> CompiledDenseCausalAttentionPlan<B> {
    pub fn new(
        batch: usize,
        heads: usize,
        value_heads: usize,
        time: usize,
        latent: usize,
        value_dim: usize,
        device: &B::Device,
    ) -> Self {
        let meta = BurnTensor::<B, 1>::from_data(
            TensorData::new(
                vec![
                    batch as f32,
                    heads as f32,
                    value_heads as f32,
                    time as f32,
                    latent as f32,
                    value_dim as f32,
                ],
                [META_LEN],
            ),
            device,
        );
        Self {
            meta,
            batch,
            heads,
            value_heads,
            time,
            latent,
            value_dim,
        }
    }

    fn matches(&self, query: &BurnTensor<B, 4>, value: &BurnTensor<B, 4>) -> bool {
        query.shape().dims::<4>() == [self.batch, self.heads, self.time, self.latent]
            && value.shape().dims::<4>()
                == [self.batch, self.value_heads, self.time, self.value_dim]
    }

    fn meta(&self) -> BurnTensor<B, 1> {
        self.meta.clone()
    }
}

pub fn supports_dense_causal_attention_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u32>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u8>>()
        || {
            #[cfg(feature = "cuda")]
            {
                matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>()
                    || matches_type::<B::FloatTensorPrimitive, CudaCubeAutodiffTensor>()
                    || matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<u32>>()
                    || matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<u8>>()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        }
}

pub fn try_fused_dense_causal_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_causal_attention_backend::<B>() {
        return None;
    }

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [value_batch, value_heads, value_time, value_dim] = value.shape().dims::<4>();
    if batch == 0 || heads == 0 || time == 0 || latent == 0 || value_dim == 0 {
        return None;
    }
    if time > MAX_FUSED_TIME {
        return None;
    }
    if value_batch != batch || value_time != time {
        return None;
    }
    if value_heads != 1 && value_heads != heads {
        return None;
    }
    let plan = CompiledDenseCausalAttentionPlan::new(
        batch,
        heads,
        value_heads,
        time,
        latent,
        value_dim,
        &query.device(),
    );
    try_fused_dense_causal_attention_wgpu_with_plan(query, value, decay, &plan)
}

pub fn try_fused_dense_causal_attention_wgpu_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    plan: &CompiledDenseCausalAttentionPlan<B>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_causal_attention_backend::<B>() || !plan.matches(query, value) {
        return None;
    }
    if decay.shape().dims::<1>() != [plan.heads] {
        return None;
    }

    let meta = plan.meta();

    try_fusion_path_runtime::<B, u32, WgpuRuntime>(query, value, decay, &meta)
        .or_else(|| try_fusion_path_runtime::<B, u8, WgpuRuntime>(query, value, decay, &meta))
        .or_else(|| {
            try_fusion_path_autodiff_runtime::<B, u32, WgpuRuntime>(query, value, decay, &meta)
        })
        .or_else(|| {
            try_fusion_path_autodiff_runtime::<B, u8, WgpuRuntime>(query, value, decay, &meta)
        })
        .or_else(|| try_direct_path_runtime::<B, WgpuRuntime>(query, value, decay, &meta))
        .or_else(|| {
            try_direct_path_autodiff_cube_runtime::<B, WgpuRuntime>(query, value, decay, &meta)
        })
        .or_else(|| {
            #[cfg(feature = "cuda")]
            {
                try_fusion_path_runtime::<B, u32, CudaRuntime>(query, value, decay, &meta)
                    .or_else(|| {
                        try_fusion_path_runtime::<B, u8, CudaRuntime>(query, value, decay, &meta)
                    })
                    .or_else(|| {
                        try_fusion_path_autodiff_runtime::<B, u32, CudaRuntime>(
                            query, value, decay, &meta,
                        )
                    })
                    .or_else(|| {
                        try_fusion_path_autodiff_runtime::<B, u8, CudaRuntime>(
                            query, value, decay, &meta,
                        )
                    })
                    .or_else(|| {
                        try_direct_path_runtime::<B, CudaRuntime>(query, value, decay, &meta)
                    })
                    .or_else(|| {
                        try_direct_path_autodiff_cube_runtime::<B, CudaRuntime>(
                            query, value, decay, &meta,
                        )
                    })
            }
            #[cfg(not(feature = "cuda"))]
            {
                None
            }
        })
}

pub(crate) fn dense_causal_scores_reference<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
) -> BurnTensor<B, 4> {
    let [_, heads, time, _] = query.shape().dims::<4>();
    let pos_row = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, 1, time]);
    let diff = (pos_row - pos_col).tril(-1);
    let decay_matrix = decay
        .reshape([1, heads, 1, 1])
        .repeat_dim(2, time)
        .repeat_dim(3, time)
        .powf(diff);
    query.clone().matmul(query.swap_dims(2, 3)).tril(-1) * decay_matrix
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dense_causal_attention_reference<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
) -> BurnTensor<B, 4> {
    let [batch, heads, time, _latent] = query.shape().dims::<4>();
    let value_dim = value.shape().dims::<4>()[3];
    let value_heads = value.shape().dims::<4>()[1];
    let scores = dense_causal_scores_reference(query, decay);
    let value_flat = if value_heads == heads {
        value.reshape([batch * heads, time, value_dim])
    } else {
        value
            .reshape([batch, 1, time, value_dim])
            .repeat_dim(1, heads)
            .reshape([batch * heads, time, value_dim])
    };
    scores
        .reshape([batch * heads, time, time])
        .matmul(value_flat)
        .reshape([batch, heads, time, value_dim])
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cube(launch)]
fn dense_causal_attention_cube_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    context: &mut Tensor<f32>,
    decay: &Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_fused_time: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let value_dim = u32::cast_from(params[5]) as usize;

    let h = CUBE_POS_Y as usize;
    let batch_row = CUBE_POS_Z as usize;
    let b = batch_row / time;
    let row = batch_row % time;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let lane = UNIT_POS_X as usize;

    if b >= batch || h >= heads || row >= time || time > max_fused_time {
        terminate!();
    }

    let mut row_scores = SharedMemory::<f32>::new_aligned(max_fused_time, 1usize);
    let decay_value = decay[h * decay.stride(0)];

    let mut col = lane;
    while col < row {
        let mut dot = f32::cast_from(0u32);
        let mut l = 0usize;
        while l < latent {
            let q_row = query[b * query.stride(0)
                + h * query.stride(1)
                + row * query.stride(2)
                + l * query.stride(3)];
            let q_col = query[b * query.stride(0)
                + h * query.stride(1)
                + col * query.stride(2)
                + l * query.stride(3)];
            dot += q_row * q_col;
            l += 1usize;
        }
        let exponent = f32::cast_from((row - col) as u32);
        row_scores[col] = dot * decay_value.powf(exponent);
        col += CUBE_DIM_X as usize;
    }

    sync_cube();

    if e >= value_dim {
        terminate!();
    }

    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut acc = f32::cast_from(0u32);
    col = 0usize;
    while col < row {
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + col * value.stride(2)
            + e * value.stride(3);
        acc += row_scores[col] * value[value_index];
        col += 1usize;
    }

    let out_index = b * context.stride(0)
        + h * context.stride(1)
        + row * context.stride(2)
        + e * context.stride(3);
    context[out_index] = acc;
}

fn resolve_fusion_tensor_runtime<B, BT, R, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<R>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<R>> = try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn matches_autodiff_fusion_type<B, BT, R>() -> bool
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<BT>>()
    } else {
        #[cfg(feature = "cuda")]
        {
            if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
                return matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<BT>>();
            }
        }
        false
    }
}

fn extract_fusion_autodiff_inner<B, BT, R>(
    value: B::FloatTensorPrimitive,
) -> Option<FusionTensor<FusionCubeRuntime<R>>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let query_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(value)?;
        let inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query_ad);
        let boxed: Box<dyn Any> = Box::new(inner);
        return boxed
            .downcast::<FusionTensor<FusionCubeRuntime<R>>>()
            .ok()
            .map(|boxed| *boxed);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let query_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(value)?;
            let inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query_ad);
            let boxed: Box<dyn Any> = Box::new(inner);
            return boxed
                .downcast::<FusionTensor<FusionCubeRuntime<R>>>()
                .ok()
                .map(|boxed| *boxed);
        }
    }
    None
}

fn wrap_fusion_autodiff_inner<B, BT, R>(
    value: FusionTensor<FusionCubeRuntime<R>>,
) -> Option<B::FloatTensorPrimitive>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let boxed: Box<dyn Any> = Box::new(value);
        let inner = boxed
            .downcast::<FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
            .ok()
            .map(|boxed| *boxed)?;
        let ad = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
        return try_cast_backend::<B, _>(ad);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let boxed: Box<dyn Any> = Box::new(value);
            let inner = boxed
                .downcast::<FusionTensor<FusionCubeRuntime<CudaRuntime>>>()
                .ok()
                .map(|boxed| *boxed)?;
            let ad = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
            return try_cast_backend::<B, _>(ad);
        }
    }
    None
}

#[derive(Clone)]
struct DenseCausalAttentionKernel;

impl KernelSource for DenseCausalAttentionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(DENSE_CAUSAL_ATTENTION_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

#[cfg(test)]
mod tests;
