use std::any::{Any, TypeId};

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Int, Shape, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::{prelude::*, server::Bindings};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::{Fusion, FusionTensor};
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;

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

fn try_fusion_path_runtime<B, BT, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R, BT>>>() {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let fusion_query: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim_query)?;
    let fusion_client = fusion_query.client.clone();
    let query = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let value = resolve_fusion_tensor_runtime::<B, BT, R, 4>(value)?;
    let decay = resolve_fusion_tensor_runtime::<B, BT, R, 1>(decay)?;
    let meta = resolve_fusion_tensor_runtime::<B, BT, R, 1>(meta)?;
    let context = dense_causal_attention_runtime::<R>(query, value, decay, meta);
    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_direct_path_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query: CubeTensor<R> = try_cast_primitive::<B, _>(prim_query)?;
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value: CubeTensor<R> = try_cast_primitive::<B, _>(prim_value)?;
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_decay = decay.clone().into_primitive().tensor();
    let decay: CubeTensor<R> = try_cast_primitive::<B, _>(prim_decay)?;
    if decay.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta: CubeTensor<R> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let context = dense_causal_attention_runtime::<R>(query, value, decay, meta);
    let context_prim = try_cast_backend::<B, _>(context)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_direct_path_autodiff_cube_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_query)?;
    let query_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(query_ad.clone())?;
    if query_inner.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_value)?;
    let value_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(value_ad.clone())?;
    if value_inner.dtype != DType::F32 {
        return None;
    }

    let prim_decay = decay.clone().into_primitive().tensor();
    let decay_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_decay)?;
    let decay_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(decay_ad.clone())?;
    if decay_inner.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_meta)?;
    let meta_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(meta_ad)?;
    if meta_inner.dtype != DType::F32 {
        return None;
    }

    let context =
        dense_causal_attention_runtime::<R>(query_inner, value_inner, decay_inner, meta_inner);
    let context_ad = wrap_autodiff_inner::<B, R>(context)?;
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_fusion_path_autodiff_runtime<B, BT, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_autodiff_fusion_type::<B, BT, R>() {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_query)?;
    let fusion_query: FusionTensor<FusionCubeRuntime<R, BT>> =
        extract_fusion_autodiff_inner::<B, BT, R>(query_ad.clone())?;
    let fusion_client = fusion_query.client.clone();
    let query = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_value)?;
    let fusion_value: FusionTensor<FusionCubeRuntime<R, BT>> =
        extract_fusion_autodiff_inner::<B, BT, R>(value_ad.clone())?;
    let value = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_value);
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_decay = decay.clone().into_primitive().tensor();
    let decay_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_decay)?;
    let fusion_decay: FusionTensor<FusionCubeRuntime<R, BT>> =
        extract_fusion_autodiff_inner::<B, BT, R>(decay_ad.clone())?;
    let decay = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_decay);
    if decay.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_meta)?;
    let fusion_meta: FusionTensor<FusionCubeRuntime<R, BT>> =
        extract_fusion_autodiff_inner::<B, BT, R>(meta_ad)?;
    let meta = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_meta);
    if meta.dtype != DType::F32 {
        return None;
    }

    let context = dense_causal_attention_wgsl_runtime::<R>(query, value, decay, meta);
    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let context_ad = wrap_fusion_autodiff_inner::<B, BT, R>(context_fusion)?;
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn dense_causal_attention_runtime<R: CubeRuntime + 'static>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            return dense_causal_attention_cube_runtime::<R>(query, value, decay, meta);
        }
    }
    dense_causal_attention_wgsl_runtime::<R>(query, value, decay, meta)
}

fn dense_causal_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let value_dim = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, value_dim]),
    );

    let workgroups_x = div_ceil_u32(value_dim as u32, WORKGROUP_SIZE_X);
    let workgroups_z = (batch * time) as u32;
    let count = CubeCount::Static(workgroups_x, heads as u32, workgroups_z);
    let kernel = SourceKernel::new(
        DenseCausalAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        context.handle.clone().binding(),
        decay.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch dense causal attention kernel");
    context
}

fn dense_causal_attention_cube_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let value_dim = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, value_dim]),
    );

    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(value_dim as u32, WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );

    let _ = dense_causal_attention_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        query.as_tensor_arg(1),
        value.as_tensor_arg(1),
        context.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        meta.as_tensor_arg(1),
        MAX_FUSED_TIME,
    );

    context
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cube(launch)]
fn dense_causal_attention_cube_kernel(
    query: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    context: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
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

    let mut row_scores = SharedMemory::<f32>::new_lined(max_fused_time, 1usize);
    let decay_value = decay[h * decay.stride(0)];

    let mut col = lane;
    while col < row {
        let mut dot = Line::cast_from(0u32);
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
        let exponent = Line::cast_from((row - col) as u32);
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
    let mut acc = Line::cast_from(0u32);
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
    let fusion: FusionTensor<FusionCubeRuntime<R, BT>> = try_cast_primitive::<B, _>(prim)?;
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

fn extract_autodiff_inner<B, R>(value: B::FloatTensorPrimitive) -> Option<CubeTensor<R>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let query_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(value)?;
        let inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad);
        let boxed: Box<dyn Any> = Box::new(inner);
        return boxed.downcast::<CubeTensor<R>>().ok().map(|boxed| *boxed);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let query_ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(value)?;
            let inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad);
            let boxed: Box<dyn Any> = Box::new(inner);
            return boxed.downcast::<CubeTensor<R>>().ok().map(|boxed| *boxed);
        }
    }
    None
}

fn wrap_autodiff_inner<B, R>(value: CubeTensor<R>) -> Option<B::FloatTensorPrimitive>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let boxed: Box<dyn Any> = Box::new(value);
        let inner = boxed
            .downcast::<CubeTensor<WgpuRuntime>>()
            .ok()
            .map(|boxed| *boxed)?;
        let ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(inner);
        return try_cast_backend::<B, _>(ad);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let boxed: Box<dyn Any> = Box::new(value);
            let inner = boxed
                .downcast::<CubeTensor<CudaRuntime>>()
                .ok()
                .map(|boxed| *boxed)?;
            let ad = <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(inner);
            return try_cast_backend::<B, _>(ad);
        }
    }
    None
}

fn extract_fusion_autodiff_inner<B, BT, R>(
    value: B::FloatTensorPrimitive,
) -> Option<FusionTensor<FusionCubeRuntime<R, BT>>>
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
            .downcast::<FusionTensor<FusionCubeRuntime<R, BT>>>()
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
                .downcast::<FusionTensor<FusionCubeRuntime<R, BT>>>()
                .ok()
                .map(|boxed| *boxed);
        }
    }
    None
}

fn wrap_fusion_autodiff_inner<B, BT, R>(
    value: FusionTensor<FusionCubeRuntime<R, BT>>,
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
            .downcast::<FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>>()
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
                .downcast::<FusionTensor<FusionCubeRuntime<CudaRuntime, BT>>>()
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
mod tests {
    use super::*;
    use burn::tensor::{Distribution, Tensor};
    #[cfg(feature = "cuda")]
    use burn_cuda::Cuda;
    use burn_wgpu::{CubeBackend, RuntimeOptions, graphics};

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn init_runtime(device: &<Backend as BackendTrait>::Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn assert_close_backend<B: BackendTrait>(
        lhs: Tensor<B, 4>,
        rhs: Tensor<B, 4>,
        atol: f32,
        rtol: f32,
    ) {
        let lhs_data = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs_data = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");
        let mut max_diff = 0.0_f32;
        let mut max_tol = 0.0_f32;
        let mut max_lhs = 0.0_f32;
        let mut max_rhs = 0.0_f32;
        for (a, b) in lhs_data.iter().zip(rhs_data.iter()) {
            let diff = (a - b).abs();
            let tol = atol + rtol * b.abs();
            if diff > max_diff {
                max_diff = diff;
                max_tol = tol;
                max_lhs = *a;
                max_rhs = *b;
            }
        }
        assert!(
            max_diff <= max_tol,
            "max difference {max_diff} exceeds tolerance {max_tol} (lhs={max_lhs}, rhs={max_rhs})"
        );
    }

    fn assert_close(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>, atol: f32, rtol: f32) {
        assert_close_backend(lhs, rhs, atol, rtol);
    }

    fn reference_attention(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        decay: Tensor<Backend, 1>,
    ) -> Tensor<Backend, 4> {
        dense_causal_attention_reference(query, value, decay)
    }

    #[test]
    fn dense_causal_attention_matches_reference_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 17);

        let query =
            Tensor::<Backend, 4>::random([2, 4, 16, 32], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 16, 24], Distribution::Normal(0.0, 1.0), &device);
        let decay = Tensor::<Backend, 1>::from_floats([0.97, 0.93, 0.89, 0.85], &device);

        let fused = try_fused_dense_causal_attention_wgpu::<Backend>(&query, &value, &decay)
            .expect("wgpu dense causal attention");
        let expected = reference_attention(query, value, decay);
        assert_close(fused, expected, 2e-4, 2e-4);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dense_causal_attention_supports_cuda_backend_types() {
        type CudaBackend = Cuda<f32, i32>;
        type CudaAutodiffBackend = Autodiff<CudaBackend>;

        assert!(supports_dense_causal_attention_backend::<CudaBackend>());
        assert!(supports_dense_causal_attention_backend::<CudaAutodiffBackend>());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn dense_causal_attention_matches_reference_on_cuda() {
        type CudaBackend = Cuda<f32, i32>;
        let device = <CudaBackend as BackendTrait>::Device::default();
        <CudaBackend as BackendTrait>::seed(&device, 17);

        let query = Tensor::<CudaBackend, 4>::random(
            [1, 2, 8, 16],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let value = Tensor::<CudaBackend, 4>::random(
            [1, 1, 8, 12],
            Distribution::Normal(0.0, 1.0),
            &device,
        );
        let decay = Tensor::<CudaBackend, 1>::from_floats([0.97, 0.93], &device);

        let fused = try_fused_dense_causal_attention_wgpu::<CudaBackend>(&query, &value, &decay)
            .expect("cuda dense causal attention");
        let expected = dense_causal_attention_reference(query, value, decay);
        let _ = <CudaBackend as BackendTrait>::sync(&device);
        assert_close_backend(fused, expected, 2e-2, 2e-2);
    }
}
