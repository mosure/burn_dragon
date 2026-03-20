use std::any::{Any, TypeId};
use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Shape, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::{base::Checkpointer, strategy::NoCheckpointing};
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops, OpsKind};
use burn_cubecl::cubecl::{prelude::*, server::Bindings};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::{Fusion, FusionTensor};
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::dense_scores::{ROW_NORM_EPS, dense_row_l1_scores_reference};
use crate::fusion_compat::register_fusion_float_tensor;

const WORKGROUP_SIZE_X: u32 = 64;
const MAX_FUSED_TIME: usize = 1024;
const META_LEN: usize = 8;
const DENSE_ATTENTION_SHADER: &str = include_str!("dense_attention.wgsl");

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
type WgpuFusionAutodiffBackend<BT> = Autodiff<WgpuFusionBackend<BT>>;
type WgpuFusionAutodiffTensor<BT> =
    <WgpuFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;

#[derive(Debug, Clone)]
pub struct CompiledDenseAttentionPlan<B: BackendTrait> {
    meta: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    value_dim: usize,
}

impl<B: BackendTrait> CompiledDenseAttentionPlan<B> {
    pub fn new(
        batch: usize,
        heads: usize,
        value_heads: usize,
        time: usize,
        latent: usize,
        value_dim: usize,
        device: &B::Device,
    ) -> Self {
        let inv_scale = 1.0 / (latent as f32).sqrt().max(1.0);
        let meta = BurnTensor::<B, 1>::from_data(
            TensorData::new(
                vec![
                    batch as f32,
                    heads as f32,
                    value_heads as f32,
                    time as f32,
                    latent as f32,
                    value_dim as f32,
                    inv_scale,
                    ROW_NORM_EPS,
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

pub fn supports_dense_attention_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u32>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u8>>()
}

pub fn try_fused_dense_row_l1_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_attention_backend::<B>() {
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
    let plan = CompiledDenseAttentionPlan::new(
        batch,
        heads,
        value_heads,
        time,
        latent,
        value_dim,
        &query.device(),
    );
    try_fused_dense_row_l1_attention_wgpu_with_plan(query, value, slopes, &plan)
}

pub fn try_fused_dense_row_l1_attention_wgpu_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    plan: &CompiledDenseAttentionPlan<B>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_attention_backend::<B>() || !plan.matches(query, value) {
        return None;
    }
    if slopes.shape().dims::<1>() != [plan.heads] {
        return None;
    }

    let meta = plan.meta();

    try_fusion_path_wgpu::<B, u32>(query, value, slopes, &meta)
        .or_else(|| try_fusion_path_wgpu::<B, u8>(query, value, slopes, &meta))
        .or_else(|| try_fusion_path_autodiff_wgpu::<B, u32>(query, value, slopes, &meta))
        .or_else(|| try_fusion_path_autodiff_wgpu::<B, u8>(query, value, slopes, &meta))
        .or_else(|| try_direct_path::<B>(query, value, slopes, &meta))
        .or_else(|| try_direct_path_autodiff_wgpu_cube::<B>(query, value, slopes, &meta))
}

fn try_fusion_path_wgpu<B, BT>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>>()
    {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let fusion_query: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim_query)?;
    let fusion_client = fusion_query.client.clone();
    let query =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let value = resolve_fusion_tensor_wgpu::<B, BT, 4>(value)?;
    let slopes = resolve_fusion_tensor_wgpu::<B, BT, 1>(slopes)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;
    let context = dense_row_l1_attention_wgsl_runtime::<WgpuRuntime>(query, value, slopes, meta);
    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_direct_path<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_query)?;
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_value)?;
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_slopes = slopes.clone().into_primitive().tensor();
    let slopes: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_slopes)?;
    if slopes.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let context = dense_row_l1_attention_wgsl_runtime::<WgpuRuntime>(query, value, slopes, meta);
    let context_prim = try_cast_backend::<B, _>(context)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_direct_path_autodiff_wgpu_cube<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_query)?;
    let query_inner: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    if query_inner.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_value)?;
    let value_inner: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    if value_inner.dtype != DType::F32 {
        return None;
    }

    let prim_slopes = slopes.clone().into_primitive().tensor();
    let slopes_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_slopes)?;
    let slopes_inner: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(slopes_ad.clone());
    if slopes_inner.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_meta)?;
    let meta_inner: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);
    if meta_inner.dtype != DType::F32 {
        return None;
    }

    let context_ad = fused_dense_attention_autodiff_wgpu(query_ad, value_ad, slopes_ad, meta_inner);
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

fn try_fusion_path_autodiff_wgpu<B, BT>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<BT>>() {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_query)?;
    let fusion_query: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query_ad.clone());
    let fusion_client = fusion_query.client.clone();
    let query =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_value)?;
    let fusion_value: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(value_ad.clone());
    let value =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_value);
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_slopes = slopes.clone().into_primitive().tensor();
    let slopes_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_slopes)?;
    let fusion_slopes: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(slopes_ad.clone());
    let slopes =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_slopes);
    if slopes.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_meta)?;
    let fusion_meta: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(meta_ad);
    let meta =
        fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_meta);
    if meta.dtype != DType::F32 {
        return None;
    }

    let context = dense_row_l1_attention_wgsl_runtime::<WgpuRuntime>(query, value, slopes, meta);
    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let context_ad = fused_dense_attention_autodiff_fusion_wgpu::<BT>(
        query_ad,
        value_ad,
        slopes_ad,
        context_fusion,
    );
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        context_prim,
    )))
}

#[derive(Debug)]
struct FusedDenseAttentionBackward<B>(PhantomData<B>);

fn fused_dense_attention_backward_impl<B: BackendTrait>(
    ops: Ops<
        (
            B::FloatTensorPrimitive,
            B::FloatTensorPrimitive,
            B::FloatTensorPrimitive,
        ),
        2,
    >,
    grads: &mut Gradients,
) {
    let grad_output = grads.consume::<B>(&ops.node);
    let (query_inner, value_inner, slopes_inner) = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query_inner));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value_inner));
    let slopes = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(slopes_inner));
    let grad_output = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_output));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_dims = value.shape().dims::<4>();
    let value_heads = value_dims[1];
    let value_dim = value_dims[3];

    let scores = dense_row_l1_scores_reference(query.clone(), slopes.clone());
    let value_flat = if value_heads == heads {
        value.clone().reshape([batch * heads, time, value_dim])
    } else {
        value
            .clone()
            .reshape([batch, 1, time, value_dim])
            .repeat_dim(1, heads)
            .reshape([batch * heads, time, value_dim])
    };
    let grad_scores = grad_output
        .clone()
        .reshape([batch * heads, time, value_dim])
        .matmul(value_flat.swap_dims(1, 2))
        .reshape([batch, heads, time, time]);

    if let Some(parent) = &parents[0] {
        let raw_scores = crate::dense_scores::dense_row_l1_raw_scores(query.clone(), slopes);
        let denom = raw_scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
        let row_dot = (grad_scores.clone() * raw_scores.clone()).sum_dim(3);
        let grad_raw =
            grad_scores / denom.clone() - raw_scores.sign() * (row_dot / (denom.clone() * denom));
        let grad_query = (grad_raw.clone().reshape([batch * heads, time, time])
            + grad_raw
                .clone()
                .reshape([batch * heads, time, time])
                .swap_dims(1, 2))
        .matmul(query.clone().reshape([batch * heads, time, latent]))
        .div_scalar((latent as f32).sqrt().max(1.0))
        .reshape([batch, heads, time, latent]);
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }

    if let Some(parent) = &parents[1] {
        let grad_value_per_head = scores
            .reshape([batch * heads, time, time])
            .swap_dims(1, 2)
            .matmul(grad_output.reshape([batch * heads, time, value_dim]))
            .reshape([batch, heads, time, value_dim]);
        let grad_value = if value_heads == heads {
            grad_value_per_head
        } else {
            grad_value_per_head
                .sum_dim(1)
                .reshape([batch, 1, time, value_dim])
        };
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 2> for FusedDenseAttentionBackward<WgpuCubeBackend> {
    type State = (
        CubeTensor<WgpuRuntime>,
        CubeTensor<WgpuRuntime>,
        CubeTensor<WgpuRuntime>,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        fused_dense_attention_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl<BT> Backward<WgpuFusionBackend<BT>, 2> for FusedDenseAttentionBackward<WgpuFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = (
        FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        fused_dense_attention_backward_impl::<WgpuFusionBackend<BT>>(ops, grads);
    }
}

fn fused_dense_attention_autodiff_wgpu(
    query: WgpuCubeAutodiffTensor,
    value: WgpuCubeAutodiffTensor,
    slopes: WgpuCubeAutodiffTensor,
    meta: CubeTensor<WgpuRuntime>,
) -> WgpuCubeAutodiffTensor {
    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value.clone());
    let slopes_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(slopes.clone());
    let context = dense_row_l1_attention_wgsl_runtime::<WgpuRuntime>(
        query_inner.clone(),
        value_inner.clone(),
        slopes_inner.clone(),
        meta,
    );

    match FusedDenseAttentionBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([query.node.clone(), value.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((query_inner, value_inner, slopes_inner), context),
        OpsKind::UnTracked(prep) => prep.finish(context),
    }
}

fn fused_dense_attention_autodiff_fusion_wgpu<BT: BoolElement + 'static>(
    query: WgpuFusionAutodiffTensor<BT>,
    value: WgpuFusionAutodiffTensor<BT>,
    slopes: WgpuFusionAutodiffTensor<BT>,
    context: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
) -> WgpuFusionAutodiffTensor<BT> {
    let query_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query.clone());
    let value_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(value.clone());
    let slopes_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(slopes.clone());

    match FusedDenseAttentionBackward::<WgpuFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([query.node.clone(), value.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((query_inner, value_inner, slopes_inner), context),
        OpsKind::UnTracked(prep) => prep.finish(context),
    }
}

fn dense_row_l1_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    slopes: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let slopes = into_contiguous(slopes);
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
    debug_assert_ne!(
        query.handle, context.handle,
        "dense attention fused context unexpectedly aliased query handle"
    );
    debug_assert_ne!(
        value.handle, context.handle,
        "dense attention fused context unexpectedly aliased value handle"
    );
    debug_assert_ne!(
        slopes.handle, context.handle,
        "dense attention fused context unexpectedly aliased slopes handle"
    );
    debug_assert_ne!(
        meta.handle, context.handle,
        "dense attention fused context unexpectedly aliased meta handle"
    );

    let workgroups_x = div_ceil_u32(value_dim as u32, WORKGROUP_SIZE_X);
    let workgroups_z = (batch * time) as u32;
    let count = CubeCount::Static(workgroups_x, heads as u32, workgroups_z);
    let kernel = SourceKernel::new(
        DenseRowL1AttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        context.handle.clone().binding(),
        slopes.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch dense row_l1 attention kernel");
    context
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn resolve_fusion_tensor_wgpu<B, BT, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

#[derive(Clone)]
struct DenseRowL1AttentionKernel;

impl KernelSource for DenseRowL1AttentionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(DENSE_ATTENTION_SHADER)
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
    use burn_wgpu::{RuntimeOptions, graphics};

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type AutodiffBackendImpl = Autodiff<Backend>;

    fn init_runtime(device: &<Backend as BackendTrait>::Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn reference_attention(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        slopes: Tensor<Backend, 1>,
    ) -> Tensor<Backend, 4> {
        let scores = dense_row_l1_scores_reference(query, slopes);
        let [batch, heads, time, _] = scores.shape().dims::<4>();
        let value_heads = value.shape().dims::<4>()[1];
        let value_dim = value.shape().dims::<4>()[3];
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

    fn reference_attention_autodiff(
        query: Tensor<AutodiffBackendImpl, 4>,
        value: Tensor<AutodiffBackendImpl, 4>,
        slopes: Tensor<AutodiffBackendImpl, 1>,
    ) -> Tensor<AutodiffBackendImpl, 4> {
        let scores = dense_row_l1_scores_reference(query, slopes);
        let [batch, heads, time, _] = scores.shape().dims::<4>();
        let value_heads = value.shape().dims::<4>()[1];
        let value_dim = value.shape().dims::<4>()[3];
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

    fn assert_close<const D: usize>(
        lhs: Tensor<Backend, D>,
        rhs: Tensor<Backend, D>,
        atol: f32,
        rtol: f32,
    ) {
        let lhs = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");
        for (index, (a, b)) in lhs.into_iter().zip(rhs).enumerate() {
            let diff = (a - b).abs();
            let tol = atol + rtol * b.abs();
            assert!(
                diff <= tol,
                "dense row_l1 attention mismatch at index {index}: lhs={a}, rhs={b}, diff={diff}, tol={tol}"
            );
        }
    }

    #[test]
    fn dense_row_l1_attention_matches_reference_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 4, 11, 8], Distribution::Default, &device);
        let value = Tensor::<Backend, 4>::random([2, 1, 11, 16], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..4).map(|i| 0.02 * i as f32).collect(), [4]),
            &device,
        );

        let actual = try_fused_dense_row_l1_attention_wgpu::<Backend>(&query, &value, &slopes)
            .expect("wgpu attention");
        let expected = reference_attention(query, value, slopes);
        assert_close(actual, expected, 2e-3, 2e-3);
    }

    #[test]
    fn dense_row_l1_attention_matches_reference_query_value_gradients_on_wgpu_autodiff() {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(
                (0..12).map(|i| (i as f32) * 0.05 - 0.2).collect(),
                [1, 1, 3, 4],
            ),
            &device,
        )
        .require_grad();
        let value = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(
                (0..18).map(|i| (i as f32) * 0.03 - 0.15).collect(),
                [1, 1, 3, 6],
            ),
            &device,
        )
        .require_grad();
        let slopes =
            Tensor::<AutodiffBackendImpl, 1>::from_data(TensorData::new(vec![0.1], [1]), &device);
        let weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(
                vec![
                    0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2, 0.1, -0.4, 0.3, -0.1, 0.2, -0.3,
                    0.4, 0.1, -0.2, 0.3,
                ],
                [1, 1, 3, 6],
            ),
            &device,
        );

        let fused =
            try_fused_dense_row_l1_attention_wgpu::<AutodiffBackendImpl>(&query, &value, &slopes)
                .expect("fused autodiff attention");
        let reference = reference_attention_autodiff(query.clone(), value.clone(), slopes.clone());

        let fused_grads = (fused * weights.clone()).sum().backward();
        let reference_grads = (reference * weights).sum().backward();

        let fused_query_grad = query
            .grad(&fused_grads)
            .expect("fused query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused query grad vec");
        let reference_query_grad = query
            .grad(&reference_grads)
            .expect("reference query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference query grad vec");
        let fused_value_grad = value
            .grad(&fused_grads)
            .expect("fused value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused value grad vec");
        let reference_value_grad = value
            .grad(&reference_grads)
            .expect("reference value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference value grad vec");

        assert_eq!(fused_query_grad.len(), reference_query_grad.len());
        assert_eq!(fused_value_grad.len(), reference_value_grad.len());
        for (index, (lhs, rhs)) in fused_query_grad
            .into_iter()
            .zip(reference_query_grad)
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() <= 2e-3,
                "query grad mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
        for (index, (lhs, rhs)) in fused_value_grad
            .into_iter()
            .zip(reference_value_grad)
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() <= 2e-3,
                "value grad mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn dense_row_l1_attention_matches_reference_query_value_gradients_on_wgpu_autodiff_medium_shape()
     {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 4, 65, 16],
            Distribution::Default,
            &device,
        )
        .require_grad();
        let value = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 1, 65, 64],
            Distribution::Default,
            &device,
        )
        .require_grad();
        let slopes = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new((0..4).map(|i| 0.02 * i as f32).collect(), [4]),
            &device,
        );
        let weights = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 4, 65, 64],
            Distribution::Default,
            &device,
        );

        let fused =
            try_fused_dense_row_l1_attention_wgpu::<AutodiffBackendImpl>(&query, &value, &slopes)
                .expect("fused autodiff attention");
        let reference = reference_attention_autodiff(query.clone(), value.clone(), slopes.clone());

        let fused_grads = (fused * weights.clone()).mean().backward();
        let reference_grads = (reference * weights).mean().backward();

        let fused_query_grad = query
            .grad(&fused_grads)
            .expect("fused query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused query grad vec");
        let reference_query_grad = query
            .grad(&reference_grads)
            .expect("reference query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference query grad vec");
        let fused_value_grad = value
            .grad(&fused_grads)
            .expect("fused value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused value grad vec");
        let reference_value_grad = value
            .grad(&reference_grads)
            .expect("reference value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference value grad vec");

        assert_eq!(fused_query_grad.len(), reference_query_grad.len());
        assert_eq!(fused_value_grad.len(), reference_value_grad.len());

        let mut max_query_diff = 0.0_f32;
        let mut max_value_diff = 0.0_f32;
        for (lhs, rhs) in fused_query_grad.into_iter().zip(reference_query_grad) {
            max_query_diff = max_query_diff.max((lhs - rhs).abs());
        }
        for (lhs, rhs) in fused_value_grad.into_iter().zip(reference_value_grad) {
            max_value_diff = max_value_diff.max((lhs - rhs).abs());
        }

        assert!(
            max_query_diff <= 5e-3,
            "medium-shape query grad drift too high: {max_query_diff}"
        );
        assert!(
            max_value_diff <= 5e-3,
            "medium-shape value grad drift too high: {max_value_diff}"
        );
    }

    #[test]
    fn dense_row_l1_attention_matches_reference_query_value_gradients_on_wgpu_autodiff_long_sequence()
     {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 8, 401, 16],
            Distribution::Default,
            &device,
        )
        .require_grad();
        let value = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 1, 401, 64],
            Distribution::Default,
            &device,
        )
        .require_grad();
        let slopes = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new((0..8).map(|i| 0.01 * i as f32).collect(), [8]),
            &device,
        );
        let weights = Tensor::<AutodiffBackendImpl, 4>::random(
            [1, 8, 401, 64],
            Distribution::Default,
            &device,
        );

        let fused =
            try_fused_dense_row_l1_attention_wgpu::<AutodiffBackendImpl>(&query, &value, &slopes)
                .expect("fused autodiff attention");
        let reference = reference_attention_autodiff(query.clone(), value.clone(), slopes.clone());

        let fused_grads = (fused * weights.clone()).mean().backward();
        let reference_grads = (reference * weights).mean().backward();

        let fused_query_grad = query
            .grad(&fused_grads)
            .expect("fused query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused query grad vec");
        let reference_query_grad = query
            .grad(&reference_grads)
            .expect("reference query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference query grad vec");
        let fused_value_grad = value
            .grad(&fused_grads)
            .expect("fused value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused value grad vec");
        let reference_value_grad = value
            .grad(&reference_grads)
            .expect("reference value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference value grad vec");

        let mut max_query_diff = 0.0_f32;
        let mut max_value_diff = 0.0_f32;
        for (lhs, rhs) in fused_query_grad.into_iter().zip(reference_query_grad) {
            max_query_diff = max_query_diff.max((lhs - rhs).abs());
        }
        for (lhs, rhs) in fused_value_grad.into_iter().zip(reference_value_grad) {
            max_value_diff = max_value_diff.max((lhs - rhs).abs());
        }

        assert!(
            max_query_diff <= 1e-2,
            "long-sequence query grad drift too high: {max_query_diff}"
        );
        assert!(
            max_value_diff <= 1e-2,
            "long-sequence value grad drift too high: {max_value_diff}"
        );
    }

    #[test]
    fn dense_row_l1_attention_matches_reference_with_compiled_plan_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 4, 11, 8], Distribution::Default, &device);
        let value = Tensor::<Backend, 4>::random([2, 1, 11, 16], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..4).map(|i| 0.02 * i as f32).collect(), [4]),
            &device,
        );
        let plan = CompiledDenseAttentionPlan::new(2, 4, 1, 11, 8, 16, &device);

        let actual = try_fused_dense_row_l1_attention_wgpu_with_plan::<Backend>(
            &query, &value, &slopes, &plan,
        )
        .expect("wgpu attention with plan");
        let expected = reference_attention(query, value, slopes);
        assert_close(actual, expected, 2e-3, 2e-3);
    }
}
