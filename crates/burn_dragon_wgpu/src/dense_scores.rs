use std::any::{Any, TypeId};
use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Int, Shape, TensorData, TensorPrimitive};
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

use crate::fusion_compat::register_fusion_float_tensor;

const WORKGROUP_SIZE_X: u32 = 64;
const META_LEN: usize = 6;
const DENSE_SCORES_SHADER: &str = include_str!("dense_scores.wgsl");
const DENSE_SCORES_WITH_DENOM_SHADER: &str = include_str!("dense_scores_with_denom.wgsl");
pub(crate) const ROW_NORM_EPS: f32 = 1e-6;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
type WgpuFusionAutodiffBackend<BT> = Autodiff<WgpuFusionBackend<BT>>;
type WgpuFusionAutodiffTensor<BT> =
    <WgpuFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;

#[derive(Debug, Clone)]
pub struct CompiledDenseScoresPlan<B: BackendTrait> {
    meta: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
}

impl<B: BackendTrait> CompiledDenseScoresPlan<B> {
    pub fn new(batch: usize, heads: usize, time: usize, latent: usize, device: &B::Device) -> Self {
        let inv_scale = 1.0 / (latent as f32).sqrt().max(1.0);
        let meta = BurnTensor::<B, 1>::from_data(
            TensorData::new(
                vec![
                    batch as f32,
                    heads as f32,
                    time as f32,
                    latent as f32,
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
            time,
            latent,
        }
    }

    fn matches(&self, query: &BurnTensor<B, 4>) -> bool {
        query.shape().dims::<4>() == [self.batch, self.heads, self.time, self.latent]
    }

    fn meta(&self) -> BurnTensor<B, 1> {
        self.meta.clone()
    }
}

pub fn supports_dense_scores_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u32>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<u8>>()
}

pub fn try_fused_dense_row_l1_scores_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_scores_backend::<B>() {
        return None;
    }

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    if batch == 0 || heads == 0 || time == 0 || latent == 0 {
        return None;
    }
    let plan = CompiledDenseScoresPlan::new(batch, heads, time, latent, &query.device());
    try_fused_dense_row_l1_scores_wgpu_with_plan(query, slopes, &plan)
}

pub fn try_fused_dense_row_l1_scores_wgpu_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    plan: &CompiledDenseScoresPlan<B>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_scores_backend::<B>() || !plan.matches(query) {
        return None;
    }
    if slopes.shape().dims::<1>() != [plan.heads] {
        return None;
    }

    let meta = plan.meta();

    try_fusion_path_wgpu::<B, u32>(query, slopes, &meta)
        .or_else(|| try_fusion_path_wgpu::<B, u8>(query, slopes, &meta))
        .or_else(|| try_fusion_path_autodiff_wgpu::<B, u32>(query, slopes, &meta))
        .or_else(|| try_fusion_path_autodiff_wgpu::<B, u8>(query, slopes, &meta))
        .or_else(|| try_direct_path::<B>(query, slopes, &meta))
        .or_else(|| try_direct_path_autodiff_wgpu_cube::<B>(query, slopes, &meta))
}

#[allow(dead_code)]
pub(crate) fn try_fused_dense_row_l1_scores_and_denom_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_scores_backend::<B>() {
        return None;
    }

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    if batch == 0 || heads == 0 || time == 0 || latent == 0 {
        return None;
    }
    let plan = CompiledDenseScoresPlan::new(batch, heads, time, latent, &query.device());
    try_fused_dense_row_l1_scores_and_denom_wgpu_with_plan(query, slopes, &plan)
}

#[allow(dead_code)]
pub(crate) fn try_fused_dense_row_l1_scores_and_denom_wgpu_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    plan: &CompiledDenseScoresPlan<B>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_dense_scores_backend::<B>() || !plan.matches(query) {
        return None;
    }
    if slopes.shape().dims::<1>() != [plan.heads] {
        return None;
    }

    let meta = plan.meta();

    try_fusion_path_wgpu_scores_and_denom::<B, u32>(query, slopes, &meta)
        .or_else(|| try_fusion_path_wgpu_scores_and_denom::<B, u8>(query, slopes, &meta))
        .or_else(|| try_direct_path_scores_and_denom::<B>(query, slopes, &meta))
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn dense_row_l1_scores_reference<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    slopes: BurnTensor<B, 1>,
) -> BurnTensor<B, 4> {
    let scores = dense_row_l1_raw_scores(query, slopes);
    let denom = dense_row_l1_denom_from_raw_scores(scores.clone());
    scores / denom
}

#[allow(dead_code)]
pub(crate) fn dense_row_l1_scores_reference_with_denom<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    slopes: BurnTensor<B, 1>,
) -> (BurnTensor<B, 4>, BurnTensor<B, 4>) {
    let scores = dense_row_l1_raw_scores(query, slopes);
    let denom = dense_row_l1_denom_from_raw_scores(scores.clone());
    let normalized = scores / denom.clone();
    (normalized, denom)
}

#[allow(dead_code)]
pub(crate) fn dense_row_l1_denom_from_raw_scores<B: BackendTrait>(
    scores: BurnTensor<B, 4>,
) -> BurnTensor<B, 4> {
    let [batch, heads, time, _] = scores.shape().dims::<4>();
    scores
        .abs()
        .sum_dim(3)
        .reshape([batch, heads, time, 1])
        .add_scalar(ROW_NORM_EPS)
}

pub(crate) fn dense_row_l1_raw_scores<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    slopes: BurnTensor<B, 1>,
) -> BurnTensor<B, 4> {
    let latent = query.shape().dims::<4>()[3] as f32;
    let scale = latent.sqrt().max(1.0);
    let [_, heads, time, _] = query.shape().dims::<4>();
    let slopes = slopes.reshape([1, heads, 1, 1]);
    let pos_row = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, 1, time]);
    query
        .clone()
        .div_scalar(scale)
        .matmul(query.swap_dims(2, 3))
        + slopes * (pos_col - pos_row)
}

fn try_fusion_path_wgpu<B, BT>(
    query: &BurnTensor<B, 4>,
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

    let slopes = resolve_fusion_tensor_wgpu::<B, BT, 1>(slopes)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;
    let scores = dense_row_l1_scores_wgsl_runtime::<WgpuRuntime>(query, slopes, meta);
    let scores_fusion = register_fusion_float_tensor(&fusion_client, scores);
    let scores_prim = try_cast_backend::<B, _>(scores_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        scores_prim,
    )))
}

fn try_direct_path<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
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

    let scores = dense_row_l1_scores_wgsl_runtime::<WgpuRuntime>(query, slopes, meta);
    let scores_prim = try_cast_backend::<B, _>(scores)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        scores_prim,
    )))
}

fn try_fusion_path_wgpu_scores_and_denom<B, BT>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
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

    let slopes = resolve_fusion_tensor_wgpu::<B, BT, 1>(slopes)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;
    let (scores, denom) =
        dense_row_l1_scores_and_denom_wgsl_runtime::<WgpuRuntime>(query, slopes, meta);
    let scores_fusion = register_fusion_float_tensor(&fusion_client, scores);
    let denom_fusion = register_fusion_float_tensor(&fusion_client, denom);
    let scores_prim = try_cast_backend::<B, _>(scores_fusion)?;
    let denom_prim = try_cast_backend::<B, _>(denom_fusion)?;
    Some((
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(scores_prim)),
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(denom_prim)),
    ))
}

fn try_direct_path_scores_and_denom<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    slopes: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_query)?;
    if query.dtype != DType::F32 {
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

    let (scores, denom) =
        dense_row_l1_scores_and_denom_wgsl_runtime::<WgpuRuntime>(query, slopes, meta);
    let scores_prim = try_cast_backend::<B, _>(scores)?;
    let denom_prim = try_cast_backend::<B, _>(denom)?;
    Some((
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(scores_prim)),
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(denom_prim)),
    ))
}

fn try_direct_path_autodiff_wgpu_cube<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
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

    let scores_ad = fused_dense_scores_autodiff_wgpu(query_ad, slopes_ad, meta_inner);
    let scores_prim = try_cast_backend::<B, _>(scores_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        scores_prim,
    )))
}

fn try_fusion_path_autodiff_wgpu<B, BT>(
    query: &BurnTensor<B, 4>,
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

    let scores = dense_row_l1_scores_wgsl_runtime::<WgpuRuntime>(query, slopes, meta);
    let scores_fusion = register_fusion_float_tensor(&fusion_client, scores);
    let scores_ad =
        fused_dense_scores_autodiff_fusion_wgpu::<BT>(query_ad, slopes_ad, scores_fusion);
    let scores_prim = try_cast_backend::<B, _>(scores_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        scores_prim,
    )))
}

#[derive(Debug)]
struct FusedDenseScoresBackward<B>(PhantomData<B>);

fn fused_dense_scores_backward_impl<B: BackendTrait>(
    ops: Ops<(B::FloatTensorPrimitive, B::FloatTensorPrimitive), 2>,
    grads: &mut Gradients,
) {
    let grad_output = grads.consume::<B>(&ops.node);
    let (query_inner, slopes_inner) = ops.state;
    let parents = ops.parents;
    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query_inner));
    let slopes = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(slopes_inner));
    let grad_output = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_output));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let raw_scores = dense_row_l1_raw_scores(query.clone(), slopes.clone());
    let denom = raw_scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
    let row_dot = (grad_output.clone() * raw_scores.clone()).sum_dim(3);
    let grad_raw =
        grad_output / denom.clone() - raw_scores.sign() * (row_dot / (denom.clone() * denom));

    if let Some(parent) = &parents[0] {
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
        let pos_row = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
            .float()
            .reshape([1, 1, time, 1]);
        let pos_col = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
            .float()
            .reshape([1, 1, 1, time]);
        let grad_slopes = (grad_raw * (pos_col - pos_row))
            .sum_dim(0)
            .sum_dim(2)
            .sum_dim(3)
            .reshape([heads]);
        grads.register::<B>(parent.id, grad_slopes.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 2> for FusedDenseScoresBackward<WgpuCubeBackend> {
    type State = (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>);

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        fused_dense_scores_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl<BT> Backward<WgpuFusionBackend<BT>, 2> for FusedDenseScoresBackward<WgpuFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = (
        FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
    );

    fn backward(
        self,
        ops: Ops<Self::State, 2>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        fused_dense_scores_backward_impl::<WgpuFusionBackend<BT>>(ops, grads);
    }
}

fn fused_dense_scores_autodiff_wgpu(
    query: WgpuCubeAutodiffTensor,
    slopes: WgpuCubeAutodiffTensor,
    meta: CubeTensor<WgpuRuntime>,
) -> WgpuCubeAutodiffTensor {
    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query.clone());
    let slopes_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(slopes.clone());
    let scores = dense_row_l1_scores_wgsl_runtime::<WgpuRuntime>(
        query_inner.clone(),
        slopes_inner.clone(),
        meta,
    );

    match FusedDenseScoresBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([query.node.clone(), slopes.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((query_inner, slopes_inner), scores),
        OpsKind::UnTracked(prep) => prep.finish(scores),
    }
}

fn fused_dense_scores_autodiff_fusion_wgpu<BT: BoolElement + 'static>(
    query: WgpuFusionAutodiffTensor<BT>,
    slopes: WgpuFusionAutodiffTensor<BT>,
    scores: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>,
) -> WgpuFusionAutodiffTensor<BT> {
    let query_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query.clone());
    let slopes_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(slopes.clone());

    match FusedDenseScoresBackward::<WgpuFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([query.node.clone(), slopes.node.clone()])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish((query_inner, slopes_inner), scores),
        OpsKind::UnTracked(prep) => prep.finish(scores),
    }
}

fn dense_row_l1_scores_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    slopes: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let slopes = into_contiguous(slopes);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let client = query.client.clone();
    let device = query.device.clone();
    let scores = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, time]),
    );

    let workgroups_x = div_ceil_u32(time as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(workgroups_x, heads as u32, batch as u32);
    let kernel = SourceKernel::new(
        DenseRowL1ScoresKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        scores.handle.clone().binding(),
        slopes.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch dense row_l1 scores kernel");
    scores
}

fn dense_row_l1_scores_and_denom_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    slopes: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let query = into_contiguous(query);
    let slopes = into_contiguous(slopes);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let client = query.client.clone();
    let device = query.device.clone();
    let scores = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, time]),
    );
    let denom = empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time, 1]));

    let workgroups_x = div_ceil_u32(time as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(workgroups_x, heads as u32, batch as u32);
    let kernel = SourceKernel::new(
        DenseRowL1ScoresWithDenomKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        scores.handle.clone().binding(),
        slopes.handle.clone().binding(),
        meta.handle.clone().binding(),
        denom.handle.clone().binding(),
    ]);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch dense row_l1 scores+denom kernel");
    (scores, denom)
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
struct DenseRowL1ScoresKernel;

impl KernelSource for DenseRowL1ScoresKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(DENSE_SCORES_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Clone)]
struct DenseRowL1ScoresWithDenomKernel;

impl KernelSource for DenseRowL1ScoresWithDenomKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(DENSE_SCORES_WITH_DENOM_SHADER)
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

    fn reference_scores(
        query: Tensor<Backend, 4>,
        slopes: Tensor<Backend, 1>,
    ) -> Tensor<Backend, 4> {
        dense_row_l1_scores_reference(query, slopes)
    }

    fn reference_scores_with_denom(
        query: Tensor<Backend, 4>,
        slopes: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
        dense_row_l1_scores_reference_with_denom(query, slopes)
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
        for (index, (a, b)) in lhs.into_iter().zip(rhs.into_iter()).enumerate() {
            let diff = (a - b).abs();
            let tol = atol + rtol * b.abs();
            assert!(
                diff <= tol,
                "dense row_l1 score mismatch at index {index}: lhs={a}, rhs={b}, diff={diff}, tol={tol}"
            );
        }
    }

    #[test]
    fn dense_row_l1_scores_match_reference_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 6, 17, 8], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..6).map(|i| 0.01 * i as f32).collect(), [6]),
            &device,
        );

        let actual =
            try_fused_dense_row_l1_scores_wgpu::<Backend>(&query, &slopes).expect("wgpu scores");
        let expected = reference_scores(query, slopes);
        assert_close(actual, expected, 2e-3, 2e-3);
    }

    #[test]
    fn dense_row_l1_scores_and_denom_match_reference_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 6, 17, 8], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..6).map(|i| 0.01 * i as f32).collect(), [6]),
            &device,
        );

        let (actual_scores, actual_denom) =
            try_fused_dense_row_l1_scores_and_denom_wgpu::<Backend>(&query, &slopes)
                .expect("wgpu scores+denom");
        let (expected_scores, expected_denom) = reference_scores_with_denom(query, slopes);
        assert_close(actual_scores, expected_scores, 2e-3, 2e-3);
        assert_close(actual_denom, expected_denom, 2e-3, 2e-3);
    }

    #[test]
    fn dense_row_l1_scores_match_reference_on_wgpu_autodiff_inner() {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        init_runtime(&device);
        let query =
            Tensor::<AutodiffBackendImpl, 4>::random([2, 4, 11, 8], Distribution::Default, &device);
        let slopes = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new((0..4).map(|i| 0.02 * i as f32).collect(), [4]),
            &device,
        );

        let actual = try_fused_dense_row_l1_scores_wgpu::<AutodiffBackendImpl>(&query, &slopes)
            .expect("wgpu autodiff scores")
            .inner();
        let expected = reference_scores(query.inner(), slopes.inner());
        assert_close(actual, expected, 2e-3, 2e-3);
    }

    #[test]
    fn dense_row_l1_scores_match_reference_query_gradients_on_wgpu_autodiff() {
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
        let slopes =
            Tensor::<AutodiffBackendImpl, 1>::from_data(TensorData::new(vec![0.1], [1]), &device);
        let weights = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(
                vec![0.1, -0.2, 0.3, -0.1, 0.2, -0.3, 0.4, -0.2, 0.1],
                [1, 1, 3, 3],
            ),
            &device,
        );

        let fused = try_fused_dense_row_l1_scores_wgpu::<AutodiffBackendImpl>(&query, &slopes)
            .expect("fused autodiff scores");
        let reference = dense_row_l1_scores_reference(query.clone(), slopes.clone());

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

        assert_eq!(fused_query_grad.len(), reference_query_grad.len());
        for (index, (lhs, rhs)) in fused_query_grad
            .into_iter()
            .zip(reference_query_grad.into_iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() <= 2e-3,
                "query grad mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn dense_row_l1_scores_match_reference_with_compiled_plan_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 6, 17, 8], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..6).map(|i| 0.01 * i as f32).collect(), [6]),
            &device,
        );
        let plan = CompiledDenseScoresPlan::new(2, 6, 17, 8, &device);

        let actual =
            try_fused_dense_row_l1_scores_wgpu_with_plan::<Backend>(&query, &slopes, &plan)
                .expect("wgpu scores with plan");
        let expected = reference_scores(query, slopes);
        assert_close(actual, expected, 2e-3, 2e-3);
    }

    #[test]
    fn dense_row_l1_scores_and_denom_match_reference_with_compiled_plan_on_wgpu() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        let query = Tensor::<Backend, 4>::random([2, 6, 17, 8], Distribution::Default, &device);
        let slopes = Tensor::<Backend, 1>::from_data(
            TensorData::new((0..6).map(|i| 0.01 * i as f32).collect(), [6]),
            &device,
        );
        let plan = CompiledDenseScoresPlan::new(2, 6, 17, 8, &device);

        let (actual_scores, actual_denom) =
            try_fused_dense_row_l1_scores_and_denom_wgpu_with_plan::<Backend>(
                &query, &slopes, &plan,
            )
            .expect("wgpu scores+denom with plan");
        let (expected_scores, expected_denom) = reference_scores_with_denom(query, slopes);
        assert_close(actual_scores, expected_scores, 2e-3, 2e-3);
        assert_close(actual_denom, expected_denom, 2e-3, 2e-3);
    }
}
