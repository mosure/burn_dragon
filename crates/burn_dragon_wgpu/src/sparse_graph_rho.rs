use std::any::{Any, TypeId};
use std::time::Instant;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Int, Shape, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_cubecl::cubecl::{prelude::*, server::Bindings};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::FusionTensor;
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;
use crate::profiling::{
    KernelProfileSite, KernelProfileSnapshot, profile_enabled, profile_record, profile_reset,
    profile_snapshot,
};

const WORKGROUP_SIZE_X: u32 = 8;
const WORKGROUP_SIZE_Y: u32 = 8;
const META_LEN: usize = 5;
const SPARSE_GRAPH_RHO_SHADER: &str = include_str!("sparse_graph_rho.wgsl");
type WgpuCubeAutodiffBackend = Autodiff<CubeBackend<WgpuRuntime, f32, i32, u32>>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuCubeAutodiffIntTensor = <WgpuCubeAutodiffBackend as BackendTrait>::IntTensorPrimitive;
static SPARSE_GRAPH_RHO_PROFILE: KernelProfileSite = KernelProfileSite::new();

pub type SparseGraphRhoProfileSnapshot = KernelProfileSnapshot;

#[derive(Debug)]
pub struct SparseGraphRhoAttentionOutput<B: BackendTrait> {
    pub context: BurnTensor<B, 3>,
    pub rho: BurnTensor<B, 4>,
}

pub fn sparse_graph_rho_profile_reset() {
    profile_reset(&SPARSE_GRAPH_RHO_PROFILE);
}

pub fn sparse_graph_rho_profile_snapshot() -> SparseGraphRhoProfileSnapshot {
    profile_snapshot(&SPARSE_GRAPH_RHO_PROFILE)
}

#[derive(Clone, Debug)]
pub struct SparseGraphCsr<B: BackendTrait> {
    source_offsets: BurnTensor<B, 1, Int>,
    source_indices: BurnTensor<B, 1, Int>,
    incoming_offsets: BurnTensor<B, 1, Int>,
    incoming_indices: BurnTensor<B, 1, Int>,
}

impl<B: BackendTrait> SparseGraphCsr<B> {
    pub fn from_usize_slices(
        source_offsets: &[usize],
        source_indices: &[usize],
        incoming_offsets: &[usize],
        incoming_indices: &[usize],
        device: &B::Device,
    ) -> Self {
        Self {
            source_offsets: BurnTensor::<B, 1, Int>::from_data(
                TensorData::new(
                    source_offsets
                        .iter()
                        .map(|&value| value as i32)
                        .collect::<Vec<_>>(),
                    [source_offsets.len()],
                ),
                device,
            ),
            source_indices: BurnTensor::<B, 1, Int>::from_data(
                TensorData::new(
                    source_indices
                        .iter()
                        .map(|&value| value as i32)
                        .collect::<Vec<_>>(),
                    [source_indices.len()],
                ),
                device,
            ),
            incoming_offsets: BurnTensor::<B, 1, Int>::from_data(
                TensorData::new(
                    incoming_offsets
                        .iter()
                        .map(|&value| value as i32)
                        .collect::<Vec<_>>(),
                    [incoming_offsets.len()],
                ),
                device,
            ),
            incoming_indices: BurnTensor::<B, 1, Int>::from_data(
                TensorData::new(
                    incoming_indices
                        .iter()
                        .map(|&value| value as i32)
                        .collect::<Vec<_>>(),
                    [incoming_indices.len()],
                ),
                device,
            ),
        }
    }

    pub fn from_tensors(
        source_offsets: BurnTensor<B, 1, Int>,
        source_indices: BurnTensor<B, 1, Int>,
        incoming_offsets: BurnTensor<B, 1, Int>,
        incoming_indices: BurnTensor<B, 1, Int>,
    ) -> Result<Self, SparseGraphRhoAttentionError> {
        let csr = Self {
            source_offsets,
            source_indices,
            incoming_offsets,
            incoming_indices,
        };
        csr.validate(None)?;
        Ok(csr)
    }

    pub fn source_offsets(&self) -> &BurnTensor<B, 1, Int> {
        &self.source_offsets
    }

    pub fn source_indices(&self) -> &BurnTensor<B, 1, Int> {
        &self.source_indices
    }

    pub fn incoming_offsets(&self) -> &BurnTensor<B, 1, Int> {
        &self.incoming_offsets
    }

    pub fn incoming_indices(&self) -> &BurnTensor<B, 1, Int> {
        &self.incoming_indices
    }

    fn validate(
        &self,
        source_count_hint: Option<usize>,
    ) -> Result<usize, SparseGraphRhoAttentionError> {
        if let Some(source_count) = source_count_hint
            && self.source_offsets.shape().dims::<1>()[0] != source_count + 1
        {
            return Err(SparseGraphRhoAttentionError::InvalidRouteLayout {
                field: "source_offsets",
            });
        }
        let target_count = self.incoming_offsets.shape().dims::<1>()[0]
            .checked_sub(1)
            .ok_or(SparseGraphRhoAttentionError::InvalidRouteLayout {
                field: "incoming_offsets",
            })?;
        if self.incoming_indices.shape().dims::<1>()[0]
            != self.source_indices.shape().dims::<1>()[0]
        {
            return Err(SparseGraphRhoAttentionError::InvalidRouteLayout {
                field: "edge_count",
            });
        }
        Ok(target_count)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SparseGraphRhoAttentionError {
    UnsupportedBackend,
    InvalidQueryShape,
    InvalidValueShape,
    InvalidRouteLayout { field: &'static str },
    InvalidRhoShape,
}

pub fn supports_sparse_graph_rho_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
}

pub fn fused_sparse_graph_rho_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 3>,
    value: &BurnTensor<B, 3>,
    rho: Option<&BurnTensor<B, 4>>,
    csr: &SparseGraphCsr<B>,
    decay: f32,
) -> Result<SparseGraphRhoAttentionOutput<B>, SparseGraphRhoAttentionError>
where
    B::FloatTensorPrimitive: 'static,
{
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    if !supports_sparse_graph_rho_backend::<B>() {
        return Err(SparseGraphRhoAttentionError::UnsupportedBackend);
    }

    let setup_start = prof_enabled.then(Instant::now);
    let [batch, source_count, latent] = query.shape().dims::<3>();
    let [value_batch, value_count, embd] = value.shape().dims::<3>();
    if batch == 0 || source_count == 0 || latent == 0 || embd == 0 {
        return Err(SparseGraphRhoAttentionError::InvalidQueryShape);
    }
    if value_batch != batch || value_count != source_count {
        return Err(SparseGraphRhoAttentionError::InvalidValueShape);
    }
    let target_count = csr.validate(Some(source_count))?;

    let device = query.device();
    let expected_rho = [batch, target_count, latent, embd];
    let rho = match rho {
        Some(existing) if existing.shape().dims::<4>() == expected_rho => existing.clone(),
        Some(_) => return Err(SparseGraphRhoAttentionError::InvalidRhoShape),
        None => BurnTensor::<B, 4>::zeros(expected_rho, &device),
    };

    let meta = BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                source_count as f32,
                target_count as f32,
                latent as f32,
                embd as f32,
            ],
            [META_LEN],
        ),
        &device,
    );
    let decay = BurnTensor::<B, 1>::from_data(TensorData::new(vec![decay], [1]), &device);

    let setup_ns = setup_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    let copy_start = prof_enabled.then(Instant::now);
    let query_copy = query.clone();
    let value_copy = value.clone();
    let rho_copy = rho.add_scalar(0.0);
    let source_offsets_copy = csr.source_offsets().clone();
    let source_indices_copy = csr.source_indices().clone();
    let incoming_offsets_copy = csr.incoming_offsets().clone();
    let incoming_indices_copy = csr.incoming_indices().clone();
    let decay_copy = decay.clone();
    let meta_copy = meta.clone();
    let copy_ns = copy_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    let output = try_fusion_path_wgpu::<B, u32>(
        &query_copy,
        &value_copy,
        &rho_copy,
        &source_offsets_copy,
        &source_indices_copy,
        &incoming_offsets_copy,
        &incoming_indices_copy,
        &decay_copy,
        &meta_copy,
    )
    .or_else(|| {
        try_fusion_path_wgpu::<B, u8>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &source_offsets_copy,
            &source_indices_copy,
            &incoming_offsets_copy,
            &incoming_indices_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| {
        try_direct_path::<B>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &source_offsets_copy,
            &source_indices_copy,
            &incoming_offsets_copy,
            &incoming_indices_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| {
        try_direct_path_autodiff_wgpu_cube::<B>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &source_offsets_copy,
            &source_indices_copy,
            &incoming_offsets_copy,
            &incoming_indices_copy,
            &decay_copy,
            &meta_copy,
        )
    });

    if let Some(start) = total_start {
        profile_record(&SPARSE_GRAPH_RHO_PROFILE, |state| {
            state.calls = state.calls.saturating_add(u64::from(output.is_some()));
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            state.setup_ns = state.setup_ns.saturating_add(setup_ns);
            state.copy_ns = state.copy_ns.saturating_add(copy_ns);
            state.transient_allocations = state.transient_allocations.saturating_add(9);
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_add(4);
            state.metadata_reuse_bytes = state.metadata_reuse_bytes.saturating_add(
                ((csr.source_offsets().shape().dims::<1>()[0]
                    + csr.source_indices().shape().dims::<1>()[0]
                    + csr.incoming_offsets().shape().dims::<1>()[0]
                    + csr.incoming_indices().shape().dims::<1>()[0])
                    * core::mem::size_of::<i32>()) as u64,
            );
            state.metadata_upload_bytes = state
                .metadata_upload_bytes
                .saturating_add((META_LEN * core::mem::size_of::<f32>()) as u64);
        });
    }

    output.ok_or(SparseGraphRhoAttentionError::UnsupportedBackend)
}

#[allow(clippy::too_many_arguments)]
pub fn try_fused_sparse_graph_rho_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 3>,
    value: &BurnTensor<B, 3>,
    rho: Option<&BurnTensor<B, 4>>,
    source_offsets: &BurnTensor<B, 1, Int>,
    source_indices: &BurnTensor<B, 1, Int>,
    incoming_offsets: &BurnTensor<B, 1, Int>,
    incoming_indices: &BurnTensor<B, 1, Int>,
    decay: f32,
) -> Option<SparseGraphRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let csr = SparseGraphCsr::from_tensors(
        source_offsets.clone(),
        source_indices.clone(),
        incoming_offsets.clone(),
        incoming_indices.clone(),
    )
    .ok()?;
    fused_sparse_graph_rho_attention_wgpu(query, value, rho, &csr, decay).ok()
}

#[allow(clippy::too_many_arguments)]
fn try_fusion_path_wgpu<B, BT>(
    query: &BurnTensor<B, 3>,
    value: &BurnTensor<B, 3>,
    rho: &BurnTensor<B, 4>,
    source_offsets: &BurnTensor<B, 1, Int>,
    source_indices: &BurnTensor<B, 1, Int>,
    incoming_offsets: &BurnTensor<B, 1, Int>,
    incoming_indices: &BurnTensor<B, 1, Int>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<SparseGraphRhoAttentionOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
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

    let value = resolve_fusion_tensor_wgpu::<B, BT, 3>(value)?;
    let rho = resolve_fusion_tensor_wgpu::<B, BT, 4>(rho)?;
    let source_offsets = resolve_fusion_int_tensor_wgpu::<B, BT, 1>(source_offsets)?;
    let source_indices = resolve_fusion_int_tensor_wgpu::<B, BT, 1>(source_indices)?;
    let incoming_offsets = resolve_fusion_int_tensor_wgpu::<B, BT, 1>(incoming_offsets)?;
    let incoming_indices = resolve_fusion_int_tensor_wgpu::<B, BT, 1>(incoming_indices)?;
    let decay = resolve_fusion_tensor_wgpu::<B, BT, 1>(decay)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;

    let (context, rho) = sparse_graph_rho_attention_wgsl_runtime::<WgpuRuntime>(
        query,
        value,
        rho,
        source_offsets,
        source_indices,
        incoming_offsets,
        incoming_indices,
        decay,
        meta,
    );

    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let rho_fusion = register_fusion_float_tensor(&fusion_client, rho);

    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    let rho_prim = try_cast_backend::<B, _>(rho_fusion)?;

    Some(SparseGraphRhoAttentionOutput {
        context: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

#[allow(clippy::too_many_arguments)]
fn try_direct_path<B: BackendTrait>(
    query: &BurnTensor<B, 3>,
    value: &BurnTensor<B, 3>,
    rho: &BurnTensor<B, 4>,
    source_offsets: &BurnTensor<B, 1, Int>,
    source_indices: &BurnTensor<B, 1, Int>,
    incoming_offsets: &BurnTensor<B, 1, Int>,
    incoming_indices: &BurnTensor<B, 1, Int>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<SparseGraphRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
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
    let prim_rho = rho.clone().into_primitive().tensor();
    let rho: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_rho)?;
    if rho.dtype != DType::F32 {
        return None;
    }
    let prim_source_offsets = source_offsets.clone().into_primitive();
    let source_offsets: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(prim_source_offsets)?;
    if source_offsets.dtype != DType::I32 {
        return None;
    }
    let prim_source_indices = source_indices.clone().into_primitive();
    let source_indices: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(prim_source_indices)?;
    if source_indices.dtype != DType::I32 {
        return None;
    }
    let prim_incoming_offsets = incoming_offsets.clone().into_primitive();
    let incoming_offsets: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(prim_incoming_offsets)?;
    if incoming_offsets.dtype != DType::I32 {
        return None;
    }
    let prim_incoming_indices = incoming_indices.clone().into_primitive();
    let incoming_indices: CubeTensor<WgpuRuntime> =
        try_cast_int_primitive::<B, _>(prim_incoming_indices)?;
    if incoming_indices.dtype != DType::I32 {
        return None;
    }
    let prim_decay = decay.clone().into_primitive().tensor();
    let decay: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_decay)?;
    if decay.dtype != DType::F32 {
        return None;
    }
    let prim_meta = meta.clone().into_primitive().tensor();
    let meta: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let (context, rho) = sparse_graph_rho_attention_wgsl_runtime::<WgpuRuntime>(
        query,
        value,
        rho,
        source_offsets,
        source_indices,
        incoming_offsets,
        incoming_indices,
        decay,
        meta,
    );

    let context_prim = try_cast_backend::<B, _>(context)?;
    let rho_prim = try_cast_backend::<B, _>(rho)?;

    Some(SparseGraphRhoAttentionOutput {
        context: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

#[allow(clippy::too_many_arguments)]
fn try_direct_path_autodiff_wgpu_cube<B: BackendTrait>(
    query: &BurnTensor<B, 3>,
    value: &BurnTensor<B, 3>,
    rho: &BurnTensor<B, 4>,
    source_offsets: &BurnTensor<B, 1, Int>,
    source_indices: &BurnTensor<B, 1, Int>,
    incoming_offsets: &BurnTensor<B, 1, Int>,
    incoming_indices: &BurnTensor<B, 1, Int>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<SparseGraphRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_query)?;
    let query: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad);
    if query.dtype != DType::F32 {
        return None;
    }
    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_value)?;
    let value: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad);
    if value.dtype != DType::F32 {
        return None;
    }
    let prim_rho = rho.clone().into_primitive().tensor();
    let rho_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_rho)?;
    let rho: CubeTensor<WgpuRuntime> = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(rho_ad);
    if rho.dtype != DType::F32 {
        return None;
    }
    let prim_source_offsets = source_offsets.clone().into_primitive();
    let source_offsets_ad: WgpuCubeAutodiffIntTensor =
        try_cast_int_primitive::<B, _>(prim_source_offsets)?;
    let source_offsets: CubeTensor<WgpuRuntime> = source_offsets_ad;
    if source_offsets.dtype != DType::I32 {
        return None;
    }
    let prim_source_indices = source_indices.clone().into_primitive();
    let source_indices_ad: WgpuCubeAutodiffIntTensor =
        try_cast_int_primitive::<B, _>(prim_source_indices)?;
    let source_indices: CubeTensor<WgpuRuntime> = source_indices_ad;
    if source_indices.dtype != DType::I32 {
        return None;
    }
    let prim_incoming_offsets = incoming_offsets.clone().into_primitive();
    let incoming_offsets_ad: WgpuCubeAutodiffIntTensor =
        try_cast_int_primitive::<B, _>(prim_incoming_offsets)?;
    let incoming_offsets: CubeTensor<WgpuRuntime> = incoming_offsets_ad;
    if incoming_offsets.dtype != DType::I32 {
        return None;
    }
    let prim_incoming_indices = incoming_indices.clone().into_primitive();
    let incoming_indices_ad: WgpuCubeAutodiffIntTensor =
        try_cast_int_primitive::<B, _>(prim_incoming_indices)?;
    let incoming_indices: CubeTensor<WgpuRuntime> = incoming_indices_ad;
    if incoming_indices.dtype != DType::I32 {
        return None;
    }
    let prim_decay = decay.clone().into_primitive().tensor();
    let decay_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_decay)?;
    let decay: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad);
    if decay.dtype != DType::F32 {
        return None;
    }
    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_meta)?;
    let meta: CubeTensor<WgpuRuntime> =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);
    if meta.dtype != DType::F32 {
        return None;
    }

    let (context, rho) = sparse_graph_rho_attention_wgsl_runtime::<WgpuRuntime>(
        query,
        value,
        rho,
        source_offsets,
        source_indices,
        incoming_offsets,
        incoming_indices,
        decay,
        meta,
    );

    let context_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(context);
    let rho_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho);
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    let rho_prim = try_cast_backend::<B, _>(rho_ad)?;

    Some(SparseGraphRhoAttentionOutput {
        context: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

#[allow(clippy::too_many_arguments)]
fn sparse_graph_rho_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    source_offsets: CubeTensor<R>,
    source_indices: CubeTensor<R>,
    incoming_offsets: CubeTensor<R>,
    incoming_indices: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho = into_contiguous(rho);
    let source_offsets = into_contiguous(source_offsets);
    let source_indices = into_contiguous(source_indices);
    let incoming_offsets = into_contiguous(incoming_offsets);
    let incoming_indices = into_contiguous(incoming_indices);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, source_count, _latent] = query.meta.shape.dims::<3>();
    let [_, target_count, latent, embd] = rho.meta.shape.dims::<4>();
    let max_items = source_count.max(target_count);

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, source_count, embd]),
    );
    let rho_next = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, target_count, latent, embd]),
    );

    let workgroups_x = div_ceil_u32(embd as u32, WORKGROUP_SIZE_X);
    let workgroups_y = div_ceil_u32(max_items as u32, WORKGROUP_SIZE_Y);
    let count = CubeCount::Static(workgroups_x, workgroups_y, batch as u32);

    let kernel = SourceKernel::new(
        SparseGraphRhoAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        rho.handle.clone().binding(),
        source_offsets.handle.clone().binding(),
        source_indices.handle.clone().binding(),
        incoming_offsets.handle.clone().binding(),
        incoming_indices.handle.clone().binding(),
        decay.handle.clone().binding(),
        context.handle.clone().binding(),
        rho_next.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    let dispatch_start = profile_enabled().then(Instant::now);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch sparse graph rho kernel");
    if let Some(start) = dispatch_start {
        profile_record(&SPARSE_GRAPH_RHO_PROFILE, |state| {
            state.launches = state.launches.saturating_add(1);
            state.dispatch_ns = state.dispatch_ns.saturating_add(start.elapsed().as_nanos());
        });
    }

    (context, rho_next)
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

fn resolve_fusion_int_tensor_wgpu<B, BT, const D: usize>(
    tensor: &BurnTensor<B, D, Int>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B: BackendTrait,
    B::IntTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_int_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_int::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::I32 {
        return None;
    }
    Some(cube)
}

struct SparseGraphRhoAttentionKernel;

impl KernelSource for SparseGraphRhoAttentionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(SPARSE_GRAPH_RHO_SHADER)
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

fn try_cast_int_primitive<B: BackendTrait, T: 'static>(value: B::IntTensorPrimitive) -> Option<T>
where
    B::IntTensorPrimitive: 'static,
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
    use burn_wgpu::{CubeBackend, RuntimeOptions, graphics};

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type GraphRouteTensors = (
        Tensor<Backend, 1, Int>,
        Tensor<Backend, 1, Int>,
        Tensor<Backend, 1, Int>,
        Tensor<Backend, 1, Int>,
    );

    fn init_runtime(device: &<Backend as BackendTrait>::Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn assert_close<const D: usize>(
        lhs: Tensor<Backend, D>,
        rhs: Tensor<Backend, D>,
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

    #[allow(clippy::too_many_arguments)]
    fn reference_sparse_graph_rho(
        query: Tensor<Backend, 3>,
        value: Tensor<Backend, 3>,
        rho: Tensor<Backend, 4>,
        source_offsets: &[usize],
        source_indices: &[usize],
        incoming_offsets: &[usize],
        incoming_indices: &[usize],
        decay: f32,
    ) -> (Tensor<Backend, 3>, Tensor<Backend, 4>) {
        let [batch, source_count, latent] = query.shape().dims::<3>();
        let [_, target_count, _, embd] = rho.shape().dims::<4>();
        let mut context = Vec::with_capacity(source_count);
        let mut next = rho.clone().mul_scalar(decay);

        for source in 0..source_count {
            let q_s = query.clone().slice_dim(1, source..source + 1);
            let mut acc = Tensor::<Backend, 3>::zeros([batch, 1, embd], &query.device());
            for &target in source_indices
                .iter()
                .take(source_offsets[source + 1])
                .skip(source_offsets[source])
            {
                if target >= target_count {
                    continue;
                }
                let rho_t = rho.clone().slice_dim(1, target..target + 1);
                let msg = rho_t
                    .mul(q_s.clone().unsqueeze_dim::<4>(3))
                    .sum_dims_squeeze::<3, usize>(&[2]);
                acc = acc + msg;
            }
            context.push(acc);
        }

        for target in 0..target_count {
            let mut update = Tensor::<Backend, 4>::zeros([batch, 1, latent, embd], &query.device());
            for &source in incoming_indices
                .iter()
                .take(incoming_offsets[target + 1])
                .skip(incoming_offsets[target])
            {
                if source >= source_count {
                    continue;
                }
                let outer = query
                    .clone()
                    .slice_dim(1, source..source + 1)
                    .unsqueeze_dim::<4>(3)
                    .mul(
                        value
                            .clone()
                            .slice_dim(1, source..source + 1)
                            .unsqueeze_dim::<4>(2),
                    );
                update = update + outer;
            }
            let current = next.clone().slice_dim(1, target..target + 1).add(update);
            next = next.slice_assign([0..batch, target..target + 1, 0..latent, 0..embd], current);
        }

        (Tensor::cat(context, 1), next)
    }

    fn graph_tensors(
        device: &<Backend as BackendTrait>::Device,
        source_offsets: &[usize],
        source_indices: &[usize],
        incoming_offsets: &[usize],
        incoming_indices: &[usize],
    ) -> GraphRouteTensors {
        let source_offsets = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(
                source_offsets.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                [source_offsets.len()],
            ),
            device,
        );
        let source_indices = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(
                source_indices.iter().map(|&x| x as i32).collect::<Vec<_>>(),
                [source_indices.len()],
            ),
            device,
        );
        let incoming_offsets = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(
                incoming_offsets
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                [incoming_offsets.len()],
            ),
            device,
        );
        let incoming_indices = Tensor::<Backend, 1, Int>::from_data(
            TensorData::new(
                incoming_indices
                    .iter()
                    .map(|&x| x as i32)
                    .collect::<Vec<_>>(),
                [incoming_indices.len()],
            ),
            device,
        );
        (
            source_offsets,
            source_indices,
            incoming_offsets,
            incoming_indices,
        )
    }

    #[test]
    fn fused_sparse_graph_rho_matches_reference_square_route() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 31);

        let query =
            Tensor::<Backend, 3>::random([2, 4, 5], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 3>::random([2, 4, 7], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 4>::random([2, 4, 5, 7], Distribution::Normal(0.0, 1.0), &device);
        let source_offsets = [0, 2, 4, 5, 6];
        let source_indices = [0, 1, 0, 2, 3, 1];
        let incoming_offsets = [0, 2, 4, 5, 6];
        let incoming_indices = [0, 1, 0, 3, 1, 2];
        let (source_offsets_t, source_indices_t, incoming_offsets_t, incoming_indices_t) =
            graph_tensors(
                &device,
                &source_offsets,
                &source_indices,
                &incoming_offsets,
                &incoming_indices,
            );

        let fused = try_fused_sparse_graph_rho_attention_wgpu::<Backend>(
            &query,
            &value,
            Some(&rho),
            &source_offsets_t,
            &source_indices_t,
            &incoming_offsets_t,
            &incoming_indices_t,
            0.9,
        )
        .expect("wgpu sparse graph rho output");
        let (reference_context, reference_rho) = reference_sparse_graph_rho(
            query,
            value,
            rho,
            &source_offsets,
            &source_indices,
            &incoming_offsets,
            &incoming_indices,
            0.9,
        );

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }

    #[test]
    fn fused_sparse_graph_rho_matches_reference_rectangular_route() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 7);

        let query =
            Tensor::<Backend, 3>::random([1, 3, 4], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 3>::random([1, 3, 6], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 4>::random([1, 2, 4, 6], Distribution::Normal(0.0, 1.0), &device);
        let source_offsets = [0, 1, 3, 4];
        let source_indices = [0, 0, 1, 1];
        let incoming_offsets = [0, 2, 4];
        let incoming_indices = [0, 1, 1, 2];
        let (source_offsets_t, source_indices_t, incoming_offsets_t, incoming_indices_t) =
            graph_tensors(
                &device,
                &source_offsets,
                &source_indices,
                &incoming_offsets,
                &incoming_indices,
            );

        let fused = try_fused_sparse_graph_rho_attention_wgpu::<Backend>(
            &query,
            &value,
            Some(&rho),
            &source_offsets_t,
            &source_indices_t,
            &incoming_offsets_t,
            &incoming_indices_t,
            0.85,
        )
        .expect("wgpu sparse graph rho output");
        let (reference_context, reference_rho) = reference_sparse_graph_rho(
            query,
            value,
            rho,
            &source_offsets,
            &source_indices,
            &incoming_offsets,
            &incoming_indices,
            0.85,
        );

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }
}
