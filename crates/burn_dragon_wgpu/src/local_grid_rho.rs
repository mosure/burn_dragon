use std::any::{Any, TypeId};
use std::time::Instant;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{DType, Shape, TensorData, TensorPrimitive};
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
const META_LEN: usize = 11;
const LOCAL_GRID_RHO_SHADER: &str = include_str!("local_grid_rho.wgsl");
type WgpuCubeAutodiffBackend = Autodiff<CubeBackend<WgpuRuntime, f32, i32, u32>>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
static LOCAL_GRID_RHO_PROFILE: KernelProfileSite = KernelProfileSite::new();

pub type LocalGridRhoProfileSnapshot = KernelProfileSnapshot;

/// Logical 2D token layout for a specialized local-grid rho kernel.
///
/// This kernel is intentionally reusable across 2D lattices such as image
/// patches or sudoku cells, but it remains a topology-specific fast path. More
/// general graph/topology recurrent routing should use separate state contracts
/// and dedicated kernels rather than forcing those topologies through this API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalGridShape2d {
    pub height: usize,
    pub width: usize,
}

impl LocalGridShape2d {
    pub const fn new(height: usize, width: usize) -> Self {
        Self { height, width }
    }

    pub fn token_count(self) -> usize {
        self.height * self.width
    }
}

/// Sparse local message-passing pattern over a 2D token lattice.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LocalGridNeighborhood {
    pub radius: usize,
    pub diagonals: bool,
    pub self_edges: bool,
}

impl LocalGridNeighborhood {
    pub const fn von_neumann(radius: usize) -> Self {
        Self {
            radius,
            diagonals: false,
            self_edges: true,
        }
    }

    pub const fn moore(radius: usize) -> Self {
        Self {
            radius,
            diagonals: true,
            self_edges: true,
        }
    }

    pub const fn with_self_edges(self, self_edges: bool) -> Self {
        Self {
            radius: self.radius,
            diagonals: self.diagonals,
            self_edges,
        }
    }
}

#[derive(Debug)]
pub struct LocalGridRhoAttentionOutput<B: BackendTrait> {
    pub context: BurnTensor<B, 4>,
    pub rho: BurnTensor<B, 5>,
}

#[derive(Debug, Clone)]
pub struct CompiledLocalGridRhoPlan<B: BackendTrait> {
    meta: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    value_heads: usize,
    patch_tokens: usize,
    latent: usize,
    embd: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct LocalGridRhoPlanSpec {
    pub batch: usize,
    pub heads: usize,
    pub value_heads: usize,
    pub patch_tokens: usize,
    pub latent: usize,
    pub embd: usize,
    pub grid: LocalGridShape2d,
    pub neighborhood: LocalGridNeighborhood,
}

impl<B: BackendTrait> CompiledLocalGridRhoPlan<B> {
    pub fn new(spec: LocalGridRhoPlanSpec, device: &B::Device) -> Self {
        let meta = BurnTensor::<B, 1>::from_data(
            TensorData::new(
                vec![
                    spec.batch as f32,
                    spec.heads as f32,
                    spec.value_heads as f32,
                    spec.patch_tokens as f32,
                    spec.latent as f32,
                    spec.embd as f32,
                    spec.grid.height as f32,
                    spec.grid.width as f32,
                    spec.neighborhood.radius as f32,
                    if spec.neighborhood.diagonals { 1.0 } else { 0.0 },
                    if spec.neighborhood.self_edges { 1.0 } else { 0.0 },
                ],
                [META_LEN],
            ),
            device,
        );
        Self {
            meta,
            batch: spec.batch,
            heads: spec.heads,
            value_heads: spec.value_heads,
            patch_tokens: spec.patch_tokens,
            latent: spec.latent,
            embd: spec.embd,
        }
    }

    fn matches(&self, query: &BurnTensor<B, 4>, value: &BurnTensor<B, 4>) -> bool {
        query.shape().dims::<4>() == [self.batch, self.heads, self.patch_tokens, self.latent]
            && value.shape().dims::<4>()
                == [self.batch, self.value_heads, self.patch_tokens, self.embd]
    }

    fn meta(&self) -> BurnTensor<B, 1> {
        self.meta.clone()
    }
}

pub fn local_grid_rho_profile_reset() {
    profile_reset(&LOCAL_GRID_RHO_PROFILE);
}

pub fn local_grid_rho_profile_snapshot() -> LocalGridRhoProfileSnapshot {
    profile_snapshot(&LOCAL_GRID_RHO_PROFILE)
}

pub fn supports_local_grid_rho_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
}

pub fn try_fused_local_grid_rho_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 5>>,
    grid: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
    decay: f32,
) -> Option<LocalGridRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let heads = query.shape().dims::<4>()[1];
    let device = query.device();
    let decay = BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![decay; heads.max(1)], [heads.max(1)]),
        &device,
    );
    try_fused_local_grid_rho_attention_wgpu_head_decay(
        query,
        value,
        rho,
        grid,
        neighborhood,
        &decay,
    )
}

pub fn try_fused_local_grid_rho_attention_wgpu_head_decay<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 5>>,
    grid: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
    decay: &BurnTensor<B, 1>,
) -> Option<LocalGridRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, patch_tokens, latent] = query.shape().dims::<4>();
    let [value_batch, value_heads, value_time, embd] = value.shape().dims::<4>();
    if batch == 0 || heads == 0 || patch_tokens == 0 || latent == 0 || embd == 0 {
        return None;
    }
    if grid.height == 0 || grid.width == 0 {
        return None;
    }
    if value_batch != batch || value_time != patch_tokens {
        return None;
    }
    if value_heads != 1 && value_heads != heads {
        return None;
    }
    if grid.token_count() != patch_tokens {
        return None;
    }
    let plan = CompiledLocalGridRhoPlan::new(
        LocalGridRhoPlanSpec {
            batch,
            heads,
            value_heads,
            patch_tokens,
            latent,
            embd,
            grid,
            neighborhood,
        },
        &query.device(),
    );
    let output =
        try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan(query, value, rho, decay, &plan);
    if output.is_some() {
        profile_record(&LOCAL_GRID_RHO_PROFILE, |state| {
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_sub(1);
            state.metadata_reuse_bytes = state
                .metadata_reuse_bytes
                .saturating_sub((META_LEN * core::mem::size_of::<f32>()) as u64);
            state.metadata_upload_bytes = state.metadata_upload_bytes.saturating_add(
                ((META_LEN + heads.max(1)) * core::mem::size_of::<f32>()) as u64,
            );
        });
    }
    output
}

pub fn try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 5>>,
    decay: &BurnTensor<B, 1>,
    plan: &CompiledLocalGridRhoPlan<B>,
) -> Option<LocalGridRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    if !supports_local_grid_rho_backend::<B>() || !plan.matches(query, value) {
        return None;
    }

    let setup_start = prof_enabled.then(Instant::now);
    let [batch, heads, patch_tokens, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let expected_rho = [batch, heads, patch_tokens, latent, embd];
    let rho = match rho {
        Some(existing) if existing.shape().dims::<5>() == expected_rho => existing.clone(),
        _ => BurnTensor::<B, 5>::zeros(expected_rho, &device),
    };
    let decay = match decay.shape().dims::<1>()[0] {
        1 => decay.clone().repeat_dim(0, heads.max(1)),
        count if count == heads => decay.clone(),
        _ => return None,
    };
    let meta = plan.meta();
    let setup_ns = setup_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    let copy_start = prof_enabled.then(Instant::now);
    let query_copy = query.clone();
    let value_copy = value.clone();
    let rho_copy = rho.clone();
    let decay_copy = decay.clone();
    let meta_copy = meta.clone();
    let copy_ns = copy_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();
    let output = try_fusion_path_wgpu::<B, u32>(
        &query_copy,
        &value_copy,
        &rho_copy,
        &decay_copy,
        &meta_copy,
    )
    .or_else(|| {
        try_fusion_path_wgpu::<B, u8>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| try_direct_path::<B>(&query_copy, &value_copy, &rho_copy, &decay_copy, &meta_copy))
    .or_else(|| {
        try_direct_path_autodiff_wgpu_cube::<B>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &decay_copy,
            &meta_copy,
        )
    });

    if let Some(start) = total_start {
        profile_record(&LOCAL_GRID_RHO_PROFILE, |state| {
            state.calls = state.calls.saturating_add(u64::from(output.is_some()));
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            state.setup_ns = state.setup_ns.saturating_add(setup_ns);
            state.copy_ns = state.copy_ns.saturating_add(copy_ns);
            state.transient_allocations = state.transient_allocations.saturating_add(4);
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_add(1);
            state.metadata_reuse_bytes = state
                .metadata_reuse_bytes
                .saturating_add((META_LEN * core::mem::size_of::<f32>()) as u64);
        });
    }

    output
}

fn try_fusion_path_wgpu<B, BT>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 5>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<LocalGridRhoAttentionOutput<B>>
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
    let rho = resolve_fusion_tensor_wgpu::<B, BT, 5>(rho)?;
    let decay = resolve_fusion_tensor_wgpu::<B, BT, 1>(decay)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;

    let (context, rho) =
        local_grid_rho_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let rho_fusion = register_fusion_float_tensor(&fusion_client, rho);

    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    let rho_prim = try_cast_backend::<B, _>(rho_fusion)?;

    Some(LocalGridRhoAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn try_direct_path<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 5>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<LocalGridRhoAttentionOutput<B>>
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

    let prim_rho = rho.clone().into_primitive().tensor();
    let rho: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_rho)?;
    if rho.dtype != DType::F32 {
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

    let (context, rho) =
        local_grid_rho_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_prim = try_cast_backend::<B, _>(context)?;
    let rho_prim = try_cast_backend::<B, _>(rho)?;

    Some(LocalGridRhoAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn try_direct_path_autodiff_wgpu_cube<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 5>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<LocalGridRhoAttentionOutput<B>>
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

    let (context, rho) =
        local_grid_rho_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(context);
    let rho_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho);
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    let rho_prim = try_cast_backend::<B, _>(rho_ad)?;

    Some(LocalGridRhoAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn local_grid_rho_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho = into_contiguous(rho);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, patch_tokens, latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, patch_tokens, embd]),
    );
    let rho_next = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, patch_tokens, latent, embd]),
    );

    let workgroups_x = div_ceil_u32(embd as u32, WORKGROUP_SIZE_X);
    let workgroups_y = div_ceil_u32(patch_tokens as u32, WORKGROUP_SIZE_Y);
    let count = CubeCount::Static(workgroups_x, workgroups_y, (batch * heads) as u32);

    let kernel = SourceKernel::new(
        LocalGridRhoAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        rho.handle.clone().binding(),
        decay.handle.clone().binding(),
        context.handle.clone().binding(),
        rho_next.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    let dispatch_start = profile_enabled().then(Instant::now);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch local grid rho kernel");
    if let Some(start) = dispatch_start {
        profile_record(&LOCAL_GRID_RHO_PROFILE, |state| {
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

struct LocalGridRhoAttentionKernel;

impl KernelSource for LocalGridRhoAttentionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOCAL_GRID_RHO_SHADER)
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
    use burn_cubecl::cubecl::Runtime;
    use burn_wgpu::{CubeBackend, RuntimeOptions, graphics};

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;

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

    #[derive(Clone, Copy)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
    }

    fn memory_snapshot(device: &<Backend as BackendTrait>::Device) -> MemorySnapshot {
        let usage = <WgpuRuntime as Runtime>::client(device).memory_usage();
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }

    fn assert_memory_growth_bounded(
        label: &str,
        snapshots: &[MemorySnapshot],
        max_reserved_growth: u64,
        max_in_use_growth: u64,
    ) {
        assert!(!snapshots.is_empty(), "{label}: no memory snapshots");
        let first = snapshots[0];
        let last = snapshots[snapshots.len() - 1];
        let reserved_growth = last.reserved.saturating_sub(first.reserved);
        let in_use_growth = last.in_use.saturating_sub(first.in_use);
        assert!(
            reserved_growth <= max_reserved_growth,
            "{label}: reserved growth {} exceeded {}",
            reserved_growth,
            max_reserved_growth
        );
        assert!(
            in_use_growth <= max_in_use_growth,
            "{label}: in_use growth {} exceeded {}",
            in_use_growth,
            max_in_use_growth
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_local_grid_rho(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        rho: Tensor<Backend, 5>,
        grid_height: usize,
        grid_width: usize,
        local_radius: usize,
        local_diagonals: bool,
        local_self: bool,
        decay: f32,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 5>) {
        let heads = query.shape().dims::<4>()[1];
        let decay = Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![decay; heads.max(1)], [heads.max(1)]),
            &query.device(),
        );
        reference_local_grid_rho_head_decay(
            query,
            value,
            rho,
            grid_height,
            grid_width,
            local_radius,
            local_diagonals,
            local_self,
            decay,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn reference_local_grid_rho_head_decay(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        rho: Tensor<Backend, 5>,
        grid_height: usize,
        grid_width: usize,
        local_radius: usize,
        local_diagonals: bool,
        local_self: bool,
        decay: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 5>) {
        let [batch, heads, patch_tokens, latent] = query.shape().dims::<4>();
        let value_heads = value.shape().dims::<4>()[1];
        let embd = value.shape().dims::<4>()[3];
        let decay = decay.reshape([1, heads, 1, 1, 1]);

        let mut state = rho;
        let mut outputs: Vec<Tensor<Backend, 4>> = Vec::with_capacity(patch_tokens);

        for target in 0..patch_tokens {
            let ty = target / grid_width.max(1);
            let tx = target % grid_width.max(1);
            let q_t = query.clone().slice_dim(2, target..target + 1);
            let mut context = Tensor::<Backend, 4>::zeros([batch, heads, 1, embd], &query.device());
            for dy in -(local_radius as isize)..=(local_radius as isize) {
                for dx in -(local_radius as isize)..=(local_radius as isize) {
                    if dy == 0 && dx == 0 && !local_self {
                        continue;
                    }
                    if !local_diagonals && dy != 0 && dx != 0 {
                        continue;
                    }
                    let sy = ty as isize + dy;
                    let sx = tx as isize + dx;
                    if sy < 0 || sy >= grid_height as isize || sx < 0 || sx >= grid_width as isize {
                        continue;
                    }
                    let source = sy as usize * grid_width + sx as usize;
                    let source_state = state.clone().slice_dim(2, source..source + 1);
                    let msg = source_state
                        .mul(q_t.clone().unsqueeze_dim::<5>(4))
                        .sum_dims_squeeze::<4, usize>(&[3]);
                    context = context + msg;
                }
            }
            outputs.push(context);
        }

        for target in 0..patch_tokens {
            let q_t = query
                .clone()
                .slice_dim(2, target..target + 1)
                .unsqueeze_dim::<5>(4);
            let v_t = if value_heads == 1 {
                value.clone().slice_dim(1, 0..1)
            } else {
                value.clone().slice_dim(1, 0..heads)
            }
            .slice_dim(2, target..target + 1)
            .unsqueeze_dim::<5>(3);
            let next = state
                .clone()
                .slice_dim(2, target..target + 1)
                .mul(decay.clone())
                .add(q_t.mul(v_t));
            state = state.slice_assign(
                [0..batch, 0..heads, target..target + 1, 0..latent, 0..embd],
                next,
            );
        }

        (Tensor::cat(outputs, 2), state)
    }

    #[test]
    fn fused_local_grid_rho_matches_reference() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 19);

        let query =
            Tensor::<Backend, 4>::random([2, 3, 4, 5], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 4, 7], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 5>::random([2, 3, 4, 5, 7], Distribution::Normal(0.0, 1.0), &device);

        let fused = try_fused_local_grid_rho_attention_wgpu::<Backend>(
            &query,
            &value,
            Some(&rho),
            LocalGridShape2d::new(2, 2),
            LocalGridNeighborhood::von_neumann(1),
            0.9,
        )
        .expect("wgpu fused local grid rho output");
        let (reference_context, reference_rho) =
            reference_local_grid_rho(query, value, rho, 2, 2, 1, false, true, 0.9);

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }

    #[test]
    fn fused_local_grid_rho_matches_reference_with_head_decay_vector() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 123);

        let query =
            Tensor::<Backend, 4>::random([1, 2, 4, 3], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([1, 1, 4, 5], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 5>::random([1, 2, 4, 3, 5], Distribution::Normal(0.0, 1.0), &device);
        let decay =
            Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.95, 0.55], [2]), &device);

        let fused = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
            &query,
            &value,
            Some(&rho),
            LocalGridShape2d::new(2, 2),
            LocalGridNeighborhood::von_neumann(1),
            &decay,
        )
        .expect("wgpu fused local grid rho output");
        let (reference_context, reference_rho) =
            reference_local_grid_rho_head_decay(query, value, rho, 2, 2, 1, false, true, decay);

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }

    #[test]
    fn fused_local_grid_rho_supports_sudoku_shaped_grid() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 81);

        let query =
            Tensor::<Backend, 4>::random([1, 2, 81, 4], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([1, 1, 81, 6], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 5>::random([1, 2, 81, 4, 6], Distribution::Normal(0.0, 1.0), &device);

        let fused = try_fused_local_grid_rho_attention_wgpu::<Backend>(
            &query,
            &value,
            Some(&rho),
            LocalGridShape2d::new(9, 9),
            LocalGridNeighborhood::von_neumann(1),
            0.85,
        )
        .expect("wgpu fused local grid rho output");
        let (reference_context, reference_rho) =
            reference_local_grid_rho(query, value, rho, 9, 9, 1, false, true, 0.85);

        assert_close(fused.context, reference_context, 4e-4, 4e-4);
        assert_close(fused.rho, reference_rho, 4e-4, 4e-4);
    }

    #[test]
    fn fused_local_grid_rho_supports_self_only_updates() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 7);

        let query =
            Tensor::<Backend, 4>::random([1, 1, 6, 3], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([1, 1, 6, 5], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 5>::random([1, 1, 6, 3, 5], Distribution::Normal(0.0, 1.0), &device);

        let fused = try_fused_local_grid_rho_attention_wgpu::<Backend>(
            &query,
            &value,
            Some(&rho),
            LocalGridShape2d::new(2, 3),
            LocalGridNeighborhood::von_neumann(0),
            0.7,
        )
        .expect("wgpu fused local grid rho output");
        let (reference_context, reference_rho) =
            reference_local_grid_rho(query, value, rho, 2, 3, 0, false, true, 0.7);

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }

    #[test]
    fn fused_local_grid_rho_matches_reference_on_single_token_grid() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 6_060);

        let query =
            Tensor::<Backend, 4>::random([2, 2, 1, 1], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 1, 4], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 5>::random([2, 2, 1, 1, 4], Distribution::Normal(0.0, 1.0), &device);
        let decay = Tensor::<Backend, 1>::from_floats([0.85, 0.95], &device);

        let fused = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
            &query,
            &value,
            Some(&rho),
            LocalGridShape2d::new(1, 1),
            LocalGridNeighborhood::moore(1),
            &decay,
        )
        .expect("wgpu fused local grid rho output");
        let (reference_context, reference_rho) =
            reference_local_grid_rho_head_decay(query, value, rho, 1, 1, 1, true, true, decay);

        assert_close(fused.context, reference_context, 3e-4, 3e-4);
        assert_close(fused.rho, reference_rho, 3e-4, 3e-4);
    }

    #[test]
    fn fused_local_grid_rho_memory_stays_bounded_across_repeated_calls() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 41);

        let query =
            Tensor::<Backend, 4>::random([2, 2, 16, 4], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 16, 6], Distribution::Normal(0.0, 1.0), &device);
        let decay = Tensor::<Backend, 1>::from_floats([0.9, 0.85], &device);
        let mut rho = Tensor::<Backend, 5>::zeros([2, 2, 16, 4, 6], &device);

        for _ in 0..2 {
            let output = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
                &query,
                &value,
                Some(&rho),
                LocalGridShape2d::new(4, 4),
                LocalGridNeighborhood::moore(1),
                &decay,
            )
            .expect("fused local-grid");
            rho = output.rho;
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);

        let mut snapshots = Vec::with_capacity(24);
        for step in 0..32 {
            let output = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
                &query,
                &value,
                Some(&rho),
                LocalGridShape2d::new(4, 4),
                LocalGridNeighborhood::moore(1),
                &decay,
            )
            .expect("fused local-grid");
            rho = output.rho;
            let _ = Backend::sync(&device);
            Backend::memory_cleanup(&device);
            let _ = Backend::sync(&device);
            if step >= 8 {
                snapshots.push(memory_snapshot(&device));
            }
        }

        assert_memory_growth_bounded(
            "wgpu_local_grid_rho",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }
}
