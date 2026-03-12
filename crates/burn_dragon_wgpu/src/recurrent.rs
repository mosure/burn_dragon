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

const WORKGROUP_SIZE_X: u32 = 64;
const META_LEN: usize = 6;
const RECURRENT_ATTENTION_SHADER: &str = include_str!("recurrent.wgsl");
type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

pub type RecurrentProfileSnapshot = KernelProfileSnapshot;

static RECURRENT_PROFILE: KernelProfileSite = KernelProfileSite::new();

pub fn recurrent_profile_reset() {
    profile_reset(&RECURRENT_PROFILE);
}

pub fn recurrent_profile_snapshot() -> RecurrentProfileSnapshot {
    profile_snapshot(&RECURRENT_PROFILE)
}

#[derive(Debug)]
pub struct RecurrentAttentionOutput<B: BackendTrait> {
    pub context: BurnTensor<B, 4>,
    pub rho: BurnTensor<B, 4>,
}

#[derive(Debug, Clone)]
pub struct CompiledRecurrentAttentionPlan<B: BackendTrait> {
    meta: BurnTensor<B, 1>,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
}

impl<B: BackendTrait> CompiledRecurrentAttentionPlan<B> {
    pub fn new(
        batch: usize,
        heads: usize,
        value_heads: usize,
        time: usize,
        latent: usize,
        embd: usize,
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
                    embd as f32,
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
            embd,
        }
    }

    fn matches(&self, query: &BurnTensor<B, 4>, value: &BurnTensor<B, 4>) -> bool {
        query.shape().dims::<4>() == [self.batch, self.heads, self.time, self.latent]
            && value.shape().dims::<4>() == [self.batch, self.value_heads, self.time, self.embd]
    }

    fn meta(&self) -> BurnTensor<B, 1> {
        self.meta.clone()
    }
}

pub fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
        || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
}

pub fn try_fused_recurrent_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 4>>,
    decay: Option<&BurnTensor<B, 1>>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [value_batch, value_heads, value_time, embd] = value.shape().dims::<4>();

    if batch == 0 || heads == 0 || time == 0 || latent == 0 || embd == 0 {
        return None;
    }
    if value_batch != batch || value_time != time {
        return None;
    }
    if value_heads != 1 && value_heads != heads {
        return None;
    }

    let plan = CompiledRecurrentAttentionPlan::new(
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        &query.device(),
    );
    let output = try_fused_recurrent_attention_wgpu_with_plan(query, value, rho, decay, &plan);
    if output.is_some() {
        profile_record(&RECURRENT_PROFILE, |state| {
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

pub fn try_fused_recurrent_attention_wgpu_with_plan<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 4>>,
    decay: Option<&BurnTensor<B, 1>>,
    plan: &CompiledRecurrentAttentionPlan<B>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);

    if !supports_backend::<B>() || !plan.matches(query, value) {
        return None;
    }

    let setup_start = prof_enabled.then(Instant::now);
    let [batch, heads, _time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let expected_rho = [batch, heads, latent, embd];
    let rho = match rho {
        Some(existing) if existing.shape().dims::<4>() == expected_rho => existing.clone(),
        _ => BurnTensor::<B, 4>::zeros(expected_rho, &device),
    };
    let decay = match decay {
        Some(existing) if existing.shape().dims::<1>()[0] == heads => existing.clone(),
        _ => BurnTensor::<B, 1>::ones([heads], &device),
    };
    let meta = plan.meta();
    let setup_ns = setup_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    let copy_start = prof_enabled.then(Instant::now);
    let query_copy = query.clone();
    let value_copy = value.clone();
    let rho_copy = rho.add_scalar(0.0);
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
        try_fusion_path_wgpu::<B, u8>(&query_copy, &value_copy, &rho_copy, &decay_copy, &meta_copy)
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
        profile_record(&RECURRENT_PROFILE, |state| {
            state.calls = state.calls.saturating_add(u64::from(output.is_some()));
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            state.setup_ns = state.setup_ns.saturating_add(setup_ns);
            state.copy_ns = state.copy_ns.saturating_add(copy_ns);
            state.transient_allocations = state.transient_allocations.saturating_add(5);
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
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
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
    let rho = resolve_fusion_tensor_wgpu::<B, BT, 4>(rho)?;
    let decay = resolve_fusion_tensor_wgpu::<B, BT, 1>(decay)?;
    let meta = resolve_fusion_tensor_wgpu::<B, BT, 1>(meta)?;

    let (context, rho) =
        recurrent_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_fusion = register_fusion_float_tensor(&fusion_client, context);
    let rho_fusion = register_fusion_float_tensor(&fusion_client, rho);

    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    let rho_prim = try_cast_backend::<B, _>(rho_fusion)?;

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn try_direct_path<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
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
        recurrent_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_prim = try_cast_backend::<B, _>(context)?;
    let rho_prim = try_cast_backend::<B, _>(rho)?;

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn try_direct_path_autodiff_wgpu_cube<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
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
        recurrent_attention_wgsl_runtime::<WgpuRuntime>(query, value, rho, decay, meta);

    let context_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(context);
    let rho_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho);
    let context_prim = try_cast_backend::<B, _>(context_ad)?;
    let rho_prim = try_cast_backend::<B, _>(rho_ad)?;

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

fn recurrent_attention_wgsl_runtime<R: CubeRuntime>(
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

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );

    let workgroups_x = div_ceil_u32(embd as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(workgroups_x, heads as u32, batch as u32);

    let kernel = SourceKernel::new(
        RecurrentAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = Bindings::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        rho.handle.clone().binding(),
        decay.handle.clone().binding(),
        context.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);

    let dispatch_start = profile_enabled().then(Instant::now);
    client
        .launch(Box::new(kernel), count, bindings)
        .expect("launch recurrent attention kernel");
    if let Some(start) = dispatch_start {
        let dispatch_ns = start.elapsed().as_nanos();
        profile_record(&RECURRENT_PROFILE, |state| {
            state.launches = state.launches.saturating_add(1);
            state.dispatch_ns = state.dispatch_ns.saturating_add(dispatch_ns);
        });
    }

    (context, rho)
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
struct RecurrentAttentionKernel;

impl KernelSource for RecurrentAttentionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(RECURRENT_ATTENTION_SHADER)
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

    fn assert_close(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>, atol: f32, rtol: f32) {
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

    fn reference_recurrent(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        rho: Tensor<Backend, 4>,
        decay: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
        let [batch, heads, time, _latent] = query.shape().dims::<4>();
        let value_heads = value.shape().dims::<4>()[1];
        let embd = value.shape().dims::<4>()[3];

        let decay = decay.reshape([1, heads, 1, 1]);
        let mut state = rho;
        let mut outputs: Vec<Tensor<Backend, 4>> = Vec::with_capacity(time);

        for t in 0..time {
            let q_t = query.clone().slice_dim(2, t..t + 1);
            let h_value = if value_heads == 1 {
                value.clone().slice_dim(1, 0..1)
            } else {
                value.clone().slice_dim(1, 0..heads)
            };
            let v_t = h_value.slice_dim(2, t..t + 1);

            let q_latent = q_t.swap_dims(2, 3);
            let context = (state.clone() * q_latent.clone())
                .sum_dim(2)
                .reshape([batch, heads, 1, embd]);
            outputs.push(context);

            state = (state + q_latent * v_t) * decay.clone();
        }

        (Tensor::cat(outputs, 2), state)
    }

    #[test]
    fn fused_recurrent_matches_reference_with_decay() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 7);

        let query =
            Tensor::<Backend, 4>::random([2, 4, 6, 16], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 6, 24], Distribution::Normal(0.0, 1.0), &device);
        let rho =
            Tensor::<Backend, 4>::random([2, 4, 16, 24], Distribution::Normal(0.0, 1.0), &device);
        let decay_values = [0.95_f32, 0.9, 0.85, 0.8];
        let decay = Tensor::<Backend, 1>::from_floats(decay_values.as_slice(), &device);

        let fused =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("wgpu fused recurrent output");
        let (reference_context, reference_rho) = reference_recurrent(query, value, rho, decay);

        assert_close(fused.context, reference_context, 2e-4, 2e-4);
        assert_close(fused.rho, reference_rho, 2e-4, 2e-4);
    }

    #[test]
    fn fused_recurrent_matches_reference_without_decay() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 11);

        let query =
            Tensor::<Backend, 4>::random([1, 2, 5, 8], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([1, 2, 5, 10], Distribution::Normal(0.0, 1.0), &device);
        let rho = Tensor::<Backend, 4>::zeros([1, 2, 8, 10], &device);
        let decay = Tensor::<Backend, 1>::ones([2], &device);

        let fused =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("wgpu fused recurrent output");
        let (reference_context, reference_rho) = reference_recurrent(query, value, rho, decay);

        assert_close(fused.context, reference_context, 2e-4, 2e-4);
        assert_close(fused.rho, reference_rho, 2e-4, 2e-4);
    }

    #[test]
    fn fused_recurrent_memory_stays_bounded_across_repeated_calls() {
        let device = <Backend as BackendTrait>::Device::default();
        init_runtime(&device);
        <Backend as BackendTrait>::seed(&device, 23);

        let query =
            Tensor::<Backend, 4>::random([2, 4, 16, 8], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<Backend, 4>::random([2, 1, 16, 12], Distribution::Normal(0.0, 1.0), &device);
        let decay = Tensor::<Backend, 1>::from_floats([0.9, 0.91, 0.92, 0.93], &device);
        let mut rho = Tensor::<Backend, 4>::zeros([2, 4, 8, 12], &device);

        for _ in 0..2 {
            let output =
                try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                    .expect("fused recurrent");
            rho = output.rho;
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);

        let mut snapshots = Vec::with_capacity(24);
        for step in 0..32 {
            let output =
                try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                    .expect("fused recurrent");
            rho = output.rho;
            let _ = Backend::sync(&device);
            Backend::memory_cleanup(&device);
            let _ = Backend::sync(&device);
            if step >= 8 {
                snapshots.push(memory_snapshot(&device));
            }
        }

        assert_memory_growth_bounded(
            "wgpu_recurrent",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }
}
