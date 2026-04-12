use std::any::{Any, TypeId};
use std::marker::PhantomData;
use std::time::Instant;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{DType, Shape, TensorData, TensorPrimitive};
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
use burn_fusion::FusionTensor;
use burn_wgpu::{CubeBackend, KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;
use crate::profiling::{
    KernelProfileSite, KernelProfileSnapshot, profile_enabled, profile_record, profile_reset,
    profile_snapshot,
};

mod backward_runtime;
mod forward_runtime;

use self::backward_runtime::recurrent_attention_autodiff_custom;
use self::forward_runtime::{
    try_direct_path_autodiff_cube_runtime, try_direct_path_runtime,
    try_direct_path_runtime_with_state_history, try_fusion_path_runtime,
};

const WORKGROUP_SIZE_X: u32 = 64;
const RECURRENT_TILED_WORKGROUP_SIZE_X: u32 = 128;
const META_LEN: usize = 6;
const RECURRENT_ATTENTION_SHADER: &str = include_str!("recurrent.wgsl");
type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

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

pub(super) struct RecurrentRuntimeOutput<R: CubeRuntime> {
    pub context: CubeTensor<R>,
    pub rho: CubeTensor<R>,
    pub state_history: Option<CubeTensor<R>>,
}

pub(super) struct RecurrentAttentionCapturedOutput<B: BackendTrait> {
    pub context: BurnTensor<B, 4>,
    pub rho: BurnTensor<B, 4>,
    pub state_history: BurnTensor<B, 5>,
}

#[derive(Debug, Clone)]
pub(super) struct RecurrentAttentionBackwardState<T> {
    pub query: T,
    pub value: T,
    pub rho: T,
    pub decay: T,
    pub state_history: T,
}

#[derive(Debug)]
struct FusedRecurrentAttentionBackward<B>(PhantomData<B>);

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
    #[cfg(feature = "cuda")]
    {
        matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
            || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>(
            )
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>(
            )
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>()
            || matches_type::<B::FloatTensorPrimitive, CudaCubeAutodiffTensor>()
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>(
            )
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>(
            )
    }
    #[cfg(not(feature = "cuda"))]
    {
        matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
            || matches_type::<B::FloatTensorPrimitive, WgpuCubeAutodiffTensor>()
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>(
            )
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>(
            )
    }
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
            state.metadata_upload_bytes = state
                .metadata_upload_bytes
                .saturating_add(((META_LEN + heads.max(1)) * core::mem::size_of::<f32>()) as u64);
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

    let output = try_fusion_path_runtime::<B, u32, WgpuRuntime>(
        &query_copy,
        &value_copy,
        &rho_copy,
        &decay_copy,
        &meta_copy,
    )
    .or_else(|| {
        try_fusion_path_runtime::<B, u8, WgpuRuntime>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| {
        try_direct_path_runtime::<B, WgpuRuntime>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| {
        try_direct_path_autodiff_cube_runtime::<B, WgpuRuntime>(
            &query_copy,
            &value_copy,
            &rho_copy,
            &decay_copy,
            &meta_copy,
        )
    })
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_fusion_path_runtime::<B, u32, CudaRuntime>(
                &query_copy,
                &value_copy,
                &rho_copy,
                &decay_copy,
                &meta_copy,
            )
            .or_else(|| {
                try_fusion_path_runtime::<B, u8, CudaRuntime>(
                    &query_copy,
                    &value_copy,
                    &rho_copy,
                    &decay_copy,
                    &meta_copy,
                )
            })
            .or_else(|| {
                try_direct_path_runtime::<B, CudaRuntime>(
                    &query_copy,
                    &value_copy,
                    &rho_copy,
                    &decay_copy,
                    &meta_copy,
                )
            })
            .or_else(|| {
                try_direct_path_autodiff_cube_runtime::<B, CudaRuntime>(
                    &query_copy,
                    &value_copy,
                    &rho_copy,
                    &decay_copy,
                    &meta_copy,
                )
            })
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
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

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cube(launch)]
fn recurrent_attention_cube_exact_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    rho_state: &mut Tensor<f32>,
    decay: &Tensor<f32>,
    context: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let decay_value = decay[h * decay.stride(0)];
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut t = 0usize;
    while t < time {
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + t * value.stride(2)
            + e * value.stride(3);
        let value_t = value[value_index];

        let mut acc = f32::cast_from(0u32);
        let mut l = 0usize;
        while l < latent {
            let query_index = b * query.stride(0)
                + h * query.stride(1)
                + t * query.stride(2)
                + l * query.stride(3);
            let rho_index = b * rho_state.stride(0)
                + h * rho_state.stride(1)
                + l * rho_state.stride(2)
                + e * rho_state.stride(3);
            let q = query[query_index];
            let rho_prev = rho_state[rho_index];
            acc += rho_prev * q;
            rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
            l += 1usize;
        }

        let out_index = b * context.stride(0)
            + h * context.stride(1)
            + t * context.stride(2)
            + e * context.stride(3);
        context[out_index] = acc;
        t += 1usize;
    }
}

#[cube(launch)]
fn recurrent_attention_cube_exact_history_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    rho_state: &mut Tensor<f32>,
    decay: &Tensor<f32>,
    context: &mut Tensor<f32>,
    state_history: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let decay_value = decay[h * decay.stride(0)];
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut t = 0usize;
    while t < time {
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + t * value.stride(2)
            + e * value.stride(3);
        let value_t = value[value_index];

        let mut acc = f32::cast_from(0u32);
        let mut l = 0usize;
        while l < latent {
            let query_index = b * query.stride(0)
                + h * query.stride(1)
                + t * query.stride(2)
                + l * query.stride(3);
            let rho_index = b * rho_state.stride(0)
                + h * rho_state.stride(1)
                + l * rho_state.stride(2)
                + e * rho_state.stride(3);
            let history_index = b * state_history.stride(0)
                + h * state_history.stride(1)
                + t * state_history.stride(2)
                + l * state_history.stride(3)
                + e * state_history.stride(4);
            let q = query[query_index];
            let rho_prev = rho_state[rho_index];
            state_history[history_index] = rho_prev;
            acc += rho_prev * q;
            rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
            l += 1usize;
        }

        let out_index = b * context.stride(0)
            + h * context.stride(1)
            + t * context.stride(2)
            + e * context.stride(3);
        context[out_index] = acc;
        t += 1usize;
    }
}

#[cube(launch)]
fn recurrent_attention_cube_tiled_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    rho_state: &mut Tensor<f32>,
    decay: &Tensor<f32>,
    context: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] query_tile_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || h >= heads {
        terminate!();
    }
    let active_e = e < embd;

    let mut query_tile = SharedMemory::<f32>::new_aligned(query_tile_size, 1usize);
    let decay_value = decay[h * decay.stride(0)];
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }

    let mut t = 0usize;
    while t < time {
        let mut value_t = f32::cast_from(0u32);
        if active_e {
            let value_index = b * value.stride(0)
                + value_head * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            value_t = value[value_index];
        }

        let mut acc = f32::cast_from(0u32);
        let mut latent_base = 0usize;
        while latent_base < latent {
            let mut load_offset = lane;
            while load_offset < query_tile_size {
                if latent_base + load_offset < latent {
                    let query_index = b * query.stride(0)
                        + h * query.stride(1)
                        + t * query.stride(2)
                        + (latent_base + load_offset) * query.stride(3);
                    query_tile[load_offset] = query[query_index];
                } else {
                    query_tile[load_offset] = f32::cast_from(0u32);
                }
                load_offset += CUBE_DIM_X as usize;
            }
            sync_cube();

            let mut tile_offset = 0usize;
            while tile_offset < query_tile_size {
                let l = latent_base + tile_offset;
                if active_e && l < latent {
                    let rho_index = b * rho_state.stride(0)
                        + h * rho_state.stride(1)
                        + l * rho_state.stride(2)
                        + e * rho_state.stride(3);
                    let q = query_tile[tile_offset];
                    let rho_prev = rho_state[rho_index];
                    acc += rho_prev * q;
                    rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
                }
                tile_offset += 1usize;
            }

            sync_cube();
            latent_base += query_tile_size;
        }

        if active_e {
            let out_index = b * context.stride(0)
                + h * context.stride(1)
                + t * context.stride(2)
                + e * context.stride(3);
            context[out_index] = acc;
        }
        t += 1usize;
    }
}

#[cube(launch)]
fn recurrent_attention_cube_tiled_history_kernel(
    query: &Tensor<f32>,
    value: &Tensor<f32>,
    rho_state: &mut Tensor<f32>,
    decay: &Tensor<f32>,
    context: &mut Tensor<f32>,
    state_history: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] query_tile_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || h >= heads {
        terminate!();
    }
    let active_e = e < embd;

    let mut query_tile = SharedMemory::<f32>::new_aligned(query_tile_size, 1usize);
    let decay_value = decay[h * decay.stride(0)];
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }

    let mut t = 0usize;
    while t < time {
        let mut value_t = f32::cast_from(0u32);
        if active_e {
            let value_index = b * value.stride(0)
                + value_head * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            value_t = value[value_index];
        }

        let mut acc = f32::cast_from(0u32);
        let mut latent_base = 0usize;
        while latent_base < latent {
            let mut load_offset = lane;
            while load_offset < query_tile_size {
                if latent_base + load_offset < latent {
                    let query_index = b * query.stride(0)
                        + h * query.stride(1)
                        + t * query.stride(2)
                        + (latent_base + load_offset) * query.stride(3);
                    query_tile[load_offset] = query[query_index];
                } else {
                    query_tile[load_offset] = f32::cast_from(0u32);
                }
                load_offset += CUBE_DIM_X as usize;
            }
            sync_cube();

            let mut tile_offset = 0usize;
            while tile_offset < query_tile_size {
                let l = latent_base + tile_offset;
                if active_e && l < latent {
                    let rho_index = b * rho_state.stride(0)
                        + h * rho_state.stride(1)
                        + l * rho_state.stride(2)
                        + e * rho_state.stride(3);
                    let history_index = b * state_history.stride(0)
                        + h * state_history.stride(1)
                        + t * state_history.stride(2)
                        + l * state_history.stride(3)
                        + e * state_history.stride(4);
                    let q = query_tile[tile_offset];
                    let rho_prev = rho_state[rho_index];
                    state_history[history_index] = rho_prev;
                    acc += rho_prev * q;
                    rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
                }
                tile_offset += 1usize;
            }

            sync_cube();
            latent_base += query_tile_size;
        }

        if active_e {
            let out_index = b * context.stride(0)
                + h * context.stride(1)
                + t * context.stride(2)
                + e * context.stride(3);
            context[out_index] = acc;
        }
        t += 1usize;
    }
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

pub(super) fn try_direct_path_autodiff_cube_runtime_custom<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    recurrent_attention_autodiff_custom::<B, R>(query, value, rho, decay, meta)
}

#[cfg(test)]
mod tests;
