use std::marker::PhantomData;

use burn::tensor::Int;
use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::Backend as BackendTrait;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::{CubeBackend, WgpuRuntime};

use super::forward::{
    rwkv8_forward_recurrence_before_4d, rwkv8_forward_recurrence_before_5d, value_outer_5d,
};
use super::runtime::{
    try_rwkv8_runtime_backward_finalize, try_rwkv8_runtime_backward_prepare,
    try_rwkv8_runtime_chunk_boundaries, try_rwkv8_runtime_chunk_boundaries_from_zero,
    try_rwkv8_runtime_forward, try_rwkv8_runtime_norm_history_recurrence,
    try_rwkv8_runtime_norm_history_recurrence_from_zero,
    try_rwkv8_runtime_norm_recurrence_from_zero, try_rwkv8_runtime_state_history_recurrence,
    try_rwkv8_runtime_state_history_recurrence_from_zero,
    try_rwkv8_runtime_state_recurrence_from_zero,
};

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

const EPS: f32 = 1.0e-6;

/// The local RWKV8 tensorized path now owns a real analytical backward over the BDH-aligned
/// decayed normalized recurrence. This is still not a monolithic fused CUDA kernel, but it is no
/// longer the old nested-autodiff recompute wrapper.
pub const AVAILABLE: bool = true;

#[derive(Debug, Clone)]
pub(crate) struct TensorizedRwkv8BackwardState<FT> {
    pub(crate) query: FT,
    pub(crate) value: FT,
    pub(crate) rho_state: Option<FT>,
    pub(crate) rho_norm_state: Option<FT>,
    pub(crate) decay: FT,
    pub(crate) chunk_size: usize,
    pub(crate) chunk_rho_start: Option<FT>,
    pub(crate) chunk_rho_norm_start: Option<FT>,
    pub(crate) chunk_rho_before: Option<FT>,
    pub(crate) chunk_rho_norm_before: Option<FT>,
    pub(crate) tail_rho_start: Option<FT>,
    pub(crate) tail_rho_norm_start: Option<FT>,
    pub(crate) tail_rho_before: Option<FT>,
    pub(crate) tail_rho_norm_before: Option<FT>,
}

#[derive(Debug)]
pub(crate) struct TensorizedRwkv8Backward<B>(pub(crate) PhantomData<B>);

fn resolve_rho_state<B: BackendTrait>(
    rho_state: Option<BurnTensor<B, 4>>,
    batch: usize,
    heads: usize,
    latent: usize,
    embd: usize,
    device: &B::Device,
) -> BurnTensor<B, 4> {
    match rho_state {
        Some(existing) if existing.shape().dims::<4>() == [batch, heads, latent, embd] => existing,
        _ => BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], device),
    }
}

fn resolve_rho_norm_state<B: BackendTrait>(
    rho_norm_state: Option<BurnTensor<B, 3>>,
    batch: usize,
    heads: usize,
    latent: usize,
    device: &B::Device,
) -> BurnTensor<B, 3> {
    match rho_norm_state {
        Some(existing) if existing.shape().dims::<3>() == [batch, heads, latent] => existing,
        _ => BurnTensor::<B, 3>::zeros([batch, heads, latent], device),
    }
}

fn reverse_time_indices<B: BackendTrait>(time: usize, device: &B::Device) -> BurnTensor<B, 1, Int> {
    BurnTensor::<B, 1, Int>::arange(0..time as i64, device).flip([0])
}

fn reverse_time_tensor5<B: BackendTrait>(tensor: BurnTensor<B, 5>) -> BurnTensor<B, 5> {
    let time = tensor.shape().dims::<5>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn reverse_time_tensor4<B: BackendTrait>(tensor: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
    let time = tensor.shape().dims::<4>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn compute_chunk_traces<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: BurnTensor<B, 4>,
    rho_norm_state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
) -> (
    BurnTensor<B, 5>,
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
) {
    let delta = value_outer_5d(query.clone(), value);
    let (rho_before, rho) = rwkv8_forward_recurrence_before_5d(delta, rho_state, decay.clone());
    let (rho_norm_before, rho_norm) =
        rwkv8_forward_recurrence_before_4d(query, rho_norm_state, decay);

    (rho_before, rho_norm_before, rho, rho_norm)
}

fn backward_chunk_impl<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_before: BurnTensor<B, 5>,
    rho_norm_before: BurnTensor<B, 4>,
    decay: BurnTensor<B, 3>,
    grad_output: BurnTensor<B, 4>,
    boundary_grad_rho_after: BurnTensor<B, 4>,
    boundary_grad_rho_norm_after: BurnTensor<B, 3>,
) -> (
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [_, value_heads, _, embd] = value.shape().dims::<4>();

    let query_denom = query
        .clone()
        .sum_dim(3)
        .add_scalar(EPS)
        .reshape([batch, heads, time, 1]);
    let query_weights = query.clone().div(query_denom.clone());
    let rho_norm_denom = rho_norm_before.clone().add_scalar(EPS);
    let rho_norm_denom5 = rho_norm_denom.clone().unsqueeze_dim::<5>(4);
    let normalized_rho = rho_before.clone().div(rho_norm_denom5.clone());

    let grad_query_weights = grad_output
        .clone()
        .unsqueeze_dim::<5>(3)
        .mul(normalized_rho.clone())
        .sum_dim(4)
        .reshape([batch, heads, time, latent]);
    let grad_normalized_rho = grad_output
        .clone()
        .unsqueeze_dim::<5>(3)
        .mul(query_weights.clone().unsqueeze_dim::<5>(4));

    let grad_rho_from_context = grad_normalized_rho.clone().div(rho_norm_denom5);
    let grad_rho_norm_from_context = grad_normalized_rho
        .clone()
        .mul(rho_before.clone())
        .sum_dim(4)
        .reshape([batch, heads, time, latent])
        .div(rho_norm_denom.clone().powf_scalar(2.0))
        .mul_scalar(-1.0);

    let grad_query_from_weights = grad_query_weights
        .clone()
        .sub(
            grad_query_weights
                .clone()
                .mul(query_weights.clone())
                .sum_dim(3)
                .reshape([batch, heads, time, 1]),
        )
        .div(query_denom);

    let (grad_rho_carry_rev, prev_boundary_grad_rho) = rwkv8_forward_recurrence_before_5d(
        reverse_time_tensor5(grad_rho_from_context.clone()),
        boundary_grad_rho_after,
        decay.clone(),
    );
    let (grad_rho_norm_carry_rev, prev_boundary_grad_rho_norm) = rwkv8_forward_recurrence_before_4d(
        reverse_time_tensor4(grad_rho_norm_from_context.clone()),
        boundary_grad_rho_norm_after,
        decay.clone(),
    );
    let grad_rho_carry = reverse_time_tensor5(grad_rho_carry_rev);
    let grad_rho_norm_carry = reverse_time_tensor4(grad_rho_norm_carry_rev);

    let grad_query_from_state = grad_rho_carry
        .clone()
        .mul(match value_heads {
            1 => value.clone().reshape([batch, 1, time, 1, embd]),
            existing if existing == heads => value.clone().reshape([batch, heads, time, 1, embd]),
            existing => panic!("value heads {existing} must be 1 or {heads}"),
        })
        .sum_dim(4)
        .reshape([batch, heads, time, latent])
        .add(grad_rho_norm_carry.clone());
    let grad_value = grad_rho_carry
        .clone()
        .mul(query.clone().unsqueeze_dim::<5>(4))
        .sum_dim(3)
        .reshape([batch, heads, time, embd]);

    let grad_decay = grad_rho_carry
        .clone()
        .mul(rho_before.clone())
        .sum_dim(4)
        .reshape([batch, heads, time, latent])
        .add(grad_rho_norm_carry.clone().mul(rho_norm_before.clone()))
        .sum_dim(2)
        .sum_dim(0)
        .reshape([1, heads, latent]);

    (
        grad_query_from_weights.add(grad_query_from_state),
        grad_value,
        grad_decay,
        prev_boundary_grad_rho,
        prev_boundary_grad_rho_norm,
    )
}

fn try_tensorized_rwkv8_backward_runtime_chunked<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: Option<BurnTensor<B, 4>>,
    rho_norm_state: Option<BurnTensor<B, 3>>,
    chunk_rho_start: Option<BurnTensor<B, 5>>,
    chunk_rho_norm_start: Option<BurnTensor<B, 4>>,
    chunk_rho_before: Option<BurnTensor<B, 6>>,
    chunk_rho_norm_before: Option<BurnTensor<B, 5>>,
    tail_rho_start: Option<BurnTensor<B, 4>>,
    tail_rho_norm_start: Option<BurnTensor<B, 3>>,
    tail_rho_before: Option<BurnTensor<B, 5>>,
    tail_rho_norm_before: Option<BurnTensor<B, 4>>,
    decay: BurnTensor<B, 3>,
    grad_output: BurnTensor<B, 4>,
    chunk_size: usize,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>, BurnTensor<B, 3>)>
where
    B::FloatTensorPrimitive: 'static,
{
    #[derive(Clone)]
    struct PreparedChunk<B: BackendTrait> {
        query_chunk: BurnTensor<B, 4>,
        value_chunk: BurnTensor<B, 4>,
        rho_before: BurnTensor<B, 5>,
        rho_norm_before: BurnTensor<B, 4>,
        grad_query_weights: BurnTensor<B, 4>,
        reversed_grad_rho_from_context: BurnTensor<B, 5>,
        reversed_grad_rho_norm_from_context: BurnTensor<B, 4>,
        zero_prev_boundary_grad_rho: BurnTensor<B, 4>,
        zero_prev_boundary_grad_rho_norm: BurnTensor<B, 3>,
    }

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let chunk = chunk_size.min(time.max(1)).max(1);
    let starts_from_zero = rho_state.is_none() && rho_norm_state.is_none();

    let resolved_rho = resolve_rho_state(rho_state, batch, heads, latent, embd, &device);
    let resolved_rho_norm = resolve_rho_norm_state(rho_norm_state, batch, heads, latent, &device);
    let precomputed_full_chunks = chunk_rho_start
        .as_ref()
        .filter(|start| {
            let dims = start.shape().dims::<5>();
            dims[0] == batch && dims[2] == heads && dims[3] == latent && dims[4] == embd
        })
        .map(|start| start.shape().dims::<5>()[1]);
    let full_chunks = precomputed_full_chunks.unwrap_or(time / chunk);
    let full_time = full_chunks * chunk;
    let tail_exists = full_time < time;
    let (
        full_rho_start,
        full_rho_norm_start,
        resolved_tail_rho_start,
        resolved_tail_rho_norm_start,
    ) = if let (Some(rho_start), Some(rho_norm_start)) = (chunk_rho_start, chunk_rho_norm_start) {
        (
            rho_start,
            rho_norm_start,
            tail_rho_start,
            tail_rho_norm_start,
        )
    } else {
        let chunk_boundaries = if starts_from_zero {
            try_rwkv8_runtime_chunk_boundaries_from_zero(
                query.clone(),
                value.clone(),
                decay.clone(),
                chunk,
            )?
        } else {
            try_rwkv8_runtime_chunk_boundaries(
                query.clone(),
                value.clone(),
                resolved_rho.clone(),
                resolved_rho_norm.clone(),
                decay.clone(),
                chunk,
            )?
        };
        (
            chunk_boundaries.rho_start,
            chunk_boundaries.rho_norm_start,
            if tail_exists {
                Some(chunk_boundaries.final_rho)
            } else {
                None
            },
            if tail_exists {
                Some(chunk_boundaries.final_rho_norm)
            } else {
                None
            },
        )
    };
    let mut full_query = None;
    let mut full_value = None;
    let mut full_rho_before = None;
    let mut full_rho_norm_before = None;
    let mut full_grad_query_weights = None;
    let mut full_reversed_grad_rho_from_context = None;
    let mut full_reversed_grad_rho_norm_from_context = None;
    let mut full_zero_prev_boundary_grad_rho = None;
    let mut full_zero_prev_boundary_grad_rho_norm = None;
    let mut tail_prepared = None;

    if full_chunks > 0 {
        let query_full = query
            .clone()
            .slice_dim(2, 0..full_time)
            .reshape([batch, heads, full_chunks, chunk, latent])
            .swap_dims(1, 2);
        let value_full = value
            .clone()
            .slice_dim(2, 0..full_time)
            .reshape([batch, value_heads, full_chunks, chunk, embd])
            .swap_dims(1, 2);
        let grad_output_full = grad_output
            .clone()
            .slice_dim(2, 0..full_time)
            .reshape([batch, heads, full_chunks, chunk, embd])
            .swap_dims(1, 2);
        let flat_batch = batch * full_chunks;
        let stored_rho_before = chunk_rho_before.as_ref().filter(|tensor| {
            tensor.shape().dims::<6>() == [batch, full_chunks, heads, chunk, latent, embd]
        });
        let stored_rho_norm_before = chunk_rho_norm_before.as_ref().filter(|tensor| {
            tensor.shape().dims::<5>() == [batch, full_chunks, heads, chunk, latent]
        });
        let fallback_captured = if stored_rho_before.is_none() || stored_rho_norm_before.is_none() {
            Some(try_rwkv8_runtime_forward(
                query_full
                    .clone()
                    .reshape([flat_batch, heads, chunk, latent]),
                value_full
                    .clone()
                    .reshape([flat_batch, value_heads, chunk, embd]),
                full_rho_start
                    .clone()
                    .reshape([flat_batch, heads, latent, embd]),
                full_rho_norm_start
                    .clone()
                    .reshape([flat_batch, heads, latent]),
                decay.clone(),
                true,
            )?)
        } else {
            None
        };
        let rho_before = stored_rho_before
            .cloned()
            .or_else(|| {
                fallback_captured.as_ref().and_then(|captured| {
                    captured.rho_before.clone().map(|tensor| {
                        tensor.reshape([batch, full_chunks, heads, chunk, latent, embd])
                    })
                })
            })
            .expect("rwkv8 runtime forward must capture rho history");
        let rho_norm_before = stored_rho_norm_before
            .cloned()
            .or_else(|| {
                fallback_captured.as_ref().and_then(|captured| {
                    captured
                        .rho_norm_before
                        .clone()
                        .map(|tensor| tensor.reshape([batch, full_chunks, heads, chunk, latent]))
                })
            })
            .expect("rwkv8 runtime forward must capture rho_norm history");
        let prepared = try_rwkv8_runtime_backward_prepare(
            query_full
                .clone()
                .reshape([flat_batch, heads, chunk, latent]),
            rho_norm_before
                .clone()
                .reshape([flat_batch, heads, chunk, latent]),
            grad_output_full
                .clone()
                .reshape([flat_batch, heads, chunk, embd]),
            rho_before
                .clone()
                .reshape([flat_batch, heads, chunk, latent, embd]),
        )?;
        let zero_rho_scan = try_rwkv8_runtime_state_recurrence_from_zero(
            prepared.reversed_grad_rho_from_context.clone(),
            decay.clone(),
        )?;
        let zero_rho_norm_scan = try_rwkv8_runtime_norm_recurrence_from_zero(
            prepared.reversed_grad_rho_norm_from_context.clone(),
            decay.clone(),
        )?;
        let grad_query_weights =
            prepared
                .grad_query_weights
                .reshape([batch, full_chunks, heads, chunk, latent]);
        let reversed_grad_rho_from_context = prepared.reversed_grad_rho_from_context.reshape([
            batch,
            full_chunks,
            heads,
            chunk,
            latent,
            embd,
        ]);
        let reversed_grad_rho_norm_from_context = prepared
            .reversed_grad_rho_norm_from_context
            .reshape([batch, full_chunks, heads, chunk, latent]);
        let zero_prev_boundary_grad_rho =
            zero_rho_scan
                .final_state
                .reshape([batch, full_chunks, heads, latent, embd]);
        let zero_prev_boundary_grad_rho_norm =
            zero_rho_norm_scan
                .final_state
                .reshape([batch, full_chunks, heads, latent]);
        full_query = Some(query_full);
        full_value = Some(value_full);
        full_rho_before = Some(rho_before);
        full_rho_norm_before = Some(rho_norm_before);
        full_grad_query_weights = Some(grad_query_weights);
        full_reversed_grad_rho_from_context = Some(reversed_grad_rho_from_context);
        full_reversed_grad_rho_norm_from_context = Some(reversed_grad_rho_norm_from_context);
        full_zero_prev_boundary_grad_rho = Some(zero_prev_boundary_grad_rho);
        full_zero_prev_boundary_grad_rho_norm = Some(zero_prev_boundary_grad_rho_norm);
    }

    if tail_exists {
        let start = full_time;
        let query_chunk = query.clone().slice_dim(2, start..time);
        let value_chunk = value.clone().slice_dim(2, start..time);
        let grad_output_chunk = grad_output.clone().slice_dim(2, start..time);
        let tail_time = time - start;
        let stored_tail_rho_before = tail_rho_before
            .as_ref()
            .filter(|tensor| tensor.shape().dims::<5>() == [batch, heads, tail_time, latent, embd]);
        let stored_tail_rho_norm_before = tail_rho_norm_before
            .as_ref()
            .filter(|tensor| tensor.shape().dims::<4>() == [batch, heads, tail_time, latent]);
        let fallback_tail =
            if stored_tail_rho_before.is_none() || stored_tail_rho_norm_before.is_none() {
                Some(try_rwkv8_runtime_forward(
                    query_chunk.clone(),
                    value_chunk.clone(),
                    resolved_tail_rho_start
                        .clone()
                        .unwrap_or_else(|| resolved_rho.clone()),
                    resolved_tail_rho_norm_start
                        .clone()
                        .unwrap_or_else(|| resolved_rho_norm.clone()),
                    decay.clone(),
                    true,
                )?)
            } else {
                None
            };
        let rho_before = stored_tail_rho_before
            .cloned()
            .or_else(|| {
                fallback_tail
                    .as_ref()
                    .and_then(|captured| captured.rho_before.clone())
            })
            .expect("rwkv8 runtime forward must capture rho history");
        let rho_norm_before = stored_tail_rho_norm_before
            .cloned()
            .or_else(|| {
                fallback_tail
                    .as_ref()
                    .and_then(|captured| captured.rho_norm_before.clone())
            })
            .expect("rwkv8 runtime forward must capture rho_norm history");
        let prepared = try_rwkv8_runtime_backward_prepare(
            query_chunk.clone(),
            rho_norm_before.clone(),
            grad_output_chunk,
            rho_before.clone(),
        )?;
        let zero_rho_scan = try_rwkv8_runtime_state_recurrence_from_zero(
            prepared.reversed_grad_rho_from_context.clone(),
            decay.clone(),
        )?;
        let zero_rho_norm_scan = try_rwkv8_runtime_norm_recurrence_from_zero(
            prepared.reversed_grad_rho_norm_from_context.clone(),
            decay.clone(),
        )?;
        tail_prepared = Some(PreparedChunk {
            query_chunk,
            value_chunk,
            rho_before,
            rho_norm_before,
            grad_query_weights: prepared.grad_query_weights,
            reversed_grad_rho_from_context: prepared.reversed_grad_rho_from_context,
            reversed_grad_rho_norm_from_context: prepared.reversed_grad_rho_norm_from_context,
            zero_prev_boundary_grad_rho: zero_rho_scan.final_state,
            zero_prev_boundary_grad_rho_norm: zero_rho_norm_scan.final_state,
        });
    }

    let full_boundary_after_rho =
        if let Some(zero_prev_boundary_grad_rho) = full_zero_prev_boundary_grad_rho.as_ref() {
            let reverse_chunks = reverse_time_indices::<B>(full_chunks, &device);
            let reversed_deltas = zero_prev_boundary_grad_rho
                .clone()
                .select(1, reverse_chunks.clone())
                .swap_dims(1, 2);
            let chunk_decay = decay.clone().powf_scalar(chunk as f32);
            let history = if let Some(tail_prepared) = tail_prepared.as_ref() {
                try_rwkv8_runtime_state_history_recurrence(
                    reversed_deltas,
                    tail_prepared.zero_prev_boundary_grad_rho.clone(),
                    chunk_decay,
                )?
            } else {
                try_rwkv8_runtime_state_history_recurrence_from_zero(reversed_deltas, chunk_decay)?
            };
            Some(history.select(2, reverse_chunks).swap_dims(1, 2))
        } else {
            None
        };
    let full_boundary_after_rho_norm = if let Some(zero_prev_boundary_grad_rho_norm) =
        full_zero_prev_boundary_grad_rho_norm.as_ref()
    {
        let reverse_chunks = reverse_time_indices::<B>(full_chunks, &device);
        let reversed_deltas = zero_prev_boundary_grad_rho_norm
            .clone()
            .select(1, reverse_chunks.clone())
            .swap_dims(1, 2);
        let chunk_decay = decay.clone().powf_scalar(chunk as f32);
        let history = if let Some(tail_prepared) = tail_prepared.as_ref() {
            try_rwkv8_runtime_norm_history_recurrence(
                reversed_deltas,
                tail_prepared.zero_prev_boundary_grad_rho_norm.clone(),
                chunk_decay,
            )?
        } else {
            try_rwkv8_runtime_norm_history_recurrence_from_zero(reversed_deltas, chunk_decay)?
        };
        Some(history.select(2, reverse_chunks).swap_dims(1, 2))
    } else {
        None
    };

    let mut grad_query_chunks = Vec::with_capacity(1 + usize::from(tail_exists));
    let mut grad_value_chunks = Vec::with_capacity(1 + usize::from(tail_exists));
    let mut grad_decay = BurnTensor::<B, 3>::zeros([1, heads, latent], &device);

    if let (
        Some(full_query),
        Some(full_value),
        Some(full_rho_before),
        Some(full_rho_norm_before),
        Some(full_grad_query_weights),
        Some(full_reversed_grad_rho_from_context),
        Some(full_reversed_grad_rho_norm_from_context),
        Some(full_boundary_after_rho),
        Some(full_boundary_after_rho_norm),
    ) = (
        full_query,
        full_value,
        full_rho_before,
        full_rho_norm_before,
        full_grad_query_weights,
        full_reversed_grad_rho_from_context,
        full_reversed_grad_rho_norm_from_context,
        full_boundary_after_rho,
        full_boundary_after_rho_norm,
    ) {
        let flat_batch = batch * full_chunks;
        let grad_rho_carry = try_rwkv8_runtime_state_history_recurrence(
            full_reversed_grad_rho_from_context.reshape([flat_batch, heads, chunk, latent, embd]),
            full_boundary_after_rho.reshape([flat_batch, heads, latent, embd]),
            decay.clone(),
        )?;
        let grad_rho_norm_carry = try_rwkv8_runtime_norm_history_recurrence(
            full_reversed_grad_rho_norm_from_context.reshape([flat_batch, heads, chunk, latent]),
            full_boundary_after_rho_norm.reshape([flat_batch, heads, latent]),
            decay.clone(),
        )?;
        let finalized = try_rwkv8_runtime_backward_finalize(
            full_query.reshape([flat_batch, heads, chunk, latent]),
            full_value.reshape([flat_batch, value_heads, chunk, embd]),
            full_rho_before.reshape([flat_batch, heads, chunk, latent, embd]),
            full_rho_norm_before.reshape([flat_batch, heads, chunk, latent]),
            full_grad_query_weights.reshape([flat_batch, heads, chunk, latent]),
            reverse_time_tensor5(grad_rho_carry),
            reverse_time_tensor4(grad_rho_norm_carry),
        )?;
        grad_query_chunks.push(
            finalized
                .grad_query
                .reshape([batch, full_chunks, heads, chunk, latent])
                .swap_dims(1, 2)
                .reshape([batch, heads, full_time, latent]),
        );
        grad_value_chunks.push(if value_heads == 1 {
            finalized
                .grad_value
                .reshape([batch, full_chunks, 1, chunk, embd])
                .swap_dims(1, 2)
                .reshape([batch, 1, full_time, embd])
        } else {
            finalized
                .grad_value
                .reshape([batch, full_chunks, heads, chunk, embd])
                .swap_dims(1, 2)
                .reshape([batch, heads, full_time, embd])
        });
        grad_decay = grad_decay.add(finalized.grad_decay);
    }

    if tail_exists {
        let chunk_prepared = tail_prepared.expect("tail chunk exists for rwkv8 backward");
        let grad_rho_carry = try_rwkv8_runtime_state_history_recurrence_from_zero(
            chunk_prepared.reversed_grad_rho_from_context,
            decay.clone(),
        )?;
        let grad_rho_norm_carry = try_rwkv8_runtime_norm_history_recurrence_from_zero(
            chunk_prepared.reversed_grad_rho_norm_from_context,
            decay.clone(),
        )?;
        let finalized = try_rwkv8_runtime_backward_finalize(
            chunk_prepared.query_chunk,
            chunk_prepared.value_chunk,
            chunk_prepared.rho_before,
            chunk_prepared.rho_norm_before,
            chunk_prepared.grad_query_weights,
            reverse_time_tensor5(grad_rho_carry),
            reverse_time_tensor4(grad_rho_norm_carry),
        )?;
        grad_query_chunks.push(finalized.grad_query);
        grad_value_chunks.push(finalized.grad_value);
        grad_decay = grad_decay.add(finalized.grad_decay);
    }

    let grad_query = BurnTensor::cat(grad_query_chunks, 2);
    let grad_value = if value_heads == 1 {
        BurnTensor::cat(grad_value_chunks, 2).reshape([batch, 1, time, embd])
    } else {
        BurnTensor::cat(grad_value_chunks, 2)
    };

    Some((grad_query, grad_value, grad_decay))
}

pub(crate) fn tensorized_rwkv8_backward_impl<B>(
    ops: Ops<TensorizedRwkv8BackwardState<B::FloatTensorPrimitive>, 3>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
{
    let grad_output =
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
    let TensorizedRwkv8BackwardState {
        query: query_inner,
        value: value_inner,
        rho_state: rho_state_inner,
        rho_norm_state: rho_norm_state_inner,
        decay: decay_inner,
        chunk_size,
        chunk_rho_start,
        chunk_rho_norm_start,
        chunk_rho_before,
        chunk_rho_norm_before,
        tail_rho_start,
        tail_rho_norm_start,
        tail_rho_before,
        tail_rho_norm_before,
    } = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query_inner));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value_inner));
    let decay = BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(decay_inner));
    let rho_state = rho_state_inner
        .map(|inner| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(inner)));
    let rho_norm_state = rho_norm_state_inner
        .map(|inner| BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(inner)));
    let chunk_rho_start = chunk_rho_start
        .map(|inner| BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(inner)));
    let chunk_rho_norm_start = chunk_rho_norm_start
        .map(|inner| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(inner)));
    let chunk_rho_before = chunk_rho_before
        .map(|inner| BurnTensor::<B, 6>::from_primitive(TensorPrimitive::Float(inner)));
    let chunk_rho_norm_before = chunk_rho_norm_before
        .map(|inner| BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(inner)));
    let tail_rho_start = tail_rho_start
        .map(|inner| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(inner)));
    let tail_rho_norm_start = tail_rho_norm_start
        .map(|inner| BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(inner)));
    let tail_rho_before = tail_rho_before
        .map(|inner| BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(inner)));
    let tail_rho_norm_before = tail_rho_norm_before
        .map(|inner| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(inner)));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [_, value_heads, _, embd] = value.shape().dims::<4>();
    let device = query.device();
    let chunk = chunk_size.min(time.max(1)).max(1);

    if let Some((grad_query, grad_value, grad_decay)) =
        try_tensorized_rwkv8_backward_runtime_chunked(
            query.clone(),
            value.clone(),
            rho_state.clone(),
            rho_norm_state.clone(),
            chunk_rho_start,
            chunk_rho_norm_start,
            chunk_rho_before,
            chunk_rho_norm_before,
            tail_rho_start,
            tail_rho_norm_start,
            tail_rho_before,
            tail_rho_norm_before,
            decay.clone(),
            grad_output.clone(),
            chunk,
        )
    {
        if let Some(parent) = &parents[0] {
            grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
        }
        if let Some(parent) = &parents[1] {
            grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
        }
        if let Some(parent) = &parents[2] {
            grads.register::<B>(parent.id, grad_decay.into_primitive().tensor());
        }
        return;
    }

    let mut grad_query_chunks = Vec::with_capacity(time.div_ceil(chunk));
    let mut grad_value_chunks = Vec::with_capacity(time.div_ceil(chunk));
    let mut grad_decay = BurnTensor::<B, 3>::zeros([1, heads, latent], &device);
    let mut chunk_entries = Vec::with_capacity(time.div_ceil(chunk));

    let mut rho = resolve_rho_state(rho_state, batch, heads, latent, embd, &device);
    let mut rho_norm = resolve_rho_norm_state(rho_norm_state, batch, heads, latent, &device);
    for start in (0..time).step_by(chunk) {
        let end = (start + chunk).min(time);
        let query_chunk = query.clone().slice_dim(2, start..end);
        let value_chunk = value.clone().slice_dim(2, start..end);
        let (rho_before, rho_norm_before, next_rho, next_rho_norm) = compute_chunk_traces(
            query_chunk.clone(),
            value_chunk.clone(),
            rho.clone(),
            rho_norm.clone(),
            decay.clone(),
        );
        chunk_entries.push((
            start,
            end,
            query_chunk,
            value_chunk,
            rho_before,
            rho_norm_before,
        ));
        rho = next_rho;
        rho_norm = next_rho_norm;
    }

    let mut boundary_grad_rho_after =
        BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], &device);
    let mut boundary_grad_rho_norm_after =
        BurnTensor::<B, 3>::zeros([batch, heads, latent], &device);

    for (start, end, query_chunk, value_chunk, rho_before, rho_norm_before) in
        chunk_entries.into_iter().rev()
    {
        let grad_output_chunk = grad_output.clone().slice_dim(2, start..end);
        let (
            grad_query_chunk,
            grad_value_chunk,
            grad_decay_chunk,
            prev_boundary_grad_rho,
            prev_boundary_grad_rho_norm,
        ) = backward_chunk_impl(
            query_chunk,
            value_chunk,
            rho_before,
            rho_norm_before,
            decay.clone(),
            grad_output_chunk,
            boundary_grad_rho_after,
            boundary_grad_rho_norm_after,
        );
        grad_query_chunks.push(grad_query_chunk);
        grad_value_chunks.push(grad_value_chunk);
        grad_decay = grad_decay.add(grad_decay_chunk);
        boundary_grad_rho_after = prev_boundary_grad_rho;
        boundary_grad_rho_norm_after = prev_boundary_grad_rho_norm;
    }

    grad_query_chunks.reverse();
    grad_value_chunks.reverse();

    let grad_query = BurnTensor::cat(grad_query_chunks, 2);
    let grad_value_expanded = BurnTensor::cat(grad_value_chunks, 2);
    let grad_value = if value_heads == 1 {
        grad_value_expanded
            .sum_dim(1)
            .reshape([batch, 1, time, embd])
    } else {
        grad_value_expanded
    };

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_decay.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 3> for TensorizedRwkv8Backward<WgpuCubeBackend> {
    type State = TensorizedRwkv8BackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_rwkv8_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 3> for TensorizedRwkv8Backward<CudaCubeBackend> {
    type State = TensorizedRwkv8BackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_rwkv8_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}
