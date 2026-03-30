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
    try_rwkv8_runtime_advance_state, try_rwkv8_runtime_backward_chunk, try_rwkv8_runtime_forward,
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
    decay: BurnTensor<B, 3>,
    grad_output: BurnTensor<B, 4>,
    chunk_size: usize,
) -> Option<(
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let chunk = chunk_size.min(time.max(1)).max(1);

    let mut rho = resolve_rho_state(rho_state, batch, heads, latent, embd, &device);
    let mut rho_norm = resolve_rho_norm_state(rho_norm_state, batch, heads, latent, &device);
    let mut chunk_starts = Vec::with_capacity(time.div_ceil(chunk));
    for start in (0..time).step_by(chunk) {
        let end = (start + chunk).min(time);
        chunk_starts.push((start, end, rho.clone(), rho_norm.clone()));
        let next = try_rwkv8_runtime_advance_state(
            query.clone().slice_dim(2, start..end),
            value.clone().slice_dim(2, start..end),
            rho,
            rho_norm,
            decay.clone(),
        )?;
        rho = next.rho;
        rho_norm = next.rho_norm;
    }

    let mut grad_query_chunks = Vec::with_capacity(chunk_starts.len());
    let mut grad_value_chunks = Vec::with_capacity(chunk_starts.len());
    let mut grad_decay = BurnTensor::<B, 3>::zeros([1, heads, latent], &device);
    let mut boundary_grad_rho_after =
        BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], &device);
    let mut boundary_grad_rho_norm_after =
        BurnTensor::<B, 3>::zeros([batch, heads, latent], &device);

    for (start, end, rho_start, rho_norm_start) in chunk_starts.into_iter().rev() {
        let query_chunk = query.clone().slice_dim(2, start..end);
        let value_chunk = value.clone().slice_dim(2, start..end);
        let grad_output_chunk = grad_output.clone().slice_dim(2, start..end);
        let captured = try_rwkv8_runtime_forward(
            query_chunk.clone(),
            value_chunk.clone(),
            rho_start,
            rho_norm_start,
            decay.clone(),
            true,
        )?;
        let backward = try_rwkv8_runtime_backward_chunk(
            query_chunk,
            value_chunk,
            captured
                .rho_before
                .expect("rwkv8 runtime forward must capture rho history"),
            captured
                .rho_norm_before
                .expect("rwkv8 runtime forward must capture rho_norm history"),
            decay.clone(),
            grad_output_chunk,
            boundary_grad_rho_after,
            boundary_grad_rho_norm_after,
        )?;
        grad_query_chunks.push(backward.grad_query);
        grad_value_chunks.push(backward.grad_value);
        grad_decay = grad_decay.add(backward.grad_decay);
        boundary_grad_rho_after = backward.prev_boundary_grad_rho;
        boundary_grad_rho_norm_after = backward.prev_boundary_grad_rho_norm;
    }

    grad_query_chunks.reverse();
    grad_value_chunks.reverse();

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
    } = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query_inner));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value_inner));
    let decay = BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(decay_inner));
    let rho_state = rho_state_inner
        .map(|inner| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(inner)));
    let rho_norm_state = rho_norm_state_inner
        .map(|inner| BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(inner)));

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
