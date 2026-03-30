use std::any::Any;
use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData, TensorPrimitive};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::ops::{Backward, OpsKind};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_wgpu::{CubeBackend, WgpuRuntime};

use crate::kernels::sequence::rwkv8::backward::{
    TensorizedRwkv8Backward, TensorizedRwkv8BackwardState,
};
use crate::kernels::sequence::rwkv8::runtime::try_rwkv8_runtime_forward;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

/// The analytical wrapper now uses a real Cube runtime kernel for the RWKV forward shell.
pub const AVAILABLE: bool = true;

#[derive(Debug)]
pub struct Rwkv8ForwardOutput<B: BackendTrait> {
    pub context: Tensor<B, 4>,
    pub rho: Tensor<B, 4>,
    pub rho_norm: Tensor<B, 3>,
}

pub fn use_tensorized_rwkv8_forward_experimental() -> bool {
    match std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_tensorized_rwkv8_train_wrapper() -> bool {
    match std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn tensorized_rwkv8_runtime_chunked_forward<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
    chunk: usize,
) -> Option<Rwkv8ForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let mut rho_state = match rho_state {
        Some(existing) if existing.shape().dims::<4>() == [batch, heads, latent, embd] => existing,
        _ => Tensor::<B, 4>::zeros([batch, heads, latent, embd], &device),
    };
    let mut rho_norm_state = match rho_norm_state {
        Some(existing) if existing.shape().dims::<3>() == [batch, heads, latent] => existing,
        _ => Tensor::<B, 3>::zeros([batch, heads, latent], &device),
    };

    let mut outputs = Vec::with_capacity(time.div_ceil(chunk.max(1)));
    for start in (0..time).step_by(chunk.max(1)) {
        let end = (start + chunk.max(1)).min(time);
        let chunk_output = try_rwkv8_runtime_forward(
            query.clone().slice_dim(2, start..end),
            value.clone().slice_dim(2, start..end),
            rho_state,
            rho_norm_state,
            decay.clone(),
            false,
        )?;
        outputs.push(chunk_output.context);
        rho_state = chunk_output.rho;
        rho_norm_state = chunk_output.rho_norm;
    }

    Some(Rwkv8ForwardOutput {
        context: Tensor::cat(outputs, 2),
        rho: rho_state,
        rho_norm: rho_norm_state,
    })
}

pub fn tensorized_rwkv8_forward<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    if use_tensorized_rwkv8_train_wrapper() {
        if let Some(output) = try_tensorized_rwkv8_autodiff_cube::<B>(
            query.clone(),
            value.clone(),
            rho_state.clone(),
            rho_norm_state.clone(),
            decay.clone(),
        ) {
            return output;
        }
    }
    tensorized_rwkv8_forward_impl(query, value, rho_state, rho_norm_state, decay)
}

pub fn tensorized_rwkv8_forward_direct_graph<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    tensorized_rwkv8_forward_reference_graph(query, value, rho_state, rho_norm_state, decay)
}

fn tensorized_rwkv8_forward_reference_graph<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let chunk = rwkv8_tensorized_chunk_size::<B>(batch, heads, time, latent, embd);
    if chunk >= time {
        return tensorized_rwkv8_forward_scan_chunk(query, value, rho_state, rho_norm_state, decay);
    }

    let mut outputs = Vec::with_capacity(time.div_ceil(chunk));
    let mut rho_state = rho_state;
    let mut rho_norm_state = rho_norm_state;
    for start in (0..time).step_by(chunk) {
        let end = (start + chunk).min(time);
        let chunk_out = tensorized_rwkv8_forward_scan_chunk(
            query.clone().slice_dim(2, start..end),
            value.clone().slice_dim(2, start..end),
            rho_state,
            rho_norm_state,
            decay.clone(),
        );
        outputs.push(chunk_out.context);
        rho_state = Some(chunk_out.rho);
        rho_norm_state = Some(chunk_out.rho_norm);
    }

    Rwkv8ForwardOutput {
        context: Tensor::cat(outputs, 2),
        rho: rho_state.expect("rwkv8 reference graph forward must produce rho"),
        rho_norm: rho_norm_state.expect("rwkv8 reference graph forward must produce rho_norm"),
    }
}

pub(crate) fn tensorized_rwkv8_forward_impl<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let chunk = rwkv8_tensorized_chunk_size::<B>(batch, heads, time, latent, embd);
    if chunk >= time {
        return tensorized_rwkv8_forward_single_chunk(
            query,
            value,
            rho_state,
            rho_norm_state,
            decay,
        );
    }

    let mut outputs = Vec::with_capacity(time.div_ceil(chunk));
    let mut rho_state = rho_state;
    let mut rho_norm_state = rho_norm_state;
    for start in (0..time).step_by(chunk) {
        let end = (start + chunk).min(time);
        let chunk_out = tensorized_rwkv8_forward_single_chunk(
            query.clone().slice_dim(2, start..end),
            value.clone().slice_dim(2, start..end),
            rho_state,
            rho_norm_state,
            decay.clone(),
        );
        outputs.push(chunk_out.context);
        rho_state = Some(chunk_out.rho);
        rho_norm_state = Some(chunk_out.rho_norm);
    }

    Rwkv8ForwardOutput {
        context: Tensor::cat(outputs, 2),
        rho: rho_state.expect("rwkv8 chunked forward must produce rho"),
        rho_norm: rho_norm_state.expect("rwkv8 chunked forward must produce rho_norm"),
    }
}

fn try_tensorized_rwkv8_autodiff_cube<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Option<Rwkv8ForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_tensorized_rwkv8_autodiff_wgpu::<B>(
        query.clone(),
        value.clone(),
        rho_state.clone(),
        rho_norm_state.clone(),
        decay.clone(),
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_tensorized_rwkv8_autodiff_cuda::<B>(query, value, rho_state, rho_norm_state, decay)
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

fn try_tensorized_rwkv8_autodiff_wgpu<B>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Option<Rwkv8ForwardOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
{
    let query_shape = query.shape().dims::<4>();
    let value_shape = value.shape().dims::<4>();
    let query_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.into_primitive().tensor())?;
    let value_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.into_primitive().tensor())?;
    let decay_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.into_primitive().tensor())?;
    let rho_state_inner = match rho_state {
        Some(state) => {
            let state_ad: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let rho_norm_state_inner = match rho_norm_state {
        Some(state) => {
            let state_ad: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let decay_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());

    let chunk_size = rwkv8_tensorized_chunk_size::<WgpuCubeBackend>(
        query_shape[0],
        query_shape[1],
        query_shape[2],
        query_shape[3],
        value_shape[3],
    );
    let output = tensorized_rwkv8_runtime_chunked_forward(
        BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        rho_state_inner.clone().map(|inner| {
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(inner))
        }),
        rho_norm_state_inner.clone().map(|inner| {
            BurnTensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(inner))
        }),
        BurnTensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        chunk_size,
    )
    .unwrap_or_else(|| {
        tensorized_rwkv8_forward_impl(
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                query_inner.clone(),
            )),
            BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                value_inner.clone(),
            )),
            rho_state_inner.clone().map(|inner| {
                BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(inner))
            }),
            rho_norm_state_inner.clone().map(|inner| {
                BurnTensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(inner))
            }),
            BurnTensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(
                decay_inner.clone(),
            )),
        )
    });
    let context_inner = output.context.into_primitive().tensor();
    let rho_inner = output.rho.into_primitive().tensor();
    let rho_norm_inner = output.rho_norm.into_primitive().tensor();

    let context_ad = match TensorizedRwkv8Backward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            TensorizedRwkv8BackwardState {
                query: query_inner,
                value: value_inner,
                rho_state: rho_state_inner,
                rho_norm_state: rho_norm_state_inner,
                decay: decay_inner,
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };
    let rho_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner);
    let rho_norm_ad = <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_norm_inner);
    let context_primitive = try_cast_backend::<B, _>(context_ad)?;
    let rho_primitive = try_cast_backend::<B, _>(rho_ad)?;
    let rho_norm_primitive = try_cast_backend::<B, _>(rho_norm_ad)?;

    Some(Rwkv8ForwardOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_primitive)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_primitive)),
        rho_norm: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(rho_norm_primitive)),
    })
}

#[cfg(feature = "cuda")]
fn try_tensorized_rwkv8_autodiff_cuda<B>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Option<Rwkv8ForwardOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
{
    let query_shape = query.shape().dims::<4>();
    let value_shape = value.shape().dims::<4>();
    let query_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.into_primitive().tensor())?;
    let value_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.into_primitive().tensor())?;
    let decay_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.into_primitive().tensor())?;
    let rho_state_inner = match rho_state {
        Some(state) => {
            let state_ad: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let rho_norm_state_inner = match rho_norm_state {
        Some(state) => {
            let state_ad: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let query_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let decay_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());

    let chunk_size = rwkv8_tensorized_chunk_size::<CudaCubeBackend>(
        query_shape[0],
        query_shape[1],
        query_shape[2],
        query_shape[3],
        value_shape[3],
    );
    let output = tensorized_rwkv8_runtime_chunked_forward(
        BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        rho_state_inner.clone().map(|inner| {
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(inner))
        }),
        rho_norm_state_inner.clone().map(|inner| {
            BurnTensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(inner))
        }),
        BurnTensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        chunk_size,
    )
    .unwrap_or_else(|| {
        tensorized_rwkv8_forward_impl(
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                query_inner.clone(),
            )),
            BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                value_inner.clone(),
            )),
            rho_state_inner.clone().map(|inner| {
                BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(inner))
            }),
            rho_norm_state_inner.clone().map(|inner| {
                BurnTensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(inner))
            }),
            BurnTensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(
                decay_inner.clone(),
            )),
        )
    });
    let context_inner = output.context.into_primitive().tensor();
    let rho_inner = output.rho.into_primitive().tensor();
    let rho_norm_inner = output.rho_norm.into_primitive().tensor();

    let context_ad = match TensorizedRwkv8Backward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            TensorizedRwkv8BackwardState {
                query: query_inner,
                value: value_inner,
                rho_state: rho_state_inner,
                rho_norm_state: rho_norm_state_inner,
                decay: decay_inner,
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };
    let rho_ad = <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner);
    let rho_norm_ad = <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_norm_inner);
    let context_primitive = try_cast_backend::<B, _>(context_ad)?;
    let rho_primitive = try_cast_backend::<B, _>(rho_ad)?;
    let rho_norm_primitive = try_cast_backend::<B, _>(rho_norm_ad)?;

    Some(Rwkv8ForwardOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_primitive)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_primitive)),
        rho_norm: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(rho_norm_primitive)),
    })
}

pub(crate) fn rwkv8_forward_recurrence_before_5d<B: BackendTrait>(
    delta: Tensor<B, 5>,
    rho_state: Tensor<B, 4>,
    decay: Tensor<B, 3>,
) -> (Tensor<B, 5>, Tensor<B, 4>) {
    let [batch, heads, time, latent, embd] = delta.shape().dims::<5>();
    let device = delta.device();
    let time_idx = Tensor::<B, 1, Int>::arange(0..time as i64, &device).float();
    let decay5 = decay
        .clone()
        .reshape([1, heads, 1, latent, 1])
        .repeat_dim(2, time);
    let state_exp5 = time_idx
        .clone()
        .reshape([1, 1, time, 1, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let inv_exp5 = time_idx
        .clone()
        .add_scalar(1.0)
        .mul_scalar(-1.0)
        .reshape([1, 1, time, 1, 1])
        .repeat_dim(1, heads)
        .repeat_dim(3, latent);
    let state_powers = decay5.clone().powf(state_exp5.clone());
    let rho_before = exclusive_prefix_sum_time_5d(delta.clone() * decay5.powf(inv_exp5))
        * state_powers.clone()
        + rho_state
            .reshape([batch, heads, 1, latent, embd])
            .repeat_dim(2, time)
            .mul(state_powers);
    let last_rho_before = rho_before
        .clone()
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let last_delta = delta
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let rho = last_rho_before
        .mul(decay.reshape([1, heads, latent, 1]))
        .add(last_delta);
    (rho_before, rho)
}

pub(crate) fn rwkv8_forward_recurrence_before_4d<B: BackendTrait>(
    query: Tensor<B, 4>,
    rho_norm_state: Tensor<B, 3>,
    decay: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let decay = decay.reshape([1, heads, latent]);
    let mut rho_norm = rho_norm_state;
    let mut rho_norm_before = Vec::with_capacity(time);
    for step in 0..time {
        rho_norm_before.push(rho_norm.clone().reshape([batch, heads, 1, latent]));
        let query_t = query
            .clone()
            .slice_dim(2, step..step + 1)
            .squeeze_dim::<3>(2);
        rho_norm = rho_norm.mul(decay.clone()).add(query_t);
    }
    (Tensor::cat(rho_norm_before, 2), rho_norm)
}

fn tensorized_rwkv8_forward_single_chunk<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    if rwkv8_tensorized_forward_should_use_scan::<B>(batch, heads, time, latent, embd) {
        if rwkv8_tensorized_forward_should_use_matmul_chunk(time) {
            return tensorized_rwkv8_forward_matmul_chunk(
                query,
                value,
                rho_state,
                rho_norm_state,
                decay,
            );
        }
        return tensorized_rwkv8_forward_scan_chunk(query, value, rho_state, rho_norm_state, decay);
    }

    let device = query.device();
    let delta = value_outer_5d(query.clone(), value);
    let rho_state = match rho_state {
        Some(existing) if existing.shape().dims::<4>() == [batch, heads, latent, embd] => existing,
        _ => Tensor::<B, 4>::zeros([batch, heads, latent, embd], &device),
    };
    let rho_norm_state = match rho_norm_state {
        Some(existing) if existing.shape().dims::<3>() == [batch, heads, latent] => existing,
        _ => Tensor::<B, 3>::zeros([batch, heads, latent], &device),
    };
    let (rho_before, rho) =
        rwkv8_forward_recurrence_before_5d(delta.clone(), rho_state, decay.clone());
    let (rho_norm_before, rho_norm) =
        rwkv8_forward_recurrence_before_4d(query.clone(), rho_norm_state, decay.clone());

    let q_weights = query.clone().div(
        query
            .clone()
            .sum_dim(3)
            .add_scalar(1.0e-6)
            .reshape([batch, heads, time, 1]),
    );
    let context = rho_before
        .clone()
        .div(
            rho_norm_before
                .clone()
                .add_scalar(1.0e-6)
                .unsqueeze_dim::<5>(4),
        )
        .mul(q_weights.unsqueeze_dim::<5>(4))
        .sum_dim(3)
        .reshape([batch, heads, time, embd]);

    Rwkv8ForwardOutput {
        context,
        rho,
        rho_norm,
    }
}

fn tensorized_rwkv8_forward_scan_chunk<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let value_heads = value.shape().dims::<4>()[1];
    let decay = decay.reshape([1, heads, latent]);

    let mut rho = match rho_state {
        Some(existing) if existing.shape().dims::<4>() == [batch, heads, latent, embd] => existing,
        _ => Tensor::<B, 4>::zeros([batch, heads, latent, embd], &device),
    };
    let mut rho_norm = match rho_norm_state {
        Some(existing) if existing.shape().dims::<3>() == [batch, heads, latent] => existing,
        _ => Tensor::<B, 3>::zeros([batch, heads, latent], &device),
    };

    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let query_t = query
            .clone()
            .slice_dim(2, step..step + 1)
            .squeeze_dim::<3>(2);
        let value_t = value.clone().slice_dim(2, step..step + 1).squeeze_dim::<3>(2);

        let q_weights = rwkv8_query_weights_step(query_t.clone());
        let context_t = rho
            .clone()
            .div(
                rho_norm
                    .clone()
                    .add_scalar(1.0e-6)
                    .reshape([batch, heads, latent, 1]),
            )
            .mul(q_weights.reshape([batch, heads, latent, 1]))
            .sum_dim(2)
            .reshape([batch, heads, 1, embd]);
        outputs.push(context_t);

        let delta_t = query_t.clone().reshape([batch, heads, latent, 1])
            * match value_heads {
                1 => value_t.reshape([batch, 1, 1, embd]),
                existing if existing == heads => value_t.reshape([batch, heads, 1, embd]),
                existing => panic!("value heads {existing} must be 1 or {heads}"),
            };
        rho = rho
            .mul(decay.clone().reshape([1, heads, latent, 1]))
            .add(delta_t);
        rho_norm = rho_norm.mul(decay.clone()).add(query_t);
    }

    Rwkv8ForwardOutput {
        context: Tensor::cat(outputs, 2),
        rho,
        rho_norm,
    }
}

fn tensorized_rwkv8_forward_matmul_chunk<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> Rwkv8ForwardOutput<B> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let value = expand_value_heads(value, heads);
    let heads_latent = heads * latent;

    let delta = query.clone().unsqueeze_dim::<5>(4) * value.clone().unsqueeze_dim::<5>(3);
    let delta_flat = delta
        .clone()
        .swap_dims(2, 3)
        .reshape([batch * heads_latent, time, embd]);
    let query_flat = query
        .clone()
        .swap_dims(2, 3)
        .reshape([batch * heads_latent, time, 1]);

    let weights = rwkv8_transition_weights(decay.clone(), time, &device).repeat_dim(0, batch);
    let rho_before_flat = weights.clone().matmul(delta_flat);
    let rho_norm_before_flat = weights
        .matmul(query_flat)
        .reshape([batch * heads_latent, time]);

    let mut rho_before = rho_before_flat
        .reshape([batch, heads, latent, time, embd])
        .swap_dims(2, 3);
    let mut rho_norm_before = rho_norm_before_flat
        .reshape([batch, heads, latent, time])
        .swap_dims(2, 3);

    let state_decay = rwkv8_state_decay_weights(decay.clone(), time, &device).repeat_dim(0, batch);
    if let Some(rho_state) =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, embd])
    {
        let rho_state_flat = rho_state.reshape([batch * heads_latent, 1, embd]);
        rho_before = rho_before
            + (state_decay.clone().reshape([batch * heads_latent, time, 1]) * rho_state_flat)
                .reshape([batch, heads, latent, time, embd])
                .swap_dims(2, 3);
    }
    if let Some(rho_norm_state) =
        rho_norm_state.filter(|state| state.shape().dims::<3>() == [batch, heads, latent])
    {
        rho_norm_before = rho_norm_before
            + (state_decay * rho_norm_state.reshape([batch * heads_latent, 1]))
                .reshape([batch, heads, latent, time])
                .swap_dims(2, 3);
    }

    let q_weights = query.clone().div(
        query
            .clone()
            .sum_dim(3)
            .add_scalar(1.0e-6)
            .reshape([batch, heads, time, 1]),
    );
    let context = rho_before
        .clone()
        .div(
            rho_norm_before
                .clone()
                .add_scalar(1.0e-6)
                .unsqueeze_dim::<5>(4),
        )
        .mul(q_weights.unsqueeze_dim::<5>(4))
        .sum_dim(3)
        .reshape([batch, heads, time, embd]);

    let last_rho_before = rho_before
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let last_rho_norm_before = rho_norm_before
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent]);
    let last_delta = delta
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, latent, embd]);
    let last_query = query
        .slice_dim(2, time - 1..time)
        .squeeze_dim::<3>(2);

    let rho = last_rho_before
        .mul(decay.clone().reshape([1, heads, latent, 1]))
        .add(last_delta);
    let rho_norm = last_rho_norm_before
        .mul(decay.reshape([1, heads, latent]))
        .add(last_query);

    Rwkv8ForwardOutput {
        context,
        rho,
        rho_norm,
    }
}

pub(crate) fn rwkv8_tensorized_chunk_size<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
) -> usize {
    if let Some(explicit) = std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return explicit.min(time.max(1));
    }

    let threshold_bytes = rwkv8_tensorized_scan_threshold_bytes::<B>();
    let backend_name = std::any::type_name::<B>();
    let mut chunk = if backend_name.contains("CudaRuntime") {
        // Real CUDA training at the Shakespeare base shape prefers a much wider
        // chunk window than the older conservative default once the runtime
        // forward/backward shell is fully fused and state outputs are pure.
        time.min(128).max(1)
    } else {
        time.max(1)
    };
    while chunk > 1
        && rwkv8_tensorized_scratch_bytes(batch, heads, chunk, latent, embd) > threshold_bytes
    {
        chunk = chunk.div_ceil(2);
    }
    chunk.max(1)
}

fn rwkv8_query_weights_step<B: BackendTrait>(query_t: Tensor<B, 3>) -> Tensor<B, 3> {
    query_t
        .clone()
        .div(query_t.clone().sum_dim(2).add_scalar(1.0e-6).reshape([
            query_t.shape().dims::<3>()[0],
            query_t.shape().dims::<3>()[1],
            1,
        ]))
}

fn rwkv8_tensorized_forward_should_use_scan<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
) -> bool {
    let threshold_bytes = rwkv8_tensorized_scan_threshold_bytes::<B>();
    let tensorized_scratch_bytes = rwkv8_tensorized_scratch_bytes(batch, heads, time, latent, embd);
    tensorized_scratch_bytes >= threshold_bytes
}

fn rwkv8_tensorized_forward_should_use_matmul_chunk(time: usize) -> bool {
    std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_MATMUL_MAX_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8)
        .min(time.max(1))
        >= time
}

fn rwkv8_tensorized_scan_threshold_bytes<B: BackendTrait>() -> usize {
    std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or_else(|| {
            let backend_name = std::any::type_name::<B>();
            if backend_name.contains("CudaRuntime") {
                4 * 1024 * 1024 * 1024
            } else {
                128 * 1024 * 1024
            }
        })
}

fn rwkv8_tensorized_scratch_bytes(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
) -> usize {
    let bhte = batch
        .saturating_mul(heads)
        .saturating_mul(time)
        .saturating_mul(latent)
        .saturating_mul(embd);
    let bhtl = batch
        .saturating_mul(heads)
        .saturating_mul(time)
        .saturating_mul(latent);

    bhte.saturating_mul(3)
        .saturating_add(bhtl.saturating_mul(2))
        .saturating_mul(std::mem::size_of::<f32>())
}

pub(crate) fn value_outer_5d<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
) -> Tensor<B, 5> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let [_, value_heads, _, embd] = value.shape().dims::<4>();
    let value = match value_heads {
        1 => value.reshape([batch, 1, time, 1, embd]),
        existing if existing == heads => value.reshape([batch, heads, time, 1, embd]),
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    };
    query.reshape([batch, heads, time, latent, 1]) * value
}

fn expand_value_heads<B: BackendTrait>(value: Tensor<B, 4>, heads: usize) -> Tensor<B, 4> {
    match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    }
}

pub(crate) fn rwkv8_transition_weights<B: BackendTrait>(
    decay: Tensor<B, 3>,
    time: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let heads = decay.shape().dims::<3>()[1];
    let latent = decay.shape().dims::<3>()[2];
    let decay = decay.reshape([heads * latent, 1, 1]);
    let (exponents, mask) = rwkv8_transition_weight_data(time);
    let exponents = Tensor::<B, 3>::from_data(TensorData::new(exponents, [1, time, time]), device)
        .repeat_dim(0, heads * latent);
    let mask = Tensor::<B, 3>::from_data(TensorData::new(mask, [1, time, time]), device)
        .repeat_dim(0, heads * latent);
    decay.powf(exponents) * mask
}

pub(crate) fn rwkv8_state_decay_weights<B: BackendTrait>(
    decay: Tensor<B, 3>,
    time: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let heads = decay.shape().dims::<3>()[1];
    let latent = decay.shape().dims::<3>()[2];
    let decay = decay.reshape([heads * latent, 1]);
    let exponents = Tensor::<B, 2>::from_data(
        TensorData::new((0..time).map(|step| step as f32).collect(), [1, time]),
        device,
    )
    .repeat_dim(0, heads * latent);
    decay.powf(exponents)
}

fn rwkv8_transition_weight_data(time: usize) -> (Vec<f32>, Vec<f32>) {
    let mut exponents = Vec::with_capacity(time * time);
    let mut mask = Vec::with_capacity(time * time);
    for row in 0..time {
        for col in 0..time {
            if col < row {
                exponents.push((row - col - 1) as f32);
                mask.push(1.0);
            } else {
                exponents.push(0.0);
                mask.push(0.0);
            }
        }
    }
    (exponents, mask)
}

fn exclusive_prefix_sum_time_5d<B: BackendTrait>(tensor: Tensor<B, 5>) -> Tensor<B, 5> {
    let [batch, heads, time, latent, embd] = tensor.shape().dims::<5>();
    let prefix = tensor.cumsum(2);
    let zero = Tensor::<B, 5>::zeros([batch, heads, 1, latent, embd], &prefix.device());
    if time == 1 {
        zero
    } else {
        Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
    }
}

fn exclusive_prefix_sum_time_4d<B: BackendTrait>(tensor: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = tensor.shape().dims::<4>();
    let prefix = tensor.cumsum(2);
    let zero = Tensor::<B, 4>::zeros([batch, heads, 1, latent], &prefix.device());
    if time == 1 {
        zero
    } else {
        Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
    }
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
