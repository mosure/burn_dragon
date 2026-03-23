use std::any::Any;
use std::marker::PhantomData;
use std::sync::Once;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Tensor, TensorPrimitive, activation};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::ops::{Backward, OpsKind};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_wgpu::{CubeBackend, WgpuRuntime};

use crate::kernels::sequence::mamba::selective_scan_backward::{
    MambaTensorizedBackwardState, TensorizedMambaBackward,
};

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

/// This remains false until a real custom fused kernel lands. The current implementation is a
/// tensorized experimental forward path with a custom recompute backward wrapper, not a true fused
/// selective-scan kernel.
pub const AVAILABLE: bool = false;

#[derive(Debug, Clone)]
pub struct MambaTensorizedState<B: BackendTrait> {
    pub conv: Tensor<B, 4>,
    pub ssm: Tensor<B, 4>,
}

#[derive(Debug)]
pub struct MambaTensorizedOutput<B: BackendTrait> {
    pub context: Tensor<B, 4>,
    pub state: MambaTensorizedState<B>,
}

pub fn use_tensorized_mamba_forward_experimental() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA_TENSORIZED_FORWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_tensorized_mamba_train_wrapper() -> bool {
    matches!(
        std::env::var("BURN_DRAGON_MAMBA_TENSORIZED_TRAIN_WRAPPER")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

fn should_log_mamba_path_selection() -> bool {
    matches!(
        std::env::var("BURN_DRAGON_MAMBA_LOG_PATH_SELECTION")
            .ok()
            .as_deref(),
        Some("1") | Some("true") | Some("TRUE") | Some("on") | Some("ON")
    )
}

fn log_mamba_path_selection_once(message: &str) {
    static ONCE: Once = Once::new();
    if should_log_mamba_path_selection() {
        ONCE.call_once(|| eprintln!("{message}"));
    }
}

#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba_forward<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    x_proj: Tensor<B, 2>,
    dt_proj_weight: Tensor<B, 2>,
    dt_proj_bias: Tensor<B, 1>,
    a_log: Tensor<B, 2>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<MambaTensorizedState<B>>,
) -> MambaTensorizedOutput<B> {
    if use_tensorized_mamba_train_wrapper() {
        if let Some(output) = try_tensorized_mamba_autodiff_cube(
            hidden_states.clone(),
            d_inner,
            d_state,
            d_conv,
            dt_rank,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            x_proj.clone(),
            dt_proj_weight.clone(),
            dt_proj_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            out_proj.clone(),
            state.clone(),
        ) {
            log_mamba_path_selection_once(
                "mamba tensorized path: using cube autodiff train wrapper",
            );
            return output;
        }

        log_mamba_path_selection_once(
            "mamba tensorized path: train wrapper unavailable, falling back to direct tensorized implementation",
        );
    } else {
        log_mamba_path_selection_once(
            "mamba tensorized path: using direct tensorized implementation (train wrapper disabled)",
        );
    }

    tensorized_mamba_forward_impl(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        dt_rank,
        in_proj,
        conv_weight,
        conv_bias,
        x_proj,
        dt_proj_weight,
        dt_proj_bias,
        a_log,
        d_skip,
        out_proj,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorized_mamba_forward_impl<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    _d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    x_proj: Tensor<B, 2>,
    dt_proj_weight: Tensor<B, 2>,
    dt_proj_bias: Tensor<B, 1>,
    a_log: Tensor<B, 2>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<MambaTensorizedState<B>>,
) -> MambaTensorizedOutput<B> {
    let [batch, views, time, d_model] = hidden_states.shape().dims::<4>();
    assert_eq!(views, 1, "mamba tensorized path expects a single view");

    let xz = hidden_states
        .clone()
        .reshape([batch * time, d_model])
        .matmul(in_proj)
        .reshape([batch, time, d_inner * 2]);
    let x = xz
        .clone()
        .slice_dim(2, 0..d_inner)
        .swap_dims(1, 2)
        .reshape([batch, 1, d_inner, time]);
    let z = xz
        .slice_dim(2, d_inner..(d_inner * 2))
        .swap_dims(1, 2)
        .reshape([batch, 1, d_inner, time]);

    let (u, final_conv_state) = super::conv::tensorized_mamba_depthwise_conv(
        x,
        conv_weight,
        conv_bias,
        state.as_ref().map(|s| s.conv.clone()),
    );
    let u_seq = u.clone().swap_dims(2, 3).reshape([batch * time, d_inner]);
    let x_db = u_seq
        .matmul(x_proj)
        .reshape([batch, time, dt_rank + d_state * 2]);

    let dt = activation::softplus(
        x_db.clone()
            .slice_dim(2, 0..dt_rank)
            .reshape([batch * time, dt_rank])
            .matmul(dt_proj_weight)
            .reshape([batch, 1, time, d_inner])
            + dt_proj_bias.reshape([1, 1, 1, d_inner]),
        1.0,
    );
    let b_t = x_db
        .clone()
        .slice_dim(2, dt_rank..(dt_rank + d_state))
        .reshape([batch, 1, time, d_state]);
    let c_t = x_db
        .slice_dim(2, (dt_rank + d_state)..(dt_rank + d_state * 2))
        .reshape([batch, 1, time, d_state]);

    let a = a_log.exp().neg().reshape([1, 1, 1, d_inner, d_state]);
    let d_a = (dt.clone().unsqueeze_dim::<5>(4) * a).exp();
    let d_b = dt.clone().unsqueeze_dim::<5>(4) * b_t.clone().unsqueeze_dim::<5>(3);
    let drive = u.clone().swap_dims(2, 3).unsqueeze_dim::<5>(4) * d_b;

    let prefix_a = d_a.clone().cumprod(2);
    let mut ssm = prefix_a.clone() * drive.div(prefix_a.clone().add_scalar(1.0e-12)).cumsum(2);
    if let Some(initial_ssm) = state
        .as_ref()
        .map(|existing| existing.ssm.clone())
        .filter(|existing| existing.shape().dims::<4>() == [batch, 1, d_inner, d_state])
    {
        ssm = ssm
            + prefix_a
                * initial_ssm
                    .reshape([batch, 1, 1, d_inner, d_state])
                    .repeat_dim(2, time);
    }

    let y = (ssm.clone() * c_t.unsqueeze_dim::<5>(3))
        .sum_dim(4)
        .reshape([batch, 1, time, d_inner])
        + d_skip.reshape([1, 1, 1, d_inner]) * u.swap_dims(2, 3);
    let gated = y * silu(z.swap_dims(2, 3));
    let context = gated
        .reshape([batch * time, d_inner])
        .matmul(out_proj)
        .reshape([batch, 1, time, d_model]);
    let final_ssm_state = ssm
        .slice_dim(2, time - 1..time)
        .reshape([batch, 1, d_inner, d_state]);

    MambaTensorizedOutput {
        context,
        state: MambaTensorizedState {
            conv: final_conv_state,
            ssm: final_ssm_state,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba_autodiff_cube<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    x_proj: Tensor<B, 2>,
    dt_proj_weight: Tensor<B, 2>,
    dt_proj_bias: Tensor<B, 1>,
    a_log: Tensor<B, 2>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<MambaTensorizedState<B>>,
) -> Option<MambaTensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let conv_bias = conv_bias?;
    try_tensorized_mamba_autodiff_wgpu(
        hidden_states.clone(),
        d_inner,
        d_state,
        d_conv,
        dt_rank,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        x_proj.clone(),
        dt_proj_weight.clone(),
        dt_proj_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        out_proj.clone(),
        state.clone(),
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_tensorized_mamba_autodiff_cuda(
                hidden_states,
                d_inner,
                d_state,
                d_conv,
                dt_rank,
                in_proj,
                conv_weight,
                conv_bias,
                x_proj,
                dt_proj_weight,
                dt_proj_bias,
                a_log,
                d_skip,
                out_proj,
                state,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba_autodiff_wgpu<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    x_proj: Tensor<B, 2>,
    dt_proj_weight: Tensor<B, 2>,
    dt_proj_bias: Tensor<B, 1>,
    a_log: Tensor<B, 2>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<MambaTensorizedState<B>>,
) -> Option<MambaTensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let hidden_states_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let conv_weight_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let x_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(x_proj.into_primitive().tensor())?;
    let dt_proj_weight_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_proj_weight.into_primitive().tensor())?;
    let dt_proj_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_proj_bias.into_primitive().tensor())?;
    let a_log_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let out_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;
    let initial_conv_inner = state.as_ref().map(|state| {
        let state_ad: WgpuCubeAutodiffTensor =
            try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
        Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
            state_ad,
        ))
    })??;
    let initial_ssm_inner = state.as_ref().map(|state| {
        let state_ad: WgpuCubeAutodiffTensor =
            try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
        Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
            state_ad,
        ))
    })??;

    let hidden_states_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(conv_bias_ad.clone());
    let x_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(x_proj_ad.clone());
    let dt_proj_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(dt_proj_weight_ad.clone());
    let dt_proj_bias_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(dt_proj_bias_ad.clone());
    let a_log_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let out_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let output = tensorized_mamba_forward_impl(
        BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        d_conv,
        dt_rank,
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            in_proj_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            conv_weight_inner.clone(),
        )),
        Some(BurnTensor::<WgpuCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(conv_bias_inner.clone()),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            x_proj_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            dt_proj_weight_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            dt_proj_bias_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            a_log_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            d_skip_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        Some(MambaTensorizedState {
            conv: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                initial_conv_inner.clone(),
            )),
            ssm: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                initial_ssm_inner.clone(),
            )),
        }),
    );
    let context_inner = output.context.into_primitive().tensor();
    let conv_inner = output.state.conv.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();

    let context_ad = match TensorizedMambaBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            x_proj_ad.node.clone(),
            dt_proj_weight_ad.node.clone(),
            dt_proj_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            MambaTensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                x_proj: x_proj_inner,
                dt_proj_weight: dt_proj_weight_inner,
                dt_proj_bias: dt_proj_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                out_proj: out_proj_inner,
                initial_conv: Some(initial_conv_inner),
                initial_ssm: Some(initial_ssm_inner),
                d_inner,
                d_state,
                d_conv,
                dt_rank,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(MambaTensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: MambaTensorizedState {
            conv: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
                B,
                _,
            >(
                <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(conv_inner),
            )?)),
            ssm: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
                B,
                _,
            >(
                <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(ssm_inner),
            )?)),
        },
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba_autodiff_cuda<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    x_proj: Tensor<B, 2>,
    dt_proj_weight: Tensor<B, 2>,
    dt_proj_bias: Tensor<B, 1>,
    a_log: Tensor<B, 2>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<MambaTensorizedState<B>>,
) -> Option<MambaTensorizedOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
{
    let hidden_states_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let conv_weight_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let x_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(x_proj.into_primitive().tensor())?;
    let dt_proj_weight_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_proj_weight.into_primitive().tensor())?;
    let dt_proj_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_proj_bias.into_primitive().tensor())?;
    let a_log_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let out_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;
    let initial_conv_inner = state.as_ref().map(|state| {
        let state_ad: CudaCubeAutodiffTensor =
            try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
        Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
            state_ad,
        ))
    })??;
    let initial_ssm_inner = state.as_ref().map(|state| {
        let state_ad: CudaCubeAutodiffTensor =
            try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
        Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
            state_ad,
        ))
    })??;

    let hidden_states_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(conv_bias_ad.clone());
    let x_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(x_proj_ad.clone());
    let dt_proj_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(dt_proj_weight_ad.clone());
    let dt_proj_bias_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(dt_proj_bias_ad.clone());
    let a_log_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let out_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let output = tensorized_mamba_forward_impl(
        BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        d_conv,
        dt_rank,
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            in_proj_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            conv_weight_inner.clone(),
        )),
        Some(BurnTensor::<CudaCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(conv_bias_inner.clone()),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            x_proj_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            dt_proj_weight_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            dt_proj_bias_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            a_log_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            d_skip_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        Some(MambaTensorizedState {
            conv: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                initial_conv_inner.clone(),
            )),
            ssm: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                initial_ssm_inner.clone(),
            )),
        }),
    );
    let context_inner = output.context.into_primitive().tensor();
    let conv_inner = output.state.conv.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();

    let context_ad = match TensorizedMambaBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            x_proj_ad.node.clone(),
            dt_proj_weight_ad.node.clone(),
            dt_proj_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            MambaTensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                x_proj: x_proj_inner,
                dt_proj_weight: dt_proj_weight_inner,
                dt_proj_bias: dt_proj_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                out_proj: out_proj_inner,
                initial_conv: Some(initial_conv_inner),
                initial_ssm: Some(initial_ssm_inner),
                d_inner,
                d_state,
                d_conv,
                dt_rank,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(MambaTensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: MambaTensorizedState {
            conv: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
                B,
                _,
            >(
                <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(conv_inner),
            )?)),
            ssm: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
                B,
                _,
            >(
                <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(ssm_inner),
            )?)),
        },
    })
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
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
