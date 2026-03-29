use std::any::{Any, TypeId};
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
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
use burn_fusion::{Fusion, FusionTensor};
use burn_wgpu::{CubeBackend, WgpuRuntime};

use crate::fusion_compat::register_fusion_float_tensor;
use crate::kernels::sequence::mamba::conv::tensorized_mamba_depthwise_conv;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba::conv_runtime::{
    MambaDepthwiseConvCudaForwardOutput, fused_mamba_depthwise_conv_forward_cuda,
};
use crate::kernels::sequence::mamba2::backward::{
    Mamba2TensorizedBackwardState, TensorizedMamba2Backward,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba2::rmsnorm_runtime::{
    Mamba2RmsnormGatedCudaForwardOutput, fused_mamba2_rmsnorm_gated_forward_cuda,
};
use crate::kernels::sequence::mamba2::rmsnorm_runtime::{
    Mamba2RmsnormGatedWgpuForwardOutput, fused_mamba2_rmsnorm_gated_forward_wgpu,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba2::ssd_runtime::fused_mamba2_ssd_forward_cuda;
use crate::kernels::sequence::mamba2::ssd_runtime::fused_mamba2_ssd_forward_wgpu;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
type WgpuFusionAutodiffBackend<BT> = Autodiff<WgpuFusionBackend<BT>>;
type WgpuFusionAutodiffTensor<BT> =
    <WgpuFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaFusionBackend<BT> = Fusion<CubeBackend<CudaRuntime, f32, i32, BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffBackend<BT> = Autodiff<CudaFusionBackend<BT>>;
#[cfg(feature = "cuda")]
type CudaFusionAutodiffTensor<BT> =
    <CudaFusionAutodiffBackend<BT> as BackendTrait>::FloatTensorPrimitive;

#[derive(Debug, Clone)]
pub struct Mamba2TensorizedState<B: BackendTrait> {
    pub conv: Tensor<B, 4>,
    pub ssm: Tensor<B, 4>,
}

#[derive(Debug)]
pub struct Mamba2TensorizedOutput<B: BackendTrait> {
    pub context: Tensor<B, 4>,
    pub state: Mamba2TensorizedState<B>,
}

struct Mamba2TensorizedInternalOutput<B: BackendTrait> {
    output: Mamba2TensorizedOutput<B>,
    ssd_state_history: Option<Tensor<B, 6>>,
    rmsnorm_inv_rms: Option<Tensor<B, 2>>,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaSsdCoreMode {
    RuntimeDefault,
    ForcedEnabled,
    ForcedDisabled,
}

#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaShellCoreMode {
    RuntimeDefault,
    ForcedEnabled,
    ForcedDisabled,
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_ssd_core_mode_enabled(mode: CudaSsdCoreMode) -> bool {
    match mode {
        CudaSsdCoreMode::RuntimeDefault => use_tensorized_mamba2_cuda_fused_ssd_core(),
        CudaSsdCoreMode::ForcedEnabled => true,
        CudaSsdCoreMode::ForcedDisabled => false,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn cuda_shell_core_mode_enabled(mode: CudaShellCoreMode) -> bool {
    match mode {
        CudaShellCoreMode::RuntimeDefault => use_tensorized_mamba2_cuda_fused_shell_core(),
        CudaShellCoreMode::ForcedEnabled => true,
        CudaShellCoreMode::ForcedDisabled => false,
    }
}

pub fn use_tensorized_mamba2_forward_experimental() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA2_TENSORIZED_FORWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_tensorized_mamba2_train_wrapper() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA2_TENSORIZED_TRAIN_WRAPPER")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(feature = "cuda")]
fn use_tensorized_mamba2_cuda_train_wrapper() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA2_CUDA_TENSORIZED_TRAIN_WRAPPER")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn use_tensorized_mamba2_cuda_fused_ssd_core() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA2_CUDA_FUSED_SSD_CORE")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn use_tensorized_mamba2_cuda_fused_shell_core() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA2_CUDA_FUSED_SHELL_CORE")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn log_mamba2_path_selection_once(message: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{message}"));
}

#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
) -> Mamba2TensorizedOutput<B> {
    if use_tensorized_mamba2_train_wrapper() {
        if let Some(output) = try_tensorized_mamba2_autodiff_cube(
            hidden_states.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            norm_eps,
            out_proj.clone(),
            state.clone(),
            CudaSsdCoreMode::RuntimeDefault,
            CudaShellCoreMode::RuntimeDefault,
        ) {
            log_mamba2_path_selection_once(
                "mamba2 tensorized path: using custom analytic backward wrapper with fused CUDA SSD and shell cores",
            );
            return output;
        }

        log_mamba2_path_selection_once(
            "mamba2 tensorized path: custom analytic backward wrapper unavailable, falling back to direct tensorized graph",
        );
    } else {
        log_mamba2_path_selection_once(
            "mamba2 tensorized path: using direct tensorized graph (custom analytic backward wrapper disabled)",
        );
    }

    tensorized_mamba2_forward_impl(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward_direct_graph<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
) -> Mamba2TensorizedOutput<B> {
    tensorized_mamba2_forward_impl(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward_direct_graph_with_ssd_core_mode<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
) -> Mamba2TensorizedOutput<B> {
    tensorized_mamba2_forward_impl_with_cuda_ssd_mode(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        cuda_ssd_core_mode,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward_custom_backward<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    tensorized_mamba2_forward_custom_backward_with_cuda_modes(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        CudaSsdCoreMode::RuntimeDefault,
        CudaShellCoreMode::RuntimeDefault,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward_custom_backward_with_cuda_modes<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_tensorized_mamba2_autodiff_cube(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
    )
}

#[doc(hidden)]
#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba2_forward_custom_backward_with_cuda_ssd_mode<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    tensorized_mamba2_forward_custom_backward_with_cuda_modes(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        cuda_ssd_core_mode,
        CudaShellCoreMode::RuntimeDefault,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorized_mamba2_forward_impl<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
) -> Mamba2TensorizedOutput<B> {
    tensorized_mamba2_forward_impl_with_cuda_modes(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        CudaSsdCoreMode::RuntimeDefault,
        CudaShellCoreMode::RuntimeDefault,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorized_mamba2_forward_impl_with_cuda_ssd_mode<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
) -> Mamba2TensorizedOutput<B> {
    tensorized_mamba2_forward_impl_with_cuda_modes(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        cuda_ssd_core_mode,
        CudaShellCoreMode::RuntimeDefault,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorized_mamba2_forward_impl_with_cuda_modes<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Mamba2TensorizedOutput<B> {
    tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode(
        hidden_states,
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj,
        conv_weight,
        conv_bias,
        dt_bias,
        a_log,
        d_skip,
        norm_weight,
        norm_eps,
        out_proj,
        state,
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
        false,
    )
    .output
}

#[allow(clippy::too_many_arguments)]
fn tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
    capture_cuda_ssd_state_history: bool,
) -> Mamba2TensorizedInternalOutput<B> {
    let [batch, views, time, d_model] = hidden_states.shape().dims::<4>();
    let device = hidden_states.device();
    assert_eq!(views, 1, "mamba2 tensorized path expects a single view");
    assert_eq!(
        d_inner % headdim,
        0,
        "mamba2 tensorized path requires d_inner divisible by headdim"
    );
    assert!(ngroups > 0, "mamba2 tensorized path requires ngroups > 0");
    let nheads = d_inner / headdim;
    let heads_per_group = nheads / ngroups;
    assert_eq!(
        nheads % ngroups,
        0,
        "mamba2 tensorized path requires nheads divisible by ngroups"
    );
    let conv_dim = d_inner + 2 * ngroups * d_state;
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + nheads;
    assert_eq!(
        conv_weight.shape().dims::<2>(),
        [conv_dim, d_conv],
        "mamba2 tensorized path requires conv_weight=[conv_dim, d_conv]"
    );
    if let Some(conv_bias) = conv_bias.as_ref() {
        assert_eq!(
            conv_bias.shape().dims::<1>(),
            [conv_dim],
            "mamba2 tensorized path requires conv_bias=[conv_dim]"
        );
    }
    assert_eq!(
        in_proj.shape().dims::<2>(),
        [d_model, in_proj_dim],
        "mamba2 tensorized path requires in_proj=[d_model, in_proj_dim]"
    );
    assert_eq!(
        dt_bias.shape().dims::<1>(),
        [nheads],
        "mamba2 tensorized path requires dt_bias=[nheads]"
    );
    assert_eq!(
        a_log.shape().dims::<1>(),
        [nheads],
        "mamba2 tensorized path requires a_log=[nheads]"
    );
    assert_eq!(
        d_skip.shape().dims::<1>(),
        [nheads],
        "mamba2 tensorized path requires d_skip=[nheads]"
    );
    assert_eq!(
        norm_weight.shape().dims::<1>(),
        [d_inner],
        "mamba2 tensorized path requires norm_weight=[d_inner]"
    );
    assert_eq!(
        out_proj.shape().dims::<2>(),
        [d_inner, d_model],
        "mamba2 tensorized path requires out_proj=[d_inner, d_model]"
    );
    let initial_ssm = state
        .as_ref()
        .filter(|existing| existing.ssm.shape().dims::<4>() == [batch, nheads, headdim, d_state])
        .map(|existing| existing.ssm.clone());

    let zxbcdt = hidden_states
        .clone()
        .reshape([batch * time, d_model])
        .matmul(in_proj)
        .reshape([batch, time, in_proj_dim]);
    let z = zxbcdt
        .clone()
        .slice_dim(2, 0..d_inner)
        .reshape([batch, time, d_inner]);
    let xbc = zxbcdt
        .clone()
        .slice_dim(2, d_inner..(d_inner + conv_dim))
        .reshape([batch, time, conv_dim]);
    let dt = zxbcdt
        .slice_dim(2, (d_inner + conv_dim)..in_proj_dim)
        .reshape([batch, time, nheads]);

    let xbc_input = xbc.swap_dims(1, 2).reshape([batch, 1, conv_dim, time]);
    let (xbc_conv, final_conv_state) = if capture_cuda_ssd_state_history {
        if let Some(output) = try_accelerated_depthwise_conv_forward_core(
            xbc_input.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            state.as_ref().map(|existing| existing.conv.clone()),
            cuda_shell_core_mode,
        ) {
            (output.activated, output.next_state)
        } else {
            tensorized_mamba_depthwise_conv(
                xbc_input,
                conv_weight,
                conv_bias,
                state.map(|existing| existing.conv),
            )
        }
    } else {
        tensorized_mamba_depthwise_conv(
            xbc_input,
            conv_weight,
            conv_bias,
            state.map(|existing| existing.conv),
        )
    };
    let xbc_conv = xbc_conv.swap_dims(2, 3).reshape([batch, time, conv_dim]);

    let x = xbc_conv
        .clone()
        .slice_dim(2, 0..d_inner)
        .reshape([batch, time, nheads, headdim]);
    let b_group = xbc_conv
        .clone()
        .slice_dim(2, d_inner..(d_inner + ngroups * d_state))
        .reshape([batch, time, ngroups, d_state]);
    let c_group = xbc_conv
        .slice_dim(2, (d_inner + ngroups * d_state)..conv_dim)
        .reshape([batch, time, ngroups, d_state]);
    let dt = activation::softplus(dt + dt_bias.reshape([1, 1, nheads]), 1.0);
    let x_grouped = x.reshape([batch, time, ngroups, heads_per_group, headdim]);
    let dt_grouped = dt.reshape([batch, time, ngroups, heads_per_group]);
    let (y_grouped, final_ssm_state, ssd_state_history) = if let Some(output) =
        try_accelerated_ssd_forward_core::<B>(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            d_skip.clone(),
            initial_ssm
                .clone()
                .map(|state| state.reshape([batch, ngroups, heads_per_group, headdim, d_state])),
            cuda_ssd_core_mode,
            capture_cuda_ssd_state_history,
        ) {
        (
            output.y_grouped,
            output.final_ssm.reshape([batch, nheads, headdim, d_state]),
            output.ssd_state_history,
        )
    } else {
        let a = a_log
            .exp()
            .neg()
            .reshape([1, ngroups, heads_per_group, 1, 1]);
        let d_skip = d_skip.reshape([1, ngroups, heads_per_group, 1]);
        let mut ssm_state = initial_ssm
            .map(|state| state.reshape([batch, ngroups, heads_per_group, headdim, d_state]))
            .unwrap_or_else(|| {
                Tensor::<B, 5>::zeros([batch, ngroups, heads_per_group, headdim, d_state], &device)
            });
        let mut outputs = Vec::with_capacity(time);
        let mut state_history = capture_cuda_ssd_state_history.then(|| Vec::with_capacity(time));

        for step in 0..time {
            let x_t = x_grouped.clone().slice_dim(1, step..step + 1).reshape([
                batch,
                ngroups,
                heads_per_group,
                headdim,
            ]);
            let b_t = b_group
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, ngroups, d_state]);
            let c_t = c_group
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, ngroups, d_state]);
            let dt_t = dt_grouped.clone().slice_dim(1, step..step + 1).reshape([
                batch,
                ngroups,
                heads_per_group,
            ]);

            let decay = (dt_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, 1, 1])
                * a.clone())
            .exp();
            let input_term = dt_t.reshape([batch, ngroups, heads_per_group, 1, 1])
                * b_t.reshape([batch, ngroups, 1, 1, d_state])
                * x_t
                    .clone()
                    .reshape([batch, ngroups, heads_per_group, headdim, 1]);
            ssm_state = ssm_state * decay + input_term;

            if let Some(history) = state_history.as_mut() {
                history.push(ssm_state.clone().reshape([
                    batch,
                    1,
                    ngroups,
                    heads_per_group,
                    headdim,
                    d_state,
                ]));
            }

            let y_t = (ssm_state.clone() * c_t.reshape([batch, ngroups, 1, 1, d_state]))
                .sum_dim(4)
                .reshape([batch, ngroups, heads_per_group, headdim])
                + d_skip.clone() * x_t;
            outputs.push(y_t.reshape([batch, 1, ngroups, heads_per_group, headdim]));
        }

        (
            Tensor::cat(outputs, 1),
            ssm_state.reshape([batch, nheads, headdim, d_state]),
            state_history.map(|history| Tensor::cat(history, 1)),
        )
    };
    let y_flat = y_grouped.reshape([batch, time, d_inner]);
    let (gated, rmsnorm_inv_rms) = if capture_cuda_ssd_state_history {
        if let Some(output) = try_accelerated_rmsnorm_gated_forward_core(
            y_flat.clone(),
            z.clone(),
            norm_weight.clone(),
            norm_eps,
            cuda_shell_core_mode,
        ) {
            (output.gated, Some(output.inv_rms))
        } else {
            let inv_rms = y_flat
                .clone()
                .powf_scalar(2.0)
                .mean_dim(2)
                .add_scalar(norm_eps)
                .sqrt()
                .recip()
                .reshape([batch, time]);
            (
                rmsnorm_gated_from_inv_rms(y_flat, z, norm_weight, inv_rms.clone()),
                Some(inv_rms),
            )
        }
    } else {
        (rmsnorm_gated(y_flat, z, norm_weight, norm_eps), None)
    };
    let context = gated
        .reshape([batch * time, d_inner])
        .matmul(out_proj)
        .reshape([batch, 1, time, d_model]);

    Mamba2TensorizedInternalOutput {
        output: Mamba2TensorizedOutput {
            context,
            state: Mamba2TensorizedState {
                conv: final_conv_state,
                ssm: final_ssm_state,
            },
        },
        ssd_state_history,
        rmsnorm_inv_rms,
    }
}

struct AcceleratedDepthwiseConvForwardOutput<B: BackendTrait> {
    activated: Tensor<B, 4>,
    next_state: Tensor<B, 4>,
}

#[cfg(feature = "cuda")]
fn try_accelerated_depthwise_conv_forward_core<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    state: Option<Tensor<B, 4>>,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<AcceleratedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_shell_core_mode_enabled(cuda_shell_core_mode) {
        return None;
    }
    let conv_bias = conv_bias?;
    try_accelerated_depthwise_conv_forward_core_cuda_direct(
        x.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        state.clone(),
    )
    .or_else(|| {
        try_accelerated_depthwise_conv_forward_core_cuda_fusion::<B, u8>(
            x.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            state.clone(),
        )
    })
    .or_else(|| {
        try_accelerated_depthwise_conv_forward_core_cuda_fusion::<B, u32>(
            x,
            conv_weight,
            conv_bias,
            state,
        )
    })
}

#[cfg(not(feature = "cuda"))]
fn try_accelerated_depthwise_conv_forward_core<B: BackendTrait>(
    _x: Tensor<B, 4>,
    _conv_weight: Tensor<B, 2>,
    _conv_bias: Option<Tensor<B, 1>>,
    _state: Option<Tensor<B, 4>>,
    _cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<AcceleratedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
fn try_accelerated_depthwise_conv_forward_core_cuda_direct<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    state: Option<Tensor<B, 4>>,
) -> Option<AcceleratedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, views, channels, _time] = x.shape().dims::<4>();
    let d_conv = conv_weight.shape().dims::<2>()[1];
    let x_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let conv_weight_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let state_raw: CubeTensor<CudaRuntime> = match state {
        Some(state) => try_cast_primitive::<B, _>(state.into_primitive().tensor())?,
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };
    let output: MambaDepthwiseConvCudaForwardOutput =
        fused_mamba_depthwise_conv_forward_cuda(x_raw, conv_weight_raw, conv_bias_raw, state_raw);
    Some(AcceleratedDepthwiseConvForwardOutput {
        activated: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.activated
        )?)),
        next_state: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.next_state,
        )?)),
    })
}

#[cfg(feature = "cuda")]
fn try_accelerated_depthwise_conv_forward_core_cuda_fusion<
    B: BackendTrait,
    BT: BoolElement + 'static,
>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    state: Option<Tensor<B, 4>>,
) -> Option<AcceleratedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime, BT>>>()
    {
        return None;
    }

    let x_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let client = x_fusion.client.clone();
    let conv_weight_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let [batch, views, channels, _time] = client
        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(x_fusion.clone())
        .meta
        .shape
        .dims::<4>();
    let d_conv = client
        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_weight_fusion.clone())
        .meta
        .shape
        .dims::<2>()[1];

    let x_raw = client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(x_fusion);
    let conv_weight_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_weight_fusion);
    let conv_bias_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_bias_fusion);
    let state_raw = match state {
        Some(state) => {
            let state_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(state_fusion)
        }
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };

    let output =
        fused_mamba_depthwise_conv_forward_cuda(x_raw, conv_weight_raw, conv_bias_raw, state_raw);
    Some(AcceleratedDepthwiseConvForwardOutput {
        activated: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            register_fusion_float_tensor(&client, output.activated),
        )?)),
        next_state: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            register_fusion_float_tensor(&client, output.next_state),
        )?)),
    })
}

struct AcceleratedRmsnormGatedForwardOutput<B: BackendTrait> {
    gated: Tensor<B, 3>,
    inv_rms: Tensor<B, 2>,
}

fn try_accelerated_rmsnorm_gated_forward_core_wgpu<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Option<AcceleratedRmsnormGatedForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let y_raw: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(y.into_primitive().tensor())?;
    let z_raw: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(z.into_primitive().tensor())?;
    let weight_raw: CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(weight.into_primitive().tensor())?;
    let output: Mamba2RmsnormGatedWgpuForwardOutput =
        fused_mamba2_rmsnorm_gated_forward_wgpu(y_raw, z_raw, weight_raw, eps);
    Some(AcceleratedRmsnormGatedForwardOutput {
        gated: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.gated)?,
        )),
        inv_rms: BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.inv_rms
        )?)),
    })
}

fn try_accelerated_rmsnorm_gated_forward_core<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<AcceleratedRmsnormGatedForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_accelerated_rmsnorm_gated_forward_core_wgpu(y.clone(), z.clone(), weight.clone(), eps)
        .or_else(|| {
            #[cfg(feature = "cuda")]
            {
                try_accelerated_rmsnorm_gated_forward_core_cuda(
                    y,
                    z,
                    weight,
                    eps,
                    cuda_shell_core_mode,
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = cuda_shell_core_mode;
                None
            }
        })
}

#[cfg(feature = "cuda")]
fn try_accelerated_rmsnorm_gated_forward_core_cuda<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<AcceleratedRmsnormGatedForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_shell_core_mode_enabled(cuda_shell_core_mode) {
        return None;
    }
    try_accelerated_rmsnorm_gated_forward_core_cuda_direct(
        y.clone(),
        z.clone(),
        weight.clone(),
        eps,
    )
    .or_else(|| {
        try_accelerated_rmsnorm_gated_forward_core_cuda_fusion::<B, u8>(
            y.clone(),
            z.clone(),
            weight.clone(),
            eps,
        )
    })
    .or_else(|| try_accelerated_rmsnorm_gated_forward_core_cuda_fusion::<B, u32>(y, z, weight, eps))
}

#[cfg(feature = "cuda")]
fn try_accelerated_rmsnorm_gated_forward_core_cuda_direct<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Option<AcceleratedRmsnormGatedForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let y_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(y.into_primitive().tensor())?;
    let z_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(z.into_primitive().tensor())?;
    let weight_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(weight.into_primitive().tensor())?;
    let output: Mamba2RmsnormGatedCudaForwardOutput =
        fused_mamba2_rmsnorm_gated_forward_cuda(y_raw, z_raw, weight_raw, eps);
    Some(AcceleratedRmsnormGatedForwardOutput {
        gated: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.gated)?,
        )),
        inv_rms: BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.inv_rms
        )?)),
    })
}

#[cfg(feature = "cuda")]
fn try_accelerated_rmsnorm_gated_forward_core_cuda_fusion<
    B: BackendTrait,
    BT: BoolElement + 'static,
>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Option<AcceleratedRmsnormGatedForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime, BT>>>()
    {
        return None;
    }

    let y_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(y.into_primitive().tensor())?;
    let client = y_fusion.client.clone();
    let z_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(z.into_primitive().tensor())?;
    let weight_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime, BT>> =
        try_cast_primitive::<B, _>(weight.into_primitive().tensor())?;

    let y_raw = client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(y_fusion);
    let z_raw = client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(z_fusion);
    let weight_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(weight_fusion);
    let output = fused_mamba2_rmsnorm_gated_forward_cuda(y_raw, z_raw, weight_raw, eps);
    Some(AcceleratedRmsnormGatedForwardOutput {
        gated: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.gated))?,
        )),
        inv_rms: BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            register_fusion_float_tensor(&client, output.inv_rms),
        )?)),
    })
}

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba2_autodiff_cube<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let conv_bias = conv_bias?;
    try_tensorized_mamba2_autodiff_wgpu_fusion::<B, u32>(
        hidden_states.clone(),
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        in_proj.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        dt_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        norm_weight.clone(),
        norm_eps,
        out_proj.clone(),
        state.clone(),
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
    )
    .or_else(|| {
        try_tensorized_mamba2_autodiff_wgpu_fusion::<B, u8>(
            hidden_states.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            norm_eps,
            out_proj.clone(),
            state.clone(),
            cuda_ssd_core_mode,
            cuda_shell_core_mode,
        )
    })
    .or_else(|| {
        try_tensorized_mamba2_autodiff_wgpu(
            hidden_states.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            norm_eps,
            out_proj.clone(),
            state.clone(),
            cuda_ssd_core_mode,
            cuda_shell_core_mode,
        )
    })
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            if use_tensorized_mamba2_cuda_train_wrapper() {
                try_tensorized_mamba2_autodiff_cuda_fusion::<B, u32>(
                    hidden_states.clone(),
                    d_inner,
                    d_state,
                    d_conv,
                    headdim,
                    ngroups,
                    in_proj.clone(),
                    conv_weight.clone(),
                    conv_bias.clone(),
                    dt_bias.clone(),
                    a_log.clone(),
                    d_skip.clone(),
                    norm_weight.clone(),
                    norm_eps,
                    out_proj.clone(),
                    state.clone(),
                    cuda_ssd_core_mode,
                    cuda_shell_core_mode,
                )
                .or_else(|| {
                    try_tensorized_mamba2_autodiff_cuda_fusion::<B, u8>(
                        hidden_states.clone(),
                        d_inner,
                        d_state,
                        d_conv,
                        headdim,
                        ngroups,
                        in_proj.clone(),
                        conv_weight.clone(),
                        conv_bias.clone(),
                        dt_bias.clone(),
                        a_log.clone(),
                        d_skip.clone(),
                        norm_weight.clone(),
                        norm_eps,
                        out_proj.clone(),
                        state.clone(),
                        cuda_ssd_core_mode,
                        cuda_shell_core_mode,
                    )
                })
                .or_else(|| {
                    try_tensorized_mamba2_autodiff_cuda(
                        hidden_states,
                        d_inner,
                        d_state,
                        d_conv,
                        headdim,
                        ngroups,
                        in_proj,
                        conv_weight,
                        conv_bias,
                        dt_bias,
                        a_log,
                        d_skip,
                        norm_weight,
                        norm_eps,
                        out_proj,
                        state,
                        cuda_ssd_core_mode,
                        cuda_shell_core_mode,
                    )
                })
            } else {
                None
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba2_autodiff_wgpu<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
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
    let dt_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let a_log_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let norm_weight_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(norm_weight.into_primitive().tensor())?;
    let out_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;
    let initial_conv_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };

    let hidden_states_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(conv_bias_ad.clone());
    let dt_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(dt_bias_ad.clone());
    let a_log_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let norm_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(norm_weight_ad.clone());
    let out_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let internal_output = tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode(
        BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            in_proj_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            conv_weight_inner.clone(),
        )),
        Some(BurnTensor::<WgpuCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(conv_bias_inner.clone()),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            dt_bias_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            a_log_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            d_skip_inner.clone(),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            norm_weight_inner.clone(),
        )),
        norm_eps,
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        match (initial_conv_inner.clone(), initial_ssm_inner.clone()) {
            (Some(conv), Some(ssm)) => Some(Mamba2TensorizedState {
                conv: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    conv,
                )),
                ssm: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(ssm)),
            }),
            _ => None,
        },
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
        true,
    );
    let Mamba2TensorizedInternalOutput {
        output,
        ssd_state_history,
        rmsnorm_inv_rms,
    } = internal_output;
    let context_inner = output.context.into_primitive().tensor();
    let conv_inner = output.state.conv.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();
    let ssd_state_history_inner =
        ssd_state_history.map(|history| history.into_primitive().tensor());
    let rmsnorm_inv_rms_inner = rmsnorm_inv_rms.map(|inv_rms| inv_rms.into_primitive().tensor());

    let context_ad = match TensorizedMamba2Backward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            dt_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            norm_weight_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba2TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                dt_bias: dt_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                norm_weight: norm_weight_inner,
                out_proj: out_proj_inner,
                initial_conv: initial_conv_inner,
                initial_ssm: initial_ssm_inner,
                rmsnorm_inv_rms: rmsnorm_inv_rms_inner,
                d_inner,
                d_state,
                d_conv,
                headdim,
                ngroups,
                norm_eps,
                ssd_state_history: ssd_state_history_inner,
                cuda_ssd_core_mode,
                cuda_shell_core_mode,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(Mamba2TensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: Mamba2TensorizedState {
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

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba2_autodiff_wgpu_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_autodiff_fusion_type::<B, BT, WgpuRuntime>() {
        return None;
    }

    let hidden_states_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let conv_weight_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let dt_bias_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let a_log_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let norm_weight_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(norm_weight.into_primitive().tensor())?;
    let out_proj_ad: WgpuFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;

    let hidden_states_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(hidden_states_ad.clone());
    let fusion_client = hidden_states_inner.client.clone();
    let in_proj_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(conv_bias_ad.clone());
    let dt_bias_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(dt_bias_ad.clone());
    let a_log_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(d_skip_ad.clone());
    let norm_weight_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(norm_weight_ad.clone());
    let out_proj_inner =
        <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(out_proj_ad.clone());
    let initial_conv_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: WgpuFusionAutodiffTensor<BT> =
                try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
            Some(<WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: WgpuFusionAutodiffTensor<BT> =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };

    let internal_output = tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode(
        BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                hidden_states_inner.clone(),
            ),
        )),
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                in_proj_inner.clone(),
            ),
        )),
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                conv_weight_inner.clone(),
            ),
        )),
        Some(BurnTensor::<WgpuCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(
                fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                    conv_bias_inner.clone(),
                ),
            ),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                dt_bias_inner.clone(),
            ),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                a_log_inner.clone(),
            ),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                d_skip_inner.clone(),
            ),
        )),
        BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                norm_weight_inner.clone(),
            ),
        )),
        norm_eps,
        BurnTensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(
                out_proj_inner.clone(),
            ),
        )),
        match (initial_conv_inner.clone(), initial_ssm_inner.clone()) {
            (Some(conv), Some(ssm)) => Some(Mamba2TensorizedState {
                conv: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    fusion_client
                        .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(conv),
                )),
                ssm: BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    fusion_client
                        .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(ssm),
                )),
            }),
            _ => None,
        },
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
        true,
    );
    let Mamba2TensorizedInternalOutput {
        output,
        ssd_state_history,
        rmsnorm_inv_rms,
    } = internal_output;
    let context_fusion =
        register_fusion_float_tensor(&fusion_client, output.context.into_primitive().tensor());
    let conv_fusion =
        register_fusion_float_tensor(&fusion_client, output.state.conv.into_primitive().tensor());
    let ssm_fusion =
        register_fusion_float_tensor(&fusion_client, output.state.ssm.into_primitive().tensor());
    let ssd_state_history_fusion = ssd_state_history.map(|history| {
        register_fusion_float_tensor(&fusion_client, history.into_primitive().tensor())
    });
    let rmsnorm_inv_rms_fusion = rmsnorm_inv_rms.map(|inv_rms| {
        register_fusion_float_tensor(&fusion_client, inv_rms.into_primitive().tensor())
    });

    let context_ad = match TensorizedMamba2Backward::<WgpuFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            dt_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            norm_weight_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba2TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                dt_bias: dt_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                norm_weight: norm_weight_inner,
                out_proj: out_proj_inner,
                initial_conv: initial_conv_inner,
                initial_ssm: initial_ssm_inner,
                rmsnorm_inv_rms: rmsnorm_inv_rms_fusion,
                d_inner,
                d_state,
                d_conv,
                headdim,
                ngroups,
                norm_eps,
                ssd_state_history: ssd_state_history_fusion,
                cuda_ssd_core_mode,
                cuda_shell_core_mode,
            },
            context_fusion,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_fusion),
    };

    Some(Mamba2TensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: Mamba2TensorizedState {
            conv: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
                wrap_fusion_autodiff_inner::<B, BT, WgpuRuntime>(conv_fusion)?,
            )),
            ssm: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
                wrap_fusion_autodiff_inner::<B, BT, WgpuRuntime>(ssm_fusion)?,
            )),
        },
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba2_autodiff_cuda_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_autodiff_fusion_type::<B, BT, CudaRuntime>() {
        return None;
    }

    let hidden_states_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let conv_weight_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let dt_bias_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let a_log_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let norm_weight_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(norm_weight.into_primitive().tensor())?;
    let out_proj_ad: CudaFusionAutodiffTensor<BT> =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;

    let hidden_states_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(hidden_states_ad.clone());
    let fusion_client = hidden_states_inner.client.clone();
    let in_proj_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(conv_bias_ad.clone());
    let dt_bias_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(dt_bias_ad.clone());
    let a_log_inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(d_skip_ad.clone());
    let norm_weight_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(norm_weight_ad.clone());
    let out_proj_inner =
        <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(out_proj_ad.clone());
    let initial_conv_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: CudaFusionAutodiffTensor<BT> =
                try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
            Some(<CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: CudaFusionAutodiffTensor<BT> =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };

    let internal_output = tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode(
        BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                hidden_states_inner.clone(),
            ),
        )),
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                in_proj_inner.clone(),
            ),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                conv_weight_inner.clone(),
            ),
        )),
        Some(BurnTensor::<CudaCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(
                fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                    conv_bias_inner.clone(),
                ),
            ),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                dt_bias_inner.clone(),
            ),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                a_log_inner.clone(),
            ),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                d_skip_inner.clone(),
            ),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                norm_weight_inner.clone(),
            ),
        )),
        norm_eps,
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            fusion_client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(
                out_proj_inner.clone(),
            ),
        )),
        match (initial_conv_inner.clone(), initial_ssm_inner.clone()) {
            (Some(conv), Some(ssm)) => Some(Mamba2TensorizedState {
                conv: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    fusion_client
                        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv),
                )),
                ssm: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    fusion_client
                        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(ssm),
                )),
            }),
            _ => None,
        },
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
        true,
    );
    let Mamba2TensorizedInternalOutput {
        output,
        ssd_state_history,
        rmsnorm_inv_rms,
    } = internal_output;
    let context_fusion =
        register_fusion_float_tensor(&fusion_client, output.context.into_primitive().tensor());
    let conv_fusion =
        register_fusion_float_tensor(&fusion_client, output.state.conv.into_primitive().tensor());
    let ssm_fusion =
        register_fusion_float_tensor(&fusion_client, output.state.ssm.into_primitive().tensor());
    let ssd_state_history_fusion = ssd_state_history.map(|history| {
        register_fusion_float_tensor(&fusion_client, history.into_primitive().tensor())
    });
    let rmsnorm_inv_rms_fusion = rmsnorm_inv_rms.map(|inv_rms| {
        register_fusion_float_tensor(&fusion_client, inv_rms.into_primitive().tensor())
    });

    let context_ad = match TensorizedMamba2Backward::<CudaFusionBackend<BT>>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            dt_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            norm_weight_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba2TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                dt_bias: dt_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                norm_weight: norm_weight_inner,
                out_proj: out_proj_inner,
                initial_conv: initial_conv_inner,
                initial_ssm: initial_ssm_inner,
                rmsnorm_inv_rms: rmsnorm_inv_rms_fusion,
                d_inner,
                d_state,
                d_conv,
                headdim,
                ngroups,
                norm_eps,
                ssd_state_history: ssd_state_history_fusion,
                cuda_ssd_core_mode,
                cuda_shell_core_mode,
            },
            context_fusion,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_fusion),
    };

    Some(Mamba2TensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: Mamba2TensorizedState {
            conv: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
                wrap_fusion_autodiff_inner::<B, BT, CudaRuntime>(conv_fusion)?,
            )),
            ssm: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
                wrap_fusion_autodiff_inner::<B, BT, CudaRuntime>(ssm_fusion)?,
            )),
        },
    })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba2_autodiff_cuda<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    in_proj: Tensor<B, 2>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    dt_bias: Tensor<B, 1>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    norm_weight: Tensor<B, 1>,
    norm_eps: f32,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba2TensorizedState<B>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<Mamba2TensorizedOutput<B>>
where
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
    let dt_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let a_log_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let norm_weight_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(norm_weight.into_primitive().tensor())?;
    let out_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;
    let initial_conv_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.conv.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };
    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let state_ad: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(
                state_ad,
            ))
        }
        None => None,
    };

    let hidden_states_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let conv_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(conv_weight_ad.clone());
    let conv_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(conv_bias_ad.clone());
    let dt_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(dt_bias_ad.clone());
    let a_log_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(a_log_ad.clone());
    let d_skip_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let norm_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(norm_weight_ad.clone());
    let out_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let internal_output = tensorized_mamba2_forward_impl_internal_with_cuda_ssd_mode(
        BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        d_conv,
        headdim,
        ngroups,
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            in_proj_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            conv_weight_inner.clone(),
        )),
        Some(BurnTensor::<CudaCubeBackend, 1>::from_primitive(
            TensorPrimitive::Float(conv_bias_inner.clone()),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            dt_bias_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            a_log_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            d_skip_inner.clone(),
        )),
        BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            norm_weight_inner.clone(),
        )),
        norm_eps,
        BurnTensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        match (initial_conv_inner.clone(), initial_ssm_inner.clone()) {
            (Some(conv), Some(ssm)) => Some(Mamba2TensorizedState {
                conv: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
                    conv,
                )),
                ssm: BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(ssm)),
            }),
            _ => None,
        },
        cuda_ssd_core_mode,
        cuda_shell_core_mode,
        true,
    );
    let Mamba2TensorizedInternalOutput {
        output,
        ssd_state_history,
        rmsnorm_inv_rms,
    } = internal_output;
    let context_inner = output.context.into_primitive().tensor();
    let conv_inner = output.state.conv.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();
    let ssd_state_history_inner =
        ssd_state_history.map(|history| history.into_primitive().tensor());
    let rmsnorm_inv_rms_inner = rmsnorm_inv_rms.map(|inv_rms| inv_rms.into_primitive().tensor());

    let context_ad = match TensorizedMamba2Backward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            conv_weight_ad.node.clone(),
            conv_bias_ad.node.clone(),
            dt_bias_ad.node.clone(),
            a_log_ad.node.clone(),
            d_skip_ad.node.clone(),
            norm_weight_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba2TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                conv_weight: conv_weight_inner,
                conv_bias: conv_bias_inner,
                dt_bias: dt_bias_inner,
                a_log: a_log_inner,
                d_skip: d_skip_inner,
                norm_weight: norm_weight_inner,
                out_proj: out_proj_inner,
                initial_conv: initial_conv_inner,
                initial_ssm: initial_ssm_inner,
                rmsnorm_inv_rms: rmsnorm_inv_rms_inner,
                d_inner,
                d_state,
                d_conv,
                headdim,
                ngroups,
                norm_eps,
                ssd_state_history: ssd_state_history_inner,
                cuda_ssd_core_mode,
                cuda_shell_core_mode,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(Mamba2TensorizedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        state: Mamba2TensorizedState {
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

fn rmsnorm_gated<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let inv_rms = y
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt()
        .recip()
        .reshape([y.shape().dims::<3>()[0], y.shape().dims::<3>()[1]]);
    rmsnorm_gated_from_inv_rms(y, z, weight, inv_rms)
}

fn rmsnorm_gated_from_inv_rms<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    inv_rms: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch, time, _width] = y.shape().dims::<3>();
    let width = weight.shape().dims::<1>()[0];
    (y * inv_rms.reshape([batch, time, 1])) * weight.reshape([1, 1, width]) * silu(z)
}

struct AcceleratedSsdForwardOutput<B: BackendTrait> {
    y_grouped: Tensor<B, 5>,
    final_ssm: Tensor<B, 5>,
    ssd_state_history: Option<Tensor<B, 6>>,
}

fn try_accelerated_ssd_forward_core<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    capture_cuda_ssd_state_history: bool,
) -> Option<AcceleratedSsdForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if cuda_ssd_core_mode != CudaSsdCoreMode::ForcedDisabled {
        if let Some(output) = try_wgpu_fused_ssd_forward_core(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            d_skip.clone(),
            initial_ssm.clone(),
            capture_cuda_ssd_state_history,
        ) {
            return Some(output);
        }
    }

    #[cfg(feature = "cuda")]
    {
        try_cuda_fused_ssd_forward_core(
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            initial_ssm,
            cuda_ssd_core_mode,
            capture_cuda_ssd_state_history,
        )
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            initial_ssm,
            capture_cuda_ssd_state_history,
        );
        None
    }
}

fn try_wgpu_fused_ssd_forward_core<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    capture_ssd_state_history: bool,
) -> Option<AcceleratedSsdForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let x_grouped: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(x_grouped.into_primitive().tensor())?;
    let b_group: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(b_group.into_primitive().tensor())?;
    let c_group: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(c_group.into_primitive().tensor())?;
    let dt_grouped: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(dt_grouped.into_primitive().tensor())?;
    let a_log: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip: burn_cubecl::tensor::CubeTensor<WgpuRuntime> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let initial_ssm = match initial_ssm {
        Some(state) => Some(try_cast_primitive::<B, _>(state.into_primitive().tensor())?),
        None => None,
    };
    let output = fused_mamba2_ssd_forward_wgpu(
        x_grouped,
        b_group,
        c_group,
        dt_grouped,
        a_log,
        d_skip,
        initial_ssm,
        capture_ssd_state_history,
    );
    let ssd_state_history = match output.state_history {
        Some(history) => Some(BurnTensor::<B, 6>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(history)?,
        ))),
        None => None,
    };
    Some(AcceleratedSsdForwardOutput {
        y_grouped: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.y_grouped
        )?)),
        final_ssm: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.final_ssm
        )?)),
        ssd_state_history,
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_ssd_forward_core<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
    capture_cuda_ssd_state_history: bool,
) -> Option<AcceleratedSsdForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_ssd_core_mode_enabled(cuda_ssd_core_mode) {
        return None;
    }
    let x_grouped: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(x_grouped.into_primitive().tensor())?;
    let b_group: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(b_group.into_primitive().tensor())?;
    let c_group: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(c_group.into_primitive().tensor())?;
    let dt_grouped: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(dt_grouped.into_primitive().tensor())?;
    let a_log: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let initial_ssm = match initial_ssm {
        Some(state) => Some(try_cast_primitive::<B, _>(state.into_primitive().tensor())?),
        None => None,
    };
    let output = fused_mamba2_ssd_forward_cuda(
        x_grouped,
        b_group,
        c_group,
        dt_grouped,
        a_log,
        d_skip,
        initial_ssm,
        capture_cuda_ssd_state_history,
    );
    let ssd_state_history = match output.state_history {
        Some(history) => Some(BurnTensor::<B, 6>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(history)?,
        ))),
        None => None,
    };
    Some(AcceleratedSsdForwardOutput {
        y_grouped: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.y_grouped
        )?)),
        final_ssm: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.final_ssm
        )?)),
        ssd_state_history,
    })
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[cfg(not(feature = "cuda"))]
fn try_cuda_fused_ssd_forward_core<B: BackendTrait>(
    _x_grouped: Tensor<B, 5>,
    _b_group: Tensor<B, 4>,
    _c_group: Tensor<B, 4>,
    _dt_grouped: Tensor<B, 4>,
    _a_log: Tensor<B, 1>,
    _d_skip: Tensor<B, 1>,
    _initial_ssm: Option<Tensor<B, 5>>,
    _cuda_ssd_core_mode: CudaSsdCoreMode,
    _capture_cuda_ssd_state_history: bool,
) -> Option<AcceleratedSsdForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

fn matches_autodiff_fusion_type<B, BT, R>() -> bool
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        matches_type::<B::FloatTensorPrimitive, WgpuFusionAutodiffTensor<BT>>()
    } else {
        #[cfg(feature = "cuda")]
        {
            if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
                return matches_type::<B::FloatTensorPrimitive, CudaFusionAutodiffTensor<BT>>();
            }
        }
        false
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

fn wrap_fusion_autodiff_inner<B, BT, R>(
    value: FusionTensor<FusionCubeRuntime<R, BT>>,
) -> Option<B::FloatTensorPrimitive>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let boxed: Box<dyn Any> = Box::new(value);
        let inner = boxed
            .downcast::<FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>>()
            .ok()
            .map(|boxed| *boxed)?;
        let ad = <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
        return try_cast_backend::<B, _>(ad);
    }
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            let boxed: Box<dyn Any> = Box::new(value);
            let inner = boxed
                .downcast::<FusionTensor<FusionCubeRuntime<CudaRuntime, BT>>>()
                .ok()
                .map(|boxed| *boxed)?;
            let ad = <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::from_inner(inner);
            return try_cast_backend::<B, _>(ad);
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::sequence::mamba2::backward::{
        Mamba2TensorizedBackwardState, TensorizedMamba2Backward,
    };
    use burn::tensor::ElementConversion;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_autodiff::checkpoint::strategy::NoCheckpointing;
    use burn_autodiff::ops::OpsKind;
    #[cfg(feature = "cuda")]
    use burn_cuda::Cuda;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;
    type AutodiffBackendImpl = Autodiff<Backend>;

    fn assert_close_backend<B: BackendTrait, const D: usize>(
        lhs: Tensor<B, D>,
        rhs: Tensor<B, D>,
        atol: f32,
        rtol: f32,
    ) {
        let max_rhs = rhs.clone().abs().max().into_scalar().elem::<f32>();
        let max_diff = lhs.sub(rhs).abs().max().into_scalar().elem::<f32>();
        let max_tol = atol + rtol * max_rhs;

        assert!(
            max_diff <= max_tol,
            "max difference {max_diff} exceeds tolerance {max_tol} (rhs max {max_rhs})"
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn mamba2_reference_step(
        x_t: Tensor<Backend, 3>,
        z_t: Tensor<Backend, 3>,
        dt_t: Tensor<Backend, 2>,
        b_t: Tensor<Backend, 3>,
        c_t: Tensor<Backend, 3>,
        ssm_state: Tensor<Backend, 4>,
        dt_bias: Tensor<Backend, 1>,
        a_log: Tensor<Backend, 1>,
        d_skip: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 3>, Tensor<Backend, 4>) {
        let [batch, nheads, headdim] = x_t.shape().dims::<3>();
        let d_state = b_t.shape().dims::<3>()[2];
        let _ = z_t;
        let dt = activation::softplus(dt_t + dt_bias.reshape([1, nheads]), 1.0);
        let a = a_log.exp().neg().reshape([1, nheads, 1, 1]);
        let next_ssm = ssm_state * (dt.clone().reshape([batch, nheads, 1, 1]) * a).exp()
            + dt.clone().reshape([batch, nheads, 1, 1])
                * b_t.reshape([batch, nheads, 1, d_state])
                * x_t.clone().reshape([batch, nheads, headdim, 1]);
        let y = (next_ssm.clone() * c_t.reshape([batch, nheads, 1, d_state]))
            .sum_dim(3)
            .reshape([batch, nheads, headdim])
            + d_skip.reshape([1, nheads, 1]) * x_t;
        (y, next_ssm)
    }

    #[test]
    fn tensorized_mamba2_matches_local_reference() {
        let device = Default::default();
        let batch = 1;
        let time = 3;
        let d_model = 4;
        let d_inner = 8;
        let d_state = 2;
        let d_conv = 2;
        let headdim = 4;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * d_model))
                    .map(|idx| ((idx % 29) as f32) / 29.0 - 0.4)
                    .collect::<Vec<_>>(),
                [batch, 1, time, d_model],
            ),
            &device,
        );
        let in_proj = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (0..(d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)))
                    .map(|idx| ((idx % 13) as f32) / 100.0 - 0.06)
                    .collect::<Vec<_>>(),
                [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
            ),
            &device,
        );
        let conv_weight = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (0..(conv_dim * d_conv))
                    .map(|idx| ((idx % 11) as f32) / 80.0 - 0.05)
                    .collect::<Vec<_>>(),
                [conv_dim, d_conv],
            ),
            &device,
        );
        let conv_bias = Tensor::<Backend, 1>::zeros([conv_dim], &device);
        let dt_bias =
            Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.01; nheads], [nheads]), &device);
        let a_log = Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            &device,
        );
        let d_skip = Tensor::<Backend, 1>::ones([nheads], &device);
        let norm_weight = Tensor::<Backend, 1>::ones([d_inner], &device);
        let out_proj = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (0..(d_inner * d_model))
                    .map(|idx| ((idx % 17) as f32) / 90.0 - 0.04)
                    .collect::<Vec<_>>(),
                [d_inner, d_model],
            ),
            &device,
        );

        let output = tensorized_mamba2_forward(
            hidden.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            Some(conv_bias.clone()),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight,
            1.0e-5,
            out_proj.clone(),
            None,
        );

        let zxbcdt = hidden
            .clone()
            .reshape([batch * time, d_model])
            .matmul(in_proj)
            .reshape([batch, time, 2 * d_inner + 2 * ngroups * d_state + nheads]);
        let z = zxbcdt
            .clone()
            .slice_dim(2, 0..d_inner)
            .reshape([batch, time, d_inner]);
        let xbc = zxbcdt
            .clone()
            .slice_dim(2, d_inner..(d_inner + conv_dim))
            .reshape([batch, time, conv_dim]);
        let dt = zxbcdt
            .slice_dim(
                2,
                (d_inner + conv_dim)..(2 * d_inner + 2 * ngroups * d_state + nheads),
            )
            .reshape([batch, time, nheads]);
        let (xbc_conv, _) = tensorized_mamba_depthwise_conv(
            xbc.swap_dims(1, 2).reshape([batch, 1, conv_dim, time]),
            conv_weight,
            Some(conv_bias),
            None,
        );
        let xbc_conv = xbc_conv.swap_dims(2, 3).reshape([batch, time, conv_dim]);

        let mut ssm = Tensor::<Backend, 4>::zeros([batch, nheads, headdim, d_state], &device);
        let mut ys = Vec::new();
        for step in 0..time {
            let x_t = xbc_conv
                .clone()
                .slice_dim(1, step..step + 1)
                .slice_dim(2, 0..d_inner)
                .reshape([batch, nheads, headdim]);
            let b_t = xbc_conv
                .clone()
                .slice_dim(1, step..step + 1)
                .slice_dim(2, d_inner..(d_inner + d_state))
                .reshape([batch, ngroups, d_state]);
            let c_t = xbc_conv
                .clone()
                .slice_dim(1, step..step + 1)
                .slice_dim(2, (d_inner + d_state)..conv_dim)
                .reshape([batch, ngroups, d_state]);
            let z_t = z
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, nheads, headdim]);
            let dt_t = dt
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, nheads]);
            let (y_t, next_ssm) = mamba2_reference_step(
                x_t,
                z_t,
                dt_t,
                repeat_groups_to_heads_runtime(b_t, nheads),
                repeat_groups_to_heads_runtime(c_t, nheads),
                ssm,
                dt_bias.clone(),
                a_log.clone(),
                d_skip.clone(),
            );
            ys.push(
                rmsnorm_gated(
                    y_t.reshape([batch, 1, d_inner]),
                    z.clone().slice_dim(1, step..step + 1),
                    Tensor::<Backend, 1>::ones([d_inner], &device),
                    1.0e-5,
                )
                .reshape([batch, d_inner])
                .matmul(out_proj.clone())
                .reshape([batch, 1, 1, d_model]),
            );
            ssm = next_ssm;
        }
        let reference = Tensor::cat(ys, 2);
        let diff = output.context.sub(reference).abs().max().into_scalar();
        assert!(diff <= 1.0e-4, "mamba2 tensorized diff {diff}");
    }

    #[test]
    fn tensorized_mamba2_wgpu_fused_ssd_core_matches_direct_graph() {
        let device = <WgpuCubeBackend as BackendTrait>::Device::default();
        let batch = 1;
        let time = 4;
        let d_model = 8;
        let d_inner = 16;
        let d_state = 4;
        let d_conv = 3;
        let headdim = 4;
        let ngroups = 2;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden = Tensor::<WgpuCubeBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * d_model))
                    .map(|idx| ((idx % 37) as f32) / 70.0 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, 1, time, d_model],
            ),
            &device,
        );
        let in_proj = Tensor::<WgpuCubeBackend, 2>::from_data(
            TensorData::new(
                (0..(d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)))
                    .map(|idx| ((idx % 31) as f32) / 120.0 - 0.1)
                    .collect::<Vec<_>>(),
                [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
            ),
            &device,
        );
        let conv_weight = Tensor::<WgpuCubeBackend, 2>::from_data(
            TensorData::new(
                (0..(conv_dim * d_conv))
                    .map(|idx| ((idx % 23) as f32) / 90.0 - 0.07)
                    .collect::<Vec<_>>(),
                [conv_dim, d_conv],
            ),
            &device,
        );
        let conv_bias = Tensor::<WgpuCubeBackend, 1>::from_data(
            TensorData::new(
                (0..conv_dim)
                    .map(|idx| ((idx % 11) as f32) / 200.0 - 0.02)
                    .collect::<Vec<_>>(),
                [conv_dim],
            ),
            &device,
        );
        let dt_bias = Tensor::<WgpuCubeBackend, 1>::from_data(
            TensorData::new(vec![0.01; nheads], [nheads]),
            &device,
        );
        let a_log = Tensor::<WgpuCubeBackend, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            &device,
        );
        let d_skip = Tensor::<WgpuCubeBackend, 1>::ones([nheads], &device);
        let norm_weight = Tensor::<WgpuCubeBackend, 1>::ones([d_inner], &device);
        let out_proj = Tensor::<WgpuCubeBackend, 2>::from_data(
            TensorData::new(
                (0..(d_inner * d_model))
                    .map(|idx| ((idx % 19) as f32) / 100.0 - 0.05)
                    .collect::<Vec<_>>(),
                [d_inner, d_model],
            ),
            &device,
        );
        let initial_state = Mamba2TensorizedState {
            conv: Tensor::<WgpuCubeBackend, 4>::from_data(
                TensorData::new(
                    (0..(batch * conv_dim * d_conv))
                        .map(|idx| ((idx % 17) as f32) / 150.0 - 0.015)
                        .collect::<Vec<_>>(),
                    [batch, 1, conv_dim, d_conv],
                ),
                &device,
            ),
            ssm: Tensor::<WgpuCubeBackend, 4>::from_data(
                TensorData::new(
                    (0..(batch * nheads * headdim * d_state))
                        .map(|idx| ((idx % 29) as f32) / 160.0 - 0.01)
                        .collect::<Vec<_>>(),
                    [batch, nheads, headdim, d_state],
                ),
                &device,
            ),
        };

        let baseline = tensorized_mamba2_forward_direct_graph_with_ssd_core_mode(
            hidden.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            Some(conv_bias.clone()),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            1.0e-5,
            out_proj.clone(),
            Some(initial_state.clone()),
            CudaSsdCoreMode::ForcedDisabled,
        );
        let fused = tensorized_mamba2_forward_direct_graph_with_ssd_core_mode(
            hidden,
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj,
            conv_weight,
            Some(conv_bias),
            dt_bias,
            a_log,
            d_skip,
            norm_weight,
            1.0e-5,
            out_proj,
            Some(initial_state),
            CudaSsdCoreMode::ForcedEnabled,
        );

        let _ = <WgpuCubeBackend as BackendTrait>::sync(&device);
        assert_close_backend(baseline.context, fused.context, 2.0e-4, 2.0e-4);
        assert_close_backend(baseline.state.conv, fused.state.conv, 2.0e-4, 2.0e-4);
        assert_close_backend(baseline.state.ssm, fused.state.ssm, 2.0e-4, 2.0e-4);
    }

    #[test]
    #[ignore = "expensive custom backward parity regression; run explicitly when changing the wrapper"]
    fn tensorized_mamba2_custom_backward_matches_graph_backward_on_ndarray_autodiff() {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();

        let batch = 1;
        let time = 1;
        let d_model = 2;
        let d_inner = 2;
        let d_state = 1;
        let d_conv = 1;
        let headdim = 1;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 31) as f32) / 31.0 - 0.35)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)))
                .map(|idx| ((idx % 17) as f32) / 120.0 - 0.05)
                .collect::<Vec<_>>(),
            [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
        );
        let conv_weight_data = TensorData::new(
            (0..(conv_dim * d_conv))
                .map(|idx| ((idx % 13) as f32) / 90.0 - 0.04)
                .collect::<Vec<_>>(),
            [conv_dim, d_conv],
        );
        let conv_bias_data = TensorData::new(
            (0..conv_dim)
                .map(|idx| ((idx % 7) as f32) / 200.0 - 0.01)
                .collect::<Vec<_>>(),
            [conv_dim],
        );
        let dt_bias_data = TensorData::new(vec![0.01; nheads], [nheads]);
        let a_log_data = TensorData::new(vec![1.0f32.ln(); nheads], [nheads]);
        let d_skip_data = TensorData::new(vec![1.0; nheads], [nheads]);
        let norm_weight_data = TensorData::new(vec![1.0; d_inner], [d_inner]);
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 19) as f32) / 100.0 - 0.045)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_conv_data = TensorData::new(
            (0..(batch * conv_dim * d_conv))
                .map(|idx| ((idx % 11) as f32) / 140.0 - 0.02)
                .collect::<Vec<_>>(),
            [batch, 1, conv_dim, d_conv],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 23) as f32) / 160.0 - 0.015)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 29) as f32) / 29.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_graph =
            Tensor::<AutodiffBackendImpl, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_graph =
            Tensor::<AutodiffBackendImpl, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let conv_weight_graph =
            Tensor::<AutodiffBackendImpl, 2>::from_data(conv_weight_data.clone(), &device)
                .require_grad();
        let conv_bias_graph =
            Tensor::<AutodiffBackendImpl, 1>::from_data(conv_bias_data.clone(), &device)
                .require_grad();
        let dt_bias_graph =
            Tensor::<AutodiffBackendImpl, 1>::from_data(dt_bias_data.clone(), &device)
                .require_grad();
        let a_log_graph =
            Tensor::<AutodiffBackendImpl, 1>::from_data(a_log_data.clone(), &device).require_grad();
        let d_skip_graph =
            Tensor::<AutodiffBackendImpl, 1>::from_data(d_skip_data.clone(), &device)
                .require_grad();
        let norm_weight_graph =
            Tensor::<AutodiffBackendImpl, 1>::from_data(norm_weight_data.clone(), &device)
                .require_grad();
        let out_proj_graph =
            Tensor::<AutodiffBackendImpl, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();
        let graph_initial_state = Mamba2TensorizedState {
            conv: Tensor::<AutodiffBackendImpl, 4>::from_data(initial_conv_data.clone(), &device),
            ssm: Tensor::<AutodiffBackendImpl, 4>::from_data(initial_ssm_data.clone(), &device),
        };

        let hidden_wrapper =
            Tensor::<AutodiffBackendImpl, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_wrapper =
            Tensor::<AutodiffBackendImpl, 2>::from_data(in_proj_data, &device).require_grad();
        let conv_weight_wrapper =
            Tensor::<AutodiffBackendImpl, 2>::from_data(conv_weight_data, &device).require_grad();
        let conv_bias_wrapper =
            Tensor::<AutodiffBackendImpl, 1>::from_data(conv_bias_data, &device).require_grad();
        let dt_bias_wrapper =
            Tensor::<AutodiffBackendImpl, 1>::from_data(dt_bias_data, &device).require_grad();
        let a_log_wrapper =
            Tensor::<AutodiffBackendImpl, 1>::from_data(a_log_data, &device).require_grad();
        let d_skip_wrapper =
            Tensor::<AutodiffBackendImpl, 1>::from_data(d_skip_data, &device).require_grad();
        let norm_weight_wrapper =
            Tensor::<AutodiffBackendImpl, 1>::from_data(norm_weight_data, &device).require_grad();
        let out_proj_wrapper =
            Tensor::<AutodiffBackendImpl, 2>::from_data(out_proj_data, &device).require_grad();
        let wrapper_initial_state = Mamba2TensorizedState {
            conv: Tensor::<AutodiffBackendImpl, 4>::from_data(initial_conv_data, &device),
            ssm: Tensor::<AutodiffBackendImpl, 4>::from_data(initial_ssm_data, &device),
        };

        let graph = tensorized_mamba2_forward_impl(
            hidden_graph.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_graph.clone(),
            conv_weight_graph.clone(),
            Some(conv_bias_graph.clone()),
            dt_bias_graph.clone(),
            a_log_graph.clone(),
            d_skip_graph.clone(),
            norm_weight_graph.clone(),
            1.0e-5,
            out_proj_graph.clone(),
            Some(graph_initial_state),
        );
        let wrapped = tensorized_mamba2_custom_backward_ndarray(
            hidden_wrapper.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_wrapper.clone(),
            conv_weight_wrapper.clone(),
            conv_bias_wrapper.clone(),
            dt_bias_wrapper.clone(),
            a_log_wrapper.clone(),
            d_skip_wrapper.clone(),
            norm_weight_wrapper.clone(),
            1.0e-5,
            out_proj_wrapper.clone(),
            Some(wrapper_initial_state),
        );

        assert_close_backend(
            graph.context.clone(),
            wrapped.context.clone(),
            1.0e-4,
            1.0e-4,
        );
        assert_close_backend(
            graph.state.conv.clone(),
            wrapped.state.conv.clone(),
            1.0e-4,
            1.0e-4,
        );
        assert_close_backend(
            graph.state.ssm.clone(),
            wrapped.state.ssm.clone(),
            1.0e-4,
            1.0e-4,
        );

        let output_weights =
            Tensor::<AutodiffBackendImpl, 4>::from_data(output_weight_data, &device);
        let graph_grads = (graph.context * output_weights.clone()).sum().backward();
        let wrapper_grads = (wrapped.context * output_weights).sum().backward();

        assert_close_backend(
            hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
            hidden_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper hidden grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            in_proj_graph
                .grad(&graph_grads)
                .expect("graph in_proj grad"),
            in_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper in_proj grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            conv_weight_graph
                .grad(&graph_grads)
                .expect("graph conv weight grad"),
            conv_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv weight grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            conv_bias_graph
                .grad(&graph_grads)
                .expect("graph conv bias grad"),
            conv_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv bias grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            dt_bias_graph
                .grad(&graph_grads)
                .expect("graph dt bias grad"),
            dt_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper dt bias grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            a_log_graph.grad(&graph_grads).expect("graph a_log grad"),
            a_log_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper a_log grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            d_skip_graph.grad(&graph_grads).expect("graph d_skip grad"),
            d_skip_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper d_skip grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            norm_weight_graph
                .grad(&graph_grads)
                .expect("graph norm weight grad"),
            norm_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper norm weight grad"),
            2.0e-4,
            2.0e-4,
        );
        assert_close_backend(
            out_proj_graph
                .grad(&graph_grads)
                .expect("graph out_proj grad"),
            out_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper out_proj grad"),
            2.0e-4,
            2.0e-4,
        );
    }

    #[test]
    fn tensorized_mamba2_custom_backward_wrapper_matches_forward_output_on_ndarray_autodiff() {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        let batch = 1;
        let time = 1;
        let d_model = 2;
        let d_inner = 2;
        let d_state = 1;
        let d_conv = 1;
        let headdim = 1;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(vec![0.1, -0.2], [batch, 1, time, d_model]),
            &device,
        )
        .require_grad();
        let in_proj = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(
                vec![0.05; d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)],
                [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
            ),
            &device,
        )
        .require_grad();
        let conv_weight = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(vec![0.02; conv_dim * d_conv], [conv_dim, d_conv]),
            &device,
        )
        .require_grad();
        let conv_bias = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![0.0; conv_dim], [conv_dim]),
            &device,
        )
        .require_grad();
        let dt_bias = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![0.01; nheads], [nheads]),
            &device,
        )
        .require_grad();
        let a_log = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            &device,
        )
        .require_grad();
        let d_skip = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0; nheads], [nheads]),
            &device,
        )
        .require_grad();
        let norm_weight = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0; d_inner], [d_inner]),
            &device,
        )
        .require_grad();
        let out_proj = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(vec![0.04; d_inner * d_model], [d_inner, d_model]),
            &device,
        )
        .require_grad();

        let direct = tensorized_mamba2_forward_impl(
            hidden.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            Some(conv_bias.clone()),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            1.0e-5,
            out_proj.clone(),
            None,
        );
        let wrapped = tensorized_mamba2_custom_backward_ndarray(
            hidden,
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj,
            conv_weight,
            conv_bias,
            dt_bias,
            a_log,
            d_skip,
            norm_weight,
            1.0e-5,
            out_proj,
            None,
        );

        assert_eq!(
            wrapped.context.shape().dims::<4>(),
            [batch, 1, time, d_model]
        );
        assert_eq!(
            wrapped.state.conv.shape().dims::<4>(),
            [batch, 1, conv_dim, d_conv]
        );
        assert_eq!(
            wrapped.state.ssm.shape().dims::<4>(),
            [batch, nheads, headdim, d_state]
        );
        assert_close_backend(direct.context, wrapped.context, 1.0e-5, 1.0e-5);
        assert_close_backend(direct.state.conv, wrapped.state.conv, 1.0e-5, 1.0e-5);
        assert_close_backend(direct.state.ssm, wrapped.state.ssm, 1.0e-5, 1.0e-5);
    }

    #[test]
    fn tensorized_mamba2_custom_backward_wrapper_runs_backward_on_ndarray_autodiff() {
        let device = <AutodiffBackendImpl as BackendTrait>::Device::default();
        let batch = 1;
        let time = 1;
        let d_model = 2;
        let d_inner = 2;
        let d_state = 1;
        let d_conv = 1;
        let headdim = 1;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden = Tensor::<AutodiffBackendImpl, 4>::from_data(
            TensorData::new(vec![0.1, -0.2], [batch, 1, time, d_model]),
            &device,
        )
        .require_grad();
        let in_proj = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(
                vec![0.05; d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)],
                [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
            ),
            &device,
        )
        .require_grad();
        let conv_weight = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(vec![0.02; conv_dim * d_conv], [conv_dim, d_conv]),
            &device,
        )
        .require_grad();
        let conv_bias = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![0.0; conv_dim], [conv_dim]),
            &device,
        )
        .require_grad();
        let dt_bias = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![0.01; nheads], [nheads]),
            &device,
        )
        .require_grad();
        let a_log = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            &device,
        )
        .require_grad();
        let d_skip = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0; nheads], [nheads]),
            &device,
        )
        .require_grad();
        let norm_weight = Tensor::<AutodiffBackendImpl, 1>::from_data(
            TensorData::new(vec![1.0; d_inner], [d_inner]),
            &device,
        )
        .require_grad();
        let out_proj = Tensor::<AutodiffBackendImpl, 2>::from_data(
            TensorData::new(vec![0.04; d_inner * d_model], [d_inner, d_model]),
            &device,
        )
        .require_grad();

        let wrapped = tensorized_mamba2_custom_backward_ndarray(
            hidden.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            dt_bias.clone(),
            a_log.clone(),
            d_skip.clone(),
            norm_weight.clone(),
            1.0e-5,
            out_proj.clone(),
            None,
        );
        let grads = wrapped.context.sum().backward();

        assert!(hidden.grad(&grads).is_some(), "hidden grad missing");
        assert!(in_proj.grad(&grads).is_some(), "in_proj grad missing");
        assert!(
            conv_weight.grad(&grads).is_some(),
            "conv_weight grad missing"
        );
        assert!(conv_bias.grad(&grads).is_some(), "conv_bias grad missing");
        assert!(dt_bias.grad(&grads).is_some(), "dt_bias grad missing");
        assert!(a_log.grad(&grads).is_some(), "a_log grad missing");
        assert!(d_skip.grad(&grads).is_some(), "d_skip grad missing");
        assert!(
            norm_weight.grad(&grads).is_some(),
            "norm_weight grad missing"
        );
        assert!(out_proj.grad(&grads).is_some(), "out_proj grad missing");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tensorized_mamba2_custom_backward_matches_graph_backward_on_cuda_autodiff() {
        type CudaBackend = Cuda<f32, i32>;
        type CudaAutodiffBackendImpl = Autodiff<CudaBackend>;

        let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();
        let batch = 1;
        let time = 2;
        let d_model = 4;
        let d_inner = 4;
        let d_state = 2;
        let d_conv = 2;
        let headdim = 2;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 37) as f32) / 37.0 - 0.3)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)))
                .map(|idx| ((idx % 23) as f32) / 150.0 - 0.06)
                .collect::<Vec<_>>(),
            [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
        );
        let conv_weight_data = TensorData::new(
            (0..(conv_dim * d_conv))
                .map(|idx| ((idx % 19) as f32) / 120.0 - 0.05)
                .collect::<Vec<_>>(),
            [conv_dim, d_conv],
        );
        let conv_bias_data = TensorData::new(
            (0..conv_dim)
                .map(|idx| ((idx % 11) as f32) / 180.0 - 0.02)
                .collect::<Vec<_>>(),
            [conv_dim],
        );
        let dt_bias_data = TensorData::new(vec![0.01; nheads], [nheads]);
        let a_log_data = TensorData::new(vec![1.0f32.ln(); nheads], [nheads]);
        let d_skip_data = TensorData::new(vec![1.0; nheads], [nheads]);
        let norm_weight_data = TensorData::new(vec![1.0; d_inner], [d_inner]);
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 29) as f32) / 140.0 - 0.045)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_conv_data = TensorData::new(
            (0..(batch * conv_dim * d_conv))
                .map(|idx| ((idx % 17) as f32) / 200.0 - 0.025)
                .collect::<Vec<_>>(),
            [batch, 1, conv_dim, d_conv],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 31) as f32) / 220.0 - 0.015)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 41) as f32) / 41.0 - 0.2)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_graph =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_graph =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let conv_weight_graph =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(conv_weight_data.clone(), &device)
                .require_grad();
        let conv_bias_graph =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(conv_bias_data.clone(), &device)
                .require_grad();
        let dt_bias_graph =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(dt_bias_data.clone(), &device)
                .require_grad();
        let a_log_graph =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(a_log_data.clone(), &device)
                .require_grad();
        let d_skip_graph =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(d_skip_data.clone(), &device)
                .require_grad();
        let norm_weight_graph =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(norm_weight_data.clone(), &device)
                .require_grad();
        let out_proj_graph =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();

        let hidden_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(in_proj_data, &device).require_grad();
        let conv_weight_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(conv_weight_data, &device)
                .require_grad();
        let conv_bias_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(conv_bias_data, &device).require_grad();
        let dt_bias_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(dt_bias_data, &device).require_grad();
        let a_log_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(a_log_data, &device).require_grad();
        let d_skip_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(d_skip_data, &device).require_grad();
        let norm_weight_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(norm_weight_data, &device)
                .require_grad();
        let out_proj_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(out_proj_data, &device).require_grad();

        let graph = tensorized_mamba2_forward_direct_graph(
            hidden_graph.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_graph.clone(),
            conv_weight_graph.clone(),
            Some(conv_bias_graph.clone()),
            dt_bias_graph.clone(),
            a_log_graph.clone(),
            d_skip_graph.clone(),
            norm_weight_graph.clone(),
            1.0e-5,
            out_proj_graph.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(
                    initial_conv_data.clone(),
                    &device,
                ),
                ssm: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(
                    initial_ssm_data.clone(),
                    &device,
                ),
            }),
        );
        let wrapped = tensorized_mamba2_forward_custom_backward(
            hidden_wrapper.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_wrapper.clone(),
            conv_weight_wrapper.clone(),
            Some(conv_bias_wrapper.clone()),
            dt_bias_wrapper.clone(),
            a_log_wrapper.clone(),
            d_skip_wrapper.clone(),
            norm_weight_wrapper.clone(),
            1.0e-5,
            out_proj_wrapper.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(initial_conv_data, &device),
                ssm: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(initial_ssm_data, &device),
            }),
        )
        .expect("cuda custom backward path available");

        let _ = <CudaAutodiffBackendImpl as BackendTrait>::sync(&device);
        assert_close_backend(
            graph.context.clone(),
            wrapped.context.clone(),
            3.0e-2,
            3.0e-2,
        );
        assert_close_backend(
            graph.state.conv.clone(),
            wrapped.state.conv.clone(),
            3.0e-2,
            3.0e-2,
        );
        assert_close_backend(
            graph.state.ssm.clone(),
            wrapped.state.ssm.clone(),
            3.0e-2,
            3.0e-2,
        );

        let output_weights =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(output_weight_data, &device);
        let graph_grads = (graph.context * output_weights.clone()).sum().backward();
        let wrapper_grads = (wrapped.context * output_weights).sum().backward();
        let _ = <CudaAutodiffBackendImpl as BackendTrait>::sync(&device);

        assert_close_backend(
            hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
            hidden_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper hidden grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            in_proj_graph
                .grad(&graph_grads)
                .expect("graph in_proj grad"),
            in_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper in_proj grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            conv_weight_graph
                .grad(&graph_grads)
                .expect("graph conv weight grad"),
            conv_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv weight grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            conv_bias_graph
                .grad(&graph_grads)
                .expect("graph conv bias grad"),
            conv_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv bias grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            dt_bias_graph
                .grad(&graph_grads)
                .expect("graph dt bias grad"),
            dt_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper dt bias grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            a_log_graph.grad(&graph_grads).expect("graph a_log grad"),
            a_log_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper a_log grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            d_skip_graph.grad(&graph_grads).expect("graph d_skip grad"),
            d_skip_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper d_skip grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            norm_weight_graph
                .grad(&graph_grads)
                .expect("graph norm weight grad"),
            norm_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper norm weight grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            out_proj_graph
                .grad(&graph_grads)
                .expect("graph out_proj grad"),
            out_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper out_proj grad"),
            5.0e-2,
            5.0e-2,
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tensorized_mamba2_shell_fused_path_matches_tensorized_wrapper_on_cuda_autodiff() {
        type CudaBackend = burn_cuda::Cuda<f32, i32>;
        type CudaAutodiffBackendImpl = Autodiff<CudaBackend>;

        let device = <CudaAutodiffBackendImpl as BackendTrait>::Device::default();
        let batch = 1;
        let time = 2;
        let d_model = 4;
        let d_inner = 4;
        let d_state = 2;
        let d_conv = 2;
        let headdim = 2;
        let ngroups = 1;
        let nheads = d_inner / headdim;
        let conv_dim = d_inner + 2 * ngroups * d_state;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 47) as f32) / 47.0 - 0.3)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * (2 * d_inner + 2 * ngroups * d_state + nheads)))
                .map(|idx| ((idx % 29) as f32) / 170.0 - 0.06)
                .collect::<Vec<_>>(),
            [d_model, 2 * d_inner + 2 * ngroups * d_state + nheads],
        );
        let conv_weight_data = TensorData::new(
            (0..(conv_dim * d_conv))
                .map(|idx| ((idx % 23) as f32) / 130.0 - 0.04)
                .collect::<Vec<_>>(),
            [conv_dim, d_conv],
        );
        let conv_bias_data = TensorData::new(
            (0..conv_dim)
                .map(|idx| ((idx % 13) as f32) / 220.0 - 0.02)
                .collect::<Vec<_>>(),
            [conv_dim],
        );
        let dt_bias_data = TensorData::new(vec![0.01; nheads], [nheads]);
        let a_log_data = TensorData::new(vec![1.0f32.ln(); nheads], [nheads]);
        let d_skip_data = TensorData::new(vec![1.0; nheads], [nheads]);
        let norm_weight_data = TensorData::new(vec![1.0; d_inner], [d_inner]);
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 31) as f32) / 160.0 - 0.045)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_conv_data = TensorData::new(
            (0..(batch * conv_dim * d_conv))
                .map(|idx| ((idx % 19) as f32) / 210.0 - 0.02)
                .collect::<Vec<_>>(),
            [batch, 1, conv_dim, d_conv],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 37) as f32) / 240.0 - 0.015)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 53) as f32) / 53.0 - 0.22)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let conv_weight_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(conv_weight_data.clone(), &device)
                .require_grad();
        let conv_bias_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(conv_bias_data.clone(), &device)
                .require_grad();
        let dt_bias_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(dt_bias_data.clone(), &device)
                .require_grad();
        let a_log_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(a_log_data.clone(), &device)
                .require_grad();
        let d_skip_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(d_skip_data.clone(), &device)
                .require_grad();
        let norm_weight_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(norm_weight_data.clone(), &device)
                .require_grad();
        let out_proj_wrapper =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();

        let hidden_fused =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_fused =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(in_proj_data, &device).require_grad();
        let conv_weight_fused =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(conv_weight_data, &device)
                .require_grad();
        let conv_bias_fused =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(conv_bias_data, &device).require_grad();
        let dt_bias_fused =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(dt_bias_data, &device).require_grad();
        let a_log_fused =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(a_log_data, &device).require_grad();
        let d_skip_fused =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(d_skip_data, &device).require_grad();
        let norm_weight_fused =
            Tensor::<CudaAutodiffBackendImpl, 1>::from_data(norm_weight_data, &device)
                .require_grad();
        let out_proj_fused =
            Tensor::<CudaAutodiffBackendImpl, 2>::from_data(out_proj_data, &device).require_grad();

        let wrapper = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
            hidden_wrapper.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_wrapper.clone(),
            conv_weight_wrapper.clone(),
            Some(conv_bias_wrapper.clone()),
            dt_bias_wrapper.clone(),
            a_log_wrapper.clone(),
            d_skip_wrapper.clone(),
            norm_weight_wrapper.clone(),
            1.0e-5,
            out_proj_wrapper.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(
                    initial_conv_data.clone(),
                    &device,
                ),
                ssm: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(
                    initial_ssm_data.clone(),
                    &device,
                ),
            }),
            CudaSsdCoreMode::ForcedDisabled,
            CudaShellCoreMode::ForcedDisabled,
        )
        .expect("cuda tensorized wrapper available");
        let fused = tensorized_mamba2_forward_custom_backward_with_cuda_modes(
            hidden_fused.clone(),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            in_proj_fused.clone(),
            conv_weight_fused.clone(),
            Some(conv_bias_fused.clone()),
            dt_bias_fused.clone(),
            a_log_fused.clone(),
            d_skip_fused.clone(),
            norm_weight_fused.clone(),
            1.0e-5,
            out_proj_fused.clone(),
            Some(Mamba2TensorizedState {
                conv: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(initial_conv_data, &device),
                ssm: Tensor::<CudaAutodiffBackendImpl, 4>::from_data(initial_ssm_data, &device),
            }),
            CudaSsdCoreMode::ForcedEnabled,
            CudaShellCoreMode::ForcedEnabled,
        )
        .expect("cuda fused shell path available");

        let _ = <CudaAutodiffBackendImpl as BackendTrait>::sync(&device);
        assert_close_backend(
            wrapper.context.clone(),
            fused.context.clone(),
            3.0e-2,
            3.0e-2,
        );
        assert_close_backend(
            wrapper.state.conv.clone(),
            fused.state.conv.clone(),
            3.0e-2,
            3.0e-2,
        );
        assert_close_backend(
            wrapper.state.ssm.clone(),
            fused.state.ssm.clone(),
            3.0e-2,
            3.0e-2,
        );

        let output_weights =
            Tensor::<CudaAutodiffBackendImpl, 4>::from_data(output_weight_data, &device);
        let wrapper_grads = (wrapper.context * output_weights.clone()).sum().backward();
        let fused_grads = (fused.context * output_weights).sum().backward();
        let _ = <CudaAutodiffBackendImpl as BackendTrait>::sync(&device);

        assert_close_backend(
            hidden_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper hidden grad"),
            hidden_fused.grad(&fused_grads).expect("fused hidden grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            in_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper in_proj grad"),
            in_proj_fused
                .grad(&fused_grads)
                .expect("fused in_proj grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            conv_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv_weight grad"),
            conv_weight_fused
                .grad(&fused_grads)
                .expect("fused conv_weight grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            conv_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper conv_bias grad"),
            conv_bias_fused
                .grad(&fused_grads)
                .expect("fused conv_bias grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            dt_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper dt_bias grad"),
            dt_bias_fused
                .grad(&fused_grads)
                .expect("fused dt_bias grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            a_log_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper a_log grad"),
            a_log_fused.grad(&fused_grads).expect("fused a_log grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            d_skip_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper d_skip grad"),
            d_skip_fused.grad(&fused_grads).expect("fused d_skip grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            norm_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper norm_weight grad"),
            norm_weight_fused
                .grad(&fused_grads)
                .expect("fused norm_weight grad"),
            5.0e-2,
            5.0e-2,
        );
        assert_close_backend(
            out_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper out_proj grad"),
            out_proj_fused
                .grad(&fused_grads)
                .expect("fused out_proj grad"),
            5.0e-2,
            5.0e-2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn tensorized_mamba2_custom_backward_ndarray(
        hidden_states: Tensor<AutodiffBackendImpl, 4>,
        d_inner: usize,
        d_state: usize,
        d_conv: usize,
        headdim: usize,
        ngroups: usize,
        in_proj: Tensor<AutodiffBackendImpl, 2>,
        conv_weight: Tensor<AutodiffBackendImpl, 2>,
        conv_bias: Tensor<AutodiffBackendImpl, 1>,
        dt_bias: Tensor<AutodiffBackendImpl, 1>,
        a_log: Tensor<AutodiffBackendImpl, 1>,
        d_skip: Tensor<AutodiffBackendImpl, 1>,
        norm_weight: Tensor<AutodiffBackendImpl, 1>,
        norm_eps: f32,
        out_proj: Tensor<AutodiffBackendImpl, 2>,
        state: Option<Mamba2TensorizedState<AutodiffBackendImpl>>,
    ) -> Mamba2TensorizedOutput<AutodiffBackendImpl> {
        let hidden_prim = hidden_states.clone().into_primitive().tensor();
        let in_proj_prim = in_proj.clone().into_primitive().tensor();
        let conv_weight_prim = conv_weight.clone().into_primitive().tensor();
        let conv_bias_prim = conv_bias.clone().into_primitive().tensor();
        let dt_bias_prim = dt_bias.clone().into_primitive().tensor();
        let a_log_prim = a_log.clone().into_primitive().tensor();
        let d_skip_prim = d_skip.clone().into_primitive().tensor();
        let norm_weight_prim = norm_weight.clone().into_primitive().tensor();
        let out_proj_prim = out_proj.clone().into_primitive().tensor();
        let initial_conv_prim = state
            .as_ref()
            .map(|state| state.conv.clone().into_primitive().tensor());
        let initial_ssm_prim = state
            .as_ref()
            .map(|state| state.ssm.clone().into_primitive().tensor());

        let hidden_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(hidden_prim.clone());
        let in_proj_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(in_proj_prim.clone());
        let conv_weight_inner =
            <AutodiffBackendImpl as AutodiffBackend>::inner(conv_weight_prim.clone());
        let conv_bias_inner =
            <AutodiffBackendImpl as AutodiffBackend>::inner(conv_bias_prim.clone());
        let dt_bias_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(dt_bias_prim.clone());
        let a_log_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(a_log_prim.clone());
        let d_skip_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(d_skip_prim.clone());
        let norm_weight_inner =
            <AutodiffBackendImpl as AutodiffBackend>::inner(norm_weight_prim.clone());
        let out_proj_inner = <AutodiffBackendImpl as AutodiffBackend>::inner(out_proj_prim.clone());
        let initial_conv_inner = initial_conv_prim
            .clone()
            .map(<AutodiffBackendImpl as AutodiffBackend>::inner);
        let initial_ssm_inner = initial_ssm_prim
            .clone()
            .map(<AutodiffBackendImpl as AutodiffBackend>::inner);

        let output = tensorized_mamba2_forward_impl(
            BurnTensor::<Backend, 4>::from_primitive(TensorPrimitive::Float(hidden_inner.clone())),
            d_inner,
            d_state,
            d_conv,
            headdim,
            ngroups,
            BurnTensor::<Backend, 2>::from_primitive(TensorPrimitive::Float(in_proj_inner.clone())),
            BurnTensor::<Backend, 2>::from_primitive(TensorPrimitive::Float(
                conv_weight_inner.clone(),
            )),
            Some(BurnTensor::<Backend, 1>::from_primitive(
                TensorPrimitive::Float(conv_bias_inner.clone()),
            )),
            BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(dt_bias_inner.clone())),
            BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(a_log_inner.clone())),
            BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(d_skip_inner.clone())),
            BurnTensor::<Backend, 1>::from_primitive(TensorPrimitive::Float(
                norm_weight_inner.clone(),
            )),
            norm_eps,
            BurnTensor::<Backend, 2>::from_primitive(TensorPrimitive::Float(
                out_proj_inner.clone(),
            )),
            match (initial_conv_inner.clone(), initial_ssm_inner.clone()) {
                (Some(conv), Some(ssm)) => Some(Mamba2TensorizedState {
                    conv: BurnTensor::<Backend, 4>::from_primitive(TensorPrimitive::Float(conv)),
                    ssm: BurnTensor::<Backend, 4>::from_primitive(TensorPrimitive::Float(ssm)),
                }),
                _ => None,
            },
        );
        let context_inner = output.context.into_primitive().tensor();
        let conv_inner = output.state.conv.into_primitive().tensor();
        let ssm_inner = output.state.ssm.into_primitive().tensor();

        let context = match TensorizedMamba2Backward::<Backend>(PhantomData)
            .prepare::<NoCheckpointing>([
                hidden_prim.node.clone(),
                in_proj_prim.node.clone(),
                conv_weight_prim.node.clone(),
                conv_bias_prim.node.clone(),
                dt_bias_prim.node.clone(),
                a_log_prim.node.clone(),
                d_skip_prim.node.clone(),
                norm_weight_prim.node.clone(),
                out_proj_prim.node.clone(),
            ])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(prep) => prep.finish(
                Mamba2TensorizedBackwardState {
                    hidden_states: hidden_inner,
                    in_proj: in_proj_inner,
                    conv_weight: conv_weight_inner,
                    conv_bias: conv_bias_inner,
                    dt_bias: dt_bias_inner,
                    a_log: a_log_inner,
                    d_skip: d_skip_inner,
                    norm_weight: norm_weight_inner,
                    out_proj: out_proj_inner,
                    initial_conv: initial_conv_inner,
                    initial_ssm: initial_ssm_inner,
                    rmsnorm_inv_rms: None,
                    ssd_state_history: None,
                    d_inner,
                    d_state,
                    d_conv,
                    headdim,
                    ngroups,
                    norm_eps,
                    cuda_ssd_core_mode: CudaSsdCoreMode::ForcedDisabled,
                    cuda_shell_core_mode: CudaShellCoreMode::ForcedDisabled,
                },
                context_inner,
            ),
            OpsKind::UnTracked(prep) => prep.finish(context_inner),
        };

        Mamba2TensorizedOutput {
            context: BurnTensor::<AutodiffBackendImpl, 4>::from_primitive(TensorPrimitive::Float(
                context,
            )),
            state: Mamba2TensorizedState {
                conv: BurnTensor::<AutodiffBackendImpl, 4>::from_primitive(TensorPrimitive::Float(
                    <AutodiffBackendImpl as AutodiffBackend>::from_inner(conv_inner),
                )),
                ssm: BurnTensor::<AutodiffBackendImpl, 4>::from_primitive(TensorPrimitive::Float(
                    <AutodiffBackendImpl as AutodiffBackend>::from_inner(ssm_inner),
                )),
            },
        }
    }

    fn repeat_groups_to_heads_runtime(
        grouped: Tensor<Backend, 3>,
        nheads: usize,
    ) -> Tensor<Backend, 3> {
        let [batch, ngroups, d_state] = grouped.shape().dims::<3>();
        assert_eq!(nheads % ngroups, 0);
        grouped
            .reshape([batch, ngroups, 1, d_state])
            .repeat_dim(2, nheads / ngroups)
            .reshape([batch, nheads, d_state])
    }
}
