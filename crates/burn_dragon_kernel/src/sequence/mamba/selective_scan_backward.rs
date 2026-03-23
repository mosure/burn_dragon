use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::{CubeBackend, WgpuRuntime};

use crate::kernels::sequence::mamba::selective_scan_forward::{
    MambaTensorizedState, tensorized_mamba_forward_impl,
};

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;

/// Real accelerated Mamba selective-scan backward kernels are still pending.
///
/// This module now owns the custom recompute-backward wrapper used by the experimental tensorized
/// forward path, but that remains distinct from a true fused selective-scan backward kernel.
pub const AVAILABLE: bool = false;

#[derive(Debug, Clone)]
pub(crate) struct MambaTensorizedBackwardState<FT> {
    pub(crate) hidden_states: FT,
    pub(crate) in_proj: FT,
    pub(crate) conv_weight: FT,
    pub(crate) conv_bias: FT,
    pub(crate) x_proj: FT,
    pub(crate) dt_proj_weight: FT,
    pub(crate) dt_proj_bias: FT,
    pub(crate) a_log: FT,
    pub(crate) d_skip: FT,
    pub(crate) out_proj: FT,
    pub(crate) initial_conv: Option<FT>,
    pub(crate) initial_ssm: Option<FT>,
    pub(crate) d_inner: usize,
    pub(crate) d_state: usize,
    pub(crate) d_conv: usize,
    pub(crate) dt_rank: usize,
}

#[derive(Debug)]
pub(crate) struct TensorizedMambaBackward<B>(pub(crate) PhantomData<B>);

pub(crate) fn tensorized_mamba_backward_impl<B, AB>(
    ops: Ops<MambaTensorizedBackwardState<B::FloatTensorPrimitive>, 10>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
    AB: AutodiffBackend<InnerBackend = B>,
{
    let grad_output = grads.consume::<B>(&ops.node);
    let state = ops.state;
    let parents = ops.parents;

    let hidden_states = BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
        TensorPrimitive::Float(state.hidden_states),
    ))
    .require_grad();
    let in_proj = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.in_proj),
    ))
    .require_grad();
    let conv_weight = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.conv_weight),
    ))
    .require_grad();
    let conv_bias = BurnTensor::<AB, 1>::from_inner(BurnTensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(state.conv_bias),
    ))
    .require_grad();
    let x_proj = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.x_proj),
    ))
    .require_grad();
    let dt_proj_weight = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.dt_proj_weight),
    ))
    .require_grad();
    let dt_proj_bias = BurnTensor::<AB, 1>::from_inner(BurnTensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(state.dt_proj_bias),
    ))
    .require_grad();
    let a_log = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.a_log),
    ))
    .require_grad();
    let d_skip = BurnTensor::<AB, 1>::from_inner(BurnTensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(state.d_skip),
    ))
    .require_grad();
    let out_proj = BurnTensor::<AB, 2>::from_inner(BurnTensor::<B, 2>::from_primitive(
        TensorPrimitive::Float(state.out_proj),
    ))
    .require_grad();
    let initial_state = match (state.initial_conv, state.initial_ssm) {
        (Some(conv), Some(ssm)) => Some(MambaTensorizedState {
            conv: BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
                TensorPrimitive::Float(conv),
            )),
            ssm: BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
                TensorPrimitive::Float(ssm),
            )),
        }),
        _ => None,
    };
    let grad_output = BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
        TensorPrimitive::Float(grad_output),
    ));

    let output = tensorized_mamba_forward_impl(
        hidden_states.clone(),
        state.d_inner,
        state.d_state,
        state.d_conv,
        state.dt_rank,
        in_proj.clone(),
        conv_weight.clone(),
        Some(conv_bias.clone()),
        x_proj.clone(),
        dt_proj_weight.clone(),
        dt_proj_bias.clone(),
        a_log.clone(),
        d_skip.clone(),
        out_proj.clone(),
        initial_state,
    );
    let backward_grads = (output.context * grad_output).sum().backward();

    if let Some(parent) = &parents[0] {
        if let Some(grad) = hidden_states.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[1] {
        if let Some(grad) = in_proj.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[2] {
        if let Some(grad) = conv_weight.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[3] {
        if let Some(grad) = conv_bias.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[4] {
        if let Some(grad) = x_proj.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[5] {
        if let Some(grad) = dt_proj_weight.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[6] {
        if let Some(grad) = dt_proj_bias.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[7] {
        if let Some(grad) = a_log.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[8] {
        if let Some(grad) = d_skip.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[9] {
        if let Some(grad) = out_proj.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
}

impl Backward<WgpuCubeBackend, 10> for TensorizedMambaBackward<WgpuCubeBackend> {
    type State = MambaTensorizedBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 10>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba_backward_impl::<WgpuCubeBackend, WgpuCubeAutodiffBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 10> for TensorizedMambaBackward<CudaCubeBackend> {
    type State = MambaTensorizedBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 10>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba_backward_impl::<CudaCubeBackend, CudaCubeAutodiffBackend>(ops, grads);
    }
}
