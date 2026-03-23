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

use crate::kernels::sequence::rwkv8::forward::tensorized_rwkv8_forward_impl;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;

/// Real accelerated RWKV backward kernels are still pending.
///
/// This module now owns the custom recompute-backward wrapper used by the experimental tensorized
/// forward path, but that is still not a true fused backward kernel.
pub const AVAILABLE: bool = false;

pub(crate) type TensorizedRwkv8BackwardState<FT> = (FT, FT, Option<FT>, Option<FT>, FT);

#[derive(Debug)]
pub(crate) struct TensorizedRwkv8Backward<B>(pub(crate) PhantomData<B>);

pub(crate) fn tensorized_rwkv8_backward_impl<B, AB>(
    ops: Ops<TensorizedRwkv8BackwardState<B::FloatTensorPrimitive>, 3>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
    AB: AutodiffBackend<InnerBackend = B>,
{
    let grad_output = grads.consume::<B>(&ops.node);
    let (query_inner, value_inner, rho_state_inner, rho_norm_state_inner, decay_inner) = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
        TensorPrimitive::Float(query_inner),
    ))
    .require_grad();
    let value = BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
        TensorPrimitive::Float(value_inner),
    ))
    .require_grad();
    let decay = BurnTensor::<AB, 3>::from_inner(BurnTensor::<B, 3>::from_primitive(
        TensorPrimitive::Float(decay_inner),
    ))
    .require_grad();
    let rho_state = rho_state_inner.map(|inner| {
        BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            inner,
        )))
    });
    let rho_norm_state = rho_norm_state_inner.map(|inner| {
        BurnTensor::<AB, 3>::from_inner(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            inner,
        )))
    });
    let grad_output = BurnTensor::<AB, 4>::from_inner(BurnTensor::<B, 4>::from_primitive(
        TensorPrimitive::Float(grad_output),
    ));

    let output = tensorized_rwkv8_forward_impl(
        query.clone(),
        value.clone(),
        rho_state,
        rho_norm_state,
        decay.clone(),
    );
    let backward_grads = (output.context * grad_output).sum().backward();

    if let Some(parent) = &parents[0] {
        if let Some(grad) = query.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[1] {
        if let Some(grad) = value.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
    }
    if let Some(parent) = &parents[2] {
        if let Some(grad) = decay.grad(&backward_grads) {
            grads.register::<B>(parent.id, grad.into_primitive().tensor());
        }
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
        tensorized_rwkv8_backward_impl::<WgpuCubeBackend, WgpuCubeAutodiffBackend>(ops, grads);
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
        tensorized_rwkv8_backward_impl::<CudaCubeBackend, CudaCubeAutodiffBackend>(ops, grads);
    }
}
