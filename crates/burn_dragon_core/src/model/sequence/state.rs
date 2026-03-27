use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::model::state::LayerState;

#[derive(Debug, Clone)]
pub struct Rwkv8State<B: Backend> {
    pub rho: Option<Tensor<B, 4>>,
    pub rho_norm: Tensor<B, 3>,
}

#[derive(Debug, Clone)]
pub struct MambaState<B: Backend> {
    pub ssm: Tensor<B, 4>,
    pub conv: Tensor<B, 4>,
}

pub fn mamba_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    device: &B::Device,
) -> MambaState<B> {
    let ssm = match layer_state.rho.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, 1, d_inner, d_state] => state.clone(),
        _ => Tensor::<B, 4>::zeros([batch, 1, d_inner, d_state], device),
    };
    let conv = match layer_state.sequence_aux.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, 1, d_inner, d_conv] => state.clone(),
        _ => Tensor::<B, 4>::zeros([batch, 1, d_inner, d_conv], device),
    };
    MambaState { ssm, conv }
}

pub fn write_mamba_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    ssm: Tensor<B, 4>,
    conv: Tensor<B, 4>,
) {
    layer_state.rho = Some(ssm);
    layer_state.packed_rho = None;
    layer_state.packed_rho_int8_device = None;
    layer_state.rho_norm = None;
    layer_state.sequence_aux = Some(conv);
}
