use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::model::state::LayerState;

#[derive(Debug, Clone)]
pub struct LinearAttentionState<B: Backend> {
    pub rho: Option<Tensor<B, 4>>,
}

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

pub fn linear_attention_state<B: Backend>(layer_state: &LayerState<B>) -> LinearAttentionState<B> {
    LinearAttentionState {
        rho: layer_state.rho.as_ref().cloned(),
    }
}

pub fn write_linear_attention_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    rho: Tensor<B, 4>,
) {
    layer_state.rho = Some(rho);
    layer_state.rho_norm = None;
    layer_state.sequence_aux = None;
}

pub fn rwkv8_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    heads: usize,
    latent: usize,
    device: &B::Device,
) -> Rwkv8State<B> {
    let rho_norm = match layer_state.rho_norm.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, heads, latent] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, heads, latent], device),
    };

    Rwkv8State {
        rho: layer_state.rho.as_ref().cloned(),
        rho_norm,
    }
}

pub fn write_rwkv8_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    rho: Tensor<B, 4>,
    rho_norm: Tensor<B, 3>,
) {
    layer_state.rho = Some(rho);
    layer_state.rho_norm = Some(rho_norm);
    layer_state.sequence_aux = None;
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
    layer_state.rho_norm = None;
    layer_state.sequence_aux = Some(conv);
}
