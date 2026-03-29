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

#[derive(Debug, Clone)]
pub struct Mamba3State<B: Backend> {
    pub ssm: Tensor<B, 4>,
    pub angle: Tensor<B, 3>,
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}

pub fn mamba_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    ssm_heads: usize,
    ssm_width: usize,
    d_state: usize,
    conv_channels: usize,
    d_conv: usize,
    device: &B::Device,
) -> MambaState<B> {
    let ssm = match layer_state.rho.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, ssm_heads, ssm_width, d_state] => {
            state.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, ssm_heads, ssm_width, d_state], device),
    };
    let conv = match layer_state.sequence_aux.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, 1, conv_channels, d_conv] => {
            state.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, 1, conv_channels, d_conv], device),
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
    layer_state.mamba_angle_state = None;
    layer_state.mamba_k_state = None;
    layer_state.mamba_v_state = None;
}

pub fn mamba3_state<B: Backend>(
    layer_state: &LayerState<B>,
    batch: usize,
    nheads: usize,
    headdim: usize,
    d_state: usize,
    angle_dim: usize,
    device: &B::Device,
) -> Mamba3State<B> {
    let ssm = match layer_state.rho.as_ref() {
        Some(state) if state.shape().dims::<4>() == [batch, nheads, headdim, d_state] => {
            state.clone()
        }
        _ => Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], device),
    };
    let angle = match layer_state.mamba_angle_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, angle_dim] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, angle_dim], device),
    };
    let k = match layer_state.mamba_k_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, d_state] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, d_state], device),
    };
    let v = match layer_state.mamba_v_state.as_ref() {
        Some(state) if state.shape().dims::<3>() == [batch, nheads, headdim] => state.clone(),
        _ => Tensor::<B, 3>::zeros([batch, nheads, headdim], device),
    };
    Mamba3State { ssm, angle, k, v }
}

pub fn write_mamba3_state<B: Backend>(
    layer_state: &mut LayerState<B>,
    ssm: Tensor<B, 4>,
    angle: Tensor<B, 3>,
    k: Tensor<B, 3>,
    v: Tensor<B, 3>,
) {
    layer_state.rho = Some(ssm);
    layer_state.packed_rho = None;
    layer_state.packed_rho_int8_device = None;
    layer_state.rho_norm = None;
    layer_state.sequence_aux = None;
    layer_state.mamba_angle_state = Some(angle);
    layer_state.mamba_k_state = Some(k);
    layer_state.mamba_v_state = Some(v);
}
