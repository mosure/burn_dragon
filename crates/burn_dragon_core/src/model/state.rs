use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::model::low_bit_runtime::{PackedRhoInt8DeviceState, PackedRhoInt8State};

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub rho: Option<Tensor<B, 4>>,
    pub packed_rho_int8: Option<PackedRhoInt8State>,
    pub packed_rho_int8_device: Option<PackedRhoInt8DeviceState<B>>,
    pub rho_norm: Option<Tensor<B, 3>>,
    pub sequence_aux: Option<Tensor<B, 4>>,
    pub y_neuron_state: Option<Tensor<B, 3>>,
    pub clocked_slow_hidden: Option<Tensor<B, 4>>,
    pub summary_memory_hidden: Option<Tensor<B, 4>>,
    #[cfg(any(feature = "viz", feature = "probe"))]
    pub viz: Option<LayerVizState<B>>,
}

#[derive(Debug, Clone)]
pub struct ModelState<B: Backend> {
    pub layers: Vec<LayerState<B>>,
    pub position: usize,
}

#[cfg(any(feature = "viz", feature = "probe"))]
#[derive(Debug, Clone)]
pub struct LayerVizState<B: Backend> {
    pub x_neuron_last: Tensor<B, 2>,
    pub y_gate_last: Tensor<B, 2>,
    pub y_neuron_last: Tensor<B, 2>,
    pub rho_last: Tensor<B, 2>,
}

impl<B: Backend> ModelState<B> {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    rho: None,
                    packed_rho_int8: None,
                    packed_rho_int8_device: None,
                    rho_norm: None,
                    sequence_aux: None,
                    y_neuron_state: None,
                    clocked_slow_hidden: None,
                    summary_memory_hidden: None,
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: None,
                })
                .collect(),
            position: 0,
        }
    }

    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.rho = None;
            layer.packed_rho_int8 = None;
            layer.packed_rho_int8_device = None;
            layer.rho_norm = None;
            layer.sequence_aux = None;
            layer.y_neuron_state = None;
            layer.clocked_slow_hidden = None;
            layer.summary_memory_hidden = None;
        }
        self.position = 0;
    }

    pub fn len(&self) -> usize {
        self.position
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn trim(&mut self, max_len: usize) {
        let _ = max_len;
    }

    pub fn detach_in_place(&mut self) {
        for layer in &mut self.layers {
            layer.rho = layer.rho.take().map(|tensor| tensor.detach());
            layer.packed_rho_int8_device = layer
                .packed_rho_int8_device
                .take()
                .map(|state| state.detach());
            layer.rho_norm = layer.rho_norm.take().map(|tensor| tensor.detach());
            layer.sequence_aux = layer.sequence_aux.take().map(|tensor| tensor.detach());
            layer.y_neuron_state = layer.y_neuron_state.take().map(|tensor| tensor.detach());
            layer.clocked_slow_hidden = layer
                .clocked_slow_hidden
                .take()
                .map(|tensor| tensor.detach());
            layer.summary_memory_hidden = layer
                .summary_memory_hidden
                .take()
                .map(|tensor| tensor.detach());
        }
    }

    #[cfg(any(feature = "viz", feature = "probe"))]
    pub fn take_viz(&mut self) -> Vec<Option<LayerVizState<B>>> {
        self.layers
            .iter_mut()
            .map(|layer| layer.viz.take())
            .collect()
    }

    #[cfg(any(feature = "viz", feature = "probe"))]
    pub fn clear_viz(&mut self) {
        for layer in &mut self.layers {
            layer.viz = None;
        }
    }
}

#[cfg(any(feature = "viz", feature = "probe"))]
impl<B: Backend> LayerState<B> {
    pub fn take_viz(&mut self) -> Option<LayerVizState<B>> {
        self.viz.take()
    }
}
