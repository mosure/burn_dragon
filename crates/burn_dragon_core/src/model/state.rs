use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};

use crate::model::low_bit_runtime::{PackedRhoBlockState, PackedRhoInt8DeviceState};

#[derive(Debug, Clone)]
pub struct LayerState<B: Backend> {
    pub persist_sequence_state: bool,
    pub rho: Option<Tensor<B, 4>>,
    pub packed_rho: Option<PackedRhoBlockState>,
    pub packed_rho_int8_device: Option<PackedRhoInt8DeviceState<B>>,
    pub rho_norm: Option<Tensor<B, 3>>,
    pub sequence_aux: Option<Tensor<B, 4>>,
    pub mamba_angle_state: Option<Tensor<B, 3>>,
    pub mamba_k_state: Option<Tensor<B, 3>>,
    pub mamba_v_state: Option<Tensor<B, 3>>,
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
        Self::with_sequence_state_persistence(num_layers, true)
    }

    pub fn new_ephemeral(num_layers: usize) -> Self {
        Self::with_sequence_state_persistence(num_layers, false)
    }

    fn with_sequence_state_persistence(num_layers: usize, persist_sequence_state: bool) -> Self {
        Self {
            layers: (0..num_layers)
                .map(|_| LayerState {
                    persist_sequence_state,
                    rho: None,
                    packed_rho: None,
                    packed_rho_int8_device: None,
                    rho_norm: None,
                    sequence_aux: None,
                    mamba_angle_state: None,
                    mamba_k_state: None,
                    mamba_v_state: None,
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
            layer.packed_rho = None;
            layer.packed_rho_int8_device = None;
            layer.rho_norm = None;
            layer.sequence_aux = None;
            layer.mamba_angle_state = None;
            layer.mamba_k_state = None;
            layer.mamba_v_state = None;
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
            layer.mamba_angle_state = layer.mamba_angle_state.take().map(|tensor| tensor.detach());
            layer.mamba_k_state = layer.mamba_k_state.take().map(|tensor| tensor.detach());
            layer.mamba_v_state = layer.mamba_v_state.take().map(|tensor| tensor.detach());
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

    pub fn detached_clone(&self) -> Self {
        let mut detached = self.clone();
        detached.detach_in_place();
        detached
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

impl<B: AutodiffBackend> ModelState<B> {
    pub fn inner_cloned(&self) -> ModelState<B::InnerBackend> {
        ModelState {
            layers: self
                .layers
                .iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    rho: layer.rho.clone().map(Tensor::inner),
                    packed_rho: layer.packed_rho.clone(),
                    packed_rho_int8_device: layer.packed_rho_int8_device.clone().map(|state| {
                        PackedRhoInt8DeviceState {
                            logical_shape: state.logical_shape,
                            block_size: state.block_size,
                            scales: state.scales.inner(),
                            packed: state.packed.inner(),
                        }
                    }),
                    rho_norm: layer.rho_norm.clone().map(Tensor::inner),
                    sequence_aux: layer.sequence_aux.clone().map(Tensor::inner),
                    mamba_angle_state: layer.mamba_angle_state.clone().map(Tensor::inner),
                    mamba_k_state: layer.mamba_k_state.clone().map(Tensor::inner),
                    mamba_v_state: layer.mamba_v_state.clone().map(Tensor::inner),
                    y_neuron_state: layer.y_neuron_state.clone().map(Tensor::inner),
                    clocked_slow_hidden: layer.clocked_slow_hidden.clone().map(Tensor::inner),
                    summary_memory_hidden: layer.summary_memory_hidden.clone().map(Tensor::inner),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.clone().map(|viz| LayerVizState {
                        x_neuron_last: viz.x_neuron_last.inner(),
                        y_gate_last: viz.y_gate_last.inner(),
                        y_neuron_last: viz.y_neuron_last.inner(),
                        rho_last: viz.rho_last.inner(),
                    }),
                })
                .collect(),
            position: self.position,
        }
    }

    pub fn from_inner_cloned(state: ModelState<B::InnerBackend>) -> Self {
        ModelState {
            layers: state
                .layers
                .into_iter()
                .map(|layer| LayerState {
                    persist_sequence_state: layer.persist_sequence_state,
                    rho: layer.rho.map(Tensor::from_inner),
                    packed_rho: layer.packed_rho,
                    packed_rho_int8_device: layer.packed_rho_int8_device.map(|state| {
                        PackedRhoInt8DeviceState {
                            logical_shape: state.logical_shape,
                            block_size: state.block_size,
                            scales: Tensor::from_inner(state.scales),
                            packed: Tensor::from_inner(state.packed),
                        }
                    }),
                    rho_norm: layer.rho_norm.map(Tensor::from_inner),
                    sequence_aux: layer.sequence_aux.map(Tensor::from_inner),
                    mamba_angle_state: layer.mamba_angle_state.map(Tensor::from_inner),
                    mamba_k_state: layer.mamba_k_state.map(Tensor::from_inner),
                    mamba_v_state: layer.mamba_v_state.map(Tensor::from_inner),
                    y_neuron_state: layer.y_neuron_state.map(Tensor::from_inner),
                    clocked_slow_hidden: layer.clocked_slow_hidden.map(Tensor::from_inner),
                    summary_memory_hidden: layer.summary_memory_hidden.map(Tensor::from_inner),
                    #[cfg(any(feature = "viz", feature = "probe"))]
                    viz: layer.viz.map(|viz| LayerVizState {
                        x_neuron_last: Tensor::from_inner(viz.x_neuron_last),
                        y_gate_last: Tensor::from_inner(viz.y_gate_last),
                        y_neuron_last: Tensor::from_inner(viz.y_neuron_last),
                        rho_last: Tensor::from_inner(viz.rho_last),
                    }),
                })
                .collect(),
            position: state.position,
        }
    }
}

#[cfg(any(feature = "viz", feature = "probe"))]
impl<B: Backend> LayerState<B> {
    pub fn take_viz(&mut self) -> Option<LayerVizState<B>> {
        self.viz.take()
    }
}
