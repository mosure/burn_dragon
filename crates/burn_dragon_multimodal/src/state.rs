use burn::tensor::Tensor;
use burn::tensor::backend::{AutodiffBackend, Backend};
#[cfg(any(feature = "viz", feature = "probe"))]
use burn_dragon_core::LayerVizState;
use burn_dragon_core::api::state::{BankedRhoState, ModelState, StructuredTopologyState};
use burn_dragon_core::model::LayerState;
use burn_dragon_stream::{
    FusionCarryPolicy, StateCarryPolicy, StreamBoundary, StreamStepMetadata, TbpttWindow,
};
use burn_dragon_vision::api::model::VisionCellularState;

#[derive(Clone)]
pub enum VisionMultimodalState<B: Backend> {
    Pyramid(StructuredTopologyState<B>),
    Cellular(VisionCellularState<B>),
}

impl<B: Backend> VisionMultimodalState<B> {
    pub fn detach(&self) -> Self {
        match self {
            Self::Pyramid(state) => Self::Pyramid(state.detach()),
            Self::Cellular(state) => Self::Cellular(state.detach()),
        }
    }
}

impl<B: AutodiffBackend> VisionMultimodalState<B> {
    pub fn inner(&self) -> VisionMultimodalState<B::InnerBackend> {
        match self {
            Self::Pyramid(state) => VisionMultimodalState::Pyramid(StructuredTopologyState {
                primary_state: state.primary_state.clone().inner(),
                context_state: state.context_state.clone().inner(),
                rho: BankedRhoState {
                    primary_rho: state.rho.primary_rho.clone().inner(),
                    context_rho: state.rho.context_rho.clone().inner(),
                    global_rho: state.rho.global_rho.clone().inner(),
                },
                temporal_position: state.temporal_position,
                prediction_age: state.prediction_age,
            }),
            Self::Cellular(state) => VisionMultimodalState::Cellular(VisionCellularState {
                token_state: state.token_state.clone().inner(),
                rho: state.rho.clone().inner(),
                temporal_position: state.temporal_position,
                prediction_age: state.prediction_age,
            }),
        }
    }

    pub fn from_inner(state: VisionMultimodalState<B::InnerBackend>) -> Self {
        match state {
            VisionMultimodalState::Pyramid(state) => Self::Pyramid(StructuredTopologyState {
                primary_state: Tensor::from_inner(state.primary_state),
                context_state: Tensor::from_inner(state.context_state),
                rho: BankedRhoState {
                    primary_rho: Tensor::from_inner(state.rho.primary_rho),
                    context_rho: Tensor::from_inner(state.rho.context_rho),
                    global_rho: Tensor::from_inner(state.rho.global_rho),
                },
                temporal_position: state.temporal_position,
                prediction_age: state.prediction_age,
            }),
            VisionMultimodalState::Cellular(state) => Self::Cellular(VisionCellularState {
                token_state: Tensor::from_inner(state.token_state),
                rho: Tensor::from_inner(state.rho),
                temporal_position: state.temporal_position,
                prediction_age: state.prediction_age,
            }),
        }
    }
}

#[derive(Clone)]
pub struct MultimodalDragonState<B: Backend> {
    pub vision: Option<VisionMultimodalState<B>>,
    pub query_text: Option<ModelState<B>>,
    pub fusion: ModelState<B>,
    pub cached_vision_tokens: Option<Tensor<B, 3>>,
    pub cached_query_tokens: Option<Tensor<B, 3>>,
    pub cached_slot_tokens: Option<Tensor<B, 3>>,
}

impl<B: Backend> MultimodalDragonState<B> {
    pub fn new(query_layers: usize, fusion_layers: usize) -> Self {
        Self {
            vision: None,
            query_text: Some(ModelState::new(query_layers)),
            fusion: ModelState::new(fusion_layers),
            cached_vision_tokens: None,
            cached_query_tokens: None,
            cached_slot_tokens: None,
        }
    }

    pub fn detach(&self) -> Self {
        Self {
            vision: self.vision.as_ref().map(VisionMultimodalState::detach),
            query_text: self.query_text.as_ref().map(detach_model_state),
            fusion: detach_model_state(&self.fusion),
            cached_vision_tokens: self.cached_vision_tokens.clone().map(Tensor::detach),
            cached_query_tokens: self.cached_query_tokens.clone().map(Tensor::detach),
            cached_slot_tokens: self.cached_slot_tokens.clone().map(Tensor::detach),
        }
    }

    pub fn reset_modality_state(&mut self) {
        self.vision = None;
        if let Some(state) = &mut self.query_text {
            state.reset();
        }
    }

    pub fn reset_fusion_state(&mut self) {
        self.fusion.reset();
        self.cached_vision_tokens = None;
        self.cached_query_tokens = None;
        self.cached_slot_tokens = None;
    }

    pub fn apply_stream_controls(
        &mut self,
        stream: &StreamStepMetadata,
        window: TbpttWindow,
        state_carry: StateCarryPolicy,
        fusion_carry: FusionCarryPolicy,
    ) {
        let reset_modality = match state_carry {
            StateCarryPolicy::Always => false,
            StateCarryPolicy::Never => true,
            StateCarryPolicy::UntilBoundary => stream.should_reset_state(),
        };
        let reset_fusion = match fusion_carry {
            FusionCarryPolicy::Always => false,
            FusionCarryPolicy::Never => true,
            FusionCarryPolicy::UntilBoundary => matches!(
                stream.boundary,
                StreamBoundary::ResetEpisode | StreamBoundary::ResetAll
            ),
        };

        if reset_modality {
            self.reset_modality_state();
        }
        if reset_fusion {
            self.reset_fusion_state();
        }

        if window.requires_detach() && stream.step_index < window.detach_prefix_steps() {
            let detached = self.detach();
            *self = detached;
        }
    }
}

pub fn detach_model_state<B: Backend>(state: &ModelState<B>) -> ModelState<B> {
    ModelState {
        layers: state
            .layers
            .iter()
            .map(|layer| LayerState {
                rho: layer.rho.clone().map(Tensor::detach),
                packed_rho: layer.packed_rho.clone(),
                packed_rho_int8_device: layer
                    .packed_rho_int8_device
                    .clone()
                    .map(|state| state.detach()),
                rho_norm: layer.rho_norm.clone().map(Tensor::detach),
                sequence_aux: layer.sequence_aux.clone().map(Tensor::detach),
                mamba_angle_state: layer.mamba_angle_state.clone().map(Tensor::detach),
                mamba_k_state: layer.mamba_k_state.clone().map(Tensor::detach),
                mamba_v_state: layer.mamba_v_state.clone().map(Tensor::detach),
                y_neuron_state: layer.y_neuron_state.clone().map(Tensor::detach),
                clocked_slow_hidden: layer.clocked_slow_hidden.clone().map(Tensor::detach),
                summary_memory_hidden: layer.summary_memory_hidden.clone().map(Tensor::detach),
                #[cfg(any(feature = "viz", feature = "probe"))]
                viz: layer.viz.clone().map(|viz| LayerVizState {
                    x_neuron_last: viz.x_neuron_last.detach(),
                    y_gate_last: viz.y_gate_last.detach(),
                    y_neuron_last: viz.y_neuron_last.detach(),
                    rho_last: viz.rho_last.detach(),
                }),
            })
            .collect(),
        position: state.position,
    }
}

pub fn model_state_inner<B: AutodiffBackend>(state: &ModelState<B>) -> ModelState<B::InnerBackend> {
    ModelState {
        layers: state
            .layers
            .iter()
            .map(|layer| LayerState {
                rho: layer.rho.clone().map(Tensor::inner),
                packed_rho: layer.packed_rho.clone(),
                packed_rho_int8_device: layer.packed_rho_int8_device.clone().map(|state| {
                    burn_dragon_core::PackedRhoInt8DeviceState {
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
        position: state.position,
    }
}

pub fn model_state_from_inner<B: AutodiffBackend>(
    state: ModelState<B::InnerBackend>,
) -> ModelState<B> {
    ModelState {
        layers: state
            .layers
            .into_iter()
            .map(|layer| LayerState {
                rho: layer.rho.map(Tensor::from_inner),
                packed_rho: layer.packed_rho,
                packed_rho_int8_device: layer.packed_rho_int8_device.map(|state| {
                    burn_dragon_core::PackedRhoInt8DeviceState {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_dragon_stream::{StreamSampleId, StreamStepMetadata};
    use burn_ndarray::NdArray;

    #[test]
    fn stream_controls_reset_and_detach_state() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut state = MultimodalDragonState::<Backend>::new(2, 3);
        state.fusion.position = 5;
        state.cached_vision_tokens = Some(Tensor::<Backend, 3>::zeros([1, 2, 3], &device));
        state.cached_query_tokens = Some(Tensor::<Backend, 3>::zeros([1, 1, 3], &device));
        state.cached_slot_tokens = Some(Tensor::<Backend, 3>::zeros([1, 1, 3], &device));
        state.query_text.as_mut().expect("query state").position = 4;

        let stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 1,
                episode_id: 2,
                segment_id: 3,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 3,
            absolute_time: 3,
        };
        state.apply_stream_controls(
            &stream,
            TbpttWindow::new(4, 2),
            StateCarryPolicy::UntilBoundary,
            FusionCarryPolicy::UntilBoundary,
        );

        assert_eq!(state.fusion.position, 0);
        assert!(state.cached_vision_tokens.is_none());
        assert!(state.cached_query_tokens.is_none());
        assert!(state.cached_slot_tokens.is_none());
        assert_eq!(state.query_text.expect("query state").position, 0);
    }

    #[test]
    fn detach_model_state_preserves_position_and_values() {
        type Backend = NdArray<f32>;
        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0_f32, 2.0, 3.0, 4.0], [1, 1, 2, 2]),
            &Default::default(),
        );
        let mut state = ModelState::<Backend>::new(1);
        state.position = 7;
        state.layers[0].rho = Some(rho);

        let detached = detach_model_state(&state);
        assert_eq!(detached.position, 7);
        let values = detached.layers[0]
            .rho
            .as_ref()
            .expect("rho")
            .clone()
            .to_data()
            .to_vec::<f32>()
            .expect("f32 values");
        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0]);
    }
}
