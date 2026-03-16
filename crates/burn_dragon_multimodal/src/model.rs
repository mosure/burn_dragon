use crate::adapters::{
    TargetTextDragonEncoderAdapter, TargetTextEmbeddingOutput, TargetTextEncoderAdapter,
    TextDragonFusionAdapter, TextFusionAdapter, TextFusionOutput, VisionDragonFusionAdapter,
    VisionFusionAdapter, VisionFusionOutput,
};
use crate::config::VlJepaDragonConfig;
use crate::data::{MultimodalStepMode, VideoLanguageTripletBatch, VisionLanguageTripletBatch};
use crate::state::{MultimodalDragonState, VisionMultimodalState};
use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::activation;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution, Tensor};
use burn_dragon_core::api::recurrent::BDH;
use burn_dragon_vision::api::model::VisionDragon;

#[derive(Clone)]
pub struct FusionInputBatch<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub vision_token_count: usize,
    pub query_token_count: usize,
    pub fusion_slot_count: usize,
}

#[derive(Clone)]
pub struct FusionCoreOutput<B: Backend> {
    pub hidden: Tensor<B, 3>,
    pub predicted_target_embedding: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct VlJepaTargets<B: Backend> {
    pub target_embedding_y: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct FrozenMultimodalCoreSet<B: AutodiffBackend> {
    pub vision_x_encoder: Option<VisionDragon<B::InnerBackend>>,
    pub query_q_encoder: Option<BDH<B::InnerBackend>>,
    pub target_y_encoder: Option<BDH<B::InnerBackend>>,
}

#[derive(Clone)]
pub struct VlJepaForwardOutput<B: Backend> {
    pub fusion_input: FusionInputBatch<B>,
    pub fusion: FusionCoreOutput<B>,
    pub targets: VlJepaTargets<B>,
    pub state: MultimodalDragonState<B>,
}

#[derive(Module, Debug)]
pub struct VlJepaDragon<B: Backend> {
    pub vision_x_encoder: VisionDragonFusionAdapter<B>,
    pub query_q_encoder: TextDragonFusionAdapter<B>,
    pub target_y_encoder: TargetTextDragonEncoderAdapter<B>,
    pub fusion_core: BDH<B>,
    fusion_slot_tokens: Param<Tensor<B, 2>>,
    fusion_modality_tokens: Option<Param<Tensor<B, 2>>>,
    target_predictor_in: Linear<B>,
    target_predictor_out: Linear<B>,
    target_norm: LayerNorm<B>,
    fusion_dim: usize,
    fusion_slot_count: usize,
    query_layer_count: usize,
    fusion_layer_count: usize,
    video_interleave_refine_steps: usize,
    freeze_vision_x_encoder: bool,
    freeze_query_q_encoder: bool,
    freeze_target_y_encoder: bool,
}

impl<B: Backend> VlJepaDragon<B> {
    pub fn new(config: VlJepaDragonConfig, device: &B::Device) -> Self {
        let fusion_slot_count = config.fusion_slots.slot_count.max(1);
        let fusion_slot_tokens = Param::from_tensor(Tensor::<B, 2>::random(
            [fusion_slot_count, config.fusion_dim],
            Distribution::Normal(0.0, 0.02),
            device,
        ));
        let fusion_modality_tokens = if config.fusion_slots.use_modality_type_embeddings {
            Some(Param::from_tensor(Tensor::<B, 2>::random(
                [3, config.fusion_dim],
                Distribution::Normal(0.0, 0.02),
                device,
            )))
        } else {
            None
        };
        Self {
            vision_x_encoder: VisionDragonFusionAdapter::new(&config, device),
            query_q_encoder: TextDragonFusionAdapter::new(
                &config.query_text,
                config.fusion_dim,
                device,
            ),
            target_y_encoder: TargetTextDragonEncoderAdapter::with_kind(
                &config.target_text,
                config.target_dim,
                config.target_text_encoder,
                device,
            ),
            fusion_core: BDH::new(config.fusion.clone(), device),
            fusion_slot_tokens,
            fusion_modality_tokens,
            target_predictor_in: LinearConfig::new(config.fusion_dim * 2, config.fusion_dim * 2)
                .init(device),
            target_predictor_out: LinearConfig::new(config.fusion_dim * 2, config.target_dim)
                .init(device),
            target_norm: LayerNormConfig::new(config.target_dim).init(device),
            fusion_dim: config.fusion_dim,
            fusion_slot_count,
            query_layer_count: config.query_text.n_layer,
            fusion_layer_count: config.fusion.n_layer,
            video_interleave_refine_steps: config.video_interleave_refine_steps,
            freeze_vision_x_encoder: false,
            freeze_query_q_encoder: false,
            freeze_target_y_encoder: false,
        }
    }

    pub fn set_frozen_modalities(
        &mut self,
        freeze_vision_x_encoder: bool,
        freeze_query_q_encoder: bool,
        freeze_target_y_encoder: bool,
    ) {
        self.freeze_vision_x_encoder = freeze_vision_x_encoder;
        self.freeze_query_q_encoder = freeze_query_q_encoder;
        self.freeze_target_y_encoder = freeze_target_y_encoder;
        self.vision_x_encoder
            .set_freeze_encoder_core(freeze_vision_x_encoder);
        self.query_q_encoder
            .set_freeze_encoder_core(freeze_query_q_encoder);
        self.target_y_encoder
            .set_freeze_encoder_core(freeze_target_y_encoder);
    }

    pub fn init_state(&self) -> MultimodalDragonState<B> {
        MultimodalDragonState::new(self.query_layer_count, self.fusion_layer_count)
    }

    fn expand_fusion_slots(&self, batch: usize) -> Tensor<B, 3> {
        self.fusion_slot_tokens
            .val()
            .clone()
            .unsqueeze_dim::<3>(0)
            .repeat_dim(0, batch)
    }

    fn apply_modality_embeddings(
        &self,
        vision: Tensor<B, 3>,
        query: Tensor<B, 3>,
        slots: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let Some(embeddings) = self.fusion_modality_tokens.as_ref() else {
            return (vision, query, slots);
        };
        let modality = embeddings.val();
        let vision = vision
            + modality
                .clone()
                .slice([0..1, 0..self.fusion_dim])
                .unsqueeze_dim::<3>(0);
        let query = query
            + modality
                .clone()
                .slice([1..2, 0..self.fusion_dim])
                .unsqueeze_dim::<3>(0);
        let slots = slots
            + modality
                .clone()
                .slice([2..3, 0..self.fusion_dim])
                .unsqueeze_dim::<3>(0);
        (vision, query, slots)
    }

    fn resolve_fusion_slots(&self, state: &MultimodalDragonState<B>, batch: usize) -> Tensor<B, 3> {
        let Some(slots) = state.cached_slot_tokens.as_ref() else {
            return self.expand_fusion_slots(batch);
        };
        let [slot_batch, slot_count, slot_dim] = slots.shape().dims::<3>();
        if slot_batch == batch
            && slot_count == self.fusion_slot_count
            && slot_dim == self.fusion_dim
        {
            slots.clone()
        } else {
            self.expand_fusion_slots(batch)
        }
    }

    pub fn pack_fusion_inputs(
        &self,
        vision: &VisionFusionOutput<B, VisionMultimodalState<B>>,
        query: &TextFusionOutput<B, burn_dragon_core::api::state::ModelState<B>>,
        state: &MultimodalDragonState<B>,
    ) -> FusionInputBatch<B> {
        let batch = vision.fusion_tokens.shape().dims::<3>()[0];
        let vision_tokens = Tensor::cat(
            vec![
                vision.summary_token.clone().unsqueeze_dim::<3>(1),
                vision.fusion_tokens.clone(),
            ],
            1,
        );
        let query_tokens = Tensor::cat(
            vec![
                query.summary_token.clone().unsqueeze_dim::<3>(1),
                query.fusion_tokens.clone(),
            ],
            1,
        );
        let slots = self.resolve_fusion_slots(state, batch);
        let (vision_tokens, query_tokens, slot_tokens) =
            self.apply_modality_embeddings(vision_tokens, query_tokens, slots);
        FusionInputBatch {
            vision_token_count: vision_tokens.shape().dims::<3>()[1],
            query_token_count: query_tokens.shape().dims::<3>()[1],
            fusion_slot_count: slot_tokens.shape().dims::<3>()[1],
            tokens: Tensor::cat(vec![vision_tokens, query_tokens, slot_tokens], 1),
        }
    }

    fn fusion_predict(
        &self,
        packed: FusionInputBatch<B>,
        mut state: MultimodalDragonState<B>,
    ) -> (FusionCoreOutput<B>, MultimodalDragonState<B>) {
        let (hidden, _logits) = self
            .fusion_core
            .forward_with_hidden_and_state_embedded(packed.tokens.clone(), &mut state.fusion);
        let [batch, total_tokens, dim] = hidden.shape().dims::<3>();
        let query_start = packed.vision_token_count;
        let slot_start = query_start + packed.query_token_count;
        state.cached_vision_tokens =
            Some(
                packed
                    .tokens
                    .clone()
                    .slice([0..batch, 0..query_start, 0..dim]),
            );
        state.cached_query_tokens =
            Some(
                packed
                    .tokens
                    .clone()
                    .slice([0..batch, query_start..slot_start, 0..dim]),
            );
        let slot_hidden = hidden
            .clone()
            .slice([0..batch, slot_start..total_tokens, 0..dim]);
        state.cached_slot_tokens = Some(slot_hidden.clone());
        let slot_summary = slot_hidden.mean_dim(1).reshape([batch, dim]);
        let context_summary = hidden.clone().mean_dim(1).reshape([batch, dim]);
        let predictor_input = Tensor::cat(vec![slot_summary, context_summary], 1);
        let predictor_hidden = activation::gelu(self.target_predictor_in.forward(predictor_input));
        let predicted_target_embedding = self
            .target_norm
            .forward(self.target_predictor_out.forward(predictor_hidden));
        (
            FusionCoreOutput {
                hidden,
                predicted_target_embedding,
            },
            state,
        )
    }

    pub fn predict_fusion_from_packed(
        &self,
        packed: FusionInputBatch<B>,
        state: MultimodalDragonState<B>,
    ) -> (FusionCoreOutput<B>, MultimodalDragonState<B>) {
        self.fusion_predict(packed, state)
    }

    pub fn replace_fusion_core(&mut self, fusion_core: BDH<B>) {
        self.fusion_core = fusion_core;
    }

    pub fn encode_target_bank(
        &self,
        target_tokens: Tensor<B, 2, burn::tensor::Int>,
        target_mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 2> {
        self.target_y_encoder
            .encode_y((target_tokens, target_mask))
            .target_embedding
    }

    pub fn forward_x_q_y(
        &self,
        batch: VisionLanguageTripletBatch<B>,
        mut state: MultimodalDragonState<B>,
        mode: MultimodalStepMode,
    ) -> VlJepaForwardOutput<B> {
        let vision = self
            .vision_x_encoder
            .observe_x(batch.vision_x, state.vision.take(), mode);
        state.vision = vision.state.clone();
        let query = self.query_q_encoder.observe_q(
            (batch.query_q_tokens, batch.query_q_mask),
            state.query_text.take(),
        );
        state.query_text = query.state.clone();
        let packed = self.pack_fusion_inputs(&vision, &query, &state);
        let (fusion, state) = self.fusion_predict(packed.clone(), state);
        let target: TargetTextEmbeddingOutput<B> = self
            .target_y_encoder
            .encode_y((batch.target_y_tokens, batch.target_y_mask));
        VlJepaForwardOutput {
            fusion_input: packed,
            fusion,
            targets: VlJepaTargets {
                target_embedding_y: target.target_embedding,
            },
            state,
        }
    }

    pub fn forward_video_x_q_y(
        &self,
        batch: VideoLanguageTripletBatch<B>,
        mut state: MultimodalDragonState<B>,
    ) -> VlJepaForwardOutput<B> {
        let query = self.query_q_encoder.observe_q(
            (batch.query_q_tokens, batch.query_q_mask),
            state.query_text.take(),
        );
        state.query_text = query.state.clone();
        let [batch_size, frames, channels, height, width] = batch.video_x.shape().dims::<5>();
        let mut last_packed = None;
        let mut last_fusion = None;
        for frame_index in 0..frames {
            let frame = batch
                .video_x
                .clone()
                .slice([
                    0..batch_size,
                    frame_index..frame_index + 1,
                    0..channels,
                    0..height,
                    0..width,
                ])
                .reshape([batch_size, channels, height, width]);
            let vision = self.vision_x_encoder.observe_x(
                frame,
                state.vision.take(),
                MultimodalStepMode::Observe,
            );
            state.vision = vision.state.clone();
            let packed = self.pack_fusion_inputs(&vision, &query, &state);
            let (fusion, next_state) = self.fusion_predict(packed.clone(), state);
            state = next_state;
            last_packed = Some(packed);
            last_fusion = Some(fusion);
            for _ in 0..self.video_interleave_refine_steps {
                let Some((fusion, next_state)) = self.refine(state.clone()) else {
                    break;
                };
                state = next_state;
                last_fusion = Some(fusion);
            }
        }
        let target = self
            .target_y_encoder
            .encode_y((batch.target_y_tokens, batch.target_y_mask));
        VlJepaForwardOutput {
            fusion_input: last_packed.expect("video must contain at least one frame"),
            fusion: last_fusion.expect("video must contain at least one frame"),
            targets: VlJepaTargets {
                target_embedding_y: target.target_embedding,
            },
            state,
        }
    }

    pub fn refine(
        &self,
        mut state: MultimodalDragonState<B>,
    ) -> Option<(FusionCoreOutput<B>, MultimodalDragonState<B>)> {
        if let Some(vision_state) = state.vision.clone()
            && let Some(refined_vision) = self.vision_x_encoder.refine_state(vision_state)
        {
            state.vision = refined_vision.state.clone();
            state.cached_vision_tokens = Some(refined_vision.fusion_tokens);
        }
        let vision_tokens = state.cached_vision_tokens.clone()?;
        let query_tokens = state.cached_query_tokens.clone()?;
        let slot_tokens = state.cached_slot_tokens.clone()?;
        let batch = vision_tokens.shape().dims::<3>()[0];
        let vision_token_count = vision_tokens.shape().dims::<3>()[1];
        let query_token_count = query_tokens.shape().dims::<3>()[1];
        let slot_token_count = slot_tokens.shape().dims::<3>()[1];
        let total_token_count = vision_token_count + query_token_count + slot_token_count;
        let packed = FusionInputBatch {
            vision_token_count,
            query_token_count,
            fusion_slot_count: slot_token_count,
            tokens: Tensor::cat(vec![vision_tokens, query_tokens, slot_tokens], 1).reshape([
                batch,
                total_token_count,
                self.fusion_dim,
            ]),
        };
        Some(self.fusion_predict(packed, state))
    }

    pub fn predict(
        &self,
        state: MultimodalDragonState<B>,
    ) -> Option<(FusionCoreOutput<B>, MultimodalDragonState<B>)> {
        self.refine(state)
    }
}

impl<B: AutodiffBackend> VlJepaDragon<B> {
    pub fn frozen_core_set(&self) -> FrozenMultimodalCoreSet<B> {
        FrozenMultimodalCoreSet {
            vision_x_encoder: self
                .freeze_vision_x_encoder
                .then(|| self.vision_x_encoder.valid_encoder_core()),
            query_q_encoder: self
                .freeze_query_q_encoder
                .then(|| self.query_q_encoder.valid_encoder_core()),
            target_y_encoder: self
                .freeze_target_y_encoder
                .then(|| self.target_y_encoder.valid_encoder_core()),
        }
    }

    pub fn forward_x_q_y_frozen_aware(
        &self,
        batch: VisionLanguageTripletBatch<B>,
        state: MultimodalDragonState<B>,
        mode: MultimodalStepMode,
    ) -> VlJepaForwardOutput<B> {
        self.forward_x_q_y_with_frozen_cores(None, batch, state, mode)
    }

    pub fn forward_x_q_y_with_frozen_cores(
        &self,
        frozen_cores: Option<&FrozenMultimodalCoreSet<B>>,
        batch: VisionLanguageTripletBatch<B>,
        mut state: MultimodalDragonState<B>,
        mode: MultimodalStepMode,
    ) -> VlJepaForwardOutput<B> {
        let vision = if self.freeze_vision_x_encoder {
            let valid_encoder = frozen_cores
                .and_then(|cores| cores.vision_x_encoder.as_ref())
                .expect("frozen vision core set missing persistent encoder");
            self.vision_x_encoder.observe_x_frozen_core(
                valid_encoder,
                batch.vision_x,
                state.vision.take(),
                mode,
            )
        } else {
            self.vision_x_encoder
                .observe_x(batch.vision_x, state.vision.take(), mode)
        };
        state.vision = vision.state.clone();
        let query = if self.freeze_query_q_encoder {
            let valid_encoder = frozen_cores
                .and_then(|cores| cores.query_q_encoder.as_ref())
                .expect("frozen text query core set missing persistent encoder");
            self.query_q_encoder.encode_q_frozen_core(
                valid_encoder,
                batch.query_q_tokens,
                batch.query_q_mask,
                state.query_text.take(),
            )
        } else {
            self.query_q_encoder.observe_q(
                (batch.query_q_tokens, batch.query_q_mask),
                state.query_text.take(),
            )
        };
        state.query_text = query.state.clone();
        let packed = self.pack_fusion_inputs(&vision, &query, &state);
        if self.freeze_vision_x_encoder {
            state.vision = None;
        }
        let (fusion, state) = self.fusion_predict(packed.clone(), state);
        let target = if self.freeze_target_y_encoder {
            let valid_encoder = frozen_cores
                .and_then(|cores| cores.target_y_encoder.as_ref())
                .expect("frozen target text core set missing persistent encoder");
            self.target_y_encoder
                .encode_y_frozen_core(valid_encoder, (batch.target_y_tokens, batch.target_y_mask))
        } else {
            self.target_y_encoder
                .encode_y((batch.target_y_tokens, batch.target_y_mask))
        };
        VlJepaForwardOutput {
            fusion_input: packed,
            fusion,
            targets: VlJepaTargets {
                target_embedding_y: target.target_embedding,
            },
            state,
        }
    }

    pub fn forward_video_x_q_y_frozen_aware(
        &self,
        batch: VideoLanguageTripletBatch<B>,
        state: MultimodalDragonState<B>,
    ) -> VlJepaForwardOutput<B> {
        self.forward_video_x_q_y_with_frozen_cores(None, batch, state)
    }

    pub fn forward_video_x_q_y_with_frozen_cores(
        &self,
        frozen_cores: Option<&FrozenMultimodalCoreSet<B>>,
        batch: VideoLanguageTripletBatch<B>,
        mut state: MultimodalDragonState<B>,
    ) -> VlJepaForwardOutput<B> {
        let query = if self.freeze_query_q_encoder {
            let valid_encoder = frozen_cores
                .and_then(|cores| cores.query_q_encoder.as_ref())
                .expect("frozen text query core set missing persistent encoder");
            self.query_q_encoder.encode_q_frozen_core(
                valid_encoder,
                batch.query_q_tokens,
                batch.query_q_mask,
                state.query_text.take(),
            )
        } else {
            self.query_q_encoder.observe_q(
                (batch.query_q_tokens, batch.query_q_mask),
                state.query_text.take(),
            )
        };
        state.query_text = query.state.clone();
        let [batch_size, frames, channels, height, width] = batch.video_x.shape().dims::<5>();
        let mut last_packed = None;
        let mut last_fusion = None;
        for frame_index in 0..frames {
            let frame = batch
                .video_x
                .clone()
                .slice([
                    0..batch_size,
                    frame_index..frame_index + 1,
                    0..channels,
                    0..height,
                    0..width,
                ])
                .reshape([batch_size, channels, height, width]);
            let vision = if self.freeze_vision_x_encoder {
                let valid_encoder = frozen_cores
                    .and_then(|cores| cores.vision_x_encoder.as_ref())
                    .expect("frozen vision core set missing persistent encoder");
                self.vision_x_encoder.observe_x_frozen_core(
                    valid_encoder,
                    frame,
                    state.vision.take(),
                    MultimodalStepMode::Observe,
                )
            } else {
                self.vision_x_encoder.observe_x(
                    frame,
                    state.vision.take(),
                    MultimodalStepMode::Observe,
                )
            };
            state.vision = vision.state.clone();
            let packed = self.pack_fusion_inputs(&vision, &query, &state);
            let (fusion, next_state) = self.fusion_predict(packed.clone(), state);
            state = next_state;
            last_packed = Some(packed);
            last_fusion = Some(fusion);
            for _ in 0..self.video_interleave_refine_steps {
                let Some((fusion, next_state)) =
                    self.refine_with_frozen_cores(frozen_cores, state.clone())
                else {
                    break;
                };
                state = next_state;
                last_fusion = Some(fusion);
            }
        }
        if self.freeze_vision_x_encoder {
            state.vision = None;
        }
        let target = if self.freeze_target_y_encoder {
            let valid_encoder = frozen_cores
                .and_then(|cores| cores.target_y_encoder.as_ref())
                .expect("frozen target text core set missing persistent encoder");
            self.target_y_encoder
                .encode_y_frozen_core(valid_encoder, (batch.target_y_tokens, batch.target_y_mask))
        } else {
            self.target_y_encoder
                .encode_y((batch.target_y_tokens, batch.target_y_mask))
        };
        VlJepaForwardOutput {
            fusion_input: last_packed.expect("video must contain at least one frame"),
            fusion: last_fusion.expect("video must contain at least one frame"),
            targets: VlJepaTargets {
                target_embedding_y: target.target_embedding,
            },
            state,
        }
    }

    pub fn refine_frozen_aware(
        &self,
        state: MultimodalDragonState<B>,
    ) -> Option<(FusionCoreOutput<B>, MultimodalDragonState<B>)> {
        self.refine_with_frozen_cores(None, state)
    }

    pub fn refine_with_frozen_cores(
        &self,
        _frozen_cores: Option<&FrozenMultimodalCoreSet<B>>,
        mut state: MultimodalDragonState<B>,
    ) -> Option<(FusionCoreOutput<B>, MultimodalDragonState<B>)> {
        if !self.freeze_vision_x_encoder
            && let Some(vision_state) = state.vision.clone()
        {
            let refined = self.vision_x_encoder.refine_state(vision_state);
            if let Some(refined_vision) = refined {
                state.vision = refined_vision.state.clone();
                state.cached_vision_tokens = Some(refined_vision.fusion_tokens);
            }
        }
        let vision_tokens = state.cached_vision_tokens.clone()?;
        let query_tokens = state.cached_query_tokens.clone()?;
        let slot_tokens = state.cached_slot_tokens.clone()?;
        let batch = vision_tokens.shape().dims::<3>()[0];
        let vision_token_count = vision_tokens.shape().dims::<3>()[1];
        let query_token_count = query_tokens.shape().dims::<3>()[1];
        let slot_token_count = slot_tokens.shape().dims::<3>()[1];
        let total_token_count = vision_token_count + query_token_count + slot_token_count;
        let packed = FusionInputBatch {
            vision_token_count,
            query_token_count,
            fusion_slot_count: slot_token_count,
            tokens: Tensor::cat(vec![vision_tokens, query_tokens, slot_tokens], 1).reshape([
                batch,
                total_token_count,
                self.fusion_dim,
            ]),
        };
        Some(self.fusion_predict(packed, state))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FusionSlotConfig;
    use crate::config::VlJepaDragonConfig;
    use burn::tensor::Int;
    use burn_dragon_core::api::config::BDHConfig;
    use burn_dragon_core::api::state::ModelState;
    use burn_dragon_vision::api::model::VisionDragonConfig;
    use burn_ndarray::NdArray;

    #[test]
    fn forward_x_q_y_runs_and_threads_separate_states() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.steps = 1;
        config.vision.backbone = burn_dragon_vision::api::model::VisionBackboneKind::Dense;
        config.query_text.n_embd = 32;
        config.target_text.n_embd = 32;
        config.fusion.n_embd = 32;
        config.fusion_dim = 32;
        config.target_dim = 32;
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            target_y_mask: None,
        };
        let output = model.forward_x_q_y(batch, model.init_state(), MultimodalStepMode::Observe);
        assert_eq!(
            output.fusion.predicted_target_embedding.shape().dims(),
            [2, 32]
        );
        assert_eq!(output.targets.target_embedding_y.shape().dims(), [2, 32]);
        assert!(output.state.query_text.is_some());
        assert!(output.state.fusion.position > 0);
    }

    #[test]
    fn refine_advances_fusion_state_without_rethreading_query_state() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_embd = 16;
        config.target_text.n_embd = 16;
        config.fusion.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_head = 2;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;
        let model = VlJepaDragon::<Backend>::new(config, &device);
        let batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let output = model.forward_x_q_y(batch, model.init_state(), MultimodalStepMode::Observe);
        let query_position = output
            .state
            .query_text
            .as_ref()
            .expect("query state")
            .position;
        let fusion_position = output.state.fusion.position;
        let (_, refined_state) = model.refine(output.state).expect("refine output");
        assert_eq!(
            refined_state
                .query_text
                .as_ref()
                .expect("query state")
                .position,
            query_position
        );
        assert!(refined_state.fusion.position > fusion_position);
    }

    #[test]
    fn forward_video_x_q_y_threads_fusion_state_across_frames() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_embd = 16;
        config.target_text.n_embd = 16;
        config.fusion.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_head = 2;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;
        let model = VlJepaDragon::<Backend>::new(config, &device);
        let image_batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let single =
            model.forward_x_q_y(image_batch, model.init_state(), MultimodalStepMode::Observe);
        let video_batch = VideoLanguageTripletBatch {
            video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let video = model.forward_video_x_q_y(video_batch, model.init_state());
        assert!(video.state.fusion.position > single.state.fusion.position);
    }

    #[test]
    fn video_interleave_refine_steps_advance_fusion_state_between_frames() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut base = VlJepaDragonConfig::default();
        base.vision.embed_dim = 16;
        base.vision.projection_dim = 16;
        base.vision.projection_hidden_dim = 16;
        base.vision.steps = 1;
        base.query_text.n_embd = 16;
        base.target_text.n_embd = 16;
        base.fusion.n_embd = 16;
        base.query_text.n_head = 2;
        base.target_text.n_head = 2;
        base.fusion.n_head = 2;
        base.fusion_dim = 16;
        base.target_dim = 16;
        let mut refined = base.clone();
        refined.video_interleave_refine_steps = 2;
        let plain_model = VlJepaDragon::<Backend>::new(base, &device);
        let refined_model = VlJepaDragon::<Backend>::new(refined, &device);
        let video_batch = VideoLanguageTripletBatch {
            video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let plain = plain_model.forward_video_x_q_y(video_batch.clone(), plain_model.init_state());
        let refined = refined_model.forward_video_x_q_y(video_batch, refined_model.init_state());
        assert!(refined.state.fusion.position > plain.state.fusion.position);
    }

    #[test]
    fn single_frame_video_matches_image_observation_path() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_embd = 16;
        config.target_text.n_embd = 16;
        config.fusion.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_head = 2;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;
        let model = VlJepaDragon::<Backend>::new(config, &device);
        let vision_x = Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device);
        let image_batch = VisionLanguageTripletBatch {
            vision_x: vision_x.clone(),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let video_batch = VideoLanguageTripletBatch {
            video_x: vision_x.unsqueeze_dim::<5>(1),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: None,
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: None,
        };
        let image =
            model.forward_x_q_y(image_batch, model.init_state(), MultimodalStepMode::Observe);
        let video = model.forward_video_x_q_y(video_batch, model.init_state());
        let image_values = image
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let video_values = video
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(image_values, video_values);
    }

    #[test]
    fn pack_fusion_inputs_reuses_cached_slot_tokens_when_available() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let config = VlJepaDragonConfig {
            fusion_dim: 8,
            target_dim: 8,
            fusion_slots: FusionSlotConfig {
                slot_count: 2,
                use_modality_type_embeddings: false,
            },
            vision: VisionDragonConfig {
                embed_dim: 8,
                projection_dim: 8,
                projection_hidden_dim: 8,
                ..Default::default()
            },
            query_text: BDHConfig {
                n_embd: 8,
                n_head: 2,
                ..Default::default()
            },
            target_text: BDHConfig {
                n_embd: 8,
                n_head: 2,
                ..Default::default()
            },
            fusion: BDHConfig {
                n_embd: 8,
                n_head: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let model = VlJepaDragon::<Backend>::new(config, &device);
        let mut state = model.init_state();
        let carried_slots = Tensor::<Backend, 3>::ones([1, 2, 8], &device);
        state.cached_slot_tokens = Some(carried_slots.clone());
        let vision = VisionFusionOutput {
            fusion_tokens: Tensor::<Backend, 3>::zeros([1, 3, 8], &device),
            summary_token: Tensor::<Backend, 2>::zeros([1, 8], &device),
            state: None,
        };
        let query = TextFusionOutput {
            fusion_tokens: Tensor::<Backend, 3>::zeros([1, 4, 8], &device),
            summary_token: Tensor::<Backend, 2>::zeros([1, 8], &device),
            state: Some(ModelState::new(model.query_layer_count)),
        };
        let packed = model.pack_fusion_inputs(&vision, &query, &state);
        let slot_start = packed.vision_token_count + packed.query_token_count;
        let slot_end = slot_start + packed.fusion_slot_count;
        let carried = packed
            .tokens
            .slice([0..1, slot_start..slot_end, 0..8])
            .into_data()
            .to_vec::<f32>()
            .unwrap();
        let expected = carried_slots.into_data().to_vec::<f32>().unwrap();
        assert_eq!(carried, expected);
    }
}
