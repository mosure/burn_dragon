use crate::config::{TargetTextEncoderKind, VlJepaDragonConfig};
use crate::data::MultimodalStepMode;
use crate::state::{
    VisionMultimodalState, detach_model_state, model_state_from_inner, model_state_inner,
};
use burn::module::{AutodiffModule, Module};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor};
use burn_dragon_core::api::config::BDHConfig;
use burn_dragon_core::api::recurrent::BDH;
use burn_dragon_core::api::state::{ModelState, StructuredStepMode};
use burn_dragon_vision::api::model::{VisionBackboneKind, VisionDragon};

#[derive(Clone)]
pub struct VisionFusionOutput<B: Backend, S> {
    pub fusion_tokens: Tensor<B, 3>,
    pub summary_token: Tensor<B, 2>,
    pub state: Option<S>,
}

#[derive(Clone)]
pub struct TextFusionOutput<B: Backend, S> {
    pub fusion_tokens: Tensor<B, 3>,
    pub summary_token: Tensor<B, 2>,
    pub state: Option<S>,
}

#[derive(Clone)]
pub struct TargetTextEmbeddingOutput<B: Backend> {
    pub target_embedding: Tensor<B, 2>,
}

pub trait VisionFusionAdapter<B: Backend> {
    type State;
    type Input;

    fn observe_x(
        &self,
        input: Self::Input,
        state: Option<Self::State>,
        mode: MultimodalStepMode,
    ) -> VisionFusionOutput<B, Self::State>;
}

pub trait TextFusionAdapter<B: Backend> {
    type State;
    type Input;

    fn observe_q(
        &self,
        input: Self::Input,
        state: Option<Self::State>,
    ) -> TextFusionOutput<B, Self::State>;
}

pub trait TargetTextEncoderAdapter<B: Backend> {
    type Input;

    fn encode_y(&self, input: Self::Input) -> TargetTextEmbeddingOutput<B>;
}

fn masked_mean<B: Backend>(
    hidden: Tensor<B, 3>,
    mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
) -> Tensor<B, 2> {
    let [batch, _time, dim] = hidden.shape().dims::<3>();
    match mask {
        Some(mask) => {
            let mask = mask.float().unsqueeze_dim::<3>(2);
            let numer = (hidden * mask.clone()).sum_dim(1).reshape([batch, dim]);
            let denom = mask.sum_dim(1).reshape([batch, 1]).clamp_min(1e-6);
            numer / denom
        }
        None => hidden.mean_dim(1).reshape([batch, dim]),
    }
}

#[derive(Module, Debug)]
pub struct VisionDragonFusionAdapter<B: Backend> {
    encoder: VisionDragon<B>,
    token_projector: Linear<B>,
    summary_projector: Linear<B>,
    summary_norm: LayerNorm<B>,
    #[module(ignore)]
    backbone: VisionBackboneKind,
    fusion_dim: usize,
    rollout_steps: usize,
    backprop_steps: usize,
    freeze_encoder_core: bool,
    passthrough_projection: bool,
    force_projection: bool,
}

impl<B: Backend> VisionDragonFusionAdapter<B> {
    pub fn new(config: &VlJepaDragonConfig, device: &B::Device) -> Self {
        let encoder = VisionDragon::new(config.vision.clone(), device);
        let token_projector =
            LinearConfig::new(config.vision.embed_dim, config.fusion_dim).init(device);
        let summary_projector =
            LinearConfig::new(config.vision.embed_dim, config.fusion_dim).init(device);
        let summary_norm = LayerNormConfig::new(config.fusion_dim).init(device);
        Self {
            encoder,
            token_projector,
            summary_projector,
            summary_norm,
            backbone: config.vision.backbone,
            fusion_dim: config.fusion_dim,
            rollout_steps: config.vision_rollout_steps.max(1),
            backprop_steps: config.vision_backprop_steps.max(1),
            freeze_encoder_core: false,
            passthrough_projection: config.vision.embed_dim == config.fusion_dim,
            force_projection: false,
        }
    }

    pub fn set_freeze_encoder_core(&mut self, freeze: bool) {
        if freeze && !self.freeze_encoder_core {
            self.encoder = self.encoder.clone().no_grad();
        }
        self.freeze_encoder_core = freeze;
    }

    pub fn set_force_projection(&mut self, force_projection: bool) {
        self.force_projection = force_projection;
    }

    fn project_output(
        &self,
        patch_tokens: Tensor<B, 3>,
        summary_token: Tensor<B, 2>,
        state: Option<VisionMultimodalState<B>>,
    ) -> VisionFusionOutput<B, VisionMultimodalState<B>> {
        let [batch, token_count, dim] = patch_tokens.shape().dims::<3>();
        let patch_tokens = if self.freeze_encoder_core {
            patch_tokens.detach()
        } else {
            patch_tokens
        };
        let summary_token = if self.freeze_encoder_core {
            summary_token.detach()
        } else {
            summary_token
        };
        let state = if self.freeze_encoder_core {
            state.as_ref().map(VisionMultimodalState::detach)
        } else {
            state
        };
        let use_passthrough_projection = self.passthrough_projection && !self.force_projection;
        let fusion_tokens = if use_passthrough_projection {
            patch_tokens
        } else {
            self.token_projector
                .forward(patch_tokens.reshape([batch * token_count, dim]))
                .reshape([batch, token_count, self.fusion_dim])
        };
        let summary_token = if use_passthrough_projection {
            summary_token
        } else {
            self.summary_norm
                .forward(self.summary_projector.forward(summary_token))
        };
        VisionFusionOutput {
            fusion_tokens,
            summary_token,
            state,
        }
    }

    fn observe_images(
        &self,
        images: Tensor<B, 4>,
        state: Option<VisionMultimodalState<B>>,
        mode: MultimodalStepMode,
    ) -> VisionFusionOutput<B, VisionMultimodalState<B>> {
        match (self.backbone, state) {
            (VisionBackboneKind::Pyramid, Some(VisionMultimodalState::Pyramid(state))) => {
                let patch = self.encoder.embed_images(images).tokens;
                let next_state = match mode {
                    MultimodalStepMode::Observe => self.encoder.observe_pyramid_state(
                        state,
                        patch,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Refine => self.encoder.refine_pyramid_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Predict => self.encoder.predict_pyramid_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                };
                self.project_output(
                    self.encoder.pyramid_patch_tokens(&next_state),
                    self.encoder.pyramid_summary(&next_state),
                    Some(VisionMultimodalState::Pyramid(next_state)),
                )
            }
            (VisionBackboneKind::Pyramid, _) => {
                let patch = self.encoder.embed_images(images).tokens;
                let state = self.encoder.pyramid_state_from_patch_tokens(patch);
                let next_state = self.encoder.forward_pyramid_state_rollout_mode_unbounded(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                    StructuredStepMode::Observe,
                );
                self.project_output(
                    self.encoder.pyramid_patch_tokens(&next_state),
                    self.encoder.pyramid_summary(&next_state),
                    Some(VisionMultimodalState::Pyramid(next_state)),
                )
            }
            _ => {
                let output = self.encoder.forward_images_embed_steps_rollout_unbounded(
                    images,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                self.project_output(output.patch_tokens, output.cls_token, None)
            }
        }
    }

    pub fn observe_video_x(
        &self,
        video: Tensor<B, 5>,
        state: Option<VisionMultimodalState<B>>,
    ) -> VisionFusionOutput<B, VisionMultimodalState<B>> {
        let [batch, frames, channels, height, width] = video.shape().dims::<5>();
        let mut carried = state;
        let mut last_tokens = None;
        let mut last_summary = None;
        for frame in 0..frames {
            let image = video
                .clone()
                .slice([0..batch, frame..frame + 1, 0..channels, 0..height, 0..width])
                .reshape([batch, channels, height, width]);
            let observed = self.observe_images(image, carried, MultimodalStepMode::Observe);
            carried = observed.state;
            last_tokens = Some(observed.fusion_tokens);
            last_summary = Some(observed.summary_token);
        }
        VisionFusionOutput {
            fusion_tokens: last_tokens.expect("video must contain at least one frame"),
            summary_token: last_summary.expect("video must contain at least one frame"),
            state: carried,
        }
    }

    pub fn refine_state(
        &self,
        state: VisionMultimodalState<B>,
    ) -> Option<VisionFusionOutput<B, VisionMultimodalState<B>>> {
        match (self.backbone, state) {
            (VisionBackboneKind::Pyramid, VisionMultimodalState::Pyramid(state)) => {
                let next_state = self.encoder.refine_pyramid_state(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                Some(self.project_output(
                    self.encoder.pyramid_patch_tokens(&next_state),
                    self.encoder.pyramid_summary(&next_state),
                    Some(VisionMultimodalState::Pyramid(next_state)),
                ))
            }
            (VisionBackboneKind::Cellular, VisionMultimodalState::Cellular(state)) => {
                let next_state = self.encoder.refine_cellular_state(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                let output = self.encoder.forward_cellular_state_embed(&next_state);
                Some(self.project_output(
                    output.patch_tokens,
                    output.cls_token,
                    Some(VisionMultimodalState::Cellular(next_state)),
                ))
            }
            _ => None,
        }
    }

    pub fn replace_encoder(&mut self, encoder: VisionDragon<B>) {
        self.encoder = encoder;
    }
}

impl<B: AutodiffBackend> VisionDragonFusionAdapter<B> {
    pub fn valid_encoder_core(&self) -> VisionDragon<B::InnerBackend> {
        self.encoder.valid()
    }

    pub fn observe_x_frozen_core(
        &self,
        valid_encoder: &VisionDragon<B::InnerBackend>,
        images: Tensor<B, 4>,
        state: Option<VisionMultimodalState<B>>,
        mode: MultimodalStepMode,
    ) -> VisionFusionOutput<B, VisionMultimodalState<B>> {
        let images = images.inner();
        let state = state.as_ref().map(VisionMultimodalState::inner);
        match (self.backbone, state) {
            (VisionBackboneKind::Pyramid, Some(VisionMultimodalState::Pyramid(state))) => {
                let patch = valid_encoder.embed_images(images).tokens;
                let next_state = match mode {
                    MultimodalStepMode::Observe => valid_encoder.observe_pyramid_state(
                        state,
                        patch,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Refine => valid_encoder.refine_pyramid_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Predict => valid_encoder.predict_pyramid_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                };
                self.project_output(
                    Tensor::from_inner(valid_encoder.pyramid_patch_tokens(&next_state)),
                    Tensor::from_inner(valid_encoder.pyramid_summary(&next_state)),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Pyramid(next_state),
                    )),
                )
            }
            (VisionBackboneKind::Pyramid, _) => {
                let patch = valid_encoder.embed_images(images).tokens;
                let state = valid_encoder.pyramid_state_from_patch_tokens(patch);
                let next_state = valid_encoder.forward_pyramid_state_rollout_mode_unbounded(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                    StructuredStepMode::Observe,
                );
                self.project_output(
                    Tensor::from_inner(valid_encoder.pyramid_patch_tokens(&next_state)),
                    Tensor::from_inner(valid_encoder.pyramid_summary(&next_state)),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Pyramid(next_state),
                    )),
                )
            }
            (VisionBackboneKind::Cellular, Some(VisionMultimodalState::Cellular(state))) => {
                let next_state = match mode {
                    MultimodalStepMode::Observe => valid_encoder.observe_cellular_state(
                        state,
                        valid_encoder.embed_images(images).tokens,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Refine => valid_encoder.refine_cellular_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                    MultimodalStepMode::Predict => valid_encoder.predict_cellular_state(
                        state,
                        self.rollout_steps,
                        self.backprop_steps,
                    ),
                };
                let output = valid_encoder.forward_cellular_state_embed(&next_state);
                self.project_output(
                    Tensor::from_inner(output.patch_tokens),
                    Tensor::from_inner(output.cls_token),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Cellular(next_state),
                    )),
                )
            }
            (VisionBackboneKind::Cellular, _) => {
                let patch = valid_encoder.embed_images(images).tokens;
                let state = valid_encoder.cellular_state_from_tokens(patch);
                let next_state = valid_encoder.forward_cellular_state_rollout_mode_unbounded(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                    StructuredStepMode::Observe,
                );
                let output = valid_encoder.forward_cellular_state_embed(&next_state);
                self.project_output(
                    Tensor::from_inner(output.patch_tokens),
                    Tensor::from_inner(output.cls_token),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Cellular(next_state),
                    )),
                )
            }
            _ => {
                let output = valid_encoder.forward_images_embed_steps_rollout_unbounded(
                    images,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                self.project_output(
                    Tensor::from_inner(output.patch_tokens),
                    Tensor::from_inner(output.cls_token),
                    None,
                )
            }
        }
    }

    pub fn refine_state_frozen_core(
        &self,
        valid_encoder: &VisionDragon<B::InnerBackend>,
        state: VisionMultimodalState<B>,
    ) -> Option<VisionFusionOutput<B, VisionMultimodalState<B>>> {
        match (self.backbone, state.inner()) {
            (VisionBackboneKind::Pyramid, VisionMultimodalState::Pyramid(state)) => {
                let next_state = valid_encoder.refine_pyramid_state(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                Some(self.project_output(
                    Tensor::from_inner(valid_encoder.pyramid_patch_tokens(&next_state)),
                    Tensor::from_inner(valid_encoder.pyramid_summary(&next_state)),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Pyramid(next_state),
                    )),
                ))
            }
            (VisionBackboneKind::Cellular, VisionMultimodalState::Cellular(state)) => {
                let next_state = valid_encoder.refine_cellular_state(
                    state,
                    self.rollout_steps,
                    self.backprop_steps,
                );
                let output = valid_encoder.forward_cellular_state_embed(&next_state);
                Some(self.project_output(
                    Tensor::from_inner(output.patch_tokens),
                    Tensor::from_inner(output.cls_token),
                    Some(VisionMultimodalState::from_inner(
                        VisionMultimodalState::Cellular(next_state),
                    )),
                ))
            }
            _ => None,
        }
    }
}

impl<B: Backend> VisionFusionAdapter<B> for VisionDragonFusionAdapter<B> {
    type State = VisionMultimodalState<B>;
    type Input = Tensor<B, 4>;

    fn observe_x(
        &self,
        input: Self::Input,
        state: Option<Self::State>,
        mode: MultimodalStepMode,
    ) -> VisionFusionOutput<B, Self::State> {
        self.observe_images(input, state, mode)
    }
}

#[derive(Module, Debug)]
pub struct TextDragonFusionAdapter<B: Backend> {
    encoder: BDH<B>,
    token_projector: Linear<B>,
    summary_projector: Linear<B>,
    summary_norm: LayerNorm<B>,
    fusion_dim: usize,
    layer_count: usize,
    freeze_encoder_core: bool,
    passthrough_projection: bool,
    force_projection: bool,
}

impl<B: Backend> TextDragonFusionAdapter<B> {
    pub fn new(config: &BDHConfig, fusion_dim: usize, device: &B::Device) -> Self {
        Self {
            encoder: BDH::new(config.clone(), device),
            token_projector: LinearConfig::new(config.n_embd, fusion_dim).init(device),
            summary_projector: LinearConfig::new(config.n_embd, fusion_dim).init(device),
            summary_norm: LayerNormConfig::new(fusion_dim).init(device),
            fusion_dim,
            layer_count: config.n_layer,
            freeze_encoder_core: false,
            passthrough_projection: config.n_embd == fusion_dim,
            force_projection: false,
        }
    }

    pub fn layer_count(&self) -> usize {
        self.layer_count
    }

    pub fn set_freeze_encoder_core(&mut self, freeze: bool) {
        if freeze && !self.freeze_encoder_core {
            self.encoder = self.encoder.clone().no_grad();
        }
        self.freeze_encoder_core = freeze;
    }

    pub fn set_force_projection(&mut self, force_projection: bool) {
        self.force_projection = force_projection;
    }

    pub fn encode_q(
        &self,
        tokens: Tensor<B, 2, Int>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        state: Option<ModelState<B>>,
    ) -> TextFusionOutput<B, ModelState<B>> {
        let mut carried = state.unwrap_or_else(|| self.encoder.init_state());
        let (hidden, _logits) = self
            .encoder
            .forward_with_hidden_and_state(tokens, &mut carried);
        let hidden = if self.freeze_encoder_core {
            hidden.detach()
        } else {
            hidden
        };
        let carried = if self.freeze_encoder_core {
            detach_model_state(&carried)
        } else {
            carried
        };
        let [batch, time, dim] = hidden.shape().dims::<3>();
        let use_passthrough_projection = self.passthrough_projection && !self.force_projection;
        let fusion_tokens = if use_passthrough_projection {
            hidden.clone()
        } else {
            self.token_projector
                .forward(hidden.clone().reshape([batch * time, dim]))
                .reshape([batch, time, self.fusion_dim])
        };
        let summary = masked_mean(hidden, mask);
        let summary = if use_passthrough_projection {
            summary
        } else {
            self.summary_norm
                .forward(self.summary_projector.forward(summary))
        };
        TextFusionOutput {
            fusion_tokens,
            summary_token: summary,
            state: Some(carried),
        }
    }

    pub fn replace_encoder(&mut self, encoder: BDH<B>) {
        self.encoder = encoder;
    }
}

impl<B: AutodiffBackend> TextDragonFusionAdapter<B> {
    pub fn valid_encoder_core(&self) -> BDH<B::InnerBackend> {
        self.encoder.valid()
    }

    pub fn encode_q_frozen_core(
        &self,
        valid_encoder: &BDH<B::InnerBackend>,
        tokens: Tensor<B, 2, Int>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
        state: Option<ModelState<B>>,
    ) -> TextFusionOutput<B, ModelState<B>> {
        let mut carried = state
            .as_ref()
            .map(model_state_inner)
            .unwrap_or_else(|| valid_encoder.init_state());
        let (hidden, _logits) =
            valid_encoder.forward_with_hidden_and_state(tokens.inner(), &mut carried);
        let hidden = Tensor::from_inner(hidden);
        let carried = model_state_from_inner(carried);
        let [batch, time, dim] = hidden.shape().dims::<3>();
        let use_passthrough_projection = self.passthrough_projection && !self.force_projection;
        let fusion_tokens = if use_passthrough_projection {
            hidden.clone()
        } else {
            self.token_projector
                .forward(hidden.clone().reshape([batch * time, dim]))
                .reshape([batch, time, self.fusion_dim])
        };
        let summary = masked_mean(hidden, mask);
        let summary = if use_passthrough_projection {
            summary
        } else {
            self.summary_norm
                .forward(self.summary_projector.forward(summary))
        };
        TextFusionOutput {
            fusion_tokens,
            summary_token: summary,
            state: Some(carried),
        }
    }
}

impl<B: Backend> TextFusionAdapter<B> for TextDragonFusionAdapter<B> {
    type State = ModelState<B>;
    type Input = (Tensor<B, 2, Int>, Option<Tensor<B, 2, burn::tensor::Bool>>);

    fn observe_q(
        &self,
        input: Self::Input,
        state: Option<Self::State>,
    ) -> TextFusionOutput<B, Self::State> {
        self.encode_q(input.0, input.1, state)
    }
}

#[derive(Module, Debug)]
pub struct TargetTextDragonEncoderAdapter<B: Backend> {
    encoder: BDH<B>,
    projector: Linear<B>,
    norm: LayerNorm<B>,
    #[module(ignore)]
    use_fixed_fourier_mean: bool,
    #[module(ignore)]
    target_dim: usize,
    #[module(ignore)]
    passthrough_projection: bool,
    #[module(ignore)]
    freeze_encoder_core: bool,
    #[module(ignore)]
    force_projection: bool,
}

impl<B: Backend> TargetTextDragonEncoderAdapter<B> {
    pub fn new(config: &BDHConfig, target_dim: usize, device: &B::Device) -> Self {
        Self {
            encoder: BDH::new(config.clone(), device),
            projector: LinearConfig::new(config.n_embd, target_dim).init(device),
            norm: LayerNormConfig::new(target_dim).init(device),
            use_fixed_fourier_mean: false,
            target_dim,
            passthrough_projection: config.n_embd == target_dim,
            freeze_encoder_core: false,
            force_projection: false,
        }
    }

    pub fn with_kind(
        config: &BDHConfig,
        target_dim: usize,
        kind: TargetTextEncoderKind,
        device: &B::Device,
    ) -> Self {
        let mut adapter = Self::new(config, target_dim, device);
        adapter.use_fixed_fourier_mean = matches!(kind, TargetTextEncoderKind::FixedFourierMean);
        adapter
    }

    fn fixed_fourier_mean(
        &self,
        tokens: Tensor<B, 2, Int>,
        mask: Option<Tensor<B, 2, burn::tensor::Bool>>,
    ) -> Tensor<B, 2> {
        let [batch, time] = tokens.shape().dims::<2>();
        let device = tokens.device();
        let token_values = tokens.float().unsqueeze_dim::<3>(2);
        let half = (self.target_dim.max(2)) / 2;
        let mut freq_values = Vec::with_capacity(half);
        for index in 0..half {
            let ratio = index as f32 / half.max(1) as f32;
            freq_values.push((1.0_f32 / 32.0).powf(ratio).max(1e-3));
        }
        let freqs =
            Tensor::<B, 1>::from_floats(freq_values.as_slice(), &device).reshape([1, 1, half]);
        let phases = token_values * freqs;
        let mut features = Tensor::cat(vec![phases.clone().sin(), phases.cos()], 2);
        let feature_dim = features.shape().dims::<3>()[2];
        if feature_dim < self.target_dim {
            let pad = Tensor::<B, 3>::zeros([batch, time, self.target_dim - feature_dim], &device);
            features = Tensor::cat(vec![features, pad], 2);
        } else if feature_dim > self.target_dim {
            features = features.slice([0..batch, 0..time, 0..self.target_dim]);
        }
        masked_mean(features, mask)
    }

    pub fn replace_encoder(&mut self, encoder: BDH<B>) {
        self.encoder = encoder;
    }

    pub fn set_freeze_encoder_core(&mut self, freeze: bool) {
        if freeze && !self.freeze_encoder_core {
            self.encoder = self.encoder.clone().no_grad();
        }
        self.freeze_encoder_core = freeze;
    }

    pub fn set_force_projection(&mut self, force_projection: bool) {
        self.force_projection = force_projection;
    }
}

impl<B: Backend> TargetTextEncoderAdapter<B> for TargetTextDragonEncoderAdapter<B> {
    type Input = (Tensor<B, 2, Int>, Option<Tensor<B, 2, burn::tensor::Bool>>);

    fn encode_y(&self, input: Self::Input) -> TargetTextEmbeddingOutput<B> {
        let target_embedding = if self.use_fixed_fourier_mean {
            self.fixed_fourier_mean(input.0, input.1)
        } else {
            let mut state = self.encoder.init_state();
            let (hidden, _logits) = self
                .encoder
                .forward_with_hidden_and_state(input.0, &mut state);
            let hidden = if self.freeze_encoder_core {
                hidden.detach()
            } else {
                hidden
            };
            let summary = masked_mean(hidden, input.1);
            let use_passthrough_projection = self.passthrough_projection && !self.force_projection;
            if use_passthrough_projection {
                summary
            } else {
                self.norm.forward(self.projector.forward(summary))
            }
        };
        TargetTextEmbeddingOutput { target_embedding }
    }
}

impl<B: AutodiffBackend> TargetTextDragonEncoderAdapter<B> {
    pub fn valid_encoder_core(&self) -> BDH<B::InnerBackend> {
        self.encoder.valid()
    }

    pub fn encode_y_frozen_core(
        &self,
        valid_encoder: &BDH<B::InnerBackend>,
        input: (Tensor<B, 2, Int>, Option<Tensor<B, 2, burn::tensor::Bool>>),
    ) -> TargetTextEmbeddingOutput<B> {
        let target_embedding = if self.use_fixed_fourier_mean {
            self.fixed_fourier_mean(input.0, input.1)
        } else {
            let mut state = valid_encoder.init_state();
            let (hidden, _logits) =
                valid_encoder.forward_with_hidden_and_state(input.0.inner(), &mut state);
            let hidden = Tensor::from_inner(hidden);
            let summary = masked_mean(hidden, input.1);
            let use_passthrough_projection = self.passthrough_projection && !self.force_projection;
            if use_passthrough_projection {
                summary
            } else {
                self.norm.forward(self.projector.forward(summary))
            }
        };
        TargetTextEmbeddingOutput { target_embedding }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VlJepaDragonConfig;
    use crate::state::VisionMultimodalState;
    use burn_dragon_vision::api::model::VisionBackboneKind;
    use burn_ndarray::NdArray;

    #[test]
    fn text_adapter_threads_state_and_projects_to_fusion_dim() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let config = VlJepaDragonConfig::default();
        let adapter =
            TextDragonFusionAdapter::<Backend>::new(&config.query_text, config.fusion_dim, &device);
        let tokens = Tensor::<Backend, 2, Int>::zeros([2, 4], &device);
        let output = adapter.observe_q((tokens, None), None);
        assert_eq!(
            output.fusion_tokens.shape().dims(),
            [2, 4, config.fusion_dim]
        );
        assert_eq!(output.summary_token.shape().dims(), [2, config.fusion_dim]);
        assert!(output.state.is_some());
    }

    #[test]
    fn vision_adapter_projects_embed_space_even_when_projection_dim_differs() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.backbone = VisionBackboneKind::Dense;
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 48;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.vision.pos_max_height = 8;
        config.vision.pos_max_width = 8;
        config.fusion_dim = 24;
        let adapter = VisionDragonFusionAdapter::<Backend>::new(&config, &device);
        let images = Tensor::<Backend, 4>::zeros([2, 3, 32, 32], &device);
        let output = adapter.observe_x(images, None, MultimodalStepMode::Observe);
        assert_eq!(output.fusion_tokens.shape().dims::<3>()[0], 2);
        assert_eq!(
            output.fusion_tokens.shape().dims::<3>()[2],
            config.fusion_dim
        );
        assert_eq!(output.summary_token.shape().dims(), [2, config.fusion_dim]);
    }

    #[test]
    fn dense_vision_adapter_keeps_no_persistent_vision_state() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.backbone = VisionBackboneKind::Dense;
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.vision.image_size = 32;
        config.vision.pos_max_height = 8;
        config.vision.pos_max_width = 8;
        config.fusion_dim = 32;
        let adapter = VisionDragonFusionAdapter::<Backend>::new(&config, &device);
        let images = Tensor::<Backend, 4>::zeros([2, 3, 32, 32], &device);
        let output = adapter.observe_x(images, None, MultimodalStepMode::Observe);
        assert!(output.state.is_none());
    }

    #[test]
    fn pyramid_vision_adapter_threads_state_and_can_refine() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.backbone = VisionBackboneKind::Pyramid;
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.vision.image_size = 32;
        config.vision.pos_max_height = 8;
        config.vision.pos_max_width = 8;
        config.fusion_dim = 32;
        let adapter = VisionDragonFusionAdapter::<Backend>::new(&config, &device);
        let images = Tensor::<Backend, 4>::zeros([2, 3, 32, 32], &device);
        let output = adapter.observe_x(images, None, MultimodalStepMode::Observe);
        let state = output.state.expect("pyramid backbone should return state");
        assert!(matches!(state, VisionMultimodalState::Pyramid(_)));
        let refined = adapter.refine_state(state);
        assert!(refined.is_some());
    }

    #[test]
    fn dense_vision_adapter_video_observe_has_no_persistent_state() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.backbone = VisionBackboneKind::Dense;
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.vision.image_size = 32;
        config.vision.pos_max_height = 8;
        config.vision.pos_max_width = 8;
        config.fusion_dim = 32;
        let adapter = VisionDragonFusionAdapter::<Backend>::new(&config, &device);
        let video = Tensor::<Backend, 5>::zeros([2, 3, 3, 32, 32], &device);
        let output = adapter.observe_video_x(video, None);
        assert!(output.state.is_none());
    }

    #[test]
    fn pyramid_vision_adapter_video_observe_carries_state_across_frames() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.backbone = VisionBackboneKind::Pyramid;
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.vision.image_size = 32;
        config.vision.pos_max_height = 8;
        config.vision.pos_max_width = 8;
        config.fusion_dim = 32;
        let adapter = VisionDragonFusionAdapter::<Backend>::new(&config, &device);
        let video = Tensor::<Backend, 5>::zeros([2, 3, 3, 32, 32], &device);
        let output = adapter.observe_video_x(video, None);
        let state = output
            .state
            .expect("pyramid video observe should carry vision state");
        assert!(matches!(state, VisionMultimodalState::Pyramid(_)));
        let refined = adapter.refine_state(state);
        assert!(refined.is_some());
    }

    #[test]
    fn fixed_fourier_target_encoder_distinguishes_different_text_sequences() {
        type Backend = NdArray<f32>;
        let device = Default::default();
        let config = VlJepaDragonConfig {
            target_text_encoder: TargetTextEncoderKind::FixedFourierMean,
            target_dim: 16,
            ..Default::default()
        };
        let adapter = TargetTextDragonEncoderAdapter::<Backend>::with_kind(
            &config.target_text,
            config.target_dim,
            config.target_text_encoder,
            &device,
        );
        let a = Tensor::<Backend, 2, Int>::from_data([[1_i64, 2, 3, 0]], &device);
        let b = Tensor::<Backend, 2, Int>::from_data([[4_i64, 5, 6, 0]], &device);
        let a_embed = adapter.encode_y((a, None)).target_embedding;
        let b_embed = adapter.encode_y((b, None)).target_embedding;
        let a_values = a_embed.to_data().to_vec::<f32>().expect("a values");
        let b_values = b_embed.to_data().to_vec::<f32>().expect("b values");
        assert_ne!(a_values, b_values);
    }
}
