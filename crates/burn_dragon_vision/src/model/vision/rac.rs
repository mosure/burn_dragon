use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::TensorData;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

use burn_dragon_core::DragonNorm;

use crate::config::VisionRacConfig;

use super::{VisionBackboneKind, VisionDragon, VisionDragonConfig, VisionRolloutState, unpatchify};

#[derive(Clone)]
pub struct VisionRacVelocityOutput<B: Backend> {
    pub rollout_state: VisionRolloutState<B>,
    pub patch_tokens: Tensor<B, 3>,
    pub velocity_patches: Tensor<B, 3>,
    pub velocity_frames: Tensor<B, 4>,
    pub summary: Tensor<B, 2>,
    pub memory_read_norm: Tensor<B, 2>,
    pub memory_write_norm: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct VisionRacVelocityBackbone<B: Backend> {
    pub state_model: VisionDragon<B>,
    input_lift: Conv2d<B>,
    token_norm: DragonNorm<B>,
    hidden: Option<Linear<B>>,
    out: Linear<B>,
    #[module(ignore)]
    state_channels: usize,
    #[module(ignore)]
    observe_steps: usize,
    #[module(ignore)]
    backprop_steps: usize,
    #[module(ignore)]
    disable_writes: bool,
    #[module(ignore)]
    velocity_tanh_scale: f32,
}

impl<B: Backend> VisionRacVelocityBackbone<B> {
    pub fn new(
        state_model: VisionDragon<B>,
        vision: &VisionDragonConfig,
        config: &VisionRacConfig,
        device: &B::Device,
    ) -> Self {
        let input_channels = config.state_channels.max(1) + 3;
        let lifted_channels = vision.in_channels.max(1);
        let input_lift = Conv2dConfig::new([input_channels, lifted_channels], [1, 1]).init(device);
        let projection_dim = vision.projection_dim.max(1);
        let token_norm = DragonNorm::new(
            state_model.normalization_config(),
            projection_dim.max(1),
            device,
        );
        let hidden_dim = config.velocity_hidden_dim.max(0);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(projection_dim, hidden_dim).init(device))
        } else {
            None
        };
        let patch_dim = config
            .state_channels
            .max(1)
            .saturating_mul(vision.patch_size.max(1))
            .saturating_mul(vision.patch_size.max(1));
        let out_in = hidden_dim.max(projection_dim);
        let out = LinearConfig::new(out_in.max(1), patch_dim.max(1)).init(device);

        Self {
            state_model,
            input_lift,
            token_norm,
            hidden,
            out,
            state_channels: config.state_channels.max(1),
            observe_steps: config.memory.observe_steps.max(1),
            backprop_steps: config.memory.backprop_steps.max(1),
            disable_writes: config.memory.disable_writes,
            velocity_tanh_scale: config.velocity_tanh_scale.max(0.0),
        }
    }

    pub fn state_channels(&self) -> usize {
        self.state_channels
    }

    pub fn memory_component_names(&self) -> Vec<String> {
        match self.state_model.backbone_kind() {
            VisionBackboneKind::Dense => vec!["no_memory".to_string()],
            VisionBackboneKind::Cellular => vec!["rho".to_string()],
            VisionBackboneKind::Pyramid => vec![
                "patch_rho".to_string(),
                "coarse_rho".to_string(),
                "hub_rho".to_string(),
            ],
        }
    }

    fn zero_memory_norm(&self, batch: usize, device: &B::Device) -> Tensor<B, 2> {
        let width = self.memory_component_names().len().max(1);
        Tensor::<B, 2>::zeros([batch, width], device)
    }

    fn mean_abs_rho5(&self, rho: Tensor<B, 5>) -> Tensor<B, 2> {
        let [batch, ..] = rho.shape().dims::<5>();
        rho.abs()
            .mean_dim(4)
            .mean_dim(3)
            .mean_dim(2)
            .mean_dim(1)
            .reshape([batch, 1])
    }

    fn mean_abs_rho4(&self, rho: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, ..] = rho.shape().dims::<4>();
        rho.abs()
            .mean_dim(3)
            .mean_dim(2)
            .mean_dim(1)
            .reshape([batch, 1])
    }

    fn summarize_memory_state(&self, state: &VisionRolloutState<B>) -> Tensor<B, 2> {
        match state {
            VisionRolloutState::Dense { token_state } => {
                let [batch, ..] = token_state.shape().dims::<3>();
                self.zero_memory_norm(batch, &token_state.device())
            }
            VisionRolloutState::Cellular(state) => self.mean_abs_rho5(state.rho.clone()),
            VisionRolloutState::Pyramid(state) => Tensor::cat(
                vec![
                    self.mean_abs_rho5(state.patch_rho().clone()),
                    self.mean_abs_rho5(state.coarse_rho().clone()),
                    self.mean_abs_rho4(state.hub_rho().clone()),
                ],
                1,
            ),
        }
    }

    fn summarize_memory_delta(
        &self,
        previous: Option<&VisionRolloutState<B>>,
        updated: &VisionRolloutState<B>,
    ) -> Tensor<B, 2> {
        match (previous, updated) {
            (
                Some(VisionRolloutState::Cellular(previous)),
                VisionRolloutState::Cellular(updated),
            ) => self.mean_abs_rho5(updated.rho.clone() - previous.rho.clone()),
            (Some(VisionRolloutState::Pyramid(previous)), VisionRolloutState::Pyramid(updated)) => {
                Tensor::cat(
                    vec![
                        self.mean_abs_rho5(
                            updated.patch_rho().clone() - previous.patch_rho().clone(),
                        ),
                        self.mean_abs_rho5(
                            updated.coarse_rho().clone() - previous.coarse_rho().clone(),
                        ),
                        self.mean_abs_rho4(updated.hub_rho().clone() - previous.hub_rho().clone()),
                    ],
                    1,
                )
            }
            (Some(VisionRolloutState::Dense { token_state }), VisionRolloutState::Dense { .. }) => {
                let [batch, ..] = token_state.shape().dims::<3>();
                self.zero_memory_norm(batch, &token_state.device())
            }
            (None, VisionRolloutState::Dense { token_state }) => {
                let [batch, ..] = token_state.shape().dims::<3>();
                self.zero_memory_norm(batch, &token_state.device())
            }
            (None, VisionRolloutState::Cellular(updated)) => {
                self.mean_abs_rho5(updated.rho.clone())
            }
            (None, VisionRolloutState::Pyramid(updated)) => Tensor::cat(
                vec![
                    self.mean_abs_rho5(updated.patch_rho().clone()),
                    self.mean_abs_rho5(updated.coarse_rho().clone()),
                    self.mean_abs_rho4(updated.hub_rho().clone()),
                ],
                1,
            ),
            _ => {
                let device = match updated {
                    VisionRolloutState::Dense { token_state } => token_state.device(),
                    VisionRolloutState::Cellular(state) => state.rho.device(),
                    VisionRolloutState::Pyramid(state) => state.patch_rho().device(),
                };
                let batch = match updated {
                    VisionRolloutState::Dense { token_state } => token_state.shape().dims::<3>()[0],
                    VisionRolloutState::Cellular(state) => state.rho.shape().dims::<5>()[0],
                    VisionRolloutState::Pyramid(state) => state.patch_rho().shape().dims::<5>()[0],
                };
                self.zero_memory_norm(batch, &device)
            }
        }
    }

    fn augment_state(
        &self,
        current_state: Tensor<B, 4>,
        time_value: f32,
        delta_t: f32,
        direction: f32,
    ) -> Tensor<B, 4> {
        let [batch, _channels, height, width] = current_state.shape().dims::<4>();
        let device = current_state.device();
        let pixels = batch.saturating_mul(height).saturating_mul(width);
        let time_map = Tensor::<B, 4>::from_data(
            TensorData::new(vec![time_value; pixels], [batch, 1, height, width]),
            &device,
        );
        let direction_map = Tensor::<B, 4>::from_data(
            TensorData::new(vec![direction; pixels], [batch, 1, height, width]),
            &device,
        );
        let dt_map = Tensor::<B, 4>::from_data(
            TensorData::new(vec![delta_t; pixels], [batch, 1, height, width]),
            &device,
        );
        Tensor::cat(vec![current_state, time_map, dt_map, direction_map], 1)
    }

    fn decode_velocity_patches(
        &self,
        patch_tokens: Tensor<B, 3>,
        height: usize,
        width: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 4>) {
        let patch_tokens = self.token_norm.forward(patch_tokens);
        let patch_tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(patch_tokens))
        } else {
            patch_tokens
        };
        let velocity_patches =
            activation::tanh(self.out.forward(patch_tokens)).mul_scalar(self.velocity_tanh_scale);
        let [batch, tokens, patch_dim] = velocity_patches.shape().dims::<3>();
        let velocity_frames = unpatchify(
            velocity_patches.clone(),
            self.state_model.patch_size().max(1),
            height,
            width,
            self.state_channels,
        )
        .reshape([batch, self.state_channels, height, width]);
        debug_assert!(tokens > 0);
        debug_assert!(patch_dim > 0);
        (velocity_patches, velocity_frames)
    }

    fn restore_persistent_memory(
        &self,
        previous: VisionRolloutState<B>,
        updated: VisionRolloutState<B>,
    ) -> VisionRolloutState<B> {
        match (previous, updated) {
            (VisionRolloutState::Pyramid(previous), VisionRolloutState::Pyramid(mut updated)) => {
                updated.rho = previous.rho;
                VisionRolloutState::Pyramid(updated)
            }
            (VisionRolloutState::Cellular(previous), VisionRolloutState::Cellular(mut updated)) => {
                updated.rho = previous.rho;
                VisionRolloutState::Cellular(updated)
            }
            (_, updated) => updated,
        }
    }

    pub fn flow_step(
        &self,
        current_state: Tensor<B, 4>,
        time_value: f32,
        delta_t: f32,
        direction: f32,
        rollout_state: Option<VisionRolloutState<B>>,
        reset_memory: bool,
    ) -> VisionRacVelocityOutput<B> {
        let [_batch, _channels, height, width] = current_state.shape().dims::<4>();
        let observed = self.augment_state(current_state, time_value, delta_t, direction);
        let lifted = self.input_lift.forward(observed);
        let previous_rollout_state = if reset_memory {
            None
        } else {
            rollout_state.clone()
        };
        let rollout_state = match (rollout_state, reset_memory) {
            (Some(state), false) => self.state_model.observe_rollout_state_unbounded(
                state,
                lifted,
                self.observe_steps,
                self.backprop_steps,
            ),
            _ => self.state_model.rollout_state_from_images(lifted),
        };
        let output = self.state_model.forward_rollout_state(&rollout_state);
        let patch_tokens = output.patch_tokens;
        let summary = output.cls_token;
        let (velocity_patches, velocity_frames) =
            self.decode_velocity_patches(patch_tokens.clone(), height, width);
        let rollout_state = if self.disable_writes {
            match previous_rollout_state.as_ref() {
                Some(previous) => self.restore_persistent_memory(previous.clone(), rollout_state),
                None => rollout_state,
            }
        } else {
            rollout_state
        };
        let memory_read_norm = match previous_rollout_state.as_ref() {
            Some(previous) => self.summarize_memory_state(previous),
            None => {
                let batch = patch_tokens.shape().dims::<3>()[0];
                self.zero_memory_norm(batch, &patch_tokens.device())
            }
        };
        let memory_write_norm =
            self.summarize_memory_delta(previous_rollout_state.as_ref(), &rollout_state);

        VisionRacVelocityOutput {
            rollout_state,
            patch_tokens,
            velocity_patches,
            velocity_frames,
            summary,
            memory_read_norm,
            memory_write_norm,
        }
    }
}
