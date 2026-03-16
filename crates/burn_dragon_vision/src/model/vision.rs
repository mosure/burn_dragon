use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};

use burn_dragon_core::{
    BankedRhoState, DragonNorm, FusedAttentionExecutor, FusedKernelConfig,
    ManifoldHyperConnections,
    StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState, lowrank_residual_step,
    mhc_merge_with_coefficients, mhc_split_with_coefficients, near_critical_residual_output_std,
    structured_dense_update_tokens, target_major_decay_add,
};
use burn_dragon_wgpu::api::spatial::{
    CompiledLocalGridRhoPlan, CompiledStructuredPyramidRhoPlan, LocalGridNeighborhood,
    LocalGridRhoPlanSpec, LocalGridShape2d, StructuredPyramidBankMode,
    StructuredPyramidCoarseOnlyNoPatchStepInput, StructuredPyramidCoarseOnlyStepOutput,
    StructuredPyramidRhoStepInput, StructuredPyramidRhoStepOutput, StructuredPyramidShape,
    StructuredPyramidSplitRhoStepInput, reference_structured_pyramid_rho_step,
    supports_local_grid_rho_backend, supports_structured_pyramid_rho_backend,
    try_fused_local_grid_rho_attention_wgpu_head_decay,
    try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan,
    try_fused_structured_pyramid_coarse_only_no_patch_step_wgpu_with_plan,
    try_fused_structured_pyramid_rho_step_wgpu_with_plan,
    try_fused_structured_pyramid_split_step_wgpu_with_plan,
};
use std::collections::BTreeMap;

const ROW_NORM_EPS: f32 = 1e-6;

mod cellular_state;
#[cfg(feature = "benchmark")]
mod benchmark;
mod config;
mod embedding;
mod pyramid_ops;
mod rho_stream;
mod rollout_state;
mod token_ops;

pub use cellular_state::VisionCellularState;
#[cfg(feature = "benchmark")]
pub use benchmark::{
    VisionDenseAttentionBenchAdapter, VisionDenseBenchAdapter, VisionRolloutScheduleBenchAdapter,
};
pub use config::*;
pub use embedding::{
    PatchEmbed, PatchEmbedOutput, SpatialPositionalEncoding, VisionProjectionHead, patchify,
    pool_patch_tokens, unpatchify,
};
pub use pyramid_ops::{
    StageAwareHostProfileSnapshot, stage_aware_host_profile_reset,
    stage_aware_host_profile_snapshot,
};
pub use rollout_state::VisionRolloutState;

fn centered_mode_offset_data(modes: usize, width: usize, scale: f32) -> Vec<f32> {
    let center = (modes.saturating_sub(1)) as f32 / 2.0;
    let mut values = Vec::with_capacity(modes.saturating_mul(width));
    for mode in 0..modes {
        let offset = (mode as f32 - center) * scale;
        values.extend(std::iter::repeat_n(offset, width));
    }
    values
}

#[derive(Clone)]
pub struct VisionDragonOutput<B: Backend> {
    /// Dense-space patch-token activations after the selected backbone update path.
    pub patch_tokens: Tensor<B, 3>,
    /// Dense-space global token. In the cellular path this is pooled from patch context rather
    /// than backed by its own persistent `rho`.
    pub cls_token: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct VisionDragonMultiOutput<B: Backend> {
    /// Dense-space patch-token activations across multiple views/eyes.
    pub patch_tokens: Tensor<B, 4>,
    /// Dense-space global token across multiple views/eyes.
    pub cls_token: Tensor<B, 3>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RhoStreamRolloutExecutorMode {
    HostLoop,
    WgpuFused,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PyramidRolloutExecutorMode {
    HostLoop,
    WgpuLocalPlans,
    WgpuFused,
}

#[derive(Module, Debug)]
pub struct VisionDragon<B: Backend> {
    steps: usize,
    n_head: usize,
    embed_dim: usize,
    mlp_internal_dim_multiplier: usize,
    use_cls_token: bool,
    #[module(ignore)]
    backbone_kind: VisionBackboneKind,
    attention_mode: VisionAttentionMode,
    use_alibi: bool,
    alibi_slopes: Option<Tensor<B, 1>>,
    latent_activation: VisionLatentActivation,
    kernel: FusedKernelConfig,
    #[module(ignore)]
    trm_graph: VisionTrmGraphConfig,
    #[module(ignore)]
    rho_stream: VisionRhoStreamConfig,
    cellular_step_mode_embeddings: Option<Param<Tensor<B, 2>>>,
    cellular_query_mode_offsets: Option<Param<Tensor<B, 2>>>,
    cellular_value_mode_offsets: Option<Param<Tensor<B, 2>>>,
    pyramid_patch_x_neuron_proj: Option<Linear<B>>,
    pyramid_patch_to_coarse_query_proj: Option<Linear<B>>,
    pyramid_patch_to_global_query_proj: Option<Linear<B>>,
    pyramid_coarse_x_neuron_proj: Option<Linear<B>>,
    pyramid_coarse_to_global_query_proj: Option<Linear<B>>,
    pyramid_write_value_proj: Option<Linear<B>>,
    pyramid_patch_y_gate_proj: Option<Linear<B>>,
    pyramid_patch_delta_proj: Option<Linear<B>>,
    pyramid_coarse_y_gate_proj: Option<Linear<B>>,
    pyramid_coarse_delta_proj: Option<Linear<B>>,
    pyramid_value_norm: Option<DragonNorm<B>>,
    pyramid_hub_gate: Option<Linear<B>>,
    grid_height: usize,
    grid_width: usize,
    patch_embed: PatchEmbed<B>,
    dropout: Dropout,
    token_norm: Option<DragonNorm<B>>,
    mhc_layers: Option<Vec<ManifoldHyperConnections<B>>>,
    eye_token: Option<Param<Tensor<B, 2>>>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    projection: VisionProjectionHead<B>,
    cls_token: Option<Param<Tensor<B, 2>>>,
    cls_pos: Option<Param<Tensor<B, 2>>>,
    cls_sync_alpha: f32,
    cross_eye_steps: usize,
}

impl<B: Backend> VisionDragon<B> {
    pub fn new(config: VisionDragonConfig, device: &B::Device) -> Self {
        let backbone_kind = config.backbone;
        let patch_embed = PatchEmbed::new(&config, device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let token_norm = if config.token_state_norm {
            Some(DragonNorm::new(
                &config.normalization,
                config.embed_dim,
                device,
            ))
        } else {
            None
        };
        let mhc_layers = if config.mhc.enabled && config.mhc.num_streams > 1 {
            let mut layers = Vec::with_capacity(config.steps.max(1));
            for layer_idx in 0..config.steps.max(1) {
                layers.push(ManifoldHyperConnections::new(
                    &config.mhc,
                    layer_idx,
                    device,
                ));
            }
            Some(layers)
        } else {
            None
        };
        let eye_token = if config.num_eyes.max(1) > 1 {
            let eye = Tensor::<B, 2>::random(
                [config.num_eyes.max(1), config.embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            Some(Param::from_tensor(eye))
        } else {
            None
        };

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();
        let encoder_std =
            near_critical_residual_output_std(config.embed_dim, latent_per_head, config.steps);
        let decoder_std =
            near_critical_residual_output_std(latent_total, config.embed_dim, config.steps);

        let encoder = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.embed_dim, latent_per_head],
            TensorDistribution::Normal(0.0, encoder_std),
            device,
        ));
        let encoder_v = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.embed_dim, latent_per_head],
            TensorDistribution::Normal(0.0, encoder_std),
            device,
        ));
        let decoder = Param::from_tensor(Tensor::<B, 2>::random(
            [latent_total, config.embed_dim],
            TensorDistribution::Normal(0.0, decoder_std),
            device,
        ));

        let projection = VisionProjectionHead::new(
            config.embed_dim,
            config.projection_hidden_dim.max(1),
            config.projection_dim.max(1),
            config.dropout,
            &config.normalization,
            device,
        );

        let trm_graph = config.trm_graph.clone();
        let rho_stream = config.rho_stream.clone();
        let latent_total = config.latent_total();
        let cellular_mode_enabled =
            matches!(backbone_kind, VisionBackboneKind::Cellular) && rho_stream.mode_embeddings;
        let cellular_step_mode_embeddings = if cellular_mode_enabled {
            Some(Param::from_tensor(Tensor::<B, 2>::random(
                [StructuredStepMode::COUNT, config.embed_dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            )))
        } else {
            None
        };
        let cellular_query_mode_offsets = if cellular_mode_enabled {
            Some(Param::from_tensor(Tensor::<B, 2>::from_data(
                TensorData::new(
                    centered_mode_offset_data(StructuredStepMode::COUNT, latent_total, 0.05),
                    [StructuredStepMode::COUNT, latent_total],
                ),
                device,
            )))
        } else {
            None
        };
        let cellular_value_mode_offsets = if cellular_mode_enabled {
            Some(Param::from_tensor(Tensor::<B, 2>::from_data(
                TensorData::new(
                    centered_mode_offset_data(StructuredStepMode::COUNT, config.embed_dim, 0.05),
                    [StructuredStepMode::COUNT, config.embed_dim],
                ),
                device,
            )))
        } else {
            None
        };
        let (
            pyramid_patch_x_neuron_proj,
            pyramid_patch_to_coarse_query_proj,
            pyramid_patch_to_global_query_proj,
            pyramid_coarse_x_neuron_proj,
            pyramid_coarse_to_global_query_proj,
            pyramid_write_value_proj,
            pyramid_patch_y_gate_proj,
            pyramid_patch_delta_proj,
            pyramid_coarse_y_gate_proj,
            pyramid_coarse_delta_proj,
            pyramid_value_norm,
            pyramid_hub_gate,
        ) = if matches!(backbone_kind, VisionBackboneKind::Pyramid) {
            let patch_rank = trm_graph.patch_rank_resolved();
            let coarse_rank = trm_graph.coarse_rank_resolved();
            let global_rank = trm_graph.global_rank_resolved();
            let pyramid_patch_x_neuron_proj =
                LinearConfig::new(config.embed_dim, patch_rank).init(device);
            let pyramid_patch_to_coarse_query_proj =
                LinearConfig::new(config.embed_dim, coarse_rank).init(device);
            let pyramid_patch_to_global_query_proj =
                LinearConfig::new(config.embed_dim, global_rank).init(device);
            let pyramid_coarse_x_neuron_proj =
                LinearConfig::new(config.embed_dim, coarse_rank).init(device);
            let pyramid_coarse_to_global_query_proj =
                LinearConfig::new(config.embed_dim, global_rank).init(device);
            let pyramid_write_value_proj =
                LinearConfig::new(config.embed_dim, trm_graph.value_dim.max(1)).init(device);
            let pyramid_patch_y_gate_proj =
                LinearConfig::new(trm_graph.value_dim.max(1), patch_rank).init(device);
            let pyramid_patch_delta_proj =
                LinearConfig::new(patch_rank, config.embed_dim).init(device);
            let pyramid_coarse_y_gate_proj =
                LinearConfig::new(trm_graph.value_dim.max(1), coarse_rank).init(device);
            let pyramid_coarse_delta_proj =
                LinearConfig::new(coarse_rank, config.embed_dim).init(device);
            let pyramid_value_norm =
                DragonNorm::new(&config.normalization, trm_graph.value_dim.max(1), device);
            let pyramid_hub_gate = if trm_graph.hub_count > 1 && trm_graph.hub_gates {
                Some(LinearConfig::new(config.embed_dim, trm_graph.hub_count).init(device))
            } else {
                None
            };
            (
                Some(pyramid_patch_x_neuron_proj),
                Some(pyramid_patch_to_coarse_query_proj),
                Some(pyramid_patch_to_global_query_proj),
                Some(pyramid_coarse_x_neuron_proj),
                Some(pyramid_coarse_to_global_query_proj),
                Some(pyramid_write_value_proj),
                Some(pyramid_patch_y_gate_proj),
                Some(pyramid_patch_delta_proj),
                Some(pyramid_coarse_y_gate_proj),
                Some(pyramid_coarse_delta_proj),
                Some(pyramid_value_norm),
                pyramid_hub_gate,
            )
        } else {
            (
                None, None, None, None, None, None, None, None, None, None, None, None,
            )
        };

        let (cls_token, cls_pos) = if config.use_cls_token {
            let cls_token = Tensor::<B, 2>::random(
                [1, config.embed_dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            let cls_pos = if config.pos_encoding == SpatialPositionalEncodingKind::Learned2d {
                Some(Param::from_tensor(Tensor::<B, 2>::random(
                    [1, config.embed_dim],
                    TensorDistribution::Normal(0.0, 0.02),
                    device,
                )))
            } else {
                None
            };
            (Some(Param::from_tensor(cls_token)), cls_pos)
        } else {
            (None, None)
        };
        let (use_alibi, alibi_slopes) = if config.use_alibi {
            let slopes = burn_dragon_core::kernel::linear_attention::default_alibi_slopes(
                config.n_head.max(1),
            );
            let slopes =
                Tensor::<B, 1>::from_data(TensorData::new(slopes, [config.n_head.max(1)]), device);
            (true, Some(slopes))
        } else {
            (false, None)
        };

        Self {
            steps: config.steps.max(1),
            n_head: config.n_head,
            embed_dim: config.embed_dim,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            use_cls_token: config.use_cls_token,
            backbone_kind,
            attention_mode: config.attention_mode,
            use_alibi,
            alibi_slopes,
            latent_activation: config.latent_activation,
            kernel: config.fused_kernels,
            patch_embed,
            dropout,
            token_norm,
            mhc_layers,
            eye_token,
            encoder,
            encoder_v,
            decoder,
            projection,
            cls_token,
            cls_pos,
            cls_sync_alpha: config.cls_sync_alpha,
            cross_eye_steps: config.cross_eye_steps,
            trm_graph,
            rho_stream,
            cellular_step_mode_embeddings,
            cellular_query_mode_offsets,
            cellular_value_mode_offsets,
            pyramid_patch_x_neuron_proj,
            pyramid_patch_to_coarse_query_proj,
            pyramid_patch_to_global_query_proj,
            pyramid_coarse_x_neuron_proj,
            pyramid_coarse_to_global_query_proj,
            pyramid_write_value_proj,
            pyramid_patch_y_gate_proj,
            pyramid_patch_delta_proj,
            pyramid_coarse_y_gate_proj,
            pyramid_coarse_delta_proj,
            pyramid_value_norm,
            pyramid_hub_gate,
            grid_height: config.pos_max_height.max(1),
            grid_width: config.pos_max_width.max(1),
        }
    }

    pub fn patch_embed(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward(images)
    }

    pub fn patch_embed_raw(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward_raw(images)
    }

    pub fn patch_size(&self) -> usize {
        self.patch_embed.patch_size()
    }

    pub fn add_patch_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        self.patch_embed.add_position(tokens, grid)
    }

    pub fn add_patch_position_multi(&self, tokens: Tensor<B, 4>, grid: PatchGrid) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        let flat = tokens.reshape([batch * streams, time, dim]);
        let flat = self.patch_embed.add_position(flat, grid);
        flat.reshape([batch, streams, time, dim])
    }

    pub fn project_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.projection.forward(tokens)
    }

    pub fn backbone_kind(&self) -> VisionBackboneKind {
        self.backbone_kind
    }

    pub fn pyramid_backbone_enabled(&self) -> bool {
        matches!(self.backbone_kind, VisionBackboneKind::Pyramid)
    }

    pub fn cellular_backbone_enabled(&self) -> bool {
        matches!(self.backbone_kind, VisionBackboneKind::Cellular)
    }

    pub fn trm_graph_enabled(&self) -> bool {
        self.pyramid_backbone_enabled()
    }

    pub fn rho_stream_enabled(&self) -> bool {
        self.cellular_backbone_enabled()
    }

    pub fn backbone_routing_spec(&self) -> StructuredRoutingSpec {
        match self.backbone_kind {
            VisionBackboneKind::Dense => StructuredRoutingSpec::new(),
            VisionBackboneKind::Cellular => self.cellular_routing_spec(),
            VisionBackboneKind::Pyramid => self.pyramid_routing_spec(),
        }
    }

    pub fn cellular_routing_spec(&self) -> StructuredRoutingSpec {
        StructuredRoutingSpec::new()
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Local,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Identity,
            ))
    }

    pub fn pyramid_routing_spec(&self) -> StructuredRoutingSpec {
        StructuredRoutingSpec::new()
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Local,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Context,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Local,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Broadcast,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Global,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Broadcast,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Global,
                StructuredBankRole::Context,
                StructuredRouteOperation::Read,
                StructuredRoutePattern::Broadcast,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Primary,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Identity,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Context,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Identity,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Context,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Pool,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Primary,
                StructuredBankRole::Global,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Pool,
            ))
            .with_route(StructuredRouteSpec::new(
                StructuredBankRole::Context,
                StructuredBankRole::Global,
                StructuredRouteOperation::Write,
                StructuredRoutePattern::Pool,
            ))
    }

    pub fn forward_images(&self, images: Tensor<B, 4>) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens(patch.tokens)
    }

    pub fn embed_images(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward(images)
    }

    pub fn forward_images_steps(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps(patch.tokens, steps)
    }

    pub fn forward_images_steps_rollout(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps_rollout(patch.tokens, steps, backprop_steps)
    }

    /// Same as `forward_images_steps_rollout`, but does not clamp `steps` to `self.steps`.
    /// Useful for validation-time extrapolation beyond the training rollout depth.
    pub fn forward_images_steps_rollout_unbounded(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps_rollout_unbounded(patch.tokens, steps, backprop_steps)
    }

    pub fn forward_images_embed_steps_rollout_unbounded(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_embed_steps_rollout_unbounded(patch.tokens, steps, backprop_steps)
    }

    pub fn forward_patches(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
    ) -> VisionDragonOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens(tokens)
    }

    pub fn forward_patches_steps(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens_steps(tokens, steps)
    }

    pub fn forward_tokens(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens(tokens);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout(tokens, steps, backprop_steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    /// Same as `forward_tokens_steps_rollout`, but does not clamp `steps` to `self.steps`.
    /// Useful for validation-time extrapolation beyond the training rollout depth.
    pub fn forward_tokens_steps_rollout_unbounded(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout_unbounded(tokens, steps, backprop_steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_embed(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens(tokens);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout(tokens, steps, backprop_steps);
        self.split_output(tokens)
    }

    /// Same as `forward_tokens_embed_steps_rollout`, but does not clamp `steps` to `self.steps`.
    /// Useful for validation-time extrapolation beyond the training rollout depth.
    pub fn forward_tokens_embed_steps_rollout_unbounded(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout_unbounded(tokens, steps, backprop_steps);
        self.split_output(tokens)
    }

    /// Evaluate multiple rollout checkpoints more efficiently than repeated full re-encodes.
    ///
    /// For the dense backbone this reuses detached prefix states over rollout depth, which acts
    /// like TBPTT over step-time: each requested `(steps, backprop_steps)` pair reruns only the
    /// tail segment that is still allowed to carry gradients.
    ///
    /// Backbones with richer persistent state contracts use the explicit `VisionRolloutState`
    /// schedule path, caching detached prefix states and rerunning only the requested tail.
    pub fn forward_images_steps_rollout_schedule(
        &self,
        images: Tensor<B, 4>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps_rollout_schedule_impl(patch.tokens, schedule, false)
    }

    /// Same as `forward_images_steps_rollout_schedule`, but does not clamp the requested rollout
    /// depths to `self.steps`.
    pub fn forward_images_steps_rollout_schedule_unbounded(
        &self,
        images: Tensor<B, 4>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps_rollout_schedule_impl(patch.tokens, schedule, true)
    }

    /// Evaluate multiple predictive rollout checkpoints from an explicit state with cached
    /// detached prefixes between requested horizons.
    pub fn predict_rollout_state_schedule(
        &self,
        state: VisionRolloutState<B>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionRolloutState<B>)> {
        self.forward_rollout_state_schedule_impl(
            state,
            schedule,
            StructuredStepMode::Predict,
            false,
        )
    }

    /// Same as `predict_rollout_state_schedule`, but does not clamp requested rollout depths to
    /// `self.steps`.
    pub fn predict_rollout_state_schedule_unbounded(
        &self,
        state: VisionRolloutState<B>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionRolloutState<B>)> {
        self.forward_rollout_state_schedule_impl(state, schedule, StructuredStepMode::Predict, true)
    }

    /// Evaluate multiple refinement rollout checkpoints from an explicit state with cached
    /// detached prefixes between requested horizons.
    pub fn refine_rollout_state_schedule(
        &self,
        state: VisionRolloutState<B>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionRolloutState<B>)> {
        self.forward_rollout_state_schedule_impl(state, schedule, StructuredStepMode::Refine, false)
    }

    /// Same as `refine_rollout_state_schedule`, but does not clamp requested rollout depths to
    /// `self.steps`.
    pub fn refine_rollout_state_schedule_unbounded(
        &self,
        state: VisionRolloutState<B>,
        schedule: &[(usize, usize)],
    ) -> Vec<(usize, VisionRolloutState<B>)> {
        self.forward_rollout_state_schedule_impl(state, schedule, StructuredStepMode::Refine, true)
    }

    /// Initializes a topology-appropriate rollout state directly from observed images.
    pub fn rollout_state_from_images(&self, images: Tensor<B, 4>) -> VisionRolloutState<B> {
        let patch = self.patch_embed.forward(images);
        self.rollout_state_from_tokens(patch.tokens)
    }

    /// Initializes a topology-appropriate rollout state from positioned patch tokens.
    pub fn rollout_state_from_tokens(&self, tokens: Tensor<B, 3>) -> VisionRolloutState<B> {
        match self.backbone_kind {
            VisionBackboneKind::Dense => VisionRolloutState::Dense {
                token_state: self.prepare_token_state(tokens, true),
            },
            VisionBackboneKind::Pyramid => {
                VisionRolloutState::Pyramid(self.pyramid_state_from_patch_tokens(tokens))
            }
            VisionBackboneKind::Cellular => {
                VisionRolloutState::Cellular(self.cellular_state_from_tokens(tokens))
            }
        }
    }

    /// Replaces the current observed image while preserving any persistent recurrent banks.
    pub fn observe_rollout_state_unbounded(
        &self,
        state: VisionRolloutState<B>,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionRolloutState<B> {
        let patch = self.patch_embed.forward(images);
        self.observe_rollout_state_with_tokens_unbounded(state, patch.tokens, steps, backprop_steps)
    }

    /// Replaces the current observed token stream while preserving any persistent recurrent banks.
    pub fn observe_rollout_state_with_tokens_unbounded(
        &self,
        state: VisionRolloutState<B>,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionRolloutState<B> {
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        match state {
            VisionRolloutState::Dense { .. } => VisionRolloutState::Dense {
                token_state: self.rollout_dense_prepared_state(
                    self.prepare_token_state(tokens, true),
                    steps,
                    detach_until,
                ),
            },
            VisionRolloutState::Pyramid(state) => VisionRolloutState::Pyramid(
                self.observe_pyramid_state(state, tokens, steps, backprop_steps),
            ),
            VisionRolloutState::Cellular(state) => VisionRolloutState::Cellular(
                self.observe_cellular_state(state, tokens, steps, backprop_steps),
            ),
        }
    }

    /// Runs a refinement rollout from an explicit recurrent state without advancing temporal time.
    pub fn refine_rollout_state_unbounded(
        &self,
        state: VisionRolloutState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionRolloutState<B> {
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        match state {
            VisionRolloutState::Dense { token_state } => VisionRolloutState::Dense {
                token_state: self.rollout_dense_prepared_state(token_state, steps, detach_until),
            },
            VisionRolloutState::Pyramid(state) => {
                VisionRolloutState::Pyramid(self.refine_pyramid_state(state, steps, backprop_steps))
            }
            VisionRolloutState::Cellular(state) => VisionRolloutState::Cellular(
                self.refine_cellular_state(state, steps, backprop_steps),
            ),
        }
    }

    /// Runs a predictive rollout from an explicit recurrent state, advancing temporal state where
    /// the active backbone supports it.
    pub fn predict_rollout_state_unbounded(
        &self,
        state: VisionRolloutState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionRolloutState<B> {
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        match state {
            VisionRolloutState::Dense { token_state } => VisionRolloutState::Dense {
                token_state: self.rollout_dense_prepared_state(token_state, steps, detach_until),
            },
            VisionRolloutState::Pyramid(state) => VisionRolloutState::Pyramid(
                self.predict_pyramid_state(state, steps, backprop_steps),
            ),
            VisionRolloutState::Cellular(state) => VisionRolloutState::Cellular(
                self.predict_cellular_state(state, steps, backprop_steps),
            ),
        }
    }

    /// Reads out the current explicit rollout state through the normal projection head.
    pub fn forward_rollout_state(&self, state: &VisionRolloutState<B>) -> VisionDragonOutput<B> {
        match state {
            VisionRolloutState::Dense { token_state } => {
                let projected = self.projection.forward(token_state.clone());
                self.split_output(projected)
            }
            VisionRolloutState::Pyramid(state) => {
                let patch_tokens = self.pyramid_patch_tokens(state);
                let [batch, _time, dim] = patch_tokens.shape().dims::<3>();
                let tokens = if self.use_cls_token {
                    let cls = patch_tokens.clone().mean_dim(1).reshape([batch, 1, dim]);
                    Tensor::cat(vec![cls, patch_tokens], 1)
                } else {
                    patch_tokens
                };
                let projected = self.projection.forward(tokens);
                self.split_output(projected)
            }
            VisionRolloutState::Cellular(state) => self.forward_cellular_state(state),
        }
    }

    /// Initializes the cellular recurrent state from an observed token sequence.
    ///
    /// Persistence contract:
    /// - `token_state` carries the dense residual stream across recurrent calls.
    /// - `rho` stores the persistent patch-local associative memory.
    /// - If CLS is enabled, the CLS token is part of `token_state` but does not own a separate
    ///   `rho` slot; recurrent memory belongs only to patch tokens, and CLS is updated from pooled
    ///   patch context during recurrent reads.
    /// - `temporal_position` advances only during `Predict`; `prediction_age` resets on
    ///   `Observe` and advances during `Predict`.
    pub fn cellular_state_from_tokens(&self, tokens: Tensor<B, 3>) -> VisionCellularState<B> {
        assert!(
            self.cellular_backbone_enabled(),
            "cellular state requires vision.backbone = \"cellular\""
        );
        let token_state = self.prepare_token_state(tokens, true);
        let rho = self.empty_cellular_rho(&token_state);
        VisionCellularState {
            token_state,
            rho,
            temporal_position: 0,
            prediction_age: 0,
        }
    }

    /// Replaces the dense token observation while preserving the persistent cellular `rho`.
    pub fn cellular_state_with_tokens(
        &self,
        state: VisionCellularState<B>,
        tokens: Tensor<B, 3>,
    ) -> VisionCellularState<B> {
        assert!(
            self.cellular_backbone_enabled(),
            "cellular state requires vision.backbone = \"cellular\""
        );
        self.validate_cellular_state(&state);
        let token_state = self.prepare_token_state(tokens, true);
        self.assert_cellular_rho_matches_tokens(&state.rho, &token_state);
        VisionCellularState {
            token_state,
            rho: state.rho,
            temporal_position: state.temporal_position,
            prediction_age: state.prediction_age,
        }
    }

    /// Observation step for the cellular backbone.
    ///
    /// This keeps `rho` resident, replaces the dense token observation, and then rolls the
    /// recurrent block in `Observe` mode without advancing temporal time.
    pub fn observe_cellular_state(
        &self,
        state: VisionCellularState<B>,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionCellularState<B> {
        let state = self.cellular_state_with_tokens(state, tokens);
        self.forward_cellular_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Observe,
        )
    }

    /// Refinement step for the cellular backbone.
    ///
    /// This reuses the current dense token state and `rho` without advancing temporal time.
    pub fn refine_cellular_state(
        &self,
        state: VisionCellularState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionCellularState<B> {
        self.forward_cellular_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Refine,
        )
    }

    /// Predictive temporal rollout for the cellular backbone.
    ///
    /// This reuses the current dense token state and `rho`, advances temporal counters, and
    /// applies predictive recurrent decay.
    pub fn predict_cellular_state(
        &self,
        state: VisionCellularState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionCellularState<B> {
        self.forward_cellular_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn forward_cellular_state(&self, state: &VisionCellularState<B>) -> VisionDragonOutput<B> {
        assert!(
            self.cellular_backbone_enabled(),
            "cellular state requires vision.backbone = \"cellular\""
        );
        self.validate_cellular_state(state);
        let projected = self.projection.forward(state.token_state.clone());
        self.split_output(projected)
    }

    pub fn forward_cellular_state_embed(
        &self,
        state: &VisionCellularState<B>,
    ) -> VisionDragonOutput<B> {
        assert!(
            self.cellular_backbone_enabled(),
            "cellular state requires vision.backbone = \"cellular\""
        );
        self.validate_cellular_state(state);
        self.split_output(state.token_state.clone())
    }

    /// Rolls the cellular recurrent block forward from an explicit state without clamping to
    /// `self.steps`. Both `token_state` and `rho` persist across calls; `y_gate` / `y_neuron`
    /// remain per-step activations and are never stored in the state.
    pub fn forward_cellular_state_rollout_unbounded(
        &self,
        state: VisionCellularState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionCellularState<B> {
        self.forward_cellular_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn forward_cellular_state_rollout_mode_unbounded(
        &self,
        state: VisionCellularState<B>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> VisionCellularState<B> {
        assert!(
            self.cellular_backbone_enabled(),
            "cellular state requires vision.backbone = \"cellular\""
        );
        self.validate_cellular_state(&state);
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.rollout_cellular_state_unbounded(state, steps, detach_until, mode)
    }

    /// Initializes the pyramid recurrent state from observed patch tokens.
    ///
    /// Persistence contract:
    /// - `primary_state` and `context_state` are the dense patch/coarse residual streams.
    /// - `rho.primary_rho` and `rho.context_rho` store the local and coarse associative banks.
    /// - `rho.global_rho` / `hub_rho` is the explicit global recurrent memory for the pyramid
    ///   backbone.
    /// - CLS is not part of this persistent state. When higher-level APIs request a CLS token, it
    ///   is derived as a readout-only summary of patch tokens and does not own its own `rho`.
    /// - `temporal_position` advances only during `Predict`; `prediction_age` resets on
    ///   `Observe` and advances during `Predict`.
    pub fn pyramid_state_from_patch_tokens(
        &self,
        patch_tokens: Tensor<B, 3>,
    ) -> StructuredTopologyState<B> {
        assert!(
            self.pyramid_backbone_enabled(),
            "structured pyramid state requires vision.backbone = \"pyramid\""
        );
        let [batch, patch_count, _dim] = patch_tokens.shape().dims::<3>();
        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        assert_eq!(
            patch_count,
            grid_height * grid_width,
            "structured pyramid state requires patch count {} to match grid {}x{}",
            patch_count,
            grid_height,
            grid_width
        );
        let h8 = self.pyramid_patch_tokens_to_spatial(patch_tokens);
        let coarse_state = self.pyramid_pool_patch_state(h8.clone());
        let patch_rank = self.trm_graph.patch_rank_resolved();
        let coarse_rank = self.trm_graph.coarse_rank_resolved();
        let global_rank = self.trm_graph.global_rank_resolved();
        let value_dim = self.trm_graph.value_dim.max(1);
        let [_, _, h32_height, h32_width] = coarse_state.shape().dims::<4>();
        let device = h8.device();
        StructuredTopologyState {
            primary_state: h8,
            context_state: coarse_state,
            rho: BankedRhoState {
                primary_rho: Tensor::<B, 5>::zeros(
                    [batch, patch_rank, value_dim, grid_height, grid_width],
                    &device,
                ),
                context_rho: Tensor::<B, 5>::zeros(
                    [
                        batch,
                        coarse_rank,
                        value_dim,
                        h32_height.max(1),
                        h32_width.max(1),
                    ],
                    &device,
                ),
                global_rho: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        self.trm_graph.hub_count.max(1),
                        global_rank,
                        value_dim,
                    ],
                    &device,
                ),
            },
            temporal_position: 0,
            prediction_age: 0,
        }
    }

    /// Replaces the dense patch/coarse observation while preserving local/coarse/global `rho`.
    pub fn pyramid_state_with_patch_tokens(
        &self,
        mut state: StructuredTopologyState<B>,
        patch_tokens: Tensor<B, 3>,
    ) -> StructuredTopologyState<B> {
        let next_patch = self.pyramid_patch_tokens_to_spatial(patch_tokens);
        let next_coarse = self.pyramid_pool_patch_state(next_patch.clone());
        *state.primary_state_mut() = next_patch;
        *state.context_state_mut() = next_coarse;
        state
    }

    /// Replaces the dense patch observation in spatial form while preserving local/coarse/global
    /// `rho`.
    pub fn pyramid_state_with_patch_state(
        &self,
        mut state: StructuredTopologyState<B>,
        patch_state: Tensor<B, 4>,
    ) -> StructuredTopologyState<B> {
        let next_patch = self.apply_embed_norm_spatial(patch_state);
        let next_coarse = self.pyramid_pool_patch_state(next_patch.clone());
        *state.primary_state_mut() = next_patch;
        *state.context_state_mut() = next_coarse;
        state
    }

    /// Observation step for the pyramid backbone.
    ///
    /// This preserves local/coarse/global recurrent banks, replaces the current observed dense
    /// patch state, and runs the recurrent block in `Observe` mode without advancing time.
    pub fn observe_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        patch_tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        let state = self.pyramid_state_with_patch_tokens(state, patch_tokens);
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Observe,
        )
    }

    /// Refinement step for the pyramid backbone.
    ///
    /// This reuses the current patch/coarse states and local/coarse/global `rho` without
    /// advancing temporal time.
    pub fn refine_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Refine,
        )
    }

    /// Predictive temporal rollout for the pyramid backbone.
    ///
    /// This reuses the current patch/coarse states and local/coarse/global banks, advances
    /// temporal counters, and applies predictive recurrent decay.
    pub fn predict_pyramid_state(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn pyramid_patch_tokens(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 3> {
        self.pyramid_spatial_to_patch_tokens(state.primary_state().clone())
    }

    /// Returns the pyramid readout-only global summary.
    ///
    /// This is derived from the current patch token state and does not own its own recurrent
    /// `rho`; persistent global memory lives in `state.hub_rho()`.
    pub fn pyramid_summary(&self, state: &StructuredTopologyState<B>) -> Tensor<B, 2> {
        let [batch, dim, _height, _width] = state.primary_state().shape().dims::<4>();
        state
            .primary_state()
            .clone()
            .mean_dim(2)
            .mean_dim(3)
            .reshape([batch, dim])
    }

    fn pyramid_decay_by_rank(
        &self,
        rank: usize,
        temporal_dt: usize,
        device: &B::Device,
        decay_scale: f32,
    ) -> Tensor<B, 1> {
        if temporal_dt == 0 {
            return Tensor::<B, 1>::ones([rank.max(1)], device);
        }

        let base_decay = self.trm_graph.decay.clamp(0.0, 1.0);
        if base_decay <= 0.0 {
            return Tensor::<B, 1>::zeros([rank.max(1)], device);
        }
        if base_decay >= 1.0 {
            return Tensor::<B, 1>::ones([rank.max(1)], device);
        }

        let slopes = if self.use_alibi {
            burn_dragon_core::kernel::linear_attention::default_alibi_slopes(rank.max(1))
        } else {
            vec![1.0; rank.max(1)]
        };
        let dt = (temporal_dt as f32) * decay_scale.max(0.0);
        let values = slopes
            .into_iter()
            .map(|slope| base_decay.powf(slope * dt))
            .collect::<Vec<_>>();
        Tensor::<B, 1>::from_data(TensorData::new(values, [rank.max(1)]), device)
    }

    fn pyramid_rollout_executor_mode(&self) -> PyramidRolloutExecutorMode {
        if self.kernel.enabled && supports_structured_pyramid_rho_backend::<B>() {
            return PyramidRolloutExecutorMode::WgpuFused;
        }
        if self.kernel.enabled && supports_local_grid_rho_backend::<B>() {
            return PyramidRolloutExecutorMode::WgpuLocalPlans;
        }
        PyramidRolloutExecutorMode::HostLoop
    }

    fn pyramid_bank_mode_for_kernel(
        bank_mode: &VisionTrmGraphBankModeConfig,
    ) -> StructuredPyramidBankMode {
        StructuredPyramidBankMode {
            patch_local_read: bank_mode.patch_local_read,
            patch_local_write: bank_mode.patch_local_write,
            coarse_local_read: bank_mode.coarse_local_read,
            coarse_local_write: bank_mode.coarse_local_write,
            patch_from_coarse_read: bank_mode.patch_from_coarse_read,
            patch_from_hub_read: bank_mode.patch_from_hub_read,
            coarse_from_hub_read: bank_mode.coarse_from_hub_read,
            patch_to_coarse_write: bank_mode.patch_to_coarse_write,
            patch_to_global_write: bank_mode.patch_to_global_write,
            coarse_to_global_write: bank_mode.coarse_to_global_write,
        }
    }

    fn pyramid_shape(&self) -> StructuredPyramidShape {
        StructuredPyramidShape {
            patch: LocalGridShape2d::new(self.grid_height.max(1), self.grid_width.max(1)),
            coarse: LocalGridShape2d::new(
                (self.grid_height.max(1) / self.trm_graph.coarse_stride.max(1)).max(1),
                (self.grid_width.max(1) / self.trm_graph.coarse_stride.max(1)).max(1),
            ),
            coarse_stride: self.trm_graph.coarse_stride.max(1),
            hub_count: self.trm_graph.hub_count.max(1),
        }
    }

    fn pyramid_rho_step_with_plan(
        &self,
        shape: StructuredPyramidShape,
        input: StructuredPyramidRhoStepInput<B>,
        fused_plan: Option<&CompiledStructuredPyramidRhoPlan<B>>,
    ) -> StructuredPyramidRhoStepOutput<B> {
        if let Some(plan) = fused_plan
            && let Some(fused) =
                try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input.clone(), plan)
        {
            return fused;
        }
        reference_structured_pyramid_rho_step(shape, input)
    }

    #[allow(clippy::too_many_arguments)]
    fn pyramid_reference_step_split(
        &self,
        patch_query: Tensor<B, 4>,
        patch_query_for_coarse: Tensor<B, 4>,
        patch_query_for_global: Tensor<B, 4>,
        patch_value: Tensor<B, 4>,
        coarse_query: Tensor<B, 4>,
        coarse_query_for_global: Tensor<B, 4>,
        coarse_value: Tensor<B, 4>,
        patch_rho: Tensor<B, 5>,
        coarse_rho: Tensor<B, 5>,
        global_rho: Tensor<B, 4>,
        patch_hub_weights: Option<Tensor<B, 4>>,
        coarse_hub_weights: Option<Tensor<B, 4>>,
        patch_decay: Tensor<B, 1>,
        coarse_decay: Tensor<B, 1>,
        global_decay: Tensor<B, 1>,
        bank_mode: &VisionTrmGraphBankModeConfig,
    ) -> StructuredPyramidRhoStepOutput<B> {
        let shape = self.pyramid_shape();
        let [patch_batch, patch_value_dim, patch_height, patch_width] =
            patch_value.shape().dims::<4>();
        let [coarse_batch, coarse_value_dim, coarse_height, coarse_width] =
            coarse_value.shape().dims::<4>();
        let patch_zero = || {
            Tensor::<B, 4>::zeros(
                [patch_batch, patch_value_dim, patch_height, patch_width],
                &patch_value.device(),
            )
        };
        let coarse_zero = || {
            Tensor::<B, 4>::zeros(
                [coarse_batch, coarse_value_dim, coarse_height, coarse_width],
                &coarse_value.device(),
            )
        };

        let patch_local_context = if bank_mode.patch_local_read {
            self.pyramid_local_read(patch_rho.clone(), patch_query.clone(), false)
        } else {
            patch_zero()
        };
        let coarse_local_context = if bank_mode.coarse_local_read {
            self.pyramid_local_read(coarse_rho.clone(), coarse_query.clone(), true)
        } else {
            coarse_zero()
        };
        let patch_from_coarse_context = if bank_mode.patch_from_coarse_read {
            self.pyramid_cross_scale_read(
                coarse_rho.clone(),
                patch_query_for_coarse.clone(),
                shape.coarse_stride.max(1),
            )
        } else {
            patch_zero()
        };
        let patch_from_hub_context = if bank_mode.patch_from_hub_read {
            self.pyramid_hub_read(
                global_rho.clone(),
                patch_query_for_global.clone(),
                patch_hub_weights.clone(),
            )
        } else {
            patch_zero()
        };
        let coarse_from_hub_context = if bank_mode.coarse_from_hub_read {
            self.pyramid_hub_read(
                global_rho.clone(),
                coarse_query_for_global.clone(),
                coarse_hub_weights.clone(),
            )
        } else {
            coarse_zero()
        };

        let patch_update = if bank_mode.patch_local_write {
            self.pyramid_outer_product(patch_query.clone(), patch_value.clone())
        } else {
            Tensor::<B, 5>::zeros(patch_rho.shape().dims::<5>(), &patch_rho.device())
        };
        let patch_to_coarse_update = bank_mode.patch_to_coarse_write.then(|| {
            self.pyramid_pool_outer(
                self.pyramid_outer_product(patch_query_for_coarse, patch_value.clone()),
                shape.coarse_stride.max(1),
            )
        });
        let patch_to_global_update = bank_mode
            .patch_to_global_write
            .then(|| self.pyramid_outer_product(patch_query_for_global, patch_value));
        let coarse_update = if bank_mode.coarse_local_write {
            self.pyramid_outer_product(coarse_query.clone(), coarse_value.clone())
        } else {
            Tensor::<B, 5>::zeros(coarse_rho.shape().dims::<5>(), &coarse_rho.device())
        };
        let coarse_to_global_update = bank_mode
            .coarse_to_global_write
            .then(|| self.pyramid_outer_product(coarse_query_for_global, coarse_value));

        let next_patch_rho = target_major_decay_add(
            Self::pyramid_rho_to_target_major(patch_rho),
            Self::pyramid_rho_to_target_major(patch_update),
            patch_decay,
        );
        let next_patch_rho = Self::pyramid_rho_from_target_major(
            next_patch_rho,
            shape.patch.height,
            shape.patch.width,
        );

        let coarse_rho_shape = coarse_rho.shape().dims::<5>();
        let coarse_rank = coarse_rho_shape[1];
        let coarse_rho_device = coarse_rho.device();
        let next_coarse_rho = target_major_decay_add(
            Self::pyramid_rho_to_target_major(coarse_rho),
            Self::pyramid_rho_to_target_major(coarse_update.clone()).add(
                patch_to_coarse_update
                    .map(Self::pyramid_rho_to_target_major)
                    .unwrap_or_else(|| {
                        Tensor::<B, 4>::zeros(
                            [
                                coarse_batch,
                                coarse_height * coarse_width,
                                coarse_rank,
                                coarse_value_dim,
                            ],
                            &coarse_rho_device,
                        )
                    }),
            ),
            coarse_decay,
        );
        let next_coarse_rho = Self::pyramid_rho_from_target_major(
            next_coarse_rho,
            shape.coarse.height,
            shape.coarse.width,
        );

        let next_hub_rho = self.pyramid_update_hub(
            global_rho,
            patch_to_global_update,
            coarse_to_global_update,
            patch_hub_weights,
            coarse_hub_weights,
            shape.hub_count.max(1),
            global_decay,
        );

        StructuredPyramidRhoStepOutput {
            patch_local_context,
            coarse_local_context,
            patch_from_coarse_context,
            patch_from_hub_context,
            coarse_from_hub_context,
            next_patch_rho,
            next_coarse_rho,
            next_hub_rho,
        }
    }

    pub fn forward_pyramid_state_rollout_unbounded(
        &self,
        state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        )
    }

    pub fn forward_pyramid_state_rollout_mode_unbounded(
        &self,
        mut state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> StructuredTopologyState<B> {
        assert!(
            self.pyramid_backbone_enabled(),
            "structured pyramid rollout requires vision.backbone = \"pyramid\""
        );
        let pyramid_patch_x_neuron_proj = self
            .pyramid_patch_x_neuron_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_x_neuron_proj");
        let pyramid_patch_to_coarse_query_proj = self
            .pyramid_patch_to_coarse_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_to_coarse_query_proj");
        let pyramid_patch_to_global_query_proj = self
            .pyramid_patch_to_global_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_to_global_query_proj");
        let pyramid_coarse_x_neuron_proj = self
            .pyramid_coarse_x_neuron_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_x_neuron_proj");
        let pyramid_coarse_to_global_query_proj = self
            .pyramid_coarse_to_global_query_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_to_global_query_proj");
        let pyramid_write_value_proj = self
            .pyramid_write_value_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_write_value_proj");
        let pyramid_patch_y_gate_proj = self
            .pyramid_patch_y_gate_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_y_gate_proj");
        let pyramid_patch_delta_proj = self
            .pyramid_patch_delta_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_patch_delta_proj");
        let pyramid_coarse_y_gate_proj = self
            .pyramid_coarse_y_gate_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_y_gate_proj");
        let pyramid_coarse_delta_proj = self
            .pyramid_coarse_delta_proj
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_coarse_delta_proj");
        let pyramid_value_norm = self
            .pyramid_value_norm
            .as_ref()
            .expect("structured pyramid rollout requires pyramid_value_norm");
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        let hub_count = self.trm_graph.hub_count.max(1);
        let patch_rank = self.trm_graph.patch_rank_resolved();
        let coarse_rank = self.trm_graph.coarse_rank_resolved();
        let global_rank = self.trm_graph.global_rank_resolved();
        let temporal_dt = mode.temporal_dt();
        let bank_mode = self.trm_graph.bank_mode(mode).clone();
        let predict_coarse_substeps = if matches!(mode, StructuredStepMode::Predict) {
            self.trm_graph.predict_coarse_substeps.max(1)
        } else {
            1
        };
        let patch_decay = self.pyramid_decay_by_rank(
            patch_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.patch_decay_scale,
        );
        let coarse_decay = self.pyramid_decay_by_rank(
            coarse_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.coarse_decay_scale,
        );
        let global_decay = self.pyramid_decay_by_rank(
            global_rank,
            temporal_dt,
            &state.primary_state().device(),
            bank_mode.global_decay_scale,
        );
        let pyramid_shape = self.pyramid_shape();
        let executor_mode = self.pyramid_rollout_executor_mode();
        let uniform_fused_eligible = self.trm_graph.ranks_uniform()
            && self.trm_graph.uses_uniform_local_topology()
            && self.trm_graph.uses_default_bank_schedule()
            && predict_coarse_substeps == 1;
        let fused_plan = match executor_mode {
            PyramidRolloutExecutorMode::WgpuFused if uniform_fused_eligible => {
                Some(CompiledStructuredPyramidRhoPlan::new(
                    state.primary_state().shape().dims::<4>()[0],
                    patch_rank,
                    self.trm_graph.value_dim.max(1),
                    pyramid_shape,
                    self.resolve_rho_stream_neighborhood(),
                    &state.primary_state().device(),
                ))
            }
            _ => None,
        };
        let split_fused_plan = match executor_mode {
            // For custom schedules / heterogeneous ranks, the lighter local-plan executor still
            // benchmarks better overall than the current split-fused path.
            PyramidRolloutExecutorMode::WgpuFused if !uniform_fused_eligible => None,
            _ => None,
        };
        let stage_aware_plan = match executor_mode {
            PyramidRolloutExecutorMode::HostLoop => None,
            PyramidRolloutExecutorMode::WgpuLocalPlans | PyramidRolloutExecutorMode::WgpuFused => {
                Some(pyramid_ops::CompiledStageAwarePyramidLocalPlan::new(
                    pyramid_ops::CompiledStageAwarePyramidLocalPlanSpec {
                        batch: state.primary_state().shape().dims::<4>()[0],
                        patch_rank,
                        coarse_rank,
                        value_dim: self.trm_graph.value_dim.max(1),
                        patch_shape: pyramid_shape.patch,
                        coarse_shape: pyramid_shape.coarse,
                        patch_neighborhood: self.pyramid_patch_neighborhood(),
                        coarse_neighborhood: self.pyramid_coarse_neighborhood(),
                        device: &state.primary_state().device(),
                    },
                ))
            }
        };

        for step_idx in 0..steps {
            let h8 = state.primary_state().clone();
            let h32 = state.context_state().clone();
            let patch_rho = state.patch_rho().clone();
            let coarse_rho = state.coarse_rho().clone();
            let hub_rho = state.hub_rho().clone();
            let [patch_batch, _, patch_height, patch_width] = h8.shape().dims::<4>();
            let patch_need_coarse_query =
                bank_mode.patch_from_coarse_read || bank_mode.patch_to_coarse_write;
            let patch_need_global_query =
                bank_mode.patch_from_hub_read || bank_mode.patch_to_global_write;
            let patch_need_value = bank_mode.patch_local_write
                || bank_mode.patch_to_coarse_write
                || bank_mode.patch_to_global_write;
            let mut patch_layers = vec![pyramid_patch_x_neuron_proj];
            let patch_coarse_query_enabled = patch_need_coarse_query;
            if patch_coarse_query_enabled {
                patch_layers.push(pyramid_patch_to_coarse_query_proj);
            }
            let patch_global_query_enabled = patch_need_global_query;
            if patch_global_query_enabled {
                patch_layers.push(pyramid_patch_to_global_query_proj);
            }
            let patch_value_enabled = patch_need_value;
            if patch_value_enabled {
                patch_layers.push(pyramid_write_value_proj);
            }
            let patch_hub_gate_enabled = hub_count > 1 && self.pyramid_hub_gate.is_some();
            if let Some(hub_gate) = self.pyramid_hub_gate.as_ref() {
                if hub_count > 1 {
                    patch_layers.push(hub_gate);
                }
            }
            let mut patch_proj = self
                .project_spatial_many(h8.clone(), &patch_layers)
                .into_iter();
            let patch_x = activation::relu(
                patch_proj
                    .next()
                    .expect("patch multi-projection should include x_neuron output"),
            );
            let patch_coarse_query = if patch_coarse_query_enabled {
                activation::relu(
                    patch_proj
                        .next()
                        .expect("patch multi-projection should include coarse-query output"),
                )
            } else {
                patch_x.clone()
            };
            let patch_global_query = if patch_global_query_enabled {
                activation::relu(
                    patch_proj
                        .next()
                        .expect("patch multi-projection should include global-query output"),
                )
            } else {
                patch_x.clone()
            };
            let v8 = if patch_value_enabled {
                patch_proj
                    .next()
                    .expect("patch multi-projection should include write-value output")
            } else {
                Tensor::<B, 4>::zeros(
                    [
                        patch_batch,
                        self.trm_graph.value_dim.max(1),
                        patch_height,
                        patch_width,
                    ],
                    &h8.device(),
                )
            };
            let hub_w8 = if patch_hub_gate_enabled {
                Some(
                    self.normalize_hub_weights(
                        patch_proj
                            .next()
                            .expect("patch multi-projection should include hub-gate output"),
                    ),
                )
            } else if hub_count > 1 {
                Some(self.pyramid_hub_weights_single(h8.clone(), hub_count, None))
            } else {
                None
            };
            let mut current_coarse_state = h32;
            let current_patch_rho = patch_rho;
            let mut current_coarse_rho = coarse_rho;
            let mut current_hub_rho = hub_rho;
            let ones_patch_decay =
                Tensor::<B, 1>::ones([patch_rank.max(1)], &state.primary_state().device());
            let ones_coarse_decay =
                Tensor::<B, 1>::ones([coarse_rank.max(1)], &state.primary_state().device());
            let ones_global_decay =
                Tensor::<B, 1>::ones([global_rank.max(1)], &state.primary_state().device());
            let coarse_only_bank_mode = bank_mode.coarse_only_predict_substep();
            let coarse_hub_gate_enabled = hub_count > 1 && self.pyramid_hub_gate.is_some();
            let mut coarse_layers = vec![
                pyramid_coarse_x_neuron_proj,
                pyramid_coarse_to_global_query_proj,
                pyramid_write_value_proj,
            ];
            if let Some(hub_gate) = self.pyramid_hub_gate.as_ref()
                && hub_count > 1
            {
                coarse_layers.push(hub_gate);
            }
            for _ in 1..predict_coarse_substeps {
                let mut coarse_proj = self
                    .project_spatial_many(current_coarse_state.clone(), &coarse_layers)
                    .into_iter();
                let coarse_x = activation::relu(
                    coarse_proj
                        .next()
                        .expect("coarse multi-projection should include x_neuron output"),
                );
                let coarse_global_query = activation::relu(
                    coarse_proj
                        .next()
                        .expect("coarse multi-projection should include global-query output"),
                );
                let v32 = coarse_proj
                    .next()
                    .expect("coarse multi-projection should include write-value output");
                let hub_w32 = if coarse_hub_gate_enabled {
                    Some(
                        self.normalize_hub_weights(
                            coarse_proj
                                .next()
                                .expect("coarse multi-projection should include hub-gate output"),
                        ),
                    )
                } else if hub_count > 1 {
                    Some(self.pyramid_hub_weights_single(
                        current_coarse_state.clone(),
                        hub_count,
                        None,
                    ))
                } else {
                    None
                };
                if let Some(plan) = split_fused_plan.as_ref() {
                    let rho_step =
                        try_fused_structured_pyramid_coarse_only_no_patch_step_wgpu_with_plan(
                            pyramid_shape,
                            StructuredPyramidCoarseOnlyNoPatchStepInput {
                                coarse_local_query: coarse_x.clone(),
                                coarse_query_for_global: coarse_global_query.clone(),
                                coarse_value: v32.clone(),
                                coarse_rho: current_coarse_rho.clone(),
                                hub_rho: current_hub_rho.clone(),
                                coarse_hub_weights: hub_w32.clone(),
                                coarse_decay: ones_coarse_decay.clone(),
                                global_decay: ones_global_decay.clone(),
                                bank_mode: Self::pyramid_bank_mode_for_kernel(
                                    &coarse_only_bank_mode,
                                ),
                            },
                            plan,
                        )
                        .unwrap_or_else(|| {
                            let rho_step = self.pyramid_reference_step_split(
                                patch_x.clone(),
                                patch_coarse_query.clone(),
                                patch_global_query.clone(),
                                v8.clone(),
                                coarse_x.clone(),
                                coarse_global_query,
                                v32,
                                current_patch_rho.clone(),
                                current_coarse_rho,
                                current_hub_rho,
                                hub_w8.clone(),
                                hub_w32.clone(),
                                ones_patch_decay.clone(),
                                ones_coarse_decay.clone(),
                                ones_global_decay.clone(),
                                &coarse_only_bank_mode,
                            );
                            StructuredPyramidCoarseOnlyStepOutput {
                                coarse_local_context: rho_step.coarse_local_context,
                                coarse_from_hub_context: rho_step.coarse_from_hub_context,
                                next_coarse_rho: rho_step.next_coarse_rho,
                                next_hub_rho: rho_step.next_hub_rho,
                            }
                        });
                    current_coarse_state = self.pyramid_update_state(
                        current_coarse_state,
                        coarse_x.clone(),
                        rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                        pyramid_coarse_y_gate_proj,
                        pyramid_coarse_delta_proj,
                        pyramid_value_norm,
                    );
                    current_coarse_rho = rho_step.next_coarse_rho;
                    current_hub_rho = rho_step.next_hub_rho;
                } else if let Some(plan) = stage_aware_plan.as_ref() {
                    let rho_step = self.pyramid_stage_aware_coarse_only_step_with_plan(
                        coarse_x.clone(),
                        coarse_global_query,
                        v32,
                        current_coarse_rho,
                        current_hub_rho,
                        hub_w32.clone(),
                        ones_coarse_decay.clone(),
                        ones_global_decay.clone(),
                        &coarse_only_bank_mode,
                        plan,
                    );
                    current_coarse_state = self.pyramid_update_state(
                        current_coarse_state,
                        coarse_x.clone(),
                        rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                        pyramid_coarse_y_gate_proj,
                        pyramid_coarse_delta_proj,
                        pyramid_value_norm,
                    );
                    current_coarse_rho = rho_step.next_coarse_rho;
                    current_hub_rho = rho_step.next_hub_rho;
                } else {
                    let rho_step = self.pyramid_reference_step_split(
                        patch_x.clone(),
                        patch_coarse_query.clone(),
                        patch_global_query.clone(),
                        v8.clone(),
                        coarse_x.clone(),
                        coarse_global_query,
                        v32,
                        current_patch_rho.clone(),
                        current_coarse_rho,
                        current_hub_rho,
                        hub_w8.clone(),
                        hub_w32.clone(),
                        ones_patch_decay.clone(),
                        ones_coarse_decay.clone(),
                        ones_global_decay.clone(),
                        &coarse_only_bank_mode,
                    );
                    current_coarse_state = self.pyramid_update_state(
                        current_coarse_state,
                        coarse_x,
                        rho_step.coarse_local_context + rho_step.coarse_from_hub_context,
                        pyramid_coarse_y_gate_proj,
                        pyramid_coarse_delta_proj,
                        pyramid_value_norm,
                    );
                    current_coarse_rho = rho_step.next_coarse_rho;
                    current_hub_rho = rho_step.next_hub_rho;
                }
            }
            let mut coarse_layers = vec![
                pyramid_coarse_x_neuron_proj,
                pyramid_coarse_to_global_query_proj,
                pyramid_write_value_proj,
            ];
            let coarse_hub_gate_enabled = hub_count > 1 && self.pyramid_hub_gate.is_some();
            if let Some(hub_gate) = self.pyramid_hub_gate.as_ref() {
                if hub_count > 1 {
                    coarse_layers.push(hub_gate);
                }
            }
            let mut coarse_proj = self
                .project_spatial_many(current_coarse_state.clone(), &coarse_layers)
                .into_iter();
            let coarse_x = activation::relu(
                coarse_proj
                    .next()
                    .expect("coarse multi-projection should include x_neuron output"),
            );
            let coarse_global_query = activation::relu(
                coarse_proj
                    .next()
                    .expect("coarse multi-projection should include global-query output"),
            );
            let v32 = coarse_proj
                .next()
                .expect("coarse multi-projection should include write-value output");
            let hub_w32 = if coarse_hub_gate_enabled {
                Some(
                    self.normalize_hub_weights(
                        coarse_proj
                            .next()
                            .expect("coarse multi-projection should include hub-gate output"),
                    ),
                )
            } else if hub_count > 1 {
                Some(self.pyramid_hub_weights_single(current_coarse_state.clone(), hub_count, None))
            } else {
                None
            };
            let rho_step = if let Some(plan) = split_fused_plan.as_ref() {
                try_fused_structured_pyramid_split_step_wgpu_with_plan(
                    pyramid_shape,
                    StructuredPyramidSplitRhoStepInput {
                        patch_local_query: patch_x.clone(),
                        patch_query_for_coarse: patch_coarse_query.clone(),
                        patch_query_for_global: patch_global_query.clone(),
                        patch_value: v8.clone(),
                        coarse_local_query: coarse_x.clone(),
                        coarse_query_for_global: coarse_global_query.clone(),
                        coarse_value: v32.clone(),
                        patch_rho: current_patch_rho.clone(),
                        coarse_rho: current_coarse_rho.clone(),
                        hub_rho: current_hub_rho.clone(),
                        patch_hub_weights: hub_w8.clone(),
                        coarse_hub_weights: hub_w32.clone(),
                        patch_decay: patch_decay.clone(),
                        coarse_decay: coarse_decay.clone(),
                        global_decay: global_decay.clone(),
                        bank_mode: Self::pyramid_bank_mode_for_kernel(&bank_mode),
                    },
                    plan,
                )
                .unwrap_or_else(|| {
                    self.pyramid_reference_step_split(
                        patch_x.clone(),
                        patch_coarse_query.clone(),
                        patch_global_query.clone(),
                        v8.clone(),
                        coarse_x.clone(),
                        coarse_global_query.clone(),
                        v32.clone(),
                        current_patch_rho,
                        current_coarse_rho,
                        current_hub_rho,
                        hub_w8.clone(),
                        hub_w32.clone(),
                        patch_decay.clone(),
                        coarse_decay.clone(),
                        global_decay.clone(),
                        &bank_mode,
                    )
                })
            } else if let Some(plan) = stage_aware_plan.as_ref() {
                self.pyramid_stage_aware_step_split_with_plan(
                    patch_x.clone(),
                    patch_coarse_query.clone(),
                    patch_global_query.clone(),
                    v8.clone(),
                    coarse_x.clone(),
                    coarse_global_query.clone(),
                    v32.clone(),
                    current_patch_rho,
                    current_coarse_rho,
                    current_hub_rho,
                    hub_w8.clone(),
                    hub_w32.clone(),
                    patch_decay.clone(),
                    coarse_decay.clone(),
                    global_decay.clone(),
                    &bank_mode,
                    plan,
                )
            } else if fused_plan.is_some() {
                self.pyramid_rho_step_with_plan(
                    pyramid_shape,
                    StructuredPyramidRhoStepInput {
                        patch_query: patch_x.clone(),
                        patch_value: v8.clone(),
                        coarse_query: coarse_x.clone(),
                        coarse_value: v32.clone(),
                        patch_rho: current_patch_rho,
                        coarse_rho: current_coarse_rho,
                        hub_rho: current_hub_rho,
                        patch_hub_weights: hub_w8.clone(),
                        coarse_hub_weights: hub_w32.clone(),
                        neighborhood: self.resolve_rho_stream_neighborhood(),
                        decay: patch_decay.clone(),
                    },
                    fused_plan.as_ref(),
                )
            } else {
                self.pyramid_reference_step_split(
                    patch_x.clone(),
                    patch_coarse_query.clone(),
                    patch_global_query.clone(),
                    v8.clone(),
                    coarse_x.clone(),
                    coarse_global_query.clone(),
                    v32.clone(),
                    current_patch_rho,
                    current_coarse_rho,
                    current_hub_rho,
                    hub_w8.clone(),
                    hub_w32.clone(),
                    patch_decay.clone(),
                    coarse_decay.clone(),
                    global_decay.clone(),
                    &bank_mode,
                )
            };

            let next_patch_state = self.pyramid_update_state(
                h8,
                patch_x.clone(),
                rho_step.patch_local_context.clone()
                    + rho_step.patch_from_coarse_context.clone()
                    + rho_step.patch_from_hub_context.clone(),
                pyramid_patch_y_gate_proj,
                pyramid_patch_delta_proj,
                pyramid_value_norm,
            );
            let next_coarse_state = self.pyramid_update_state(
                current_coarse_state,
                coarse_x.clone(),
                rho_step.coarse_local_context.clone() + rho_step.coarse_from_hub_context.clone(),
                pyramid_coarse_y_gate_proj,
                pyramid_coarse_delta_proj,
                pyramid_value_norm,
            );

            *state.primary_state_mut() = if step_idx < detach_until {
                next_patch_state.detach()
            } else {
                next_patch_state
            };
            *state.context_state_mut() = if step_idx < detach_until {
                next_coarse_state.detach()
            } else {
                next_coarse_state
            };
            *state.patch_rho_mut() = if step_idx < detach_until {
                rho_step.next_patch_rho.detach()
            } else {
                rho_step.next_patch_rho
            };
            *state.coarse_rho_mut() = if step_idx < detach_until {
                rho_step.next_coarse_rho.detach()
            } else {
                rho_step.next_coarse_rho
            };
            *state.hub_rho_mut() = if step_idx < detach_until {
                rho_step.next_hub_rho.detach()
            } else {
                rho_step.next_hub_rho
            };
        }

        if temporal_dt > 0 {
            state.temporal_position = state.temporal_position.saturating_add(temporal_dt);
            state.prediction_age = state.prediction_age.saturating_add(temporal_dt);
        } else if mode.resets_prediction_age() {
            state.prediction_age = 0;
        }

        state
    }

    pub fn forward_tokens_embed_steps_rollout_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonMultiOutput<B> {
        let tokens = self.encode_tokens_steps_rollout_multi(tokens, steps, backprop_steps);
        self.split_output_multi(tokens)
    }

    fn prepare_token_state(&self, tokens: Tensor<B, 3>, add_cls: bool) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, _] = tokens.shape().dims::<3>();
        let current = tokens.reshape([batch, 1, time, self.embed_dim]);
        self.apply_token_norm(current)
            .reshape([batch, time, self.embed_dim])
    }

    fn apply_cellular_step_mode(
        &self,
        token_state: Tensor<B, 3>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        let Some(step_mode_embeddings) = &self.cellular_step_mode_embeddings else {
            return token_state;
        };
        let [batch, time, embed_dim] = token_state.shape().dims::<3>();
        let bias = step_mode_embeddings
            .val()
            .slice_dim(0, mode.index()..mode.index() + 1)
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, time);
        token_state + bias
    }

    fn apply_cellular_step_mode_4d(
        &self,
        current: Tensor<B, 4>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 4> {
        let [batch, views, time, dim] = current.shape().dims::<4>();
        let flat = current.reshape([batch * views, time, dim]);
        let flat = self.apply_cellular_step_mode(flat, mode);
        flat.reshape([batch, views, time, dim])
    }

    fn apply_cellular_recurrent_mode(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        mode: StructuredStepMode,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let Some(query_offsets) = &self.cellular_query_mode_offsets else {
            return (query, value);
        };
        let Some(value_offsets) = &self.cellular_value_mode_offsets else {
            return (query, value);
        };

        let [_, heads, _, latent] = query.shape().dims::<4>();
        let [_, _, _, embd] = value.shape().dims::<4>();
        let query_scale = query_offsets
            .val()
            .slice_dim(0, mode.index()..mode.index() + 1)
            .reshape([1, heads, 1, latent])
            .add_scalar(1.0);
        let value_scale = value_offsets
            .val()
            .slice_dim(0, mode.index()..mode.index() + 1)
            .reshape([1, 1, 1, embd])
            .add_scalar(1.0);

        (query * query_scale, value * value_scale)
    }

    fn cellular_patch_tokens_from_time(&self, time: usize) -> usize {
        if self.use_cls_token && time > 1 {
            time - 1
        } else {
            time
        }
    }

    fn cellular_rho_expected_shape(&self, token_state: &Tensor<B, 3>) -> [usize; 5] {
        let [batch, time, embd] = token_state.shape().dims::<3>();
        let latent = self.encoder.val().shape().dims::<3>()[2];
        [
            batch,
            self.n_head,
            self.cellular_patch_tokens_from_time(time),
            latent,
            embd,
        ]
    }

    fn empty_cellular_rho(&self, token_state: &Tensor<B, 3>) -> Tensor<B, 5> {
        Tensor::<B, 5>::zeros(
            self.cellular_rho_expected_shape(token_state),
            &token_state.device(),
        )
    }

    fn assert_cellular_rho_matches_tokens(&self, rho: &Tensor<B, 5>, token_state: &Tensor<B, 3>) {
        let actual = rho.shape().dims::<5>();
        let expected = self.cellular_rho_expected_shape(token_state);
        assert_eq!(
            actual, expected,
            "cellular rho state shape {:?} does not match token state contract {:?}",
            actual, expected
        );
    }

    fn validate_cellular_state(&self, state: &VisionCellularState<B>) {
        self.assert_cellular_rho_matches_tokens(&state.rho, &state.token_state);
    }

    fn scalar_cellular_decay(&self, decay: f32, device: &B::Device) -> Tensor<B, 1> {
        Tensor::<B, 1>::from_data(
            TensorData::new(vec![decay; self.n_head.max(1)], [self.n_head.max(1)]),
            device,
        )
    }

    fn cellular_decay_by_mode(&self, mode: StructuredStepMode, device: &B::Device) -> Tensor<B, 1> {
        let dt = mode.temporal_dt();
        if dt == 0 {
            return Tensor::<B, 1>::ones([self.n_head.max(1)], device);
        }
        if self.use_alibi
            && let Some(slopes) = self.alibi_slopes.as_ref()
        {
            return slopes.clone().mul_scalar(-(dt as f32)).exp();
        }
        self.scalar_cellular_decay(
            self.rho_stream.decay.clamp(0.0, 1.0).powf(dt as f32),
            device,
        )
    }

    fn rollout_cellular_state_unbounded(
        &self,
        mut state: VisionCellularState<B>,
        steps: usize,
        detach_until: usize,
        mode: StructuredStepMode,
    ) -> VisionCellularState<B> {
        let decay = self.cellular_decay_by_mode(mode, &state.token_state.device());
        let [batch, time, _] = state.token_state.shape().dims::<3>();
        let current = state.token_state.reshape([batch, 1, time, self.embed_dim]);
        let (current, rho) = match self.rho_stream_rollout_executor_mode() {
            RhoStreamRolloutExecutorMode::HostLoop => self
                .rollout_rho_stream_from_current_host_loop(
                    current,
                    Some(state.rho),
                    steps,
                    detach_until,
                    decay,
                    mode,
                ),
            RhoStreamRolloutExecutorMode::WgpuFused => self
                .rollout_rho_stream_from_current_wgpu_rollout_fused(
                    current,
                    Some(state.rho),
                    steps,
                    detach_until,
                    decay,
                    mode,
                ),
        };

        state.token_state = current.reshape([batch, time, self.embed_dim]);
        state.rho = rho.expect("cellular rollout must return rho state");
        let temporal_dt = mode.temporal_dt();
        if temporal_dt > 0 {
            state.temporal_position = state.temporal_position.saturating_add(temporal_dt);
            state.prediction_age = state.prediction_age.saturating_add(temporal_dt);
        } else if mode.resets_prediction_age() {
            state.prediction_age = 0;
        }
        state
    }

    fn encode_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encode_tokens_steps(tokens, self.steps)
    }

    fn encode_tokens_steps(&self, tokens: Tensor<B, 3>, steps: usize) -> Tensor<B, 3> {
        let steps = steps.max(1).min(self.steps);
        self.encode_tokens_steps_inner(tokens, steps, 0, true)
    }

    fn encode_tokens_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 3> {
        let steps = steps.max(1).min(self.steps);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner(tokens, steps, detach_until, true)
    }

    fn encode_tokens_steps_rollout_unbounded(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 3> {
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner(tokens, steps, detach_until, true)
    }

    fn encode_tokens_steps_rollout_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 4> {
        let steps = steps.max(1).min(self.steps);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner_multi(tokens, steps, detach_until)
    }

    fn encode_tokens_steps_inner(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        match self.backbone_kind {
            VisionBackboneKind::Cellular => {
                return self.encode_tokens_steps_inner_rho_stream(
                    tokens,
                    steps,
                    detach_until,
                    add_cls,
                );
            }
            VisionBackboneKind::Pyramid => {
                return self.encode_tokens_steps_inner_pyramid(
                    tokens,
                    steps,
                    detach_until,
                    add_cls,
                );
            }
            VisionBackboneKind::Dense => {}
        }

        self.encode_tokens_steps_inner_default(tokens, steps, detach_until, add_cls)
    }

    fn pyramid_backbone_fallback(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        reason: &str,
    ) -> Tensor<B, 3> {
        match self.trm_graph.grid_mismatch_policy {
            VisionTrmGridMismatchPolicy::FallbackDefault => {
                self.encode_tokens_steps_inner_default(tokens, steps, detach_until, false)
            }
            VisionTrmGridMismatchPolicy::Error => {
                panic!(
                    "pyramid backbone path unavailable: {reason}. Set `vision.trm_graph.grid_mismatch_policy = \"fallback_default\"` to allow explicit fallback."
                )
            }
        }
    }

    fn normalize_rollout_schedule(
        &self,
        schedule: &[(usize, usize)],
        unbounded: bool,
    ) -> Vec<(usize, usize)> {
        let max_supported_steps = self.steps.max(1);
        let mut normalized = schedule
            .iter()
            .map(|(step, backprop_steps)| {
                let step = if unbounded {
                    (*step).max(1)
                } else {
                    (*step).max(1).min(max_supported_steps)
                };
                let backprop_steps = if *backprop_steps == 0 {
                    step
                } else {
                    (*backprop_steps).max(1).min(step)
                };
                (step, backprop_steps)
            })
            .collect::<Vec<_>>();
        normalized.sort_unstable_by_key(|(step, _)| *step);
        let mut deduped: Vec<(usize, usize)> = Vec::with_capacity(normalized.len());
        for (step, backprop_steps) in normalized {
            if let Some((last_step, last_backprop)) = deduped.last_mut()
                && *last_step == step
            {
                *last_backprop = (*last_backprop).max(backprop_steps);
            } else {
                deduped.push((step, backprop_steps));
            }
        }
        deduped
    }

    fn advance_rollout_state_unbounded(
        &self,
        state: VisionRolloutState<B>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> VisionRolloutState<B> {
        match mode {
            StructuredStepMode::Refine => {
                self.refine_rollout_state_unbounded(state, steps, backprop_steps)
            }
            StructuredStepMode::Predict => {
                self.predict_rollout_state_unbounded(state, steps, backprop_steps)
            }
            StructuredStepMode::Observe => {
                panic!("observe rollout scheduling requires explicit observation tokens")
            }
        }
    }

    fn forward_rollout_state_schedule_impl(
        &self,
        initial_state: VisionRolloutState<B>,
        schedule: &[(usize, usize)],
        mode: StructuredStepMode,
        unbounded: bool,
    ) -> Vec<(usize, VisionRolloutState<B>)> {
        let normalized = self.normalize_rollout_schedule(schedule, unbounded);
        if normalized.is_empty() {
            return Vec::new();
        }
        if normalized.len() == 1 {
            let (step, backprop_steps) = normalized[0];
            return vec![(
                step,
                self.advance_rollout_state_unbounded(initial_state, step, backprop_steps, mode),
            )];
        }

        let mut starts = normalized
            .iter()
            .map(|(step, backprop_steps)| step.saturating_sub(*backprop_steps))
            .collect::<Vec<_>>();
        starts.push(0);
        starts.sort_unstable();
        starts.dedup();

        let mut cached = BTreeMap::new();
        cached.insert(0usize, initial_state);
        let mut previous_start = 0usize;
        for &start in starts.iter().skip(1) {
            let previous_state = cached
                .get(&previous_start)
                .expect("rollout prefix state")
                .clone();
            let delta = start.saturating_sub(previous_start);
            let state = self
                .advance_rollout_state_unbounded(previous_state, delta, delta, mode)
                .detach();
            cached.insert(start, state);
            previous_start = start;
        }

        normalized
            .into_iter()
            .map(|(step, backprop_steps)| {
                let start = step.saturating_sub(backprop_steps);
                let start_state = cached.get(&start).expect("rollout start state").clone();
                let final_state = self.advance_rollout_state_unbounded(
                    start_state,
                    backprop_steps,
                    backprop_steps,
                    mode,
                );
                (step, final_state)
            })
            .collect()
    }

    fn forward_tokens_steps_rollout_schedule_impl(
        &self,
        tokens: Tensor<B, 3>,
        schedule: &[(usize, usize)],
        unbounded: bool,
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        let normalized = self.normalize_rollout_schedule(schedule, unbounded);
        if normalized.is_empty() {
            return Vec::new();
        }

        if self.backbone_kind != VisionBackboneKind::Dense {
            let states = self.predict_rollout_state_schedule_unbounded(
                self.rollout_state_from_tokens(tokens),
                &normalized,
            );
            return states
                .into_iter()
                .map(|(step, state)| (step, self.forward_rollout_state(&state)))
                .collect();
        }

        if normalized.len() == 1 {
            return normalized
                .into_iter()
                .map(|(step, backprop_steps)| {
                    let output = if unbounded {
                        self.forward_tokens_steps_rollout_unbounded(
                            tokens.clone(),
                            step,
                            backprop_steps,
                        )
                    } else {
                        self.forward_tokens_steps_rollout(tokens.clone(), step, backprop_steps)
                    };
                    (step, output)
                })
                .collect();
        }

        let initial_state = self.prepare_token_state(tokens, true);
        let cached = self.cache_dense_rollout_prefix_states(initial_state, &normalized);
        self.forward_dense_rollout_schedule_from_cached(&normalized, &cached)
    }

    fn cache_dense_rollout_prefix_states(
        &self,
        initial_state: Tensor<B, 3>,
        schedule: &[(usize, usize)],
    ) -> BTreeMap<usize, Tensor<B, 3>> {
        let mut starts = schedule
            .iter()
            .map(|(step, backprop_steps)| step.saturating_sub(*backprop_steps))
            .collect::<Vec<_>>();
        starts.push(0);
        starts.sort_unstable();
        starts.dedup();

        let mut cached = BTreeMap::new();
        cached.insert(0usize, initial_state);
        let mut previous_start = 0usize;
        for &start in starts.iter().skip(1) {
            let previous_state = cached
                .get(&previous_start)
                .expect("rollout prefix state")
                .clone();
            let delta = start.saturating_sub(previous_start);
            let state = self.rollout_dense_prepared_state(previous_state, delta, delta);
            cached.insert(start, state);
            previous_start = start;
        }
        cached
    }

    fn forward_dense_rollout_schedule_from_cached(
        &self,
        schedule: &[(usize, usize)],
        cached: &BTreeMap<usize, Tensor<B, 3>>,
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        if matches!(
            self.kernel.attention_executor,
            FusedAttentionExecutor::ScoresOnly
        ) {
            let min_backprop = schedule.iter().map(|(_, backprop_steps)| *backprop_steps).min();
            let max_backprop = schedule.iter().map(|(_, backprop_steps)| *backprop_steps).max();
            if let (Some(min_backprop), Some(max_backprop)) = (min_backprop, max_backprop) {
                if max_backprop > min_backprop && max_backprop.saturating_sub(min_backprop) == 1 {
                    return self.forward_dense_rollout_schedule_from_cached_unified(
                        schedule,
                        cached,
                        max_backprop,
                    );
                }
            }
        }

        let batch = cached
            .get(&0)
            .expect("initial dense rollout state")
            .shape()
            .dims::<3>()[0];
        let mut grouped: BTreeMap<usize, Vec<(usize, usize, Tensor<B, 3>)>> = BTreeMap::new();
        for (index, &(step, backprop_steps)) in schedule.iter().enumerate() {
            let start = step.saturating_sub(backprop_steps);
            let start_state = cached.get(&start).expect("rollout start state").clone();
            grouped
                .entry(backprop_steps)
                .or_default()
                .push((index, step, start_state));
        }

        let mut outputs: Vec<Option<(usize, VisionDragonOutput<B>)>> = vec![None; schedule.len()];
        for (backprop_steps, requests) in grouped {
            if requests.len() == 1 {
                let (index, step, start_state) = requests
                    .into_iter()
                    .next()
                    .expect("single dense rollout schedule request");
                let final_state = self.rollout_dense_prepared_state(start_state, backprop_steps, 0);
                let projected = self.projection.forward(final_state);
                outputs[index] = Some((step, self.split_output(projected)));
                continue;
            }

            let packed_start = Tensor::cat(
                requests
                    .iter()
                    .map(|(_, _, start_state)| start_state.clone())
                    .collect(),
                0,
            );
            let packed_final = self.rollout_dense_prepared_state(packed_start, backprop_steps, 0);
            let packed_projected = self.projection.forward(packed_final);

            let mut batch_offset = 0usize;
            for (index, step, _) in requests {
                let next_offset = batch_offset + batch;
                let projected = packed_projected
                    .clone()
                    .slice_dim(0, batch_offset..next_offset);
                outputs[index] = Some((step, self.split_output(projected)));
                batch_offset = next_offset;
            }
        }

        outputs
            .into_iter()
            .map(|output| output.expect("dense rollout schedule output"))
            .collect()
    }

    fn forward_dense_rollout_schedule_from_cached_unified(
        &self,
        schedule: &[(usize, usize)],
        cached: &BTreeMap<usize, Tensor<B, 3>>,
        exec_steps: usize,
    ) -> Vec<(usize, VisionDragonOutput<B>)> {
        let batch = cached
            .get(&0)
            .expect("initial dense rollout state")
            .shape()
            .dims::<3>()[0];
        let mut requests = schedule
            .iter()
            .enumerate()
            .map(|(index, &(step, backprop_steps))| {
                let start = step.saturating_sub(backprop_steps);
                let checkpoint = step.saturating_sub(start).max(1);
                let start_state = cached.get(&start).expect("rollout start state").clone();
                (checkpoint, index, step, start_state)
            })
            .collect::<Vec<_>>();
        requests.sort_by_key(|(checkpoint, _, _, _)| *checkpoint);

        let packed_start = Tensor::cat(
            requests
                .iter()
                .map(|(_, _, _, start_state)| start_state.clone())
                .collect(),
            0,
        );
        let mut outputs: Vec<Option<(usize, VisionDragonOutput<B>)>> = vec![None; schedule.len()];
        let [packed_batch, time, _] = packed_start.shape().dims::<3>();
        let mut current = packed_start.reshape([packed_batch, 1, time, self.embed_dim]);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };

        let mut request_offset = 0usize;
        for step_idx in 0..exec_steps {
            let output = lowrank_residual_step(
                current,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                sparse_mask.clone(),
                |query, value| self.full_attention(query, value),
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );
            current = output.next;

            let completed_step = step_idx + 1;
            while request_offset < requests.len() && requests[request_offset].0 == completed_step {
                let mut next_offset = request_offset + 1;
                while next_offset < requests.len() && requests[next_offset].0 == completed_step {
                    next_offset += 1;
                }

                let state = current
                    .clone()
                    .reshape([packed_batch, time, self.embed_dim])
                    .slice_dim(0, request_offset * batch..next_offset * batch);
                let projected = self.projection.forward(state);

                let mut batch_offset = 0usize;
                for (_, index, step, _) in &requests[request_offset..next_offset] {
                    let next_batch = batch_offset + batch;
                    let projected = projected.clone().slice_dim(0, batch_offset..next_batch);
                    outputs[*index] = Some((*step, self.split_output(projected)));
                    batch_offset = next_batch;
                }

                request_offset = next_offset;
            }
        }

        outputs
            .into_iter()
            .map(|output| output.expect("dense rollout schedule output"))
            .collect()
    }

    fn encode_tokens_steps_inner_default(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, _] = tokens.shape().dims::<3>();
        let current = tokens.reshape([batch, 1, time, self.embed_dim]);
        let current = self.apply_token_norm(current);
        let current = self.rollout_dense_prepared_state(
            current.reshape([batch, time, self.embed_dim]),
            steps,
            detach_until,
        );
        current
    }

    fn rollout_dense_prepared_state(
        &self,
        current: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
    ) -> Tensor<B, 3> {
        let [batch, time, _] = current.shape().dims::<3>();
        let mut current = current.reshape([batch, 1, time, self.embed_dim]);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };

        for step_idx in 0..steps {
            let output = lowrank_residual_step(
                current,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                sparse_mask.clone(),
                |query, value| self.full_attention(query, value),
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );
            current = output.next;
            if step_idx < detach_until {
                current = current.detach();
            }
        }

        current.reshape([batch, time, self.embed_dim])
    }

    fn encode_tokens_steps_inner_rho_stream(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        match self.rho_stream_rollout_executor_mode() {
            RhoStreamRolloutExecutorMode::HostLoop => self
                .encode_tokens_steps_inner_rho_stream_host_loop(
                    tokens,
                    steps,
                    detach_until,
                    add_cls,
                ),
            RhoStreamRolloutExecutorMode::WgpuFused => self
                .encode_tokens_steps_inner_rho_stream_wgpu_rollout_fused(
                    tokens,
                    steps,
                    detach_until,
                    add_cls,
                ),
        }
    }

    fn rollout_rho_stream_from_current_host_loop(
        &self,
        mut current: Tensor<B, 4>,
        mut rho_state: Option<Tensor<B, 5>>,
        steps: usize,
        detach_until: usize,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> (Tensor<B, 4>, Option<Tensor<B, 5>>) {
        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };

        for step_idx in 0..steps {
            let step_current = self.apply_cellular_step_mode_4d(current, mode);
            let output = lowrank_residual_step(
                step_current,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                sparse_mask.clone(),
                |query, value| {
                    self.rho_stream_attention_with_decay(
                        query,
                        value,
                        &mut rho_state,
                        decay.clone(),
                        mode,
                    )
                },
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );
            current = output.next;
            if step_idx < detach_until {
                current = current.detach();
                rho_state = rho_state.map(|state| state.detach());
            }
        }

        (current, rho_state)
    }

    fn encode_tokens_steps_inner_rho_stream_host_loop(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let current = self.prepare_token_state(tokens, add_cls);
        let [batch, time, _] = current.shape().dims::<3>();
        let current = current.reshape([batch, 1, time, self.embed_dim]);
        let device = current.device();
        let (current, _) = self.rollout_rho_stream_from_current_host_loop(
            current,
            None,
            steps,
            detach_until,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        );
        current.reshape([batch, time, self.embed_dim])
    }

    fn rollout_rho_stream_from_current_wgpu_rollout_fused(
        &self,
        mut current: Tensor<B, 4>,
        mut rho_state: Option<Tensor<B, 5>>,
        steps: usize,
        detach_until: usize,
        decay: Tensor<B, 1>,
        mode: StructuredStepMode,
    ) -> (Tensor<B, 4>, Option<Tensor<B, 5>>) {
        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        let patch_tokens = if self.use_cls_token {
            current.shape().dims::<4>()[2].saturating_sub(1)
        } else {
            current.shape().dims::<4>()[2]
        };
        let fused_plan = if patch_tokens > 0 && self.rho_stream_wgpu_forward_enabled() {
            let grid = self.resolve_rho_stream_grid(patch_tokens);
            Some(CompiledLocalGridRhoPlan::new(
                LocalGridRhoPlanSpec {
                    batch: current.shape().dims::<4>()[0],
                    heads,
                    value_heads: 1,
                    patch_tokens,
                    latent,
                    embd: self.embed_dim,
                    grid: LocalGridShape2d::new(grid.height, grid.width),
                    neighborhood: self.resolve_rho_stream_neighborhood(),
                },
                &current.device(),
            ))
        } else {
            None
        };

        for step_idx in 0..steps {
            let step_current = self.apply_cellular_step_mode_4d(current, mode);
            let output = lowrank_residual_step(
                step_current,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                sparse_mask.clone(),
                |query, value| {
                    self.rho_stream_attention_fused_with_decay_plan(
                        query,
                        value,
                        &mut rho_state,
                        decay.clone(),
                        mode,
                        fused_plan.as_ref(),
                    )
                },
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );
            current = output.next;
            if step_idx < detach_until {
                current = current.detach();
                rho_state = rho_state.map(|state| state.detach());
            }
        }

        (current, rho_state)
    }

    fn encode_tokens_steps_inner_rho_stream_wgpu_rollout_fused(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let current = self.prepare_token_state(tokens, add_cls);
        let [batch, time, _] = current.shape().dims::<3>();
        let current = current.reshape([batch, 1, time, self.embed_dim]);
        let device = current.device();
        let (current, _) = self.rollout_rho_stream_from_current_wgpu_rollout_fused(
            current,
            None,
            steps,
            detach_until,
            self.scalar_cellular_decay(self.rho_stream.decay.clamp(0.0, 1.0), &device),
            StructuredStepMode::Predict,
        );
        current.reshape([batch, time, self.embed_dim])
    }

    fn encode_tokens_steps_inner_pyramid(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, dim] = tokens.shape().dims::<3>();
        if batch == 0 || time == 0 || dim == 0 {
            return tokens;
        }

        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        let patch_count = grid_height * grid_width;
        let (patch_tokens, has_cls) = if self.use_cls_token && time == patch_count + 1 {
            let patch = tokens.clone().slice_dim(1, 1..time);
            (patch, true)
        } else {
            (tokens.clone(), false)
        };

        let patch_len = patch_tokens.shape().dims::<3>()[1];
        if patch_len != patch_count {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                &format!(
                    "token count mismatch (got {patch_len}, expected {patch_count} from grid {}x{})",
                    grid_height, grid_width
                ),
            );
        }

        if self.pyramid_patch_x_neuron_proj.is_none()
            || self.pyramid_patch_to_coarse_query_proj.is_none()
            || self.pyramid_patch_to_global_query_proj.is_none()
            || self.pyramid_coarse_x_neuron_proj.is_none()
            || self.pyramid_coarse_to_global_query_proj.is_none()
        {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone query projection layers",
            );
        }
        if self.pyramid_write_value_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone projection layer `pyramid_write_value_proj`",
            );
        }
        if self.pyramid_patch_y_gate_proj.is_none() || self.pyramid_coarse_y_gate_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone y-gate projection layers",
            );
        }
        if self.pyramid_patch_delta_proj.is_none() || self.pyramid_coarse_delta_proj.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone delta projection layers",
            );
        }
        if self.pyramid_value_norm.is_none() {
            return self.pyramid_backbone_fallback(
                tokens,
                steps,
                detach_until,
                "missing pyramid backbone normalization layer `pyramid_value_norm`",
            );
        }

        let backprop_steps = steps.saturating_sub(detach_until).max(1);
        let state = self.pyramid_state_from_patch_tokens(patch_tokens);
        let state = self.forward_pyramid_state_rollout_mode_unbounded(
            state,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        );
        let patch_tokens = self
            .pyramid_patch_tokens(&state)
            .reshape([batch, patch_count, dim]);
        if has_cls {
            let cls = patch_tokens.clone().mean_dim(1).reshape([batch, 1, dim]);
            Tensor::cat(vec![cls, patch_tokens], 1)
        } else {
            patch_tokens
        }
    }

    fn encode_tokens_steps_inner_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        detach_until: usize,
    ) -> Tensor<B, 4> {
        let tokens = if self.use_cls_token {
            let [batch, streams, time, dim] = tokens.shape().dims::<4>();
            let flat = tokens.reshape([batch * streams, time, dim]);
            let flat = self.prepend_cls(flat);
            let [flat_batch, time, dim] = flat.shape().dims::<3>();
            let streams = (flat_batch / batch).max(1);
            flat.reshape([batch, streams, time, dim])
        } else {
            tokens
        };

        let [batch, streams, time, _] = tokens.shape().dims::<4>();
        let mut current = tokens.reshape([batch, streams, time, self.embed_dim]);
        current = self.apply_token_norm(current);

        if let Some(eye_token) = &self.eye_token {
            let eye = eye_token
                .val()
                .reshape([1, streams, 1, self.embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(2, time);
            current = current + eye;
        }
        current = self.sync_cls_tokens_multi(current);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };

        let mhc_coefficients = self.mhc_layers.as_ref().map(|layers| {
            layers
                .iter()
                .map(|mhc| mhc.coefficients())
                .collect::<Vec<_>>()
        });

        for step_idx in 0..steps {
            let mhc = self
                .mhc_layers
                .as_ref()
                .map(|layers| &layers[step_idx.min(layers.len().saturating_sub(1))]);
            let mhc_coefficients = mhc_coefficients.as_ref().and_then(|coefficients| {
                coefficients.get(step_idx.min(coefficients.len().saturating_sub(1)))
            });
            let (branch_input, residuals_base, beta) =
                mhc_split_with_coefficients(mhc, current, mhc_coefficients);

            let [batch, views, time, dim] = branch_input.shape().dims::<4>();
            let branch_flat = branch_input.reshape([batch * views, 1, time, dim]);

            let output = lowrank_residual_step(
                branch_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                sparse_mask.clone(),
                |query, value| self.full_attention(query, value),
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );

            let branch_out = output.next.reshape([batch, views, time, dim]);
            let next = mhc_merge_with_coefficients(
                mhc,
                branch_out,
                residuals_base,
                mhc_coefficients,
                beta,
            );

            current = self.sync_cls_tokens_multi(self.apply_token_norm(next));
            if step_idx < detach_until {
                current = current.detach();
            }
        }

        if self.cross_eye_steps > 0 && streams > 1 && time > 0 {
            let cross_steps = self.cross_eye_steps.min(self.steps);
            if cross_steps > 0 {
                let flat = current.reshape([batch, streams * time, self.embed_dim]);
                let mixed = self.encode_tokens_steps_inner(flat, cross_steps, 0, false);
                current = self.sync_cls_tokens_multi(self.apply_token_norm(mixed.reshape([
                    batch,
                    streams,
                    time,
                    self.embed_dim,
                ])));
            }
        }

        current.reshape([batch, streams, time, self.embed_dim])
    }
}

#[cfg(test)]
mod rho_stream_tests;
#[cfg(test)]
mod rollout_state_tests;
