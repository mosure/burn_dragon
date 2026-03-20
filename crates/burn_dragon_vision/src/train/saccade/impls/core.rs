#![allow(clippy::too_many_arguments)]

use crate::train::gdpo;
use crate::train::prelude::*;

mod recon;

type SaccadeFinalizeOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Option<Vec<GdpoPolicyInputs<B>>>,
    Option<(Tensor<B, 3>, Tensor<B, 3>)>,
);
type SaccadeArtifacts<B> = Option<(
    Vec<Tensor<B, 4>>,
    Tensor<B, 3>,
    Option<Tensor<B, 5>>,
    Vec<String>,
)>;
type SaccadeReconLossOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    SaccadeArtifacts<B>,
    Option<Vec<GdpoPolicyInputs<B>>>,
);
type ReconPerSampleOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Option<(Tensor<B, 3>, Tensor<B, 3>)>,
);

struct SaccadeRolloutContext<B: BackendTrait> {
    device: B::Device,
    images: Tensor<B, 4>,
    view_images: Vec<Tensor<B, 4>>,
    view_embed: Option<Tensor<B, 4>>,
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    patch_size: usize,
    embed_dim: usize,
    tokens: usize,
    mip_levels: Vec<Vec<SaccadeMipLevel<B>>>,
    grids: Vec<PatchGrid>,
    target_patches: Vec<Tensor<B, 3>>,
    loss_masks: Vec<Tensor<B, 2>>,
    laplacian_images: Option<Vec<SaccadeLaplacianImages<B>>>,
    base_grid: Tensor<B, 4>,
    traj_len: usize,
    num_eyes: usize,
    inner_steps: usize,
    traj_update_alpha: f32,
    rollout_steps: usize,
    detach_until: usize,
    tbptt_step_count: usize,
    tbptt_enabled: bool,
    low_mem_pre_rollout: bool,
    capture_traj: bool,
    capture_artifacts: bool,
    gdpo_enabled: bool,
    gdpo_group: usize,
    gdpo_policy_enabled: bool,
    info_reward_enabled: bool,
    info_stride: usize,
    detach_policy_from_recon: bool,
}

struct SaccadeArtifactState<B: BackendTrait> {
    traj_steps: Vec<Vec<(Tensor<B, 2>, Tensor<B, 2>)>>,
    frame_steps: Vec<Tensor<B, 4>>,
    last_patch_views: Vec<Tensor<B, 4>>,
}

impl<B: BackendTrait> SaccadeArtifactState<B> {
    fn new(rollout_steps: usize, capture_traj: bool, capture_artifacts: bool) -> Self {
        let traj_steps = if capture_traj {
            Vec::with_capacity(rollout_steps)
        } else {
            Vec::new()
        };
        let frame_steps = if capture_artifacts {
            Vec::with_capacity(rollout_steps)
        } else {
            Vec::new()
        };
        Self {
            traj_steps,
            frame_steps,
            last_patch_views: Vec::new(),
        }
    }
}

struct SaccadeRolloutState<B: BackendTrait> {
    trajs: Vec<Tensor<B, 3>>,
    state_levels: Vec<Tensor<B, 3>>,
    artifacts: SaccadeArtifactState<B>,
    log_prob_sum: Option<Tensor<B, 2>>,
    log_prob_sum_old: Option<Tensor<B, 2>>,
    hard_reward: Option<Tensor<B, 1>>,
    policy_steps: usize,
    clamp_rate_sum: Option<Tensor<B, 1>>,
    clamp_rate_count: usize,
    tbptt_policy_inputs: Vec<GdpoPolicyInputs<B>>,
    tbptt_step_idx: usize,
    tbptt_chunks: usize,
    tbptt_loss_sum: Option<Tensor<B, 1>>,
    tbptt_mask_sum: Option<Tensor<B, 1>>,
    tbptt_inv_sum: Option<Tensor<B, 1>>,
    tbptt_sigreg_sum: Option<Tensor<B, 1>>,
    started_backprop: bool,
}

impl<B: BackendTrait> SaccadeRolloutState<B> {
    fn new(
        trajs: Vec<Tensor<B, 3>>,
        state_levels: Vec<Tensor<B, 3>>,
        log_prob_sum: Option<Tensor<B, 2>>,
        log_prob_sum_old: Option<Tensor<B, 2>>,
        hard_reward: Option<Tensor<B, 1>>,
        capture_traj: bool,
        capture_artifacts: bool,
        rollout_steps: usize,
    ) -> Self {
        let artifacts = SaccadeArtifactState::new(rollout_steps, capture_traj, capture_artifacts);
        let clamp_rate_sum = log_prob_sum.as_ref().map(|log_prob_sum| {
            let [batch, _] = log_prob_sum.shape().dims::<2>();
            Tensor::<B, 1>::zeros([batch], &log_prob_sum.device())
        });
        Self {
            trajs,
            state_levels,
            artifacts,
            log_prob_sum,
            log_prob_sum_old,
            hard_reward,
            policy_steps: 0,
            clamp_rate_sum,
            clamp_rate_count: 0,
            tbptt_policy_inputs: Vec::new(),
            tbptt_step_idx: 0,
            tbptt_chunks: 0,
            tbptt_loss_sum: None,
            tbptt_mask_sum: None,
            tbptt_inv_sum: None,
            tbptt_sigreg_sum: None,
            started_backprop: false,
        }
    }

    fn reset_policy_accumulators(
        &mut self,
        gdpo_enabled: bool,
        info_reward_enabled: bool,
        batch: usize,
        device: &B::Device,
    ) {
        if gdpo_enabled {
            self.log_prob_sum = Some(Tensor::<B, 2>::zeros([batch, 1], device));
            self.log_prob_sum_old = Some(Tensor::<B, 2>::zeros([batch, 1], device));
            self.clamp_rate_sum = Some(Tensor::<B, 1>::zeros([batch], device));
        } else {
            self.log_prob_sum = None;
            self.log_prob_sum_old = None;
            self.clamp_rate_sum = None;
        }
        if info_reward_enabled {
            self.hard_reward = Some(Tensor::<B, 1>::zeros([batch], device));
        }
        self.policy_steps = 0;
        self.clamp_rate_count = 0;
    }
}

struct SaccadeStepScratch<B: BackendTrait> {
    updates: Vec<Tensor<B, 3>>,
    updates_null: Vec<Tensor<B, 3>>,
    step_capture: SaccadeStepCapture<B>,
    next_trajs: Vec<Tensor<B, 3>>,
}

struct SaccadeStepCapture<B: BackendTrait> {
    traj: Vec<(Tensor<B, 2>, Tensor<B, 2>)>,
    patches: Vec<Tensor<B, 4>>,
}

impl<B: BackendTrait> SaccadeStepCapture<B> {
    fn new() -> Self {
        Self {
            traj: Vec::new(),
            patches: Vec::new(),
        }
    }

    fn reset(&mut self, num_eyes: usize, capture_traj: bool, capture_artifacts: bool) {
        self.traj.clear();
        self.patches.clear();
        if capture_traj || capture_artifacts {
            self.traj.reserve(num_eyes);
        }
        if capture_artifacts {
            self.patches.reserve(num_eyes);
        }
    }
}

impl<B: BackendTrait> SaccadeStepScratch<B> {
    fn new() -> Self {
        Self {
            updates: Vec::new(),
            updates_null: Vec::new(),
            step_capture: SaccadeStepCapture::new(),
            next_trajs: Vec::new(),
        }
    }

    fn reset_for_step(
        &mut self,
        state_levels: &[Tensor<B, 3>],
        device: &B::Device,
        num_eyes: usize,
        capture_traj: bool,
        capture_artifacts: bool,
        collect_info: bool,
    ) {
        self.updates = state_levels
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), device))
            .collect();
        self.updates_null = if collect_info {
            state_levels
                .iter()
                .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), device))
                .collect()
        } else {
            Vec::new()
        };
        self.step_capture
            .reset(num_eyes, capture_traj, capture_artifacts);
        self.next_trajs.clear();
        self.next_trajs.reserve(num_eyes);
    }
}

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn new(
        model: VisionDragon<B>,
        config: VisionSaccadeConfig,
        embed_dim: usize,
        patch_size: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        train_repeats: usize,
        train_repeat_chunk: usize,
        device: &B::Device,
    ) -> Self {
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.loss.recon.hidden_dim,
            recon_patch_dim,
            config.loss.recon.recon_head_norm,
            &DragonNormConfig::default(),
            device,
        );
        let traj_tokens = config.traj_tokens.max(1);
        let trajectory_token = Tensor::<B, 2>::zeros([traj_tokens, embed_dim.max(1)], device);
        let num_eyes = config.num_eyes.max(1);
        let eye_token = if num_eyes > 1 {
            Tensor::<B, 2>::random(
                [num_eyes, embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            )
        } else {
            Tensor::<B, 2>::zeros([num_eyes, embed_dim.max(1)], device)
        };
        let view_embed = if config.cross_view.enabled && num_eyes > 1 {
            Some(LinearConfig::new(4, embed_dim.max(1)).init(device))
        } else {
            None
        };
        let cache_entries = config.cache.max_entries;
        let pyramid_dim = config
            .pyramid_feature_dim
            .filter(|&value| value > 0)
            .unwrap_or(embed_dim)
            .max(1);
        let input_proj = VisionSaccadeInputProjection::new(
            embed_dim,
            patch_size,
            &config.input_projection,
            device,
        );
        let fovea_proj =
            VisionSaccadeProjection::new(3, embed_dim, &DragonNormConfig::default(), device);
        let pyramid_in_proj = if pyramid_dim != embed_dim {
            Some(VisionSaccadeProjection::new(
                embed_dim,
                pyramid_dim,
                &DragonNormConfig::default(),
                device,
            ))
        } else {
            None
        };
        let pyramid_out_proj = if pyramid_dim != embed_dim {
            Some(VisionSaccadeProjection::new(
                pyramid_dim,
                embed_dim,
                &DragonNormConfig::default(),
                device,
            ))
        } else {
            None
        };
        let pyramid_norm = LayerNormConfig::new(pyramid_dim).init(device);
        let residual_proj = VisionSaccadeProjection::new(
            embed_dim,
            pyramid_dim,
            &DragonNormConfig::default(),
            device,
        );
        let saccade_head = VisionSaccadeHead::new(embed_dim, &DragonNormConfig::default(), device);
        Self {
            model,
            recon,
            trajectory_token: Param::from_tensor(trajectory_token),
            eye_token: Param::from_tensor(eye_token),
            view_embed,
            input_proj,
            fovea_proj,
            pyramid_in_proj,
            pyramid_out_proj,
            pyramid_norm,
            residual_proj,
            saccade_head,
            config,
            level_coords_cache: LevelCoordsCache::new(cache_entries),
            upsample_weights_cache: UpsampleWeightsCache::new(cache_entries),
            fovea_grid_cache: FoveaBaseGridCache::new(cache_entries),
            fovea_jitter_cache: FoveaJitterCache::new(cache_entries),
            pyramid_dim,
            rollout,
            train_repeats: train_repeats.max(1),
            train_repeat_chunk,
        }
    }

    pub(crate) fn detach_if<const D: usize>(tensor: Tensor<B, D>, detach: bool) -> Tensor<B, D> {
        if detach { tensor.detach() } else { tensor }
    }

    #[cfg(any(feature = "benchmark", test))]
    pub(crate) fn pyramid_feature_dim(&self) -> usize {
        self.pyramid_dim
    }

    pub(crate) fn project_pyramid_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(proj) = &self.pyramid_in_proj {
            proj.forward(tokens)
        } else {
            tokens
        }
    }

    pub(crate) fn project_pyramid_context(&self, context: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(proj) = &self.pyramid_out_proj {
            proj.forward(context)
        } else {
            self.pyramid_norm.forward(context)
        }
    }

    pub(crate) fn project_pyramid_level(&self, level: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_pyramid_context(level)
    }

    pub(crate) fn project_pyramid_levels(&self, levels: &[Tensor<B, 3>]) -> Vec<Tensor<B, 3>> {
        if let Some(proj) = &self.pyramid_out_proj {
            levels
                .iter()
                .map(|level| proj.forward(level.clone()))
                .collect()
        } else {
            levels.to_vec()
        }
    }

    fn recon_loss_per_sample_from_projected_levels_inner(
        &self,
        state_levels: &[Tensor<B, 3>],
        target_patches: &[Tensor<B, 3>],
        loss_masks: Option<&[Tensor<B, 2>]>,
        capture_base: bool,
    ) -> ReconPerSampleOutput<B> {
        let device = state_levels
            .first()
            .map(|level| level.device())
            .unwrap_or_default();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut mask_sum: Option<Tensor<B, 1>> = None;
        let mut base_pair = None;
        for (level_idx, (state_level, target_level)) in
            state_levels.iter().zip(target_patches.iter()).enumerate()
        {
            let pred_patches = self.recon.forward(state_level.clone());
            let [batch, level_tokens, patch_dim] = pred_patches.shape().dims::<3>();
            if batch == 0 || level_tokens == 0 || patch_dim == 0 {
                continue;
            }
            let diff = pred_patches.clone() - target_level.clone();
            let (per_sample, mask_sum_level) =
                if let Some(mask) = loss_masks.and_then(|masks| masks.get(level_idx)) {
                    let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
                    let per_sample = diff
                        .powf_scalar(2.0)
                        .mul(mask_expanded)
                        .sum_dim(2)
                        .sum_dim(1)
                        .reshape([batch]);
                    let mask_sum_level = mask
                        .clone()
                        .sum_dim(1)
                        .reshape([batch])
                        .mul_scalar(patch_dim as f32);
                    (per_sample, mask_sum_level)
                } else {
                    let per_sample = diff.powf_scalar(2.0).sum_dim(2).sum_dim(1).reshape([batch]);
                    let mask_sum_level = Tensor::<B, 1>::ones([batch], &device)
                        .mul_scalar((level_tokens * patch_dim) as f32);
                    (per_sample, mask_sum_level)
                };
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + per_sample.clone(),
                None => per_sample,
            });
            mask_sum = Some(match mask_sum {
                Some(accum) => accum + mask_sum_level,
                None => mask_sum_level,
            });
            if capture_base && level_idx == 0 {
                base_pair = Some((pred_patches, target_level.clone()));
            }
        }
        let Some(loss_sum) = loss_sum else {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        };
        let Some(mask_sum) = mask_sum else {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        };
        (loss_sum, mask_sum, base_pair)
    }

    fn recon_loss_per_sample_from_projected_levels(
        &self,
        state_levels: &[Tensor<B, 3>],
        target_patches: &[Tensor<B, 3>],
        loss_masks: Option<&[Tensor<B, 2>]>,
        capture_base: bool,
    ) -> ReconPerSampleOutput<B> {
        let device = state_levels
            .first()
            .map(|level| level.device())
            .unwrap_or_default();
        let [batch, _, _] = state_levels
            .first()
            .map(|level| level.shape().dims::<3>())
            .unwrap_or([0, 0, 0]);
        if batch == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let chunk_override = if self.config.recon_batch_chunk > 0 {
            Some(self.config.recon_batch_chunk)
        } else {
            None
        };
        let max_elems = self.config.recon_max_elems.max(1);
        let [_, tokens, dim] = state_levels
            .first()
            .map(|level| level.shape().dims::<3>())
            .unwrap_or([batch, 0, 0]);
        let denom = tokens.max(1).saturating_mul(dim.max(1));
        let auto_chunk = max_elems
            .checked_div(denom)
            .unwrap_or(batch)
            .max(1)
            .min(batch);
        let chunk = chunk_override.unwrap_or(auto_chunk).max(1).min(batch);
        if batch <= chunk {
            return self.recon_loss_per_sample_from_projected_levels_inner(
                state_levels,
                target_patches,
                loss_masks,
                capture_base,
            );
        }

        let mut loss_chunks = Vec::new();
        let mut mask_chunks = Vec::new();
        let mut base_pair = None;
        let mut start = 0;
        while start < batch {
            let end = (start + chunk).min(batch);
            let state_chunk: Vec<Tensor<B, 3>> = state_levels
                .iter()
                .map(|level| level.clone().slice_dim(0, start..end))
                .collect();
            let target_chunk: Vec<Tensor<B, 3>> = target_patches
                .iter()
                .map(|level| level.clone().slice_dim(0, start..end))
                .collect();
            let mask_chunk: Option<Vec<Tensor<B, 2>>> = loss_masks.map(|masks| {
                masks
                    .iter()
                    .map(|mask| mask.clone().slice_dim(0, start..end))
                    .collect()
            });
            let (loss_chunk, mask_chunk, base_chunk) = self
                .recon_loss_per_sample_from_projected_levels_inner(
                    &state_chunk,
                    &target_chunk,
                    mask_chunk.as_deref(),
                    capture_base && base_pair.is_none(),
                );
            loss_chunks.push(loss_chunk);
            mask_chunks.push(mask_chunk);
            if base_pair.is_none() {
                base_pair = base_chunk;
            }
            start = end;
        }

        let loss_sum = if loss_chunks.is_empty() {
            Tensor::<B, 1>::zeros([1], &device)
        } else {
            Tensor::cat(loss_chunks, 0)
        };
        let mask_sum = if mask_chunks.is_empty() {
            Tensor::<B, 1>::zeros([1], &device)
        } else {
            Tensor::cat(mask_chunks, 0)
        };
        (loss_sum, mask_sum, base_pair)
    }

    fn recon_loss_per_sample_from_state_inner(
        &self,
        state_levels: &[Tensor<B, 3>],
        grids: &[PatchGrid],
        target_patches: &[Tensor<B, 3>],
        loss_masks: Option<&[Tensor<B, 2>]>,
        capture_base: bool,
    ) -> ReconPerSampleOutput<B> {
        let state_composed = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => state_levels.to_vec(),
            VisionPyramidMode::Laplacian => self.compose_pyramid(state_levels, grids),
        };
        let state_composed_embed = self.project_pyramid_levels(&state_composed);
        self.recon_loss_per_sample_from_projected_levels_inner(
            &state_composed_embed,
            target_patches,
            loss_masks,
            capture_base,
        )
    }

    pub(crate) fn recon_loss_per_sample_from_state(
        &self,
        state_levels: &[Tensor<B, 3>],
        grids: &[PatchGrid],
        target_patches: &[Tensor<B, 3>],
        loss_masks: Option<&[Tensor<B, 2>]>,
        capture_base: bool,
    ) -> ReconPerSampleOutput<B> {
        let device = state_levels
            .first()
            .map(|level| level.device())
            .unwrap_or_default();
        let [batch, _, _] = state_levels
            .first()
            .map(|level| level.shape().dims::<3>())
            .unwrap_or([0, 0, 0]);
        if batch == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let chunk_override = if self.config.recon_batch_chunk > 0 {
            Some(self.config.recon_batch_chunk)
        } else {
            None
        };
        let max_elems = self.config.recon_max_elems.max(1);
        let [_, tokens, dim] = state_levels
            .first()
            .map(|level| level.shape().dims::<3>())
            .unwrap_or([batch, 0, 0]);
        let mut approx_dim = dim.max(1);
        if self.pyramid_out_proj.is_some() {
            approx_dim = approx_dim.saturating_mul(2);
        }
        let denom = tokens.max(1).saturating_mul(approx_dim.max(1));
        let auto_chunk = max_elems
            .checked_div(denom)
            .unwrap_or(batch)
            .max(1)
            .min(batch);
        let chunk = chunk_override.unwrap_or(auto_chunk).max(1).min(batch);
        if batch <= chunk {
            return self.recon_loss_per_sample_from_state_inner(
                state_levels,
                grids,
                target_patches,
                loss_masks,
                capture_base,
            );
        }

        let mut loss_chunks = Vec::new();
        let mut mask_chunks = Vec::new();
        let mut base_pair = None;
        let mut start = 0;
        while start < batch {
            let end = (start + chunk).min(batch);
            let state_chunk: Vec<Tensor<B, 3>> = state_levels
                .iter()
                .map(|level| level.clone().slice_dim(0, start..end))
                .collect();
            let target_chunk: Vec<Tensor<B, 3>> = target_patches
                .iter()
                .map(|level| level.clone().slice_dim(0, start..end))
                .collect();
            let mask_chunk: Option<Vec<Tensor<B, 2>>> = loss_masks.map(|masks| {
                masks
                    .iter()
                    .map(|mask| mask.clone().slice_dim(0, start..end))
                    .collect()
            });
            let (loss_chunk, mask_chunk, base_chunk) = self.recon_loss_per_sample_from_state_inner(
                &state_chunk,
                grids,
                &target_chunk,
                mask_chunk.as_deref(),
                capture_base && base_pair.is_none(),
            );
            loss_chunks.push(loss_chunk);
            mask_chunks.push(mask_chunk);
            if base_pair.is_none() {
                base_pair = base_chunk;
            }
            start = end;
        }
        let loss_sum = if loss_chunks.is_empty() {
            Tensor::<B, 1>::zeros([1], &device)
        } else {
            Tensor::cat(loss_chunks, 0)
        };
        let mask_sum = if mask_chunks.is_empty() {
            Tensor::<B, 1>::zeros([1], &device)
        } else {
            Tensor::cat(mask_chunks, 0)
        };
        (loss_sum, mask_sum, base_pair)
    }

    pub(crate) fn fovea_base_grid(&self, patch_size: usize, device: &B::Device) -> Tensor<B, 4> {
        self.fovea_grid_cache.get_or_build(patch_size, device)
    }

    pub(crate) fn fovea_jitter(
        &self,
        patch_size: usize,
        subsamples_axis: usize,
        device: &B::Device,
    ) -> FoveaJitter<B> {
        self.fovea_jitter_cache
            .get_or_build(patch_size, subsamples_axis, device)
    }

    pub(crate) fn build_gdpo_policy_loss<F>(
        &self,
        gdpo: &burn_dragon_train::GdpoConfig,
        inputs: GdpoPolicyInputs<B>,
        advantage_fn: F,
    ) -> Option<Tensor<B, 1>>
    where
        F: FnOnce(Tensor<B, 2>, Tensor<B, 2>, &burn_dragon_train::GdpoConfig) -> Tensor<B, 2>,
    {
        let gdpo_group = inputs.gdpo_group;
        let batch = inputs.hard_reward.shape().dims::<1>()[0];
        if gdpo_group == 0 || batch == 0 || batch % gdpo_group != 0 {
            return None;
        }
        let scene_batch = batch / gdpo_group;
        let hard = inputs
            .hard_reward
            .detach()
            .reshape([scene_batch, gdpo_group]);
        let easy = inputs
            .recon_per_sample
            .mul_scalar(-1.0)
            .detach()
            .reshape([scene_batch, gdpo_group]);
        let advantage = advantage_fn(hard, easy, gdpo).reshape([batch, 1]).detach();
        Some(gdpo::gdpo_policy_loss(
            inputs.log_prob_sum,
            inputs.log_prob_sum_old,
            advantage,
            gdpo,
        ))
    }

    fn policy_log_prob_stats(
        &self,
        log_prob_sum: &Tensor<B, 2>,
        policy_steps: usize,
        num_eyes: usize,
        traj_len: usize,
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let device = log_prob_sum.device();
        let [batch, _] = log_prob_sum.shape().dims::<2>();
        if batch == 0 {
            let zero = Tensor::<B, 1>::zeros([batch.max(1)], &device);
            return (zero.clone(), zero);
        }
        let denom = (policy_steps.max(1) * num_eyes.max(1) * traj_len.max(1)) as f32;
        let log_prob_mean = log_prob_sum.clone().div_scalar(denom).reshape([batch]);
        let entropy = log_prob_mean.clone().mul_scalar(-1.0);
        (log_prob_mean, entropy)
    }

    fn policy_action_clamp_rate(
        &self,
        clamp_rate_sum: Option<Tensor<B, 1>>,
        clamp_rate_count: usize,
        batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 1> {
        if let Some(clamp_rate_sum) = clamp_rate_sum {
            clamp_rate_sum.div_scalar(clamp_rate_count.max(1) as f32)
        } else {
            Tensor::<B, 1>::zeros([batch.max(1)], device)
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        is_validation: bool,
    ) -> VisionSaccadeLosses<B> {
        let gdpo = &self.config.policy.gdpo;
        self.forward_losses_with_policy(
            batch,
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
            is_validation,
            |inputs| {
                self.build_gdpo_policy_loss(gdpo, inputs, |hard, easy, gdpo| {
                    gdpo::gdpo_advantage(hard, easy, gdpo)
                })
            },
        )
    }

    pub(crate) fn forward_losses_with_policy<F>(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        is_validation: bool,
        mut policy_loss_fn: F,
    ) -> VisionSaccadeLosses<B>
    where
        F: FnMut(GdpoPolicyInputs<B>) -> Option<Tensor<B, 1>>,
    {
        let ImageNetBatch {
            images,
            view_images,
            view_crops,
            labels,
            ..
        } = batch;
        let gdpo = &self.config.policy.gdpo;
        let gdpo_group = gdpo.group_size.max(1);
        let gdpo_active = gdpo.enabled && !capture_artifacts;
        let loss_on_all_patches = self
            .config
            .loss
            .recon
            .loss_on_all_patches_for(is_validation);
        let (images, labels) = if gdpo_active && gdpo_group > 1 {
            (
                images.repeat_dim(0, gdpo_group),
                labels.repeat_dim(0, gdpo_group),
            )
        } else {
            (images, labels)
        };
        let (loss_sum, mask_sum, inv, sigreg, artifacts, gdpo_inputs) = self.recon_loss(
            images,
            view_images,
            view_crops,
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
            loss_on_all_patches,
        );
        let sigreg = if gdpo_active && gdpo_group > 1 {
            sigreg.mul_scalar(1.0 / gdpo_group as f32)
        } else {
            sigreg
        };
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let recon_psnr = recon_psnr(recon.clone());
        let lambda = if self.config.loss.lejepa.enabled {
            self.config.loss.lejepa.lambda.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
        let recon_weight = self.config.loss.recon.weight.max(0.0);
        if recon_weight > 0.0 {
            total = total + recon.clone().mul_scalar(recon_weight);
        }
        let zero = Tensor::<B, 1>::zeros([1], &total.device());
        let mut policy_loss_sum: Option<Tensor<B, 1>> = None;
        let mut policy_loss_count = 0usize;
        let mut adv_abs_sum: Option<Tensor<B, 1>> = None;
        let mut adv_std_sum: Option<Tensor<B, 1>> = None;
        let mut adv_count = 0usize;
        let mut log_prob_sum: Option<Tensor<B, 1>> = None;
        let mut entropy_sum: Option<Tensor<B, 1>> = None;
        let mut clamp_sum: Option<Tensor<B, 1>> = None;
        let mut stat_count = 0usize;

        if let Some(inputs) = gdpo_inputs {
            for input in inputs {
                stat_count += 1;
                log_prob_sum = Some(match log_prob_sum {
                    Some(accum) => accum + input.log_prob_mean.clone(),
                    None => input.log_prob_mean.clone(),
                });
                entropy_sum = Some(match entropy_sum {
                    Some(accum) => accum + input.entropy.clone(),
                    None => input.entropy.clone(),
                });
                clamp_sum = Some(match clamp_sum {
                    Some(accum) => accum + input.action_clamp_rate.clone(),
                    None => input.action_clamp_rate.clone(),
                });

                let batch = input.hard_reward.shape().dims::<1>()[0];
                let gdpo_group = input.gdpo_group;
                if gdpo_group > 0 && batch > 0 && batch % gdpo_group == 0 {
                    let scene_batch = batch / gdpo_group;
                    let hard = input
                        .hard_reward
                        .clone()
                        .detach()
                        .reshape([scene_batch, gdpo_group]);
                    let easy = input
                        .recon_per_sample
                        .clone()
                        .mul_scalar(-1.0)
                        .detach()
                        .reshape([scene_batch, gdpo_group]);
                    let advantage = gdpo::gdpo_advantage(hard, easy, gdpo);
                    let adv_abs = advantage.clone().abs().mean();
                    let adv_mean = advantage.clone().mean();
                    let adv_mean_sq = adv_mean.clone().powf_scalar(2.0);
                    let adv_sq_mean = advantage.clone().powf_scalar(2.0).mean();
                    let adv_std = adv_sq_mean.sub(adv_mean_sq).clamp_min(0.0).sqrt();
                    adv_abs_sum = Some(match adv_abs_sum {
                        Some(accum) => accum + adv_abs,
                        None => adv_abs,
                    });
                    adv_std_sum = Some(match adv_std_sum {
                        Some(accum) => accum + adv_std,
                        None => adv_std,
                    });
                    adv_count += 1;
                }

                if let Some(loss) = policy_loss_fn(input) {
                    policy_loss_count += 1;
                    policy_loss_sum = Some(match policy_loss_sum {
                        Some(accum) => accum + loss,
                        None => loss,
                    });
                }
            }
        }

        let policy_loss = policy_loss_sum.map(|loss| {
            if policy_loss_count > 1 {
                loss.mul_scalar(1.0 / policy_loss_count as f32)
            } else {
                loss
            }
        });
        let policy = policy_loss.clone().unwrap_or_else(|| zero.clone());
        if let Some(policy_loss) = policy_loss {
            total = total + policy_loss;
        }
        let policy_advantage_abs_mean = if adv_count > 0 {
            adv_abs_sum
                .unwrap_or_else(|| zero.clone())
                .mul_scalar(1.0 / adv_count as f32)
        } else {
            zero.clone()
        };
        let policy_advantage_std = if adv_count > 0 {
            adv_std_sum
                .unwrap_or_else(|| zero.clone())
                .mul_scalar(1.0 / adv_count as f32)
        } else {
            zero.clone()
        };
        let policy_log_prob_mean = if stat_count > 0 {
            log_prob_sum
                .unwrap_or_else(|| zero.clone())
                .mul_scalar(1.0 / stat_count as f32)
        } else {
            zero.clone()
        };
        let policy_entropy = if stat_count > 0 {
            entropy_sum
                .unwrap_or_else(|| zero.clone())
                .mul_scalar(1.0 / stat_count as f32)
        } else {
            zero.clone()
        };
        let policy_action_clamp_rate = if stat_count > 0 {
            clamp_sum
                .unwrap_or_else(|| zero.clone())
                .mul_scalar(1.0 / stat_count as f32)
        } else {
            zero.clone()
        };

        let artifacts = artifacts.and_then(|(views, residual, frames, legend)| {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    ..VisionLejepaConfig::default()
                },
                &views,
                LejepaArtifactBuildInput {
                    frames,
                    first_patch: Some(residual),
                    pca_source: None,
                    patch_norms_steps: None,
                    pca_rgb_steps: None,
                    probe_logits: None,
                    labels: Some(labels),
                    legend: Some(legend),
                },
            )
        });

        VisionSaccadeLosses {
            total,
            inv,
            sigreg,
            recon,
            recon_psnr,
            policy,
            policy_advantage_abs_mean,
            policy_advantage_std,
            policy_log_prob_mean,
            policy_entropy,
            policy_action_clamp_rate,
            artifacts,
        }
    }
}
