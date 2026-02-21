#![allow(clippy::too_many_arguments)]

use crate::train::gdpo;
use crate::train::prelude::*;

type SaccadeFinalizeOutput<B> = (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Option<Vec<GdpoPolicyInputs<B>>>,
    Option<(Tensor<B, 3>, Tensor<B, 3>)>,
);
type SaccadeArtifacts<B> =
    Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>, Option<Tensor<B, 5>>, Vec<String>)>;
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
        let fovea_proj = VisionSaccadeProjection::new(3, embed_dim, device);
        let pyramid_in_proj = if pyramid_dim != embed_dim {
            Some(VisionSaccadeProjection::new(embed_dim, pyramid_dim, device))
        } else {
            None
        };
        let pyramid_out_proj = if pyramid_dim != embed_dim {
            Some(VisionSaccadeProjection::new(pyramid_dim, embed_dim, device))
        } else {
            None
        };
        let pyramid_norm = LayerNormConfig::new(pyramid_dim).init(device);
        let residual_proj = VisionSaccadeProjection::new(embed_dim, pyramid_dim, device);
        let saccade_head = VisionSaccadeHead::new(embed_dim, device);
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
            let (per_sample, mask_sum_level) = if let Some(mask) =
                loss_masks.and_then(|masks| masks.get(level_idx))
            {
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
                let per_sample = diff
                    .powf_scalar(2.0)
                    .sum_dim(2)
                    .sum_dim(1)
                    .reshape([batch]);
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
        let auto_chunk = if denom == 0 {
            batch
        } else {
            (max_elems / denom).max(1).min(batch)
        };
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
        let auto_chunk = if denom == 0 {
            batch
        } else {
            (max_elems / denom).max(1).min(batch)
        };
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
        F: FnOnce(
            Tensor<B, 2>,
            Tensor<B, 2>,
            &burn_dragon_train::GdpoConfig,
        ) -> Tensor<B, 2>,
    {
        let gdpo_group = inputs.gdpo_group;
        let batch = inputs.hard_reward.shape().dims::<1>()[0];
        if gdpo_group == 0 || batch == 0 || !batch.is_multiple_of(gdpo_group) {
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
                frames,
                Some(residual),
                None,
                None,
                None,
                None,
                Some(labels),
                Some(legend),
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

    fn run_recon_rollout(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
        scratch: &mut SaccadeStepScratch<B>,
    ) {
        for step_idx in 0..ctx.rollout_steps {
            let pre_rollout = ctx.low_mem_pre_rollout && step_idx < ctx.detach_until;
            let in_backprop = step_idx >= ctx.detach_until;
            if ctx.tbptt_enabled && in_backprop && !state.started_backprop {
                state.started_backprop = true;
                state.reset_policy_accumulators(
                    ctx.gdpo_enabled,
                    ctx.info_reward_enabled,
                    ctx.batch,
                    &ctx.device,
                );
            }
            let anchor_traj = ctx.low_mem_pre_rollout
                && ctx.detach_until > 0
                && step_idx + 1 == ctx.detach_until + 1;
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state.state_levels.clone(),
                VisionPyramidMode::Laplacian => {
                    self.compose_pyramid(&state.state_levels, &ctx.grids)
                }
            };
            let collect_info = ctx.info_reward_enabled && (step_idx % ctx.info_stride == 0);
            scratch.reset_for_step(
                &state.state_levels,
                &ctx.device,
                ctx.num_eyes,
                ctx.capture_traj,
                ctx.capture_artifacts,
                collect_info,
            );
            let mut eye_trajs = Vec::with_capacity(ctx.num_eyes);
            let mut eye_weights = Vec::with_capacity(ctx.num_eyes);
            let mut tokens_in_multi = Vec::with_capacity(ctx.num_eyes);
            let mut tokens_in_null_multi = if collect_info {
                Some(Vec::with_capacity(ctx.num_eyes))
            } else {
                None
            };
            let traj_denom = ctx.traj_len.max(1) as f32;

            for eye_idx in 0..ctx.num_eyes {
                let mut traj = state.trajs[eye_idx].clone();
                if pre_rollout {
                    traj = traj.detach();
                } else if anchor_traj {
                    let anchor = self
                        .trajectory_token
                        .val()
                        .reshape([1, ctx.traj_len, ctx.embed_dim])
                        .repeat_dim(0, ctx.batch);
                    traj = traj + (anchor.clone() - anchor.detach());
                }
                let eye_embed = self
                    .eye_token
                    .val()
                    .slice_dim(0, eye_idx..eye_idx + 1)
                    .reshape([1, 1, ctx.embed_dim])
                    .repeat_dim(0, ctx.batch)
                    .repeat_dim(1, ctx.traj_len);
                let eye_embed = Self::detach_if(eye_embed, pre_rollout);
                let mut traj_with_eye = traj.clone() + eye_embed.clone();
                if let Some(view_embed) = ctx.view_embed.as_ref() {
                    let [_, eyes, _, _] = view_embed.shape().dims::<4>();
                    if eye_idx < eyes {
                        let view_bias = view_embed
                            .clone()
                            .slice_dim(1, eye_idx..eye_idx + 1)
                            .reshape([ctx.batch, 1, ctx.embed_dim])
                            .repeat_dim(1, ctx.traj_len);
                        let view_bias = Self::detach_if(view_bias, pre_rollout);
                        traj_with_eye = traj_with_eye + view_bias;
                    }
                }
                let traj_with_eye = Self::detach_if(traj_with_eye, pre_rollout);
                let traj_summary = traj_with_eye
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1, ctx.embed_dim]);
                let params = self.saccade_head.forward(traj_summary);
                let params = Self::detach_if(params, pre_rollout);
                let (mean_raw, sigma_raw) = self.decode_saccade_params(params);
                let mean_raw = Self::detach_if(mean_raw, pre_rollout);
                let sigma_raw = Self::detach_if(sigma_raw, pre_rollout);
                let (mean_action, sigma_action) = if ctx.gdpo_enabled {
                    let sample = self.sample_policy_action(mean_raw.clone(), sigma_raw.clone());
                    let log_prob_eye = sample.log_prob.sum_dim(1).reshape([ctx.batch, 1]);
                    if let Some(log_prob_sum) = state.log_prob_sum.as_mut() {
                        *log_prob_sum = log_prob_sum.clone() + log_prob_eye.clone();
                    }
                    if let Some(log_prob_sum_old) = state.log_prob_sum_old.as_mut() {
                        *log_prob_sum_old = log_prob_sum_old.clone() + log_prob_eye.detach();
                    }
                    if let Some(clamp_rate_sum) = state.clamp_rate_sum.as_mut() {
                        *clamp_rate_sum = clamp_rate_sum.clone() + sample.clamp_rate.clone();
                    }
                    state.clamp_rate_count += 1;
                    (sample.mean, sample.sigma)
                } else {
                    (mean_raw.clone(), sigma_raw.clone())
                };
                let mean = if ctx.detach_policy_from_recon {
                    mean_action.clone().detach()
                } else {
                    mean_action.clone()
                };
                let sigma = if ctx.detach_policy_from_recon {
                    sigma_action.clone().detach()
                } else {
                    sigma_action.clone()
                };
                let mean_step = mean
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 2]);
                let sigma_step = sigma
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1]);
                let mean_detached = mean_step.clone().detach();
                let sigma_detached = sigma_step.clone().detach();
                if ctx.capture_traj || ctx.capture_artifacts {
                    scratch
                        .step_capture
                        .traj
                        .push((mean_detached.clone(), sigma_detached.clone()));
                }
                let eye_levels = ctx
                    .mip_levels
                    .get(eye_idx)
                    .unwrap_or_else(|| ctx.mip_levels.first().expect("mip levels"));
                let weights_context =
                    self.mip_gaussian_weights(eye_levels, mean.clone(), sigma.clone());
                let weights_scatter = weights_context.clone();
                let patch_image = self.foveated_patch_image(
                    eye_levels,
                    &ctx.base_grid,
                    mean_step.clone(),
                    sigma_step.clone(),
                    ctx.laplacian_images
                        .as_ref()
                        .and_then(|images| images.get(eye_idx)),
                );
                let patch_tokens = self.model.patch_embed_raw(patch_image.clone()).tokens;
                let patch_tokens = Self::detach_if(patch_tokens, pre_rollout);
                if ctx.capture_artifacts {
                    scratch.step_capture.patches.push(patch_image);
                }
                let input_context = patch_tokens.clone();
                let state_context = {
                    let context = self.mip_weighted_sum(&state_composed, &weights_context);
                    self.project_pyramid_context(context)
                };
                let input_tokens = self.build_input_tokens(
                    input_context,
                    state_context.clone(),
                    mean.clone(),
                    sigma.clone(),
                );
                let input_tokens = Self::detach_if(input_tokens, pre_rollout);
                let input_tokens = input_tokens.repeat_dim(1, ctx.traj_len);
                let mut tokens_in = traj_with_eye.clone() + input_tokens;
                tokens_in = Self::detach_if(tokens_in, pre_rollout);
                tokens_in_multi.push(tokens_in.unsqueeze_dim::<4>(1));

                if let Some(tokens_in_null_multi) = tokens_in_null_multi.as_mut() {
                    let null_patch_tokens = self.null_patch_tokens(&patch_tokens);
                    let input_tokens_null = self.build_input_tokens(
                        null_patch_tokens,
                        state_context.clone(),
                        mean.clone(),
                        sigma.clone(),
                    );
                    let input_tokens_null = Self::detach_if(input_tokens_null, pre_rollout);
                    let input_tokens_null = input_tokens_null.repeat_dim(1, ctx.traj_len);
                    let mut tokens_in_null = traj_with_eye.clone() + input_tokens_null;
                    tokens_in_null = Self::detach_if(tokens_in_null, pre_rollout);
                    tokens_in_null_multi.push(tokens_in_null.unsqueeze_dim::<4>(1));
                }

                eye_trajs.push(traj);
                eye_weights.push(weights_scatter);
            }

            let tokens_in_multi = Tensor::cat(tokens_in_multi, 1);
            let mut out_tokens_multi = self
                .model
                .forward_tokens_embed_steps_rollout_multi(
                    tokens_in_multi,
                    ctx.inner_steps,
                    ctx.inner_steps,
                )
                .patch_tokens;
            out_tokens_multi = Self::detach_if(out_tokens_multi, pre_rollout);

            let out_tokens_null_multi = if let Some(tokens_in_null_multi) = tokens_in_null_multi {
                let tokens_in_null = Tensor::cat(tokens_in_null_multi, 1);
                let mut out_tokens_null = self
                    .model
                    .forward_tokens_embed_steps_rollout_multi(
                        tokens_in_null,
                        ctx.inner_steps,
                        ctx.inner_steps,
                    )
                    .patch_tokens;
                out_tokens_null = Self::detach_if(out_tokens_null, pre_rollout);
                Some(out_tokens_null)
            } else {
                None
            };

            for (eye_idx, weights_for_eye) in eye_weights.iter().enumerate().take(ctx.num_eyes) {
                let traj = eye_trajs
                    .get(eye_idx)
                    .cloned()
                    .unwrap_or_else(|| state.trajs[eye_idx].clone());
                let out_tokens = out_tokens_multi
                    .clone()
                    .slice_dim(1, eye_idx..eye_idx + 1)
                    .reshape([ctx.batch, ctx.traj_len, ctx.embed_dim]);
                let residual = self.residual_proj.forward(out_tokens.clone());
                let residual = Self::detach_if(residual, pre_rollout);
                let residual_pool = residual
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1, self.pyramid_dim]);
                let next_traj = if ctx.traj_update_alpha >= 1.0 {
                    out_tokens
                } else {
                    let keep = 1.0 - ctx.traj_update_alpha;
                    traj.clone().mul_scalar(keep)
                        + out_tokens.clone().mul_scalar(ctx.traj_update_alpha)
                };
                for (update, weights) in scratch
                    .updates
                    .iter_mut()
                    .zip(weights_for_eye.iter())
                {
                    let update_eye = self.weighted_sum_tokens(
                        weights.clone().swap_dims(1, 2),
                        residual_pool.clone(),
                    );
                    *update = update.clone() + update_eye;
                }
                if let Some(out_tokens_null_multi) = out_tokens_null_multi.as_ref() {
                    let out_tokens_null = out_tokens_null_multi
                        .clone()
                        .slice_dim(1, eye_idx..eye_idx + 1)
                        .reshape([ctx.batch, ctx.traj_len, ctx.embed_dim]);
                    let residual_null = self.residual_proj.forward(out_tokens_null);
                    let residual_null = Self::detach_if(residual_null, pre_rollout);
                    let residual_pool_null = residual_null
                        .sum_dim(1)
                        .mul_scalar(1.0 / traj_denom)
                        .reshape([ctx.batch, 1, self.pyramid_dim]);
                    for (update, weights) in scratch
                        .updates_null
                        .iter_mut()
                        .zip(weights_for_eye.iter())
                    {
                        let update_eye_null = self.weighted_sum_tokens(
                            weights.clone().swap_dims(1, 2),
                            residual_pool_null.clone(),
                        );
                        *update = update.clone() + update_eye_null;
                    }
                }
                scratch.next_trajs.push(next_traj);
            }
            if ctx.gdpo_enabled {
                state.policy_steps += 1;
            }
            let state_real: Vec<Tensor<B, 3>> = state
                .state_levels
                .iter()
                .zip(scratch.updates.iter())
                .map(|(state, update)| state.clone() + update.clone())
                .collect();
            let state_null = if collect_info {
                Some(
                    state
                        .state_levels
                        .iter()
                        .zip(scratch.updates_null.iter())
                        .map(|(state, update)| state.clone() + update.clone())
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            state.state_levels = state_real;
            if collect_info
                && let (Some(hard_reward), Some(state_null)) =
                    (state.hard_reward.as_mut(), state_null)
            {
                let (real_sum, real_mask, _) = self.recon_loss_per_sample_from_state(
                    &state.state_levels,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    false,
                );
                let (null_sum, null_mask, _) = self.recon_loss_per_sample_from_state(
                    &state_null,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    false,
                );
                let real = real_sum / real_mask.add_scalar(LEJEPA_EPS);
                let null = null_sum / null_mask.add_scalar(LEJEPA_EPS);
                *hard_reward = hard_reward.clone() + (null - real);
            }
            if ctx.capture_traj || ctx.capture_artifacts {
                let step_traj = std::mem::take(&mut scratch.step_capture.traj);
                if ctx.capture_artifacts {
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state.state_levels.clone(),
                        VisionPyramidMode::Laplacian => {
                            self.compose_pyramid(&state.state_levels, &ctx.grids)
                        }
                    };
                    let pred_patches = self
                        .recon
                        .forward(self.project_pyramid_level(state_composed[0].clone()));
                    let recon_view = unpatchify(
                        pred_patches,
                        ctx.patch_size,
                        ctx.height,
                        ctx.width,
                        ctx.channels,
                    );
                    let mut combined_frame = ctx
                        .view_images
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ctx.images.clone());
                    let mut appended_patch = false;
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            combined_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            combined_frame = overlay;
                        }
                    }
                    let mut frame_views = Vec::new();
                    let push_view = |views: &mut Vec<Tensor<B, 4>>, view: Tensor<B, 4>| {
                        if !views.is_empty() && SACCADE_VIEW_GAP > 0 {
                            views.push(view_separator_like(&view, SACCADE_VIEW_GAP));
                        }
                        views.push(view);
                    };
                    push_view(&mut frame_views, combined_frame);
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        let mut eye_frame = ctx
                            .view_images
                            .get(eye_idx)
                            .cloned()
                            .unwrap_or_else(|| ctx.images.clone());
                        if let Some(overlay) = saccade_circle_overlay(
                            eye_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            eye_frame = overlay;
                        }
                        push_view(&mut frame_views, eye_frame);
                    }
                    let step_patches = std::mem::take(&mut scratch.step_capture.patches);
                    if !step_patches.is_empty()
                        && let Some(patch_views) = saccade_patch_views(step_patches, ctx.height)
                    {
                        state.artifacts.last_patch_views = patch_views
                            .iter()
                            .map(|view| view.clone().detach())
                            .collect();
                        for patch_view in patch_views {
                            push_view(&mut frame_views, patch_view);
                        }
                        appended_patch = true;
                    }
                    if !appended_patch && !state.artifacts.last_patch_views.is_empty() {
                        for patch_view in &state.artifacts.last_patch_views {
                            push_view(&mut frame_views, patch_view.clone());
                        }
                    }
                    push_view(&mut frame_views, recon_view);
                    let frame = Tensor::cat(frame_views, 3);
                    state.artifacts.frame_steps.push(frame);
                }
                if ctx.capture_traj {
                    state.artifacts.traj_steps.push(step_traj);
                }
            }
            state.trajs.clear();
            state.trajs.append(&mut scratch.next_trajs);
            if step_idx < ctx.detach_until {
                for traj in &mut state.trajs {
                    *traj = traj.clone().detach();
                }
                for level in &mut state.state_levels {
                    *level = level.clone().detach();
                }
            }
            if ctx.tbptt_enabled && in_backprop {
                state.tbptt_step_idx += 1;
                let chunk_done = state.tbptt_step_idx >= ctx.tbptt_step_count
                    || step_idx + 1 == ctx.rollout_steps;
                if chunk_done {
                    state.tbptt_step_idx = 0;
                    state.tbptt_chunks += 1;
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state.state_levels.clone(),
                        VisionPyramidMode::Laplacian => {
                            self.compose_pyramid(&state.state_levels, &ctx.grids)
                        }
                    };
                    let state_composed_embed = self.project_pyramid_levels(&state_composed);
                    let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                        self.pyramid_lejepa_loss(&state_composed_embed)
                    } else {
                        let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
                        (zero.clone(), zero)
                    };
                    let (loss_per_sample, mask_per_sample, _) = self
                        .recon_loss_per_sample_from_projected_levels(
                            &state_composed_embed,
                            &ctx.target_patches,
                            Some(&ctx.loss_masks),
                            false,
                        );
                    let loss_sum = loss_per_sample.clone().sum();
                    let mask_sum = mask_per_sample.clone().sum();
                    state.tbptt_loss_sum = Some(match state.tbptt_loss_sum.take() {
                        Some(accum) => accum + loss_sum.clone(),
                        None => loss_sum,
                    });
                    state.tbptt_mask_sum = Some(match state.tbptt_mask_sum.take() {
                        Some(accum) => accum + mask_sum.clone(),
                        None => mask_sum,
                    });
                    state.tbptt_inv_sum = Some(match state.tbptt_inv_sum.take() {
                        Some(accum) => accum + inv.clone(),
                        None => inv,
                    });
                    state.tbptt_sigreg_sum = Some(match state.tbptt_sigreg_sum.take() {
                        Some(accum) => accum + sigreg.clone(),
                        None => sigreg,
                    });
                    if ctx.gdpo_policy_enabled {
                        let recon_per_sample =
                            loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
                        let hard_reward = state
                            .hard_reward
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 1>::zeros([ctx.batch], &ctx.device));
                        let log_prob_sum = state
                            .log_prob_sum
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                        let log_prob_sum_old = state
                            .log_prob_sum_old
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                        let (log_prob_mean, entropy) = self.policy_log_prob_stats(
                            &log_prob_sum,
                            state.policy_steps,
                            ctx.num_eyes,
                            ctx.traj_len,
                        );
                        let action_clamp_rate = self.policy_action_clamp_rate(
                            state.clamp_rate_sum.take(),
                            state.clamp_rate_count,
                            ctx.batch,
                            &ctx.device,
                        );
                        state.tbptt_policy_inputs.push(GdpoPolicyInputs {
                            hard_reward,
                            recon_per_sample,
                            log_prob_sum,
                            log_prob_sum_old,
                            log_prob_mean,
                            entropy,
                            action_clamp_rate,
                            gdpo_group: ctx.gdpo_group,
                        });
                    }
                    if step_idx + 1 < ctx.rollout_steps {
                        state.reset_policy_accumulators(
                            ctx.gdpo_enabled,
                            ctx.info_reward_enabled,
                            ctx.batch,
                            &ctx.device,
                        );
                        for traj in &mut state.trajs {
                            *traj = traj.clone().detach();
                        }
                        for level in &mut state.state_levels {
                            *level = level.clone().detach();
                        }
                    }
                }
            }
        }
    }

    fn finalize_recon_loss(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
    ) -> SaccadeFinalizeOutput<B> {
        if ctx.tbptt_enabled {
            let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
            let chunk_count = state.tbptt_chunks.max(1) as f32;
            let inv_sum = state.tbptt_inv_sum.take().unwrap_or_else(|| zero.clone());
            let sigreg_sum = state
                .tbptt_sigreg_sum
                .take()
                .unwrap_or_else(|| zero.clone());
            let inv = inv_sum.mul_scalar(1.0 / chunk_count);
            let sigreg = sigreg_sum.mul_scalar(1.0 / chunk_count);
            let loss_sum = state.tbptt_loss_sum.take().unwrap_or_else(|| zero.clone());
            let mask_sum = state.tbptt_mask_sum.take().unwrap_or_else(|| zero.clone());
            let base_pair = if ctx.capture_artifacts {
                let (_, _, base_pair) = self.recon_loss_per_sample_from_state(
                    &state.state_levels,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    true,
                );
                base_pair
            } else {
                None
            };
            let gdpo_inputs = if ctx.gdpo_policy_enabled {
                Some(std::mem::take(&mut state.tbptt_policy_inputs))
            } else {
                None
            };
            (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair)
        } else {
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state.state_levels.clone(),
                VisionPyramidMode::Laplacian => {
                    self.compose_pyramid(&state.state_levels, &ctx.grids)
                }
            };
            let state_composed_embed = self.project_pyramid_levels(&state_composed);
            let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                self.pyramid_lejepa_loss(&state_composed_embed)
            } else {
                let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
                (zero.clone(), zero)
            };
            let (loss_per_sample, mask_per_sample, base_pair) = self
                .recon_loss_per_sample_from_projected_levels(
                    &state_composed_embed,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    ctx.capture_artifacts,
                );
            let loss_sum = loss_per_sample.clone().sum();
            let mask_sum = mask_per_sample.clone().sum();
            let recon_per_sample = loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
            let gdpo_inputs = if ctx.gdpo_policy_enabled {
                let hard_reward = state
                    .hard_reward
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 1>::zeros([ctx.batch], &ctx.device));
                let log_prob_sum = state
                    .log_prob_sum
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                let log_prob_sum_old = state
                    .log_prob_sum_old
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                let (log_prob_mean, entropy) = self.policy_log_prob_stats(
                    &log_prob_sum,
                    state.policy_steps,
                    ctx.num_eyes,
                    ctx.traj_len,
                );
                let action_clamp_rate = self.policy_action_clamp_rate(
                    state.clamp_rate_sum.take(),
                    state.clamp_rate_count,
                    ctx.batch,
                    &ctx.device,
                );
                Some(vec![GdpoPolicyInputs {
                    hard_reward,
                    recon_per_sample,
                    log_prob_sum,
                    log_prob_sum_old,
                    log_prob_mean,
                    entropy,
                    action_clamp_rate,
                    gdpo_group: ctx.gdpo_group,
                }])
            } else {
                None
            };
            (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair)
        }
    }

    fn build_recon_artifacts(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
        base_pair: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) -> SaccadeArtifacts<B> {
        if !ctx.capture_artifacts || ctx.batch == 0 || ctx.tokens == 0 {
            return None;
        }
        let (pred_base, target_base) = if let Some((pred, target)) = base_pair {
            (Some(pred), Some(target))
        } else {
            (None, None)
        };
        let pred_first = pred_base.clone().unwrap_or_else(|| {
            Tensor::<B, 3>::zeros(
                [
                    ctx.batch,
                    ctx.tokens,
                    ctx.patch_size * ctx.patch_size * ctx.channels,
                ],
                &ctx.device,
            )
        });
        let target_first = target_base.clone().unwrap_or_else(|| {
            Tensor::<B, 3>::zeros(
                [
                    ctx.batch,
                    ctx.tokens,
                    ctx.patch_size * ctx.patch_size * ctx.channels,
                ],
                &ctx.device,
            )
        });
        let recon_view = unpatchify(
            pred_first.clone(),
            ctx.patch_size,
            ctx.height,
            ctx.width,
            ctx.channels,
        );
        let residual = pred_first - target_first;
        let mut target_width = ctx.width;
        for view in &ctx.view_images {
            target_width = target_width.max(view.shape().dims::<4>()[3]);
        }
        if !state.artifacts.last_patch_views.is_empty() {
            for patch_view in &state.artifacts.last_patch_views {
                target_width = target_width.max(patch_view.shape().dims::<4>()[3]);
            }
        }
        let mut combined_view = ctx
            .view_images
            .first()
            .cloned()
            .unwrap_or_else(|| ctx.images.clone());
        let mut per_eye_views = Vec::new();
        if let Some(last_step) = state.artifacts.traj_steps.last() {
            for (eye_idx, (mean, sigma)) in last_step.iter().enumerate() {
                if let Some(overlay) = saccade_circle_overlay(
                    combined_view.clone(),
                    mean.clone(),
                    sigma.clone(),
                    saccade_eye_color(eye_idx),
                ) {
                    combined_view = overlay;
                }
                let mut eye_view = ctx
                    .view_images
                    .get(eye_idx)
                    .cloned()
                    .unwrap_or_else(|| ctx.images.clone());
                if let Some(overlay) = saccade_circle_overlay(
                    eye_view.clone(),
                    mean.clone(),
                    sigma.clone(),
                    saccade_eye_color(eye_idx),
                ) {
                    eye_view = overlay;
                }
                per_eye_views.push((eye_idx, eye_view));
            }
        }
        let combined_view = pad_view_width(combined_view, target_width);
        let recon_view = pad_view_width(recon_view, target_width);
        let patch_views = if state.artifacts.last_patch_views.is_empty() {
            None
        } else {
            let patch_views = std::mem::take(&mut state.artifacts.last_patch_views);
            Some(
                patch_views
                    .into_iter()
                    .map(|patch_view| pad_view_width_centered(patch_view, target_width))
                    .collect::<Vec<_>>(),
            )
        };
        let mut views = Vec::new();
        let mut legend = Vec::new();
        views.push(combined_view);
        legend.push("input_with_fovea".to_string());
        for (eye_idx, eye_view) in per_eye_views {
            views.push(pad_view_width(eye_view, target_width));
            legend.push(format!("input_with_fovea_eye_{eye_idx}"));
        }
        if let Some(patch_views) = patch_views {
            for (eye_idx, patch_view) in patch_views.into_iter().enumerate() {
                views.push(patch_view);
                legend.push(format!("foveated_patch_eye_{eye_idx}"));
            }
        }
        views.push(recon_view);
        legend.push("reconstruction".to_string());
        if !state.artifacts.traj_steps.is_empty() {
            let steps = std::mem::take(&mut state.artifacts.traj_steps);
            let max_extra = self.config.artifact_max_views.saturating_sub(views.len());
            let mut remaining = max_extra;
            for idx in select_trajectory_indices(steps.len(), max_extra) {
                for (eye_idx, (mean, sigma)) in steps[idx].iter().enumerate() {
                    if remaining == 0 {
                        break;
                    }
                    let base_view = ctx
                        .view_images
                        .get(eye_idx)
                        .cloned()
                        .unwrap_or_else(|| ctx.images.clone());
                    if let Some(view) = saccade_circle_overlay(
                        base_view,
                        mean.clone(),
                        sigma.clone(),
                        saccade_eye_color(eye_idx),
                    ) {
                        views.push(pad_view_width(view, target_width));
                        legend.push(format!("trajectory_overlay_step_{idx}_eye_{eye_idx}"));
                        remaining = remaining.saturating_sub(1);
                    }
                }
                if remaining == 0 {
                    break;
                }
            }
        }
        let frames = if state.artifacts.frame_steps.is_empty() {
            None
        } else {
            let frames = std::mem::take(&mut state.artifacts.frame_steps);
            if frames.is_empty() {
                None
            } else {
                let mut max_width = 0;
                for frame in &frames {
                    let width = frame.shape().dims::<4>()[3];
                    max_width = max_width.max(width);
                }
                let mut stacked = Vec::with_capacity(frames.len());
                for frame in frames {
                    let frame = pad_view_width(frame, max_width);
                    stacked.push(frame.unsqueeze_dim::<5>(1));
                }
                Some(Tensor::cat(stacked, 1))
            }
        };
        Some((views, residual, frames, legend))
    }

    pub(crate) fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        view_images: Option<Tensor<B, 5>>,
        view_crops: Option<Tensor<B, 3>>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        loss_on_all_patches: bool,
    ) -> SaccadeReconLossOutput<B> {
        let device = images.device();
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch_size = self.model.patch_size().max(1);
        let num_eyes = self.config.num_eyes.max(1);
        let cross_view = self.config.cross_view.enabled && view_images.is_some() && num_eyes > 1;

        let (mut eye_views, view_crops) = if cross_view {
            let views = view_images.expect("view images");
            let [view_batch, view_count, _, _, _] = views.shape().dims::<5>();
            if view_batch != batch || view_count == 0 {
                let mut eye_views = Vec::with_capacity(num_eyes);
                for _ in 0..num_eyes {
                    eye_views.push(images.clone());
                }
                (eye_views, None)
            } else {
                let usable_views = view_count.min(num_eyes).max(1);
                let mut eye_views = Vec::with_capacity(num_eyes);
                for eye_idx in 0..num_eyes {
                    if eye_idx < usable_views {
                        let view = views
                            .clone()
                            .slice_dim(1, eye_idx..eye_idx + 1)
                            .reshape([batch, channels, height, width]);
                        eye_views.push(view);
                    } else {
                        eye_views.push(images.clone());
                    }
                }
                let view_crops = view_crops.and_then(|crops| {
                    let [crop_batch, crop_views, _] = crops.shape().dims::<3>();
                    if crop_batch != batch || crop_views == 0 {
                        return None;
                    }
                    let crops = if crop_views >= num_eyes {
                        crops.slice_dim(1, 0..num_eyes)
                    } else {
                        let pad =
                            Tensor::<B, 3>::zeros([batch, num_eyes - crop_views, 4], &device);
                        Tensor::cat(vec![crops, pad], 1)
                    };
                    Some(crops)
                });
                (eye_views, view_crops)
            }
        } else {
            let mut eye_views = Vec::with_capacity(num_eyes);
            for _ in 0..num_eyes {
                eye_views.push(images.clone());
            }
            (eye_views, None)
        };

        let masked_eye = if cross_view {
            self.config
                .cross_view
                .masked_eye
                .min(num_eyes.saturating_sub(1))
        } else {
            0
        };
        let mask_ratio = if cross_view {
            self.config.loss.recon.mask_ratio
        } else {
            0.0
        };

        let (mip_levels, target_patches, loss_masks, grids, input_levels) = if cross_view {
            let (masked_levels, target_patches, masks) = self.build_masked_mip_pyramid(
                eye_views[masked_eye].clone(),
                patch_size,
                mask_ratio,
                randomize_mask,
                view_crops.clone(),
                masked_eye,
            );
            if masked_levels.is_empty() {
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
            }
            eye_views[masked_eye] = masked_levels[0].image.clone();
            let grids: Vec<PatchGrid> = masked_levels.iter().map(|level| level.grid).collect();
            let input_levels: Vec<Tensor<B, 3>> = masked_levels
                .iter()
                .map(|level| level.tokens.clone())
                .collect();
            let mut per_eye_levels = Vec::with_capacity(num_eyes);
            for (eye_idx, eye_view) in eye_views.iter().enumerate().take(num_eyes) {
                if eye_idx == masked_eye {
                    per_eye_levels.push(masked_levels.clone());
                } else {
                    let levels = self.build_mip_pyramid(eye_view.clone(), patch_size);
                    if levels.is_empty() {
                        let zero = Tensor::<B, 1>::zeros([1], &device);
                        return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
                    }
                    per_eye_levels.push(levels);
                }
            }
            let use_masks = !loss_on_all_patches && mask_ratio > 0.0;
            let loss_masks = if use_masks {
                masks.clone()
            } else {
                masks
                    .iter()
                    .map(|mask| {
                        let [mask_batch, mask_tokens] = mask.shape().dims::<2>();
                        Tensor::<B, 2>::ones([mask_batch, mask_tokens], &device)
                    })
                    .collect()
            };
            (per_eye_levels, target_patches, loss_masks, grids, input_levels)
        } else {
            let base_levels = self.build_mip_pyramid(images.clone(), patch_size);
            if base_levels.is_empty() {
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
            }
            let grids: Vec<PatchGrid> = base_levels.iter().map(|level| level.grid).collect();
            let input_levels: Vec<Tensor<B, 3>> = base_levels
                .iter()
                .map(|level| level.tokens.clone())
                .collect();
            let target_patches: Vec<Tensor<B, 3>> = base_levels
                .iter()
                .map(|level| patchify(level.image.clone(), patch_size))
                .collect();
            let loss_masks: Vec<Tensor<B, 2>> = target_patches
                .iter()
                .map(|patches| {
                    let [mask_batch, mask_tokens, _] = patches.shape().dims::<3>();
                    Tensor::<B, 2>::ones([mask_batch, mask_tokens], &device)
                })
                .collect();
            (vec![base_levels; num_eyes], target_patches, loss_masks, grids, input_levels)
        };

        let embed_dim = input_levels
            .first()
            .map(|level| level.shape().dims::<3>()[2])
            .unwrap_or(0)
            .max(1);
        let tokens = grids.first().map(|grid| grid.num_patches()).unwrap_or(0);
        let view_embed = if let (Some(view_crops), Some(view_embed)) =
            (view_crops, self.view_embed.as_ref())
        {
            let [crop_batch, crop_eyes, _] = view_crops.shape().dims::<3>();
            if crop_batch == 0 || crop_eyes == 0 {
                None
            } else {
                let flat = view_crops.clone().reshape([crop_batch * crop_eyes, 4]);
                let embed = view_embed
                    .forward(flat)
                    .reshape([crop_batch, crop_eyes, 1, embed_dim]);
                Some(embed)
            }
        } else {
            None
        };

        let input_state_levels = if cross_view {
            let mut averaged = Vec::with_capacity(input_levels.len());
            for level_idx in 0..input_levels.len() {
                let mut sum: Option<Tensor<B, 3>> = None;
                let mut count = 0.0f32;
                for levels in &mip_levels {
                    if let Some(level) = levels.get(level_idx) {
                        sum = Some(match sum {
                            Some(accum) => accum + level.tokens.clone(),
                            None => level.tokens.clone(),
                        });
                        count += 1.0;
                    }
                }
                let fallback = input_levels
                    .get(level_idx)
                    .cloned()
                    .unwrap_or_else(|| {
                        let [batch, tokens, dim] =
                            input_levels.first().map(|t| t.shape().dims::<3>()).unwrap_or([0, 0, 0]);
                        Tensor::<B, 3>::zeros([batch, tokens, dim], &device)
                    });
                let avg = sum
                    .map(|sum| sum.mul_scalar(1.0 / count.max(1.0)))
                    .unwrap_or(fallback);
                averaged.push(avg);
            }
            averaged
        } else {
            input_levels.clone()
        };
        let input_residuals = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => input_state_levels.clone(),
            VisionPyramidMode::Laplacian => self.decompose_pyramid(&input_state_levels, &grids),
        };
        let laplacian_images = if matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            let mut images = Vec::with_capacity(num_eyes);
            let mut ok = true;
            for levels in &mip_levels {
                if let Some(laplacian) = self.build_laplacian_images(levels) {
                    images.push(laplacian);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok { Some(images) } else { None }
        } else {
            None
        };
        let base_grid = self.fovea_base_grid(patch_size, &device);
        let traj_len = self.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let inner_steps = self.config.inner_steps.max(1);
        let traj_update_alpha = self.config.traj_update_alpha;

        let base_traj = self
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let trajs = vec![base_traj; num_eyes];
        let state_levels: Vec<Tensor<B, 3>> = if cross_view {
            input_residuals.clone()
        } else {
            input_residuals
                .iter()
                .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
                .collect()
        };
        let rollout_steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(rollout_steps);
        let detach_until = rollout_steps.saturating_sub(backprop_steps);
        let tbptt_step_count = self.config.tbptt.step_count;
        let tbptt_step_count = if tbptt_step_count == 0 {
            0
        } else {
            tbptt_step_count.max(1).min(backprop_steps)
        };
        let tbptt_enabled = tbptt_step_count > 0;
        let low_mem_pre_rollout = self.config.low_mem_pre_rollout;
        let capture_traj =
            capture_artifacts && self.config.artifact_max_views.saturating_sub(3) > 0;
        let gdpo = &self.config.policy.gdpo;
        let gdpo_enabled = gdpo.enabled && !capture_artifacts;
        let gdpo_group = gdpo.group_size.max(1);
        let gdpo_policy_enabled = gdpo_enabled && gdpo.policy_weight > 0.0;
        let info_reward_enabled = gdpo_enabled && self.config.policy.info_reward.enabled;
        let info_stride = self.config.policy.info_reward.stride.max(1);
        let detach_policy_from_recon = self.config.policy.detach_policy_from_recon;
        let log_prob_sum = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let log_prob_sum_old = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let hard_reward = if info_reward_enabled {
            Some(Tensor::<B, 1>::zeros([batch], &device))
        } else {
            None
        };
        // Heap-allocate rollout buffers to keep the stack frame small.
        let ctx = Box::new(SaccadeRolloutContext {
            device,
            images: eye_views
                .first()
                .cloned()
                .unwrap_or_else(|| images.clone()),
            view_images: eye_views,
            view_embed,
            batch,
            channels,
            height,
            width,
            patch_size,
            embed_dim,
            tokens,
            mip_levels,
            grids,
            target_patches,
            loss_masks,
            laplacian_images,
            base_grid,
            traj_len,
            num_eyes,
            inner_steps,
            traj_update_alpha,
            rollout_steps,
            detach_until,
            tbptt_step_count,
            tbptt_enabled,
            low_mem_pre_rollout,
            capture_traj,
            capture_artifacts,
            gdpo_enabled,
            gdpo_group,
            gdpo_policy_enabled,
            info_reward_enabled,
            info_stride,
            detach_policy_from_recon,
        });
        let mut state = Box::new(SaccadeRolloutState::new(
            trajs,
            state_levels,
            log_prob_sum,
            log_prob_sum_old,
            hard_reward,
            capture_traj,
            capture_artifacts,
            rollout_steps,
        ));
        let mut scratch = Box::new(SaccadeStepScratch::new());

        self.run_recon_rollout(&ctx, &mut state, &mut scratch);
        let (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair) =
            self.finalize_recon_loss(&ctx, &mut state);
        let artifacts = self.build_recon_artifacts(&ctx, &mut state, base_pair);

        (loss_sum, mask_sum, inv, sigreg, artifacts, gdpo_inputs)
    }
}

