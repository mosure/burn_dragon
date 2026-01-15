use crate::train::prelude::*;
use crate::train::gdpo;

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
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
            device,
        );
        let traj_tokens = config.traj_tokens.max(1);
        let trajectory_token =
            Tensor::<B, 2>::zeros([traj_tokens, embed_dim.max(1)], device);
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
        let residual_proj = VisionSaccadeProjection::new(embed_dim, pyramid_dim, device);
        let saccade_head = VisionSaccadeHead::new(embed_dim, device);
        Self {
            model,
            recon,
            trajectory_token: Param::from_tensor(trajectory_token),
            eye_token: Param::from_tensor(eye_token),
            input_proj,
            fovea_proj,
            pyramid_in_proj,
            pyramid_out_proj,
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
        if detach {
            tensor.detach()
        } else {
            tensor
        }
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
            context
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
            levels.iter().cloned().collect()
        }
    }

    fn recon_loss_per_sample_from_projected_levels_inner(
        &self,
        state_levels: &[Tensor<B, 3>],
        target_patches: &[Tensor<B, 3>],
        capture_base: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) {
        let device = state_levels
            .first()
            .map(|level| level.device())
            .unwrap_or_default();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut mask_sum_value = 0.0f32;
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
            let per_sample = diff
                .powf_scalar(2.0)
                .sum_dim(2)
                .sum_dim(1)
                .reshape([batch]);
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + per_sample,
                None => per_sample,
            });
            mask_sum_value += (level_tokens * patch_dim) as f32;
            if capture_base && level_idx == 0 {
                base_pair = Some((pred_patches, target_level.clone()));
            }
        }
        let Some(loss_sum) = loss_sum else {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        };
        if mask_sum_value == 0.0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let [batch] = loss_sum.shape().dims::<1>();
        let mask_sum = Tensor::<B, 1>::ones([batch], &device).mul_scalar(mask_sum_value);
        (loss_sum, mask_sum, base_pair)
    }

    fn recon_loss_per_sample_from_projected_levels(
        &self,
        state_levels: &[Tensor<B, 3>],
        target_patches: &[Tensor<B, 3>],
        capture_base: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) {
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
            let (loss_chunk, mask_chunk, base_chunk) =
                self.recon_loss_per_sample_from_projected_levels_inner(
                    &state_chunk,
                    &target_chunk,
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
        capture_base: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) {
        let state_composed = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => state_levels.to_vec(),
            VisionPyramidMode::Laplacian => self.compose_pyramid(state_levels, grids),
        };
        let state_composed_embed = self.project_pyramid_levels(&state_composed);
        self.recon_loss_per_sample_from_projected_levels_inner(
            &state_composed_embed,
            target_patches,
            capture_base,
        )
    }

    pub(crate) fn recon_loss_per_sample_from_state(
        &self,
        state_levels: &[Tensor<B, 3>],
        grids: &[PatchGrid],
        target_patches: &[Tensor<B, 3>],
        capture_base: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) {
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
            let (loss_chunk, mask_chunk, base_chunk) =
                self.recon_loss_per_sample_from_state_inner(
                    &state_chunk,
                    grids,
                    &target_chunk,
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
        gdpo: &crate::GdpoConfig,
        inputs: GdpoPolicyInputs<B>,
        advantage_fn: F,
    ) -> Option<Tensor<B, 1>>
    where
        F: FnOnce(Tensor<B, 2>, Tensor<B, 2>, &crate::GdpoConfig) -> Tensor<B, 2>,
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
        let advantage = advantage_fn(hard, easy, gdpo)
            .reshape([batch, 1])
            .detach();
        Some(gdpo::gdpo_policy_loss(
            inputs.log_prob_sum,
            inputs.log_prob_sum_old,
            advantage,
            gdpo,
        ))
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionSaccadeLosses<B> {
        let gdpo = &self.config.policy.gdpo;
        self.forward_losses_with_policy(
            batch,
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
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
        mut policy_loss_fn: F,
    ) -> VisionSaccadeLosses<B>
    where
        F: FnMut(GdpoPolicyInputs<B>) -> Option<Tensor<B, 1>>,
    {
        let ImageNetBatch { images, labels, .. } = batch;
        let gdpo = &self.config.policy.gdpo;
        let gdpo_group = gdpo.group_size.max(1);
        let gdpo_active = gdpo.enabled && !capture_artifacts;
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
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
        );
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
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
        let policy_loss = gdpo_inputs.and_then(|inputs| {
            let mut sum: Option<Tensor<B, 1>> = None;
            let mut count = 0usize;
            for input in inputs {
                if let Some(loss) = policy_loss_fn(input) {
                    count += 1;
                    sum = Some(match sum {
                        Some(accum) => accum + loss,
                        None => loss,
                    });
                }
            }
            sum.map(|loss| {
                if count > 1 {
                    loss.mul_scalar(1.0 / count as f32)
                } else {
                    loss
                }
            })
        });
        if let Some(policy_loss) = policy_loss {
            total = total + policy_loss;
        }

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
                Some(labels),
                Some(legend),
            )
        });

        VisionSaccadeLosses {
            total,
            inv,
            sigreg,
            recon,
            artifacts,
        }
    }

    pub(crate) fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>, Option<Tensor<B, 5>>, Vec<String>)>,
        Option<Vec<GdpoPolicyInputs<B>>>,
    ) {
        let device = images.device();
        let _ = randomize_mask;
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch_size = self.model.patch_size().max(1);
        let mip_levels = self.build_mip_pyramid(images.clone(), patch_size);
        if mip_levels.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
        }
        let input_levels: Vec<Tensor<B, 3>> =
            mip_levels.iter().map(|level| level.tokens.clone()).collect();
        let embed_dim = input_levels
            .first()
            .map(|level| level.shape().dims::<3>()[2])
            .unwrap_or(0)
            .max(1);
        let grids: Vec<PatchGrid> = mip_levels.iter().map(|level| level.grid).collect();
        let tokens = grids.first().map(|grid| grid.num_patches()).unwrap_or(0);
        let target_patches: Vec<Tensor<B, 3>> = mip_levels
            .iter()
            .map(|level| patchify(level.image.clone(), patch_size))
            .collect();
        let input_residuals = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => input_levels.clone(),
            VisionPyramidMode::Laplacian => {
                self.decompose_pyramid(&input_levels, &grids)
            }
        };
        let laplacian_images = if matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            self.build_laplacian_images(&mip_levels)
        } else {
            None
        };
        let base_grid = self.fovea_base_grid(patch_size, &device);
        let traj_len = self.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let num_eyes = self.config.num_eyes.max(1);
        let inner_steps = self.config.inner_steps.max(1);
        let traj_update_alpha = self.config.traj_update_alpha;

        let base_traj = self
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let mut trajs = vec![base_traj; num_eyes];
        let mut state_levels: Vec<Tensor<B, 3>> = input_residuals
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
            .collect();
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
        let mut tbptt_step_idx = 0usize;
        let mut tbptt_chunks = 0usize;
        let mut tbptt_loss_sum: Option<Tensor<B, 1>> = None;
        let mut tbptt_mask_sum: Option<Tensor<B, 1>> = None;
        let mut tbptt_inv_sum: Option<Tensor<B, 1>> = None;
        let mut tbptt_sigreg_sum: Option<Tensor<B, 1>> = None;
        let low_mem_pre_rollout = self.config.low_mem_pre_rollout;
        let capture_traj = capture_artifacts
            && self
                .config
                .artifact_max_views
                .saturating_sub(3)
            > 0;
        let mut traj_steps = if capture_traj {
            Some(Vec::with_capacity(rollout_steps))
        } else {
            None
        };
        let mut frame_steps = if capture_artifacts {
            Some(Vec::with_capacity(rollout_steps))
        } else {
            None
        };
        let mut last_patch_views: Option<Vec<Tensor<B, 4>>> = None;
        let gdpo = &self.config.policy.gdpo;
        let gdpo_enabled = gdpo.enabled && !capture_artifacts;
        let gdpo_group = gdpo.group_size.max(1);
        let gdpo_policy_enabled = gdpo_enabled && gdpo.policy_weight > 0.0;
        let info_reward_enabled = gdpo_enabled && self.config.policy.info_reward.enabled;
        let info_stride = self.config.policy.info_reward.stride.max(1);
        let detach_policy_from_recon = self.config.policy.detach_policy_from_recon;
        let mut log_prob_sum = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let mut log_prob_sum_old = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let mut hard_reward = if info_reward_enabled {
            Some(Tensor::<B, 1>::zeros([batch], &device))
        } else {
            None
        };
        let mut tbptt_policy_inputs = if gdpo_policy_enabled {
            Some(Vec::new())
        } else {
            None
        };
        let mut started_backprop = false;
        for step_idx in 0..rollout_steps {
            let pre_rollout = low_mem_pre_rollout && step_idx + 1 <= detach_until;
            let in_backprop = step_idx + 1 > detach_until;
            if tbptt_enabled && in_backprop && !started_backprop {
                started_backprop = true;
                if gdpo_enabled {
                    log_prob_sum = Some(Tensor::<B, 2>::zeros([batch, 1], &device));
                    log_prob_sum_old = Some(Tensor::<B, 2>::zeros([batch, 1], &device));
                }
                if info_reward_enabled {
                    hard_reward = Some(Tensor::<B, 1>::zeros([batch], &device));
                }
            }
            let anchor_traj =
                low_mem_pre_rollout && detach_until > 0 && step_idx + 1 == detach_until + 1;
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state_levels.clone(),
                VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
            };
            let mut updates: Vec<Tensor<B, 3>> = state_levels
                .iter()
                .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
                .collect();
            let collect_info = info_reward_enabled && (step_idx % info_stride == 0);
            let mut updates_null: Option<Vec<Tensor<B, 3>>> = if collect_info {
                Some(
                    state_levels
                        .iter()
                        .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
                        .collect(),
                )
            } else {
                None
            };
            let mut step_traj = if capture_traj || capture_artifacts {
                Some(Vec::with_capacity(num_eyes))
            } else {
                None
            };
            let mut step_patches = if capture_artifacts {
                Some(Vec::with_capacity(num_eyes))
            } else {
                None
            };
            let mut next_trajs = Vec::with_capacity(num_eyes);
            for eye_idx in 0..num_eyes {
                let mut traj = trajs[eye_idx].clone();
                if pre_rollout {
                    traj = traj.detach();
                } else if anchor_traj {
                    let anchor = self
                        .trajectory_token
                        .val()
                        .reshape([1, traj_len, embed_dim])
                        .repeat_dim(0, batch);
                    traj = traj + (anchor.clone() - anchor.detach());
                }
                let eye_embed = self
                    .eye_token
                    .val()
                    .slice_dim(0, eye_idx..eye_idx + 1)
                    .reshape([1, 1, embed_dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, traj_len);
                let eye_embed = Self::detach_if(eye_embed, pre_rollout);
                let traj_with_eye = traj.clone() + eye_embed.clone();
                let traj_with_eye = Self::detach_if(traj_with_eye, pre_rollout);
                let traj_summary = traj_with_eye
                    .clone()
                    .mean_dim(1)
                    .reshape([batch, 1, embed_dim]);
                let params = self.saccade_head.forward(traj_summary);
                let params = Self::detach_if(params, pre_rollout);
                let (mean_raw, sigma_raw) = self.decode_saccade_params(params);
                let mean_raw = Self::detach_if(mean_raw, pre_rollout);
                let sigma_raw = Self::detach_if(sigma_raw, pre_rollout);
                let (mean_action, sigma_action) = if gdpo_enabled {
                    let sample = self.sample_policy_action(mean_raw.clone(), sigma_raw.clone());
                    let log_prob_eye = sample.log_prob.sum_dim(1).reshape([batch, 1]);
                    if let Some(log_prob_sum) = log_prob_sum.as_mut() {
                        *log_prob_sum = log_prob_sum.clone() + log_prob_eye.clone();
                    }
                    if let Some(log_prob_sum_old) = log_prob_sum_old.as_mut() {
                        *log_prob_sum_old = log_prob_sum_old.clone() + log_prob_eye.detach();
                    }
                    (sample.mean, sample.sigma)
                } else {
                    (mean_raw.clone(), sigma_raw.clone())
                };
                let mean = if detach_policy_from_recon {
                    mean_action.clone().detach()
                } else {
                    mean_action.clone()
                };
                let sigma = if detach_policy_from_recon {
                    sigma_action.clone().detach()
                } else {
                    sigma_action.clone()
                };
                let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
                let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
                let mean_detached = mean_step.clone().detach();
                let sigma_detached = sigma_step.clone().detach();
                if let Some(steps) = &mut step_traj {
                    steps.push((mean_detached.clone(), sigma_detached.clone()));
                }
                let weights_context =
                    self.mip_gaussian_weights(&mip_levels, mean.clone(), sigma.clone());
                let weights_scatter = if matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian) {
                    self.mip_spatial_weights(&mip_levels, mean.clone(), sigma.clone())
                } else {
                    weights_context.clone()
                };
                let patch_image = {
                    self.foveated_patch_image(
                        &mip_levels,
                        &base_grid,
                        mean_step.clone(),
                        sigma_step.clone(),
                        laplacian_images.as_ref(),
                    )
                };
                let patch_tokens = {
                    self.model.patch_embed_raw(patch_image.clone()).tokens
                };
                let patch_tokens = Self::detach_if(patch_tokens, pre_rollout);
                let null_patch_tokens = if collect_info {
                    Some(self.null_patch_tokens(&patch_tokens))
                } else {
                    None
                };
                if let Some(step_patches) = step_patches.as_mut() {
                    step_patches.push(patch_image);
                }
                let input_context = patch_tokens;
                let state_context = {
                    let context = self.mip_weighted_sum(&state_composed, &weights_context);
                    self.project_pyramid_context(context)
                };
                let input_tokens =
                    self.build_input_tokens(input_context, state_context.clone(), mean.clone(), sigma.clone());
                let input_tokens = Self::detach_if(input_tokens, pre_rollout);
                let input_tokens = input_tokens.repeat_dim(1, traj_len);
                let tokens_in = traj_with_eye.clone() + input_tokens;
                let out_tokens = {
                    self.model
                        .forward_tokens_embed_steps(tokens_in, inner_steps)
                        .patch_tokens
                };
                let out_tokens = Self::detach_if(out_tokens, pre_rollout);
                let residual = {
                    self.residual_proj.forward(out_tokens.clone())
                };
                let residual = Self::detach_if(residual, pre_rollout);
                let residual_pool = residual
                    .clone()
                    .mean_dim(1)
                    .reshape([batch, 1, self.pyramid_dim]);
                let next_traj = if traj_update_alpha >= 1.0 {
                    out_tokens
                } else {
                    let keep = 1.0 - traj_update_alpha;
                    traj.clone().mul_scalar(keep)
                        + out_tokens.clone().mul_scalar(traj_update_alpha)
                };
                for (update, weights) in updates.iter_mut().zip(weights_scatter.iter()) {
                    let update_eye = {
                        self.weighted_sum_tokens(
                            weights.clone().swap_dims(1, 2),
                            residual_pool.clone(),
                        )
                    };
                    *update = update.clone() + update_eye;
                }
                if let Some(updates_null) = updates_null.as_mut() {
                    if let Some(null_patch_tokens) = null_patch_tokens {
                        let input_tokens_null = self.build_input_tokens(
                            null_patch_tokens,
                            state_context.clone(),
                            mean.clone(),
                            sigma.clone(),
                        );
                        let input_tokens_null = input_tokens_null.repeat_dim(1, traj_len);
                        let tokens_in_null = traj_with_eye.clone() + input_tokens_null;
                        let out_tokens_null = self
                            .model
                            .forward_tokens_embed_steps(tokens_in_null, inner_steps)
                            .patch_tokens;
                        let residual_null = self.residual_proj.forward(out_tokens_null);
                        let residual_pool_null = residual_null
                            .mean_dim(1)
                            .reshape([batch, 1, self.pyramid_dim]);
                        for (update, weights) in updates_null.iter_mut().zip(weights_scatter.iter()) {
                            let update_eye_null = self.weighted_sum_tokens(
                                weights.clone().swap_dims(1, 2),
                                residual_pool_null.clone(),
                            );
                            *update = update.clone() + update_eye_null;
                        }
                    }
                }
                next_trajs.push(next_traj);
            }
            let state_real: Vec<Tensor<B, 3>> = state_levels
                .iter()
                .zip(updates.iter())
                .map(|(state, update)| state.clone() + update.clone())
                .collect();
            let state_null = updates_null.as_ref().map(|updates_null| {
                state_levels
                    .iter()
                    .zip(updates_null.iter())
                    .map(|(state, update)| state.clone() + update.clone())
                    .collect::<Vec<_>>()
            });
            state_levels = state_real;
            if collect_info {
                if let (Some(hard_reward), Some(state_null)) =
                    (hard_reward.as_mut(), state_null)
                {
                    let (real_sum, real_mask, _) = self.recon_loss_per_sample_from_state(
                        &state_levels,
                        &grids,
                        &target_patches,
                        false,
                    );
                    let (null_sum, null_mask, _) = self.recon_loss_per_sample_from_state(
                        &state_null,
                        &grids,
                        &target_patches,
                        false,
                    );
                    let real = real_sum / real_mask.add_scalar(LEJEPA_EPS);
                    let null = null_sum / null_mask.add_scalar(LEJEPA_EPS);
                    *hard_reward = hard_reward.clone() + (null - real);
                }
            }
            if let Some(step_traj) = step_traj {
                if capture_traj {
                    if let Some(steps) = &mut traj_steps {
                        steps.push(step_traj.clone());
                    }
                }
                if let Some(frames) = &mut frame_steps {
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state_levels.clone(),
                        VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
                    };
                    let pred_patches = self
                        .recon
                        .forward(self.project_pyramid_level(state_composed[0].clone()));
                    let recon_view =
                        unpatchify(pred_patches, patch_size, height, width, channels);
                    let mut input_frame = images.clone();
                    let mut appended_patch = false;
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            input_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            input_frame = overlay;
                        }
                    }
                    let mut frame_views = Vec::new();
                    let push_view = |views: &mut Vec<Tensor<B, 4>>, view: Tensor<B, 4>| {
                        if !views.is_empty() && SACCADE_VIEW_GAP > 0 {
                            views.push(view_separator_like(&view, SACCADE_VIEW_GAP));
                        }
                        views.push(view);
                    };
                    push_view(&mut frame_views, input_frame);
                    if let Some(step_patches) = step_patches {
                        if let Some(patch_views) =
                            saccade_patch_views(step_patches, height)
                        {
                            last_patch_views =
                                Some(patch_views.iter().map(|view| view.clone().detach()).collect());
                            for patch_view in patch_views {
                                push_view(&mut frame_views, patch_view);
                            }
                            appended_patch = true;
                        }
                    }
                    if !appended_patch {
                        if let Some(patch_views) = last_patch_views.clone() {
                            for patch_view in patch_views {
                                push_view(&mut frame_views, patch_view);
                            }
                        }
                    }
                    push_view(&mut frame_views, recon_view);
                    let frame = Tensor::cat(frame_views, 3);
                    frames.push(frame);
                }
            }
            trajs = next_trajs;
            if step_idx + 1 <= detach_until {
                trajs = trajs.into_iter().map(|traj| traj.detach()).collect();
                for level in &mut state_levels {
                    *level = level.clone().detach();
                }
            }
            if tbptt_enabled && in_backprop {
                tbptt_step_idx += 1;
                let chunk_done = tbptt_step_idx >= tbptt_step_count || step_idx + 1 == rollout_steps;
                if chunk_done {
                    tbptt_step_idx = 0;
                    tbptt_chunks += 1;
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state_levels.clone(),
                        VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
                    };
                    let state_composed_embed = self.project_pyramid_levels(&state_composed);
                    let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                        self.pyramid_lejepa_loss(&state_composed_embed)
                    } else {
                        let zero = Tensor::<B, 1>::zeros([1], &device);
                        (zero.clone(), zero)
                    };
                    let (loss_per_sample, mask_per_sample, _) = self
                        .recon_loss_per_sample_from_projected_levels(
                            &state_composed_embed,
                            &target_patches,
                            false,
                        );
                    let loss_sum = loss_per_sample.clone().sum();
                    let mask_sum = mask_per_sample.clone().sum();
                    tbptt_loss_sum = Some(match tbptt_loss_sum {
                        Some(accum) => accum + loss_sum.clone(),
                        None => loss_sum,
                    });
                    tbptt_mask_sum = Some(match tbptt_mask_sum {
                        Some(accum) => accum + mask_sum.clone(),
                        None => mask_sum,
                    });
                    tbptt_inv_sum = Some(match tbptt_inv_sum {
                        Some(accum) => accum + inv.clone(),
                        None => inv,
                    });
                    tbptt_sigreg_sum = Some(match tbptt_sigreg_sum {
                        Some(accum) => accum + sigreg.clone(),
                        None => sigreg,
                    });
                    if let Some(gdpo_inputs) = tbptt_policy_inputs.as_mut() {
                        let recon_per_sample =
                            loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
                        let hard_reward = hard_reward
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 1>::zeros([batch], &device));
                        let log_prob_sum = log_prob_sum
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch, 1], &device));
                        let log_prob_sum_old = log_prob_sum_old
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch, 1], &device));
                        gdpo_inputs.push(GdpoPolicyInputs {
                            hard_reward,
                            recon_per_sample,
                            log_prob_sum,
                            log_prob_sum_old,
                            gdpo_group,
                        });
                    }
                    if step_idx + 1 < rollout_steps {
                        if gdpo_enabled {
                            log_prob_sum = Some(Tensor::<B, 2>::zeros([batch, 1], &device));
                            log_prob_sum_old = Some(Tensor::<B, 2>::zeros([batch, 1], &device));
                        }
                        if info_reward_enabled {
                            hard_reward = Some(Tensor::<B, 1>::zeros([batch], &device));
                        }
                        trajs = trajs.into_iter().map(|traj| traj.detach()).collect();
                        for level in &mut state_levels {
                            *level = level.clone().detach();
                        }
                    }
                }
            }
        }

        let (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair) = if tbptt_enabled {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            let chunk_count = tbptt_chunks.max(1) as f32;
            let inv_sum = tbptt_inv_sum.unwrap_or_else(|| zero.clone());
            let sigreg_sum = tbptt_sigreg_sum.unwrap_or_else(|| zero.clone());
            let inv = inv_sum.mul_scalar(1.0 / chunk_count);
            let sigreg = sigreg_sum.mul_scalar(1.0 / chunk_count);
            let loss_sum = tbptt_loss_sum.unwrap_or_else(|| zero.clone());
            let mask_sum = tbptt_mask_sum.unwrap_or_else(|| zero.clone());
            let base_pair = if capture_artifacts {
                let (_, _, base_pair) = self.recon_loss_per_sample_from_state(
                    &state_levels,
                    &grids,
                    &target_patches,
                    true,
                );
                base_pair
            } else {
                None
            };
            (
                loss_sum,
                mask_sum,
                inv,
                sigreg,
                tbptt_policy_inputs,
                base_pair,
            )
        } else {
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state_levels.clone(),
                VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
            };
            let state_composed_embed = self.project_pyramid_levels(&state_composed);
            let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                self.pyramid_lejepa_loss(&state_composed_embed)
            } else {
                let zero = Tensor::<B, 1>::zeros([1], &device);
                (zero.clone(), zero)
            };

            let (loss_per_sample, mask_per_sample, base_pair) =
                self.recon_loss_per_sample_from_projected_levels(
                    &state_composed_embed,
                    &target_patches,
                    capture_artifacts,
                );
            let loss_sum = loss_per_sample.clone().sum();
            let mask_sum = mask_per_sample.clone().sum();
            let recon_per_sample = loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
            let gdpo_inputs = if gdpo_policy_enabled {
                let hard_reward = hard_reward
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 1>::zeros([batch], &device));
                let log_prob_sum = log_prob_sum
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch, 1], &device));
                let log_prob_sum_old = log_prob_sum_old
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([batch, 1], &device));
                Some(vec![GdpoPolicyInputs {
                    hard_reward,
                    recon_per_sample,
                    log_prob_sum,
                    log_prob_sum_old,
                    gdpo_group,
                }])
            } else {
                None
            };
            (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair)
        };
        let (pred_base, target_base) = if let Some((pred, target)) = base_pair {
            (Some(pred), Some(target))
        } else {
            (None, None)
        };

        let artifacts = if capture_artifacts && batch > 0 && tokens > 0 {
            let pred_first = pred_base.clone().unwrap_or_else(|| {
                Tensor::<B, 3>::zeros([batch, tokens, patch_size * patch_size * channels], &device)
            });
            let target_first = target_base.clone().unwrap_or_else(|| {
                Tensor::<B, 3>::zeros([batch, tokens, patch_size * patch_size * channels], &device)
            });
            let recon_view = unpatchify(pred_first.clone(), patch_size, height, width, channels);
            let residual = pred_first - target_first;
            let mut target_width = width;
            if let Some(patch_views) = &last_patch_views {
                for patch_view in patch_views {
                    target_width = target_width.max(patch_view.shape().dims::<4>()[3]);
                }
            }
            let mut images_view = images.clone();
            if let Some(steps) = traj_steps.as_ref() {
                if let Some(last_step) = steps.last() {
                    for (eye_idx, (mean, sigma)) in last_step.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            images_view.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            images_view = overlay;
                        }
                    }
                }
            }
            let images_view = pad_view_width(images_view, target_width);
            let recon_view = pad_view_width(recon_view, target_width);
            let patch_views = last_patch_views.map(|patch_views| {
                patch_views
                    .into_iter()
                    .map(|patch_view| pad_view_width_centered(patch_view, target_width))
                    .collect::<Vec<_>>()
            });
            let mut views = Vec::new();
            let mut legend = Vec::new();
            views.push(images_view);
            legend.push("input_with_fovea".to_string());
            if let Some(patch_views) = patch_views {
                for (eye_idx, patch_view) in patch_views.into_iter().enumerate() {
                    views.push(patch_view);
                    legend.push(format!("foveated_patch_eye_{eye_idx}"));
                }
            }
            views.push(recon_view);
            legend.push("reconstruction".to_string());
            if let Some(steps) = traj_steps {
                let max_extra = self
                    .config
                    .artifact_max_views
                    .saturating_sub(views.len());
                let mut remaining = max_extra;
                for idx in select_trajectory_indices(steps.len(), max_extra) {
                    for (eye_idx, (mean, sigma)) in steps[idx].iter().enumerate() {
                        if remaining == 0 {
                            break;
                        }
                        if let Some(view) = saccade_circle_overlay(
                            images.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            views.push(pad_view_width(view, target_width));
                            legend.push(format!(
                                "trajectory_overlay_step_{idx}_eye_{eye_idx}"
                            ));
                            remaining = remaining.saturating_sub(1);
                        }
                    }
                    if remaining == 0 {
                        break;
                    }
                }
            }
            let frames = frame_steps.and_then(|frames| {
                if frames.is_empty() {
                    return None;
                }
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
            });
            Some((views, residual, frames, legend))
        } else {
            None
        };

        (loss_sum, mask_sum, inv, sigreg, artifacts, gdpo_inputs)
    }

}

