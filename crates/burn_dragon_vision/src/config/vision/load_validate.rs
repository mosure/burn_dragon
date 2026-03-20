use super::*;
use crate::loss::VisionDistillationLossConfig;
use burn_dragon_core::BDHConfig;

pub fn load_vision_training_config(paths: &[PathBuf]) -> Result<VisionTrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<VisionTrainingConfig>()
        .map_err(|err| anyhow!(err))
}

pub(super) fn validate_vision_rollout(
    training: &VisionTrainingHyperparameters,
    max_steps: usize,
) -> Result<()> {
    let max_steps = max_steps.max(1);
    let min_steps = training.rollout_min_steps.unwrap_or(max_steps);
    let max_steps_cfg = training.rollout_max_steps.unwrap_or(max_steps);
    let backprop_steps = training.rollout_backprop_steps.unwrap_or(max_steps_cfg);
    if min_steps == 0 || max_steps_cfg == 0 {
        return Err(anyhow!(
            "vision rollout steps must be > 0 (min={min_steps}, max={max_steps_cfg})"
        ));
    }
    if min_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_min_steps ({min_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    if max_steps_cfg > max_steps {
        return Err(anyhow!(
            "vision rollout_max_steps ({max_steps_cfg}) exceeds vision.steps ({max_steps})"
        ));
    }
    if backprop_steps > 0 && backprop_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_backprop_steps ({backprop_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    Ok(())
}

pub(super) fn validate_vision_mhc(vision: &VisionModelConfig) -> Result<()> {
    if !vision.mhc.enabled {
        return Ok(());
    }
    let num_eyes = vision.num_eyes.max(1);
    if vision.mhc.num_streams != 0 && vision.mhc.num_streams != num_eyes {
        return Err(anyhow!(
            "vision.mhc.num_streams ({}) must match vision.num_eyes ({})",
            vision.mhc.num_streams,
            num_eyes
        ));
    }
    if vision.mhc.num_views != 0 && vision.mhc.num_views != num_eyes {
        return Err(anyhow!(
            "vision.mhc.num_views ({}) must match vision.num_eyes ({})",
            vision.mhc.num_views,
            num_eyes
        ));
    }
    if vision.mhc.mhc_iters == 0 {
        return Err(anyhow!("vision.mhc.mhc_iters must be > 0"));
    }
    if vision.mhc.mhc_tau <= 0.0 {
        return Err(anyhow!(
            "vision.mhc.mhc_tau must be > 0 (got {})",
            vision.mhc.mhc_tau
        ));
    }
    if vision.mhc.dropout < 0.0 {
        return Err(anyhow!("vision.mhc.dropout must be >= 0"));
    }
    Ok(())
}

pub(super) fn validate_vision_trm_graph(vision: &VisionModelConfig) -> Result<()> {
    if !matches!(
        vision.resolved_backbone_kind()?,
        VisionBackboneKind::Pyramid
    ) {
        return Ok(());
    }
    if vision.trm_graph.rank == 0 {
        return Err(anyhow!("vision.trm_graph.rank must be > 0"));
    }
    if vision.trm_graph.patch_rank == Some(0) {
        return Err(anyhow!("vision.trm_graph.patch_rank must be > 0 when set"));
    }
    if vision.trm_graph.coarse_rank == Some(0) {
        return Err(anyhow!("vision.trm_graph.coarse_rank must be > 0 when set"));
    }
    if vision.trm_graph.global_rank == Some(0) {
        return Err(anyhow!("vision.trm_graph.global_rank must be > 0 when set"));
    }
    if vision.trm_graph.value_dim == 0 {
        return Err(anyhow!("vision.trm_graph.value_dim must be > 0"));
    }
    if vision.trm_graph.local_radius == 0 && !vision.trm_graph.local_self {
        return Err(anyhow!(
            "vision.trm_graph.local_radius must be > 0 unless vision.trm_graph.local_self is true"
        ));
    }
    if vision.trm_graph.hub_count == 0 {
        return Err(anyhow!("vision.trm_graph.hub_count must be > 0"));
    }
    if vision.trm_graph.coarse_stride == 0 {
        return Err(anyhow!("vision.trm_graph.coarse_stride must be > 0"));
    }
    if !(0.0..=1.0).contains(&vision.trm_graph.decay) {
        return Err(anyhow!(
            "vision.trm_graph.decay must be in [0, 1] (got {})",
            vision.trm_graph.decay
        ));
    }
    let patch_size = vision.patch_size.max(1);
    let grid = vision.image_size.div_ceil(patch_size);
    let grid_h = vision.pos_max_height.unwrap_or(grid);
    let grid_w = vision.pos_max_width.unwrap_or(grid);
    if grid_h % vision.trm_graph.coarse_stride != 0 {
        return Err(anyhow!(
            "vision.trm_graph.coarse_stride ({}) must divide grid height ({})",
            vision.trm_graph.coarse_stride,
            grid_h
        ));
    }
    if grid_w % vision.trm_graph.coarse_stride != 0 {
        return Err(anyhow!(
            "vision.trm_graph.coarse_stride ({}) must divide grid width ({})",
            vision.trm_graph.coarse_stride,
            grid_w
        ));
    }
    Ok(())
}

pub(super) fn validate_vision_rho_stream(vision: &VisionModelConfig) -> Result<()> {
    if !matches!(
        vision.resolved_backbone_kind()?,
        VisionBackboneKind::Cellular
    ) {
        return Ok(());
    }
    if vision.rho_stream.local_radius == 0 {
        return Err(anyhow!("vision.rho_stream.local_radius must be > 0"));
    }
    if !(0.0..=1.0).contains(&vision.rho_stream.decay) {
        return Err(anyhow!(
            "vision.rho_stream.decay must be in [0, 1] (got {})",
            vision.rho_stream.decay
        ));
    }
    if vision.rho_stream.wgpu_rollout_fused && !vision.rho_stream.wgpu_forward_kernel {
        return Err(anyhow!(
            "vision.rho_stream.wgpu_rollout_fused requires vision.rho_stream.wgpu_forward_kernel = true"
        ));
    }
    if vision.num_eyes.max(1) > 1 {
        return Err(anyhow!(
            "vision.backbone = \"cellular\" currently supports single-stream rollout only; set vision.num_eyes = 1"
        ));
    }
    Ok(())
}

pub(super) fn validate_vision_mode(
    mode: &VisionTrainingModeConfig,
    vision: &VisionModelConfig,
    dataset: &VisionDatasetConfig,
    augment: &VisionAugmentationConfig,
) -> Result<()> {
    match mode {
        VisionTrainingModeConfig::Distill(distill) => {
            if dataset.source != VisionDatasetSource::Imagenet {
                return Err(anyhow!(
                    "distill mode currently requires dataset.source = \"imagenet\""
                ));
            }
            validate_distill_loss(&distill.loss)?;
            validate_distill_supervision(distill)?;
            validate_distill_primary_teacher(&distill.teacher, vision)?;
            validate_distill_auxiliary_teacher_targets(distill, vision)?;

            let has_feature_targets = distill
                .resolved_teacher_targets()
                .into_iter()
                .any(|target| matches!(target.teacher, VisionTeacherConfig::Features(_)));
            if has_feature_targets && !distill_train_features_are_deterministic(augment) {
                return Err(anyhow!(
                    "distill mode with feature-file teachers requires deterministic train augmentations so precomputed teacher features match the student view"
                ));
            }
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            if dataset.source != VisionDatasetSource::Imagenet {
                return Err(anyhow!(
                    "image LEJEPA currently requires dataset.source = \"imagenet\""
                ));
            }
            validate_momentum_teacher("mode.teacher_ema", &lejepa.teacher_ema)?;
            if lejepa.views == 0 {
                return Err(anyhow!("mode.views must be > 0"));
            }
            if lejepa.local_image_size == 0 {
                return Err(anyhow!("mode.local_image_size must be > 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.local_min_scale) {
                return Err(anyhow!(
                    "mode.local_min_scale must be in [0, 1] (got {})",
                    lejepa.local_min_scale
                ));
            }
            if !(0.0..=1.0).contains(&lejepa.local_max_scale) {
                return Err(anyhow!(
                    "mode.local_max_scale must be in [0, 1] (got {})",
                    lejepa.local_max_scale
                ));
            }
            if lejepa.local_min_scale > lejepa.local_max_scale {
                return Err(anyhow!(
                    "mode.local_min_scale ({}) must be <= mode.local_max_scale ({})",
                    lejepa.local_min_scale,
                    lejepa.local_max_scale
                ));
            }
            if !(0.0..=1.0).contains(&lejepa.min_view_overlap) {
                return Err(anyhow!(
                    "mode.min_view_overlap must be in [0, 1] (got {})",
                    lejepa.min_view_overlap
                ));
            }
            if lejepa.view_overlap_attempts == 0 {
                return Err(anyhow!("mode.view_overlap_attempts must be > 0"));
            }
            if matches!(
                vision.resolved_backbone_kind()?,
                VisionBackboneKind::Pyramid
            ) && vision.trm_graph.grid_mismatch_policy == VisionTrmGridMismatchPolicy::Error
                && lejepa.local_views > 0
            {
                let patch = vision.patch_size.max(1);
                let global_grid = vision.image_size.div_ceil(patch);
                let grid_h = vision.pos_max_height.unwrap_or(global_grid);
                let grid_w = vision.pos_max_width.unwrap_or(global_grid);
                let local_grid = lejepa.local_image_size.div_ceil(patch);
                if local_grid != grid_h || local_grid != grid_w {
                    return Err(anyhow!(
                        "pyramid backbone strict mode requires local view grid ({local_grid}x{local_grid}) to match vision grid ({grid_h}x{grid_w}); adjust mode.local_image_size / vision.patch_size / vision.pos_max_*, set mode.local_views=0, or set vision.trm_graph.grid_mismatch_policy = \"fallback_default\""
                    ));
                }
            }
            validate_lejepa_loss(&lejepa.loss.lejepa)?;
            validate_recon_loss("mode.loss.recon", &lejepa.loss.recon)?;
        }
        VisionTrainingModeConfig::VideoLejepa(video) => {
            if dataset.source != VisionDatasetSource::MovingMnist {
                return Err(anyhow!(
                    "video LEJEPA requires dataset.source = \"moving_mnist\""
                ));
            }
            validate_momentum_teacher("mode.teacher_ema", &video.teacher_ema)?;
            if !vision.use_cls_token {
                return Err(anyhow!("video LEJEPA requires vision.use_cls_token = true"));
            }
            if vision.in_channels != 3 {
                return Err(anyhow!(
                    "video LEJEPA currently requires vision.in_channels = 3"
                ));
            }
            if video.context_frames == 0 {
                return Err(anyhow!("mode.context_frames must be > 0"));
            }
            if video.target_frames == 0 {
                return Err(anyhow!("mode.target_frames must be > 0"));
            }
            if video.train_target_frames_max > 0
                && video.train_target_frames_min > 0
                && video.train_target_frames_max < video.train_target_frames_min
            {
                return Err(anyhow!(
                    "mode.train_target_frames_max ({}) must be >= mode.train_target_frames_min ({})",
                    video.train_target_frames_max,
                    video.train_target_frames_min
                ));
            }
            if video.frame_stride == 0 {
                return Err(anyhow!("mode.frame_stride must be > 0"));
            }
            if video.temporal.n_layer == 0 {
                return Err(anyhow!("mode.temporal.n_layer must be > 0"));
            }
            if video.temporal.n_head == 0 {
                return Err(anyhow!("mode.temporal.n_head must be > 0"));
            }
            if video.temporal.mlp_internal_dim_multiplier == 0 {
                return Err(anyhow!(
                    "mode.temporal.mlp_internal_dim_multiplier must be > 0"
                ));
            }
            if !BDHConfig::is_valid_rollout_fast_steps(
                video.temporal.rollout_fast_steps_per_slow_step,
            ) {
                return Err(anyhow!(
                    "mode.temporal.rollout_fast_steps_per_slow_step must be one of {:?} (got {})",
                    BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS,
                    video.temporal.rollout_fast_steps_per_slow_step
                ));
            }
            if vision.embed_dim % video.temporal.n_head != 0 {
                return Err(anyhow!(
                    "vision.embed_dim ({}) must be divisible by mode.temporal.n_head ({})",
                    vision.embed_dim,
                    video.temporal.n_head
                ));
            }
            if video.temporal.latent_block_size == 0 {
                return Err(anyhow!("mode.temporal.latent_block_size must be > 0"));
            }
            if video.temporal.time_block_size == 0 {
                return Err(anyhow!("mode.temporal.time_block_size must be > 0"));
            }
            if video.temporal.wgpu_rollout_fused && !video.temporal.wgpu_recurrent_kernel {
                return Err(anyhow!(
                    "mode.temporal.wgpu_rollout_fused requires mode.temporal.wgpu_recurrent_kernel = true"
                ));
            }
            if video.loss.prediction_weight < 0.0 {
                return Err(anyhow!("mode.loss.prediction_weight must be >= 0"));
            }
            if video.loss.observe_weight < 0.0 {
                return Err(anyhow!("mode.loss.observe_weight must be >= 0"));
            }
            if video.loss.cosine_weight < 0.0 {
                return Err(anyhow!("mode.loss.cosine_weight must be >= 0"));
            }
            if video.loss.probe_weight < 0.0 {
                return Err(anyhow!("mode.loss.probe_weight must be >= 0"));
            }
            if video.loss.debug_recon_weight < 0.0 {
                return Err(anyhow!("mode.loss.debug_recon_weight must be >= 0"));
            }
            validate_lejepa_loss(&video.loss.sigreg)?;
            if video.artifact_upscale == 0 {
                return Err(anyhow!("mode.artifact_upscale must be > 0"));
            }
            if video.artifact_future_frames > 0
                && video.artifact_future_frames < video.target_frames
            {
                return Err(anyhow!(
                    "mode.artifact_future_frames ({}) must be >= mode.target_frames ({}) when set",
                    video.artifact_future_frames,
                    video.target_frames
                ));
            }
            let moving = &dataset.moving_mnist;
            if moving.digit_size == 0 {
                return Err(anyhow!("dataset.moving_mnist.digit_size must be > 0"));
            }
            if moving.digit_size > vision.image_size {
                return Err(anyhow!(
                    "dataset.moving_mnist.digit_size ({}) must be <= vision.image_size ({})",
                    moving.digit_size,
                    vision.image_size
                ));
            }
            if moving.min_velocity <= 0.0 {
                return Err(anyhow!("dataset.moving_mnist.min_velocity must be > 0"));
            }
            if moving.max_velocity < moving.min_velocity {
                return Err(anyhow!(
                    "dataset.moving_mnist.max_velocity ({}) must be >= min_velocity ({})",
                    moving.max_velocity,
                    moving.min_velocity
                ));
            }
        }
        VisionTrainingModeConfig::Mae(mae) => {
            if dataset.source != VisionDatasetSource::Imagenet {
                return Err(anyhow!(
                    "mae mode currently requires dataset.source = \"imagenet\""
                ));
            }
            validate_recon_loss("mode.loss.recon", &mae.loss.recon)?;
            if mae.pyramid_levels == 0 {
                return Err(anyhow!("mode.pyramid_levels must be > 0"));
            }
            if mae.cross_view.enabled {
                let num_eyes = vision.num_eyes.max(1);
                if num_eyes < 2 {
                    return Err(anyhow!(
                        "vision.num_eyes must be >= 2 when cross_view is enabled"
                    ));
                }
                if !(0.0..=1.0).contains(&mae.cross_view.min_overlap) {
                    return Err(anyhow!(
                        "mode.cross_view.min_overlap must be in [0, 1] (got {})",
                        mae.cross_view.min_overlap
                    ));
                }
                if mae.cross_view.max_attempts == 0 {
                    return Err(anyhow!("mode.cross_view.max_attempts must be > 0"));
                }
                if !(0.0..=1.0).contains(&mae.cross_view.fuse_alpha) {
                    return Err(anyhow!(
                        "mode.cross_view.fuse_alpha must be in [0, 1] (got {})",
                        mae.cross_view.fuse_alpha
                    ));
                }
                if mae.cross_view.visible_weight < 0.0 {
                    return Err(anyhow!(
                        "mode.cross_view.visible_weight must be >= 0 (got {})",
                        mae.cross_view.visible_weight
                    ));
                }
                if mae.cross_view.masked_eye >= num_eyes {
                    return Err(anyhow!(
                        "mode.cross_view.masked_eye ({}) must be < vision.num_eyes ({})",
                        mae.cross_view.masked_eye,
                        num_eyes
                    ));
                }
                if !vision.mhc.enabled {
                    return Err(anyhow!(
                        "vision.mhc.enabled must be true when mode.cross_view.enabled is true"
                    ));
                }
                if vision.mhc.num_streams != 0 && vision.mhc.num_streams != num_eyes {
                    return Err(anyhow!(
                        "vision.mhc.num_streams ({}) must match vision.num_eyes ({})",
                        vision.mhc.num_streams,
                        num_eyes
                    ));
                }
                if vision.mhc.num_views != 0 && vision.mhc.num_views != num_eyes {
                    return Err(anyhow!(
                        "vision.mhc.num_views ({}) must match vision.num_eyes ({})",
                        vision.mhc.num_views,
                        num_eyes
                    ));
                }
            }
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            if dataset.source != VisionDatasetSource::Imagenet {
                return Err(anyhow!(
                    "saccade mode currently requires dataset.source = \"imagenet\""
                ));
            }
            let num_eyes = if saccade.num_eyes == 0 {
                vision.num_eyes
            } else {
                saccade.num_eyes
            };
            if num_eyes == 0 {
                return Err(anyhow!("vision.num_eyes must be > 0"));
            }
            if saccade.num_eyes != 0 && saccade.num_eyes != vision.num_eyes {
                return Err(anyhow!(
                    "saccade.num_eyes ({}) must match vision.num_eyes ({})",
                    saccade.num_eyes,
                    vision.num_eyes
                ));
            }
            if saccade.traj_tokens == 0 {
                return Err(anyhow!("saccade.traj_tokens must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.traj_update_alpha) {
                return Err(anyhow!(
                    "saccade.traj_update_alpha must be in [0, 1] (got {})",
                    saccade.traj_update_alpha
                ));
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if saccade.fovea_subsamples == 0 {
                return Err(anyhow!("saccade.fovea_subsamples must be > 0"));
            }
            if saccade.fovea_radius_scale <= 0.0 {
                return Err(anyhow!(
                    "saccade.fovea_radius_scale must be > 0 (got {})",
                    saccade.fovea_radius_scale
                ));
            }
            if saccade.grid_sample_max_mb == 0 {
                return Err(anyhow!("saccade.grid_sample_max_mb must be > 0"));
            }
            if saccade.mip_concat_max_mb == 0 {
                return Err(anyhow!("saccade.mip_concat_max_mb must be > 0"));
            }
            if saccade.recon_max_elems == 0 {
                return Err(anyhow!("saccade.recon_max_elems must be > 0"));
            }
            validate_input_projection(&saccade.input_projection)?;
            if saccade.fovea_subpatch_size > 0 && saccade.fovea_subpatch_size > vision.patch_size {
                return Err(anyhow!(
                    "saccade.fovea_subpatch_size ({}) must be <= vision.patch_size ({})",
                    saccade.fovea_subpatch_size,
                    vision.patch_size
                ));
            }
            if matches!(saccade.pyramid_feature_dim, Some(0)) {
                return Err(anyhow!("saccade.pyramid_feature_dim must be > 0 when set"));
            }
            if saccade.cache.max_entries == 0 {
                return Err(anyhow!("saccade.cache.max_entries must be > 0"));
            }
            if saccade.policy.info_reward.stride == 0 {
                return Err(anyhow!("saccade.policy.info_reward.stride must be > 0"));
            }
            if saccade.policy.location_embedding.quantize_bins < 2 {
                return Err(anyhow!(
                    "saccade.policy.location_embedding.quantize_bins must be >= 2"
                ));
            }
            validate_recon_loss("saccade.loss.recon", &saccade.loss.recon)?;
            validate_lejepa_loss(&saccade.loss.lejepa)?;
            if saccade.cross_view.enabled {
                if num_eyes < 2 {
                    return Err(anyhow!(
                        "vision.num_eyes must be >= 2 when mode.cross_view.enabled is true"
                    ));
                }
                if !(0.0..=1.0).contains(&saccade.cross_view.min_overlap) {
                    return Err(anyhow!(
                        "mode.cross_view.min_overlap must be in [0, 1] (got {})",
                        saccade.cross_view.min_overlap
                    ));
                }
                if saccade.cross_view.max_attempts == 0 {
                    return Err(anyhow!(
                        "mode.cross_view.max_attempts must be > 0 when mode.cross_view.enabled is true"
                    ));
                }
                if saccade.cross_view.masked_eye >= num_eyes {
                    return Err(anyhow!(
                        "mode.cross_view.masked_eye ({}) must be < vision.num_eyes ({})",
                        saccade.cross_view.masked_eye,
                        num_eyes
                    ));
                }
            }

            if saccade.policy.gdpo.enabled {
                if saccade.policy.gdpo.group_size == 0 {
                    return Err(anyhow!("saccade.policy.gdpo.group_size must be > 0"));
                }
                if saccade.policy.action_noise_std <= 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.action_noise_std must be > 0 when gdpo is enabled"
                    ));
                }
                if saccade.policy.gdpo.hard_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.hard_weight must be >= 0"));
                }
                if saccade.policy.gdpo.easy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.easy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.policy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_clip_range < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.policy_clip_range must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.advantage_clip < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_clip must be >= 0 (got {})",
                        saccade.policy.gdpo.advantage_clip
                    ));
                }
                if !(0.0..1.0).contains(&saccade.policy.gdpo.advantage_ema_decay) {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_ema_decay must be in [0, 1) (got {})",
                        saccade.policy.gdpo.advantage_ema_decay
                    ));
                }
                if let GdpoHardGate::Percentile { quantile } = saccade.policy.gdpo.hard_gate
                    && !(0.0..=1.0).contains(&quantile)
                {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                        quantile
                    ));
                }
            }
        }
    }
    Ok(())
}

fn validate_distill_supervision(distill: &VisionDistillConfig) -> Result<()> {
    if distill.rollout_supervision_frames == 0 {
        return Err(anyhow!(
            "mode.rollout_supervision_frames must be > 0 for distill mode"
        ));
    }
    if distill.rollout_supervision_groups == 0 {
        return Err(anyhow!(
            "mode.rollout_supervision_groups must be > 0 for distill mode"
        ));
    }
    if distill.rollout_supervision_explicit_steps.contains(&0) {
        return Err(anyhow!(
            "mode.rollout_supervision_explicit_steps entries must be > 0 for distill mode"
        ));
    }
    if distill
        .rollout_supervision_explicit_groups
        .iter()
        .flatten()
        .any(|step| *step == 0)
    {
        return Err(anyhow!(
            "mode.rollout_supervision_explicit_groups entries must be > 0 for distill mode"
        ));
    }
    if distill.rollout_supervision_stride == 0 {
        return Err(anyhow!(
            "mode.rollout_supervision_stride must be > 0 for distill mode"
        ));
    }
    if distill.rollout_supervision_power < 0.0 {
        return Err(anyhow!(
            "mode.rollout_supervision_power must be >= 0 for distill mode"
        ));
    }
    if distill.rollout_sampling_power < 0.0 {
        return Err(anyhow!(
            "mode.rollout_sampling_power must be >= 0 for distill mode"
        ));
    }
    if distill.rollout_improvement_weight < 0.0 {
        return Err(anyhow!(
            "mode.rollout_improvement_weight must be >= 0 for distill mode"
        ));
    }
    if distill.rollout_improvement_margin < 0.0 {
        return Err(anyhow!(
            "mode.rollout_improvement_margin must be >= 0 for distill mode"
        ));
    }
    Ok(())
}

fn validate_distill_primary_teacher(
    teacher: &VisionTeacherConfig,
    vision: &VisionModelConfig,
) -> Result<()> {
    match teacher {
        VisionTeacherConfig::Features(config) => {
            if config.feature_dim == 0 {
                return Err(anyhow!("mode.teacher.feature_dim must be > 0"));
            }
            if !config.has_patch_targets() {
                return Err(anyhow!(
                    "mode.teacher feature teachers currently require train_patch_path and val_patch_path"
                ));
            }
            if matches!(config.patch_tokens, Some(0)) {
                return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
            }
            if config.feature_dim != vision.projection_dim {
                return Err(anyhow!(
                    "mode.teacher.feature_dim ({}) must match vision.projection_dim ({})",
                    config.feature_dim,
                    vision.projection_dim
                ));
            }
        }
        VisionTeacherConfig::Model(config) => {
            if matches!(config.image_size, Some(0)) {
                return Err(anyhow!("mode.teacher.image_size must be > 0 when set"));
            }
            if matches!(config.patch_size, Some(0)) {
                return Err(anyhow!("mode.teacher.patch_size must be > 0 when set"));
            }
            if matches!(config.feature_dim, Some(0)) {
                return Err(anyhow!("mode.teacher.feature_dim must be > 0 when set"));
            }
            if matches!(config.patch_tokens, Some(0)) {
                return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
            }
        }
    }
    Ok(())
}

fn validate_distill_auxiliary_teacher_targets(
    distill: &VisionDistillConfig,
    vision: &VisionModelConfig,
) -> Result<()> {
    let student_patch_tokens = vision.image_size.div_ceil(vision.patch_size.max(1)).pow(2);
    let mut names = std::collections::BTreeSet::new();
    for (index, target) in distill.teacher_targets.iter().enumerate() {
        let prefix = format!("mode.teacher_targets[{index}]");
        let name = target.name.trim();
        if name.is_empty() {
            return Err(anyhow!("{prefix}.name must not be empty"));
        }
        if name == VisionDistillConfig::PRIMARY_TEACHER_NAME {
            return Err(anyhow!(
                "{prefix}.name \"{}\" is reserved for the legacy primary teacher",
                VisionDistillConfig::PRIMARY_TEACHER_NAME
            ));
        }
        if !names.insert(name.to_string()) {
            return Err(anyhow!("{prefix}.name must be unique"));
        }
        if target.weight < 0.0 {
            return Err(anyhow!("{prefix}.weight must be >= 0"));
        }
        if matches!(target.decoder_hidden_dim, Some(0)) {
            return Err(anyhow!("{prefix}.decoder_hidden_dim must be > 0 when set"));
        }
        match &target.teacher {
            VisionTeacherConfig::Features(config) => {
                if config.feature_dim == 0 {
                    return Err(anyhow!("{prefix}.teacher.feature_dim must be > 0"));
                }
                if matches!(config.patch_tokens, Some(0)) {
                    return Err(anyhow!(
                        "{prefix}.teacher.patch_tokens must be > 0 when set"
                    ));
                }
                if matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls)
                    && !config.has_patch_targets()
                {
                    return Err(anyhow!(
                        "{prefix}.teacher requires train_patch_path and val_patch_path for target_kind = \"patch_and_cls\""
                    ));
                }
                match target.decoder_mode {
                    VisionTeacherDecoderMode::SharedProjection => {
                        if config.feature_dim != vision.projection_dim {
                            return Err(anyhow!(
                                "{prefix}.teacher.feature_dim ({}) must match vision.projection_dim ({}) for decoder_mode = \"shared_projection\"",
                                config.feature_dim,
                                vision.projection_dim
                            ));
                        }
                        if matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls) {
                            if let Some(tokens) = config
                                .patch_tokens
                                .filter(|tokens| *tokens != student_patch_tokens)
                            {
                                return Err(anyhow!(
                                    "{prefix}.teacher.patch_tokens ({tokens}) must match the student patch count ({student_patch_tokens}) for decoder_mode = \"shared_projection\""
                                ));
                            }
                        }
                    }
                    VisionTeacherDecoderMode::DedicatedProjection => {
                        if matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls)
                            && let Some(tokens) = config
                                .patch_tokens
                                .filter(|tokens| *tokens != student_patch_tokens)
                        {
                            return Err(anyhow!(
                                "{prefix}.teacher.patch_tokens ({tokens}) must match the student patch count ({student_patch_tokens}) for decoder_mode = \"dedicated_projection\"; use \"dedicated_spatial_projection\" to supervise a different spatial grid"
                            ));
                        }
                    }
                    VisionTeacherDecoderMode::DedicatedSpatialProjection => {
                        if matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls) {
                            let patch_tokens = config.patch_tokens.ok_or_else(|| {
                                anyhow!(
                                    "{prefix}.teacher.patch_tokens is required for decoder_mode = \"dedicated_spatial_projection\" with patch-and-cls targets"
                                )
                            })?;
                            let side = (patch_tokens as f64).sqrt().round() as usize;
                            if side.saturating_mul(side) != patch_tokens {
                                return Err(anyhow!(
                                    "{prefix}.teacher.patch_tokens ({patch_tokens}) must form a square patch grid for decoder_mode = \"dedicated_spatial_projection\""
                                ));
                            }
                        }
                    }
                }
            }
            VisionTeacherConfig::Model(_) => {
                return Err(anyhow!(
                    "{prefix}.teacher model-backed auxiliary targets are not supported yet; use precomputed feature targets"
                ));
            }
        }
    }
    Ok(())
}

fn distill_train_features_are_deterministic(augment: &VisionAugmentationConfig) -> bool {
    augment.flip_prob <= 0.0
        && augment.color_jitter_prob <= 0.0
        && augment.grayscale_prob <= 0.0
        && augment.blur_prob <= 0.0
        && augment.solarize_prob <= 0.0
        && (augment.min_scale - 1.0).abs() <= f32::EPSILON
        && (augment.max_scale - 1.0).abs() <= f32::EPSILON
        && (augment.min_aspect_ratio - 1.0).abs() <= f32::EPSILON
        && (augment.max_aspect_ratio - 1.0).abs() <= f32::EPSILON
}

fn validate_input_projection(config: &VisionSaccadeInputProjectionConfig) -> Result<()> {
    match config {
        VisionSaccadeInputProjectionConfig::Linear => Ok(()),
        VisionSaccadeInputProjectionConfig::Cnn(cfg) => {
            if matches!(cfg.channels, Some(0)) {
                return Err(anyhow!(
                    "saccade.input_projection.channels must be > 0 when set"
                ));
            }
            if cfg.expansion == 0 {
                return Err(anyhow!("saccade.input_projection.expansion must be > 0"));
            }
            if cfg.kernel != 0 && cfg.kernel % 2 == 0 {
                return Err(anyhow!(
                    "saccade.input_projection.kernel must be odd when set"
                ));
            }
            Ok(())
        }
        VisionSaccadeInputProjectionConfig::RadialMicroVit(cfg) => {
            if cfg.mlp_ratio == 0 {
                return Err(anyhow!("saccade.input_projection.mlp_ratio must be > 0"));
            }
            if cfg.radial_scale <= 0.0 {
                return Err(anyhow!("saccade.input_projection.radial_scale must be > 0"));
            }
            Ok(())
        }
    }
}

fn validate_recon_loss(label: &str, loss: &VisionReconLossConfig) -> Result<()> {
    if !(0.0..=1.0).contains(&loss.mask_ratio) {
        return Err(anyhow!(
            "{label}.mask_ratio must be in [0, 1] (got {})",
            loss.mask_ratio
        ));
    }
    if loss.weight < 0.0 {
        return Err(anyhow!("{label}.weight must be >= 0"));
    }
    Ok(())
}

fn validate_momentum_teacher(label: &str, teacher: &VisionMomentumTeacherConfig) -> Result<()> {
    if !(0.0..1.0).contains(&teacher.decay) {
        return Err(anyhow!(
            "{label}.decay must be in [0, 1) (got {})",
            teacher.decay
        ));
    }
    Ok(())
}

fn validate_lejepa_loss(loss: &VisionLejepaLossConfig) -> Result<()> {
    if loss.enabled {
        if !(0.0..=1.0).contains(&loss.lambda) {
            return Err(anyhow!(
                "mode.loss.lejepa.lambda must be in [0, 1] (got {})",
                loss.lambda
            ));
        }
        if loss.sigreg_knots == 0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_knots must be > 0"));
        }
        if loss.sigreg_t_max <= 0.0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_t_max must be > 0"));
        }
        if loss.sigreg_proj_dim == 0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_proj_dim must be > 0"));
        }
    }
    Ok(())
}

fn validate_distill_loss(loss: &VisionDistillationLossConfig) -> Result<()> {
    if loss.patch_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.patch_mse_weight must be >= 0"));
    }
    if loss.cls_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_mse_weight must be >= 0"));
    }
    if loss.cls_cosine_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_cosine_weight must be >= 0"));
    }
    if loss.rel_weight < 0.0 {
        return Err(anyhow!("mode.loss.rel_weight must be >= 0"));
    }
    if loss.rel_tau <= 0.0 {
        return Err(anyhow!("mode.loss.rel_tau must be > 0"));
    }
    if matches!(loss.rel_sample_tokens, Some(0)) {
        return Err(anyhow!("mode.loss.rel_sample_tokens must be > 0 when set"));
    }
    Ok(())
}

fn load_value(path: &Path) -> Result<Value> {
    let mut stack = Vec::new();
    load_value_recursive(path, &mut stack)
}

fn load_value_recursive(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let canonical = fs::canonicalize(path).with_context(|| {
        format!(
            "failed to canonicalize configuration file {}",
            path.display()
        )
    })?;
    if let Some(idx) = stack.iter().position(|seen| seen == &canonical) {
        let mut cycle = stack[idx..]
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>();
        cycle.push(canonical.display().to_string());
        return Err(anyhow!(
            "config extends cycle detected: {}",
            cycle.join(" -> ")
        ));
    }

    stack.push(canonical.clone());
    let result = (|| {
        let content = fs::read_to_string(&canonical).with_context(|| {
            format!("failed to read configuration file {}", canonical.display())
        })?;
        let ext = canonical
            .extension()
            .and_then(|value| value.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        let mut value = if ext == "yml" || ext == "yaml" {
            let value: serde_yaml::Value = serde_yaml::from_str(&content)
                .with_context(|| format!("failed to parse {} as YAML", canonical.display()))?;
            yaml_to_toml(value)
                .with_context(|| format!("failed to convert {} from YAML", canonical.display()))?
        } else {
            let table: toml::value::Table = toml::from_str(&content)
                .with_context(|| format!("failed to parse {} as TOML", canonical.display()))?;
            Value::Table(table)
        };

        let extends = take_extends(&mut value)
            .with_context(|| format!("failed to parse extends in {}", canonical.display()))?;
        if let Some(extends) = extends {
            let base_dir = canonical.parent().unwrap_or_else(|| Path::new("."));
            let mut merged = Value::Table(toml::value::Table::new());
            for extend in extends {
                let extend_path = base_dir.join(extend);
                let base = load_value_recursive(&extend_path, stack)?;
                merge_values(&mut merged, base);
            }
            merge_values(&mut merged, value);
            Ok(merged)
        } else {
            Ok(value)
        }
    })();
    stack.pop();
    result
}

fn take_extends(value: &mut Value) -> Result<Option<Vec<PathBuf>>> {
    let Value::Table(table) = value else {
        return Ok(None);
    };
    let Some(extends) = table.remove("extends") else {
        return Ok(None);
    };
    match extends {
        Value::String(path) => Ok(Some(vec![PathBuf::from(path)])),
        Value::Array(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    Value::String(path) => out.push(PathBuf::from(path)),
                    other => {
                        return Err(anyhow!(
                            "extends entries must be strings, got {}",
                            other.type_str()
                        ));
                    }
                }
            }
            Ok(Some(out))
        }
        other => Err(anyhow!(
            "extends must be a string or array of strings, got {}",
            other.type_str()
        )),
    }
}

fn yaml_to_toml(value: serde_yaml::Value) -> Result<Value> {
    match value {
        serde_yaml::Value::Null => Err(anyhow!("null values are not supported in config")),
        serde_yaml::Value::Bool(value) => Ok(Value::Boolean(value)),
        serde_yaml::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                Ok(Value::Integer(value))
            } else if let Some(value) = value.as_u64() {
                Ok(Value::Integer(value as i64))
            } else if let Some(value) = value.as_f64() {
                Ok(Value::Float(value))
            } else {
                Err(anyhow!("unsupported YAML number"))
            }
        }
        serde_yaml::Value::String(value) => Ok(Value::String(value)),
        serde_yaml::Value::Sequence(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                out.push(yaml_to_toml(value)?);
            }
            Ok(Value::Array(out))
        }
        serde_yaml::Value::Mapping(values) => {
            let mut table = toml::value::Table::new();
            for (key, value) in values {
                let key = match key {
                    serde_yaml::Value::String(value) => value,
                    serde_yaml::Value::Number(value) => value.to_string(),
                    serde_yaml::Value::Bool(value) => value.to_string(),
                    serde_yaml::Value::Null => "null".to_string(),
                    other => format!("{other:?}"),
                };
                table.insert(key, yaml_to_toml(value)?);
            }
            Ok(Value::Table(table))
        }
        serde_yaml::Value::Tagged(tagged) => Err(anyhow!(
            "tagged YAML values are not supported: {:?}",
            tagged.tag
        )),
    }
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}
