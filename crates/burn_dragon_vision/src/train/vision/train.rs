use super::distill_runtime::build_distill_datasets_and_teacher;
use crate::train::prelude::*;

pub fn train_vision_backend<B, Init>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;
    if training.batch_size == 0 {
        return Err(anyhow!("vision training batch_size must be > 0"));
    }

    let vision_config = config.vision.build();
    if vision_config.patch_size == 0 {
        return Err(anyhow!("vision.patch_size must be > 0"));
    }
    if config.augment.image_size != vision_config.image_size {
        return Err(anyhow!(
            "augment.image_size ({}) must match vision.image_size ({})",
            config.augment.image_size,
            vision_config.image_size
        ));
    }
    let rollout = resolve_vision_rollout(training, vision_config.steps)?;
    info!(
        "vision rollout steps: min={}, max={}, backprop={}",
        rollout.min_steps, rollout.max_steps, rollout.backprop_steps
    );

    maybe_download_vision_dataset(&config.dataset)?;

    if let VisionTrainingModeConfig::VideoLejepa(video) = &config.mode {
        return train_video_lejepa_backend::<B>(
            config,
            backend_name,
            &device,
            &vision_config,
            video,
            rollout,
            optimizer_cfg,
        );
    }

    let grid = vision_config.image_size.div_ceil(vision_config.patch_size);
    let student_patch_tokens = grid * grid;

    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_aug = ImageNetAugmentations::new(
        ImageNetSplit::Train,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let val_aug = ImageNetAugmentations::new(
        ImageNetSplit::Val,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );

    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);

    enum VisionMode<B: BackendTrait> {
        Distill {
            teacher: Option<Box<crate::train::vision::models::DistillTeacherModel<B>>>,
        },
        Lejepa {
            config: VisionLejepaConfig,
        },
        Mae {
            config: VisionMaeConfig,
        },
        Saccade {
            config: Box<VisionSaccadeConfig>,
        },
    }

    let (train_dataset, val_dataset, mode) = match &config.mode {
        VisionTrainingModeConfig::Distill(_distill) => {
            let (train_dataset, val_dataset, teacher) = build_distill_datasets_and_teacher::<B>(
                config,
                &vision_config,
                normalize,
                train_aug,
                val_aug,
                &train_root,
                &val_root,
                student_patch_tokens,
                &device,
            )?;

            (
                train_dataset,
                val_dataset,
                VisionMode::Distill {
                    teacher: teacher.map(Box::new),
                },
            )
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            let multi_crop = lejepa.global_views > 0 || lejepa.local_views > 0;
            let global_views = if multi_crop {
                lejepa.global_views.max(1)
            } else {
                lejepa.views.max(1)
            };
            let local_views = if multi_crop { lejepa.local_views } else { 0 };
            if global_views + local_views == 0 {
                return Err(anyhow!(
                    "lejepa must have at least one global or local view"
                ));
            }
            if local_views > 0 {
                if lejepa.local_image_size == 0 {
                    return Err(anyhow!("lejepa.local_image_size must be > 0"));
                }
                if !lejepa
                    .local_image_size
                    .is_multiple_of(vision_config.patch_size)
                {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be divisible by patch_size ({})",
                        lejepa.local_image_size,
                        vision_config.patch_size
                    ));
                }
                if lejepa.local_image_size > vision_config.image_size {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be <= vision.image_size ({})",
                        lejepa.local_image_size,
                        vision_config.image_size
                    ));
                }
            }
            if lejepa.loss.recon.weight < 0.0 {
                return Err(anyhow!("lejepa.loss.recon.weight must be >= 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "lejepa.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    lejepa.loss.recon.mask_ratio
                ));
            }
            if lejepa.loss.lejepa.enabled {
                if !(0.0..=1.0).contains(&lejepa.loss.lejepa.lambda) {
                    return Err(anyhow!(
                        "lejepa.loss.lejepa.lambda must be in [0, 1] (got {})",
                        lejepa.loss.lejepa.lambda
                    ));
                }
                if lejepa.loss.lejepa.sigreg_knots == 0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_knots must be > 0"));
                }
                if lejepa.loss.lejepa.sigreg_t_max <= 0.0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_t_max must be > 0"));
                }
                if lejepa.loss.lejepa.sigreg_proj_dim == 0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_proj_dim must be > 0"));
                }
            }
            let local_train_aug = if local_views > 0 {
                Some(ImageNetAugmentations::new(
                    ImageNetSplit::Train,
                    lejepa.local_image_size,
                    lejepa.local_image_size,
                    lejepa.local_min_scale,
                    lejepa.local_max_scale,
                    config.augment.min_aspect_ratio,
                    config.augment.max_aspect_ratio,
                    config.augment.flip_prob,
                    config.augment.color_jitter_prob,
                    config.augment.brightness,
                    config.augment.contrast,
                    config.augment.saturation,
                    config.augment.hue,
                    config.augment.grayscale_prob,
                    config.augment.blur_prob,
                    config.augment.blur_sigma_min,
                    config.augment.blur_sigma_max,
                    config.augment.solarize_prob,
                    config.augment.solarize_threshold,
                ))
            } else {
                None
            };
            let local_val_aug = if local_views > 0 {
                Some(ImageNetAugmentations::new(
                    ImageNetSplit::Val,
                    lejepa.local_image_size,
                    lejepa.local_image_size,
                    lejepa.local_min_scale,
                    lejepa.local_max_scale,
                    config.augment.min_aspect_ratio,
                    config.augment.max_aspect_ratio,
                    config.augment.flip_prob,
                    config.augment.color_jitter_prob,
                    config.augment.brightness,
                    config.augment.contrast,
                    config.augment.saturation,
                    config.augment.hue,
                    config.augment.grayscale_prob,
                    config.augment.blur_prob,
                    config.augment.blur_sigma_min,
                    config.augment.blur_sigma_max,
                    config.augment.solarize_prob,
                    config.augment.solarize_threshold,
                ))
            } else {
                None
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: local_train_aug.clone(),
                normalize,
                teacher: None,
                views: global_views,
                local_views,
                min_view_overlap: lejepa.min_view_overlap,
                view_overlap_attempts: lejepa.view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: local_val_aug.clone(),
                normalize,
                teacher: None,
                views: global_views,
                local_views,
                min_view_overlap: lejepa.min_view_overlap,
                view_overlap_attempts: lejepa.view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Lejepa {
                    config: lejepa.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Mae(mae) => {
            if !(0.0..=1.0).contains(&mae.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "mae.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    mae.loss.recon.mask_ratio
                ));
            }
            if mae.loss.recon.weight < 0.0 {
                return Err(anyhow!("mae.loss.recon.weight must be >= 0"));
            }
            if mae.pyramid_levels == 0 {
                return Err(anyhow!("mae.pyramid_levels must be > 0"));
            }
            let views = if mae.cross_view.enabled {
                config.vision.num_eyes.max(1)
            } else {
                1
            };
            let min_view_overlap = if mae.cross_view.enabled {
                mae.cross_view.min_overlap.max(0.0)
            } else {
                0.0
            };
            let view_overlap_attempts = if mae.cross_view.enabled {
                mae.cross_view.max_attempts.max(1)
            } else {
                1
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Mae {
                    config: mae.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            let mut saccade = (**saccade).clone();
            if saccade.num_eyes == 0 {
                saccade.num_eyes = config.vision.num_eyes.max(1);
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "saccade.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    saccade.loss.recon.mask_ratio
                ));
            }
            if saccade.loss.recon.weight < 0.0 {
                return Err(anyhow!("saccade.loss.recon.weight must be >= 0"));
            }
            if saccade.loss.lejepa.enabled {
                if !(0.0..=1.0).contains(&saccade.loss.lejepa.lambda) {
                    return Err(anyhow!(
                        "saccade.loss.lejepa.lambda must be in [0, 1] (got {})",
                        saccade.loss.lejepa.lambda
                    ));
                }
                if saccade.loss.lejepa.sigreg_knots == 0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_knots must be > 0"));
                }
                if saccade.loss.lejepa.sigreg_t_max <= 0.0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_t_max must be > 0"));
                }
                if saccade.loss.lejepa.sigreg_proj_dim == 0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_proj_dim must be > 0"));
                }
            }
            if saccade.policy.info_reward.stride == 0 {
                return Err(anyhow!("saccade.policy.info_reward.stride must be > 0"));
            }
            if saccade.policy.location_embedding.quantize_bins < 2 {
                return Err(anyhow!(
                    "saccade.policy.location_embedding.quantize_bins must be >= 2"
                ));
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
                match saccade.policy.gdpo.hard_gate {
                    GdpoHardGate::Off => {}
                    GdpoHardGate::Fixed { .. } => {}
                    GdpoHardGate::Percentile { quantile } => {
                        if !(0.0..=1.0).contains(&quantile) {
                            return Err(anyhow!(
                                "saccade.policy.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                                quantile
                            ));
                        }
                    }
                }
            }
            let views = if saccade.cross_view.enabled {
                saccade.num_eyes.max(1)
            } else {
                1
            };
            let min_view_overlap = if saccade.cross_view.enabled {
                saccade.cross_view.min_overlap.max(0.0)
            } else {
                0.0
            };
            let view_overlap_attempts = if saccade.cross_view.enabled {
                saccade.cross_view.max_attempts.max(1)
            } else {
                1
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Saccade {
                    config: Box::new(saccade.clone()),
                },
            )
        }
        VisionTrainingModeConfig::VideoLejepa(_) => {
            unreachable!("video LEJEPA is handled by the dedicated early-return path")
        }
    };

    let steps_per_epoch = train_dataset.steps_per_epoch(training.batch_size);
    let schedule = resolve_vision_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "vision schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );

    let prefetch_to_device = config.dataset.prefetch_to_device;

    let train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>> =
        Arc::new(ImageNetDataLoader::<B>::new(
            Arc::clone(&train_dataset),
            training.batch_size,
            &device,
            steps_per_epoch,
            Some(total_steps),
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            prefetch_to_device,
        ));

    let val_steps_per_epoch = val_dataset.steps_per_epoch(training.batch_size);
    let valid_steps =
        resolve_valid_steps_per_epoch(total_steps, training.log_frequency, val_steps_per_epoch);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>> =
        Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&val_dataset),
            training.batch_size,
            &valid_device,
            valid_steps,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            prefetch_to_device,
        ));

    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_vision_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &vision_config)?;

    let run_root = PathBuf::from("runs").join("vision");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    info!("vision run name: {run_name}");
    let context = VisionTrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };

    match mode {
        VisionMode::Distill { teacher } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let teacher = teacher.map(|teacher| *teacher);
            let distill = match &config.mode {
                VisionTrainingModeConfig::Distill(distill) => distill.clone(),
                _ => unreachable!("distill mode branch should only run for distill configs"),
            };
            let mut model = Some(VisionDistillModel::new(model, distill, teacher, rollout));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionDistillModel<B>>());
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "distill".to_string(),
                inv: false,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: false,
                recon: false,
                policy: false,
                probe: false,
                distill: true,
                distill_rollout: true,
                artifact_every: 0,
                artifact_output: VisionArtifactOutputMode::Images,
                artifact_overwrite: false,
                artifact_max_images: 0,
                artifact_fps: 1,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics,
                )?,
            }
        }
        VisionMode::Lejepa { config: lejepa } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let recon = VisionReconstructionInit {
                patch_dim: recon_patch_dim,
                normalize_std: config.augment.normalize_std,
                patch_size: vision_config.patch_size,
                in_channels: vision_config.in_channels,
            };
            let mut model = Some(VisionLejepaModel::new(
                model,
                lejepa,
                VisionLejepaInit {
                    embed_dim: vision_config.embed_dim,
                    num_classes: train_dataset.num_classes(),
                    rollout,
                    recon,
                },
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionLejepaModel<B>>());
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "lejepa".to_string(),
                distill: false,
                distill_rollout: false,
                inv: model.as_ref().expect("model").config.loss.lejepa.enabled,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: model.as_ref().expect("model").config.loss.lejepa.enabled,
                recon: model.as_ref().expect("model").config.loss.recon.weight > 0.0,
                policy: false,
                probe: true,
                artifact_every: model.as_ref().expect("model").config.artifact_every,
                artifact_output: model.as_ref().expect("model").config.artifact_output,
                artifact_overwrite: model.as_ref().expect("model").config.artifact_overwrite,
                artifact_max_images: model.as_ref().expect("model").config.artifact_max_images,
                artifact_fps: model.as_ref().expect("model").config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Mae { config: mae } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let recon = VisionReconstructionInit {
                patch_dim: recon_patch_dim,
                normalize_std: config.augment.normalize_std,
                patch_size: vision_config.patch_size,
                in_channels: vision_config.in_channels,
            };
            let mut model = Some(VisionMaeModel::new(
                model,
                mae,
                VisionMaeInit {
                    num_eyes: vision_config.num_eyes,
                    embed_dim: vision_config.embed_dim,
                    rollout,
                    recon,
                },
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionMaeModel<B>>());
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "mae".to_string(),
                distill: false,
                distill_rollout: false,
                inv: false,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: false,
                recon: model_ref.config.loss.recon.weight > 0.0,
                policy: false,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Saccade { config: saccade } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionSaccadeModel::new(
                model,
                *saccade,
                vision_config.embed_dim,
                vision_config.patch_size,
                rollout,
                recon_patch_dim,
                training.batch_repeats,
                training.train_repeat_chunk,
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionSaccadeModel<B>>());
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "saccade".to_string(),
                distill: false,
                distill_rollout: false,
                inv: model_ref.config.loss.lejepa.enabled,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: model_ref.config.loss.lejepa.enabled,
                recon: model_ref.config.loss.recon.weight > 0.0,
                policy: model_ref.config.policy.gdpo.enabled,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
    }

    info!("Vision training complete on {backend_name}");

    Ok(())
}

fn train_video_lejepa_backend<B>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    device: &B::Device,
    vision_config: &VisionDragonConfig,
    video: &VisionVideoLejepaConfig,
    rollout: VisionRollout,
    optimizer_cfg: &OptimizerConfig,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    crate::device::pin_stream_zero();
    let training = &config.training;
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_target_frames_min = video.effective_train_target_frames_min();
    let train_target_frames_max = video.effective_train_target_frames_max();
    let target_horizon_curriculum = Some(VideoTargetHorizonCurriculum {
        min_target_len: train_target_frames_min,
        max_target_len: train_target_frames_max,
        warmup_steps: video.train_target_warmup_steps,
        seed: config.dataset.moving_mnist.train_seed ^ 0xA11B_1C0E_5EED_u64,
    });
    let train_dataset = Arc::new(MovingMnistVideoDataset::new_from_mnist(
        MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Train,
            frame_size: vision_config.image_size,
            digit_size: config.dataset.moving_mnist.digit_size,
            in_channels: vision_config.in_channels,
            context_len: video.context_frames,
            target_len: train_target_frames_max,
            extra_future_frames: 0,
            frame_stride: video.frame_stride,
            max_records: config.dataset.max_records,
            normalize,
            min_velocity: config.dataset.moving_mnist.min_velocity,
            max_velocity: config.dataset.moving_mnist.max_velocity,
            seed: config.dataset.moving_mnist.train_seed,
        },
    )?);
    let val_extra_future_frames = video
        .artifact_future_frames
        .saturating_sub(video.target_frames);
    let val_dataset = Arc::new(MovingMnistVideoDataset::new_from_mnist(
        MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Val,
            frame_size: vision_config.image_size,
            digit_size: config.dataset.moving_mnist.digit_size,
            in_channels: vision_config.in_channels,
            context_len: video.context_frames,
            target_len: video.target_frames,
            extra_future_frames: val_extra_future_frames,
            frame_stride: video.frame_stride,
            max_records: config.dataset.max_records,
            normalize,
            min_velocity: config.dataset.moving_mnist.min_velocity,
            max_velocity: config.dataset.moving_mnist.max_velocity,
            seed: config.dataset.moving_mnist.val_seed,
        },
    )?);

    let steps_per_epoch = train_dataset.steps_per_epoch(training.batch_size);
    let schedule = resolve_vision_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "vision video schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );
    info!(
        "video train target horizon: min={}, max={}, warmup_steps={}",
        train_target_frames_min, train_target_frames_max, video.train_target_warmup_steps
    );

    let train_loader: Arc<dyn DataLoader<B, VideoClipBatch<B>>> =
        Arc::new(MovingMnistVideoDataLoader::<B>::new(
            Arc::clone(&train_dataset),
            training.batch_size,
            device,
            steps_per_epoch,
            Some(total_steps),
            target_horizon_curriculum,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            config.dataset.prefetch_to_device,
            false,
            0,
            0,
            0,
        ));

    let long_rollout_validation =
        video.artifact_future_frames > video.max_supervised_target_frames();
    let valid_batch_size = if long_rollout_validation {
        training.batch_size.min(video.artifact_max_images.max(1))
    } else {
        training.batch_size
    };
    let val_steps_per_epoch = val_dataset.steps_per_epoch(valid_batch_size);
    let valid_steps =
        resolve_valid_steps_per_epoch(total_steps, training.log_frequency, val_steps_per_epoch);
    let valid_device = device.clone();
    let artifact_capture_every = if video.artifact_every > 0 && video.artifact_max_images > 0 {
        video.artifact_every
    } else {
        0
    };
    let artifact_capture_images = if artifact_capture_every > 0 {
        video.artifact_max_images
    } else {
        0
    };
    let artifact_extra_future_frames = if artifact_capture_every > 0 {
        val_extra_future_frames
    } else {
        0
    };
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, VideoClipBatch<ValidBackend<B>>>> =
        Arc::new(MovingMnistVideoDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&val_dataset),
            valid_batch_size,
            &valid_device,
            valid_steps,
            None,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            false,
            true,
            artifact_capture_every,
            artifact_capture_images,
            artifact_extra_future_frames,
        ));
    info!(
        "video valid loader: batch_size={valid_batch_size}, steps_per_epoch={val_steps_per_epoch}, long_rollout_validation={long_rollout_validation}"
    );

    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_vision_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, vision_config)?;

    let run_root = PathBuf::from("runs").join("vision");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    info!("vision run name: {run_name}");

    let context = VisionTrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };

    let model = VisionDragon::<B>::new(vision_config.clone(), device);
    let mut model = Some(VisionVideoLejepaModel::new(
        model,
        video.clone(),
        vision_config,
        rollout,
        train_dataset.num_classes(),
        device,
    ));
    let mut optim =
        Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionVideoLejepaModel<B>>());
    let diagnostics = Some(VisionDiagnostics {
        metric_prefix: "video_lejepa".to_string(),
        distill: false,
        distill_rollout: false,
        inv: true,
        observe: model.as_ref().expect("model").config.loss.observe_weight > 0.0,
        mode_separation: true,
        rollout_horizon_metrics: true,
        sigreg: model.as_ref().expect("model").config.loss.sigreg.enabled,
        recon: model
            .as_ref()
            .expect("model")
            .config
            .loss
            .debug_recon_weight
            > 0.0,
        policy: false,
        probe: true,
        artifact_every: model.as_ref().expect("model").config.artifact_every,
        artifact_output: model.as_ref().expect("model").config.artifact_output,
        artifact_overwrite: model.as_ref().expect("model").config.artifact_overwrite,
        artifact_max_images: model.as_ref().expect("model").config.artifact_max_images,
        artifact_fps: model.as_ref().expect("model").config.artifact_fps,
        normalize_mean: config.augment.normalize_mean,
        normalize_std: config.augment.normalize_std,
        ffmpeg_path: training.ffmpeg_path.clone(),
    });

    match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
            diagnostics,
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
    }

    info!("Vision video LEJEPA training complete on {backend_name}");

    Ok(())
}

#[cfg(feature = "integration_test")]
pub fn train_vision_backend_for_test<B, Init>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_vision_backend::<B, Init>(config, backend_name, init_backend)
}
