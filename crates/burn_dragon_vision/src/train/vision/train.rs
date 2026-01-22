use crate::train::prelude::*;

pub(crate) fn train_vision_backend<B, Init>(
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

    let grid = vision_config
        .image_size
        .div_ceil(vision_config.patch_size);
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
            loss: VisionDistillationLossConfig,
            teacher: Option<Box<DinoVisionTransformer<B>>>,
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
        VisionTrainingModeConfig::Distill(distill) => {
            let (train_dataset, val_dataset, teacher) = match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    if teacher.feature_dim != vision_config.projection_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                            teacher.feature_dim,
                            vision_config.projection_dim
                        ));
                    }
                    if let Some(tokens) = teacher
                        .patch_tokens
                        .filter(|tokens| *tokens != student_patch_tokens)
                    {
                        return Err(anyhow!(
                            "teacher.patch_tokens ({}) must match ceil(image_size/patch_size)^2 ({})",
                            tokens,
                            student_patch_tokens
                        ));
                    }
                    let teacher_tokens = teacher.patch_tokens.unwrap_or(student_patch_tokens);

                    let mut train_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                        root: train_root,
                        split: ImageNetSplit::Train,
                        max_records: config.dataset.max_records,
                        augmentations: train_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        min_view_overlap: 0.0,
                        view_overlap_attempts: 1,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                        cache_preprocessed: config.dataset.cache_preprocessed,
                    })?;
                    let train_records = train_dataset.len();
                    let train_teacher = Arc::new(DinoFeatureStore::new(
                        &teacher.train_cls_path,
                        &teacher.train_patch_path,
                        teacher.feature_dim,
                        teacher_tokens,
                        Some(train_records),
                    )?);
                    train_dataset = train_dataset.with_teacher(Arc::clone(&train_teacher));
                    let train_dataset = Arc::new(train_dataset);

                    let mut val_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                        root: val_root,
                        split: ImageNetSplit::Val,
                        max_records: config.dataset.max_records,
                        augmentations: val_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        min_view_overlap: 0.0,
                        view_overlap_attempts: 1,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                        cache_preprocessed: config.dataset.cache_preprocessed,
                    })?;
                    let val_records = val_dataset.len();
                    let val_teacher = Arc::new(DinoFeatureStore::new(
                        &teacher.val_cls_path,
                        &teacher.val_patch_path,
                        teacher.feature_dim,
                        teacher_tokens,
                        Some(val_records),
                    )?);
                    val_dataset = val_dataset.with_teacher(Arc::clone(&val_teacher));
                    let val_dataset = Arc::new(val_dataset);
                    (train_dataset, val_dataset, None)
                }
                VisionTeacherConfig::Model(teacher) => {
                    let image_size = teacher.image_size.unwrap_or(vision_config.image_size);
                    let patch_size = teacher.patch_size.unwrap_or(vision_config.patch_size);
                    if patch_size == 0 {
                        return Err(anyhow!("teacher.patch_size must be > 0"));
                    }
                    if !image_size.is_multiple_of(patch_size) {
                        return Err(anyhow!(
                            "teacher image_size must be divisible by patch_size ({} % {} != 0)",
                            image_size,
                            patch_size
                        ));
                    }
                    let teacher_grid = image_size.div_ceil(patch_size);
                    let teacher_tokens = teacher_grid * teacher_grid;
                    if teacher_tokens != student_patch_tokens {
                        return Err(anyhow!(
                            "teacher patch tokens ({}) must match student tokens ({})",
                            teacher_tokens,
                            student_patch_tokens
                        ));
                    }
                    if let Some(tokens) = teacher
                        .patch_tokens
                        .filter(|tokens| *tokens != teacher_tokens)
                    {
                        return Err(anyhow!(
                            "teacher.patch_tokens ({}) must match ceil(image_size/patch_size)^2 ({})",
                            tokens,
                            teacher_tokens
                        ));
                    }

                    let feature_dim = teacher
                        .feature_dim
                        .unwrap_or_else(|| teacher_variant_dim(teacher.variant));
                    if feature_dim != vision_config.projection_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                            feature_dim,
                            vision_config.projection_dim
                        ));
                    }

                    let mut dino_config =
                        build_dino_config(teacher.variant, image_size, patch_size);
                    if teacher.register_tokens > 0 {
                        dino_config = dino_config.with_register_tokens(teacher.register_tokens);
                    }
                    if dino_config.embedding_dimension != feature_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match DINO embedding dim ({})",
                            feature_dim,
                            dino_config.embedding_dimension
                        ));
                    }

                    let teacher_model = load_model_from_checkpoint::<B>(
                        &dino_config,
                        &teacher.checkpoint_path,
                        &device,
                    )
                    .map_err(|err| {
                        anyhow!(
                            "failed to load teacher checkpoint {}: {err}",
                            teacher.checkpoint_path.display()
                        )
                    })?
                    .no_grad();

                    let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                        root: train_root,
                        split: ImageNetSplit::Train,
                        max_records: config.dataset.max_records,
                        augmentations: train_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        min_view_overlap: 0.0,
                        view_overlap_attempts: 1,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                        cache_preprocessed: config.dataset.cache_preprocessed,
                    })?);
                    let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                        root: val_root,
                        split: ImageNetSplit::Val,
                        max_records: config.dataset.max_records,
                        augmentations: val_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        min_view_overlap: 0.0,
                        view_overlap_attempts: 1,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                        cache_preprocessed: config.dataset.cache_preprocessed,
                    })?);

                    (train_dataset, val_dataset, Some(Box::new(teacher_model)))
                }
            };

            (
                train_dataset,
                val_dataset,
                VisionMode::Distill {
                    loss: distill.loss.clone(),
                    teacher,
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
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
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
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
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
            let mut saccade = saccade.clone();
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
    let desired_valid_steps = usize::max(1, total_steps / training.log_frequency.max(1));
    let valid_steps = desired_valid_steps.min(val_steps_per_epoch).max(1);

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
        VisionMode::Distill { loss, teacher } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let teacher = teacher.map(|teacher| *teacher);
            let mut model = Some(VisionDistillModel::new(model, loss, teacher, rollout));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionDistillModel<B>>());
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    None,
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
            }
        }
        VisionMode::Lejepa { config: lejepa } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionLejepaModel::new(
                model,
                lejepa,
                vision_config.embed_dim,
                train_dataset.num_classes(),
                rollout,
                recon_patch_dim,
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionLejepaModel<B>>());
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "lejepa".to_string(),
                inv: model.as_ref().expect("model").config.loss.lejepa.enabled,
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
            let mut model = Some(VisionMaeModel::new(
                model,
                mae,
                vision_config.num_eyes,
                vision_config.embed_dim,
                rollout,
                recon_patch_dim,
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionMaeModel<B>>());
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "mae".to_string(),
                inv: false,
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
                inv: model_ref.config.loss.lejepa.enabled,
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

