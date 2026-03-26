use super::*;
use crate::train::vision::rac;

pub(super) fn train_rac_backend<B>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: Option<PlannedRunArtifacts>,
    backend_name: &str,
    device: &B::Device,
    vision_config: &VisionDragonConfig,
    rac: &VisionRacConfig,
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
    let build_imagenet_dataset = |split: ImageNetSplit| -> Result<Arc<ImageNetDataset>> {
        maybe_download_vision_dataset(&config.dataset)?;
        let root = match split {
            ImageNetSplit::Train => config.dataset.imagenet_root.join(&config.dataset.train_dir),
            ImageNetSplit::Val => config.dataset.imagenet_root.join(&config.dataset.val_dir),
        };
        let augment = &config.augment;
        let mut dataset = ImageNetDataset::new(ImageNetDatasetConfig {
            root,
            split,
            max_records: config.dataset.max_records,
            augmentations: ImageNetAugmentations::new(
                split,
                vision_config.image_size,
                augment.resize_short,
                augment.min_scale,
                augment.max_scale,
                augment.min_aspect_ratio,
                augment.max_aspect_ratio,
                augment.flip_prob,
                augment.color_jitter_prob,
                augment.brightness,
                augment.contrast,
                augment.saturation,
                augment.hue,
                augment.grayscale_prob,
                augment.blur_prob,
                augment.blur_sigma_min,
                augment.blur_sigma_max,
                augment.solarize_prob,
                augment.solarize_threshold,
            ),
            local_augmentations: None,
            normalize,
            teacher: None,
            rac_teacher_latent: None,
            teacher_targets: Vec::new(),
            views: 1,
            local_views: 0,
            min_view_overlap: 0.0,
            view_overlap_attempts: 1,
            cache_decoded: config.dataset.cache_decoded,
            cache_capacity: config.dataset.cache_capacity,
            cache_preprocessed: config.dataset.cache_preprocessed,
        })?;
        if let Some(store) = rac::build_rac_semantic_teacher_store(
            split,
            rac,
            dataset.len(),
            config.dataset.cache_teacher_features_in_memory,
        )? {
            dataset = dataset.with_teacher(store);
        }
        if let Some(store) = rac::build_rac_teacher_latent_store(
            split,
            rac,
            dataset.len(),
            config.dataset.cache_teacher_features_in_memory,
        )? {
            dataset = dataset.with_rac_teacher_latent(store);
        }
        Ok(Arc::new(dataset))
    };
    let (
        train_loader,
        valid_loader,
        steps_per_epoch,
        total_epochs,
        total_steps,
        schedule_source,
        num_classes,
        source_label,
    ): (
        Arc<dyn DataLoader<B, VisionRacBatch<B>>>,
        Arc<dyn DataLoader<ValidBackend<B>, VisionRacBatch<ValidBackend<B>>>>,
        usize,
        usize,
        usize,
        ScheduleSource,
        usize,
        &'static str,
    ) = match config.dataset.source {
        VisionDatasetSource::Cifar10 | VisionDatasetSource::Cifar100 => {
            let cifar_type = match config.dataset.source {
                VisionDatasetSource::Cifar10 => CifarType::Cifar10,
                VisionDatasetSource::Cifar100 => CifarType::Cifar100,
                _ => unreachable!(),
            };
            let train_dataset = Arc::new(
                CifarDataset::new(&config.dataset.cifar_root, cifar_type, CifarSplit::Train)?
                    .with_max_records(config.dataset.max_records),
            );
            let val_dataset = Arc::new(
                CifarDataset::new(&config.dataset.cifar_root, cifar_type, CifarSplit::Test)?
                    .with_max_records(config.dataset.max_records),
            );
            let schedule = resolve_vision_train_schedule(
                training,
                train_dataset.steps_per_epoch(training.batch_size),
            )?;
            let steps_per_epoch = schedule.steps_per_epoch;
            let total_epochs = schedule.total_epochs;
            let total_steps = schedule.total_steps;
            let valid_steps = resolve_valid_steps_per_epoch(
                total_steps,
                training.log_frequency,
                val_dataset.steps_per_epoch(training.batch_size),
            );
            let train_loader: Arc<dyn DataLoader<B, VisionRacBatch<B>>> =
                Arc::new(VisionRacBatchLoader::new(
                    Arc::new(CifarDataLoader::<B>::new(
                        Arc::clone(&train_dataset),
                        training.batch_size,
                        device,
                        steps_per_epoch,
                        Some(total_steps),
                    )),
                    rac_batch_from_cifar::<B>,
                ));
            let valid_loader: Arc<
                dyn DataLoader<ValidBackend<B>, VisionRacBatch<ValidBackend<B>>>,
            > = Arc::new(VisionRacBatchLoader::new(
                Arc::new(CifarDataLoader::<ValidBackend<B>>::new(
                    Arc::clone(&val_dataset),
                    training.batch_size,
                    device,
                    valid_steps,
                    None,
                )),
                rac_batch_from_cifar::<ValidBackend<B>>,
            ));
            let source_label = if matches!(cifar_type, CifarType::Cifar10) {
                "cifar10"
            } else {
                "cifar100"
            };
            (
                train_loader,
                valid_loader,
                steps_per_epoch,
                total_epochs,
                total_steps,
                schedule.source,
                if matches!(cifar_type, CifarType::Cifar10) {
                    10usize
                } else {
                    100usize
                },
                source_label,
            )
        }
        VisionDatasetSource::Imagenet => {
            let train_dataset = build_imagenet_dataset(ImageNetSplit::Train)?;
            let val_dataset = build_imagenet_dataset(ImageNetSplit::Val)?;
            let schedule = resolve_vision_train_schedule(
                training,
                train_dataset.steps_per_epoch(training.batch_size),
            )?;
            let steps_per_epoch = schedule.steps_per_epoch;
            let total_epochs = schedule.total_epochs;
            let total_steps = schedule.total_steps;
            let valid_steps = resolve_valid_steps_per_epoch(
                total_steps,
                training.log_frequency,
                val_dataset.steps_per_epoch(training.batch_size),
            );
            let train_loader: Arc<dyn DataLoader<B, VisionRacBatch<B>>> =
                Arc::new(VisionRacBatchLoader::new(
                    Arc::new(ImageNetDataLoader::<B>::new(
                        Arc::clone(&train_dataset),
                        training.batch_size,
                        device,
                        steps_per_epoch,
                        Some(total_steps),
                        config.dataset.prefetch_batches,
                        config.dataset.prefetch_workers,
                        config.dataset.prefetch_to_device,
                    )),
                    rac_batch_from_imagenet::<B>,
                ));
            let valid_loader: Arc<
                dyn DataLoader<ValidBackend<B>, VisionRacBatch<ValidBackend<B>>>,
            > = Arc::new(VisionRacBatchLoader::new(
                Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
                    Arc::clone(&val_dataset),
                    training.batch_size,
                    device,
                    valid_steps,
                    None,
                    config.dataset.prefetch_batches,
                    config.dataset.prefetch_workers,
                    false,
                )),
                rac_batch_from_imagenet::<ValidBackend<B>>,
            ));
            (
                train_loader,
                valid_loader,
                steps_per_epoch,
                total_epochs,
                total_steps,
                schedule.source,
                train_dataset.num_classes(),
                "imagenet1k",
            )
        }
        other => {
            return Err(anyhow!(
                "rac backend requires cifar10, cifar100, or imagenet dataset source, got {other:?}"
            ));
        }
    };

    info!(
        "vision rac schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule_source.as_str()
    );
    info!(
        "vision rac data source: {source_label}, backbone={}, sample_steps={}, random_time_grid={}, reset_each_step={}, detach_each_step={}, flow_backprop_steps={:?}, disable_writes={}, eval_wipe_after_step={:?}",
        vision_config.backbone,
        rac.sample_steps.max(1),
        rac.random_time_grid,
        rac.memory.reset_each_step,
        rac.memory.detach_each_step,
        rac.memory.flow_backprop_steps,
        rac.memory.disable_writes,
        rac.memory.eval_wipe_after_step,
    );

    let scheduler_iters = match schedule_source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_vision_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, vision_config)?;

    let planned_run = resolve_vision_run_artifacts(config, config_paths, planned_run)?;
    activate_planned_run(&planned_run)?;
    let run_dir = planned_run.run_dir;
    let run_name = planned_run.run_name;
    crate::write_training_snapshot(config, &run_dir)?;
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
    let mut model = Some(VisionRacModel::new(
        model,
        rac.clone(),
        vision_config,
        num_classes,
        device,
    ));
    let mut optim = Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionRacModel<B>>());
    let diagnostics = Some(VisionDiagnostics {
        metric_prefix: "rac".to_string(),
        distill: false,
        distill_rollout: false,
        inv: true,
        observe: rac.loss.path_weight > 0.0
            || rac.loss.latent_weight > 0.0
            || rac.loss.velocity_weight > 0.0,
        mode_separation: false,
        rollout_horizon_metrics: false,
        sigreg: false,
        recon: rac.loss.recon_weight > 0.0 || rac.loss.roundtrip_weight > 0.0,
        directional: true,
        policy: false,
        probe: rac.loss.probe_weight > 0.0,
        artifact_every: rac.artifact_every,
        artifact_output: rac.artifact_output,
        artifact_overwrite: rac.artifact_overwrite,
        artifact_max_images: rac.artifact_max_images,
        artifact_fps: rac.artifact_fps,
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
        ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
            diagnostics,
        )?,
    }

    write_rac_best_checkpoint_report(&run_dir, &run_name)?;
    info!("Vision RAC training complete on {backend_name}");
    Ok(())
}
