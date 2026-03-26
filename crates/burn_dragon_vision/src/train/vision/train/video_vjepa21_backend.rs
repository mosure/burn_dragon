use super::*;

pub(super) fn train_video_vjepa21_backend<B>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: Option<PlannedRunArtifacts>,
    backend_name: &str,
    device: &B::Device,
    vision_config: &VisionDragonConfig,
    video: &VisionVideoLejepaConfig,
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
    let clip_frames = video.vjepa21.clip_frames.max(1);
    let build_imagenet_dataset = |split: ImageNetSplit| -> Result<Arc<ImageNetDataset>> {
        maybe_download_vision_dataset(&config.dataset)?;
        let root = match split {
            ImageNetSplit::Train => config.dataset.imagenet_root.join(&config.dataset.train_dir),
            ImageNetSplit::Val => config.dataset.imagenet_root.join(&config.dataset.val_dir),
        };
        let augment = &config.augment;
        let views = clip_frames.max(1);
        let min_view_overlap = if matches!(split, ImageNetSplit::Train) && views > 1 {
            0.5
        } else {
            0.0
        };
        let view_overlap_attempts = if views > 1 { 8 } else { 1 };
        Ok(Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
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
            views,
            local_views: 0,
            min_view_overlap,
            view_overlap_attempts,
            cache_decoded: config.dataset.cache_decoded,
            cache_capacity: config.dataset.cache_capacity,
            cache_preprocessed: config.dataset.cache_preprocessed,
        })?))
    };

    let (
        train_loader,
        valid_loader,
        steps_per_epoch,
        total_epochs,
        total_steps,
        schedule_source,
        num_classes,
        data_source_label,
    ): (
        Arc<dyn DataLoader<B, VideoClipBatch<B>>>,
        Arc<dyn DataLoader<ValidBackend<B>, VideoClipBatch<ValidBackend<B>>>>,
        usize,
        usize,
        usize,
        ScheduleSource,
        usize,
        &'static str,
    ) = match config.dataset.source {
        VisionDatasetSource::MovingMnist => {
            let train_dataset = Arc::new(MovingMnistVideoDataset::new_from_mnist(
                MovingMnistVideoDatasetConfig {
                    split: MovingMnistSplit::Train,
                    frame_size: vision_config.image_size,
                    digit_size: config.dataset.moving_mnist.digit_size,
                    in_channels: vision_config.in_channels,
                    context_len: clip_frames,
                    target_len: 0,
                    extra_future_frames: 0,
                    frame_stride: video.frame_stride.max(1),
                    max_records: config.dataset.max_records,
                    normalize,
                    min_velocity: config.dataset.moving_mnist.min_velocity,
                    max_velocity: config.dataset.moving_mnist.max_velocity,
                    seed: config.dataset.moving_mnist.train_seed,
                },
            )?);
            let val_dataset = Arc::new(MovingMnistVideoDataset::new_from_mnist(
                MovingMnistVideoDatasetConfig {
                    split: MovingMnistSplit::Val,
                    frame_size: vision_config.image_size,
                    digit_size: config.dataset.moving_mnist.digit_size,
                    in_channels: vision_config.in_channels,
                    context_len: clip_frames,
                    target_len: 0,
                    extra_future_frames: 0,
                    frame_stride: video.frame_stride.max(1),
                    max_records: config.dataset.max_records,
                    normalize,
                    min_velocity: config.dataset.moving_mnist.min_velocity,
                    max_velocity: config.dataset.moving_mnist.max_velocity,
                    seed: config.dataset.moving_mnist.val_seed,
                },
            )?);

            let schedule = resolve_vision_train_schedule(
                training,
                train_dataset.steps_per_epoch(training.batch_size),
            )?;
            let steps_per_epoch = schedule.steps_per_epoch;
            let total_epochs = schedule.total_epochs;
            let total_steps = schedule.total_steps;
            let train_loader: Arc<dyn DataLoader<B, VideoClipBatch<B>>> =
                Arc::new(MovingMnistVideoDataLoader::<B>::new(
                    Arc::clone(&train_dataset),
                    device,
                    MovingMnistVideoLoaderConfig {
                        batch_size: training.batch_size,
                        steps_per_epoch,
                        total_steps: Some(total_steps),
                        target_horizon_curriculum: None,
                        prefetch_batches: config.dataset.prefetch_batches,
                        prefetch_workers: config.dataset.prefetch_workers,
                        prefetch_to_device: config.dataset.prefetch_to_device,
                        sequential: false,
                        artifact_capture_every: 0,
                        artifact_capture_images: 0,
                        artifact_extra_future_frames: 0,
                    },
                ));
            let val_steps_per_epoch = val_dataset.steps_per_epoch(training.batch_size);
            let valid_steps = resolve_valid_steps_per_epoch(
                total_steps,
                training.log_frequency,
                val_steps_per_epoch,
            );
            let valid_loader: Arc<
                dyn DataLoader<ValidBackend<B>, VideoClipBatch<ValidBackend<B>>>,
            > = Arc::new(MovingMnistVideoDataLoader::<ValidBackend<B>>::new(
                Arc::clone(&val_dataset),
                device,
                MovingMnistVideoLoaderConfig {
                    batch_size: training.batch_size,
                    steps_per_epoch: valid_steps,
                    total_steps: None,
                    target_horizon_curriculum: None,
                    prefetch_batches: config.dataset.prefetch_batches,
                    prefetch_workers: config.dataset.prefetch_workers,
                    prefetch_to_device: false,
                    sequential: true,
                    artifact_capture_every: video.artifact_every,
                    artifact_capture_images: video.artifact_max_images,
                    artifact_extra_future_frames: 0,
                },
            ));
            (
                train_loader,
                valid_loader,
                steps_per_epoch,
                total_epochs,
                total_steps,
                schedule.source,
                train_dataset.num_classes(),
                "moving_mnist",
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
            let train_image_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>> =
                Arc::new(ImageNetDataLoader::<B>::new(
                    Arc::clone(&train_dataset),
                    training.batch_size,
                    device,
                    steps_per_epoch,
                    Some(total_steps),
                    config.dataset.prefetch_batches,
                    config.dataset.prefetch_workers,
                    config.dataset.prefetch_to_device,
                ));
            let train_loader: Arc<dyn DataLoader<B, VideoClipBatch<B>>> = Arc::new(
                ImageNetVideoDataLoader::<B>::new(train_image_loader, clip_frames, 0, 0),
            );
            let val_steps_per_epoch = val_dataset.steps_per_epoch(training.batch_size);
            let valid_steps = resolve_valid_steps_per_epoch(
                total_steps,
                training.log_frequency,
                val_steps_per_epoch,
            );
            let valid_image_loader: Arc<
                dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>,
            > = Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
                Arc::clone(&val_dataset),
                training.batch_size,
                device,
                valid_steps,
                None,
                config.dataset.prefetch_batches,
                config.dataset.prefetch_workers,
                false,
            ));
            let valid_loader: Arc<
                dyn DataLoader<ValidBackend<B>, VideoClipBatch<ValidBackend<B>>>,
            > = Arc::new(ImageNetVideoDataLoader::<ValidBackend<B>>::new(
                valid_image_loader,
                clip_frames,
                video.artifact_every,
                video.artifact_max_images,
            ));
            (
                train_loader,
                valid_loader,
                steps_per_epoch,
                total_epochs,
                total_steps,
                schedule.source,
                train_dataset.num_classes(),
                "imagenet",
            )
        }
        VisionDatasetSource::Cifar10 | VisionDatasetSource::Cifar100 => {
            unreachable!("V-JEPA 2.1 backend does not support CIFAR datasets");
        }
    };

    info!(
        "vision vjepa21 schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule_source.as_str()
    );
    info!(
        "video vjepa21 data source: {data_source_label}, clip_frames={clip_frames}, frame_stride={}",
        video.frame_stride.max(1)
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
    let mut model = Some(VisionVideoVjepa21Model::new(
        model,
        video.clone(),
        vision_config,
        num_classes,
        device,
    ));
    let mut optim =
        Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionVideoVjepa21Model<B>>());
    let diagnostics = Some(VisionDiagnostics {
        metric_prefix: "video_vjepa21".to_string(),
        distill: false,
        distill_rollout: false,
        inv: true,
        observe: video.vjepa21.loss.predict_all && video.vjepa21.loss.context_weight > 0.0,
        mode_separation: false,
        rollout_horizon_metrics: false,
        sigreg: false,
        recon: video.loss.debug_recon_weight > 0.0,
        directional: false,
        policy: false,
        probe: video.loss.probe_weight > 0.0,
        artifact_every: video.artifact_every,
        artifact_output: video.artifact_output,
        artifact_overwrite: video.artifact_overwrite,
        artifact_max_images: video.artifact_max_images,
        artifact_fps: video.artifact_fps,
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

    info!("Vision video V-JEPA 2.1 training complete on {backend_name}");
    Ok(())
}
