use super::*;

pub(super) fn train_video_lejepa_backend<B>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: Option<PlannedRunArtifacts>,
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
            device,
            MovingMnistVideoLoaderConfig {
                batch_size: training.batch_size,
                steps_per_epoch,
                total_steps: Some(total_steps),
                target_horizon_curriculum,
                prefetch_batches: config.dataset.prefetch_batches,
                prefetch_workers: config.dataset.prefetch_workers,
                prefetch_to_device: config.dataset.prefetch_to_device,
                sequential: false,
                artifact_capture_every: 0,
                artifact_capture_images: 0,
                artifact_extra_future_frames: 0,
            },
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
            &valid_device,
            MovingMnistVideoLoaderConfig {
                batch_size: valid_batch_size,
                steps_per_epoch: valid_steps,
                total_steps: None,
                target_horizon_curriculum: None,
                prefetch_batches: config.dataset.prefetch_batches,
                prefetch_workers: config.dataset.prefetch_workers,
                prefetch_to_device: false,
                sequential: true,
                artifact_capture_every,
                artifact_capture_images,
                artifact_extra_future_frames,
            },
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
        directional: false,
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
        ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
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
