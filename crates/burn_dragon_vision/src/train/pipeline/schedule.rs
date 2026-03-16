use crate::train::prelude::*;
use burn_train::metric::IterationSpeedMetric;

pub use burn_dragon_train::train::pipeline::{ResolvedLrScheduler, ScheduleSource, TrainSchedule};

pub struct VisionTrainEnvironment<'a, B, TrainBatch, ValidBatch = TrainBatch>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a VisionTrainingHyperparameters,
    pub device: &'a B::Device,
    pub train_loader: Arc<dyn DataLoader<B, TrainBatch>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ValidBatch>>,
    pub epochs: usize,
}

#[derive(Clone, Copy, Debug, Module)]
pub struct VisionRollout {
    pub min_steps: usize,
    pub max_steps: usize,
    pub backprop_steps: usize,
}

impl VisionRollout {
    pub fn sample_steps(&self) -> usize {
        if self.min_steps >= self.max_steps {
            self.max_steps
        } else {
            thread_rng().gen_range(self.min_steps..=self.max_steps)
        }
    }

    pub fn backprop_steps(&self, steps: usize) -> usize {
        if self.backprop_steps == 0 {
            steps.max(1)
        } else {
            self.backprop_steps.min(steps).max(1)
        }
    }
}

#[derive(Clone)]
pub struct VisionDiagnostics {
    pub metric_prefix: String,
    pub distill: bool,
    pub distill_rollout: bool,
    pub inv: bool,
    pub observe: bool,
    pub mode_separation: bool,
    pub rollout_horizon_metrics: bool,
    pub sigreg: bool,
    pub recon: bool,
    pub policy: bool,
    pub probe: bool,
    pub artifact_every: usize,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_overwrite: bool,
    pub artifact_max_images: usize,
    pub artifact_fps: u32,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
    pub ffmpeg_path: Option<PathBuf>,
}

pub fn train_vision_with_scheduler<B, S, M, TrainBatch, ValidBatch>(
    env: &VisionTrainEnvironment<'_, B, TrainBatch, ValidBatch>,
    model: M,
    optimizer: OptimizerAdaptor<AdamW, M, B>,
    scheduler: S,
    vision_diagnostics: Option<VisionDiagnostics>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    TrainBatch: Send + 'static,
    ValidBatch: Send + 'static,
    M: AutodiffModule<B>
        + TrainStep<Input = TrainBatch, Output = VisionTrainItem<B>>
        + core::fmt::Display
        + Clone
        + 'static,
    M::InnerModule: ValidStep<Input = ValidBatch, Output = VisionOutput<ValidBackend<B>>>,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let metric_every = env.training.log_frequency.max(1);
    let loss_every = 1;
    let enable_checkpoints = should_enable_vision_checkpoints(env.training, env.backend_name);
    if env.training.enable_checkpoints && !enable_checkpoints {
        tracing::warn!(
            "vision checkpoints disabled for backend {} on this platform to avoid stack overflow",
            env.backend_name
        );
    }
    let mut builder = SupervisedTraining::new(
        env.run_dir,
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    )
    .num_epochs(env.epochs)
    .grads_accumulation(env.training.gradient_accumulation_steps.max(1))
    .with_training_strategy(LearningStrategy::SingleDevice(env.device.clone()));
    if enable_checkpoints {
        builder = builder.with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new());
    }
    builder = builder
        .metric_train_numeric(IterationSpeedMetric::new())
        .metric_train_numeric(
            ScalarMetric::<MetricsBackend, LossValue<MetricsBackend>>::new_every(
                "Loss", loss_every,
            ),
        )
        .metric_valid_numeric(LossMetric::<MetricsBackend>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name))
        .summary();

    info!("vision run name: {}", env.run_name);

    #[cfg(feature = "integration_test")]
    if env.training.trace_train_loss {
        builder = builder.metric_train(burn_dragon_train::train::metrics::LossTraceMetric::<
            MetricsBackend,
        >::new(
            "loss_trace", env.training.trace_train_loss_every
        ));
    }

    let train_cleanup_every = env.training.memory_cleanup_every;
    let train_cleanup_iters = env.training.memory_cleanup_iters;
    let valid_cleanup_every = train_cleanup_every;
    let valid_cleanup_iters = env.training.memory_cleanup_iters;
    if train_cleanup_every > 0
        || train_cleanup_iters > 0
        || valid_cleanup_every > 0
        || valid_cleanup_iters > 0
    {
        let allow_cuda_cleanup = !env.training.disable_cuda_memory_cleanup;
        if train_cleanup_every > 0 || train_cleanup_iters > 0 {
            builder = builder.metric_train(MemoryCleanupMetric::<B>::new(
                env.device,
                train_cleanup_every,
                train_cleanup_iters,
                allow_cuda_cleanup,
            ));
        }
        if valid_cleanup_every > 0 || valid_cleanup_iters > 0 {
            builder = builder.metric_valid(MemoryCleanupMetric::<ValidBackend<B>>::new(
                env.device,
                valid_cleanup_every,
                valid_cleanup_iters,
                allow_cuda_cleanup,
            ));
        }
    }

    let memory_check_every = env.training.device_memory_check_every;
    let max_device_memory_mb = env.training.max_device_memory_mb;
    if max_device_memory_mb > 0 || memory_check_every > 0 {
        let check_every = if memory_check_every == 0 {
            metric_every
        } else {
            memory_check_every
        };
        let allow_cuda_cleanup = !env.training.disable_cuda_memory_cleanup;
        builder = builder
            .metric_train(DeviceMemoryMetric::<B>::new(
                env.device,
                check_every,
                max_device_memory_mb,
                allow_cuda_cleanup,
            ))
            .metric_valid(DeviceMemoryMetric::<ValidBackend<B>>::new(
                env.device,
                check_every,
                max_device_memory_mb,
                allow_cuda_cleanup,
            ));
    }

    if let Some(diagnostics) = &vision_diagnostics {
        let prefix = diagnostics.metric_prefix.as_str();
        if diagnostics.distill {
            let patch_name = format!("{prefix}_patch_loss");
            let cls_name = format!("{prefix}_cls_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<MetricsBackend, InvLossInput<MetricsBackend>>::new_every(
                        patch_name.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<MetricsBackend, InvLossInput<MetricsBackend>>::new_every(
                        patch_name.as_str(),
                        metric_every,
                    ),
                )
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ObserveLossInput<MetricsBackend>,
                >::new_every(cls_name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ObserveLossInput<MetricsBackend>,
                >::new_every(cls_name.as_str(), metric_every));
        }
        if diagnostics.distill_rollout {
            for (index, step) in VISION_ROLLOUT_HORIZON_CAPS.into_iter().enumerate() {
                let total_name = format!("{prefix}_total_to_s{step}");
                let patch_name = format!("{prefix}_patch_to_s{step}");
                let cls_name = format!("{prefix}_cls_to_s{step}");
                builder = match index {
                    0 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 0>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    1 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 1>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    2 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 2>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    3 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 3>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    4 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 4>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    5 => builder
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutInvToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            total_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateNormRatioToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            patch_name.as_str(), metric_every
                        ))
                        .metric_train_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        ))
                        .metric_valid_numeric(ScalarMetric::<
                            MetricsBackend,
                            RolloutStateMotionToHorizonInput<MetricsBackend, 5>,
                        >::new_every(
                            cls_name.as_str(), metric_every
                        )),
                    _ => unreachable!("unexpected distill rollout horizon index"),
                };
            }
        }
        if diagnostics.inv {
            let name = format!("{prefix}_inv_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<MetricsBackend, InvLossInput<MetricsBackend>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<MetricsBackend, InvLossInput<MetricsBackend>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                );
        }
        if diagnostics.observe {
            let name = format!("{prefix}_observe_loss");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ObserveLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ObserveLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every));
        }
        if diagnostics.mode_separation {
            let name = format!("{prefix}_mode_separation_ratio");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ModeSeparationRatioInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ModeSeparationRatioInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every));
        }
        if diagnostics.rollout_horizon_metrics {
            for (index, horizon) in VISION_ROLLOUT_HORIZON_CAPS.into_iter().enumerate() {
                let inv_name = format!("{prefix}_rollout_inv_to_h{horizon}");
                let norm_name = format!("{prefix}_rollout_state_norm_ratio_to_h{horizon}");
                let motion_name = format!("{prefix}_rollout_state_motion_to_h{horizon}");
                let long_inv_name = format!("{prefix}_long_rollout_inv_to_h{horizon}");
                let long_norm_name =
                    format!("{prefix}_long_rollout_state_norm_ratio_to_h{horizon}");
                let long_motion_name = format!("{prefix}_long_rollout_state_motion_to_h{horizon}");
                match index {
                    0 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 0>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    1 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 1>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    2 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 2>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    3 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 3>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    4 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 4>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    5 => {
                        builder = builder
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutInvToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                inv_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateNormRatioToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                norm_name.as_str(), metric_every
                            ))
                            .metric_train_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(ScalarMetric::<
                                MetricsBackend,
                                RolloutStateMotionToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                motion_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutInvToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                long_inv_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateNormRatioToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                long_norm_name.as_str(), metric_every
                            ))
                            .metric_valid_numeric(OptionalScalarMetric::<
                                MetricsBackend,
                                LongRolloutStateMotionToHorizonInput<MetricsBackend, 5>,
                            >::new_every(
                                long_motion_name.as_str(), metric_every
                            ));
                    }
                    _ => unreachable!("unexpected rollout horizon index"),
                }
            }
            let com_name = format!("{prefix}_rollout_com_error_to_h24");
            let velocity_name = format!("{prefix}_rollout_velocity_error_to_h24");
            let long_com_name = format!("{prefix}_long_rollout_com_error_to_h24");
            let long_velocity_name = format!("{prefix}_long_rollout_velocity_error_to_h24");
            builder = builder
                .metric_valid_numeric(OptionalScalarMetric::<
                    MetricsBackend,
                    RolloutComErrorToH24Input<MetricsBackend>,
                >::new_every(com_name.as_str(), metric_every))
                .metric_valid_numeric(OptionalScalarMetric::<
                    MetricsBackend,
                    RolloutVelocityErrorToH24Input<MetricsBackend>,
                >::new_every(
                    velocity_name.as_str(), metric_every
                ))
                .metric_valid_numeric(OptionalScalarMetric::<
                    MetricsBackend,
                    LongRolloutComErrorToH24Input<MetricsBackend>,
                >::new_every(
                    long_com_name.as_str(), metric_every
                ))
                .metric_valid_numeric(OptionalScalarMetric::<
                    MetricsBackend,
                    LongRolloutVelocityErrorToH24Input<MetricsBackend>,
                >::new_every(
                    long_velocity_name.as_str(), metric_every
                ));
        }
        if diagnostics.sigreg {
            let name = format!("{prefix}_sigreg_loss");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    SigRegLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    SigRegLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every));
        }
        if diagnostics.recon {
            let name = format!("{prefix}_recon_loss");
            let psnr_masked = format!("{prefix}_recon_psnr_masked");
            let psnr_full = format!("{prefix}_recon_psnr_full");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<MetricsBackend, ReconLossInput<MetricsBackend>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<MetricsBackend, ReconLossInput<MetricsBackend>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                );
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ReconPsnrMaskedInput<MetricsBackend>,
                >::new_every(
                    psnr_masked.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ReconPsnrMaskedInput<MetricsBackend>,
                >::new_every(
                    psnr_masked.as_str(), metric_every
                ));
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ReconPsnrFullInput<MetricsBackend>,
                >::new_every(
                    psnr_full.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ReconPsnrFullInput<MetricsBackend>,
                >::new_every(
                    psnr_full.as_str(), metric_every
                ));
        }
        if diagnostics.policy {
            let name = format!("{prefix}_policy_loss");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    PolicyLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    PolicyLossInput<MetricsBackend>,
                >::new_every(name.as_str(), metric_every));
            let adv_abs = format!("{prefix}_advantage_abs_mean");
            let adv_std = format!("{prefix}_advantage_std");
            let log_prob = format!("{prefix}_log_prob_mean");
            let entropy = format!("{prefix}_entropy");
            let clamp_rate = format!("{prefix}_action_clamp_rate");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    AdvantageAbsMeanInput<MetricsBackend>,
                >::new_every(adv_abs.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    AdvantageAbsMeanInput<MetricsBackend>,
                >::new_every(adv_abs.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    AdvantageStdInput<MetricsBackend>,
                >::new_every(adv_std.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    AdvantageStdInput<MetricsBackend>,
                >::new_every(adv_std.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    LogProbMeanInput<MetricsBackend>,
                >::new_every(log_prob.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    LogProbMeanInput<MetricsBackend>,
                >::new_every(log_prob.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    PolicyEntropyInput<MetricsBackend>,
                >::new_every(entropy.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    PolicyEntropyInput<MetricsBackend>,
                >::new_every(entropy.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    MetricsBackend,
                    ActionClampRateInput<MetricsBackend>,
                >::new_every(
                    clamp_rate.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    MetricsBackend,
                    ActionClampRateInput<MetricsBackend>,
                >::new_every(
                    clamp_rate.as_str(), metric_every
                ));
        }
        if diagnostics.probe {
            let probe_loss = format!("{prefix}_probe_loss");
            let probe_acc = format!("{prefix}_probe_acc");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<MetricsBackend, ProbeLossInput<MetricsBackend>>::new_every(
                        probe_loss.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<MetricsBackend, ProbeLossInput<MetricsBackend>>::new_every(
                        probe_loss.as_str(),
                        metric_every,
                    ),
                )
                .metric_train_numeric(
                    ScalarMetric::<MetricsBackend, ProbeAccInput<MetricsBackend>>::new_every(
                        probe_acc.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<MetricsBackend, ProbeAccInput<MetricsBackend>>::new_every(
                        probe_acc.as_str(),
                        metric_every,
                    ),
                );
        }

        if diagnostics.artifact_every > 0 {
            let artifact_dir = env.run_dir.join("artifacts");
            builder = builder.metric_valid(VisionArtifactMetric::<MetricsBackend>::new(
                artifact_dir,
                diagnostics.artifact_every,
                diagnostics.artifact_output,
                diagnostics.artifact_max_images,
                diagnostics.artifact_fps,
                diagnostics.normalize_mean,
                diagnostics.normalize_std,
                diagnostics.artifact_overwrite,
                diagnostics.ffmpeg_path.clone(),
            ));
        }
    }

    let learner = burn_train::Learner::new(model, optimizer, scheduler);
    let _result = builder.launch(learner);

    Ok(())
}

fn should_enable_vision_checkpoints(
    training: &VisionTrainingHyperparameters,
    backend_name: &str,
) -> bool {
    if !training.enable_checkpoints {
        return false;
    }
    if cfg!(windows) {
        let backend = backend_name.to_ascii_lowercase();
        if backend.contains("wgpu") {
            return false;
        }
    }
    true
}

pub fn resolve_vision_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &VisionDragonConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.embed_dim,
    )
}

pub fn resolve_vision_train_schedule(
    training: &VisionTrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    let epochs = match training.schedule_mode {
        Some(crate::VisionTrainScheduleMode::MaxIters) => None,
        Some(crate::VisionTrainScheduleMode::Epochs) | None => training.epochs,
    };
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        epochs,
        training.max_iters,
        steps_per_epoch,
        "vision training",
    )
}

pub fn resolve_vision_rollout(
    training: &VisionTrainingHyperparameters,
    max_steps: usize,
) -> Result<VisionRollout> {
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
    Ok(VisionRollout {
        min_steps,
        max_steps: max_steps_cfg,
        backprop_steps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vision_checkpoints_disabled_when_flag_off() {
        let training = VisionTrainingHyperparameters {
            enable_checkpoints: false,
            ..Default::default()
        };
        assert!(!should_enable_vision_checkpoints(&training, "wgpu"));
        assert!(!should_enable_vision_checkpoints(&training, "ndarray"));
    }

    #[test]
    fn vision_checkpoints_guard_wgpu_on_windows() {
        let training = VisionTrainingHyperparameters::default();
        let enabled = should_enable_vision_checkpoints(&training, "wgpu");
        if cfg!(windows) {
            assert!(!enabled);
        } else {
            assert!(enabled);
        }
    }
}
