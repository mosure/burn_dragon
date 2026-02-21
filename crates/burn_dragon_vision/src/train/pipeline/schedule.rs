use crate::train::prelude::*;
use burn_train::metric::IterationSpeedMetric;

pub enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(CosineAnnealingLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScheduleSource {
    Epochs,
    MaxIters,
}

impl ScheduleSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ScheduleSource::Epochs => "epochs",
            ScheduleSource::MaxIters => "max_iters",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrainSchedule {
    pub steps_per_epoch: usize,
    pub total_steps: usize,
    pub total_epochs: usize,
    pub source: ScheduleSource,
}

pub struct VisionTrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a VisionTrainingHyperparameters,
    pub device: &'a B::Device,
    pub train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>>,
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
    pub inv: bool,
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

pub fn train_vision_with_scheduler<B, S, M>(
    env: &VisionTrainEnvironment<'_, B>,
    model: M,
    optimizer: OptimizerAdaptor<AdamW, M, B>,
    scheduler: S,
    vision_diagnostics: Option<VisionDiagnostics>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    M: AutodiffModule<B>
        + TrainStep<ImageNetBatch<B>, VisionTrainItem<B>>
        + core::fmt::Display
        + Clone
        + 'static,
    M::InnerModule: ValidStep<ImageNetBatch<ValidBackend<B>>, VisionOutput<ValidBackend<B>>>,
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
    let mut builder = LearnerBuilder::new(env.run_dir)
        .num_epochs(env.epochs)
        .learning_strategy(LearningStrategy::SingleDevice(env.device.clone()));
    if enable_checkpoints {
        builder = builder.with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new());
    }
    builder = builder
        .metric_train_numeric(IterationSpeedMetric::new())
        .metric_train_numeric(
            ScalarMetric::<ValidBackend<B>, LossValue<ValidBackend<B>>>::new_every(
                "Loss", loss_every,
            ),
        )
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name))
        .summary();

    info!("vision run name: {}", env.run_name);

    #[cfg(feature = "integration_test")]
    if env.training.trace_train_loss {
        builder = builder.metric_train(burn_dragon_train::train::metrics::LossTraceMetric::<
            ValidBackend<B>,
        >::new(
            "loss_trace", env.training.trace_train_loss_every
        ));
    }

    let cleanup_iters = env.training.memory_cleanup_iters;
    if env.training.memory_cleanup_every > 0 || cleanup_iters > 0 {
        let allow_cuda_cleanup = !env.training.disable_cuda_memory_cleanup;
        builder = builder
            .metric_train(MemoryCleanupMetric::<B>::new(
                env.device,
                env.training.memory_cleanup_every,
                cleanup_iters,
                allow_cuda_cleanup,
            ))
            .metric_valid(MemoryCleanupMetric::<ValidBackend<B>>::new(
                env.device,
                env.training.memory_cleanup_every,
                cleanup_iters,
                allow_cuda_cleanup,
            ));
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
        if diagnostics.inv {
            let name = format!("{prefix}_inv_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new_every(
                        name.as_str(),
                        metric_every,
                    ),
                );
        }
        if diagnostics.sigreg {
            let name = format!("{prefix}_sigreg_loss");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    SigRegLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    SigRegLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every));
        }
        if diagnostics.recon {
            let name = format!("{prefix}_recon_loss");
            let psnr_masked = format!("{prefix}_recon_psnr_masked");
            let psnr_full = format!("{prefix}_recon_psnr_full");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every));
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconPsnrMaskedInput<ValidBackend<B>>,
                >::new_every(
                    psnr_masked.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconPsnrMaskedInput<ValidBackend<B>>,
                >::new_every(
                    psnr_masked.as_str(), metric_every
                ));
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconPsnrFullInput<ValidBackend<B>>,
                >::new_every(
                    psnr_full.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ReconPsnrFullInput<ValidBackend<B>>,
                >::new_every(
                    psnr_full.as_str(), metric_every
                ));
        }
        if diagnostics.policy {
            let name = format!("{prefix}_policy_loss");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    PolicyLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    PolicyLossInput<ValidBackend<B>>,
                >::new_every(name.as_str(), metric_every));
            let adv_abs = format!("{prefix}_advantage_abs_mean");
            let adv_std = format!("{prefix}_advantage_std");
            let log_prob = format!("{prefix}_log_prob_mean");
            let entropy = format!("{prefix}_entropy");
            let clamp_rate = format!("{prefix}_action_clamp_rate");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    AdvantageAbsMeanInput<ValidBackend<B>>,
                >::new_every(adv_abs.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    AdvantageAbsMeanInput<ValidBackend<B>>,
                >::new_every(adv_abs.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    AdvantageStdInput<ValidBackend<B>>,
                >::new_every(adv_std.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    AdvantageStdInput<ValidBackend<B>>,
                >::new_every(adv_std.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    LogProbMeanInput<ValidBackend<B>>,
                >::new_every(log_prob.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    LogProbMeanInput<ValidBackend<B>>,
                >::new_every(log_prob.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    PolicyEntropyInput<ValidBackend<B>>,
                >::new_every(entropy.as_str(), metric_every))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    PolicyEntropyInput<ValidBackend<B>>,
                >::new_every(entropy.as_str(), metric_every))
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ActionClampRateInput<ValidBackend<B>>,
                >::new_every(
                    clamp_rate.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ActionClampRateInput<ValidBackend<B>>,
                >::new_every(
                    clamp_rate.as_str(), metric_every
                ));
        }
        if diagnostics.probe {
            let probe_loss = format!("{prefix}_probe_loss");
            let probe_acc = format!("{prefix}_probe_acc");
            builder = builder
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ProbeLossInput<ValidBackend<B>>,
                >::new_every(
                    probe_loss.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ProbeLossInput<ValidBackend<B>>,
                >::new_every(
                    probe_loss.as_str(), metric_every
                ))
                .metric_train_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ProbeAccInput<ValidBackend<B>>,
                >::new_every(
                    probe_acc.as_str(), metric_every
                ))
                .metric_valid_numeric(ScalarMetric::<
                    ValidBackend<B>,
                    ProbeAccInput<ValidBackend<B>>,
                >::new_every(
                    probe_acc.as_str(), metric_every
                ));
        }

        if diagnostics.artifact_every > 0 {
            let artifact_dir = env.run_dir.join("artifacts");
            builder = builder.metric_valid(VisionArtifactMetric::<ValidBackend<B>>::new(
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

    let learner = builder.build(model, optimizer, scheduler);

    let _result = learner.fit(Arc::clone(&env.train_loader), Arc::clone(&env.valid_loader));

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
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = total_steps.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => ResolvedLrScheduler::Constant(base_lr),
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = CosineAnnealingLrSchedulerConfig::new(
                init_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .with_min_lr(min_lr.unwrap_or(0.0))
            .init()
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(model_config.embed_dim).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

pub fn resolve_vision_train_schedule(
    training: &VisionTrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    let steps_per_epoch = steps_per_epoch.max(1);
    match training.epochs {
        Some(epochs) => {
            let total_epochs = epochs.max(1);
            let total_steps = steps_per_epoch
                .checked_mul(total_epochs)
                .ok_or_else(|| {
                    anyhow!(
                        "vision training.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
                    )
                })?
                .max(1);
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::Epochs,
            })
        }
        None => {
            let total_steps = training.max_iters.max(1);
            let total_epochs = usize::max(1, total_steps.div_ceil(steps_per_epoch));
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::MaxIters,
            })
        }
    }
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
