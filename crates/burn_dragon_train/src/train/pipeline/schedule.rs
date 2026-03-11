use crate::train::prelude::*;

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

pub fn resolve_valid_steps_per_epoch(
    total_steps: usize,
    log_frequency: usize,
    val_steps_per_epoch: usize,
) -> usize {
    let desired_valid_steps = usize::max(1, total_steps / log_frequency.max(1));
    desired_valid_steps.min(val_steps_per_epoch.max(1)).max(1)
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    default_model_size: usize,
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
            config = config.with_model_size(model_size.unwrap_or(default_model_size).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

pub fn resolve_train_schedule(
    epochs: Option<usize>,
    max_iters: usize,
    steps_per_epoch: usize,
    label: &str,
) -> Result<TrainSchedule> {
    let steps_per_epoch = steps_per_epoch.max(1);
    match epochs {
        Some(epochs) => {
            let total_epochs = epochs.max(1);
            let total_steps = steps_per_epoch
                .checked_mul(total_epochs)
                .ok_or_else(|| {
                    anyhow!(
                        "{label}.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
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
            let total_steps = max_iters.max(1);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn optimizer(
        learning_rate: f64,
        lr_schedule: Option<LearningRateScheduleConfig>,
    ) -> OptimizerConfig {
        OptimizerConfig {
            learning_rate,
            weight_decay: 0.0,
            lr_schedule,
            grad_clip_norm: None,
            grad_clip_value: None,
        }
    }

    #[test]
    fn resolve_train_schedule_prefers_epochs_when_configured() {
        let schedule = resolve_train_schedule(Some(3), 99, 10, "training").expect("schedule");

        assert_eq!(schedule.steps_per_epoch, 10);
        assert_eq!(schedule.total_steps, 30);
        assert_eq!(schedule.total_epochs, 3);
        assert_eq!(schedule.source, ScheduleSource::Epochs);
    }

    #[test]
    fn resolve_train_schedule_uses_max_iters_when_epochs_missing() {
        let schedule = resolve_train_schedule(None, 25, 8, "training").expect("schedule");

        assert_eq!(schedule.steps_per_epoch, 8);
        assert_eq!(schedule.total_steps, 25);
        assert_eq!(schedule.total_epochs, 4);
        assert_eq!(schedule.source, ScheduleSource::MaxIters);
    }

    #[test]
    fn resolve_train_schedule_rejects_overflow() {
        let result = resolve_train_schedule(Some(2), 1, usize::MAX, "training");

        assert!(result.is_err());
        let err = result.expect_err("overflow should fail");
        assert!(err.to_string().contains("training.epochs overflow"));
    }

    #[test]
    fn resolve_lr_scheduler_returns_expected_variants() {
        let constant =
            resolve_lr_scheduler(&optimizer(1e-3, None), 100, None, 64).expect("constant schedule");
        match constant {
            ResolvedLrScheduler::Constant(lr) => assert_eq!(lr, 1e-3),
            _ => panic!("expected constant scheduler"),
        }

        let step = resolve_lr_scheduler(
            &optimizer(
                2e-3,
                Some(LearningRateScheduleConfig::Step {
                    initial_lr: Some(3e-3),
                    gamma: 0.5,
                    step_size: Some(12),
                }),
            ),
            100,
            None,
            64,
        )
        .expect("step schedule");
        assert!(matches!(step, ResolvedLrScheduler::Step(_)));

        let noam = resolve_lr_scheduler(
            &optimizer(
                2e-3,
                Some(LearningRateScheduleConfig::Noam {
                    initial_lr: None,
                    warmup_steps: Some(16),
                    model_size: None,
                }),
            ),
            200,
            None,
            256,
        )
        .expect("noam schedule");
        assert!(matches!(noam, ResolvedLrScheduler::Noam(_)));
    }

    #[test]
    fn resolve_valid_steps_per_epoch_is_bounded() {
        assert_eq!(resolve_valid_steps_per_epoch(100, 10, 20), 10);
        assert_eq!(resolve_valid_steps_per_epoch(100, 1, 5), 5);
        assert_eq!(resolve_valid_steps_per_epoch(3, 100, 0), 1);
    }
}
