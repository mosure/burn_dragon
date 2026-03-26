use crate::OptimizerScheduleMode;
use crate::train::prelude::*;

pub enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(CosineAnnealingLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
    BitNetTwoStage(BitNetTwoStageLrScheduler),
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

const DEFAULT_BITNET_FINAL_LR_RATIO: f64 = 2.0 / 3.0;
const DEFAULT_BITNET_SECOND_STAGE_START: f32 = 0.5;

#[derive(Clone, Copy, Debug, PartialEq)]
struct BitNetTwoStageProfile {
    peak_lr: LearningRate,
    final_lr: LearningRate,
    warmup_steps: usize,
    second_stage_start: f32,
    total_steps: usize,
}

#[derive(Clone, Debug)]
pub struct BitNetTwoStageLrScheduler {
    peak_lr: LearningRate,
    final_lr: LearningRate,
    warmup_steps: usize,
    second_stage_start_step: usize,
    total_steps: usize,
    current_step: usize,
}

#[derive(Record, Clone, Debug)]
pub struct BitNetTwoStageLrSchedulerRecord {
    peak_lr: LearningRate,
    final_lr: LearningRate,
    warmup_steps: usize,
    second_stage_start_step: usize,
    total_steps: usize,
    current_step: usize,
}

impl BitNetTwoStageLrScheduler {
    fn from_profile(profile: BitNetTwoStageProfile) -> Self {
        let total_steps = profile.total_steps.max(1);
        let last_step_index = total_steps.saturating_sub(1);
        let stage_start_index = ((last_step_index as f32)
            * profile.second_stage_start.clamp(0.0, 1.0))
        .round() as usize;
        let second_stage_start_step = stage_start_index.saturating_add(1).min(total_steps);

        Self {
            peak_lr: profile.peak_lr,
            final_lr: profile.final_lr,
            warmup_steps: profile.warmup_steps.max(1).min(total_steps),
            second_stage_start_step: second_stage_start_step.max(1),
            total_steps,
            current_step: 0,
        }
    }
}

impl LrScheduler for BitNetTwoStageLrScheduler {
    type Record<B: BackendTrait> = BitNetTwoStageLrSchedulerRecord;

    fn step(&mut self) -> LearningRate {
        self.current_step = self.current_step.saturating_add(1).min(self.total_steps);

        if self.current_step <= self.warmup_steps {
            return self.peak_lr * (self.current_step as f64 / self.warmup_steps as f64);
        }

        if self.current_step <= self.second_stage_start_step {
            return self.peak_lr;
        }

        if self.current_step >= self.total_steps {
            return self.final_lr;
        }

        let decay_span = self
            .total_steps
            .saturating_sub(self.second_stage_start_step)
            .max(1);
        let decay_progress = self
            .current_step
            .saturating_sub(self.second_stage_start_step) as f64
            / decay_span as f64;
        self.peak_lr + (self.final_lr - self.peak_lr) * decay_progress
    }

    fn to_record<B: BackendTrait>(&self) -> Self::Record<B> {
        BitNetTwoStageLrSchedulerRecord {
            peak_lr: self.peak_lr,
            final_lr: self.final_lr,
            warmup_steps: self.warmup_steps,
            second_stage_start_step: self.second_stage_start_step,
            total_steps: self.total_steps,
            current_step: self.current_step,
        }
    }

    fn load_record<B: BackendTrait>(self, record: Self::Record<B>) -> Self {
        Self {
            peak_lr: record.peak_lr,
            final_lr: record.final_lr,
            warmup_steps: record.warmup_steps,
            second_stage_start_step: record.second_stage_start_step,
            total_steps: record.total_steps,
            current_step: record.current_step,
        }
    }
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
        None => match optimizer_cfg.schedule_mode {
            OptimizerScheduleMode::BdhReference => ResolvedLrScheduler::Constant(base_lr),
            OptimizerScheduleMode::BitnetB158Reference | OptimizerScheduleMode::Hybrid => {
                let profile = resolve_bitnet_two_stage_profile(optimizer_cfg, fallback_iters);
                ResolvedLrScheduler::BitNetTwoStage(BitNetTwoStageLrScheduler::from_profile(
                    profile,
                ))
            }
        },
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

fn resolve_bitnet_two_stage_profile(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
) -> BitNetTwoStageProfile {
    let total_steps = total_steps.max(1);
    let warmup_steps = total_steps.div_ceil(20).min(375).max(1);

    BitNetTwoStageProfile {
        peak_lr: optimizer_cfg.learning_rate,
        final_lr: optimizer_cfg.learning_rate * DEFAULT_BITNET_FINAL_LR_RATIO,
        warmup_steps,
        second_stage_start: DEFAULT_BITNET_SECOND_STAGE_START,
        total_steps,
    }
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
    use crate::{OptimizerKind, OptimizerScheduleMode};

    fn optimizer(
        learning_rate: f64,
        lr_schedule: Option<LearningRateScheduleConfig>,
    ) -> OptimizerConfig {
        OptimizerConfig {
            name: OptimizerKind::default(),
            learning_rate,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule,
            schedule_mode: OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
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
    fn resolve_lr_scheduler_uses_bitnet_two_stage_modes_when_explicit_schedule_missing() {
        let mut config = optimizer(1.2e-3, None);
        config.schedule_mode = OptimizerScheduleMode::BitnetB158Reference;
        let bitnet = resolve_lr_scheduler(&config, 100, None, 64).expect("bitnet schedule");
        assert!(matches!(bitnet, ResolvedLrScheduler::BitNetTwoStage(_)));

        config.schedule_mode = OptimizerScheduleMode::Hybrid;
        let hybrid = resolve_lr_scheduler(&config, 100, None, 64).expect("hybrid schedule");
        assert!(matches!(hybrid, ResolvedLrScheduler::BitNetTwoStage(_)));
    }

    #[test]
    fn bitnet_two_stage_scheduler_warms_up_then_decays() {
        let mut scheduler = BitNetTwoStageLrScheduler::from_profile(BitNetTwoStageProfile {
            peak_lr: 1.2e-3,
            final_lr: 8.0e-4,
            warmup_steps: 2,
            second_stage_start: 0.5,
            total_steps: 6,
        });

        let step1 = scheduler.step();
        let step2 = scheduler.step();
        let step3 = scheduler.step();
        let step4 = scheduler.step();
        let step5 = scheduler.step();
        let step6 = scheduler.step();

        assert!((step1 - 6.0e-4).abs() < 1.0e-12);
        assert!((step2 - 1.2e-3).abs() < 1.0e-12);
        assert!((step3 - 1.2e-3).abs() < 1.0e-12);
        assert!((step4 - 1.2e-3).abs() < 1.0e-12);
        assert!(step5 < step4);
        assert!((step6 - 8.0e-4).abs() < 1.0e-12);
    }

    #[test]
    fn bitnet_two_stage_scheduler_round_trips_record_state() {
        let scheduler = BitNetTwoStageLrScheduler::from_profile(BitNetTwoStageProfile {
            peak_lr: 1.2e-3,
            final_lr: 8.0e-4,
            warmup_steps: 3,
            second_stage_start: 0.5,
            total_steps: 12,
        });
        let mut truth = scheduler.clone();
        let mut persisted = scheduler.clone();
        for _ in 0..4 {
            truth.step();
            persisted.step();
        }
        let record = persisted.to_record::<burn_ndarray::NdArray<f32>>();
        let mut restored = persisted.load_record::<burn_ndarray::NdArray<f32>>(record);
        for _ in 0..8 {
            let expected = truth.step();
            let actual = restored.step();
            assert!((expected - actual).abs() < 1.0e-12);
        }
    }

    #[test]
    fn resolve_valid_steps_per_epoch_is_bounded() {
        assert_eq!(resolve_valid_steps_per_epoch(100, 10, 20), 10);
        assert_eq!(resolve_valid_steps_per_epoch(100, 1, 5), 5);
        assert_eq!(resolve_valid_steps_per_epoch(3, 100, 0), 1);
    }
}
