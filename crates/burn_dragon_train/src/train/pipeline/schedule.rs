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
