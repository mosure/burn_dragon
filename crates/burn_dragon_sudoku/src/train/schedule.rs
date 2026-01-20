use crate::train::metrics::{
    SudokuAccInput, SudokuAdvantageAbsMeanInput, SudokuAdvantageStdInput, SudokuEasyRewardInput,
    SudokuHaltLossInput, SudokuHaltProbInput, SudokuHaltTargetInput, SudokuHardRewardInput,
    SudokuLogProbMeanInput, SudokuPolicyEntropyInput, SudokuPolicyLossInput, SudokuReconLossInput,
};
use crate::train::prelude::*;
use crate::train::steps::SudokuTrainer;

pub struct SudokuTrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a SudokuTrainingHyperparameters,
    pub device: &'a B::Device,
    pub train_loader: Arc<dyn DataLoader<B, SudokuBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SudokuBatch<ValidBackend<B>>>>,
    pub epochs: usize,
}

pub fn train_with_scheduler<B, S>(
    env: &SudokuTrainEnvironment<'_, B>,
    model: SudokuTrainer<B>,
    optimizer: OptimizerAdaptor<AdamW, SudokuTrainer<B>, B>,
    scheduler: S,
) -> Result<SudokuTrainer<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let metric_every = env.training.log_frequency.max(1);
    let builder = LearnerBuilder::new(env.run_dir)
        .num_epochs(env.epochs)
        .learning_strategy(LearningStrategy::SingleDevice(env.device.clone()))
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(
            ScalarMetric::<ValidBackend<B>, LossValue<ValidBackend<B>>>::new_every(
                "Loss",
                metric_every,
            ),
        )
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuReconLossInput<ValidBackend<B>>,
        >::new_every("sudoku_recon_loss", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuReconLossInput<ValidBackend<B>>,
        >::new_every("sudoku_recon_loss", metric_every))
        .metric_train_numeric(ScalarMetric::<ValidBackend<B>, SudokuAccInput<ValidBackend<B>>>::new_every(
            "sudoku_acc",
            metric_every,
        ))
        .metric_valid_numeric(ScalarMetric::<ValidBackend<B>, SudokuAccInput<ValidBackend<B>>>::new_every(
            "sudoku_acc",
            metric_every,
        ))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuPolicyLossInput<ValidBackend<B>>,
        >::new_every("sudoku_policy_loss", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuPolicyLossInput<ValidBackend<B>>,
        >::new_every("sudoku_policy_loss", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltLossInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_loss", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltLossInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_loss", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltProbInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_prob", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltProbInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_prob", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltTargetInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_target", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHaltTargetInput<ValidBackend<B>>,
        >::new_every("sudoku_halt_target", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuAdvantageAbsMeanInput<ValidBackend<B>>,
        >::new_every("sudoku_adv_abs_mean", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuAdvantageAbsMeanInput<ValidBackend<B>>,
        >::new_every("sudoku_adv_abs_mean", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuAdvantageStdInput<ValidBackend<B>>,
        >::new_every("sudoku_adv_std", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuAdvantageStdInput<ValidBackend<B>>,
        >::new_every("sudoku_adv_std", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuLogProbMeanInput<ValidBackend<B>>,
        >::new_every("sudoku_log_prob_mean", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuLogProbMeanInput<ValidBackend<B>>,
        >::new_every("sudoku_log_prob_mean", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuPolicyEntropyInput<ValidBackend<B>>,
        >::new_every("sudoku_entropy", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuPolicyEntropyInput<ValidBackend<B>>,
        >::new_every("sudoku_entropy", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHardRewardInput<ValidBackend<B>>,
        >::new_every("sudoku_hard_reward", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuHardRewardInput<ValidBackend<B>>,
        >::new_every("sudoku_hard_reward", metric_every))
        .metric_train_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuEasyRewardInput<ValidBackend<B>>,
        >::new_every("sudoku_easy_reward", metric_every))
        .metric_valid_numeric(ScalarMetric::<
            ValidBackend<B>,
            SudokuEasyRewardInput<ValidBackend<B>>,
        >::new_every("sudoku_easy_reward", metric_every))
        .summary();

    info!("sudoku run name: {}", env.run_name);

    #[cfg(feature = "integration_test")]
    let builder = builder.metric_train(
        crate::train::metrics::LossTraceMetric::<ValidBackend<B>>::new("loss_trace", 1),
    );

    let learner = builder.build(model, optimizer, scheduler);
    let TrainingResult { model, .. } =
        learner.fit(Arc::clone(&env.train_loader), Arc::clone(&env.valid_loader));

    Ok(model)
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &SudokuModelConfig,
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
            config = config.with_model_size(model_size.unwrap_or(model_config.n_embd).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

pub fn resolve_train_schedule(
    training: &SudokuTrainingHyperparameters,
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
                        "training.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
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
