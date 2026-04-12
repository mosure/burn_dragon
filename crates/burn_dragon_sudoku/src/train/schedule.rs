use crate::train::metrics::{
    SudokuAccInput, SudokuAdvantageAbsMeanInput, SudokuAdvantageStdInput, SudokuEasyRewardInput,
    SudokuExactAccInput, SudokuHaltLossInput, SudokuHaltProbInput, SudokuHaltTargetInput,
    SudokuHardRewardInput, SudokuLogProbMeanInput, SudokuPolicyEntropyAlphaInput,
    SudokuPolicyEntropyInput, SudokuPolicyEntropyTargetInput, SudokuPolicyLossInput,
    SudokuReconLossInput, SudokuSaccadeRepeatRateInput, SudokuSaccadeRevisitRateInput,
    SudokuSaccadeUniqueFracInput, SudokuSaccadeUnknownFracInput, SudokuShapingAccuracyInput,
    SudokuShapingConflictInput, SudokuShapingIncorrectInput, SudokuShapingUnknownInput,
    SudokuSolveRateInput, SudokuWriteGateMeanInput, SudokuWriteRateInput,
};
use crate::train::prelude::*;
use crate::train::steps::SudokuTrainer;
use burn_train::metric::IterationSpeedMetric;

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
    let gdpo_active = env.training.gdpo.enabled;
    let entropy_active =
        env.training.policy.entropy_adaptive || env.training.policy.entropy_weight > 0.0;
    let halt_active = env.training.halt.weight > 0.0;

    let mut builder = SupervisedTraining::new(
        env.run_dir,
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    )
    .num_epochs(env.epochs)
    .with_training_strategy(LearningStrategy::Default(ExecutionStrategy::single(
        env.device.clone(),
    )))
    .with_application_logger(None)
    .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
    .metric_train_numeric(IterationSpeedMetric::new())
    .metric_train_numeric(
        ScalarMetric::<ValidBackend<B>, LossValue<ValidBackend<B>>>::new_every(
            "Loss",
            metric_every,
        ),
    )
    .metric_valid_numeric(
        ScalarMetric::<ValidBackend<B>, LossValue<ValidBackend<B>>>::new_every(
            "Loss",
            metric_every,
        ),
    )
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
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuAccInput<ValidBackend<B>>,
    >::new_every("sudoku_acc", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuAccInput<ValidBackend<B>>,
    >::new_every("sudoku_acc", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuExactAccInput<ValidBackend<B>>,
    >::new_every("sudoku_exact_acc", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuExactAccInput<ValidBackend<B>>,
    >::new_every("sudoku_exact_acc", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSolveRateInput<ValidBackend<B>>,
    >::new_every("sudoku_solve_rate", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSolveRateInput<ValidBackend<B>>,
    >::new_every("sudoku_solve_rate", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeRevisitRateInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_revisit_rate", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeRevisitRateInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_revisit_rate", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeRepeatRateInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_repeat_rate", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeRepeatRateInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_repeat_rate", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeUnknownFracInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_unknown_frac", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeUnknownFracInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_unknown_frac", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeUniqueFracInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_unique_frac", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuSaccadeUniqueFracInput<ValidBackend<B>>,
    >::new_every("sudoku_saccade_unique_frac", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuWriteGateMeanInput<ValidBackend<B>>,
    >::new_every("sudoku_write_gate_mean", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuWriteGateMeanInput<ValidBackend<B>>,
    >::new_every("sudoku_write_gate_mean", metric_every))
    .metric_train_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuWriteRateInput<ValidBackend<B>>,
    >::new_every("sudoku_write_rate", metric_every))
    .metric_valid_numeric(ScalarMetric::<
        ValidBackend<B>,
        SudokuWriteRateInput<ValidBackend<B>>,
    >::new_every("sudoku_write_rate", metric_every));

    if gdpo_active {
        builder = builder
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuPolicyLossInput<ValidBackend<B>>,
            >::new_every("sudoku_policy_loss", metric_every))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuAdvantageAbsMeanInput<ValidBackend<B>>,
            >::new_every("sudoku_adv_abs_mean", metric_every))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuAdvantageStdInput<ValidBackend<B>>,
            >::new_every("sudoku_adv_std", metric_every))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuLogProbMeanInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_log_prob_mean", metric_every
            ))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuHardRewardInput<ValidBackend<B>>,
            >::new_every("sudoku_hard_reward", metric_every))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuEasyRewardInput<ValidBackend<B>>,
            >::new_every("sudoku_easy_reward", metric_every));
        builder = builder
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuShapingConflictInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_shaping_conflict", metric_every
            ))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuShapingUnknownInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_shaping_unknown", metric_every
            ))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuShapingAccuracyInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_shaping_accuracy", metric_every
            ))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuShapingIncorrectInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_shaping_incorrect", metric_every
            ));
    }

    if halt_active {
        builder = builder
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
            >::new_every("sudoku_halt_target", metric_every));
    }

    if entropy_active {
        builder = builder
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuPolicyEntropyInput<ValidBackend<B>>,
            >::new_every("sudoku_entropy", metric_every))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuPolicyEntropyAlphaInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_entropy_alpha", metric_every
            ))
            .metric_train_numeric(ScalarMetric::<
                ValidBackend<B>,
                SudokuPolicyEntropyTargetInput<ValidBackend<B>>,
            >::new_every(
                "sudoku_entropy_target", metric_every
            ));
    }

    info!("sudoku run name: {}", env.run_name);

    #[cfg(feature = "integration_test")]
    let builder = builder
        .metric_train(
            crate::train::metrics::LossTraceMetric::<ValidBackend<B>>::new("loss_trace", 1),
        )
        .metric_valid(
            crate::train::metrics::SolveRateTraceMetric::<ValidBackend<B>>::new(
                "solve_rate_trace",
                1,
            ),
        )
        .metric_valid(
            crate::train::metrics::HaltProbTraceMetric::<ValidBackend<B>>::new(
                "halt_prob_trace",
                1,
            ),
        );

    let builder = builder.summary();

    let learner = burn_train::Learner::new(model, optimizer, scheduler);
    let TrainingResult { model, .. } = builder.launch(learner);

    Ok(model)
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &SudokuModelConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.n_embd,
    )
}

pub fn resolve_train_schedule(
    training: &SudokuTrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        training.epochs,
        training.max_iters,
        steps_per_epoch,
        "training",
    )
}
