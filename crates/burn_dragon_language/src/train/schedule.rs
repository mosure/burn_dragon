use crate::train::prelude::*;
use crate::train::utils::log_theoretical_profile;

pub struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a TrainingHyperparameters,
    pub model_config: &'a BDHConfig,
    pub device: &'a B::Device,
    pub train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    pub epochs: usize,
}

pub fn train_with_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: BDH<B>,
    optimizer: OptimizerAdaptor<AdamW, BDH<B>, B>,
    scheduler: S,
) -> Result<BDH<ValidBackend<B>>>
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
        .summary();

    info!("run name: {}", env.run_name);

    let learner = builder.build(model, optimizer, scheduler);

    let TrainingResult { model, .. } =
        learner.fit(Arc::clone(&env.train_loader), Arc::clone(&env.valid_loader));

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model)
}

pub fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &BDHConfig,
) -> Result<ResolvedLrScheduler> {
    burn_dragon_train::train::pipeline::resolve_lr_scheduler(
        optimizer_cfg,
        total_steps,
        override_num_iters,
        model_config.n_embd,
    )
}

pub fn resolve_train_schedule(
    training: &TrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    burn_dragon_train::train::pipeline::resolve_train_schedule(
        training.epochs,
        training.max_iters,
        steps_per_epoch,
        "training",
    )
}
