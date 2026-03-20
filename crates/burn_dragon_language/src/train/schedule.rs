use crate::train::prelude::*;
use crate::train::utils::log_theoretical_profile;
#[cfg(feature = "ddp")]
use burn::tensor::TensorPrimitive;
#[cfg(feature = "ddp")]
use std::marker::PhantomData;

const CHECKPOINT_KEEP_LAST: usize = 2;

pub struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    pub parallel_runtime: &'a ParallelRuntime,
    pub parallel_config: &'a ParallelConfig,
    pub run_dir: &'a Path,
    pub run_name: &'a str,
    pub backend_name: &'a str,
    pub training: &'a TrainingHyperparameters,
    pub resume_checkpoint_epoch: Option<usize>,
    pub model_config: &'a BDHConfig,
    pub device: &'a B::Device,
    pub devices: &'a [B::Device],
    pub train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    pub valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    pub epochs: usize,
}

pub(crate) fn train_with_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
    scheduler: S,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let metric_every = env.training.log_frequency.max(1);
    #[cfg(feature = "ddp")]
    if env.parallel_runtime.mode == ParallelismKind::Ddp
        && env.parallel_runtime.is_process_group_launch()
    {
        return train_with_process_group_scheduler(env, model, optimizer, scheduler);
    }
    let training_strategy = match env.parallel_runtime.mode {
        ParallelismKind::Single => LearningStrategy::SingleDevice(env.device.clone()),
        ParallelismKind::Ddp => {
            #[cfg(feature = "ddp")]
            {
                LearningStrategy::DistributedDataParallel {
                    devices: env.devices.to_vec(),
                    config: resolve_collective_config(env.parallel_runtime, env.parallel_config)?,
                }
            }
            #[cfg(not(feature = "ddp"))]
            {
                LearningStrategy::MultiDevice(
                    env.devices.to_vec(),
                    MultiDeviceOptim::OptimMainDevice,
                )
            }
        }
        mode => {
            return Err(anyhow!(
                "parallel.mode={mode:?} is not wired into language training yet"
            ));
        }
    };
    let builder = SupervisedTraining::new(
        env.run_dir,
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    )
    .num_epochs(env.epochs)
    .grads_accumulation(env.training.gradient_accumulation_steps.max(1))
    .with_training_strategy(training_strategy)
    .with_application_logger(None)
    .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
    .with_checkpointing_strategy(
        burn_train::checkpoint::ComposedCheckpointingStrategy::builder()
            .add(burn_train::checkpoint::KeepLastNCheckpoints::new(
                CHECKPOINT_KEEP_LAST,
            ))
            .add(burn_train::checkpoint::MetricCheckpointingStrategy::new(
                &LossMetric::<ValidBackend<B>>::new(),
                burn_train::metric::store::Aggregate::Mean,
                burn_train::metric::store::Direction::Lowest,
                burn_train::metric::store::Split::Valid,
            ))
            .build(),
    )
    .metric_train_numeric(
        ScalarMetric::<ValidBackend<B>, LossValue<ValidBackend<B>>>::new_every(
            "Loss",
            metric_every,
        ),
    )
    .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
    .metric_train_numeric(LearningRateMetric::new())
    .metric_train(DeviceMetric::new("device", env.backend_name))
    .metric_valid(DeviceMetric::new("device", env.backend_name));
    #[cfg(feature = "rerun")]
    let builder = crate::train::rerun::attach_metric_loggers(builder, env.run_dir);
    let builder = builder.summary();
    let builder = match env.resume_checkpoint_epoch {
        Some(checkpoint) => builder.checkpoint(checkpoint),
        None => builder,
    };

    info!("run name: {}", env.run_name);
    info!(
        "training strategy: mode={:?} replicas={}",
        env.parallel_runtime.mode,
        env.devices.len()
    );
    info!(
        "checkpoint policy: logical_epoch_steps={} keep_last={} keep_best_valid_loss=true",
        env.train_loader.num_items(),
        CHECKPOINT_KEEP_LAST
    );

    let learner = burn_train::Learner::new(model, optimizer, scheduler);
    let TrainingResult { model, .. } = builder.launch(learner);

    log_theoretical_profile(
        env.model_config,
        env.training
            .batch_size
            .saturating_mul(env.training.gradient_accumulation_steps.max(1)),
        env.training.block_size,
        env.backend_name,
    );

    Ok(model.model)
}

#[cfg(feature = "ddp")]
struct CollectiveSessionGuard<B: BackendTrait> {
    peer_id: PeerId,
    _marker: PhantomData<B>,
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> CollectiveSessionGuard<B> {
    fn register(
        peer_id: PeerId,
        device: B::Device,
        config: burn_collective::CollectiveConfig,
    ) -> Result<Self> {
        info!("registering DDP collective session for peer_id={peer_id}");
        register::<B>(peer_id, device, config)
            .map_err(|err| anyhow!("failed to register DDP collective session: {err:?}"))?;
        info!("registered DDP collective session for peer_id={peer_id}");
        Ok(Self {
            peer_id,
            _marker: PhantomData,
        })
    }
}

#[cfg(feature = "ddp")]
impl<B: BackendTrait> Drop for CollectiveSessionGuard<B> {
    fn drop(&mut self) {
        let _ = finish_collective::<B>(self.peer_id);
    }
}

#[cfg(feature = "ddp")]
fn shard_bounds(
    total_items: usize,
    shard_index: usize,
    shard_count: usize,
) -> Result<(usize, usize)> {
    if shard_count == 0 {
        return Err(anyhow!("cannot shard a dataloader across zero ranks"));
    }
    if shard_index >= shard_count {
        return Err(anyhow!(
            "rank-local dataloader shard {shard_index} is out of range for shard_count={shard_count}"
        ));
    }
    if total_items < shard_count {
        return Err(anyhow!(
            "rank-local dataloader sharding requires at least one step per rank (steps={total_items}, world_size={shard_count})"
        ));
    }

    let base = total_items / shard_count;
    let remainder = total_items % shard_count;
    let start = shard_index * base + shard_index.min(remainder);
    let width = base + usize::from(shard_index < remainder);
    Ok((start, start + width))
}

#[cfg(feature = "ddp")]
fn shard_dataloader<B, I>(
    loader: Arc<dyn DataLoader<B, I>>,
    shard_index: usize,
    shard_count: usize,
    label: &str,
) -> Result<Arc<dyn DataLoader<B, I>>>
where
    B: BackendTrait + 'static,
    I: 'static,
{
    if shard_count <= 1 {
        return Ok(loader);
    }

    let total_items = loader.num_items();
    let (start, end) = shard_bounds(total_items, shard_index, shard_count)
        .with_context(|| format!("failed to shard {label} dataloader"))?;
    Ok(loader.slice(start, end))
}

#[cfg(feature = "ddp")]
fn mean_scalar_from_tensor<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap_or(0.0)
}

#[cfg(feature = "ddp")]
fn reduce_mean_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Mean,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar metric: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn process_group_peer_id(runtime: &ParallelRuntime) -> PeerId {
    runtime.global_rank.into()
}

#[cfg(feature = "ddp")]
fn all_reduce_gradients_in_module_order<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    peer_id: PeerId,
    op: ReduceOperation,
) -> Result<()>
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    struct GradientAllReduceVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        peer_id: PeerId,
        op: ReduceOperation,
        trace_grads: bool,
        index: usize,
        error: Option<anyhow::Error>,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientAllReduceVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if self.error.is_some() {
                return;
            }

            self.index += 1;
            let grad_index = self.index;

            let grad = match self.grads.remove::<B::InnerBackend, D>(param.id) {
                Some(grad) => grad,
                None => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] missing, zero-filling shape={:?}",
                            self.peer_id,
                            param.val().shape().dims::<D>()
                        );
                    }
                    param.val().inner().zeros_like()
                }
            };

            if self.trace_grads && grad_index <= 12 {
                info!(
                    "process-group DDP peer_id={} gradient[{grad_index}] entering all-reduce shape={:?}",
                    self.peer_id,
                    grad.shape().dims::<D>()
                );
            }

            match all_reduce::<B::InnerBackend>(
                self.peer_id,
                grad.into_primitive().tensor(),
                self.op,
            ) {
                Ok(reduced) => {
                    if self.trace_grads && grad_index <= 12 {
                        info!(
                            "process-group DDP peer_id={} gradient[{grad_index}] completed all-reduce",
                            self.peer_id
                        );
                    }
                    self.grads.register(
                        param.id,
                        Tensor::<B::InnerBackend, D>::from_primitive(TensorPrimitive::Float(
                            reduced,
                        )),
                    )
                }
                Err(err) => {
                    self.error = Some(anyhow!(
                        "failed to all-reduce process-group DDP gradients: {err:?}"
                    ));
                }
            }
        }
    }

    let trace_grads = true;
    let mut visitor = GradientAllReduceVisitor::<B> {
        grads,
        peer_id,
        op,
        trace_grads,
        index: 0,
        error: None,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);

    if let Some(err) = visitor.error {
        return Err(err);
    }

    Ok(())
}

#[cfg(feature = "ddp")]
fn save_process_group_checkpoint<B, S>(
    run_dir: &Path,
    epoch: usize,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<
            B,
            S,
            LanguageTrainModel<B>,
            OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
        >,
    >,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    FileCheckpointer::new(recorder, &checkpoint_dir, "model")
        .save(epoch, learner.model().model.into_record())
        .with_context(|| {
            format!(
                "failed to save process-group model checkpoint {epoch} in {}",
                checkpoint_dir.display()
            )
        })?;
    Ok(())
}

#[cfg(feature = "ddp")]
fn load_process_group_checkpoint<B, S>(
    run_dir: &Path,
    epoch: usize,
    device: &B::Device,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<
            B,
            S,
            LanguageTrainModel<B>,
            OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
        >,
    >,
) -> Result<
    burn_train::Learner<
        burn_train::LearningComponentsMarker<
            B,
            S,
            LanguageTrainModel<B>,
            OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
        >,
    >,
>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    let checkpoint_dir = run_dir.join("checkpoint");
    let recorder = BinFileRecorder::<FullPrecisionSettings>::new();
    let model_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "model")
        .restore(epoch, device)
        .with_context(|| {
            format!(
                "failed to restore process-group model checkpoint {epoch} from {}",
                checkpoint_dir.display()
            )
        })?;
    learner.load_model(model_record);

    let optim_path = checkpoint_dir.join(format!("optim-{epoch}.bin"));
    if optim_path.is_file() {
        let optim_record = FileCheckpointer::new(recorder.clone(), &checkpoint_dir, "optim")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group optimizer checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_optim(optim_record);
    }

    let scheduler_path = checkpoint_dir.join(format!("scheduler-{epoch}.bin"));
    if scheduler_path.is_file() {
        let scheduler_record = FileCheckpointer::new(recorder, &checkpoint_dir, "scheduler")
            .restore(epoch, device)
            .with_context(|| {
                format!(
                    "failed to restore process-group scheduler checkpoint {epoch} from {}",
                    checkpoint_dir.display()
                )
            })?;
        learner.load_scheduler(scheduler_record);
    }

    Ok(learner)
}

#[cfg(feature = "ddp")]
fn run_process_group_validation<B, S>(
    env: &TrainEnvironment<'_, B>,
    learner: &burn_train::Learner<
        burn_train::LearningComponentsMarker<
            B,
            S,
            LanguageTrainModel<B>,
            OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
        >,
    >,
) -> Option<f64>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    if !env.parallel_runtime.is_primary() {
        return None;
    }

    let model = learner.model().valid();
    let mut iterator = env.valid_loader.iter();
    let mut total = 0.0;
    let mut count = 0usize;

    while let Some(item) = iterator.next() {
        let output = model.step(item);
        let loss_value: LossValue<ValidBackend<B>> = output.adapt();
        total += mean_scalar_from_tensor(loss_value.value());
        count += 1;
    }

    (count > 0).then_some(total / count as f64)
}

#[cfg(feature = "ddp")]
fn train_with_collective_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
    scheduler: S,
    collective: burn_collective::CollectiveConfig,
    peer_id: PeerId,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    let _session = CollectiveSessionGuard::<B::InnerBackend>::register(
        peer_id,
        env.device.clone(),
        collective,
    )?;

    let local_train_loader = shard_dataloader(
        Arc::clone(&env.train_loader),
        env.parallel_runtime.global_rank,
        env.parallel_runtime.world_size,
        "train",
    )?;

    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let local_train_steps = local_train_loader.num_items();
    let mut learner = burn_train::Learner::new(model, optimizer, scheduler);
    if let Some(checkpoint) = env.resume_checkpoint_epoch {
        learner =
            load_process_group_checkpoint::<B, S>(env.run_dir, checkpoint, env.device, learner)?;
    }
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);

    info!(
        "training strategy: mode={:?} replicas={} local_rank={} global_rank={} local_train_steps={} start_epoch={}",
        env.parallel_runtime.mode,
        env.parallel_runtime.world_size,
        env.parallel_runtime.local_rank,
        env.parallel_runtime.global_rank,
        local_train_steps,
        start_epoch
    );

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group DDP epoch {} on global_rank={}",
            epoch, env.parallel_runtime.global_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        while let Some(item) = iterator.next() {
            iteration += 1;
            for _ in 0..env.parallel_runtime.world_size {
                learner.lr_step();
            }

            let item = learner.train_step(item);
            let train_output = item.item.sync();
            let loss_value: LossValue<ValidBackend<B>> = train_output.adapt();
            info!(
                "process-group DDP rank={} iteration={} entering scalar loss all-reduce",
                env.parallel_runtime.global_rank, iteration
            );
            let mean_train_loss =
                reduce_mean_scalar::<ValidBackend<B>>(peer_id, loss_value.value())?;
            info!(
                "process-group DDP rank={} iteration={} completed scalar loss all-reduce",
                env.parallel_runtime.global_rank, iteration
            );

            accumulator.accumulate(&learner.model(), item.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                info!(
                    "process-group DDP rank={} iteration={} entering gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                let mut grads = accumulator.grads();
                // Fresh multi-process launches instantiate random ParamIds per rank, so
                // cross-rank gradient sync must follow deterministic module traversal order.
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Mean,
                )?;
                info!(
                    "process-group DDP rank={} iteration={} completed gradient all-reduce",
                    env.parallel_runtime.global_rank, iteration
                );
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && (iteration % metric_every == 0 || iteration == local_train_steps)
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(
                        env.parallel_runtime
                            .world_size
                            .saturating_mul(local_train_steps),
                    )
                    .saturating_add(iteration.saturating_mul(env.parallel_runtime.world_size));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    mean_train_loss,
                    learner.lr_current(),
                    epoch,
                    env.epochs
                );
            }
        }

        if env.parallel_runtime.is_primary() {
            if let Some(valid_loss) = run_process_group_validation(env, &learner) {
                info!("valid epoch={} loss={valid_loss:.4}", epoch);
            }
            save_process_group_checkpoint::<B, S>(env.run_dir, epoch, &learner)?;
        }
    }

    Ok(learner.model().valid().model)
}

#[cfg(feature = "ddp")]
fn train_with_process_group_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
    scheduler: S,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    let collective = resolve_collective_config(env.parallel_runtime, env.parallel_config)?;
    train_with_collective_scheduler(
        env,
        model,
        optimizer,
        scheduler,
        collective,
        process_group_peer_id(env.parallel_runtime),
    )
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::{DataLoaderIterator, Progress};
    #[cfg(feature = "ddp")]
    use burn::module::list_param_ids;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    #[cfg(feature = "ddp")]
    use burn_collective::reset_collective;
    use burn_ndarray::NdArray;
    #[cfg(feature = "ddp")]
    use std::sync::{Mutex, OnceLock};
    #[cfg(feature = "ddp")]
    use tempfile::tempdir;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestValidBackend = ValidBackend<TestBackend>;

    #[derive(Clone)]
    struct StaticSequenceLoader<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
    }

    impl<B: BackendTrait> StaticSequenceLoader<B> {
        fn new(items: Vec<SequenceBatch<B>>) -> Self {
            Self { items }
        }
    }

    struct StaticSequenceIterator<B: BackendTrait> {
        items: Vec<SequenceBatch<B>>,
        index: usize,
    }

    impl<B: BackendTrait> Iterator for StaticSequenceIterator<B> {
        type Item = SequenceBatch<B>;

        fn next(&mut self) -> Option<Self::Item> {
            let item = self.items.get(self.index).cloned();
            if item.is_some() {
                self.index += 1;
            }
            item
        }
    }

    impl<B: BackendTrait> DataLoaderIterator<SequenceBatch<B>> for StaticSequenceIterator<B> {
        fn progress(&self) -> Progress {
            Progress::new(self.index, self.items.len())
        }
    }

    impl<B> DataLoader<B, SequenceBatch<B>> for StaticSequenceLoader<B>
    where
        B: BackendTrait + 'static,
    {
        fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
            Box::new(StaticSequenceIterator {
                items: self.items.clone(),
                index: 0,
            })
        }

        fn num_items(&self) -> usize {
            self.items.len()
        }

        fn to_device(&self, _device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            Arc::new(self.clone())
        }

        fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
            let len = self.items.len();
            let start = start.min(len);
            let end = end.min(len);
            Arc::new(Self {
                items: self.items[start..end].to_vec(),
            })
        }
    }

    fn make_batch<B: BackendTrait>(
        device: &B::Device,
        inputs: &[i64],
        targets: &[i64],
        shape: [usize; 2],
    ) -> SequenceBatch<B> {
        SequenceBatch::new(
            Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
            Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
            None,
        )
    }

    fn tiny_model_config() -> BDHConfig {
        BDHConfig {
            n_layer: 1,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..Default::default()
        }
    }

    fn tiny_training_hparams() -> TrainingHyperparameters {
        TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 2,
            checkpoint_interval_iters: 2000,
            log_frequency: 1,
            fast_train: false,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        }
    }

    fn tiny_training_hparams_with_epochs(
        epochs: usize,
        resume_checkpoint_epoch: Option<usize>,
    ) -> TrainingHyperparameters {
        let mut training = tiny_training_hparams();
        training.epochs = Some(epochs);
        training.resume_checkpoint_epoch = resume_checkpoint_epoch;
        training
    }

    #[cfg(feature = "ddp")]
    fn collective_test_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[cfg(feature = "ddp")]
    fn flatten_gradients_in_module_order<B, M>(module: &M, mut grads: GradientsParams) -> Vec<f32>
    where
        B: AutodiffBackend,
        M: AutodiffModule<B>,
    {
        #[derive(Default)]
        struct GradientCollector {
            values: Vec<f32>,
        }

        struct GradientCollectorVisitor<'a> {
            collector: &'a mut GradientCollector,
            grads: &'a mut GradientsParams,
        }

        impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientCollectorVisitor<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                let grad = self
                    .grads
                    .remove::<B::InnerBackend, D>(param.id)
                    .unwrap_or_else(|| param.val().inner().zeros_like());
                let values = grad
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("gradient data");
                self.collector.values.extend(values);
            }
        }

        let mut collector = GradientCollector::default();
        let mut visitor = GradientCollectorVisitor {
            collector: &mut collector,
            grads: &mut grads,
        };
        module.visit(&mut visitor);
        collector.values
    }

    #[cfg(feature = "ddp")]
    fn mean_abs_diff(left: &[f32], right: &[f32]) -> f32 {
        left.iter()
            .zip(right.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / left.len().max(1) as f32
    }

    #[cfg(feature = "ddp")]
    fn l2_norm(values: &[f32]) -> f32 {
        values.iter().map(|value| value * value).sum::<f32>().sqrt()
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_scheduler_executes_local_ddp_on_ndarray() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve local ddp runtime");

        let primary_device = <TestBackend as BackendTrait>::Device::default();
        let devices =
            resolve_training_devices::<TestBackend>(&parallel_runtime, &primary_device).unwrap();
        assert_eq!(devices.len(), 2, "expected 2 local replicas");

        let valid_device = <TestValidBackend as BackendTrait>::Device::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "ddp-ndarray-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            epochs: 1,
        };

        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let trained = train_with_scheduler(&env, model, optimizer, 1e-3).expect("ddp train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite(), "ddp smoke loss must be finite");
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn shard_bounds_evenly_distribute_remainder_steps() {
        assert_eq!(shard_bounds(5, 0, 2).expect("rank0"), (0, 3));
        assert_eq!(shard_bounds(5, 1, 2).expect("rank1"), (3, 5));
        assert!(shard_bounds(1, 1, 2).is_err());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn gradient_mean_matches_combined_batch_reference_in_module_order() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let config = tiny_model_config();
        let reference = LanguageTrainModel::new(BDH::<TestBackend>::new(config, &device));
        let combined_model = reference.clone();
        let shard_a_model = reference.clone();
        let shard_b_model = reference;

        let shard_a = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7],
            &[1, 2, 3, 4, 5, 6, 7, 0],
            [2, 4],
        );
        let shard_b = make_batch::<TestBackend>(
            &device,
            &[7, 6, 5, 4, 3, 2, 1, 0],
            &[6, 5, 4, 3, 2, 1, 0, 7],
            [2, 4],
        );
        let combined = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 4, 5, 6, 7, 7, 6, 5, 4, 3, 2, 1, 0],
            &[1, 2, 3, 4, 5, 6, 7, 0, 6, 5, 4, 3, 2, 1, 0, 7],
            [4, 4],
        );

        let combined_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &combined_model,
            burn_train::TrainStep::step(&combined_model, combined).grads,
        );
        let shard_a_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_a_model,
            burn_train::TrainStep::step(&shard_a_model, shard_a).grads,
        );
        let shard_b_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &shard_b_model,
            burn_train::TrainStep::step(&shard_b_model, shard_b).grads,
        );

        assert_eq!(combined_grads.len(), shard_a_grads.len());
        assert_eq!(combined_grads.len(), shard_b_grads.len());

        let averaged_shards = shard_a_grads
            .iter()
            .zip(shard_b_grads.iter())
            .map(|(lhs, rhs)| (lhs + rhs) * 0.5)
            .collect::<Vec<_>>();

        let mean_abs = mean_abs_diff(&combined_grads, &averaged_shards);
        let combined_norm = l2_norm(&combined_grads);
        let averaged_norm = l2_norm(&averaged_shards);

        assert!(
            mean_abs <= 1.0e-5,
            "combined-batch reference and mean rank-local gradients drifted: mean_abs_diff={mean_abs}"
        );
        assert!(
            (combined_norm - averaged_norm).abs() <= 1.0e-5,
            "gradient norms drifted: combined_norm={combined_norm} averaged_norm={averaged_norm}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_runs_single_rank_and_writes_checkpoint() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = <TestBackend as BackendTrait>::Device::default();
        let valid_device = <TestValidBackend as BackendTrait>::Device::default();
        let train_batches = vec![
            make_batch::<TestBackend>(
                &primary_device,
                &[0, 1, 2, 3, 4, 5, 6, 7],
                &[1, 2, 3, 4, 5, 6, 7, 0],
                [2, 4],
            ),
            make_batch::<TestBackend>(
                &primary_device,
                &[7, 6, 5, 4, 3, 2, 1, 0],
                &[6, 5, 4, 3, 2, 1, 0, 7],
                [2, 4],
            ),
        ];
        let valid_batches = vec![make_batch::<TestValidBackend>(
            &valid_device,
            &[0, 0, 1, 1, 2, 2, 3, 3],
            &[0, 1, 1, 2, 2, 3, 3, 0],
            [2, 4],
        )];

        let training = tiny_training_hparams();
        let model_config = tiny_model_config();
        let devices = vec![primary_device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-single-rank",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            epochs: 1,
        };
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let trained =
            train_with_collective_scheduler(&env, model, optimizer, 1e-3, collective, 0.into())
                .expect("collective train");
        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(trained.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_with_collective_scheduler_resumes_from_checkpoint_family() {
        let _lock = collective_test_lock().lock().expect("collective lock");
        reset_collective::<TestValidBackend>();

        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let parallel_config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            data: burn_dragon_train::ParallelDataConfig {
                size: 1,
                ..Default::default()
            },
            ..Default::default()
        };
        let parallel_runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };

        let primary_device = <TestBackend as BackendTrait>::Device::default();
        let valid_device = <TestValidBackend as BackendTrait>::Device::default();
        let train_loader: Arc<dyn DataLoader<TestBackend, SequenceBatch<TestBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![
                make_batch::<TestBackend>(
                    &primary_device,
                    &[0, 1, 2, 3, 4, 5, 6, 7],
                    &[1, 2, 3, 4, 5, 6, 7, 0],
                    [2, 4],
                ),
                make_batch::<TestBackend>(
                    &primary_device,
                    &[7, 6, 5, 4, 3, 2, 1, 0],
                    &[6, 5, 4, 3, 2, 1, 0, 7],
                    [2, 4],
                ),
            ]));
        let valid_loader: Arc<dyn DataLoader<TestValidBackend, SequenceBatch<TestValidBackend>>> =
            Arc::new(StaticSequenceLoader::new(vec![make_batch::<
                TestValidBackend,
            >(
                &valid_device,
                &[0, 0, 1, 1, 2, 2, 3, 3],
                &[0, 1, 1, 2, 2, 3, 3, 0],
                [2, 4],
            )]));
        let devices = vec![primary_device.clone()];
        let model_config = tiny_model_config();
        let collective =
            resolve_collective_config(&parallel_runtime, &parallel_config).expect("collective");

        let training_first = tiny_training_hparams_with_epochs(1, None);
        let env_first = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_first,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_loader: Arc::clone(&train_loader),
            valid_loader: Arc::clone(&valid_loader),
            epochs: 1,
        };
        let model_first = LanguageTrainModel::new(BDH::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_first = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        train_with_collective_scheduler(
            &env_first,
            model_first,
            optimizer_first,
            1e-3,
            collective.clone(),
            0.into(),
        )
        .expect("first collective train");
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());

        reset_collective::<TestValidBackend>();

        let training_resume = tiny_training_hparams_with_epochs(2, Some(1));
        let env_resume = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "collective-resume",
            backend_name: "cpu",
            training: &training_resume,
            resume_checkpoint_epoch: Some(1),
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_loader,
            valid_loader,
            epochs: 2,
        };
        let model_resume = LanguageTrainModel::new(BDH::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer_resume = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();
        let resumed = train_with_collective_scheduler(
            &env_resume,
            model_resume,
            optimizer_resume,
            1e-3,
            collective,
            0.into(),
        )
        .expect("resumed collective train");

        let probe = make_batch::<TestValidBackend>(
            &valid_device,
            &[1, 2, 3, 4, 4, 3, 2, 1],
            &[2, 3, 4, 5, 3, 2, 1, 0],
            [2, 4],
        );
        let loss =
            language_model_loss::<TestValidBackend>(resumed.forward(probe.inputs), probe.targets)
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("loss vec")[0];

        assert!(loss.is_finite());
        assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());

        reset_collective::<TestValidBackend>();
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn process_group_peer_id_uses_global_rank() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 4,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };

        assert_eq!(process_group_peer_id(&runtime), 3usize.into());
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn fresh_models_use_random_param_ids_but_stable_module_traversal_shapes() {
        #[derive(Default)]
        struct ShapeCollector {
            shapes: Vec<Vec<usize>>,
        }

        impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ShapeCollector {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
                self.shapes.push(param.val().shape().dims::<D>().into());
            }
        }

        let device = <TestBackend as BackendTrait>::Device::default();
        let config = tiny_model_config();
        let model_a = LanguageTrainModel::new(BDH::<TestBackend>::new(config.clone(), &device));
        let model_b = LanguageTrainModel::new(BDH::<TestBackend>::new(config, &device));

        let ids_a = list_param_ids(&model_a);
        let ids_b = list_param_ids(&model_b);
        let mut shapes_a = ShapeCollector::default();
        let mut shapes_b = ShapeCollector::default();
        model_a.visit(&mut shapes_a);
        model_b.visit(&mut shapes_b);

        assert_eq!(ids_a.len(), ids_b.len());
        assert_ne!(
            ids_a, ids_b,
            "fresh models should not rely on matching ParamIds"
        );
        assert_eq!(shapes_a.shapes, shapes_b.shapes);
    }
}
