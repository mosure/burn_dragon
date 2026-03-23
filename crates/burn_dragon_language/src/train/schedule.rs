use crate::train::prelude::*;
use crate::train::utils::log_theoretical_profile;
#[cfg(feature = "ddp")]
use burn::tensor::TensorPrimitive;
use std::collections::BTreeSet;
#[cfg(feature = "ddp")]
use std::collections::HashMap;
#[cfg(feature = "ddp")]
use std::marker::PhantomData;

const CHECKPOINT_KEEP_LAST: usize = 2;

struct FileMetricBestCheckpointingStrategy {
    run_dir: PathBuf,
    metric_name: String,
    direction: burn_train::metric::store::Direction,
    split: burn_train::metric::store::Split,
    best_epoch: Option<usize>,
    best_value: Option<f64>,
}

impl FileMetricBestCheckpointingStrategy {
    fn new<M>(
        run_dir: &Path,
        metric: &M,
        direction: burn_train::metric::store::Direction,
        split: burn_train::metric::store::Split,
    ) -> Self
    where
        M: burn_train::metric::Metric,
    {
        Self {
            run_dir: run_dir.to_path_buf(),
            metric_name: metric.name().to_string(),
            direction,
            split,
            best_epoch: None,
            best_value: None,
        }
    }

    fn is_better(&self, candidate: f64, current: f64) -> bool {
        match self.direction {
            burn_train::metric::store::Direction::Lowest => candidate < current,
            burn_train::metric::store::Direction::Highest => candidate > current,
        }
    }

    fn checkpoint_path(&self, epoch: usize) -> PathBuf {
        self.run_dir
            .join("checkpoint")
            .join(format!("model-{epoch}.bin"))
    }

    fn metric_log_path(&self, epoch: usize) -> PathBuf {
        let split_dir = match self.split {
            burn_train::metric::store::Split::Train => "train",
            burn_train::metric::store::Split::Valid => "valid",
            burn_train::metric::store::Split::Test(_) => "test",
        };
        self.run_dir
            .join(split_dir)
            .join(format!("epoch-{epoch}"))
            .join(format!("{}.log", self.metric_name))
    }

    fn checkpoint_exists(&self, epoch: usize) -> bool {
        self.checkpoint_path(epoch).is_file()
    }

    fn existing_checkpoint_epochs(&self) -> BTreeSet<usize> {
        let checkpoint_dir = self.run_dir.join("checkpoint");
        let Ok(entries) = fs::read_dir(checkpoint_dir) else {
            return BTreeSet::new();
        };

        entries
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                name.strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()
            })
            .collect()
    }

    fn metric_mean_from_log(&self, epoch: usize) -> Option<f64> {
        let path = self.metric_log_path(epoch);
        let content = fs::read_to_string(path).ok()?;
        let mut sum = 0.0;
        let mut count = 0usize;

        for line in content.lines() {
            let field = line.split(',').next()?.trim();
            let value = field.parse::<f64>().ok()?;
            sum += value;
            count += 1;
        }

        (count > 0).then_some(sum / count as f64)
    }

    fn update_best_candidate(&mut self, epoch: usize, value: f64) -> Option<usize> {
        let should_replace = self
            .best_value
            .is_none_or(|current| self.is_better(value, current));

        if !should_replace {
            return None;
        }

        let previous_best = self.best_epoch.replace(epoch);
        self.best_value = Some(value);
        previous_best.filter(|previous_best| *previous_best != epoch)
    }

    fn refresh_best_from_history(&mut self, last_epoch: usize) {
        self.best_epoch = None;
        self.best_value = None;

        for epoch in 1..=last_epoch {
            if let Some(value) = self.metric_mean_from_log(epoch) {
                let _ = self.update_best_candidate(epoch, value);
            }
        }
    }

    fn actions_for_epoch(
        &mut self,
        epoch: usize,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.refresh_best_from_history(epoch);

        let mut keep_epochs = BTreeSet::new();
        keep_epochs.extend(epoch.saturating_sub(CHECKPOINT_KEEP_LAST - 1).max(1)..=epoch);
        if let Some(best_epoch) = self.best_epoch {
            keep_epochs.insert(best_epoch);
        }

        let existing_epochs = self.existing_checkpoint_epochs();
        let mut actions = vec![burn_train::checkpoint::CheckpointingAction::Save];
        actions.extend(
            existing_epochs
                .into_iter()
                .filter(|existing_epoch| !keep_epochs.contains(existing_epoch))
                .map(burn_train::checkpoint::CheckpointingAction::Delete),
        );
        actions
    }
}

impl burn_train::checkpoint::CheckpointingStrategy for FileMetricBestCheckpointingStrategy {
    fn checkpointing(
        &mut self,
        epoch: usize,
        _store: &burn_train::metric::store::EventStoreClient,
    ) -> Vec<burn_train::checkpoint::CheckpointingAction> {
        self.actions_for_epoch(epoch)
    }
}

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
    .with_checkpointing_strategy(FileMetricBestCheckpointingStrategy::new(
        env.run_dir,
        &LossMetric::<ValidBackend<B>>::new(),
        burn_train::metric::store::Direction::Lowest,
        burn_train::metric::store::Split::Valid,
    ))
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
fn process_group_data_shard(
    runtime: &ParallelRuntime,
    config: &ParallelConfig,
) -> Result<(
    usize,
    usize,
    Option<PipelineRankAssignment>,
    Option<PipelineParallelLayout>,
)> {
    let layout = resolve_pipeline_parallel_layout(runtime, config)?;
    if let Some(layout) = layout {
        let assignment = layout.assignment(runtime.global_rank).clone();
        return Ok((
            assignment.data_parallel_rank,
            layout.data_parallel_size,
            Some(assignment),
            Some(layout),
        ));
    }

    Ok((runtime.global_rank, runtime.world_size, None, None))
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
fn scale_gradients_in_module_order<B, M>(module: &M, grads: &mut GradientsParams, scale: f32)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    if (scale - 1.0).abs() <= f32::EPSILON {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        scale: f32,
        _marker: PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(self.scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        scale,
        _marker: PhantomData,
    };
    module.visit(&mut visitor);
}

#[cfg(feature = "ddp")]
fn reduce_sum_scalar<B: BackendTrait>(peer_id: PeerId, tensor: Tensor<B, 1>) -> Result<f64> {
    let reduced = all_reduce::<B>(
        peer_id,
        tensor.into_primitive().tensor(),
        ReduceOperation::Sum,
    )
    .map_err(|err| anyhow!("failed to all-reduce scalar sum: {err:?}"))?;
    Ok(mean_scalar_from_tensor(Tensor::<B, 1>::from_primitive(
        TensorPrimitive::Float(reduced),
    )))
}

#[cfg(feature = "ddp")]
fn scalar_tensor<B: BackendTrait>(device: &B::Device, value: f32) -> Tensor<B, 1> {
    Tensor::<B, 1>::from_floats([value], device)
}

#[cfg(feature = "ddp")]
fn scalar_flag<B: BackendTrait>(device: &B::Device, enabled: bool) -> Tensor<B, 1> {
    scalar_tensor::<B>(device, if enabled { 1.0 } else { 0.0 })
}

#[cfg(feature = "ddp")]
fn broadcast_float_tensor_rooted<B: BackendTrait, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D>>,
) -> Result<Tensor<B, D>> {
    let root_tensor = if global_rank == root_rank {
        Some(
            tensor
                .ok_or_else(|| anyhow!("collective root rank {root_rank} is missing a tensor"))?
                .into_primitive()
                .tensor(),
        )
    } else {
        None
    };
    let broadcasted = broadcast::<B>(peer_id, root_tensor).map_err(|err| {
        anyhow!("failed to broadcast rooted tensor from rank {root_rank}: {err:?}")
    })?;
    Ok(Tensor::<B, D>::from_primitive(TensorPrimitive::Float(
        broadcasted,
    )))
}

#[cfg(feature = "ddp")]
fn broadcast_usize_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<usize>,
) -> Result<usize> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_tensor::<B>(device, value as f32)),
    )?;
    Ok(mean_scalar_from_tensor(tensor).round().max(0.0) as usize)
}

#[cfg(feature = "ddp")]
fn broadcast_bool_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    value: Option<bool>,
) -> Result<bool> {
    let tensor = broadcast_float_tensor_rooted::<B, 1>(
        peer_id,
        global_rank,
        root_rank,
        value.map(|value| scalar_flag::<B>(device, value)),
    )?;
    Ok(mean_scalar_from_tensor(tensor) >= 0.5)
}

#[cfg(feature = "ddp")]
fn broadcast_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Tensor<B, D, Int>> {
    let broadcasted = broadcast_float_tensor_rooted::<B::InnerBackend, D>(
        peer_id,
        global_rank,
        root_rank,
        tensor.map(|tensor| tensor.float().inner()),
    )?;
    Ok(Tensor::<B, D>::from_inner(broadcasted).int())
}

#[cfg(feature = "ddp")]
fn broadcast_optional_int_tensor_rooted<B: AutodiffBackend, const D: usize>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    tensor: Option<Tensor<B, D, Int>>,
) -> Result<Option<Tensor<B, D, Int>>> {
    let has_tensor = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(tensor.is_some()),
    )?;
    if !has_tensor {
        return Ok(None);
    }
    broadcast_int_tensor_rooted(peer_id, global_rank, root_rank, tensor).map(Some)
}

#[cfg(feature = "ddp")]
fn broadcast_sequence_batch_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    batch: Option<SequenceBatch<B>>,
) -> Result<SequenceBatch<B>> {
    let inputs = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.inputs.clone()),
    )?;
    let targets = broadcast_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        batch.as_ref().map(|batch| batch.targets.clone()),
    )?;
    let summary_event_mask = broadcast_optional_int_tensor_rooted(
        peer_id,
        global_rank,
        root_rank,
        device,
        batch
            .as_ref()
            .and_then(|batch| batch.summary_event_mask.clone()),
    )?;
    let reset_stream_state = broadcast_bool_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        Some(batch.as_ref().is_some_and(|batch| batch.reset_stream_state)),
    )?;

    Ok(SequenceBatch {
        inputs,
        targets,
        summary_event_mask,
        reset_stream_state,
    })
}

#[cfg(feature = "ddp")]
fn detach_pipeline_state_to_inner<B: AutodiffBackend>(
    state: &LanguagePipelineState<B>,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        state.current().clone().detach().inner(),
        state
            .residual_history()
            .iter()
            .cloned()
            .map(|tensor| tensor.detach().inner())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn attach_pipeline_state_require_grad<B: AutodiffBackend>(
    state: LanguagePipelineState<B::InnerBackend>,
) -> LanguagePipelineState<B> {
    let (current, residual_history) = state.into_parts();
    LanguagePipelineState::from_parts(
        Tensor::<B, 4>::from_inner(current).require_grad(),
        residual_history
            .into_iter()
            .map(|tensor| Tensor::<B, 4>::from_inner(tensor).require_grad())
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_rooted<B: AutodiffBackend>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B::InnerBackend>> {
    let history_len = broadcast_usize_rooted::<B::InnerBackend>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone().detach().inner()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B::InnerBackend, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone().detach().inner()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn broadcast_pipeline_state_inner_rooted<B: BackendTrait>(
    peer_id: PeerId,
    global_rank: usize,
    root_rank: usize,
    device: &B::Device,
    state: Option<&LanguagePipelineState<B>>,
) -> Result<LanguagePipelineState<B>> {
    let history_len = broadcast_usize_rooted::<B>(
        peer_id,
        global_rank,
        root_rank,
        device,
        state.map(|state| state.residual_history().len()),
    )?;
    let current = broadcast_float_tensor_rooted::<B, 4>(
        peer_id,
        global_rank,
        root_rank,
        state.map(|state| state.current().clone()),
    )?;
    let residual_history = (0..history_len)
        .map(|index| {
            broadcast_float_tensor_rooted::<B, 4>(
                peer_id,
                global_rank,
                root_rank,
                state.map(|state| state.residual_history()[index].clone()),
            )
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(LanguagePipelineState::from_parts(current, residual_history))
}

#[cfg(feature = "ddp")]
fn pipeline_surrogate_loss<B: AutodiffBackend>(
    output_state: &LanguagePipelineState<B>,
    grad_state: LanguagePipelineState<B::InnerBackend>,
) -> Tensor<B, 1> {
    let (grad_current, grad_history) = grad_state.into_parts();
    assert_eq!(
        output_state.residual_history().len(),
        grad_history.len(),
        "pipeline residual history length mismatch"
    );

    let mut surrogate = output_state
        .current()
        .clone()
        .mul(Tensor::<B, 4>::from_inner(grad_current))
        .sum();
    for (residual, grad) in output_state
        .residual_history()
        .iter()
        .zip(grad_history.into_iter())
    {
        surrogate = surrogate + residual.clone().mul(Tensor::<B, 4>::from_inner(grad)).sum();
    }
    surrogate
}

#[cfg(feature = "ddp")]
fn pipeline_input_grad_state<B: AutodiffBackend>(
    input_state: &LanguagePipelineState<B>,
    grads: &mut B::Gradients,
) -> LanguagePipelineState<B::InnerBackend> {
    LanguagePipelineState::from_parts(
        input_state
            .current()
            .grad_remove(grads)
            .unwrap_or_else(|| input_state.current().clone().inner().zeros_like()),
        input_state
            .residual_history()
            .iter()
            .map(|tensor| {
                tensor
                    .grad_remove(grads)
                    .unwrap_or_else(|| tensor.clone().inner().zeros_like())
            })
            .collect(),
    )
}

#[cfg(feature = "ddp")]
fn slice_batch_int<B: BackendTrait>(
    tensor: Tensor<B, 2, Int>,
    range: std::ops::Range<usize>,
) -> Tensor<B, 2, Int> {
    let [_batch, block_size] = tensor.shape().dims();
    tensor.slice([range.start..range.end, 0..block_size])
}

#[cfg(feature = "ddp")]
fn pipeline_replica_root_rank(layout: &PipelineParallelLayout, data_parallel_rank: usize) -> usize {
    data_parallel_rank * layout.stage_count
}

#[cfg(feature = "ddp")]
fn global_rank_for_virtual_stage(
    plan: &PipelinePlan,
    layout: &PipelineParallelLayout,
    data_parallel_rank: usize,
    virtual_stage_id: usize,
) -> usize {
    let physical_stage_id = plan.assignment(virtual_stage_id).physical_stage_id;
    data_parallel_rank * layout.stage_count + physical_stage_id
}

#[cfg(feature = "ddp")]
struct DistributedPipelineForwardCache<B: AutodiffBackend> {
    input_state: Option<LanguagePipelineState<B>>,
    output_state: Option<LanguagePipelineState<B>>,
    loss: Option<Tensor<B, 1>>,
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
struct DistributedPipelineTrainStepResult {
    grads: GradientsParams,
    mean_train_loss: f64,
}

#[cfg(feature = "ddp")]
fn distributed_pipeline_train_step<B>(
    peer_id: PeerId,
    model: &LanguageTrainModel<B>,
    batch: SequenceBatch<B>,
    layout: &PipelineParallelLayout,
    assignment: &PipelineRankAssignment,
    device: &B::Device,
) -> Result<DistributedPipelineTrainStepResult>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if layout.data_parallel_size != 1 {
        return Err(anyhow!(
            "distributed pipeline execution currently requires parallel.data.size = 1 (got {})",
            layout.data_parallel_size
        ));
    }

    let plan = model
        .pipeline_plan
        .as_ref()
        .ok_or_else(|| anyhow!("distributed pipeline step requires a pipeline plan"))?;
    let [batch_size, _block_size] = batch.inputs.shape().dims();
    let ranges = split_microbatch_ranges(batch_size, plan.microbatches)?;
    let chunk_inputs = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.inputs.clone(), range))
        .collect::<Vec<_>>();
    let chunk_targets = ranges
        .iter()
        .cloned()
        .map(|range| slice_batch_int(batch.targets.clone(), range))
        .collect::<Vec<_>>();
    let chunk_masks = ranges
        .iter()
        .cloned()
        .map(|range| {
            batch
                .summary_event_mask
                .clone()
                .map(|mask| slice_batch_int(mask, range))
        })
        .collect::<Vec<_>>();
    let mut chunk_states = (0..plan.microbatches)
        .map(|_| model.model.init_state())
        .collect::<Vec<ModelState<B>>>();
    let mut forward_cache = HashMap::<(usize, usize), DistributedPipelineForwardCache<B>>::new();
    let mut incoming_forward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut incoming_backward =
        HashMap::<(usize, usize), LanguagePipelineState<B::InnerBackend>>::new();
    let mut local_accumulator = GradientsAccumulator::new();
    let mut local_loss: Option<Tensor<B::InnerBackend, 1>> = None;
    let last_virtual_stage_id = plan.total_virtual_stages.saturating_sub(1);
    let last_stage_rank = global_rank_for_virtual_stage(plan, layout, 0, last_virtual_stage_id);

    for event in &plan.events {
        let microbatch_id = event.microbatch_id;
        let local_forward_output = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let input_state = if event.virtual_stage_id == 0 {
                model
                    .model
                    .begin_language_pipeline(chunk_inputs[microbatch_id].clone())
            } else {
                let input_state = incoming_forward
                    .remove(&(event.virtual_stage_id, microbatch_id))
                    .ok_or_else(|| {
                        anyhow!(
                            "missing forward pipeline state for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?;
                attach_pipeline_state_require_grad::<B>(input_state)
            };
            let cached_input = (event.virtual_stage_id > 0).then_some(input_state.clone());
            let output_state = model.model.forward_language_pipeline_stage_with_state(
                input_state,
                &mut chunk_states[microbatch_id],
                plan.assignment(event.virtual_stage_id).layer_range.clone(),
                chunk_masks[microbatch_id].clone(),
            );

            if event.virtual_stage_id == last_virtual_stage_id {
                let (_hidden, logits) = model.model.finish_language_pipeline_with_state(
                    output_state,
                    &mut chunk_states[microbatch_id],
                );
                let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
                let loss = language_model_loss::<B>(logits, chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
                local_loss = Some(match local_loss {
                    Some(accumulated) => accumulated + loss.clone().detach().inner(),
                    None => loss.clone().detach().inner(),
                });
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: None,
                        loss: Some(loss),
                    },
                );
                None
            } else {
                forward_cache.insert(
                    (event.virtual_stage_id, microbatch_id),
                    DistributedPipelineForwardCache {
                        input_state: cached_input,
                        output_state: Some(output_state.clone()),
                        loss: None,
                    },
                );
                Some(output_state)
            }
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            && event.virtual_stage_id < last_virtual_stage_id
        {
            let sender_rank =
                global_rank_for_virtual_stage(plan, layout, 0, event.virtual_stage_id);
            let receiver_rank =
                global_rank_for_virtual_stage(plan, layout, 0, event.virtual_stage_id + 1);
            let broadcasted = broadcast_pipeline_state_rooted(
                peer_id,
                assignment.global_rank,
                sender_rank,
                device,
                local_forward_output.as_ref(),
            )?;
            if assignment.global_rank == receiver_rank {
                incoming_forward.insert((event.virtual_stage_id + 1, microbatch_id), broadcasted);
            }
        }

        let local_backward_grad = if event.kind
            == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.physical_stage_id == assignment.pipeline_stage_id
        {
            let cached = forward_cache
                .remove(&(event.virtual_stage_id, microbatch_id))
                .ok_or_else(|| {
                    anyhow!(
                        "missing backward cache for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;

            let mut grads = if event.virtual_stage_id == last_virtual_stage_id {
                cached
                    .loss
                    .ok_or_else(|| {
                        anyhow!(
                            "missing terminal loss for virtual_stage={} microbatch={microbatch_id}",
                            event.virtual_stage_id
                        )
                    })?
                    .backward()
            } else {
                let output_state = cached.output_state.as_ref().ok_or_else(|| {
                    anyhow!(
                        "missing stage output for virtual_stage={} microbatch={microbatch_id}",
                        event.virtual_stage_id
                    )
                })?;
                let grad_state = incoming_backward
                        .remove(&(event.virtual_stage_id, microbatch_id))
                        .ok_or_else(|| {
                            anyhow!(
                                "missing backward pipeline gradient for virtual_stage={} microbatch={microbatch_id}",
                                event.virtual_stage_id
                            )
                        })?;
                pipeline_surrogate_loss(output_state, grad_state).backward()
            };

            let input_grad = cached
                .input_state
                .as_ref()
                .map(|input_state| pipeline_input_grad_state(input_state, &mut grads));
            local_accumulator.accumulate(model, GradientsParams::from_grads(grads, model));
            input_grad
        } else {
            None
        };

        if event.kind == burn_dragon_train::train::pipeline::PipelineEventKind::Backward
            && event.virtual_stage_id > 0
        {
            let sender_rank =
                global_rank_for_virtual_stage(plan, layout, 0, event.virtual_stage_id);
            let receiver_rank =
                global_rank_for_virtual_stage(plan, layout, 0, event.virtual_stage_id - 1);
            let broadcasted = broadcast_pipeline_state_inner_rooted::<B::InnerBackend>(
                peer_id,
                assignment.global_rank,
                sender_rank,
                device,
                local_backward_grad.as_ref(),
            )?;
            if assignment.global_rank == receiver_rank {
                incoming_backward.insert((event.virtual_stage_id - 1, microbatch_id), broadcasted);
            }
        }
    }

    let reduced_loss = reduce_sum_scalar::<B::InnerBackend>(
        peer_id,
        if assignment.global_rank == last_stage_rank {
            local_loss.unwrap_or_else(|| Tensor::<B::InnerBackend, 1>::zeros([1], device))
        } else {
            Tensor::<B::InnerBackend, 1>::zeros([1], device)
        },
    )?;

    Ok(DistributedPipelineTrainStepResult {
        grads: local_accumulator.grads(),
        mean_train_loss: reduced_loss,
    })
}

#[cfg(feature = "ddp")]
fn train_with_collective_pipeline_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    mut learner: burn_train::Learner<
        burn_train::LearningComponentsMarker<
            B,
            S,
            LanguageTrainModel<B>,
            OptimizerAdaptor<AdamW, LanguageTrainModel<B>, B>,
        >,
    >,
    local_train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    peer_id: PeerId,
    layout: PipelineParallelLayout,
    assignment: PipelineRankAssignment,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    if layout.data_parallel_size != 1 {
        return Err(anyhow!(
            "parallel.pipeline.enabled process-group execution currently supports only parallel.data.size = 1 (got {})",
            layout.data_parallel_size
        ));
    }

    let local_train_steps = local_train_loader.num_items();
    let metric_every = env.training.log_frequency.max(1);
    let grad_accumulation = env.training.gradient_accumulation_steps.max(1);
    let logical_replica_count = layout.data_parallel_size;
    let start_epoch = env
        .resume_checkpoint_epoch
        .map(|epoch| epoch + 1)
        .unwrap_or(1);
    let batch_root_rank = pipeline_replica_root_rank(&layout, assignment.data_parallel_rank);

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group pipeline epoch {} on global_rank={} stage={}",
            epoch, assignment.global_rank, assignment.pipeline_stage_id
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;

        while iteration < local_train_steps {
            let local_batch = if assignment.global_rank == batch_root_rank {
                iterator.next()
            } else {
                None
            };
            let batch = broadcast_sequence_batch_rooted(
                peer_id,
                assignment.global_rank,
                batch_root_rank,
                env.device,
                local_batch,
            )?;

            iteration += 1;
            for _ in 0..logical_replica_count {
                learner.lr_step();
            }

            let step = distributed_pipeline_train_step(
                peer_id,
                &learner.model(),
                batch,
                &layout,
                &assignment,
                env.device,
            )?;

            accumulator.accumulate(&learner.model(), step.grads);
            accumulation_current += 1;

            if grad_accumulation <= accumulation_current {
                let mut grads = accumulator.grads();
                all_reduce_gradients_in_module_order(
                    &learner.model(),
                    &mut grads,
                    peer_id,
                    ReduceOperation::Sum,
                )?;
                learner.optimizer_step(grads);
                accumulation_current = 0;
            }

            if env.parallel_runtime.is_primary()
                && (iteration % metric_every == 0 || iteration == local_train_steps)
            {
                let progress = iterator.progress();
                let global_iteration = epoch
                    .saturating_sub(1)
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
                info!(
                    "train epoch={} local_step={}/{} global_iteration={} loss={:.4} lr={:.6} global_progress={}/{}",
                    epoch,
                    progress.items_processed,
                    progress.items_total,
                    global_iteration,
                    step.mean_train_loss,
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

    let (data_shard_index, data_shard_count, pipeline_assignment, pipeline_layout) =
        process_group_data_shard(env.parallel_runtime, env.parallel_config)?;

    let local_train_loader = shard_dataloader(
        Arc::clone(&env.train_loader),
        data_shard_index,
        data_shard_count,
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
    if let (Some(layout), Some(assignment)) = (&pipeline_layout, &pipeline_assignment) {
        info!(
            "process-group pipeline topology: {} rank={} stage={} dp_rank={} predecessor={:?} successor={:?} pipeline_group={:?} dp_group={:?}",
            layout.summary(),
            assignment.global_rank,
            assignment.pipeline_stage_id,
            assignment.data_parallel_rank,
            assignment.predecessor_global_rank,
            assignment.successor_global_rank,
            assignment.pipeline_group_ranks,
            assignment.data_parallel_group_ranks,
        );
    }

    if let (Some(layout), Some(assignment)) = (pipeline_layout.clone(), pipeline_assignment.clone())
    {
        return train_with_collective_pipeline_scheduler(
            env,
            learner,
            local_train_loader,
            peer_id,
            layout,
            assignment,
        );
    }

    for epoch in start_epoch..=env.epochs {
        info!(
            "Executing process-group DDP epoch {} on global_rank={}",
            epoch, env.parallel_runtime.global_rank
        );

        let mut iterator = local_train_loader.iter();
        let mut iteration = 0usize;
        let mut accumulator = GradientsAccumulator::new();
        let mut accumulation_current = 0usize;
        let logical_replica_count = env.parallel_runtime.world_size;
        while let Some(item) = iterator.next() {
            iteration += 1;
            for _ in 0..logical_replica_count {
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
                    .saturating_mul(logical_replica_count.saturating_mul(local_train_steps))
                    .saturating_add(iteration.saturating_mul(logical_replica_count));
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
    use burn_train::checkpoint::CheckpointingAction;
    #[cfg(feature = "ddp")]
    use std::sync::{Mutex, OnceLock};
    #[cfg(feature = "ddp")]
    use tempfile::tempdir;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestValidBackend = ValidBackend<TestBackend>;

    #[test]
    fn file_metric_best_strategy_tracks_best_value() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let previous_best = strategy.update_best_candidate(1, 3.5);

        assert_eq!(previous_best, None);
        assert_eq!(strategy.best_epoch, Some(1));
        assert_eq!(strategy.best_value, Some(3.5));
    }

    #[test]
    fn file_metric_best_strategy_replaces_only_on_improvement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );
        strategy.best_epoch = Some(2);
        strategy.best_value = Some(3.2);

        let worse_previous_best = strategy.update_best_candidate(3, 3.3);
        assert_eq!(worse_previous_best, None);
        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(strategy.best_value, Some(3.2));

        let better_previous_best = strategy.update_best_candidate(4, 3.1);
        assert_eq!(better_previous_best, Some(2));
        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(strategy.best_value, Some(3.1));
    }

    fn write_metric_log(run_dir: &Path, split: &str, epoch: usize, values: &[f64]) {
        let epoch_dir = run_dir.join(split).join(format!("epoch-{epoch}"));
        fs::create_dir_all(&epoch_dir).expect("create epoch dir");
        let path = epoch_dir.join("Loss.log");
        let content = values
            .iter()
            .map(|value| format!("{value},1"))
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(path, content).expect("write metric log");
    }

    fn apply_checkpoint_actions(run_dir: &Path, epoch: usize, actions: &[CheckpointingAction]) {
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        for action in actions {
            match action {
                CheckpointingAction::Save => {
                    for prefix in ["model", "optim", "scheduler"] {
                        fs::write(
                            checkpoint_dir.join(format!("{prefix}-{epoch}.bin")),
                            format!("{prefix}-{epoch}"),
                        )
                        .expect("write checkpoint file");
                    }
                }
                CheckpointingAction::Delete(epoch) => {
                    for prefix in ["model", "optim", "scheduler"] {
                        let path = checkpoint_dir.join(format!("{prefix}-{epoch}.bin"));
                        if path.exists() {
                            fs::remove_file(path).expect("remove checkpoint file");
                        }
                    }
                }
            }
        }
    }

    fn retained_model_epochs(run_dir: &Path) -> Vec<usize> {
        let checkpoint_dir = run_dir.join("checkpoint");
        let mut epochs = fs::read_dir(&checkpoint_dir)
            .expect("read checkpoint dir")
            .filter_map(|entry| {
                let path = entry.ok()?.path();
                let name = path.file_name()?.to_str()?;
                let epoch = name
                    .strip_prefix("model-")?
                    .strip_suffix(".bin")?
                    .parse::<usize>()
                    .ok()?;
                Some(epoch)
            })
            .collect::<Vec<_>>();
        epochs.sort_unstable();
        epochs
    }

    #[test]
    fn file_metric_best_strategy_preserves_old_best_outside_keep_last_window() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let means = [
            2.0, 1.9, 1.8, 1.7, 1.6, 1.55, 1.53, 1.52, 1.515, 1.51, 1.509, 1.508, 1.507, 1.506,
            1.505, 1.504, 1.503, 1.502, 1.497, 1.501, 1.510, 1.512, 1.511, 1.499, 1.513, 1.514,
            1.502, 1.520, 1.506, 1.530,
        ];

        for (index, mean) in means.iter().enumerate() {
            let epoch = index + 1;
            write_metric_log(dir.path(), "valid", epoch, &[*mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(19));
        assert_eq!(retained_model_epochs(dir.path()), vec![19, 29, 30]);
    }

    #[test]
    fn file_metric_best_strategy_deletes_old_best_after_replacement() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 2.0), (3, 2.5), (4, 1.5)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        assert_eq!(strategy.best_epoch, Some(4));
        assert_eq!(retained_model_epochs(dir.path()), vec![3, 4]);
    }

    #[test]
    fn file_metric_best_strategy_rehydrates_history_when_resuming() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for (epoch, mean) in [(1, 3.0), (2, 1.5), (3, 2.0), (4, 2.1), (5, 2.2), (6, 2.3)] {
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
        }
        for epoch in [2, 5, 6] {
            apply_checkpoint_actions(dir.path(), epoch, &[CheckpointingAction::Save]);
        }

        write_metric_log(dir.path(), "valid", 7, &[2.4]);
        let actions = strategy.actions_for_epoch(7);
        apply_checkpoint_actions(dir.path(), 7, &actions);

        assert_eq!(strategy.best_epoch, Some(2));
        assert_eq!(retained_model_epochs(dir.path()), vec![2, 6, 7]);
    }

    #[test]
    fn file_metric_best_strategy_recomputes_history_when_new_best_log_arrives_late() {
        let dir = tempfile::tempdir().expect("tempdir");
        let mut strategy = FileMetricBestCheckpointingStrategy::new(
            dir.path(),
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        for epoch in 1..=23 {
            let mean = if epoch == 23 {
                1.50
            } else {
                2.0 + epoch as f64 * 0.01
            };
            write_metric_log(dir.path(), "valid", epoch, &[mean]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        for epoch in 24..=28 {
            write_metric_log(dir.path(), "valid", epoch, &[1.60 + epoch as f64 * 0.001]);
            let actions = strategy.actions_for_epoch(epoch);
            apply_checkpoint_actions(dir.path(), epoch, &actions);
        }

        let actions = strategy.actions_for_epoch(29);
        apply_checkpoint_actions(dir.path(), 29, &actions);
        write_metric_log(dir.path(), "valid", 29, &[1.48]);

        write_metric_log(dir.path(), "valid", 30, &[1.49]);
        let actions = strategy.actions_for_epoch(30);
        apply_checkpoint_actions(dir.path(), 30, &actions);

        assert_eq!(strategy.best_epoch, Some(29));
        assert_eq!(retained_model_epochs(dir.path()), vec![29, 30]);
    }

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

    #[test]
    fn train_with_scheduler_retains_best_valid_and_last_checkpoints() {
        let dir = tempfile::tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");

        let parallel_config = burn_dragon_train::ParallelConfig::default();
        let parallel_runtime =
            resolve_parallel_runtime(&parallel_config).expect("resolve single runtime");

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

        let training = tiny_training_hparams_with_epochs(4, None);
        let model_config = tiny_model_config();
        let devices = vec![primary_device.clone()];
        let env = TrainEnvironment {
            parallel_runtime: &parallel_runtime,
            parallel_config: &parallel_config,
            run_dir: &run_dir,
            run_name: "single-retention-smoke",
            backend_name: "cpu",
            training: &training,
            resume_checkpoint_epoch: None,
            model_config: &model_config,
            device: &primary_device,
            devices: &devices,
            train_loader: Arc::new(StaticSequenceLoader::new(train_batches)),
            valid_loader: Arc::new(StaticSequenceLoader::new(valid_batches)),
            epochs: 4,
        };
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            model_config.clone(),
            &primary_device,
        ));
        let optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<TestBackend, LanguageTrainModel<TestBackend>>();

        let _trained =
            train_with_scheduler(&env, model, optimizer, 1e-3).expect("single-device train");

        let strategy = FileMetricBestCheckpointingStrategy::new(
            &run_dir,
            &LossMetric::<TestValidBackend>::new(),
            burn_train::metric::store::Direction::Lowest,
            burn_train::metric::store::Split::Valid,
        );

        let best_epoch = (1..=4)
            .map(|epoch| {
                (
                    epoch,
                    strategy
                        .metric_mean_from_log(epoch)
                        .expect("metric log for every epoch"),
                )
            })
            .min_by(|left, right| left.1.total_cmp(&right.1))
            .map(|(epoch, _)| epoch)
            .expect("best epoch");

        let retained = retained_model_epochs(&run_dir);
        let mut expected = vec![best_epoch, 3, 4];
        expected.sort_unstable();
        expected.dedup();

        assert_eq!(retained, expected);
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
    fn pipeline_stage_surrogate_backward_matches_full_pipeline_gradients() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let mut config = tiny_model_config();
        config.n_layer = 2;
        let pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        let plan = build_pipeline_plan(config.n_layer, &pipeline).expect("plan");
        let reference_model =
            LanguageTrainModel::new(BDH::<TestBackend>::new(config.clone(), &device))
                .with_pipeline_plan(Some(plan.clone()));
        let split_model = reference_model.clone();

        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let reference_grads = flatten_gradients_in_module_order::<TestBackend, _>(
            &reference_model,
            burn_train::TrainStep::step(&reference_model, batch.clone()).grads,
        );

        let [batch_size, _] = batch.inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches).expect("ranges");
        let chunk_inputs = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.inputs.clone(), range))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .cloned()
            .map(|range| slice_batch_int(batch.targets.clone(), range))
            .collect::<Vec<_>>();
        let mut chunk_states = (0..plan.microbatches)
            .map(|_| split_model.model.init_state())
            .collect::<Vec<_>>();
        let mut accumulator = GradientsAccumulator::new();

        for microbatch_id in 0..plan.microbatches {
            let stage0_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    split_model
                        .model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                    &mut chunk_states[microbatch_id],
                    plan.assignment(0).layer_range.clone(),
                    None,
                );
            let stage1_input = attach_pipeline_state_require_grad::<TestBackend>(
                detach_pipeline_state_to_inner(&stage0_output),
            );
            let stage1_input_for_grad = stage1_input.clone();
            let stage1_output = split_model
                .model
                .forward_language_pipeline_stage_with_state(
                    stage1_input,
                    &mut chunk_states[microbatch_id],
                    plan.assignment(1).layer_range.clone(),
                    None,
                );
            let (_hidden, logits) = split_model.model.finish_language_pipeline_with_state(
                stage1_output,
                &mut chunk_states[microbatch_id],
            );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let loss =
                language_model_loss::<TestBackend>(logits, chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
            let mut stage1_grads = loss.backward();
            let grad_to_stage0 =
                pipeline_input_grad_state(&stage1_input_for_grad, &mut stage1_grads);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage1_grads, &split_model),
            );

            let stage0_surrogate = pipeline_surrogate_loss(&stage0_output, grad_to_stage0);
            accumulator.accumulate(
                &split_model,
                GradientsParams::from_grads(stage0_surrogate.backward(), &split_model),
            );
        }

        let split_grads =
            flatten_gradients_in_module_order::<TestBackend, _>(&split_model, accumulator.grads());
        let mean_abs = mean_abs_diff(&reference_grads, &split_grads);
        let reference_norm = l2_norm(&reference_grads);
        let split_norm = l2_norm(&split_grads);

        assert!(
            mean_abs <= 1.0e-5,
            "surrogate split pipeline gradients drifted from full pipeline reference: mean_abs_diff={mean_abs}"
        );
        assert!(
            (reference_norm - split_norm).abs() <= 1.0e-5,
            "split pipeline gradient norm drifted from reference: reference_norm={reference_norm} split_norm={split_norm}"
        );
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
    fn process_group_data_shard_uses_data_parallel_rank_when_pipeline_enabled() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            global_rank: 3,
            local_rank: 1,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };
        let config = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Ddp,
            world_size: 4,
            data: burn_dragon_train::ParallelDataConfig {
                size: 2,
                ..Default::default()
            },
            pipeline: burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let (shard_index, shard_count, assignment, layout) =
            process_group_data_shard(&runtime, &config).expect("pipeline shard");

        assert_eq!(shard_index, 1);
        assert_eq!(shard_count, 2);
        let assignment = assignment.expect("rank assignment");
        let layout = layout.expect("layout");
        assert_eq!(assignment.pipeline_stage_id, 1);
        assert_eq!(assignment.data_parallel_rank, 1);
        assert_eq!(assignment.pipeline_group_ranks, vec![2, 3]);
        assert_eq!(assignment.data_parallel_group_ranks, vec![1, 3]);
        assert_eq!(
            layout.summary(),
            "pipeline_layout=replica_major stage_count=2 virtual_stages_per_rank=1 data_parallel_size=2 world_size=4"
        );
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
