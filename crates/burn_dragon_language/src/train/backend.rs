use crate::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV};
use crate::train::prelude::*;
use crate::train::schedule::{
    TrainEnvironment, resolve_lr_scheduler, resolve_train_schedule, train_with_scheduler,
};
use crate::train::startup_autotune::{
    resolve_gradient_accumulation_steps, resolve_startup_batch_size,
};
use crate::train::utils::write_run_config;
use crate::write_training_snapshot;
use std::time::Instant;
use tracing::warn;

const PROCESS_GROUP_RUN_DIR_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_DIR";
const PROCESS_GROUP_RUN_NAME_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_NAME";

fn resolve_run_root() -> PathBuf {
    crate::checkpoint::resolve_run_root()
}

fn resolve_checkpoint_steps_per_epoch(
    training: &TrainingHyperparameters,
    dataset_steps_per_epoch: usize,
) -> usize {
    match training.epochs {
        Some(_) => dataset_steps_per_epoch.max(1),
        None => training
            .checkpoint_interval_iters
            .min(training.max_iters.max(1))
            .max(1),
    }
}

fn derive_run_name(run_dir: &Path) -> Result<String> {
    run_dir
        .file_name()
        .and_then(|name| name.to_str())
        .map(ToOwned::to_owned)
        .ok_or_else(|| anyhow!("failed to derive run name from {}", run_dir.display()))
}

fn resolve_run_artifacts(
    parallel_runtime: &ParallelRuntime,
    run_root: &Path,
    training: &TrainingHyperparameters,
) -> Result<(PathBuf, String)> {
    if let Some(resume_run_dir) = &training.resume_run_dir {
        let run_dir = resume_run_dir.clone();
        let run_name = derive_run_name(&run_dir)?;
        if !parallel_runtime.is_process_group_launch() {
            if !run_dir.is_dir() {
                return Err(anyhow!(
                    "training.resume_run_dir does not exist or is not a directory: {}",
                    run_dir.display()
                ));
            }
            return Ok((run_dir, run_name));
        }
        let env_run_dir = std::env::var_os(PROCESS_GROUP_RUN_DIR_ENV)
            .map(PathBuf::from)
            .ok_or_else(|| {
                anyhow!(
                    "parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_DIR_ENV}"
                )
            })?;
        let env_run_name = std::env::var(PROCESS_GROUP_RUN_NAME_ENV).map_err(|_| {
            anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_NAME_ENV}")
        })?;
        if env_run_dir != run_dir || env_run_name != run_name {
            return Err(anyhow!(
                "process-group resume requires launcher env run_dir/run_name to match training.resume_run_dir (env={} name={}, resume={} name={})",
                env_run_dir.display(),
                env_run_name,
                run_dir.display(),
                run_name
            ));
        }
        return Ok((run_dir, run_name));
    }

    let env_run_dir = std::env::var_os(RUN_DIR_ENV).map(PathBuf::from);
    let env_run_name = std::env::var(RUN_NAME_ENV).ok();
    if !parallel_runtime.is_process_group_launch() {
        match (env_run_dir, env_run_name) {
            (Some(run_dir), Some(run_name)) => {
                fs::create_dir_all(&run_dir).with_context(|| {
                    format!(
                        "failed to create preassigned run directory {}",
                        run_dir.display()
                    )
                })?;
                return Ok((run_dir, run_name));
            }
            (None, None) => return create_run_dir(run_root),
            _ => {
                return Err(anyhow!(
                    "single-process launches require both {RUN_DIR_ENV} and {RUN_NAME_ENV} when either one is set"
                ));
            }
        }
    }

    let run_dir = std::env::var_os(PROCESS_GROUP_RUN_DIR_ENV)
        .map(PathBuf::from)
        .ok_or_else(|| {
            anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_DIR_ENV}")
        })?;
    let run_name = std::env::var(PROCESS_GROUP_RUN_NAME_ENV).map_err(|_| {
        anyhow!("parallel.mode=ddp process-group launches require {PROCESS_GROUP_RUN_NAME_ENV}")
    })?;
    Ok((run_dir, run_name))
}

fn resolve_resume_checkpoint_epoch(
    training: &TrainingHyperparameters,
    run_dir: &Path,
) -> Result<Option<usize>> {
    let Some(_) = training.resume_run_dir else {
        return Ok(None);
    };
    let checkpoint_dir = run_dir.join("checkpoint");
    let (_, epoch) = crate::checkpoint::resolve_checkpoint_base(
        &checkpoint_dir,
        training.resume_checkpoint_epoch,
    )
    .with_context(|| {
        format!(
            "failed to resolve resume checkpoint in {}",
            checkpoint_dir.display()
        )
    })?;
    Ok(Some(epoch))
}

fn initialize_model_from_checkpoint<B: BackendTrait>(
    training: &TrainingHyperparameters,
    model: &mut BDH<B>,
    device: &B::Device,
) -> Result<()> {
    let Some(checkpoint_path) = &training.init_checkpoint_path else {
        return Ok(());
    };
    let (checkpoint_base, epoch) =
        crate::checkpoint::resolve_checkpoint_base(checkpoint_path, training.init_checkpoint_epoch)
            .with_context(|| {
                format!(
                    "failed to resolve init checkpoint from {}",
                    checkpoint_path.display()
                )
            })?;
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .with_context(|| {
            format!(
                "failed to load init checkpoint epoch {epoch} from {}",
                checkpoint_base.display()
            )
        })?;
    *model = model.clone().load_record(record);
    info!(
        "initialized model weights from checkpoint epoch {epoch} at {}",
        checkpoint_base.display()
    );
    Ok(())
}

pub fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + 'static,
    Init: Fn(&B::Device),
{
    let stage_profile = crate::train::profile::enabled();
    if stage_profile {
        crate::train::profile::reset();
    }
    let train_wall_start = stage_profile.then(Instant::now);

    let parallel_runtime = resolve_parallel_runtime(&config.parallel)?;
    info!("parallel runtime: {}", parallel_runtime.summary());

    let primary_device = B::Device::default();
    let devices = resolve_training_devices::<B>(&parallel_runtime, &primary_device)?;
    for device in &devices {
        B::seed(device, config.training.seed);
        init_backend(device);
    }
    let device = devices
        .first()
        .cloned()
        .expect("at least one training device");
    info!("resolved training devices: {}", devices.len());

    let mut resolved_config = config.clone();
    let startup_autotune =
        resolve_startup_batch_size::<B>(&resolved_config, &dataset, backend_name, &device)?;
    if let Some(report) = &startup_autotune {
        resolved_config.training.batch_size = report.resolved_batch_size;
        resolved_config.training.gradient_accumulation_steps =
            report.resolved_gradient_accumulation_steps;
    }
    if startup_autotune.is_none() {
        resolved_config.training.gradient_accumulation_steps = resolve_gradient_accumulation_steps(
            resolved_config.training.batch_size,
            resolved_config.training.gradient_accumulation_steps,
            resolved_config.training.target_effective_batch_size,
        );
    }

    let datasets = if resolved_config.training.batch_size == config.training.batch_size {
        crate::train::utils::PreparedDatasets {
            train: Arc::clone(&dataset),
            valid: Arc::clone(&dataset),
        }
    } else {
        crate::train::utils::prepare_datasets(&resolved_config.dataset, &resolved_config.training)?
    };

    let training = &resolved_config.training;
    let optimizer_cfg = &config.optimizer;

    let tokenizer = datasets.train.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &resolved_config.model,
        training.block_size,
        tokenizer.as_ref(),
    )?;
    if let Some(sequence_kernel) = training.sequence_kernel_override {
        model_config.sequence_kernel = sequence_kernel;
    }
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        resolved_config.wgpu.training.fused_core_recurrent,
        resolved_config.wgpu.training.fused_core_rollout,
    );
    if backend_name.eq_ignore_ascii_case("cuda") && model_config.fused_kernels.enabled {
        warn!(
            "cuda language training still mixes burn_dragon_kernel fused kernels with generic Burn tensor ops; only selected recurrent/projection paths are accelerated today"
        );
    }
    let pipeline_plan = if resolved_config.parallel.pipeline.enabled {
        let pipeline_plan =
            build_pipeline_plan(model_config.n_layer, &resolved_config.parallel.pipeline)?;
        info!("resolved pipeline plan: {}", pipeline_plan.summary());
        if resolved_config.parallel.pipeline.communication
            == burn_dragon_train::PipelineCommunicationKind::BlockResidualCache
            && resolved_config.model.residual_connector
                == Some(burn_dragon_core::ResidualConnectorKind::BlockAttentionResidual)
        {
            let layers_per_block = resolved_config
                .model
                .block_attention_residual
                .as_ref()
                .map(|cfg| cfg.layers_per_block.max(1))
                .unwrap_or(1);
            let payload_bytes = model_config
                .n_embd
                .saturating_mul(training.block_size.max(1))
                .saturating_mul(std::mem::size_of::<f32>());
            let communication = simulate_pipeline_communication(
                &pipeline_plan,
                resolved_config.parallel.pipeline.communication,
                &resolved_config.parallel.pipeline.cache,
                layers_per_block,
                payload_bytes,
            )?;
            info!(
                "resolved pipeline communication: requested_bytes={} transmitted_bytes={} bytes_saved={} cache_hits={} cache_misses={} backward_reuse_hits={} hit_rate={:.3}",
                communication.raw_payload_bytes_requested,
                communication.payload_bytes_transmitted,
                communication.bytes_saved(),
                communication.cache_hits,
                communication.cache_misses,
                communication.backward_reuse_hits,
                communication.cache_hit_rate(),
            );
        }
        if parallel_runtime.mode != ParallelismKind::Single {
            let layout =
                resolve_pipeline_parallel_layout(&parallel_runtime, &resolved_config.parallel)?
                    .ok_or_else(|| {
                        anyhow!("parallel.pipeline.enabled requires a resolved DDP pipeline layout")
                    })?;
            let assignment = layout.assignment(parallel_runtime.global_rank).clone();
            let workload = build_pipeline_rank_workload(
                &pipeline_plan,
                assignment.global_rank,
                assignment.pipeline_stage_id,
                assignment.data_parallel_rank,
            );
            info!(
                "resolved distributed pipeline rank workload: {} rank={} stage={} dp_rank={} assignments={} forward_events={} backward_events={}",
                layout.summary(),
                assignment.global_rank,
                assignment.pipeline_stage_id,
                assignment.data_parallel_rank,
                workload.stage_assignments.len(),
                workload.forward_events.len(),
                workload.backward_events.len(),
            );
            if parallel_runtime.mode != ParallelismKind::Ddp
                || !parallel_runtime.is_process_group_launch()
            {
                return Err(anyhow!(
                    "parallel.pipeline.enabled distributed execution currently requires a process-group DDP launch"
                ));
            }
            if layout.data_parallel_size > 1 {
                return Err(anyhow!(
                    "parallel.pipeline.enabled process-group execution is currently implemented for layer pipeline parallelism with parallel.data.size = 1; resolved {}",
                    layout.summary(),
                ));
            }
        }
        if training.tbptt_chunk_size.is_some() || training.tbptt_persist_across_steps {
            return Err(anyhow!(
                "parallel.pipeline.enabled does not yet support tbptt chunking or persistent stream state"
            ));
        }
        if model_config.rollout_fast_steps_per_slow_step != 1 {
            return Err(anyhow!(
                "parallel.pipeline.enabled requires rollout_fast_steps_per_slow_step = 1 (got {})",
                model_config.rollout_fast_steps_per_slow_step
            ));
        }
        if model_config.y_neuron_recurrence.enabled {
            return Err(anyhow!(
                "parallel.pipeline.enabled does not yet support y_neuron_recurrence"
            ));
        }
        Some(pipeline_plan)
    } else {
        None
    };
    let summary_event_token_ids = model_config.summary_memory.write_trigger_token_ids.clone();

    let dataset_steps_per_epoch = datasets.train.steps_per_epoch(DatasetSplit::Train);
    let checkpoint_steps_per_epoch =
        resolve_checkpoint_steps_per_epoch(training, dataset_steps_per_epoch);
    let schedule = resolve_train_schedule(training, checkpoint_steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "train schedule: dataset_steps_per_epoch={dataset_steps_per_epoch}, logical_steps_per_epoch={steps_per_epoch}, checkpoint_interval_iters={}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        training.checkpoint_interval_iters,
        schedule.source.as_str()
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> =
        if training.tbptt_persist_across_steps {
            Arc::new(
                StreamingDataLoader::<B>::new(
                    Arc::clone(&datasets.train),
                    DatasetSplit::Train,
                    &device,
                    steps_per_epoch,
                    Some(total_steps),
                    training.min_logical_block_size,
                    training.seed,
                )
                .with_summary_event_token_ids(summary_event_token_ids.clone()),
            )
        } else {
            Arc::new(
                RandomDataLoader::<B>::new(
                    Arc::clone(&datasets.train),
                    DatasetSplit::Train,
                    &device,
                    steps_per_epoch,
                    Some(total_steps),
                )
                .with_summary_event_token_ids(summary_event_token_ids.clone()),
            )
        };

    let val_steps_per_epoch = datasets.valid.steps_per_epoch(DatasetSplit::Val);
    let valid_steps =
        resolve_valid_steps_per_epoch(total_steps, training.log_frequency, val_steps_per_epoch);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>> =
        Arc::new(
            RandomDataLoader::<ValidBackend<B>>::new(
                Arc::clone(&datasets.valid),
                DatasetSplit::Val,
                &valid_device,
                valid_steps,
                None,
            )
            .with_summary_event_token_ids(summary_event_token_ids),
        );

    let mut base_model = BDH::<B>::new(model_config.clone(), &device);
    initialize_model_from_checkpoint(training, &mut base_model, &device)?;
    let mut model = Some(
        LanguageTrainModel::new(base_model)
            .with_pipeline_plan(pipeline_plan.clone())
            .with_tbptt_chunk_size(training.tbptt_chunk_size)
            .with_tbptt_persist_across_steps(training.tbptt_persist_across_steps),
    );
    let mut optim =
        Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, LanguageTrainModel<B>>());
    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &model_config)?;

    let run_root = resolve_run_root();
    let (run_dir, run_name) = resolve_run_artifacts(&parallel_runtime, &run_root, training)?;
    let resume_checkpoint_epoch = resolve_resume_checkpoint_epoch(training, &run_dir)?;
    if parallel_runtime.is_primary() {
        write_latest_run(&run_root, &run_name)?;
        write_run_config(
            &resolved_config,
            &model_config,
            &run_dir,
            &run_name,
            backend_name,
            startup_autotune.as_ref(),
        )?;
        write_training_snapshot(&resolved_config, &run_dir, dataset.tokenizer().as_ref())?;
    }
    info!("run name: {run_name}");
    if let Some(report) = &startup_autotune {
        info!(
            "startup autotune: backend={} target_device_memory_mb={} resolved_batch_size={} resolved_gradient_accumulation_steps={} resolved_effective_batch_size={} probes={}",
            report.backend_name,
            report.target_device_memory_mb,
            report.resolved_batch_size,
            report.resolved_gradient_accumulation_steps,
            report.resolved_effective_batch_size,
            report
                .probes
                .iter()
                .map(|probe| match (probe.reserved_mb, probe.in_use_mb) {
                    (Some(reserved), Some(in_use)) => format!(
                        "bs{}:{}:{reserved:.1}/{in_use:.1}MiB",
                        probe.batch_size, probe.status
                    ),
                    _ => format!("bs{}:{}", probe.batch_size, probe.status),
                })
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    info!(
        "training batching: micro_batch_size={} gradient_accumulation_steps={} effective_batch_size={} tbptt_chunk_size={} tbptt_persist_across_steps={} min_logical_block_size={}",
        resolved_config.training.batch_size,
        resolved_config.training.gradient_accumulation_steps,
        resolved_config
            .training
            .batch_size
            .saturating_mul(resolved_config.training.gradient_accumulation_steps),
        resolved_config
            .training
            .tbptt_chunk_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string()),
        resolved_config.training.tbptt_persist_across_steps,
        resolved_config
            .training
            .min_logical_block_size
            .map(|value| value.to_string())
            .unwrap_or_else(|| "disabled".to_string())
    );
    let context = TrainEnvironment {
        parallel_runtime: &parallel_runtime,
        parallel_config: &resolved_config.parallel,
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        resume_checkpoint_epoch,
        model_config: &model_config,
        device: &device,
        devices: &devices,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };
    let _model = match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
    };

    info!("Training complete on {backend_name}");

    if let Some(start) = train_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let snapshot = crate::train::profile::snapshot();
        info!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.loss_backward_ns,
            snapshot.embed_probe_ns,
            snapshot.first_layer_forward_probe_ns,
            snapshot.first_layer_probe_ns,
            snapshot.logits_loss_probe_ns,
            snapshot.hidden_logits_loss_probe_ns,
            snapshot.hidden_model_forward_probe_ns,
            snapshot.hidden_model_probe_ns,
            snapshot.detail_probe_steps,
            snapshot.train_steps,
        );
        eprintln!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.loss_backward_ns,
            snapshot.embed_probe_ns,
            snapshot.first_layer_forward_probe_ns,
            snapshot.first_layer_probe_ns,
            snapshot.logits_loss_probe_ns,
            snapshot.hidden_logits_loss_probe_ns,
            snapshot.hidden_model_forward_probe_ns,
            snapshot.hidden_model_probe_ns,
            snapshot.detail_probe_steps,
            snapshot.train_steps,
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV};
    #[cfg(feature = "ddp")]
    use crate::checkpoint::{
        load_language_core_from_checkpoint, load_training_snapshot_from_run_dir,
    };
    #[cfg(feature = "ddp")]
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;
    use std::env;
    use std::fs;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    type TestBackend = Autodiff<NdArray<f32>>;
    type InferenceBackend = NdArray<f32>;

    fn cwd_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    struct CurrentDirGuard {
        previous: PathBuf,
    }

    impl CurrentDirGuard {
        fn enter(path: &Path) -> Self {
            let previous = std::env::current_dir().expect("current dir");
            std::env::set_current_dir(path).expect("set current dir");
            Self { previous }
        }
    }

    impl Drop for CurrentDirGuard {
        fn drop(&mut self) {
            let _ = std::env::set_current_dir(&self.previous);
        }
    }

    fn tiny_training_config(cache_dir: &Path) -> TrainingConfig {
        TrainingConfig {
            dataset: DatasetConfig {
                cache_dir: cache_dir.to_path_buf(),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 8,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 4,
                seed: 1337,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 1,
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
            },
            optimizer: OptimizerConfig {
                learning_rate: 1e-3,
                weight_decay: 0.0,
                lr_schedule: None,
                grad_clip_norm: None,
                grad_clip_value: None,
            },
            parallel: burn_dragon_train::ParallelConfig {
                mode: ParallelismKind::Ddp,
                world_size: 2,
                data: burn_dragon_train::ParallelDataConfig {
                    size: 2,
                    ..Default::default()
                },
                ..Default::default()
            },
            generation: GenerationConfig {
                prompt: String::new(),
                max_tokens: Some(1),
                max_chars: None,
                temperature: 1.0,
                top_k: None,
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: WgpuRuntimeConfig::default(),
            model: ModelOverrides {
                n_layer: Some(1),
                n_embd: Some(8),
                n_head: Some(1),
                mlp_internal_dim_multiplier: Some(1),
                dropout: Some(0.0),
                ..Default::default()
            },
        }
    }

    #[test]
    fn checkpoint_steps_per_epoch_uses_interval_for_max_iters_runs() {
        let training = TrainingHyperparameters {
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 4,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: None,
            max_iters: 6_000,
            checkpoint_interval_iters: 2_000,
            log_frequency: 100,
            fast_train: false,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        };

        assert_eq!(
            resolve_checkpoint_steps_per_epoch(&training, 31_228_052),
            2_000
        );
    }

    #[test]
    fn checkpoint_steps_per_epoch_uses_dataset_epoch_for_epoch_runs() {
        let training = TrainingHyperparameters {
            block_size: 8,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 4,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 6_000,
            checkpoint_interval_iters: 2_000,
            log_frequency: 100,
            fast_train: false,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        };

        assert_eq!(resolve_checkpoint_steps_per_epoch(&training, 512), 512);
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_backend_local_ddp_writes_reloadable_checkpoint() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(512),
        )
        .expect("write tiny shakespeare");

        let config = tiny_training_config(&cache_dir);
        let dataset = crate::train::utils::prepare_dataset(&config.dataset, &config.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        train_backend::<TestBackend, _>(&config, dataset, "cpu", |_| {}).expect("train backend");

        let latest = fs::read_to_string(dir.path().join("runs/latest")).expect("read latest");
        let run_dir = dir.path().join("runs").join(latest.trim());
        assert!(run_dir.join("config.json").is_file(), "expected run config");
        assert!(
            run_dir.join("training_config.json").is_file(),
            "expected training snapshot"
        );
        assert!(
            run_dir.join("checkpoint").is_dir(),
            "expected checkpoint directory"
        );

        let snapshot =
            load_training_snapshot_from_run_dir(&run_dir).expect("load training snapshot");
        assert_eq!(snapshot.parallel.mode, ParallelismKind::Ddp);
        assert_eq!(snapshot.parallel.data.size, 2);

        let device = <InferenceBackend as BackendTrait>::Device::default();
        let model = load_language_core_from_checkpoint::<InferenceBackend>(
            &run_dir.join("checkpoint"),
            Some(1),
            &[],
            "cpu",
            &device,
        )
        .expect("reload checkpoint");
        let logits = model.forward(Tensor::<InferenceBackend, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7], [1, 8]),
            &device,
        ));
        let values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits vec");
        assert!(!values.is_empty());
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn train_backend_max_iters_checkpoints_on_logical_intervals() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(512),
        )
        .expect("write tiny shakespeare");

        let mut config = tiny_training_config(&cache_dir);
        config.parallel = burn_dragon_train::ParallelConfig::default();
        config.training.max_iters = 3;
        config.training.checkpoint_interval_iters = 1;
        let dataset = crate::train::utils::prepare_dataset(&config.dataset, &config.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        train_backend::<TestBackend, _>(&config, dataset, "cpu", |_| {}).expect("train backend");

        let latest = fs::read_to_string(dir.path().join("runs/latest")).expect("read latest");
        let run_dir = dir.path().join("runs").join(latest.trim());
        let checkpoint_dir = run_dir.join("checkpoint");
        assert!(checkpoint_dir.join("model-2.bin").is_file());
        assert!(checkpoint_dir.join("model-3.bin").is_file());
    }

    #[test]
    fn train_backend_single_process_pipeline_runs_and_writes_checkpoint() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(64),
        )
        .expect("write tiny shakespeare");

        let mut config = tiny_training_config(&cache_dir);
        config.parallel = burn_dragon_train::ParallelConfig {
            mode: ParallelismKind::Single,
            world_size: 1,
            pipeline: burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
                microbatches: 2,
                ..Default::default()
            },
            ..Default::default()
        };
        config.model.n_layer = Some(2);
        let dataset = crate::train::utils::prepare_dataset(&config.dataset, &config.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        train_backend::<TestBackend, _>(&config, dataset, "cpu", |_| {})
            .expect("single-process pipeline train backend");
        let latest = fs::read_to_string(dir.path().join("runs/latest")).expect("read latest");
        let run_dir = dir.path().join("runs").join(latest.trim());
        assert!(run_dir.join("config.json").is_file(), "expected run config");
        assert!(
            run_dir.join("checkpoint").join("model-1.bin").is_file(),
            "expected checkpoint"
        );
    }

    #[test]
    fn train_backend_rejects_pipeline_for_non_single_runtime() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(64),
        )
        .expect("write tiny shakespeare");

        let mut config = tiny_training_config(&cache_dir);
        config.parallel.data.size = 1;
        config.parallel.pipeline = burn_dragon_train::ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
            microbatches: 2,
            ..Default::default()
        };
        config.model.n_layer = Some(2);
        let dataset = crate::train::utils::prepare_dataset(&config.dataset, &config.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        let err = train_backend::<TestBackend, _>(&config, dataset, "cpu", |_| {})
            .expect_err("ddp pipeline runtime should fail explicitly");
        assert!(
            err.to_string()
                .contains("parallel.pipeline.enabled distributed execution currently requires a process-group DDP launch"),
            "unexpected error: {err:#}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn train_backend_resume_run_dir_reuses_checkpoint_family() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(64),
        )
        .expect("write tiny shakespeare");

        let mut initial = tiny_training_config(&cache_dir);
        initial.parallel = burn_dragon_train::ParallelConfig::default();
        initial.training.epochs = Some(1);
        let dataset = crate::train::utils::prepare_dataset(&initial.dataset, &initial.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        train_backend::<TestBackend, _>(&initial, Arc::clone(&dataset), "cpu", |_| {})
            .expect("initial train backend");

        let latest = fs::read_to_string(dir.path().join("runs/latest")).expect("read latest");
        let run_dir = dir.path().join("runs").join(latest.trim());
        assert!(run_dir.join("checkpoint").join("model-1.bin").is_file());
        assert!(run_dir.join("checkpoint").join("optim-1.bin").is_file());
        assert!(run_dir.join("checkpoint").join("scheduler-1.bin").is_file());

        let mut resumed = initial.clone();
        resumed.training.epochs = Some(2);
        resumed.training.resume_run_dir = Some(run_dir.clone());
        resumed.training.resume_checkpoint_epoch = Some(1);
        train_backend::<TestBackend, _>(&resumed, dataset, "cpu", |_| {})
            .expect("resumed train backend");

        assert!(run_dir.join("checkpoint").join("model-2.bin").is_file());
        assert!(run_dir.join("checkpoint").join("optim-2.bin").is_file());
        assert!(run_dir.join("checkpoint").join("scheduler-2.bin").is_file());

        let device = <InferenceBackend as BackendTrait>::Device::default();
        let model = load_language_core_from_checkpoint::<InferenceBackend>(
            &run_dir.join("checkpoint"),
            Some(2),
            &[],
            "cpu",
            &device,
        )
        .expect("reload resumed checkpoint");
        let logits = model.forward(Tensor::<InferenceBackend, 2, Int>::from_data(
            TensorData::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7], [1, 8]),
            &device,
        ));
        let values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logits vec");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn train_backend_init_checkpoint_bootstraps_new_run() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let dir = tempdir().expect("tempdir");
        let cache_dir = dir.path().join("cache");
        fs::create_dir_all(&cache_dir).expect("cache dir");
        fs::write(
            cache_dir.join("tinyshakespeare.txt"),
            b"Once more unto the breach, dear friends, once more.\n".repeat(128),
        )
        .expect("write tiny shakespeare");

        let mut initial = tiny_training_config(&cache_dir);
        initial.parallel = burn_dragon_train::ParallelConfig::default();
        let dataset = crate::train::utils::prepare_dataset(&initial.dataset, &initial.training)
            .expect("prepare dataset");

        let _cwd = CurrentDirGuard::enter(dir.path());
        train_backend::<TestBackend, _>(&initial, Arc::clone(&dataset), "cpu", |_| {})
            .expect("initial train backend");

        let first_latest = fs::read_to_string(dir.path().join("runs/latest")).expect("read latest");
        let first_run_dir = dir.path().join("runs").join(first_latest.trim());
        let checkpoint_dir = first_run_dir.join("checkpoint");
        assert!(checkpoint_dir.join("model-1.bin").is_file());

        let mut warmstart = tiny_training_config(&cache_dir);
        warmstart.parallel = burn_dragon_train::ParallelConfig::default();
        warmstart.training.init_checkpoint_path = Some(checkpoint_dir.clone());
        warmstart.training.init_checkpoint_epoch = Some(1);
        train_backend::<TestBackend, _>(&warmstart, dataset, "cpu", |_| {})
            .expect("warmstart train backend");

        let second_latest =
            fs::read_to_string(dir.path().join("runs/latest")).expect("read second latest");
        let second_run_dir = dir.path().join("runs").join(second_latest.trim());
        assert_ne!(first_run_dir, second_run_dir);
        assert!(
            second_run_dir
                .join("checkpoint")
                .join("model-1.bin")
                .is_file()
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_run_artifacts_requires_shared_env_for_process_group_launches() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };

        unsafe {
            env::remove_var(PROCESS_GROUP_RUN_DIR_ENV);
            env::remove_var(PROCESS_GROUP_RUN_NAME_ENV);
        }
        let err = resolve_run_artifacts(
            &runtime,
            Path::new("runs"),
            &tiny_training_config(Path::new("data")).training,
        )
        .expect_err("missing env");
        assert!(
            err.to_string().contains(PROCESS_GROUP_RUN_DIR_ENV),
            "expected missing run-dir env error, got {err:#}"
        );
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_run_artifacts_uses_shared_env_for_process_group_launches() {
        let _cwd_guard = cwd_lock().lock().expect("cwd lock");
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Ddp,
            world_size: 2,
            global_rank: 1,
            local_rank: 1,
            data_parallel_size: 2,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: true,
        };
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("shared-run");

        unsafe {
            env::set_var(PROCESS_GROUP_RUN_DIR_ENV, &run_dir);
            env::set_var(PROCESS_GROUP_RUN_NAME_ENV, "shared-run");
        }
        let (resolved_dir, resolved_name) = resolve_run_artifacts(
            &runtime,
            Path::new("runs"),
            &tiny_training_config(Path::new("data")).training,
        )
        .expect("shared run artifacts");
        assert_eq!(resolved_dir, run_dir);
        assert_eq!(resolved_name, "shared-run");

        unsafe {
            env::remove_var(PROCESS_GROUP_RUN_DIR_ENV);
            env::remove_var(PROCESS_GROUP_RUN_NAME_ENV);
        }
    }

    #[cfg(feature = "ddp")]
    #[test]
    fn resolve_run_artifacts_prefers_resume_run_dir_for_single_process() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Single,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };
        let dir = tempdir().expect("tempdir");
        let resume_run_dir = dir.path().join("existing-run");
        fs::create_dir_all(&resume_run_dir).expect("resume run dir");
        let mut training = tiny_training_config(Path::new("data")).training;
        training.resume_run_dir = Some(resume_run_dir.clone());

        let (resolved_dir, resolved_name) =
            resolve_run_artifacts(&runtime, Path::new("runs"), &training)
                .expect("resume run artifacts");
        assert_eq!(resolved_dir, resume_run_dir);
        assert_eq!(resolved_name, "existing-run");
    }

    #[test]
    fn resolve_run_artifacts_uses_preassigned_single_process_run_env() {
        let runtime = ParallelRuntime {
            mode: ParallelismKind::Single,
            world_size: 1,
            global_rank: 0,
            local_rank: 0,
            data_parallel_size: 1,
            local_data_parallel_size: 1,
            tensor_parallel_size: 1,
            process_group_launch: false,
        };
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("preassigned-run");
        unsafe {
            env::set_var(RUN_DIR_ENV, &run_dir);
            env::set_var(RUN_NAME_ENV, "preassigned-run");
        }

        let (resolved_dir, resolved_name) = resolve_run_artifacts(
            &runtime,
            Path::new("runs"),
            &tiny_training_config(Path::new("data")).training,
        )
        .expect("preassigned run artifacts");
        assert_eq!(resolved_dir, run_dir);
        assert_eq!(resolved_name, "preassigned-run");

        unsafe {
            env::remove_var(RUN_DIR_ENV);
            env::remove_var(RUN_NAME_ENV);
        }
    }
}
