use crate::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV};
use crate::train::prelude::*;
use crate::train::schedule::{
    TrainEnvironment, resolve_lr_scheduler, resolve_train_schedule, train_with_scheduler,
};
use crate::train::startup_autotune::{
    resolve_gradient_accumulation_steps, resolve_startup_batch_size,
};
use crate::train::utils::{build_training_execution_form, write_run_config};
use crate::write_training_snapshot;
use burn_dragon_core::SequenceMemorySystem;
use std::time::Instant;
use tracing::warn;

const PROCESS_GROUP_RUN_DIR_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_DIR";
const PROCESS_GROUP_RUN_NAME_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_NAME";
const CUDA_LINEAR_DENSE_SCORE_AUTO_BLOCK_LIMIT: usize = 1024;

fn cuda_rwkv8_tensorized_scan_threshold_bytes() -> usize {
    std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(4 * 1024 * 1024 * 1024)
}

fn cuda_rwkv8_tensorized_scratch_bytes(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
) -> usize {
    let bhte = batch
        .saturating_mul(heads)
        .saturating_mul(time)
        .saturating_mul(latent)
        .saturating_mul(embd);
    let bhtl = batch
        .saturating_mul(heads)
        .saturating_mul(time)
        .saturating_mul(latent);

    bhte.saturating_mul(3)
        .saturating_add(bhtl.saturating_mul(2))
        .saturating_mul(std::mem::size_of::<f32>())
}

fn cuda_rwkv8_tensorized_chunk_size(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
) -> usize {
    if let Some(explicit) = std::env::var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_CHUNK")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
    {
        return explicit.min(time.max(1));
    }

    let threshold_bytes = cuda_rwkv8_tensorized_scan_threshold_bytes();
    let mut chunk = time.min(128).max(1);
    while chunk > 1
        && cuda_rwkv8_tensorized_scratch_bytes(batch, heads, chunk, latent, embd) > threshold_bytes
    {
        chunk = chunk.div_ceil(2);
    }
    chunk.max(1)
}

fn cuda_rwkv8_training_geometry_summary(
    model_config: &BDHConfig,
    micro_batch_size: usize,
    training_kernel_block_size: usize,
) -> Option<String> {
    if !matches!(
        model_config.sequence_kernel.memory_system,
        SequenceMemorySystem::Rwkv8StateSpace
    ) {
        return None;
    }

    let heads = model_config.n_head.max(1);
    let latent_total = model_config.latent_total_for_layer(0);
    let latent_per_head = model_config.latent_per_head_for_layer(0);
    let runtime_chunk = cuda_rwkv8_tensorized_chunk_size(
        micro_batch_size,
        heads,
        training_kernel_block_size,
        latent_per_head,
        model_config.n_embd,
    );

    Some(format!(
        "cuda rwkv8 geometry: micro_batch={} kernel_block={} tokens/micro_batch={} latent_total={} latent/head={} nheads={} value_heads=1 embd={} runtime_chunk={}",
        micro_batch_size,
        training_kernel_block_size,
        micro_batch_size.saturating_mul(training_kernel_block_size),
        latent_total,
        latent_per_head,
        heads,
        model_config.n_embd,
        runtime_chunk,
    ))
}

fn cuda_mamba_training_geometry_summary(
    model_config: &BDHConfig,
    micro_batch_size: usize,
    training_kernel_block_size: usize,
) -> Option<String> {
    match model_config.sequence_kernel.memory_system {
        SequenceMemorySystem::Mamba2StateSpaceDuality => {
            let resolved = model_config.mamba.resolve(
                model_config.n_embd,
                SequenceMemorySystem::Mamba2StateSpaceDuality,
            );
            Some(format!(
                "cuda mamba2 geometry: micro_batch={} kernel_block={} tokens/micro_batch={} d_inner={} headdim={} nheads={} ngroups={} d_state={} d_conv={}",
                micro_batch_size,
                training_kernel_block_size,
                micro_batch_size.saturating_mul(training_kernel_block_size),
                resolved.d_inner,
                resolved.headdim,
                resolved.nheads,
                resolved.ngroups,
                resolved.d_state,
                resolved.d_conv,
            ))
        }
        SequenceMemorySystem::Mamba3StateSpaceDuality => {
            let resolved = model_config.mamba.resolve(
                model_config.n_embd,
                SequenceMemorySystem::Mamba3StateSpaceDuality,
            );
            Some(format!(
                "cuda mamba3 geometry: micro_batch={} kernel_block={} tokens/micro_batch={} d_inner={} headdim={} nheads={} ngroups={} d_state={} rope_angles={} chunk_size={}",
                micro_batch_size,
                training_kernel_block_size,
                micro_batch_size.saturating_mul(training_kernel_block_size),
                resolved.d_inner,
                resolved.headdim,
                resolved.nheads,
                resolved.ngroups,
                resolved.d_state,
                resolved.num_rope_angles,
                resolved.chunk_size,
            ))
        }
        _ => None,
    }
}

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

fn train_with_resolved_scheduler<B, O>(
    context: &TrainEnvironment<'_, B>,
    model: LanguageTrainModel<B>,
    optimizer: O,
    scheduler: ResolvedLrScheduler,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    O: Optimizer<LanguageTrainModel<B>, B> + 'static,
{
    match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(context, model, optimizer, lr),
        ResolvedLrScheduler::Cosine(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
        ResolvedLrScheduler::Linear(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
        ResolvedLrScheduler::Exponential(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
        ResolvedLrScheduler::Step(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
        ResolvedLrScheduler::Noam(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
        ResolvedLrScheduler::BitNetTwoStage(scheduler) => {
            train_with_scheduler(context, model, optimizer, scheduler)
        }
    }
}

fn resolve_effective_training_sequence_kernel(
    configured_kernel: SequenceKernelConfig,
    training_override: Option<SequenceKernelConfig>,
    backend_name: &str,
    training_kernel_block_size: usize,
) -> (
    SequenceKernelConfig,
    Option<SequenceKernelConfig>,
    Option<&'static str>,
) {
    if let Some(explicit) = training_override {
        return (explicit, Some(explicit), None);
    }

    if backend_name.eq_ignore_ascii_case("cuda")
        && configured_kernel
            == SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention)
        && training_kernel_block_size <= CUDA_LINEAR_DENSE_SCORE_AUTO_BLOCK_LIMIT
    {
        let promoted = SequenceKernelConfig::dense_score_short_context();
        return (
            promoted,
            Some(promoted),
            Some(
                "auto-promoted short-context CUDA linear-attention training to dense_score_short_context",
            ),
        );
    }

    (configured_kernel, None, None)
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
    let training_kernel_block_size =
        crate::train::utils::effective_training_kernel_block_size(training);

    let tokenizer = datasets.train.tokenizer();
    let mut model_config = build_model_config_with_tokenizer(
        &resolved_config.model,
        training_kernel_block_size,
        tokenizer.as_ref(),
    )?;
    let configured_sequence_kernel = model_config.sequence_kernel;
    let (effective_sequence_kernel, effective_training_sequence_kernel_override, promotion_reason) =
        resolve_effective_training_sequence_kernel(
            configured_sequence_kernel,
            training.sequence_kernel_override,
            backend_name,
            training_kernel_block_size,
        );
    model_config.sequence_kernel = effective_sequence_kernel;
    apply_wgpu_fused_core_override(
        &mut model_config,
        backend_name,
        WgpuFusedCoreOverride {
            recurrent: resolved_config.wgpu.training.fused_core_recurrent,
            rollout: resolved_config.wgpu.training.fused_core_rollout,
        },
    );
    info!(
        "training path fingerprint: backend={} execution_form={} launch_mode={:?} effective_sequence_kernel={:?} sequence_kernel_override={:?} tbptt_chunk_size={:?} kernel_block_size={} pipeline_enabled={}",
        backend_name,
        build_training_execution_form(&resolved_config),
        training.launch_mode,
        model_config.sequence_kernel,
        effective_training_sequence_kernel_override,
        training.tbptt_chunk_size,
        training_kernel_block_size,
        resolved_config.parallel.pipeline.enabled,
    );
    if let Some(reason) = promotion_reason {
        info!(
            "training sequence kernel promotion: configured={:?} effective={:?} reason={reason}",
            configured_sequence_kernel, model_config.sequence_kernel,
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda") && model_config.fused_kernels.enabled {
        warn!(
            "cuda language training still mixes burn_dragon_kernel fused kernels with generic Burn tensor ops; only selected recurrent/projection paths are accelerated today"
        );
    }
    if backend_name.eq_ignore_ascii_case("cuda")
        && matches!(
            model_config.sequence_kernel.memory_system,
            SequenceMemorySystem::Mamba2StateSpaceDuality
                | SequenceMemorySystem::Mamba3StateSpaceDuality
        )
    {
        if let Some(summary) = cuda_mamba_training_geometry_summary(
            &model_config,
            resolved_config.training.batch_size,
            training_kernel_block_size,
        ) {
            info!("{summary}");
        }
        match model_config.sequence_kernel.memory_system {
            SequenceMemorySystem::Mamba2StateSpaceDuality => warn!(
                "cuda mamba2 training is on the tensorized SSD path with the custom analytic backward wrapper; the fused SSD recurrence core and shell fusion path are enabled by default on CUDA"
            ),
            SequenceMemorySystem::Mamba3StateSpaceDuality => warn!(
                "cuda mamba3 training defaults to the tensorized custom analytical backward wrapper over the chunked SISO path; set BURN_DRAGON_MAMBA3_CUDA_TENSORIZED_TRAIN_WRAPPER=0 to force the direct graph baseline"
            ),
            _ => {}
        }
    }
    if backend_name.eq_ignore_ascii_case("cuda")
        && model_config.fused_kernels.enabled
        && matches!(
            model_config.sequence_kernel.memory_system,
            SequenceMemorySystem::Rwkv8StateSpace
        )
    {
        if let Some(summary) = cuda_rwkv8_training_geometry_summary(
            &model_config,
            resolved_config.training.batch_size,
            training_kernel_block_size,
        ) {
            info!("{summary}");
        }
        warn!(
            "cuda rwkv8 training defaults to the tensorized custom analytical backward wrapper over the decayed normalized recurrence; set BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER=0 to force the direct graph baseline"
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
            if parallel_runtime.mode != ParallelismKind::Single {
                warn!(
                    "parallel.pipeline.communication=block_residual_cache currently reports simulated savings, but live distributed pipeline transport still sends full pipeline states until compressed block-residual transport is implemented"
                );
            }
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
    let mut optim = Some(resolve_optimizer::<B, LanguageTrainModel<B>>(
        optimizer_cfg,
        total_steps,
    )?);
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
            effective_training_sequence_kernel_override,
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
    info!(
        "optimizer fingerprint: name={:?} schedule_mode={:?} learning_rate={} weight_decay={} weight_decay_final={:?}",
        optimizer_cfg.name,
        optimizer_cfg.schedule_mode,
        optimizer_cfg.learning_rate,
        optimizer_cfg.weight_decay,
        optimizer_cfg.weight_decay_final,
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
    let _model = match optim.take().expect("optimizer initialized") {
        ResolvedOptimizer::AdamW(optimizer) => train_with_resolved_scheduler(
            &context,
            model.take().expect("model initialized"),
            optimizer,
            scheduler,
        )?,
        ResolvedOptimizer::BitNetAdamW(optimizer) => train_with_resolved_scheduler(
            &context,
            model.take().expect("model initialized"),
            optimizer,
            scheduler,
        )?,
        ResolvedOptimizer::MuonHybrid(optimizer) => train_with_resolved_scheduler(
            &context,
            model.take().expect("model initialized"),
            optimizer,
            scheduler,
        )?,
    };

    info!("Training complete on {backend_name}");

    if let Some(start) = train_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let snapshot = crate::train::profile::snapshot();
        info!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={} max_step_reserved_before_bytes={} max_step_in_use_before_bytes={} max_step_reserved_after_forward_bytes={} max_step_in_use_after_forward_bytes={} max_step_reserved_after_backward_bytes={} max_step_in_use_after_backward_bytes={}",
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
            snapshot.max_step_reserved_before_bytes,
            snapshot.max_step_in_use_before_bytes,
            snapshot.max_step_reserved_after_forward_bytes,
            snapshot.max_step_in_use_after_forward_bytes,
            snapshot.max_step_reserved_after_backward_bytes,
            snapshot.max_step_in_use_after_backward_bytes,
        );
        eprintln!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} embed_probe_ns={} first_layer_forward_probe_ns={} first_layer_probe_ns={} logits_loss_probe_ns={} hidden_logits_loss_probe_ns={} hidden_model_forward_probe_ns={} hidden_model_probe_ns={} detail_probe_steps={} train_steps={} max_step_reserved_before_bytes={} max_step_in_use_before_bytes={} max_step_reserved_after_forward_bytes={} max_step_in_use_after_forward_bytes={} max_step_reserved_after_backward_bytes={} max_step_in_use_after_backward_bytes={}",
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
            snapshot.max_step_reserved_before_bytes,
            snapshot.max_step_in_use_before_bytes,
            snapshot.max_step_reserved_after_forward_bytes,
            snapshot.max_step_in_use_after_forward_bytes,
            snapshot.max_step_reserved_after_backward_bytes,
            snapshot.max_step_in_use_after_backward_bytes,
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
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: OptimizerKind::default(),
                learning_rate: 1e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
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
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
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
    fn cuda_mamba2_training_geometry_summary_reports_resolved_shape() {
        let model_config = burn_dragon_core::BDHConfig {
            n_embd: 128,
            sequence_kernel: SequenceKernelConfig::reference(
                SequenceMemorySystem::Mamba2StateSpaceDuality,
            ),
            mamba: burn_dragon_core::MambaSequenceConfig {
                headdim: 128,
                ngroups: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        let summary =
            cuda_mamba_training_geometry_summary(&model_config, 24, 512).expect("summary");
        assert!(summary.contains("tokens/micro_batch=12288"), "{summary}");
        assert!(summary.contains("headdim=128"), "{summary}");
        assert!(summary.contains("nheads=2"), "{summary}");
    }

    #[test]
    fn cuda_mamba3_training_geometry_summary_reports_resolved_shape() {
        let model_config = burn_dragon_core::BDHConfig {
            n_embd: 128,
            sequence_kernel: SequenceKernelConfig::reference(
                SequenceMemorySystem::Mamba3StateSpaceDuality,
            ),
            mamba: burn_dragon_core::MambaSequenceConfig {
                headdim: 64,
                ngroups: 4,
                rope_fraction: 0.5,
                chunk_size: 64,
                ..Default::default()
            },
            ..Default::default()
        };

        let summary =
            cuda_mamba_training_geometry_summary(&model_config, 24, 512).expect("summary");
        assert!(summary.contains("tokens/micro_batch=12288"), "{summary}");
        assert!(summary.contains("headdim=64"), "{summary}");
        assert!(summary.contains("nheads=4"), "{summary}");
        assert!(summary.contains("rope_angles=4"), "{summary}");
    }

    #[test]
    fn cuda_mamba_training_geometry_summary_skips_other_kernels() {
        let model_config = burn_dragon_core::BDHConfig {
            sequence_kernel: SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
            mamba: burn_dragon_core::MambaSequenceConfig {
                headdim: 128,
                ngroups: 1,
                ..Default::default()
            },
            ..Default::default()
        };

        assert!(cuda_mamba_training_geometry_summary(&model_config, 24, 512).is_none());
    }

    #[test]
    fn cuda_rwkv8_training_geometry_summary_reports_resolved_shape() {
        let model_config = burn_dragon_core::BDHConfig {
            n_embd: 128,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            sequence_kernel: SequenceKernelConfig::reference(SequenceMemorySystem::Rwkv8StateSpace),
            ..Default::default()
        };

        let summary =
            cuda_rwkv8_training_geometry_summary(&model_config, 24, 512).expect("summary");
        assert!(summary.contains("tokens/micro_batch=12288"), "{summary}");
        assert!(summary.contains("latent_total=512"), "{summary}");
        assert!(summary.contains("latent/head=128"), "{summary}");
        assert!(summary.contains("runtime_chunk=128"), "{summary}");
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
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
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
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
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

    #[test]
    fn short_context_cuda_linear_attention_auto_promotes_to_dense_score() {
        let (effective, recorded_override, reason) = resolve_effective_training_sequence_kernel(
            SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
            None,
            "cuda",
            512,
        );

        assert_eq!(effective, SequenceKernelConfig::dense_score_short_context());
        assert_eq!(
            recorded_override,
            Some(SequenceKernelConfig::dense_score_short_context())
        );
        assert!(reason.is_some());
    }

    #[test]
    fn explicit_training_override_prevents_auto_promotion() {
        let explicit = SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention);
        let (effective, recorded_override, reason) = resolve_effective_training_sequence_kernel(
            SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
            Some(explicit),
            "cuda",
            512,
        );

        assert_eq!(effective, explicit);
        assert_eq!(recorded_override, Some(explicit));
        assert!(reason.is_none());
    }

    #[test]
    fn non_cuda_linear_attention_stays_on_reference_without_override() {
        let configured = SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention);
        let (effective, recorded_override, reason) =
            resolve_effective_training_sequence_kernel(configured, None, "cpu", 512);

        assert_eq!(effective, configured);
        assert_eq!(recorded_override, None);
        assert!(reason.is_none());
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
