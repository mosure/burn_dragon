#![recursion_limit = "256"]

#[cfg(feature = "language-ddp")]
use std::fs;
use std::path::{Path, PathBuf};
#[cfg(feature = "language-ddp")]
use std::process::{Command as ProcessCommand, Stdio};
#[cfg(feature = "language-ddp")]
use std::time::{SystemTime, UNIX_EPOCH};

#[cfg(feature = "language-ddp")]
use anyhow::Context;
use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::BalancedCheckpointing;
#[cfg(feature = "language-ddp")]
use burn_collective::start_global_orchestrator;
use burn_dragon::api::core::recurrent::{
    logits_projection_profile_reset, logits_projection_profile_snapshot,
    low_bit_training_lowrank_memory_profile_snapshot,
    lowrank_residual_memory_profile_reset, lowrank_residual_memory_profile_snapshot,
    lowrank_residual_profile_reset, lowrank_residual_profile_snapshot,
};
use burn_dragon_kernel::api::projection::{
    relu_lowrank_forward_profile_reset, relu_lowrank_forward_profile_snapshot,
    relu_lowrank_grad_input_profile_reset, relu_lowrank_grad_input_profile_snapshot,
    relu_lowrank_grad_weight_profile_reset, relu_lowrank_grad_weight_profile_snapshot,
};
use burn_dragon_kernel::api::recurrent::{recurrent_profile_reset, recurrent_profile_snapshot};
use burn_dragon_language::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV, RUN_ROOT_ENV};
use burn_dragon_language::train::{
    build_vocab_only, prepare_dataset, train_backend as train_language_backend,
};
use burn_dragon_language::{
    TrainingConfig as LanguageTrainingConfig, load_training_config as load_language_training_config,
};
use burn_dragon_train::cli::init_experiment_tracing;
use burn_dragon_train::train::pipeline::{
    PlannedRunArtifacts, TrainingLaunchMode, plan_run_artifacts, resolve_resume_run_dir,
    resolve_run_root_for_config_paths,
};
use burn_dragon_train::wgpu::init_runtime;
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, WgpuRuntime};
use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(feature = "language-cuda")]
use burn_cuda::Cuda;

type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;

const PROCESS_GROUP_RUN_DIR_ENV: &str = "BURN_DRAGON_PROCESS_GROUP_RUN_DIR";

fn balanced_checkpointing_enabled() -> bool {
    std::env::var_os("BURN_DRAGON_BALANCED_CHECKPOINTING").is_some()
}

fn print_low_bit_training_memory_profile() {
    let profile = low_bit_training_lowrank_memory_profile_snapshot();
    for (stage_name, stage) in [
        ("after_weight_codes", profile.after_weight_codes),
        ("after_activation_codes", profile.after_activation_codes),
        ("after_output", profile.after_output),
    ] {
        eprintln!(
            "[stage-profile][training-lowbit-lowrank-memory] calls={} stage={} reserved_bytes={} in_use_bytes={} tracked_tensor_bytes={}",
            profile.calls,
            stage_name,
            stage.reserved_bytes,
            stage.in_use_bytes,
            stage.tracked_tensor_bytes,
        );
    }
}

fn default_or_explicit_config_paths(default_base: &str, explicit: &[PathBuf]) -> Vec<PathBuf> {
    if explicit.is_empty() {
        vec![PathBuf::from(default_base)]
    } else {
        explicit.to_vec()
    }
}

#[derive(Parser, Debug)]
#[command(author, version, about = "Language-only Dragon training CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Train language models and optional vocabulary build.
    Language(LanguageArgs),
    /// Launch local multi-process language DDP with a shared run directory and Burn orchestrator.
    #[cfg(feature = "language-ddp")]
    LanguageLocalDdp(LanguageLocalDdpArgs),
    /// Run the Burn collective orchestrator used for multi-process DDP experiments.
    #[cfg(feature = "language-ddp")]
    CollectiveOrchestrator(CollectiveOrchestratorArgs),
}

enum PreparedCommand {
    Language(PreparedLanguageCommand),
    #[cfg(feature = "language-ddp")]
    LanguageLocalDdp(LanguageLocalDdpArgs),
    #[cfg(feature = "language-ddp")]
    CollectiveOrchestrator(CollectiveOrchestratorArgs),
}

#[derive(Args, Debug)]
struct LanguageArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Explicit launch intent for fresh runs, exact resumes, or latest-checkpoint reuse.
    #[arg(long, value_enum, default_value_t = LaunchModeArg::FromConfig)]
    launch_mode: LaunchModeArg,
    /// Build vocabulary and exit without training.
    #[arg(long)]
    build_vocab_only: bool,
    /// Serve live training metrics to a local rerun server.
    #[arg(long)]
    rerun: bool,
    /// IP address to bind the rerun gRPC server on.
    #[arg(long, default_value = "127.0.0.1")]
    rerun_bind_ip: String,
    /// TCP port to bind the rerun gRPC server on.
    #[arg(long, default_value_t = 9876)]
    rerun_port: u16,
    /// GPU telemetry sampling interval for rerun, in seconds.
    #[arg(long, default_value_t = 5)]
    rerun_telemetry_interval_secs: u64,
}

#[cfg(feature = "language-ddp")]
#[derive(Args, Debug)]
struct LanguageLocalDdpArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
    backend: BackendArg,
    /// Explicit launch intent for fresh runs, exact resumes, or latest-checkpoint reuse.
    #[arg(long, value_enum, default_value_t = LaunchModeArg::FromConfig)]
    launch_mode: LaunchModeArg,
    /// Number of launched ranks.
    #[arg(long, default_value_t = 2)]
    world_size: usize,
    /// Websocket port used by the Burn collective orchestrator.
    #[arg(long, default_value_t = 32100)]
    orchestrator_port: u16,
}

#[cfg(feature = "language-ddp")]
#[derive(Args, Debug)]
struct CollectiveOrchestratorArgs {
    /// Websocket port to bind the orchestrator on.
    #[arg(long)]
    port: u16,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
    WgpuNoFusion,
    Ndarray,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum LaunchModeArg {
    FromConfig,
    Fresh,
    ResumeExactRun,
    ResumeLatestCheckpointIfPresent,
    InitFromCheckpoint,
}

impl LaunchModeArg {
    fn resolve(self, config_mode: TrainingLaunchMode) -> TrainingLaunchMode {
        match self {
            Self::FromConfig => config_mode,
            Self::Fresh => TrainingLaunchMode::Fresh,
            Self::ResumeExactRun => TrainingLaunchMode::ResumeExactRun,
            Self::ResumeLatestCheckpointIfPresent => {
                TrainingLaunchMode::ResumeLatestCheckpointIfPresent
            }
            Self::InitFromCheckpoint => TrainingLaunchMode::InitFromCheckpoint,
        }
    }

    #[cfg(feature = "language-ddp")]
    fn as_cli_value(self) -> &'static str {
        match self {
            Self::FromConfig => "from-config",
            Self::Fresh => "fresh",
            Self::ResumeExactRun => "resume-exact-run",
            Self::ResumeLatestCheckpointIfPresent => "resume-latest-checkpoint-if-present",
            Self::InitFromCheckpoint => "init-from-checkpoint",
        }
    }
}

#[derive(Debug)]
struct PreparedLanguageCommand {
    args: LanguageArgs,
    config: LanguageTrainingConfig,
    run_root: PathBuf,
    planned_run: Option<PlannedRunArtifacts>,
}

#[cfg(feature = "language-rerun")]
struct RerunSessionGuard {
    active: bool,
}

#[cfg(feature = "language-rerun")]
impl Drop for RerunSessionGuard {
    fn drop(&mut self) {
        if self.active {
            burn_dragon_language::train::shutdown_training_rerun();
        }
    }
}

fn resolve_cli_run_root(config: &LanguageTrainingConfig, config_paths: &[PathBuf]) -> PathBuf {
    std::env::var_os(RUN_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            resolve_run_root_for_config_paths("language", &config.run_layout, config_paths)
        })
}

fn apply_run_root_and_resume_policy(
    mut config: LanguageTrainingConfig,
    config_paths: &[PathBuf],
    launch_mode_override: LaunchModeArg,
) -> Result<(LanguageTrainingConfig, PathBuf)> {
    let run_root = resolve_cli_run_root(&config, config_paths);
    let resolved_launch_mode = launch_mode_override.resolve(config.training.launch_mode);
    config.training.launch_mode = resolved_launch_mode;
    config.training.resume_run_dir = resolve_resume_run_dir(
        &run_root,
        config.training.resume_run_dir.as_deref(),
        resolved_launch_mode,
    )?;
    Ok((config, run_root))
}

fn plan_single_process_run_artifacts(
    config: &LanguageTrainingConfig,
    run_root: &Path,
) -> Result<Option<PlannedRunArtifacts>> {
    if config.training.resume_run_dir.is_none() && config.training.max_iters == 0 {
        return Ok(None);
    }
    plan_run_artifacts(run_root, config.training.resume_run_dir.as_deref()).map(Some)
}

fn apply_planned_run_env(planned_run: &PlannedRunArtifacts) {
    unsafe {
        std::env::set_var(RUN_ROOT_ENV, &planned_run.run_root);
        std::env::set_var(RUN_DIR_ENV, &planned_run.run_dir);
        std::env::set_var(RUN_NAME_ENV, &planned_run.run_name);
    }
}

fn prepare_language_command(args: LanguageArgs) -> Result<PreparedLanguageCommand> {
    let config_paths = default_or_explicit_config_paths("config/language/base.toml", &args.config);
    let config = load_language_training_config(&config_paths)?;
    let (config, run_root) =
        apply_run_root_and_resume_policy(config, &config_paths, args.launch_mode)?;
    let planned_run =
        if args.build_vocab_only || std::env::var_os(PROCESS_GROUP_RUN_DIR_ENV).is_some() {
            None
        } else {
            plan_single_process_run_artifacts(&config, &run_root)?
        };

    Ok(PreparedLanguageCommand {
        args,
        config,
        run_root,
        planned_run,
    })
}

fn run_in_training_thread<F, T>(name: &str, work: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    #[cfg(target_os = "windows")]
    {
        let stack_mb = std::env::var("BDH_TRAIN_STACK_MB")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(64);
        let stack_bytes = stack_mb.max(8) * 1024 * 1024;
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .stack_size(stack_bytes)
            .spawn(work)
            .context("spawn training thread")?;
        handle
            .join()
            .map_err(|_| anyhow::anyhow!("training thread panicked"))?
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = name;
        work()
    }
}

fn train_language<B, Init>(
    config: &LanguageTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    train_language_backend::<B, _>(config, dataset, backend_name, init)
}

fn run_language(prepared: PreparedLanguageCommand) -> Result<()> {
    let PreparedLanguageCommand {
        args,
        config,
        run_root,
        planned_run,
    } = prepared;

    if args.build_vocab_only {
        build_vocab_only(&config)?;
        return Ok(());
    }

    if let Some(planned_run) = &planned_run {
        apply_planned_run_env(planned_run);
    } else {
        unsafe { std::env::set_var(RUN_ROOT_ENV, &run_root) };
    }

    #[cfg(feature = "language-rerun")]
    let _rerun_guard = if args.rerun {
        let planned_run = planned_run.as_ref().ok_or_else(|| {
            anyhow::anyhow!("rerun requires planned run artifacts for language training")
        })?;
        let info = burn_dragon_language::train::initialize_training_rerun(
            &burn_dragon_language::train::TrainingRerunConfig {
                run_name: planned_run.run_name.clone(),
                bind_ip: args.rerun_bind_ip.clone(),
                port: args.rerun_port,
                telemetry_interval: std::time::Duration::from_secs(
                    args.rerun_telemetry_interval_secs.max(1),
                ),
            },
        )?;
        tracing::info!("rerun server url: {}", info.server_url);
        tracing::info!("rerun viewer url: {}", info.viewer_url);
        Some(RerunSessionGuard { active: true })
    } else {
        None
    };
    #[cfg(not(feature = "language-rerun"))]
    if args.rerun {
        return Err(anyhow::anyhow!(
            "--rerun requires rebuilding language_train with `--features language-rerun`"
        ));
    }

    run_in_training_thread("language-train", move || match args.backend {
        BackendArg::Ndarray => {
            if balanced_checkpointing_enabled() {
                train_language::<Autodiff<NdArray<f32>, BalancedCheckpointing>, _>(
                    &config,
                    "cpu",
                    |_| {},
                )
            } else {
                train_language::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})
            }
        }
        BackendArg::Wgpu => {
            let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
            if stage_profile {
                recurrent_profile_reset();
                relu_lowrank_forward_profile_reset();
                relu_lowrank_grad_input_profile_reset();
                relu_lowrank_grad_weight_profile_reset();
                logits_projection_profile_reset();
                lowrank_residual_memory_profile_reset();
                lowrank_residual_profile_reset();
            }
            let wgpu_config = config.wgpu.clone();
            let backend_name = if config.wgpu.training.fused_core_recurrent == Some(true) {
                "wgpu-fused-core"
            } else {
                "wgpu-nofusion"
            };
            eprintln!(
                "language training: routing --backend wgpu through CubeBackend (backend={backend_name}) for the best measured Shakespeare training throughput"
            );
            let result = if balanced_checkpointing_enabled() {
                train_language::<Autodiff<WgpuNoFusion, BalancedCheckpointing>, _>(
                    &config,
                    backend_name,
                    move |device| init_runtime(device, &wgpu_config),
                )
            } else {
                train_language::<Autodiff<WgpuNoFusion>, _>(&config, backend_name, move |device| {
                    init_runtime(device, &wgpu_config)
                })
            };
            if stage_profile {
                let lowrank_forward = relu_lowrank_forward_profile_snapshot();
                let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                let logits_projection = logits_projection_profile_snapshot();
                let residual_memory = lowrank_residual_memory_profile_snapshot();
                let residual = lowrank_residual_profile_snapshot();
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                    lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-grad-input] calls={} launches={} total_ns={}",
                    lowrank_grad_input.calls,
                    lowrank_grad_input.launches,
                    lowrank_grad_input.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-grad-weight] calls={} total_ns={}",
                    lowrank_grad_weight.calls, lowrank_grad_weight.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-logits-projection] calls={} total_ns={}",
                    logits_projection.calls, logits_projection.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-residual-step] calls={} total_ns={} attention_norm_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                    residual.calls,
                    residual.total_ns,
                    residual.attention_norm_ns,
                    residual.decoder_tail_ns,
                    residual.mlp_norm_ns,
                    residual.residual_combine_ns,
                );
                for (stage_name, stage) in [
                    ("after_attention_norm", residual_memory.after_attention_norm),
                    ("after_y_projection", residual_memory.after_y_projection),
                    ("after_y_post_quant", residual_memory.after_y_post_quant),
                    ("after_y_neuron", residual_memory.after_y_neuron),
                    ("after_decoder_tail", residual_memory.after_decoder_tail),
                    ("after_mlp_norm", residual_memory.after_mlp_norm),
                ] {
                    eprintln!(
                        "[stage-profile][training-residual-memory] calls={} stage={} reserved_bytes={} in_use_bytes={} tracked_tensor_bytes={}",
                        residual_memory.calls,
                        stage_name,
                        stage.reserved_bytes,
                        stage.in_use_bytes,
                        stage.tracked_tensor_bytes,
                    );
                }
                eprintln!(
                    "[stage-profile][training-recurrent] calls={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={}",
                    snapshot.calls,
                    snapshot.total_ns,
                    snapshot.setup_ns,
                    snapshot.copy_ns,
                    snapshot.dispatch_ns,
                );
                print_low_bit_training_memory_profile();
            }
            result
        }
        BackendArg::WgpuNoFusion => {
            let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
            if stage_profile {
                recurrent_profile_reset();
                relu_lowrank_forward_profile_reset();
                relu_lowrank_grad_input_profile_reset();
                relu_lowrank_grad_weight_profile_reset();
                logits_projection_profile_reset();
                lowrank_residual_memory_profile_reset();
                lowrank_residual_profile_reset();
            }
            let wgpu_config = config.wgpu.clone();
            let backend_name = if config.wgpu.training.fused_core_recurrent == Some(true) {
                "wgpu-fused-core"
            } else {
                "wgpu-nofusion"
            };
            let result = if balanced_checkpointing_enabled() {
                train_language::<Autodiff<WgpuNoFusion, BalancedCheckpointing>, _>(
                    &config,
                    backend_name,
                    move |device| init_runtime(device, &wgpu_config),
                )
            } else {
                train_language::<Autodiff<WgpuNoFusion>, _>(&config, backend_name, move |device| {
                    init_runtime(device, &wgpu_config)
                })
            };
            if stage_profile {
                let lowrank_forward = relu_lowrank_forward_profile_snapshot();
                let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                let logits_projection = logits_projection_profile_snapshot();
                let residual_memory = lowrank_residual_memory_profile_snapshot();
                let residual = lowrank_residual_profile_snapshot();
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                    lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-grad-input] calls={} launches={} total_ns={}",
                    lowrank_grad_input.calls,
                    lowrank_grad_input.launches,
                    lowrank_grad_input.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-grad-weight] calls={} total_ns={}",
                    lowrank_grad_weight.calls, lowrank_grad_weight.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-logits-projection] calls={} total_ns={}",
                    logits_projection.calls, logits_projection.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-residual-step] calls={} total_ns={} attention_norm_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                    residual.calls,
                    residual.total_ns,
                    residual.attention_norm_ns,
                    residual.decoder_tail_ns,
                    residual.mlp_norm_ns,
                    residual.residual_combine_ns,
                );
                for (stage_name, stage) in [
                    ("after_attention_norm", residual_memory.after_attention_norm),
                    ("after_y_projection", residual_memory.after_y_projection),
                    ("after_y_post_quant", residual_memory.after_y_post_quant),
                    ("after_y_neuron", residual_memory.after_y_neuron),
                    ("after_decoder_tail", residual_memory.after_decoder_tail),
                    ("after_mlp_norm", residual_memory.after_mlp_norm),
                ] {
                    eprintln!(
                        "[stage-profile][training-residual-memory] calls={} stage={} reserved_bytes={} in_use_bytes={} tracked_tensor_bytes={}",
                        residual_memory.calls,
                        stage_name,
                        stage.reserved_bytes,
                        stage.in_use_bytes,
                        stage.tracked_tensor_bytes,
                    );
                }
                eprintln!(
                    "[stage-profile][training-recurrent] calls={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={}",
                    snapshot.calls,
                    snapshot.total_ns,
                    snapshot.setup_ns,
                    snapshot.copy_ns,
                    snapshot.dispatch_ns,
                );
                print_low_bit_training_memory_profile();
            }
            result
        }
        BackendArg::Cuda => {
            #[cfg(feature = "language-cuda")]
            {
                let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
                if stage_profile {
                    recurrent_profile_reset();
                    relu_lowrank_forward_profile_reset();
                    relu_lowrank_grad_input_profile_reset();
                    relu_lowrank_grad_weight_profile_reset();
                    logits_projection_profile_reset();
                    lowrank_residual_memory_profile_reset();
                    lowrank_residual_profile_reset();
                }
                let result = if balanced_checkpointing_enabled() {
                    train_language::<Autodiff<Cuda<f32>, BalancedCheckpointing>, _>(
                        &config,
                        "cuda",
                        |_| {},
                    )
                } else {
                    train_language::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
                };
                if stage_profile {
                    let lowrank_forward = relu_lowrank_forward_profile_snapshot();
                    let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                    let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                    let logits_projection = logits_projection_profile_snapshot();
                    let residual_memory = lowrank_residual_memory_profile_snapshot();
                    let residual = lowrank_residual_profile_snapshot();
                    let snapshot = recurrent_profile_snapshot();
                    eprintln!(
                        "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                        lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-lowrank-grad-input] calls={} launches={} total_ns={}",
                        lowrank_grad_input.calls,
                        lowrank_grad_input.launches,
                        lowrank_grad_input.total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-lowrank-grad-weight] calls={} total_ns={}",
                        lowrank_grad_weight.calls, lowrank_grad_weight.total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-logits-projection] calls={} total_ns={}",
                        logits_projection.calls, logits_projection.total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-residual-step] calls={} total_ns={} attention_norm_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                        residual.calls,
                        residual.total_ns,
                        residual.attention_norm_ns,
                        residual.decoder_tail_ns,
                        residual.mlp_norm_ns,
                        residual.residual_combine_ns,
                    );
                    for (stage_name, stage) in [
                        ("after_attention_norm", residual_memory.after_attention_norm),
                        ("after_y_projection", residual_memory.after_y_projection),
                        ("after_y_post_quant", residual_memory.after_y_post_quant),
                        ("after_y_neuron", residual_memory.after_y_neuron),
                        ("after_decoder_tail", residual_memory.after_decoder_tail),
                        ("after_mlp_norm", residual_memory.after_mlp_norm),
                    ] {
                        eprintln!(
                            "[stage-profile][training-residual-memory] calls={} stage={} reserved_bytes={} in_use_bytes={} tracked_tensor_bytes={}",
                            residual_memory.calls,
                            stage_name,
                            stage.reserved_bytes,
                            stage.in_use_bytes,
                            stage.tracked_tensor_bytes,
                        );
                    }
                    eprintln!(
                        "[stage-profile][training-recurrent] calls={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={}",
                        snapshot.calls,
                        snapshot.total_ns,
                        snapshot.setup_ns,
                        snapshot.copy_ns,
                        snapshot.dispatch_ns,
                    );
                    print_low_bit_training_memory_profile();
                }
                result
            }
            #[cfg(not(feature = "language-cuda"))]
            {
                Err(anyhow::anyhow!(
                    "cuda backend selected but this build lacks `language-cuda`; rebuild with `--features language-cuda`"
                ))
            }
        }
    })
}

#[cfg(feature = "language-ddp")]
fn local_ddp_run_name() -> Result<String> {
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow::anyhow!("failed to read system time: {err}"))?
        .as_secs();
    Ok(format!(
        "language-local-ddp-{}-{suffix}",
        std::process::id()
    ))
}

#[cfg(feature = "language-ddp")]
fn backend_arg_cli_value(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Cuda => "cuda",
        BackendArg::Wgpu => "wgpu",
        BackendArg::WgpuNoFusion => "wgpu-no-fusion",
        BackendArg::Ndarray => "ndarray",
    }
}

#[cfg(feature = "language-ddp")]
fn write_language_local_ddp_overlay(
    root: &PathBuf,
    rank: usize,
    world_size: usize,
    orchestrator_port: u16,
) -> Result<PathBuf> {
    fs::create_dir_all(root).map_err(|err| {
        anyhow::anyhow!(
            "failed to create DDP overlay directory {}: {err}",
            root.display()
        )
    })?;
    let node_port = orchestrator_port
        .checked_add(1)
        .and_then(|base| base.checked_add(rank as u16))
        .ok_or_else(|| anyhow::anyhow!("orchestrator port overflow for local DDP launch"))?;
    let overlay = root.join(format!("rank-{rank}.toml"));
    let contents = format!(
        r#"[parallel]
mode = "ddp"
world_size = {world_size}

[parallel.data]
size = {world_size}
collective_num_nodes = {world_size}
collective_global_address = "ws://127.0.0.1:{orchestrator_port}"
collective_node_address = "ws://127.0.0.1:{node_port}"
collective_data_service_port = {node_port}
"#
    );
    fs::write(&overlay, contents).map_err(|err| {
        anyhow::anyhow!("failed to write DDP overlay {}: {err}", overlay.display())
    })?;
    Ok(overlay)
}

#[cfg(feature = "language-ddp")]
fn run_language_local_ddp(args: LanguageLocalDdpArgs) -> Result<()> {
    if args.world_size == 0 {
        return Err(anyhow::anyhow!("--world-size must be at least 1"));
    }

    let current_exe = std::env::current_exe().context("resolve current language_train binary")?;
    let config_paths = default_or_explicit_config_paths("config/language/base.toml", &args.config);
    let config = load_language_training_config(&config_paths)?;
    let (config, run_root) =
        apply_run_root_and_resume_policy(config, &config_paths, args.launch_mode)?;
    let (run_name, run_dir) = match &config.training.resume_run_dir {
        Some(run_dir) => {
            if !run_dir.is_dir() {
                return Err(anyhow::anyhow!(
                    "training.resume_run_dir does not exist or is not a directory: {}",
                    run_dir.display()
                ));
            }
            let run_name = run_dir
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| {
                    anyhow::anyhow!("failed to derive run name from {}", run_dir.display())
                })?
                .to_string();
            (run_name, run_dir.clone())
        }
        None => {
            let run_name = local_ddp_run_name()?;
            let run_dir = run_root.join(&run_name);
            fs::create_dir_all(&run_dir).map_err(|err| {
                anyhow::anyhow!(
                    "failed to create shared DDP run directory {}: {err}",
                    run_dir.display()
                )
            })?;
            (run_name, run_dir)
        }
    };

    let overlay_root = std::env::temp_dir().join(format!(
        "burn_dragon_language_local_ddp_{}_{}",
        std::process::id(),
        run_name
    ));
    let overlays = (0..args.world_size)
        .map(|rank| {
            write_language_local_ddp_overlay(
                &overlay_root,
                rank,
                args.world_size,
                args.orchestrator_port,
            )
        })
        .collect::<Result<Vec<_>>>()?;

    let orchestrator_stdout = fs::File::create(run_dir.join("orchestrator.stdout.log"))
        .context("create orchestrator stdout log")?;
    let orchestrator_stderr = fs::File::create(run_dir.join("orchestrator.stderr.log"))
        .context("create orchestrator stderr log")?;
    let mut orchestrator = ProcessCommand::new(&current_exe)
        .arg("collective-orchestrator")
        .arg("--port")
        .arg(args.orchestrator_port.to_string())
        .stdout(Stdio::from(orchestrator_stdout))
        .stderr(Stdio::from(orchestrator_stderr))
        .spawn()
        .context("spawn Burn collective orchestrator")?;
    std::thread::sleep(std::time::Duration::from_millis(500));

    let mut children = Vec::with_capacity(args.world_size);
    for (rank, overlay) in overlays.iter().enumerate() {
        let rank_stdout = fs::File::create(run_dir.join(format!("rank-{rank}.stdout.log")))
            .with_context(|| format!("create stdout log for local DDP rank {rank}"))?;
        let rank_stderr = fs::File::create(run_dir.join(format!("rank-{rank}.stderr.log")))
            .with_context(|| format!("create stderr log for local DDP rank {rank}"))?;
        let mut command = ProcessCommand::new(&current_exe);
        command.arg("language");
        for config in &config_paths {
            command.arg("--config").arg(config);
        }
        command.arg("--config").arg(overlay);
        command
            .arg("--launch-mode")
            .arg(args.launch_mode.as_cli_value());
        command
            .arg("--backend")
            .arg(backend_arg_cli_value(args.backend))
            .env(RUN_ROOT_ENV, &run_root)
            .env("WORLD_SIZE", args.world_size.to_string())
            .env("RANK", rank.to_string())
            .env("LOCAL_RANK", rank.to_string())
            .env("BURN_DRAGON_PROCESS_GROUP_RUN_DIR", &run_dir)
            .env("BURN_DRAGON_PROCESS_GROUP_RUN_NAME", &run_name)
            .stdout(Stdio::from(rank_stdout))
            .stderr(Stdio::from(rank_stderr));
        children.push(
            command
                .spawn()
                .with_context(|| format!("spawn local DDP rank {rank}"))?,
        );
    }

    let mut first_error = None;
    for (rank, mut child) in children.into_iter().enumerate() {
        let status = child
            .wait()
            .with_context(|| format!("wait for local DDP rank {rank}"))?;
        if !status.success() && first_error.is_none() {
            first_error = Some(anyhow::anyhow!(
                "local DDP rank {rank} exited with status {status}"
            ));
        }
    }

    let _ = orchestrator.kill();
    let _ = orchestrator.wait();

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(())
}

#[cfg(feature = "language-ddp")]
fn run_collective_orchestrator(args: CollectiveOrchestratorArgs) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(start_global_orchestrator(args.port));
    Ok(())
}

#[cfg(all(test, feature = "language-train"))]
mod tests {
    use super::{
        Cli, Command, LaunchModeArg, PreparedCommand, PreparedLanguageCommand,
        apply_run_root_and_resume_policy, default_or_explicit_config_paths,
        plan_single_process_run_artifacts, prepare_command, resolve_cli_run_root,
    };
    use burn_dragon_language::TrainingConfig as LanguageTrainingConfig;
    use burn_dragon_language::checkpoint::RUN_ROOT_ENV;
    use burn_dragon_language::config::{
        ContextStrategyConfig, DatasetConfig, DatasetSourceConfig, GenerationConfig,
        ModelOverrides, TrainingHyperparameters,
    };
    use burn_dragon_language::tokenizer::TokenizerConfig;
    use burn_dragon_train::{OptimizerConfig, ParallelConfig, WgpuRuntimeConfig};
    use clap::Parser;
    use std::fs;
    use std::path::PathBuf;
    use std::sync::{Mutex, OnceLock};
    use tempfile::tempdir;

    #[cfg(feature = "language-ddp")]
    use super::{backend_arg_cli_value, write_language_local_ddp_overlay};

    #[test]
    fn default_or_explicit_config_paths_uses_default_only_when_no_explicit_configs() {
        assert_eq!(
            default_or_explicit_config_paths("config/language/base.toml", &[]),
            vec![PathBuf::from("config/language/base.toml")]
        );
        assert_eq!(
            default_or_explicit_config_paths(
                "config/language/base.toml",
                &[PathBuf::from("config/language/custom.toml")]
            ),
            vec![PathBuf::from("config/language/custom.toml")]
        );
    }

    fn tiny_training_config() -> LanguageTrainingConfig {
        LanguageTrainingConfig {
            dataset: DatasetConfig {
                cache_dir: PathBuf::from("data"),
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 32,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 2,
                seed: 1337,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: None,
                max_iters: 4,
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
                name: burn_dragon_train::OptimizerKind::default(),
                learning_rate: 1.0e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: burn_dragon_train::OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: ParallelConfig::default(),
            generation: GenerationConfig {
                prompt: "abc".to_string(),
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
            model: ModelOverrides::default(),
        }
    }

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    #[test]
    fn resolve_cli_run_root_mirrors_last_matching_config_stem() {
        let run_root = resolve_cli_run_root(
            &tiny_training_config(),
            &[
                PathBuf::from("config/language/base.toml"),
                PathBuf::from("config/language/baselines/current_best_large.toml"),
            ],
        );

        assert_eq!(
            run_root,
            PathBuf::from("runs/language/baselines/current_best_large")
        );
    }

    #[test]
    fn apply_run_root_and_resume_policy_sets_resume_dir_from_latest() {
        let _env_guard = env_lock().lock().expect("env lock");
        let dir = tempdir().expect("tempdir");
        let explicit_root = dir.path().join("family-root");
        let run_dir = explicit_root.join("resume-me");
        fs::create_dir_all(&run_dir).expect("create run dir");
        fs::create_dir_all(run_dir.join("checkpoint")).expect("checkpoint dir");
        fs::write(run_dir.join("checkpoint/model-1.bin"), b"checkpoint").expect("checkpoint");
        fs::create_dir_all(&explicit_root).expect("create explicit root");
        fs::write(explicit_root.join("latest"), "resume-me").expect("write latest");
        unsafe { std::env::set_var(RUN_ROOT_ENV, &explicit_root) };

        let config = tiny_training_config();
        let (resolved, run_root) = apply_run_root_and_resume_policy(
            config,
            &[PathBuf::from(
                "config/language/baselines/current_best_large.toml",
            )],
            LaunchModeArg::ResumeLatestCheckpointIfPresent,
        )
        .expect("apply resume policy");
        assert_eq!(run_root, explicit_root);
        assert_eq!(resolved.training.resume_run_dir, Some(run_dir));

        unsafe { std::env::remove_var(RUN_ROOT_ENV) };
    }

    #[test]
    fn plan_single_process_run_artifacts_creates_named_run_under_family_root() {
        let dir = tempdir().expect("tempdir");
        let run_root = dir.path().join("runs").join("family");
        let config = tiny_training_config();

        let planned = plan_single_process_run_artifacts(&config, &run_root)
            .expect("plan run")
            .expect("run");
        assert_eq!(planned.run_root, run_root);
        assert!(planned.run_dir.starts_with(&planned.run_root));
        assert_eq!(
            planned.run_dir.file_name().and_then(|v| v.to_str()),
            Some(planned.run_name.as_str())
        );
    }

    #[test]
    fn prepare_command_skips_single_process_run_planning_for_process_group_launches() {
        let _env_guard = env_lock().lock().expect("env lock");
        let dir = tempdir().expect("tempdir");
        let base_config = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../config/language/base.toml")
            .canonicalize()
            .expect("canonicalize base config");
        unsafe {
            std::env::set_var(
                "BURN_DRAGON_PROCESS_GROUP_RUN_DIR",
                dir.path().join("shared"),
            );
        }
        let cli = Cli::parse_from([
            "language_train",
            "language",
            "--config",
            base_config.to_str().expect("utf8 config path"),
        ]);
        let prepared = prepare_command(cli).expect("prepare command");
        match prepared {
            PreparedCommand::Language(PreparedLanguageCommand { planned_run, .. }) => {
                assert!(planned_run.is_none());
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected language command"),
        }
        unsafe {
            std::env::remove_var("BURN_DRAGON_PROCESS_GROUP_RUN_DIR");
        }
    }

    #[test]
    fn clap_parses_rerun_flags_for_language_training() {
        let cli = Cli::parse_from([
            "language_train",
            "language",
            "--rerun",
            "--rerun-bind-ip",
            "0.0.0.0",
            "--rerun-port",
            "9988",
            "--rerun-telemetry-interval-secs",
            "7",
        ]);
        match cli.command {
            Command::Language(args) => {
                assert!(args.rerun);
                assert_eq!(args.rerun_bind_ip, "0.0.0.0");
                assert_eq!(args.rerun_port, 9988);
                assert_eq!(args.rerun_telemetry_interval_secs, 7);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected language command"),
        }
    }

    #[cfg(feature = "language-ddp")]
    #[test]
    fn backend_arg_cli_value_formats_expected_ddp_backend_flags() {
        assert_eq!(backend_arg_cli_value(super::BackendArg::Ndarray), "ndarray");
        assert_eq!(backend_arg_cli_value(super::BackendArg::Cuda), "cuda");
        assert_eq!(backend_arg_cli_value(super::BackendArg::Wgpu), "wgpu");
        assert_eq!(
            backend_arg_cli_value(super::BackendArg::WgpuNoFusion),
            "wgpu-no-fusion"
        );
    }

    #[cfg(feature = "language-ddp")]
    #[test]
    fn write_language_local_ddp_overlay_emits_rank_specific_collective_config() {
        let dir = tempdir().expect("tempdir");
        let overlay = write_language_local_ddp_overlay(&dir.path().join("overlays"), 1, 2, 32100)
            .expect("overlay");
        let contents = std::fs::read_to_string(&overlay).expect("read overlay");

        assert!(contents.contains("mode = \"ddp\""));
        assert!(contents.contains("world_size = 2"));
        assert!(contents.contains("collective_num_nodes = 2"));
        assert!(contents.contains("collective_global_address = \"ws://127.0.0.1:32100\""));
        assert!(contents.contains("collective_node_address = \"ws://127.0.0.1:32102\""));
        assert!(contents.contains("collective_data_service_port = 32102"));
    }
}

fn prepare_command(cli: Cli) -> Result<PreparedCommand> {
    match cli.command {
        Command::Language(cmd) => Ok(PreparedCommand::Language(prepare_language_command(cmd)?)),
        #[cfg(feature = "language-ddp")]
        Command::LanguageLocalDdp(cmd) => Ok(PreparedCommand::LanguageLocalDdp(cmd)),
        #[cfg(feature = "language-ddp")]
        Command::CollectiveOrchestrator(cmd) => Ok(PreparedCommand::CollectiveOrchestrator(cmd)),
    }
}

fn run(prepared: PreparedCommand) -> Result<()> {
    match prepared {
        PreparedCommand::Language(cmd) => run_language(cmd),
        #[cfg(feature = "language-ddp")]
        PreparedCommand::LanguageLocalDdp(cmd) => run_language_local_ddp(cmd),
        #[cfg(feature = "language-ddp")]
        PreparedCommand::CollectiveOrchestrator(cmd) => run_collective_orchestrator(cmd),
    }
}

fn main() {
    let cli = Cli::parse();
    let prepared = match prepare_command(cli) {
        Ok(prepared) => prepared,
        Err(err) => {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    };
    let log_path = match &prepared {
        PreparedCommand::Language(cmd) => cmd
            .planned_run
            .as_ref()
            .map(|planned_run| planned_run.run_dir.join("experiment.log")),
        #[cfg(feature = "language-ddp")]
        PreparedCommand::LanguageLocalDdp(_) => None,
        #[cfg(feature = "language-ddp")]
        PreparedCommand::CollectiveOrchestrator(_) => None,
    };
    let _log_guard = match init_experiment_tracing(log_path.as_deref()) {
        Ok(guard) => guard,
        Err(err) => {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    };
    if let Err(err) = run(prepared) {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}
