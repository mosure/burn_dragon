#![recursion_limit = "256"]

#[cfg(feature = "language-ddp")]
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
#[cfg(feature = "language-ddp")]
use std::process::{Command as ProcessCommand, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};
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
    low_bit_training_lowrank_memory_profile_snapshot, low_bit_training_quantize_profile_snapshot,
    lowrank_residual_memory_profile_reset, lowrank_residual_memory_profile_snapshot,
    lowrank_residual_profile_reset, lowrank_residual_profile_snapshot,
};
use burn_dragon_kernel::api::projection::{
    relu_lowrank_forward_profile_reset, relu_lowrank_forward_profile_snapshot,
    relu_lowrank_forward_route_profile_reset, relu_lowrank_forward_route_profile_snapshot,
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
use serde::Serialize;

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
    /// Persist `nvidia-smi` power/utilization samples into the run directory every N seconds.
    /// Set to 0 to disable sidecar sampling.
    #[arg(long, default_value_t = 0)]
    gpu_telemetry_interval_secs: u64,
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

#[derive(Debug, Clone, Serialize)]
struct GpuTelemetryDeviceSummary {
    gpu_index: u32,
    samples: u64,
    mean_utilization_pct: f32,
    max_utilization_pct: f32,
    mean_power_watts: f32,
    max_power_watts: f32,
    max_memory_used_mib: u64,
}

#[derive(Debug, Clone, Serialize)]
struct GpuTelemetrySummary {
    sample_interval_secs: u64,
    devices: Vec<GpuTelemetryDeviceSummary>,
}

#[derive(Debug, Clone, Serialize)]
struct GpuTelemetryActiveTrainSummary {
    warmup_iterations_skipped: usize,
    train_iterations: usize,
    micro_batch_size: usize,
    kernel_block_size: usize,
    tokens_per_step: usize,
    telemetry_samples: u64,
    mean_step_secs: f64,
    median_step_secs: f64,
    approx_tokens_per_sec: f64,
    mean_utilization_pct: f32,
    mean_power_watts: f32,
    max_power_watts: f32,
    max_memory_used_mib: u64,
}

#[derive(Debug, Clone, Copy)]
struct GpuTelemetrySample {
    gpu_index: u32,
    utilization_pct: f32,
    power_watts: f32,
    memory_used_mib: u64,
}

#[derive(Debug, Default)]
struct GpuTelemetryAccumulator {
    samples: u64,
    sum_utilization_pct: f64,
    max_utilization_pct: f32,
    sum_power_watts: f64,
    max_power_watts: f32,
    max_memory_used_mib: u64,
}

#[derive(Debug, Clone, Copy)]
struct ActiveTrainWindow {
    warmup_iterations_skipped: usize,
    train_iterations: usize,
    train_start_elapsed_secs: f64,
    validation_start_elapsed_secs: f64,
    mean_step_secs: f64,
    median_step_secs: f64,
}

struct GpuTelemetrySidecarGuard {
    shutdown: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

impl Drop for GpuTelemetrySidecarGuard {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
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

fn backend_uses_nvidia_gpu(backend: BackendArg) -> bool {
    matches!(
        backend,
        BackendArg::Cuda | BackendArg::Wgpu | BackendArg::WgpuNoFusion
    )
}

fn parse_gpu_telemetry_samples(stdout: &str) -> Result<Vec<GpuTelemetrySample>> {
    stdout
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let fields = line
                .split(',')
                .map(|field| field.trim())
                .collect::<Vec<_>>();
            if fields.len() != 4 {
                return Err(anyhow::anyhow!(
                    "expected 4 telemetry fields from nvidia-smi, got {} in line: {}",
                    fields.len(),
                    line
                ));
            }
            Ok(GpuTelemetrySample {
                gpu_index: fields[0].parse()?,
                utilization_pct: fields[1].parse()?,
                power_watts: fields[2].parse()?,
                memory_used_mib: fields[3].parse()?,
            })
        })
        .collect()
}

fn query_gpu_telemetry_samples() -> Result<Vec<GpuTelemetrySample>> {
    let output = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()?;
    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "nvidia-smi exited with status {}: {}",
            output.status,
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    parse_gpu_telemetry_samples(&String::from_utf8_lossy(&output.stdout))
}

fn spawn_gpu_telemetry_sidecar(
    run_dir: &Path,
    interval: Duration,
) -> Result<GpuTelemetrySidecarGuard> {
    let csv_path = run_dir.join("gpu_telemetry.csv");
    let summary_path = run_dir.join("gpu_telemetry_summary.json");
    let shutdown = Arc::new(AtomicBool::new(false));
    let worker_shutdown = Arc::clone(&shutdown);
    let csv_path_for_thread = csv_path.clone();
    let summary_path_for_thread = summary_path.clone();
    let interval_secs = interval.as_secs().max(1);

    let worker = thread::Builder::new()
        .name("language-gpu-telemetry".into())
        .spawn(move || {
            let mut csv = match std::fs::File::create(&csv_path_for_thread) {
                Ok(file) => file,
                Err(err) => {
                    tracing::warn!(
                        "failed to create gpu telemetry sidecar {}: {err}",
                        csv_path_for_thread.display()
                    );
                    return;
                }
            };
            if writeln!(
                csv,
                "sample,elapsed_secs,gpu_index,utilization_pct,power_watts,memory_used_mib"
            )
            .is_err()
            {
                tracing::warn!(
                    "failed to write gpu telemetry header to {}",
                    csv_path_for_thread.display()
                );
                return;
            }

            let start = Instant::now();
            let mut sample_index = 0u64;
            let mut devices: std::collections::BTreeMap<u32, GpuTelemetryAccumulator> =
                std::collections::BTreeMap::new();

            while !worker_shutdown.load(Ordering::Relaxed) {
                match query_gpu_telemetry_samples() {
                    Ok(samples) => {
                        sample_index += 1;
                        let elapsed = start.elapsed().as_secs_f32();
                        for sample in samples {
                            let _ = writeln!(
                                csv,
                                "{sample_index},{elapsed:.3},{},{:.1},{:.1},{}",
                                sample.gpu_index,
                                sample.utilization_pct,
                                sample.power_watts,
                                sample.memory_used_mib
                            );
                            let entry = devices.entry(sample.gpu_index).or_default();
                            entry.samples += 1;
                            entry.sum_utilization_pct += f64::from(sample.utilization_pct);
                            entry.max_utilization_pct =
                                entry.max_utilization_pct.max(sample.utilization_pct);
                            entry.sum_power_watts += f64::from(sample.power_watts);
                            entry.max_power_watts = entry.max_power_watts.max(sample.power_watts);
                            entry.max_memory_used_mib =
                                entry.max_memory_used_mib.max(sample.memory_used_mib);
                        }
                        let _ = csv.flush();
                    }
                    Err(err) => {
                        tracing::warn!("gpu telemetry sidecar sample failed: {err}");
                        break;
                    }
                }

                let mut slept = Duration::ZERO;
                while slept < interval && !worker_shutdown.load(Ordering::Relaxed) {
                    let step = (interval - slept).min(Duration::from_millis(200));
                    thread::sleep(step);
                    slept += step;
                }
            }

            let summary = GpuTelemetrySummary {
                sample_interval_secs: interval_secs,
                devices: devices
                    .into_iter()
                    .map(|(gpu_index, entry)| GpuTelemetryDeviceSummary {
                        gpu_index,
                        samples: entry.samples,
                        mean_utilization_pct: if entry.samples == 0 {
                            0.0
                        } else {
                            (entry.sum_utilization_pct / entry.samples as f64) as f32
                        },
                        max_utilization_pct: entry.max_utilization_pct,
                        mean_power_watts: if entry.samples == 0 {
                            0.0
                        } else {
                            (entry.sum_power_watts / entry.samples as f64) as f32
                        },
                        max_power_watts: entry.max_power_watts,
                        max_memory_used_mib: entry.max_memory_used_mib,
                    })
                    .collect(),
            };
            match serde_json::to_string_pretty(&summary) {
                Ok(json) => {
                    let _ = std::fs::write(&summary_path_for_thread, json);
                }
                Err(err) => {
                    tracing::warn!(
                        "failed to serialize gpu telemetry summary {}: {err}",
                        summary_path_for_thread.display()
                    );
                }
            }
        })?;

    tracing::info!(
        "gpu telemetry sidecar writing {} and {} every {}s",
        csv_path.display(),
        summary_path.display(),
        interval_secs
    );

    Ok(GpuTelemetrySidecarGuard {
        shutdown,
        worker: Some(worker),
    })
}

fn parse_log_clock_secs(line: &str) -> Option<f64> {
    let token = line.split_whitespace().next()?;
    let time = token.split('T').nth(1)?.trim_end_matches('Z');
    let mut parts = time.split(':');
    let hour = parts.next()?.parse::<f64>().ok()?;
    let minute = parts.next()?.parse::<f64>().ok()?;
    let second = parts.next()?.parse::<f64>().ok()?;
    Some(hour * 3600.0 + minute * 60.0 + second)
}

fn parse_iteration_index(line: &str) -> Option<usize> {
    line.split("INFO Iteration ")
        .nth(1)?
        .trim()
        .parse::<usize>()
        .ok()
}

fn effective_training_kernel_block_size(
    training: &burn_dragon_language::TrainingHyperparameters,
) -> usize {
    training
        .tbptt_chunk_size
        .filter(|chunk| *chunk > 0 && *chunk < training.block_size)
        .unwrap_or(training.block_size)
        .max(1)
}

fn resolve_active_train_window(experiment_log: &Path) -> Result<Option<ActiveTrainWindow>> {
    use anyhow::Context as _;

    let content = std::fs::read_to_string(experiment_log)
        .with_context(|| format!("failed to read {}", experiment_log.display()))?;
    let mut base_secs = None;
    let mut iteration_times = Vec::new();
    let mut validation_start_secs = None;
    for line in content.lines() {
        let Some(clock_secs) = parse_log_clock_secs(line) else {
            continue;
        };
        let base = *base_secs.get_or_insert(clock_secs);
        let elapsed_secs = clock_secs - base;
        if let Some(iteration) = parse_iteration_index(line) {
            iteration_times.push((iteration, elapsed_secs));
        }
        if validation_start_secs.is_none() && line.contains("INFO Executing validation step") {
            validation_start_secs = Some(elapsed_secs);
        }
    }
    let Some(validation_start_elapsed_secs) = validation_start_secs else {
        return Ok(None);
    };
    if iteration_times.len() < 2 {
        return Ok(None);
    }
    let train_iterations = iteration_times
        .last()
        .map(|(iteration, _)| *iteration)
        .unwrap_or_default();
    if train_iterations < 2 {
        return Ok(None);
    }
    let warmup_iterations_skipped = if train_iterations > 20 {
        10
    } else {
        (train_iterations / 4).clamp(2, 10)
    };
    let train_start_elapsed_secs = iteration_times
        .iter()
        .find(|(iteration, _)| *iteration >= warmup_iterations_skipped)
        .map(|(_, elapsed)| *elapsed)
        .or_else(|| iteration_times.first().map(|(_, elapsed)| *elapsed))
        .unwrap_or(0.0);
    let mut step_deltas = Vec::new();
    for window in iteration_times.windows(2) {
        let [
            (prev_iteration, prev_elapsed),
            (next_iteration, next_elapsed),
        ] = window
        else {
            continue;
        };
        let _ = prev_iteration;
        if *next_iteration >= warmup_iterations_skipped {
            step_deltas.push(next_elapsed - prev_elapsed);
        }
    }
    if step_deltas.is_empty() {
        return Ok(None);
    }
    let mean_step_secs = step_deltas.iter().sum::<f64>() / step_deltas.len() as f64;
    step_deltas.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    let median_step_secs = step_deltas[step_deltas.len() / 2];
    Ok(Some(ActiveTrainWindow {
        warmup_iterations_skipped,
        train_iterations,
        train_start_elapsed_secs,
        validation_start_elapsed_secs,
        mean_step_secs,
        median_step_secs,
    }))
}

fn write_active_train_telemetry_summary(run_dir: &Path) -> Result<()> {
    use anyhow::Context as _;

    let config_path = run_dir.join("training_config.json");
    let config_payload = std::fs::read(&config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    let config: LanguageTrainingConfig = serde_json::from_slice(&config_payload)
        .with_context(|| format!("failed to parse {}", config_path.display()))?;
    let Some(window) = resolve_active_train_window(&run_dir.join("experiment.log"))? else {
        return Ok(());
    };
    let csv_path = run_dir.join("gpu_telemetry.csv");
    let csv_payload = std::fs::read_to_string(&csv_path)
        .with_context(|| format!("failed to read {}", csv_path.display()))?;
    let mut telemetry_samples = 0u64;
    let mut sum_utilization_pct = 0.0f64;
    let mut sum_power_watts = 0.0f64;
    let mut max_power_watts = 0.0f32;
    let mut max_memory_used_mib = 0u64;
    for line in csv_payload.lines().skip(1) {
        let mut parts = line.split(',');
        let _sample = parts.next();
        let Some(elapsed_secs) = parts.next().and_then(|value| value.parse::<f64>().ok()) else {
            continue;
        };
        let _gpu_index = parts.next();
        let Some(utilization_pct) = parts.next().and_then(|value| value.parse::<f32>().ok()) else {
            continue;
        };
        let Some(power_watts) = parts.next().and_then(|value| value.parse::<f32>().ok()) else {
            continue;
        };
        let Some(memory_used_mib) = parts.next().and_then(|value| value.parse::<u64>().ok()) else {
            continue;
        };
        if elapsed_secs < window.train_start_elapsed_secs
            || elapsed_secs > window.validation_start_elapsed_secs
        {
            continue;
        }
        telemetry_samples += 1;
        sum_utilization_pct += f64::from(utilization_pct);
        sum_power_watts += f64::from(power_watts);
        max_power_watts = max_power_watts.max(power_watts);
        max_memory_used_mib = max_memory_used_mib.max(memory_used_mib);
    }
    if telemetry_samples == 0 {
        return Ok(());
    }
    let kernel_block_size = effective_training_kernel_block_size(&config.training);
    let tokens_per_step = config.training.batch_size.saturating_mul(kernel_block_size);
    let summary = GpuTelemetryActiveTrainSummary {
        warmup_iterations_skipped: window.warmup_iterations_skipped,
        train_iterations: window.train_iterations,
        micro_batch_size: config.training.batch_size,
        kernel_block_size,
        tokens_per_step,
        telemetry_samples,
        mean_step_secs: window.mean_step_secs,
        median_step_secs: window.median_step_secs,
        approx_tokens_per_sec: tokens_per_step as f64 / window.mean_step_secs.max(1.0e-9),
        mean_utilization_pct: (sum_utilization_pct / telemetry_samples as f64) as f32,
        mean_power_watts: (sum_power_watts / telemetry_samples as f64) as f32,
        max_power_watts,
        max_memory_used_mib,
    };
    let summary_path = run_dir.join("gpu_telemetry_active_train_summary.json");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)
        .with_context(|| format!("failed to write {}", summary_path.display()))?;
    tracing::info!(
        "active train summary: tokens_per_sec={:.1} mean_power_watts={:.1} mean_utilization_pct={:.1} summary={}",
        summary.approx_tokens_per_sec,
        summary.mean_power_watts,
        summary.mean_utilization_pct,
        summary_path.display(),
    );
    Ok(())
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
    let telemetry_run_dir = planned_run
        .as_ref()
        .map(|planned_run| planned_run.run_dir.clone());

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

    let gpu_telemetry_guard =
        if backend_uses_nvidia_gpu(args.backend) && args.gpu_telemetry_interval_secs > 0 {
            planned_run
                .as_ref()
                .map(|planned_run| {
                    spawn_gpu_telemetry_sidecar(
                        &planned_run.run_dir,
                        Duration::from_secs(args.gpu_telemetry_interval_secs.max(1)),
                    )
                })
                .transpose()?
        } else {
            None
        };

    let result = run_in_training_thread("language-train", move || match args.backend {
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
                relu_lowrank_forward_route_profile_reset();
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
                let lowrank_forward_route = relu_lowrank_forward_route_profile_snapshot();
                let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                let logits_projection = logits_projection_profile_snapshot();
                let lowbit_quantize = low_bit_training_quantize_profile_snapshot();
                let residual_memory = lowrank_residual_memory_profile_snapshot();
                let residual = lowrank_residual_profile_snapshot();
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                    lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-forward-route] attempts={} successes={} fallbacks={} wgpu_fusion_autodiff={} wgpu_direct_autodiff={} wgpu_fusion_runtime={} wgpu_direct_runtime={} cuda_fusion_autodiff={} cuda_direct_autodiff={} cuda_fusion_runtime={} cuda_direct_runtime={}",
                    lowrank_forward_route.attempts,
                    lowrank_forward_route.successes(),
                    lowrank_forward_route.fallbacks(),
                    lowrank_forward_route.wgpu_fusion_autodiff,
                    lowrank_forward_route.wgpu_direct_autodiff,
                    lowrank_forward_route.wgpu_fusion_runtime,
                    lowrank_forward_route.wgpu_direct_runtime,
                    lowrank_forward_route.cuda_fusion_autodiff,
                    lowrank_forward_route.cuda_direct_autodiff,
                    lowrank_forward_route.cuda_fusion_runtime,
                    lowrank_forward_route.cuda_direct_runtime,
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
                    "[stage-profile][training-lowbit-quantize] lowrank_direct_calls={} lowrank_direct_total_ns={} lowrank_fallback_calls={} lowrank_fallback_total_ns={} decoder_tail_calls={} decoder_tail_total_ns={}",
                    lowbit_quantize.lowrank_direct_calls,
                    lowbit_quantize.lowrank_direct_total_ns,
                    lowbit_quantize.lowrank_fallback_calls,
                    lowbit_quantize.lowrank_fallback_total_ns,
                    lowbit_quantize.decoder_tail_calls,
                    lowbit_quantize.decoder_tail_total_ns,
                );
                eprintln!(
                    "[stage-profile][training-residual-step] calls={} total_ns={} x_projection_ns={} x_post_quant_ns={} attention_norm_ns={} attention_mixer_ns={} attention_post_norm_ns={} y_projection_ns={} y_post_quant_ns={} y_neuron_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                    residual.calls,
                    residual.total_ns,
                    residual.x_projection_ns,
                    residual.x_post_quant_ns,
                    residual.attention_norm_ns,
                    residual.attention_mixer_ns,
                    residual.attention_post_norm_ns,
                    residual.y_projection_ns,
                    residual.y_post_quant_ns,
                    residual.y_neuron_ns,
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
                relu_lowrank_forward_route_profile_reset();
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
                let lowrank_forward_route = relu_lowrank_forward_route_profile_snapshot();
                let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                let logits_projection = logits_projection_profile_snapshot();
                let lowbit_quantize = low_bit_training_quantize_profile_snapshot();
                let residual_memory = lowrank_residual_memory_profile_snapshot();
                let residual = lowrank_residual_profile_snapshot();
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                    lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                );
                eprintln!(
                    "[stage-profile][training-lowrank-forward-route] attempts={} successes={} fallbacks={} wgpu_fusion_autodiff={} wgpu_direct_autodiff={} wgpu_fusion_runtime={} wgpu_direct_runtime={} cuda_fusion_autodiff={} cuda_direct_autodiff={} cuda_fusion_runtime={} cuda_direct_runtime={}",
                    lowrank_forward_route.attempts,
                    lowrank_forward_route.successes(),
                    lowrank_forward_route.fallbacks(),
                    lowrank_forward_route.wgpu_fusion_autodiff,
                    lowrank_forward_route.wgpu_direct_autodiff,
                    lowrank_forward_route.wgpu_fusion_runtime,
                    lowrank_forward_route.wgpu_direct_runtime,
                    lowrank_forward_route.cuda_fusion_autodiff,
                    lowrank_forward_route.cuda_direct_autodiff,
                    lowrank_forward_route.cuda_fusion_runtime,
                    lowrank_forward_route.cuda_direct_runtime,
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
                    "[stage-profile][training-lowbit-quantize] lowrank_direct_calls={} lowrank_direct_total_ns={} lowrank_fallback_calls={} lowrank_fallback_total_ns={} decoder_tail_calls={} decoder_tail_total_ns={}",
                    lowbit_quantize.lowrank_direct_calls,
                    lowbit_quantize.lowrank_direct_total_ns,
                    lowbit_quantize.lowrank_fallback_calls,
                    lowbit_quantize.lowrank_fallback_total_ns,
                    lowbit_quantize.decoder_tail_calls,
                    lowbit_quantize.decoder_tail_total_ns,
                );
                eprintln!(
                    "[stage-profile][training-residual-step] calls={} total_ns={} x_projection_ns={} x_post_quant_ns={} attention_norm_ns={} attention_mixer_ns={} attention_post_norm_ns={} y_projection_ns={} y_post_quant_ns={} y_neuron_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                    residual.calls,
                    residual.total_ns,
                    residual.x_projection_ns,
                    residual.x_post_quant_ns,
                    residual.attention_norm_ns,
                    residual.attention_mixer_ns,
                    residual.attention_post_norm_ns,
                    residual.y_projection_ns,
                    residual.y_post_quant_ns,
                    residual.y_neuron_ns,
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
                    relu_lowrank_forward_route_profile_reset();
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
                    let lowrank_forward_route = relu_lowrank_forward_route_profile_snapshot();
                    let lowrank_grad_input = relu_lowrank_grad_input_profile_snapshot();
                    let lowrank_grad_weight = relu_lowrank_grad_weight_profile_snapshot();
                    let logits_projection = logits_projection_profile_snapshot();
                    let lowbit_quantize = low_bit_training_quantize_profile_snapshot();
                    let residual_memory = lowrank_residual_memory_profile_snapshot();
                    let residual = lowrank_residual_profile_snapshot();
                    let snapshot = recurrent_profile_snapshot();
                    eprintln!(
                        "[stage-profile][training-lowrank-forward] calls={} launches={} total_ns={}",
                        lowrank_forward.calls, lowrank_forward.launches, lowrank_forward.total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-lowrank-forward-route] attempts={} successes={} fallbacks={} wgpu_fusion_autodiff={} wgpu_direct_autodiff={} wgpu_fusion_runtime={} wgpu_direct_runtime={} cuda_fusion_autodiff={} cuda_direct_autodiff={} cuda_fusion_runtime={} cuda_direct_runtime={}",
                        lowrank_forward_route.attempts,
                        lowrank_forward_route.successes(),
                        lowrank_forward_route.fallbacks(),
                        lowrank_forward_route.wgpu_fusion_autodiff,
                        lowrank_forward_route.wgpu_direct_autodiff,
                        lowrank_forward_route.wgpu_fusion_runtime,
                        lowrank_forward_route.wgpu_direct_runtime,
                        lowrank_forward_route.cuda_fusion_autodiff,
                        lowrank_forward_route.cuda_direct_autodiff,
                        lowrank_forward_route.cuda_fusion_runtime,
                        lowrank_forward_route.cuda_direct_runtime,
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
                        "[stage-profile][training-lowbit-quantize] lowrank_direct_calls={} lowrank_direct_total_ns={} lowrank_fallback_calls={} lowrank_fallback_total_ns={} decoder_tail_calls={} decoder_tail_total_ns={}",
                        lowbit_quantize.lowrank_direct_calls,
                        lowbit_quantize.lowrank_direct_total_ns,
                        lowbit_quantize.lowrank_fallback_calls,
                        lowbit_quantize.lowrank_fallback_total_ns,
                        lowbit_quantize.decoder_tail_calls,
                        lowbit_quantize.decoder_tail_total_ns,
                    );
                    eprintln!(
                        "[stage-profile][training-residual-step] calls={} total_ns={} x_projection_ns={} x_post_quant_ns={} attention_norm_ns={} attention_mixer_ns={} attention_post_norm_ns={} y_projection_ns={} y_post_quant_ns={} y_neuron_ns={} decoder_tail_ns={} mlp_norm_ns={} residual_combine_ns={}",
                        residual.calls,
                        residual.total_ns,
                        residual.x_projection_ns,
                        residual.x_post_quant_ns,
                        residual.attention_norm_ns,
                        residual.attention_mixer_ns,
                        residual.attention_post_norm_ns,
                        residual.y_projection_ns,
                        residual.y_post_quant_ns,
                        residual.y_neuron_ns,
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
    });
    drop(gpu_telemetry_guard);
    if result.is_ok() {
        if let Some(run_dir) = telemetry_run_dir.as_ref() {
            if let Err(err) = write_active_train_telemetry_summary(run_dir) {
                tracing::warn!(
                    "failed to write active train telemetry summary for {}: {err:#}",
                    run_dir.display()
                );
            }
        }
    }
    result
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
        plan_single_process_run_artifacts, prepare_command, resolve_active_train_window,
        resolve_cli_run_root, write_active_train_telemetry_summary,
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
    fn resolve_cli_run_root_mirrors_local_overlay_configs_under_local_namespace() {
        let run_root = resolve_cli_run_root(
            &tiny_training_config(),
            &[PathBuf::from(
                "config/local/shakespeare_kernel_ablation/mamba2.toml",
            )],
        );

        assert_eq!(
            run_root,
            PathBuf::from("runs/language/local/shakespeare_kernel_ablation/mamba2")
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
                assert_eq!(args.gpu_telemetry_interval_secs, 0);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected language command"),
        }
    }

    #[test]
    fn clap_defaults_gpu_telemetry_to_disabled_for_language_training() {
        let cli = Cli::parse_from(["language_train", "language"]);
        match cli.command {
            Command::Language(args) => {
                assert_eq!(args.gpu_telemetry_interval_secs, 0);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected language command"),
        }
    }

    #[test]
    fn clap_parses_gpu_telemetry_interval_for_language_training() {
        let cli = Cli::parse_from([
            "language_train",
            "language",
            "--gpu-telemetry-interval-secs",
            "4",
        ]);
        match cli.command {
            Command::Language(args) => {
                assert_eq!(args.gpu_telemetry_interval_secs, 4);
            }
            #[allow(unreachable_patterns)]
            _ => panic!("expected language command"),
        }
    }

    #[test]
    fn parse_gpu_telemetry_samples_accepts_nvidia_smi_csv_output() {
        let samples =
            super::parse_gpu_telemetry_samples("0, 91, 287.5, 14321\n1, 88, 250.0, 12001\n")
                .expect("samples");
        assert_eq!(samples.len(), 2);
        assert_eq!(samples[0].gpu_index, 0);
        assert_eq!(samples[0].utilization_pct, 91.0);
        assert_eq!(samples[0].power_watts, 287.5);
        assert_eq!(samples[0].memory_used_mib, 14321);
        assert_eq!(samples[1].gpu_index, 1);
    }

    #[test]
    fn resolve_active_train_window_extracts_steady_state_window() {
        let dir = tempdir().expect("tempdir");
        let log_path = dir.path().join("experiment.log");
        fs::write(
            &log_path,
            "\
2026-03-29T06:00:00.000Z  INFO start\n\
2026-03-29T06:00:01.000Z  INFO Iteration 1\n\
2026-03-29T06:00:02.000Z  INFO Iteration 2\n\
2026-03-29T06:00:03.000Z  INFO Iteration 3\n\
2026-03-29T06:00:04.000Z  INFO Iteration 4\n\
2026-03-29T06:00:05.000Z  INFO Iteration 5\n\
2026-03-29T06:00:06.000Z  INFO Iteration 6\n\
2026-03-29T06:00:07.000Z  INFO Iteration 7\n\
2026-03-29T06:00:08.000Z  INFO Iteration 8\n\
2026-03-29T06:00:09.000Z  INFO Executing validation step for epoch 1\n",
        )
        .expect("write log");

        let window = resolve_active_train_window(&log_path)
            .expect("window")
            .expect("some window");
        assert_eq!(window.warmup_iterations_skipped, 2);
        assert_eq!(window.train_iterations, 8);
        assert!((window.mean_step_secs - 1.0).abs() < 1.0e-6);
        assert!((window.median_step_secs - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn write_active_train_telemetry_summary_emits_tokens_per_sec_summary() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path();
        let mut config = tiny_training_config();
        config.training.batch_size = 4;
        config.training.block_size = 8;
        fs::write(
            run_dir.join("training_config.json"),
            serde_json::to_vec_pretty(&config).expect("serialize config"),
        )
        .expect("write config");
        fs::write(
            run_dir.join("experiment.log"),
            "\
2026-03-29T06:00:00.000Z  INFO start\n\
2026-03-29T06:00:01.000Z  INFO Iteration 1\n\
2026-03-29T06:00:02.000Z  INFO Iteration 2\n\
2026-03-29T06:00:03.000Z  INFO Iteration 3\n\
2026-03-29T06:00:04.000Z  INFO Iteration 4\n\
2026-03-29T06:00:05.000Z  INFO Iteration 5\n\
2026-03-29T06:00:06.000Z  INFO Iteration 6\n\
2026-03-29T06:00:07.000Z  INFO Iteration 7\n\
2026-03-29T06:00:08.000Z  INFO Iteration 8\n\
2026-03-29T06:00:09.000Z  INFO Executing validation step for epoch 1\n",
        )
        .expect("write log");
        fs::write(
            run_dir.join("gpu_telemetry.csv"),
            "\
sample,elapsed_secs,gpu_index,utilization_pct,power_watts,memory_used_mib\n\
1,5.200,0,97.0,410.0,12000\n\
2,6.200,0,98.0,420.0,12032\n\
3,7.200,0,99.0,430.0,12064\n\
4,9.200,0,10.0,100.0,2048\n",
        )
        .expect("write telemetry");

        write_active_train_telemetry_summary(run_dir).expect("write summary");
        let payload = fs::read_to_string(run_dir.join("gpu_telemetry_active_train_summary.json"))
            .expect("read summary");
        let json: serde_json::Value = serde_json::from_str(&payload).expect("parse summary");
        assert_eq!(json["tokens_per_step"], 32);
        assert_eq!(json["telemetry_samples"], 3);
        assert_eq!(json["max_memory_used_mib"], 12064);
        assert_eq!(json["mean_step_secs"], 1.0);
        assert_eq!(json["approx_tokens_per_sec"], 32.0);
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
