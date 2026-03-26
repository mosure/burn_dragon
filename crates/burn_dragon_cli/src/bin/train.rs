#![recursion_limit = "256"]

#[cfg(feature = "train")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "train")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(all(feature = "train", feature = "ddp"))]
use burn_collective::start_global_orchestrator;
#[cfg(feature = "train")]
use burn_dragon::language::train::{
    build_vocab_only, prepare_dataset, train_backend as train_language_backend,
};
#[cfg(feature = "train")]
use burn_dragon::language::{
    TrainingConfig as LanguageTrainingConfig, load_training_config as load_language_training_config,
};
#[cfg(feature = "train")]
use burn_dragon::multimodal::{
    MultimodalRuntimeConfig, MultimodalTrainingConfig, MultimodalVideoTrainingConfig,
    load_multimodal_runtime_config, train_backend as train_multimodal_backend,
    train_video_backend as train_multimodal_video_backend,
};
#[cfg(all(feature = "train", feature = "ddp"))]
use burn_dragon::train::train::pipeline::{
    plan_run_artifacts, resolve_resume_run_dir, resolve_run_root_for_config_paths,
};
#[cfg(feature = "train")]
use burn_dragon::train::wgpu::{init_runtime, is_wgpu_backend_name};
#[cfg(feature = "train")]
use burn_dragon::vision::train::train_vision_backend;
#[cfg(feature = "train")]
use burn_dragon::vision::{VisionTrainingConfig, load_vision_training_config};
#[cfg(feature = "train")]
use burn_dragon_kernel::api::recurrent::{recurrent_profile_reset, recurrent_profile_snapshot};
#[cfg(feature = "train")]
use burn_dragon_sudoku::config::{
    SudokuTrainingConfig, load_training_config as load_sudoku_training_config,
};
#[cfg(feature = "train")]
use burn_dragon_sudoku::train::train_backend as train_sudoku_backend;
#[cfg(feature = "train")]
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use burn_wgpu::{CubeBackend, Wgpu, WgpuRuntime};
#[cfg(feature = "train")]
use clap::{Args, Parser, Subcommand, ValueEnum};
#[cfg(feature = "train")]
use std::fs;
#[cfg(feature = "train")]
use std::path::PathBuf;
#[cfg(all(feature = "train", feature = "ddp"))]
use std::process::{Command as ProcessCommand, Stdio};

#[cfg(all(feature = "train", feature = "cuda"))]
use burn_cuda::Cuda;

#[cfg(feature = "train")]
type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(feature = "train")]
fn default_or_explicit_config_paths(default_base: &str, explicit: &[PathBuf]) -> Vec<PathBuf> {
    if explicit.is_empty() {
        vec![PathBuf::from(default_base)]
    } else {
        explicit.to_vec()
    }
}

#[cfg(feature = "train")]
fn apply_wgpu_vision_training_overrides(config: &mut VisionTrainingConfig, backend_name: &str) {
    if !is_wgpu_backend_name(backend_name) {
        return;
    }

    let fused_override = config
        .wgpu
        .training
        .fused_core_rollout
        .or(config.wgpu.training.fused_core_recurrent);
    if let Some(enabled) = fused_override {
        config.vision.fused_kernels = enabled;
    }
}

#[cfg(feature = "train")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Unified burn_dragon training CLI")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[cfg(feature = "train")]
#[derive(Subcommand, Debug)]
enum Command {
    /// Train language models and optional vocabulary build.
    Language(LanguageArgs),
    /// Launch local multi-process language DDP with a shared run directory and Burn orchestrator.
    #[cfg(feature = "ddp")]
    LanguageLocalDdp(LanguageLocalDdpArgs),
    /// Run the Burn collective orchestrator used for multi-process DDP experiments.
    #[cfg(feature = "ddp")]
    CollectiveOrchestrator(CollectiveOrchestratorArgs),
    /// Train vision distill/LeJEPA models.
    Vision(VisionArgs),
    /// Train sudoku models.
    Sudoku(SudokuArgs),
    /// Train multimodal VL-JEPA models.
    Multimodal(MultimodalArgs),
}

#[cfg(feature = "train")]
#[derive(Args, Debug)]
struct LanguageArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    /// Build vocabulary and exit without training.
    #[arg(long)]
    build_vocab_only: bool,
}

#[cfg(all(feature = "train", feature = "ddp"))]
#[derive(Args, Debug)]
struct LanguageLocalDdpArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
    backend: BackendArg,
    /// Number of launched ranks.
    #[arg(long, default_value_t = 2)]
    world_size: usize,
    /// Websocket port used by the Burn collective orchestrator.
    #[arg(long, default_value_t = 32100)]
    orchestrator_port: u16,
}

#[cfg(all(feature = "train", feature = "ddp"))]
#[derive(Args, Debug)]
struct CollectiveOrchestratorArgs {
    /// Websocket port to bind the orchestrator on.
    #[arg(long)]
    port: u16,
}

#[cfg(feature = "train")]
#[derive(Args, Debug)]
struct VisionArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[cfg(feature = "train")]
#[derive(Args, Debug)]
struct SudokuArgs {
    /// Configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH", required = true)]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[cfg(feature = "train")]
#[derive(Args, Debug)]
struct MultimodalArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH")]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Wgpu)]
    backend: BackendArg,
}

#[cfg(feature = "train")]
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
    WgpuNoFusion,
    Ndarray,
}

#[cfg(feature = "train")]
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
            .map_err(|_| anyhow!("training thread panicked"))?
    }
    #[cfg(not(target_os = "windows"))]
    {
        let _ = name;
        work()
    }
}

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
fn train_sudoku<B, Init>(
    config: &SudokuTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_sudoku_backend::<B, _>(config, backend_name, init)
}

#[cfg(feature = "train")]
fn train_multimodal<B, Init>(
    config: &MultimodalTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let report = train_multimodal_backend::<B, _>(config, backend_name, init)?;
    eprintln!(
        "multimodal run: {} ({}) checkpoints={} artifacts={}",
        report.run_name,
        report.run_dir.display(),
        report.checkpoint_paths.len(),
        report.artifact_paths.len(),
    );
    Ok(())
}

#[cfg(feature = "train")]
fn train_multimodal_video<B, Init>(
    config: &MultimodalVideoTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let report = train_multimodal_video_backend::<B, _>(config, backend_name, init)?;
    eprintln!(
        "multimodal video run: {} ({}) checkpoints={} artifacts={}",
        report.run_name,
        report.run_dir.display(),
        report.checkpoint_paths.len(),
        report.artifact_paths.len(),
    );
    Ok(())
}

#[cfg(feature = "train")]
fn run_language(args: LanguageArgs) -> Result<()> {
    let config_paths = default_or_explicit_config_paths("config/language/base.toml", &args.config);
    let config = load_language_training_config(&config_paths)?;

    if args.build_vocab_only {
        build_vocab_only(&config)?;
        return Ok(());
    }

    run_in_training_thread("language-train", move || match args.backend {
        BackendArg::Ndarray => train_language::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {}),
        BackendArg::Wgpu => {
            let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
            if stage_profile {
                recurrent_profile_reset();
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
            let result =
                train_language::<Autodiff<WgpuNoFusion>, _>(&config, backend_name, move |device| {
                    init_runtime(device, &wgpu_config)
                });
            if stage_profile {
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-recurrent] calls={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={}",
                    snapshot.calls,
                    snapshot.total_ns,
                    snapshot.setup_ns,
                    snapshot.copy_ns,
                    snapshot.dispatch_ns,
                );
            }
            result
        }
        BackendArg::WgpuNoFusion => {
            let stage_profile = std::env::var_os("BDH_STAGE_PROFILE").is_some();
            if stage_profile {
                recurrent_profile_reset();
            }
            let wgpu_config = config.wgpu.clone();
            let backend_name = if config.wgpu.training.fused_core_recurrent == Some(true) {
                "wgpu-fused-core"
            } else {
                "wgpu-nofusion"
            };
            let result =
                train_language::<Autodiff<WgpuNoFusion>, _>(&config, backend_name, move |device| {
                    init_runtime(device, &wgpu_config)
                });
            if stage_profile {
                let snapshot = recurrent_profile_snapshot();
                eprintln!(
                    "[stage-profile][training-recurrent] calls={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={}",
                    snapshot.calls,
                    snapshot.total_ns,
                    snapshot.setup_ns,
                    snapshot.copy_ns,
                    snapshot.dispatch_ns,
                );
            }
            result
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_language::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    })
}

#[cfg(all(feature = "train", feature = "ddp"))]
fn backend_arg_cli_value(backend: BackendArg) -> &'static str {
    match backend {
        BackendArg::Cuda => "cuda",
        BackendArg::Wgpu => "wgpu",
        BackendArg::WgpuNoFusion => "wgpu-no-fusion",
        BackendArg::Ndarray => "ndarray",
    }
}

#[cfg(all(feature = "train", feature = "ddp"))]
fn write_language_local_ddp_overlay(
    root: &PathBuf,
    rank: usize,
    world_size: usize,
    orchestrator_port: u16,
) -> Result<PathBuf> {
    fs::create_dir_all(root).map_err(|err| {
        anyhow!(
            "failed to create DDP overlay directory {}: {err}",
            root.display()
        )
    })?;
    let node_port = orchestrator_port
        .checked_add(1)
        .and_then(|base| base.checked_add(rank as u16))
        .ok_or_else(|| anyhow!("orchestrator port overflow for local DDP launch"))?;
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
    fs::write(&overlay, contents)
        .map_err(|err| anyhow!("failed to write DDP overlay {}: {err}", overlay.display()))?;
    Ok(overlay)
}

#[cfg(all(feature = "train", feature = "ddp"))]
fn run_language_local_ddp(args: LanguageLocalDdpArgs) -> Result<()> {
    if args.world_size == 0 {
        return Err(anyhow!("--world-size must be at least 1"));
    }

    let current_exe = std::env::current_exe().context("resolve current train binary")?;
    let config_paths = default_or_explicit_config_paths("config/language/base.toml", &args.config);
    let config = load_language_training_config(&config_paths)?;
    let run_root = resolve_run_root_for_config_paths("language", &config.run_layout, &config_paths);
    let resume_run_dir = resolve_resume_run_dir(
        &run_root,
        config.training.resume_run_dir.as_deref(),
        config.training.launch_mode,
    )?;
    let planned_run = plan_run_artifacts(&run_root, resume_run_dir.as_deref())?;
    let run_name = planned_run.run_name;
    let run_dir = planned_run.run_dir;

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
            .arg("--backend")
            .arg(backend_arg_cli_value(args.backend))
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
            first_error = Some(anyhow!("local DDP rank {rank} exited with status {status}"));
        }
    }

    let _ = orchestrator.kill();
    let _ = orchestrator.wait();

    if let Some(err) = first_error {
        return Err(err);
    }

    Ok(())
}

#[cfg(all(feature = "train", feature = "ddp"))]
fn run_collective_orchestrator(args: CollectiveOrchestratorArgs) -> Result<()> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(start_global_orchestrator(args.port));
    Ok(())
}

#[cfg(all(test, feature = "train"))]
mod tests {
    use super::{
        apply_wgpu_vision_training_overrides, backend_arg_cli_value,
        default_or_explicit_config_paths, write_language_local_ddp_overlay,
    };
    use burn_dragon::vision::VisionTrainingConfig;
    use std::path::PathBuf;
    use tempfile::tempdir;

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

    #[test]
    fn apply_wgpu_vision_training_overrides_enables_fused_kernels() {
        let mut config =
            VisionTrainingConfig::scene_slot_graph_bridge_multimode_spatial_imagenet1k_medium_launch();
        config.vision.fused_kernels = false;
        config.wgpu.training.fused_core_recurrent = Some(true);

        apply_wgpu_vision_training_overrides(&mut config, "wgpu");

        assert!(config.vision.fused_kernels);
    }

    #[test]
    fn apply_wgpu_vision_training_overrides_ignores_non_wgpu_backend() {
        let mut config =
            VisionTrainingConfig::scene_slot_graph_bridge_multimode_spatial_imagenet1k_medium_launch();
        config.vision.fused_kernels = false;
        config.wgpu.training.fused_core_recurrent = Some(true);

        apply_wgpu_vision_training_overrides(&mut config, "cuda");

        assert!(!config.vision.fused_kernels);
    }

    #[cfg(feature = "ddp")]
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

    #[cfg(feature = "ddp")]
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

#[cfg(feature = "train")]
fn run_vision(args: VisionArgs) -> Result<()> {
    let mut config_paths = vec![PathBuf::from("config/vision/base.toml")];
    config_paths.extend(args.config);
    let config = load_vision_training_config(&config_paths)?;

    run_in_training_thread("vision-train", move || match args.backend {
        BackendArg::Ndarray => {
            train_vision_backend::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})
        }
        BackendArg::Wgpu => {
            let mut resolved = config.clone();
            apply_wgpu_vision_training_overrides(&mut resolved, "wgpu");
            let wgpu_config = resolved.wgpu.clone();
            let backend_name = if resolved.vision.fused_kernels {
                "wgpu-fused-core"
            } else {
                "wgpu-nofusion"
            };
            eprintln!(
                "vision training: routing --backend wgpu through CubeBackend (backend={backend_name}) to enable the measured fused pyramid path"
            );
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(
                &resolved,
                backend_name,
                move |device| init_runtime(device, &wgpu_config),
            )
        }
        BackendArg::WgpuNoFusion => {
            let mut resolved = config.clone();
            resolved.vision.fused_kernels = false;
            let wgpu_config = resolved.wgpu.clone();
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(
                &resolved,
                "wgpu-nofusion",
                move |device| init_runtime(device, &wgpu_config),
            )
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_vision_backend::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    })
}

#[cfg(feature = "train")]
fn run_sudoku(args: SudokuArgs) -> Result<()> {
    let config = load_sudoku_training_config(&args.config)?;

    run_in_training_thread("sudoku-train", move || match args.backend {
        BackendArg::Ndarray => train_sudoku::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {}),
        BackendArg::Wgpu => {
            let wgpu_config = config.wgpu.clone();
            train_sudoku::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", move |device| {
                init_runtime(device, &wgpu_config)
            })
        }
        BackendArg::WgpuNoFusion => {
            let wgpu_config = config.wgpu.clone();
            train_sudoku::<Autodiff<WgpuNoFusion>, _>(&config, "wgpu-nofusion", move |device| {
                init_runtime(device, &wgpu_config)
            })
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_sudoku::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    })
}

#[cfg(feature = "train")]
fn run_multimodal(args: MultimodalArgs) -> Result<()> {
    let mut config_paths = vec![PathBuf::from("config/multimodal/base.toml")];
    config_paths.extend(args.config);
    let config = load_multimodal_runtime_config(&config_paths)?;

    run_in_training_thread("multimodal-train", move || match config {
        MultimodalRuntimeConfig::ImageText(config) => match args.backend {
            BackendArg::Ndarray => {
                train_multimodal::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})
            }
            BackendArg::Wgpu => {
                let wgpu_config = config.wgpu.clone();
                train_multimodal::<Autodiff<WgpuNoFusion>, _>(&config, "wgpu", move |device| {
                    init_runtime(device, &wgpu_config)
                })
            }
            BackendArg::WgpuNoFusion => {
                let wgpu_config = config.wgpu.clone();
                train_multimodal::<Autodiff<WgpuNoFusion>, _>(
                    &config,
                    "wgpu-nofusion",
                    move |device| init_runtime(device, &wgpu_config),
                )
            }
            BackendArg::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    train_multimodal::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow!(
                        "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                    ))
                }
            }
        },
        MultimodalRuntimeConfig::VideoText(config) => match args.backend {
            BackendArg::Ndarray => {
                train_multimodal_video::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})
            }
            BackendArg::Wgpu => {
                let wgpu_config = config.wgpu.clone();
                train_multimodal_video::<Autodiff<WgpuNoFusion>, _>(
                    &config,
                    "wgpu",
                    move |device| init_runtime(device, &wgpu_config),
                )
            }
            BackendArg::WgpuNoFusion => {
                let wgpu_config = config.wgpu.clone();
                train_multimodal_video::<Autodiff<WgpuNoFusion>, _>(
                    &config,
                    "wgpu-nofusion",
                    move |device| init_runtime(device, &wgpu_config),
                )
            }
            BackendArg::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    train_multimodal_video::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow!(
                        "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                    ))
                }
            }
        },
    })
}

#[cfg(feature = "train")]
fn run() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Language(cmd) => run_language(cmd),
        #[cfg(feature = "ddp")]
        Command::LanguageLocalDdp(cmd) => run_language_local_ddp(cmd),
        #[cfg(feature = "ddp")]
        Command::CollectiveOrchestrator(cmd) => run_collective_orchestrator(cmd),
        Command::Vision(cmd) => run_vision(cmd),
        Command::Sudoku(cmd) => run_sudoku(cmd),
        Command::Multimodal(cmd) => run_multimodal(cmd),
    }
}

#[cfg(feature = "train")]
fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

#[cfg(not(feature = "train"))]
fn main() {
    eprintln!("burn_dragon_cli train binary requires the `train` feature.");
    std::process::exit(1);
}
