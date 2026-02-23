#![recursion_limit = "256"]

#[cfg(feature = "train")]
use std::path::PathBuf;
#[cfg(feature = "train")]
use std::sync::atomic::Ordering;

#[cfg(all(feature = "train", target_os = "windows"))]
use anyhow::Context;
#[cfg(feature = "train")]
use anyhow::Result;
#[cfg(all(feature = "train", any(target_os = "windows", not(feature = "cuda"))))]
use anyhow::anyhow;
#[cfg(feature = "train")]
use burn::tensor::backend::AutodiffBackend;
#[cfg(feature = "train")]
use burn_autodiff::Autodiff;
#[cfg(feature = "train")]
use burn_dragon::language::train::{
    build_vocab_only, prepare_dataset, train_backend as train_language_backend,
};
#[cfg(feature = "train")]
use burn_dragon::language::{
    TrainingConfig as LanguageTrainingConfig, load_training_config as load_language_training_config,
};
#[cfg(feature = "train")]
use burn_dragon::train::train::constants::FAST_TRAIN;
#[cfg(feature = "train")]
use burn_dragon::train::wgpu::init_runtime;
#[cfg(feature = "train")]
use burn_dragon::vision::load_vision_training_config;
#[cfg(feature = "train")]
use burn_dragon::vision::train::train_vision_backend;
#[cfg(feature = "train")]
use burn_dragon_sudoku::config::{
    SudokuTrainingConfig, load_training_config as load_sudoku_training_config,
};
#[cfg(feature = "train")]
use burn_dragon_sudoku::train::train_backend as train_sudoku_backend;
#[cfg(feature = "train")]
use burn_dragon_wgpu::{recurrent_profile_reset, recurrent_profile_snapshot};
#[cfg(feature = "train")]
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use burn_wgpu::{CubeBackend, Wgpu, WgpuRuntime};
#[cfg(feature = "train")]
use clap::{Args, Parser, Subcommand, ValueEnum};

#[cfg(all(feature = "train", feature = "cuda"))]
use burn_cuda::Cuda;

#[cfg(feature = "train")]
type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;

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
    /// Train vision distill/LeJEPA models.
    Vision(VisionArgs),
    /// Train sudoku models.
    Sudoku(SudokuArgs),
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
fn run_language(args: LanguageArgs) -> Result<()> {
    let mut config_paths = vec![PathBuf::from("config/language/base.toml")];
    config_paths.extend(args.config);
    let config = load_language_training_config(&config_paths)?;

    if args.build_vocab_only {
        build_vocab_only(&config)?;
        return Ok(());
    }

    FAST_TRAIN.store(config.training.fast_train, Ordering::Relaxed);

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
                "wgpu"
            };
            let result =
                train_language::<Autodiff<Wgpu<f32>>, _>(&config, backend_name, move |device| {
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
            let wgpu_config = config.wgpu.clone();
            train_vision_backend::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", move |device| {
                init_runtime(device, &wgpu_config)
            })
        }
        BackendArg::WgpuNoFusion => {
            let wgpu_config = config.wgpu.clone();
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(
                &config,
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
fn run() -> Result<()> {
    let args = Cli::parse();
    match args.command {
        Command::Language(cmd) => run_language(cmd),
        Command::Vision(cmd) => run_vision(cmd),
        Command::Sudoku(cmd) => run_sudoku(cmd),
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
