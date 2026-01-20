#[cfg(feature = "cli")]
use crate::train::prelude::*;
#[cfg(feature = "cli")]
use crate::train::vision::train_vision_backend;
#[cfg(feature = "cli")]
use burn_dragon_language::train::train_backend;
#[cfg(feature = "cli")]
use burn_dragon_language::{load_training_config, train::build_vocab_only, train::prepare_dataset};

#[cfg(feature = "cli")]
fn run_in_training_thread<F, T>(name: &str, work: F) -> Result<T>
where
    F: FnOnce() -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    #[cfg(target_os = "windows")]
    {
        let handle = std::thread::Builder::new()
            .name(name.to_string())
            .spawn(work)
            .context("spawn training thread")?;
        handle
            .join()
            .map_err(|_| anyhow!("training thread panicked"))?
    }
    #[cfg(not(target_os = "windows"))]
    {
        work()
    }
}

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Cli {
    #[command(flatten)]
    train: TrainArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[cfg(feature = "cli")]
#[derive(ClapArgs, Debug)]
struct TrainArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH", global = true)]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[cfg(feature = "cli")]
#[derive(Subcommand, Debug)]
enum Command {
    /// Build the character-level vocabulary and exit.
    BuildVocab,
    /// Train the vision model (distill or LeJEPA).
    Vision,
}

#[cfg(feature = "cli")]
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
    WgpuNoFusion,
}

#[cfg(feature = "cli")]
pub fn run_cli() -> Result<()> {
    let args = Cli::parse();

    if matches!(args.command, Some(Command::Vision)) {
        let backend = args.train.backend;
        let mut config_paths = vec![PathBuf::from("config/vision/base.toml")];
        config_paths.extend(args.train.config.clone());
        let config = load_vision_training_config(&config_paths)?;
        return run_in_training_thread("vision-train", move || match backend {
            BackendArg::Wgpu => {
                train_vision_backend::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", |device| {
                    init_runtime(device, &config.wgpu)
                })
            }
            BackendArg::WgpuNoFusion => {
                use burn_wgpu::{CubeBackend, WgpuRuntime};
                type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
                train_vision_backend::<Autodiff<WgpuNoFusion>, _>(&config, "wgpu-nofusion", |device| {
                    init_runtime(device, &config.wgpu)
                })
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
        });
    }

    let backend = args.train.backend;
    let mut config_paths = vec![PathBuf::from("config/language/base.toml")];
    config_paths.extend(args.train.config.clone());
    let config = load_training_config(&config_paths)?;
    FAST_TRAIN.store(config.training.fast_train, Ordering::Relaxed);

    if matches!(args.command, Some(Command::BuildVocab)) {
        build_vocab_only(&config)?;
        return Ok(());
    }

    let dataset = prepare_dataset(&config.dataset, &config.training)?;

    run_in_training_thread("train", move || match backend {
        BackendArg::Wgpu => train_backend::<Autodiff<Wgpu<f32>>, _>(
            &config,
            Arc::clone(&dataset),
            "wgpu",
            |device| init_runtime(device, &config.wgpu),
        ),
        BackendArg::WgpuNoFusion => {
            use burn_wgpu::{CubeBackend, WgpuRuntime};
            type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
            train_backend::<Autodiff<WgpuNoFusion>, _>(
                &config,
                Arc::clone(&dataset),
                "wgpu-nofusion",
                |device| init_runtime(device, &config.wgpu),
            )
        }
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<Autodiff<Cuda<f32>>, _>(&config, dataset, "cuda", |_| {})
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
