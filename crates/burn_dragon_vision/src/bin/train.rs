#![recursion_limit = "256"]

#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
use burn_dragon_vision::VisionTrainingConfig;
#[cfg(feature = "cli")]
use burn_dragon_vision::config::load_vision_training_config;
#[cfg(feature = "cli")]
use burn_dragon_vision::train::train_vision_backend;

#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cli")]
use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
use burn_wgpu::{CubeBackend, WgpuRuntime};

#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::init_runtime;
#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::is_wgpu_backend_name;

#[cfg(all(feature = "cuda", feature = "cli"))]
use burn_cuda::Cuda;

#[cfg(feature = "cli")]
type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, value_name = "PATH", required = true)]
    config: Vec<PathBuf>,
    #[arg(long, default_value = "cpu")]
    backend: String,
}

#[cfg(feature = "cli")]
fn main() -> Result<()> {
    let args = Args::parse();
    let stack_mb = std::env::var("BDH_TRAIN_STACK_MB")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(64);
    let stack_bytes = stack_mb.max(8) * 1024 * 1024;
    let handle = std::thread::Builder::new()
        .name("vision-train".to_string())
        .stack_size(stack_bytes)
        .spawn(move || run(args))
        .map_err(|err| anyhow!("failed to spawn training thread: {err}"))?;
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(anyhow!("training thread panicked")),
    }
}

#[cfg(feature = "cli")]
fn run(args: Args) -> Result<()> {
    let config = load_vision_training_config(&args.config)?;

    match args.backend.as_str() {
        "cpu" | "ndarray" => {
            train_vision_backend::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})?;
        }
        "wgpu" => {
            let mut resolved = config.clone();
            apply_wgpu_vision_training_overrides(&mut resolved, "wgpu");
            let backend_name = if resolved.vision.fused_kernels {
                "wgpu-fused-core"
            } else {
                "wgpu-nofusion"
            };
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(&resolved, backend_name, |device| {
                init_runtime(device, &resolved.wgpu)
            })?;
        }
        "wgpu-nofusion" | "wgpu-no-fusion" => {
            let mut resolved = config.clone();
            resolved.vision.fused_kernels = false;
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(
                &resolved,
                "wgpu-nofusion",
                |device| init_runtime(device, &resolved.wgpu),
            )?;
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                train_vision_backend::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!(
                    "cuda backend requested but feature `cuda` is not enabled"
                ));
            }
        }
        other => {
            return Err(anyhow!(
                "unknown backend `{other}` (expected cpu, wgpu, wgpu-nofusion/wgpu-no-fusion, or cuda)"
            ));
        }
    }

    Ok(())
}

#[cfg(feature = "cli")]
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

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("burn_dragon_vision train binary requires the `cli` feature.");
}
