#![recursion_limit = "256"]

#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
use burn_dragon_vision::config::load_vision_training_config;
#[cfg(feature = "cli")]
use burn_dragon_vision::train::train_vision_backend;

#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cli")]
use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
use burn_wgpu::Wgpu;

#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::init_runtime;

#[cfg(all(feature = "cuda", feature = "cli"))]
use burn_cuda::Cuda;

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
            train_vision_backend::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", |device| {
                init_runtime(device, &config.wgpu)
            })?;
        }
        "wgpu-nofusion" => {
            use burn_wgpu::{CubeBackend, WgpuRuntime};
            type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(&config, "wgpu-nofusion", |device| {
                init_runtime(device, &config.wgpu)
            })?;
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                train_vision_backend::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!("cuda backend requested but feature `cuda` is not enabled"));
            }
        }
        other => {
            return Err(anyhow!(
                "unknown backend `{other}` (expected cpu, wgpu, wgpu-nofusion, or cuda)"
            ));
        }
    }

    Ok(())
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("burn_dragon_vision train binary requires the `cli` feature.");
}
