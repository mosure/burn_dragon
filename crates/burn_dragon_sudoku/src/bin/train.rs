#![recursion_limit = "256"]

#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
use burn_dragon_sudoku::config::load_training_config;
#[cfg(feature = "cli")]
use burn_dragon_sudoku::train::train_backend;

#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cli")]
use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
use burn_wgpu::Wgpu;

#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::{init_runtime, WgpuDevice};

#[cfg(feature = "cuda")]
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
    let config = load_training_config(&args.config)?;

    match args.backend.as_str() {
        "cpu" | "ndarray" => {
            train_backend::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {})?;
        }
        "wgpu" => {
            let device = WgpuDevice::default();
            init_runtime(&device, &config.wgpu);
            train_backend::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", |_| {})?;
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                return Err(anyhow!("cuda backend requested but feature `cuda` is not enabled"));
            }
        }
        other => {
            return Err(anyhow!(
                "unknown backend `{other}` (expected cpu, wgpu, or cuda)"
            ));
        }
    }

    Ok(())
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("burn_dragon_sudoku train binary requires the `cli` feature.");
}
