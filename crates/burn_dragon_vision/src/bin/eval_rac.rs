#![recursion_limit = "256"]

#[cfg(feature = "cli")]
use std::path::PathBuf;

#[cfg(feature = "cli")]
use anyhow::{Result, anyhow};
#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(all(feature = "cuda", feature = "cli"))]
use burn_cuda::Cuda;
#[cfg(feature = "cli")]
use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
use burn_wgpu::Wgpu;

#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::init_runtime;
#[cfg(feature = "cli")]
use burn_dragon_vision::config::load_vision_training_config;
#[cfg(feature = "cli")]
use burn_dragon_vision::train::eval_vision_rac_checkpoint_backend;

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, value_name = "PATH", required = true)]
    config: Vec<PathBuf>,
    #[arg(long, value_name = "PATH")]
    checkpoint: PathBuf,
    #[arg(long, value_name = "PATH")]
    output_dir: Option<PathBuf>,
    #[arg(long, value_name = "N")]
    max_images: Option<usize>,
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
        .name("vision-rac-eval".to_string())
        .stack_size(stack_bytes)
        .spawn(move || run(args))
        .map_err(|err| anyhow!("failed to spawn eval thread: {err}"))?;
    match handle.join() {
        Ok(result) => result,
        Err(_) => Err(anyhow!("eval thread panicked")),
    }
}

#[cfg(feature = "cli")]
fn run(args: Args) -> Result<()> {
    let config = load_vision_training_config(&args.config)?;
    let summary = match args.backend.as_str() {
        "cpu" | "ndarray" => eval_vision_rac_checkpoint_backend::<Autodiff<NdArray<f32>>, _>(
            &config,
            &args.checkpoint,
            "cpu",
            |_| {},
            args.output_dir.as_deref(),
            args.max_images,
        )?,
        "wgpu" => eval_vision_rac_checkpoint_backend::<Autodiff<Wgpu<f32>>, _>(
            &config,
            &args.checkpoint,
            "wgpu",
            |device| init_runtime(device, &config.wgpu),
            args.output_dir.as_deref(),
            args.max_images,
        )?,
        "wgpu-nofusion" => {
            use burn_wgpu::{CubeBackend, WgpuRuntime};
            type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
            eval_vision_rac_checkpoint_backend::<Autodiff<WgpuNoFusion>, _>(
                &config,
                &args.checkpoint,
                "wgpu-nofusion",
                |device| init_runtime(device, &config.wgpu),
                args.output_dir.as_deref(),
                args.max_images,
            )?
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                eval_vision_rac_checkpoint_backend::<Autodiff<Cuda<f32>>, _>(
                    &config,
                    &args.checkpoint,
                    "cuda",
                    |_| {},
                    args.output_dir.as_deref(),
                    args.max_images,
                )?
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
                "unknown backend `{other}` (expected cpu, wgpu, wgpu-nofusion, or cuda)"
            ));
        }
    };

    println!(
        "rac checkpoint eval: backend={} checkpoint={} export_dir={} images={}",
        summary.backend,
        summary.checkpoint.display(),
        summary.export_dir.display(),
        summary.images
    );
    println!(
        "roundtrip: recon={:.6} psnr={:.6} | forward: recon={:.6} psnr={:.6}",
        summary.roundtrip_recon_loss,
        summary.roundtrip_recon_psnr,
        summary.forward_recon_loss,
        summary.forward_recon_psnr,
    );
    println!(
        "directional: path={:.6} vel={:.6} reverse_latent={:.6} reverse_to_init={:.6}",
        summary.forward_path_loss,
        summary.forward_velocity_loss,
        summary.reverse_latent_loss,
        summary.reverse_to_init_loss,
    );
    if let Some(semantic_loss) = summary.semantic_loss {
        println!("semantic: loss={semantic_loss:.6}");
    }

    Ok(())
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("burn_dragon_vision eval_rac binary requires the `cli` feature.");
}
