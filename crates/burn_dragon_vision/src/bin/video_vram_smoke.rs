#![recursion_limit = "256"]

#[cfg(feature = "cli")]
use std::collections::HashSet;
#[cfg(feature = "cli")]
use std::fs;
#[cfg(feature = "cli")]
use std::path::{Path, PathBuf};

#[cfg(feature = "cli")]
use anyhow::{Context, Result, anyhow};
#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cli")]
use burn_dragon_train::wgpu::init_runtime;
#[cfg(feature = "cli")]
use burn_dragon_vision::{config::load_vision_training_config, train::train_vision_backend};
#[cfg(feature = "cli")]
use burn_wgpu::Wgpu;
#[cfg(feature = "cli")]
use clap::Parser;

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about)]
struct Args {
    #[arg(short, long, value_name = "PATH", required = true)]
    config: Vec<PathBuf>,
    #[arg(long, default_value = "wgpu-nofusion")]
    backend: String,
    #[arg(long, default_value_t = 1024.0)]
    max_reserved_growth_mb: f64,
    #[arg(long, default_value_t = 256.0)]
    max_in_use_growth_mb: f64,
    #[arg(long, default_value_t = 16)]
    min_artifacts: usize,
    #[arg(long, default_value_t = 2)]
    warmup_epochs: usize,
}

#[cfg(feature = "cli")]
fn vision_run_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("runs")
        .join("vision")
}

#[cfg(feature = "cli")]
fn current_run_dirs(root: &Path) -> HashSet<PathBuf> {
    fs::read_dir(root)
        .ok()
        .into_iter()
        .flat_map(|entries| entries.filter_map(Result::ok))
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_dir() && path.file_name().and_then(|n| n.to_str()) != Some("latest") {
                Some(path)
            } else {
                None
            }
        })
        .collect()
}

#[cfg(feature = "cli")]
fn newest_added_run_dir(before: &HashSet<PathBuf>, root: &Path) -> Result<PathBuf> {
    let mut added: Vec<_> = current_run_dirs(root).difference(before).cloned().collect();
    if added.is_empty() {
        let latest = root.join("latest");
        if latest.exists() {
            return fs::canonicalize(latest).context("canonicalize latest run");
        }
        return Err(anyhow!(
            "no new run directory created under {}",
            root.display()
        ));
    }
    added.sort();
    Ok(added.pop().expect("new run dir"))
}

#[cfg(feature = "cli")]
fn parse_device_memory_log(path: &Path) -> Result<Vec<(f64, f64)>> {
    let contents = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut values = Vec::new();
    for line in contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let metric = line.split(',').next().unwrap_or(line).trim();
        let mut parts = metric.split('/');
        let reserved = parts
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches(" MiB")
            .parse::<f64>()
            .with_context(|| format!("parse reserved from `{metric}`"))?;
        let in_use = parts
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches(" MiB")
            .parse::<f64>()
            .with_context(|| format!("parse in-use from `{metric}`"))?;
        values.push((reserved, in_use));
    }
    if values.is_empty() {
        return Err(anyhow!("no device memory samples in {}", path.display()));
    }
    Ok(values)
}

#[cfg(feature = "cli")]
fn max_reserved(values: &[(f64, f64)]) -> f64 {
    values
        .iter()
        .map(|(reserved, _)| *reserved)
        .fold(0.0, f64::max)
}

#[cfg(feature = "cli")]
fn max_in_use(values: &[(f64, f64)]) -> f64 {
    values.iter().map(|(_, in_use)| *in_use).fold(0.0, f64::max)
}

#[cfg(feature = "cli")]
fn main() -> Result<()> {
    let args = Args::parse();
    let config = load_vision_training_config(&args.config)?;
    let total_epochs = config
        .training
        .epochs
        .ok_or_else(|| anyhow!("video_vram_smoke requires training.epochs to be set"))?;
    if total_epochs <= args.warmup_epochs {
        return Err(anyhow!(
            "training.epochs={} must be greater than warmup_epochs={}",
            total_epochs,
            args.warmup_epochs
        ));
    }

    let run_root = vision_run_root();
    fs::create_dir_all(&run_root).context("create vision run root")?;
    let before = current_run_dirs(&run_root);

    match args.backend.as_str() {
        "wgpu" => {
            train_vision_backend::<Autodiff<Wgpu<f32>>, _>(&config, "wgpu", |device| {
                init_runtime(device, &config.wgpu)
            })?;
        }
        "wgpu-nofusion" => {
            use burn_wgpu::{CubeBackend, WgpuRuntime};
            type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
            train_vision_backend::<Autodiff<WgpuNoFusion>, _>(
                &config,
                "wgpu-nofusion",
                |device| init_runtime(device, &config.wgpu),
            )?;
        }
        other => {
            return Err(anyhow!(
                "unsupported backend `{other}` for video_vram_smoke (expected wgpu or wgpu-nofusion)"
            ));
        }
    }

    let run_dir = newest_added_run_dir(&before, &run_root)?;
    let previous_epoch = total_epochs - 1;
    let last_epoch = total_epochs;
    let valid_previous = parse_device_memory_log(
        &run_dir.join(format!("valid/epoch-{previous_epoch}/device_memory_mb.log")),
    )?;
    let valid_last = parse_device_memory_log(
        &run_dir.join(format!("valid/epoch-{last_epoch}/device_memory_mb.log")),
    )?;

    let reserved_growth = max_reserved(&valid_last) - max_reserved(&valid_previous);
    let in_use_growth = max_in_use(&valid_last) - max_in_use(&valid_previous);

    if reserved_growth > args.max_reserved_growth_mb {
        return Err(anyhow!(
            "reserved device memory grew by {reserved_growth:.1} MiB (> {:.1}); run_dir={}",
            args.max_reserved_growth_mb,
            run_dir.display()
        ));
    }
    if in_use_growth > args.max_in_use_growth_mb {
        return Err(anyhow!(
            "in-use device memory grew by {in_use_growth:.1} MiB (> {:.1}); run_dir={}",
            args.max_in_use_growth_mb,
            run_dir.display()
        ));
    }

    let artifact_count = fs::read_dir(run_dir.join("artifacts"))
        .with_context(|| format!("read {}/artifacts", run_dir.display()))?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("mp4"))
        .count();

    if artifact_count < args.min_artifacts {
        return Err(anyhow!(
            "expected at least {} artifact mp4s, found {} in {}",
            args.min_artifacts,
            artifact_count,
            run_dir.display()
        ));
    }

    println!(
        "video_vram_smoke passed: run={} compare_epochs={}->{} reserved_growth_mb={:.1} in_use_growth_mb={:.1} artifacts={}",
        run_dir.display(),
        previous_epoch,
        last_epoch,
        reserved_growth,
        in_use_growth,
        artifact_count
    );

    Ok(())
}

#[cfg(not(feature = "cli"))]
fn main() {
    eprintln!("video_vram_smoke requires the `cli` feature.");
}
