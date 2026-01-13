#![cfg(feature = "integration_test")]

use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use burn_dragon_hatchling_core::load_vision_training_config;
use burn_dragon_hatchling_core::train::{
    gdpo_cpu_fallbacks, gdpo_reset_cpu_fallbacks, train_vision_backend_for_test,
};
use burn_dragon_hatchling_core::VisionTrainingModeConfig;

#[cfg(feature = "cuda")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

fn vision_saccade_tiny_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision_saccade_tiny.toml")
}

#[cfg(feature = "cuda")]
fn query_nvidia_smi() -> Option<(f32, f32)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout.lines().next()?.trim();
    let mut parts = line.split(',');
    let util = parts.next()?.trim().parse::<f32>().ok()?;
    let mem = parts.next()?.trim().parse::<f32>().ok()?;
    Some((util, mem))
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_vision_saccade_tiny_training_gpu_utilization() {
    if std::env::var("BDH_SKIP_GPU_UTIL").is_ok() {
        eprintln!("BDH_SKIP_GPU_UTIL set; skipping GPU utilization check.");
        return;
    }

    if query_nvidia_smi().is_none() {
        panic!("nvidia-smi not available; set BDH_SKIP_GPU_UTIL=1 to skip");
    }

    unsafe {
        std::env::set_var("BDH_MEMORY_CLEANUP_ITERS", "1");
    }

    let config_path = vision_saccade_tiny_path();
    let mut config =
        load_vision_training_config(&[config_path]).expect("load vision_saccade_tiny");
    let max_iters = std::env::var("BDH_TRAIN_MAX_ITERS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);
    config.training.max_iters = max_iters.max(1);
    config.training.epochs = None;
    config.training.log_frequency = max_iters.max(1);
    config.dataset.max_records = Some(64);

    if let VisionTrainingModeConfig::Saccade(ref mut saccade) = config.mode {
        saccade.artifact_every = 0;
    }

    gdpo_reset_cpu_fallbacks();

    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let handle = thread::spawn(move || {
        let result = train_vision_backend_for_test::<Autodiff<Cuda<f32>>, _>(
            &config,
            "cuda",
            |_| {},
        );
        done_clone.store(true, Ordering::Relaxed);
        result
    });

    let sample_interval_ms = std::env::var("BDH_GPU_UTIL_INTERVAL_MS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(250);
    let timeout_secs = std::env::var("BDH_GPU_UTIL_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(90);
    let warmup_samples = std::env::var("BDH_GPU_UTIL_WARMUP_SAMPLES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);

    let mut util_samples = Vec::new();
    let mut mem_samples = Vec::new();
    let start = Instant::now();
    loop {
        if let Some((util, mem)) = query_nvidia_smi() {
            util_samples.push(util);
            mem_samples.push(mem);
        }

        if done.load(Ordering::Relaxed) || handle.is_finished() {
            break;
        }
        if start.elapsed() > Duration::from_secs(timeout_secs) {
            panic!("training timed out after {timeout_secs}s");
        }
        thread::sleep(Duration::from_millis(sample_interval_ms));
    }

    let result = handle.join().expect("train thread panicked");
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let cpu_fallbacks = gdpo_cpu_fallbacks();
    assert!(
        cpu_fallbacks == 0,
        "gdpo percentile fell back to CPU {cpu_fallbacks} times"
    );

    let slice_start = warmup_samples.min(util_samples.len());
    let slice = &util_samples[slice_start..];
    let avg_util = if slice.is_empty() {
        0.0
    } else {
        slice.iter().sum::<f32>() / slice.len() as f32
    };
    let min_util = std::env::var("BDH_MIN_GPU_UTIL")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(15.0);
    eprintln!(
        "GPU utilization avg={avg_util:.2}%, samples={}, min_required={min_util:.2}%",
        util_samples.len()
    );
    assert!(
        avg_util >= min_util,
        "avg GPU utilization too low: {avg_util:.2}% (< {min_util:.2}%)"
    );

    let (min_mem, max_mem) = mem_samples.iter().fold(
        (f32::INFINITY, f32::NEG_INFINITY),
        |(min_val, max_val), &value| (min_val.min(value), max_val.max(value)),
    );
    let mem_growth = (max_mem - min_mem).max(0.0);
    eprintln!(
        "GPU memory growth={mem_growth:.1} MiB (min={min_mem:.1}, max={max_mem:.1})"
    );
    let max_growth = std::env::var("BDH_MAX_MEM_GROWTH_MB")
        .ok()
        .and_then(|value| value.parse::<f32>().ok())
        .unwrap_or(2048.0);
    assert!(
        mem_growth <= max_growth,
        "GPU memory growth too high: {mem_growth:.1} MiB (> {max_growth:.1} MiB)"
    );
}
