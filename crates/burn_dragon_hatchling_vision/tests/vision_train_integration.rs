#![cfg(feature = "integration_test")]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use serde::Deserialize;

use burn_dragon_hatchling_core::load_vision_training_config;
use burn_dragon_hatchling_vision::train::{
    gdpo_cpu_fallbacks, gdpo_reset_cpu_fallbacks, loss_trace_len, loss_trace_reset,
    loss_trace_take, train_vision_backend_for_test,
};
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

fn vision_saccade_tiny_integration_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision_saccade_tiny_integration.toml")
}

#[derive(Debug, Deserialize)]
#[serde(default)]
struct IntegrationSettings {
    skip_gpu_util: bool,
    gpu_util_interval_ms: u64,
    gpu_util_timeout_secs: u64,
    gpu_util_warmup_samples: usize,
    min_train_loss: f32,
    max_train_loss: f32,
    min_gpu_util: f32,
    max_mem_growth_mb: f32,
}

impl Default for IntegrationSettings {
    fn default() -> Self {
        Self {
            skip_gpu_util: false,
            gpu_util_interval_ms: 250,
            gpu_util_timeout_secs: 1800,
            gpu_util_warmup_samples: 4,
            min_train_loss: 0.05,
            max_train_loss: 10.0,
            min_gpu_util: 70.0,
            max_mem_growth_mb: 8192.0,
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct TestSettings {
    #[serde(default)]
    test: TestSection,
}

#[derive(Debug, Default, Deserialize)]
struct TestSection {
    #[serde(default)]
    integration: IntegrationSettings,
}

fn load_integration_settings(path: &Path) -> IntegrationSettings {
    let contents = fs::read_to_string(path).expect("read integration config");
    let settings: TestSettings = toml::from_str(&contents).expect("parse integration config");
    settings.test.integration
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
    let integration_path = vision_saccade_tiny_integration_path();
    let settings = load_integration_settings(&integration_path);
    if settings.skip_gpu_util {
        eprintln!("GPU utilization check disabled in config.");
        return;
    }

    if query_nvidia_smi().is_none() {
        panic!("nvidia-smi not available; disable test.integration.skip_gpu_util to skip");
    }

    let config_path = vision_saccade_tiny_path();
    let config =
        load_vision_training_config(&[config_path, integration_path]).expect("load test config");

    gdpo_reset_cpu_fallbacks();
    loss_trace_reset();

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

    let sample_interval_ms = settings.gpu_util_interval_ms;
    let timeout_secs = settings.gpu_util_timeout_secs;
    let warmup_samples = settings.gpu_util_warmup_samples;

    let mut util_samples = Vec::new();
    let mut mem_samples = Vec::new();
    let start = Instant::now();

    while loss_trace_len() == 0 {
        if done.load(Ordering::Relaxed) || handle.is_finished() {
            break;
        }
        if start.elapsed() > Duration::from_secs(timeout_secs) {
            panic!("training timed out after {timeout_secs}s");
        }
        thread::sleep(Duration::from_millis(sample_interval_ms.min(100)));
    }
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

    let losses = loss_trace_take();
    assert!(
        !losses.is_empty(),
        "no training loss samples captured; enable training.trace_train_loss in the integration config"
    );
    let mut min_loss = f32::INFINITY;
    let mut max_loss = f32::NEG_INFINITY;
    let mut sum_loss = 0.0f32;
    for loss in &losses {
        assert!(loss.is_finite(), "loss contains non-finite value: {loss}");
        min_loss = min_loss.min(*loss);
        max_loss = max_loss.max(*loss);
        sum_loss += loss;
    }
    let avg_loss = sum_loss / losses.len().max(1) as f32;
    let min_expected = settings.min_train_loss;
    let max_expected = settings.max_train_loss;
    eprintln!(
        "train loss avg={avg_loss:.4} min={min_loss:.4} max={max_loss:.4} samples={}",
        losses.len()
    );
    assert!(
        avg_loss >= min_expected,
        "avg loss too low: {avg_loss:.4} (< {min_expected:.4})"
    );
    assert!(
        avg_loss <= max_expected,
        "avg loss too high: {avg_loss:.4} (> {max_expected:.4})"
    );

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
    let min_util = settings.min_gpu_util;
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
    let max_growth = settings.max_mem_growth_mb;
    assert!(
        mem_growth <= max_growth,
        "GPU memory growth too high: {mem_growth:.1} MiB (> {max_growth:.1} MiB)"
    );
}
