#![cfg(feature = "integration_test")]

#[cfg(feature = "train")]
use std::collections::HashSet;
#[cfg(any(feature = "cuda", feature = "train"))]
use std::fs;
#[cfg(feature = "cuda")]
use std::path::Path;
use std::path::PathBuf;
#[cfg(feature = "cuda")]
use std::process::Command;
#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "cuda")]
use std::thread;
#[cfg(feature = "cuda")]
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
use serde::Deserialize;

use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cubecl::CubeBackend;
#[cfg(feature = "train")]
use burn_dragon_train::train::pipeline::{
    resolve_latest_run_dir_in, resolve_run_root_for_config_paths,
};
#[cfg(feature = "train")]
use burn_dragon_train::wgpu::init_runtime;
use burn_dragon_vision::load_vision_training_config;
#[cfg(feature = "cuda")]
use burn_dragon_vision::train::{gdpo_cpu_fallbacks, loss_trace_len};
use burn_dragon_vision::train::{
    gdpo_reset_cpu_fallbacks, loss_trace_reset, loss_trace_take, train_vision_backend_for_test,
    train_vision_backend_with_config_paths,
};
use burn_ndarray::NdArray;
#[cfg(feature = "train")]
use burn_wgpu::Wgpu;
#[cfg(feature = "train")]
use burn_wgpu::{CubeBackend, WgpuRuntime};
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;

#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u32>;

#[cfg(feature = "cuda")]
fn vision_saccade_tiny_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/saccade/tiny.toml")
}

#[cfg(feature = "cuda")]
fn vision_saccade_tiny_integration_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/saccade/integration.toml")
}

#[cfg(feature = "cuda")]
fn vision_croco_tiny_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/croco/tiny.toml")
}

#[cfg(feature = "cuda")]
fn vision_croco_tiny_integration_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/croco/integration.toml")
}

fn vision_identity_tiny_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/identity/tiny.toml")
}

fn vision_identity_tiny_integration_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision/identity/integration.toml")
}

#[cfg(feature = "train")]
fn moving_mnist_trm_retention_check_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("vision")
        .join("video_lejepa")
        .join("moving_mnist_trm_retention_check.toml")
}

#[cfg(feature = "train")]
fn vision_run_root(
    config: &burn_dragon_vision::VisionTrainingConfig,
    paths: &[PathBuf],
) -> PathBuf {
    resolve_run_root_for_config_paths("vision", &config.run_layout, paths)
}

#[cfg(feature = "train")]
fn current_run_dirs(root: &std::path::Path) -> HashSet<PathBuf> {
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

#[cfg(feature = "train")]
fn newest_added_run_dir(before: &HashSet<PathBuf>, root: &std::path::Path) -> PathBuf {
    let mut added: Vec<_> = current_run_dirs(root).difference(before).cloned().collect();
    if added.is_empty() {
        if let Some(latest_run_dir) = resolve_latest_run_dir_in(root) {
            return latest_run_dir;
        }
        panic!("no new run directory created under {}", root.display());
    }
    added.sort();
    added.pop().expect("new run dir")
}

#[cfg(feature = "train")]
fn parse_device_memory_log(path: &std::path::Path) -> Vec<(f64, f64)> {
    let contents = fs::read_to_string(path)
        .unwrap_or_else(|err| panic!("read {} failed: {err}", path.display()));
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
            .unwrap_or_else(|err| panic!("parse reserved from `{metric}` failed: {err}"));
        let in_use = parts
            .next()
            .unwrap_or_default()
            .trim()
            .trim_end_matches(" MiB")
            .parse::<f64>()
            .unwrap_or_else(|err| panic!("parse in-use from `{metric}` failed: {err}"));
        values.push((reserved, in_use));
    }
    assert!(
        !values.is_empty(),
        "no device memory samples in {}",
        path.display()
    );
    values
}

#[cfg(feature = "cuda")]
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
    max_gpu_mem_mb: f32,
}

#[cfg(feature = "cuda")]
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
            max_gpu_mem_mb: 0.0,
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default, Deserialize)]
struct TestSettings {
    #[serde(default)]
    test: TestSection,
}

#[cfg(feature = "cuda")]
#[derive(Debug, Default, Deserialize)]
struct TestSection {
    #[serde(default)]
    integration: IntegrationSettings,
}

#[cfg(feature = "cuda")]
fn load_integration_settings(path: &Path) -> IntegrationSettings {
    let contents = fs::read_to_string(path).expect("read integration config");
    let settings: TestSettings = toml::from_str(&contents).expect("parse integration config");
    settings.test.integration
}

#[test]
fn cpu_vision_identity_tiny_training_loss_decreases() {
    let config_path = vision_identity_tiny_path();
    let integration_path = vision_identity_tiny_integration_path();
    let config = load_vision_training_config(&[config_path, integration_path])
        .expect("load vision_identity_tiny");

    gdpo_reset_cpu_fallbacks();
    loss_trace_reset();

    let result = train_vision_backend_for_test::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {});
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let losses = loss_trace_take();
    assert!(
        !losses.is_empty(),
        "no training loss samples captured; enable training.trace_train_loss in the integration config"
    );
    let first = losses.first().copied().unwrap_or(f32::INFINITY);
    let min_loss = losses
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, value| acc.min(value));
    assert!(first.is_finite(), "initial loss not finite: {first}");
    assert!(min_loss.is_finite(), "min loss not finite: {min_loss}");
    assert!(
        min_loss < first,
        "expected loss to decrease at least once (initial={first}, min={min_loss})"
    );
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn wgpu_video_trm_artifact_validation_memory_stays_bounded() {
    let config_path = moving_mnist_trm_retention_check_path();
    let config = load_vision_training_config(&[config_path]).expect("load retention config");
    let run_root = vision_run_root(&config, &[moving_mnist_trm_retention_check_path()]);
    fs::create_dir_all(&run_root).expect("create run root");
    let before = current_run_dirs(&run_root);

    type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
    let result = train_vision_backend_with_config_paths::<Autodiff<WgpuNoFusion>, _>(
        &config,
        &[moving_mnist_trm_retention_check_path()],
        "wgpu-nofusion",
        |device| init_runtime(device, &config.wgpu),
    );
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let run_dir = newest_added_run_dir(&before, &run_root);
    let valid_epoch1 = parse_device_memory_log(&run_dir.join("valid/epoch-1/device_memory_mb.log"));
    let valid_epoch2 = parse_device_memory_log(&run_dir.join("valid/epoch-2/device_memory_mb.log"));

    let epoch1_reserved_max = valid_epoch1
        .iter()
        .map(|(reserved, _)| *reserved)
        .fold(0.0, f64::max);
    let epoch2_reserved_max = valid_epoch2
        .iter()
        .map(|(reserved, _)| *reserved)
        .fold(0.0, f64::max);
    let epoch1_in_use_max = valid_epoch1
        .iter()
        .map(|(_, in_use)| *in_use)
        .fold(0.0, f64::max);
    let epoch2_in_use_max = valid_epoch2
        .iter()
        .map(|(_, in_use)| *in_use)
        .fold(0.0, f64::max);

    let reserved_growth = epoch2_reserved_max - epoch1_reserved_max;
    let in_use_growth = epoch2_in_use_max - epoch1_in_use_max;
    assert!(
        reserved_growth <= 1024.0,
        "reserved device memory grew by {reserved_growth:.1} MiB across validation epochs; run_dir={}",
        run_dir.display()
    );
    assert!(
        in_use_growth <= 256.0,
        "in-use device memory grew by {in_use_growth:.1} MiB across validation epochs; run_dir={}",
        run_dir.display()
    );

    let artifact_count = fs::read_dir(run_dir.join("artifacts"))
        .expect("artifact dir")
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().and_then(|ext| ext.to_str()) == Some("mp4"))
        .count();
    assert!(
        artifact_count >= 16,
        "expected at least 16 artifact mp4s, found {artifact_count} in {}",
        run_dir.display()
    );
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
fn run_cuda_training_gpu_utilization(config_path: PathBuf, integration_path: PathBuf) {
    let settings = load_integration_settings(&integration_path);
    if settings.skip_gpu_util {
        eprintln!("GPU utilization check disabled in config.");
        return;
    }

    if query_nvidia_smi().is_none() {
        panic!("nvidia-smi not available; disable test.integration.skip_gpu_util to skip");
    }

    let config =
        load_vision_training_config(&[config_path, integration_path]).expect("load test config");

    gdpo_reset_cpu_fallbacks();
    loss_trace_reset();

    let done = Arc::new(AtomicBool::new(false));
    let done_clone = Arc::clone(&done);
    let handle = thread::spawn(move || {
        let result = train_vision_backend_for_test::<Autodiff<CudaCubeBackend>, _>(
            &config,
            "cuda-cubecl",
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
    eprintln!("GPU memory growth={mem_growth:.1} MiB (min={min_mem:.1}, max={max_mem:.1})");
    let max_growth = settings.max_mem_growth_mb;
    assert!(
        mem_growth <= max_growth,
        "GPU memory growth too high: {mem_growth:.1} MiB (> {max_growth:.1} MiB)"
    );
    let max_mem_cap = settings.max_gpu_mem_mb;
    if max_mem_cap > 0.0 {
        assert!(
            max_mem <= max_mem_cap,
            "GPU memory exceeded cap: {max_mem:.1} MiB (> {max_mem_cap:.1} MiB)"
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_vision_saccade_tiny_training_gpu_utilization() {
    run_cuda_training_gpu_utilization(
        vision_saccade_tiny_path(),
        vision_saccade_tiny_integration_path(),
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_vision_croco_tiny_training_gpu_utilization() {
    run_cuda_training_gpu_utilization(
        vision_croco_tiny_path(),
        vision_croco_tiny_integration_path(),
    );
}
