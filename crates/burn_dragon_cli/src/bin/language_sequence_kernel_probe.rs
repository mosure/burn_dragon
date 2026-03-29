use std::process::Command;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Instant;

use anyhow::Result;
#[cfg(not(any(feature = "cuda", feature = "language-cuda")))]
use anyhow::anyhow;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(any(feature = "cuda", feature = "language-cuda"))]
use burn_cuda::Cuda;
use burn_dragon::core::{
    BDH, BDHConfig, FusedKernelConfig, SequenceKernelConfig, SequenceMemorySystem,
    SequenceTrainingExecutor,
};
use burn_dragon_kernel::kernels::sequence::{
    mamba as mamba_kernel, mamba2 as mamba2_kernel, rwkv8 as rwkv8_kernel,
};
use burn_ndarray::NdArray;
use clap::{Parser, ValueEnum};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "language_sequence_kernel_probe")]
struct Args {
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 256)]
    block: usize,
    #[arg(long, default_value_t = 8)]
    n_layer: usize,
    #[arg(long, default_value_t = 256)]
    n_embd: usize,
    #[arg(long, default_value_t = 1)]
    n_head: usize,
    #[arg(long, default_value_t = 32768)]
    latent_total: usize,
    #[arg(long, default_value_t = 50257)]
    vocab_size: usize,
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    repetitions: usize,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Ndarray,
    Cuda,
}

#[derive(Debug, Clone, Serialize)]
struct KernelProbeResult {
    sequence_kernel: SequenceKernelConfig,
    implementation_status: String,
    algorithmic_executor_shortcut: bool,
    forward_kernel_available: bool,
    backward_kernel_available: bool,
    upstream_anchor: Option<String>,
    full_forward_ms: f64,
    full_forward_tokens_per_s: f64,
    full_forward_gpu: SampledGpuTelemetry,
    recurrent_ms: f64,
    recurrent_tokens_per_s: f64,
    recurrent_gpu: SampledGpuTelemetry,
    checksum: f64,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeReport {
    backend: String,
    batch: usize,
    block: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    vocab_size: usize,
    warmup: usize,
    repetitions: usize,
    results: Vec<KernelProbeResult>,
}

#[derive(Debug, Clone, Serialize, Default)]
struct SampledGpuTelemetry {
    samples: usize,
    mean_util_pct: Option<f32>,
    max_util_pct: Option<f32>,
    mean_power_w: Option<f32>,
    max_power_w: Option<f32>,
    peak_memory_mib: Option<f32>,
}

#[derive(Debug, Clone, Copy)]
struct GpuSample {
    util_pct: f32,
    power_w: f32,
    memory_mib: f32,
}

fn sample_tokens<B: BackendTrait>(
    batch: usize,
    block: usize,
    vocab_size: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let token_count = batch * block;
    let tokens: Vec<i64> = (0..token_count)
        .map(|idx| (idx % vocab_size.max(2)) as i64)
        .collect();
    Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [batch, block]), device)
}

fn build_config(args: &Args, kernel: SequenceKernelConfig) -> BDHConfig {
    assert!(
        args.latent_total % args.n_embd == 0,
        "latent_total must be divisible by n_embd"
    );
    let mut config = BDHConfig {
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        mlp_internal_dim_multiplier: args.latent_total / args.n_embd,
        vocab_size: args.vocab_size,
        dropout: 0.0,
        sequence_kernel: kernel,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            ..Default::default()
        },
        ..Default::default()
    };
    config
        .fused_kernels
        .set_block_sizes(8.min(args.n_embd), 8.min(args.block.max(1)));
    config
}

fn probe_cases() -> [SequenceKernelConfig; 5] {
    [
        SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
        SequenceKernelConfig::dense_score_short_context(),
        SequenceKernelConfig::reference(SequenceMemorySystem::Rwkv8StateSpace),
        SequenceKernelConfig::reference(SequenceMemorySystem::Mamba1SelectiveScan),
        SequenceKernelConfig::reference(SequenceMemorySystem::Mamba2StateSpaceDuality),
    ]
}

fn implementation_metadata(
    kernel_config: SequenceKernelConfig,
) -> (&'static str, bool, bool, bool, Option<String>) {
    match (kernel_config.memory_system, kernel_config.executor) {
        (
            SequenceMemorySystem::LinearAttention,
            SequenceTrainingExecutor::DenseScoreShortContext,
        ) => ("algorithmic_executor_shortcut", true, false, false, None),
        (SequenceMemorySystem::LinearAttention, SequenceTrainingExecutor::Reference) => {
            ("reference_only", false, false, false, None)
        }
        (SequenceMemorySystem::Rwkv8StateSpace, SequenceTrainingExecutor::Reference) => (
            rwkv8_kernel::STATUS,
            false,
            rwkv8_kernel::FORWARD_ACCELERATION_AVAILABLE,
            rwkv8_kernel::BACKWARD_ACCELERATION_AVAILABLE,
            Some(format!(
                "{} :: {} ({})",
                rwkv8_kernel::UPSTREAM_MODEL_REPO,
                rwkv8_kernel::UPSTREAM_KERNEL_REPO,
                rwkv8_kernel::UPSTREAM_TARGET_KIND
            )),
        ),
        (SequenceMemorySystem::Mamba1SelectiveScan, SequenceTrainingExecutor::Reference) => (
            mamba_kernel::STATUS,
            false,
            mamba_kernel::FORWARD_ACCELERATION_AVAILABLE,
            mamba_kernel::BACKWARD_ACCELERATION_AVAILABLE,
            Some(format!(
                "{}@{} ({})",
                mamba_kernel::UPSTREAM_REPO,
                mamba_kernel::UPSTREAM_COMMIT,
                mamba_kernel::UPSTREAM_TARGET_KIND
            )),
        ),
        (SequenceMemorySystem::Mamba2StateSpaceDuality, SequenceTrainingExecutor::Reference) => (
            mamba2_kernel::STATUS,
            false,
            mamba2_kernel::FORWARD_ACCELERATION_AVAILABLE,
            mamba2_kernel::BACKWARD_ACCELERATION_AVAILABLE,
            Some(format!(
                "{} ({})",
                mamba2_kernel::UPSTREAM_REPO,
                mamba2_kernel::UPSTREAM_TARGET_KIND
            )),
        ),
        (memory_system, executor) => (
            "unclassified",
            false,
            false,
            false,
            Some(format!(
                "memory_system={memory_system:?}, executor={executor:?}"
            )),
        ),
    }
}

fn query_gpu_sample() -> Option<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut max_util = None;
    let mut max_power = None;
    let mut max_memory = None;
    for line in stdout.lines() {
        let mut fields = line
            .split(',')
            .map(|field| field.trim().parse::<f32>().ok());
        let util = fields.next().flatten()?;
        let power = fields.next().flatten()?;
        let memory = fields.next().flatten()?;
        max_util = Some(max_util.map_or(util, |current: f32| current.max(util)));
        max_power = Some(max_power.map_or(power, |current: f32| current.max(power)));
        max_memory = Some(max_memory.map_or(memory, |current: f32| current.max(memory)));
    }

    Some(GpuSample {
        util_pct: max_util?,
        power_w: max_power?,
        memory_mib: max_memory?,
    })
}

fn summarize_gpu_samples(samples: &[GpuSample]) -> SampledGpuTelemetry {
    if samples.is_empty() {
        return SampledGpuTelemetry::default();
    }
    let count = samples.len() as f32;
    SampledGpuTelemetry {
        samples: samples.len(),
        mean_util_pct: Some(samples.iter().map(|sample| sample.util_pct).sum::<f32>() / count),
        max_util_pct: Some(
            samples
                .iter()
                .map(|sample| sample.util_pct)
                .fold(0.0_f32, f32::max),
        ),
        mean_power_w: Some(samples.iter().map(|sample| sample.power_w).sum::<f32>() / count),
        max_power_w: Some(
            samples
                .iter()
                .map(|sample| sample.power_w)
                .fold(0.0_f32, f32::max),
        ),
        peak_memory_mib: Some(
            samples
                .iter()
                .map(|sample| sample.memory_mib)
                .fold(0.0_f32, f32::max),
        ),
    }
}

fn run_with_gpu_sampling<F, T>(enabled: bool, run: F) -> (T, SampledGpuTelemetry)
where
    F: FnOnce() -> T,
{
    if !enabled {
        return (run(), SampledGpuTelemetry::default());
    }

    let stop = Arc::new(AtomicBool::new(false));
    let samples = Arc::new(Mutex::new(Vec::<GpuSample>::new()));
    let stop_for_thread = stop.clone();
    let samples_for_thread = samples.clone();
    let handle = thread::spawn(move || {
        while !stop_for_thread.load(Ordering::Relaxed) {
            if let Some(sample) = query_gpu_sample()
                && let Ok(mut shared) = samples_for_thread.lock()
            {
                shared.push(sample);
            }
            thread::sleep(std::time::Duration::from_millis(100));
        }
        if let Some(sample) = query_gpu_sample()
            && let Ok(mut shared) = samples_for_thread.lock()
        {
            shared.push(sample);
        }
    });

    let result = run();
    stop.store(true, Ordering::Relaxed);
    let _ = handle.join();
    let telemetry = samples
        .lock()
        .map(|shared| summarize_gpu_samples(&shared))
        .unwrap_or_default();
    (result, telemetry)
}

fn timed_full_forward<B: BackendTrait>(
    model: &BDH<B>,
    tokens: &Tensor<B, 2, Int>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
    sample_gpu: bool,
) -> (f64, f64, f64, SampledGpuTelemetry) {
    for _ in 0..warmup {
        let _ = model.forward(tokens.clone());
        let _ = B::sync(device);
    }

    let ((total_ms, checksum), telemetry) = run_with_gpu_sampling(sample_gpu, || {
        let mut total_ms = 0.0;
        let mut checksum = 0.0;
        for _ in 0..repetitions {
            let _ = B::sync(device);
            let started = Instant::now();
            let logits = model.forward(tokens.clone());
            let _ = B::sync(device);
            total_ms += started.elapsed().as_secs_f64() * 1_000.0;
            checksum = logits
                .sum()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("full forward checksum")[0] as f64;
        }
        (total_ms, checksum)
    });

    let avg_ms = total_ms / repetitions.max(1) as f64;
    let tokens_per_s = (tokens.shape().dims::<2>()[0] * tokens.shape().dims::<2>()[1]) as f64
        / (avg_ms / 1_000.0).max(f64::EPSILON);
    (avg_ms, tokens_per_s, checksum, telemetry)
}

fn timed_recurrent<B: BackendTrait>(
    model: &BDH<B>,
    tokens: &Tensor<B, 2, Int>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
    sample_gpu: bool,
) -> (f64, f64, SampledGpuTelemetry) {
    let block = tokens.shape().dims::<2>()[1];
    for _ in 0..warmup {
        let mut state = model.init_state();
        for step in 0..block {
            let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
            let _ = model.forward_with_state(step_tokens, &mut state);
        }
        let _ = B::sync(device);
    }

    let (total_ms, telemetry) = run_with_gpu_sampling(sample_gpu, || {
        let mut total_ms = 0.0;
        for _ in 0..repetitions {
            let mut state = model.init_state();
            let _ = B::sync(device);
            let started = Instant::now();
            for step in 0..block {
                let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
                let _ = model.forward_with_state(step_tokens, &mut state);
            }
            let _ = B::sync(device);
            total_ms += started.elapsed().as_secs_f64() * 1_000.0;
        }
        total_ms
    });

    let avg_ms = total_ms / repetitions.max(1) as f64;
    let tokens_per_s =
        (tokens.shape().dims::<2>()[0] * block) as f64 / (avg_ms / 1_000.0).max(f64::EPSILON);
    (avg_ms, tokens_per_s, telemetry)
}

fn run_probe<B: BackendTrait>(args: &Args, backend_name: &str, device: &B::Device) -> Result<()> {
    <B as BackendTrait>::seed(device, 2026);
    let tokens = sample_tokens::<B>(args.batch, args.block, args.vocab_size, device);
    let sample_gpu = backend_name == "cuda";
    let results = probe_cases()
        .into_iter()
        .map(|kernel_config| {
            let (
                implementation_status,
                algorithmic_executor_shortcut,
                forward_kernel_available,
                backward_kernel_available,
                upstream_anchor,
            ) = implementation_metadata(kernel_config);
            <B as BackendTrait>::seed(device, 2026);
            let model = BDH::<B>::new(build_config(args, kernel_config), device);
            let (full_forward_ms, full_forward_tokens_per_s, checksum, full_forward_gpu) =
                timed_full_forward(
                    &model,
                    &tokens,
                    device,
                    args.warmup,
                    args.repetitions,
                    sample_gpu,
                );
            let (recurrent_ms, recurrent_tokens_per_s, recurrent_gpu) = timed_recurrent(
                &model,
                &tokens,
                device,
                args.warmup,
                args.repetitions,
                sample_gpu,
            );
            KernelProbeResult {
                sequence_kernel: kernel_config,
                implementation_status: implementation_status.to_string(),
                algorithmic_executor_shortcut,
                forward_kernel_available,
                backward_kernel_available,
                upstream_anchor,
                full_forward_ms,
                full_forward_tokens_per_s,
                full_forward_gpu,
                recurrent_ms,
                recurrent_tokens_per_s,
                recurrent_gpu,
                checksum,
            }
        })
        .collect::<Vec<_>>();

    let report = ProbeReport {
        backend: backend_name.to_string(),
        batch: args.batch,
        block: args.block,
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        latent_total: args.latent_total,
        vocab_size: args.vocab_size,
        warmup: args.warmup,
        repetitions: args.repetitions,
        results,
    };

    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    match args.backend {
        BackendArg::Ndarray => {
            let device = <NdArray<f32> as BackendTrait>::Device::default();
            run_probe::<NdArray<f32>>(&args, "ndarray", &device)
        }
        BackendArg::Cuda => {
            #[cfg(any(feature = "cuda", feature = "language-cuda"))]
            {
                let device = <Cuda<f32> as BackendTrait>::Device::default();
                run_probe::<Cuda<f32>>(&args, "cuda", &device)
            }
            #[cfg(not(any(feature = "cuda", feature = "language-cuda")))]
            {
                Err(anyhow!(
                    "cuda backend selected but this binary was built without a CUDA-enabled feature"
                ))
            }
        }
    }
}
