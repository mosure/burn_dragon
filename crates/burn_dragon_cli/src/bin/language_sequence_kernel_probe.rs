use std::time::Instant;

use anyhow::Result;
#[cfg(not(any(feature = "cuda", feature = "language-cuda")))]
use anyhow::anyhow;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(any(feature = "cuda", feature = "language-cuda"))]
use burn_cuda::Cuda;
use burn_dragon::core::{
    BDH, BDHConfig, FusedKernelConfig, SequenceKernelConfig, SequenceKernelFamily,
    SequenceKernelKind, SequenceTrainingExecutor,
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
    legacy_kind: Option<SequenceKernelKind>,
    family: SequenceKernelFamily,
    executor: SequenceTrainingExecutor,
    full_forward_ms: f64,
    full_forward_tokens_per_s: f64,
    recurrent_ms: f64,
    recurrent_tokens_per_s: f64,
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

fn build_config(args: &Args, kernel: SequenceKernelKind) -> BDHConfig {
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

fn probe_cases() -> [SequenceKernelConfig; 4] {
    [
        SequenceKernelKind::BdhLinearAttention.resolved_config(),
        SequenceKernelKind::BdhLinearDenseScoreExperimental.resolved_config(),
        SequenceKernelKind::Rwkv8StateSpaceExperimental.resolved_config(),
        SequenceKernelKind::MambaSelectiveSsmExperimental.resolved_config(),
    ]
}

fn timed_full_forward<B: BackendTrait>(
    model: &BDH<B>,
    tokens: &Tensor<B, 2, Int>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
) -> (f64, f64, f64) {
    for _ in 0..warmup {
        let _ = model.forward(tokens.clone());
        let _ = B::sync(device);
    }

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

    let avg_ms = total_ms / repetitions.max(1) as f64;
    let tokens_per_s = (tokens.shape().dims::<2>()[0] * tokens.shape().dims::<2>()[1]) as f64
        / (avg_ms / 1_000.0).max(f64::EPSILON);
    (avg_ms, tokens_per_s, checksum)
}

fn timed_recurrent<B: BackendTrait>(
    model: &BDH<B>,
    tokens: &Tensor<B, 2, Int>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
) -> (f64, f64) {
    let block = tokens.shape().dims::<2>()[1];
    for _ in 0..warmup {
        let mut state = model.init_state();
        for step in 0..block {
            let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
            let _ = model.forward_with_state(step_tokens, &mut state);
        }
        let _ = B::sync(device);
    }

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

    let avg_ms = total_ms / repetitions.max(1) as f64;
    let tokens_per_s =
        (tokens.shape().dims::<2>()[0] * block) as f64 / (avg_ms / 1_000.0).max(f64::EPSILON);
    (avg_ms, tokens_per_s)
}

fn run_probe<B: BackendTrait>(args: &Args, backend_name: &str, device: &B::Device) -> Result<()> {
    <B as BackendTrait>::seed(device, 2026);
    let tokens = sample_tokens::<B>(args.batch, args.block, args.vocab_size, device);
    let results = probe_cases()
        .into_iter()
        .map(|kernel_config| {
            let kernel = kernel_config
                .legacy_kind()
                .expect("probe currently only supports legacy-backed families");
            <B as BackendTrait>::seed(device, 2026);
            let model = BDH::<B>::new(build_config(args, kernel), device);
            let (full_forward_ms, full_forward_tokens_per_s, checksum) =
                timed_full_forward(&model, &tokens, device, args.warmup, args.repetitions);
            let (recurrent_ms, recurrent_tokens_per_s) =
                timed_recurrent(&model, &tokens, device, args.warmup, args.repetitions);
            KernelProbeResult {
                legacy_kind: Some(kernel),
                family: kernel_config.family,
                executor: kernel_config.executor,
                full_forward_ms,
                full_forward_tokens_per_s,
                recurrent_ms,
                recurrent_tokens_per_s,
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
