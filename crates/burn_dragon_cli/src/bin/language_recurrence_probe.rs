use std::time::Instant;

use anyhow::Result;
#[cfg(not(any(feature = "cuda", feature = "language-cuda")))]
use anyhow::anyhow;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(any(feature = "cuda", feature = "language-cuda"))]
use burn_cuda::Cuda;
use burn_dragon::core::{
    BDH, BDHConfig, FusedKernelConfig, ModelState, SequenceKernelConfig, SequenceMemorySystem,
};
use burn_ndarray::NdArray;
use clap::{Parser, ValueEnum};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(name = "language_recurrence_probe")]
struct Args {
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
    #[arg(long, default_value_t = 1)]
    batch: usize,
    #[arg(long, default_value_t = 256)]
    block: usize,
    #[arg(long, default_value_t = 32)]
    chunk_tokens: usize,
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
struct LinearDenseScoreCandidateResult {
    host_loop_ms: f64,
    host_loop_tokens_per_s: f64,
    dense_score_ms: f64,
    dense_score_tokens_per_s: f64,
    context_max_diff: f64,
    context_rmse: f64,
    context_relative_max_diff: f64,
    context_reference_max_abs: f64,
    rho_max_diff: f64,
    rho_rmse: f64,
    rho_relative_max_diff: f64,
    rho_reference_max_abs: f64,
}

#[derive(Debug, Clone, Serialize)]
struct KernelProbeResult {
    sequence_kernel: SequenceKernelConfig,
    full_forward_ms: f64,
    full_forward_tokens_per_s: f64,
    stateful_full_ms: f64,
    stateful_full_tokens_per_s: f64,
    token_step_ms: f64,
    token_step_tokens_per_s: f64,
    chunked_ms: f64,
    chunked_tokens_per_s: f64,
    full_vs_stateful_max_diff: f64,
    full_vs_step_max_diff: f64,
    full_vs_chunked_max_diff: f64,
    stateful_vs_step_state_max_diff: f64,
    stateful_vs_chunked_state_max_diff: f64,
    linear_dense_score_candidate: Option<LinearDenseScoreCandidateResult>,
}

#[derive(Debug, Clone, Serialize)]
struct ProbeReport {
    backend: String,
    batch: usize,
    block: usize,
    chunk_tokens: usize,
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

fn build_config(args: &Args, kernel: SequenceKernelConfig) -> BDHConfig {
    assert!(
        args.latent_total % args.n_embd == 0,
        "latent_total must be divisible by n_embd"
    );
    BDHConfig {
        n_layer: args.n_layer,
        n_embd: args.n_embd,
        n_head: args.n_head,
        mlp_internal_dim_multiplier: args.latent_total / args.n_embd,
        vocab_size: args.vocab_size,
        dropout: 0.0,
        sequence_kernel: kernel,
        fused_kernels: FusedKernelConfig {
            enabled: false,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn sample_recurrence_query<B: BackendTrait>(
    batch: usize,
    heads: usize,
    block: usize,
    latent_per_head: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let total = batch * heads * block * latent_per_head;
    let values = (0..total)
        .map(|idx| (((idx % 97) as f32) / 97.0) + 0.01)
        .collect::<Vec<_>>();
    Tensor::<B, 4>::from_data(
        TensorData::new(values, [batch, heads, block, latent_per_head]),
        device,
    )
}

fn sample_recurrence_value<B: BackendTrait>(
    batch: usize,
    block: usize,
    n_embd: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let total = batch * block * n_embd;
    let values = (0..total)
        .map(|idx| (((idx % 53) as f32) / 53.0) + 0.02)
        .collect::<Vec<_>>();
    Tensor::<B, 4>::from_data(TensorData::new(values, [batch, 1, block, n_embd]), device)
}

fn repeat_value_heads<B: BackendTrait>(value: Tensor<B, 4>, heads: usize) -> Tensor<B, 4> {
    match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    }
}

fn linear_recurrence_host_loop<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let decay = decay.map(|tensor| tensor.reshape([1, heads, 1, 1]));
    let value = repeat_value_heads(value, heads);

    let mut rho = Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device);
    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

    for t in 0..time {
        let q_t = query.clone().slice_dim(2, t..t + 1);
        let v_t = value.clone().slice_dim(2, t..t + 1);
        let q_latent = q_t.swap_dims(2, 3);

        let context_t = (rho.clone() * q_latent.clone())
            .sum_dim(2)
            .reshape([batch, heads, 1, n_embd]);
        outputs.push(context_t);

        rho = rho + q_latent * v_t;
        if let Some(decay) = &decay {
            rho = rho * decay.clone();
        }
    }

    (Tensor::cat(outputs, 2), rho)
}

fn linear_recurrence_dense_score<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let embd = value.shape().dims::<4>()[3];
    let device = query.device();
    let value = repeat_value_heads(value, heads);

    let mut scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
    let rho = if let Some(decay) = decay {
        let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);
        let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, 1, time]);
        let diff = (pos_row.clone() - pos_col.clone())
            .tril(-1)
            .repeat_dim(1, heads);
        let decay_score = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .repeat_dim(3, time);
        scores = scores * decay_score.powf(diff);

        let final_exponents = pos_row
            .mul_scalar(-1.0)
            .add_scalar(time as f32)
            .repeat_dim(1, heads);
        let decay_final = decay
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(final_exponents);
        query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
    } else {
        query.swap_dims(2, 3).matmul(value.clone())
    };

    let context = scores.matmul(value).reshape([batch, heads, time, embd]);
    assert_eq!(rho.shape().dims::<4>(), [batch, heads, latent, embd]);
    (context, rho)
}

fn tensor_max_abs_diff<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> f64 {
    let stats = tensor_diff_stats(lhs, rhs);
    stats.max_abs_diff
}

#[derive(Debug, Clone, Copy)]
struct TensorDiffStats {
    max_abs_diff: f64,
    rmse: f64,
    relative_max_diff: f64,
    reference_max_abs: f64,
}

fn tensor_diff_stats<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> TensorDiffStats {
    let lhs_vec = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs_vec = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");

    let mut max_abs_diff = 0.0f64;
    let mut sq_error_sum = 0.0f64;
    let mut reference_max_abs = 0.0f64;

    for (lhs, rhs) in lhs_vec.iter().zip(rhs_vec.iter()) {
        let lhs = *lhs as f64;
        let rhs = *rhs as f64;
        let abs_diff = (lhs - rhs).abs();
        max_abs_diff = max_abs_diff.max(abs_diff);
        sq_error_sum += abs_diff * abs_diff;
        reference_max_abs = reference_max_abs.max(lhs.abs());
    }

    let rmse = if lhs_vec.is_empty() {
        0.0
    } else {
        (sq_error_sum / lhs_vec.len() as f64).sqrt()
    };
    let relative_max_diff = max_abs_diff / reference_max_abs.max(f64::EPSILON);

    TensorDiffStats {
        max_abs_diff,
        rmse,
        relative_max_diff,
        reference_max_abs,
    }
}

fn option_tensor_max_abs_diff<B: BackendTrait, const D: usize>(
    lhs: &Option<Tensor<B, D>>,
    rhs: &Option<Tensor<B, D>>,
) -> f64 {
    match (lhs, rhs) {
        (None, None) => 0.0,
        (Some(lhs), Some(rhs)) => tensor_max_abs_diff(lhs.clone(), rhs.clone()),
        _ => f64::INFINITY,
    }
}

fn model_state_max_abs_diff<B: BackendTrait>(lhs: &ModelState<B>, rhs: &ModelState<B>) -> f64 {
    if lhs.position != rhs.position || lhs.layers.len() != rhs.layers.len() {
        return f64::INFINITY;
    }

    let mut max_diff = 0.0f64;
    for (lhs_layer, rhs_layer) in lhs.layers.iter().zip(rhs.layers.iter()) {
        max_diff = max_diff.max(option_tensor_max_abs_diff(&lhs_layer.rho, &rhs_layer.rho));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.rho_norm,
            &rhs_layer.rho_norm,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.y_neuron_state,
            &rhs_layer.y_neuron_state,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.clocked_slow_hidden,
            &rhs_layer.clocked_slow_hidden,
        ));
        max_diff = max_diff.max(option_tensor_max_abs_diff(
            &lhs_layer.summary_memory_hidden,
            &rhs_layer.summary_memory_hidden,
        ));
    }
    max_diff
}

fn timed_ms<F, B: BackendTrait>(
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
    mut f: F,
) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        f();
        let _ = B::sync(device);
    }

    let mut total_ms = 0.0;
    for _ in 0..repetitions {
        let _ = B::sync(device);
        let started = Instant::now();
        f();
        let _ = B::sync(device);
        total_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    total_ms / repetitions.max(1) as f64
}

fn throughput(tokens: usize, ms: f64) -> f64 {
    tokens as f64 / (ms / 1_000.0).max(f64::EPSILON)
}

fn run_probe<B: BackendTrait>(args: &Args, backend_name: &str, device: &B::Device) -> Result<()> {
    <B as BackendTrait>::seed(device, 2026);
    let tokens = sample_tokens::<B>(args.batch, args.block, args.vocab_size, device);
    let total_tokens = args.batch * args.block;
    let chunk_tokens = args.chunk_tokens.max(1);

    let results = [
        SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention),
        SequenceKernelConfig::dense_score_short_context(),
        SequenceKernelConfig::reference(SequenceMemorySystem::Rwkv8StateSpace),
    ]
    .into_iter()
    .map(|kernel| {
        <B as BackendTrait>::seed(device, 2026);
        let model = BDH::<B>::new(build_config(args, kernel), device);

        let full_forward_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
            let _ = model.forward(tokens.clone());
        });

        let stateful_full_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
            let mut state = model.init_state();
            let _ = model.forward_with_state(tokens.clone(), &mut state);
        });

        let token_step_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
            let mut state = model.init_state();
            for step in 0..args.block {
                let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
                let _ = model.forward_with_state(step_tokens, &mut state);
            }
        });

        let chunked_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
            let mut state = model.init_state();
            for chunk_start in (0..args.block).step_by(chunk_tokens) {
                let chunk_end = (chunk_start + chunk_tokens).min(args.block);
                let chunk = tokens.clone().slice_dim(1, chunk_start..chunk_end);
                let _ = model.forward_with_state(chunk, &mut state);
            }
        });

        let logits_full = model.forward(tokens.clone());
        let mut stateful_state = model.init_state();
        let logits_stateful = model.forward_with_state(tokens.clone(), &mut stateful_state);

        let mut step_state = model.init_state();
        let mut step_logits = Vec::with_capacity(args.block);
        for step in 0..args.block {
            let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
            step_logits.push(model.forward_with_state(step_tokens, &mut step_state));
        }
        let logits_step = Tensor::cat(step_logits, 1);

        let mut chunked_state = model.init_state();
        let mut chunked_logits = Vec::with_capacity(args.block.div_ceil(chunk_tokens));
        for chunk_start in (0..args.block).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(args.block);
            let chunk = tokens.clone().slice_dim(1, chunk_start..chunk_end);
            chunked_logits.push(model.forward_with_state(chunk, &mut chunked_state));
        }
        let logits_chunked = Tensor::cat(chunked_logits, 1);

        let linear_dense_score_candidate = if kernel
            == SequenceKernelConfig::reference(SequenceMemorySystem::LinearAttention)
        {
            let latent_per_head = args.latent_total / args.n_head.max(1);
            let query = sample_recurrence_query::<B>(
                args.batch,
                args.n_head,
                args.block,
                latent_per_head,
                device,
            );
            let value = sample_recurrence_value::<B>(args.batch, args.block, args.n_embd, device);

            let host_loop_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
                let _ = linear_recurrence_host_loop(query.clone(), value.clone(), None);
            });
            let dense_score_ms = timed_ms::<_, B>(device, args.warmup, args.repetitions, || {
                let _ = linear_recurrence_dense_score(query.clone(), value.clone(), None);
            });

            let (context_host, rho_host) =
                linear_recurrence_host_loop(query.clone(), value.clone(), None);
            let (context_dense, rho_dense) = linear_recurrence_dense_score(query, value, None);
            let context_stats = tensor_diff_stats(context_host, context_dense);
            let rho_stats = tensor_diff_stats(rho_host, rho_dense);

            Some(LinearDenseScoreCandidateResult {
                host_loop_ms,
                host_loop_tokens_per_s: throughput(total_tokens, host_loop_ms),
                dense_score_ms,
                dense_score_tokens_per_s: throughput(total_tokens, dense_score_ms),
                context_max_diff: context_stats.max_abs_diff,
                context_rmse: context_stats.rmse,
                context_relative_max_diff: context_stats.relative_max_diff,
                context_reference_max_abs: context_stats.reference_max_abs,
                rho_max_diff: rho_stats.max_abs_diff,
                rho_rmse: rho_stats.rmse,
                rho_relative_max_diff: rho_stats.relative_max_diff,
                rho_reference_max_abs: rho_stats.reference_max_abs,
            })
        } else {
            None
        };

        KernelProbeResult {
            sequence_kernel: kernel,
            full_forward_ms,
            full_forward_tokens_per_s: throughput(total_tokens, full_forward_ms),
            stateful_full_ms,
            stateful_full_tokens_per_s: throughput(total_tokens, stateful_full_ms),
            token_step_ms,
            token_step_tokens_per_s: throughput(total_tokens, token_step_ms),
            chunked_ms,
            chunked_tokens_per_s: throughput(total_tokens, chunked_ms),
            full_vs_stateful_max_diff: tensor_max_abs_diff(logits_full.clone(), logits_stateful),
            full_vs_step_max_diff: tensor_max_abs_diff(logits_full.clone(), logits_step),
            full_vs_chunked_max_diff: tensor_max_abs_diff(logits_full, logits_chunked),
            stateful_vs_step_state_max_diff: model_state_max_abs_diff(&stateful_state, &step_state),
            stateful_vs_chunked_state_max_diff: model_state_max_abs_diff(
                &stateful_state,
                &chunked_state,
            ),
            linear_dense_score_candidate,
        }
    })
    .collect::<Vec<_>>();

    let report = ProbeReport {
        backend: backend_name.to_string(),
        batch: args.batch,
        block: args.block,
        chunk_tokens,
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
