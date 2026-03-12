use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon::core::{BDH, BDHConfig, FusedKernelConfig};
use burn_dragon::language::loss::language_model_loss;
use burn_dragon_wgpu::api::recurrent::{recurrent_profile_reset, recurrent_profile_snapshot};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;
use serde::Serialize;

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InnerBackend>;
type Device = <TrainBackend as BackendTrait>::Device;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
    #[arg(long)]
    markdown_path: Option<PathBuf>,
    #[arg(long)]
    json_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    batch: usize,
    sequence_len: usize,
}

#[derive(Clone, Copy, Serialize)]
struct ErrorMetrics {
    max_abs: f32,
    mean_abs: f32,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    problem_shape: String,
    rollout_fast_steps: usize,
    warmup: usize,
    iterations: usize,
    baseline_forward_backward_ms: f64,
    fused_forward_backward_ms: f64,
    baseline_tokens_per_sec: f64,
    fused_tokens_per_sec: f64,
    speedup_x: f64,
    loss_abs_diff: f32,
    logits_error: ErrorMetrics,
    baseline_recurrent_calls: f64,
    fused_recurrent_calls: f64,
    baseline_recurrent_launches: f64,
    fused_recurrent_launches: f64,
    baseline_dispatch_ns: f64,
    fused_dispatch_ns: f64,
    baseline_transient_allocations: f64,
    fused_transient_allocations: f64,
    baseline_metadata_upload_bytes: f64,
    fused_metadata_upload_bytes: f64,
}

#[derive(Clone, Serialize)]
struct Report {
    benchmark: &'static str,
    adapter: String,
    warmup: usize,
    iterations: usize,
    cases: Vec<CaseResult>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "tiny",
        n_layer: 2,
        n_embd: 32,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 128,
        batch: 4,
        sequence_len: 32,
    },
    BenchCase {
        name: "wide",
        n_layer: 3,
        n_embd: 64,
        n_head: 8,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 256,
        batch: 4,
        sequence_len: 32,
    },
    BenchCase {
        name: "deep",
        n_layer: 4,
        n_embd: 96,
        n_head: 8,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 384,
        batch: 2,
        sequence_len: 48,
    },
];

fn main() {
    let args = Args::parse();
    let device = Device::default();
    init_runtime(&device);

    let report = Report {
        benchmark: "burn_dragon core forward+backward wgpu fused rollout benchmark",
        adapter: adapter_info(),
        warmup: args.warmup,
        iterations: args.iterations,
        cases: run_all_cases(&device, &args),
    };

    let markdown = format_markdown(&report);
    let json = serde_json::to_string_pretty(&report).expect("serialize core fb report");

    println!("{markdown}");

    if let Some(path) = args.markdown_path.as_ref() {
        write_text_artifact(path, &markdown, "markdown artifact");
    }
    if let Some(path) = args.json_path.as_ref() {
        write_text_artifact(path, &json, "json artifact");
    }
}

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn adapter_info() -> String {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu adapter");
    let info = adapter.get_info();
    format!("{} ({:?})", info.name, info.device_type)
}

fn build_case_config(
    case: BenchCase,
    rollout_fast_steps: usize,
    wgpu_recurrent_kernel: bool,
    wgpu_rollout_fused: bool,
) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: case.n_layer,
        n_embd: case.n_embd,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: case.mlp_internal_dim_multiplier,
        vocab_size: case.vocab_size,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_recurrent_kernel,
            wgpu_rollout_fused,
            ..Default::default()
        },
        ..Default::default()
    };
    config.fused_kernels.set_block_sizes(8, 8);
    config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    config
}

fn sample_batch(
    case: BenchCase,
    device: &Device,
) -> (Tensor<TrainBackend, 2, Int>, Tensor<TrainBackend, 2, Int>) {
    let tokens = case.batch * case.sequence_len;
    let inputs: Vec<i64> = (0..tokens)
        .map(|idx| (idx % case.vocab_size) as i64)
        .collect();
    let targets: Vec<i64> = (0..tokens)
        .map(|idx| ((idx + 1) % case.vocab_size) as i64)
        .collect();
    (
        Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(inputs, [case.batch, case.sequence_len]),
            device,
        ),
        Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(targets, [case.batch, case.sequence_len]),
            device,
        ),
    )
}

fn run_all_cases(device: &Device, args: &Args) -> Vec<CaseResult> {
    let mut results = Vec::new();
    for case in CASES.iter().copied() {
        for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
            results.push(run_case(case, rollout_fast_steps, device, args));
        }
    }
    results
}

fn run_case(
    case: BenchCase,
    rollout_fast_steps: usize,
    device: &Device,
    args: &Args,
) -> CaseResult {
    let (inputs, targets) = sample_batch(case, device);

    <TrainBackend as BackendTrait>::seed(device, 2_026 + rollout_fast_steps as u64);
    let baseline = BDH::<TrainBackend>::new(
        build_case_config(case, rollout_fast_steps, false, false),
        device,
    );
    <TrainBackend as BackendTrait>::seed(device, 2_026 + rollout_fast_steps as u64);
    let fused = BDH::<TrainBackend>::new(
        build_case_config(case, rollout_fast_steps, true, true),
        device,
    );

    let (loss_abs_diff, logits_error) =
        parity_snapshot(&baseline, &fused, inputs.clone(), targets.clone(), device);

    for _ in 0..args.warmup {
        let _ = run_forward_backward(&baseline, inputs.clone(), targets.clone(), device);
        let _ = run_forward_backward(&fused, inputs.clone(), targets.clone(), device);
    }

    let mut baseline_ns = Vec::with_capacity(args.iterations);
    let mut baseline_calls = Vec::with_capacity(args.iterations);
    let mut baseline_launches = Vec::with_capacity(args.iterations);
    let mut baseline_dispatch = Vec::with_capacity(args.iterations);
    let mut baseline_allocs = Vec::with_capacity(args.iterations);
    let mut baseline_metadata = Vec::with_capacity(args.iterations);
    let mut fused_ns = Vec::with_capacity(args.iterations);
    let mut fused_calls = Vec::with_capacity(args.iterations);
    let mut fused_launches = Vec::with_capacity(args.iterations);
    let mut fused_dispatch = Vec::with_capacity(args.iterations);
    let mut fused_allocs = Vec::with_capacity(args.iterations);
    let mut fused_metadata = Vec::with_capacity(args.iterations);

    for _ in 0..args.iterations {
        let baseline_metrics =
            run_forward_backward(&baseline, inputs.clone(), targets.clone(), device);
        baseline_ns.push(baseline_metrics.elapsed_ns);
        baseline_calls.push(baseline_metrics.recurrent_calls as f64);
        baseline_launches.push(baseline_metrics.recurrent_launches as f64);
        baseline_dispatch.push(baseline_metrics.dispatch_ns as f64);
        baseline_allocs.push(baseline_metrics.transient_allocations as f64);
        baseline_metadata.push(baseline_metrics.metadata_upload_bytes as f64);

        let fused_metrics = run_forward_backward(&fused, inputs.clone(), targets.clone(), device);
        fused_ns.push(fused_metrics.elapsed_ns);
        fused_calls.push(fused_metrics.recurrent_calls as f64);
        fused_launches.push(fused_metrics.recurrent_launches as f64);
        fused_dispatch.push(fused_metrics.dispatch_ns as f64);
        fused_allocs.push(fused_metrics.transient_allocations as f64);
        fused_metadata.push(fused_metrics.metadata_upload_bytes as f64);
    }

    let baseline_avg_ns = mean_u128(&baseline_ns);
    let fused_avg_ns = mean_u128(&fused_ns);
    let tokens_per_step = (case.batch * case.sequence_len) as f64;
    let baseline_tokens_per_sec = tokens_per_step / (baseline_avg_ns / 1e9);
    let fused_tokens_per_sec = tokens_per_step / (fused_avg_ns / 1e9);

    CaseResult {
        case,
        problem_shape: format!(
            "b{} t{} layer{} embd{} head{} vocab{} fast{}",
            case.batch,
            case.sequence_len,
            case.n_layer,
            case.n_embd,
            case.n_head,
            case.vocab_size,
            rollout_fast_steps
        ),
        rollout_fast_steps,
        warmup: args.warmup,
        iterations: args.iterations,
        baseline_forward_backward_ms: baseline_avg_ns / 1e6,
        fused_forward_backward_ms: fused_avg_ns / 1e6,
        baseline_tokens_per_sec,
        fused_tokens_per_sec,
        speedup_x: safe_ratio(baseline_avg_ns, fused_avg_ns),
        loss_abs_diff,
        logits_error,
        baseline_recurrent_calls: mean_f64(&baseline_calls),
        fused_recurrent_calls: mean_f64(&fused_calls),
        baseline_recurrent_launches: mean_f64(&baseline_launches),
        fused_recurrent_launches: mean_f64(&fused_launches),
        baseline_dispatch_ns: mean_f64(&baseline_dispatch),
        fused_dispatch_ns: mean_f64(&fused_dispatch),
        baseline_transient_allocations: mean_f64(&baseline_allocs),
        fused_transient_allocations: mean_f64(&fused_allocs),
        baseline_metadata_upload_bytes: mean_f64(&baseline_metadata),
        fused_metadata_upload_bytes: mean_f64(&fused_metadata),
    }
}

#[derive(Clone, Copy)]
struct StepMetrics {
    elapsed_ns: u128,
    recurrent_calls: u64,
    recurrent_launches: u64,
    dispatch_ns: u128,
    transient_allocations: u64,
    metadata_upload_bytes: u64,
}

fn run_forward_backward(
    model: &BDH<TrainBackend>,
    inputs: Tensor<TrainBackend, 2, Int>,
    targets: Tensor<TrainBackend, 2, Int>,
    device: &Device,
) -> StepMetrics {
    let _ = <TrainBackend as BackendTrait>::sync(device);
    recurrent_profile_reset();
    let started = Instant::now();
    let logits = model.forward(inputs);
    let loss = language_model_loss::<TrainBackend>(logits, targets);
    let _ = loss.backward();
    let _ = <TrainBackend as BackendTrait>::sync(device);
    let profile = recurrent_profile_snapshot();

    StepMetrics {
        elapsed_ns: started.elapsed().as_nanos(),
        recurrent_calls: profile.calls,
        recurrent_launches: profile.launches,
        dispatch_ns: profile.dispatch_ns,
        transient_allocations: profile.transient_allocations,
        metadata_upload_bytes: profile.metadata_upload_bytes,
    }
}

fn parity_snapshot(
    baseline: &BDH<TrainBackend>,
    fused: &BDH<TrainBackend>,
    inputs: Tensor<TrainBackend, 2, Int>,
    targets: Tensor<TrainBackend, 2, Int>,
    device: &Device,
) -> (f32, ErrorMetrics) {
    let baseline_logits = baseline.forward(inputs.clone());
    let fused_logits = fused.forward(inputs.clone());
    let baseline_loss =
        language_model_loss::<TrainBackend>(baseline_logits.clone(), targets.clone());
    let fused_loss = language_model_loss::<TrainBackend>(fused_logits.clone(), targets);

    let baseline_loss_value = scalar_to_f32(baseline_loss);
    let fused_loss_value = scalar_to_f32(fused_loss);
    let _ = <TrainBackend as BackendTrait>::sync(device);

    let baseline_logits = baseline_logits.inner();
    let fused_logits = fused_logits.inner();
    (
        (baseline_loss_value - fused_loss_value).abs(),
        error_metrics(baseline_logits, fused_logits),
    )
}

fn scalar_to_f32(tensor: Tensor<TrainBackend, 1>) -> f32 {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scalar vec")[0]
}

fn error_metrics<const D: usize>(
    lhs: Tensor<InnerBackend, D>,
    rhs: Tensor<InnerBackend, D>,
) -> ErrorMetrics {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");

    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    let mean_abs = if lhs.is_empty() {
        0.0
    } else {
        sum_abs / lhs.len() as f32
    };

    ErrorMetrics { max_abs, mean_abs }
}

fn mean_u128(values: &[u128]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<u128>() as f64 / values.len() as f64
    }
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn safe_ratio(numer: f64, denom: f64) -> f64 {
    if denom == 0.0 { 0.0 } else { numer / denom }
}

fn format_markdown(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Core Forward+Backward Benchmark");
    let _ = writeln!(out);
    let _ = writeln!(out, "- Benchmark: {}", report.benchmark);
    let _ = writeln!(out, "- Adapter: {}", report.adapter);
    let _ = writeln!(out, "- Warmup: {}", report.warmup);
    let _ = writeln!(out, "- Iterations: {}", report.iterations);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| case | fast_steps | baseline ms | fused ms | speedup x | baseline tok/s | fused tok/s | loss abs diff | logits max abs | logits mean abs | fused recurrent calls | fused launches | fused dispatch ms | fused meta bytes |"
    );
    let _ = writeln!(
        out,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for row in &report.cases {
        let _ = writeln!(
            out,
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.6} | {:.6} | {:.6} | {:.1} | {:.1} | {:.3} | {:.1} |",
            row.case.name,
            row.rollout_fast_steps,
            row.baseline_forward_backward_ms,
            row.fused_forward_backward_ms,
            row.speedup_x,
            row.baseline_tokens_per_sec,
            row.fused_tokens_per_sec,
            row.loss_abs_diff,
            row.logits_error.max_abs,
            row.logits_error.mean_abs,
            row.fused_recurrent_calls,
            row.fused_recurrent_launches,
            row.fused_dispatch_ns / 1e6,
            row.fused_metadata_upload_bytes,
        );
    }
    out
}

fn write_text_artifact(path: &Path, content: &str, label: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create artifact directory");
    }
    fs::write(path, content).expect("write artifact");
    eprintln!("wrote {label} to {}", path.display());
}
