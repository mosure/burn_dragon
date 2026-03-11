use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon::language::loss::language_model_loss;
use burn_dragon::{BDH, BDHConfig, FusedKernelConfig};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;
use serde::Serialize;

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InnerBackend>;
type Device = <TrainBackend as BackendTrait>::Device;

#[derive(Parser, Debug)]
#[command(name = "core_train_bench")]
struct Args {
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 4)]
    repetitions: usize,
    #[arg(long)]
    output_dir: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    block: usize,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    rollout_fast_steps: usize,
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct StepTiming {
    forward_ms: f64,
    backward_ms: f64,
    optimizer_ms: f64,
    total_ms: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct RunSummary {
    forward_ms: f64,
    backward_ms: f64,
    optimizer_ms: f64,
    total_ms: f64,
    slow_tokens_per_s: f64,
    effective_rollout_tokens_per_s: f64,
}

#[derive(Clone, Copy, Debug, Serialize)]
struct ParitySummary {
    loss_abs_diff: f64,
    logits_max_abs: f64,
    logits_mean_abs: f64,
}

#[derive(Clone, Debug, Serialize)]
struct CaseResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    baseline: RunSummary,
    fused: RunSummary,
    total_speedup_x: f64,
    forward_speedup_x: f64,
    backward_speedup_x: f64,
    optimizer_speedup_x: f64,
    parity: ParitySummary,
}

#[derive(Clone, Debug, Serialize)]
struct Report {
    benchmark: &'static str,
    warmup: usize,
    repetitions: usize,
    cases: Vec<CaseResult>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "tiny_fs1",
        batch: 4,
        block: 32,
        n_layer: 2,
        n_embd: 32,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 256,
        rollout_fast_steps: 1,
    },
    BenchCase {
        name: "tiny_fs4",
        batch: 4,
        block: 32,
        n_layer: 2,
        n_embd: 32,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 256,
        rollout_fast_steps: 4,
    },
    BenchCase {
        name: "base_fs4",
        batch: 4,
        block: 64,
        n_layer: 4,
        n_embd: 64,
        n_head: 8,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 512,
        rollout_fast_steps: 4,
    },
];

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn build_config(case: &BenchCase, fused: bool) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: case.n_layer,
        n_embd: case.n_embd,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: case.mlp_internal_dim_multiplier,
        vocab_size: case.vocab_size,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_recurrent_kernel: fused,
            wgpu_rollout_fused: fused,
            ..Default::default()
        },
        ..Default::default()
    };
    config.fused_kernels.set_block_sizes(8, 8);
    config.set_rollout_fast_steps_per_slow_step(case.rollout_fast_steps);
    config
}

fn sample_tokens(
    case: &BenchCase,
    device: &Device,
) -> (Tensor<TrainBackend, 2, Int>, Tensor<TrainBackend, 2, Int>) {
    let token_count = case.batch * case.block;
    let inputs: Vec<i64> = (0..token_count)
        .map(|idx| (idx % case.vocab_size.max(2)) as i64)
        .collect();
    let targets: Vec<i64> = inputs
        .iter()
        .map(|token| ((*token as usize + 1) % case.vocab_size.max(2)) as i64)
        .collect();
    let inputs = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(inputs, [case.batch, case.block]),
        device,
    );
    let targets = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(targets, [case.batch, case.block]),
        device,
    );
    (inputs, targets)
}

fn train_step_timed(
    model: BDH<TrainBackend>,
    inputs: Tensor<TrainBackend, 2, Int>,
    targets: Tensor<TrainBackend, 2, Int>,
    device: &Device,
) -> StepTiming {
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TrainBackend, BDH<TrainBackend>>();
    let lr: LearningRate = 1e-3;

    let _ = TrainBackend::sync(device);
    let start_forward = Instant::now();
    let logits = model.forward(inputs);
    let loss = language_model_loss::<TrainBackend>(logits, targets);
    let _ = TrainBackend::sync(device);
    let forward_ms = start_forward.elapsed().as_secs_f64() * 1_000.0;

    let start_backward = Instant::now();
    let grads = loss.backward();
    let _ = TrainBackend::sync(device);
    let backward_ms = start_backward.elapsed().as_secs_f64() * 1_000.0;

    let start_optimizer = Instant::now();
    let grads = GradientsParams::from_grads(grads, &model);
    let _updated = optimizer.step(lr, model, grads);
    let _ = TrainBackend::sync(device);
    let optimizer_ms = start_optimizer.elapsed().as_secs_f64() * 1_000.0;

    StepTiming {
        forward_ms,
        backward_ms,
        optimizer_ms,
        total_ms: forward_ms + backward_ms + optimizer_ms,
    }
}

fn average_timing(samples: &[StepTiming], case: &BenchCase) -> RunSummary {
    let repetitions = samples.len().max(1) as f64;
    let forward_ms = samples.iter().map(|sample| sample.forward_ms).sum::<f64>() / repetitions;
    let backward_ms = samples.iter().map(|sample| sample.backward_ms).sum::<f64>() / repetitions;
    let optimizer_ms = samples
        .iter()
        .map(|sample| sample.optimizer_ms)
        .sum::<f64>()
        / repetitions;
    let total_ms = samples.iter().map(|sample| sample.total_ms).sum::<f64>() / repetitions;
    let slow_tokens = (case.batch * case.block) as f64;
    let effective_tokens = slow_tokens * case.rollout_fast_steps as f64;
    let seconds = (total_ms / 1_000.0).max(f64::EPSILON);

    RunSummary {
        forward_ms,
        backward_ms,
        optimizer_ms,
        total_ms,
        slow_tokens_per_s: slow_tokens / seconds,
        effective_rollout_tokens_per_s: effective_tokens / seconds,
    }
}

fn diff_metrics<const D: usize>(
    lhs: Tensor<InnerBackend, D>,
    rhs: Tensor<InnerBackend, D>,
) -> (f64, f64) {
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

    let mut max_abs = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
        let diff = f64::from((*lhs_value - *rhs_value).abs());
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }

    let mean_abs = if lhs.is_empty() {
        0.0
    } else {
        sum_abs / lhs.len() as f64
    };

    (max_abs, mean_abs)
}

fn train_one_step_for_parity(
    model: BDH<TrainBackend>,
    inputs: Tensor<TrainBackend, 2, Int>,
    targets: Tensor<TrainBackend, 2, Int>,
) -> (BDH<TrainBackend>, f32) {
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TrainBackend, BDH<TrainBackend>>();
    let lr: LearningRate = 1e-3;
    let logits = model.forward(inputs);
    let loss = language_model_loss::<TrainBackend>(logits, targets);
    let loss_value = loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let model = optimizer.step(lr, model, grads);
    (model, loss_value)
}

fn parity_summary(case: &BenchCase, device: &Device) -> ParitySummary {
    <TrainBackend as BackendTrait>::seed(device, 20_260 + case.rollout_fast_steps as u64);
    let baseline = BDH::<TrainBackend>::new(build_config(case, false), device);
    <TrainBackend as BackendTrait>::seed(device, 20_260 + case.rollout_fast_steps as u64);
    let fused = BDH::<TrainBackend>::new(build_config(case, true), device);
    let (inputs, targets) = sample_tokens(case, device);

    let (baseline, baseline_loss) =
        train_one_step_for_parity(baseline, inputs.clone(), targets.clone());
    let (fused, fused_loss) = train_one_step_for_parity(fused, inputs.clone(), targets);
    let loss_abs_diff = f64::from((baseline_loss - fused_loss).abs());

    let baseline_logits = baseline.forward(inputs.clone()).inner();
    let fused_logits = fused.forward(inputs).inner();
    let (logits_max_abs, logits_mean_abs) = diff_metrics(baseline_logits, fused_logits);

    ParitySummary {
        loss_abs_diff,
        logits_max_abs,
        logits_mean_abs,
    }
}

fn run_case(case: &BenchCase, args: &Args, device: &Device) -> CaseResult {
    let (inputs, targets) = sample_tokens(case, device);
    let baseline_model = BDH::<TrainBackend>::new(build_config(case, false), device);
    let fused_model = BDH::<TrainBackend>::new(build_config(case, true), device);

    for _ in 0..args.warmup {
        let _ = train_step_timed(
            baseline_model.clone(),
            inputs.clone(),
            targets.clone(),
            device,
        );
        let _ = train_step_timed(fused_model.clone(), inputs.clone(), targets.clone(), device);
    }

    let mut baseline_samples = Vec::with_capacity(args.repetitions);
    let mut fused_samples = Vec::with_capacity(args.repetitions);
    for _ in 0..args.repetitions {
        baseline_samples.push(train_step_timed(
            baseline_model.clone(),
            inputs.clone(),
            targets.clone(),
            device,
        ));
        fused_samples.push(train_step_timed(
            fused_model.clone(),
            inputs.clone(),
            targets.clone(),
            device,
        ));
    }

    let baseline = average_timing(&baseline_samples, case);
    let fused = average_timing(&fused_samples, case);
    let parity = parity_summary(case, device);

    CaseResult {
        case: *case,
        warmup: args.warmup,
        repetitions: args.repetitions,
        baseline,
        fused,
        total_speedup_x: baseline.total_ms / fused.total_ms.max(f64::EPSILON),
        forward_speedup_x: baseline.forward_ms / fused.forward_ms.max(f64::EPSILON),
        backward_speedup_x: baseline.backward_ms / fused.backward_ms.max(f64::EPSILON),
        optimizer_speedup_x: baseline.optimizer_ms / fused.optimizer_ms.max(f64::EPSILON),
        parity,
    }
}

fn format_markdown(report: &Report) -> String {
    let mut markdown = String::new();
    writeln!(&mut markdown, "# {}", report.benchmark).unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "warmup={} repetitions={}",
        report.warmup, report.repetitions
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "| case | fast_steps | total_ms_base | total_ms_fused | speedup_x | base tok/s | fused tok/s | base rollout tok/s | fused rollout tok/s | loss_abs | logits_max_abs | logits_mean_abs |"
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();

    for result in &report.cases {
        writeln!(
            &mut markdown,
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {:.1} | {:.6} | {:.6} | {:.6} |",
            result.case.name,
            result.case.rollout_fast_steps,
            result.baseline.total_ms,
            result.fused.total_ms,
            result.total_speedup_x,
            result.baseline.slow_tokens_per_s,
            result.fused.slow_tokens_per_s,
            result.baseline.effective_rollout_tokens_per_s,
            result.fused.effective_rollout_tokens_per_s,
            result.parity.loss_abs_diff,
            result.parity.logits_max_abs,
            result.parity.logits_mean_abs,
        )
        .unwrap();
    }

    markdown
}

fn format_csv(report: &Report) -> String {
    let mut csv = String::from(
        "case,batch,block,n_layer,n_embd,n_head,fast_steps,warmup,repetitions,baseline_forward_ms,baseline_backward_ms,baseline_optimizer_ms,baseline_total_ms,fused_forward_ms,fused_backward_ms,fused_optimizer_ms,fused_total_ms,total_speedup_x,forward_speedup_x,backward_speedup_x,optimizer_speedup_x,baseline_slow_tokens_per_s,fused_slow_tokens_per_s,baseline_effective_rollout_tokens_per_s,fused_effective_rollout_tokens_per_s,loss_abs_diff,logits_max_abs,logits_mean_abs\n",
    );
    for result in &report.cases {
        let case = result.case;
        let _ = writeln!(
            &mut csv,
            "{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            case.name,
            case.batch,
            case.block,
            case.n_layer,
            case.n_embd,
            case.n_head,
            case.rollout_fast_steps,
            result.warmup,
            result.repetitions,
            result.baseline.forward_ms,
            result.baseline.backward_ms,
            result.baseline.optimizer_ms,
            result.baseline.total_ms,
            result.fused.forward_ms,
            result.fused.backward_ms,
            result.fused.optimizer_ms,
            result.fused.total_ms,
            result.total_speedup_x,
            result.forward_speedup_x,
            result.backward_speedup_x,
            result.optimizer_speedup_x,
            result.baseline.slow_tokens_per_s,
            result.fused.slow_tokens_per_s,
            result.baseline.effective_rollout_tokens_per_s,
            result.fused.effective_rollout_tokens_per_s,
            result.parity.loss_abs_diff,
            result.parity.logits_max_abs,
            result.parity.logits_mean_abs,
        );
    }
    csv
}

fn write_text(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create output directory");
    }
    fs::write(path, contents).expect("write artifact");
}

fn main() {
    let args = Args::parse();
    let device = Device::default();
    init_runtime(&device);

    let mut cases = Vec::with_capacity(CASES.len());
    for case in CASES {
        eprintln!("running {}...", case.name);
        cases.push(run_case(case, &args, &device));
    }

    let report = Report {
        benchmark: "burn_dragon core fused forward+backward train-step benchmark",
        warmup: args.warmup,
        repetitions: args.repetitions,
        cases,
    };

    let markdown = format_markdown(&report);
    let csv = format_csv(&report);
    let json = serde_json::to_string_pretty(&report).expect("serialize report");

    println!("{markdown}");

    if let Some(output_dir) = args.output_dir.as_ref() {
        write_text(&output_dir.join("core_train_bench.md"), &markdown);
        write_text(&output_dir.join("core_train_bench.csv"), &csv);
        write_text(&output_dir.join("core_train_bench.json"), &json);
    }
}
