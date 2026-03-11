#![recursion_limit = "256"]

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon::language::{
    ContextStrategy, GenerationSettings, generate_tokens, generate_tokens_chunked,
    generation_profile_reset, generation_profile_snapshot,
};
use burn_dragon::{BDH, BDHConfig, FusedKernelConfig};
use burn_dragon_wgpu::{recurrent_profile_reset, recurrent_profile_snapshot};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;
use serde::Serialize;

type InferBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InferBackend>;

#[derive(Parser, Debug, Clone)]
#[command(
    author,
    version,
    about = "Rollout generation benchmark sweep for task_32678880"
)]
struct Args {
    #[arg(
        long,
        default_value = "docs/artifacts/task_32678880-9135-421e-9bcc-fdd88f1fa1c0"
    )]
    output_dir: PathBuf,
    #[arg(long, value_delimiter = ',', default_value = "1,2,4,8,16")]
    rollout_fast_steps: Vec<usize>,
    #[arg(long, value_delimiter = ',', default_value = "64,256")]
    sequence_lengths: Vec<usize>,
    #[arg(long, default_value_t = 96)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 2)]
    inference_warmup: usize,
    #[arg(long, default_value_t = 4)]
    inference_iters: usize,
    #[arg(long, default_value_t = 0)]
    training_warmup: usize,
    #[arg(long, default_value_t = 1)]
    training_iters: usize,
    #[arg(long, default_value_t = 8)]
    chunk_tokens: usize,
    #[arg(long, default_value_t = 64)]
    device_buffer_tokens: usize,
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
enum Mode {
    Baseline,
    RolloutChunked,
}

impl Mode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Baseline => "baseline",
            Self::RolloutChunked => "rollout_chunked",
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq, PartialOrd, Ord)]
struct CaseKey {
    rollout_fast_steps: usize,
    sequence_len: usize,
    mode: Mode,
}

#[derive(Clone, Debug, Serialize)]
struct CorrectnessRow {
    rollout_fast_steps: usize,
    sequence_len: usize,
    pass: bool,
    mismatch_index: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct InferenceRow {
    mode: Mode,
    rollout_fast_steps: usize,
    sequence_len: usize,
    generated_tokens: usize,
    iterations: usize,
    avg_elapsed_ms: f64,
    avg_tokens_per_sec: f64,
    avg_recurrent_calls: f64,
    avg_host_sync_points: f64,
    avg_host_to_device_copy_bytes: f64,
    avg_device_to_host_copy_bytes: f64,
    avg_chunk_flushes: f64,
    avg_chunk_flush_ns: f64,
    avg_dispatch_ns: f64,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingRow {
    mode: Mode,
    rollout_fast_steps: usize,
    sequence_len: usize,
    train_tokens_per_step: usize,
    iterations: usize,
    avg_step_ms: f64,
    avg_tokens_per_sec: f64,
    avg_recurrent_calls: f64,
    avg_dispatch_ns: f64,
}

#[derive(Clone, Debug, Serialize)]
struct DeltaRow {
    rollout_fast_steps: usize,
    sequence_len: usize,
    inference_speedup_x: f64,
    inference_tokens_per_sec_delta_pct: f64,
    recurrent_call_reduction_pct: f64,
    host_sync_reduction_pct: f64,
    host_to_device_copy_reduction_pct: f64,
    device_to_host_copy_reduction_pct: f64,
    chunk_flush_overhead_pct: f64,
    training_speedup_x: f64,
}

#[derive(Clone, Debug, Serialize)]
struct RolloutBenchReport {
    benchmark: &'static str,
    max_new_tokens: usize,
    rollout_fast_steps: Vec<usize>,
    sequence_lengths: Vec<usize>,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
    correctness: Vec<CorrectnessRow>,
    inference: Vec<InferenceRow>,
    training: Vec<TrainingRow>,
    deltas: Vec<DeltaRow>,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let args = Args::parse();
    if std::env::var_os("BDH_STAGE_PROFILE").is_none() {
        return Err(anyhow!(
            "BDH_STAGE_PROFILE is required for rollout profiling metrics (example: BDH_STAGE_PROFILE=1 cargo run -p burn_dragon_cli --bin rollout_bench -- ...)"
        ));
    }
    if args.rollout_fast_steps.is_empty() {
        return Err(anyhow!("--rollout-fast-steps must not be empty"));
    }
    if args.sequence_lengths.is_empty() {
        return Err(anyhow!("--sequence-lengths must not be empty"));
    }
    if args.inference_iters == 0 {
        return Err(anyhow!("--inference-iters must be >= 1"));
    }

    let output_dir = args.output_dir.clone();
    fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;

    let infer_device = <InferBackend as BackendTrait>::Device::default();
    init_wgpu_runtime(&infer_device);
    <InferBackend as BackendTrait>::seed(&infer_device, 2026);

    let train_device = <TrainBackend as BackendTrait>::Device::default();
    <TrainBackend as BackendTrait>::seed(&train_device, 2026);

    let mut correctness = Vec::new();
    let mut inference = Vec::new();
    let mut training = Vec::new();

    for &rollout_fast_steps in &args.rollout_fast_steps {
        if !BDHConfig::is_valid_rollout_fast_steps(rollout_fast_steps) {
            return Err(anyhow!(
                "unsupported rollout_fast_steps={rollout_fast_steps}; expected one of {:?}",
                BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS
            ));
        }

        for &sequence_len in &args.sequence_lengths {
            correctness.push(run_correctness_case(
                rollout_fast_steps,
                sequence_len,
                args.max_new_tokens.min(32),
                args.chunk_tokens,
                args.device_buffer_tokens,
                &infer_device,
            ));

            inference.push(run_inference_case(
                Mode::Baseline,
                rollout_fast_steps,
                sequence_len,
                &args,
                &infer_device,
            )?);
            inference.push(run_inference_case(
                Mode::RolloutChunked,
                rollout_fast_steps,
                sequence_len,
                &args,
                &infer_device,
            )?);

            training.push(run_training_case(
                Mode::Baseline,
                rollout_fast_steps,
                sequence_len,
                &args,
                &train_device,
            )?);
            training.push(run_training_case(
                Mode::RolloutChunked,
                rollout_fast_steps,
                sequence_len,
                &args,
                &train_device,
            )?);
        }
    }

    let deltas = compute_deltas(&inference, &training);
    let report = RolloutBenchReport {
        benchmark: "burn_bdh rollout generation sweep",
        max_new_tokens: args.max_new_tokens,
        rollout_fast_steps: args.rollout_fast_steps.clone(),
        sequence_lengths: args.sequence_lengths.clone(),
        chunk_tokens: args.chunk_tokens.max(1),
        device_buffer_tokens: args.device_buffer_tokens.max(args.chunk_tokens.max(1)),
        correctness,
        inference,
        training,
        deltas,
    };

    let json_path = output_dir.join("rollout_bench.json");
    let inference_csv_path = output_dir.join("rollout_bench_inference.csv");
    let training_csv_path = output_dir.join("rollout_bench_training.csv");
    let delta_csv_path = output_dir.join("rollout_bench_deltas.csv");
    let markdown_path = output_dir.join("rollout_bench_summary.md");

    let report_json = serde_json::to_string_pretty(&report).context("serialize benchmark json")?;
    fs::write(&json_path, report_json)
        .with_context(|| format!("failed to write {}", json_path.display()))?;
    fs::write(&inference_csv_path, format_inference_csv(&report.inference))
        .with_context(|| format!("failed to write {}", inference_csv_path.display()))?;
    fs::write(&training_csv_path, format_training_csv(&report.training))
        .with_context(|| format!("failed to write {}", training_csv_path.display()))?;
    fs::write(&delta_csv_path, format_delta_csv(&report.deltas))
        .with_context(|| format!("failed to write {}", delta_csv_path.display()))?;
    fs::write(&markdown_path, format_markdown_summary(&report))
        .with_context(|| format!("failed to write {}", markdown_path.display()))?;

    println!(
        "rollout benchmark artifacts:\n- {}\n- {}\n- {}\n- {}\n- {}",
        json_path.display(),
        inference_csv_path.display(),
        training_csv_path.display(),
        delta_csv_path.display(),
        markdown_path.display()
    );
    Ok(())
}

fn init_wgpu_runtime(device: &<InferBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn build_model_config(rollout_fast_steps: usize, mode: Mode) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        vocab_size: 256,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: true,
            wgpu_recurrent_kernel: true,
            wgpu_rollout_fused: mode == Mode::RolloutChunked,
            ..Default::default()
        },
        ..Default::default()
    };
    config.fused_kernels.set_block_sizes(32, 32);
    config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    config
}

fn make_prompt(sequence_len: usize, vocab_size: usize) -> Vec<i64> {
    (0..sequence_len)
        .map(|idx| (idx % vocab_size) as i64)
        .collect()
}

fn run_correctness_case(
    rollout_fast_steps: usize,
    sequence_len: usize,
    max_new_tokens: usize,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
    device: &<InferBackend as BackendTrait>::Device,
) -> CorrectnessRow {
    let baseline_config = build_model_config(rollout_fast_steps, Mode::Baseline);
    <InferBackend as BackendTrait>::seed(device, 2_026);
    let baseline_model = BDH::<InferBackend>::new(baseline_config.clone(), device);
    let settings = GenerationSettings {
        max_new_tokens: Some(max_new_tokens),
        temperature: 1.0,
        top_k: Some(1),
        strategy: ContextStrategy::Infinite,
    };
    let prompt = make_prompt(sequence_len, baseline_config.vocab_size);
    let baseline_tokens = generate_tokens(&baseline_model, prompt.clone(), device, settings, None)
        .expect("baseline generation should succeed");

    let chunked_config = build_model_config(rollout_fast_steps, Mode::RolloutChunked);
    <InferBackend as BackendTrait>::seed(device, 2_026);
    let chunked_model = BDH::<InferBackend>::new(chunked_config, device);
    let chunked_tokens = generate_tokens_chunked(
        &chunked_model,
        prompt,
        device,
        settings,
        chunk_tokens.max(1),
        device_buffer_tokens.max(chunk_tokens.max(1)),
        None,
    )
    .expect("chunked generation should succeed");

    let mismatch_index = baseline_tokens
        .iter()
        .zip(chunked_tokens.iter())
        .position(|(lhs, rhs)| lhs != rhs);
    let pass = baseline_tokens.len() == chunked_tokens.len() && mismatch_index.is_none();
    CorrectnessRow {
        rollout_fast_steps,
        sequence_len,
        pass,
        mismatch_index,
    }
}

fn run_inference_case(
    mode: Mode,
    rollout_fast_steps: usize,
    sequence_len: usize,
    args: &Args,
    device: &<InferBackend as BackendTrait>::Device,
) -> Result<InferenceRow> {
    let config = build_model_config(rollout_fast_steps, mode);
    let prompt = make_prompt(sequence_len, config.vocab_size);
    let settings = GenerationSettings {
        max_new_tokens: Some(args.max_new_tokens),
        temperature: 1.0,
        top_k: Some(1),
        strategy: ContextStrategy::Infinite,
    };
    <InferBackend as BackendTrait>::seed(device, 4_000 + rollout_fast_steps as u64);
    let model = BDH::<InferBackend>::new(config, device);

    for _ in 0..args.inference_warmup {
        let _ = run_generation(
            &model,
            prompt.clone(),
            device,
            settings,
            mode,
            args.chunk_tokens,
            args.device_buffer_tokens,
        )?;
    }

    let mut elapsed_ns = Vec::with_capacity(args.inference_iters);
    let mut recurrent_calls = Vec::with_capacity(args.inference_iters);
    let mut host_sync_points = Vec::with_capacity(args.inference_iters);
    let mut host_to_device_copy_bytes = Vec::with_capacity(args.inference_iters);
    let mut device_to_host_copy_bytes = Vec::with_capacity(args.inference_iters);
    let mut chunk_flushes = Vec::with_capacity(args.inference_iters);
    let mut chunk_flush_ns = Vec::with_capacity(args.inference_iters);
    let mut dispatch_ns = Vec::with_capacity(args.inference_iters);

    for _ in 0..args.inference_iters {
        generation_profile_reset();
        recurrent_profile_reset();
        let started = Instant::now();
        let generated = run_generation(
            &model,
            prompt.clone(),
            device,
            settings,
            mode,
            args.chunk_tokens,
            args.device_buffer_tokens,
        )?;
        let elapsed = started.elapsed().as_nanos();
        let expected = prompt.len() + args.max_new_tokens;
        if generated.len() != expected {
            return Err(anyhow!(
                "inference generated unexpected token count for mode={} rollout={} seq={}: expected {}, got {}",
                mode.as_str(),
                rollout_fast_steps,
                sequence_len,
                expected,
                generated.len()
            ));
        }
        let g = generation_profile_snapshot();
        let r = recurrent_profile_snapshot();
        elapsed_ns.push(elapsed);
        recurrent_calls.push(r.calls as f64);
        host_sync_points.push(g.host_sync_points as f64);
        host_to_device_copy_bytes.push(g.host_to_device_copy_bytes as f64);
        device_to_host_copy_bytes.push(g.device_to_host_copy_bytes as f64);
        chunk_flushes.push(g.chunk_flushes as f64);
        chunk_flush_ns.push(g.chunk_flush_ns as f64);
        dispatch_ns.push(r.dispatch_ns as f64);
    }

    let avg_elapsed_ns = mean(&elapsed_ns);
    let generated_tokens = args.max_new_tokens;
    let tokens_per_sec = generated_tokens as f64 / (avg_elapsed_ns / 1e9);

    Ok(InferenceRow {
        mode,
        rollout_fast_steps,
        sequence_len,
        generated_tokens,
        iterations: args.inference_iters,
        avg_elapsed_ms: avg_elapsed_ns / 1e6,
        avg_tokens_per_sec: tokens_per_sec,
        avg_recurrent_calls: mean_f64(&recurrent_calls),
        avg_host_sync_points: mean_f64(&host_sync_points),
        avg_host_to_device_copy_bytes: mean_f64(&host_to_device_copy_bytes),
        avg_device_to_host_copy_bytes: mean_f64(&device_to_host_copy_bytes),
        avg_chunk_flushes: mean_f64(&chunk_flushes),
        avg_chunk_flush_ns: mean_f64(&chunk_flush_ns),
        avg_dispatch_ns: mean_f64(&dispatch_ns),
    })
}

fn run_generation(
    model: &BDH<InferBackend>,
    prompt: Vec<i64>,
    device: &<InferBackend as BackendTrait>::Device,
    settings: GenerationSettings,
    mode: Mode,
    chunk_tokens: usize,
    device_buffer_tokens: usize,
) -> Result<Vec<i64>> {
    match mode {
        Mode::Baseline => generate_tokens(model, prompt, device, settings, None),
        Mode::RolloutChunked => generate_tokens_chunked(
            model,
            prompt,
            device,
            settings,
            chunk_tokens.max(1),
            device_buffer_tokens.max(chunk_tokens.max(1)),
            None,
        ),
    }
}

fn run_training_case(
    mode: Mode,
    rollout_fast_steps: usize,
    sequence_len: usize,
    args: &Args,
    device: &<TrainBackend as BackendTrait>::Device,
) -> Result<TrainingRow> {
    if args.training_iters == 0 {
        return Ok(TrainingRow {
            mode,
            rollout_fast_steps,
            sequence_len,
            train_tokens_per_step: 0,
            iterations: 0,
            avg_step_ms: 0.0,
            avg_tokens_per_sec: 0.0,
            avg_recurrent_calls: 0.0,
            avg_dispatch_ns: 0.0,
        });
    }

    let config = build_model_config(rollout_fast_steps, mode);
    <TrainBackend as BackendTrait>::seed(device, 9_000 + rollout_fast_steps as u64);
    let model = BDH::<TrainBackend>::new(config, device);
    let batch = 2usize;
    let train_tokens_per_step = batch * sequence_len;
    let token_values: Vec<i64> = (0..train_tokens_per_step)
        .map(|idx| (idx % 255) as i64)
        .collect();
    let inputs = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(token_values, [batch, sequence_len]),
        device,
    );

    for _ in 0..args.training_warmup {
        let _ = model.forward(inputs.clone());
    }

    let mut step_elapsed_ns = Vec::with_capacity(args.training_iters);
    let mut recurrent_calls = Vec::with_capacity(args.training_iters);
    let mut dispatch_ns = Vec::with_capacity(args.training_iters);

    for _ in 0..args.training_iters {
        recurrent_profile_reset();
        let started = Instant::now();
        let _ = model.forward(inputs.clone());
        let elapsed = started.elapsed().as_nanos();
        let recurrent = recurrent_profile_snapshot();
        step_elapsed_ns.push(elapsed);
        recurrent_calls.push(recurrent.calls as f64);
        dispatch_ns.push(recurrent.dispatch_ns as f64);
    }

    let avg_step_ns = mean(&step_elapsed_ns);
    let tokens_per_sec = train_tokens_per_step as f64 / (avg_step_ns / 1e9);

    Ok(TrainingRow {
        mode,
        rollout_fast_steps,
        sequence_len,
        train_tokens_per_step,
        iterations: args.training_iters,
        avg_step_ms: avg_step_ns / 1e6,
        avg_tokens_per_sec: tokens_per_sec,
        avg_recurrent_calls: mean_f64(&recurrent_calls),
        avg_dispatch_ns: mean_f64(&dispatch_ns),
    })
}

fn compute_deltas(inference: &[InferenceRow], training: &[TrainingRow]) -> Vec<DeltaRow> {
    let mut inf_map = BTreeMap::new();
    for row in inference {
        inf_map.insert(
            CaseKey {
                rollout_fast_steps: row.rollout_fast_steps,
                sequence_len: row.sequence_len,
                mode: row.mode,
            },
            row,
        );
    }
    let mut train_map = BTreeMap::new();
    for row in training {
        train_map.insert(
            CaseKey {
                rollout_fast_steps: row.rollout_fast_steps,
                sequence_len: row.sequence_len,
                mode: row.mode,
            },
            row,
        );
    }

    let mut deltas = Vec::new();
    for baseline in inference.iter().filter(|row| row.mode == Mode::Baseline) {
        let key_chunked = CaseKey {
            rollout_fast_steps: baseline.rollout_fast_steps,
            sequence_len: baseline.sequence_len,
            mode: Mode::RolloutChunked,
        };
        let Some(chunked) = inf_map.get(&key_chunked) else {
            continue;
        };
        let train_baseline = train_map.get(&CaseKey {
            rollout_fast_steps: baseline.rollout_fast_steps,
            sequence_len: baseline.sequence_len,
            mode: Mode::Baseline,
        });
        let train_chunked = train_map.get(&CaseKey {
            rollout_fast_steps: baseline.rollout_fast_steps,
            sequence_len: baseline.sequence_len,
            mode: Mode::RolloutChunked,
        });

        let inference_speedup_x =
            safe_ratio(chunked.avg_tokens_per_sec, baseline.avg_tokens_per_sec);
        let inference_tokens_per_sec_delta_pct =
            safe_delta_pct(chunked.avg_tokens_per_sec, baseline.avg_tokens_per_sec);
        let recurrent_call_reduction_pct =
            safe_reduction_pct(chunked.avg_recurrent_calls, baseline.avg_recurrent_calls);
        let host_sync_reduction_pct =
            safe_reduction_pct(chunked.avg_host_sync_points, baseline.avg_host_sync_points);
        let host_to_device_copy_reduction_pct = safe_reduction_pct(
            chunked.avg_host_to_device_copy_bytes,
            baseline.avg_host_to_device_copy_bytes,
        );
        let device_to_host_copy_reduction_pct = safe_reduction_pct(
            chunked.avg_device_to_host_copy_bytes,
            baseline.avg_device_to_host_copy_bytes,
        );
        let chunk_flush_overhead_pct = safe_ratio(
            chunked.avg_chunk_flush_ns,
            chunked.avg_elapsed_ms.max(1e-9) * 1e6,
        ) * 100.0;
        let training_speedup_x = match (train_baseline, train_chunked) {
            (Some(base), Some(opt)) => safe_ratio(opt.avg_tokens_per_sec, base.avg_tokens_per_sec),
            _ => 1.0,
        };

        deltas.push(DeltaRow {
            rollout_fast_steps: baseline.rollout_fast_steps,
            sequence_len: baseline.sequence_len,
            inference_speedup_x,
            inference_tokens_per_sec_delta_pct,
            recurrent_call_reduction_pct,
            host_sync_reduction_pct,
            host_to_device_copy_reduction_pct,
            device_to_host_copy_reduction_pct,
            chunk_flush_overhead_pct,
            training_speedup_x,
        });
    }

    deltas
}

fn safe_ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator <= 0.0 {
        return 0.0;
    }
    numerator / denominator
}

fn safe_delta_pct(new_value: f64, base_value: f64) -> f64 {
    if base_value.abs() < f64::EPSILON {
        return 0.0;
    }
    (new_value - base_value) / base_value * 100.0
}

fn safe_reduction_pct(new_value: f64, base_value: f64) -> f64 {
    if base_value.abs() < f64::EPSILON {
        return 0.0;
    }
    (base_value - new_value) / base_value * 100.0
}

fn mean(values: &[u128]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let sum: u128 = values.iter().copied().sum();
    sum as f64 / values.len() as f64
}

fn mean_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().copied().sum::<f64>() / values.len() as f64
}

fn format_inference_csv(rows: &[InferenceRow]) -> String {
    let mut csv = String::from(
        "mode,rollout_fast_steps,sequence_len,generated_tokens,iterations,avg_elapsed_ms,avg_tokens_per_sec,avg_recurrent_calls,avg_host_sync_points,avg_host_to_device_copy_bytes,avg_device_to_host_copy_bytes,avg_chunk_flushes,avg_chunk_flush_ns,avg_dispatch_ns\n",
    );
    for row in rows {
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{:.6},{:.6},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3},{:.3}",
            row.mode.as_str(),
            row.rollout_fast_steps,
            row.sequence_len,
            row.generated_tokens,
            row.iterations,
            row.avg_elapsed_ms,
            row.avg_tokens_per_sec,
            row.avg_recurrent_calls,
            row.avg_host_sync_points,
            row.avg_host_to_device_copy_bytes,
            row.avg_device_to_host_copy_bytes,
            row.avg_chunk_flushes,
            row.avg_chunk_flush_ns,
            row.avg_dispatch_ns,
        );
    }
    csv
}

fn format_training_csv(rows: &[TrainingRow]) -> String {
    let mut csv = String::from(
        "mode,rollout_fast_steps,sequence_len,train_tokens_per_step,iterations,avg_step_ms,avg_tokens_per_sec,avg_recurrent_calls,avg_dispatch_ns\n",
    );
    for row in rows {
        let _ = writeln!(
            csv,
            "{},{},{},{},{},{:.6},{:.6},{:.3},{:.3}",
            row.mode.as_str(),
            row.rollout_fast_steps,
            row.sequence_len,
            row.train_tokens_per_step,
            row.iterations,
            row.avg_step_ms,
            row.avg_tokens_per_sec,
            row.avg_recurrent_calls,
            row.avg_dispatch_ns,
        );
    }
    csv
}

fn format_delta_csv(rows: &[DeltaRow]) -> String {
    let mut csv = String::from(
        "rollout_fast_steps,sequence_len,inference_speedup_x,inference_tokens_per_sec_delta_pct,recurrent_call_reduction_pct,host_sync_reduction_pct,host_to_device_copy_reduction_pct,device_to_host_copy_reduction_pct,chunk_flush_overhead_pct,training_speedup_x\n",
    );
    for row in rows {
        let _ = writeln!(
            csv,
            "{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
            row.rollout_fast_steps,
            row.sequence_len,
            row.inference_speedup_x,
            row.inference_tokens_per_sec_delta_pct,
            row.recurrent_call_reduction_pct,
            row.host_sync_reduction_pct,
            row.host_to_device_copy_reduction_pct,
            row.device_to_host_copy_reduction_pct,
            row.chunk_flush_overhead_pct,
            row.training_speedup_x,
        );
    }
    csv
}

fn format_markdown_summary(report: &RolloutBenchReport) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# Rollout Benchmark Summary");
    let _ = writeln!(out, "");
    let _ = writeln!(out, "- benchmark: {}", report.benchmark);
    let _ = writeln!(out, "- max_new_tokens: {}", report.max_new_tokens);
    let _ = writeln!(out, "- rollout_fast_steps: {:?}", report.rollout_fast_steps);
    let _ = writeln!(out, "- sequence_lengths: {:?}", report.sequence_lengths);
    let _ = writeln!(out, "- chunk_tokens: {}", report.chunk_tokens);
    let _ = writeln!(
        out,
        "- device_buffer_tokens: {}",
        report.device_buffer_tokens
    );
    let pass_count = report.correctness.iter().filter(|row| row.pass).count();
    let _ = writeln!(
        out,
        "- correctness: {}/{} deterministic parity checks passed",
        pass_count,
        report.correctness.len()
    );
    let _ = writeln!(out, "");
    let _ = writeln!(out, "## Delta Table");
    let _ = writeln!(out, "");
    let _ = writeln!(
        out,
        "| rollout | seq | inf speedup x | inf delta % | recurrent call reduction % | host sync reduction % | H2D copy reduction % | D2H copy reduction % | chunk flush overhead % | train speedup x |"
    );
    let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
    for row in &report.deltas {
        let _ = writeln!(
            out,
            "| {} | {} | {:.3} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.3} |",
            row.rollout_fast_steps,
            row.sequence_len,
            row.inference_speedup_x,
            row.inference_tokens_per_sec_delta_pct,
            row.recurrent_call_reduction_pct,
            row.host_sync_reduction_pct,
            row.host_to_device_copy_reduction_pct,
            row.device_to_host_copy_reduction_pct,
            row.chunk_flush_overhead_pct,
            row.training_speedup_x,
        );
    }
    out
}

#[allow(dead_code)]
fn write_text(path: &Path, contents: &str) -> Result<()> {
    fs::write(path, contents).with_context(|| format!("failed to write {}", path.display()))
}
