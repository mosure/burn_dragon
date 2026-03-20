use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_cubecl::cubecl::Runtime;
use burn_dragon_kernel::api::recurrent::{
    RecurrentAttentionOutput, try_fused_recurrent_attention_wgpu,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

const CORE_FUSED_PATH_DISTINCT: bool = false;
const CORE_FUSED_PATH_DESCRIPTION: &str = "burn_dragon_core recurrent path delegates directly to burn_dragon_kernel::api::recurrent::try_fused_recurrent_attention_wgpu when wgpu_recurrent_kernel is enabled";
const STRICT_CONTEXT_MAX_ABS: f64 = 1e-3;
const STRICT_CONTEXT_RMSE_MAX: f64 = 2e-4;
const STRICT_RHO_MAX_ABS: f64 = 1e-3;
const STRICT_RHO_RMSE_MAX: f64 = 2e-4;

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
}

#[derive(Clone, Copy, Serialize)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

#[derive(Clone, Copy, Serialize)]
struct ErrorMetrics {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
    max_rel: f64,
}

#[derive(Clone, Copy, Serialize)]
struct StepMetrics {
    step: usize,
    context_max_abs: f64,
    context_mean_abs: f64,
    context_rmse: f64,
    context_max_rel: f64,
    rho_max_abs: f64,
    rho_mean_abs: f64,
    rho_rmse: f64,
    rho_max_rel: f64,
}

#[derive(Clone, Copy, Serialize)]
struct StepConsistencySummary {
    context_max_abs: f64,
    context_rmse_max: f64,
    rho_max_abs: f64,
    rho_rmse_max: f64,
    strict_pass: bool,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    extra_repetitions_applied: usize,
    sampled_gpu_utilization_pct: Option<f32>,
    baseline_ms: f64,
    raw_wgsl_ms: f64,
    core_fused_ms: f64,
    raw_speedup_x: f64,
    core_speedup_x: f64,
    core_fused_mode: &'static str,
    full_context_error: ErrorMetrics,
    final_rho_error: ErrorMetrics,
    step_consistency: StepConsistencySummary,
    step_metrics: Vec<StepMetrics>,
}

#[derive(Clone, Serialize)]
struct SweepReport {
    benchmark: &'static str,
    adapter: String,
    warmup: usize,
    repetitions: usize,
    extra_repetitions: usize,
    heavy_min_embd: usize,
    gpu_util_threshold: f32,
    core_fused_path_distinct: bool,
    core_fused_path_description: &'static str,
    strict_thresholds: StrictThresholds,
    memory_before_bytes: MemorySnapshot,
    memory_after_bytes: MemorySnapshot,
    cases: Vec<CaseResult>,
}

#[derive(Clone, Copy, Serialize)]
struct StrictThresholds {
    context_max_abs: f64,
    context_rmse_max: f64,
    rho_max_abs: f64,
    rho_rmse_max: f64,
}

#[derive(Clone)]
struct BenchConfig {
    warmup: usize,
    repetitions: usize,
    extra_repetitions: usize,
    heavy_min_embd: usize,
    gpu_util_threshold: f32,
    output_dir: Option<PathBuf>,
    markdown_path: Option<PathBuf>,
    csv_path: Option<PathBuf>,
    json_path: Option<PathBuf>,
    steps_csv_path: Option<PathBuf>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e96",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 96,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e128",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 128,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e192",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 192,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e256",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 256,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e384",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 384,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e512",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 512,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e640",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 640,
    },
    BenchCase {
        name: "sweep_b1_h8_t64_l64_e768",
        batch: 1,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 768,
    },
];

fn main() {
    let config = parse_args();

    let device = Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 1337);

    let (adapter, adapter_type) = adapter_info();
    assert!(
        !matches!(adapter_type, wgpu::DeviceType::Cpu),
        "wgpu benchmark selected a CPU adapter; refusing to continue (adapter: {adapter})"
    );
    let before = memory_snapshot(&device);

    let mut results = Vec::with_capacity(CASES.len());
    for case in CASES {
        results.push(run_case(case, &device, &config));
    }

    let after = memory_snapshot(&device);

    let report = SweepReport {
        benchmark: "burn_dragon_kernel recurrent scaling sweep",
        adapter,
        warmup: config.warmup,
        repetitions: config.repetitions,
        extra_repetitions: config.extra_repetitions,
        heavy_min_embd: config.heavy_min_embd,
        gpu_util_threshold: config.gpu_util_threshold,
        core_fused_path_distinct: CORE_FUSED_PATH_DISTINCT,
        core_fused_path_description: CORE_FUSED_PATH_DESCRIPTION,
        strict_thresholds: StrictThresholds {
            context_max_abs: STRICT_CONTEXT_MAX_ABS,
            context_rmse_max: STRICT_CONTEXT_RMSE_MAX,
            rho_max_abs: STRICT_RHO_MAX_ABS,
            rho_rmse_max: STRICT_RHO_RMSE_MAX,
        },
        memory_before_bytes: before,
        memory_after_bytes: after,
        cases: results,
    };

    let markdown = format_markdown(&report);
    let summary_csv = format_summary_csv(&report.cases);
    let steps_csv = format_steps_csv(&report.cases);
    let json = serde_json::to_string_pretty(&report).expect("serialize sweep report");

    println!("{markdown}");

    if let Some(path) = config.markdown_path.as_ref() {
        write_text_artifact(path, &markdown, "markdown artifact");
    }
    if let Some(path) = config.csv_path.as_ref() {
        write_text_artifact(path, &summary_csv, "csv artifact");
    }
    if let Some(path) = config.steps_csv_path.as_ref() {
        write_text_artifact(path, &steps_csv, "step csv artifact");
    }
    if let Some(path) = config.json_path.as_ref() {
        write_text_artifact(path, &json, "json artifact");
    }
}

fn parse_args() -> BenchConfig {
    let mut config = BenchConfig {
        warmup: 2,
        repetitions: 4,
        extra_repetitions: 3,
        heavy_min_embd: 512,
        gpu_util_threshold: 25.0,
        output_dir: None,
        markdown_path: None,
        csv_path: None,
        json_path: None,
        steps_csv_path: None,
    };

    let args: Vec<String> = std::env::args().collect();
    let mut idx = 1usize;
    while idx < args.len() {
        match args[idx].as_str() {
            "--warmup" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.warmup = value.parse().unwrap_or(config.warmup);
                }
            }
            "--repetitions" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.repetitions = value.parse().unwrap_or(config.repetitions);
                }
            }
            "--extra-repetitions" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.extra_repetitions = value.parse().unwrap_or(config.extra_repetitions);
                }
            }
            "--heavy-min-embd" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.heavy_min_embd = value.parse().unwrap_or(config.heavy_min_embd);
                }
            }
            "--gpu-util-threshold" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.gpu_util_threshold = value.parse().unwrap_or(config.gpu_util_threshold);
                }
            }
            "--output-dir" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.output_dir = Some(PathBuf::from(value));
                }
            }
            "--artifact" | "--artifact-md" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.markdown_path = Some(PathBuf::from(value));
                }
            }
            "--artifact-csv" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.csv_path = Some(PathBuf::from(value));
                }
            }
            "--artifact-json" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.json_path = Some(PathBuf::from(value));
                }
            }
            "--artifact-steps-csv" => {
                idx += 1;
                if let Some(value) = args.get(idx) {
                    config.steps_csv_path = Some(PathBuf::from(value));
                }
            }
            _ => {}
        }

        idx += 1;
    }

    config.warmup = config.warmup.max(1);
    config.repetitions = config.repetitions.max(1);

    if let Some(output_dir) = config.output_dir.clone() {
        if config.markdown_path.is_none() {
            config.markdown_path = Some(output_dir.join("recurrent_scaling.md"));
        }
        if config.csv_path.is_none() {
            config.csv_path = Some(output_dir.join("recurrent_scaling.csv"));
        }
        if config.json_path.is_none() {
            config.json_path = Some(output_dir.join("recurrent_scaling.json"));
        }
        if config.steps_csv_path.is_none() {
            config.steps_csv_path = Some(output_dir.join("recurrent_scaling_steps.csv"));
        }
    }

    config
}

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn adapter_info() -> (String, wgpu::DeviceType) {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu adapter");
    let info = adapter.get_info();

    let summary = format!(
        "name={} backend={:?} type={:?} vendor={} device={}",
        info.name, info.backend, info.device_type, info.vendor, info.device
    );

    (summary, info.device_type)
}

fn memory_snapshot(device: &Device) -> MemorySnapshot {
    let usage = <burn_wgpu::WgpuRuntime as Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

fn run_case(case: &BenchCase, device: &Device, config: &BenchConfig) -> CaseResult {
    let query = Tensor::<Backend, 4>::random(
        [case.batch, case.heads, case.time, case.latent],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = Tensor::<Backend, 4>::random(
        [case.batch, case.value_heads, case.time, case.embd],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let rho = Tensor::<Backend, 4>::zeros([case.batch, case.heads, case.latent, case.embd], device);

    let decay_values: Vec<f32> = (0..case.heads)
        .map(|idx| 0.85_f32 + 0.1_f32 * (idx as f32 / case.heads as f32))
        .collect();
    let decay = Tensor::<Backend, 1>::from_floats(decay_values.as_slice(), device);

    let sampled_gpu_utilization_pct = query_gpu_utilization_percent();
    let allow_heavy_loop = sampled_gpu_utilization_pct
        .map(|util| util <= config.gpu_util_threshold)
        .unwrap_or(false);
    let extra_repetitions_applied =
        if allow_heavy_loop && case.embd >= config.heavy_min_embd && config.extra_repetitions > 0 {
            config.extra_repetitions
        } else {
            0
        };
    let repetitions = config.repetitions + extra_repetitions_applied;

    for _ in 0..config.warmup {
        let (context, _) = baseline_recurrent(
            query.clone(),
            value.clone(),
            rho.clone(),
            Some(decay.clone()),
        );
        sync_tensor(context);
    }

    for _ in 0..config.warmup {
        let output =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("raw fused recurrent output");
        sync_tensor(output.context);
    }

    let baseline_total = measure_avg(repetitions, || {
        let (context, _) = baseline_recurrent(
            query.clone(),
            value.clone(),
            rho.clone(),
            Some(decay.clone()),
        );
        sync_tensor(context);
    });

    let raw_total = measure_avg(repetitions, || {
        let output =
            try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
                .expect("raw fused recurrent output");
        sync_tensor(output.context);
    });

    let baseline_ms = duration_ms(baseline_total);
    let raw_wgsl_ms = duration_ms(raw_total);

    let (core_fused_ms, core_fused_mode) = if CORE_FUSED_PATH_DISTINCT {
        let core_total = measure_avg(repetitions, || {
            let output = core_fused_recurrent_path(&query, &value, &rho, &decay);
            sync_tensor(output.context);
        });
        (duration_ms(core_total), "measured_distinct")
    } else {
        (raw_wgsl_ms, "alias_raw_wgsl")
    };

    let (baseline_context, baseline_rho) = baseline_recurrent(
        query.clone(),
        value.clone(),
        rho.clone(),
        Some(decay.clone()),
    );
    let raw_output =
        try_fused_recurrent_attention_wgpu::<Backend>(&query, &value, Some(&rho), Some(&decay))
            .expect("raw fused recurrent output");

    let full_context_error = compare_tensors(baseline_context.clone(), raw_output.context.clone());
    let final_rho_error = compare_tensors(baseline_rho.clone(), raw_output.rho.clone());

    let (step_metrics, step_consistency) = stepwise_consistency(case, &query, &value, &rho, &decay);

    CaseResult {
        case: *case,
        warmup: config.warmup,
        repetitions,
        extra_repetitions_applied,
        sampled_gpu_utilization_pct,
        baseline_ms,
        raw_wgsl_ms,
        core_fused_ms,
        raw_speedup_x: baseline_ms / raw_wgsl_ms,
        core_speedup_x: baseline_ms / core_fused_ms,
        core_fused_mode,
        full_context_error,
        final_rho_error,
        step_consistency,
        step_metrics,
    }
}

fn stepwise_consistency(
    case: &BenchCase,
    query: &Tensor<Backend, 4>,
    value: &Tensor<Backend, 4>,
    rho: &Tensor<Backend, 4>,
    decay: &Tensor<Backend, 1>,
) -> (Vec<StepMetrics>, StepConsistencySummary) {
    let mut baseline_state = rho.clone();
    let mut raw_state = rho.clone();

    let mut context_max_abs = 0.0_f64;
    let mut context_rmse_max = 0.0_f64;
    let mut rho_max_abs = 0.0_f64;
    let mut rho_rmse_max = 0.0_f64;

    let mut step_metrics = Vec::with_capacity(case.time);

    for step in 0..case.time {
        let q_step = query.clone().slice_dim(2, step..step + 1);
        let v_step = value.clone().slice_dim(2, step..step + 1);

        let (baseline_context, next_baseline_state) = baseline_recurrent(
            q_step.clone(),
            v_step.clone(),
            baseline_state,
            Some(decay.clone()),
        );

        let raw_output = try_fused_recurrent_attention_wgpu::<Backend>(
            &q_step,
            &v_step,
            Some(&raw_state),
            Some(decay),
        )
        .expect("raw fused recurrent output");

        let context_error = compare_tensors(baseline_context, raw_output.context);
        let rho_error = compare_tensors(next_baseline_state.clone(), raw_output.rho.clone());

        context_max_abs = context_max_abs.max(context_error.max_abs);
        context_rmse_max = context_rmse_max.max(context_error.rmse);
        rho_max_abs = rho_max_abs.max(rho_error.max_abs);
        rho_rmse_max = rho_rmse_max.max(rho_error.rmse);

        step_metrics.push(StepMetrics {
            step,
            context_max_abs: context_error.max_abs,
            context_mean_abs: context_error.mean_abs,
            context_rmse: context_error.rmse,
            context_max_rel: context_error.max_rel,
            rho_max_abs: rho_error.max_abs,
            rho_mean_abs: rho_error.mean_abs,
            rho_rmse: rho_error.rmse,
            rho_max_rel: rho_error.max_rel,
        });

        baseline_state = next_baseline_state;
        raw_state = raw_output.rho;
    }

    let strict_pass = context_max_abs <= STRICT_CONTEXT_MAX_ABS
        && context_rmse_max <= STRICT_CONTEXT_RMSE_MAX
        && rho_max_abs <= STRICT_RHO_MAX_ABS
        && rho_rmse_max <= STRICT_RHO_RMSE_MAX;

    (
        step_metrics,
        StepConsistencySummary {
            context_max_abs,
            context_rmse_max,
            rho_max_abs,
            rho_rmse_max,
            strict_pass,
        },
    )
}

fn compare_tensors<const D: usize>(
    lhs: Tensor<Backend, D>,
    rhs: Tensor<Backend, D>,
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

    assert_eq!(lhs.len(), rhs.len(), "tensor length mismatch");

    if lhs.is_empty() {
        return ErrorMetrics {
            max_abs: 0.0,
            mean_abs: 0.0,
            rmse: 0.0,
            max_rel: 0.0,
        };
    }

    let mut max_abs = 0.0_f64;
    let mut max_rel = 0.0_f64;
    let mut sum_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;

    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a as f64 - *b as f64).abs();
        let denom = (*b as f64).abs().max(1e-6);
        let rel = diff / denom;

        max_abs = max_abs.max(diff);
        max_rel = max_rel.max(rel);
        sum_abs += diff;
        sum_sq += diff * diff;
    }

    let n = lhs.len() as f64;

    ErrorMetrics {
        max_abs,
        mean_abs: sum_abs / n,
        rmse: (sum_sq / n).sqrt(),
        max_rel,
    }
}

fn core_fused_recurrent_path(
    query: &Tensor<Backend, 4>,
    value: &Tensor<Backend, 4>,
    rho: &Tensor<Backend, 4>,
    decay: &Tensor<Backend, 1>,
) -> RecurrentAttentionOutput<Backend> {
    // This mirrors the burn_dragon_core recurrent fused dispatch branch.
    try_fused_recurrent_attention_wgpu::<Backend>(query, value, Some(rho), Some(decay))
        .expect("core fused recurrent output")
}

fn query_gpu_utilization_percent() -> Option<f32> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let stdout = String::from_utf8(output.stdout).ok()?;
    let mut max_util: Option<f32> = None;

    for line in stdout.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        if let Ok(value) = trimmed.parse::<f32>() {
            max_util = Some(match max_util {
                Some(current) => current.max(value),
                None => value,
            });
        }
    }

    max_util
}

fn baseline_recurrent(
    query: Tensor<Backend, 4>,
    value: Tensor<Backend, 4>,
    rho: Tensor<Backend, 4>,
    decay: Option<Tensor<Backend, 1>>,
) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
    let [batch, heads, time, _latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];

    let decay = decay.map(|tensor| tensor.reshape([1, heads, 1, 1]));
    let mut state = rho;
    let mut outputs = Vec::with_capacity(time);

    for t in 0..time {
        let q_t = query.clone().slice_dim(2, t..t + 1);
        let v_t = if value_heads == 1 {
            value.clone().slice_dim(1, 0..1).repeat_dim(1, heads)
        } else {
            value.clone().slice_dim(1, 0..heads)
        }
        .slice_dim(2, t..t + 1);

        let q_latent = q_t.swap_dims(2, 3);
        let context = (state.clone() * q_latent.clone())
            .sum_dim(2)
            .reshape([batch, heads, 1, embd]);
        outputs.push(context);

        state = state + q_latent * v_t;
        if let Some(decay) = &decay {
            state = state * decay.clone();
        }
    }

    (Tensor::cat(outputs, 2), state)
}

fn measure_avg<F: FnMut()>(repetitions: usize, mut f: F) -> Duration {
    let mut total = Duration::ZERO;
    for _ in 0..repetitions {
        let start = Instant::now();
        f();
        total += start.elapsed();
    }
    total.div_f64(repetitions as f64)
}

fn sync_tensor(tensor: Tensor<Backend, 4>) {
    let _ = tensor
        .slice_dim(0, 0..1)
        .slice_dim(1, 0..1)
        .slice_dim(2, 0..1)
        .slice_dim(3, 0..1)
        .to_data();
}

fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

fn format_markdown(report: &SweepReport) -> String {
    let mut markdown = String::new();
    let strict_pass = report
        .cases
        .iter()
        .filter(|case| case.step_consistency.strict_pass)
        .count();

    writeln!(&mut markdown, "{}", report.benchmark).unwrap();
    writeln!(&mut markdown, "adapter: {}", report.adapter).unwrap();
    writeln!(
        &mut markdown,
        "warmup: {}, repetitions: {}, extra_repetitions: {}, heavy_min_embd: {}, gpu_util_threshold_pct: {:.1}",
        report.warmup,
        report.repetitions,
        report.extra_repetitions,
        report.heavy_min_embd,
        report.gpu_util_threshold
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "core_fused_path_distinct: {}",
        report.core_fused_path_distinct
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "core_fused_path_description: {}",
        report.core_fused_path_description
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "strict_consistency_thresholds: context_max_abs<={:.2e}, context_rmse<={:.2e}, rho_max_abs<={:.2e}, rho_rmse<={:.2e}",
        report.strict_thresholds.context_max_abs,
        report.strict_thresholds.context_rmse_max,
        report.strict_thresholds.rho_max_abs,
        report.strict_thresholds.rho_rmse_max,
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "strict_consistency_pass_cases: {}/{}",
        strict_pass,
        report.cases.len()
    )
    .unwrap();
    writeln!(&mut markdown).unwrap();

    writeln!(&mut markdown, "| case | embd | baseline_ms | raw_wgsl_ms | core_fused_ms | raw_speedup_x | core_speedup_x | step_ctx_max_abs | step_rho_max_abs | strict_pass | reps | gpu_util_pct | core_mode |",).unwrap();
    writeln!(
        &mut markdown,
        "| :-- | --: | --: | --: | --: | --: | --: | --: | --: | :--: | --: | --: | :-- |",
    )
    .unwrap();

    for case in &report.cases {
        let util = case
            .sampled_gpu_utilization_pct
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".to_string());

        writeln!(
            &mut markdown,
            "| {} | {} | {:.3} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2e} | {:.2e} | {} | {} | {} | {} |",
            case.case.name,
            case.case.embd,
            case.baseline_ms,
            case.raw_wgsl_ms,
            case.core_fused_ms,
            case.raw_speedup_x,
            case.core_speedup_x,
            case.step_consistency.context_max_abs,
            case.step_consistency.rho_max_abs,
            if case.step_consistency.strict_pass {
                "pass"
            } else {
                "fail"
            },
            case.repetitions,
            util,
            case.core_fused_mode,
        )
        .unwrap();
    }

    writeln!(&mut markdown).unwrap();
    writeln!(
        &mut markdown,
        "memory_before_bytes: reserved={} in_use={}",
        report.memory_before_bytes.reserved, report.memory_before_bytes.in_use
    )
    .unwrap();
    writeln!(
        &mut markdown,
        "memory_after_bytes: reserved={} in_use={}",
        report.memory_after_bytes.reserved, report.memory_after_bytes.in_use
    )
    .unwrap();

    markdown
}

fn format_summary_csv(results: &[CaseResult]) -> String {
    let mut csv = String::new();
    writeln!(
        &mut csv,
        "case,embd,batch,heads,value_heads,time,latent,warmup,repetitions,extra_repetitions_applied,sampled_gpu_utilization_pct,baseline_ms,raw_wgsl_ms,core_fused_ms,raw_speedup_x,core_speedup_x,full_context_max_abs,full_context_mean_abs,full_context_rmse,full_context_max_rel,final_rho_max_abs,final_rho_mean_abs,final_rho_rmse,final_rho_max_rel,step_context_max_abs,step_context_rmse_max,step_rho_max_abs,step_rho_rmse_max,strict_consistency_pass,core_fused_mode"
    )
    .unwrap();

    for result in results {
        let util = result
            .sampled_gpu_utilization_pct
            .map(|value| format!("{value:.3}"))
            .unwrap_or_default();

        writeln!(
            &mut csv,
            "{},{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{},{}",
            result.case.name,
            result.case.embd,
            result.case.batch,
            result.case.heads,
            result.case.value_heads,
            result.case.time,
            result.case.latent,
            result.warmup,
            result.repetitions,
            result.extra_repetitions_applied,
            util,
            result.baseline_ms,
            result.raw_wgsl_ms,
            result.core_fused_ms,
            result.raw_speedup_x,
            result.core_speedup_x,
            result.full_context_error.max_abs,
            result.full_context_error.mean_abs,
            result.full_context_error.rmse,
            result.full_context_error.max_rel,
            result.final_rho_error.max_abs,
            result.final_rho_error.mean_abs,
            result.final_rho_error.rmse,
            result.final_rho_error.max_rel,
            result.step_consistency.context_max_abs,
            result.step_consistency.context_rmse_max,
            result.step_consistency.rho_max_abs,
            result.step_consistency.rho_rmse_max,
            result.step_consistency.strict_pass,
            result.core_fused_mode,
        )
        .unwrap();
    }

    csv
}

fn format_steps_csv(results: &[CaseResult]) -> String {
    let mut csv = String::new();
    writeln!(
        &mut csv,
        "case,embd,step,context_max_abs,context_mean_abs,context_rmse,context_max_rel,rho_max_abs,rho_mean_abs,rho_rmse,rho_max_rel"
    )
    .unwrap();

    for result in results {
        for step in &result.step_metrics {
            writeln!(
                &mut csv,
                "{},{},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e}",
                result.case.name,
                result.case.embd,
                step.step,
                step.context_max_abs,
                step.context_mean_abs,
                step.context_rmse,
                step.context_max_rel,
                step.rho_max_abs,
                step.rho_mean_abs,
                step.rho_rmse,
                step.rho_max_rel,
            )
            .unwrap();
        }
    }

    csv
}

fn write_text_artifact(path: &Path, content: &str, label: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create artifact dir");
    }
    fs::write(path, content).expect("write artifact");
    eprintln!("wrote {} to {}", label, path.display());
}
