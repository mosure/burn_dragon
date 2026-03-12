use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon_wgpu::api::spatial::{
    CompiledStructuredPyramidRhoPlan, LocalGridNeighborhood, LocalGridShape2d,
    StructuredPyramidRhoStepInput, StructuredPyramidRhoStepOutput, StructuredPyramidShape,
    reference_structured_pyramid_rho_step, structured_pyramid_profile_reset,
    structured_pyramid_profile_snapshot,
    try_fused_structured_pyramid_rho_step_wgpu_with_plan,
};
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
    repetitions: usize,
    #[arg(long, default_value_t = 4)]
    steps: usize,
    #[arg(long)]
    markdown_path: Option<PathBuf>,
    #[arg(long)]
    json_path: Option<PathBuf>,
}

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    patch_height: usize,
    patch_width: usize,
    coarse_stride: usize,
    hub_count: usize,
    rank: usize,
    value_dim: usize,
}

#[derive(Clone, Copy, Serialize)]
struct ErrorMetrics {
    max_abs: f32,
    mean_abs: f32,
}

#[derive(Clone, Serialize)]
struct MeasurementResult {
    mode: &'static str,
    steps: usize,
    baseline_time_ms: f64,
    fused_time_ms: f64,
    speedup_x: f64,
    baseline_steps_per_sec: f64,
    fused_steps_per_sec: f64,
    baseline_kernel_calls: f64,
    fused_kernel_calls: f64,
    baseline_launches: f64,
    fused_launches: f64,
    baseline_dispatch_ms: f64,
    fused_dispatch_ms: f64,
    baseline_allocations: f64,
    fused_allocations: f64,
    baseline_metadata_bytes: f64,
    fused_metadata_bytes: f64,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    problem_shape: String,
    one_step: MeasurementResult,
    resident_rollout: MeasurementResult,
    loss_abs_diff: f32,
    patch_context_error: ErrorMetrics,
    coarse_context_error: ErrorMetrics,
    patch_rho_error: ErrorMetrics,
    coarse_rho_error: ErrorMetrics,
    hub_rho_error: ErrorMetrics,
}

#[derive(Clone, Serialize)]
struct Report {
    benchmark: &'static str,
    adapter: String,
    warmup: usize,
    repetitions: usize,
    steps: usize,
    cases: Vec<CaseResult>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "small",
        batch: 2,
        patch_height: 8,
        patch_width: 8,
        coarse_stride: 2,
        hub_count: 2,
        rank: 8,
        value_dim: 32,
    },
    BenchCase {
        name: "medium",
        batch: 2,
        patch_height: 16,
        patch_width: 16,
        coarse_stride: 2,
        hub_count: 4,
        rank: 12,
        value_dim: 48,
    },
    BenchCase {
        name: "large",
        batch: 1,
        patch_height: 24,
        patch_width: 24,
        coarse_stride: 3,
        hub_count: 4,
        rank: 16,
        value_dim: 64,
    },
];

fn main() {
    let args = Args::parse();
    let device = Device::default();
    init_runtime(&device);

    let report = Report {
        benchmark: "burn_dragon vision structured pyramid forward+backward benchmark",
        adapter: adapter_info(),
        warmup: args.warmup,
        repetitions: args.repetitions,
        steps: args.steps.max(1),
        cases: CASES
            .iter()
            .copied()
            .map(|case| run_case(case, &device, &args))
            .collect(),
    };

    let markdown = format_markdown(&report);
    let json = serde_json::to_string_pretty(&report).expect("serialize structured pyramid fb report");

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

fn run_case(case: BenchCase, device: &Device, args: &Args) -> CaseResult {
    let coarse_height = (case.patch_height / case.coarse_stride.max(1)).max(1);
    let coarse_width = (case.patch_width / case.coarse_stride.max(1)).max(1);
    let shape = StructuredPyramidShape {
        patch: LocalGridShape2d::new(case.patch_height, case.patch_width),
        coarse: LocalGridShape2d::new(coarse_height, coarse_width),
        coarse_stride: case.coarse_stride,
        hub_count: case.hub_count,
    };
    let neighborhood = LocalGridNeighborhood::moore(1);
    <TrainBackend as BackendTrait>::seed(device, 8_080 + case.rank as u64 + case.value_dim as u64);

    let patch_query = Tensor::<TrainBackend, 4>::random(
        [case.batch, case.rank, case.patch_height, case.patch_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let patch_value = Tensor::<TrainBackend, 4>::random(
        [case.batch, case.value_dim, case.patch_height, case.patch_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let coarse_query = Tensor::<TrainBackend, 4>::random(
        [case.batch, case.rank, coarse_height, coarse_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let coarse_value = Tensor::<TrainBackend, 4>::random(
        [case.batch, case.value_dim, coarse_height, coarse_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let patch_rho = Tensor::<TrainBackend, 5>::zeros(
        [
            case.batch,
            case.rank,
            case.value_dim,
            case.patch_height,
            case.patch_width,
        ],
        device,
    );
    let coarse_rho = Tensor::<TrainBackend, 5>::zeros(
        [case.batch, case.rank, case.value_dim, coarse_height, coarse_width],
        device,
    );
    let hub_rho =
        Tensor::<TrainBackend, 4>::zeros([case.batch, case.hub_count, case.rank, case.value_dim], device);
    let patch_hub_weights = Some(
        Tensor::<TrainBackend, 4>::ones(
            [case.batch, case.hub_count, case.patch_height, case.patch_width],
            device,
        )
        .div_scalar(case.hub_count as f32),
    );
    let coarse_hub_weights = Some(
        Tensor::<TrainBackend, 4>::ones(
            [case.batch, case.hub_count, coarse_height, coarse_width],
            device,
        )
        .div_scalar(case.hub_count as f32),
    );
    let decay_values = (0..case.rank)
        .map(|idx| 0.85_f32 + 0.1_f32 * (idx as f32 / case.rank.max(1) as f32))
        .collect::<Vec<_>>();
    let decay =
        Tensor::<TrainBackend, 1>::from_data(TensorData::new(decay_values, [case.rank]), device);
    let plan = CompiledStructuredPyramidRhoPlan::new(
        case.batch,
        case.rank,
        case.value_dim,
        shape,
        neighborhood,
        device,
    );

    let input = StructuredPyramidRhoStepInput {
        patch_query,
        patch_value,
        coarse_query,
        coarse_value,
        patch_rho,
        coarse_rho,
        hub_rho,
        patch_hub_weights,
        coarse_hub_weights,
        neighborhood,
        decay,
    };

    let (loss_abs_diff, patch_context_error, coarse_context_error, patch_rho_error, coarse_rho_error, hub_rho_error) =
        parity_snapshot(shape, input.clone(), &plan, args.steps.max(1), device);

    for _ in 0..args.warmup {
        let _ = run_forward_backward(shape, input.clone(), None, 1, device);
        let _ = run_forward_backward(shape, input.clone(), Some(&plan), 1, device);
        let _ = run_forward_backward(shape, input.clone(), None, args.steps.max(1), device);
        let _ = run_forward_backward(shape, input.clone(), Some(&plan), args.steps.max(1), device);
    }

    let mut baseline_one_step = Vec::with_capacity(args.repetitions);
    let mut fused_one_step = Vec::with_capacity(args.repetitions);
    let mut baseline_rollout = Vec::with_capacity(args.repetitions);
    let mut fused_rollout = Vec::with_capacity(args.repetitions);

    for _ in 0..args.repetitions {
        baseline_one_step.push(run_forward_backward(shape, input.clone(), None, 1, device));
        fused_one_step.push(run_forward_backward(shape, input.clone(), Some(&plan), 1, device));
        baseline_rollout.push(run_forward_backward(
            shape,
            input.clone(),
            None,
            args.steps.max(1),
            device,
        ));
        fused_rollout.push(run_forward_backward(
            shape,
            input.clone(),
            Some(&plan),
            args.steps.max(1),
            device,
        ));
    }

    CaseResult {
        case,
        problem_shape: format!(
            "b{} patch{}x{} coarse_stride{} hubs{} rank{} value{}",
            case.batch,
            case.patch_height,
            case.patch_width,
            case.coarse_stride,
            case.hub_count,
            case.rank,
            case.value_dim
        ),
        one_step: summarize_measurements("one_step", 1, &baseline_one_step, &fused_one_step),
        resident_rollout: summarize_measurements(
            "resident_rollout",
            args.steps.max(1),
            &baseline_rollout,
            &fused_rollout,
        ),
        loss_abs_diff,
        patch_context_error,
        coarse_context_error,
        patch_rho_error,
        coarse_rho_error,
        hub_rho_error,
    }
}

#[derive(Clone, Copy)]
struct StepMetrics {
    elapsed_ns: u128,
    kernel_calls: u64,
    launches: u64,
    dispatch_ns: u128,
    allocations: u64,
    metadata_upload_bytes: u64,
}

fn run_forward_backward(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<TrainBackend>,
    plan: Option<&CompiledStructuredPyramidRhoPlan<TrainBackend>>,
    steps: usize,
    device: &Device,
) -> StepMetrics {
    let _ = <TrainBackend as BackendTrait>::sync(device);
    structured_pyramid_profile_reset();
    let started = Instant::now();

    let mut patch_rho = input.patch_rho.clone();
    let mut coarse_rho = input.coarse_rho.clone();
    let mut hub_rho = input.hub_rho.clone();
    let mut loss: Option<Tensor<TrainBackend, 1>> = None;

    for _ in 0..steps.max(1) {
        let output = match plan {
            Some(plan) => try_fused_structured_pyramid_rho_step_wgpu_with_plan(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: input.patch_query.clone(),
                    patch_value: input.patch_value.clone(),
                    coarse_query: input.coarse_query.clone(),
                    coarse_value: input.coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: input.patch_hub_weights.clone(),
                    coarse_hub_weights: input.coarse_hub_weights.clone(),
                    neighborhood: input.neighborhood,
                    decay: input.decay.clone(),
                },
                plan,
            )
            .expect("structured pyramid fused output"),
            None => reference_structured_pyramid_rho_step(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: input.patch_query.clone(),
                    patch_value: input.patch_value.clone(),
                    coarse_query: input.coarse_query.clone(),
                    coarse_value: input.coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: input.patch_hub_weights.clone(),
                    coarse_hub_weights: input.coarse_hub_weights.clone(),
                    neighborhood: input.neighborhood,
                    decay: input.decay.clone(),
                },
            ),
        };
        patch_rho = output.next_patch_rho.clone();
        coarse_rho = output.next_coarse_rho.clone();
        hub_rho = output.next_hub_rho.clone();

        let step_loss = output.patch_local_context.mean()
            + output.coarse_local_context.mean()
            + output.patch_from_coarse_context.mean()
            + output.patch_from_hub_context.mean()
            + output.coarse_from_hub_context.mean()
            + output.next_patch_rho.mean()
            + output.next_coarse_rho.mean()
            + output.next_hub_rho.mean();
        loss = Some(match loss {
            Some(acc) => acc + step_loss,
            None => step_loss,
        });
    }

    let _ = loss.expect("structured pyramid loss").backward();
    let _ = <TrainBackend as BackendTrait>::sync(device);
    let profile = structured_pyramid_profile_snapshot();

    StepMetrics {
        elapsed_ns: started.elapsed().as_nanos(),
        kernel_calls: profile.calls,
        launches: profile.launches,
        dispatch_ns: profile.dispatch_ns,
        allocations: profile.transient_allocations,
        metadata_upload_bytes: profile.metadata_upload_bytes,
    }
}

#[allow(clippy::type_complexity)]
fn parity_snapshot(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<TrainBackend>,
    plan: &CompiledStructuredPyramidRhoPlan<TrainBackend>,
    steps: usize,
    device: &Device,
) -> (
    f32,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
) {
    let reference = rollout(shape, input.clone(), None, steps);
    let fused = rollout(shape, input, Some(plan), steps);

    let reference_loss = output_loss(reference.clone());
    let fused_loss = output_loss(fused.clone());
    let loss_abs_diff = (scalar_to_f32(reference_loss) - scalar_to_f32(fused_loss)).abs();
    let _ = <TrainBackend as BackendTrait>::sync(device);

    (
        loss_abs_diff,
        error_metrics(reference.patch_local_context.inner(), fused.patch_local_context.inner()),
        error_metrics(
            reference.coarse_local_context.inner(),
            fused.coarse_local_context.inner(),
        ),
        error_metrics(reference.next_patch_rho.inner(), fused.next_patch_rho.inner()),
        error_metrics(reference.next_coarse_rho.inner(), fused.next_coarse_rho.inner()),
        error_metrics(reference.next_hub_rho.inner(), fused.next_hub_rho.inner()),
    )
}

fn rollout(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<TrainBackend>,
    plan: Option<&CompiledStructuredPyramidRhoPlan<TrainBackend>>,
    steps: usize,
) -> StructuredPyramidRhoStepOutput<TrainBackend> {
    let mut patch_rho = input.patch_rho.clone();
    let mut coarse_rho = input.coarse_rho.clone();
    let mut hub_rho = input.hub_rho.clone();
    let mut last = None;

    for _ in 0..steps.max(1) {
        let output = match plan {
            Some(plan) => try_fused_structured_pyramid_rho_step_wgpu_with_plan(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: input.patch_query.clone(),
                    patch_value: input.patch_value.clone(),
                    coarse_query: input.coarse_query.clone(),
                    coarse_value: input.coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: input.patch_hub_weights.clone(),
                    coarse_hub_weights: input.coarse_hub_weights.clone(),
                    neighborhood: input.neighborhood,
                    decay: input.decay.clone(),
                },
                plan,
            )
            .expect("structured pyramid fused rollout output"),
            None => reference_structured_pyramid_rho_step(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: input.patch_query.clone(),
                    patch_value: input.patch_value.clone(),
                    coarse_query: input.coarse_query.clone(),
                    coarse_value: input.coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: input.patch_hub_weights.clone(),
                    coarse_hub_weights: input.coarse_hub_weights.clone(),
                    neighborhood: input.neighborhood,
                    decay: input.decay.clone(),
                },
            ),
        };
        patch_rho = output.next_patch_rho.clone();
        coarse_rho = output.next_coarse_rho.clone();
        hub_rho = output.next_hub_rho.clone();
        last = Some(output);
    }

    last.expect("structured pyramid rollout produced output")
}

fn output_loss(output: StructuredPyramidRhoStepOutput<TrainBackend>) -> Tensor<TrainBackend, 1> {
    output.patch_local_context.mean()
        + output.coarse_local_context.mean()
        + output.patch_from_coarse_context.mean()
        + output.patch_from_hub_context.mean()
        + output.coarse_from_hub_context.mean()
        + output.next_patch_rho.mean()
        + output.next_coarse_rho.mean()
        + output.next_hub_rho.mean()
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

fn summarize_measurements(
    mode: &'static str,
    steps: usize,
    baseline: &[StepMetrics],
    fused: &[StepMetrics],
) -> MeasurementResult {
    let baseline_ns = mean_u128(&baseline.iter().map(|m| m.elapsed_ns).collect::<Vec<_>>());
    let fused_ns = mean_u128(&fused.iter().map(|m| m.elapsed_ns).collect::<Vec<_>>());
    let baseline_steps_per_sec = steps as f64 / (baseline_ns / 1e9);
    let fused_steps_per_sec = steps as f64 / (fused_ns / 1e9);

    MeasurementResult {
        mode,
        steps,
        baseline_time_ms: baseline_ns / 1e6,
        fused_time_ms: fused_ns / 1e6,
        speedup_x: safe_ratio(baseline_ns, fused_ns),
        baseline_steps_per_sec,
        fused_steps_per_sec,
        baseline_kernel_calls: mean_f64(
            &baseline
                .iter()
                .map(|m| m.kernel_calls as f64)
                .collect::<Vec<_>>(),
        ),
        fused_kernel_calls: mean_f64(
            &fused.iter().map(|m| m.kernel_calls as f64).collect::<Vec<_>>(),
        ),
        baseline_launches: mean_f64(
            &baseline.iter().map(|m| m.launches as f64).collect::<Vec<_>>(),
        ),
        fused_launches: mean_f64(&fused.iter().map(|m| m.launches as f64).collect::<Vec<_>>()),
        baseline_dispatch_ms: mean_f64(
            &baseline
                .iter()
                .map(|m| m.dispatch_ns as f64 / 1e6)
                .collect::<Vec<_>>(),
        ),
        fused_dispatch_ms: mean_f64(
            &fused
                .iter()
                .map(|m| m.dispatch_ns as f64 / 1e6)
                .collect::<Vec<_>>(),
        ),
        baseline_allocations: mean_f64(
            &baseline
                .iter()
                .map(|m| m.allocations as f64)
                .collect::<Vec<_>>(),
        ),
        fused_allocations: mean_f64(
            &fused.iter().map(|m| m.allocations as f64).collect::<Vec<_>>(),
        ),
        baseline_metadata_bytes: mean_f64(
            &baseline
                .iter()
                .map(|m| m.metadata_upload_bytes as f64)
                .collect::<Vec<_>>(),
        ),
        fused_metadata_bytes: mean_f64(
            &fused
                .iter()
                .map(|m| m.metadata_upload_bytes as f64)
                .collect::<Vec<_>>(),
        ),
    }
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
    let _ = writeln!(out, "# Vision Pyramid Forward+Backward Benchmark");
    let _ = writeln!(out);
    let _ = writeln!(out, "- Benchmark: {}", report.benchmark);
    let _ = writeln!(out, "- Adapter: {}", report.adapter);
    let _ = writeln!(out, "- Warmup: {}", report.warmup);
    let _ = writeln!(out, "- Repetitions: {}", report.repetitions);
    let _ = writeln!(out, "- Rollout steps: {}", report.steps);
    let _ = writeln!(out);
    let _ = writeln!(
        out,
        "| case | mode | steps | baseline ms | fused ms | speedup x | baseline step/s | fused step/s | fused calls | fused launches | fused dispatch ms | fused allocs | loss abs diff | patch max abs | coarse max abs | patch rho max abs | coarse rho max abs | hub rho max abs |"
    );
    let _ = writeln!(
        out,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for row in &report.cases {
        for measurement in [&row.one_step, &row.resident_rollout] {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {:.1} | {:.1} | {:.3} | {:.1} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
                row.case.name,
                measurement.mode,
                measurement.steps,
                measurement.baseline_time_ms,
                measurement.fused_time_ms,
                measurement.speedup_x,
                measurement.baseline_steps_per_sec,
                measurement.fused_steps_per_sec,
                measurement.fused_kernel_calls,
                measurement.fused_launches,
                measurement.fused_dispatch_ms,
                measurement.fused_allocations,
                row.loss_abs_diff,
                row.patch_context_error.max_abs,
                row.coarse_context_error.max_abs,
                row.patch_rho_error.max_abs,
                row.coarse_rho_error.max_abs,
                row.hub_rho_error.max_abs,
            );
        }
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
