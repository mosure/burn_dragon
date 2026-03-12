use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_dragon_wgpu::api::spatial::{
    CompiledStructuredPyramidRhoPlan, LocalGridNeighborhood, LocalGridShape2d,
    StructuredPyramidRhoStepInput, StructuredPyramidShape, reference_structured_pyramid_rho_step,
    structured_pyramid_profile_reset, structured_pyramid_profile_snapshot,
    try_fused_structured_pyramid_rho_step_wgpu_with_plan,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    repetitions: usize,
    #[arg(long, default_value_t = 8)]
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
    reference_time_ms: f64,
    fused_time_ms: f64,
    speedup_x: f64,
    throughput_per_sec: f64,
    fused_kernel_calls: u64,
    fused_kernel_launches: u64,
    fused_kernel_dispatch_ms: f64,
    fused_transient_allocations: u64,
    fused_metadata_upload_bytes: u64,
    fused_metadata_reuse_bytes: u64,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    problem_shape: String,
    forward: MeasurementResult,
    resident_rollout: MeasurementResult,
    patch_local_error: ErrorMetrics,
    coarse_local_error: ErrorMetrics,
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
        benchmark: "burn_dragon vision structured pyramid benchmark",
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
    let json = serde_json::to_string_pretty(&report).expect("serialize structured pyramid report");
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

    let patch_query = Tensor::<Backend, 4>::random(
        [case.batch, case.rank, case.patch_height, case.patch_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let patch_value = Tensor::<Backend, 4>::random(
        [case.batch, case.value_dim, case.patch_height, case.patch_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let coarse_query = Tensor::<Backend, 4>::random(
        [case.batch, case.rank, coarse_height, coarse_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let coarse_value = Tensor::<Backend, 4>::random(
        [case.batch, case.value_dim, coarse_height, coarse_width],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let patch_rho = Tensor::<Backend, 5>::zeros(
        [
            case.batch,
            case.rank,
            case.value_dim,
            case.patch_height,
            case.patch_width,
        ],
        device,
    );
    let coarse_rho =
        Tensor::<Backend, 5>::zeros([case.batch, case.rank, case.value_dim, coarse_height, coarse_width], device);
    let hub_rho =
        Tensor::<Backend, 4>::zeros([case.batch, case.hub_count, case.rank, case.value_dim], device);
    let patch_hub_weights = Some(
        Tensor::<Backend, 4>::ones(
            [case.batch, case.hub_count, case.patch_height, case.patch_width],
            device,
        )
        .div_scalar(case.hub_count as f32),
    );
    let coarse_hub_weights = Some(
        Tensor::<Backend, 4>::ones(
            [case.batch, case.hub_count, coarse_height, coarse_width],
            device,
        )
        .div_scalar(case.hub_count as f32),
    );
    let decay_values = (0..case.rank)
        .map(|idx| 0.85_f32 + 0.1_f32 * (idx as f32 / case.rank.max(1) as f32))
        .collect::<Vec<_>>();
    let decay = Tensor::<Backend, 1>::from_data(TensorData::new(decay_values, [case.rank]), device);
    let plan = CompiledStructuredPyramidRhoPlan::new(
        case.batch,
        case.rank,
        case.value_dim,
        shape,
        neighborhood,
        device,
    );

    let input = StructuredPyramidRhoStepInput {
        patch_query: patch_query.clone(),
        patch_value: patch_value.clone(),
        coarse_query: coarse_query.clone(),
        coarse_value: coarse_value.clone(),
        patch_rho: patch_rho.clone(),
        coarse_rho: coarse_rho.clone(),
        hub_rho: hub_rho.clone(),
        patch_hub_weights: patch_hub_weights.clone(),
        coarse_hub_weights: coarse_hub_weights.clone(),
        neighborhood,
        decay: decay.clone(),
    };
    let reference = reference_structured_pyramid_rho_step(shape, input.clone());
    let fused = try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan)
        .expect("fused structured pyramid output");

    for _ in 0..args.warmup {
        let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(
            shape,
            StructuredPyramidRhoStepInput {
                patch_query: patch_query.clone(),
                patch_value: patch_value.clone(),
                coarse_query: coarse_query.clone(),
                coarse_value: coarse_value.clone(),
                patch_rho: patch_rho.clone(),
                coarse_rho: coarse_rho.clone(),
                hub_rho: hub_rho.clone(),
                patch_hub_weights: patch_hub_weights.clone(),
                coarse_hub_weights: coarse_hub_weights.clone(),
                neighborhood,
                decay: decay.clone(),
            },
            &plan,
        )
        .expect("structured pyramid warmup");
        sync_tensor(output.patch_local_context);
    }

    let forward_reference_ns = measure_avg(args.repetitions, || {
        let output = reference_structured_pyramid_rho_step(
            shape,
            StructuredPyramidRhoStepInput {
                patch_query: patch_query.clone(),
                patch_value: patch_value.clone(),
                coarse_query: coarse_query.clone(),
                coarse_value: coarse_value.clone(),
                patch_rho: patch_rho.clone(),
                coarse_rho: coarse_rho.clone(),
                hub_rho: hub_rho.clone(),
                patch_hub_weights: patch_hub_weights.clone(),
                coarse_hub_weights: coarse_hub_weights.clone(),
                neighborhood,
                decay: decay.clone(),
            },
        );
        sync_tensor(output.patch_local_context);
    });

    structured_pyramid_profile_reset();
    let forward_fused_ns = measure_avg(args.repetitions, || {
        let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(
            shape,
            StructuredPyramidRhoStepInput {
                patch_query: patch_query.clone(),
                patch_value: patch_value.clone(),
                coarse_query: coarse_query.clone(),
                coarse_value: coarse_value.clone(),
                patch_rho: patch_rho.clone(),
                coarse_rho: coarse_rho.clone(),
                hub_rho: hub_rho.clone(),
                patch_hub_weights: patch_hub_weights.clone(),
                coarse_hub_weights: coarse_hub_weights.clone(),
                neighborhood,
                decay: decay.clone(),
            },
            &plan,
        )
        .expect("structured pyramid forward");
        sync_tensor(output.patch_local_context);
    });
    let forward_profile = structured_pyramid_profile_snapshot();

    let rollout_reference_ns = measure_avg(args.repetitions, || {
        let mut patch_rho = patch_rho.clone();
        let mut coarse_rho = coarse_rho.clone();
        let mut hub_rho = hub_rho.clone();
        for _ in 0..args.steps {
            let output = reference_structured_pyramid_rho_step(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: patch_query.clone(),
                    patch_value: patch_value.clone(),
                    coarse_query: coarse_query.clone(),
                    coarse_value: coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: patch_hub_weights.clone(),
                    coarse_hub_weights: coarse_hub_weights.clone(),
                    neighborhood,
                    decay: decay.clone(),
                },
            );
            patch_rho = output.next_patch_rho;
            coarse_rho = output.next_coarse_rho;
            hub_rho = output.next_hub_rho;
        }
        sync_tensor(hub_rho);
    });

    structured_pyramid_profile_reset();
    let rollout_fused_ns = measure_avg(args.repetitions, || {
        let mut patch_rho = patch_rho.clone();
        let mut coarse_rho = coarse_rho.clone();
        let mut hub_rho = hub_rho.clone();
        for _ in 0..args.steps {
            let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: patch_query.clone(),
                    patch_value: patch_value.clone(),
                    coarse_query: coarse_query.clone(),
                    coarse_value: coarse_value.clone(),
                    patch_rho,
                    coarse_rho,
                    hub_rho,
                    patch_hub_weights: patch_hub_weights.clone(),
                    coarse_hub_weights: coarse_hub_weights.clone(),
                    neighborhood,
                    decay: decay.clone(),
                },
                &plan,
            )
            .expect("structured pyramid rollout");
            patch_rho = output.next_patch_rho;
            coarse_rho = output.next_coarse_rho;
            hub_rho = output.next_hub_rho;
        }
        sync_tensor(hub_rho);
    });
    let rollout_profile = structured_pyramid_profile_snapshot();

    CaseResult {
        case,
        problem_shape: format!(
            "b{} patch{}x{} coarse{}x{} hub{} r{} v{}",
            case.batch,
            case.patch_height,
            case.patch_width,
            coarse_height,
            coarse_width,
            case.hub_count,
            case.rank,
            case.value_dim
        ),
        forward: measurement(
            "forward_only",
            forward_reference_ns,
            forward_fused_ns,
            1.0,
            &forward_profile,
        ),
        resident_rollout: measurement(
            "resident_rollout",
            rollout_reference_ns,
            rollout_fused_ns,
            args.steps as f64,
            &rollout_profile,
        ),
        patch_local_error: error_metrics(fused.patch_local_context, reference.patch_local_context),
        coarse_local_error: error_metrics(
            fused.coarse_local_context,
            reference.coarse_local_context,
        ),
        patch_rho_error: error_metrics(fused.next_patch_rho, reference.next_patch_rho),
        coarse_rho_error: error_metrics(fused.next_coarse_rho, reference.next_coarse_rho),
        hub_rho_error: error_metrics(fused.next_hub_rho, reference.next_hub_rho),
    }
}

fn measurement(
    mode: &'static str,
    reference_ns: u128,
    fused_ns: u128,
    work_items: f64,
    profile: &burn_dragon_wgpu::api::spatial::StructuredPyramidProfileSnapshot,
) -> MeasurementResult {
    let reference_ms = nanos_to_ms(reference_ns);
    let fused_ms = nanos_to_ms(fused_ns);
    MeasurementResult {
        mode,
        reference_time_ms: reference_ms,
        fused_time_ms: fused_ms,
        speedup_x: if fused_ms > 0.0 { reference_ms / fused_ms } else { 0.0 },
        throughput_per_sec: if fused_ms > 0.0 {
            work_items / (fused_ms / 1_000.0)
        } else {
            0.0
        },
        fused_kernel_calls: profile.calls,
        fused_kernel_launches: profile.launches,
        fused_kernel_dispatch_ms: nanos_to_ms(profile.dispatch_ns),
        fused_transient_allocations: profile.transient_allocations,
        fused_metadata_upload_bytes: profile.metadata_upload_bytes,
        fused_metadata_reuse_bytes: profile.metadata_reuse_bytes,
    }
}

fn error_metrics<const D: usize>(lhs: Tensor<Backend, D>, rhs: Tensor<Backend, D>) -> ErrorMetrics {
    let diff = lhs
        .sub(rhs)
        .abs()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("error vec");
    let max_abs = diff.iter().copied().fold(0.0_f32, f32::max);
    let mean_abs = if diff.is_empty() {
        0.0
    } else {
        diff.iter().copied().sum::<f32>() / diff.len() as f32
    };
    ErrorMetrics { max_abs, mean_abs }
}

fn measure_avg(repetitions: usize, mut f: impl FnMut()) -> u128 {
    let reps = repetitions.max(1);
    let start = Instant::now();
    for _ in 0..reps {
        f();
    }
    start.elapsed().as_nanos() / reps as u128
}

fn sync_tensor<const D: usize>(tensor: Tensor<Backend, D>) {
    let _ = tensor.into_data();
}

fn nanos_to_ms(ns: u128) -> f64 {
    ns as f64 / 1_000_000.0
}

fn format_markdown(report: &Report) -> String {
    let mut out = String::new();
    let _ = writeln!(&mut out, "# {}", report.benchmark);
    let _ = writeln!(&mut out);
    let _ = writeln!(&mut out, "- adapter: {}", report.adapter);
    let _ = writeln!(&mut out, "- warmup: {}", report.warmup);
    let _ = writeln!(&mut out, "- repetitions: {}", report.repetitions);
    let _ = writeln!(&mut out, "- steps: {}", report.steps);
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "| case | shape | mode | ref ms | fused ms | speedup | throughput/s | launches | dispatch ms | meta reuse bytes |"
    );
    let _ = writeln!(
        &mut out,
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for case in &report.cases {
        for measurement in [&case.forward, &case.resident_rollout] {
            let _ = writeln!(
                &mut out,
                "| {} | {} | {} | {:.3} | {:.3} | {:.2}x | {:.2} | {} | {:.3} | {} |",
                case.case.name,
                case.problem_shape,
                measurement.mode,
                measurement.reference_time_ms,
                measurement.fused_time_ms,
                measurement.speedup_x,
                measurement.throughput_per_sec,
                measurement.fused_kernel_launches,
                measurement.fused_kernel_dispatch_ms,
                measurement.fused_metadata_reuse_bytes,
            );
        }
    }
    out
}

fn write_text_artifact(path: &Path, text: &str, label: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|err| {
            panic!("failed to create {label} parent {}: {err}", parent.display())
        });
    }
    fs::write(path, text)
        .unwrap_or_else(|err| panic!("failed to write {label} {}: {err}", path.display()));
}
