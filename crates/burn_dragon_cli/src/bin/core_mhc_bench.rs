use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon::core::{
    ManifoldHyperConnectionCoefficients, ManifoldHyperConnectionCoefficientPolicy,
    ManifoldHyperConnections, ManifoldHyperConnectionsConfig, mhc_passthrough_with_coefficients,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 2)]
    warmup: usize,
    #[arg(long, default_value_t = 5)]
    iterations: usize,
}

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    num_streams: usize,
    num_views: usize,
    batch: usize,
    time: usize,
    dim: usize,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    coefficient_ms: f64,
    width_baseline_ms: f64,
    width_optimized_ms: f64,
    width_speedup_x: f64,
    depth_baseline_ms: f64,
    depth_optimized_ms: f64,
    depth_speedup_x: f64,
    passthrough_baseline_ms: f64,
    passthrough_optimized_ms: f64,
    passthrough_speedup_x: f64,
    passthrough_reuse_baseline_ms: f64,
    passthrough_reuse_optimized_ms: f64,
    passthrough_reuse_speedup_x: f64,
    width_max_abs_diff: f32,
    passthrough_max_abs_diff: f32,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "expand_1x4",
        num_streams: 1,
        num_views: 4,
        batch: 8,
        time: 64,
        dim: 128,
    },
    BenchCase {
        name: "reduce_4x1",
        num_streams: 4,
        num_views: 1,
        batch: 8,
        time: 64,
        dim: 128,
    },
    BenchCase {
        name: "square_4x4",
        num_streams: 4,
        num_views: 4,
        batch: 8,
        time: 64,
        dim: 128,
    },
    BenchCase {
        name: "square_8x8",
        num_streams: 8,
        num_views: 8,
        batch: 4,
        time: 64,
        dim: 96,
    },
];

fn main() {
    let args = Args::parse();
    let device = Device::default();
    init_runtime(&device);
    let adapter = adapter_info();

    println!("# burn_dragon core mHC forward benchmark");
    println!();
    println!("- adapter: {adapter}");
    println!("- warmup: {}", args.warmup);
    println!("- iterations: {}", args.iterations);
    println!();

    for case in CASES.iter().copied() {
        let result = run_case(case, &device, &args);
        println!("## {}", case.name);
        println!(
            "- coefficient: {:.3} ms",
            result.coefficient_ms,
        );
        println!(
            "- width: baseline {:.3} ms, optimized {:.3} ms, speedup {:.2}x, max_abs_diff {:.3e}",
            result.width_baseline_ms,
            result.width_optimized_ms,
            result.width_speedup_x,
            result.width_max_abs_diff,
        );
        println!(
            "- depth: baseline {:.3} ms, optimized {:.3} ms, speedup {:.2}x",
            result.depth_baseline_ms,
            result.depth_optimized_ms,
            result.depth_speedup_x,
        );
        println!(
            "- passthrough: baseline {:.3} ms, optimized {:.3} ms, speedup {:.2}x, max_abs_diff {:.3e}",
            result.passthrough_baseline_ms,
            result.passthrough_optimized_ms,
            result.passthrough_speedup_x,
            result.passthrough_max_abs_diff,
        );
        println!(
            "- passthrough_reuse_coeffs: baseline {:.3} ms, optimized {:.3} ms, speedup {:.2}x",
            result.passthrough_reuse_baseline_ms,
            result.passthrough_reuse_optimized_ms,
            result.passthrough_reuse_speedup_x,
        );
        println!();
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

fn build_config(case: BenchCase) -> ManifoldHyperConnectionsConfig {
    ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: case.num_streams,
        num_views: case.num_views,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
        mhc_iters: 10,
        mhc_tau: 0.05,
        add_branch_out_to_residual: true,
        dropout: 0.0,
    }
}

fn sample_residuals(case: BenchCase, device: &Device) -> Tensor<Backend, 4> {
    <Backend as BackendTrait>::seed(device, 2_026);
    Tensor::<Backend, 4>::random(
        [case.batch, case.num_streams, case.time, case.dim],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn reference_mix_streams(
    residuals: Tensor<Backend, 4>,
    weights: Tensor<Backend, 2>,
) -> Tensor<Backend, 4> {
    let [batch, streams, time, dim] = residuals.shape().dims::<4>();
    let [in_streams, out_streams] = weights.shape().dims::<2>();
    assert_eq!(streams, in_streams);
    let flat = residuals
        .swap_dims(1, 2)
        .swap_dims(2, 3)
        .reshape([batch * time * dim, streams]);
    let mixed = flat.matmul(weights);
    mixed
        .reshape([batch, time, dim, out_streams])
        .swap_dims(2, 3)
        .swap_dims(1, 2)
}

fn reference_width(
    residuals: Tensor<Backend, 4>,
    coefficients: &ManifoldHyperConnectionCoefficients<Backend>,
) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
    let residuals_out =
        reference_mix_streams(residuals.clone(), coefficients.residual_weights.clone());
    let branch_input =
        reference_mix_streams(residuals, coefficients.branch_input_weights.clone());
    (branch_input, residuals_out)
}

fn reference_depth(
    branch_output: Tensor<Backend, 4>,
    residuals: Tensor<Backend, 4>,
    coefficients: &ManifoldHyperConnectionCoefficients<Backend>,
) -> Tensor<Backend, 4> {
    let Some(beta) = coefficients.branch_output_weights.clone() else {
        return residuals;
    };
    residuals + reference_mix_streams(branch_output, beta)
}

fn reference_passthrough(
    residuals: Tensor<Backend, 4>,
    coefficients: &ManifoldHyperConnectionCoefficients<Backend>,
) -> Tensor<Backend, 4> {
    let (branch_input, residuals_out) = reference_width(residuals, coefficients);
    reference_depth(branch_input, residuals_out, coefficients)
}

fn measure_ms<F>(warmup: usize, iterations: usize, mut func: F, device: &Device) -> f64
where
    F: FnMut(),
{
    for _ in 0..warmup {
        func();
        let _ = Backend::sync(device);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        func();
        let _ = Backend::sync(device);
    }
    start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
}

fn max_abs_diff(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f32 {
    lhs.sub(rhs).abs().max().into_scalar()
}

fn run_case(case: BenchCase, device: &Device, args: &Args) -> CaseResult {
    let config = build_config(case);
    let mhc = ManifoldHyperConnections::<Backend>::new(&config, 0, device);
    let residuals = sample_residuals(case, device);
    let coefficients = mhc.coefficients();

    let coefficient_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = mhc.coefficients();
        },
        device,
    );

    let width_reference = reference_width(residuals.clone(), &coefficients);
    let width_optimized = mhc.width_connection_with_coefficients(residuals.clone(), &coefficients);
    let width_max_abs_diff = max_abs_diff(
        width_reference.0.clone(),
        width_optimized.branch_input.clone(),
    )
    .max(max_abs_diff(
        width_reference.1.clone(),
        width_optimized.residuals_out.clone(),
    ));

    let width_baseline_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = reference_width(residuals.clone(), &coefficients);
        },
        device,
    );
    let width_optimized_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = mhc.width_connection_with_coefficients(residuals.clone(), &coefficients);
        },
        device,
    );

    let depth_baseline_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = reference_depth(
                width_reference.0.clone(),
                width_reference.1.clone(),
                &coefficients,
            );
        },
        device,
    );
    let depth_optimized_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = mhc.depth_connection_with_coefficients(
                width_optimized.branch_input.clone(),
                width_optimized.residuals_out.clone(),
                &coefficients,
            );
        },
        device,
    );

    let passthrough_reference = reference_passthrough(residuals.clone(), &coefficients);
    let passthrough_optimized = mhc.passthrough(residuals.clone());
    let passthrough_max_abs_diff =
        max_abs_diff(passthrough_reference.clone(), passthrough_optimized.clone());

    let passthrough_baseline_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let coefficients = mhc.coefficients();
            let _ = reference_passthrough(residuals.clone(), &coefficients);
        },
        device,
    );
    let passthrough_optimized_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = mhc.passthrough(residuals.clone());
        },
        device,
    );

    let passthrough_reuse_baseline_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = reference_passthrough(residuals.clone(), &coefficients);
        },
        device,
    );
    let passthrough_reuse_optimized_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let _ = mhc_passthrough_with_coefficients(
                Some(&mhc),
                residuals.clone(),
                Some(&coefficients),
            );
        },
        device,
    );

    CaseResult {
        case,
        coefficient_ms,
        width_baseline_ms,
        width_optimized_ms,
        width_speedup_x: width_baseline_ms / width_optimized_ms,
        depth_baseline_ms,
        depth_optimized_ms,
        depth_speedup_x: depth_baseline_ms / depth_optimized_ms,
        passthrough_baseline_ms,
        passthrough_optimized_ms,
        passthrough_speedup_x: passthrough_baseline_ms / passthrough_optimized_ms,
        passthrough_reuse_baseline_ms,
        passthrough_reuse_optimized_ms,
        passthrough_reuse_speedup_x: passthrough_reuse_baseline_ms / passthrough_reuse_optimized_ms,
        width_max_abs_diff,
        passthrough_max_abs_diff,
    }
}
