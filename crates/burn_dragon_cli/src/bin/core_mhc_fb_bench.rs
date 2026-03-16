use std::time::Instant;

use burn::module::Module;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_dragon::core::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use clap::Parser;

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InnerBackend>;
type Device = <TrainBackend as BackendTrait>::Device;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long, default_value_t = 1)]
    warmup: usize,
    #[arg(long, default_value_t = 3)]
    iterations: usize,
}

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    num_streams: usize,
    num_views: usize,
    batch: usize,
    time: usize,
    dim: usize,
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
];

fn main() {
    let args = Args::parse();
    let device = Device::default();
    init_runtime(&device);

    println!("# burn_dragon core mHC forward+backward benchmark");
    println!();
    println!("- warmup: {}", args.warmup);
    println!("- iterations: {}", args.iterations);
    println!();

    for case in CASES.iter().copied() {
        let (baseline_ms, optimized_ms, loss_abs_diff) = run_case(case, &device, &args);
        println!("## {}", case.name);
        println!(
            "- baseline: {:.3} ms, optimized: {:.3} ms, speedup: {:.2}x, loss_abs_diff: {:.3e}",
            baseline_ms,
            optimized_ms,
            baseline_ms / optimized_ms,
            loss_abs_diff,
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

fn build_config(case: BenchCase) -> ManifoldHyperConnectionsConfig {
    ManifoldHyperConnectionsConfig {
        enabled: true,
        num_streams: case.num_streams,
        num_views: case.num_views,
        last_layers: None,
        coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
        mhc_iters: 10,
        mhc_tau: 0.05,
        add_branch_out_to_residual: true,
        dropout: 0.0,
    }
}

fn sample_residuals(case: BenchCase, device: &Device) -> Tensor<TrainBackend, 4> {
    <TrainBackend as BackendTrait>::seed(device, 2_026);
    Tensor::<TrainBackend, 4>::random(
        [case.batch, case.num_streams, case.time, case.dim],
        Distribution::Normal(0.0, 1.0),
        device,
    )
}

fn reference_mix_streams(
    residuals: Tensor<TrainBackend, 4>,
    weights: Tensor<TrainBackend, 2>,
) -> Tensor<TrainBackend, 4> {
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

fn reference_passthrough(
    residuals: Tensor<TrainBackend, 4>,
    coefficients: &ManifoldHyperConnectionCoefficients<TrainBackend>,
) -> Tensor<TrainBackend, 4> {
    let residuals_out =
        reference_mix_streams(residuals.clone(), coefficients.residual_weights.clone());
    let branch_input = reference_mix_streams(residuals, coefficients.branch_input_weights.clone());
    let Some(beta) = coefficients.branch_output_weights.clone() else {
        return residuals_out;
    };
    residuals_out + reference_mix_streams(branch_input, beta)
}

fn measure_ms<F>(warmup: usize, iterations: usize, mut func: F, device: &Device) -> f64
where
    F: FnMut() -> f32,
{
    for _ in 0..warmup {
        let _ = func();
        let _ = TrainBackend::sync(device);
    }
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = func();
        let _ = TrainBackend::sync(device);
    }
    start.elapsed().as_secs_f64() * 1_000.0 / iterations as f64
}

fn run_case(case: BenchCase, device: &Device, args: &Args) -> (f64, f64, f32) {
    let config = build_config(case);
    let baseline = ManifoldHyperConnections::<TrainBackend>::new(&config, 0, device);
    let optimized = baseline.clone().load_record(baseline.clone().into_record());
    let residuals = sample_residuals(case, device);

    let coefficients = baseline.coefficients();
    let baseline_loss = reference_passthrough(residuals.clone(), &coefficients).mean();
    let optimized_loss = optimized.passthrough(residuals.clone()).mean();
    let baseline_value = baseline_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("baseline loss")[0];
    let optimized_value = optimized_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("optimized loss")[0];

    let baseline_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let coefficients = baseline.coefficients();
            let loss = reference_passthrough(residuals.clone(), &coefficients).mean();
            let value = loss
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("baseline iter loss")[0];
            let _ = loss.backward();
            value
        },
        device,
    );

    let optimized_ms = measure_ms(
        args.warmup,
        args.iterations,
        || {
            let loss = optimized.passthrough(residuals.clone()).mean();
            let value = loss
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("optimized iter loss")[0];
            let _ = loss.backward();
            value
        },
        device,
    );

    (
        baseline_ms,
        optimized_ms,
        (baseline_value - optimized_value).abs(),
    )
}
