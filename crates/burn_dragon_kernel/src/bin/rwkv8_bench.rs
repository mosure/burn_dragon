use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_dragon_kernel::kernels::sequence::rwkv8::forward::tensorized_rwkv8_forward;
use burn_wgpu::{CubeBackend, WgpuRuntime};
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackend = Autodiff<Backend>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
}

#[derive(Serialize)]
struct BenchResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    host_loop_forward_ms: f64,
    tensorized_forward_ms: f64,
    forward_speedup_x: f64,
    host_loop_backward_ms: f64,
    tensorized_backward_ms: f64,
    backward_speedup_x: f64,
    context_max_abs: f64,
    rho_max_abs: f64,
    rho_norm_max_abs: f64,
}

#[derive(Serialize)]
struct BenchReport {
    benchmark: &'static str,
    profile: &'static str,
    results: Vec<BenchResult>,
}

const COMPACT_CASES: &[BenchCase] = &[BenchCase {
    name: "b1_h1_t8_l16_e16",
    batch: 1,
    heads: 1,
    time: 8,
    latent: 16,
    embd: 16,
}];

const FULL_CASES: &[BenchCase] = &[
    BenchCase {
        name: "b1_h8_t64_l64_e128",
        batch: 1,
        heads: 8,
        time: 64,
        latent: 64,
        embd: 128,
    },
    BenchCase {
        name: "b1_h8_t64_l128_e128",
        batch: 1,
        heads: 8,
        time: 64,
        latent: 128,
        embd: 128,
    },
];

fn main() {
    let device = Device::default();
    <Backend as BackendTrait>::seed(&device, 2026);

    let profile = bench_profile();
    let warmup = if profile == "full" { 1 } else { 0 };
    let repetitions = if profile == "full" { 2 } else { 1 };
    let results = bench_cases(profile)
        .iter()
        .copied()
        .map(|case| run_case(case, &device, warmup, repetitions))
        .collect::<Vec<_>>();

    println!(
        "{}",
        serde_json::to_string_pretty(&BenchReport {
            benchmark: "burn_dragon_kernel rwkv8 forward+backward microbench",
            profile,
            results,
        })
        .expect("serialize rwkv8 bench report")
    );
}

fn bench_profile() -> &'static str {
    match std::env::var("BURN_DRAGON_BENCH_PROFILE")
        .unwrap_or_else(|_| "compact".to_string())
        .as_str()
    {
        "full" => "full",
        _ => "compact",
    }
}

fn bench_cases(profile: &'static str) -> &'static [BenchCase] {
    if profile == "full" {
        FULL_CASES
    } else {
        COMPACT_CASES
    }
}

fn run_case(case: BenchCase, device: &Device, warmup: usize, repetitions: usize) -> BenchResult {
    let query = Tensor::<Backend, 4>::random(
        [case.batch, case.heads, case.time, case.latent],
        Distribution::Uniform(0.0, 1.0),
        device,
    );
    let value = Tensor::<Backend, 4>::random(
        [case.batch, 1, case.time, case.embd],
        Distribution::Uniform(0.0, 1.0),
        device,
    );
    let decay = Tensor::<Backend, 3>::random(
        [1, case.heads, case.latent],
        Distribution::Uniform(0.75, 0.99),
        device,
    );

    for _ in 0..warmup {
        let _ = rwkv8_host_loop_reference(query.clone(), value.clone(), None, None, decay.clone());
        let _ = tensorized_rwkv8_forward(query.clone(), value.clone(), None, None, decay.clone());
        let _ = Backend::sync(device);
    }

    let mut host_loop_forward_ms = 0.0;
    let mut tensorized_forward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = rwkv8_host_loop_reference(query.clone(), value.clone(), None, None, decay.clone());
        let _ = Backend::sync(device);
        host_loop_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = tensorized_rwkv8_forward(query.clone(), value.clone(), None, None, decay.clone());
        let _ = Backend::sync(device);
        tensorized_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let query_ad = Tensor::<AutodiffBackend, 4>::random(
        [case.batch, case.heads, case.time, case.latent],
        Distribution::Uniform(0.0, 1.0),
        device,
    )
    .require_grad();
    let value_ad = Tensor::<AutodiffBackend, 4>::random(
        [case.batch, 1, case.time, case.embd],
        Distribution::Uniform(0.0, 1.0),
        device,
    )
    .require_grad();
    let decay_ad = Tensor::<AutodiffBackend, 3>::random(
        [1, case.heads, case.latent],
        Distribution::Uniform(0.75, 0.99),
        device,
    )
    .require_grad();

    for _ in 0..warmup {
        let host_loss = rwkv8_host_loop_reference(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .0
        .sum();
        let _ = host_loss.backward();
        let _ = AutodiffBackend::sync(device);

        let tensorized_loss = tensorized_rwkv8_forward(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
        .sum();
        let _ = tensorized_loss.backward();
        let _ = AutodiffBackend::sync(device);
    }

    let mut host_loop_backward_ms = 0.0;
    let mut tensorized_backward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let host_loss = rwkv8_host_loop_reference(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .0
        .sum();
        let _ = host_loss.backward();
        let _ = AutodiffBackend::sync(device);
        host_loop_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let tensorized_loss = tensorized_rwkv8_forward(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
        .sum();
        let _ = tensorized_loss.backward();
        let _ = AutodiffBackend::sync(device);
        tensorized_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let host_output =
        rwkv8_host_loop_reference(query.clone(), value.clone(), None, None, decay.clone());
    let tensorized_output = tensorized_rwkv8_forward(query, value, None, None, decay);

    BenchResult {
        case,
        warmup,
        repetitions,
        host_loop_forward_ms: host_loop_forward_ms / repetitions.max(1) as f64,
        tensorized_forward_ms: tensorized_forward_ms / repetitions.max(1) as f64,
        forward_speedup_x: host_loop_forward_ms / tensorized_forward_ms.max(f64::EPSILON),
        host_loop_backward_ms: host_loop_backward_ms / repetitions.max(1) as f64,
        tensorized_backward_ms: tensorized_backward_ms / repetitions.max(1) as f64,
        backward_speedup_x: host_loop_backward_ms / tensorized_backward_ms.max(f64::EPSILON),
        context_max_abs: max_abs_4(host_output.0, tensorized_output.context),
        rho_max_abs: max_abs_4(host_output.1, tensorized_output.rho),
        rho_norm_max_abs: max_abs_3(host_output.2, tensorized_output.rho_norm),
    }
}

fn max_abs_4(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}

fn max_abs_3(lhs: Tensor<Backend, 3>, rhs: Tensor<Backend, 3>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}

fn rwkv8_host_loop_reference<B: BackendTrait>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();

    let mut rho = rho_state
        .filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd])
        .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device));
    let mut rho_norm = rho_norm_state
        .filter(|state| state.shape().dims::<3>() == [batch, heads, latent])
        .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, heads, latent], &device));

    let mut outputs = Vec::with_capacity(time);
    for t in 0..time {
        let q_t = query
            .clone()
            .slice_dim(2, t..t + 1)
            .reshape([batch, heads, latent]);
        let q_weights = q_t.clone().div(
            q_t.clone()
                .sum_dim(2)
                .add_scalar(1.0e-6)
                .reshape([batch, heads, 1]),
        );
        let value_t = value
            .clone()
            .slice_dim(2, t..t + 1)
            .repeat_dim(1, heads)
            .reshape([batch, heads, n_embd]);
        let context_t = (rho.clone().div(
            rho_norm
                .clone()
                .add_scalar(1.0e-6)
                .reshape([batch, heads, latent, 1]),
        ) * q_weights.reshape([batch, heads, latent, 1]))
        .sum_dim(2)
        .reshape([batch, heads, 1, n_embd]);
        outputs.push(context_t);

        rho = rho.mul(decay.clone().reshape([1, heads, latent, 1])).add(
            q_t.clone().reshape([batch, heads, latent, 1])
                * value_t.reshape([batch, heads, 1, n_embd]),
        );
        rho_norm = rho_norm.mul(decay.clone()).add(q_t);
    }

    (Tensor::cat(outputs, 2), rho, rho_norm)
}
