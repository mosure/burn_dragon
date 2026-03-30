#![cfg_attr(not(feature = "cuda"), allow(unused))]

#[cfg(feature = "cuda")]
use std::time::Instant;

#[cfg(feature = "cuda")]
use burn::tensor::backend::Backend as BackendTrait;
#[cfg(feature = "cuda")]
use burn::tensor::{Distribution, Tensor};
#[cfg(feature = "cuda")]
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use burn_dragon_kernel::kernels::sequence::rwkv8::forward::{
    tensorized_rwkv8_forward, tensorized_rwkv8_forward_direct_graph,
};
#[cfg(feature = "cuda")]
use burn_wgpu::CubeBackend;
#[cfg(feature = "cuda")]
use serde::Serialize;

#[cfg(feature = "cuda")]
type Backend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type AutodiffBackend = Autodiff<Backend>;
#[cfg(feature = "cuda")]
type Device = <Backend as BackendTrait>::Device;

#[cfg(feature = "cuda")]
struct EnvVarGuard {
    key: &'static str,
    previous: Option<String>,
}

#[cfg(feature = "cuda")]
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let previous = std::env::var(key).ok();
        unsafe { std::env::set_var(key, value) };
        Self { key, previous }
    }
}

#[cfg(feature = "cuda")]
impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match &self.previous {
            Some(previous) => unsafe { std::env::set_var(self.key, previous) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
}

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct BenchResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    direct_graph_forward_ms: f64,
    analytic_wrapper_forward_ms: f64,
    forward_speedup_x: f64,
    direct_graph_backward_ms: f64,
    analytic_wrapper_backward_ms: f64,
    backward_speedup_x: f64,
    context_max_abs: f64,
    rho_max_abs: f64,
    rho_norm_max_abs: f64,
}

#[cfg(feature = "cuda")]
#[derive(Serialize)]
struct BenchReport {
    benchmark: &'static str,
    profile: &'static str,
    results: Vec<BenchResult>,
}

#[cfg(feature = "cuda")]
const COMPACT_CASES: &[BenchCase] = &[BenchCase {
    name: "cuda_b1_h2_t16_l16_e32",
    batch: 1,
    heads: 2,
    time: 16,
    latent: 16,
    embd: 32,
}];

#[cfg(feature = "cuda")]
const FULL_CASES: &[BenchCase] = &[
    BenchCase {
        name: "cuda_b1_h4_t64_l32_e64",
        batch: 1,
        heads: 4,
        time: 64,
        latent: 32,
        embd: 64,
    },
    BenchCase {
        name: "cuda_b1_h4_t128_l64_e128",
        batch: 1,
        heads: 4,
        time: 128,
        latent: 64,
        embd: 128,
    },
];

#[cfg(feature = "cuda")]
fn main() {
    let _guard = EnvVarGuard::set("BURN_DRAGON_RWKV8_TENSORIZED_TRAIN_WRAPPER", "1");
    let device = Device::default();
    <Backend as BackendTrait>::seed(&device, 2026);

    let profile = bench_profile();
    let warmup = 1;
    let repetitions = if profile == "full" { 3 } else { 2 };
    let results = bench_cases(profile)
        .iter()
        .copied()
        .map(|case| run_case(case, &device, warmup, repetitions))
        .collect::<Vec<_>>();

    println!(
        "{}",
        serde_json::to_string_pretty(&BenchReport {
            benchmark: "burn_dragon_kernel rwkv8 cuda direct-graph vs analytical wrapper",
            profile,
            results,
        })
        .expect("serialize rwkv8 cuda bench report")
    );
}

#[cfg(not(feature = "cuda"))]
fn main() {
    panic!("rwkv8_cuda_bench requires --features cuda");
}

#[cfg(feature = "cuda")]
fn bench_profile() -> &'static str {
    match std::env::var("BURN_DRAGON_BENCH_PROFILE")
        .unwrap_or_else(|_| "compact".to_string())
        .as_str()
    {
        "full" => "full",
        _ => "compact",
    }
}

#[cfg(feature = "cuda")]
fn bench_cases(profile: &'static str) -> &'static [BenchCase] {
    if profile == "full" {
        FULL_CASES
    } else {
        COMPACT_CASES
    }
}

#[cfg(feature = "cuda")]
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
        let _ = tensorized_rwkv8_forward_direct_graph(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );
        let _ = tensorized_rwkv8_forward(query.clone(), value.clone(), None, None, decay.clone());
        let _ = Backend::sync(device);
    }

    let mut direct_graph_forward_ms = 0.0;
    let mut analytic_wrapper_forward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = tensorized_rwkv8_forward_direct_graph(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );
        let _ = Backend::sync(device);
        direct_graph_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = tensorized_rwkv8_forward(query.clone(), value.clone(), None, None, decay.clone());
        let _ = Backend::sync(device);
        analytic_wrapper_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;
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
    let weights = Tensor::<AutodiffBackend, 4>::random(
        [case.batch, case.heads, case.time, case.embd],
        Distribution::Uniform(-0.25, 0.25),
        device,
    );

    for _ in 0..warmup {
        let direct_graph_loss = (tensorized_rwkv8_forward_direct_graph(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
            * weights.clone())
        .sum();
        let _ = direct_graph_loss.backward();
        let _ = AutodiffBackend::sync(device);

        let analytic_wrapper_loss = (tensorized_rwkv8_forward(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
            * weights.clone())
        .sum();
        let _ = analytic_wrapper_loss.backward();
        let _ = AutodiffBackend::sync(device);
    }

    let mut direct_graph_backward_ms = 0.0;
    let mut analytic_wrapper_backward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let direct_graph_loss = (tensorized_rwkv8_forward_direct_graph(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
            * weights.clone())
        .sum();
        let _ = direct_graph_loss.backward();
        let _ = AutodiffBackend::sync(device);
        direct_graph_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let analytic_wrapper_loss = (tensorized_rwkv8_forward(
            query_ad.clone(),
            value_ad.clone(),
            None,
            None,
            decay_ad.clone(),
        )
        .context
            * weights.clone())
        .sum();
        let _ = analytic_wrapper_loss.backward();
        let _ = AutodiffBackend::sync(device);
        analytic_wrapper_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let direct_graph_output = tensorized_rwkv8_forward_direct_graph(
        query.clone(),
        value.clone(),
        None,
        None,
        decay.clone(),
    );
    let analytic_wrapper_output = tensorized_rwkv8_forward(query, value, None, None, decay);

    let direct_graph_forward_ms = direct_graph_forward_ms / repetitions.max(1) as f64;
    let analytic_wrapper_forward_ms = analytic_wrapper_forward_ms / repetitions.max(1) as f64;
    let direct_graph_backward_ms = direct_graph_backward_ms / repetitions.max(1) as f64;
    let analytic_wrapper_backward_ms = analytic_wrapper_backward_ms / repetitions.max(1) as f64;

    BenchResult {
        case,
        warmup,
        repetitions,
        direct_graph_forward_ms,
        analytic_wrapper_forward_ms,
        forward_speedup_x: direct_graph_forward_ms / analytic_wrapper_forward_ms.max(f64::EPSILON),
        direct_graph_backward_ms,
        analytic_wrapper_backward_ms,
        backward_speedup_x: direct_graph_backward_ms
            / analytic_wrapper_backward_ms.max(f64::EPSILON),
        context_max_abs: max_abs_4(direct_graph_output.context, analytic_wrapper_output.context),
        rho_max_abs: max_abs_4(direct_graph_output.rho, analytic_wrapper_output.rho),
        rho_norm_max_abs: max_abs_3(
            direct_graph_output.rho_norm,
            analytic_wrapper_output.rho_norm,
        ),
    }
}

#[cfg(feature = "cuda")]
fn max_abs_4(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}

#[cfg(feature = "cuda")]
fn max_abs_3(lhs: Tensor<Backend, 3>, rhs: Tensor<Backend, 3>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}
