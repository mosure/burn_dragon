#![allow(dead_code)]

use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor, TensorData, activation};
use burn_autodiff::Autodiff;
use burn_dragon_kernel::kernels::sequence::mamba2::forward::{
    CudaSsdCoreMode, Mamba2TensorizedState, tensorized_mamba2_forward_custom_backward,
    tensorized_mamba2_forward_direct_graph,
    tensorized_mamba2_forward_direct_graph_with_ssd_core_mode,
};
use burn_wgpu::{CubeBackend, WgpuRuntime};
use serde::{Deserialize, Serialize};

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackend = Autodiff<Backend>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    time: usize,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
    headdim: usize,
    ngroups: usize,
}

#[derive(Deserialize, Serialize)]
struct BenchCaseReport {
    name: String,
    batch: usize,
    time: usize,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
    headdim: usize,
    ngroups: usize,
}

impl From<BenchCase> for BenchCaseReport {
    fn from(value: BenchCase) -> Self {
        Self {
            name: value.name.to_string(),
            batch: value.batch,
            time: value.time,
            d_model: value.d_model,
            d_state: value.d_state,
            d_conv: value.d_conv,
            expand: value.expand,
            headdim: value.headdim,
            ngroups: value.ngroups,
        }
    }
}

#[derive(Deserialize, Serialize)]
struct BenchResult {
    case: BenchCaseReport,
    warmup: usize,
    repetitions: usize,
    reference_forward_ms: f64,
    tensorized_baseline_forward_ms: f64,
    tensorized_forward_ms: f64,
    accelerated_vs_baseline_forward_speedup_x: f64,
    forward_speedup_x: f64,
    tensorized_graph_backward_ms: f64,
    tensorized_custom_backward_ms: f64,
    custom_vs_graph_backward_speedup_x: f64,
    output_max_abs: f64,
    conv_state_max_abs: f64,
    ssm_state_max_abs: f64,
}

#[derive(Serialize)]
struct BenchReport {
    benchmark: &'static str,
    profile: &'static str,
    results: Vec<BenchResult>,
}

#[derive(Clone)]
struct Params {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    nheads: usize,
    norm_eps: f32,
    in_proj: Tensor<Backend, 2>,
    conv_weight: Tensor<Backend, 2>,
    conv_bias: Tensor<Backend, 1>,
    dt_bias: Tensor<Backend, 1>,
    a_log: Tensor<Backend, 1>,
    d_skip: Tensor<Backend, 1>,
    norm_weight: Tensor<Backend, 1>,
    out_proj: Tensor<Backend, 2>,
}

#[derive(Clone)]
struct ParamsAutodiff {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    headdim: usize,
    ngroups: usize,
    nheads: usize,
    norm_eps: f32,
    in_proj: Tensor<AutodiffBackend, 2>,
    conv_weight: Tensor<AutodiffBackend, 2>,
    conv_bias: Tensor<AutodiffBackend, 1>,
    dt_bias: Tensor<AutodiffBackend, 1>,
    a_log: Tensor<AutodiffBackend, 1>,
    d_skip: Tensor<AutodiffBackend, 1>,
    norm_weight: Tensor<AutodiffBackend, 1>,
    out_proj: Tensor<AutodiffBackend, 2>,
}

#[derive(Clone)]
struct State {
    conv: Tensor<Backend, 4>,
    ssm: Tensor<Backend, 4>,
}

#[derive(Clone)]
struct StateAutodiff {
    conv: Tensor<AutodiffBackend, 4>,
    ssm: Tensor<AutodiffBackend, 4>,
}

const COMPACT_CASES: &[BenchCase] = &[BenchCase {
    name: "b1_t4_dm8_ds1_dc2_e1_h4_g1",
    batch: 1,
    time: 4,
    d_model: 8,
    d_state: 1,
    d_conv: 2,
    expand: 1,
    headdim: 4,
    ngroups: 1,
}];

const FULL_CASES: &[BenchCase] = &[
    BenchCase {
        name: "b1_t64_dm256_ds16_dc4_e2_h64_g1",
        batch: 1,
        time: 64,
        d_model: 256,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        headdim: 64,
        ngroups: 1,
    },
    BenchCase {
        name: "b1_t128_dm256_ds16_dc4_e2_h64_g1",
        batch: 1,
        time: 128,
        d_model: 256,
        d_state: 16,
        d_conv: 4,
        expand: 2,
        headdim: 64,
        ngroups: 1,
    },
];

fn main() {
    let profile = bench_profile();
    if let Some(case_name) = child_case_name() {
        let device = Device::default();
        <Backend as BackendTrait>::seed(&device, 2026);
        let warmup = if profile == "full" { 1 } else { 0 };
        let repetitions = if profile == "full" { 2 } else { 1 };
        let case = bench_cases(profile)
            .iter()
            .copied()
            .find(|candidate| candidate.name == case_name)
            .unwrap_or_else(|| panic!("unknown bench case {case_name}"));
        println!(
            "{}",
            serde_json::to_string(&run_case(case, &device, warmup, repetitions))
                .expect("serialize child mamba2 bench result")
        );
        return;
    }

    let timeout = bench_case_timeout(profile);
    let results = bench_cases(profile)
        .iter()
        .copied()
        .map(|case| run_case_with_timeout(case, profile, timeout))
        .collect::<Vec<_>>();

    println!(
        "{}",
        serde_json::to_string_pretty(&BenchReport {
            benchmark: "burn_dragon_kernel mamba2 forward+backward microbench",
            profile,
            results,
        })
        .expect("serialize mamba2 bench report")
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

fn child_case_name() -> Option<String> {
    std::env::var("BURN_DRAGON_BENCH_CHILD_CASE").ok()
}

fn bench_case_timeout(profile: &'static str) -> Duration {
    let default_secs = if profile == "full" { 180 } else { 30 };
    let secs = std::env::var("BURN_DRAGON_BENCH_TIMEOUT_SECS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(default_secs);
    Duration::from_secs(secs)
}

fn run_case_with_timeout(case: BenchCase, profile: &'static str, timeout: Duration) -> BenchResult {
    let mut child = Command::new(std::env::current_exe().expect("current exe"))
        .env("BURN_DRAGON_BENCH_PROFILE", profile)
        .env("BURN_DRAGON_BENCH_CHILD_CASE", case.name)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn mamba2 bench child");
    let started = Instant::now();

    loop {
        if let Some(status) = child.try_wait().expect("poll mamba2 bench child") {
            let output = child
                .wait_with_output()
                .expect("collect mamba2 bench child output");
            if !status.success() {
                panic!(
                    "mamba2 bench child {} failed: {}",
                    case.name,
                    String::from_utf8_lossy(&output.stderr)
                );
            }
            return serde_json::from_slice::<BenchResult>(&output.stdout).unwrap_or_else(|error| {
                panic!(
                    "parse mamba2 bench child {} output failed: {error}; stdout={}",
                    case.name,
                    String::from_utf8_lossy(&output.stdout)
                )
            });
        }
        if started.elapsed() > timeout {
            let _ = child.kill();
            let output = child
                .wait_with_output()
                .expect("collect timed-out mamba2 bench child output");
            panic!(
                "mamba2 bench child {} exceeded {:?}; stderr={}",
                case.name,
                timeout,
                String::from_utf8_lossy(&output.stderr)
            );
        }
        thread::sleep(Duration::from_millis(100));
    }
}

fn run_case(case: BenchCase, device: &Device, warmup: usize, repetitions: usize) -> BenchResult {
    let hidden = Tensor::<Backend, 4>::random(
        [case.batch, 1, case.time, case.d_model],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let params = build_params(case, device);

    for _ in 0..warmup {
        let _ = mamba2_reference(hidden.clone(), &params, None);
        let _ = mamba2_tensorized_with_ssd_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedDisabled,
        );
        let _ = mamba2_tensorized_with_ssd_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedEnabled,
        );
        let _ = Backend::sync(device);
    }

    let mut reference_forward_ms = 0.0;
    let mut tensorized_baseline_forward_ms = 0.0;
    let mut tensorized_forward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = mamba2_reference(hidden.clone(), &params, None);
        let _ = Backend::sync(device);
        reference_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = mamba2_tensorized_with_ssd_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedDisabled,
        );
        let _ = Backend::sync(device);
        tensorized_baseline_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = mamba2_tensorized_with_ssd_mode(
            hidden.clone(),
            &params,
            CudaSsdCoreMode::ForcedEnabled,
        );
        let _ = Backend::sync(device);
        tensorized_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let hidden_ad = Tensor::<AutodiffBackend, 4>::random(
        [case.batch, 1, case.time, case.d_model],
        Distribution::Uniform(-0.5, 0.5),
        device,
    )
    .require_grad();
    let params_ad = build_params_autodiff(case, device);

    for _ in 0..warmup {
        let tensorized_graph_loss = mamba2_tensorized_autodiff_graph(hidden_ad.clone(), &params_ad)
            .0
            .sum();
        let _ = tensorized_graph_loss.backward();
        let _ = AutodiffBackend::sync(device);

        let tensorized_custom_loss =
            mamba2_tensorized_autodiff_custom(hidden_ad.clone(), &params_ad)
                .0
                .sum();
        let _ = tensorized_custom_loss.backward();
        let _ = AutodiffBackend::sync(device);
    }

    let mut tensorized_graph_backward_ms = 0.0;
    let mut tensorized_custom_backward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let tensorized_graph_loss = mamba2_tensorized_autodiff_graph(hidden_ad.clone(), &params_ad)
            .0
            .sum();
        let _ = tensorized_graph_loss.backward();
        let _ = AutodiffBackend::sync(device);
        tensorized_graph_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let tensorized_custom_loss =
            mamba2_tensorized_autodiff_custom(hidden_ad.clone(), &params_ad)
                .0
                .sum();
        let _ = tensorized_custom_loss.backward();
        let _ = AutodiffBackend::sync(device);
        tensorized_custom_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let (reference_output, reference_state) = mamba2_reference(hidden.clone(), &params, None);
    let (tensorized_output, tensorized_state) =
        mamba2_tensorized_with_ssd_mode(hidden, &params, CudaSsdCoreMode::ForcedEnabled);

    BenchResult {
        case: case.into(),
        warmup,
        repetitions,
        reference_forward_ms: reference_forward_ms / repetitions.max(1) as f64,
        tensorized_baseline_forward_ms: tensorized_baseline_forward_ms / repetitions.max(1) as f64,
        tensorized_forward_ms: tensorized_forward_ms / repetitions.max(1) as f64,
        accelerated_vs_baseline_forward_speedup_x: tensorized_baseline_forward_ms
            / tensorized_forward_ms.max(f64::EPSILON),
        forward_speedup_x: reference_forward_ms / tensorized_forward_ms.max(f64::EPSILON),
        tensorized_graph_backward_ms: tensorized_graph_backward_ms / repetitions.max(1) as f64,
        tensorized_custom_backward_ms: tensorized_custom_backward_ms / repetitions.max(1) as f64,
        custom_vs_graph_backward_speedup_x: tensorized_graph_backward_ms
            / tensorized_custom_backward_ms.max(f64::EPSILON),
        output_max_abs: max_abs_4(reference_output, tensorized_output),
        conv_state_max_abs: max_abs_4(reference_state.conv, tensorized_state.conv),
        ssm_state_max_abs: max_abs_4(reference_state.ssm, tensorized_state.ssm),
    }
}

fn build_params(case: BenchCase, device: &Device) -> Params {
    let d_inner = case.d_model * case.expand;
    let nheads = d_inner / case.headdim;
    let conv_dim = d_inner + 2 * case.ngroups * case.d_state;
    Params {
        d_model: case.d_model,
        d_inner,
        d_state: case.d_state,
        d_conv: case.d_conv,
        headdim: case.headdim,
        ngroups: case.ngroups,
        nheads,
        norm_eps: 1.0e-5,
        in_proj: Tensor::<Backend, 2>::random(
            [
                case.d_model,
                2 * d_inner + 2 * case.ngroups * case.d_state + nheads,
            ],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        conv_weight: Tensor::<Backend, 2>::random(
            [conv_dim, case.d_conv],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        conv_bias: Tensor::<Backend, 1>::zeros([conv_dim], device),
        dt_bias: Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![0.01; nheads], [nheads]),
            device,
        ),
        a_log: Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            device,
        ),
        d_skip: Tensor::<Backend, 1>::ones([nheads], device),
        norm_weight: Tensor::<Backend, 1>::ones([d_inner], device),
        out_proj: Tensor::<Backend, 2>::random(
            [d_inner, case.d_model],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
    }
}

fn build_params_autodiff(case: BenchCase, device: &Device) -> ParamsAutodiff {
    let d_inner = case.d_model * case.expand;
    let nheads = d_inner / case.headdim;
    let conv_dim = d_inner + 2 * case.ngroups * case.d_state;
    ParamsAutodiff {
        d_model: case.d_model,
        d_inner,
        d_state: case.d_state,
        d_conv: case.d_conv,
        headdim: case.headdim,
        ngroups: case.ngroups,
        nheads,
        norm_eps: 1.0e-5,
        in_proj: Tensor::<AutodiffBackend, 2>::random(
            [
                case.d_model,
                2 * d_inner + 2 * case.ngroups * case.d_state + nheads,
            ],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        conv_weight: Tensor::<AutodiffBackend, 2>::random(
            [conv_dim, case.d_conv],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        conv_bias: Tensor::<AutodiffBackend, 1>::zeros([conv_dim], device).require_grad(),
        dt_bias: Tensor::<AutodiffBackend, 1>::from_data(
            TensorData::new(vec![0.01; nheads], [nheads]),
            device,
        )
        .require_grad(),
        a_log: Tensor::<AutodiffBackend, 1>::from_data(
            TensorData::new(vec![1.0f32.ln(); nheads], [nheads]),
            device,
        )
        .require_grad(),
        d_skip: Tensor::<AutodiffBackend, 1>::ones([nheads], device).require_grad(),
        norm_weight: Tensor::<AutodiffBackend, 1>::ones([d_inner], device).require_grad(),
        out_proj: Tensor::<AutodiffBackend, 2>::random(
            [d_inner, case.d_model],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
    }
}

fn mamba2_tensorized(
    hidden_states: Tensor<Backend, 4>,
    params: &Params,
) -> (Tensor<Backend, 4>, State) {
    mamba2_tensorized_with_ssd_mode(hidden_states, params, CudaSsdCoreMode::ForcedEnabled)
}

fn mamba2_tensorized_with_ssd_mode(
    hidden_states: Tensor<Backend, 4>,
    params: &Params,
    ssd_core_mode: CudaSsdCoreMode,
) -> (Tensor<Backend, 4>, State) {
    let batch = hidden_states.shape().dims::<4>()[0];
    let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
    let device = hidden_states.device();
    let output = tensorized_mamba2_forward_direct_graph_with_ssd_core_mode(
        hidden_states,
        params.d_inner,
        params.d_state,
        params.d_conv,
        params.headdim,
        params.ngroups,
        params.in_proj.clone(),
        params.conv_weight.clone(),
        Some(params.conv_bias.clone()),
        params.dt_bias.clone(),
        params.a_log.clone(),
        params.d_skip.clone(),
        params.norm_weight.clone(),
        params.norm_eps,
        params.out_proj.clone(),
        Some(Mamba2TensorizedState {
            conv: Tensor::<Backend, 4>::zeros([batch, 1, conv_dim, params.d_conv], &device),
            ssm: Tensor::<Backend, 4>::zeros(
                [batch, params.nheads, params.headdim, params.d_state],
                &device,
            ),
        }),
        ssd_core_mode,
    );
    (
        output.context,
        State {
            conv: output.state.conv,
            ssm: output.state.ssm,
        },
    )
}

fn mamba2_tensorized_autodiff(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    mamba2_tensorized_autodiff_custom(hidden_states, params)
}

fn mamba2_tensorized_autodiff_graph(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    let batch = hidden_states.shape().dims::<4>()[0];
    let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
    let device = hidden_states.device();
    let output = tensorized_mamba2_forward_direct_graph(
        hidden_states,
        params.d_inner,
        params.d_state,
        params.d_conv,
        params.headdim,
        params.ngroups,
        params.in_proj.clone(),
        params.conv_weight.clone(),
        Some(params.conv_bias.clone()),
        params.dt_bias.clone(),
        params.a_log.clone(),
        params.d_skip.clone(),
        params.norm_weight.clone(),
        params.norm_eps,
        params.out_proj.clone(),
        Some(Mamba2TensorizedState {
            conv: Tensor::<AutodiffBackend, 4>::zeros([batch, 1, conv_dim, params.d_conv], &device),
            ssm: Tensor::<AutodiffBackend, 4>::zeros(
                [batch, params.nheads, params.headdim, params.d_state],
                &device,
            ),
        }),
    );
    (
        output.context,
        StateAutodiff {
            conv: output.state.conv,
            ssm: output.state.ssm,
        },
    )
}

fn mamba2_tensorized_autodiff_custom(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    let batch = hidden_states.shape().dims::<4>()[0];
    let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
    let device = hidden_states.device();
    let output = tensorized_mamba2_forward_custom_backward(
        hidden_states,
        params.d_inner,
        params.d_state,
        params.d_conv,
        params.headdim,
        params.ngroups,
        params.in_proj.clone(),
        params.conv_weight.clone(),
        Some(params.conv_bias.clone()),
        params.dt_bias.clone(),
        params.a_log.clone(),
        params.d_skip.clone(),
        params.norm_weight.clone(),
        params.norm_eps,
        params.out_proj.clone(),
        Some(Mamba2TensorizedState {
            conv: Tensor::<AutodiffBackend, 4>::zeros([batch, 1, conv_dim, params.d_conv], &device),
            ssm: Tensor::<AutodiffBackend, 4>::zeros(
                [batch, params.nheads, params.headdim, params.d_state],
                &device,
            ),
        }),
    )
    .expect("custom backward path available");
    (
        output.context,
        StateAutodiff {
            conv: output.state.conv,
            ssm: output.state.ssm,
        },
    )
}

fn mamba2_reference(
    hidden_states: Tensor<Backend, 4>,
    params: &Params,
    state: Option<State>,
) -> (Tensor<Backend, 4>, State) {
    let [batch, _, time, _] = hidden_states.shape().dims::<4>();
    let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
    let device = hidden_states.device();

    let mut conv_state = state
        .as_ref()
        .filter(|state| state.conv.shape().dims::<4>() == [batch, 1, conv_dim, params.d_conv])
        .map(|state| state.conv.clone())
        .unwrap_or_else(|| {
            Tensor::<Backend, 4>::zeros([batch, 1, conv_dim, params.d_conv], &device)
        });
    let mut ssm_state = state
        .as_ref()
        .filter(|state| {
            state.ssm.shape().dims::<4>() == [batch, params.nheads, params.headdim, params.d_state]
        })
        .map(|state| state.ssm.clone())
        .unwrap_or_else(|| {
            Tensor::<Backend, 4>::zeros(
                [batch, params.nheads, params.headdim, params.d_state],
                &device,
            )
        });

    let zxbcdt = hidden_states
        .clone()
        .reshape([batch * time, params.d_model])
        .matmul(params.in_proj.clone())
        .reshape([
            batch,
            time,
            2 * params.d_inner + 2 * params.ngroups * params.d_state + params.nheads,
        ]);
    let z = zxbcdt
        .clone()
        .slice_dim(2, 0..params.d_inner)
        .reshape([batch, time, params.d_inner]);
    let xbc = zxbcdt
        .clone()
        .slice_dim(2, params.d_inner..(params.d_inner + conv_dim))
        .reshape([batch, time, conv_dim]);
    let dt = zxbcdt
        .slice_dim(
            2,
            (params.d_inner + conv_dim)
                ..(2 * params.d_inner + 2 * params.ngroups * params.d_state + params.nheads),
        )
        .reshape([batch, time, params.nheads]);

    let history = Tensor::cat(
        vec![
            conv_state.clone(),
            xbc.swap_dims(1, 2).reshape([batch, 1, conv_dim, time]),
        ],
        3,
    );
    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let mut xbc_t = Tensor::<Backend, 3>::zeros([batch, 1, conv_dim], &device);
        for tap in 0..params.d_conv {
            let window = history
                .clone()
                .slice_dim(3, tap + step + 1..tap + step + 2)
                .reshape([batch, 1, conv_dim])
                .mul(
                    params
                        .conv_weight
                        .clone()
                        .slice_dim(1, tap..tap + 1)
                        .reshape([1, 1, conv_dim]),
                );
            xbc_t = xbc_t + window;
        }
        xbc_t = silu(xbc_t + params.conv_bias.clone().reshape([1, 1, conv_dim]));
        let x_t = xbc_t.clone().slice_dim(2, 0..params.d_inner).reshape([
            batch,
            params.nheads,
            params.headdim,
        ]);
        let b_t = repeat_groups_to_heads_runtime(
            xbc_t
                .clone()
                .slice_dim(
                    2,
                    params.d_inner..(params.d_inner + params.ngroups * params.d_state),
                )
                .reshape([batch, params.ngroups, params.d_state]),
            params.nheads,
        );
        let c_t = repeat_groups_to_heads_runtime(
            xbc_t
                .clone()
                .slice_dim(
                    2,
                    (params.d_inner + params.ngroups * params.d_state)..conv_dim,
                )
                .reshape([batch, params.ngroups, params.d_state]),
            params.nheads,
        );
        let z_t =
            z.clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, params.nheads, params.headdim]);
        let dt_t = dt
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, params.nheads]);
        let (y_t, next_ssm_state) =
            mamba2_reference_step(x_t, z_t, dt_t, b_t, c_t, ssm_state, params);
        outputs.push(
            rmsnorm_gated_runtime(
                y_t.reshape([batch, 1, params.d_inner]),
                z.clone().slice_dim(1, step..step + 1),
                params.norm_weight.clone(),
                params.norm_eps,
            )
            .reshape([batch, params.d_inner])
            .matmul(params.out_proj.clone())
            .reshape([batch, 1, 1, params.d_model]),
        );
        ssm_state = next_ssm_state;
    }
    conv_state = history.slice_dim(3, time..time + params.d_conv);

    (
        Tensor::cat(outputs, 2),
        State {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

fn mamba2_reference_autodiff(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
    state: Option<StateAutodiff>,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    let [batch, _, time, _] = hidden_states.shape().dims::<4>();
    let conv_dim = params.d_inner + 2 * params.ngroups * params.d_state;
    let device = hidden_states.device();

    let mut conv_state = state
        .as_ref()
        .filter(|state| state.conv.shape().dims::<4>() == [batch, 1, conv_dim, params.d_conv])
        .map(|state| state.conv.clone())
        .unwrap_or_else(|| {
            Tensor::<AutodiffBackend, 4>::zeros([batch, 1, conv_dim, params.d_conv], &device)
        });
    let mut ssm_state = state
        .as_ref()
        .filter(|state| {
            state.ssm.shape().dims::<4>() == [batch, params.nheads, params.headdim, params.d_state]
        })
        .map(|state| state.ssm.clone())
        .unwrap_or_else(|| {
            Tensor::<AutodiffBackend, 4>::zeros(
                [batch, params.nheads, params.headdim, params.d_state],
                &device,
            )
        });

    let zxbcdt = hidden_states
        .clone()
        .reshape([batch * time, params.d_model])
        .matmul(params.in_proj.clone())
        .reshape([
            batch,
            time,
            2 * params.d_inner + 2 * params.ngroups * params.d_state + params.nheads,
        ]);
    let z = zxbcdt
        .clone()
        .slice_dim(2, 0..params.d_inner)
        .reshape([batch, time, params.d_inner]);
    let xbc = zxbcdt
        .clone()
        .slice_dim(2, params.d_inner..(params.d_inner + conv_dim))
        .reshape([batch, time, conv_dim]);
    let dt = zxbcdt
        .slice_dim(
            2,
            (params.d_inner + conv_dim)
                ..(2 * params.d_inner + 2 * params.ngroups * params.d_state + params.nheads),
        )
        .reshape([batch, time, params.nheads]);

    let history = Tensor::cat(
        vec![
            conv_state.clone(),
            xbc.swap_dims(1, 2).reshape([batch, 1, conv_dim, time]),
        ],
        3,
    );
    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let mut xbc_t = Tensor::<AutodiffBackend, 3>::zeros([batch, 1, conv_dim], &device);
        for tap in 0..params.d_conv {
            let window = history
                .clone()
                .slice_dim(3, tap + step + 1..tap + step + 2)
                .reshape([batch, 1, conv_dim])
                .mul(
                    params
                        .conv_weight
                        .clone()
                        .slice_dim(1, tap..tap + 1)
                        .reshape([1, 1, conv_dim]),
                );
            xbc_t = xbc_t + window;
        }
        xbc_t = silu(xbc_t + params.conv_bias.clone().reshape([1, 1, conv_dim]));
        let x_t = xbc_t.clone().slice_dim(2, 0..params.d_inner).reshape([
            batch,
            params.nheads,
            params.headdim,
        ]);
        let b_t = repeat_groups_to_heads_autodiff(
            xbc_t
                .clone()
                .slice_dim(
                    2,
                    params.d_inner..(params.d_inner + params.ngroups * params.d_state),
                )
                .reshape([batch, params.ngroups, params.d_state]),
            params.nheads,
        );
        let c_t = repeat_groups_to_heads_autodiff(
            xbc_t
                .clone()
                .slice_dim(
                    2,
                    (params.d_inner + params.ngroups * params.d_state)..conv_dim,
                )
                .reshape([batch, params.ngroups, params.d_state]),
            params.nheads,
        );
        let z_t =
            z.clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, params.nheads, params.headdim]);
        let dt_t = dt
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, params.nheads]);
        let (y_t, next_ssm_state) =
            mamba2_reference_step_autodiff(x_t, z_t, dt_t, b_t, c_t, ssm_state, params);
        outputs.push(
            rmsnorm_gated_autodiff(
                y_t.reshape([batch, 1, params.d_inner]),
                z.clone().slice_dim(1, step..step + 1),
                params.norm_weight.clone(),
                params.norm_eps,
            )
            .reshape([batch, params.d_inner])
            .matmul(params.out_proj.clone())
            .reshape([batch, 1, 1, params.d_model]),
        );
        ssm_state = next_ssm_state;
    }
    conv_state = history.slice_dim(3, time..time + params.d_conv);

    (
        Tensor::cat(outputs, 2),
        StateAutodiff {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

fn mamba2_reference_step(
    x_t: Tensor<Backend, 3>,
    z_t: Tensor<Backend, 3>,
    dt_t: Tensor<Backend, 2>,
    b_t: Tensor<Backend, 3>,
    c_t: Tensor<Backend, 3>,
    ssm_state: Tensor<Backend, 4>,
    params: &Params,
) -> (Tensor<Backend, 3>, Tensor<Backend, 4>) {
    let [batch, nheads, headdim] = x_t.shape().dims::<3>();
    let _ = z_t;
    let dt = activation::softplus(dt_t + params.dt_bias.clone().reshape([1, nheads]), 1.0);
    let a = params.a_log.clone().exp().neg().reshape([1, nheads, 1, 1]);
    let next_ssm = ssm_state * (dt.clone().reshape([batch, nheads, 1, 1]) * a).exp()
        + dt.clone().reshape([batch, nheads, 1, 1])
            * b_t.reshape([batch, nheads, 1, params.d_state])
            * x_t.clone().reshape([batch, nheads, headdim, 1]);
    let y = (next_ssm.clone() * c_t.reshape([batch, nheads, 1, params.d_state]))
        .sum_dim(3)
        .reshape([batch, nheads, headdim])
        + params.d_skip.clone().reshape([1, nheads, 1]) * x_t;
    (y, next_ssm)
}

fn mamba2_reference_step_autodiff(
    x_t: Tensor<AutodiffBackend, 3>,
    z_t: Tensor<AutodiffBackend, 3>,
    dt_t: Tensor<AutodiffBackend, 2>,
    b_t: Tensor<AutodiffBackend, 3>,
    c_t: Tensor<AutodiffBackend, 3>,
    ssm_state: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 3>, Tensor<AutodiffBackend, 4>) {
    let [batch, nheads, headdim] = x_t.shape().dims::<3>();
    let _ = z_t;
    let dt = activation::softplus(dt_t + params.dt_bias.clone().reshape([1, nheads]), 1.0);
    let a = params.a_log.clone().exp().neg().reshape([1, nheads, 1, 1]);
    let next_ssm = ssm_state * (dt.clone().reshape([batch, nheads, 1, 1]) * a).exp()
        + dt.clone().reshape([batch, nheads, 1, 1])
            * b_t.reshape([batch, nheads, 1, params.d_state])
            * x_t.clone().reshape([batch, nheads, headdim, 1]);
    let y = (next_ssm.clone() * c_t.reshape([batch, nheads, 1, params.d_state]))
        .sum_dim(3)
        .reshape([batch, nheads, headdim])
        + params.d_skip.clone().reshape([1, nheads, 1]) * x_t;
    (y, next_ssm)
}

fn repeat_groups_to_heads_runtime(
    grouped: Tensor<Backend, 3>,
    nheads: usize,
) -> Tensor<Backend, 3> {
    let [batch, ngroups, d_state] = grouped.shape().dims::<3>();
    grouped
        .reshape([batch, ngroups, 1, d_state])
        .repeat_dim(2, nheads / ngroups)
        .reshape([batch, nheads, d_state])
}

fn repeat_groups_to_heads_autodiff(
    grouped: Tensor<AutodiffBackend, 3>,
    nheads: usize,
) -> Tensor<AutodiffBackend, 3> {
    let [batch, ngroups, d_state] = grouped.shape().dims::<3>();
    grouped
        .reshape([batch, ngroups, 1, d_state])
        .repeat_dim(2, nheads / ngroups)
        .reshape([batch, nheads, d_state])
}

fn rmsnorm_gated_runtime(
    y: Tensor<Backend, 3>,
    z: Tensor<Backend, 3>,
    weight: Tensor<Backend, 1>,
    eps: f32,
) -> Tensor<Backend, 3> {
    let width = weight.shape().dims::<1>()[0];
    let [batch, time, _] = y.shape().dims::<3>();
    let rms = y
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt()
        .reshape([batch, time, 1]);
    (y / rms) * weight.reshape([1, 1, width]) * silu(z)
}

fn rmsnorm_gated_autodiff(
    y: Tensor<AutodiffBackend, 3>,
    z: Tensor<AutodiffBackend, 3>,
    weight: Tensor<AutodiffBackend, 1>,
    eps: f32,
) -> Tensor<AutodiffBackend, 3> {
    let width = weight.shape().dims::<1>()[0];
    let [batch, time, _] = y.shape().dims::<3>();
    let rms = y
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt()
        .reshape([batch, time, 1]);
    (y / rms) * weight.reshape([1, 1, width]) * silu(z)
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

fn max_abs_4(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}
