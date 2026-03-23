use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor, TensorData, activation};
use burn_autodiff::Autodiff;
use burn_dragon_kernel::kernels::sequence::mamba::selective_scan_forward::{
    MambaTensorizedState, tensorized_mamba_forward,
};
use burn_wgpu::{CubeBackend, WgpuRuntime};
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type AutodiffBackend = Autodiff<Backend>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone)]
struct Params {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<Backend, 2>,
    conv_weight: Tensor<Backend, 2>,
    conv_bias: Tensor<Backend, 1>,
    x_proj: Tensor<Backend, 2>,
    dt_proj_weight: Tensor<Backend, 2>,
    dt_proj_bias: Tensor<Backend, 1>,
    a_log: Tensor<Backend, 2>,
    d_skip: Tensor<Backend, 1>,
    out_proj: Tensor<Backend, 2>,
}

#[derive(Clone)]
struct State {
    conv: Tensor<Backend, 4>,
    ssm: Tensor<Backend, 4>,
}

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    time: usize,
    d_model: usize,
    d_state: usize,
    d_conv: usize,
    expand: usize,
}

#[derive(Serialize)]
struct BenchResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    reference_forward_ms: f64,
    tensorized_forward_ms: f64,
    forward_speedup_x: f64,
    reference_backward_ms: f64,
    tensorized_backward_ms: f64,
    backward_speedup_x: f64,
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

const COMPACT_CASES: &[BenchCase] = &[BenchCase {
    name: "b1_t32_dm64_ds8_dc4_e2",
    batch: 1,
    time: 32,
    d_model: 64,
    d_state: 8,
    d_conv: 4,
    expand: 2,
}];

const FULL_CASES: &[BenchCase] = &[
    BenchCase {
        name: "b1_t64_dm256_ds16_dc4_e2",
        batch: 1,
        time: 64,
        d_model: 256,
        d_state: 16,
        d_conv: 4,
        expand: 2,
    },
    BenchCase {
        name: "b1_t128_dm256_ds16_dc4_e2",
        batch: 1,
        time: 128,
        d_model: 256,
        d_state: 16,
        d_conv: 4,
        expand: 2,
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
            benchmark: "burn_dragon_kernel mamba forward+backward microbench",
            profile,
            results,
        })
        .expect("serialize mamba bench report")
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
    let hidden = Tensor::<Backend, 4>::random(
        [case.batch, 1, case.time, case.d_model],
        Distribution::Uniform(-0.5, 0.5),
        device,
    );
    let params = build_params(case, device);

    for _ in 0..warmup {
        let _ = mamba_reference(hidden.clone(), &params, None);
        let _ = mamba_tensorized(hidden.clone(), &params);
        let _ = Backend::sync(device);
    }

    let mut reference_forward_ms = 0.0;
    let mut tensorized_forward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = mamba_reference(hidden.clone(), &params, None);
        let _ = Backend::sync(device);
        reference_forward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = Backend::sync(device);
        let started = Instant::now();
        let _ = mamba_tensorized(hidden.clone(), &params);
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
        let reference_loss = mamba_reference_autodiff(hidden_ad.clone(), &params_ad, None)
            .0
            .sum();
        let _ = reference_loss.backward();
        let _ = AutodiffBackend::sync(device);

        let tensorized_loss = mamba_tensorized_autodiff(hidden_ad.clone(), &params_ad)
            .0
            .sum();
        let _ = tensorized_loss.backward();
        let _ = AutodiffBackend::sync(device);
    }

    let mut reference_backward_ms = 0.0;
    let mut tensorized_backward_ms = 0.0;
    for _ in 0..repetitions {
        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let reference_loss = mamba_reference_autodiff(hidden_ad.clone(), &params_ad, None)
            .0
            .sum();
        let _ = reference_loss.backward();
        let _ = AutodiffBackend::sync(device);
        reference_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;

        let _ = AutodiffBackend::sync(device);
        let started = Instant::now();
        let tensorized_loss = mamba_tensorized_autodiff(hidden_ad.clone(), &params_ad)
            .0
            .sum();
        let _ = tensorized_loss.backward();
        let _ = AutodiffBackend::sync(device);
        tensorized_backward_ms += started.elapsed().as_secs_f64() * 1_000.0;
    }

    let (full_output, full_state) = mamba_reference(hidden.clone(), &params, None);
    let (tensorized_output, tensorized_state) = mamba_tensorized(hidden, &params);

    BenchResult {
        case,
        warmup,
        repetitions,
        reference_forward_ms: reference_forward_ms / repetitions.max(1) as f64,
        tensorized_forward_ms: tensorized_forward_ms / repetitions.max(1) as f64,
        forward_speedup_x: reference_forward_ms / tensorized_forward_ms.max(f64::EPSILON),
        reference_backward_ms: reference_backward_ms / repetitions.max(1) as f64,
        tensorized_backward_ms: tensorized_backward_ms / repetitions.max(1) as f64,
        backward_speedup_x: reference_backward_ms / tensorized_backward_ms.max(f64::EPSILON),
        output_max_abs: max_abs_4(full_output, tensorized_output),
        conv_state_max_abs: max_abs_4(full_state.conv, tensorized_state.conv),
        ssm_state_max_abs: max_abs_4(full_state.ssm, tensorized_state.ssm),
    }
}

fn build_params(case: BenchCase, device: &Device) -> Params {
    let d_inner = case.d_model * case.expand;
    let dt_rank = case.d_model.div_ceil(16);
    let dt_bias = 1.0e-2_f32 + (-(-1.0e-2_f32).exp_m1()).ln();
    let a_values = (0..d_inner)
        .flat_map(|_| (1..=case.d_state).map(|value| (value as f32).ln()))
        .collect::<Vec<_>>();

    Params {
        d_model: case.d_model,
        d_inner,
        d_state: case.d_state,
        d_conv: case.d_conv,
        dt_rank,
        in_proj: Tensor::<Backend, 2>::random(
            [case.d_model, d_inner * 2],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        conv_weight: Tensor::<Backend, 2>::random(
            [d_inner, case.d_conv],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        conv_bias: Tensor::<Backend, 1>::zeros([d_inner], device),
        x_proj: Tensor::<Backend, 2>::random(
            [d_inner, dt_rank + case.d_state * 2],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        dt_proj_weight: Tensor::<Backend, 2>::random(
            [dt_rank, d_inner],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
        dt_proj_bias: Tensor::<Backend, 1>::from_data(
            TensorData::new(vec![dt_bias; d_inner], [d_inner]),
            device,
        ),
        a_log: Tensor::<Backend, 2>::from_data(
            TensorData::new(a_values, [d_inner, case.d_state]),
            device,
        ),
        d_skip: Tensor::<Backend, 1>::ones([d_inner], device),
        out_proj: Tensor::<Backend, 2>::random(
            [d_inner, case.d_model],
            Distribution::Uniform(-0.05, 0.05),
            device,
        ),
    }
}

fn build_params_autodiff(case: BenchCase, device: &Device) -> ParamsAutodiff {
    let d_inner = case.d_model * case.expand;
    let dt_rank = case.d_model.div_ceil(16);
    let dt_bias = 1.0e-2_f32 + (-(-1.0e-2_f32).exp_m1()).ln();
    let a_values = (0..d_inner)
        .flat_map(|_| (1..=case.d_state).map(|value| (value as f32).ln()))
        .collect::<Vec<_>>();

    ParamsAutodiff {
        d_model: case.d_model,
        d_inner,
        d_state: case.d_state,
        d_conv: case.d_conv,
        dt_rank,
        in_proj: Tensor::<AutodiffBackend, 2>::random(
            [case.d_model, d_inner * 2],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        conv_weight: Tensor::<AutodiffBackend, 2>::random(
            [d_inner, case.d_conv],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        conv_bias: Tensor::<AutodiffBackend, 1>::zeros([d_inner], device).require_grad(),
        x_proj: Tensor::<AutodiffBackend, 2>::random(
            [d_inner, dt_rank + case.d_state * 2],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        dt_proj_weight: Tensor::<AutodiffBackend, 2>::random(
            [dt_rank, d_inner],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
        dt_proj_bias: Tensor::<AutodiffBackend, 1>::from_data(
            TensorData::new(vec![dt_bias; d_inner], [d_inner]),
            device,
        )
        .require_grad(),
        a_log: Tensor::<AutodiffBackend, 2>::from_data(
            TensorData::new(a_values, [d_inner, case.d_state]),
            device,
        )
        .require_grad(),
        d_skip: Tensor::<AutodiffBackend, 1>::ones([d_inner], device).require_grad(),
        out_proj: Tensor::<AutodiffBackend, 2>::random(
            [d_inner, case.d_model],
            Distribution::Uniform(-0.05, 0.05),
            device,
        )
        .require_grad(),
    }
}

#[derive(Clone)]
struct ParamsAutodiff {
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    d_conv: usize,
    dt_rank: usize,
    in_proj: Tensor<AutodiffBackend, 2>,
    conv_weight: Tensor<AutodiffBackend, 2>,
    conv_bias: Tensor<AutodiffBackend, 1>,
    x_proj: Tensor<AutodiffBackend, 2>,
    dt_proj_weight: Tensor<AutodiffBackend, 2>,
    dt_proj_bias: Tensor<AutodiffBackend, 1>,
    a_log: Tensor<AutodiffBackend, 2>,
    d_skip: Tensor<AutodiffBackend, 1>,
    out_proj: Tensor<AutodiffBackend, 2>,
}

#[derive(Clone)]
struct StateAutodiff {
    conv: Tensor<AutodiffBackend, 4>,
    ssm: Tensor<AutodiffBackend, 4>,
}

fn mamba_reference(
    hidden_states: Tensor<Backend, 4>,
    params: &Params,
    state: Option<State>,
) -> (Tensor<Backend, 4>, State) {
    let [batch, views, time, dim] = hidden_states.shape().dims::<4>();
    assert_eq!(views, 1);
    assert_eq!(dim, params.d_model);
    let device = hidden_states.device();

    let mut conv_state = state
        .as_ref()
        .filter(|state| state.conv.shape().dims::<4>() == [batch, 1, params.d_inner, params.d_conv])
        .map(|state| state.conv.clone())
        .unwrap_or_else(|| {
            Tensor::<Backend, 4>::zeros([batch, 1, params.d_inner, params.d_conv], &device)
        });
    let mut ssm_state = state
        .as_ref()
        .filter(|state| state.ssm.shape().dims::<4>() == [batch, 1, params.d_inner, params.d_state])
        .map(|state| state.ssm.clone())
        .unwrap_or_else(|| {
            Tensor::<Backend, 4>::zeros([batch, 1, params.d_inner, params.d_state], &device)
        });

    let xz = hidden_states
        .clone()
        .reshape([batch * time, params.d_model])
        .matmul(params.in_proj.clone())
        .reshape([batch, time, params.d_inner * 2]);
    let x = xz
        .clone()
        .slice_dim(2, 0..params.d_inner)
        .swap_dims(1, 2)
        .reshape([batch, 1, params.d_inner, time]);
    let z = xz
        .slice_dim(2, params.d_inner..(params.d_inner * 2))
        .swap_dims(1, 2)
        .reshape([batch, 1, params.d_inner, time]);

    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let x_t = x
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, params.d_inner]);
        let z_t = z
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, params.d_inner]);
        let (u_t, next_conv_state) = mamba_depthwise_conv_step_reference(
            x_t,
            conv_state,
            params.conv_weight.clone(),
            params.conv_bias.clone(),
        );
        conv_state = next_conv_state;
        let (y_t, next_ssm_state) =
            mamba_selective_scan_step_reference(u_t, z_t, ssm_state, params);
        ssm_state = next_ssm_state;
        outputs.push(
            y_t.reshape([batch, params.d_inner])
                .matmul(params.out_proj.clone())
                .reshape([batch, 1, 1, params.d_model]),
        );
    }

    (
        Tensor::cat(outputs, 2),
        State {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

fn mamba_reference_autodiff(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
    state: Option<StateAutodiff>,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    let [batch, views, time, dim] = hidden_states.shape().dims::<4>();
    assert_eq!(views, 1);
    assert_eq!(dim, params.d_model);
    let device = hidden_states.device();

    let mut conv_state = state
        .as_ref()
        .filter(|state| state.conv.shape().dims::<4>() == [batch, 1, params.d_inner, params.d_conv])
        .map(|state| state.conv.clone())
        .unwrap_or_else(|| {
            Tensor::<AutodiffBackend, 4>::zeros([batch, 1, params.d_inner, params.d_conv], &device)
        });
    let mut ssm_state = state
        .as_ref()
        .filter(|state| state.ssm.shape().dims::<4>() == [batch, 1, params.d_inner, params.d_state])
        .map(|state| state.ssm.clone())
        .unwrap_or_else(|| {
            Tensor::<AutodiffBackend, 4>::zeros([batch, 1, params.d_inner, params.d_state], &device)
        });

    let xz = hidden_states
        .clone()
        .reshape([batch * time, params.d_model])
        .matmul(params.in_proj.clone())
        .reshape([batch, time, params.d_inner * 2]);
    let x = xz
        .clone()
        .slice_dim(2, 0..params.d_inner)
        .swap_dims(1, 2)
        .reshape([batch, 1, params.d_inner, time]);
    let z = xz
        .slice_dim(2, params.d_inner..(params.d_inner * 2))
        .swap_dims(1, 2)
        .reshape([batch, 1, params.d_inner, time]);

    let mut outputs = Vec::with_capacity(time);
    for step in 0..time {
        let x_t = x
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, params.d_inner]);
        let z_t = z
            .clone()
            .slice_dim(3, step..step + 1)
            .reshape([batch, 1, params.d_inner]);
        let (u_t, next_conv_state) = mamba_depthwise_conv_step_reference_autodiff(
            x_t,
            conv_state,
            params.conv_weight.clone(),
            Some(params.conv_bias.clone()),
        );
        let (y_t, next_ssm_state) =
            mamba_selective_scan_step_reference_autodiff(u_t, z_t, ssm_state, params);
        outputs.push(y_t.unsqueeze_dim::<4>(2));
        conv_state = next_conv_state;
        ssm_state = next_ssm_state;
    }

    (
        Tensor::cat(outputs, 2),
        StateAutodiff {
            conv: conv_state,
            ssm: ssm_state,
        },
    )
}

fn mamba_tensorized_autodiff(
    hidden_states: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 4>, StateAutodiff) {
    let output = tensorized_mamba_forward(
        hidden_states,
        params.d_inner,
        params.d_state,
        params.d_conv,
        params.dt_rank,
        params.in_proj.clone(),
        params.conv_weight.clone(),
        Some(params.conv_bias.clone()),
        params.x_proj.clone(),
        params.dt_proj_weight.clone(),
        params.dt_proj_bias.clone(),
        params.a_log.clone(),
        params.d_skip.clone(),
        params.out_proj.clone(),
        None,
    );
    (
        output.context,
        StateAutodiff {
            conv: output.state.conv,
            ssm: output.state.ssm,
        },
    )
}

fn mamba_depthwise_conv_step_reference_autodiff(
    x_t: Tensor<AutodiffBackend, 3>,
    conv_state: Tensor<AutodiffBackend, 4>,
    conv_weight: Tensor<AutodiffBackend, 2>,
    conv_bias: Option<Tensor<AutodiffBackend, 1>>,
) -> (Tensor<AutodiffBackend, 3>, Tensor<AutodiffBackend, 4>) {
    let [batch, views, d_inner] = x_t.shape().dims::<3>();
    let d_conv = conv_state.shape().dims::<4>()[3];
    let device = x_t.device();
    let conv_tail = if d_conv > 1 {
        conv_state.clone().slice_dim(3, 1..d_conv)
    } else {
        Tensor::<AutodiffBackend, 4>::zeros([batch, views, d_inner, 0], &device)
    };
    let next_conv_state = Tensor::cat(vec![conv_tail, x_t.clone().unsqueeze_dim::<4>(3)], 3);
    let mut u_t = (next_conv_state.clone() * conv_weight.reshape([1, 1, d_inner, d_conv]))
        .sum_dim(3)
        .reshape([batch, views, d_inner]);
    if let Some(bias) = conv_bias {
        u_t = u_t + bias.reshape([1, 1, d_inner]);
    }
    (silu(u_t), next_conv_state)
}

fn mamba_selective_scan_step_reference_autodiff(
    u_t: Tensor<AutodiffBackend, 3>,
    z_t: Tensor<AutodiffBackend, 3>,
    ssm_state: Tensor<AutodiffBackend, 4>,
    params: &ParamsAutodiff,
) -> (Tensor<AutodiffBackend, 3>, Tensor<AutodiffBackend, 4>) {
    let [batch, views, d_inner] = u_t.shape().dims::<3>();
    let a = params
        .a_log
        .clone()
        .exp()
        .neg()
        .reshape([1, 1, params.d_inner, params.d_state]);
    let d_skip = params.d_skip.clone().reshape([1, 1, params.d_inner]);

    let x_db = u_t
        .clone()
        .reshape([batch, d_inner])
        .matmul(params.x_proj.clone())
        .reshape([batch, params.dt_rank + params.d_state * 2]);
    let dt = activation::softplus(
        x_db.clone()
            .slice_dim(1, 0..params.dt_rank)
            .matmul(params.dt_proj_weight.clone())
            .reshape([batch, views, params.d_inner])
            + params.dt_proj_bias.clone().reshape([1, 1, params.d_inner]),
        1.0,
    );
    let b_t = x_db
        .clone()
        .slice_dim(1, params.dt_rank..(params.dt_rank + params.d_state))
        .reshape([batch, views, params.d_state]);
    let c_t = x_db
        .slice_dim(
            1,
            (params.dt_rank + params.d_state)..(params.dt_rank + params.d_state * 2),
        )
        .reshape([batch, views, params.d_state]);

    let delta_a = (dt.clone().unsqueeze_dim::<4>(3) * a).exp();
    let delta_b_u = dt.clone().unsqueeze_dim::<4>(3)
        * b_t.reshape([batch, views, 1, params.d_state])
        * u_t.clone().reshape([batch, views, params.d_inner, 1]);
    let next_ssm_state = delta_a * ssm_state + delta_b_u;
    let y_t = (next_ssm_state.clone() * c_t.reshape([batch, views, 1, params.d_state]))
        .sum_dim(3)
        .reshape([batch, views, params.d_inner])
        + d_skip * u_t;
    (y_t * silu(z_t), next_ssm_state)
}

fn mamba_tensorized(
    hidden_states: Tensor<Backend, 4>,
    params: &Params,
) -> (Tensor<Backend, 4>, State) {
    let output = tensorized_mamba_forward(
        hidden_states,
        params.d_inner,
        params.d_state,
        params.d_conv,
        params.dt_rank,
        params.in_proj.clone(),
        params.conv_weight.clone(),
        Some(params.conv_bias.clone()),
        params.x_proj.clone(),
        params.dt_proj_weight.clone(),
        params.dt_proj_bias.clone(),
        params.a_log.clone(),
        params.d_skip.clone(),
        params.out_proj.clone(),
        None::<MambaTensorizedState<Backend>>,
    );
    (
        output.context,
        State {
            conv: output.state.conv,
            ssm: output.state.ssm,
        },
    )
}

fn mamba_depthwise_conv_step_reference(
    x_t: Tensor<Backend, 3>,
    conv_state: Tensor<Backend, 4>,
    conv_weight: Tensor<Backend, 2>,
    conv_bias: Tensor<Backend, 1>,
) -> (Tensor<Backend, 3>, Tensor<Backend, 4>) {
    let [batch, views, d_inner] = x_t.shape().dims::<3>();
    let d_conv = conv_state.shape().dims::<4>()[3];
    let device = x_t.device();
    let conv_tail = if d_conv > 1 {
        conv_state.clone().slice_dim(3, 1..d_conv)
    } else {
        Tensor::<Backend, 4>::zeros([batch, views, d_inner, 0], &device)
    };
    let next_conv_state = Tensor::cat(vec![conv_tail, x_t.clone().unsqueeze_dim::<4>(3)], 3);
    let u_t = (next_conv_state.clone() * conv_weight.reshape([1, 1, d_inner, d_conv]))
        .sum_dim(3)
        .reshape([batch, views, d_inner])
        + conv_bias.reshape([1, 1, d_inner]);
    (silu(u_t), next_conv_state)
}

fn mamba_selective_scan_step_reference(
    u_t: Tensor<Backend, 3>,
    z_t: Tensor<Backend, 3>,
    ssm_state: Tensor<Backend, 4>,
    params: &Params,
) -> (Tensor<Backend, 3>, Tensor<Backend, 4>) {
    let [batch, views, d_inner] = u_t.shape().dims::<3>();
    let a = params
        .a_log
        .clone()
        .exp()
        .neg()
        .reshape([1, 1, params.d_inner, params.d_state]);
    let d_skip = params.d_skip.clone().reshape([1, 1, params.d_inner]);
    let x_db = u_t
        .clone()
        .reshape([batch, d_inner])
        .matmul(params.x_proj.clone())
        .reshape([batch, params.dt_rank + params.d_state * 2]);
    let dt = activation::softplus(
        x_db.clone()
            .slice_dim(1, 0..params.dt_rank)
            .matmul(params.dt_proj_weight.clone())
            .reshape([batch, views, params.d_inner])
            + params.dt_proj_bias.clone().reshape([1, 1, params.d_inner]),
        1.0,
    );
    let b_t = x_db
        .clone()
        .slice_dim(1, params.dt_rank..(params.dt_rank + params.d_state))
        .reshape([batch, views, params.d_state]);
    let c_t = x_db
        .slice_dim(
            1,
            (params.dt_rank + params.d_state)..(params.dt_rank + params.d_state * 2),
        )
        .reshape([batch, views, params.d_state]);

    let d_a = (dt.clone().unsqueeze_dim::<4>(3) * a).exp();
    let d_b = dt.clone().unsqueeze_dim::<4>(3) * b_t.clone().unsqueeze_dim::<4>(2);
    let next_ssm_state = ssm_state * d_a + u_t.clone().unsqueeze_dim::<4>(3) * d_b;
    let y_t = (next_ssm_state.clone() * c_t.unsqueeze_dim::<4>(2))
        .sum_dim(3)
        .reshape([batch, views, params.d_inner])
        + d_skip * u_t;
    (y_t * silu(z_t), next_ssm_state)
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

fn max_abs_4(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f64 {
    lhs.sub(rhs).abs().max().into_scalar() as f64
}
