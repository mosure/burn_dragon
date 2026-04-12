use std::sync::Once;
use std::time::Instant;

use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Tensor, TensorData};
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_dragon_kernel::kernels::sequence::mamba3::forward::{
    Mamba3TensorizedState, tensorized_mamba3_forward_custom_backward,
    tensorized_mamba3_forward_direct_graph,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, graphics};
use serde::Serialize;

type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;
type WgpuDevice = <WgpuAutodiffBackend as BackendTrait>::Device;

#[cfg(feature = "cuda")]
type CudaBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaAutodiffBackend = Autodiff<CudaBackend>;
#[cfg(feature = "cuda")]
type CudaDevice = <CudaAutodiffBackend as BackendTrait>::Device;

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    time: usize,
    d_model: usize,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
}

#[derive(Clone)]
struct RawInputs {
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    hidden: TensorData,
    in_proj: TensorData,
    dt_bias: TensorData,
    b_bias: TensorData,
    c_bias: TensorData,
    b_norm_weight: TensorData,
    c_norm_weight: TensorData,
    d_skip: TensorData,
    out_proj: TensorData,
    state_ssm: TensorData,
    state_angle: TensorData,
    state_k: TensorData,
    state_v: TensorData,
    output_weight: TensorData,
}

#[derive(Clone)]
struct Inputs<B: BackendTrait> {
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    hidden: Tensor<B, 4>,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Mamba3TensorizedState<B>,
    output_weight: Tensor<B, 4>,
}

#[derive(Clone, Copy, Serialize)]
struct TimedMetrics {
    forward_ms: f64,
    backward_ms: f64,
    tokens_per_s: f64,
}

#[derive(Clone, Copy, Serialize)]
struct ErrorMetrics {
    max_abs: f64,
    mean_abs: f64,
    rmse: f64,
}

#[derive(Clone, Serialize)]
struct BackendBenchResult {
    backend: &'static str,
    direct_graph: TimedMetrics,
    custom_backward: TimedMetrics,
    forward_speedup_x: f64,
    backward_speedup_x: f64,
    output_error: ErrorMetrics,
    ssm_state_error: ErrorMetrics,
    angle_state_error: ErrorMetrics,
    k_state_error: ErrorMetrics,
    v_state_error: ErrorMetrics,
    hidden_grad_error: ErrorMetrics,
    in_proj_grad_error: ErrorMetrics,
    out_proj_grad_error: ErrorMetrics,
}

#[derive(Clone, Serialize)]
struct Report {
    benchmark: &'static str,
    case: BenchCase,
    chunk_size: usize,
    warmup: usize,
    repetitions: usize,
    wgpu: BackendBenchResult,
    #[cfg(feature = "cuda")]
    cuda: BackendBenchResult,
}

const COMPACT_CASE: BenchCase = BenchCase {
    name: "b1_t32_dm128_di256_ds16_h64_g4_r8",
    batch: 1,
    time: 32,
    d_model: 128,
    d_inner: 256,
    d_state: 16,
    headdim: 64,
    ngroups: 4,
    num_rope_angles: 8,
};

const FULL_CASE: BenchCase = BenchCase {
    name: "b2_t64_dm128_di256_ds16_h64_g4_r8",
    batch: 2,
    time: 64,
    d_model: 128,
    d_inner: 256,
    d_state: 16,
    headdim: 64,
    ngroups: 4,
    num_rope_angles: 8,
};

fn main() {
    let profile =
        std::env::var("BURN_DRAGON_BENCH_PROFILE").unwrap_or_else(|_| "compact".to_string());
    let chunk_size = std::env::var("BURN_DRAGON_MAMBA3_BENCH_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32);
    let (case, warmup, repetitions) = if profile.eq_ignore_ascii_case("full") {
        (FULL_CASE, 2usize, 5usize)
    } else {
        (COMPACT_CASE, 1usize, 3usize)
    };

    let raw_inputs = sample_raw_inputs(case);

    let wgpu_device = WgpuDevice::default();
    init_wgpu_runtime(&wgpu_device);
    <WgpuAutodiffBackend as BackendTrait>::seed(&wgpu_device, 20260411);
    let wgpu_inputs = materialize_inputs::<WgpuAutodiffBackend>(&raw_inputs, &wgpu_device);
    let wgpu =
        run_backend_case::<WgpuAutodiffBackend>(&wgpu_inputs, &wgpu_device, warmup, repetitions);

    let report = Report {
        benchmark: "burn_dragon_kernel mamba3 training microbench",
        case,
        chunk_size,
        warmup,
        repetitions,
        wgpu,
        #[cfg(feature = "cuda")]
        cuda: {
            let cuda_device = CudaDevice::default();
            <CudaAutodiffBackend as BackendTrait>::seed(&cuda_device, 20260411);
            let cuda_inputs = materialize_inputs::<CudaAutodiffBackend>(&raw_inputs, &cuda_device);
            run_backend_case::<CudaAutodiffBackend>(&cuda_inputs, &cuda_device, warmup, repetitions)
        },
    };

    println!(
        "{}",
        serde_json::to_string_pretty(&report).expect("serialize mamba3 bench report")
    );
}

fn init_wgpu_runtime(device: &WgpuDevice) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn sample_raw_inputs(case: BenchCase) -> RawInputs {
    let nheads = case.d_inner / case.headdim;
    let in_proj_dim =
        2 * case.d_inner + 2 * case.ngroups * case.d_state + 3 * nheads + case.num_rope_angles;

    RawInputs {
        d_inner: case.d_inner,
        d_state: case.d_state,
        headdim: case.headdim,
        ngroups: case.ngroups,
        num_rope_angles: case.num_rope_angles,
        hidden: shaped_data(
            [case.batch, 1, case.time, case.d_model],
            case.batch * case.time * case.d_model,
            257,
            -0.5,
        ),
        in_proj: shaped_data(
            [case.d_model, in_proj_dim],
            case.d_model * in_proj_dim,
            263,
            -0.45,
        ),
        dt_bias: shaped_data([nheads], nheads, 269, -0.35),
        b_bias: shaped_data([nheads, case.d_state], nheads * case.d_state, 271, -0.4),
        c_bias: shaped_data([nheads, case.d_state], nheads * case.d_state, 277, -0.42),
        b_norm_weight: shaped_data([case.d_state], case.d_state, 281, 0.9),
        c_norm_weight: shaped_data([case.d_state], case.d_state, 283, 0.85),
        d_skip: shaped_data([nheads], nheads, 293, 0.75),
        out_proj: shaped_data(
            [case.d_inner, case.d_model],
            case.d_inner * case.d_model,
            307,
            -0.45,
        ),
        state_ssm: shaped_data(
            [case.batch, nheads, case.headdim, case.d_state],
            case.batch * nheads * case.headdim * case.d_state,
            311,
            -0.25,
        ),
        state_angle: shaped_data(
            [case.batch, nheads, case.num_rope_angles],
            case.batch * nheads * case.num_rope_angles,
            313,
            -0.15,
        ),
        state_k: shaped_data(
            [case.batch, nheads, case.d_state],
            case.batch * nheads * case.d_state,
            317,
            -0.2,
        ),
        state_v: shaped_data(
            [case.batch, nheads, case.headdim],
            case.batch * nheads * case.headdim,
            331,
            -0.25,
        ),
        output_weight: shaped_data(
            [case.batch, 1, case.time, case.d_model],
            case.batch * case.time * case.d_model,
            337,
            -0.35,
        ),
    }
}

fn shaped_data<const D: usize>(
    shape: [usize; D],
    len: usize,
    modulus: usize,
    offset: f32,
) -> TensorData {
    TensorData::new(
        (0..len)
            .map(|idx| ((idx % modulus) as f32) / modulus as f32 + offset)
            .collect::<Vec<_>>(),
        shape,
    )
}

fn materialize_inputs<B: BackendTrait>(raw: &RawInputs, device: &B::Device) -> Inputs<B> {
    Inputs {
        d_inner: raw.d_inner,
        d_state: raw.d_state,
        headdim: raw.headdim,
        ngroups: raw.ngroups,
        num_rope_angles: raw.num_rope_angles,
        hidden: Tensor::<B, 4>::from_data(raw.hidden.clone(), device),
        in_proj: Tensor::<B, 2>::from_data(raw.in_proj.clone(), device),
        dt_bias: Tensor::<B, 1>::from_data(raw.dt_bias.clone(), device),
        b_bias: Tensor::<B, 2>::from_data(raw.b_bias.clone(), device),
        c_bias: Tensor::<B, 2>::from_data(raw.c_bias.clone(), device),
        b_norm_weight: Tensor::<B, 1>::from_data(raw.b_norm_weight.clone(), device),
        c_norm_weight: Tensor::<B, 1>::from_data(raw.c_norm_weight.clone(), device),
        d_skip: Tensor::<B, 1>::from_data(raw.d_skip.clone(), device),
        out_proj: Tensor::<B, 2>::from_data(raw.out_proj.clone(), device),
        state: Mamba3TensorizedState {
            ssm: Tensor::<B, 4>::from_data(raw.state_ssm.clone(), device),
            angle: Tensor::<B, 3>::from_data(raw.state_angle.clone(), device),
            k: Tensor::<B, 3>::from_data(raw.state_k.clone(), device),
            v: Tensor::<B, 3>::from_data(raw.state_v.clone(), device),
        },
        output_weight: Tensor::<B, 4>::from_data(raw.output_weight.clone(), device),
    }
}

fn run_backend_case<B: AutodiffBackend>(
    inputs: &Inputs<B>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
) -> BackendBenchResult {
    let direct_graph = timed_run::<B>(inputs, device, warmup, repetitions, false);
    let custom_backward = timed_run::<B>(inputs, device, warmup, repetitions, true);
    let (
        output_error,
        ssm_state_error,
        angle_state_error,
        k_state_error,
        v_state_error,
        hidden_grad_error,
        in_proj_grad_error,
        out_proj_grad_error,
    ) = accuracy_case::<B>(inputs, device);

    BackendBenchResult {
        backend: backend_name::<B>(),
        forward_speedup_x: direct_graph.forward_ms / custom_backward.forward_ms,
        backward_speedup_x: direct_graph.backward_ms / custom_backward.backward_ms,
        direct_graph,
        custom_backward,
        output_error,
        ssm_state_error,
        angle_state_error,
        k_state_error,
        v_state_error,
        hidden_grad_error,
        in_proj_grad_error,
        out_proj_grad_error,
    }
}

fn backend_name<B: BackendTrait>() -> &'static str {
    let name = std::any::type_name::<B>();
    if name.contains("cubecl_wgpu") {
        "wgpu"
    } else if name.contains("cubecl::cuda") || name.contains("CudaRuntime") {
        "cuda"
    } else {
        name
    }
}

fn timed_run<B: AutodiffBackend>(
    inputs: &Inputs<B>,
    device: &B::Device,
    warmup: usize,
    repetitions: usize,
    custom_backward: bool,
) -> TimedMetrics {
    let chunk_size = std::env::var("BURN_DRAGON_MAMBA3_BENCH_CHUNK_SIZE")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(32);
    let mut forward_total_ms = 0.0;
    let mut backward_total_ms = 0.0;
    let total_iters = warmup + repetitions;
    let [batch, _, time, _] = inputs.hidden.shape().dims::<4>();
    let tokens = (batch * time) as f64;

    for step in 0..total_iters {
        let hidden = inputs.hidden.clone().require_grad();
        let in_proj = inputs.in_proj.clone().require_grad();
        let dt_bias = inputs.dt_bias.clone().require_grad();
        let b_bias = inputs.b_bias.clone().require_grad();
        let c_bias = inputs.c_bias.clone().require_grad();
        let b_norm_weight = inputs.b_norm_weight.clone().require_grad();
        let c_norm_weight = inputs.c_norm_weight.clone().require_grad();
        let d_skip = inputs.d_skip.clone().require_grad();
        let out_proj = inputs.out_proj.clone().require_grad();

        let start_forward = Instant::now();
        let output = if custom_backward {
            tensorized_mamba3_forward_custom_backward(
                hidden.clone(),
                inputs.d_inner,
                inputs.d_state,
                inputs.headdim,
                inputs.ngroups,
                inputs.num_rope_angles,
                1.0e-5,
                1.0e-4,
                chunk_size,
                in_proj.clone(),
                dt_bias.clone(),
                b_bias.clone(),
                c_bias.clone(),
                b_norm_weight.clone(),
                c_norm_weight.clone(),
                d_skip.clone(),
                out_proj.clone(),
                Some(inputs.state.clone()),
            )
        } else {
            tensorized_mamba3_forward_direct_graph(
                hidden.clone(),
                inputs.d_inner,
                inputs.d_state,
                inputs.headdim,
                inputs.ngroups,
                inputs.num_rope_angles,
                1.0e-5,
                1.0e-4,
                chunk_size,
                in_proj.clone(),
                dt_bias.clone(),
                b_bias.clone(),
                c_bias.clone(),
                b_norm_weight.clone(),
                c_norm_weight.clone(),
                d_skip.clone(),
                out_proj.clone(),
                Some(inputs.state.clone()),
            )
        };
        let loss = (output.context * inputs.output_weight.clone()).sum();
        let _ = B::sync(device);
        let forward_ms = start_forward.elapsed().as_secs_f64() * 1_000.0;

        let start_backward = Instant::now();
        let _ = loss.backward();
        let _ = B::sync(device);
        let backward_ms = start_backward.elapsed().as_secs_f64() * 1_000.0;

        if step >= warmup {
            forward_total_ms += forward_ms;
            backward_total_ms += backward_ms;
        }
    }

    let repetitions_f64 = repetitions as f64;
    let total_ms = forward_total_ms + backward_total_ms;
    TimedMetrics {
        forward_ms: forward_total_ms / repetitions_f64,
        backward_ms: backward_total_ms / repetitions_f64,
        tokens_per_s: repetitions_f64 * tokens / (total_ms / 1_000.0),
    }
}

#[allow(clippy::type_complexity)]
fn accuracy_case<B: AutodiffBackend>(
    inputs: &Inputs<B>,
    device: &B::Device,
) -> (
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
) {
    let direct_hidden = inputs.hidden.clone().require_grad();
    let direct_in_proj = inputs.in_proj.clone().require_grad();
    let direct_out_proj = inputs.out_proj.clone().require_grad();
    let direct = tensorized_mamba3_forward_direct_graph(
        direct_hidden.clone(),
        inputs.d_inner,
        inputs.d_state,
        inputs.headdim,
        inputs.ngroups,
        inputs.num_rope_angles,
        1.0e-5,
        1.0e-4,
        32,
        direct_in_proj.clone(),
        inputs.dt_bias.clone(),
        inputs.b_bias.clone(),
        inputs.c_bias.clone(),
        inputs.b_norm_weight.clone(),
        inputs.c_norm_weight.clone(),
        inputs.d_skip.clone(),
        direct_out_proj.clone(),
        Some(inputs.state.clone()),
    );

    let custom_hidden = inputs.hidden.clone().require_grad();
    let custom_in_proj = inputs.in_proj.clone().require_grad();
    let custom_out_proj = inputs.out_proj.clone().require_grad();
    let custom = tensorized_mamba3_forward_custom_backward(
        custom_hidden.clone(),
        inputs.d_inner,
        inputs.d_state,
        inputs.headdim,
        inputs.ngroups,
        inputs.num_rope_angles,
        1.0e-5,
        1.0e-4,
        32,
        custom_in_proj.clone(),
        inputs.dt_bias.clone(),
        inputs.b_bias.clone(),
        inputs.c_bias.clone(),
        inputs.b_norm_weight.clone(),
        inputs.c_norm_weight.clone(),
        inputs.d_skip.clone(),
        custom_out_proj.clone(),
        Some(inputs.state.clone()),
    );

    let direct_grads = (direct.context.clone() * inputs.output_weight.clone())
        .sum()
        .backward();
    let custom_grads = (custom.context.clone() * inputs.output_weight.clone())
        .sum()
        .backward();
    let _ = B::sync(device);

    (
        tensor_error(direct.context, custom.context),
        tensor_error(direct.state.ssm, custom.state.ssm),
        tensor_error(direct.state.angle, custom.state.angle),
        tensor_error(direct.state.k, custom.state.k),
        tensor_error(direct.state.v, custom.state.v),
        tensor_error(
            direct_hidden
                .grad(&direct_grads)
                .expect("direct hidden grad"),
            custom_hidden
                .grad(&custom_grads)
                .expect("custom hidden grad"),
        ),
        tensor_error(
            direct_in_proj
                .grad(&direct_grads)
                .expect("direct in_proj grad"),
            custom_in_proj
                .grad(&custom_grads)
                .expect("custom in_proj grad"),
        ),
        tensor_error(
            direct_out_proj
                .grad(&direct_grads)
                .expect("direct out_proj grad"),
            custom_out_proj
                .grad(&custom_grads)
                .expect("custom out_proj grad"),
        ),
    )
}

fn tensor_error<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
) -> ErrorMetrics {
    let lhs_data = lhs.into_data().to_vec::<f32>().expect("lhs data");
    let rhs_data = rhs.into_data().to_vec::<f32>().expect("rhs data");
    let len = lhs_data.len().max(1) as f64;
    let mut max_abs = 0.0f64;
    let mut sum_abs = 0.0f64;
    let mut sum_sq = 0.0f64;

    for (lhs, rhs) in lhs_data.iter().zip(rhs_data.iter()) {
        let diff = (*lhs - *rhs) as f64;
        let abs = diff.abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        sum_sq += diff * diff;
    }

    ErrorMetrics {
        max_abs,
        mean_abs: sum_abs / len,
        rmse: (sum_sq / len).sqrt(),
    }
}
