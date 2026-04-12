#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::marker::PhantomData;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorPrimitive, activation};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::tensor::CubeTensor;
#[cfg(test)]
use burn_ndarray::NdArray;
use burn_wgpu::CubeBackend;

use crate::kernels::sequence::mamba3::backward_runtime::{
    try_carry_backward, try_current_score_backward, try_fused_score_carry_backward,
    try_reverse_cumsum_bhl, try_reverse_cumsum_blhr,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_backward_cuda;
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_backward_wgpu;
use crate::kernels::sequence::mamba3::forward::silu;
use crate::kernels::sequence::mamba3::preprocess_runtime::fused_mamba3_preprocess_backward_wgpu;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_backward_cuda;
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_backward_wgpu;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(test)]
type NdArrayBackend = NdArray<f32>;

const PI: f32 = std::f32::consts::PI;

pub const AVAILABLE: bool = true;

#[cfg(test)]
static WGPU_BC_BACKWARD_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(test)]
static WGPU_ROTARY_BACKWARD_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(test)]
static WGPU_REVERSE_CUMSUM_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(test)]
static CUDA_REVERSE_CUMSUM_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

fn mamba3_backward_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BURN_DRAGON_MAMBA3_BACKWARD_PROFILE")
            .ok()
            .as_deref()
            .map(|value| !matches!(value, "0" | "false" | "FALSE" | "off" | "OFF"))
            .unwrap_or(false)
    })
}

fn log_mamba3_backward_profile(
    label: &str,
    start: Instant,
    profile_start: Instant,
    backend: &str,
    chunk_start: usize,
    chunk_end: usize,
) {
    if !mamba3_backward_profile_enabled() {
        return;
    }
    eprintln!(
        "[mamba3-backward:{backend}] chunk={chunk_start}..{chunk_end} stage={label} stage_ms={:.3} total_ms={:.3}",
        start.elapsed().as_secs_f64() * 1_000.0,
        profile_start.elapsed().as_secs_f64() * 1_000.0,
    );
}

fn use_mamba3_wgpu_bc_backward_runtime() -> bool {
    #[cfg(test)]
    match WGPU_BC_BACKWARD_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_BC_BACKWARD_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_mamba3_wgpu_rotary_backward_runtime() -> bool {
    #[cfg(test)]
    match WGPU_ROTARY_BACKWARD_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_ROTARY_BACKWARD_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_mamba3_wgpu_reverse_cumsum_runtime() -> bool {
    #[cfg(test)]
    match WGPU_REVERSE_CUMSUM_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_REVERSE_CUMSUM_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(feature = "cuda")]
fn use_mamba3_cuda_reverse_cumsum_runtime() -> bool {
    #[cfg(test)]
    match CUDA_REVERSE_CUMSUM_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_CUDA_REVERSE_CUMSUM_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg(test)]
pub(crate) fn set_mamba3_wgpu_bc_backward_runtime_for_tests(value: Option<bool>) {
    WGPU_BC_BACKWARD_RUNTIME_OVERRIDE.store(
        match value {
            Some(false) => 0,
            Some(true) => 1,
            None => -1,
        },
        Ordering::Relaxed,
    );
}

#[cfg(test)]
pub(crate) fn set_mamba3_wgpu_rotary_backward_runtime_for_tests(value: Option<bool>) {
    WGPU_ROTARY_BACKWARD_RUNTIME_OVERRIDE.store(
        match value {
            Some(false) => 0,
            Some(true) => 1,
            None => -1,
        },
        Ordering::Relaxed,
    );
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_mamba3_wgpu_reverse_cumsum_runtime_for_tests(value: Option<bool>) {
    WGPU_REVERSE_CUMSUM_RUNTIME_OVERRIDE.store(
        match value {
            Some(false) => 0,
            Some(true) => 1,
            None => -1,
        },
        Ordering::Relaxed,
    );
}

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn set_mamba3_cuda_reverse_cumsum_runtime_for_tests(value: Option<bool>) {
    CUDA_REVERSE_CUMSUM_RUNTIME_OVERRIDE.store(
        match value {
            Some(false) => 0,
            Some(true) => 1,
            None => -1,
        },
        Ordering::Relaxed,
    );
}

#[derive(Debug, Clone)]
pub(crate) struct Mamba3TensorizedBackwardState<FT> {
    pub(crate) hidden_states: FT,
    pub(crate) in_proj: FT,
    pub(crate) dt_bias: FT,
    pub(crate) b_bias: FT,
    pub(crate) c_bias: FT,
    pub(crate) b_norm_weight: FT,
    pub(crate) c_norm_weight: FT,
    pub(crate) d_skip: FT,
    pub(crate) out_proj: FT,
    pub(crate) chunks: Vec<Mamba3ChunkTrace<FT>>,
    pub(crate) d_inner: usize,
    pub(crate) d_state: usize,
    pub(crate) headdim: usize,
    pub(crate) ngroups: usize,
    pub(crate) num_rope_angles: usize,
    pub(crate) norm_eps: f32,
    pub(crate) a_floor: f32,
    pub(crate) chunk_size: usize,
}

#[derive(Debug)]
pub(crate) struct TensorizedMamba3Backward<B>(pub(crate) PhantomData<B>);

#[derive(Debug)]
pub(crate) struct TensorizedMamba3CurrentScoreBackward<B>(pub(crate) PhantomData<B>);

#[derive(Debug)]
pub(crate) struct TensorizedMamba3StateUpdateBackward<B>(pub(crate) PhantomData<B>);

#[derive(Debug)]
pub(crate) struct TensorizedMamba3PreprocessBackward<B>(pub(crate) PhantomData<B>);

#[derive(Debug, Clone)]
pub(crate) struct Mamba3CurrentScoreBackwardState<FT> {
    pub(crate) q_head: FT,
    pub(crate) k_head: FT,
    pub(crate) v_head: FT,
    pub(crate) da_prefix: FT,
    pub(crate) raw_scores: FT,
}

#[derive(Debug, Clone)]
pub(crate) struct Mamba3StateUpdateBackwardState<FT> {
    pub(crate) state_tilde: FT,
    pub(crate) da_prefix: FT,
    pub(crate) v_head: FT,
    pub(crate) k_head: FT,
}

#[derive(Debug, Clone)]
pub(crate) struct Mamba3PreprocessBackwardState<FT> {
    pub(crate) q_pre: FT,
    pub(crate) k_pre: FT,
    pub(crate) angles: FT,
    pub(crate) gamma: FT,
    pub(crate) scale: FT,
}

#[derive(Debug, Clone)]
pub(crate) struct Mamba3ChunkTrace<FT> {
    pub(crate) chunk_start: usize,
    pub(crate) chunk_end: usize,
    pub(crate) k_state: FT,
    pub(crate) v_state: FT,
    pub(crate) q_pre: FT,
    pub(crate) k_pre: FT,
    pub(crate) b_input: FT,
    pub(crate) c_input: FT,
    pub(crate) b_inv_rms: FT,
    pub(crate) c_inv_rms: FT,
    pub(crate) dt_pre: FT,
    pub(crate) dt: FT,
    pub(crate) a_unclamped: FT,
    pub(crate) a: FT,
    pub(crate) trap: FT,
    pub(crate) gamma: FT,
    pub(crate) scale: FT,
    pub(crate) tanh_angle: FT,
    pub(crate) angle_chunk: FT,
    pub(crate) q_head: FT,
    pub(crate) k_rot_chunk: FT,
    pub(crate) k_head: FT,
    pub(crate) v_head: FT,
    pub(crate) z_head: FT,
    pub(crate) state_tilde: FT,
    pub(crate) da_prefix: FT,
    pub(crate) exp_da_prefix: FT,
    pub(crate) qk_inner: FT,
    pub(crate) y_pre: FT,
}

#[derive(Debug, Clone)]
struct ChunkEntry<B: BackendTrait> {
    chunk_start: usize,
    chunk_end: usize,
    k_state: Tensor<B, 3>,
    v_state: Tensor<B, 3>,
    q_pre: Tensor<B, 4>,
    k_pre: Tensor<B, 4>,
    b_input: Tensor<B, 4>,
    c_input: Tensor<B, 4>,
    b_inv_rms: Tensor<B, 3>,
    c_inv_rms: Tensor<B, 3>,
    dt_pre: Tensor<B, 3>,
    dt: Tensor<B, 3>,
    a_unclamped: Tensor<B, 3>,
    a: Tensor<B, 3>,
    trap: Tensor<B, 3>,
    gamma: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    tanh_angle: Tensor<B, 4>,
    angle_chunk: Tensor<B, 4>,
    q_head: Tensor<B, 4>,
    k_rot_chunk: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
    v_head: Tensor<B, 4>,
    z_head: Tensor<B, 4>,
    state_tilde: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
    exp_da_prefix: Tensor<B, 3>,
    qk_inner: Tensor<B, 3>,
    y_pre: Tensor<B, 4>,
}

fn unpack_chunk_entry<B: BackendTrait>(
    trace: &Mamba3ChunkTrace<B::FloatTensorPrimitive>,
) -> ChunkEntry<B>
where
    B::FloatTensorPrimitive: 'static,
{
    ChunkEntry {
        chunk_start: trace.chunk_start,
        chunk_end: trace.chunk_end,
        k_state: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.k_state.clone())),
        v_state: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.v_state.clone())),
        q_pre: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.q_pre.clone())),
        k_pre: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.k_pre.clone())),
        b_input: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.b_input.clone())),
        c_input: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.c_input.clone())),
        b_inv_rms: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.b_inv_rms.clone())),
        c_inv_rms: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.c_inv_rms.clone())),
        dt_pre: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.dt_pre.clone())),
        dt: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.dt.clone())),
        a_unclamped: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            trace.a_unclamped.clone(),
        )),
        a: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.a.clone())),
        trap: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.trap.clone())),
        gamma: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.gamma.clone())),
        scale: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.scale.clone())),
        tanh_angle: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            trace.tanh_angle.clone(),
        )),
        angle_chunk: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            trace.angle_chunk.clone(),
        )),
        q_head: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.q_head.clone())),
        k_rot_chunk: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            trace.k_rot_chunk.clone(),
        )),
        k_head: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.k_head.clone())),
        v_head: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.v_head.clone())),
        z_head: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.z_head.clone())),
        state_tilde: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            trace.state_tilde.clone(),
        )),
        da_prefix: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.da_prefix.clone())),
        exp_da_prefix: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            trace.exp_da_prefix.clone(),
        )),
        qk_inner: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(trace.qk_inner.clone())),
        y_pre: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(trace.y_pre.clone())),
    }
}

pub(crate) fn tensorized_mamba3_backward_impl<B>(
    ops: Ops<Mamba3TensorizedBackwardState<B::FloatTensorPrimitive>, 9>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
{
    let grad_output = grads.consume::<B>(&ops.node);
    let state = ops.state;
    let parents = ops.parents;

    let hidden_states =
        Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.hidden_states.clone()));
    let in_proj = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.in_proj.clone()));
    let _dt_bias = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.dt_bias.clone()));
    let _b_bias = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.b_bias.clone()));
    let _c_bias = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.c_bias.clone()));
    let b_norm_weight =
        Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.b_norm_weight.clone()));
    let c_norm_weight =
        Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.c_norm_weight.clone()));
    let d_skip = Tensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.d_skip.clone()));
    let out_proj = Tensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.out_proj.clone()));
    let grad_output = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_output.clone()));

    let d_inner = state.d_inner;
    let d_state = state.d_state;
    let headdim = state.headdim;
    let ngroups = state.ngroups;
    let num_rope_angles = state.num_rope_angles;
    let _norm_eps = state.norm_eps;
    let a_floor = state.a_floor;
    let _chunk_size = state.chunk_size;

    let [batch, _views, time, d_model] = hidden_states.shape().dims::<4>();
    let nheads = d_inner / headdim;
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles;
    let device = hidden_states.device();

    let mut grad_out_proj = Tensor::<B, 2>::zeros([d_inner, d_model], &device);
    let mut grad_z_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_x_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_b_input_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_c_input_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_angle_proj_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_gamma_direct_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_scale_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_dt_direct_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_a_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_trap_direct_chunks = Vec::with_capacity(state.chunks.len());
    let mut dt_pre_chunks = Vec::with_capacity(state.chunks.len());
    let mut dt_chunks = Vec::with_capacity(state.chunks.len());
    let mut a_unclamped_chunks = Vec::with_capacity(state.chunks.len());
    let mut trap_chunks = Vec::with_capacity(state.chunks.len());
    let mut grad_b_bias_total = Tensor::<B, 2>::zeros([nheads, d_state], &device);
    let mut grad_c_bias_total = Tensor::<B, 2>::zeros([nheads, d_state], &device);
    let mut grad_b_norm_weight_total = Tensor::<B, 1>::zeros([d_state], &device);
    let mut grad_c_norm_weight_total = Tensor::<B, 1>::zeros([d_state], &device);
    let mut grad_d_skip_total = Tensor::<B, 1>::zeros([nheads], &device);

    let mut grad_ssm_carry = Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], &device);
    let mut grad_angle_carry = Tensor::<B, 3>::zeros([batch, nheads, num_rope_angles], &device);
    let mut grad_k_carry = Tensor::<B, 3>::zeros([batch, nheads, d_state], &device);
    let mut grad_v_carry = Tensor::<B, 3>::zeros([batch, nheads, headdim], &device);
    let backend_name = if std::any::type_name::<B>().contains("CudaRuntime") {
        "cuda"
    } else {
        "wgpu"
    };

    for chunk_trace in state.chunks.iter().rev() {
        let chunk = unpack_chunk_entry(chunk_trace);
        let chunk_start = chunk.chunk_start;
        let chunk_end = chunk.chunk_end;
        let chunk_len = chunk_end - chunk_start;
        let profile_start = Instant::now();

        let q_pre_chunk = chunk.q_pre.clone();
        let k_pre_chunk = chunk.k_pre.clone();
        let b_input_chunk = chunk.b_input.clone();
        let c_input_chunk = chunk.c_input.clone();
        let b_inv_rms_chunk = chunk.b_inv_rms.clone();
        let c_inv_rms_chunk = chunk.c_inv_rms.clone();
        let dt_pre_chunk = chunk.dt_pre.clone();
        let dt_chunk = chunk.dt.clone();
        let a_unclamped_chunk = chunk.a_unclamped.clone();
        let a_chunk = chunk.a.clone();
        let trap_chunk = chunk.trap.clone();
        let gamma_chunk = chunk.gamma.clone();
        let scale_chunk = chunk.scale.clone();
        let tanh_angle = chunk.tanh_angle.clone();
        let angle_chunk = chunk.angle_chunk.clone();
        let q_head = chunk.q_head.clone();
        let k_rot_chunk = chunk.k_rot_chunk.clone();
        let k_head = chunk.k_head.clone();
        let v_head = chunk.v_head.clone();
        let z_head = chunk.z_head.clone();
        let state_tilde = chunk.state_tilde.clone();
        let da_prefix = chunk.da_prefix.clone();
        let exp_da_prefix = chunk.exp_da_prefix.clone();
        let qk_inner = chunk.qk_inner.clone();
        let y_pre = chunk.y_pre.clone();

        let alpha0 = dt_chunk.clone().slice_dim(1, 0..1).reshape([batch, nheads])
            * trap_chunk
                .clone()
                .slice_dim(1, 0..1)
                .reshape([batch, nheads])
                .neg()
                .add_scalar(1.0);
        let outer_prev = chunk.v_state.clone().unsqueeze_dim::<4>(3)
            * chunk.k_state.clone().unsqueeze_dim::<4>(2);
        let qk_dot_chunk = qk_inner.clone() * gamma_chunk.clone();
        let prev_scores = q_head.clone().matmul(state_tilde.clone().swap_dims(2, 3));
        let prev_out = prev_scores.clone() * exp_da_prefix.clone().unsqueeze_dim::<4>(3);
        let diff =
            da_prefix.clone().unsqueeze_dim::<4>(3) - da_prefix.clone().unsqueeze_dim::<4>(2);
        let decay = diff.clone().clamp_max(0.0).exp();
        let raw_scores = q_head.clone().matmul(k_head.clone().swap_dims(2, 3));
        let tril_scores = (raw_scores.clone() * decay.clone()).tril(-1);
        let qk_dot_head = qk_dot_chunk
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, nheads, chunk_len, 1]);
        let d_skip_head = d_skip.clone().reshape([1, nheads, 1, 1]);
        let gate = silu(z_head.clone());
        let y = gate.clone() * y_pre.clone();
        let output_proj_backward_start = Instant::now();
        let y_flat = y
            .clone()
            .swap_dims(1, 2)
            .reshape([batch * chunk_len, d_inner]);
        let grad_output_chunk = grad_output
            .clone()
            .slice_dim(2, chunk_start..chunk_end)
            .reshape([batch * chunk_len, d_model]);
        grad_out_proj = grad_out_proj
            + y_flat
                .clone()
                .swap_dims(0, 1)
                .matmul(grad_output_chunk.clone());
        let grad_y = grad_output_chunk
            .matmul(out_proj.clone().swap_dims(0, 1))
            .reshape([batch, chunk_len, nheads, headdim])
            .swap_dims(1, 2);
        log_mamba3_backward_profile(
            "output_proj_backward",
            output_proj_backward_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );

        let grad_y_pre = grad_y.clone() * gate.clone();
        let sigmoid_z = activation::sigmoid(z_head.clone());
        let ones = sigmoid_z.clone().ones_like();
        let grad_z_head = grad_y
            * y_pre.clone()
            * (sigmoid_z.clone() * (ones.clone() + z_head.clone() * (ones - sigmoid_z)));

        let grad_prev_out = grad_y_pre.clone();
        let grad_current_out = grad_y_pre.clone();
        let grad_skip = grad_y_pre;

        let mut grad_v_head = grad_skip.clone() * (d_skip_head.clone() + qk_dot_head.clone());
        let grad_skip_scale = (grad_skip.clone() * v_head.clone())
            .sum_dim(3)
            .reshape([batch, nheads, chunk_len]);
        grad_d_skip_total = grad_d_skip_total
            + grad_skip_scale
                .clone()
                .sum_dim(0)
                .sum_dim(2)
                .reshape([nheads]);
        let grad_qk_dot_chunk = grad_skip_scale
            .clone()
            .swap_dims(1, 2)
            .reshape([batch, chunk_len, nheads]);

        let grad_gamma_chunk = grad_qk_dot_chunk.clone() * qk_inner.clone();
        let mut grad_q_pre_chunk = grad_qk_dot_chunk
            .clone()
            .reshape([batch, chunk_len, nheads, 1])
            * gamma_chunk.clone().unsqueeze_dim::<4>(3)
            * k_pre_chunk.clone();
        let mut grad_k_pre_chunk = grad_qk_dot_chunk
            .clone()
            .reshape([batch, chunk_len, nheads, 1])
            * gamma_chunk.clone().unsqueeze_dim::<4>(3)
            * q_pre_chunk.clone();

        let score_backward_start = Instant::now();
        let grad_prev_scores = grad_prev_out.clone() * exp_da_prefix.clone().unsqueeze_dim::<4>(3);
        let mut grad_da_prefix = (grad_prev_out.clone() * prev_out.clone())
            .sum_dim(3)
            .reshape([batch, nheads, chunk_len]);
        let mut grad_q_head = grad_prev_scores.clone().matmul(state_tilde.clone());
        let mut grad_state_tilde = grad_prev_scores.swap_dims(2, 3).matmul(q_head.clone());
        let da_last = da_prefix
            .clone()
            .slice_dim(2, chunk_len - 1..chunk_len)
            .reshape([batch, nheads]);
        let weighted_scale = (da_last.clone().unsqueeze_dim::<3>(2) - da_prefix.clone()).exp();
        let mut grad_da_last = (grad_ssm_carry.clone()
            * state_tilde.clone()
            * da_last.clone().reshape([batch, nheads, 1, 1]).exp())
        .sum_dim(2)
        .sum_dim(3)
        .reshape([batch, nheads]);
        grad_state_tilde = grad_state_tilde
            + grad_ssm_carry.clone() * da_last.clone().reshape([batch, nheads, 1, 1]).exp();

        let (
            used_fused_score_carry,
            grad_q_current_add,
            grad_k_head,
            grad_v_current_add,
            grad_da_current_add,
        ) = if let Some((grad_q_add, grad_k_add, grad_v_add, grad_da_add)) =
            try_fused_score_carry_backward(
                grad_current_out.clone(),
                v_head.clone(),
                q_head.clone(),
                k_head.clone(),
                raw_scores.clone(),
                decay.clone(),
                grad_ssm_carry.clone(),
                weighted_scale.clone(),
            ) {
            (true, grad_q_add, grad_k_add, grad_v_add, grad_da_add)
        } else {
            let grad_tril_scores = grad_current_out
                .clone()
                .matmul(v_head.clone().swap_dims(2, 3));
            let grad_v_current_add = tril_scores
                .clone()
                .swap_dims(2, 3)
                .matmul(grad_current_out.clone());
            let grad_current_scores = grad_tril_scores.tril(-1);
            let grad_raw_scores = grad_current_scores.clone() * decay.clone();
            let grad_decay = grad_current_scores * raw_scores.clone();
            let grad_q_add = grad_raw_scores.clone().matmul(k_head.clone());
            let mut grad_k_add = grad_raw_scores
                .clone()
                .swap_dims(2, 3)
                .matmul(q_head.clone());
            let decay_mask = diff.lower_equal_elem(0.0).float();
            let grad_diff = grad_decay * decay.clone() * decay_mask;
            let mut grad_da_add = grad_diff
                .clone()
                .sum_dim(3)
                .reshape([batch, nheads, chunk_len])
                - grad_diff.sum_dim(2).reshape([batch, nheads, chunk_len]);

            let weighted_v = v_head.clone() * weighted_scale.clone().unsqueeze_dim::<4>(3);
            let grad_weighted_v_t = grad_ssm_carry
                .clone()
                .matmul(k_head.clone().swap_dims(2, 3));
            let grad_weighted_v = grad_weighted_v_t.swap_dims(2, 3);
            grad_k_add = grad_k_add + weighted_v.matmul(grad_ssm_carry.clone());
            let grad_v_add = grad_v_current_add
                + grad_weighted_v.clone() * weighted_scale.clone().unsqueeze_dim::<4>(3);
            let grad_weighted_scale = (grad_weighted_v * v_head.clone())
                .sum_dim(3)
                .reshape([batch, nheads, chunk_len]);
            grad_da_last = grad_da_last
                + (grad_weighted_scale.clone() * weighted_scale.clone())
                    .sum_dim(2)
                    .reshape([batch, nheads]);
            grad_da_add = grad_da_add - grad_weighted_scale * weighted_scale.clone();
            grad_da_add = add_last_time_grad_bhl(grad_da_add, grad_da_last.clone());
            (false, grad_q_add, grad_k_add, grad_v_add, grad_da_add)
        };
        grad_q_head = grad_q_head + grad_q_current_add;
        grad_v_head = grad_v_head + grad_v_current_add;
        grad_da_prefix = grad_da_prefix + grad_da_current_add;
        if used_fused_score_carry {
            grad_da_prefix = add_last_time_grad_bhl(grad_da_prefix, grad_da_last);
        }
        log_mamba3_backward_profile(
            "score_carry_backward",
            score_backward_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );

        let grad_state_tilde_outer =
            grad_state_tilde.clone() * alpha0.clone().reshape([batch, nheads, 1, 1]);
        let grad_v_prev = (grad_state_tilde_outer.clone()
            * chunk.k_state.clone().unsqueeze_dim::<4>(2))
        .sum_dim(3)
        .reshape([batch, nheads, headdim]);
        let grad_k_prev = (grad_state_tilde_outer.clone()
            * chunk.v_state.clone().unsqueeze_dim::<4>(3))
        .sum_dim(2)
        .reshape([batch, nheads, d_state]);
        let grad_alpha0 = (grad_state_tilde.clone() * outer_prev.clone())
            .sum_dim(2)
            .sum_dim(3)
            .reshape([batch, nheads]);

        let grad_k_rot_from_scaled =
            grad_k_head.clone() * scale_chunk.clone().swap_dims(1, 2).unsqueeze_dim::<4>(3);
        let grad_scale_chunk = (grad_k_head * k_rot_chunk.clone().swap_dims(1, 2))
            .sum_dim(3)
            .reshape([batch, nheads, chunk_len]);
        let mut grad_k_rot_chunk = grad_k_rot_from_scaled.swap_dims(1, 2);
        grad_k_rot_chunk = add_last_time_grad_blhd(grad_k_rot_chunk, grad_k_carry.clone());
        let mut grad_angle_total =
            Tensor::<B, 4>::zeros([batch, chunk_len, nheads, num_rope_angles], &device);
        let grad_v_head = add_last_time_grad_bhld(grad_v_head, grad_v_carry.clone());
        let rotary_backward_start = Instant::now();
        let (grad_q_from_rot, grad_k_from_rot, grad_angles_from_rot) =
            rotate_pairwise_qk_backward_with_angles(
                q_pre_chunk.clone(),
                k_pre_chunk.clone(),
                angle_chunk.clone(),
                grad_q_head.swap_dims(1, 2),
                grad_k_rot_chunk.clone(),
                num_rope_angles,
            );
        grad_q_pre_chunk = grad_q_pre_chunk + grad_q_from_rot;
        grad_k_pre_chunk = grad_k_pre_chunk + grad_k_from_rot;
        grad_angle_total = grad_angle_total + grad_angles_from_rot;
        grad_angle_total = add_last_time_grad_blhr(grad_angle_total, grad_angle_carry.clone());
        log_mamba3_backward_profile(
            "rotary_backward",
            rotary_backward_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );

        let grad_angle_state_prev =
            grad_angle_total
                .clone()
                .sum_dim(1)
                .reshape([batch, nheads, num_rope_angles]);
        let reverse_cumsum_start = Instant::now();
        let grad_angle_delta = reverse_cumsum_blhr(grad_angle_total);
        let grad_angle_proj_chunk = grad_angle_delta.clone()
            * dt_chunk.clone().unsqueeze_dim::<4>(3)
            * PI
            * (Tensor::<B, 4>::ones([batch, chunk_len, nheads, num_rope_angles], &device)
                - tanh_angle.clone().powf_scalar(2.0));
        let mut grad_dt_chunk = (grad_angle_delta * tanh_angle.clone() * PI)
            .sum_dim(3)
            .reshape([batch, chunk_len, nheads]);

        let grad_da_chunk = reverse_cumsum_bhl(grad_da_prefix.clone());
        log_mamba3_backward_profile(
            "reverse_cumsum",
            reverse_cumsum_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );
        grad_dt_chunk = grad_dt_chunk
            + (grad_da_chunk.clone() * a_chunk.clone().swap_dims(1, 2)).swap_dims(1, 2);
        let grad_a_chunk = (grad_da_chunk * dt_chunk.clone().swap_dims(1, 2)).swap_dims(1, 2);

        let grad_dt0 = grad_alpha0.clone()
            * trap_chunk
                .clone()
                .slice_dim(1, 0..1)
                .reshape([batch, nheads])
                .neg()
                .add_scalar(1.0);
        let grad_trap0 =
            grad_alpha0.neg() * dt_chunk.clone().slice_dim(1, 0..1).reshape([batch, nheads]);
        let first_dt = grad_dt_chunk
            .clone()
            .slice_dim(1, 0..1)
            .reshape([batch, nheads])
            + grad_dt0;
        grad_dt_chunk = grad_dt_chunk.slice_assign(
            [0..batch, 0..1, 0..nheads],
            first_dt.reshape([batch, 1, nheads]),
        );
        let grad_trap_chunk = Tensor::<B, 3>::zeros([batch, chunk_len, nheads], &device)
            .slice_assign(
                [0..batch, 0..1, 0..nheads],
                grad_trap0.reshape([batch, 1, nheads]),
            );

        let grad_x_chunk = grad_v_head.swap_dims(1, 2);
        let grad_scale_chunk_bt = grad_scale_chunk.swap_dims(1, 2);
        let grad_gamma_chunk_bt = grad_gamma_chunk;
        let grad_angle_proj_bt = grad_angle_proj_chunk;

        grad_z_chunks.push(grad_z_head.swap_dims(1, 2));
        grad_x_chunks.push(grad_x_chunk);
        grad_angle_proj_chunks.push(grad_angle_proj_bt);
        grad_gamma_direct_chunks.push(grad_gamma_chunk_bt);
        grad_scale_chunks.push(grad_scale_chunk_bt);
        grad_dt_direct_chunks.push(grad_dt_chunk);
        grad_a_chunks.push(grad_a_chunk);
        grad_trap_direct_chunks.push(grad_trap_chunk);
        dt_pre_chunks.push(dt_pre_chunk.clone());
        dt_chunks.push(dt_chunk.clone());
        a_unclamped_chunks.push(a_unclamped_chunk.clone());
        trap_chunks.push(trap_chunk.clone());

        grad_c_bias_total = grad_c_bias_total
            + grad_q_pre_chunk
                .clone()
                .sum_dim(0)
                .sum_dim(1)
                .reshape([nheads, d_state]);
        grad_b_bias_total = grad_b_bias_total
            + grad_k_pre_chunk
                .clone()
                .sum_dim(0)
                .sum_dim(1)
                .reshape([nheads, d_state]);

        let bc_backward_start = Instant::now();
        let (grad_c_input_chunk, grad_c_norm_weight_chunk) =
            try_fused_group_rmsnorm_expand_bias_backward(
                c_input_chunk.clone(),
                c_norm_weight.clone(),
                grad_q_pre_chunk.clone(),
                c_inv_rms_chunk.clone(),
                nheads,
            )
            .unwrap_or_else(|| {
                let grad_c_norm_heads =
                    repeat_groups_to_heads_backward_4d(grad_q_pre_chunk.clone(), ngroups);
                rmsnorm_last_dim_backward_4d(
                    c_input_chunk,
                    c_norm_weight.clone(),
                    c_inv_rms_chunk,
                    grad_c_norm_heads,
                )
            });
        log_mamba3_backward_profile(
            "bc_backward",
            bc_backward_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );
        let b_backward_start = Instant::now();
        let (grad_b_input_chunk, grad_b_norm_weight_chunk) =
            try_fused_group_rmsnorm_expand_bias_backward(
                b_input_chunk.clone(),
                b_norm_weight.clone(),
                grad_k_pre_chunk.clone(),
                b_inv_rms_chunk.clone(),
                nheads,
            )
            .unwrap_or_else(|| {
                let grad_b_norm_heads =
                    repeat_groups_to_heads_backward_4d(grad_k_pre_chunk.clone(), ngroups);
                rmsnorm_last_dim_backward_4d(
                    b_input_chunk,
                    b_norm_weight.clone(),
                    b_inv_rms_chunk,
                    grad_b_norm_heads,
                )
            });
        log_mamba3_backward_profile(
            "b_backward",
            b_backward_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );
        log_mamba3_backward_profile(
            "chunk_done",
            profile_start,
            profile_start,
            backend_name,
            chunk_start,
            chunk_end,
        );
        grad_c_norm_weight_total = grad_c_norm_weight_total + grad_c_norm_weight_chunk;
        grad_b_norm_weight_total = grad_b_norm_weight_total + grad_b_norm_weight_chunk;
        grad_c_input_chunks.push(grad_c_input_chunk);
        grad_b_input_chunks.push(grad_b_input_chunk);

        grad_ssm_carry = grad_state_tilde;
        grad_angle_carry = grad_angle_state_prev;
        grad_k_carry = grad_k_prev;
        grad_v_carry = grad_v_prev;
    }

    grad_z_chunks.reverse();
    grad_x_chunks.reverse();
    grad_b_input_chunks.reverse();
    grad_c_input_chunks.reverse();
    grad_angle_proj_chunks.reverse();
    grad_gamma_direct_chunks.reverse();
    grad_scale_chunks.reverse();
    grad_dt_direct_chunks.reverse();
    grad_a_chunks.reverse();
    grad_trap_direct_chunks.reverse();
    dt_pre_chunks.reverse();
    dt_chunks.reverse();
    a_unclamped_chunks.reverse();
    trap_chunks.reverse();

    let grad_z_full = Tensor::cat(grad_z_chunks, 1);
    let grad_x_full = Tensor::cat(grad_x_chunks, 1);
    let grad_b_input_full = Tensor::cat(grad_b_input_chunks, 1);
    let grad_c_input_full = Tensor::cat(grad_c_input_chunks, 1);
    let grad_angle_proj_full = Tensor::cat(grad_angle_proj_chunks, 1);
    let grad_gamma_direct_full = Tensor::cat(grad_gamma_direct_chunks, 1);
    let grad_scale_full = Tensor::cat(grad_scale_chunks, 1);
    let grad_dt_direct_full = Tensor::cat(grad_dt_direct_chunks, 1);
    let grad_a_full = Tensor::cat(grad_a_chunks, 1);
    let grad_trap_full = Tensor::cat(grad_trap_direct_chunks, 1);
    let dt_pre_full = Tensor::cat(dt_pre_chunks, 1);
    let dt_full = Tensor::cat(dt_chunks, 1);
    let a_unclamped_full = Tensor::cat(a_unclamped_chunks, 1);
    let trap_full = Tensor::cat(trap_chunks, 1);

    let mut grad_dt_total = grad_dt_direct_full.clone();
    let mut grad_trap_total = grad_trap_full.clone();
    let grad_gamma_total = grad_gamma_direct_full.clone() + grad_scale_full.clone();
    grad_dt_total = grad_dt_total + grad_gamma_total.clone() * trap_full.clone();
    grad_trap_total = grad_trap_total + grad_gamma_total.clone() * dt_full.clone();
    if time > 1 {
        let shifted_consumer = grad_scale_full.clone().slice_dim(1, 0..time - 1);
        let dt_shift = shifted_consumer.clone()
            * trap_full
                .clone()
                .slice_dim(1, 1..time)
                .neg()
                .add_scalar(1.0);
        let trap_shift = shifted_consumer.neg() * dt_full.clone().slice_dim(1, 1..time);
        let updated_dt_tail = grad_dt_total.clone().slice_dim(1, 1..time) + dt_shift;
        let updated_trap_tail = grad_trap_total.clone().slice_dim(1, 1..time) + trap_shift;
        grad_dt_total = grad_dt_total.slice_assign([0..batch, 1..time, 0..nheads], updated_dt_tail);
        grad_trap_total =
            grad_trap_total.slice_assign([0..batch, 1..time, 0..nheads], updated_trap_tail);
    }

    let grad_dt_pre = grad_dt_total * activation::sigmoid(dt_pre_full.clone());
    let a_mask = a_unclamped_full.clone().lower_equal_elem(-a_floor).float();
    let grad_dd_a = grad_a_full
        * a_mask
        * (Tensor::<B, 3>::ones([batch, time, nheads], &device) - a_unclamped_full.clone().exp())
            .neg();
    let grad_trap_pre =
        grad_trap_total * trap_full.clone() * trap_full.clone().neg().add_scalar(1.0);
    let grad_dt_bias = grad_dt_pre.clone().sum_dim(0).sum_dim(1).reshape([nheads]);
    let grad_angle_proj_shared =
        grad_angle_proj_full
            .sum_dim(2)
            .reshape([batch, time, num_rope_angles]);

    let grad_projected = Tensor::cat(
        vec![
            grad_z_full.clone().reshape([batch, time, d_inner]),
            grad_x_full.clone().reshape([batch, time, d_inner]),
            grad_b_input_full
                .clone()
                .reshape([batch, time, ngroups * d_state]),
            grad_c_input_full
                .clone()
                .reshape([batch, time, ngroups * d_state]),
            grad_dt_pre,
            grad_dd_a,
            grad_trap_pre,
            grad_angle_proj_shared,
        ],
        2,
    );
    let final_projection_start = Instant::now();
    let grad_projected_flat = grad_projected.reshape([batch * time, in_proj_dim]);
    let hidden_flat = hidden_states.clone().reshape([batch * time, d_model]);
    let grad_hidden = grad_projected_flat
        .clone()
        .matmul(in_proj.clone().swap_dims(0, 1))
        .reshape([batch, 1, time, d_model]);
    let grad_in_proj = hidden_flat.swap_dims(0, 1).matmul(grad_projected_flat);
    log_mamba3_backward_profile(
        "final_projection",
        final_projection_start,
        final_projection_start,
        backend_name,
        0,
        time,
    );

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_hidden.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_in_proj.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_dt_bias.into_primitive().tensor());
    }
    if let Some(parent) = &parents[3] {
        grads.register::<B>(parent.id, grad_b_bias_total.into_primitive().tensor());
    }
    if let Some(parent) = &parents[4] {
        grads.register::<B>(parent.id, grad_c_bias_total.into_primitive().tensor());
    }
    if let Some(parent) = &parents[5] {
        grads.register::<B>(
            parent.id,
            grad_b_norm_weight_total.into_primitive().tensor(),
        );
    }
    if let Some(parent) = &parents[6] {
        grads.register::<B>(
            parent.id,
            grad_c_norm_weight_total.into_primitive().tensor(),
        );
    }
    if let Some(parent) = &parents[7] {
        grads.register::<B>(parent.id, grad_d_skip_total.into_primitive().tensor());
    }
    if let Some(parent) = &parents[8] {
        grads.register::<B>(parent.id, grad_out_proj.into_primitive().tensor());
    }
}

#[allow(dead_code)]
fn repeat_groups_to_heads_4d<B: BackendTrait>(
    grouped: Tensor<B, 4>,
    nheads: usize,
) -> Tensor<B, 4> {
    let [batch, time, ngroups, d_state] = grouped.shape().dims::<4>();
    grouped
        .reshape([batch, time, ngroups, 1, d_state])
        .repeat_dim(3, nheads / ngroups)
        .reshape([batch, time, nheads, d_state])
}

fn repeat_groups_to_heads_backward_4d<B: BackendTrait>(
    grad_heads: Tensor<B, 4>,
    ngroups: usize,
) -> Tensor<B, 4> {
    let [batch, time, nheads, d_state] = grad_heads.shape().dims::<4>();
    assert_eq!(nheads % ngroups, 0);
    grad_heads
        .reshape([batch, time, ngroups, nheads / ngroups, d_state])
        .sum_dim(3)
        .reshape([batch, time, ngroups, d_state])
}

#[allow(dead_code)]
fn rmsnorm_last_dim_forward_4d<B: BackendTrait>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, time, heads, width] = input.shape().dims::<4>();
    let inv_rms = input
        .clone()
        .powf_scalar(2.0)
        .mean_dim(3)
        .add_scalar(eps)
        .sqrt()
        .recip()
        .reshape([batch, time, heads]);
    let output = input.clone()
        * inv_rms.clone().reshape([batch, time, heads, 1])
        * weight.reshape([1, 1, 1, width]);
    (output, inv_rms)
}

fn rmsnorm_last_dim_backward_4d<B: BackendTrait>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    inv_rms: Tensor<B, 3>,
    grad_output: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 1>) {
    let [batch, time, heads, width] = input.shape().dims::<4>();
    let inv_rms = inv_rms.reshape([batch, time, heads, 1]);
    let normalized = input.clone() * inv_rms.clone();
    let grad_weight = (grad_output.clone() * normalized.clone())
        .sum_dim(0)
        .sum_dim(1)
        .sum_dim(2)
        .reshape([width]);
    let grad_normalized = grad_output * weight.reshape([1, 1, 1, width]);
    let dot = (grad_normalized.clone() * input.clone())
        .sum_dim(3)
        .reshape([batch, time, heads, 1]);
    let grad_input = grad_normalized * inv_rms.clone()
        - input
            * dot
                .mul(inv_rms.clone().powf_scalar(3.0))
                .div_scalar(width as f32);
    (grad_input, grad_weight)
}

fn try_fused_group_rmsnorm_expand_bias_backward<B: BackendTrait>(
    grouped_input: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    grad_expanded: Tensor<B, 4>,
    inv_rms_group: Tensor<B, 3>,
    nheads: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 1>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let width = weight.shape().dims::<1>()[0];
    let input_raw = grouped_input.into_primitive().tensor();
    let weight_raw = weight.into_primitive().tensor();
    let grad_raw = grad_expanded.into_primitive().tensor();
    let inv_raw = inv_rms_group.into_primitive().tensor();

    if use_mamba3_wgpu_bc_backward_runtime()
        && let (Some(input_cube), Some(weight_cube), Some(grad_cube), Some(inv_cube)) = (
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(input_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(weight_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(inv_raw.clone()),
        )
    {
        let output =
            fused_mamba3_bc_backward_wgpu(input_cube, weight_cube, grad_cube, inv_cube, nheads);
        let grad_input = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_input)?,
        ));
        let grad_weight_contrib = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_weight_contrib)?,
        ));
        let grad_weight = grad_weight_contrib
            .sum_dim(0)
            .sum_dim(1)
            .sum_dim(2)
            .reshape([width]);
        return Some((grad_input, grad_weight));
    }

    #[cfg(feature = "cuda")]
    if let (Some(input_cube), Some(weight_cube), Some(grad_cube), Some(inv_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(input_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(weight_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(inv_raw.clone()),
    ) {
        let output =
            fused_mamba3_bc_backward_cuda(input_cube, weight_cube, grad_cube, inv_cube, nheads);
        let grad_input = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_input)?,
        ));
        let grad_weight_contrib = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_weight_contrib)?,
        ));
        let grad_weight = grad_weight_contrib
            .sum_dim(0)
            .sum_dim(1)
            .sum_dim(2)
            .reshape([width]);
        return Some((grad_input, grad_weight));
    }

    None
}

#[allow(dead_code)]
fn rotate_pairwise_qk_with_angles<B: BackendTrait>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    num_rope_angles: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, time, nheads, width] = q.shape().dims::<4>();
    let rotary_dim = num_rope_angles * 2;
    let cos = angles.clone().cos();
    let sin = angles.clone().sin();

    let q_rot = q.clone().slice_dim(3, 0..rotary_dim);
    let k_rot = k.clone().slice_dim(3, 0..rotary_dim);
    let q_tail = (rotary_dim < width).then(|| q.slice_dim(3, rotary_dim..width));
    let k_tail = (rotary_dim < width).then(|| k.slice_dim(3, rotary_dim..width));

    let q_pairs = q_rot.reshape([batch, time, nheads, num_rope_angles, 2]);
    let k_pairs = k_rot.reshape([batch, time, nheads, num_rope_angles, 2]);
    let q0 = q_pairs
        .clone()
        .slice_dim(4, 0..1)
        .reshape([batch, time, nheads, num_rope_angles]);
    let q1 = q_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);
    let k0 = k_pairs
        .clone()
        .slice_dim(4, 0..1)
        .reshape([batch, time, nheads, num_rope_angles]);
    let k1 = k_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);

    let q_rotated = Tensor::cat(
        vec![
            (q0.clone() * cos.clone() - q1.clone() * sin.clone()).unsqueeze_dim::<5>(4),
            (q0 * sin.clone() + q1 * cos.clone()).unsqueeze_dim::<5>(4),
        ],
        4,
    )
    .reshape([batch, time, nheads, rotary_dim]);
    let k_rotated = Tensor::cat(
        vec![
            (k0.clone() * cos.clone() - k1.clone() * sin.clone()).unsqueeze_dim::<5>(4),
            (k0 * sin + k1 * cos).unsqueeze_dim::<5>(4),
        ],
        4,
    )
    .reshape([batch, time, nheads, rotary_dim]);

    let q_out = if let Some(tail) = q_tail {
        Tensor::cat(vec![q_rotated, tail], 3)
    } else {
        q_rotated
    };
    let k_out = if let Some(tail) = k_tail {
        Tensor::cat(vec![k_rotated, tail], 3)
    } else {
        k_rotated
    };
    (q_out, k_out)
}

fn rotate_pairwise_qk_backward_with_angles<B: BackendTrait>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    grad_q_rot: Tensor<B, 4>,
    grad_k_rot: Tensor<B, 4>,
    num_rope_angles: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
    if let Some(output) = try_rotate_pairwise_qk_backward_with_angles_runtime(
        q.clone(),
        k.clone(),
        angles.clone(),
        grad_q_rot.clone(),
        grad_k_rot.clone(),
        num_rope_angles,
    ) {
        return output;
    }
    let [batch, time, nheads, width] = q.shape().dims::<4>();
    let rotary_dim = num_rope_angles * 2;
    let cos = angles.clone().cos();
    let sin = angles.clone().sin();

    let q_rotary = q.clone().slice_dim(3, 0..rotary_dim);
    let k_rotary = k.clone().slice_dim(3, 0..rotary_dim);
    let grad_q_rotary = grad_q_rot.clone().slice_dim(3, 0..rotary_dim);
    let grad_k_rotary = grad_k_rot.clone().slice_dim(3, 0..rotary_dim);
    let grad_q_tail = (rotary_dim < width).then(|| grad_q_rot.slice_dim(3, rotary_dim..width));
    let grad_k_tail = (rotary_dim < width).then(|| grad_k_rot.slice_dim(3, rotary_dim..width));

    let q_pairs = q_rotary.reshape([batch, time, nheads, num_rope_angles, 2]);
    let k_pairs = k_rotary.reshape([batch, time, nheads, num_rope_angles, 2]);
    let grad_q_pairs = grad_q_rotary.reshape([batch, time, nheads, num_rope_angles, 2]);
    let grad_k_pairs = grad_k_rotary.reshape([batch, time, nheads, num_rope_angles, 2]);

    let q0 = q_pairs
        .clone()
        .slice_dim(4, 0..1)
        .reshape([batch, time, nheads, num_rope_angles]);
    let q1 = q_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);
    let k0 = k_pairs
        .clone()
        .slice_dim(4, 0..1)
        .reshape([batch, time, nheads, num_rope_angles]);
    let k1 = k_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);
    let gq0 =
        grad_q_pairs
            .clone()
            .slice_dim(4, 0..1)
            .reshape([batch, time, nheads, num_rope_angles]);
    let gq1 = grad_q_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);
    let gk0 =
        grad_k_pairs
            .clone()
            .slice_dim(4, 0..1)
            .reshape([batch, time, nheads, num_rope_angles]);
    let gk1 = grad_k_pairs
        .slice_dim(4, 1..2)
        .reshape([batch, time, nheads, num_rope_angles]);

    let grad_q0 = gq0.clone() * cos.clone() + gq1.clone() * sin.clone();
    let grad_q1 = gq1.clone() * cos.clone() - gq0.clone() * sin.clone();
    let grad_k0 = gk0.clone() * cos.clone() + gk1.clone() * sin.clone();
    let grad_k1 = gk1.clone() * cos.clone() - gk0.clone() * sin.clone();

    let grad_angle_q = (gq0.clone() * q0.clone() + gq1.clone() * q1.clone()).neg() * sin.clone()
        + (gq1 * q0 - gq0 * q1) * cos.clone();
    let grad_angle_k = (gk0.clone() * k0.clone() + gk1.clone() * k1.clone()).neg() * sin
        + (gk1 * k0 - gk0 * k1) * cos;
    let grad_angle = grad_angle_q + grad_angle_k;

    let grad_q_rotary = Tensor::cat(
        vec![grad_q0.unsqueeze_dim::<5>(4), grad_q1.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, time, nheads, rotary_dim]);
    let grad_k_rotary = Tensor::cat(
        vec![grad_k0.unsqueeze_dim::<5>(4), grad_k1.unsqueeze_dim::<5>(4)],
        4,
    )
    .reshape([batch, time, nheads, rotary_dim]);

    let grad_q = if let Some(tail) = grad_q_tail {
        Tensor::cat(vec![grad_q_rotary, tail], 3)
    } else {
        grad_q_rotary
    };
    let grad_k = if let Some(tail) = grad_k_tail {
        Tensor::cat(vec![grad_k_rotary, tail], 3)
    } else {
        grad_k_rotary
    };

    (grad_q, grad_k, grad_angle)
}

fn try_rotate_pairwise_qk_backward_with_angles_runtime<B: BackendTrait>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    grad_q_rot: Tensor<B, 4>,
    grad_k_rot: Tensor<B, 4>,
    num_rope_angles: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let q_raw = q.into_primitive().tensor();
    let k_raw = k.into_primitive().tensor();
    let angles_raw = angles.into_primitive().tensor();
    let grad_q_rot_raw = grad_q_rot.into_primitive().tensor();
    let grad_k_rot_raw = grad_k_rot.into_primitive().tensor();

    if use_mamba3_wgpu_rotary_backward_runtime()
        && let (
            Some(q_cube),
            Some(k_cube),
            Some(angles_cube),
            Some(grad_q_rot_cube),
            Some(grad_k_rot_cube),
        ) = (
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(angles_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_q_rot_raw.clone()),
            try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_k_rot_raw.clone()),
        )
    {
        let output = fused_mamba3_rotary_backward_wgpu(
            q_cube,
            k_cube,
            angles_cube,
            grad_q_rot_cube,
            grad_k_rot_cube,
            num_rope_angles,
        );
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_q,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_k,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_angle,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (
        Some(q_cube),
        Some(k_cube),
        Some(angles_cube),
        Some(grad_q_rot_cube),
        Some(grad_k_rot_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(angles_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_q_rot_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_k_rot_raw.clone()),
    ) {
        let output = fused_mamba3_rotary_backward_cuda(
            q_cube,
            k_cube,
            angles_cube,
            grad_q_rot_cube,
            grad_k_rot_cube,
            num_rope_angles,
        );
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_q,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_k,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.grad_angle,
            )?)),
        ));
    }

    None
}

fn reverse_cumsum_bhl<B: BackendTrait>(values: Tensor<B, 3>) -> Tensor<B, 3> {
    let use_runtime = if std::any::type_name::<B>().contains("CudaRuntime") {
        #[cfg(feature = "cuda")]
        {
            use_mamba3_cuda_reverse_cumsum_runtime()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    } else {
        use_mamba3_wgpu_reverse_cumsum_runtime()
    };
    if use_runtime && let Some(output) = try_reverse_cumsum_bhl(values.clone()) {
        return output;
    }
    let [batch, heads, time] = values.shape().dims::<3>();
    let device = values.device();
    let reverse_index = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .mul_scalar(-1)
        .add_scalar(time as i64 - 1)
        .reshape([1, 1, time])
        .repeat_dim(0, batch)
        .repeat_dim(1, heads);
    let reversed = values.clone().gather(2, reverse_index.clone());
    reversed.cumsum(2).gather(2, reverse_index)
}

fn reverse_cumsum_blhr<B: BackendTrait>(values: Tensor<B, 4>) -> Tensor<B, 4> {
    let use_runtime = if std::any::type_name::<B>().contains("CudaRuntime") {
        #[cfg(feature = "cuda")]
        {
            use_mamba3_cuda_reverse_cumsum_runtime()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    } else {
        use_mamba3_wgpu_reverse_cumsum_runtime()
    };
    if use_runtime && let Some(output) = try_reverse_cumsum_blhr(values.clone()) {
        return output;
    }
    let [batch, time, heads, width] = values.shape().dims::<4>();
    let device = values.device();
    let reverse_index = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .mul_scalar(-1)
        .add_scalar(time as i64 - 1)
        .reshape([1, time, 1, 1])
        .repeat_dim(0, batch)
        .repeat_dim(2, heads)
        .repeat_dim(3, width);
    let reversed = values.clone().gather(1, reverse_index.clone());
    reversed.cumsum(1).gather(1, reverse_index)
}

fn add_last_time_grad_bhl<B: BackendTrait>(
    values: Tensor<B, 3>,
    last_grad: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let [batch, heads, time] = values.shape().dims::<3>();
    let updated_last = values
        .clone()
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads])
        + last_grad;
    values.slice_assign(
        [0..batch, 0..heads, time - 1..time],
        updated_last.reshape([batch, heads, 1]),
    )
}

fn add_last_time_grad_blhr<B: BackendTrait>(
    values: Tensor<B, 4>,
    last_grad: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, time, heads, width] = values.shape().dims::<4>();
    let updated_last = values
        .clone()
        .slice_dim(1, time - 1..time)
        .reshape([batch, heads, width])
        + last_grad;
    values.slice_assign(
        [0..batch, time - 1..time, 0..heads, 0..width],
        updated_last.reshape([batch, 1, heads, width]),
    )
}

fn add_last_time_grad_blhd<B: BackendTrait>(
    values: Tensor<B, 4>,
    last_grad: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, time, heads, width] = values.shape().dims::<4>();
    let updated_last = values
        .clone()
        .slice_dim(1, time - 1..time)
        .reshape([batch, heads, width])
        + last_grad;
    values.slice_assign(
        [0..batch, time - 1..time, 0..heads, 0..width],
        updated_last.reshape([batch, 1, heads, width]),
    )
}

fn add_last_time_grad_bhld<B: BackendTrait>(
    values: Tensor<B, 4>,
    last_grad: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, heads, time, width] = values.shape().dims::<4>();
    let updated_last = values
        .clone()
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads, width])
        + last_grad;
    values.slice_assign(
        [0..batch, 0..heads, time - 1..time, 0..width],
        updated_last.reshape([batch, heads, 1, width]),
    )
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

fn tensorized_mamba3_current_score_backward_impl<B: BackendTrait>(
    ops: Ops<Mamba3CurrentScoreBackwardState<B::FloatTensorPrimitive>, 4>,
    grads: &mut Gradients,
) {
    let grad_current_out = grads.consume::<B>(&ops.node);
    let parents = ops.parents;
    let state = ops.state;

    let q_head = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.q_head));
    let k_head = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.k_head));
    let v_head = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.v_head));
    let da_prefix = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(state.da_prefix));
    let raw_scores = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.raw_scores));
    let grad_current_out = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_current_out));

    let decay = (da_prefix.clone().unsqueeze_dim::<4>(3) - da_prefix.clone().unsqueeze_dim::<4>(2))
        .clamp_max(0.0)
        .exp();
    let (grad_q_add, grad_k_add, grad_v_add, grad_da_add) =
        try_current_score_backward(grad_current_out, v_head, q_head, k_head, raw_scores, decay)
            .expect("mamba3 current score backward runtime");

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_q_add.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_k_add.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_v_add.into_primitive().tensor());
    }
    if let Some(parent) = &parents[3] {
        grads.register::<B>(parent.id, grad_da_add.into_primitive().tensor());
    }
}

fn tensorized_mamba3_state_update_backward_impl<B: BackendTrait>(
    ops: Ops<Mamba3StateUpdateBackwardState<B::FloatTensorPrimitive>, 4>,
    grads: &mut Gradients,
) {
    let grad_ssm_state = grads.consume::<B>(&ops.node);
    let parents = ops.parents;
    let state = ops.state;

    let state_tilde = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.state_tilde));
    let da_prefix = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(state.da_prefix));
    let v_head = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.v_head));
    let k_head = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.k_head));
    let grad_ssm_state = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_ssm_state));

    let [batch, heads, _headdim, _d_state] = state_tilde.shape().dims::<4>();
    let time = da_prefix.shape().dims::<3>()[2];
    let da_last = da_prefix
        .clone()
        .slice_dim(2, time - 1..time)
        .reshape([batch, heads]);
    let scale_last = da_last.clone().reshape([batch, heads, 1, 1]).exp();
    let weighted_scale = (da_last.clone().unsqueeze_dim::<3>(2) - da_prefix.clone()).exp();

    let grad_state_tilde = grad_ssm_state.clone() * scale_last.clone();
    let grad_da_last_from_state = (grad_ssm_state.clone() * state_tilde.clone() * scale_last)
        .sum_dim(2)
        .sum_dim(3)
        .reshape([batch, heads]);

    let (grad_k_add, grad_v_add, grad_weighted_scale) =
        try_carry_backward(grad_ssm_state, k_head, v_head, weighted_scale.clone())
            .expect("mamba3 state update backward runtime");

    let grad_da_scale = grad_weighted_scale * weighted_scale;
    let grad_da_last =
        grad_da_last_from_state + grad_da_scale.clone().sum_dim(2).reshape([batch, heads]);
    let grad_da_add = add_last_time_grad_bhl(grad_da_scale.neg(), grad_da_last);

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_state_tilde.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_da_add.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_v_add.into_primitive().tensor());
    }
    if let Some(parent) = &parents[3] {
        grads.register::<B>(parent.id, grad_k_add.into_primitive().tensor());
    }
}

fn tensorized_mamba3_preprocess_backward_impl<B: BackendTrait>(
    ops: Ops<Mamba3PreprocessBackwardState<B::FloatTensorPrimitive>, 5>,
    grads: &mut Gradients,
) {
    let grad_packed = grads.consume::<B>(&ops.node);
    let parents = ops.parents;
    let state = ops.state;

    let q_pre = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.q_pre));
    let k_pre = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.k_pre));
    let angles = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.angles));
    let gamma = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(state.gamma));
    let scale = Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(state.scale));
    let grad_packed = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_packed));

    let q_raw = q_pre.clone().into_primitive().tensor();
    let k_raw = k_pre.clone().into_primitive().tensor();
    let angles_raw = angles.clone().into_primitive().tensor();
    let gamma_raw = gamma.clone().into_primitive().tensor();
    let scale_raw = scale.clone().into_primitive().tensor();
    let grad_packed_raw = grad_packed.into_primitive().tensor();

    if let (
        Some(q_cube),
        Some(k_cube),
        Some(angles_cube),
        Some(gamma_cube),
        Some(scale_cube),
        Some(grad_packed_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(angles_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(gamma_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(scale_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_packed_raw.clone()),
    ) {
        let output = fused_mamba3_preprocess_backward_wgpu(
            q_cube,
            k_cube,
            angles_cube,
            gamma_cube,
            scale_cube,
            grad_packed_cube,
        );
        if let Some(parent) = &parents[0] {
            grads.register::<B>(
                parent.id,
                try_cast_backend::<B, _>(output.grad_q).expect("mamba3 preprocess grad_q cast"),
            );
        }
        if let Some(parent) = &parents[1] {
            grads.register::<B>(
                parent.id,
                try_cast_backend::<B, _>(output.grad_k).expect("mamba3 preprocess grad_k cast"),
            );
        }
        if let Some(parent) = &parents[2] {
            grads.register::<B>(
                parent.id,
                try_cast_backend::<B, _>(output.grad_angle)
                    .expect("mamba3 preprocess grad_angle cast"),
            );
        }
        if let Some(parent) = &parents[3] {
            grads.register::<B>(
                parent.id,
                try_cast_backend::<B, _>(output.grad_gamma)
                    .expect("mamba3 preprocess grad_gamma cast"),
            );
        }
        if let Some(parent) = &parents[4] {
            grads.register::<B>(
                parent.id,
                try_cast_backend::<B, _>(output.grad_scale)
                    .expect("mamba3 preprocess grad_scale cast"),
            );
        }
        return;
    }

    unreachable!("mamba3 preprocess backward runtime requires wgpu cube backend");
}

impl Backward<WgpuCubeBackend, 9> for TensorizedMamba3Backward<WgpuCubeBackend> {
    type State = Mamba3TensorizedBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl Backward<WgpuCubeBackend, 4> for TensorizedMamba3CurrentScoreBackward<WgpuCubeBackend> {
    type State = Mamba3CurrentScoreBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_current_score_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl Backward<WgpuCubeBackend, 4> for TensorizedMamba3StateUpdateBackward<WgpuCubeBackend> {
    type State = Mamba3StateUpdateBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_state_update_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl Backward<WgpuCubeBackend, 5> for TensorizedMamba3PreprocessBackward<WgpuCubeBackend> {
    type State = Mamba3PreprocessBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 5>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_preprocess_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 9> for TensorizedMamba3Backward<CudaCubeBackend> {
    type State = Mamba3TensorizedBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 4> for TensorizedMamba3CurrentScoreBackward<CudaCubeBackend> {
    type State = Mamba3CurrentScoreBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_current_score_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 4> for TensorizedMamba3StateUpdateBackward<CudaCubeBackend> {
    type State = Mamba3StateUpdateBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_state_update_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

#[cfg(test)]
impl Backward<NdArrayBackend, 9> for TensorizedMamba3Backward<NdArrayBackend> {
    type State =
        Mamba3TensorizedBackwardState<<NdArrayBackend as BackendTrait>::FloatTensorPrimitive>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba3_backward_impl::<NdArrayBackend>(ops, grads);
    }
}
