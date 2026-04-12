use std::any::Any;
use std::sync::Once;
use std::sync::OnceLock;
#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};
use std::time::Instant;

use burn::tensor::TensorPrimitive;
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, activation};
use burn_autodiff::Autodiff;
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
use burn_autodiff::ops::{Backward, OpsKind};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::CubeBackend;
use std::marker::PhantomData;

use crate::kernels::sequence::mamba3::backward::{
    Mamba3ChunkTrace, Mamba3CurrentScoreBackwardState, Mamba3PreprocessBackwardState,
    Mamba3StateUpdateBackwardState, Mamba3TensorizedBackwardState, TensorizedMamba3Backward,
    TensorizedMamba3CurrentScoreBackward, TensorizedMamba3PreprocessBackward,
    TensorizedMamba3StateUpdateBackward,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_forward_cuda;
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_forward_wgpu;
use crate::kernels::sequence::mamba3::forward_runtime::{
    try_current_score_forward, try_state_update_forward,
};
use crate::kernels::sequence::mamba3::preprocess_runtime::fused_mamba3_preprocess_forward_wgpu;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_forward_cuda;
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_forward_wgpu;
type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuCubeAutodiffBackend = Autodiff<WgpuCubeBackend>;
type WgpuCubeAutodiffTensor = <WgpuCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

const PI: f32 = std::f32::consts::PI;

#[cfg(test)]
static WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(test)]
static WGPU_STATE_UPDATE_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(test)]
static WGPU_PREPROCESS_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(all(test, feature = "cuda"))]
static CUDA_CURRENT_SCORE_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);
#[cfg(all(test, feature = "cuda"))]
static CUDA_STATE_UPDATE_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

fn mamba3_forward_profile_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("BURN_DRAGON_MAMBA3_FORWARD_PROFILE")
            .ok()
            .as_deref()
            .map(|value| !matches!(value, "0" | "false" | "FALSE" | "off" | "OFF"))
            .unwrap_or(false)
    })
}

fn mamba3_forward_backend_name<B: BackendTrait>() -> &'static str {
    let name = std::any::type_name::<B>();
    if name.contains("cubecl_wgpu") || name.contains("WgpuRuntime") {
        "wgpu"
    } else if name.contains("cubecl::cuda") || name.contains("CudaRuntime") {
        "cuda"
    } else {
        name
    }
}

fn log_mamba3_forward_profile(
    label: &str,
    start: Instant,
    profile_start: Instant,
    backend: &str,
    chunk_start: Option<usize>,
    chunk_end: Option<usize>,
) {
    if !mamba3_forward_profile_enabled() {
        return;
    }
    match (chunk_start, chunk_end) {
        (Some(chunk_start), Some(chunk_end)) => eprintln!(
            "[mamba3-forward:{backend}] chunk={chunk_start}..{chunk_end} stage={label} stage_ms={:.3} total_ms={:.3}",
            start.elapsed().as_secs_f64() * 1_000.0,
            profile_start.elapsed().as_secs_f64() * 1_000.0,
        ),
        _ => eprintln!(
            "[mamba3-forward:{backend}] stage={label} stage_ms={:.3} total_ms={:.3}",
            start.elapsed().as_secs_f64() * 1_000.0,
            profile_start.elapsed().as_secs_f64() * 1_000.0,
        ),
    }
}

#[derive(Debug, Clone)]
pub struct Mamba3TensorizedState<B: BackendTrait> {
    pub ssm: Tensor<B, 4>,
    pub angle: Tensor<B, 3>,
    pub k: Tensor<B, 3>,
    pub v: Tensor<B, 3>,
}

#[derive(Debug)]
pub struct Mamba3TensorizedOutput<B: BackendTrait> {
    pub context: Tensor<B, 4>,
    pub state: Mamba3TensorizedState<B>,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
#[derive(Debug)]
struct Mamba3ForwardTrace<B: BackendTrait> {
    chunks: Vec<Mamba3ChunkTrace<B::FloatTensorPrimitive>>,
}

pub(crate) fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

pub(crate) fn tanh_reference<B: BackendTrait, const D: usize>(
    values: Tensor<B, D>,
) -> Tensor<B, D> {
    activation::sigmoid(values.mul_scalar(2.0))
        .mul_scalar(2.0)
        .sub_scalar(1.0)
}

fn repeat_groups_to_heads_4d<B: BackendTrait>(
    grouped: Tensor<B, 4>,
    nheads: usize,
) -> Tensor<B, 4> {
    let [batch, time, ngroups, d_state] = grouped.shape().dims::<4>();
    assert_eq!(
        nheads % ngroups,
        0,
        "Mamba-3 requires nheads divisible by ngroups"
    );
    grouped
        .reshape([batch, time, ngroups, 1, d_state])
        .repeat_dim(3, nheads / ngroups)
        .reshape([batch, time, nheads, d_state])
}

fn rmsnorm_last_dim_forward_4d<B: BackendTrait>(
    values: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, time, heads, width] = values.shape().dims::<4>();
    let inv_rms = values
        .clone()
        .powf_scalar(2.0)
        .mean_dim(3)
        .add_scalar(eps)
        .sqrt()
        .recip()
        .reshape([batch, time, heads]);
    let output = values
        * inv_rms.clone().reshape([batch, time, heads, 1])
        * weight.reshape([1, 1, 1, width]);
    (output, inv_rms)
}

pub(crate) fn try_fused_group_rmsnorm_expand_bias_forward<B: BackendTrait>(
    grouped: Tensor<B, 4>,
    weight: Tensor<B, 1>,
    bias: Tensor<B, 2>,
    nheads: usize,
    eps: f32,
) -> Option<(Tensor<B, 4>, Tensor<B, 3>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let grouped_raw = grouped.into_primitive().tensor();
    let weight_raw = weight.into_primitive().tensor();
    let bias_raw = bias.into_primitive().tensor();

    if let (Some(grouped_cube), Some(weight_cube), Some(bias_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grouped_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(weight_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(bias_raw.clone()),
    ) {
        let output =
            fused_mamba3_bc_forward_wgpu(grouped_cube, weight_cube, bias_cube, nheads, eps);
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.expanded,
            )?)),
            Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.inv_rms,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (Some(grouped_cube), Some(weight_cube), Some(bias_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grouped_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(weight_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(bias_raw.clone()),
    ) {
        let output =
            fused_mamba3_bc_forward_cuda(grouped_cube, weight_cube, bias_cube, nheads, eps);
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.expanded,
            )?)),
            Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.inv_rms,
            )?)),
        ));
    }

    None
}

fn rotate_pairwise_qk_with_angles<B: BackendTrait>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    num_rope_angles: usize,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    if let Some(output) = try_rotate_pairwise_qk_with_angles_runtime(
        q.clone(),
        k.clone(),
        angles.clone(),
        num_rope_angles,
    ) {
        return output;
    }
    let [batch, time, nheads, width] = q.shape().dims::<4>();
    assert_eq!(k.shape().dims::<4>(), [batch, time, nheads, width]);
    assert_eq!(
        angles.shape().dims::<4>(),
        [batch, time, nheads, num_rope_angles]
    );
    let rotary_dim = num_rope_angles * 2;
    assert!(
        rotary_dim <= width,
        "Mamba-3 rotary dim {} must be <= q/k width {}",
        rotary_dim,
        width
    );
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

fn try_rotate_pairwise_qk_with_angles_runtime<B: BackendTrait>(
    q: Tensor<B, 4>,
    k: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    num_rope_angles: usize,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let q_raw = q.into_primitive().tensor();
    let k_raw = k.into_primitive().tensor();
    let angles_raw = angles.into_primitive().tensor();

    if let (Some(q_cube), Some(k_cube), Some(angles_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(angles_raw.clone()),
    ) {
        let output = fused_mamba3_rotary_forward_wgpu(q_cube, k_cube, angles_cube, num_rope_angles);
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.q_rot,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.k_rot,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (Some(q_cube), Some(k_cube), Some(angles_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(angles_raw.clone()),
    ) {
        let output = fused_mamba3_rotary_forward_cuda(q_cube, k_cube, angles_cube, num_rope_angles);
        return Some((
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.q_rot,
            )?)),
            Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                output.k_rot,
            )?)),
        ));
    }

    None
}

fn log_mamba3_path_selection_once(message: &str) {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| eprintln!("{message}"));
}

fn mamba3_chunk_size_override(env_key: &str) -> Option<usize> {
    std::env::var(env_key)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
}

fn is_wgpu_mamba_backend<B: BackendTrait>(hidden_states: &Tensor<B, 4>) -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    let raw = hidden_states.clone().into_primitive().tensor();
    try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(raw.clone()).is_some()
        || try_cast_primitive::<B, WgpuCubeAutodiffTensor>(raw).is_some()
}

#[cfg(feature = "cuda")]
fn is_cuda_mamba_backend<B: BackendTrait>(hidden_states: &Tensor<B, 4>) -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    let raw = hidden_states.clone().into_primitive().tensor();
    try_cast_primitive::<B, CubeTensor<CudaRuntime>>(raw.clone()).is_some()
        || try_cast_primitive::<B, CudaCubeAutodiffTensor>(raw).is_some()
}

#[cfg(any(test, feature = "cuda"))]
fn recommended_cuda_short_context_chunk_size(configured_chunk_size: usize, time: usize) -> usize {
    configured_chunk_size.max(time.min(256)).min(time.max(1))
}

fn recommended_wgpu_short_context_chunk_size(configured_chunk_size: usize, time: usize) -> usize {
    configured_chunk_size.max(time.min(128)).min(time.max(1))
}

fn resolve_tensorized_mamba3_chunk_size<B: BackendTrait>(
    _hidden_states: &Tensor<B, 4>,
    configured_chunk_size: usize,
) -> usize
where
    B::FloatTensorPrimitive: 'static,
{
    assert!(
        configured_chunk_size > 0,
        "mamba3 tensorized path requires chunk_size > 0"
    );
    let time = _hidden_states.shape().dims::<4>()[2];

    if let Some(override_size) =
        mamba3_chunk_size_override("BURN_DRAGON_MAMBA3_CHUNK_SIZE_OVERRIDE")
    {
        return override_size;
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_mamba_backend(_hidden_states) {
            let effective_chunk_size =
                recommended_cuda_short_context_chunk_size(configured_chunk_size, time);
            if effective_chunk_size != configured_chunk_size {
                log_mamba3_path_selection_once(&format!(
                    "mamba3 tensorized path: auto-promoting short-context chunk_size from {} to {} on cuda",
                    configured_chunk_size, effective_chunk_size
                ));
            }
            return effective_chunk_size;
        }
    }

    if is_wgpu_mamba_backend(_hidden_states) {
        if let Some(override_size) =
            mamba3_chunk_size_override("BURN_DRAGON_MAMBA3_WGPU_CHUNK_SIZE_OVERRIDE")
        {
            return override_size;
        }

        let effective_chunk_size =
            recommended_wgpu_short_context_chunk_size(configured_chunk_size, time);
        if effective_chunk_size != configured_chunk_size {
            log_mamba3_path_selection_once(&format!(
                "mamba3 tensorized path: auto-promoting short-context chunk_size from {} to {} on wgpu",
                configured_chunk_size, effective_chunk_size
            ));
        }
        return effective_chunk_size;
    }

    configured_chunk_size
}

pub fn use_tensorized_mamba3_forward_experimental() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA3_TENSORIZED_FORWARD")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

fn use_mamba3_wgpu_current_score_runtime() -> bool {
    #[cfg(test)]
    match WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_CURRENT_SCORE_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => false,
    }
}

fn use_mamba3_wgpu_preprocess_runtime() -> bool {
    #[cfg(test)]
    match WGPU_PREPROCESS_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    // Keep the fused preprocess boundary on by default for WGPU. It is the current
    // best-foot-forward path on the full training microbench, even though the margin
    // is somewhat noisy across runs.
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_PREPROCESS_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[cfg_attr(not(test), allow(dead_code))]
fn use_mamba3_wgpu_state_update_runtime() -> bool {
    mamba3_wgpu_state_update_runtime_override().unwrap_or(false)
}

fn mamba3_wgpu_state_update_runtime_override() -> Option<bool> {
    #[cfg(test)]
    match WGPU_STATE_UPDATE_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return Some(false),
        1 => return Some(true),
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_STATE_UPDATE_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => Some(false),
        Some(_) => Some(true),
        None => None,
    }
}

#[cfg(feature = "cuda")]
fn use_mamba3_cuda_current_score_runtime() -> bool {
    #[cfg(test)]
    match CUDA_CURRENT_SCORE_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_CUDA_CURRENT_SCORE_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => false,
    }
}

#[cfg(feature = "cuda")]
fn use_mamba3_cuda_state_update_runtime() -> bool {
    mamba3_cuda_state_update_runtime_override().unwrap_or(false)
}

#[cfg(feature = "cuda")]
fn mamba3_cuda_state_update_runtime_override() -> Option<bool> {
    #[cfg(test)]
    match CUDA_STATE_UPDATE_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return Some(false),
        1 => return Some(true),
        _ => {}
    }
    match std::env::var("BURN_DRAGON_MAMBA3_CUDA_STATE_UPDATE_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => Some(false),
        Some(_) => Some(true),
        None => None,
    }
}

fn use_tensorized_mamba3_wgpu_train_wrapper<B: BackendTrait>(hidden_states: &Tensor<B, 4>) -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_TENSORIZED_TRAIN_WRAPPER")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => return false,
        Some(_) => return true,
        None => {}
    }

    if !is_wgpu_mamba_backend(hidden_states) {
        return false;
    }

    // WGPU ships the direct-graph path by default because the custom analytic wrapper
    // is still materially slower on the promoted training shapes.
    false
}

#[cfg(feature = "cuda")]
fn use_tensorized_mamba3_cuda_train_wrapper() -> bool {
    match std::env::var("BURN_DRAGON_MAMBA3_CUDA_TENSORIZED_TRAIN_WRAPPER")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => true,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba3_forward<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Mamba3TensorizedOutput<B> {
    let chunk_size = resolve_tensorized_mamba3_chunk_size(&hidden_states, chunk_size);
    if use_tensorized_mamba3_wgpu_train_wrapper(&hidden_states) {
        if let Some(output) = try_tensorized_mamba3_autodiff_wgpu(
            hidden_states.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            norm_eps,
            a_floor,
            chunk_size,
            in_proj.clone(),
            dt_bias.clone(),
            b_bias.clone(),
            c_bias.clone(),
            b_norm_weight.clone(),
            c_norm_weight.clone(),
            d_skip.clone(),
            out_proj.clone(),
            state.clone(),
        ) {
            log_mamba3_path_selection_once(
                "mamba3 tensorized path: using WGPU custom analytic backward wrapper over chunked SISO recurrent angle/ssm/k/v state",
            );
            return output;
        }
        log_mamba3_path_selection_once(
            "mamba3 tensorized path: WGPU custom analytic backward wrapper unavailable, falling back to chunked direct SISO graph",
        );
    } else if is_wgpu_mamba_backend(&hidden_states) {
        log_mamba3_path_selection_once(
            "mamba3 tensorized path: using WGPU chunked direct SISO graph by default; set BURN_DRAGON_MAMBA3_WGPU_TENSORIZED_TRAIN_WRAPPER=1 to force the custom analytic wrapper",
        );
    }
    #[cfg(feature = "cuda")]
    {
        if use_tensorized_mamba3_cuda_train_wrapper() {
            if let Some(output) = try_tensorized_mamba3_autodiff_cuda(
                hidden_states.clone(),
                d_inner,
                d_state,
                headdim,
                ngroups,
                num_rope_angles,
                norm_eps,
                a_floor,
                chunk_size,
                in_proj.clone(),
                dt_bias.clone(),
                b_bias.clone(),
                c_bias.clone(),
                b_norm_weight.clone(),
                c_norm_weight.clone(),
                d_skip.clone(),
                out_proj.clone(),
                state.clone(),
            ) {
                log_mamba3_path_selection_once(
                    "mamba3 tensorized path: using custom analytic backward wrapper over chunked SISO recurrent angle/ssm/k/v state",
                );
                return output;
            }
            log_mamba3_path_selection_once(
                "mamba3 tensorized path: custom analytic backward wrapper unavailable, falling back to chunked direct SISO graph",
            );
        } else {
            log_mamba3_path_selection_once(
                "mamba3 tensorized path: using chunked direct SISO graph (custom analytic backward wrapper disabled by env)",
            );
        }
    }
    #[cfg(not(feature = "cuda"))]
    {
        log_mamba3_path_selection_once(
            "mamba3 tensorized path: using chunked direct SISO graph with recurrent angle/ssm/k/v state",
        );
    }
    tensorized_mamba3_forward_impl(
        hidden_states,
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        in_proj,
        dt_bias,
        b_bias,
        c_bias,
        b_norm_weight,
        c_norm_weight,
        d_skip,
        out_proj,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba3_forward_direct_graph<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Mamba3TensorizedOutput<B> {
    let chunk_size = resolve_tensorized_mamba3_chunk_size(&hidden_states, chunk_size);
    tensorized_mamba3_forward_impl(
        hidden_states,
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        in_proj,
        dt_bias,
        b_bias,
        c_bias,
        b_norm_weight,
        c_norm_weight,
        d_skip,
        out_proj,
        state,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn tensorized_mamba3_forward_custom_backward<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Mamba3TensorizedOutput<B> {
    let chunk_size = resolve_tensorized_mamba3_chunk_size(&hidden_states, chunk_size);
    if let Some(output) = try_tensorized_mamba3_autodiff_wgpu(
        hidden_states.clone(),
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        in_proj.clone(),
        dt_bias.clone(),
        b_bias.clone(),
        c_bias.clone(),
        b_norm_weight.clone(),
        c_norm_weight.clone(),
        d_skip.clone(),
        out_proj.clone(),
        state.clone(),
    ) {
        return output;
    }
    #[cfg(feature = "cuda")]
    {
        return try_tensorized_mamba3_autodiff_cuda(
            hidden_states,
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            norm_eps,
            a_floor,
            chunk_size,
            in_proj,
            dt_bias,
            b_bias,
            c_bias,
            b_norm_weight,
            c_norm_weight,
            d_skip,
            out_proj,
            state,
        )
        .expect("mamba3 custom backward wrapper requires wgpu or cuda autodiff cube backend");
    }
    #[cfg(not(feature = "cuda"))]
    {
        panic!("mamba3 custom backward wrapper requires wgpu autodiff cube backend");
    }
}

#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba3_autodiff_wgpu<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Option<Mamba3TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let hidden_states_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let dt_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let b_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(b_bias.into_primitive().tensor())?;
    let c_bias_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(c_bias.into_primitive().tensor())?;
    let b_norm_weight_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(b_norm_weight.into_primitive().tensor())?;
    let c_norm_weight_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(c_norm_weight.into_primitive().tensor())?;
    let d_skip_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let out_proj_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;

    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let tensor: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_angle_inner = match state.as_ref() {
        Some(state) => {
            let tensor: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.angle.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_k_inner = match state.as_ref() {
        Some(state) => {
            let tensor: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.k.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_v_inner = match state.as_ref() {
        Some(state) => {
            let tensor: WgpuCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.v.clone().into_primitive().tensor())?;
            Some(<WgpuCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };

    let hidden_states_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let dt_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(dt_bias_ad.clone());
    let b_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(b_bias_ad.clone());
    let c_bias_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(c_bias_ad.clone());
    let b_norm_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(b_norm_weight_ad.clone());
    let c_norm_weight_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(c_norm_weight_ad.clone());
    let d_skip_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let out_proj_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let (output, trace) = tensorized_mamba3_forward_impl_traced(
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        Tensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(in_proj_inner.clone())),
        Tensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(dt_bias_inner.clone())),
        Tensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(b_bias_inner.clone())),
        Tensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(c_bias_inner.clone())),
        Tensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            b_norm_weight_inner.clone(),
        )),
        Tensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            c_norm_weight_inner.clone(),
        )),
        Tensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(d_skip_inner.clone())),
        Tensor::<WgpuCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        match (
            initial_ssm_inner.clone(),
            initial_angle_inner.clone(),
            initial_k_inner.clone(),
            initial_v_inner.clone(),
        ) {
            (Some(ssm), Some(angle), Some(k), Some(v)) => Some(Mamba3TensorizedState {
                ssm: Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(ssm)),
                angle: Tensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(angle)),
                k: Tensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(k)),
                v: Tensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(v)),
            }),
            _ => None,
        },
        true,
    );
    let trace = trace.expect("mamba3 traced forward must return trace when requested");
    let context_inner = output.context.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();
    let angle_inner = output.state.angle.into_primitive().tensor();
    let k_inner = output.state.k.into_primitive().tensor();
    let v_inner = output.state.v.into_primitive().tensor();

    let context_ad = match TensorizedMamba3Backward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            dt_bias_ad.node.clone(),
            b_bias_ad.node.clone(),
            c_bias_ad.node.clone(),
            b_norm_weight_ad.node.clone(),
            c_norm_weight_ad.node.clone(),
            d_skip_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                dt_bias: dt_bias_inner,
                b_bias: b_bias_inner,
                c_bias: c_bias_inner,
                b_norm_weight: b_norm_weight_inner,
                c_norm_weight: c_norm_weight_inner,
                d_skip: d_skip_inner,
                out_proj: out_proj_inner,
                chunks: trace.chunks,
                d_inner,
                d_state,
                headdim,
                ngroups,
                num_rope_angles,
                norm_eps,
                a_floor,
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(Mamba3TensorizedOutput {
        context: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            context_ad,
        )?)),
        state: Mamba3TensorizedState {
            ssm: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(ssm_inner),
            )?)),
            angle: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(
                try_cast_backend::<B, _>(
                    <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(angle_inner),
                )?,
            )),
            k: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(k_inner),
            )?)),
            v: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(v_inner),
            )?)),
        },
    })
}

fn try_tensorized_mamba3_preprocess_autodiff_wgpu<B: BackendTrait>(
    q_pre: Tensor<B, 4>,
    k_pre: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    gamma: Tensor<B, 3>,
    scale: Tensor<B, 3>,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let q_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(q_pre.into_primitive().tensor())?;
    let k_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(k_pre.into_primitive().tensor())?;
    let angles_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(angles.into_primitive().tensor())?;
    let gamma_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(gamma.into_primitive().tensor())?;
    let scale_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(scale.into_primitive().tensor())?;

    let q_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(q_ad.clone());
    let k_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(k_ad.clone());
    let angles_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(angles_ad.clone());
    let gamma_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(gamma_ad.clone());
    let scale_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(scale_ad.clone());
    let [batch, time, heads, width] = q_inner.meta.shape.dims::<4>();

    let packed = fused_mamba3_preprocess_forward_wgpu(
        q_inner.clone(),
        k_inner.clone(),
        angles_inner.clone(),
        gamma_inner.clone(),
        scale_inner.clone(),
    )
    .packed;

    let packed_ad = match TensorizedMamba3PreprocessBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            q_ad.node.clone(),
            k_ad.node.clone(),
            angles_ad.node.clone(),
            gamma_ad.node.clone(),
            scale_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3PreprocessBackwardState {
                q_pre: q_inner,
                k_pre: k_inner,
                angles: angles_inner,
                gamma: gamma_inner,
                scale: scale_inner,
            },
            packed,
        ),
        OpsKind::UnTracked(prep) => prep.finish(packed),
    };

    let packed_tensor = Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
        B,
        _,
    >(packed_ad)?));
    let q_rot = packed_tensor.clone().slice_dim(3, 0..width);
    let k_scaled = packed_tensor.clone().slice_dim(3, width..(width * 2));
    let qk_dot = packed_tensor
        .slice_dim(3, width * 2..(width * 2 + 1))
        .reshape([batch, time, heads]);
    Some((q_rot, k_scaled, qk_dot))
}

fn try_tensorized_mamba3_preprocess<B: BackendTrait>(
    q_pre: Tensor<B, 4>,
    k_pre: Tensor<B, 4>,
    angles: Tensor<B, 4>,
    gamma: Tensor<B, 3>,
    scale: Tensor<B, 3>,
    capture_trace: bool,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>)>
where
    B::FloatTensorPrimitive: 'static,
{
    if capture_trace {
        return None;
    }
    if mamba3_forward_backend_name::<B>() == "wgpu" && use_mamba3_wgpu_preprocess_runtime() {
        return try_tensorized_mamba3_preprocess_autodiff_wgpu(
            q_pre.clone(),
            k_pre.clone(),
            angles.clone(),
            gamma.clone(),
            scale.clone(),
        );
    }
    None
}

fn try_tensorized_mamba3_current_score_autodiff_wgpu<B: BackendTrait>(
    q_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
    v_head: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let q_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(q_head.into_primitive().tensor())?;
    let k_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(k_head.into_primitive().tensor())?;
    let v_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(v_head.into_primitive().tensor())?;
    let da_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(da_prefix.into_primitive().tensor())?;

    let q_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(q_ad.clone());
    let k_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(k_ad.clone());
    let v_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(v_ad.clone());
    let da_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(da_ad.clone());

    let (current_out, raw_scores) = try_current_score_forward(
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(q_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(k_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(v_inner.clone())),
        Tensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(da_inner.clone())),
    )?;

    let current_out_inner = current_out.into_primitive().tensor();
    let raw_scores_inner = raw_scores.into_primitive().tensor();
    let current_out_ad = match TensorizedMamba3CurrentScoreBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            q_ad.node.clone(),
            k_ad.node.clone(),
            v_ad.node.clone(),
            da_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3CurrentScoreBackwardState {
                q_head: q_inner,
                k_head: k_inner,
                v_head: v_inner,
                da_prefix: da_inner,
                raw_scores: raw_scores_inner,
            },
            current_out_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(current_out_inner),
    };

    Some(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(current_out_ad)?,
    )))
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_tensorized_mamba3_autodiff_cuda<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Option<Mamba3TensorizedOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let hidden_states_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(hidden_states.into_primitive().tensor())?;
    let in_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(in_proj.into_primitive().tensor())?;
    let dt_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(dt_bias.into_primitive().tensor())?;
    let b_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(b_bias.into_primitive().tensor())?;
    let c_bias_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(c_bias.into_primitive().tensor())?;
    let b_norm_weight_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(b_norm_weight.into_primitive().tensor())?;
    let c_norm_weight_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(c_norm_weight.into_primitive().tensor())?;
    let d_skip_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let out_proj_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(out_proj.into_primitive().tensor())?;

    let initial_ssm_inner = match state.as_ref() {
        Some(state) => {
            let tensor: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.ssm.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_angle_inner = match state.as_ref() {
        Some(state) => {
            let tensor: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.angle.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_k_inner = match state.as_ref() {
        Some(state) => {
            let tensor: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.k.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };
    let initial_v_inner = match state.as_ref() {
        Some(state) => {
            let tensor: CudaCubeAutodiffTensor =
                try_cast_primitive::<B, _>(state.v.clone().into_primitive().tensor())?;
            Some(<CudaCubeAutodiffBackend as AutodiffBackend>::inner(tensor))
        }
        None => None,
    };

    let hidden_states_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(hidden_states_ad.clone());
    let in_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(in_proj_ad.clone());
    let dt_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(dt_bias_ad.clone());
    let b_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(b_bias_ad.clone());
    let c_bias_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(c_bias_ad.clone());
    let b_norm_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(b_norm_weight_ad.clone());
    let c_norm_weight_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(c_norm_weight_ad.clone());
    let d_skip_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(d_skip_ad.clone());
    let out_proj_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(out_proj_ad.clone());

    let (output, trace) = tensorized_mamba3_forward_impl_traced(
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            hidden_states_inner.clone(),
        )),
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        Tensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(in_proj_inner.clone())),
        Tensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(dt_bias_inner.clone())),
        Tensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(b_bias_inner.clone())),
        Tensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(c_bias_inner.clone())),
        Tensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            b_norm_weight_inner.clone(),
        )),
        Tensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            c_norm_weight_inner.clone(),
        )),
        Tensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(d_skip_inner.clone())),
        Tensor::<CudaCubeBackend, 2>::from_primitive(TensorPrimitive::Float(
            out_proj_inner.clone(),
        )),
        match (
            initial_ssm_inner.clone(),
            initial_angle_inner.clone(),
            initial_k_inner.clone(),
            initial_v_inner.clone(),
        ) {
            (Some(ssm), Some(angle), Some(k), Some(v)) => Some(Mamba3TensorizedState {
                ssm: Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(ssm)),
                angle: Tensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(angle)),
                k: Tensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(k)),
                v: Tensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(v)),
            }),
            _ => None,
        },
        true,
    );
    let trace = trace.expect("mamba3 traced forward must return trace when requested");
    let context_inner = output.context.into_primitive().tensor();
    let ssm_inner = output.state.ssm.into_primitive().tensor();
    let angle_inner = output.state.angle.into_primitive().tensor();
    let k_inner = output.state.k.into_primitive().tensor();
    let v_inner = output.state.v.into_primitive().tensor();

    let context_ad = match TensorizedMamba3Backward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            hidden_states_ad.node.clone(),
            in_proj_ad.node.clone(),
            dt_bias_ad.node.clone(),
            b_bias_ad.node.clone(),
            c_bias_ad.node.clone(),
            b_norm_weight_ad.node.clone(),
            c_norm_weight_ad.node.clone(),
            d_skip_ad.node.clone(),
            out_proj_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3TensorizedBackwardState {
                hidden_states: hidden_states_inner,
                in_proj: in_proj_inner,
                dt_bias: dt_bias_inner,
                b_bias: b_bias_inner,
                c_bias: c_bias_inner,
                b_norm_weight: b_norm_weight_inner,
                c_norm_weight: c_norm_weight_inner,
                d_skip: d_skip_inner,
                out_proj: out_proj_inner,
                chunks: trace.chunks,
                d_inner,
                d_state,
                headdim,
                ngroups,
                num_rope_angles,
                norm_eps,
                a_floor,
                chunk_size,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(Mamba3TensorizedOutput {
        context: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            context_ad,
        )?)),
        state: Mamba3TensorizedState {
            ssm: Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(ssm_inner),
            )?)),
            angle: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(
                try_cast_backend::<B, _>(
                    <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(angle_inner),
                )?,
            )),
            k: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(k_inner),
            )?)),
            v: Tensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(v_inner),
            )?)),
        },
    })
}

#[cfg(feature = "cuda")]
fn try_tensorized_mamba3_current_score_autodiff_cuda<B: BackendTrait>(
    q_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
    v_head: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let q_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(q_head.into_primitive().tensor())?;
    let k_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(k_head.into_primitive().tensor())?;
    let v_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(v_head.into_primitive().tensor())?;
    let da_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(da_prefix.into_primitive().tensor())?;

    let q_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(q_ad.clone());
    let k_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(k_ad.clone());
    let v_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(v_ad.clone());
    let da_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(da_ad.clone());

    let (current_out, raw_scores) = try_current_score_forward(
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(q_inner.clone())),
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(k_inner.clone())),
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(v_inner.clone())),
        Tensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(da_inner.clone())),
    )?;

    let current_out_inner = current_out.into_primitive().tensor();
    let raw_scores_inner = raw_scores.into_primitive().tensor();
    let current_out_ad = match TensorizedMamba3CurrentScoreBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            q_ad.node.clone(),
            k_ad.node.clone(),
            v_ad.node.clone(),
            da_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3CurrentScoreBackwardState {
                q_head: q_inner,
                k_head: k_inner,
                v_head: v_inner,
                da_prefix: da_inner,
                raw_scores: raw_scores_inner,
            },
            current_out_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(current_out_inner),
    };

    Some(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(current_out_ad)?,
    )))
}

fn try_tensorized_mamba3_current_score<B: BackendTrait>(
    q_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
    v_head: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
    capture_trace: bool,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if capture_trace {
        return None;
    }

    if use_mamba3_wgpu_current_score_runtime() {
        if let Some(output) = try_tensorized_mamba3_current_score_autodiff_wgpu(
            q_head.clone(),
            k_head.clone(),
            v_head.clone(),
            da_prefix.clone(),
        ) {
            return Some(output);
        }
    }

    #[cfg(feature = "cuda")]
    if use_mamba3_cuda_current_score_runtime() {
        if let Some(output) = try_tensorized_mamba3_current_score_autodiff_cuda(
            q_head.clone(),
            k_head.clone(),
            v_head.clone(),
            da_prefix.clone(),
        ) {
            return Some(output);
        }
    }

    try_current_score_forward(q_head, k_head, v_head, da_prefix).map(|(current_out, _)| current_out)
}

fn try_tensorized_mamba3_state_update_autodiff_wgpu<B: BackendTrait>(
    state_tilde: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
    v_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let state_tilde_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(state_tilde.into_primitive().tensor())?;
    let da_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(da_prefix.into_primitive().tensor())?;
    let v_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(v_head.into_primitive().tensor())?;
    let k_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(k_head.into_primitive().tensor())?;

    let state_tilde_inner =
        <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(state_tilde_ad.clone());
    let da_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(da_ad.clone());
    let v_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(v_ad.clone());
    let k_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(k_ad.clone());

    let ssm_state = try_state_update_forward(
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            state_tilde_inner.clone(),
        )),
        Tensor::<WgpuCubeBackend, 3>::from_primitive(TensorPrimitive::Float(da_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(v_inner.clone())),
        Tensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(k_inner.clone())),
    )?;

    let ssm_state_inner = ssm_state.into_primitive().tensor();
    let ssm_state_ad = match TensorizedMamba3StateUpdateBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            state_tilde_ad.node.clone(),
            da_ad.node.clone(),
            v_ad.node.clone(),
            k_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3StateUpdateBackwardState {
                state_tilde: state_tilde_inner,
                da_prefix: da_inner,
                v_head: v_inner,
                k_head: k_inner,
            },
            ssm_state_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(ssm_state_inner),
    };

    Some(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(ssm_state_ad)?,
    )))
}

#[cfg(feature = "cuda")]
fn try_tensorized_mamba3_state_update_autodiff_cuda<B: BackendTrait>(
    state_tilde: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
    v_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let state_tilde_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(state_tilde.into_primitive().tensor())?;
    let da_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(da_prefix.into_primitive().tensor())?;
    let v_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(v_head.into_primitive().tensor())?;
    let k_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(k_head.into_primitive().tensor())?;

    let state_tilde_inner =
        <CudaCubeAutodiffBackend as AutodiffBackend>::inner(state_tilde_ad.clone());
    let da_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(da_ad.clone());
    let v_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(v_ad.clone());
    let k_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(k_ad.clone());

    let ssm_state = try_state_update_forward(
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            state_tilde_inner.clone(),
        )),
        Tensor::<CudaCubeBackend, 3>::from_primitive(TensorPrimitive::Float(da_inner.clone())),
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(v_inner.clone())),
        Tensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(k_inner.clone())),
    )?;

    let ssm_state_inner = ssm_state.into_primitive().tensor();
    let ssm_state_ad = match TensorizedMamba3StateUpdateBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            state_tilde_ad.node.clone(),
            da_ad.node.clone(),
            v_ad.node.clone(),
            k_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            Mamba3StateUpdateBackwardState {
                state_tilde: state_tilde_inner,
                da_prefix: da_inner,
                v_head: v_inner,
                k_head: k_inner,
            },
            ssm_state_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(ssm_state_inner),
    };

    Some(Tensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(ssm_state_ad)?,
    )))
}

fn try_tensorized_mamba3_state_update<B: BackendTrait>(
    state_tilde: Tensor<B, 4>,
    da_prefix: Tensor<B, 3>,
    v_head: Tensor<B, 4>,
    k_head: Tensor<B, 4>,
    capture_trace: bool,
) -> Option<Tensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if capture_trace {
        return None;
    }

    let use_wgpu_runtime = should_use_mamba3_wgpu_state_update_runtime(&da_prefix);
    if use_wgpu_runtime {
        if let Some(output) = try_tensorized_mamba3_state_update_autodiff_wgpu(
            state_tilde.clone(),
            da_prefix.clone(),
            v_head.clone(),
            k_head.clone(),
        ) {
            return Some(output);
        }
    }

    #[cfg(feature = "cuda")]
    if use_mamba3_cuda_state_update_runtime() {
        if let Some(output) = try_tensorized_mamba3_state_update_autodiff_cuda(
            state_tilde.clone(),
            da_prefix.clone(),
            v_head.clone(),
            k_head.clone(),
        ) {
            return Some(output);
        }
    }

    try_state_update_forward(state_tilde, da_prefix, v_head, k_head)
}

fn should_use_mamba3_wgpu_state_update_runtime<B: BackendTrait>(da_prefix: &Tensor<B, 3>) -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    match mamba3_wgpu_state_update_runtime_override() {
        Some(value) => value,
        None => {
            mamba3_forward_backend_name::<B>() == "wgpu" && da_prefix.shape().dims::<3>()[2] >= 64
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn tensorized_mamba3_forward_impl<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
) -> Mamba3TensorizedOutput<B> {
    tensorized_mamba3_forward_impl_traced(
        hidden_states,
        d_inner,
        d_state,
        headdim,
        ngroups,
        num_rope_angles,
        norm_eps,
        a_floor,
        chunk_size,
        in_proj,
        dt_bias,
        b_bias,
        c_bias,
        b_norm_weight,
        c_norm_weight,
        d_skip,
        out_proj,
        state,
        false,
    )
    .0
}

#[allow(clippy::too_many_arguments)]
fn tensorized_mamba3_forward_impl_traced<B: BackendTrait>(
    hidden_states: Tensor<B, 4>,
    d_inner: usize,
    d_state: usize,
    headdim: usize,
    ngroups: usize,
    num_rope_angles: usize,
    norm_eps: f32,
    a_floor: f32,
    chunk_size: usize,
    in_proj: Tensor<B, 2>,
    dt_bias: Tensor<B, 1>,
    b_bias: Tensor<B, 2>,
    c_bias: Tensor<B, 2>,
    b_norm_weight: Tensor<B, 1>,
    c_norm_weight: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    out_proj: Tensor<B, 2>,
    state: Option<Mamba3TensorizedState<B>>,
    capture_trace: bool,
) -> (Mamba3TensorizedOutput<B>, Option<Mamba3ForwardTrace<B>>) {
    let profile_enabled = mamba3_forward_profile_enabled() && !capture_trace;
    let backend_name = if profile_enabled {
        Some(mamba3_forward_backend_name::<B>())
    } else {
        None
    };
    let profile_start = Instant::now();
    let [batch, views, time, d_model] = hidden_states.shape().dims::<4>();
    assert_eq!(views, 1, "mamba3 tensorized path expects a single view");
    assert_eq!(
        d_inner % headdim,
        0,
        "mamba3 tensorized path requires d_inner divisible by headdim"
    );
    let nheads = d_inner / headdim;
    assert!(ngroups > 0, "mamba3 tensorized path requires ngroups > 0");
    assert_eq!(
        nheads % ngroups,
        0,
        "mamba3 tensorized path requires nheads divisible by ngroups"
    );
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles;
    assert_eq!(
        in_proj.shape().dims::<2>(),
        [d_model, in_proj_dim],
        "mamba3 tensorized path requires in_proj=[d_model, in_proj_dim]"
    );
    assert_eq!(
        dt_bias.shape().dims::<1>(),
        [nheads],
        "mamba3 tensorized path requires dt_bias=[nheads]"
    );
    assert_eq!(
        b_bias.shape().dims::<2>(),
        [nheads, d_state],
        "mamba3 tensorized path requires b_bias=[nheads, d_state]"
    );
    assert_eq!(
        c_bias.shape().dims::<2>(),
        [nheads, d_state],
        "mamba3 tensorized path requires c_bias=[nheads, d_state]"
    );
    assert_eq!(
        b_norm_weight.shape().dims::<1>(),
        [d_state],
        "mamba3 tensorized path requires b_norm_weight=[d_state]"
    );
    assert_eq!(
        c_norm_weight.shape().dims::<1>(),
        [d_state],
        "mamba3 tensorized path requires c_norm_weight=[d_state]"
    );
    assert_eq!(
        d_skip.shape().dims::<1>(),
        [nheads],
        "mamba3 tensorized path requires d_skip=[nheads]"
    );
    assert_eq!(
        out_proj.shape().dims::<2>(),
        [d_inner, d_model],
        "mamba3 tensorized path requires out_proj=[d_inner, d_model]"
    );
    let device = hidden_states.device();
    let projection_start = Instant::now();
    let projected = hidden_states
        .clone()
        .reshape([batch * time, d_model])
        .matmul(in_proj)
        .reshape([batch, time, in_proj_dim]);
    if let Some(backend_name) = backend_name {
        log_mamba3_forward_profile(
            "projection",
            projection_start,
            profile_start,
            backend_name,
            None,
            None,
        );
    }
    let z_flat = projected.clone().slice_dim(2, 0..d_inner);
    let x_flat = projected.clone().slice_dim(2, d_inner..(2 * d_inner));
    let b_flat = projected
        .clone()
        .slice_dim(2, (2 * d_inner)..(2 * d_inner + ngroups * d_state));
    let c_flat = projected.clone().slice_dim(
        2,
        (2 * d_inner + ngroups * d_state)..(2 * d_inner + 2 * ngroups * d_state),
    );
    let dd_dt = projected.clone().slice_dim(
        2,
        (2 * d_inner + 2 * ngroups * d_state)..(2 * d_inner + 2 * ngroups * d_state + nheads),
    );
    let dd_a = projected.clone().slice_dim(
        2,
        (2 * d_inner + 2 * ngroups * d_state + nheads)
            ..(2 * d_inner + 2 * ngroups * d_state + 2 * nheads),
    );
    let trap = projected.clone().slice_dim(
        2,
        (2 * d_inner + 2 * ngroups * d_state + 2 * nheads)
            ..(2 * d_inner + 2 * ngroups * d_state + 3 * nheads),
    );
    let angle_proj_shared = projected.clone().slice_dim(
        2,
        (2 * d_inner + 2 * ngroups * d_state + 3 * nheads)..in_proj_dim,
    );

    let z = z_flat.reshape([batch, time, nheads, headdim]);
    let x = x_flat.reshape([batch, time, nheads, headdim]);
    let b_input_full = b_flat.reshape([batch, time, ngroups, d_state]);
    let c_input_full = c_flat.reshape([batch, time, ngroups, d_state]);
    let bc_start = Instant::now();
    let (k_pre, b_inv_rms_full) = try_fused_group_rmsnorm_expand_bias_forward(
        b_input_full.clone(),
        b_norm_weight.clone(),
        b_bias.clone(),
        nheads,
        norm_eps,
    )
    .unwrap_or_else(|| {
        let (b, b_inv_rms_full) =
            rmsnorm_last_dim_forward_4d(b_input_full.clone(), b_norm_weight.clone(), norm_eps);
        let b_heads = repeat_groups_to_heads_4d(b, nheads);
        (
            b_heads + b_bias.reshape([1, 1, nheads, d_state]),
            b_inv_rms_full,
        )
    });
    if let Some(backend_name) = backend_name {
        log_mamba3_forward_profile(
            "bc_expand_norm",
            bc_start,
            profile_start,
            backend_name,
            None,
            None,
        );
    }
    let qc_start = Instant::now();
    let (q_pre, c_inv_rms_full) = try_fused_group_rmsnorm_expand_bias_forward(
        c_input_full.clone(),
        c_norm_weight.clone(),
        c_bias.clone(),
        nheads,
        norm_eps,
    )
    .unwrap_or_else(|| {
        let (c, c_inv_rms_full) =
            rmsnorm_last_dim_forward_4d(c_input_full.clone(), c_norm_weight.clone(), norm_eps);
        let c_heads = repeat_groups_to_heads_4d(c, nheads);
        (
            c_heads + c_bias.reshape([1, 1, nheads, d_state]),
            c_inv_rms_full,
        )
    });
    if let Some(backend_name) = backend_name {
        log_mamba3_forward_profile(
            "qc_expand_norm",
            qc_start,
            profile_start,
            backend_name,
            None,
            None,
        );
    }
    let setup_start = Instant::now();
    let dt_pre = dt_bias.reshape([1, 1, nheads]) + dd_dt;
    let dt = activation::softplus(dt_pre.clone(), 1.0);
    let a_unclamped = activation::softplus(dd_a, 1.0).neg();
    let a = a_unclamped.clone().clamp_max(-a_floor);
    let trap = activation::sigmoid(trap);
    let angle_proj = angle_proj_shared
        .reshape([batch, time, 1, num_rope_angles])
        .repeat_dim(2, nheads);
    let gamma = dt.clone() * trap.clone();
    let shifted_gamma = if time > 1 {
        Tensor::cat(
            vec![
                dt.clone().slice_dim(1, 1..time)
                    * trap.clone().slice_dim(1, 1..time).neg().add_scalar(1.0),
                Tensor::<B, 3>::zeros([batch, 1, nheads], &device),
            ],
            1,
        )
    } else {
        Tensor::<B, 3>::zeros([batch, 1, nheads], &device)
    };
    let scale = gamma.clone() + shifted_gamma;
    if let Some(backend_name) = backend_name {
        log_mamba3_forward_profile(
            "setup_scalars",
            setup_start,
            profile_start,
            backend_name,
            None,
            None,
        );
    }

    let mut ssm_state = state
        .as_ref()
        .map(|state| state.ssm.clone())
        .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, nheads, headdim, d_state], &device));
    let mut angle_state = state
        .as_ref()
        .map(|state| state.angle.clone())
        .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, nheads, num_rope_angles], &device));
    let mut k_state = state
        .as_ref()
        .map(|state| state.k.clone())
        .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, nheads, d_state], &device));
    let mut v_state = state
        .as_ref()
        .map(|state| state.v.clone())
        .unwrap_or_else(|| Tensor::<B, 3>::zeros([batch, nheads, headdim], &device));

    let mut outputs = Vec::with_capacity(time);
    #[allow(unused_mut)]
    let mut chunk_traces = if capture_trace {
        Some(Vec::with_capacity(time.div_ceil(chunk_size)))
    } else {
        None
    };
    let d_skip = d_skip.reshape([1, nheads, 1, 1]);
    for chunk_start in (0..time).step_by(chunk_size) {
        let chunk_profile_start = Instant::now();
        let chunk_end = (chunk_start + chunk_size).min(time);
        let chunk_len = chunk_end - chunk_start;

        let slice_start = Instant::now();
        let z_chunk = z.clone().slice_dim(1, chunk_start..chunk_end);
        let x_chunk = x.clone().slice_dim(1, chunk_start..chunk_end);
        let b_input_chunk = b_input_full.clone().slice_dim(1, chunk_start..chunk_end);
        let c_input_chunk = c_input_full.clone().slice_dim(1, chunk_start..chunk_end);
        let b_inv_rms_chunk = b_inv_rms_full.clone().slice_dim(1, chunk_start..chunk_end);
        let c_inv_rms_chunk = c_inv_rms_full.clone().slice_dim(1, chunk_start..chunk_end);
        let q_pre_chunk = q_pre.clone().slice_dim(1, chunk_start..chunk_end);
        let k_pre_chunk = k_pre.clone().slice_dim(1, chunk_start..chunk_end);
        let dt_pre_chunk = dt_pre.clone().slice_dim(1, chunk_start..chunk_end);
        let dt_chunk = dt.clone().slice_dim(1, chunk_start..chunk_end);
        let a_unclamped_chunk = a_unclamped.clone().slice_dim(1, chunk_start..chunk_end);
        let a_chunk = a.clone().slice_dim(1, chunk_start..chunk_end);
        let trap_chunk = trap.clone().slice_dim(1, chunk_start..chunk_end);
        let gamma_chunk = gamma.clone().slice_dim(1, chunk_start..chunk_end);
        let scale_chunk = scale.clone().slice_dim(1, chunk_start..chunk_end);
        let angle_proj_chunk = angle_proj.clone().slice_dim(1, chunk_start..chunk_end);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "chunk_slice",
                slice_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }

        let state_tilde_start = Instant::now();
        let state_tilde = ssm_state.clone()
            + v_state.clone().unsqueeze_dim::<4>(3)
                * k_state.clone().unsqueeze_dim::<4>(2)
                * (dt_chunk
                    .clone()
                    .slice_dim(1, 0..1)
                    .reshape([batch, nheads, 1])
                    * trap_chunk
                        .clone()
                        .slice_dim(1, 0..1)
                        .reshape([batch, nheads, 1])
                        .neg()
                        .add_scalar(1.0))
                .reshape([batch, nheads, 1, 1]);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "state_tilde",
                state_tilde_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }

        let rotary_start = Instant::now();
        let tanh_angle = tanh_reference(angle_proj_chunk);
        let angle_delta = tanh_angle.clone() * dt_chunk.clone().unsqueeze_dim::<4>(3) * PI;
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "angle_delta",
                rotary_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        let angle_prefix_start = Instant::now();
        let angle_chunk = angle_delta.cumsum(1)
            + angle_state
                .clone()
                .reshape([batch, 1, nheads, num_rope_angles]);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "angle_prefix",
                angle_prefix_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        let rotary_kernel_start = Instant::now();
        let (q_rot_chunk, k_chunk, qk_dot_chunk, k_rot_chunk_opt, qk_inner_opt) =
            if let Some((q_rot_chunk, k_scaled_chunk, qk_dot_chunk)) =
                try_tensorized_mamba3_preprocess(
                    q_pre_chunk.clone(),
                    k_pre_chunk.clone(),
                    angle_chunk.clone(),
                    gamma_chunk.clone(),
                    scale_chunk.clone(),
                    capture_trace,
                )
            {
                (
                    q_rot_chunk.clone(),
                    k_scaled_chunk.swap_dims(1, 2),
                    qk_dot_chunk,
                    None,
                    None,
                )
            } else {
                let (q_rot_chunk, k_rot_chunk) = rotate_pairwise_qk_with_angles(
                    q_pre_chunk.clone(),
                    k_pre_chunk.clone(),
                    angle_chunk.clone(),
                    num_rope_angles,
                );
                let qk_inner = (q_pre_chunk.clone() * k_pre_chunk.clone())
                    .sum_dim(3)
                    .reshape([batch, chunk_len, nheads]);
                let qk_dot_chunk = qk_inner.clone() * gamma_chunk.clone();
                (
                    q_rot_chunk,
                    (k_rot_chunk.clone() * scale_chunk.clone().unsqueeze_dim::<4>(3))
                        .swap_dims(1, 2),
                    qk_dot_chunk,
                    Some(k_rot_chunk),
                    Some(qk_inner),
                )
            };
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "rotary_kernel",
                rotary_kernel_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        let qk_pack_start = Instant::now();
        let q_chunk = q_rot_chunk.swap_dims(1, 2);
        let v_chunk = x_chunk.clone().swap_dims(1, 2);
        let z_chunk = z_chunk.swap_dims(1, 2);
        let da_chunk = (a_chunk.clone() * dt_chunk.clone()).swap_dims(1, 2);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "qk_pack",
                qk_pack_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        let da_prefix_start = Instant::now();
        let da_prefix = da_chunk.clone().cumsum(2);
        let exp_da_prefix = da_prefix.clone().exp();
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "da_prefix_exp",
                da_prefix_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "rotary_and_prefix",
                rotary_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }

        let prev_out_start = Instant::now();
        let prev_out = q_chunk.clone().matmul(state_tilde.clone().swap_dims(2, 3))
            * exp_da_prefix.clone().unsqueeze_dim::<4>(3);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "prev_out",
                prev_out_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }

        let current_out_start = Instant::now();
        let current_out = try_tensorized_mamba3_current_score(
            q_chunk.clone(),
            k_chunk.clone(),
            v_chunk.clone(),
            da_prefix.clone(),
            capture_trace,
        )
        .unwrap_or_else(|| {
            let decay = (da_prefix.clone().unsqueeze_dim::<4>(3)
                - da_prefix.clone().unsqueeze_dim::<4>(2))
            .clamp_max(0.0)
            .exp();
            let raw_scores = q_chunk.clone().matmul(k_chunk.clone().swap_dims(2, 3));
            let current_scores = raw_scores * decay;
            current_scores.tril(-1).matmul(v_chunk.clone())
        });
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "current_out",
                current_out_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
        let output_start = Instant::now();
        let y_pre = prev_out
            + current_out.clone()
            + (d_skip.clone()
                + qk_dot_chunk
                    .clone()
                    .swap_dims(1, 2)
                    .reshape([batch, nheads, chunk_len, 1]))
                * v_chunk.clone();
        let y_chunk = silu(z_chunk.clone()) * y_pre.clone();
        outputs.push(
            y_chunk
                .swap_dims(1, 2)
                .reshape([batch * chunk_len, d_inner])
                .matmul(out_proj.clone())
                .reshape([batch, 1, chunk_len, d_model]),
        );
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "output_proj",
                output_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }

        if let Some(traces) = chunk_traces.as_mut() {
            let trace_start = Instant::now();
            let k_rot_chunk = k_rot_chunk_opt
                .clone()
                .expect("mamba3 trace capture expects explicit rotary chunk");
            let qk_inner = qk_inner_opt
                .clone()
                .expect("mamba3 trace capture expects explicit qk inner");
            traces.push(Mamba3ChunkTrace {
                chunk_start,
                chunk_end,
                k_state: k_state.clone().into_primitive().tensor(),
                v_state: v_state.clone().into_primitive().tensor(),
                q_pre: q_pre_chunk.clone().into_primitive().tensor(),
                k_pre: k_pre_chunk.clone().into_primitive().tensor(),
                b_input: b_input_chunk.clone().into_primitive().tensor(),
                c_input: c_input_chunk.clone().into_primitive().tensor(),
                b_inv_rms: b_inv_rms_chunk.clone().into_primitive().tensor(),
                c_inv_rms: c_inv_rms_chunk.clone().into_primitive().tensor(),
                dt_pre: dt_pre_chunk.clone().into_primitive().tensor(),
                dt: dt_chunk.clone().into_primitive().tensor(),
                a_unclamped: a_unclamped_chunk.clone().into_primitive().tensor(),
                a: a_chunk.clone().into_primitive().tensor(),
                trap: trap_chunk.clone().into_primitive().tensor(),
                gamma: gamma_chunk.clone().into_primitive().tensor(),
                scale: scale_chunk.clone().into_primitive().tensor(),
                tanh_angle: tanh_angle.clone().into_primitive().tensor(),
                angle_chunk: angle_chunk.clone().into_primitive().tensor(),
                q_head: q_chunk.clone().into_primitive().tensor(),
                k_rot_chunk: k_rot_chunk.clone().into_primitive().tensor(),
                k_head: k_chunk.clone().into_primitive().tensor(),
                v_head: v_chunk.clone().into_primitive().tensor(),
                z_head: z_chunk.clone().into_primitive().tensor(),
                state_tilde: state_tilde.clone().into_primitive().tensor(),
                da_prefix: da_prefix.clone().into_primitive().tensor(),
                exp_da_prefix: exp_da_prefix.clone().into_primitive().tensor(),
                qk_inner: qk_inner.clone().into_primitive().tensor(),
                y_pre: y_pre.clone().into_primitive().tensor(),
            });
            if let Some(backend_name) = backend_name {
                log_mamba3_forward_profile(
                    "trace_capture",
                    trace_start,
                    chunk_profile_start,
                    backend_name,
                    Some(chunk_start),
                    Some(chunk_end),
                );
            }
        }

        let state_update_start = Instant::now();
        let da_last = da_prefix
            .clone()
            .slice_dim(2, chunk_len - 1..chunk_len)
            .reshape([batch, nheads]);
        ssm_state = try_tensorized_mamba3_state_update(
            state_tilde.clone(),
            da_prefix.clone(),
            v_chunk.clone(),
            k_chunk.clone(),
            capture_trace,
        )
        .unwrap_or_else(|| {
            let weighted_v = v_chunk.clone()
                * (da_last.clone().unsqueeze_dim::<3>(2) - da_prefix.clone())
                    .exp()
                    .unsqueeze_dim::<4>(3);
            state_tilde.clone() * da_last.clone().reshape([batch, nheads, 1, 1]).exp()
                + weighted_v.swap_dims(2, 3).matmul(k_chunk.clone())
        });
        angle_state = angle_chunk.slice_dim(1, chunk_len - 1..chunk_len).reshape([
            batch,
            nheads,
            num_rope_angles,
        ]);
        k_state = if let Some(k_rot_chunk) = k_rot_chunk_opt {
            k_rot_chunk
                .slice_dim(1, chunk_len - 1..chunk_len)
                .reshape([batch, nheads, d_state])
        } else {
            let k_scaled_last = k_chunk
                .clone()
                .slice_dim(2, chunk_len - 1..chunk_len)
                .reshape([batch, nheads, d_state]);
            let scale_last = scale_chunk
                .clone()
                .slice_dim(1, chunk_len - 1..chunk_len)
                .reshape([batch, nheads, 1]);
            k_scaled_last / scale_last
        };
        v_state = x_chunk
            .slice_dim(1, chunk_len - 1..chunk_len)
            .reshape([batch, nheads, headdim]);
        if let Some(backend_name) = backend_name {
            log_mamba3_forward_profile(
                "state_update",
                state_update_start,
                chunk_profile_start,
                backend_name,
                Some(chunk_start),
                Some(chunk_end),
            );
        }
    }

    let output = Mamba3TensorizedOutput {
        context: Tensor::cat(outputs, 2),
        state: Mamba3TensorizedState {
            ssm: ssm_state,
            angle: angle_state,
            k: k_state,
            v: v_state,
        },
    };
    let trace = chunk_traces.map(|chunks| Mamba3ForwardTrace { chunks });
    (output, trace)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    #[test]
    fn recommended_cuda_short_context_chunk_size_tracks_short_block_cap() {
        assert_eq!(recommended_cuda_short_context_chunk_size(64, 64), 64);
        assert_eq!(recommended_cuda_short_context_chunk_size(64, 128), 128);
        assert_eq!(recommended_cuda_short_context_chunk_size(64, 256), 256);
        assert_eq!(recommended_cuda_short_context_chunk_size(64, 512), 256);
        assert_eq!(recommended_cuda_short_context_chunk_size(256, 512), 256);
        assert_eq!(recommended_cuda_short_context_chunk_size(512, 512), 512);
    }

    #[test]
    fn recommended_wgpu_short_context_chunk_size_tracks_short_block_cap() {
        assert_eq!(recommended_wgpu_short_context_chunk_size(32, 32), 32);
        assert_eq!(recommended_wgpu_short_context_chunk_size(32, 64), 64);
        assert_eq!(recommended_wgpu_short_context_chunk_size(32, 128), 128);
        assert_eq!(recommended_wgpu_short_context_chunk_size(32, 256), 128);
        assert_eq!(recommended_wgpu_short_context_chunk_size(128, 256), 128);
        assert_eq!(recommended_wgpu_short_context_chunk_size(256, 256), 256);
    }

    fn deterministic_tensor<const D: usize>(
        shape: [usize; D],
        period: usize,
    ) -> Tensor<Backend, D> {
        let len = shape.iter().product::<usize>();
        Tensor::<Backend, D>::from_data(
            TensorData::new(
                (0..len)
                    .map(|idx| ((idx % period) as f32) / period as f32 - 0.5)
                    .collect::<Vec<_>>(),
                shape,
            ),
            &Default::default(),
        )
    }

    fn tensor_max_abs_diff<const D: usize>(
        lhs: Tensor<Backend, D>,
        rhs: Tensor<Backend, D>,
    ) -> f32 {
        let lhs = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs");
        let rhs = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs");
        lhs.into_iter()
            .zip(rhs)
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn tensorized_mamba3_chunked_state_matches_full_sequence() {
        let batch = 1;
        let time = 32;
        let split = 11;
        let d_model = 128;
        let d_inner = 256;
        let d_state = 16;
        let headdim = 64;
        let ngroups = 4;
        let nheads = d_inner / headdim;
        let num_rope_angles = 4;
        let hidden = deterministic_tensor([batch, 1, time, d_model], 257);
        let in_proj = deterministic_tensor(
            [
                d_model,
                2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles,
            ],
            263,
        );
        let dt_bias = deterministic_tensor([nheads], 269);
        let b_bias = deterministic_tensor([nheads, d_state], 271);
        let c_bias = deterministic_tensor([nheads, d_state], 277);
        let b_norm_weight = deterministic_tensor([d_state], 281);
        let c_norm_weight = deterministic_tensor([d_state], 283);
        let d_skip = deterministic_tensor([nheads], 293);
        let out_proj = deterministic_tensor([d_inner, d_model], 307);

        let full = tensorized_mamba3_forward(
            hidden.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            64,
            in_proj.clone(),
            dt_bias.clone(),
            b_bias.clone(),
            c_bias.clone(),
            b_norm_weight.clone(),
            c_norm_weight.clone(),
            d_skip.clone(),
            out_proj.clone(),
            None::<Mamba3TensorizedState<Backend>>,
        );
        let prefix = tensorized_mamba3_forward(
            hidden.clone().slice_dim(2, 0..split),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            64,
            in_proj.clone(),
            dt_bias.clone(),
            b_bias.clone(),
            c_bias.clone(),
            b_norm_weight.clone(),
            c_norm_weight.clone(),
            d_skip.clone(),
            out_proj.clone(),
            None::<Mamba3TensorizedState<Backend>>,
        );
        let suffix = tensorized_mamba3_forward(
            hidden.slice_dim(2, split..time),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            64,
            in_proj,
            dt_bias,
            b_bias,
            c_bias,
            b_norm_weight,
            c_norm_weight,
            d_skip,
            out_proj,
            Some(prefix.state),
        );
        let chunked_context = Tensor::cat(vec![prefix.context, suffix.context], 2);
        assert!(tensor_max_abs_diff(full.context, chunked_context) <= 2.0e-3);
        assert!(tensor_max_abs_diff(full.state.ssm, suffix.state.ssm) <= 2.0e-3);
        assert!(tensor_max_abs_diff(full.state.angle, suffix.state.angle) <= 2.0e-4);
        assert!(tensor_max_abs_diff(full.state.k, suffix.state.k) <= 2.0e-4);
        assert!(tensor_max_abs_diff(full.state.v, suffix.state.v) <= 2.0e-4);
    }
}

#[cfg(test)]
mod wgpu_tests {
    use super::*;
    use crate::kernels::sequence::mamba3::backward::{
        set_mamba3_wgpu_bc_backward_runtime_for_tests,
        set_mamba3_wgpu_rotary_backward_runtime_for_tests,
    };
    use burn::tensor::{ElementConversion, TensorData};
    use burn_wgpu::{RuntimeOptions, graphics};

    fn init_runtime(device: &<WgpuCubeAutodiffBackend as BackendTrait>::Device) {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn assert_close_backend<B: BackendTrait, const D: usize>(
        lhs: Tensor<B, D>,
        rhs: Tensor<B, D>,
        atol: f32,
        rtol: f32,
    ) {
        let max_rhs = rhs.clone().abs().max().into_scalar().elem::<f32>();
        let max_diff = lhs.sub(rhs).abs().max().into_scalar().elem::<f32>();
        let max_tol = atol + rtol * max_rhs;
        assert!(
            max_diff <= max_tol,
            "max difference {max_diff} exceeds tolerance {max_tol} (rhs max {max_rhs})"
        );
    }

    fn run_tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_autodiff(
        use_bc_runtime: Option<bool>,
        use_rotary_runtime: Option<bool>,
    ) {
        unsafe {
            std::env::set_var("RUST_BACKTRACE", "1");
        }
        set_mamba3_wgpu_bc_backward_runtime_for_tests(use_bc_runtime);
        set_mamba3_wgpu_rotary_backward_runtime_for_tests(use_rotary_runtime);
        let device = <WgpuCubeAutodiffBackend as BackendTrait>::Device::default();
        init_runtime(&device);

        let batch = 1;
        let time = 16;
        let d_model = 64;
        let d_inner = 128;
        let d_state = 8;
        let headdim = 32;
        let ngroups = 4;
        let nheads = d_inner / headdim;
        let num_rope_angles = 4;
        let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * in_proj_dim))
                .map(|idx| ((idx % 263) as f32) / 263.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_model, in_proj_dim],
        );
        let dt_bias_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 269) as f32) / 269.0 - 0.35)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let b_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 271) as f32) / 271.0 - 0.4)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let c_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 277) as f32) / 277.0 - 0.42)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let b_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 281) as f32) / 281.0 + 0.9)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let c_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 283) as f32) / 283.0 + 0.85)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let d_skip_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 293) as f32) / 293.0 + 0.75)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 307) as f32) / 307.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 311) as f32) / 311.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let initial_angle_data = TensorData::new(
            (0..(batch * nheads * num_rope_angles))
                .map(|idx| ((idx % 313) as f32) / 313.0 - 0.15)
                .collect::<Vec<_>>(),
            [batch, nheads, num_rope_angles],
        );
        let initial_k_data = TensorData::new(
            (0..(batch * nheads * d_state))
                .map(|idx| ((idx % 317) as f32) / 317.0 - 0.2)
                .collect::<Vec<_>>(),
            [batch, nheads, d_state],
        );
        let initial_v_data = TensorData::new(
            (0..(batch * nheads * headdim))
                .map(|idx| ((idx % 331) as f32) / 331.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 337) as f32) / 337.0 - 0.35)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_graph =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_graph =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let dt_bias_graph =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(dt_bias_data.clone(), &device)
                .require_grad();
        let b_bias_graph =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(b_bias_data.clone(), &device)
                .require_grad();
        let c_bias_graph =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(c_bias_data.clone(), &device)
                .require_grad();
        let b_norm_weight_graph =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(b_norm_weight_data.clone(), &device)
                .require_grad();
        let c_norm_weight_graph =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(c_norm_weight_data.clone(), &device)
                .require_grad();
        let d_skip_graph =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(d_skip_data.clone(), &device)
                .require_grad();
        let out_proj_graph =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();

        let hidden_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(in_proj_data, &device).require_grad();
        let dt_bias_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(dt_bias_data, &device).require_grad();
        let b_bias_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(b_bias_data, &device).require_grad();
        let c_bias_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(c_bias_data, &device).require_grad();
        let b_norm_weight_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(b_norm_weight_data, &device)
                .require_grad();
        let c_norm_weight_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(c_norm_weight_data, &device)
                .require_grad();
        let d_skip_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(d_skip_data, &device).require_grad();
        let out_proj_wrapper =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(out_proj_data, &device).require_grad();

        let graph = tensorized_mamba3_forward_direct_graph(
            hidden_graph.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            32,
            in_proj_graph.clone(),
            dt_bias_graph.clone(),
            b_bias_graph.clone(),
            c_bias_graph.clone(),
            b_norm_weight_graph.clone(),
            c_norm_weight_graph.clone(),
            d_skip_graph.clone(),
            out_proj_graph.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
                    initial_ssm_data.clone(),
                    &device,
                ),
                angle: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
                    initial_angle_data.clone(),
                    &device,
                ),
                k: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_k_data.clone(), &device),
                v: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_v_data.clone(), &device),
            }),
        );
        let wrapped = tensorized_mamba3_forward_custom_backward(
            hidden_wrapper.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            32,
            in_proj_wrapper.clone(),
            dt_bias_wrapper.clone(),
            b_bias_wrapper.clone(),
            c_bias_wrapper.clone(),
            b_norm_weight_wrapper.clone(),
            c_norm_weight_wrapper.clone(),
            d_skip_wrapper.clone(),
            out_proj_wrapper.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(initial_ssm_data, &device),
                angle: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_angle_data, &device),
                k: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_k_data, &device),
                v: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_v_data, &device),
            }),
        );

        let _ = <WgpuCubeAutodiffBackend as BackendTrait>::sync(&device);
        assert_close_backend(
            graph.context.clone(),
            wrapped.context.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.ssm.clone(),
            wrapped.state.ssm.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.angle.clone(),
            wrapped.state.angle.clone(),
            5.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            graph.state.k.clone(),
            wrapped.state.k.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.v.clone(),
            wrapped.state.v.clone(),
            5.0e-3,
            5.0e-3,
        );

        let output_weights =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(output_weight_data, &device);
        let graph_grads = (graph.context * output_weights.clone()).sum().backward();
        let wrapper_grads = (wrapped.context * output_weights).sum().backward();
        let _ = <WgpuCubeAutodiffBackend as BackendTrait>::sync(&device);

        assert_close_backend(
            hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
            hidden_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper hidden grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            in_proj_graph
                .grad(&graph_grads)
                .expect("graph in_proj grad"),
            in_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper in_proj grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            out_proj_graph
                .grad(&graph_grads)
                .expect("graph out_proj grad"),
            out_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper out_proj grad"),
            1.0e-2,
            1.0e-2,
        );

        set_mamba3_wgpu_bc_backward_runtime_for_tests(None);
        set_mamba3_wgpu_rotary_backward_runtime_for_tests(None);
    }

    #[test]
    fn tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_autodiff() {
        run_tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_autodiff(None, None);
    }

    #[test]
    fn wgpu_train_wrapper_defaults_to_direct_graph() {
        let device = <WgpuCubeBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        let single_chunk = Tensor::<WgpuCubeBackend, 4>::zeros([1, 1, 32, 8], &device);
        let multi_chunk = Tensor::<WgpuCubeBackend, 4>::zeros([1, 1, 64, 8], &device);

        assert!(!use_tensorized_mamba3_wgpu_train_wrapper(&single_chunk));
        assert!(!use_tensorized_mamba3_wgpu_train_wrapper(&multi_chunk));
    }

    #[test]
    fn current_score_runtime_defaults_off() {
        WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);
        assert!(!use_mamba3_wgpu_current_score_runtime());
    }

    #[test]
    fn state_update_runtime_defaults_off() {
        WGPU_STATE_UPDATE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);
        assert!(!use_mamba3_wgpu_state_update_runtime());
    }

    #[test]
    fn state_update_runtime_heuristic_prefers_long_wgpu_chunks() {
        let device = <WgpuCubeBackend as BackendTrait>::Device::default();
        init_runtime(&device);
        WGPU_STATE_UPDATE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);
        let short = Tensor::<WgpuCubeBackend, 3>::zeros([1, 2, 32], &device);
        let long = Tensor::<WgpuCubeBackend, 3>::zeros([1, 2, 64], &device);
        assert!(!should_use_mamba3_wgpu_state_update_runtime(&short));
        assert!(should_use_mamba3_wgpu_state_update_runtime(&long));
    }

    #[test]
    fn wgpu_current_score_runtime_matches_direct_graph_reference() {
        let device = <WgpuCubeAutodiffBackend as BackendTrait>::Device::default();
        init_runtime(&device);

        let batch = 1;
        let time = 16;
        let d_model = 64;
        let d_inner = 128;
        let d_state = 8;
        let headdim = 32;
        let ngroups = 4;
        let nheads = d_inner / headdim;
        let num_rope_angles = 4;
        let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * in_proj_dim))
                .map(|idx| ((idx % 263) as f32) / 263.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_model, in_proj_dim],
        );
        let dt_bias_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 269) as f32) / 269.0 - 0.35)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let b_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 271) as f32) / 271.0 - 0.4)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let c_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 277) as f32) / 277.0 - 0.42)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let b_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 281) as f32) / 281.0 + 0.9)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let c_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 283) as f32) / 283.0 + 0.85)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let d_skip_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 293) as f32) / 293.0 + 0.75)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 307) as f32) / 307.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 311) as f32) / 311.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let initial_angle_data = TensorData::new(
            (0..(batch * nheads * num_rope_angles))
                .map(|idx| ((idx % 313) as f32) / 313.0 - 0.15)
                .collect::<Vec<_>>(),
            [batch, nheads, num_rope_angles],
        );
        let initial_k_data = TensorData::new(
            (0..(batch * nheads * d_state))
                .map(|idx| ((idx % 317) as f32) / 317.0 - 0.2)
                .collect::<Vec<_>>(),
            [batch, nheads, d_state],
        );
        let initial_v_data = TensorData::new(
            (0..(batch * nheads * headdim))
                .map(|idx| ((idx % 331) as f32) / 331.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 337) as f32) / 337.0 - 0.35)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_ref =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_ref =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let out_proj_ref =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();

        let hidden_runtime =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_runtime =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(in_proj_data, &device).require_grad();
        let out_proj_runtime =
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(out_proj_data, &device).require_grad();

        WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE.store(0, Ordering::Relaxed);
        let reference = tensorized_mamba3_forward_direct_graph(
            hidden_ref.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            32,
            in_proj_ref.clone(),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(dt_bias_data.clone(), &device),
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(b_bias_data.clone(), &device),
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(c_bias_data.clone(), &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(b_norm_weight_data.clone(), &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(c_norm_weight_data.clone(), &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(d_skip_data.clone(), &device),
            out_proj_ref.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
                    initial_ssm_data.clone(),
                    &device,
                ),
                angle: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
                    initial_angle_data.clone(),
                    &device,
                ),
                k: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_k_data.clone(), &device),
                v: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_v_data.clone(), &device),
            }),
        );

        WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE.store(1, Ordering::Relaxed);
        let runtime = tensorized_mamba3_forward_direct_graph(
            hidden_runtime.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            32,
            in_proj_runtime.clone(),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(dt_bias_data, &device),
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(b_bias_data, &device),
            Tensor::<WgpuCubeAutodiffBackend, 2>::from_data(c_bias_data, &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(b_norm_weight_data, &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(c_norm_weight_data, &device),
            Tensor::<WgpuCubeAutodiffBackend, 1>::from_data(d_skip_data, &device),
            out_proj_runtime.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(initial_ssm_data, &device),
                angle: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_angle_data, &device),
                k: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_k_data, &device),
                v: Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(initial_v_data, &device),
            }),
        );
        WGPU_CURRENT_SCORE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);

        let output_weight =
            Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(output_weight_data, &device);
        let reference_grads = (reference.context.clone() * output_weight.clone())
            .sum()
            .backward();
        let runtime_grads = (runtime.context.clone() * output_weight).sum().backward();

        assert_close_backend(reference.context, runtime.context, 1.0e-4, 5.0e-4);
        assert_close_backend(reference.state.ssm, runtime.state.ssm, 1.0e-4, 5.0e-4);
        assert_close_backend(reference.state.angle, runtime.state.angle, 1.0e-4, 5.0e-4);
        assert_close_backend(reference.state.k, runtime.state.k, 1.0e-4, 5.0e-4);
        assert_close_backend(reference.state.v, runtime.state.v, 1.0e-4, 5.0e-4);
        assert_close_backend(
            hidden_ref
                .grad(&reference_grads)
                .expect("reference hidden grad"),
            hidden_runtime
                .grad(&runtime_grads)
                .expect("runtime hidden grad"),
            1.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            in_proj_ref
                .grad(&reference_grads)
                .expect("reference in proj grad"),
            in_proj_runtime
                .grad(&runtime_grads)
                .expect("runtime in proj grad"),
            1.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            out_proj_ref
                .grad(&reference_grads)
                .expect("reference out proj grad"),
            out_proj_runtime
                .grad(&runtime_grads)
                .expect("runtime out proj grad"),
            1.0e-4,
            5.0e-4,
        );
    }

    #[test]
    fn wgpu_preprocess_runtime_matches_direct_graph_reference() {
        let device = <WgpuCubeAutodiffBackend as BackendTrait>::Device::default();
        init_runtime(&device);

        let batch = 1;
        let time = 8;
        let heads = 3;
        let width = 8;
        let num_rope_angles = 3;

        let q_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * heads * width))
                    .map(|idx| ((idx % 97) as f32) / 97.0 - 0.4)
                    .collect::<Vec<_>>(),
                [batch, time, heads, width],
            ),
            &device,
        )
        .require_grad();
        let k_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * heads * width))
                    .map(|idx| ((idx % 89) as f32) / 89.0 - 0.3)
                    .collect::<Vec<_>>(),
                [batch, time, heads, width],
            ),
            &device,
        )
        .require_grad();
        let angles_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * heads * num_rope_angles))
                    .map(|idx| ((idx % 83) as f32) / 83.0 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, time, heads, num_rope_angles],
            ),
            &device,
        )
        .require_grad();
        let gamma_ref = Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * time * heads))
                    .map(|idx| ((idx % 79) as f32) / 79.0 + 0.2)
                    .collect::<Vec<_>>(),
                [batch, time, heads],
            ),
            &device,
        )
        .require_grad();
        let scale_ref = Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * time * heads))
                    .map(|idx| ((idx % 73) as f32) / 73.0 + 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, heads],
            ),
            &device,
        )
        .require_grad();

        let q_runtime = q_ref.clone().detach().require_grad();
        let k_runtime = k_ref.clone().detach().require_grad();
        let angles_runtime = angles_ref.clone().detach().require_grad();
        let gamma_runtime = gamma_ref.clone().detach().require_grad();
        let scale_runtime = scale_ref.clone().detach().require_grad();

        WGPU_PREPROCESS_RUNTIME_OVERRIDE.store(0, Ordering::Relaxed);
        let (q_rot_ref, k_rot_ref) = rotate_pairwise_qk_with_angles(
            q_ref.clone(),
            k_ref.clone(),
            angles_ref.clone(),
            num_rope_angles,
        );
        let k_scaled_ref = k_rot_ref.clone() * scale_ref.clone().unsqueeze_dim::<4>(3);
        let qk_dot_ref = (q_ref.clone() * k_ref.clone())
            .sum_dim(3)
            .reshape([batch, time, heads])
            * gamma_ref.clone();

        WGPU_PREPROCESS_RUNTIME_OVERRIDE.store(1, Ordering::Relaxed);
        let (q_rot_runtime, k_scaled_runtime, qk_dot_runtime) = try_tensorized_mamba3_preprocess(
            q_runtime.clone(),
            k_runtime.clone(),
            angles_runtime.clone(),
            gamma_runtime.clone(),
            scale_runtime.clone(),
            false,
        )
        .expect("wgpu preprocess runtime");
        WGPU_PREPROCESS_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);

        let q_weight = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * heads * width))
                    .map(|idx| ((idx % 67) as f32) / 67.0 - 0.1)
                    .collect::<Vec<_>>(),
                [batch, time, heads, width],
            ),
            &device,
        );
        let k_weight = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * heads * width))
                    .map(|idx| ((idx % 61) as f32) / 61.0 - 0.15)
                    .collect::<Vec<_>>(),
                [batch, time, heads, width],
            ),
            &device,
        );
        let qk_weight = Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * time * heads))
                    .map(|idx| ((idx % 59) as f32) / 59.0 - 0.05)
                    .collect::<Vec<_>>(),
                [batch, time, heads],
            ),
            &device,
        );

        let reference_grads = ((q_rot_ref.clone() * q_weight.clone()).sum()
            + (k_scaled_ref.clone() * k_weight.clone()).sum()
            + (qk_dot_ref.clone() * qk_weight.clone()).sum())
        .backward();
        let runtime_grads = ((q_rot_runtime.clone() * q_weight).sum()
            + (k_scaled_runtime.clone() * k_weight).sum()
            + (qk_dot_runtime.clone() * qk_weight).sum())
        .backward();

        assert_close_backend(q_rot_ref, q_rot_runtime, 1.0e-4, 5.0e-4);
        assert_close_backend(k_scaled_ref, k_scaled_runtime, 1.0e-4, 5.0e-4);
        assert_close_backend(qk_dot_ref, qk_dot_runtime, 1.0e-4, 5.0e-4);
        assert_close_backend(
            q_ref.grad(&reference_grads).expect("reference q grad"),
            q_runtime.grad(&runtime_grads).expect("runtime q grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            k_ref.grad(&reference_grads).expect("reference k grad"),
            k_runtime.grad(&runtime_grads).expect("runtime k grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            angles_ref
                .grad(&reference_grads)
                .expect("reference angle grad"),
            angles_runtime
                .grad(&runtime_grads)
                .expect("runtime angle grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            gamma_ref
                .grad(&reference_grads)
                .expect("reference gamma grad"),
            gamma_runtime
                .grad(&runtime_grads)
                .expect("runtime gamma grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            scale_ref
                .grad(&reference_grads)
                .expect("reference scale grad"),
            scale_runtime
                .grad(&runtime_grads)
                .expect("runtime scale grad"),
            1.0e-4,
            5.0e-4,
        );
    }

    #[test]
    fn wgpu_state_update_runtime_matches_direct_graph_reference() {
        let device = <WgpuCubeAutodiffBackend as BackendTrait>::Device::default();
        init_runtime(&device);

        let batch = 1;
        let heads = 2;
        let time = 6;
        let headdim = 4;
        let d_state = 3;

        let state_tilde_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..24)
                    .map(|idx| ((idx % 41) as f32) / 41.0 - 0.25)
                    .collect::<Vec<_>>(),
                [batch, heads, headdim, d_state],
            ),
            &device,
        )
        .require_grad();
        let da_prefix_ref = Tensor::<WgpuCubeAutodiffBackend, 3>::from_data(
            TensorData::new(
                (0..12)
                    .map(|idx| ((idx % 17) as f32) / 17.0 - 0.3)
                    .collect::<Vec<_>>(),
                [batch, heads, time],
            ),
            &device,
        )
        .require_grad();
        let v_head_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..48)
                    .map(|idx| ((idx % 29) as f32) / 29.0 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, heads, time, headdim],
            ),
            &device,
        )
        .require_grad();
        let k_head_ref = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..36)
                    .map(|idx| ((idx % 31) as f32) / 31.0 - 0.1)
                    .collect::<Vec<_>>(),
                [batch, heads, time, d_state],
            ),
            &device,
        )
        .require_grad();

        let state_tilde_runtime = state_tilde_ref.clone().detach().require_grad();
        let da_prefix_runtime = da_prefix_ref.clone().detach().require_grad();
        let v_head_runtime = v_head_ref.clone().detach().require_grad();
        let k_head_runtime = k_head_ref.clone().detach().require_grad();

        let da_last_ref = da_prefix_ref
            .clone()
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads]);
        let weighted_v_ref = v_head_ref.clone()
            * (da_last_ref.clone().unsqueeze_dim::<3>(2) - da_prefix_ref.clone())
                .exp()
                .unsqueeze_dim::<4>(3);
        let reference = state_tilde_ref.clone() * da_last_ref.reshape([batch, heads, 1, 1]).exp()
            + weighted_v_ref.swap_dims(2, 3).matmul(k_head_ref.clone());

        WGPU_STATE_UPDATE_RUNTIME_OVERRIDE.store(1, Ordering::Relaxed);
        let runtime = try_tensorized_mamba3_state_update(
            state_tilde_runtime.clone(),
            da_prefix_runtime.clone(),
            v_head_runtime.clone(),
            k_head_runtime.clone(),
            false,
        )
        .expect("runtime state update");
        WGPU_STATE_UPDATE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);

        let output_weight = Tensor::<WgpuCubeAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..24)
                    .map(|idx| ((idx % 37) as f32) / 37.0 - 0.15)
                    .collect::<Vec<_>>(),
                [batch, heads, headdim, d_state],
            ),
            &device,
        );
        let reference_grads = (reference.clone() * output_weight.clone()).sum().backward();
        let runtime_grads = (runtime.clone() * output_weight).sum().backward();

        assert_close_backend(reference, runtime, 1.0e-4, 5.0e-4);
        assert_close_backend(
            state_tilde_ref
                .grad(&reference_grads)
                .expect("reference state_tilde grad"),
            state_tilde_runtime
                .grad(&runtime_grads)
                .expect("runtime state_tilde grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            da_prefix_ref
                .grad(&reference_grads)
                .expect("reference da_prefix grad"),
            da_prefix_runtime
                .grad(&runtime_grads)
                .expect("runtime da_prefix grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            v_head_ref
                .grad(&reference_grads)
                .expect("reference v_head grad"),
            v_head_runtime
                .grad(&runtime_grads)
                .expect("runtime v_head grad"),
            1.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            k_head_ref
                .grad(&reference_grads)
                .expect("reference k_head grad"),
            k_head_runtime
                .grad(&runtime_grads)
                .expect("runtime k_head grad"),
            1.0e-4,
            5.0e-4,
        );
    }

    #[test]
    #[ignore = "diagnostic fallback check"]
    fn tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_without_bc_runtime() {
        run_tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_autodiff(
            Some(false),
            None,
        );
    }

    #[test]
    #[ignore = "diagnostic fallback check"]
    fn tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_without_rotary_runtime() {
        run_tensorized_mamba3_custom_backward_matches_direct_graph_on_wgpu_autodiff(
            None,
            Some(false),
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use burn::tensor::{ElementConversion, TensorData};

    fn assert_close_backend<B: BackendTrait, const D: usize>(
        lhs: Tensor<B, D>,
        rhs: Tensor<B, D>,
        atol: f32,
        rtol: f32,
    ) {
        let max_rhs = rhs.clone().abs().max().into_scalar().elem::<f32>();
        let max_diff = lhs.sub(rhs).abs().max().into_scalar().elem::<f32>();
        let max_tol = atol + rtol * max_rhs;
        assert!(
            max_diff <= max_tol,
            "max difference {max_diff} exceeds tolerance {max_tol} (rhs max {max_rhs})"
        );
    }

    #[test]
    fn tensorized_mamba3_custom_backward_matches_direct_graph_on_cuda_autodiff() {
        let device = <CudaCubeAutodiffBackend as BackendTrait>::Device::default();
        let batch = 1;
        let time = 32;
        let d_model = 128;
        let d_inner = 256;
        let d_state = 16;
        let headdim = 64;
        let ngroups = 4;
        let nheads = d_inner / headdim;
        let num_rope_angles = 4;
        let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + 3 * nheads + num_rope_angles;

        let hidden_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );
        let in_proj_data = TensorData::new(
            (0..(d_model * in_proj_dim))
                .map(|idx| ((idx % 263) as f32) / 263.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_model, in_proj_dim],
        );
        let dt_bias_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 269) as f32) / 269.0 - 0.35)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let b_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 271) as f32) / 271.0 - 0.4)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let c_bias_data = TensorData::new(
            (0..(nheads * d_state))
                .map(|idx| ((idx % 277) as f32) / 277.0 - 0.42)
                .collect::<Vec<_>>(),
            [nheads, d_state],
        );
        let b_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 281) as f32) / 281.0 + 0.9)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let c_norm_weight_data = TensorData::new(
            (0..d_state)
                .map(|idx| ((idx % 283) as f32) / 283.0 + 0.85)
                .collect::<Vec<_>>(),
            [d_state],
        );
        let d_skip_data = TensorData::new(
            (0..nheads)
                .map(|idx| ((idx % 293) as f32) / 293.0 + 0.75)
                .collect::<Vec<_>>(),
            [nheads],
        );
        let out_proj_data = TensorData::new(
            (0..(d_inner * d_model))
                .map(|idx| ((idx % 307) as f32) / 307.0 - 0.45)
                .collect::<Vec<_>>(),
            [d_inner, d_model],
        );
        let initial_ssm_data = TensorData::new(
            (0..(batch * nheads * headdim * d_state))
                .map(|idx| ((idx % 311) as f32) / 311.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim, d_state],
        );
        let initial_angle_data = TensorData::new(
            (0..(batch * nheads * num_rope_angles))
                .map(|idx| ((idx % 313) as f32) / 313.0 - 0.15)
                .collect::<Vec<_>>(),
            [batch, nheads, num_rope_angles],
        );
        let initial_k_data = TensorData::new(
            (0..(batch * nheads * d_state))
                .map(|idx| ((idx % 317) as f32) / 317.0 - 0.2)
                .collect::<Vec<_>>(),
            [batch, nheads, d_state],
        );
        let initial_v_data = TensorData::new(
            (0..(batch * nheads * headdim))
                .map(|idx| ((idx % 331) as f32) / 331.0 - 0.25)
                .collect::<Vec<_>>(),
            [batch, nheads, headdim],
        );
        let output_weight_data = TensorData::new(
            (0..(batch * time * d_model))
                .map(|idx| ((idx % 337) as f32) / 337.0 - 0.35)
                .collect::<Vec<_>>(),
            [batch, 1, time, d_model],
        );

        let hidden_graph =
            Tensor::<CudaCubeAutodiffBackend, 4>::from_data(hidden_data.clone(), &device)
                .require_grad();
        let in_proj_graph =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(in_proj_data.clone(), &device)
                .require_grad();
        let dt_bias_graph =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(dt_bias_data.clone(), &device)
                .require_grad();
        let b_bias_graph =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(b_bias_data.clone(), &device)
                .require_grad();
        let c_bias_graph =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(c_bias_data.clone(), &device)
                .require_grad();
        let b_norm_weight_graph =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(b_norm_weight_data.clone(), &device)
                .require_grad();
        let c_norm_weight_graph =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(c_norm_weight_data.clone(), &device)
                .require_grad();
        let d_skip_graph =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(d_skip_data.clone(), &device)
                .require_grad();
        let out_proj_graph =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(out_proj_data.clone(), &device)
                .require_grad();

        let hidden_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 4>::from_data(hidden_data, &device).require_grad();
        let in_proj_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(in_proj_data, &device).require_grad();
        let dt_bias_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(dt_bias_data, &device).require_grad();
        let b_bias_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(b_bias_data, &device).require_grad();
        let c_bias_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(c_bias_data, &device).require_grad();
        let b_norm_weight_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(b_norm_weight_data, &device)
                .require_grad();
        let c_norm_weight_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(c_norm_weight_data, &device)
                .require_grad();
        let d_skip_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 1>::from_data(d_skip_data, &device).require_grad();
        let out_proj_wrapper =
            Tensor::<CudaCubeAutodiffBackend, 2>::from_data(out_proj_data, &device).require_grad();

        let graph = tensorized_mamba3_forward_direct_graph(
            hidden_graph.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            64,
            in_proj_graph.clone(),
            dt_bias_graph.clone(),
            b_bias_graph.clone(),
            c_bias_graph.clone(),
            b_norm_weight_graph.clone(),
            c_norm_weight_graph.clone(),
            d_skip_graph.clone(),
            out_proj_graph.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<CudaCubeAutodiffBackend, 4>::from_data(
                    initial_ssm_data.clone(),
                    &device,
                ),
                angle: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(
                    initial_angle_data.clone(),
                    &device,
                ),
                k: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(initial_k_data.clone(), &device),
                v: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(initial_v_data.clone(), &device),
            }),
        );
        let wrapped = tensorized_mamba3_forward_custom_backward(
            hidden_wrapper.clone(),
            d_inner,
            d_state,
            headdim,
            ngroups,
            num_rope_angles,
            1.0e-5,
            1.0e-4,
            64,
            in_proj_wrapper.clone(),
            dt_bias_wrapper.clone(),
            b_bias_wrapper.clone(),
            c_bias_wrapper.clone(),
            b_norm_weight_wrapper.clone(),
            c_norm_weight_wrapper.clone(),
            d_skip_wrapper.clone(),
            out_proj_wrapper.clone(),
            Some(Mamba3TensorizedState {
                ssm: Tensor::<CudaCubeAutodiffBackend, 4>::from_data(initial_ssm_data, &device),
                angle: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(initial_angle_data, &device),
                k: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(initial_k_data, &device),
                v: Tensor::<CudaCubeAutodiffBackend, 3>::from_data(initial_v_data, &device),
            }),
        );

        let _ = <CudaCubeAutodiffBackend as BackendTrait>::sync(&device);
        assert_close_backend(
            graph.context.clone(),
            wrapped.context.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.ssm.clone(),
            wrapped.state.ssm.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.angle.clone(),
            wrapped.state.angle.clone(),
            5.0e-4,
            5.0e-4,
        );
        assert_close_backend(
            graph.state.k.clone(),
            wrapped.state.k.clone(),
            5.0e-3,
            5.0e-3,
        );
        assert_close_backend(
            graph.state.v.clone(),
            wrapped.state.v.clone(),
            5.0e-3,
            5.0e-3,
        );

        let output_weights =
            Tensor::<CudaCubeAutodiffBackend, 4>::from_data(output_weight_data, &device);
        let graph_grads = (graph.context * output_weights.clone()).sum().backward();
        let wrapper_grads = (wrapped.context * output_weights).sum().backward();
        let _ = <CudaCubeAutodiffBackend as BackendTrait>::sync(&device);

        assert_close_backend(
            hidden_graph.grad(&graph_grads).expect("graph hidden grad"),
            hidden_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper hidden grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            in_proj_graph
                .grad(&graph_grads)
                .expect("graph in_proj grad"),
            in_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper in_proj grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            dt_bias_graph
                .grad(&graph_grads)
                .expect("graph dt bias grad"),
            dt_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper dt bias grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            b_bias_graph.grad(&graph_grads).expect("graph b bias grad"),
            b_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper b bias grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            c_bias_graph.grad(&graph_grads).expect("graph c bias grad"),
            c_bias_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper c bias grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            b_norm_weight_graph
                .grad(&graph_grads)
                .expect("graph b norm grad"),
            b_norm_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper b norm grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            c_norm_weight_graph
                .grad(&graph_grads)
                .expect("graph c norm grad"),
            c_norm_weight_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper c norm grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            d_skip_graph.grad(&graph_grads).expect("graph d skip grad"),
            d_skip_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper d skip grad"),
            1.0e-2,
            1.0e-2,
        );
        assert_close_backend(
            out_proj_graph
                .grad(&graph_grads)
                .expect("graph out proj grad"),
            out_proj_wrapper
                .grad(&wrapper_grads)
                .expect("wrapper out proj grad"),
            1.0e-2,
            1.0e-2,
        );
    }
}
