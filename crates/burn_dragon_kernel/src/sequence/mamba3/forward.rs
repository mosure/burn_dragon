use std::any::Any;
use std::sync::Once;

use burn::tensor::TensorPrimitive;
#[cfg(feature = "cuda")]
use burn::tensor::backend::AutodiffBackend;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, activation};
#[cfg(any(test, feature = "cuda"))]
use burn_autodiff::Autodiff;
#[cfg(feature = "cuda")]
use burn_autodiff::checkpoint::strategy::NoCheckpointing;
#[cfg(feature = "cuda")]
use burn_autodiff::ops::{Backward, OpsKind};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::tensor::CubeTensor;
#[cfg(feature = "cuda")]
use burn_wgpu::CubeBackend;
#[cfg(feature = "cuda")]
use std::marker::PhantomData;

use crate::kernels::sequence::mamba3::backward::Mamba3ChunkTrace;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::backward::{
    Mamba3TensorizedBackwardState, TensorizedMamba3Backward,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_forward_cuda;
use crate::kernels::sequence::mamba3::bc_runtime::fused_mamba3_bc_forward_wgpu;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_forward_cuda;
use crate::kernels::sequence::mamba3::rotary_runtime::fused_mamba3_rotary_forward_wgpu;

#[cfg(test)]
use burn_ndarray::NdArray;
#[cfg(test)]
type NdArrayBackend = NdArray<f32>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffBackend = Autodiff<CudaCubeBackend>;
#[cfg(feature = "cuda")]
type CudaCubeAutodiffTensor = <CudaCubeAutodiffBackend as BackendTrait>::FloatTensorPrimitive;

const PI: f32 = std::f32::consts::PI;

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

fn rmsnorm_last_dim_3d<B: BackendTrait>(
    values: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    eps: f32,
) -> Tensor<B, 3> {
    let [batch, heads, width] = values.shape().dims::<3>();
    let rms = values
        .clone()
        .powf_scalar(2.0)
        .mean_dim(2)
        .add_scalar(eps)
        .sqrt()
        .reshape([batch, heads, 1]);
    (values / rms) * weight.reshape([1, 1, width])
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

fn try_fused_group_rmsnorm_expand_bias_forward<B: BackendTrait>(
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

#[cfg(any(test, feature = "cuda"))]
fn recommended_cuda_short_context_chunk_size(configured_chunk_size: usize, time: usize) -> usize {
    configured_chunk_size.max(time.min(256)).min(time.max(1))
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

    #[cfg(feature = "cuda")]
    {
        let time = _hidden_states.shape().dims::<4>()[2];
        let raw = _hidden_states.clone().into_primitive().tensor();
        if try_cast_primitive::<B, CubeTensor<CudaRuntime>>(raw).is_some() {
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

#[cfg(feature = "cuda")]
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
    try_tensorized_mamba3_autodiff_cuda(
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
    .expect("mamba3 custom backward wrapper requires cuda autodiff cube backend")
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
    let projected = hidden_states
        .clone()
        .reshape([batch * time, d_model])
        .matmul(in_proj)
        .reshape([batch, time, in_proj_dim]);
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
        let chunk_end = (chunk_start + chunk_size).min(time);
        let chunk_len = chunk_end - chunk_start;

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

        let tanh_angle = tanh_reference(angle_proj_chunk);
        let angle_delta = tanh_angle.clone() * dt_chunk.clone().unsqueeze_dim::<4>(3) * PI;
        let angle_chunk = angle_delta.cumsum(1)
            + angle_state
                .clone()
                .reshape([batch, 1, nheads, num_rope_angles]);
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

        let q_chunk = q_rot_chunk.swap_dims(1, 2);
        let k_chunk =
            (k_rot_chunk.clone() * scale_chunk.clone().unsqueeze_dim::<4>(3)).swap_dims(1, 2);
        let v_chunk = x_chunk.clone().swap_dims(1, 2);
        let z_chunk = z_chunk.swap_dims(1, 2);
        let da_chunk = (a_chunk.clone() * dt_chunk.clone()).swap_dims(1, 2);
        let da_prefix = da_chunk.clone().cumsum(2);
        let exp_da_prefix = da_prefix.clone().exp();
        let prev_out = q_chunk.clone().matmul(state_tilde.clone().swap_dims(2, 3))
            * exp_da_prefix.clone().unsqueeze_dim::<4>(3);

        let decay = (da_prefix.clone().unsqueeze_dim::<4>(3)
            - da_prefix.clone().unsqueeze_dim::<4>(2))
        .clamp_max(0.0)
        .exp();
        let raw_scores = q_chunk.clone().matmul(k_chunk.clone().swap_dims(2, 3));
        let current_scores = raw_scores.clone() * decay.clone();
        let tril_scores = current_scores.tril(-1);
        let current_out = tril_scores.clone().matmul(v_chunk.clone());
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

        if let Some(traces) = chunk_traces.as_mut() {
            traces.push(Mamba3ChunkTrace {
                chunk_start,
                chunk_end,
                k_state: k_state.clone().into_primitive().tensor(),
                v_state: v_state.clone().into_primitive().tensor(),
                q_pre: q_pre_chunk.into_primitive().tensor(),
                k_pre: k_pre_chunk.into_primitive().tensor(),
                b_input: b_input_chunk.into_primitive().tensor(),
                c_input: c_input_chunk.into_primitive().tensor(),
                b_inv_rms: b_inv_rms_chunk.into_primitive().tensor(),
                c_inv_rms: c_inv_rms_chunk.into_primitive().tensor(),
                dt_pre: dt_pre_chunk.into_primitive().tensor(),
                dt: dt_chunk.clone().into_primitive().tensor(),
                a_unclamped: a_unclamped_chunk.into_primitive().tensor(),
                a: a_chunk.clone().into_primitive().tensor(),
                trap: trap_chunk.clone().into_primitive().tensor(),
                gamma: gamma_chunk.clone().into_primitive().tensor(),
                scale: scale_chunk.clone().into_primitive().tensor(),
                tanh_angle: tanh_angle.into_primitive().tensor(),
                angle_chunk: angle_chunk.clone().into_primitive().tensor(),
                q_head: q_chunk.clone().into_primitive().tensor(),
                k_rot_chunk: k_rot_chunk.clone().into_primitive().tensor(),
                k_head: k_chunk.clone().into_primitive().tensor(),
                v_head: v_chunk.clone().into_primitive().tensor(),
                z_head: z_chunk.clone().into_primitive().tensor(),
                state_tilde: state_tilde.clone().into_primitive().tensor(),
                da_prefix: da_prefix.clone().into_primitive().tensor(),
                exp_da_prefix: exp_da_prefix.into_primitive().tensor(),
                qk_inner: qk_inner.into_primitive().tensor(),
                y_pre: y_pre.into_primitive().tensor(),
            });
        }

        let da_last = da_prefix
            .clone()
            .slice_dim(2, chunk_len - 1..chunk_len)
            .reshape([batch, nheads]);
        let weighted_v = v_chunk
            * (da_last.clone().unsqueeze_dim::<3>(2) - da_prefix.clone())
                .exp()
                .unsqueeze_dim::<4>(3);
        ssm_state = state_tilde * da_last.clone().reshape([batch, nheads, 1, 1]).exp()
            + weighted_v.swap_dims(2, 3).matmul(k_chunk.clone());
        angle_state = angle_chunk.slice_dim(1, chunk_len - 1..chunk_len).reshape([
            batch,
            nheads,
            num_rope_angles,
        ]);
        k_state = k_rot_chunk
            .slice_dim(1, chunk_len - 1..chunk_len)
            .reshape([batch, nheads, d_state]);
        v_state = x_chunk
            .slice_dim(1, chunk_len - 1..chunk_len)
            .reshape([batch, nheads, headdim]);
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
