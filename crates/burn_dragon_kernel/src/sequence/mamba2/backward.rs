#[cfg(feature = "cuda")]
use std::any::{Any, TypeId};
use std::marker::PhantomData;

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::TensorPrimitive;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, activation};
use burn_autodiff::checkpoint::base::Checkpointer;
use burn_autodiff::grads::Gradients;
use burn_autodiff::ops::{Backward, Ops};
use burn_cubecl::BoolElement;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::{Fusion, FusionTensor};
#[cfg(test)]
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, WgpuRuntime};

#[cfg(feature = "cuda")]
use crate::fusion_compat::register_fusion_float_tensor;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba::conv_runtime::{
    MambaDepthwiseConvCudaBackwardOutput, MambaDepthwiseConvCudaForwardOutput,
    fused_mamba_depthwise_conv_backward_cuda, fused_mamba_depthwise_conv_forward_cuda,
};
#[cfg(not(feature = "cuda"))]
use crate::kernels::sequence::mamba2::forward::{CudaShellCoreMode, CudaSsdCoreMode};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba2::forward::{
    CudaShellCoreMode, CudaSsdCoreMode, cuda_shell_core_mode_enabled, cuda_ssd_core_mode_enabled,
};
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba2::rmsnorm_runtime::fused_mamba2_rmsnorm_gated_backward_cuda;
#[cfg(feature = "cuda")]
use crate::kernels::sequence::mamba2::ssd_runtime::fused_mamba2_ssd_backward_cuda;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuFusionBackend<BT> = Fusion<CubeBackend<WgpuRuntime, f32, i32, BT>>;
#[cfg(test)]
type NdArrayBackend = NdArray<f32>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaFusionBackend<BT> = Fusion<CubeBackend<CudaRuntime, f32, i32, BT>>;

/// The Mamba-2 backward path uses a custom analytic backward over the SSD core and block shell.
/// On CUDA, the default wrapper can route the SSD recurrence, depthwise causal convolution, and
/// RMSNorm-gated shell through fused kernels while the GEMM shell remains tensorized.
pub const AVAILABLE: bool = true;

#[derive(Debug, Clone)]
pub(crate) struct Mamba2TensorizedBackwardState<FT> {
    pub(crate) hidden_states: FT,
    pub(crate) in_proj: FT,
    pub(crate) conv_weight: FT,
    pub(crate) conv_bias: FT,
    pub(crate) dt_bias: FT,
    pub(crate) a_log: FT,
    pub(crate) d_skip: FT,
    pub(crate) norm_weight: FT,
    pub(crate) out_proj: FT,
    pub(crate) initial_conv: Option<FT>,
    pub(crate) initial_ssm: Option<FT>,
    pub(crate) rmsnorm_inv_rms: Option<FT>,
    pub(crate) ssd_state_history: Option<FT>,
    pub(crate) d_inner: usize,
    pub(crate) d_state: usize,
    pub(crate) d_conv: usize,
    pub(crate) headdim: usize,
    pub(crate) ngroups: usize,
    pub(crate) norm_eps: f32,
    pub(crate) cuda_ssd_core_mode: CudaSsdCoreMode,
    pub(crate) cuda_shell_core_mode: CudaShellCoreMode,
}

#[derive(Debug)]
pub(crate) struct TensorizedMamba2Backward<B>(pub(crate) PhantomData<B>);

pub(crate) fn tensorized_mamba2_backward_impl<B>(
    ops: Ops<Mamba2TensorizedBackwardState<B::FloatTensorPrimitive>, 9>,
    grads: &mut Gradients,
) where
    B: BackendTrait,
{
    let grad_output = grads.consume::<B>(&ops.node);
    let state = ops.state;
    let parents = ops.parents;
    let d_inner = state.d_inner;
    let d_state = state.d_state;
    let _d_conv = state.d_conv;
    let headdim = state.headdim;
    let ngroups = state.ngroups;
    let norm_eps = state.norm_eps;
    let cuda_ssd_core_mode = state.cuda_ssd_core_mode;
    let cuda_shell_core_mode = state.cuda_shell_core_mode;

    let hidden_states_b =
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(state.hidden_states.clone()));
    let in_proj_b =
        BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.in_proj.clone()));
    let conv_weight_b =
        BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.conv_weight.clone()));
    let conv_bias_b =
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.conv_bias.clone()));
    let dt_bias_b =
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.dt_bias.clone()));
    let a_log_b = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.a_log.clone()));
    let d_skip_b = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.d_skip.clone()));
    let norm_weight_b =
        BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(state.norm_weight.clone()));
    let out_proj_b =
        BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(state.out_proj.clone()));
    let grad_output_b = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grad_output));

    let [batch, _views, time, d_model] = hidden_states_b.shape().dims::<4>();
    let nheads = d_inner / headdim;
    let heads_per_group = nheads / ngroups;
    let conv_dim = d_inner + 2 * ngroups * d_state;
    let in_proj_dim = 2 * d_inner + 2 * ngroups * d_state + nheads;

    let initial_conv_b = state
        .initial_conv
        .as_ref()
        .map(|conv| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(conv.clone())));
    let initial_ssm_b = state
        .initial_ssm
        .as_ref()
        .map(|ssm| BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(ssm.clone())));
    let rmsnorm_inv_rms_b = state
        .rmsnorm_inv_rms
        .as_ref()
        .map(|inv_rms| BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(inv_rms.clone())));
    let ssd_state_history_b = state
        .ssd_state_history
        .as_ref()
        .map(|history| BurnTensor::<B, 6>::from_primitive(TensorPrimitive::Float(history.clone())));

    let zxbcdt = hidden_states_b
        .clone()
        .reshape([batch * time, d_model])
        .matmul(in_proj_b.clone())
        .reshape([batch, time, in_proj_dim]);
    let z = zxbcdt
        .clone()
        .slice_dim(2, 0..d_inner)
        .reshape([batch, time, d_inner]);
    let xbc = zxbcdt
        .clone()
        .slice_dim(2, d_inner..(d_inner + conv_dim))
        .reshape([batch, time, conv_dim]);
    let dt_raw = zxbcdt
        .clone()
        .slice_dim(2, (d_inner + conv_dim)..in_proj_dim)
        .reshape([batch, time, nheads]);
    let xbc_input = xbc
        .clone()
        .swap_dims(1, 2)
        .reshape([batch, 1, conv_dim, time]);
    let accelerated_conv_forward = try_cuda_fused_depthwise_conv_forward_core::<B>(
        xbc_input.clone(),
        conv_weight_b.clone(),
        conv_bias_b.clone(),
        initial_conv_b.clone(),
        cuda_shell_core_mode,
    );
    let (xbc_preact_raw, xbc_history) = match accelerated_conv_forward {
        Some(output) => (output.preact, None),
        None => {
            let (preact, history) = depthwise_conv_preact_with_history(
                xbc_input.clone(),
                conv_weight_b.clone(),
                Some(conv_bias_b.clone()),
                initial_conv_b.clone(),
            );
            (preact, Some(history))
        }
    };
    let xbc_conv_raw = silu(xbc_preact_raw.clone());
    let xbc_conv = xbc_conv_raw
        .swap_dims(2, 3)
        .reshape([batch, time, conv_dim]);

    let x = xbc_conv
        .clone()
        .slice_dim(2, 0..d_inner)
        .reshape([batch, time, nheads, headdim]);
    let b_group = xbc_conv
        .clone()
        .slice_dim(2, d_inner..(d_inner + ngroups * d_state))
        .reshape([batch, time, ngroups, d_state]);
    let c_group = xbc_conv
        .clone()
        .slice_dim(2, (d_inner + ngroups * d_state)..conv_dim)
        .reshape([batch, time, ngroups, d_state]);
    let x_grouped = x.reshape([batch, time, ngroups, heads_per_group, headdim]);

    let dt_pre = dt_raw.clone() + dt_bias_b.clone().reshape([1, 1, nheads]);
    let dt = activation::softplus(dt_pre.clone(), 1.0);
    let dt_grouped = dt.clone().reshape([batch, time, ngroups, heads_per_group]);
    let a = a_log_b
        .clone()
        .exp()
        .neg()
        .reshape([1, 1, ngroups, heads_per_group, 1, 1]);
    let d_skip_grouped = d_skip_b
        .clone()
        .reshape([ngroups, heads_per_group])
        .reshape([1, 1, ngroups, heads_per_group, 1]);

    let d_a = (dt_grouped
        .clone()
        .reshape([batch, time, ngroups, heads_per_group, 1, 1])
        * a.clone())
    .exp();
    let drive = dt_grouped
        .clone()
        .reshape([batch, time, ngroups, heads_per_group, 1, 1])
        * b_group
            .clone()
            .reshape([batch, time, ngroups, 1, 1, d_state])
        * x_grouped
            .clone()
            .reshape([batch, time, ngroups, heads_per_group, headdim, 1]);
    let prefix_a = d_a.clone().cumprod(1);
    let mut ssm = prefix_a.clone() * drive.div(prefix_a.clone().add_scalar(1.0e-12)).cumsum(1);
    if let Some(initial_ssm) = initial_ssm_b.clone() {
        ssm = ssm
            + prefix_a.clone()
                * initial_ssm
                    .reshape([batch, 1, ngroups, heads_per_group, headdim, d_state])
                    .repeat_dim(1, time);
    }

    let y_grouped = (ssm.clone()
        * c_group
            .clone()
            .reshape([batch, time, ngroups, 1, 1, d_state]))
    .sum_dim(5)
    .reshape([batch, time, ngroups, heads_per_group, headdim])
        + d_skip_grouped.clone() * x_grouped.clone();
    let y_flat = y_grouped.clone().reshape([batch, time, d_inner]);
    let inv_rms = rmsnorm_inv_rms_b.clone().unwrap_or_else(|| {
        y_flat
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .add_scalar(norm_eps)
            .sqrt()
            .recip()
            .reshape([batch, time])
    });
    let gated = rmsnorm_gated_forward_from_inv_rms(
        y_flat.clone(),
        z.clone(),
        norm_weight_b.clone(),
        inv_rms.clone(),
    );
    let gated_flat = gated.clone().reshape([batch * time, d_inner]);
    let grad_context_flat = grad_output_b.reshape([batch * time, d_model]);
    let grad_out_proj = gated_flat
        .clone()
        .swap_dims(0, 1)
        .matmul(grad_context_flat.clone());
    let grad_gated_flat = grad_context_flat.matmul(out_proj_b.clone().swap_dims(0, 1));
    let (grad_y_flat, grad_z, grad_norm_weight) = if let Some(fused) =
        try_cuda_fused_rmsnorm_gated_backward_core::<B>(
            y_flat.clone(),
            z.clone(),
            norm_weight_b.clone(),
            grad_gated_flat.clone().reshape([batch, time, d_inner]),
            inv_rms.clone(),
            cuda_shell_core_mode,
        ) {
        (fused.grad_y, fused.grad_z, fused.grad_weight)
    } else {
        rmsnorm_gated_backward_from_inv_rms(
            y_flat,
            z,
            norm_weight_b.clone(),
            inv_rms.clone(),
            grad_gated_flat.reshape([batch, time, d_inner]),
        )
    };
    let grad_y_grouped = grad_y_flat.reshape([batch, time, ngroups, heads_per_group, headdim]);
    let (grad_x_grouped, grad_d_skip, grad_b_group, grad_c_group, grad_dt_grouped, grad_a_log) =
        if let Some(fused) = try_cuda_fused_ssd_backward_core::<B>(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log_b.clone(),
            d_skip_b.clone(),
            initial_ssm_b
                .clone()
                .map(|state| state.reshape([batch, ngroups, heads_per_group, headdim, d_state])),
            ssd_state_history_b.clone(),
            grad_y_grouped.clone(),
            cuda_ssd_core_mode,
        ) {
            (
                fused.grad_x_grouped,
                fused.grad_d_skip,
                fused.grad_b_group,
                fused.grad_c_group,
                fused.grad_dt_grouped,
                fused.grad_a_log,
            )
        } else {
            let reference = ssd_backward_reference(
                x_grouped.clone(),
                b_group.clone(),
                c_group.clone(),
                dt_grouped.clone(),
                a_log_b.clone(),
                d_skip_b.clone(),
                initial_ssm_b.clone().map(|state| {
                    state.reshape([batch, ngroups, heads_per_group, headdim, d_state])
                }),
                grad_y_grouped.clone(),
            );
            (
                reference.grad_x_grouped,
                reference.grad_d_skip,
                reference.grad_b_group,
                reference.grad_c_group,
                reference.grad_dt_grouped,
                reference.grad_a_log,
            )
        };

    let sigmoid_dt_pre = activation::sigmoid(dt_pre.clone());
    let grad_dt_pre = grad_dt_grouped.reshape([batch, time, nheads]) * sigmoid_dt_pre;
    let grad_dt_bias = grad_dt_pre.clone().sum_dim(0).sum_dim(1).reshape([nheads]);
    let grad_dt_raw = grad_dt_pre;

    let grad_xbc_conv = Tensor::cat(
        vec![
            grad_x_grouped.reshape([batch, time, d_inner]),
            grad_b_group.reshape([batch, time, ngroups * d_state]),
            grad_c_group.reshape([batch, time, ngroups * d_state]),
        ],
        2,
    );

    let grad_xbc_conv_raw = grad_xbc_conv
        .clone()
        .swap_dims(1, 2)
        .reshape([batch, 1, conv_dim, time]);
    let conv_sigmoid = activation::sigmoid(xbc_preact_raw.clone());
    let conv_ones = conv_sigmoid.clone().ones_like();
    let grad_conv_preact = grad_xbc_conv_raw
        * (conv_sigmoid.clone()
            * (conv_ones.clone() + xbc_preact_raw.clone() * (conv_ones - conv_sigmoid)));
    let (grad_conv_weight, grad_conv_bias, grad_xbc) = if let Some(fused) =
        try_cuda_fused_depthwise_conv_backward_core::<B>(
            xbc_input,
            conv_weight_b.clone(),
            initial_conv_b.clone(),
            grad_conv_preact.clone(),
            cuda_shell_core_mode,
        ) {
        (
            fused.grad_weight,
            fused.grad_bias,
            fused
                .grad_x
                .reshape([batch, conv_dim, time])
                .swap_dims(1, 2)
                .reshape([batch, time, conv_dim]),
        )
    } else {
        let xbc_history = xbc_history.expect("tensorized conv fallback should provide history");
        let grad_conv_bias = grad_conv_preact
            .clone()
            .sum_dim(0)
            .sum_dim(1)
            .sum_dim(3)
            .reshape([conv_dim]);
        let mut grad_conv_weight_cols = Vec::with_capacity(_d_conv);
        let mut grad_history =
            Tensor::<B, 4>::zeros([batch, 1, conv_dim, time + _d_conv], &xbc_history.device());
        for tap in 0.._d_conv {
            let history_window = xbc_history.clone().slice_dim(3, tap + 1..tap + 1 + time);
            let grad_weight_tap = (grad_conv_preact.clone() * history_window.clone())
                .sum_dim(0)
                .sum_dim(1)
                .sum_dim(3)
                .reshape([conv_dim, 1]);
            grad_conv_weight_cols.push(grad_weight_tap);

            let grad_window = grad_conv_preact.clone()
                * conv_weight_b
                    .clone()
                    .slice_dim(1, tap..tap + 1)
                    .reshape([1, 1, conv_dim, 1]);
            let updated_history_window =
                grad_history.clone().slice_dim(3, tap + 1..tap + 1 + time) + grad_window;
            grad_history = grad_history.slice_assign(
                [0..batch, 0..1, 0..conv_dim, tap + 1..tap + 1 + time],
                updated_history_window,
            );
        }
        let grad_conv_weight = Tensor::cat(grad_conv_weight_cols, 1);
        let grad_xbc = grad_history
            .slice_dim(3, _d_conv.._d_conv + time)
            .reshape([batch, conv_dim, time])
            .swap_dims(1, 2)
            .reshape([batch, time, conv_dim]);
        (grad_conv_weight, grad_conv_bias, grad_xbc)
    };
    let grad_zxbcdt = Tensor::cat(vec![grad_z, grad_xbc, grad_dt_raw], 2);
    let grad_zxbcdt_flat = grad_zxbcdt.clone().reshape([batch * time, in_proj_dim]);
    let hidden_states_flat = hidden_states_b.clone().reshape([batch * time, d_model]);
    let grad_hidden_states = grad_zxbcdt_flat
        .clone()
        .matmul(in_proj_b.clone().swap_dims(0, 1))
        .reshape([batch, time, d_model])
        .reshape([batch, 1, time, d_model]);
    let grad_in_proj = hidden_states_flat.swap_dims(0, 1).matmul(grad_zxbcdt_flat);

    if let Some(parent) = &parents[0] {
        grads.register::<B>(parent.id, grad_hidden_states.into_primitive().tensor());
    }
    if let Some(parent) = &parents[1] {
        grads.register::<B>(parent.id, grad_in_proj.into_primitive().tensor());
    }
    if let Some(parent) = &parents[2] {
        grads.register::<B>(parent.id, grad_conv_weight.into_primitive().tensor());
    }
    if let Some(parent) = &parents[3] {
        grads.register::<B>(parent.id, grad_conv_bias.into_primitive().tensor());
    }
    if let Some(parent) = &parents[4] {
        grads.register::<B>(parent.id, grad_dt_bias.into_primitive().tensor());
    }
    if let Some(parent) = &parents[5] {
        grads.register::<B>(parent.id, grad_a_log.into_primitive().tensor());
    }
    if let Some(parent) = &parents[6] {
        grads.register::<B>(parent.id, grad_d_skip.into_primitive().tensor());
    }
    if let Some(parent) = &parents[7] {
        grads.register::<B>(parent.id, grad_norm_weight.into_primitive().tensor());
    }
    if let Some(parent) = &parents[8] {
        grads.register::<B>(parent.id, grad_out_proj.into_primitive().tensor());
    }
}

struct CudaFusedSsdBackwardGrads<B: BackendTrait> {
    grad_x_grouped: Tensor<B, 5>,
    grad_b_group: Tensor<B, 4>,
    grad_c_group: Tensor<B, 4>,
    grad_dt_grouped: Tensor<B, 4>,
    grad_a_log: Tensor<B, 1>,
    grad_d_skip: Tensor<B, 1>,
}

fn ssd_forward_state_history_reference<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
) -> Tensor<B, 6> {
    let [batch, time, ngroups, heads_per_group, headdim] = x_grouped.shape().dims::<5>();
    let d_state = b_group.shape().dims::<4>()[3];
    let device = x_grouped.device();
    let a = a_log
        .exp()
        .neg()
        .reshape([1, ngroups, heads_per_group, 1, 1]);
    let mut ssm_state = initial_ssm.unwrap_or_else(|| {
        Tensor::<B, 5>::zeros([batch, ngroups, heads_per_group, headdim, d_state], &device)
    });
    let mut history = Vec::with_capacity(time);

    for step in 0..time {
        let x_t = x_grouped.clone().slice_dim(1, step..step + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
        ]);
        let b_t = b_group
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, ngroups, d_state]);
        let dt_t = dt_grouped.clone().slice_dim(1, step..step + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
        ]);

        let decay = (dt_t
            .clone()
            .reshape([batch, ngroups, heads_per_group, 1, 1])
            * a.clone())
        .exp();
        let input_term = dt_t.reshape([batch, ngroups, heads_per_group, 1, 1])
            * b_t.reshape([batch, ngroups, 1, 1, d_state])
            * x_t.reshape([batch, ngroups, heads_per_group, headdim, 1]);
        ssm_state = ssm_state * decay + input_term;
        history.push(ssm_state.clone().reshape([
            batch,
            1,
            ngroups,
            heads_per_group,
            headdim,
            d_state,
        ]));
    }

    Tensor::cat(history, 1)
}

fn ssd_backward_reference<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    grad_y_grouped: Tensor<B, 5>,
) -> CudaFusedSsdBackwardGrads<B> {
    let [batch, time, ngroups, heads_per_group, headdim] = x_grouped.shape().dims::<5>();
    let d_state = b_group.shape().dims::<4>()[3];
    let nheads = ngroups * heads_per_group;
    let device = x_grouped.device();
    let d_skip_grouped = d_skip.clone().reshape([ngroups, heads_per_group]).reshape([
        1,
        1,
        ngroups,
        heads_per_group,
        1,
    ]);
    let ssm_history = ssd_forward_state_history_reference(
        x_grouped.clone(),
        b_group.clone(),
        dt_grouped.clone(),
        a_log.clone(),
        initial_ssm.clone(),
    );
    let prev_ssm_all = if time > 1 {
        Tensor::cat(
            vec![
                initial_ssm
                    .clone()
                    .unwrap_or_else(|| {
                        Tensor::<B, 5>::zeros(
                            [batch, ngroups, heads_per_group, headdim, d_state],
                            &device,
                        )
                    })
                    .reshape([batch, 1, ngroups, heads_per_group, headdim, d_state]),
                ssm_history.clone().slice_dim(1, 0..time - 1),
            ],
            1,
        )
    } else {
        initial_ssm
            .clone()
            .unwrap_or_else(|| {
                Tensor::<B, 5>::zeros([batch, ngroups, heads_per_group, headdim, d_state], &device)
            })
            .reshape([batch, 1, ngroups, heads_per_group, headdim, d_state])
    };

    let a = a_log
        .clone()
        .exp()
        .neg()
        .reshape([1, 1, ngroups, heads_per_group, 1, 1]);
    let d_a = (dt_grouped
        .clone()
        .reshape([batch, time, ngroups, heads_per_group, 1, 1])
        * a.clone())
    .exp();
    let mut grad_x_grouped = grad_y_grouped.clone() * d_skip_grouped;
    let grad_d_skip = (grad_y_grouped.clone() * x_grouped.clone())
        .sum_dim(0)
        .sum_dim(1)
        .sum_dim(4)
        .reshape([nheads]);
    let mut grad_b_group = Tensor::<B, 4>::zeros([batch, time, ngroups, d_state], &device);
    let mut grad_c_group = Tensor::<B, 4>::zeros([batch, time, ngroups, d_state], &device);
    let mut grad_dt_grouped =
        Tensor::<B, 4>::zeros([batch, time, ngroups, heads_per_group], &device);
    let mut grad_a_grouped =
        Tensor::<B, 4>::zeros([batch, time, ngroups, heads_per_group], &device);
    let mut grad_state_carry =
        Tensor::<B, 5>::zeros([batch, ngroups, heads_per_group, headdim, d_state], &device);

    for t in (0..time).rev() {
        let grad_y_t = grad_y_grouped.clone().slice_dim(1, t..t + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
        ]);
        let c_t = c_group
            .clone()
            .slice_dim(1, t..t + 1)
            .reshape([batch, ngroups, d_state]);
        let x_t = x_grouped.clone().slice_dim(1, t..t + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
        ]);
        let b_t = b_group
            .clone()
            .slice_dim(1, t..t + 1)
            .reshape([batch, ngroups, d_state]);
        let dt_t =
            dt_grouped
                .clone()
                .slice_dim(1, t..t + 1)
                .reshape([batch, ngroups, heads_per_group]);
        let da_t =
            d_a.clone()
                .slice_dim(1, t..t + 1)
                .reshape([batch, ngroups, heads_per_group, 1, 1]);
        let state_t = ssm_history.clone().slice_dim(1, t..t + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
            d_state,
        ]);
        let prev_state_t = prev_ssm_all.clone().slice_dim(1, t..t + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
            d_state,
        ]);

        let grad_state_local =
            grad_y_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, headdim, 1])
                * c_t.clone().reshape([batch, ngroups, 1, 1, d_state]);
        let grad_state = grad_state_local + grad_state_carry.clone();

        let grad_c_t = (grad_y_t
            .clone()
            .reshape([batch, ngroups, heads_per_group, headdim, 1])
            * state_t.clone())
        .sum_dim(2)
        .sum_dim(3)
        .reshape([batch, ngroups, d_state]);
        grad_c_group = grad_c_group.slice_assign(
            [0..batch, t..t + 1, 0..ngroups, 0..d_state],
            grad_c_t.reshape([batch, 1, ngroups, d_state]),
        );

        let grad_da_t = (grad_state.clone() * prev_state_t.clone())
            .sum_dim(3)
            .sum_dim(4)
            .reshape([batch, ngroups, heads_per_group]);
        let grad_drive_t = grad_state.clone();
        grad_state_carry = grad_state * da_t.clone();

        let grad_dt_drive_t = (grad_drive_t.clone()
            * b_t.clone().reshape([batch, ngroups, 1, 1, d_state])
            * x_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, headdim, 1]))
        .sum_dim(3)
        .sum_dim(4)
        .reshape([batch, ngroups, heads_per_group]);
        let grad_b_t = (grad_drive_t.clone()
            * dt_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, 1, 1])
            * x_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, headdim, 1]))
        .sum_dim(2)
        .sum_dim(3)
        .reshape([batch, ngroups, d_state]);
        grad_b_group = grad_b_group.slice_assign(
            [0..batch, t..t + 1, 0..ngroups, 0..d_state],
            grad_b_t.reshape([batch, 1, ngroups, d_state]),
        );

        let grad_x_drive_t = (grad_drive_t
            * dt_t
                .clone()
                .reshape([batch, ngroups, heads_per_group, 1, 1])
            * b_t.reshape([batch, ngroups, 1, 1, d_state]))
        .sum_dim(4)
        .reshape([batch, ngroups, heads_per_group, headdim]);
        let grad_dt_da_t = grad_da_t.clone()
            * da_t.clone().reshape([batch, ngroups, heads_per_group])
            * a.clone().reshape([1, ngroups, heads_per_group]);
        grad_dt_grouped = grad_dt_grouped.slice_assign(
            [0..batch, t..t + 1, 0..ngroups, 0..heads_per_group],
            (grad_dt_drive_t + grad_dt_da_t).reshape([batch, 1, ngroups, heads_per_group]),
        );
        grad_a_grouped =
            grad_a_grouped.slice_assign(
                [0..batch, t..t + 1, 0..ngroups, 0..heads_per_group],
                (grad_da_t * da_t.reshape([batch, ngroups, heads_per_group]) * dt_t.clone())
                    .reshape([batch, 1, ngroups, heads_per_group]),
            );
        let updated_grad_x_t = grad_x_grouped.clone().slice_dim(1, t..t + 1).reshape([
            batch,
            ngroups,
            heads_per_group,
            headdim,
        ]) + grad_x_drive_t;
        grad_x_grouped = grad_x_grouped.slice_assign(
            [
                0..batch,
                t..t + 1,
                0..ngroups,
                0..heads_per_group,
                0..headdim,
            ],
            updated_grad_x_t.reshape([batch, 1, ngroups, heads_per_group, headdim]),
        );
    }

    let grad_a_log =
        grad_a_grouped.sum_dim(0).sum_dim(1).reshape([nheads]) * a_log.clone().exp().neg();
    CudaFusedSsdBackwardGrads {
        grad_x_grouped,
        grad_b_group,
        grad_c_group,
        grad_dt_grouped,
        grad_a_log,
        grad_d_skip,
    }
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_ssd_backward_core<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    ssd_state_history: Option<Tensor<B, 6>>,
    grad_y_grouped: Tensor<B, 5>,
    cuda_ssd_core_mode: CudaSsdCoreMode,
) -> Option<CudaFusedSsdBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_ssd_core_mode_enabled(cuda_ssd_core_mode) {
        return None;
    }
    try_cuda_fused_ssd_backward_core_direct(
        x_grouped.clone(),
        b_group.clone(),
        c_group.clone(),
        dt_grouped.clone(),
        a_log.clone(),
        d_skip.clone(),
        initial_ssm.clone(),
        ssd_state_history.clone(),
        grad_y_grouped.clone(),
    )
    .or_else(|| {
        try_cuda_fused_ssd_backward_core_fusion::<B, u8>(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            d_skip.clone(),
            initial_ssm.clone(),
            ssd_state_history.clone(),
            grad_y_grouped.clone(),
        )
    })
    .or_else(|| {
        try_cuda_fused_ssd_backward_core_fusion::<B, u32>(
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            initial_ssm,
            ssd_state_history,
            grad_y_grouped,
        )
    })
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_fused_ssd_backward_core<B: BackendTrait>(
    _x_grouped: Tensor<B, 5>,
    _b_group: Tensor<B, 4>,
    _c_group: Tensor<B, 4>,
    _dt_grouped: Tensor<B, 4>,
    _a_log: Tensor<B, 1>,
    _d_skip: Tensor<B, 1>,
    _initial_ssm: Option<Tensor<B, 5>>,
    _ssd_state_history: Option<Tensor<B, 6>>,
    _grad_y_grouped: Tensor<B, 5>,
    _cuda_ssd_core_mode: CudaSsdCoreMode,
) -> Option<CudaFusedSsdBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_ssd_backward_core_direct<B: BackendTrait>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    ssd_state_history: Option<Tensor<B, 6>>,
    grad_y_grouped: Tensor<B, 5>,
) -> Option<CudaFusedSsdBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let x_grouped_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(x_grouped.into_primitive().tensor())?;
    let b_group_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(b_group.into_primitive().tensor())?;
    let c_group_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(c_group.into_primitive().tensor())?;
    let dt_grouped_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(dt_grouped.into_primitive().tensor())?;
    let a_log_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let grad_y_grouped_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(grad_y_grouped.into_primitive().tensor())?;
    let initial_ssm_raw = match initial_ssm {
        Some(state) => Some(try_cast_primitive::<B, _>(state.into_primitive().tensor())?),
        None => None,
    };
    let ssd_state_history_raw = match ssd_state_history {
        Some(history) => Some(try_cast_primitive::<B, _>(
            history.into_primitive().tensor(),
        )?),
        None => None,
    };
    let output = fused_mamba2_ssd_backward_cuda(
        x_grouped_raw,
        b_group_raw,
        c_group_raw,
        dt_grouped_raw,
        a_log_raw,
        d_skip_raw,
        initial_ssm_raw,
        grad_y_grouped_raw,
        ssd_state_history_raw,
    );
    Some(CudaFusedSsdBackwardGrads {
        grad_x_grouped: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_x_grouped)?,
        )),
        grad_b_group: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_b_group)?,
        )),
        grad_c_group: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_c_group)?,
        )),
        grad_dt_grouped: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_dt_grouped)?,
        )),
        grad_a_log: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.grad_a_log,
        )?)),
        grad_d_skip: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_d_skip)?,
        )),
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_ssd_backward_core_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    x_grouped: Tensor<B, 5>,
    b_group: Tensor<B, 4>,
    c_group: Tensor<B, 4>,
    dt_grouped: Tensor<B, 4>,
    a_log: Tensor<B, 1>,
    d_skip: Tensor<B, 1>,
    initial_ssm: Option<Tensor<B, 5>>,
    ssd_state_history: Option<Tensor<B, 6>>,
    grad_y_grouped: Tensor<B, 5>,
) -> Option<CudaFusedSsdBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>() {
        return None;
    }

    let x_grouped_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(x_grouped.into_primitive().tensor())?;
    let client = x_grouped_fusion.client.clone();
    let b_group_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(b_group.into_primitive().tensor())?;
    let c_group_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(c_group.into_primitive().tensor())?;
    let dt_grouped_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(dt_grouped.into_primitive().tensor())?;
    let a_log_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(a_log.into_primitive().tensor())?;
    let d_skip_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(d_skip.into_primitive().tensor())?;
    let grad_y_grouped_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(grad_y_grouped.into_primitive().tensor())?;

    let x_grouped_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(x_grouped_fusion);
    let b_group_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(b_group_fusion);
    let c_group_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(c_group_fusion);
    let dt_grouped_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(dt_grouped_fusion);
    let a_log_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(a_log_fusion);
    let d_skip_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(d_skip_fusion);
    let grad_y_grouped_raw = client
        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(grad_y_grouped_fusion);
    let initial_ssm_raw = match initial_ssm {
        Some(state) => {
            let fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            Some(client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion))
        }
        None => None,
    };
    let ssd_state_history_raw = match ssd_state_history {
        Some(history) => {
            let fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
                try_cast_primitive::<B, _>(history.into_primitive().tensor())?;
            Some(client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion))
        }
        None => None,
    };
    let output = fused_mamba2_ssd_backward_cuda(
        x_grouped_raw,
        b_group_raw,
        c_group_raw,
        dt_grouped_raw,
        a_log_raw,
        d_skip_raw,
        initial_ssm_raw,
        grad_y_grouped_raw,
        ssd_state_history_raw,
    );
    Some(CudaFusedSsdBackwardGrads {
        grad_x_grouped: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_x_grouped))?,
        )),
        grad_b_group: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_b_group))?,
        )),
        grad_c_group: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_c_group))?,
        )),
        grad_dt_grouped: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(
                &client,
                output.grad_dt_grouped,
            ))?,
        )),
        grad_a_log: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            register_fusion_float_tensor(&client, output.grad_a_log),
        )?)),
        grad_d_skip: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_d_skip))?,
        )),
    })
}

#[cfg(feature = "cuda")]
fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

#[cfg(feature = "cuda")]
fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

#[cfg(feature = "cuda")]
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

struct CudaFusedRmsnormGatedBackwardGrads<B: BackendTrait> {
    grad_y: Tensor<B, 3>,
    grad_z: Tensor<B, 3>,
    grad_weight: Tensor<B, 1>,
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_rmsnorm_gated_backward_core<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    grad_output: Tensor<B, 3>,
    inv_rms: Tensor<B, 2>,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedRmsnormGatedBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_shell_core_mode_enabled(cuda_shell_core_mode) {
        return None;
    }
    try_cuda_fused_rmsnorm_gated_backward_core_direct(
        y.clone(),
        z.clone(),
        weight.clone(),
        grad_output.clone(),
        inv_rms.clone(),
    )
    .or_else(|| {
        try_cuda_fused_rmsnorm_gated_backward_core_fusion::<B, u8>(
            y.clone(),
            z.clone(),
            weight.clone(),
            grad_output.clone(),
            inv_rms.clone(),
        )
    })
    .or_else(|| {
        try_cuda_fused_rmsnorm_gated_backward_core_fusion::<B, u32>(
            y,
            z,
            weight,
            grad_output,
            inv_rms,
        )
    })
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_fused_rmsnorm_gated_backward_core<B: BackendTrait>(
    _y: Tensor<B, 3>,
    _z: Tensor<B, 3>,
    _weight: Tensor<B, 1>,
    _grad_output: Tensor<B, 3>,
    _inv_rms: Tensor<B, 2>,
    _cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedRmsnormGatedBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_rmsnorm_gated_backward_core_direct<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    grad_output: Tensor<B, 3>,
    inv_rms: Tensor<B, 2>,
) -> Option<CudaFusedRmsnormGatedBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let y_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(y.into_primitive().tensor())?;
    let z_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(z.into_primitive().tensor())?;
    let weight_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(weight.into_primitive().tensor())?;
    let grad_output_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(grad_output.into_primitive().tensor())?;
    let inv_rms_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(inv_rms.into_primitive().tensor())?;
    let output = fused_mamba2_rmsnorm_gated_backward_cuda(
        y_raw,
        z_raw,
        weight_raw,
        grad_output_raw,
        inv_rms_raw,
    );
    Some(CudaFusedRmsnormGatedBackwardGrads {
        grad_y: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_y)?,
        )),
        grad_z: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_z)?,
        )),
        grad_weight: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_weight)?,
        )),
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_rmsnorm_gated_backward_core_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    grad_output: Tensor<B, 3>,
    inv_rms: Tensor<B, 2>,
) -> Option<CudaFusedRmsnormGatedBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>() {
        return None;
    }

    let y_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(y.into_primitive().tensor())?;
    let client = y_fusion.client.clone();
    let z_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(z.into_primitive().tensor())?;
    let weight_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(weight.into_primitive().tensor())?;
    let grad_output_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(grad_output.into_primitive().tensor())?;
    let inv_rms_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(inv_rms.into_primitive().tensor())?;

    let y_raw = client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(y_fusion);
    let z_raw = client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(z_fusion);
    let weight_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(weight_fusion);
    let grad_output_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(grad_output_fusion);
    let inv_rms_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(inv_rms_fusion);
    let output = fused_mamba2_rmsnorm_gated_backward_cuda(
        y_raw,
        z_raw,
        weight_raw,
        grad_output_raw,
        inv_rms_raw,
    );
    Some(CudaFusedRmsnormGatedBackwardGrads {
        grad_y: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_y))?,
        )),
        grad_z: BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_z))?,
        )),
        grad_weight: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_weight))?,
        )),
    })
}

struct CudaFusedDepthwiseConvForwardOutput<B: BackendTrait> {
    preact: Tensor<B, 4>,
}

struct CudaFusedDepthwiseConvBackwardGrads<B: BackendTrait> {
    grad_x: Tensor<B, 4>,
    grad_weight: Tensor<B, 2>,
    grad_bias: Tensor<B, 1>,
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_forward_core<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    state: Option<Tensor<B, 4>>,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_shell_core_mode_enabled(cuda_shell_core_mode) {
        return None;
    }
    try_cuda_fused_depthwise_conv_forward_core_direct(
        x.clone(),
        conv_weight.clone(),
        conv_bias.clone(),
        state.clone(),
    )
    .or_else(|| {
        try_cuda_fused_depthwise_conv_forward_core_fusion::<B, u8>(
            x.clone(),
            conv_weight.clone(),
            conv_bias.clone(),
            state.clone(),
        )
    })
    .or_else(|| {
        try_cuda_fused_depthwise_conv_forward_core_fusion::<B, u32>(
            x,
            conv_weight,
            conv_bias,
            state,
        )
    })
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_fused_depthwise_conv_forward_core<B: BackendTrait>(
    _x: Tensor<B, 4>,
    _conv_weight: Tensor<B, 2>,
    _conv_bias: Tensor<B, 1>,
    _state: Option<Tensor<B, 4>>,
    _cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_forward_core_direct<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    state: Option<Tensor<B, 4>>,
) -> Option<CudaFusedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, views, channels, _time] = x.shape().dims::<4>();
    let d_conv = conv_weight.shape().dims::<2>()[1];
    let x_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let conv_weight_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let state_raw: CubeTensor<CudaRuntime> = match state {
        Some(state) => try_cast_primitive::<B, _>(state.into_primitive().tensor())?,
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };
    let output: MambaDepthwiseConvCudaForwardOutput =
        fused_mamba_depthwise_conv_forward_cuda(x_raw, conv_weight_raw, conv_bias_raw, state_raw);
    Some(CudaFusedDepthwiseConvForwardOutput {
        preact: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.preact)?,
        )),
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_forward_core_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Tensor<B, 1>,
    state: Option<Tensor<B, 4>>,
) -> Option<CudaFusedDepthwiseConvForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>() {
        return None;
    }

    let x_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let client = x_fusion.client.clone();
    let conv_weight_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let conv_bias_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(conv_bias.into_primitive().tensor())?;
    let x_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(x_fusion.clone());
    let [batch, views, channels, _time] = x_raw.meta.shape.dims::<4>();
    let conv_weight_raw = client
        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_weight_fusion.clone());
    let d_conv = conv_weight_raw.meta.shape.dims::<2>()[1];
    let conv_bias_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_bias_fusion);
    let state_raw = match state {
        Some(state) => {
            let state_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(state_fusion)
        }
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };
    let output =
        fused_mamba_depthwise_conv_forward_cuda(x_raw, conv_weight_raw, conv_bias_raw, state_raw);
    Some(CudaFusedDepthwiseConvForwardOutput {
        preact: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.preact))?,
        )),
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_backward_core<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    state: Option<Tensor<B, 4>>,
    grad_preact: Tensor<B, 4>,
    cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedDepthwiseConvBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !cuda_shell_core_mode_enabled(cuda_shell_core_mode) {
        return None;
    }
    try_cuda_fused_depthwise_conv_backward_core_direct(
        x.clone(),
        conv_weight.clone(),
        state.clone(),
        grad_preact.clone(),
    )
    .or_else(|| {
        try_cuda_fused_depthwise_conv_backward_core_fusion::<B, u8>(
            x.clone(),
            conv_weight.clone(),
            state.clone(),
            grad_preact.clone(),
        )
    })
    .or_else(|| {
        try_cuda_fused_depthwise_conv_backward_core_fusion::<B, u32>(
            x,
            conv_weight,
            state,
            grad_preact,
        )
    })
}

#[cfg(not(feature = "cuda"))]
fn try_cuda_fused_depthwise_conv_backward_core<B: BackendTrait>(
    _x: Tensor<B, 4>,
    _conv_weight: Tensor<B, 2>,
    _state: Option<Tensor<B, 4>>,
    _grad_preact: Tensor<B, 4>,
    _cuda_shell_core_mode: CudaShellCoreMode,
) -> Option<CudaFusedDepthwiseConvBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_backward_core_direct<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    state: Option<Tensor<B, 4>>,
    grad_preact: Tensor<B, 4>,
) -> Option<CudaFusedDepthwiseConvBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, views, channels, _time] = x.shape().dims::<4>();
    let d_conv = conv_weight.shape().dims::<2>()[1];
    let x_raw: CubeTensor<CudaRuntime> = try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let conv_weight_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let grad_preact_raw: CubeTensor<CudaRuntime> =
        try_cast_primitive::<B, _>(grad_preact.into_primitive().tensor())?;
    let state_raw: CubeTensor<CudaRuntime> = match state {
        Some(state) => try_cast_primitive::<B, _>(state.into_primitive().tensor())?,
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };
    let output: MambaDepthwiseConvCudaBackwardOutput = fused_mamba_depthwise_conv_backward_cuda(
        x_raw,
        conv_weight_raw,
        state_raw,
        grad_preact_raw,
    );
    Some(CudaFusedDepthwiseConvBackwardGrads {
        grad_x: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_x)?,
        )),
        grad_weight: BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output.grad_weight)?,
        )),
        grad_bias: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            output.grad_bias
        )?)),
    })
}

#[cfg(feature = "cuda")]
fn try_cuda_fused_depthwise_conv_backward_core_fusion<B: BackendTrait, BT: BoolElement + 'static>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    state: Option<Tensor<B, 4>>,
    grad_preact: Tensor<B, 4>,
) -> Option<CudaFusedDepthwiseConvBackwardGrads<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>() {
        return None;
    }

    let x_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(x.into_primitive().tensor())?;
    let client = x_fusion.client.clone();
    let conv_weight_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(conv_weight.into_primitive().tensor())?;
    let grad_preact_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
        try_cast_primitive::<B, _>(grad_preact.into_primitive().tensor())?;
    let x_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(x_fusion.clone());
    let [batch, views, channels, _time] = x_raw.meta.shape.dims::<4>();
    let conv_weight_raw = client
        .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(conv_weight_fusion.clone());
    let d_conv = conv_weight_raw.meta.shape.dims::<2>()[1];
    let grad_preact_raw =
        client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(grad_preact_fusion);
    let state_raw = match state {
        Some(state) => {
            let state_fusion: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
                try_cast_primitive::<B, _>(state.into_primitive().tensor())?;
            client.resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(state_fusion)
        }
        None => {
            BurnTensor::<CudaCubeBackend, 4>::zeros([batch, views, channels, d_conv], &x_raw.device)
                .into_primitive()
                .tensor()
        }
    };
    let output = fused_mamba_depthwise_conv_backward_cuda(
        x_raw,
        conv_weight_raw,
        state_raw,
        grad_preact_raw,
    );
    Some(CudaFusedDepthwiseConvBackwardGrads {
        grad_x: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_x))?,
        )),
        grad_weight: BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(register_fusion_float_tensor(&client, output.grad_weight))?,
        )),
        grad_bias: BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(
            register_fusion_float_tensor(&client, output.grad_bias),
        )?)),
    })
}

fn depthwise_conv_preact_with_history<B: BackendTrait>(
    x: Tensor<B, 4>,
    conv_weight: Tensor<B, 2>,
    conv_bias: Option<Tensor<B, 1>>,
    state: Option<Tensor<B, 4>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, views, d_inner, time] = x.shape().dims::<4>();
    let [weight_inner, d_conv] = conv_weight.shape().dims::<2>();
    assert_eq!(weight_inner, d_inner, "conv weight inner dim mismatch");

    let device = x.device();
    let initial_state = state
        .filter(|existing| existing.shape().dims::<4>() == [batch, views, d_inner, d_conv])
        .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, views, d_inner, d_conv], &device));
    let history = Tensor::cat(vec![initial_state, x], 3);

    let mut preact = Tensor::<B, 4>::zeros([batch, views, d_inner, time], &device);
    for tap in 0..d_conv {
        let window = history.clone().slice_dim(3, tap + 1..tap + 1 + time).mul(
            conv_weight
                .clone()
                .slice_dim(1, tap..tap + 1)
                .reshape([1, 1, d_inner, 1]),
        );
        preact = preact + window;
    }

    if let Some(bias) = conv_bias {
        preact = preact + bias.reshape([1, 1, d_inner, 1]);
    }

    (preact, history)
}

fn rmsnorm_gated_forward_from_inv_rms<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    inv_rms: Tensor<B, 2>,
) -> Tensor<B, 3> {
    let width = weight.shape().dims::<1>()[0];
    let [batch, time, _width] = y.shape().dims::<3>();
    (y * inv_rms.reshape([batch, time, 1])) * weight.reshape([1, 1, width]) * silu(z)
}

fn rmsnorm_gated_backward_from_inv_rms<B: BackendTrait>(
    y: Tensor<B, 3>,
    z: Tensor<B, 3>,
    weight: Tensor<B, 1>,
    inv_rms: Tensor<B, 2>,
    grad_output: Tensor<B, 3>,
) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 1>) {
    let [batch, time, width] = y.shape().dims::<3>();
    let inv_rms = inv_rms.reshape([batch, time, 1]);
    let normalized = y.clone() * inv_rms.clone();
    let gate = silu(z.clone());
    let weighted = normalized.clone() * weight.clone().reshape([1, 1, width]);

    let grad_weighted = grad_output.clone() * gate.clone();
    let grad_gate = grad_output * weighted.clone();
    let grad_weight = (grad_weighted.clone() * normalized.clone())
        .sum_dim(0)
        .sum_dim(1)
        .reshape([width]);
    let grad_normalized = grad_weighted * weight.reshape([1, 1, width]);
    let dot = (grad_normalized.clone() * y.clone())
        .sum_dim(2)
        .reshape([batch, time, 1]);
    let grad_y = grad_normalized * inv_rms.clone()
        - y * dot
            .mul(inv_rms.clone().powf_scalar(3.0))
            .div_scalar(width as f32);

    let sigmoid_z = activation::sigmoid(z.clone());
    let ones = sigmoid_z.clone().ones_like();
    let grad_z = grad_gate * (sigmoid_z.clone() * (ones.clone() + z * (ones - sigmoid_z)));

    (grad_y, grad_z, grad_weight)
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}

impl Backward<WgpuCubeBackend, 9> for TensorizedMamba2Backward<WgpuCubeBackend> {
    type State = Mamba2TensorizedBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba2_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

impl<BT> Backward<WgpuFusionBackend<BT>, 9> for TensorizedMamba2Backward<WgpuFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = Mamba2TensorizedBackwardState<FusionTensor<FusionCubeRuntime<WgpuRuntime>>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba2_backward_impl::<WgpuFusionBackend<BT>>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 9> for TensorizedMamba2Backward<CudaCubeBackend> {
    type State = Mamba2TensorizedBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba2_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl<BT> Backward<CudaFusionBackend<BT>, 9> for TensorizedMamba2Backward<CudaFusionBackend<BT>>
where
    BT: BoolElement + 'static,
{
    type State = Mamba2TensorizedBackwardState<FusionTensor<FusionCubeRuntime<CudaRuntime>>>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba2_backward_impl::<CudaFusionBackend<BT>>(ops, grads);
    }
}

#[cfg(test)]
impl Backward<NdArrayBackend, 9> for TensorizedMamba2Backward<NdArrayBackend> {
    type State =
        Mamba2TensorizedBackwardState<<NdArrayBackend as BackendTrait>::FloatTensorPrimitive>;

    fn backward(
        self,
        ops: Ops<Self::State, 9>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        tensorized_mamba2_backward_impl::<NdArrayBackend>(ops, grads);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::{
        CudaCubeBackend, ssd_backward_reference, ssd_forward_state_history_reference,
        try_cuda_fused_ssd_backward_core_direct,
    };
    use burn::prelude::ElementConversion;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Tensor, TensorData};

    fn max_abs_diff<const D: usize>(
        lhs: Tensor<CudaCubeBackend, D>,
        rhs: Tensor<CudaCubeBackend, D>,
    ) -> f32 {
        lhs.sub(rhs).abs().max().into_scalar().elem::<f32>()
    }

    #[allow(clippy::type_complexity)]
    fn make_realistic_cuda_inputs() -> (
        Tensor<CudaCubeBackend, 5>,
        Tensor<CudaCubeBackend, 4>,
        Tensor<CudaCubeBackend, 4>,
        Tensor<CudaCubeBackend, 4>,
        Tensor<CudaCubeBackend, 1>,
        Tensor<CudaCubeBackend, 1>,
        Tensor<CudaCubeBackend, 5>,
        Tensor<CudaCubeBackend, 5>,
    ) {
        let device = <CudaCubeBackend as BackendTrait>::Device::default();
        let batch = 1;
        let time = 32;
        let ngroups = 4;
        let heads_per_group = 1;
        let headdim = 64;
        let d_state = 16;
        let nheads = ngroups * heads_per_group;

        let x_grouped = Tensor::<CudaCubeBackend, 5>::from_data(
            TensorData::new(
                (0..(batch * time * ngroups * heads_per_group * headdim))
                    .map(|idx| ((idx % 257) as f32) / 257.0 - 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, ngroups, heads_per_group, headdim],
            ),
            &device,
        );
        let b_group = Tensor::<CudaCubeBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * ngroups * d_state))
                    .map(|idx| ((idx % 263) as f32) / 263.0 - 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, ngroups, d_state],
            ),
            &device,
        );
        let c_group = Tensor::<CudaCubeBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * ngroups * d_state))
                    .map(|idx| ((idx % 269) as f32) / 269.0 - 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, ngroups, d_state],
            ),
            &device,
        );
        let dt_grouped = Tensor::<CudaCubeBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * time * ngroups * heads_per_group))
                    .map(|idx| ((idx % 271) as f32) / 271.0 + 0.05)
                    .collect::<Vec<_>>(),
                [batch, time, ngroups, heads_per_group],
            ),
            &device,
        );
        let a_log = Tensor::<CudaCubeBackend, 1>::from_data(
            TensorData::new(
                (0..nheads)
                    .map(|idx| ((idx % 277) as f32) / 277.0 - 0.25)
                    .collect::<Vec<_>>(),
                [nheads],
            ),
            &device,
        );
        let d_skip = Tensor::<CudaCubeBackend, 1>::from_data(
            TensorData::new(
                (0..nheads)
                    .map(|idx| ((idx % 281) as f32) / 281.0 - 0.25)
                    .collect::<Vec<_>>(),
                [nheads],
            ),
            &device,
        );
        let initial_ssm = Tensor::<CudaCubeBackend, 5>::from_data(
            TensorData::new(
                (0..(batch * ngroups * heads_per_group * headdim * d_state))
                    .map(|idx| ((idx % 283) as f32) / 283.0 - 0.5)
                    .collect::<Vec<_>>(),
                [batch, ngroups, heads_per_group, headdim, d_state],
            ),
            &device,
        );
        let grad_y_grouped = Tensor::<CudaCubeBackend, 5>::from_data(
            TensorData::new(
                (0..(batch * time * ngroups * heads_per_group * headdim))
                    .map(|idx| ((idx % 293) as f32) / 293.0 - 0.5)
                    .collect::<Vec<_>>(),
                [batch, time, ngroups, heads_per_group, headdim],
            ),
            &device,
        );

        (
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            initial_ssm,
            grad_y_grouped,
        )
    }

    #[test]
    fn fused_ssd_backward_with_history_matches_reference_on_cuda_realistic_shape() {
        let (x_grouped, b_group, c_group, dt_grouped, a_log, d_skip, initial_ssm, grad_y_grouped) =
            make_realistic_cuda_inputs();
        let state_history = ssd_forward_state_history_reference(
            x_grouped.clone(),
            b_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            Some(initial_ssm.clone()),
        );
        let reference = ssd_backward_reference(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            d_skip.clone(),
            Some(initial_ssm.clone()),
            grad_y_grouped.clone(),
        );
        let fused = try_cuda_fused_ssd_backward_core_direct::<CudaCubeBackend>(
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            Some(initial_ssm),
            Some(state_history),
            grad_y_grouped,
        )
        .expect("direct cuda fused ssd backward");

        let grad_x_diff = max_abs_diff(reference.grad_x_grouped, fused.grad_x_grouped);
        let grad_b_diff = max_abs_diff(reference.grad_b_group, fused.grad_b_group);
        let grad_c_diff = max_abs_diff(reference.grad_c_group, fused.grad_c_group);
        let grad_dt_diff = max_abs_diff(reference.grad_dt_grouped, fused.grad_dt_grouped);
        let grad_a_diff = max_abs_diff(reference.grad_a_log, fused.grad_a_log);
        let grad_d_diff = max_abs_diff(reference.grad_d_skip, fused.grad_d_skip);

        assert!(
            grad_x_diff <= 5.0e-4,
            "expected grad_x parity, max diff {grad_x_diff}"
        );
        assert!(
            grad_b_diff <= 5.0e-4,
            "expected grad_b parity, max diff {grad_b_diff}"
        );
        assert!(
            grad_c_diff <= 5.0e-4,
            "expected grad_c parity, max diff {grad_c_diff}"
        );
        assert!(
            grad_dt_diff <= 5.0e-4,
            "expected grad_dt parity, max diff {grad_dt_diff}"
        );
        assert!(
            grad_a_diff <= 5.0e-4,
            "expected grad_a parity, max diff {grad_a_diff}"
        );
        assert!(
            grad_d_diff <= 5.0e-4,
            "expected grad_d parity, max diff {grad_d_diff}"
        );
    }

    #[test]
    fn fused_ssd_backward_recompute_matches_reference_on_cuda_realistic_shape() {
        let (x_grouped, b_group, c_group, dt_grouped, a_log, d_skip, initial_ssm, grad_y_grouped) =
            make_realistic_cuda_inputs();
        let reference = ssd_backward_reference(
            x_grouped.clone(),
            b_group.clone(),
            c_group.clone(),
            dt_grouped.clone(),
            a_log.clone(),
            d_skip.clone(),
            Some(initial_ssm.clone()),
            grad_y_grouped.clone(),
        );
        let fused = try_cuda_fused_ssd_backward_core_direct::<CudaCubeBackend>(
            x_grouped,
            b_group,
            c_group,
            dt_grouped,
            a_log,
            d_skip,
            Some(initial_ssm),
            None,
            grad_y_grouped,
        )
        .expect("direct cuda fused ssd backward");

        let grad_x_diff = max_abs_diff(reference.grad_x_grouped, fused.grad_x_grouped);
        let grad_b_diff = max_abs_diff(reference.grad_b_group, fused.grad_b_group);
        let grad_c_diff = max_abs_diff(reference.grad_c_group, fused.grad_c_group);
        let grad_dt_diff = max_abs_diff(reference.grad_dt_grouped, fused.grad_dt_grouped);
        let grad_a_diff = max_abs_diff(reference.grad_a_log, fused.grad_a_log);
        let grad_d_diff = max_abs_diff(reference.grad_d_skip, fused.grad_d_skip);

        assert!(
            grad_x_diff <= 5.0e-4,
            "expected recompute grad_x parity, max diff {grad_x_diff}"
        );
        assert!(
            grad_b_diff <= 5.0e-4,
            "expected recompute grad_b parity, max diff {grad_b_diff}"
        );
        assert!(
            grad_c_diff <= 5.0e-4,
            "expected recompute grad_c parity, max diff {grad_c_diff}"
        );
        assert!(
            grad_dt_diff <= 5.0e-4,
            "expected recompute grad_dt parity, max diff {grad_dt_diff}"
        );
        assert!(
            grad_a_diff <= 5.0e-4,
            "expected recompute grad_a parity, max diff {grad_a_diff}"
        );
        assert!(
            grad_d_diff <= 5.0e-4,
            "expected recompute grad_d parity, max diff {grad_d_diff}"
        );
    }
}
