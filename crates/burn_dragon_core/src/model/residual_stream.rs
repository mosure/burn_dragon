use burn::nn::Dropout;
use burn::tensor::backend::Backend;
use burn::tensor::Tensor;

use crate::kernel::{BlockPattern1d, relu_lowrank};

use super::residual::ManifoldHyperConnections;

#[derive(Debug)]
pub struct LowRankResidualOutput<B: Backend> {
    pub next: Tensor<B, 4>,
    pub x_sparse: Tensor<B, 4>,
    pub y_sparse: Tensor<B, 4>,
    pub xy_sparse: Tensor<B, 4>,
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    mut attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let dim = current.shape().dims::<4>()[3];

    let x_sparse = if use_fused {
        relu_lowrank::fused_forward(
            current.clone(),
            encoder.clone(),
            None,
            relu_threshold,
            latent_pattern,
        )
    } else {
        let mut x_latent = current.clone().matmul(encoder);
        if apply_threshold && relu_threshold != 0.0 {
            x_latent = x_latent.sub_scalar(relu_threshold);
        }
        apply_latent(x_latent)
    };

    let attn = attention(x_sparse.clone(), current.clone());
    let attn = apply_norm(attn);

    let y_sparse = if use_fused {
        relu_lowrank::fused_forward(attn.clone(), encoder_v, None, relu_threshold, latent_pattern)
    } else {
        let mut y_latent = attn.matmul(encoder_v);
        if apply_threshold && relu_threshold != 0.0 {
            y_latent = y_latent.sub_scalar(relu_threshold);
        }
        apply_latent(y_latent)
    };

    let xy_sparse = dropout.forward(x_sparse.clone() * y_sparse.clone());
    let mixed = xy_sparse.clone().swap_dims(1, 2);
    let [batch, time, heads, latent] = mixed.shape().dims();
    let mixed_flat = mixed.reshape([batch * time, heads * latent]);
    let mlp_flat = mixed_flat.matmul(decoder);
    let mlp_out = mlp_flat.reshape([batch, 1, time, dim]);
    let mlp_out = apply_norm(mlp_out);
    let next = apply_norm(current + mlp_out);

    LowRankResidualOutput {
        next,
        x_sparse,
        y_sparse,
        xy_sparse,
    }
}

pub fn mhc_split<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2>>) {
    if let Some(mhc) = mhc {
        mhc.width_connection(residuals)
    } else {
        (residuals.clone(), residuals, None)
    }
}

pub fn mhc_merge<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    branch_output: Tensor<B, 4>,
    residuals: Tensor<B, 4>,
    beta: Option<Tensor<B, 2>>,
) -> Tensor<B, 4> {
    if let Some(mhc) = mhc {
        mhc.depth_connection(branch_output, residuals, beta)
    } else {
        branch_output
    }
}

pub fn mhc_passthrough<B: Backend>(
    mhc: Option<&ManifoldHyperConnections<B>>,
    residuals: Tensor<B, 4>,
) -> Tensor<B, 4> {
    if let Some(mhc) = mhc {
        let (branch_input, residuals_out, beta) = mhc.width_connection(residuals);
        mhc.depth_connection(branch_input, residuals_out, beta)
    } else {
        residuals
    }
}
