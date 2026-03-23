use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, activation};

/// Real accelerated Mamba depthwise causal-conv kernels are still pending. The current path is a
/// tensorized experimental implementation that avoids a per-token host loop, but it is not yet a
/// true fused kernel.
pub const AVAILABLE: bool = false;

pub fn tensorized_mamba_depthwise_conv<B: BackendTrait>(
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

    let mut u = Tensor::<B, 4>::zeros([batch, views, d_inner, time], &device);
    for tap in 0..d_conv {
        let window = history.clone().slice_dim(3, tap + 1..tap + 1 + time).mul(
            conv_weight
                .clone()
                .slice_dim(1, tap..tap + 1)
                .reshape([1, 1, d_inner, 1]),
        );
        u = u + window;
    }

    if let Some(bias) = conv_bias {
        u = u + bias.reshape([1, 1, d_inner, 1]);
    }

    let next_state = history.slice_dim(3, time..time + d_conv);
    (silu(u), next_state)
}

fn silu<B: BackendTrait, const D: usize>(values: Tensor<B, D>) -> Tensor<B, D> {
    values.clone() * activation::sigmoid(values)
}
