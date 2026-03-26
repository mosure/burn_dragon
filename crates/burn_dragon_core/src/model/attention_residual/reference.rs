use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};

use super::AttentionResidualConfig;

#[derive(Module, Debug)]
pub struct AttentionResidual<B: Backend> {
    num_heads: usize,
    head_dim: usize,
    history_window: Option<usize>,
    recency_bias: f32,
    dropout: Dropout,
    mix_gate: Param<Tensor<B, 1>>,
    query: Param<Tensor<B, 2>>,
    key_proj: Param<Tensor<B, 2>>,
    value_proj: Param<Tensor<B, 2>>,
    out_proj: Param<Tensor<B, 2>>,
}

impl<B: Backend> AttentionResidual<B> {
    pub fn new(config: &AttentionResidualConfig, dense_dim: usize, device: &B::Device) -> Self {
        let num_heads = config.resolved_num_heads(dense_dim);
        let head_dim = dense_dim / num_heads.max(1);
        let mix_gate = Param::from_tensor(Tensor::<B, 1>::zeros([1], device));
        let query = Param::from_tensor(Tensor::<B, 2>::zeros([num_heads, head_dim], device));
        let key_proj = Param::from_tensor(identity_matrix(dense_dim, device));
        let value_proj = Param::from_tensor(identity_matrix(dense_dim, device));
        let out_proj = Param::from_tensor(identity_matrix(dense_dim, device));

        Self {
            num_heads,
            head_dim,
            history_window: config.history_window,
            recency_bias: config.recency_bias,
            dropout: DropoutConfig::new(config.dropout).init(),
            mix_gate,
            query,
            key_proj,
            value_proj,
            out_proj,
        }
    }

    pub fn history_window(&self) -> Option<usize> {
        self.history_window
    }

    pub fn branch_input(
        &self,
        current: Tensor<B, 4>,
        residual_history: &[Tensor<B, 4>],
    ) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = current.shape().dims::<4>();
        debug_assert_eq!(streams, 1);
        debug_assert_eq!(dim, self.num_heads * self.head_dim);

        let mut candidates = residual_history.to_vec();
        if candidates.is_empty() {
            candidates.push(current.clone());
        }
        let window = self.history_window.unwrap_or(candidates.len()).max(1);
        let start = candidates.len().saturating_sub(window);
        let history = Tensor::cat(candidates[start..].to_vec(), 1);
        let history_len = history.shape().dims::<4>()[1];
        let mixed =
            if history_len == 1 {
                history
            } else {
                let query = self
                    .query
                    .val()
                    .reshape([1, 1, self.num_heads, self.head_dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, time);
                let key = rms_norm_last_dim(
                    history
                        .clone()
                        .reshape([batch * history_len * time, dim])
                        .matmul(self.key_proj.val())
                        .reshape([batch, history_len, time, self.num_heads, self.head_dim])
                        .swap_dims(1, 2)
                        .swap_dims(2, 3),
                    1.0e-6,
                );
                let value = history
                    .reshape([batch * history_len * time, dim])
                    .matmul(self.value_proj.val())
                    .reshape([batch, history_len, time, self.num_heads, self.head_dim])
                    .swap_dims(1, 2)
                    .swap_dims(2, 3);

                let scale = (self.head_dim.max(1) as f32).sqrt();
                let recency_bias = (0..history_len)
                    .map(|index| -self.recency_bias * (history_len - 1 - index) as f32)
                    .collect::<Vec<_>>();
                let recency_bias =
                    Tensor::<B, 1>::from_floats(recency_bias.as_slice(), &current.device())
                        .reshape([1, 1, 1, history_len, 1]);
                let weights = activation::softmax(
                    query
                        .unsqueeze_dim::<5>(3)
                        .mul(key)
                        .sum_dim(4)
                        .div_scalar(scale)
                        .add(recency_bias),
                    3,
                );
                let attended = weights
                    .mul(value)
                    .sum_dim(3)
                    .reshape([batch * time, dim])
                    .matmul(self.out_proj.val())
                    .reshape([batch, 1, time, dim]);
                self.dropout.forward(attended)
            };
        let gate = self.mix_gate.val().tanh().reshape([1, 1, 1, 1]);
        current.clone().add(mixed.clone().sub(current).mul(gate))
    }

    #[cfg(test)]
    pub(crate) fn debug_set_mix_gate_raw(&mut self, raw: f32, device: &B::Device) {
        self.mix_gate = Param::from_tensor(Tensor::<B, 1>::from_floats([raw], device));
    }
}

fn identity_matrix<B: Backend>(dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = vec![0.0f32; dim * dim];
    for idx in 0..dim {
        values[idx * dim + idx] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [dim, dim]), device)
}

fn rms_norm_last_dim<B: Backend>(tensor: Tensor<B, 5>, eps: f32) -> Tensor<B, 5> {
    let rms = tensor
        .clone()
        .powf_scalar(2.0)
        .mean_dim(4)
        .add_scalar(eps)
        .sqrt();
    tensor.div(rms)
}
