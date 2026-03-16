use burn::nn::Dropout;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use crate::kernel::{BlockPattern1d, relu_lowrank};

#[derive(Debug)]
pub struct LowRankResidualOutput<B: Backend> {
    pub next: Tensor<B, 4>,
    pub x_neuron: Tensor<B, 4>,
    pub y_gate: Tensor<B, 4>,
    pub y_neuron: Tensor<B, 4>,
}

fn decode_y_neuron_tail<B: Backend>(y_neuron: Tensor<B, 4>, decoder: Tensor<B, 2>) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let dim = decoder.shape().dims::<2>()[1];

    if heads == 1 {
        return y_neuron
            .reshape([batch * time, latent])
            .matmul(decoder)
            .reshape([batch, 1, time, dim]);
    }

    let decoder_by_head = decoder.reshape([heads, latent, dim]);
    let mixed_by_head = y_neuron.swap_dims(0, 1).reshape([heads, batch * time, latent]);
    mixed_by_head
        .matmul(decoder_by_head)
        .sum_dim(0)
        .reshape([batch, 1, time, dim])
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
    sparse_mask: Option<Tensor<B, 4>>,
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
    let sparse_mask = if use_fused && latent_pattern.is_sparse() {
        sparse_mask.or_else(|| {
            let latent = encoder.shape().dims::<4>()[3];
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        })
    } else {
        None
    };

    let x_neuron = if use_fused {
        relu_lowrank::fused_forward(
            current.clone(),
            encoder.clone(),
            None,
            relu_threshold,
            latent_pattern,
            sparse_mask.clone(),
        )
    } else {
        let mut x_latent = current.clone().matmul(encoder);
        if apply_threshold && relu_threshold != 0.0 {
            x_latent = x_latent.sub_scalar(relu_threshold);
        }
        apply_latent(x_latent)
    };

    let attn = attention(x_neuron.clone(), current.clone());
    let attn = apply_norm(attn);

    let y_gate = if use_fused {
        relu_lowrank::fused_forward(
            attn.clone(),
            encoder_v,
            None,
            relu_threshold,
            latent_pattern,
            sparse_mask,
        )
    } else {
        let mut y_latent = attn.matmul(encoder_v);
        if apply_threshold && relu_threshold != 0.0 {
            y_latent = y_latent.sub_scalar(relu_threshold);
        }
        apply_latent(y_latent)
    };

    let y_neuron = dropout.forward(x_neuron.clone() * y_gate.clone());
    let mlp_out = decode_y_neuron_tail(y_neuron.clone(), decoder);
    let mlp_out = apply_norm(mlp_out);
    let next = apply_norm(current + mlp_out);

    LowRankResidualOutput {
        next,
        x_neuron,
        y_gate,
        y_neuron,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BlockPattern1d;
    use burn::nn::DropoutConfig;
    use burn::tensor::{TensorData, backend::Backend as BackendTrait};
    use burn_ndarray::NdArray;

    fn assert_close(actual: Vec<f32>, expected: Vec<f32>, tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "mismatch at index {index}: actual={a}, expected={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn decode_y_neuron_tail_matches_flat_decoder_projection_multi_head() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let y_neuron = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=24).map(|value| value as f32 * 0.1).collect::<Vec<_>>(), [2, 2, 2, 3]),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new((1..=30).map(|value| value as f32 * 0.05).collect::<Vec<_>>(), [6, 5]),
            &device,
        );

        let actual = decode_y_neuron_tail(y_neuron.clone(), decoder.clone())
            .into_data()
            .to_vec::<f32>()
            .expect("actual vec");
        let expected = y_neuron
            .swap_dims(1, 2)
            .reshape([4, 6])
            .matmul(decoder)
            .reshape([2, 1, 2, 5])
            .into_data()
            .to_vec::<f32>()
            .expect("expected vec");

        assert_close(actual, expected, 1.0e-6);
    }

    #[test]
    fn decode_y_neuron_tail_matches_flat_decoder_projection_single_head() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let y_neuron = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=12).map(|value| value as f32 * 0.2).collect::<Vec<_>>(), [2, 1, 2, 3]),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new((1..=12).map(|value| value as f32 * 0.04).collect::<Vec<_>>(), [3, 4]),
            &device,
        );

        let actual = decode_y_neuron_tail(y_neuron.clone(), decoder.clone())
            .into_data()
            .to_vec::<f32>()
            .expect("actual vec");
        let expected = y_neuron
            .reshape([4, 3])
            .matmul(decoder)
            .reshape([2, 1, 2, 4])
            .into_data()
            .to_vec::<f32>()
            .expect("expected vec");

        assert_close(actual, expected, 1.0e-6);
    }

    #[test]
    fn lowrank_residual_step_matches_paper_neuron_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let current =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![1.0, 2.0], [1, 1, 1, 2]), &device);
        let encoder = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 1, 2, 2]),
            &device,
        );
        let encoder_v = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![3.0, 0.0, 0.0, 4.0], [1, 1, 2, 2]),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]),
            &device,
        );
        let dropout = DropoutConfig::new(0.0).init();

        let output = lowrank_residual_step(
            current.clone(),
            encoder,
            encoder_v,
            decoder,
            &dropout,
            false,
            0.0,
            false,
            &BlockPattern1d::dense(2),
            None,
            |query, _current| query,
            |values| values,
            |values| values,
        );

        let expected_x_neuron = vec![1.0, 2.0];
        let expected_y_gate = vec![3.0, 8.0];
        let expected_y_neuron = vec![3.0, 16.0];
        let expected_next = vec![4.0, 18.0];

        let x_neuron = output
            .x_neuron
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("x_neuron vec");
        let y_gate = output
            .y_gate
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("y_gate vec");
        let y_neuron = output
            .y_neuron
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("y_neuron vec");
        let next = output
            .next
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("next vec");

        assert_eq!(x_neuron, expected_x_neuron);
        assert_eq!(y_gate, expected_y_gate);
        assert_eq!(y_neuron, expected_y_neuron);
        assert_eq!(next, expected_next);
    }
}
