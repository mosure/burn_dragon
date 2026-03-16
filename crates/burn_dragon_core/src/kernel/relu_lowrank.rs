use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

use super::block_sparse::BlockPattern1d;

fn single_stream_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, streams, time, embd] = input.shape().dims::<4>();
    let [weight_batch, heads, weight_embd, latent] = weight.shape().dims::<4>();
    if streams != 1 || weight_batch != 1 || embd != weight_embd {
        return None;
    }

    let input_flat = input.reshape([batch * time, embd]);
    let weight_flat = weight
        .reshape([heads, embd, latent])
        .swap_dims(0, 1)
        .reshape([embd, heads * latent]);
    let projected = input_flat.matmul(weight_flat);
    Some(projected.reshape([batch, time, heads, latent]).swap_dims(1, 2))
}

fn head_aligned_projection_flat<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, heads, time, embd] = input.shape().dims::<4>();
    let [weight_batch, weight_heads, weight_embd, latent] = weight.shape().dims::<4>();
    if weight_batch != 1 || heads != weight_heads || embd != weight_embd {
        return None;
    }

    let input_by_head = input.swap_dims(0, 1).reshape([heads, batch * time, embd]);
    let weight_by_head = weight.reshape([heads, embd, latent]);
    let projected = input_by_head.matmul(weight_by_head);
    Some(projected.reshape([heads, batch, time, latent]).swap_dims(0, 1))
}

fn head_aligned_projection_block_dense<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
) -> Option<Tensor<B, 4>> {
    let [batch, heads, time, embd] = input.shape().dims::<4>();
    let [weight_batch, weight_heads, weight_embd, latent] = weight.shape().dims::<4>();
    if weight_batch != 1 || heads != weight_heads || embd != weight_embd {
        return None;
    }

    let device = input.device();
    let input_flat = input.swap_dims(1, 2).reshape([batch * time, heads * embd]);
    let mut row_blocks = Vec::with_capacity(heads);
    for head in 0..heads {
        let head_weight = weight
            .clone()
            .slice_dim(1, head..head + 1)
            .reshape([embd, latent]);
        let left_latent = head * latent;
        let right_latent = (heads - head - 1) * latent;
        let row = match (left_latent, right_latent) {
            (0, 0) => head_weight,
            (0, _) => Tensor::cat(
                vec![
                    head_weight,
                    Tensor::<B, 2>::zeros([embd, right_latent], &device),
                ],
                1,
            ),
            (_, 0) => Tensor::cat(
                vec![
                    Tensor::<B, 2>::zeros([embd, left_latent], &device),
                    head_weight,
                ],
                1,
            ),
            (_, _) => Tensor::cat(
                vec![
                    Tensor::<B, 2>::zeros([embd, left_latent], &device),
                    head_weight,
                    Tensor::<B, 2>::zeros([embd, right_latent], &device),
                ],
                1,
            ),
        };
        row_blocks.push(row);
    }

    let weight_flat = Tensor::cat(row_blocks, 0);
    let projected = input_flat.matmul(weight_flat);
    Some(projected.reshape([batch, time, heads, latent]).swap_dims(1, 2))
}

pub fn fused_forward<B: Backend>(
    input: Tensor<B, 4>,
    weight: Tensor<B, 4>,
    bias: Option<Tensor<B, 3>>,
    threshold: f32,
    layout: &BlockPattern1d,
    sparse_mask: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    let device = input.device();
    let latent = weight.shape().dims::<4>()[3];

    let mut projected = single_stream_projection_flat(input.clone(), weight.clone())
        .or_else(|| head_aligned_projection_block_dense(input.clone(), weight.clone()))
        .or_else(|| head_aligned_projection_flat(input.clone(), weight.clone()))
        .unwrap_or_else(|| input.matmul(weight));

    if let Some(bias) = bias {
        let dims = bias.shape().dims::<3>();
        let bias = bias.reshape([1, dims[0], 1, dims[2]]);
        projected = projected + bias;
    }

    if threshold != 0.0 {
        projected = projected.sub_scalar(threshold);
    }

    let mut activated = activation::relu(projected);

    if layout.is_sparse() {
        let mask = sparse_mask.unwrap_or_else(|| layout.mask::<B>(latent, &device));
        activated = activated * mask;
    }

    activated
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{TensorData, backend::Backend as BackendTrait};
    use burn_ndarray::NdArray;

    #[test]
    fn fused_forward_matches_reference_single_stream_projection() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let input = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=12).map(|v| v as f32).collect::<Vec<_>>(), [2, 1, 2, 3]),
            &device,
        );
        let weight = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=24).map(|v| (v as f32) * 0.05).collect::<Vec<_>>(),
                [1, 2, 3, 4],
            ),
            &device,
        );

        let actual = fused_forward(
            input.clone(),
            weight.clone(),
            None,
            0.0,
            &BlockPattern1d::dense(4),
            None,
        );
        let expected = activation::relu(input.matmul(weight));
        let actual = actual.into_data().to_vec::<f32>().expect("actual");
        let expected = expected.into_data().to_vec::<f32>().expect("expected");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.into_iter().zip(expected.into_iter()) {
            assert!((a - b).abs() <= 1e-6, "single-stream mismatch: {a} vs {b}");
        }
    }

    #[test]
    fn fused_forward_matches_reference_head_aligned_projection() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let input = Tensor::<Backend, 4>::from_data(
            TensorData::new((1..=24).map(|v| v as f32).collect::<Vec<_>>(), [2, 2, 2, 3]),
            &device,
        );
        let weight = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=24).map(|v| (v as f32) * 0.03).collect::<Vec<_>>(),
                [1, 2, 3, 4],
            ),
            &device,
        );

        let actual = fused_forward(
            input.clone(),
            weight.clone(),
            None,
            0.0,
            &BlockPattern1d::dense(4),
            None,
        );
        let expected = activation::relu(input.matmul(weight));
        let actual = actual.into_data().to_vec::<f32>().expect("actual");
        let expected = expected.into_data().to_vec::<f32>().expect("expected");
        assert_eq!(actual.len(), expected.len());
        for (a, b) in actual.into_iter().zip(expected.into_iter()) {
            assert!((a - b).abs() <= 1e-6, "head-aligned mismatch: {a} vs {b}");
        }
    }
}
