use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation;

use super::{DragonNorm, StructuredStepMode};

#[derive(Clone)]
pub struct StructuredDenseUpdateOutput<B: Backend> {
    pub y_gate: Tensor<B, 3>,
    pub y_neuron: Tensor<B, 3>,
    pub delta_dense: Tensor<B, 3>,
}

/// Shared temporal decay policy for banked recurrent updates.
///
/// `Observe` and `Refine` reuse the current recurrent state without applying predictive temporal
/// decay, while `Predict` applies the configured recurrent decay.
pub fn structured_predict_decay(mode: StructuredStepMode, predict_decay: f32) -> f32 {
    if mode.temporal_dt() == 0 {
        1.0
    } else {
        predict_decay.clamp(0.0, 1.0)
    }
}

pub fn target_major_identity_read<B: Backend>(
    query: Tensor<B, 3>,
    rho: Tensor<B, 4>,
) -> Tensor<B, 3> {
    let [batch, targets, rank] = query.shape().dims::<3>();
    let [rho_batch, rho_targets, rho_rank, value_dim] = rho.shape().dims::<4>();
    assert_eq!(rho_batch, batch, "rho batch must match query batch");
    assert_eq!(rho_targets, targets, "rho targets must match query targets");
    assert_eq!(rho_rank, rank, "rho rank must match query rank");
    rho.mul(query.unsqueeze_dim::<4>(3))
        .sum_dims_squeeze::<3, usize>(&[2])
        .reshape([batch, targets, value_dim])
}

pub fn target_major_outer_product<B: Backend>(
    query: Tensor<B, 3>,
    value: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, targets, rank] = query.shape().dims::<3>();
    let [value_batch, value_targets, value_dim] = value.shape().dims::<3>();
    assert_eq!(value_batch, batch, "value batch must match query batch");
    assert_eq!(
        value_targets, targets,
        "value targets must match query targets"
    );
    query
        .unsqueeze_dim::<4>(3)
        .mul(value.unsqueeze_dim::<4>(2))
        .reshape([batch, targets, rank, value_dim])
}

pub fn target_major_apply_decay<B: Backend>(
    rho: Tensor<B, 4>,
    decay: Tensor<B, 1>,
) -> Tensor<B, 4> {
    let [batch, targets, rank, value_dim] = rho.shape().dims::<4>();
    let [decay_len] = decay.shape().dims::<1>();
    let decay = match decay_len {
        1 => decay.repeat_dim(0, rank.max(1)),
        len if len == rank => decay,
        _ => panic!(
            "target-major decay length {} must be 1 or equal to rank {}",
            decay_len, rank
        ),
    };
    rho.mul(decay.reshape([1, 1, rank, 1]))
        .reshape([batch, targets, rank, value_dim])
}

pub fn target_major_decay_add<B: Backend>(
    rho: Tensor<B, 4>,
    update: Tensor<B, 4>,
    decay: Tensor<B, 1>,
) -> Tensor<B, 4> {
    target_major_apply_decay(rho, decay).add(update)
}

pub fn target_major_identity_write<B: Backend>(
    rho: Tensor<B, 4>,
    query: Tensor<B, 3>,
    value: Tensor<B, 3>,
    decay: Tensor<B, 1>,
) -> Tensor<B, 4> {
    target_major_decay_add(rho, target_major_outer_product(query, value), decay)
}

pub fn structured_dense_update_tokens<B: Backend>(
    x_neuron: Tensor<B, 3>,
    a_dense: Tensor<B, 3>,
    y_gate_proj: &Linear<B>,
    delta_proj: &Linear<B>,
    value_norm: Option<&DragonNorm<B>>,
) -> StructuredDenseUpdateOutput<B> {
    let [batch, targets, value_dim] = a_dense.shape().dims::<3>();
    let [x_batch, x_targets, rank] = x_neuron.shape().dims::<3>();
    assert_eq!(x_batch, batch, "x_neuron batch must match a_dense batch");
    assert_eq!(
        x_targets, targets,
        "x_neuron targets must match a_dense targets"
    );

    let a_dense = if let Some(norm) = value_norm {
        norm.forward(a_dense)
    } else {
        a_dense
    };

    let y_gate = y_gate_proj
        .forward(a_dense.clone().reshape([batch * targets, value_dim]))
        .reshape([batch, targets, rank]);
    let y_gate = activation::relu(y_gate);
    let y_neuron = y_gate.clone().mul(x_neuron);
    let delta_dense = delta_proj.forward(y_neuron.clone().reshape([batch * targets, rank]));
    let delta_dim = delta_dense.shape().dims::<2>()[1];
    let delta_dense = delta_dense.reshape([batch, targets, delta_dim]);

    StructuredDenseUpdateOutput {
        y_gate,
        y_neuron,
        delta_dense,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::Param;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    fn device() -> <Backend as BackendTrait>::Device {
        <Backend as BackendTrait>::Device::default()
    }

    fn approx_eq(actual: TensorData, expected: &[f32]) {
        let actual = actual.to_vec::<f32>().expect("f32 tensor data");
        assert_eq!(actual.len(), expected.len());
        for (lhs, rhs) in actual.iter().zip(expected.iter()) {
            let diff = (lhs - rhs).abs();
            assert!(diff <= 1e-5, "expected {rhs}, got {lhs}, diff={diff}");
        }
    }

    #[test]
    fn target_major_identity_ops_match_reference_contract() {
        let device = device();
        let query =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![2.0, 3.0], [1, 1, 2]), &device);
        let value = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![5.0, 7.0, 11.0], [1, 1, 3]),
            &device,
        );
        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![13.0, 17.0, 19.0, 23.0, 29.0, 31.0], [1, 1, 2, 3]),
            &device,
        );
        let decay = Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.5, 0.25], [2]), &device);

        let read = target_major_identity_read(query.clone(), rho.clone());
        let update = target_major_outer_product(query.clone(), value.clone());
        let next = target_major_identity_write(rho, query, value, decay);

        approx_eq(read.into_data(), &[95.0, 121.0, 131.0]);
        approx_eq(update.into_data(), &[10.0, 14.0, 22.0, 15.0, 21.0, 33.0]);
        approx_eq(next.into_data(), &[16.5, 22.5, 31.5, 20.75, 28.25, 40.75]);
    }

    #[test]
    fn structured_dense_update_tokens_matches_gated_delta_contract() {
        let device = device();
        let x_neuron =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![2.0, 4.0], [1, 1, 2]), &device);
        let a_dense =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![3.0, 5.0], [1, 1, 2]), &device);
        let y_gate_proj = Linear {
            weight: Param::from_tensor(Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]),
                &device,
            )),
            bias: Some(Param::from_tensor(Tensor::<Backend, 1>::from_data(
                TensorData::new(vec![0.0, 0.0], [2]),
                &device,
            ))),
        };
        let delta_proj = Linear {
            weight: Param::from_tensor(Tensor::<Backend, 2>::from_data(
                TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [2, 2]),
                &device,
            )),
            bias: Some(Param::from_tensor(Tensor::<Backend, 1>::from_data(
                TensorData::new(vec![0.0, 0.0], [2]),
                &device,
            ))),
        };
        let output =
            structured_dense_update_tokens(x_neuron, a_dense, &y_gate_proj, &delta_proj, None);

        approx_eq(output.y_gate.into_data(), &[3.0, 5.0]);
        approx_eq(output.y_neuron.into_data(), &[6.0, 20.0]);
        approx_eq(output.delta_dense.into_data(), &[6.0, 20.0]);
    }

    #[test]
    fn structured_predict_decay_matches_observe_refine_predict_contract() {
        assert_eq!(
            structured_predict_decay(StructuredStepMode::Observe, 0.25),
            1.0
        );
        assert_eq!(
            structured_predict_decay(StructuredStepMode::Refine, 0.25),
            1.0
        );
        assert_eq!(
            structured_predict_decay(StructuredStepMode::Predict, 0.25),
            0.25
        );
        assert_eq!(
            structured_predict_decay(StructuredStepMode::Predict, -1.0),
            0.0
        );
        assert_eq!(
            structured_predict_decay(StructuredStepMode::Predict, 2.0),
            1.0
        );
    }
}
