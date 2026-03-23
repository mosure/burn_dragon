use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

pub(crate) fn rwkv8_query_weights<B: Backend>(query_t: Tensor<B, 3>) -> Tensor<B, 3> {
    query_t
        .clone()
        .div(query_t.clone().sum_dim(2).add_scalar(1.0e-6).reshape([
            query_t.shape().dims::<3>()[0],
            query_t.shape().dims::<3>()[1],
            1,
        ]))
}

pub(crate) fn rwkv8_read_context_step<B: Backend>(
    rho: Tensor<B, 4>,
    rho_norm: Tensor<B, 3>,
    query_t: Tensor<B, 3>,
) -> Tensor<B, 4> {
    let [batch, heads, latent, embd] = rho.shape().dims::<4>();
    let q_weights = rwkv8_query_weights(query_t);
    (rho.div(
        rho_norm
            .add_scalar(1.0e-6)
            .reshape([batch, heads, latent, 1]),
    ) * q_weights.reshape([batch, heads, latent, 1]))
    .sum_dim(2)
    .reshape([batch, heads, 1, embd])
}

pub(crate) fn rwkv8_update_state_step<B: Backend>(
    rho: Tensor<B, 4>,
    rho_norm: Tensor<B, 3>,
    query_t: Tensor<B, 3>,
    value_t: Tensor<B, 3>,
    decay: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, heads, latent] = query_t.shape().dims::<3>();
    let n_embd = value_t.shape().dims::<3>()[2];
    let next_rho = rho.mul(decay.clone().reshape([1, heads, latent, 1])).add(
        query_t.clone().reshape([batch, heads, latent, 1])
            * value_t.reshape([batch, heads, 1, n_embd]),
    );
    let next_rho_norm = rho_norm.mul(decay).add(query_t);
    (next_rho, next_rho_norm)
}

pub fn recurrent_rwkv8_state_space_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    rho_norm_state: Option<Tensor<B, 3>>,
    decay: Tensor<B, 3>,
) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
    let [batch, heads, time, latent] = query.shape().dims();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let decay = decay.reshape([1, heads, latent]);

    let mut rho = match rho_state {
        Some(existing) => {
            let dims = existing.shape().dims::<4>();
            if dims == [batch, heads, latent, n_embd] {
                existing
            } else {
                Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device)
            }
        }
        None => Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device),
    };

    let mut rho_norm = match rho_norm_state {
        Some(existing) => {
            let dims = existing.shape().dims::<3>();
            if dims == [batch, heads, latent] {
                existing
            } else {
                Tensor::<B, 3>::zeros([batch, heads, latent], &device)
            }
        }
        None => Tensor::<B, 3>::zeros([batch, heads, latent], &device),
    };

    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

    for t in 0..time {
        let q_t = query
            .clone()
            .slice_dim(2, t..t + 1)
            .reshape([batch, heads, latent]);
        let value_t = value
            .clone()
            .slice_dim(2, t..t + 1)
            .repeat_dim(1, heads)
            .reshape([batch, heads, n_embd]);

        let context_t = rwkv8_read_context_step(rho.clone(), rho_norm.clone(), q_t.clone());
        outputs.push(context_t);

        (rho, rho_norm) = rwkv8_update_state_step(rho, rho_norm, q_t, value_t, decay.clone());
    }

    (Tensor::cat(outputs, 2), rho, rho_norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    fn test_inputs(
        device: &<Backend as BackendTrait>::Device,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 4>, Tensor<Backend, 3>) {
        let batch = 1;
        let heads = 2;
        let time = 5;
        let latent = 4;
        let n_embd = 3;
        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.2, 0.3, 0.4, 0.5, 0.3, 0.4, 0.5, 0.6, 0.4, 0.5, 0.6, 0.7, 0.5, 0.6, 0.7, 0.8,
                    0.6, 0.7, 0.8, 0.9, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.6, 0.5, 0.4, 0.3,
                    0.7, 0.6, 0.5, 0.4, 0.8, 0.7, 0.6, 0.5,
                ],
                [batch, heads, time, latent],
            ),
            device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.1, 0.2, 0.3, 0.2, 0.3, 0.4, 0.3, 0.4, 0.5, 0.4, 0.5, 0.6, 0.5, 0.6, 0.7,
                ],
                [batch, 1, time, n_embd],
            ),
            device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(
                vec![0.97, 0.95, 0.93, 0.91, 0.96, 0.94, 0.92, 0.90],
                [1, heads, latent],
            ),
            device,
        );
        (query, value, decay)
    }

    fn max_abs_4(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> f32 {
        lhs.sub(rhs).abs().max().into_scalar()
    }

    fn max_abs_3(lhs: Tensor<Backend, 3>, rhs: Tensor<Backend, 3>) -> f32 {
        lhs.sub(rhs).abs().max().into_scalar()
    }

    #[test]
    fn rwkv8_query_weights_sum_to_one() {
        let device = <Backend as BackendTrait>::Device::default();
        let query_t = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.2, 0.3, 0.5, 0.4, 0.1, 0.5], [1, 2, 3]),
            &device,
        );
        let weights = rwkv8_query_weights(query_t);
        let sums = weights
            .sum_dim(2)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap();
        for actual in sums {
            assert!((actual - 1.0).abs() <= 1.0e-6, "weight sum {actual}");
        }
    }

    #[test]
    fn rwkv8_step_helpers_match_single_step_reference() {
        let device = <Backend as BackendTrait>::Device::default();
        let query_t = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.4, 0.6, 0.3, 0.7], [1, 1, 4]),
            &device,
        );
        let value_t = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.25, 0.5, 0.75], [1, 1, 3]),
            &device,
        );
        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.5, 1.0, 1.5, 0.75, 1.25, 1.75, 1.0, 1.5, 2.0, 1.25, 1.75, 2.25,
                ],
                [1, 1, 4, 3],
            ),
            &device,
        );
        let rho_norm = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![1.2, 0.8, 1.5, 0.9], [1, 1, 4]),
            &device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.95, 0.9, 0.85, 0.8], [1, 1, 4]),
            &device,
        );

        let helper_context =
            rwkv8_read_context_step(rho.clone(), rho_norm.clone(), query_t.clone());
        let (helper_rho, helper_rho_norm) = rwkv8_update_state_step(
            rho.clone(),
            rho_norm.clone(),
            query_t.clone(),
            value_t.clone(),
            decay.clone(),
        );

        let (reference_context, reference_rho, reference_rho_norm) =
            recurrent_rwkv8_state_space_reference(
                query_t.reshape([1, 1, 1, 4]),
                value_t.reshape([1, 1, 1, 3]),
                Some(rho),
                Some(rho_norm),
                decay,
            );

        assert!(max_abs_4(helper_context, reference_context) <= 1.0e-6);
        assert!(max_abs_4(helper_rho, reference_rho) <= 1.0e-6);
        assert!(max_abs_3(helper_rho_norm, reference_rho_norm) <= 1.0e-6);
    }

    #[test]
    fn rwkv8_reference_step_state_matches_full_sequence() {
        let device = <Backend as BackendTrait>::Device::default();
        let (query, value, decay) = test_inputs(&device);
        let [_, _, time, _] = query.shape().dims::<4>();

        let (full_out, full_rho, full_rho_norm) = recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );

        let mut outputs = Vec::with_capacity(time);
        let mut rho_state = None;
        let mut rho_norm_state = None;
        for step in 0..time {
            let step_query = query.clone().slice_dim(2, step..step + 1);
            let step_value = value.clone().slice_dim(2, step..step + 1);
            let (step_out, next_rho, next_rho_norm) = recurrent_rwkv8_state_space_reference(
                step_query,
                step_value,
                rho_state,
                rho_norm_state,
                decay.clone(),
            );
            outputs.push(step_out);
            rho_state = Some(next_rho);
            rho_norm_state = Some(next_rho_norm);
        }

        let step_out = Tensor::cat(outputs, 2);
        let step_rho = rho_state.expect("rwkv rho state");
        let step_rho_norm = rho_norm_state.expect("rwkv rho_norm state");

        assert!(max_abs_4(step_out, full_out) <= 1.0e-6);
        assert!(max_abs_4(step_rho, full_rho) <= 1.0e-6);
        assert!(max_abs_3(step_rho_norm, full_rho_norm) <= 1.0e-6);
    }

    #[test]
    fn rwkv8_reference_chunked_state_matches_full_sequence() {
        let device = <Backend as BackendTrait>::Device::default();
        let (query, value, decay) = test_inputs(&device);

        let (full_out, full_rho, full_rho_norm) = recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );

        let (prefix_out, prefix_rho, prefix_rho_norm) = recurrent_rwkv8_state_space_reference(
            query.clone().slice_dim(2, 0..2),
            value.clone().slice_dim(2, 0..2),
            None,
            None,
            decay.clone(),
        );
        let (suffix_out, suffix_rho, suffix_rho_norm) = recurrent_rwkv8_state_space_reference(
            query.clone().slice_dim(2, 2..5),
            value.clone().slice_dim(2, 2..5),
            Some(prefix_rho),
            Some(prefix_rho_norm),
            decay.clone(),
        );

        let chunked_out = Tensor::cat(vec![prefix_out, suffix_out], 2);
        assert!(max_abs_4(chunked_out, full_out) <= 1.0e-6);
        assert!(max_abs_4(suffix_rho, full_rho) <= 1.0e-6);
        assert!(max_abs_3(suffix_rho_norm, full_rho_norm) <= 1.0e-6);
    }
}
