use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

const DENSE_SCORE_REFERENCE_ROW_CHUNK: usize = 256;

pub fn expand_attention_values_to_heads<B: Backend>(
    value: Tensor<B, 4>,
    heads: usize,
) -> Tensor<B, 4> {
    match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    }
}

pub fn recurrent_attention_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let decay = decay.map(|tensor| tensor.reshape([1, heads, 1, 1]));

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

    let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

    for t in 0..time {
        let x_t = query.clone().slice_dim(2, t..t + 1);
        let v_t = value.clone().slice_dim(2, t..t + 1).repeat_dim(1, heads);
        let x_t_latent = x_t.swap_dims(2, 3);

        let attn_t = (rho.clone() * x_t_latent.clone())
            .sum_dim(2)
            .reshape([batch, heads, 1, n_embd]);
        outputs.push(attn_t);

        rho = rho + x_t_latent * v_t;
        if let Some(decay) = &decay {
            rho = rho * decay.clone();
        }
    }

    (Tensor::cat(outputs, 2), rho)
}

pub fn recurrent_attention_dense_score_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();

    if time <= DENSE_SCORE_REFERENCE_ROW_CHUNK {
        return recurrent_attention_dense_score_reference_full(query, value, rho_state, decay);
    }

    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);
    let query_key = query.clone().swap_dims(2, 3);
    let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, 1, time]);
    let decay_heads = decay.clone().map(|tensor| tensor.reshape([1, heads, 1, 1]));

    let rho = recurrent_attention_dense_score_final_rho_reference(
        query.clone(),
        value.clone(),
        rho_state.clone(),
        decay.clone(),
    );

    let mut outputs: Vec<Tensor<B, 4>> =
        Vec::with_capacity(time.div_ceil(DENSE_SCORE_REFERENCE_ROW_CHUNK));
    for start in (0..time).step_by(DENSE_SCORE_REFERENCE_ROW_CHUNK) {
        let end = (start + DENSE_SCORE_REFERENCE_ROW_CHUNK).min(time);
        let rows = end.saturating_sub(start);
        let q_chunk = query.clone().slice_dim(2, start..end);
        let mut score_chunk = q_chunk
            .clone()
            .matmul(query_key.clone())
            .tril(start as i64 - 1);
        let initial_context_chunk = if let Some(decay_heads) = decay_heads.clone() {
            let pos_row = Tensor::<B, 1, Int>::arange(start as i64..end as i64, &device)
                .float()
                .reshape([1, 1, rows, 1]);
            let diff = (pos_row.clone() - pos_col.clone())
                .tril(start as i64 - 1)
                .repeat_dim(1, heads);
            let decay_score = decay_heads.clone().repeat_dim(2, rows).repeat_dim(3, time);
            score_chunk = score_chunk * decay_score.powf(diff);

            if let Some(rho_state) = rho_state.clone() {
                let decay_state = decay_heads
                    .clone()
                    .repeat_dim(2, rows)
                    .powf(pos_row.repeat_dim(1, heads));
                q_chunk
                    .clone()
                    .mul(decay_state)
                    .matmul(rho_state)
                    .reshape([batch, heads, rows, n_embd])
            } else {
                Tensor::<B, 4>::zeros([batch, heads, rows, n_embd], &device)
            }
        } else if let Some(rho_state) = rho_state.clone() {
            q_chunk
                .clone()
                .matmul(rho_state)
                .reshape([batch, heads, rows, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, rows, n_embd], &device)
        };

        let chunk_context = initial_context_chunk
            + score_chunk
                .matmul(value.clone())
                .reshape([batch, heads, rows, n_embd]);
        outputs.push(chunk_context);
    }

    (Tensor::cat(outputs, 2), rho)
}

fn recurrent_attention_dense_score_reference_full<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> (Tensor<B, 4>, Tensor<B, 4>) {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
        .float()
        .reshape([1, 1, 1, time]);

    let mut scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
    let (initial_context, rho) = if let Some(decay) = decay {
        let diff = (pos_row.clone() - pos_col.clone())
            .tril(-1)
            .repeat_dim(1, heads);
        let decay_score = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .repeat_dim(3, time);
        scores = scores * decay_score.powf(diff);

        let state_exp = pos_row.clone().repeat_dim(1, heads);
        let decay_state = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(state_exp.clone());
        let initial_context = if let Some(rho_state) = rho_state.clone() {
            query
                .clone()
                .mul(decay_state.clone())
                .matmul(rho_state)
                .reshape([batch, heads, time, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device)
        };

        let final_exponents = pos_row
            .mul_scalar(-1.0)
            .add_scalar(time as f32)
            .repeat_dim(1, heads);
        let decay_final = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(final_exponents);
        let rho = if let Some(rho_state) = rho_state {
            rho_state.mul(
                decay
                    .clone()
                    .reshape([1, heads, 1, 1])
                    .powf_scalar(time as f32),
            ) + query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
        } else {
            query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
        };

        (initial_context, rho)
    } else {
        let initial_context = if let Some(rho_state) = rho_state.clone() {
            query
                .clone()
                .matmul(rho_state)
                .reshape([batch, heads, time, n_embd])
        } else {
            Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device)
        };
        let rho = if let Some(rho_state) = rho_state {
            rho_state + query.swap_dims(2, 3).matmul(value.clone())
        } else {
            query.swap_dims(2, 3).matmul(value.clone())
        };
        (initial_context, rho)
    };

    let context = initial_context + scores.matmul(value).reshape([batch, heads, time, n_embd]);
    (context, rho)
}

pub fn recurrent_attention_dense_score_final_rho_reference<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let n_embd = value.shape().dims::<4>()[3];
    let device = value.device();
    let value = expand_attention_values_to_heads(value, heads);
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    if let Some(decay) = decay {
        let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);
        let final_exponents = pos_row
            .mul_scalar(-1.0)
            .add_scalar(time as f32)
            .repeat_dim(1, heads);
        let decay_final = decay
            .clone()
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(final_exponents);
        let contribution = query.mul(decay_final).swap_dims(2, 3).matmul(value);
        if let Some(rho_state) = rho_state {
            rho_state.mul(decay.reshape([1, heads, 1, 1]).powf_scalar(time as f32)) + contribution
        } else {
            contribution
        }
    } else {
        let contribution = query.swap_dims(2, 3).matmul(value);
        if let Some(rho_state) = rho_state {
            rho_state + contribution
        } else {
            contribution
        }
    }
}

pub fn recurrent_attention_dense_score_initial_context_reference<B: Backend>(
    query: Tensor<B, 4>,
    rho_state: Option<Tensor<B, 4>>,
    decay: Option<Tensor<B, 1>>,
    n_embd: usize,
) -> Tensor<B, 4> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let device = query.device();
    let rho_state =
        rho_state.filter(|state| state.shape().dims::<4>() == [batch, heads, latent, n_embd]);

    if let Some(decay) = decay {
        let Some(rho_state) = rho_state else {
            return Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device);
        };
        let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);
        let decay_state = decay
            .reshape([1, heads, 1, 1])
            .repeat_dim(2, time)
            .powf(pos_row.repeat_dim(1, heads));
        query
            .mul(decay_state)
            .matmul(rho_state)
            .reshape([batch, heads, time, n_embd])
    } else {
        let Some(rho_state) = rho_state else {
            return Tensor::<B, 4>::zeros([batch, heads, time, n_embd], &device);
        };
        query
            .matmul(rho_state)
            .reshape([batch, heads, time, n_embd])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn tensor4(values: Vec<f32>, shape: [usize; 4]) -> Tensor<TestBackend, 4> {
        Tensor::<TestBackend, 4>::from_data(TensorData::new(values, shape), &Default::default())
    }

    fn max_abs_diff(lhs: Tensor<TestBackend, 4>, rhs: Tensor<TestBackend, 4>) -> f32 {
        let lhs = lhs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs = rhs
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");
        lhs.into_iter()
            .zip(rhs)
            .map(|(left, right)| (left - right).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn chunked_dense_score_reference_matches_full_without_decay() {
        let shape = [2, 3, 320, 8];
        let value_shape = [2, 1, 320, 8];
        let query = tensor4(
            (0..shape.iter().product::<usize>())
                .map(|index| (index % 97) as f32 / 97.0)
                .collect(),
            shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 3) % 89) as f32 / 89.0)
                .collect(),
            value_shape,
        );

        let (chunked_context, chunked_rho) =
            recurrent_attention_dense_score_reference(query.clone(), value.clone(), None, None);
        let (full_context, full_rho) =
            recurrent_attention_dense_score_reference_full(query, value, None, None);

        assert!(max_abs_diff(chunked_context, full_context) < 1.0e-4);
        assert!(max_abs_diff(chunked_rho, full_rho) < 1.0e-4);
    }

    #[test]
    fn chunked_dense_score_reference_matches_full_with_decay_and_state() {
        let shape = [1, 4, 384, 6];
        let value_shape = [1, 1, 384, 5];
        let rho_shape = [1, 4, 6, 5];
        let query = tensor4(
            (0..shape.iter().product::<usize>())
                .map(|index| ((index * 5) % 113) as f32 / 113.0)
                .collect(),
            shape,
        );
        let value = tensor4(
            (0..value_shape.iter().product::<usize>())
                .map(|index| ((index * 7) % 101) as f32 / 101.0)
                .collect(),
            value_shape,
        );
        let rho_state = tensor4(
            (0..rho_shape.iter().product::<usize>())
                .map(|index| ((index * 11) % 79) as f32 / 79.0)
                .collect(),
            rho_shape,
        );
        let decay = Tensor::<TestBackend, 1>::from_data(
            TensorData::new(vec![0.91f32, 0.93, 0.95, 0.97], [4]),
            &Default::default(),
        );

        let (chunked_context, chunked_rho) = recurrent_attention_dense_score_reference(
            query.clone(),
            value.clone(),
            Some(rho_state.clone()),
            Some(decay.clone()),
        );
        let (full_context, full_rho) = recurrent_attention_dense_score_reference_full(
            query,
            value,
            Some(rho_state),
            Some(decay),
        );

        assert!(max_abs_diff(chunked_context, full_context) < 2.0e-4);
        assert!(max_abs_diff(chunked_rho, full_rho) < 2.0e-4);
    }
}
