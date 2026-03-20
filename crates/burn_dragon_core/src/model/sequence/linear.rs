use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

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
