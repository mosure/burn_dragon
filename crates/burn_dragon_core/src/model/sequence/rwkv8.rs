use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

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
        let q_weights = q_t.clone().div(
            q_t.clone()
                .sum_dim(2)
                .add_scalar(1.0e-6)
                .reshape([batch, heads, 1]),
        );
        let value_t = value
            .clone()
            .slice_dim(2, t..t + 1)
            .repeat_dim(1, heads)
            .reshape([batch, heads, n_embd]);

        let read = rho.clone().div(
            rho_norm
                .clone()
                .add_scalar(1.0e-6)
                .reshape([batch, heads, latent, 1]),
        );
        let context_t = (read * q_weights.reshape([batch, heads, latent, 1]))
            .sum_dim(2)
            .reshape([batch, heads, 1, n_embd]);
        outputs.push(context_t);

        rho = rho.mul(decay.clone().reshape([1, heads, latent, 1])).add(
            q_t.clone().reshape([batch, heads, latent, 1])
                * value_t.reshape([batch, heads, 1, n_embd]),
        );
        rho_norm = rho_norm.mul(decay.clone()).add(q_t);
    }

    (Tensor::cat(outputs, 2), rho, rho_norm)
}
