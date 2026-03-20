use burn::nn::Dropout;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::kernel::{BlockPattern1d, relu_lowrank};

#[derive(Debug)]
pub struct LowRankResidualOutput<B: Backend> {
    pub next: Tensor<B, 4>,
    pub x_neuron: Tensor<B, 4>,
    pub y_gate: Tensor<B, 4>,
    pub y_neuron: Tensor<B, 4>,
}

struct LowRankResidualInternal<B: Backend> {
    next: Tensor<B, 4>,
    x_neuron: Option<Tensor<B, 4>>,
    y_gate: Option<Tensor<B, 4>>,
    y_neuron: Option<Tensor<B, 4>>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LowRankResidualProfileSnapshot {
    pub calls: u64,
    pub total_ns: u128,
    pub attention_norm_ns: u128,
    pub decoder_tail_ns: u128,
    pub mlp_norm_ns: u128,
    pub residual_combine_ns: u128,
}

static LOWRANK_RESIDUAL_PROFILE: OnceLock<Mutex<LowRankResidualProfileSnapshot>> = OnceLock::new();
static LOWRANK_RESIDUAL_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();

fn lowrank_residual_profile_enabled() -> bool {
    *LOWRANK_RESIDUAL_PROFILE_ENABLED
        .get_or_init(|| std::env::var_os("BDH_STAGE_PROFILE").is_some())
}

fn lowrank_residual_profile_state() -> &'static Mutex<LowRankResidualProfileSnapshot> {
    LOWRANK_RESIDUAL_PROFILE.get_or_init(|| Mutex::new(LowRankResidualProfileSnapshot::default()))
}

pub fn lowrank_residual_profile_reset() {
    if let Ok(mut state) = lowrank_residual_profile_state().lock() {
        *state = LowRankResidualProfileSnapshot::default();
    }
}

pub fn lowrank_residual_profile_snapshot() -> LowRankResidualProfileSnapshot {
    lowrank_residual_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

fn lowrank_residual_profile_record(
    total_ns: u128,
    attention_norm_ns: u128,
    decoder_tail_ns: u128,
    mlp_norm_ns: u128,
    residual_combine_ns: u128,
) {
    if let Ok(mut state) = lowrank_residual_profile_state().lock() {
        state.calls = state.calls.saturating_add(1);
        state.total_ns = state.total_ns.saturating_add(total_ns);
        state.attention_norm_ns = state.attention_norm_ns.saturating_add(attention_norm_ns);
        state.decoder_tail_ns = state.decoder_tail_ns.saturating_add(decoder_tail_ns);
        state.mlp_norm_ns = state.mlp_norm_ns.saturating_add(mlp_norm_ns);
        state.residual_combine_ns = state
            .residual_combine_ns
            .saturating_add(residual_combine_ns);
    }
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
    let mixed_by_head = y_neuron
        .swap_dims(0, 1)
        .reshape([heads, batch * time, latent]);
    mixed_by_head
        .matmul(decoder_by_head)
        .sum_dim(0)
        .reshape([batch, 1, time, dim])
}

#[allow(clippy::too_many_arguments)]
fn lowrank_residual_step_impl<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    mut attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
    keep_aux: bool,
) -> LowRankResidualInternal<B>
where
    B: Backend,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let prof_enabled = lowrank_residual_profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let mut attention_norm_ns = 0;
    let mut decoder_tail_ns = 0;
    let mut mlp_norm_ns = 0;
    let mut residual_combine_ns = 0;

    let use_fused_any = use_fused_x || use_fused_y;
    let x_grad_input_executor = match lowrank_grad_input_executor {
        LowrankGradInputExecutor::KernelTiled => LowrankGradInputExecutor::AlignedMatmul,
        other => other,
    };
    let y_grad_input_executor = lowrank_grad_input_executor;
    let sparse_mask = if use_fused_any && latent_pattern.is_sparse() {
        sparse_mask.or_else(|| {
            let latent = encoder.shape().dims::<4>()[3];
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        })
    } else {
        None
    };

    let x_neuron = if use_fused_x {
        relu_lowrank::fused_forward_with_executor(
            current.clone(),
            encoder.clone(),
            None,
            relu_threshold,
            latent_pattern,
            sparse_mask.clone(),
            x_grad_input_executor,
        )
    } else {
        let mut x_latent = current.clone().matmul(encoder);
        if apply_threshold && relu_threshold != 0.0 {
            x_latent = x_latent.sub_scalar(relu_threshold);
        }
        apply_latent(x_latent)
    };

    let attention_start = prof_enabled.then(Instant::now);
    let attn = attention(x_neuron.clone(), current.clone());
    let attn = apply_norm(attn);
    if let Some(start) = attention_start {
        attention_norm_ns = start.elapsed().as_nanos();
    }

    let y_gate = if use_fused_y {
        relu_lowrank::fused_forward_with_executor(
            attn.clone(),
            encoder_v,
            None,
            relu_threshold,
            latent_pattern,
            sparse_mask,
            y_grad_input_executor,
        )
    } else {
        let mut y_latent = attn.matmul(encoder_v);
        if apply_threshold && relu_threshold != 0.0 {
            y_latent = y_latent.sub_scalar(relu_threshold);
        }
        apply_latent(y_latent)
    };

    let (y_neuron, x_neuron_out, y_gate_out, y_neuron_out) = if keep_aux {
        let y_neuron = dropout.forward(x_neuron.clone() * y_gate.clone());
        (
            y_neuron.clone(),
            Some(x_neuron),
            Some(y_gate),
            Some(y_neuron),
        )
    } else {
        let y_neuron = dropout.forward(x_neuron * y_gate);
        (y_neuron, None, None, None)
    };
    let decoder_tail_start = prof_enabled.then(Instant::now);
    let mlp_out = decode_y_neuron_tail(y_neuron.clone(), decoder);
    if let Some(start) = decoder_tail_start {
        decoder_tail_ns = start.elapsed().as_nanos();
    }
    let mlp_norm_start = prof_enabled.then(Instant::now);
    let mlp_out = apply_norm(mlp_out);
    if let Some(start) = mlp_norm_start {
        mlp_norm_ns = start.elapsed().as_nanos();
    }
    let residual_combine_start = prof_enabled.then(Instant::now);
    let next = apply_norm(current + mlp_out);
    if let Some(start) = residual_combine_start {
        residual_combine_ns = start.elapsed().as_nanos();
    }

    if let Some(start) = total_start {
        lowrank_residual_profile_record(
            start.elapsed().as_nanos(),
            attention_norm_ns,
            decoder_tail_ns,
            mlp_norm_ns,
            residual_combine_ns,
        );
    }

    LowRankResidualInternal {
        next,
        x_neuron: x_neuron_out,
        y_gate: y_gate_out,
        y_neuron: y_neuron_out,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> LowRankResidualOutput<B>
where
    B: Backend,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    let output = lowrank_residual_step_impl(
        current,
        encoder,
        encoder_v,
        decoder,
        dropout,
        use_fused_x,
        use_fused_y,
        relu_threshold,
        apply_threshold,
        latent_pattern,
        lowrank_grad_input_executor,
        sparse_mask,
        attention,
        apply_latent,
        apply_norm,
        true,
    );
    LowRankResidualOutput {
        next: output.next,
        x_neuron: output.x_neuron.expect("x_neuron for full residual output"),
        y_gate: output.y_gate.expect("y_gate for full residual output"),
        y_neuron: output.y_neuron.expect("y_neuron for full residual output"),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn lowrank_residual_step_next<B, FAttn, FNorm, FAct>(
    current: Tensor<B, 4>,
    encoder: Tensor<B, 4>,
    encoder_v: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    dropout: &Dropout,
    use_fused_x: bool,
    use_fused_y: bool,
    relu_threshold: f32,
    apply_threshold: bool,
    latent_pattern: &BlockPattern1d,
    lowrank_grad_input_executor: LowrankGradInputExecutor,
    sparse_mask: Option<Tensor<B, 4>>,
    attention: FAttn,
    apply_latent: FAct,
    apply_norm: FNorm,
) -> Tensor<B, 4>
where
    B: Backend,
    B::FloatTensorPrimitive: 'static,
    FAttn: FnMut(Tensor<B, 4>, Tensor<B, 4>) -> Tensor<B, 4>,
    FNorm: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
    FAct: Fn(Tensor<B, 4>) -> Tensor<B, 4>,
{
    lowrank_residual_step_impl(
        current,
        encoder,
        encoder_v,
        decoder,
        dropout,
        use_fused_x,
        use_fused_y,
        relu_threshold,
        apply_threshold,
        latent_pattern,
        lowrank_grad_input_executor,
        sparse_mask,
        attention,
        apply_latent,
        apply_norm,
        false,
    )
    .next
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
        for (index, (a, b)) in actual.into_iter().zip(expected).enumerate() {
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
            TensorData::new(
                (1..=24).map(|value| value as f32 * 0.1).collect::<Vec<_>>(),
                [2, 2, 2, 3],
            ),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (1..=30)
                    .map(|value| value as f32 * 0.05)
                    .collect::<Vec<_>>(),
                [6, 5],
            ),
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
            TensorData::new(
                (1..=12).map(|value| value as f32 * 0.2).collect::<Vec<_>>(),
                [2, 1, 2, 3],
            ),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (1..=12)
                    .map(|value| value as f32 * 0.04)
                    .collect::<Vec<_>>(),
                [3, 4],
            ),
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
            false,
            0.0,
            false,
            &BlockPattern1d::dense(2),
            LowrankGradInputExecutor::Auto,
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

    #[test]
    fn lowrank_residual_step_next_matches_full_output_next() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let current = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=8).map(|value| value as f32 * 0.1).collect::<Vec<_>>(),
                [1, 1, 2, 4],
            ),
            &device,
        );
        let encoder = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=32)
                    .map(|value| value as f32 * 0.02)
                    .collect::<Vec<_>>(),
                [1, 1, 4, 8],
            ),
            &device,
        );
        let encoder_v = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (1..=32)
                    .map(|value| value as f32 * 0.03)
                    .collect::<Vec<_>>(),
                [1, 1, 4, 8],
            ),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                (1..=32)
                    .map(|value| value as f32 * 0.01)
                    .collect::<Vec<_>>(),
                [8, 4],
            ),
            &device,
        );
        let dropout = DropoutConfig::new(0.0).init();
        let layout = BlockPattern1d::dense(8);

        let full = lowrank_residual_step(
            current.clone(),
            encoder.clone(),
            encoder_v.clone(),
            decoder.clone(),
            &dropout,
            false,
            false,
            0.0,
            false,
            &layout,
            LowrankGradInputExecutor::Auto,
            None,
            |_query, current| current,
            |values| values,
            |values| values,
        )
        .next
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("full next vec");

        let next_only = lowrank_residual_step_next(
            current,
            encoder,
            encoder_v,
            decoder,
            &dropout,
            false,
            false,
            0.0,
            false,
            &layout,
            LowrankGradInputExecutor::Auto,
            None,
            |_query, current| current,
            |values| values,
            |values| values,
        )
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("next only vec");

        assert_eq!(next_only, full);
    }
}
