use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use burn_dragon::language::loss::language_model_loss;
use burn_dragon::{BDH, BDHConfig, FusedKernelConfig, RotaryEmbedding};

type TrainBackend = Autodiff<NdArray<f32>>;
type InferBackend = NdArray<f32>;

fn build_config(rotary: RotaryEmbedding, fused: bool) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: 2,
        n_embd: 16,
        n_head: 2,
        mlp_internal_dim_multiplier: 4,
        vocab_size: 32,
        dropout: 0.0,
        fused_kernels: FusedKernelConfig {
            enabled: fused,
            ..Default::default()
        },
        ..Default::default()
    };
    config.fused_kernels.set_block_sizes(4, 4);
    config.fused_kernels.set_rotary_embedding(rotary);
    config
}

fn sample_tokens<B: Backend>(device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
    let tokens = vec![0, 1, 2, 3, 4, 5, 6, 7];
    let targets = vec![1, 2, 3, 4, 5, 6, 7, 0];
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(tokens, [2, 4]), device);
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [2, 4]), device);
    (inputs, targets)
}

#[test]
fn training_paths_run_across_configs() {
    let device = <TrainBackend as Backend>::Device::default();
    let configs = [
        (RotaryEmbedding::Rope, false),
        (RotaryEmbedding::Pope, false),
        (RotaryEmbedding::Alibi, false),
        (RotaryEmbedding::Alibi, true),
    ];

    for (rotary, fused) in configs {
        let config = build_config(rotary, fused);
        let model = BDH::<TrainBackend>::new(config, &device);
        let (inputs, targets) = sample_tokens::<TrainBackend>(&device);

        let logits_fast = model.forward_fast(inputs.clone());
        let [batch, time, vocab] = logits_fast.shape().dims();
        assert_eq!([batch, time, vocab], [2, 4, 32]);

        let loss_fast = language_model_loss::<TrainBackend>(logits_fast, targets.clone());
        let _ = loss_fast.backward();

        let mut state = model.init_state();
        let logits_rec = model.forward_with_state(inputs.clone(), &mut state);
        let [batch, time, vocab] = logits_rec.shape().dims();
        assert_eq!([batch, time, vocab], [2, 4, 32]);
        assert_eq!(state.position, 4);

        let loss_rec = language_model_loss::<TrainBackend>(logits_rec, targets.clone());
        let _ = loss_rec.backward();
    }
}

#[test]
fn inference_paths_run_across_configs() {
    let device = <InferBackend as Backend>::Device::default();
    let configs = [
        (RotaryEmbedding::Rope, false),
        (RotaryEmbedding::Pope, false),
        (RotaryEmbedding::Alibi, false),
        (RotaryEmbedding::Alibi, true),
    ];

    for (rotary, fused) in configs {
        let config = build_config(rotary, fused);
        let model = BDH::<InferBackend>::new(config, &device);
        let (inputs, _targets) = sample_tokens::<InferBackend>(&device);

        let logits_fast = model.forward_fast(inputs.clone());
        let [batch, time, vocab] = logits_fast.shape().dims();
        assert_eq!([batch, time, vocab], [2, 4, 32]);

        let mut state = model.init_state();
        let logits_rec = model.forward_with_state(inputs.clone(), &mut state);
        let [batch, time, vocab] = logits_rec.shape().dims();
        assert_eq!([batch, time, vocab], [2, 4, 32]);
        assert_eq!(state.position, 4);
    }
}
