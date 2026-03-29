use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use burn_dragon::core::{BDH, BDHConfig, FusedKernelConfig, RotaryEmbedding};
use burn_dragon::language::loss::language_model_loss;
use burn_dragon::language::{
    ContextStrategy, GenerationSettings, generate_tokens, generate_tokens_chunked,
};

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

fn assert_close<const D: usize>(lhs: Tensor<InferBackend, D>, rhs: Tensor<InferBackend, D>) {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        assert!(
            diff <= 1e-5,
            "difference {diff} exceeds tolerance 1e-5 (lhs={a}, rhs={b})"
        );
    }
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
        for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
            let mut config = build_config(rotary, fused);
            config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
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
        for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
            let mut config = build_config(rotary, fused);
            config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
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
}

#[test]
fn rollout_keeps_slow_token_emission_semantics() {
    let device = <InferBackend as Backend>::Device::default();
    let mut config = build_config(RotaryEmbedding::Alibi, true);
    config.set_rollout_fast_steps_per_slow_step(8);
    let model = BDH::<InferBackend>::new(config, &device);

    let tokens = Tensor::<InferBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4, 5], [1, 6]),
        &device,
    );

    let logits_full = model.forward(tokens.clone());

    let mut state = model.init_state();
    let mut streamed = Vec::with_capacity(6);
    for step in 0..6 {
        let token_step = tokens.clone().slice_dim(1, step..step + 1);
        let logits = model.forward_with_state(token_step, &mut state);
        streamed.push(logits);
    }
    let logits_stream = Tensor::cat(streamed, 1);

    let [batch, time, vocab] = logits_stream.shape().dims::<3>();
    assert_eq!([batch, time, vocab], [1, 6, 32]);
    assert_eq!(state.position, 6);
    assert_close(logits_full, logits_stream);
}

#[test]
fn rollout_chunked_generation_matches_baseline_greedy_across_fast_steps() {
    let device = <InferBackend as Backend>::Device::default();
    let prompt = vec![0, 1, 2, 3, 4, 5];
    let settings = GenerationSettings {
        max_new_tokens: Some(24),
        temperature: 1.0,
        top_k: Some(1),
        strategy: ContextStrategy::Infinite,
    };

    for rollout_fast_steps in BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS {
        let mut config = build_config(RotaryEmbedding::Alibi, true);
        config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
        config.fused_kernels.set_wgpu_rollout_fused(true);
        let model = BDH::<InferBackend>::new(config, &device);

        let baseline =
            generate_tokens(&model, prompt.clone(), &device, settings, None).expect("baseline");
        let mut streamed = Vec::new();
        let chunked = generate_tokens_chunked(
            &model,
            prompt.clone(),
            &device,
            settings,
            4,
            16,
            None,
            Some(&mut |chunk: &[i64]| streamed.extend_from_slice(chunk)),
        )
        .expect("chunked");

        assert_eq!(
            baseline, chunked,
            "chunked generation diverged at rollout_fast_steps={rollout_fast_steps}"
        );
        assert_eq!(
            &chunked[prompt.len()..],
            streamed.as_slice(),
            "streamed chunks should reconstruct generated tail at rollout_fast_steps={rollout_fast_steps}"
        );
    }
}
