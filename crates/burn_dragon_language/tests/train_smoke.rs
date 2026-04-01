#![cfg(feature = "train")]

use std::path::PathBuf;

use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;

use burn_dragon_core::{BDH, LanguageHeadConfig};
use burn_dragon_language::loss::language_model_loss;
use burn_dragon_language::{build_model_config, load_training_config};

type TrainBackend = Autodiff<NdArray<f32>>;

#[test]
fn training_forward_backward_from_configs() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let config_paths = [
        repo_root.join("config/language/base.toml"),
        repo_root.join("config/language/baselines/small.toml"),
    ];
    let config = load_training_config(&config_paths).expect("load training config");
    let mut model_config = build_model_config(&config.model, config.training.block_size);
    model_config.vocab_size = 64;

    let device = <TrainBackend as Backend>::Device::default();
    let model = BDH::<TrainBackend>::new(model_config, &device);

    let batch = 2;
    let time = 4;
    let inputs = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 2, 3, 4, 5, 6, 7], [batch, time]),
        &device,
    );
    let targets = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![1, 2, 3, 4, 5, 6, 7, 0], [batch, time]),
        &device,
    );

    let logits = model.forward(inputs.clone());
    let loss = language_model_loss::<TrainBackend>(logits, targets.clone());
    let _ = loss.backward();

    let logits_fast = model.forward_fast(inputs.clone());
    let loss_fast = language_model_loss::<TrainBackend>(logits_fast, targets);
    let _ = loss_fast.backward();
}

#[test]
fn nca_factorized_head_forward_backward_smoke() {
    let device = <TrainBackend as Backend>::Device::default();
    let mut model_config = burn_dragon_core::BDHConfig::default();
    model_config.n_layer = 2;
    model_config.n_embd = 8;
    model_config.n_head = 1;
    model_config.mlp_internal_dim_multiplier = 1;
    model_config.dropout = 0.0;
    model_config.vocab_size = 19;
    model_config.language_head = LanguageHeadConfig::NcaFactorizedPatch {
        state_count: 2,
        patch_size: 2,
        frame_special_tokens: true,
        eos_id: Some(18),
    };

    let model = BDH::<TrainBackend>::new(model_config, &device);

    let inputs = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![16, 0, 1, 17, 16, 2, 3, 18], [2, 4]),
        &device,
    );
    let targets = Tensor::<TrainBackend, 2, Int>::from_data(
        TensorData::new(vec![0, 1, 17, 18, 2, 3, 16, 18], [2, 4]),
        &device,
    );

    let hidden = model.forward_hidden(inputs.clone());
    let loss = model.language_loss_from_hidden(hidden, targets.clone());
    let _ = loss.backward();

    model.language_loss_from_hidden(model.forward_hidden(inputs), targets);
}
