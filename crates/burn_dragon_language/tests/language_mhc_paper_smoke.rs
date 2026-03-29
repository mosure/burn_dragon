#![cfg(feature = "train")]

use std::fs;

use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn::tensor::backend::Backend as BackendTrait;
use burn_autodiff::Autodiff;
use burn_dragon_core::{
    BDH, BDHConfig, ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionsConfig,
};
use burn_dragon_language::dataset::{ShakespeareDataset, ShakespeareSplit};
use burn_dragon_language::loss::language_model_loss;
use burn_dragon_language::tokenizer::TokenizerConfig;
use burn_ndarray::NdArray;
use tempfile::tempdir;

type Backend = Autodiff<NdArray<f32>>;

#[derive(Clone)]
struct SmokeCase {
    name: &'static str,
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mhc: Option<ManifoldHyperConnectionsConfig>,
}

fn build_dataset() -> ShakespeareDataset {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.keep();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"Friends, Romans, countrymen, lend me your ears.\n".repeat(1024);
    fs::write(&file_path, content).expect("write dataset");
    ShakespeareDataset::new(cache_dir, 48, 8, 0.9, &TokenizerConfig::default()).expect("dataset")
}

fn run_case(case: SmokeCase) -> (f32, f32) {
    let dataset = build_dataset();
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 1337);

    let mut config = BDHConfig {
        n_layer: case.n_layer,
        n_embd: case.n_embd,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        ..Default::default()
    };
    if let Some(mhc) = case.mhc {
        config.mhc = mhc;
    }
    config.vocab_size = dataset.tokenizer().len();

    let mut model = BDH::<Backend>::new(config, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.05)
        .init::<Backend, BDH<Backend>>();
    let lr: LearningRate = 5e-4;
    let mut initial_loss = None;
    let mut final_loss = 0.0f32;

    for step in 0..8 {
        let batch = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);
        let logits = model.forward(batch.inputs.clone());
        let loss = language_model_loss::<Backend>(logits, batch.targets.clone());
        let loss_scalar = loss
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        if step == 0 {
            initial_loss = Some(loss_scalar);
        }
        final_loss = loss_scalar;
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &model);
        model = optimizer.step(lr, model, grads);
    }

    (initial_loss.expect("initial loss"), final_loss)
}

#[test]
fn language_mhc_smoke_reports_ablation() {
    let cases = [
        SmokeCase {
            name: "baseline",
            n_layer: 2,
            n_embd: 32,
            n_head: 2,
            mhc: None,
        },
        SmokeCase {
            name: "static_streams2",
            n_layer: 2,
            n_embd: 32,
            n_head: 2,
            mhc: Some(ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
        },
        SmokeCase {
            name: "dynamic_streams2",
            n_layer: 2,
            n_embd: 32,
            n_head: 2,
            mhc: Some(ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                coefficient_policy: ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
        },
    ];

    for case in cases {
        let (initial_loss, final_loss) = run_case(case.clone());
        eprintln!(
            "SMOKE_RESULT name={} initial_loss={:.6} final_loss={:.6}",
            case.name, initial_loss, final_loss
        );
        assert!(initial_loss.is_finite());
        assert!(final_loss.is_finite());
    }
}
