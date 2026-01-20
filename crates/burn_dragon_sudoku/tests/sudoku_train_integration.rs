#![cfg(feature = "integration_test")]

use std::fs;
use std::path::Path;

use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use tempfile::tempdir;

use burn_dragon_sudoku::config::{
    SudokuArtifactConfig, SudokuDatasetConfig, SudokuDatasetSourceConfig, SudokuLocalConfig,
    SudokuModelConfig, SudokuRecordFormat, SudokuTrainingConfig, SudokuTrainingHyperparameters,
};
use burn_dragon_sudoku::train::{loss_trace_reset, loss_trace_take, train_backend_for_test};
use burn_dragon_train::{GdpoConfig, OptimizerConfig, WgpuRuntimeConfig};

fn write_dataset(root: &Path) {
    let payload = [
        r#"{"puzzle":"530070000600195000098000060800060003400803001700020006060000280000419005000080079","solution":"534678912672195348198342567859761423426853791713924856961537284287419635345286179"}"#,
        r#"{"puzzle":"003020600900305001001806400008102900700000008006708200002609500800203009005010300","solution":"483921657967345821251876493548132976729564138136798245372689514814253769695417382"}"#,
        r#"{"puzzle":"200080300060070084030500209000105408000000000402706000301007040720040060004010003","solution":"245986371169273584837541269976135428513824697482796135391657842728349156654412793"}"#,
        r#"{"puzzle":"000000907000420180000705026100904000050000040000507009920108000034059000507000000","solution":"483651927659423181217895326176934852952781643348567219926178534834259761571346298"}"#,
    ]
    .join("\n");
    let path = root.join("train.jsonl");
    fs::write(&path, payload).expect("write dataset");
}

#[test]
fn cpu_sudoku_training_loss_decreases() {
    let dir = tempdir().expect("tempdir");
    write_dataset(dir.path());

    let dataset = SudokuDatasetConfig {
        cache_dir: dir.path().to_path_buf(),
        train_split_ratio: 0.9,
        source: SudokuDatasetSourceConfig::Local(SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: Vec::new(),
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            max_records: Some(64),
        }),
    };

    let training = SudokuTrainingHyperparameters {
        batch_size: 4,
        epochs: None,
        max_iters: 40,
        log_frequency: 5,
        rollout_steps: 4,
        halt_weight: 0.1,
        policy_noise: 0.2,
        gdpo: GdpoConfig {
            enabled: true,
            group_size: 2,
            ..GdpoConfig::default()
        },
    };

    let config = SudokuTrainingConfig {
        dataset,
        training,
        optimizer: OptimizerConfig {
            learning_rate: 0.005,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        artifacts: SudokuArtifactConfig::default(),
        wgpu: WgpuRuntimeConfig::default(),
        model: SudokuModelConfig {
            n_layer: 2,
            n_embd: 64,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
        },
    };

    loss_trace_reset();
    let result = train_backend_for_test::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {});
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let losses = loss_trace_take();
    assert!(
        !losses.is_empty(),
        "expected loss trace samples from integration run"
    );
    let first = losses.first().copied().unwrap_or(f32::INFINITY);
    let min_loss = losses
        .iter()
        .copied()
        .fold(f32::INFINITY, |acc, value| acc.min(value));
    assert!(first.is_finite(), "initial loss not finite: {first}");
    assert!(min_loss.is_finite(), "min loss not finite: {min_loss}");
    assert!(
        min_loss < first,
        "expected loss to decrease at least once (initial={first}, min={min_loss})"
    );
}
