#![cfg(feature = "integration_test")]

use std::fs;
use std::path::Path;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use std::path::PathBuf;

use burn_autodiff::Autodiff;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_cuda::{Cuda, CudaDevice};
use burn_ndarray::NdArray;
use tempfile::tempdir;

#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn::tensor::backend::Backend;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::config::load_training_config;
use burn_dragon_sudoku::config::{
    SudokuArtifactConfig, SudokuCacheMhcConfig, SudokuCacheUpdateConfig, SudokuDatasetConfig,
    SudokuDatasetSourceConfig, SudokuGridPositional, SudokuHaltConfig, SudokuLocalConfig,
    SudokuLossMask, SudokuModelConfig, SudokuPolicyConfig, SudokuPolicyHead, SudokuReconConfig,
    SudokuReconLoss, SudokuRecordFormat, SudokuRevisitConfig, SudokuRewardConfig,
    SudokuRolloutConfig, SudokuTrainingConfig, SudokuTrainingHyperparameters,
    SudokuValidationConfig,
};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::dataset::SudokuBatch;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::model::SudokuSaccadeModel;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::train::SudokuTrainer;
use burn_dragon_sudoku::train::{
    halt_prob_trace_reset, halt_prob_trace_take, loss_trace_reset, loss_trace_take,
    solve_rate_trace_reset, solve_rate_trace_take, train_backend_for_test,
};
use burn_dragon_sudoku::vocab::GRID_LEN;
use burn_dragon_train::{GdpoConfig, OptimizerConfig, WgpuRuntimeConfig};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_train::TrainStep;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use cubecl::Runtime;

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

fn write_trivial_dataset(root: &Path) {
    let puzzle = "0".repeat(GRID_LEN);
    let solution = "1".repeat(GRID_LEN);
    let line = format!(r#"{{"puzzle":"{puzzle}","solution":"{solution}"}}"#);
    let train_payload = [line.as_str(), line.as_str()].join("\n");
    let valid_payload = line;

    fs::write(root.join("train.jsonl"), train_payload).expect("write train dataset");
    fs::write(root.join("valid.jsonl"), valid_payload).expect("write valid dataset");
}

#[test]
fn cpu_sudoku_training_loss_decreases() {
    let dir = tempdir().expect("tempdir");
    write_dataset(dir.path());

    let dataset = SudokuDatasetConfig {
        cache_dir: dir.path().to_path_buf(),
        train_split_ratio: 0.9,
        augment: false,
        augment_prob: 0.0,
        source: SudokuDatasetSourceConfig::Local(SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: Vec::new(),
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            train_max_records: None,
            validation_max_records: None,
            max_records: Some(64),
        }),
    };

    let training = SudokuTrainingHyperparameters {
        batch_size: 4,
        epochs: None,
        max_iters: 40,
        log_frequency: 5,
        rollout: SudokuRolloutConfig {
            steps: 4,
            min_steps: 0,
            max_steps: 0,
            max_steps_warmup_iters: 0,
            max_steps_warmup_cap: 0,
            backprop_steps: None,
            ..SudokuRolloutConfig::default()
        },
        halt: SudokuHaltConfig {
            weight: 0.1,
            exploration_prob: 0.0,
            min_steps: 1,
        },
        policy: SudokuPolicyConfig {
            noise: 0.2,
            epsilon: 0.0,
            epsilon_final: 0.0,
            epsilon_anneal_steps: 0,
            teacher_forcing_prob: 1.0,
            teacher_forcing_final: 1.0,
            teacher_forcing_anneal_steps: 0,
            temperature: 1.0,
            temperature_final: 1.0,
            temperature_anneal_steps: 0,
            entropy_weight: 0.0,
            entropy_weight_final: 0.0,
            entropy_anneal_steps: 0,
            entropy_adaptive: false,
            entropy_target_scale: 1.0,
            entropy_alpha: 0.0,
            entropy_alpha_lr: 0.0,
            visit_penalty: 0.0,
            revisit_penalty: 0.0,
            recon_weight: 0.0,
            ..SudokuPolicyConfig::default()
        },
        revisit: SudokuRevisitConfig {
            min_filled_frac: 0.0,
            min_filled_final: 0.0,
            min_filled_anneal_steps: 0,
        },
        reward: SudokuRewardConfig {
            unknown_power: 0.0,
            ..SudokuRewardConfig::default()
        },
        recon: SudokuReconConfig {
            loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            loss_interval_steps: 1,
            global_loss_samples: 8,
            global_loss_weight: 0.2,
            ..SudokuReconConfig::default()
        },
        validation: SudokuValidationConfig::default(),
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
            summary_tokens: 1,
            policy_heads: 1,
            policy_head: SudokuPolicyHead::Cache,
            policy_mlp_hidden_mult: 2,
            rotary_embedding: Default::default(),
            grid_positional: SudokuGridPositional::Additive,
            grid_rope_theta: 65_536.0,
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
            cache_mhc: SudokuCacheMhcConfig::default(),
            cache_update: SudokuCacheUpdateConfig::default(),
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

#[test]
fn cpu_sudoku_validation_solve_rate_gate() {
    let dir = tempdir().expect("tempdir");
    write_trivial_dataset(dir.path());

    let dataset = SudokuDatasetConfig {
        cache_dir: dir.path().to_path_buf(),
        train_split_ratio: 0.9,
        augment: false,
        augment_prob: 0.0,
        source: SudokuDatasetSourceConfig::Local(SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: vec!["valid.jsonl".to_string()],
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            train_max_records: None,
            validation_max_records: None,
            max_records: Some(32),
        }),
    };

    let training = SudokuTrainingHyperparameters {
        batch_size: 2,
        epochs: None,
        max_iters: 60,
        log_frequency: 10,
        rollout: SudokuRolloutConfig {
            steps: GRID_LEN * 2,
            min_steps: 0,
            max_steps: 0,
            max_steps_warmup_iters: 0,
            max_steps_warmup_cap: 0,
            backprop_steps: None,
            ..SudokuRolloutConfig::default()
        },
        halt: SudokuHaltConfig {
            weight: 0.2,
            exploration_prob: 0.0,
            min_steps: 1,
        },
        policy: SudokuPolicyConfig {
            noise: 0.0,
            epsilon: 0.0,
            epsilon_final: 0.0,
            epsilon_anneal_steps: 0,
            teacher_forcing_prob: 1.0,
            teacher_forcing_final: 0.0,
            teacher_forcing_anneal_steps: 10,
            temperature: 1.0,
            temperature_final: 1.0,
            temperature_anneal_steps: 0,
            entropy_weight: 0.0,
            entropy_weight_final: 0.0,
            entropy_anneal_steps: 0,
            entropy_adaptive: false,
            entropy_target_scale: 1.0,
            entropy_alpha: 0.0,
            entropy_alpha_lr: 0.0,
            visit_penalty: 0.0,
            revisit_penalty: 0.0,
            recon_weight: 0.0,
            ..SudokuPolicyConfig::default()
        },
        revisit: SudokuRevisitConfig {
            min_filled_frac: 1.0,
            min_filled_final: 1.0,
            min_filled_anneal_steps: 0,
        },
        reward: SudokuRewardConfig {
            unknown_power: 1.0,
            ..SudokuRewardConfig::default()
        },
        recon: SudokuReconConfig {
            loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            loss_interval_steps: 1,
            global_loss_samples: 8,
            global_loss_weight: 0.2,
            ..SudokuReconConfig::default()
        },
        validation: SudokuValidationConfig::default(),
        gdpo: GdpoConfig {
            enabled: false,
            group_size: 1,
            ..GdpoConfig::default()
        },
    };

    let config = SudokuTrainingConfig {
        dataset,
        training,
        optimizer: OptimizerConfig {
            learning_rate: 0.01,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        artifacts: SudokuArtifactConfig::default(),
        wgpu: WgpuRuntimeConfig::default(),
        model: SudokuModelConfig {
            n_layer: 1,
            n_embd: 32,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            summary_tokens: 1,
            policy_heads: 1,
            policy_head: SudokuPolicyHead::Cache,
            policy_mlp_hidden_mult: 2,
            rotary_embedding: Default::default(),
            grid_positional: SudokuGridPositional::Additive,
            grid_rope_theta: 65_536.0,
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
            cache_mhc: SudokuCacheMhcConfig::default(),
            cache_update: SudokuCacheUpdateConfig::default(),
        },
    };

    solve_rate_trace_reset();
    halt_prob_trace_reset();
    let result = train_backend_for_test::<Autodiff<NdArray<f32>>, _>(&config, "cpu", |_| {});
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let solve_rates = solve_rate_trace_take();
    let halt_probs = halt_prob_trace_take();
    assert!(
        !solve_rates.is_empty(),
        "expected solve rate trace samples from validation run"
    );
    assert!(
        !halt_probs.is_empty(),
        "expected halt prob trace samples from validation run"
    );

    let final_solve = *solve_rates.last().unwrap_or(&0.0);
    let final_halt = *halt_probs.last().unwrap_or(&0.0);
    assert!(
        final_solve >= 0.9,
        "expected final solve rate >= 0.9, got {final_solve}"
    );
    assert!(
        final_halt >= 0.1,
        "expected final halt prob >= 0.1, got {final_halt}"
    );
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
#[derive(Clone, Copy, Debug)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
fn cuda_snapshot(device: &CudaDevice) -> Option<MemorySnapshot> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let usage = <cubecl::cuda::CudaRuntime as Runtime>::client(device).memory_usage();
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }))
    .ok()
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
fn run_single_cuda_step(device: &CudaDevice, rollout_steps: usize) -> Option<MemorySnapshot> {
    type Backend = Autodiff<Cuda<f32>>;
    let model = SudokuSaccadeModel::new(
        &SudokuModelConfig {
            n_layer: 1,
            n_embd: 32,
            n_head: 1,
            mlp_internal_dim_multiplier: 2,
            summary_tokens: 1,
            policy_heads: 1,
            policy_head: SudokuPolicyHead::Cache,
            policy_mlp_hidden_mult: 2,
            rotary_embedding: Default::default(),
            grid_positional: SudokuGridPositional::Additive,
            grid_rope_theta: 65_536.0,
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
            cache_mhc: SudokuCacheMhcConfig::default(),
            cache_update: SudokuCacheUpdateConfig::default(),
        },
        device,
    );

    let training = SudokuTrainingHyperparameters {
        batch_size: 2,
        epochs: None,
        max_iters: 1,
        log_frequency: 1,
        rollout: SudokuRolloutConfig {
            steps: rollout_steps,
            min_steps: 0,
            max_steps: 0,
            max_steps_warmup_iters: 0,
            max_steps_warmup_cap: 0,
            backprop_steps: Some(8),
            ..SudokuRolloutConfig::default()
        },
        halt: SudokuHaltConfig {
            weight: 0.1,
            exploration_prob: 0.0,
            min_steps: 1,
        },
        policy: SudokuPolicyConfig {
            noise: 0.0,
            epsilon: 0.0,
            epsilon_final: 0.0,
            epsilon_anneal_steps: 0,
            teacher_forcing_prob: 0.0,
            teacher_forcing_final: 0.0,
            teacher_forcing_anneal_steps: 0,
            temperature: 1.0,
            temperature_final: 1.0,
            temperature_anneal_steps: 0,
            entropy_weight: 0.0,
            entropy_weight_final: 0.0,
            entropy_anneal_steps: 0,
            entropy_adaptive: false,
            entropy_target_scale: 1.0,
            entropy_alpha: 0.0,
            entropy_alpha_lr: 0.0,
            visit_penalty: 0.0,
            revisit_penalty: 0.0,
            recon_weight: 0.0,
            ..SudokuPolicyConfig::default()
        },
        revisit: SudokuRevisitConfig {
            min_filled_frac: 0.0,
            min_filled_final: 0.0,
            min_filled_anneal_steps: 0,
        },
        reward: SudokuRewardConfig {
            unknown_power: 0.0,
            ..SudokuRewardConfig::default()
        },
        recon: SudokuReconConfig {
            loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            loss_interval_steps: 1,
            global_loss_samples: 8,
            global_loss_weight: 0.2,
            ..SudokuReconConfig::default()
        },
        validation: SudokuValidationConfig::default(),
        gdpo: GdpoConfig {
            enabled: false,
            group_size: 1,
            ..GdpoConfig::default()
        },
    };

    let trainer = SudokuTrainer::new(model, training, 1);
    let puzzles = Tensor::<Backend, 2, Int>::zeros([2, GRID_LEN], device);
    let solution_values = vec![1i64; 2 * GRID_LEN];
    let solutions = Tensor::<Backend, 2, Int>::from_data(
        TensorData::new(solution_values, [2, GRID_LEN]),
        device,
    );
    let batch = SudokuBatch::new(puzzles, solutions);

    let output = trainer.step(batch);
    drop(output);
    let _ = Cuda::<f32>::sync(device);

    cuda_snapshot(device)
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
#[test]
fn cuda_vram_constant_across_rollout_steps() {
    let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::default)).ok();
    let Some(device) = device else {
        eprintln!("cuda device unavailable; skipping vram scaling test");
        return;
    };

    let small = run_single_cuda_step(&device, 16);
    let large = run_single_cuda_step(&device, 128);
    let (Some(small), Some(large)) = (small, large) else {
        eprintln!("cuda memory snapshot unavailable; skipping vram scaling test");
        return;
    };

    let growth_reserved = large.reserved.saturating_sub(small.reserved);
    let growth_in_use = large.in_use.saturating_sub(small.in_use);
    let max_growth = 64 * 1024 * 1024;
    let enforce_reserved = std::env::var("SUDOKU_ASSERT_RESERVED_VRAM")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true);

    if enforce_reserved {
        assert!(
            growth_reserved <= max_growth,
            "reserved VRAM grew by {} bytes (> {}); small={:?} large={:?}",
            growth_reserved,
            max_growth,
            small,
            large
        );
    } else if growth_reserved > max_growth {
        eprintln!(
            "reserved VRAM grew by {} bytes (> {}); set SUDOKU_ASSERT_RESERVED_VRAM=1 to enforce. small={:?} large={:?}",
            growth_reserved, max_growth, small, large
        );
    }
    assert!(
        growth_in_use <= max_growth,
        "in-use VRAM grew by {} bytes (> {}); small={:?} large={:?}",
        growth_in_use,
        max_growth,
        small,
        large
    );
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
#[test]
fn cuda_sudoku_training_tiny_like_smoke() {
    let dir = tempdir().expect("tempdir");
    write_dataset(dir.path());

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("sudoku")
        .join("tiny.toml");
    let mut config = load_training_config(&[config_path]).expect("load config/sudoku/tiny.toml");
    let train_split_ratio = config.dataset.train_split_ratio;
    let augment = config.dataset.augment;
    let augment_prob = config.dataset.augment_prob;
    config.dataset = SudokuDatasetConfig {
        cache_dir: dir.path().to_path_buf(),
        train_split_ratio,
        augment,
        augment_prob,
        source: SudokuDatasetSourceConfig::Local(SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: Vec::new(),
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            train_max_records: None,
            validation_max_records: None,
            max_records: Some(64),
        }),
    };
    config.training.max_iters = 10;
    config.training.log_frequency = 5;

    loss_trace_reset();
    solve_rate_trace_reset();
    let result = train_backend_for_test::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {});
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let losses = loss_trace_take();
    assert!(
        !losses.is_empty(),
        "expected loss trace samples from cuda integration run"
    );
    assert!(
        losses.iter().all(|value| value.is_finite()),
        "expected finite loss values from cuda integration run"
    );

    let solve_rates = solve_rate_trace_take();
    assert!(
        !solve_rates.is_empty(),
        "expected solve rate trace samples from cuda integration run"
    );
    assert!(
        solve_rates
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0 && *value <= 1.0),
        "expected solve rates in [0, 1] from cuda integration run"
    );
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
#[test]
fn cuda_sudoku_training_trm_tiny_smoke() {
    let dir = tempdir().expect("tempdir");
    write_dataset(dir.path());

    let config_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("sudoku")
        .join("trm")
        .join("tiny.toml");
    let mut config =
        load_training_config(&[config_path]).expect("load config/sudoku/trm/tiny.toml");
    let train_split_ratio = config.dataset.train_split_ratio;
    let augment = config.dataset.augment;
    let augment_prob = config.dataset.augment_prob;
    config.dataset = SudokuDatasetConfig {
        cache_dir: dir.path().to_path_buf(),
        train_split_ratio,
        augment,
        augment_prob,
        source: SudokuDatasetSourceConfig::Local(SudokuLocalConfig {
            root: dir.path().to_path_buf(),
            format: SudokuRecordFormat::Jsonl,
            train_files: vec!["train.jsonl".to_string()],
            validation_files: Vec::new(),
            puzzle_field: "puzzle".to_string(),
            solution_field: "solution".to_string(),
            train_max_records: None,
            validation_max_records: None,
            max_records: Some(64),
        }),
    };
    config.training.max_iters = 1;
    config.training.log_frequency = 1;
    config.artifacts.max_samples = 0;

    assert_eq!(config.training.rollout.steps, 64);
    assert_eq!(config.training.rollout.backprop_steps, Some(32));

    loss_trace_reset();
    solve_rate_trace_reset();
    let result = train_backend_for_test::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {});
    if let Err(err) = result {
        panic!("training failed: {err}");
    }

    let losses = loss_trace_take();
    assert!(
        !losses.is_empty(),
        "expected loss trace samples from cuda TRM integration run"
    );
    assert!(
        losses.iter().all(|value| value.is_finite()),
        "expected finite loss values from cuda TRM integration run"
    );
}
