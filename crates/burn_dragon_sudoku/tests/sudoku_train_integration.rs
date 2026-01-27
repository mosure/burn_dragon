#![cfg(feature = "integration_test")]

use std::fs;
use std::path::{Path, PathBuf};

use burn_autodiff::Autodiff;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_cuda::{Cuda, CudaDevice};
use burn_ndarray::NdArray;
use tempfile::tempdir;

use burn_dragon_sudoku::config::{
    load_training_config, SudokuArtifactConfig, SudokuDatasetConfig, SudokuDatasetSourceConfig,
    SudokuLocalConfig, SudokuLossMask, SudokuModelConfig, SudokuPolicyHead, SudokuReconLoss,
    SudokuRecordFormat, SudokuTrainingConfig, SudokuTrainingHyperparameters,
};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::model::SudokuSaccadeModel;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::dataset::SudokuBatch;
use burn_dragon_sudoku::train::{
    halt_prob_trace_reset, halt_prob_trace_take, loss_trace_reset, loss_trace_take,
    solve_rate_trace_reset, solve_rate_trace_take, train_backend_for_test,
};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_dragon_sudoku::train::SudokuTrainer;
use burn_dragon_sudoku::vocab::GRID_LEN;
use burn_dragon_train::{GdpoConfig, OptimizerConfig, WgpuRuntimeConfig};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn::tensor::{Int, Tensor, TensorData};
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn::tensor::backend::Backend;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use cubecl::Runtime;
#[cfg(all(feature = "cuda", feature = "integration_test"))]
use burn_train::TrainStep;

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
    let puzzle: String = std::iter::repeat('0').take(GRID_LEN).collect();
    let solution: String = std::iter::repeat('1').take(GRID_LEN).collect();
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
            max_records: Some(64),
        }),
    };

    let training = SudokuTrainingHyperparameters {
        batch_size: 4,
        epochs: None,
        max_iters: 40,
        log_frequency: 5,
        rollout_steps: 4,
        rollout_min_steps: 0,
        rollout_max_steps: 0,
        rollout_max_steps_warmup_iters: 0,
        rollout_max_steps_warmup_cap: 0,
        rollout_backprop_steps: None,
        halt_weight: 0.1,
        halt_exploration_prob: 0.0,
        halt_min_steps: 1,
        policy_noise: 0.2,
        policy_epsilon: 0.0,
        policy_epsilon_final: 0.0,
        policy_epsilon_anneal_steps: 0,
        teacher_forcing_prob: 1.0,
        teacher_forcing_final: 1.0,
        teacher_forcing_anneal_steps: 0,
        policy_temperature: 1.0,
        policy_temperature_final: 1.0,
        policy_temperature_anneal_steps: 0,
        policy_entropy_weight: 0.0,
        policy_entropy_weight_final: 0.0,
        policy_entropy_anneal_steps: 0,
        policy_entropy_adaptive: false,
        policy_entropy_target_scale: 1.0,
        policy_entropy_alpha: 0.0,
        policy_entropy_alpha_lr: 0.0,
        policy_visit_penalty: 0.0,\r\n        policy_revisit_penalty: 0.0,
        policy_recon_weight: 0.0,
        revisit_min_filled_frac: 0.0,
        revisit_min_filled_final: 0.0,
        revisit_min_filled_anneal_steps: 0,
        reward_unknown_power: 0.0,
        saccade_step_cells: 1,
        recon_loss: SudokuReconLoss::Softmax,
        loss_mask: SudokuLossMask::All,
        recon_loss_interval_steps: 1,
        global_loss_samples: 8,
        global_loss_weight: 0.2,
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
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
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
            max_records: Some(32),
        }),
    };

    let training = SudokuTrainingHyperparameters {
        batch_size: 2,
        epochs: None,
        max_iters: 60,
        log_frequency: 10,
        rollout_steps: GRID_LEN * 2,
        rollout_min_steps: 0,
        rollout_max_steps: 0,
        rollout_max_steps_warmup_iters: 0,
        rollout_max_steps_warmup_cap: 0,
        rollout_backprop_steps: None,
        halt_weight: 0.2,
        halt_exploration_prob: 0.0,
        halt_min_steps: 1,
        policy_noise: 0.0,
        policy_epsilon: 0.0,
        policy_epsilon_final: 0.0,
        policy_epsilon_anneal_steps: 0,
        teacher_forcing_prob: 1.0,
        teacher_forcing_final: 0.0,
        teacher_forcing_anneal_steps: 10,
        policy_temperature: 1.0,
        policy_temperature_final: 1.0,
        policy_temperature_anneal_steps: 0,
        policy_entropy_weight: 0.0,
        policy_entropy_weight_final: 0.0,
        policy_entropy_anneal_steps: 0,
        policy_entropy_adaptive: false,
        policy_entropy_target_scale: 1.0,
        policy_entropy_alpha: 0.0,
        policy_entropy_alpha_lr: 0.0,
        policy_visit_penalty: 0.0,\r\n        policy_revisit_penalty: 0.0,
        policy_recon_weight: 0.0,
        revisit_min_filled_frac: 1.0,
        revisit_min_filled_final: 1.0,
        revisit_min_filled_anneal_steps: 0,
        reward_unknown_power: 1.0,
        saccade_step_cells: 1,
        recon_loss: SudokuReconLoss::Softmax,
        loss_mask: SudokuLossMask::All,
        recon_loss_interval_steps: 1,
        global_loss_samples: 8,
        global_loss_weight: 0.2,
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
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
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
            dropout: 0.0,
            fused_kernels: false,
            relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
        },
        device,
    );

    let training = SudokuTrainingHyperparameters {
        batch_size: 2,
        epochs: None,
        max_iters: 1,
        log_frequency: 1,
        rollout_steps,
        rollout_min_steps: 0,
        rollout_max_steps: 0,
        rollout_max_steps_warmup_iters: 0,
        rollout_max_steps_warmup_cap: 0,
        rollout_backprop_steps: Some(8),
        halt_weight: 0.1,
        halt_exploration_prob: 0.0,
        halt_min_steps: 1,
        policy_noise: 0.0,
        policy_epsilon: 0.0,
        policy_epsilon_final: 0.0,
        policy_epsilon_anneal_steps: 0,
        teacher_forcing_prob: 0.0,
        teacher_forcing_final: 0.0,
        teacher_forcing_anneal_steps: 0,
        policy_temperature: 1.0,
        policy_temperature_final: 1.0,
        policy_temperature_anneal_steps: 0,
        policy_entropy_weight: 0.0,
        policy_entropy_weight_final: 0.0,
        policy_entropy_anneal_steps: 0,
        policy_entropy_adaptive: false,
        policy_entropy_target_scale: 1.0,
        policy_entropy_alpha: 0.0,
        policy_entropy_alpha_lr: 0.0,
        policy_visit_penalty: 0.0,\r\n        policy_revisit_penalty: 0.0,
        policy_recon_weight: 0.0,
        revisit_min_filled_frac: 0.0,
        revisit_min_filled_final: 0.0,
        revisit_min_filled_anneal_steps: 0,
        reward_unknown_power: 0.0,
        saccade_step_cells: 1,
        recon_loss: SudokuReconLoss::Softmax,
        loss_mask: SudokuLossMask::All,
        recon_loss_interval_steps: 1,
        global_loss_samples: 8,
        global_loss_weight: 0.2,
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
    Cuda::<f32>::sync(device);

    cuda_snapshot(device)
}

#[cfg(all(feature = "cuda", feature = "integration_test"))]
#[test]
fn cuda_vram_constant_across_rollout_steps() {
    let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::default))
        .ok();
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
            growth_reserved,
            max_growth,
            small,
            large
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
    let mut config =
        load_training_config(&[config_path]).expect("load config/sudoku/tiny.toml");
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


