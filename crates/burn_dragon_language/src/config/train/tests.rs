use std::fs;
use std::path::{Path, PathBuf};

use burn_dragon_core::ManifoldHyperConnectionCoefficientPolicy;
use burn_dragon_train::WgpuGenerationExecutor;
use tempfile::tempdir;

use super::super::ContextStrategyConfig;
use super::*;

fn write_config(dir: &Path, name: &str, contents: &str) -> PathBuf {
    let path = dir.join(name);
    let trimmed_lines: Vec<&str> = contents.lines().map(|line| line.trim_start()).collect();
    let mut formatted = trimmed_lines.join("\n");
    if formatted.starts_with('\n') {
        formatted = formatted.trim_start_matches('\n').to_string();
    }
    fs::write(&path, formatted).expect("write config");
    path
}

#[test]
fn load_merges_in_order() {
    let dir = tempdir().expect("tempdir");

    let base_contents = [
        "[dataset]",
        "cache_dir = \"data\"",
        "train_split_ratio = 0.8",
        "type = \"shakespeare\"",
        "",
        "[training]",
        "block_size = 256",
        "batch_size = 16",
        "max_iters = 1000",
        "log_frequency = 50",
        "",
        "[optimizer]",
        "learning_rate = 0.001",
        "weight_decay = 0.05",
        "",
        "[optimizer.lr_schedule]",
        "type = \"cosine\"",
        "min_lr = 0.00005",
        "num_iters = 100",
        "",
        "[generation]",
        "prompt = \"Base prompt\"",
        "max_tokens = 64",
        "temperature = 0.9",
        "top_k = 4",
        "",
        "[model]",
        "n_layer = 6",
        "n_embd = 256",
        "n_head = 4",
        "mlp_internal_dim_multiplier = 4",
        "dropout = 0.1",
        "fused_kernels = false",
        "rollout_fast_steps_per_slow_step = 2",
        "rotary_embedding = \"alibi\"",
    ]
    .join("\n");
    let base = write_config(dir.path(), "base.toml", &base_contents);

    let override_contents = [
        "[training]",
        "max_iters = 2000",
        "",
        "[optimizer]",
        "learning_rate = 0.0005",
        "",
        "[optimizer.lr_schedule]",
        "type = \"linear\"",
        "final_lr = 0.0002",
        "num_iters = 50",
        "",
        "[model]",
        "n_embd = 320",
        "fused_kernels = true",
        "block_size = 256",
        "rollout_fast_steps_per_slow_step = 8",
    ]
    .join("\n");
    let override_cfg = write_config(dir.path(), "override.toml", &override_contents);

    let config = load_training_config(&[base, override_cfg]).expect("load config");

    assert_eq!(
        config.training,
        TrainingHyperparameters {
            block_size: 256,
            batch_size: 16,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: None,
            max_iters: 2000,
            log_frequency: 50,
            fast_train: false,
            context_strategy: ContextStrategyConfig::Infinite,
            gdpo: None,
        }
    );
    assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
    assert!((config.optimizer.weight_decay - 0.05).abs() < f32::EPSILON);
    assert_eq!(
        config.optimizer.lr_schedule,
        Some(burn_dragon_train::LearningRateScheduleConfig::Linear {
            initial_lr: None,
            final_lr: 0.0002,
            num_iters: Some(50),
        })
    );
    assert_eq!(config.dataset.tokenizer, TokenizerConfig::default());
    assert!((config.dataset.train_split_ratio - 0.8).abs() < f32::EPSILON);
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::Shakespeare { url: None }
    );
    assert_eq!(config.generation.max_tokens, Some(64));
    assert_eq!(
        config.training.context_strategy,
        ContextStrategyConfig::Infinite
    );
    assert_eq!(
        config.generation.context_strategy,
        ContextStrategyConfig::Infinite
    );
    assert_eq!(config.model.n_layer, Some(6));
    assert_eq!(config.model.n_embd, Some(320));
    assert_eq!(config.model.n_head, Some(4));
    assert_eq!(config.model.mlp_internal_dim_multiplier, Some(4));
    assert_eq!(config.model.dropout, Some(0.1));
    assert_eq!(config.model.fused_kernels, Some(true));
    assert_eq!(config.model.block_size, Some(256));
    assert_eq!(config.model.rollout_fast_steps_per_slow_step, Some(8));
    assert_eq!(
        config.model.rotary_embedding,
        Some(burn_dragon_core::RotaryEmbedding::Alibi)
    );
}

#[test]
fn validate_rejects_invalid_rollout_fast_steps() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        gradient_accumulation_steps = 3
        target_effective_batch_size = 48
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model]
        rollout_fast_steps_per_slow_step = 3
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("invalid rollout fast steps should fail validation");
    assert!(
        err.to_string()
            .contains("model.rollout_fast_steps_per_slow_step"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn schedule_constant_round_trips() {
    let text = r#"
        learning_rate = 0.002
        weight_decay = 0.1

        [lr_schedule]
        type = "constant"
    "#;
    let optimizer: burn_dragon_train::OptimizerConfig =
        toml::from_str(text).expect("parse optimizer config");
    assert_eq!(
        optimizer.lr_schedule,
        Some(burn_dragon_train::LearningRateScheduleConfig::Constant { initial_lr: None })
    );
}

#[test]
fn huggingface_dataset_config_parses() {
    let text = r#"
        cache_dir = "data"
        train_split_ratio = 0.75
        type = "hugging_face"
        repo_id = "zwhe99/DeepMath-103K"
        revision = "main"
        format = "parquet"
        train_files = [
            "data/train-00000-of-00010.parquet",
            "data/train-00001-of-00010.parquet",
        ]
        validation_files = []
        text_fields = ["question", "final_answer"]
        field_separator = "\n\n"
        template = "{question}\n{final_answer}"
        max_records = 1000
    "#;
    let dataset: DatasetConfig = toml::from_str(text).expect("parse dataset config");
    assert_eq!(dataset.train_split_ratio, 0.75);
    match &dataset.source {
        DatasetSourceConfig::HuggingFace(hf) => {
            assert_eq!(hf.repo_id, "zwhe99/DeepMath-103K");
            assert_eq!(hf.revision.as_deref(), Some("main"));
            assert_eq!(hf.format, HuggingFaceRecordFormat::Parquet);
            assert_eq!(
                hf.train_files,
                vec![
                    "data/train-00000-of-00010.parquet".to_string(),
                    "data/train-00001-of-00010.parquet".to_string()
                ]
            );
            assert!(hf.validation_files.is_empty());
            assert_eq!(hf.text_fields, vec!["question", "final_answer"]);
            assert_eq!(hf.field_separator, "\n\n");
            assert_eq!(hf.template.as_deref(), Some("{question}\n{final_answer}"));
            assert_eq!(hf.max_records, Some(1000));
        }
        other => panic!("unexpected dataset source: {other:?}"),
    }
}

#[test]
fn wgpu_training_and_inference_core_switches_parse() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        gradient_accumulation_steps = 3
        target_effective_batch_size = 48
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [wgpu.training]
        fused_core_recurrent = true
        fused_core_rollout = true

        [wgpu.training.startup_autotune]
        enabled = true
        target_device_memory_mb = 4096
        min_batch_size = 4
        max_batch_size = 64
        probe_steps = 2
        binary_search = false

        [wgpu.inference]
        fused_core_recurrent = false
        fused_core_rollout = false
        generation_executor = "rollout_chunked"
        generation_chunk_tokens = 16
        generation_device_buffer_tokens = 96
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    assert_eq!(config.wgpu.training.fused_core_recurrent, Some(true));
    assert_eq!(config.wgpu.training.fused_core_rollout, Some(true));
    assert_eq!(config.training.gradient_accumulation_steps, 3);
    assert_eq!(config.training.target_effective_batch_size, Some(48));
    assert!(config.wgpu.training.startup_autotune.enabled);
    assert_eq!(config.wgpu.training.startup_autotune.target_device_memory_mb, 4096);
    assert_eq!(config.wgpu.training.startup_autotune.min_batch_size, 4);
    assert_eq!(config.wgpu.training.startup_autotune.max_batch_size, Some(64));
    assert_eq!(config.wgpu.training.startup_autotune.probe_steps, 2);
    assert!(!config.wgpu.training.startup_autotune.binary_search);
    assert_eq!(config.wgpu.inference.fused_core_recurrent, Some(false));
    assert_eq!(config.wgpu.inference.fused_core_rollout, Some(false));
    assert_eq!(
        config.wgpu.inference.generation_executor,
        WgpuGenerationExecutor::RolloutChunked
    );
    assert_eq!(config.wgpu.inference.generation_chunk_tokens, 16);
    assert_eq!(config.wgpu.inference.generation_device_buffer_tokens, 96);
}

#[test]
fn validate_rejects_invalid_wgpu_startup_autotune_config() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        gradient_accumulation_steps = 0
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [wgpu.training.startup_autotune]
        enabled = true
        target_device_memory_mb = 0
        min_batch_size = 8
        max_batch_size = 4
        probe_steps = 0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let error = config.validate().expect_err("invalid autotune config should fail");
    assert!(
        error
            .to_string()
            .contains("training.gradient_accumulation_steps must be > 0")
    );
}

#[test]
fn mhc_override_parses_and_validates_for_language_bdh() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.mhc]
        enabled = true
        num_streams = 1
        num_views = 4
        coefficient_policy = "static_sinkhorn"
        mhc_iters = 8
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("language mHC config should validate");
    let mhc = config.model.mhc.expect("mHC override");
    assert!(mhc.enabled);
    assert_eq!(mhc.num_streams, 1);
    assert_eq!(mhc.num_views, 4);
    assert_eq!(
        mhc.coefficient_policy,
        ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn
    );
}

#[test]
fn y_neuron_recurrence_override_parses_and_validates_for_language_bdh() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.y_neuron_recurrence]
        enabled = true
        carry_in_scale = 0.125
        last_layers = 1
        chunk_tokens = 4
        state_decay = 0.5
        state_update_scale = 1.5
        state_rms_cap = 0.75
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("language y_neuron recurrence should validate");
    let recurrence = config
        .model
        .y_neuron_recurrence
        .expect("y_neuron recurrence override");
    assert!(recurrence.enabled);
    assert_eq!(recurrence.carry_in_scale, 0.125);
    assert_eq!(recurrence.last_layers, Some(1));
    assert_eq!(recurrence.chunk_tokens, 4);
    assert_eq!(recurrence.state_decay, 0.5);
    assert_eq!(recurrence.state_update_scale, 1.5);
    assert_eq!(recurrence.state_rms_cap, Some(0.75));
}

#[test]
fn normalization_override_parses_and_validates_for_language_bdh() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.normalization]
        kind = "rms_norm"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("language normalization config should validate");
    let normalization = config.model.normalization.expect("normalization override");
    assert_eq!(normalization.kind, burn_dragon_core::DragonNormKind::RmsNorm);
}

#[test]
fn y_sparse_recurrence_alias_parses_into_y_neuron_recurrence() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.y_sparse_recurrence]
        enabled = true
        carry_in_scale = 0.2
        last_layers = 2
        chunk_tokens = 8
        state_decay = 0.25
        state_update_scale = 2.0
        state_rms_cap = 0.5
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("legacy y_sparse alias should validate");
    let recurrence = config
        .model
        .y_neuron_recurrence
        .expect("aliased y_neuron recurrence override");
    assert!(recurrence.enabled);
    assert_eq!(recurrence.carry_in_scale, 0.2);
    assert_eq!(recurrence.last_layers, Some(2));
    assert_eq!(recurrence.chunk_tokens, 8);
    assert_eq!(recurrence.state_decay, 0.25);
    assert_eq!(recurrence.state_update_scale, 2.0);
    assert_eq!(recurrence.state_rms_cap, Some(0.5));
}

#[test]
fn validate_rejects_zero_last_layers_for_y_neuron_recurrence() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.y_neuron_recurrence]
        enabled = true
        last_layers = 0
        chunk_tokens = 4
        state_decay = 0.5
        state_update_scale = 1.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config.validate().expect_err("zero last_layers should be rejected");
    assert!(
        err.to_string().contains("last_layers"),
        "expected last_layers validation error, got {err}"
    );
}

#[test]
fn validate_rejects_language_mhc_multi_streams_for_now() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model.mhc]
        enabled = true
        num_streams = 2
        num_views = 2
        mhc_iters = 8
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("language multi-stream mHC should fail validation for now");
    assert!(
        err.to_string().contains("model.mhc.num_streams"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn wgpu_inference_generation_defaults_parse() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    assert_eq!(
        config.wgpu.inference.generation_executor,
        WgpuGenerationExecutor::Baseline
    );
    assert_eq!(config.wgpu.inference.generation_chunk_tokens, 8);
    assert_eq!(config.wgpu.inference.generation_device_buffer_tokens, 64);
}
