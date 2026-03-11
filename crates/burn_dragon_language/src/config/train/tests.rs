use std::fs;
use std::path::{Path, PathBuf};

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
