use std::fs;
use std::path::{Path, PathBuf};

use burn_dragon_core::{
    LatentFanoutScheduleConfig, ManifoldHyperConnectionCoefficientPolicy, ResidualConnectorKind,
};
use burn_dragon_train::{
    ParallelConfig, ParallelismKind, TensorParallelPartitionKind, WgpuGenerationExecutor,
};
use tempfile::tempdir;

use super::super::{
    ContextStrategyConfig, GenerationOutputFormat, GenerationTokenizerSourceConfig,
};
use super::*;
use crate::tokenizer::{ByteTokenizerConfig, PretokenizedTokenizerConfig, TokenizerKind};

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
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 16,
            seed: 1337,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: None,
            max_iters: 2000,
            checkpoint_interval_iters: 2000,
            log_frequency: 50,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            module_lr_scales: Vec::new(),
            context_strategy: ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        }
    );
    assert_eq!(config.parallel, ParallelConfig::default());
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
    assert_eq!(config.model.latent_total, None);
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
fn validate_accepts_explicit_latent_total_override() {
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
        n_embd = 256
        latent_total = 32768
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(config.model.latent_total, Some(32768));
}

#[test]
fn validate_accepts_sequence_kernel_override() {
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
        sequence_kernel = "rwkv8_state_space"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(
        config.model.sequence_kernel,
        Some(burn_dragon_core::SequenceKernelConfig::reference(
            burn_dragon_core::SequenceMemorySystem::Rwkv8StateSpace,
        ))
    );
}

#[test]
fn load_parses_module_lr_scale_schedule_block() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [[training.module_lr_scales]]
        target = "mamba"
        scale = 0.5

        [training.module_lr_scales.schedule]
        final_scale = 1.0
        start_fraction = 0.25
        end_fraction = 0.75

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(config.training.module_lr_scales.len(), 1);
    let entry = &config.training.module_lr_scales[0];
    assert_eq!(
        entry.target,
        burn_dragon_core::LanguageModuleLrScaleTarget::Mamba
    );
    assert!((entry.scale - 0.5).abs() < f32::EPSILON);
    let schedule = entry.schedule.as_ref().expect("scheduled module lr scale");
    assert!((schedule.final_scale - 1.0).abs() < f32::EPSILON);
    assert!((schedule.start_fraction - 0.25).abs() < f32::EPSILON);
    assert!((schedule.end_fraction - 0.75).abs() < f32::EPSILON);
}

#[test]
fn validate_accepts_training_sequence_kernel_override() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1
        sequence_kernel_override = { memory_system = "linear_attention", executor = "dense_score_short_context" }

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model]
        sequence_kernel = "linear_attention"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(
        config.training.sequence_kernel_override,
        Some(burn_dragon_core::SequenceKernelConfig::dense_score_short_context())
    );
    assert_eq!(
        config.model.sequence_kernel,
        Some(burn_dragon_core::SequenceKernelConfig::reference(
            burn_dragon_core::SequenceMemorySystem::LinearAttention,
        ))
    );
}

#[test]
fn validate_accepts_mamba2_state_space_duality_request() {
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
        sequence_kernel = "mamba2_state_space_duality"

        [model.mamba]
        headdim = 64
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("mamba2_state_space_duality should validate");
}

#[test]
fn load_training_config_rejects_removed_fast_train_key() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1
        fast_train = true

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"

        [model]
        sequence_kernel = "linear_attention"
    "#;
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("removed-fast-train.toml");
    std::fs::write(&path, text).expect("write config");
    let err = load_training_config(&[path]).expect_err("fast_train should be rejected");
    assert!(
        err.to_string()
            .contains("training.fast_train has been removed"),
        "unexpected load error: {err}"
    );
}

#[test]
fn generation_tokenizer_overrides_parse_with_distinct_source_and_tokenizer_tags() {
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
        prompt = "1 2 3"
        output_format = "decoded_text"

        [generation.prompt_tokenizer]
        source = "config"
        type = "pretokenized"
        vocab_size = 50257
        bos_id = 1

        [generation.decode_tokenizer]
        source = "config"
        cache_dir = "data/tokenizers"
        type = "byte"
        add_special_tokens = false
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");

    assert_eq!(
        config.generation.output_format,
        GenerationOutputFormat::DecodedText
    );
    assert_eq!(
        config.generation.prompt_tokenizer,
        GenerationTokenizerSourceConfig::Config {
            cache_dir: None,
            tokenizer: TokenizerConfig {
                vocab_path: None,
                kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                    vocab_size: 50257,
                    bos_id: Some(1),
                    eos_id: None,
                    pad_id: None,
                    unk_id: None,
                }),
            },
        }
    );
    assert_eq!(
        config.generation.decode_tokenizer,
        GenerationTokenizerSourceConfig::Config {
            cache_dir: Some(PathBuf::from("data/tokenizers")),
            tokenizer: TokenizerConfig {
                vocab_path: None,
                kind: TokenizerKind::Byte(ByteTokenizerConfig {
                    add_special_tokens: false,
                }),
            },
        }
    );
}

#[test]
fn universality_manifest_dataset_config_parses_with_pretokenized_tokenizer() {
    let text = r#"
        [dataset]
        cache_dir = "data/universality/nca"
        type = "universality_manifest"
        manifest = "data/universality/nca/manifest.json"

        [dataset.tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256

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
    config.validate().expect("valid config");
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::UniversalityManifest {
            manifest: "data/universality/nca/manifest.json".into()
        }
    );
}

#[test]
fn universality_nca_dataset_config_parses_with_pretokenized_tokenizer() {
    let text = r#"
        [dataset]
        cache_dir = "data/universality/runtime"
        type = "universality_nca"
        config = "config/universality/nca_paper_aligned_smoke.toml"

        [dataset.tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256

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
    config.validate().expect("valid config");
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::UniversalityNca {
            config: "config/universality/nca_paper_aligned_smoke.toml".into()
        }
    );
}

#[test]
fn local_text_dataset_config_parses_with_byte_tokenizer() {
    let text = r#"
        [dataset]
        cache_dir = "data/local_text"
        type = "local_text"
        path = "data/local_text/train.txt"

        [dataset.tokenizer]
        type = "byte"
        add_special_tokens = true

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "track|n=3|init=ABC____|s=AB,__,__,__,__|q=A|a="
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::LocalText {
            path: "data/local_text/train.txt".into()
        }
    );
    assert!(matches!(
        config.dataset.tokenizer.kind,
        TokenizerKind::Byte(_)
    ));
}

#[test]
fn validate_accepts_init_checkpoint_path_with_optional_epoch() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1
        launch_mode = "init_from_checkpoint"
        init_checkpoint_path = "runs/example/checkpoint"
        init_checkpoint_epoch = 3

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(
        config.training.init_checkpoint_path,
        Some(PathBuf::from("runs/example/checkpoint"))
    );
    assert_eq!(config.training.init_checkpoint_epoch, Some(3));
}

#[test]
fn parse_defaults_parallel_and_seed_for_legacy_configs() {
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
    assert_eq!(config.training.seed, 1337);
    assert_eq!(config.parallel, ParallelConfig::default());
    assert_eq!(config.training.resume_run_dir, None);
    assert_eq!(config.training.resume_checkpoint_epoch, None);
    config
        .validate()
        .expect("legacy config should still validate");
}

#[test]
fn validate_accepts_resume_run_dir_with_optional_epoch() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        log_frequency = 1
        launch_mode = "resume_exact_run"
        resume_run_dir = "runs/example"
        resume_checkpoint_epoch = 3

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid resume config");
    assert_eq!(
        config.training.resume_run_dir,
        Some(PathBuf::from("runs/example"))
    );
    assert_eq!(config.training.resume_checkpoint_epoch, Some(3));
}

#[test]
fn validate_rejects_zero_checkpoint_interval() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        max_iters = 4
        checkpoint_interval_iters = 0
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("zero checkpoint interval should fail");
    assert!(
        err.to_string()
            .contains("training.checkpoint_interval_iters must be > 0")
    );
}

#[test]
fn validate_accepts_explicit_parallel_config() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "shakespeare"

        [training]
        block_size = 32
        batch_size = 2
        seed = 4242
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [parallel]
        mode = "tensor_parallel_neuron"
        world_size = 4

        [parallel.data]
        size = 1

        [parallel.tensor]
        size = 4
        partition = "head_aligned"

        [generation]
        prompt = "abc"

        [model]
        n_embd = 256
        n_head = 4
        latent_total = 32768
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    assert_eq!(config.training.seed, 4242);
    assert_eq!(config.parallel.mode, ParallelismKind::TensorParallelNeuron);
    assert_eq!(
        config.parallel.tensor.partition,
        TensorParallelPartitionKind::HeadAligned
    );
    config
        .validate()
        .expect("explicit parallel config should validate");
}

#[test]
fn validate_accepts_pipeline_cache_parallel_config() {
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

        [parallel]
        mode = "ddp"
        world_size = 4

        [parallel.data]
        size = 2

        [parallel.pipeline]
        enabled = true
        stage_count = 2
        virtual_stages_per_rank = 1
        schedule = "interleaved_1f1b"
        microbatches = 2
        communication = "block_residual_cache"

        [parallel.pipeline.cache]
        enabled = true
        policy = "resident_block_summaries"
        reuse_across_backward = true
        max_inflight_microbatches = 2
        eviction = "step_boundary"
        transport_dtype = "bf16"

        [generation]
        prompt = "abc"

        [model]
        residual_connector = "block_attention_residual"

        [model.block_attention_residual]
        enabled = true
        layers_per_block = 2
        num_heads = 2
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    assert!(config.parallel.pipeline.enabled);
    assert!(config.parallel.pipeline.cache.enabled);
    config
        .validate()
        .expect("pipeline cache config should validate");
}

#[test]
fn validate_rejects_pipeline_cache_without_pipeline() {
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

        [parallel.pipeline.cache]
        enabled = true
        policy = "resident_block_summaries"
        max_inflight_microbatches = 2

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("pipeline cache without pipeline should fail validation");
    assert!(
        err.to_string()
            .contains("parallel.pipeline.cache.enabled requires parallel.pipeline.enabled"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_parallel_world_size_mismatch() {
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

        [parallel]
        mode = "ddp"
        world_size = 4

        [parallel.data]
        size = 2

        [parallel.tensor]
        size = 1

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("parallel world size mismatch should fail validation");
    assert!(
        err.to_string()
            .contains("parallel.data.size * parallel.tensor.size * pipeline_stage_multiplier"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_accepts_single_process_pipeline_simulation_config() {
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

        [parallel]
        mode = "single"
        world_size = 1

        [parallel.data]
        size = 1

        [parallel.tensor]
        size = 1

        [parallel.pipeline]
        enabled = true
        stage_count = 2
        virtual_stages_per_rank = 1
        schedule = "interleaved_1f1b"
        microbatches = 2

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("single-process pipeline simulation config should validate");
}

#[test]
fn validate_rejects_pipeline_virtual_stages_exceeding_stage_count() {
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

        [parallel]
        mode = "single"
        world_size = 1

        [parallel.pipeline]
        enabled = true
        stage_count = 2
        virtual_stages_per_rank = 3
        schedule = "interleaved_1f1b"
        microbatches = 2

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("virtual stages larger than stage count should fail validation");
    assert!(
        err.to_string().contains(
            "parallel.pipeline.virtual_stages_per_rank must be <= parallel.pipeline.stage_count"
        ),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_pipeline_microbatches_exceeding_batch_size() {
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

        [parallel]
        mode = "single"
        world_size = 1

        [parallel.pipeline]
        enabled = true
        stage_count = 2
        virtual_stages_per_rank = 1
        schedule = "interleaved_1f1b"
        microbatches = 3

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("microbatches above batch size should fail validation");
    assert!(
        err.to_string()
            .contains("parallel.pipeline.microbatches must be <= training.batch_size"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_block_residual_cache_without_block_connector() {
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

        [parallel]
        mode = "ddp"
        world_size = 4

        [parallel.data]
        size = 2

        [parallel.pipeline]
        enabled = true
        stage_count = 2
        virtual_stages_per_rank = 1
        schedule = "interleaved_1f1b"
        microbatches = 2
        communication = "block_residual_cache"

        [generation]
        prompt = "abc"

        [model]
        residual_connector = "attention_residual"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("block residual cache should require block connector");
    assert!(
        err.to_string().contains(
            "parallel.pipeline.communication = \"block_residual_cache\" requires model.residual_connector = \"block_attention_residual\""
        ),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_accepts_complete_collective_global_config() {
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

        [parallel]
        mode = "ddp"
        world_size = 2

        [parallel.data]
        size = 2
        collective_num_nodes = 2
        collective_global_address = "127.0.0.1:32000"
        collective_node_address = "127.0.0.1:32001"
        collective_data_service_port = 32001

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
}

#[test]
fn validate_rejects_partial_collective_global_config() {
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

        [parallel]
        mode = "ddp"
        world_size = 2

        [parallel.data]
        size = 2
        collective_num_nodes = 2
        collective_global_address = "127.0.0.1:32000"

        [generation]
        prompt = "abc"
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("partial collective config should fail validation");
    assert!(
        err.to_string().contains("collective global settings"),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_accepts_latent_fanout_schedule() {
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
        n_layer = 8
        n_embd = 256
        n_head = 4
        latent_total = 32768

        [model.latent_fanout_schedule]
        type = "late_layer"
        base_latent_total = 8192
        last_layers = 4
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config.validate().expect("valid config");
    assert_eq!(
        config.model.latent_fanout_schedule,
        Some(LatentFanoutScheduleConfig::LateLayer {
            base_latent_total: 8192,
            last_layers: 4,
        })
    );
}

#[test]
fn validate_rejects_invalid_latent_fanout_schedule() {
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
        n_layer = 8
        n_embd = 256
        n_head = 4
        latent_total = 32768

        [model.latent_fanout_schedule]
        type = "late_layer"
        base_latent_total = 9000
        last_layers = 0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("invalid latent schedule should fail validation");
    assert!(
        err.to_string().contains("model.latent_fanout_schedule"),
        "unexpected error: {err:#}"
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
            assert_eq!(hf.sequence_field, None);
            assert_eq!(hf.field_separator, "\n\n");
            assert_eq!(hf.template.as_deref(), Some("{question}\n{final_answer}"));
            assert_eq!(hf.max_records, Some(1000));
            assert!(!hf.auto_discover_train_files);
        }
        other => panic!("unexpected dataset source: {other:?}"),
    }
}

#[test]
fn validation_override_huggingface_config_parses() {
    let text = r#"
        cache_dir = "data"
        type = "nemotron_climb_mix"
        max_records = 1024

        [validation]
        cache_dir = "data/validation"
        train_split_ratio = 0.8
        type = "hugging_face"
        repo_id = "example/openwebtext-gpt2-ids"
        format = "jsonl"
        train_files = []
        validation_files = ["validation.jsonl"]
        sequence_field = "tokens"

        [tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256
    "#;
    let dataset: DatasetConfig = toml::from_str(text).expect("parse dataset config");
    let validation = dataset.validation.expect("validation override");
    assert_eq!(
        validation.cache_dir.as_deref(),
        Some(Path::new("data/validation"))
    );
    assert_eq!(validation.train_split_ratio, Some(0.8));
    match validation.source {
        DatasetSourceConfig::HuggingFace(hf) => {
            assert_eq!(hf.repo_id, "example/openwebtext-gpt2-ids");
            assert!(hf.train_files.is_empty());
            assert_eq!(hf.validation_files, vec!["validation.jsonl"]);
            assert_eq!(hf.sequence_field.as_deref(), Some("tokens"));
        }
        other => panic!("unexpected validation dataset source: {other:?}"),
    }
}

#[test]
fn nemotron_climbmix_dataset_config_parses_with_pretokenized_tokenizer() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "nemotron_climb_mix"
        max_records = 1024

        [dataset.tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256

        [training]
        block_size = 128
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "464 329 262"
    "#;

    let config: TrainingConfig = toml::from_str(text).expect("parse nemotron config");
    config.validate().expect("nemotron config should validate");
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::NemotronClimbMix {
            revision: None,
            max_records: Some(1024),
        }
    );
    match config.dataset.tokenizer.kind {
        crate::tokenizer::TokenizerKind::Pretokenized(config) => {
            assert_eq!(config.vocab_size, 50_257);
            assert_eq!(config.eos_id, Some(50_256));
        }
        other => panic!("expected pretokenized tokenizer, got {other:?}"),
    }
}

#[test]
fn openwebtext_gpt2_dataset_config_parses_with_pretokenized_tokenizer() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "openwebtext_gpt2"
        max_records = 4096

        [dataset.tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256

        [training]
        block_size = 128
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "464 329 262"
    "#;

    let config: TrainingConfig = toml::from_str(text).expect("parse openwebtext config");
    config
        .validate()
        .expect("openwebtext config should validate");
    assert_eq!(
        config.dataset.source,
        DatasetSourceConfig::OpenWebTextGpt2 {
            revision: None,
            max_records: Some(4096),
        }
    );
}

#[test]
fn validate_accepts_validation_only_huggingface_override() {
    let text = r#"
        [dataset]
        cache_dir = "data"
        type = "nemotron_climb_mix"
        max_records = 1024

        [dataset.validation]
        cache_dir = "data/validation"
        type = "hugging_face"
        repo_id = "example/openwebtext-gpt2-ids"
        format = "jsonl"
        train_files = []
        validation_files = ["validation.jsonl"]
        sequence_field = "tokens"

        [dataset.tokenizer]
        type = "pretokenized"
        vocab_size = 50257
        eos_id = 50256

        [training]
        block_size = 128
        batch_size = 2
        max_iters = 4
        log_frequency = 1

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.0

        [generation]
        prompt = "464 329 262"
    "#;

    let config: TrainingConfig = toml::from_str(text).expect("parse config");
    config
        .validate()
        .expect("validation-only huggingface override should validate");
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
    assert_eq!(
        config
            .wgpu
            .training
            .startup_autotune
            .target_device_memory_mb,
        4096
    );
    assert_eq!(config.wgpu.training.startup_autotune.min_batch_size, 4);
    assert_eq!(
        config.wgpu.training.startup_autotune.max_batch_size,
        Some(64)
    );
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
    let error = config
        .validate()
        .expect_err("invalid autotune config should fail");
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

        [model]
        residual_connector = "mhc"

        [model.mhc]
        enabled = true
        num_streams = 1
        num_views = 4
        last_layers = 1
        coefficient_policy = "static_sinkhorn"
        mhc_iters = 8
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("language mHC config should validate");
    let mhc = config.model.mhc.expect("mHC override");
    assert!(mhc.enabled);
    assert_eq!(mhc.num_streams, 1);
    assert_eq!(mhc.num_views, 4);
    assert_eq!(mhc.last_layers, Some(1));
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
    config
        .validate()
        .expect("language y_neuron recurrence should validate");
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
    config
        .validate()
        .expect("language normalization config should validate");
    let normalization = config.model.normalization.expect("normalization override");
    assert_eq!(
        normalization.kind,
        burn_dragon_core::DragonNormKind::RmsNorm
    );
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
    config
        .validate()
        .expect("legacy y_sparse alias should validate");
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
    let err = config
        .validate()
        .expect_err("zero last_layers should be rejected");
    assert!(
        err.to_string().contains("last_layers"),
        "expected last_layers validation error, got {err}"
    );
}

#[test]
fn language_mhc_multi_streams_validate_for_language_bdh() {
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
        residual_connector = "mhc"

        [model.mhc]
        enabled = true
        num_streams = 2
        num_views = 2
        last_layers = 1
        mhc_iters = 8
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("language multi-stream mHC should now validate");
}

#[test]
fn attention_residual_override_parses_and_validates_for_language_bdh() {
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
        residual_connector = "attention_residual"

        [model.attention_residual]
        enabled = true
        last_layers = 2
        num_heads = 4
        history_window = 3
        dropout = 0.0
        recency_bias = 1.5
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("attention residual config should validate");
    assert_eq!(
        config.model.residual_connector,
        Some(ResidualConnectorKind::AttentionResidual)
    );
    let attention_residual = config
        .model
        .attention_residual
        .expect("attention residual override");
    assert!(attention_residual.enabled);
    assert_eq!(attention_residual.last_layers, Some(2));
    assert_eq!(attention_residual.num_heads, 4);
    assert_eq!(attention_residual.history_window, Some(3));
}

#[test]
fn block_attention_residual_override_parses_and_validates_for_language_bdh() {
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
        residual_connector = "block_attention_residual"

        [model.block_attention_residual]
        enabled = true
        last_layers = 2
        num_heads = 4
        layers_per_block = 2
        block_history_window = 3
        intra_block_history_window = 1
        summary_mode = "learned_projection"
        dropout = 0.0
        recency_bias = 1.5
        cache_block_summaries = true
        two_phase_compute = true
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    config
        .validate()
        .expect("block attention residual config should validate");
    assert_eq!(
        config.model.residual_connector,
        Some(ResidualConnectorKind::BlockAttentionResidual)
    );
    let block_attention_residual = config
        .model
        .block_attention_residual
        .expect("block attention residual override");
    assert!(block_attention_residual.enabled);
    assert_eq!(block_attention_residual.last_layers, Some(2));
    assert_eq!(block_attention_residual.num_heads, 4);
    assert_eq!(block_attention_residual.layers_per_block, 2);
    assert_eq!(block_attention_residual.block_history_window, Some(3));
    assert_eq!(block_attention_residual.intra_block_history_window, Some(1));
}

#[test]
fn validate_rejects_enabled_mhc_without_matching_connector_enum() {
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
        num_views = 1
        mhc_iters = 4
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("enabled mHC should require explicit connector enum selection");
    assert!(
        err.to_string()
            .contains("model.residual_connector = \"mhc\""),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_enabled_attention_residual_without_matching_connector_enum() {
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

        [model.attention_residual]
        enabled = true
        num_heads = 4
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("enabled attention residual should require explicit connector enum selection");
    assert!(
        err.to_string()
            .contains("model.residual_connector = \"attention_residual\""),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_enabled_block_attention_residual_without_matching_connector_enum() {
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

        [model.block_attention_residual]
        enabled = true
        num_heads = 4
        layers_per_block = 2
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config.validate().expect_err(
        "enabled block attention residual should require explicit connector enum selection",
    );
    assert!(
        err.to_string()
            .contains("model.residual_connector = \"block_attention_residual\""),
        "unexpected error: {err:#}"
    );
}

#[test]
fn validate_rejects_zero_last_layers_for_mhc() {
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
        residual_connector = "mhc"

        [model.mhc]
        enabled = true
        num_streams = 2
        num_views = 1
        last_layers = 0
        mhc_iters = 8
        mhc_tau = 0.1
        add_branch_out_to_residual = true
        dropout = 0.0
    "#;
    let config: TrainingConfig = toml::from_str(text).expect("parse training config");
    let err = config
        .validate()
        .expect_err("zero mhc last_layers should be rejected");
    assert!(
        err.to_string().contains("model.mhc.last_layers"),
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
