use std::collections::BTreeSet;
use std::path::PathBuf;
use std::{fs, path::Path};

use burn_dragon_core::ResidualConnectorKind;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tempfile::tempdir;

use super::train::{TrainingConfig, load_training_config};
use crate::config::train::DatasetSourceConfig;
use crate::stages::load_experiment_bundle_config;
use crate::tokenizer::TokenizerKind;

#[derive(serde::Deserialize)]
struct PromotedBaselineRegistry {
    entries: Vec<PromotedBaselineRegistryEntry>,
}

#[derive(serde::Deserialize)]
struct PromotedBaselineRegistryEntry {
    name: String,
    kind: PromotedBaselineKind,
    family: String,
    path: PathBuf,
}

#[derive(Clone, Copy, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PromotedBaselineKind {
    TrainingConfig,
    BundleConfig,
}

fn config_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("language")
}

fn load_config_from_root(relative_path: &str) -> TrainingConfig {
    let root = config_root();
    let path = root.join(relative_path);
    load_training_config(&[path.clone()])
        .unwrap_or_else(|err| panic!("failed to load language config {path:?}: {err}"))
}

fn roundtrip_config<T>(config: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let serialized = toml::to_string(config).expect("serialize config");
    toml::from_str(&serialized).expect("parse serialized config")
}

#[test]
fn language_configs_parse_serialize_validate() {
    let root = config_root();
    let files = [
        "base.toml",
        "tiny.toml",
        "small.toml",
        "large.toml",
        "baselines/current_best_large/nca_stage_base.toml",
        "baselines/current_best_large/climbmix_stage_base.toml",
        "baselines/current_best_large/climbmix_stage_48h.toml",
        "baselines/shakespeare_deployed_repro.toml",
        "baselines/shakespeare_deployed_dense_score_train.toml",
        "baselines/smoke.toml",
        "baselines/tiny.toml",
        "baselines/small.toml",
        "baselines/current_best.toml",
        "baselines/base.toml",
        "baselines/reasoning.toml",
        "baselines/tool_boundary.toml",
        "tiny_chat_short.toml",
        "tiny_chat_fixedtime_short.toml",
        "tiny_chat_fixedtime_short_small.toml",
        "tiny_chat_fixedtime_short_small_promoted.toml",
        "deep_math_fixediters32_base.toml",
        "deep_math_fixediters64_base.toml",
        "deep_math_fixediters96_base.toml",
        "deep_math_tool_boundary_fixediters64_base.toml",
        "deep_math_tool_boundary_fixediters96_base.toml",
        "deep_math_tool_boundary_fixediters128_base.toml",
        "deep_math_tool_boundary_multiscale_clock8_topquarter_fixediters64.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_fixediters64.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_fixediters96.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_fixediters128.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_summary16_fixediters128.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_summary32_lowresid_fixediters128.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_summary32_lowresid_eventwrite_fixediters128.toml",
        "deep_math_tool_boundary_multiscale_clock4_tophalf_summary32_lowresid_surprise0p05_fixediters128.toml",
        "deep_math_multiscale_clock8_topquarter_fixediters32.toml",
        "deep_math_multiscale_clock8_topquarter_fixediters64.toml",
        "deep_math_multiscale_clock8_topquarter_fixediters96.toml",
        "deep_math_multiscale_clock8_topquarter_mhc2_top1_iter2_fixediters64.toml",
        "deep_math_multiscale_clock8_topquarter_mhc2_top1_iter2_fixediters96.toml",
        "deep_math_multiscale_clock8_topquarter_mhc2_top2_iter2_fixediters64.toml",
        "gsm8k_fixediters32_base.toml",
        "gsm8k_multiscale_clock8_topquarter_fixediters32.toml",
        "gsm8k_fixediters64_base.toml",
        "gsm8k_multiscale_clock8_topquarter_fixediters64.toml",
        "gsm8k_multiscale_clock8_topquarter_mhc2_top1_iter2_fixediters64.toml",
        "gsm8k_tool_boundary_fixediters64_base.toml",
        "gsm8k_tool_boundary_multiscale_clock8_topquarter_fixediters64.toml",
        "gsm8k_tool_boundary_multiscale_clock4_tophalf_fixediters64.toml",
        "gsm8k_tool_boundary_fixediters96_base.toml",
        "gsm8k_tool_boundary_multiscale_clock4_tophalf_fixediters96.toml",
        "gsm8k_tool_boundary_fixediters128_base.toml",
        "gsm8k_tool_boundary_multiscale_clock4_tophalf_fixediters128.toml",
        "gsm8k_tool_boundary_multiscale_clock4_tophalf_summary32_lowresid_eventwrite_fixediters128.toml",
        "gsm8k_tool_boundary_multiscale_clock4_tophalf_summary64_lowresid_eventwrite_fixediters128.toml",
        "orca_math_fixediters32_base.toml",
        "orca_math_multiscale_clock8_topquarter_fixediters32.toml",
        "orca_math_fixediters64_base.toml",
        "orca_math_multiscale_clock8_topquarter_fixediters64.toml",
        "orca_math_multiscale_clock8_topquarter_mhc2_top1_iter2_fixediters64.toml",
        "orca_math_tool_boundary_fixediters64_base.toml",
        "orca_math_tool_boundary_multiscale_clock8_topquarter_fixediters64.toml",
        "orca_math_tool_boundary_multiscale_clock4_tophalf_fixediters64.toml",
        "orca_math_tool_boundary_fixediters96_base.toml",
        "orca_math_tool_boundary_multiscale_clock8_topquarter_fixediters96.toml",
        "orca_math_tool_boundary_multiscale_clock4_tophalf_fixediters96.toml",
        "orca_math_tool_boundary_fixediters128_base.toml",
        "orca_math_tool_boundary_multiscale_clock4_tophalf_fixediters128.toml",
        "orca_math_tool_boundary_multiscale_clock4_tophalf_summary32_lowresid_eventwrite_fixediters128.toml",
        "webscale_rl_fixediters32_base.toml",
        "webscale_rl_multiscale_clock8_topquarter_fixediters32.toml",
        "webscale_rl_fixediters64_base.toml",
        "webscale_rl_multiscale_clock8_topquarter_fixediters64.toml",
        "tiny_chat_fixedtime_short_rho2x.toml",
        "tiny_chat_fixedtime_short_rho2x_matchwall.toml",
        "tiny_chat_multiscale_fast2_short.toml",
        "tiny_chat_multiscale_fast4_short.toml",
        "tiny_chat_multiscale_fast4_ycarry_top1_chunk16_short.toml",
        "tiny_chat_multiscale_fast2_ycarry_top1_chunk16_short.toml",
        "tiny_chat_multiscale_ycarry_top2_chunk32_short.toml",
        "tiny_chat_multiscale_screen_base.toml",
        "tiny_chat_multiscale_fast2_screen.toml",
        "tiny_chat_multiscale_fast4_screen.toml",
        "tiny_chat_multiscale_ycarry_top1_chunk16_screen.toml",
        "tiny_chat_multiscale_fast2_ycarry_top1_chunk16_screen.toml",
        "tiny_chat_multiscale_ycarry_top2_chunk32_screen.toml",
        "tiny_chat_multiscale_clock4_tophalf_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter2_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter1_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter2_tau0p2_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter2_nobranch_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter2_tau0p05_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top2_iter2_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top2_iter2_matchwall_screen.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top2_iter2_fixediters32.toml",
        "tiny_chat_multiscale_clock8_topquarter_mhc2_top1_iter2_fixediters32.toml",
        "tiny_chat_multiscale_clock8_topquarter_fixediters32.toml",
        "tiny_chat_multiscale_clock8_topquarter_short.toml",
        "tiny_chat_multiscale_3bank_clock8_summary32_screen.toml",
        "tiny_chat_multiscale_3bank_clock8_summary32_lowresid_screen.toml",
        "tiny_chat_multiscale_3bank_clock8_summary16_screen.toml",
        "tiny_chat_multiscale_3bank_clock8_summary32_lowdecay_screen.toml",
        "tiny_chat_multiscale_micro_base.toml",
        "tiny_chat_multiscale_fast2_micro.toml",
        "tiny_chat_multiscale_fast4_micro.toml",
        "experiments/nemotron_climbmix_sequence_kernel_smoke_base.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_trainfast_dense_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_mamba_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_64step_base.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_64step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_64step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_mamba_64step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_64step_owt_valid_base.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_64step_owt_valid.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_dense_score_64step_owt_valid.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_64step_owt_valid.toml",
        "experiments/nemotron_climbmix_sequence_kernel_mamba_64step_owt_valid.toml",
        "experiments/nemotron_climbmix_sequence_kernel_256step_base.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_256step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_256step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_mamba_256step.toml",
        "experiments/nemotron_climbmix_sequence_kernel_linear_65536_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_65536_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_mamba_65536_smoke.toml",
        "experiments/nemotron_climbmix_sequence_kernel_rwkv8_100m_smoke.toml",
        "experiments/nemotron_climbmix_bdh_mhc_phase4_constant_latent65536_trainfast_dense_large.toml",
        "experiments/nemotron_climbmix_bdh_mhc_phase4_constant_latent65536_trainfast_dense_large_owt_valid.toml",
    ];
    let base_path = root.join("base.toml");

    for file in files {
        let paths = if file == "base.toml" {
            vec![base_path.clone()]
        } else {
            vec![base_path.clone(), root.join(file)]
        };
        let config: TrainingConfig = load_training_config(&paths).unwrap_or_else(|err| {
            panic!("failed to load language config from {paths:?}: {err}");
        });
        config
            .validate()
            .unwrap_or_else(|err| panic!("language config validation failed: {err}"));

        let roundtripped: TrainingConfig = roundtrip_config(&config);
        roundtripped
            .validate()
            .unwrap_or_else(|err| panic!("roundtripped config validation failed: {err}"));
    }
}

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
fn language_loader_supports_relative_extends() {
    let dir = tempdir().expect("tempdir");

    let base = write_config(
        dir.path(),
        "base.toml",
        r#"
        [dataset]
        cache_dir = "data/tiny_chat"
        train_split_ratio = 0.95
        type = "tiny_chat"
        max_records = 64

        [training]
        block_size = 128
        batch_size = 8
        max_iters = 16
        epochs = 1
        log_frequency = 4

        [optimizer]
        learning_rate = 0.001
        weight_decay = 0.1

        [optimizer.lr_schedule]
        type = "cosine"
        min_lr = 0.00005
        num_iters = 16

        [generation]
        prompt = "User: hello\nAssistant:"
        temperature = 0.9
        top_k = 1

        [model]
        n_layer = 4
        n_embd = 192
        n_head = 4
        mlp_internal_dim_multiplier = 4
        dropout = 0.1
        "#,
    );
    let overlay = write_config(
        dir.path(),
        "overlay.toml",
        r#"
        extends = "base.toml"

        [training]
        batch_size = 11

        [model]
        rollout_fast_steps_per_slow_step = 4
        "#,
    );

    let config = load_training_config(&[overlay]).expect("load extended config");
    assert_eq!(config.training.batch_size, 11);
    assert_eq!(config.training.block_size, 128);
    assert_eq!(config.model.rollout_fast_steps_per_slow_step, Some(4));
    assert!(matches!(
        config.dataset.source,
        super::train::DatasetSourceConfig::TinyChat {
            max_records: Some(64),
            ..
        }
    ));

    let config_with_base =
        load_training_config(&[base, dir.path().join("overlay.toml")]).expect("load config");
    assert_eq!(config_with_base.training.batch_size, 11);
    assert_eq!(
        config_with_base.model.rollout_fast_steps_per_slow_step,
        Some(4)
    );
}

#[test]
fn baseline_configs_do_not_extend_ambiguous_sibling_base_toml() {
    fn visit(dir: &Path, files: &mut Vec<PathBuf>) {
        for entry in fs::read_dir(dir).expect("read baseline dir") {
            let entry = entry.expect("dir entry");
            let path = entry.path();
            if path.is_dir() {
                visit(&path, files);
            } else if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                files.push(path);
            }
        }
    }

    let baselines_root = config_root().join("baselines");
    let mut files = Vec::new();
    visit(&baselines_root, &mut files);

    for path in files {
        if path.file_name().and_then(|name| name.to_str()) == Some("base.toml") {
            continue;
        }
        let content = fs::read_to_string(&path).expect("read baseline config");
        let value: toml::Value =
            toml::from_str(&content).unwrap_or_else(|err| panic!("parse {path:?}: {err}"));
        let extends = value.get("extends");
        let has_ambiguous_local_base = match extends {
            Some(toml::Value::String(value)) => value == "base.toml",
            Some(toml::Value::Array(values)) => values.iter().any(|value| match value {
                toml::Value::String(value) => value == "base.toml",
                _ => false,
            }),
            _ => false,
        };
        assert!(
            !has_ambiguous_local_base,
            "baseline config {} must not extend ambiguous sibling base.toml; use ../base.toml or a named fragment instead",
            path.display()
        );
    }
}

#[test]
fn promoted_baseline_registry_entries_exist_and_load() {
    let baselines_root = config_root().join("baselines");
    let registry_path = baselines_root.join("registry.toml");
    let registry: PromotedBaselineRegistry =
        toml::from_str(&fs::read_to_string(&registry_path).expect("read registry"))
            .expect("parse promoted baseline registry");

    let mut seen_names = BTreeSet::new();
    let mut seen_paths = BTreeSet::new();
    assert!(
        !registry.entries.is_empty(),
        "promoted baseline registry must not be empty"
    );

    for entry in registry.entries {
        assert!(
            seen_names.insert(entry.name.clone()),
            "duplicate promoted baseline name `{}`",
            entry.name
        );
        assert!(
            seen_paths.insert(entry.path.clone()),
            "duplicate promoted baseline path `{}`",
            entry.path.display()
        );
        assert!(
            !entry.family.trim().is_empty(),
            "promoted baseline `{}` must declare a family",
            entry.name
        );
        let full_path = baselines_root.join(&entry.path);
        assert!(
            full_path.is_file(),
            "promoted baseline `{}` points to missing file {}",
            entry.name,
            full_path.display()
        );
        match entry.kind {
            PromotedBaselineKind::TrainingConfig => {
                let config = load_training_config(&[full_path.clone()]).unwrap_or_else(|err| {
                    panic!("load training config {}: {err}", full_path.display())
                });
                config.validate().unwrap_or_else(|err| {
                    panic!("validate training config {}: {err}", full_path.display())
                });
            }
            PromotedBaselineKind::BundleConfig => {
                load_experiment_bundle_config(&full_path).unwrap_or_else(|err| {
                    panic!("load bundle config {}: {err}", full_path.display())
                });
            }
        }
    }
}

#[test]
fn shakespeare_deployed_repro_resolves_to_the_old_shakespeare_recipe() {
    let config = load_config_from_root("baselines/shakespeare_deployed_repro.toml");
    config.validate().expect("validate shakespeare repro");

    assert!(matches!(
        config.dataset.source,
        DatasetSourceConfig::Shakespeare { .. }
    ));
    assert!(matches!(
        config.dataset.tokenizer.kind,
        TokenizerKind::Char(_)
    ));
    assert_eq!(config.training.block_size, 512);
    assert_eq!(config.training.batch_size, 24);
    assert_eq!(config.training.epochs, Some(30));
    assert_eq!(config.training.max_iters, 3000);
    assert_eq!(config.model.n_layer, Some(4));
    assert_eq!(config.model.n_embd, Some(128));
    assert_eq!(config.model.n_head, Some(4));
    assert_eq!(config.training.sequence_kernel_override, None);
    assert_eq!(
        config.model.residual_connector,
        Some(ResidualConnectorKind::Vanilla)
    );
    assert!(matches!(
        config.model.mhc.as_ref(),
        Some(mhc) if !mhc.enabled
    ));
    assert!(matches!(
        config.model.attention_residual.as_ref(),
        Some(attn) if !attn.enabled
    ));
}

#[test]
fn shakespeare_deployed_dense_score_train_only_differs_by_training_kernel_override() {
    let repro = load_config_from_root("baselines/shakespeare_deployed_repro.toml");
    repro.validate().expect("validate shakespeare repro");

    let dense = load_config_from_root("baselines/shakespeare_deployed_dense_score_train.toml");
    dense
        .validate()
        .expect("validate shakespeare dense-score training baseline");

    assert!(matches!(
        dense.dataset.source,
        DatasetSourceConfig::Shakespeare { .. }
    ));
    assert!(matches!(
        dense.dataset.tokenizer.kind,
        TokenizerKind::Char(_)
    ));
    assert_eq!(dense.training.block_size, repro.training.block_size);
    assert_eq!(dense.training.batch_size, repro.training.batch_size);
    assert_eq!(dense.training.epochs, repro.training.epochs);
    assert_eq!(dense.training.max_iters, repro.training.max_iters);
    assert_eq!(dense.model.n_layer, repro.model.n_layer);
    assert_eq!(dense.model.n_embd, repro.model.n_embd);
    assert_eq!(dense.model.n_head, repro.model.n_head);
    assert_eq!(
        dense.model.residual_connector,
        repro.model.residual_connector
    );
    assert_eq!(
        dense.training.sequence_kernel_override,
        Some(burn_dragon_core::SequenceKernelConfig::dense_score_short_context())
    );
    assert_eq!(repro.training.sequence_kernel_override, None);
}

#[test]
fn shakespeare_small_attention_residual_baseline_matches_deployed_recipe_except_connector() {
    let config = load_config_from_root("baselines/shakespeare_small_attention_residual.toml");
    config
        .validate()
        .expect("validate shakespeare attention residual baseline");

    assert!(matches!(
        config.dataset.source,
        DatasetSourceConfig::Shakespeare { .. }
    ));
    assert!(matches!(
        config.dataset.tokenizer.kind,
        TokenizerKind::Char(_)
    ));
    assert_eq!(config.training.block_size, 512);
    assert_eq!(config.training.batch_size, 24);
    assert_eq!(config.training.epochs, Some(30));
    assert_eq!(config.model.n_embd, Some(128));
    assert_eq!(config.model.n_layer, Some(4));
    assert_eq!(config.model.n_head, Some(4));
    assert_eq!(
        config.model.residual_connector,
        Some(ResidualConnectorKind::AttentionResidual)
    );
    assert!(matches!(
        config.model.mhc.as_ref(),
        Some(mhc) if !mhc.enabled
    ));
    assert!(matches!(
        config.model.attention_residual.as_ref(),
        Some(attn) if attn.enabled
    ));
}
