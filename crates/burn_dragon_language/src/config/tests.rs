use std::path::PathBuf;
use std::{fs, path::Path};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tempfile::tempdir;

use super::train::{TrainingConfig, load_training_config};

fn config_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
        .join("language")
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
        "baselines/smoke.toml",
        "baselines/tiny.toml",
        "baselines/small.toml",
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
        fast_train = false

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
