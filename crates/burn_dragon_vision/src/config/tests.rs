use std::path::PathBuf;
use std::{fs, path::Path};

use serde::Serialize;
use serde::de::DeserializeOwned;
use tempfile::tempdir;

use super::{VisionTrainingConfig, VisionTrainingModeConfig, load_vision_training_config};

fn config_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("config")
}

fn roundtrip_config<T>(config: &T) -> T
where
    T: Serialize + DeserializeOwned,
{
    let serialized = toml::to_string(config).expect("serialize config");
    toml::from_str(&serialized).expect("parse serialized config")
}

#[test]
fn vision_configs_parse_serialize_validate() {
    let root = config_root();
    let files = [
        "vision/base.toml",
        "vision/identity/tiny.toml",
        "vision/mae/tiny.toml",
        "vision/croco/tiny.toml",
        "vision/lejepa/tiny.toml",
        "vision/saccade/tiny.toml",
        "vision/video_lejepa/moving_mnist_trm_artifact_validate.toml",
    ];

    for file in files {
        let paths = vec![root.join(file)];
        let config: VisionTrainingConfig =
            load_vision_training_config(&paths).unwrap_or_else(|err| {
                panic!("failed to load vision config from {paths:?}: {err}");
            });
        config.validate().unwrap_or_else(|err| {
            panic!(
                "vision config validation failed for {}: {err}",
                paths[0].display()
            )
        });

        let roundtripped: VisionTrainingConfig = roundtrip_config(&config);
        roundtripped.validate().unwrap_or_else(|err| {
            panic!(
                "roundtripped vision config validation failed for {}: {err}",
                paths[0].display()
            )
        });
    }
}

#[test]
fn video_lejepa_configs_parse_serialize_validate() {
    let root = config_root().join("vision").join("video_lejepa");
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", root.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .collect::<Vec<_>>();
    files.sort();

    assert!(
        !files.is_empty(),
        "expected at least one video_lejepa overlay config in {}",
        root.display()
    );

    for path in files {
        let config: VisionTrainingConfig =
            load_vision_training_config(std::slice::from_ref(&path)).unwrap_or_else(|err| {
                panic!("failed to load video_lejepa config {}: {err}", path.display());
            });
        config
            .validate()
            .unwrap_or_else(|err| panic!("video_lejepa config validation failed: {err}"));

        let roundtripped: VisionTrainingConfig = roundtrip_config(&config);
        roundtripped.validate().unwrap_or_else(|err| {
            panic!(
                "roundtripped video_lejepa config validation failed for {}: {err}",
                path.display()
            )
        });
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
fn vision_loader_supports_relative_extends() {
    let dir = tempdir().expect("tempdir");

    let base_contents = [
        "[dataset]",
        "source = \"imagenet\"",
        "",
        "[training]",
        "batch_size = 4",
        "max_iters = 8",
        "log_frequency = 1",
        "",
        "[optimizer]",
        "learning_rate = 0.001",
        "weight_decay = 0.0",
        "",
        "[vision]",
        "image_size = 32",
        "patch_size = 4",
        "in_channels = 3",
        "embed_dim = 64",
        "steps = 2",
        "n_head = 8",
        "mlp_internal_dim_multiplier = 2",
        "dropout = 0.0",
        "projection_dim = 32",
        "projection_hidden_dim = 64",
        "use_cls_token = true",
        "cls_sync_alpha = 0.0",
        "num_eyes = 1",
        "cross_eye_steps = 0",
        "token_state_norm = true",
        "pos_encoding = \"learned2d\"",
        "attention_mode = \"row_l1\"",
        "use_alibi = true",
        "fused_kernels = false",
        "",
        "[augment]",
        "image_size = 32",
        "resize_short = 32",
        "min_scale = 1.0",
        "max_scale = 1.0",
        "min_aspect_ratio = 1.0",
        "max_aspect_ratio = 1.0",
        "flip_prob = 0.0",
        "color_jitter_prob = 0.0",
        "brightness = 0.0",
        "contrast = 0.0",
        "saturation = 0.0",
        "hue = 0.0",
        "grayscale_prob = 0.0",
        "blur_prob = 0.0",
        "blur_sigma_min = 0.1",
        "blur_sigma_max = 2.0",
        "solarize_prob = 0.0",
        "solarize_threshold = 128",
        "",
        "[mode]",
        "type = \"distill\"",
        "views = 2",
    ]
    .join("\n");
    let _base = write_config(dir.path(), "base.toml", &base_contents);

    let overlay_contents = [
        "extends = \"base.toml\"",
        "",
        "[training]",
        "batch_size = 7",
    ]
    .join("\n");
    let overlay = write_config(dir.path(), "overlay.toml", &overlay_contents);

    let config = load_vision_training_config(&[overlay]).expect("load extended config");
    assert_eq!(config.training.batch_size, 7);
    assert_eq!(config.training.max_iters, 8);
    assert!(matches!(config.mode, VisionTrainingModeConfig::Distill(_)));
}
