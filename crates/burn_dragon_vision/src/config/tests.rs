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
        "vision/distill/baselines/smoke.toml",
        "vision/distill/baselines/tiny.toml",
        "vision/distill/baselines/small.toml",
        "vision/distill/baselines/base.toml",
        "vision/distill/baselines/balanced_224.toml",
        "vision/distill/baselines/richer_280.toml",
        "vision/distill/frontier/graph_bridge_multiteacher_medium_280.toml",
        "vision/distill/frontier/graph_bridge_multimode_spatial_medium_280.toml",
        "vision/distill/frontier/graph_bridge_multiteacher_medium_280_spatial_smoke.toml",
        "vision/distill/frontier/graph_bridge_multiteacher_base_336.toml",
        "vision/distill/frontier/graph_bridge_multimode_spatial_base_336.toml",
        "vision/trm/baselines/graph_scene_slots_imagenette.toml",
        "vision/trm/baselines/graph_bridge_imagenette.toml",
        "vision/rac/base.toml",
        "vision/rac/baselines/tiny.toml",
        "vision/rac/baselines/smoke.toml",
        "vision/rac/cifar10_artifact_validate.toml",
        "vision/rac/experiments/cifar10/cellular_short.toml",
        "vision/rac/experiments/cifar10/dense_short.toml",
        "vision/rac/experiments/cifar10/cellular_reset_short.toml",
        "vision/rac/experiments/cifar10/cellular_write_disabled_short.toml",
        "vision/rac/experiments/cifar10/cellular_phase2_k8_64.toml",
        "vision/rac/experiments/cifar10/dense_sequential_phase2_k8_64.toml",
        "vision/rac/experiments/cifar10/cellular_full_bptt_k8_medium.toml",
        "vision/rac/experiments/cifar10/cellular_tbptt_k8_medium.toml",
        "vision/rac/experiments/cifar10/dense_sequential_k8_medium.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_smoke.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_long.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_precomputed_taesd_smoke.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_precomputed_taesd_smoke_subset.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_precomputed_taesd_medium_subset.toml",
        "vision/rac/experiments/imagenet1k/dense_subset_precomputed_taesd_long_subset.toml",
        "vision/video_lejepa/baselines/smoke.toml",
        "vision/video_lejepa/baselines/tiny.toml",
        "vision/video_lejepa/baselines/small.toml",
        "vision/video_lejepa/baselines/base.toml",
        "vision/video_lejepa/baselines/vjepa21_dense_promoted.toml",
        "vision/video_lejepa/baselines/vjepa21_imagenet1k_dense_smoke.toml",
        "vision/video_lejepa/baselines/vjepa21_imagenet1k_dense_long.toml",
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
fn scene_slot_graph_imagenette_programmatic_baseline_matches_checked_in_config() {
    let root = config_root();
    let path = root.join("vision/trm/baselines/graph_scene_slots_imagenette.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic = VisionTrainingConfig::scene_slot_graph_imagenette_baseline();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic graph scene-slot baseline should validate");
}

#[test]
fn scene_slot_graph_bridge_imagenette_programmatic_baseline_matches_checked_in_config() {
    let root = config_root();
    let path = root.join("vision/trm/baselines/graph_bridge_imagenette.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic = VisionTrainingConfig::scene_slot_graph_bridge_imagenette_baseline();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic graph bridge baseline should validate");
}

#[test]
fn scene_slot_graph_bridge_multiteacher_medium_programmatic_baseline_matches_checked_in_config() {
    let root = config_root();
    let path = root.join("vision/distill/frontier/graph_bridge_multiteacher_medium_280.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic =
        VisionTrainingConfig::scene_slot_graph_bridge_multiteacher_imagenet1k_medium_launch();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic multiteacher medium baseline should validate");
}

#[test]
fn scene_slot_graph_bridge_multiteacher_base_programmatic_baseline_matches_checked_in_config() {
    let root = config_root();
    let path = root.join("vision/distill/frontier/graph_bridge_multiteacher_base_336.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic =
        VisionTrainingConfig::scene_slot_graph_bridge_multiteacher_imagenet1k_base_launch();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic multiteacher base baseline should validate");
}

#[test]
fn scene_slot_graph_bridge_multimode_spatial_medium_programmatic_baseline_matches_checked_in_config()
 {
    let root = config_root();
    let path = root.join("vision/distill/frontier/graph_bridge_multimode_spatial_medium_280.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic =
        VisionTrainingConfig::scene_slot_graph_bridge_multimode_spatial_imagenet1k_medium_launch();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic multimode spatial medium baseline should validate");
}

#[test]
fn scene_slot_graph_bridge_multimode_spatial_base_programmatic_baseline_matches_checked_in_config()
{
    let root = config_root();
    let path = root.join("vision/distill/frontier/graph_bridge_multimode_spatial_base_336.toml");
    let loaded: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&path)).expect("load checked-in config");
    let programmatic =
        VisionTrainingConfig::scene_slot_graph_bridge_multimode_spatial_imagenet1k_base_launch();

    assert_eq!(programmatic, loaded);
    programmatic
        .validate()
        .expect("programmatic multimode spatial base baseline should validate");
}

#[test]
fn rac_configs_parse_serialize_validate() {
    let root = config_root().join("vision").join("rac");
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", root.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .collect::<Vec<_>>();
    files.sort();

    assert!(
        !files.is_empty(),
        "expected at least one rac overlay config in {}",
        root.display()
    );

    for path in files {
        let config: VisionTrainingConfig = load_vision_training_config(std::slice::from_ref(&path))
            .unwrap_or_else(|err| {
                panic!("failed to load rac config {}: {err}", path.display());
            });
        config
            .validate()
            .unwrap_or_else(|err| panic!("rac config validation failed: {err}"));

        let roundtripped: VisionTrainingConfig = roundtrip_config(&config);
        roundtripped.validate().unwrap_or_else(|err| {
            panic!(
                "roundtripped rac config validation failed for {}: {err}",
                path.display()
            )
        });
    }
}

#[test]
fn rac_experiment_configs_parse_serialize_validate() {
    let root = config_root()
        .join("vision")
        .join("rac")
        .join("experiments")
        .join("cifar10");
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", root.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .collect::<Vec<_>>();
    files.sort();

    assert!(
        !files.is_empty(),
        "expected at least one rac experiment config in {}",
        root.display()
    );

    for path in files {
        let config: VisionTrainingConfig = load_vision_training_config(std::slice::from_ref(&path))
            .unwrap_or_else(|err| {
                panic!(
                    "failed to load rac experiment config {}: {err}",
                    path.display()
                );
            });
        config
            .validate()
            .unwrap_or_else(|err| panic!("rac experiment config validation failed: {err}"));

        let roundtripped: VisionTrainingConfig = roundtrip_config(&config);
        roundtripped.validate().unwrap_or_else(|err| {
            panic!(
                "roundtripped rac experiment config validation failed for {}: {err}",
                path.display()
            )
        });
    }
}

#[test]
fn rac_cifar10_experiment_configs_stay_curated() {
    let root = config_root()
        .join("vision")
        .join("rac")
        .join("experiments")
        .join("cifar10");
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", root.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .filter_map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    files.sort();

    assert_eq!(
        files,
        vec![
            "cellular_full_bptt_k8_medium.toml".to_string(),
            "cellular_phase2_k8_64.toml".to_string(),
            "cellular_reset_short.toml".to_string(),
            "cellular_short.toml".to_string(),
            "cellular_tbptt_k8_medium.toml".to_string(),
            "cellular_write_disabled_short.toml".to_string(),
            "dense_sequential_k8_medium.toml".to_string(),
            "dense_sequential_phase2_k8_64.toml".to_string(),
            "dense_short.toml".to_string(),
        ]
    );
}

#[test]
fn rac_imagenet1k_experiment_configs_stay_curated() {
    let root = config_root()
        .join("vision")
        .join("rac")
        .join("experiments")
        .join("imagenet1k");
    let mut files = fs::read_dir(&root)
        .unwrap_or_else(|err| panic!("failed to read {}: {err}", root.display()))
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|ext| ext.to_str()) == Some("toml"))
        .filter_map(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(str::to_string)
        })
        .collect::<Vec<_>>();
    files.sort();

    assert_eq!(
        files,
        vec![
            "dense_subset_long.toml".to_string(),
            "dense_subset_precomputed_taesd_long_subset.toml".to_string(),
            "dense_subset_precomputed_taesd_medium_subset.toml".to_string(),
            "dense_subset_precomputed_taesd_smoke.toml".to_string(),
            "dense_subset_precomputed_taesd_smoke_subset.toml".to_string(),
            "dense_subset_smoke.toml".to_string(),
        ]
    );
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
        let config: VisionTrainingConfig = load_vision_training_config(std::slice::from_ref(&path))
            .unwrap_or_else(|err| {
                panic!(
                    "failed to load video_lejepa config {}: {err}",
                    path.display()
                );
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

#[test]
fn moving_mnist_vjepa21_dense_promoted_baseline_matches_latest_wide_recipe() {
    let root = config_root().join("vision").join("video_lejepa");
    let baseline_path = root.join("baselines").join("vjepa21_dense_promoted.toml");
    let latest_path =
        root.join("moving_mnist_vjepa21_dense_wgpu_diag_probe025_recon1_256_wide.toml");

    let baseline: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&baseline_path))
            .unwrap_or_else(|err| panic!("failed to load {}: {err}", baseline_path.display()));
    let latest: VisionTrainingConfig =
        load_vision_training_config(std::slice::from_ref(&latest_path))
            .unwrap_or_else(|err| panic!("failed to load {}: {err}", latest_path.display()));

    assert_eq!(baseline, latest);
    baseline
        .validate()
        .expect("promoted dense V-JEPA 2.1 baseline should validate");
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

#[test]
fn all_vision_config_extends_targets_exist() {
    fn collect_toml_files(root: &Path) -> Vec<PathBuf> {
        let mut stack = vec![root.to_path_buf()];
        let mut files = Vec::new();
        let local_root = root.join("local");
        while let Some(dir) = stack.pop() {
            if dir == local_root {
                continue;
            }
            let entries = fs::read_dir(&dir)
                .unwrap_or_else(|err| panic!("failed to read {}: {err}", dir.display()));
            for entry in entries {
                let path = entry
                    .unwrap_or_else(|err| {
                        panic!("failed to read dir entry in {}: {err}", dir.display())
                    })
                    .path();
                if path.is_dir() {
                    stack.push(path);
                } else if path.extension().and_then(|ext| ext.to_str()) == Some("toml") {
                    files.push(path);
                }
            }
        }
        files.sort();
        files
    }

    fn extract_extends(path: &Path) -> Vec<PathBuf> {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read {}: {err}", path.display()));
        let value: toml::Value = toml::from_str(&text)
            .unwrap_or_else(|err| panic!("failed to parse {}: {err}", path.display()));
        let Some(extends) = value.get("extends") else {
            return Vec::new();
        };
        match extends {
            toml::Value::String(value) => vec![PathBuf::from(value)],
            toml::Value::Array(values) => values
                .iter()
                .map(|value| match value {
                    toml::Value::String(value) => PathBuf::from(value),
                    other => panic!(
                        "extends in {} must contain only strings, got {other:?}",
                        path.display()
                    ),
                })
                .collect(),
            other => panic!(
                "extends in {} must be string or array of strings, got {other:?}",
                path.display()
            ),
        }
    }

    let root = config_root().join("vision");
    let files = collect_toml_files(&root);
    assert!(
        !files.is_empty(),
        "expected at least one config file under {}",
        root.display()
    );

    let mut missing = Vec::new();
    for path in files {
        let parent = path.parent().expect("config parent");
        for extend in extract_extends(&path) {
            let resolved = parent.join(&extend);
            if !resolved.exists() {
                missing.push(format!("{} -> {}", path.display(), resolved.display()));
            }
        }
    }

    assert!(
        missing.is_empty(),
        "missing extends targets:\n{}",
        missing.join("\n")
    );
}
