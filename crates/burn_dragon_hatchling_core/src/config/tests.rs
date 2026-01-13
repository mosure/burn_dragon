use std::path::PathBuf;

use serde::de::DeserializeOwned;
use serde::Serialize;

use super::{TrainingConfig, VisionTrainingConfig, load_training_config, load_vision_training_config};

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
fn text_configs_parse_serialize_validate() {
    let root = config_root();
    let base = root.join("base.toml");
    let configs: Vec<Vec<PathBuf>> = vec![
        vec![base.clone()],
        vec![base.clone(), root.join("tiny.toml")],
        vec![base.clone(), root.join("small.toml")],
        vec![base.clone(), root.join("large.toml")],
    ];

    for paths in configs {
        let config: TrainingConfig = load_training_config(&paths).unwrap_or_else(|err| {
            panic!("failed to load training config from {paths:?}: {err}");
        });
        config
            .validate()
            .unwrap_or_else(|err| panic!("training config validation failed: {err}"));

        let roundtripped: TrainingConfig = roundtrip_config(&config);
        roundtripped
            .validate()
            .unwrap_or_else(|err| panic!("roundtripped training config validation failed: {err}"));
    }
}

#[test]
fn vision_configs_parse_serialize_validate() {
    let root = config_root();
    let files = [
        "vision_base.toml",
        "vision_mae_tiny.toml",
        "vision_lejepa_tiny.toml",
        "vision_saccade_tiny.toml",
    ];

    for file in files {
        let paths = vec![root.join(file)];
        let config: VisionTrainingConfig =
            load_vision_training_config(&paths).unwrap_or_else(|err| {
                panic!("failed to load vision config from {paths:?}: {err}");
            });
        config
            .validate()
            .unwrap_or_else(|err| panic!("vision config validation failed: {err}"));

        let roundtripped: VisionTrainingConfig = roundtrip_config(&config);
        roundtripped
            .validate()
            .unwrap_or_else(|err| panic!("roundtripped vision config validation failed: {err}"));
    }
}
