use std::path::PathBuf;

use serde::Serialize;
use serde::de::DeserializeOwned;

use super::{VisionTrainingConfig, load_vision_training_config};

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
