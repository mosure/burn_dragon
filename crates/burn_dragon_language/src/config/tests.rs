use std::path::PathBuf;

use serde::Serialize;
use serde::de::DeserializeOwned;

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
    let files = ["base.toml", "tiny.toml", "small.toml", "large.toml"];
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
