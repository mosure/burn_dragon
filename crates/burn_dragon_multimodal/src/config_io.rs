use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::Value as JsonValue;

pub(crate) fn load_merged_config<T>(config_paths: &[PathBuf]) -> Result<T>
where
    T: Default + DeserializeOwned + Serialize,
{
    let merged = load_merged_value(config_paths, T::default())?;
    serde_json::from_value(merged).context("failed to deserialize merged config")
}

pub(crate) fn load_merged_value<T>(config_paths: &[PathBuf], defaults: T) -> Result<JsonValue>
where
    T: Serialize,
{
    let mut merged = serde_json::to_value(defaults).context("failed to seed config defaults")?;
    for path in config_paths {
        let value = load_config_value(path)?;
        merge_json_value(&mut merged, value);
    }
    Ok(merged)
}

fn load_config_value(path: &Path) -> Result<JsonValue> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read config {}", path.display()))?;
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display())),
        Some("toml") => {
            let value: toml::Value = toml::from_str(&contents)
                .with_context(|| format!("failed to parse {}", path.display()))?;
            serde_json::to_value(value).context("failed to convert TOML config to JSON value")
        }
        _ => Err(anyhow!(
            "config {} must use .json or .toml",
            path.display()
        )),
    }
}

pub(crate) fn merge_json_value(base: &mut JsonValue, incoming: JsonValue) {
    match (base, incoming) {
        (JsonValue::Object(base_map), JsonValue::Object(incoming_map)) => {
            for (key, value) in incoming_map {
                match base_map.get_mut(&key) {
                    Some(existing) => merge_json_value(existing, value),
                    None => {
                        base_map.insert(key, value);
                    }
                }
            }
        }
        (base_slot, incoming) => {
            *base_slot = incoming;
        }
    }
}
