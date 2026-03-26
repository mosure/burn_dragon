use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::de::DeserializeOwned;
use toml::Value;

pub(crate) fn load_merged_toml<T>(paths: &[PathBuf]) -> Result<T>
where
    T: DeserializeOwned,
{
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    reject_removed_training_fast_train(&value)?;
    value.try_into::<T>().map_err(|err| anyhow!(err))
}

fn reject_removed_training_fast_train(value: &Value) -> Result<()> {
    match value {
        Value::Table(table) => {
            if let Some(Value::Table(training)) = table.get("training")
                && training.contains_key("fast_train")
            {
                return Err(anyhow!(
                    "training.fast_train has been removed from the language config schema; use training.launch_mode and training.sequence_kernel_override explicitly"
                ));
            }
            for child in table.values() {
                reject_removed_training_fast_train(child)?;
            }
        }
        Value::Array(values) => {
            for child in values {
                reject_removed_training_fast_train(child)?;
            }
        }
        _ => {}
    }
    Ok(())
}

fn load_value(path: &Path) -> Result<Value> {
    let mut stack = Vec::new();
    load_value_recursive(path, &mut stack)
}

fn load_value_recursive(path: &Path, stack: &mut Vec<PathBuf>) -> Result<Value> {
    let canonical = fs::canonicalize(path).with_context(|| {
        format!(
            "failed to canonicalize configuration file {}",
            path.display()
        )
    })?;
    if let Some(idx) = stack.iter().position(|seen| seen == &canonical) {
        let mut cycle = stack[idx..]
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>();
        cycle.push(canonical.display().to_string());
        return Err(anyhow!(
            "config extends cycle detected: {}",
            cycle.join(" -> ")
        ));
    }

    stack.push(canonical.clone());
    let result = (|| {
        let content = fs::read_to_string(&canonical).with_context(|| {
            format!("failed to read configuration file {}", canonical.display())
        })?;
        let table: toml::value::Table = toml::from_str(&content)
            .with_context(|| format!("failed to parse {} as TOML", canonical.display()))?;
        let mut value = Value::Table(table);
        let extends = take_extends(&mut value)
            .with_context(|| format!("failed to parse extends in {}", canonical.display()))?;
        if let Some(extends) = extends {
            let base_dir = canonical.parent().unwrap_or_else(|| Path::new("."));
            let mut merged = Value::Table(toml::value::Table::new());
            for extend in extends {
                let extend_path = base_dir.join(extend);
                let base = load_value_recursive(&extend_path, stack)?;
                merge_values(&mut merged, base);
            }
            merge_values(&mut merged, value);
            Ok(merged)
        } else {
            Ok(value)
        }
    })();
    stack.pop();
    result
}

fn take_extends(value: &mut Value) -> Result<Option<Vec<PathBuf>>> {
    let Value::Table(table) = value else {
        return Ok(None);
    };
    let Some(extends) = table.remove("extends") else {
        return Ok(None);
    };
    match extends {
        Value::String(path) => Ok(Some(vec![PathBuf::from(path)])),
        Value::Array(values) => {
            let mut out = Vec::with_capacity(values.len());
            for value in values {
                match value {
                    Value::String(path) => out.push(PathBuf::from(path)),
                    other => {
                        return Err(anyhow!(
                            "extends entries must be strings, got {}",
                            other.type_str()
                        ));
                    }
                }
            }
            Ok(Some(out))
        }
        other => Err(anyhow!(
            "extends must be a string or array of strings, got {}",
            other.type_str()
        )),
    }
}

pub(crate) fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}
