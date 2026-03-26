use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::Serialize;

pub fn validate_named_stage_bundle<Stage, StageNameFn, StageDepsFn, ValidateStageFn>(
    bundle_name: &str,
    output_dir: &Path,
    stages: &[Stage],
    stage_name: StageNameFn,
    stage_dependencies: StageDepsFn,
    validate_stage: ValidateStageFn,
) -> Result<()>
where
    StageNameFn: Fn(&Stage) -> &str,
    StageDepsFn: Fn(&Stage) -> &[String],
    ValidateStageFn: Fn(usize, &Stage, &[Stage]) -> Result<()>,
{
    if bundle_name.trim().is_empty() {
        return Err(anyhow!("bundle name must not be empty"));
    }
    if output_dir.as_os_str().is_empty() {
        return Err(anyhow!("bundle output_dir must not be empty"));
    }
    if stages.is_empty() {
        return Err(anyhow!("bundle must contain at least one stage"));
    }

    let mut seen = HashSet::new();
    for (index, stage) in stages.iter().enumerate() {
        let name = stage_name(stage);
        if name.trim().is_empty() {
            return Err(anyhow!("stages[{index}].name must not be empty"));
        }
        if !seen.insert(name.to_string()) {
            return Err(anyhow!("duplicate stage name `{name}`"));
        }
        validate_stage(index, stage, stages)?;
        for dependency in stage_dependencies(stage) {
            if dependency == name {
                return Err(anyhow!(
                    "stages[{index}].depends_on cannot reference itself"
                ));
            }
            let Some(dep_index) = stages
                .iter()
                .position(|candidate| stage_name(candidate) == dependency.as_str())
            else {
                return Err(anyhow!(
                    "stages[{index}].depends_on references unknown stage `{dependency}`"
                ));
            };
            if dep_index >= index {
                return Err(anyhow!(
                    "stages[{index}].depends_on must point to earlier stages only"
                ));
            }
        }
    }

    Ok(())
}

pub fn resolve_bundle_root(output_dir: &Path) -> PathBuf {
    if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(output_dir)
    }
}

pub fn resolve_named_stage_dir(bundle_root: &Path, index: usize, stage_name: &str) -> PathBuf {
    bundle_root
        .join("stages")
        .join(format!("{index:02}_{stage_name}"))
}

pub fn write_resolved_config<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let payload = toml::to_string_pretty(value).context("serialize resolved config")?;
    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::anyhow;
    use tempfile::tempdir;

    use crate::train::pipeline::{
        resolve_bundle_root, resolve_named_stage_dir, validate_named_stage_bundle,
        write_resolved_config,
    };

    #[derive(Debug, Clone)]
    struct TestStage {
        name: &'static str,
        deps: Vec<String>,
        value: usize,
    }

    #[test]
    fn validate_named_stage_bundle_rejects_duplicate_or_forward_dependencies() {
        let err = validate_named_stage_bundle(
            "demo",
            Path::new("runs/demo"),
            &[
                TestStage {
                    name: "one",
                    deps: Vec::new(),
                    value: 1,
                },
                TestStage {
                    name: "one",
                    deps: vec!["two".to_string()],
                    value: 2,
                },
            ],
            |stage| stage.name,
            |stage| stage.deps.as_slice(),
            |_index, _stage, _all| Ok(()),
        )
        .expect_err("duplicate name should fail");

        assert!(err.to_string().contains("duplicate stage name"));
    }

    #[test]
    fn validate_named_stage_bundle_delegates_stage_specific_validation() {
        let err = validate_named_stage_bundle(
            "demo",
            Path::new("runs/demo"),
            &[TestStage {
                name: "one",
                deps: Vec::new(),
                value: 0,
            }],
            |stage| stage.name,
            |stage| stage.deps.as_slice(),
            |index, stage, _all| {
                if stage.value == 0 {
                    return Err(anyhow!("stages[{index}].value must be nonzero"));
                }
                Ok(())
            },
        )
        .expect_err("stage-specific validation should fail");

        assert!(err.to_string().contains("value must be nonzero"));
    }

    #[test]
    fn bundle_layout_helpers_are_deterministic() {
        let root = resolve_bundle_root(Path::new("runs/demo"));
        assert!(root.ends_with(Path::new("runs/demo")));

        let stage_dir = resolve_named_stage_dir(&PathBuf::from("/tmp/demo"), 3, "train");
        assert_eq!(stage_dir, PathBuf::from("/tmp/demo/stages/03_train"));
    }

    #[test]
    fn write_resolved_config_writes_pretty_toml() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("resolved.toml");
        let value = serde_json::json!({
            "name": "demo",
            "count": 3
        });

        write_resolved_config(&path, &value).expect("write resolved config");
        let contents = fs::read_to_string(&path).expect("read resolved config");
        assert!(contents.contains("name = \"demo\""));
        assert!(contents.contains("count = 3"));
    }
}
