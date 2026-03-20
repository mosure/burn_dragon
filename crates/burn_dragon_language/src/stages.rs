#![cfg(feature = "train")]

use std::collections::{BTreeMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use burn_dragon_checkpoint::resolve_checkpoint_base;
use burn_dragon_universality::{NcaCorpusConfig, load_nca_config};
use serde::{Deserialize, Serialize};

use crate::config::merge::load_merged_toml;
use crate::config::{DatasetSourceConfig, TrainingConfig, load_training_config};
use crate::tokenizer::TokenizerKind;

pub const BUNDLE_STATE_FILE_NAME: &str = "bundle_state.json";
pub const STAGE_STATE_FILE_NAME: &str = "stage_state.json";
pub const RESOLVED_CONFIG_FILE_NAME: &str = "resolved_config.toml";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentBackend {
    #[default]
    Ndarray,
    Cuda,
    Wgpu,
    WgpuNoFusion,
}

impl ExperimentBackend {
    pub fn as_cli_arg(self) -> &'static str {
        match self {
            Self::Ndarray => "ndarray",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::WgpuNoFusion => "wgpu-no-fusion",
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentBundleConfig {
    pub name: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_true")]
    pub resume_from_last_completed_stage: bool,
    pub stages: Vec<ExperimentStageConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentStageConfig {
    pub name: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(flatten)]
    pub kind: ExperimentStageKind,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ExperimentStageKind {
    UniversalityGenerate {
        config: PathBuf,
    },
    LanguageTrain {
        config: PathBuf,
        #[serde(default)]
        backend: ExperimentBackend,
        #[serde(default)]
        dataset_manifest_from_stage: Option<String>,
        #[serde(default)]
        init_checkpoint_from_stage: Option<String>,
        #[serde(default)]
        resume_from_last_checkpoint: bool,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExperimentStageStatus {
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
pub struct ExperimentStageArtifact {
    #[serde(default)]
    pub corpus_output_dir: Option<PathBuf>,
    #[serde(default)]
    pub manifest_path: Option<PathBuf>,
    #[serde(default)]
    pub sample_records_path: Option<PathBuf>,
    #[serde(default)]
    pub preview_dir: Option<PathBuf>,
    #[serde(default)]
    pub run_root: Option<PathBuf>,
    #[serde(default)]
    pub latest_run_dir: Option<PathBuf>,
    #[serde(default)]
    pub latest_checkpoint_dir: Option<PathBuf>,
    #[serde(default)]
    pub latest_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub resolved_config_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentStageState {
    pub stage_name: String,
    pub status: ExperimentStageStatus,
    #[serde(default)]
    pub started_at_unix_secs: Option<u64>,
    #[serde(default)]
    pub completed_at_unix_secs: Option<u64>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub artifact: ExperimentStageArtifact,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ExperimentBundleState {
    pub bundle_name: String,
    pub bundle_root: PathBuf,
    #[serde(default)]
    pub latest_completed_stage: Option<String>,
    pub stages: Vec<ExperimentStageState>,
}

pub fn load_experiment_bundle_config(path: &Path) -> Result<ExperimentBundleConfig> {
    let config: ExperimentBundleConfig = load_merged_toml(&[path.to_path_buf()])?;
    config.validate(path)
}

impl ExperimentBundleConfig {
    pub fn validate(&self, config_path: &Path) -> Result<Self> {
        if self.name.trim().is_empty() {
            return Err(anyhow!("bundle name must not be empty"));
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(anyhow!("bundle output_dir must not be empty"));
        }
        if self.stages.is_empty() {
            return Err(anyhow!("bundle must contain at least one stage"));
        }

        let mut seen = HashSet::new();
        for (index, stage) in self.stages.iter().enumerate() {
            if stage.name.trim().is_empty() {
                return Err(anyhow!("stages[{index}].name must not be empty"));
            }
            if !seen.insert(stage.name.clone()) {
                return Err(anyhow!("duplicate stage name `{}`", stage.name));
            }
            match &stage.kind {
                ExperimentStageKind::UniversalityGenerate { config } => {
                    if config.as_os_str().is_empty() {
                        return Err(anyhow!(
                            "stages[{index}].config must not be empty for universality_generate"
                        ));
                    }
                }
                ExperimentStageKind::LanguageTrain {
                    config,
                    dataset_manifest_from_stage,
                    init_checkpoint_from_stage,
                    ..
                } => {
                    if config.as_os_str().is_empty() {
                        return Err(anyhow!(
                            "stages[{index}].config must not be empty for language_train"
                        ));
                    }
                    for reference in [dataset_manifest_from_stage, init_checkpoint_from_stage]
                        .into_iter()
                        .flatten()
                    {
                        let Some(dep_index) = self
                            .stages
                            .iter()
                            .position(|candidate| candidate.name == *reference)
                        else {
                            return Err(anyhow!(
                                "stages[{index}] references unknown stage `{reference}`"
                            ));
                        };
                        if dep_index >= index {
                            return Err(anyhow!(
                                "stages[{index}] reference `{reference}` must point to an earlier stage"
                            ));
                        }
                    }
                }
            }
            for dependency in &stage.depends_on {
                if dependency == &stage.name {
                    return Err(anyhow!(
                        "stages[{index}].depends_on cannot reference itself"
                    ));
                }
                let Some(dep_index) = self
                    .stages
                    .iter()
                    .position(|candidate| candidate.name == *dependency)
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

        let _ = config_path;
        Ok(self.clone())
    }
}

pub fn resolve_bundle_root(bundle_config_path: &Path, config: &ExperimentBundleConfig) -> PathBuf {
    if config.output_dir.is_absolute() {
        config.output_dir.clone()
    } else {
        let _ = bundle_config_path;
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(&config.output_dir)
    }
}

pub fn resolve_stage_dir(
    bundle_root: &Path,
    index: usize,
    stage: &ExperimentStageConfig,
) -> PathBuf {
    bundle_root
        .join("stages")
        .join(format!("{index:02}_{}", stage.name))
}

pub fn stage_state_path(stage_dir: &Path) -> PathBuf {
    stage_dir.join(STAGE_STATE_FILE_NAME)
}

pub fn bundle_state_path(bundle_root: &Path) -> PathBuf {
    bundle_root.join(BUNDLE_STATE_FILE_NAME)
}

pub fn resolved_stage_config_path(stage_dir: &Path) -> PathBuf {
    stage_dir.join(RESOLVED_CONFIG_FILE_NAME)
}

pub fn load_stage_state(stage_dir: &Path) -> Result<Option<ExperimentStageState>> {
    let path = stage_state_path(stage_dir);
    if !path.is_file() {
        return Ok(None);
    }
    let payload =
        fs::read_to_string(&path).with_context(|| format!("failed to read {}", path.display()))?;
    let state = serde_json::from_str(&payload)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(Some(state))
}

pub fn write_stage_state(stage_dir: &Path, state: &ExperimentStageState) -> Result<()> {
    fs::create_dir_all(stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;
    let path = stage_state_path(stage_dir);
    let payload = serde_json::to_string_pretty(state).context("serialize stage state")?;
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn build_bundle_state(
    config: &ExperimentBundleConfig,
    bundle_root: &Path,
    stage_states: Vec<ExperimentStageState>,
) -> ExperimentBundleState {
    let latest_completed_stage = stage_states
        .iter()
        .rev()
        .find(|stage| stage.status == ExperimentStageStatus::Completed)
        .map(|stage| stage.stage_name.clone());
    ExperimentBundleState {
        bundle_name: config.name.clone(),
        bundle_root: bundle_root.to_path_buf(),
        latest_completed_stage,
        stages: stage_states,
    }
}

pub fn write_bundle_state(bundle_root: &Path, state: &ExperimentBundleState) -> Result<()> {
    fs::create_dir_all(bundle_root)
        .with_context(|| format!("failed to create {}", bundle_root.display()))?;
    let path = bundle_state_path(bundle_root);
    let payload = serde_json::to_string_pretty(state).context("serialize bundle state")?;
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn resolve_stage_dependency_artifacts(
    config: &ExperimentBundleConfig,
    bundle_root: &Path,
) -> Result<BTreeMap<String, ExperimentStageArtifact>> {
    let mut artifacts = BTreeMap::new();
    for (index, stage) in config.stages.iter().enumerate() {
        let stage_dir = resolve_stage_dir(bundle_root, index, stage);
        if let Some(state) = load_stage_state(&stage_dir)?
            && state.status == ExperimentStageStatus::Completed
        {
            artifacts.insert(stage.name.clone(), state.artifact);
        }
    }
    Ok(artifacts)
}

pub fn prepare_universality_stage_config(
    bundle_config_path: &Path,
    stage_dir: &Path,
    source_config_path: &Path,
) -> Result<NcaCorpusConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let mut config = load_nca_config(&source_path)?;
    config.output_dir = stage_dir.join("output");
    Ok(config)
}

pub fn prepare_language_stage_config(
    bundle_config_path: &Path,
    source_config_path: &Path,
    stage_dir: &Path,
    stage: &ExperimentStageConfig,
    dependency_artifacts: &BTreeMap<String, ExperimentStageArtifact>,
) -> Result<TrainingConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let mut config = load_training_config(&[source_path])?;

    if let ExperimentStageKind::LanguageTrain {
        dataset_manifest_from_stage,
        init_checkpoint_from_stage,
        ..
    } = &stage.kind
    {
        if let Some(stage_name) = dataset_manifest_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let manifest_path = artifact
                .manifest_path
                .clone()
                .ok_or_else(|| anyhow!("stage `{stage_name}` did not produce a manifest_path"))?;
            config.dataset.cache_dir = artifact.corpus_output_dir.clone().unwrap_or_else(|| {
                manifest_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .to_path_buf()
            });
            config.dataset.source = DatasetSourceConfig::UniversalityManifest {
                manifest: manifest_path,
            };
            if !matches!(
                config.dataset.tokenizer.kind,
                TokenizerKind::Pretokenized(_)
            ) {
                return Err(anyhow!(
                    "language stage `{}` requires tokenizer.type = `pretokenized` when sourcing a universality manifest",
                    stage.name
                ));
            }
        }
        if let Some(stage_name) = init_checkpoint_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let checkpoint_dir = artifact.latest_checkpoint_dir.clone().ok_or_else(|| {
                anyhow!("stage `{stage_name}` did not produce a latest_checkpoint_dir")
            })?;
            config.training.init_checkpoint_path = Some(checkpoint_dir);
            config.training.init_checkpoint_epoch = artifact.latest_checkpoint_epoch;
            config.training.resume_run_dir = None;
            config.training.resume_checkpoint_epoch = None;
        }
    }

    let stage_run_root = stage_dir.join("runs");
    if let ExperimentStageKind::LanguageTrain {
        resume_from_last_checkpoint: true,
        ..
    } = &stage.kind
        && let Some(latest_run_dir) = crate::checkpoint::resolve_latest_run_dir_in(&stage_run_root)
    {
        config.training.resume_run_dir = Some(latest_run_dir);
        config.training.resume_checkpoint_epoch = None;
        config.training.init_checkpoint_path = None;
        config.training.init_checkpoint_epoch = None;
    }

    Ok(config)
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

pub fn resolve_training_stage_artifact(stage_dir: &Path) -> Result<ExperimentStageArtifact> {
    let run_root = stage_dir.join("runs");
    let latest_run_dir = crate::checkpoint::resolve_latest_run_dir_in(&run_root)
        .ok_or_else(|| anyhow!("no latest run available under {}", run_root.display()))?;
    let checkpoint_dir = latest_run_dir.join("checkpoint");
    let (_, epoch) = resolve_checkpoint_base(&checkpoint_dir, None).with_context(|| {
        format!(
            "failed to resolve latest checkpoint in {}",
            checkpoint_dir.display()
        )
    })?;
    Ok(ExperimentStageArtifact {
        run_root: Some(run_root),
        latest_run_dir: Some(latest_run_dir),
        latest_checkpoint_dir: Some(checkpoint_dir),
        latest_checkpoint_epoch: Some(epoch),
        resolved_config_path: Some(resolved_stage_config_path(stage_dir)),
        ..ExperimentStageArtifact::default()
    })
}

pub fn unix_timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn resolve_relative_to(bundle_config_path: &Path, relative_or_absolute: &Path) -> PathBuf {
    if relative_or_absolute.is_absolute() {
        relative_or_absolute.to_path_buf()
    } else {
        bundle_config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(relative_or_absolute)
    }
}

fn default_true() -> bool {
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn bundle_config_validates_stage_references() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("bundle.toml");
        fs::write(
            &config_path,
            r#"
name = "demo"
output_dir = "runs/demo"

[[stages]]
name = "gen"
type = "universality_generate"
config = "gen.toml"

[[stages]]
name = "train"
type = "language_train"
config = "train.toml"
backend = "ndarray"
dataset_manifest_from_stage = "gen"
"#,
        )
        .expect("write config");
        let config = load_experiment_bundle_config(&config_path).expect("bundle config");
        assert_eq!(config.stages.len(), 2);
    }

    #[test]
    fn prepare_language_stage_config_injects_manifest_and_checkpoint() {
        let dir = tempdir().expect("tempdir");
        let bundle_path = dir.path().join("bundle.toml");
        fs::write(
            &bundle_path,
            "name = \"demo\"\noutput_dir = \"runs/demo\"\n",
        )
        .expect("write bundle");
        let train_cfg = dir.path().join("train.toml");
        fs::write(
            &train_cfg,
            r#"
[dataset]
cache_dir = "cache"
type = "universality_manifest"
manifest = "placeholder.json"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[training]
block_size = 64
batch_size = 2
max_iters = 4
log_frequency = 1

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "1 2 3"

[model]
n_layer = 1
n_embd = 8
n_head = 1
mlp_internal_dim_multiplier = 1
fused_kernels = true
rotary_embedding = "alibi"
"#,
        )
        .expect("write training config");

        let stage = ExperimentStageConfig {
            name: "train".to_string(),
            depends_on: Vec::new(),
            kind: ExperimentStageKind::LanguageTrain {
                config: PathBuf::from("train.toml"),
                backend: ExperimentBackend::Ndarray,
                dataset_manifest_from_stage: Some("gen".to_string()),
                init_checkpoint_from_stage: Some("pre".to_string()),
                resume_from_last_checkpoint: false,
            },
        };
        let mut artifacts = BTreeMap::new();
        artifacts.insert(
            "gen".to_string(),
            ExperimentStageArtifact {
                manifest_path: Some(dir.path().join("gen/manifest.json")),
                corpus_output_dir: Some(dir.path().join("gen")),
                ..ExperimentStageArtifact::default()
            },
        );
        artifacts.insert(
            "pre".to_string(),
            ExperimentStageArtifact {
                latest_checkpoint_dir: Some(dir.path().join("pre/checkpoint")),
                latest_checkpoint_epoch: Some(3),
                ..ExperimentStageArtifact::default()
            },
        );

        let config = prepare_language_stage_config(
            &bundle_path,
            Path::new("train.toml"),
            &dir.path().join("stage"),
            &stage,
            &artifacts,
        )
        .expect("prepare stage config");

        assert!(matches!(
            config.dataset.source,
            DatasetSourceConfig::UniversalityManifest { .. }
        ));
        assert_eq!(
            config.training.init_checkpoint_path,
            Some(dir.path().join("pre/checkpoint"))
        );
        assert_eq!(config.training.init_checkpoint_epoch, Some(3));
    }

    #[test]
    fn relative_bundle_output_dir_resolves_from_cwd() {
        let config = ExperimentBundleConfig {
            name: "demo".to_string(),
            output_dir: PathBuf::from("runs/demo"),
            resume_from_last_completed_stage: true,
            stages: vec![ExperimentStageConfig {
                name: "gen".to_string(),
                depends_on: Vec::new(),
                kind: ExperimentStageKind::UniversalityGenerate {
                    config: PathBuf::from("gen.toml"),
                },
            }],
        };
        let bundle_path = PathBuf::from("config/language/bundles/demo.toml");
        let expected = std::env::current_dir().expect("cwd").join("runs/demo");
        assert_eq!(resolve_bundle_root(&bundle_path, &config), expected);
    }

    #[test]
    fn current_best_large_baseline_bundle_loads() {
        let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("config")
            .join("language")
            .join("baselines");
        let bundle_path = root.join("current_best_large.toml");
        let config = load_experiment_bundle_config(&bundle_path).expect("baseline bundle");
        assert_eq!(config.name, "current_best_large_48h");
        assert_eq!(config.stages.len(), 2);
        assert_eq!(config.stages[0].name, "nca_prepretrain");
        assert_eq!(config.stages[1].name, "climbmix_pretrain");
    }
}
