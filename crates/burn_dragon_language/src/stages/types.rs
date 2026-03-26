use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use burn_dragon_train::train::pipeline::{
    BundleState, StageState, StageStatus, TrainingLaunchMode, validate_named_stage_bundle,
};
use serde::{Deserialize, Serialize};

use crate::config::merge::load_merged_toml;

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
        launch_mode: TrainingLaunchMode,
    },
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

pub type ExperimentStageStatus = StageStatus;
pub type ExperimentStageState = StageState<ExperimentStageArtifact>;
pub type ExperimentBundleState = BundleState<ExperimentStageArtifact>;

pub fn load_experiment_bundle_config(path: &Path) -> Result<ExperimentBundleConfig> {
    let config: ExperimentBundleConfig = load_merged_toml(&[path.to_path_buf()])?;
    validate_experiment_bundle_config(config)
}

fn validate_experiment_bundle_config(
    config: ExperimentBundleConfig,
) -> Result<ExperimentBundleConfig> {
    validate_named_stage_bundle(
        &config.name,
        &config.output_dir,
        &config.stages,
        |stage| stage.name.as_str(),
        |stage| stage.depends_on.as_slice(),
        validate_experiment_stage_config,
    )?;
    Ok(config)
}

fn validate_experiment_stage_config(
    index: usize,
    stage: &ExperimentStageConfig,
    stages: &[ExperimentStageConfig],
) -> Result<()> {
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
                let Some(dep_index) = stages
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
    Ok(())
}

fn default_true() -> bool {
    true
}
