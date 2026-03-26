#![cfg(feature = "train")]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use burn_dragon_train::train::pipeline::{
    build_bundle_state as build_bundle_state_shared, bundle_state_path as bundle_state_path_shared,
    load_stage_state as load_stage_state_shared, resolve_bundle_root as resolve_bundle_root_shared,
    resolve_completed_stage_artifacts, resolve_named_stage_dir,
    resolved_stage_config_path as resolved_stage_config_path_shared,
    stage_state_path as stage_state_path_shared, unix_timestamp_now as unix_timestamp_now_shared,
    write_bundle_state as write_bundle_state_shared,
    write_resolved_config as write_resolved_config_shared,
    write_stage_state as write_stage_state_shared,
};
use serde::Serialize;

mod artifacts;
mod prepare;
#[cfg(test)]
mod tests;
mod types;

pub use artifacts::resolve_training_stage_artifact;
pub use prepare::{prepare_language_stage_config, prepare_universality_stage_config};
pub use types::{
    ExperimentBackend, ExperimentBundleConfig, ExperimentBundleState, ExperimentStageArtifact,
    ExperimentStageConfig, ExperimentStageKind, ExperimentStageState, ExperimentStageStatus,
    load_experiment_bundle_config,
};

pub use burn_dragon_train::train::pipeline::{
    BUNDLE_STATE_FILE_NAME, RESOLVED_CONFIG_FILE_NAME, STAGE_STATE_FILE_NAME,
};

pub fn resolve_bundle_root(config: &ExperimentBundleConfig) -> PathBuf {
    resolve_bundle_root_shared(&config.output_dir)
}

pub fn resolve_stage_dir(
    bundle_root: &Path,
    index: usize,
    stage: &ExperimentStageConfig,
) -> PathBuf {
    resolve_named_stage_dir(bundle_root, index, &stage.name)
}

pub fn stage_state_path(stage_dir: &Path) -> PathBuf {
    stage_state_path_shared(stage_dir)
}

pub fn bundle_state_path(bundle_root: &Path) -> PathBuf {
    bundle_state_path_shared(bundle_root)
}

pub fn resolved_stage_config_path(stage_dir: &Path) -> PathBuf {
    resolved_stage_config_path_shared(stage_dir)
}

pub fn load_stage_state(stage_dir: &Path) -> Result<Option<ExperimentStageState>> {
    load_stage_state_shared(stage_dir)
}

pub fn write_stage_state(stage_dir: &Path, state: &ExperimentStageState) -> Result<()> {
    write_stage_state_shared(stage_dir, state)
}

pub fn build_bundle_state(
    config: &ExperimentBundleConfig,
    bundle_root: &Path,
    stage_states: Vec<ExperimentStageState>,
) -> ExperimentBundleState {
    build_bundle_state_shared(config.name.clone(), bundle_root, stage_states)
}

pub fn write_bundle_state(bundle_root: &Path, state: &ExperimentBundleState) -> Result<()> {
    write_bundle_state_shared(bundle_root, state)
}

pub fn resolve_stage_dependency_artifacts(
    config: &ExperimentBundleConfig,
    bundle_root: &Path,
) -> Result<BTreeMap<String, ExperimentStageArtifact>> {
    resolve_completed_stage_artifacts(
        &config.stages,
        bundle_root,
        |stage| stage.name.as_str(),
        resolve_stage_dir,
    )
}

pub fn write_resolved_config<T>(path: &Path, value: &T) -> Result<()>
where
    T: Serialize,
{
    write_resolved_config_shared(path, value)
}

pub fn unix_timestamp_now() -> u64 {
    unix_timestamp_now_shared()
}
