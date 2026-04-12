use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::Result;
use burn_dragon_language::TrainingConfig;

use super::{
    DragonCapabilityDowngradeRecord, capability_scope_fingerprint, record_is_still_binding,
};
use crate::capability::{DragonNativeCapabilityAssessment, DragonTrainingFootprint};
use crate::config::{DragonExperimentKind, DragonNativeTarget};

const NATIVE_STATE_FILE_NAME: &str = "dragon-capability-downgrades.json";

fn native_state_path(storage_root: &Path) -> PathBuf {
    storage_root.join("state").join(NATIVE_STATE_FILE_NAME)
}

fn load_native_state_map(
    storage_root: &Path,
) -> Result<BTreeMap<String, DragonCapabilityDowngradeRecord>> {
    let path = native_state_path(storage_root);
    if !path.is_file() {
        return Ok(BTreeMap::new());
    }
    let bytes = fs::read(&path)?;
    Ok(serde_json::from_slice(&bytes)?)
}

fn persist_native_state_map(
    storage_root: &Path,
    state: &BTreeMap<String, DragonCapabilityDowngradeRecord>,
) -> Result<()> {
    let path = native_state_path(storage_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_vec_pretty(state)?)?;
    Ok(())
}

pub fn load_matching_native_downgrade<M: serde::Serialize>(
    storage_root: &Path,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &M,
    batch_size: usize,
    block_size: usize,
    current_trainer_budget_bytes: Option<u64>,
) -> Result<Option<DragonCapabilityDowngradeRecord>> {
    let scope_fingerprint = capability_scope_fingerprint(
        experiment_kind,
        backend_label,
        model_config,
        batch_size,
        block_size,
    );
    let state = load_native_state_map(storage_root)?;
    Ok(state
        .get(&scope_fingerprint)
        .cloned()
        .filter(|record| record_is_still_binding(record, current_trainer_budget_bytes)))
}

pub fn persist_native_downgrade<M: serde::Serialize>(
    storage_root: &Path,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &M,
    batch_size: usize,
    block_size: usize,
    footprint: &DragonTrainingFootprint,
    trainer_budget_bytes: Option<u64>,
    downgrade_to: &str,
    reason: &str,
    source: &str,
) -> Result<DragonCapabilityDowngradeRecord> {
    let scope_fingerprint = capability_scope_fingerprint(
        experiment_kind,
        backend_label,
        model_config,
        batch_size,
        block_size,
    );
    let mut state = load_native_state_map(storage_root)?;
    let mut record =
        state
            .get(&scope_fingerprint)
            .cloned()
            .unwrap_or(DragonCapabilityDowngradeRecord {
                scope_fingerprint: scope_fingerprint.clone(),
                experiment_kind,
                backend_label: backend_label.to_owned(),
                downgrade_to: downgrade_to.to_owned(),
                observed_training_bytes: footprint.estimated_training_bytes,
                trainer_budget_bytes,
                reason: reason.to_owned(),
                source: source.to_owned(),
                observed_at: chrono::Utc::now(),
                failure_count: 0,
            });
    record.downgrade_to = downgrade_to.to_owned();
    record.observed_training_bytes = footprint.estimated_training_bytes;
    record.trainer_budget_bytes = trainer_budget_bytes;
    record.reason = reason.to_owned();
    record.source = source.to_owned();
    record.observed_at = chrono::Utc::now();
    record.failure_count = record.failure_count.saturating_add(1);
    state.insert(scope_fingerprint, record.clone());
    persist_native_state_map(storage_root, &state)?;
    Ok(record)
}

pub fn clear_native_downgrade<M: serde::Serialize>(
    storage_root: &Path,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &M,
    batch_size: usize,
    block_size: usize,
) -> Result<()> {
    let scope_fingerprint = capability_scope_fingerprint(
        experiment_kind,
        backend_label,
        model_config,
        batch_size,
        block_size,
    );
    let mut state = load_native_state_map(storage_root)?;
    state.remove(&scope_fingerprint);
    persist_native_state_map(storage_root, &state)
}

pub fn apply_native_downgrade_state(
    storage_root: &Path,
    config: &TrainingConfig,
    mut assessment: DragonNativeCapabilityAssessment,
) -> Result<DragonNativeCapabilityAssessment> {
    if let Some(record) = load_matching_native_downgrade(
        storage_root,
        assessment.experiment_kind,
        &assessment.backend_label,
        &assessment.model_config,
        config.training.batch_size,
        config.training.block_size,
        assessment.target_decision.trainer_memory_budget_bytes,
    )? {
        assessment.target_decision.effective_target = DragonNativeTarget::Validator;
        assessment.target_decision.can_train = false;
        assessment.target_decision.downgrade_reason = Some(format!(
            "persisted trainer failure for this workload fingerprint at {}: {}; holding validator role until the trainer budget increases or the workload changes",
            record.observed_at, record.reason
        ));
    }
    Ok(assessment)
}

#[cfg(test)]
mod tests {
    use burn_dragon_language::BDHConfig;

    use super::*;

    #[test]
    fn native_downgrade_record_unbinds_when_budget_increases() {
        let root = tempfile::tempdir().expect("tempdir");
        let model = BDHConfig {
            n_layer: 2,
            n_embd: 32,
            n_head: 4,
            vocab_size: 256,
            ..BDHConfig::default()
        };
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 128,
            estimated_optimizer_state_bytes: 256,
            estimated_activation_bytes: 256,
            estimated_training_bytes: 1024,
            estimated_checkpoint_bytes: 512,
            estimated_shard_bytes: 256,
            estimated_tokens_per_second: 123.0,
        };
        persist_native_downgrade(
            root.path(),
            DragonExperimentKind::NcaPrepretraining,
            "wgpu",
            &model,
            2,
            64,
            &footprint,
            Some(512),
            "validator",
            "oom",
            "runtime",
        )
        .expect("persist downgrade");

        assert!(
            load_matching_native_downgrade(
                root.path(),
                DragonExperimentKind::NcaPrepretraining,
                "wgpu",
                &model,
                2,
                64,
                Some(512),
            )
            .expect("load downgrade")
            .is_some()
        );
        assert!(
            load_matching_native_downgrade(
                root.path(),
                DragonExperimentKind::NcaPrepretraining,
                "wgpu",
                &model,
                2,
                64,
                Some(2048),
            )
            .expect("load downgrade")
            .is_none()
        );
    }
}
