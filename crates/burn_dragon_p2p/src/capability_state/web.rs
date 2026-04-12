use anyhow::{Result, anyhow};
use burn_p2p_browser::{BrowserAppTarget, BrowserRuntimeRole, BrowserWorkerSupport};
use sha2::{Digest, Sha256};

use super::{
    DragonCapabilityDowngradeRecord, capability_scope_fingerprint, record_is_still_binding,
};
use crate::capability::DragonBrowserCapabilityDecision;
use crate::config::DragonBrowserTrainingConfig;

const BROWSER_STATE_PREFIX: &str = "burn-dragon-p2p.capability-downgrade.v1";

fn browser_storage() -> Result<web_sys::Storage> {
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .local_storage()
        .map_err(|error| anyhow!("localStorage unavailable: {error:?}"))?
        .ok_or_else(|| anyhow!("localStorage unavailable"))
}

fn browser_storage_key(scope_fingerprint: &str, edge_base_url: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(edge_base_url.trim_end_matches('/').as_bytes());
    format!(
        "{BROWSER_STATE_PREFIX}:{}:{scope_fingerprint}",
        hex::encode(hasher.finalize())
    )
}

fn browser_scope_fingerprint(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    backend_label: &str,
) -> (String, String) {
    let scope_fingerprint = capability_scope_fingerprint(
        config.experiment_kind,
        backend_label,
        &config.model_config,
        config.batch_size,
        config.block_size,
    );
    let storage_key = browser_storage_key(&scope_fingerprint, edge_base_url);
    (scope_fingerprint, storage_key)
}

pub fn load_browser_downgrade(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    backend_label: &str,
    current_trainer_budget_bytes: Option<u64>,
) -> Result<Option<DragonCapabilityDowngradeRecord>> {
    let (_scope_fingerprint, storage_key) =
        browser_scope_fingerprint(edge_base_url, config, backend_label);
    let Some(encoded) = browser_storage()?
        .get_item(&storage_key)
        .map_err(|error| anyhow!("failed to read browser downgrade state: {error:?}"))?
    else {
        return Ok(None);
    };
    let record: DragonCapabilityDowngradeRecord = serde_json::from_str(&encoded)?;
    Ok(record_is_still_binding(&record, current_trainer_budget_bytes).then_some(record))
}

pub fn persist_browser_downgrade(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    backend_label: &str,
    decision: &DragonBrowserCapabilityDecision,
    reason: &str,
    source: &str,
) -> Result<DragonCapabilityDowngradeRecord> {
    let (scope_fingerprint, storage_key) =
        browser_scope_fingerprint(edge_base_url, config, backend_label);
    let footprint = decision
        .footprint
        .as_ref()
        .ok_or_else(|| anyhow!("missing browser footprint for downgrade persistence"))?;
    let existing = browser_storage()?
        .get_item(&storage_key)
        .map_err(|error| anyhow!("failed to read browser downgrade state: {error:?}"))?;
    let mut record = existing
        .as_deref()
        .and_then(|value| serde_json::from_str::<DragonCapabilityDowngradeRecord>(value).ok())
        .unwrap_or(DragonCapabilityDowngradeRecord {
            scope_fingerprint,
            experiment_kind: config.experiment_kind,
            backend_label: backend_label.to_owned(),
            downgrade_to: "browser_verifier".into(),
            observed_training_bytes: footprint.estimated_training_bytes,
            trainer_budget_bytes: decision.trainer_memory_budget_bytes,
            reason: reason.to_owned(),
            source: source.to_owned(),
            observed_at: chrono::Utc::now(),
            failure_count: 0,
        });
    record.downgrade_to = "browser_verifier".into();
    record.observed_training_bytes = footprint.estimated_training_bytes;
    record.trainer_budget_bytes = decision.trainer_memory_budget_bytes;
    record.reason = reason.to_owned();
    record.source = source.to_owned();
    record.observed_at = chrono::Utc::now();
    record.failure_count = record.failure_count.saturating_add(1);
    browser_storage()?
        .set_item(&storage_key, &serde_json::to_string(&record)?)
        .map_err(|error| anyhow!("failed to persist browser downgrade state: {error:?}"))?;
    Ok(record)
}

pub fn clear_browser_downgrade(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    backend_label: &str,
) -> Result<()> {
    let (_scope_fingerprint, storage_key) =
        browser_scope_fingerprint(edge_base_url, config, backend_label);
    let _ = browser_storage()?
        .remove_item(&storage_key)
        .map_err(|error| anyhow!("failed to clear browser downgrade state: {error:?}"))?;
    Ok(())
}

pub fn apply_browser_downgrade_state(
    edge_base_url: &str,
    config: &DragonBrowserTrainingConfig,
    backend_label: &str,
    mut decision: DragonBrowserCapabilityDecision,
) -> DragonBrowserCapabilityDecision {
    let Ok(Some(record)) = load_browser_downgrade(
        edge_base_url,
        config,
        backend_label,
        decision.trainer_memory_budget_bytes,
    ) else {
        return decision;
    };

    let can_verifier = config.capability_policy.allow_browser_verifier_fallback
        && matches!(
            decision.capability.dedicated_worker,
            BrowserWorkerSupport::DedicatedWorker
        );
    decision.can_train = false;
    decision.training_budget = None;
    decision.capability.recommended_role = if can_verifier {
        BrowserRuntimeRole::BrowserVerifier
    } else {
        BrowserRuntimeRole::BrowserObserver
    };
    decision.connect_target = if can_verifier {
        BrowserAppTarget::Validate
    } else {
        BrowserAppTarget::Observe
    };
    decision.downgrade_reason = Some(format!(
        "persisted trainer failure for this workload fingerprint at {}: {}; holding browser verifier/observer role until the trainer budget increases or the workload changes",
        record.observed_at, record.reason
    ));
    decision
}

#[cfg(test)]
mod tests {
    use wasm_bindgen_test::wasm_bindgen_test;

    use super::*;
    use crate::capability::{DragonBrowserHostCapabilityProbe, decide_browser_capability};
    use crate::config::{
        DragonBrowserExecutionBackend, DragonBrowserTokenSource, DragonCapabilityPolicy,
        DragonExperimentKind,
    };

    #[wasm_bindgen_test]
    fn browser_persisted_downgrade_unbinds_when_budget_increases() {
        let config = crate::config::DragonBrowserTrainingConfig {
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            model_config: burn_dragon_core::BDHConfig {
                n_layer: 2,
                n_embd: 32,
                n_head: 4,
                vocab_size: 256,
                ..burn_dragon_core::BDHConfig::default()
            },
            execution_backend: DragonBrowserExecutionBackend::Wgpu,
            block_size: 8,
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            batch_size: 2,
            max_train_batches: Some(1),
            max_eval_batches: Some(1),
            capability_policy: DragonCapabilityPolicy::default(),
            training_lease: None,
            train_source: DragonBrowserTokenSource::Inline {
                records: Vec::new(),
            },
            eval_source: None,
            live_participant: None,
        };
        let probe = DragonBrowserHostCapabilityProbe {
            navigator_gpu_exposed: true,
            worker_gpu_exposed: true,
            dedicated_worker_exposed: true,
            persistent_storage_exposed: true,
            web_transport_exposed: true,
            web_rtc_exposed: true,
            system_memory_bytes: Some(8 * 1024 * 1024 * 1024),
        };
        let edge_base_url = "https://edge.example";
        let decision: DragonBrowserCapabilityDecision =
            decide_browser_capability(Some(&config), &probe);
        clear_browser_downgrade(edge_base_url, &config, "wgpu").expect("clear prior state");
        persist_browser_downgrade(edge_base_url, &config, "wgpu", &decision, "oom", "runtime")
            .expect("persist browser downgrade");
        let blocked =
            apply_browser_downgrade_state(edge_base_url, &config, "wgpu", decision.clone());
        assert!(!blocked.can_train);

        let mut raised_budget = config.clone();
        let required = decision
            .footprint
            .as_ref()
            .expect("footprint")
            .estimated_training_bytes;
        let failed_budget = decision.trainer_memory_budget_bytes.unwrap_or(required);
        raised_budget
            .capability_policy
            .browser_wgpu_memory_budget_bytes = Some(failed_budget.max(required).saturating_mul(2));
        let raised_decision = decide_browser_capability(Some(&raised_budget), &probe);
        let unblocked =
            apply_browser_downgrade_state(edge_base_url, &raised_budget, "wgpu", raised_decision);
        assert!(unblocked.can_train);
        clear_browser_downgrade(edge_base_url, &config, "wgpu").expect("clear state");
    }
}
