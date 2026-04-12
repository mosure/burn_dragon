use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::config::DragonExperimentKind;

#[cfg(feature = "native")]
mod native;
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
mod web;

#[cfg(feature = "native")]
pub use native::{
    apply_native_downgrade_state, clear_native_downgrade, load_matching_native_downgrade,
    persist_native_downgrade,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub use web::{
    apply_browser_downgrade_state, clear_browser_downgrade, load_browser_downgrade,
    persist_browser_downgrade,
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonCapabilityDowngradeRecord {
    pub scope_fingerprint: String,
    pub experiment_kind: DragonExperimentKind,
    pub backend_label: String,
    pub downgrade_to: String,
    pub observed_training_bytes: u64,
    pub trainer_budget_bytes: Option<u64>,
    pub reason: String,
    pub source: String,
    pub observed_at: DateTime<Utc>,
    pub failure_count: u32,
}

#[derive(Serialize)]
struct DragonCapabilityScope<'a, M> {
    experiment_kind: DragonExperimentKind,
    backend_label: &'a str,
    model_config: &'a M,
    batch_size: usize,
    block_size: usize,
}

pub fn capability_scope_fingerprint<M: Serialize>(
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &M,
    batch_size: usize,
    block_size: usize,
) -> String {
    let scope = DragonCapabilityScope {
        experiment_kind,
        backend_label,
        model_config,
        batch_size,
        block_size,
    };
    let encoded = serde_json::to_vec(&scope).expect("capability scope should serialize");
    let mut hasher = Sha256::new();
    hasher.update(encoded);
    format!("{:x}", hasher.finalize())
}

fn record_is_still_binding(
    record: &DragonCapabilityDowngradeRecord,
    current_trainer_budget_bytes: Option<u64>,
) -> bool {
    let failed_budget_bytes = record
        .trainer_budget_bytes
        .unwrap_or(record.observed_training_bytes)
        .max(record.observed_training_bytes);
    match current_trainer_budget_bytes {
        Some(current_budget) => current_budget <= failed_budget_bytes,
        None => true,
    }
}
