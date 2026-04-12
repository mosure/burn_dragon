use anyhow::{Result, anyhow};
use burn_p2p::ExperimentDirectoryEntry;
use burn_p2p_bootstrap::{AdminAction, AdminResult, AuthPolicyRollout};

use crate::auth::fetch_edge_snapshot;

pub async fn fetch_directory_entries(edge_base_url: &str) -> Result<Vec<ExperimentDirectoryEntry>> {
    Ok(fetch_edge_snapshot(edge_base_url).await?.directory.entries)
}

pub fn upsert_directory_entry(
    entries: &mut Vec<ExperimentDirectoryEntry>,
    replacement: ExperimentDirectoryEntry,
) {
    if let Some(entry) = entries.iter_mut().find(|entry| {
        entry.study_id == replacement.study_id && entry.experiment_id == replacement.experiment_id
    }) {
        *entry = replacement;
    } else {
        entries.push(replacement);
    }
}

pub async fn rollout_directory_entries(
    edge_base_url: &str,
    session_id: &str,
    directory_entries: Vec<ExperimentDirectoryEntry>,
) -> Result<AdminResult> {
    let response = reqwest::Client::new()
        .post(format!("{}/admin", edge_base_url.trim_end_matches('/')))
        .header("x-session-id", session_id)
        .json(&AdminAction::RolloutAuthPolicy(AuthPolicyRollout {
            minimum_revocation_epoch: None,
            directory_entries: Some(directory_entries),
            trusted_issuers: None,
            reenrollment: None,
        }))
        .send()
        .await?;
    let status = response.status();
    if !status.is_success() {
        let body = response.text().await.unwrap_or_else(|_| String::new());
        return Err(anyhow!(
            "admin rollout request failed with {status}: {}",
            body.trim()
        ));
    }
    response.json::<AdminResult>().await.map_err(Into::into)
}
