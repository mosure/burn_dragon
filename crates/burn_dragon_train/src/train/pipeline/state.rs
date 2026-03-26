use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

pub const BUNDLE_STATE_FILE_NAME: &str = "bundle_state.json";
pub const STAGE_STATE_FILE_NAME: &str = "stage_state.json";
pub const RESOLVED_CONFIG_FILE_NAME: &str = "resolved_config.toml";

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StageStatus {
    #[default]
    Pending,
    Running,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(bound(
    deserialize = "Artifact: Deserialize<'de> + Default",
    serialize = "Artifact: Serialize"
))]
pub struct StageState<Artifact> {
    pub stage_name: String,
    pub status: StageStatus,
    #[serde(default)]
    pub started_at_unix_secs: Option<u64>,
    #[serde(default)]
    pub completed_at_unix_secs: Option<u64>,
    #[serde(default)]
    pub last_error: Option<String>,
    #[serde(default)]
    pub artifact: Artifact,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(bound(
    deserialize = "Artifact: Deserialize<'de> + Default",
    serialize = "Artifact: Serialize"
))]
pub struct BundleState<Artifact> {
    pub bundle_name: String,
    pub bundle_root: PathBuf,
    #[serde(default)]
    pub latest_completed_stage: Option<String>,
    pub stages: Vec<StageState<Artifact>>,
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

pub fn load_stage_state<Artifact>(stage_dir: &Path) -> Result<Option<StageState<Artifact>>>
where
    Artifact: DeserializeOwned + Default,
{
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

pub fn write_stage_state<Artifact>(stage_dir: &Path, state: &StageState<Artifact>) -> Result<()>
where
    Artifact: Serialize,
{
    fs::create_dir_all(stage_dir)
        .with_context(|| format!("failed to create {}", stage_dir.display()))?;
    let path = stage_state_path(stage_dir);
    let payload = serde_json::to_string_pretty(state).context("serialize stage state")?;
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn build_bundle_state<Artifact>(
    bundle_name: impl Into<String>,
    bundle_root: &Path,
    stage_states: Vec<StageState<Artifact>>,
) -> BundleState<Artifact> {
    let latest_completed_stage = stage_states
        .iter()
        .rev()
        .find(|stage| stage.status == StageStatus::Completed)
        .map(|stage| stage.stage_name.clone());
    BundleState {
        bundle_name: bundle_name.into(),
        bundle_root: bundle_root.to_path_buf(),
        latest_completed_stage,
        stages: stage_states,
    }
}

pub fn write_bundle_state<Artifact>(bundle_root: &Path, state: &BundleState<Artifact>) -> Result<()>
where
    Artifact: Serialize,
{
    fs::create_dir_all(bundle_root)
        .with_context(|| format!("failed to create {}", bundle_root.display()))?;
    let path = bundle_state_path(bundle_root);
    let payload = serde_json::to_string_pretty(state).context("serialize bundle state")?;
    fs::write(&path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn unix_timestamp_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

#[cfg(test)]
mod tests {
    use super::{
        BUNDLE_STATE_FILE_NAME, BundleState, RESOLVED_CONFIG_FILE_NAME, STAGE_STATE_FILE_NAME,
        StageState, StageStatus, build_bundle_state, bundle_state_path, load_stage_state,
        resolved_stage_config_path, stage_state_path, write_bundle_state, write_stage_state,
    };
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    #[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
    struct TestArtifact {
        value: usize,
    }

    #[test]
    fn stage_state_roundtrips_through_json_file() {
        let dir = tempdir().expect("tempdir");
        let stage_dir = dir.path().join("stage");
        let state = StageState {
            stage_name: "train".to_string(),
            status: StageStatus::Running,
            started_at_unix_secs: Some(11),
            completed_at_unix_secs: None,
            last_error: None,
            artifact: TestArtifact { value: 7 },
        };

        write_stage_state(&stage_dir, &state).expect("write stage state");
        let loaded = load_stage_state::<TestArtifact>(&stage_dir)
            .expect("load stage state")
            .expect("stage state present");

        assert_eq!(
            stage_state_path(&stage_dir),
            stage_dir.join(STAGE_STATE_FILE_NAME)
        );
        assert_eq!(loaded, state);
    }

    #[test]
    fn build_bundle_state_tracks_latest_completed_stage() {
        let dir = tempdir().expect("tempdir");
        let bundle_root = dir.path().join("bundle");
        let stages = vec![
            StageState {
                stage_name: "gen".to_string(),
                status: StageStatus::Completed,
                started_at_unix_secs: Some(1),
                completed_at_unix_secs: Some(2),
                last_error: None,
                artifact: TestArtifact { value: 1 },
            },
            StageState {
                stage_name: "train".to_string(),
                status: StageStatus::Running,
                started_at_unix_secs: Some(3),
                completed_at_unix_secs: None,
                last_error: None,
                artifact: TestArtifact { value: 2 },
            },
        ];

        let bundle = build_bundle_state("demo", &bundle_root, stages.clone());
        assert_eq!(
            bundle,
            BundleState {
                bundle_name: "demo".to_string(),
                bundle_root: bundle_root.clone(),
                latest_completed_stage: Some("gen".to_string()),
                stages,
            }
        );

        write_bundle_state(&bundle_root, &bundle).expect("write bundle state");
        assert_eq!(
            bundle_state_path(&bundle_root),
            bundle_root.join(BUNDLE_STATE_FILE_NAME)
        );
        assert_eq!(
            resolved_stage_config_path(&bundle_root),
            bundle_root.join(RESOLVED_CONFIG_FILE_NAME)
        );
    }
}
