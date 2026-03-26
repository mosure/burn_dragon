use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::Result;
use serde::Serialize;
use serde::de::DeserializeOwned;

use crate::train::pipeline::{
    BundleState, StageState, StageStatus, build_bundle_state, load_stage_state, unix_timestamp_now,
    write_bundle_state, write_stage_state,
};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct BundleExecutionOptions {
    pub resume_from_last_completed_stage: bool,
    pub stop_after_stage: Option<String>,
}

pub fn resolve_completed_stage_artifacts<Stage, Artifact, StageNameFn, StageDirFn>(
    stages: &[Stage],
    bundle_root: &Path,
    stage_name: StageNameFn,
    stage_dir_for: StageDirFn,
) -> Result<BTreeMap<String, Artifact>>
where
    Artifact: Clone + Default + DeserializeOwned,
    StageNameFn: Fn(&Stage) -> &str,
    StageDirFn: Fn(&Path, usize, &Stage) -> PathBuf,
{
    let mut artifacts = BTreeMap::new();
    for (index, stage) in stages.iter().enumerate() {
        let stage_dir = stage_dir_for(bundle_root, index, stage);
        if let Some(state) = load_stage_state::<Artifact>(&stage_dir)?
            && state.status == StageStatus::Completed
        {
            artifacts.insert(stage_name(stage).to_string(), state.artifact);
        }
    }
    Ok(artifacts)
}

pub fn execute_bundle<Stage, Artifact, StageNameFn, StageDirFn, ValidateFn, RunFn>(
    bundle_name: &str,
    bundle_root: &Path,
    stages: &[Stage],
    options: &BundleExecutionOptions,
    stage_name: StageNameFn,
    stage_dir_for: StageDirFn,
    validate_stage: ValidateFn,
    run_stage: RunFn,
) -> Result<BundleState<Artifact>>
where
    Artifact: Clone + Default + DeserializeOwned + Serialize,
    StageNameFn: Fn(&Stage) -> &str,
    StageDirFn: Fn(&Path, usize, &Stage) -> PathBuf,
    ValidateFn: Fn(&Stage, &BTreeMap<String, Artifact>) -> Result<()>,
    RunFn: Fn(usize, &Stage, &Path, &BTreeMap<String, Artifact>) -> Result<Artifact>,
{
    let mut dependency_artifacts =
        resolve_completed_stage_artifacts(stages, bundle_root, &stage_name, &stage_dir_for)?;
    let mut stage_states = Vec::with_capacity(stages.len());

    for (index, stage) in stages.iter().enumerate() {
        let stage_dir = stage_dir_for(bundle_root, index, stage);
        let stage_name_value = stage_name(stage).to_string();
        let prior_state = load_stage_state::<Artifact>(&stage_dir)?;

        if options.resume_from_last_completed_stage
            && matches!(
                prior_state.as_ref().map(|state| state.status),
                Some(StageStatus::Completed)
            )
        {
            let state = prior_state.expect("completed state");
            dependency_artifacts.insert(stage_name_value.clone(), state.artifact.clone());
            stage_states.push(state);
            if options.stop_after_stage.as_deref() == Some(stage_name_value.as_str()) {
                break;
            }
            continue;
        }

        validate_stage(stage, &dependency_artifacts)?;

        let started_at = unix_timestamp_now();
        let running_state = StageState {
            stage_name: stage_name_value.clone(),
            status: StageStatus::Running,
            started_at_unix_secs: Some(started_at),
            completed_at_unix_secs: None,
            last_error: None,
            artifact: Artifact::default(),
        };
        write_stage_state(&stage_dir, &running_state)?;

        let state = match run_stage(index, stage, &stage_dir, &dependency_artifacts) {
            Ok(artifact) => StageState {
                stage_name: stage_name_value.clone(),
                status: StageStatus::Completed,
                started_at_unix_secs: Some(started_at),
                completed_at_unix_secs: Some(unix_timestamp_now()),
                last_error: None,
                artifact,
            },
            Err(err) => {
                let state = StageState {
                    stage_name: stage_name_value,
                    status: StageStatus::Failed,
                    started_at_unix_secs: Some(started_at),
                    completed_at_unix_secs: Some(unix_timestamp_now()),
                    last_error: Some(err.to_string()),
                    artifact: Artifact::default(),
                };
                write_stage_state(&stage_dir, &state)?;
                stage_states.push(state);
                write_bundle_state(
                    bundle_root,
                    &build_bundle_state(bundle_name.to_string(), bundle_root, stage_states),
                )?;
                return Err(err);
            }
        };

        write_stage_state(&stage_dir, &state)?;
        dependency_artifacts.insert(state.stage_name.clone(), state.artifact.clone());
        stage_states.push(state);
        write_bundle_state(
            bundle_root,
            &build_bundle_state(bundle_name.to_string(), bundle_root, stage_states.clone()),
        )?;

        if options.stop_after_stage.as_deref() == Some(stage_name_value.as_str()) {
            break;
        }
    }

    let bundle_state = build_bundle_state(bundle_name.to_string(), bundle_root, stage_states);
    write_bundle_state(bundle_root, &bundle_state)?;
    Ok(bundle_state)
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::collections::BTreeMap;
    use std::rc::Rc;

    use anyhow::anyhow;
    use serde::{Deserialize, Serialize};
    use tempfile::tempdir;

    use crate::train::pipeline::{
        BundleExecutionOptions, StageState, StageStatus, bundle_state_path, execute_bundle,
        resolve_completed_stage_artifacts, stage_state_path, write_stage_state,
    };

    #[derive(Debug, Clone)]
    struct TestStage {
        name: &'static str,
    }

    #[derive(Debug, Clone, Default, Deserialize, Serialize, PartialEq, Eq)]
    struct TestArtifact {
        value: usize,
    }

    #[test]
    fn resolve_completed_stage_artifacts_reads_only_completed_states() {
        let dir = tempdir().expect("tempdir");
        let bundle_root = dir.path().join("bundle");
        let stages = vec![TestStage { name: "one" }, TestStage { name: "two" }];

        let completed_dir = bundle_root.join("stages/00_one");
        write_stage_state(
            &completed_dir,
            &StageState {
                stage_name: "one".to_string(),
                status: StageStatus::Completed,
                started_at_unix_secs: Some(1),
                completed_at_unix_secs: Some(2),
                last_error: None,
                artifact: TestArtifact { value: 7 },
            },
        )
        .expect("write completed state");
        let failed_dir = bundle_root.join("stages/01_two");
        write_stage_state(
            &failed_dir,
            &StageState {
                stage_name: "two".to_string(),
                status: StageStatus::Failed,
                started_at_unix_secs: Some(3),
                completed_at_unix_secs: Some(4),
                last_error: Some("boom".to_string()),
                artifact: TestArtifact { value: 9 },
            },
        )
        .expect("write failed state");

        let artifacts = resolve_completed_stage_artifacts(
            &stages,
            &bundle_root,
            |stage| stage.name,
            |root, index, stage| {
                root.join("stages")
                    .join(format!("{index:02}_{}", stage.name))
            },
        )
        .expect("resolve completed artifacts");

        assert_eq!(
            artifacts,
            BTreeMap::from([("one".to_string(), TestArtifact { value: 7 })])
        );
    }

    #[test]
    fn execute_bundle_resumes_completed_stage_and_runs_remaining_stage() {
        let dir = tempdir().expect("tempdir");
        let bundle_root = dir.path().join("bundle");
        let stages = vec![TestStage { name: "one" }, TestStage { name: "two" }];
        let first_stage_dir = bundle_root.join("stages/00_one");
        write_stage_state(
            &first_stage_dir,
            &StageState {
                stage_name: "one".to_string(),
                status: StageStatus::Completed,
                started_at_unix_secs: Some(1),
                completed_at_unix_secs: Some(2),
                last_error: None,
                artifact: TestArtifact { value: 5 },
            },
        )
        .expect("write completed state");

        let run_count = Rc::new(Cell::new(0usize));
        let run_count_cloned = Rc::clone(&run_count);
        let bundle_state = execute_bundle(
            "demo",
            &bundle_root,
            &stages,
            &BundleExecutionOptions {
                resume_from_last_completed_stage: true,
                stop_after_stage: None,
            },
            |stage| stage.name,
            |root, index, stage| {
                root.join("stages")
                    .join(format!("{index:02}_{}", stage.name))
            },
            |_stage, _artifacts| Ok(()),
            |_index, stage, stage_dir, artifacts: &BTreeMap<String, TestArtifact>| {
                run_count_cloned.set(run_count_cloned.get() + 1);
                if stage.name != "two" {
                    return Err(anyhow!("unexpected stage executed"));
                }
                let prior = artifacts.get("one").expect("first artifact available");
                assert_eq!(prior.value, 5);
                assert_eq!(
                    stage_state_path(stage_dir),
                    stage_dir.join("stage_state.json")
                );
                Ok(TestArtifact { value: 9 })
            },
        )
        .expect("execute bundle");

        assert_eq!(run_count.get(), 1);
        assert_eq!(bundle_state.latest_completed_stage, Some("two".to_string()));
        assert_eq!(bundle_state.stages.len(), 2);
        assert_eq!(bundle_state.stages[0].artifact, TestArtifact { value: 5 });
        assert_eq!(bundle_state.stages[1].artifact, TestArtifact { value: 9 });
        assert!(bundle_state_path(&bundle_root).is_file());
    }
}
