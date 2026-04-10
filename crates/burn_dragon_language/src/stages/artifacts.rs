use std::path::Path;

use anyhow::{Context, Result, anyhow};
use burn_dragon_checkpoint::resolve_checkpoint_base;
use burn_dragon_train::train::pipeline::resolve_latest_run_dir_in;

use super::{ExperimentStageArtifact, resolved_stage_config_path};

pub fn resolve_training_stage_artifact(stage_dir: &Path) -> Result<ExperimentStageArtifact> {
    let run_root = stage_dir.join("runs");
    let effective_run_root = if resolve_latest_run_dir_in(&run_root).is_some() {
        run_root.clone()
    } else {
        let language_run_root = run_root.join("language");
        if resolve_latest_run_dir_in(&language_run_root).is_some() {
            language_run_root
        } else {
            return Err(anyhow!(
                "no latest run available under {}",
                run_root.display()
            ));
        }
    };
    let latest_run_dir = resolve_latest_run_dir_in(&effective_run_root).ok_or_else(|| {
        anyhow!(
            "no latest run available under {}",
            effective_run_root.display()
        )
    })?;
    let checkpoint_dir = latest_run_dir.join("checkpoint");
    let (_, epoch) = resolve_checkpoint_base(&checkpoint_dir, None).with_context(|| {
        format!(
            "failed to resolve latest checkpoint in {}",
            checkpoint_dir.display()
        )
    })?;
    Ok(ExperimentStageArtifact {
        run_root: Some(effective_run_root),
        latest_run_dir: Some(latest_run_dir),
        latest_checkpoint_dir: Some(checkpoint_dir),
        latest_checkpoint_epoch: Some(epoch),
        resolved_config_path: Some(resolved_stage_config_path(stage_dir)),
        ..ExperimentStageArtifact::default()
    })
}
