use crate::{
    MovingMnistDreamerRunSummary, MovingMnistDreamerTrainConfig, MovingMnistTokenizerRunSummary,
    train_moving_mnist, train_moving_mnist_tokenizer,
};
use anyhow::{Context, Result};
use burn_dragon_train::train::pipeline::resolve_named_stage_dir;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MovingMnistDreamerBundleConfig {
    pub name: String,
    pub run_root: Option<PathBuf>,
    pub tokenizer: MovingMnistDreamerTrainConfig,
    pub dynamics: MovingMnistDreamerTrainConfig,
}

impl Default for MovingMnistDreamerBundleConfig {
    fn default() -> Self {
        let tokenizer = MovingMnistDreamerTrainConfig::default();
        let dynamics = MovingMnistDreamerTrainConfig::default();
        Self {
            name: "moving_mnist_bundle".to_string(),
            run_root: Some(PathBuf::from("runs/burn_dragon_dreamer/bundles")),
            tokenizer,
            dynamics,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MovingMnistDreamerBundleRunSummary {
    pub name: String,
    pub tokenizer_checkpoint: PathBuf,
    pub tokenizer: MovingMnistTokenizerRunSummary,
    pub dynamics: MovingMnistDreamerRunSummary,
}

fn apply_bundle_stage_run_roots(bundle: &mut MovingMnistDreamerBundleConfig) {
    if let Some(root) = bundle.run_root.clone() {
        let bundle_root = root.join(&bundle.name);
        if bundle.tokenizer.run_root.is_none() {
            bundle.tokenizer.run_root = Some(resolve_named_stage_dir(&bundle_root, 0, "tokenizer"));
        }
        if bundle.dynamics.run_root.is_none() {
            bundle.dynamics.run_root = Some(resolve_named_stage_dir(&bundle_root, 1, "dynamics"));
        }
    }
}

pub fn train_moving_mnist_bundle(
    mut bundle: MovingMnistDreamerBundleConfig,
) -> Result<MovingMnistDreamerBundleRunSummary> {
    apply_bundle_stage_run_roots(&mut bundle);

    let tokenizer_summary = train_moving_mnist_tokenizer(bundle.tokenizer.clone())?;
    let tokenizer_checkpoint = tokenizer_summary
        .checkpoint_base
        .clone()
        .context("tokenizer stage did not produce a checkpoint")?;

    bundle.dynamics.tokenizer_checkpoint = Some(tokenizer_checkpoint.clone());
    if !bundle.dynamics.freeze_tokenizer {
        bundle.dynamics.freeze_tokenizer = true;
    }
    let dynamics_summary = train_moving_mnist(bundle.dynamics)?;

    Ok(MovingMnistDreamerBundleRunSummary {
        name: bundle.name,
        tokenizer_checkpoint,
        tokenizer: tokenizer_summary,
        dynamics: dynamics_summary,
    })
}

#[cfg(test)]
mod tests {
    use super::{MovingMnistDreamerBundleConfig, apply_bundle_stage_run_roots};
    use std::path::PathBuf;

    #[test]
    fn bundle_stage_roots_use_shared_stage_layout() {
        let mut bundle = MovingMnistDreamerBundleConfig {
            name: "demo".to_string(),
            run_root: Some(PathBuf::from("runs/burn_dragon_dreamer/bundles")),
            ..MovingMnistDreamerBundleConfig::default()
        };

        bundle.tokenizer.run_root = None;
        bundle.dynamics.run_root = None;
        apply_bundle_stage_run_roots(&mut bundle);

        assert_eq!(
            bundle.tokenizer.run_root,
            Some(PathBuf::from(
                "runs/burn_dragon_dreamer/bundles/demo/stages/00_tokenizer"
            ))
        );
        assert_eq!(
            bundle.dynamics.run_root,
            Some(PathBuf::from(
                "runs/burn_dragon_dreamer/bundles/demo/stages/01_dynamics"
            ))
        );
    }
}
