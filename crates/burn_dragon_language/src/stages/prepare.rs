use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use burn_dragon_train::train::pipeline::resolve_resume_run_dir;
use burn_dragon_universality::{NcaCorpusConfig, load_nca_config};

use crate::config::{DatasetSourceConfig, TrainingConfig, load_training_config};
use crate::tokenizer::TokenizerKind;

use super::{ExperimentStageArtifact, ExperimentStageConfig, ExperimentStageKind};

pub fn prepare_universality_stage_config(
    bundle_config_path: &Path,
    stage_dir: &Path,
    source_config_path: &Path,
) -> Result<NcaCorpusConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let mut config = load_nca_config(&source_path)?;
    config.output_dir = stage_dir.join("output");
    Ok(config)
}

pub fn prepare_language_stage_config(
    bundle_config_path: &Path,
    source_config_path: &Path,
    stage_dir: &Path,
    stage: &ExperimentStageConfig,
    dependency_artifacts: &BTreeMap<String, ExperimentStageArtifact>,
) -> Result<TrainingConfig> {
    let source_path = resolve_relative_to(bundle_config_path, source_config_path);
    let mut config = load_training_config(&[source_path])?;
    config.run_layout.base_dir = Some(stage_dir.join("runs"));
    config.run_layout.category = None;
    config.run_layout.mirror_config_path = false;
    let stage_launch_mode = match &stage.kind {
        ExperimentStageKind::LanguageTrain { launch_mode, .. } => *launch_mode,
        ExperimentStageKind::UniversalityGenerate { .. } => unreachable!(),
    };
    config.training.launch_mode = stage_launch_mode;

    if let ExperimentStageKind::LanguageTrain {
        dataset_manifest_from_stage,
        init_checkpoint_from_stage,
        ..
    } = &stage.kind
    {
        if let Some(stage_name) = dataset_manifest_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let manifest_path = artifact
                .manifest_path
                .clone()
                .ok_or_else(|| anyhow!("stage `{stage_name}` did not produce a manifest_path"))?;
            config.dataset.cache_dir = artifact.corpus_output_dir.clone().unwrap_or_else(|| {
                manifest_path
                    .parent()
                    .unwrap_or(Path::new("."))
                    .to_path_buf()
            });
            config.dataset.source = DatasetSourceConfig::UniversalityManifest {
                manifest: manifest_path,
            };
            if !matches!(
                config.dataset.tokenizer.kind,
                TokenizerKind::Pretokenized(_)
            ) {
                return Err(anyhow!(
                    "language stage `{}` requires tokenizer.type = `pretokenized` when sourcing a universality manifest",
                    stage.name
                ));
            }
        }
        if let Some(stage_name) = init_checkpoint_from_stage {
            let artifact = dependency_artifacts.get(stage_name).ok_or_else(|| {
                anyhow!("stage `{}` has no completed artifact available", stage_name)
            })?;
            let checkpoint_dir = artifact.latest_checkpoint_dir.clone().ok_or_else(|| {
                anyhow!("stage `{stage_name}` did not produce a latest_checkpoint_dir")
            })?;
            config.training.init_checkpoint_path = Some(checkpoint_dir);
            config.training.init_checkpoint_epoch = artifact.latest_checkpoint_epoch;
            config.training.resume_run_dir = None;
            config.training.resume_checkpoint_epoch = None;
        }
    }

    let stage_run_root = config
        .run_layout
        .base_dir
        .clone()
        .unwrap_or_else(|| stage_dir.join("runs"));
    config.training.resume_run_dir = resolve_resume_run_dir(
        &stage_run_root,
        config.training.resume_run_dir.as_deref(),
        stage_launch_mode,
    )?;
    if config.training.resume_run_dir.is_some() {
        config.training.resume_checkpoint_epoch = None;
        config.training.init_checkpoint_path = None;
        config.training.init_checkpoint_epoch = None;
    }

    Ok(config)
}

fn resolve_relative_to(bundle_config_path: &Path, relative_or_absolute: &Path) -> PathBuf {
    if relative_or_absolute.is_absolute() {
        relative_or_absolute.to_path_buf()
    } else {
        bundle_config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join(relative_or_absolute)
    }
}
