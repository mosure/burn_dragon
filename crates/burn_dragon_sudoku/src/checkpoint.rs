#![cfg(feature = "train")]

use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_checkpoint::{
    BurnpackBundleExportOptions, CheckpointExportReport, export_model_to_burnpack_bundle,
    format_checkpoint_load_error, load_json_snapshot,
    resolve_checkpoint_base as resolve_checkpoint_base_shared,
    resolve_checkpoint_run_dir as resolve_checkpoint_run_dir_shared, run_snapshot_path,
    write_json_snapshot,
};
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};

use crate::config::{
    SudokuDatasetConfig, SudokuModelConfig, SudokuTrainingConfig, SudokuTrainingHyperparameters,
    load_training_config,
};
use crate::model::SudokuSaccadeModel;
use burn_dragon_train::WgpuRuntimeConfig;

const TRAINING_SNAPSHOT_FILE_NAME: &str = "sudoku_training_config.json";
const RUN_CONFIG_FILE_NAME: &str = "config.json";

type ExportBackend = NdArray<f32>;

pub type SudokuBurnpackExportReport = CheckpointExportReport;

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SudokuRunExportSnapshot {
    #[serde(default)]
    run_name: String,
    training: SudokuTrainingHyperparameters,
    model: SudokuModelConfig,
    dataset: SudokuDatasetConfig,
}

pub fn write_training_snapshot(config: &SudokuTrainingConfig, run_dir: &Path) -> Result<()> {
    write_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME, config)
}

pub fn load_training_snapshot_from_run_dir(run_dir: &Path) -> Result<SudokuTrainingConfig> {
    load_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub fn export_sudoku_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<SudokuBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let (model_config, core_wgpu) =
        load_model_export_config_for_checkpoint(config_paths, checkpoint)?;

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let core_config = model_config.to_bdh_config_for_backend("wgpu", &core_wgpu);
    let mut model = SudokuSaccadeModel::<ExportBackend>::new_with_bdh_config(
        &model_config,
        core_config,
        &device,
    );
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<SudokuSaccadeModel<ExportBackend> as Module<ExportBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);

    let bundle = export_model_to_burnpack_bundle(&model, output_base, options)
        .map_err(|err| anyhow!(err))?;

    Ok(CheckpointExportReport {
        checkpoint_base,
        epoch,
        run_dir: resolve_checkpoint_run_dir(checkpoint),
        bundle,
    })
}

pub fn training_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub(crate) fn resolve_checkpoint_run_dir(checkpoint: &Path) -> Option<PathBuf> {
    resolve_checkpoint_run_dir_shared(checkpoint)
}

pub(crate) fn resolve_checkpoint_base(
    path: &Path,
    epoch: Option<usize>,
) -> Result<(PathBuf, usize)> {
    resolve_checkpoint_base_shared(path, epoch)
}

fn load_model_export_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: &Path,
) -> Result<(SudokuModelConfig, WgpuRuntimeConfig)> {
    if !config_paths.is_empty() {
        let config = load_training_config(config_paths)?;
        return Ok((config.model, config.wgpu));
    }

    if let Some(run_dir) = resolve_checkpoint_run_dir(checkpoint) {
        let snapshot_path = training_snapshot_path(&run_dir);
        if snapshot_path.is_file() {
            let config = load_training_snapshot_from_run_dir(&run_dir)?;
            return Ok((config.model, config.wgpu));
        }

        let run_config_path = run_snapshot_path(&run_dir, RUN_CONFIG_FILE_NAME);
        if run_config_path.is_file() {
            let snapshot: SudokuRunExportSnapshot =
                load_json_snapshot(&run_dir, RUN_CONFIG_FILE_NAME)?;
            return Ok((snapshot.model, WgpuRuntimeConfig::default()));
        }
    }

    Err(anyhow!(
        "sudoku export requires explicit config overlays, a run-local {} snapshot, or {}",
        TRAINING_SNAPSHOT_FILE_NAME,
        RUN_CONFIG_FILE_NAME
    ))
}

#[cfg(test)]
mod tests {
    use super::{
        ExportBackend, export_sudoku_checkpoint_to_burnpack, load_training_snapshot_from_run_dir,
        training_snapshot_path, write_training_snapshot,
    };
    use crate::config::load_training_config;
    use crate::model::SudokuSaccadeModel;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_checkpoint::{
        BurnpackBundleExportOptions, BurnpackFloatPrecision, burnpack_parts_manifest_path,
        manifest_is_complete,
    };
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    fn load_tiny_config() -> crate::config::SudokuTrainingConfig {
        let config_path =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../config/sudoku/saccade/tiny.toml");
        load_training_config(&[config_path]).expect("load tiny sudoku config")
    }

    #[test]
    fn writes_and_loads_sudoku_training_snapshot() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("sudoku-run");
        let config = load_tiny_config();

        write_training_snapshot(&config, &run_dir).expect("write sudoku snapshot");
        let loaded = load_training_snapshot_from_run_dir(&run_dir).expect("load sudoku snapshot");

        assert!(training_snapshot_path(&run_dir).is_file());
        assert_eq!(loaded.model, config.model);
        assert_eq!(loaded.training, config.training);
    }

    #[test]
    fn exports_sudoku_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("sudoku-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let config = load_tiny_config();
        write_training_snapshot(&config, &run_dir).expect("write sudoku snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = SudokuSaccadeModel::<ExportBackend>::new(&config.model, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_sudoku_checkpoint_to_burnpack(
            &checkpoint_dir,
            Some(0),
            &[],
            &run_dir.join("deploy/model"),
            &BurnpackBundleExportOptions {
                precision: BurnpackFloatPrecision::F16,
                max_part_size_mib: Some(1),
                overwrite_parts: true,
                ..BurnpackBundleExportOptions::default()
            },
        )
        .expect("export sudoku burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }
}
