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

use crate::config::VlJepaDragonConfig;
use crate::config_io::load_merged_config;
use crate::model::VlJepaDragon;

const MULTIMODAL_CONFIG_SNAPSHOT_FILE_NAME: &str = "multimodal_vl_jepa_config.json";

type ExportBackend = NdArray<f32>;

pub type MultimodalBurnpackExportReport = CheckpointExportReport;
pub type MultimodalRunConfigSnapshot = VlJepaDragonConfig;

pub fn default_checkpoint_dir(run_dir: impl AsRef<Path>) -> PathBuf {
    run_dir.as_ref().join("checkpoint")
}

pub fn write_training_snapshot(
    run_dir: impl AsRef<Path>,
    config: &VlJepaDragonConfig,
) -> Result<PathBuf> {
    let run_dir = run_dir.as_ref();
    write_json_snapshot(run_dir, MULTIMODAL_CONFIG_SNAPSHOT_FILE_NAME, config)?;
    Ok(training_snapshot_path(run_dir))
}

pub fn load_training_snapshot_from_run_dir(run_dir: &Path) -> Result<VlJepaDragonConfig> {
    load_json_snapshot(run_dir, MULTIMODAL_CONFIG_SNAPSHOT_FILE_NAME)
}

pub fn load_training_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: &Path,
) -> Result<VlJepaDragonConfig> {
    if !config_paths.is_empty() {
        return load_multimodal_training_config(config_paths);
    }

    if let Some(run_dir) = resolve_checkpoint_run_dir(checkpoint) {
        let snapshot_path = training_snapshot_path(&run_dir);
        if snapshot_path.is_file() {
            return load_training_snapshot_from_run_dir(&run_dir);
        }
    }

    Err(anyhow!(
        "multimodal export requires explicit config overlays or a run-local {} snapshot",
        MULTIMODAL_CONFIG_SNAPSHOT_FILE_NAME
    ))
}

pub fn export_multimodal_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<MultimodalBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let config = load_training_config_for_checkpoint(config_paths, checkpoint)?;

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let mut model = VlJepaDragon::<ExportBackend>::new(config, &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<VlJepaDragon<ExportBackend> as Module<ExportBackend>>::Record>(
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
    run_snapshot_path(run_dir, MULTIMODAL_CONFIG_SNAPSHOT_FILE_NAME)
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

fn load_multimodal_training_config(config_paths: &[PathBuf]) -> Result<VlJepaDragonConfig> {
    load_merged_config(config_paths)
}

#[cfg(test)]
mod tests {
    use super::{
        ExportBackend, export_multimodal_checkpoint_to_burnpack,
        load_training_config_for_checkpoint, training_snapshot_path, write_training_snapshot,
    };
    use crate::config::VlJepaDragonConfig;
    use crate::model::VlJepaDragon;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_checkpoint::{
        BurnpackBundleExportOptions, BurnpackFloatPrecision, burnpack_parts_manifest_path,
        manifest_is_complete,
    };
    use std::fs;
    use tempfile::tempdir;

    fn test_config() -> VlJepaDragonConfig {
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.projection_hidden_dim = 32;
        config.vision.steps = 1;
        config.vision.patch_size = 4;
        config.vision.in_channels = 3;
        config.query_text.n_layer = 2;
        config.query_text.n_embd = 32;
        config.query_text.n_head = 4;
        config.target_text.n_layer = 2;
        config.target_text.n_embd = 32;
        config.target_text.n_head = 4;
        config.fusion.n_layer = 2;
        config.fusion.n_embd = 32;
        config.fusion.n_head = 4;
        config.fusion_dim = 32;
        config.target_dim = 32;
        config
    }

    #[test]
    fn writes_and_loads_multimodal_training_snapshot() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("multimodal-run");
        let config = test_config();

        write_training_snapshot(&run_dir, &config).expect("write multimodal snapshot");
        let loaded = load_training_config_for_checkpoint(&[], &run_dir.join("checkpoint"))
            .expect("load snapshot");

        assert!(training_snapshot_path(&run_dir).is_file());
        assert_eq!(loaded, config);
    }

    #[test]
    fn merges_multimodal_config_overlays_in_order() {
        let dir = tempdir().expect("tempdir");
        let base_path = dir.path().join("base.toml");
        let override_path = dir.path().join("override.toml");
        fs::write(
            &base_path,
            r#"
fusion_dim = 32
target_dim = 32
vision_rollout_steps = 2
vision_backprop_steps = 1
temperature = 0.07

[fusion_slots]
slot_count = 2
use_modality_type_embeddings = true

[tbptt]
unroll_steps = 4
backprop_steps = 2
state_carry_policy = "until_boundary"
fusion_carry_policy = "until_boundary"
target_alignment_policy = "same_step"

[vision]
embed_dim = 32
projection_dim = 32
projection_hidden_dim = 32
patch_size = 4
steps = 1

[query_text]
n_layer = 2
n_embd = 32
n_head = 4
vocab_size = 256

[target_text]
n_layer = 2
n_embd = 32
n_head = 4
vocab_size = 256

[fusion]
n_layer = 2
n_embd = 32
n_head = 4
vocab_size = 1
"#,
        )
        .expect("write base");
        fs::write(
            &override_path,
            r#"
temperature = 0.11

[fusion_slots]
slot_count = 4
"#,
        )
        .expect("write override");

        let loaded = load_training_config_for_checkpoint(&[base_path, override_path], dir.path())
            .expect("load merged overlays");

        assert_eq!(loaded.temperature, 0.11);
        assert_eq!(loaded.fusion_slots.slot_count, 4);
        assert_eq!(loaded.vision.embed_dim, 32);
    }

    #[test]
    fn exports_multimodal_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("multimodal-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let config = test_config();
        write_training_snapshot(&run_dir, &config).expect("write multimodal snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = VlJepaDragon::<ExportBackend>::new(config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_multimodal_checkpoint_to_burnpack(
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
        .expect("export multimodal burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }
}
