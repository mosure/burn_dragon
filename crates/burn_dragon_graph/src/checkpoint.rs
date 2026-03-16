use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
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

use crate::{GraphDragon, GraphDragonConfig};

const GRAPH_CONFIG_SNAPSHOT_FILE_NAME: &str = "graph_dragon_config.json";

type ExportBackend = NdArray<f32>;

pub type GraphBurnpackExportReport = CheckpointExportReport;

pub fn write_graph_config_snapshot(config: &GraphDragonConfig, run_dir: &Path) -> Result<()> {
    write_json_snapshot(run_dir, GRAPH_CONFIG_SNAPSHOT_FILE_NAME, config)
}

pub fn load_graph_config_snapshot_from_run_dir(run_dir: &Path) -> Result<GraphDragonConfig> {
    load_json_snapshot(run_dir, GRAPH_CONFIG_SNAPSHOT_FILE_NAME)
}

pub fn load_graph_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: &Path,
) -> Result<GraphDragonConfig> {
    match config_paths.len() {
        0 => {
            if let Some(run_dir) = resolve_checkpoint_run_dir(checkpoint) {
                let snapshot_path = graph_config_snapshot_path(&run_dir);
                if snapshot_path.is_file() {
                    return load_graph_config_snapshot_from_run_dir(&run_dir);
                }
            }
            Err(anyhow!(
                "graph export requires exactly one config file or a run-local {} snapshot",
                GRAPH_CONFIG_SNAPSHOT_FILE_NAME
            ))
        }
        1 => load_graph_config_file(&config_paths[0]),
        _ => Err(anyhow!(
            "graph export expects exactly one config file; received {}",
            config_paths.len()
        )),
    }
}

pub fn export_graph_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<GraphBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let config = load_graph_config_for_checkpoint(config_paths, checkpoint)?;

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let mut model = GraphDragon::<ExportBackend>::new(config, &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<GraphDragon<ExportBackend> as Module<ExportBackend>>::Record>(
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

pub fn graph_config_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, GRAPH_CONFIG_SNAPSHOT_FILE_NAME)
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

fn load_graph_config_file(path: &Path) -> Result<GraphDragonConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read graph config {}", path.display()))?;
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("json") => serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display())),
        Some("toml") => {
            toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))
        }
        _ => Err(anyhow!(
            "graph export config {} must use .json or .toml",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ExportBackend, export_graph_checkpoint_to_burnpack, graph_config_snapshot_path,
        load_graph_config_for_checkpoint, write_graph_config_snapshot,
    };
    use crate::{GraphDragon, GraphDragonConfig};
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_checkpoint::{
        BurnpackBundleExportOptions, BurnpackFloatPrecision, burnpack_parts_manifest_path,
        manifest_is_complete,
    };
    use std::fs;
    use tempfile::tempdir;

    fn test_config() -> GraphDragonConfig {
        GraphDragonConfig {
            embed_dim: 32,
            rank: 8,
            value_dim: 32,
            predict_decay: 0.95,
            mode_embeddings: true,
        }
    }

    #[test]
    fn writes_and_loads_graph_config_snapshot() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("graph-run");
        let config = test_config();

        write_graph_config_snapshot(&config, &run_dir).expect("write graph snapshot");
        let loaded = load_graph_config_for_checkpoint(&[], &run_dir.join("checkpoint"))
            .expect("load graph snapshot");

        assert!(graph_config_snapshot_path(&run_dir).is_file());
        assert_eq!(loaded, config);
    }

    #[test]
    fn exports_graph_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("graph-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let config = test_config();
        write_graph_config_snapshot(&config, &run_dir).expect("write graph snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = GraphDragon::<ExportBackend>::new(config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_graph_checkpoint_to_burnpack(
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
        .expect("export graph burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }
}
