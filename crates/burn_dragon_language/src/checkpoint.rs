#![cfg(feature = "train")]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_checkpoint::{
    BurnpackBundleExportOptions, BurnpackBundleExportReport,
    export_model_to_burnpack_bundle, format_checkpoint_load_error, load_json_snapshot,
    resolve_checkpoint_base as resolve_checkpoint_base_shared,
    resolve_checkpoint_run_dir as resolve_checkpoint_run_dir_shared, run_snapshot_path,
    write_json_snapshot,
};
use burn_ndarray::NdArray;
use serde::{Deserialize, Serialize};

use crate::config::load_training_config;
use crate::tokenizer::Tokenizer;
use crate::{BDH, ModelOverrides, TrainingConfig, build_model_config};

const RUN_CONFIG_FILE_NAME: &str = "config.json";
const TRAINING_SNAPSHOT_FILE_NAME: &str = "training_config.json";
const TOKENIZER_SNAPSHOT_FILE_NAME: &str = "tokenizer.json";

type ExportBackend = NdArray<f32>;

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct LanguageRunConfigSnapshot {
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub overrides: ModelOverrides,
}

#[derive(Debug, Clone)]
pub struct LanguageBurnpackExportReport {
    pub checkpoint_base: PathBuf,
    pub epoch: usize,
    pub vocab_size: usize,
    pub run_dir: Option<PathBuf>,
    pub bundle: BurnpackBundleExportReport,
}

pub fn write_training_snapshot(
    config: &TrainingConfig,
    run_dir: &Path,
    tokenizer: &dyn Tokenizer,
) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let mut snapshot = config.clone();
    if snapshot
        .dataset
        .tokenizer
        .storage_path(Path::new("."))
        .is_some()
    {
        let tokenizer_path = tokenizer_snapshot_path(run_dir);
        snapshot
            .dataset
            .tokenizer
            .save(tokenizer, &tokenizer_path)
            .with_context(|| {
                format!(
                    "failed to save tokenizer snapshot {}",
                    tokenizer_path.display()
                )
            })?;
        snapshot.dataset.cache_dir = PathBuf::from(".");
        snapshot.dataset.tokenizer.vocab_path = Some(PathBuf::from(TOKENIZER_SNAPSHOT_FILE_NAME));
    }

    write_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME, &snapshot)
}

pub fn load_training_snapshot_from_run_dir(run_dir: &Path) -> Result<TrainingConfig> {
    let mut config: TrainingConfig = load_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME)?;
    apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
    absolutize_snapshot_cache_dir(&mut config, run_dir);
    Ok(config)
}

pub fn load_training_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Result<TrainingConfig> {
    let run_dir = resolve_checkpoint_run_dir(checkpoint, backend_name);

    if !config_paths.is_empty() {
        let mut config = load_training_config(config_paths)?;
        if let Some(run_dir) = run_dir.as_deref() {
            apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
        }
        return Ok(config);
    }

    if let Some(run_dir) = run_dir.as_deref() {
        let snapshot_path = training_snapshot_path(run_dir);
        if snapshot_path.is_file() {
            return load_training_snapshot_from_run_dir(run_dir);
        }
    }

    let mut config = load_training_config(&[PathBuf::from("config/language/base.toml")])?;
    if let Some(path) = resolve_run_config_path(checkpoint, backend_name) {
        let contents = fs::read_to_string(&path)
            .with_context(|| format!("failed to read run config {}", path.display()))?;
        let run_config: LanguageRunConfigSnapshot = serde_json::from_str(&contents)
            .with_context(|| format!("failed to parse {}", path.display()))?;
        apply_run_config(&mut config, &run_config);
    }
    if let Some(run_dir) = run_dir.as_deref() {
        apply_run_dir_tokenizer_snapshot(&mut config, run_dir);
    }
    Ok(config)
}

pub fn export_language_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<LanguageBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let checkpoint_path = checkpoint.to_path_buf();
    let config =
        load_training_config_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;

    let tokenizer_path = config
        .dataset
        .tokenizer
        .storage_path(&config.dataset.cache_dir);
    let tokenizer = if let Some(path) = tokenizer_path {
        config
            .dataset
            .tokenizer
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display()))?
    } else {
        config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer")?
    };

    let mut model_config = build_model_config(&config.model, config.training.block_size);
    model_config.vocab_size = tokenizer.len();

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let mut model = BDH::<ExportBackend>::new(model_config, &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<ExportBackend> as Module<ExportBackend>>::Record>(checkpoint_base.clone(), &device)
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);

    let bundle = export_model_to_burnpack_bundle(&model, output_base, options)
        .map_err(|err| anyhow!(err))?;

    Ok(LanguageBurnpackExportReport {
        checkpoint_base,
        epoch,
        vocab_size: tokenizer.len(),
        run_dir: resolve_checkpoint_run_dir(Some(&checkpoint_path), backend_name),
        bundle,
    })
}

pub fn apply_run_config(config: &mut TrainingConfig, run_config: &LanguageRunConfigSnapshot) {
    let block_override = run_config
        .block_size
        .or(run_config.overrides.block_size)
        .map(|value| value.max(1));
    if let Some(block_size) = block_override {
        config.training.block_size = block_size;
    }
    merge_model_overrides(&mut config.model, &run_config.overrides);
}

pub fn merge_model_overrides(base: &mut ModelOverrides, incoming: &ModelOverrides) {
    if let Some(value) = incoming.n_layer {
        base.n_layer = Some(value);
    }
    if let Some(value) = incoming.n_embd {
        base.n_embd = Some(value);
    }
    if let Some(value) = incoming.n_head {
        base.n_head = Some(value);
    }
    if let Some(value) = incoming.mlp_internal_dim_multiplier {
        base.mlp_internal_dim_multiplier = Some(value);
    }
    if let Some(value) = incoming.relu_threshold {
        base.relu_threshold = Some(value);
    }
    if let Some(value) = incoming.dropout {
        base.dropout = Some(value);
    }
    if let Some(value) = incoming.fused_kernels {
        base.fused_kernels = Some(value);
    }
    if let Some(value) = incoming.block_size {
        base.block_size = Some(value);
    }
    if let Some(value) = incoming.rollout_fast_steps_per_slow_step {
        base.rollout_fast_steps_per_slow_step = Some(value);
    }
    if let Some(value) = incoming.rotary_embedding {
        base.rotary_embedding = Some(value);
    }
    if let Some(value) = &incoming.y_neuron_recurrence {
        base.y_neuron_recurrence = Some(value.clone());
    }
    if let Some(value) = &incoming.mhc {
        base.mhc = Some(value.clone());
    }
}

pub fn resolve_run_config_path(
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Option<PathBuf> {
    resolve_checkpoint_run_dir(checkpoint, backend_name).and_then(|run_dir| {
        let path = run_dir.join(RUN_CONFIG_FILE_NAME);
        path.is_file().then_some(path)
    })
}

pub(crate) fn resolve_checkpoint_run_dir(
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Option<PathBuf> {
    let checkpoint_path = checkpoint
        .cloned()
        .unwrap_or_else(|| default_checkpoint_dir(backend_name));
    resolve_checkpoint_run_dir_shared(&checkpoint_path)
}

pub fn default_checkpoint_dir(backend_name: &str) -> PathBuf {
    resolve_latest_run_dir(backend_name)
        .map(|dir| dir.join("checkpoint"))
        .unwrap_or_else(|| PathBuf::from("runs").join("checkpoint"))
}

pub fn resolve_latest_run_dir(backend_name: &str) -> Option<PathBuf> {
    let run_root = PathBuf::from("runs");
    resolve_latest_run_dir_from(&run_root).or_else(|| {
        let device_root = run_root.join(backend_name);
        resolve_latest_run_dir_from(&device_root)
    })
}

pub fn training_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub fn tokenizer_snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TOKENIZER_SNAPSHOT_FILE_NAME)
}

pub(crate) fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
    resolve_checkpoint_base_shared(path, epoch)
}

fn apply_run_dir_tokenizer_snapshot(config: &mut TrainingConfig, run_dir: &Path) {
    let tokenizer_path = tokenizer_snapshot_path(run_dir);
    if tokenizer_path.is_file() {
        config.dataset.cache_dir = run_dir.to_path_buf();
        config.dataset.tokenizer.vocab_path = Some(PathBuf::from(TOKENIZER_SNAPSHOT_FILE_NAME));
    }
}

fn absolutize_snapshot_cache_dir(config: &mut TrainingConfig, run_dir: &Path) {
    if !config.dataset.cache_dir.is_absolute() {
        config.dataset.cache_dir = run_dir.join(&config.dataset.cache_dir);
    }
}

fn resolve_latest_run_dir_from(run_root: &Path) -> Option<PathBuf> {
    let latest_path = run_root.join("latest");
    let contents = fs::read_to_string(&latest_path).ok()?;
    let name = contents.trim();
    if name.is_empty() {
        return None;
    }
    Some(run_root.join(name))
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackBundleExportOptions, ExportBackend, export_language_checkpoint_to_burnpack,
        load_training_config_for_checkpoint, resolve_checkpoint_base,
        tokenizer_snapshot_path, training_snapshot_path, write_training_snapshot,
    };
    use crate::config::{
        ContextStrategyConfig, DatasetConfig, DatasetSourceConfig, GenerationConfig,
        ModelOverrides, TrainingConfig, TrainingHyperparameters,
    };
    use crate::tokenizer::TokenizerConfig;
    use crate::BDH;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_checkpoint::{
        BurnpackFloatPrecision, burnpack_parts_manifest_path, manifest_is_complete,
    };
    use burn_dragon_train::{OptimizerConfig, WgpuRuntimeConfig};
    use std::fs;
    use std::path::{Path, PathBuf};
    use tempfile::tempdir;

    #[test]
    fn writes_and_loads_training_snapshot_with_tokenizer_copy() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let config = test_config(dir.path().join("cache"));
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["To be"].into_iter())
            .expect("fit tokenizer");

        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let loaded = load_training_config_for_checkpoint(
            &[],
            Some(&run_dir.join("checkpoint")),
            "wgpu",
        )
        .expect("load checkpoint config");

        assert!(training_snapshot_path(&run_dir).is_file());
        assert!(tokenizer_snapshot_path(&run_dir).is_file());
        assert_eq!(loaded.dataset.cache_dir, run_dir);
        assert_eq!(
            loaded.dataset.tokenizer.vocab_path.as_deref(),
            Some(Path::new("tokenizer.json"))
        );
    }

    #[test]
    fn exports_language_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let config = test_config(dir.path().join("cache"));
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");

        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let mut model_config = crate::build_model_config(&config.model, config.training.block_size);
        model_config.vocab_size = tokenizer.len();
        let model = BDH::<ExportBackend>::new(model_config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let report = export_language_checkpoint_to_burnpack(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &run_dir.join("deploy/model"),
            &BurnpackBundleExportOptions {
                precision: BurnpackFloatPrecision::F16,
                max_part_size_mib: Some(1),
                overwrite_parts: true,
                ..BurnpackBundleExportOptions::default()
            },
        )
        .expect("export burnpack");

        assert_eq!(
            resolve_checkpoint_base(&checkpoint_dir, Some(0))
                .expect("resolve checkpoint")
                .1,
            0
        );
        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
        assert_eq!(report.vocab_size, tokenizer.len());
    }

    fn test_config(cache_dir: PathBuf) -> TrainingConfig {
        TrainingConfig {
            dataset: DatasetConfig {
                cache_dir,
                train_split_ratio: 0.9,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 8,
                batch_size: 2,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: Some(1),
                max_iters: 1,
                log_frequency: 1,
                fast_train: true,
                context_strategy: ContextStrategyConfig::Infinite,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                learning_rate: 1e-3,
                weight_decay: 0.0,
                lr_schedule: None,
                grad_clip_norm: None,
                grad_clip_value: None,
            },
            generation: GenerationConfig {
                prompt: "To be".to_string(),
                max_tokens: Some(4),
                temperature: 1.0,
                top_k: Some(4),
                context_strategy: ContextStrategyConfig::Infinite,
            },
            wgpu: WgpuRuntimeConfig::default(),
            model: ModelOverrides::default(),
        }
    }
}
