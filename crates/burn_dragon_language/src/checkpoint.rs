#![cfg(feature = "train")]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_checkpoint::{
    BurnpackBundleExportOptions, BurnpackBundleExportReport, BurnpackFloatPrecision,
    BurnpackLoadPolicy, BurnpackPrecisionPreference, apply_burnpack_part_bytes,
    convert_burnpack_precision, export_model_to_burnpack_bundle, format_checkpoint_load_error,
    load_json_snapshot, resolve_checkpoint_base as resolve_checkpoint_base_shared,
    resolve_checkpoint_run_dir as resolve_checkpoint_run_dir_shared, run_snapshot_path,
    write_json_snapshot,
};
use burn_dragon_train::train::metrics::MetricsSinkSpec;
use burn_dragon_train::train::pipeline::resolve_latest_run_dir_in as resolve_latest_run_dir_shared;
use burn_dragon_train::{KernelSpec, ModelSpec, OptimizerSpec, ParallelSpec, StateLayout};
use burn_ndarray::NdArray;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::io::{Read, Write};

use crate::bitnet_artifact::{
    BITNET_ARTIFACT_BINARY_MAGIC, LanguageBitNetArtifactBundle, deserialize_bitnet_artifact_binary,
    serialize_bitnet_artifact_binary,
};
use crate::config::load_training_config;
use crate::tokenizer::{SharedTokenizer, Tokenizer};
use crate::{BDH, ModelOverrides, TrainingConfig, build_model_config_with_tokenizer};

const RUN_CONFIG_FILE_NAME: &str = "config.json";
const TRAINING_SNAPSHOT_FILE_NAME: &str = "training_config.json";
const TOKENIZER_SNAPSHOT_FILE_NAME: &str = "tokenizer.json";
pub const RUN_ROOT_ENV: &str = "BURN_DRAGON_RUN_ROOT";
pub const RUN_DIR_ENV: &str = "BURN_DRAGON_RUN_DIR";
pub const RUN_NAME_ENV: &str = "BURN_DRAGON_RUN_NAME";

type ExportBackend = NdArray<f32>;

#[derive(Debug, Clone, Deserialize, Serialize, Default, PartialEq)]
pub struct LanguageRunConfigSnapshot {
    #[serde(default)]
    pub block_size: Option<usize>,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub training_execution_form: Option<String>,
    #[serde(default)]
    pub training_launch_mode_requested:
        Option<burn_dragon_train::train::pipeline::TrainingLaunchMode>,
    #[serde(default)]
    pub training_sequence_kernel_override: Option<burn_dragon_core::SequenceKernelConfig>,
    #[serde(default)]
    pub arch_version: Option<String>,
    #[serde(default)]
    pub shard_layout_version: Option<u32>,
    #[serde(default)]
    pub overrides: ModelOverrides,
    #[serde(default)]
    pub model_spec: Option<ModelSpec>,
    #[serde(default)]
    pub optimizer_spec: Option<OptimizerSpec>,
    #[serde(default)]
    pub parallel_spec: Option<ParallelSpec>,
    #[serde(default)]
    pub kernel_spec: Option<KernelSpec>,
    #[serde(default)]
    pub state_layout: Option<StateLayout>,
    #[serde(default)]
    pub metrics_sink: Option<MetricsSinkSpec>,
}

#[derive(Debug, Clone)]
pub struct LanguageBurnpackExportReport {
    pub checkpoint_base: PathBuf,
    pub epoch: usize,
    pub vocab_size: usize,
    pub run_dir: Option<PathBuf>,
    pub bundle: BurnpackBundleExportReport,
}

#[derive(Debug, Clone)]
pub struct LanguageBitNetArtifactExportReport {
    pub checkpoint_base: PathBuf,
    pub epoch: usize,
    pub run_dir: Option<PathBuf>,
    pub artifact_path: PathBuf,
    pub bundle: LanguageBitNetArtifactBundle,
}

#[derive(Debug, Clone)]
struct BurnpackApplySummary {
    applied: Vec<String>,
    unused: Vec<String>,
    error_count: usize,
}

fn bitnet_artifact_path_is_supported(path: &Path) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .is_some_and(|name| {
            name.ends_with(".bitnet_artifact.bin") || name.ends_with(".bitnet_artifact.bin.gz")
        })
}

fn ensure_supported_bitnet_artifact_path(path: &Path) -> Result<()> {
    if bitnet_artifact_path_is_supported(path) {
        return Ok(());
    }
    Err(anyhow!(
        "unsupported BitNet artifact path {}; expected a `.bitnet_artifact.bin` or `.bitnet_artifact.bin.gz` file",
        path.display()
    ))
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
        let source_tokenizer_path = snapshot
            .dataset
            .tokenizer
            .storage_path(&snapshot.dataset.cache_dir);
        if let Err(error) = snapshot.dataset.tokenizer.save(tokenizer, &tokenizer_path) {
            let copied = source_tokenizer_path
                .as_ref()
                .filter(|path| path.is_file())
                .and_then(|source_path| {
                    if fs::copy(source_path, &tokenizer_path).is_ok() {
                        Some(())
                    } else {
                        None
                    }
                })
                .is_some();
            if !copied {
                return Err(error).with_context(|| {
                    format!(
                        "failed to save tokenizer snapshot {}",
                        tokenizer_path.display()
                    )
                });
            }
        }
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

fn export_bitnet_deploy_base_burnpack_bytes(model: &BDH<ExportBackend>) -> Result<Vec<u8>> {
    let temp_dir = tempfile::tempdir().context("create temp dir for BitNet deploy scaffold")?;
    let burnpack_base = temp_dir.path().join("bitnet_deploy_base");
    let scaffold = model.export_bitnet_deploy_scaffold();
    let report = export_model_to_burnpack_bundle(
        &scaffold,
        &burnpack_base,
        &BurnpackBundleExportOptions {
            precision: BurnpackFloatPrecision::F16,
            max_part_size_mib: None,
            overwrite_parts: false,
            keep_intermediate_f32: false,
            ..BurnpackBundleExportOptions::default()
        },
    )
    .map_err(|err| anyhow!("failed to serialize BitNet deploy scaffold burnpack: {err}"))?;
    fs::read(&report.burnpack_path)
        .with_context(|| format!("failed to read {}", report.burnpack_path.display()))
}

fn convert_burnpack_bytes_precision(
    burnpack_bytes: &[u8],
    precision: BurnpackFloatPrecision,
) -> Result<Vec<u8>> {
    let temp_dir = tempfile::tempdir().context("create temp dir for burnpack conversion")?;
    let source = temp_dir.path().join("source_f16.bpk");
    fs::write(&source, burnpack_bytes)
        .with_context(|| format!("failed to write {}", source.display()))?;
    let converted = convert_burnpack_precision(
        &source,
        &temp_dir.path().join("converted"),
        precision,
        BurnpackLoadPolicy::default().with_precision(BurnpackPrecisionPreference::PreferF32),
    )
    .map_err(|err| anyhow!("failed to convert burnpack precision: {err}"))?;
    fs::read(&converted).with_context(|| format!("failed to read {}", converted.display()))
}

fn backend_requires_f32_burnpack_apply<B: BackendTrait>() -> bool {
    let backend = std::any::type_name::<B>();
    backend.contains("burn_ndarray::NdArray") || backend.contains("burn_ndarray")
}

fn apply_deploy_base_burnpack_bytes_to_model<B: BackendTrait>(
    model: &mut BDH<B>,
    deploy_base_burnpack: Vec<u8>,
) -> Result<BurnpackApplySummary> {
    let deploy_base_burnpack = if backend_requires_f32_burnpack_apply::<B>() {
        convert_burnpack_bytes_precision(&deploy_base_burnpack, BurnpackFloatPrecision::F32)
            .context("convert BitNet deploy scaffold burnpack to backend-compatible f32")?
    } else {
        deploy_base_burnpack
    };
    match apply_burnpack_part_bytes(model, deploy_base_burnpack.clone()) {
        Ok(result) => Ok(BurnpackApplySummary {
            applied: result.applied,
            unused: result.unused,
            error_count: result.errors.len(),
        }),
        Err(err) if err.contains("Unsupported dtype F16") => {
            let converted = convert_burnpack_bytes_precision(
                &deploy_base_burnpack,
                BurnpackFloatPrecision::F32,
            )
            .context("convert BitNet deploy scaffold burnpack to f32 fallback")?;
            apply_burnpack_part_bytes(model, converted)
                .map(|result| BurnpackApplySummary {
                    applied: result.applied,
                    unused: result.unused,
                    error_count: result.errors.len(),
                })
                .map_err(|retry_err| {
                    anyhow!(
                        "failed to apply BitNet deploy scaffold burnpack after f16->f32 fallback: {retry_err}"
                    )
                })
        }
        Err(err) => Err(anyhow!(
            "failed to apply BitNet deploy scaffold burnpack: {err}"
        )),
    }
}

pub fn apply_bitnet_artifact_bundle_to_model<B: BackendTrait>(
    model: &mut BDH<B>,
    artifact_bundle: &LanguageBitNetArtifactBundle,
    device: &B::Device,
) -> Result<()> {
    if let Some(deploy_base_burnpack) = artifact_bundle.deploy_base_burnpack.clone() {
        let apply_result = apply_deploy_base_burnpack_bytes_to_model(model, deploy_base_burnpack)?;
        if apply_result.error_count > 0 {
            return Err(anyhow!(
                "BitNet deploy scaffold burnpack reported {} apply errors",
                apply_result.error_count
            ));
        }
        if !apply_result.unused.is_empty() {
            return Err(anyhow!(
                "BitNet deploy scaffold burnpack had {} unmatched tensor entries",
                apply_result.unused.len()
            ));
        }
        if apply_result.applied.is_empty() {
            return Err(anyhow!(
                "BitNet deploy scaffold burnpack did not apply any tensors"
            ));
        }
    }

    model
        .apply_bitnet_static_artifacts(&artifact_bundle.static_weights, device)
        .context("apply bitnet static artifacts")
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

    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let mut model = BDH::<ExportBackend>::new(model_config, &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<ExportBackend> as Module<ExportBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
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

pub fn export_language_checkpoint_to_bitnet_artifact(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    output_path: &Path,
) -> Result<LanguageBitNetArtifactExportReport> {
    ensure_supported_bitnet_artifact_path(output_path)?;
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let checkpoint_path = checkpoint.to_path_buf();
    let config =
        load_training_config_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let run_dir = resolve_checkpoint_run_dir(Some(&checkpoint_path), backend_name);
    let config_hash = sha256_json(&config).context("hash training config for bitnet artifact")?;

    let tokenizer =
        load_tokenizer_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;
    if !model_config.quant.enable {
        return Err(anyhow!(
            "language BitNet export requires model.quant.enable = true"
        ));
    }
    if !matches!(
        model_config.quant.inference_mode,
        burn_dragon_core::LowBitInferenceMode::OfflinePack
    ) {
        return Err(anyhow!(
            "language BitNet export currently requires model.quant.inference_mode = \"offline_pack\""
        ));
    }

    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);
    let mut model = BDH::<ExportBackend>::new(model_config.clone(), &device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<ExportBackend> as Module<ExportBackend>>::Record>(
            checkpoint_base.clone(),
            &device,
        )
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);

    let static_weights = model.export_bitnet_static_artifacts();
    if static_weights.decoder_x.is_none()
        && static_weights.decoder_y.is_none()
        && static_weights.encoder.is_none()
    {
        return Err(anyhow!(
            "language BitNet export found no quantized static matrices to pack"
        ));
    }

    let bundle = LanguageBitNetArtifactBundle {
        schema_version: 2,
        source_checkpoint_epoch: epoch,
        source_training_config_sha256: config_hash,
        source_run_dir: run_dir.clone(),
        kernel_abi_version: model_config.quant.enable.then_some(1),
        quant: model_config.quant,
        rho: model_config.rho,
        deploy_base_burnpack: Some(
            export_bitnet_deploy_base_burnpack_bytes(&model)
                .context("export standalone BitNet deploy scaffold")?,
        ),
        static_weights,
    };

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let binary_bytes =
        serialize_bitnet_artifact_binary(&bundle).context("serialize binary bitnet artifact")?;
    if output_path.extension().is_some_and(|ext| ext == "gz") {
        let file = fs::File::create(output_path)
            .with_context(|| format!("failed to create {}", output_path.display()))?;
        let mut encoder = GzEncoder::new(file, Compression::best());
        encoder
            .write_all(&binary_bytes)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
        encoder
            .finish()
            .with_context(|| format!("failed to finalize {}", output_path.display()))?;
    } else {
        fs::write(output_path, binary_bytes)
            .with_context(|| format!("failed to write {}", output_path.display()))?;
    }

    Ok(LanguageBitNetArtifactExportReport {
        checkpoint_base,
        epoch,
        run_dir,
        artifact_path: output_path.to_path_buf(),
        bundle,
    })
}

pub fn load_bitnet_artifact_bundle(path: &Path) -> Result<LanguageBitNetArtifactBundle> {
    ensure_supported_bitnet_artifact_path(path)?;
    if path.extension().is_some_and(|ext| ext == "gz") {
        let file =
            fs::File::open(path).with_context(|| format!("failed to read {}", path.display()))?;
        let mut decoder = GzDecoder::new(file);
        let mut contents = Vec::new();
        decoder
            .read_to_end(&mut contents)
            .with_context(|| format!("failed to decompress {}", path.display()))?;
        if !contents.starts_with(BITNET_ARTIFACT_BINARY_MAGIC) {
            return Err(anyhow!(
                "BitNet artifact {} is not in the supported binary format; re-export it as `.bitnet_artifact.bin.gz`",
                path.display()
            ));
        }
        deserialize_bitnet_artifact_binary(&contents, &path.display().to_string())
    } else {
        let contents =
            fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
        deserialize_bitnet_artifact_binary(&contents, &path.display().to_string())
    }
}

pub fn load_tokenizer_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: Option<&PathBuf>,
    backend_name: &str,
) -> Result<SharedTokenizer> {
    let config = load_training_config_for_checkpoint(config_paths, checkpoint, backend_name)?;
    let tokenizer_path = config
        .dataset
        .tokenizer
        .storage_path(&config.dataset.cache_dir);
    match tokenizer_path {
        Some(path) => config
            .dataset
            .tokenizer
            .load(&path)
            .with_context(|| format!("failed to load tokenizer {}", path.display())),
        None => config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())
            .context("failed to initialize tokenizer"),
    }
}

pub fn load_language_core_from_checkpoint<B: BackendTrait>(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    device: &B::Device,
) -> Result<BDH<B>> {
    let (checkpoint_base, _epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let checkpoint_path = checkpoint.to_path_buf();
    let config =
        load_training_config_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let tokenizer =
        load_tokenizer_for_checkpoint(config_paths, Some(&checkpoint_path), backend_name)?;
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )?;
    let mut model = BDH::<B>::new(model_config, device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    model = model.load_record(record);
    Ok(model)
}

pub fn apply_init_checkpoint_to_language_core<B: BackendTrait>(
    target_model: &BDH<B>,
    target_config: &TrainingConfig,
    init_checkpoint_path: &Path,
    init_checkpoint_epoch: Option<usize>,
    backend_name: &str,
    device: &B::Device,
) -> Result<BDH<B>> {
    let checkpoint_path = init_checkpoint_path.to_path_buf();
    let (checkpoint_base, epoch) = resolve_checkpoint_base(&checkpoint_path, init_checkpoint_epoch)
        .with_context(|| {
            format!(
                "failed to resolve init checkpoint from {}",
                checkpoint_path.display()
            )
        })?;
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<BDH<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
    let source_config =
        load_training_config_for_checkpoint(&[], Some(&checkpoint_path), backend_name)
            .with_context(|| {
                format!(
                    "failed to load source training config for init checkpoint {}",
                    checkpoint_path.display()
                )
            })?;
    let current_language_head = target_config
        .model
        .language_head
        .clone()
        .unwrap_or_default();
    let source_language_head = source_config
        .model
        .language_head
        .clone()
        .unwrap_or_default();
    let preserve_input_embedding =
        source_config.dataset.tokenizer.kind != target_config.dataset.tokenizer.kind;
    let preserve_output_head =
        preserve_input_embedding || source_language_head != current_language_head;
    let loaded = if preserve_input_embedding || preserve_output_head {
        target_model.load_record_preserving_tokenizer_surfaces(
            record,
            preserve_input_embedding,
            preserve_output_head,
        )
    } else {
        target_model.clone().load_record(record)
    };
    let interface_reference = if let Some(interface_checkpoint_path) = target_config
        .training
        .init_transfer
        .interface_checkpoint_path
        .as_ref()
    {
        let (interface_base, interface_epoch) = resolve_checkpoint_base(
            interface_checkpoint_path,
            target_config
                .training
                .init_transfer
                .interface_checkpoint_epoch,
        )
        .with_context(|| {
            format!(
                "failed to resolve init transfer interface checkpoint from {}",
                interface_checkpoint_path.display()
            )
        })?;
        let interface_record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load::<<BDH<B> as Module<B>>::Record>(interface_base.clone(), device)
            .map_err(|err| anyhow!(format_checkpoint_load_error(&interface_base, err)))?;
        let interface_config =
            load_training_config_for_checkpoint(&[], Some(interface_checkpoint_path), backend_name)
                .with_context(|| {
                    format!(
                        "failed to load interface training config for checkpoint {}",
                        interface_checkpoint_path.display()
                    )
                })?;
        let interface_language_head = interface_config
            .model
            .language_head
            .clone()
            .unwrap_or_default();
        let preserve_interface_embedding =
            interface_config.dataset.tokenizer.kind != target_config.dataset.tokenizer.kind;
        let preserve_interface_head =
            preserve_interface_embedding || interface_language_head != current_language_head;
        let interface_model = if preserve_interface_embedding || preserve_interface_head {
            target_model.load_record_preserving_tokenizer_surfaces(
                interface_record,
                preserve_interface_embedding,
                preserve_interface_head,
            )
        } else {
            target_model.clone().load_record(interface_record)
        };
        Some((interface_model, interface_base, interface_epoch))
    } else {
        None
    };
    let reference_model = interface_reference
        .as_ref()
        .map(|(model, _, _)| model)
        .unwrap_or(target_model);
    let loaded = if let Some((interface_model, _, _)) = interface_reference.as_ref() {
        let interface_checkpoint_config = load_training_config_for_checkpoint(
            &[],
            target_config
                .training
                .init_transfer
                .interface_checkpoint_path
                .as_ref(),
            backend_name,
        )?;
        if target_config
            .training
            .init_transfer
            .preserve_interface_input_embedding
            || target_config
                .training
                .init_transfer
                .preserve_interface_output_head
            || target_config
                .training
                .init_transfer
                .interface_output_head_blend_alpha
                .is_some()
        {
            anyhow::ensure!(
                interface_checkpoint_config.dataset.tokenizer.kind
                    == target_config.dataset.tokenizer.kind,
                "training.init_transfer.preserve_interface_input_embedding/output_head requires interface tokenizer kind to match target tokenizer kind"
            );
            anyhow::ensure!(
                interface_checkpoint_config
                    .model
                    .language_head
                    .clone()
                    .unwrap_or_default()
                    == current_language_head,
                "training.init_transfer.preserve_interface_output_head requires interface language head to match target language head"
            );
        }
        loaded
            .with_tokenizer_surfaces_from(
                interface_model,
                target_config
                    .training
                    .init_transfer
                    .preserve_interface_input_embedding,
                target_config
                    .training
                    .init_transfer
                    .preserve_interface_output_head,
            )
            .with_output_head_blended_from(
                interface_model,
                target_config
                    .training
                    .init_transfer
                    .interface_output_head_blend_alpha
                    .unwrap_or(0.0),
            )
    } else {
        loaded
    };
    let loaded = loaded.adapted_transferred_backbone(
        reference_model,
        target_config.training.init_transfer.backbone_blend_alpha,
        target_config.training.init_transfer.decoder_blend_alpha,
        target_config.training.init_transfer.norm_blend_alpha,
        target_config.training.init_transfer.fresh_top_layers,
        target_config.training.init_transfer.preserve_fresh_decoder,
        target_config.training.init_transfer.preserve_fresh_norm,
        target_config.training.init_transfer.match_fresh_rms,
    );
    tracing::info!(
        "initialized model weights from checkpoint epoch {epoch} at {} (interface_checkpoint={}, interface_epoch={:?}, interface_embed={}, interface_head={}, interface_head_blend_alpha={:?}, blend_alpha={:?}, decoder_blend_alpha={:?}, norm_blend_alpha={:?}, fresh_top_layers={:?}, preserve_fresh_decoder={}, preserve_fresh_norm={}, match_fresh_rms={})",
        checkpoint_base.display(),
        interface_reference
            .as_ref()
            .map(|(_, base, _)| base.display().to_string())
            .unwrap_or_else(|| "none".to_string()),
        interface_reference.as_ref().map(|(_, _, epoch)| *epoch),
        target_config
            .training
            .init_transfer
            .preserve_interface_input_embedding,
        target_config
            .training
            .init_transfer
            .preserve_interface_output_head,
        target_config
            .training
            .init_transfer
            .interface_output_head_blend_alpha,
        target_config.training.init_transfer.backbone_blend_alpha,
        target_config.training.init_transfer.decoder_blend_alpha,
        target_config.training.init_transfer.norm_blend_alpha,
        target_config.training.init_transfer.fresh_top_layers,
        target_config.training.init_transfer.preserve_fresh_decoder,
        target_config.training.init_transfer.preserve_fresh_norm,
        target_config.training.init_transfer.match_fresh_rms
    );
    Ok(loaded)
}

pub fn load_language_core_from_checkpoint_with_bitnet_artifact<B: BackendTrait>(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    backend_name: &str,
    artifact_path: &Path,
    device: &B::Device,
) -> Result<BDH<B>> {
    let mut model =
        load_language_core_from_checkpoint(checkpoint, epoch, config_paths, backend_name, device)?;
    let artifact_bundle = load_bitnet_artifact_bundle(artifact_path)?;
    apply_bitnet_artifact_bundle_to_model(&mut model, &artifact_bundle, device)?;
    Ok(model)
}

pub fn apply_run_config(config: &mut TrainingConfig, run_config: &LanguageRunConfigSnapshot) {
    let block_override = run_config
        .block_size
        .or(run_config.overrides.block_size)
        .map(|value| value.max(1));
    if let Some(block_size) = block_override {
        config.training.block_size = block_size;
    }
    if let Some(sequence_kernel_override) = run_config.training_sequence_kernel_override {
        config.training.sequence_kernel_override = Some(sequence_kernel_override);
    }
    if let Some(launch_mode) = run_config.training_launch_mode_requested {
        config.training.launch_mode = launch_mode;
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
    if let Some(value) = incoming.latent_total {
        base.latent_total = Some(value);
    }
    if let Some(value) = &incoming.initialization {
        base.initialization = Some(value.clone());
    }
    if let Some(value) = incoming.sequence_kernel {
        base.sequence_kernel = Some(value);
    }
    if let Some(value) = &incoming.mamba {
        base.mamba = Some(value.clone());
    }
    if let Some(value) = incoming.residual_connector {
        base.residual_connector = Some(value);
    }
    if let Some(value) = &incoming.attention_residual {
        base.attention_residual = Some(value.clone());
    }
    if let Some(value) = &incoming.block_attention_residual {
        base.block_attention_residual = Some(value.clone());
    }
    if let Some(value) = &incoming.latent_fanout_schedule {
        base.latent_fanout_schedule = Some(value.clone());
    }
    if let Some(value) = incoming.relu_threshold {
        base.relu_threshold = Some(value);
    }
    if let Some(value) = incoming.dropout {
        base.dropout = Some(value);
    }
    if let Some(value) = &incoming.normalization {
        base.normalization = Some(value.clone());
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
    if let Some(value) = &incoming.clocked_slow_memory {
        base.clocked_slow_memory = Some(value.clone());
    }
    if let Some(value) = &incoming.summary_memory {
        base.summary_memory = Some(value.clone());
    }
    if let Some(value) = &incoming.mhc {
        base.mhc = Some(value.clone());
    }
    if let Some(value) = &incoming.quant {
        base.quant = Some(value.clone());
    }
    if let Some(value) = &incoming.rho {
        base.rho = Some(value.clone());
    }
}

pub fn default_bitnet_artifact_path(checkpoint_base: &Path, epoch: usize) -> PathBuf {
    let checkpoint_dir = checkpoint_base
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let run_dir = checkpoint_dir
        .file_name()
        .is_some_and(|name| name == "checkpoint")
        .then(|| checkpoint_dir.parent().map(Path::to_path_buf))
        .flatten()
        .unwrap_or(checkpoint_dir);
    run_dir
        .join("deploy")
        .join(format!("model-{epoch}.bitnet_artifact.bin.gz"))
}

pub fn candidate_bitnet_artifact_paths(checkpoint_base: &Path, epoch: usize) -> [PathBuf; 2] {
    let preferred = default_bitnet_artifact_path(checkpoint_base, epoch);
    let deploy_dir = preferred
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    [
        preferred,
        deploy_dir.join(format!("model-{epoch}.bitnet_artifact.bin")),
    ]
}

fn sha256_json<T: Serialize>(value: &T) -> Result<String> {
    let json = serde_json::to_vec(value).context("serialize json for sha256")?;
    let mut hasher = Sha256::new();
    hasher.update(json.as_slice());
    Ok(format!("{:x}", hasher.finalize()))
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
        .unwrap_or_else(|| resolve_run_root().join("checkpoint"))
}

pub fn resolve_latest_run_dir(backend_name: &str) -> Option<PathBuf> {
    let run_root = resolve_run_root();
    resolve_latest_run_dir_shared(&run_root).or_else(|| {
        let device_root = run_root.join(backend_name);
        resolve_latest_run_dir_shared(&device_root)
    })
}

pub fn resolve_latest_run_dir_in(run_root: &Path) -> Option<PathBuf> {
    resolve_latest_run_dir_shared(run_root)
}

pub fn resolve_run_root() -> PathBuf {
    std::env::var_os(RUN_ROOT_ENV)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("runs"))
}

pub fn training_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub fn tokenizer_snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TOKENIZER_SNAPSHOT_FILE_NAME)
}

pub(crate) fn resolve_checkpoint_base(
    path: &Path,
    epoch: Option<usize>,
) -> Result<(PathBuf, usize)> {
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
        let cwd_relative = std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(&config.dataset.cache_dir));
        config.dataset.cache_dir = match cwd_relative {
            Some(path) if path.exists() => path,
            _ => run_dir.join(&config.dataset.cache_dir),
        };
    }
    if let Some(validation) = &mut config.dataset.validation
        && let Some(cache_dir) = &mut validation.cache_dir
        && !cache_dir.is_absolute()
    {
        let cwd_relative = std::env::current_dir()
            .ok()
            .map(|cwd| cwd.join(&*cache_dir));
        *cache_dir = match cwd_relative {
            Some(path) if path.exists() => path,
            _ => run_dir.join(&*cache_dir),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::{
        BurnpackBundleExportOptions, Compression, ExportBackend, GzEncoder,
        LanguageBitNetArtifactBundle, LanguageRunConfigSnapshot,
        apply_bitnet_artifact_bundle_to_model, apply_run_config, candidate_bitnet_artifact_paths,
        default_bitnet_artifact_path, export_language_checkpoint_to_bitnet_artifact,
        export_language_checkpoint_to_burnpack, load_bitnet_artifact_bundle,
        load_language_core_from_checkpoint,
        load_language_core_from_checkpoint_with_bitnet_artifact,
        load_training_config_for_checkpoint, resolve_checkpoint_base, tokenizer_snapshot_path,
        training_snapshot_path, write_training_snapshot,
    };
    use crate::BDH;
    use crate::config::{
        ContextStrategyConfig, DatasetConfig, DatasetSourceConfig, GenerationConfig,
        ModelOverrides, TrainingConfig, TrainingHyperparameters,
    };
    use crate::tokenizer::TokenizerConfig;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor};
    use burn_dragon_checkpoint::{
        BurnpackFloatPrecision, burnpack_parts_manifest_path, checkpoint_bin_path,
        manifest_is_complete, save_model_to_burnpack,
    };
    use burn_dragon_train::{
        OptimizerConfig, OptimizerKind, OptimizerScheduleMode, WgpuRuntimeConfig,
    };
    use std::fs;
    use std::io::Write;
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

        let loaded =
            load_training_config_for_checkpoint(&[], Some(&run_dir.join("checkpoint")), "wgpu")
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
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
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

    #[test]
    fn exports_language_checkpoint_to_bitnet_artifact_bundle() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let mut config = test_config(dir.path().join("cache"));
        config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Ternary158,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::Encoder,
                burn_dragon_core::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Int8,
            ..Default::default()
        });
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");

        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let artifact_path = run_dir.join("deploy/model-0.bitnet_artifact.bin");
        let report = export_language_checkpoint_to_bitnet_artifact(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &artifact_path,
        )
        .expect("export bitnet artifact");

        assert_eq!(report.epoch, 0);
        assert_eq!(report.artifact_path, artifact_path);
        assert!(report.bundle.deploy_base_burnpack.is_some());
        assert!(report.bundle.static_weights.decoder_x.is_none());
        assert!(report.bundle.static_weights.decoder_y.is_some());
        assert!(report.bundle.static_weights.encoder.is_some());
        assert_eq!(report.bundle.kernel_abi_version, Some(1));
        assert!(report.artifact_path.is_file());
    }

    #[test]
    fn bitnet_deploy_scaffold_burnpack_uses_smaller_f16_payload() {
        let dir = tempdir().expect("tempdir");
        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let config = test_config(dir.path().join("cache"));
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config, &device);
        let scaffold = model.export_bitnet_deploy_scaffold();
        let f32_burnpack = save_model_to_burnpack(&scaffold, &dir.path().join("scaffold_f32"))
            .expect("write f32 scaffold burnpack");
        let f32_bytes = fs::read(&f32_burnpack).expect("read f32 scaffold burnpack");
        let f16_bytes =
            super::export_bitnet_deploy_base_burnpack_bytes(&model).expect("export f16 scaffold");

        assert!(
            f16_bytes.len() < f32_bytes.len(),
            "expected f16 scaffold burnpack to be smaller than f32 (f16={}, f32={})",
            f16_bytes.len(),
            f32_bytes.len()
        );
        assert!(
            f16_bytes.windows(3).any(|window| window == b"F16"),
            "expected scaffold burnpack metadata to encode f16 tensors"
        );
    }

    #[test]
    fn bitnet_deploy_scaffold_only_applies_fp_remainder() {
        let dir = tempdir().expect("tempdir");
        let tokenizer = test_config(dir.path().join("cache"))
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");
        let device = <ExportBackend as BackendTrait>::Device::default();

        let mut dy_config = test_config(dir.path().join("cache_dy"));
        dy_config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Int8,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![burn_dragon_core::LowBitTargetModule::DecoderY],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Fp16,
            ..Default::default()
        });
        let dy_model_config = crate::build_model_config_with_tokenizer(
            &dy_config.model,
            dy_config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build dy model config");
        ExportBackend::seed(&device, 1337);
        let dy_model = BDH::<ExportBackend>::new(dy_model_config.clone(), &device);
        let dy_scaffold_bytes =
            super::export_bitnet_deploy_base_burnpack_bytes(&dy_model).expect("export dy scaffold");
        let mut dy_apply_model = BDH::<ExportBackend>::new(dy_model_config, &device);
        let dy_apply = super::apply_deploy_base_burnpack_bytes_to_model(
            &mut dy_apply_model,
            dy_scaffold_bytes,
        )
        .expect("apply dy scaffold");

        let mut xy_config = test_config(dir.path().join("cache_xy"));
        xy_config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Int8,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::DecoderX,
                burn_dragon_core::LowBitTargetModule::DecoderY,
                burn_dragon_core::LowBitTargetModule::Encoder,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Int8,
            encoder_mode: Some(burn_dragon_core::LowBitWeightFormat::Int8),
            ..Default::default()
        });
        let xy_model_config = crate::build_model_config_with_tokenizer(
            &xy_config.model,
            xy_config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build x+y+enc model config");
        ExportBackend::seed(&device, 1337);
        let xy_model = BDH::<ExportBackend>::new(xy_model_config.clone(), &device);
        let xy_scaffold_bytes =
            super::export_bitnet_deploy_base_burnpack_bytes(&xy_model).expect("export xy scaffold");
        let mut xy_apply_model = BDH::<ExportBackend>::new(xy_model_config, &device);
        let xy_apply = super::apply_deploy_base_burnpack_bytes_to_model(
            &mut xy_apply_model,
            xy_scaffold_bytes,
        )
        .expect("apply xy scaffold");

        let forbidden = ["encoder", "encoder_v", "decoder"];
        for key in dy_apply.applied.iter().chain(xy_apply.applied.iter()) {
            for name in forbidden {
                let matches_name = key == name
                    || key.starts_with(&format!("{name}."))
                    || key.contains(&format!(".{name}."))
                    || key.ends_with(&format!(".{name}"));
                assert!(
                    !matches_name,
                    "BitNet deploy scaffold should not apply targeted low-bit matrix `{name}`; applied key: {key}"
                );
            }
        }

        assert!(
            dy_apply.applied.iter().any(|key| key.contains("embed")),
            "expected deploy scaffold to carry fp remainder tensors like embed"
        );
        assert!(
            dy_apply.applied.iter().any(|key| key.contains("lm_head")),
            "expected deploy scaffold to carry fp remainder tensors like lm_head"
        );
    }

    #[test]
    fn loaded_bitnet_artifact_model_matches_fake_quant_checkpoint_logits() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let mut config = test_config(dir.path().join("cache"));
        config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Ternary158,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::Encoder,
                burn_dragon_core::LowBitTargetModule::DecoderX,
                burn_dragon_core::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Sign1,
            ..Default::default()
        });
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");
        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let artifact_path = run_dir.join("deploy/model-0.bitnet_artifact.bin");
        export_language_checkpoint_to_bitnet_artifact(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &artifact_path,
        )
        .expect("export bitnet artifact");

        let reference_model = load_language_core_from_checkpoint::<ExportBackend>(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &device,
        )
        .expect("load reference model");
        let packed_model =
            load_language_core_from_checkpoint_with_bitnet_artifact::<ExportBackend>(
                &checkpoint_dir,
                Some(0),
                &[],
                "wgpu",
                &artifact_path,
                &device,
            )
            .expect("load packed model");
        let tokens = Tensor::<ExportBackend, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7], [1, 8]),
            &device,
        );
        let reference_logits = reference_model.forward(tokens.clone());
        let packed_logits = packed_model.forward(tokens);
        let reference = reference_logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference vec");
        let packed = packed_logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("packed vec");
        let max_diff = reference
            .iter()
            .zip(packed.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1.0e-4,
            "expected packed bitnet artifact model to match fake-quant logits, max diff {max_diff}"
        );
    }

    #[test]
    fn standalone_bitnet_artifact_bundle_reloads_without_checkpoint_weights() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let mut config = test_config(dir.path().join("cache"));
        config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Ternary158,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::Encoder,
                burn_dragon_core::LowBitTargetModule::DecoderX,
                burn_dragon_core::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Packed2,
            ..Default::default()
        });
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");
        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config.clone(), &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let artifact_path = run_dir.join("deploy/model-0.bitnet_artifact.bin");
        export_language_checkpoint_to_bitnet_artifact(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &artifact_path,
        )
        .expect("export bitnet artifact");
        let artifact_bundle = load_bitnet_artifact_bundle(&artifact_path).expect("load artifact");
        assert!(artifact_bundle.deploy_base_burnpack.is_some());

        let packed_model =
            load_language_core_from_checkpoint_with_bitnet_artifact::<ExportBackend>(
                &checkpoint_dir,
                Some(0),
                &[],
                "wgpu",
                &artifact_path,
                &device,
            )
            .expect("load checkpoint-backed packed model");
        let tokens = Tensor::<ExportBackend, 2, Int>::from_data(
            burn::tensor::TensorData::new(vec![0i64, 1, 2, 3, 4, 5, 6, 7], [1, 8]),
            &device,
        );
        let reference_logits = packed_model.forward(tokens.clone());
        let (checkpoint_base, _) =
            resolve_checkpoint_base(&checkpoint_dir, Some(0)).expect("resolve checkpoint base");
        fs::remove_file(checkpoint_bin_path(&checkpoint_base)).expect("remove checkpoint file");

        let mut standalone_model = BDH::<ExportBackend>::new(model_config, &device);
        apply_bitnet_artifact_bundle_to_model(&mut standalone_model, &artifact_bundle, &device)
            .expect("apply standalone artifact bundle");
        let standalone_logits = standalone_model.forward(tokens);

        let reference = reference_logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference vec");
        let standalone = standalone_logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("standalone vec");
        let max_diff = reference
            .iter()
            .zip(standalone.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff <= 1.0e-4,
            "standalone bundle should match checkpoint-backed packed logits, max diff {max_diff}"
        );
    }

    fn test_config(cache_dir: PathBuf) -> TrainingConfig {
        TrainingConfig {
            dataset: DatasetConfig {
                cache_dir,
                train_split_ratio: 0.9,
                validation: None,
                source: DatasetSourceConfig::Shakespeare { url: None },
                tokenizer: TokenizerConfig::default(),
            },
            training: TrainingHyperparameters {
                block_size: 8,
                tbptt_chunk_size: None,
                tbptt_persist_across_steps: false,
                min_logical_block_size: None,
                batch_size: 2,
                seed: 1337,
                gradient_accumulation_steps: 1,
                target_effective_batch_size: None,
                epochs: Some(1),
                max_iters: 1,
                checkpoint_interval_iters: 2000,
                log_frequency: 1,
                launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
                resume_run_dir: None,
                resume_checkpoint_epoch: None,
                init_checkpoint_path: None,
                init_checkpoint_epoch: None,
                init_transfer: Default::default(),
                continual_backprop: Default::default(),
                module_lr_scales: Vec::new(),
                context_strategy: ContextStrategyConfig::Infinite,
                sequence_kernel_override: None,
                gdpo: None,
            },
            optimizer: OptimizerConfig {
                name: OptimizerKind::default(),
                learning_rate: 1e-3,
                weight_decay: 0.0,
                weight_decay_final: None,
                lr_schedule: None,
                schedule_mode: OptimizerScheduleMode::default(),
                grad_clip_norm: None,
                grad_clip_value: None,
                muon: None,
            },
            parallel: Default::default(),
            generation: GenerationConfig {
                prompt: "To be".to_string(),
                max_tokens: Some(4),
                max_chars: None,
                temperature: 1.0,
                top_k: Some(4),
                context_strategy: ContextStrategyConfig::Infinite,
                prompt_tokenizer: Default::default(),
                decode_tokenizer: Default::default(),
                output_format: Default::default(),
            },
            wgpu: WgpuRuntimeConfig::default(),
            run_layout: burn_dragon_train::RunLayoutConfig::default(),
            model: ModelOverrides::default(),
        }
    }

    #[test]
    fn parse_run_config_snapshot_accepts_phase0_metadata() {
        let snapshot: super::LanguageRunConfigSnapshot = serde_json::from_str(
            r#"{
                "block_size": 128,
                "seed": 4242,
                "arch_version": "dragon_bdh_v1",
                "shard_layout_version": 1,
                "overrides": {
                    "n_embd": 256,
                    "latent_total": 32768
                },
                "model_spec": {
                    "arch": "dragon_bdh",
                    "n_embd": 256,
                    "n_head": 4,
                    "n_layer": 8,
                    "latent_total": 32768,
                    "latent_per_head": 8192,
                    "shared_layer_weights": true,
                    "sequence_kernel": "linear_attention"
                },
                "parallel_spec": {
                    "mode": "single",
                    "world_size": 1,
                    "data_parallel_size": 1,
                    "tensor_parallel_size": 1,
                    "tensor_parallel_axis": "neuron",
                    "tensor_parallel_partition": "contiguous",
                    "fsdp_enabled": false,
                    "checkpoint_format": "unsharded_v1"
                },
                "kernel_spec": {
                    "sequence_kernel": "linear_attention",
                    "fused_kernels_enabled": true,
                    "rollout_fast_steps_per_slow_step": 1,
                    "wgpu_fused_core_recurrent": true,
                    "wgpu_fused_core_rollout": false
                },
                "state_layout": {
                    "state_family": "bdh_model_state",
                    "position_tracked": true,
                    "layers": [
                        {
                            "layer_index": 0,
                            "latent_total": 32768,
                            "latent_per_head": 8192,
                            "tensors": [
                                {
                                    "name": "rho",
                                    "axes": [
                                        {"name": "batch_views", "size": null},
                                        {"name": "heads", "size": 4},
                                        {"name": "latent_per_head", "size": 8192},
                                        {"name": "dense_dim", "size": 256}
                                    ]
                                }
                            ]
                        }
                    ]
                },
                "metrics_sink": {
                    "family": "language_bdh_burn_train_v1",
                    "entries": [
                        {
                            "name": "Loss",
                            "split": "train",
                            "value_kind": "numeric",
                            "every_steps": 1
                        },
                        {
                            "name": "Loss",
                            "split": "valid",
                            "value_kind": "numeric",
                            "every_steps": 1
                        }
                    ]
                }
            }"#,
        )
        .expect("parse snapshot");

        assert_eq!(snapshot.seed, Some(4242));
        assert_eq!(snapshot.arch_version.as_deref(), Some("dragon_bdh_v1"));
        assert_eq!(snapshot.shard_layout_version, Some(1));
        assert_eq!(snapshot.overrides.latent_total, Some(32768));
        assert_eq!(
            snapshot
                .model_spec
                .as_ref()
                .expect("model spec")
                .latent_per_head,
            8192
        );
        assert_eq!(
            snapshot.state_layout.as_ref().expect("state layout").layers[0].tensors[0].name,
            "rho"
        );
        assert_eq!(
            snapshot.metrics_sink.as_ref().expect("metrics sink").family,
            "language_bdh_burn_train_v1"
        );
    }

    #[test]
    fn apply_run_config_merges_sequence_kernel_override() {
        let mut config = test_config(PathBuf::from("data"));
        let snapshot = LanguageRunConfigSnapshot {
            training_execution_form: Some("default_stateful".to_string()),
            training_launch_mode_requested: Some(
                burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            ),
            training_sequence_kernel_override: Some(
                burn_dragon_core::SequenceKernelConfig::dense_score_short_context(),
            ),
            overrides: ModelOverrides {
                sequence_kernel: Some(
                    burn_dragon_core::SequenceKernelConfig::dense_score_short_context(),
                ),
                ..ModelOverrides::default()
            },
            ..LanguageRunConfigSnapshot::default()
        };

        apply_run_config(&mut config, &snapshot);

        assert_eq!(
            config.training.sequence_kernel_override,
            Some(burn_dragon_core::SequenceKernelConfig::dense_score_short_context())
        );
        assert_eq!(
            config.model.sequence_kernel,
            Some(burn_dragon_core::SequenceKernelConfig::dense_score_short_context())
        );
    }

    #[test]
    fn apply_run_config_merges_initialization_override() {
        let mut config = test_config(PathBuf::from("data"));
        let snapshot = LanguageRunConfigSnapshot {
            overrides: ModelOverrides {
                initialization: Some(burn_dragon_core::BdhInitializationConfig {
                    kind: burn_dragon_core::BdhInitializationKind::SimpleNormal,
                    simple_normal_std: 0.015,
                    ..Default::default()
                }),
                ..ModelOverrides::default()
            },
            ..LanguageRunConfigSnapshot::default()
        };

        apply_run_config(&mut config, &snapshot);

        assert_eq!(
            config.model.initialization,
            Some(burn_dragon_core::BdhInitializationConfig {
                kind: burn_dragon_core::BdhInitializationKind::SimpleNormal,
                simple_normal_std: 0.015,
                ..Default::default()
            })
        );
    }

    #[test]
    fn default_bitnet_artifact_path_uses_run_deploy_layout() {
        let path = default_bitnet_artifact_path(Path::new("runs/example/checkpoint/model-3"), 3);
        assert_eq!(
            path,
            PathBuf::from("runs/example/deploy/model-3.bitnet_artifact.bin.gz")
        );
        assert_eq!(
            candidate_bitnet_artifact_paths(Path::new("runs/example/checkpoint/model-3"), 3),
            [
                PathBuf::from("runs/example/deploy/model-3.bitnet_artifact.bin.gz"),
                PathBuf::from("runs/example/deploy/model-3.bitnet_artifact.bin"),
            ]
        );
    }

    #[test]
    fn loads_gzip_bitnet_artifact_bundle() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let mut config = test_config(dir.path().join("cache"));
        config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Ternary158,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::Encoder,
                burn_dragon_core::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Int8,
            ..Default::default()
        });
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");

        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let artifact_path = default_bitnet_artifact_path(&checkpoint_dir, 0);
        let report = export_language_checkpoint_to_bitnet_artifact(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &artifact_path,
        )
        .expect("export bitnet artifact");
        let loaded = load_bitnet_artifact_bundle(&artifact_path).expect("load gzip artifact");

        assert_eq!(report.artifact_path, artifact_path);
        assert!(artifact_path.is_file());
        assert_eq!(loaded.kernel_abi_version, Some(1));
        assert!(loaded.deploy_base_burnpack.is_some());
        assert!(loaded.static_weights.encoder.is_some());
        assert!(loaded.static_weights.decoder_y.is_some());
    }

    #[test]
    fn rejects_legacy_json_gzip_bitnet_artifact_bundle() {
        let dir = tempdir().expect("tempdir");
        let legacy_path = dir.path().join("model-0.bitnet_artifact.json.gz");
        let bundle = LanguageBitNetArtifactBundle {
            schema_version: 1,
            source_checkpoint_epoch: 0,
            source_training_config_sha256: "abc123".to_string(),
            source_run_dir: None,
            kernel_abi_version: Some(1),
            quant: burn_dragon_core::LowBitQuantizationConfig::default(),
            rho: burn_dragon_core::LowBitRhoConfig::default(),
            deploy_base_burnpack: None,
            static_weights:
                burn_dragon_core::experimental::bitnet_reference::BdhBitNetStaticArtifacts {
                    decoder_x: None,
                    decoder_y: Some(
                        burn_dragon_core::experimental::bitnet_reference::PackedWeightArtifact {
                            encoding: burn_dragon_core::experimental::bitnet_reference::PackedWeightEncoding::Ternary2,
                            logical_shape: vec![2, 4],
                            scale: 0.25,
                            packed: vec![0b10_01_00_10, 0b01_10_01_00],
                            len: 8,
                        },
                    ),
                    encoder: None,
                },
        };
        let json = serde_json::to_vec(&bundle).expect("serialize legacy json bundle");
        let file = fs::File::create(&legacy_path).expect("create legacy gzip file");
        let mut encoder = GzEncoder::new(file, Compression::best());
        encoder.write_all(&json).expect("write legacy gzip payload");
        encoder.finish().expect("finish legacy gzip");

        let err = load_bitnet_artifact_bundle(&legacy_path).expect_err("reject legacy json gzip");
        assert!(
            err.to_string().contains("unsupported BitNet artifact path"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn rejects_plain_bin_without_magic_prefix() {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("model-0.bitnet_artifact.bin");
        fs::write(&path, b"not-a-bitnet-bundle").expect("write invalid artifact");

        let err = load_bitnet_artifact_bundle(&path).expect_err("reject invalid plain bin");
        assert!(
            err.to_string().contains("supported binary format"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn rejects_legacy_json_output_path_for_bitnet_export() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        let mut config = test_config(dir.path().join("cache"));
        config.model.n_embd = Some(128);
        config.model.n_layer = Some(4);
        config.model.n_head = Some(4);
        config.model.latent_total = Some(4096);
        config.model.quant = Some(burn_dragon_core::LowBitQuantizationConfig {
            enable: true,
            protocol: burn_dragon_core::BitNetLowBitProtocol::BitnetB158,
            weight_format: burn_dragon_core::LowBitWeightFormat::Ternary158,
            act_format: burn_dragon_core::LowBitActivationFormat::Int8,
            target_modules: vec![
                burn_dragon_core::LowBitTargetModule::Encoder,
                burn_dragon_core::LowBitTargetModule::DecoderX,
                burn_dragon_core::LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: burn_dragon_core::LowBitWeightFormat::Sign1,
            ..Default::default()
        });
        let tokenizer = config
            .dataset
            .tokenizer
            .fit(["All the world's a stage"].into_iter())
            .expect("fit tokenizer");

        write_training_snapshot(&config, &run_dir, tokenizer.as_ref()).expect("write snapshot");

        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model_config = crate::build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .expect("build model config with tokenizer");
        let model = BDH::<ExportBackend>::new(model_config, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write bin checkpoint");

        let legacy_json_path = run_dir.join("deploy/model-0.bitnet_artifact.json.gz");
        let err = export_language_checkpoint_to_bitnet_artifact(
            &checkpoint_dir,
            Some(0),
            &[],
            "wgpu",
            &legacy_json_path,
        )
        .expect_err("reject legacy json output path");
        assert!(
            err.to_string().contains("unsupported BitNet artifact path"),
            "unexpected error: {err:#}"
        );
    }
}
