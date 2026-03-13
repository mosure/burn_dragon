#![cfg(feature = "train")]

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

use crate::config::{VisionTrainingConfig, VisionTrainingModeConfig, load_vision_training_config};
use crate::train::{
    VisionDistillModel, VisionLejepaInit, VisionLejepaModel, VisionReconstructionInit,
    resolve_vision_rollout,
};
use crate::train::vision::VisionVideoLejepaModel;
use crate::VisionDragon;

const TRAINING_SNAPSHOT_FILE_NAME: &str = "vision_training_config.json";

type ExportBackend = NdArray<f32>;

pub type VisionBurnpackExportReport = CheckpointExportReport;

pub fn write_training_snapshot(config: &VisionTrainingConfig, run_dir: &Path) -> Result<()> {
    write_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME, config)
}

pub fn load_training_snapshot_from_run_dir(run_dir: &Path) -> Result<VisionTrainingConfig> {
    load_json_snapshot(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub fn load_training_config_for_checkpoint(
    config_paths: &[PathBuf],
    checkpoint: &Path,
) -> Result<VisionTrainingConfig> {
    if !config_paths.is_empty() {
        return load_vision_training_config(config_paths);
    }

    if let Some(run_dir) = resolve_checkpoint_run_dir(checkpoint) {
        let snapshot_path = training_snapshot_path(&run_dir);
        if snapshot_path.is_file() {
            return load_training_snapshot_from_run_dir(&run_dir);
        }
    }

    Err(anyhow!(
        "vision export requires explicit config overlays or a run-local {} snapshot",
        TRAINING_SNAPSHOT_FILE_NAME
    ))
}

pub fn export_vision_encoder_checkpoint_to_burnpack(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    output_base: &Path,
    options: &BurnpackBundleExportOptions,
) -> Result<VisionBurnpackExportReport> {
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let config = load_training_config_for_checkpoint(config_paths, checkpoint)?;
    let vision_config = config.vision.build();
    let rollout = resolve_vision_rollout(&config.training, vision_config.steps)?;
    let device = <ExportBackend as BackendTrait>::Device::default();
    ExportBackend::seed(&device, 1337);

    let bundle = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            let model = VisionDragon::<ExportBackend>::new(vision_config.clone(), &device);
            let mut distill_model =
                VisionDistillModel::new(model, distill.clone(), None, rollout);
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionDistillModel<ExportBackend> as Module<ExportBackend>>::Record>(
                    checkpoint_base.clone(),
                    &device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            distill_model = distill_model.load_record(record);
            export_model_to_burnpack_bundle(&distill_model.model, output_base, options)
                .map_err(|err| anyhow!(err))?
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            let model = VisionDragon::<ExportBackend>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let num_classes = infer_image_dataset_num_classes(&config)?;
            let mut lejepa_model = VisionLejepaModel::new(
                model,
                lejepa.clone(),
                VisionLejepaInit {
                    embed_dim: vision_config.embed_dim,
                    num_classes,
                    rollout,
                    recon: VisionReconstructionInit {
                        patch_dim: recon_patch_dim,
                        normalize_std: config.augment.normalize_std,
                        patch_size: vision_config.patch_size,
                        in_channels: vision_config.in_channels,
                    },
                    normalization: vision_config.normalization.clone(),
                },
                &device,
            );
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionLejepaModel<ExportBackend> as Module<ExportBackend>>::Record>(
                    checkpoint_base.clone(),
                    &device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            lejepa_model = lejepa_model.load_record(record);
            export_model_to_burnpack_bundle(&lejepa_model.model, output_base, options)
                .map_err(|err| anyhow!(err))?
        }
        VisionTrainingModeConfig::VideoLejepa(video) => {
            let model = VisionDragon::<ExportBackend>::new(vision_config.clone(), &device);
            let mut video_model = VisionVideoLejepaModel::new(
                model,
                video.clone(),
                &vision_config,
                rollout,
                infer_video_dataset_num_classes(&config)?,
                &device,
            );
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionVideoLejepaModel<ExportBackend> as Module<ExportBackend>>::Record>(
                    checkpoint_base.clone(),
                    &device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            video_model = video_model.load_record(record);
            export_model_to_burnpack_bundle(&video_model.frame_model, output_base, options)
                .map_err(|err| anyhow!(err))?
        }
        _ => {
            return Err(anyhow!(
                "vision encoder export currently supports only mode.type = \"distill\", \"lejepa\", or \"video_lejepa\""
            ));
        }
    };

    Ok(CheckpointExportReport {
        checkpoint_base,
        epoch,
        run_dir: resolve_checkpoint_run_dir(checkpoint),
        bundle,
    })
}

pub fn load_vision_encoder_from_checkpoint<B: BackendTrait>(
    checkpoint: &Path,
    epoch: Option<usize>,
    config_paths: &[PathBuf],
    device: &B::Device,
) -> Result<VisionDragon<B>> {
    let (checkpoint_base, _epoch) = resolve_checkpoint_base(checkpoint, epoch)?;
    let config = load_training_config_for_checkpoint(config_paths, checkpoint)?;
    let vision_config = config.vision.build();
    let rollout = resolve_vision_rollout(&config.training, vision_config.steps)?;

    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            let model = VisionDragon::<B>::new(vision_config, device);
            let mut distill_model = VisionDistillModel::new(model, distill.clone(), None, rollout);
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionDistillModel<B> as Module<B>>::Record>(
                    checkpoint_base.clone(),
                    device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            distill_model = distill_model.load_record(record);
            Ok(distill_model.model)
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            let model = VisionDragon::<B>::new(vision_config.clone(), device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let num_classes = infer_image_dataset_num_classes(&config)?;
            let mut lejepa_model = VisionLejepaModel::new(
                model,
                lejepa.clone(),
                VisionLejepaInit {
                    embed_dim: vision_config.embed_dim,
                    num_classes,
                    rollout,
                    recon: VisionReconstructionInit {
                        patch_dim: recon_patch_dim,
                        normalize_std: config.augment.normalize_std,
                        patch_size: vision_config.patch_size,
                        in_channels: vision_config.in_channels,
                    },
                    normalization: vision_config.normalization.clone(),
                },
                device,
            );
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionLejepaModel<B> as Module<B>>::Record>(
                    checkpoint_base.clone(),
                    device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            lejepa_model = lejepa_model.load_record(record);
            Ok(lejepa_model.model)
        }
        VisionTrainingModeConfig::VideoLejepa(video) => {
            let model = VisionDragon::<B>::new(vision_config.clone(), device);
            let mut video_model = VisionVideoLejepaModel::new(
                model,
                video.clone(),
                &vision_config,
                rollout,
                infer_video_dataset_num_classes(&config)?,
                device,
            );
            let record = BinFileRecorder::<FullPrecisionSettings>::new()
                .load::<<VisionVideoLejepaModel<B> as Module<B>>::Record>(
                    checkpoint_base.clone(),
                    device,
                )
                .map_err(|err| anyhow!(format_checkpoint_load_error(&checkpoint_base, err)))?;
            video_model = video_model.load_record(record);
            Ok(video_model.frame_model)
        }
        _ => Err(anyhow!(
            "vision encoder load currently supports only mode.type = \"distill\", \"lejepa\", or \"video_lejepa\""
        )),
    }
}

pub fn training_snapshot_path(run_dir: &Path) -> PathBuf {
    run_snapshot_path(run_dir, TRAINING_SNAPSHOT_FILE_NAME)
}

pub(crate) fn resolve_checkpoint_run_dir(checkpoint: &Path) -> Option<PathBuf> {
    resolve_checkpoint_run_dir_shared(checkpoint)
}

pub(crate) fn resolve_checkpoint_base(path: &Path, epoch: Option<usize>) -> Result<(PathBuf, usize)> {
    resolve_checkpoint_base_shared(path, epoch)
}

fn infer_image_dataset_num_classes(config: &VisionTrainingConfig) -> Result<usize> {
    crate::train::vision::maybe_download_vision_dataset(&config.dataset)?;
    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let count = std::fs::read_dir(&train_root)
        .with_context(|| format!("failed to read {}", train_root.display()))?
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_type().ok())
        .filter(|kind| kind.is_dir())
        .count();
    if count == 0 {
        return Err(anyhow!(
            "no image-class directories found under {}",
            train_root.display()
        ));
    }
    Ok(count)
}

fn infer_video_dataset_num_classes(config: &VisionTrainingConfig) -> Result<usize> {
    if !matches!(config.mode, VisionTrainingModeConfig::VideoLejepa(_)) {
        return Err(anyhow!(
            "video dataset class inference requires mode.type = \"video_lejepa\""
        ));
    }
    Ok(10)
}

#[cfg(test)]
mod tests {
    use super::{
        ExportBackend, export_vision_encoder_checkpoint_to_burnpack,
        load_training_config_for_checkpoint,
        training_snapshot_path, write_training_snapshot,
    };
    use crate::{
        VisionBackboneKind, VisionDistillationLossConfig, VisionPatchEmbedMode,
    };
    use crate::config::{
        VisionAugmentationConfig, VisionDatasetConfig, VisionDatasetSource, VisionDistillConfig,
        VisionLejepaConfig, VisionModelConfig, VisionTeacherConfig, VisionTeacherFeatureConfig,
        VisionTrainingConfig, VisionTrainingHyperparameters, VisionTrainingModeConfig,
        VisionVideoLejepaConfig,
    };
    use crate::train::{
        VisionDistillModel, VisionLejepaInit, VisionLejepaModel, VisionReconstructionInit,
        resolve_vision_rollout,
    };
    use crate::train::vision::VisionVideoLejepaModel;
    use crate::VisionDragon;
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_checkpoint::{
        BurnpackBundleExportOptions, BurnpackFloatPrecision, burnpack_parts_manifest_path,
        manifest_is_complete,
    };
    use burn_dragon_train::{OptimizerConfig, WgpuRuntimeConfig};
    use std::fs;
    use std::path::PathBuf;
    use tempfile::tempdir;

    #[test]
    fn writes_and_loads_vision_training_snapshot() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("vision-run");
        let config = test_distill_config();

        write_training_snapshot(&config, &run_dir).expect("write vision snapshot");
        let loaded = load_training_config_for_checkpoint(&[], &run_dir.join("checkpoint"))
            .expect("load vision snapshot");

        assert!(training_snapshot_path(&run_dir).is_file());
        assert_eq!(loaded.mode, config.mode);
        assert_eq!(loaded.vision.backbone, config.vision.backbone);
    }

    #[test]
    fn exports_vision_distill_encoder_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("vision-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");

        let config = test_distill_config();
        write_training_snapshot(&config, &run_dir).expect("write vision snapshot");

        let vision_config = config.vision.build();
        let rollout = resolve_vision_rollout(&config.training, vision_config.steps).expect("rollout");
        let distill = match &config.mode {
            VisionTrainingModeConfig::Distill(distill) => distill.clone(),
            _ => unreachable!("test config uses distill mode"),
        };
        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = VisionDragon::<ExportBackend>::new(vision_config, &device);
        let distill_model = VisionDistillModel::new(model, distill, None, rollout);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(distill_model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_vision_encoder_checkpoint_to_burnpack(
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
        .expect("export vision burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }

    #[test]
    fn exports_vision_lejepa_encoder_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("vision-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        let train_root = dir.path().join("imagenet/train/class0");
        let val_root = dir.path().join("imagenet/val/class0");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");
        fs::create_dir_all(&train_root).expect("create train class dir");
        fs::create_dir_all(&val_root).expect("create val class dir");

        let config = test_lejepa_config(dir.path().join("imagenet"));
        write_training_snapshot(&config, &run_dir).expect("write vision snapshot");

        let vision_config = config.vision.build();
        let rollout =
            resolve_vision_rollout(&config.training, vision_config.steps).expect("rollout");
        let lejepa = match &config.mode {
            VisionTrainingModeConfig::Lejepa(lejepa) => lejepa.clone(),
            _ => unreachable!("test config uses lejepa mode"),
        };
        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = VisionDragon::<ExportBackend>::new(vision_config.clone(), &device);
        let recon = VisionReconstructionInit {
            patch_dim: vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels),
            normalize_std: config.augment.normalize_std,
            patch_size: vision_config.patch_size,
            in_channels: vision_config.in_channels,
        };
        let lejepa_model = VisionLejepaModel::new(
            model,
            lejepa,
            VisionLejepaInit {
                embed_dim: vision_config.embed_dim,
                num_classes: 1,
                rollout,
                recon,
                normalization: vision_config.normalization.clone(),
            },
            &device,
        );
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(lejepa_model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_vision_encoder_checkpoint_to_burnpack(
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
        .expect("export vision burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }

    #[test]
    fn exports_vision_video_lejepa_encoder_checkpoint_to_f16_burnpack_parts() {
        let dir = tempdir().expect("tempdir");
        let run_dir = dir.path().join("vision-run");
        let checkpoint_dir = run_dir.join("checkpoint");
        fs::create_dir_all(&checkpoint_dir).expect("create checkpoint dir");

        let config = test_video_lejepa_config();
        write_training_snapshot(&config, &run_dir).expect("write vision snapshot");

        let vision_config = config.vision.build();
        let rollout =
            resolve_vision_rollout(&config.training, vision_config.steps).expect("rollout");
        let video = match &config.mode {
            VisionTrainingModeConfig::VideoLejepa(video) => video.clone(),
            _ => unreachable!("test config uses video lejepa mode"),
        };
        let device = <ExportBackend as BackendTrait>::Device::default();
        ExportBackend::seed(&device, 1337);
        let model = VisionDragon::<ExportBackend>::new(vision_config.clone(), &device);
        let video_model =
            VisionVideoLejepaModel::new(model, video, &vision_config, rollout, 10, &device);
        BinFileRecorder::<FullPrecisionSettings>::new()
            .record(video_model.into_record(), checkpoint_dir.join("model-0"))
            .expect("write checkpoint");

        let report = export_vision_encoder_checkpoint_to_burnpack(
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
        .expect("export vision burnpack");

        assert!(report.bundle.burnpack_path.is_file());
        let manifest_path = burnpack_parts_manifest_path(&report.bundle.burnpack_path);
        assert!(
            manifest_is_complete(&manifest_path).expect("manifest status"),
            "exported multipart manifest should be complete"
        );
    }

    fn test_distill_config() -> VisionTrainingConfig {
        VisionTrainingConfig {
            dataset: VisionDatasetConfig {
                source: VisionDatasetSource::Imagenet,
                imagenet_root: PathBuf::from("data/imagenet"),
                train_dir: "train".to_string(),
                val_dir: "val".to_string(),
                max_records: Some(1),
                download: None,
                moving_mnist: Default::default(),
                prefetch_batches: 0,
                prefetch_workers: 0,
                prefetch_to_device: false,
                cache_decoded: false,
                cache_capacity: 1,
                cache_preprocessed: false,
            },
            training: VisionTrainingHyperparameters {
                batch_size: 1,
                epochs: Some(1),
                max_iters: 1,
                log_frequency: 1,
                batch_repeats: 1,
                train_repeat_chunk: 0,
                memory_cleanup_every: 0,
                memory_cleanup_iters: 0,
                device_memory_check_every: 0,
                max_device_memory_mb: 0,
                disable_cuda_memory_cleanup: false,
                enable_checkpoints: true,
                trace_train_loss: false,
                trace_train_loss_every: 1,
                rollout_min_steps: None,
                rollout_max_steps: None,
                rollout_backprop_steps: None,
                ffmpeg_path: None,
            },
            optimizer: OptimizerConfig {
                learning_rate: 1e-3,
                weight_decay: 0.0,
                lr_schedule: None,
                grad_clip_norm: None,
                grad_clip_value: None,
            },
            wgpu: WgpuRuntimeConfig::default(),
            vision: VisionModelConfig {
                image_size: 16,
                patch_size: 4,
                patch_embed_mode: VisionPatchEmbedMode::Linear,
                backbone: Some(VisionBackboneKind::Dense),
                in_channels: 3,
                embed_dim: 32,
                steps: 2,
                n_head: 4,
                mlp_internal_dim_multiplier: 2,
                dropout: 0.0,
                projection_dim: 16,
                projection_hidden_dim: 32,
                ..VisionModelConfig::default()
            },
            augment: VisionAugmentationConfig {
                image_size: 16,
                resize_short: 16,
                ..VisionAugmentationConfig::default()
            },
            mode: VisionTrainingModeConfig::Distill(VisionDistillConfig {
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: PathBuf::from("teacher/train_cls.bin"),
                    train_patch_path: PathBuf::from("teacher/train_patch.bin"),
                    val_cls_path: PathBuf::from("teacher/val_cls.bin"),
                    val_patch_path: PathBuf::from("teacher/val_patch.bin"),
                    feature_dim: 16,
                    patch_tokens: Some(16),
                }),
                loss: VisionDistillationLossConfig::default(),
                rollout_supervision_frames: 1,
                rollout_supervision_power: 1.0,
                rollout_sampling_power: 0.0,
            }),
        }
    }

    fn test_lejepa_config(imagenet_root: PathBuf) -> VisionTrainingConfig {
        let mut config = test_distill_config();
        config.dataset.imagenet_root = imagenet_root;
        config.mode = VisionTrainingModeConfig::Lejepa(VisionLejepaConfig::default());
        config
    }

    fn test_video_lejepa_config() -> VisionTrainingConfig {
        let mut config = test_distill_config();
        config.dataset.source = VisionDatasetSource::MovingMnist;
        config.dataset.train_dir = "train".to_string();
        config.dataset.val_dir = "val".to_string();
        config.mode = VisionTrainingModeConfig::VideoLejepa(VisionVideoLejepaConfig::default());
        config
    }
}
