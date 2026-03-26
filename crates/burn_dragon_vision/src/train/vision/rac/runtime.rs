use std::fmt::Write as _;

use ::image::RgbImage;
use burn::record::Recorder;
use serde::Serialize;

use crate::checkpoint::{infer_image_dataset_num_classes, resolve_checkpoint_base};
use crate::train::prelude::*;
use crate::train::vision::VisionRacBatchLoader;

use super::{
    VisionRacBatch, VisionRacModel, build_rac_semantic_teacher_store,
    build_rac_teacher_latent_store,
};

#[derive(Debug, Clone, Serialize)]
pub struct VisionRacCheckpointEvalSummary {
    pub backend: String,
    pub checkpoint: PathBuf,
    pub export_dir: PathBuf,
    pub split: String,
    pub teacher_kind: String,
    pub teacher_state_mapping: String,
    pub teacher_latent_downsample: usize,
    pub batches: usize,
    pub images: usize,
    pub forward_recon_loss: f64,
    pub forward_recon_psnr: f64,
    pub roundtrip_recon_loss: f64,
    pub roundtrip_recon_psnr: f64,
    pub forward_path_loss: f64,
    pub forward_velocity_loss: f64,
    pub reverse_latent_loss: f64,
    pub reverse_to_init_loss: f64,
    pub semantic_loss: Option<f64>,
}

fn tensor_scalar_f64<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .ok()
        .and_then(|values| values.into_iter().next())
        .unwrap_or(0.0) as f64
}

fn create_unique_dir(base: &Path) -> Result<PathBuf> {
    if !base.exists() {
        fs::create_dir_all(base)?;
        return Ok(base.to_path_buf());
    }

    let stem = base
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| anyhow!("invalid output directory {}", base.display()))?;
    let parent = base
        .parent()
        .ok_or_else(|| anyhow!("invalid output directory {}", base.display()))?;
    for suffix in 1..=999 {
        let candidate = parent.join(format!("{stem}_{suffix:03}"));
        if !candidate.exists() {
            fs::create_dir_all(&candidate)?;
            return Ok(candidate);
        }
    }
    Err(anyhow!(
        "failed to allocate unique export dir under {}",
        parent.display()
    ))
}

fn default_export_dir(checkpoint: &Path) -> Result<PathBuf> {
    let run_dir = crate::checkpoint::resolve_checkpoint_run_dir(checkpoint).ok_or_else(|| {
        anyhow!(
            "failed to resolve RAC run dir from checkpoint {}",
            checkpoint.display()
        )
    })?;
    let (checkpoint_base, epoch) = resolve_checkpoint_base(checkpoint, None)?;
    let checkpoint_name = checkpoint_base
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("model");
    create_unique_dir(
        &run_dir
            .join("artifacts")
            .join(format!("rac_eval_epoch_{epoch:03}_{checkpoint_name}")),
    )
}

fn denormalize_channel(value: f32, channel: usize, mean: &[f32], std: &[f32]) -> u8 {
    let mean = mean.get(channel).copied().unwrap_or(0.0);
    let std = std.get(channel).copied().unwrap_or(1.0).max(1.0e-6);
    let pixel = (value * std + mean).clamp(0.0, 1.0);
    (pixel * 255.0).round() as u8
}

fn save_tensor_images<B: BackendTrait>(
    tensor: Tensor<B, 4>,
    output_dir: &Path,
    offset: usize,
    mean: &[f32],
    std: &[f32],
) -> Result<usize> {
    let [batch, channels, height, width] = tensor.shape().dims::<4>();
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .map_err(|_| anyhow!("failed to materialize tensor image data"))?;
    let channel_stride = height.saturating_mul(width);
    let image_stride = channels.saturating_mul(channel_stride);
    for batch_idx in 0..batch {
        let start = batch_idx.saturating_mul(image_stride);
        let end = start.saturating_add(image_stride).min(values.len());
        let slice = &values[start..end];
        let mut rgb = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
        for y in 0..height {
            for x in 0..width {
                let pixel_idx = y.saturating_mul(width).saturating_add(x);
                let r = if channels > 0 {
                    denormalize_channel(slice[pixel_idx], 0, mean, std)
                } else {
                    0
                };
                let g = if channels > 1 {
                    denormalize_channel(slice[channel_stride + pixel_idx], 1, mean, std)
                } else {
                    r
                };
                let b = if channels > 2 {
                    denormalize_channel(slice[2 * channel_stride + pixel_idx], 2, mean, std)
                } else {
                    g
                };
                let out = pixel_idx.saturating_mul(3);
                rgb[out] = r;
                rgb[out + 1] = g;
                rgb[out + 2] = b;
            }
        }
        let image = RgbImage::from_vec(width as u32, height as u32, rgb)
            .ok_or_else(|| anyhow!("failed to build PNG for batch item {batch_idx}"))?;
        image.save(output_dir.join(format!("{:06}.png", offset + batch_idx)))?;
    }
    Ok(batch)
}

fn write_summary_markdown(summary: &VisionRacCheckpointEvalSummary) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# RAC Checkpoint Eval").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- Backend: `{}`", summary.backend).unwrap();
    writeln!(&mut out, "- Checkpoint: `{}`", summary.checkpoint.display()).unwrap();
    writeln!(&mut out, "- Export dir: `{}`", summary.export_dir.display()).unwrap();
    writeln!(&mut out, "- Split: `{}`", summary.split).unwrap();
    writeln!(&mut out, "- Teacher kind: `{}`", summary.teacher_kind).unwrap();
    writeln!(
        &mut out,
        "- Teacher state mapping: `{}`",
        summary.teacher_state_mapping
    )
    .unwrap();
    writeln!(
        &mut out,
        "- Teacher latent downsample: `{}`",
        summary.teacher_latent_downsample
    )
    .unwrap();
    writeln!(&mut out, "- Batches: `{}`", summary.batches).unwrap();
    writeln!(&mut out, "- Images: `{}`", summary.images).unwrap();
    writeln!(&mut out).unwrap();
    if summary.teacher_kind == "pooled_image" {
        writeln!(&mut out, "## Comparability Note").unwrap();
        writeln!(
            &mut out,
            "- This eval uses the repo's legacy `pooled_image` shortcut rather than the compact-latent RAC structure shown in the public nano implementation."
        )
        .unwrap();
        writeln!(
            &mut out,
            "- Smaller `teacher_latent_downsample` values make the reconstruction task much easier than a compact-latent RAC setting."
        )
        .unwrap();
        writeln!(&mut out).unwrap();
    }
    if summary.teacher_kind == "precomputed_latent" {
        writeln!(&mut out, "## Comparability Note").unwrap();
        writeln!(
            &mut out,
            "- This eval uses a compact precomputed latent teacher and a shared image-state rollout, which is intended to match the public RAC nano demonstration more closely than the older pooled-image shortcut."
        )
        .unwrap();
        writeln!(
            &mut out,
            "- This should be treated as `nano`-compatible scaffolding, not as a claim that the full paper training protocol is reproduced end-to-end."
        )
        .unwrap();
        writeln!(
            &mut out,
            "- `forward/` and `roundtrip/` exports are first-three-channel state projections unless an explicit teacher decoder is added; they are not guaranteed to be decoded RGB reconstructions."
        )
        .unwrap();
        writeln!(
            &mut out,
            "- Validation PSNR here is state-domain PSNR over the padded image-state tensor, not decoded-image PSNR or rFID."
        )
        .unwrap();
        writeln!(&mut out).unwrap();
    }
    writeln!(&mut out, "## Forward Decode").unwrap();
    writeln!(&mut out, "- recon loss: {:.6}", summary.forward_recon_loss).unwrap();
    writeln!(&mut out, "- PSNR: {:.6}", summary.forward_recon_psnr).unwrap();
    writeln!(&mut out, "- path loss: {:.6}", summary.forward_path_loss).unwrap();
    writeln!(
        &mut out,
        "- velocity loss: {:.6}",
        summary.forward_velocity_loss
    )
    .unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Roundtrip Reconstruction").unwrap();
    writeln!(
        &mut out,
        "- recon loss: {:.6}",
        summary.roundtrip_recon_loss
    )
    .unwrap();
    writeln!(&mut out, "- PSNR: {:.6}", summary.roundtrip_recon_psnr).unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Reverse Encode").unwrap();
    writeln!(
        &mut out,
        "- latent loss: {:.6}",
        summary.reverse_latent_loss
    )
    .unwrap();
    writeln!(
        &mut out,
        "- to-init loss: {:.6}",
        summary.reverse_to_init_loss
    )
    .unwrap();
    if let Some(semantic_loss) = summary.semantic_loss {
        writeln!(&mut out, "- semantic loss: {:.6}", semantic_loss).unwrap();
    }
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "## Exports").unwrap();
    writeln!(&mut out, "- real images: `real/`").unwrap();
    writeln!(&mut out, "- roundtrip reconstructions: `roundtrip/`").unwrap();
    writeln!(&mut out, "- forward decode images: `forward/`").unwrap();
    writeln!(&mut out).unwrap();
    if summary.teacher_kind == "pooled_image" {
        writeln!(
            &mut out,
            "Use `scripts/vision/eval_rfid.py` on `real/` vs `roundtrip/` only for legacy shortcut comparisons."
        )
        .unwrap();
    } else {
        writeln!(
            &mut out,
            "Do not interpret `real/` vs `roundtrip/` rFID as paper-aligned reconstruction quality unless an explicit teacher decoder is added to the export path."
        )
        .unwrap();
    }
    out
}

fn rac_batch_from_cifar_valid<B: BackendTrait>(batch: CifarBatch<B>) -> VisionRacBatch<B> {
    batch.into()
}

fn rac_batch_from_imagenet_valid<B: BackendTrait>(batch: ImageNetBatch<B>) -> VisionRacBatch<B> {
    batch.into()
}

pub fn eval_vision_rac_checkpoint_backend<B, Init>(
    config: &VisionTrainingConfig,
    checkpoint: &Path,
    backend_name: &str,
    init_backend: Init,
    output_dir: Option<&Path>,
    max_images: Option<usize>,
) -> Result<VisionRacCheckpointEvalSummary>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    if training.batch_size == 0 {
        return Err(anyhow!("vision RAC eval batch_size must be > 0"));
    }

    let vision_config = config.vision.build();
    let rac = match &config.mode {
        VisionTrainingModeConfig::Rac(rac) => rac.as_ref().clone(),
        _ => {
            return Err(anyhow!(
                "vision RAC checkpoint eval requires mode.type = \"rac\""
            ));
        }
    };
    maybe_download_vision_dataset(&config.dataset)?;

    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let valid_device = device.clone();
    let (valid_loader, num_classes): (
        Arc<dyn DataLoader<ValidBackend<B>, VisionRacBatch<ValidBackend<B>>>>,
        usize,
    ) = match config.dataset.source {
        VisionDatasetSource::Cifar10 | VisionDatasetSource::Cifar100 => {
            let cifar_type = if matches!(config.dataset.source, VisionDatasetSource::Cifar10) {
                CifarType::Cifar10
            } else {
                CifarType::Cifar100
            };
            let dataset = Arc::new(
                CifarDataset::new(&config.dataset.cifar_root, cifar_type, CifarSplit::Test)?
                    .with_max_records(config.dataset.max_records),
            );
            let valid_steps = dataset.steps_per_epoch(training.batch_size);
            (
                Arc::new(VisionRacBatchLoader::new(
                    Arc::new(CifarDataLoader::<ValidBackend<B>>::new(
                        Arc::clone(&dataset),
                        training.batch_size,
                        &valid_device,
                        valid_steps,
                        None,
                    )),
                    rac_batch_from_cifar_valid::<ValidBackend<B>>,
                )),
                if matches!(cifar_type, CifarType::Cifar10) {
                    10
                } else {
                    100
                },
            )
        }
        VisionDatasetSource::Imagenet => {
            let root = config.dataset.imagenet_root.join(&config.dataset.val_dir);
            let augment = &config.augment;
            let mut dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: ImageNetAugmentations::new(
                    ImageNetSplit::Val,
                    vision_config.image_size,
                    augment.resize_short,
                    augment.min_scale,
                    augment.max_scale,
                    augment.min_aspect_ratio,
                    augment.max_aspect_ratio,
                    augment.flip_prob,
                    augment.color_jitter_prob,
                    augment.brightness,
                    augment.contrast,
                    augment.saturation,
                    augment.hue,
                    augment.grayscale_prob,
                    augment.blur_prob,
                    augment.blur_sigma_min,
                    augment.blur_sigma_max,
                    augment.solarize_prob,
                    augment.solarize_threshold,
                ),
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views: 1,
                local_views: 0,
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?;
            if let Some(store) = build_rac_semantic_teacher_store(
                ImageNetSplit::Val,
                &rac,
                dataset.len(),
                config.dataset.cache_teacher_features_in_memory,
            )? {
                dataset = dataset.with_teacher(store);
            }
            if let Some(store) = build_rac_teacher_latent_store(
                ImageNetSplit::Val,
                &rac,
                dataset.len(),
                config.dataset.cache_teacher_features_in_memory,
            )? {
                dataset = dataset.with_rac_teacher_latent(store);
            }
            let dataset = Arc::new(dataset);
            let valid_steps = dataset.steps_per_epoch(training.batch_size);
            (
                Arc::new(VisionRacBatchLoader::new(
                    Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
                        Arc::clone(&dataset),
                        training.batch_size,
                        &valid_device,
                        valid_steps,
                        None,
                        config.dataset.prefetch_batches,
                        config.dataset.prefetch_workers,
                        false,
                    )),
                    rac_batch_from_imagenet_valid::<ValidBackend<B>>,
                )),
                infer_image_dataset_num_classes(config)?,
            )
        }
        other => {
            return Err(anyhow!(
                "vision RAC checkpoint eval does not support dataset source {other:?}"
            ));
        }
    };

    let checkpoint_base = resolve_checkpoint_base(checkpoint, None)?.0;
    let checkpoint_path = checkpoint_base.with_extension("bin");
    if !checkpoint_path.exists() {
        return Err(anyhow!(
            "checkpoint file {} not found",
            checkpoint_path.display()
        ));
    }
    let teacher_kind = rac.teacher.kind.to_string();
    let teacher_state_mapping = rac.teacher.state_mapping.to_string();
    let teacher_latent_downsample = rac.teacher.latent_downsample;
    let model = VisionDragon::<ValidBackend<B>>::new(vision_config.clone(), &valid_device);
    let mut valid_model =
        VisionRacModel::new(model, rac, &vision_config, num_classes, &valid_device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<VisionRacModel<ValidBackend<B>> as Module<ValidBackend<B>>>::Record>(
            checkpoint_base,
            &valid_device,
        )
        .with_context(|| format!("failed to load checkpoint {}", checkpoint_path.display()))?;
    valid_model = valid_model.load_record(record);

    let export_dir = match output_dir {
        Some(path) => create_unique_dir(path)?,
        None => default_export_dir(checkpoint)?,
    };
    let real_dir = export_dir.join("real");
    let roundtrip_dir = export_dir.join("roundtrip");
    let forward_dir = export_dir.join("forward");
    fs::create_dir_all(&real_dir)?;
    fs::create_dir_all(&roundtrip_dir)?;
    fs::create_dir_all(&forward_dir)?;

    let mut batches = 0usize;
    let mut images = 0usize;
    let mut forward_recon_loss = 0.0f64;
    let mut forward_recon_psnr = 0.0f64;
    let mut roundtrip_recon_loss = 0.0f64;
    let mut roundtrip_recon_psnr = 0.0f64;
    let mut forward_path_loss = 0.0f64;
    let mut forward_velocity_loss = 0.0f64;
    let mut reverse_latent_loss = 0.0f64;
    let mut reverse_to_init_loss = 0.0f64;
    let mut semantic_loss = 0.0f64;
    let mut semantic_batches = 0usize;
    let export_cap = max_images.unwrap_or(usize::MAX);

    for batch in valid_loader.iter() {
        if images >= export_cap {
            break;
        }
        let output = valid_model.eval_batch(batch, false);
        let batch_size = output.images.shape().dims::<4>()[0];
        let write_count = batch_size.min(export_cap.saturating_sub(images));

        let real_images = if write_count < batch_size {
            output.images.clone().slice([
                0..write_count,
                0..3,
                0..vision_config.image_size,
                0..vision_config.image_size,
            ])
        } else {
            output.images.clone()
        };
        let roundtrip_images = if write_count < batch_size {
            output.roundtrip_state.clone().slice([
                0..write_count,
                0..3,
                0..vision_config.image_size,
                0..vision_config.image_size,
            ])
        } else {
            output.roundtrip_state.clone()
        };
        let forward_images = if write_count < batch_size {
            output.forward_state.clone().slice([
                0..write_count,
                0..3,
                0..vision_config.image_size,
                0..vision_config.image_size,
            ])
        } else {
            output.forward_state.clone()
        };
        save_tensor_images(
            real_images,
            &real_dir,
            images,
            &config.augment.normalize_mean,
            &config.augment.normalize_std,
        )?;
        save_tensor_images(
            roundtrip_images,
            &roundtrip_dir,
            images,
            &config.augment.normalize_mean,
            &config.augment.normalize_std,
        )?;
        save_tensor_images(
            forward_images,
            &forward_dir,
            images,
            &config.augment.normalize_mean,
            &config.augment.normalize_std,
        )?;

        forward_recon_loss += tensor_scalar_f64(output.recon);
        forward_recon_psnr += tensor_scalar_f64(output.recon_psnr);
        roundtrip_recon_loss += tensor_scalar_f64(output.roundtrip);
        roundtrip_recon_psnr += tensor_scalar_f64(output.roundtrip_psnr);
        forward_path_loss += tensor_scalar_f64(output.forward_path);
        forward_velocity_loss += tensor_scalar_f64(output.forward_velocity);
        reverse_latent_loss += tensor_scalar_f64(output.reverse_latent);
        reverse_to_init_loss += tensor_scalar_f64(output.reverse_to_init);
        if let Some(value) = output.semantic_loss {
            semantic_loss += tensor_scalar_f64(value);
            semantic_batches += 1;
        }
        batches += 1;
        images += write_count;
    }

    if batches == 0 || images == 0 {
        return Err(anyhow!(
            "RAC checkpoint eval produced zero validation batches"
        ));
    }

    let inv_batches = 1.0 / batches as f64;
    let summary = VisionRacCheckpointEvalSummary {
        backend: backend_name.to_string(),
        checkpoint: checkpoint_path,
        export_dir: export_dir.clone(),
        split: "valid".to_string(),
        teacher_kind,
        teacher_state_mapping,
        teacher_latent_downsample,
        batches,
        images,
        forward_recon_loss: forward_recon_loss * inv_batches,
        forward_recon_psnr: forward_recon_psnr * inv_batches,
        roundtrip_recon_loss: roundtrip_recon_loss * inv_batches,
        roundtrip_recon_psnr: roundtrip_recon_psnr * inv_batches,
        forward_path_loss: forward_path_loss * inv_batches,
        forward_velocity_loss: forward_velocity_loss * inv_batches,
        reverse_latent_loss: reverse_latent_loss * inv_batches,
        reverse_to_init_loss: reverse_to_init_loss * inv_batches,
        semantic_loss: (semantic_batches > 0).then_some(semantic_loss / semantic_batches as f64),
    };

    fs::write(
        export_dir.join("rac_checkpoint_eval_summary.md"),
        write_summary_markdown(&summary),
    )?;
    fs::write(
        export_dir.join("rac_checkpoint_eval_summary.json"),
        serde_json::to_string_pretty(&summary)?,
    )?;
    Ok(summary)
}
