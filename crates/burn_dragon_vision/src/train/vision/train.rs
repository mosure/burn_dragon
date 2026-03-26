use super::distill_runtime::{DistillDatasetRequest, build_distill_datasets_and_teacher};
use super::video::dataset::MovingMnistVideoLoaderConfig;
use super::write_rac_best_checkpoint_report;
use crate::model::{stage_aware_host_profile_reset, stage_aware_host_profile_snapshot};
use crate::train::prelude::*;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn::record::Recorder;
use burn_dragon_kernel::api::spatial::{
    structured_pyramid_profile_reset, structured_pyramid_profile_snapshot,
};
use burn_dragon_train::train::pipeline::activate_planned_run;
use std::time::Instant;

mod rac_backend;
mod video_lejepa_backend;
mod video_vjepa21_backend;

use rac_backend::train_rac_backend;
use video_lejepa_backend::train_video_lejepa_backend;
use video_vjepa21_backend::train_video_vjepa21_backend;

pub(crate) struct VisionRacBatchLoader<B, SrcBatch>
where
    B: BackendTrait + 'static,
    B::Device: Clone,
    SrcBatch: Send + 'static,
{
    inner: Arc<dyn DataLoader<B, SrcBatch>>,
    map: fn(SrcBatch) -> VisionRacBatch<B>,
}

impl<B, SrcBatch> VisionRacBatchLoader<B, SrcBatch>
where
    B: BackendTrait + 'static,
    B::Device: Clone,
    SrcBatch: Send + 'static,
{
    pub(crate) fn new(
        inner: Arc<dyn DataLoader<B, SrcBatch>>,
        map: fn(SrcBatch) -> VisionRacBatch<B>,
    ) -> Self {
        Self { inner, map }
    }
}

struct VisionRacBatchIterator<'a, B, SrcBatch>
where
    B: BackendTrait + 'static,
    SrcBatch: Send + 'static,
{
    inner: Box<dyn DataLoaderIterator<SrcBatch> + 'a>,
    map: fn(SrcBatch) -> VisionRacBatch<B>,
}

impl<B, SrcBatch> Iterator for VisionRacBatchIterator<'_, B, SrcBatch>
where
    B: BackendTrait + 'static,
    SrcBatch: Send + 'static,
{
    type Item = VisionRacBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next().map(self.map)
    }
}

impl<B, SrcBatch> DataLoaderIterator<VisionRacBatch<B>> for VisionRacBatchIterator<'_, B, SrcBatch>
where
    B: BackendTrait + 'static,
    SrcBatch: Send + 'static,
{
    fn progress(&self) -> Progress {
        self.inner.progress()
    }
}

impl<B, SrcBatch> DataLoader<B, VisionRacBatch<B>> for VisionRacBatchLoader<B, SrcBatch>
where
    B: BackendTrait + 'static,
    B::Device: Clone,
    SrcBatch: Send + 'static,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<VisionRacBatch<B>> + 'a> {
        Box::new(VisionRacBatchIterator {
            inner: self.inner.iter(),
            map: self.map,
        })
    }

    fn num_items(&self) -> usize {
        self.inner.num_items()
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, VisionRacBatch<B>>> {
        Arc::new(Self::new(self.inner.to_device(device), self.map))
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, VisionRacBatch<B>>> {
        Arc::new(Self::new(self.inner.slice(start, end), self.map))
    }
}

fn rac_batch_from_cifar<B: BackendTrait>(batch: CifarBatch<B>) -> VisionRacBatch<B> {
    batch.into()
}

fn rac_batch_from_imagenet<B: BackendTrait>(batch: ImageNetBatch<B>) -> VisionRacBatch<B> {
    batch.into()
}

pub fn train_vision_backend<B, Init>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_vision_backend_with_context::<B, Init>(config, &[], None, backend_name, init_backend)
}

pub fn train_vision_backend_with_config_paths<B, Init>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_vision_backend_with_context::<B, Init>(
        config,
        config_paths,
        None,
        backend_name,
        init_backend,
    )
}

pub fn train_vision_backend_with_planned_run<B, Init>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: PlannedRunArtifacts,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_vision_backend_with_context::<B, Init>(
        config,
        config_paths,
        Some(planned_run),
        backend_name,
        init_backend,
    )
}

fn train_vision_backend_with_context<B, Init>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: Option<PlannedRunArtifacts>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let stage_profile = crate::train::profile::enabled();
    if stage_profile {
        crate::train::profile::reset();
        stage_aware_host_profile_reset();
        structured_pyramid_profile_reset();
    }
    let train_wall_start = stage_profile.then(Instant::now);

    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;
    if training.batch_size == 0 {
        return Err(anyhow!("vision training batch_size must be > 0"));
    }

    let vision_config = config.vision.build();
    if vision_config.patch_size == 0 {
        return Err(anyhow!("vision.patch_size must be > 0"));
    }
    if config.augment.image_size != vision_config.image_size {
        return Err(anyhow!(
            "augment.image_size ({}) must match vision.image_size ({})",
            config.augment.image_size,
            vision_config.image_size
        ));
    }
    let rollout = resolve_vision_rollout(training, vision_config.steps)?;
    info!(
        "vision rollout steps: min={}, max={}, backprop={}",
        rollout.min_steps, rollout.max_steps, rollout.backprop_steps
    );

    maybe_download_vision_dataset(&config.dataset)?;

    if let VisionTrainingModeConfig::Rac(rac) = &config.mode {
        return train_rac_backend::<B>(
            config,
            config_paths,
            planned_run,
            backend_name,
            &device,
            &vision_config,
            rac,
            optimizer_cfg,
        );
    }

    if let VisionTrainingModeConfig::VideoLejepa(video) = &config.mode {
        if video.is_vjepa21() {
            return train_video_vjepa21_backend::<B>(
                config,
                config_paths,
                planned_run.clone(),
                backend_name,
                &device,
                &vision_config,
                video,
                optimizer_cfg,
            );
        }
        return train_video_lejepa_backend::<B>(
            config,
            config_paths,
            planned_run,
            backend_name,
            &device,
            &vision_config,
            video,
            rollout,
            optimizer_cfg,
        );
    }

    let grid = vision_config.image_size.div_ceil(vision_config.patch_size);
    let student_patch_tokens = grid * grid;

    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_aug = ImageNetAugmentations::new(
        ImageNetSplit::Train,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let val_aug = ImageNetAugmentations::new(
        ImageNetSplit::Val,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );

    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);

    enum VisionMode<B: BackendTrait> {
        Distill {
            teacher: Option<Box<crate::train::vision::models::DistillTeacherModel<B>>>,
        },
        Lejepa {
            config: VisionLejepaConfig,
        },
        Mae {
            config: VisionMaeConfig,
        },
        Saccade {
            config: Box<VisionSaccadeConfig>,
        },
    }

    let (train_dataset, val_dataset, mode) = match &config.mode {
        VisionTrainingModeConfig::Distill(_distill) => {
            let (train_dataset, val_dataset, teacher) =
                build_distill_datasets_and_teacher::<B>(DistillDatasetRequest {
                    config,
                    vision_config: &vision_config,
                    normalize,
                    train_aug,
                    val_aug,
                    train_root: &train_root,
                    val_root: &val_root,
                    student_patch_tokens,
                    device: &device,
                })?;

            (
                train_dataset,
                val_dataset,
                VisionMode::Distill {
                    teacher: teacher.map(Box::new),
                },
            )
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            let multi_crop = lejepa.global_views > 0 || lejepa.local_views > 0;
            let global_views = if multi_crop {
                lejepa.global_views.max(1)
            } else {
                lejepa.views.max(1)
            };
            let local_views = if multi_crop { lejepa.local_views } else { 0 };
            if global_views + local_views == 0 {
                return Err(anyhow!(
                    "lejepa must have at least one global or local view"
                ));
            }
            if local_views > 0 {
                if lejepa.local_image_size == 0 {
                    return Err(anyhow!("lejepa.local_image_size must be > 0"));
                }
                if lejepa.local_image_size % vision_config.patch_size != 0 {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be divisible by patch_size ({})",
                        lejepa.local_image_size,
                        vision_config.patch_size
                    ));
                }
                if lejepa.local_image_size > vision_config.image_size {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be <= vision.image_size ({})",
                        lejepa.local_image_size,
                        vision_config.image_size
                    ));
                }
            }
            if lejepa.loss.recon.weight < 0.0 {
                return Err(anyhow!("lejepa.loss.recon.weight must be >= 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "lejepa.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    lejepa.loss.recon.mask_ratio
                ));
            }
            if lejepa.loss.lejepa.enabled {
                if !(0.0..=1.0).contains(&lejepa.loss.lejepa.lambda) {
                    return Err(anyhow!(
                        "lejepa.loss.lejepa.lambda must be in [0, 1] (got {})",
                        lejepa.loss.lejepa.lambda
                    ));
                }
                if lejepa.loss.lejepa.sigreg_knots == 0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_knots must be > 0"));
                }
                if lejepa.loss.lejepa.sigreg_t_max <= 0.0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_t_max must be > 0"));
                }
                if lejepa.loss.lejepa.sigreg_proj_dim == 0 {
                    return Err(anyhow!("lejepa.loss.lejepa.sigreg_proj_dim must be > 0"));
                }
            }
            let local_train_aug = if local_views > 0 {
                Some(ImageNetAugmentations::new(
                    ImageNetSplit::Train,
                    lejepa.local_image_size,
                    lejepa.local_image_size,
                    lejepa.local_min_scale,
                    lejepa.local_max_scale,
                    config.augment.min_aspect_ratio,
                    config.augment.max_aspect_ratio,
                    config.augment.flip_prob,
                    config.augment.color_jitter_prob,
                    config.augment.brightness,
                    config.augment.contrast,
                    config.augment.saturation,
                    config.augment.hue,
                    config.augment.grayscale_prob,
                    config.augment.blur_prob,
                    config.augment.blur_sigma_min,
                    config.augment.blur_sigma_max,
                    config.augment.solarize_prob,
                    config.augment.solarize_threshold,
                ))
            } else {
                None
            };
            let local_val_aug = if local_views > 0 {
                Some(ImageNetAugmentations::new(
                    ImageNetSplit::Val,
                    lejepa.local_image_size,
                    lejepa.local_image_size,
                    lejepa.local_min_scale,
                    lejepa.local_max_scale,
                    config.augment.min_aspect_ratio,
                    config.augment.max_aspect_ratio,
                    config.augment.flip_prob,
                    config.augment.color_jitter_prob,
                    config.augment.brightness,
                    config.augment.contrast,
                    config.augment.saturation,
                    config.augment.hue,
                    config.augment.grayscale_prob,
                    config.augment.blur_prob,
                    config.augment.blur_sigma_min,
                    config.augment.blur_sigma_max,
                    config.augment.solarize_prob,
                    config.augment.solarize_threshold,
                ))
            } else {
                None
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: local_train_aug.clone(),
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views: global_views,
                local_views,
                min_view_overlap: lejepa.min_view_overlap,
                view_overlap_attempts: lejepa.view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: local_val_aug.clone(),
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views: global_views,
                local_views,
                min_view_overlap: lejepa.min_view_overlap,
                view_overlap_attempts: lejepa.view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Lejepa {
                    config: lejepa.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Mae(mae) => {
            if !(0.0..=1.0).contains(&mae.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "mae.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    mae.loss.recon.mask_ratio
                ));
            }
            if mae.loss.recon.weight < 0.0 {
                return Err(anyhow!("mae.loss.recon.weight must be >= 0"));
            }
            if mae.pyramid_levels == 0 {
                return Err(anyhow!("mae.pyramid_levels must be > 0"));
            }
            let views = if mae.cross_view.enabled {
                config.vision.num_eyes.max(1)
            } else {
                1
            };
            let min_view_overlap = if mae.cross_view.enabled {
                mae.cross_view.min_overlap.max(0.0)
            } else {
                0.0
            };
            let view_overlap_attempts = if mae.cross_view.enabled {
                mae.cross_view.max_attempts.max(1)
            } else {
                1
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Mae {
                    config: mae.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Rac(_) => {
            unreachable!("rac mode should be dispatched before the generic image trainer");
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            let mut saccade = (**saccade).clone();
            if saccade.num_eyes == 0 {
                saccade.num_eyes = config.vision.num_eyes.max(1);
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.loss.recon.mask_ratio) {
                return Err(anyhow!(
                    "saccade.loss.recon.mask_ratio must be in [0, 1] (got {})",
                    saccade.loss.recon.mask_ratio
                ));
            }
            if saccade.loss.recon.weight < 0.0 {
                return Err(anyhow!("saccade.loss.recon.weight must be >= 0"));
            }
            if saccade.loss.lejepa.enabled {
                if !(0.0..=1.0).contains(&saccade.loss.lejepa.lambda) {
                    return Err(anyhow!(
                        "saccade.loss.lejepa.lambda must be in [0, 1] (got {})",
                        saccade.loss.lejepa.lambda
                    ));
                }
                if saccade.loss.lejepa.sigreg_knots == 0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_knots must be > 0"));
                }
                if saccade.loss.lejepa.sigreg_t_max <= 0.0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_t_max must be > 0"));
                }
                if saccade.loss.lejepa.sigreg_proj_dim == 0 {
                    return Err(anyhow!("saccade.loss.lejepa.sigreg_proj_dim must be > 0"));
                }
            }
            if saccade.policy.info_reward.stride == 0 {
                return Err(anyhow!("saccade.policy.info_reward.stride must be > 0"));
            }
            if saccade.policy.location_embedding.quantize_bins < 2 {
                return Err(anyhow!(
                    "saccade.policy.location_embedding.quantize_bins must be >= 2"
                ));
            }
            if saccade.policy.gdpo.enabled {
                if saccade.policy.gdpo.group_size == 0 {
                    return Err(anyhow!("saccade.policy.gdpo.group_size must be > 0"));
                }
                if saccade.policy.action_noise_std <= 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.action_noise_std must be > 0 when gdpo is enabled"
                    ));
                }
                if saccade.policy.gdpo.hard_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.hard_weight must be >= 0"));
                }
                if saccade.policy.gdpo.easy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.easy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_weight < 0.0 {
                    return Err(anyhow!("saccade.policy.gdpo.policy_weight must be >= 0"));
                }
                if saccade.policy.gdpo.policy_clip_range < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.policy_clip_range must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.advantage_clip < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_clip must be >= 0 (got {})",
                        saccade.policy.gdpo.advantage_clip
                    ));
                }
                if !(0.0..1.0).contains(&saccade.policy.gdpo.advantage_ema_decay) {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.advantage_ema_decay must be in [0, 1) (got {})",
                        saccade.policy.gdpo.advantage_ema_decay
                    ));
                }
                match saccade.policy.gdpo.hard_gate {
                    GdpoHardGate::Off => {}
                    GdpoHardGate::Fixed { .. } => {}
                    GdpoHardGate::Percentile { quantile } => {
                        if !(0.0..=1.0).contains(&quantile) {
                            return Err(anyhow!(
                                "saccade.policy.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                                quantile
                            ));
                        }
                    }
                }
            }
            let views = if saccade.cross_view.enabled {
                saccade.num_eyes.max(1)
            } else {
                1
            };
            let min_view_overlap = if saccade.cross_view.enabled {
                saccade.cross_view.min_overlap.max(0.0)
            } else {
                0.0
            };
            let view_overlap_attempts = if saccade.cross_view.enabled {
                saccade.cross_view.max_attempts.max(1)
            } else {
                1
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views,
                local_views: 0,
                min_view_overlap,
                view_overlap_attempts,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Saccade {
                    config: Box::new(saccade.clone()),
                },
            )
        }
        VisionTrainingModeConfig::VideoLejepa(_) => {
            unreachable!("video LEJEPA is handled by the dedicated early-return path")
        }
    };

    let steps_per_epoch = train_dataset.steps_per_epoch(training.batch_size);
    let schedule = resolve_vision_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "vision schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );

    let prefetch_to_device = config.dataset.prefetch_to_device;

    let train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>> =
        Arc::new(ImageNetDataLoader::<B>::new(
            Arc::clone(&train_dataset),
            training.batch_size,
            &device,
            steps_per_epoch,
            Some(total_steps),
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            prefetch_to_device,
        ));

    let val_steps_per_epoch = val_dataset.steps_per_epoch(training.batch_size);
    let valid_steps =
        resolve_valid_steps_per_epoch(total_steps, training.log_frequency, val_steps_per_epoch);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>> =
        Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&val_dataset),
            training.batch_size,
            &valid_device,
            valid_steps,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            prefetch_to_device,
        ));

    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_vision_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &vision_config)?;

    let planned_run = resolve_vision_run_artifacts(config, config_paths, planned_run)?;
    activate_planned_run(&planned_run)?;
    let run_dir = planned_run.run_dir;
    let run_name = planned_run.run_name;
    crate::write_training_snapshot(config, &run_dir)?;
    info!("vision run name: {run_name}");
    info!(
        "vision training batching: micro_batch_size={} gradient_accumulation_steps={} effective_batch_size={}",
        training.batch_size,
        training.gradient_accumulation_steps,
        training
            .batch_size
            .saturating_mul(training.gradient_accumulation_steps.max(1))
    );
    let context = VisionTrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };

    match mode {
        VisionMode::Distill { teacher } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let teacher = teacher.map(|teacher| *teacher);
            let distill = match &config.mode {
                VisionTrainingModeConfig::Distill(distill) => distill.clone(),
                _ => unreachable!("distill mode branch should only run for distill configs"),
            };
            let mut distill_model =
                VisionDistillModel::new(model, distill.clone(), teacher, rollout, &device);
            if let Some(checkpoint) = distill.student_checkpoint.as_ref() {
                let (checkpoint_base, epoch) =
                    crate::checkpoint::resolve_checkpoint_base(checkpoint, None)?;
                let record = BinFileRecorder::<FullPrecisionSettings>::new()
                    .load::<<VisionDistillModel<B> as Module<B>>::Record>(
                        checkpoint_base.clone(),
                        &device,
                    )
                    .map_err(|err| {
                        anyhow!(burn_dragon_checkpoint::format_checkpoint_load_error(
                            &checkpoint_base,
                            err
                        ))
                    })?;
                distill_model = distill_model.load_record(record);
                info!(
                    "loaded distill student warm start from {} (epoch {})",
                    checkpoint.display(),
                    epoch
                );
            }
            let mut model = Some(distill_model);
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionDistillModel<B>>());
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "distill".to_string(),
                inv: false,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: false,
                recon: false,
                directional: false,
                policy: false,
                probe: false,
                distill: true,
                distill_rollout: true,
                artifact_every: 0,
                artifact_output: VisionArtifactOutputMode::Images,
                artifact_overwrite: false,
                artifact_max_images: 0,
                artifact_fps: 1,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics,
                )?,
                ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics,
                )?,
            }
        }
        VisionMode::Lejepa { config: lejepa } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let recon = VisionReconstructionInit {
                patch_dim: recon_patch_dim,
                normalize_std: config.augment.normalize_std,
                patch_size: vision_config.patch_size,
                in_channels: vision_config.in_channels,
            };
            let mut model = Some(VisionLejepaModel::new(
                model,
                lejepa,
                VisionLejepaInit {
                    embed_dim: vision_config.embed_dim,
                    num_classes: train_dataset.num_classes(),
                    rollout,
                    recon,
                    normalization: vision_config.normalization.clone(),
                },
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionLejepaModel<B>>());
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "lejepa".to_string(),
                distill: false,
                distill_rollout: false,
                inv: model.as_ref().expect("model").config.loss.lejepa.enabled,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: model.as_ref().expect("model").config.loss.lejepa.enabled,
                recon: model.as_ref().expect("model").config.loss.recon.weight > 0.0,
                directional: false,
                policy: false,
                probe: true,
                artifact_every: model.as_ref().expect("model").config.artifact_every,
                artifact_output: model.as_ref().expect("model").config.artifact_output,
                artifact_overwrite: model.as_ref().expect("model").config.artifact_overwrite,
                artifact_max_images: model.as_ref().expect("model").config.artifact_max_images,
                artifact_fps: model.as_ref().expect("model").config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Mae { config: mae } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let recon = VisionReconstructionInit {
                patch_dim: recon_patch_dim,
                normalize_std: config.augment.normalize_std,
                patch_size: vision_config.patch_size,
                in_channels: vision_config.in_channels,
            };
            let mut model = Some(VisionMaeModel::new(
                model,
                mae,
                VisionMaeInit {
                    num_eyes: vision_config.num_eyes,
                    embed_dim: vision_config.embed_dim,
                    rollout,
                    recon,
                    normalization: vision_config.normalization.clone(),
                },
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionMaeModel<B>>());
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "mae".to_string(),
                distill: false,
                distill_rollout: false,
                inv: false,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: false,
                recon: model_ref.config.loss.recon.weight > 0.0,
                directional: false,
                policy: false,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Saccade { config: saccade } => {
            let model = VisionDragon::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionSaccadeModel::new(
                model,
                *saccade,
                vision_config.embed_dim,
                vision_config.patch_size,
                rollout,
                recon_patch_dim,
                training.batch_repeats,
                training.train_repeat_chunk,
                &device,
            ));
            let mut optim =
                Some(adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionSaccadeModel<B>>());
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "saccade".to_string(),
                distill: false,
                distill_rollout: false,
                inv: model_ref.config.loss.lejepa.enabled,
                observe: false,
                mode_separation: false,
                rollout_horizon_metrics: false,
                sigreg: model_ref.config.loss.lejepa.enabled,
                recon: model_ref.config.loss.recon.weight > 0.0,
                directional: false,
                policy: model_ref.config.policy.gdpo.enabled,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
                ffmpeg_path: training.ffmpeg_path.clone(),
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::BitNetTwoStage(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
    }

    info!("Vision training complete on {backend_name}");
    if let Some(start) = train_wall_start {
        let elapsed_ns = start.elapsed().as_nanos();
        let snapshot = crate::train::profile::snapshot();
        let structured = structured_pyramid_profile_snapshot();
        let stage_aware = stage_aware_host_profile_snapshot();
        info!(
            "[stage-profile][training] total_ns={elapsed_ns} dataloader_cpu_ns={} dataloader_image_load_ns={} dataloader_image_transform_ns={} dataloader_teacher_load_ns={} dataloader_tensor_copy_ns={} dataloader_host_to_device_copy_bytes={} host_sync_points={} forward_ns={} loss_backward_ns={} optimizer_ns={} train_steps={} optimizer_steps={}",
            snapshot.dataloader_cpu_ns,
            snapshot.dataloader_image_load_ns,
            snapshot.dataloader_image_transform_ns,
            snapshot.dataloader_teacher_load_ns,
            snapshot.dataloader_tensor_copy_ns,
            snapshot.dataloader_host_to_device_copy_bytes,
            snapshot.host_sync_points,
            snapshot.forward_ns,
            snapshot.loss_backward_ns,
            snapshot.optimizer_ns,
            snapshot.train_steps,
            snapshot.optimizer_steps,
        );
        info!(
            "[stage-profile][training-structured] calls={} launches={} total_ns={} setup_ns={} copy_ns={} dispatch_ns={} transient_allocations={} metadata_upload_bytes={} metadata_reuse_hits={} metadata_reuse_bytes={} resident_rollout_steps={}",
            structured.calls,
            structured.launches,
            structured.total_ns,
            structured.setup_ns,
            structured.copy_ns,
            structured.dispatch_ns,
            structured.transient_allocations,
            structured.metadata_upload_bytes,
            structured.metadata_reuse_hits,
            structured.metadata_reuse_bytes,
            structured.resident_rollout_steps,
        );
        info!(
            "[stage-profile][training-stageaware] step_calls={} coarse_only_step_calls={} patch_local_ns={} coarse_local_ns={} patch_from_coarse_ns={} hub_read_ns={} patch_to_coarse_ns={} hub_update_ns={}",
            stage_aware.step_calls,
            stage_aware.coarse_only_step_calls,
            stage_aware.patch_local_ns,
            stage_aware.coarse_local_ns,
            stage_aware.patch_from_coarse_ns,
            stage_aware.hub_read_ns,
            stage_aware.patch_to_coarse_ns,
            stage_aware.hub_update_ns,
        );
    }

    Ok(())
}

fn resolve_vision_run_artifacts(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    planned_run: Option<PlannedRunArtifacts>,
) -> Result<PlannedRunArtifacts> {
    match planned_run {
        Some(planned_run) => Ok(planned_run),
        None => {
            let run_root =
                resolve_run_root_for_config_paths("vision", &config.run_layout, config_paths);
            plan_run_artifacts(&run_root, None)
        }
    }
}

#[cfg(feature = "integration_test")]
pub fn train_vision_backend_for_test<B, Init>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    train_vision_backend::<B, Init>(config, backend_name, init_backend)
}
