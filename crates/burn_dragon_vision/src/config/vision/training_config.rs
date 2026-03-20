use super::*;
use crate::loss::VisionDistillationLossConfig;
use burn_dragon_train::LearningRateScheduleConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SiglipTeacherSurface {
    GlobalOnly224,
    Spatial224,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionTrainingConfig {
    pub dataset: VisionDatasetConfig,
    pub training: VisionTrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    pub vision: VisionModelConfig,
    #[serde(default)]
    pub augment: VisionAugmentationConfig,
    #[serde(default)]
    pub mode: VisionTrainingModeConfig,
}

impl VisionTrainingConfig {
    /// Construct the matched scene-slot graph control used in the broader Imagenette sweeps.
    pub fn scene_slot_graph_imagenette_baseline() -> Self {
        Self::scene_slot_graph_imagenette_common(VisionModelConfig::scene_slot_graph_baseline_224())
    }

    /// Construct the promoted local-bridge scene-slot graph baseline used in the broader
    /// Imagenette sweeps.
    pub fn scene_slot_graph_bridge_imagenette_baseline() -> Self {
        Self::scene_slot_graph_imagenette_common(
            VisionModelConfig::scene_slot_graph_bridge_baseline_224(),
        )
    }

    /// Construct the medium-width ImageNet-1k multi-teacher graph-bridge launch config.
    pub fn scene_slot_graph_bridge_multiteacher_imagenet1k_medium_launch() -> Self {
        Self::scene_slot_graph_multiteacher_imagenet1k_common(
            VisionModelConfig::scene_slot_graph_bridge_multiteacher_medium_280(),
            280,
            128,
            180_000,
            SiglipTeacherSurface::GlobalOnly224,
        )
    }

    /// Construct the base-width ImageNet-1k multi-teacher graph-bridge launch config.
    pub fn scene_slot_graph_bridge_multiteacher_imagenet1k_base_launch() -> Self {
        Self::scene_slot_graph_multiteacher_imagenet1k_common(
            VisionModelConfig::scene_slot_graph_bridge_multiteacher_base_336(),
            336,
            96,
            220_000,
            SiglipTeacherSurface::GlobalOnly224,
        )
    }

    /// Construct the promoted medium-width ImageNet-1k multi-mode graph-bridge launch config.
    ///
    /// This is the preferred frontier recipe when full SigLIP2 spatial features are available.
    pub fn scene_slot_graph_bridge_multimode_spatial_imagenet1k_medium_launch() -> Self {
        Self::scene_slot_graph_multiteacher_imagenet1k_common(
            VisionModelConfig::scene_slot_graph_bridge_multiteacher_medium_280(),
            280,
            128,
            180_000,
            SiglipTeacherSurface::Spatial224,
        )
    }

    /// Construct the promoted base-width ImageNet-1k multi-mode graph-bridge launch config.
    ///
    /// This is the preferred frontier recipe when full SigLIP2 spatial features are available.
    pub fn scene_slot_graph_bridge_multimode_spatial_imagenet1k_base_launch() -> Self {
        Self::scene_slot_graph_multiteacher_imagenet1k_common(
            VisionModelConfig::scene_slot_graph_bridge_multiteacher_base_336(),
            336,
            96,
            220_000,
            SiglipTeacherSurface::Spatial224,
        )
    }

    fn scene_slot_graph_imagenette_common(vision: VisionModelConfig) -> Self {
        let dataset = VisionDatasetConfig {
            imagenet_root: "data/imagenette2-160".into(),
            train_dir: "train".into(),
            val_dir: "val".into(),
            download: Some(VisionDatasetDownloadConfig::Imagenette {
                variant: ImagenetteVariant::Imagenette2_160,
            }),
            prefetch_batches: 8,
            prefetch_workers: 6,
            prefetch_to_device: true,
            cache_decoded: true,
            cache_capacity: 512,
            cache_preprocessed: true,
            ..VisionDatasetConfig::default()
        };
        let training = VisionTrainingHyperparameters {
            batch_size: 64,
            max_iters: 20_000,
            log_frequency: 200,
            rollout_min_steps: Some(2),
            rollout_max_steps: Some(vision.steps),
            rollout_backprop_steps: Some(vision.steps),
            ..VisionTrainingHyperparameters::default()
        };
        let optimizer = OptimizerConfig {
            learning_rate: 5e-4,
            weight_decay: 0.05,
            lr_schedule: Some(LearningRateScheduleConfig::Constant { initial_lr: None }),
            grad_clip_norm: Some(1.0),
            grad_clip_value: None,
        };
        let mode = VisionTrainingModeConfig::Lejepa(VisionLejepaConfig {
            views: 4,
            global_views: 2,
            local_views: 2,
            local_image_size: 96,
            local_min_scale: 0.05,
            local_max_scale: 0.3,
            min_view_overlap: 0.2,
            view_overlap_attempts: 8,
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 8,
            artifact_every: 1,
            artifact_max_images: 4,
            artifact_max_views: 3,
            artifact_overwrite: false,
            loss: VisionLossConfig {
                lejepa: VisionLejepaLossConfig::default(),
                recon: VisionReconLossConfig {
                    weight: 1.0,
                    mask_ratio: 0.6,
                    loss_on_all_patches: false,
                    loss_on_all_patches_valid_only: false,
                    recon_head_norm: true,
                    hidden_dim: 256,
                },
            },
            ..VisionLejepaConfig::default()
        });

        Self {
            dataset,
            training,
            optimizer,
            wgpu: WgpuRuntimeConfig::default(),
            vision,
            augment: VisionAugmentationConfig::default(),
            mode,
        }
    }

    fn scene_slot_graph_multiteacher_imagenet1k_common(
        vision: VisionModelConfig,
        image_size: usize,
        batch_size: usize,
        max_iters: usize,
        siglip_surface: SiglipTeacherSurface,
    ) -> Self {
        let patch_grid = image_size.div_ceil(vision.patch_size.max(1)).max(1);
        let patch_tokens = patch_grid * patch_grid;
        let resize_short = ((image_size as f32) * (256.0 / 224.0)).round() as usize;
        let dataset = VisionDatasetConfig {
            imagenet_root: "data/imagenet1k".into(),
            train_dir: "train".into(),
            val_dir: "val".into(),
            prefetch_batches: 8,
            prefetch_workers: 12,
            prefetch_to_device: true,
            cache_decoded: true,
            cache_capacity: 2048,
            cache_preprocessed: false,
            cache_teacher_features_in_memory: false,
            ..VisionDatasetConfig::default()
        };
        let training = VisionTrainingHyperparameters {
            batch_size,
            gradient_accumulation_steps: 2,
            epochs: Some(90),
            max_iters,
            log_frequency: 100,
            device_memory_check_every: 25,
            max_device_memory_mb: 96 * 1024,
            rollout_min_steps: Some(2),
            rollout_max_steps: Some(vision.steps),
            rollout_backprop_steps: Some(vision.steps),
            ..VisionTrainingHyperparameters::default()
        };
        let optimizer = OptimizerConfig {
            learning_rate: 3e-4,
            weight_decay: 0.05,
            lr_schedule: Some(LearningRateScheduleConfig::Cosine {
                initial_lr: None,
                min_lr: Some(3e-5),
                num_iters: Some(max_iters),
            }),
            grad_clip_norm: Some(1.0),
            grad_clip_value: None,
        };
        let siglip_target = match siglip_surface {
            SiglipTeacherSurface::GlobalOnly224 => VisionTeacherTargetConfig {
                name: "siglip2_global".to_string(),
                weight: 0.35,
                target_kind: VisionTeacherTargetKind::GlobalOnly,
                decoder_mode: VisionTeacherDecoderMode::DedicatedProjection,
                decoder_hidden_dim: Some(1536),
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224/train_cls.bin",
                    ),
                    train_patch_path: None,
                    val_cls_path: PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224/val_cls.bin",
                    ),
                    val_patch_path: None,
                    feature_dim: 768,
                    patch_tokens: None,
                }),
            },
            SiglipTeacherSurface::Spatial224 => VisionTeacherTargetConfig {
                name: "siglip2_spatial".to_string(),
                weight: 0.35,
                target_kind: VisionTeacherTargetKind::PatchAndCls,
                decoder_mode: VisionTeacherDecoderMode::DedicatedSpatialProjection,
                decoder_hidden_dim: Some(1536),
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224_spatial/train_cls.bin",
                    ),
                    train_patch_path: Some(PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224_spatial/train_patch.bin",
                    )),
                    val_cls_path: PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224_spatial/val_cls.bin",
                    ),
                    val_patch_path: Some(PathBuf::from(
                        "data/imagenet1k/features/siglip2_base_224_spatial/val_patch.bin",
                    )),
                    feature_dim: 768,
                    patch_tokens: Some(14 * 14),
                }),
            },
        };
        let mode = VisionTrainingModeConfig::Distill(VisionDistillConfig {
            teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                train_cls_path: format!(
                    "data/imagenet1k/features/dinov2_base_{image_size}/train_cls.bin"
                )
                .into(),
                train_patch_path: Some(
                    format!("data/imagenet1k/features/dinov2_base_{image_size}/train_patch.bin")
                        .into(),
                ),
                val_cls_path: format!(
                    "data/imagenet1k/features/dinov2_base_{image_size}/val_cls.bin"
                )
                .into(),
                val_patch_path: Some(
                    format!("data/imagenet1k/features/dinov2_base_{image_size}/val_patch.bin")
                        .into(),
                ),
                feature_dim: 768,
                patch_tokens: Some(patch_tokens),
            }),
            teacher_targets: vec![siglip_target],
            student_checkpoint: None,
            loss: VisionDistillationLossConfig {
                patch_mse_weight: 1.0,
                cls_mse_weight: 0.25,
                cls_cosine_weight: 1.0,
                rel_weight: 0.1,
                rel_tau: 0.07,
                rel_sample_tokens: Some(128),
            },
            rollout_supervision_frames: 2,
            rollout_supervision_stride: 1,
            rollout_supervision_groups: 1,
            rollout_supervision_explicit_steps: Vec::new(),
            rollout_supervision_explicit_groups: Vec::new(),
            rollout_supervision_include_step1: true,
            rollout_supervision_power: 1.0,
            rollout_sampling_power: 0.5,
            rollout_improvement_weight: 0.0,
            rollout_improvement_margin: 0.0,
        });
        let augment = VisionAugmentationConfig {
            image_size,
            resize_short,
            min_scale: 1.0,
            max_scale: 1.0,
            min_aspect_ratio: 1.0,
            max_aspect_ratio: 1.0,
            flip_prob: 0.0,
            color_jitter_prob: 0.0,
            brightness: 0.0,
            contrast: 0.0,
            saturation: 0.0,
            hue: 0.0,
            grayscale_prob: 0.0,
            blur_prob: 0.0,
            blur_sigma_min: 0.1,
            blur_sigma_max: 2.0,
            solarize_prob: 0.0,
            solarize_threshold: 128,
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
        };

        Self {
            dataset,
            training,
            optimizer,
            wgpu: WgpuRuntimeConfig::default(),
            vision,
            augment,
            mode,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.batch_repeats == 0 {
            return Err(anyhow!("training.batch_repeats must be > 0"));
        }
        if self.training.trace_train_loss_every == 0 {
            return Err(anyhow!("training.trace_train_loss_every must be > 0"));
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        if matches!(
            self.training.schedule_mode,
            Some(VisionTrainScheduleMode::Epochs)
        ) && self.training.epochs.is_none()
        {
            return Err(anyhow!(
                "training.schedule_mode=epochs requires training.epochs to be set"
            ));
        }
        self.optimizer.validate()?;

        if self.vision.image_size == 0 {
            return Err(anyhow!("vision.image_size must be > 0"));
        }
        if self.vision.patch_size == 0 {
            return Err(anyhow!("vision.patch_size must be > 0"));
        }
        if self.vision.in_channels == 0 {
            return Err(anyhow!("vision.in_channels must be > 0"));
        }
        if self.vision.embed_dim == 0 {
            return Err(anyhow!("vision.embed_dim must be > 0"));
        }
        if self.vision.steps == 0 {
            return Err(anyhow!("vision.steps must be > 0"));
        }
        if self.vision.cross_eye_steps > self.vision.steps {
            return Err(anyhow!(
                "vision.cross_eye_steps ({}) must be <= vision.steps ({})",
                self.vision.cross_eye_steps,
                self.vision.steps
            ));
        }
        if self.vision.n_head == 0 {
            return Err(anyhow!("vision.n_head must be > 0"));
        }
        if self.vision.mlp_internal_dim_multiplier == 0 {
            return Err(anyhow!("vision.mlp_internal_dim_multiplier must be > 0"));
        }
        if self.vision.projection_dim == 0 {
            return Err(anyhow!("vision.projection_dim must be > 0"));
        }
        if self.vision.projection_hidden_dim == 0 {
            return Err(anyhow!("vision.projection_hidden_dim must be > 0"));
        }
        if self.vision.num_eyes == 0 {
            return Err(anyhow!("vision.num_eyes must be > 0"));
        }
        if self.vision.cls_sync_alpha < 0.0 || self.vision.cls_sync_alpha > 1.0 {
            return Err(anyhow!("vision.cls_sync_alpha must be between 0.0 and 1.0"));
        }
        if self.vision.dropout < 0.0 {
            return Err(anyhow!("vision.dropout must be >= 0"));
        }
        if matches!(self.vision.pos_max_height, Some(0)) {
            return Err(anyhow!("vision.pos_max_height must be > 0 when set"));
        }
        if matches!(self.vision.pos_max_width, Some(0)) {
            return Err(anyhow!("vision.pos_max_width must be > 0 when set"));
        }
        if !self.vision.allow_softmax_attention
            && matches!(self.vision.attention_mode, VisionAttentionMode::Softmax)
        {
            return Err(anyhow!(
                "vision.attention_mode=softmax requires vision.allow_softmax_attention=true"
            ));
        }

        load_validate::validate_vision_mhc(&self.vision)?;
        load_validate::validate_vision_trm_graph(&self.vision)?;
        load_validate::validate_vision_rho_stream(&self.vision)?;
        load_validate::validate_vision_rollout(&self.training, self.vision.steps)?;
        load_validate::validate_vision_mode(
            &self.mode,
            &self.vision,
            &self.dataset,
            &self.augment,
        )?;

        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.optimizer.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }

        Ok(())
    }
}
