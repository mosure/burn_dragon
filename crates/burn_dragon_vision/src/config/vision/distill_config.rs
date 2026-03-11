use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::loss::VisionDistillationLossConfig;
use burn_dragon_train::VisionTeacherVariant;

use super::{VisionLejepaConfig, VisionMaeConfig, VisionSaccadeConfig, VisionVideoLejepaConfig};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionTrainingModeConfig {
    Distill(VisionDistillConfig),
    Lejepa(VisionLejepaConfig),
    VideoLejepa(VisionVideoLejepaConfig),
    Mae(VisionMaeConfig),
    Saccade(Box<VisionSaccadeConfig>),
}

impl Default for VisionTrainingModeConfig {
    fn default() -> Self {
        Self::Distill(VisionDistillConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionDistillConfig {
    #[serde(default)]
    pub teacher: VisionTeacherConfig,
    #[serde(default)]
    pub loss: VisionDistillationLossConfig,
    #[serde(default = "default_distill_rollout_supervision_frames")]
    pub rollout_supervision_frames: usize,
    #[serde(default = "default_distill_rollout_supervision_power")]
    pub rollout_supervision_power: f32,
    #[serde(default = "default_distill_rollout_sampling_power")]
    pub rollout_sampling_power: f32,
}

impl Default for VisionDistillConfig {
    fn default() -> Self {
        Self {
            teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig::default()),
            loss: VisionDistillationLossConfig::default(),
            rollout_supervision_frames: default_distill_rollout_supervision_frames(),
            rollout_supervision_power: default_distill_rollout_supervision_power(),
            rollout_sampling_power: default_distill_rollout_sampling_power(),
        }
    }
}

const fn default_distill_rollout_supervision_frames() -> usize {
    4
}

const fn default_distill_rollout_supervision_power() -> f32 {
    1.0
}

const fn default_distill_rollout_sampling_power() -> f32 {
    0.0
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionTeacherConfig {
    Features(VisionTeacherFeatureConfig),
    Model(VisionTeacherModelConfig),
}

impl Default for VisionTeacherConfig {
    fn default() -> Self {
        Self::Features(VisionTeacherFeatureConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionTeacherFeatureConfig {
    pub train_cls_path: PathBuf,
    pub train_patch_path: PathBuf,
    pub val_cls_path: PathBuf,
    pub val_patch_path: PathBuf,
    pub feature_dim: usize,
    pub patch_tokens: Option<usize>,
}

impl Default for VisionTeacherFeatureConfig {
    fn default() -> Self {
        Self {
            train_cls_path: PathBuf::from("data/imagenet1k/features/dinov3_small/train_cls.bin"),
            train_patch_path: PathBuf::from(
                "data/imagenet1k/features/dinov3_small/train_patch.bin",
            ),
            val_cls_path: PathBuf::from("data/imagenet1k/features/dinov3_small/val_cls.bin"),
            val_patch_path: PathBuf::from("data/imagenet1k/features/dinov3_small/val_patch.bin"),
            feature_dim: 384,
            patch_tokens: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionTeacherModelConfig {
    pub checkpoint_path: PathBuf,
    #[serde(default)]
    pub variant: VisionTeacherVariant,
    #[serde(default)]
    pub image_size: Option<usize>,
    #[serde(default)]
    pub patch_size: Option<usize>,
    #[serde(default)]
    pub register_tokens: usize,
    #[serde(default)]
    pub feature_dim: Option<usize>,
    #[serde(default)]
    pub patch_tokens: Option<usize>,
}
