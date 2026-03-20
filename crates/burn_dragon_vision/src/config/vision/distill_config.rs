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
    pub teacher_targets: Vec<VisionTeacherTargetConfig>,
    #[serde(default)]
    pub student_checkpoint: Option<PathBuf>,
    #[serde(default)]
    pub loss: VisionDistillationLossConfig,
    #[serde(default = "default_distill_rollout_supervision_frames")]
    pub rollout_supervision_frames: usize,
    #[serde(default = "default_distill_rollout_supervision_stride")]
    pub rollout_supervision_stride: usize,
    #[serde(default = "default_distill_rollout_supervision_groups")]
    pub rollout_supervision_groups: usize,
    #[serde(default)]
    pub rollout_supervision_explicit_steps: Vec<usize>,
    #[serde(default)]
    pub rollout_supervision_explicit_groups: Vec<Vec<usize>>,
    #[serde(default = "default_distill_rollout_supervision_include_step1")]
    pub rollout_supervision_include_step1: bool,
    #[serde(default = "default_distill_rollout_supervision_power")]
    pub rollout_supervision_power: f32,
    #[serde(default = "default_distill_rollout_sampling_power")]
    pub rollout_sampling_power: f32,
    #[serde(default = "default_distill_rollout_improvement_weight")]
    pub rollout_improvement_weight: f32,
    #[serde(default = "default_distill_rollout_improvement_margin")]
    pub rollout_improvement_margin: f32,
}

impl Default for VisionDistillConfig {
    fn default() -> Self {
        Self {
            teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig::default()),
            teacher_targets: Vec::new(),
            student_checkpoint: None,
            loss: VisionDistillationLossConfig::default(),
            rollout_supervision_frames: default_distill_rollout_supervision_frames(),
            rollout_supervision_stride: default_distill_rollout_supervision_stride(),
            rollout_supervision_groups: default_distill_rollout_supervision_groups(),
            rollout_supervision_explicit_steps: Vec::new(),
            rollout_supervision_explicit_groups: Vec::new(),
            rollout_supervision_include_step1: default_distill_rollout_supervision_include_step1(),
            rollout_supervision_power: default_distill_rollout_supervision_power(),
            rollout_sampling_power: default_distill_rollout_sampling_power(),
            rollout_improvement_weight: default_distill_rollout_improvement_weight(),
            rollout_improvement_margin: default_distill_rollout_improvement_margin(),
        }
    }
}

impl VisionDistillConfig {
    pub const PRIMARY_TEACHER_NAME: &str = "primary";

    pub fn primary_teacher_target(&self) -> VisionTeacherTargetConfig {
        VisionTeacherTargetConfig {
            name: Self::PRIMARY_TEACHER_NAME.to_string(),
            weight: default_teacher_target_weight(),
            target_kind: VisionTeacherTargetKind::PatchAndCls,
            decoder_mode: VisionTeacherDecoderMode::SharedProjection,
            decoder_hidden_dim: None,
            teacher: self.teacher.clone(),
        }
    }

    pub fn resolved_teacher_targets(&self) -> Vec<VisionTeacherTargetConfig> {
        let mut targets = Vec::with_capacity(self.teacher_targets.len() + 1);
        targets.push(self.primary_teacher_target());
        targets.extend(self.teacher_targets.iter().cloned());
        targets
    }

    pub fn auxiliary_teacher_targets(&self) -> &[VisionTeacherTargetConfig] {
        &self.teacher_targets
    }
}

const fn default_distill_rollout_supervision_frames() -> usize {
    4
}

const fn default_distill_rollout_supervision_stride() -> usize {
    1
}

const fn default_distill_rollout_supervision_groups() -> usize {
    1
}

const fn default_distill_rollout_supervision_include_step1() -> bool {
    true
}

const fn default_distill_rollout_supervision_power() -> f32 {
    1.0
}

const fn default_distill_rollout_sampling_power() -> f32 {
    0.0
}

const fn default_distill_rollout_improvement_weight() -> f32 {
    0.0
}

const fn default_distill_rollout_improvement_margin() -> f32 {
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTeacherTargetKind {
    #[default]
    PatchAndCls,
    ClsOnly,
    GlobalOnly,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTeacherDecoderMode {
    #[default]
    SharedProjection,
    DedicatedProjection,
    DedicatedSpatialProjection,
}

impl VisionTeacherDecoderMode {
    pub fn uses_dedicated_projection(self) -> bool {
        !matches!(self, Self::SharedProjection)
    }

    pub fn supports_spatial_resampling(self) -> bool {
        matches!(self, Self::DedicatedSpatialProjection)
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionTeacherTargetConfig {
    pub name: String,
    pub weight: f32,
    pub target_kind: VisionTeacherTargetKind,
    pub decoder_mode: VisionTeacherDecoderMode,
    pub decoder_hidden_dim: Option<usize>,
    pub teacher: VisionTeacherConfig,
}

impl Default for VisionTeacherTargetConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            weight: default_teacher_target_weight(),
            target_kind: VisionTeacherTargetKind::default(),
            decoder_mode: VisionTeacherDecoderMode::default(),
            decoder_hidden_dim: None,
            teacher: VisionTeacherConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionTeacherFeatureConfig {
    pub train_cls_path: PathBuf,
    #[serde(default)]
    pub train_patch_path: Option<PathBuf>,
    pub val_cls_path: PathBuf,
    #[serde(default)]
    pub val_patch_path: Option<PathBuf>,
    pub feature_dim: usize,
    pub patch_tokens: Option<usize>,
}

impl Default for VisionTeacherFeatureConfig {
    fn default() -> Self {
        Self {
            train_cls_path: PathBuf::from("data/imagenet1k/features/dinov3_small/train_cls.bin"),
            train_patch_path: Some(PathBuf::from(
                "data/imagenet1k/features/dinov3_small/train_patch.bin",
            )),
            val_cls_path: PathBuf::from("data/imagenet1k/features/dinov3_small/val_cls.bin"),
            val_patch_path: Some(PathBuf::from(
                "data/imagenet1k/features/dinov3_small/val_patch.bin",
            )),
            feature_dim: 384,
            patch_tokens: None,
        }
    }
}

impl VisionTeacherFeatureConfig {
    pub fn has_patch_targets(&self) -> bool {
        self.train_patch_path.is_some() && self.val_patch_path.is_some()
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

const fn default_teacher_target_weight() -> f32 {
    1.0
}
