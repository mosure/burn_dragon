use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};
use toml::Value;

use crate::model::{FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode};
use crate::model::VisionDistillationLossConfig;

use super::{GdpoConfig, GdpoHardGate, OptimizerConfig, WgpuRuntimeConfig};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionPyramidMode {
    Stacked,
    Laplacian,
}

impl Default for VisionPyramidMode {
    fn default() -> Self {
        Self::Laplacian
    }
}

impl fmt::Display for VisionPyramidMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Stacked => write!(f, "stacked"),
            Self::Laplacian => write!(f, "laplacian"),
        }
    }
}

impl ModuleDisplayDefault for VisionPyramidMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionPyramidMode {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaSamplingMode {
    Batched,
    Sequential,
    Subpatch,
    Cubecl,
    Wgsl,
}

impl Default for VisionFoveaSamplingMode {
    fn default() -> Self {
        Self::Sequential
    }
}

impl fmt::Display for VisionFoveaSamplingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Batched => write!(f, "batched"),
            Self::Sequential => write!(f, "sequential"),
            Self::Subpatch => write!(f, "subpatch"),
            Self::Cubecl => write!(f, "cubecl"),
            Self::Wgsl => write!(f, "wgsl"),
        }
    }
}

impl ModuleDisplayDefault for VisionFoveaSamplingMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionFoveaSamplingMode {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaWarpMode {
    Warped,
    Patched,
}

impl Default for VisionFoveaWarpMode {
    fn default() -> Self {
        Self::Warped
    }
}

impl fmt::Display for VisionFoveaWarpMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Warped => write!(f, "warped"),
            Self::Patched => write!(f, "patched"),
        }
    }
}

impl ModuleDisplayDefault for VisionFoveaWarpMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionFoveaWarpMode {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaScatterMode {
    Tensor,
    Cubecl,
    Wgsl,
}

impl Default for VisionFoveaScatterMode {
    fn default() -> Self {
        Self::Tensor
    }
}

impl fmt::Display for VisionFoveaScatterMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tensor => write!(f, "tensor"),
            Self::Cubecl => write!(f, "cubecl"),
            Self::Wgsl => write!(f, "wgsl"),
        }
    }
}

impl ModuleDisplayDefault for VisionFoveaScatterMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionFoveaScatterMode {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionLocationEmbeddingMode {
    None,
    Learned,
    Sinusoidal,
    Quantized,
    Rope,
    Pope,
}

impl Default for VisionLocationEmbeddingMode {
    fn default() -> Self {
        Self::Pope
    }
}

impl fmt::Display for VisionLocationEmbeddingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => write!(f, "none"),
            Self::Learned => write!(f, "learned"),
            Self::Sinusoidal => write!(f, "sinusoidal"),
            Self::Quantized => write!(f, "quantized"),
            Self::Rope => write!(f, "rope"),
            Self::Pope => write!(f, "pope"),
        }
    }
}

impl ModuleDisplayDefault for VisionLocationEmbeddingMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionLocationEmbeddingMode {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionLocationEmbeddingConfig {
    pub mode: VisionLocationEmbeddingMode,
    pub embed_dim: usize,
    pub quantize_bins: usize,
    pub noise_std: f32,
}

impl Default for VisionLocationEmbeddingConfig {
    fn default() -> Self {
        Self {
            mode: VisionLocationEmbeddingMode::default(),
            embed_dim: 12,
            quantize_bins: 32,
            noise_std: 0.0,
        }
    }
}

impl ModuleDisplayDefault for VisionLocationEmbeddingConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("mode", &self.mode)
            .add("embed_dim", &self.embed_dim)
            .add("quantize_bins", &self.quantize_bins)
            .add("noise_std", &self.noise_std)
            .optional()
    }
}

impl ModuleDisplay for VisionLocationEmbeddingConfig {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionNullGlimpseMode {
    Zero,
    Noise,
}

impl Default for VisionNullGlimpseMode {
    fn default() -> Self {
        Self::Zero
    }
}

impl fmt::Display for VisionNullGlimpseMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Zero => write!(f, "zero"),
            Self::Noise => write!(f, "noise"),
        }
    }
}

impl ModuleDisplayDefault for VisionNullGlimpseMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionNullGlimpseMode {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeInfoRewardConfig {
    pub enabled: bool,
    pub stride: usize,
    pub null_mode: VisionNullGlimpseMode,
    pub null_noise_std: f32,
}

impl Default for VisionSaccadeInfoRewardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            stride: 1,
            null_mode: VisionNullGlimpseMode::Zero,
            null_noise_std: 0.05,
        }
    }
}

impl ModuleDisplayDefault for VisionSaccadeInfoRewardConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("stride", &self.stride)
            .add("null_mode", &self.null_mode)
            .add("null_noise_std", &self.null_noise_std)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadeInfoRewardConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadePolicyConfig {
    pub location_embedding: VisionLocationEmbeddingConfig,
    pub detach_policy_from_recon: bool,
    pub action_noise_std: f32,
    pub info_reward: VisionSaccadeInfoRewardConfig,
    pub gdpo: GdpoConfig,
}

impl Default for VisionSaccadePolicyConfig {
    fn default() -> Self {
        Self {
            location_embedding: VisionLocationEmbeddingConfig::default(),
            detach_policy_from_recon: false,
            action_noise_std: 0.05,
            info_reward: VisionSaccadeInfoRewardConfig::default(),
            gdpo: GdpoConfig::default(),
        }
    }
}

impl ModuleDisplayDefault for VisionSaccadePolicyConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("location_embedding", &self.location_embedding)
            .add("detach_policy_from_recon", &self.detach_policy_from_recon)
            .add("action_noise_std", &self.action_noise_std)
            .add("info_reward", &self.info_reward)
            .add("gdpo", &self.gdpo)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadePolicyConfig {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionArtifactOutputMode {
    Images,
    Avi,
    Mp4,
}

impl Default for VisionArtifactOutputMode {
    fn default() -> Self {
        Self::Images
    }
}

impl fmt::Display for VisionArtifactOutputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Images => write!(f, "images"),
            Self::Avi => write!(f, "avi"),
            Self::Mp4 => write!(f, "mp4"),
        }
    }
}

impl ModuleDisplayDefault for VisionArtifactOutputMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionArtifactOutputMode {}

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
    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
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
            return Err(anyhow!(
                "training.trace_train_loss_every must be > 0"
            ));
        }
        if let Some(epochs) = self.training.epochs {
            if epochs == 0 {
                return Err(anyhow!("training.epochs must be > 0"));
            }
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
        if self.vision.n_head == 0 {
            return Err(anyhow!("vision.n_head must be > 0"));
        }
        if self.vision.mlp_internal_dim_multiplier == 0 {
            return Err(anyhow!(
                "vision.mlp_internal_dim_multiplier must be > 0"
            ));
        }
        if self.vision.projection_dim == 0 {
            return Err(anyhow!("vision.projection_dim must be > 0"));
        }
        if self.vision.projection_hidden_dim == 0 {
            return Err(anyhow!("vision.projection_hidden_dim must be > 0"));
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

        validate_vision_rollout(&self.training, self.vision.steps)?;
        validate_vision_mode(&self.mode, &self.vision)?;

        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.optimizer.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionTrainingModeConfig {
    Distill(VisionDistillConfig),
    Lejepa(VisionLejepaConfig),
    Mae(VisionMaeConfig),
    Saccade(VisionSaccadeConfig),
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
}

impl Default for VisionDistillConfig {
    fn default() -> Self {
        Self {
            teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig::default()),
            loss: VisionDistillationLossConfig::default(),
        }
    }
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
            train_patch_path: PathBuf::from("data/imagenet1k/features/dinov3_small/train_patch.bin"),
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTeacherVariant {
    #[default]
    Vits,
    Vitb,
    Vitl,
    Vitg,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionLejepaLossConfig {
    pub enabled: bool,
    pub lambda: f32,
    pub sigreg_knots: usize,
    pub sigreg_t_max: f32,
    pub sigreg_proj_dim: usize,
}

impl Default for VisionLejepaLossConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            lambda: 0.02,
            sigreg_knots: 17,
            sigreg_t_max: 3.0,
            sigreg_proj_dim: 256,
        }
    }
}

impl ModuleDisplayDefault for VisionLejepaLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("lambda", &self.lambda)
            .add("sigreg_knots", &self.sigreg_knots)
            .add("sigreg_t_max", &self.sigreg_t_max)
            .add("sigreg_proj_dim", &self.sigreg_proj_dim)
            .optional()
    }
}

impl ModuleDisplay for VisionLejepaLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionReconLossConfig {
    pub weight: f32,
    pub mask_ratio: f32,
    pub hidden_dim: usize,
}

impl Default for VisionReconLossConfig {
    fn default() -> Self {
        Self {
            weight: 0.0,
            mask_ratio: 0.75,
            hidden_dim: 256,
        }
    }
}

impl ModuleDisplayDefault for VisionReconLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("mask_ratio", &self.mask_ratio)
            .add("hidden_dim", &self.hidden_dim)
            .optional()
    }
}

impl ModuleDisplay for VisionReconLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionLossConfig {
    pub lejepa: VisionLejepaLossConfig,
    pub recon: VisionReconLossConfig,
}

impl Default for VisionLossConfig {
    fn default() -> Self {
        Self {
            lejepa: VisionLejepaLossConfig::default(),
            recon: VisionReconLossConfig::default(),
        }
    }
}

impl ModuleDisplayDefault for VisionLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("lejepa", &self.lejepa)
            .add("recon", &self.recon)
            .optional()
    }
}

impl ModuleDisplay for VisionLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionMaeLossConfig {
    pub recon: VisionReconLossConfig,
}

impl Default for VisionMaeLossConfig {
    fn default() -> Self {
        Self {
            recon: VisionReconLossConfig {
                weight: 1.0,
                mask_ratio: 0.75,
                hidden_dim: 256,
            },
        }
    }
}

impl ModuleDisplayDefault for VisionMaeLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content.add("recon", &self.recon).optional()
    }
}

impl ModuleDisplay for VisionMaeLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionLejepaConfig {
    pub loss: VisionLossConfig,
    pub views: usize,
    pub global_views: usize,
    pub local_views: usize,
    pub local_image_size: usize,
    pub local_min_scale: f32,
    pub local_max_scale: f32,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_max_views: usize,
    pub artifact_overwrite: bool,
}

impl Default for VisionLejepaConfig {
    fn default() -> Self {
        Self {
            loss: VisionLossConfig::default(),
            views: 4,
            global_views: 0,
            local_views: 0,
            local_image_size: 96,
            local_min_scale: 0.05,
            local_max_scale: 0.3,
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 4,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_max_views: 3,
            artifact_overwrite: true,
        }
    }
}

impl<B: Backend> Module<B> for VisionLejepaConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionLejepaConfig {
    type InnerModule = VisionLejepaConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for VisionLejepaConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("loss", &self.loss)
            .add("views", &self.views)
            .add("global_views", &self.global_views)
            .add("local_views", &self.local_views)
            .add("local_image_size", &self.local_image_size)
            .add("local_min_scale", &self.local_min_scale)
            .add("local_max_scale", &self.local_max_scale)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_max_views", &self.artifact_max_views)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .optional()
    }
}

impl ModuleDisplay for VisionLejepaConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionMaeConfig {
    pub loss: VisionMaeLossConfig,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_max_views: usize,
    pub artifact_overwrite: bool,
}

impl Default for VisionMaeConfig {
    fn default() -> Self {
        Self {
            loss: VisionMaeLossConfig::default(),
            artifact_output: VisionArtifactOutputMode::Images,
            artifact_fps: 4,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_max_views: 3,
            artifact_overwrite: true,
        }
    }
}

impl<B: Backend> Module<B> for VisionMaeConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionMaeConfig {
    type InnerModule = VisionMaeConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for VisionMaeConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("loss", &self.loss)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_max_views", &self.artifact_max_views)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .optional()
    }
}

impl ModuleDisplay for VisionMaeConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeCacheConfig {
    pub max_entries: usize,
}

impl Default for VisionSaccadeCacheConfig {
    fn default() -> Self {
        Self { max_entries: 64 }
    }
}

impl ModuleDisplayDefault for VisionSaccadeCacheConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content.add("max_entries", &self.max_entries).optional()
    }
}

impl ModuleDisplay for VisionSaccadeCacheConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionTbpttConfig {
    pub step_count: usize,
}

impl Default for VisionTbpttConfig {
    fn default() -> Self {
        Self { step_count: 0 }
    }
}

impl ModuleDisplayDefault for VisionTbpttConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content.add("step_count", &self.step_count).optional()
    }
}

impl ModuleDisplay for VisionTbpttConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionSaccadeInputProjectionConfig {
    Linear,
    Cnn(VisionSaccadeInputProjectionCnnConfig),
    RadialMicroVit(VisionSaccadeInputProjectionMicroVitConfig),
}

impl Default for VisionSaccadeInputProjectionConfig {
    fn default() -> Self {
        Self::Linear
    }
}

impl ModuleDisplayDefault for VisionSaccadeInputProjectionConfig {
    fn content(&self, content: Content) -> Option<Content> {
        match self {
            VisionSaccadeInputProjectionConfig::Linear => {
                content.add("type", "linear").optional()
            }
            VisionSaccadeInputProjectionConfig::Cnn(cfg) => content
                .add("type", "cnn")
                .add("config", cfg)
                .optional(),
            VisionSaccadeInputProjectionConfig::RadialMicroVit(cfg) => content
                .add("type", "radial_micro_vit")
                .add("config", cfg)
                .optional(),
        }
    }
}

impl ModuleDisplay for VisionSaccadeInputProjectionConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeInputProjectionCnnConfig {
    pub channels: Option<usize>,
    pub blocks: usize,
    pub kernel: usize,
    pub expansion: usize,
}

impl Default for VisionSaccadeInputProjectionCnnConfig {
    fn default() -> Self {
        Self {
            channels: None,
            blocks: 0,
            kernel: 0,
            expansion: 2,
        }
    }
}

impl ModuleDisplayDefault for VisionSaccadeInputProjectionCnnConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("channels", &self.channels)
            .add("blocks", &self.blocks)
            .add("kernel", &self.kernel)
            .add("expansion", &self.expansion)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadeInputProjectionCnnConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeInputProjectionMicroVitConfig {
    pub layers: usize,
    pub heads: usize,
    pub mlp_ratio: usize,
    pub radial_hidden_dim: usize,
    pub radial_scale: f32,
}

impl Default for VisionSaccadeInputProjectionMicroVitConfig {
    fn default() -> Self {
        Self {
            layers: 0,
            heads: 0,
            mlp_ratio: 2,
            radial_hidden_dim: 0,
            radial_scale: 1.0,
        }
    }
}

impl ModuleDisplayDefault for VisionSaccadeInputProjectionMicroVitConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("layers", &self.layers)
            .add("heads", &self.heads)
            .add("mlp_ratio", &self.mlp_ratio)
            .add("radial_hidden_dim", &self.radial_hidden_dim)
            .add("radial_scale", &self.radial_scale)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadeInputProjectionMicroVitConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeConfig {
    pub num_eyes: usize,
    pub traj_tokens: usize,
    #[serde(default = "default_traj_update_alpha")]
    pub traj_update_alpha: f32,
    pub mip_levels: usize,
    pub pyramid_mode: VisionPyramidMode,
    pub fovea_sampling_mode: VisionFoveaSamplingMode,
    pub fovea_warp_mode: VisionFoveaWarpMode,
    #[serde(default = "default_fovea_subsamples")]
    pub fovea_subsamples: usize,
    #[serde(default = "default_fovea_radius_scale")]
    pub fovea_radius_scale: f32,
    pub fovea_subpatch_size: usize,
    pub fovea_scatter_mode: VisionFoveaScatterMode,
    #[serde(default)]
    pub input_projection: VisionSaccadeInputProjectionConfig,
    #[serde(default = "default_grid_sample_max_mb")]
    pub grid_sample_max_mb: usize,
    #[serde(default = "default_mip_concat_max_mb")]
    pub mip_concat_max_mb: usize,
    pub pyramid_feature_dim: Option<usize>,
    pub inner_steps: usize,
    pub low_mem_pre_rollout: bool,
    #[serde(default)]
    pub recon_batch_chunk: usize,
    #[serde(default = "default_recon_max_elems")]
    pub recon_max_elems: usize,
    pub tbptt: VisionTbpttConfig,
    pub policy: VisionSaccadePolicyConfig,
    pub cache: VisionSaccadeCacheConfig,
    pub loss: VisionLossConfig,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_max_views: usize,
    pub artifact_overwrite: bool,
}

impl Default for VisionSaccadeConfig {
    fn default() -> Self {
        Self {
            num_eyes: 1,
            traj_tokens: 1,
            traj_update_alpha: default_traj_update_alpha(),
            mip_levels: 4,
            pyramid_mode: VisionPyramidMode::Laplacian,
            fovea_sampling_mode: VisionFoveaSamplingMode::Batched,
            fovea_warp_mode: VisionFoveaWarpMode::Warped,
            fovea_subsamples: default_fovea_subsamples(),
            fovea_radius_scale: default_fovea_radius_scale(),
            fovea_subpatch_size: 0,
            fovea_scatter_mode: VisionFoveaScatterMode::Tensor,
            input_projection: VisionSaccadeInputProjectionConfig::default(),
            grid_sample_max_mb: default_grid_sample_max_mb(),
            mip_concat_max_mb: default_mip_concat_max_mb(),
            pyramid_feature_dim: None,
            inner_steps: 1,
            low_mem_pre_rollout: true,
            recon_batch_chunk: 0,
            recon_max_elems: default_recon_max_elems(),
            tbptt: VisionTbpttConfig::default(),
            policy: VisionSaccadePolicyConfig::default(),
            cache: VisionSaccadeCacheConfig::default(),
            loss: VisionLossConfig::default(),
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 8,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_max_views: 4,
            artifact_overwrite: true,
        }
    }
}

impl<B: Backend> Module<B> for VisionSaccadeConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionSaccadeConfig {
    type InnerModule = VisionSaccadeConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

impl ModuleDisplayDefault for VisionSaccadeConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("num_eyes", &self.num_eyes)
            .add("traj_tokens", &self.traj_tokens)
            .add("traj_update_alpha", &self.traj_update_alpha)
            .add("mip_levels", &self.mip_levels)
            .add("pyramid_mode", &self.pyramid_mode)
            .add("fovea_sampling_mode", &self.fovea_sampling_mode)
            .add("fovea_warp_mode", &self.fovea_warp_mode)
            .add("fovea_subsamples", &self.fovea_subsamples)
            .add("fovea_radius_scale", &self.fovea_radius_scale)
            .add("fovea_subpatch_size", &self.fovea_subpatch_size)
            .add("fovea_scatter_mode", &self.fovea_scatter_mode)
            .add("input_projection", &self.input_projection)
            .add("grid_sample_max_mb", &self.grid_sample_max_mb)
            .add("mip_concat_max_mb", &self.mip_concat_max_mb)
            .add("pyramid_feature_dim", &self.pyramid_feature_dim)
            .add("inner_steps", &self.inner_steps)
            .add("low_mem_pre_rollout", &self.low_mem_pre_rollout)
            .add("recon_batch_chunk", &self.recon_batch_chunk)
            .add("recon_max_elems", &self.recon_max_elems)
            .add("tbptt", &self.tbptt)
            .add("policy", &self.policy)
            .add("cache", &self.cache)
            .add("loss", &self.loss)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_max_views", &self.artifact_max_views)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadeConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionDatasetDownloadConfig {
    Imagenette {
        #[serde(default)]
        variant: ImagenetteVariant,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ImagenetteVariant {
    #[default]
    Imagenette2_160,
    Imagenette2_320,
}

fn default_prefetch_batches() -> usize {
    4
}

fn default_batch_repeats() -> usize {
    1
}

fn default_trace_train_loss_every() -> usize {
    1
}

fn default_fovea_subsamples() -> usize {
    1
}

fn default_fovea_radius_scale() -> f32 {
    1.0
}

fn default_traj_update_alpha() -> f32 {
    1.0
}

fn default_prefetch_workers() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().min(8))
        .unwrap_or(4)
}

fn default_prefetch_to_device() -> bool {
    true
}

fn default_cache_decoded() -> bool {
    true
}

fn default_cache_capacity() -> usize {
    512
}

fn default_cache_preprocessed() -> bool {
    false
}

fn default_grid_sample_max_mb() -> usize {
    512
}

fn default_mip_concat_max_mb() -> usize {
    512
}

fn default_recon_max_elems() -> usize {
    50_000_000
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionDatasetConfig {
    pub imagenet_root: PathBuf,
    pub train_dir: String,
    pub val_dir: String,
    pub max_records: Option<usize>,
    pub download: Option<VisionDatasetDownloadConfig>,
    #[serde(default = "default_prefetch_batches")]
    pub prefetch_batches: usize,
    #[serde(default = "default_prefetch_workers")]
    pub prefetch_workers: usize,
    #[serde(default = "default_prefetch_to_device")]
    pub prefetch_to_device: bool,
    #[serde(default = "default_cache_decoded")]
    pub cache_decoded: bool,
    #[serde(default = "default_cache_capacity")]
    pub cache_capacity: usize,
    #[serde(default = "default_cache_preprocessed")]
    pub cache_preprocessed: bool,
}

impl Default for VisionDatasetConfig {
    fn default() -> Self {
        Self {
            imagenet_root: PathBuf::from("data/imagenet1k"),
            train_dir: "train".to_string(),
            val_dir: "val".to_string(),
            max_records: None,
            download: None,
            prefetch_batches: default_prefetch_batches(),
            prefetch_workers: default_prefetch_workers(),
            prefetch_to_device: default_prefetch_to_device(),
            cache_decoded: default_cache_decoded(),
            cache_capacity: default_cache_capacity(),
            cache_preprocessed: default_cache_preprocessed(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionTrainingHyperparameters {
    pub batch_size: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default = "default_batch_repeats")]
    pub batch_repeats: usize,
    #[serde(default)]
    pub train_repeat_chunk: usize,
    #[serde(default)]
    pub memory_cleanup_every: usize,
    #[serde(default)]
    pub memory_cleanup_iters: usize,
    #[serde(default)]
    pub disable_cuda_memory_cleanup: bool,
    #[serde(default)]
    pub trace_train_loss: bool,
    #[serde(default = "default_trace_train_loss_every")]
    pub trace_train_loss_every: usize,
    #[serde(default)]
    pub rollout_min_steps: Option<usize>,
    #[serde(default)]
    pub rollout_max_steps: Option<usize>,
    #[serde(default)]
    pub rollout_backprop_steps: Option<usize>,
    #[serde(default)]
    pub ffmpeg_path: Option<PathBuf>,
}

impl Default for VisionTrainingHyperparameters {
    fn default() -> Self {
        Self {
            batch_size: 64,
            epochs: None,
            max_iters: 1000,
            log_frequency: 50,
            batch_repeats: 1,
            train_repeat_chunk: 0,
            memory_cleanup_every: 0,
            memory_cleanup_iters: 0,
            disable_cuda_memory_cleanup: false,
            trace_train_loss: false,
            trace_train_loss_every: default_trace_train_loss_every(),
            rollout_min_steps: None,
            rollout_max_steps: None,
            rollout_backprop_steps: None,
            ffmpeg_path: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionModelConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub steps: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub dropout: f64,
    pub projection_dim: usize,
    pub projection_hidden_dim: usize,
    pub use_cls_token: bool,
    pub pos_encoding: SpatialPositionalEncodingKind,
    pub pos_max_height: Option<usize>,
    pub pos_max_width: Option<usize>,
    pub attention_mode: VisionAttentionMode,
    pub fused_kernels: bool,
    pub relu_threshold: f32,
}

impl Default for VisionModelConfig {
    fn default() -> Self {
        let image_size = 224;
        let patch_size = 16;
        Self {
            image_size,
            patch_size,
            in_channels: 3,
            embed_dim: 256,
            steps: 6,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            dropout: 0.1,
            projection_dim: 384,
            projection_hidden_dim: 512,
            use_cls_token: true,
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: None,
            pos_max_width: None,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: false,
            relu_threshold: 0.0,
        }
    }
}

impl VisionModelConfig {
    pub fn build(&self) -> crate::model::VisionDragonHatchlingConfig {
        let patch_size = self.patch_size.max(1);
        let grid = (self.image_size + patch_size - 1) / patch_size;
        let kernels = FusedKernelConfig {
            enabled: self.fused_kernels,
            relu_threshold: self.relu_threshold,
            ..Default::default()
        };

        crate::model::VisionDragonHatchlingConfig {
            image_size: self.image_size,
            patch_size: self.patch_size,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
            steps: self.steps,
            n_head: self.n_head,
            mlp_internal_dim_multiplier: self.mlp_internal_dim_multiplier,
            dropout: self.dropout,
            projection_dim: self.projection_dim,
            projection_hidden_dim: self.projection_hidden_dim,
            use_cls_token: self.use_cls_token,
            pos_encoding: self.pos_encoding,
            pos_max_height: self.pos_max_height.unwrap_or(grid),
            pos_max_width: self.pos_max_width.unwrap_or(grid),
            attention_mode: self.attention_mode,
            fused_kernels: kernels,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionAugmentationConfig {
    pub image_size: usize,
    pub resize_short: usize,
    pub min_scale: f32,
    pub max_scale: f32,
    pub min_aspect_ratio: f32,
    pub max_aspect_ratio: f32,
    pub flip_prob: f32,
    pub color_jitter_prob: f32,
    pub brightness: f32,
    pub contrast: f32,
    pub saturation: f32,
    pub hue: f32,
    pub grayscale_prob: f32,
    pub blur_prob: f32,
    pub blur_sigma_min: f32,
    pub blur_sigma_max: f32,
    pub solarize_prob: f32,
    pub solarize_threshold: u8,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
}

impl Default for VisionAugmentationConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            resize_short: 256,
            min_scale: 0.08,
            max_scale: 1.0,
            min_aspect_ratio: 0.75,
            max_aspect_ratio: 1.3333334,
            flip_prob: 0.5,
            color_jitter_prob: 1.0,
            brightness: 0.4,
            contrast: 0.4,
            saturation: 0.0,
            hue: 0.1,
            grayscale_prob: 0.0,
            blur_prob: 0.0,
            blur_sigma_min: 0.1,
            blur_sigma_max: 2.0,
            solarize_prob: 0.0,
            solarize_threshold: 128,
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
        }
    }
}

pub fn load_vision_training_config(paths: &[PathBuf]) -> Result<VisionTrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<VisionTrainingConfig>()
        .map_err(|err| anyhow!(err))
}

fn validate_vision_rollout(
    training: &VisionTrainingHyperparameters,
    max_steps: usize,
) -> Result<()> {
    let max_steps = max_steps.max(1);
    let min_steps = training.rollout_min_steps.unwrap_or(max_steps);
    let max_steps_cfg = training.rollout_max_steps.unwrap_or(max_steps);
    let backprop_steps = training.rollout_backprop_steps.unwrap_or(max_steps_cfg);
    if min_steps == 0 || max_steps_cfg == 0 {
        return Err(anyhow!(
            "vision rollout steps must be > 0 (min={min_steps}, max={max_steps_cfg})"
        ));
    }
    if backprop_steps == 0 {
        return Err(anyhow!(
            "vision rollout_backprop_steps must be > 0 (value={backprop_steps})"
        ));
    }
    if min_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_min_steps ({min_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    if max_steps_cfg > max_steps {
        return Err(anyhow!(
            "vision rollout_max_steps ({max_steps_cfg}) exceeds vision.steps ({max_steps})"
        ));
    }
    if backprop_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_backprop_steps ({backprop_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    Ok(())
}

fn validate_vision_mode(mode: &VisionTrainingModeConfig, vision: &VisionModelConfig) -> Result<()> {
    match mode {
        VisionTrainingModeConfig::Distill(distill) => {
            validate_distill_loss(&distill.loss)?;
            match &distill.teacher {
                VisionTeacherConfig::Features(config) => {
                    if config.feature_dim == 0 {
                        return Err(anyhow!("mode.teacher.feature_dim must be > 0"));
                    }
                    if matches!(config.patch_tokens, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
                    }
                }
                VisionTeacherConfig::Model(config) => {
                    if matches!(config.image_size, Some(0)) {
                        return Err(anyhow!("mode.teacher.image_size must be > 0 when set"));
                    }
                    if matches!(config.patch_size, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_size must be > 0 when set"));
                    }
                    if matches!(config.feature_dim, Some(0)) {
                        return Err(anyhow!("mode.teacher.feature_dim must be > 0 when set"));
                    }
                    if matches!(config.patch_tokens, Some(0)) {
                        return Err(anyhow!("mode.teacher.patch_tokens must be > 0 when set"));
                    }
                }
            }
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            if lejepa.views == 0 {
                return Err(anyhow!("mode.views must be > 0"));
            }
            if lejepa.local_image_size == 0 {
                return Err(anyhow!("mode.local_image_size must be > 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.local_min_scale) {
                return Err(anyhow!(
                    "mode.local_min_scale must be in [0, 1] (got {})",
                    lejepa.local_min_scale
                ));
            }
            if !(0.0..=1.0).contains(&lejepa.local_max_scale) {
                return Err(anyhow!(
                    "mode.local_max_scale must be in [0, 1] (got {})",
                    lejepa.local_max_scale
                ));
            }
            if lejepa.local_min_scale > lejepa.local_max_scale {
                return Err(anyhow!(
                    "mode.local_min_scale ({}) must be <= mode.local_max_scale ({})",
                    lejepa.local_min_scale,
                    lejepa.local_max_scale
                ));
            }
            validate_lejepa_loss(&lejepa.loss.lejepa)?;
            validate_recon_loss("mode.loss.recon", &lejepa.loss.recon)?;
        }
        VisionTrainingModeConfig::Mae(mae) => {
            validate_recon_loss("mode.loss.recon", &mae.loss.recon)?;
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            if saccade.num_eyes == 0 {
                return Err(anyhow!("saccade.num_eyes must be > 0"));
            }
            if saccade.traj_tokens == 0 {
                return Err(anyhow!("saccade.traj_tokens must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.traj_update_alpha) {
                return Err(anyhow!(
                    "saccade.traj_update_alpha must be in [0, 1] (got {})",
                    saccade.traj_update_alpha
                ));
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if saccade.fovea_subsamples == 0 {
                return Err(anyhow!("saccade.fovea_subsamples must be > 0"));
            }
            if saccade.fovea_radius_scale <= 0.0 {
                return Err(anyhow!(
                    "saccade.fovea_radius_scale must be > 0 (got {})",
                    saccade.fovea_radius_scale
                ));
            }
            if saccade.grid_sample_max_mb == 0 {
                return Err(anyhow!(
                    "saccade.grid_sample_max_mb must be > 0"
                ));
            }
            if saccade.mip_concat_max_mb == 0 {
                return Err(anyhow!(
                    "saccade.mip_concat_max_mb must be > 0"
                ));
            }
            if saccade.recon_max_elems == 0 {
                return Err(anyhow!(
                    "saccade.recon_max_elems must be > 0"
                ));
            }
            validate_input_projection(&saccade.input_projection)?;
            if saccade.fovea_subpatch_size > 0
                && saccade.fovea_subpatch_size > vision.patch_size
            {
                return Err(anyhow!(
                    "saccade.fovea_subpatch_size ({}) must be <= vision.patch_size ({})",
                    saccade.fovea_subpatch_size,
                    vision.patch_size
                ));
            }
            if matches!(saccade.pyramid_feature_dim, Some(0)) {
                return Err(anyhow!(
                    "saccade.pyramid_feature_dim must be > 0 when set"
                ));
            }
            if saccade.cache.max_entries == 0 {
                return Err(anyhow!("saccade.cache.max_entries must be > 0"));
            }
            if saccade.policy.info_reward.stride == 0 {
                return Err(anyhow!(
                    "saccade.policy.info_reward.stride must be > 0"
                ));
            }
            if saccade.policy.location_embedding.quantize_bins < 2 {
                return Err(anyhow!(
                    "saccade.policy.location_embedding.quantize_bins must be >= 2"
                ));
            }
            validate_recon_loss("saccade.loss.recon", &saccade.loss.recon)?;
            validate_lejepa_loss(&saccade.loss.lejepa)?;

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
                    return Err(anyhow!(
                        "saccade.policy.gdpo.hard_weight must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.easy_weight < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.easy_weight must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.policy_weight < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.policy_weight must be >= 0"
                    ));
                }
                if saccade.policy.gdpo.policy_clip_range < 0.0 {
                    return Err(anyhow!(
                        "saccade.policy.gdpo.policy_clip_range must be >= 0"
                    ));
                }
                if let GdpoHardGate::Percentile { quantile } = saccade.policy.gdpo.hard_gate
                {
                    if !(0.0..=1.0).contains(&quantile) {
                        return Err(anyhow!(
                            "saccade.policy.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                            quantile
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

fn validate_input_projection(config: &VisionSaccadeInputProjectionConfig) -> Result<()> {
    match config {
        VisionSaccadeInputProjectionConfig::Linear => Ok(()),
        VisionSaccadeInputProjectionConfig::Cnn(cfg) => {
            if matches!(cfg.channels, Some(0)) {
                return Err(anyhow!("saccade.input_projection.channels must be > 0 when set"));
            }
            if cfg.expansion == 0 {
                return Err(anyhow!("saccade.input_projection.expansion must be > 0"));
            }
            if cfg.kernel != 0 && cfg.kernel % 2 == 0 {
                return Err(anyhow!("saccade.input_projection.kernel must be odd when set"));
            }
            Ok(())
        }
        VisionSaccadeInputProjectionConfig::RadialMicroVit(cfg) => {
            if cfg.mlp_ratio == 0 {
                return Err(anyhow!("saccade.input_projection.mlp_ratio must be > 0"));
            }
            if cfg.radial_scale <= 0.0 {
                return Err(anyhow!(
                    "saccade.input_projection.radial_scale must be > 0"
                ));
            }
            Ok(())
        }
    }
}

fn validate_recon_loss(label: &str, loss: &VisionReconLossConfig) -> Result<()> {
    if !(0.0..=1.0).contains(&loss.mask_ratio) {
        return Err(anyhow!(
            "{label}.mask_ratio must be in [0, 1] (got {})",
            loss.mask_ratio
        ));
    }
    if loss.weight < 0.0 {
        return Err(anyhow!("{label}.weight must be >= 0"));
    }
    if loss.hidden_dim == 0 {
        return Err(anyhow!("{label}.hidden_dim must be > 0"));
    }
    Ok(())
}

fn validate_lejepa_loss(loss: &VisionLejepaLossConfig) -> Result<()> {
    if loss.enabled {
        if !(0.0..=1.0).contains(&loss.lambda) {
            return Err(anyhow!(
                "mode.loss.lejepa.lambda must be in [0, 1] (got {})",
                loss.lambda
            ));
        }
        if loss.sigreg_knots == 0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_knots must be > 0"));
        }
        if loss.sigreg_t_max <= 0.0 {
            return Err(anyhow!("mode.loss.lejepa.sigreg_t_max must be > 0"));
        }
        if loss.sigreg_proj_dim == 0 {
            return Err(anyhow!(
                "mode.loss.lejepa.sigreg_proj_dim must be > 0"
            ));
        }
    }
    Ok(())
}

fn validate_distill_loss(loss: &VisionDistillationLossConfig) -> Result<()> {
    if loss.patch_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.patch_mse_weight must be >= 0"));
    }
    if loss.cls_mse_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_mse_weight must be >= 0"));
    }
    if loss.cls_cosine_weight < 0.0 {
        return Err(anyhow!("mode.loss.cls_cosine_weight must be >= 0"));
    }
    if loss.rel_weight < 0.0 {
        return Err(anyhow!("mode.loss.rel_weight must be >= 0"));
    }
    if loss.rel_tau <= 0.0 {
        return Err(anyhow!("mode.loss.rel_tau must be > 0"));
    }
    if matches!(loss.rel_sample_tokens, Some(0)) {
        return Err(anyhow!("mode.loss.rel_sample_tokens must be > 0 when set"));
    }
    Ok(())
}

fn load_value(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration file {}", path.display()))?;
    let table: toml::value::Table = toml::from_str(&content)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
    Ok(Value::Table(table))
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distill_mode_parses() {
        let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2
            rollout_min_steps = 2
            rollout_max_steps = 3
            rollout_backprop_steps = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "distill"

            [mode.teacher]
            type = "features"
            train_cls_path = "train_cls.bin"
            train_patch_path = "train_patch.bin"
            val_cls_path = "val_cls.bin"
            val_patch_path = "val_patch.bin"
            feature_dim = 384
            patch_tokens = 256
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse distill config");
        assert_eq!(config.training.rollout_min_steps, Some(2));
        assert_eq!(config.training.rollout_max_steps, Some(3));
        assert_eq!(config.training.rollout_backprop_steps, Some(2));
        match config.mode {
            VisionTrainingModeConfig::Distill(distill) => match distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    assert_eq!(teacher.feature_dim, 384);
                    assert_eq!(teacher.patch_tokens, Some(256));
                }
                other => panic!("unexpected teacher config: {other:?}"),
            },
            other => panic!("unexpected mode: {other:?}"),
        }
    }

    #[test]
    fn lejepa_mode_parses() {
        let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2
            rollout_min_steps = 1
            rollout_max_steps = 4
            rollout_backprop_steps = 1

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "lejepa"
            views = 4
            global_views = 2
            local_views = 6
            local_image_size = 96
            local_min_scale = 0.05
            local_max_scale = 0.3
            artifact_output = "avi"
            artifact_fps = 6
            artifact_every = 5
            artifact_max_images = 3
            artifact_max_views = 2
            artifact_overwrite = true

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128

            [mode.loss.recon]
            weight = 0.7
            mask_ratio = 0.6
            hidden_dim = 192
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse lejepa config");
        assert_eq!(config.training.rollout_min_steps, Some(1));
        assert_eq!(config.training.rollout_max_steps, Some(4));
        assert_eq!(config.training.rollout_backprop_steps, Some(1));
        match config.mode {
            VisionTrainingModeConfig::Lejepa(lejepa) => {
                assert!(lejepa.loss.lejepa.enabled);
                assert!((lejepa.loss.lejepa.lambda - 0.05).abs() < f32::EPSILON);
                assert_eq!(lejepa.loss.lejepa.sigreg_knots, 19);
                assert!((lejepa.loss.lejepa.sigreg_t_max - 2.5).abs() < f32::EPSILON);
                assert_eq!(lejepa.loss.lejepa.sigreg_proj_dim, 128);
                assert!((lejepa.loss.recon.weight - 0.7).abs() < f32::EPSILON);
                assert!((lejepa.loss.recon.mask_ratio - 0.6).abs() < f32::EPSILON);
                assert_eq!(lejepa.loss.recon.hidden_dim, 192);
                assert_eq!(lejepa.views, 4);
                assert_eq!(lejepa.global_views, 2);
                assert_eq!(lejepa.local_views, 6);
                assert_eq!(lejepa.local_image_size, 96);
                assert!((lejepa.local_min_scale - 0.05).abs() < f32::EPSILON);
                assert!((lejepa.local_max_scale - 0.3).abs() < f32::EPSILON);
                assert_eq!(lejepa.artifact_output, VisionArtifactOutputMode::Avi);
                assert_eq!(lejepa.artifact_fps, 6);
                assert_eq!(lejepa.artifact_every, 5);
                assert_eq!(lejepa.artifact_max_images, 3);
                assert_eq!(lejepa.artifact_max_views, 2);
                assert!(lejepa.artifact_overwrite);
            }
            other => panic!("unexpected mode: {other:?}"),
        }
    }

    #[test]
    fn mae_mode_parses() {
        let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "mae"
            artifact_output = "images"
            artifact_fps = 5
            artifact_every = 3
            artifact_max_images = 2
            artifact_max_views = 1
            artifact_overwrite = true

            [mode.loss.recon]
            weight = 1.2
            mask_ratio = 0.8
            hidden_dim = 192
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse mae config");
        match config.mode {
            VisionTrainingModeConfig::Mae(mae) => {
                assert!((mae.loss.recon.mask_ratio - 0.8).abs() < f32::EPSILON);
                assert!((mae.loss.recon.weight - 1.2).abs() < f32::EPSILON);
                assert_eq!(mae.loss.recon.hidden_dim, 192);
                assert_eq!(mae.artifact_output, VisionArtifactOutputMode::Images);
                assert_eq!(mae.artifact_fps, 5);
                assert_eq!(mae.artifact_every, 3);
                assert_eq!(mae.artifact_max_images, 2);
                assert_eq!(mae.artifact_max_views, 1);
                assert!(mae.artifact_overwrite);
            }
            other => panic!("unexpected mode: {other:?}"),
        }
    }

    #[test]
    fn saccade_mode_parses() {
        let text = r#"
            [dataset]
            imagenet_root = "data/imagenet1k"
            train_dir = "train"
            val_dir = "val"

            [training]
            batch_size = 8
            max_iters = 10
            log_frequency = 2

            [optimizer]
            learning_rate = 0.001
            weight_decay = 0.1

            [vision]
            image_size = 224
            patch_size = 14
            in_channels = 3
            embed_dim = 256
            steps = 4
            n_head = 4
            mlp_internal_dim_multiplier = 4
            dropout = 0.1
            projection_dim = 384
            projection_hidden_dim = 512
            use_cls_token = true
            pos_encoding = "learned2d"
            attention_mode = "row_l1"
            fused_kernels = false
            relu_threshold = 0.0

            [mode]
            type = "saccade"
            num_eyes = 2
            mip_levels = 4
            pyramid_mode = "laplacian"
            fovea_sampling_mode = "subpatch"
            fovea_warp_mode = "patched"
            fovea_subpatch_size = 12
            inner_steps = 2
            artifact_output = "avi"
            artifact_fps = 7
            artifact_every = 4
            artifact_max_images = 3
            artifact_max_views = 2
            artifact_overwrite = false

            [mode.loss.lejepa]
            enabled = true
            lambda = 0.05
            sigreg_knots = 9
            sigreg_t_max = 2.0
            sigreg_proj_dim = 192

            [mode.loss.recon]
            weight = 0.9
            mask_ratio = 0.7
            hidden_dim = 320
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse saccade config");
        match config.mode {
            VisionTrainingModeConfig::Saccade(saccade) => {
                assert_eq!(saccade.num_eyes, 2);
                assert_eq!(saccade.mip_levels, 4);
                assert_eq!(saccade.pyramid_mode, VisionPyramidMode::Laplacian);
                assert_eq!(saccade.fovea_sampling_mode, VisionFoveaSamplingMode::Subpatch);
                assert_eq!(saccade.fovea_warp_mode, VisionFoveaWarpMode::Patched);
                assert_eq!(saccade.fovea_subpatch_size, 12);
                assert_eq!(saccade.inner_steps, 2);
                assert!(saccade.loss.lejepa.enabled);
                assert!((saccade.loss.lejepa.lambda - 0.05).abs() < f32::EPSILON);
                assert_eq!(saccade.loss.lejepa.sigreg_knots, 9);
                assert!((saccade.loss.lejepa.sigreg_t_max - 2.0).abs() < f32::EPSILON);
                assert_eq!(saccade.loss.lejepa.sigreg_proj_dim, 192);
                assert!((saccade.loss.recon.weight - 0.9).abs() < f32::EPSILON);
                assert!((saccade.loss.recon.mask_ratio - 0.7).abs() < f32::EPSILON);
                assert_eq!(saccade.loss.recon.hidden_dim, 320);
                assert_eq!(saccade.artifact_output, VisionArtifactOutputMode::Avi);
                assert_eq!(saccade.artifact_fps, 7);
                assert_eq!(saccade.artifact_every, 4);
                assert_eq!(saccade.artifact_max_images, 3);
                assert_eq!(saccade.artifact_max_views, 2);
                assert!(!saccade.artifact_overwrite);
            }
            other => panic!("unexpected mode: {other:?}"),
        }
    }
}
