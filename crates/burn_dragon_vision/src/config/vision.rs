use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};
use toml::Value;

use crate::loss::VisionDistillationLossConfig;
use crate::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionDragonConfig, VisionLatentActivation,
    VisionPatchEmbedMode, VisionTrmGraphConfig, VisionTrmGridMismatchPolicy,
};
use burn_dragon_core::FusedKernelConfig;
use burn_dragon_train::{
    GdpoConfig, GdpoHardGate, OptimizerConfig, VisionArtifactOutputMode, VisionTeacherVariant,
    WgpuRuntimeConfig,
};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionPyramidMode {
    Stacked,
    #[default]
    Laplacian,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaSamplingMode {
    Batched,
    #[default]
    Sequential,
    Subpatch,
    Cubecl,
    Wgsl,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaWarpMode {
    #[default]
    Warped,
    Patched,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionFoveaScatterMode {
    #[default]
    Tensor,
    Cubecl,
    Wgsl,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionLocationEmbeddingMode {
    None,
    Learned,
    Sinusoidal,
    Quantized,
    Rope,
    #[default]
    Pope,
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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionNullGlimpseMode {
    #[default]
    Zero,
    Noise,
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
            return Err(anyhow!("training.trace_train_loss_every must be > 0"));
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
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
        load_validate::validate_vision_rollout(&self.training, self.vision.steps)?;
        load_validate::validate_vision_mode(&self.mode, &self.vision)?;

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
    /// When true, compute reconstruction loss on all patches (not only masked ones).
    #[serde(alias = "full_loss")]
    pub loss_on_all_patches: bool,
    /// When true, force full-patch reconstruction loss during validation only.
    #[serde(alias = "full_loss_valid_only")]
    pub loss_on_all_patches_valid_only: bool,
    /// Enable LayerNorm before the reconstruction head.
    #[serde(alias = "norm")]
    pub recon_head_norm: bool,
    /// Hidden dimension for the recon head MLP (set to 0 for a linear head).
    pub hidden_dim: usize,
}

impl Default for VisionReconLossConfig {
    fn default() -> Self {
        Self {
            weight: 0.0,
            mask_ratio: 0.75,
            loss_on_all_patches: false,
            loss_on_all_patches_valid_only: false,
            recon_head_norm: true,
            hidden_dim: 256,
        }
    }
}

impl VisionReconLossConfig {
    pub(crate) fn loss_on_all_patches_for(&self, is_validation: bool) -> bool {
        if is_validation && self.loss_on_all_patches_valid_only {
            true
        } else {
            self.loss_on_all_patches
        }
    }
}

impl ModuleDisplayDefault for VisionReconLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("mask_ratio", &self.mask_ratio)
            .add("loss_on_all_patches", &self.loss_on_all_patches)
            .add(
                "loss_on_all_patches_valid_only",
                &self.loss_on_all_patches_valid_only,
            )
            .add("recon_head_norm", &self.recon_head_norm)
            .add("hidden_dim", &self.hidden_dim)
            .optional()
    }
}

impl ModuleDisplay for VisionReconLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct VisionLossConfig {
    pub lejepa: VisionLejepaLossConfig,
    pub recon: VisionReconLossConfig,
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
                ..VisionReconLossConfig::default()
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
pub struct VisionMaeCrossViewConfig {
    pub enabled: bool,
    pub min_overlap: f32,
    pub max_attempts: usize,
    /// Index into `vision.num_eyes` for the masked view.
    pub masked_eye: usize,
    pub fuse_alpha: f32,
    pub visible_weight: f32,
}

impl Default for VisionMaeCrossViewConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_overlap: 0.3,
            max_attempts: 10,
            masked_eye: 1,
            fuse_alpha: 0.0,
            visible_weight: 0.0,
        }
    }
}

impl ModuleDisplayDefault for VisionMaeCrossViewConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("min_overlap", &self.min_overlap)
            .add("max_attempts", &self.max_attempts)
            .add("masked_eye", &self.masked_eye)
            .add("fuse_alpha", &self.fuse_alpha)
            .add("visible_weight", &self.visible_weight)
            .optional()
    }
}

impl ModuleDisplay for VisionMaeCrossViewConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeCrossViewConfig {
    pub enabled: bool,
    pub min_overlap: f32,
    pub max_attempts: usize,
    /// Index into `vision.num_eyes` for the masked view.
    pub masked_eye: usize,
}

impl Default for VisionSaccadeCrossViewConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            min_overlap: 0.3,
            max_attempts: 10,
            masked_eye: 1,
        }
    }
}

impl ModuleDisplayDefault for VisionSaccadeCrossViewConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("min_overlap", &self.min_overlap)
            .add("max_attempts", &self.max_attempts)
            .add("masked_eye", &self.masked_eye)
            .optional()
    }
}

impl ModuleDisplay for VisionSaccadeCrossViewConfig {}

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
    pub min_view_overlap: f32,
    pub view_overlap_attempts: usize,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_max_views: usize,
    /// Optional eval-only rollout depth for artifact generation.
    /// `0` uses the normal validation rollout depth.
    pub artifact_rollout_steps: usize,
    /// Optional cap on artifact trajectory frame count.
    /// `0` keeps all rollout frames.
    pub artifact_rollout_frames: usize,
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
            min_view_overlap: 0.0,
            view_overlap_attempts: 1,
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 4,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_max_views: 3,
            artifact_rollout_steps: 0,
            artifact_rollout_frames: 0,
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
            .add("min_view_overlap", &self.min_view_overlap)
            .add("view_overlap_attempts", &self.view_overlap_attempts)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_max_views", &self.artifact_max_views)
            .add("artifact_rollout_steps", &self.artifact_rollout_steps)
            .add("artifact_rollout_frames", &self.artifact_rollout_frames)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .optional()
    }
}

impl ModuleDisplay for VisionLejepaConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionMaeConfig {
    pub loss: VisionMaeLossConfig,
    pub cross_view: VisionMaeCrossViewConfig,
    #[serde(default = "default_mae_pyramid_levels")]
    pub pyramid_levels: usize,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_max_views: usize,
    /// Optional eval-only rollout depth for artifact generation.
    /// `0` uses the normal validation rollout depth.
    pub artifact_rollout_steps: usize,
    /// Optional cap on artifact trajectory frame count.
    /// `0` keeps all rollout frames.
    pub artifact_rollout_frames: usize,
    pub artifact_overwrite: bool,
}

impl Default for VisionMaeConfig {
    fn default() -> Self {
        Self {
            loss: VisionMaeLossConfig::default(),
            cross_view: VisionMaeCrossViewConfig::default(),
            pyramid_levels: default_mae_pyramid_levels(),
            artifact_output: VisionArtifactOutputMode::Images,
            artifact_fps: 4,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_max_views: 3,
            artifact_rollout_steps: 0,
            artifact_rollout_frames: 0,
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
            .add("cross_view", &self.cross_view)
            .add("pyramid_levels", &self.pyramid_levels)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_max_views", &self.artifact_max_views)
            .add("artifact_rollout_steps", &self.artifact_rollout_steps)
            .add("artifact_rollout_frames", &self.artifact_rollout_frames)
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

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct VisionTbpttConfig {
    pub step_count: usize,
}

impl ModuleDisplayDefault for VisionTbpttConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content.add("step_count", &self.step_count).optional()
    }
}

impl ModuleDisplay for VisionTbpttConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionSaccadeInputProjectionConfig {
    #[default]
    Linear,
    Cnn(VisionSaccadeInputProjectionCnnConfig),
    RadialMicroVit(VisionSaccadeInputProjectionMicroVitConfig),
}

impl ModuleDisplayDefault for VisionSaccadeInputProjectionConfig {
    fn content(&self, content: Content) -> Option<Content> {
        match self {
            VisionSaccadeInputProjectionConfig::Linear => content.add("type", "linear").optional(),
            VisionSaccadeInputProjectionConfig::Cnn(cfg) => {
                content.add("type", "cnn").add("config", cfg).optional()
            }
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
    pub cross_view: VisionSaccadeCrossViewConfig,
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
            num_eyes: 0,
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
            cross_view: VisionSaccadeCrossViewConfig::default(),
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
            .add("cross_view", &self.cross_view)
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

fn default_enable_checkpoints() -> bool {
    true
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

fn default_mae_pyramid_levels() -> usize {
    1
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
    pub device_memory_check_every: usize,
    #[serde(default)]
    pub max_device_memory_mb: usize,
    #[serde(default)]
    pub disable_cuda_memory_cleanup: bool,
    #[serde(default = "default_enable_checkpoints")]
    pub enable_checkpoints: bool,
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
            device_memory_check_every: 0,
            max_device_memory_mb: 0,
            disable_cuda_memory_cleanup: false,
            enable_checkpoints: default_enable_checkpoints(),
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
pub struct VisionManifoldHyperConnectionsConfig {
    pub enabled: bool,
    pub num_streams: usize,
    pub num_views: usize,
    pub mhc_iters: usize,
    pub mhc_tau: f32,
    pub add_branch_out_to_residual: bool,
    pub dropout: f64,
}

impl Default for VisionManifoldHyperConnectionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_streams: 0,
            num_views: 0,
            mhc_iters: 10,
            mhc_tau: 0.05,
            add_branch_out_to_residual: true,
            dropout: 0.0,
        }
    }
}

impl ModuleDisplayDefault for VisionManifoldHyperConnectionsConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("num_streams", &self.num_streams)
            .add("num_views", &self.num_views)
            .add("mhc_iters", &self.mhc_iters)
            .add("mhc_tau", &self.mhc_tau)
            .add(
                "add_branch_out_to_residual",
                &self.add_branch_out_to_residual,
            )
            .add("dropout", &self.dropout)
            .optional()
    }
}

impl ModuleDisplay for VisionManifoldHyperConnectionsConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionModelConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub patch_embed_mode: VisionPatchEmbedMode,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub steps: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub dropout: f64,
    pub projection_dim: usize,
    pub projection_hidden_dim: usize,
    pub use_cls_token: bool,
    pub cls_sync_alpha: f32,
    pub num_eyes: usize,
    pub cross_eye_steps: usize,
    /// Enable LayerNorm on the token state/residual stream.
    #[serde(alias = "token_norm")]
    pub token_state_norm: bool,
    pub latent_activation: VisionLatentActivation,
    pub pos_encoding: SpatialPositionalEncodingKind,
    pub pos_max_height: Option<usize>,
    pub pos_max_width: Option<usize>,
    pub attention_mode: VisionAttentionMode,
    /// Allow non-core attention modes (e.g. softmax) for experiments.
    pub allow_softmax_attention: bool,
    /// Enable ALiBi bias on the recurrent time axis.
    pub use_alibi: bool,
    pub fused_kernels: bool,
    pub relu_threshold: f32,
    pub mhc: VisionManifoldHyperConnectionsConfig,
    pub trm_graph: VisionTrmGraphConfig,
}

impl Default for VisionModelConfig {
    fn default() -> Self {
        let image_size = 224;
        let patch_size = 16;
        Self {
            image_size,
            patch_size,
            patch_embed_mode: VisionPatchEmbedMode::default(),
            in_channels: 3,
            embed_dim: 256,
            steps: 6,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            dropout: 0.1,
            projection_dim: 384,
            projection_hidden_dim: 512,
            use_cls_token: true,
            cls_sync_alpha: 0.0,
            num_eyes: 1,
            cross_eye_steps: 0,
            token_state_norm: true,
            latent_activation: VisionLatentActivation::default(),
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: None,
            pos_max_width: None,
            attention_mode: VisionAttentionMode::RowL1,
            allow_softmax_attention: false,
            use_alibi: true,
            fused_kernels: false,
            relu_threshold: 0.0,
            mhc: VisionManifoldHyperConnectionsConfig::default(),
            trm_graph: VisionTrmGraphConfig::default(),
        }
    }
}

impl VisionModelConfig {
    pub fn build(&self) -> VisionDragonConfig {
        let patch_size = self.patch_size.max(1);
        let grid = self.image_size.div_ceil(patch_size);
        let num_eyes = self.num_eyes.max(1);
        let mhc_streams = if self.mhc.num_streams == 0 {
            num_eyes
        } else {
            self.mhc.num_streams
        };
        let mhc_views = if self.mhc.num_views == 0 {
            num_eyes
        } else {
            self.mhc.num_views
        };
        let kernels = FusedKernelConfig {
            enabled: self.fused_kernels,
            relu_threshold: self.relu_threshold,
            ..Default::default()
        };

        VisionDragonConfig {
            image_size: self.image_size,
            patch_size: self.patch_size,
            patch_embed_mode: self.patch_embed_mode,
            in_channels: self.in_channels,
            embed_dim: self.embed_dim,
            steps: self.steps,
            n_head: self.n_head,
            mlp_internal_dim_multiplier: self.mlp_internal_dim_multiplier,
            dropout: self.dropout,
            projection_dim: self.projection_dim,
            projection_hidden_dim: self.projection_hidden_dim,
            use_cls_token: self.use_cls_token,
            cls_sync_alpha: self.cls_sync_alpha,
            num_eyes,
            cross_eye_steps: self.cross_eye_steps,
            token_state_norm: self.token_state_norm,
            latent_activation: self.latent_activation,
            pos_encoding: self.pos_encoding,
            pos_max_height: self.pos_max_height.unwrap_or(grid),
            pos_max_width: self.pos_max_width.unwrap_or(grid),
            attention_mode: if self.allow_softmax_attention {
                self.attention_mode
            } else {
                VisionAttentionMode::RowL1
            },
            use_alibi: self.use_alibi,
            fused_kernels: kernels,
            mhc: burn_dragon_core::ManifoldHyperConnectionsConfig {
                enabled: self.mhc.enabled,
                num_streams: mhc_streams,
                num_views: mhc_views,
                mhc_iters: self.mhc.mhc_iters,
                mhc_tau: self.mhc.mhc_tau,
                add_branch_out_to_residual: self.mhc.add_branch_out_to_residual,
                dropout: self.mhc.dropout,
            },
            trm_graph: self.trm_graph.clone(),
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

mod load_validate;

pub use load_validate::load_vision_training_config;

#[cfg(test)]
mod tests;
