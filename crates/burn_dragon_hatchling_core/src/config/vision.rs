use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::Deserialize;
use toml::Value;

use crate::model::{FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode};
use crate::model::VisionDistillationLossConfig;

use super::OptimizerConfig;

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
pub struct VisionTrainingConfig {
    pub dataset: VisionDatasetConfig,
    pub training: VisionTrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    pub vision: VisionModelConfig,
    #[serde(default)]
    pub augment: VisionAugmentationConfig,
    #[serde(default)]
    pub mode: VisionTrainingModeConfig,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTeacherVariant {
    #[default]
    Vits,
    Vitb,
    Vitl,
    Vitg,
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionLejepaConfig {
    pub lambda: f32,
    pub sigreg_knots: usize,
    pub sigreg_t_max: f32,
    pub sigreg_proj_dim: usize,
    pub recon_weight: f32,
    pub recon_mask_ratio: f32,
    pub recon_hidden_dim: usize,
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
            lambda: 0.02,
            sigreg_knots: 17,
            sigreg_t_max: 3.0,
            sigreg_proj_dim: 256,
            recon_weight: 0.0,
            recon_mask_ratio: 0.75,
            recon_hidden_dim: 256,
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
            .add("lambda", &self.lambda)
            .add("sigreg_knots", &self.sigreg_knots)
            .add("sigreg_t_max", &self.sigreg_t_max)
            .add("sigreg_proj_dim", &self.sigreg_proj_dim)
            .add("recon_weight", &self.recon_weight)
            .add("recon_mask_ratio", &self.recon_mask_ratio)
            .add("recon_hidden_dim", &self.recon_hidden_dim)
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionMaeConfig {
    pub mask_ratio: f32,
    pub recon_weight: f32,
    pub recon_hidden_dim: usize,
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
            mask_ratio: 0.75,
            recon_weight: 1.0,
            recon_hidden_dim: 256,
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
            .add("mask_ratio", &self.mask_ratio)
            .add("recon_weight", &self.recon_weight)
            .add("recon_hidden_dim", &self.recon_hidden_dim)
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionSaccadeConfig {
    pub num_eyes: usize,
    pub mip_levels: usize,
    pub pyramid_mode: VisionPyramidMode,
    pub fovea_sampling_mode: VisionFoveaSamplingMode,
    pub fovea_warp_mode: VisionFoveaWarpMode,
    pub fovea_subpatch_size: usize,
    pub fovea_scatter_mode: VisionFoveaScatterMode,
    pub pyramid_feature_dim: Option<usize>,
    pub inner_steps: usize,
    pub low_mem_pre_rollout: bool,
    pub lambda: f32,
    pub sigreg_knots: usize,
    pub sigreg_t_max: f32,
    pub sigreg_proj_dim: usize,
    pub recon_weight: f32,
    pub recon_mask_ratio: f32,
    pub recon_hidden_dim: usize,
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
            num_eyes: 2,
            mip_levels: 3,
            pyramid_mode: VisionPyramidMode::Laplacian,
            fovea_sampling_mode: VisionFoveaSamplingMode::Sequential,
            fovea_warp_mode: VisionFoveaWarpMode::Warped,
            fovea_subpatch_size: 0,
            fovea_scatter_mode: VisionFoveaScatterMode::Tensor,
            pyramid_feature_dim: None,
            inner_steps: 1,
            low_mem_pre_rollout: true,
            lambda: 0.02,
            sigreg_knots: 17,
            sigreg_t_max: 3.0,
            sigreg_proj_dim: 256,
            recon_weight: 0.0,
            recon_mask_ratio: 0.75,
            recon_hidden_dim: 256,
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 4,
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
            .add("mip_levels", &self.mip_levels)
            .add("pyramid_mode", &self.pyramid_mode)
            .add("fovea_sampling_mode", &self.fovea_sampling_mode)
            .add("fovea_warp_mode", &self.fovea_warp_mode)
            .add("fovea_subpatch_size", &self.fovea_subpatch_size)
            .add("fovea_scatter_mode", &self.fovea_scatter_mode)
            .add("pyramid_feature_dim", &self.pyramid_feature_dim)
            .add("inner_steps", &self.inner_steps)
            .add("low_mem_pre_rollout", &self.low_mem_pre_rollout)
            .add("lambda", &self.lambda)
            .add("sigreg_knots", &self.sigreg_knots)
            .add("sigreg_t_max", &self.sigreg_t_max)
            .add("sigreg_proj_dim", &self.sigreg_proj_dim)
            .add("recon_weight", &self.recon_weight)
            .add("recon_mask_ratio", &self.recon_mask_ratio)
            .add("recon_hidden_dim", &self.recon_hidden_dim)
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum VisionDatasetDownloadConfig {
    Imagenette {
        #[serde(default)]
        variant: ImagenetteVariant,
    },
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq, Default)]
#[serde(rename_all = "kebab-case")]
pub enum ImagenetteVariant {
    #[default]
    Imagenette2_160,
    Imagenette2_320,
}

fn default_prefetch_batches() -> usize {
    2
}

fn default_prefetch_workers() -> usize {
    std::thread::available_parallelism()
        .map(|count| count.get().min(4))
        .unwrap_or(2)
}

fn default_cache_decoded() -> bool {
    true
}

fn default_cache_capacity() -> usize {
    512
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
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
    #[serde(default = "default_cache_decoded")]
    pub cache_decoded: bool,
    #[serde(default = "default_cache_capacity")]
    pub cache_capacity: usize,
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
            cache_decoded: default_cache_decoded(),
            cache_capacity: default_cache_capacity(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
#[serde(default)]
pub struct VisionTrainingHyperparameters {
    pub batch_size: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub rollout_min_steps: Option<usize>,
    #[serde(default)]
    pub rollout_max_steps: Option<usize>,
    #[serde(default)]
    pub rollout_backprop_steps: Option<usize>,
}

impl Default for VisionTrainingHyperparameters {
    fn default() -> Self {
        Self {
            batch_size: 64,
            epochs: None,
            max_iters: 1000,
            log_frequency: 50,
            rollout_min_steps: None,
            rollout_max_steps: None,
            rollout_backprop_steps: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, PartialEq)]
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
        let grid = (self.image_size / self.patch_size).max(1);
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

#[derive(Debug, Clone, Deserialize, PartialEq)]
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
            lambda = 0.05
            sigreg_knots = 19
            sigreg_t_max = 2.5
            sigreg_proj_dim = 128
            recon_weight = 0.7
            recon_mask_ratio = 0.6
            recon_hidden_dim = 192
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
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse lejepa config");
        assert_eq!(config.training.rollout_min_steps, Some(1));
        assert_eq!(config.training.rollout_max_steps, Some(4));
        assert_eq!(config.training.rollout_backprop_steps, Some(1));
        match config.mode {
            VisionTrainingModeConfig::Lejepa(lejepa) => {
                assert!((lejepa.lambda - 0.05).abs() < f32::EPSILON);
                assert_eq!(lejepa.sigreg_knots, 19);
                assert!((lejepa.sigreg_t_max - 2.5).abs() < f32::EPSILON);
                assert_eq!(lejepa.sigreg_proj_dim, 128);
                assert!((lejepa.recon_weight - 0.7).abs() < f32::EPSILON);
                assert!((lejepa.recon_mask_ratio - 0.6).abs() < f32::EPSILON);
                assert_eq!(lejepa.recon_hidden_dim, 192);
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
            mask_ratio = 0.8
            recon_weight = 1.2
            recon_hidden_dim = 192
            artifact_output = "images"
            artifact_fps = 5
            artifact_every = 3
            artifact_max_images = 2
            artifact_max_views = 1
            artifact_overwrite = true
        "#;

        let config: VisionTrainingConfig = toml::from_str(text).expect("parse mae config");
        match config.mode {
            VisionTrainingModeConfig::Mae(mae) => {
                assert!((mae.mask_ratio - 0.8).abs() < f32::EPSILON);
                assert!((mae.recon_weight - 1.2).abs() < f32::EPSILON);
                assert_eq!(mae.recon_hidden_dim, 192);
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
            lambda = 0.05
            sigreg_knots = 9
            sigreg_t_max = 2.0
            sigreg_proj_dim = 192
            recon_weight = 0.9
            recon_mask_ratio = 0.7
            recon_hidden_dim = 320
            artifact_output = "avi"
            artifact_fps = 7
            artifact_every = 4
            artifact_max_images = 3
            artifact_max_views = 2
            artifact_overwrite = false
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
                assert!((saccade.lambda - 0.05).abs() < f32::EPSILON);
                assert_eq!(saccade.sigreg_knots, 9);
                assert!((saccade.sigreg_t_max - 2.0).abs() < f32::EPSILON);
                assert_eq!(saccade.sigreg_proj_dim, 192);
                assert!((saccade.recon_weight - 0.9).abs() < f32::EPSILON);
                assert!((saccade.recon_mask_ratio - 0.7).abs() < f32::EPSILON);
                assert_eq!(saccade.recon_hidden_dim, 320);
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
