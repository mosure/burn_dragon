use std::fmt;
use std::path::PathBuf;

use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use super::distill_config::VisionTeacherConfig;
use burn_dragon_train::VisionArtifactOutputMode;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionRacTeacherKind {
    #[default]
    PooledImage,
    PrecomputedLatent,
}

impl fmt::Display for VisionRacTeacherKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PooledImage => write!(f, "pooled_image"),
            Self::PrecomputedLatent => write!(f, "precomputed_latent"),
        }
    }
}

impl ModuleDisplayDefault for VisionRacTeacherKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionRacTeacherKind {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionRacStateMappingKind {
    #[default]
    ExpandNearest,
    CenteredSubpixel,
}

impl fmt::Display for VisionRacStateMappingKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExpandNearest => write!(f, "expand_nearest"),
            Self::CenteredSubpixel => write!(f, "centered_subpixel"),
        }
    }
}

impl ModuleDisplayDefault for VisionRacStateMappingKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionRacStateMappingKind {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacPrecomputedLatentConfig {
    pub train_path: PathBuf,
    pub val_path: PathBuf,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

impl Default for VisionRacPrecomputedLatentConfig {
    fn default() -> Self {
        Self {
            train_path: PathBuf::new(),
            val_path: PathBuf::new(),
            channels: 4,
            height: 0,
            width: 0,
        }
    }
}

impl ModuleDisplayDefault for VisionRacPrecomputedLatentConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("channels", &self.channels)
            .add("height", &self.height)
            .add("width", &self.width)
            .optional()
    }
}

impl ModuleDisplay for VisionRacPrecomputedLatentConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacTeacherConfig {
    pub kind: VisionRacTeacherKind,
    pub latent_downsample: usize,
    pub state_mapping: VisionRacStateMappingKind,
    pub precomputed_latent: Option<VisionRacPrecomputedLatentConfig>,
}

impl Default for VisionRacTeacherConfig {
    fn default() -> Self {
        Self {
            kind: VisionRacTeacherKind::default(),
            latent_downsample: 4,
            state_mapping: VisionRacStateMappingKind::default(),
            precomputed_latent: None,
        }
    }
}

impl ModuleDisplayDefault for VisionRacTeacherConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("kind", &self.kind)
            .add("latent_downsample", &self.latent_downsample)
            .add("state_mapping", &self.state_mapping)
            .add("precomputed_latent", &self.precomputed_latent)
            .optional()
    }
}

impl ModuleDisplay for VisionRacTeacherConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacSemanticTeacherConfig {
    pub weight: f32,
    pub cls_weight: f32,
    pub patch_weight: f32,
    pub hidden_dim: Option<usize>,
    pub teacher: VisionTeacherConfig,
}

impl Default for VisionRacSemanticTeacherConfig {
    fn default() -> Self {
        Self {
            weight: 0.0,
            cls_weight: 1.0,
            patch_weight: 1.0,
            hidden_dim: None,
            teacher: VisionTeacherConfig::default(),
        }
    }
}

impl ModuleDisplayDefault for VisionRacSemanticTeacherConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("weight", &self.weight)
            .add("cls_weight", &self.cls_weight)
            .add("patch_weight", &self.patch_weight)
            .add("hidden_dim", &self.hidden_dim)
            .optional()
    }
}

impl ModuleDisplay for VisionRacSemanticTeacherConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacMemoryConfig {
    pub observe_steps: usize,
    pub backprop_steps: usize,
    pub flow_backprop_steps: Option<usize>,
    pub reset_each_step: bool,
    pub detach_each_step: bool,
    pub disable_writes: bool,
    pub eval_wipe_after_step: Option<usize>,
}

impl Default for VisionRacMemoryConfig {
    fn default() -> Self {
        Self {
            observe_steps: 1,
            backprop_steps: 1,
            flow_backprop_steps: None,
            reset_each_step: false,
            detach_each_step: false,
            disable_writes: false,
            eval_wipe_after_step: None,
        }
    }
}

impl ModuleDisplayDefault for VisionRacMemoryConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("observe_steps", &self.observe_steps)
            .add("backprop_steps", &self.backprop_steps)
            .add("flow_backprop_steps", &self.flow_backprop_steps)
            .add("reset_each_step", &self.reset_each_step)
            .add("detach_each_step", &self.detach_each_step)
            .add("disable_writes", &self.disable_writes)
            .add("eval_wipe_after_step", &self.eval_wipe_after_step)
            .optional()
    }
}

impl ModuleDisplay for VisionRacMemoryConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacLossConfig {
    pub recon_weight: f32,
    pub path_weight: f32,
    pub reverse_path_weight: f32,
    pub latent_weight: f32,
    pub roundtrip_weight: f32,
    pub roundtrip_state_weight: f32,
    pub state_align_weight: f32,
    pub block_const_weight: f32,
    pub velocity_weight: f32,
    pub probe_weight: f32,
}

impl Default for VisionRacLossConfig {
    fn default() -> Self {
        Self {
            recon_weight: 1.0,
            path_weight: 0.25,
            reverse_path_weight: 0.0,
            latent_weight: 0.5,
            roundtrip_weight: 0.5,
            roundtrip_state_weight: 0.0,
            state_align_weight: 0.0,
            block_const_weight: 0.0,
            velocity_weight: 0.1,
            probe_weight: 0.25,
        }
    }
}

impl ModuleDisplayDefault for VisionRacLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("recon_weight", &self.recon_weight)
            .add("path_weight", &self.path_weight)
            .add("reverse_path_weight", &self.reverse_path_weight)
            .add("latent_weight", &self.latent_weight)
            .add("roundtrip_weight", &self.roundtrip_weight)
            .add("roundtrip_state_weight", &self.roundtrip_state_weight)
            .add("state_align_weight", &self.state_align_weight)
            .add("block_const_weight", &self.block_const_weight)
            .add("velocity_weight", &self.velocity_weight)
            .add("probe_weight", &self.probe_weight)
            .optional()
    }
}

impl ModuleDisplay for VisionRacLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacNoiseConfig {
    pub input_std_scale: f32,
    pub input_std_min: f32,
    pub reverse_noise: bool,
}

impl Default for VisionRacNoiseConfig {
    fn default() -> Self {
        Self {
            input_std_scale: 0.0,
            input_std_min: 0.0,
            reverse_noise: true,
        }
    }
}

impl ModuleDisplayDefault for VisionRacNoiseConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("input_std_scale", &self.input_std_scale)
            .add("input_std_min", &self.input_std_min)
            .add("reverse_noise", &self.reverse_noise)
            .optional()
    }
}

impl ModuleDisplay for VisionRacNoiseConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionRacConfig {
    pub sample_steps: usize,
    pub random_time_grid: bool,
    pub state_channels: usize,
    pub velocity_hidden_dim: usize,
    pub velocity_tanh_scale: f32,
    pub state_clamp_min: f32,
    pub state_clamp_max: f32,
    #[serde(default)]
    pub teacher: VisionRacTeacherConfig,
    #[serde(default)]
    pub semantic_teacher: VisionRacSemanticTeacherConfig,
    #[serde(default)]
    pub memory: VisionRacMemoryConfig,
    #[serde(default)]
    pub loss: VisionRacLossConfig,
    #[serde(default)]
    pub noise: VisionRacNoiseConfig,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_upscale: usize,
    pub artifact_overwrite: bool,
}

impl Default for VisionRacConfig {
    fn default() -> Self {
        Self {
            sample_steps: 4,
            random_time_grid: true,
            state_channels: 3,
            velocity_hidden_dim: 0,
            velocity_tanh_scale: 1.0,
            state_clamp_min: 0.0,
            state_clamp_max: 1.0,
            teacher: VisionRacTeacherConfig::default(),
            semantic_teacher: VisionRacSemanticTeacherConfig::default(),
            memory: VisionRacMemoryConfig::default(),
            loss: VisionRacLossConfig::default(),
            noise: VisionRacNoiseConfig::default(),
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 8,
            artifact_every: 1,
            artifact_max_images: 4,
            artifact_upscale: 4,
            artifact_overwrite: true,
        }
    }
}

impl<B: Backend> Module<B> for VisionRacConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionRacConfig {
    type InnerModule = VisionRacConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionRacConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("sample_steps", &self.sample_steps)
            .add("random_time_grid", &self.random_time_grid)
            .add("state_channels", &self.state_channels)
            .add("velocity_hidden_dim", &self.velocity_hidden_dim)
            .add("velocity_tanh_scale", &self.velocity_tanh_scale)
            .add("state_clamp_min", &self.state_clamp_min)
            .add("state_clamp_max", &self.state_clamp_max)
            .add("teacher", &self.teacher)
            .add("semantic_teacher", &self.semantic_teacher)
            .add("memory", &self.memory)
            .add("loss", &self.loss)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_upscale", &self.artifact_upscale)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .optional()
    }
}

impl ModuleDisplay for VisionRacConfig {}
