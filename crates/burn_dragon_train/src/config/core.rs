use std::fmt;

use burn::module::{Content, ModuleDisplay, ModuleDisplayDefault};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WgpuBackend {
    #[default]
    Auto,
    Vulkan,
    Dx12,
    Metal,
    #[serde(rename = "opengl")]
    OpenGl,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub enum WgpuMemoryConfig {
    #[serde(rename = "subslices")]
    #[default]
    SubSlices,
    Exclusive,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum WgpuGenerationExecutor {
    #[default]
    Baseline,
    RolloutChunked,
}

fn default_generation_chunk_tokens() -> usize {
    8
}

fn default_generation_device_buffer_tokens() -> usize {
    64
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct WgpuInferenceConfig {
    pub fused_core_recurrent: Option<bool>,
    pub fused_core_rollout: Option<bool>,
    pub generation_executor: WgpuGenerationExecutor,
    #[serde(default = "default_generation_chunk_tokens")]
    pub generation_chunk_tokens: usize,
    #[serde(default = "default_generation_device_buffer_tokens")]
    pub generation_device_buffer_tokens: usize,
}

impl Default for WgpuInferenceConfig {
    fn default() -> Self {
        Self {
            fused_core_recurrent: None,
            fused_core_rollout: None,
            generation_executor: WgpuGenerationExecutor::Baseline,
            generation_chunk_tokens: default_generation_chunk_tokens(),
            generation_device_buffer_tokens: default_generation_device_buffer_tokens(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct WgpuTrainingConfig {
    pub fused_core_recurrent: Option<bool>,
    pub fused_core_rollout: Option<bool>,
    pub startup_autotune: WgpuStartupAutotuneConfig,
}

fn default_startup_autotune_min_batch_size() -> usize {
    1
}

fn default_startup_autotune_probe_steps() -> usize {
    1
}

fn default_startup_autotune_binary_search() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct WgpuStartupAutotuneConfig {
    pub enabled: bool,
    pub target_device_memory_mb: usize,
    #[serde(default = "default_startup_autotune_min_batch_size")]
    pub min_batch_size: usize,
    pub max_batch_size: Option<usize>,
    #[serde(default = "default_startup_autotune_probe_steps")]
    pub probe_steps: usize,
    #[serde(default = "default_startup_autotune_binary_search")]
    pub binary_search: bool,
}

impl Default for WgpuStartupAutotuneConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target_device_memory_mb: 0,
            min_batch_size: default_startup_autotune_min_batch_size(),
            max_batch_size: None,
            probe_steps: default_startup_autotune_probe_steps(),
            binary_search: default_startup_autotune_binary_search(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct WgpuRuntimeConfig {
    pub backend: WgpuBackend,
    pub tasks_max: Option<usize>,
    pub memory: WgpuMemoryConfig,
    pub training: WgpuTrainingConfig,
    pub inference: WgpuInferenceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct GdpoConfig {
    pub enabled: bool,
    pub group_size: usize,
    pub hard_weight: f32,
    pub easy_weight: f32,
    pub policy_weight: f32,
    pub policy_clip_range: f32,
    pub hard_gate: GdpoHardGate,
    pub norm_epsilon: f32,
    pub advantage_clip: f32,
    pub advantage_ema_decay: f32,
}

impl Default for GdpoConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            group_size: 1,
            hard_weight: 1.0,
            easy_weight: 1.0,
            policy_weight: 1.0,
            policy_clip_range: 0.2,
            hard_gate: GdpoHardGate::Percentile { quantile: 0.5 },
            norm_epsilon: 1e-6,
            advantage_clip: 0.0,
            advantage_ema_decay: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GdpoHardGate {
    Off,
    Fixed { threshold: f32 },
    Percentile { quantile: f32 },
}

impl Default for GdpoHardGate {
    fn default() -> Self {
        Self::Percentile { quantile: 0.5 }
    }
}

impl fmt::Display for GdpoHardGate {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Off => write!(f, "off"),
            Self::Fixed { threshold } => write!(f, "fixed(threshold={threshold:.4})"),
            Self::Percentile { quantile } => write!(f, "percentile(quantile={quantile:.3})"),
        }
    }
}

impl ModuleDisplayDefault for GdpoHardGate {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for GdpoHardGate {}

impl ModuleDisplayDefault for GdpoConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("group_size", &self.group_size)
            .add("hard_weight", &self.hard_weight)
            .add("easy_weight", &self.easy_weight)
            .add("policy_weight", &self.policy_weight)
            .add("policy_clip_range", &self.policy_clip_range)
            .add("hard_gate", &self.hard_gate)
            .add("norm_epsilon", &self.norm_epsilon)
            .add("advantage_clip", &self.advantage_clip)
            .add("advantage_ema_decay", &self.advantage_ema_decay)
            .optional()
    }
}

impl ModuleDisplay for GdpoConfig {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTeacherVariant {
    #[default]
    Vits,
    Vitb,
    Vitl,
    Vitg,
}
