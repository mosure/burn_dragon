use std::fmt;

use burn::module::{Content, ModuleDisplay, ModuleDisplayDefault};
use serde::{Deserialize, Serialize};

use crate::positional::RotaryEmbedding;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum WgpuBackend {
    Auto,
    Vulkan,
    Dx12,
    Metal,
    #[serde(rename = "opengl")]
    OpenGl,
}

impl Default for WgpuBackend {
    fn default() -> Self {
        Self::Auto
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub enum WgpuMemoryConfig {
    #[serde(rename = "subslices")]
    SubSlices,
    Exclusive,
}

impl Default for WgpuMemoryConfig {
    fn default() -> Self {
        Self::SubSlices
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct WgpuRuntimeConfig {
    pub backend: WgpuBackend,
    pub tasks_max: Option<usize>,
    pub memory: WgpuMemoryConfig,
}

impl Default for WgpuRuntimeConfig {
    fn default() -> Self {
        Self {
            backend: WgpuBackend::default(),
            tasks_max: None,
            memory: WgpuMemoryConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    pub batch_size: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub fast_train: bool,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
    #[serde(default)]
    pub gdpo: Option<GdpoConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct GenerationConfig {
    pub prompt: String,
    #[serde(default)]
    pub max_tokens: Option<i64>,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default)]
    pub top_k: Option<usize>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContextStrategyConfig {
    #[default]
    Infinite,
    Sliding {
        window: usize,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct ModelOverrides {
    pub n_layer: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub mlp_internal_dim_multiplier: Option<usize>,
    pub relu_threshold: Option<f32>,
    pub dropout: Option<f64>,
    pub fused_kernels: Option<bool>,
    pub block_size: Option<usize>,
    pub rotary_embedding: Option<RotaryEmbedding>,
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
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GdpoHardGate {
    Off,
    Fixed {
        threshold: f32,
    },
    Percentile {
        quantile: f32,
    },
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
            .optional()
    }
}

impl ModuleDisplay for GdpoConfig {}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

fn default_temperature() -> f32 {
    1.0
}
