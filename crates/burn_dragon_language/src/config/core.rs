use serde::{Deserialize, Serialize};

use burn_dragon_core::{
    ClockedSlowMemoryConfig, DragonNormConfig, LatentFanoutScheduleConfig,
    MambaSequenceConfig, ManifoldHyperConnectionsConfig, RotaryEmbedding, SequenceKernelKind,
    SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};

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
    #[serde(alias = "neuron_space_dim")]
    pub latent_total: Option<usize>,
    pub sequence_kernel: Option<SequenceKernelKind>,
    pub mamba: Option<MambaSequenceConfig>,
    pub latent_fanout_schedule: Option<LatentFanoutScheduleConfig>,
    pub relu_threshold: Option<f32>,
    pub dropout: Option<f64>,
    pub normalization: Option<DragonNormConfig>,
    pub fused_kernels: Option<bool>,
    pub block_size: Option<usize>,
    #[serde(alias = "rollout_fast_steps")]
    pub rollout_fast_steps_per_slow_step: Option<usize>,
    pub rotary_embedding: Option<RotaryEmbedding>,
    #[serde(alias = "y_sparse_recurrence")]
    pub y_neuron_recurrence: Option<YNeuronRecurrenceConfig>,
    pub clocked_slow_memory: Option<ClockedSlowMemoryConfig>,
    pub summary_memory: Option<SummaryMemoryConfig>,
    pub mhc: Option<ManifoldHyperConnectionsConfig>,
}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

fn default_temperature() -> f32 {
    1.0
}
