use std::fmt;

use burn::module::{Content, ModuleDisplay, ModuleDisplayDefault};
pub use burn_dragon_core::SequenceKernelKind;
use serde::{Deserialize, Serialize};

fn default_parallel_world_size() -> usize {
    1
}

fn default_parallel_group_size() -> usize {
    1
}

fn default_find_unused_parameters() -> bool {
    false
}

fn default_gradient_as_bucket_view() -> bool {
    true
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParallelismKind {
    #[default]
    Single,
    Ddp,
    Fsdp,
    TensorParallelNeuron,
    Hybrid2D,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParallelCommunicationBackend {
    #[default]
    Auto,
    Nccl,
    Gloo,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TensorParallelAxis {
    #[default]
    Neuron,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TensorParallelPartitionKind {
    #[default]
    Contiguous,
    HeadAligned,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ParallelCheckpointFormat {
    #[default]
    UnshardedV1,
    ShardedV2,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum FsdpMixedPrecisionKind {
    #[default]
    Disabled,
    Bf16,
    F16,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ParallelDataConfig {
    #[serde(default = "default_parallel_group_size")]
    pub size: usize,
    pub backend: ParallelCommunicationBackend,
    #[serde(default = "default_find_unused_parameters")]
    pub find_unused_parameters: bool,
    #[serde(default = "default_gradient_as_bucket_view")]
    pub gradient_as_bucket_view: bool,
    #[serde(default)]
    pub collective_num_nodes: Option<u32>,
    #[serde(default)]
    pub collective_global_address: Option<String>,
    #[serde(default)]
    pub collective_node_address: Option<String>,
    #[serde(default)]
    pub collective_data_service_port: Option<u16>,
}

impl Default for ParallelDataConfig {
    fn default() -> Self {
        Self {
            size: default_parallel_group_size(),
            backend: ParallelCommunicationBackend::default(),
            find_unused_parameters: default_find_unused_parameters(),
            gradient_as_bucket_view: default_gradient_as_bucket_view(),
            collective_num_nodes: None,
            collective_global_address: None,
            collective_node_address: None,
            collective_data_service_port: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ParallelTensorConfig {
    #[serde(default = "default_parallel_group_size")]
    pub size: usize,
    pub axis: TensorParallelAxis,
    pub partition: TensorParallelPartitionKind,
    pub sequence_parallel: bool,
}

impl Default for ParallelTensorConfig {
    fn default() -> Self {
        Self {
            size: default_parallel_group_size(),
            axis: TensorParallelAxis::default(),
            partition: TensorParallelPartitionKind::default(),
            sequence_parallel: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct ParallelFsdpConfig {
    pub enabled: bool,
    pub reshard_after_forward: bool,
    pub cpu_offload: bool,
    pub mixed_precision: FsdpMixedPrecisionKind,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
pub struct ParallelCheckpointConfig {
    pub format: ParallelCheckpointFormat,
    pub async_write: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ParallelConfig {
    pub mode: ParallelismKind,
    #[serde(default = "default_parallel_world_size")]
    pub world_size: usize,
    pub data: ParallelDataConfig,
    pub tensor: ParallelTensorConfig,
    pub fsdp: ParallelFsdpConfig,
    pub checkpoint: ParallelCheckpointConfig,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            mode: ParallelismKind::Single,
            world_size: default_parallel_world_size(),
            data: ParallelDataConfig::default(),
            tensor: ParallelTensorConfig::default(),
            fsdp: ParallelFsdpConfig::default(),
            checkpoint: ParallelCheckpointConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ModelSpec {
    pub arch: String,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    pub latent_total: usize,
    pub latent_per_head: usize,
    pub shared_layer_weights: bool,
    pub sequence_kernel: SequenceKernelKind,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct ParallelSpec {
    pub mode: ParallelismKind,
    pub world_size: usize,
    pub data_parallel_size: usize,
    pub tensor_parallel_size: usize,
    pub tensor_parallel_axis: TensorParallelAxis,
    pub tensor_parallel_partition: TensorParallelPartitionKind,
    pub fsdp_enabled: bool,
    pub checkpoint_format: ParallelCheckpointFormat,
    pub collective_num_nodes: Option<u32>,
    pub collective_global_address: Option<String>,
    pub collective_node_address: Option<String>,
    pub collective_data_service_port: Option<u16>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct KernelSpec {
    pub sequence_kernel: SequenceKernelKind,
    pub fused_kernels_enabled: bool,
    pub rollout_fast_steps_per_slow_step: usize,
    pub wgpu_fused_core_recurrent: Option<bool>,
    pub wgpu_fused_core_rollout: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct StateAxisSpec {
    pub name: String,
    pub size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct StateTensorSpec {
    pub name: String,
    pub axes: Vec<StateAxisSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct LayerStateSpec {
    pub layer_index: usize,
    pub latent_total: usize,
    pub latent_per_head: usize,
    pub tensors: Vec<StateTensorSpec>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct StateLayout {
    pub state_family: String,
    pub position_tracked: bool,
    pub layers: Vec<LayerStateSpec>,
}

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
