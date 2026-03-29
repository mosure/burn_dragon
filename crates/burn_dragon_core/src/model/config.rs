use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
use serde::{Deserialize, Serialize};

use crate::kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
use crate::model::attention_residual::{
    AttentionResidualConfig, BlockAttentionResidualConfig, ResidualConnectorKind,
};
use crate::model::init::BdhInitializationConfig;
use crate::model::low_bit::{LowBitQuantizationConfig, LowBitRhoConfig};
use crate::model::mhc::ManifoldHyperConnectionsConfig;
use crate::model::norm::DragonNormConfig;
use crate::model::sequence::{MambaSequenceConfig, SequenceKernelConfig};
use crate::positional::RotaryEmbedding;

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusedAttentionExecutor {
    ScoresOnly,
    #[default]
    AttentionContext,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FusedProjectionExecutor {
    None,
    XOnly,
    YOnly,
    #[default]
    Both,
}

impl FusedProjectionExecutor {
    pub fn use_x(self) -> bool {
        matches!(self, Self::XOnly | Self::Both)
    }

    pub fn use_y(self) -> bool {
        matches!(self, Self::YOnly | Self::Both)
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct FusedKernelConfig {
    pub enabled: bool,
    pub wgpu_recurrent_kernel: bool,
    pub wgpu_rollout_fused: bool,
    #[serde(default)]
    pub attention_executor: FusedAttentionExecutor,
    #[serde(default)]
    pub projection_executor: FusedProjectionExecutor,
    #[serde(default)]
    pub lowrank_grad_input_executor: LowrankGradInputExecutor,
    pub block_sparse: BlockSparseConfig,
    pub rope_theta: f32,
    pub relu_threshold: f32,
    pub alibi_slopes: Option<Vec<f32>>,
    pub rotary_embedding: RotaryEmbedding,
}

impl Default for FusedKernelConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            wgpu_recurrent_kernel: true,
            wgpu_rollout_fused: false,
            attention_executor: FusedAttentionExecutor::default(),
            projection_executor: FusedProjectionExecutor::default(),
            lowrank_grad_input_executor: LowrankGradInputExecutor::default(),
            block_sparse: BlockSparseConfig::dense(64, 64),
            rope_theta: 65_536.0,
            relu_threshold: 0.0,
            alibi_slopes: None,
            rotary_embedding: RotaryEmbedding::default(),
        }
    }
}

impl FusedKernelConfig {
    pub fn with_block_sizes(mut self, latent: usize, time: usize) -> Self {
        self.set_block_sizes(latent, time);
        self
    }

    pub fn set_block_sizes(&mut self, latent: usize, time: usize) {
        self.block_sparse = BlockSparseConfig {
            latent: BlockPattern1d::dense(latent),
            time: BlockPattern2d::dense(time),
        };
    }

    pub fn set_alibi_slopes(&mut self, slopes: Vec<f32>) {
        self.alibi_slopes = Some(slopes);
    }

    pub fn set_rotary_embedding(&mut self, rotary_embedding: RotaryEmbedding) {
        self.rotary_embedding = rotary_embedding;
    }

    pub fn set_wgpu_recurrent_kernel(&mut self, enabled: bool) {
        self.wgpu_recurrent_kernel = enabled;
    }

    pub fn set_wgpu_rollout_fused(&mut self, enabled: bool) {
        self.wgpu_rollout_fused = enabled;
    }

    pub fn set_attention_executor(&mut self, executor: FusedAttentionExecutor) {
        self.attention_executor = executor;
    }

    pub fn set_projection_executor(&mut self, executor: FusedProjectionExecutor) {
        self.projection_executor = executor;
    }

    pub fn set_lowrank_grad_input_executor(&mut self, executor: LowrankGradInputExecutor) {
        self.lowrank_grad_input_executor = executor;
    }
}

impl<B: Backend> Module<B> for FusedKernelConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for FusedKernelConfig {
    type InnerModule = FusedKernelConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for FusedKernelConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, wgpu_recurrent_kernel={}, wgpu_rollout_fused={}, attention_executor={:?}, projection_executor={:?}, lowrank_grad_input_executor={:?}, rotary_embedding={}, relu_threshold={}, rope_theta={}, latent_block={}, time_block={}, custom_alibi={}",
            self.enabled,
            self.wgpu_recurrent_kernel,
            self.wgpu_rollout_fused,
            self.attention_executor,
            self.projection_executor,
            self.lowrank_grad_input_executor,
            self.rotary_embedding,
            self.relu_threshold,
            self.rope_theta,
            self.block_sparse.latent.block_size(),
            self.block_sparse.time.block_size(),
            self.alibi_slopes.as_ref().map(|s| s.len()).unwrap_or(0)
        );

        content
            .set_top_level_type("FusedKernelConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for FusedKernelConfig {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct YNeuronRecurrenceConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_y_neuron_carry_in_scale")]
    pub carry_in_scale: f32,
    #[serde(default)]
    pub last_layers: Option<usize>,
    #[serde(default = "default_y_neuron_chunk_tokens")]
    pub chunk_tokens: usize,
    #[serde(default = "default_y_neuron_state_decay")]
    pub state_decay: f32,
    #[serde(default = "default_y_neuron_state_update_scale")]
    pub state_update_scale: f32,
    #[serde(default = "default_y_neuron_state_rms_cap")]
    pub state_rms_cap: Option<f32>,
}

impl Default for YNeuronRecurrenceConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            carry_in_scale: default_y_neuron_carry_in_scale(),
            last_layers: None,
            chunk_tokens: default_y_neuron_chunk_tokens(),
            state_decay: default_y_neuron_state_decay(),
            state_update_scale: default_y_neuron_state_update_scale(),
            state_rms_cap: default_y_neuron_state_rms_cap(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct ClockedSlowMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub last_layers: Option<usize>,
    #[serde(default = "default_clocked_slow_chunk_tokens")]
    pub chunk_tokens: usize,
    #[serde(default = "default_clocked_slow_residual_scale")]
    pub residual_scale: f32,
}

impl Default for ClockedSlowMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            last_layers: None,
            chunk_tokens: default_clocked_slow_chunk_tokens(),
            residual_scale: default_clocked_slow_residual_scale(),
        }
    }
}

fn default_clocked_slow_chunk_tokens() -> usize {
    4
}

fn default_clocked_slow_residual_scale() -> f32 {
    1.0
}

impl<B: Backend> Module<B> for ClockedSlowMemoryConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for ClockedSlowMemoryConfig {
    type InnerModule = ClockedSlowMemoryConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for ClockedSlowMemoryConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, last_layers={}, chunk_tokens={}, residual_scale={}",
            self.enabled,
            self.last_layers
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.chunk_tokens,
            self.residual_scale,
        );

        content
            .set_top_level_type("ClockedSlowMemoryConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for ClockedSlowMemoryConfig {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SummaryMemoryConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub last_layers: Option<usize>,
    #[serde(default = "default_summary_memory_chunk_tokens")]
    pub chunk_tokens: usize,
    #[serde(default = "default_summary_memory_residual_scale")]
    pub residual_scale: f32,
    #[serde(default = "default_summary_memory_state_decay")]
    pub state_decay: f32,
    #[serde(default = "default_summary_memory_state_update_scale")]
    pub state_update_scale: f32,
    #[serde(default = "default_summary_memory_surprise_gate_threshold")]
    pub surprise_gate_threshold: f32,
    #[serde(default = "default_summary_memory_surprise_gate_sharpness")]
    pub surprise_gate_sharpness: f32,
    #[serde(default)]
    pub write_trigger_text: Option<String>,
    #[serde(default)]
    pub write_trigger_token_ids: Option<Vec<u32>>,
}

impl Default for SummaryMemoryConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            last_layers: None,
            chunk_tokens: default_summary_memory_chunk_tokens(),
            residual_scale: default_summary_memory_residual_scale(),
            state_decay: default_summary_memory_state_decay(),
            state_update_scale: default_summary_memory_state_update_scale(),
            surprise_gate_threshold: default_summary_memory_surprise_gate_threshold(),
            surprise_gate_sharpness: default_summary_memory_surprise_gate_sharpness(),
            write_trigger_text: None,
            write_trigger_token_ids: None,
        }
    }
}

fn default_summary_memory_chunk_tokens() -> usize {
    32
}

fn default_summary_memory_residual_scale() -> f32 {
    0.25
}

fn default_summary_memory_state_decay() -> f32 {
    0.75
}

fn default_summary_memory_state_update_scale() -> f32 {
    0.5
}

fn default_summary_memory_surprise_gate_threshold() -> f32 {
    0.0
}

fn default_summary_memory_surprise_gate_sharpness() -> f32 {
    8.0
}

impl<B: Backend> Module<B> for SummaryMemoryConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SummaryMemoryConfig {
    type InnerModule = SummaryMemoryConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SummaryMemoryConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, last_layers={}, chunk_tokens={}, residual_scale={}, state_decay={}, state_update_scale={}, surprise_gate_threshold={}, surprise_gate_sharpness={}, write_trigger_text_chars={}, write_trigger_tokens={}",
            self.enabled,
            self.last_layers
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.chunk_tokens,
            self.residual_scale,
            self.state_decay,
            self.state_update_scale,
            self.surprise_gate_threshold,
            self.surprise_gate_sharpness,
            self.write_trigger_text
                .as_ref()
                .map(|value| value.chars().count())
                .unwrap_or(0),
            self.write_trigger_token_ids
                .as_ref()
                .map(|value| value.len())
                .unwrap_or(0),
        );

        content
            .set_top_level_type("SummaryMemoryConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for SummaryMemoryConfig {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LatentFanoutScheduleConfig {
    LateLayer {
        base_latent_total: usize,
        last_layers: usize,
    },
    Geometric {
        min_latent_total: usize,
    },
}

fn default_y_neuron_carry_in_scale() -> f32 {
    0.125
}

fn default_y_neuron_chunk_tokens() -> usize {
    1
}

fn default_y_neuron_state_decay() -> f32 {
    0.5
}

fn default_y_neuron_state_update_scale() -> f32 {
    1.0
}

fn default_y_neuron_state_rms_cap() -> Option<f32> {
    Some(1.0)
}

impl<B: Backend> Module<B> for YNeuronRecurrenceConfig {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for YNeuronRecurrenceConfig {
    type InnerModule = YNeuronRecurrenceConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for YNeuronRecurrenceConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, carry_in_scale={}, last_layers={}, chunk_tokens={}, state_decay={}, state_update_scale={}, state_rms_cap={}",
            self.enabled,
            self.carry_in_scale,
            self.last_layers
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.chunk_tokens,
            self.state_decay,
            self.state_update_scale,
            self.state_rms_cap
                .map(|value| value.to_string())
                .unwrap_or_else(|| "none".to_string())
        );

        content
            .set_top_level_type("YNeuronRecurrenceConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for YNeuronRecurrenceConfig {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BDHConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub dropout: f64,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    #[serde(default)]
    pub initialization: BdhInitializationConfig,
    #[serde(default)]
    pub sequence_kernel: SequenceKernelConfig,
    #[serde(default)]
    pub latent_fanout_schedule: Option<LatentFanoutScheduleConfig>,
    #[serde(default)]
    pub mamba: MambaSequenceConfig,
    pub n_expert: usize,
    pub vocab_size: usize,
    /// Number of fast internal recurrent updates to run before each slow token emission.
    /// Valid values: 1, 2, 4, 8, 16.
    pub rollout_fast_steps_per_slow_step: usize,
    pub fused_kernels: FusedKernelConfig,
    pub normalization: DragonNormConfig,
    #[serde(default)]
    pub residual_connector: ResidualConnectorKind,
    #[serde(default)]
    pub quant: LowBitQuantizationConfig,
    #[serde(default)]
    pub rho: LowBitRhoConfig,
    pub mhc: ManifoldHyperConnectionsConfig,
    #[serde(default)]
    pub attention_residual: AttentionResidualConfig,
    #[serde(default)]
    pub block_attention_residual: BlockAttentionResidualConfig,
    pub y_neuron_recurrence: YNeuronRecurrenceConfig,
    pub clocked_slow_memory: ClockedSlowMemoryConfig,
    pub summary_memory: SummaryMemoryConfig,
}

impl Default for BDHConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_embd: 256,
            dropout: 0.1,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            initialization: BdhInitializationConfig::default(),
            sequence_kernel: SequenceKernelConfig::default(),
            latent_fanout_schedule: None,
            mamba: MambaSequenceConfig::default(),
            n_expert: 1,
            vocab_size: 256,
            rollout_fast_steps_per_slow_step: 1,
            fused_kernels: FusedKernelConfig::default(),
            normalization: DragonNormConfig::default(),
            residual_connector: ResidualConnectorKind::default(),
            quant: LowBitQuantizationConfig::default(),
            rho: LowBitRhoConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            attention_residual: AttentionResidualConfig::default(),
            block_attention_residual: BlockAttentionResidualConfig::default(),
            y_neuron_recurrence: YNeuronRecurrenceConfig::default(),
            clocked_slow_memory: ClockedSlowMemoryConfig::default(),
            summary_memory: SummaryMemoryConfig::default(),
        }
    }
}

impl BDHConfig {
    pub const SUPPORTED_ROLLOUT_FAST_STEPS: [usize; 5] = [1, 2, 4, 8, 16];

    pub fn resolved_residual_connector_kind(&self) -> ResidualConnectorKind {
        match self.residual_connector {
            ResidualConnectorKind::Mhc => ResidualConnectorKind::Mhc,
            ResidualConnectorKind::AttentionResidual => ResidualConnectorKind::AttentionResidual,
            ResidualConnectorKind::BlockAttentionResidual => {
                ResidualConnectorKind::BlockAttentionResidual
            }
            ResidualConnectorKind::Vanilla => {
                if self.block_attention_residual.enabled {
                    ResidualConnectorKind::BlockAttentionResidual
                } else if self.attention_residual.enabled {
                    ResidualConnectorKind::AttentionResidual
                } else if self.mhc.enabled {
                    ResidualConnectorKind::Mhc
                } else {
                    ResidualConnectorKind::Vanilla
                }
            }
        }
    }

    pub fn is_valid_rollout_fast_steps(value: usize) -> bool {
        Self::SUPPORTED_ROLLOUT_FAST_STEPS.contains(&value)
    }

    pub fn set_rollout_fast_steps_per_slow_step(&mut self, value: usize) {
        assert!(
            Self::is_valid_rollout_fast_steps(value),
            "rollout_fast_steps_per_slow_step must be one of {:?} (got {value})",
            Self::SUPPORTED_ROLLOUT_FAST_STEPS
        );
        self.rollout_fast_steps_per_slow_step = value;
    }

    pub fn latent_per_head(&self) -> usize {
        let total = self.max_latent_total();
        assert!(
            total % self.n_head == 0,
            "latent size must be divisible by the number of heads"
        );
        let latent_per_head = total / self.n_head;
        assert!(
            latent_per_head % self.n_expert == 0,
            "latent per head {} must be divisible by experts {}",
            latent_per_head,
            self.n_expert
        );
        latent_per_head
    }

    pub fn latent_total(&self) -> usize {
        self.latent_per_head() * self.n_head
    }

    pub fn max_latent_total(&self) -> usize {
        self.mlp_internal_dim_multiplier * self.n_embd
    }

    pub fn latent_total_for_layer(&self, layer_idx: usize) -> usize {
        let max_total = self.max_latent_total();
        match &self.latent_fanout_schedule {
            None => max_total,
            Some(LatentFanoutScheduleConfig::LateLayer {
                base_latent_total,
                last_layers,
            }) => {
                let first_full_layer = self.n_layer.max(1).saturating_sub((*last_layers).max(1));
                if layer_idx >= first_full_layer {
                    max_total
                } else {
                    *base_latent_total
                }
            }
            Some(LatentFanoutScheduleConfig::Geometric { min_latent_total }) => {
                if self.n_layer <= 1 {
                    return max_total;
                }
                if layer_idx == 0 {
                    return *min_latent_total;
                }
                if layer_idx + 1 >= self.n_layer {
                    return max_total;
                }

                let progress = layer_idx as f64 / (self.n_layer - 1) as f64;
                let raw_total = (*min_latent_total as f64)
                    * ((max_total as f64) / (*min_latent_total as f64)).powf(progress);
                self.snap_latent_total(raw_total.round() as usize, *min_latent_total, max_total)
            }
        }
    }

    pub fn latent_per_head_for_layer(&self, layer_idx: usize) -> usize {
        let total = self.latent_total_for_layer(layer_idx);
        assert!(
            total % self.n_head == 0,
            "layer latent size must be divisible by the number of heads"
        );
        let latent_per_head = total / self.n_head;
        assert!(
            latent_per_head % self.n_expert == 0,
            "layer latent per head {} must be divisible by experts {}",
            latent_per_head,
            self.n_expert
        );
        latent_per_head
    }

    pub fn latent_per_expert(&self) -> usize {
        self.latent_per_head() / self.n_expert
    }

    /// Dragon Hatchling paper terminology: dense/token value space dimension.
    pub fn dense_space_dim(&self) -> usize {
        self.n_embd
    }

    /// Dragon Hatchling paper terminology: total neuron-space dimension.
    pub fn neuron_space_dim(&self) -> usize {
        self.latent_total()
    }

    /// Dragon Hatchling paper terminology: neuron-space dimension per head.
    pub fn neuron_space_dim_per_head(&self) -> usize {
        self.latent_per_head()
    }

    pub fn validate_latent_fanout_schedule(
        &self,
        schedule: &LatentFanoutScheduleConfig,
    ) -> Result<(), String> {
        let max_total = self.max_latent_total();
        let quantum = self.latent_total_quantum();
        let validate_total = |label: &str, total: usize| -> Result<(), String> {
            if total == 0 {
                return Err(format!("model.latent_fanout_schedule.{label} must be > 0"));
            }
            if total > max_total {
                return Err(format!(
                    "model.latent_fanout_schedule.{label} must be <= model.latent_total (got {} > {})",
                    total, max_total
                ));
            }
            if total % quantum != 0 {
                return Err(format!(
                    "model.latent_fanout_schedule.{label} must be divisible by lcm(n_embd, n_head*n_expert) (got total={} n_embd={} n_head={} n_expert={})",
                    total, self.n_embd, self.n_head, self.n_expert
                ));
            }
            Ok(())
        };

        match schedule {
            LatentFanoutScheduleConfig::LateLayer {
                base_latent_total,
                last_layers,
            } => {
                validate_total("base_latent_total", *base_latent_total)?;
                if *last_layers == 0 {
                    return Err("model.latent_fanout_schedule.last_layers must be > 0".to_string());
                }
            }
            LatentFanoutScheduleConfig::Geometric { min_latent_total } => {
                validate_total("min_latent_total", *min_latent_total)?;
            }
        }

        Ok(())
    }

    fn latent_total_quantum(&self) -> usize {
        lcm(
            self.n_embd.max(1),
            self.n_head.max(1) * self.n_expert.max(1),
        )
    }

    fn snap_latent_total(&self, total: usize, min_total: usize, max_total: usize) -> usize {
        let quantum = self.latent_total_quantum();
        let clamped = total.clamp(min_total, max_total);
        let lower = ((clamped / quantum) * quantum).max(min_total);
        let upper = lower.saturating_add(quantum).min(max_total);
        if upper < min_total {
            return min_total;
        }
        if clamped.saturating_sub(lower) <= upper.saturating_sub(clamped) {
            lower
        } else {
            upper
        }
    }
}

fn gcd(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn lcm(lhs: usize, rhs: usize) -> usize {
    lhs.max(1) / gcd(lhs.max(1), rhs.max(1)) * rhs.max(1)
}

#[cfg(test)]
mod tests {
    use super::{BDHConfig, LatentFanoutScheduleConfig};

    #[test]
    fn late_layer_schedule_uses_base_then_full_latent_total() {
        let config = BDHConfig {
            n_layer: 8,
            n_embd: 256,
            n_head: 4,
            mlp_internal_dim_multiplier: 128,
            latent_fanout_schedule: Some(LatentFanoutScheduleConfig::LateLayer {
                base_latent_total: 8192,
                last_layers: 4,
            }),
            ..Default::default()
        };

        let totals = (0..config.n_layer)
            .map(|layer_idx| config.latent_total_for_layer(layer_idx))
            .collect::<Vec<_>>();

        assert_eq!(
            totals,
            vec![8192, 8192, 8192, 8192, 32768, 32768, 32768, 32768]
        );
    }

    #[test]
    fn geometric_schedule_is_monotonic_and_hits_endpoints() {
        let config = BDHConfig {
            n_layer: 8,
            n_embd: 256,
            n_head: 4,
            mlp_internal_dim_multiplier: 128,
            latent_fanout_schedule: Some(LatentFanoutScheduleConfig::Geometric {
                min_latent_total: 8192,
            }),
            ..Default::default()
        };

        let totals = (0..config.n_layer)
            .map(|layer_idx| config.latent_total_for_layer(layer_idx))
            .collect::<Vec<_>>();

        assert_eq!(totals.first().copied(), Some(8192));
        assert_eq!(totals.last().copied(), Some(32768));
        assert!(totals.windows(2).all(|window| window[0] <= window[1]));
        assert!(totals.iter().all(|total| total % config.n_embd == 0));
    }
}
