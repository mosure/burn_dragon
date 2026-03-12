use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use crate::kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
use crate::model::mhc::ManifoldHyperConnectionsConfig;
use crate::positional::RotaryEmbedding;

#[derive(Clone, Debug)]
pub struct FusedKernelConfig {
    pub enabled: bool,
    pub wgpu_recurrent_kernel: bool,
    pub wgpu_rollout_fused: bool,
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
            "enabled={}, wgpu_recurrent_kernel={}, wgpu_rollout_fused={}, rotary_embedding={}, relu_threshold={}, rope_theta={}, latent_block={}, time_block={}, custom_alibi={}",
            self.enabled,
            self.wgpu_recurrent_kernel,
            self.wgpu_rollout_fused,
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

#[derive(Clone, Debug)]
pub struct BDHConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub dropout: f64,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub n_expert: usize,
    pub vocab_size: usize,
    /// Number of fast internal recurrent updates to run before each slow token emission.
    /// Valid values: 1, 2, 4, 8, 16.
    pub rollout_fast_steps_per_slow_step: usize,
    pub fused_kernels: FusedKernelConfig,
    pub mhc: ManifoldHyperConnectionsConfig,
    pub y_neuron_recurrence: YNeuronRecurrenceConfig,
}

impl Default for BDHConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_embd: 256,
            dropout: 0.1,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            n_expert: 1,
            vocab_size: 256,
            rollout_fast_steps_per_slow_step: 1,
            fused_kernels: FusedKernelConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            y_neuron_recurrence: YNeuronRecurrenceConfig::default(),
        }
    }
}

impl BDHConfig {
    pub const SUPPORTED_ROLLOUT_FAST_STEPS: [usize; 5] = [1, 2, 4, 8, 16];

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
        let total = self.mlp_internal_dim_multiplier * self.n_embd;
        assert!(
            total.is_multiple_of(self.n_head),
            "latent size must be divisible by the number of heads"
        );
        let latent_per_head = total / self.n_head;
        assert!(
            latent_per_head.is_multiple_of(self.n_expert),
            "latent per head {} must be divisible by experts {}",
            latent_per_head,
            self.n_expert
        );
        latent_per_head
    }

    pub fn latent_total(&self) -> usize {
        self.latent_per_head() * self.n_head
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
}
