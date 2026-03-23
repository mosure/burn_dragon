use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResidualConnectorKind {
    #[default]
    Vanilla,
    Mhc,
    AttentionResidual,
    BlockAttentionResidual,
}

impl<B: Backend> Module<B> for ResidualConnectorKind {
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

impl<B: AutodiffBackend> AutodiffModule<B> for ResidualConnectorKind {
    type InnerModule = ResidualConnectorKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for ResidualConnectorKind {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("ResidualConnectorKind")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for ResidualConnectorKind {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct AttentionResidualConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub last_layers: Option<usize>,
    #[serde(default = "default_attention_residual_num_heads")]
    pub num_heads: usize,
    #[serde(default)]
    pub history_window: Option<usize>,
    #[serde(default = "default_attention_residual_dropout")]
    pub dropout: f64,
    #[serde(default = "default_attention_residual_recency_bias")]
    pub recency_bias: f32,
}

impl Default for AttentionResidualConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            last_layers: None,
            num_heads: default_attention_residual_num_heads(),
            history_window: None,
            dropout: default_attention_residual_dropout(),
            recency_bias: default_attention_residual_recency_bias(),
        }
    }
}

const fn default_attention_residual_num_heads() -> usize {
    4
}

const fn default_attention_residual_dropout() -> f64 {
    0.0
}

const fn default_attention_residual_recency_bias() -> f32 {
    2.0
}

impl AttentionResidualConfig {
    pub fn resolved_num_heads(&self, dense_dim: usize) -> usize {
        self.num_heads.max(1).min(dense_dim.max(1))
    }
}

impl<B: Backend> Module<B> for AttentionResidualConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for AttentionResidualConfig {
    type InnerModule = AttentionResidualConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for AttentionResidualConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, last_layers={}, num_heads={}, history_window={}, dropout={}, recency_bias={}",
            self.enabled,
            self.last_layers
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.num_heads,
            self.history_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.dropout,
            self.recency_bias,
        );

        content
            .set_top_level_type("AttentionResidualConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for AttentionResidualConfig {}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BlockAttentionResidualSummaryMode {
    MeanPool,
    #[default]
    LearnedProjection,
}

impl<B: Backend> Module<B> for BlockAttentionResidualSummaryMode {
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

impl<B: AutodiffBackend> AutodiffModule<B> for BlockAttentionResidualSummaryMode {
    type InnerModule = BlockAttentionResidualSummaryMode;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for BlockAttentionResidualSummaryMode {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .set_top_level_type("BlockAttentionResidualSummaryMode")
            .add_formatted(&format!("{self:?}"))
            .optional()
    }
}

impl ModuleDisplay for BlockAttentionResidualSummaryMode {}

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct BlockAttentionResidualConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub last_layers: Option<usize>,
    #[serde(default = "default_attention_residual_num_heads")]
    pub num_heads: usize,
    #[serde(default = "default_block_attention_residual_layers_per_block")]
    pub layers_per_block: usize,
    #[serde(default)]
    pub block_history_window: Option<usize>,
    #[serde(default)]
    pub intra_block_history_window: Option<usize>,
    #[serde(default)]
    pub summary_mode: BlockAttentionResidualSummaryMode,
    #[serde(default = "default_attention_residual_dropout")]
    pub dropout: f64,
    #[serde(default = "default_attention_residual_recency_bias")]
    pub recency_bias: f32,
    #[serde(default = "default_true")]
    pub cache_block_summaries: bool,
    #[serde(default = "default_true")]
    pub two_phase_compute: bool,
}

impl Default for BlockAttentionResidualConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            last_layers: None,
            num_heads: default_attention_residual_num_heads(),
            layers_per_block: default_block_attention_residual_layers_per_block(),
            block_history_window: None,
            intra_block_history_window: None,
            summary_mode: BlockAttentionResidualSummaryMode::default(),
            dropout: default_attention_residual_dropout(),
            recency_bias: default_attention_residual_recency_bias(),
            cache_block_summaries: default_true(),
            two_phase_compute: default_true(),
        }
    }
}

const fn default_block_attention_residual_layers_per_block() -> usize {
    2
}

const fn default_true() -> bool {
    true
}

impl BlockAttentionResidualConfig {
    pub fn resolved_num_heads(&self, dense_dim: usize) -> usize {
        self.num_heads.max(1).min(dense_dim.max(1))
    }

    pub fn resolved_layers_per_block(&self) -> usize {
        self.layers_per_block.max(1)
    }

    pub fn resolved_intra_block_history_window(&self) -> usize {
        self.intra_block_history_window
            .unwrap_or_else(|| self.resolved_layers_per_block())
            .max(1)
    }
}

impl<B: Backend> Module<B> for BlockAttentionResidualConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for BlockAttentionResidualConfig {
    type InnerModule = BlockAttentionResidualConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for BlockAttentionResidualConfig {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!(
            "enabled={}, last_layers={}, num_heads={}, layers_per_block={}, block_history_window={}, intra_block_history_window={}, summary_mode={:?}, dropout={}, recency_bias={}, cache_block_summaries={}, two_phase_compute={}",
            self.enabled,
            self.last_layers
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.num_heads,
            self.layers_per_block,
            self.block_history_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "all".to_string()),
            self.intra_block_history_window
                .map(|value| value.to_string())
                .unwrap_or_else(|| "auto".to_string()),
            self.summary_mode,
            self.dropout,
            self.recency_bias,
            self.cache_block_summaries,
            self.two_phase_compute,
        );

        content
            .set_top_level_type("BlockAttentionResidualConfig")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for BlockAttentionResidualConfig {}
