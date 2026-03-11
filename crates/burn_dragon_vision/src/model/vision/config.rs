use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use burn_dragon_core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpatialPositionalEncodingKind {
    None,
    #[default]
    Learned2d,
    SineCosine2d,
    Rope,
}

impl core::fmt::Display for SpatialPositionalEncodingKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for SpatialPositionalEncodingKind {
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

impl<B: AutodiffBackend> AutodiffModule<B> for SpatialPositionalEncodingKind {
    type InnerModule = SpatialPositionalEncodingKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SpatialPositionalEncodingKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for SpatialPositionalEncodingKind {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionAttentionMode {
    #[default]
    RowL1,
    Softmax,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionPatchEmbedMode {
    #[default]
    Conv,
    Linear,
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionBackboneKind {
    /// Parallel dense-space refinement with no persistent `rho` across calls.
    #[default]
    Dense,
    /// Structured multi-bank recurrent refinement over primary/context/global banks.
    Pyramid,
    /// Token-local recurrent `rho` over a bounded neighborhood.
    Cellular,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionLatentActivation {
    /// Paper-aligned positive neuron-space activation for `x_neuron`.
    #[default]
    Relu,
    Gelu,
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTrmGridMismatchPolicy {
    #[default]
    Error,
    FallbackDefault,
}

impl core::fmt::Display for VisionAttentionMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionPatchEmbedMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionBackboneKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionLatentActivation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionTrmGridMismatchPolicy {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for VisionAttentionMode {
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

impl<B: Backend> Module<B> for VisionPatchEmbedMode {
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

impl<B: Backend> Module<B> for VisionBackboneKind {
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

impl<B: Backend> Module<B> for VisionLatentActivation {
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

impl<B: Backend> Module<B> for VisionTrmGridMismatchPolicy {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionAttentionMode {
    type InnerModule = VisionAttentionMode;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionPatchEmbedMode {
    type InnerModule = VisionPatchEmbedMode;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionBackboneKind {
    type InnerModule = VisionBackboneKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionLatentActivation {
    type InnerModule = VisionLatentActivation;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmGridMismatchPolicy {
    type InnerModule = VisionTrmGridMismatchPolicy;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionAttentionMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionPatchEmbedMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionBackboneKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionLatentActivation {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionTrmGridMismatchPolicy {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionAttentionMode {}

impl ModuleDisplay for VisionPatchEmbedMode {}

impl ModuleDisplay for VisionBackboneKind {}

impl ModuleDisplay for VisionLatentActivation {}

impl ModuleDisplay for VisionTrmGridMismatchPolicy {}

impl ModuleDisplayDefault for VisionTrmGraphConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("coarse_stride", &self.coarse_stride)
            .add("hub_count", &self.hub_count)
            .add("rank", &self.rank)
            .add("value_dim", &self.value_dim)
            .add("local_radius", &self.local_radius)
            .add("local_diagonals", &self.local_diagonals)
            .add("local_self", &self.local_self)
            .add("decay", &self.decay)
            .add("hub_gates", &self.hub_gates)
            .add("grid_mismatch_policy", &self.grid_mismatch_policy)
            .optional()
    }
}

impl ModuleDisplay for VisionTrmGraphConfig {}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionTrmGraphConfig {
    pub enabled: bool,
    pub coarse_stride: usize,
    pub hub_count: usize,
    pub rank: usize,
    pub value_dim: usize,
    pub local_radius: usize,
    pub local_diagonals: bool,
    pub local_self: bool,
    pub decay: f32,
    pub hub_gates: bool,
    pub grid_mismatch_policy: VisionTrmGridMismatchPolicy,
}

impl Default for VisionTrmGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coarse_stride: 4,
            hub_count: 1,
            rank: 4,
            value_dim: 32,
            local_radius: 1,
            local_diagonals: false,
            local_self: false,
            decay: 0.9,
            hub_gates: true,
            grid_mismatch_policy: VisionTrmGridMismatchPolicy::Error,
        }
    }
}

impl<B: Backend> Module<B> for VisionTrmGraphConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmGraphConfig {
    type InnerModule = VisionTrmGraphConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionRhoStreamConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("local_radius", &self.local_radius)
            .add("local_diagonals", &self.local_diagonals)
            .add("local_self", &self.local_self)
            .add("decay", &self.decay)
            .add("mode_embeddings", &self.mode_embeddings)
            .add("wgpu_forward_kernel", &self.wgpu_forward_kernel)
            .add("wgpu_rollout_fused", &self.wgpu_rollout_fused)
            .optional()
    }
}

impl ModuleDisplay for VisionRhoStreamConfig {}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionRhoStreamConfig {
    pub enabled: bool,
    pub local_radius: usize,
    pub local_diagonals: bool,
    pub local_self: bool,
    pub decay: f32,
    pub mode_embeddings: bool,
    pub wgpu_forward_kernel: bool,
    pub wgpu_rollout_fused: bool,
}

impl Default for VisionRhoStreamConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            decay: 0.9,
            mode_embeddings: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
        }
    }
}

impl<B: Backend> Module<B> for VisionRhoStreamConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionRhoStreamConfig {
    type InnerModule = VisionRhoStreamConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

#[derive(Clone, Debug)]
pub struct VisionDragonConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub patch_embed_mode: VisionPatchEmbedMode,
    pub backbone: VisionBackboneKind,
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
    pub token_state_norm: bool,
    pub latent_activation: VisionLatentActivation,
    pub pos_encoding: SpatialPositionalEncodingKind,
    pub pos_max_height: usize,
    pub pos_max_width: usize,
    pub attention_mode: VisionAttentionMode,
    pub use_alibi: bool,
    pub fused_kernels: FusedKernelConfig,
    pub mhc: ManifoldHyperConnectionsConfig,
    pub trm_graph: VisionTrmGraphConfig,
    pub rho_stream: VisionRhoStreamConfig,
}

impl Default for VisionDragonConfig {
    fn default() -> Self {
        let image_size: usize = 224;
        let patch_size: usize = 16;
        let grid = image_size.div_ceil(patch_size).max(1);
        Self {
            image_size,
            patch_size,
            patch_embed_mode: VisionPatchEmbedMode::default(),
            backbone: VisionBackboneKind::default(),
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
            pos_max_height: grid,
            pos_max_width: grid,
            attention_mode: VisionAttentionMode::RowL1,
            use_alibi: true,
            fused_kernels: FusedKernelConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            trm_graph: VisionTrmGraphConfig::default(),
            rho_stream: VisionRhoStreamConfig::default(),
        }
    }
}

impl VisionDragonConfig {
    pub fn latent_per_head(&self) -> usize {
        let total = self.mlp_internal_dim_multiplier * self.embed_dim;
        assert!(
            total.is_multiple_of(self.n_head),
            "latent size must be divisible by the number of heads"
        );
        total / self.n_head
    }

    pub fn latent_total(&self) -> usize {
        self.latent_per_head() * self.n_head
    }

    /// Dragon Hatchling paper terminology: dense/token value space dimension.
    pub fn dense_space_dim(&self) -> usize {
        self.embed_dim
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

pub type VisionPyramidConfig = VisionTrmGraphConfig;
pub type VisionCellularConfig = VisionRhoStreamConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchGrid {
    pub height: usize,
    pub width: usize,
}

impl PatchGrid {
    pub fn num_patches(&self) -> usize {
        self.height * self.width
    }
}
