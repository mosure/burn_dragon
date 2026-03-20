use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use serde::{Deserialize, Serialize};

use burn_dragon_core::{
    DragonNormConfig, FusedKernelConfig, ManifoldHyperConnectionsConfig, StructuredStepMode,
};

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
    /// Hierarchical ConvNeXt-style patch stem with depthwise and pointwise mixing.
    #[default]
    Conv,
    /// Explicit alias for the ConvNeXt-style patch stem used by `Conv`.
    #[serde(alias = "convnext")]
    ConvNext,
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTrmPredictSubstepKind {
    #[default]
    CoarseOnly,
    LocalBridge,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTrmClsReadoutKind {
    #[default]
    PatchMean,
    Hub,
    HubAndCoarse,
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

impl core::fmt::Display for VisionTrmPredictSubstepKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionTrmClsReadoutKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl ModuleDisplayDefault for VisionTrmClsReadoutKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionTrmClsReadoutKind {}

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

impl<B: Backend> Module<B> for VisionTrmPredictSubstepKind {
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

impl<B: Backend> Module<B> for VisionTrmClsReadoutKind {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmPredictSubstepKind {
    type InnerModule = VisionTrmPredictSubstepKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmClsReadoutKind {
    type InnerModule = VisionTrmClsReadoutKind;

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

impl ModuleDisplayDefault for VisionTrmPredictSubstepKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionAttentionMode {}

impl ModuleDisplay for VisionPatchEmbedMode {}

impl ModuleDisplay for VisionBackboneKind {}

impl ModuleDisplay for VisionLatentActivation {}

impl ModuleDisplay for VisionTrmGridMismatchPolicy {}

impl ModuleDisplay for VisionTrmPredictSubstepKind {}

impl ModuleDisplayDefault for VisionTrmGraphBankModeConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("patch_local_read", &self.patch_local_read)
            .add("patch_local_write", &self.patch_local_write)
            .add("coarse_local_read", &self.coarse_local_read)
            .add("coarse_local_write", &self.coarse_local_write)
            .add("patch_from_coarse_read", &self.patch_from_coarse_read)
            .add("patch_from_hub_read", &self.patch_from_hub_read)
            .add("coarse_from_hub_read", &self.coarse_from_hub_read)
            .add("patch_to_coarse_write", &self.patch_to_coarse_write)
            .add("patch_to_global_write", &self.patch_to_global_write)
            .add("coarse_to_global_write", &self.coarse_to_global_write)
            .add("patch_decay_scale", &self.patch_decay_scale)
            .add("coarse_decay_scale", &self.coarse_decay_scale)
            .add("global_decay_scale", &self.global_decay_scale)
            .optional()
    }
}

impl ModuleDisplay for VisionTrmGraphBankModeConfig {}

impl ModuleDisplayDefault for VisionTrmGraphBankScheduleConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("observe", &self.observe)
            .add("refine", &self.refine)
            .add("predict", &self.predict)
            .optional()
    }
}

impl ModuleDisplay for VisionTrmGraphBankScheduleConfig {}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionTrmGraphBankModeConfig {
    pub patch_local_read: bool,
    pub patch_local_write: bool,
    pub coarse_local_read: bool,
    pub coarse_local_write: bool,
    pub patch_from_coarse_read: bool,
    pub patch_from_hub_read: bool,
    pub coarse_from_hub_read: bool,
    pub patch_to_coarse_write: bool,
    pub patch_to_global_write: bool,
    pub coarse_to_global_write: bool,
    pub patch_decay_scale: f32,
    pub coarse_decay_scale: f32,
    pub global_decay_scale: f32,
}

impl Default for VisionTrmGraphBankModeConfig {
    fn default() -> Self {
        Self {
            patch_local_read: true,
            patch_local_write: true,
            coarse_local_read: true,
            coarse_local_write: true,
            patch_from_coarse_read: true,
            patch_from_hub_read: true,
            coarse_from_hub_read: true,
            patch_to_coarse_write: true,
            patch_to_global_write: true,
            coarse_to_global_write: true,
            patch_decay_scale: 1.0,
            coarse_decay_scale: 1.0,
            global_decay_scale: 1.0,
        }
    }
}

impl VisionTrmGraphBankModeConfig {
    /// Predict-time schedule for scene-slot style graph routing.
    ///
    /// The patch path reads its late predict context primarily from the hub / scene-slot bank,
    /// while the coarse path continues local refinement and hub updates.
    pub fn scene_slot_predict_preset() -> Self {
        Self {
            patch_local_read: false,
            patch_local_write: false,
            coarse_local_read: true,
            coarse_local_write: true,
            patch_from_coarse_read: false,
            patch_from_hub_read: true,
            coarse_from_hub_read: true,
            patch_to_coarse_write: false,
            patch_to_global_write: false,
            coarse_to_global_write: true,
            patch_decay_scale: 1.0,
            coarse_decay_scale: 1.0,
            global_decay_scale: 1.0,
        }
    }

    /// Coarse-only bridge substep used between full predict updates.
    ///
    /// This preserves the coarse / hub routing surface while freezing the patch bank for the
    /// intermediate bridge pass.
    pub fn bridge_predict_substep(&self) -> Self {
        Self {
            patch_local_read: false,
            patch_local_write: false,
            coarse_local_read: self.coarse_local_read,
            coarse_local_write: self.coarse_local_write,
            patch_from_coarse_read: false,
            patch_from_hub_read: false,
            coarse_from_hub_read: self.coarse_from_hub_read,
            patch_to_coarse_write: false,
            patch_to_global_write: false,
            coarse_to_global_write: self.coarse_to_global_write,
            patch_decay_scale: self.patch_decay_scale,
            coarse_decay_scale: self.coarse_decay_scale,
            global_decay_scale: self.global_decay_scale,
        }
    }

    /// Local diffusion bridge substep used by the promoted graph-bridge image recipe.
    ///
    /// This keeps patch/coarse local state evolution active while suppressing cross-bank reads
    /// and writes during the intermediate bridge pass.
    pub fn local_bridge_predict_substep(&self) -> Self {
        Self {
            patch_local_read: true,
            patch_local_write: true,
            coarse_local_read: true,
            coarse_local_write: true,
            patch_from_coarse_read: false,
            patch_from_hub_read: false,
            coarse_from_hub_read: false,
            patch_to_coarse_write: false,
            patch_to_global_write: false,
            coarse_to_global_write: false,
            patch_decay_scale: self.patch_decay_scale,
            coarse_decay_scale: self.coarse_decay_scale,
            global_decay_scale: self.global_decay_scale,
        }
    }

    pub fn coarse_only_predict_substep(&self) -> Self {
        self.bridge_predict_substep()
    }
}

#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionTrmGraphBankScheduleConfig {
    pub observe: VisionTrmGraphBankModeConfig,
    pub refine: VisionTrmGraphBankModeConfig,
    pub predict: VisionTrmGraphBankModeConfig,
}

impl ModuleDisplayDefault for VisionTrmGraphConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("coarse_stride", &self.coarse_stride)
            .add("hub_count", &self.hub_count)
            .add("rank", &self.rank)
            .add("patch_rank", &self.patch_rank)
            .add("coarse_rank", &self.coarse_rank)
            .add("global_rank", &self.global_rank)
            .add("value_dim", &self.value_dim)
            .add("local_radius", &self.local_radius)
            .add("local_diagonals", &self.local_diagonals)
            .add("local_self", &self.local_self)
            .add("coarse_local_radius", &self.coarse_local_radius)
            .add("coarse_local_diagonals", &self.coarse_local_diagonals)
            .add("coarse_local_self", &self.coarse_local_self)
            .add("predict_coarse_substeps", &self.predict_coarse_substeps)
            .add("predict_substep_kind", &self.predict_substep_kind)
            .add("cls_readout", &self.cls_readout)
            .add("decay", &self.decay)
            .add("hub_gates", &self.hub_gates)
            .add("bank_schedule", &self.bank_schedule)
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
    pub patch_rank: Option<usize>,
    pub coarse_rank: Option<usize>,
    pub global_rank: Option<usize>,
    pub value_dim: usize,
    pub local_radius: usize,
    pub local_diagonals: bool,
    pub local_self: bool,
    pub coarse_local_radius: Option<usize>,
    pub coarse_local_diagonals: Option<bool>,
    pub coarse_local_self: Option<bool>,
    pub predict_coarse_substeps: usize,
    pub predict_substep_kind: VisionTrmPredictSubstepKind,
    pub cls_readout: VisionTrmClsReadoutKind,
    pub decay: f32,
    pub hub_gates: bool,
    pub bank_schedule: VisionTrmGraphBankScheduleConfig,
    pub grid_mismatch_policy: VisionTrmGridMismatchPolicy,
}

impl Default for VisionTrmGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coarse_stride: 4,
            hub_count: 1,
            rank: 4,
            patch_rank: None,
            coarse_rank: None,
            global_rank: None,
            value_dim: 32,
            local_radius: 1,
            local_diagonals: false,
            local_self: false,
            coarse_local_radius: None,
            coarse_local_diagonals: None,
            coarse_local_self: None,
            predict_coarse_substeps: 1,
            predict_substep_kind: VisionTrmPredictSubstepKind::CoarseOnly,
            cls_readout: VisionTrmClsReadoutKind::PatchMean,
            decay: 0.9,
            hub_gates: true,
            bank_schedule: VisionTrmGraphBankScheduleConfig::default(),
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

impl VisionTrmGraphConfig {
    /// Scene-slot style graph baseline distilled from the current broader-validation image sweeps.
    pub fn scene_slot_graph_preset() -> Self {
        Self {
            enabled: true,
            coarse_stride: 2,
            hub_count: 8,
            rank: 8,
            patch_rank: None,
            coarse_rank: None,
            global_rank: None,
            value_dim: 32,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            coarse_local_radius: Some(1),
            coarse_local_diagonals: Some(true),
            coarse_local_self: Some(true),
            predict_coarse_substeps: 1,
            predict_substep_kind: VisionTrmPredictSubstepKind::CoarseOnly,
            cls_readout: VisionTrmClsReadoutKind::PatchMean,
            decay: 0.9,
            hub_gates: true,
            bank_schedule: VisionTrmGraphBankScheduleConfig {
                observe: VisionTrmGraphBankModeConfig::default(),
                refine: VisionTrmGraphBankModeConfig::default(),
                predict: VisionTrmGraphBankModeConfig::scene_slot_predict_preset(),
            },
            grid_mismatch_policy: VisionTrmGridMismatchPolicy::FallbackDefault,
        }
    }

    /// Promoted graph-bridge preset: scene-slot hubs plus one extra coarse-only bridge substep.
    pub fn scene_slot_graph_bridge_preset() -> Self {
        let mut preset = Self::scene_slot_graph_preset();
        preset.predict_coarse_substeps = 2;
        preset.predict_substep_kind = VisionTrmPredictSubstepKind::LocalBridge;
        preset
    }

    pub fn patch_rank_resolved(&self) -> usize {
        self.patch_rank.unwrap_or(self.rank).max(1)
    }

    pub fn coarse_rank_resolved(&self) -> usize {
        self.coarse_rank.unwrap_or(self.rank).max(1)
    }

    pub fn global_rank_resolved(&self) -> usize {
        self.global_rank.unwrap_or(self.rank).max(1)
    }

    pub fn coarse_local_radius_resolved(&self) -> usize {
        self.coarse_local_radius.unwrap_or(self.local_radius)
    }

    pub fn coarse_local_diagonals_resolved(&self) -> bool {
        self.coarse_local_diagonals.unwrap_or(self.local_diagonals)
    }

    pub fn coarse_local_self_resolved(&self) -> bool {
        self.coarse_local_self.unwrap_or(self.local_self)
    }

    pub fn ranks_uniform(&self) -> bool {
        let patch = self.patch_rank_resolved();
        let coarse = self.coarse_rank_resolved();
        let global = self.global_rank_resolved();
        patch == coarse && coarse == global
    }

    pub fn uses_uniform_local_topology(&self) -> bool {
        self.coarse_local_radius.is_none()
            && self.coarse_local_diagonals.is_none()
            && self.coarse_local_self.is_none()
    }

    pub fn bank_mode(&self, mode: StructuredStepMode) -> &VisionTrmGraphBankModeConfig {
        match mode {
            StructuredStepMode::Observe => &self.bank_schedule.observe,
            StructuredStepMode::Refine => &self.bank_schedule.refine,
            StructuredStepMode::Predict => &self.bank_schedule.predict,
        }
    }

    pub fn uses_default_bank_schedule(&self) -> bool {
        self.bank_schedule == VisionTrmGraphBankScheduleConfig::default()
    }
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

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
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
    pub normalization: DragonNormConfig,
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
            normalization: DragonNormConfig::default(),
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
    /// Construct the matched 224px scene-slot graph control from the current sweeps.
    pub fn scene_slot_graph_baseline_224() -> Self {
        let mut config = Self::default();
        config.apply_scene_slot_graph_baseline_224();
        config
    }

    /// Construct the promoted 224px graph-bridge image baseline from the current sweeps.
    pub fn scene_slot_graph_bridge_baseline_224() -> Self {
        let mut config = Self::default();
        config.apply_scene_slot_graph_bridge_baseline_224();
        config
    }

    /// Construct a medium-width ImageNet-1k graph-bridge launch preset sized for 280px,
    /// multi-teacher distillation against 768-dim target spaces.
    pub fn scene_slot_graph_bridge_multiteacher_medium_280() -> Self {
        let mut config = Self::default();
        config.apply_scene_slot_graph_bridge_multiteacher_medium_280();
        config
    }

    /// Construct a base-width ImageNet-1k graph-bridge launch preset sized for 336px and
    /// 768-dim teacher target spaces.
    pub fn scene_slot_graph_bridge_multiteacher_base_336() -> Self {
        let mut config = Self::default();
        config.apply_scene_slot_graph_bridge_multiteacher_base_336();
        config
    }

    /// Apply the validated scene-slot graph baseline to an existing vision config.
    pub fn apply_scene_slot_graph_preset(&mut self) -> &mut Self {
        self.backbone = VisionBackboneKind::Pyramid;
        self.steps = self.steps.max(3);
        self.trm_graph = VisionTrmGraphConfig::scene_slot_graph_preset();
        self.rho_stream = VisionRhoStreamConfig::default();
        self
    }

    /// Apply the promoted graph-bridge preset derived from the current vision sweeps.
    pub fn apply_scene_slot_graph_bridge_preset(&mut self) -> &mut Self {
        self.backbone = VisionBackboneKind::Pyramid;
        self.steps = self.steps.max(4);
        self.trm_graph = VisionTrmGraphConfig::scene_slot_graph_bridge_preset();
        self.rho_stream = VisionRhoStreamConfig::default();
        self
    }

    /// Apply the matched 224px scene-slot graph control baseline.
    ///
    /// This is the simpler slot-graph control paired against the promoted local-bridge variant.
    pub fn apply_scene_slot_graph_baseline_224(&mut self) -> &mut Self {
        self.image_size = 224;
        self.patch_size = 16;
        self.patch_embed_mode = VisionPatchEmbedMode::default();
        self.backbone = VisionBackboneKind::Pyramid;
        self.in_channels = 3;
        self.embed_dim = 160;
        self.steps = 3;
        self.n_head = 5;
        self.mlp_internal_dim_multiplier = 4;
        self.dropout = 0.1;
        self.projection_dim = 384;
        self.projection_hidden_dim = 512;
        self.use_cls_token = true;
        self.cls_sync_alpha = 0.0;
        self.num_eyes = 1;
        self.cross_eye_steps = 0;
        self.token_state_norm = true;
        self.latent_activation = VisionLatentActivation::default();
        self.pos_encoding = SpatialPositionalEncodingKind::Learned2d;
        let grid = self.image_size.div_ceil(self.patch_size).max(1);
        self.pos_max_height = grid;
        self.pos_max_width = grid;
        self.attention_mode = VisionAttentionMode::RowL1;
        self.use_alibi = true;
        self.apply_scene_slot_graph_preset();
        self
    }

    /// Apply the promoted 224px graph-bridge image baseline.
    ///
    /// This captures the currently strongest small image backbone recipe surfaced by the broader
    /// graph validation work: `patch16 / embed160 / heads5 / steps4 / local_bridge`.
    pub fn apply_scene_slot_graph_bridge_baseline_224(&mut self) -> &mut Self {
        self.image_size = 224;
        self.patch_size = 16;
        self.patch_embed_mode = VisionPatchEmbedMode::default();
        self.backbone = VisionBackboneKind::Pyramid;
        self.in_channels = 3;
        self.embed_dim = 160;
        self.steps = 4;
        self.n_head = 5;
        self.mlp_internal_dim_multiplier = 4;
        self.dropout = 0.1;
        self.projection_dim = 384;
        self.projection_hidden_dim = 512;
        self.use_cls_token = true;
        self.cls_sync_alpha = 0.0;
        self.num_eyes = 1;
        self.cross_eye_steps = 0;
        self.token_state_norm = true;
        self.latent_activation = VisionLatentActivation::default();
        self.pos_encoding = SpatialPositionalEncodingKind::Learned2d;
        let grid = self.image_size.div_ceil(self.patch_size).max(1);
        self.pos_max_height = grid;
        self.pos_max_width = grid;
        self.attention_mode = VisionAttentionMode::RowL1;
        self.use_alibi = true;
        self.apply_scene_slot_graph_bridge_preset();
        self
    }

    /// Apply the medium-width ImageNet-1k graph-bridge launch preset.
    pub fn apply_scene_slot_graph_bridge_multiteacher_medium_280(&mut self) -> &mut Self {
        self.image_size = 280;
        self.patch_size = 14;
        self.patch_embed_mode = VisionPatchEmbedMode::default();
        self.backbone = VisionBackboneKind::Pyramid;
        self.in_channels = 3;
        self.embed_dim = 320;
        self.steps = 4;
        self.n_head = 8;
        self.mlp_internal_dim_multiplier = 4;
        self.dropout = 0.0;
        self.projection_dim = 768;
        self.projection_hidden_dim = 1536;
        self.use_cls_token = true;
        self.cls_sync_alpha = 0.0;
        self.num_eyes = 1;
        self.cross_eye_steps = 0;
        self.token_state_norm = true;
        self.latent_activation = VisionLatentActivation::default();
        self.pos_encoding = SpatialPositionalEncodingKind::Learned2d;
        let grid = self.image_size.div_ceil(self.patch_size).max(1);
        self.pos_max_height = grid;
        self.pos_max_width = grid;
        self.attention_mode = VisionAttentionMode::RowL1;
        self.use_alibi = true;
        self.fused_kernels = FusedKernelConfig::default();
        self.apply_scene_slot_graph_bridge_preset();
        self.trm_graph.hub_count = 12;
        self.trm_graph.rank = 12;
        self.trm_graph.value_dim = 64;
        self
    }

    /// Apply the base-width ImageNet-1k graph-bridge launch preset.
    pub fn apply_scene_slot_graph_bridge_multiteacher_base_336(&mut self) -> &mut Self {
        self.image_size = 336;
        self.patch_size = 14;
        self.patch_embed_mode = VisionPatchEmbedMode::default();
        self.backbone = VisionBackboneKind::Pyramid;
        self.in_channels = 3;
        self.embed_dim = 384;
        self.steps = 5;
        self.n_head = 12;
        self.mlp_internal_dim_multiplier = 4;
        self.dropout = 0.0;
        self.projection_dim = 768;
        self.projection_hidden_dim = 2048;
        self.use_cls_token = true;
        self.cls_sync_alpha = 0.0;
        self.num_eyes = 1;
        self.cross_eye_steps = 0;
        self.token_state_norm = true;
        self.latent_activation = VisionLatentActivation::default();
        self.pos_encoding = SpatialPositionalEncodingKind::Learned2d;
        let grid = self.image_size.div_ceil(self.patch_size).max(1);
        self.pos_max_height = grid;
        self.pos_max_width = grid;
        self.attention_mode = VisionAttentionMode::RowL1;
        self.use_alibi = true;
        self.fused_kernels = FusedKernelConfig::default();
        self.apply_scene_slot_graph_bridge_preset();
        self.trm_graph.hub_count = 16;
        self.trm_graph.rank = 16;
        self.trm_graph.value_dim = 96;
        self
    }

    pub fn latent_per_head(&self) -> usize {
        let total = self.mlp_internal_dim_multiplier * self.embed_dim;
        assert!(
            total % self.n_head == 0,
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
