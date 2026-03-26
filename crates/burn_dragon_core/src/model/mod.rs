mod attention;
mod attention_residual;
mod bdh;
mod bdh_support;
mod config;
mod halt;
mod init;
mod low_bit;
mod low_bit_runtime;
mod mhc;
mod norm;
mod residual_stream;
mod sequence;
mod state;
mod structured_mode;
mod structured_routing;
mod structured_state;
mod structured_step;

pub use attention_residual::{
    AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
    BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode, ResidualConnectorKind,
};
pub use bdh::BDH;
#[cfg(any(feature = "probe", test))]
pub use bdh_support::LanguageBdhInitLayerDiagnostics;
pub use bdh_support::{
    LanguageMhcLayerDiagnostics, LanguagePipelineState, LogitsProjectionProfileSnapshot,
    logits_projection_profile_reset, logits_projection_profile_snapshot,
};
pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
pub use config::{
    BDHConfig, ClockedSlowMemoryConfig, FusedAttentionExecutor, FusedKernelConfig,
    FusedProjectionExecutor, LatentFanoutScheduleConfig, SequenceKernelKind, SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};
pub use halt::HaltHead;
pub use init::{
    BdhActivationThresholds, BdhFiringTargetConfig, BdhFiringTargetKind, BdhInitializationConfig,
    BdhInitializationKind, BdhInitializer, BdhNeuronGainConfig, BdhNeuronGainKind,
    BdhProjectionRole, BdhResidualScalingConfig, BdhResidualScalingKind, BdhTopologyPriorConfig,
    BdhTopologyPriorKind, near_critical_embedding_initializer, near_critical_embedding_std,
    near_critical_projection_std, near_critical_residual_output_std,
};
pub use low_bit::{
    BitNetLowBitProtocol, LowBitActivationFormat, LowBitActivationGrouping, LowBitInferenceMode,
    LowBitQuantizationConfig, LowBitRhoConfig, LowBitSavedActivationConfig,
    LowBitSavedActivationMode, LowBitTargetModule, LowBitTrainingMode, LowBitWeightFormat,
    LowBitWeightGrouping, RhoCompressionConfig, RhoCompressionInterval, RhoPrecisionConfig,
};
pub use low_bit_runtime::{
    LowBitKernelCapabilities, LowBitKernelFallbackReason, LowBitKernelPlan,
    LowBitKernelRuntimeKind, LowBitMemoryBucketEstimate, LowBitMemoryEstimateInput,
    LowBitNativeProjectionProfileSnapshot, LowBitProjectionPlan, LowBitSavedActivationInventory,
    LowBitSavedActivationRecomputePolicy, LowBitSavedActivationTensorInventoryEntry,
    LowBitTrainingProjectionMemoryProfileSnapshot, LowBitTrainingProjectionMemoryStageSnapshot,
    PackedLowBitProjectionArtifacts, PackedRhoInt8DeviceState, PackedRhoInt8State,
    PackedSavedActivationBuffer, PackedSavedActivationState, RhoCompressionQualityGate,
    RhoCompressionStatsSnapshot, build_low_bit_saved_activation_inventory,
    estimate_low_bit_memory_buckets, fake_quantize_activation_ste, fake_quantize_weight_ste,
    fraction_nonzero, low_bit_kernel_capabilities, low_bit_kernel_capabilities_for_backend_name,
    low_bit_native_decoder_tail_profile_snapshot, low_bit_native_lowrank_profile_snapshot,
    low_bit_native_projection_profile_reset, low_bit_training_lowrank_memory_profile_snapshot,
    pack_saved_activation_state,
    resolve_low_bit_kernel_plan, resolve_low_bit_kernel_plan_for_backend_name,
    rho_compression_profile_reset, rho_compression_profile_snapshot,
    rho_compression_snapshot_passes_gate, unpack_saved_activation_state,
};
pub use mhc::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnectionStreamOutput,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
    mhc_merge, mhc_merge_with_coefficients, mhc_passthrough, mhc_passthrough_with_coefficients,
    mhc_split, mhc_split_with_coefficients,
};
pub use norm::{DragonNorm, DragonNormConfig, DragonNormKind};
pub use residual_stream::{
    LowBitSavedActivationCache, LowRankResidualMemoryProfileSnapshot,
    LowRankResidualMemoryStageSnapshot, LowRankResidualOutput, LowRankResidualProfileSnapshot,
    lowrank_residual_memory_profile_reset, lowrank_residual_memory_profile_snapshot,
    lowrank_residual_profile_reset, lowrank_residual_profile_snapshot, lowrank_residual_step,
    lowrank_residual_step_next,
};
pub use sequence::{
    MambaSequenceConfig, SequenceKernelConfig, SequenceKernelFamily, SequenceTrainingExecutor,
};
#[cfg(any(feature = "viz", feature = "probe"))]
pub use state::LayerVizState;
pub use state::{LayerState, ModelState};
pub use structured_mode::StructuredStepMode;
pub use structured_routing::{
    StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec,
};
pub use structured_state::{BankedRhoState, StructuredGridState, StructuredTopologyState};
pub use structured_step::{
    StructuredDenseUpdateOutput, structured_dense_update_tokens, structured_predict_decay,
    target_major_apply_decay, target_major_decay_add, target_major_identity_read,
    target_major_identity_write, target_major_outer_product,
};
