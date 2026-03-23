mod attention;
mod attention_residual;
mod bdh;
mod config;
mod halt;
mod init;
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
pub use bdh::{
    BDH, LanguageMhcLayerDiagnostics, LanguagePipelineState, LogitsProjectionProfileSnapshot,
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
    near_critical_embedding_initializer, near_critical_embedding_std, near_critical_projection_std,
    near_critical_residual_output_std,
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
    LowRankResidualOutput, LowRankResidualProfileSnapshot, lowrank_residual_profile_reset,
    lowrank_residual_profile_snapshot, lowrank_residual_step,
};
pub use sequence::{
    MambaSequenceConfig, SequenceKernelConfig, SequenceKernelFamily, SequenceTrainingExecutor,
};
#[cfg(feature = "viz")]
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
