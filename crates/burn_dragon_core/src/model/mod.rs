mod attention;
mod bdh;
mod config;
mod halt;
mod init;
mod mhc;
mod residual_stream;
mod state;
mod structured_mode;
mod structured_routing;
mod structured_state;
mod structured_step;

pub use bdh::BDH;
pub use config::{BDHConfig, FusedKernelConfig, YNeuronRecurrenceConfig};
pub use halt::HaltHead;
pub use init::{
    near_critical_embedding_initializer, near_critical_embedding_std, near_critical_projection_std,
    near_critical_residual_output_std,
};
pub use mhc::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnections,
    ManifoldHyperConnectionsConfig, mhc_merge, mhc_merge_with_coefficients, mhc_passthrough,
    mhc_passthrough_with_coefficients, mhc_split, mhc_split_with_coefficients,
};
pub use residual_stream::{LowRankResidualOutput, lowrank_residual_step};
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
