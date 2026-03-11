mod attention;
mod bdh;
mod config;
mod halt;
mod init;
mod residual;
mod residual_stream;
mod state;
mod structured_mode;
mod structured_routing;
mod structured_state;

pub use bdh::BDH;
pub use config::{BDHConfig, FusedKernelConfig};
pub use halt::HaltHead;
pub use init::{
    near_critical_embedding_initializer, near_critical_embedding_std, near_critical_projection_std,
    near_critical_residual_output_std,
};
pub use residual::{ManifoldHyperConnections, ManifoldHyperConnectionsConfig};
pub use residual_stream::{
    LowRankResidualOutput, lowrank_residual_step, mhc_merge, mhc_passthrough, mhc_split,
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
