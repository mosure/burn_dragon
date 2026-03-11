#![recursion_limit = "256"]

pub mod constants;
pub mod kernel;
pub mod model;
pub mod positional;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
#[cfg(feature = "viz")]
pub use model::LayerVizState;
pub use model::{
    BDH, BDHConfig, BankedRhoState, FusedKernelConfig, HaltHead, LowRankResidualOutput,
    ManifoldHyperConnections, ManifoldHyperConnectionsConfig, ModelState, StructuredBankRole,
    StructuredGridState, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState, lowrank_residual_step,
    mhc_merge, mhc_passthrough, mhc_split, near_critical_embedding_initializer,
    near_critical_embedding_std, near_critical_projection_std, near_critical_residual_output_std,
};
pub use positional::RotaryEmbedding;
