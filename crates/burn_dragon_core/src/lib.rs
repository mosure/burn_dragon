#![recursion_limit = "256"]

pub mod constants;
pub mod kernel;
pub mod model;
pub mod positional;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
#[cfg(feature = "viz")]
pub use model::LayerVizState;
pub use model::{
    BDH, BDHConfig, FusedKernelConfig, HaltHead, LowRankResidualOutput, ManifoldHyperConnections,
    ManifoldHyperConnectionsConfig, ModelState, lowrank_residual_step, mhc_merge, mhc_passthrough,
    mhc_split,
};
pub use positional::RotaryEmbedding;
