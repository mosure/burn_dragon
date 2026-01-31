mod attention;
mod bdh;
mod config;
mod halt;
mod residual;
mod residual_stream;
mod state;

pub use bdh::BDH;
pub use config::{BDHConfig, FusedKernelConfig};
pub use halt::HaltHead;
#[cfg(feature = "viz")]
pub use state::LayerVizState;
pub use state::{LayerState, ModelState};
pub use residual::{ManifoldHyperConnections, ManifoldHyperConnectionsConfig};
pub use residual_stream::{LowRankResidualOutput, lowrank_residual_step, mhc_merge, mhc_passthrough, mhc_split};

