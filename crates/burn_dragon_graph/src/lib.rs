#![recursion_limit = "256"]

//! Graph routing and recurrent adapters over the shared Dragon `rho` contract.
//!
//! Paper mapping:
//! - dense node / cluster activations live in dense space
//! - sparse node / cluster / global banks carry persistent `rho`
//! - graph execution adapters project dense state into neuron-space write activations and merge
//!   dense recurrent readouts back into the activation stream

mod compiled_routing;
mod config;
mod executor;
mod model;
mod routing;
mod state;

pub use burn_dragon_core::{
    BankedRhoState, StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern,
    StructuredRouteSpec, StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState,
};
pub use compiled_routing::{CompiledGraphRoute, CompiledGraphRouting};
pub use config::GraphTopologyConfig;
pub use executor::{
    GraphExecutionError, GraphRhoStepConfig, GraphStepInputs, GraphStepOutput, GraphStepReadouts,
    graph_reference_step,
};
pub use model::{GraphDragon, GraphDragonConfig};
pub use routing::{GraphCsrAdjacency, GraphRoutingError, GraphTopologyRouting};
pub use state::{GraphTopologyLayout, GraphTopologyState};
