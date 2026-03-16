#![recursion_limit = "256"]

//! Graph routing and recurrent adapters over the shared Dragon `rho` contract.
//!
//! Paper mapping:
//! - dense node / cluster activations live in dense space
//! - sparse node / cluster / global banks carry persistent `rho`
//! - graph execution adapters project dense state into neuron-space write activations and merge
//!   dense recurrent readouts back into the activation stream

mod checkpoint;
mod compiled_executor;
mod compiled_routing;
mod config;
mod executor;
mod model;
mod routing;
mod state;

pub mod api {
    //! Curated graph-facing Dragon API.

    pub mod core {
        pub use burn_dragon_core::api::state::{
            BankedRhoState, StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState,
        };
        pub use burn_dragon_core::{
            StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern,
            StructuredRouteSpec,
        };
    }

    pub mod config {
        pub use crate::config::GraphTopologyConfig;
    }

    pub mod checkpoint {
        pub use crate::checkpoint::{
            GraphBurnpackExportReport, export_graph_checkpoint_to_burnpack,
            load_graph_config_for_checkpoint, load_graph_config_snapshot_from_run_dir,
            write_graph_config_snapshot,
        };
    }

    pub mod routing {
        pub use crate::routing::{GraphCsrAdjacency, GraphRoutingError, GraphTopologyRouting};
        pub use crate::state::{GraphTopologyLayout, GraphTopologyState};
    }

    pub mod execution {
        pub use crate::compiled_executor::GraphCompiledExecutor;
        pub use crate::executor::{
            GraphExecutionError, GraphRhoStepConfig, GraphStepInputs, GraphStepOutput,
            GraphStepReadouts, graph_reference_step,
        };
        pub use crate::model::{GraphDragon, GraphDragonConfig};
    }

    pub mod expert {
        pub use crate::compiled_routing::{CompiledGraphRoute, CompiledGraphRouting};
    }
}

pub use burn_dragon_core::{
    BankedRhoState, StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern,
    StructuredRouteSpec, StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState,
};
pub use checkpoint::{
    GraphBurnpackExportReport, export_graph_checkpoint_to_burnpack,
    load_graph_config_for_checkpoint, load_graph_config_snapshot_from_run_dir,
    write_graph_config_snapshot,
};
pub use compiled_executor::GraphCompiledExecutor;
pub use config::GraphTopologyConfig;
pub use executor::{
    GraphExecutionError, GraphRhoStepConfig, GraphStepInputs, GraphStepOutput, GraphStepReadouts,
    graph_reference_step,
};
pub use model::{GraphDragon, GraphDragonConfig};
pub use routing::{GraphCsrAdjacency, GraphRoutingError, GraphTopologyRouting};
pub use state::{GraphTopologyLayout, GraphTopologyState};
