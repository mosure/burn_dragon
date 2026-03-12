#![recursion_limit = "256"]

//! Top-level Dragon framework facade.
//!
//! Preferred usage:
//! - import the specific domain crate through `burn_dragon::{core, language, vision, graph, sudoku}`
//! - or use the curated `burn_dragon::api::*` modules for the smaller recommended surface
//! - use `burn_dragon::api::expert::*` only when the stable-ish curated surface is insufficient

pub mod api {
    //! Curated top-level framework surface.
    //!
    //! These modules are intended as the preferred import paths for library users.

    pub use burn_dragon_checkpoint::api as checkpoint;
    pub use burn_dragon_core::api as core;
    pub use burn_dragon_graph::api as graph;
    pub use burn_dragon_language::api as language;
    pub use burn_dragon_sudoku::api as sudoku;
    #[cfg(feature = "train")]
    pub use burn_dragon_train::api as train;
    pub use burn_dragon_vision::api as vision;
    pub use burn_dragon_wgpu::api as wgpu;

    pub mod expert {
        //! Lower-level crate surfaces for advanced callers.

        pub use burn_dragon_checkpoint;
        pub use burn_dragon_core::api::expert as core;
        pub use burn_dragon_graph::api::expert as graph;
        pub use burn_dragon_language;
        pub use burn_dragon_sudoku;
        pub use burn_dragon_vision;
        pub use burn_dragon_wgpu::api::expert as wgpu;
    }
}

pub use burn_dragon_checkpoint as checkpoint;
pub use burn_dragon_core as core;
pub use burn_dragon_graph as graph;
#[cfg(feature = "viz")]
pub use bevy_dragon as viz;
pub use burn_dragon_language as language;
pub use burn_dragon_sudoku as sudoku;
#[cfg(feature = "train")]
pub use burn_dragon_train as train;
pub use burn_dragon_vision as vision;
pub use burn_dragon_wgpu as wgpu;
#[cfg(feature = "web")]
pub use burn_dragon_web as web;
