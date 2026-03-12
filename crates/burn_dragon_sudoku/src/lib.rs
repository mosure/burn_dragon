//! Sudoku task adapters over the shared Dragon BDH core.
//!
//! The paper-faithful `x_neuron` / `y_gate` / `y_neuron` / `rho` contract lives in
//! `burn_dragon_core::BDH`; this crate adds Sudoku-specific policy, cache, and training adapters
//! around that core.

pub mod vocab;

#[cfg(feature = "train")]
pub mod artifacts;
#[cfg(feature = "train")]
pub mod checkpoint;
#[cfg(feature = "train")]
pub mod config;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod model;
#[cfg(feature = "train")]
pub mod train;

pub mod api {
    //! Curated Sudoku-facing Dragon API.

    pub mod vocab {
        pub use crate::vocab::*;
    }

    #[cfg(feature = "train")]
    pub mod checkpoint {
        pub use crate::checkpoint::{
            SudokuBurnpackExportReport, export_sudoku_checkpoint_to_burnpack,
            load_training_snapshot_from_run_dir, write_training_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod config {
        pub use crate::config::*;
    }

    #[cfg(feature = "train")]
    pub mod data {
        pub use crate::dataset::*;
    }

    #[cfg(feature = "train")]
    pub mod model {
        pub use crate::model::*;
    }

    #[cfg(feature = "train")]
    pub mod train {
        pub use crate::artifacts::*;
        pub use crate::train::*;
    }
}

#[cfg(feature = "train")]
pub use checkpoint::{
    SudokuBurnpackExportReport, export_sudoku_checkpoint_to_burnpack,
    load_training_snapshot_from_run_dir, write_training_snapshot,
};
