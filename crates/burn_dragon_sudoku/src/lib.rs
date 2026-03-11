//! Sudoku task adapters over the shared Dragon BDH core.
//!
//! The paper-faithful `x_neuron` / `y_gate` / `y_neuron` / `rho` contract lives in
//! `burn_dragon_core::BDH`; this crate adds Sudoku-specific policy, cache, and training adapters
//! around that core.

pub mod vocab;

#[cfg(feature = "train")]
pub mod artifacts;
#[cfg(feature = "train")]
pub mod config;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod model;
#[cfg(feature = "train")]
pub mod train;
