#![cfg_attr(not(feature = "viz"), allow(dead_code, unused_imports))]

//! Bevy-based visualization helpers for Dragon model state and diagnostics.

#[cfg(feature = "viz")]
/// Visualization systems and plugins.
pub mod viz;

#[cfg(feature = "viz")]
pub use viz::*;
