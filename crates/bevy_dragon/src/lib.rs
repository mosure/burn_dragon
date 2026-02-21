#![cfg_attr(not(feature = "viz"), allow(dead_code, unused_imports))]

#[cfg(feature = "viz")]
pub mod viz;

#[cfg(feature = "viz")]
pub use viz::*;
