#![cfg_attr(not(target_arch = "wasm32"), allow(dead_code, unused_imports))]

//! Web/WASM-facing Dragon bindings.

#[cfg(target_arch = "wasm32")]
mod wasm;
#[cfg(target_arch = "wasm32")]
pub use wasm::*;

#[cfg(not(target_arch = "wasm32"))]
mod native;
#[cfg(not(target_arch = "wasm32"))]
pub use native::*;
