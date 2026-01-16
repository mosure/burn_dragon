#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod prelude;

pub mod artifacts;
pub mod constants;
pub mod gdpo;
pub mod metrics;
pub mod teacher;

pub mod train;

#[cfg(feature = "integration_test")]
pub use gdpo::{gdpo_cpu_fallbacks, gdpo_reset_cpu_fallbacks};
#[cfg(feature = "integration_test")]
pub use metrics::{loss_trace_len, loss_trace_reset, loss_trace_take};
