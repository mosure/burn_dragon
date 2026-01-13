#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod prelude;

mod artifacts;
mod cli;
mod constants;
mod gdpo;
mod metrics;
mod teacher;

mod foveation;
mod scatter;
mod saccade;
mod train;
mod vision;
#[cfg(feature = "benchmark")]
pub mod bench;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod lejepa_tests;

#[cfg(feature = "integration_test")]
pub use gdpo::{gdpo_cpu_fallbacks, gdpo_reset_cpu_fallbacks};
#[cfg(feature = "cli")]
pub use cli::run_cli;
#[cfg(feature = "integration_test")]
pub use vision::train::train_vision_backend_for_test;
pub use saccade::SaccadeFoveationSampler;
