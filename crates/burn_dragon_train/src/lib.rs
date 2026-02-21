#![recursion_limit = "256"]

pub mod config;
pub mod constants;
pub mod device;
pub mod wgpu;

#[cfg(feature = "train")]
pub mod train;

pub use config::{
    GdpoConfig, GdpoHardGate, VisionTeacherVariant, WgpuBackend, WgpuMemoryConfig,
    WgpuRuntimeConfig,
};
#[cfg(feature = "train")]
pub use config::{LearningRateScheduleConfig, OptimizerConfig, VisionArtifactOutputMode};
