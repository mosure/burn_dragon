pub mod core;
#[cfg(feature = "train")]
pub mod artifacts;
#[cfg(feature = "train")]
pub mod optimizer;

pub use core::{
    GdpoConfig, GdpoHardGate, VisionTeacherVariant, WgpuBackend, WgpuMemoryConfig,
    WgpuRuntimeConfig,
};
#[cfg(feature = "train")]
pub use artifacts::VisionArtifactOutputMode;
#[cfg(feature = "train")]
pub use optimizer::{LearningRateScheduleConfig, OptimizerConfig};
