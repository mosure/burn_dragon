#[cfg(feature = "train")]
pub mod artifacts;
pub mod core;
#[cfg(feature = "train")]
pub mod optimizer;

#[cfg(feature = "train")]
pub use artifacts::VisionArtifactOutputMode;
pub use core::{
    GdpoConfig, GdpoHardGate, VisionTeacherVariant, WgpuBackend, WgpuGenerationExecutor,
    WgpuInferenceConfig, WgpuMemoryConfig, WgpuRuntimeConfig, WgpuTrainingConfig,
};
#[cfg(feature = "train")]
pub use optimizer::{LearningRateScheduleConfig, OptimizerConfig};
