#![recursion_limit = "256"]

pub mod config;
pub mod constants;
pub mod device;
pub mod wgpu;

#[cfg(feature = "train")]
pub mod train;

pub mod api {
    //! Curated shared training/runtime API.

    pub mod config {
        pub use crate::config::{
            GdpoConfig, GdpoHardGate, VisionTeacherVariant, WgpuBackend,
            WgpuGenerationExecutor, WgpuInferenceConfig, WgpuMemoryConfig,
            WgpuRuntimeConfig, WgpuStartupAutotuneConfig, WgpuTrainingConfig,
        };
        #[cfg(feature = "train")]
        pub use crate::config::{LearningRateScheduleConfig, OptimizerConfig, VisionArtifactOutputMode};
    }

    pub mod runtime {
        #[cfg(feature = "train")]
        pub use crate::train::runtime::{
            DeviceMemoryUsage, bytes_to_mb, cleanup_device_memory, cleanup_device_memory_allowed,
            device_memory_usage, device_memory_usage_safe,
        };
    }

    pub mod wgpu {
        pub use crate::wgpu::{
            WgpuDevice, apply_wgpu_fused_core_override, init_runtime, is_wgpu_backend_name,
        };
    }

    #[cfg(feature = "train")]
    pub mod expert {
        pub use crate::train;
    }
}

pub use config::{
    GdpoConfig, GdpoHardGate, VisionTeacherVariant, WgpuBackend, WgpuGenerationExecutor,
    WgpuInferenceConfig, WgpuMemoryConfig, WgpuRuntimeConfig, WgpuStartupAutotuneConfig,
    WgpuTrainingConfig,
};
#[cfg(feature = "train")]
pub use config::{LearningRateScheduleConfig, OptimizerConfig, VisionArtifactOutputMode};
