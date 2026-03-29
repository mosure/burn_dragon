#![recursion_limit = "256"]

//! Shared Dragon training/runtime helpers.
//!
//! Preferred library-facing surface:
//! - [`api::config`] for backend/runtime/training config types
//! - [`api::runtime`] for train-time memory helpers
//! - [`api::wgpu`] for backend/device initialization helpers

#[cfg(feature = "cli")]
/// Shared CLI helpers for train binaries.
pub mod cli;
/// Shared training/runtime configuration helpers.
pub mod config;
/// Constants used across Dragon training/runtime adapters.
pub mod constants;
/// Device and backend resolution helpers.
pub mod device;
/// WGPU-specific runtime/device helpers.
pub mod wgpu;

#[cfg(feature = "train")]
/// Training-loop integration and runtime instrumentation.
pub mod train;

pub mod api {
    //! Curated shared training/runtime API.

    pub mod config {
        pub use crate::config::{
            FsdpMixedPrecisionKind, GdpoConfig, GdpoHardGate, KernelSpec, LayerStateSpec,
            LowBitMemorySpec, LowBitModelSpec, LowBitSavedActivationInventorySpec,
            LowBitSavedActivationTensorSpec, ModelSpec, OptimizerSpec, ParallelCheckpointConfig,
            ParallelCheckpointFormat, ParallelCommunicationBackend, ParallelConfig,
            ParallelDataConfig, ParallelFsdpConfig, ParallelPipelineCacheConfig,
            ParallelPipelineConfig, ParallelSpec, ParallelTensorConfig, ParallelismKind,
            PipelineCacheEvictionKind, PipelineCachePolicy, PipelineCommunicationKind,
            PipelinePartitionKind, PipelineScheduleKind, PipelineSharedWeightSyncKind,
            PipelineTransportDtype, RunLayoutConfig, SequenceKernelConfig, StateAxisSpec,
            StateLayout, StateTensorSpec, TensorParallelAxis, TensorParallelPartitionKind,
            VisionTeacherVariant, WgpuBackend, WgpuGenerationExecutor, WgpuInferenceConfig,
            WgpuMemoryConfig, WgpuRuntimeConfig, WgpuStartupAutotuneConfig, WgpuTrainingConfig,
        };
        #[cfg(feature = "train")]
        pub use crate::config::{
            LearningRateScheduleConfig, MuonAdjustLrFn, MuonHybridConfig, OptimizerConfig,
            OptimizerKind, OptimizerScheduleMode, VisionArtifactOutputMode,
        };
    }

    pub mod runtime {
        #[cfg(all(feature = "train", feature = "ddp"))]
        pub use crate::train::runtime::resolve_collective_config;
        #[cfg(feature = "train")]
        pub use crate::train::runtime::{
            DeviceMemoryUsage, ParallelRuntime, PipelineParallelLayout, PipelineRankAssignment,
            bytes_to_mb, cleanup_device_memory, cleanup_device_memory_allowed, device_memory_usage,
            device_memory_usage_safe, resolve_parallel_runtime, resolve_pipeline_parallel_layout,
            resolve_training_devices,
        };
    }

    pub mod wgpu {
        pub use crate::wgpu::{
            WgpuDevice, WgpuFusedCoreOverride, apply_wgpu_fused_core_override, init_runtime,
            is_wgpu_backend_name,
        };
    }

    #[cfg(feature = "train")]
    pub mod expert {
        pub use crate::train;
    }
}

pub use config::{
    FsdpMixedPrecisionKind, GdpoConfig, GdpoHardGate, KernelSpec, LayerStateSpec, LowBitMemorySpec,
    LowBitModelSpec, LowBitSavedActivationInventorySpec, LowBitSavedActivationTensorSpec,
    ModelSpec, OptimizerSpec, ParallelCheckpointConfig, ParallelCheckpointFormat,
    ParallelCommunicationBackend, ParallelConfig, ParallelDataConfig, ParallelFsdpConfig,
    ParallelPipelineCacheConfig, ParallelPipelineConfig, ParallelSpec, ParallelTensorConfig,
    ParallelismKind, PipelineCacheEvictionKind, PipelineCachePolicy, PipelineCommunicationKind,
    PipelinePartitionKind, PipelineScheduleKind, PipelineSharedWeightSyncKind,
    PipelineTransportDtype, RunLayoutConfig, SequenceKernelConfig, StateAxisSpec, StateLayout,
    StateTensorSpec, TensorParallelAxis, TensorParallelPartitionKind, VisionTeacherVariant,
    WgpuBackend, WgpuGenerationExecutor, WgpuInferenceConfig, WgpuMemoryConfig, WgpuRuntimeConfig,
    WgpuStartupAutotuneConfig, WgpuTrainingConfig,
};
#[cfg(feature = "train")]
pub use config::{
    LearningRateScheduleConfig, MuonAdjustLrFn, MuonHybridConfig, OptimizerConfig, OptimizerKind,
    OptimizerScheduleMode, VisionArtifactOutputMode,
};
