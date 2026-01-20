#![recursion_limit = "256"]

pub mod config;
pub mod constants;
pub mod device;
pub mod wgpu;

#[cfg(feature = "train")]
pub mod train;

pub use config::{
    GdpoConfig, GdpoHardGate, WgpuBackend, WgpuMemoryConfig, WgpuRuntimeConfig,
};
#[cfg(feature = "train")]
pub use config::{
    ImagenetteVariant, LearningRateScheduleConfig, OptimizerConfig,
    VisionArtifactOutputMode, VisionAugmentationConfig, VisionDatasetConfig,
    VisionDatasetDownloadConfig, VisionDistillConfig, VisionFoveaSamplingMode,
    VisionFoveaScatterMode, VisionFoveaWarpMode, VisionLejepaConfig, VisionLejepaLossConfig,
    VisionLocationEmbeddingConfig, VisionLocationEmbeddingMode, VisionLossConfig, VisionMaeConfig,
    VisionMaeCrossViewConfig, VisionMaeLossConfig, VisionModelConfig, VisionNullGlimpseMode,
    VisionPyramidMode, VisionReconLossConfig, VisionSaccadeCacheConfig,
    VisionSaccadeCrossViewConfig, VisionSaccadeConfig, VisionSaccadeInfoRewardConfig,
    VisionSaccadeInputProjectionCnnConfig, VisionSaccadeInputProjectionConfig,
    VisionSaccadeInputProjectionMicroVitConfig, VisionSaccadePolicyConfig, VisionTbpttConfig,
    VisionTeacherConfig, VisionTeacherFeatureConfig, VisionTeacherModelConfig,
    VisionTeacherVariant, VisionTrainingConfig, VisionTrainingHyperparameters,
    VisionTrainingModeConfig, load_vision_training_config,
};
