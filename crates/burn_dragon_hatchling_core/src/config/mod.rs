pub mod core;
#[cfg(feature = "train")]
pub mod train;
#[cfg(feature = "train")]
pub mod vision;

pub use core::{
    ContextStrategyConfig, GdpoConfig, GdpoHardGate, GenerationConfig, ModelOverrides,
    TrainingHyperparameters, WgpuBackend, WgpuMemoryConfig, WgpuRuntimeConfig,
};
#[cfg(feature = "train")]
pub use train::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    LearningRateScheduleConfig, OptimizerConfig, TrainingConfig, load_training_config,
};
#[cfg(feature = "train")]
pub use vision::{
    ImagenetteVariant, VisionAugmentationConfig, VisionDatasetConfig, VisionDatasetDownloadConfig,
    VisionArtifactOutputMode, VisionDistillConfig, VisionFoveaSamplingMode, VisionFoveaScatterMode,
    VisionFoveaWarpMode, VisionLejepaConfig, VisionLejepaLossConfig,
    VisionLocationEmbeddingConfig, VisionLocationEmbeddingMode, VisionLossConfig, VisionMaeConfig,
    VisionMaeLossConfig, VisionModelConfig, VisionNullGlimpseMode,
    VisionPyramidMode, VisionReconLossConfig,
    VisionSaccadeCacheConfig, VisionSaccadeConfig,
    VisionSaccadeInputProjectionConfig, VisionSaccadeInputProjectionCnnConfig,
    VisionSaccadeInputProjectionMicroVitConfig, VisionTbpttConfig,
    VisionSaccadeInfoRewardConfig, VisionSaccadePolicyConfig, VisionTeacherConfig,
    VisionTeacherFeatureConfig, VisionTeacherModelConfig, VisionTeacherVariant, VisionTrainingConfig,
    VisionTrainingHyperparameters, VisionTrainingModeConfig, load_vision_training_config,
};

#[cfg(all(test, feature = "train"))]
mod tests;
