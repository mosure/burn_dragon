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
    ImagenetteVariant, VisionArtifactOutputMode, VisionAugmentationConfig, VisionDatasetConfig,
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

#[cfg(all(test, feature = "train"))]
mod tests;
