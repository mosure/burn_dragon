#![recursion_limit = "256"]

pub mod config;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod foveation;
pub mod generation;
pub mod inference;
pub mod kernel;
pub mod model;
pub mod positional;
pub mod tokenizer;
#[cfg(feature = "train")]
pub mod train;
#[cfg(feature = "viz")]
pub mod viz;
#[cfg(feature = "cli")]
pub mod wgpu;
#[cfg(all(target_arch = "wasm32", feature = "web"))]
pub mod web;

pub use config::{ContextStrategyConfig, GenerationConfig, ModelOverrides, TrainingHyperparameters};
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    ImagenetteVariant, LearningRateScheduleConfig, OptimizerConfig, TrainingConfig,
    VisionArtifactOutputMode, VisionAugmentationConfig, VisionDatasetConfig,
    VisionDatasetDownloadConfig, VisionDistillConfig, VisionLejepaConfig, VisionMaeConfig,
    VisionModelConfig, VisionPyramidMode, VisionSaccadeConfig, VisionTeacherConfig,
    VisionTeacherFeatureConfig, VisionTeacherModelConfig, VisionTeacherVariant,
    VisionTrainingConfig, VisionTrainingHyperparameters, VisionTrainingModeConfig,
    load_training_config, load_vision_training_config,
};
#[cfg(feature = "train")]
pub use dataset::{
    Dataset, DatasetSplit, HuggingFaceDataset, RandomDataLoader, SequenceBatch, ShakespeareBatch,
    ShakespeareDataset, ShakespeareRandomDataLoader, ShakespeareSplit, TokenSequenceDataset,
    build_dataset,
};
pub use generation::{
    ContextStrategy, GenerationSettings, generate_text, generate_tokens, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use inference::build_model_config;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{
    BDH, BDHConfig, FusedKernelConfig, ModelState, PatchEmbed, PatchEmbedOutput, PatchGrid,
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionDragonHatchling,
    VisionDragonHatchlingConfig, VisionDragonHatchlingOutput, VisionDistillationLossConfig,
    language_model_loss, patchify, pool_patch_tokens, unpatchify, vision_distillation_loss,
};
#[cfg(feature = "train")]
pub use model::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};
pub use positional::RotaryEmbedding;
pub use tokenizer::char_vocab::CharVocab;
