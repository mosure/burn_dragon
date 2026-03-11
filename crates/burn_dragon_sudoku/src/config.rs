use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};

use burn_dragon_core::{ManifoldHyperConnectionsConfig, RotaryEmbedding};
use burn_dragon_train::{GdpoConfig, OptimizerConfig, WgpuRuntimeConfig};

mod dataset;
mod defaults;
mod load;
mod model;
#[cfg(test)]
mod tests;
mod training;
mod validate;

pub use dataset::{
    SudokuDatasetConfig, SudokuDatasetSourceConfig, SudokuHuggingFaceConfig, SudokuLocalConfig,
    SudokuRecordFormat,
};
pub use load::load_training_config;
pub use model::{
    SudokuArtifactConfig, SudokuCacheMhcConfig, SudokuCacheUpdateConfig, SudokuCacheUpdateMode,
    SudokuGridPositional, SudokuModelConfig,
};
pub use training::{
    SudokuEasyRewardMode, SudokuHaltConfig, SudokuHardRewardMode, SudokuInfoRewardConfig,
    SudokuLossMask, SudokuPolicyConfig, SudokuPolicyHead, SudokuReconConfig, SudokuReconLoss,
    SudokuRevisitConfig, SudokuRewardBaselineConfig, SudokuRewardConfig,
    SudokuRewardShapingConfig, SudokuRewardShapingMetric, SudokuRolloutConfig,
    SudokuRolloutSchedule, SudokuTrainingHyperparameters, SudokuTraversal, SudokuTrmMode,
    SudokuValidationConfig, SudokuWriteGateMode,
};

pub(crate) use defaults::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuTrainingConfig {
    pub dataset: SudokuDatasetConfig,
    pub training: SudokuTrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub artifacts: SudokuArtifactConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    #[serde(default)]
    pub model: SudokuModelConfig,
}
