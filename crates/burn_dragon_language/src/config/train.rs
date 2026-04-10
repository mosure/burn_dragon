use serde::{Deserialize, Serialize};

mod load;
mod schema;
#[cfg(test)]
mod tests;
mod validate;

pub use load::load_training_config;
pub use schema::{
    ContinualBackpropConfig, ContinualBackpropLrCoupling, ContinualBackpropTarget, DatasetConfig,
    DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat, ModuleLrScaleEntry,
    ModuleLrScaleScheduleConfig, TrainingConfig, TrainingHyperparameters, ValidationDatasetConfig,
};

use crate::tokenizer::TokenizerConfig;

use super::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
