#![recursion_limit = "256"]

pub mod config;
pub mod generation;
pub mod inference;
pub mod loss;
pub mod tokenizer;

#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod train;

pub use config::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
pub use loss::language_model_loss;
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingConfig, TrainingHyperparameters, load_training_config,
};
pub use generation::{
    ContextStrategy, GenerationSettings, generate_text, generate_tokens, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use inference::build_model_config;
pub use tokenizer::char_vocab::CharVocab;
