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
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingConfig, TrainingHyperparameters, load_training_config,
};
pub use generation::{
    ContextStrategy, GenerationProfileSnapshot, GenerationSettings, generate_text, generate_tokens,
    generation_profile_reset, generation_profile_snapshot, prefill_state, resolve_context_strategy,
    sample_next_token,
};
pub use inference::{apply_wgpu_fused_core_override, build_model_config, is_wgpu_backend_name};
pub use loss::language_model_loss;
pub use tokenizer::char_vocab::CharVocab;
