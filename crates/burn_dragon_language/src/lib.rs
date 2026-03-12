#![recursion_limit = "256"]

//! Language training and inference adapters over the shared Dragon BDH core.
//!
//! Paper mapping:
//! - `burn_dragon_core::BDH` owns the paper-faithful `x_neuron`, `y_gate`, `y_neuron`, and
//!   per-layer recurrent `rho` contract
//! - this crate layers tokenization, datasets, generation, and training schedules on top of that
//!   core without redefining the recurrent state semantics

pub mod config;
pub mod generation;
pub mod inference;
pub mod loss;
pub mod tokenizer;

#[cfg(feature = "train")]
pub mod checkpoint;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod train;

pub mod api {
    //! Curated language-facing Dragon API.

    pub mod core {
        pub use burn_dragon_core::api::config::{BDHConfig, YNeuronRecurrenceConfig};
        pub use burn_dragon_core::api::state::ModelState;
        pub use burn_dragon_core::api::recurrent::BDH;
    }

    pub mod config {
        pub use crate::config::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
        #[cfg(feature = "train")]
        pub use crate::config::{
            DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig,
            HuggingFaceRecordFormat, TrainingConfig, TrainingHyperparameters,
            load_training_config,
        };
    }

    pub mod inference {
        pub use crate::generation::{
            ContextStrategy, GenerationProfileSnapshot, GenerationSettings, generate_text,
            generate_tokens, generate_tokens_chunked, generation_profile_reset,
            generation_profile_snapshot, prefill_state, resolve_context_strategy,
            sample_next_token,
        };
        pub use crate::inference::{
            apply_wgpu_fused_core_override, build_model_config, is_wgpu_backend_name,
        };
        pub use crate::loss::language_model_loss;
        pub use crate::tokenizer::char_vocab::CharVocab;
    }

    #[cfg(feature = "train")]
    pub mod checkpoint {
        pub use crate::checkpoint::{
            LanguageBurnpackExportReport, LanguageRunConfigSnapshot,
            default_checkpoint_dir, export_language_checkpoint_to_burnpack,
            load_training_config_for_checkpoint, write_training_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod train {
        pub use crate::dataset;
        pub use crate::train;
    }
}

pub use burn_dragon_core::{BDH, BDHConfig, ModelState};
pub use config::{ContextStrategyConfig, GenerationConfig, ModelOverrides};
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingConfig, TrainingHyperparameters, load_training_config,
};
#[cfg(feature = "train")]
pub use checkpoint::{
    LanguageBurnpackExportReport, LanguageRunConfigSnapshot, default_checkpoint_dir,
    export_language_checkpoint_to_burnpack, load_training_config_for_checkpoint,
    write_training_snapshot,
};
pub use generation::{
    ContextStrategy, GenerationProfileSnapshot, GenerationSettings, generate_text, generate_tokens,
    generate_tokens_chunked, generation_profile_reset, generation_profile_snapshot, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use inference::{apply_wgpu_fused_core_override, build_model_config, is_wgpu_backend_name};
pub use loss::language_model_loss;
pub use tokenizer::char_vocab::CharVocab;
