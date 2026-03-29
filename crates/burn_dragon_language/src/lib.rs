#![recursion_limit = "256"]

//! Language training and inference adapters over the shared Dragon BDH core.
//!
//! Paper mapping:
//! - `burn_dragon_core::BDH` owns the paper-faithful `x_neuron`, `y_gate`, `y_neuron`, and
//!   per-layer recurrent `rho` contract
//! - this crate layers tokenization, datasets, generation, and training schedules on top of that
//!   core without redefining the recurrent state semantics

pub mod bitnet_artifact;
pub mod config;
pub mod generation;
pub mod inference;
pub mod loss;
pub mod summary_events;
pub mod tokenizer;

#[cfg(feature = "train")]
pub mod checkpoint;
#[cfg(feature = "train")]
pub mod dataset;
#[cfg(feature = "train")]
pub mod stages;
#[cfg(feature = "train")]
pub mod train;

pub mod api {
    //! Curated language-facing Dragon API.

    pub mod core {
        pub use burn_dragon_core::api::config::{
            BDHConfig, SequenceKernelConfig, SequenceMemorySystem, SequenceTrainingExecutor,
            YNeuronRecurrenceConfig,
        };
        pub use burn_dragon_core::api::recurrent::BDH;
        pub use burn_dragon_core::api::state::ModelState;
    }

    pub mod config {
        pub use crate::config::{
            ContextStrategyConfig, GenerationConfig, GenerationOutputFormat,
            GenerationTokenizerSourceConfig, ModelOverrides,
        };
        #[cfg(feature = "train")]
        pub use crate::config::{
            DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
            TrainingConfig, TrainingHyperparameters, load_training_config,
        };
    }

    pub mod inference {
        pub use crate::bitnet_artifact::{
            BITNET_ARTIFACT_BINARY_MAGIC, LanguageBitNetArtifactBundle,
            deserialize_bitnet_artifact_binary, serialize_bitnet_artifact_binary,
        };
        pub use crate::generation::{
            ContextStrategy, GenerationProfileSnapshot, GenerationSettings, generate_text,
            generate_tokens, generate_tokens_chunked, generation_profile_reset,
            generation_profile_snapshot, prefill_state, resolve_context_strategy,
            sample_next_token,
        };
        pub use crate::inference::{
            WgpuFusedCoreOverride, apply_wgpu_fused_core_override, build_model_config,
            build_model_config_with_tokenizer, is_wgpu_backend_name,
        };
        pub use crate::loss::language_model_loss;
        pub use crate::tokenizer::char_vocab::CharVocab;
    }

    #[cfg(feature = "train")]
    pub mod checkpoint {
        pub use crate::bitnet_artifact::LanguageBitNetArtifactBundle;
        pub use crate::checkpoint::{
            LanguageBitNetArtifactExportReport, LanguageBurnpackExportReport,
            LanguageRunConfigSnapshot, apply_bitnet_artifact_bundle_to_model,
            candidate_bitnet_artifact_paths, default_bitnet_artifact_path, default_checkpoint_dir,
            export_language_checkpoint_to_bitnet_artifact, export_language_checkpoint_to_burnpack,
            load_bitnet_artifact_bundle, load_language_core_from_checkpoint,
            load_language_core_from_checkpoint_with_bitnet_artifact, load_tokenizer_for_checkpoint,
            load_training_config_for_checkpoint, write_training_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod train {
        pub use crate::dataset;
        pub use crate::stages;
        pub use crate::train;
    }
}

pub use bitnet_artifact::{
    BITNET_ARTIFACT_BINARY_MAGIC, LanguageBitNetArtifactBundle, deserialize_bitnet_artifact_binary,
    serialize_bitnet_artifact_binary,
};
pub use burn_dragon_core::{
    BDH, BDHConfig, ModelState, SequenceKernelConfig, SequenceMemorySystem,
    SequenceTrainingExecutor,
};
#[cfg(feature = "train")]
pub use checkpoint::{
    LanguageBitNetArtifactExportReport, LanguageBurnpackExportReport, LanguageRunConfigSnapshot,
    apply_bitnet_artifact_bundle_to_model, candidate_bitnet_artifact_paths,
    default_bitnet_artifact_path, default_checkpoint_dir,
    export_language_checkpoint_to_bitnet_artifact, export_language_checkpoint_to_burnpack,
    load_bitnet_artifact_bundle, load_language_core_from_checkpoint,
    load_language_core_from_checkpoint_with_bitnet_artifact, load_tokenizer_for_checkpoint,
    load_training_config_for_checkpoint, write_training_snapshot,
};
pub use config::{
    ContextStrategyConfig, GenerationConfig, GenerationOutputFormat,
    GenerationTokenizerSourceConfig, ModelOverrides,
};
#[cfg(feature = "train")]
pub use config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingConfig, TrainingHyperparameters, load_training_config,
};
pub use generation::{
    ContextStrategy, GenerationProfileSnapshot, GenerationSettings, generate_text, generate_tokens,
    generate_tokens_chunked, generation_profile_reset, generation_profile_snapshot, prefill_state,
    resolve_context_strategy, sample_next_token,
};
pub use inference::{
    WgpuFusedCoreOverride, apply_wgpu_fused_core_override, build_model_config,
    build_model_config_with_tokenizer, is_wgpu_backend_name,
};
pub use loss::language_model_loss;
#[cfg(feature = "train")]
pub use stages::{
    BUNDLE_STATE_FILE_NAME, ExperimentBackend, ExperimentBundleConfig, ExperimentBundleState,
    ExperimentStageArtifact, ExperimentStageConfig, ExperimentStageKind, ExperimentStageState,
    ExperimentStageStatus, RESOLVED_CONFIG_FILE_NAME, STAGE_STATE_FILE_NAME, build_bundle_state,
    bundle_state_path, load_experiment_bundle_config, load_stage_state,
    prepare_language_stage_config, prepare_universality_stage_config, resolve_bundle_root,
    resolve_stage_dependency_artifacts, resolve_stage_dir, resolve_training_stage_artifact,
    resolved_stage_config_path, stage_state_path, unix_timestamp_now, write_bundle_state,
    write_resolved_config, write_stage_state,
};
pub use summary_events::{
    resolve_summary_memory_write_triggers, summary_event_mask_from_flat_batch,
    summary_event_mask_from_tokens, summary_event_mask_tensor,
};
pub use tokenizer::char_vocab::CharVocab;
