#![recursion_limit = "256"]

//! Synthetic corpus generation focused on universality-style pre-pretraining.
//!
//! The initial implementation centers on Neural Cellular Automata-like rollouts
//! serialized into language-model token streams.

pub mod config;
pub mod generate;
pub mod manifest;
pub mod nca;
pub mod runtime;
pub mod stats;
pub mod tokenize;

pub mod api {
    //! Curated universality surface.

    pub mod config {
        pub use crate::config::{
            FloatRangeConfig, NcaComplexityBand, NcaComplexityMetric, NcaCorpusConfig,
            NcaFamilyConfig, NcaFamilyKind, NcaRuleFilterConfig, NcaSerializationConfig,
            NcaTokenizationConfig, UsizeRangeConfig, default_families,
            default_rule_filter_for_band, load_nca_config,
        };
    }

    pub mod manifest {
        pub use crate::manifest::{
            CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
            UniversalitySampleRecord, UniversalityTokenizerManifest, load_manifest,
        };
    }

    pub mod generate {
        pub use crate::generate::{GeneratedCorpusReport, generate_nca_corpus};
    }

    pub mod runtime {
        pub use crate::runtime::{
            OnlineNcaCorpus, RuntimeCorpusSummary, RuntimeSampleDocument,
            fixed_document_token_count,
        };
    }

    pub mod stats {
        pub use crate::stats::{
            ComplexityHistogramBin, CorpusStats, SampleStats, build_complexity_histogram,
        };
    }

    pub mod expert {
        pub use crate::{config, generate, manifest, nca, stats, tokenize};
    }
}

pub use config::{
    FloatRangeConfig, NcaComplexityBand, NcaComplexityMetric, NcaCorpusConfig, NcaFamilyConfig,
    NcaFamilyKind, NcaRuleFilterConfig, NcaSerializationConfig, NcaTokenizationConfig,
    UsizeRangeConfig, default_families, default_rule_filter_for_band, load_nca_config,
};
pub use generate::{GeneratedCorpusReport, generate_nca_corpus};
pub use manifest::{
    CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
    UniversalitySampleRecord, UniversalityTokenizerManifest, load_manifest,
};
pub use runtime::{
    OnlineNcaCorpus, RuntimeCorpusSummary, RuntimeSampleDocument, fixed_document_token_count,
};
pub use stats::{ComplexityHistogramBin, CorpusStats, SampleStats, build_complexity_histogram};
