#![recursion_limit = "256"]

//! Shared Dragon/Hatchling core math and state contracts.
//!
//! Preferred library-facing surface:
//! - [`api::config`]
//! - [`api::state`]
//! - [`api::recurrent`]
//! - [`api::mhc`]
//! - [`api::expert`] for lower-level implementation details

pub mod constants;
pub mod kernel;
pub mod model;
pub mod positional;

pub mod api {
    //! Curated public surface for the Dragon core.
    //!
    //! These modules group the primary stable-ish concepts rather than exposing the full internal
    //! implementation layout.

    pub mod config {
        pub use crate::{
            BDHConfig, ClockedSlowMemoryConfig, DragonNormConfig, DragonNormKind,
            FusedAttentionExecutor, FusedKernelConfig, FusedProjectionExecutor,
            LatentFanoutScheduleConfig, MambaSequenceConfig,
            ManifoldHyperConnectionCoefficientPolicy,
            ManifoldHyperConnectionsConfig, SequenceKernelConfig, SequenceKernelFamily,
            SequenceKernelKind, SequenceTrainingExecutor, SummaryMemoryConfig,
            YNeuronRecurrenceConfig,
        };
        pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
    }

    pub mod state {
        pub use crate::{
            BankedRhoState, ModelState, StructuredGridState, StructuredRoutingSpec,
            StructuredStepMode, StructuredTopologyState,
        };
    }

    pub mod recurrent {
        pub use crate::{
            BDH, DragonNorm, HaltHead, LanguageMhcLayerDiagnostics,
            LogitsProjectionProfileSnapshot, LowRankResidualOutput, LowRankResidualProfileSnapshot,
            StructuredDenseUpdateOutput, logits_projection_profile_reset,
            logits_projection_profile_snapshot, lowrank_residual_profile_reset,
            lowrank_residual_profile_snapshot, lowrank_residual_step,
            structured_dense_update_tokens,
        };
    }

    pub mod mhc {
        pub use crate::{
            ManifoldHyperConnectionCoefficients, ManifoldHyperConnectionStreamCoefficients,
            ManifoldHyperConnectionStreamOutput, ManifoldHyperConnectionWidthOutput,
            ManifoldHyperConnections, mhc_merge, mhc_merge_with_coefficients, mhc_passthrough,
            mhc_passthrough_with_coefficients, mhc_split, mhc_split_with_coefficients,
        };
    }

    pub mod expert {
        //! Lower-level implementation surface for advanced callers.
        //!
        //! Prefer the smaller `config` / `state` / `recurrent` / `mhc` modules unless you
        //! explicitly need internal kernel/layout utilities.

        pub use crate::constants;
        pub use crate::kernel;
        pub use crate::model;
        pub use crate::positional;
    }
}

pub use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
#[cfg(feature = "viz")]
pub use model::LayerVizState;
pub use model::{
    BDH, BDHConfig, BankedRhoState, ClockedSlowMemoryConfig, DragonNorm, DragonNormConfig,
    DragonNormKind, FusedAttentionExecutor, FusedKernelConfig, FusedProjectionExecutor, HaltHead,
    LanguageMhcLayerDiagnostics, LatentFanoutScheduleConfig, LogitsProjectionProfileSnapshot,
    MambaSequenceConfig,
    LowRankResidualOutput, LowRankResidualProfileSnapshot,
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnectionStreamOutput,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
    ModelState, SequenceKernelConfig, SequenceKernelFamily, SequenceKernelKind,
    SequenceTrainingExecutor, StructuredBankRole, StructuredDenseUpdateOutput, StructuredGridState,
    StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec, StructuredRoutingSpec,
    StructuredStepMode, StructuredTopologyState, SummaryMemoryConfig, YNeuronRecurrenceConfig,
    logits_projection_profile_reset, logits_projection_profile_snapshot,
    lowrank_residual_profile_reset, lowrank_residual_profile_snapshot, lowrank_residual_step,
    mhc_merge, mhc_merge_with_coefficients, mhc_passthrough, mhc_passthrough_with_coefficients,
    mhc_split, mhc_split_with_coefficients, near_critical_embedding_initializer,
    near_critical_embedding_std, near_critical_projection_std, near_critical_residual_output_std,
    structured_dense_update_tokens, structured_predict_decay, target_major_apply_decay,
    target_major_decay_add, target_major_identity_read, target_major_identity_write,
    target_major_outer_product,
};
pub use positional::RotaryEmbedding;
