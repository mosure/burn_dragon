#![recursion_limit = "256"]

//! Shared Dragon/Hatchling core math and state contracts.
//!
//! Preferred library-facing surface:
//! - [`api::config`]
//! - [`api::state`]
//! - [`api::recurrent`]
//! - [`api::mhc`]
//! - [`api::experimental`] for research-only connector and sequence-kernel surfaces
//! - [`api::expert`] for lower-level implementation details

pub mod constants;
pub mod experimental;
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
            AttentionResidualConfig, BDHConfig, BdhFiringTargetConfig, BdhFiringTargetKind,
            BdhInitializationConfig, BdhInitializationKind, BdhNeuronGainConfig, BdhNeuronGainKind,
            BdhResidualScalingConfig, BdhResidualScalingKind, BdhTopologyPriorConfig,
            BdhTopologyPriorKind, BitNetLowBitProtocol, BlockAttentionResidualConfig,
            BlockAttentionResidualSummaryMode, ClockedSlowMemoryConfig, DragonNormConfig,
            DragonNormKind, FusedAttentionExecutor, FusedKernelConfig, FusedProjectionExecutor,
            LatentFanoutScheduleConfig, LowBitActivationFormat, LowBitActivationGrouping,
            LowBitInferenceMode, LowBitMemoryBucketEstimate, LowBitMemoryEstimateInput,
            LowBitNativeProjectionProfileSnapshot, LowBitQuantizationConfig, LowBitRhoConfig,
            LowBitSavedActivationConfig, LowBitSavedActivationInventory, LowBitSavedActivationMode,
            LowBitSavedActivationRecomputePolicy, LowBitSavedActivationTensorInventoryEntry,
            LowBitTargetModule, LowBitTrainingMode, LowBitWeightFormat, LowBitWeightGrouping,
            MambaSequenceConfig, ManifoldHyperConnectionCoefficientPolicy,
            ManifoldHyperConnectionsConfig, ResidualConnectorKind, RhoCompressionConfig,
            RhoCompressionInterval, RhoPrecisionConfig, SequenceKernelConfig, SequenceKernelFamily,
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
        #[cfg(any(feature = "probe", test))]
        pub use crate::LanguageBdhInitLayerDiagnostics;
        pub use crate::{
            BDH, DragonNorm, HaltHead, LanguageMhcLayerDiagnostics, LanguagePipelineState,
            LogitsProjectionProfileSnapshot, LowBitTrainingProjectionMemoryProfileSnapshot,
            LowBitTrainingProjectionMemoryStageSnapshot, LowBitTrainingQuantizeProfileSnapshot,
            LowRankResidualMemoryProfileSnapshot, LowRankResidualMemoryStageSnapshot,
            LowRankResidualOutput, LowRankResidualProfileSnapshot, StructuredDenseUpdateOutput,
            logits_projection_profile_reset, logits_projection_profile_snapshot,
            low_bit_training_lowrank_memory_profile_snapshot,
            low_bit_training_quantize_profile_snapshot, lowrank_residual_memory_profile_reset,
            lowrank_residual_memory_profile_snapshot, lowrank_residual_profile_reset,
            lowrank_residual_profile_snapshot, lowrank_residual_step, lowrank_residual_step_next,
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

    pub mod experimental {
        //! Experimental BDH surfaces that are still expected to evolve quickly.
        //!
        //! These are available for research and ablation work, but they are intentionally grouped
        //! away from the smaller stable-ish `config` / `state` / `recurrent` surface.

        pub mod connectors {
            pub use crate::{
                AttentionResidual, AttentionResidualConfig, BlockAttentionResidual,
                BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode,
                ResidualConnectorKind,
            };
        }

        pub mod sequence {
            pub use crate::{
                MambaSequenceConfig, SequenceKernelConfig, SequenceKernelFamily,
                SequenceKernelKind, SequenceTrainingExecutor,
            };
        }

        pub mod bitnet_reference {
            pub use crate::experimental::bitnet_reference::{
                BdhBitNetStaticArtifacts, BitLinearReferenceSpec, BitNetReferenceActivationMode,
                BitNetReferenceWeightMode, PackedTernaryBuffer, PackedWeightArtifact,
                PackedWeightEncoding, QuantizedBuffer, bitlinear_reference_forward,
                dequantize_activation_i8, dequantize_weight_codes, pack_binary_1bit,
                pack_ternary_2bit, pack_weight_artifact_from_format,
                quantize_activation_symmetric_i8, quantize_binary_sign, quantize_ternary_absmean,
                ste_passthrough, unpack_binary_1bit, unpack_ternary_2bit,
                unpack_weight_artifact_to_f32,
            };
        }
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
#[cfg(any(feature = "probe", test))]
pub use model::LanguageBdhInitLayerDiagnostics;
#[cfg(any(feature = "viz", feature = "probe"))]
pub use model::LayerVizState;
pub use model::{
    AttentionResidual, AttentionResidualConfig, BDH, BDHConfig, BankedRhoState,
    BdhActivationThresholds, BdhBitNetDeployScaffold, BdhFiringTargetConfig, BdhFiringTargetKind,
    BdhInitializationConfig, BdhInitializationKind, BdhInitializer, BdhNeuronGainConfig,
    BdhNeuronGainKind, BdhProjectionRole, BdhResidualScalingConfig, BdhResidualScalingKind,
    BdhTopologyPriorConfig, BdhTopologyPriorKind, BitNetLowBitProtocol, BlockAttentionResidual,
    BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode, ClockedSlowMemoryConfig,
    DragonNorm, DragonNormConfig, DragonNormKind, FusedAttentionExecutor, FusedKernelConfig,
    FusedProjectionExecutor, HaltHead, LanguageMhcLayerDiagnostics, LanguagePipelineState,
    LatentFanoutScheduleConfig, LogitsProjectionProfileSnapshot, LowBitActivationFormat,
    LowBitActivationGrouping, LowBitInferenceMode, LowBitKernelCapabilities,
    LowBitKernelFallbackReason, LowBitKernelPlan, LowBitKernelRuntimeKind,
    LowBitMemoryBucketEstimate, LowBitMemoryEstimateInput, LowBitNativeProjectionProfileSnapshot,
    LowBitProjectionPlan, LowBitQuantizationConfig, LowBitRhoConfig, LowBitSavedActivationCache,
    LowBitSavedActivationConfig, LowBitSavedActivationInventory, LowBitSavedActivationMode,
    LowBitSavedActivationRecomputePolicy, LowBitSavedActivationTensorInventoryEntry,
    LowBitTargetModule, LowBitTrainingMode, LowBitTrainingProjectionMemoryProfileSnapshot,
    LowBitTrainingProjectionMemoryStageSnapshot, LowBitTrainingQuantizeProfileSnapshot,
    LowBitWeightFormat, LowBitWeightGrouping, LowRankResidualMemoryProfileSnapshot,
    LowRankResidualMemoryStageSnapshot, LowRankResidualOutput, LowRankResidualProfileSnapshot,
    MambaSequenceConfig, ManifoldHyperConnectionCoefficientPolicy,
    ManifoldHyperConnectionCoefficients, ManifoldHyperConnectionStreamCoefficients,
    ManifoldHyperConnectionStreamOutput, ManifoldHyperConnectionWidthOutput,
    ManifoldHyperConnections, ManifoldHyperConnectionsConfig, MicroTransformerBlock, ModelState,
    PackedLowBitProjectionArtifacts, PackedRhoBlockEncoding, PackedRhoBlockState,
    PackedRhoInt8DeviceState, PackedSavedActivationBuffer, PackedSavedActivationState,
    ResidualConnectorKind, RhoCompressionConfig, RhoCompressionInterval, RhoCompressionQualityGate,
    RhoCompressionStatsSnapshot, RhoPrecisionConfig, SequenceKernelConfig, SequenceKernelFamily,
    SequenceKernelKind, SequenceTrainingExecutor, StructuredBankRole, StructuredDenseUpdateOutput,
    StructuredGridState, StructuredRouteOperation, StructuredRoutePattern, StructuredRouteSpec,
    StructuredRoutingSpec, StructuredStepMode, StructuredTopologyState, SummaryMemoryConfig,
    YNeuronRecurrenceConfig, build_low_bit_saved_activation_inventory,
    estimate_low_bit_memory_buckets, logits_projection_profile_reset,
    logits_projection_profile_snapshot, low_bit_kernel_capabilities,
    low_bit_kernel_capabilities_for_backend_name, low_bit_native_decoder_tail_profile_snapshot,
    low_bit_native_lowrank_profile_snapshot, low_bit_native_projection_profile_reset,
    low_bit_training_lowrank_memory_profile_snapshot, low_bit_training_quantize_profile_snapshot,
    lowrank_residual_memory_profile_reset, lowrank_residual_memory_profile_snapshot,
    lowrank_residual_profile_reset, lowrank_residual_profile_snapshot, lowrank_residual_step,
    lowrank_residual_step_next, mhc_merge, mhc_merge_with_coefficients, mhc_passthrough,
    mhc_passthrough_with_coefficients, mhc_split, mhc_split_with_coefficients,
    near_critical_embedding_initializer, near_critical_embedding_std, near_critical_projection_std,
    near_critical_residual_output_std, pack_saved_activation_state, resolve_low_bit_kernel_plan,
    resolve_low_bit_kernel_plan_for_backend_name, rho_compression_profile_reset,
    rho_compression_profile_snapshot, rho_compression_snapshot_passes_gate,
    structured_dense_update_tokens, structured_predict_decay, target_major_apply_decay,
    target_major_decay_add, target_major_identity_read, target_major_identity_write,
    target_major_outer_product, unpack_saved_activation_state,
};
pub use positional::RotaryEmbedding;
