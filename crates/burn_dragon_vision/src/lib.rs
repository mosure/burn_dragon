#![recursion_limit = "256"]

//! Vision backbones and training adapters over the shared Dragon terminology/state contract.
//!
//! Paper mapping:
//! - `Dense` is dense-space latent refinement without persistent `rho`
//! - `Cellular` exposes token-local persistent `rho`
//! - `Pyramid` exposes structured primary/context/global recurrent banks built on shared core
//!   `BankedRhoState`
//!
//! Current image-side recommendation from the broader validation sweeps:
//! - start graph-backed work from
//!   `VisionDragonConfig::scene_slot_graph_bridge_baseline_224()` or
//!   `VisionTrainingConfig::scene_slot_graph_bridge_imagenette_baseline()`
//! - use `VisionDragonConfig::scene_slot_graph_baseline_224()` or the checked-in
//!   `config/vision/trm/baselines/graph_scene_slots_imagenette.toml` file as the matched control
//! - use `config/vision/trm/baselines/graph_bridge_imagenette.toml` as the promoted training
//!   baseline
//! - for ImageNet-1k distill, treat
//!   `VisionTrainingConfig::scene_slot_graph_bridge_multimode_spatial_imagenet1k_medium_launch()`
//!   as the promoted multi-mode recipe when full SigLIP2 spatial features are available
//! - use
//!   `VisionTrainingConfig::scene_slot_graph_bridge_multiteacher_imagenet1k_medium_launch()`
//!   as the lower-storage fallback when only SigLIP2 global features are available

#[cfg(feature = "train")]
pub mod checkpoint;
#[cfg(feature = "train")]
pub mod config;
pub mod constants;
mod device;
pub mod foveation;
#[cfg(feature = "train")]
pub mod loss;
pub mod model;
#[cfg(feature = "train")]
pub mod train;
pub mod wgsl;

pub mod api {
    //! Curated vision-facing Dragon API.

    pub mod core {
        pub use burn_dragon_core::api::state::{
            BankedRhoState, StructuredGridState, StructuredRoutingSpec, StructuredStepMode,
            StructuredTopologyState,
        };
        pub use burn_dragon_core::{
            StructuredBankRole, StructuredRouteOperation, StructuredRoutePattern,
            StructuredRouteSpec,
        };
    }

    pub mod model {
        pub use crate::model::{
            PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind,
            StageAwareHostProfileSnapshot, VisionAttentionMode, VisionBackboneKind,
            VisionCellularConfig, VisionCellularState, VisionDragon, VisionDragonConfig,
            VisionDragonMultiOutput, VisionDragonOutput, VisionLatentActivation,
            VisionPatchEmbedMode, VisionPyramidConfig, VisionRhoStreamConfig, VisionRolloutState,
            VisionTrmGraphConfig, VisionTrmGridMismatchPolicy, VisionTrmPredictSubstepKind,
            patchify, pool_patch_tokens, stage_aware_host_profile_reset,
            stage_aware_host_profile_snapshot, unpatchify,
        };
    }

    #[cfg(feature = "benchmark")]
    pub mod bench {
        pub use crate::model::{
            VisionDenseAttentionBenchAdapter, VisionDenseBenchAdapter,
            VisionRolloutScheduleBenchAdapter,
        };
        pub use crate::train::{
            VISION_ARTIFACT_SCHEMA_VERSION, VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION,
            VISION_DISTILL_LINEAR_PROBE_HARNESS_VERSION, VisionArtifactHeader,
            VisionDistillDeploySmokePrecision, VisionDistillDeploySmokeReport,
            VisionDistillFeatureExportReport, VisionDistillFeatureProbeAccuracyReport,
            VisionDistillFeatureProbeBackend, VisionDistillFeatureProbeDevice,
            VisionDistillFeatureProbeReport, VisionDistillFeatureProbeStepAccuracy,
            VisionDistillLinearProbeAccuracyReport, VisionDistillLinearProbeReport,
            VisionDistillLinearProbeStepAccuracy, VisionDistillServingBenchmarkBackend,
            VisionDistillServingBenchmarkDevice, VisionDistillServingBenchmarkReport,
            VisionDistillServingStepMetrics, export_vision_distill_feature_embeddings,
            push_vision_artifact_markdown_prelude, run_vision_distill_deploy_smoke,
            run_vision_distill_feature_probe, run_vision_distill_feature_probe_with_seed,
            run_vision_distill_linear_probe, run_vision_distill_linear_probe_with_seed,
            run_vision_distill_serving_benchmark,
        };
    }

    pub mod foveation {
        pub use crate::foveation::{
            CpuImageLevel, CpuPyramidCache, FoveaWarpMode, PyramidMode, build_pyramid_cache,
            image_from_nchw, lod_sigma_from_sigma, render_foveated_patch,
            render_foveated_patch_with_radius, sigma_from_unit,
        };
    }

    #[cfg(feature = "train")]
    pub mod checkpoint {
        pub use crate::checkpoint::{
            VisionBurnpackExportReport, export_vision_encoder_checkpoint_to_burnpack,
            load_training_config_for_checkpoint, load_vision_encoder_from_checkpoint,
            write_training_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod config {
        pub use crate::config::*;
    }

    #[cfg(feature = "train")]
    pub mod train {
        pub use crate::loss::{
            VisionDistillationLossConfig, vision_distillation_loss, vision_distillation_loss_terms,
        };
        pub use crate::train::{
            CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
            ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
            ImageNetDatasetConfig, ImageNetSplit, ImageNetTeacherTargetBatch,
            ImageTeacherTargetStore, MovingMnistRenderedClip, MovingMnistSplit,
            MovingMnistVideoDataLoader, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
            VideoClipBatch, VisionNormalize, VisionVideoTrainProfileSnapshot,
            video_train_profile_reset, video_train_profile_snapshot,
        };
    }

    pub mod shaders {
        pub use crate::wgsl::{
            FOVEATION_BUFFER_SHADER, FOVEATION_SHADER, PYRAMID_SHADER, SCATTER_BUFFER_SHADER,
        };
    }
}

pub use burn_dragon_core::{
    BankedRhoState, StructuredBankRole, StructuredGridState, StructuredRouteOperation,
    StructuredRoutePattern, StructuredRouteSpec, StructuredRoutingSpec, StructuredStepMode,
    StructuredTopologyState,
};
pub use constants::{FOVEA_AA_THRESHOLD, SACCADE_FOVEA_SUBSAMPLES};
pub use foveation::{
    CpuImageLevel, CpuPyramidCache, FoveaWarpMode, PyramidMode, build_pyramid_cache,
    image_from_nchw, lod_sigma_from_sigma, render_foveated_patch,
    render_foveated_patch_with_radius, sigma_from_unit,
};
pub use model::{
    PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind,
    StageAwareHostProfileSnapshot, VisionAttentionMode, VisionBackboneKind, VisionCellularConfig,
    VisionCellularState, VisionDragon, VisionDragonConfig, VisionDragonMultiOutput,
    VisionDragonOutput, VisionLatentActivation, VisionPatchEmbedMode, VisionPyramidConfig,
    VisionRhoStreamConfig, VisionTrmGraphConfig, VisionTrmGridMismatchPolicy,
    VisionTrmPredictSubstepKind, patchify, pool_patch_tokens, stage_aware_host_profile_reset,
    stage_aware_host_profile_snapshot, unpatchify,
};
#[cfg(feature = "benchmark")]
pub use model::{
    VisionDenseAttentionBenchAdapter, VisionDenseBenchAdapter, VisionRolloutScheduleBenchAdapter,
};
#[cfg(all(feature = "benchmark", feature = "cuda"))]
pub use train::run_vision_distill_decode_probe_cuda_with_seed;
#[cfg(feature = "train")]
pub use train::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, ImageNetTeacherTargetBatch, ImageTeacherTargetStore,
    MovingMnistRenderedClip, MovingMnistSplit, MovingMnistVideoDataLoader, MovingMnistVideoDataset,
    MovingMnistVideoDatasetConfig, VideoClipBatch, VisionNormalize,
    VisionVideoTrainProfileSnapshot, video_train_profile_reset, video_train_profile_snapshot,
};
#[cfg(feature = "benchmark")]
pub use train::{
    VISION_ARTIFACT_SCHEMA_VERSION, VISION_DISTILL_DECODE_PROBE_HARNESS_VERSION,
    VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION, VISION_DISTILL_LINEAR_PROBE_HARNESS_VERSION,
    VisionArtifactHeader, VisionDistillDecodeProbeReport, VisionDistillDecodeProbeStepMetrics,
    VisionDistillDeploySmokePrecision, VisionDistillDeploySmokeReport,
    VisionDistillFeatureExportReport, VisionDistillFeatureProbeAccuracyReport,
    VisionDistillFeatureProbeBackend, VisionDistillFeatureProbeDevice,
    VisionDistillFeatureProbeReport, VisionDistillFeatureProbeStepAccuracy,
    VisionDistillLinearProbeAccuracyReport, VisionDistillLinearProbeReport,
    VisionDistillLinearProbeStepAccuracy, VisionDistillServingBenchmarkBackend,
    VisionDistillServingBenchmarkDevice, VisionDistillServingBenchmarkReport,
    VisionDistillServingStepMetrics, export_vision_distill_feature_embeddings,
    push_vision_artifact_markdown_prelude, run_vision_distill_decode_probe_with_seed,
    run_vision_distill_deploy_smoke, run_vision_distill_feature_probe,
    run_vision_distill_feature_probe_for_teacher_with_seed,
    run_vision_distill_feature_probe_with_seed, run_vision_distill_linear_probe,
    run_vision_distill_linear_probe_for_teacher_with_seed,
    run_vision_distill_linear_probe_with_seed, run_vision_distill_serving_benchmark,
};

#[cfg(feature = "train")]
pub use checkpoint::{
    VisionBurnpackExportReport, export_vision_encoder_checkpoint_to_burnpack,
    load_training_config_for_checkpoint, load_vision_encoder_from_checkpoint,
    write_training_snapshot,
};
#[cfg(feature = "train")]
pub use config::*;
#[cfg(feature = "train")]
pub use loss::{
    VisionDistillationLossConfig, vision_distillation_loss, vision_distillation_loss_terms,
};

pub use wgsl::{FOVEATION_BUFFER_SHADER, FOVEATION_SHADER, PYRAMID_SHADER, SCATTER_BUFFER_SHADER};
