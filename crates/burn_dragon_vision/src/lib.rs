#![recursion_limit = "256"]

//! Vision backbones and training adapters over the shared Dragon terminology/state contract.
//!
//! Paper mapping:
//! - `Dense` is dense-space latent refinement without persistent `rho`
//! - `Cellular` exposes token-local persistent `rho`
//! - `Pyramid` exposes structured primary/context/global recurrent banks built on shared core
//!   `BankedRhoState`

#[cfg(feature = "train")]
pub mod config;
pub mod constants;
#[cfg(feature = "train")]
pub mod checkpoint;
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
        pub use burn_dragon_core::{StructuredBankRole, StructuredRouteOperation,
            StructuredRoutePattern, StructuredRouteSpec};
    }

    pub mod model {
        pub use crate::model::{
            PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind,
            StageAwareHostProfileSnapshot, VisionAttentionMode, VisionBackboneKind, VisionCellularConfig,
            VisionCellularState, VisionDragon, VisionDragonConfig, VisionDragonMultiOutput,
            VisionDragonOutput, VisionLatentActivation, VisionPatchEmbedMode,
            VisionPyramidConfig, VisionRhoStreamConfig, VisionTrmGraphConfig,
            VisionTrmGridMismatchPolicy, patchify, pool_patch_tokens,
            stage_aware_host_profile_reset, stage_aware_host_profile_snapshot, unpatchify,
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
        pub use crate::loss::{VisionDistillationLossConfig, vision_distillation_loss};
        pub use crate::train::{
            CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
            ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
            ImageNetDatasetConfig, ImageNetSplit, MovingMnistSplit, MovingMnistVideoDataLoader,
            MovingMnistRenderedClip, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
            VideoClipBatch,
            VisionNormalize,
            VisionVideoTrainProfileSnapshot, video_train_profile_reset,
            video_train_profile_snapshot,
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
    PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind, StageAwareHostProfileSnapshot,
    VisionAttentionMode, VisionBackboneKind, VisionCellularConfig, VisionCellularState, VisionDragon,
    VisionDragonConfig, VisionDragonMultiOutput, VisionDragonOutput, VisionLatentActivation,
    VisionPatchEmbedMode, VisionPyramidConfig, VisionRhoStreamConfig, VisionTrmGraphConfig,
    VisionTrmGridMismatchPolicy, patchify, pool_patch_tokens, stage_aware_host_profile_reset,
    stage_aware_host_profile_snapshot, unpatchify,
};
#[cfg(feature = "train")]
pub use train::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, MovingMnistSplit, MovingMnistVideoDataLoader,
    MovingMnistRenderedClip, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
    VideoClipBatch, VisionNormalize, VisionVideoTrainProfileSnapshot,
    video_train_profile_reset, video_train_profile_snapshot,
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
pub use loss::{VisionDistillationLossConfig, vision_distillation_loss};

pub use wgsl::{FOVEATION_BUFFER_SHADER, FOVEATION_SHADER, PYRAMID_SHADER, SCATTER_BUFFER_SHADER};
