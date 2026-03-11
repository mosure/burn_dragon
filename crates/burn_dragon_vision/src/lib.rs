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
mod device;
pub mod foveation;
#[cfg(feature = "train")]
pub mod loss;
pub mod model;
#[cfg(feature = "train")]
pub mod train;
pub mod wgsl;

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
    PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind, VisionAttentionMode,
    VisionBackboneKind, VisionCellularConfig, VisionCellularState, VisionDragon,
    VisionDragonConfig, VisionDragonMultiOutput, VisionDragonOutput, VisionLatentActivation,
    VisionPatchEmbedMode, VisionPyramidConfig, VisionRhoStreamConfig, VisionTrmGraphConfig,
    VisionTrmGridMismatchPolicy, patchify, pool_patch_tokens, unpatchify,
};
#[cfg(feature = "train")]
pub use train::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, MovingMnistSplit, MovingMnistVideoDataLoader,
    MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VideoClipBatch, VisionNormalize,
};

#[cfg(feature = "train")]
pub use config::*;
#[cfg(feature = "train")]
pub use loss::{VisionDistillationLossConfig, vision_distillation_loss};

pub use wgsl::{FOVEATION_BUFFER_SHADER, FOVEATION_SHADER, PYRAMID_SHADER, SCATTER_BUFFER_SHADER};
