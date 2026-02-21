#![recursion_limit = "256"]

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

pub use constants::{FOVEA_AA_THRESHOLD, SACCADE_FOVEA_SUBSAMPLES};
pub use foveation::{
    CpuImageLevel, CpuPyramidCache, FoveaWarpMode, PyramidMode, build_pyramid_cache,
    image_from_nchw, lod_sigma_from_sigma, render_foveated_patch,
    render_foveated_patch_with_radius, sigma_from_unit,
};
pub use model::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, PatchEmbed, PatchEmbedOutput, PatchGrid,
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionDragon, VisionDragonConfig,
    VisionDragonMultiOutput, VisionDragonOutput, VisionLatentActivation, VisionNormalize,
    VisionPatchEmbedMode, VisionTrmGraphConfig, VisionTrmGridMismatchPolicy, patchify,
    pool_patch_tokens, unpatchify,
};

#[cfg(feature = "train")]
pub use config::*;
#[cfg(feature = "train")]
pub use loss::{VisionDistillationLossConfig, vision_distillation_loss};

pub use wgsl::{FOVEATION_BUFFER_SHADER, FOVEATION_SHADER, PYRAMID_SHADER, SCATTER_BUFFER_SHADER};
