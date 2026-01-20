#![recursion_limit = "256"]

pub mod constants;
mod device;
pub mod kernel;
pub mod model;
pub mod positional;
pub use kernel::{BlockPattern1d, BlockPattern2d, BlockSparseConfig};
pub use model::{
    BDH, BDHConfig, FusedKernelConfig, ManifoldHyperConnections,
    ManifoldHyperConnectionsConfig, ModelState, PatchEmbed, PatchEmbedOutput, PatchGrid,
    SpatialPositionalEncodingKind, VisionAttentionMode,
    VisionDragonHatchling, VisionDragonHatchlingConfig, VisionDragonHatchlingOutput,
    VisionLatentActivation, VisionPatchEmbedMode, patchify, pool_patch_tokens, unpatchify,
};
#[cfg(feature = "viz")]
pub use model::LayerVizState;
#[cfg(feature = "train")]
pub use model::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};
pub use positional::RotaryEmbedding;
