mod attention;
mod bdh;
mod config;
mod loss;
mod residual;
mod state;
mod vision;

pub use bdh::BDH;
pub use config::{BDHConfig, FusedKernelConfig};
pub use loss::{VisionDistillationLossConfig, language_model_loss, vision_distillation_loss};
#[cfg(feature = "viz")]
pub use state::LayerVizState;
pub use state::{LayerState, ModelState};
#[cfg(feature = "train")]
pub use vision::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};
pub use vision::{
    PatchEmbed, PatchEmbedOutput, PatchGrid, SpatialPositionalEncodingKind, VisionAttentionMode,
    VisionDragonHatchling, VisionDragonHatchlingConfig, VisionDragonHatchlingOutput,
    VisionLatentActivation, VisionPatchEmbedMode, patchify, pool_patch_tokens, unpatchify,
};
pub use residual::{ManifoldHyperConnections, ManifoldHyperConnectionsConfig};
