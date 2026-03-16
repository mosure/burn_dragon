#![cfg_attr(not(feature = "cli"), allow(dead_code))]

mod prelude;

pub(crate) mod constants;
pub(crate) mod gdpo;
pub(crate) mod metrics;
pub(crate) mod pipeline;
pub(crate) mod profile;

pub(crate) mod foveation;
pub(crate) mod saccade;
pub(crate) mod scatter;
pub(crate) mod vision;

#[cfg(feature = "benchmark")]
pub mod bench;
#[cfg(test)]
mod lejepa_tests;
#[cfg(test)]
mod mae_tests;
#[cfg(test)]
mod test_utils;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use test_utils::{init_wgpu_test_runtime, wgpu_test_guard};

#[cfg(feature = "integration_test")]
pub use burn_dragon_train::train::gdpo::{gdpo_cpu_fallbacks, gdpo_reset_cpu_fallbacks};
#[cfg(feature = "integration_test")]
pub use burn_dragon_train::train::metrics::{loss_trace_len, loss_trace_reset, loss_trace_take};
pub(crate) use pipeline::resolve_vision_rollout;
pub use saccade::SaccadeFoveationSampler;
pub use vision::train::train_vision_backend;
#[cfg(feature = "integration_test")]
pub use vision::train::train_vision_backend_for_test;
pub use vision::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, MovingMnistRenderedClip, MovingMnistSplit,
    MovingMnistVideoDataLoader, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
    VideoClipBatch, VideoTargetHorizonCurriculum, VisionNormalize, VisionVideoTrainProfileSnapshot,
    video_train_profile_reset, video_train_profile_snapshot,
};
pub use vision::{VisionDistillCheckpointEvalSummary, eval_vision_distill_checkpoint_backend};
#[cfg(feature = "benchmark")]
pub use vision::{
    VISION_ARTIFACT_SCHEMA_VERSION, VisionArtifactHeader,
    VisionDistillDeploySmokePrecision, VisionDistillDeploySmokeReport,
    VisionDistillFeatureProbeAccuracyReport, VisionDistillFeatureProbeBackend,
    VisionDistillFeatureProbeDevice, VisionDistillFeatureProbeReport,
    VisionDistillFeatureProbeStepAccuracy,
    VisionDistillServingBenchmarkBackend, VisionDistillServingBenchmarkDevice,
    VisionDistillServingBenchmarkReport, VisionDistillServingStepMetrics,
    push_vision_artifact_markdown_prelude,
    run_vision_distill_deploy_smoke, run_vision_distill_feature_probe,
    run_vision_distill_serving_benchmark,
};
pub(crate) use vision::{
    VisionDistillModel, VisionLejepaInit, VisionLejepaModel, VisionReconstructionInit,
};
