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
#[cfg(all(feature = "benchmark", feature = "cuda"))]
pub use vision::run_vision_distill_decode_probe_cuda_with_seed;
#[cfg(feature = "integration_test")]
pub use vision::train::train_vision_backend_for_test;
pub use vision::train::{
    train_vision_backend, train_vision_backend_with_config_paths,
    train_vision_backend_with_planned_run,
};
pub use vision::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, ImageNetTeacherTargetBatch, ImageNetVideoDataLoader,
    ImageTeacherTargetStore, ImageTensorStore, MovingMnistRenderedClip, MovingMnistSplit,
    MovingMnistVideoDataLoader, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
    VideoClipBatch, VideoTargetHorizonCurriculum, VisionNormalize, VisionVideoTrainProfileSnapshot,
    video_train_profile_reset, video_train_profile_snapshot,
};
#[cfg(feature = "benchmark")]
pub use vision::{
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
pub use vision::{
    VisionDistillCheckpointEvalSummary, VisionRacCheckpointEvalSummary,
    eval_vision_distill_checkpoint_backend, eval_vision_rac_checkpoint_backend,
};
pub(crate) use vision::{
    VisionDistillModel, VisionLejepaInit, VisionLejepaModel, VisionReconstructionInit,
};
