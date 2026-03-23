#[cfg(feature = "benchmark")]
pub(crate) mod artifact;
pub(crate) mod dataset;
pub(crate) mod distill;
pub(crate) mod distill_runtime;
pub(crate) mod ema;
pub(crate) mod image_data;
pub(crate) mod losses;
pub(crate) mod models;
#[cfg(feature = "benchmark")]
pub(crate) mod probe;
#[cfg(feature = "benchmark")]
pub(crate) mod serving;
pub(crate) mod train;
pub(crate) mod video;

pub use image_data::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, ImageNetTeacherTargetBatch, ImageTeacherTargetStore,
    VisionNormalize,
};

#[cfg(feature = "benchmark")]
pub use artifact::{
    VISION_ARTIFACT_SCHEMA_VERSION, VisionArtifactHeader, push_vision_artifact_markdown_prelude,
};
pub(crate) use dataset::maybe_download_vision_dataset;
pub use distill_runtime::{
    VisionDistillCheckpointEvalSummary, eval_vision_distill_checkpoint_backend,
};
pub(crate) use ema::{
    ema_update_module, init_momentum_teacher, restore_optional_teacher_from_student,
    sync_optional_teacher_from_student,
};
pub(crate) use losses::{
    CollectedViews, LejepaArtifactBuildInput, build_lejepa_artifacts, collect_views,
    lejepa_invariance_loss, lejepa_sigreg_loss, lejepa_sigreg_loss_params,
    lejepa_teacher_invariance_loss, normalize_artifact_legend, normalize_columns,
    patch_heatmap_or_norm, pca_patch_heatmap, pca_patch_rgb, recon_psnr, sample_patch_mask,
    select_trajectory_indices, split_view_tensor, stack_views,
};
pub(crate) use models::{
    VisionDistillModel, VisionLejepaInit, VisionLejepaLosses, VisionLejepaModel, VisionMaeInit,
    VisionMaeLosses, VisionMaeModel, VisionProbe, VisionReconstructionHead,
    VisionReconstructionInit, VisionSaccadeHead, VisionSaccadeInputProjection,
    VisionSaccadeProjection,
};
#[cfg(all(feature = "benchmark", feature = "cuda"))]
pub use probe::run_vision_distill_decode_probe_cuda_with_seed;
#[cfg(feature = "benchmark")]
pub use probe::{
    VISION_DISTILL_DECODE_PROBE_HARNESS_VERSION, VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION,
    VISION_DISTILL_LINEAR_PROBE_HARNESS_VERSION, VisionDistillDecodeProbeReport,
    VisionDistillDecodeProbeStepMetrics, VisionDistillFeatureExportReport,
    VisionDistillFeatureProbeAccuracyReport, VisionDistillFeatureProbeBackend,
    VisionDistillFeatureProbeDevice, VisionDistillFeatureProbeReport,
    VisionDistillFeatureProbeStepAccuracy, VisionDistillLinearProbeAccuracyReport,
    VisionDistillLinearProbeReport, VisionDistillLinearProbeStepAccuracy,
    export_vision_distill_feature_embeddings, run_vision_distill_decode_probe_with_seed,
    run_vision_distill_feature_probe, run_vision_distill_feature_probe_for_teacher_with_seed,
    run_vision_distill_feature_probe_with_seed, run_vision_distill_linear_probe,
    run_vision_distill_linear_probe_for_teacher_with_seed,
    run_vision_distill_linear_probe_with_seed,
};
#[cfg(feature = "benchmark")]
pub use serving::{
    VisionDistillDeploySmokePrecision, VisionDistillDeploySmokeReport,
    VisionDistillServingBenchmarkBackend, VisionDistillServingBenchmarkDevice,
    VisionDistillServingBenchmarkReport, VisionDistillServingStepMetrics,
    run_vision_distill_deploy_smoke, run_vision_distill_serving_benchmark,
};
pub(crate) use train::train_vision_backend;
#[cfg(feature = "integration_test")]
pub(crate) use train::train_vision_backend_for_test;
pub use video::dataset::{
    MovingMnistRenderedClip, MovingMnistSplit, MovingMnistVideoDataLoader, MovingMnistVideoDataset,
    MovingMnistVideoDatasetConfig, VideoClipBatch, VideoTargetHorizonCurriculum,
};
pub(crate) use video::models::{VisionVideoLejepaLosses, VisionVideoLejepaModel};
pub use video::profile::{
    VisionVideoTrainProfileSnapshot, video_train_profile_reset, video_train_profile_snapshot,
};
pub(crate) use video::vjepa21::VisionVideoVjepa21Model;
