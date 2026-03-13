#![allow(unused_imports)]

pub(crate) use std::any::TypeId;
pub(crate) use std::collections::{HashMap, VecDeque};
pub(crate) use std::fs;
pub(crate) use std::io;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::Mutex;
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) use anyhow::{Context, Result, anyhow};
#[cfg(feature = "cli")]
pub(crate) use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
pub(crate) use names::Generator;
pub(crate) use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};

pub(crate) use burn::data::dataloader::DataLoader;
pub(crate) use burn::lr_scheduler::{
    LrScheduler,
    cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
    exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    noam::{NoamLrScheduler, NoamLrSchedulerConfig},
    step::{StepLrScheduler, StepLrSchedulerConfig},
};
pub(crate) use burn::module::{
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, Param,
};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::grad_clipping::GradientClippingConfig;
pub(crate) use burn::optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, LearningRate,
};
pub(crate) use burn::tensor::Distribution as TensorDistribution;
pub(crate) use burn::tensor::activation;
pub(crate) use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
pub(crate) use burn::tensor::module::conv2d;
pub(crate) use burn::tensor::ops::{ConvOptions, InterpolateMode};
pub(crate) use burn::tensor::{Int, Tensor, TensorData};
#[cfg(feature = "cli")]
pub(crate) use burn_autodiff::Autodiff;
#[cfg(any(feature = "train", feature = "cli"))]
pub(crate) use burn_ndarray::NdArrayDevice;
pub(crate) use burn_train::metric::{LearningRateMetric, LossMetric};
pub(crate) use burn_train::{
    InferenceStep as ValidStep, LearningResult as TrainingResult, SupervisedTraining, TrainOutput,
    TrainStep, TrainingStrategy as LearningStrategy,
};
#[cfg(feature = "cli")]
pub(crate) use burn_wgpu::Wgpu;
#[cfg(any(feature = "train", feature = "cli"))]
pub(crate) use burn_wgpu::WgpuDevice;
pub(crate) use tracing::info;

#[cfg(all(feature = "cuda", any(feature = "cli", test)))]
pub(crate) use burn_cuda::Cuda;

pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings};
pub(crate) use burn_dragon_core::{DragonNorm, DragonNormConfig};

#[cfg(feature = "cli")]
pub(crate) use crate::config::load_vision_training_config;
pub(crate) use crate::config::{
    ImagenetteVariant, VisionDatasetConfig, VisionDatasetDownloadConfig, VisionDatasetSource,
    VisionDistillConfig, VisionFoveaSamplingMode, VisionFoveaScatterMode, VisionFoveaWarpMode,
    VisionLejepaConfig, VisionLejepaLossConfig, VisionMaeConfig, VisionMomentumTeacherConfig,
    VisionMovingMnistConfig, VisionPyramidMode, VisionSaccadeConfig,
    VisionSaccadeInputProjectionCnnConfig, VisionSaccadeInputProjectionConfig,
    VisionSaccadeInputProjectionMicroVitConfig, VisionTeacherConfig, VisionTeacherVariant,
    VisionTrainingConfig, VisionTrainingHyperparameters, VisionTrainingModeConfig,
    VisionVideoLejepaConfig, VisionVideoTemporalConfig,
};
pub(crate) use crate::loss::{
    VisionDistillationLossConfig, VisionDistillationLossTerms, vision_distillation_loss,
    vision_distillation_loss_terms,
};
pub(crate) use crate::train::VideoTargetHorizonCurriculum;
pub(crate) use crate::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, PatchGrid, SpatialPositionalEncodingKind,
    VisionAttentionMode, VisionDragon, VisionDragonConfig, VisionLatentActivation, VisionNormalize,
    VisionPatchEmbedMode, patchify, unpatchify,
};
#[cfg(feature = "burn_dino")]
pub(crate) use burn_dino::correctness::load_model_from_checkpoint;
#[cfg(feature = "burn_dino")]
pub(crate) use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
#[cfg(feature = "cli")]
pub(crate) use burn_dragon_train::wgpu::init_runtime;
pub(crate) use burn_dragon_train::{
    GdpoHardGate, LearningRateScheduleConfig, OptimizerConfig, VisionArtifactOutputMode,
};
pub(crate) use serde::Serialize;

pub(crate) use crate::train::constants::*;
pub(crate) use crate::train::metrics::{
    ActionClampRateInput, AdvantageAbsMeanInput, AdvantageStdInput, InvLossInput,
    LogProbMeanInput, LongRolloutComErrorToH24Input, LongRolloutInvToHorizonInput,
    LongRolloutStateMotionToHorizonInput, LongRolloutStateNormRatioToHorizonInput,
    LongRolloutVelocityErrorToH24Input, ModeSeparationRatioInput, ObserveLossInput,
    PolicyEntropyInput, PolicyLossInput, ProbeAccInput, ProbeLossInput, ReconLossInput,
    ReconPsnrFullInput, ReconPsnrMaskedInput, RolloutComErrorToH24Input,
    RolloutInvToHorizonInput, RolloutStateMotionToHorizonInput,
    RolloutStateNormRatioToHorizonInput, RolloutVelocityErrorToH24Input, SigRegLossInput,
    VISION_ROLLOUT_HORIZON_CAPS, VISION_ROLLOUT_HORIZON_COUNT, VisionArtifactInput,
    VisionArtifactMetric, VisionOutput, VisionTrainItem,
};
pub(crate) use crate::train::pipeline::*;
pub(crate) use crate::train::saccade::*;
#[cfg(feature = "integration_test")]
pub(crate) use crate::train::vision::train_vision_backend_for_test;
pub(crate) use crate::train::vision::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, CollectedViews,
    LejepaArtifactBuildInput, MovingMnistSplit, MovingMnistVideoDataLoader,
    MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VideoClipBatch, VisionDistillModel,
    VisionLejepaInit, VisionLejepaLosses, VisionLejepaModel, VisionMaeInit, VisionMaeLosses,
    VisionMaeModel, VisionProbe, VisionReconstructionHead, VisionReconstructionInit,
    VisionSaccadeHead, VisionSaccadeInputProjection, VisionSaccadeProjection,
    VisionVideoLejepaLosses, VisionVideoLejepaModel, build_lejepa_artifacts, collect_views,
    ema_update_module, init_momentum_teacher, lejepa_invariance_loss, lejepa_sigreg_loss,
    lejepa_sigreg_loss_params, lejepa_teacher_invariance_loss, maybe_download_vision_dataset,
    normalize_artifact_legend, normalize_columns, patch_heatmap_or_norm, pca_patch_heatmap,
    pca_patch_rgb, recon_psnr, restore_optional_teacher_from_student, sample_patch_mask,
    select_trajectory_indices, split_view_tensor, stack_views, sync_optional_teacher_from_student,
    train_vision_backend,
};
pub(crate) use burn_dragon_train::train::teacher::*;

pub(crate) use burn_dragon_train::train::metrics::{
    DeviceMemoryMetric, DeviceMetric, LanguageModelOutput, LanguageModelTrainItem, LossValue,
    MemoryCleanupMetric, MetricsBackend, OptionalScalarMetric, ScalarMetric,
};
