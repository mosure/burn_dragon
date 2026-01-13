pub(crate) use std::any::TypeId;
pub(crate) use std::collections::{HashMap, VecDeque};
pub(crate) use std::fs;
pub(crate) use std::io;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
pub(crate) use std::sync::Mutex;
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
pub(crate) use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::{AdamW, AdamWConfig, GradientsParams, LearningRate};
pub(crate) use burn::tensor::Distribution as TensorDistribution;
pub(crate) use burn::tensor::activation;
pub(crate) use burn::tensor::module::conv2d;
pub(crate) use burn::tensor::ops::{ConvOptions, InterpolateMode};
pub(crate) use burn::tensor::{Int, Tensor, TensorData};
pub(crate) use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
#[cfg(feature = "cli")]
pub(crate) use burn_autodiff::Autodiff;
#[cfg(any(feature = "train", feature = "cli"))]
pub(crate) use burn_ndarray::NdArrayDevice;
#[cfg(any(feature = "train", feature = "cli"))]
pub(crate) use burn_wgpu::WgpuDevice;
pub(crate) use burn_train::metric::{LearningRateMetric, LossMetric};
pub(crate) use burn_train::{
    LearnerBuilder,
    LearningStrategy,
    TrainingResult,
    TrainOutput,
    TrainStep,
    ValidStep,
};
#[cfg(feature = "cli")]
pub(crate) use burn_wgpu::Wgpu;
pub(crate) use tracing::info;

#[cfg(all(feature = "cuda", any(feature = "cli", test)))]
pub(crate) use burn_cuda::Cuda;

pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings};

#[cfg(feature = "cli")]
pub(crate) use crate::wgpu::init_runtime;
#[cfg(test)]
pub(crate) use burn_dragon_hatchling_vision::foveation;
#[cfg(feature = "cli")]
pub(crate) use crate::{load_training_config, load_vision_training_config};
pub(crate) use crate::{
    BDH, BDHConfig, Dataset, DatasetConfig, DatasetSplit, DinoFeatureStore, GdpoHardGate,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset, ImageNetDatasetConfig,
    ImageNetSplit, ImagenetteVariant, LearningRateScheduleConfig, ModelOverrides, OptimizerConfig,
    PatchGrid, RandomDataLoader, SequenceBatch, TrainingConfig, TrainingHyperparameters,
    VisionArtifactOutputMode, VisionDatasetConfig, VisionDatasetDownloadConfig,
    VisionDragonHatchling, VisionDragonHatchlingConfig, VisionDistillationLossConfig,
    VisionFoveaSamplingMode, VisionFoveaScatterMode, VisionFoveaWarpMode, VisionLejepaConfig,
    VisionLejepaLossConfig, VisionMaeConfig, VisionNormalize,
    VisionPyramidMode, VisionSaccadeConfig, VisionTeacherConfig,
    VisionTeacherVariant, VisionTrainingConfig,
    VisionTrainingHyperparameters, VisionTrainingModeConfig, build_dataset, build_model_config,
    language_model_loss, patchify, unpatchify, vision_distillation_loss,
};
pub(crate) use burn_dino::correctness::load_model_from_checkpoint;
pub(crate) use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
pub(crate) use serde::Serialize;

pub(crate) use crate::train::constants::*;
pub(crate) use crate::train::saccade::*;
pub(crate) use crate::train::teacher::*;
pub(crate) use crate::train::train::*;
pub(crate) use crate::train::vision::*;

pub(crate) use crate::train::metrics::{
    DeviceMetric, InvLossInput, LanguageModelOutput, LanguageModelTrainItem, LossValue,
    MemoryCleanupMetric, ProbeAccInput, ProbeLossInput, ReconLossInput, ScalarMetric,
    SigRegLossInput, VisionArtifactInput, VisionArtifactMetric, VisionOutput, VisionTrainItem,
};
