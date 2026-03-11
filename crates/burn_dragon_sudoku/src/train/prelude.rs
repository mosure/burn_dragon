#![allow(unused_imports)]

pub(crate) use std::fs;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
pub(crate) use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) use anyhow::{Context, Result, anyhow};
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
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::grad_clipping::GradientClippingConfig;
pub(crate) use burn::optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, LearningRate,
};
pub(crate) use burn::tensor::Distribution as TensorDistribution;
pub(crate) use burn::tensor::activation;
pub(crate) use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
pub(crate) use burn::tensor::{Int, Tensor, TensorData};
pub(crate) use burn_train::metric::{LearningRateMetric, LossMetric};
pub(crate) use burn_train::{
    InferenceStep as ValidStep, LearningResult as TrainingResult, SupervisedTraining, TrainOutput,
    TrainStep, TrainingStrategy as LearningStrategy,
};
pub(crate) use tracing::info;

pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings};

#[cfg(feature = "cli")]
pub(crate) use burn_autodiff::Autodiff;
#[cfg(all(feature = "cuda", test))]
pub(crate) use burn_cuda::Cuda;
#[cfg(feature = "cli")]
pub(crate) use burn_dragon_train::wgpu::{WgpuDevice, init_runtime};
#[cfg(feature = "cli")]
pub(crate) use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
pub(crate) use burn_wgpu::Wgpu;

pub(crate) use serde::Serialize;

pub(crate) use crate::config::{
    SudokuArtifactConfig, SudokuCacheMhcConfig, SudokuCacheUpdateConfig, SudokuDatasetConfig,
    SudokuDatasetSourceConfig, SudokuEasyRewardMode, SudokuGridPositional, SudokuHaltConfig,
    SudokuHardRewardMode, SudokuInfoRewardConfig, SudokuLossMask, SudokuModelConfig,
    SudokuPolicyConfig, SudokuPolicyHead, SudokuReconConfig, SudokuReconLoss, SudokuRevisitConfig,
    SudokuRewardBaselineConfig, SudokuRewardConfig, SudokuRewardShapingConfig,
    SudokuRewardShapingMetric, SudokuRolloutConfig, SudokuRolloutSchedule, SudokuTrainingConfig,
    SudokuTrainingHyperparameters, SudokuTraversal, SudokuTrmMode, SudokuValidationConfig,
    SudokuWriteGateMode,
};
pub(crate) use crate::dataset::{SudokuBatch, SudokuDataset, SudokuRandomDataLoader, SudokuSplit};
pub(crate) use crate::model::SudokuSaccadeModel;

pub(crate) use burn_dragon_train::train::constants::ValidBackend;
pub(crate) use burn_dragon_train::train::metrics::{
    DeviceMetric, LossValue, ScalarMetric, ScalarValue,
};
pub(crate) use burn_dragon_train::train::pipeline::{
    ResolvedLrScheduler, ScheduleSource, TrainSchedule, adamw_config_from_optimizer,
    create_run_dir, resolve_valid_steps_per_epoch, write_latest_run,
};
pub(crate) use burn_dragon_train::{
    GdpoConfig, GdpoHardGate, LearningRateScheduleConfig, OptimizerConfig,
    VisionArtifactOutputMode, WgpuRuntimeConfig,
};
