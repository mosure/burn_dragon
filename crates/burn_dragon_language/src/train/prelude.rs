#![allow(unused_imports)]

pub(crate) use std::fs;
pub(crate) use std::path::{Path, PathBuf};
pub(crate) use std::sync::Arc;
pub(crate) use std::sync::atomic::{AtomicBool, Ordering};
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
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param,
};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::grad_clipping::GradientClippingConfig;
pub(crate) use burn::optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, LearningRate,
};
pub(crate) use burn::tensor::Distribution as TensorDistribution;
pub(crate) use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
pub(crate) use burn::tensor::{Int, Tensor, TensorData};
pub(crate) use burn_train::metric::{LearningRateMetric, LossMetric};
pub(crate) use burn_train::{
    LearnerBuilder, LearningStrategy, TrainOutput, TrainStep, TrainingResult, ValidStep,
};
pub(crate) use tracing::info;

pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings};

#[cfg(all(feature = "cuda", test))]
pub(crate) use burn_cuda::Cuda;

pub(crate) use serde::Serialize;

pub(crate) use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingConfig, TrainingHyperparameters,
};
pub(crate) use crate::dataset::{
    Dataset, DatasetSplit, RandomDataLoader, SequenceBatch, build_dataset,
};
pub(crate) use crate::inference::build_model_config;
pub(crate) use crate::tokenizer::TokenizerConfig;
pub(crate) use crate::{ContextStrategyConfig, GenerationConfig, ModelOverrides};

pub(crate) use crate::loss::language_model_loss;
pub(crate) use burn_dragon_core::{BDH, BDHConfig};
pub(crate) use burn_dragon_train::train::constants::{
    FAST_TRAIN, ValidBackend, fast_train_enabled,
};
pub(crate) use burn_dragon_train::train::metrics::{
    DeviceMetric, LanguageModelOutput, LanguageModelTrainItem, LossValue, ScalarMetric,
};
pub(crate) use burn_dragon_train::train::pipeline::{
    ResolvedLrScheduler, ScheduleSource, TrainSchedule, adamw_config_from_optimizer,
    create_run_dir, write_latest_run,
};
pub(crate) use burn_dragon_train::{
    GdpoConfig, GdpoHardGate, LearningRateScheduleConfig, OptimizerConfig, WgpuRuntimeConfig,
};
