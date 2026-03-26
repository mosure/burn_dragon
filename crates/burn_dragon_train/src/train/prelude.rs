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
    AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param,
};
pub(crate) use burn::nn::loss::CrossEntropyLossConfig;
pub(crate) use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
pub(crate) use burn::optim::adaptor::OptimizerAdaptor;
pub(crate) use burn::optim::grad_clipping::GradientClippingConfig;
pub(crate) use burn::optim::{
    AdamW, AdamWConfig, GradientsAccumulator, GradientsParams, LearningRate, Optimizer,
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
    InferenceStep as ValidStep, LearningResult as TrainingResult, MultiDeviceOptim,
    SupervisedTraining, TrainOutput, TrainStep, TrainingStrategy as LearningStrategy,
};
#[cfg(feature = "cli")]
pub(crate) use burn_wgpu::Wgpu;
#[cfg(any(feature = "train", feature = "cli"))]
pub(crate) use burn_wgpu::WgpuDevice;
pub(crate) use tracing::info;

#[cfg(all(feature = "cuda", any(feature = "cli", test)))]
pub(crate) use burn_cuda::Cuda;

pub(crate) use burn::record::{BinFileRecorder, FullPrecisionSettings, Record};

#[cfg(feature = "cli")]
pub(crate) use crate::wgpu::init_runtime;
pub(crate) use crate::{
    GdpoHardGate, LearningRateScheduleConfig, OptimizerConfig, VisionArtifactOutputMode,
    VisionTeacherVariant,
};
#[cfg(feature = "burn_dino")]
pub(crate) use burn_dino::correctness::load_model_from_checkpoint;
#[cfg(feature = "burn_dino")]
pub(crate) use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
pub(crate) use serde::Serialize;

pub(crate) use crate::train::constants::*;
pub(crate) use crate::train::pipeline::*;
pub(crate) use crate::train::teacher::*;

pub(crate) use crate::train::metrics::{
    DeviceMemoryMetric, DeviceMetric, LossValue, MemoryCleanupMetric, MetricsBackend, ScalarMetric,
};
