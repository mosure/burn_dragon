#![cfg_attr(not(feature = "cli"), allow(dead_code))]

use std::any::TypeId;
use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
#[cfg(feature = "cli")]
use clap::{Args as ClapArgs, Parser, Subcommand, ValueEnum};
use names::Generator;
use rand::{Rng, SeedableRng, rngs::StdRng, thread_rng};

use burn::data::dataloader::DataLoader;
use burn::lr_scheduler::{
    LrScheduler,
    cosine::{CosineAnnealingLrScheduler, CosineAnnealingLrSchedulerConfig},
    exponential::{ExponentialLrScheduler, ExponentialLrSchedulerConfig},
    linear::{LinearLrScheduler, LinearLrSchedulerConfig},
    noam::{NoamLrScheduler, NoamLrSchedulerConfig},
    step::{StepLrScheduler, StepLrSchedulerConfig},
};
use burn::module::{AutodiffModule, Content, Module, ModuleDisplay, ModuleDisplayDefault, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, LearningRate};
use burn::tensor::Distribution as TensorDistribution;
use burn::tensor::activation;
use burn::tensor::module::conv2d;
use burn::tensor::ops::{ConvOptions, InterpolateMode};
use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
#[cfg(feature = "cli")]
use burn_autodiff::Autodiff;
#[cfg(any(feature = "train", feature = "cli"))]
use burn_ndarray::NdArrayDevice;
use burn_train::metric::{LearningRateMetric, LossMetric};
use burn_train::{
    LearnerBuilder,
    LearningStrategy,
    TrainingResult,
    TrainOutput,
    TrainStep,
    ValidStep,
};
#[cfg(feature = "cli")]
use burn_wgpu::Wgpu;
use tracing::info;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

use burn::record::{BinFileRecorder, FullPrecisionSettings};

#[cfg(feature = "cli")]
use crate::wgpu::init_runtime;
#[cfg(test)]
use crate::foveation;
#[cfg(feature = "cli")]
use crate::{load_training_config, load_vision_training_config};
use crate::{
    BDH, BDHConfig, Dataset, DatasetConfig, DatasetSplit, DinoFeatureStore, ImageNetAugmentations,
    ImageNetBatch, ImageNetDataLoader, ImageNetDataset, ImageNetDatasetConfig, ImageNetSplit,
    ImagenetteVariant, LearningRateScheduleConfig, ModelOverrides, OptimizerConfig, PatchGrid,
    RandomDataLoader, SequenceBatch, TrainingConfig, TrainingHyperparameters,
    VisionArtifactOutputMode, VisionDatasetConfig, VisionDatasetDownloadConfig,
    VisionDragonHatchling, VisionDragonHatchlingConfig, VisionDistillationLossConfig,
    VisionFoveaSamplingMode, VisionLejepaConfig, VisionMaeConfig, VisionNormalize,
    VisionPyramidMode, VisionSaccadeConfig, VisionTeacherConfig, VisionTeacherVariant,
    VisionTrainingConfig,
    VisionTrainingHyperparameters, VisionTrainingModeConfig, build_dataset, build_model_config,
    language_model_loss, patchify, unpatchify, vision_distillation_loss,
};
use burn_dino::correctness::load_model_from_checkpoint;
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use serde::Serialize;

mod metrics;
mod artifacts;
mod foveation_cubecl;
#[cfg(feature = "benchmark")]
pub mod bench;

use metrics::{
    DeviceMetric, InvLossInput, LanguageModelOutput, LanguageModelTrainItem, ProbeAccInput,
    ProbeLossInput, ReconLossInput, ScalarMetric, SigRegLossInput, VisionArtifactInput,
    VisionArtifactMetric, VisionOutput, VisionTrainItem,
};

#[cfg(feature = "cli")]
#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Cli {
    #[command(flatten)]
    train: TrainArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[cfg(feature = "cli")]
#[derive(ClapArgs, Debug)]
struct TrainArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH", global = true)]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[cfg(feature = "cli")]
#[derive(Subcommand, Debug)]
enum Command {
    /// Build the character-level vocabulary and exit.
    BuildVocab,
    /// Train the vision model (distill or LeJEPA).
    Vision,
}

#[cfg(feature = "cli")]
#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
}

static FAST_TRAIN: AtomicBool = AtomicBool::new(false);
const LEJEPA_EPS: f32 = 1e-6;
const SACCADE_EPS: f32 = 1e-6;
// Saccade uses a single trajectory token per eye; next fovea params decode from it.
const SACCADE_TRAJ_TOKENS: usize = 1;
const SACCADE_SIGMA_MIN: f32 = 0.03;
const SACCADE_SIGMA_MAX: f32 = 0.5;
const SACCADE_LN_2: f32 = 0.69314718056;
const SACCADE_LOD_LOG2_MIN: f32 = -2.0;
const SACCADE_LOD_LOG2_MAX: f32 = 1.0;
const SACCADE_RING_WIDTH: f32 = 0.02;
const SACCADE_RING_INTENSITY: f32 = 2.0;
const SACCADE_RING_OUTER_SCALE: f32 = 2.5;
const SACCADE_RING_OUTER_INTENSITY: f32 = 0.7;
const SACCADE_VIEW_GAP: usize = 2;
const SACCADE_FOVEA_SUBSAMPLES: usize = 4;
const SACCADE_FOVEA_LOD_WINDOW: f32 = 3.0;
const SACCADE_FOVEA_SQRT2: f32 = 1.41421356237;
const SACCADE_FOVEA_PI: f32 = 3.14159265359;
const SACCADE_FOVEA_ERF_A: f32 = 0.147;
const SACCADE_FOVEA_SQRT_PI_OVER_2: f32 = 0.88622692545;

fn fast_train_enabled() -> bool {
    FAST_TRAIN.load(Ordering::Relaxed)
}

type ValidBackend<B> = <B as AutodiffBackend>::InnerBackend;

#[derive(Module, Debug)]
struct VisionDistillModel<B: BackendTrait> {
    model: VisionDragonHatchling<B>,
    loss: VisionDistillationLossConfig,
    teacher: Option<DinoVisionTransformer<B>>,
    #[module(ignore)]
    rollout: VisionRollout,
}

impl<B: BackendTrait> VisionDistillModel<B> {
    fn new(
        model: VisionDragonHatchling<B>,
        loss: VisionDistillationLossConfig,
        teacher: Option<DinoVisionTransformer<B>>,
        rollout: VisionRollout,
    ) -> Self {
        Self {
            model,
            loss,
            teacher,
            rollout,
        }
    }
}

#[derive(Module, Debug)]
struct VisionProbe<B: BackendTrait> {
    norm: LayerNorm<B>,
    head: Linear<B>,
}

impl<B: BackendTrait> VisionProbe<B> {
    fn new(embed_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let head = LinearConfig::new(embed_dim, num_classes.max(1)).init(device);
        Self { norm, head }
    }

    fn forward(&self, tokens: Tensor<B, 2>) -> Tensor<B, 2> {
        let tokens = self.norm.forward(tokens);
        self.head.forward(tokens)
    }
}

#[derive(Module, Debug)]
struct VisionReconstructionHead<B: BackendTrait> {
    norm: LayerNorm<B>,
    hidden: Option<Linear<B>>,
    out: Linear<B>,
}

impl<B: BackendTrait> VisionReconstructionHead<B> {
    fn new(embed_dim: usize, hidden_dim: usize, patch_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(embed_dim, hidden_dim).init(device))
        } else {
            None
        };
        let out_dim = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            embed_dim.max(1)
        };
        let out = LinearConfig::new(out_dim, patch_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
struct VisionSaccadeHead<B: BackendTrait> {
    norm: LayerNorm<B>,
    proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeHead<B> {
    fn new(embed_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let proj = LinearConfig::new(embed_dim, 3).init(device);
        Self { norm, proj }
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
struct VisionSaccadeProjection<B: BackendTrait> {
    norm: LayerNorm<B>,
    proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeProjection<B> {
    fn new(embed_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let proj = LinearConfig::new(embed_dim, out_dim).init(device);
        Self { norm, proj }
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
struct VisionLejepaModel<B: BackendTrait> {
    model: VisionDragonHatchling<B>,
    probe: VisionProbe<B>,
    probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    recon: Option<VisionReconstructionHead<B>>,
    mask_token: Option<Param<Tensor<B, 2>>>,
    config: VisionLejepaConfig,
    #[module(ignore)]
    rollout: VisionRollout,
}

struct VisionLejepaLosses<B: BackendTrait> {
    total: Tensor<B, 1>,
    inv: Tensor<B, 1>,
    sigreg: Tensor<B, 1>,
    recon: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

struct ViewGroupOutput<B: BackendTrait> {
    proj: Tensor<B, 3>,
    embed: Tensor<B, 3>,
    patch_tokens: Tensor<B, 3>,
}

impl<B: BackendTrait> VisionLejepaModel<B> {
    fn new(
        model: VisionDragonHatchling<B>,
        config: VisionLejepaConfig,
        embed_dim: usize,
        num_classes: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let probe = VisionProbe::new(embed_dim, num_classes, device);
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let recon = if config.recon_weight > 0.0 {
            if recon_patch_dim == 0 {
                None
            } else {
                Some(VisionReconstructionHead::new(
                    embed_dim,
                    config.recon_hidden_dim,
                    recon_patch_dim,
                    device,
                ))
            }
        } else {
            None
        };
        let mask_token = recon.as_ref().map(|_| {
            let token = Tensor::<B, 2>::random(
                [1, embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            Param::from_tensor(token)
        });
        Self {
            model,
            probe,
            probe_loss,
            recon,
            mask_token,
            config,
            rollout,
        }
    }

    fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
    ) -> VisionLejepaLosses<B> {
        let ImageNetBatch {
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
            labels,
            ..
        } = batch;

        let device = labels.device();
        let collected = collect_views(
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
        );
        let mut proj_groups = Vec::new();
        let mut embed_groups = Vec::new();
        let mut heatmap_source = None;
        let mut artifact_views = None;
        let mut probe_primary = None;
        let mut probe_embed = None;
        let mut recon_loss_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut recon_mask_sum = Tensor::<B, 1>::zeros([1], &device);
        let recon_enabled = self.recon.is_some();

        if !collected.global.is_empty() {
            let output = self.forward_view_group(&collected.global, steps, backprop_steps);
            probe_embed = Some(output.embed.clone());
            let [view_count, batch, dim] = output.embed.shape().dims::<3>();
            if view_count > 0 {
                probe_primary = Some(
                    output
                        .embed
                        .clone()
                        .slice_dim(0, 0..1)
                        .reshape([batch, dim]),
                );
                if heatmap_source.is_none() {
                    heatmap_source =
                        Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
                }
            }

            if recon_enabled {
                let (loss_sum, mask_sum, artifacts) =
                    self.recon_group_loss(
                        &collected.global,
                        steps,
                        backprop_steps,
                        true,
                        randomize_mask,
                    );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
                if let Some((views, residual)) = artifacts {
                    artifact_views = Some(views);
                    heatmap_source = Some(residual);
                }
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if !collected.local.is_empty() {
            let output = self.forward_view_group(&collected.local, steps, backprop_steps);
            if heatmap_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                heatmap_source =
                    Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if recon_enabled {
                let (loss_sum, mask_sum, _) = self.recon_group_loss(
                    &collected.local,
                    steps,
                    backprop_steps,
                    false,
                    randomize_mask,
                );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if proj_groups.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return VisionLejepaLosses {
                total: zero.clone(),
                inv: zero.clone(),
                sigreg: zero.clone(),
                recon: zero.clone(),
                probe_loss: zero.clone(),
                probe_acc: zero,
                artifacts: None,
            };
        }

        let proj = if proj_groups.len() == 1 {
            proj_groups.pop().expect("proj group")
        } else {
            Tensor::cat(proj_groups, 0)
        };
        let embed = if embed_groups.len() == 1 {
            embed_groups.pop().expect("embed group")
        } else {
            Tensor::cat(embed_groups, 0)
        };
        let inv = lejepa_invariance_loss(proj.clone());
        let sigreg = lejepa_sigreg_loss(proj.clone(), &self.config);
        let lambda = self.config.lambda.clamp(0.0, 1.0);
        let mut total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
        let recon = if recon_enabled {
            let denom = recon_mask_sum.clone().add_scalar(LEJEPA_EPS);
            let recon = recon_loss_sum / denom;
            let weight = self.config.recon_weight.max(0.0);
            total = total + recon.clone().mul_scalar(weight);
            recon
        } else {
            Tensor::<B, 1>::zeros([1], &device)
        };

        let probe_source = probe_embed.as_ref().unwrap_or(&embed);
        let [view_count, batch, embed_dim] = probe_source.shape().dims::<3>();
        let embed_flat = probe_source
            .clone()
            .reshape([view_count * batch, embed_dim])
            .detach();
        let labels_flat = labels.clone().repeat_dim(0, view_count);
        let probe_logits = self.probe.forward(embed_flat);
        let probe_loss = self.probe_loss.forward(probe_logits.clone(), labels_flat.clone());
        let probe_pred = probe_logits
            .clone()
            .argmax(1)
            .reshape([view_count * batch]);
        let probe_acc = probe_pred
            .equal(labels_flat)
            .float()
            .mean();

        let probe_primary = probe_primary.map(|embed| self.probe.forward(embed.detach()));
        let artifact_views = artifact_views.unwrap_or_else(|| collected.artifact_views());
        let legend = if artifact_views.is_empty() {
            None
        } else if recon_enabled && artifact_views.len() == 3 {
            Some(vec![
                "input".to_string(),
                "masked_input".to_string(),
                "reconstruction".to_string(),
            ])
        } else if artifact_views.len() == 1 {
            Some(vec!["input".to_string()])
        } else {
            Some(
                (0..artifact_views.len())
                    .map(|idx| format!("view_{idx}"))
                    .collect(),
            )
        };
        let artifacts = build_lejepa_artifacts(
            &self.config,
            &artifact_views,
            None,
            heatmap_source,
            probe_primary,
            Some(labels),
            legend,
        );

        VisionLejepaLosses {
            total,
            inv,
            sigreg,
            recon,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }

    fn recon_group_loss(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
        capture_artifacts: bool,
        randomize_mask: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let recon = match &self.recon {
            Some(recon) => recon,
            None => {
                let device = views
                    .get(0)
                    .map(|view| view.device())
                    .unwrap_or_default();
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero, None);
            }
        };
        if views.is_empty() {
            let device = <B as BackendTrait>::Device::default();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let device = views[0].device();
        let [batch, channels, height, width] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed_raw(stacked.clone());
        let [total, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let patch_size = height / grid_h;
        if patch_size == 0 || height % grid_h != 0 || width % grid_w != 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let target_patches = patchify(stacked, patch_size);
        let mask = sample_patch_mask(
            &device,
            total,
            tokens,
            self.config.recon_mask_ratio,
            randomize_mask,
        );
        let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
        let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
        if let Some(mask_token) = &self.mask_token {
            let token = mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, total)
                .repeat_dim(1, tokens);
            masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
        }
        let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
        let embed_out =
            self.model
                .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

        let pred_patches = recon.forward(embed_out.patch_tokens);
        let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
        debug_assert_eq!(
            target_patches.shape().dims::<3>(),
            pred_patches.shape().dims::<3>(),
            "recon patches shape mismatch"
        );
        if total == 0 || tokens == 0 || patch_dim == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .powf_scalar(2.0)
            .mul(mask_expanded.clone())
            .sum();
        let mask_sum = mask.clone().sum().mul_scalar(patch_dim as f32);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches =
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(mask_expanded);
            Some((vec![views[0].clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (loss_sum, mask_sum, artifacts)
    }

    fn forward_view_group(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
    ) -> ViewGroupOutput<B> {
        let view_count = views.len();
        let [batch, _, _, _] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed(stacked);
        let embed_out = self
            .model
            .forward_tokens_embed_steps_rollout(patch.tokens, steps, backprop_steps);
        let cls_embed = embed_out.cls_token;
        let patch_tokens = embed_out.patch_tokens;
        let [total, embed_dim] = cls_embed.shape().dims::<2>();
        debug_assert_eq!(total, view_count * batch, "lejepa embed mismatch");
        let tokens = Tensor::cat(
            vec![cls_embed.clone().unsqueeze_dim::<3>(1), patch_tokens.clone()],
            1,
        );
        let proj_tokens = self.model.project_tokens(tokens);
        let proj_dim = proj_tokens.shape().dims::<3>()[2];
        let proj_cls = proj_tokens
            .slice_dim(1, 0..1)
            .reshape([view_count, batch, proj_dim]);
        let embed_cls = cls_embed.reshape([view_count, batch, embed_dim]);
        ViewGroupOutput {
            proj: proj_cls,
            embed: embed_cls,
            patch_tokens,
        }
    }
}

#[derive(Module, Debug)]
struct VisionMaeModel<B: BackendTrait> {
    model: VisionDragonHatchling<B>,
    recon: VisionReconstructionHead<B>,
    mask_token: Param<Tensor<B, 2>>,
    config: VisionMaeConfig,
    #[module(ignore)]
    rollout: VisionRollout,
}

struct VisionMaeLosses<B: BackendTrait> {
    total: Tensor<B, 1>,
    recon: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionMaeModel<B> {
    fn new(
        model: VisionDragonHatchling<B>,
        config: VisionMaeConfig,
        embed_dim: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.recon_hidden_dim,
            recon_patch_dim,
            device,
        );
        let token = Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let mask_token = Param::from_tensor(token);
        Self {
            model,
            recon,
            mask_token,
            config,
            rollout,
        }
    }

    fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionMaeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, artifacts) =
            self.recon_loss(images, steps, backprop_steps, randomize_mask, capture_artifacts);
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let total = recon.clone().mul_scalar(self.config.recon_weight.max(0.0));

        let artifacts = artifacts.and_then(|(views, residual)| {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    ..VisionLejepaConfig::default()
                },
                &views,
                None,
                Some(residual),
                None,
                Some(labels),
                Some(vec![
                    "input".to_string(),
                    "masked_input".to_string(),
                    "reconstruction".to_string(),
                ]),
            )
        });

        VisionMaeLosses {
            total,
            recon,
            artifacts,
        }
    }

    fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let device = images.device();
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch = self.model.patch_embed_raw(images.clone());
        let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let patch_size = height / grid_h;
        if patch_size == 0 || height % grid_h != 0 || width % grid_w != 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let target_patches = patchify(images.clone(), patch_size);
        let mask = sample_patch_mask(
            &device,
            batch,
            tokens,
            self.config.mask_ratio,
            randomize_mask,
        );
        let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
        let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
        let token = self
            .mask_token
            .val()
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, tokens);
        masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
        let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
        let embed_out =
            self.model
                .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

        let pred_patches = self.recon.forward(embed_out.patch_tokens);
        let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
        if total == 0 || tokens == 0 || patch_dim == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .powf_scalar(2.0)
            .mul(mask_expanded.clone())
            .sum();
        let mask_sum = mask.clone().sum().mul_scalar(patch_dim as f32);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches =
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(mask_expanded);
            Some((vec![images.clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (loss_sum, mask_sum, artifacts)
    }
}

#[derive(Clone, Debug)]
struct LevelCoordsCache<B: BackendTrait> {
    inner: Arc<Mutex<HashMap<(usize, usize), Tensor<B, 2>>>>,
}

impl<B: BackendTrait> LevelCoordsCache<B> {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_or_build(&self, grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
        let key = (grid.height, grid.width);
        if let Ok(cache) = self.inner.lock() {
            if let Some(coords) = cache.get(&key) {
                return coords.clone();
            }
        }
        let coords = build_level_coords::<B>(grid, device);
        if let Ok(mut cache) = self.inner.lock() {
            cache.insert(key, coords.clone());
        }
        coords
    }
}

impl<B: BackendTrait> Module<B> for LevelCoordsCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for LevelCoordsCache<B> {
    type InnerModule = LevelCoordsCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        LevelCoordsCache::new()
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for LevelCoordsCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let entries = self.inner.lock().map(|cache| cache.len()).unwrap_or(0);
        content.add("entries", &entries).optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for LevelCoordsCache<B> {}

#[derive(Clone, Debug)]
struct UpsampleWeightsCache<B: BackendTrait> {
    inner: Arc<Mutex<HashMap<(usize, usize, usize, usize), Tensor<B, 2>>>>,
}

impl<B: BackendTrait> UpsampleWeightsCache<B> {
    fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    fn get_or_build(&self, from: PatchGrid, to: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
        let key = (from.height, from.width, to.height, to.width);
        if let Ok(cache) = self.inner.lock() {
            if let Some(weights) = cache.get(&key) {
                return weights.clone();
            }
        }
        let from_tokens = from.num_patches();
        let to_tokens = to.num_patches();
        let mut mapping = vec![0.0f32; to_tokens * from_tokens];
        for ty in 0..to.height {
            let src_y = (ty as f32 * from.height as f32 / to.height as f32)
                .floor()
                .min((from.height - 1) as f32) as usize;
            for tx in 0..to.width {
                let src_x = (tx as f32 * from.width as f32 / to.width as f32)
                    .floor()
                    .min((from.width - 1) as f32) as usize;
                let src_idx = src_y * from.width + src_x;
                let dst_idx = ty * to.width + tx;
                mapping[dst_idx * from_tokens + src_idx] = 1.0;
            }
        }
        let weights =
            Tensor::<B, 2>::from_data(TensorData::new(mapping, [to_tokens, from_tokens]), device);
        if let Ok(mut cache) = self.inner.lock() {
            cache.insert(key, weights.clone());
        }
        weights
    }
}

impl<B: BackendTrait> Module<B> for UpsampleWeightsCache<B> {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for UpsampleWeightsCache<B> {
    type InnerModule = UpsampleWeightsCache<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        UpsampleWeightsCache::new()
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for UpsampleWeightsCache<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let entries = self.inner.lock().map(|cache| cache.len()).unwrap_or(0);
        content.add("entries", &entries).optional()
    }
}

impl<B: BackendTrait> ModuleDisplay for UpsampleWeightsCache<B> {}

#[derive(Module, Debug)]
struct VisionSaccadeModel<B: BackendTrait> {
    model: VisionDragonHatchling<B>,
    recon: VisionReconstructionHead<B>,
    // Learned initial trajectory state for the recurrent rollout.
    trajectory_token: Param<Tensor<B, 2>>,
    // Per-eye identity bias to keep multi-eye rollouts disentangled.
    eye_token: Param<Tensor<B, 2>>,
    input_proj: VisionSaccadeProjection<B>,
    fovea_proj: VisionSaccadeProjection<B>,
    residual_proj: VisionSaccadeProjection<B>,
    saccade_head: VisionSaccadeHead<B>,
    config: VisionSaccadeConfig,
    level_coords_cache: LevelCoordsCache<B>,
    upsample_weights_cache: UpsampleWeightsCache<B>,
    #[module(ignore)]
    rollout: VisionRollout,
}

struct VisionSaccadeLosses<B: BackendTrait> {
    total: Tensor<B, 1>,
    inv: Tensor<B, 1>,
    sigreg: Tensor<B, 1>,
    recon: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

struct SaccadeMipLevel<B: BackendTrait> {
    tokens: Tensor<B, 3>,
    grid: PatchGrid,
    image: Tensor<B, 4>,
}

struct SaccadeLaplacianImages<B: BackendTrait> {
    residuals: Vec<Tensor<B, 4>>,
    coarse: Tensor<B, 4>,
}

impl<B: BackendTrait> VisionSaccadeModel<B> {
    fn new(
        model: VisionDragonHatchling<B>,
        config: VisionSaccadeConfig,
        embed_dim: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.recon_hidden_dim,
            recon_patch_dim,
            device,
        );
        let trajectory_token = Tensor::<B, 2>::random(
            [SACCADE_TRAJ_TOKENS, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let num_eyes = config.num_eyes.max(1);
        let eye_token = if num_eyes > 1 {
            Tensor::<B, 2>::random(
                [num_eyes, embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            )
        } else {
            Tensor::<B, 2>::zeros([num_eyes, embed_dim.max(1)], device)
        };
        let input_proj = VisionSaccadeProjection::new(embed_dim, embed_dim, device);
        let fovea_proj = VisionSaccadeProjection::new(3, embed_dim, device);
        let residual_proj = VisionSaccadeProjection::new(embed_dim, embed_dim, device);
        let saccade_head = VisionSaccadeHead::new(embed_dim, device);
        Self {
            model,
            recon,
            trajectory_token: Param::from_tensor(trajectory_token),
            eye_token: Param::from_tensor(eye_token),
            input_proj,
            fovea_proj,
            residual_proj,
            saccade_head,
            config,
            level_coords_cache: LevelCoordsCache::new(),
            upsample_weights_cache: UpsampleWeightsCache::new(),
            rollout,
        }
    }

    fn detach_if<const D: usize>(tensor: Tensor<B, D>, detach: bool) -> Tensor<B, D> {
        if detach {
            tensor.detach()
        } else {
            tensor
        }
    }

    fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionSaccadeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, inv, sigreg, artifacts) =
            self.recon_loss(images, steps, backprop_steps, randomize_mask, capture_artifacts);
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let lambda = self.config.lambda.clamp(0.0, 1.0);
        let mut total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
        let recon_weight = self.config.recon_weight.max(0.0);
        if recon_weight > 0.0 {
            total = total + recon.clone().mul_scalar(recon_weight);
        }

        let artifacts = artifacts.and_then(|(views, residual, frames, legend)| {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    ..VisionLejepaConfig::default()
                },
                &views,
                frames,
                Some(residual),
                None,
                Some(labels),
                Some(legend),
            )
        });

        VisionSaccadeLosses {
            total,
            inv,
            sigreg,
            recon,
            artifacts,
        }
    }

    fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>, Option<Tensor<B, 5>>, Vec<String>)>,
    ) {
        let device = images.device();
        let _ = randomize_mask;
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch = self.model.patch_embed_raw(images.clone());
        let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero.clone(), zero.clone(), zero, None);
        }
        let patch_size = height / grid_h;
        if patch_size == 0 || height % grid_h != 0 || width % grid_w != 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero.clone(), zero.clone(), zero, None);
        }

        let mip_levels = self.build_mip_pyramid(images.clone(), patch_size);
        if mip_levels.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero.clone(), zero.clone(), zero, None);
        }
        let input_levels: Vec<Tensor<B, 3>> =
            mip_levels.iter().map(|level| level.tokens.clone()).collect();
        let grids: Vec<PatchGrid> = mip_levels.iter().map(|level| level.grid).collect();
        let input_residuals = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => input_levels.clone(),
            VisionPyramidMode::Laplacian => self.decompose_pyramid(&input_levels, &grids),
        };
        let laplacian_images = if matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            self.build_laplacian_images(&mip_levels)
        } else {
            None
        };
        let base_grid = build_foveated_base_grid::<B>(patch_size, &device);
        let traj_len = self.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let num_eyes = self.config.num_eyes.max(1);
        let inner_steps = self.config.inner_steps.max(1);

        let base_traj = self
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let mut trajs = vec![base_traj; num_eyes];
        let mut state_levels: Vec<Tensor<B, 3>> = input_residuals
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
            .collect();
        let rollout_steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(rollout_steps);
        let detach_until = rollout_steps.saturating_sub(backprop_steps);
        let low_mem_pre_rollout = self.config.low_mem_pre_rollout;
        let capture_traj = capture_artifacts
            && self
                .config
                .artifact_max_views
                .saturating_sub(3)
            > 0;
        let mut traj_steps = if capture_traj {
            Some(Vec::with_capacity(rollout_steps))
        } else {
            None
        };
        let mut frame_steps = if capture_artifacts {
            Some(Vec::with_capacity(rollout_steps))
        } else {
            None
        };
        let mut last_patch_views: Option<Vec<Tensor<B, 4>>> = None;
        for step_idx in 0..rollout_steps {
            let pre_rollout = low_mem_pre_rollout && step_idx + 1 <= detach_until;
            let anchor_traj =
                low_mem_pre_rollout && detach_until > 0 && step_idx + 1 == detach_until + 1;
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state_levels.clone(),
                VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
            };
            let mut updates: Vec<Tensor<B, 3>> = state_levels
                .iter()
                .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
                .collect();
            let mut step_traj = if capture_traj || capture_artifacts {
                Some(Vec::with_capacity(num_eyes))
            } else {
                None
            };
            let mut step_patches = if capture_artifacts {
                Some(Vec::with_capacity(num_eyes))
            } else {
                None
            };
            let mut next_trajs = Vec::with_capacity(num_eyes);
            for eye_idx in 0..num_eyes {
                let mut traj = trajs[eye_idx].clone();
                if pre_rollout {
                    traj = traj.detach();
                } else if anchor_traj {
                    let anchor = self
                        .trajectory_token
                        .val()
                        .reshape([1, traj_len, embed_dim])
                        .repeat_dim(0, batch);
                    traj = traj + (anchor.clone() - anchor.detach());
                }
                let eye_embed = self
                    .eye_token
                    .val()
                    .slice_dim(0, eye_idx..eye_idx + 1)
                    .reshape([1, 1, embed_dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, traj_len);
                let eye_embed = Self::detach_if(eye_embed, pre_rollout);
                let traj_with_eye = traj.clone() + eye_embed.clone();
                let traj_with_eye = Self::detach_if(traj_with_eye, pre_rollout);
                let traj_summary = traj_with_eye
                    .clone()
                    .mean_dim(1)
                    .reshape([batch, 1, embed_dim]);
                let params = self.saccade_head.forward(traj_summary);
                let params = Self::detach_if(params, pre_rollout);
                let (mean, sigma) = self.decode_saccade_params(params);
                let mean = Self::detach_if(mean, pre_rollout);
                let sigma = Self::detach_if(sigma, pre_rollout);
                let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
                let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
                let mean_detached = mean_step.clone().detach();
                let sigma_detached = sigma_step.clone().detach();
                if let Some(steps) = &mut step_traj {
                    steps.push((mean_detached.clone(), sigma_detached.clone()));
                }
                // Reuse fovea weights for both context and residual scatter to keep updates localized
                // to the sampled region of the pyramid.
                let weights = self.mip_gaussian_weights(&mip_levels, mean.clone(), sigma.clone());
                let patch_image = self.foveated_patch_image(
                    &mip_levels,
                    &base_grid,
                    mean_step.clone(),
                    sigma_step.clone(),
                    laplacian_images.as_ref(),
                );
                let patch_tokens = self.model.patch_embed_raw(patch_image.clone()).tokens;
                let patch_tokens = Self::detach_if(patch_tokens, pre_rollout);
                if let Some(step_patches) = step_patches.as_mut() {
                    step_patches.push(patch_image);
                }
                let input_context = patch_tokens;
                let state_context = self.mip_weighted_sum(&state_composed, &weights);
                let fovea_params = Tensor::cat(vec![mean, sigma], 2);
                let fovea_embed = self.fovea_proj.forward(fovea_params);
                let fovea_embed = Self::detach_if(fovea_embed, pre_rollout);
                let input_tokens =
                    self.input_proj.forward(input_context) + state_context + fovea_embed;
                let input_tokens = Self::detach_if(input_tokens, pre_rollout);
                let input_tokens = input_tokens.repeat_dim(1, traj_len);
                let tokens_in = traj_with_eye + input_tokens;
                let out_tokens = self
                    .model
                    .forward_tokens_embed_steps(tokens_in, inner_steps)
                    .patch_tokens;
                let out_tokens = Self::detach_if(out_tokens, pre_rollout);
                let residual = self.residual_proj.forward(out_tokens.clone());
                let residual = Self::detach_if(residual, pre_rollout);
                let residual_pool = residual
                    .clone()
                    .mean_dim(1)
                    .reshape([batch, 1, embed_dim]);
                let next_traj = out_tokens;
                for (update, weights) in updates.iter_mut().zip(weights.iter()) {
                    let update_eye =
                        self.weighted_sum_tokens(weights.clone().swap_dims(1, 2), residual_pool.clone());
                    *update = update.clone() + update_eye;
                }
                next_trajs.push(next_traj);
            }
            for (state, update) in state_levels.iter_mut().zip(updates.iter()) {
                *state = state.clone() + update.clone();
            }
            if let Some(step_traj) = step_traj {
                if capture_traj {
                    if let Some(steps) = &mut traj_steps {
                        steps.push(step_traj.clone());
                    }
                }
                if let Some(frames) = &mut frame_steps {
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state_levels.clone(),
                        VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
                    };
                    let pred_patches = self.recon.forward(state_composed[0].clone());
                    let recon_view =
                        unpatchify(pred_patches, patch_size, height, width, channels);
                    let mut input_frame = images.clone();
                    let mut appended_patch = false;
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            input_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            input_frame = overlay;
                        }
                    }
                    let mut frame_views = Vec::new();
                    let push_view = |views: &mut Vec<Tensor<B, 4>>, view: Tensor<B, 4>| {
                        if !views.is_empty() && SACCADE_VIEW_GAP > 0 {
                            views.push(view_separator_like(&view, SACCADE_VIEW_GAP));
                        }
                        views.push(view);
                    };
                    push_view(&mut frame_views, input_frame);
                    if let Some(step_patches) = step_patches {
                        if let Some(patch_views) =
                            saccade_patch_views(step_patches, height)
                        {
                            last_patch_views =
                                Some(patch_views.iter().map(|view| view.clone().detach()).collect());
                            for patch_view in patch_views {
                                push_view(&mut frame_views, patch_view);
                            }
                            appended_patch = true;
                        }
                    }
                    if !appended_patch {
                        if let Some(patch_views) = last_patch_views.clone() {
                            for patch_view in patch_views {
                                push_view(&mut frame_views, patch_view);
                            }
                        }
                    }
                    push_view(&mut frame_views, recon_view);
                    let frame = Tensor::cat(frame_views, 3);
                    frames.push(frame);
                }
            }
            trajs = next_trajs;
            if step_idx + 1 <= detach_until {
                trajs = trajs.into_iter().map(|traj| traj.detach()).collect();
                for level in &mut state_levels {
                    *level = level.clone().detach();
                }
            }
        }

        let state_composed = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => state_levels.clone(),
            VisionPyramidMode::Laplacian => self.compose_pyramid(&state_levels, &grids),
        };
        let (inv, sigreg) = self.pyramid_lejepa_loss(&state_composed);

        let mut loss_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut mask_sum_value = 0.0f32;
        let mut pred_base = None;
        let mut target_base = None;
        for (level_idx, level) in mip_levels.iter().enumerate() {
            let pred_patches = self.recon.forward(state_composed[level_idx].clone());
            let [total, level_tokens, patch_dim] = pred_patches.shape().dims::<3>();
            if total == 0 || level_tokens == 0 || patch_dim == 0 {
                continue;
            }
            let target_patches = patchify(level.image.clone(), patch_size);
            let diff = pred_patches.clone() - target_patches.clone();
            loss_sum = loss_sum + diff.powf_scalar(2.0).sum();
            mask_sum_value += (total * level_tokens * patch_dim) as f32;
            if level_idx == 0 {
                pred_base = Some(pred_patches);
                target_base = Some(target_patches);
            }
        }
        if mask_sum_value == 0.0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero.clone(), zero.clone(), zero, None);
        }
        let mask_sum =
            Tensor::<B, 1>::from_data(TensorData::new(vec![mask_sum_value], [1]), &device);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_base.clone().unwrap_or_else(|| {
                Tensor::<B, 3>::zeros([batch, tokens, patch_size * patch_size * channels], &device)
            });
            let target_first = target_base.clone().unwrap_or_else(|| {
                Tensor::<B, 3>::zeros([batch, tokens, patch_size * patch_size * channels], &device)
            });
            let recon_view = unpatchify(pred_first.clone(), patch_size, height, width, channels);
            let residual = pred_first - target_first;
            let mut target_width = width;
            if let Some(patch_views) = &last_patch_views {
                for patch_view in patch_views {
                    target_width = target_width.max(patch_view.shape().dims::<4>()[3]);
                }
            }
            let mut images_view = images.clone();
            if let Some(steps) = traj_steps.as_ref() {
                if let Some(last_step) = steps.last() {
                    for (eye_idx, (mean, sigma)) in last_step.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            images_view.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            images_view = overlay;
                        }
                    }
                }
            }
            let images_view = pad_view_width(images_view, target_width);
            let recon_view = pad_view_width(recon_view, target_width);
            let patch_views = last_patch_views.map(|patch_views| {
                patch_views
                    .into_iter()
                    .map(|patch_view| pad_view_width_centered(patch_view, target_width))
                    .collect::<Vec<_>>()
            });
            let mut views = Vec::new();
            let mut legend = Vec::new();
            views.push(images_view);
            legend.push("input_with_fovea".to_string());
            if let Some(patch_views) = patch_views {
                for (eye_idx, patch_view) in patch_views.into_iter().enumerate() {
                    views.push(patch_view);
                    legend.push(format!("foveated_patch_eye_{eye_idx}"));
                }
            }
            views.push(recon_view);
            legend.push("reconstruction".to_string());
            if let Some(steps) = traj_steps {
                let max_extra = self
                    .config
                    .artifact_max_views
                    .saturating_sub(views.len());
                let mut remaining = max_extra;
                for idx in select_trajectory_indices(steps.len(), max_extra) {
                    for (eye_idx, (mean, sigma)) in steps[idx].iter().enumerate() {
                        if remaining == 0 {
                            break;
                        }
                        if let Some(view) = saccade_circle_overlay(
                            images.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            views.push(pad_view_width(view, target_width));
                            legend.push(format!(
                                "trajectory_overlay_step_{idx}_eye_{eye_idx}"
                            ));
                            remaining = remaining.saturating_sub(1);
                        }
                    }
                    if remaining == 0 {
                        break;
                    }
                }
            }
            let frames = frame_steps.and_then(|frames| {
                if frames.is_empty() {
                    return None;
                }
                let mut max_width = 0;
                for frame in &frames {
                    let width = frame.shape().dims::<4>()[3];
                    max_width = max_width.max(width);
                }
                let mut stacked = Vec::with_capacity(frames.len());
                for frame in frames {
                    let frame = pad_view_width(frame, max_width);
                    stacked.push(frame.unsqueeze_dim::<5>(1));
                }
                Some(Tensor::cat(stacked, 1))
            });
            Some((views, residual, frames, legend))
        } else {
            None
        };

        (loss_sum, mask_sum, inv, sigreg, artifacts)
    }

    fn decode_saccade_params(
        &self,
        params: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mean = activation::sigmoid(params.clone().slice_dim(2, 0..2))
            .mul_scalar(1.0 - 2.0 * SACCADE_EPS)
            .add_scalar(SACCADE_EPS);
        let sigma = activation::sigmoid(params.slice_dim(2, 2..3))
            .mul_scalar(SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN)
            .add_scalar(SACCADE_SIGMA_MIN);
        (mean, sigma)
    }

    fn build_mip_pyramid(
        &self,
        images: Tensor<B, 4>,
        patch_size: usize,
    ) -> Vec<SaccadeMipLevel<B>> {
        let max_levels = self.config.mip_levels.max(1);
        let mut levels = Vec::new();
        let mut current = images;
        for level in 0..max_levels {
            let [_, _, height, width] = current.shape().dims::<4>();
            if height < patch_size || width < patch_size {
                break;
            }
            let crop_h = height - (height % patch_size);
            let crop_w = width - (width % patch_size);
            if crop_h == 0 || crop_w == 0 {
                break;
            }
            let cropped = if crop_h != height || crop_w != width {
                current
                    .clone()
                    .slice_dim(2, 0..crop_h)
                    .slice_dim(3, 0..crop_w)
            } else {
                current.clone()
            };
            let patch = self.model.patch_embed_raw(cropped.clone());
            let grid = patch.grid;
            if grid.height == 0 || grid.width == 0 {
                break;
            }
            levels.push(SaccadeMipLevel {
                tokens: patch.tokens,
                grid,
                image: cropped.clone(),
            });

            if level + 1 == max_levels {
                break;
            }
            let next = downsample_image(cropped.clone());
            if let Some(next) = next {
                current = next;
            } else {
                break;
            }
        }
        levels
    }

    fn build_laplacian_images(
        &self,
        levels: &[SaccadeMipLevel<B>],
    ) -> Option<SaccadeLaplacianImages<B>> {
        if levels.len() < 2 {
            return None;
        }
        let device = levels
            .first()
            .map(|level| level.image.device())
            .unwrap_or_default();
        let mut residuals = Vec::with_capacity(levels.len().saturating_sub(1));
        for idx in 0..levels.len().saturating_sub(1) {
            let current = &levels[idx].image;
            let next = &levels[idx + 1].image;
            let [batch, _, level_h, level_w] = current.shape().dims::<4>();
            let [_, _, next_h, next_w] = next.shape().dims::<4>();
            if level_h == 0 || level_w == 0 {
                return None;
            }
            let grid = build_image_grid::<B>(level_h, level_w, next_h, next_w, &device);
            let grid = if grid.shape().dims::<4>()[0] == batch {
                grid
            } else {
                grid.repeat_dim(0, batch)
            };
            let upsampled = grid_sample_2d_bilinear::<B>(next.clone(), grid);
            residuals.push(current.clone() - upsampled);
        }
        let coarse = levels
            .last()
            .expect("levels not empty")
            .image
            .clone();
        Some(SaccadeLaplacianImages { residuals, coarse })
    }

    fn decompose_pyramid(
        &self,
        levels: &[Tensor<B, 3>],
        grids: &[PatchGrid],
    ) -> Vec<Tensor<B, 3>> {
        let mut residuals = Vec::with_capacity(levels.len());
        for idx in 0..levels.len() {
            if idx + 1 < levels.len() {
                let upsampled =
                    self.upsample_tokens(levels[idx + 1].clone(), grids[idx + 1], grids[idx]);
                residuals.push(levels[idx].clone() - upsampled);
            } else {
                residuals.push(levels[idx].clone());
            }
        }
        residuals
    }

    fn compose_pyramid(
        &self,
        residuals: &[Tensor<B, 3>],
        grids: &[PatchGrid],
    ) -> Vec<Tensor<B, 3>> {
        if residuals.is_empty() {
            return Vec::new();
        }
        let mut composed_rev = Vec::with_capacity(residuals.len());
        let mut current = residuals
            .last()
            .expect("residuals not empty")
            .clone();
        composed_rev.push(current.clone());
        if residuals.len() > 1 {
            for idx in (0..residuals.len() - 1).rev() {
                let upsampled = self.upsample_tokens(current, grids[idx + 1], grids[idx]);
                current = residuals[idx].clone() + upsampled;
                composed_rev.push(current.clone());
            }
        }
        composed_rev.reverse();
        composed_rev
    }

    fn level_coords_cached(&self, grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
        self.level_coords_cache.get_or_build(grid, device)
    }

    fn upsample_weights_cached(
        &self,
        from: PatchGrid,
        to: PatchGrid,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        self.upsample_weights_cache.get_or_build(from, to, device)
    }

    fn upsample_tokens(
        &self,
        tokens: Tensor<B, 3>,
        from: PatchGrid,
        to: PatchGrid,
    ) -> Tensor<B, 3> {
        let [batch, tokens_len, dim] = tokens.shape().dims::<3>();
        if from.height == 0 || from.width == 0 || to.height == 0 || to.width == 0 {
            return Tensor::<B, 3>::zeros([batch, to.num_patches().max(1), dim], &tokens.device());
        }
        if tokens_len == 0 || tokens_len != from.num_patches() {
            return Tensor::<B, 3>::zeros([batch, to.num_patches().max(1), dim], &tokens.device());
        }
        if from.height == to.height && from.width == to.width {
            return tokens;
        }
        let device = tokens.device();
        let weights = self.upsample_weights_cached(from, to, &device);
        let weights = weights.unsqueeze_dim::<3>(0).repeat_dim(0, batch);
        self.weighted_sum_tokens(weights, tokens)
    }

    fn pyramid_lejepa_loss(
        &self,
        levels: &[Tensor<B, 3>],
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let Some(first) = levels.first() else {
            let device = self.trajectory_token.val().device();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero);
        };
        let device = first.device();
        let mut views = Vec::with_capacity(levels.len());
        for level in levels {
            let [batch, tokens, dim] = level.shape().dims::<3>();
            if batch == 0 || tokens == 0 || dim == 0 {
                continue;
            }
            let pooled = level.clone().mean_dim(1).reshape([batch, 1, dim]);
            let proj = self.model.project_tokens(pooled);
            let proj_dim = proj.shape().dims::<3>()[2];
            views.push(proj.reshape([1, batch, proj_dim]));
        }
        if views.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero);
        }
        let proj = if views.len() == 1 {
            views.pop().expect("single view")
        } else {
            Tensor::cat(views, 0)
        };
        let inv = lejepa_invariance_loss(proj.clone());
        let sigreg = lejepa_sigreg_loss_params(
            proj,
            self.config.sigreg_knots,
            self.config.sigreg_t_max,
            self.config.sigreg_proj_dim,
        );
        (inv, sigreg)
    }

    fn mip_gaussian_weights(
        &self,
        levels: &[SaccadeMipLevel<B>],
        mean: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
    ) -> Vec<Tensor<B, 3>> {
        let [batch, traj_tokens, _] = mean.shape().dims::<3>();
        let device = mean.device();
        let level_count = levels.len();
        if level_count == 0 {
            return Vec::new();
        }

        let mean_flat = mean.reshape([batch * traj_tokens, 2]);
        let sigma_flat = sigma.reshape([batch * traj_tokens, 1]);
        let lod_sigma = self
            .lod_sigma_from_sigma(sigma_flat.clone())
            .clamp_min(SACCADE_EPS);
        let max_level = level_count.saturating_sub(1) as f32;

        let mut total = Tensor::<B, 2>::zeros([batch * traj_tokens, 1], &device);
        let mut raw_weights = Vec::with_capacity(level_count);
        for (level_idx, level) in levels.iter().enumerate() {
            let coords = self.level_coords_cached(level.grid, &device);
            let tokens_len = level.tokens.shape().dims::<3>()[1].max(1);
            let coords = coords.reshape([1, tokens_len, 2]);
            let diff = mean_flat.clone().unsqueeze_dim::<3>(1) - coords;
            let dist2 = diff
                .powf_scalar(2.0)
                .sum_dim(2)
                .reshape([batch * traj_tokens, tokens_len]);
            let sigma2 = sigma_flat
                .clone()
                .powf_scalar(2.0)
                .add_scalar(SACCADE_EPS)
                .repeat_dim(1, tokens_len);
            let spatial = (dist2.clone() / sigma2.mul_scalar(2.0))
                .mul_scalar(-1.0)
                .exp();

            let sigma_tokens = sigma_flat.clone().repeat_dim(1, tokens_len);
            let dist = dist2.add_scalar(SACCADE_EPS).sqrt();
            let dist_norm = dist / sigma_tokens;
            let lod_center = dist_norm
                .clamp_min(1.0)
                .log()
                .div_scalar(SACCADE_LN_2)
                .clamp_max(max_level);
            let lod_sigma = lod_sigma.clone().repeat_dim(1, tokens_len);
            let diff = lod_center.clone().sub_scalar(level_idx as f32) / lod_sigma;
            let lod_weight = diff.powf_scalar(2.0).mul_scalar(-0.5).exp();
            let lod_window = lod_center
                .sub_scalar(level_idx as f32)
                .abs()
                .lower_equal_elem(SACCADE_FOVEA_LOD_WINDOW);
            let lod_weight = Tensor::<B, 2>::zeros(lod_weight.shape().dims::<2>(), &device)
                .mask_where(lod_window, lod_weight);

            let weights = spatial * lod_weight;
            let sum = weights.clone().sum_dim(1).reshape([batch * traj_tokens, 1]);
            total = total + sum;
            raw_weights.push(weights);
        }

        let total = total.add_scalar(SACCADE_EPS);
        let mut weights_out = Vec::with_capacity(level_count);
        for (level, weights) in levels.iter().zip(raw_weights.into_iter()) {
            let tokens_len = level.tokens.shape().dims::<3>()[1].max(1);
            let denom = total.clone().repeat_dim(1, tokens_len);
            let weights = (weights / denom).reshape([batch, traj_tokens, tokens_len]);
            weights_out.push(weights);
        }
        weights_out
    }

    fn erfinv_approx(&self, values: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = values.device();
        let shape = values.shape().dims::<3>();
        let ones = Tensor::<B, 3>::ones(shape, &device);
        let sign = ones
            .clone()
            .mul_scalar(-1.0)
            .mask_where(values.clone().greater_equal_elem(0.0), ones.clone());
        let xx = values.clamp_min(-0.999).clamp_max(0.999);
        let ln = ones.clone().sub(xx.clone().powf_scalar(2.0)).log();
        let term = ln
            .clone()
            .mul_scalar(0.5)
            .add_scalar(2.0 / (SACCADE_FOVEA_PI * SACCADE_FOVEA_ERF_A));
        let inside = term
            .clone()
            .powf_scalar(2.0)
            .sub(ln.div_scalar(SACCADE_FOVEA_ERF_A))
            .clamp_min(0.0);
        let result = inside.sqrt().sub(term).clamp_min(0.0).sqrt();
        sign * result
    }

    fn erf_approx(&self, values: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = values.device();
        let shape = values.shape().dims::<3>();
        let ones = Tensor::<B, 3>::ones(shape, &device);
        let sign = ones
            .clone()
            .mul_scalar(-1.0)
            .mask_where(values.clone().greater_equal_elem(0.0), ones.clone());
        let ax = values.abs();
        let t = ones.clone().div(ax.clone().mul_scalar(0.3275911).add_scalar(1.0));
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let poly = t
            .clone()
            .mul_scalar(a5)
            .add_scalar(a4)
            .mul(t.clone())
            .add_scalar(a3)
            .mul(t.clone())
            .add_scalar(a2)
            .mul(t.clone())
            .add_scalar(a1)
            .mul(t);
        let y = ones.clone().sub(poly.mul((ax.clone().mul(ax).mul_scalar(-1.0)).exp()));
        sign * y
    }

    fn foveated_warp(
        &self,
        u: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
        radius: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let sigma_safe = sigma.clamp_min(SACCADE_EPS);
        let radius_safe = radius.clamp_min(SACCADE_EPS);
        let k = radius_safe / sigma_safe.clone();
        let u_max = self
            .erf_approx(k.div_scalar(SACCADE_FOVEA_SQRT2))
            .clamp_max(0.999);
        let u_scaled = u.clamp_min(-1.0).clamp_max(1.0) * u_max.clone();
        let erf_inv = self.erfinv_approx(u_scaled);
        let offset = erf_inv.clone().mul(sigma_safe.clone()).mul_scalar(SACCADE_FOVEA_SQRT2);
        let deriv = sigma_safe
            .mul_scalar(SACCADE_FOVEA_SQRT2)
            .mul(u_max)
            .mul_scalar(SACCADE_FOVEA_SQRT_PI_OVER_2)
            .mul(erf_inv.clone().powf_scalar(2.0).exp());
        (offset, deriv)
    }

    // Foveated patch sampling on the image pyramid (GPU tensor path).
    fn foveated_patch_image(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    ) -> Tensor<B, 4> {
        self.foveated_patch_image_with_radius(
            levels,
            base_grid,
            mean,
            sigma.clone(),
            sigma,
            laplacian_images,
        )
    }

    fn foveated_patch_image_with_radius(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        radius: Tensor<B, 2>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    ) -> Tensor<B, 4> {
        let device = mean.device();
        let Some(first) = levels.first() else {
            let [batch, _] = mean.shape().dims::<2>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let full_patch_h = patch_h;
        if batch == 0 || channels == 0 || patch_h == 0 || patch_w == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), channels.max(1), patch_h.max(1), patch_w.max(1)],
                &device,
            );
        }
        let base_grid = if base_grid.shape().dims::<4>()[0] == batch {
            base_grid.clone()
        } else {
            base_grid.clone().repeat_dim(0, batch)
        };

        let use_laplacian = matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian);
        let laplacian_fallback = if use_laplacian && laplacian_images.is_none() {
            self.build_laplacian_images(levels)
        } else {
            None
        };
        let laplacian_images = if use_laplacian {
            laplacian_images.or(laplacian_fallback.as_ref())
        } else {
            None
        };

        let min_side = width.min(height) as f32;
        let mean = mean.clamp_min(SACCADE_EPS).clamp_max(1.0 - SACCADE_EPS);
        let mean_x = mean.clone().slice_dim(1, 0..1).reshape([batch, 1, 1]);
        let mean_y = mean.slice_dim(1, 1..2).reshape([batch, 1, 1]);
        let radius_norm = radius.clamp_min(SACCADE_EPS).reshape([batch, 1, 1]);
        let sigma_norm = sigma
            .clamp_min(SACCADE_EPS)
            .reshape([batch, 1, 1])
            .min_pair(radius_norm.clone());
        let sigma_px = sigma_norm.clone().mul_scalar(min_side);
        let radius_px = radius_norm.clone().mul_scalar(min_side);
        let lod_sigma = self
            .lod_sigma_from_sigma(sigma_norm.clone().reshape([batch, 1]))
            .reshape([batch, 1, 1])
            .clamp_min(SACCADE_EPS);
        let center_x = mean_x.mul_scalar(width as f32);
        let center_y = mean_y.mul_scalar(height as f32);
        match self.config.fovea_sampling_mode {
            VisionFoveaSamplingMode::Batched => self.foveated_patch_sample_batched(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            ),
            VisionFoveaSamplingMode::Sequential => self.foveated_patch_sample_sequential(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            ),
            VisionFoveaSamplingMode::Cubecl => {
                if let Some(patch) = foveation_cubecl::try_foveated_patch_cubecl(
                    levels,
                    &base_grid,
                    &center_x,
                    &center_y,
                    &sigma_px,
                    &radius_px,
                    &lod_sigma,
                    laplacian_images,
                ) {
                    patch
                } else {
                    self.foveated_patch_sample_sequential(
                        levels,
                        base_grid,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        laplacian_images,
                        full_patch_h,
                    )
                }
            }
            VisionFoveaSamplingMode::Subpatch => {
                let subpatch = self.config.fovea_subpatch_size;
                if subpatch == 0 {
                    return self.foveated_patch_sample_sequential(
                        levels,
                        base_grid,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        laplacian_images,
                        full_patch_h,
                    );
                }
                self.foveated_patch_sample_subpatch(
                    levels,
                    base_grid,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    laplacian_images,
                    subpatch,
                    full_patch_h,
                )
            }
        }
    }

    fn foveated_patch_sample_batched(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        let device = base_grid.device();
        let Some(first) = levels.first() else {
            let [batch, _, _, _] = base_grid.shape().dims::<4>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let full_half = full_patch_h as f32 * 0.5;
        let pixel_du = 1.0 / full_half.max(1.0);
        let subsamples = SACCADE_FOVEA_SUBSAMPLES * SACCADE_FOVEA_SUBSAMPLES;
        let mut jitter_values = Vec::with_capacity(subsamples * 2);
        for sy in 0..SACCADE_FOVEA_SUBSAMPLES {
            for sx in 0..SACCADE_FOVEA_SUBSAMPLES {
                let jitter_x =
                    (sx as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
                let jitter_y =
                    (sy as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
                jitter_values.push(jitter_x / full_half);
                jitter_values.push(jitter_y / full_half);
            }
        }
        let jitter = Tensor::<B, 5>::from_data(
            TensorData::new(jitter_values, [subsamples, 1, 1, 1, 2]),
            &device,
        );
        let base_grid = base_grid
            .unsqueeze_dim::<5>(0)
            .repeat_dim(0, subsamples);
        let grid = (base_grid + jitter).reshape([subsamples * batch, patch_h, patch_w, 2]);

        let expand_3d = |tensor: Tensor<B, 3>| -> Tensor<B, 3> {
            let [_, h, w] = tensor.shape().dims::<3>();
            tensor
                .unsqueeze_dim::<4>(0)
                .repeat_dim(0, subsamples)
                .reshape([subsamples * batch, h, w])
        };
        let expand_4d = |tensor: Tensor<B, 4>| -> Tensor<B, 4> {
            let [_, ch, h, w] = tensor.shape().dims::<4>();
            tensor
                .unsqueeze_dim::<5>(0)
                .repeat_dim(0, subsamples)
                .reshape([subsamples * batch, ch, h, w])
        };

        let sigma_px = expand_3d(sigma_px);
        let radius_px = expand_3d(radius_px);
        let center_x = expand_3d(center_x);
        let center_y = expand_3d(center_y);
        let lod_sigma = expand_3d(lod_sigma);

        let ux = grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let uy = grid.slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let (dx, dx_deriv) = self.foveated_warp(ux, sigma_px.clone(), radius_px.clone());
        let (dy, dy_deriv) = self.foveated_warp(uy, sigma_px.clone(), radius_px.clone());
        let local_scale = dx_deriv.abs().max_pair(dy_deriv.abs()).mul_scalar(pixel_du);
        let img_x = center_x + dx.clone();
        let img_y = center_y + dy.clone();
        let fx = img_x.div_scalar(width as f32);
        let fy = img_y.div_scalar(height as f32);

        let sigma_sq = sigma_px.clone().powf_scalar(2.0);
        let dist = dx
            .clone()
            .powf_scalar(2.0)
            .div(sigma_sq.clone())
            .add(dy.clone().powf_scalar(2.0).div(sigma_sq))
            .sqrt();
        let zeros = Tensor::<B, 3>::zeros(dist.shape().dims::<3>(), &device);
        let dist_safe = dist.clone().clamp_min(1.0);
        let lod_dist = dist_safe
            .log()
            .div_scalar(SACCADE_LN_2)
            .mask_where(dist.lower_equal_elem(1.0), zeros.clone());
        let scale_safe = local_scale.clone().clamp_min(1.0);
        let lod_scale = scale_safe
            .log()
            .div_scalar(SACCADE_LN_2)
            .mask_where(local_scale.lower_equal_elem(1.0), zeros);
        let max_level = levels.len().saturating_sub(1) as f32;
        let lod = lod_dist
            .max_pair(lod_scale)
            .clamp_min(0.0)
            .clamp_max(max_level);

        let grid_shape = fx.shape().dims::<3>();
        let make_grid = |fx: &Tensor<B, 3>, fy: &Tensor<B, 3>, level_w: usize, level_h: usize| {
            let grid_x = if level_w > 1 {
                fx.clone()
                    .mul_scalar(level_w as f32)
                    .sub_scalar(0.5)
                    .mul_scalar(2.0 / (level_w - 1) as f32)
                    .add_scalar(-1.0)
            } else {
                Tensor::<B, 3>::zeros(grid_shape, &device)
            };
            let grid_y = if level_h > 1 {
                fy.clone()
                    .mul_scalar(level_h as f32)
                    .sub_scalar(0.5)
                    .mul_scalar(2.0 / (level_h - 1) as f32)
                    .add_scalar(-1.0)
            } else {
                Tensor::<B, 3>::zeros(grid_shape, &device)
            };
            Tensor::cat(
                vec![grid_x.unsqueeze_dim::<4>(3), grid_y.unsqueeze_dim::<4>(3)],
                3,
            )
        };

        let laplacian_samples = if let Some(laplacian) = laplacian_images {
            let [_, _, coarse_h, coarse_w] = laplacian.coarse.shape().dims::<4>();
            let coarse_grid = make_grid(&fx, &fy, coarse_w, coarse_h);
            let coarse_sample =
                grid_sample_2d_bilinear::<B>(expand_4d(laplacian.coarse.clone()), coarse_grid);
            let mut residual_samples = Vec::with_capacity(laplacian.residuals.len());
            for residual in laplacian.residuals.iter() {
                let [_, _, res_h, res_w] = residual.shape().dims::<4>();
                let residual_grid = make_grid(&fx, &fy, res_w, res_h);
                residual_samples.push(grid_sample_2d_bilinear::<B>(
                    expand_4d(residual.clone()),
                    residual_grid,
                ));
            }
            let mut recon_samples = Vec::with_capacity(levels.len());
            let mut current = coarse_sample;
            recon_samples.push(current.clone());
            for residual in residual_samples.iter().rev() {
                current = current + residual.clone();
                recon_samples.push(current.clone());
            }
            recon_samples.reverse();
            Some(recon_samples)
        } else {
            None
        };

        let mut color =
            Tensor::<B, 4>::zeros([subsamples * batch, channels, patch_h, patch_w], &device);
        let mut weight_sum =
            Tensor::<B, 3>::zeros([subsamples * batch, patch_h, patch_w], &device);
        for (level_idx, level) in levels.iter().enumerate() {
            let level_f = level_idx as f32;
            let diff = lod.clone().sub_scalar(level_f).div(lod_sigma.clone());
            let weight = diff.powf_scalar(2.0).mul_scalar(-0.5).exp();
            let window_mask = lod
                .clone()
                .sub_scalar(level_f)
                .abs()
                .lower_equal_elem(SACCADE_FOVEA_LOD_WINDOW);
            let weight = Tensor::<B, 3>::zeros(weight.shape().dims::<3>(), &device)
                .mask_where(window_mask, weight);
            let sample = if let Some(laplacian_samples) = laplacian_samples.as_ref() {
                laplacian_samples[level_idx].clone()
            } else {
                let [_, _, level_h, level_w] = level.image.shape().dims::<4>();
                let level_grid = make_grid(&fx, &fy, level_w, level_h);
                grid_sample_2d_bilinear::<B>(expand_4d(level.image.clone()), level_grid)
            };
            color = color + sample * weight.clone().unsqueeze_dim::<4>(1);
            weight_sum = weight_sum + weight;
        }
        let weight_sum = weight_sum.clamp_min(SACCADE_EPS);
        let sample = color / weight_sum.unsqueeze_dim::<4>(1);
        let sample = sample.reshape([subsamples, batch, channels, patch_h, patch_w]);
        let mut accum =
            Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
        for idx in 0..subsamples {
            let slice = sample
                .clone()
                .slice_dim(0, idx..idx + 1)
                .squeeze_dim::<4>(0);
            accum = accum + slice;
        }
        accum.mul_scalar(1.0 / subsamples as f32)
    }

    fn foveated_patch_sample_sequential(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        let device = base_grid.device();
        let Some(first) = levels.first() else {
            let [batch, _, _, _] = base_grid.shape().dims::<4>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let full_half = full_patch_h as f32 * 0.5;
        let pixel_du = 1.0 / full_half.max(1.0);
        let subsamples = SACCADE_FOVEA_SUBSAMPLES * SACCADE_FOVEA_SUBSAMPLES;
        let mut accum =
            Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);

        for sy in 0..SACCADE_FOVEA_SUBSAMPLES {
            for sx in 0..SACCADE_FOVEA_SUBSAMPLES {
                let jitter_x =
                    (sx as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
                let jitter_y =
                    (sy as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
                let jitter = Tensor::<B, 4>::from_data(
                    TensorData::new(vec![jitter_x / full_half, jitter_y / full_half], [1, 1, 1, 2]),
                    &device,
                );
                let grid = base_grid.clone() + jitter;

                let ux = grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
                let uy = grid.slice_dim(3, 1..2).squeeze_dim::<3>(3);
                let (dx, dx_deriv) =
                    self.foveated_warp(ux, sigma_px.clone(), radius_px.clone());
                let (dy, dy_deriv) =
                    self.foveated_warp(uy, sigma_px.clone(), radius_px.clone());
                let local_scale = dx_deriv.abs().max_pair(dy_deriv.abs()).mul_scalar(pixel_du);
                let img_x = center_x.clone() + dx.clone();
                let img_y = center_y.clone() + dy.clone();
                let fx = img_x.div_scalar(width as f32);
                let fy = img_y.div_scalar(height as f32);

                let sigma_sq = sigma_px.clone().powf_scalar(2.0);
                let dist = dx
                    .clone()
                    .powf_scalar(2.0)
                    .div(sigma_sq.clone())
                    .add(dy.clone().powf_scalar(2.0).div(sigma_sq))
                    .sqrt();
                let zeros = Tensor::<B, 3>::zeros(dist.shape().dims::<3>(), &device);
                let dist_safe = dist.clone().clamp_min(1.0);
                let lod_dist = dist_safe
                    .log()
                    .div_scalar(SACCADE_LN_2)
                    .mask_where(dist.lower_equal_elem(1.0), zeros.clone());
                let scale_safe = local_scale.clone().clamp_min(1.0);
                let lod_scale = scale_safe
                    .log()
                    .div_scalar(SACCADE_LN_2)
                    .mask_where(local_scale.lower_equal_elem(1.0), zeros);
                let max_level = levels.len().saturating_sub(1) as f32;
                let lod = lod_dist
                    .max_pair(lod_scale)
                    .clamp_min(0.0)
                    .clamp_max(max_level);

                let grid_shape = fx.shape().dims::<3>();
                let make_grid =
                    |fx: &Tensor<B, 3>, fy: &Tensor<B, 3>, level_w: usize, level_h: usize| {
                        let grid_x = if level_w > 1 {
                            fx.clone()
                                .mul_scalar(level_w as f32)
                                .sub_scalar(0.5)
                                .mul_scalar(2.0 / (level_w - 1) as f32)
                                .add_scalar(-1.0)
                        } else {
                            Tensor::<B, 3>::zeros(grid_shape, &device)
                        };
                        let grid_y = if level_h > 1 {
                            fy.clone()
                                .mul_scalar(level_h as f32)
                                .sub_scalar(0.5)
                                .mul_scalar(2.0 / (level_h - 1) as f32)
                                .add_scalar(-1.0)
                        } else {
                            Tensor::<B, 3>::zeros(grid_shape, &device)
                        };
                        Tensor::cat(
                            vec![grid_x.unsqueeze_dim::<4>(3), grid_y.unsqueeze_dim::<4>(3)],
                            3,
                        )
                    };

                let laplacian_samples = if let Some(laplacian) = laplacian_images {
                    let [_, _, coarse_h, coarse_w] = laplacian.coarse.shape().dims::<4>();
                    let coarse_grid = make_grid(&fx, &fy, coarse_w, coarse_h);
                    let coarse_sample =
                        grid_sample_2d_bilinear::<B>(laplacian.coarse.clone(), coarse_grid);
                    let mut residual_samples = Vec::with_capacity(laplacian.residuals.len());
                    for residual in laplacian.residuals.iter() {
                        let [_, _, res_h, res_w] = residual.shape().dims::<4>();
                        let residual_grid = make_grid(&fx, &fy, res_w, res_h);
                        residual_samples.push(grid_sample_2d_bilinear::<B>(
                            residual.clone(),
                            residual_grid,
                        ));
                    }
                    let mut recon_samples = Vec::with_capacity(levels.len());
                    let mut current = coarse_sample;
                    recon_samples.push(current.clone());
                    for residual in residual_samples.iter().rev() {
                        current = current + residual.clone();
                        recon_samples.push(current.clone());
                    }
                    recon_samples.reverse();
                    Some(recon_samples)
                } else {
                    None
                };

                let mut color =
                    Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
                let mut weight_sum =
                    Tensor::<B, 3>::zeros([batch, patch_h, patch_w], &device);
                for (level_idx, level) in levels.iter().enumerate() {
                    let level_f = level_idx as f32;
                    let diff = lod.clone().sub_scalar(level_f).div(lod_sigma.clone());
                    let weight = diff.powf_scalar(2.0).mul_scalar(-0.5).exp();
                    let window_mask = lod
                        .clone()
                        .sub_scalar(level_f)
                        .abs()
                        .lower_equal_elem(SACCADE_FOVEA_LOD_WINDOW);
                    let weight = Tensor::<B, 3>::zeros(weight.shape().dims::<3>(), &device)
                        .mask_where(window_mask, weight);
                    let sample = if let Some(laplacian_samples) = laplacian_samples.as_ref() {
                        laplacian_samples[level_idx].clone()
                    } else {
                        let [_, _, level_h, level_w] = level.image.shape().dims::<4>();
                        let level_grid = make_grid(&fx, &fy, level_w, level_h);
                        grid_sample_2d_bilinear::<B>(level.image.clone(), level_grid)
                    };
                    color = color + sample * weight.clone().unsqueeze_dim::<4>(1);
                    weight_sum = weight_sum + weight;
                }
                let weight_sum = weight_sum.clamp_min(SACCADE_EPS);
                let sample = color / weight_sum.unsqueeze_dim::<4>(1);
                accum = accum + sample;
            }
        }
        accum.mul_scalar(1.0 / subsamples as f32)
    }

    fn foveated_patch_sample_subpatch(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        subpatch_size: usize,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        let device = base_grid.device();
        let [batch, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        if patch_h == 0 || patch_w == 0 {
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        }
        let mut tile = subpatch_size.min(patch_h).min(patch_w);
        while tile > 1 && (patch_h % tile != 0 || patch_w % tile != 0) {
            tile -= 1;
        }
        if tile >= patch_h && tile >= patch_w {
            return self.foveated_patch_sample_sequential(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            );
        }

        let mut rows = Vec::new();
        let mut y = 0;
        while y < patch_h {
            let y_end = (y + tile).min(patch_h);
            let mut row_tiles = Vec::new();
            let mut x = 0;
            while x < patch_w {
                let x_end = (x + tile).min(patch_w);
                let tile_grid = base_grid
                    .clone()
                    .slice_dim(1, y..y_end)
                    .slice_dim(2, x..x_end);
                let tile_patch = self.foveated_patch_sample_sequential(
                    levels,
                    tile_grid,
                    center_x.clone(),
                    center_y.clone(),
                    sigma_px.clone(),
                    radius_px.clone(),
                    lod_sigma.clone(),
                    laplacian_images,
                    full_patch_h,
                );
                row_tiles.push(tile_patch);
                x += tile;
            }
            let row = Tensor::cat(row_tiles, 3);
            rows.push(row);
            y += tile;
        }
        Tensor::cat(rows, 2)
    }

    fn lod_sigma_from_sigma(&self, sigma: Tensor<B, 2>) -> Tensor<B, 2> {
        let range = (SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN).max(SACCADE_EPS);
        let t = sigma
            .sub_scalar(SACCADE_SIGMA_MIN)
            .div_scalar(range)
            .clamp_min(0.0)
            .clamp_max(1.0);
        let log2 = t
            .mul_scalar(SACCADE_LOD_LOG2_MAX - SACCADE_LOD_LOG2_MIN)
            .add_scalar(SACCADE_LOD_LOG2_MIN);
        (log2.mul_scalar(SACCADE_LN_2)).exp()
    }

    fn mip_weighted_sum(
        &self,
        levels: &[Tensor<B, 3>],
        weights: &[Tensor<B, 3>],
    ) -> Tensor<B, 3> {
        let device = if let Some(weight) = weights.first() {
            weight.device()
        } else if let Some(level) = levels.first() {
            level.device()
        } else {
            return Tensor::<B, 3>::zeros([0, 0, 0], &B::Device::default());
        };
        let Some(weight) = weights.first() else {
            return Tensor::<B, 3>::zeros([0, 0, 0], &device);
        };
        let [batch, traj_tokens, _] = weight.shape().dims::<3>();
        let embed_dim = levels
            .first()
            .map(|tokens| tokens.shape().dims::<3>()[2])
            .unwrap_or(0);
        let mut context = Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim.max(1)], &device);
        for (tokens, weights) in levels.iter().zip(weights.iter()) {
            context = context + self.weighted_sum_tokens(weights.clone(), tokens.clone());
        }
        context
    }

    fn weighted_sum_tokens(
        &self,
        weights: Tensor<B, 3>,
        tokens: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = weights.device();
        let [batch, out_tokens, in_tokens] = weights.shape().dims::<3>();
        let dim = tokens.shape().dims::<3>()[2];
        if batch == 0 || out_tokens == 0 || in_tokens == 0 || dim == 0 {
            return Tensor::<B, 3>::zeros(
                [batch, out_tokens.max(1), dim.max(1)],
                &device,
            );
        }
        let weights = weights.unsqueeze_dim::<4>(3);
        let tokens = tokens.unsqueeze_dim::<4>(1);
        (weights * tokens).sum_dim(2).squeeze_dim::<3>(2)
    }

    #[cfg(test)]
    fn apply_mip_residual(
        &self,
        state_levels: &mut [Tensor<B, 3>],
        weights: &[Tensor<B, 3>],
        residual: Tensor<B, 3>,
    ) {
        for (state, weights) in state_levels.iter_mut().zip(weights.iter()) {
            let update = self
                .weighted_sum_tokens(weights.clone().swap_dims(1, 2), residual.clone());
            *state = state.clone() + update;
        }
    }
}

/// GPU foveation sampler that reuses the saccade pipeline's mip + patch sampling.
pub struct SaccadeFoveationSampler<B: BackendTrait> {
    saccade: VisionSaccadeModel<B>,
    base_grid: Tensor<B, 4>,
    patch_size: usize,
    levels: Vec<SaccadeMipLevel<B>>,
    laplacian: Option<SaccadeLaplacianImages<B>>,
}

impl<B: BackendTrait> SaccadeFoveationSampler<B> {
    pub fn new(
        vision: VisionDragonHatchlingConfig,
        saccade: VisionSaccadeConfig,
        device: &B::Device,
    ) -> Self {
        let model = VisionDragonHatchling::<B>::new(vision.clone(), device);
        let recon_patch_dim = vision.patch_size * vision.patch_size * vision.in_channels;
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 1,
            backprop_steps: 1,
        };
        let saccade =
            VisionSaccadeModel::new(model, saccade, vision.embed_dim, rollout, recon_patch_dim, device);
        let base_grid = build_foveated_base_grid::<B>(vision.patch_size, device);
        Self {
            saccade,
            base_grid,
            patch_size: vision.patch_size,
            levels: Vec::new(),
            laplacian: None,
        }
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn mip_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn update_image(&mut self, images: Tensor<B, 4>) {
        self.levels = build_sampling_pyramid::<B>(images, self.saccade.config.mip_levels);
        self.laplacian = if matches!(self.saccade.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            self.saccade.build_laplacian_images(&self.levels)
        } else {
            None
        };
    }

    pub fn sample_patch(
        &self,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        self.saccade.foveated_patch_image(
            &self.levels,
            &self.base_grid,
            mean,
            sigma,
            self.laplacian.as_ref(),
        )
    }

    pub fn sample_patch_with_radius(
        &self,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        radius: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        self.saccade.foveated_patch_image_with_radius(
            &self.levels,
            &self.base_grid,
            mean,
            sigma,
            radius,
            self.laplacian.as_ref(),
        )
    }
}

fn build_sampling_pyramid<B: BackendTrait>(
    images: Tensor<B, 4>,
    mip_levels: usize,
) -> Vec<SaccadeMipLevel<B>> {
    let max_levels = mip_levels.max(1);
    let device = images.device();
    let [batch, _channels, height, width] = images.shape().dims::<4>();
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let mut levels = Vec::with_capacity(max_levels);
    let mut current = images;
    for level in 0..max_levels {
        let tokens = Tensor::<B, 3>::zeros([batch.max(1), 1, 1], &device);
        levels.push(SaccadeMipLevel {
            tokens,
            grid: PatchGrid { height: 1, width: 1 },
            image: current.clone(),
        });
        if level + 1 == max_levels {
            break;
        }
        let Some(next) = downsample_image(current) else {
            break;
        };
        current = next;
    }
    levels
}

impl<B: AutodiffBackend> TrainStep<SequenceBatch<B>, LanguageModelTrainItem<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let logits = if fast_train_enabled() {
            self.forward_fast(batch.inputs)
        } else {
            self.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        let grads = loss.backward();

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep<SequenceBatch<B>, LanguageModelOutput<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let logits = if fast_train_enabled() {
            self.forward_fast(batch.inputs)
        } else {
            self.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>>
    for VisionDistillModel<B>
{
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let (teacher_patch, teacher_cls) = if let Some(teacher) = &self.teacher {
            let output = teacher.forward(images.clone(), None);
            (output.x_norm_patchtokens, output.x_norm_clstoken)
        } else {
            let teacher_patch = teacher_patch.expect("teacher patch features required");
            let teacher_cls = teacher_cls.expect("teacher cls features required");
            (teacher_patch, teacher_cls)
        };

        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let output = self
            .model
            .forward_images_steps_rollout(images, rollout_steps, backprop_steps);
        let loss = vision_distillation_loss(
            output.patch_tokens,
            teacher_patch,
            output.cls_token,
            teacher_cls,
            &self.loss,
        );
        let zero = Tensor::<B, 1>::zeros([1], &loss.device());
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                loss,
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ),
        )
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionDistillModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let (teacher_patch, teacher_cls) = if let Some(teacher) = &self.teacher {
            let output = teacher.forward(images.clone(), None);
            (output.x_norm_patchtokens, output.x_norm_clstoken)
        } else {
            let teacher_patch = teacher_patch.expect("teacher patch features required");
            let teacher_cls = teacher_cls.expect("teacher cls features required");
            (teacher_patch, teacher_cls)
        };

        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let output = self
            .model
            .forward_images_steps_rollout(images, self.rollout.max_steps, backprop_steps);
        let loss = vision_distillation_loss(
            output.patch_tokens,
            teacher_patch,
            output.cls_token,
            teacher_cls,
            &self.loss,
        );
        let zero = Tensor::<B, 1>::zeros([1], &loss.device());
        VisionOutput::new(
            loss,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            None,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionLejepaModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = self.forward_losses(batch, rollout_steps, backprop_steps, true);
        let total_for_backprop = losses.total.clone() + losses.probe_loss.clone();
        let grads = total_for_backprop.backward();

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                losses.total,
                losses.inv,
                losses.sigreg,
                losses.recon,
                losses.probe_loss,
                losses.probe_acc,
            ),
        )
    }

    fn optimize<BB, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: burn::optim::Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        optim.step(lr, self, grads)
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionLejepaModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let losses = self.forward_losses(batch, self.rollout.max_steps, backprop_steps, false);

        VisionOutput::new(
            losses.total,
            losses.inv,
            losses.sigreg,
            losses.recon,
            losses.probe_loss,
            losses.probe_acc,
            losses.artifacts,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionMaeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = self.forward_losses(batch, rollout_steps, backprop_steps, true, false);
        let grads = losses.total.clone().backward();
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                losses.total,
                zero.clone(),
                zero.clone(),
                losses.recon,
                zero.clone(),
                zero,
            ),
        )
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionMaeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let losses = self.forward_losses(batch, self.rollout.max_steps, backprop_steps, false, true);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            zero.clone(),
            zero.clone(),
            losses.recon,
            zero.clone(),
            zero,
            losses.artifacts,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionSaccadeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = self.forward_losses(batch, rollout_steps, backprop_steps, true, false);
        let grads = losses.total.clone().backward();
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                losses.total,
                losses.inv,
                losses.sigreg,
                losses.recon,
                zero.clone(),
                zero,
            ),
        )
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionSaccadeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let losses = self.forward_losses(batch, self.rollout.max_steps, backprop_steps, false, true);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            losses.inv,
            losses.sigreg,
            losses.recon,
            zero.clone(),
            zero,
            losses.artifacts,
        )
    }
}

#[cfg(feature = "cli")]
pub fn run_cli() -> Result<()> {
    let args = Cli::parse();

    if matches!(args.command, Some(Command::Vision)) {
        let mut config_paths = vec![PathBuf::from("config/vision_base.toml")];
        config_paths.extend(args.train.config.clone());
        let config = load_vision_training_config(&config_paths)?;
        return match args.train.backend {
            BackendArg::Wgpu => train_vision_backend::<Autodiff<Wgpu<f32>>, _>(
                &config,
                "wgpu",
                init_runtime,
            ),
            BackendArg::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    train_vision_backend::<Autodiff<Cuda<f32>>, _>(&config, "cuda", |_| {})
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow!(
                        "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                    ))
                }
            }
        };
    }

    let mut config_paths = vec![PathBuf::from("config/base.toml")];
    config_paths.extend(args.train.config.clone());
    let config = load_training_config(&config_paths)?;
    FAST_TRAIN.store(config.training.fast_train, Ordering::Relaxed);

    if matches!(args.command, Some(Command::BuildVocab)) {
        build_vocab_only(&config)?;
        return Ok(());
    }

    let dataset = prepare_dataset(&config.dataset, &config.training)?;

    match args.train.backend {
        BackendArg::Wgpu => train_backend::<Autodiff<Wgpu<f32>>, _>(
            &config,
            Arc::clone(&dataset),
            "wgpu",
            init_runtime,
        ),
        BackendArg::Cuda => {
            #[cfg(feature = "cuda")]
            {
                train_backend::<Autodiff<Cuda<f32>>, _>(&config, dataset, "cuda", |_| {})
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(anyhow!(
                    "cuda backend selected but this build lacks `cuda` feature; rebuild with `--features cuda`"
                ))
            }
        }
    }
}

fn train_backend<B, Init>(
    config: &TrainingConfig,
    dataset: Arc<Dataset>,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;

    let mut model_config = build_model_config(&config.model, training.block_size);
    let tokenizer = dataset.tokenizer();
    model_config.vocab_size = tokenizer.len();

    let steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Train);
    let schedule = resolve_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "train schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );
    let train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>> =
        Arc::new(RandomDataLoader::<B>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            steps_per_epoch,
            Some(total_steps),
        ));

    let val_steps_per_epoch = dataset.steps_per_epoch(DatasetSplit::Val);
    let desired_valid_steps = usize::max(1, total_steps / training.log_frequency.max(1));
    let valid_steps = desired_valid_steps.min(val_steps_per_epoch).max(1);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>> =
        Arc::new(RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&dataset),
            DatasetSplit::Val,
            &valid_device,
            valid_steps,
            None,
        ));

    let mut model = Some(BDH::<B>::new(model_config.clone(), &device));
    let mut optim = Some(
        AdamWConfig::new()
            .with_weight_decay(optimizer_cfg.weight_decay)
            .init::<B, BDH<B>>(),
    );
    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &model_config)?;

    let run_root = PathBuf::from("runs");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    write_run_config(config, &run_dir, &run_name)?;
    info!("run name: {run_name}");
    let context = TrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        training,
        model_config: &model_config,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };
    let _model = match scheduler {
        ResolvedLrScheduler::Constant(lr) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            lr,
        )?,
        ResolvedLrScheduler::Cosine(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Linear(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Exponential(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Step(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
        ResolvedLrScheduler::Noam(scheduler) => train_with_scheduler(
            &context,
            model.take().expect("model initialized"),
            optim.take().expect("optimizer initialized"),
            scheduler,
        )?,
    };

    info!("Training complete on {backend_name}");

    Ok(())
}

fn train_vision_backend<B, Init>(
    config: &VisionTrainingConfig,
    backend_name: &str,
    init_backend: Init,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    let optimizer_cfg = &config.optimizer;
    if training.batch_size == 0 {
        return Err(anyhow!("vision training batch_size must be > 0"));
    }

    let vision_config = config.vision.build();
    if vision_config.patch_size == 0 {
        return Err(anyhow!("vision.patch_size must be > 0"));
    }
    if !vision_config
        .image_size
        .is_multiple_of(vision_config.patch_size)
    {
        return Err(anyhow!(
            "vision.image_size must be divisible by vision.patch_size ({} % {} != 0)",
            vision_config.image_size,
            vision_config.patch_size
        ));
    }
    if config.augment.image_size != vision_config.image_size {
        return Err(anyhow!(
            "augment.image_size ({}) must match vision.image_size ({})",
            config.augment.image_size,
            vision_config.image_size
        ));
    }
    let rollout = resolve_vision_rollout(training, vision_config.steps)?;
    info!(
        "vision rollout steps: min={}, max={}, backprop={}",
        rollout.min_steps, rollout.max_steps, rollout.backprop_steps
    );

    maybe_download_vision_dataset(&config.dataset)?;

    let grid = vision_config.image_size / vision_config.patch_size;
    let student_patch_tokens = grid * grid;

    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_aug = ImageNetAugmentations::new(
        ImageNetSplit::Train,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let val_aug = ImageNetAugmentations::new(
        ImageNetSplit::Val,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );

    let train_root = config
        .dataset
        .imagenet_root
        .join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);

    enum VisionMode<B: BackendTrait> {
        Distill {
            loss: VisionDistillationLossConfig,
            teacher: Option<Box<DinoVisionTransformer<B>>>,
        },
        Lejepa {
            config: VisionLejepaConfig,
        },
        Mae {
            config: VisionMaeConfig,
        },
        Saccade {
            config: VisionSaccadeConfig,
        },
    }

    let (train_dataset, val_dataset, mode) = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => {
            let (train_dataset, val_dataset, teacher) = match &distill.teacher {
                VisionTeacherConfig::Features(teacher) => {
                    if teacher.feature_dim != vision_config.projection_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                            teacher.feature_dim,
                            vision_config.projection_dim
                        ));
                    }
                    if let Some(tokens) = teacher
                        .patch_tokens
                        .filter(|tokens| *tokens != student_patch_tokens)
                    {
                        return Err(anyhow!(
                            "teacher.patch_tokens ({}) must match (image_size/patch_size)^2 ({})",
                            tokens,
                            student_patch_tokens
                        ));
                    }
                    let teacher_tokens = teacher.patch_tokens.unwrap_or(student_patch_tokens);

                    let mut train_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                        root: train_root,
                        split: ImageNetSplit::Train,
                        max_records: config.dataset.max_records,
                        augmentations: train_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                    })?;
                    let train_records = train_dataset.len();
                    let train_teacher = Arc::new(DinoFeatureStore::new(
                        &teacher.train_cls_path,
                        &teacher.train_patch_path,
                        teacher.feature_dim,
                        teacher_tokens,
                        Some(train_records),
                    )?);
                    train_dataset = train_dataset.with_teacher(Arc::clone(&train_teacher));
                    let train_dataset = Arc::new(train_dataset);

                    let mut val_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                        root: val_root,
                        split: ImageNetSplit::Val,
                        max_records: config.dataset.max_records,
                        augmentations: val_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                    })?;
                    let val_records = val_dataset.len();
                    let val_teacher = Arc::new(DinoFeatureStore::new(
                        &teacher.val_cls_path,
                        &teacher.val_patch_path,
                        teacher.feature_dim,
                        teacher_tokens,
                        Some(val_records),
                    )?);
                    val_dataset = val_dataset.with_teacher(Arc::clone(&val_teacher));
                    let val_dataset = Arc::new(val_dataset);
                    (train_dataset, val_dataset, None)
                }
                VisionTeacherConfig::Model(teacher) => {
                    let image_size = teacher
                        .image_size
                        .unwrap_or(vision_config.image_size);
                    let patch_size = teacher
                        .patch_size
                        .unwrap_or(vision_config.patch_size);
                    if patch_size == 0 {
                        return Err(anyhow!("teacher.patch_size must be > 0"));
                    }
                    if !image_size.is_multiple_of(patch_size) {
                        return Err(anyhow!(
                            "teacher image_size must be divisible by patch_size ({} % {} != 0)",
                            image_size,
                            patch_size
                        ));
                    }
                    let teacher_grid = image_size / patch_size;
                    let teacher_tokens = teacher_grid * teacher_grid;
                    if teacher_tokens != student_patch_tokens {
                        return Err(anyhow!(
                            "teacher patch tokens ({}) must match student tokens ({})",
                            teacher_tokens,
                            student_patch_tokens
                        ));
                    }
                    if let Some(tokens) = teacher
                        .patch_tokens
                        .filter(|tokens| *tokens != teacher_tokens)
                    {
                        return Err(anyhow!(
                            "teacher.patch_tokens ({}) must match (image_size/patch_size)^2 ({})",
                            tokens,
                            teacher_tokens
                        ));
                    }

                    let feature_dim = teacher
                        .feature_dim
                        .unwrap_or_else(|| teacher_variant_dim(teacher.variant));
                    if feature_dim != vision_config.projection_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                            feature_dim,
                            vision_config.projection_dim
                        ));
                    }

                    let mut dino_config =
                        build_dino_config(teacher.variant, image_size, patch_size);
                    if teacher.register_tokens > 0 {
                        dino_config = dino_config.with_register_tokens(teacher.register_tokens);
                    }
                    if dino_config.embedding_dimension != feature_dim {
                        return Err(anyhow!(
                            "teacher.feature_dim ({}) must match DINO embedding dim ({})",
                            feature_dim,
                            dino_config.embedding_dimension
                        ));
                    }

                    let teacher_model = load_model_from_checkpoint::<B>(
                        &dino_config,
                        &teacher.checkpoint_path,
                        &device,
                    )
                    .map_err(|err| {
                        anyhow!(
                            "failed to load teacher checkpoint {}: {err}",
                            teacher.checkpoint_path.display()
                        )
                    })?
                    .no_grad();

                    let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                        root: train_root,
                        split: ImageNetSplit::Train,
                        max_records: config.dataset.max_records,
                        augmentations: train_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                    })?);
                    let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                        root: val_root,
                        split: ImageNetSplit::Val,
                        max_records: config.dataset.max_records,
                        augmentations: val_aug,
                        local_augmentations: None,
                        normalize,
                        teacher: None,
                        views: 1,
                        local_views: 0,
                        cache_decoded: config.dataset.cache_decoded,
                        cache_capacity: config.dataset.cache_capacity,
                    })?);

                    (train_dataset, val_dataset, Some(Box::new(teacher_model)))
                }
            };

            (
                train_dataset,
                val_dataset,
                VisionMode::Distill {
                    loss: distill.loss.clone(),
                    teacher,
                },
            )
        }
        VisionTrainingModeConfig::Lejepa(lejepa) => {
            let multi_crop = lejepa.global_views > 0 || lejepa.local_views > 0;
            let global_views = if multi_crop {
                lejepa.global_views.max(1)
            } else {
                lejepa.views.max(1)
            };
            let local_views = if multi_crop { lejepa.local_views } else { 0 };
            if global_views + local_views == 0 {
                return Err(anyhow!(
                    "lejepa must have at least one global or local view"
                ));
            }
            if local_views > 0 {
                if lejepa.local_image_size == 0 {
                    return Err(anyhow!("lejepa.local_image_size must be > 0"));
                }
                if !lejepa.local_image_size.is_multiple_of(vision_config.patch_size) {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be divisible by patch_size ({})",
                        lejepa.local_image_size,
                        vision_config.patch_size
                    ));
                }
                if lejepa.local_image_size > vision_config.image_size {
                    return Err(anyhow!(
                        "lejepa.local_image_size ({}) must be <= vision.image_size ({})",
                        lejepa.local_image_size,
                        vision_config.image_size
                    ));
                }
            }
            if lejepa.recon_weight < 0.0 {
                return Err(anyhow!("lejepa.recon_weight must be >= 0"));
            }
            if !(0.0..=1.0).contains(&lejepa.recon_mask_ratio) {
                return Err(anyhow!(
                    "lejepa.recon_mask_ratio must be in [0, 1] (got {})",
                    lejepa.recon_mask_ratio
                ));
            }
            let local_train_aug = if local_views > 0 {
                Some(ImageNetAugmentations::new(
                    ImageNetSplit::Train,
                    lejepa.local_image_size,
                    lejepa.local_image_size,
                    lejepa.local_min_scale,
                    lejepa.local_max_scale,
                    config.augment.min_aspect_ratio,
                    config.augment.max_aspect_ratio,
                    config.augment.flip_prob,
                    config.augment.color_jitter_prob,
                    config.augment.brightness,
                    config.augment.contrast,
                    config.augment.saturation,
                    config.augment.hue,
                    config.augment.grayscale_prob,
                    config.augment.blur_prob,
                    config.augment.blur_sigma_min,
                    config.augment.blur_sigma_max,
                    config.augment.solarize_prob,
                    config.augment.solarize_threshold,
                ))
            } else {
                None
            };
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: local_train_aug.clone(),
                normalize,
                teacher: None,
                views: global_views,
                local_views,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: local_train_aug.clone(),
                normalize,
                teacher: None,
                views: global_views,
                local_views,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Lejepa {
                    config: lejepa.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Mae(mae) => {
            if !(0.0..=1.0).contains(&mae.mask_ratio) {
                return Err(anyhow!(
                    "mae.mask_ratio must be in [0, 1] (got {})",
                    mae.mask_ratio
                ));
            }
            if mae.recon_weight < 0.0 {
                return Err(anyhow!("mae.recon_weight must be >= 0"));
            }
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views: 1,
                local_views: 0,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views: 1,
                local_views: 0,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Mae {
                    config: mae.clone(),
                },
            )
        }
        VisionTrainingModeConfig::Saccade(saccade) => {
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
            }
            if saccade.inner_steps == 0 {
                return Err(anyhow!("saccade.inner_steps must be > 0"));
            }
            if !(0.0..=1.0).contains(&saccade.recon_mask_ratio) {
                return Err(anyhow!(
                    "saccade.recon_mask_ratio must be in [0, 1] (got {})",
                    saccade.recon_mask_ratio
                ));
            }
            if saccade.recon_weight < 0.0 {
                return Err(anyhow!("saccade.recon_weight must be >= 0"));
            }
            if !(0.0..=1.0).contains(&saccade.lambda) {
                return Err(anyhow!(
                    "saccade.lambda must be in [0, 1] (got {})",
                    saccade.lambda
                ));
            }
            if saccade.sigreg_knots == 0 {
                return Err(anyhow!("saccade.sigreg_knots must be > 0"));
            }
            if saccade.sigreg_t_max <= 0.0 {
                return Err(anyhow!("saccade.sigreg_t_max must be > 0"));
            }
            if saccade.sigreg_proj_dim == 0 {
                return Err(anyhow!("saccade.sigreg_proj_dim must be > 0"));
            }
            let train_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root,
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views: 1,
                local_views: 0,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);
            let val_dataset = Arc::new(ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root,
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: train_aug.clone(),
                local_augmentations: None,
                normalize,
                teacher: None,
                views: 1,
                local_views: 0,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
            })?);

            (
                train_dataset,
                val_dataset,
                VisionMode::Saccade {
                    config: saccade.clone(),
                },
            )
        }
    };

    let steps_per_epoch = train_dataset.steps_per_epoch(training.batch_size);
    let schedule = resolve_vision_train_schedule(training, steps_per_epoch)?;
    let steps_per_epoch = schedule.steps_per_epoch;
    let total_epochs = schedule.total_epochs;
    let total_steps = schedule.total_steps;

    info!(
        "vision schedule: steps_per_epoch={steps_per_epoch}, total_steps={total_steps}, epochs={total_epochs}, source={}",
        schedule.source.as_str()
    );

    let train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>> =
        Arc::new(ImageNetDataLoader::<B>::new(
            Arc::clone(&train_dataset),
            training.batch_size,
            &device,
            steps_per_epoch,
            Some(total_steps),
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
        ));

    let val_steps_per_epoch = val_dataset.steps_per_epoch(training.batch_size);
    let desired_valid_steps = usize::max(1, total_steps / training.log_frequency.max(1));
    let valid_steps = desired_valid_steps.min(val_steps_per_epoch).max(1);

    let valid_device = device.clone();
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>> =
        Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&val_dataset),
            training.batch_size,
            &valid_device,
            valid_steps,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
        ));

    let scheduler_iters = match schedule.source {
        ScheduleSource::Epochs => Some(total_steps),
        ScheduleSource::MaxIters => None,
    };
    let scheduler =
        resolve_vision_lr_scheduler(optimizer_cfg, total_steps, scheduler_iters, &vision_config)?;

    let run_root = PathBuf::from("runs").join("vision");
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    info!("vision run name: {run_name}");
    let context = VisionTrainEnvironment {
        run_dir: &run_dir,
        run_name: &run_name,
        backend_name,
        device: &device,
        train_loader,
        valid_loader,
        epochs: total_epochs,
    };

    match mode {
        VisionMode::Distill { loss, teacher } => {
            let model = VisionDragonHatchling::<B>::new(vision_config.clone(), &device);
            let teacher = teacher.map(|teacher| *teacher);
            let mut model = Some(VisionDistillModel::new(model, loss, teacher, rollout));
            let mut optim = Some(
                AdamWConfig::new()
                    .with_weight_decay(optimizer_cfg.weight_decay)
                    .init::<B, VisionDistillModel<B>>(),
            );
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    None,
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    None,
                )?,
            }
        }
        VisionMode::Lejepa { config: lejepa } => {
            let model = VisionDragonHatchling::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionLejepaModel::new(
                model,
                lejepa,
                vision_config.embed_dim,
                train_dataset.num_classes(),
                rollout,
                recon_patch_dim,
                &device,
            ));
            let mut optim = Some(
                AdamWConfig::new()
                    .with_weight_decay(optimizer_cfg.weight_decay)
                    .init::<B, VisionLejepaModel<B>>(),
            );
            let diagnostics = Some(VisionDiagnostics {
                metric_prefix: "lejepa".to_string(),
                inv: true,
                sigreg: true,
                recon: model
                    .as_ref()
                    .expect("model")
                    .config
                    .recon_weight
                    > 0.0,
                probe: true,
                artifact_every: model.as_ref().expect("model").config.artifact_every,
                artifact_output: model.as_ref().expect("model").config.artifact_output,
                artifact_overwrite: model.as_ref().expect("model").config.artifact_overwrite,
                artifact_max_images: model.as_ref().expect("model").config.artifact_max_images,
                artifact_fps: model.as_ref().expect("model").config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Mae { config: mae } => {
            let model = VisionDragonHatchling::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionMaeModel::new(
                model,
                mae,
                vision_config.embed_dim,
                rollout,
                recon_patch_dim,
                &device,
            ));
            let mut optim = Some(
                AdamWConfig::new()
                    .with_weight_decay(optimizer_cfg.weight_decay)
                    .init::<B, VisionMaeModel<B>>(),
            );
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "mae".to_string(),
                inv: false,
                sigreg: false,
                recon: model_ref.config.recon_weight > 0.0,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
        VisionMode::Saccade { config: saccade } => {
            let model = VisionDragonHatchling::<B>::new(vision_config.clone(), &device);
            let recon_patch_dim = vision_config
                .patch_size
                .saturating_mul(vision_config.patch_size)
                .saturating_mul(vision_config.in_channels);
            let mut model = Some(VisionSaccadeModel::new(
                model,
                saccade,
                vision_config.embed_dim,
                rollout,
                recon_patch_dim,
                &device,
            ));
            let mut optim = Some(
                AdamWConfig::new()
                    .with_weight_decay(optimizer_cfg.weight_decay)
                    .init::<B, VisionSaccadeModel<B>>(),
            );
            let diagnostics = model.as_ref().map(|model_ref| VisionDiagnostics {
                metric_prefix: "saccade".to_string(),
                inv: true,
                sigreg: true,
                recon: true,
                probe: false,
                artifact_every: model_ref.config.artifact_every,
                artifact_output: model_ref.config.artifact_output,
                artifact_overwrite: model_ref.config.artifact_overwrite,
                artifact_max_images: model_ref.config.artifact_max_images,
                artifact_fps: model_ref.config.artifact_fps,
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    diagnostics.clone(),
                )?,
            }
        }
    }

    info!("Vision training complete on {backend_name}");

    Ok(())
}

fn maybe_download_vision_dataset(config: &VisionDatasetConfig) -> Result<()> {
    let Some(download) = &config.download else {
        return Ok(());
    };

    let train_root = config.imagenet_root.join(&config.train_dir);
    let val_root = config.imagenet_root.join(&config.val_dir);
    if vision_split_has_images(&train_root)? && vision_split_has_images(&val_root)? {
        return Ok(());
    }

    match download {
        VisionDatasetDownloadConfig::Imagenette { variant } => {
            download_imagenette(config, *variant)
        }
    }
}

fn vision_split_has_images(root: &Path) -> Result<bool> {
    if !root.is_dir() {
        return Ok(false);
    }
    for entry in fs::read_dir(root)
        .with_context(|| format!("failed to read {}", root.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        for file in fs::read_dir(&path)
            .with_context(|| format!("failed to read {}", path.display()))?
        {
            let file = file?;
            let path = file.path();
            if path.is_file() && is_image_file(&path) {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => matches!(
            ext.to_ascii_lowercase().as_str(),
            "jpg" | "jpeg" | "png"
        ),
        None => false,
    }
}

fn download_imagenette(config: &VisionDatasetConfig, variant: ImagenetteVariant) -> Result<()> {
    if config.train_dir != "train" || config.val_dir != "val" {
        return Err(anyhow!(
            "imagenette download expects train_dir='train' and val_dir='val'"
        ));
    }

    let root = &config.imagenet_root;
    let train_root = root.join(&config.train_dir);
    let val_root = root.join(&config.val_dir);
    if vision_split_has_images(&train_root)? && vision_split_has_images(&val_root)? {
        return Ok(());
    }
    if root.exists() {
        let has_entries = fs::read_dir(root)
            .map(|mut iter| iter.next().is_some())
            .unwrap_or(false);
        if has_entries {
            return Err(anyhow!(
                "imagenet_root {} exists but doesn't look like imagenette; move it or disable download",
                root.display()
            ));
        }
    }

    if let Some(parent) = root.parent() {
        fs::create_dir_all(parent)?;
    }

    let cache_root = root.parent().unwrap_or_else(|| Path::new("."));
    let folder = imagenette_folder_name(variant);
    let cache_dir = cache_root.join(".vision_cache").join(folder);
    fs::create_dir_all(&cache_dir)?;

    let archive_path = cache_dir.join(format!("{folder}.tgz"));
    if !archive_path.is_file() {
        download_file(imagenette_url(variant), &archive_path)?;
    }

    let extract_dir = cache_dir.join("extract");
    if extract_dir.exists() {
        fs::remove_dir_all(&extract_dir)?;
    }
    fs::create_dir_all(&extract_dir)?;

    let archive_file = fs::File::open(&archive_path)
        .with_context(|| format!("failed to open {}", archive_path.display()))?;
    let decoder = flate2::read::GzDecoder::new(archive_file);
    let mut archive = tar::Archive::new(decoder);
    archive
        .unpack(&extract_dir)
        .with_context(|| format!("failed to unpack {}", archive_path.display()))?;

    let candidate = extract_dir.join(folder);
    let source_dir = if candidate.is_dir() {
        candidate
    } else if extract_dir.join(&config.train_dir).is_dir() {
        extract_dir.clone()
    } else {
        return Err(anyhow!(
            "unexpected imagenette archive layout under {}",
            extract_dir.display()
        ));
    };

    if root.exists() {
        let has_entries = fs::read_dir(root)
            .map(|mut iter| iter.next().is_some())
            .unwrap_or(false);
        if has_entries {
            return Err(anyhow!(
                "imagenet_root {} exists but is not empty",
                root.display()
            ));
        }
        fs::remove_dir_all(root)?;
    }

    if let Err(err) = fs::rename(&source_dir, root) {
        copy_dir_all(&source_dir, root).map_err(|copy_err| {
            anyhow!(
                "failed to move imagenette data into {}: {err}; copy error: {copy_err}",
                root.display()
            )
        })?;
    }

    Ok(())
}

fn imagenette_url(variant: ImagenetteVariant) -> &'static str {
    match variant {
        ImagenetteVariant::Imagenette2_160 => {
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
        }
        ImagenetteVariant::Imagenette2_320 => {
            "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        }
    }
}

fn imagenette_folder_name(variant: ImagenetteVariant) -> &'static str {
    match variant {
        ImagenetteVariant::Imagenette2_160 => "imagenette2-160",
        ImagenetteVariant::Imagenette2_320 => "imagenette2-320",
    }
}

fn download_file(url: &str, dest: &Path) -> Result<()> {
    let parent = dest
        .parent()
        .ok_or_else(|| anyhow!("download destination missing parent"))?;
    fs::create_dir_all(parent)?;

    info!("Downloading {url}");
    let response = ureq::get(url)
        .call()
        .map_err(|err| anyhow!("failed to download {url}: {err}"))?;
    let mut reader = response.into_reader();
    let tmp_path = dest.with_extension("tmp");
    let mut file =
        fs::File::create(&tmp_path).with_context(|| format!("failed to create {}", tmp_path.display()))?;
    io::copy(&mut reader, &mut file)
        .with_context(|| format!("failed to write {}", tmp_path.display()))?;
    fs::rename(&tmp_path, dest)
        .with_context(|| format!("failed to rename {} to {}", tmp_path.display(), dest.display()))?;
    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src).with_context(|| format!("failed to read {}", src.display()))? {
        let entry = entry?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());
        if src_path.is_dir() {
            copy_dir_all(&src_path, &dst_path)?;
        } else {
            fs::copy(&src_path, &dst_path).with_context(|| {
                format!(
                    "failed to copy {} to {}",
                    src_path.display(),
                    dst_path.display()
                )
            })?;
        }
    }
    Ok(())
}

enum ResolvedLrScheduler {
    Constant(LearningRate),
    Cosine(CosineAnnealingLrScheduler),
    Linear(LinearLrScheduler),
    Exponential(ExponentialLrScheduler),
    Step(StepLrScheduler),
    Noam(NoamLrScheduler),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScheduleSource {
    Epochs,
    MaxIters,
}

impl ScheduleSource {
    fn as_str(self) -> &'static str {
        match self {
            ScheduleSource::Epochs => "epochs",
            ScheduleSource::MaxIters => "max_iters",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TrainSchedule {
    steps_per_epoch: usize,
    total_steps: usize,
    total_epochs: usize,
    source: ScheduleSource,
}

struct TrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    run_dir: &'a Path,
    run_name: &'a str,
    backend_name: &'a str,
    training: &'a TrainingHyperparameters,
    model_config: &'a BDHConfig,
    device: &'a B::Device,
    train_loader: Arc<dyn DataLoader<B, SequenceBatch<B>>>,
    valid_loader: Arc<dyn DataLoader<ValidBackend<B>, SequenceBatch<ValidBackend<B>>>>,
    epochs: usize,
}

struct VisionTrainEnvironment<'a, B>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    run_dir: &'a Path,
    run_name: &'a str,
    backend_name: &'a str,
    device: &'a B::Device,
    train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>>,
    valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>>,
    epochs: usize,
}

#[derive(Clone, Copy, Debug, Module)]
struct VisionRollout {
    min_steps: usize,
    max_steps: usize,
    backprop_steps: usize,
}

impl VisionRollout {
    fn sample_steps(&self) -> usize {
        if self.min_steps >= self.max_steps {
            self.max_steps
        } else {
            thread_rng().gen_range(self.min_steps..=self.max_steps)
        }
    }

    fn backprop_steps(&self, steps: usize) -> usize {
        self.backprop_steps.min(steps).max(1)
    }
}

#[derive(Clone)]
struct VisionDiagnostics {
    metric_prefix: String,
    inv: bool,
    sigreg: bool,
    recon: bool,
    probe: bool,
    artifact_every: usize,
    artifact_output: VisionArtifactOutputMode,
    artifact_overwrite: bool,
    artifact_max_images: usize,
    artifact_fps: u32,
    normalize_mean: [f32; 3],
    normalize_std: [f32; 3],
}

fn train_with_scheduler<B, S>(
    env: &TrainEnvironment<'_, B>,
    model: BDH<B>,
    optimizer: OptimizerAdaptor<AdamW, BDH<B>, B>,
    scheduler: S,
) -> Result<BDH<ValidBackend<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let builder = LearnerBuilder::new(env.run_dir)
        .num_epochs(env.epochs)
        .learning_strategy(LearningStrategy::SingleDevice(env.device.clone()))
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name))
        .summary();

    info!("run name: {}", env.run_name);

    let learner = builder.build(model, optimizer, scheduler);

    let TrainingResult { model, .. } = learner.fit(
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    );

    log_theoretical_profile(
        env.model_config,
        env.training.batch_size,
        env.training.block_size,
        env.backend_name,
    );

    Ok(model)
}

fn train_vision_with_scheduler<B, S, M>(
    env: &VisionTrainEnvironment<'_, B>,
    model: M,
    optimizer: OptimizerAdaptor<AdamW, M, B>,
    scheduler: S,
    vision_diagnostics: Option<VisionDiagnostics>,
) -> Result<()>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    M: AutodiffModule<B>
        + TrainStep<ImageNetBatch<B>, VisionTrainItem<B>>
        + core::fmt::Display
        + Clone
        + 'static,
    M::InnerModule: ValidStep<ImageNetBatch<ValidBackend<B>>, VisionOutput<ValidBackend<B>>>,
    S: LrScheduler + 'static,
{
    fs::create_dir_all(env.run_dir)?;

    let mut builder = LearnerBuilder::new(env.run_dir)
        .num_epochs(env.epochs)
        .learning_strategy(LearningStrategy::SingleDevice(env.device.clone()))
        .with_file_checkpointer(BinFileRecorder::<FullPrecisionSettings>::new())
        .metric_train_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_valid_numeric(LossMetric::<ValidBackend<B>>::new())
        .metric_train_numeric(LearningRateMetric::new())
        .metric_train(DeviceMetric::new("device", env.backend_name))
        .metric_valid(DeviceMetric::new("device", env.backend_name))
        .summary();

    info!("vision run name: {}", env.run_name);

    if let Some(diagnostics) = &vision_diagnostics {
        let prefix = diagnostics.metric_prefix.as_str();
        if diagnostics.inv {
            let name = format!("{prefix}_inv_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                );
        }
        if diagnostics.sigreg {
            let name = format!("{prefix}_sigreg_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, SigRegLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, SigRegLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                );
        }
        if diagnostics.recon {
            let name = format!("{prefix}_recon_loss");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, ReconLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, ReconLossInput<ValidBackend<B>>>::new(
                        name.as_str(),
                    ),
                );
        }
        if diagnostics.probe {
            let probe_loss = format!("{prefix}_probe_loss");
            let probe_acc = format!("{prefix}_probe_acc");
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, ProbeLossInput<ValidBackend<B>>>::new(
                        probe_loss.as_str(),
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, ProbeLossInput<ValidBackend<B>>>::new(
                        probe_loss.as_str(),
                    ),
                )
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, ProbeAccInput<ValidBackend<B>>>::new(
                        probe_acc.as_str(),
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, ProbeAccInput<ValidBackend<B>>>::new(
                        probe_acc.as_str(),
                    ),
                );
        }

        if diagnostics.artifact_every > 0 {
            let artifact_dir = env.run_dir.join("artifacts");
            builder = builder.metric_valid(VisionArtifactMetric::<ValidBackend<B>>::new(
                artifact_dir,
                diagnostics.artifact_every,
                diagnostics.artifact_output,
                diagnostics.artifact_max_images,
                diagnostics.artifact_fps,
                diagnostics.normalize_mean,
                diagnostics.normalize_std,
                diagnostics.artifact_overwrite,
            ));
        }
    }

    let learner = builder.build(model, optimizer, scheduler);

    let _result = learner.fit(
        Arc::clone(&env.train_loader),
        Arc::clone(&env.valid_loader),
    );

    Ok(())
}

fn resolve_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &BDHConfig,
) -> Result<ResolvedLrScheduler> {
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = total_steps.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => ResolvedLrScheduler::Constant(base_lr),
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = CosineAnnealingLrSchedulerConfig::new(
                init_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .with_min_lr(min_lr.unwrap_or(0.0))
            .init()
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(model_config.n_embd).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

fn resolve_vision_lr_scheduler(
    optimizer_cfg: &OptimizerConfig,
    total_steps: usize,
    override_num_iters: Option<usize>,
    model_config: &VisionDragonHatchlingConfig,
) -> Result<ResolvedLrScheduler> {
    let base_lr = optimizer_cfg.learning_rate;
    let fallback_iters = total_steps.max(1);

    let schedule = match &optimizer_cfg.lr_schedule {
        None => ResolvedLrScheduler::Constant(base_lr),
        Some(LearningRateScheduleConfig::Constant { initial_lr }) => {
            ResolvedLrScheduler::Constant(initial_lr.unwrap_or(base_lr))
        }
        Some(LearningRateScheduleConfig::Cosine {
            initial_lr,
            min_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = CosineAnnealingLrSchedulerConfig::new(
                init_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .with_min_lr(min_lr.unwrap_or(0.0))
            .init()
            .map_err(|err| anyhow!("failed to initialize cosine lr scheduler: {err}"))?;
            ResolvedLrScheduler::Cosine(scheduler)
        }
        Some(LearningRateScheduleConfig::Linear {
            initial_lr,
            final_lr,
            num_iters,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = LinearLrSchedulerConfig::new(
                init_lr,
                *final_lr,
                override_num_iters
                    .unwrap_or_else(|| num_iters.unwrap_or(fallback_iters))
                    .max(1),
            )
            .init()
            .map_err(|err| anyhow!("failed to initialize linear lr scheduler: {err}"))?;
            ResolvedLrScheduler::Linear(scheduler)
        }
        Some(LearningRateScheduleConfig::Exponential { initial_lr, gamma }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler = ExponentialLrSchedulerConfig::new(init_lr, *gamma)
                .init()
                .map_err(|err| anyhow!("failed to initialize exponential lr scheduler: {err}"))?;
            ResolvedLrScheduler::Exponential(scheduler)
        }
        Some(LearningRateScheduleConfig::Step {
            initial_lr,
            gamma,
            step_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let scheduler =
                StepLrSchedulerConfig::new(init_lr, step_size.unwrap_or(fallback_iters).max(1))
                    .with_gamma(*gamma)
                    .init()
                    .map_err(|err| anyhow!("failed to initialize step lr scheduler: {err}"))?;
            ResolvedLrScheduler::Step(scheduler)
        }
        Some(LearningRateScheduleConfig::Noam {
            initial_lr,
            warmup_steps,
            model_size,
        }) => {
            let init_lr = initial_lr.unwrap_or(base_lr);
            let mut config = NoamLrSchedulerConfig::new(init_lr);
            config = config.with_warmup_steps(warmup_steps.unwrap_or(fallback_iters).max(1));
            config = config.with_model_size(model_size.unwrap_or(model_config.embed_dim).max(1));
            let scheduler = config
                .init()
                .map_err(|err| anyhow!("failed to initialize noam lr scheduler: {err}"))?;
            ResolvedLrScheduler::Noam(scheduler)
        }
    };

    Ok(schedule)
}

fn resolve_train_schedule(
    training: &TrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    let steps_per_epoch = steps_per_epoch.max(1);
    match training.epochs {
        Some(epochs) => {
            let total_epochs = epochs.max(1);
            let total_steps = steps_per_epoch
                .checked_mul(total_epochs)
                .ok_or_else(|| {
                    anyhow!(
                        "training.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
                    )
                })?
                .max(1);
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::Epochs,
            })
        }
        None => {
            let total_steps = training.max_iters.max(1);
            let total_epochs = usize::max(1, total_steps.div_ceil(steps_per_epoch));
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::MaxIters,
            })
        }
    }
}

fn resolve_vision_train_schedule(
    training: &VisionTrainingHyperparameters,
    steps_per_epoch: usize,
) -> Result<TrainSchedule> {
    let steps_per_epoch = steps_per_epoch.max(1);
    match training.epochs {
        Some(epochs) => {
            let total_epochs = epochs.max(1);
            let total_steps = steps_per_epoch
                .checked_mul(total_epochs)
                .ok_or_else(|| {
                    anyhow!(
                        "vision training.epochs overflow: steps_per_epoch={steps_per_epoch}, epochs={total_epochs}"
                    )
                })?
                .max(1);
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::Epochs,
            })
        }
        None => {
            let total_steps = training.max_iters.max(1);
            let total_epochs = usize::max(1, total_steps.div_ceil(steps_per_epoch));
            Ok(TrainSchedule {
                steps_per_epoch,
                total_steps,
                total_epochs,
                source: ScheduleSource::MaxIters,
            })
        }
    }
}

fn resolve_vision_rollout(
    training: &VisionTrainingHyperparameters,
    max_steps: usize,
) -> Result<VisionRollout> {
    let max_steps = max_steps.max(1);
    let min_steps = training.rollout_min_steps.unwrap_or(max_steps);
    let max_steps_cfg = training.rollout_max_steps.unwrap_or(max_steps);
    let backprop_steps = training.rollout_backprop_steps.unwrap_or(max_steps_cfg);
    if min_steps == 0 || max_steps_cfg == 0 {
        return Err(anyhow!(
            "vision rollout steps must be > 0 (min={min_steps}, max={max_steps_cfg})"
        ));
    }
    if backprop_steps == 0 {
        return Err(anyhow!(
            "vision rollout_backprop_steps must be > 0 (value={backprop_steps})"
        ));
    }
    if min_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_min_steps ({min_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    if max_steps_cfg > max_steps {
        return Err(anyhow!(
            "vision rollout_max_steps ({max_steps_cfg}) exceeds vision.steps ({max_steps})"
        ));
    }
    if backprop_steps > max_steps_cfg {
        return Err(anyhow!(
            "vision rollout_backprop_steps ({backprop_steps}) must be <= rollout_max_steps ({max_steps_cfg})"
        ));
    }
    Ok(VisionRollout {
        min_steps,
        max_steps: max_steps_cfg,
        backprop_steps,
    })
}

struct CollectedViews<B: BackendTrait> {
    global: Vec<Tensor<B, 4>>,
    local: Vec<Tensor<B, 4>>,
    all: Vec<Tensor<B, 4>>,
}

impl<B: BackendTrait> CollectedViews<B> {
    fn artifact_views(&self) -> Vec<Tensor<B, 4>> {
        if !self.global.is_empty() {
            self.global.clone()
        } else if !self.all.is_empty() {
            self.all.clone()
        } else {
            Vec::new()
        }
    }
}

fn split_view_tensor<B: BackendTrait>(views: &Tensor<B, 5>) -> Vec<Tensor<B, 4>> {
    let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
    let mut out = Vec::with_capacity(view_count);
    for view_idx in 0..view_count {
        let view = views
            .clone()
            .slice_dim(1, view_idx..view_idx + 1)
            .reshape([batch, channels, height, width]);
        out.push(view);
    }
    out
}

fn collect_views<B: BackendTrait>(
    images: Tensor<B, 4>,
    target_images: Option<Tensor<B, 4>>,
    view_images: Option<Tensor<B, 5>>,
    global_view_images: Option<Tensor<B, 5>>,
    local_view_images: Option<Tensor<B, 5>>,
) -> CollectedViews<B> {
    let mut global = Vec::new();
    let mut local = Vec::new();
    let mut all = Vec::new();

    if let Some(global_views) = global_view_images {
        let views = split_view_tensor(&global_views);
        global.extend(views.clone());
        all.extend(views);
    }
    if let Some(local_views) = local_view_images {
        let views = split_view_tensor(&local_views);
        local.extend(views.clone());
        all.extend(views);
    }
    if all.is_empty() {
        if let Some(view_images) = view_images {
            let views = split_view_tensor(&view_images);
            global.extend(views.clone());
            all.extend(views);
        } else if let Some(target) = target_images {
            global.push(images.clone());
            global.push(target.clone());
            all.push(images);
            all.push(target);
        } else {
            global.push(images.clone());
            all.push(images);
        }
    }

    CollectedViews { global, local, all }
}

fn stack_views<B: BackendTrait>(views: &[Tensor<B, 4>]) -> Tensor<B, 4> {
    let view_count = views.len();
    if view_count == 1 {
        views[0].clone()
    } else {
        Tensor::cat(views.iter().cloned().collect(), 0)
    }
}

fn sample_patch_mask<B: BackendTrait>(
    device: &B::Device,
    batch: usize,
    tokens: usize,
    mask_ratio: f32,
    randomize_mask: bool,
) -> Tensor<B, 2> {
    if batch == 0 || tokens == 0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device);
    }
    let mask_ratio = mask_ratio.clamp(0.0, 1.0);
    if mask_ratio <= 0.0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device);
    }
    if mask_ratio >= 1.0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device).add_scalar(1.0);
    }
    if randomize_mask {
        return Tensor::<B, 2>::random(
            [batch, tokens],
            TensorDistribution::Uniform(0.0, 1.0),
            device,
        )
        .lower_elem(mask_ratio)
        .float();
    }

    let total = batch * tokens;
    let mut rng = StdRng::seed_from_u64(0);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let value = if rng.r#gen::<f32>() < mask_ratio {
            1.0
        } else {
            0.0
        };
        data.push(value);
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [batch, tokens]), device)
}

fn lejepa_invariance_loss<B: BackendTrait>(proj: Tensor<B, 3>) -> Tensor<B, 1> {
    let device = proj.device();
    let [views, batch, dim] = proj.shape().dims::<3>();
    if views == 0 || batch == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let mean = proj.clone().mean_dim(0);
    (proj - mean).powf_scalar(2.0).mean()
}

fn normalize_columns<B: BackendTrait>(matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = matrix
        .clone()
        .powf_scalar(2.0)
        .sum_dim(0)
        .sqrt()
        .add_scalar(LEJEPA_EPS);
    matrix / norm
}

fn lejepa_sigreg_loss<B: BackendTrait>(
    proj: Tensor<B, 3>,
    config: &VisionLejepaConfig,
) -> Tensor<B, 1> {
    lejepa_sigreg_loss_params(
        proj,
        config.sigreg_knots,
        config.sigreg_t_max,
        config.sigreg_proj_dim,
    )
}

fn lejepa_sigreg_loss_params<B: BackendTrait>(
    proj: Tensor<B, 3>,
    sigreg_knots: usize,
    sigreg_t_max: f32,
    sigreg_proj_dim: usize,
) -> Tensor<B, 1> {
    let device = proj.device();
    let [views, batch, dim] = proj.shape().dims::<3>();
    if views == 0 || batch == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }

    let knots = sigreg_knots.max(2);
    let t_max = sigreg_t_max.max(LEJEPA_EPS);
    let dt = t_max / (knots as f32 - 1.0);
    let mut t = Vec::with_capacity(knots);
    let mut phi = Vec::with_capacity(knots);
    let mut weights = Vec::with_capacity(knots);
    for i in 0..knots {
        let value = i as f32 * dt;
        let window = (-0.5 * value * value).exp();
        let weight = if i == 0 || i + 1 == knots { dt } else { 2.0 * dt };
        t.push(value);
        phi.push(window);
        weights.push(weight * window);
    }

    let t = Tensor::<B, 1>::from_data(TensorData::new(t, [knots]), &device)
        .reshape([1, 1, 1, knots]);
    let phi = Tensor::<B, 1>::from_data(TensorData::new(phi, [knots]), &device)
        .reshape([1, 1, knots]);
    let weights = Tensor::<B, 1>::from_data(TensorData::new(weights, [knots]), &device)
        .reshape([1, 1, knots]);

    let sketch_dim = sigreg_proj_dim.max(1);
    let a = Tensor::<B, 2>::random(
        [dim, sketch_dim],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );
    let a = normalize_columns(a);

    let proj_flat = proj.reshape([views * batch, dim]);
    let sketched = proj_flat.matmul(a).reshape([views, batch, sketch_dim]);
    let x_t = sketched.unsqueeze_dim::<4>(3).mul(t);
    let cos = x_t
        .clone()
        .cos()
        .mean_dim(1)
        .reshape([views, sketch_dim, knots]);
    let sin = x_t
        .sin()
        .mean_dim(1)
        .reshape([views, sketch_dim, knots]);
    let phi = phi.repeat_dim(0, views).repeat_dim(1, sketch_dim);
    let weights = weights.repeat_dim(0, views).repeat_dim(1, sketch_dim);
    let err = (cos - phi).powf_scalar(2.0) + sin.powf_scalar(2.0);
    let statistic = err.mul(weights).sum_dim(2).mul_scalar(batch as f32);
    statistic.mean()
}

fn normalize_artifact_legend(legend: Option<Vec<String>>, view_count: usize) -> Option<Vec<String>> {
    if view_count == 0 {
        return None;
    }
    let mut legend = legend.unwrap_or_else(|| {
        (0..view_count)
            .map(|idx| format!("view_{idx}"))
            .collect()
    });
    if legend.len() < view_count {
        for idx in legend.len()..view_count {
            legend.push(format!("view_{idx}"));
        }
    } else if legend.len() > view_count {
        legend.truncate(view_count);
    }
    Some(legend)
}

fn build_lejepa_artifacts<B: BackendTrait>(
    config: &VisionLejepaConfig,
    views: &[Tensor<B, 4>],
    frames: Option<Tensor<B, 5>>,
    first_patch: Option<Tensor<B, 3>>,
    probe_logits: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1, Int>>,
    legend: Option<Vec<String>>,
) -> Option<VisionArtifactInput<B>> {
    let max_images = config.artifact_max_images;
    let max_views = config.artifact_max_views;
    if max_images == 0 || max_views == 0 || views.is_empty() {
        return None;
    }
    let [batch, _, _, _] = views[0].shape().dims::<4>();
    let image_count = max_images.min(batch);
    if image_count == 0 {
        return None;
    }
    let view_count = max_views.min(views.len()).max(1);
    let legend = normalize_artifact_legend(legend, view_count);
    let mut stacked = Vec::with_capacity(view_count);
    for view in views.iter().take(view_count) {
        let view = view.clone().slice_dim(0, 0..image_count);
        stacked.push(view.unsqueeze_dim::<5>(1));
    }
    let views_tensor = Tensor::cat(stacked, 1);

    let patch_norms = first_patch.and_then(|patch| {
        let [batch, tokens, _] = patch.shape().dims::<3>();
        if batch == 0 || tokens == 0 {
            return None;
        }
        let grid = (tokens as f64).sqrt().round() as usize;
        if grid * grid != tokens {
            return None;
        }
        let norms = patch.powf_scalar(2.0).sum_dim(2).sqrt();
        let norms = norms.reshape([batch, grid, grid]);
        Some(norms.slice_dim(0, 0..image_count))
    });

    let probe_logits = probe_logits.map(|logits| logits.slice_dim(0, 0..image_count));
    let labels = labels.map(|labels| labels.slice_dim(0, 0..image_count));
    let frames = frames.map(|frames| frames.slice_dim(0, 0..image_count));

    Some(VisionArtifactInput {
        views: Some(views_tensor),
        frames,
        patch_norms,
        probe_logits,
        labels,
        legend,
    })
}

fn select_trajectory_indices(total: usize, max: usize) -> Vec<usize> {
    if total == 0 || max == 0 {
        return Vec::new();
    }
    if max >= total {
        return (0..total).collect();
    }
    if max == 1 {
        return vec![total - 1];
    }
    let last = (total - 1) as f32;
    let denom = (max - 1) as f32;
    let mut indices = Vec::with_capacity(max);
    for i in 0..max {
        let idx = ((i as f32) * last / denom).round() as usize;
        indices.push(idx.min(total - 1));
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

fn gaussian_downsample_kernel<B: BackendTrait>(channels: usize, device: &B::Device) -> Tensor<B, 4> {
    let weights = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
    let mut kernel = vec![0.0_f32; channels * 5 * 5];
    for c in 0..channels {
        let base = c * 25;
        for ky in 0..5 {
            for kx in 0..5 {
                kernel[base + ky * 5 + kx] = (weights[ky] * weights[kx]) / 256.0;
            }
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(kernel, [channels, 1, 5, 5]), device)
}

fn replicate_pad2d<B: BackendTrait>(images: Tensor<B, 4>, pad: usize) -> Tensor<B, 4> {
    if pad == 0 {
        return images;
    }
    let [_, _, height, width] = images.shape().dims::<4>();
    if height == 0 || width == 0 {
        return images;
    }
    let top = images
        .clone()
        .slice_dim(2, 0..1)
        .repeat_dim(2, pad);
    let bottom = images
        .clone()
        .slice_dim(2, height - 1..height)
        .repeat_dim(2, pad);
    let padded_v = Tensor::cat(vec![top, images, bottom], 2);
    let left = padded_v
        .clone()
        .slice_dim(3, 0..1)
        .repeat_dim(3, pad);
    let right = padded_v
        .clone()
        .slice_dim(3, width - 1..width)
        .repeat_dim(3, pad);
    Tensor::cat(vec![left, padded_v, right], 3)
}

fn downsample_image<B: BackendTrait>(images: Tensor<B, 4>) -> Option<Tensor<B, 4>> {
    let [_batch, channels, height, width] = images.shape().dims::<4>();
    if channels == 0 || height < 2 || width < 2 {
        return None;
    }
    let even_h = height - (height % 2);
    let even_w = width - (width % 2);
    if even_h == 0 || even_w == 0 {
        return None;
    }
    let device = images.device();
    let images = images
        .slice_dim(2, 0..even_h)
        .slice_dim(3, 0..even_w);
    let padded = replicate_pad2d(images, 2);
    let kernel = gaussian_downsample_kernel::<B>(channels, &device);
    let options = ConvOptions::new([2, 2], [0, 0], [1, 1], channels.max(1));
    Some(conv2d(padded, kernel, None, options))
}

fn should_fix_grid<B: BackendTrait>() -> bool
where
    B::Device: 'static,
{
    #[cfg(any(feature = "train", feature = "cli"))]
    {
        if TypeId::of::<B::Device>() == TypeId::of::<NdArrayDevice>() {
            return false;
        }
    }
    true
}

fn fix_grid_for_burn<B: BackendTrait>(
    grid: Tensor<B, 4>,
    height_in: usize,
    width_in: usize,
) -> Tensor<B, 4> {
    if !should_fix_grid::<B>() {
        return grid;
    }
    if width_in <= 1 || height_in <= 1 {
        return grid;
    }
    let x_half = (width_in - 1) as f32 * 0.5;
    let y_half = (height_in - 1) as f32 * 0.5;
    if (x_half - y_half).abs() <= f32::EPSILON {
        return grid;
    }
    // burn-tensor's default grid_sample scales y by x_half; compensate for all backends.
    let scale = y_half / x_half;
    let grid_x = grid.clone().slice_dim(3, 0..1);
    let grid_y = grid.slice_dim(3, 1..2).mul_scalar(scale);
    Tensor::cat(vec![grid_x, grid_y], 3)
}

fn grid_sample_2d_bilinear<B: BackendTrait>(
    tensor: Tensor<B, 4>,
    grid: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, height_in, width_in] = tensor.shape().dims::<4>();
    let grid = fix_grid_for_burn::<B>(grid, height_in, width_in);
    tensor.grid_sample_2d(grid, InterpolateMode::Bilinear)
}

fn build_foveated_base_grid<B: BackendTrait>(
    patch_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let patch = patch_size.max(1);
    let half = patch as f32 * 0.5;
    let mut coords = Vec::with_capacity(patch * patch * 2);
    for y in 0..patch {
        for x in 0..patch {
            let ux = (x as f32 + 0.5 - half) / half;
            let uy = (y as f32 + 0.5 - half) / half;
            coords.push(ux);
            coords.push(uy);
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(coords, [patch * patch * 2]), device)
        .reshape([patch, patch, 2])
        .unsqueeze_dim::<4>(0)
}

fn build_image_grid<B: BackendTrait>(
    out_height: usize,
    out_width: usize,
    in_height: usize,
    in_width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let out_height = out_height.max(1);
    let out_width = out_width.max(1);
    let in_height = in_height.max(1);
    let in_width = in_width.max(1);
    let mut coords = Vec::with_capacity(out_height * out_width * 2);
    let scale_x = if in_width > 1 {
        2.0 / (in_width as f32 - 1.0)
    } else {
        0.0
    };
    let scale_y = if in_height > 1 {
        2.0 / (in_height as f32 - 1.0)
    } else {
        0.0
    };
    let denom_w = out_width as f32;
    let denom_h = out_height as f32;
    for y in 0..out_height {
        let fy = (y as f32 + 0.5) / denom_h;
        let gy = if in_height > 1 {
            (fy * in_height as f32 - 0.5) * scale_y - 1.0
        } else {
            0.0
        };
        for x in 0..out_width {
            let fx = (x as f32 + 0.5) / denom_w;
            let gx = if in_width > 1 {
                (fx * in_width as f32 - 0.5) * scale_x - 1.0
            } else {
                0.0
            };
            coords.push(gx);
            coords.push(gy);
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(coords, [out_height * out_width * 2]), device)
        .reshape([out_height, out_width, 2])
        .unsqueeze_dim::<4>(0)
}

fn build_level_coords<B: BackendTrait>(grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
    let mut coords = Vec::with_capacity(grid.height * grid.width * 2);
    let inv_w = 1.0 / (grid.width.max(1) as f32);
    let inv_h = 1.0 / (grid.height.max(1) as f32);
    for y in 0..grid.height {
        let cy = (y as f32 + 0.5) * inv_h;
        for x in 0..grid.width {
            let cx = (x as f32 + 0.5) * inv_w;
            coords.push(cx);
            coords.push(cy);
        }
    }
    Tensor::<B, 1>::from_data(
        TensorData::new(coords, [grid.height * grid.width * 2]),
        device,
    )
    .reshape([grid.height * grid.width, 2])
}

fn saccade_eye_color(eye: usize) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 6] = [
        [0.95, 0.25, 0.25],
        [0.25, 0.65, 0.95],
        [0.25, 0.85, 0.4],
        [0.95, 0.75, 0.25],
        [0.75, 0.35, 0.95],
        [0.9, 0.9, 0.2],
    ];
    PALETTE[eye % PALETTE.len()]
}

fn saccade_circle_overlay<B: BackendTrait>(
    images: Tensor<B, 4>,
    mean: Tensor<B, 2>,
    sigma: Tensor<B, 2>,
    color: [f32; 3],
) -> Option<Tensor<B, 4>> {
    let outer = sigma
        .clone()
        .mul_scalar(SACCADE_RING_OUTER_SCALE)
        .max_pair(sigma.clone())
        .clamp_max(1.0 - SACCADE_EPS);
    let images = saccade_ring_overlay(
        images,
        mean.clone(),
        outer,
        color,
        SACCADE_RING_OUTER_INTENSITY,
    )?;
    saccade_ring_overlay(images, mean, sigma, color, 1.0)
}

fn saccade_ring_overlay<B: BackendTrait>(
    images: Tensor<B, 4>,
    mean: Tensor<B, 2>,
    radius: Tensor<B, 2>,
    color: [f32; 3],
    intensity_scale: f32,
) -> Option<Tensor<B, 4>> {
    let device = images.device();
    let [batch, channels, height, width] = images.shape().dims::<4>();
    if batch == 0 || channels < 3 || height == 0 || width == 0 {
        return None;
    }
    let x_coords = Tensor::<B, 1>::from_data(
        TensorData::new(
            (0..width)
                .map(|x| (x as f32 + 0.5) / width as f32)
                .collect::<Vec<_>>(),
            [width],
        ),
        &device,
    )
    .reshape([1, 1, 1, width]);
    let y_coords = Tensor::<B, 1>::from_data(
        TensorData::new(
            (0..height)
                .map(|y| (y as f32 + 0.5) / height as f32)
                .collect::<Vec<_>>(),
            [height],
        ),
        &device,
    )
    .reshape([1, 1, height, 1]);

    let cx = mean
        .clone()
        .slice_dim(1, 0..1)
        .reshape([batch, 1, 1, 1]);
    let cy = mean
        .slice_dim(1, 1..2)
        .reshape([batch, 1, 1, 1]);
    let radius = radius.reshape([batch, 1, 1, 1]);

    let dx = x_coords - cx;
    let dy = y_coords - cy;
    let dist = (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)).sqrt();
    let ring = dist.sub(radius).abs();
    let ring_mask = activation::relu(
        ring.mul_scalar(-1.0).add_scalar(SACCADE_RING_WIDTH),
    )
    .div_scalar(SACCADE_RING_WIDTH.max(SACCADE_EPS));
    let color_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(vec![color[0], color[1], color[2]], [3]),
        &device,
    )
    .reshape([1, 3, 1, 1]);
    let ring_rgb = ring_mask
        .clone()
        .repeat_dim(1, 3)
        .mul(color_tensor)
        .mul_scalar(SACCADE_RING_INTENSITY * intensity_scale);
    let inv_mask = ring_mask.mul_scalar(-1.0).add_scalar(1.0);
    let overlay = images.mul(inv_mask) + ring_rgb;
    Some(overlay)
}

fn saccade_patch_views<B: BackendTrait>(
    patches: Vec<Tensor<B, 4>>,
    target_height: usize,
) -> Option<Vec<Tensor<B, 4>>> {
    if patches.is_empty() || target_height == 0 {
        return None;
    }
    let mut views = Vec::with_capacity(patches.len());
    for patch in patches {
        let view = pad_view_height_centered(patch, target_height);
        views.push(view);
    }
    Some(views)
}

fn pad_view_width<B: BackendTrait>(view: Tensor<B, 4>, target_width: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_width <= width {
        return view;
    }
    let pad = target_width - width;
    if pad == 0 {
        return view;
    }
    let device = view.device();
    let padding = Tensor::<B, 4>::zeros([batch, channels, height, pad], &device);
    Tensor::cat(vec![view, padding], 3)
}

fn pad_view_width_centered<B: BackendTrait>(
    view: Tensor<B, 4>,
    target_width: usize,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_width <= width {
        return view;
    }
    let pad = target_width - width;
    if pad == 0 {
        return view;
    }
    let left = pad / 2;
    let right = pad - left;
    let device = view.device();
    let padding_left = Tensor::<B, 4>::zeros([batch, channels, height, left], &device);
    let padding_right = Tensor::<B, 4>::zeros([batch, channels, height, right], &device);
    Tensor::cat(vec![padding_left, view, padding_right], 3)
}

fn view_separator_like<B: BackendTrait>(like: &Tensor<B, 4>, width: usize) -> Tensor<B, 4> {
    let [batch, channels, height, _] = like.shape().dims::<4>();
    if width == 0 || batch == 0 || channels == 0 || height == 0 {
        return Tensor::<B, 4>::zeros([batch.max(1), channels.max(1), height.max(1), width.max(1)], &like.device());
    }
    Tensor::<B, 4>::zeros([batch, channels, height, width], &like.device())
}

fn pad_view_height_centered<B: BackendTrait>(
    view: Tensor<B, 4>,
    target_height: usize,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_height <= height {
        return view;
    }
    let pad = target_height - height;
    if pad == 0 {
        return view;
    }
    let top = pad / 2;
    let bottom = pad - top;
    let device = view.device();
    let padding_top = Tensor::<B, 4>::zeros([batch, channels, top, width], &device);
    let padding_bottom = Tensor::<B, 4>::zeros([batch, channels, bottom, width], &device);
    Tensor::cat(vec![padding_top, view, padding_bottom], 2)
}

#[cfg(test)]
mod lejepa_tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    #[test]
    fn lejepa_invariance_loss_is_finite() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
        let loss = lejepa_invariance_loss(proj);
        let value = loss
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        assert!(value.is_finite());
    }

    #[test]
    fn lejepa_sigreg_loss_is_finite() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let config = VisionLejepaConfig::default();

        let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
        let loss = lejepa_sigreg_loss(proj, &config);
        let value = loss
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        assert!(value.is_finite());
    }

    #[test]
    fn patchify_roundtrip() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();

        let batch = 1;
        let channels = 3;
        let height = 4;
        let width = 4;
        let patch_size = 2;
        let total = batch * channels * height * width;
        let data: Vec<f32> = (0..total).map(|v| v as f32).collect();

        let images = Tensor::<Backend, 4>::from_data(
            TensorData::new(data.clone(), [batch, channels, height, width]),
            &device,
        );
        let patches = patchify(images.clone(), patch_size);
        let [patch_batch, tokens, patch_dim] = patches.shape().dims::<3>();
        assert_eq!(patch_batch, batch);
        assert_eq!(tokens, (height / patch_size) * (width / patch_size));
        assert_eq!(patch_dim, channels * patch_size * patch_size);

        let recon = unpatchify(patches, patch_size, height, width, channels);
        let out = recon
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("recon vec");
        assert_eq!(data, out);
    }
}

fn teacher_variant_dim(variant: VisionTeacherVariant) -> usize {
    match variant {
        VisionTeacherVariant::Vits => 384,
        VisionTeacherVariant::Vitb => 768,
        VisionTeacherVariant::Vitl => 1024,
        VisionTeacherVariant::Vitg => 1536,
    }
}

fn build_dino_config(
    variant: VisionTeacherVariant,
    image_size: usize,
    patch_size: usize,
) -> DinoVisionTransformerConfig {
    match variant {
        VisionTeacherVariant::Vits => {
            DinoVisionTransformerConfig::vits(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitb => {
            DinoVisionTransformerConfig::vitb(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitl => {
            DinoVisionTransformerConfig::vitl(Some(image_size), Some(patch_size))
        }
        VisionTeacherVariant::Vitg => {
            DinoVisionTransformerConfig::vitg(Some(image_size), Some(patch_size))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ContextStrategyConfig;
    use crate::{
        FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode, VisionPyramidMode,
    };
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    fn make_training(max_iters: usize, epochs: Option<usize>) -> TrainingHyperparameters {
        TrainingHyperparameters {
            block_size: 16,
            batch_size: 2,
            epochs,
            max_iters,
            log_frequency: 10,
            fast_train: false,
            context_strategy: ContextStrategyConfig::Infinite,
        }
    }

    fn make_saccade_model<B: BackendTrait>(
        device: &B::Device,
        num_eyes: usize,
    ) -> (VisionSaccadeModel<B>, VisionDragonHatchlingConfig) {
        make_saccade_model_with_dims(device, num_eyes, 8, 8, 4)
    }

    fn make_saccade_model_with_dims<B: BackendTrait>(
        device: &B::Device,
        num_eyes: usize,
        image_width: usize,
        image_height: usize,
        patch_size: usize,
    ) -> (VisionSaccadeModel<B>, VisionDragonHatchlingConfig) {
        let patch_size = patch_size.max(1);
        let image_size = image_width.max(image_height).max(patch_size);
        let grid_w = (image_width / patch_size).max(1);
        let grid_h = (image_height / patch_size).max(1);
        let vision_config = VisionDragonHatchlingConfig {
            image_size,
            patch_size,
            in_channels: 3,
            embed_dim: 16,
            steps: 2,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            dropout: 0.0,
            projection_dim: 8,
            projection_hidden_dim: 16,
            use_cls_token: true,
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: grid_h,
            pos_max_width: grid_w,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: FusedKernelConfig::default(),
        };
        let model = VisionDragonHatchling::<B>::new(vision_config.clone(), device);
        let saccade_config = VisionSaccadeConfig {
            num_eyes,
            mip_levels: 3,
            pyramid_mode: VisionPyramidMode::Laplacian,
            fovea_sampling_mode: VisionFoveaSamplingMode::Sequential,
            fovea_subpatch_size: 0,
            inner_steps: 1,
            low_mem_pre_rollout: true,
            lambda: 0.02,
            sigreg_knots: 5,
            sigreg_t_max: 1.0,
            sigreg_proj_dim: 8,
            recon_weight: 0.0,
            recon_mask_ratio: 0.0,
            recon_hidden_dim: 16,
            artifact_output: VisionArtifactOutputMode::Images,
            artifact_fps: 4,
            artifact_every: 0,
            artifact_max_images: 0,
            artifact_max_views: 0,
            artifact_overwrite: true,
        };
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 2,
            backprop_steps: 2,
        };
        let recon_patch_dim =
            vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
        let saccade = VisionSaccadeModel::new(
            model,
            saccade_config,
            vision_config.embed_dim,
            rollout,
            recon_patch_dim,
            device,
        );
        (saccade, vision_config)
    }

    fn make_test_image(channels: usize, height: usize, width: usize) -> Vec<f32> {
        let mut data = Vec::with_capacity(channels * height * width);
        let denom_w = (width - 1).max(1) as f32;
        let denom_h = (height - 1).max(1) as f32;
        for c in 0..channels {
            for y in 0..height {
                let gy = y as f32 / denom_h;
                for x in 0..width {
                    let gx = x as f32 / denom_w;
                    let checker = ((x / 2 + y / 3 + c) % 2) as f32;
                    let value = match c {
                        0 => gx,
                        1 => gy,
                        _ => 0.55 * gx + 0.35 * gy + 0.1 * checker,
                    };
                    data.push(value);
                }
            }
        }
        data
    }

    fn run_foveation_equivalence<B: BackendTrait>(
        device: &B::Device,
        backend_label: &str,
    ) {
        let batch = 1;
        let channels = 3;
        let cases = [
            ([0.5, 0.5], 0.2, 0.2),
            ([0.2, 0.8], 0.1, 0.3),
            ([0.8, 0.2], 0.35, 0.4),
            ([0.6, 0.4], 0.4, 0.2),
            ([0.02, 0.98], 0.05, 0.15),
            ([0.98, 0.02], 0.25, 0.1),
            ([0.1, 0.1], 0.02, 0.05),
            ([0.9, 0.9], 0.45, 0.45),
        ];
        let size_configs = [
            (8, 8, 4, 2),
            (12, 8, 4, 2),
            (12, 12, 6, 3),
            (16, 16, 8, 4),
        ];
        let pyramid_configs = [
            (VisionPyramidMode::Stacked, 2),
            (VisionPyramidMode::Laplacian, 3),
        ];
        let mut sampling_modes = vec![
            VisionFoveaSamplingMode::Batched,
            VisionFoveaSamplingMode::Sequential,
            VisionFoveaSamplingMode::Subpatch,
        ];
        if super::foveation_cubecl::supports_backend::<B>() {
            sampling_modes.push(VisionFoveaSamplingMode::Cubecl);
        }

        for (size_idx, (width, height, patch_size, subpatch_size)) in
            size_configs.iter().copied().enumerate()
        {
            let data = make_test_image(channels, height, width);
            let (min_val, max_val) = data.iter().fold(
                (f32::INFINITY, f32::NEG_INFINITY),
                |(min_v, max_v), &val| (min_v.min(val), max_v.max(val)),
            );
            assert!(max_val - min_val > 0.25, "test image too flat");
            let images = Tensor::<B, 4>::from_data(
                TensorData::new(data.clone(), [batch, channels, height, width]),
                device,
            );

            for (pyramid_mode, mip_levels) in pyramid_configs.iter().copied() {
                for sampling_mode in sampling_modes.iter().copied() {
                    let (mut saccade, vision_config) =
                        make_saccade_model_with_dims::<B>(
                            device,
                            1,
                            width,
                            height,
                            patch_size,
                        );
                    saccade.config.pyramid_mode = pyramid_mode;
                    saccade.config.fovea_sampling_mode = sampling_mode;
                    saccade.config.fovea_subpatch_size = if matches!(
                        sampling_mode,
                        VisionFoveaSamplingMode::Subpatch
                    ) {
                        subpatch_size
                    } else {
                        0
                    };
                    saccade.config.mip_levels = mip_levels;

                    let mut sampler = SaccadeFoveationSampler::<B>::new(
                        vision_config,
                        saccade.config.clone(),
                        device,
                    );
                    sampler.update_image(images.clone());
                    let patch_size = sampler.patch_size();
                    let cpu_depth = sampler.mip_levels().max(1);
                    let cache = foveation::build_pyramid_cache(
                        foveation::image_from_nchw(&data, 0, channels, height, width)
                            .expect("cpu image"),
                        cpu_depth,
                        saccade.config.pyramid_mode,
                    );

                    for (case_idx, (mean_vals, sigma_val, radius_val)) in
                        cases.iter().enumerate()
                    {
                        let mean_vals = *mean_vals;
                        let sigma_val = *sigma_val;
                        let radius_val = *radius_val;
                        let mean = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![mean_vals[0], mean_vals[1]], [batch, 2]),
                            device,
                        );
                        let sigma = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![sigma_val], [batch, 1]),
                            device,
                        );
                        let radius = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![radius_val], [batch, 1]),
                            device,
                        );
                        let patch_view =
                            sampler.sample_patch_with_radius(mean, sigma, radius);
                        let patch_vec = patch_view
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("patch vec");

                        let expected_patch = foveation::render_foveated_patch_with_radius(
                            &cache,
                            mean_vals,
                            sigma_val,
                            radius_val,
                            patch_size,
                        );
                        let mut expected = vec![0.0f32; channels * patch_size * patch_size];
                        let channel_stride = patch_size * patch_size;
                        for y in 0..patch_size {
                            for x in 0..patch_size {
                                let src = (y * patch_size + x) * 3;
                                let dst = y * patch_size + x;
                                expected[dst] = expected_patch[src];
                                expected[dst + channel_stride] =
                                    expected_patch[src + 1];
                                expected[dst + 2 * channel_stride] =
                                    expected_patch[src + 2];
                            }
                        }
                        let mut mse = 0.0f32;
                        let mut max_abs = 0.0f32;
                        for (lhs, rhs) in patch_vec.iter().zip(expected.iter()) {
                            let diff = lhs - rhs;
                            let abs = diff.abs();
                            if abs > max_abs {
                                max_abs = abs;
                            }
                            mse += diff * diff;
                        }
                        mse /= patch_vec.len().max(1) as f32;
                        let max_abs_threshold = 1e-3;
                        let mse_threshold = 1e-6;
                        assert!(
                            max_abs < max_abs_threshold && mse < mse_threshold,
                            "backend {backend_label} size {size_idx} {width}x{height} patch {patch_size} pyramid {pyramid_mode:?} sampling {sampling_mode:?} case {case_idx} max_abs {max_abs} mse {mse}"
                        );
                    }
                }
            }
        }
    }

    fn make_level<B: BackendTrait>(
        device: &B::Device,
        tokens: usize,
        dim: usize,
        offset: f32,
    ) -> Tensor<B, 3> {
        let mut data = Vec::with_capacity(tokens * dim);
        for token in 0..tokens {
            for d in 0..dim {
                data.push(offset + token as f32 + d as f32 * 0.1);
            }
        }
        Tensor::<B, 2>::from_data(TensorData::new(data, [tokens, dim]), device)
            .reshape([1, tokens, dim])
    }

    fn assert_mse_below<B: BackendTrait>(tensor: Tensor<B, 1>, threshold: f32) {
        let value = tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("mse vec")[0];
        assert!(value < threshold, "mse {value} >= {threshold}");
    }

    fn assert_mse_above<B: BackendTrait>(tensor: Tensor<B, 1>, threshold: f32) {
        let value = tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("mse vec")[0];
        assert!(value > threshold, "mse {value} <= {threshold}");
    }

    fn assert_tensor_finite<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) {
        let values = tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor vec");
        assert!(!values.is_empty(), "tensor has no values");
        assert!(
            values.iter().all(|value| value.is_finite()),
            "tensor contains non-finite values"
        );
    }

    fn saccade_eye_step<B: BackendTrait>(
        saccade: &VisionSaccadeModel<B>,
        images: Tensor<B, 4>,
        eye_idx: usize,
    ) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
        let device = images.device();
        let [batch, _channels, height, _width] = images.shape().dims::<4>();
        let patch = saccade.model.patch_embed_raw(images.clone());
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        assert!(grid_h > 0 && grid_w > 0);
        let patch_size = height / grid_h;
        assert!(patch_size > 0);

        let mip_levels = saccade.build_mip_pyramid(images, patch_size);
        assert!(!mip_levels.is_empty());
        let input_levels: Vec<Tensor<B, 3>> =
            mip_levels.iter().map(|level| level.tokens.clone()).collect();
        let grids: Vec<PatchGrid> = mip_levels.iter().map(|level| level.grid).collect();
        let input_residuals = match saccade.config.pyramid_mode {
            VisionPyramidMode::Stacked => input_levels.clone(),
            VisionPyramidMode::Laplacian => saccade.decompose_pyramid(&input_levels, &grids),
        };
        let laplacian_images = if matches!(saccade.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            saccade.build_laplacian_images(&mip_levels)
        } else {
            None
        };
        let embed_dim = patch.tokens.shape().dims::<3>()[2];
        let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let base_traj = saccade
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let state_levels: Vec<Tensor<B, 3>> = input_residuals
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
            .collect();
        let state_composed = match saccade.config.pyramid_mode {
            VisionPyramidMode::Stacked => state_levels.clone(),
            VisionPyramidMode::Laplacian => saccade.compose_pyramid(&state_levels, &grids),
        };
        let eye_embed = saccade
            .eye_token
            .val()
            .slice_dim(0, eye_idx..eye_idx + 1)
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, traj_len);
        let traj_with_eye = base_traj + eye_embed;
        let traj_summary = traj_with_eye
            .clone()
            .mean_dim(1)
            .reshape([batch, 1, embed_dim]);
        let params = saccade.saccade_head.forward(traj_summary);
        let (mean, sigma) = saccade.decode_saccade_params(params);
        let weights = saccade.mip_gaussian_weights(&mip_levels, mean.clone(), sigma.clone());
        let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
        let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
        let base_grid = build_foveated_base_grid::<B>(patch_size, &device);
        let patch_image = saccade.foveated_patch_image(
            &mip_levels,
            &base_grid,
            mean_step,
            sigma_step,
            laplacian_images.as_ref(),
        );
        let patch_tokens = saccade.model.patch_embed_raw(patch_image).tokens;
        let input_context = patch_tokens;
        let state_context = saccade.mip_weighted_sum(&state_composed, &weights);
        let fovea_params = Tensor::cat(vec![mean, sigma], 2);
        let fovea_embed = saccade.fovea_proj.forward(fovea_params);
        let input_tokens = saccade.input_proj.forward(input_context) + state_context + fovea_embed;
        let input_tokens = input_tokens.repeat_dim(1, traj_len);
        let tokens_in = traj_with_eye + input_tokens;
        let inner_steps = saccade.config.inner_steps.max(1);
        let out_tokens = saccade
            .model
            .forward_tokens_embed_steps(tokens_in, inner_steps)
            .patch_tokens;
        let residual = saccade.residual_proj.forward(out_tokens.clone());
        let residual_pool = residual
            .clone()
            .mean_dim(1)
            .reshape([batch, 1, embed_dim]);
        let next_traj = out_tokens;
        let mut updates = Vec::with_capacity(weights.len());
        for weights in &weights {
            updates.push(
                saccade.weighted_sum_tokens(weights.clone().swap_dims(1, 2), residual_pool.clone()),
            );
        }
        (next_traj, updates)
    }

    fn saccade_fovea_params<B: BackendTrait>(
        saccade: &VisionSaccadeModel<B>,
        traj_with_eye: Tensor<B, 3>,
        embed_dim: usize,
    ) -> Tensor<B, 3> {
        let [batch, _, _] = traj_with_eye.shape().dims::<3>();
        let traj_summary = traj_with_eye.mean_dim(1).reshape([batch, 1, embed_dim]);
        let params = saccade.saccade_head.forward(traj_summary);
        let (mean, sigma) = saccade.decode_saccade_params(params);
        Tensor::cat(vec![mean, sigma], 2)
    }

    fn saccade_weights_for_eye<B: BackendTrait>(
        saccade: &VisionSaccadeModel<B>,
        traj_with_eye: Tensor<B, 3>,
        mip_levels: &[SaccadeMipLevel<B>],
        embed_dim: usize,
    ) -> Vec<Tensor<B, 3>> {
        let [batch, _, _] = traj_with_eye.shape().dims::<3>();
        let traj_summary = traj_with_eye.mean_dim(1).reshape([batch, 1, embed_dim]);
        let params = saccade.saccade_head.forward(traj_summary);
        let (mean, sigma) = saccade.decode_saccade_params(params);
        saccade.mip_gaussian_weights(mip_levels, mean, sigma)
    }

    #[test]
    fn epochs_schedule_overrides_max_iters() {
        let training = make_training(5, Some(3));
        let schedule = resolve_train_schedule(&training, 4).expect("schedule");

        assert_eq!(schedule.source, ScheduleSource::Epochs);
        assert_eq!(schedule.steps_per_epoch, 4);
        assert_eq!(schedule.total_epochs, 3);
        assert_eq!(schedule.total_steps, 12);
        assert_eq!(schedule.total_steps % schedule.steps_per_epoch, 0);
    }

    #[test]
    fn max_iters_schedule_uses_step_limit() {
        let training = make_training(12, None);
        let schedule = resolve_train_schedule(&training, 5).expect("schedule");

        assert_eq!(schedule.source, ScheduleSource::MaxIters);
        assert_eq!(schedule.steps_per_epoch, 5);
        assert_eq!(schedule.total_steps, 12);
        assert_eq!(schedule.total_epochs, 3);
    }

    #[test]
    fn saccade_recon_loss_smoke() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
        let losses = saccade.forward_losses(batch, 2, 1, true, false);
        let value = losses
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        assert!(value.is_finite());
    }

    #[test]
    fn saccade_multi_eye_loss_smoke() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
        let losses = saccade.forward_losses(batch, 2, 1, true, false);
        let value = losses
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0];
        assert!(value.is_finite());
    }

    #[test]
    fn saccade_multi_eye_trajectory_states_diverge() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
        let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
        let mut eye_values = vec![0.0; embed_dim];
        eye_values.extend(std::iter::repeat(1.0).take(embed_dim));
        let eye_token =
            Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
        saccade.eye_token = Param::from_tensor(eye_token);

        let mut image_values = Vec::with_capacity(1 * 3 * 8 * 8);
        for idx in 0..(1 * 3 * 8 * 8) {
            image_values.push(idx as f32 / 255.0);
        }
        let images =
            Tensor::<Backend, 4>::from_data(TensorData::new(image_values, [1, 3, 8, 8]), &device);

        let (traj0, _) = saccade_eye_step(&saccade, images.clone(), 0);
        let (traj1, _) = saccade_eye_step(&saccade, images, 1);
        let mse = (traj0 - traj1).powf_scalar(2.0).mean();
        assert_mse_above(mse, 0.0);
    }

    #[test]
    fn saccade_patch_view_matches_cpu_foveation() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        run_foveation_equivalence::<Backend>(&device, "ndarray");
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn saccade_patch_view_matches_cpu_foveation_wgpu() {
        use burn_wgpu::{self, RuntimeOptions, Wgpu, graphics};
        use std::sync::Once;

        static INIT: Once = Once::new();
        type Backend = Wgpu<f32>;
        let device = burn_wgpu::WgpuDevice::default();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(&device, RuntimeOptions::default());
        });
        run_foveation_equivalence::<Backend>(&device, "wgpu");
    }

    #[test]
    fn saccade_multi_eye_updates_are_additive() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
        let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
        let mut eye_values = vec![0.0; embed_dim];
        eye_values.extend(std::iter::repeat(1.0).take(embed_dim));
        let eye_token =
            Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
        saccade.eye_token = Param::from_tensor(eye_token);

        let images = Tensor::<Backend, 4>::random([1, 3, 8, 8], TensorDistribution::Default, &device);
        let (_, updates0) = saccade_eye_step(&saccade, images.clone(), 0);
        let (_, updates1) = saccade_eye_step(&saccade, images, 1);

        let mut state_sum: Vec<Tensor<Backend, 3>> = updates0
            .iter()
            .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
            .collect();
        for (state, update0, update1) in
            state_sum.iter_mut().zip(updates0.iter()).zip(updates1.iter()).map(
                |((state, update0), update1)| (state, update0, update1),
            )
        {
            *state = state.clone() + update0.clone() + update1.clone();
        }

        let mut state_seq: Vec<Tensor<Backend, 3>> = updates0
            .iter()
            .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
            .collect();
        for (state, update) in state_seq.iter_mut().zip(updates0.iter()) {
            *state = state.clone() + update.clone();
        }
        for (state, update) in state_seq.iter_mut().zip(updates1.iter()) {
            *state = state.clone() + update.clone();
        }

        for (sum, seq) in state_sum.iter().zip(state_seq.iter()) {
            let mse = (sum.clone() - seq.clone()).powf_scalar(2.0).mean();
            assert_mse_below(mse, 1e-6);
        }
    }

    #[test]
    fn saccade_multi_eye_step_produces_finite_grads() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
        let images =
            Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
        let losses = saccade.forward_losses(batch, 1, 1, true, false);
        let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

        let eye_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
            .expect("eye_token grad");
        let traj_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
            .expect("trajectory_token grad");
        assert_tensor_finite(eye_grad);
        assert_tensor_finite(traj_grad);
    }

    #[test]
    fn saccade_artifact_frames_match_steps() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
        saccade.config.artifact_max_images = 1;
        saccade.config.artifact_max_views = 4;
        saccade.config.artifact_every = 1;

        let images = Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([1], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);

        let steps = 3;
        let losses = saccade.forward_losses(batch, steps, 1, false, true);
        let artifacts = losses.artifacts.expect("artifacts");
        let frames = artifacts.frames.expect("frames");
        let views = artifacts.views.expect("views");
        let [batch, frame_count, _, _, _] = frames.shape().dims::<5>();
        let [view_batch, view_count, _, _, _] = views.shape().dims::<5>();
        assert_eq!(batch, 1);
        assert_eq!(frame_count, steps);
        assert_eq!(view_batch, 1);
        assert_eq!(view_count, 4);
    }

    #[test]
    fn saccade_laplacian_roundtrip_is_exact() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let grids = vec![
            PatchGrid { height: 4, width: 4 },
            PatchGrid { height: 2, width: 2 },
            PatchGrid { height: 1, width: 1 },
        ];
        let levels = vec![
            make_level::<Backend>(&device, 16, 2, 0.0),
            make_level::<Backend>(&device, 4, 2, 10.0),
            make_level::<Backend>(&device, 1, 2, 20.0),
        ];
        let residuals = saccade.decompose_pyramid(&levels, &grids);
        let composed = saccade.compose_pyramid(&residuals, &grids);
        for (orig, recon) in levels.iter().zip(composed.iter()) {
            let mse = (orig.clone() - recon.clone()).powf_scalar(2.0).mean();
            assert_mse_below(mse, 1e-6);
        }
    }

    #[test]
    fn saccade_laplacian_drop_residual_matches_upsample() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let grids = vec![
            PatchGrid { height: 4, width: 4 },
            PatchGrid { height: 2, width: 2 },
            PatchGrid { height: 1, width: 1 },
        ];
        let levels = vec![
            make_level::<Backend>(&device, 16, 2, 0.0),
            make_level::<Backend>(&device, 4, 2, 10.0),
            make_level::<Backend>(&device, 1, 2, 20.0),
        ];
        let residuals = saccade.decompose_pyramid(&levels, &grids);
        let mut truncated = residuals.clone();
        truncated[0] = Tensor::<Backend, 3>::zeros([1, 16, 2], &device);
        let composed = saccade.compose_pyramid(&truncated, &grids);
        let upsampled = saccade.upsample_tokens(levels[1].clone(), grids[1], grids[0]);
        let mse = (composed[0].clone() - upsampled).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }

    #[test]
    fn saccade_mip_gaussian_weights_normalize() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let levels = vec![
            SaccadeMipLevel {
                tokens: Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
                grid: PatchGrid { height: 2, width: 2 },
                image: Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device),
            },
            SaccadeMipLevel {
                tokens: Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
                grid: PatchGrid { height: 1, width: 1 },
                image: Tensor::<Backend, 4>::zeros([2, 3, 4, 4], &device),
            },
        ];
        let mean = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.2, 0.4, 0.7, 0.9], [2, 1, 2]),
            &device,
        );
        let sigma = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.3, 0.5], [2, 1, 1]),
            &device,
        );
        let weights = saccade.mip_gaussian_weights(&levels, mean, sigma);
        let mut total = weights[0].clone().sum_dim(2);
        for weight in weights.iter().skip(1) {
            total = total + weight.clone().sum_dim(2);
        }
        let ones = Tensor::<Backend, 3>::ones([2, 1, 1], &device);
        let diff = total.add(ones.mul_scalar(-1.0));
        let mse = diff.powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }

    #[test]
    fn saccade_fovea_params_use_single_trajectory_token() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
        let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
        assert_eq!(traj_len, SACCADE_TRAJ_TOKENS);

        let base_traj = saccade
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim]);
        let eye_embed = saccade
            .eye_token
            .val()
            .reshape([1, 1, embed_dim])
            .repeat_dim(1, traj_len);
        let traj_with_eye = base_traj + eye_embed;
        let fovea_params = saccade_fovea_params(&saccade, traj_with_eye.clone(), embed_dim);
        assert_eq!(fovea_params.shape().dims::<3>(), [1, 1, 3]);

        let levels = vec![
            SaccadeMipLevel {
                tokens: Tensor::<Backend, 3>::zeros([1, 4, 3], &device),
                grid: PatchGrid { height: 2, width: 2 },
                image: Tensor::<Backend, 4>::zeros([1, 3, 4, 4], &device),
            },
            SaccadeMipLevel {
                tokens: Tensor::<Backend, 3>::zeros([1, 1, 3], &device),
                grid: PatchGrid { height: 1, width: 1 },
                image: Tensor::<Backend, 4>::zeros([1, 3, 2, 2], &device),
            },
        ];
        let weights = saccade_weights_for_eye(&saccade, traj_with_eye, &levels, embed_dim);
        for weight in weights {
            let shape = weight.shape().dims::<3>();
            assert_eq!(shape[1], 1, "fovea weights should use a single token");
        }
    }

    #[test]
    fn saccade_mip_scatter_gather_one_hot() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let residual = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 1, 3]),
            &device,
        );
        let mut state_levels = vec![
            Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
            Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
        ];
        let weights_level0 = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 1, 4]),
            &device,
        );
        let weights_level1 = Tensor::<Backend, 3>::zeros([2, 1, 1], &device);
        let weights = vec![weights_level0, weights_level1];

        saccade.apply_mip_residual(&mut state_levels, &weights, residual.clone());
        let gathered = saccade.mip_weighted_sum(&state_levels, &weights);
        let mse = (gathered - residual).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }

    #[test]
    fn saccade_upsample_tokens_mismatch_returns_zero() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);

        let tokens = Tensor::<Backend, 3>::zeros([1, 1, 2], &device);
        let from = PatchGrid { height: 2, width: 2 };
        let to = PatchGrid { height: 3, width: 3 };
        let upsampled = saccade.upsample_tokens(tokens, from, to);

        assert_eq!(upsampled.shape().dims(), [1, 9, 2]);
        let value = upsampled
            .sum()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("sum vec")[0];
        assert_eq!(value, 0.0);
    }

    #[test]
    fn saccade_level_coords_cache_is_bounded() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let grid = PatchGrid { height: 2, width: 2 };

        let _ = saccade.level_coords_cached(grid, &device);
        let _ = saccade.level_coords_cached(grid, &device);

        let len = saccade
            .level_coords_cache
            .inner
            .lock()
            .expect("level coords cache lock")
            .len();
        assert_eq!(len, 1);
    }

    #[test]
    fn saccade_upsample_weights_cache_is_bounded() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let from = PatchGrid { height: 2, width: 2 };
        let to = PatchGrid { height: 4, width: 4 };

        let _ = saccade.upsample_weights_cached(from, to, &device);
        let _ = saccade.upsample_weights_cached(from, to, &device);

        let len = saccade
            .upsample_weights_cache
            .inner
            .lock()
            .expect("upsample weights cache lock")
            .len();
        assert_eq!(len, 1);
    }

    #[test]
    fn saccade_step_produces_finite_grads() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let images =
            Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
        let losses = saccade.forward_losses(batch, 1, 1, true, false);
        let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

        let token_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
            .expect("trajectory_token grad");
        let eye_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
            .expect("eye_token grad");
        assert_tensor_finite(token_grad);
        assert_tensor_finite(eye_grad);
    }
}

fn build_vocab_only(config: &TrainingConfig) -> Result<()> {
    let dataset = prepare_dataset(&config.dataset, &config.training)?;
    let tokenizer = dataset.tokenizer();
    info!(
        "Tokenizer `{}` ready with {} tokens",
        config.dataset.tokenizer.kind_name(),
        tokenizer.len()
    );
    Ok(())
}

fn prepare_dataset(
    dataset_cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<Arc<Dataset>> {
    let tokenizer_path = dataset_cfg.tokenizer.storage_path(&dataset_cfg.cache_dir);
    let tokenizer_preexists = tokenizer_path
        .as_ref()
        .map(|path| path.is_file())
        .unwrap_or(false);

    let (dataset_enum, dataset_summary) = build_dataset(dataset_cfg, training)?;
    let dataset = Arc::new(dataset_enum);

    let tokenizer = dataset.tokenizer();
    match tokenizer_path {
        Some(path) if tokenizer_preexists => info!(
            "Loaded {} tokenizer with {} tokens from {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        Some(path) => info!(
            "Built {} tokenizer with {} tokens at {}",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len(),
            path.display()
        ),
        None => info!(
            "Initialized {} tokenizer with {} tokens (no persistence required)",
            dataset_cfg.tokenizer.kind_name(),
            tokenizer.len()
        ),
    };

    info!("{dataset_summary}");

    Ok(dataset)
}

fn log_theoretical_profile(config: &BDHConfig, batch: usize, block: usize, backend: &str) {
    let batch = batch as u64;
    let time = block as u64;
    let embed = config.n_embd as u64;
    let latent_per_head = config.latent_per_head() as u64;
    let latent_total = config.latent_total() as u64;
    let heads = config.n_head as u64;
    let bt = batch * time;

    let encoder_matmul = 2 * bt * embed * latent_total;
    let attn_scores = 2 * batch * heads * time * time * latent_per_head;
    let attn_value = 2 * batch * heads * time * time * embed;
    let decoder_matmul = 2 * bt * latent_total * embed;
    let total = encoder_matmul + attn_scores + attn_value + decoder_matmul;

    info!(
        "[train:{backend}] approx forward GFLOPs: total={total_gflops:.2}, encoder={enc:.2}, \
         attn_scores={scores:.2}, attn_value={value:.2}, decoder={dec:.2} (backward ~2x forward)",
        total_gflops = total as f64 / 1e9,
        enc = encoder_matmul as f64 / 1e9,
        scores = attn_scores as f64 / 1e9,
        value = attn_value as f64 / 1e9,
        dec = decoder_matmul as f64 / 1e9,
    );
}

fn create_run_dir(run_root: &Path) -> Result<(PathBuf, String)> {
    let mut generator = Generator::default();

    for _ in 0..64 {
        let name = generator
            .next()
            .unwrap_or_else(|| "nameless-hatchling".to_string());
        let candidate = run_root.join(&name);
        if !candidate.exists() {
            return Ok((candidate, name));
        }
    }

    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|err| anyhow!("failed to read system time: {err}"))?
        .as_secs();
    let name = format!("run-{suffix}");
    Ok((run_root.join(&name), name))
}

fn write_latest_run(run_root: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_root)
        .with_context(|| format!("failed to create run directory {}", run_root.display()))?;
    let path = run_root.join("latest");
    fs::write(&path, run_name)
        .with_context(|| format!("failed to write latest run {}", path.display()))?;
    Ok(())
}

#[derive(Serialize)]
struct WebConfigOutput {
    run_name: String,
    block_size: usize,
    overrides: ModelOverrides,
}

fn write_run_config(config: &TrainingConfig, run_dir: &Path, run_name: &str) -> Result<()> {
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;

    let block_size = config
        .model
        .block_size
        .unwrap_or(config.training.block_size)
        .max(1);
    let output = WebConfigOutput {
        run_name: run_name.to_string(),
        block_size,
        overrides: config.model.clone(),
    };
    let payload =
        serde_json::to_string_pretty(&output).context("failed to serialize web config")?;
    let path = run_dir.join("config.json");
    fs::write(&path, payload)
        .with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}
