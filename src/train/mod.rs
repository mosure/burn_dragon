use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
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
use burn::module::{AutodiffModule, Module, Param};
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::optim::adaptor::OptimizerAdaptor;
use burn::optim::{AdamW, AdamWConfig, GradientsParams, LearningRate};
use burn::tensor::Distribution as TensorDistribution;
use burn::tensor::activation;
use burn::tensor::{Int, Tensor, TensorData};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_autodiff::Autodiff;
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

use crate::wgpu::init_runtime;
use crate::{
    BDH, BDHConfig, Dataset, DatasetConfig, DatasetSplit, DinoFeatureStore, ImageNetAugmentations,
    ImageNetBatch, ImageNetDataLoader, ImageNetDataset, ImageNetDatasetConfig, ImageNetSplit,
    ImagenetteVariant, LearningRateScheduleConfig, ModelOverrides, OptimizerConfig, PatchGrid,
    RandomDataLoader, SequenceBatch, TrainingConfig, TrainingHyperparameters,
    VisionArtifactOutputMode, VisionDatasetConfig, VisionDatasetDownloadConfig,
    VisionDragonHatchling, VisionDragonHatchlingConfig, VisionDistillationLossConfig,
    VisionLejepaConfig, VisionMaeConfig, VisionNormalize, VisionPyramidMode, VisionSaccadeConfig,
    VisionTeacherConfig, VisionTeacherVariant, VisionTrainingConfig,
    VisionTrainingHyperparameters, VisionTrainingModeConfig, build_dataset, build_model_config,
    language_model_loss, load_training_config, load_vision_training_config, patchify, unpatchify,
    vision_distillation_loss,
};
use burn_dino::correctness::load_model_from_checkpoint;
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use serde::Serialize;

mod metrics;
mod artifacts;

use metrics::{
    InvLossInput, LanguageModelOutput, LanguageModelTrainItem, ProbeAccInput, ProbeLossInput,
    ReconLossInput, ScalarMetric, SigRegLossInput, VisionArtifactInput, VisionArtifactMetric,
    VisionOutput, VisionTrainItem,
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
const SACCADE_SIGMA_MIN: f32 = 0.03;
const SACCADE_SIGMA_MAX: f32 = 0.5;
const SACCADE_LEVEL_TEMP: f32 = 8.0;
const SACCADE_RING_WIDTH: f32 = 0.02;
const SACCADE_RING_INTENSITY: f32 = 2.0;

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
            let output = self.forward_view_group(&collected.global, steps);
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
                    self.recon_group_loss(&collected.global, steps, true, randomize_mask);
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
            let output = self.forward_view_group(&collected.local, steps);
            if heatmap_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                heatmap_source =
                    Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if recon_enabled {
                let (loss_sum, mask_sum, _) =
                    self.recon_group_loss(&collected.local, steps, false, randomize_mask);
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
        let artifacts = build_lejepa_artifacts(
            &self.config,
            &artifact_views,
            None,
            heatmap_source,
            probe_primary,
            Some(labels),
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
        let embed_out = self.model.forward_tokens_embed_steps(masked_tokens, steps);

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
    ) -> ViewGroupOutput<B> {
        let view_count = views.len();
        let [batch, _, _, _] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed(stacked);
        let embed_out = self.model.forward_tokens_embed_steps(patch.tokens, steps);
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
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionMaeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, artifacts) =
            self.recon_loss(images, steps, randomize_mask, capture_artifacts);
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
        let embed_out = self.model.forward_tokens_embed_steps(masked_tokens, steps);

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

#[derive(Module, Debug)]
struct VisionSaccadeModel<B: BackendTrait> {
    model: VisionDragonHatchling<B>,
    recon: VisionReconstructionHead<B>,
    trajectory_token: Param<Tensor<B, 2>>,
    trajectory_query: Param<Tensor<B, 2>>,
    eye_token: Param<Tensor<B, 2>>,
    input_proj: VisionSaccadeProjection<B>,
    saccade_proj: VisionSaccadeProjection<B>,
    residual_proj: VisionSaccadeProjection<B>,
    saccade_head: VisionSaccadeHead<B>,
    config: VisionSaccadeConfig,
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
            [config.trajectory_tokens.max(1), embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let trajectory_query = Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
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
        let saccade_proj = VisionSaccadeProjection::new(embed_dim, embed_dim, device);
        let residual_proj = VisionSaccadeProjection::new(embed_dim, embed_dim, device);
        let saccade_head = VisionSaccadeHead::new(embed_dim, device);
        Self {
            model,
            recon,
            trajectory_token: Param::from_tensor(trajectory_token),
            trajectory_query: Param::from_tensor(trajectory_query),
            eye_token: Param::from_tensor(eye_token),
            input_proj,
            saccade_proj,
            residual_proj,
            saccade_head,
            config,
            rollout,
        }
    }

    fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionSaccadeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, inv, sigreg, artifacts) =
            self.recon_loss(images, steps, randomize_mask, capture_artifacts);
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let lambda = self.config.lambda.clamp(0.0, 1.0);
        let mut total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
        let recon_weight = self.config.recon_weight.max(0.0);
        if recon_weight > 0.0 {
            total = total + recon.clone().mul_scalar(recon_weight);
        }

        let artifacts = artifacts.and_then(|(views, residual, frames)| {
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
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>, Option<Tensor<B, 5>>)>,
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
        let (input_residuals, input_sample_levels) = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => (input_levels.clone(), input_levels.clone()),
            VisionPyramidMode::Laplacian => {
                let residuals = self.decompose_pyramid(&input_levels, &grids);
                (residuals.clone(), residuals)
            }
        };
        let traj_len = self.config.trajectory_tokens.max(1);
        let num_eyes = self.config.num_eyes.max(1);

        let base_traj = self
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let mut trajs = vec![base_traj; num_eyes];
        let traj_query = self
            .trajectory_query
            .val()
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, traj_len);
        let mut state_levels: Vec<Tensor<B, 3>> = input_residuals
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
            .collect();
        let rollout_steps = steps.max(1);
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
        for step_idx in 0..rollout_steps {
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
            let mut next_trajs = Vec::with_capacity(num_eyes);
            for eye_idx in 0..num_eyes {
                let traj = trajs[eye_idx].clone();
                let eye_embed = self
                    .eye_token
                    .val()
                    .slice_dim(0, eye_idx..eye_idx + 1)
                    .reshape([1, 1, embed_dim])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, traj_len);
                let traj_with_eye = traj.clone() + eye_embed.clone();
                let params = self.saccade_head.forward(traj_with_eye.clone());
                let (mean, sigma) = self.decode_saccade_params(params);
                if let Some(steps) = &mut step_traj {
                    let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
                    let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
                    steps.push((mean_step.detach(), sigma_step.detach()));
                }
                let weights = self.mip_gaussian_weights(&mip_levels, mean, sigma);
                let input_context = self.mip_weighted_sum(&input_sample_levels, &weights);
                let state_context = self.mip_weighted_sum(&state_composed, &weights);
                let input_tokens = self.input_proj.forward(input_context) + state_context;
                let tokens_in = traj_with_eye + input_tokens + traj_query.clone();
                let out_tokens = self
                    .model
                    .forward_tokens_embed_steps(tokens_in, steps)
                    .patch_tokens;
                let residual = self.residual_proj.forward(out_tokens.clone());
                let next_traj = self.saccade_proj.forward(out_tokens);
                for (update, weights) in updates.iter_mut().zip(weights.iter()) {
                    let update_eye = weights.clone().swap_dims(1, 2).matmul(residual.clone());
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
                    let mut frame = recon_view;
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            frame = overlay;
                        }
                    }
                    frames.push(frame);
                }
            }
            trajs = next_trajs;
            if step_idx + 1 < rollout_steps {
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
            let mut views = vec![images.clone(), recon_view];
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
                            views.push(view);
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
                let mut stacked = Vec::with_capacity(frames.len());
                for frame in frames {
                    stacked.push(frame.unsqueeze_dim::<5>(1));
                }
                Some(Tensor::cat(stacked, 1))
            });
            Some((views, residual, frames))
        } else {
            None
        };

        (loss_sum, mask_sum, inv, sigreg, artifacts)
    }

    fn decode_saccade_params(
        &self,
        params: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mean = activation::sigmoid(params.clone().slice_dim(2, 0..2));
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

    fn upsample_tokens(
        &self,
        tokens: Tensor<B, 3>,
        from: PatchGrid,
        to: PatchGrid,
    ) -> Tensor<B, 3> {
        let [batch, _, dim] = tokens.shape().dims::<3>();
        if from.height == 0 || from.width == 0 || to.height == 0 || to.width == 0 {
            return Tensor::<B, 3>::zeros([batch, to.num_patches().max(1), dim], &tokens.device());
        }
        if to.height % from.height != 0 || to.width % from.width != 0 {
            return tokens;
        }
        let scale_h = to.height / from.height;
        let scale_w = to.width / from.width;
        if scale_h == 0 || scale_w == 0 {
            return tokens;
        }
        tokens
            .reshape([batch, from.height, from.width, dim])
            .repeat_dim(1, scale_h)
            .repeat_dim(2, scale_w)
            .reshape([batch, to.height * to.width, dim])
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

        let level_refs = build_level_sigmas::<B>(level_count, &device);
        let mean_flat = mean.reshape([batch * traj_tokens, 2]);
        let sigma_flat = sigma.reshape([batch * traj_tokens, 1]);
        let diff = sigma_flat.clone() - level_refs;
        let level_scores = diff
            .powf_scalar(2.0)
            .mul_scalar(-SACCADE_LEVEL_TEMP);
        let level_weights = activation::softmax(level_scores, 1);

        let mut weights_out = Vec::with_capacity(level_count);
        for (level_idx, level) in levels.iter().enumerate() {
            let coords = build_level_coords::<B>(level.grid, &device);
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
            let scaled = dist2 / sigma2.mul_scalar(2.0);
            let spatial = activation::softmax(scaled.mul_scalar(-1.0), 1);
            let spatial = spatial.reshape([batch, traj_tokens, tokens_len]);
            let level_weight = level_weights
                .clone()
                .slice_dim(1, level_idx..level_idx + 1)
                .reshape([batch, traj_tokens, 1]);
            weights_out.push(spatial * level_weight);
        }

        weights_out
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
            context = context + weights.clone().matmul(tokens.clone());
        }
        context
    }

    #[cfg(test)]
    fn apply_mip_residual(
        &self,
        state_levels: &mut [Tensor<B, 3>],
        weights: &[Tensor<B, 3>],
        residual: Tensor<B, 3>,
    ) {
        for (state, weights) in state_levels.iter_mut().zip(weights.iter()) {
            let update = weights.clone().swap_dims(1, 2).matmul(residual.clone());
            *state = state.clone() + update;
        }
    }
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
        let output = self.model.forward_images_steps(images, rollout_steps);
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

        let output = self
            .model
            .forward_images_steps(images, self.rollout.max_steps);
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
        let losses = self.forward_losses(batch, rollout_steps, true);
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
        let losses = self.forward_losses(batch, self.rollout.max_steps, false);

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
        let losses = self.forward_losses(batch, rollout_steps, true, false);
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
        let losses = self.forward_losses(batch, self.rollout.max_steps, false, true);
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
        let losses = self.forward_losses(batch, rollout_steps, true, false);
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
        let losses = self.forward_losses(batch, self.rollout.max_steps, false, true);
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
        "vision rollout steps: min={}, max={}",
        rollout.min_steps, rollout.max_steps
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
            if saccade.trajectory_tokens == 0 {
                return Err(anyhow!("saccade.trajectory_tokens must be > 0"));
            }
            if saccade.mip_levels == 0 {
                return Err(anyhow!("saccade.mip_levels must be > 0"));
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
    device: &'a B::Device,
    train_loader: Arc<dyn DataLoader<B, ImageNetBatch<B>>>,
    valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>>,
    epochs: usize,
}

#[derive(Clone, Copy, Debug, Module)]
struct VisionRollout {
    min_steps: usize,
    max_steps: usize,
}

impl VisionRollout {
    fn sample_steps(&self) -> usize {
        if self.min_steps >= self.max_steps {
            self.max_steps
        } else {
            thread_rng().gen_range(self.min_steps..=self.max_steps)
        }
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
    if min_steps == 0 || max_steps_cfg == 0 {
        return Err(anyhow!(
            "vision rollout steps must be > 0 (min={min_steps}, max={max_steps_cfg})"
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
    Ok(VisionRollout {
        min_steps,
        max_steps: max_steps_cfg,
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

fn build_lejepa_artifacts<B: BackendTrait>(
    config: &VisionLejepaConfig,
    views: &[Tensor<B, 4>],
    frames: Option<Tensor<B, 5>>,
    first_patch: Option<Tensor<B, 3>>,
    probe_logits: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1, Int>>,
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

fn downsample_image<B: BackendTrait>(images: Tensor<B, 4>) -> Option<Tensor<B, 4>> {
    let [batch, channels, height, width] = images.shape().dims::<4>();
    if height < 2 || width < 2 {
        return None;
    }
    let even_h = height - (height % 2);
    let even_w = width - (width % 2);
    if even_h == 0 || even_w == 0 {
        return None;
    }
    let images = images
        .slice_dim(2, 0..even_h)
        .slice_dim(3, 0..even_w)
        .reshape([batch, channels, even_h / 2, 2, even_w / 2, 2])
        .mean_dim(3)
        .mean_dim(5)
        .reshape([batch, channels, even_h / 2, even_w / 2]);
    Some(images)
}

fn build_level_sigmas<B: BackendTrait>(levels: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = Vec::with_capacity(levels);
    let mut sigma = SACCADE_SIGMA_MIN.max(SACCADE_EPS);
    for _ in 0..levels {
        values.push(sigma.min(SACCADE_SIGMA_MAX));
        sigma *= 2.0;
    }
    Tensor::<B, 1>::from_data(TensorData::new(values, [levels]), device).reshape([1, levels])
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
    let radius = sigma.reshape([batch, 1, 1, 1]);

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
        .mul_scalar(SACCADE_RING_INTENSITY);
    let inv_mask = ring_mask.mul_scalar(-1.0).add_scalar(1.0);
    let overlay = images.mul(inv_mask) + ring_rgb;
    Some(overlay)
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
        let vision_config = VisionDragonHatchlingConfig {
            image_size: 8,
            patch_size: 4,
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
            pos_max_height: 2,
            pos_max_width: 2,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: FusedKernelConfig::default(),
        };
        let model = VisionDragonHatchling::<B>::new(vision_config.clone(), device);
        let saccade_config = VisionSaccadeConfig {
            num_eyes,
            trajectory_tokens: 4,
            mip_levels: 3,
            pyramid_mode: VisionPyramidMode::Laplacian,
            lambda: 0.02,
            sigreg_knots: 5,
            sigreg_t_max: 1.0,
            sigreg_proj_dim: 8,
            recon_weight: 0.0,
            recon_mask_ratio: 0.0,
            recon_hidden_dim: 16,
            artifact_output: VisionArtifactOutputMode::Images,
            artifact_every: 0,
            artifact_max_images: 0,
            artifact_max_views: 0,
            artifact_overwrite: true,
        };
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 2,
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
        steps: usize,
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
        let (input_residuals, input_sample_levels) = match saccade.config.pyramid_mode {
            VisionPyramidMode::Stacked => (input_levels.clone(), input_levels.clone()),
            VisionPyramidMode::Laplacian => {
                let residuals = saccade.decompose_pyramid(&input_levels, &grids);
                (residuals.clone(), residuals)
            }
        };
        let embed_dim = patch.tokens.shape().dims::<3>()[2];
        let traj_len = saccade.config.trajectory_tokens.max(1);
        let base_traj = saccade
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let traj_query = saccade
            .trajectory_query
            .val()
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, traj_len);
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
        let params = saccade.saccade_head.forward(traj_with_eye.clone());
        let (mean, sigma) = saccade.decode_saccade_params(params);
        let weights = saccade.mip_gaussian_weights(&mip_levels, mean, sigma);
        let input_context = saccade.mip_weighted_sum(&input_sample_levels, &weights);
        let state_context = saccade.mip_weighted_sum(&state_composed, &weights);
        let input_tokens = saccade.input_proj.forward(input_context) + state_context;
        let tokens_in = traj_with_eye + input_tokens + traj_query;
        let out_tokens = saccade
            .model
            .forward_tokens_embed_steps(tokens_in, steps)
            .patch_tokens;
        let residual = saccade.residual_proj.forward(out_tokens.clone());
        let next_traj = saccade.saccade_proj.forward(out_tokens);
        let mut updates = Vec::with_capacity(weights.len());
        for weights in &weights {
            updates.push(weights.clone().swap_dims(1, 2).matmul(residual.clone()));
        }
        (next_traj, updates)
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
        let losses = saccade.forward_losses(batch, 2, true, false);
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
        let losses = saccade.forward_losses(batch, 2, true, false);
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

        let (traj0, _) = saccade_eye_step(&saccade, images.clone(), 1, 0);
        let (traj1, _) = saccade_eye_step(&saccade, images, 1, 1);
        let mse = (traj0 - traj1).powf_scalar(2.0).mean();
        assert_mse_above(mse, 0.0);
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
        let (_, updates0) = saccade_eye_step(&saccade, images.clone(), 1, 0);
        let (_, updates1) = saccade_eye_step(&saccade, images, 1, 1);

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
        let losses = saccade.forward_losses(batch, 1, true, false);
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
    fn saccade_step_produces_finite_grads() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = <Backend as BackendTrait>::Device::default();
        let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
        let images =
            Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
        let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
        let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
        let losses = saccade.forward_losses(batch, 1, true, false);
        let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

        let token_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
            .expect("trajectory_token grad");
        let query_grad = grads
            .get::<ValidBackend<Backend>, 2>(saccade.trajectory_query.id)
            .expect("trajectory_query grad");
        assert_tensor_finite(token_grad);
        assert_tensor_finite(query_grad);
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
