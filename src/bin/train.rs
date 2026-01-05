#![recursion_limit = "256"]

use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
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
use burn_train::metric::{Adaptor, ItemLazy, LearningRateMetric, LossInput, LossMetric};
use burn_train::{
    LearnerBuilder,
    LearningStrategy,
    TrainingResult,
    TrainOutput,
    TrainStep,
    ValidStep,
};
use burn_wgpu::Wgpu;
use tracing::info;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

use burn::record::{BinFileRecorder, FullPrecisionSettings};

use burn_dragon_hatchling::wgpu::init_runtime;
use burn_dragon_hatchling::{
    BDH, BDHConfig, Dataset, DatasetConfig, DatasetSplit, DinoFeatureStore, ImageNetAugmentations,
    ImageNetBatch, ImageNetDataLoader, ImageNetDataset, ImageNetDatasetConfig, ImageNetSplit,
    ImagenetteVariant, LearningRateScheduleConfig, OptimizerConfig, RandomDataLoader,
    SequenceBatch, TrainingConfig, TrainingHyperparameters, VisionDatasetConfig,
    VisionDatasetDownloadConfig, VisionDragonHatchling, VisionDragonHatchlingConfig,
    VisionDistillationLossConfig, VisionLejepaConfig, VisionNormalize, VisionTeacherConfig,
    VisionTeacherVariant, VisionTrainingConfig, VisionTrainingHyperparameters,
    VisionTrainingModeConfig, build_dataset, build_model_config, language_model_loss,
    load_training_config, load_vision_training_config, vision_distillation_loss,
};
use burn_dino::correctness::load_model_from_checkpoint;
use burn_dino::model::dino::{DinoVisionTransformer, DinoVisionTransformerConfig};
use serde::Serialize;

#[derive(Parser, Debug)]
#[command(author, version, about = "Train the Baby Dragon Hatchling model")]
struct Cli {
    #[command(flatten)]
    train: TrainArgs,
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(ClapArgs, Debug)]
struct TrainArgs {
    /// Additional configuration files applied in order (later files override earlier ones).
    #[arg(short = 'c', long = "config", value_name = "PATH", global = true)]
    config: Vec<PathBuf>,
    /// Backend to use for training.
    #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
    backend: BackendArg,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Build the character-level vocabulary and exit.
    BuildVocab,
    /// Train the vision model (distill or LeJEPA).
    Vision,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum BackendArg {
    Cuda,
    Wgpu,
}

static FAST_TRAIN: AtomicBool = AtomicBool::new(false);
const LEJEPA_EPS: f32 = 1e-6;

fn fast_train_enabled() -> bool {
    FAST_TRAIN.load(Ordering::Relaxed)
}

#[derive(Clone)]
struct LanguageModelOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
}

impl<B: BackendTrait> LanguageModelOutput<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: BackendTrait> ItemLazy for LanguageModelOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

struct LanguageModelTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LanguageModelTrainItem<B> {
    fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: AutodiffBackend> ItemLazy for LanguageModelTrainItem<B> {
    type ItemSync = LanguageModelOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        LanguageModelOutput::new(self.loss.inner())
    }
}

type ValidBackend<B> = <B as AutodiffBackend>::InnerBackend;

#[derive(Clone)]
struct VisionArtifactInput<B: BackendTrait> {
    views: Option<Tensor<B, 5>>,
    patch_norms: Option<Tensor<B, 3>>,
    probe_logits: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1, Int>>,
}

impl<B: BackendTrait> VisionArtifactInput<B> {
    fn empty() -> Self {
        Self {
            views: None,
            patch_norms: None,
            probe_logits: None,
            labels: None,
        }
    }
}

#[derive(Clone)]
struct VisionOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionOutput<B> {
    fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
        artifacts: Option<VisionArtifactInput<B>>,
    ) -> Self {
        Self {
            loss,
            inv_loss,
            sigreg_loss,
            recon_loss,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }
}

impl<B: BackendTrait> ItemLazy for VisionOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Clone)]
struct InvLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> InvLossInput<B> {
    fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
struct SigRegLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SigRegLossInput<B> {
    fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
struct ReconLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ReconLossInput<B> {
    fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
struct ProbeLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeLossInput<B> {
    fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
struct ProbeAccInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeAccInput<B> {
    fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

impl<B: BackendTrait> Adaptor<InvLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> InvLossInput<B> {
        InvLossInput::new(self.inv_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SigRegLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> SigRegLossInput<B> {
        SigRegLossInput::new(self.sigreg_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconLossInput<B> {
        ReconLossInput::new(self.recon_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeLossInput<B> {
        ProbeLossInput::new(self.probe_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeAccInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeAccInput<B> {
        ProbeAccInput::new(self.probe_acc.clone())
    }
}

impl<B: BackendTrait> Adaptor<VisionArtifactInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> VisionArtifactInput<B> {
        self.artifacts.clone().unwrap_or_else(VisionArtifactInput::empty)
    }
}

struct VisionTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
}

impl<B: AutodiffBackend> VisionTrainItem<B> {
    fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
    ) -> Self {
        Self {
            loss,
            inv_loss,
            sigreg_loss,
            recon_loss,
            probe_loss,
            probe_acc,
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for VisionTrainItem<B> {
    type ItemSync = VisionOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        VisionOutput::new(
            self.loss.inner(),
            self.inv_loss.inner(),
            self.sigreg_loss.inner(),
            self.recon_loss.inner(),
            self.probe_loss.inner(),
            self.probe_acc.inner(),
            None,
        )
    }
}

trait ScalarValue<B: BackendTrait> {
    fn value(&self) -> Tensor<B, 1>;
}

impl<B: BackendTrait> ScalarValue<B> for InvLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SigRegLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ReconLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeAccInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

struct ScalarMetric<B: BackendTrait, I: ScalarValue<B>> {
    name: Arc<String>,
    last: f64,
    _marker: std::marker::PhantomData<(B, I)>,
}

impl<B: BackendTrait, I: ScalarValue<B>> Clone for ScalarMetric<B, I> {
    fn clone(&self) -> Self {
        Self {
            name: Arc::clone(&self.name),
            last: self.last,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B>> ScalarMetric<B, I> {
    fn new(name: &str) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            last: 0.0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Metric
    for ScalarMetric<B, I>
{
    type Input = I;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        let value = item
            .value()
            .mean()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("metric value");
        let value = value.first().copied().unwrap_or(0.0) as f64;
        self.last = value;
        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            burn_train::metric::format_float(value, 4),
            value.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last = 0.0;
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Numeric
    for ScalarMetric<B, I>
{
    fn value(&self) -> burn_train::metric::NumericEntry {
        burn_train::metric::NumericEntry::Value(self.last)
    }
}

#[derive(Clone)]
struct VisionArtifactMetric<B: BackendTrait> {
    name: Arc<String>,
    output_dir: PathBuf,
    every: usize,
    mean: [f32; 3],
    std: [f32; 3],
    overwrite: bool,
    _marker: std::marker::PhantomData<B>,
}

impl<B: BackendTrait> VisionArtifactMetric<B> {
    fn new(
        output_dir: PathBuf,
        every: usize,
        mean: [f32; 3],
        std: [f32; 3],
        overwrite: bool,
    ) -> Self {
        Self {
            name: Arc::new("vision_artifacts".to_string()),
            output_dir,
            every,
            mean,
            std,
            overwrite,
            _marker: std::marker::PhantomData,
        }
    }

    fn denormalize_channel(&self, value: f32, channel: usize) -> u8 {
        let mut value = value * self.std[channel] + self.mean[channel];
        value = value.clamp(0.0, 1.0);
        (value * 255.0).round() as u8
    }
}

impl<B: BackendTrait> burn_train::metric::Metric for VisionArtifactMetric<B> {
    type Input = VisionArtifactInput<B>;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        if self.every == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "disabled".to_string(),
                "0".to_string(),
            );
        }
        if metadata.iteration % self.every != 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "skip".to_string(),
                "0".to_string(),
            );
        }
        let Some(views) = &item.views else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_views".to_string(),
                "0".to_string(),
            );
        };
        let Some(patch_norms) = &item.patch_norms else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_patch_norms".to_string(),
                "0".to_string(),
            );
        };

        let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
        if batch == 0 || view_count == 0 || channels == 0 || height == 0 || width == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "empty".to_string(),
                "0".to_string(),
            );
        }

        let [norm_batch, grid_h, grid_w] = patch_norms.shape().dims::<3>();
        if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "empty_norms".to_string(),
                "0".to_string(),
            );
        }

        let views_vec = match views
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        {
            Ok(vec) => vec,
            Err(_) => {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "view_copy_failed".to_string(),
                    "0".to_string(),
                );
            }
        };
        let patch_vec = match patch_norms
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        {
            Ok(vec) => vec,
            Err(_) => {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "patch_copy_failed".to_string(),
                    "0".to_string(),
                );
            }
        };
        let probe_preds = if let (Some(logits), Some(labels)) =
            (&item.probe_logits, &item.labels)
        {
            let preds = logits
                .clone()
                .argmax(1)
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .ok();
            let labels = labels
                .clone()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .ok();
            preds.zip(labels)
        } else {
            None
        };

        if let Err(err) = fs::create_dir_all(&self.output_dir) {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                format!("mkdir_failed: {err}"),
                "0".to_string(),
            );
        }

        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
        if heat_patch_h == 0 || heat_patch_w == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "heatmap_scale_invalid".to_string(),
                "0".to_string(),
            );
        }

        let mut saved = 0usize;
        let mut log_lines = Vec::new();
        let width_total = width * (view_count + 1);
        for batch_idx in 0..batch {
            let mut canvas = vec![0u8; width_total * height * 3];
            for view_idx in 0..view_count {
                for y in 0..height {
                    for x in 0..width {
                        let base = (((batch_idx * view_count + view_idx) * channels + 0) * height
                            + y)
                            * width
                            + x;
                        let r = self.denormalize_channel(views_vec[base], 0);
                        let g = self.denormalize_channel(
                            views_vec[base + height * width],
                            1,
                        );
                        let b = self.denormalize_channel(
                            views_vec[base + 2 * height * width],
                            2,
                        );
                        let out_x = view_idx * width + x;
                        let offset = (y * width_total + out_x) * 3;
                        canvas[offset] = r;
                        canvas[offset + 1] = g;
                        canvas[offset + 2] = b;
                    }
                }
            }

            let patch_offset = batch_idx * grid_h * grid_w;
            let patch_slice = &patch_vec[patch_offset..patch_offset + grid_h * grid_w];
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for value in patch_slice {
                min_val = min_val.min(*value);
                max_val = max_val.max(*value);
            }
            let denom = (max_val - min_val).max(LEJEPA_EPS);
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let value = (patch_slice[gy * grid_w + gx] - min_val) / denom;
                    let pix = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    for y in (gy * heat_patch_h)..((gy + 1) * heat_patch_h) {
                        for x in (gx * heat_patch_w)..((gx + 1) * heat_patch_w) {
                            let out_x = view_count * width + x;
                            let offset = (y * width_total + out_x) * 3;
                            canvas[offset] = pix;
                            canvas[offset + 1] = pix;
                            canvas[offset + 2] = pix;
                        }
                    }
                }
            }

            let is_correct = probe_preds.as_ref().and_then(|(preds, labels)| {
                let pred = preds.get(batch_idx)?;
                let label = labels.get(batch_idx)?;
                Some(pred == label)
            });
            if let Some(is_correct) = is_correct {
                let (r, g, b) = if is_correct {
                    (0u8, 200u8, 0u8)
                } else {
                    (200u8, 0u8, 0u8)
                };
                for x in 0..width_total {
                    let top = x * 3;
                    canvas[top] = r;
                    canvas[top + 1] = g;
                    canvas[top + 2] = b;
                    let bottom = ((height - 1) * width_total + x) * 3;
                    canvas[bottom] = r;
                    canvas[bottom + 1] = g;
                    canvas[bottom + 2] = b;
                }
                for y in 0..height {
                    let left = (y * width_total) * 3;
                    canvas[left] = r;
                    canvas[left + 1] = g;
                    canvas[left + 2] = b;
                    let right = (y * width_total + (width_total - 1)) * 3;
                    canvas[right] = r;
                    canvas[right + 1] = g;
                    canvas[right + 2] = b;
                }
            }

            if let Some(image) = image::RgbImage::from_vec(
                width_total as u32,
                height as u32,
                canvas,
            ) {
                let filename = if self.overwrite {
                    format!("sample_{:02}.png", batch_idx)
                } else if let Some((preds, labels)) = &probe_preds {
                    let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                    let label = labels.get(batch_idx).copied().unwrap_or(-1);
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}_pred_{pred}_label_{label}.png",
                        metadata.iteration,
                        batch_idx
                    )
                } else {
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}.png",
                        metadata.iteration,
                        batch_idx
                    )
                };
                let path = self.output_dir.join(filename);
                if image.save(path).is_ok() {
                    saved += 1;
                }
            }

            if let Some((preds, labels)) = &probe_preds {
                let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                let label = labels.get(batch_idx).copied().unwrap_or(-1);
                let correct = if pred == label { "1" } else { "0" };
                log_lines.push(format!(
                    "{},{},{},{},{}",
                    metadata.iteration, batch_idx, pred, label, correct
                ));
            }
        }

        if !log_lines.is_empty() {
            let log_path = self.output_dir.join("vision_artifacts.log");
            let mut contents = String::new();
            contents.push_str("iteration,batch_idx,pred,label,correct\n");
            contents.push_str(&log_lines.join("\n"));
            if self.overwrite {
                let _ = fs::write(log_path, contents);
            } else if let Ok(mut file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                let _ = writeln!(file, "{contents}");
            }
        }

        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            format!("saved={saved}"),
            saved.to_string(),
        )
    }

    fn clear(&mut self) {}
}

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

pub fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err:#}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
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
            let lejepa_diagnostics = Some(VisionLejepaDiagnostics {
                config: model.as_ref().expect("model").config.clone(),
                normalize_mean: config.augment.normalize_mean,
                normalize_std: config.augment.normalize_std,
            });
            match scheduler {
                ResolvedLrScheduler::Constant(lr) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    lr,
                    lejepa_diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Cosine(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    lejepa_diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Linear(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    lejepa_diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Exponential(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    lejepa_diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Step(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    lejepa_diagnostics.clone(),
                )?,
                ResolvedLrScheduler::Noam(scheduler) => train_vision_with_scheduler(
                    &context,
                    model.take().expect("model initialized"),
                    optim.take().expect("optimizer initialized"),
                    scheduler,
                    lejepa_diagnostics.clone(),
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
struct VisionLejepaDiagnostics {
    config: VisionLejepaConfig,
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
    lejepa_diagnostics: Option<VisionLejepaDiagnostics>,
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

    if let Some(diagnostics) = &lejepa_diagnostics {
        builder = builder
            .metric_train_numeric(ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new(
                "lejepa_inv_loss",
            ))
            .metric_valid_numeric(ScalarMetric::<ValidBackend<B>, InvLossInput<ValidBackend<B>>>::new(
                "lejepa_inv_loss",
            ))
            .metric_train_numeric(
                ScalarMetric::<ValidBackend<B>, SigRegLossInput<ValidBackend<B>>>::new(
                    "lejepa_sigreg_loss",
                ),
            )
            .metric_valid_numeric(
                ScalarMetric::<ValidBackend<B>, SigRegLossInput<ValidBackend<B>>>::new(
                    "lejepa_sigreg_loss",
                ),
            );
        if diagnostics.config.recon_weight > 0.0 {
            builder = builder
                .metric_train_numeric(
                    ScalarMetric::<ValidBackend<B>, ReconLossInput<ValidBackend<B>>>::new(
                        "lejepa_recon_loss",
                    ),
                )
                .metric_valid_numeric(
                    ScalarMetric::<ValidBackend<B>, ReconLossInput<ValidBackend<B>>>::new(
                        "lejepa_recon_loss",
                    ),
                );
        }
        builder = builder
            .metric_train_numeric(
                ScalarMetric::<ValidBackend<B>, ProbeLossInput<ValidBackend<B>>>::new(
                    "lejepa_probe_loss",
                ),
            )
            .metric_valid_numeric(
                ScalarMetric::<ValidBackend<B>, ProbeLossInput<ValidBackend<B>>>::new(
                    "lejepa_probe_loss",
                ),
            )
            .metric_train_numeric(
                ScalarMetric::<ValidBackend<B>, ProbeAccInput<ValidBackend<B>>>::new(
                    "lejepa_probe_acc",
                ),
            )
            .metric_valid_numeric(
                ScalarMetric::<ValidBackend<B>, ProbeAccInput<ValidBackend<B>>>::new(
                    "lejepa_probe_acc",
                ),
            );

        if diagnostics.config.artifact_every > 0 {
            let artifact_dir = env.run_dir.join("artifacts");
            builder = builder.metric_valid(VisionArtifactMetric::<ValidBackend<B>>::new(
                artifact_dir,
                diagnostics.config.artifact_every,
                diagnostics.normalize_mean,
                diagnostics.normalize_std,
                diagnostics.config.artifact_overwrite,
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

fn patchify<B: BackendTrait>(images: Tensor<B, 4>, patch_size: usize) -> Tensor<B, 3> {
    let [batch, channels, height, width] = images.shape().dims::<4>();
    assert!(
        height.is_multiple_of(patch_size) && width.is_multiple_of(patch_size),
        "patchify expects height/width divisible by patch size"
    );
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    images
        .reshape([batch, channels, grid_h, patch_size, grid_w, patch_size])
        .swap_dims(1, 2)
        .swap_dims(2, 4)
        .swap_dims(3, 4)
        .reshape([
            batch,
            grid_h * grid_w,
            channels * patch_size * patch_size,
        ])
}

fn unpatchify<B: BackendTrait>(
    patches: Tensor<B, 3>,
    patch_size: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Tensor<B, 4> {
    let [batch, tokens, patch_dim] = patches.shape().dims::<3>();
    assert!(patch_dim > 0, "unpatchify expects non-empty patch dim");
    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    assert!(
        grid_h * grid_w == tokens,
        "unpatchify expects token count to match grid"
    );
    patches
        .reshape([batch, grid_h, grid_w, channels, patch_size, patch_size])
        .swap_dims(3, 4)
        .swap_dims(2, 4)
        .swap_dims(1, 2)
        .reshape([batch, channels, height, width])
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
    let device = proj.device();
    let [views, batch, dim] = proj.shape().dims::<3>();
    if views == 0 || batch == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }

    let knots = config.sigreg_knots.max(2);
    let t_max = config.sigreg_t_max.max(LEJEPA_EPS);
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

    let sketch_dim = config.sigreg_proj_dim.max(1);
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

    Some(VisionArtifactInput {
        views: Some(views_tensor),
        patch_norms,
        probe_logits,
        labels,
    })
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
    use burn_dragon_hatchling::ContextStrategyConfig;

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
    overrides: burn_dragon_hatchling::ModelOverrides,
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
