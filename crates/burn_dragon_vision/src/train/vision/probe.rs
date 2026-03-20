use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
use burn_ndarray::NdArray;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use super::artifact::{VisionArtifactHeader, push_vision_artifact_markdown_prelude};
use crate::checkpoint::{load_vision_encoder_from_checkpoint, resolve_checkpoint_base};
use crate::config::{
    VisionDistillConfig, VisionTeacherConfig, VisionTeacherDecoderMode, VisionTeacherFeatureConfig,
    VisionTeacherTargetKind, VisionTrainingConfig, VisionTrainingModeConfig,
};
use crate::loss::{VisionDistillationLossConfig, vision_distillation_loss_terms};
use crate::train::{DinoFeatureStore, ImageNetAugmentations, ImageNetSplit, VisionNormalize};
use crate::train::{VisionDistillModel, resolve_vision_rollout};

pub type VisionDistillFeatureProbeBackend = NdArray<f32>;
pub type VisionDistillFeatureProbeDevice =
    <VisionDistillFeatureProbeBackend as burn::tensor::backend::Backend>::Device;
pub const VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION: &str = "vision_distill_feature_probe_v1";
pub const VISION_DISTILL_LINEAR_PROBE_HARNESS_VERSION: &str = "vision_distill_linear_probe_v1";
pub const VISION_DISTILL_DECODE_PROBE_HARNESS_VERSION: &str = "vision_distill_decode_probe_v1";

#[derive(Clone, Serialize)]
pub struct VisionDistillFeatureProbeAccuracyReport {
    pub teacher_acc: f64,
    pub step_accuracies: Vec<VisionDistillFeatureProbeStepAccuracy>,
    pub best_step: usize,
    pub best_acc: f64,
    pub best_minus_s1: f64,
    pub teacher_minus_best_gap: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillFeatureProbeStepAccuracy {
    pub step: usize,
    pub acc: f64,
    pub delta_vs_s1: f64,
    pub teacher_gap: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillFeatureProbeReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub harness_version: &'static str,
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub steps: Vec<usize>,
    pub batch_size: usize,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub subset_seed: u64,
    pub teacher_target: String,
    pub image_size: usize,
    pub feature_dim: usize,
    pub num_classes: usize,
    pub train_records: usize,
    pub val_records: usize,
    pub accuracy: VisionDistillFeatureProbeAccuracyReport,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillFeatureExportReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub step: usize,
    pub batch_size: usize,
    pub max_val_samples: Option<usize>,
    pub image_size: usize,
    pub feature_dim: usize,
    pub records: usize,
    pub labels: Vec<usize>,
    pub teacher_cls: Vec<Vec<f32>>,
    pub student_cls: Vec<Vec<f32>>,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillLinearProbeAccuracyReport {
    pub teacher_acc: f64,
    pub step_accuracies: Vec<VisionDistillLinearProbeStepAccuracy>,
    pub best_step: usize,
    pub best_acc: f64,
    pub best_minus_s1: f64,
    pub teacher_minus_best_gap: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillLinearProbeStepAccuracy {
    pub step: usize,
    pub acc: f64,
    pub delta_vs_s1: f64,
    pub teacher_gap: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillLinearProbeReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub harness_version: &'static str,
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub steps: Vec<usize>,
    pub batch_size: usize,
    pub max_train_samples: Option<usize>,
    pub max_val_samples: Option<usize>,
    pub subset_seed: u64,
    pub teacher_target: String,
    pub train_seed: u64,
    pub epochs: usize,
    pub learning_rate: f32,
    pub weight_decay: f32,
    pub image_size: usize,
    pub teacher_feature_dim: usize,
    pub student_feature_dim: usize,
    pub num_classes: usize,
    pub train_records: usize,
    pub val_records: usize,
    pub accuracy: VisionDistillLinearProbeAccuracyReport,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillDecodeProbeStepMetrics {
    pub step: usize,
    pub distill_total: f64,
    pub distill_patch: f64,
    pub distill_cls: f64,
    pub delta_total_vs_s1: f64,
    pub delta_patch_vs_s1: f64,
    pub delta_cls_vs_s1: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillDecodeProbeReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub harness_version: &'static str,
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub steps: Vec<usize>,
    pub batch_size: usize,
    pub max_val_samples: Option<usize>,
    pub subset_seed: u64,
    pub teacher_target: String,
    pub teacher_weight: f32,
    pub teacher_target_kind: String,
    pub teacher_decoder_mode: String,
    pub image_size: usize,
    pub feature_dim: usize,
    pub val_records: usize,
    pub best_step: usize,
    pub best_total: f64,
    pub best_minus_s1: f64,
    pub steps_report: Vec<VisionDistillDecodeProbeStepMetrics>,
}

#[derive(Clone)]
struct ProbeSample {
    path: PathBuf,
    label: usize,
    teacher_index: usize,
}

struct ProbeSplitSpec {
    samples: Vec<ProbeSample>,
    cls_store: DinoFeatureStore,
}

#[derive(Clone)]
struct ProbeTeacherSpec {
    name: String,
    weight: f32,
    target_kind: VisionTeacherTargetKind,
    decoder_mode: VisionTeacherDecoderMode,
    teacher: VisionTeacherFeatureConfig,
}

#[derive(Clone)]
struct ProbeFeatureMatrix {
    labels: Vec<usize>,
    flat: Vec<f32>,
    dim: usize,
}

struct LinearProbeModel {
    weights: Vec<f32>,
    bias: Vec<f32>,
    classes: usize,
    dim: usize,
}

impl VisionDistillFeatureProbeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Feature Probe",
            &self.artifact,
        );
        writeln!(&mut out, "- checkpoint: {}", self.checkpoint.display()).unwrap();
        writeln!(&mut out, "- harness version: {}", self.harness_version).unwrap();
        writeln!(
            &mut out,
            "- config: {}",
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out, "- teacher target: {}", self.teacher_target).unwrap();
        writeln!(&mut out, "- image size: {}", self.image_size).unwrap();
        writeln!(&mut out, "- feature dim: {}", self.feature_dim).unwrap();
        writeln!(&mut out, "- classes: {}", self.num_classes).unwrap();
        writeln!(&mut out, "- train records: {}", self.train_records).unwrap();
        writeln!(&mut out, "- val records: {}", self.val_records).unwrap();
        writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
        if let Some(limit) = self.max_train_samples {
            writeln!(&mut out, "- max train samples: {}", limit).unwrap();
        }
        if let Some(limit) = self.max_val_samples {
            writeln!(&mut out, "- max val samples: {}", limit).unwrap();
        }
        writeln!(&mut out, "- subset seed: {}", self.subset_seed).unwrap();
        writeln!(
            &mut out,
            "- rollout steps: {}",
            self.steps
                .iter()
                .map(|step| format!("s{step}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "| source | accuracy |").unwrap();
        writeln!(&mut out, "|---|---:|").unwrap();
        writeln!(
            &mut out,
            "| teacher cls | {:.4} |",
            self.accuracy.teacher_acc
        )
        .unwrap();
        for step in &self.accuracy.step_accuracies {
            writeln!(&mut out, "| dragon s{} cls | {:.4} |", step.step, step.acc).unwrap();
        }
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "- best step: s{}", self.accuracy.best_step).unwrap();
        writeln!(&mut out, "- best accuracy: {:.4}", self.accuracy.best_acc).unwrap();
        writeln!(
            &mut out,
            "- best minus s1: {:.4}",
            self.accuracy.best_minus_s1
        )
        .unwrap();
        writeln!(
            &mut out,
            "- teacher minus best gap: {:.4}",
            self.accuracy.teacher_minus_best_gap
        )
        .unwrap();
        out
    }
}

impl VisionDistillLinearProbeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Linear Probe",
            &self.artifact,
        );
        writeln!(&mut out, "- checkpoint: {}", self.checkpoint.display()).unwrap();
        writeln!(&mut out, "- harness version: {}", self.harness_version).unwrap();
        writeln!(
            &mut out,
            "- config: {}",
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out, "- teacher target: {}", self.teacher_target).unwrap();
        writeln!(&mut out, "- image size: {}", self.image_size).unwrap();
        writeln!(
            &mut out,
            "- teacher feature dim: {}",
            self.teacher_feature_dim
        )
        .unwrap();
        writeln!(
            &mut out,
            "- student feature dim: {}",
            self.student_feature_dim
        )
        .unwrap();
        writeln!(&mut out, "- classes: {}", self.num_classes).unwrap();
        writeln!(&mut out, "- train records: {}", self.train_records).unwrap();
        writeln!(&mut out, "- val records: {}", self.val_records).unwrap();
        writeln!(&mut out, "- feature batch size: {}", self.batch_size).unwrap();
        writeln!(&mut out, "- probe epochs: {}", self.epochs).unwrap();
        writeln!(&mut out, "- probe learning rate: {:.5}", self.learning_rate).unwrap();
        writeln!(&mut out, "- probe weight decay: {:.6}", self.weight_decay).unwrap();
        writeln!(&mut out, "- subset seed: {}", self.subset_seed).unwrap();
        writeln!(&mut out, "- train seed: {}", self.train_seed).unwrap();
        if let Some(limit) = self.max_train_samples {
            writeln!(&mut out, "- max train samples: {}", limit).unwrap();
        }
        if let Some(limit) = self.max_val_samples {
            writeln!(&mut out, "- max val samples: {}", limit).unwrap();
        }
        writeln!(
            &mut out,
            "- rollout steps: {}",
            self.steps
                .iter()
                .map(|step| format!("s{step}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "| source | top-1 acc |").unwrap();
        writeln!(&mut out, "|---|---:|").unwrap();
        writeln!(
            &mut out,
            "| teacher linear probe | {:.4} |",
            self.accuracy.teacher_acc
        )
        .unwrap();
        for step in &self.accuracy.step_accuracies {
            writeln!(
                &mut out,
                "| dragon s{} linear probe | {:.4} |",
                step.step, step.acc
            )
            .unwrap();
        }
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "- best step: s{}", self.accuracy.best_step).unwrap();
        writeln!(&mut out, "- best accuracy: {:.4}", self.accuracy.best_acc).unwrap();
        writeln!(
            &mut out,
            "- best minus s1: {:.4}",
            self.accuracy.best_minus_s1
        )
        .unwrap();
        writeln!(
            &mut out,
            "- teacher minus best gap: {:.4}",
            self.accuracy.teacher_minus_best_gap
        )
        .unwrap();
        out
    }
}

impl VisionDistillDecodeProbeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Decode Probe",
            &self.artifact,
        );
        writeln!(&mut out, "- checkpoint: {}", self.checkpoint.display()).unwrap();
        writeln!(&mut out, "- harness version: {}", self.harness_version).unwrap();
        writeln!(
            &mut out,
            "- config: {}",
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out, "- teacher target: {}", self.teacher_target).unwrap();
        writeln!(&mut out, "- teacher weight: {:.4}", self.teacher_weight).unwrap();
        writeln!(
            &mut out,
            "- teacher target kind: {}",
            self.teacher_target_kind
        )
        .unwrap();
        writeln!(
            &mut out,
            "- teacher decoder mode: {}",
            self.teacher_decoder_mode
        )
        .unwrap();
        writeln!(&mut out, "- image size: {}", self.image_size).unwrap();
        writeln!(&mut out, "- feature dim: {}", self.feature_dim).unwrap();
        writeln!(&mut out, "- val records: {}", self.val_records).unwrap();
        writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
        if let Some(limit) = self.max_val_samples {
            writeln!(&mut out, "- max val samples: {}", limit).unwrap();
        }
        writeln!(&mut out, "- subset seed: {}", self.subset_seed).unwrap();
        writeln!(
            &mut out,
            "- rollout steps: {}",
            self.steps
                .iter()
                .map(|step| format!("s{step}"))
                .collect::<Vec<_>>()
                .join(", ")
        )
        .unwrap();
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "| source | total | patch | cls |").unwrap();
        writeln!(&mut out, "|---|---:|---:|---:|").unwrap();
        for step in &self.steps_report {
            writeln!(
                &mut out,
                "| dragon s{} -> {} | {:.4} | {:.4} | {:.4} |",
                step.step,
                self.teacher_target,
                step.distill_total,
                step.distill_patch,
                step.distill_cls
            )
            .unwrap();
        }
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "- best step: s{}", self.best_step).unwrap();
        writeln!(&mut out, "- best total: {:.4}", self.best_total).unwrap();
        writeln!(&mut out, "- best minus s1: {:.4}", self.best_minus_s1).unwrap();
        out
    }
}

pub fn run_vision_distill_feature_probe(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
) -> Result<VisionDistillFeatureProbeReport> {
    run_vision_distill_feature_probe_for_teacher_with_seed(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        0,
        None,
    )
}

pub fn run_vision_distill_feature_probe_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
) -> Result<VisionDistillFeatureProbeReport> {
    run_vision_distill_feature_probe_for_teacher_with_seed(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        subset_seed,
        None,
    )
}

pub fn run_vision_distill_feature_probe_for_teacher_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    teacher_target: Option<&str>,
) -> Result<VisionDistillFeatureProbeReport> {
    config.validate()?;
    let device = VisionDistillFeatureProbeDevice::default();
    let model = load_vision_encoder_from_checkpoint::<VisionDistillFeatureProbeBackend>(
        checkpoint,
        None,
        config_paths,
        &device,
    )?;
    let vision = config.vision.build();
    let batch_size = batch_size.max(1);
    let mut steps = steps.iter().map(|step| (*step).max(1)).collect::<Vec<_>>();
    steps.sort_unstable();
    steps.dedup();

    let teacher_spec = resolve_probe_teacher_spec(config, teacher_target)?;
    let (train_spec, val_spec, num_classes) = load_probe_split_specs_for_teacher(
        config,
        &teacher_spec,
        max_train_samples,
        max_val_samples,
        subset_seed,
    )?;
    let val_aug = build_val_augmentations(config);
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);

    let teacher_centroids = build_teacher_centroids(&train_spec, num_classes, batch_size, &device)?;
    let teacher_acc = eval_teacher_accuracy(&teacher_centroids, &val_spec, batch_size, &device)?;
    let mut step_accuracies = Vec::with_capacity(steps.len());
    for step in &steps {
        let centroids = build_dragon_centroids(
            &model,
            vision.projection_dim,
            &train_spec.samples,
            &val_aug,
            normalize,
            *step,
            batch_size,
            &device,
        )?;
        let acc = eval_dragon_accuracy(
            &centroids,
            &model,
            vision.projection_dim,
            &val_spec.samples,
            &val_aug,
            normalize,
            *step,
            batch_size,
            &device,
        )?;
        step_accuracies.push((*step, acc));
    }
    let s1_acc = step_accuracies
        .iter()
        .find(|(step, _)| *step == 1)
        .map(|(_, acc)| *acc)
        .ok_or_else(|| anyhow!("feature probe requires step 1 to be present"))?;
    let (best_step, best_acc) = step_accuracies
        .iter()
        .copied()
        .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
        .ok_or_else(|| anyhow!("no step accuracies recorded"))?;
    let step_accuracies = step_accuracies
        .into_iter()
        .map(|(step, acc)| VisionDistillFeatureProbeStepAccuracy {
            step,
            acc,
            delta_vs_s1: acc - s1_acc,
            teacher_gap: teacher_acc - acc,
        })
        .collect::<Vec<_>>();

    Ok(VisionDistillFeatureProbeReport {
        artifact: VisionArtifactHeader::new("vision_distill_feature_probe"),
        benchmark: "burn_dragon vision distill feature probe",
        harness_version: VISION_DISTILL_FEATURE_PROBE_HARNESS_VERSION,
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        subset_seed,
        teacher_target: teacher_spec.name,
        image_size: vision.image_size,
        feature_dim: vision.projection_dim,
        num_classes,
        train_records: train_spec.samples.len(),
        val_records: val_spec.samples.len(),
        accuracy: VisionDistillFeatureProbeAccuracyReport {
            teacher_acc,
            step_accuracies,
            best_step,
            best_acc,
            best_minus_s1: best_acc - s1_acc,
            teacher_minus_best_gap: teacher_acc - best_acc,
        },
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_vision_distill_linear_probe(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    epochs: usize,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<VisionDistillLinearProbeReport> {
    run_vision_distill_linear_probe_for_teacher_with_seed(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        0,
        1337,
        epochs,
        learning_rate,
        weight_decay,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_vision_distill_linear_probe_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    train_seed: u64,
    epochs: usize,
    learning_rate: f32,
    weight_decay: f32,
) -> Result<VisionDistillLinearProbeReport> {
    run_vision_distill_linear_probe_for_teacher_with_seed(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        subset_seed,
        train_seed,
        epochs,
        learning_rate,
        weight_decay,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn run_vision_distill_linear_probe_for_teacher_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    train_seed: u64,
    epochs: usize,
    learning_rate: f32,
    weight_decay: f32,
    teacher_target: Option<&str>,
) -> Result<VisionDistillLinearProbeReport> {
    config.validate()?;
    let device = VisionDistillFeatureProbeDevice::default();
    let model = load_vision_encoder_from_checkpoint::<VisionDistillFeatureProbeBackend>(
        checkpoint,
        None,
        config_paths,
        &device,
    )?;
    let vision = config.vision.build();
    let batch_size = batch_size.max(1);
    let epochs = epochs.max(1);
    let learning_rate = learning_rate.max(1e-5);
    let weight_decay = weight_decay.max(0.0);
    let mut steps = steps.iter().map(|step| (*step).max(1)).collect::<Vec<_>>();
    steps.sort_unstable();
    steps.dedup();

    let teacher_spec = resolve_probe_teacher_spec(config, teacher_target)?;
    let (train_spec, val_spec, num_classes) = load_probe_split_specs_for_teacher(
        config,
        &teacher_spec,
        max_train_samples,
        max_val_samples,
        subset_seed,
    )?;
    let val_aug = build_val_augmentations(config);
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);

    let teacher_train = collect_teacher_features(&train_spec, batch_size, &device)?;
    let teacher_val = collect_teacher_features(&val_spec, batch_size, &device)?;
    let teacher_probe = train_linear_probe_classifier(
        &teacher_train,
        num_classes,
        epochs,
        learning_rate,
        weight_decay,
        train_seed,
    );
    let teacher_acc = eval_linear_probe_classifier(&teacher_probe, &teacher_val);

    let mut step_accuracies = Vec::with_capacity(steps.len());
    for step in &steps {
        let train_features = collect_dragon_features(
            &model,
            vision.projection_dim,
            &train_spec.samples,
            &val_aug,
            normalize,
            *step,
            batch_size,
            &device,
        )?;
        let val_features = collect_dragon_features(
            &model,
            vision.projection_dim,
            &val_spec.samples,
            &val_aug,
            normalize,
            *step,
            batch_size,
            &device,
        )?;
        let probe = train_linear_probe_classifier(
            &train_features,
            num_classes,
            epochs,
            learning_rate,
            weight_decay,
            train_seed ^ ((*step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let acc = eval_linear_probe_classifier(&probe, &val_features);
        step_accuracies.push(VisionDistillLinearProbeStepAccuracy {
            step: *step,
            acc,
            delta_vs_s1: 0.0,
            teacher_gap: teacher_acc - acc,
        });
    }

    let s1_acc = step_accuracies
        .first()
        .map(|step| step.acc)
        .ok_or_else(|| anyhow!("vision_distill_linear_probe requires at least one step"))?;
    for step in &mut step_accuracies {
        step.delta_vs_s1 = step.acc - s1_acc;
    }
    let best = step_accuracies
        .iter()
        .max_by(|lhs, rhs| lhs.acc.total_cmp(&rhs.acc))
        .ok_or_else(|| anyhow!("no linear-probe results"))?;
    let best_step = best.step;
    let best_acc = best.acc;

    Ok(VisionDistillLinearProbeReport {
        artifact: VisionArtifactHeader::new("vision_distill_linear_probe"),
        benchmark: "burn_dragon vision distill linear probe",
        harness_version: VISION_DISTILL_LINEAR_PROBE_HARNESS_VERSION,
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        steps,
        batch_size,
        max_train_samples,
        max_val_samples,
        subset_seed,
        teacher_target: teacher_spec.name,
        train_seed,
        epochs,
        learning_rate,
        weight_decay,
        image_size: vision.image_size,
        teacher_feature_dim: teacher_train.dim,
        student_feature_dim: vision.projection_dim,
        num_classes,
        train_records: train_spec.samples.len(),
        val_records: val_spec.samples.len(),
        accuracy: VisionDistillLinearProbeAccuracyReport {
            teacher_acc,
            step_accuracies,
            best_step,
            best_acc,
            best_minus_s1: best_acc - s1_acc,
            teacher_minus_best_gap: teacher_acc - best_acc,
        },
    })
}

pub fn export_vision_distill_feature_embeddings(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    step: usize,
    batch_size: usize,
    max_val_samples: Option<usize>,
) -> Result<VisionDistillFeatureExportReport> {
    config.validate()?;
    let device = VisionDistillFeatureProbeDevice::default();
    let model = load_vision_encoder_from_checkpoint::<VisionDistillFeatureProbeBackend>(
        checkpoint,
        None,
        config_paths,
        &device,
    )?;
    let vision = config.vision.build();
    let batch_size = batch_size.max(1);
    let step = step.max(1);
    let (_, val_spec, _num_classes) = load_probe_split_specs(config, None, max_val_samples, 0)?;
    let val_aug = build_val_augmentations(config);
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let feature_dim = vision.projection_dim;
    let mut labels = Vec::with_capacity(val_spec.samples.len());
    let mut teacher_cls = Vec::with_capacity(val_spec.samples.len());
    let mut student_cls = Vec::with_capacity(val_spec.samples.len());

    for start in (0..val_spec.samples.len()).step_by(batch_size) {
        let end = (start + batch_size).min(val_spec.samples.len());
        let samples = &val_spec.samples[start..end];
        let teacher_indices = samples
            .iter()
            .map(|sample| sample.teacher_index)
            .collect::<Vec<_>>();
        let (teacher_batch, _) = val_spec
            .cls_store
            .load_batch::<VisionDistillFeatureProbeBackend>(&teacher_indices, &device)?;
        let batch = build_image_batch(samples, &val_aug, normalize, &device)?;
        let student_batch = model.forward_images_steps_rollout_unbounded(batch, step, step);
        let teacher_flat = teacher_batch
            .into_data()
            .to_vec::<f32>()
            .expect("teacher cls");
        let student_flat = student_batch
            .cls_token
            .into_data()
            .to_vec::<f32>()
            .expect("student cls");
        for (row, sample) in samples.iter().enumerate() {
            let offset = row * feature_dim;
            labels.push(sample.label);
            teacher_cls.push(teacher_flat[offset..offset + feature_dim].to_vec());
            student_cls.push(student_flat[offset..offset + feature_dim].to_vec());
        }
    }

    Ok(VisionDistillFeatureExportReport {
        artifact: VisionArtifactHeader::new("vision_distill_feature_export"),
        benchmark: "burn_dragon vision distill feature export",
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        step,
        batch_size,
        max_val_samples,
        image_size: vision.image_size,
        feature_dim,
        records: labels.len(),
        labels,
        teacher_cls,
        student_cls,
    })
}

pub fn run_vision_distill_decode_probe_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    teacher_target: Option<&str>,
) -> Result<VisionDistillDecodeProbeReport> {
    run_vision_distill_decode_probe_with_seed_backend::<VisionDistillFeatureProbeBackend>(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_val_samples,
        subset_seed,
        teacher_target,
    )
}

#[cfg(feature = "cuda")]
pub fn run_vision_distill_decode_probe_cuda_with_seed(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    teacher_target: Option<&str>,
) -> Result<VisionDistillDecodeProbeReport> {
    run_vision_distill_decode_probe_with_seed_backend::<Cuda<f32>>(
        config,
        config_paths,
        checkpoint,
        steps,
        batch_size,
        max_val_samples,
        subset_seed,
        teacher_target,
    )
}

fn run_vision_distill_decode_probe_with_seed_backend<B: burn::tensor::backend::Backend>(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
    max_val_samples: Option<usize>,
    subset_seed: u64,
    teacher_target: Option<&str>,
) -> Result<VisionDistillDecodeProbeReport> {
    config.validate()?;
    let device = B::Device::default();
    let distill_model =
        load_vision_distill_model_from_checkpoint::<B>(config, checkpoint, &device)?;
    let vision = config.vision.build();
    let batch_size = batch_size.max(1);
    let mut steps = steps.iter().map(|step| (*step).max(1)).collect::<Vec<_>>();
    steps.sort_unstable();
    steps.dedup();

    let teacher_spec = resolve_probe_teacher_spec(config, teacher_target)?;
    let (_train_spec, val_spec, _num_classes) = load_probe_split_specs_for_teacher(
        config,
        &teacher_spec,
        None,
        max_val_samples,
        subset_seed,
    )?;
    let val_aug = build_val_augmentations(config);
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let loss_config = distill_loss_config_for_probe(
        distill_loss_config(config)?,
        teacher_spec.target_kind,
        teacher_spec.teacher.has_patch_targets(),
    );

    let mut steps_report = Vec::with_capacity(steps.len());
    for step in &steps {
        let mut total_sum = 0.0f64;
        let mut patch_sum = 0.0f64;
        let mut cls_sum = 0.0f64;
        let mut batches = 0usize;
        for start in (0..val_spec.samples.len()).step_by(batch_size) {
            let end = (start + batch_size).min(val_spec.samples.len());
            let samples = &val_spec.samples[start..end];
            let teacher_indices = samples
                .iter()
                .map(|sample| sample.teacher_index)
                .collect::<Vec<_>>();
            let (teacher_cls, teacher_patch) = val_spec
                .cls_store
                .load_batch::<B>(&teacher_indices, &device)?;
            let batch = build_image_batch(samples, &val_aug, normalize, &device)?;
            let output = distill_model
                .model
                .forward_images_steps_rollout_unbounded(batch, *step, *step);
            let terms = decode_probe_terms(
                &distill_model,
                &teacher_spec,
                output.patch_tokens,
                output.cls_token,
                teacher_patch,
                teacher_cls,
                &loss_config,
            )?;
            total_sum += scalar_tensor(&terms.total);
            patch_sum += scalar_tensor(&terms.patch);
            cls_sum += scalar_tensor(&terms.cls);
            batches += 1;
        }
        if batches == 0 {
            return Err(anyhow!("decode probe validation split is empty"));
        }
        let inv_batches = 1.0 / batches as f64;
        steps_report.push(VisionDistillDecodeProbeStepMetrics {
            step: *step,
            distill_total: total_sum * inv_batches,
            distill_patch: patch_sum * inv_batches,
            distill_cls: cls_sum * inv_batches,
            delta_total_vs_s1: 0.0,
            delta_patch_vs_s1: 0.0,
            delta_cls_vs_s1: 0.0,
        });
    }
    let s1 = steps_report
        .iter()
        .find(|step| step.step == 1)
        .cloned()
        .ok_or_else(|| anyhow!("decode probe requires step 1 to be present"))?;
    for step in &mut steps_report {
        step.delta_total_vs_s1 = s1.distill_total - step.distill_total;
        step.delta_patch_vs_s1 = s1.distill_patch - step.distill_patch;
        step.delta_cls_vs_s1 = s1.distill_cls - step.distill_cls;
    }
    let best = steps_report
        .iter()
        .min_by(|lhs, rhs| lhs.distill_total.total_cmp(&rhs.distill_total))
        .ok_or_else(|| anyhow!("no decode-probe results"))?;

    Ok(VisionDistillDecodeProbeReport {
        artifact: VisionArtifactHeader::new("vision_distill_decode_probe"),
        benchmark: "burn_dragon vision distill decode probe",
        harness_version: VISION_DISTILL_DECODE_PROBE_HARNESS_VERSION,
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        steps,
        batch_size,
        max_val_samples,
        subset_seed,
        teacher_target: teacher_spec.name,
        teacher_weight: teacher_spec.weight,
        teacher_target_kind: format!("{:?}", teacher_spec.target_kind),
        teacher_decoder_mode: format!("{:?}", teacher_spec.decoder_mode),
        image_size: vision.image_size,
        feature_dim: teacher_spec.teacher.feature_dim,
        val_records: val_spec.samples.len(),
        best_step: best.step,
        best_total: best.distill_total,
        best_minus_s1: s1.distill_total - best.distill_total,
        steps_report,
    })
}

fn load_probe_split_specs(
    config: &VisionTrainingConfig,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
) -> Result<(ProbeSplitSpec, ProbeSplitSpec, usize)> {
    let teacher_spec = resolve_probe_teacher_spec(config, None)?;
    load_probe_split_specs_for_teacher(
        config,
        &teacher_spec,
        max_train_samples,
        max_val_samples,
        subset_seed,
    )
}

fn distill_loss_config(config: &VisionTrainingConfig) -> Result<&VisionDistillationLossConfig> {
    match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => Ok(&distill.loss),
        other => Err(anyhow!(
            "vision distill probe requires distill mode, got {other:?}"
        )),
    }
}

fn distill_loss_config_for_probe(
    config: &VisionDistillationLossConfig,
    target_kind: VisionTeacherTargetKind,
    has_patch: bool,
) -> VisionDistillationLossConfig {
    let mut config = config.clone();
    if !matches!(target_kind, VisionTeacherTargetKind::PatchAndCls) || !has_patch {
        config.patch_mse_weight = 0.0;
        config.rel_weight = 0.0;
    }
    config
}

fn load_vision_distill_model_from_checkpoint<B: burn::tensor::backend::Backend>(
    config: &VisionTrainingConfig,
    checkpoint: &Path,
    device: &B::Device,
) -> Result<VisionDistillModel<B>> {
    let (checkpoint_base, _epoch) = resolve_checkpoint_base(checkpoint, None)?;
    let vision_config = config.vision.build();
    let rollout = resolve_vision_rollout(&config.training, vision_config.steps)?;
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        other => {
            return Err(anyhow!(
                "vision decode probe requires distill mode, got {other:?}"
            ));
        }
    };
    let model = crate::VisionDragon::<B>::new(vision_config, device);
    let mut distill_model = VisionDistillModel::new(model, distill, None, rollout, device);
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<VisionDistillModel<B> as Module<B>>::Record>(checkpoint_base.clone(), device)
        .with_context(|| format!("failed to load checkpoint {}", checkpoint.display()))?;
    distill_model = distill_model.load_record(record);
    Ok(distill_model)
}

fn resolve_probe_teacher_spec(
    config: &VisionTrainingConfig,
    teacher_target: Option<&str>,
) -> Result<ProbeTeacherSpec> {
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill,
        other => {
            return Err(anyhow!(
                "vision distill probe requires distill mode, got {other:?}"
            ));
        }
    };
    let teacher_target = teacher_target.unwrap_or(VisionDistillConfig::PRIMARY_TEACHER_NAME);
    if teacher_target == VisionDistillConfig::PRIMARY_TEACHER_NAME {
        let teacher = match &distill.teacher {
            VisionTeacherConfig::Features(teacher) => teacher.clone(),
            other => {
                return Err(anyhow!(
                    "vision distill probe requires precomputed feature teachers, got {other:?}"
                ));
            }
        };
        return Ok(ProbeTeacherSpec {
            name: VisionDistillConfig::PRIMARY_TEACHER_NAME.to_string(),
            weight: 1.0,
            target_kind: VisionTeacherTargetKind::PatchAndCls,
            decoder_mode: VisionTeacherDecoderMode::SharedProjection,
            teacher,
        });
    }

    let target = distill
        .teacher_targets
        .iter()
        .find(|target| target.name == teacher_target)
        .ok_or_else(|| anyhow!("unknown teacher target `{teacher_target}`"))?;
    let teacher = match &target.teacher {
        VisionTeacherConfig::Features(teacher) => teacher.clone(),
        other => {
            return Err(anyhow!(
                "vision distill probe requires precomputed feature teachers for `{teacher_target}`, got {other:?}"
            ));
        }
    };
    Ok(ProbeTeacherSpec {
        name: target.name.clone(),
        weight: target.weight,
        target_kind: target.target_kind,
        decoder_mode: target.decoder_mode,
        teacher,
    })
}

fn load_probe_split_specs_for_teacher(
    config: &VisionTrainingConfig,
    teacher_spec: &ProbeTeacherSpec,
    max_train_samples: Option<usize>,
    max_val_samples: Option<usize>,
    subset_seed: u64,
) -> Result<(ProbeSplitSpec, ProbeSplitSpec, usize)> {
    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);
    let mut train_samples = collect_samples(&train_root)?;
    let mut val_samples = collect_samples(&val_root)?;
    if let Some(max_records) = config.dataset.max_records {
        train_samples = limit_samples_balanced(&train_samples, max_records, subset_seed);
    }
    if let Some(max_train_samples) = max_train_samples {
        train_samples = limit_samples_balanced(&train_samples, max_train_samples, subset_seed);
    }
    if let Some(max_val_samples) = max_val_samples {
        val_samples = limit_samples_balanced(
            &val_samples,
            max_val_samples,
            subset_seed ^ 0xA5A5_A5A5_A5A5_A5A5,
        );
    }
    let num_classes = train_samples
        .iter()
        .chain(val_samples.iter())
        .map(|sample| sample.label)
        .max()
        .map(|value| value + 1)
        .ok_or_else(|| anyhow!("dataset is empty"))?;
    let train_store = DinoFeatureStore::new_optional_patch_with_options(
        &teacher_spec.teacher.train_cls_path,
        teacher_spec.teacher.train_patch_path.as_deref(),
        teacher_spec.teacher.feature_dim,
        teacher_spec.teacher.patch_tokens,
        Some(train_samples.len()),
        false,
    )?;
    let val_store = DinoFeatureStore::new_optional_patch_with_options(
        &teacher_spec.teacher.val_cls_path,
        teacher_spec.teacher.val_patch_path.as_deref(),
        teacher_spec.teacher.feature_dim,
        teacher_spec.teacher.patch_tokens,
        Some(val_samples.len()),
        false,
    )?;

    Ok((
        ProbeSplitSpec {
            samples: train_samples,
            cls_store: train_store,
        },
        ProbeSplitSpec {
            samples: val_samples,
            cls_store: val_store,
        },
        num_classes,
    ))
}

fn patch_grid_from_tokens(tokens: usize, label: &str) -> Result<(usize, usize)> {
    let side = (tokens as f64).sqrt().round() as usize;
    if side.saturating_mul(side) != tokens {
        return Err(anyhow!(
            "{label} token count {tokens} must form a square patch grid"
        ));
    }
    Ok((side, side))
}

fn resample_patch_tokens_to_target<B: burn::tensor::backend::Backend>(
    patch_tokens: Tensor<B, 3>,
    target_tokens: usize,
) -> Result<Tensor<B, 3>> {
    use burn::tensor::module::{adaptive_avg_pool2d, interpolate};
    use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

    let [batch, token_count, dim] = patch_tokens.shape().dims::<3>();
    let (src_h, src_w) = patch_grid_from_tokens(token_count, "student patch")?;
    let (dst_h, dst_w) = patch_grid_from_tokens(target_tokens, "teacher patch")?;
    if src_h == dst_h && src_w == dst_w {
        return Ok(patch_tokens);
    }
    let patch_state = patch_tokens
        .swap_dims(1, 2)
        .reshape([batch, dim, src_h, src_w]);
    let resized = if dst_h <= src_h && dst_w <= src_w {
        adaptive_avg_pool2d(patch_state, [dst_h, dst_w])
    } else {
        interpolate(
            patch_state,
            [dst_h, dst_w],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    };
    Ok(resized.reshape([batch, dim, dst_h * dst_w]).swap_dims(1, 2))
}

fn scalar_tensor<B: burn::tensor::backend::Backend>(tensor: &Tensor<B, 1>) -> f64 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor")[0] as f64
}

fn decode_probe_terms<B: burn::tensor::backend::Backend>(
    model: &VisionDistillModel<B>,
    teacher_spec: &ProbeTeacherSpec,
    student_patch: Tensor<B, 3>,
    student_cls: Tensor<B, 2>,
    teacher_patch: Option<Tensor<B, 3>>,
    teacher_cls: Tensor<B, 2>,
    loss_config: &VisionDistillationLossConfig,
) -> Result<crate::loss::VisionDistillationLossTerms<B>> {
    let patch_enabled = matches!(
        teacher_spec.target_kind,
        VisionTeacherTargetKind::PatchAndCls
    ) && teacher_patch.is_some();

    let (student_patch, student_cls, teacher_patch) =
        if teacher_spec.name == VisionDistillConfig::PRIMARY_TEACHER_NAME {
            (student_patch, student_cls, teacher_patch)
        } else {
            let (head, decoder_mode, target_patch_tokens) = model
                .auxiliary_teacher_head(&teacher_spec.name)
                .ok_or_else(|| anyhow!("missing auxiliary teacher head `{}`", teacher_spec.name))?;
            let projected_patch = if patch_enabled {
                let target_tokens = target_patch_tokens
                    .or(teacher_spec.teacher.patch_tokens)
                    .or_else(|| {
                        teacher_patch
                            .as_ref()
                            .map(|patch| patch.shape().dims::<3>()[1])
                    })
                    .ok_or_else(|| {
                        anyhow!(
                            "teacher target `{}` requires patch tokens",
                            teacher_spec.name
                        )
                    })?;
                let patch_input = if decoder_mode.supports_spatial_resampling() {
                    resample_patch_tokens_to_target(student_patch, target_tokens)?
                } else {
                    student_patch
                };
                Some(
                    head.patch_head
                        .as_ref()
                        .map(|head| head.forward(patch_input.clone()))
                        .unwrap_or(patch_input),
                )
            } else {
                None
            };
            let projected_cls = head.cls_head.forward(student_cls);
            (
                projected_patch.unwrap_or_else(|| dummy_patch_tokens(&projected_cls)),
                projected_cls,
                teacher_patch,
            )
        };

    let teacher_patch = teacher_patch.unwrap_or_else(|| dummy_patch_tokens(&student_cls));
    Ok(vision_distillation_loss_terms(
        student_patch,
        teacher_patch,
        student_cls,
        teacher_cls,
        loss_config,
    ))
}

fn dummy_patch_tokens<B: burn::tensor::backend::Backend>(cls: &Tensor<B, 2>) -> Tensor<B, 3> {
    let [batch, _dim] = cls.shape().dims::<2>();
    Tensor::<B, 3>::zeros([batch, 1, 1], &cls.device())
}

fn limit_samples_balanced(
    samples: &[ProbeSample],
    max_samples: usize,
    subset_seed: u64,
) -> Vec<ProbeSample> {
    if max_samples >= samples.len() {
        return samples.to_vec();
    }
    let num_classes = samples
        .iter()
        .map(|sample| sample.label)
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    if num_classes == 0 || max_samples == 0 {
        return Vec::new();
    }

    let mut buckets = vec![Vec::new(); num_classes];
    for sample in samples {
        buckets[sample.label].push(sample.clone());
    }
    for (class, bucket) in buckets.iter_mut().enumerate() {
        let mut rng = StdRng::seed_from_u64(
            subset_seed ^ ((class as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        use rand::seq::SliceRandom;
        bucket.shuffle(&mut rng);
    }
    let mut offsets = vec![0usize; num_classes];
    let mut limited = Vec::with_capacity(max_samples);
    while limited.len() < max_samples {
        let mut made_progress = false;
        for class in 0..num_classes {
            let offset = &mut offsets[class];
            if *offset < buckets[class].len() {
                limited.push(buckets[class][*offset].clone());
                *offset += 1;
                made_progress = true;
                if limited.len() >= max_samples {
                    break;
                }
            }
        }
        if !made_progress {
            break;
        }
    }
    limited
}

fn build_val_augmentations(config: &VisionTrainingConfig) -> ImageNetAugmentations {
    ImageNetAugmentations::new(
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
    )
}

fn collect_samples(root: &Path) -> Result<Vec<ProbeSample>> {
    let mut class_dirs: Vec<PathBuf> = fs::read_dir(root)
        .with_context(|| format!("failed to read {}", root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    class_dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    if class_dirs.is_empty() {
        return Err(anyhow!("no class directories found in {}", root.display()));
    }

    let mut samples = Vec::new();
    let mut teacher_index = 0usize;
    for (label, class_dir) in class_dirs.iter().enumerate() {
        let mut images = collect_images(class_dir)?;
        images.sort();
        for image in images {
            samples.push(ProbeSample {
                path: image,
                label,
                teacher_index,
            });
            teacher_index += 1;
        }
    }
    Ok(samples)
}

fn collect_images(root: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
        {
            let entry =
                entry.with_context(|| format!("failed to read dir entry in {}", dir.display()))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_image_file(&path) {
                images.push(path);
            }
        }
    }
    Ok(images)
}

fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"),
        None => false,
    }
}

fn build_teacher_centroids(
    split: &ProbeSplitSpec,
    num_classes: usize,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<Vec<Vec<f32>>> {
    let dim = split.cls_store.feature_dim();
    let mut sums = vec![vec![0.0f64; dim]; num_classes];
    let mut counts = vec![0usize; num_classes];
    for start in (0..split.samples.len()).step_by(batch_size) {
        let end = (start + batch_size).min(split.samples.len());
        let indices: Vec<usize> = split.samples[start..end]
            .iter()
            .map(|sample| sample.teacher_index)
            .collect();
        let (cls, _) = split
            .cls_store
            .load_batch::<VisionDistillFeatureProbeBackend>(&indices, device)?;
        let flat = cls.into_data().to_vec::<f32>().expect("teacher cls");
        for (row, sample) in split.samples[start..end].iter().enumerate() {
            let offset = row * dim;
            for feat in 0..dim {
                sums[sample.label][feat] += flat[offset + feat] as f64;
            }
            counts[sample.label] += 1;
        }
    }
    Ok(finalize_centroids(sums, counts))
}

fn build_dragon_centroids(
    model: &crate::model::VisionDragon<VisionDistillFeatureProbeBackend>,
    dim: usize,
    samples: &[ProbeSample],
    augment: &ImageNetAugmentations,
    normalize: VisionNormalize,
    step: usize,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<Vec<Vec<f32>>> {
    let num_classes = samples
        .iter()
        .map(|sample| sample.label)
        .max()
        .map(|value| value + 1)
        .ok_or_else(|| anyhow!("empty sample set"))?;
    let mut sums = vec![vec![0.0f64; dim]; num_classes];
    let mut counts = vec![0usize; num_classes];
    for start in (0..samples.len()).step_by(batch_size) {
        let end = (start + batch_size).min(samples.len());
        let batch = build_image_batch(&samples[start..end], augment, normalize, device)?;
        let output = model.forward_images_steps_rollout_unbounded(batch, step, step);
        let flat = output
            .cls_token
            .into_data()
            .to_vec::<f32>()
            .expect("dragon cls");
        for (row, sample) in samples[start..end].iter().enumerate() {
            let offset = row * dim;
            for feat in 0..dim {
                sums[sample.label][feat] += flat[offset + feat] as f64;
            }
            counts[sample.label] += 1;
        }
    }
    Ok(finalize_centroids(sums, counts))
}

fn eval_teacher_accuracy(
    centroids: &[Vec<f32>],
    split: &ProbeSplitSpec,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<f64> {
    let dim = split.cls_store.feature_dim();
    let mut correct = 0usize;
    let mut total = 0usize;
    for start in (0..split.samples.len()).step_by(batch_size) {
        let end = (start + batch_size).min(split.samples.len());
        let indices: Vec<usize> = split.samples[start..end]
            .iter()
            .map(|sample| sample.teacher_index)
            .collect();
        let (cls, _) = split
            .cls_store
            .load_batch::<VisionDistillFeatureProbeBackend>(&indices, device)?;
        let flat = cls.into_data().to_vec::<f32>().expect("teacher cls");
        for (row, sample) in split.samples[start..end].iter().enumerate() {
            let offset = row * dim;
            let pred = predict_centroid(&flat[offset..offset + dim], centroids);
            if pred == sample.label {
                correct += 1;
            }
            total += 1;
        }
    }
    Ok(correct as f64 / total.max(1) as f64)
}

fn eval_dragon_accuracy(
    centroids: &[Vec<f32>],
    model: &crate::model::VisionDragon<VisionDistillFeatureProbeBackend>,
    dim: usize,
    samples: &[ProbeSample],
    augment: &ImageNetAugmentations,
    normalize: VisionNormalize,
    step: usize,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<f64> {
    let mut correct = 0usize;
    let mut total = 0usize;
    for start in (0..samples.len()).step_by(batch_size) {
        let end = (start + batch_size).min(samples.len());
        let batch = build_image_batch(&samples[start..end], augment, normalize, device)?;
        let output = model.forward_images_steps_rollout_unbounded(batch, step, step);
        let flat = output
            .cls_token
            .into_data()
            .to_vec::<f32>()
            .expect("dragon cls");
        for (row, sample) in samples[start..end].iter().enumerate() {
            let offset = row * dim;
            let pred = predict_centroid(&flat[offset..offset + dim], centroids);
            if pred == sample.label {
                correct += 1;
            }
            total += 1;
        }
    }
    Ok(correct as f64 / total.max(1) as f64)
}

fn build_image_batch<B: burn::tensor::backend::Backend>(
    samples: &[ProbeSample],
    augment: &ImageNetAugmentations,
    normalize: VisionNormalize,
    device: &B::Device,
) -> Result<Tensor<B, 4>> {
    let image_size = augment.image_size();
    let mut buffer = Vec::with_capacity(samples.len() * 3 * image_size * image_size);
    let mut rng = StdRng::seed_from_u64(0);
    for sample in samples {
        let image = image::ImageReader::open(&sample.path)
            .with_context(|| format!("failed to open {}", sample.path.display()))?
            .decode()
            .with_context(|| format!("failed to decode {}", sample.path.display()))?;
        let rgb = augment.apply(&image, &mut rng);
        normalize.apply(&rgb, &mut buffer);
    }
    Ok(Tensor::<B, 4>::from_data(
        TensorData::new(buffer, [samples.len(), 3, image_size, image_size]),
        device,
    ))
}

fn collect_teacher_features(
    split: &ProbeSplitSpec,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<ProbeFeatureMatrix> {
    let dim = split.cls_store.feature_dim();
    let mut labels = Vec::with_capacity(split.samples.len());
    let mut flat = Vec::with_capacity(split.samples.len() * dim);
    for start in (0..split.samples.len()).step_by(batch_size.max(1)) {
        let end = (start + batch_size.max(1)).min(split.samples.len());
        let indices = split.samples[start..end]
            .iter()
            .map(|sample| sample.teacher_index)
            .collect::<Vec<_>>();
        let (cls, _) = split
            .cls_store
            .load_batch::<VisionDistillFeatureProbeBackend>(&indices, device)?;
        let batch_flat = cls.into_data().to_vec::<f32>().expect("teacher cls");
        for (row, sample) in split.samples[start..end].iter().enumerate() {
            let offset = row * dim;
            labels.push(sample.label);
            flat.extend_from_slice(&batch_flat[offset..offset + dim]);
        }
    }
    l2_normalize_rows(&mut flat, dim);
    Ok(ProbeFeatureMatrix { labels, flat, dim })
}

fn collect_dragon_features(
    model: &crate::model::VisionDragon<VisionDistillFeatureProbeBackend>,
    dim: usize,
    samples: &[ProbeSample],
    augment: &ImageNetAugmentations,
    normalize: VisionNormalize,
    step: usize,
    batch_size: usize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<ProbeFeatureMatrix> {
    let mut labels = Vec::with_capacity(samples.len());
    let mut flat = Vec::with_capacity(samples.len() * dim);
    for start in (0..samples.len()).step_by(batch_size.max(1)) {
        let end = (start + batch_size.max(1)).min(samples.len());
        let batch = build_image_batch(&samples[start..end], augment, normalize, device)?;
        let output = model.forward_images_steps_rollout_unbounded(batch, step, step);
        let batch_flat = output
            .cls_token
            .into_data()
            .to_vec::<f32>()
            .expect("dragon cls");
        for (row, sample) in samples[start..end].iter().enumerate() {
            let offset = row * dim;
            labels.push(sample.label);
            flat.extend_from_slice(&batch_flat[offset..offset + dim]);
        }
    }
    l2_normalize_rows(&mut flat, dim);
    Ok(ProbeFeatureMatrix { labels, flat, dim })
}

fn l2_normalize_rows(values: &mut [f32], dim: usize) {
    if dim == 0 {
        return;
    }
    for row in values.chunks_mut(dim) {
        l2_normalize_f32(row);
    }
}

fn train_linear_probe_classifier(
    train: &ProbeFeatureMatrix,
    num_classes: usize,
    epochs: usize,
    learning_rate: f32,
    weight_decay: f32,
    seed: u64,
) -> LinearProbeModel {
    let dim = train.dim.max(1);
    let classes = num_classes.max(1);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut weights = vec![0.0f32; classes * dim];
    let mut bias = vec![0.0f32; classes];
    for weight in &mut weights {
        *weight = rng.gen_range(-0.01f32..0.01f32);
    }
    let mut grad_w = vec![0.0f32; classes * dim];
    let mut grad_b = vec![0.0f32; classes];
    let mut logits = vec![0.0f32; classes];
    let mut probs = vec![0.0f32; classes];
    let records = train.labels.len().max(1);

    for epoch in 0..epochs.max(1) {
        grad_w.fill(0.0);
        grad_b.fill(0.0);
        for (row, label) in train.labels.iter().copied().enumerate() {
            let feature = &train.flat[row * dim..(row + 1) * dim];
            for class in 0..classes {
                let mut sum = bias[class];
                let weight_row = &weights[class * dim..(class + 1) * dim];
                for feat in 0..dim {
                    sum += weight_row[feat] * feature[feat];
                }
                logits[class] = sum;
            }
            let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut denom = 0.0f32;
            for class in 0..classes {
                let value = (logits[class] - max_logit).exp();
                probs[class] = value;
                denom += value;
            }
            let denom = denom.max(1e-12);
            for class in 0..classes {
                let prob = probs[class] / denom;
                let target = if class == label { 1.0 } else { 0.0 };
                let error = prob - target;
                grad_b[class] += error;
                let grad_row = &mut grad_w[class * dim..(class + 1) * dim];
                for feat in 0..dim {
                    grad_row[feat] += error * feature[feat];
                }
            }
        }

        let inv_records = 1.0f32 / records as f32;
        let lr = learning_rate
            * 0.5
            * (1.0 + (core::f32::consts::PI * epoch as f32 / epochs.max(1) as f32).cos());
        for class in 0..classes {
            bias[class] -= lr * grad_b[class] * inv_records;
            let row = &mut weights[class * dim..(class + 1) * dim];
            let grad_row = &grad_w[class * dim..(class + 1) * dim];
            for feat in 0..dim {
                let grad = grad_row[feat] * inv_records + weight_decay * row[feat];
                row[feat] -= lr * grad;
            }
        }
    }

    LinearProbeModel {
        weights,
        bias,
        classes,
        dim,
    }
}

fn eval_linear_probe_classifier(model: &LinearProbeModel, features: &ProbeFeatureMatrix) -> f64 {
    if features.labels.is_empty() || model.dim == 0 || model.classes == 0 {
        return 0.0;
    }
    let mut correct = 0usize;
    for (row, label) in features.labels.iter().copied().enumerate() {
        let feature = &features.flat[row * model.dim..(row + 1) * model.dim];
        let mut best_class = 0usize;
        let mut best_score = f32::NEG_INFINITY;
        for class in 0..model.classes {
            let mut score = model.bias[class];
            let row_weights = &model.weights[class * model.dim..(class + 1) * model.dim];
            for feat in 0..model.dim {
                score += row_weights[feat] * feature[feat];
            }
            if score > best_score {
                best_score = score;
                best_class = class;
            }
        }
        if best_class == label {
            correct += 1;
        }
    }
    correct as f64 / features.labels.len().max(1) as f64
}

fn finalize_centroids(mut sums: Vec<Vec<f64>>, counts: Vec<usize>) -> Vec<Vec<f32>> {
    for (class, centroid) in sums.iter_mut().enumerate() {
        let count = counts[class].max(1) as f64;
        for value in centroid.iter_mut() {
            *value /= count;
        }
        l2_normalize_f64(centroid);
    }
    sums.into_iter()
        .map(|centroid| centroid.into_iter().map(|value| value as f32).collect())
        .collect()
}

fn l2_normalize_f64(values: &mut [f64]) {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

fn predict_centroid(feature: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut normalized = feature.to_vec();
    l2_normalize_f32(&mut normalized);
    let mut best_idx = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    for (idx, centroid) in centroids.iter().enumerate() {
        let score = normalized
            .iter()
            .zip(centroid)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f32>();
        if score > best_score {
            best_score = score;
            best_idx = idx;
        }
    }
    best_idx
}

fn l2_normalize_f32(values: &mut [f32]) {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ProbeFeatureMatrix, ProbeSample, eval_linear_probe_classifier, limit_samples_balanced,
        train_linear_probe_classifier,
    };
    use std::path::PathBuf;

    fn sample(label: usize, teacher_index: usize) -> ProbeSample {
        ProbeSample {
            path: PathBuf::from(format!("{label}_{teacher_index}.jpg")),
            label,
            teacher_index,
        }
    }

    #[test]
    fn balanced_limit_is_seeded_and_class_balanced() {
        let samples = vec![
            sample(0, 0),
            sample(0, 1),
            sample(0, 2),
            sample(1, 3),
            sample(1, 4),
            sample(1, 5),
        ];

        let limited_a = limit_samples_balanced(&samples, 4, 7);
        let limited_b = limit_samples_balanced(&samples, 4, 7);
        let limited_c = limit_samples_balanced(&samples, 4, 11);

        assert_eq!(limited_a.len(), 4);
        assert_eq!(
            limited_a.iter().filter(|sample| sample.label == 0).count(),
            2
        );
        assert_eq!(
            limited_a.iter().filter(|sample| sample.label == 1).count(),
            2
        );
        assert_eq!(
            limited_a
                .iter()
                .map(|sample| sample.teacher_index)
                .collect::<Vec<_>>(),
            limited_b
                .iter()
                .map(|sample| sample.teacher_index)
                .collect::<Vec<_>>()
        );
        assert_ne!(
            limited_a
                .iter()
                .map(|sample| sample.teacher_index)
                .collect::<Vec<_>>(),
            limited_c
                .iter()
                .map(|sample| sample.teacher_index)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn linear_probe_fits_separable_toy_features() {
        let train = ProbeFeatureMatrix {
            labels: vec![0, 0, 1, 1],
            flat: vec![
                -1.0, -1.0, //
                -1.0, -0.5, //
                1.0, 0.5, //
                1.0, 1.0, //
            ],
            dim: 2,
        };
        let val = ProbeFeatureMatrix {
            labels: vec![0, 1],
            flat: vec![
                -0.8, -0.9, //
                0.8, 0.9, //
            ],
            dim: 2,
        };
        let model = train_linear_probe_classifier(&train, 2, 200, 0.2, 1e-4, 1337);
        let acc = eval_linear_probe_classifier(&model, &val);
        assert!(acc >= 1.0);
    }
}
