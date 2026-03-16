use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::tensor::{Tensor, TensorData};
use burn_ndarray::NdArray;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde::Serialize;

use super::artifact::{VisionArtifactHeader, push_vision_artifact_markdown_prelude};
use crate::checkpoint::load_vision_encoder_from_checkpoint;
use crate::config::{VisionTeacherConfig, VisionTrainingConfig, VisionTrainingModeConfig};
use crate::train::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetSplit, VisionNormalize,
};

pub type VisionDistillFeatureProbeBackend = NdArray<f32>;
pub type VisionDistillFeatureProbeDevice =
    <VisionDistillFeatureProbeBackend as burn::tensor::backend::Backend>::Device;

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
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub steps: Vec<usize>,
    pub batch_size: usize,
    pub image_size: usize,
    pub num_classes: usize,
    pub train_records: usize,
    pub val_records: usize,
    pub accuracy: VisionDistillFeatureProbeAccuracyReport,
}

#[derive(Clone)]
struct ProbeSample {
    path: PathBuf,
    label: usize,
}

struct ProbeSplitSpec {
    samples: Vec<ProbeSample>,
    cls_store: DinoFeatureStore,
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
        writeln!(&mut out, "- image size: {}", self.image_size).unwrap();
        writeln!(&mut out, "- classes: {}", self.num_classes).unwrap();
        writeln!(&mut out, "- train records: {}", self.train_records).unwrap();
        writeln!(&mut out, "- val records: {}", self.val_records).unwrap();
        writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
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
        writeln!(&mut out, "| teacher cls | {:.4} |", self.accuracy.teacher_acc).unwrap();
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

pub fn run_vision_distill_feature_probe(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    steps: &[usize],
    batch_size: usize,
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

    let (train_spec, val_spec, num_classes) = load_probe_split_specs(config)?;
    let val_aug = build_val_augmentations(config);
    let normalize = VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);

    let teacher_centroids =
        build_teacher_centroids(&train_spec, num_classes, batch_size, &device)?;
    let teacher_acc = eval_teacher_accuracy(&teacher_centroids, &val_spec, batch_size, &device)?;
    let mut step_accuracies = Vec::with_capacity(steps.len());
    for step in &steps {
        let centroids = build_dragon_centroids(
            &model,
            vision.embed_dim,
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
            vision.embed_dim,
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
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        steps,
        batch_size,
        image_size: vision.image_size,
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

fn load_probe_split_specs(
    config: &VisionTrainingConfig,
) -> Result<(ProbeSplitSpec, ProbeSplitSpec, usize)> {
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill,
        other => {
            return Err(anyhow!(
                "vision_distill_feature_probe requires distill mode, got {other:?}"
            ));
        }
    };
    let teacher = match &distill.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        other => {
            return Err(anyhow!(
                "vision_distill_feature_probe requires precomputed feature teachers, got {other:?}"
            ));
        }
    };

    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);
    let train_samples = collect_samples(&train_root)?;
    let val_samples = collect_samples(&val_root)?;
    let num_classes = train_samples
        .iter()
        .chain(val_samples.iter())
        .map(|sample| sample.label)
        .max()
        .map(|value| value + 1)
        .ok_or_else(|| anyhow!("dataset is empty"))?;
    let patch_tokens = teacher
        .patch_tokens
        .ok_or_else(|| anyhow!("teacher patch_tokens missing"))?;

    let train_store = DinoFeatureStore::new(
        &teacher.train_cls_path,
        &teacher.train_patch_path,
        teacher.feature_dim,
        patch_tokens,
        Some(train_samples.len()),
    )?;
    let val_store = DinoFeatureStore::new(
        &teacher.val_cls_path,
        &teacher.val_patch_path,
        teacher.feature_dim,
        patch_tokens,
        Some(val_samples.len()),
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
    for (label, class_dir) in class_dirs.iter().enumerate() {
        let mut images = collect_images(class_dir)?;
        images.sort();
        for image in images {
            samples.push(ProbeSample { path: image, label });
        }
    }
    Ok(samples)
}

fn collect_images(root: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).with_context(|| format!("failed to read {}", dir.display()))?
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
        let indices: Vec<usize> = (start..end).collect();
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
        let flat = output.cls_token.into_data().to_vec::<f32>().expect("dragon cls");
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
        let indices: Vec<usize> = (start..end).collect();
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
        let flat = output.cls_token.into_data().to_vec::<f32>().expect("dragon cls");
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

fn build_image_batch(
    samples: &[ProbeSample],
    augment: &ImageNetAugmentations,
    normalize: VisionNormalize,
    device: &VisionDistillFeatureProbeDevice,
) -> Result<Tensor<VisionDistillFeatureProbeBackend, 4>> {
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
    Ok(Tensor::<VisionDistillFeatureProbeBackend, 4>::from_data(
        TensorData::new(buffer, [samples.len(), 3, image_size, image_size]),
        device,
    ))
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
