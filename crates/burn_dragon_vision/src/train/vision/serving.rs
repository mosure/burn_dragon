use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_checkpoint::{
    BurnpackBundleExportOptions, BurnpackFloatPrecision, load_model_from_burnpack_candidates,
};
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

use super::artifact::{VisionArtifactHeader, push_vision_artifact_markdown_prelude};
use crate::checkpoint::{
    export_vision_encoder_checkpoint_to_burnpack, load_vision_encoder_from_checkpoint,
};
use crate::config::{VisionTeacherConfig, VisionTrainingConfig, VisionTrainingModeConfig};
use crate::loss::{VisionDistillationLossConfig, vision_distillation_loss_terms};
use crate::model::{VisionDragon, VisionDragonOutput};
use crate::train::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetBatch, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};

pub type VisionDistillServingBenchmarkBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionDistillServingBenchmarkDevice =
    <VisionDistillServingBenchmarkBackend as BackendTrait>::Device;
pub type VisionDistillDeploySmokeBackend = NdArray<f32>;
pub type VisionDistillDeploySmokeDevice =
    <VisionDistillDeploySmokeBackend as BackendTrait>::Device;

#[derive(Clone, Serialize)]
pub struct VisionDistillServingStepMetrics {
    pub step: usize,
    pub backprop_steps: usize,
    pub forward_ms: f64,
    pub samples_per_sec: f64,
    pub tokens_per_sec: f64,
    pub distill_total: f64,
    pub distill_patch: f64,
    pub distill_cls: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDistillServingBenchmarkReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub config: Vec<PathBuf>,
    pub checkpoint: Option<PathBuf>,
    pub image_size: usize,
    pub patch_tokens_per_image: usize,
    pub batch_size: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub serving: VisionDistillServingStepMetrics,
    pub baseline: VisionDistillServingStepMetrics,
    pub total_gain_vs_baseline: f64,
    pub patch_gain_vs_baseline: f64,
    pub cls_gain_vs_baseline: f64,
    pub latency_scale_vs_baseline: f64,
    pub total_gain_per_extra_ms: Option<f64>,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub enum VisionDistillDeploySmokePrecision {
    F16,
    F32,
}

impl From<VisionDistillDeploySmokePrecision> for BurnpackFloatPrecision {
    fn from(value: VisionDistillDeploySmokePrecision) -> Self {
        match value {
            VisionDistillDeploySmokePrecision::F16 => Self::F16,
            VisionDistillDeploySmokePrecision::F32 => Self::F32,
        }
    }
}

#[derive(Clone, Serialize)]
pub struct VisionDistillDeploySmokeReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub config: Vec<PathBuf>,
    pub checkpoint: PathBuf,
    pub burnpack: PathBuf,
    pub precision: VisionDistillDeploySmokePrecision,
    pub image_size: usize,
    pub batch_size: usize,
    pub step: usize,
    pub backprop_steps: usize,
    pub checkpoint_forward_ms: f64,
    pub burnpack_forward_ms: f64,
    pub latency_scale: f64,
    pub checkpoint_distill_total: f64,
    pub burnpack_distill_total: f64,
    pub checkpoint_distill_patch: f64,
    pub burnpack_distill_patch: f64,
    pub checkpoint_distill_cls: f64,
    pub burnpack_distill_cls: f64,
    pub distill_total_abs_diff: f64,
    pub distill_patch_abs_diff: f64,
    pub distill_cls_abs_diff: f64,
    pub patch_max_abs_diff: f64,
    pub cls_max_abs_diff: f64,
}

impl VisionDistillServingBenchmarkReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Serving Benchmark",
            &self.artifact,
        );
        writeln!(&mut out, "- adapter: {}", self.adapter).unwrap();
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
        if let Some(checkpoint) = self.checkpoint.as_ref() {
            writeln!(&mut out, "- checkpoint: {}", checkpoint.display()).unwrap();
        }
        writeln!(&mut out, "- image size: {}", self.image_size).unwrap();
        writeln!(
            &mut out,
            "- patch tokens / image: {}",
            self.patch_tokens_per_image
        )
        .unwrap();
        writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
        writeln!(
            &mut out,
            "- serving step: s{} (bptt {})",
            self.serving.step, self.serving.backprop_steps
        )
        .unwrap();
        writeln!(
            &mut out,
            "- baseline step: s{} (bptt {})",
            self.baseline.step, self.baseline.backprop_steps
        )
        .unwrap();
        writeln!(
            &mut out,
            "- total gain vs baseline: {:.4}",
            self.total_gain_vs_baseline
        )
        .unwrap();
        writeln!(
            &mut out,
            "- patch gain vs baseline: {:.4}",
            self.patch_gain_vs_baseline
        )
        .unwrap();
        writeln!(
            &mut out,
            "- cls gain vs baseline: {:.4}",
            self.cls_gain_vs_baseline
        )
        .unwrap();
        writeln!(
            &mut out,
            "- latency scale vs baseline: {:.3}",
            self.latency_scale_vs_baseline
        )
        .unwrap();
        if let Some(gain_per_extra_ms) = self.total_gain_per_extra_ms {
            writeln!(&mut out, "- total gain / extra ms: {:.6}", gain_per_extra_ms).unwrap();
        }
        writeln!(&mut out).unwrap();
        writeln!(
            &mut out,
            "| profile | step | bptt | forward ms | samples/s | tokens/s | total | patch | cls |"
        )
        .unwrap();
        writeln!(&mut out, "|---|---:|---:|---:|---:|---:|---:|---:|---:|").unwrap();
        for (label, step) in [("baseline", &self.baseline), ("serving", &self.serving)] {
            writeln!(
                &mut out,
                "| {label} | {} | {} | {:.3} | {:.3} | {:.1} | {:.4} | {:.4} | {:.4} |",
                step.step,
                step.backprop_steps,
                step.forward_ms,
                step.samples_per_sec,
                step.tokens_per_sec,
                step.distill_total,
                step.distill_patch,
                step.distill_cls
            )
            .unwrap();
        }
        out
    }
}

impl VisionDistillDeploySmokeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Deploy Smoke",
            &self.artifact,
        );
        writeln!(&mut out, "- checkpoint: {}", self.checkpoint.display()).unwrap();
        writeln!(&mut out, "- burnpack: {}", self.burnpack.display()).unwrap();
        writeln!(&mut out, "- burnpack precision: {:?}", self.precision).unwrap();
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
        writeln!(&mut out, "- batch size: {}", self.batch_size).unwrap();
        writeln!(
            &mut out,
            "- rollout step: s{} (bptt {})",
            self.step, self.backprop_steps
        )
        .unwrap();
        writeln!(
            &mut out,
            "- checkpoint forward: {:.3} ms",
            self.checkpoint_forward_ms
        )
        .unwrap();
        writeln!(&mut out, "- burnpack forward: {:.3} ms", self.burnpack_forward_ms).unwrap();
        writeln!(&mut out, "- latency scale: {:.3}", self.latency_scale).unwrap();
        writeln!(
            &mut out,
            "- patch max abs diff: {:.8}",
            self.patch_max_abs_diff
        )
        .unwrap();
        writeln!(&mut out, "- cls max abs diff: {:.8}", self.cls_max_abs_diff).unwrap();
        writeln!(
            &mut out,
            "- distill total abs diff: {:.8}",
            self.distill_total_abs_diff
        )
        .unwrap();
        writeln!(
            &mut out,
            "- distill patch abs diff: {:.8}",
            self.distill_patch_abs_diff
        )
        .unwrap();
        writeln!(
            &mut out,
            "- distill cls abs diff: {:.8}",
            self.distill_cls_abs_diff
        )
        .unwrap();
        writeln!(&mut out).unwrap();
        writeln!(&mut out, "| source | forward ms | total | patch | cls |").unwrap();
        writeln!(&mut out, "|---|---:|---:|---:|---:|").unwrap();
        writeln!(
            &mut out,
            "| checkpoint | {:.3} | {:.6} | {:.6} | {:.6} |",
            self.checkpoint_forward_ms,
            self.checkpoint_distill_total,
            self.checkpoint_distill_patch,
            self.checkpoint_distill_cls
        )
        .unwrap();
        writeln!(
            &mut out,
            "| burnpack | {:.3} | {:.6} | {:.6} | {:.6} |",
            self.burnpack_forward_ms,
            self.burnpack_distill_total,
            self.burnpack_distill_patch,
            self.burnpack_distill_cls
        )
        .unwrap();
        out
    }
}

pub fn init_vision_serving_benchmark_runtime(device: &VisionDistillServingBenchmarkDevice) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

#[allow(clippy::too_many_arguments)]
pub fn run_vision_distill_serving_benchmark(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: Option<&Path>,
    adapter: String,
    step: usize,
    backprop_steps: Option<usize>,
    baseline_step: usize,
    warmup: usize,
    iterations: usize,
    batch_size: usize,
) -> Result<VisionDistillServingBenchmarkReport> {
    let device = VisionDistillServingBenchmarkDevice::default();
    init_vision_serving_benchmark_runtime(&device);
    config.validate()?;
    let batch_size = batch_size.max(1);
    let probe = build_validation_probe_batch::<VisionDistillServingBenchmarkBackend>(
        config, batch_size, &device, "vision_distill_serving_bench",
    )?;
    let model = load_or_init_serving_model(config, checkpoint, config_paths, &device)?;
    let vision = config.vision.build();
    let patch_grid = vision.image_size.div_ceil(vision.patch_size.max(1)).max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;

    let serving_step = step.max(1);
    let serving_backprop = backprop_steps
        .unwrap_or(
            config
                .training
                .rollout_backprop_steps
                .unwrap_or(serving_step)
                .min(serving_step)
                .max(1),
        )
        .min(serving_step)
        .max(1);
    let baseline_step = baseline_step.max(1);
    let baseline_backprop = config
        .training
        .rollout_backprop_steps
        .unwrap_or(baseline_step)
        .min(baseline_step)
        .max(1);

    let baseline = run_serving_step_case(
        &model,
        &probe.batch,
        &probe.distill_loss,
        baseline_step,
        baseline_backprop,
        warmup,
        iterations,
    );
    let serving = run_serving_step_case(
        &model,
        &probe.batch,
        &probe.distill_loss,
        serving_step,
        serving_backprop,
        warmup,
        iterations,
    );

    let total_gain_vs_baseline = baseline.distill_total - serving.distill_total;
    let patch_gain_vs_baseline = baseline.distill_patch - serving.distill_patch;
    let cls_gain_vs_baseline = baseline.distill_cls - serving.distill_cls;
    let extra_ms = serving.forward_ms - baseline.forward_ms;
    let total_gain_per_extra_ms = if extra_ms > 0.0 {
        Some(total_gain_vs_baseline / extra_ms)
    } else {
        None
    };
    let latency_scale_vs_baseline = if baseline.forward_ms > 0.0 {
        serving.forward_ms / baseline.forward_ms
    } else {
        0.0
    };

    Ok(VisionDistillServingBenchmarkReport {
        artifact: VisionArtifactHeader::new("vision_distill_serving_benchmark"),
        benchmark: "burn_dragon vision distill serving benchmark",
        adapter,
        config: config_paths.to_vec(),
        checkpoint: checkpoint.map(PathBuf::from),
        image_size: vision.image_size,
        patch_tokens_per_image,
        batch_size,
        warmup,
        iterations,
        serving,
        baseline,
        total_gain_vs_baseline,
        patch_gain_vs_baseline,
        cls_gain_vs_baseline,
        latency_scale_vs_baseline,
        total_gain_per_extra_ms,
    })
}

#[allow(clippy::too_many_arguments)]
pub fn run_vision_distill_deploy_smoke(
    config: &VisionTrainingConfig,
    config_paths: &[PathBuf],
    checkpoint: &Path,
    step: usize,
    backprop_steps: Option<usize>,
    batch_size: usize,
    warmup: usize,
    iterations: usize,
    precision: VisionDistillDeploySmokePrecision,
) -> Result<VisionDistillDeploySmokeReport> {
    config.validate()?;
    let device = VisionDistillDeploySmokeDevice::default();
    let vision = config.vision.build();
    let step = step.max(1);
    let backprop_steps = backprop_steps
        .unwrap_or(
            config
                .training
                .rollout_backprop_steps
                .unwrap_or(step)
                .min(step)
                .max(1),
        )
        .min(step)
        .max(1);
    let batch_size = batch_size.max(1);

    let checkpoint_model = load_vision_encoder_from_checkpoint::<VisionDistillDeploySmokeBackend>(
        checkpoint,
        None,
        config_paths,
        &device,
    )?;
    let export_report = export_vision_encoder_checkpoint_to_burnpack(
        checkpoint,
        None,
        config_paths,
        &deploy_output_base(checkpoint)?,
        &BurnpackBundleExportOptions {
            precision: precision.into(),
            overwrite_parts: true,
            ..BurnpackBundleExportOptions::default()
        },
    )?;
    let burnpack_path = export_report.bundle.burnpack_path.clone();
    let burnpack_model = load_model_from_burnpack_candidates(
        std::slice::from_ref(&burnpack_path),
        "VisionDragon",
        true,
        || VisionDragon::<VisionDistillDeploySmokeBackend>::new(vision.clone(), &device),
    )
    .map_err(|err| anyhow!(err))?
    .0;

    let probe = build_validation_probe_batch::<VisionDistillDeploySmokeBackend>(
        config, batch_size, &device, "vision_distill_deploy_smoke",
    )?;
    let checkpoint_output = checkpoint_model.forward_images_steps_rollout_unbounded(
        probe.batch.images.clone(),
        step,
        backprop_steps,
    );
    let burnpack_output = burnpack_model.forward_images_steps_rollout_unbounded(
        probe.batch.images.clone(),
        step,
        backprop_steps,
    );
    let checkpoint_terms =
        distill_terms_from_output(&checkpoint_output, &probe.batch, &probe.distill_loss);
    let burnpack_terms =
        distill_terms_from_output(&burnpack_output, &probe.batch, &probe.distill_loss);

    let checkpoint_forward_ms = measure_avg_ms(
        iterations,
        || {
            sync_output(checkpoint_model.forward_images_steps_rollout_unbounded(
                probe.batch.images.clone(),
                step,
                backprop_steps,
            ));
        },
        warmup,
    );
    let burnpack_forward_ms = measure_avg_ms(
        iterations,
        || {
            sync_output(burnpack_model.forward_images_steps_rollout_unbounded(
                probe.batch.images.clone(),
                step,
                backprop_steps,
            ));
        },
        warmup,
    );

    let patch_max_abs_diff = max_abs_diff_3d(
        checkpoint_output.patch_tokens.clone(),
        burnpack_output.patch_tokens.clone(),
    );
    let cls_max_abs_diff =
        max_abs_diff_2d(checkpoint_output.cls_token.clone(), burnpack_output.cls_token.clone());

    Ok(VisionDistillDeploySmokeReport {
        artifact: VisionArtifactHeader::new("vision_distill_deploy_smoke"),
        benchmark: "burn_dragon vision distill deploy smoke",
        config: config_paths.to_vec(),
        checkpoint: checkpoint.to_path_buf(),
        burnpack: burnpack_path,
        precision,
        image_size: vision.image_size,
        batch_size,
        step,
        backprop_steps,
        checkpoint_forward_ms,
        burnpack_forward_ms,
        latency_scale: if checkpoint_forward_ms > 0.0 {
            burnpack_forward_ms / checkpoint_forward_ms
        } else {
            0.0
        },
        checkpoint_distill_total: checkpoint_terms.0,
        burnpack_distill_total: burnpack_terms.0,
        checkpoint_distill_patch: checkpoint_terms.1,
        burnpack_distill_patch: burnpack_terms.1,
        checkpoint_distill_cls: checkpoint_terms.2,
        burnpack_distill_cls: burnpack_terms.2,
        distill_total_abs_diff: (checkpoint_terms.0 - burnpack_terms.0).abs(),
        distill_patch_abs_diff: (checkpoint_terms.1 - burnpack_terms.1).abs(),
        distill_cls_abs_diff: (checkpoint_terms.2 - burnpack_terms.2).abs(),
        patch_max_abs_diff,
        cls_max_abs_diff,
    })
}

struct DistillValidationProbeBatch<B: BackendTrait> {
    batch: ImageNetBatch<B>,
    distill_loss: VisionDistillationLossConfig,
}

fn build_validation_probe_batch<B: BackendTrait>(
    config: &VisionTrainingConfig,
    batch_size: usize,
    device: &B::Device,
    tool_name: &str,
) -> Result<DistillValidationProbeBatch<B>> {
    let vision = config.vision.build();
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        other => {
            return Err(anyhow!(
                "{tool_name} requires distill mode, got {other:?}"
            ));
        }
    };

    let teacher = match &distill.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        other => {
            return Err(anyhow!(
                "{tool_name} currently requires precomputed feature teachers, got {other:?}"
            ));
        }
    };

    let student_patch_tokens = vision.image_size.div_ceil(vision.patch_size.max(1)).pow(2);
    let teacher_patch_tokens = teacher.patch_tokens.unwrap_or(student_patch_tokens);
    if teacher_patch_tokens != student_patch_tokens {
        return Err(anyhow!(
            "teacher patch tokens ({teacher_patch_tokens}) must match student tokens ({student_patch_tokens})"
        ));
    }

    let normalize = VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
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
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);
    let mut dataset = ImageNetDataset::new(ImageNetDatasetConfig {
        root: val_root,
        split: ImageNetSplit::Val,
        max_records: config.dataset.max_records,
        augmentations: val_aug,
        local_augmentations: None,
        normalize,
        teacher: None,
        views: 1,
        local_views: 0,
        min_view_overlap: 0.0,
        view_overlap_attempts: 1,
        cache_decoded: config.dataset.cache_decoded,
        cache_capacity: config.dataset.cache_capacity,
        cache_preprocessed: config.dataset.cache_preprocessed,
    })?;
    let record_count = dataset.len();
    let teacher_store = Arc::new(
        DinoFeatureStore::new(
            &teacher.val_cls_path,
            &teacher.val_patch_path,
            teacher.feature_dim,
            teacher_patch_tokens,
            Some(record_count),
        )
        .with_context(|| {
            format!(
                "failed to open teacher features patch={} cls={}",
                teacher.val_patch_path.display(),
                teacher.val_cls_path.display()
            )
        })?,
    );
    dataset = dataset.with_teacher(teacher_store);

    Ok(DistillValidationProbeBatch {
        batch: dataset.sample_batch::<B>(batch_size, device),
        distill_loss: distill.loss,
    })
}

fn load_or_init_serving_model(
    config: &VisionTrainingConfig,
    checkpoint: Option<&Path>,
    config_paths: &[PathBuf],
    device: &VisionDistillServingBenchmarkDevice,
) -> Result<VisionDragon<VisionDistillServingBenchmarkBackend>> {
    match checkpoint {
        Some(checkpoint) => load_vision_encoder_from_checkpoint::<VisionDistillServingBenchmarkBackend>(
            checkpoint,
            None,
            config_paths,
            device,
        ),
        None => Ok(VisionDragon::new(config.vision.build(), device)),
    }
}

fn run_serving_step_case(
    model: &VisionDragon<VisionDistillServingBenchmarkBackend>,
    batch: &ImageNetBatch<VisionDistillServingBenchmarkBackend>,
    distill_loss: &VisionDistillationLossConfig,
    step: usize,
    backprop_steps: usize,
    warmup: usize,
    iterations: usize,
) -> VisionDistillServingStepMetrics {
    for _ in 0..warmup {
        sync_scalar_tensor(run_forward_loss(
            model,
            batch,
            distill_loss,
            step,
            backprop_steps,
        ));
    }

    let elapsed_ms = measure_avg_ns(iterations, || {
        sync_scalar_tensor(run_forward_loss(
            model,
            batch,
            distill_loss,
            step,
            backprop_steps,
        ));
    }) / 1_000_000.0;

    let output = model.forward_images_steps_rollout_unbounded(
        batch.images.clone(),
        step,
        backprop_steps,
    );
    let teacher_patch = batch
        .teacher_patch
        .clone()
        .expect("teacher patch features required for serving benchmark");
    let teacher_cls = batch
        .teacher_cls
        .clone()
        .expect("teacher cls features required for serving benchmark");
    let terms = vision_distillation_loss_terms(
        output.patch_tokens,
        teacher_patch.clone(),
        output.cls_token,
        teacher_cls,
        distill_loss,
    );

    let batch_size = batch.images.shape().dims::<4>()[0];
    let patch_tokens = teacher_patch.shape().dims::<3>()[1] * batch_size;
    let seconds = elapsed_ms / 1000.0;
    VisionDistillServingStepMetrics {
        step,
        backprop_steps,
        forward_ms: elapsed_ms,
        samples_per_sec: batch_size as f64 / seconds,
        tokens_per_sec: patch_tokens as f64 / seconds,
        distill_total: scalar_f32(terms.total),
        distill_patch: scalar_f32(terms.patch),
        distill_cls: scalar_f32(terms.cls),
    }
}

fn run_forward_loss(
    model: &VisionDragon<VisionDistillServingBenchmarkBackend>,
    batch: &ImageNetBatch<VisionDistillServingBenchmarkBackend>,
    distill_loss: &VisionDistillationLossConfig,
    step: usize,
    backprop_steps: usize,
) -> Tensor<VisionDistillServingBenchmarkBackend, 1> {
    let output = model.forward_images_steps_rollout_unbounded(
        batch.images.clone(),
        step,
        backprop_steps,
    );
    let teacher_patch = batch
        .teacher_patch
        .clone()
        .expect("teacher patch features required for serving benchmark");
    let teacher_cls = batch
        .teacher_cls
        .clone()
        .expect("teacher cls features required for serving benchmark");
    let terms = vision_distillation_loss_terms(
        output.patch_tokens,
        teacher_patch,
        output.cls_token,
        teacher_cls,
        distill_loss,
    );
    terms.total.reshape([1])
}

fn distill_terms_from_output(
    output: &VisionDragonOutput<VisionDistillDeploySmokeBackend>,
    batch: &ImageNetBatch<VisionDistillDeploySmokeBackend>,
    distill_loss: &VisionDistillationLossConfig,
) -> (f64, f64, f64) {
    let teacher_patch = batch
        .teacher_patch
        .clone()
        .expect("teacher patch features required for deploy smoke");
    let teacher_cls = batch
        .teacher_cls
        .clone()
        .expect("teacher cls features required for deploy smoke");
    let terms = vision_distillation_loss_terms(
        output.patch_tokens.clone(),
        teacher_patch,
        output.cls_token.clone(),
        teacher_cls,
        distill_loss,
    );
    (
        scalar_f32(terms.total),
        scalar_f32(terms.patch),
        scalar_f32(terms.cls),
    )
}

fn deploy_output_base(checkpoint: &Path) -> Result<PathBuf> {
    let checkpoint_dir = checkpoint_base(checkpoint);
    let run_dir = checkpoint_dir
        .parent()
        .and_then(|parent| parent.parent())
        .ok_or_else(|| anyhow!("failed to resolve run dir from {}", checkpoint.display()))?;
    Ok(run_dir.join("deploy").join("smoke").join("model"))
}

fn checkpoint_base(checkpoint: &Path) -> PathBuf {
    if checkpoint.is_dir() {
        checkpoint.join("model-0")
    } else if checkpoint.extension().is_some() {
        checkpoint.with_extension("")
    } else {
        checkpoint.to_path_buf()
    }
}

fn max_abs_diff_3d(
    lhs: Tensor<VisionDistillDeploySmokeBackend, 3>,
    rhs: Tensor<VisionDistillDeploySmokeBackend, 3>,
) -> f64 {
    let lhs = lhs.into_data().to_vec::<f32>().expect("lhs tensor");
    let rhs = rhs.into_data().to_vec::<f32>().expect("rhs tensor");
    lhs.into_iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0, f64::max)
}

fn max_abs_diff_2d(
    lhs: Tensor<VisionDistillDeploySmokeBackend, 2>,
    rhs: Tensor<VisionDistillDeploySmokeBackend, 2>,
) -> f64 {
    let lhs = lhs.into_data().to_vec::<f32>().expect("lhs tensor");
    let rhs = rhs.into_data().to_vec::<f32>().expect("rhs tensor");
    lhs.into_iter()
        .zip(rhs)
        .map(|(a, b)| (a - b).abs() as f64)
        .fold(0.0, f64::max)
}

fn scalar_f32<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor.into_data().to_vec::<f32>().expect("scalar tensor")[0] as f64
}

fn measure_avg_ns<F: FnMut()>(iterations: usize, mut f: F) -> f64 {
    let iterations = iterations.max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn measure_avg_ms<F: FnMut()>(iterations: usize, mut f: F, warmup: usize) -> f64 {
    for _ in 0..warmup {
        f();
    }
    let iterations = iterations.max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64 / 1_000_000.0
}

fn sync_scalar_tensor(tensor: Tensor<VisionDistillServingBenchmarkBackend, 1>) {
    let _ = tensor.into_data();
}

fn sync_output(output: VisionDragonOutput<VisionDistillDeploySmokeBackend>) {
    let _ = output.patch_tokens.into_data();
    let _ = output.cls_token.into_data();
}
