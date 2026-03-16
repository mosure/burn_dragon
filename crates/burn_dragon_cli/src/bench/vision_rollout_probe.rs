use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon::vision::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetBatch, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionArtifactHeader, VisionBackboneKind, VisionDragon,
    VisionNormalize, VisionTeacherConfig, VisionTrainingConfig, VisionTrainingModeConfig,
    load_vision_encoder_from_checkpoint, push_vision_artifact_markdown_prelude,
    vision_distillation_loss_terms,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

pub type VisionRolloutProbeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionRolloutProbeDevice = <VisionRolloutProbeBackend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VisionRolloutProbeConfig {
    pub checkpoint: Option<PathBuf>,
    pub steps: Vec<usize>,
    pub warmup: usize,
    pub iterations: usize,
    pub batch_size: Option<usize>,
}

#[derive(Clone, Serialize)]
pub struct VisionRolloutProbeStepCase {
    pub step: usize,
    pub backprop_steps: usize,
    pub forward_ms: f64,
    pub samples_per_sec: f64,
    pub tokens_per_sec: f64,
    pub forward_scale_vs_step1: f64,
    pub forward_alpha_vs_step1: Option<f64>,
    pub distill_total: f64,
    pub distill_patch: f64,
    pub distill_cls: f64,
    pub gain_vs_step1_total: f64,
    pub gain_vs_step1_patch: f64,
    pub gain_vs_step1_cls: f64,
    pub total_gain_per_extra_ms: Option<f64>,
    pub patch_delta_vs_prev: Option<f64>,
    pub cls_delta_vs_prev: Option<f64>,
    pub patch_delta_vs_step1: Option<f64>,
    pub cls_delta_vs_step1: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct VisionRolloutProbeSummary {
    pub best_total_step: usize,
    pub best_total: f64,
    pub best_patch_step: usize,
    pub best_patch: f64,
    pub best_cls_step: usize,
    pub best_cls: f64,
    pub gain_vs_step1_total: f64,
    pub gain_vs_step1_patch: f64,
    pub gain_vs_step1_cls: f64,
    pub forward_alpha_fit_all_steps: Option<f64>,
    pub forward_ms_per_extra_step_fit: Option<f64>,
    pub best_efficiency_step: Option<usize>,
    pub best_efficiency_gain_per_extra_ms: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct VisionRolloutProbeReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub config: Vec<PathBuf>,
    pub checkpoint: Option<PathBuf>,
    pub backbone: String,
    pub video_stateful_recurrence: bool,
    pub image_size: usize,
    pub patch_tokens_per_image: usize,
    pub batch_size: usize,
    pub warmup: usize,
    pub iterations: usize,
    pub cases: Vec<VisionRolloutProbeStepCase>,
    pub summary: VisionRolloutProbeSummary,
}

pub fn init_vision_rollout_probe_runtime(device: &VisionRolloutProbeDevice) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

pub fn detect_wgpu_adapter_info() -> String {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu adapter");
    let info = adapter.get_info();
    format!("{} ({:?})", info.name, info.device_type)
}

impl VisionRolloutProbeReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let forward_alpha_fit = self
            .summary
            .forward_alpha_fit_all_steps
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let ms_per_step_fit = self
            .summary
            .forward_ms_per_extra_step_fit
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let best_efficiency_step = self
            .summary
            .best_efficiency_step
            .map(|step| format!("s{step}"))
            .unwrap_or_else(|| "-".to_string());
        let best_efficiency_gain = self
            .summary
            .best_efficiency_gain_per_extra_ms
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string());
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Rollout Probe",
            &self.artifact,
        );
        let _ = writeln!(
            out,
            "- adapter: {}\n- config: {}\n- checkpoint: {}\n- backbone: {}\n- stateful video recurrence: {}\n- image size: {}\n- patch tokens / image: {}\n- batch size: {}\n- warmup: {}\n- iterations: {}\n- best total step: s{} ({:.4})\n- gain vs step1 total: {:.4}\n- gain vs step1 patch: {:.4}\n- gain vs step1 cls: {:.4}\n- forward alpha fit over all steps: {}\n- forward ms / extra step fit: {}\n- best total gain / extra ms: {} ({})\n",
            self.adapter,
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            self.checkpoint
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "(random init)".to_string()),
            self.backbone,
            self.video_stateful_recurrence,
            self.image_size,
            self.patch_tokens_per_image,
            self.batch_size,
            self.warmup,
            self.iterations,
            self.summary.best_total_step,
            self.summary.best_total,
            self.summary.gain_vs_step1_total,
            self.summary.gain_vs_step1_patch,
            self.summary.gain_vs_step1_cls,
            forward_alpha_fit,
            ms_per_step_fit,
            best_efficiency_step,
            best_efficiency_gain,
        );
        let _ = writeln!(
            out,
            "| step | bptt | forward ms | fwd x(s1) | alpha | gain(s1) | gain/ms | samples/s | tokens/s | total | patch | cls | patch d(prev) | cls d(prev) | patch d(s1) | cls d(s1) |"
        );
        let _ = writeln!(
            out,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        );
        for case in &self.cases {
            let forward_alpha = case
                .forward_alpha_vs_step1
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let gain_per_ms = case
                .total_gain_per_extra_ms
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let patch_prev = case
                .patch_delta_vs_prev
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let cls_prev = case
                .cls_delta_vs_prev
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let patch_s1 = case
                .patch_delta_vs_step1
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let cls_s1 = case
                .cls_delta_vs_step1
                .map(|value| format!("{value:.4}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {} | {:.3} | {:.3} | {} | {:.4} | {} | {:.3} | {:.1} | {:.4} | {:.4} | {:.4} | {} | {} | {} | {} |",
                case.step,
                case.backprop_steps,
                case.forward_ms,
                case.forward_scale_vs_step1,
                forward_alpha,
                case.gain_vs_step1_total,
                gain_per_ms,
                case.samples_per_sec,
                case.tokens_per_sec,
                case.distill_total,
                case.distill_patch,
                case.distill_cls,
                patch_prev,
                cls_prev,
                patch_s1,
                cls_s1,
            );
        }
        out
    }
}

pub fn run_vision_rollout_probe(
    config: &VisionTrainingConfig,
    probe: &VisionRolloutProbeConfig,
    config_paths: &[PathBuf],
) -> Result<VisionRolloutProbeReport> {
    let device = VisionRolloutProbeDevice::default();
    init_vision_rollout_probe_runtime(&device);
    let batch_size = probe.batch_size.unwrap_or(config.training.batch_size).max(1);
    let (batch, distill_loss) = build_validation_probe_batch(config, batch_size, &device)?;
    let vision = config.vision.build();
    let patch_grid = vision.image_size.div_ceil(vision.patch_size.max(1)).max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let model = load_or_init_model(config, probe.checkpoint.as_deref(), config_paths, &device)?;

    let mut steps = probe
        .steps
        .iter()
        .copied()
        .map(|step| step.max(1))
        .collect::<Vec<_>>();
    steps.push(1);
    steps.sort_unstable();
    steps.dedup();

    let mut cases = run_probe_cases(&model, &batch, &distill_loss, config, &steps, probe);
    annotate_step_scaling(&mut cases);
    let summary = summarize(&cases);
    Ok(VisionRolloutProbeReport {
        artifact: VisionArtifactHeader::new("vision_distill_rollout_probe"),
        benchmark: "burn_dragon vision distill rollout inference probe",
        adapter: detect_wgpu_adapter_info(),
        config: config_paths.to_vec(),
        checkpoint: probe.checkpoint.clone(),
        backbone: format!("{:?}", vision.backbone),
        video_stateful_recurrence: matches!(
            vision.backbone,
            VisionBackboneKind::Pyramid | VisionBackboneKind::Cellular
        ),
        image_size: vision.image_size,
        patch_tokens_per_image,
        batch_size,
        warmup: probe.warmup,
        iterations: probe.iterations,
        cases,
        summary,
    })
}

fn build_validation_probe_batch(
    config: &VisionTrainingConfig,
    batch_size: usize,
    device: &VisionRolloutProbeDevice,
) -> Result<(
    ImageNetBatch<VisionRolloutProbeBackend>,
    burn_dragon::vision::VisionDistillationLossConfig,
)> {
    let vision = config.vision.build();
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        other => {
            return Err(anyhow!(
                "vision_distill_rollout_probe requires distill mode, got {other:?}"
            ));
        }
    };

    let teacher = match &distill.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        other => {
            return Err(anyhow!(
                "vision_distill_rollout_probe currently requires precomputed feature teacher paths, got {other:?}"
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
    let teacher_store = Arc::new(DinoFeatureStore::new(
        &teacher.val_cls_path,
        &teacher.val_patch_path,
        teacher.feature_dim,
        teacher_patch_tokens,
        Some(record_count),
    )?);
    dataset = dataset.with_teacher(teacher_store);
    Ok((dataset.sample_batch::<VisionRolloutProbeBackend>(batch_size, device), distill.loss))
}

fn load_or_init_model(
    config: &VisionTrainingConfig,
    checkpoint: Option<&Path>,
    config_paths: &[PathBuf],
    device: &VisionRolloutProbeDevice,
) -> Result<VisionDragon<VisionRolloutProbeBackend>> {
    if let Some(checkpoint) = checkpoint {
        load_vision_encoder_from_checkpoint::<VisionRolloutProbeBackend>(
            checkpoint,
            None,
            config_paths,
            device,
        )
        .with_context(|| format!("load encoder checkpoint {}", checkpoint.display()))
    } else {
        Ok(VisionDragon::<VisionRolloutProbeBackend>::new(config.vision.build(), device))
    }
}

fn run_probe_cases(
    model: &VisionDragon<VisionRolloutProbeBackend>,
    batch: &ImageNetBatch<VisionRolloutProbeBackend>,
    loss: &burn_dragon::vision::VisionDistillationLossConfig,
    config: &VisionTrainingConfig,
    steps: &[usize],
    probe: &VisionRolloutProbeConfig,
) -> Vec<VisionRolloutProbeStepCase> {
    let images = batch.images.clone();
    let teacher_patch = batch
        .teacher_patch
        .clone()
        .expect("teacher patch features required for rollout probe");
    let teacher_cls = batch
        .teacher_cls
        .clone()
        .expect("teacher cls features required for rollout probe");
    let batch_size = images.shape().dims::<4>()[0];
    let patch_tokens_per_image = teacher_patch.shape().dims::<3>()[1];
    let total_patch_tokens = patch_tokens_per_image * batch_size;
    let configured_backprop = config.training.rollout_backprop_steps;

    let mut cases = Vec::with_capacity(steps.len());
    let mut baseline_patch = None;
    let mut baseline_cls = None;
    let mut prev_patch = None;
    let mut prev_cls = None;

    for step in steps {
        let backprop_steps = configured_backprop.unwrap_or(*step).min(*step).max(1);
        for _ in 0..probe.warmup {
            let output = model.forward_images_steps_rollout_unbounded(
                images.clone(),
                *step,
                backprop_steps,
            );
            sync_tensor(output.patch_tokens.sum() + output.cls_token.sum());
        }
        let forward_ns = measure_avg(probe.iterations, || {
            let output =
                model.forward_images_steps_rollout_unbounded(images.clone(), *step, backprop_steps);
            sync_tensor(output.patch_tokens.sum() + output.cls_token.sum());
        });

        let output =
            model.forward_images_steps_rollout_unbounded(images.clone(), *step, backprop_steps);
        let terms = vision_distillation_loss_terms(
            output.patch_tokens.clone(),
            teacher_patch.clone(),
            output.cls_token.clone(),
            teacher_cls.clone(),
            loss,
        );

        let patch_delta_vs_prev = prev_patch
            .as_ref()
            .map(|prev: &Tensor<VisionRolloutProbeBackend, 3>| {
                mse(output.patch_tokens.clone(), prev.clone())
            });
        let cls_delta_vs_prev = prev_cls
            .as_ref()
            .map(|prev: &Tensor<VisionRolloutProbeBackend, 2>| mse(output.cls_token.clone(), prev.clone()));
        let patch_delta_vs_step1 = baseline_patch
            .as_ref()
            .map(|base: &Tensor<VisionRolloutProbeBackend, 3>| {
                mse(output.patch_tokens.clone(), base.clone())
            });
        let cls_delta_vs_step1 = baseline_cls
            .as_ref()
            .map(|base: &Tensor<VisionRolloutProbeBackend, 2>| mse(output.cls_token.clone(), base.clone()));

        if baseline_patch.is_none() {
            baseline_patch = Some(output.patch_tokens.clone());
            baseline_cls = Some(output.cls_token.clone());
        }
        prev_patch = Some(output.patch_tokens);
        prev_cls = Some(output.cls_token);

        let forward_ms = forward_ns / 1_000_000.0;
        let seconds = forward_ns / 1_000_000_000.0;
        cases.push(VisionRolloutProbeStepCase {
            step: *step,
            backprop_steps,
            forward_ms,
            samples_per_sec: batch_size as f64 / seconds,
            tokens_per_sec: total_patch_tokens as f64 / seconds,
            forward_scale_vs_step1: 1.0,
            forward_alpha_vs_step1: None,
            distill_total: scalar(terms.total),
            distill_patch: scalar(terms.patch),
            distill_cls: scalar(terms.cls),
            gain_vs_step1_total: 0.0,
            gain_vs_step1_patch: 0.0,
            gain_vs_step1_cls: 0.0,
            total_gain_per_extra_ms: None,
            patch_delta_vs_prev,
            cls_delta_vs_prev,
            patch_delta_vs_step1,
            cls_delta_vs_step1,
        });
    }

    cases
}

fn annotate_step_scaling(cases: &mut [VisionRolloutProbeStepCase]) {
    if cases.is_empty() {
        return;
    }
    let base_step = cases[0].step as f64;
    let base_forward = cases[0].forward_ms.max(1e-9);
    let base_total = cases[0].distill_total;
    let base_patch = cases[0].distill_patch;
    let base_cls = cases[0].distill_cls;

    for case in cases.iter_mut() {
        case.forward_scale_vs_step1 = case.forward_ms / base_forward;
        case.gain_vs_step1_total = base_total - case.distill_total;
        case.gain_vs_step1_patch = base_patch - case.distill_patch;
        case.gain_vs_step1_cls = base_cls - case.distill_cls;

        let step_ratio = case.step as f64 / base_step;
        if step_ratio > 1.0 {
            case.forward_alpha_vs_step1 =
                Some(case.forward_scale_vs_step1.ln() / step_ratio.ln());
            let extra_ms = case.forward_ms - base_forward;
            if extra_ms > 0.0 {
                case.total_gain_per_extra_ms = Some(case.gain_vs_step1_total / extra_ms);
            }
        }
    }
}

fn summarize(cases: &[VisionRolloutProbeStepCase]) -> VisionRolloutProbeSummary {
    let first = cases.first().expect("at least one rollout probe case");
    let best_total = cases
        .iter()
        .min_by(|a, b| a.distill_total.total_cmp(&b.distill_total))
        .expect("best total case");
    let best_patch = cases
        .iter()
        .min_by(|a, b| a.distill_patch.total_cmp(&b.distill_patch))
        .expect("best patch case");
    let best_cls = cases
        .iter()
        .min_by(|a, b| a.distill_cls.total_cmp(&b.distill_cls))
        .expect("best cls case");
    let best_efficiency = cases
        .iter()
        .filter_map(|case| {
            case.total_gain_per_extra_ms
                .filter(|gain| gain.is_finite())
                .map(|gain| (case.step, gain))
        })
        .max_by(|a, b| a.1.total_cmp(&b.1));

    VisionRolloutProbeSummary {
        best_total_step: best_total.step,
        best_total: best_total.distill_total,
        best_patch_step: best_patch.step,
        best_patch: best_patch.distill_patch,
        best_cls_step: best_cls.step,
        best_cls: best_cls.distill_cls,
        gain_vs_step1_total: first.distill_total - best_total.distill_total,
        gain_vs_step1_patch: first.distill_patch - best_patch.distill_patch,
        gain_vs_step1_cls: first.distill_cls - best_cls.distill_cls,
        forward_alpha_fit_all_steps: fit_alpha_by_step(cases, |case| case.forward_ms),
        forward_ms_per_extra_step_fit: fit_linear_ms_per_step(cases, |case| case.forward_ms),
        best_efficiency_step: best_efficiency.map(|(step, _)| step),
        best_efficiency_gain_per_extra_ms: best_efficiency.map(|(_, gain)| gain),
    }
}

fn measure_avg(mut iterations: usize, mut f: impl FnMut()) -> f64 {
    iterations = iterations.max(1);
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    start.elapsed().as_nanos() as f64 / iterations as f64
}

fn sync_tensor<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

fn scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
    tensor
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor data")[0] as f64
}

fn mse<B: BackendTrait, const D: usize>(a: Tensor<B, D>, b: Tensor<B, D>) -> f64 {
    scalar((a - b).powf_scalar(2.0).mean())
}

fn fit_alpha_by_step(
    cases: &[VisionRolloutProbeStepCase],
    metric: impl Fn(&VisionRolloutProbeStepCase) -> f64,
) -> Option<f64> {
    if cases.len() < 2 {
        return None;
    }
    let points = cases
        .iter()
        .filter_map(|case| {
            let x = (case.step as f64).ln();
            let y = metric(case).max(1e-9).ln();
            if x.is_finite() && y.is_finite() {
                Some((x, y))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    fit_line_slope(&points)
}

fn fit_linear_ms_per_step(
    cases: &[VisionRolloutProbeStepCase],
    metric: impl Fn(&VisionRolloutProbeStepCase) -> f64,
) -> Option<f64> {
    if cases.len() < 2 {
        return None;
    }
    let points = cases
        .iter()
        .filter_map(|case| {
            let x = case.step as f64;
            let y = metric(case);
            if x.is_finite() && y.is_finite() {
                Some((x, y))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    fit_line_slope(&points)
}

fn fit_line_slope(points: &[(f64, f64)]) -> Option<f64> {
    if points.len() < 2 {
        return None;
    }
    let n = points.len() as f64;
    let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
    let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;
    let denom = points
        .iter()
        .map(|(x, _)| {
            let dx = *x - mean_x;
            dx * dx
        })
        .sum::<f64>();
    if denom <= 0.0 {
        return None;
    }
    let numer = points
        .iter()
        .map(|(x, y)| (*x - mean_x) * (*y - mean_y))
        .sum::<f64>();
    Some(numer / denom)
}
