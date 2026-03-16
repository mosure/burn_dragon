use std::fmt::Write as _;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_dragon::vision::train::bench::VisionDistillTrainStepBench;
use burn_dragon::vision::{
    ImageNetBatch, VisionArtifactHeader, VisionDragon, VisionTeacherConfig, VisionTrainingConfig,
    VisionTrainingModeConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

pub type VisionResolutionBenchInnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionResolutionBenchTrainBackend = Autodiff<VisionResolutionBenchInnerBackend>;
pub type VisionResolutionBenchDevice =
    <VisionResolutionBenchTrainBackend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VisionResolutionBenchConfig {
    pub config_paths: Vec<PathBuf>,
    pub resolutions: Vec<usize>,
    pub warmup: usize,
    pub iterations: usize,
    pub batch_size: Option<usize>,
}

#[derive(Clone, Serialize)]
pub struct VisionResolutionBenchCaseResult {
    pub resolution: usize,
    pub patch_grid: usize,
    pub patch_tokens_per_image: usize,
    pub total_patch_tokens: usize,
    pub batch_size: usize,
    pub rollout_steps: usize,
    pub backprop_steps: usize,
    pub forward_ms: f64,
    pub train_step_ms: f64,
    pub forward_samples_per_sec: f64,
    pub train_samples_per_sec: f64,
    pub forward_tokens_per_sec: f64,
    pub train_tokens_per_sec: f64,
    pub forward_scale_vs_base: f64,
    pub train_scale_vs_base: f64,
    pub forward_alpha_vs_base: Option<f64>,
    pub train_alpha_vs_base: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct VisionResolutionBenchScalingSummary {
    pub forward_alpha_fit_all: Option<f64>,
    pub train_alpha_fit_all: Option<f64>,
}

#[derive(Clone, Serialize)]
pub struct VisionResolutionBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub config: Vec<PathBuf>,
    pub warmup: usize,
    pub iterations: usize,
    pub cases: Vec<VisionResolutionBenchCaseResult>,
    pub summary: VisionResolutionBenchScalingSummary,
}

pub fn init_vision_resolution_bench_runtime(device: &VisionResolutionBenchDevice) {
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

impl VisionResolutionBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        let forward_fit = self
            .summary
            .forward_alpha_fit_all
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string());
        let train_fit = self
            .summary
            .train_alpha_fit_all
            .map(|value| format!("{value:.3}"))
            .unwrap_or_else(|| "-".to_string());
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Distill Resolution Bench",
            &self.artifact,
        );
        let _ = writeln!(
            out,
            "- adapter: {}\n- warmup: {}\n- iterations: {}\n- config: {}\n- forward alpha fit over all resolutions: {}\n- train-step alpha fit over all resolutions: {}\n",
            self.adapter,
            self.warmup,
            self.iterations,
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
            forward_fit,
            train_fit,
        );
        let _ = writeln!(
            out,
            "| res | grid | tokens/img | total tokens | forward ms | train-step ms | forward tok/s | train tok/s | fwd x | train x | fwd alpha | train alpha |"
        );
        let _ = writeln!(
            out,
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        );
        for case in &self.cases {
            let forward_alpha = case
                .forward_alpha_vs_base
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let train_alpha = case
                .train_alpha_vs_base
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {:.3} | {:.3} | {:.1} | {:.1} | {:.3} | {:.3} | {} | {} |",
                case.resolution,
                case.patch_grid,
                case.patch_tokens_per_image,
                case.total_patch_tokens,
                case.forward_ms,
                case.train_step_ms,
                case.forward_tokens_per_sec,
                case.train_tokens_per_sec,
                case.forward_scale_vs_base,
                case.train_scale_vs_base,
                forward_alpha,
                train_alpha,
            );
        }
        out
    }
}

pub fn run_vision_resolution_bench(
    base_config: &VisionTrainingConfig,
    bench: &VisionResolutionBenchConfig,
) -> Result<VisionResolutionBenchReport> {
    let device = VisionResolutionBenchDevice::default();
    init_vision_resolution_bench_runtime(&device);
    let mut resolutions = bench.resolutions.clone();
    resolutions.sort_unstable();
    resolutions.dedup();

    let batch_size = bench
        .batch_size
        .unwrap_or(base_config.training.batch_size)
        .max(1);
    let mut cases = resolutions
        .into_iter()
        .map(|resolution| run_case(base_config, resolution, batch_size, &device, bench))
        .collect::<Vec<_>>();
    annotate_scaling(&mut cases);

    let summary = VisionResolutionBenchScalingSummary {
        forward_alpha_fit_all: fit_alpha(&cases, |case| case.forward_ms),
        train_alpha_fit_all: fit_alpha(&cases, |case| case.train_step_ms),
    };

    Ok(VisionResolutionBenchReport {
        artifact: VisionArtifactHeader::new("vision_distill_resolution_bench"),
        benchmark: "burn_dragon vision distill resolution scaling benchmark",
        adapter: detect_wgpu_adapter_info(),
        config: bench.config_paths.clone(),
        warmup: bench.warmup,
        iterations: bench.iterations,
        cases,
        summary,
    })
}

fn run_case(
    base_config: &VisionTrainingConfig,
    resolution: usize,
    batch_size: usize,
    device: &VisionResolutionBenchDevice,
    bench: &VisionResolutionBenchConfig,
) -> VisionResolutionBenchCaseResult {
    let config = mutate_resolution(base_config.clone(), resolution, batch_size);
    config.validate().unwrap_or_else(|err| {
        panic!("invalid mutated config at resolution {resolution}: {err}")
    });

    let vision = config.vision.build();
    let rollout_steps = config
        .training
        .rollout_max_steps
        .unwrap_or(vision.steps)
        .max(1);
    let backprop_steps = config
        .training
        .rollout_backprop_steps
        .unwrap_or(rollout_steps)
        .min(rollout_steps)
        .max(1);
    let patch_grid = vision.image_size.div_ceil(vision.patch_size.max(1)).max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let total_patch_tokens = patch_tokens_per_image * batch_size;

    let images_inner = Tensor::<VisionResolutionBenchInnerBackend, 4>::random(
        [
            batch_size,
            vision.in_channels,
            vision.image_size,
            vision.image_size,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let model_inner = VisionDragon::<VisionResolutionBenchInnerBackend>::new(vision.clone(), device);
    for _ in 0..bench.warmup {
        sync_tensor(
            model_inner
                .forward_images_steps_rollout(
                    images_inner.clone(),
                    rollout_steps,
                    backprop_steps,
                )
                .patch_tokens
                .sum(),
        );
    }
    let forward_ns = measure_avg(bench.iterations, || {
        let output =
            model_inner.forward_images_steps_rollout(images_inner.clone(), rollout_steps, backprop_steps);
        sync_tensor(output.patch_tokens.sum() + output.cls_token.sum());
    });

    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        other => panic!("vision_distill_resolution_bench requires distill mode, got {other:?}"),
    };
    let synthetic_batch =
        make_synthetic_distill_batch(&vision, rollout_steps, backprop_steps, batch_size, device);
    let mut train_bench = VisionDistillTrainStepBench::<VisionResolutionBenchTrainBackend>::new(
        vision,
        distill,
        &config.training,
        &config.optimizer,
        device,
    )
    .expect("distill train-step bench");

    for _ in 0..bench.warmup {
        sync_tensor(train_bench.train_step(synthetic_batch.clone()));
    }
    let train_step_ns = measure_avg(bench.iterations, || {
        sync_tensor(train_bench.train_step(synthetic_batch.clone()));
    });

    let forward_ms = forward_ns / 1_000_000.0;
    let train_step_ms = train_step_ns / 1_000_000.0;
    let forward_samples_per_sec = batch_size as f64 / (forward_ns / 1_000_000_000.0);
    let train_samples_per_sec = batch_size as f64 / (train_step_ns / 1_000_000_000.0);
    let forward_tokens_per_sec = total_patch_tokens as f64 / (forward_ns / 1_000_000_000.0);
    let train_tokens_per_sec = total_patch_tokens as f64 / (train_step_ns / 1_000_000_000.0);

    VisionResolutionBenchCaseResult {
        resolution,
        patch_grid,
        patch_tokens_per_image,
        total_patch_tokens,
        batch_size,
        rollout_steps,
        backprop_steps,
        forward_ms,
        train_step_ms,
        forward_samples_per_sec,
        train_samples_per_sec,
        forward_tokens_per_sec,
        train_tokens_per_sec,
        forward_scale_vs_base: 1.0,
        train_scale_vs_base: 1.0,
        forward_alpha_vs_base: None,
        train_alpha_vs_base: None,
    }
}

fn mutate_resolution(
    mut config: VisionTrainingConfig,
    resolution: usize,
    batch_size: usize,
) -> VisionTrainingConfig {
    let patch_size = config.vision.patch_size.max(1);
    let patch_grid = resolution.div_ceil(patch_size).max(1);
    config.training.batch_size = batch_size;
    config.vision.image_size = resolution;
    config.vision.pos_max_height = Some(patch_grid);
    config.vision.pos_max_width = Some(patch_grid);
    config.augment.image_size = resolution;
    config.augment.resize_short = resolution;
    if let VisionTrainingModeConfig::Distill(distill) = &mut config.mode {
        match &mut distill.teacher {
            VisionTeacherConfig::Features(teacher) => {
                if teacher.patch_tokens.is_some() {
                    teacher.patch_tokens = Some(patch_grid * patch_grid);
                }
            }
            VisionTeacherConfig::Model(teacher) => {
                if teacher.image_size.is_some() {
                    teacher.image_size = Some(resolution);
                }
                if teacher.patch_size.is_some() {
                    teacher.patch_size = Some(patch_size);
                }
                if teacher.patch_tokens.is_some() {
                    teacher.patch_tokens = Some(patch_grid * patch_grid);
                }
            }
        }
    }
    config
}

fn make_synthetic_distill_batch(
    vision: &burn_dragon::vision::VisionDragonConfig,
    rollout_steps: usize,
    backprop_steps: usize,
    batch_size: usize,
    device: &VisionResolutionBenchDevice,
) -> ImageNetBatch<VisionResolutionBenchTrainBackend> {
    let images = Tensor::<VisionResolutionBenchTrainBackend, 4>::random(
        [
            batch_size,
            vision.in_channels,
            vision.image_size,
            vision.image_size,
        ],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let labels = Tensor::<VisionResolutionBenchTrainBackend, 1, Int>::from_data(
        TensorData::new(vec![0i64; batch_size], [batch_size]),
        device,
    );
    let dry_model = VisionDragon::<VisionResolutionBenchTrainBackend>::new(vision.clone(), device);
    let dry_output =
        dry_model.forward_images_steps_rollout(images.clone(), rollout_steps, backprop_steps);
    let patch_shape = dry_output.patch_tokens.shape().dims::<3>();
    let cls_shape = dry_output.cls_token.shape().dims::<2>();
    let teacher_patch = Tensor::<VisionResolutionBenchTrainBackend, 3>::random(
        patch_shape,
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let teacher_cls = Tensor::<VisionResolutionBenchTrainBackend, 2>::random(
        cls_shape,
        Distribution::Normal(0.0, 1.0),
        device,
    );
    ImageNetBatch::new(
        images,
        None,
        None,
        None,
        None,
        None,
        labels,
        Some(teacher_patch),
        Some(teacher_cls),
    )
}

fn annotate_scaling(cases: &mut [VisionResolutionBenchCaseResult]) {
    if cases.is_empty() {
        return;
    }
    let base_tokens = cases[0].total_patch_tokens as f64;
    let base_forward = cases[0].forward_ms.max(1e-9);
    let base_train = cases[0].train_step_ms.max(1e-9);

    for case in cases.iter_mut() {
        case.forward_scale_vs_base = case.forward_ms / base_forward;
        case.train_scale_vs_base = case.train_step_ms / base_train;

        let token_ratio = case.total_patch_tokens as f64 / base_tokens;
        if token_ratio > 1.0 {
            case.forward_alpha_vs_base = Some(case.forward_scale_vs_base.ln() / token_ratio.ln());
            case.train_alpha_vs_base = Some(case.train_scale_vs_base.ln() / token_ratio.ln());
        }
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

fn fit_alpha(
    cases: &[VisionResolutionBenchCaseResult],
    metric: impl Fn(&VisionResolutionBenchCaseResult) -> f64,
) -> Option<f64> {
    if cases.len() < 2 {
        return None;
    }
    let points = cases
        .iter()
        .filter_map(|case| {
            let x = (case.total_patch_tokens as f64).ln();
            let y = metric(case).max(1e-9).ln();
            if x.is_finite() && y.is_finite() {
                Some((x, y))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
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
