use std::fmt::Write as _;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn_autodiff::Autodiff;
use burn_dragon::core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
use burn_dragon::vision::train::bench::VisionVideoLejepaTrainStepBench;
use burn_dragon::vision::{
    MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
    SpatialPositionalEncodingKind, VisionArtifactHeader, VisionAttentionMode, VisionBackboneKind,
    VisionDragonConfig, VisionLatentActivation, VisionNormalize, VisionPatchEmbedMode,
    VisionVideoLejepaConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

pub type VideoLejepaBenchInnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VideoLejepaBenchTrainBackend = Autodiff<VideoLejepaBenchInnerBackend>;
pub type VideoLejepaBenchDevice = <VideoLejepaBenchTrainBackend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VideoLejepaBenchConfig {
    pub warmup: usize,
    pub iterations: usize,
}

#[derive(Clone, Copy, Serialize)]
pub struct VideoLejepaBenchCase {
    pub name: &'static str,
    pub batch_size: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub projection_dim: usize,
    pub projection_hidden_dim: usize,
    pub spatial_steps: usize,
    pub n_head: usize,
    pub temporal_layers: usize,
    pub temporal_heads: usize,
    pub temporal_mlp_multiplier: usize,
    pub rollout_fast_steps: usize,
    pub context_frames: usize,
    pub target_frames: usize,
    pub max_records: usize,
}

#[derive(Clone, Serialize)]
pub struct VideoLejepaBenchCaseResult {
    pub case: VideoLejepaBenchCase,
    pub warmup: usize,
    pub iterations: usize,
    pub baseline_forward_ms: f64,
    pub fused_forward_ms: f64,
    pub forward_speedup_x: f64,
    pub baseline_train_step_ms: f64,
    pub fused_train_step_ms: f64,
    pub train_step_speedup_x: f64,
    pub baseline_forward_frames_per_sec: f64,
    pub fused_forward_frames_per_sec: f64,
    pub baseline_train_frames_per_sec: f64,
    pub fused_train_frames_per_sec: f64,
    pub forward_loss_abs_diff: f32,
    pub train_loss_abs_diff: f32,
}

#[derive(Clone, Serialize)]
pub struct VideoLejepaBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub warmup: usize,
    pub iterations: usize,
    pub cases: Vec<VideoLejepaBenchCaseResult>,
}

const CASES: &[VideoLejepaBenchCase] = &[
    VideoLejepaBenchCase {
        name: "tiny_fs1",
        batch_size: 8,
        image_size: 32,
        patch_size: 4,
        embed_dim: 32,
        projection_dim: 16,
        projection_hidden_dim: 32,
        spatial_steps: 2,
        n_head: 4,
        temporal_layers: 2,
        temporal_heads: 4,
        temporal_mlp_multiplier: 2,
        rollout_fast_steps: 1,
        context_frames: 4,
        target_frames: 2,
        max_records: 64,
    },
    VideoLejepaBenchCase {
        name: "tiny_fs4",
        batch_size: 8,
        image_size: 32,
        patch_size: 4,
        embed_dim: 32,
        projection_dim: 16,
        projection_hidden_dim: 32,
        spatial_steps: 2,
        n_head: 4,
        temporal_layers: 2,
        temporal_heads: 4,
        temporal_mlp_multiplier: 2,
        rollout_fast_steps: 4,
        context_frames: 4,
        target_frames: 2,
        max_records: 64,
    },
    VideoLejepaBenchCase {
        name: "small_fs4",
        batch_size: 8,
        image_size: 32,
        patch_size: 4,
        embed_dim: 64,
        projection_dim: 32,
        projection_hidden_dim: 64,
        spatial_steps: 2,
        n_head: 8,
        temporal_layers: 3,
        temporal_heads: 8,
        temporal_mlp_multiplier: 2,
        rollout_fast_steps: 4,
        context_frames: 4,
        target_frames: 2,
        max_records: 64,
    },
];

pub fn init_video_lejepa_bench_runtime(device: &VideoLejepaBenchDevice) {
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

impl VideoLejepaBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Video LEJEPA Fused Benchmark",
            &self.artifact,
        );
        let _ = writeln!(&mut out, "- adapter: {}", self.adapter);
        let _ = writeln!(&mut out, "- warmup: {}", self.warmup);
        let _ = writeln!(&mut out, "- iterations: {}", self.iterations);
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| case | forward ms (base) | forward ms (fused) | forward speedup | train ms (base) | train ms (fused) | train speedup | fused train fps | forward loss drift | train loss drift |"
        );
        let _ = writeln!(
            &mut out,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for case in &self.cases {
            let _ = writeln!(
                &mut out,
                "| {} | {:.2} | {:.2} | {:.2}x | {:.2} | {:.2} | {:.2}x | {:.1} | {:.5} | {:.5} |",
                case.case.name,
                case.baseline_forward_ms,
                case.fused_forward_ms,
                case.forward_speedup_x,
                case.baseline_train_step_ms,
                case.fused_train_step_ms,
                case.train_step_speedup_x,
                case.fused_train_frames_per_sec,
                case.forward_loss_abs_diff,
                case.train_loss_abs_diff,
            );
        }
        out
    }
}

pub fn run_video_lejepa_bench(config: &VideoLejepaBenchConfig) -> Result<VideoLejepaBenchReport> {
    let device = VideoLejepaBenchDevice::default();
    init_video_lejepa_bench_runtime(&device);
    Ok(VideoLejepaBenchReport {
        artifact: VisionArtifactHeader::new("video_lejepa_bench"),
        benchmark: "burn_dragon video LEJEPA temporal fused benchmark",
        adapter: detect_wgpu_adapter_info(),
        warmup: config.warmup,
        iterations: config.iterations,
        cases: run_all_cases(&device, config),
    })
}

fn run_all_cases(
    device: &VideoLejepaBenchDevice,
    config: &VideoLejepaBenchConfig,
) -> Vec<VideoLejepaBenchCaseResult> {
    CASES
        .iter()
        .copied()
        .map(|case| run_case(case, device, config))
        .collect()
}

fn run_case(
    case: VideoLejepaBenchCase,
    device: &VideoLejepaBenchDevice,
    config: &VideoLejepaBenchConfig,
) -> VideoLejepaBenchCaseResult {
    let batch = sample_batch(case, device);
    let batch_frames = (case.batch_size * (case.context_frames + case.target_frames)) as f64;

    let (forward_loss_abs_diff, train_loss_abs_diff) = parity_snapshot(case, batch.clone(), device);

    let baseline_forward_bench = build_bench(case, false, device);
    let fused_forward_bench = build_bench(case, true, device);
    let mut baseline_train_bench = build_bench(case, false, device);
    let mut fused_train_bench = build_bench(case, true, device);

    for _ in 0..config.warmup {
        let _ = baseline_forward_bench.forward_loss(batch.clone());
        let _ = fused_forward_bench.forward_loss(batch.clone());
        let _ = baseline_train_bench.train_step(batch.clone());
        let _ = fused_train_bench.train_step(batch.clone());
    }

    let baseline_forward_ns = (0..config.iterations)
        .map(|_| {
            time_ns(|| {
                let _ = baseline_forward_bench.forward_loss(batch.clone());
            })
        })
        .collect::<Vec<_>>();
    let fused_forward_ns = (0..config.iterations)
        .map(|_| {
            time_ns(|| {
                let _ = fused_forward_bench.forward_loss(batch.clone());
            })
        })
        .collect::<Vec<_>>();
    let baseline_train_ns = (0..config.iterations)
        .map(|_| {
            time_ns(|| {
                let _ = baseline_train_bench.train_step(batch.clone());
            })
        })
        .collect::<Vec<_>>();
    let fused_train_ns = (0..config.iterations)
        .map(|_| {
            time_ns(|| {
                let _ = fused_train_bench.train_step(batch.clone());
            })
        })
        .collect::<Vec<_>>();

    let baseline_forward_avg = mean_u128(&baseline_forward_ns);
    let fused_forward_avg = mean_u128(&fused_forward_ns);
    let baseline_train_avg = mean_u128(&baseline_train_ns);
    let fused_train_avg = mean_u128(&fused_train_ns);

    VideoLejepaBenchCaseResult {
        case,
        warmup: config.warmup,
        iterations: config.iterations,
        baseline_forward_ms: ns_to_ms(baseline_forward_avg),
        fused_forward_ms: ns_to_ms(fused_forward_avg),
        forward_speedup_x: baseline_forward_avg / fused_forward_avg,
        baseline_train_step_ms: ns_to_ms(baseline_train_avg),
        fused_train_step_ms: ns_to_ms(fused_train_avg),
        train_step_speedup_x: baseline_train_avg / fused_train_avg,
        baseline_forward_frames_per_sec: batch_frames / (baseline_forward_avg / 1e9),
        fused_forward_frames_per_sec: batch_frames / (fused_forward_avg / 1e9),
        baseline_train_frames_per_sec: batch_frames / (baseline_train_avg / 1e9),
        fused_train_frames_per_sec: batch_frames / (fused_train_avg / 1e9),
        forward_loss_abs_diff,
        train_loss_abs_diff,
    }
}

fn sample_batch(
    case: VideoLejepaBenchCase,
    device: &VideoLejepaBenchDevice,
) -> burn_dragon::vision::VideoClipBatch<VideoLejepaBenchTrainBackend> {
    let normalize = VisionNormalize::new([0.5; 3], [0.5; 3]);
    let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split: MovingMnistSplit::Train,
        frame_size: case.image_size,
        digit_size: case.image_size.saturating_sub(12).max(12),
        in_channels: 3,
        context_len: case.context_frames,
        target_len: case.target_frames,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records: Some(case.max_records),
        normalize,
        min_velocity: 0.8,
        max_velocity: 2.0,
        seed: 2026,
    })
    .expect("moving mnist dataset");
    dataset.sample_batch::<VideoLejepaBenchTrainBackend>(case.batch_size, device)
}

fn build_bench(
    case: VideoLejepaBenchCase,
    fused_temporal: bool,
    device: &VideoLejepaBenchDevice,
) -> VisionVideoLejepaTrainStepBench<VideoLejepaBenchTrainBackend> {
    let vision = build_vision_config(case);
    let video = build_video_config(case, fused_temporal);
    let training = burn_dragon::vision::VisionTrainingHyperparameters {
        batch_size: case.batch_size,
        max_iters: 1,
        log_frequency: 1,
        rollout_min_steps: Some(case.spatial_steps),
        rollout_max_steps: Some(case.spatial_steps),
        rollout_backprop_steps: Some(case.spatial_steps),
        ..Default::default()
    };
    let optimizer = burn_dragon::train::OptimizerConfig {
        learning_rate: 1e-3,
        weight_decay: 0.0,
        lr_schedule: None,
        grad_clip_norm: None,
        grad_clip_value: None,
    };
    <VideoLejepaBenchTrainBackend as BackendTrait>::seed(device, 4_242);
    VisionVideoLejepaTrainStepBench::<VideoLejepaBenchTrainBackend>::new(
        vision, video, &training, &optimizer, 10, device,
    )
    .expect("video bench")
}

fn build_vision_config(case: VideoLejepaBenchCase) -> VisionDragonConfig {
    VisionDragonConfig {
        image_size: case.image_size,
        patch_size: case.patch_size,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Dense,
        in_channels: 3,
        embed_dim: case.embed_dim,
        steps: case.spatial_steps,
        n_head: case.n_head,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: case.projection_dim,
        projection_hidden_dim: case.projection_hidden_dim,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        normalization: burn_dragon::core::DragonNormConfig::default(),
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: case.image_size.div_ceil(case.patch_size),
        pos_max_width: case.image_size.div_ceil(case.patch_size),
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: Default::default(),
    }
}

fn build_video_config(case: VideoLejepaBenchCase, fused_temporal: bool) -> VisionVideoLejepaConfig {
    let mut video = VisionVideoLejepaConfig {
        context_frames: case.context_frames,
        target_frames: case.target_frames,
        ..VisionVideoLejepaConfig::default()
    };
    video.teacher_ema.enabled = true;
    video.teacher_ema.decay = 0.996;
    video.loss.probe_weight = 0.25;
    video.loss.cosine_weight = 0.1;
    video.loss.sigreg.enabled = true;
    video.loss.sigreg.lambda = 0.02;
    video.temporal.n_layer = case.temporal_layers;
    video.temporal.n_head = case.temporal_heads;
    video.temporal.mlp_internal_dim_multiplier = case.temporal_mlp_multiplier;
    video.temporal.rollout_fast_steps_per_slow_step = case.rollout_fast_steps;
    video.temporal.fused = true;
    video.temporal.wgpu_recurrent_kernel = fused_temporal;
    video.temporal.wgpu_rollout_fused = fused_temporal;
    video.temporal.latent_block_size = 8;
    video.temporal.time_block_size = 8;
    video
}

fn parity_snapshot(
    case: VideoLejepaBenchCase,
    batch: burn_dragon::vision::VideoClipBatch<VideoLejepaBenchTrainBackend>,
    device: &VideoLejepaBenchDevice,
) -> (f32, f32) {
    let mut baseline = build_bench(case, false, device);
    let mut fused = build_bench(case, true, device);

    let baseline_forward = baseline
        .forward_loss(batch.clone())
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("baseline forward")[0];
    let fused_forward = fused
        .forward_loss(batch.clone())
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused forward")[0];

    let baseline_train = baseline
        .train_step(batch.clone())
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("baseline train")[0];
    let fused_train = fused
        .train_step(batch)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused train")[0];

    (
        (baseline_forward - fused_forward).abs(),
        (baseline_train - fused_train).abs(),
    )
}

fn time_ns<F>(mut f: F) -> u128
where
    F: FnMut(),
{
    let start = Instant::now();
    f();
    start.elapsed().as_nanos()
}

fn mean_u128(values: &[u128]) -> f64 {
    let total = values.iter().copied().sum::<u128>() as f64;
    total / values.len().max(1) as f64
}

fn ns_to_ms(value: f64) -> f64 {
    value / 1_000_000.0
}
