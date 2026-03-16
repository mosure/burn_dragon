use std::fmt::Write as _;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon::vision::{
    VisionArtifactHeader, VisionBackboneKind, VisionDragon, VisionRolloutScheduleBenchAdapter,
    VisionTrainingConfig, VisionTrainingModeConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

pub type VisionRolloutScheduleBenchBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionRolloutScheduleBenchDevice =
    <VisionRolloutScheduleBenchBackend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VisionRolloutScheduleBenchConfig {
    pub steps: Option<Vec<usize>>,
    pub warmup: usize,
    pub iterations: usize,
    pub batch_size: Option<usize>,
}

#[derive(Clone, Serialize)]
pub struct VisionRolloutScheduleBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub backbone: VisionBackboneKind,
    pub batch_size: usize,
    pub image_size: usize,
    pub patch_tokens_per_image: usize,
    pub supervision_steps: Vec<usize>,
    pub repeated_ms: f64,
    pub scheduled_ms: f64,
    pub speedup: f64,
    pub repeated_tokens_per_sec: f64,
    pub scheduled_tokens_per_sec: f64,
    pub dense_schedule_reuse_enabled: bool,
}

pub fn init_vision_rollout_schedule_bench_runtime(device: &VisionRolloutScheduleBenchDevice) {
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

impl VisionRolloutScheduleBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(&mut out, self.benchmark, &self.artifact);
        let _ = writeln!(out, "- Adapter: {}", self.adapter);
        let _ = writeln!(out, "- Backbone: {:?}", self.backbone);
        let _ = writeln!(out, "- Batch size: {}", self.batch_size);
        let _ = writeln!(out, "- Image size: {}", self.image_size);
        let _ = writeln!(out, "- Patch tokens/image: {}", self.patch_tokens_per_image);
        let _ = writeln!(out, "- Supervision steps: {:?}", self.supervision_steps);
        let _ = writeln!(
            out,
            "- Dense schedule reuse enabled: {}",
            self.dense_schedule_reuse_enabled
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "## Timing");
        let _ = writeln!(out);
        let _ = writeln!(out, "- Repeated rollout ms: {:.3}", self.repeated_ms);
        let _ = writeln!(out, "- Scheduled rollout ms: {:.3}", self.scheduled_ms);
        let _ = writeln!(out, "- Speedup: {:.3}x", self.speedup);
        let _ = writeln!(out, "- Repeated tokens/sec: {:.2}", self.repeated_tokens_per_sec);
        let _ = writeln!(out, "- Scheduled tokens/sec: {:.2}", self.scheduled_tokens_per_sec);
        out
    }
}

pub fn run_vision_rollout_schedule_bench(
    config: &VisionTrainingConfig,
    bench: &VisionRolloutScheduleBenchConfig,
) -> Result<VisionRolloutScheduleBenchReport> {
    let device = VisionRolloutScheduleBenchDevice::default();
    init_vision_rollout_schedule_bench_runtime(&device);
    config.validate()?;
    let vision = config.vision.build();
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        other => panic!("vision_distill_rollout_schedule_bench requires distill mode, got {other:?}"),
    };
    let batch_size = bench.batch_size.unwrap_or(config.training.batch_size).max(1);
    let rollout_steps = config
        .training
        .rollout_max_steps
        .unwrap_or(vision.steps)
        .max(1);
    let model = VisionDragon::<VisionRolloutScheduleBenchBackend>::new(vision.clone(), &device);
    let rollout_bench = VisionRolloutScheduleBenchAdapter::new(&model);
    let steps = bench.steps.clone().unwrap_or_else(|| {
        VisionRolloutScheduleBenchAdapter::<VisionRolloutScheduleBenchBackend>::default_distill_steps(
            &distill,
            rollout_steps,
        )
    });
    let steps = VisionRolloutScheduleBenchAdapter::<VisionRolloutScheduleBenchBackend>::normalize_steps(
        &steps,
        rollout_steps,
    );
    assert!(
        steps.len() >= 2,
        "vision_distill_rollout_schedule_bench requires at least two supervision steps"
    );

    let patch_grid = vision.image_size.div_ceil(vision.patch_size.max(1)).max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let total_tokens = patch_tokens_per_image * batch_size * steps.len();
    let images = Tensor::<VisionRolloutScheduleBenchBackend, 4>::random(
        [batch_size, vision.in_channels, vision.image_size, vision.image_size],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let teacher_patch = Tensor::<VisionRolloutScheduleBenchBackend, 3>::random(
        [batch_size, patch_tokens_per_image, vision.projection_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let teacher_cls = Tensor::<VisionRolloutScheduleBenchBackend, 2>::random(
        [batch_size, vision.projection_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let schedule = steps
        .iter()
        .map(|step| {
            (
                *step,
                config
                    .training
                    .rollout_backprop_steps
                    .unwrap_or(*step)
                    .min(*step)
                    .max(1),
            )
        })
        .collect::<Vec<_>>();
    for _ in 0..bench.warmup {
        sync_tensor(run_repeated(
            &rollout_bench,
            images.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &schedule,
            &distill.loss,
        ));
        sync_tensor(run_scheduled(
            &rollout_bench,
            images.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &schedule,
            &distill.loss,
        ));
    }

    let repeated_ns = measure_avg(bench.iterations, || {
        sync_tensor(run_repeated(
            &rollout_bench,
            images.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &schedule,
            &distill.loss,
        ));
    });
    let scheduled_ns = measure_avg(bench.iterations, || {
        sync_tensor(run_scheduled(
            &rollout_bench,
            images.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &schedule,
            &distill.loss,
        ));
    });

    let repeated_ms = repeated_ns / 1_000_000.0;
    let scheduled_ms = scheduled_ns / 1_000_000.0;
    let repeated_tokens_per_sec = total_tokens as f64 / (repeated_ns / 1_000_000_000.0);
    let scheduled_tokens_per_sec = total_tokens as f64 / (scheduled_ns / 1_000_000_000.0);
    let speedup = if scheduled_ns > 0.0 {
        repeated_ns / scheduled_ns
    } else {
        0.0
    };

    Ok(VisionRolloutScheduleBenchReport {
        artifact: VisionArtifactHeader::new("vision_distill_rollout_schedule_bench"),
        benchmark: "burn_dragon vision distill rollout schedule benchmark",
        adapter: detect_wgpu_adapter_info(),
        backbone: vision.backbone,
        batch_size,
        image_size: vision.image_size,
        patch_tokens_per_image,
        supervision_steps: steps,
        repeated_ms,
        scheduled_ms,
        speedup,
        repeated_tokens_per_sec,
        scheduled_tokens_per_sec,
        dense_schedule_reuse_enabled: vision.backbone == VisionBackboneKind::Dense
            && schedule.len() > 1,
    })
}

fn run_repeated(
    rollout_bench: &VisionRolloutScheduleBenchAdapter<'_, VisionRolloutScheduleBenchBackend>,
    images: Tensor<VisionRolloutScheduleBenchBackend, 4>,
    teacher_patch: Tensor<VisionRolloutScheduleBenchBackend, 3>,
    teacher_cls: Tensor<VisionRolloutScheduleBenchBackend, 2>,
    schedule: &[(usize, usize)],
    loss: &burn_dragon::vision::VisionDistillationLossConfig,
) -> Tensor<VisionRolloutScheduleBenchBackend, 1> {
    rollout_bench.repeated_distill_loss(images, teacher_patch, teacher_cls, schedule, loss)
}

fn run_scheduled(
    rollout_bench: &VisionRolloutScheduleBenchAdapter<'_, VisionRolloutScheduleBenchBackend>,
    images: Tensor<VisionRolloutScheduleBenchBackend, 4>,
    teacher_patch: Tensor<VisionRolloutScheduleBenchBackend, 3>,
    teacher_cls: Tensor<VisionRolloutScheduleBenchBackend, 2>,
    schedule: &[(usize, usize)],
    loss: &burn_dragon::vision::VisionDistillationLossConfig,
) -> Tensor<VisionRolloutScheduleBenchBackend, 1> {
    rollout_bench.scheduled_distill_loss(images, teacher_patch, teacher_cls, schedule, loss)
}

fn measure_avg(iterations: usize, mut f: impl FnMut()) -> f64 {
    let iterations = iterations.max(1);
    let mut total_ns = 0f64;
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total_ns += start.elapsed().as_secs_f64() * 1_000_000_000.0;
    }
    total_ns / iterations as f64
}

fn sync_tensor(tensor: Tensor<VisionRolloutScheduleBenchBackend, 1>) {
    let _ = tensor.into_data();
}
