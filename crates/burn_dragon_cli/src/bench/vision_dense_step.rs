use std::fmt::Write as _;
use std::process::Command;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon::core::FusedAttentionExecutor;
use burn_dragon::vision::model::VisionRolloutState;
use burn_dragon::vision::{
    VisionArtifactHeader, VisionBackboneKind, VisionDenseBenchAdapter, VisionDragon,
    VisionTrainingConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

pub type VisionDenseStepBenchBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionDenseStepBenchDevice =
    <VisionDenseStepBenchBackend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VisionDenseStepBenchConfig {
    pub batch_size: Option<usize>,
    pub warmup: usize,
    pub iterations: usize,
    pub rollout_steps: Vec<usize>,
    pub gpu_index: usize,
    pub power_sample_ms: u64,
    pub power_phase_ms: u64,
}

#[derive(Clone, Serialize)]
pub struct VisionDenseStepRolloutReport {
    pub steps: usize,
    pub total_ms: f64,
    pub per_step_ms: f64,
    pub factor_vs_single_step: f64,
    pub power: Option<VisionDenseStepPowerReport>,
}

#[derive(Clone, Serialize)]
pub struct VisionDenseStepPowerReport {
    pub sample_count: usize,
    pub mean_power_w: f64,
    pub p90_power_w: f64,
    pub mean_util_pct: f64,
    pub p90_util_pct: f64,
    pub mean_memory_mb: f64,
}

#[derive(Clone, Serialize)]
pub struct VisionDenseStepBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub config: Vec<std::path::PathBuf>,
    pub batch_size: usize,
    pub image_size: usize,
    pub patch_tokens_per_image: usize,
    pub sequence_len: usize,
    pub heads: usize,
    pub latent_per_head: usize,
    pub embed_dim: usize,
    pub fused_enabled: bool,
    pub attention_executor: FusedAttentionExecutor,
    pub x_projection_ms: f64,
    pub attention_context_ms: f64,
    pub y_projection_ms: f64,
    pub tail_ms: f64,
    pub component_sum_ms: f64,
    pub full_step_ms: f64,
    pub full_step_vs_component_sum: f64,
    pub dominant_component: String,
    pub x_projection_power: Option<VisionDenseStepPowerReport>,
    pub attention_context_power: Option<VisionDenseStepPowerReport>,
    pub y_projection_power: Option<VisionDenseStepPowerReport>,
    pub tail_power: Option<VisionDenseStepPowerReport>,
    pub full_step_power: Option<VisionDenseStepPowerReport>,
    pub rollout: Vec<VisionDenseStepRolloutReport>,
}

pub fn init_vision_dense_step_bench_runtime(device: &VisionDenseStepBenchDevice) {
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

impl VisionDenseStepBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(&mut out, self.benchmark, &self.artifact);
        let _ = writeln!(out, "- Adapter: {}", self.adapter);
        let _ = writeln!(out, "- Batch size: {}", self.batch_size);
        let _ = writeln!(out, "- Image size: {}", self.image_size);
        let _ = writeln!(out, "- Patch tokens/image: {}", self.patch_tokens_per_image);
        let _ = writeln!(out, "- Sequence len: {}", self.sequence_len);
        let _ = writeln!(out, "- Heads: {}", self.heads);
        let _ = writeln!(out, "- Latent/head: {}", self.latent_per_head);
        let _ = writeln!(out, "- Embed dim: {}", self.embed_dim);
        let _ = writeln!(out, "- Fused kernels: {}", self.fused_enabled);
        let _ = writeln!(out, "- Attention executor: `{:?}`", self.attention_executor);
        let _ = writeln!(out, "- Dominant component: `{}`", self.dominant_component);
        let _ = writeln!(out);
        let _ = writeln!(out, "## Step Breakdown");
        let _ = writeln!(out);
        let _ = writeln!(out, "- x projection: `{:.3} ms`", self.x_projection_ms);
        let _ = writeln!(
            out,
            "- attention/context: `{:.3} ms`",
            self.attention_context_ms
        );
        let _ = writeln!(out, "- y projection: `{:.3} ms`", self.y_projection_ms);
        let _ = writeln!(out, "- tail: `{:.3} ms`", self.tail_ms);
        let _ = writeln!(out, "- component sum: `{:.3} ms`", self.component_sum_ms);
        let _ = writeln!(out, "- full step: `{:.3} ms`", self.full_step_ms);
        let _ = writeln!(
            out,
            "- full/component ratio: `{:.3}`",
            self.full_step_vs_component_sum
        );
        if let Some(power) = &self.x_projection_power {
            let _ = writeln!(
                out,
                "- x projection power: `{:.1} W` mean, `{:.1} W` p90, util `{:.1}%`",
                power.mean_power_w, power.p90_power_w, power.mean_util_pct
            );
        }
        if let Some(power) = &self.attention_context_power {
            let _ = writeln!(
                out,
                "- attention/context power: `{:.1} W` mean, `{:.1} W` p90, util `{:.1}%`",
                power.mean_power_w, power.p90_power_w, power.mean_util_pct
            );
        }
        if let Some(power) = &self.y_projection_power {
            let _ = writeln!(
                out,
                "- y projection power: `{:.1} W` mean, `{:.1} W` p90, util `{:.1}%`",
                power.mean_power_w, power.p90_power_w, power.mean_util_pct
            );
        }
        if let Some(power) = &self.tail_power {
            let _ = writeln!(
                out,
                "- tail power: `{:.1} W` mean, `{:.1} W` p90, util `{:.1}%`",
                power.mean_power_w, power.p90_power_w, power.mean_util_pct
            );
        }
        if let Some(power) = &self.full_step_power {
            let _ = writeln!(
                out,
                "- full step power: `{:.1} W` mean, `{:.1} W` p90, util `{:.1}%`",
                power.mean_power_w, power.p90_power_w, power.mean_util_pct
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(out, "## Rollout");
        let _ = writeln!(out);
        for rollout in &self.rollout {
            let _ = writeln!(
                out,
                "- `{} step`: total `{:.3} ms`, per-step `{:.3} ms`, factor-vs-single `{:.3}`{}",
                rollout.steps,
                rollout.total_ms,
                rollout.per_step_ms,
                rollout.factor_vs_single_step,
                rollout
                    .power
                    .as_ref()
                    .map(|power| format!(
                        ", power `{:.1} W` mean / `{:.1} W` p90, util `{:.1}%`",
                        power.mean_power_w, power.p90_power_w, power.mean_util_pct
                    ))
                    .unwrap_or_default(),
            );
        }
        out
    }
}

pub fn run_vision_dense_step_bench(
    config: &VisionTrainingConfig,
    bench: &VisionDenseStepBenchConfig,
    config_paths: &[std::path::PathBuf],
) -> Result<VisionDenseStepBenchReport> {
    let device = VisionDenseStepBenchDevice::default();
    init_vision_dense_step_bench_runtime(&device);
    config.validate()?;
    let vision_config = config.vision.build();
    assert_eq!(
        vision_config.backbone,
        VisionBackboneKind::Dense,
        "vision_dense_step_bench only supports dense backbones",
    );

    let model = VisionDragon::<VisionDenseStepBenchBackend>::new(vision_config.clone(), &device);
    let batch_size = bench.batch_size.unwrap_or(config.training.batch_size).max(1);
    let patch_grid = vision_config
        .image_size
        .div_ceil(vision_config.patch_size.max(1))
        .max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let sequence_len = patch_tokens_per_image + usize::from(vision_config.use_cls_token);
    let heads = vision_config.n_head.max(1);
    let latent_per_head = vision_config.latent_per_head().max(1);
    let embed_dim = vision_config.embed_dim.max(1);

    let images = Tensor::<VisionDenseStepBenchBackend, 4>::random(
        [
            batch_size,
            vision_config.in_channels.max(1),
            vision_config.image_size.max(1),
            vision_config.image_size.max(1),
        ],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let initial_state = model.rollout_state_from_images(images);
    let VisionRolloutState::Dense { token_state } = initial_state.clone() else {
        panic!("expected dense rollout state")
    };
    let [batch, time, dim] = token_state.shape().dims::<3>();
    let current = token_state.reshape([batch, 1, time, dim]);

    let dense = VisionDenseBenchAdapter::new(&model);

    for _ in 0..bench.warmup {
        let x_neuron = dense.x_projection(current.clone());
        let attn = dense.attention_context(x_neuron.clone(), current.clone());
        let y_gate = dense.y_projection(attn);
        sync_tensor(dense.tail(current.clone(), x_neuron, y_gate));
        sync_tensor(dense.step(current.clone()));
        for &steps in &bench.rollout_steps {
            sync_rollout_state(model.refine_rollout_state_unbounded(
                initial_state.clone(),
                steps.max(1),
                steps.max(1),
            ));
        }
    }

    let x_projection_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.x_projection(current.clone()));
    });
    let x_projection_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.x_projection(current.clone())),
    );
    let x_neuron = dense.x_projection(current.clone());
    let attention_context_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.attention_context(x_neuron.clone(), current.clone()));
    });
    let attention_context_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.attention_context(x_neuron.clone(), current.clone())),
    );
    let attn = dense.attention_context(x_neuron.clone(), current.clone());
    let y_projection_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.y_projection(attn.clone()));
    });
    let y_projection_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.y_projection(attn.clone())),
    );
    let y_gate = dense.y_projection(attn);
    let tail_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.tail(current.clone(), x_neuron.clone(), y_gate.clone()));
    });
    let tail_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.tail(current.clone(), x_neuron.clone(), y_gate.clone())),
    );
    let full_step_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.step(current.clone()));
    });
    let full_step_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.step(current.clone())),
    );
    let component_sum_ms = x_projection_ms + attention_context_ms + y_projection_ms + tail_ms;
    let full_step_vs_component_sum = if component_sum_ms > 0.0 {
        full_step_ms / component_sum_ms
    } else {
        0.0
    };
    let dominant_component = [
        ("x_projection", x_projection_ms),
        ("attention_context", attention_context_ms),
        ("y_projection", y_projection_ms),
        ("tail", tail_ms),
    ]
    .into_iter()
    .max_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1))
    .map(|(name, _)| name.to_string())
    .unwrap_or_else(|| "unknown".to_string());

    let rollout = bench
        .rollout_steps
        .iter()
        .copied()
        .map(|steps| {
            let total_ms = measure_avg(bench.iterations, || {
                sync_rollout_state(model.refine_rollout_state_unbounded(
                    initial_state.clone(),
                    steps.max(1),
                    steps.max(1),
                ));
            });
            let power = measure_power_phase(
                bench.power_phase_ms,
                bench.power_sample_ms,
                bench.gpu_index,
                || {
                    sync_rollout_state(model.refine_rollout_state_unbounded(
                        initial_state.clone(),
                        steps.max(1),
                        steps.max(1),
                    ))
                },
            );
            let per_step_ms = total_ms / steps.max(1) as f64;
            let factor_vs_single_step = if full_step_ms > 0.0 {
                per_step_ms / full_step_ms
            } else {
                0.0
            };
            VisionDenseStepRolloutReport {
                steps,
                total_ms,
                per_step_ms,
                factor_vs_single_step,
                power,
            }
        })
        .collect::<Vec<_>>();

    Ok(VisionDenseStepBenchReport {
        artifact: VisionArtifactHeader::new("vision_dense_step_bench"),
        benchmark: "burn_dragon dense vision recurrent step breakdown benchmark",
        adapter: detect_wgpu_adapter_info(),
        config: config_paths.to_vec(),
        batch_size,
        image_size: vision_config.image_size,
        patch_tokens_per_image,
        sequence_len,
        heads,
        latent_per_head,
        embed_dim,
        fused_enabled: vision_config.fused_kernels.enabled,
        attention_executor: vision_config.fused_kernels.attention_executor,
        x_projection_ms,
        attention_context_ms,
        y_projection_ms,
        tail_ms,
        component_sum_ms,
        full_step_ms,
        full_step_vs_component_sum,
        dominant_component,
        x_projection_power,
        attention_context_power,
        y_projection_power,
        tail_power,
        full_step_power,
        rollout,
    })
}

fn measure_avg(iterations: usize, mut f: impl FnMut()) -> f64 {
    let iterations = iterations.max(1);
    let mut total_ms = 0f64;
    for _ in 0..iterations {
        let start = Instant::now();
        f();
        total_ms += start.elapsed().as_secs_f64() * 1_000.0;
    }
    total_ms / iterations as f64
}

fn sync_tensor<const D: usize>(tensor: Tensor<VisionDenseStepBenchBackend, D>) {
    let _ = tensor.into_data();
}

fn sync_rollout_state(state: VisionRolloutState<VisionDenseStepBenchBackend>) {
    match state {
        VisionRolloutState::Dense { token_state } => sync_tensor(token_state),
        VisionRolloutState::Pyramid(_) | VisionRolloutState::Cellular(_) => {
            panic!("vision_dense_step_bench only supports dense rollout states")
        }
    }
}

fn measure_power_phase(
    phase_ms: u64,
    sample_ms: u64,
    gpu_index: usize,
    mut f: impl FnMut(),
) -> Option<VisionDenseStepPowerReport> {
    if phase_ms == 0 || sample_ms == 0 {
        return None;
    }
    let stop = Arc::new(AtomicBool::new(false));
    let samples = Arc::new(Mutex::new(Vec::<(f64, f64, f64)>::new()));
    let stop_reader = Arc::clone(&stop);
    let samples_reader = Arc::clone(&samples);
    let handle = thread::spawn(move || {
        while !stop_reader.load(Ordering::Relaxed) {
            if let Some(sample) = sample_gpu_once(gpu_index)
                && let Ok(mut locked) = samples_reader.lock()
            {
                locked.push(sample);
            }
            thread::sleep(Duration::from_millis(sample_ms.max(1)));
        }
    });

    let deadline = Instant::now() + Duration::from_millis(phase_ms);
    while Instant::now() < deadline {
        f();
    }
    stop.store(true, Ordering::Relaxed);
    let _ = handle.join();
    let Ok(samples) = samples.lock() else {
        return None;
    };
    summarize_power(&samples)
}

fn sample_gpu_once(gpu_index: usize) -> Option<(f64, f64, f64)> {
    let output = Command::new("nvidia-smi")
        .args([
            "--id",
            &gpu_index.to_string(),
            "--query-gpu=power.draw.instant,utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let mut parts = line.trim().split(',').map(str::trim);
    let power = parts.next()?.parse::<f64>().ok()?;
    let util = parts.next()?.parse::<f64>().ok()?;
    let memory = parts.next()?.parse::<f64>().ok()?;
    Some((power, util, memory))
}

fn summarize_power(samples: &[(f64, f64, f64)]) -> Option<VisionDenseStepPowerReport> {
    if samples.is_empty() {
        return None;
    }
    let mut power = samples.iter().map(|sample| sample.0).collect::<Vec<_>>();
    let mut util = samples.iter().map(|sample| sample.1).collect::<Vec<_>>();
    let mean_memory_mb = samples.iter().map(|sample| sample.2).sum::<f64>() / samples.len() as f64;
    power.sort_by(f64::total_cmp);
    util.sort_by(f64::total_cmp);
    let mean_power_w = power.iter().sum::<f64>() / power.len() as f64;
    let mean_util_pct = util.iter().sum::<f64>() / util.len() as f64;
    let p90_idx = (((power.len() as f64) * 0.9).floor() as usize).min(power.len() - 1);
    Some(VisionDenseStepPowerReport {
        sample_count: samples.len(),
        mean_power_w,
        p90_power_w: power[p90_idx],
        mean_util_pct,
        p90_util_pct: util[p90_idx],
        mean_memory_mb,
    })
}
