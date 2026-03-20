use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_dragon::vision::api::model::VisionRolloutState;
use burn_dragon::vision::{
    VisionArtifactHeader, VisionBackboneKind, VisionDenseBenchAdapter, VisionDragon,
    VisionTrainingConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, Wgpu, WgpuDevice, WgpuRuntime, graphics};
use serde::Serialize;

pub type VisionLowrankProjectionNoFusionBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionLowrankProjectionFusionBackend = Wgpu<f32>;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum VisionLowrankProjectionBenchBackendKind {
    Wgpu,
    WgpuNoFusion,
}

impl VisionLowrankProjectionBenchBackendKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Wgpu => "wgpu",
            Self::WgpuNoFusion => "wgpu-no-fusion",
        }
    }
}

/// Bench configuration for direct low-rank x/y projection measurement.
#[derive(Clone, Debug)]
pub struct VisionLowrankProjectionBenchConfig {
    pub config_paths: Vec<PathBuf>,
    pub backend: VisionLowrankProjectionBenchBackendKind,
    pub batch_size: Option<usize>,
    pub warmup: usize,
    pub iterations: usize,
    pub gpu_index: usize,
    pub power_sample_ms: u64,
    pub power_phase_ms: u64,
}

/// GPU power summary for a single measured benchmark phase.
#[derive(Clone, Debug, Serialize)]
pub struct VisionLowrankProjectionPhasePower {
    pub mean_w: f64,
    pub p90_w: f64,
    pub mean_util_pct: f64,
}

/// Forward reference-vs-fused comparison for a single projection path.
#[derive(Clone, Debug, Serialize)]
pub struct VisionLowrankProjectionPairReport {
    pub reference_ms: f64,
    pub fused_ms: f64,
    pub speedup_vs_reference: f64,
    pub reference_power: VisionLowrankProjectionPhasePower,
    pub fused_power: VisionLowrankProjectionPhasePower,
}

/// Current backward cost for the reference training path.
#[derive(Clone, Debug, Serialize)]
pub struct VisionLowrankProjectionBackwardPhaseReport {
    pub total_ms: f64,
    pub input_only_ms: f64,
    pub weight_only_ms: f64,
    pub total_power: VisionLowrankProjectionPhasePower,
    pub input_only_power: VisionLowrankProjectionPhasePower,
    pub weight_only_power: VisionLowrankProjectionPhasePower,
}

/// Current backward cost for the training path.
#[derive(Clone, Debug, Serialize)]
pub struct VisionLowrankProjectionBackwardReport {
    pub x: VisionLowrankProjectionBackwardPhaseReport,
    pub y: VisionLowrankProjectionBackwardPhaseReport,
}

/// Full report for the dedicated low-rank projection benchmark.
#[derive(Clone, Debug, Serialize)]
pub struct VisionLowrankProjectionBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub backend: String,
    pub config: Vec<PathBuf>,
    pub batch_size: usize,
    pub image_size: usize,
    pub sequence_len: usize,
    pub heads: usize,
    pub latent_per_head: usize,
    pub embed_dim: usize,
    pub x_projection: VisionLowrankProjectionPairReport,
    pub y_projection: VisionLowrankProjectionPairReport,
    pub y_path: VisionLowrankProjectionPairReport,
    pub backward: VisionLowrankProjectionBackwardReport,
    pub y_path_backward: VisionLowrankProjectionBackwardPhaseReport,
}

pub fn init_vision_lowrank_projection_bench_runtime(device: &WgpuDevice) {
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

impl VisionLowrankProjectionBenchReport {
    pub fn to_markdown(&self) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Vision Lowrank Projection Bench",
            &self.artifact,
        );
        let _ = writeln!(
            out,
            "- adapter: {}\n- backend: {}\n- batch size: {}\n- image size: {}\n- sequence len: {}\n- heads: {}\n- latent/head: {}\n- embed dim: {}\n- config: {}\n",
            self.adapter,
            self.backend,
            self.batch_size,
            self.image_size,
            self.sequence_len,
            self.heads,
            self.latent_per_head,
            self.embed_dim,
            self.config
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
        let _ = writeln!(out, "## Forward");
        let _ = writeln!(
            out,
            "- x projection: ref `{:0.3} ms`, fused `{:0.3} ms`, speedup `{:0.3}x`, power ref `{:0.1} W`, fused `{:0.1} W`",
            self.x_projection.reference_ms,
            self.x_projection.fused_ms,
            self.x_projection.speedup_vs_reference,
            self.x_projection.reference_power.mean_w,
            self.x_projection.fused_power.mean_w,
        );
        let _ = writeln!(
            out,
            "- y projection: ref `{:0.3} ms`, fused `{:0.3} ms`, speedup `{:0.3}x`, power ref `{:0.1} W`, fused `{:0.1} W`\n",
            self.y_projection.reference_ms,
            self.y_projection.fused_ms,
            self.y_projection.speedup_vs_reference,
            self.y_projection.reference_power.mean_w,
            self.y_projection.fused_power.mean_w,
        );
        let _ = writeln!(
            out,
            "- y path (projection + tail): ref `{:0.3} ms`, fused `{:0.3} ms`, speedup `{:0.3}x`, power ref `{:0.1} W`, fused `{:0.1} W`\n",
            self.y_path.reference_ms,
            self.y_path.fused_ms,
            self.y_path.speedup_vs_reference,
            self.y_path.reference_power.mean_w,
            self.y_path.fused_power.mean_w,
        );
        let _ = writeln!(out, "## Backward");
        let _ = writeln!(
            out,
            "- x backward total: `{:0.3} ms`, input-only `{:0.3} ms`, weight-only `{:0.3} ms`, power total `{:0.1} W`",
            self.backward.x.total_ms,
            self.backward.x.input_only_ms,
            self.backward.x.weight_only_ms,
            self.backward.x.total_power.mean_w,
        );
        let _ = writeln!(
            out,
            "- y backward total: `{:0.3} ms`, input-only `{:0.3} ms`, weight-only `{:0.3} ms`, power total `{:0.1} W`",
            self.backward.y.total_ms,
            self.backward.y.input_only_ms,
            self.backward.y.weight_only_ms,
            self.backward.y.total_power.mean_w,
        );
        let _ = writeln!(
            out,
            "- y path backward total: `{:0.3} ms`, input-only `{:0.3} ms`, weight-only `{:0.3} ms`, power total `{:0.1} W`",
            self.y_path_backward.total_ms,
            self.y_path_backward.input_only_ms,
            self.y_path_backward.weight_only_ms,
            self.y_path_backward.total_power.mean_w,
        );
        out
    }
}

/// Runs the direct low-rank projection benchmark on the configured dense vision model.
pub fn run_vision_lowrank_projection_bench(
    config: &VisionTrainingConfig,
    bench: &VisionLowrankProjectionBenchConfig,
) -> Result<VisionLowrankProjectionBenchReport> {
    match bench.backend {
        VisionLowrankProjectionBenchBackendKind::Wgpu => {
            run_vision_lowrank_projection_bench_backend::<VisionLowrankProjectionFusionBackend>(
                config, bench,
            )
        }
        VisionLowrankProjectionBenchBackendKind::WgpuNoFusion => {
            run_vision_lowrank_projection_bench_backend::<VisionLowrankProjectionNoFusionBackend>(
                config, bench,
            )
        }
    }
}

fn run_vision_lowrank_projection_bench_backend<BenchBackend>(
    config: &VisionTrainingConfig,
    bench: &VisionLowrankProjectionBenchConfig,
) -> Result<VisionLowrankProjectionBenchReport>
where
    BenchBackend: BackendTrait<Device = WgpuDevice>,
    BenchBackend::FloatTensorPrimitive: 'static,
    Autodiff<BenchBackend>: AutodiffBackend<Device = WgpuDevice>,
    <Autodiff<BenchBackend> as BackendTrait>::FloatTensorPrimitive: 'static,
{
    config.validate()?;
    let vision_config = config.vision.build();
    assert_eq!(
        vision_config.backbone,
        VisionBackboneKind::Dense,
        "vision_lowrank_projection_bench only supports dense backbones",
    );

    let device = <BenchBackend as BackendTrait>::Device::default();
    init_vision_lowrank_projection_bench_runtime(&device);
    let model = VisionDragon::<BenchBackend>::new(vision_config.clone(), &device);
    let batch_size = bench
        .batch_size
        .unwrap_or(config.training.batch_size)
        .max(1);
    let patch_grid = vision_config
        .image_size
        .div_ceil(vision_config.patch_size.max(1))
        .max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let sequence_len = patch_tokens_per_image + usize::from(vision_config.use_cls_token);
    let heads = vision_config.n_head.max(1);
    let latent_per_head = vision_config.latent_per_head().max(1);
    let embed_dim = vision_config.embed_dim.max(1);

    let images = Tensor::<BenchBackend, 4>::random(
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
    let VisionRolloutState::Dense { token_state } = initial_state else {
        panic!("expected dense rollout state");
    };
    let [batch, time, dim] = token_state.shape().dims::<3>();
    let current = token_state.reshape([batch, 1, time, dim]);
    let dense = VisionDenseBenchAdapter::new(&model);

    let x_reference = dense.x_projection_reference(current.clone());
    let attn_reference = dense.attention_context(x_reference.clone(), current.clone());

    for _ in 0..bench.warmup {
        sync_tensor(dense.x_projection_reference(current.clone()));
        sync_tensor(dense.x_projection(current.clone()));
        sync_tensor(dense.y_projection_reference(attn_reference.clone()));
        sync_tensor(dense.y_projection(attn_reference.clone()));
        sync_tensor(dense.y_path_reference_from_attention(
            current.clone(),
            x_reference.clone(),
            attn_reference.clone(),
        ));
        sync_tensor(dense.y_path_from_attention(
            current.clone(),
            x_reference.clone(),
            attn_reference.clone(),
        ));
    }

    let x_reference_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.x_projection_reference(current.clone()));
    });
    let x_reference_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.x_projection_reference(current.clone())),
    );
    let x_fused_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.x_projection(current.clone()));
    });
    let x_fused_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.x_projection(current.clone())),
    );

    let y_reference_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.y_projection_reference(attn_reference.clone()));
    });
    let y_reference_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.y_projection_reference(attn_reference.clone())),
    );
    let y_fused_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.y_projection(attn_reference.clone()));
    });
    let y_fused_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || sync_tensor(dense.y_projection(attn_reference.clone())),
    );

    let y_path_reference_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.y_path_reference_from_attention(
            current.clone(),
            x_reference.clone(),
            attn_reference.clone(),
        ));
    });
    let y_path_reference_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(dense.y_path_reference_from_attention(
                current.clone(),
                x_reference.clone(),
                attn_reference.clone(),
            ))
        },
    );
    let y_path_fused_ms = measure_avg(bench.iterations, || {
        sync_tensor(dense.y_path_from_attention(
            current.clone(),
            x_reference.clone(),
            attn_reference.clone(),
        ));
    });
    let y_path_fused_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(dense.y_path_from_attention(
                current.clone(),
                x_reference.clone(),
                attn_reference.clone(),
            ))
        },
    );

    let train_device = <Autodiff<BenchBackend> as BackendTrait>::Device::default();
    init_vision_lowrank_projection_bench_runtime(&train_device);
    let train_model =
        VisionDragon::<Autodiff<BenchBackend>>::new(vision_config.clone(), &train_device);
    let train_images = Tensor::<Autodiff<BenchBackend>, 4>::random(
        [
            batch_size,
            vision_config.in_channels.max(1),
            vision_config.image_size.max(1),
            vision_config.image_size.max(1),
        ],
        Distribution::Normal(0.0, 1.0),
        &train_device,
    );
    let train_initial_state = train_model.rollout_state_from_images(train_images);
    let VisionRolloutState::Dense {
        token_state: train_token_state,
    } = train_initial_state
    else {
        panic!("expected dense rollout state");
    };
    let [train_batch, train_time, train_dim] = train_token_state.shape().dims::<3>();
    let current_train_base = train_token_state
        .reshape([train_batch, 1, train_time, train_dim])
        .detach();
    let dense_train = VisionDenseBenchAdapter::new(&train_model);
    let x_train_base = dense_train
        .x_projection_reference(current_train_base.clone())
        .detach();
    let attn_train_base = dense_train
        .attention_context(x_train_base.clone(), current_train_base.clone())
        .detach();

    for _ in 0..bench.warmup {
        sync_tensor(backward_x_projection_total(
            &dense_train,
            current_train_base.clone(),
        ));
        sync_tensor(backward_x_projection_input_only(
            &dense_train,
            current_train_base.clone(),
        ));
        sync_tensor(backward_x_projection_weight_only(
            &dense_train,
            current_train_base.clone(),
        ));
        sync_tensor(backward_y_projection_total(
            &dense_train,
            attn_train_base.clone(),
        ));
        sync_tensor(backward_y_projection_input_only(
            &dense_train,
            attn_train_base.clone(),
        ));
        sync_tensor(backward_y_projection_weight_only(
            &dense_train,
            attn_train_base.clone(),
        ));
        sync_tensor(backward_y_path_total(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
        sync_tensor(backward_y_path_input_only(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
        sync_tensor(backward_y_path_weight_only(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
    }

    let x_backward_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_x_projection_total(
            &dense_train,
            current_train_base.clone(),
        ));
    });
    let x_backward_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_x_projection_total(
                &dense_train,
                current_train_base.clone(),
            ))
        },
    );
    let x_backward_input_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_x_projection_input_only(
            &dense_train,
            current_train_base.clone(),
        ));
    });
    let x_backward_input_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_x_projection_input_only(
                &dense_train,
                current_train_base.clone(),
            ))
        },
    );
    let x_backward_weight_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_x_projection_weight_only(
            &dense_train,
            current_train_base.clone(),
        ));
    });
    let x_backward_weight_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_x_projection_weight_only(
                &dense_train,
                current_train_base.clone(),
            ))
        },
    );

    let y_backward_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_projection_total(
            &dense_train,
            attn_train_base.clone(),
        ));
    });
    let y_backward_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_projection_total(
                &dense_train,
                attn_train_base.clone(),
            ))
        },
    );
    let y_backward_input_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_projection_input_only(
            &dense_train,
            attn_train_base.clone(),
        ));
    });
    let y_backward_input_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_projection_input_only(
                &dense_train,
                attn_train_base.clone(),
            ))
        },
    );
    let y_backward_weight_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_projection_weight_only(
            &dense_train,
            attn_train_base.clone(),
        ));
    });
    let y_backward_weight_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_projection_weight_only(
                &dense_train,
                attn_train_base.clone(),
            ))
        },
    );

    let y_path_backward_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_path_total(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
    });
    let y_path_backward_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_path_total(
                &dense_train,
                current_train_base.clone(),
                x_train_base.clone(),
                attn_train_base.clone(),
            ))
        },
    );
    let y_path_backward_input_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_path_input_only(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
    });
    let y_path_backward_input_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_path_input_only(
                &dense_train,
                current_train_base.clone(),
                x_train_base.clone(),
                attn_train_base.clone(),
            ))
        },
    );
    let y_path_backward_weight_only_ms = measure_avg(bench.iterations, || {
        sync_tensor(backward_y_path_weight_only(
            &dense_train,
            current_train_base.clone(),
            x_train_base.clone(),
            attn_train_base.clone(),
        ));
    });
    let y_path_backward_weight_only_power = measure_power_phase(
        bench.power_phase_ms,
        bench.power_sample_ms,
        bench.gpu_index,
        || {
            sync_tensor(backward_y_path_weight_only(
                &dense_train,
                current_train_base.clone(),
                x_train_base.clone(),
                attn_train_base.clone(),
            ))
        },
    );

    Ok(VisionLowrankProjectionBenchReport {
        artifact: VisionArtifactHeader::new("vision_lowrank_projection_bench"),
        benchmark: "burn_dragon vision lowrank projection bench",
        adapter: detect_wgpu_adapter_info(),
        backend: bench.backend.as_str().to_string(),
        config: bench.config_paths.clone(),
        batch_size,
        image_size: vision_config.image_size,
        sequence_len,
        heads,
        latent_per_head,
        embed_dim,
        x_projection: VisionLowrankProjectionPairReport {
            reference_ms: x_reference_ms,
            fused_ms: x_fused_ms,
            speedup_vs_reference: safe_ratio(x_reference_ms, x_fused_ms),
            reference_power: x_reference_power,
            fused_power: x_fused_power,
        },
        y_projection: VisionLowrankProjectionPairReport {
            reference_ms: y_reference_ms,
            fused_ms: y_fused_ms,
            speedup_vs_reference: safe_ratio(y_reference_ms, y_fused_ms),
            reference_power: y_reference_power,
            fused_power: y_fused_power,
        },
        y_path: VisionLowrankProjectionPairReport {
            reference_ms: y_path_reference_ms,
            fused_ms: y_path_fused_ms,
            speedup_vs_reference: safe_ratio(y_path_reference_ms, y_path_fused_ms),
            reference_power: y_path_reference_power,
            fused_power: y_path_fused_power,
        },
        backward: VisionLowrankProjectionBackwardReport {
            x: VisionLowrankProjectionBackwardPhaseReport {
                total_ms: x_backward_ms,
                input_only_ms: x_backward_input_only_ms,
                weight_only_ms: x_backward_weight_only_ms,
                total_power: x_backward_power,
                input_only_power: x_backward_input_only_power,
                weight_only_power: x_backward_weight_only_power,
            },
            y: VisionLowrankProjectionBackwardPhaseReport {
                total_ms: y_backward_ms,
                input_only_ms: y_backward_input_only_ms,
                weight_only_ms: y_backward_weight_only_ms,
                total_power: y_backward_power,
                input_only_power: y_backward_input_only_power,
                weight_only_power: y_backward_weight_only_power,
            },
        },
        y_path_backward: VisionLowrankProjectionBackwardPhaseReport {
            total_ms: y_path_backward_ms,
            input_only_ms: y_path_backward_input_only_ms,
            weight_only_ms: y_path_backward_weight_only_ms,
            total_power: y_path_backward_power,
            input_only_power: y_path_backward_input_only_power,
            weight_only_power: y_path_backward_weight_only_power,
        },
    })
}

fn backward_x_projection_total<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let current = current.require_grad();
    let loss = dense.x_projection(current.clone()).sum();
    let grads = loss.backward();
    current.grad(&grads).expect("x input grad")
}

fn backward_x_projection_input_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let current = current.require_grad();
    let encoder = dense.x_projection_encoder().detach();
    let loss = dense
        .x_projection_with_encoder(current.clone(), encoder)
        .sum();
    let grads = loss.backward();
    current.grad(&grads).expect("x input-only grad")
}

fn backward_x_projection_weight_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let encoder = dense.x_projection_encoder().detach().require_grad();
    let loss = dense
        .x_projection_with_encoder(current.detach(), encoder.clone())
        .sum();
    let grads = loss.backward();
    encoder.grad(&grads).expect("x weight-only grad")
}

fn backward_y_projection_total<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let attn = attn.require_grad();
    let loss = dense.y_projection(attn.clone()).sum();
    let grads = loss.backward();
    attn.grad(&grads).expect("y input grad")
}

fn backward_y_projection_input_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let attn = attn.require_grad();
    let encoder_v = dense.y_projection_encoder().detach();
    let loss = dense
        .y_projection_with_encoder(attn.clone(), encoder_v)
        .sum();
    let grads = loss.backward();
    attn.grad(&grads).expect("y input-only grad")
}

fn backward_y_projection_weight_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4> {
    let encoder_v = dense.y_projection_encoder().detach().require_grad();
    let loss = dense
        .y_projection_with_encoder(attn.detach(), encoder_v.clone())
        .sum();
    let grads = loss.backward();
    encoder_v.grad(&grads).expect("y weight-only grad")
}

fn backward_y_path_total<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
    x_neuron: Tensor<B, 4>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4>
where
    B::FloatTensorPrimitive: 'static,
{
    let attn = attn.require_grad();
    let encoder_v = dense.y_projection_encoder().detach().require_grad();
    let loss = dense
        .y_path_with_encoder_from_attention(
            current.detach(),
            x_neuron.detach(),
            attn.clone(),
            encoder_v,
        )
        .sum();
    let grads = loss.backward();
    attn.grad(&grads).expect("y path input grad")
}

fn backward_y_path_input_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
    x_neuron: Tensor<B, 4>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4>
where
    B::FloatTensorPrimitive: 'static,
{
    let attn = attn.require_grad();
    let encoder_v = dense.y_projection_encoder().detach();
    let loss = dense
        .y_path_with_encoder_from_attention(
            current.detach(),
            x_neuron.detach(),
            attn.clone(),
            encoder_v,
        )
        .sum();
    let grads = loss.backward();
    attn.grad(&grads).expect("y path input-only grad")
}

fn backward_y_path_weight_only<B: AutodiffBackend>(
    dense: &VisionDenseBenchAdapter<'_, B>,
    current: Tensor<B, 4>,
    x_neuron: Tensor<B, 4>,
    attn: Tensor<B, 4>,
) -> Tensor<B::InnerBackend, 4>
where
    B::FloatTensorPrimitive: 'static,
{
    let encoder_v = dense.y_projection_encoder().detach().require_grad();
    let loss = dense
        .y_path_with_encoder_from_attention(
            current.detach(),
            x_neuron.detach(),
            attn.detach(),
            encoder_v.clone(),
        )
        .sum();
    let grads = loss.backward();
    encoder_v.grad(&grads).expect("y path weight-only grad")
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

fn sync_tensor<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) {
    let _ = tensor.into_data();
}

fn safe_ratio(numer: f64, denom: f64) -> f64 {
    if denom > 0.0 { numer / denom } else { 0.0 }
}

#[derive(Clone, Copy)]
struct GpuSample {
    power_w: f64,
    util_pct: f64,
}

fn measure_power_phase(
    phase_ms: u64,
    sample_ms: u64,
    gpu_index: usize,
    mut f: impl FnMut(),
) -> VisionLowrankProjectionPhasePower {
    let sample_ms = sample_ms.max(50);
    let phase_ms = phase_ms.max(sample_ms);
    let poll_period = Duration::from_millis(sample_ms);
    let deadline = Instant::now() + Duration::from_millis(phase_ms);
    let mut samples = Vec::new();
    while Instant::now() < deadline {
        let phase_end = Instant::now() + poll_period;
        while Instant::now() < phase_end {
            f();
        }
        if let Some(sample) = query_gpu_sample(gpu_index) {
            samples.push(sample);
        }
    }
    if samples.is_empty() {
        return VisionLowrankProjectionPhasePower {
            mean_w: 0.0,
            p90_w: 0.0,
            mean_util_pct: 0.0,
        };
    }
    let mut power = samples
        .iter()
        .map(|sample| sample.power_w)
        .collect::<Vec<_>>();
    power.sort_by(|lhs, rhs| lhs.total_cmp(rhs));
    let p90_index = ((power.len() - 1) as f64 * 0.9).round() as usize;
    VisionLowrankProjectionPhasePower {
        mean_w: power.iter().sum::<f64>() / power.len() as f64,
        p90_w: power[p90_index],
        mean_util_pct: samples.iter().map(|sample| sample.util_pct).sum::<f64>()
            / samples.len() as f64,
    }
}

fn query_gpu_sample(gpu_index: usize) -> Option<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,utilization.gpu",
            "--format=csv,noheader,nounits",
            "-i",
            &gpu_index.to_string(),
        ])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let line = String::from_utf8(output.stdout).ok()?;
    let mut parts = line.trim().split(',');
    let power_w = parts.next()?.trim().parse::<f64>().ok()?;
    let util_pct = parts.next()?.trim().parse::<f64>().ok()?;
    thread::sleep(Duration::from_millis(10));
    Some(GpuSample { power_w, util_pct })
}
