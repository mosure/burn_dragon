use std::fmt::Write as _;
use std::sync::mpsc;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon::vision::{
    VisionArtifactHeader, VisionAttentionMode, VisionDenseAttentionBenchAdapter,
    VisionTrainingConfig, push_vision_artifact_markdown_prelude,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;
use wgpu::util::DeviceExt;

pub type VisionDenseAttentionBenchBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
pub type VisionDenseAttentionBenchDevice =
    <VisionDenseAttentionBenchBackend as BackendTrait>::Device;

const ROW_NORM_EPS: f32 = 1e-6;
const RAW_WGSL_ROWL1_SHADER: &str = r#"
@group(0) @binding(0)
var<storage, read> query: array<f32>;

@group(0) @binding(1)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(2)
var<storage, read> params: array<f32>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_query(b: u32, h: u32, t: u32, l: u32, heads: u32, time: u32, latent: u32) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

fn idx_score(b: u32, h: u32, row: u32, col: u32, heads: u32, time: u32) -> u32 {
  return (((b * heads + h) * time + row) * time + col);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let time = to_u32(params[2]);
  let latent = to_u32(params[3]);
  let inv_scale = params[4];
  let eps = params[5];
  let row_offset = to_u32(params[6]);
  let row_count = to_u32(params[7]);

  let row_local = gid.x;
  let row = row_offset + row_local;
  let h = gid.y;
  let b = gid.z;
  if row_local >= row_count || row >= time || h >= heads || b >= batch {
    return;
  }

  let slope = params[8u + h];
  var denom = eps;
  var col = 0u;
  while col < time {
    var sum = 0.0;
    var l = 0u;
    while l < latent {
      let q_row = query[idx_query(b, h, row, l, heads, time, latent)] * inv_scale;
      let q_col = query[idx_query(b, h, col, l, heads, time, latent)];
      sum += q_row * q_col;
      l += 1u;
    }
    sum += slope * (f32(col) - f32(row));
    scores[idx_score(b, h, row_local, col, heads, time)] = sum;
    denom += abs(sum);
    col += 1u;
  }

  col = 0u;
  while col < time {
    let index = idx_score(b, h, row_local, col, heads, time);
    scores[index] = scores[index] / denom;
    col += 1u;
  }
}
"#;

#[derive(Clone, Debug)]
pub struct VisionDenseAttentionBenchConfig {
    pub batch_size: Option<usize>,
    pub warmup: usize,
    pub iterations: usize,
    pub attention_mode: Option<VisionAttentionMode>,
    pub use_alibi: Option<bool>,
}

#[derive(Clone, Serialize)]
pub struct VisionDenseAttentionBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub batch_size: usize,
    pub image_size: usize,
    pub patch_tokens_per_image: usize,
    pub sequence_len: usize,
    pub heads: usize,
    pub latent_per_head: usize,
    pub embed_dim: usize,
    pub attention_mode: VisionAttentionMode,
    pub use_alibi: bool,
    pub full_attention_ms: f64,
    pub qk_scores_ms: f64,
    pub qk_scores_direct_4d_ms: f64,
    pub alibi_and_row_norm_ms: f64,
    pub raw_direct_wgpu_rowl1_scores_ms: Option<f64>,
    pub raw_direct_wgpu_max_abs: Option<f64>,
    pub repeated_value_expand_ms: f64,
    pub repeated_value_matmul_ms: f64,
    pub repeated_value_total_ms: f64,
    pub shared_value_batchmatmul_ms: f64,
    pub shared_value_speedup_vs_repeat: f64,
    pub raw_direct_wgpu_speedup_vs_current_score_path: Option<f64>,
    pub blockwise64_ms: Option<f64>,
    pub blockwise128_ms: Option<f64>,
    pub dominant_component: String,
}

pub fn init_vision_dense_attention_bench_runtime(device: &VisionDenseAttentionBenchDevice) {
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

impl VisionDenseAttentionBenchReport {
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
        let _ = writeln!(out, "- Attention mode: {:?}", self.attention_mode);
        let _ = writeln!(out, "- Use ALiBi: {}", self.use_alibi);
        let _ = writeln!(out, "- Dominant component: {}", self.dominant_component);
        let _ = writeln!(out);
        let _ = writeln!(out, "## Timing");
        let _ = writeln!(out);
        let _ = writeln!(out, "- Full attention ms: {:.3}", self.full_attention_ms);
        let _ = writeln!(out, "- QK scores ms: {:.3}", self.qk_scores_ms);
        let _ = writeln!(
            out,
            "- QK scores direct 4D ms: {:.3}",
            self.qk_scores_direct_4d_ms
        );
        let _ = writeln!(
            out,
            "- ALiBi + row norm ms: {:.3}",
            self.alibi_and_row_norm_ms
        );
        if let Some(raw_ms) = self.raw_direct_wgpu_rowl1_scores_ms {
            let _ = writeln!(
                out,
                "- Raw direct WGPU fused row_l1 scores ms: {:.3}",
                raw_ms
            );
        }
        if let Some(speedup) = self.raw_direct_wgpu_speedup_vs_current_score_path {
            let _ = writeln!(
                out,
                "- Raw direct WGPU speedup vs current score path: {:.3}x",
                speedup
            );
        }
        if let Some(max_abs) = self.raw_direct_wgpu_max_abs {
            let _ = writeln!(
                out,
                "- Raw direct WGPU max abs vs current score path: {:.6}",
                max_abs
            );
        }
        let _ = writeln!(
            out,
            "- Repeated value expand ms: {:.3}",
            self.repeated_value_expand_ms
        );
        let _ = writeln!(
            out,
            "- Repeated value matmul ms: {:.3}",
            self.repeated_value_matmul_ms
        );
        let _ = writeln!(
            out,
            "- Repeated value total ms: {:.3}",
            self.repeated_value_total_ms
        );
        let _ = writeln!(
            out,
            "- Shared-value batchmatmul ms: {:.3}",
            self.shared_value_batchmatmul_ms
        );
        let _ = writeln!(
            out,
            "- Shared-value speedup vs repeat path: {:.3}x",
            self.shared_value_speedup_vs_repeat
        );
        if let Some(blockwise64_ms) = self.blockwise64_ms {
            let _ = writeln!(
                out,
                "- Blockwise row_l1 full attention (64) ms: {:.3}",
                blockwise64_ms
            );
        }
        if let Some(blockwise128_ms) = self.blockwise128_ms {
            let _ = writeln!(
                out,
                "- Blockwise row_l1 full attention (128) ms: {:.3}",
                blockwise128_ms
            );
        }
        out
    }
}

pub fn run_vision_dense_attention_bench(
    config: &VisionTrainingConfig,
    bench: &VisionDenseAttentionBenchConfig,
) -> Result<VisionDenseAttentionBenchReport> {
    let device = VisionDenseAttentionBenchDevice::default();
    init_vision_dense_attention_bench_runtime(&device);
    config.validate()?;
    let mut vision_config = config.vision.clone();
    if let Some(attention_mode) = bench.attention_mode {
        vision_config.attention_mode = attention_mode;
        vision_config.allow_softmax_attention =
            matches!(attention_mode, VisionAttentionMode::Softmax);
    }
    if let Some(use_alibi) = bench.use_alibi {
        vision_config.use_alibi = use_alibi;
    }
    let vision = vision_config.build();
    let batch_size = bench
        .batch_size
        .unwrap_or(config.training.batch_size)
        .max(1);
    let patch_grid = vision.image_size.div_ceil(vision.patch_size.max(1)).max(1);
    let patch_tokens_per_image = patch_grid * patch_grid;
    let sequence_len = patch_tokens_per_image + usize::from(vision.use_cls_token);
    let heads = vision.n_head.max(1);
    let latent_per_head = vision.latent_per_head().max(1);
    let embed_dim = vision.embed_dim.max(1);

    let query = Tensor::<VisionDenseAttentionBenchBackend, 4>::random(
        [batch_size, heads, sequence_len, latent_per_head],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let value = Tensor::<VisionDenseAttentionBenchBackend, 4>::random(
        [batch_size, 1, sequence_len, embed_dim],
        Distribution::Normal(0.0, 1.0),
        &device,
    );

    let slopes = if vision.use_alibi {
        Some(
            burn_dragon::core::kernel::linear_attention::default_alibi_slopes(heads)
                .into_iter()
                .map(|v| v as f32)
                .collect::<Vec<_>>(),
        )
    } else {
        None
    };
    let query_data = query
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("query vec");
    let attention_bench =
        VisionDenseAttentionBenchAdapter::new(vision.attention_mode, slopes.clone());
    let raw_runner = matches!(vision.attention_mode, VisionAttentionMode::RowL1).then(|| {
        RawDenseScoreRunner::new(
            &query_data,
            slopes.as_deref(),
            batch_size,
            heads,
            sequence_len,
            latent_per_head,
        )
    });
    for _ in 0..bench.warmup {
        sync_tensor(attention_bench.full_attention_current(query.clone(), value.clone()));
        let scores = attention_bench.qk_scores(query.clone());
        let normalized = attention_bench.apply_alibi_and_norm(scores);
        if matches!(vision.attention_mode, VisionAttentionMode::RowL1) {
            raw_runner
                .as_ref()
                .expect("raw direct wgpu runner")
                .run(false)
                .expect("raw direct wgpu dense row-l1 scores");
        }
        sync_tensor(attention_bench.repeated_value_attention(normalized.clone(), value.clone()));
        sync_tensor(attention_bench.shared_value_batchmatmul(normalized, value.clone()));
    }

    let full_attention_ms = measure_avg(bench.iterations, || {
        sync_tensor(attention_bench.full_attention_current(query.clone(), value.clone()));
    });
    let qk_scores_ms = measure_avg(bench.iterations, || {
        sync_tensor(attention_bench.qk_scores(query.clone()));
    });
    let qk_scores_direct_4d_ms = measure_avg(bench.iterations, || {
        sync_tensor(attention_bench.qk_scores_direct_4d(query.clone()));
    });

    let scores = attention_bench.qk_scores(query.clone());
    let alibi_and_row_norm_ms = measure_avg(bench.iterations, || {
        sync_tensor(attention_bench.apply_alibi_and_norm(scores.clone()));
    });
    let current_score_path = qk_scores_ms + alibi_and_row_norm_ms;
    let current_scores =
        attention_bench.apply_alibi_and_norm(attention_bench.qk_scores(query.clone()));
    let current_scores_vec = current_scores
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("current score vec");
    let (raw_direct_wgpu_rowl1_scores_ms, raw_direct_wgpu_max_abs) =
        if matches!(vision.attention_mode, VisionAttentionMode::RowL1) {
            let runner = raw_runner.as_ref().expect("raw direct wgpu runner");
            let ms = measure_avg(bench.iterations, || {
                runner.run(false).expect("raw direct wgpu dense row-l1");
            });
            let raw_scores = runner.run(true).expect("raw direct wgpu readback");
            let max_abs = raw_scores
                .iter()
                .zip(current_scores_vec.iter())
                .map(|(lhs, rhs)| (lhs - rhs).abs() as f64)
                .fold(0.0_f64, f64::max);
            (Some(ms), Some(max_abs))
        } else {
            (None, None)
        };

    let repeated_value_expand_ms = measure_avg(bench.iterations, || {
        sync_tensor(attention_bench.expand_shared_value_for_heads(value.clone(), heads));
    });

    let normalized_scores =
        attention_bench.apply_alibi_and_norm(attention_bench.qk_scores(query.clone()));
    let repeated_value_matmul_ms = measure_avg(bench.iterations, || {
        sync_tensor(
            attention_bench.repeated_value_attention(normalized_scores.clone(), value.clone()),
        );
    });
    let repeated_value_total_ms = repeated_value_expand_ms + repeated_value_matmul_ms;

    let shared_value_batchmatmul_ms = measure_avg(bench.iterations, || {
        sync_tensor(
            attention_bench.shared_value_batchmatmul(normalized_scores.clone(), value.clone()),
        );
    });
    let blockwise64_ms =
        if matches!(vision.attention_mode, VisionAttentionMode::RowL1) && sequence_len > 1 {
            Some(measure_avg(bench.iterations, || {
                sync_tensor(attention_bench.full_attention_blockwise_row_l1(
                    query.clone(),
                    value.clone(),
                    64,
                ));
            }))
        } else {
            None
        };
    let blockwise128_ms =
        if matches!(vision.attention_mode, VisionAttentionMode::RowL1) && sequence_len > 1 {
            Some(measure_avg(bench.iterations, || {
                sync_tensor(attention_bench.full_attention_blockwise_row_l1(
                    query.clone(),
                    value.clone(),
                    128,
                ));
            }))
        } else {
            None
        };

    let dominant_component = {
        let mut candidates = [
            ("qk_scores", qk_scores_ms),
            ("alibi_and_row_norm", alibi_and_row_norm_ms),
            (
                "raw_direct_wgpu_rowl1_scores",
                raw_direct_wgpu_rowl1_scores_ms.unwrap_or_default(),
            ),
            ("repeated_value_total", repeated_value_total_ms),
            ("shared_value_batchmatmul", shared_value_batchmatmul_ms),
        ];
        candidates.sort_by(|lhs, rhs| lhs.1.total_cmp(&rhs.1));
        candidates
            .last()
            .map(|(name, _)| (*name).to_string())
            .unwrap_or_else(|| "unknown".to_string())
    };

    Ok(VisionDenseAttentionBenchReport {
        artifact: VisionArtifactHeader::new("vision_dense_attention_bench"),
        benchmark: "burn_dragon dense vision attention breakdown benchmark",
        adapter: detect_wgpu_adapter_info(),
        batch_size,
        image_size: vision.image_size,
        patch_tokens_per_image,
        sequence_len,
        heads,
        latent_per_head,
        embed_dim,
        attention_mode: vision.attention_mode,
        use_alibi: vision.use_alibi,
        full_attention_ms,
        qk_scores_ms,
        qk_scores_direct_4d_ms,
        alibi_and_row_norm_ms,
        raw_direct_wgpu_rowl1_scores_ms,
        raw_direct_wgpu_max_abs,
        repeated_value_expand_ms,
        repeated_value_matmul_ms,
        repeated_value_total_ms,
        shared_value_batchmatmul_ms,
        shared_value_speedup_vs_repeat: if shared_value_batchmatmul_ms > 0.0 {
            repeated_value_total_ms / shared_value_batchmatmul_ms
        } else {
            0.0
        },
        raw_direct_wgpu_speedup_vs_current_score_path: raw_direct_wgpu_rowl1_scores_ms.map(|raw| {
            if raw > 0.0 {
                current_score_path / raw
            } else {
                0.0
            }
        }),
        blockwise64_ms,
        blockwise128_ms,
        dominant_component,
    })
}

struct RawDenseScoreRunner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    params_buffer: wgpu::Buffer,
    output_buffer: wgpu::Buffer,
    readback_buffer: wgpu::Buffer,
    dispatch_y: u32,
    dispatch_z: u32,
    batch: usize,
    heads: usize,
    time: usize,
    row_block: usize,
    params_base: Vec<f32>,
}

impl RawDenseScoreRunner {
    fn new(
        query: &[f32],
        slopes: Option<&[f32]>,
        batch: usize,
        heads: usize,
        time: usize,
        latent: usize,
    ) -> Self {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("wgpu adapter");
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default()))
                .expect("wgpu device");

        let row_block = usize::min(time, 64);
        let mut params = Vec::with_capacity(8 + heads);
        params.extend_from_slice(&[
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            1.0 / (latent as f32).sqrt().max(1.0),
            ROW_NORM_EPS,
            0.0,
            row_block as f32,
        ]);
        if let Some(slopes) = slopes {
            params.extend_from_slice(slopes);
        } else {
            params.resize(8 + heads, 0.0);
        }

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("raw_dense_rowl1_shader"),
            source: wgpu::ShaderSource::Wgsl(RAW_WGSL_ROWL1_SHADER.into()),
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("raw_dense_rowl1_bind_group_layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("raw_dense_rowl1_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("raw_dense_rowl1_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let output_len = batch * heads * row_block * time;
        let query_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raw_dense_query"),
            contents: bytemuck::cast_slice(query),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_dense_scores"),
            size: (output_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("raw_dense_params"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let readback_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("raw_dense_scores_readback"),
            size: (output_len * core::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("raw_dense_rowl1_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: query_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        Self {
            device,
            queue,
            pipeline,
            bind_group,
            params_buffer,
            output_buffer,
            readback_buffer,
            dispatch_y: heads as u32,
            dispatch_z: batch as u32,
            batch,
            heads,
            time,
            row_block,
            params_base: params,
        }
    }

    fn run(&self, readback: bool) -> Option<Vec<f32>> {
        let mut full =
            readback.then(|| vec![0.0f32; self.batch * self.heads * self.time * self.time]);
        let block_elems = self.batch * self.heads * self.row_block * self.time;
        let block_bytes = (block_elems * core::mem::size_of::<f32>()) as u64;

        for row_offset in (0..self.time).step_by(self.row_block) {
            let rows_this_block = usize::min(self.row_block, self.time - row_offset);
            let mut params = self.params_base.clone();
            params[6] = row_offset as f32;
            params[7] = rows_this_block as f32;
            self.queue
                .write_buffer(&self.params_buffer, 0, bytemuck::cast_slice(&params));

            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("raw_dense_rowl1_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("raw_dense_rowl1_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                pass.dispatch_workgroups(
                    rows_this_block.div_ceil(64) as u32,
                    self.dispatch_y,
                    self.dispatch_z,
                );
            }
            if readback {
                encoder.copy_buffer_to_buffer(
                    &self.output_buffer,
                    0,
                    &self.readback_buffer,
                    0,
                    block_bytes,
                );
            }
            self.queue.submit([encoder.finish()]);
            self.device.poll(wgpu::PollType::Wait).ok();

            if let Some(full_scores) = full.as_mut() {
                let slice = self.readback_buffer.slice(0..block_bytes);
                let (tx, rx) = mpsc::channel();
                slice.map_async(wgpu::MapMode::Read, move |res| {
                    tx.send(res).ok();
                });
                self.device.poll(wgpu::PollType::Wait).ok();
                rx.recv().ok()?.ok()?;
                let bytes = slice.get_mapped_range().to_vec();
                self.readback_buffer.unmap();
                let block = bytemuck::cast_slice::<u8, f32>(&bytes);
                for b in 0..self.batch {
                    for h in 0..self.heads {
                        for row_local in 0..rows_this_block {
                            let row = row_offset + row_local;
                            let dst = (((b * self.heads + h) * self.time + row) * self.time)
                                ..(((b * self.heads + h) * self.time + row + 1) * self.time);
                            let src = (((b * self.heads + h) * self.row_block + row_local)
                                * self.time)
                                ..((((b * self.heads + h) * self.row_block + row_local) + 1)
                                    * self.time);
                            full_scores[dst].copy_from_slice(&block[src]);
                        }
                    }
                }
            }
        }

        Some(full.unwrap_or_default())
    }
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

fn sync_tensor<const D: usize>(tensor: Tensor<VisionDenseAttentionBenchBackend, D>) {
    let _ = tensor.into_data();
}
