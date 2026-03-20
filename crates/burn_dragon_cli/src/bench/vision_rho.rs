use std::fmt::Write as _;
use std::time::Instant;

use anyhow::Result;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor, TensorData};
use burn_dragon::vision::{VisionArtifactHeader, push_vision_artifact_markdown_prelude};
use burn_dragon_kernel::api::spatial::{
    LocalGridNeighborhood, LocalGridShape2d, local_grid_rho_profile_reset,
    local_grid_rho_profile_snapshot, try_fused_local_grid_rho_attention_wgpu_head_decay,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone, Debug)]
pub struct VisionRhoBenchConfig {
    pub warmup: usize,
    pub repetitions: usize,
    pub steps: usize,
}

#[derive(Clone, Copy, Serialize)]
pub struct VisionRhoBenchCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    height: usize,
    width: usize,
    latent: usize,
    embd: usize,
    radius: usize,
    diagonals: bool,
    self_edges: bool,
}

#[derive(Clone, Copy, Serialize)]
pub struct VisionRhoBenchErrorMetrics {
    max_abs: f32,
    mean_abs: f32,
}

#[derive(Clone, Serialize)]
pub struct VisionRhoBenchMeasurementResult {
    mode: &'static str,
    reference_time_ms: f64,
    fused_time_ms: f64,
    speedup_x: f64,
    throughput_per_sec: f64,
    fused_kernel_calls: u64,
    fused_kernel_launches: u64,
    fused_kernel_dispatch_ms: f64,
    fused_transient_allocations: u64,
    fused_metadata_upload_bytes: u64,
}

#[derive(Clone, Serialize)]
pub struct VisionRhoBenchCaseResult {
    pub case: VisionRhoBenchCase,
    pub problem_shape: String,
    pub forward: VisionRhoBenchMeasurementResult,
    pub resident_rollout: VisionRhoBenchMeasurementResult,
    pub context_error: VisionRhoBenchErrorMetrics,
    pub rho_error: VisionRhoBenchErrorMetrics,
}

#[derive(Clone, Serialize)]
pub struct VisionRhoBenchReport {
    pub artifact: VisionArtifactHeader,
    pub benchmark: &'static str,
    pub adapter: String,
    pub warmup: usize,
    pub repetitions: usize,
    pub steps: usize,
    pub cases: Vec<VisionRhoBenchCaseResult>,
}

const CASES: &[VisionRhoBenchCase] = &[
    VisionRhoBenchCase {
        name: "small",
        batch: 2,
        heads: 4,
        height: 8,
        width: 8,
        latent: 16,
        embd: 32,
        radius: 1,
        diagonals: false,
        self_edges: true,
    },
    VisionRhoBenchCase {
        name: "medium",
        batch: 2,
        heads: 8,
        height: 16,
        width: 16,
        latent: 16,
        embd: 64,
        radius: 1,
        diagonals: true,
        self_edges: true,
    },
    VisionRhoBenchCase {
        name: "large",
        batch: 1,
        heads: 8,
        height: 24,
        width: 24,
        latent: 24,
        embd: 96,
        radius: 1,
        diagonals: true,
        self_edges: true,
    },
];

pub fn init_vision_rho_bench_runtime(device: &Device) {
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

impl VisionRhoBenchReport {
    pub fn to_markdown(&self) -> String {
        format_markdown(self)
    }
}

pub fn run_vision_rho_bench(config: &VisionRhoBenchConfig) -> Result<VisionRhoBenchReport> {
    let device = Device::default();
    init_vision_rho_bench_runtime(&device);

    Ok(VisionRhoBenchReport {
        artifact: VisionArtifactHeader::new("vision_rho_bench"),
        benchmark: "burn_dragon vision local-grid rho benchmark",
        adapter: detect_wgpu_adapter_info(),
        warmup: config.warmup,
        repetitions: config.repetitions,
        steps: config.steps.max(1),
        cases: CASES
            .iter()
            .copied()
            .map(|case| run_case(case, &device, config))
            .collect(),
    })
}

fn run_case(
    case: VisionRhoBenchCase,
    device: &Device,
    config: &VisionRhoBenchConfig,
) -> VisionRhoBenchCaseResult {
    let tokens = case.height * case.width;
    let query = Tensor::<Backend, 4>::random(
        [case.batch, case.heads, tokens, case.latent],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = Tensor::<Backend, 4>::random(
        [case.batch, 1, tokens, case.embd],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let initial_rho = Tensor::<Backend, 5>::zeros(
        [case.batch, case.heads, tokens, case.latent, case.embd],
        device,
    );
    let decay_values = (0..case.heads)
        .map(|idx| 0.85_f32 + 0.1_f32 * (idx as f32 / case.heads.max(1) as f32))
        .collect::<Vec<_>>();
    let decay =
        Tensor::<Backend, 1>::from_data(TensorData::new(decay_values, [case.heads]), device);
    let neighborhood = if case.diagonals {
        LocalGridNeighborhood::moore(case.radius)
    } else {
        LocalGridNeighborhood::von_neumann(case.radius)
    }
    .with_self_edges(case.self_edges);
    let grid = LocalGridShape2d::new(case.height, case.width);

    let (reference_context, reference_rho) = reference_local_grid_rho_head_decay(
        query.clone(),
        value.clone(),
        initial_rho.clone(),
        grid,
        neighborhood,
        decay.clone(),
    );
    let fused = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
        &query,
        &value,
        Some(&initial_rho),
        grid,
        neighborhood,
        &decay,
    )
    .expect("fused local-grid output");

    for _ in 0..config.warmup {
        sync_tensor(
            try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
                &query,
                &value,
                Some(&initial_rho),
                grid,
                neighborhood,
                &decay,
            )
            .expect("fused warmup")
            .context,
        );
    }

    let forward_reference_ns = measure_avg(config.repetitions, || {
        let (context, _) = reference_local_grid_rho_head_decay(
            query.clone(),
            value.clone(),
            initial_rho.clone(),
            grid,
            neighborhood,
            decay.clone(),
        );
        sync_tensor(context);
    });

    local_grid_rho_profile_reset();
    let forward_fused_ns = measure_avg(config.repetitions, || {
        let output = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
            &query,
            &value,
            Some(&initial_rho),
            grid,
            neighborhood,
            &decay,
        )
        .expect("fused local-grid");
        sync_tensor(output.context);
    });
    let forward_profile = local_grid_rho_profile_snapshot();

    let rollout_reference_ns = measure_avg(config.repetitions, || {
        let (_, rho) = reference_rollout(
            query.clone(),
            value.clone(),
            initial_rho.clone(),
            grid,
            neighborhood,
            decay.clone(),
            config.steps,
        );
        sync_tensor(rho);
    });

    local_grid_rho_profile_reset();
    let rollout_fused_ns = measure_avg(config.repetitions, || {
        let mut rho = initial_rho.clone();
        for _ in 0..config.steps {
            let output = try_fused_local_grid_rho_attention_wgpu_head_decay::<Backend>(
                &query,
                &value,
                Some(&rho),
                grid,
                neighborhood,
                &decay,
            )
            .expect("fused rollout");
            rho = output.rho;
        }
        sync_tensor(rho);
    });
    let rollout_profile = local_grid_rho_profile_snapshot();

    let forward_tokens = (case.batch * tokens) as f64;
    let rollout_tokens = (case.batch * tokens * config.steps.max(1)) as f64;

    VisionRhoBenchCaseResult {
        case,
        problem_shape: format!(
            "b{} h{} grid={}x{} latent={} embd={} radius={} diagonals={} self_edges={}",
            case.batch,
            case.heads,
            case.height,
            case.width,
            case.latent,
            case.embd,
            case.radius,
            case.diagonals,
            case.self_edges
        ),
        forward: VisionRhoBenchMeasurementResult {
            mode: "forward_only",
            reference_time_ms: ns_to_ms(forward_reference_ns),
            fused_time_ms: ns_to_ms(forward_fused_ns),
            speedup_x: forward_reference_ns / forward_fused_ns,
            throughput_per_sec: forward_tokens / (forward_fused_ns / 1e9),
            fused_kernel_calls: forward_profile.calls,
            fused_kernel_launches: forward_profile.launches,
            fused_kernel_dispatch_ms: forward_profile.dispatch_ns as f64 / 1e6,
            fused_transient_allocations: forward_profile.transient_allocations,
            fused_metadata_upload_bytes: forward_profile.metadata_upload_bytes,
        },
        resident_rollout: VisionRhoBenchMeasurementResult {
            mode: "resident_rollout",
            reference_time_ms: ns_to_ms(rollout_reference_ns),
            fused_time_ms: ns_to_ms(rollout_fused_ns),
            speedup_x: rollout_reference_ns / rollout_fused_ns,
            throughput_per_sec: rollout_tokens / (rollout_fused_ns / 1e9),
            fused_kernel_calls: rollout_profile.calls,
            fused_kernel_launches: rollout_profile.launches,
            fused_kernel_dispatch_ms: rollout_profile.dispatch_ns as f64 / 1e6,
            fused_transient_allocations: rollout_profile.transient_allocations,
            fused_metadata_upload_bytes: rollout_profile.metadata_upload_bytes,
        },
        context_error: compare_tensors(reference_context, fused.context),
        rho_error: compare_tensors(reference_rho, fused.rho),
    }
}

fn reference_rollout(
    query: Tensor<Backend, 4>,
    value: Tensor<Backend, 4>,
    mut rho: Tensor<Backend, 5>,
    grid: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
    decay: Tensor<Backend, 1>,
    steps: usize,
) -> (Tensor<Backend, 4>, Tensor<Backend, 5>) {
    let mut context = Tensor::<Backend, 4>::zeros(
        [
            query.shape().dims::<4>()[0],
            query.shape().dims::<4>()[1],
            grid.token_count(),
            value.shape().dims::<4>()[3],
        ],
        &query.device(),
    );
    for _ in 0..steps.max(1) {
        let output = reference_local_grid_rho_head_decay(
            query.clone(),
            value.clone(),
            rho,
            grid,
            neighborhood,
            decay.clone(),
        );
        context = output.0;
        rho = output.1;
    }
    (context, rho)
}

fn reference_local_grid_rho_head_decay(
    query: Tensor<Backend, 4>,
    value: Tensor<Backend, 4>,
    rho: Tensor<Backend, 5>,
    grid: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
    decay: Tensor<Backend, 1>,
) -> (Tensor<Backend, 4>, Tensor<Backend, 5>) {
    let [batch, heads, patch_tokens, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let decay = decay.reshape([1, heads, 1, 1, 1]);

    let mut state = rho;
    let mut outputs = Vec::with_capacity(patch_tokens);
    for target in 0..patch_tokens {
        let ty = target / grid.width.max(1);
        let tx = target % grid.width.max(1);
        let q_t = query.clone().slice_dim(2, target..target + 1);
        let mut context = Tensor::<Backend, 4>::zeros([batch, heads, 1, embd], &query.device());
        for dy in -(neighborhood.radius as isize)..=(neighborhood.radius as isize) {
            for dx in -(neighborhood.radius as isize)..=(neighborhood.radius as isize) {
                if dy == 0 && dx == 0 && !neighborhood.self_edges {
                    continue;
                }
                if !neighborhood.diagonals && dy != 0 && dx != 0 {
                    continue;
                }
                let sy = ty as isize + dy;
                let sx = tx as isize + dx;
                if sy < 0 || sy >= grid.height as isize || sx < 0 || sx >= grid.width as isize {
                    continue;
                }
                let source = sy as usize * grid.width + sx as usize;
                let source_state = state.clone().slice_dim(2, source..source + 1);
                let msg = source_state
                    .mul(q_t.clone().unsqueeze_dim::<5>(4))
                    .sum_dims_squeeze::<4, usize>(&[3]);
                context = context + msg;
            }
        }
        outputs.push(context);
    }

    for target in 0..patch_tokens {
        let q_t = query
            .clone()
            .slice_dim(2, target..target + 1)
            .unsqueeze_dim::<5>(4);
        let v_t = if value_heads == 1 {
            value.clone().slice_dim(1, 0..1)
        } else {
            value.clone().slice_dim(1, 0..heads)
        }
        .slice_dim(2, target..target + 1)
        .unsqueeze_dim::<5>(3);
        let next = state
            .clone()
            .slice_dim(2, target..target + 1)
            .mul(decay.clone())
            .add(q_t.mul(v_t));
        state = state.slice_assign(
            [0..batch, 0..heads, target..target + 1, 0..latent, 0..embd],
            next,
        );
    }

    (Tensor::cat(outputs, 2), state)
}

fn compare_tensors<const D: usize>(
    lhs: Tensor<Backend, D>,
    rhs: Tensor<Backend, D>,
) -> VisionRhoBenchErrorMetrics {
    let lhs = lhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs");
    let rhs = rhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs");
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    let mut count = 0usize;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (a - b).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
        count += 1;
    }
    VisionRhoBenchErrorMetrics {
        max_abs,
        mean_abs: sum_abs / count.max(1) as f32,
    }
}

fn measure_avg<F>(repetitions: usize, mut f: F) -> f64
where
    F: FnMut(),
{
    let mut total = 0.0_f64;
    for _ in 0..repetitions.max(1) {
        let start = Instant::now();
        f();
        total += start.elapsed().as_nanos() as f64;
    }
    total / repetitions.max(1) as f64
}

fn sync_tensor<const D: usize>(tensor: Tensor<Backend, D>) {
    let _ = tensor.into_data();
}

fn ns_to_ms(ns: f64) -> f64 {
    ns / 1_000_000.0
}

fn format_markdown(report: &VisionRhoBenchReport) -> String {
    let mut out = String::new();
    push_vision_artifact_markdown_prelude(&mut out, "Vision Rho Benchmark", &report.artifact);
    let _ = writeln!(&mut out, "- adapter: {}", report.adapter);
    let _ = writeln!(&mut out, "- warmup: {}", report.warmup);
    let _ = writeln!(&mut out, "- repetitions: {}", report.repetitions);
    let _ = writeln!(&mut out, "- steps: {}", report.steps);
    let _ = writeln!(&mut out);
    let _ = writeln!(
        &mut out,
        "| case | mode | ref ms | fused ms | speedup | fused throughput/s | launches | context max abs | rho max abs |"
    );
    let _ = writeln!(
        &mut out,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    );
    for case in &report.cases {
        for measurement in [&case.forward, &case.resident_rollout] {
            let _ = writeln!(
                &mut out,
                "| {} | {} | {:.2} | {:.2} | {:.2}x | {:.1} | {} | {:.5} | {:.5} |",
                case.case.name,
                measurement.mode,
                measurement.reference_time_ms,
                measurement.fused_time_ms,
                measurement.speedup_x,
                measurement.throughput_per_sec,
                measurement.fused_kernel_launches,
                case.context_error.max_abs,
                case.rho_error.max_abs,
            );
        }
    }
    out
}
