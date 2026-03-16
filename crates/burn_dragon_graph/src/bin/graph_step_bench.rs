use std::fmt::Write as _;
use std::fs;
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_dragon_graph::api::expert::CompiledGraphRouting;
use burn_dragon_graph::{
    GraphCsrAdjacency, GraphDragon, GraphDragonConfig, GraphTopologyRouting, StructuredStepMode,
};
use burn_dragon_wgpu::api::graph::{
    sparse_graph_rho_profile_reset, sparse_graph_rho_profile_snapshot,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone, Copy)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    node_count: usize,
    cluster_count: usize,
    global_count: usize,
    avg_degree: usize,
    embed_dim: usize,
    rank: usize,
    value_dim: usize,
}

#[derive(Clone, Copy)]
struct Args {
    warmup: usize,
    repetitions: usize,
    steps: usize,
    markdown_path: Option<&'static str>,
    csv_path: Option<&'static str>,
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    max_abs: f32,
    mean_abs: f32,
}

#[derive(Clone)]
struct CaseResult {
    case: BenchCase,
    problem_shape: String,
    compile_ms: f64,
    baseline_ms: f64,
    wrapped_fused_ms: f64,
    persistent_fused_ms: f64,
    baseline_step_ms: f64,
    wrapped_fused_step_ms: f64,
    persistent_fused_step_ms: f64,
    baseline_steps_per_sec: f64,
    wrapped_fused_steps_per_sec: f64,
    persistent_fused_steps_per_sec: f64,
    wrapped_speedup_x: f64,
    persistent_speedup_x: f64,
    persistent_vs_wrapped_speedup_x: f64,
    edges_per_step: usize,
    persistent_fused_edges_per_sec: f64,
    persistent_fused_kernel_calls: u64,
    persistent_fused_kernel_launches: u64,
    persistent_fused_kernel_dispatch_ms: f64,
    persistent_fused_metadata_reuse_bytes: u64,
    persistent_fused_transient_allocations: u64,
    node_error: ErrorMetrics,
    cluster_rho_error: ErrorMetrics,
    global_rho_error: ErrorMetrics,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "small",
        batch: 2,
        node_count: 128,
        cluster_count: 16,
        global_count: 2,
        avg_degree: 8,
        embed_dim: 64,
        rank: 16,
        value_dim: 64,
    },
    BenchCase {
        name: "medium",
        batch: 2,
        node_count: 512,
        cluster_count: 64,
        global_count: 4,
        avg_degree: 8,
        embed_dim: 96,
        rank: 24,
        value_dim: 96,
    },
    BenchCase {
        name: "large",
        batch: 1,
        node_count: 2048,
        cluster_count: 256,
        global_count: 8,
        avg_degree: 12,
        embed_dim: 128,
        rank: 32,
        value_dim: 128,
    },
];

fn main() {
    let args = parse_args();
    let device = Device::default();
    init_runtime(&device);

    let (adapter_name, adapter_type) = adapter_info();
    assert!(
        !matches!(adapter_type, wgpu::DeviceType::Cpu),
        "graph benchmark selected a CPU adapter; refusing to continue (adapter: {adapter_name})"
    );

    let mut results = Vec::with_capacity(CASES.len());
    for (case_idx, case) in CASES.iter().copied().enumerate() {
        <Backend as BackendTrait>::seed(&device, 2_026 + case_idx as u64);
        results.push(run_case(case, &device, args));
    }

    let markdown = format_markdown(&adapter_name, args, &results);
    let csv = format_csv(&results);
    println!("{markdown}");

    if let Some(path) = args.markdown_path {
        fs::write(path, &markdown).expect("write benchmark markdown");
        eprintln!("wrote markdown benchmark report to {path}");
    }
    if let Some(path) = args.csv_path {
        fs::write(path, &csv).expect("write benchmark csv");
        eprintln!("wrote csv benchmark report to {path}");
    }
}

fn parse_args() -> Args {
    let mut warmup = 3usize;
    let mut repetitions = 10usize;
    let mut steps = 8usize;
    let mut markdown_path = None;
    let mut csv_path = None;

    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let mut index = 0usize;
    while index < args.len() {
        match args[index].as_str() {
            "--warmup" => {
                index += 1;
                warmup = args
                    .get(index)
                    .expect("--warmup value")
                    .parse()
                    .expect("warmup usize");
            }
            "--repetitions" => {
                index += 1;
                repetitions = args
                    .get(index)
                    .expect("--repetitions value")
                    .parse()
                    .expect("repetitions usize");
            }
            "--steps" => {
                index += 1;
                steps = args
                    .get(index)
                    .expect("--steps value")
                    .parse()
                    .expect("steps usize");
            }
            "--markdown-path" => {
                index += 1;
                let path = args.get(index).expect("--markdown-path value").clone();
                markdown_path = Some(leak_string(path));
            }
            "--csv-path" => {
                index += 1;
                let path = args.get(index).expect("--csv-path value").clone();
                csv_path = Some(leak_string(path));
            }
            "--help" | "-h" => {
                print_help_and_exit();
            }
            other => {
                panic!("unknown arg: {other}");
            }
        }
        index += 1;
    }

    Args {
        warmup: warmup.max(1),
        repetitions: repetitions.max(1),
        steps: steps.max(1),
        markdown_path,
        csv_path,
    }
}

fn leak_string(value: String) -> &'static str {
    Box::leak(value.into_boxed_str())
}

fn print_help_and_exit() -> ! {
    println!(
        "graph_step_bench [--warmup N] [--repetitions N] [--steps N] [--markdown-path PATH] [--csv-path PATH]"
    );
    std::process::exit(0);
}

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn adapter_info() -> (String, wgpu::DeviceType) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu adapter");
    let info = adapter.get_info();
    (
        format!("{} ({:?})", info.name, info.device_type),
        info.device_type,
    )
}

fn run_case(case: BenchCase, device: &Device, args: Args) -> CaseResult {
    let routing = synthetic_routing(case);
    let compile_start = Instant::now();
    let compiled = CompiledGraphRouting::<Backend>::new(routing.clone(), device);
    let compile_ms = compile_start.elapsed().as_secs_f64() * 1000.0;

    let model = GraphDragon::new(
        GraphDragonConfig {
            embed_dim: case.embed_dim,
            rank: case.rank,
            value_dim: case.value_dim,
            predict_decay: 0.97,
            mode_embeddings: true,
        },
        device,
    );
    let node_observation = Tensor::<Backend, 3>::random(
        [case.batch, case.node_count, case.embed_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let cluster_observation = Tensor::<Backend, 3>::random(
        [case.batch, case.cluster_count, case.embed_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let state = model
        .state_from_observations(&routing, node_observation, cluster_observation)
        .expect("graph state init");

    let reference = model
        .step(state.clone(), &routing, StructuredStepMode::Predict)
        .expect("reference step");
    let fused = model
        .step_compiled(state.clone(), &compiled, StructuredStepMode::Predict)
        .expect("fused step");

    let node_error = tensor_error(reference.state.node_state(), fused.state.node_state());
    let cluster_rho_error = tensor_error(reference.state.cluster_rho(), fused.state.cluster_rho());
    let global_rho_error = tensor_error(reference.state.global_rho(), fused.state.global_rho());

    assert!(
        node_error.max_abs <= 7e-4,
        "node parity drift too large for {}: {}",
        case.name,
        node_error.max_abs
    );
    assert!(
        cluster_rho_error.max_abs <= 7e-4,
        "cluster rho parity drift too large for {}: {}",
        case.name,
        cluster_rho_error.max_abs
    );
    assert!(
        global_rho_error.max_abs <= 7e-4,
        "global rho parity drift too large for {}: {}",
        case.name,
        global_rho_error.max_abs
    );

    for _ in 0..args.warmup {
        let _ = model
            .rollout(
                state.clone(),
                &routing,
                args.steps,
                StructuredStepMode::Predict,
            )
            .expect("baseline warmup");
        let mut wrapped_state = state.clone();
        for _ in 0..args.steps {
            wrapped_state = model
                .step_compiled(wrapped_state, &compiled, StructuredStepMode::Predict)
                .expect("wrapped fused warmup")
                .state;
        }
        let _ = model
            .rollout_compiled(
                state.clone(),
                &compiled,
                args.steps,
                StructuredStepMode::Predict,
            )
            .expect("persistent fused warmup");
    }

    let baseline_ms = timed_rollout_ms(args.repetitions, || {
        let state = model
            .rollout(
                state.clone(),
                &routing,
                args.steps,
                StructuredStepMode::Predict,
            )
            .expect("baseline rollout");
        rollout_sync_tensor(&state)
    });
    let wrapped_fused_ms = timed_rollout_ms(args.repetitions, || {
        let mut state = state.clone();
        for _ in 0..args.steps {
            state = model
                .step_compiled(state, &compiled, StructuredStepMode::Predict)
                .expect("wrapped fused rollout")
                .state;
        }
        rollout_sync_tensor(&state)
    });
    sparse_graph_rho_profile_reset();
    let persistent_fused_ms = timed_rollout_ms(args.repetitions, || {
        let state = model
            .rollout_compiled(
                state.clone(),
                &compiled,
                args.steps,
                StructuredStepMode::Predict,
            )
            .expect("persistent fused rollout");
        rollout_sync_tensor(&state)
    });
    let persistent_profile = sparse_graph_rho_profile_snapshot();

    let total_steps = (args.repetitions * args.steps) as f64;
    let baseline_step_ms = baseline_ms / total_steps;
    let wrapped_fused_step_ms = wrapped_fused_ms / total_steps;
    let persistent_fused_step_ms = persistent_fused_ms / total_steps;
    let baseline_steps_per_sec = total_steps * 1000.0 / baseline_ms.max(f64::EPSILON);
    let wrapped_fused_steps_per_sec = total_steps * 1000.0 / wrapped_fused_ms.max(f64::EPSILON);
    let persistent_fused_steps_per_sec =
        total_steps * 1000.0 / persistent_fused_ms.max(f64::EPSILON);
    let edges_per_step = routing_edge_count(&routing, case);
    let persistent_fused_edges_per_sec = persistent_fused_steps_per_sec * edges_per_step as f64;

    CaseResult {
        case,
        problem_shape: format!(
            "b{} n{} c{} g{} deg{} embd{} rank{} value{}",
            case.batch,
            case.node_count,
            case.cluster_count,
            case.global_count,
            case.avg_degree,
            case.embed_dim,
            case.rank,
            case.value_dim
        ),
        compile_ms,
        baseline_ms,
        wrapped_fused_ms,
        persistent_fused_ms,
        baseline_step_ms,
        wrapped_fused_step_ms,
        persistent_fused_step_ms,
        baseline_steps_per_sec,
        wrapped_fused_steps_per_sec,
        persistent_fused_steps_per_sec,
        wrapped_speedup_x: baseline_ms / wrapped_fused_ms.max(f64::EPSILON),
        persistent_speedup_x: baseline_ms / persistent_fused_ms.max(f64::EPSILON),
        persistent_vs_wrapped_speedup_x: wrapped_fused_ms / persistent_fused_ms.max(f64::EPSILON),
        edges_per_step,
        persistent_fused_edges_per_sec,
        persistent_fused_kernel_calls: persistent_profile.calls,
        persistent_fused_kernel_launches: persistent_profile.launches,
        persistent_fused_kernel_dispatch_ms: persistent_profile.dispatch_ns as f64 / 1e6,
        persistent_fused_metadata_reuse_bytes: persistent_profile.metadata_reuse_bytes,
        persistent_fused_transient_allocations: persistent_profile.transient_allocations,
        node_error,
        cluster_rho_error,
        global_rho_error,
    }
}

fn timed_rollout_ms(mut repeats: usize, mut f: impl FnMut() -> Tensor<Backend, 1>) -> f64 {
    let start = Instant::now();
    let mut sync = None;
    while repeats > 0 {
        sync = Some(f());
        repeats -= 1;
    }
    if let Some(sync) = sync {
        let _ = sync.into_data();
    }
    start.elapsed().as_secs_f64() * 1000.0
}

fn synthetic_routing(case: BenchCase) -> GraphTopologyRouting {
    let mut edges = Vec::with_capacity(case.node_count * case.avg_degree.max(1));
    for source in 0..case.node_count {
        for hop in 0..case.avg_degree.max(1) {
            let stride = hop.saturating_mul(13).saturating_add(1);
            let target = (source + stride + (source / 7) + hop * 3) % case.node_count;
            edges.push((source, target));
        }
    }

    let mut routing = GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(case.node_count, case.node_count, &edges)
            .expect("synthetic node adjacency"),
    )
    .expect("synthetic routing");

    let cluster_assignments = (0..case.node_count)
        .map(|node| node * case.cluster_count / case.node_count.max(1))
        .map(|cluster| cluster.min(case.cluster_count.saturating_sub(1)))
        .collect::<Vec<_>>();
    routing = routing
        .with_cluster_assignments(case.cluster_count, &cluster_assignments)
        .expect("cluster assignments");

    let node_global_assignments = (0..case.node_count)
        .map(|node| node % case.global_count.max(1))
        .collect::<Vec<_>>();
    routing = routing
        .with_node_global_assignments(case.global_count.max(1), &node_global_assignments)
        .expect("node global assignments");

    let cluster_global_assignments = (0..case.cluster_count)
        .map(|cluster| cluster % case.global_count.max(1))
        .collect::<Vec<_>>();
    routing
        .with_cluster_global_assignments(case.global_count.max(1), &cluster_global_assignments)
        .expect("cluster global assignments")
}

fn routing_edge_count(routing: &GraphTopologyRouting, case: BenchCase) -> usize {
    let mut edges = routing.node_neighbors().edge_count();
    edges += case.node_count;
    edges += case.cluster_count;
    edges += routing
        .node_to_cluster()
        .map(|route| route.edge_count())
        .unwrap_or(0);
    edges += routing
        .node_to_global()
        .map(|route| route.edge_count())
        .unwrap_or(0);
    edges += routing
        .cluster_to_global()
        .map(|route| route.edge_count())
        .unwrap_or(0);
    edges
}

fn tensor_error<const D: usize>(lhs: Tensor<Backend, D>, rhs: Tensor<Backend, D>) -> ErrorMetrics {
    let lhs = lhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    let mut max_abs = 0.0_f32;
    let mut sum_abs = 0.0_f32;
    let len = lhs.len().max(1) as f32;
    for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
        let diff = (lhs - rhs).abs();
        max_abs = max_abs.max(diff);
        sum_abs += diff;
    }
    ErrorMetrics {
        max_abs,
        mean_abs: sum_abs / len,
    }
}

fn rollout_sync_tensor(
    state: &burn_dragon_graph::GraphTopologyState<Backend>,
) -> Tensor<Backend, 1> {
    state.global_rho().mean().reshape([1])
}

fn format_markdown(adapter: &str, args: Args, results: &[CaseResult]) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# burn_dragon_graph sparse graph fused benchmark").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- Adapter: {adapter}").unwrap();
    writeln!(&mut out, "- Warmup: {}", args.warmup).unwrap();
    writeln!(&mut out, "- Repetitions: {}", args.repetitions).unwrap();
    writeln!(&mut out, "- Predict steps per repetition: {}", args.steps).unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| case | graph | dims | compile ms | baseline step ms | wrapped fused step ms | persistent fused step ms | wrapped speedup | persistent speedup | persistent/wrapped | persistent fused steps/s | persistent fused edges/s | launches | dispatch ms | node max abs | cluster rho max abs | global rho max abs |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for result in results {
        writeln!(
            &mut out,
            "| {} | b{} n{} c{} g{} deg{} | e{} r{} v{} | {:.2} | {:.4} | {:.4} | {:.4} | {:.2}x | {:.2}x | {:.2}x | {:.1} | {:.0} | {} | {:.2} | {:.2e} | {:.2e} | {:.2e} |",
            result.case.name,
            result.case.batch,
            result.case.node_count,
            result.case.cluster_count,
            result.case.global_count,
            result.case.avg_degree,
            result.case.embed_dim,
            result.case.rank,
            result.case.value_dim,
            result.compile_ms,
            result.baseline_step_ms,
            result.wrapped_fused_step_ms,
            result.persistent_fused_step_ms,
            result.wrapped_speedup_x,
            result.persistent_speedup_x,
            result.persistent_vs_wrapped_speedup_x,
            result.persistent_fused_steps_per_sec,
            result.persistent_fused_edges_per_sec,
            result.persistent_fused_kernel_launches,
            result.persistent_fused_kernel_dispatch_ms,
            result.node_error.max_abs,
            result.cluster_rho_error.max_abs,
            result.global_rho_error.max_abs,
        )
        .unwrap();
    }
    out
}

fn format_csv(results: &[CaseResult]) -> String {
    let mut out = String::from(
        "case,problem_shape,batch,node_count,cluster_count,global_count,avg_degree,embed_dim,rank,value_dim,compile_ms,baseline_ms,wrapped_fused_ms,persistent_fused_ms,baseline_step_ms,wrapped_fused_step_ms,persistent_fused_step_ms,baseline_steps_per_sec,wrapped_fused_steps_per_sec,persistent_fused_steps_per_sec,wrapped_speedup_x,persistent_speedup_x,persistent_vs_wrapped_speedup_x,edges_per_step,persistent_fused_edges_per_sec,persistent_fused_kernel_calls,persistent_fused_kernel_launches,persistent_fused_kernel_dispatch_ms,persistent_fused_metadata_reuse_bytes,persistent_fused_transient_allocations,node_max_abs,node_mean_abs,cluster_rho_max_abs,cluster_rho_mean_abs,global_rho_max_abs,global_rho_mean_abs\n",
    );
    for result in results {
        writeln!(
            &mut out,
            "{},{},{},{},{},{},{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{:.6},{},{},{:.6},{},{},{:.8},{:.8},{:.8},{:.8},{:.8},{:.8}",
            result.case.name,
            result.problem_shape,
            result.case.batch,
            result.case.node_count,
            result.case.cluster_count,
            result.case.global_count,
            result.case.avg_degree,
            result.case.embed_dim,
            result.case.rank,
            result.case.value_dim,
            result.compile_ms,
            result.baseline_ms,
            result.wrapped_fused_ms,
            result.persistent_fused_ms,
            result.baseline_step_ms,
            result.wrapped_fused_step_ms,
            result.persistent_fused_step_ms,
            result.baseline_steps_per_sec,
            result.wrapped_fused_steps_per_sec,
            result.persistent_fused_steps_per_sec,
            result.wrapped_speedup_x,
            result.persistent_speedup_x,
            result.persistent_vs_wrapped_speedup_x,
            result.edges_per_step,
            result.persistent_fused_edges_per_sec,
            result.persistent_fused_kernel_calls,
            result.persistent_fused_kernel_launches,
            result.persistent_fused_kernel_dispatch_ms,
            result.persistent_fused_metadata_reuse_bytes,
            result.persistent_fused_transient_allocations,
            result.node_error.max_abs,
            result.node_error.mean_abs,
            result.cluster_rho_error.max_abs,
            result.cluster_rho_error.mean_abs,
            result.global_rho_error.max_abs,
            result.global_rho_error.mean_abs,
        )
        .unwrap();
    }
    out
}
