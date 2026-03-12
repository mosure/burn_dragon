use std::fmt::Write as _;
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_autodiff::Autodiff;
use burn_dragon_graph::{
    GraphCsrAdjacency, GraphDragon, GraphDragonConfig, GraphTopologyRouting, GraphTopologyState,
    StructuredStepMode,
};
use burn_dragon_graph::api::expert::CompiledGraphRouting;
use burn_dragon_wgpu::api::graph::{
    sparse_graph_rho_profile_reset, sparse_graph_rho_profile_snapshot,
};
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type TrainBackend = Autodiff<InnerBackend>;
type Device = <TrainBackend as BackendTrait>::Device;

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

#[derive(Clone)]
struct Args {
    warmup: usize,
    repetitions: usize,
    steps: usize,
    case_filter: Option<Vec<String>>,
}

#[derive(Clone, Copy)]
struct ErrorMetrics {
    max_abs: f32,
}

#[derive(Clone, Copy)]
struct Measurement {
    elapsed_ms: f64,
    steps_per_sec: f64,
    kernel_calls: u64,
    launches: u64,
    dispatch_ms: f64,
}

#[derive(Clone)]
struct CaseResult {
    case: BenchCase,
    mode: &'static str,
    steps: usize,
    baseline: Measurement,
    fused: Measurement,
    speedup_x: f64,
    loss_abs_diff: f32,
    node_error: ErrorMetrics,
    cluster_error: ErrorMetrics,
    node_rho_error: ErrorMetrics,
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
    let adapter = adapter_info();

    let selected_cases = CASES
        .iter()
        .copied()
        .filter(|case| {
            args.case_filter.as_ref().is_none_or(|filters| {
                filters.iter().any(|filter| filter == case.name)
            })
        })
        .collect::<Vec<_>>();
    assert!(!selected_cases.is_empty(), "graph_step_fb_bench selected no cases");

    let mut results = Vec::with_capacity(selected_cases.len() * 2);
    for (case_idx, case) in selected_cases.into_iter().enumerate() {
        <TrainBackend as BackendTrait>::seed(&device, 9_001 + case_idx as u64);
        results.push(run_case(case, "one_step", 1, &device, &args));
        results.push(run_case(
            case,
            "resident_rollout",
            args.steps.max(1),
            &device,
            &args,
        ));
    }

    println!("{}", format_markdown(&adapter, args, &results));
}

fn parse_args() -> Args {
    let mut warmup = 1usize;
    let mut repetitions = 3usize;
    let mut steps = 2usize;
    let mut case_filter = None;
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
            "--cases" => {
                index += 1;
                let value = args.get(index).expect("--cases value");
                let cases = value
                    .split(',')
                    .map(|part| part.trim().to_string())
                    .filter(|part| !part.is_empty())
                    .collect::<Vec<_>>();
                case_filter = Some(cases);
            }
            "--help" | "-h" => {
                println!(
                    "graph_step_fb_bench [--warmup N] [--repetitions N] [--steps N] [--cases small,medium]"
                );
                std::process::exit(0);
            }
            other => panic!("unknown arg: {other}"),
        }
        index += 1;
    }
    Args {
        warmup: warmup.max(1),
        repetitions: repetitions.max(1),
        steps: steps.max(1),
        case_filter,
    }
}

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn adapter_info() -> String {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("wgpu adapter");
    let info = adapter.get_info();
    format!("{} ({:?})", info.name, info.device_type)
}

fn run_case(
    case: BenchCase,
    mode_name: &'static str,
    steps: usize,
    device: &Device,
    args: &Args,
) -> CaseResult {
    let routing = synthetic_routing(case);
    let compiled = CompiledGraphRouting::<TrainBackend>::new(routing.clone(), device);
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
    let node_observation = Tensor::<TrainBackend, 3>::random(
        [case.batch, case.node_count, case.embed_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let cluster_observation = Tensor::<TrainBackend, 3>::random(
        [case.batch, case.cluster_count, case.embed_dim],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let state = model
        .state_from_observations(&routing, node_observation, cluster_observation)
        .expect("state init");

    let (loss_abs_diff, node_error, cluster_error, node_rho_error, cluster_rho_error, global_rho_error) =
        parity_snapshot(&model, state.clone(), &routing, &compiled, steps);

    for _ in 0..args.warmup {
        let _ = run_forward_backward(&model, state.clone(), &routing, None, steps, device);
        let _ = run_forward_backward(&model, state.clone(), &routing, Some(&compiled), steps, device);
    }

    let baseline = summarize_measurements((0..args.repetitions).map(|_| {
        run_forward_backward(&model, state.clone(), &routing, None, steps, device)
    }).collect::<Vec<_>>(), steps);
    let fused = summarize_measurements((0..args.repetitions).map(|_| {
        run_forward_backward(&model, state.clone(), &routing, Some(&compiled), steps, device)
    }).collect::<Vec<_>>(), steps);

    CaseResult {
        case,
        mode: mode_name,
        steps,
        baseline,
        fused,
        speedup_x: baseline.elapsed_ms / fused.elapsed_ms.max(f64::EPSILON),
        loss_abs_diff,
        node_error,
        cluster_error,
        node_rho_error,
        cluster_rho_error,
        global_rho_error,
    }
}

#[derive(Clone, Copy)]
struct StepMeasurement {
    elapsed_ms: f64,
    kernel_calls: u64,
    launches: u64,
    dispatch_ms: f64,
}

fn run_forward_backward(
    model: &GraphDragon<TrainBackend>,
    state: GraphTopologyState<TrainBackend>,
    routing: &GraphTopologyRouting,
    compiled: Option<&CompiledGraphRouting<TrainBackend>>,
    steps: usize,
    device: &Device,
) -> StepMeasurement {
    let _ = <TrainBackend as BackendTrait>::sync(device);
    sparse_graph_rho_profile_reset();
    let started = Instant::now();

    let state = match compiled {
        Some(compiled) => model
            .rollout_compiled(state, compiled, steps, StructuredStepMode::Predict)
            .expect("compiled rollout"),
        None => model
            .rollout(state, routing, steps, StructuredStepMode::Predict)
            .expect("reference rollout"),
    };
    let loss = state_loss(state);
    let _ = loss.backward();
    let _ = <TrainBackend as BackendTrait>::sync(device);
    let profile = sparse_graph_rho_profile_snapshot();

    StepMeasurement {
        elapsed_ms: started.elapsed().as_secs_f64() * 1000.0,
        kernel_calls: profile.calls,
        launches: profile.launches,
        dispatch_ms: profile.dispatch_ns as f64 / 1e6,
    }
}

fn state_loss(state: GraphTopologyState<TrainBackend>) -> Tensor<TrainBackend, 1> {
    state.node_state().tanh().powf_scalar(2.0).mean()
        + state.cluster_state().tanh().powf_scalar(2.0).mean()
        + state.node_rho().tanh().powf_scalar(2.0).mean()
        + state.cluster_rho().tanh().powf_scalar(2.0).mean()
        + state.global_rho().tanh().powf_scalar(2.0).mean()
}

fn summarize_measurements(samples: Vec<StepMeasurement>, steps: usize) -> Measurement {
    let count = samples.len().max(1) as f64;
    let elapsed_ms = samples.iter().map(|sample| sample.elapsed_ms).sum::<f64>() / count;
    let kernel_calls = (samples.iter().map(|sample| sample.kernel_calls as f64).sum::<f64>() / count).round() as u64;
    let launches = (samples.iter().map(|sample| sample.launches as f64).sum::<f64>() / count).round() as u64;
    let dispatch_ms = samples.iter().map(|sample| sample.dispatch_ms).sum::<f64>() / count;
    Measurement {
        elapsed_ms,
        steps_per_sec: (steps as f64) * 1000.0 / elapsed_ms.max(f64::EPSILON),
        kernel_calls,
        launches,
        dispatch_ms,
    }
}

#[allow(clippy::type_complexity)]
fn parity_snapshot(
    model: &GraphDragon<TrainBackend>,
    state: GraphTopologyState<TrainBackend>,
    routing: &GraphTopologyRouting,
    compiled: &CompiledGraphRouting<TrainBackend>,
    steps: usize,
) -> (
    f32,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
    ErrorMetrics,
) {
    let reference = model
        .rollout(state.clone(), routing, steps, StructuredStepMode::Predict)
        .expect("reference rollout");
    let fused = model
        .rollout_compiled(state, compiled, steps, StructuredStepMode::Predict)
        .expect("compiled rollout");
    let loss_abs_diff = (scalar_to_f32(state_loss(reference.clone())) - scalar_to_f32(state_loss(fused.clone()))).abs();

    (
        loss_abs_diff,
        error_metrics(reference.node_state(), fused.node_state()),
        error_metrics(reference.cluster_state(), fused.cluster_state()),
        error_metrics(reference.node_rho(), fused.node_rho()),
        error_metrics(reference.cluster_rho(), fused.cluster_rho()),
        error_metrics(reference.global_rho(), fused.global_rho()),
    )
}

fn scalar_to_f32(tensor: Tensor<TrainBackend, 1>) -> f32 {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("scalar vec")[0]
}

fn error_metrics<const D: usize>(
    reference: Tensor<TrainBackend, D>,
    fused: Tensor<TrainBackend, D>,
) -> ErrorMetrics {
    let reference = reference
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference vec");
    let fused = fused
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused vec");
    let mut max_abs = 0.0_f32;
    for (reference, fused) in reference.iter().zip(fused.iter()) {
        let diff = (*reference - *fused).abs();
        max_abs = max_abs.max(diff);
    }
    ErrorMetrics { max_abs }
}

fn synthetic_routing(case: BenchCase) -> GraphTopologyRouting {
    let mut edges = Vec::with_capacity(case.node_count * case.avg_degree.max(1));
    for source in 0..case.node_count {
        for offset in 1..=case.avg_degree.max(1) {
            edges.push((source, (source + offset) % case.node_count));
        }
    }
    GraphTopologyRouting::new(
        GraphCsrAdjacency::try_from_edges(case.node_count, case.node_count, &edges)
            .expect("valid node adjacency"),
    )
    .expect("valid routing")
    .with_cluster_assignments(
        case.cluster_count,
        &(0..case.node_count)
            .map(|node| node % case.cluster_count.max(1))
            .collect::<Vec<_>>(),
    )
    .expect("valid cluster assignments")
    .with_node_global_assignments(
        case.global_count,
        &(0..case.node_count)
            .map(|node| node % case.global_count.max(1))
            .collect::<Vec<_>>(),
    )
    .expect("valid node/global assignments")
    .with_cluster_global_assignments(
        case.global_count,
        &(0..case.cluster_count)
            .map(|cluster| cluster % case.global_count.max(1))
            .collect::<Vec<_>>(),
    )
    .expect("valid cluster/global assignments")
}

fn format_markdown(adapter: &str, args: Args, results: &[CaseResult]) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# burn_dragon_graph sparse graph forward+backward benchmark").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- Adapter: {adapter}").unwrap();
    writeln!(&mut out, "- Warmup: {}", args.warmup).unwrap();
    writeln!(&mut out, "- Repetitions: {}", args.repetitions).unwrap();
    writeln!(&mut out, "- Rollout steps: {}", args.steps).unwrap();
    writeln!(&mut out).unwrap();
    writeln!(
        &mut out,
        "| case | mode | steps | baseline ms | fused ms | speedup x | baseline step/s | fused step/s | fused calls | fused launches | fused dispatch ms | loss abs diff | node max abs | cluster max abs | node rho max abs | cluster rho max abs | global rho max abs |"
    )
    .unwrap();
    writeln!(
        &mut out,
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for result in results {
        writeln!(
            &mut out,
            "| {} | {} | {} | {:.3} | {:.3} | {:.3} | {:.1} | {:.1} | {} | {} | {:.3} | {:.6} | {:.2e} | {:.2e} | {:.2e} | {:.2e} | {:.2e} |",
            result.case.name,
            result.mode,
            result.steps,
            result.baseline.elapsed_ms,
            result.fused.elapsed_ms,
            result.speedup_x,
            result.baseline.steps_per_sec,
            result.fused.steps_per_sec,
            result.fused.kernel_calls,
            result.fused.launches,
            result.fused.dispatch_ms,
            result.loss_abs_diff,
            result.node_error.max_abs,
            result.cluster_error.max_abs,
            result.node_rho_error.max_abs,
            result.cluster_rho_error.max_abs,
            result.global_rho_error.max_abs,
        )
        .unwrap();
    }
    out
}
