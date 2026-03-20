use std::fmt::Write as _;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Int, Tensor, TensorData};
use burn_cubecl::cubecl::Runtime;
use burn_dragon_kernel::api::attention::try_fused_dense_causal_attention_wgpu;
use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
use serde::Serialize;

type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type Device = <Backend as BackendTrait>::Device;

#[derive(Clone, Copy, Serialize)]
struct BenchCase {
    name: &'static str,
    batch: usize,
    heads: usize,
    value_heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
}

#[derive(Clone, Serialize)]
struct CaseResult {
    case: BenchCase,
    warmup: usize,
    repetitions: usize,
    reference_ms: f64,
    fused_ms: f64,
    speedup_x: f64,
    reference_tokens_per_s: f64,
    fused_tokens_per_s: f64,
    context_max_abs: f64,
    context_rmse: f64,
    reserved_before: u64,
    reserved_after: u64,
}

#[derive(Clone, Serialize)]
struct Report {
    benchmark: &'static str,
    adapter: String,
    cases: Vec<CaseResult>,
}

const CASES: &[BenchCase] = &[
    BenchCase {
        name: "practical_t64_l32768_e256",
        batch: 1,
        heads: 1,
        value_heads: 1,
        time: 64,
        latent: 32768,
        embd: 256,
    },
    BenchCase {
        name: "longer_t128_l32768_e256",
        batch: 1,
        heads: 1,
        value_heads: 1,
        time: 128,
        latent: 32768,
        embd: 256,
    },
    BenchCase {
        name: "wider_t64_l65536_e256",
        batch: 1,
        heads: 1,
        value_heads: 1,
        time: 64,
        latent: 65536,
        embd: 256,
    },
];

fn init_runtime(device: &Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn main() {
    let output_dir = parse_output_dir();
    let device = Device::default();
    init_runtime(&device);
    <Backend as BackendTrait>::seed(&device, 2027);
    let (adapter, adapter_type) = adapter_info();
    assert!(
        !matches!(adapter_type, wgpu::DeviceType::Cpu),
        "wgpu dense causal benchmark selected a CPU adapter; refusing to continue (adapter: {adapter})"
    );

    let mut cases = Vec::with_capacity(CASES.len());
    for case in CASES {
        cases.push(run_case(*case, &device));
    }

    let report = Report {
        benchmark: "burn_dragon_kernel dense causal attention sweep",
        adapter,
        cases,
    };
    let markdown = format_markdown(&report);
    println!("{markdown}");

    if let Some(output_dir) = output_dir {
        fs::create_dir_all(&output_dir).expect("create output dir");
        fs::write(
            output_dir.join("dense_causal_attention_bench.md"),
            &markdown,
        )
        .expect("write markdown");
        fs::write(
            output_dir.join("dense_causal_attention_bench.json"),
            serde_json::to_vec_pretty(&report).expect("serialize report"),
        )
        .expect("write json");
    }
}

fn parse_output_dir() -> Option<PathBuf> {
    let mut args = std::env::args().skip(1);
    let mut output_dir = None;
    while let Some(arg) = args.next() {
        if arg == "--output-dir" {
            output_dir = args.next().map(PathBuf::from);
        } else {
            panic!("unknown arg: {arg}");
        }
    }
    output_dir
}

fn adapter_info() -> (String, wgpu::DeviceType) {
    let instance = wgpu::Instance::default();
    let adapter =
        pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
            .expect("request adapter");
    let info = adapter.get_info();
    (info.name, info.device_type)
}

fn run_case(case: BenchCase, device: &Device) -> CaseResult {
    let warmup = 2;
    let repetitions = 5;
    let query = Tensor::<Backend, 4>::random(
        [case.batch, case.heads, case.time, case.latent],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let value = Tensor::<Backend, 4>::random(
        [case.batch, case.value_heads, case.time, case.embd],
        Distribution::Normal(0.0, 1.0),
        device,
    );
    let decay = Tensor::<Backend, 1>::from_data(
        TensorData::new(vec![0.97_f32; case.heads], [case.heads]),
        device,
    );

    for _ in 0..warmup {
        let _ = dense_causal_attention_reference(query.clone(), value.clone(), decay.clone());
        let _ = try_fused_dense_causal_attention_wgpu::<Backend>(&query, &value, &decay)
            .expect("fused dense causal attention");
        let _ = Backend::sync(device);
    }

    let reserved_before = <WgpuRuntime as Runtime>::client(device)
        .memory_usage()
        .bytes_reserved;
    let reference_ms = measure_ms(repetitions, || {
        let _ = dense_causal_attention_reference(query.clone(), value.clone(), decay.clone());
        let _ = Backend::sync(device);
    });
    let fused_output = try_fused_dense_causal_attention_wgpu::<Backend>(&query, &value, &decay)
        .expect("fused dense causal attention");
    let reference_output =
        dense_causal_attention_reference(query.clone(), value.clone(), decay.clone());
    let fused_ms = measure_ms(repetitions, || {
        let _ = try_fused_dense_causal_attention_wgpu::<Backend>(&query, &value, &decay)
            .expect("fused dense causal attention");
        let _ = Backend::sync(device);
    });
    let reserved_after = <WgpuRuntime as Runtime>::client(device)
        .memory_usage()
        .bytes_reserved;

    let (context_max_abs, context_rmse) = diff_metrics(reference_output, fused_output);
    let tokens = (case.batch * case.time) as f64;

    CaseResult {
        case,
        warmup,
        repetitions,
        reference_ms,
        fused_ms,
        speedup_x: reference_ms / fused_ms,
        reference_tokens_per_s: tokens / (reference_ms / 1000.0),
        fused_tokens_per_s: tokens / (fused_ms / 1000.0),
        context_max_abs,
        context_rmse,
        reserved_before,
        reserved_after,
    }
}

fn measure_ms(repetitions: usize, mut f: impl FnMut()) -> f64 {
    let start = Instant::now();
    for _ in 0..repetitions {
        f();
    }
    (start.elapsed().as_secs_f64() * 1000.0) / repetitions as f64
}

fn dense_causal_attention_reference(
    query: Tensor<Backend, 4>,
    value: Tensor<Backend, 4>,
    decay: Tensor<Backend, 1>,
) -> Tensor<Backend, 4> {
    let [batch, heads, time, _latent] = query.shape().dims::<4>();
    let value_dim = value.shape().dims::<4>()[3];
    let value = match value.shape().dims::<4>()[1] {
        1 => value.repeat_dim(1, heads),
        existing if existing == heads => value,
        existing => panic!("value heads {existing} must be 1 or {heads}"),
    };
    let pos_row = Tensor::<Backend, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = Tensor::<Backend, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, 1, time]);
    let diff = (pos_row - pos_col).tril(-1);
    let decay_matrix = decay
        .reshape([1, heads, 1, 1])
        .repeat_dim(2, time)
        .repeat_dim(3, time)
        .powf(diff);
    let scores = query.clone().matmul(query.swap_dims(2, 3)).tril(-1) * decay_matrix;
    scores
        .reshape([batch * heads, time, time])
        .matmul(value.reshape([batch * heads, time, value_dim]))
        .reshape([batch, heads, time, value_dim])
}

fn diff_metrics(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> (f64, f64) {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    let mut max_abs = 0.0_f64;
    let mut sum_sq = 0.0_f64;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = f64::from((a - b).abs());
        max_abs = max_abs.max(diff);
        sum_sq += diff * diff;
    }
    let rmse = if lhs.is_empty() {
        0.0
    } else {
        (sum_sq / lhs.len() as f64).sqrt()
    };
    (max_abs, rmse)
}

fn format_markdown(report: &Report) -> String {
    let mut out = String::new();
    writeln!(&mut out, "# Dense causal attention benchmark").unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "- adapter: {}", report.adapter).unwrap();
    writeln!(&mut out).unwrap();
    writeln!(&mut out, "| case | ref_ms | fused_ms | speedup_x | ref_tok/s | fused_tok/s | ctx_max_abs | ctx_rmse | reserved_before | reserved_after |").unwrap();
    writeln!(
        &mut out,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    .unwrap();
    for case in &report.cases {
        writeln!(
            &mut out,
            "| {} | {:.3} | {:.3} | {:.2} | {:.2} | {:.2} | {:.6} | {:.6} | {} | {} |",
            case.case.name,
            case.reference_ms,
            case.fused_ms,
            case.speedup_x,
            case.reference_tokens_per_s,
            case.fused_tokens_per_s,
            case.context_max_abs,
            case.context_rmse,
            case.reserved_before,
            case.reserved_after,
        )
        .unwrap();
    }
    out
}
