#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("recurrent_cuda_bench requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
mod app {
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Tensor, TensorData};
    use burn_cubecl::cubecl::Runtime;
    use burn_cubecl::cubecl::cuda::CudaRuntime;
    use burn_cuda::Cuda;
    use burn_dragon_kernel::api::recurrent::{
        supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
    };
    use serde::Serialize;

    type Backend = Cuda<f32, i32>;
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

    #[derive(Clone, Copy, Serialize)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
    }

    #[derive(Clone, Copy, Serialize)]
    struct ErrorMetrics {
        max_abs: f64,
        mean_abs: f64,
        rmse: f64,
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
        context_error: ErrorMetrics,
        rho_error: ErrorMetrics,
        memory_before: MemorySnapshot,
        memory_after: MemorySnapshot,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        backend: &'static str,
        warmup: usize,
        repetitions: usize,
        cases: Vec<CaseResult>,
    }

    const CASES: &[BenchCase] = &[
        BenchCase {
            name: "cuda_b1_h1_t128_l32768_e256",
            batch: 1,
            heads: 1,
            value_heads: 1,
            time: 128,
            latent: 32768,
            embd: 256,
        },
        BenchCase {
            name: "cuda_b1_h1_t128_l65536_e256",
            batch: 1,
            heads: 1,
            value_heads: 1,
            time: 128,
            latent: 65536,
            embd: 256,
        },
    ];

    pub fn main() {
        let output_dir = std::env::args().skip(1).find_map(|arg| {
            arg.strip_prefix("--output-dir=")
                .map(PathBuf::from)
                .or_else(|| {
                    (arg == "--output-dir").then_some(PathBuf::from(
                        "artifacts/language/recurrent_cuda_bench/latest",
                    ))
                })
        });

        if !supports_recurrent_backend::<Backend>() {
            eprintln!(
                "cuda runtime support is currently disabled for the recurrent fused kernel because the kernel source is WGSL-only; port it to a portable CubeCL kernel or add a CUDA-specific source first"
            );
            std::process::exit(2);
        }
        let device = Device::default();
        <Backend as BackendTrait>::seed(&device, 20260319);

        let warmup = 1usize;
        let repetitions = 3usize;
        let cases = CASES
            .iter()
            .map(|case| run_case(case, &device, warmup, repetitions))
            .collect::<Vec<_>>();

        let report = Report {
            benchmark: "burn_dragon_kernel recurrent cuda bench",
            backend: "cuda",
            warmup,
            repetitions,
            cases,
        };

        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        println!("{json}");

        if let Some(root) = output_dir.as_ref() {
            write_artifacts(root, &report, &json);
        }
    }

    fn run_case(
        case: &BenchCase,
        device: &Device,
        warmup: usize,
        repetitions: usize,
    ) -> CaseResult {
        let query = sample_query(case, device);
        let value = sample_value(case, device);
        let rho =
            Tensor::<Backend, 4>::zeros([case.batch, case.heads, case.latent, case.embd], device);
        let decay = sample_decay(case.heads, device);

        let memory_before = memory_snapshot(device);

        let (reference_ms, reference_context, reference_rho) = timed_run(
            warmup,
            repetitions,
            || reference_recurrent(query.clone(), value.clone(), rho.clone(), decay.clone()),
            device,
        );
        let (fused_ms, fused_context, fused_rho) = timed_run(
            warmup,
            repetitions,
            || {
                let output = try_fused_recurrent_attention_wgpu::<Backend>(
                    &query,
                    &value,
                    Some(&rho),
                    Some(&decay),
                )
                .expect("fused recurrent output");
                (output.context, output.rho)
            },
            device,
        );

        let memory_after = memory_snapshot(device);
        let tokens = (case.batch * case.time) as f64;

        CaseResult {
            case: *case,
            warmup,
            repetitions,
            reference_ms,
            fused_ms,
            speedup_x: reference_ms / fused_ms,
            reference_tokens_per_s: tokens / (reference_ms / 1_000.0),
            fused_tokens_per_s: tokens / (fused_ms / 1_000.0),
            context_error: compare_tensors(reference_context, fused_context),
            rho_error: compare_tensors(reference_rho, fused_rho),
            memory_before,
            memory_after,
        }
    }

    fn sample_query(case: &BenchCase, device: &Device) -> Tensor<Backend, 4> {
        let total = case.batch * case.heads * case.time * case.latent;
        let values = (0..total)
            .map(|idx| (((idx % 97) as f32) / 97.0) + 0.01)
            .collect::<Vec<_>>();
        Tensor::<Backend, 4>::from_data(
            TensorData::new(values, [case.batch, case.heads, case.time, case.latent]),
            device,
        )
    }

    fn sample_value(case: &BenchCase, device: &Device) -> Tensor<Backend, 4> {
        let total = case.batch * case.value_heads * case.time * case.embd;
        let values = (0..total)
            .map(|idx| (((idx % 53) as f32) / 53.0) + 0.02)
            .collect::<Vec<_>>();
        Tensor::<Backend, 4>::from_data(
            TensorData::new(values, [case.batch, case.value_heads, case.time, case.embd]),
            device,
        )
    }

    fn sample_decay(heads: usize, device: &Device) -> Tensor<Backend, 1> {
        let values = (0..heads)
            .map(|idx| 0.99f32 - (idx as f32 * 0.01))
            .collect::<Vec<_>>();
        Tensor::<Backend, 1>::from_data(TensorData::new(values, [heads]), device)
    }

    fn reference_recurrent(
        query: Tensor<Backend, 4>,
        value: Tensor<Backend, 4>,
        rho: Tensor<Backend, 4>,
        decay: Tensor<Backend, 1>,
    ) -> (Tensor<Backend, 4>, Tensor<Backend, 4>) {
        let [batch, heads, time, _latent] = query.shape().dims::<4>();
        let value_heads = value.shape().dims::<4>()[1];
        let embd = value.shape().dims::<4>()[3];

        let decay = decay.reshape([1, heads, 1, 1]);
        let value = if value_heads == 1 {
            value.repeat_dim(1, heads)
        } else {
            value
        };

        let mut state = rho;
        let mut outputs = Vec::with_capacity(time);
        for t in 0..time {
            let q_t = query.clone().slice_dim(2, t..t + 1);
            let v_t = value.clone().slice_dim(2, t..t + 1);
            let q_latent = q_t.swap_dims(2, 3);
            let context = (state.clone() * q_latent.clone())
                .sum_dim(2)
                .reshape([batch, heads, 1, embd]);
            outputs.push(context);
            state = (state + q_latent * v_t) * decay.clone();
        }
        (Tensor::cat(outputs, 2), state)
    }

    fn timed_run<F>(
        warmup: usize,
        repetitions: usize,
        mut run: F,
        device: &Device,
    ) -> (f64, Tensor<Backend, 4>, Tensor<Backend, 4>)
    where
        F: FnMut() -> (Tensor<Backend, 4>, Tensor<Backend, 4>),
    {
        let mut last = run();
        let _ = <Backend as BackendTrait>::sync(device);
        for _ in 0..warmup {
            last = run();
            let _ = <Backend as BackendTrait>::sync(device);
        }
        let start = Instant::now();
        for _ in 0..repetitions {
            last = run();
            let _ = <Backend as BackendTrait>::sync(device);
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1_000.0 / repetitions as f64;
        (elapsed_ms, last.0, last.1)
    }

    fn compare_tensors(lhs: Tensor<Backend, 4>, rhs: Tensor<Backend, 4>) -> ErrorMetrics {
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

        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            let diff = (*a as f64 - *b as f64).abs();
            max_abs = max_abs.max(diff);
            sum_abs += diff;
            sum_sq += diff * diff;
        }
        let len = lhs.len().max(1) as f64;
        ErrorMetrics {
            max_abs,
            mean_abs: sum_abs / len,
            rmse: (sum_sq / len).sqrt(),
        }
    }

    fn memory_snapshot(device: &Device) -> MemorySnapshot {
        let usage = <CudaRuntime as Runtime>::client(device).memory_usage();
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }

    fn write_artifacts(root: &Path, report: &Report, json: &str) {
        fs::create_dir_all(root).expect("create output dir");
        fs::write(root.join("report.json"), json).expect("write report json");
        fs::write(root.join("summary.md"), format_markdown(report)).expect("write summary");
        fs::write(root.join("summary.csv"), format_csv(report)).expect("write csv");
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        out.push_str("# CUDA Recurrent Bench\n\n");
        out.push_str(&format!(
            "- backend: `{}`\n- warmup: `{}`\n- repetitions: `{}`\n\n",
            report.backend, report.warmup, report.repetitions
        ));
        out.push_str("| case | reference ms | fused ms | speedup | ref tok/s | fused tok/s | ctx max abs | rho max abs |\n");
        out.push_str("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n");
        for case in &report.cases {
            out.push_str(&format!(
                "| {} | {:.3} | {:.3} | {:.2}x | {:.2} | {:.2} | {:.6} | {:.6} |\n",
                case.case.name,
                case.reference_ms,
                case.fused_ms,
                case.speedup_x,
                case.reference_tokens_per_s,
                case.fused_tokens_per_s,
                case.context_error.max_abs,
                case.rho_error.max_abs,
            ));
        }
        out
    }

    fn format_csv(report: &Report) -> String {
        let mut out = String::from(
            "case,reference_ms,fused_ms,speedup_x,reference_tokens_per_s,fused_tokens_per_s,context_max_abs,rho_max_abs,reserved_before,reserved_after,in_use_before,in_use_after\n",
        );
        for case in &report.cases {
            out.push_str(&format!(
                "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.9},{:.9},{},{},{},{}\n",
                case.case.name,
                case.reference_ms,
                case.fused_ms,
                case.speedup_x,
                case.reference_tokens_per_s,
                case.fused_tokens_per_s,
                case.context_error.max_abs,
                case.rho_error.max_abs,
                case.memory_before.reserved,
                case.memory_after.reserved,
                case.memory_before.in_use,
                case.memory_after.in_use,
            ));
        }
        out
    }
}

#[cfg(feature = "cuda")]
fn main() {
    app::main();
}
