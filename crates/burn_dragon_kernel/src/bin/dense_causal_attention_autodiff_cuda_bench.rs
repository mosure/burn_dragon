#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("dense_causal_attention_autodiff_cuda_bench requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
mod app {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_cuda::Cuda;
    use burn_dragon_kernel::api::attention::try_fused_dense_causal_attention_wgpu;
    use serde::Serialize;

    type Backend = Cuda<f32, i32>;
    type AutodiffBackend = Autodiff<Backend>;
    type Device = <AutodiffBackend as BackendTrait>::Device;

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
    struct ErrorMetrics {
        max_abs: f64,
        mean_abs: f64,
        rmse: f64,
        max_rel: f64,
    }

    #[derive(Clone, Copy, Serialize)]
    struct TimedMetrics {
        forward_ms: f64,
        backward_ms: f64,
        tokens_per_s: f64,
    }

    #[derive(Clone, Serialize)]
    struct CaseResult {
        case: BenchCase,
        warmup: usize,
        repetitions: usize,
        reference: TimedMetrics,
        fused: TimedMetrics,
        forward_speedup_x: f64,
        backward_speedup_x: f64,
        query_grad_error: ErrorMetrics,
        value_grad_error: ErrorMetrics,
        decay_grad_error: ErrorMetrics,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        backend: &'static str,
        profile: &'static str,
        warmup: usize,
        repetitions: usize,
        cases: Vec<CaseResult>,
    }

    #[derive(Clone)]
    struct Inputs {
        query: Tensor<AutodiffBackend, 4>,
        value: Tensor<AutodiffBackend, 4>,
        decay: Tensor<AutodiffBackend, 1>,
        weights: Tensor<AutodiffBackend, 4>,
    }

    const COMPACT_CASES: &[BenchCase] = &[BenchCase {
        name: "cuda_b2_h8_t64_l64_e128",
        batch: 2,
        heads: 8,
        value_heads: 1,
        time: 64,
        latent: 64,
        embd: 128,
    }];

    const FULL_CASES: &[BenchCase] = &[
        BenchCase {
            name: "cuda_b2_h8_t64_l64_e128",
            batch: 2,
            heads: 8,
            value_heads: 1,
            time: 64,
            latent: 64,
            embd: 128,
        },
        BenchCase {
            name: "cuda_b2_h8_t128_l64_e128",
            batch: 2,
            heads: 8,
            value_heads: 1,
            time: 128,
            latent: 64,
            embd: 128,
        },
    ];

    pub fn main() {
        let device = Device::default();
        <AutodiffBackend as BackendTrait>::seed(&device, 20260329);

        let profile =
            std::env::var("BURN_DRAGON_BENCH_PROFILE").unwrap_or_else(|_| "compact".into());
        let (cases, warmup, repetitions, profile_name) = if profile.eq_ignore_ascii_case("full") {
            (FULL_CASES, 2usize, 5usize, "full")
        } else {
            (COMPACT_CASES, 1usize, 3usize, "compact")
        };

        let output_dir = std::env::args()
            .skip(1)
            .find_map(|arg| arg.strip_prefix("--output-dir=").map(PathBuf::from));

        let results = cases
            .iter()
            .map(|case| run_case(case, &device, warmup, repetitions))
            .collect::<Vec<_>>();

        let report = Report {
            benchmark: "burn_dragon_kernel dense causal attention autodiff cuda bench",
            backend: "cuda",
            profile: profile_name,
            warmup,
            repetitions,
            cases: results,
        };
        let json = serde_json::to_string_pretty(&report).expect("serialize report");
        println!("{json}");

        if let Some(root) = output_dir.as_ref() {
            write_artifacts(root, &json);
        }
    }

    fn run_case(
        case: &BenchCase,
        device: &Device,
        warmup: usize,
        repetitions: usize,
    ) -> CaseResult {
        let inputs = sample_inputs(case, device);

        let reference = timed_run(&inputs, device, warmup, repetitions, false);
        let fused = timed_run(&inputs, device, warmup, repetitions, true);
        let (query_grad_error, value_grad_error, decay_grad_error) = gradient_error_case(&inputs);

        CaseResult {
            case: *case,
            warmup,
            repetitions,
            forward_speedup_x: reference.forward_ms / fused.forward_ms,
            backward_speedup_x: reference.backward_ms / fused.backward_ms,
            reference,
            fused,
            query_grad_error,
            value_grad_error,
            decay_grad_error,
        }
    }

    fn timed_run(
        inputs: &Inputs,
        device: &Device,
        warmup: usize,
        repetitions: usize,
        fused: bool,
    ) -> TimedMetrics {
        let mut forward_total_ms = 0.0;
        let mut backward_total_ms = 0.0;
        let total_iters = warmup + repetitions;
        let tokens =
            (inputs.query.shape().dims::<4>()[0] * inputs.query.shape().dims::<4>()[2]) as f64;

        for step in 0..total_iters {
            let query = inputs.query.clone().require_grad();
            let value = inputs.value.clone().require_grad();
            let decay = inputs.decay.clone().require_grad();
            let weights = inputs.weights.clone();

            let start_forward = Instant::now();
            let context = if fused {
                try_fused_dense_causal_attention_wgpu::<AutodiffBackend>(&query, &value, &decay)
                    .expect("fused dense causal autodiff")
            } else {
                dense_causal_attention_reference(query.clone(), value.clone(), decay.clone())
            };
            let loss = (context * weights).sum();
            let _ = AutodiffBackend::sync(device);
            let forward_ms = start_forward.elapsed().as_secs_f64() * 1_000.0;

            let start_backward = Instant::now();
            let _grads = loss.backward();
            let _ = AutodiffBackend::sync(device);
            let backward_ms = start_backward.elapsed().as_secs_f64() * 1_000.0;

            if step >= warmup {
                forward_total_ms += forward_ms;
                backward_total_ms += backward_ms;
            }
        }

        let repetitions_f64 = repetitions as f64;
        let total_ms = forward_total_ms + backward_total_ms;
        TimedMetrics {
            forward_ms: forward_total_ms / repetitions_f64,
            backward_ms: backward_total_ms / repetitions_f64,
            tokens_per_s: repetitions_f64 * tokens / (total_ms / 1_000.0),
        }
    }

    fn gradient_error_case(inputs: &Inputs) -> (ErrorMetrics, ErrorMetrics, ErrorMetrics) {
        let fused_query = inputs.query.clone().require_grad();
        let fused_value = inputs.value.clone().require_grad();
        let fused_decay = inputs.decay.clone().require_grad();
        let fused_context = try_fused_dense_causal_attention_wgpu::<AutodiffBackend>(
            &fused_query,
            &fused_value,
            &fused_decay,
        )
        .expect("fused dense causal autodiff");
        let fused_grads = (fused_context * inputs.weights.clone()).sum().backward();

        let reference_query = inputs.query.clone().require_grad();
        let reference_value = inputs.value.clone().require_grad();
        let reference_decay = inputs.decay.clone().require_grad();
        let reference_context = dense_causal_attention_reference(
            reference_query.clone(),
            reference_value.clone(),
            reference_decay.clone(),
        );
        let reference_grads = (reference_context * inputs.weights.clone())
            .sum()
            .backward();

        let query_grad_error = diff_metrics(
            fused_query.grad(&fused_grads).expect("fused query grad"),
            reference_query
                .grad(&reference_grads)
                .expect("reference query grad"),
        );
        let value_grad_error = diff_metrics(
            fused_value.grad(&fused_grads).expect("fused value grad"),
            reference_value
                .grad(&reference_grads)
                .expect("reference value grad"),
        );
        let decay_grad_error = diff_metrics(
            fused_decay.grad(&fused_grads).expect("fused decay grad"),
            reference_decay
                .grad(&reference_grads)
                .expect("reference decay grad"),
        );

        (query_grad_error, value_grad_error, decay_grad_error)
    }

    fn sample_inputs(case: &BenchCase, device: &Device) -> Inputs {
        let query = Tensor::<AutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..case.batch * case.heads * case.time * case.latent)
                    .map(|i| ((i % 1024) as f32) * 0.0005 - 0.25)
                    .collect(),
                [case.batch, case.heads, case.time, case.latent],
            ),
            device,
        );
        let value = Tensor::<AutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..case.batch * case.value_heads * case.time * case.embd)
                    .map(|i| ((i % 2048) as f32) * 0.00025 - 0.25)
                    .collect(),
                [case.batch, case.value_heads, case.time, case.embd],
            ),
            device,
        );
        let decay = Tensor::<AutodiffBackend, 1>::from_data(
            TensorData::new(vec![0.97_f32; case.heads], [case.heads]),
            device,
        );
        let weights = Tensor::<AutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..case.batch * case.heads * case.time * case.embd)
                    .map(|i| ((i % 1536) as f32) * 0.0002 - 0.15)
                    .collect(),
                [case.batch, case.heads, case.time, case.embd],
            ),
            device,
        );
        Inputs {
            query,
            value,
            decay,
            weights,
        }
    }

    fn dense_causal_attention_reference(
        query: Tensor<AutodiffBackend, 4>,
        value: Tensor<AutodiffBackend, 4>,
        decay: Tensor<AutodiffBackend, 1>,
    ) -> Tensor<AutodiffBackend, 4> {
        let [batch, heads, time, _latent] = query.shape().dims::<4>();
        let value_dim = value.shape().dims::<4>()[3];
        let value = match value.shape().dims::<4>()[1] {
            1 => value.repeat_dim(1, heads),
            existing if existing == heads => value,
            existing => panic!("value heads {existing} must be 1 or {heads}"),
        };
        let pos_row = Tensor::<AutodiffBackend, 1, Int>::arange(0..time as i64, &query.device())
            .float()
            .reshape([1, 1, time, 1]);
        let pos_col = Tensor::<AutodiffBackend, 1, Int>::arange(0..time as i64, &query.device())
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

    fn diff_metrics<B: BackendTrait, const D: usize>(
        lhs: Tensor<B, D>,
        rhs: Tensor<B, D>,
    ) -> ErrorMetrics {
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
        let mut sum_abs = 0.0_f64;
        let mut sum_sq = 0.0_f64;
        let mut max_rel = 0.0_f64;
        let mut finite_count = 0usize;
        for (a, b) in lhs.iter().zip(rhs.iter()) {
            if !a.is_finite() || !b.is_finite() {
                continue;
            }
            let diff = f64::from((a - b).abs());
            if !diff.is_finite() {
                continue;
            }
            let denom = f64::from(b.abs()).max(1.0e-12);
            max_abs = max_abs.max(diff);
            sum_abs += diff;
            sum_sq += diff * diff;
            max_rel = max_rel.max(diff / denom);
            finite_count += 1;
        }
        let len = finite_count.max(1) as f64;
        ErrorMetrics {
            max_abs,
            mean_abs: sum_abs / len,
            rmse: (sum_sq / len).sqrt(),
            max_rel,
        }
    }

    fn write_artifacts(root: &Path, json: &str) {
        std::fs::create_dir_all(root).expect("create output dir");
        std::fs::write(
            root.join("dense_causal_attention_autodiff_cuda_bench.json"),
            json,
        )
        .expect("write json");
    }
}

#[cfg(feature = "cuda")]
fn main() {
    app::main();
}
