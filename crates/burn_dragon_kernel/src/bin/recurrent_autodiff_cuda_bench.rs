#[cfg(not(feature = "cuda"))]
fn main() {
    eprintln!("recurrent_autodiff_cuda_bench requires --features cuda");
    std::process::exit(1);
}

#[cfg(feature = "cuda")]
mod app {
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_cuda::Cuda;
    use burn_dragon_kernel::api::recurrent::try_fused_recurrent_attention_wgpu;
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
        rho_grad_error: ErrorMetrics,
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
        rho: Tensor<AutodiffBackend, 4>,
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
            benchmark: "burn_dragon_kernel recurrent autodiff cuda bench",
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

        let (query_grad_error, value_grad_error, rho_grad_error, decay_grad_error) =
            gradient_error_case(&inputs);

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
            rho_grad_error,
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
            let rho = inputs.rho.clone().require_grad();
            let decay = inputs.decay.clone().require_grad();
            let weights = inputs.weights.clone();

            let start_forward = Instant::now();
            let context = if fused {
                try_fused_recurrent_attention_wgpu::<AutodiffBackend>(
                    &query,
                    &value,
                    Some(&rho),
                    Some(&decay),
                )
                .expect("fused recurrent autodiff")
                .context
            } else {
                reference_recurrent(query.clone(), value.clone(), rho.clone(), decay.clone()).0
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

    fn gradient_error_case(
        inputs: &Inputs,
    ) -> (ErrorMetrics, ErrorMetrics, ErrorMetrics, ErrorMetrics) {
        let fused_query = inputs.query.clone().require_grad();
        let fused_value = inputs.value.clone().require_grad();
        let fused_rho = inputs.rho.clone().require_grad();
        let fused_decay = inputs.decay.clone().require_grad();
        let fused_context = try_fused_recurrent_attention_wgpu::<AutodiffBackend>(
            &fused_query,
            &fused_value,
            Some(&fused_rho),
            Some(&fused_decay),
        )
        .expect("fused recurrent autodiff")
        .context;
        let fused_grads = (fused_context * inputs.weights.clone()).sum().backward();

        let reference_query = inputs.query.clone().require_grad();
        let reference_value = inputs.value.clone().require_grad();
        let reference_rho = inputs.rho.clone().require_grad();
        let reference_decay = inputs.decay.clone().require_grad();
        let reference_context = reference_recurrent(
            reference_query.clone(),
            reference_value.clone(),
            reference_rho.clone(),
            reference_decay.clone(),
        )
        .0;
        let reference_grads = (reference_context * inputs.weights.clone())
            .sum()
            .backward();

        let query_grad_error = compare_tensors(
            fused_query.grad(&fused_grads).expect("fused query grad"),
            reference_query
                .grad(&reference_grads)
                .expect("reference query grad"),
        );
        let value_grad_error = compare_tensors(
            fused_value.grad(&fused_grads).expect("fused value grad"),
            reference_value
                .grad(&reference_grads)
                .expect("reference value grad"),
        );
        let rho_grad_error = compare_tensors(
            fused_rho.grad(&fused_grads).expect("fused rho grad"),
            reference_rho
                .grad(&reference_grads)
                .expect("reference rho grad"),
        );
        let decay_grad_error = compare_tensors(
            fused_decay.grad(&fused_grads).expect("fused decay grad"),
            reference_decay
                .grad(&reference_grads)
                .expect("reference decay grad"),
        );

        (
            query_grad_error,
            value_grad_error,
            rho_grad_error,
            decay_grad_error,
        )
    }

    fn sample_inputs(case: &BenchCase, device: &Device) -> Inputs {
        let query_total = case.batch * case.heads * case.time * case.latent;
        let value_total = case.batch * case.value_heads * case.time * case.embd;
        let rho_total = case.batch * case.heads * case.latent * case.embd;
        let weight_total = case.batch * case.heads * case.time * case.embd;

        Inputs {
            query: Tensor::<AutodiffBackend, 4>::from_data(
                TensorData::new(
                    (0..query_total)
                        .map(|idx| (idx as f32) * 0.0007 - 0.2)
                        .collect::<Vec<_>>(),
                    [case.batch, case.heads, case.time, case.latent],
                ),
                device,
            ),
            value: Tensor::<AutodiffBackend, 4>::from_data(
                TensorData::new(
                    (0..value_total)
                        .map(|idx| (idx as f32) * 0.0009 - 0.15)
                        .collect::<Vec<_>>(),
                    [case.batch, case.value_heads, case.time, case.embd],
                ),
                device,
            ),
            rho: Tensor::<AutodiffBackend, 4>::from_data(
                TensorData::new(
                    (0..rho_total)
                        .map(|idx| (idx as f32) * 0.0003 - 0.05)
                        .collect::<Vec<_>>(),
                    [case.batch, case.heads, case.latent, case.embd],
                ),
                device,
            ),
            decay: Tensor::<AutodiffBackend, 1>::from_data(
                TensorData::new(
                    (0..case.heads)
                        .map(|idx| 0.95f32 - idx as f32 * 0.01)
                        .collect::<Vec<_>>(),
                    [case.heads],
                ),
                device,
            ),
            weights: Tensor::<AutodiffBackend, 4>::from_data(
                TensorData::new(
                    (0..weight_total)
                        .map(|idx| (idx as f32) * 0.0005 - 0.1)
                        .collect::<Vec<_>>(),
                    [case.batch, case.heads, case.time, case.embd],
                ),
                device,
            ),
        }
    }

    fn reference_recurrent(
        query: Tensor<AutodiffBackend, 4>,
        value: Tensor<AutodiffBackend, 4>,
        rho: Tensor<AutodiffBackend, 4>,
        decay: Tensor<AutodiffBackend, 1>,
    ) -> (Tensor<AutodiffBackend, 4>, Tensor<AutodiffBackend, 4>) {
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

    fn compare_tensors<B: BackendTrait, const D: usize>(
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
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut sum_sq = 0.0f64;
        for (lhs, rhs) in lhs.iter().zip(rhs.iter()) {
            let diff = (*lhs as f64 - *rhs as f64).abs();
            let rel = diff / ((*rhs as f64).abs().max(1.0e-6));
            max_abs = max_abs.max(diff);
            max_rel = max_rel.max(rel);
            sum_abs += diff;
            sum_sq += diff * diff;
        }
        let count = lhs.len().max(1) as f64;
        ErrorMetrics {
            max_abs,
            mean_abs: sum_abs / count,
            rmse: (sum_sq / count).sqrt(),
            max_rel,
        }
    }

    fn write_artifacts(root: &Path, json: &str) {
        std::fs::create_dir_all(root).expect("create output dir");
        std::fs::write(root.join("report.json"), json).expect("write report");
    }
}

#[cfg(feature = "cuda")]
fn main() {
    app::main();
}
