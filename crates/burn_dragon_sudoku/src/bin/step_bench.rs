#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("sudoku_step_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_dragon_sudoku::config::{SudokuModelConfig, SudokuPolicyHead};
    use burn_dragon_sudoku::model::SudokuSaccadeModel;
    use burn_dragon_sudoku::vocab::GRID_LEN;
    use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
    use clap::Parser;
    use serde::Serialize;

    type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type TrainBackend = Autodiff<InnerBackend>;
    type Device = <TrainBackend as BackendTrait>::Device;

    #[derive(Parser, Debug)]
    #[command(name = "sudoku_step_bench")]
    struct Args {
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Copy, Serialize)]
    struct BenchCase {
        name: &'static str,
        batch: usize,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        mlp_internal_dim_multiplier: usize,
        rollout_fast_steps: usize,
    }

    #[derive(Clone, Copy, Serialize)]
    struct ErrorMetrics {
        max_abs: f32,
        mean_abs: f32,
    }

    #[derive(Clone, Serialize)]
    struct CaseResult {
        case: BenchCase,
        warmup: usize,
        iterations: usize,
        baseline_forward_backward_ms: f64,
        fused_forward_backward_ms: f64,
        baseline_cells_per_sec: f64,
        fused_cells_per_sec: f64,
        speedup_x: f64,
        loss_abs_diff: f32,
        logits_error: ErrorMetrics,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        adapter: String,
        warmup: usize,
        iterations: usize,
        cases: Vec<CaseResult>,
    }

    const CASES: &[BenchCase] = &[
        BenchCase {
            name: "tiny_fs1",
            batch: 16,
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            rollout_fast_steps: 1,
        },
        BenchCase {
            name: "tiny_fs4",
            batch: 16,
            n_layer: 2,
            n_embd: 64,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            rollout_fast_steps: 4,
        },
        BenchCase {
            name: "small_fs4",
            batch: 32,
            n_layer: 4,
            n_embd: 128,
            n_head: 8,
            mlp_internal_dim_multiplier: 2,
            rollout_fast_steps: 4,
        },
    ];

    pub fn main() {
        let args = Args::parse();
        let device = Device::default();
        init_runtime(&device);

        let report = Report {
            benchmark: "burn_dragon_sudoku fused core forward+backward benchmark",
            adapter: adapter_info(),
            warmup: args.warmup,
            iterations: args.iterations,
            cases: run_all_cases(&device, &args),
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).expect("serialize sudoku bench report");
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact");
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact");
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

    fn run_all_cases(device: &Device, args: &Args) -> Vec<CaseResult> {
        CASES
            .iter()
            .copied()
            .map(|case| run_case(case, device, args))
            .collect()
    }

    fn run_case(case: BenchCase, device: &Device, args: &Args) -> CaseResult {
        let puzzles = sample_batch(case, device);

        <TrainBackend as BackendTrait>::seed(device, 2_026 + case.rollout_fast_steps as u64);
        let baseline = build_model(case, false, device);
        <TrainBackend as BackendTrait>::seed(device, 2_026 + case.rollout_fast_steps as u64);
        let fused = build_model(case, true, device);

        let (loss_abs_diff, logits_error) =
            parity_snapshot(&baseline, &fused, puzzles.clone(), device);

        for _ in 0..args.warmup {
            let _ = run_forward_backward(&baseline, puzzles.clone(), device);
            let _ = run_forward_backward(&fused, puzzles.clone(), device);
        }

        let baseline_ns = (0..args.iterations)
            .map(|_| run_forward_backward(&baseline, puzzles.clone(), device))
            .collect::<Vec<_>>();
        let fused_ns = (0..args.iterations)
            .map(|_| run_forward_backward(&fused, puzzles.clone(), device))
            .collect::<Vec<_>>();

        let baseline_avg_ns = mean_u128(&baseline_ns);
        let fused_avg_ns = mean_u128(&fused_ns);
        let cells_per_iter = (case.batch * GRID_LEN * case.rollout_fast_steps.max(1)) as f64;

        CaseResult {
            case,
            warmup: args.warmup,
            iterations: args.iterations,
            baseline_forward_backward_ms: ns_to_ms(baseline_avg_ns),
            fused_forward_backward_ms: ns_to_ms(fused_avg_ns),
            baseline_cells_per_sec: cells_per_iter / (baseline_avg_ns / 1e9),
            fused_cells_per_sec: cells_per_iter / (fused_avg_ns / 1e9),
            speedup_x: baseline_avg_ns / fused_avg_ns.max(f64::EPSILON),
            loss_abs_diff,
            logits_error,
        }
    }

    fn build_model(
        case: BenchCase,
        fused: bool,
        device: &Device,
    ) -> SudokuSaccadeModel<TrainBackend> {
        let config = SudokuModelConfig {
            n_layer: case.n_layer,
            n_embd: case.n_embd,
            n_head: case.n_head,
            mlp_internal_dim_multiplier: case.mlp_internal_dim_multiplier,
            summary_tokens: 2,
            policy_heads: case.n_head,
            policy_head: SudokuPolicyHead::Cache,
            policy_mlp_hidden_mult: 2,
            dropout: 0.0,
            fused_kernels: fused,
            relu_threshold: 0.0,
            ..SudokuModelConfig::default()
        };
        let mut core_config = config.to_bdh_config();
        core_config.fused_kernels.enabled = true;
        core_config.fused_kernels.set_wgpu_recurrent_kernel(fused);
        core_config.fused_kernels.set_wgpu_rollout_fused(fused);
        core_config.set_rollout_fast_steps_per_slow_step(case.rollout_fast_steps);
        SudokuSaccadeModel::new_with_bdh_config(&config, core_config, device)
    }

    fn sample_batch(case: BenchCase, device: &Device) -> Tensor<TrainBackend, 2, Int> {
        let len = case.batch * GRID_LEN;
        let puzzles: Vec<i64> = (0..len)
            .map(|idx| {
                let digit = ((idx % 9) + 1) as i64;
                if idx % 4 == 0 { 0 } else { digit }
            })
            .collect();
        Tensor::<TrainBackend, 2, Int>::from_data(
            TensorData::new(puzzles, [case.batch, GRID_LEN]),
            device,
        )
    }

    fn run_forward_backward(
        model: &SudokuSaccadeModel<TrainBackend>,
        puzzles: Tensor<TrainBackend, 2, Int>,
        device: &Device,
    ) -> f64 {
        let _ = <TrainBackend as BackendTrait>::sync(device);
        let start = Instant::now();
        let loss = sudoku_loss(model, puzzles);
        let _ = loss.backward();
        let _ = <TrainBackend as BackendTrait>::sync(device);
        start.elapsed().as_nanos() as f64
    }

    fn parity_snapshot(
        baseline: &SudokuSaccadeModel<TrainBackend>,
        fused: &SudokuSaccadeModel<TrainBackend>,
        puzzles: Tensor<TrainBackend, 2, Int>,
        device: &Device,
    ) -> (f32, ErrorMetrics) {
        let baseline_logits = sudoku_logits(baseline, puzzles.clone());
        let fused_logits = sudoku_logits(fused, puzzles);

        let baseline_loss = sudoku_loss_from_logits(baseline_logits.clone());
        let fused_loss = sudoku_loss_from_logits(fused_logits.clone());
        let _ = <TrainBackend as BackendTrait>::sync(device);

        let baseline_loss = scalar_from_tensor(baseline_loss.inner());
        let fused_loss = scalar_from_tensor(fused_loss.inner());
        let logits_error = diff_metrics(baseline_logits.inner(), fused_logits.inner());

        ((baseline_loss - fused_loss).abs(), logits_error)
    }

    fn sudoku_logits(
        model: &SudokuSaccadeModel<TrainBackend>,
        puzzles: Tensor<TrainBackend, 2, Int>,
    ) -> Tensor<TrainBackend, 3> {
        let (hidden, _) = model.forward_with_hidden(puzzles);
        model.value_logits_from_hidden(hidden)
    }

    fn sudoku_loss(
        model: &SudokuSaccadeModel<TrainBackend>,
        puzzles: Tensor<TrainBackend, 2, Int>,
    ) -> Tensor<TrainBackend, 1> {
        sudoku_loss_from_logits(sudoku_logits(model, puzzles))
    }

    fn sudoku_loss_from_logits(logits: Tensor<TrainBackend, 3>) -> Tensor<TrainBackend, 1> {
        logits.tanh().powf_scalar(2.0).mean()
    }

    fn scalar_from_tensor(tensor: Tensor<InnerBackend, 1>) -> f32 {
        tensor
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tensor vec")[0]
    }

    fn diff_metrics(lhs: Tensor<InnerBackend, 3>, rhs: Tensor<InnerBackend, 3>) -> ErrorMetrics {
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

        let mut max_abs = 0.0_f32;
        let mut sum_abs = 0.0_f64;
        let mut count = 0usize;
        for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
            let diff = (*lhs_value - *rhs_value).abs();
            max_abs = max_abs.max(diff);
            sum_abs += f64::from(diff);
            count += 1;
        }

        ErrorMetrics {
            max_abs,
            mean_abs: (sum_abs / count.max(1) as f64) as f32,
        }
    }

    fn mean_u128(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len().max(1) as f64
    }

    fn ns_to_ms(value: f64) -> f64 {
        value / 1_000_000.0
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# {}", report.benchmark);
        let _ = writeln!(out);
        let _ = writeln!(out, "- adapter: {}", report.adapter);
        let _ = writeln!(out, "- warmup: {}", report.warmup);
        let _ = writeln!(out, "- iterations: {}", report.iterations);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| case | dims | baseline ms | fused ms | speedup x | baseline cells/s | fused cells/s | loss abs diff | logits max abs | logits mean abs |"
        );
        let _ = writeln!(out, "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|");
        for result in &report.cases {
            let _ = writeln!(
                out,
                "| {} | b{} l{} d{} h{} fs{} | {:.3} | {:.3} | {:.2} | {:.0} | {:.0} | {:.4e} | {:.4e} | {:.4e} |",
                result.case.name,
                result.case.batch,
                result.case.n_layer,
                result.case.n_embd,
                result.case.n_head,
                result.case.rollout_fast_steps,
                result.baseline_forward_backward_ms,
                result.fused_forward_backward_ms,
                result.speedup_x,
                result.baseline_cells_per_sec,
                result.fused_cells_per_sec,
                result.loss_abs_diff,
                result.logits_error.max_abs,
                result.logits_error.mean_abs,
            );
        }
        out
    }

    fn write_text_artifact(path: &Path, content: &str, label: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .unwrap_or_else(|err| panic!("failed to create {}: {err}", parent.display()));
        }
        fs::write(path, content)
            .unwrap_or_else(|err| panic!("failed to write {label} {}: {err}", path.display()));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
