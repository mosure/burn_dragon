#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("language_rollout_probe requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result};
    use burn::module::Module;
    use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_dataset;
    use burn_dragon::language::{
        BDH, TrainingConfig, apply_wgpu_fused_core_override, build_model_config,
        language_model_loss, load_tokenizer_for_checkpoint, load_training_config_for_checkpoint,
    };
    use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
    use clap::Parser;
    use serde::Serialize;

    type Backend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type Device = <Backend as BackendTrait>::Device;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long, value_delimiter = ',', num_args = 1.., default_values_t = [1usize, 2, 4, 8, 16])]
        fast_steps: Vec<usize>,
        #[arg(long, default_value_t = 8)]
        eval_batches: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        block_size: Option<usize>,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 3)]
        iterations: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Serialize)]
    struct ProbeCase {
        fast_steps: usize,
        avg_loss: f64,
        perplexity: f64,
        forward_ms: f64,
        tokens_per_sec: f64,
        loss_gain_vs_step1: f64,
        gain_per_extra_ms: Option<f64>,
        latency_scale_vs_step1: f64,
        latency_alpha_vs_step1: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct Summary {
        best_loss_fast_steps: usize,
        best_loss: f64,
        best_gain_vs_step1: f64,
        best_efficiency_fast_steps: Option<usize>,
        best_efficiency_gain_per_extra_ms: Option<f64>,
        latency_alpha_fit_all_steps: Option<f64>,
        latency_ms_per_extra_fast_step_fit: Option<f64>,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        config: Vec<PathBuf>,
        checkpoint: PathBuf,
        batch_size: usize,
        block_size: usize,
        eval_batches: usize,
        warmup: usize,
        iterations: usize,
        cases: Vec<ProbeCase>,
        summary: Summary,
    }

    struct EvalBatch<B: BackendTrait> {
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        token_count: usize,
    }

    pub fn main() {
        let args = Args::parse();
        let device = Device::default();
        init_runtime(&device);

        let config =
            load_training_config_for_checkpoint(&args.config, Some(&args.checkpoint), "wgpu")
                .unwrap_or_else(|err| {
                    panic!("failed to load language config/checkpoint metadata: {err}")
                });
        let dataset = prepare_dataset(&config.dataset, &config.training)
            .unwrap_or_else(|err| panic!("failed to prepare dataset: {err}"));
        let tokenizer = load_tokenizer_for_checkpoint(&args.config, Some(&args.checkpoint), "wgpu")
            .unwrap_or_else(|err| panic!("failed to load tokenizer for checkpoint: {err}"));

        let batch_size = args.batch_size.unwrap_or(config.training.batch_size).max(1);
        let block_size = args
            .block_size
            .unwrap_or(
                config
                    .model
                    .block_size
                    .unwrap_or(config.training.block_size),
            )
            .max(1);
        let eval_batches = build_eval_batches::<Backend>(
            dataset.as_ref(),
            DatasetSplit::Val,
            batch_size,
            block_size,
            args.eval_batches.max(1),
            &device,
        )
        .unwrap_or_else(|err| panic!("failed to build eval batches: {err}"));

        let mut fast_steps = args
            .fast_steps
            .iter()
            .copied()
            .map(|value| value.max(1))
            .collect::<Vec<_>>();
        fast_steps.push(1);
        fast_steps.sort_unstable();
        fast_steps.dedup();

        let mut cases = Vec::with_capacity(fast_steps.len());
        for fast in fast_steps {
            let model = load_model_for_fast_steps(
                &config,
                &args.checkpoint,
                tokenizer.as_ref(),
                fast,
                &device,
            )
            .unwrap_or_else(|err| panic!("failed to load model for fast_steps={fast}: {err}"));
            cases.push(run_case(&model, &eval_batches, fast, &args));
        }

        annotate_cases(&mut cases);
        let summary = summarize(&cases);
        let report = Report {
            benchmark: "burn_dragon language rollout quality probe",
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            batch_size,
            block_size,
            eval_batches: eval_batches.len(),
            warmup: args.warmup,
            iterations: args.iterations.max(1),
            cases,
            summary,
        };

        let markdown = format_markdown(&report);
        let json =
            serde_json::to_string_pretty(&report).expect("serialize language rollout report");
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

    fn build_eval_batches<B: BackendTrait>(
        dataset: &Dataset,
        split: DatasetSplit,
        batch_size: usize,
        block_size: usize,
        num_batches: usize,
        device: &B::Device,
    ) -> Result<Vec<EvalBatch<B>>> {
        let (offset, span) = dataset.split_offset_and_span(split);
        let tokens = dataset.tokens();
        let required = block_size.checked_add(1).context("block size overflow")?;
        if span <= required {
            anyhow::bail!("validation split too small for block_size={block_size}");
        }

        let max_start = span - required;
        let total_sequences = num_batches.saturating_mul(batch_size).max(1);
        let stride = (max_start / total_sequences.max(1)).max(1);
        let mut cursor = 0usize;
        let mut batches = Vec::with_capacity(num_batches);

        for _ in 0..num_batches {
            let mut inputs = vec![0i64; batch_size * block_size];
            let mut targets = vec![0i64; batch_size * block_size];
            for batch_idx in 0..batch_size {
                let start = offset + cursor.min(max_start);
                cursor = cursor.saturating_add(stride);
                for t in 0..block_size {
                    let idx = batch_idx * block_size + t;
                    inputs[idx] = tokens[start + t] as i64;
                    targets[idx] = tokens[start + t + 1] as i64;
                }
            }
            let inputs = Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [batch_size, block_size]),
                device,
            );
            let targets = Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, block_size]),
                device,
            );
            batches.push(EvalBatch {
                inputs,
                targets,
                token_count: batch_size * block_size,
            });
        }

        Ok(batches)
    }

    fn load_model_for_fast_steps(
        config: &TrainingConfig,
        checkpoint: &Path,
        tokenizer: &dyn burn_dragon::language::tokenizer::Tokenizer,
        fast_steps: usize,
        device: &Device,
    ) -> Result<BDH<Backend>> {
        let mut overrides = config.model.clone();
        overrides.rollout_fast_steps_per_slow_step = Some(fast_steps);
        let mut model_config = build_model_config(&overrides, config.training.block_size);
        model_config.vocab_size = tokenizer.len();
        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu",
            config.wgpu.training.fused_core_recurrent,
            config.wgpu.training.fused_core_rollout,
        );
        let mut model = BDH::<Backend>::new(model_config, device);
        let record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load::<<BDH<Backend> as Module<Backend>>::Record>(checkpoint.to_path_buf(), device)
            .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;
        model = model.load_record(record);
        Ok(model)
    }

    fn run_case(
        model: &BDH<Backend>,
        eval_batches: &[EvalBatch<Backend>],
        fast_steps: usize,
        args: &Args,
    ) -> ProbeCase {
        for _ in 0..args.warmup {
            let _ = evaluate_once(model, eval_batches);
        }

        let iterations = args.iterations.max(1);
        let mut losses = Vec::with_capacity(iterations);
        let start = Instant::now();
        let mut total_tokens = 0usize;
        for _ in 0..iterations {
            let (loss, tokens) = evaluate_once(model, eval_batches);
            losses.push(loss);
            total_tokens = total_tokens.saturating_add(tokens);
        }
        let elapsed_ns = start.elapsed().as_nanos() as f64;
        let avg_loss = losses.iter().sum::<f64>() / losses.len() as f64;
        let seconds = (elapsed_ns / 1e9).max(1e-9);
        ProbeCase {
            fast_steps,
            avg_loss,
            perplexity: avg_loss.exp(),
            forward_ms: elapsed_ns / iterations as f64 / 1e6,
            tokens_per_sec: total_tokens as f64 / seconds,
            loss_gain_vs_step1: 0.0,
            gain_per_extra_ms: None,
            latency_scale_vs_step1: 1.0,
            latency_alpha_vs_step1: None,
        }
    }

    fn evaluate_once(model: &BDH<Backend>, eval_batches: &[EvalBatch<Backend>]) -> (f64, usize) {
        let mut loss_sum = 0.0;
        let mut token_sum = 0usize;
        for batch in eval_batches {
            let logits = model.forward(batch.inputs.clone());
            let loss = scalar(language_model_loss::<Backend>(
                logits,
                batch.targets.clone(),
            ));
            loss_sum += loss;
            token_sum = token_sum.saturating_add(batch.token_count);
        }
        (loss_sum / eval_batches.len() as f64, token_sum)
    }

    fn annotate_cases(cases: &mut [ProbeCase]) {
        if cases.is_empty() {
            return;
        }
        let base_steps = cases[0].fast_steps as f64;
        let base_ms = cases[0].forward_ms.max(1e-9);
        let base_loss = cases[0].avg_loss;
        for case in cases.iter_mut() {
            case.loss_gain_vs_step1 = base_loss - case.avg_loss;
            case.latency_scale_vs_step1 = case.forward_ms / base_ms;
            let step_ratio = case.fast_steps as f64 / base_steps;
            if step_ratio > 1.0 {
                case.latency_alpha_vs_step1 =
                    Some(case.latency_scale_vs_step1.ln() / step_ratio.ln());
                let extra_ms = case.forward_ms - base_ms;
                if extra_ms > 0.0 {
                    case.gain_per_extra_ms = Some(case.loss_gain_vs_step1 / extra_ms);
                }
            }
        }
    }

    fn summarize(cases: &[ProbeCase]) -> Summary {
        let first = cases.first().expect("at least one case");
        let best = cases
            .iter()
            .min_by(|a, b| a.avg_loss.total_cmp(&b.avg_loss))
            .expect("best case");
        let best_efficiency = cases
            .iter()
            .filter_map(|case| case.gain_per_extra_ms.map(|gain| (case.fast_steps, gain)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        Summary {
            best_loss_fast_steps: best.fast_steps,
            best_loss: best.avg_loss,
            best_gain_vs_step1: first.avg_loss - best.avg_loss,
            best_efficiency_fast_steps: best_efficiency.map(|(steps, _)| steps),
            best_efficiency_gain_per_extra_ms: best_efficiency.map(|(_, gain)| gain),
            latency_alpha_fit_all_steps: fit_alpha_by_fast_steps(cases, |case| case.forward_ms),
            latency_ms_per_extra_fast_step_fit: fit_linear_ms_per_fast_step(cases, |case| {
                case.forward_ms
            }),
        }
    }

    fn scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("scalar tensor data")[0] as f64
    }

    fn fit_alpha_by_fast_steps(
        cases: &[ProbeCase],
        metric: impl Fn(&ProbeCase) -> f64,
    ) -> Option<f64> {
        if cases.len() < 2 {
            return None;
        }
        let points = cases
            .iter()
            .filter_map(|case| {
                let x = (case.fast_steps as f64).ln();
                let y = metric(case).max(1e-9).ln();
                if x.is_finite() && y.is_finite() {
                    Some((x, y))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        fit_line_slope(&points)
    }

    fn fit_linear_ms_per_fast_step(
        cases: &[ProbeCase],
        metric: impl Fn(&ProbeCase) -> f64,
    ) -> Option<f64> {
        if cases.len() < 2 {
            return None;
        }
        let points = cases
            .iter()
            .filter_map(|case| {
                let x = case.fast_steps as f64;
                let y = metric(case);
                if x.is_finite() && y.is_finite() {
                    Some((x, y))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        fit_line_slope(&points)
    }

    fn fit_line_slope(points: &[(f64, f64)]) -> Option<f64> {
        if points.len() < 2 {
            return None;
        }
        let n = points.len() as f64;
        let mean_x = points.iter().map(|(x, _)| *x).sum::<f64>() / n;
        let mean_y = points.iter().map(|(_, y)| *y).sum::<f64>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for (x, y) in points {
            let dx = *x - mean_x;
            num += dx * (*y - mean_y);
            den += dx * dx;
        }
        (den > 0.0).then_some(num / den)
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Language Rollout Probe");
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "- config: {}",
            report
                .config
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
        let _ = writeln!(out, "- checkpoint: {}", report.checkpoint.display());
        let _ = writeln!(out, "- batch size: {}", report.batch_size);
        let _ = writeln!(out, "- block size: {}", report.block_size);
        let _ = writeln!(out, "- eval batches: {}", report.eval_batches);
        let _ = writeln!(out, "- warmup: {}", report.warmup);
        let _ = writeln!(out, "- iterations: {}", report.iterations);
        let _ = writeln!(
            out,
            "- best loss fast steps: {} ({:.6})",
            report.summary.best_loss_fast_steps, report.summary.best_loss
        );
        let _ = writeln!(
            out,
            "- gain vs fast1: {:.6}",
            report.summary.best_gain_vs_step1
        );
        if let Some(alpha) = report.summary.latency_alpha_fit_all_steps {
            let _ = writeln!(out, "- latency alpha fit over all fast steps: {:.3}", alpha);
        }
        if let Some(ms) = report.summary.latency_ms_per_extra_fast_step_fit {
            let _ = writeln!(out, "- latency ms / extra fast step fit: {:.3}", ms);
        }
        if let Some(step) = report.summary.best_efficiency_fast_steps {
            let gain = report
                .summary
                .best_efficiency_gain_per_extra_ms
                .unwrap_or_default();
            let _ = writeln!(out, "- best gain / extra ms: fast{step} ({gain:.6})");
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| fast steps | avg loss | ppl | forward ms | ms x(fast1) | alpha | gain(fast1) | gain/ms | tokens/s |"
        );
        let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for case in &report.cases {
            let alpha = case
                .latency_alpha_vs_step1
                .map(|value| format!("{value:.3}"))
                .unwrap_or_else(|| "-".to_string());
            let gain_ms = case
                .gain_per_extra_ms
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {:.6} | {:.3} | {:.3} | {:.3} | {} | {:.6} | {} | {:.1} |",
                case.fast_steps,
                case.avg_loss,
                case.perplexity,
                case.forward_ms,
                case.latency_scale_vs_step1,
                alpha,
                case.loss_gain_vs_step1,
                gain_ms,
                case.tokens_per_sec,
            );
        }
        out
    }

    fn write_text_artifact(path: &Path, contents: &str, label: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).unwrap_or_else(|err| {
                panic!(
                    "failed to create {} parent {}: {err}",
                    label,
                    parent.display()
                )
            });
        }
        fs::write(path, contents)
            .unwrap_or_else(|err| panic!("failed to write {} {}: {err}", label, path.display()));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
