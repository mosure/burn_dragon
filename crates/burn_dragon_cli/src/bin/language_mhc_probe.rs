#![recursion_limit = "256"]

#[cfg(not(feature = "train"))]
fn main() {
    panic!("language_mhc_probe requires --features train");
}

#[cfg(feature = "train")]
mod real {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, Tensor, TensorData};
    use burn_dragon::core::LanguageMhcLayerDiagnostics;
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_dataset;
    use burn_dragon::language::{
        language_model_loss, load_language_core_from_checkpoint,
        load_training_config_for_checkpoint, summary_event_mask_tensor,
    };
    use burn_ndarray::NdArray;
    use clap::{Parser, ValueEnum};
    use serde::Serialize;

    #[cfg(feature = "cuda")]
    use burn_cuda::Cuda;

    const COSINE_THRESHOLD: f64 = 0.999;
    const ALPHA_ENTROPY_NORMALIZED_MAX: f64 = 0.98;
    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum)]
    enum BackendArg {
        Ndarray,
        Cuda,
    }

    impl BackendArg {
        fn as_backend_name(self) -> &'static str {
            match self {
                Self::Ndarray => "cpu",
                Self::Cuda => "cuda",
            }
        }
    }

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
        backend: BackendArg,
        #[arg(long, default_value_t = 4)]
        eval_batches: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        block_size: Option<usize>,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone)]
    struct EvalBatch<B: BackendTrait> {
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    }

    #[derive(Clone, Debug, Default)]
    struct DiagnosticsAccumulator {
        count: usize,
        num_streams: usize,
        stream_norm_mean_sum: f64,
        stream_norm_variance_sum: f64,
        pairwise_stream_cosine_mean_sum: f64,
        pairwise_stream_cosine_mean_count: usize,
        alpha_entropy_mean_sum: f64,
        alpha_entropy_normalized_mean_sum: f64,
        beta_entropy_mean_sum: f64,
        beta_entropy_mean_count: usize,
        beta_entropy_normalized_mean_sum: f64,
        beta_entropy_normalized_mean_count: usize,
        residual_distance_identity_l1_mean_sum: f64,
        residual_distance_uniform_l1_mean_sum: f64,
    }

    #[derive(Clone, Serialize)]
    struct Phase0Criteria {
        cosine_threshold: f64,
        alpha_entropy_normalized_max: f64,
        stream_divergence_pass: bool,
        passing_layers: Vec<usize>,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        config: Vec<PathBuf>,
        checkpoint: PathBuf,
        backend: &'static str,
        batch_size: usize,
        block_size: usize,
        eval_batches: usize,
        avg_loss: f64,
        layers: Vec<LanguageMhcLayerDiagnostics>,
        phase0: Phase0Criteria,
    }

    pub fn main() {
        if let Err(err) = run() {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    }

    fn run() -> Result<()> {
        let args = Args::parse();
        match args.backend {
            BackendArg::Ndarray => run_backend::<NdArray<f32>>(&args),
            BackendArg::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    run_backend::<Cuda<f32>>(&args)
                }
                #[cfg(not(feature = "cuda"))]
                {
                    Err(anyhow!(
                        "cuda backend selected but this build lacks `cuda` feature"
                    ))
                }
            }
        }
    }

    fn run_backend<B: BackendTrait>(args: &Args) -> Result<()> {
        let backend_name = args.backend.as_backend_name();
        let config =
            load_training_config_for_checkpoint(&args.config, Some(&args.checkpoint), backend_name)
                .with_context(|| {
                    format!(
                        "failed to load config/checkpoint metadata for {}",
                        args.checkpoint.display()
                    )
                })?;
        let dataset = prepare_dataset(&config.dataset, &config.training)
            .context("failed to prepare dataset")?;
        let device = B::Device::default();
        B::seed(&device, 1337);
        let model = load_language_core_from_checkpoint::<B>(
            &args.checkpoint,
            None,
            &args.config,
            backend_name,
            &device,
        )
        .with_context(|| format!("failed to load checkpoint {}", args.checkpoint.display()))?;

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
        let summary_event_token_ids = model
            .summary_memory_write_trigger_token_ids()
            .map(|ids| ids.to_vec());
        let eval_batches = build_eval_batches::<B>(
            dataset.as_ref(),
            DatasetSplit::Val,
            batch_size,
            block_size,
            args.eval_batches.max(1),
            summary_event_token_ids.as_deref(),
            &device,
        )?;

        let mut diagnostics = BTreeMap::<usize, DiagnosticsAccumulator>::new();
        let mut loss_sum = 0.0f64;
        for batch in &eval_batches {
            let logits = if let Some(mask) = batch.summary_event_mask.as_ref() {
                model.forward_with_summary_event_mask(batch.inputs.clone(), mask.clone())
            } else {
                model.forward(batch.inputs.clone())
            };
            loss_sum += scalar(language_model_loss::<B>(logits, batch.targets.clone()));

            let batch_diagnostics = if let Some(mask) = batch.summary_event_mask.as_ref() {
                model.collect_language_mhc_diagnostics_with_summary_event_mask(
                    batch.inputs.clone(),
                    mask.clone(),
                )
            } else {
                model.collect_language_mhc_diagnostics(batch.inputs.clone())
            };
            accumulate_diagnostics(&mut diagnostics, &batch_diagnostics);
        }

        let layers = finalize_diagnostics(diagnostics);
        let passing_layers = layers
            .iter()
            .filter(|diag| layer_passes_phase0(diag))
            .map(|diag| diag.layer_index)
            .collect::<Vec<_>>();
        let report = Report {
            benchmark: "burn_dragon language mHC checkpoint probe",
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            backend: backend_name,
            batch_size,
            block_size,
            eval_batches: eval_batches.len(),
            avg_loss: loss_sum / eval_batches.len().max(1) as f64,
            layers,
            phase0: Phase0Criteria {
                cosine_threshold: COSINE_THRESHOLD,
                alpha_entropy_normalized_max: ALPHA_ENTROPY_NORMALIZED_MAX,
                stream_divergence_pass: !passing_layers.is_empty(),
                passing_layers,
            },
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).context("serialize mHC probe report")?;
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact")?;
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact")?;
        }

        Ok(())
    }

    fn build_eval_batches<B: BackendTrait>(
        dataset: &Dataset,
        split: DatasetSplit,
        batch_size: usize,
        block_size: usize,
        num_batches: usize,
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Result<Vec<EvalBatch<B>>> {
        let (offset, span) = dataset.split_offset_and_span(split);
        let mut tokens = vec![0u32; dataset.token_count()];
        dataset.copy_token_range(0, &mut tokens);
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
            let summary_event_mask = summary_event_mask_tensor::<B>(
                &inputs,
                batch_size,
                block_size,
                summary_event_token_ids,
                device,
            );
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
                summary_event_mask,
            });
        }

        Ok(batches)
    }

    fn accumulate_diagnostics(
        accumulators: &mut BTreeMap<usize, DiagnosticsAccumulator>,
        diagnostics: &[LanguageMhcLayerDiagnostics],
    ) {
        for diag in diagnostics {
            let accumulator = accumulators.entry(diag.layer_index).or_default();
            accumulator.count += 1;
            accumulator.num_streams = diag.num_streams;
            accumulator.stream_norm_mean_sum += diag.stream_norm_mean;
            accumulator.stream_norm_variance_sum += diag.stream_norm_variance;
            if let Some(value) = diag.pairwise_stream_cosine_mean {
                accumulator.pairwise_stream_cosine_mean_sum += value;
                accumulator.pairwise_stream_cosine_mean_count += 1;
            }
            accumulator.alpha_entropy_mean_sum += diag.alpha_entropy_mean;
            accumulator.alpha_entropy_normalized_mean_sum += diag.alpha_entropy_normalized_mean;
            if let Some(value) = diag.beta_entropy_mean {
                accumulator.beta_entropy_mean_sum += value;
                accumulator.beta_entropy_mean_count += 1;
            }
            if let Some(value) = diag.beta_entropy_normalized_mean {
                accumulator.beta_entropy_normalized_mean_sum += value;
                accumulator.beta_entropy_normalized_mean_count += 1;
            }
            accumulator.residual_distance_identity_l1_mean_sum +=
                diag.residual_distance_identity_l1_mean;
            accumulator.residual_distance_uniform_l1_mean_sum +=
                diag.residual_distance_uniform_l1_mean;
        }
    }

    fn finalize_diagnostics(
        accumulators: BTreeMap<usize, DiagnosticsAccumulator>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        accumulators
            .into_iter()
            .map(|(layer_index, accumulator)| {
                let count = accumulator.count.max(1) as f64;
                LanguageMhcLayerDiagnostics {
                    layer_index,
                    num_streams: accumulator.num_streams,
                    stream_norm_mean: accumulator.stream_norm_mean_sum / count,
                    stream_norm_variance: accumulator.stream_norm_variance_sum / count,
                    pairwise_stream_cosine_mean: (accumulator.pairwise_stream_cosine_mean_count
                        > 0)
                    .then_some(
                        accumulator.pairwise_stream_cosine_mean_sum
                            / accumulator.pairwise_stream_cosine_mean_count as f64,
                    ),
                    alpha_entropy_mean: accumulator.alpha_entropy_mean_sum / count,
                    alpha_entropy_normalized_mean: accumulator.alpha_entropy_normalized_mean_sum
                        / count,
                    beta_entropy_mean: (accumulator.beta_entropy_mean_count > 0).then_some(
                        accumulator.beta_entropy_mean_sum
                            / accumulator.beta_entropy_mean_count as f64,
                    ),
                    beta_entropy_normalized_mean: (accumulator.beta_entropy_normalized_mean_count
                        > 0)
                    .then_some(
                        accumulator.beta_entropy_normalized_mean_sum
                            / accumulator.beta_entropy_normalized_mean_count as f64,
                    ),
                    residual_distance_identity_l1_mean: accumulator
                        .residual_distance_identity_l1_mean_sum
                        / count,
                    residual_distance_uniform_l1_mean: accumulator
                        .residual_distance_uniform_l1_mean_sum
                        / count,
                }
            })
            .collect()
    }

    fn layer_passes_phase0(diag: &LanguageMhcLayerDiagnostics) -> bool {
        diag.pairwise_stream_cosine_mean
            .is_some_and(|value| value <= COSINE_THRESHOLD)
            && diag.alpha_entropy_normalized_mean <= ALPHA_ENTROPY_NORMALIZED_MAX
    }

    fn scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
        tensor.into_data().to_vec::<f32>().expect("scalar value")[0] as f64
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Language mHC Probe");
        let _ = writeln!(out);
        let _ = writeln!(out, "- checkpoint: {}", report.checkpoint.display());
        let _ = writeln!(out, "- backend: {}", report.backend);
        let _ = writeln!(out, "- batch size: {}", report.batch_size);
        let _ = writeln!(out, "- block size: {}", report.block_size);
        let _ = writeln!(out, "- eval batches: {}", report.eval_batches);
        let _ = writeln!(out, "- avg loss: {:.6}", report.avg_loss);
        let _ = writeln!(
            out,
            "- phase0 stream divergence pass: {}",
            report.phase0.stream_divergence_pass
        );
        let _ = writeln!(
            out,
            "- criteria: cosine <= {:.3}, alpha_norm <= {:.2}",
            report.phase0.cosine_threshold, report.phase0.alpha_entropy_normalized_max,
        );
        if !report.phase0.passing_layers.is_empty() {
            let _ = writeln!(
                out,
                "- passing layers: {}",
                report
                    .phase0
                    .passing_layers
                    .iter()
                    .map(|layer| layer.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| layer | streams | norm mean | norm var | pairwise cosine | alpha H | alpha H/logS | beta H/logS | ||M-I||_1 | ||M-U||_1 |"
        );
        let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for layer in &report.layers {
            let cosine = layer
                .pairwise_stream_cosine_mean
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let beta = layer
                .beta_entropy_normalized_mean
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "-".to_string());
            let _ = writeln!(
                out,
                "| {} | {} | {:.6} | {:.6} | {} | {:.6} | {:.6} | {} | {:.6} | {:.6} |",
                layer.layer_index,
                layer.num_streams,
                layer.stream_norm_mean,
                layer.stream_norm_variance,
                cosine,
                layer.alpha_entropy_mean,
                layer.alpha_entropy_normalized_mean,
                beta,
                layer.residual_distance_identity_l1_mean,
                layer.residual_distance_uniform_l1_mean,
            );
        }
        out
    }

    fn write_text_artifact(path: &Path, contents: &str, label: &str) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| {
                format!("failed to create {} parent {}", label, parent.display())
            })?;
        }
        fs::write(path, contents)
            .with_context(|| format!("failed to write {} {}", label, path.display()))
    }
}

#[cfg(feature = "train")]
fn main() {
    real::main();
}
