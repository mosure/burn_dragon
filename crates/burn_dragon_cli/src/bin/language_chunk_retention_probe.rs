#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("language_chunk_retention_probe requires --features benchmark");
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
    #[cfg(feature = "language-cuda")]
    use burn_cuda::Cuda;
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_dataset;
    use burn_dragon::language::{
        BDH, build_model_config_with_tokenizer, load_tokenizer_for_checkpoint,
        load_training_config_for_checkpoint, summary_event_mask_tensor,
    };
    #[cfg(not(feature = "language-cuda"))]
    use burn_ndarray::NdArray;
    use clap::Parser;
    use serde::Serialize;

    #[cfg(feature = "language-cuda")]
    type Backend = Cuda<f32, i32>;
    #[cfg(not(feature = "language-cuda"))]
    type Backend = NdArray<f32>;
    type Device = <Backend as BackendTrait>::Device;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, required = true)]
        config: Vec<PathBuf>,
        #[arg(long, required = true)]
        checkpoint: PathBuf,
        #[arg(long, value_delimiter = ',', num_args = 1.., default_values_t = [32usize, 64, 128])]
        prefix_lens: Vec<usize>,
        #[arg(long, value_delimiter = ',', num_args = 1.., default_values_t = [64usize])]
        suffix_lens: Vec<usize>,
        #[arg(long, default_value_t = 32)]
        eval_cases: usize,
        #[arg(long, default_value_t = false)]
        document_aligned: bool,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Serialize)]
    struct RetentionCase {
        prefix_len: usize,
        suffix_len: usize,
        full_loss: f64,
        carry_loss: f64,
        reset_loss: f64,
        carry_gain_vs_reset: f64,
        carry_gap_to_full: f64,
        carry_ms: f64,
        reset_ms: f64,
        full_token_losses: Vec<f64>,
        carry_token_losses: Vec<f64>,
        reset_token_losses: Vec<f64>,
        carry_gain_vs_reset_by_token: Vec<f64>,
        carry_gap_to_full_by_token: Vec<f64>,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        config: Vec<PathBuf>,
        checkpoint: PathBuf,
        eval_cases: usize,
        cases: Vec<RetentionCase>,
    }

    pub fn main() {
        let args = Args::parse();
        let device = Device::default();
        let backend_name = if cfg!(feature = "language-cuda") {
            "cuda"
        } else {
            "ndarray"
        };

        let config =
            load_training_config_for_checkpoint(&args.config, Some(&args.checkpoint), backend_name)
                .unwrap_or_else(|err| {
                    panic!("failed to load language config/checkpoint metadata: {err}")
                });
        let dataset = prepare_dataset(&config.dataset, &config.training)
            .unwrap_or_else(|err| panic!("failed to prepare dataset: {err}"));
        let tokenizer =
            load_tokenizer_for_checkpoint(&args.config, Some(&args.checkpoint), backend_name)
            .unwrap_or_else(|err| panic!("failed to load tokenizer for checkpoint: {err}"));
        let model = load_model(&config, &args.checkpoint, tokenizer.as_ref(), &device)
            .unwrap_or_else(|err| panic!("failed to load model: {err}"));

        let mut prefix_lens = args
            .prefix_lens
            .iter()
            .copied()
            .map(|value| value.max(1))
            .collect::<Vec<_>>();
        prefix_lens.sort_unstable();
        prefix_lens.dedup();
        let mut suffix_lens = args
            .suffix_lens
            .iter()
            .copied()
            .map(|value| value.max(1))
            .collect::<Vec<_>>();
        suffix_lens.sort_unstable();
        suffix_lens.dedup();

        let mut cases = Vec::with_capacity(prefix_lens.len() * suffix_lens.len());
        for suffix_len in suffix_lens {
            let windows = build_windows(
                dataset.as_ref(),
                DatasetSplit::Val,
                &prefix_lens,
                suffix_len,
                args.eval_cases.max(1),
                args.document_aligned,
            )
            .unwrap_or_else(|err| panic!("failed to build retention windows: {err}"));
            for &prefix_len in &prefix_lens {
                let subset = windows
                    .iter()
                    .filter(|window| window.prefix_len == prefix_len)
                    .collect::<Vec<_>>();
                if subset.is_empty() {
                    continue;
                }
                cases.push(run_case(&model, &subset));
            }
        }

        let report = Report {
            benchmark: "burn_dragon language chunk retention probe",
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            eval_cases: args.eval_cases.max(1),
            cases,
        };

        let markdown = format_markdown(&report);
        let json =
            serde_json::to_string_pretty(&report).expect("serialize language retention report");
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact");
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact");
        }
    }

    struct RetentionWindow {
        prefix_len: usize,
        suffix_len: usize,
        prefix_tokens: Vec<i64>,
        suffix_inputs: Vec<i64>,
        suffix_targets: Vec<i64>,
        full_inputs: Vec<i64>,
        full_targets: Vec<i64>,
    }

    fn build_windows(
        dataset: &Dataset,
        split: DatasetSplit,
        prefix_lens: &[usize],
        suffix_len: usize,
        eval_cases: usize,
        document_aligned: bool,
    ) -> Result<Vec<RetentionWindow>> {
        let (offset, span) = dataset.split_offset_and_span(split);
        let max_prefix = prefix_lens
            .iter()
            .copied()
            .max()
            .context("no prefix lengths")?;
        let required = max_prefix
            .checked_add(suffix_len)
            .and_then(|value| value.checked_add(1))
            .context("prefix/suffix overflow")?;
        if span <= required {
            anyhow::bail!("validation split too small for retention windows");
        }

        let mut windows = Vec::with_capacity(prefix_lens.len() * eval_cases);

        if document_aligned {
            let logical_document_tokens = dataset
                .preferred_logical_document_tokens(split)
                .context("document-aligned retention requires preferred logical document tokens")?;
            let document_span = logical_document_tokens + 1;
            let active_prefix_lens = prefix_lens
                .iter()
                .copied()
                .filter(|prefix_len| {
                    prefix_len
                        .checked_add(suffix_len)
                        .and_then(|value| value.checked_add(1))
                        .is_some_and(|needed| needed <= document_span)
                })
                .collect::<Vec<_>>();
            if active_prefix_lens.is_empty() {
                anyhow::bail!(
                    "logical document length {} is too small for every prefix + suffix combination (max prefix {}, suffix {})",
                    logical_document_tokens,
                    max_prefix,
                    suffix_len
                );
            }
            let num_documents = (span / document_span).max(1);
            let stride = (num_documents / eval_cases.max(1)).max(1);
            let mut doc_cursor = 0usize;
            for _ in 0..eval_cases.min(num_documents) {
                let doc_index = doc_cursor.min(num_documents.saturating_sub(1));
                let start = offset + doc_index.saturating_mul(document_span);
                doc_cursor = doc_cursor.saturating_add(stride);
                append_windows_for_start(
                    dataset,
                    start,
                    &active_prefix_lens,
                    suffix_len,
                    &mut windows,
                );
            }
            return Ok(windows);
        }

        let max_start = span - required;
        let stride = (max_start / eval_cases.max(1)).max(1);
        let mut cursor = 0usize;
        for _ in 0..eval_cases {
            let start = offset + cursor.min(max_start);
            cursor = cursor.saturating_add(stride);
            append_windows_for_start(dataset, start, prefix_lens, suffix_len, &mut windows);
        }
        Ok(windows)
    }

    fn append_windows_for_start(
        dataset: &Dataset,
        start: usize,
        prefix_lens: &[usize],
        suffix_len: usize,
        windows: &mut Vec<RetentionWindow>,
    ) {
        for &prefix_len in prefix_lens {
            let prefix =
                copy_tokens(dataset, start, prefix_len).into_iter().map(i64::from).collect();
            let suffix_inputs = copy_tokens(dataset, start + prefix_len - 1, suffix_len)
                .into_iter()
                .map(i64::from)
                .collect();
            let suffix_targets = copy_tokens(dataset, start + prefix_len, suffix_len)
                .into_iter()
                .map(i64::from)
                .collect();
            let full_inputs = copy_tokens(dataset, start, prefix_len + suffix_len - 1)
                .into_iter()
                .map(i64::from)
                .collect();
            let full_targets = copy_tokens(dataset, start + 1, prefix_len + suffix_len - 1)
                .into_iter()
                .map(i64::from)
                .collect();
            windows.push(RetentionWindow {
                prefix_len,
                suffix_len,
                prefix_tokens: prefix,
                suffix_inputs,
                suffix_targets,
                full_inputs,
                full_targets,
            });
        }
    }

    fn copy_tokens(dataset: &Dataset, start: usize, len: usize) -> Vec<u32> {
        let mut buffer = vec![0u32; len];
        dataset.copy_token_range(start, &mut buffer);
        buffer
    }

    fn load_model(
        config: &burn_dragon::language::TrainingConfig,
        checkpoint: &Path,
        tokenizer: &dyn burn_dragon::language::tokenizer::Tokenizer,
        device: &Device,
    ) -> Result<BDH<Backend>> {
        let model_config = build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer,
        )?;
        let mut model = BDH::<Backend>::new(model_config, device);
        let record = BinFileRecorder::<FullPrecisionSettings>::new()
            .load::<<BDH<Backend> as Module<Backend>>::Record>(checkpoint.to_path_buf(), device)
            .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;
        model = model.load_record(record);
        Ok(model)
    }

    fn run_case(model: &BDH<Backend>, windows: &[&RetentionWindow]) -> RetentionCase {
        let prefix_len = windows.first().expect("retention windows").prefix_len;
        let suffix_len = windows.first().expect("retention windows").suffix_len;
        let mut full_losses = Vec::with_capacity(windows.len());
        let mut carry_losses = Vec::with_capacity(windows.len());
        let mut reset_losses = Vec::with_capacity(windows.len());
        let mut full_token_losses = Vec::with_capacity(windows.len());
        let mut carry_token_losses = Vec::with_capacity(windows.len());
        let mut reset_token_losses = Vec::with_capacity(windows.len());
        let carry_start = Instant::now();
        for window in windows {
            let losses = run_carry_loss_tokens(model, window);
            carry_losses.push(mean(&losses));
            carry_token_losses.push(losses);
        }
        let carry_ms = carry_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;
        let reset_start = Instant::now();
        for window in windows {
            let full = run_full_suffix_loss_tokens(model, window);
            full_losses.push(mean(&full));
            full_token_losses.push(full);
            let reset = run_reset_loss_tokens(model, window);
            reset_losses.push(mean(&reset));
            reset_token_losses.push(reset);
        }
        let reset_ms = reset_start.elapsed().as_secs_f64() * 1000.0 / (windows.len() * 2) as f64;
        let full_loss = mean(&full_losses);
        let carry_loss = mean(&carry_losses);
        let reset_loss = mean(&reset_losses);
        let full_token_losses = mean_by_position(&full_token_losses);
        let carry_token_losses = mean_by_position(&carry_token_losses);
        let reset_token_losses = mean_by_position(&reset_token_losses);
        let carry_gain_vs_reset_by_token = carry_token_losses
            .iter()
            .zip(reset_token_losses.iter())
            .map(|(carry, reset)| reset - carry)
            .collect::<Vec<_>>();
        let carry_gap_to_full_by_token = carry_token_losses
            .iter()
            .zip(full_token_losses.iter())
            .map(|(carry, full)| carry - full)
            .collect::<Vec<_>>();
        RetentionCase {
            prefix_len,
            suffix_len,
            full_loss,
            carry_loss,
            reset_loss,
            carry_gain_vs_reset: reset_loss - carry_loss,
            carry_gap_to_full: carry_loss - full_loss,
            carry_ms,
            reset_ms,
            full_token_losses,
            carry_token_losses,
            reset_token_losses,
            carry_gain_vs_reset_by_token,
            carry_gap_to_full_by_token,
        }
    }

    fn run_full_suffix_loss_tokens(model: &BDH<Backend>, window: &RetentionWindow) -> Vec<f64> {
        let device = &Device::default();
        let inputs = tensor_2d::<Backend>(&window.full_inputs, 1, window.full_inputs.len(), device);
        let targets =
            tensor_2d::<Backend>(&window.full_targets, 1, window.full_targets.len(), device);
        let hidden = match summary_event_mask_tensor::<Backend>(
            &window.full_inputs,
            1,
            window.full_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                let mut state = model.init_state();
                model.forward_hidden_with_state_and_summary_event_mask(inputs, mask, &mut state)
            }
            None => model.forward_hidden(inputs),
        };
        let suffix_start = window.prefix_len - 1;
        let suffix_end = suffix_start + window.suffix_len;
        let suffix_losses = model
            .language_token_losses_from_hidden(hidden, targets)
            .slice([0..1, suffix_start..suffix_end]);
        token_losses_from_tensor(suffix_losses)
    }

    fn run_carry_loss_tokens(model: &BDH<Backend>, window: &RetentionWindow) -> Vec<f64> {
        let device = &Device::default();
        let prefix =
            tensor_2d::<Backend>(&window.prefix_tokens, 1, window.prefix_tokens.len(), device);
        let suffix_inputs =
            tensor_2d::<Backend>(&window.suffix_inputs, 1, window.suffix_inputs.len(), device);
        let suffix_targets = tensor_2d::<Backend>(
            &window.suffix_targets,
            1,
            window.suffix_targets.len(),
            device,
        );
        let mut state = model.init_state();
        match summary_event_mask_tensor::<Backend>(
            &window.prefix_tokens,
            1,
            window.prefix_tokens.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                let _ =
                    model.forward_hidden_with_state_and_summary_event_mask(prefix, mask, &mut state);
            }
            None => {
                let _ = model.forward_hidden_with_state(prefix, &mut state);
            }
        }
        let hidden = match summary_event_mask_tensor::<Backend>(
            &window.suffix_inputs,
            1,
            window.suffix_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                model.forward_hidden_with_state_and_summary_event_mask(
                    suffix_inputs,
                    mask,
                    &mut state,
                )
            }
            None => model.forward_hidden_with_state(suffix_inputs, &mut state),
        };
        let losses = model.language_token_losses_from_hidden(hidden, suffix_targets);
        token_losses_from_tensor(losses)
    }

    fn run_reset_loss_tokens(model: &BDH<Backend>, window: &RetentionWindow) -> Vec<f64> {
        let device = &Device::default();
        let suffix_inputs =
            tensor_2d::<Backend>(&window.suffix_inputs, 1, window.suffix_inputs.len(), device);
        let suffix_targets = tensor_2d::<Backend>(
            &window.suffix_targets,
            1,
            window.suffix_targets.len(),
            device,
        );
        let mut state = model.init_state();
        let hidden = match summary_event_mask_tensor::<Backend>(
            &window.suffix_inputs,
            1,
            window.suffix_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                model.forward_hidden_with_state_and_summary_event_mask(
                    suffix_inputs,
                    mask,
                    &mut state,
                )
            }
            None => model.forward_hidden_with_state(suffix_inputs, &mut state),
        };
        let losses = model.language_token_losses_from_hidden(hidden, suffix_targets);
        token_losses_from_tensor(losses)
    }

    fn tensor_2d<B: BackendTrait>(
        values: &[i64],
        batch: usize,
        time: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), [batch, time]), device)
    }

    fn token_losses_from_tensor<B: BackendTrait>(losses: Tensor<B, 2>) -> Vec<f64> {
        let [batch, time] = losses.shape().dims();
        losses
            .reshape([batch * time])
            .into_data()
            .to_vec::<f32>()
            .expect("token loss tensor data")
            .into_iter()
            .map(|value| value as f64)
            .collect()
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len().max(1) as f64
    }

    fn mean_by_position(series: &[Vec<f64>]) -> Vec<f64> {
        let max_len = series.iter().map(Vec::len).max().unwrap_or(0);
        let mut sums = vec![0.0f64; max_len];
        let mut counts = vec![0usize; max_len];
        for values in series {
            for (index, value) in values.iter().enumerate() {
                sums[index] += value;
                counts[index] += 1;
            }
        }
        sums.into_iter()
            .zip(counts)
            .map(|(sum, count)| {
                if count == 0 {
                    f64::NAN
                } else {
                    sum / count as f64
                }
            })
            .collect()
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Language Chunk Retention Probe");
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
        let _ = writeln!(out, "- eval cases: {}", report.eval_cases);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| prefix | suffix | full loss | carry loss | reset loss | carry gain vs reset | carry gap to full | carry ms | reset ms |"
        );
        let _ = writeln!(out, "|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
        for case in &report.cases {
            let _ = writeln!(
                out,
                "| {} | {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.3} | {:.3} |",
                case.prefix_len,
                case.suffix_len,
                case.full_loss,
                case.carry_loss,
                case.reset_loss,
                case.carry_gain_vs_reset,
                case.carry_gap_to_full,
                case.carry_ms,
                case.reset_ms,
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
