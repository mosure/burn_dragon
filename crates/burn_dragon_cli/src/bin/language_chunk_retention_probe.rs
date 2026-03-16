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
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_dataset;
    use burn_dragon::language::{
        BDH, apply_wgpu_fused_core_override, build_model_config_with_tokenizer,
        language_model_loss, load_tokenizer_for_checkpoint, load_training_config_for_checkpoint,
        summary_event_mask_tensor,
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
        #[arg(long, value_delimiter = ',', num_args = 1.., default_values_t = [32usize, 64, 128])]
        prefix_lens: Vec<usize>,
        #[arg(long, default_value_t = 64)]
        suffix_len: usize,
        #[arg(long, default_value_t = 32)]
        eval_cases: usize,
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

        let windows = build_windows(
            dataset.as_ref(),
            DatasetSplit::Val,
            &prefix_lens,
            args.suffix_len.max(1),
            args.eval_cases.max(1),
        )
        .unwrap_or_else(|err| panic!("failed to build retention windows: {err}"));

        let mut cases = Vec::with_capacity(prefix_lens.len());
        for prefix_len in prefix_lens {
            let subset = windows
                .iter()
                .filter(|window| window.prefix_len == prefix_len)
                .collect::<Vec<_>>();
            cases.push(run_case(&model, &subset));
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

    fn init_runtime(device: &Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
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
    ) -> Result<Vec<RetentionWindow>> {
        let (offset, span) = dataset.split_offset_and_span(split);
        let tokens = dataset.tokens();
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
        let max_start = span - required;
        let stride = (max_start / eval_cases.max(1)).max(1);
        let mut cursor = 0usize;
        let mut windows = Vec::with_capacity(prefix_lens.len() * eval_cases);

        for _ in 0..eval_cases {
            let start = offset + cursor.min(max_start);
            cursor = cursor.saturating_add(stride);
            for &prefix_len in prefix_lens {
                let prefix = tokens[start..start + prefix_len]
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
                let suffix_inputs = tokens
                    [start + prefix_len - 1..start + prefix_len + suffix_len - 1]
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
                let suffix_targets = tokens[start + prefix_len..start + prefix_len + suffix_len]
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
                let full_inputs = tokens[start..start + prefix_len + suffix_len - 1]
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
                let full_targets = tokens[start + 1..start + prefix_len + suffix_len]
                    .iter()
                    .map(|&value| value as i64)
                    .collect::<Vec<_>>();
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

        Ok(windows)
    }

    fn load_model(
        config: &burn_dragon::language::TrainingConfig,
        checkpoint: &Path,
        tokenizer: &dyn burn_dragon::language::tokenizer::Tokenizer,
        device: &Device,
    ) -> Result<BDH<Backend>> {
        let mut model_config = build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer,
        )?;
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

    fn run_case(model: &BDH<Backend>, windows: &[&RetentionWindow]) -> RetentionCase {
        let prefix_len = windows.first().expect("retention windows").prefix_len;
        let suffix_len = windows.first().expect("retention windows").suffix_len;
        let mut full_losses = Vec::with_capacity(windows.len());
        let mut carry_losses = Vec::with_capacity(windows.len());
        let mut reset_losses = Vec::with_capacity(windows.len());
        let carry_start = Instant::now();
        for window in windows {
            carry_losses.push(run_carry_loss(model, window));
        }
        let carry_ms = carry_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;
        let reset_start = Instant::now();
        for window in windows {
            full_losses.push(run_full_suffix_loss(model, window));
            reset_losses.push(run_reset_loss(model, window));
        }
        let reset_ms = reset_start.elapsed().as_secs_f64() * 1000.0 / (windows.len() * 2) as f64;
        let full_loss = mean(&full_losses);
        let carry_loss = mean(&carry_losses);
        let reset_loss = mean(&reset_losses);
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
        }
    }

    fn run_full_suffix_loss(model: &BDH<Backend>, window: &RetentionWindow) -> f64 {
        let device = &Device::default();
        let inputs = tensor_2d::<Backend>(&window.full_inputs, 1, window.full_inputs.len(), device);
        let targets =
            tensor_2d::<Backend>(&window.full_targets, 1, window.full_targets.len(), device);
        let logits = match summary_event_mask_tensor::<Backend>(
            &window.full_inputs,
            1,
            window.full_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => model.forward_with_summary_event_mask(inputs, mask),
            None => model.forward(inputs),
        };
        let vocab = logits.shape().dims::<3>()[2];
        let suffix_start = window.prefix_len - 1;
        let suffix_end = suffix_start + window.suffix_len;
        let suffix_logits = logits.slice([0..1, suffix_start..suffix_end, 0..vocab]);
        let suffix_targets = targets.slice([0..1, suffix_start..suffix_end]);
        scalar(language_model_loss::<Backend>(
            suffix_logits,
            suffix_targets,
        ))
    }

    fn run_carry_loss(model: &BDH<Backend>, window: &RetentionWindow) -> f64 {
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
                let _ = model.forward_with_state_and_summary_event_mask(prefix, mask, &mut state);
            }
            None => {
                let _ = model.forward_with_state(prefix, &mut state);
            }
        }
        let logits = match summary_event_mask_tensor::<Backend>(
            &window.suffix_inputs,
            1,
            window.suffix_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                model.forward_with_state_and_summary_event_mask(suffix_inputs, mask, &mut state)
            }
            None => model.forward_with_state(suffix_inputs, &mut state),
        };
        scalar(language_model_loss::<Backend>(logits, suffix_targets))
    }

    fn run_reset_loss(model: &BDH<Backend>, window: &RetentionWindow) -> f64 {
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
        let logits = match summary_event_mask_tensor::<Backend>(
            &window.suffix_inputs,
            1,
            window.suffix_inputs.len(),
            model.summary_memory_write_trigger_token_ids(),
            device,
        ) {
            Some(mask) => {
                model.forward_with_state_and_summary_event_mask(suffix_inputs, mask, &mut state)
            }
            None => model.forward_with_state(suffix_inputs, &mut state),
        };
        scalar(language_model_loss::<Backend>(logits, suffix_targets))
    }

    fn tensor_2d<B: BackendTrait>(
        values: &[i64],
        batch: usize,
        time: usize,
        device: &B::Device,
    ) -> Tensor<B, 2, Int> {
        Tensor::<B, 2, Int>::from_data(TensorData::new(values.to_vec(), [batch, time]), device)
    }

    fn scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f64 {
        tensor
            .into_data()
            .to_vec::<f32>()
            .expect("scalar tensor data")[0] as f64
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len().max(1) as f64
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
