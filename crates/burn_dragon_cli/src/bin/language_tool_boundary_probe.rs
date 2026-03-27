#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("language_tool_boundary_probe requires --features benchmark");
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
        BDH, WgpuFusedCoreOverride, apply_wgpu_fused_core_override,
        build_model_config_with_tokenizer, language_model_loss, load_tokenizer_for_checkpoint,
        load_training_config_for_checkpoint, prefill_state, summary_event_mask_tensor,
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
        #[arg(long, default_value_t = 32)]
        refresh_tail_len: usize,
        #[arg(long, default_value_t = 1)]
        rehearse_passes: usize,
        #[arg(long, default_value_t = 2.0)]
        selective_surprise_threshold: f64,
        #[arg(
            long,
            default_value = "\nTool: search\nObservation: no external result\nAssistant:"
        )]
        tool_text: String,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Serialize)]
    struct ProbeCase {
        prefix_len: usize,
        suffix_len: usize,
        tool_tokens: usize,
        plain_loss: f64,
        carry_loss: f64,
        reset_loss: f64,
        rehearse_loss: f64,
        refresh_loss: f64,
        selective_loss: f64,
        oracle_loss: f64,
        carry_gain_vs_reset: f64,
        rehearse_gain_vs_carry: f64,
        refresh_gain_vs_reset: f64,
        selective_gain_vs_carry: f64,
        oracle_gain_vs_carry: f64,
        carry_gap_to_plain: f64,
        reset_gap_to_plain: f64,
        rehearse_gap_to_plain: f64,
        refresh_gap_to_plain: f64,
        selective_gap_to_plain: f64,
        oracle_gap_to_plain: f64,
        carry_ms: f64,
        reset_ms: f64,
        rehearse_ms: f64,
        refresh_ms: f64,
        selective_ms: f64,
        selective_refresh_rate: f64,
        selective_tool_surprise: f64,
        oracle_noncarry_rate: f64,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        config: Vec<PathBuf>,
        checkpoint: PathBuf,
        eval_cases: usize,
        tool_text: String,
        refresh_tail_len: usize,
        rehearse_passes: usize,
        selective_surprise_threshold: f64,
        cases: Vec<ProbeCase>,
    }

    struct ToolBoundaryWindow {
        prefix_len: usize,
        suffix_len: usize,
        prefix_tokens: Vec<i64>,
        suffix_inputs: Vec<i64>,
        suffix_targets: Vec<i64>,
        full_inputs: Vec<i64>,
        full_targets: Vec<i64>,
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
        let tool_tokens = tokenizer
            .encode(&args.tool_text, false, false)
            .into_iter()
            .map(|value| value as i64)
            .collect::<Vec<_>>();
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
        .unwrap_or_else(|err| panic!("failed to build tool-boundary windows: {err}"));

        let mut cases = Vec::with_capacity(prefix_lens.len());
        for prefix_len in prefix_lens {
            let subset = windows
                .iter()
                .filter(|window| window.prefix_len == prefix_len)
                .collect::<Vec<_>>();
            cases.push(run_case(
                &model,
                &subset,
                &tool_tokens,
                args.refresh_tail_len,
                args.rehearse_passes,
                args.selective_surprise_threshold,
                &device,
            ));
        }

        let report = Report {
            benchmark: "burn_dragon language tool-boundary probe",
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            eval_cases: args.eval_cases.max(1),
            tool_text: args.tool_text.clone(),
            refresh_tail_len: args.refresh_tail_len,
            rehearse_passes: args.rehearse_passes,
            selective_surprise_threshold: args.selective_surprise_threshold,
            cases,
        };

        let markdown = format_markdown(&report);
        let json =
            serde_json::to_string_pretty(&report).expect("serialize language tool-boundary report");
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

    fn build_windows(
        dataset: &Dataset,
        split: DatasetSplit,
        prefix_lens: &[usize],
        suffix_len: usize,
        eval_cases: usize,
    ) -> Result<Vec<ToolBoundaryWindow>> {
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
            anyhow::bail!("validation split too small for tool-boundary windows");
        }
        let max_start = span - required;
        let stride = (max_start / eval_cases.max(1)).max(1);
        let mut cursor = 0usize;
        let mut windows = Vec::with_capacity(prefix_lens.len() * eval_cases);

        for _ in 0..eval_cases {
            let start = offset + cursor.min(max_start);
            cursor = cursor.saturating_add(stride);
            for &prefix_len in prefix_lens {
                let prefix_tokens = tokens[start..start + prefix_len]
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
                windows.push(ToolBoundaryWindow {
                    prefix_len,
                    suffix_len,
                    prefix_tokens,
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
            WgpuFusedCoreOverride {
                recurrent: config.wgpu.training.fused_core_recurrent,
                rollout: config.wgpu.training.fused_core_rollout,
            },
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
        windows: &[&ToolBoundaryWindow],
        tool_tokens: &[i64],
        refresh_tail_len: usize,
        rehearse_passes: usize,
        selective_surprise_threshold: f64,
        device: &Device,
    ) -> ProbeCase {
        let prefix_len = windows.first().expect("tool-boundary windows").prefix_len;
        let suffix_len = windows.first().expect("tool-boundary windows").suffix_len;

        let mut plain_losses = Vec::with_capacity(windows.len());
        let mut carry_losses = Vec::with_capacity(windows.len());
        let mut reset_losses = Vec::with_capacity(windows.len());
        let mut rehearse_losses = Vec::with_capacity(windows.len());
        let mut refresh_losses = Vec::with_capacity(windows.len());
        let mut selective_losses = Vec::with_capacity(windows.len());
        let mut selective_refreshes = 0usize;
        let mut selective_tool_surprises = Vec::with_capacity(windows.len());
        let mut oracle_losses = Vec::with_capacity(windows.len());
        let mut oracle_noncarry = 0usize;

        let carry_start = Instant::now();
        for window in windows {
            carry_losses.push(run_carry_loss(model, window, tool_tokens, device));
        }
        let carry_ms = carry_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;

        let rehearse_start = Instant::now();
        for window in windows {
            rehearse_losses.push(run_rehearse_loss(
                model,
                window,
                tool_tokens,
                rehearse_passes,
                device,
            ));
        }
        let rehearse_ms = rehearse_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;

        let refresh_start = Instant::now();
        for window in windows {
            refresh_losses.push(run_refresh_loss(
                model,
                window,
                tool_tokens,
                refresh_tail_len,
                device,
            ));
        }
        let refresh_ms = refresh_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;

        let selective_start = Instant::now();
        for window in windows {
            let result = run_selective_refresh_loss(
                model,
                window,
                tool_tokens,
                refresh_tail_len,
                selective_surprise_threshold,
                device,
            );
            selective_losses.push(result.loss);
            selective_refreshes += usize::from(result.refreshed);
            selective_tool_surprises.push(result.tool_surprise);
        }
        let selective_ms = selective_start.elapsed().as_secs_f64() * 1000.0 / windows.len() as f64;

        let reset_start = Instant::now();
        for window in windows {
            plain_losses.push(run_plain_loss(model, window, device));
            reset_losses.push(run_reset_loss(model, window, tool_tokens, device));
        }
        let reset_ms = reset_start.elapsed().as_secs_f64() * 1000.0 / (windows.len() * 2) as f64;

        let plain_loss = mean(&plain_losses);
        let carry_loss = mean(&carry_losses);
        let reset_loss = mean(&reset_losses);
        let rehearse_loss = mean(&rehearse_losses);
        let refresh_loss = mean(&refresh_losses);
        let selective_loss = mean(&selective_losses);
        let selective_tool_surprise = mean(&selective_tool_surprises);
        for idx in 0..windows.len() {
            let carry = carry_losses[idx];
            let reset = reset_losses[idx];
            let rehearse = rehearse_losses[idx];
            let refresh = refresh_losses[idx];
            let best = carry.min(reset.min(rehearse.min(refresh)));
            oracle_losses.push(best);
            if best + 1e-9 < carry {
                oracle_noncarry += 1;
            }
        }
        let oracle_loss = mean(&oracle_losses);

        ProbeCase {
            prefix_len,
            suffix_len,
            tool_tokens: tool_tokens.len(),
            plain_loss,
            carry_loss,
            reset_loss,
            rehearse_loss,
            refresh_loss,
            selective_loss,
            oracle_loss,
            carry_gain_vs_reset: reset_loss - carry_loss,
            rehearse_gain_vs_carry: carry_loss - rehearse_loss,
            refresh_gain_vs_reset: reset_loss - refresh_loss,
            selective_gain_vs_carry: carry_loss - selective_loss,
            oracle_gain_vs_carry: carry_loss - oracle_loss,
            carry_gap_to_plain: carry_loss - plain_loss,
            reset_gap_to_plain: reset_loss - plain_loss,
            rehearse_gap_to_plain: rehearse_loss - plain_loss,
            refresh_gap_to_plain: refresh_loss - plain_loss,
            selective_gap_to_plain: selective_loss - plain_loss,
            oracle_gap_to_plain: oracle_loss - plain_loss,
            carry_ms,
            reset_ms,
            rehearse_ms,
            refresh_ms,
            selective_ms,
            selective_refresh_rate: selective_refreshes as f64 / windows.len().max(1) as f64,
            selective_tool_surprise,
            oracle_noncarry_rate: oracle_noncarry as f64 / windows.len().max(1) as f64,
        }
    }

    fn run_plain_loss(model: &BDH<Backend>, window: &ToolBoundaryWindow, device: &Device) -> f64 {
        let inputs = tensor_from_i64(&window.full_inputs, device);
        let targets = tensor_from_i64(&window.full_targets, device);
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
        language_model_loss(logits, targets).into_scalar() as f64
    }

    fn run_carry_loss(
        model: &BDH<Backend>,
        window: &ToolBoundaryWindow,
        tool_tokens: &[i64],
        device: &Device,
    ) -> f64 {
        let (mut state, _) =
            prefill_state(model, &window.prefix_tokens, device).expect("prefill prefix state");
        if !tool_tokens.is_empty() {
            let tool_inputs = tensor_from_i64(tool_tokens, device);
            let _ = model.forward_with_state(tool_inputs, &mut state);
        }
        run_suffix_loss_with_state(
            model,
            &mut state,
            &window.suffix_inputs,
            &window.suffix_targets,
            device,
        )
    }

    fn run_reset_loss(
        model: &BDH<Backend>,
        window: &ToolBoundaryWindow,
        tool_tokens: &[i64],
        device: &Device,
    ) -> f64 {
        let mut state = if tool_tokens.is_empty() {
            model.init_state()
        } else {
            let (state, _) =
                prefill_state(model, tool_tokens, device).expect("prefill tool boundary state");
            state
        };
        run_suffix_loss_with_state(
            model,
            &mut state,
            &window.suffix_inputs,
            &window.suffix_targets,
            device,
        )
    }

    fn run_rehearse_loss(
        model: &BDH<Backend>,
        window: &ToolBoundaryWindow,
        tool_tokens: &[i64],
        rehearse_passes: usize,
        device: &Device,
    ) -> f64 {
        let (mut state, _) =
            prefill_state(model, &window.prefix_tokens, device).expect("prefill prefix state");
        if !tool_tokens.is_empty() {
            let tool_inputs = tensor_from_i64(tool_tokens, device);
            let _ = model.forward_with_state(tool_inputs.clone(), &mut state);
            for _ in 0..rehearse_passes {
                let _ = model.forward_with_state(tool_inputs.clone(), &mut state);
            }
        }
        run_suffix_loss_with_state(
            model,
            &mut state,
            &window.suffix_inputs,
            &window.suffix_targets,
            device,
        )
    }

    fn run_refresh_loss(
        model: &BDH<Backend>,
        window: &ToolBoundaryWindow,
        tool_tokens: &[i64],
        refresh_tail_len: usize,
        device: &Device,
    ) -> f64 {
        let mut refresh_tokens = window
            .prefix_tokens
            .iter()
            .rev()
            .take(refresh_tail_len.max(1))
            .copied()
            .collect::<Vec<_>>();
        refresh_tokens.reverse();
        refresh_tokens.extend_from_slice(tool_tokens);
        let mut state = if refresh_tokens.is_empty() {
            model.init_state()
        } else {
            let (state, _) = prefill_state(model, &refresh_tokens, device)
                .expect("prefill refresh boundary state");
            state
        };
        run_suffix_loss_with_state(
            model,
            &mut state,
            &window.suffix_inputs,
            &window.suffix_targets,
            device,
        )
    }

    struct SelectiveRefreshResult {
        loss: f64,
        refreshed: bool,
        tool_surprise: f64,
    }

    fn run_selective_refresh_loss(
        model: &BDH<Backend>,
        window: &ToolBoundaryWindow,
        tool_tokens: &[i64],
        refresh_tail_len: usize,
        selective_surprise_threshold: f64,
        device: &Device,
    ) -> SelectiveRefreshResult {
        if tool_tokens.is_empty() {
            return SelectiveRefreshResult {
                loss: run_carry_loss(model, window, tool_tokens, device),
                refreshed: false,
                tool_surprise: 0.0,
            };
        }

        let (mut carry_state, _) =
            prefill_state(model, &window.prefix_tokens, device).expect("prefill prefix state");
        let (tool_inputs, tool_targets) = boundary_tokens(window, tool_tokens);
        let mut score_state = carry_state.clone();
        let tool_surprise = run_suffix_loss_with_state(
            model,
            &mut score_state,
            &tool_inputs,
            &tool_targets,
            device,
        );

        let refreshed = tool_surprise > selective_surprise_threshold;
        let loss = if refreshed {
            run_refresh_loss(model, window, tool_tokens, refresh_tail_len, device)
        } else {
            let tool_inputs = tensor_from_i64(tool_tokens, device);
            let _ = model.forward_with_state(tool_inputs, &mut carry_state);
            run_suffix_loss_with_state(
                model,
                &mut carry_state,
                &window.suffix_inputs,
                &window.suffix_targets,
                device,
            )
        };

        SelectiveRefreshResult {
            loss,
            refreshed,
            tool_surprise,
        }
    }

    fn boundary_tokens(window: &ToolBoundaryWindow, tool_tokens: &[i64]) -> (Vec<i64>, Vec<i64>) {
        let mut inputs = Vec::with_capacity(tool_tokens.len());
        inputs.push(
            *window
                .prefix_tokens
                .last()
                .expect("prefix tokens required for boundary probe"),
        );
        if tool_tokens.len() > 1 {
            inputs.extend_from_slice(&tool_tokens[..tool_tokens.len() - 1]);
        }
        (inputs, tool_tokens.to_vec())
    }

    fn run_suffix_loss_with_state(
        model: &BDH<Backend>,
        state: &mut burn_dragon::language::ModelState<Backend>,
        suffix_inputs: &[i64],
        suffix_targets: &[i64],
        device: &Device,
    ) -> f64 {
        let inputs = tensor_from_i64(suffix_inputs, device);
        let targets = tensor_from_i64(suffix_targets, device);
        let logits = model.forward_with_state(inputs, state);
        language_model_loss(logits, targets).into_scalar() as f64
    }

    fn tensor_from_i64(values: &[i64], device: &Device) -> Tensor<Backend, 2, Int> {
        Tensor::<Backend, 2, Int>::from_data(
            TensorData::new(values.to_vec(), [1, values.len()]),
            device,
        )
    }

    fn mean(values: &[f64]) -> f64 {
        values.iter().sum::<f64>() / values.len().max(1) as f64
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(&mut out, "# Language Tool-Boundary Probe");
        let _ = writeln!(&mut out);
        let _ = writeln!(&mut out, "- checkpoint: `{}`", report.checkpoint.display());
        let _ = writeln!(&mut out, "- eval_cases: `{}`", report.eval_cases);
        let _ = writeln!(
            &mut out,
            "- tool_text: `{}`",
            report.tool_text.replace('\n', "\\n")
        );
        let _ = writeln!(
            &mut out,
            "- refresh_tail_len: `{}`",
            report.refresh_tail_len
        );
        let _ = writeln!(&mut out, "- rehearse_passes: `{}`", report.rehearse_passes);
        let _ = writeln!(
            &mut out,
            "- selective_surprise_threshold: `{:.4}`",
            report.selective_surprise_threshold
        );
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| prefix | suffix | tool_tokens | plain_loss | carry_loss | reset_loss | rehearse_loss | refresh_loss | selective_loss | oracle_loss | carry_gain_vs_reset | rehearse_gain_vs_carry | refresh_gain_vs_reset | selective_gain_vs_carry | oracle_gain_vs_carry | carry_gap_to_plain | reset_gap_to_plain | rehearse_gap_to_plain | refresh_gap_to_plain | selective_gap_to_plain | oracle_gap_to_plain | carry_ms | reset_ms | rehearse_ms | refresh_ms | selective_ms | selective_refresh_rate | selective_tool_surprise | oracle_noncarry_rate |"
        );
        let _ = writeln!(
            &mut out,
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        );
        for case in &report.cases {
            let _ = writeln!(
                &mut out,
                "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.4} | {:.4} | {:.4} |",
                case.prefix_len,
                case.suffix_len,
                case.tool_tokens,
                case.plain_loss,
                case.carry_loss,
                case.reset_loss,
                case.rehearse_loss,
                case.refresh_loss,
                case.selective_loss,
                case.oracle_loss,
                case.carry_gain_vs_reset,
                case.rehearse_gain_vs_carry,
                case.refresh_gain_vs_reset,
                case.selective_gain_vs_carry,
                case.oracle_gain_vs_carry,
                case.carry_gap_to_plain,
                case.reset_gap_to_plain,
                case.rehearse_gap_to_plain,
                case.refresh_gap_to_plain,
                case.selective_gap_to_plain,
                case.oracle_gap_to_plain,
                case.carry_ms,
                case.reset_ms,
                case.rehearse_ms,
                case.refresh_ms,
                case.selective_ms,
                case.selective_refresh_rate,
                case.selective_tool_surprise,
                case.oracle_noncarry_rate,
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
