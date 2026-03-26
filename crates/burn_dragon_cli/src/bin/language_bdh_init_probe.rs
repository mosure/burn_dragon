#![recursion_limit = "256"]

#[cfg(not(feature = "language-probe"))]
fn main() {
    panic!("language_bdh_init_probe requires --features language-probe");
}

#[cfg(feature = "language-probe")]
mod real {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result, anyhow};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
    use burn_autodiff::Autodiff;
    use burn_dragon::core::{
        BDH, BDHConfig, BdhFiringTargetKind, BdhInitializationConfig,
        LanguageBdhInitLayerDiagnostics,
    };
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_datasets;
    use burn_dragon::language::{
        apply_wgpu_fused_core_override, build_model_config_with_tokenizer, language_model_loss,
        load_language_core_from_checkpoint, load_training_config_for_checkpoint,
        summary_event_mask_tensor,
    };
    use burn_ndarray::NdArray;
    use clap::{ArgAction, Parser, ValueEnum};
    use serde::Serialize;

    #[cfg(feature = "language-cuda")]
    use burn_cuda::Cuda;

    const P_X_MIN: f64 = 0.05;
    const P_X_MAX: f64 = 0.35;
    const P_Y_MIN: f64 = 0.01;
    const P_Y_MAX: f64 = 0.10;
    const R_RES_MIN: f64 = 0.05;
    const R_RES_MAX: f64 = 0.50;

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize)]
    #[serde(rename_all = "snake_case")]
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

    #[derive(Clone, Copy, Debug, Eq, PartialEq, ValueEnum, Serialize)]
    #[serde(rename_all = "snake_case")]
    enum CalibrationArg {
        Disabled,
        FiringThresholds,
        LsuvBdh,
    }

    #[derive(Parser, Debug)]
    #[command(name = "language_bdh_init_probe")]
    struct Args {
        #[arg(long)]
        config: Vec<PathBuf>,
        #[arg(long)]
        checkpoint: Option<PathBuf>,
        #[arg(long, value_enum, default_value_t = BackendArg::Cuda)]
        backend: BackendArg,
        #[arg(long, default_value_t = 4)]
        eval_batches: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long)]
        block_size: Option<usize>,
        #[arg(long, action = ArgAction::Set, default_value_t = true)]
        carry_state: bool,
        #[arg(long, value_enum, default_value_t = CalibrationArg::Disabled)]
        calibration: CalibrationArg,
        #[arg(long, default_value_t = 2)]
        calibration_rounds: usize,
        #[arg(long, default_value_t = 2)]
        calibration_batches: usize,
        #[arg(long, default_value_t = 12)]
        calibration_search_steps: usize,
        #[arg(long, default_value_t = 0.01)]
        calibration_tolerance: f64,
        #[arg(long, default_value_t = 0.10)]
        target_r_res: f64,
        #[arg(long, default_value_t = 8.0)]
        max_residual_gain: f64,
        #[arg(long, default_value_t = 8)]
        backward_steps: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
        #[arg(long)]
        resolved_init_toml_path: Option<PathBuf>,
    }

    #[derive(Clone, Debug)]
    struct EvalBatchData {
        inputs: Vec<i64>,
        targets: Vec<i64>,
        batch_size: usize,
        block_size: usize,
    }

    impl EvalBatchData {
        fn inputs_tensor<B: BackendTrait>(&self, device: &B::Device) -> Tensor<B, 2, Int> {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(self.inputs.clone(), [self.batch_size, self.block_size]),
                device,
            )
        }

        fn targets_tensor<B: BackendTrait>(&self, device: &B::Device) -> Tensor<B, 2, Int> {
            Tensor::<B, 2, Int>::from_data(
                TensorData::new(self.targets.clone(), [self.batch_size, self.block_size]),
                device,
            )
        }
    }

    #[derive(Clone, Debug, Default)]
    struct DiagnosticsAccumulator {
        count: usize,
        lowrank_active_count: usize,
        finite_count: usize,
        p_x_sum: f64,
        p_x_count: usize,
        p_y_sum: f64,
        p_y_count: usize,
        current_rms_sum: f64,
        current_rms_count: usize,
        recurrent_readout_rms_sum: f64,
        recurrent_readout_rms_count: usize,
        recurrent_readout_ratio_sum: f64,
        recurrent_readout_ratio_count: usize,
        residual_delta_rms_sum: f64,
        residual_delta_rms_count: usize,
        r_res_sum: f64,
        r_res_count: usize,
    }

    #[derive(Clone, Copy, Debug, Default, Serialize)]
    struct ProbeMetricSummary {
        finite: bool,
        p_x: Option<f64>,
        p_y: Option<f64>,
        r_res: Option<f64>,
        recurrent_readout_ratio: Option<f64>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct BackwardSummary {
        enabled: bool,
        finite: bool,
        completed_steps: usize,
        final_loss: Option<f64>,
    }

    #[derive(Clone, Debug, Serialize)]
    struct CalibrationSummary {
        enabled: bool,
        kind: CalibrationArg,
        rounds: usize,
        calibration_batches: usize,
        search_steps: usize,
        tolerance: Option<f64>,
        target_r_res: Option<f64>,
        final_x_threshold: Option<f64>,
        final_y_threshold: Option<f64>,
        final_residual_gain: Option<f64>,
        post_calibration: Option<ProbeMetricSummary>,
    }

    impl CalibrationSummary {
        fn disabled() -> Self {
            Self {
                enabled: false,
                kind: CalibrationArg::Disabled,
                rounds: 0,
                calibration_batches: 0,
                search_steps: 0,
                tolerance: None,
                target_r_res: None,
                final_x_threshold: None,
                final_y_threshold: None,
                final_residual_gain: None,
                post_calibration: None,
            }
        }
    }

    #[derive(Clone, Serialize)]
    struct Phase0Criteria {
        p_x_band: [f64; 2],
        p_y_band: [f64; 2],
        r_res_band: [f64; 2],
        sparse_positive_pass: bool,
        backward_finite_pass: bool,
        phase0_pass: bool,
        passing_layers: Vec<usize>,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        config: Vec<PathBuf>,
        checkpoint: Option<PathBuf>,
        backend: &'static str,
        batch_size: usize,
        block_size: usize,
        eval_batches: usize,
        carry_state: bool,
        initialization: BdhInitializationConfig,
        effective_initialization: BdhInitializationConfig,
        avg_loss: f64,
        metrics: ProbeMetricSummary,
        layers: Vec<LanguageBdhInitLayerDiagnostics>,
        calibration: CalibrationSummary,
        backward: BackwardSummary,
        phase0: Phase0Criteria,
    }

    #[derive(Clone, Copy)]
    enum BranchCalibrationTarget {
        X,
        Y,
    }

    impl BranchCalibrationTarget {
        fn target(self, init: &BdhInitializationConfig) -> f64 {
            match self {
                Self::X => init.firing_targets.x_target,
                Self::Y => init.firing_targets.y_target,
            }
        }

        fn metric(self, metrics: &ProbeMetricSummary) -> Result<f64> {
            match self {
                Self::X => metrics
                    .p_x
                    .ok_or_else(|| anyhow!("missing p_x during firing-target calibration")),
                Self::Y => metrics
                    .p_y
                    .ok_or_else(|| anyhow!("missing p_y during firing-target calibration")),
            }
        }
    }

    pub fn main() {
        if let Err(err) = run() {
            eprintln!("error: {err:#}");
            std::process::exit(1);
        }
    }

    fn run() -> Result<()> {
        let args = Args::parse();
        if args.calibration != CalibrationArg::Disabled && args.checkpoint.is_some() {
            return Err(anyhow!(
                "--calibration requires an untrained config path and is not valid with --checkpoint"
            ));
        }
        if !args.target_r_res.is_finite() || args.target_r_res <= 0.0 {
            return Err(anyhow!(
                "--target-r-res must be finite and > 0 (got {})",
                args.target_r_res
            ));
        }
        if !args.max_residual_gain.is_finite() || args.max_residual_gain <= 0.0 {
            return Err(anyhow!(
                "--max-residual-gain must be finite and > 0 (got {})",
                args.max_residual_gain
            ));
        }
        if !args.calibration_tolerance.is_finite() || args.calibration_tolerance < 0.0 {
            return Err(anyhow!(
                "--calibration-tolerance must be finite and >= 0 (got {})",
                args.calibration_tolerance
            ));
        }
        if args.calibration_search_steps == 0 {
            return Err(anyhow!("--calibration-search-steps must be >= 1 (got 0)"));
        }

        match args.backend {
            BackendArg::Ndarray => run_backend::<NdArray<f32>>(&args),
            BackendArg::Cuda => {
                #[cfg(feature = "language-cuda")]
                {
                    run_backend::<Cuda<f32>>(&args)
                }
                #[cfg(not(feature = "language-cuda"))]
                {
                    Err(anyhow!(
                        "cuda backend selected but this build lacks `language-cuda` feature"
                    ))
                }
            }
        }
    }

    fn run_backend<B: BackendTrait>(args: &Args) -> Result<()> {
        let backend_name = args.backend.as_backend_name();
        let config = load_training_config_for_checkpoint(
            &args.config,
            args.checkpoint.as_ref(),
            backend_name,
        )
        .context("load language training config for BDH init probe")?;
        let datasets = prepare_datasets(&config.dataset, &config.training)
            .context("prepare language datasets")?;
        let tokenizer = datasets.train.tokenizer();
        let mut model_config = build_model_config_with_tokenizer(
            &config.model,
            config.training.block_size,
            tokenizer.as_ref(),
        )
        .context("build model config with tokenizer")?;
        apply_wgpu_fused_core_override(
            &mut model_config,
            backend_name,
            config.wgpu.training.fused_core_recurrent,
            config.wgpu.training.fused_core_rollout,
        );

        let batch_size = args.batch_size.unwrap_or(config.training.batch_size).max(1);
        let block_size = args.block_size.unwrap_or(config.training.block_size).max(1);
        let summary_event_token_ids = model_config
            .summary_memory
            .write_trigger_token_ids
            .as_ref()
            .map(|ids| ids.to_vec());
        let eval_batches = build_eval_batches(
            datasets.valid.as_ref(),
            DatasetSplit::Val,
            batch_size,
            block_size,
            args.eval_batches.max(1),
        )?;
        let device = B::Device::default();

        let requested_initialization = model_config.initialization.clone();
        let (effective_initialization, calibration) = if args.checkpoint.is_none() {
            calibrate_initialization::<B>(
                args,
                &model_config,
                config.training.seed,
                &eval_batches,
                summary_event_token_ids.as_deref(),
                &device,
            )?
        } else {
            (
                requested_initialization.clone(),
                CalibrationSummary::disabled(),
            )
        };

        model_config.initialization = effective_initialization.clone();
        let model = if let Some(checkpoint) = args.checkpoint.as_ref() {
            load_language_core_from_checkpoint::<B>(
                checkpoint,
                None,
                &args.config,
                backend_name,
                &device,
            )
            .with_context(|| format!("load checkpoint {}", checkpoint.display()))?
        } else {
            build_untrained_model::<B>(&model_config, config.training.seed, &device)
        };

        let (layers, avg_loss, metrics) = probe_model(
            args.carry_state,
            &model,
            &eval_batches,
            summary_event_token_ids.as_deref(),
            &device,
        );
        let passing_layers = layers
            .iter()
            .filter(|diag| layer_passes_phase0(diag))
            .map(|diag| diag.layer_index)
            .collect::<Vec<_>>();
        let backward = if args.checkpoint.is_some() {
            BackwardSummary::default()
        } else {
            run_backward_check::<B>(
                args,
                &model_config,
                config.training.seed,
                &eval_batches,
                summary_event_token_ids.as_deref(),
            )?
        };
        let sparse_positive_pass = !passing_layers.is_empty();
        let backward_finite_pass = !backward.enabled || backward.finite;
        let report = Report {
            benchmark: "burn_dragon language BDH init probe",
            config: args.config.clone(),
            checkpoint: args.checkpoint.clone(),
            backend: backend_name,
            batch_size,
            block_size,
            eval_batches: eval_batches.len(),
            carry_state: args.carry_state,
            initialization: requested_initialization,
            effective_initialization: effective_initialization.clone(),
            avg_loss,
            metrics,
            layers,
            calibration,
            backward,
            phase0: Phase0Criteria {
                p_x_band: [P_X_MIN, P_X_MAX],
                p_y_band: [P_Y_MIN, P_Y_MAX],
                r_res_band: [R_RES_MIN, R_RES_MAX],
                sparse_positive_pass,
                backward_finite_pass,
                phase0_pass: sparse_positive_pass && backward_finite_pass,
                passing_layers,
            },
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).context("serialize probe report")?;
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact")?;
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact")?;
        }
        if let Some(path) = args.resolved_init_toml_path.as_ref() {
            write_text_artifact(
                path,
                &format_initialization_override_toml(&effective_initialization),
                "resolved init override",
            )?;
        }

        Ok(())
    }

    fn build_eval_batches(
        dataset: &Dataset,
        split: DatasetSplit,
        batch_size: usize,
        block_size: usize,
        num_batches: usize,
    ) -> Result<Vec<EvalBatchData>> {
        let (offset, span) = dataset.split_offset_and_span(split);
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
                let mut token_buffer = vec![0u32; required];
                dataset.copy_token_range(start, &mut token_buffer);
                for t in 0..block_size {
                    let idx = batch_idx * block_size + t;
                    inputs[idx] = token_buffer[t] as i64;
                    targets[idx] = token_buffer[t + 1] as i64;
                }
            }
            batches.push(EvalBatchData {
                inputs,
                targets,
                batch_size,
                block_size,
            });
        }

        Ok(batches)
    }

    fn build_untrained_model<B: BackendTrait>(
        model_config: &BDHConfig,
        seed: u64,
        device: &B::Device,
    ) -> BDH<B> {
        let mut config = model_config.clone();
        B::seed(device, seed);
        config.vocab_size = config.vocab_size.max(1);
        BDH::<B>::new(config, device)
    }

    fn summary_event_mask_for_batch<B: BackendTrait>(
        batch: &EvalBatchData,
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Option<Tensor<B, 2, Int>> {
        summary_event_mask_tensor::<B>(
            &batch.inputs,
            batch.batch_size,
            batch.block_size,
            summary_event_token_ids,
            device,
        )
    }

    fn probe_model<B: BackendTrait>(
        carry_state: bool,
        model: &BDH<B>,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> (
        Vec<LanguageBdhInitLayerDiagnostics>,
        f64,
        ProbeMetricSummary,
    ) {
        let mut diagnostics = BTreeMap::<usize, DiagnosticsAccumulator>::new();
        let mut loss_sum = 0.0f64;
        let mut state = model.init_state();

        for batch in eval_batches {
            let inputs = batch.inputs_tensor::<B>(device);
            let targets = batch.targets_tensor::<B>(device);
            let summary_event_mask =
                summary_event_mask_for_batch::<B>(batch, summary_event_token_ids, device);
            let logits = if let Some(mask) = summary_event_mask.clone() {
                model.forward_with_summary_event_mask(inputs.clone(), mask)
            } else {
                model.forward(inputs.clone())
            };
            loss_sum += language_model_loss::<B>(logits, targets)
                .into_scalar()
                .elem::<f64>();

            let batch_diagnostics = match (carry_state, summary_event_mask) {
                (true, Some(mask)) => model
                    .collect_language_bdh_init_diagnostics_with_state_and_summary_event_mask(
                        inputs, mask, &mut state,
                    ),
                (true, None) => {
                    model.collect_language_bdh_init_diagnostics_with_state(inputs, &mut state)
                }
                (false, Some(mask)) => model
                    .collect_language_bdh_init_diagnostics_with_summary_event_mask(inputs, mask),
                (false, None) => model.collect_language_bdh_init_diagnostics(inputs),
            };
            accumulate_diagnostics(&mut diagnostics, &batch_diagnostics);
        }

        let layers = finalize_diagnostics(diagnostics);
        let metrics = summarize_probe_metrics(&layers);
        (layers, loss_sum / eval_batches.len().max(1) as f64, metrics)
    }

    fn run_backward_check<B: BackendTrait>(
        args: &Args,
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
    ) -> Result<BackwardSummary> {
        if args.backward_steps == 0 {
            return Ok(BackwardSummary::default());
        }

        let device = <Autodiff<B> as BackendTrait>::Device::default();
        let model = build_untrained_model::<Autodiff<B>>(model_config, seed, &device);
        let mut state = model.init_state();
        let mut last_loss = None;

        for step in 0..args.backward_steps {
            let batch = &eval_batches[step % eval_batches.len().max(1)];
            let inputs = batch.inputs_tensor::<Autodiff<B>>(&device);
            let targets = batch.targets_tensor::<Autodiff<B>>(&device);
            let summary_event_mask = summary_event_mask_for_batch::<Autodiff<B>>(
                batch,
                summary_event_token_ids,
                &device,
            );
            let logits = if args.carry_state {
                if let Some(mask) = summary_event_mask {
                    model.forward_with_state_and_summary_event_mask(inputs, mask, &mut state)
                } else {
                    model.forward_with_state(inputs, &mut state)
                }
            } else if let Some(mask) = summary_event_mask {
                model.forward_with_summary_event_mask(inputs, mask)
            } else {
                model.forward(inputs)
            };
            let loss = language_model_loss::<Autodiff<B>>(logits, targets);
            let loss_value = loss.clone().into_scalar().elem::<f64>();
            if !loss_value.is_finite() {
                return Ok(BackwardSummary {
                    enabled: true,
                    finite: false,
                    completed_steps: step,
                    final_loss: Some(loss_value),
                });
            }
            let _ = loss.backward();
            last_loss = Some(loss_value);
        }

        Ok(BackwardSummary {
            enabled: true,
            finite: true,
            completed_steps: args.backward_steps,
            final_loss: last_loss,
        })
    }

    fn calibrate_initialization<B: BackendTrait>(
        args: &Args,
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        match args.calibration {
            CalibrationArg::Disabled => Ok((
                model_config.initialization.clone(),
                CalibrationSummary::disabled(),
            )),
            CalibrationArg::FiringThresholds => calibrate_firing_thresholds_only::<B>(
                args,
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                device,
            ),
            CalibrationArg::LsuvBdh => calibrate_lsuv_bdh::<B>(
                args,
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                device,
            ),
        }
    }

    fn calibrate_firing_thresholds_only<B: BackendTrait>(
        args: &Args,
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        let init = &model_config.initialization;
        let calibration_batches = args
            .calibration_batches
            .max(1)
            .min(eval_batches.len().max(1));
        let calibration_slice = &eval_batches[..calibration_batches];
        let mut x_threshold = init.firing_targets.x_threshold;
        let mut y_threshold = init.firing_targets.y_threshold;

        for _ in 0..args.calibration_rounds.max(1) {
            x_threshold = calibrate_branch_threshold::<B>(
                model_config,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
                BranchCalibrationTarget::X,
                args.calibration_search_steps,
                args.calibration_tolerance,
                x_threshold,
                y_threshold,
                init.residual_scaling.gain,
            )?;
            y_threshold = calibrate_branch_threshold::<B>(
                model_config,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
                BranchCalibrationTarget::Y,
                args.calibration_search_steps,
                args.calibration_tolerance,
                x_threshold,
                y_threshold,
                init.residual_scaling.gain,
            )?;
        }

        let mut calibrated = init.clone();
        calibrated.firing_targets.kind = BdhFiringTargetKind::ExplicitThresholds;
        calibrated.firing_targets.x_threshold = x_threshold;
        calibrated.firing_targets.y_threshold = y_threshold;
        calibrated.validate().map_err(anyhow::Error::msg)?;

        let post_calibration = mean_probe_metrics_for_config::<B>(
            model_config,
            &calibrated,
            seed,
            calibration_slice,
            summary_event_token_ids,
            args.carry_state,
            device,
        );

        Ok((
            calibrated.clone(),
            CalibrationSummary {
                enabled: true,
                kind: CalibrationArg::FiringThresholds,
                rounds: args.calibration_rounds.max(1),
                calibration_batches,
                search_steps: args.calibration_search_steps,
                tolerance: Some(args.calibration_tolerance),
                target_r_res: None,
                final_x_threshold: Some(x_threshold),
                final_y_threshold: Some(y_threshold),
                final_residual_gain: Some(calibrated.residual_scaling.gain),
                post_calibration: Some(post_calibration),
            },
        ))
    }

    fn calibrate_lsuv_bdh<B: BackendTrait>(
        args: &Args,
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> Result<(BdhInitializationConfig, CalibrationSummary)> {
        let init = &model_config.initialization;
        let calibration_batches = args
            .calibration_batches
            .max(1)
            .min(eval_batches.len().max(1));
        let calibration_slice = &eval_batches[..calibration_batches];
        let mut x_threshold = init.firing_targets.x_threshold;
        let mut y_threshold = init.firing_targets.y_threshold;
        let mut residual_gain = init.residual_scaling.gain.max(1.0e-6);
        let mut best_init =
            candidate_initialization(init, x_threshold, y_threshold, residual_gain)?;
        let mut best_metrics = mean_probe_metrics_for_config::<B>(
            model_config,
            &best_init,
            seed,
            calibration_slice,
            summary_event_token_ids,
            args.carry_state,
            device,
        );
        let mut best_error = phase0_band_error(&best_metrics);

        for _ in 0..(args.calibration_rounds.max(1) + 1) {
            x_threshold = calibrate_branch_threshold::<B>(
                model_config,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
                BranchCalibrationTarget::X,
                args.calibration_search_steps,
                args.calibration_tolerance,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            y_threshold = calibrate_branch_threshold::<B>(
                model_config,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
                BranchCalibrationTarget::Y,
                args.calibration_search_steps,
                args.calibration_tolerance,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            residual_gain = calibrate_residual_gain::<B>(
                args,
                model_config,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
                x_threshold,
                y_threshold,
                residual_gain,
            )?;
            let candidate =
                candidate_initialization(init, x_threshold, y_threshold, residual_gain)?;
            let candidate_metrics = mean_probe_metrics_for_config::<B>(
                model_config,
                &candidate,
                seed,
                calibration_slice,
                summary_event_token_ids,
                args.carry_state,
                device,
            );
            let candidate_error = phase0_band_error(&candidate_metrics);
            if candidate_error < best_error {
                best_error = candidate_error;
                best_init = candidate;
                best_metrics = candidate_metrics;
            }
            if metrics_pass_phase0_bands(&best_metrics) {
                break;
            }
        }

        Ok((
            best_init.clone(),
            CalibrationSummary {
                enabled: true,
                kind: CalibrationArg::LsuvBdh,
                rounds: args.calibration_rounds.max(1),
                calibration_batches,
                search_steps: args.calibration_search_steps,
                tolerance: Some(args.calibration_tolerance),
                target_r_res: Some(args.target_r_res),
                final_x_threshold: Some(best_init.firing_targets.x_threshold),
                final_y_threshold: Some(best_init.firing_targets.y_threshold),
                final_residual_gain: Some(best_init.residual_scaling.gain),
                post_calibration: Some(best_metrics),
            },
        ))
    }

    fn candidate_initialization(
        init: &BdhInitializationConfig,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<BdhInitializationConfig> {
        let mut calibration_init = init.clone();
        calibration_init.firing_targets.kind = BdhFiringTargetKind::ExplicitThresholds;
        calibration_init.firing_targets.x_threshold = x_threshold;
        calibration_init.firing_targets.y_threshold = y_threshold;
        calibration_init.residual_scaling.gain = residual_gain;
        calibration_init.validate().map_err(anyhow::Error::msg)?;
        Ok(calibration_init)
    }

    fn calibrate_branch_threshold<B: BackendTrait>(
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        carry_state: bool,
        device: &B::Device,
        branch: BranchCalibrationTarget,
        search_steps: usize,
        tolerance: f64,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<f64> {
        let init = &model_config.initialization;
        let target = branch.target(init);
        let current_metrics = mean_probe_metrics_for_candidate::<B>(
            model_config,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
            x_threshold,
            y_threshold,
            residual_gain,
        )?;
        let current_metric = branch.metric(&current_metrics)?;
        if (current_metric - target).abs() <= tolerance {
            return Ok(match branch {
                BranchCalibrationTarget::X => x_threshold,
                BranchCalibrationTarget::Y => y_threshold,
            });
        }
        let zero_metrics = mean_probe_metrics_for_candidate::<B>(
            model_config,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
            if matches!(branch, BranchCalibrationTarget::X) {
                0.0
            } else {
                x_threshold
            },
            if matches!(branch, BranchCalibrationTarget::Y) {
                0.0
            } else {
                y_threshold
            },
            residual_gain,
        )?;
        let zero_metric = branch.metric(&zero_metrics)?;
        if zero_metric <= target {
            return Ok(0.0);
        }

        let mut low = 0.0;
        let mut high = match branch {
            BranchCalibrationTarget::X => x_threshold.max(1.0e-4),
            BranchCalibrationTarget::Y => y_threshold.max(1.0e-4),
        };
        let mut high_metric = branch.metric(&mean_probe_metrics_for_candidate::<B>(
            model_config,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
            if matches!(branch, BranchCalibrationTarget::X) {
                high
            } else {
                x_threshold
            },
            if matches!(branch, BranchCalibrationTarget::Y) {
                high
            } else {
                y_threshold
            },
            residual_gain,
        )?)?;

        while high_metric > target && high < 1.0e3 {
            high *= 2.0;
            high_metric = branch.metric(&mean_probe_metrics_for_candidate::<B>(
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                carry_state,
                device,
                if matches!(branch, BranchCalibrationTarget::X) {
                    high
                } else {
                    x_threshold
                },
                if matches!(branch, BranchCalibrationTarget::Y) {
                    high
                } else {
                    y_threshold
                },
                residual_gain,
            )?)?;
        }

        for _ in 0..search_steps.max(1) {
            let mid = 0.5 * (low + high);
            let metric = branch.metric(&mean_probe_metrics_for_candidate::<B>(
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                carry_state,
                device,
                if matches!(branch, BranchCalibrationTarget::X) {
                    mid
                } else {
                    x_threshold
                },
                if matches!(branch, BranchCalibrationTarget::Y) {
                    mid
                } else {
                    y_threshold
                },
                residual_gain,
            )?)?;
            if metric > target {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok(high)
    }

    fn calibrate_residual_gain<B: BackendTrait>(
        args: &Args,
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        carry_state: bool,
        device: &B::Device,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<f64> {
        let min_gain = 1.0e-6;
        let target = args.target_r_res;
        let search_steps = args.calibration_search_steps.max(1);
        let current_r_res = mean_probe_metrics_for_candidate::<B>(
            model_config,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
            x_threshold,
            y_threshold,
            residual_gain,
        )?
        .r_res
        .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
        if (current_r_res - target).abs() <= args.calibration_tolerance {
            return Ok(residual_gain.max(min_gain));
        }

        let scaled_gain = if current_r_res <= min_gain {
            args.max_residual_gain
        } else {
            (residual_gain * (target / current_r_res)).clamp(min_gain, args.max_residual_gain)
        };
        let scaled_r_res = mean_probe_metrics_for_candidate::<B>(
            model_config,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
            x_threshold,
            y_threshold,
            scaled_gain,
        )?
        .r_res
        .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
        if (scaled_r_res - target).abs() <= args.calibration_tolerance || search_steps == 1 {
            return Ok(scaled_gain.max(min_gain));
        }

        let (mut low, mut high) = if current_r_res < target {
            let low = residual_gain.max(min_gain);
            let mut high = scaled_gain.max(low);
            let mut high_r_res = scaled_r_res;
            while high_r_res < target && high < args.max_residual_gain {
                high = (high * 2.0).min(args.max_residual_gain);
                high_r_res = mean_probe_metrics_for_candidate::<B>(
                    model_config,
                    seed,
                    eval_batches,
                    summary_event_token_ids,
                    carry_state,
                    device,
                    x_threshold,
                    y_threshold,
                    high,
                )?
                .r_res
                .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
                if (high - low).abs() <= f64::EPSILON {
                    break;
                }
            }
            if high_r_res < target {
                return Ok(high);
            }
            (low, high)
        } else {
            let mut high = residual_gain.max(min_gain);
            let mut low = (high * 0.5).max(min_gain);
            let mut low_r_res = mean_probe_metrics_for_candidate::<B>(
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                carry_state,
                device,
                x_threshold,
                y_threshold,
                low,
            )?
            .r_res
            .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
            while low_r_res > target && low > min_gain {
                high = low;
                low = (low * 0.5).max(min_gain);
                low_r_res = mean_probe_metrics_for_candidate::<B>(
                    model_config,
                    seed,
                    eval_batches,
                    summary_event_token_ids,
                    carry_state,
                    device,
                    x_threshold,
                    y_threshold,
                    low,
                )?
                .r_res
                .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
                if (high - low).abs() <= f64::EPSILON {
                    break;
                }
            }
            if low_r_res > target {
                return Ok(low);
            }
            (low, high)
        };

        for _ in 0..(search_steps - 1) {
            let mid = (low * high).sqrt().max(min_gain);
            let mid_r_res = mean_probe_metrics_for_candidate::<B>(
                model_config,
                seed,
                eval_batches,
                summary_event_token_ids,
                carry_state,
                device,
                x_threshold,
                y_threshold,
                mid,
            )?
            .r_res
            .ok_or_else(|| anyhow!("missing r_res during LSUV-BDH calibration"))?;
            if mid_r_res < target {
                low = mid;
            } else {
                high = mid;
            }
        }

        Ok(high.max(min_gain))
    }

    fn mean_probe_metrics_for_candidate<B: BackendTrait>(
        model_config: &BDHConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        carry_state: bool,
        device: &B::Device,
        x_threshold: f64,
        y_threshold: f64,
        residual_gain: f64,
    ) -> Result<ProbeMetricSummary> {
        let calibration_init = candidate_initialization(
            &model_config.initialization,
            x_threshold,
            y_threshold,
            residual_gain,
        )?;
        Ok(mean_probe_metrics_for_config::<B>(
            model_config,
            &calibration_init,
            seed,
            eval_batches,
            summary_event_token_ids,
            carry_state,
            device,
        ))
    }

    fn mean_probe_metrics_for_config<B: BackendTrait>(
        model_config: &BDHConfig,
        initialization: &BdhInitializationConfig,
        seed: u64,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        carry_state: bool,
        device: &B::Device,
    ) -> ProbeMetricSummary {
        let mut candidate_config = model_config.clone();
        candidate_config.initialization = initialization.clone();
        let model = build_untrained_model::<B>(&candidate_config, seed, device);
        let (_, _, metrics) = probe_model(
            carry_state,
            &model,
            eval_batches,
            summary_event_token_ids,
            device,
        );
        metrics
    }

    fn accumulate_diagnostics(
        accumulators: &mut BTreeMap<usize, DiagnosticsAccumulator>,
        diagnostics: &[LanguageBdhInitLayerDiagnostics],
    ) {
        for diag in diagnostics {
            let accumulator = accumulators.entry(diag.layer_index).or_default();
            accumulator.count += 1;
            accumulator.lowrank_active_count += usize::from(diag.lowrank_path_active);
            accumulator.finite_count += usize::from(diag.finite);
            if let Some(value) = diag.p_x {
                accumulator.p_x_sum += value;
                accumulator.p_x_count += 1;
            }
            if let Some(value) = diag.p_y {
                accumulator.p_y_sum += value;
                accumulator.p_y_count += 1;
            }
            if let Some(value) = diag.current_rms {
                accumulator.current_rms_sum += value;
                accumulator.current_rms_count += 1;
            }
            if let Some(value) = diag.recurrent_readout_rms {
                accumulator.recurrent_readout_rms_sum += value;
                accumulator.recurrent_readout_rms_count += 1;
            }
            if let Some(value) = diag.recurrent_readout_ratio {
                accumulator.recurrent_readout_ratio_sum += value;
                accumulator.recurrent_readout_ratio_count += 1;
            }
            if let Some(value) = diag.residual_delta_rms {
                accumulator.residual_delta_rms_sum += value;
                accumulator.residual_delta_rms_count += 1;
            }
            if let Some(value) = diag.r_res {
                accumulator.r_res_sum += value;
                accumulator.r_res_count += 1;
            }
        }
    }

    fn finalize_diagnostics(
        accumulators: BTreeMap<usize, DiagnosticsAccumulator>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        accumulators
            .into_iter()
            .map(
                |(layer_index, accumulator)| LanguageBdhInitLayerDiagnostics {
                    layer_index,
                    lowrank_path_active: accumulator.lowrank_active_count * 2 >= accumulator.count,
                    finite: accumulator.finite_count == accumulator.count,
                    p_x: mean_optional(accumulator.p_x_sum, accumulator.p_x_count),
                    p_y: mean_optional(accumulator.p_y_sum, accumulator.p_y_count),
                    current_rms: mean_optional(
                        accumulator.current_rms_sum,
                        accumulator.current_rms_count,
                    ),
                    recurrent_readout_rms: mean_optional(
                        accumulator.recurrent_readout_rms_sum,
                        accumulator.recurrent_readout_rms_count,
                    ),
                    recurrent_readout_ratio: mean_optional(
                        accumulator.recurrent_readout_ratio_sum,
                        accumulator.recurrent_readout_ratio_count,
                    ),
                    residual_delta_rms: mean_optional(
                        accumulator.residual_delta_rms_sum,
                        accumulator.residual_delta_rms_count,
                    ),
                    r_res: mean_optional(accumulator.r_res_sum, accumulator.r_res_count),
                },
            )
            .collect()
    }

    fn summarize_probe_metrics(layers: &[LanguageBdhInitLayerDiagnostics]) -> ProbeMetricSummary {
        ProbeMetricSummary {
            finite: !layers.is_empty() && layers.iter().all(|layer| layer.finite),
            p_x: mean_from_values(layers.iter().filter_map(|layer| layer.p_x)),
            p_y: mean_from_values(layers.iter().filter_map(|layer| layer.p_y)),
            r_res: mean_from_values(layers.iter().filter_map(|layer| layer.r_res)),
            recurrent_readout_ratio: mean_from_values(
                layers
                    .iter()
                    .filter_map(|layer| layer.recurrent_readout_ratio),
            ),
        }
    }

    fn mean_optional(sum: f64, count: usize) -> Option<f64> {
        (count > 0).then_some(sum / count as f64)
    }

    fn mean_from_values(values: impl Iterator<Item = f64>) -> Option<f64> {
        let mut sum = 0.0;
        let mut count = 0usize;
        for value in values {
            sum += value;
            count += 1;
        }
        mean_optional(sum, count)
    }

    fn metric_band_error(value: Option<f64>, min: f64, max: f64) -> f64 {
        match value {
            Some(value) if value < min => min - value,
            Some(value) if value > max => value - max,
            Some(_) => 0.0,
            None => 1.0,
        }
    }

    fn phase0_band_error(metrics: &ProbeMetricSummary) -> f64 {
        metric_band_error(metrics.p_x, P_X_MIN, P_X_MAX)
            + metric_band_error(metrics.p_y, P_Y_MIN, P_Y_MAX)
            + metric_band_error(metrics.r_res, R_RES_MIN, R_RES_MAX)
            + if metrics.finite { 0.0 } else { 1.0 }
    }

    fn metrics_pass_phase0_bands(metrics: &ProbeMetricSummary) -> bool {
        metrics.finite
            && metrics
                .p_x
                .is_some_and(|value| (P_X_MIN..=P_X_MAX).contains(&value))
            && metrics
                .p_y
                .is_some_and(|value| (P_Y_MIN..=P_Y_MAX).contains(&value))
            && metrics
                .r_res
                .is_some_and(|value| (R_RES_MIN..=R_RES_MAX).contains(&value))
    }

    fn layer_passes_phase0(diag: &LanguageBdhInitLayerDiagnostics) -> bool {
        diag.lowrank_path_active
            && diag.finite
            && diag
                .p_x
                .is_some_and(|value| (P_X_MIN..=P_X_MAX).contains(&value))
            && diag
                .p_y
                .is_some_and(|value| (P_Y_MIN..=P_Y_MAX).contains(&value))
            && diag
                .r_res
                .is_some_and(|value| (R_RES_MIN..=R_RES_MAX).contains(&value))
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "# Language BDH Init Probe");
        let _ = writeln!(out);
        let _ = writeln!(out, "- backend: {}", report.backend);
        let _ = writeln!(out, "- batch size: {}", report.batch_size);
        let _ = writeln!(out, "- block size: {}", report.block_size);
        let _ = writeln!(out, "- eval batches: {}", report.eval_batches);
        let _ = writeln!(out, "- carry state: {}", report.carry_state);
        let _ = writeln!(out, "- avg loss: {:.6}", report.avg_loss);
        let _ = writeln!(
            out,
            "- initialization: {:?}",
            report.effective_initialization.kind
        );
        let _ = writeln!(
            out,
            "- residual scaling: {:?}",
            report.effective_initialization.residual_scaling.kind
        );
        let _ = writeln!(
            out,
            "- neuron gains: {:?}",
            report.effective_initialization.neuron_gains.kind
        );
        let _ = writeln!(
            out,
            "- topology prior: {:?}",
            report.effective_initialization.topology_prior.kind
        );
        let _ = writeln!(
            out,
            "- firing targets: {:?}",
            report.effective_initialization.firing_targets.kind
        );
        if report.calibration.enabled {
            let _ = writeln!(
                out,
                "- calibration: kind={:?}, rounds={}, batches={}, x_threshold={:.6}, y_threshold={:.6}, residual_gain={:.6}",
                report.calibration.kind,
                report.calibration.rounds,
                report.calibration.calibration_batches,
                report.calibration.final_x_threshold.unwrap_or_default(),
                report.calibration.final_y_threshold.unwrap_or_default(),
                report.calibration.final_residual_gain.unwrap_or_default(),
            );
        }
        let _ = writeln!(
            out,
            "- mean metrics: finite={}, p_x={:.6}, p_y={:.6}, r_res={:.6}, readout_ratio={:.6}",
            report.metrics.finite,
            report.metrics.p_x.unwrap_or_default(),
            report.metrics.p_y.unwrap_or_default(),
            report.metrics.r_res.unwrap_or_default(),
            report.metrics.recurrent_readout_ratio.unwrap_or_default(),
        );
        if report.backward.enabled {
            let _ = writeln!(
                out,
                "- backward: finite={}, completed_steps={}, final_loss={:.6}",
                report.backward.finite,
                report.backward.completed_steps,
                report.backward.final_loss.unwrap_or_default(),
            );
        }
        let _ = writeln!(out, "- phase0 pass: {}", report.phase0.phase0_pass);
        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "| layer | active | finite | p_x | p_y | current rms | readout rms | readout ratio | residual rms | r_res |"
        );
        let _ = writeln!(out, "|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|");
        for layer in &report.layers {
            let _ = writeln!(
                out,
                "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
                layer.layer_index,
                layer.lowrank_path_active,
                layer.finite,
                fmt_opt(layer.p_x),
                fmt_opt(layer.p_y),
                fmt_opt(layer.current_rms),
                fmt_opt(layer.recurrent_readout_rms),
                fmt_opt(layer.recurrent_readout_ratio),
                fmt_opt(layer.residual_delta_rms),
                fmt_opt(layer.r_res),
            );
        }
        out
    }

    fn fmt_opt(value: Option<f64>) -> String {
        value
            .map(|value| format!("{value:.6}"))
            .unwrap_or_else(|| "-".to_string())
    }

    fn format_initialization_override_toml(initialization: &BdhInitializationConfig) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "[model.initialization]");
        let _ = writeln!(out, "kind = \"{}\"", init_kind_name(initialization));
        let _ = writeln!(
            out,
            "simple_normal_std = {:.12}",
            initialization.simple_normal_std
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "[model.initialization.residual_scaling]");
        let _ = writeln!(
            out,
            "kind = \"{}\"",
            residual_scaling_kind_name(initialization)
        );
        let _ = writeln!(out, "gain = {:.12}", initialization.residual_scaling.gain);
        let _ = writeln!(out);
        let _ = writeln!(out, "[model.initialization.neuron_gains]");
        let _ = writeln!(out, "kind = \"{}\"", neuron_gain_kind_name(initialization));
        let _ = writeln!(
            out,
            "log_sigma = {:.12}",
            initialization.neuron_gains.log_sigma
        );
        let _ = writeln!(
            out,
            "max_gain = {:.12}",
            initialization.neuron_gains.max_gain
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "[model.initialization.topology_prior]");
        let _ = writeln!(
            out,
            "kind = \"{}\"",
            topology_prior_kind_name(initialization)
        );
        let _ = writeln!(
            out,
            "community_count = {}",
            initialization.topology_prior.community_count
        );
        let _ = writeln!(
            out,
            "bridge_fraction = {:.12}",
            initialization.topology_prior.bridge_fraction
        );
        let _ = writeln!(
            out,
            "intra_community_gain = {:.12}",
            initialization.topology_prior.intra_community_gain
        );
        let _ = writeln!(
            out,
            "inter_community_gain = {:.12}",
            initialization.topology_prior.inter_community_gain
        );
        let _ = writeln!(
            out,
            "bridge_gain = {:.12}",
            initialization.topology_prior.bridge_gain
        );
        let _ = writeln!(out);
        let _ = writeln!(out, "[model.initialization.firing_targets]");
        let _ = writeln!(
            out,
            "kind = \"{}\"",
            firing_target_kind_name(initialization)
        );
        let _ = writeln!(
            out,
            "x_target = {:.12}",
            initialization.firing_targets.x_target
        );
        let _ = writeln!(
            out,
            "y_target = {:.12}",
            initialization.firing_targets.y_target
        );
        let _ = writeln!(
            out,
            "x_threshold = {:.12}",
            initialization.firing_targets.x_threshold
        );
        let _ = writeln!(
            out,
            "y_threshold = {:.12}",
            initialization.firing_targets.y_threshold
        );
        out
    }

    fn init_kind_name(initialization: &BdhInitializationConfig) -> &'static str {
        match initialization.kind {
            burn_dragon::core::BdhInitializationKind::NearCritical => "near_critical",
            burn_dragon::core::BdhInitializationKind::SimpleNormal => "simple_normal",
            burn_dragon::core::BdhInitializationKind::HeGlorot => "he_glorot",
            burn_dragon::core::BdhInitializationKind::HeadwiseSemiOrthogonal => {
                "headwise_semi_orthogonal"
            }
        }
    }

    fn residual_scaling_kind_name(initialization: &BdhInitializationConfig) -> &'static str {
        match initialization.residual_scaling.kind {
            burn_dragon::core::BdhResidualScalingKind::FamilyDefault => "family_default",
            burn_dragon::core::BdhResidualScalingKind::Disabled => "disabled",
            burn_dragon::core::BdhResidualScalingKind::DepthScaled => "depth_scaled",
        }
    }

    fn neuron_gain_kind_name(initialization: &BdhInitializationConfig) -> &'static str {
        match initialization.neuron_gains.kind {
            burn_dragon::core::BdhNeuronGainKind::Iid => "iid",
            burn_dragon::core::BdhNeuronGainKind::HeavyTailedLogNormal => "heavy_tailed_log_normal",
        }
    }

    fn topology_prior_kind_name(initialization: &BdhInitializationConfig) -> &'static str {
        match initialization.topology_prior.kind {
            burn_dragon::core::BdhTopologyPriorKind::Iid => "iid",
            burn_dragon::core::BdhTopologyPriorKind::ModularBridges => "modular_bridges",
        }
    }

    fn firing_target_kind_name(initialization: &BdhInitializationConfig) -> &'static str {
        match initialization.firing_targets.kind {
            burn_dragon::core::BdhFiringTargetKind::Disabled => "disabled",
            burn_dragon::core::BdhFiringTargetKind::GaussianEstimate => "gaussian_estimate",
            burn_dragon::core::BdhFiringTargetKind::ExplicitThresholds => "explicit_thresholds",
        }
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

#[cfg(feature = "language-probe")]
fn main() {
    real::main();
}
