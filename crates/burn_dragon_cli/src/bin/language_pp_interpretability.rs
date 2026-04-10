#![recursion_limit = "256"]

#[cfg(not(all(feature = "language-probe", feature = "language-universality")))]
fn main() {
    panic!(
        "language_pp_interpretability requires --features 'language-probe language-universality'"
    );
}

#[cfg(all(feature = "language-probe", feature = "language-universality"))]
mod real {
    use std::collections::BTreeMap;
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};

    use anyhow::{Context, Result};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
    use burn_dragon::core::{
        BDH, LanguageBdhInitLayerDiagnostics, LanguageLayerStateDeltaDiagnostics,
        LanguageLayerStateSummaryDiagnostics, LanguageLowRankLayerComparisonDiagnostics,
        LanguageLowRankLayerGeometryDiagnostics, compare_model_states, summarize_model_state,
    };
    use burn_dragon::language::dataset::{Dataset, DatasetSplit, TokenSequenceDataset};
    use burn_dragon::language::train::prepare_datasets;
    use burn_dragon::language::{
        WgpuFusedCoreOverride, apply_init_checkpoint_to_language_core,
        apply_wgpu_fused_core_override, build_model_config_with_tokenizer, language_model_loss,
        load_language_core_from_checkpoint, load_tokenizer_for_checkpoint,
        load_training_config_for_checkpoint, summary_event_mask_tensor,
    };
    use burn_ndarray::NdArray;
    use clap::{Parser, ValueEnum};
    use serde::{Deserialize, Serialize};

    #[cfg(feature = "language-cuda")]
    use burn_cuda::Cuda;

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

    #[derive(Parser, Debug)]
    #[command(name = "language_pp_interpretability")]
    struct Args {
        #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
        backend: BackendArg,
        #[arg(long, default_value_t = 6)]
        eval_batches: usize,
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        #[arg(long, default_value_t = 128)]
        block_size: usize,
        #[arg(long, default_value = "tools/pp_interpretability_spec.json")]
        spec: PathBuf,
        #[arg(long, default_value = "docs/assets/matrix/pp_interpretability.json")]
        json_path: PathBuf,
    }

    #[derive(Clone, Debug)]
    struct ArchitectureSpec {
        name: String,
        stage1_label: String,
        stage1_checkpoint: PathBuf,
        stage1_checkpoint_epoch: Option<usize>,
        stage2_scratch_checkpoint: PathBuf,
        stage2_transfer_legacy_checkpoint: PathBuf,
        stage2_transfer_improved_prestep_checkpoint: Option<PathBuf>,
        stage2_transfer_improved_final_checkpoint: Option<PathBuf>,
    }

    #[derive(Debug, Deserialize)]
    struct ProbeSpecFile {
        #[serde(default)]
        probe_runs: BTreeMap<String, ProbeRunSpec>,
    }

    #[derive(Debug, Deserialize)]
    struct ProbeRunSpec {
        stage1_label: String,
        stage1_checkpoint: PathBuf,
        stage1_checkpoint_epoch: Option<usize>,
        stage2_scratch_checkpoint: PathBuf,
        stage2_transfer_legacy_checkpoint: PathBuf,
        stage2_transfer_improved_prestep_checkpoint: Option<PathBuf>,
        stage2_transfer_improved_final_checkpoint: Option<PathBuf>,
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

    #[derive(Clone, Debug, Default, Serialize)]
    struct TensorSummary {
        finite: bool,
        mean: f64,
        std: f64,
        rms: f64,
        mean_abs: f64,
        abs_max: f64,
        positive_fraction: f64,
        nonzero_fraction: f64,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct HiddenBatchMetrics {
        tensor: TensorSummary,
        token_count: usize,
        dim: usize,
        token_norm_mean: f64,
        token_norm_cv: f64,
        mean_cosine_to_batch_mean: Option<f64>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct LogitBatchMetrics {
        tensor: TensorSummary,
        vocab_size: usize,
        mean_entropy: f64,
        mean_top1_prob: f64,
        mean_top1_top2_gap: f64,
        mean_target_prob: f64,
        mean_target_rank: f64,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct BatchCaseMetrics {
        loss: f64,
        hidden: HiddenBatchMetrics,
        logits: Option<LogitBatchMetrics>,
        init_diagnostics: Vec<LanguageBdhInitLayerDiagnostics>,
        state_summary: Vec<LanguageLayerStateSummaryDiagnostics>,
        state_delta: Vec<LanguageLayerStateDeltaDiagnostics>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct CaseReport {
        label: String,
        dataset_label: String,
        checkpoint: Option<String>,
        uses_factorized_language_head: bool,
        lowrank_geometry: Vec<LanguageLowRankLayerGeometryDiagnostics>,
        batches: Vec<BatchCaseMetrics>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct PairwiseOutputBatchMetrics {
        hidden_token_cosine_mean: Option<f64>,
        hidden_relative_l2: Option<f64>,
        logit_token_cosine_mean: Option<f64>,
        logit_relative_l2: Option<f64>,
        mean_kl_ab: Option<f64>,
        mean_kl_ba: Option<f64>,
        top1_agreement: Option<f64>,
        mean_target_logit_delta: Option<f64>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct PairwiseOutputReport {
        label: String,
        batches: Vec<PairwiseOutputBatchMetrics>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct ArchitectureReport {
        architecture: String,
        stage1_source: CaseReport,
        stage2_fresh_init: CaseReport,
        stage2_transfer_legacy_prestep: CaseReport,
        stage2_transfer_improved_prestep: Option<CaseReport>,
        stage2_scratch_final: CaseReport,
        stage2_transfer_legacy_final: CaseReport,
        stage2_transfer_improved_final: Option<CaseReport>,
        parameter_comparisons: Vec<NamedParameterComparison>,
        output_comparisons: Vec<PairwiseOutputReport>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct NamedParameterComparison {
        label: String,
        layers: Vec<LanguageLowRankLayerComparisonDiagnostics>,
    }

    #[derive(Clone, Debug, Default, Serialize)]
    struct InterpretabilityReport {
        benchmark: &'static str,
        backend: String,
        eval_batches: usize,
        batch_size: usize,
        block_size: usize,
        architectures: Vec<ArchitectureReport>,
    }

    fn load_architecture_specs(spec_path: &Path) -> Result<Vec<ArchitectureSpec>> {
        let spec_json = fs::read_to_string(spec_path)
            .with_context(|| format!("read {}", spec_path.display()))?;
        let spec_file: ProbeSpecFile =
            serde_json::from_str(&spec_json).with_context(|| format!("parse {}", spec_path.display()))?;
        if spec_file.probe_runs.is_empty() {
            anyhow::bail!(
                "no probe_runs configured in interpretability spec {}",
                spec_path.display()
            );
        }
        Ok(spec_file
            .probe_runs
            .into_iter()
            .map(|(name, probe)| ArchitectureSpec {
                name,
                stage1_label: probe.stage1_label,
                stage1_checkpoint: probe.stage1_checkpoint,
                stage1_checkpoint_epoch: probe.stage1_checkpoint_epoch,
                stage2_scratch_checkpoint: probe.stage2_scratch_checkpoint,
                stage2_transfer_legacy_checkpoint: probe.stage2_transfer_legacy_checkpoint,
                stage2_transfer_improved_prestep_checkpoint: probe
                    .stage2_transfer_improved_prestep_checkpoint,
                stage2_transfer_improved_final_checkpoint: probe
                    .stage2_transfer_improved_final_checkpoint,
            })
            .collect())
    }

    fn tensor_summary(values: &[f32]) -> TensorSummary {
        if values.is_empty() {
            return TensorSummary::default();
        }
        let len = values.len() as f64;
        let finite = values.iter().all(|value| value.is_finite());
        let mean = values.iter().map(|&value| value as f64).sum::<f64>() / len;
        let mean_square = values
            .iter()
            .map(|&value| {
                let value = value as f64;
                value * value
            })
            .sum::<f64>()
            / len;
        let variance = values
            .iter()
            .map(|&value| {
                let centered = value as f64 - mean;
                centered * centered
            })
            .sum::<f64>()
            / len;
        TensorSummary {
            finite,
            mean,
            std: variance.sqrt(),
            rms: mean_square.sqrt(),
            mean_abs: values
                .iter()
                .map(|&value| (value as f64).abs())
                .sum::<f64>()
                / len,
            abs_max: values
                .iter()
                .map(|&value| (value as f64).abs())
                .fold(0.0_f64, f64::max),
            positive_fraction: values.iter().filter(|&&value| value > 0.0).count() as f64
                / values.len() as f64,
            nonzero_fraction: values
                .iter()
                .filter(|&&value| value.abs() > 1.0e-12)
                .count() as f64
                / values.len() as f64,
        }
    }

    fn l2_norm(values: &[f64]) -> f64 {
        values.iter().map(|value| value * value).sum::<f64>().sqrt()
    }

    fn cosine_f64(lhs: &[f64], rhs: &[f64]) -> Option<f64> {
        if lhs.len() != rhs.len() || lhs.is_empty() {
            return None;
        }
        let dot = lhs
            .iter()
            .zip(rhs.iter())
            .map(|(lhs, rhs)| lhs * rhs)
            .sum::<f64>();
        let denom = l2_norm(lhs) * l2_norm(rhs);
        (denom > 0.0).then_some(dot / denom)
    }

    fn hidden_batch_metrics<B: BackendTrait>(hidden: Tensor<B, 3>) -> HiddenBatchMetrics {
        let [batch, time, dim] = hidden.shape().dims::<3>();
        let values = hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hidden values");
        let tensor = tensor_summary(&values);
        let rows = batch * time;
        let row_values = values
            .chunks(dim)
            .map(|row| row.iter().map(|&value| value as f64).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let token_norms = row_values
            .iter()
            .map(|row| l2_norm(row))
            .collect::<Vec<_>>();
        let token_norm_mean = token_norms.iter().sum::<f64>() / token_norms.len().max(1) as f64;
        let token_norm_cv = if token_norm_mean > 0.0 {
            let variance = token_norms
                .iter()
                .map(|value| {
                    let centered = value - token_norm_mean;
                    centered * centered
                })
                .sum::<f64>()
                / token_norms.len().max(1) as f64;
            variance.sqrt() / token_norm_mean
        } else {
            0.0
        };
        let mut batch_mean = vec![0.0; dim];
        for row in &row_values {
            for (dst, value) in batch_mean.iter_mut().zip(row.iter()) {
                *dst += *value;
            }
        }
        for value in &mut batch_mean {
            *value /= rows.max(1) as f64;
        }
        let mean_cosine_to_batch_mean = if l2_norm(&batch_mean) > 0.0 {
            let mut cosines = Vec::with_capacity(rows);
            for row in &row_values {
                if let Some(cosine) = cosine_f64(row, &batch_mean) {
                    cosines.push(cosine);
                }
            }
            (!cosines.is_empty()).then_some(cosines.iter().sum::<f64>() / cosines.len() as f64)
        } else {
            None
        };

        HiddenBatchMetrics {
            tensor,
            token_count: rows,
            dim,
            token_norm_mean,
            token_norm_cv,
            mean_cosine_to_batch_mean,
        }
    }

    fn softmax_row_metrics(row: &[f32], target: usize) -> (f64, f64, f64, f64, f64) {
        let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let mut exp_values = Vec::with_capacity(row.len());
        let mut sum_exp = 0.0;
        for &value in row {
            let exp_value = ((value as f64) - max_logit).exp();
            exp_values.push(exp_value);
            sum_exp += exp_value;
        }
        let mut entropy = 0.0;
        let mut top1 = 0.0;
        let mut top2 = 0.0;
        let mut target_prob = 0.0;
        for (index, exp_value) in exp_values.iter().enumerate() {
            let probability = exp_value / sum_exp.max(f64::MIN_POSITIVE);
            if probability > 0.0 {
                entropy -= probability * probability.ln();
            }
            if probability > top1 {
                top2 = top1;
                top1 = probability;
            } else if probability > top2 {
                top2 = probability;
            }
            if index == target {
                target_prob = probability;
            }
        }
        let target_logit = row[target];
        let target_rank = 1.0 + row.iter().filter(|&&value| value > target_logit).count() as f64;
        (entropy, top1, top1 - top2, target_prob, target_rank)
    }

    fn logit_batch_metrics<B: BackendTrait>(
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> LogitBatchMetrics {
        let [batch, time, vocab_size] = logits.shape().dims::<3>();
        let logits_values = logits
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("logit values");
        let target_values = targets
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("target values");
        let tensor = tensor_summary(&logits_values);
        let mut entropy_sum = 0.0;
        let mut top1_sum = 0.0;
        let mut gap_sum = 0.0;
        let mut target_prob_sum = 0.0;
        let mut target_rank_sum = 0.0;
        let rows = batch * time;
        for (row, &target) in logits_values.chunks(vocab_size).zip(target_values.iter()) {
            let (entropy, top1, gap, target_prob, target_rank) =
                softmax_row_metrics(row, target as usize);
            entropy_sum += entropy;
            top1_sum += top1;
            gap_sum += gap;
            target_prob_sum += target_prob;
            target_rank_sum += target_rank;
        }
        LogitBatchMetrics {
            tensor,
            vocab_size,
            mean_entropy: entropy_sum / rows.max(1) as f64,
            mean_top1_prob: top1_sum / rows.max(1) as f64,
            mean_top1_top2_gap: gap_sum / rows.max(1) as f64,
            mean_target_prob: target_prob_sum / rows.max(1) as f64,
            mean_target_rank: target_rank_sum / rows.max(1) as f64,
        }
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

    fn build_untrained_model<B: BackendTrait>(
        config: &burn_dragon::core::BDHConfig,
        seed: u64,
        device: &B::Device,
    ) -> BDH<B> {
        let mut config = config.clone();
        B::seed(device, seed);
        config.vocab_size = config.vocab_size.max(1);
        BDH::<B>::new(config, device)
    }

    fn analyze_case<B: BackendTrait>(
        label: &str,
        dataset_label: &str,
        checkpoint: Option<&Path>,
        model: &BDH<B>,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> CaseReport {
        let mut batches = Vec::with_capacity(eval_batches.len());
        for batch in eval_batches {
            let mut state = model.init_state();
            let before_state = state.clone();
            let inputs = batch.inputs_tensor::<B>(device);
            let targets = batch.targets_tensor::<B>(device);
            let summary_event_mask =
                summary_event_mask_for_batch::<B>(batch, summary_event_token_ids, device);
            let hidden = if let Some(mask) = summary_event_mask.clone() {
                model.forward_hidden_with_state_and_summary_event_mask(
                    inputs.clone(),
                    mask,
                    &mut state,
                )
            } else {
                model.forward_hidden_with_state(inputs.clone(), &mut state)
            };
            let loss = if model.uses_factorized_language_head() {
                model
                    .language_loss_from_hidden(hidden.clone(), targets.clone())
                    .into_scalar()
                    .elem::<f64>()
            } else {
                let logits = model.logits_from_hidden(hidden.clone());
                language_model_loss::<B>(logits, targets.clone())
                    .into_scalar()
                    .elem::<f64>()
            };

            let logits = (!model.uses_factorized_language_head())
                .then(|| model.logits_from_hidden(hidden.clone()));
            let logits_metrics = logits
                .clone()
                .map(|logits| logit_batch_metrics(logits, targets.clone()));
            let init_diagnostics = match summary_event_mask {
                Some(mask) => model
                    .collect_language_bdh_init_diagnostics_with_summary_event_mask(inputs, mask),
                None => model.collect_language_bdh_init_diagnostics(inputs),
            };

            batches.push(BatchCaseMetrics {
                loss,
                hidden: hidden_batch_metrics(hidden),
                logits: logits_metrics,
                init_diagnostics,
                state_summary: summarize_model_state(&state),
                state_delta: compare_model_states(&before_state, &state),
            });
        }

        CaseReport {
            label: label.to_string(),
            dataset_label: dataset_label.to_string(),
            checkpoint: checkpoint.map(|path| path.display().to_string()),
            uses_factorized_language_head: model.uses_factorized_language_head(),
            lowrank_geometry: model.collect_lowrank_geometry_diagnostics(),
            batches,
        }
    }

    fn compare_flat_models_on_batches<B: BackendTrait>(
        label: &str,
        lhs: &BDH<B>,
        rhs: &BDH<B>,
        eval_batches: &[EvalBatchData],
        summary_event_token_ids: Option<&[u32]>,
        device: &B::Device,
    ) -> PairwiseOutputReport {
        let mut batches = Vec::with_capacity(eval_batches.len());
        for batch in eval_batches {
            let mut lhs_state = lhs.init_state();
            let mut rhs_state = rhs.init_state();
            let inputs = batch.inputs_tensor::<B>(device);
            let targets = batch.targets_tensor::<B>(device);
            let summary_event_mask =
                summary_event_mask_for_batch::<B>(batch, summary_event_token_ids, device);
            let lhs_hidden = if let Some(mask) = summary_event_mask.clone() {
                lhs.forward_hidden_with_state_and_summary_event_mask(
                    inputs.clone(),
                    mask,
                    &mut lhs_state,
                )
            } else {
                lhs.forward_hidden_with_state(inputs.clone(), &mut lhs_state)
            };
            let rhs_hidden = if let Some(mask) = summary_event_mask {
                rhs.forward_hidden_with_state_and_summary_event_mask(inputs, mask, &mut rhs_state)
            } else {
                rhs.forward_hidden_with_state(inputs, &mut rhs_state)
            };
            let lhs_hidden_values = lhs_hidden
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("lhs hidden");
            let rhs_hidden_values = rhs_hidden
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("rhs hidden");
            let [batch_size, time, dim] = lhs_hidden.shape().dims::<3>();
            let hidden_token_cosine_mean = {
                let mut cosines = Vec::new();
                for (lhs_row, rhs_row) in lhs_hidden_values
                    .chunks(dim)
                    .zip(rhs_hidden_values.chunks(dim))
                {
                    let lhs_row = lhs_row
                        .iter()
                        .map(|&value| value as f64)
                        .collect::<Vec<_>>();
                    let rhs_row = rhs_row
                        .iter()
                        .map(|&value| value as f64)
                        .collect::<Vec<_>>();
                    if let Some(cosine) = cosine_f64(&lhs_row, &rhs_row) {
                        cosines.push(cosine);
                    }
                }
                (!cosines.is_empty()).then_some(cosines.iter().sum::<f64>() / cosines.len() as f64)
            };
            let hidden_relative_l2 = {
                let lhs_values = lhs_hidden_values
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                let rhs_values = rhs_hidden_values
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                let delta = lhs_values
                    .iter()
                    .zip(rhs_values.iter())
                    .map(|(lhs, rhs)| lhs - rhs)
                    .collect::<Vec<_>>();
                let rhs_norm = l2_norm(&rhs_values);
                (rhs_norm > 0.0).then_some(l2_norm(&delta) / rhs_norm)
            };

            let lhs_logits = lhs.logits_from_hidden(lhs_hidden);
            let rhs_logits = rhs.logits_from_hidden(rhs_hidden);
            let lhs_logits_values = lhs_logits
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("lhs logits");
            let rhs_logits_values = rhs_logits
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("rhs logits");
            let targets_values = targets
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("targets");
            let vocab = lhs_logits.shape().dims::<3>()[2];
            let mut logit_cosines = Vec::new();
            let mut kl_ab_sum = 0.0;
            let mut kl_ba_sum = 0.0;
            let mut agreement_sum = 0.0;
            let mut target_delta_sum = 0.0;
            let rows = batch_size * time;
            for row_idx in 0..rows {
                let start = row_idx * vocab;
                let end = start + vocab;
                let lhs_row = &lhs_logits_values[start..end];
                let rhs_row = &rhs_logits_values[start..end];
                let lhs_row_f64 = lhs_row
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                let rhs_row_f64 = rhs_row
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                if let Some(cosine) = cosine_f64(&lhs_row_f64, &rhs_row_f64) {
                    logit_cosines.push(cosine);
                }

                let max_lhs = lhs_row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let max_rhs = rhs_row.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let lhs_exp = lhs_row
                    .iter()
                    .map(|&value| ((value as f64) - max_lhs).exp())
                    .collect::<Vec<_>>();
                let rhs_exp = rhs_row
                    .iter()
                    .map(|&value| ((value as f64) - max_rhs).exp())
                    .collect::<Vec<_>>();
                let lhs_sum = lhs_exp.iter().sum::<f64>().max(f64::MIN_POSITIVE);
                let rhs_sum = rhs_exp.iter().sum::<f64>().max(f64::MIN_POSITIVE);
                let lhs_probs = lhs_exp
                    .iter()
                    .map(|value| value / lhs_sum)
                    .collect::<Vec<_>>();
                let rhs_probs = rhs_exp
                    .iter()
                    .map(|value| value / rhs_sum)
                    .collect::<Vec<_>>();
                kl_ab_sum += lhs_probs
                    .iter()
                    .zip(rhs_probs.iter())
                    .filter(|(lhs, rhs)| **lhs > 0.0 && **rhs > 0.0)
                    .map(|(lhs, rhs)| lhs * (lhs / rhs).ln())
                    .sum::<f64>();
                kl_ba_sum += rhs_probs
                    .iter()
                    .zip(lhs_probs.iter())
                    .filter(|(lhs, rhs)| **lhs > 0.0 && **rhs > 0.0)
                    .map(|(lhs, rhs)| lhs * (lhs / rhs).ln())
                    .sum::<f64>();
                let lhs_top1 = lhs_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                let rhs_top1 = rhs_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, lhs), (_, rhs)| lhs.total_cmp(rhs))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                agreement_sum += f64::from(lhs_top1 == rhs_top1);
                let target = targets_values[row_idx] as usize;
                target_delta_sum += lhs_row[target] as f64 - rhs_row[target] as f64;
            }

            let logit_relative_l2 = {
                let lhs_values = lhs_logits_values
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                let rhs_values = rhs_logits_values
                    .iter()
                    .map(|&value| value as f64)
                    .collect::<Vec<_>>();
                let delta = lhs_values
                    .iter()
                    .zip(rhs_values.iter())
                    .map(|(lhs, rhs)| lhs - rhs)
                    .collect::<Vec<_>>();
                let rhs_norm = l2_norm(&rhs_values);
                (rhs_norm > 0.0).then_some(l2_norm(&delta) / rhs_norm)
            };

            batches.push(PairwiseOutputBatchMetrics {
                hidden_token_cosine_mean,
                hidden_relative_l2,
                logit_token_cosine_mean: (!logit_cosines.is_empty())
                    .then_some(logit_cosines.iter().sum::<f64>() / logit_cosines.len() as f64),
                logit_relative_l2,
                mean_kl_ab: Some(kl_ab_sum / rows.max(1) as f64),
                mean_kl_ba: Some(kl_ba_sum / rows.max(1) as f64),
                top1_agreement: Some(agreement_sum / rows.max(1) as f64),
                mean_target_logit_delta: Some(target_delta_sum / rows.max(1) as f64),
            });
        }
        PairwiseOutputReport {
            label: label.to_string(),
            batches,
        }
    }

    fn run_architecture<B: BackendTrait>(
        spec: &ArchitectureSpec,
        args: &Args,
        backend_name: &str,
        device: &B::Device,
    ) -> Result<ArchitectureReport> {
        let scratch_checkpoint = spec.stage2_scratch_checkpoint.clone();
        let legacy_transfer_checkpoint = spec.stage2_transfer_legacy_checkpoint.clone();
        let improved_transfer_prestep_checkpoint =
            spec.stage2_transfer_improved_prestep_checkpoint.clone();
        let improved_transfer_final_checkpoint =
            spec.stage2_transfer_improved_final_checkpoint.clone();
        let stage1_checkpoint = spec.stage1_checkpoint.clone();

        let stage2_config =
            load_training_config_for_checkpoint(&[], Some(&scratch_checkpoint), backend_name)
                .with_context(|| {
                    format!(
                        "load stage2 training config for {}",
                        scratch_checkpoint.display()
                    )
                })?;
        let legacy_transfer_config = load_training_config_for_checkpoint(
            &[],
            Some(&legacy_transfer_checkpoint),
            backend_name,
        )
        .with_context(|| {
            format!(
                "load legacy transfer training config for {}",
                legacy_transfer_checkpoint.display()
            )
        })?;
        let improved_transfer_config = improved_transfer_prestep_checkpoint
            .as_ref()
            .map(|checkpoint| {
                load_training_config_for_checkpoint(&[], Some(checkpoint), backend_name)
                    .with_context(|| {
                        format!(
                            "load improved transfer training config for {}",
                            checkpoint.display()
                        )
                    })
            })
            .transpose()?;
        let stage2_datasets = prepare_datasets(&stage2_config.dataset, &stage2_config.training)
            .context("prepare stage2 datasets")?;
        let stage2_tokenizer =
            load_tokenizer_for_checkpoint(&[], Some(&scratch_checkpoint), backend_name)?;
        let mut stage2_model_config = build_model_config_with_tokenizer(
            &stage2_config.model,
            stage2_config.training.block_size,
            stage2_tokenizer.as_ref(),
        )?;
        apply_wgpu_fused_core_override(
            &mut stage2_model_config,
            backend_name,
            WgpuFusedCoreOverride {
                recurrent: stage2_config.wgpu.training.fused_core_recurrent,
                rollout: stage2_config.wgpu.training.fused_core_rollout,
            },
        );
        let stage2_summary_event_ids = stage2_model_config
            .summary_memory
            .write_trigger_token_ids
            .clone();
        let stage2_batches = build_eval_batches(
            stage2_datasets.valid.as_ref(),
            DatasetSplit::Val,
            args.batch_size.max(1),
            args.block_size.max(1),
            args.eval_batches.max(1),
        )?;

        let fresh_init_model =
            build_untrained_model::<B>(&stage2_model_config, stage2_config.training.seed, device);
        let legacy_transfer_prestep_model = apply_init_checkpoint_to_language_core(
            &fresh_init_model,
            &legacy_transfer_config,
            &stage1_checkpoint,
            None,
            backend_name,
            device,
        )?;
        let improved_transfer_prestep_model = improved_transfer_config
            .as_ref()
            .map(|config| {
                apply_init_checkpoint_to_language_core(
                    &fresh_init_model,
                    config,
                    &stage1_checkpoint,
                    None,
                    backend_name,
                    device,
                )
            })
            .transpose()?;
        let scratch_final_model = load_language_core_from_checkpoint::<B>(
            &scratch_checkpoint,
            None,
            &[],
            backend_name,
            device,
        )?;
        let legacy_transfer_final_model = load_language_core_from_checkpoint::<B>(
            &legacy_transfer_checkpoint,
            None,
            &[],
            backend_name,
            device,
        )?;
        let improved_transfer_final_model = improved_transfer_final_checkpoint
            .as_ref()
            .map(|checkpoint| {
                load_language_core_from_checkpoint::<B>(checkpoint, None, &[], backend_name, device)
            })
            .transpose()?;

        let stage1_source = (|| -> Result<CaseReport> {
            let stage1_config =
                load_training_config_for_checkpoint(&[], Some(&stage1_checkpoint), backend_name)
                    .with_context(|| {
                        format!(
                            "load stage1 training config for {}",
                            stage1_checkpoint.display()
                        )
                    })?;
            let stage1_datasets = prepare_datasets(&stage1_config.dataset, &stage1_config.training)
                .context("prepare stage1 datasets")?;
            let stage1_tokenizer =
                load_tokenizer_for_checkpoint(&[], Some(&stage1_checkpoint), backend_name)?;
            let mut stage1_model_config = build_model_config_with_tokenizer(
                &stage1_config.model,
                stage1_config.training.block_size,
                stage1_tokenizer.as_ref(),
            )?;
            apply_wgpu_fused_core_override(
                &mut stage1_model_config,
                backend_name,
                WgpuFusedCoreOverride {
                    recurrent: stage1_config.wgpu.training.fused_core_recurrent,
                    rollout: stage1_config.wgpu.training.fused_core_rollout,
                },
            );
            let stage1_summary_event_ids = stage1_model_config
                .summary_memory
                .write_trigger_token_ids
                .clone();
            let stage1_batches = build_eval_batches(
                stage1_datasets.valid.as_ref(),
                DatasetSplit::Val,
                args.batch_size.max(1),
                args.block_size.max(1),
                args.eval_batches.max(1),
            )?;
            let stage1_model = load_language_core_from_checkpoint::<B>(
                &stage1_checkpoint,
                spec.stage1_checkpoint_epoch,
                &[],
                backend_name,
                device,
            )?;
            Ok(analyze_case(
                &spec.stage1_label,
                "nca_val",
                Some(&stage1_checkpoint),
                &stage1_model,
                &stage1_batches,
                stage1_summary_event_ids.as_deref(),
                device,
            ))
        })()
        .unwrap_or_else(|err| {
            eprintln!(
                "warning: failed to analyze stage1 source checkpoint {}: {err:#}",
                stage1_checkpoint.display()
            );
            CaseReport {
                label: spec.stage1_label.to_string(),
                dataset_label: "nca_val_unavailable".to_string(),
                checkpoint: Some(stage1_checkpoint.display().to_string()),
                uses_factorized_language_head: false,
                lowrank_geometry: Vec::new(),
                batches: Vec::new(),
            }
        });
        let stage2_fresh_init = analyze_case(
            "stage2_fresh_init",
            "shakespeare_val",
            None,
            &fresh_init_model,
            &stage2_batches,
            stage2_summary_event_ids.as_deref(),
            device,
        );
        let stage2_transfer_legacy_prestep = analyze_case(
            "stage2_transfer_legacy_prestep",
            "shakespeare_val",
            Some(&stage1_checkpoint),
            &legacy_transfer_prestep_model,
            &stage2_batches,
            stage2_summary_event_ids.as_deref(),
            device,
        );
        let stage2_transfer_improved_prestep =
            improved_transfer_prestep_model.as_ref().map(|model| {
                analyze_case(
                    "stage2_transfer_improved_prestep",
                    "shakespeare_val",
                    Some(&stage1_checkpoint),
                    model,
                    &stage2_batches,
                    stage2_summary_event_ids.as_deref(),
                    device,
                )
            });
        let stage2_scratch_final = analyze_case(
            "stage2_scratch_final",
            "shakespeare_val",
            Some(&scratch_checkpoint),
            &scratch_final_model,
            &stage2_batches,
            stage2_summary_event_ids.as_deref(),
            device,
        );
        let stage2_transfer_legacy_final = analyze_case(
            "stage2_transfer_legacy_final",
            "shakespeare_val",
            Some(&legacy_transfer_checkpoint),
            &legacy_transfer_final_model,
            &stage2_batches,
            stage2_summary_event_ids.as_deref(),
            device,
        );
        let stage2_transfer_improved_final = improved_transfer_final_model
            .as_ref()
            .zip(improved_transfer_final_checkpoint.as_ref())
            .map(|(model, checkpoint)| {
                analyze_case(
                    "stage2_transfer_improved_final",
                    "shakespeare_val",
                    Some(checkpoint),
                    model,
                    &stage2_batches,
                    stage2_summary_event_ids.as_deref(),
                    device,
                )
            });

        let parameter_comparisons = vec![
            NamedParameterComparison {
                label: "fresh_init_vs_transfer_legacy_prestep".to_string(),
                layers: fresh_init_model.compare_lowrank_geometry(&legacy_transfer_prestep_model),
            },
            NamedParameterComparison {
                label: "fresh_init_vs_scratch_final".to_string(),
                layers: fresh_init_model.compare_lowrank_geometry(&scratch_final_model),
            },
            NamedParameterComparison {
                label: "transfer_legacy_prestep_vs_transfer_legacy_final".to_string(),
                layers: legacy_transfer_prestep_model
                    .compare_lowrank_geometry(&legacy_transfer_final_model),
            },
            NamedParameterComparison {
                label: "scratch_final_vs_transfer_legacy_final".to_string(),
                layers: scratch_final_model.compare_lowrank_geometry(&legacy_transfer_final_model),
            },
        ]
        .into_iter()
        .chain(
            improved_transfer_prestep_model
                .as_ref()
                .map(|model| NamedParameterComparison {
                    label: "fresh_init_vs_transfer_improved_prestep".to_string(),
                    layers: fresh_init_model.compare_lowrank_geometry(model),
                }),
        )
        .chain(
            improved_transfer_final_model
                .as_ref()
                .map(|model| NamedParameterComparison {
                    label: "scratch_final_vs_transfer_improved_final".to_string(),
                    layers: scratch_final_model.compare_lowrank_geometry(model),
                }),
        )
        .chain(
            improved_transfer_prestep_model
                .as_ref()
                .zip(improved_transfer_final_model.as_ref())
                .map(|(prestep, final_model)| NamedParameterComparison {
                    label: "transfer_improved_prestep_vs_transfer_improved_final".to_string(),
                    layers: prestep.compare_lowrank_geometry(final_model),
                }),
        )
        .chain(
            improved_transfer_prestep_model
                .as_ref()
                .map(|model| NamedParameterComparison {
                    label: "transfer_legacy_prestep_vs_transfer_improved_prestep".to_string(),
                    layers: legacy_transfer_prestep_model.compare_lowrank_geometry(model),
                }),
        )
        .chain(
            improved_transfer_final_model
                .as_ref()
                .map(|model| NamedParameterComparison {
                    label: "transfer_legacy_final_vs_transfer_improved_final".to_string(),
                    layers: legacy_transfer_final_model.compare_lowrank_geometry(model),
                }),
        )
        .collect();

        let output_comparisons = vec![
            compare_flat_models_on_batches(
                "fresh_init_vs_transfer_legacy_prestep",
                &fresh_init_model,
                &legacy_transfer_prestep_model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            ),
            compare_flat_models_on_batches(
                "scratch_final_vs_transfer_legacy_final",
                &scratch_final_model,
                &legacy_transfer_final_model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            ),
            compare_flat_models_on_batches(
                "transfer_legacy_prestep_vs_transfer_legacy_final",
                &legacy_transfer_prestep_model,
                &legacy_transfer_final_model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            ),
        ]
        .into_iter()
        .chain(improved_transfer_prestep_model.as_ref().map(|model| {
            compare_flat_models_on_batches(
                "fresh_init_vs_transfer_improved_prestep",
                &fresh_init_model,
                model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            )
        }))
        .chain(improved_transfer_final_model.as_ref().map(|model| {
            compare_flat_models_on_batches(
                "scratch_final_vs_transfer_improved_final",
                &scratch_final_model,
                model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            )
        }))
        .chain(
            improved_transfer_prestep_model
                .as_ref()
                .zip(improved_transfer_final_model.as_ref())
                .map(|(prestep, final_model)| {
                    compare_flat_models_on_batches(
                        "transfer_improved_prestep_vs_transfer_improved_final",
                        prestep,
                        final_model,
                        &stage2_batches,
                        stage2_summary_event_ids.as_deref(),
                        device,
                    )
                }),
        )
        .chain(improved_transfer_prestep_model.as_ref().map(|model| {
            compare_flat_models_on_batches(
                "transfer_legacy_prestep_vs_transfer_improved_prestep",
                &legacy_transfer_prestep_model,
                model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            )
        }))
        .chain(improved_transfer_final_model.as_ref().map(|model| {
            compare_flat_models_on_batches(
                "transfer_legacy_final_vs_transfer_improved_final",
                &legacy_transfer_final_model,
                model,
                &stage2_batches,
                stage2_summary_event_ids.as_deref(),
                device,
            )
        }))
        .collect();

        Ok(ArchitectureReport {
            architecture: spec.name.clone(),
            stage1_source,
            stage2_fresh_init,
            stage2_transfer_legacy_prestep,
            stage2_transfer_improved_prestep,
            stage2_scratch_final,
            stage2_transfer_legacy_final,
            stage2_transfer_improved_final,
            parameter_comparisons,
            output_comparisons,
        })
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
        let device = B::Device::default();
        let specs = load_architecture_specs(&args.spec)?;
        let mut architectures = Vec::with_capacity(specs.len());
        for spec in &specs {
            match run_architecture::<B>(spec, args, backend_name, &device) {
                Ok(report) => architectures.push(report),
                Err(err) => {
                    eprintln!(
                        "warning: skipping architecture {} in interpretability probe: {err:#}",
                        spec.name
                    );
                }
            }
        }
        if architectures.is_empty() {
            eprintln!(
                "warning: no interpretability architectures completed successfully; writing empty report"
            );
        }

        let report = InterpretabilityReport {
            benchmark: "burn_dragon pp-train interpretability",
            backend: backend_name.to_string(),
            eval_batches: args.eval_batches.max(1),
            batch_size: args.batch_size.max(1),
            block_size: args.block_size.max(1),
            architectures,
        };
        let json = serde_json::to_string_pretty(&report).context("serialize report")?;
        if let Some(parent) = args.json_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
        }
        fs::write(&args.json_path, json)
            .with_context(|| format!("write {}", args.json_path.display()))?;

        let mut stdout = String::new();
        writeln!(
            &mut stdout,
            "pp interpretability report written to {}",
            args.json_path.display()
        )
        .ok();
        for arch in &report.architectures {
            let fresh_loss = arch
                .stage2_fresh_init
                .batches
                .iter()
                .map(|batch| batch.loss)
                .sum::<f64>()
                / arch.stage2_fresh_init.batches.len().max(1) as f64;
            let legacy_prestep_loss = arch
                .stage2_transfer_legacy_prestep
                .batches
                .iter()
                .map(|batch| batch.loss)
                .sum::<f64>()
                / arch.stage2_transfer_legacy_prestep.batches.len().max(1) as f64;
            let scratch_loss = arch
                .stage2_scratch_final
                .batches
                .iter()
                .map(|batch| batch.loss)
                .sum::<f64>()
                / arch.stage2_scratch_final.batches.len().max(1) as f64;
            let legacy_transfer_loss = arch
                .stage2_transfer_legacy_final
                .batches
                .iter()
                .map(|batch| batch.loss)
                .sum::<f64>()
                / arch.stage2_transfer_legacy_final.batches.len().max(1) as f64;
            let improved_prestep_loss =
                arch.stage2_transfer_improved_prestep.as_ref().map(|case| {
                    case.batches.iter().map(|batch| batch.loss).sum::<f64>()
                        / case.batches.len().max(1) as f64
                });
            let improved_transfer_loss = arch.stage2_transfer_improved_final.as_ref().map(|case| {
                case.batches.iter().map(|batch| batch.loss).sum::<f64>()
                    / case.batches.len().max(1) as f64
            });
            writeln!(
                &mut stdout,
                "{}: fresh_init={fresh_loss:.4} legacy_prestep={legacy_prestep_loss:.4} improved_prestep={} scratch_final={scratch_loss:.4} legacy_final={legacy_transfer_loss:.4} improved_final={}",
                arch.architecture,
                improved_prestep_loss
                    .map(|loss| format!("{loss:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
                improved_transfer_loss
                    .map(|loss| format!("{loss:.4}"))
                    .unwrap_or_else(|| "n/a".to_string()),
            )
            .ok();
        }
        print!("{stdout}");
        Ok(())
    }
}

#[cfg(all(feature = "language-probe", feature = "language-universality"))]
fn main() {
    real::main();
}
