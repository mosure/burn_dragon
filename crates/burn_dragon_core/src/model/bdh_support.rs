use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use serde::Serialize;

use super::attention_residual::{AttentionResidual, BlockAttentionResidual, ResidualHistory};
use super::{ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnections};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RecurrentPositionMode {
    Sequential,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum RolloutExecutorMode {
    HostLoop,
    WgpuFused,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LogitsProjectionProfileSnapshot {
    pub calls: u64,
    pub total_ns: u128,
}

#[derive(Default)]
struct LogitsProjectionProfileState {
    calls: u64,
    total_ns: u128,
}

static LOGITS_PROJECTION_PROFILE: OnceLock<Mutex<LogitsProjectionProfileState>> = OnceLock::new();

pub(crate) fn logits_projection_profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

pub fn logits_projection_profile_reset() {
    if let Ok(mut state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        *state = LogitsProjectionProfileState::default();
    }
}

pub fn logits_projection_profile_snapshot() -> LogitsProjectionProfileSnapshot {
    if let Ok(state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        return LogitsProjectionProfileSnapshot {
            calls: state.calls,
            total_ns: state.total_ns,
        };
    }
    LogitsProjectionProfileSnapshot::default()
}

pub(crate) fn logits_projection_profile_record(elapsed_ns: u128) {
    if let Ok(mut state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        state.calls = state.calls.saturating_add(1);
        state.total_ns = state.total_ns.saturating_add(elapsed_ns);
    }
}

pub(crate) struct LanguageMhcSplitBindings<B: Backend> {
    pub branch_input: Tensor<B, 4>,
    pub merge: LanguageMhcMergeBindings<B>,
}

pub(crate) struct LanguageMhcMergeBindings<B: Backend> {
    pub residuals_base: Tensor<B, 4>,
    pub legacy_beta: Option<Tensor<B, 2>>,
    pub stream_coefficients: Option<ManifoldHyperConnectionStreamCoefficients<B>>,
}

#[derive(Clone, Debug)]
pub struct LanguagePipelineState<B: Backend> {
    pub(crate) current: Tensor<B, 4>,
    pub(crate) residual_history: ResidualHistory<B>,
}

impl<B: Backend> LanguagePipelineState<B> {
    pub fn from_parts(current: Tensor<B, 4>, residual_history: Vec<Tensor<B, 4>>) -> Self {
        Self {
            current,
            residual_history: ResidualHistory::from_entries(residual_history),
        }
    }

    pub fn into_parts(self) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        (self.current, self.residual_history.into_entries())
    }

    pub fn current(&self) -> &Tensor<B, 4> {
        &self.current
    }

    pub fn residual_history(&self) -> &[Tensor<B, 4>] {
        self.residual_history.as_slice()
    }
}

#[derive(Clone, Copy)]
pub(crate) enum ResidualConnectorRef<'a, B: Backend> {
    Vanilla,
    Mhc(&'a ManifoldHyperConnections<B>),
    AttentionResidual(&'a AttentionResidual<B>),
    BlockAttentionResidual(&'a BlockAttentionResidual<B>),
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageMhcLayerDiagnostics {
    pub layer_index: usize,
    pub num_streams: usize,
    pub stream_norm_mean: f64,
    pub stream_norm_variance: f64,
    pub pairwise_stream_cosine_mean: Option<f64>,
    pub alpha_entropy_mean: f64,
    pub alpha_entropy_normalized_mean: f64,
    pub beta_entropy_mean: Option<f64>,
    pub beta_entropy_normalized_mean: Option<f64>,
    pub residual_distance_identity_l1_mean: f64,
    pub residual_distance_uniform_l1_mean: f64,
}

#[cfg(any(feature = "probe", test))]
#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageBdhInitLayerDiagnostics {
    pub layer_index: usize,
    pub lowrank_path_active: bool,
    pub finite: bool,
    pub p_x: Option<f64>,
    pub p_y: Option<f64>,
    pub current_rms: Option<f64>,
    pub recurrent_readout_rms: Option<f64>,
    pub recurrent_readout_ratio: Option<f64>,
    pub residual_delta_rms: Option<f64>,
    pub r_res: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct LanguageMhcLayerDiagnosticsAccumulator {
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

#[cfg(any(feature = "probe", test))]
#[derive(Clone, Debug, Default)]
struct LanguageBdhInitLayerDiagnosticsAccumulator {
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

pub(crate) fn shannon_entropy(probabilities: &[f32]) -> f64 {
    let total = probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(f64::from)
        .sum::<f64>();
    if total <= 0.0 {
        return 0.0;
    }

    probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| {
            let value = value as f64 / total;
            -value * value.ln()
        })
        .sum()
}

pub(crate) fn average_language_mhc_diagnostics(
    diagnostics_runs: Vec<Vec<LanguageMhcLayerDiagnostics>>,
) -> Vec<LanguageMhcLayerDiagnostics> {
    let mut accumulators = BTreeMap::<usize, LanguageMhcLayerDiagnosticsAccumulator>::new();
    for run in diagnostics_runs {
        for diag in run {
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

    accumulators
        .into_iter()
        .map(|(layer_index, accumulator)| {
            let count = accumulator.count.max(1) as f64;
            LanguageMhcLayerDiagnostics {
                layer_index,
                num_streams: accumulator.num_streams,
                stream_norm_mean: accumulator.stream_norm_mean_sum / count,
                stream_norm_variance: accumulator.stream_norm_variance_sum / count,
                pairwise_stream_cosine_mean: (accumulator.pairwise_stream_cosine_mean_count > 0)
                    .then_some(
                        accumulator.pairwise_stream_cosine_mean_sum
                            / accumulator.pairwise_stream_cosine_mean_count as f64,
                    ),
                alpha_entropy_mean: accumulator.alpha_entropy_mean_sum / count,
                alpha_entropy_normalized_mean: accumulator.alpha_entropy_normalized_mean_sum
                    / count,
                beta_entropy_mean: (accumulator.beta_entropy_mean_count > 0).then_some(
                    accumulator.beta_entropy_mean_sum / accumulator.beta_entropy_mean_count as f64,
                ),
                beta_entropy_normalized_mean: (accumulator.beta_entropy_normalized_mean_count > 0)
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

#[cfg(any(feature = "probe", test))]
pub(crate) fn average_language_bdh_init_diagnostics(
    diagnostics_runs: Vec<Vec<LanguageBdhInitLayerDiagnostics>>,
) -> Vec<LanguageBdhInitLayerDiagnostics> {
    let mut accumulators = BTreeMap::<usize, LanguageBdhInitLayerDiagnosticsAccumulator>::new();
    for run in diagnostics_runs {
        for diag in run {
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

    accumulators
        .into_iter()
        .map(
            |(layer_index, accumulator)| LanguageBdhInitLayerDiagnostics {
                layer_index,
                lowrank_path_active: accumulator.lowrank_active_count * 2 >= accumulator.count,
                finite: accumulator.finite_count == accumulator.count,
                p_x: (accumulator.p_x_count > 0)
                    .then_some(accumulator.p_x_sum / accumulator.p_x_count as f64),
                p_y: (accumulator.p_y_count > 0)
                    .then_some(accumulator.p_y_sum / accumulator.p_y_count as f64),
                current_rms: (accumulator.current_rms_count > 0)
                    .then_some(accumulator.current_rms_sum / accumulator.current_rms_count as f64),
                recurrent_readout_rms: (accumulator.recurrent_readout_rms_count > 0).then_some(
                    accumulator.recurrent_readout_rms_sum
                        / accumulator.recurrent_readout_rms_count as f64,
                ),
                recurrent_readout_ratio: (accumulator.recurrent_readout_ratio_count > 0).then_some(
                    accumulator.recurrent_readout_ratio_sum
                        / accumulator.recurrent_readout_ratio_count as f64,
                ),
                residual_delta_rms: (accumulator.residual_delta_rms_count > 0).then_some(
                    accumulator.residual_delta_rms_sum
                        / accumulator.residual_delta_rms_count as f64,
                ),
                r_res: (accumulator.r_res_count > 0)
                    .then_some(accumulator.r_res_sum / accumulator.r_res_count as f64),
            },
        )
        .collect()
}

#[cfg(any(feature = "probe", test))]
pub(crate) fn tensor_values_f32<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor values")
}

#[cfg(any(feature = "probe", test))]
pub(crate) fn values_are_finite(values: &[f32]) -> bool {
    values.iter().all(|value| value.is_finite())
}

#[cfg(any(feature = "probe", test))]
pub(crate) fn rms_from_values(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let mean_square = values
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        / values.len() as f64;
    mean_square.sqrt()
}

#[cfg(any(feature = "probe", test))]
pub(crate) fn positive_fraction(values: &[f32]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().filter(|value| **value > 0.0).count() as f64 / values.len() as f64
}
