use super::*;
use crate::model::bdh_support::tensor_values_f32;
use serde::Serialize;

#[derive(Clone, Debug, Default, Serialize)]
pub struct TensorDistributionDiagnostics {
    pub finite: bool,
    pub mean: f64,
    pub std: f64,
    pub rms: f64,
    pub mean_abs: f64,
    pub abs_max: f64,
    pub positive_fraction: f64,
    pub nonzero_fraction: f64,
    pub tiny_fraction_0p1_rms: f64,
    pub kurtosis_excess: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct HeadTensorGeometryDiagnostics {
    pub head_count: usize,
    pub rows: usize,
    pub cols: usize,
    pub mean_head_norm: f64,
    pub head_norm_cv: f64,
    pub max_head_energy_share: f64,
    pub pairwise_cosine_mean: Option<f64>,
    pub pairwise_cosine_max: Option<f64>,
    pub nearest_neighbor_cosine_mean: Option<f64>,
    pub stable_rank_mean: f64,
    pub stable_rank_min: f64,
    pub top_singular_energy_mean: f64,
    pub tensor: TensorDistributionDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TensorComparisonDiagnostics {
    pub cosine: Option<f64>,
    pub relative_l2: Option<f64>,
    pub mean_abs_delta: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct HeadTensorComparisonDiagnostics {
    pub paired_head_cosine_mean: Option<f64>,
    pub paired_head_relative_l2_mean: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageLowRankLayerGeometryDiagnostics {
    pub layer_index: usize,
    pub latent_per_head: usize,
    pub encoder: HeadTensorGeometryDiagnostics,
    pub encoder_v: HeadTensorGeometryDiagnostics,
    pub decoder: HeadTensorGeometryDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageLowRankLayerComparisonDiagnostics {
    pub layer_index: usize,
    pub latent_per_head: usize,
    pub encoder: TensorComparisonDiagnostics,
    pub encoder_v: TensorComparisonDiagnostics,
    pub decoder: TensorComparisonDiagnostics,
    pub encoder_heads: HeadTensorComparisonDiagnostics,
    pub encoder_v_heads: HeadTensorComparisonDiagnostics,
    pub decoder_heads: HeadTensorComparisonDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TensorStateSummaryDiagnostics {
    pub present: bool,
    pub finite: bool,
    pub rms: Option<f64>,
    pub mean_abs: Option<f64>,
    pub abs_max: Option<f64>,
    pub positive_fraction: Option<f64>,
    pub nonzero_fraction: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct TensorStateDeltaDiagnostics {
    pub present_before: bool,
    pub present_after: bool,
    pub cosine: Option<f64>,
    pub delta_rms: Option<f64>,
    pub relative_update: Option<f64>,
    pub after_rms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageLayerStateSummaryDiagnostics {
    pub layer_index: usize,
    pub rho: TensorStateSummaryDiagnostics,
    pub rho_norm: TensorStateSummaryDiagnostics,
    pub sequence_aux: TensorStateSummaryDiagnostics,
    pub mamba_angle_state: TensorStateSummaryDiagnostics,
    pub mamba_k_state: TensorStateSummaryDiagnostics,
    pub mamba_v_state: TensorStateSummaryDiagnostics,
    pub y_neuron_state: TensorStateSummaryDiagnostics,
    pub clocked_slow_hidden: TensorStateSummaryDiagnostics,
    pub summary_memory_hidden: TensorStateSummaryDiagnostics,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageLayerStateDeltaDiagnostics {
    pub layer_index: usize,
    pub rho: TensorStateDeltaDiagnostics,
    pub rho_norm: TensorStateDeltaDiagnostics,
    pub sequence_aux: TensorStateDeltaDiagnostics,
    pub mamba_angle_state: TensorStateDeltaDiagnostics,
    pub mamba_k_state: TensorStateDeltaDiagnostics,
    pub mamba_v_state: TensorStateDeltaDiagnostics,
    pub y_neuron_state: TensorStateDeltaDiagnostics,
    pub clocked_slow_hidden: TensorStateDeltaDiagnostics,
    pub summary_memory_hidden: TensorStateDeltaDiagnostics,
}

fn tensor_distribution(values: &[f32]) -> TensorDistributionDiagnostics {
    if values.is_empty() {
        return TensorDistributionDiagnostics::default();
    }

    let finite = values.iter().all(|value| value.is_finite());
    let len = values.len() as f64;
    let mean = values.iter().map(|&value| value as f64).sum::<f64>() / len;
    let mean_square = values
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        / len;
    let rms = mean_square.sqrt();
    let variance = values
        .iter()
        .map(|&value| {
            let centered = value as f64 - mean;
            centered * centered
        })
        .sum::<f64>()
        / len;
    let std = variance.sqrt();
    let mean_abs = values
        .iter()
        .map(|&value| (value as f64).abs())
        .sum::<f64>()
        / len;
    let abs_max = values
        .iter()
        .map(|&value| (value as f64).abs())
        .fold(0.0_f64, f64::max);
    let positive_fraction =
        values.iter().filter(|&&value| value > 0.0).count() as f64 / values.len() as f64;
    let nonzero_fraction = values
        .iter()
        .filter(|&&value| value.abs() > 1.0e-12)
        .count() as f64
        / values.len() as f64;
    let tiny_threshold = rms * 0.1;
    let tiny_fraction_0p1_rms = values
        .iter()
        .filter(|&&value| (value as f64).abs() <= tiny_threshold)
        .count() as f64
        / values.len() as f64;
    let kurtosis_excess = if std > 0.0 {
        Some(
            values
                .iter()
                .map(|&value| {
                    let centered = (value as f64 - mean) / std;
                    centered.powi(4)
                })
                .sum::<f64>()
                / len
                - 3.0,
        )
    } else {
        None
    };

    TensorDistributionDiagnostics {
        finite,
        mean,
        std,
        rms,
        mean_abs,
        abs_max,
        positive_fraction,
        nonzero_fraction,
        tiny_fraction_0p1_rms,
        kurtosis_excess,
    }
}

fn l2_norm(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|&value| {
            let value = value as f64;
            value * value
        })
        .sum::<f64>()
        .sqrt()
}

fn cosine_similarity(lhs: &[f32], rhs: &[f32]) -> Option<f64> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }
    let mut dot = 0.0;
    let mut lhs_norm = 0.0;
    let mut rhs_norm = 0.0;
    for (&lhs_value, &rhs_value) in lhs.iter().zip(rhs.iter()) {
        let lhs_value = lhs_value as f64;
        let rhs_value = rhs_value as f64;
        dot += lhs_value * rhs_value;
        lhs_norm += lhs_value * lhs_value;
        rhs_norm += rhs_value * rhs_value;
    }
    let denom = lhs_norm.sqrt() * rhs_norm.sqrt();
    (denom > 0.0).then_some(dot / denom)
}

fn relative_l2(lhs: &[f32], rhs: &[f32]) -> Option<f64> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }
    let mut delta_square = 0.0;
    let mut rhs_square = 0.0;
    for (&lhs_value, &rhs_value) in lhs.iter().zip(rhs.iter()) {
        let delta = lhs_value as f64 - rhs_value as f64;
        delta_square += delta * delta;
        let rhs_value = rhs_value as f64;
        rhs_square += rhs_value * rhs_value;
    }
    let rhs_norm = rhs_square.sqrt();
    (rhs_norm > 0.0).then_some(delta_square.sqrt() / rhs_norm)
}

fn mean_abs_delta(lhs: &[f32], rhs: &[f32]) -> Option<f64> {
    if lhs.len() != rhs.len() || lhs.is_empty() {
        return None;
    }
    Some(
        lhs.iter()
            .zip(rhs.iter())
            .map(|(&lhs_value, &rhs_value)| (lhs_value as f64 - rhs_value as f64).abs())
            .sum::<f64>()
            / lhs.len() as f64,
    )
}

fn max_singular_value_squared(values: &[f32], rows: usize, cols: usize) -> f64 {
    if rows == 0 || cols == 0 || values.is_empty() {
        return 0.0;
    }
    let mut vector = vec![1.0 / (cols as f64).sqrt(); cols];
    for _ in 0..12 {
        let mut left = vec![0.0; rows];
        for row_idx in 0..rows {
            let row = &values[row_idx * cols..(row_idx + 1) * cols];
            left[row_idx] = row
                .iter()
                .zip(vector.iter())
                .map(|(&weight, &v)| weight as f64 * v)
                .sum::<f64>();
        }
        let left_norm = left.iter().map(|value| value * value).sum::<f64>().sqrt();
        if left_norm <= f64::EPSILON {
            return 0.0;
        }
        for value in &mut left {
            *value /= left_norm;
        }

        let mut next = vec![0.0; cols];
        for row_idx in 0..rows {
            let row = &values[row_idx * cols..(row_idx + 1) * cols];
            let coeff = left[row_idx];
            for (col_idx, &weight) in row.iter().enumerate() {
                next[col_idx] += weight as f64 * coeff;
            }
        }
        let next_norm = next.iter().map(|value| value * value).sum::<f64>().sqrt();
        if next_norm <= f64::EPSILON {
            return 0.0;
        }
        for (dst, value) in vector.iter_mut().zip(next.iter()) {
            *dst = *value / next_norm;
        }
    }

    let mut left = vec![0.0; rows];
    for row_idx in 0..rows {
        let row = &values[row_idx * cols..(row_idx + 1) * cols];
        left[row_idx] = row
            .iter()
            .zip(vector.iter())
            .map(|(&weight, &v)| weight as f64 * v)
            .sum::<f64>();
    }
    left.iter().map(|value| value * value).sum::<f64>()
}

fn head_geometry(heads: &[Vec<f32>], rows: usize, cols: usize) -> HeadTensorGeometryDiagnostics {
    let all_values = heads.iter().flatten().copied().collect::<Vec<_>>();
    let tensor = tensor_distribution(&all_values);
    if heads.is_empty() {
        return HeadTensorGeometryDiagnostics {
            head_count: 0,
            rows,
            cols,
            tensor,
            ..Default::default()
        };
    }

    let head_norms = heads.iter().map(|head| l2_norm(head)).collect::<Vec<_>>();
    let mean_head_norm = head_norms.iter().sum::<f64>() / head_norms.len() as f64;
    let variance = head_norms
        .iter()
        .map(|norm| {
            let centered = norm - mean_head_norm;
            centered * centered
        })
        .sum::<f64>()
        / head_norms.len() as f64;
    let head_norm_cv = if mean_head_norm > 0.0 {
        variance.sqrt() / mean_head_norm
    } else {
        0.0
    };
    let energies = head_norms
        .iter()
        .map(|norm| norm * norm)
        .collect::<Vec<_>>();
    let energy_sum = energies.iter().sum::<f64>();
    let max_head_energy_share = if energy_sum > 0.0 {
        energies.iter().copied().fold(0.0, f64::max) / energy_sum
    } else {
        0.0
    };

    let mut pairwise = Vec::new();
    let mut nearest = Vec::new();
    for head_idx in 0..heads.len() {
        let mut nearest_for_head = None::<f64>;
        for other_idx in (head_idx + 1)..heads.len() {
            if let Some(cosine) = cosine_similarity(&heads[head_idx], &heads[other_idx]) {
                pairwise.push(cosine);
                nearest_for_head = Some(
                    nearest_for_head
                        .map(|value| value.max(cosine))
                        .unwrap_or(cosine),
                );
            }
        }
        if let Some(value) = nearest_for_head {
            nearest.push(value);
        }
    }

    let mut stable_ranks = Vec::with_capacity(heads.len());
    let mut top_singular_energies = Vec::with_capacity(heads.len());
    for head in heads {
        let frob_sq = head
            .iter()
            .map(|&value| {
                let value = value as f64;
                value * value
            })
            .sum::<f64>();
        let sigma_max_sq = max_singular_value_squared(head, rows, cols);
        if sigma_max_sq > 0.0 && frob_sq > 0.0 {
            stable_ranks.push(frob_sq / sigma_max_sq);
            top_singular_energies.push(sigma_max_sq / frob_sq);
        } else {
            stable_ranks.push(0.0);
            top_singular_energies.push(0.0);
        }
    }

    HeadTensorGeometryDiagnostics {
        head_count: heads.len(),
        rows,
        cols,
        mean_head_norm,
        head_norm_cv,
        max_head_energy_share,
        pairwise_cosine_mean: (!pairwise.is_empty())
            .then_some(pairwise.iter().sum::<f64>() / pairwise.len() as f64),
        pairwise_cosine_max: pairwise.iter().copied().reduce(f64::max),
        nearest_neighbor_cosine_mean: (!nearest.is_empty())
            .then_some(nearest.iter().sum::<f64>() / nearest.len() as f64),
        stable_rank_mean: stable_ranks.iter().sum::<f64>() / stable_ranks.len() as f64,
        stable_rank_min: stable_ranks.iter().copied().fold(f64::INFINITY, f64::min),
        top_singular_energy_mean: top_singular_energies.iter().sum::<f64>()
            / top_singular_energies.len() as f64,
        tensor,
    }
}

fn head_tensor_comparison(
    lhs_heads: &[Vec<f32>],
    rhs_heads: &[Vec<f32>],
) -> HeadTensorComparisonDiagnostics {
    if lhs_heads.len() != rhs_heads.len() || lhs_heads.is_empty() {
        return HeadTensorComparisonDiagnostics::default();
    }
    let mut paired_cosines = Vec::new();
    let mut paired_relative_l2 = Vec::new();
    for (lhs_head, rhs_head) in lhs_heads.iter().zip(rhs_heads.iter()) {
        if let Some(value) = cosine_similarity(lhs_head, rhs_head) {
            paired_cosines.push(value);
        }
        if let Some(value) = relative_l2(lhs_head, rhs_head) {
            paired_relative_l2.push(value);
        }
    }
    HeadTensorComparisonDiagnostics {
        paired_head_cosine_mean: (!paired_cosines.is_empty())
            .then_some(paired_cosines.iter().sum::<f64>() / paired_cosines.len() as f64),
        paired_head_relative_l2_mean: (!paired_relative_l2.is_empty())
            .then_some(paired_relative_l2.iter().sum::<f64>() / paired_relative_l2.len() as f64),
    }
}

fn tensor_comparison(lhs: &[f32], rhs: &[f32]) -> TensorComparisonDiagnostics {
    TensorComparisonDiagnostics {
        cosine: cosine_similarity(lhs, rhs),
        relative_l2: relative_l2(lhs, rhs),
        mean_abs_delta: mean_abs_delta(lhs, rhs),
    }
}

fn tensor_state_summary_from_values(values: Option<Vec<f32>>) -> TensorStateSummaryDiagnostics {
    let Some(values) = values else {
        return TensorStateSummaryDiagnostics::default();
    };
    let summary = tensor_distribution(&values);
    TensorStateSummaryDiagnostics {
        present: true,
        finite: summary.finite,
        rms: Some(summary.rms),
        mean_abs: Some(summary.mean_abs),
        abs_max: Some(summary.abs_max),
        positive_fraction: Some(summary.positive_fraction),
        nonzero_fraction: Some(summary.nonzero_fraction),
    }
}

fn tensor_state_delta_from_values(
    before: Option<Vec<f32>>,
    after: Option<Vec<f32>>,
) -> TensorStateDeltaDiagnostics {
    match (before, after) {
        (Some(before), Some(after)) => {
            let delta = before
                .iter()
                .zip(after.iter())
                .map(|(&lhs, &rhs)| rhs as f64 - lhs as f64)
                .collect::<Vec<_>>();
            let delta_rms = if delta.is_empty() {
                None
            } else {
                Some(
                    (delta.iter().map(|value| value * value).sum::<f64>() / delta.len() as f64)
                        .sqrt(),
                )
            };
            let after_norm = l2_norm(&after);
            let delta_norm = delta.iter().map(|value| value * value).sum::<f64>().sqrt();
            TensorStateDeltaDiagnostics {
                present_before: true,
                present_after: true,
                cosine: cosine_similarity(&before, &after),
                delta_rms,
                relative_update: (after_norm > 0.0).then_some(delta_norm / after_norm),
                after_rms: Some(tensor_distribution(&after).rms),
            }
        }
        (None, Some(after)) => TensorStateDeltaDiagnostics {
            present_before: false,
            present_after: true,
            cosine: None,
            delta_rms: Some(tensor_distribution(&after).rms),
            relative_update: None,
            after_rms: Some(tensor_distribution(&after).rms),
        },
        (Some(_), None) => TensorStateDeltaDiagnostics {
            present_before: true,
            present_after: false,
            ..Default::default()
        },
        (None, None) => TensorStateDeltaDiagnostics::default(),
    }
}

fn optional_tensor_values_4<B: Backend>(tensor: &Option<Tensor<B, 4>>) -> Option<Vec<f32>> {
    tensor
        .as_ref()
        .map(|tensor| tensor_values_f32(tensor.clone()))
}

fn optional_tensor_values_3<B: Backend>(tensor: &Option<Tensor<B, 3>>) -> Option<Vec<f32>> {
    tensor
        .as_ref()
        .map(|tensor| tensor_values_f32(tensor.clone()))
}

fn split_head_tensor(values: &[f32], heads: usize, rows: usize, cols: usize) -> Vec<Vec<f32>> {
    let head_size = rows.saturating_mul(cols);
    (0..heads)
        .map(|head_idx| {
            let start = head_idx * head_size;
            let end = start + head_size;
            values[start..end].to_vec()
        })
        .collect()
}

#[cfg(any(feature = "probe", test))]
impl<B: Backend> BDH<B> {
    pub fn collect_lowrank_geometry_diagnostics(
        &self,
    ) -> Vec<LanguageLowRankLayerGeometryDiagnostics> {
        let mut layers = Vec::with_capacity(self.n_layer);
        for layer_idx in 0..self.n_layer {
            let (encoder, encoder_v, decoder, latent_per_head) =
                self.layer_lowrank_weights(layer_idx);
            let encoder_shape = encoder.shape().dims::<4>();
            let encoder_values = tensor_values_f32(
                encoder.reshape([encoder_shape[1] * encoder_shape[2] * encoder_shape[3]]),
            );
            let encoder_v_shape = encoder_v.shape().dims::<4>();
            let encoder_v_values = tensor_values_f32(
                encoder_v.reshape([encoder_v_shape[1] * encoder_v_shape[2] * encoder_v_shape[3]]),
            );
            let decoder_shape = decoder.shape().dims::<2>();
            let decoder_values =
                tensor_values_f32(decoder.reshape([decoder_shape[0] * decoder_shape[1]]));

            let encoder_heads =
                split_head_tensor(&encoder_values, self.n_head, self.n_embd, latent_per_head);
            let encoder_v_heads =
                split_head_tensor(&encoder_v_values, self.n_head, self.n_embd, latent_per_head);
            let decoder_heads =
                split_head_tensor(&decoder_values, self.n_head, latent_per_head, self.n_embd);

            layers.push(LanguageLowRankLayerGeometryDiagnostics {
                layer_index: layer_idx,
                latent_per_head,
                encoder: head_geometry(&encoder_heads, self.n_embd, latent_per_head),
                encoder_v: head_geometry(&encoder_v_heads, self.n_embd, latent_per_head),
                decoder: head_geometry(&decoder_heads, latent_per_head, self.n_embd),
            });
        }
        layers
    }

    pub fn compare_lowrank_geometry(
        &self,
        other: &Self,
    ) -> Vec<LanguageLowRankLayerComparisonDiagnostics> {
        let layer_count = self.n_layer.min(other.n_layer);
        let head_count = self.n_head.min(other.n_head);
        let mut layers = Vec::with_capacity(layer_count);
        for layer_idx in 0..layer_count {
            let (lhs_encoder, lhs_encoder_v, lhs_decoder, lhs_latent_per_head) =
                self.layer_lowrank_weights(layer_idx);
            let (rhs_encoder, rhs_encoder_v, rhs_decoder, rhs_latent_per_head) =
                other.layer_lowrank_weights(layer_idx);
            if lhs_latent_per_head != rhs_latent_per_head || self.n_embd != other.n_embd {
                continue;
            }

            let encoder_values = tensor_values_f32(
                lhs_encoder
                    .clone()
                    .reshape([head_count * self.n_embd * lhs_latent_per_head]),
            );
            let rhs_encoder_values = tensor_values_f32(
                rhs_encoder
                    .clone()
                    .reshape([head_count * self.n_embd * rhs_latent_per_head]),
            );
            let encoder_v_values = tensor_values_f32(
                lhs_encoder_v
                    .clone()
                    .reshape([head_count * self.n_embd * lhs_latent_per_head]),
            );
            let rhs_encoder_v_values = tensor_values_f32(
                rhs_encoder_v
                    .clone()
                    .reshape([head_count * self.n_embd * rhs_latent_per_head]),
            );
            let decoder_values = tensor_values_f32(
                lhs_decoder
                    .clone()
                    .reshape([head_count * lhs_latent_per_head * self.n_embd]),
            );
            let rhs_decoder_values = tensor_values_f32(
                rhs_decoder
                    .clone()
                    .reshape([head_count * rhs_latent_per_head * self.n_embd]),
            );

            let encoder_heads = split_head_tensor(
                &encoder_values,
                head_count,
                self.n_embd,
                lhs_latent_per_head,
            );
            let rhs_encoder_heads = split_head_tensor(
                &rhs_encoder_values,
                head_count,
                self.n_embd,
                rhs_latent_per_head,
            );
            let encoder_v_heads = split_head_tensor(
                &encoder_v_values,
                head_count,
                self.n_embd,
                lhs_latent_per_head,
            );
            let rhs_encoder_v_heads = split_head_tensor(
                &rhs_encoder_v_values,
                head_count,
                self.n_embd,
                rhs_latent_per_head,
            );
            let decoder_heads = split_head_tensor(
                &decoder_values,
                head_count,
                lhs_latent_per_head,
                self.n_embd,
            );
            let rhs_decoder_heads = split_head_tensor(
                &rhs_decoder_values,
                head_count,
                rhs_latent_per_head,
                self.n_embd,
            );

            layers.push(LanguageLowRankLayerComparisonDiagnostics {
                layer_index: layer_idx,
                latent_per_head: lhs_latent_per_head,
                encoder: tensor_comparison(&encoder_values, &rhs_encoder_values),
                encoder_v: tensor_comparison(&encoder_v_values, &rhs_encoder_v_values),
                decoder: tensor_comparison(&decoder_values, &rhs_decoder_values),
                encoder_heads: head_tensor_comparison(&encoder_heads, &rhs_encoder_heads),
                encoder_v_heads: head_tensor_comparison(&encoder_v_heads, &rhs_encoder_v_heads),
                decoder_heads: head_tensor_comparison(&decoder_heads, &rhs_decoder_heads),
            });
        }
        layers
    }
}

#[cfg(any(feature = "probe", test))]
pub fn summarize_model_state<B: Backend>(
    state: &ModelState<B>,
) -> Vec<LanguageLayerStateSummaryDiagnostics> {
    state
        .layers
        .iter()
        .enumerate()
        .map(
            |(layer_index, layer)| LanguageLayerStateSummaryDiagnostics {
                layer_index,
                rho: tensor_state_summary_from_values(optional_tensor_values_4(&layer.rho)),
                rho_norm: tensor_state_summary_from_values(optional_tensor_values_3(
                    &layer.rho_norm,
                )),
                sequence_aux: tensor_state_summary_from_values(optional_tensor_values_4(
                    &layer.sequence_aux,
                )),
                mamba_angle_state: tensor_state_summary_from_values(optional_tensor_values_3(
                    &layer.mamba_angle_state,
                )),
                mamba_k_state: tensor_state_summary_from_values(optional_tensor_values_3(
                    &layer.mamba_k_state,
                )),
                mamba_v_state: tensor_state_summary_from_values(optional_tensor_values_3(
                    &layer.mamba_v_state,
                )),
                y_neuron_state: tensor_state_summary_from_values(optional_tensor_values_3(
                    &layer.y_neuron_state,
                )),
                clocked_slow_hidden: tensor_state_summary_from_values(optional_tensor_values_4(
                    &layer.clocked_slow_hidden,
                )),
                summary_memory_hidden: tensor_state_summary_from_values(optional_tensor_values_4(
                    &layer.summary_memory_hidden,
                )),
            },
        )
        .collect()
}

#[cfg(any(feature = "probe", test))]
pub fn compare_model_states<B: Backend>(
    before: &ModelState<B>,
    after: &ModelState<B>,
) -> Vec<LanguageLayerStateDeltaDiagnostics> {
    before
        .layers
        .iter()
        .zip(after.layers.iter())
        .enumerate()
        .map(
            |(layer_index, (before_layer, after_layer))| LanguageLayerStateDeltaDiagnostics {
                layer_index,
                rho: tensor_state_delta_from_values(
                    optional_tensor_values_4(&before_layer.rho),
                    optional_tensor_values_4(&after_layer.rho),
                ),
                rho_norm: tensor_state_delta_from_values(
                    optional_tensor_values_3(&before_layer.rho_norm),
                    optional_tensor_values_3(&after_layer.rho_norm),
                ),
                sequence_aux: tensor_state_delta_from_values(
                    optional_tensor_values_4(&before_layer.sequence_aux),
                    optional_tensor_values_4(&after_layer.sequence_aux),
                ),
                mamba_angle_state: tensor_state_delta_from_values(
                    optional_tensor_values_3(&before_layer.mamba_angle_state),
                    optional_tensor_values_3(&after_layer.mamba_angle_state),
                ),
                mamba_k_state: tensor_state_delta_from_values(
                    optional_tensor_values_3(&before_layer.mamba_k_state),
                    optional_tensor_values_3(&after_layer.mamba_k_state),
                ),
                mamba_v_state: tensor_state_delta_from_values(
                    optional_tensor_values_3(&before_layer.mamba_v_state),
                    optional_tensor_values_3(&after_layer.mamba_v_state),
                ),
                y_neuron_state: tensor_state_delta_from_values(
                    optional_tensor_values_3(&before_layer.y_neuron_state),
                    optional_tensor_values_3(&after_layer.y_neuron_state),
                ),
                clocked_slow_hidden: tensor_state_delta_from_values(
                    optional_tensor_values_4(&before_layer.clocked_slow_hidden),
                    optional_tensor_values_4(&after_layer.clocked_slow_hidden),
                ),
                summary_memory_hidden: tensor_state_delta_from_values(
                    optional_tensor_values_4(&before_layer.summary_memory_hidden),
                    optional_tensor_values_4(&after_layer.summary_memory_hidden),
                ),
            },
        )
        .collect()
}
