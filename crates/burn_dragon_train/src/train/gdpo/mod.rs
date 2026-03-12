mod cubecl;

use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Tensor, TensorData};
#[cfg(feature = "integration_test")]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::{GdpoConfig, GdpoHardGate};

#[cfg(feature = "integration_test")]
static GDPO_CPU_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

#[cfg(feature = "integration_test")]
pub fn gdpo_reset_cpu_fallbacks() {
    GDPO_CPU_FALLBACKS.store(0, Ordering::Relaxed);
}

#[cfg(feature = "integration_test")]
pub fn gdpo_cpu_fallbacks() -> usize {
    GDPO_CPU_FALLBACKS.load(Ordering::Relaxed)
}

#[cfg(feature = "integration_test")]
fn note_cpu_fallback() {
    GDPO_CPU_FALLBACKS.fetch_add(1, Ordering::Relaxed);
}

const GDPO_LOG_RATIO_CLAMP: f32 = 20.0;

pub fn gdpo_advantage<B: BackendTrait>(
    hard: Tensor<B, 2>,
    easy: Tensor<B, 2>,
    config: &GdpoConfig,
) -> Tensor<B, 2> {
    let hard = nan_to_num(hard);
    let easy = nan_to_num(easy);
    let easy = gate_easy_rewards(hard.clone(), easy, config);
    let hard = hard.unsqueeze_dim::<3>(2);
    let easy = easy.unsqueeze_dim::<3>(2);
    let rewards = Tensor::cat(vec![hard, easy], 2);
    let weights = [config.hard_weight, config.easy_weight];
    let advantage = group_normalize_rewards(rewards, &weights, config.norm_epsilon.max(0.0));
    let advantage = batch_normalize_advantage(advantage, config.norm_epsilon.max(0.0));
    clip_advantage(advantage, config.advantage_clip)
}

pub fn gdpo_advantage_autodiff<B: AutodiffBackend>(
    hard: Tensor<B, 2>,
    easy: Tensor<B, 2>,
    config: &GdpoConfig,
) -> Tensor<B, 2> {
    let hard_inner = hard.inner();
    let easy_inner = easy.inner();
    let advantage_inner = gdpo_advantage::<B::InnerBackend>(hard_inner, easy_inner, config);
    Tensor::from_inner(advantage_inner)
}

pub fn gdpo_policy_loss<B: BackendTrait>(
    log_prob_new: Tensor<B, 2>,
    log_prob_old: Tensor<B, 2>,
    advantage: Tensor<B, 2>,
    config: &GdpoConfig,
) -> Tensor<B, 1> {
    let policy_weight = config.policy_weight.max(0.0);
    if policy_weight <= 0.0 {
        return Tensor::<B, 1>::zeros([1], &log_prob_new.device());
    }
    let log_prob_new = nan_to_num(log_prob_new);
    let log_prob_old = nan_to_num(log_prob_old);
    let advantage = nan_to_num(advantage);
    let clip = config.policy_clip_range.max(0.0);
    if clip <= 0.0 {
        return log_prob_new
            .mul(advantage)
            .mean()
            .mul_scalar(-policy_weight);
    }

    let log_ratio = log_prob_new
        .clone()
        .sub(log_prob_old)
        .clamp_min(-GDPO_LOG_RATIO_CLAMP)
        .clamp_max(GDPO_LOG_RATIO_CLAMP);
    let ratio = log_ratio.exp();
    let clipped = ratio.clone().clamp_min(1.0 - clip).clamp_max(1.0 + clip);
    let surrogate = ratio.mul(advantage.clone());
    let surrogate_clipped = clipped.mul(advantage);
    let use_clipped = surrogate_clipped.clone().lower_equal(surrogate.clone());
    let objective = surrogate.mask_where(use_clipped, surrogate_clipped);
    objective.mean().mul_scalar(-policy_weight)
}

fn gate_easy_rewards<B: BackendTrait>(
    hard: Tensor<B, 2>,
    easy: Tensor<B, 2>,
    config: &GdpoConfig,
) -> Tensor<B, 2> {
    match config.hard_gate {
        GdpoHardGate::Off => easy,
        GdpoHardGate::Fixed { threshold } => {
            let mask = hard.clone().greater_equal_elem(threshold).float();
            easy * mask
        }
        GdpoHardGate::Percentile { quantile } => {
            let thresholds = percentile_thresholds(hard.clone(), quantile);
            let thresholds = thresholds.repeat_dim(1, hard.shape().dims::<2>()[1].max(1));
            let mask = hard.sub(thresholds).greater_equal_elem(0.0).float();
            easy * mask
        }
    }
}

fn percentile_thresholds<B: BackendTrait>(values: Tensor<B, 2>, quantile: f32) -> Tensor<B, 2> {
    let quantile = if quantile.is_nan() {
        0.0
    } else {
        quantile.clamp(0.0, 1.0)
    };
    let [batch, group] = values.shape().dims::<2>();
    if batch == 0 || group == 0 {
        return Tensor::<B, 2>::zeros([batch.max(1), 1], &values.device());
    }
    if group == 1 {
        return nan_to_num(values).slice_dim(1, 0..1);
    }
    if group == 2 {
        let values = nan_to_num(values);
        let v0 = values.clone().slice_dim(1, 0..1);
        let v1 = values.clone().slice_dim(1, 1..2);
        let use_v0 = v0.clone().lower_equal(v1.clone());
        let lo = v1.clone().mask_where(use_v0.clone(), v0.clone());
        let hi = v0.clone().mask_where(use_v0, v1);
        return lo.clone() + (hi - lo).mul_scalar(quantile);
    }
    if group <= cubecl::MAX_GROUP
        && let Some(result) = cubecl::try_percentile_thresholds_cubecl(&values, quantile)
    {
        return result;
    }
    if group <= 8 {
        return percentile_thresholds_tensor(nan_to_num(values), quantile);
    }
    #[cfg(feature = "integration_test")]
    note_cpu_fallback();
    let data = values
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("gdpo rewards vec");
    let mut data = data;
    nan_to_num_data(&mut data);
    let mut thresholds = Vec::with_capacity(batch);
    for b in 0..batch {
        let start = b * group;
        let end = start + group;
        let mut row: Vec<f32> = data[start..end].to_vec();
        row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pos = (group.saturating_sub(1) as f32) * quantile;
        let lower = pos.floor() as usize;
        let upper = pos.ceil() as usize;
        let lower_idx = lower.min(group.saturating_sub(1));
        let upper_idx = upper.min(group.saturating_sub(1));
        let weight = pos - lower as f32;
        let lower_val = row[lower_idx];
        let upper_val = row[upper_idx];
        thresholds.push(lower_val + (upper_val - lower_val) * weight);
    }
    Tensor::<B, 2>::from_data(TensorData::new(thresholds, [batch, 1]), &values.device())
}

fn percentile_thresholds_tensor<B: BackendTrait>(
    values: Tensor<B, 2>,
    quantile: f32,
) -> Tensor<B, 2> {
    let quantile = if quantile.is_nan() {
        0.0
    } else {
        quantile.clamp(0.0, 1.0)
    };
    let [batch, group] = values.shape().dims::<2>();
    if batch == 0 || group == 0 {
        return Tensor::<B, 2>::zeros([batch.max(1), 1], &values.device());
    }
    let mut columns = Vec::with_capacity(group);
    for idx in 0..group {
        columns.push(values.clone().slice_dim(1, idx..idx + 1));
    }
    for _ in 0..group {
        for idx in 0..group.saturating_sub(1) {
            let a = columns[idx].clone();
            let b = columns[idx + 1].clone();
            let lo = a.clone().min_pair(b.clone());
            let hi = a.max_pair(b);
            columns[idx] = lo;
            columns[idx + 1] = hi;
        }
    }
    let pos = (group.saturating_sub(1) as f32) * quantile;
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    let weight = pos - lower as f32;
    let lower_idx = lower.min(group.saturating_sub(1));
    let upper_idx = upper.min(group.saturating_sub(1));
    let lower_val = columns[lower_idx].clone();
    let upper_val = columns[upper_idx].clone();
    lower_val.clone() + (upper_val - lower_val).mul_scalar(weight)
}

fn group_normalize_rewards<B: BackendTrait>(
    rewards: Tensor<B, 3>,
    weights: &[f32],
    epsilon: f32,
) -> Tensor<B, 2> {
    let [batch, group, reward_dim] = rewards.shape().dims::<3>();
    if batch == 0 || group == 0 || reward_dim == 0 {
        return Tensor::<B, 2>::zeros([batch.max(1), group.max(1)], &rewards.device());
    }

    let mean = rewards.clone().mean_dim(1).repeat_dim(1, group);
    let centered = rewards.clone() - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(1)
        .repeat_dim(1, group);
    let std = var.add_scalar(epsilon.max(1e-12)).sqrt();
    let normalized = centered / std;

    let mut weight_vec = Vec::with_capacity(reward_dim);
    for idx in 0..reward_dim {
        weight_vec.push(*weights.get(idx).unwrap_or(&0.0));
    }
    let weight = Tensor::<B, 3>::from_data(
        TensorData::new(weight_vec, [1, 1, reward_dim]),
        &rewards.device(),
    )
    .repeat_dim(0, batch)
    .repeat_dim(1, group);

    (normalized * weight).sum_dim(2).reshape([batch, group])
}

fn batch_normalize_advantage<B: BackendTrait>(
    advantage: Tensor<B, 2>,
    epsilon: f32,
) -> Tensor<B, 2> {
    let [batch, group] = advantage.shape().dims::<2>();
    if batch == 0 || group == 0 {
        return Tensor::<B, 2>::zeros([batch.max(1), group.max(1)], &advantage.device());
    }
    let mean = advantage
        .clone()
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, group);
    let centered = advantage.clone() - mean;
    let var = centered
        .clone()
        .powf_scalar(2.0)
        .mean_dim(0)
        .mean_dim(1)
        .repeat_dim(0, batch)
        .repeat_dim(1, group);
    let std = var.add_scalar(epsilon.max(1e-12)).sqrt();
    centered / std
}

fn clip_advantage<B: BackendTrait>(advantage: Tensor<B, 2>, clip: f32) -> Tensor<B, 2> {
    if clip <= 0.0 {
        return advantage;
    }
    advantage.clamp_min(-clip).clamp_max(clip)
}

fn nan_to_num<B: BackendTrait>(values: Tensor<B, 2>) -> Tensor<B, 2> {
    let [batch, group] = values.shape().dims::<2>();
    if batch == 0 || group == 0 {
        return values;
    }
    let device = values.device();
    let shape = [batch, group];
    let zeros = Tensor::<B, 2>::zeros(shape, &device);
    let ones = Tensor::<B, 2>::ones(shape, &device);
    let mut sanitized = values.clone().mask_where(values.clone().is_nan(), zeros);
    let inf_mask = sanitized.clone().is_inf();
    let pos_inf = inf_mask
        .clone()
        .bool_and(sanitized.clone().greater_equal_elem(0.0));
    let neg_inf = inf_mask.bool_and(sanitized.clone().lower_equal_elem(0.0));
    let pos_values = ones.clone().mul_scalar(f32::MAX);
    let neg_values = ones.mul_scalar(f32::MIN);
    sanitized = sanitized.mask_where(pos_inf, pos_values);
    sanitized.mask_where(neg_inf, neg_values)
}

fn nan_to_num_data(values: &mut [f32]) {
    for value in values.iter_mut() {
        if value.is_nan() {
            *value = 0.0;
        } else if value.is_infinite() {
            *value = if value.is_sign_negative() {
                f32::MIN
            } else {
                f32::MAX
            };
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::train::gdpo::*;
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn gdpo_easy_gate_fixed_threshold() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hard = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![0.1, 0.9, 0.5, 0.8], [2, 2]),
            &device,
        );
        let easy = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![1.0, 2.0, 3.0, 4.0], [2, 2]),
            &device,
        );
        let config = GdpoConfig {
            hard_gate: GdpoHardGate::Fixed { threshold: 0.8 },
            ..GdpoConfig::default()
        };
        let gated = gate_easy_rewards(hard, easy, &config)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("gated vec");
        assert_eq!(gated, vec![0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn gdpo_advantage_zero_rewards_is_zero() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hard = Tensor::<Backend, 2>::zeros([2, 3], &device);
        let easy = Tensor::<Backend, 2>::zeros([2, 3], &device);
        let config = GdpoConfig::default();
        let advantage = gdpo_advantage(hard, easy, &config)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("advantage vec");
        for value in advantage {
            assert!(value.abs() < 1e-6);
        }
    }

    #[test]
    fn gdpo_percentile_gate_uses_linear_interpolation() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hard = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![0.0, 2.0, 4.0, 6.0], [1, 4]),
            &device,
        );
        let thresholds = percentile_thresholds(hard, 0.25)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("threshold vec");
        assert_eq!(thresholds.len(), 1);
        assert!((thresholds[0] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn gdpo_percentile_gate_handles_nan_to_num() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hard = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                vec![f32::NAN, 1.0, f32::INFINITY, f32::NEG_INFINITY],
                [1, 4],
            ),
            &device,
        );
        let thresholds = percentile_thresholds(hard, 0.5)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("threshold vec");
        assert_eq!(thresholds.len(), 1);
        assert!((thresholds[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn gdpo_policy_loss_clips_ratio() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let log_prob_new =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.3], [1, 1]), &device);
        let log_prob_old = Tensor::<Backend, 2>::zeros([1, 1], &device);
        let advantage = Tensor::<Backend, 2>::ones([1, 1], &device);
        let config = GdpoConfig {
            policy_clip_range: 0.2,
            ..GdpoConfig::default()
        };
        let loss = gdpo_policy_loss(log_prob_new, log_prob_old, advantage, &config)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec");
        assert_eq!(loss.len(), 1);
        assert!((loss[0] + 1.2).abs() < 1e-3);
    }
}
