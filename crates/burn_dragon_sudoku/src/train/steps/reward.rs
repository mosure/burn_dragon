use super::*;

pub(super) fn apply_advantage_guardrails<B: BackendTrait>(
    advantage: Tensor<B, 2>,
    config: &GdpoConfig,
    stats: Option<&Arc<Mutex<GdpoAdvantageStats<B>>>>,
) -> Tensor<B, 2> {
    let mut advantage = advantage;
    let decay = config.advantage_ema_decay;
    if decay > 0.0
        && let Some(stats) = stats
        && let Ok(mut stats) = stats.lock()
    {
        let batch_mean = advantage.clone().mean();
        let batch_sq_mean = advantage.clone().powf_scalar(2.0).mean();
        let batch_var = batch_sq_mean
            .clone()
            .sub(batch_mean.clone().powf_scalar(2.0))
            .clamp_min(0.0);
        let batch_std = (batch_var + config.norm_epsilon.max(1e-12)).sqrt();
        let (ema_mean, ema_std) = stats.update(batch_mean, batch_std, decay);
        let ema_std = ema_std.clamp_min(config.norm_epsilon.max(1e-12));
        let [batch, _] = advantage.shape().dims::<2>();
        let ema_mean = ema_mean.expand([batch.max(1), 1]);
        let ema_std = ema_std.expand([batch.max(1), 1]);
        advantage = advantage.sub(ema_mean).div(ema_std);
    }
    if config.advantage_clip > 0.0 {
        advantage = advantage
            .clamp_min(-config.advantage_clip)
            .clamp_max(config.advantage_clip);
    }
    advantage
}

pub(super) fn difficulty_scale_from_unknowns<B: BackendTrait>(
    unknown_counts: Tensor<B, 1>,
    power: f32,
) -> Tensor<B, 1> {
    let device = unknown_counts.device();
    let [batch] = unknown_counts.shape().dims();
    if power <= 0.0 {
        return Tensor::<B, 1>::ones([batch.max(1)], &device);
    }
    unknown_counts
        .div_scalar(GRID_LEN as f32)
        .clamp_min(0.0)
        .clamp_max(1.0)
        .powf_scalar(power)
}

pub(super) fn build_action_index<B: BackendTrait>(
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = batch.max(1);
    Tensor::<B, 1, Int>::arange(0..GRID_LEN as i64, device)
        .unsqueeze_dim::<2>(0)
        .expand([batch, GRID_LEN])
}

pub(crate) fn static_traversal_actions<B: BackendTrait>(
    batch: usize,
    step_idx: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = batch.max(1);
    let action = (step_idx % GRID_LEN) as i64;
    let data = vec![action; batch];
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, [batch, 1]), device)
}

fn conflict_count<B: BackendTrait>(tokens: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let device = tokens.device();
    let [batch, time] = tokens.shape().dims();
    if batch == 0 || time == 0 {
        return Tensor::<B, 1>::zeros([batch.max(1)], &device);
    }
    let data = tokens.to_data().convert::<i64>();
    let values = data.into_vec::<i64>().unwrap_or_default();
    let mut counts = Vec::with_capacity(batch);
    let stride = time;
    for sample in 0..batch {
        let slice = &values[sample * stride..(sample + 1) * stride];
        let mut conflicts = 0i32;
        for row in 0..9 {
            let mut seen = [0i32; 10];
            for col in 0..9 {
                let v = slice[row * 9 + col] as usize;
                if v > 0 && v <= 9 {
                    seen[v] += 1;
                }
            }
            for &count in seen.iter().skip(1) {
                if count > 1 {
                    conflicts += count - 1;
                }
            }
        }
        for col in 0..9 {
            let mut seen = [0i32; 10];
            for row in 0..9 {
                let v = slice[row * 9 + col] as usize;
                if v > 0 && v <= 9 {
                    seen[v] += 1;
                }
            }
            for &count in seen.iter().skip(1) {
                if count > 1 {
                    conflicts += count - 1;
                }
            }
        }
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut seen = [0i32; 10];
                for row in 0..3 {
                    for col in 0..3 {
                        let idx = (box_row * 3 + row) * 9 + (box_col * 3 + col);
                        let v = slice[idx] as usize;
                        if v > 0 && v <= 9 {
                            seen[v] += 1;
                        }
                    }
                }
                for &count in seen.iter().skip(1) {
                    if count > 1 {
                        conflicts += count - 1;
                    }
                }
            }
        }
        counts.append(&mut vec![conflicts as f32]);
    }
    Tensor::<B, 1>::from_data(TensorData::new(counts, [batch]), &device)
}

pub(super) fn shaping_potential<B: BackendTrait>(
    tokens: &Tensor<B, 2, Int>,
    metric: SudokuRewardShapingMetric,
) -> Tensor<B, 1> {
    match metric {
        SudokuRewardShapingMetric::Conflict => conflict_potential(tokens),
    }
}

fn conflict_potential<B: BackendTrait>(tokens: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
    conflict_count(tokens).mul_scalar(-1.0)
}

pub(super) fn unknown_potential<B: BackendTrait>(tokens: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let [batch, _] = tokens.shape().dims();
    tokens
        .clone()
        .equal_elem(0)
        .float()
        .sum_dim(1)
        .reshape([batch.max(1)])
        .mul_scalar(-1.0)
}

pub(super) fn gae_advantage<B: BackendTrait>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    gamma: f32,
    lambda: f32,
) -> Tensor<B, 1> {
    if rewards.is_empty() {
        let device = values.first().map(|v| v.device()).unwrap_or_default();
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let device = rewards[0].device();
    let [batch] = rewards[0].shape().dims();
    let mut adv = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut adv_sum = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut next_value = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    for (reward, value) in rewards.iter().zip(values.iter()).rev() {
        let delta = reward.clone() + next_value.clone().mul_scalar(gamma) - value.clone();
        adv = delta + adv.mul_scalar(gamma * lambda);
        adv_sum = adv_sum + adv.clone();
        next_value = value.clone();
    }
    adv_sum
}

pub(super) fn gae_advantages<B: BackendTrait>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    dones: &[Tensor<B, 1>],
    next_value: Tensor<B, 1>,
    gamma: f32,
    lambda: f32,
) -> Vec<Tensor<B, 1>> {
    if rewards.is_empty() || values.is_empty() || dones.is_empty() {
        return Vec::new();
    }
    let device = rewards[0].device();
    let [batch] = rewards[0].shape().dims();
    let mut adv = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut next_value = next_value;
    let mut advantages_rev: Vec<Tensor<B, 1>> = Vec::with_capacity(rewards.len());
    for idx in (0..rewards.len()).rev() {
        let reward = &rewards[idx];
        let value = &values[idx];
        let done = &dones[idx];
        let not_done = done.clone().mul_scalar(-1.0).add_scalar(1.0);
        let delta = reward.clone() + next_value.clone().mul_scalar(gamma).mul(not_done.clone())
            - value.clone();
        adv = delta + adv.mul_scalar(gamma * lambda).mul(not_done);
        advantages_rev.push(adv.clone());
        next_value = value.clone();
    }
    advantages_rev.reverse();
    advantages_rev
}

pub(super) fn ensure_non_empty_mask<B: BackendTrait>(
    mask: Tensor<B, 2>,
    fallback: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch, time] = mask.shape().dims();
    if batch == 0 || time == 0 {
        return mask;
    }
    let device = mask.device();
    let mask_sum = mask.clone().sum_dim(1).reshape([batch, 1]);
    let has_any = mask_sum.greater_elem(0.0).float();
    let has_any_grid = has_any.clone().repeat_dim(1, time);
    let keep = has_any_grid.clone();
    let use_fallback = Tensor::<B, 2>::ones([batch, time], &device) - has_any_grid;
    mask * keep + fallback * use_fallback
}
