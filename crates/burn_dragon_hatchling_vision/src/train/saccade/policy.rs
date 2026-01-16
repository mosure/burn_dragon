use std::f32::consts::PI;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData};

use burn_dragon_hatchling_core::{VisionLocationEmbeddingMode, VisionNullGlimpseMode};

use crate::train::constants::{SACCADE_EPS, SACCADE_SIGMA_MAX, SACCADE_SIGMA_MIN};
use super::structs::VisionSaccadeModel;

pub(crate) struct SaccadePolicySample<B: BackendTrait> {
    pub(crate) mean: Tensor<B, 3>,
    pub(crate) sigma: Tensor<B, 3>,
    pub(crate) log_prob: Tensor<B, 2>,
    pub(crate) clamp_rate: Tensor<B, 1>,
}

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn build_input_tokens(
        &self,
        input_context: Tensor<B, 3>,
        state_context: Tensor<B, 3>,
        mean: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let embed_dim = input_context.shape().dims::<3>()[2];
        let mut input_tokens = self.input_proj.forward(input_context) + state_context;
        if let Some(fovea_embed) = self.fovea_embed(mean, sigma, embed_dim) {
            input_tokens = input_tokens + fovea_embed;
        }
        input_tokens
    }

    pub(crate) fn sample_policy_action(
        &self,
        mean: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
    ) -> SaccadePolicySample<B> {
        let device = mean.device();
        let noise_std = self.config.policy.action_noise_std;
        let params = Tensor::cat(vec![mean, sigma], 2);
        let [batch, traj_tokens, _] = params.shape().dims::<3>();
        if noise_std <= SACCADE_EPS {
            let log_prob = Tensor::<B, 2>::zeros([batch, traj_tokens], &device);
            let mean = params.clone().slice_dim(2, 0..2);
            let sigma = params.slice_dim(2, 2..3);
            let clamp_rate = Tensor::<B, 1>::zeros([batch], &device);
            return SaccadePolicySample {
                mean,
                sigma,
                log_prob,
                clamp_rate,
            };
        }

        let noise = Tensor::<B, 3>::random(
            params.shape().dims::<3>(),
            TensorDistribution::Normal(0.0, 1.0),
            &device,
        );
        let sample = params.clone() + noise.mul_scalar(noise_std);

        let log_norm = -0.5 * (2.0 * PI).ln() - noise_std.ln();
        let diff = sample.clone().sub(params).div_scalar(noise_std);
        let log_prob = diff
            .powf_scalar(2.0)
            .mul_scalar(-0.5)
            .add_scalar(log_norm)
            .sum_dim(2)
            .reshape([batch, traj_tokens]);

        let mean_raw = sample.clone().slice_dim(2, 0..2);
        let sigma_raw = sample.clone().slice_dim(2, 2..3);
        let mean_low = mean_raw.clone().lower_equal_elem(SACCADE_EPS).float();
        let mean_high = mean_raw
            .clone()
            .greater_equal_elem(1.0 - SACCADE_EPS)
            .float();
        let sigma_low = sigma_raw
            .clone()
            .lower_equal_elem(SACCADE_SIGMA_MIN)
            .float();
        let sigma_high = sigma_raw
            .clone()
            .greater_equal_elem(SACCADE_SIGMA_MAX)
            .float();
        let mean_clamped = (mean_low + mean_high).clamp_max(1.0);
        let sigma_clamped = (sigma_low + sigma_high).clamp_max(1.0);
        let clamp_count = mean_clamped.sum_dim(2) + sigma_clamped.sum_dim(2);
        let clamp_rate = clamp_count
            .div_scalar(3.0)
            .mean_dim(1)
            .reshape([batch]);

        let mean = mean_raw
            .clamp_min(SACCADE_EPS)
            .clamp_max(1.0 - SACCADE_EPS);
        let sigma = sigma_raw
            .clamp_min(SACCADE_SIGMA_MIN)
            .clamp_max(SACCADE_SIGMA_MAX);
        SaccadePolicySample {
            mean,
            sigma,
            log_prob,
            clamp_rate,
        }
    }

    pub(crate) fn null_patch_tokens(
        &self,
        reference: &Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = reference.device();
        match self.config.policy.info_reward.null_mode {
            VisionNullGlimpseMode::Zero => {
                Tensor::<B, 3>::zeros(reference.shape().dims::<3>(), &device)
            }
            VisionNullGlimpseMode::Noise => Tensor::<B, 3>::random(
                reference.shape().dims::<3>(),
                TensorDistribution::Normal(
                    0.0,
                    self.config.policy.info_reward.null_noise_std as f64,
                ),
                &device,
            ),
        }
    }

    fn fovea_embed(
        &self,
        mean: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
        embed_dim: usize,
    ) -> Option<Tensor<B, 3>> {
        let config = &self.config.policy.location_embedding;
        match config.mode {
            VisionLocationEmbeddingMode::None => None,
            VisionLocationEmbeddingMode::Learned => {
                let params = Tensor::cat(vec![mean, sigma], 2);
                Some(self.fovea_proj.forward(params))
            }
            VisionLocationEmbeddingMode::Sinusoidal | VisionLocationEmbeddingMode::Quantized => {
                Some(fixed_location_embedding(
                    mean,
                    sigma,
                    embed_dim,
                    config,
                ))
            }
            VisionLocationEmbeddingMode::Rope | VisionLocationEmbeddingMode::Pope => {
                Some(rotary_location_embedding(
                    mean,
                    sigma,
                    embed_dim,
                    config,
                    matches!(config.mode, VisionLocationEmbeddingMode::Pope),
                ))
            }
        }
    }
}

fn fixed_location_embedding<B: BackendTrait>(
    mean: Tensor<B, 3>,
    sigma: Tensor<B, 3>,
    embed_dim: usize,
    config: &burn_dragon_hatchling_core::VisionLocationEmbeddingConfig,
) -> Tensor<B, 3> {
    let device = mean.device();
    let [batch, traj_tokens, _] = mean.shape().dims::<3>();
    let target_dim = config.embed_dim.min(embed_dim);
    if batch == 0 || traj_tokens == 0 || target_dim == 0 {
        return Tensor::<B, 3>::zeros([batch.max(1), traj_tokens.max(1), embed_dim], &device);
    }

    let sigma_norm = sigma
        .clone()
        .sub_scalar(SACCADE_SIGMA_MIN)
        .div_scalar((SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN).max(SACCADE_EPS))
        .clamp_min(0.0)
        .clamp_max(1.0);
    let mut coords = Tensor::cat(vec![mean, sigma_norm], 2);
    if config.noise_std > 0.0 {
        let noise = Tensor::<B, 3>::random(
            coords.shape().dims::<3>(),
            TensorDistribution::Normal(0.0, config.noise_std as f64),
            &device,
        );
        coords = (coords + noise).clamp_min(0.0).clamp_max(1.0);
    }
    if matches!(config.mode, VisionLocationEmbeddingMode::Quantized) {
        let bins = config.quantize_bins.max(2) as f32;
        coords = coords
            .mul_scalar(bins - 1.0)
            .add_scalar(0.5)
            .floor()
            .div_scalar(bins - 1.0);
    }

    let freq_count = (target_dim / 6).max(1);
    let mut freqs = Vec::with_capacity(freq_count);
    for idx in 0..freq_count {
        freqs.push((2.0 * PI) * (2.0_f32).powi(idx as i32));
    }
    let freqs = Tensor::<B, 1>::from_data(TensorData::new(freqs, [freq_count]), &device)
        .reshape([1, freq_count])
        .repeat_dim(0, batch * traj_tokens);

    let coords = coords.reshape([batch * traj_tokens, 3]);
    let mut features = Vec::with_capacity(6);
    for idx in 0..3 {
        let coord = coords.clone().slice_dim(1, idx..idx + 1);
        let phase = coord.repeat_dim(1, freq_count) * freqs.clone();
        features.push(phase.clone().sin());
        features.push(phase.cos());
    }
    let mut embed = Tensor::cat(features, 1).reshape([batch, traj_tokens, freq_count * 6]);
    if freq_count * 6 > target_dim {
        embed = embed.slice_dim(2, 0..target_dim);
    } else if freq_count * 6 < target_dim {
        let pad = Tensor::<B, 3>::zeros(
            [batch, traj_tokens, target_dim - freq_count * 6],
            &device,
        );
        embed = Tensor::cat(vec![embed, pad], 2);
    }
    if target_dim < embed_dim {
        let pad = Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim - target_dim], &device);
        Tensor::cat(vec![embed, pad], 2)
    } else {
        embed
    }
}

fn rotary_location_embedding<B: BackendTrait>(
    mean: Tensor<B, 3>,
    sigma: Tensor<B, 3>,
    embed_dim: usize,
    config: &burn_dragon_hatchling_core::VisionLocationEmbeddingConfig,
    use_polar: bool,
) -> Tensor<B, 3> {
    let device = mean.device();
    let [batch, traj_tokens, _] = mean.shape().dims::<3>();
    let requested_dim = config.embed_dim.min(embed_dim);
    let (rotary_dim, target_dim) = if use_polar {
        // PoPE uses complex channels (d), which expand to 2*d real channels.
        let base_dim = config.embed_dim.min(embed_dim / 2);
        let rotary_dim = base_dim.saturating_mul(2);
        (rotary_dim, rotary_dim)
    } else {
        let rotary_dim = requested_dim - (requested_dim % 2);
        (rotary_dim, requested_dim)
    };
    if batch == 0 || traj_tokens == 0 || rotary_dim == 0 {
        return Tensor::<B, 3>::zeros([batch.max(1), traj_tokens.max(1), embed_dim], &device);
    }

    let sigma_norm = sigma
        .clone()
        .sub_scalar(SACCADE_SIGMA_MIN)
        .div_scalar((SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN).max(SACCADE_EPS))
        .clamp_min(0.0)
        .clamp_max(1.0);
    let mut coords = if use_polar {
        pope_coords(mean, sigma_norm)
    } else {
        Tensor::cat(vec![mean, sigma_norm], 2)
    };
    if config.noise_std > 0.0 {
        let noise = Tensor::<B, 3>::random(
            coords.shape().dims::<3>(),
            TensorDistribution::Normal(0.0, config.noise_std as f64),
            &device,
        );
        coords = coords + noise;
    }

    let [_, _, axes] = coords.shape().dims::<3>();
    if use_polar {
        let angle = coords
            .clone()
            .slice_dim(2, 0..1)
            .clamp_min(-0.5)
            .clamp_max(0.5);
        let rest = coords
            .slice_dim(2, 1..axes)
            .clamp_min(0.0)
            .clamp_max(1.0);
        coords = Tensor::cat(vec![angle, rest], 2);
    } else {
        coords = coords.clamp_min(0.0).clamp_max(1.0);
    }

    let dims = if use_polar {
        pope_rotary_dims_per_axis(rotary_dim, axes)
    } else {
        rotary_dims_per_axis(rotary_dim, axes)
    };
    let coords = coords.reshape([batch * traj_tokens, axes]);
    let mut features = Vec::with_capacity(rotary_dim);
    for (axis, axis_dim) in dims.iter().enumerate() {
        if *axis_dim == 0 {
            continue;
        }
        let freq_count = axis_dim / 2;
        let mut inv_freq = Vec::with_capacity(freq_count);
        let denom = freq_count.max(1) as f32;
        for idx in 0..freq_count {
            inv_freq.push(1.0 / 10000.0_f32.powf(idx as f32 / denom));
        }
        let inv_freq =
            Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [freq_count]), &device)
                .reshape([1, freq_count])
                .repeat_dim(0, batch * traj_tokens);
        let coord = coords.clone().slice_dim(1, axis..axis + 1);
        let phase = coord
            .repeat_dim(1, freq_count)
            .mul_scalar(2.0 * PI)
            * inv_freq;
        features.push(phase.clone().sin());
        features.push(phase.cos());
    }
    if features.is_empty() {
        return Tensor::<B, 3>::zeros([batch.max(1), traj_tokens.max(1), embed_dim], &device);
    }

    let mut embed = Tensor::cat(features, 1).reshape([batch, traj_tokens, rotary_dim]);
    if rotary_dim < target_dim {
        let pad = Tensor::<B, 3>::zeros([batch, traj_tokens, target_dim - rotary_dim], &device);
        embed = Tensor::cat(vec![embed, pad], 2);
    }
    if target_dim < embed_dim {
        let pad = Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim - target_dim], &device);
        Tensor::cat(vec![embed, pad], 2)
    } else {
        embed
    }
}

fn pope_coords<B: BackendTrait>(
    mean: Tensor<B, 3>,
    sigma_norm: Tensor<B, 3>,
) -> Tensor<B, 3> {
    let dx = mean.clone().sub_scalar(0.5).slice_dim(2, 0..1);
    let dy = mean.sub_scalar(0.5).slice_dim(2, 1..2);
    let r = (dx.clone().powf_scalar(2.0) + dy.clone().powf_scalar(2.0)).sqrt();
    let max_r = (0.5_f32 * 0.5_f32 + 0.5_f32 * 0.5_f32).sqrt();
    let r_norm = r
        .clone()
        .div_scalar(max_r.max(SACCADE_EPS))
        .clamp_min(0.0)
        .clamp_max(1.0);
    let theta = approx_atan2(dy, dx).div_scalar(2.0 * PI);
    Tensor::cat(vec![theta, r_norm, sigma_norm], 2)
}

fn approx_atan2<B: BackendTrait>(y: Tensor<B, 3>, x: Tensor<B, 3>) -> Tensor<B, 3> {
    let device = y.device();
    let shape = y.shape().dims::<3>();
    let ones = Tensor::<B, 3>::ones(shape, &device);
    let abs_y = y.clone().abs().add_scalar(1e-6);
    let x_ge_zero = x.clone().greater_equal_elem(0.0);
    let r_pos = (x.clone().sub(abs_y.clone())).div(x.clone().add(abs_y.clone()));
    let r_neg = (x.clone().add(abs_y.clone())).div(abs_y.clone().sub(x.clone()));
    let r = r_neg.mask_where(x_ge_zero.clone(), r_pos);
    let base_pos = ones.clone().mul_scalar(PI * 0.25);
    let base_neg = ones.clone().mul_scalar(PI * 0.75);
    let base = base_neg.mask_where(x_ge_zero.clone(), base_pos);
    let r2 = r.clone().powf_scalar(2.0);
    let angle = base + r.clone().mul(r2.mul_scalar(0.1963).add_scalar(-0.9817));
    let sign = ones
        .clone()
        .mul_scalar(-1.0)
        .mask_where(y.clone().greater_equal_elem(0.0), ones);
    sign * angle
}

fn pope_rotary_dims_per_axis(rotary_dim: usize, axes: usize) -> Vec<usize> {
    if axes == 0 || rotary_dim == 0 {
        return Vec::new();
    }
    if axes != 3 {
        return rotary_dims_per_axis(rotary_dim, axes);
    }
    let rotary_dim = rotary_dim - (rotary_dim % 2);
    if rotary_dim == 0 {
        return Vec::new();
    }
    let mut dims = vec![0; 3];
    let mut remaining = rotary_dim;
    if remaining >= 2 {
        dims[0] = 2;
        remaining -= 2;
    }
    if remaining >= 4 {
        dims[1] = 2;
        dims[2] = 2;
        remaining -= 4;
    }
    let slots = remaining / 2;
    if slots > 0 {
        let weights = [2usize, 1, 1];
        let total_weight = weights.iter().sum::<usize>().max(1);
        let mut allocated = [0usize; 3];
        for (axis, weight) in weights.iter().enumerate() {
            allocated[axis] = slots * weight / total_weight;
        }
        let used_slots: usize = allocated.iter().sum();
        let leftover = slots.saturating_sub(used_slots);
        allocated[0] += leftover;
        for axis in 0..3 {
            dims[axis] += allocated[axis] * 2;
        }
    }
    dims
}

fn rotary_dims_per_axis(rotary_dim: usize, axes: usize) -> Vec<usize> {
    if axes == 0 || rotary_dim == 0 {
        return Vec::new();
    }
    let mut dims = vec![0; axes];
    let mut remaining = rotary_dim;
    let mut idx = 0usize;
    while remaining >= 2 {
        dims[idx] += 2;
        remaining -= 2;
        idx = (idx + 1) % axes;
    }
    dims
}
