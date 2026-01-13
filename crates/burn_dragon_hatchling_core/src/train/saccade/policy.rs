use std::f32::consts::PI;

use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData};

use crate::{VisionLocationEmbeddingMode, VisionNullGlimpseMode};

use crate::train::constants::{SACCADE_EPS, SACCADE_SIGMA_MAX, SACCADE_SIGMA_MIN};
use super::structs::VisionSaccadeModel;

pub(crate) struct SaccadePolicySample<B: BackendTrait> {
    pub(crate) mean: Tensor<B, 3>,
    pub(crate) sigma: Tensor<B, 3>,
    pub(crate) log_prob: Tensor<B, 2>,
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
            return SaccadePolicySample { mean, sigma, log_prob };
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

        let mean = sample
            .clone()
            .slice_dim(2, 0..2)
            .clamp_min(SACCADE_EPS)
            .clamp_max(1.0 - SACCADE_EPS);
        let sigma = sample
            .slice_dim(2, 2..3)
            .clamp_min(SACCADE_SIGMA_MIN)
            .clamp_max(SACCADE_SIGMA_MAX);
        SaccadePolicySample { mean, sigma, log_prob }
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
        }
    }
}

fn fixed_location_embedding<B: BackendTrait>(
    mean: Tensor<B, 3>,
    sigma: Tensor<B, 3>,
    embed_dim: usize,
    config: &crate::VisionLocationEmbeddingConfig,
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
