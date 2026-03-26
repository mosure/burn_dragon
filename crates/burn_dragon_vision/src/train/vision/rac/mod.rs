use crate::model::{VisionDragon, VisionProjectionHead, VisionRacVelocityBackbone};
use crate::train::prelude::*;
use burn::tensor::ElementConversion;
use burn::tensor::module::{adaptive_avg_pool2d, interpolate};
use burn::tensor::ops::InterpolateOptions;

const RAC_STATE_PAD_VALUE: f32 = 0.5;

fn pixel_shuffle<B: BackendTrait>(x: Tensor<B, 4>, upscale_factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.shape().dims::<4>();
    let factor = upscale_factor.max(1);
    assert!(
        channels % (factor * factor) == 0,
        "channels must be divisible by upscale_factor^2"
    );
    let out_channels = channels / (factor * factor);
    x.reshape([
        batch as i32,
        out_channels as i32,
        factor as i32,
        factor as i32,
        height as i32,
        width as i32,
    ])
    .permute([0, 1, 4, 2, 5, 3])
    .reshape([
        batch as i32,
        out_channels as i32,
        (height * factor) as i32,
        (width * factor) as i32,
    ])
}

fn pixel_unshuffle<B: BackendTrait>(x: Tensor<B, 4>, downscale_factor: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = x.shape().dims::<4>();
    let factor = downscale_factor.max(1);
    assert!(
        height % factor == 0 && width % factor == 0,
        "height and width must be divisible by downscale_factor"
    );
    let out_height = height / factor;
    let out_width = width / factor;
    x.reshape([
        batch as i32,
        channels as i32,
        out_height as i32,
        factor as i32,
        out_width as i32,
        factor as i32,
    ])
    .permute([0, 1, 3, 5, 2, 4])
    .reshape([
        batch as i32,
        (channels * factor * factor) as i32,
        out_height as i32,
        out_width as i32,
    ])
}

mod artifacts;
mod report;
mod runtime;
mod teacher;
#[cfg(test)]
mod tests;

use artifacts::build_rac_artifacts;
pub(crate) use report::write_rac_best_checkpoint_report;
pub use runtime::{VisionRacCheckpointEvalSummary, eval_vision_rac_checkpoint_backend};
pub(crate) use teacher::{
    build_rac_semantic_teacher_store, build_rac_teacher_latent_store,
    rac_semantic_teacher_feature_dim,
};

#[derive(Clone)]
pub(crate) struct VisionRacBatch<B: BackendTrait> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
    pub teacher_latent: Option<Tensor<B, 4>>,
    pub teacher_patch: Option<Tensor<B, 3>>,
    pub teacher_cls: Option<Tensor<B, 2>>,
}

impl<B: BackendTrait> VisionRacBatch<B> {
    pub fn new(
        images: Tensor<B, 4>,
        labels: Tensor<B, 1, Int>,
        teacher_latent: Option<Tensor<B, 4>>,
        teacher_patch: Option<Tensor<B, 3>>,
        teacher_cls: Option<Tensor<B, 2>>,
    ) -> Self {
        Self {
            images,
            labels,
            teacher_latent,
            teacher_patch,
            teacher_cls,
        }
    }
}

impl<B: BackendTrait> From<CifarBatch<B>> for VisionRacBatch<B> {
    fn from(batch: CifarBatch<B>) -> Self {
        Self::new(batch.images, batch.labels, None, None, None)
    }
}

impl<B: BackendTrait> From<ImageNetBatch<B>> for VisionRacBatch<B> {
    fn from(batch: ImageNetBatch<B>) -> Self {
        Self::new(
            batch.images,
            batch.labels,
            batch.rac_teacher_latent,
            batch.teacher_patch,
            batch.teacher_cls,
        )
    }
}

#[derive(Clone)]
struct VisionRacLosses<B: BackendTrait> {
    total: Tensor<B, 1>,
    recon: Tensor<B, 1>,
    aux: Tensor<B, 1>,
    roundtrip: Tensor<B, 1>,
    roundtrip_state: Tensor<B, 1>,
    recon_psnr: Tensor<B, 1>,
    roundtrip_psnr: Tensor<B, 1>,
    forward_path: Tensor<B, 1>,
    reverse_path: Tensor<B, 1>,
    forward_velocity: Tensor<B, 1>,
    reverse_latent: Tensor<B, 1>,
    reverse_to_init: Tensor<B, 1>,
    block_const: Tensor<B, 1>,
    semantic_loss: Option<Tensor<B, 1>>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

#[derive(Clone)]
pub(crate) struct VisionRacEvalOutput<B: BackendTrait> {
    pub images: Tensor<B, 4>,
    pub forward_state: Tensor<B, 4>,
    pub roundtrip_state: Tensor<B, 4>,
    pub total: Tensor<B, 1>,
    pub recon: Tensor<B, 1>,
    pub aux: Tensor<B, 1>,
    pub roundtrip: Tensor<B, 1>,
    pub roundtrip_state_loss: Tensor<B, 1>,
    pub recon_psnr: Tensor<B, 1>,
    pub roundtrip_psnr: Tensor<B, 1>,
    pub forward_path: Tensor<B, 1>,
    pub reverse_path: Tensor<B, 1>,
    pub forward_velocity: Tensor<B, 1>,
    pub reverse_latent: Tensor<B, 1>,
    pub reverse_to_init: Tensor<B, 1>,
    pub block_const: Tensor<B, 1>,
    pub semantic_loss: Option<Tensor<B, 1>>,
    pub probe_loss: Tensor<B, 1>,
    pub probe_acc: Tensor<B, 1>,
    pub artifacts: Option<VisionArtifactInput<B>>,
}

struct VisionRacTrajectory<B: BackendTrait> {
    states: Vec<Tensor<B, 4>>,
    step_tokens: Vec<Tensor<B, 3>>,
    step_velocity_patches: Vec<Tensor<B, 3>>,
    step_velocity_frames: Vec<Tensor<B, 4>>,
    step_memory_read_norms: Vec<Tensor<B, 2>>,
    step_memory_write_norms: Vec<Tensor<B, 2>>,
    step_summaries: Vec<Tensor<B, 2>>,
    state_times: Vec<f32>,
    final_state: Tensor<B, 4>,
}

#[derive(Module, Debug)]
pub(crate) struct VisionRacModel<B: BackendTrait> {
    pub(crate) velocity: VisionRacVelocityBackbone<B>,
    semantic_head: Option<VisionProjectionHead<B>>,
    semantic_patch_head: Option<VisionProjectionHead<B>>,
    probe: VisionProbe<B>,
    probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    #[module(ignore)]
    pub(crate) config: VisionRacConfig,
}

impl<B: BackendTrait> VisionRacModel<B> {
    pub(crate) fn new(
        state_model: VisionDragon<B>,
        config: VisionRacConfig,
        vision: &VisionDragonConfig,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        let semantic_head =
            if config.semantic_teacher.weight > 0.0 && config.semantic_teacher.cls_weight > 0.0 {
                rac_semantic_teacher_feature_dim(&config, vision.projection_dim).map(|target_dim| {
                    let hidden_dim = config
                        .semantic_teacher
                        .hidden_dim
                        .unwrap_or_else(|| vision.projection_dim.max(target_dim).saturating_mul(2))
                        .max(1);
                    VisionProjectionHead::new(
                        vision.projection_dim.max(1),
                        hidden_dim,
                        target_dim.max(1),
                        0.0,
                        &vision.normalization,
                        device,
                    )
                })
            } else {
                None
            };
        let semantic_patch_head = match &config.semantic_teacher.teacher {
            VisionTeacherConfig::Features(teacher)
                if config.semantic_teacher.weight > 0.0
                    && config.semantic_teacher.patch_weight > 0.0
                    && teacher.train_patch_path.is_some()
                    && teacher.patch_tokens.is_some() =>
            {
                let target_dim = teacher.feature_dim.max(1);
                let hidden_dim = config
                    .semantic_teacher
                    .hidden_dim
                    .unwrap_or_else(|| vision.projection_dim.max(target_dim).saturating_mul(2))
                    .max(1);
                Some(VisionProjectionHead::new(
                    vision.projection_dim.max(1),
                    hidden_dim,
                    target_dim,
                    0.0,
                    &vision.normalization,
                    device,
                ))
            }
            _ => None,
        };
        let probe = VisionProbe::new(
            vision.projection_dim.max(1),
            num_classes.max(1),
            &vision.normalization,
            device,
        );
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let velocity = VisionRacVelocityBackbone::new(state_model, vision, &config, device);
        Self {
            velocity,
            semantic_head,
            semantic_patch_head,
            probe,
            probe_loss,
            config,
        }
    }

    fn build_time_grid(&self, randomize: bool) -> Vec<f32> {
        let steps = self.config.sample_steps.max(1);
        if randomize && steps > 1 {
            let mut values = (0..steps.saturating_sub(1))
                .map(|_| thread_rng().gen_range(0.0..1.0))
                .collect::<Vec<_>>();
            values.sort_by(|left, right| {
                left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut grid = Vec::with_capacity(steps + 1);
            grid.push(0.0);
            grid.extend(values);
            grid.push(1.0);
            grid
        } else {
            (0..=steps)
                .map(|idx| idx as f32 / steps.max(1) as f32)
                .collect::<Vec<_>>()
        }
    }

    fn mse4(&self, prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        (prediction - target).powf_scalar(2.0).mean().reshape([1])
    }

    fn rgb_state_channels(&self, tensor: &Tensor<B, 4>) -> usize {
        tensor.shape().dims::<4>()[1].min(3)
    }

    fn rgb_state(&self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = tensor.shape().dims::<4>();
        let rgb = channels.min(3);
        tensor.slice([0..batch, 0..rgb, 0..height, 0..width])
    }

    fn mse4_rgb(&self, prediction: Tensor<B, 4>, target: Tensor<B, 4>) -> Tensor<B, 1> {
        self.mse4(self.rgb_state(prediction), self.rgb_state(target))
    }

    fn add_input_noise(&self, state: Tensor<B, 4>, std_scale: f32, std_min: f32) -> Tensor<B, 4> {
        let std_scale = std_scale.max(0.0);
        let std_min = std_min.max(0.0);
        if std_scale <= 0.0 && std_min <= 0.0 {
            return state;
        }
        let rms = state
            .clone()
            .powf_scalar(2.0)
            .mean()
            .sqrt()
            .into_scalar()
            .elem::<f32>();
        let noise_std = (rms * std_scale).max(std_min);
        if noise_std <= 0.0 {
            return state;
        }
        let [batch, channels, height, width] = state.shape().dims::<4>();
        let device = state.device();
        let noise = Tensor::<B, 4>::random(
            [batch, channels, height, width],
            burn::tensor::Distribution::Normal(0.0, noise_std as f64),
            &device,
        );
        (state + noise).clamp(self.config.state_clamp_min, self.config.state_clamp_max)
    }

    fn cosine_alignment_loss(
        &self,
        prediction: Tensor<B, 2>,
        target: Tensor<B, 2>,
    ) -> Tensor<B, 1> {
        let pred_norm = prediction
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1.0e-6);
        let target_norm = target
            .clone()
            .powf_scalar(2.0)
            .sum_dim(1)
            .sqrt()
            .clamp_min(1.0e-6);
        let pred_unit = prediction.div(pred_norm);
        let target_unit = target.div(target_norm);
        let cosine = pred_unit.mul(target_unit).sum_dim(1).mean().reshape([1]);
        Tensor::<B, 1>::ones([1], &cosine.device()) - cosine
    }

    fn cosine_alignment_loss_tokens(
        &self,
        prediction: Tensor<B, 3>,
        target: Tensor<B, 3>,
    ) -> Tensor<B, 1> {
        let pred_norm = prediction
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1.0e-6)
            .reshape([
                prediction.shape().dims::<3>()[0],
                prediction.shape().dims::<3>()[1],
                1,
            ]);
        let target_norm = target
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .clamp_min(1.0e-6)
            .reshape([
                target.shape().dims::<3>()[0],
                target.shape().dims::<3>()[1],
                1,
            ]);
        let pred_unit = prediction.div(pred_norm);
        let target_unit = target.div(target_norm);
        let cosine = pred_unit.mul(target_unit).sum_dim(2).mean().reshape([1]);
        Tensor::<B, 1>::ones([1], &cosine.device()) - cosine
    }

    fn maybe_square_grid(tokens: usize) -> Option<usize> {
        let side = (tokens as f64).sqrt() as usize;
        if side.saturating_mul(side) == tokens {
            Some(side)
        } else {
            None
        }
    }

    fn resample_patch_teacher(
        &self,
        teacher_patch: Tensor<B, 3>,
        target_tokens: usize,
    ) -> Tensor<B, 3> {
        let [batch, source_tokens, feature_dim] = teacher_patch.shape().dims::<3>();
        if source_tokens == target_tokens {
            return teacher_patch;
        }

        let Some(source_side) = Self::maybe_square_grid(source_tokens) else {
            return teacher_patch.slice([
                0..batch,
                0..target_tokens.min(source_tokens),
                0..feature_dim,
            ]);
        };
        let Some(target_side) = Self::maybe_square_grid(target_tokens) else {
            return teacher_patch.slice([
                0..batch,
                0..target_tokens.min(source_tokens),
                0..feature_dim,
            ]);
        };

        let teacher_grid = teacher_patch
            .reshape([batch, source_side, source_side, feature_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let resized = interpolate(
            teacher_grid,
            [target_side, target_side],
            InterpolateOptions::new(InterpolateMode::Bilinear),
        );
        resized.swap_dims(2, 3).swap_dims(1, 3).reshape([
            batch,
            target_side * target_side,
            feature_dim,
        ])
    }

    fn align_teacher_latent_channels(&self, latent: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = latent.shape().dims::<4>();
        let state_channels = self.config.state_channels.max(1);
        if channels == state_channels {
            return latent;
        }
        if channels > state_channels {
            return latent.slice([0..batch, 0..state_channels, 0..height, 0..width]);
        }
        let pad = Tensor::<B, 4>::full(
            [batch, state_channels - channels, height, width],
            RAC_STATE_PAD_VALUE,
            &latent.device(),
        );
        Tensor::cat(vec![latent, pad], 1)
    }

    fn pooled_teacher_latent(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch, channels, height, width] = images.shape().dims::<4>();
        let downsample = self.config.teacher.latent_downsample.max(1);
        let latent_h = height.div_ceil(downsample).max(1);
        let latent_w = width.div_ceil(downsample).max(1);
        let pooled = adaptive_avg_pool2d(images, [latent_h, latent_w]);
        let [batch, pooled_channels, pooled_h, pooled_w] = pooled.shape().dims::<4>();
        let pooled = if pooled_channels == channels && pooled_h == latent_h && pooled_w == latent_w
        {
            pooled
        } else {
            pooled.reshape([batch, channels, latent_h, latent_w])
        };
        self.align_teacher_latent_channels(pooled)
    }

    fn teacher_target_latent(
        &self,
        images: Tensor<B, 4>,
        teacher_latent: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        match self.config.teacher.kind {
            VisionRacTeacherKind::PooledImage => self.pooled_teacher_latent(images),
            VisionRacTeacherKind::PrecomputedLatent => {
                let latent = teacher_latent.expect("precomputed RAC latent teacher batch");
                self.align_teacher_latent_channels(latent.detach())
            }
        }
    }

    fn image_to_state(&self, images: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let state_channels = self.config.state_channels.max(1);
        if channels == state_channels {
            return images;
        }
        if channels > state_channels {
            return images.slice([0..batch, 0..state_channels, 0..height, 0..width]);
        }
        let pad = Tensor::<B, 4>::full(
            [batch, state_channels - channels, height, width],
            RAC_STATE_PAD_VALUE,
            &images.device(),
        );
        Tensor::cat(vec![images, pad], 1)
    }

    fn expand_teacher_latent(
        &self,
        latent: Tensor<B, 4>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let [batch, channels, latent_h, latent_w] = latent.shape().dims::<4>();
        if latent_h == height && latent_w == width {
            return latent.reshape([batch, channels, height, width]);
        }
        interpolate(
            latent,
            [height, width],
            InterpolateOptions::new(InterpolateMode::Nearest),
        )
    }

    fn latent_to_centered_subpixel_state(
        &self,
        latent: Tensor<B, 4>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let [batch, channels, latent_h, latent_w] = latent.shape().dims::<4>();
        if latent_h == height && latent_w == width {
            return latent;
        }
        if height % latent_h != 0 || width % latent_w != 0 {
            return self.expand_teacher_latent(latent, height, width);
        }
        let stride_h = height / latent_h;
        let stride_w = width / latent_w;
        if stride_h != stride_w || stride_h == 0 {
            return self.expand_teacher_latent(latent, height, width);
        }
        let stride = stride_h;
        let shuffle_channels = channels.saturating_mul(stride).saturating_mul(stride);
        let mut shuffle_space = Tensor::<B, 4>::full(
            [batch, shuffle_channels, latent_h, latent_w],
            RAC_STATE_PAD_VALUE,
            &latent.device(),
        );
        let offset_y = stride / 2;
        let offset_x = stride / 2;
        let offset_idx = offset_y.saturating_mul(stride).saturating_add(offset_x);
        for channel in 0..channels {
            let source =
                latent
                    .clone()
                    .slice([0..batch, channel..channel + 1, 0..latent_h, 0..latent_w]);
            shuffle_space = shuffle_space.slice_assign(
                [
                    0..batch,
                    channel * stride * stride + offset_idx
                        ..channel * stride * stride + offset_idx + 1,
                    0..latent_h,
                    0..latent_w,
                ],
                source,
            );
        }
        pixel_shuffle(shuffle_space, stride)
    }

    fn centered_subpixel_state_to_latent(
        &self,
        state: Tensor<B, 4>,
        latent_h: usize,
        latent_w: usize,
    ) -> Tensor<B, 4> {
        let [batch, channels, height, width] = state.shape().dims::<4>();
        if height == latent_h && width == latent_w {
            return state;
        }
        if height % latent_h != 0 || width % latent_w != 0 {
            return adaptive_avg_pool2d(state, [latent_h, latent_w]);
        }
        let stride_h = height / latent_h;
        let stride_w = width / latent_w;
        if stride_h != stride_w || stride_h == 0 {
            return adaptive_avg_pool2d(state, [latent_h, latent_w]);
        }
        let stride = stride_h;
        let folded = pixel_unshuffle(state, stride);
        let offset_y = stride / 2;
        let offset_x = stride / 2;
        let offset_idx = offset_y.saturating_mul(stride).saturating_add(offset_x);
        let mut outputs = Vec::with_capacity(channels);
        for channel in 0..channels {
            outputs.push(folded.clone().slice([
                0..batch,
                channel * stride * stride + offset_idx..channel * stride * stride + offset_idx + 1,
                0..latent_h,
                0..latent_w,
            ]));
        }
        Tensor::cat(outputs, 1)
    }

    fn teacher_latent_to_state(
        &self,
        latent: Tensor<B, 4>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        match self.config.teacher.state_mapping {
            VisionRacStateMappingKind::ExpandNearest => {
                self.expand_teacher_latent(latent, height, width)
            }
            VisionRacStateMappingKind::CenteredSubpixel => {
                self.latent_to_centered_subpixel_state(latent, height, width)
            }
        }
    }

    fn state_to_teacher_latent(
        &self,
        state: Tensor<B, 4>,
        reference_latent: &Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [_batch, _channels, latent_h, latent_w] = reference_latent.shape().dims::<4>();
        let latent = match self.config.teacher.state_mapping {
            VisionRacStateMappingKind::ExpandNearest => {
                adaptive_avg_pool2d(state, [latent_h, latent_w])
            }
            VisionRacStateMappingKind::CenteredSubpixel => {
                self.centered_subpixel_state_to_latent(state, latent_h, latent_w)
            }
        };
        self.align_teacher_latent_channels(latent)
    }

    fn straight_path_state(
        &self,
        start_state: Tensor<B, 4>,
        target_state: Tensor<B, 4>,
        time_value: f32,
    ) -> Tensor<B, 4> {
        start_state.mul_scalar(1.0 - time_value) + target_state.mul_scalar(time_value)
    }

    fn integrate_flow(
        &self,
        start_state: Tensor<B, 4>,
        time_grid: &[f32],
        reverse: bool,
        wipe_after_step: Option<usize>,
    ) -> VisionRacTrajectory<B> {
        let steps = time_grid.len().saturating_sub(1).max(1);
        let mut current = start_state.clone();
        let mut states = vec![current.clone()];
        let mut state_times = vec![if reverse { 1.0 } else { 0.0 }];
        let mut step_tokens = Vec::with_capacity(steps);
        let mut step_velocity_patches = Vec::with_capacity(steps);
        let mut step_velocity_frames = Vec::with_capacity(steps);
        let mut step_memory_read_norms = Vec::with_capacity(steps);
        let mut step_memory_write_norms = Vec::with_capacity(steps);
        let mut step_summaries = Vec::with_capacity(steps);
        let mut rollout_state = None;
        let reset_each_step = self.config.memory.reset_each_step;
        let detach_each_step = self.config.memory.detach_each_step;
        let state_clamp_min = self.config.state_clamp_min.min(self.config.state_clamp_max);
        let state_clamp_max = self.config.state_clamp_min.max(self.config.state_clamp_max);
        let step_noise_std = self.config.noise.input_std_scale.max(0.0);
        let step_noise_min = self.config.noise.input_std_min.max(0.0);
        let flow_backprop_steps = if detach_each_step {
            1usize
        } else {
            self.config
                .memory
                .flow_backprop_steps
                .unwrap_or(steps)
                .min(steps)
                .max(1)
        };
        let mut steps_since_detach = 0usize;

        if reverse {
            for (step_idx, idx) in (0..steps).rev().enumerate() {
                let dt = time_grid[idx + 1] - time_grid[idx];
                let time_value = time_grid[idx + 1];
                let flow_input = if self.config.noise.reverse_noise {
                    self.add_input_noise(current.clone(), step_noise_std, step_noise_min)
                } else {
                    current.clone()
                };
                let output = self.velocity.flow_step(
                    flow_input,
                    time_value,
                    dt,
                    -1.0,
                    rollout_state,
                    reset_each_step,
                );
                let delta = output.velocity_frames.clone().mul_scalar(dt);
                // RAC rollout states live in the shared image-state space, so keep them in-range.
                current = (current - delta).clamp(state_clamp_min, state_clamp_max);
                rollout_state = Some(if detach_each_step {
                    output.rollout_state.detach()
                } else {
                    output.rollout_state
                });
                step_tokens.push(output.patch_tokens);
                step_velocity_patches.push(output.velocity_patches);
                step_velocity_frames.push(output.velocity_frames);
                step_memory_read_norms.push(output.memory_read_norm);
                step_memory_write_norms.push(output.memory_write_norm);
                step_summaries.push(output.summary);
                states.push(current.clone());
                state_times.push(time_grid[idx]);
                if wipe_after_step.is_some_and(|wipe| wipe == step_idx + 1) {
                    rollout_state = None;
                }
                steps_since_detach += 1;
                if step_idx + 1 < steps && steps_since_detach >= flow_backprop_steps {
                    current = current.detach();
                    rollout_state = rollout_state.map(|state| state.detach());
                    steps_since_detach = 0;
                }
            }
        } else {
            for idx in 0..steps {
                let dt = time_grid[idx + 1] - time_grid[idx];
                let time_value = time_grid[idx];
                let flow_input =
                    self.add_input_noise(current.clone(), step_noise_std, step_noise_min);
                let output = self.velocity.flow_step(
                    flow_input,
                    time_value,
                    dt,
                    1.0,
                    rollout_state,
                    reset_each_step,
                );
                let delta = output.velocity_frames.clone().mul_scalar(dt);
                // RAC rollout states live in the shared image-state space, so keep them in-range.
                current = (current + delta).clamp(state_clamp_min, state_clamp_max);
                rollout_state = Some(if detach_each_step {
                    output.rollout_state.detach()
                } else {
                    output.rollout_state
                });
                step_tokens.push(output.patch_tokens);
                step_velocity_patches.push(output.velocity_patches);
                step_velocity_frames.push(output.velocity_frames);
                step_memory_read_norms.push(output.memory_read_norm);
                step_memory_write_norms.push(output.memory_write_norm);
                step_summaries.push(output.summary);
                states.push(current.clone());
                state_times.push(time_grid[idx + 1]);
                if wipe_after_step.is_some_and(|wipe| wipe == idx + 1) {
                    rollout_state = None;
                }
                steps_since_detach += 1;
                if idx + 1 < steps && steps_since_detach >= flow_backprop_steps {
                    current = current.detach();
                    rollout_state = rollout_state.map(|state| state.detach());
                    steps_since_detach = 0;
                }
            }
        }

        VisionRacTrajectory {
            final_state: current,
            states,
            step_tokens,
            step_velocity_patches,
            step_velocity_frames,
            step_memory_read_norms,
            step_memory_write_norms,
            step_summaries,
            state_times,
        }
    }

    fn path_consistency_loss(
        &self,
        trajectory: &VisionRacTrajectory<B>,
        start_state: Tensor<B, 4>,
        target_state: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let Some(first) = trajectory.states.first() else {
            return Tensor::<B, 1>::zeros([1], &start_state.device());
        };
        let device = first.device();
        let mut total = Tensor::<B, 1>::zeros([1], &device);
        let mut count = 0usize;
        for (state, time_value) in trajectory.states.iter().zip(trajectory.state_times.iter()) {
            let reference =
                self.straight_path_state(start_state.clone(), target_state.clone(), *time_value);
            total = total + self.mse4(state.clone(), reference);
            count += 1;
        }
        if count == 0 {
            Tensor::<B, 1>::zeros([1], &device)
        } else {
            total.div_scalar(count as f32)
        }
    }

    fn mean_velocity_loss(
        &self,
        trajectory: &VisionRacTrajectory<B>,
        start_state: Tensor<B, 4>,
        target_state: Tensor<B, 4>,
    ) -> Tensor<B, 1> {
        let Some(first_velocity) = trajectory.step_velocity_frames.first() else {
            return Tensor::<B, 1>::zeros([1], &start_state.device());
        };
        let [batch, channels, height, width] = first_velocity.shape().dims::<4>();
        let mut mean_velocity =
            Tensor::<B, 4>::zeros([batch, channels, height, width], &first_velocity.device());
        for velocity in &trajectory.step_velocity_frames {
            mean_velocity = mean_velocity + velocity.clone();
        }
        mean_velocity =
            mean_velocity.div_scalar(trajectory.step_velocity_frames.len().max(1) as f32);
        let target_velocity = target_state - start_state;
        self.mse4(mean_velocity, target_velocity)
    }

    fn trajectory_probe_summary(&self, trajectory: &VisionRacTrajectory<B>) -> Tensor<B, 2> {
        if let Some(summary) = trajectory.step_summaries.last() {
            summary.clone()
        } else {
            let device = trajectory.final_state.device();
            let batch = trajectory.final_state.shape().dims::<4>()[0];
            Tensor::<B, 2>::zeros(
                [batch, self.velocity.state_model.projection_dim().max(1)],
                &device,
            )
        }
    }

    fn eval_batch(
        &self,
        batch: VisionRacBatch<B>,
        capture_artifacts: bool,
    ) -> VisionRacEvalOutput<B> {
        let images = batch.images;
        let labels = batch.labels;
        let teacher_latent = batch.teacher_latent;
        let teacher_patch = batch.teacher_patch;
        let teacher_cls = batch.teacher_cls;
        let [_batch, _channels, height, width] = images.shape().dims::<4>();
        let device = images.device();
        let teacher_latent = self.teacher_target_latent(images.clone().detach(), teacher_latent);
        let target_state = self.image_to_state(images.clone());
        let init_state = self.teacher_latent_to_state(teacher_latent.clone(), height, width);
        let time_grid = self.build_time_grid(self.config.random_time_grid);
        let forward = self.integrate_flow(init_state.clone(), &time_grid, false, None);
        let recon = self.mse4_rgb(forward.final_state.clone(), target_state.clone());
        let forward_path =
            self.path_consistency_loss(&forward, init_state.clone(), target_state.clone());
        let velocity = self.mean_velocity_loss(&forward, init_state.clone(), target_state.clone());

        let eval_wipe_after_step = if capture_artifacts {
            self.config.memory.eval_wipe_after_step
        } else {
            None
        };
        let reverse =
            self.integrate_flow(target_state.clone(), &time_grid, true, eval_wipe_after_step);
        let reverse_latent =
            self.state_to_teacher_latent(reverse.final_state.clone(), &teacher_latent);
        let latent = self.mse4(reverse_latent.clone(), teacher_latent.clone());
        let reverse_to_init = self.mse4(reverse.final_state.clone(), init_state.clone());
        let reverse_path =
            self.path_consistency_loss(&reverse, init_state.clone(), target_state.clone());
        let block_target = self.teacher_latent_to_state(reverse_latent.clone(), height, width);
        let block_const = self.mse4(reverse.final_state.clone(), block_target);

        let roundtrip_init = self.teacher_latent_to_state(reverse_latent.clone(), height, width);
        let roundtrip_path =
            self.integrate_flow(roundtrip_init, &time_grid, false, eval_wipe_after_step);
        let roundtrip = self.mse4_rgb(roundtrip_path.final_state.clone(), target_state.clone());
        let roundtrip_state_path = self.integrate_flow(
            reverse.final_state.clone(),
            &time_grid,
            false,
            eval_wipe_after_step,
        );
        let roundtrip_state = self.mse4_rgb(
            roundtrip_state_path.final_state.clone(),
            target_state.clone(),
        );
        let reverse_summary = self.trajectory_probe_summary(&reverse);
        let semantic_cls_loss = match (&self.semantic_head, teacher_cls) {
            (Some(head), Some(target)) if self.config.semantic_teacher.weight > 0.0 => Some(
                self.cosine_alignment_loss(head.forward(reverse_summary.clone()), target.detach())
                    .mul_scalar(self.config.semantic_teacher.cls_weight.max(0.0)),
            ),
            _ => None,
        };
        let semantic_patch_loss = match (
            &self.semantic_patch_head,
            teacher_patch,
            reverse.step_tokens.last(),
        ) {
            (Some(head), Some(target), Some(tokens))
                if self.config.semantic_teacher.weight > 0.0 =>
            {
                let predicted = head.forward(tokens.clone());
                let aligned_target =
                    self.resample_patch_teacher(target.detach(), predicted.shape().dims::<3>()[1]);
                Some(
                    self.cosine_alignment_loss_tokens(predicted, aligned_target)
                        .mul_scalar(self.config.semantic_teacher.patch_weight.max(0.0)),
                )
            }
            _ => None,
        };
        let semantic_loss = match (semantic_cls_loss.clone(), semantic_patch_loss.clone()) {
            (Some(cls), Some(patch)) => {
                let denom = (self.config.semantic_teacher.cls_weight.max(0.0)
                    + self.config.semantic_teacher.patch_weight.max(0.0))
                .max(1.0e-6);
                Some((cls + patch).div_scalar(denom))
            }
            (Some(cls), None) => Some(cls),
            (None, Some(patch)) => Some(patch),
            (None, None) => None,
        };

        let aux = forward_path
            .clone()
            .mul_scalar(self.config.loss.path_weight.max(0.0))
            + reverse_path
                .clone()
                .mul_scalar(self.config.loss.reverse_path_weight.max(0.0))
            + latent
                .clone()
                .mul_scalar(self.config.loss.latent_weight.max(0.0))
            + velocity
                .clone()
                .mul_scalar(self.config.loss.velocity_weight.max(0.0));
        let total = recon
            .clone()
            .mul_scalar(self.config.loss.recon_weight.max(0.0))
            + aux.clone()
            + reverse_to_init
                .clone()
                .mul_scalar(self.config.loss.state_align_weight.max(0.0))
            + roundtrip
                .clone()
                .mul_scalar(self.config.loss.roundtrip_weight.max(0.0))
            + roundtrip_state
                .clone()
                .mul_scalar(self.config.loss.roundtrip_state_weight.max(0.0))
            + block_const
                .clone()
                .mul_scalar(self.config.loss.block_const_weight.max(0.0));
        let total = total
            + semantic_loss
                .clone()
                .unwrap_or_else(|| Tensor::<B, 1>::zeros([1], &device))
                .mul_scalar(self.config.semantic_teacher.weight.max(0.0));

        let probe_logits = self.probe.forward(reverse_summary);
        let probe_loss = if self.config.loss.probe_weight > 0.0 {
            self.probe_loss
                .forward(probe_logits.clone(), labels.clone())
        } else {
            Tensor::<B, 1>::zeros([1], &device)
        };
        let probe_pred = probe_logits
            .clone()
            .argmax(1)
            .reshape([labels.shape().dims::<1>()[0]]);
        let probe_acc = probe_pred.equal(labels.clone()).float().mean();

        let total = total
            + probe_loss
                .clone()
                .mul_scalar(self.config.loss.probe_weight.max(0.0));
        let recon_psnr_value = recon_psnr(recon.clone());
        let roundtrip_psnr_value = recon_psnr(roundtrip.clone());

        let artifacts = if capture_artifacts
            && self.config.artifact_every > 0
            && self.config.artifact_max_images > 0
        {
            build_rac_artifacts(
                target_state.clone().detach(),
                init_state.clone().detach(),
                &forward,
                &reverse,
                &roundtrip_path,
                labels.clone(),
                probe_logits.clone(),
                &self.velocity.memory_component_names(),
                self.velocity.state_model.patch_size().max(1),
                self.config.artifact_max_images.max(1),
                self.config.artifact_upscale.max(1),
                self.config.memory.reset_each_step,
                self.config.memory.detach_each_step,
                if self.config.memory.detach_each_step {
                    Some(1)
                } else {
                    self.config.memory.flow_backprop_steps
                },
                self.config.memory.disable_writes,
                eval_wipe_after_step,
            )
        } else {
            None
        };

        VisionRacEvalOutput {
            images,
            forward_state: forward.final_state,
            roundtrip_state: roundtrip_path.final_state,
            total,
            recon,
            aux,
            roundtrip,
            roundtrip_state_loss: roundtrip_state,
            recon_psnr: recon_psnr_value,
            roundtrip_psnr: roundtrip_psnr_value,
            forward_path,
            reverse_path,
            forward_velocity: velocity,
            reverse_latent: latent,
            reverse_to_init,
            block_const,
            semantic_loss,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }

    fn forward_losses(
        &self,
        batch: VisionRacBatch<B>,
        capture_artifacts: bool,
    ) -> VisionRacLosses<B> {
        let output = self.eval_batch(batch, capture_artifacts);
        VisionRacLosses {
            total: output.total,
            recon: output.recon,
            aux: output.aux,
            roundtrip: output.roundtrip,
            roundtrip_state: output.roundtrip_state_loss,
            recon_psnr: output.recon_psnr,
            roundtrip_psnr: output.roundtrip_psnr,
            forward_path: output.forward_path,
            reverse_path: output.reverse_path,
            forward_velocity: output.forward_velocity,
            reverse_latent: output.reverse_latent,
            reverse_to_init: output.reverse_to_init,
            block_const: output.block_const,
            semantic_loss: output.semantic_loss,
            probe_loss: output.probe_loss,
            probe_acc: output.probe_acc,
            artifacts: output.artifacts,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for VisionRacModel<B> {
    type Input = VisionRacBatch<B>;
    type Output = VisionTrainItem<B>;

    fn step(&self, batch: VisionRacBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        crate::device::pin_stream_zero();
        let losses = self.forward_losses(batch, false);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), self);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        let item = VisionTrainItem::new(
            losses.total,
            losses.recon,
            losses.aux,
            zero.clone(),
            zero.clone(),
            losses.roundtrip.clone(),
            losses.recon_psnr,
            losses.roundtrip_psnr,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            losses.probe_loss,
            losses.probe_acc,
        )
        .with_directional_metrics(
            Some(losses.forward_path),
            Some(losses.reverse_path),
            Some(losses.forward_velocity),
            Some(losses.reverse_latent),
            Some(losses.reverse_to_init),
            Some(losses.roundtrip_state),
            Some(losses.block_const),
        )
        .with_semantic_loss(losses.semantic_loss);
        TrainOutput { grads, item }
    }
}

impl<B: BackendTrait> ValidStep for VisionRacModel<B> {
    type Input = VisionRacBatch<B>;
    type Output = VisionOutput<B>;

    fn step(&self, batch: VisionRacBatch<B>) -> VisionOutput<B> {
        crate::device::pin_stream_zero();
        let losses = self.forward_losses(batch, self.config.artifact_every > 0);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            losses.recon,
            losses.aux,
            zero.clone(),
            zero.clone(),
            losses.roundtrip.clone(),
            losses.recon_psnr,
            losses.roundtrip_psnr,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            losses.probe_loss,
            losses.probe_acc,
            losses.artifacts,
        )
        .with_directional_metrics(
            Some(losses.forward_path),
            Some(losses.reverse_path),
            Some(losses.forward_velocity),
            Some(losses.reverse_latent),
            Some(losses.reverse_to_init),
            Some(losses.roundtrip_state),
            Some(losses.block_const),
        )
        .with_semantic_loss(losses.semantic_loss)
    }
}
