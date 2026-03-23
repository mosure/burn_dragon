use super::dynamics::embed_clip_frames_raw_with_model;
use crate::model::VisionRolloutState;
use crate::train::prelude::*;

const VJEPA21_EPS: f32 = 1.0e-6;

#[derive(Module, Debug)]
struct VisionVideoVjepa21Predictor<B: BackendTrait> {
    norm: DragonNorm<B>,
    hidden: Option<Linear<B>>,
    out: Linear<B>,
}

impl<B: BackendTrait> VisionVideoVjepa21Predictor<B> {
    fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, input_dim.max(1), device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(input_dim.max(1), hidden_dim.max(1)).init(device))
        } else {
            None
        };
        let out_in = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            input_dim.max(1)
        };
        let out = LinearConfig::new(out_in, output_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    fn forward(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionVideoVjepa21Model<B: BackendTrait> {
    pub(crate) frame_model: VisionDragon<B>,
    predictor: VisionVideoVjepa21Predictor<B>,
    input_mask_token: Param<Tensor<B, 2>>,
    predictor_mask_bias: Option<Param<Tensor<B, 2>>>,
    pub(crate) teacher_frame_model: Option<VisionDragon<B>>,
    pub(crate) config: VisionVideoLejepaConfig,
}

#[derive(Debug, Clone)]
struct Vjepa21MaskBatch<B: BackendTrait> {
    visible: Tensor<B, 3>,
    target: Tensor<B, 3>,
    context_distance: Tensor<B, 3>,
    visible_ratio: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
struct VisionVideoVjepa21Losses<B: BackendTrait> {
    total: Tensor<B, 1>,
    masked: Tensor<B, 1>,
    context: Tensor<B, 1>,
    visible_ratio: Tensor<B, 1>,
}

impl<B: BackendTrait> VisionVideoVjepa21Model<B> {
    pub(crate) fn new(
        frame_model: VisionDragon<B>,
        config: VisionVideoLejepaConfig,
        vision: &VisionDragonConfig,
        device: &B::Device,
    ) -> Self {
        let level_count = 1 + sanitize_checkpoint_depths(&config.vjepa21.checkpoint_depths).len();
        let projection_dim = vision.projection_dim.max(1);
        let input_dim = projection_dim * level_count;
        let predictor_hidden_dim = if config.vjepa21.predictor_hidden_dim == 0 {
            input_dim.max(vision.projection_hidden_dim.max(projection_dim))
        } else {
            config.vjepa21.predictor_hidden_dim.max(1)
        };
        let predictor = VisionVideoVjepa21Predictor::new(
            input_dim,
            predictor_hidden_dim,
            input_dim,
            &vision.normalization,
            device,
        );
        let input_mask_token = Param::from_tensor(Tensor::<B, 2>::random(
            [1, vision.embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let predictor_mask_bias = config.vjepa21.use_mask_token_bias.then(|| {
            Param::from_tensor(Tensor::<B, 2>::random(
                [1, input_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            ))
        });
        let teacher_frame_model =
            init_momentum_teacher::<B, _>(&frame_model, &config.vjepa21.teacher_ema);

        Self {
            frame_model,
            predictor,
            input_mask_token,
            predictor_mask_bias,
            teacher_frame_model,
            config,
        }
    }

    pub(crate) fn sync_teacher_from_student(mut self) -> Self {
        self.teacher_frame_model = sync_optional_teacher_from_student::<B, _>(
            self.teacher_frame_model.take(),
            &self.frame_model,
            self.config.vjepa21.teacher_ema.enabled,
            self.config.vjepa21.teacher_ema.decay,
        );
        self
    }

    pub(crate) fn restore_teacher_from_student(mut self) -> Self {
        self.teacher_frame_model = restore_optional_teacher_from_student::<B, _>(
            &self.frame_model,
            self.config.vjepa21.teacher_ema.enabled,
        );
        self
    }

    fn checkpoint_depths(&self) -> Vec<usize> {
        sanitize_checkpoint_depths(&self.config.vjepa21.checkpoint_depths)
    }

    fn effective_observe_steps(&self) -> usize {
        self.config.vjepa21.observe_steps.max(1)
    }

    fn effective_observe_backprop_steps(&self) -> usize {
        self.config
            .vjepa21
            .observe_backprop_steps
            .clamp(1, self.effective_observe_steps())
    }

    fn forward_losses(&self, batch: VideoClipBatch<B>) -> VisionVideoVjepa21Losses<B> {
        let clip_frames = batch.clip_frames;
        let raw_tokens = embed_clip_frames_raw_with_model(&self.frame_model, clip_frames);
        let [batch_size, clip_len, patch_count, embed_dim] = raw_tokens.shape().dims::<4>();
        let projection_dim = self
            .frame_model
            .project_tokens(Tensor::<B, 3>::zeros(
                [1, 1, embed_dim.max(1)],
                &raw_tokens.device(),
            ))
            .shape()
            .dims::<3>()[2]
            .max(1);

        let masks = sample_mask_batch::<B>(
            batch_size,
            clip_len,
            patch_count,
            &self.config.vjepa21,
            &raw_tokens.device(),
        );
        let mask_token = self
            .input_mask_token
            .val()
            .reshape([1, 1, 1, embed_dim.max(1)])
            .repeat_dim(0, batch_size)
            .repeat_dim(1, clip_len)
            .repeat_dim(2, patch_count);
        let target_mask = masks.target.clone().unsqueeze_dim::<4>(3);
        let visible_mask = masks.visible.clone().unsqueeze_dim::<4>(3);
        let student_tokens = raw_tokens.clone() * visible_mask + mask_token * target_mask.clone();

        let checkpoint_depths = self.checkpoint_depths();
        let student_levels = encode_clip_with_checkpoints(
            &self.frame_model,
            student_tokens,
            self.effective_observe_steps(),
            self.effective_observe_backprop_steps(),
            &checkpoint_depths,
        );
        let teacher_source = self
            .teacher_frame_model
            .as_ref()
            .unwrap_or(&self.frame_model);
        let teacher_levels = encode_clip_with_checkpoints(
            teacher_source,
            raw_tokens,
            self.effective_observe_steps(),
            self.effective_observe_steps(),
            &checkpoint_depths,
        )
        .into_iter()
        .map(Tensor::detach)
        .collect::<Vec<_>>();

        let student_concat = Tensor::cat(student_levels.clone(), 3);
        let mut predictor_input = student_concat.clone();
        if let Some(mask_bias) = &self.predictor_mask_bias {
            let bias = mask_bias
                .val()
                .reshape([1, 1, 1, student_concat.shape().dims::<4>()[3]])
                .repeat_dim(0, batch_size)
                .repeat_dim(1, clip_len)
                .repeat_dim(2, patch_count);
            predictor_input = predictor_input + bias * target_mask;
        }
        let predicted_concat = self.predictor.forward(predictor_input);
        let predicted_levels = split_levels(predicted_concat, teacher_levels.len(), projection_dim);
        let visible_flat = masks.visible.reshape([batch_size, clip_len * patch_count]);
        let target_flat = masks.target.reshape([batch_size, clip_len * patch_count]);
        let distance_flat = masks
            .context_distance
            .reshape([batch_size, clip_len * patch_count]);

        let masked_loss = stacked_level_loss(
            &predicted_levels,
            &teacher_levels,
            target_flat,
            None,
            self.config.vjepa21.loss.normalize_targets,
            self.config.vjepa21.loss.loss_exp,
        );
        let context_loss = if self.config.vjepa21.loss.predict_all
            && self.config.vjepa21.loss.context_weight > 0.0
        {
            let weights = if self.config.vjepa21.loss.weight_distance_loss {
                Some(distance_flat)
            } else {
                None
            };
            stacked_level_loss(
                &predicted_levels,
                &teacher_levels,
                visible_flat,
                weights,
                self.config.vjepa21.loss.normalize_targets,
                self.config.vjepa21.loss.loss_exp,
            )
        } else {
            Tensor::<B, 1>::zeros([1], &masks.visible_ratio.device())
        };
        let total = masked_loss
            .clone()
            .mul_scalar(self.config.vjepa21.loss.masked_weight)
            + context_loss
                .clone()
                .mul_scalar(self.config.vjepa21.loss.context_weight);

        VisionVideoVjepa21Losses {
            total,
            masked: masked_loss,
            context: context_loss,
            visible_ratio: masks.visible_ratio,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for VisionVideoVjepa21Model<B> {
    type Input = VideoClipBatch<B>;
    type Output = VisionTrainItem<B>;

    fn step(&self, batch: VideoClipBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        crate::device::pin_stream_zero();
        let losses = self.forward_losses(batch);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), self);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        let item = VisionTrainItem::new(
            losses.total,
            losses.masked,
            losses.context,
            losses.visible_ratio,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        );
        TrainOutput { grads, item }
    }

    fn optimize<BB, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: burn::optim::Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        crate::device::pin_stream_zero();
        let model = optim.step(lr, self, grads);
        model.sync_teacher_from_student()
    }
}

impl<B: BackendTrait> ValidStep for VisionVideoVjepa21Model<B> {
    type Input = VideoClipBatch<B>;
    type Output = VisionOutput<B>;

    fn step(&self, batch: VideoClipBatch<B>) -> VisionOutput<B> {
        crate::device::pin_stream_zero();
        let losses = self.forward_losses(batch);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            losses.masked,
            losses.context,
            losses.visible_ratio,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            None,
        )
    }
}

fn sanitize_checkpoint_depths(depths: &[usize]) -> Vec<usize> {
    let mut out = depths
        .iter()
        .copied()
        .filter(|depth| *depth > 0)
        .collect::<Vec<_>>();
    out.sort_unstable();
    out.dedup();
    out
}

fn l2_normalize_last_dim<const D: usize, B: BackendTrait>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let norm = tensor
        .clone()
        .powf_scalar(2.0)
        .sum_dim(D - 1)
        .sqrt()
        .add_scalar(VJEPA21_EPS);
    tensor / norm
}

fn split_levels<B: BackendTrait>(
    tensor: Tensor<B, 4>,
    level_count: usize,
    projection_dim: usize,
) -> Vec<Tensor<B, 4>> {
    (0..level_count)
        .map(|level| {
            let start = level * projection_dim;
            let end = (start + projection_dim).min(tensor.shape().dims::<4>()[3]);
            tensor.clone().slice_dim(3, start..end)
        })
        .collect()
}

fn masked_token_loss<B: BackendTrait>(
    predicted: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Tensor<B, 2>,
    weights: Option<Tensor<B, 2>>,
    normalize_targets: bool,
    loss_exp: f32,
) -> Tensor<B, 1> {
    let [batch, clip_len, patch_count, _] = predicted.shape().dims::<4>();
    let total_tokens = clip_len * patch_count;
    let predicted_dim = predicted.shape().dims::<4>()[3];
    let target_dim = target.shape().dims::<4>()[3];
    let predicted = predicted.reshape([batch, total_tokens, predicted_dim]);
    let target = target.reshape([batch, total_tokens, target_dim]);
    let predicted = if normalize_targets {
        l2_normalize_last_dim(predicted)
    } else {
        predicted
    };
    let target = if normalize_targets {
        l2_normalize_last_dim(target)
    } else {
        target
    };
    let per_token = (predicted - target)
        .abs()
        .powf_scalar(loss_exp.max(1.0))
        .mean_dim(2)
        .reshape([batch, total_tokens]);
    let token_weights = if let Some(weights) = weights {
        mask.clone() * weights.recip()
    } else {
        mask
    };
    let denom = token_weights.clone().sum().add_scalar(VJEPA21_EPS);
    per_token.mul(token_weights).sum().div(denom).reshape([1])
}

fn stacked_level_loss<B: BackendTrait>(
    predicted: &[Tensor<B, 4>],
    target: &[Tensor<B, 4>],
    mask: Tensor<B, 2>,
    weights: Option<Tensor<B, 2>>,
    normalize_targets: bool,
    loss_exp: f32,
) -> Tensor<B, 1> {
    let device = predicted
        .first()
        .expect("stacked loss requires at least one level")
        .device();
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let mut count = 0usize;
    for (predicted_level, target_level) in predicted.iter().zip(target.iter()) {
        total = total
            + masked_token_loss(
                predicted_level.clone(),
                target_level.clone().detach(),
                mask.clone(),
                weights.clone(),
                normalize_targets,
                loss_exp,
            );
        count += 1;
    }
    total.div_scalar(count.max(1) as f32)
}

fn encode_clip_with_checkpoints<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_tokens: Tensor<B, 4>,
    observe_steps: usize,
    observe_backprop_steps: usize,
    checkpoint_depths: &[usize],
) -> Vec<Tensor<B, 4>> {
    let [batch_size, clip_len, patch_count, embed_dim] = clip_tokens.shape().dims::<4>();
    let level_count = 1 + checkpoint_depths.len();
    let mut level_outputs = (0..level_count)
        .map(|_| Vec::with_capacity(clip_len))
        .collect::<Vec<_>>();
    let mut carry_state: Option<VisionRolloutState<B>> = None;
    let checkpoint_schedule = checkpoint_depths
        .iter()
        .map(|depth| (*depth, *depth))
        .collect::<Vec<_>>();

    for frame_idx in 0..clip_len {
        let tokens = clip_tokens
            .clone()
            .slice_dim(1, frame_idx..frame_idx + 1)
            .reshape([batch_size, patch_count, embed_dim]);
        let base_state = if let Some(state) = carry_state.take() {
            model.observe_rollout_state_with_tokens_unbounded(
                state,
                tokens,
                observe_steps.max(1),
                observe_backprop_steps.max(1),
            )
        } else {
            model.refine_rollout_state_unbounded(
                model.rollout_state_from_tokens(tokens),
                observe_steps.max(1),
                observe_backprop_steps.max(1),
            )
        };
        level_outputs[0].push(model.forward_rollout_state(&base_state).patch_tokens);

        let mut final_state = base_state.clone();
        if !checkpoint_schedule.is_empty() {
            let scheduled =
                model.refine_rollout_state_schedule_unbounded(base_state, &checkpoint_schedule);
            for (level_idx, (_, state)) in scheduled.iter().enumerate() {
                level_outputs[level_idx + 1].push(model.forward_rollout_state(state).patch_tokens);
            }
            if let Some((_, state)) = scheduled.last() {
                final_state = state.clone();
            }
        }
        carry_state = Some(final_state);
    }

    level_outputs
        .into_iter()
        .map(|frames| Tensor::stack::<4>(frames, 1))
        .collect()
}

fn sample_mask_batch<B: BackendTrait>(
    batch_size: usize,
    clip_len: usize,
    patch_count: usize,
    config: &VisionVideoVjepa21Config,
    device: &B::Device,
) -> Vjepa21MaskBatch<B> {
    let grid_h = (patch_count as f64).sqrt() as usize;
    let grid_h = grid_h.max(1);
    let grid_w = patch_count.div_ceil(grid_h).max(1);
    let total_tokens = clip_len * patch_count;
    let mut rng = thread_rng();
    let mut visible_data = vec![0.0_f32; batch_size * total_tokens];
    let mut target_data = vec![0.0_f32; batch_size * total_tokens];
    let mut distance_data = vec![1.0_f32; batch_size * total_tokens];
    let max_context_duration = ((clip_len as f32) * config.mask.max_context_frames_ratio)
        .floor()
        .max(1.0) as usize;
    let max_context_duration = max_context_duration.clamp(1, clip_len.max(1));

    for batch_idx in 0..batch_size {
        let mut visible = vec![true; total_tokens];
        for _ in 0..config.mask.num_blocks.max(1) {
            let temporal_scale = rng.gen_range(
                config
                    .mask
                    .temporal_scale_min
                    .min(config.mask.temporal_scale_max)
                    ..=config
                        .mask
                        .temporal_scale_min
                        .max(config.mask.temporal_scale_max),
            );
            let temporal_span = ((clip_len as f32) * temporal_scale).round() as usize;
            let temporal_span = temporal_span.clamp(1, clip_len.max(1));

            let spatial_scale = rng.gen_range(
                config
                    .mask
                    .spatial_scale_min
                    .min(config.mask.spatial_scale_max)
                    ..=config
                        .mask
                        .spatial_scale_min
                        .max(config.mask.spatial_scale_max),
            );
            let spatial_keep = ((patch_count as f32) * spatial_scale).round() as usize;
            let spatial_keep = spatial_keep.clamp(1, patch_count.max(1));

            let aspect_ratio = rng.gen_range(
                config
                    .mask
                    .aspect_ratio_min
                    .min(config.mask.aspect_ratio_max)
                    ..=config
                        .mask
                        .aspect_ratio_min
                        .max(config.mask.aspect_ratio_max),
            );
            let mut block_h = ((spatial_keep as f32 * aspect_ratio).sqrt().round() as usize)
                .clamp(1, grid_h.max(1));
            let mut block_w = ((spatial_keep as f32 / aspect_ratio).sqrt().round() as usize)
                .clamp(1, grid_w.max(1));
            if block_h * block_w > patch_count {
                block_h = block_h.min(grid_h.max(1));
                block_w = block_w.min(grid_w.max(1));
            }
            let top = if grid_h > block_h {
                rng.gen_range(0..=grid_h - block_h)
            } else {
                0
            };
            let left = if grid_w > block_w {
                rng.gen_range(0..=grid_w - block_w)
            } else {
                0
            };
            let start_t = if clip_len > temporal_span {
                rng.gen_range(0..=clip_len - temporal_span)
            } else {
                0
            };

            for t in start_t..(start_t + temporal_span).min(clip_len) {
                for y in top..(top + block_h).min(grid_h) {
                    for x in left..(left + block_w).min(grid_w) {
                        let token_idx = y * grid_w + x;
                        if token_idx < patch_count {
                            visible[t * patch_count + token_idx] = false;
                        }
                    }
                }
            }
        }
        for t in max_context_duration..clip_len {
            for token_idx in 0..patch_count {
                visible[t * patch_count + token_idx] = false;
            }
        }
        if visible.iter().all(|flag| !*flag) {
            visible[0] = true;
        }
        let masked_positions = visible
            .iter()
            .enumerate()
            .filter_map(|(idx, visible)| (!*visible).then_some(idx))
            .collect::<Vec<_>>();
        if masked_positions.is_empty() {
            visible[0] = false;
        }
        let masked_positions = visible
            .iter()
            .enumerate()
            .filter_map(|(idx, visible)| (!*visible).then_some(idx))
            .collect::<Vec<_>>();
        let mut visible_count = 0usize;
        for token_idx in 0..total_tokens {
            let row = batch_idx * total_tokens + token_idx;
            let is_visible = visible[token_idx];
            visible_data[row] = if is_visible { 1.0 } else { 0.0 };
            target_data[row] = if is_visible { 0.0 } else { 1.0 };
            if is_visible {
                visible_count += 1;
                let t = token_idx / patch_count;
                let local = token_idx % patch_count;
                let y = local / grid_w;
                let x = local % grid_w;
                let mut min_distance = f32::MAX;
                for &masked in &masked_positions {
                    let masked_t = masked / patch_count;
                    let masked_local = masked % patch_count;
                    let masked_y = masked_local / grid_w;
                    let masked_x = masked_local % grid_w;
                    let dt = masked_t as f32 - t as f32;
                    let dy = masked_y as f32 - y as f32;
                    let dx = masked_x as f32 - x as f32;
                    let distance = (dt * dt + dy * dy + dx * dx).sqrt().max(1.0);
                    min_distance = min_distance.min(distance);
                }
                let offset = if config.loss.offset_context_loss {
                    (grid_w.max(grid_h) / 16).max(1) as f32
                } else {
                    1.0
                };
                distance_data[row] = (min_distance / offset).sqrt().max(1.0);
            } else {
                distance_data[row] = 1.0;
            }
        }
        let visible_ratio = visible_count as f32 / total_tokens.max(1) as f32;
        distance_data[batch_idx * total_tokens] = distance_data[batch_idx * total_tokens].max(1.0);
        target_data[batch_idx * total_tokens] = target_data[batch_idx * total_tokens].max(0.0);
        visible_data[batch_idx * total_tokens] = visible_data[batch_idx * total_tokens].max(0.0);
        let _ = visible_ratio;
    }

    let visible = Tensor::<B, 3>::from_data(
        TensorData::new(visible_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let target = Tensor::<B, 3>::from_data(
        TensorData::new(target_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let context_distance = Tensor::<B, 3>::from_data(
        TensorData::new(distance_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let visible_ratio = visible
        .clone()
        .sum()
        .div_scalar((batch_size * total_tokens).max(1) as f32)
        .reshape([1]);

    Vjepa21MaskBatch {
        visible,
        target,
        context_distance,
        visible_ratio,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::optim::{AdamWConfig, GradientsParams, LearningRate};
    use burn_autodiff::Autodiff;
    use burn_dragon_core::FusedKernelConfig;
    use burn_ndarray::NdArray;

    fn make_config(
        backbone: crate::VisionBackboneKind,
    ) -> (VisionDragonConfig, VisionVideoLejepaConfig) {
        let mut vision = VisionDragonConfig {
            image_size: 16,
            patch_size: 4,
            patch_embed_mode: VisionPatchEmbedMode::default(),
            backbone,
            in_channels: 3,
            embed_dim: 24,
            steps: 2,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            dropout: 0.0,
            projection_dim: 12,
            projection_hidden_dim: 24,
            use_cls_token: true,
            cls_sync_alpha: 0.0,
            num_eyes: 1,
            cross_eye_steps: 0,
            token_state_norm: true,
            normalization: DragonNormConfig::default(),
            latent_activation: VisionLatentActivation::default(),
            pos_encoding: SpatialPositionalEncodingKind::Rope,
            pos_max_height: 4,
            pos_max_width: 4,
            attention_mode: VisionAttentionMode::RowL1,
            use_alibi: false,
            fused_kernels: FusedKernelConfig::default(),
            mhc: Default::default(),
            trm_graph: Default::default(),
            rho_stream: crate::VisionRhoStreamConfig {
                enabled: false,
                local_radius: 1,
                local_diagonals: true,
                local_self: true,
                decay: 0.9,
                mode_embeddings: true,
                wgpu_forward_kernel: false,
                wgpu_rollout_fused: false,
            },
        };
        if matches!(backbone, crate::VisionBackboneKind::Pyramid) {
            vision.trm_graph.enabled = true;
            vision.trm_graph.coarse_stride = 2;
            vision.trm_graph.hub_count = 2;
            vision.trm_graph.rank = 4;
            vision.trm_graph.value_dim = 12;
            vision.trm_graph.local_radius = 1;
            vision.trm_graph.local_diagonals = true;
            vision.trm_graph.local_self = true;
            vision.trm_graph.decay = 0.9;
        }
        if matches!(backbone, crate::VisionBackboneKind::Cellular) {
            vision.rho_stream.enabled = true;
        }
        let mut video = VisionVideoLejepaConfig::default();
        video.paradigm = VisionVideoParadigmKind::Vjepa21;
        video.frame_stride = 1;
        video.vjepa21.clip_frames = 6;
        video.vjepa21.observe_steps = 1;
        video.vjepa21.observe_backprop_steps = 1;
        video.vjepa21.predictor_hidden_dim = 32;
        video.vjepa21.checkpoint_depths = vec![1, 2];
        video.vjepa21.loss.context_weight = 0.5;
        video.vjepa21.loss.predict_all = true;
        video.vjepa21.loss.weight_distance_loss = true;
        video.vjepa21.mask.num_blocks = 2;
        video.vjepa21.mask.temporal_scale_min = 0.5;
        video.vjepa21.mask.temporal_scale_max = 1.0;
        (vision, video)
    }

    fn toy_batch<B: BackendTrait>(device: &B::Device) -> VideoClipBatch<B> {
        let batch = 2;
        let clip_len = 6;
        let channels = 3;
        let size = 16;
        let mut data = vec![0.0_f32; batch * clip_len * channels * size * size];
        for b in 0..batch {
            for t in 0..clip_len {
                let x = (t + b) % (size - 4);
                let y = (2 * t + b) % (size - 4);
                for c in 0..channels {
                    for dy in 0..4 {
                        for dx in 0..4 {
                            let idx = ((((b * clip_len + t) * channels + c) * size + (y + dy))
                                * size)
                                + x
                                + dx;
                            data[idx] = 1.0;
                        }
                    }
                }
            }
        }
        let clip_frames = Tensor::<B, 5>::from_data(
            TensorData::new(data, [batch, clip_len, channels, size, size]),
            device,
        );
        let labels = Tensor::<B, 1, Int>::zeros([batch], device);
        VideoClipBatch::new(clip_frames, labels, clip_len, 0)
    }

    #[test]
    fn vjepa21_mask_sampler_keeps_targets_and_context() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let (_, video) = make_config(crate::VisionBackboneKind::Dense);
        let masks = sample_mask_batch::<Backend>(2, 6, 16, &video.vjepa21, &device);
        let visible = masks
            .visible
            .sum()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("visible sum")[0];
        let target = masks
            .target
            .sum()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("target sum")[0];
        assert!(visible > 0.0);
        assert!(target > 0.0);
    }

    #[test]
    fn vjepa21_forward_losses_are_finite_for_all_backbones() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        for backbone in [
            crate::VisionBackboneKind::Dense,
            crate::VisionBackboneKind::Pyramid,
            crate::VisionBackboneKind::Cellular,
        ] {
            let (vision, video) = make_config(backbone);
            let model = VisionVideoVjepa21Model::<Backend>::new(
                VisionDragon::<Backend>::new(vision.clone(), &device),
                video,
                &vision,
                &device,
            );
            let losses = model.forward_losses(toy_batch::<Backend>(&device));
            let total = losses
                .total
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("total")[0];
            assert!(total.is_finite());
            assert!(total > 0.0);
        }
    }

    #[test]
    fn vjepa21_dense_prediction_improves_on_toy_batch() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = <Backend as BackendTrait>::Device::default();
        let (vision, video) = make_config(crate::VisionBackboneKind::Dense);
        let mut model = VisionVideoVjepa21Model::<Backend>::new(
            VisionDragon::<Backend>::new(vision.clone(), &device),
            video,
            &vision,
            &device,
        );
        let batch = toy_batch::<Backend>(&device);
        let initial = model
            .forward_losses(batch.clone())
            .masked
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("initial")[0];
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<Backend, VisionVideoVjepa21Model<Backend>>();
        let lr: LearningRate = 1.0e-2;
        for _ in 0..30 {
            let losses = model.forward_losses(batch.clone());
            let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
            model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);
        }
        let final_loss = model
            .forward_losses(batch)
            .masked
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("final")[0];
        assert!(final_loss.is_finite());
        assert!(final_loss < initial);
    }
}
