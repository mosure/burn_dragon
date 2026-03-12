use super::*;

type RolloutHorizonMetricArray<B> = [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT];
type RolloutHorizonMetricArrays<B> = (
    RolloutHorizonMetricArray<B>,
    RolloutHorizonMetricArray<B>,
    RolloutHorizonMetricArray<B>,
);

pub(super) struct VideoDebugReconInput<B: BackendTrait> {
    pub clip_frames: Tensor<B, 5>,
    pub frame_patch_tokens: Tensor<B, 4>,
    pub future_patch_tokens: Tensor<B, 4>,
    pub context_len: usize,
    pub target_len: usize,
    pub future_len_all: usize,
    pub steps: usize,
    pub capture_artifacts: bool,
}

impl<B: BackendTrait> VisionVideoLejepaModel<B> {
    pub(super) fn reconstruct_patch_values_from_patch_tokens(
        &self,
        patch_tokens: Tensor<B, 4>,
        channels: usize,
    ) -> Tensor<B, 4> {
        let [batch, frames, tokens, _] = patch_tokens.shape().dims::<4>();
        let patch_size = self.frame_model.patch_size().max(1);
        let patch_dim = channels * patch_size * patch_size;
        self.recon
            .forward(patch_tokens)
            .reshape([batch, frames, tokens, patch_dim])
    }

    pub(super) fn reconstruct_frames_from_patch_values(
        &self,
        patch_values: Tensor<B, 4>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Tensor<B, 4> {
        let [batch, frames, tokens, patch_dim] = patch_values.shape().dims::<4>();
        let patch_size = self.frame_model.patch_size().max(1);
        let patches = patch_values.reshape([batch * frames, tokens, patch_dim]);
        unpatchify(patches, patch_size, height, width, channels)
    }

    #[allow(dead_code)]
    pub(super) fn reconstruct_frames_from_patch_tokens(
        &self,
        patch_tokens: Tensor<B, 4>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Tensor<B, 4> {
        let patch_values = self.reconstruct_patch_values_from_patch_tokens(patch_tokens, channels);
        self.reconstruct_frames_from_patch_values(patch_values, height, width, channels)
    }

    pub(super) fn apply_step_mode(
        &self,
        patch_tokens: Tensor<B, 3>,
        mode: StructuredStepMode,
    ) -> Tensor<B, 3> {
        let Some(step_mode_embeddings) = &self.step_mode_embeddings else {
            return patch_tokens;
        };
        let [batch, patch_count, embed_dim] = patch_tokens.shape().dims::<3>();
        let bias = step_mode_embeddings
            .val()
            .slice_dim(0, mode.index()..mode.index() + 1)
            .reshape([1, 1, embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, patch_count);
        patch_tokens + bias
    }

    pub(super) fn refine_passes(&self) -> usize {
        self.config.temporal.refine_passes
    }

    pub(super) fn maybe_refine_patch_state(
        &self,
        mut output: VisionVideoRolloutOutput<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionVideoRolloutOutput<B> {
        for _ in 0..self.refine_passes() {
            output = self.rollout_conditioned_patch_step(
                output.patch_tokens,
                None,
                steps,
                backprop_steps,
                StructuredStepMode::Refine,
            );
        }
        output
    }

    pub(super) fn rollout_pyramid_step(
        &self,
        state: StructuredTopologyState<B>,
        global_condition: Option<Tensor<B, 2>>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> StructuredTopologyState<B> {
        let mut patch_tokens = self.frame_model.pyramid_patch_tokens(&state);
        let [batch, patch_count, embed_dim] = patch_tokens.shape().dims::<3>();
        if let Some(global_condition) = global_condition {
            let cond = self
                .patch_conditioner
                .forward(global_condition)
                .reshape([batch, 1, embed_dim])
                .repeat_dim(1, patch_count);
            patch_tokens = patch_tokens + cond;
        }
        let patch_tokens = self.apply_step_mode(patch_tokens, mode);
        let state = self
            .frame_model
            .pyramid_state_with_patch_tokens(state, patch_tokens);
        self.frame_model
            .forward_pyramid_state_rollout_mode_unbounded(state, steps, backprop_steps, mode)
    }

    pub(super) fn maybe_refine_pyramid_state(
        &self,
        mut state: StructuredTopologyState<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> StructuredTopologyState<B> {
        for _ in 0..self.refine_passes() {
            state = self.rollout_pyramid_step(
                state,
                None,
                steps,
                backprop_steps,
                StructuredStepMode::Refine,
            );
        }
        state
    }

    pub(super) fn rollout_future_patch_latents(
        &self,
        seed_patch_tokens: Tensor<B, 3>,
        future_hidden_all: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 3>) {
        let [batch, future_len, embed_dim] = future_hidden_all.shape().dims::<3>();
        let [_, tokens, _] = seed_patch_tokens.shape().dims::<3>();
        if future_len == 0 || tokens == 0 {
            let device = future_hidden_all.device();
            return (
                Tensor::<B, 4>::zeros(
                    [batch, future_len.max(1), tokens.max(1), embed_dim],
                    &device,
                )
                .slice_dim(1, 0..future_len)
                .slice_dim(2, 0..tokens),
                Tensor::<B, 3>::zeros([batch, future_len.max(1), embed_dim], &device)
                    .slice_dim(1, 0..future_len),
            );
        }

        let conditioning = self.patch_conditioner.forward(future_hidden_all);
        let mut current = seed_patch_tokens;
        let mut future_patch_tokens = Vec::with_capacity(future_len);
        let mut future_cls_embed = Vec::with_capacity(future_len);

        for step_idx in 0..future_len {
            let cond = conditioning
                .clone()
                .slice_dim(1, step_idx..step_idx + 1)
                .reshape([batch, embed_dim]);
            let output = self.rollout_conditioned_patch_step(
                current.clone(),
                Some(cond),
                steps,
                backprop_steps,
                StructuredStepMode::Predict,
            );
            current = output.patch_tokens.clone();
            future_patch_tokens.push(current.clone().unsqueeze_dim::<4>(1));
            future_cls_embed.push(output.cls_token.unsqueeze_dim::<3>(1));
            if self.should_detach_predict_rollout(step_idx, future_len) {
                current = current.detach();
            }
        }

        (
            Tensor::cat(future_patch_tokens, 1),
            Tensor::cat(future_cls_embed, 1),
        )
    }

    pub(super) fn rollout_future_patch_latents_pyramid(
        &self,
        mut state: StructuredTopologyState<B>,
        future_len: usize,
        steps: usize,
        backprop_steps: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 3>, StructuredTopologyState<B>) {
        let patch_seed = self.frame_model.pyramid_patch_tokens(&state);
        let [batch, patch_count, embed_dim] = patch_seed.shape().dims::<3>();
        if future_len == 0 || patch_count == 0 || embed_dim == 0 {
            let device = patch_seed.device();
            let future_patch_tokens = Tensor::<B, 4>::zeros(
                [
                    batch,
                    future_len.max(1),
                    patch_count.max(1),
                    embed_dim.max(1),
                ],
                &device,
            )
            .slice_dim(1, 0..future_len)
            .slice_dim(2, 0..patch_count)
            .slice_dim(3, 0..embed_dim);
            let future_cls_embed =
                Tensor::<B, 3>::zeros([batch, future_len.max(1), embed_dim.max(1)], &device)
                    .slice_dim(1, 0..future_len)
                    .slice_dim(2, 0..embed_dim);
            return (future_patch_tokens, future_cls_embed, state);
        }

        let mut future_patch_tokens = Vec::with_capacity(future_len);
        let mut future_cls_embed = Vec::with_capacity(future_len);

        for step_idx in 0..future_len {
            let summary = self.frame_model.pyramid_summary(&state);
            state = self.rollout_pyramid_step(
                state,
                Some(summary),
                steps,
                backprop_steps,
                StructuredStepMode::Predict,
            );
            future_patch_tokens.push(
                self.frame_model
                    .pyramid_patch_tokens(&state)
                    .unsqueeze_dim::<4>(1),
            );
            future_cls_embed.push(
                self.frame_model
                    .pyramid_summary(&state)
                    .unsqueeze_dim::<3>(1),
            );
            if self.should_detach_predict_rollout(step_idx, future_len) {
                state = state.detach();
            }
        }

        (
            Tensor::cat(future_patch_tokens, 1),
            Tensor::cat(future_cls_embed, 1),
            state,
        )
    }

    pub(super) fn rollout_conditioned_patch_step(
        &self,
        patch_tokens: Tensor<B, 3>,
        global_condition: Option<Tensor<B, 2>>,
        steps: usize,
        backprop_steps: usize,
        mode: StructuredStepMode,
    ) -> VisionVideoRolloutOutput<B> {
        let [batch, patch_count, embed_dim] = patch_tokens.shape().dims::<3>();
        let conditioned = if let Some(global_condition) = global_condition {
            let cond = self
                .patch_conditioner
                .forward(global_condition)
                .reshape([batch, 1, embed_dim])
                .repeat_dim(1, patch_count);
            patch_tokens + cond
        } else {
            patch_tokens
        };
        let conditioned = self.apply_step_mode(conditioned, mode);
        self.frame_model
            .forward_tokens_embed_steps_rollout_unbounded(conditioned, steps, backprop_steps)
    }

    pub(super) fn filter_context_patch_latents_pyramid(
        &self,
        observation_patch_tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionVideoContextForward<B> {
        let [batch, context_len, patch_count, embed_dim] =
            observation_patch_tokens.shape().dims::<4>();
        if context_len == 0 || patch_count == 0 {
            let device = observation_patch_tokens.device();
            return VisionVideoContextForward {
                posterior_patch_tokens: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        context_len.max(1),
                        patch_count.max(1),
                        embed_dim.max(1),
                    ],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..patch_count)
                .slice_dim(3, 0..embed_dim),
                posterior_cls_embed: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                patch_tokens: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        context_len.max(1),
                        patch_count.max(1),
                        embed_dim.max(1),
                    ],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..patch_count)
                .slice_dim(3, 0..embed_dim),
                cls_embed: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                hidden: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                temporal_state: ModelState::new(self.config.temporal.n_layer.max(1)),
                structured_state: None,
            };
        }

        let mut previous_state: Option<StructuredTopologyState<B>> = None;
        let mut context_posterior_patch_tokens = Vec::with_capacity(context_len);
        let mut context_posterior_cls_embed = Vec::with_capacity(context_len);
        let mut refined_patch_tokens = Vec::with_capacity(context_len);
        let mut refined_cls_embed = Vec::with_capacity(context_len);
        let mut temporal_hidden_steps = Vec::with_capacity(context_len);

        for step_idx in 0..context_len {
            let observation = observation_patch_tokens
                .clone()
                .slice_dim(1, step_idx..step_idx + 1)
                .reshape([batch, patch_count, embed_dim]);

            let posterior_state = if let Some(previous_state) = previous_state.clone() {
                let prior_summary = self.frame_model.pyramid_summary(&previous_state);
                let prior_state = self.rollout_pyramid_step(
                    previous_state,
                    Some(prior_summary),
                    steps,
                    backprop_steps,
                    StructuredStepMode::Predict,
                );
                let prior_patch = self.frame_model.pyramid_patch_tokens(&prior_state);
                let merged = self.observation_merger.forward(prior_patch, observation);
                let posterior_state = self.frame_model.pyramid_state_with_patch_tokens(
                    prior_state,
                    self.apply_step_mode(merged, StructuredStepMode::Observe),
                );
                self.frame_model
                    .forward_pyramid_state_rollout_mode_unbounded(
                        posterior_state,
                        steps,
                        backprop_steps,
                        StructuredStepMode::Observe,
                    )
            } else {
                let initial_state = self.frame_model.pyramid_state_from_patch_tokens(
                    self.apply_step_mode(observation, StructuredStepMode::Observe),
                );
                self.frame_model
                    .forward_pyramid_state_rollout_mode_unbounded(
                        initial_state,
                        steps,
                        backprop_steps,
                        StructuredStepMode::Observe,
                    )
            };

            let posterior_patch = self.frame_model.pyramid_patch_tokens(&posterior_state);
            let posterior_cls = self.frame_model.pyramid_summary(&posterior_state);
            let refined_state =
                self.maybe_refine_pyramid_state(posterior_state, steps, backprop_steps);
            let refined_patch = self.frame_model.pyramid_patch_tokens(&refined_state);
            let refined_cls = self.frame_model.pyramid_summary(&refined_state);

            previous_state = Some(refined_state);
            context_posterior_patch_tokens.push(posterior_patch.unsqueeze_dim::<4>(1));
            context_posterior_cls_embed.push(posterior_cls.clone().unsqueeze_dim::<3>(1));
            refined_patch_tokens.push(refined_patch.unsqueeze_dim::<4>(1));
            refined_cls_embed.push(refined_cls.clone().unsqueeze_dim::<3>(1));
            temporal_hidden_steps.push(refined_cls.unsqueeze_dim::<3>(1));
        }

        VisionVideoContextForward {
            posterior_patch_tokens: Tensor::cat(context_posterior_patch_tokens, 1),
            posterior_cls_embed: Tensor::cat(context_posterior_cls_embed, 1),
            patch_tokens: Tensor::cat(refined_patch_tokens, 1),
            cls_embed: Tensor::cat(refined_cls_embed, 1),
            hidden: Tensor::cat(temporal_hidden_steps, 1),
            temporal_state: ModelState::new(self.config.temporal.n_layer.max(1)),
            structured_state: previous_state,
        }
    }

    pub(super) fn filter_context_patch_latents(
        &self,
        observation_patch_tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionVideoContextForward<B> {
        let [batch, context_len, patch_count, embed_dim] =
            observation_patch_tokens.shape().dims::<4>();
        if context_len == 0 || patch_count == 0 {
            let device = observation_patch_tokens.device();
            return VisionVideoContextForward {
                posterior_patch_tokens: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        context_len.max(1),
                        patch_count.max(1),
                        embed_dim.max(1),
                    ],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..patch_count)
                .slice_dim(3, 0..embed_dim),
                posterior_cls_embed: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                patch_tokens: Tensor::<B, 4>::zeros(
                    [
                        batch,
                        context_len.max(1),
                        patch_count.max(1),
                        embed_dim.max(1),
                    ],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..patch_count)
                .slice_dim(3, 0..embed_dim),
                cls_embed: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                hidden: Tensor::<B, 3>::zeros(
                    [batch, context_len.max(1), embed_dim.max(1)],
                    &device,
                )
                .slice_dim(1, 0..context_len)
                .slice_dim(2, 0..embed_dim),
                temporal_state: ModelState::new(self.config.temporal.n_layer.max(1)),
                structured_state: None,
            };
        }

        let mut temporal_state = self.temporal_model_ref().init_state();
        let mut previous_patch_tokens: Option<Tensor<B, 3>> = None;
        let mut previous_temporal_hidden: Option<Tensor<B, 2>> = None;
        let mut context_posterior_patch_tokens = Vec::with_capacity(context_len);
        let mut context_posterior_cls_embed = Vec::with_capacity(context_len);
        let mut posterior_patch_tokens = Vec::with_capacity(context_len);
        let mut posterior_cls_embed = Vec::with_capacity(context_len);
        let mut temporal_hidden_steps = Vec::with_capacity(context_len);

        for step_idx in 0..context_len {
            let observation = observation_patch_tokens
                .clone()
                .slice_dim(1, step_idx..step_idx + 1)
                .reshape([batch, patch_count, embed_dim]);

            let posterior_output =
                if let Some(previous_patch_tokens) = previous_patch_tokens.clone() {
                    let prior_output = self.rollout_conditioned_patch_step(
                        previous_patch_tokens,
                        previous_temporal_hidden.clone(),
                        steps,
                        backprop_steps,
                        StructuredStepMode::Predict,
                    );
                    let merged = self
                        .observation_merger
                        .forward(prior_output.patch_tokens, observation);
                    self.frame_model
                        .forward_tokens_embed_steps_rollout_unbounded(
                            self.apply_step_mode(merged, StructuredStepMode::Observe),
                            steps,
                            backprop_steps,
                        )
                } else {
                    self.frame_model
                        .forward_tokens_embed_steps_rollout_unbounded(
                            self.apply_step_mode(observation, StructuredStepMode::Observe),
                            steps,
                            backprop_steps,
                        )
                };

            let posterior_patch = posterior_output.patch_tokens.clone();
            let posterior_cls = posterior_output.cls_token.clone();
            let refined_output =
                self.maybe_refine_patch_state(posterior_output, steps, backprop_steps);
            let refined_patch = refined_output.patch_tokens;
            let refined_cls = refined_output.cls_token;
            let (hidden_t, _) = self
                .temporal_model_ref()
                .forward_with_hidden_and_state_embedded(
                    refined_cls.clone().unsqueeze_dim::<3>(1),
                    &mut temporal_state,
                );
            let hidden_t = hidden_t.reshape([batch, embed_dim]);

            previous_patch_tokens = Some(refined_patch.clone());
            previous_temporal_hidden = Some(hidden_t.clone());
            context_posterior_patch_tokens.push(posterior_patch.unsqueeze_dim::<4>(1));
            context_posterior_cls_embed.push(posterior_cls.unsqueeze_dim::<3>(1));
            posterior_patch_tokens.push(refined_patch.unsqueeze_dim::<4>(1));
            posterior_cls_embed.push(refined_cls.unsqueeze_dim::<3>(1));
            temporal_hidden_steps.push(hidden_t.unsqueeze_dim::<3>(1));
        }

        VisionVideoContextForward {
            posterior_patch_tokens: Tensor::cat(context_posterior_patch_tokens, 1),
            posterior_cls_embed: Tensor::cat(context_posterior_cls_embed, 1),
            patch_tokens: Tensor::cat(posterior_patch_tokens, 1),
            cls_embed: Tensor::cat(posterior_cls_embed, 1),
            hidden: Tensor::cat(temporal_hidden_steps, 1),
            temporal_state,
            structured_state: None,
        }
    }

    pub(super) fn debug_reconstruction_outputs(
        &self,
        forward: &VisionVideoForward<B>,
        steps: usize,
        capture_artifacts: bool,
        emit_clip_for_metrics: bool,
        sample_batch_limit: Option<usize>,
    ) -> VisionVideoDebugRecon<B> {
        let [batch, _, channels, height, width] = forward.clip_frames.shape().dims::<5>();
        let sample_batch = sample_batch_limit
            .map(|limit| limit.max(1).min(batch))
            .unwrap_or(batch);
        let clip_frames = if sample_batch == batch {
            forward.clip_frames.clone()
        } else {
            forward.clip_frames.clone().slice_dim(0, 0..sample_batch)
        };
        let frame_patch_tokens = if sample_batch == batch {
            forward.frame_patch_tokens.clone()
        } else {
            forward
                .frame_patch_tokens
                .clone()
                .slice_dim(0, 0..sample_batch)
        };
        let future_patch_tokens = if sample_batch == batch {
            forward.future_patch_tokens.clone()
        } else {
            forward
                .future_patch_tokens
                .clone()
                .slice_dim(0, 0..sample_batch)
        };
        let context_len = forward.context_len;
        let target_len = forward.target_len;
        let future_len_all = forward.future_len_all;

        let context_patch_tokens = frame_patch_tokens
            .clone()
            .slice_dim(1, 0..context_len)
            .detach();
        let context_target = clip_frames
            .clone()
            .slice_dim(1, 0..context_len)
            .detach()
            .reshape([sample_batch * context_len, channels, height, width]);
        let context_recon_patches =
            self.reconstruct_patch_values_from_patch_tokens(context_patch_tokens, channels);
        let context_target_patches = patchify(context_target, self.frame_model.patch_size().max(1));
        let [_, _, patch_tokens, patch_dim] = context_recon_patches.shape().dims::<4>();

        let future_patch_all = future_patch_tokens.clone().detach();
        let future_target_full = clip_frames
            .clone()
            .slice_dim(1, context_len..context_len + future_len_all)
            .detach()
            .reshape([sample_batch * future_len_all, channels, height, width]);
        let future_recon_full_patches =
            self.reconstruct_patch_values_from_patch_tokens(future_patch_all, channels);
        let future_target_full_patches =
            patchify(future_target_full, self.frame_model.patch_size().max(1)).reshape([
                sample_batch,
                future_len_all,
                patch_tokens,
                patch_dim,
            ]);
        let future_recon_full_patches = future_recon_full_patches.reshape([
            sample_batch,
            future_len_all,
            patch_tokens,
            patch_dim,
        ]);
        let future_sq_error = (future_recon_full_patches.clone()
            - future_target_full_patches.clone())
        .powf_scalar(2.0);
        let short_mask =
            Tensor::<B, 1, Int>::arange(0..future_len_all as i64, &clip_frames.device())
                .float()
                .lower_equal_elem(target_len.saturating_sub(1) as f32)
                .float()
                .reshape([1, future_len_all, 1, 1])
                .repeat_dim(0, sample_batch);
        let context_sq_error = (context_recon_patches.clone().reshape([
            sample_batch * context_len,
            patch_tokens,
            patch_dim,
        ]) - context_target_patches.clone())
        .powf_scalar(2.0);
        let context_sum = context_sq_error.clone().sum();
        let context_denom = (sample_batch * context_len * patch_tokens * patch_dim).max(1) as f32;
        let future_short_sum = future_sq_error.clone().mul(short_mask).sum();
        let future_short_denom =
            (sample_batch * target_len * patch_tokens * patch_dim).max(1) as f32;
        let recon_loss = (context_sum.clone() + future_short_sum.clone())
            .div_scalar(context_denom + future_short_denom);
        let short_mse = future_short_sum.div_scalar(future_short_denom);
        let psnr_short = recon_psnr(short_mse);

        let full_mse = future_sq_error
            .clone()
            .reshape([sample_batch * future_len_all, patch_tokens, patch_dim])
            .mean();
        let psnr_full = recon_psnr(full_mse);

        let artifact_capture_enabled = capture_artifacts
            && self.config.artifact_every > 0
            && self.config.artifact_max_images > 0;
        let emit_clip = emit_clip_for_metrics || artifact_capture_enabled;
        let (clip, pca_rgb_steps) = if emit_clip {
            let metrics_clip_images = self.config.artifact_max_images.max(1);
            let image_count = if artifact_capture_enabled {
                self.config.artifact_max_images.min(sample_batch)
            } else {
                metrics_clip_images.min(sample_batch)
            };
            if image_count == 0 {
                (None, None)
            } else {
                let context_clip = self
                    .reconstruct_frames_from_patch_values(
                        context_recon_patches.clone(),
                        height,
                        width,
                        channels,
                    )
                    .slice_dim(0, 0..(image_count * context_len))
                    .reshape([image_count, context_len, channels, height, width]);
                let future_clip = self
                    .reconstruct_frames_from_patch_values(
                        future_recon_full_patches.clone(),
                        height,
                        width,
                        channels,
                    )
                    .slice_dim(0, 0..(image_count * future_len_all))
                    .reshape([image_count, future_len_all, channels, height, width]);
                let recon_clip = Tensor::cat(vec![context_clip, future_clip], 1);
                let pca_rgb_steps = if artifact_capture_enabled {
                    let (_, recon_patch_tokens, _) = encode_clip_frames_with_model(
                        &self.frame_model,
                        recon_clip.clone(),
                        steps,
                        steps,
                        self.embed_dim,
                        self.projection_dim,
                    );
                    let (_, _, _patch_norms_steps, pca_rgb_steps) =
                        collect_video_feature_maps(recon_patch_tokens, image_count, false);
                    pca_rgb_steps
                } else {
                    None
                };
                (Some(recon_clip), pca_rgb_steps)
            }
        } else {
            (None, None)
        };

        VisionVideoDebugRecon {
            loss: recon_loss,
            psnr_short,
            psnr_full,
            clip,
            pca_rgb_steps,
        }
    }

    pub(super) fn debug_reconstruction_outputs_from_patch_tokens(
        &self,
        input: VideoDebugReconInput<B>,
    ) -> VisionVideoDebugRecon<B> {
        let VideoDebugReconInput {
            clip_frames,
            frame_patch_tokens,
            future_patch_tokens,
            context_len,
            target_len,
            future_len_all,
            steps,
            capture_artifacts,
        } = input;
        let forward = VisionVideoForward {
            predicted_proj: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            ),
            target_proj: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            ),
            predicted_proj_all: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..future_len_all),
            target_proj_all: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..future_len_all),
            observation_proj: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    context_len.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..context_len),
            observation_target_proj: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    context_len.max(1),
                    self.projection_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..context_len),
            context_posterior_patch_tokens: Tensor::<B, 4>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    context_len.max(1),
                    frame_patch_tokens.shape().dims::<4>()[2].max(1),
                    self.embed_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..context_len)
            .slice_dim(2, 0..frame_patch_tokens.shape().dims::<4>()[2]),
            context_posterior_cls_embed: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    context_len.max(1),
                    self.embed_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..context_len),
            context_summary: Tensor::<B, 2>::zeros(
                [clip_frames.shape().dims::<5>()[0], self.embed_dim.max(1)],
                &clip_frames.device(),
            ),
            cls_embed: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    (context_len + future_len_all).max(1),
                    self.embed_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..(context_len + future_len_all)),
            frame_patch_tokens,
            future_hidden_all: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.embed_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..future_len_all),
            future_cls_embed: Tensor::<B, 3>::zeros(
                [
                    clip_frames.shape().dims::<5>()[0],
                    future_len_all.max(1),
                    self.embed_dim.max(1),
                ],
                &clip_frames.device(),
            )
            .slice_dim(1, 0..future_len_all),
            future_patch_tokens,
            context_structured_state: None,
            probe_logits: Tensor::<B, 2>::zeros(
                [clip_frames.shape().dims::<5>()[0], 1],
                &clip_frames.device(),
            ),
            probe_labels: Tensor::<B, 1, Int>::zeros(
                [clip_frames.shape().dims::<5>()[0]],
                &clip_frames.device(),
            ),
            clip_frames,
            context_len,
            target_len,
            future_len_all,
        };
        self.debug_reconstruction_outputs(&forward, steps, capture_artifacts, false, None)
    }

    pub(super) fn forward_video(
        &self,
        batch: VideoClipBatch<B>,
        steps: usize,
        backprop_steps: usize,
        future_len_requested: usize,
    ) -> VisionVideoForward<B> {
        let clip_frames = batch.clip_frames;
        let labels = batch.labels;
        let [batch_size, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
        let context_len = batch.context_len.min(clip_len.saturating_sub(1)).max(1);
        let target_end = (context_len + batch.target_len).min(clip_len);
        let target_len = target_end.saturating_sub(context_len).max(1);
        let available_future_len = clip_len.saturating_sub(context_len).max(target_len);
        let future_len_all = available_future_len.min(future_len_requested.max(target_len));
        let context_frames = clip_frames.clone().slice_dim(1, 0..context_len);
        let context_observation_patch_tokens =
            embed_clip_frames_raw_with_model(&self.frame_model, context_frames.clone());
        let context_forward = if self.uses_pyramid_backbone() {
            self.filter_context_patch_latents_pyramid(
                context_observation_patch_tokens,
                steps,
                backprop_steps,
            )
        } else {
            self.filter_context_patch_latents(
                context_observation_patch_tokens,
                steps,
                backprop_steps,
            )
        };
        let context_target_proj = if let Some(teacher) = &self.teacher_frame_model {
            project_clip_frames_with_model(
                teacher,
                context_frames.clone(),
                steps,
                self.embed_dim,
                self.projection_dim,
            )
            .detach()
        } else {
            project_clip_frames_with_model(
                &self.frame_model,
                context_frames.clone(),
                steps,
                self.embed_dim,
                self.projection_dim,
            )
            .detach()
        };
        let future_target_frames = clip_frames
            .clone()
            .slice_dim(1, context_len..context_len + future_len_all)
            .reshape([batch_size, future_len_all, channels, height, width]);
        let target_proj_all = if let Some(teacher) = &self.teacher_frame_model {
            project_clip_frames_with_model(
                teacher,
                future_target_frames,
                steps,
                self.embed_dim,
                self.projection_dim,
            )
            .detach()
        } else {
            project_clip_frames_with_model(
                &self.frame_model,
                future_target_frames,
                steps,
                self.embed_dim,
                self.projection_dim,
            )
            .detach()
        };
        let target_proj = target_proj_all.clone();
        let observation_proj = self
            .frame_model
            .project_tokens(context_forward.posterior_cls_embed.clone());

        let (future_hidden_all, future_patch_tokens, future_cls_embed, context_structured_state) =
            if let Some(structured_state) = context_forward.structured_state.clone() {
                let (future_patch_tokens, future_cls_embed, _) = self
                    .rollout_future_patch_latents_pyramid(
                        structured_state.clone(),
                        future_len_all,
                        steps,
                        backprop_steps,
                    );
                (
                    future_cls_embed.clone(),
                    future_patch_tokens,
                    future_cls_embed,
                    Some(structured_state),
                )
            } else {
                let mut temporal_state = context_forward.temporal_state.clone();
                let future_queries =
                    repeat_last_future_query(self.future_queries_ref(), future_len_all)
                        .reshape([1, future_len_all, self.embed_dim])
                        .repeat_dim(0, batch_size);
                let (future_hidden_all, _) = self
                    .temporal_model_ref()
                    .forward_with_hidden_and_state_embedded(future_queries, &mut temporal_state);
                let seed_patch_tokens = context_forward
                    .patch_tokens
                    .clone()
                    .slice_dim(1, (context_len - 1)..context_len)
                    .reshape([
                        batch_size,
                        context_forward.patch_tokens.shape().dims::<4>()[2],
                        self.embed_dim,
                    ]);
                let (future_patch_tokens, future_cls_embed) = self.rollout_future_patch_latents(
                    seed_patch_tokens,
                    future_hidden_all.clone(),
                    steps,
                    backprop_steps,
                );
                (
                    future_hidden_all,
                    future_patch_tokens,
                    future_cls_embed,
                    None,
                )
            };
        let predicted_proj_all = self.predictor.forward(future_cls_embed.clone());
        let predicted_proj = predicted_proj_all.clone();
        let context_summary = context_forward
            .hidden
            .clone()
            .slice_dim(1, (context_len - 1)..context_len)
            .reshape([batch_size, self.embed_dim]);
        let probe_logits = self.probe.forward(context_summary.clone().detach());
        let cls_embed = Tensor::cat(
            vec![context_forward.cls_embed.clone(), future_cls_embed.clone()],
            1,
        );
        let frame_patch_tokens = Tensor::cat(
            vec![
                context_forward.patch_tokens.clone(),
                future_patch_tokens.clone(),
            ],
            1,
        );

        VisionVideoForward {
            predicted_proj,
            target_proj,
            predicted_proj_all,
            target_proj_all,
            observation_proj,
            observation_target_proj: context_target_proj,
            context_posterior_patch_tokens: context_forward.posterior_patch_tokens,
            context_posterior_cls_embed: context_forward.posterior_cls_embed,
            context_summary,
            cls_embed,
            frame_patch_tokens,
            future_hidden_all,
            future_cls_embed,
            future_patch_tokens,
            context_structured_state,
            probe_logits,
            probe_labels: labels,
            clip_frames,
            context_len,
            target_len,
            future_len_all,
        }
    }

    pub(super) fn mode_separation_ratio(
        &self,
        forward: &VisionVideoForward<B>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 1> {
        if let Some(state) = forward.context_structured_state.clone() {
            let seed = self.frame_model.pyramid_patch_tokens(&state);
            let refine_state = self.rollout_pyramid_step(
                state.clone(),
                None,
                steps,
                backprop_steps,
                StructuredStepMode::Refine,
            );
            let refine = self.frame_model.pyramid_patch_tokens(&refine_state);
            let predict_state = self.rollout_pyramid_step(
                state,
                Some(forward.context_summary.clone()),
                steps,
                backprop_steps,
                StructuredStepMode::Predict,
            );
            let predict = self.frame_model.pyramid_patch_tokens(&predict_state);
            let refine_motion = (refine - seed.clone()).abs().mean();
            let predict_motion = (predict - seed).abs().mean();
            return predict_motion.div(refine_motion.add_scalar(LEJEPA_EPS));
        }

        let [batch, context_len, patch_count, embed_dim] =
            forward.frame_patch_tokens.shape().dims::<4>();
        if batch == 0 || context_len == 0 || patch_count == 0 || embed_dim == 0 {
            return Tensor::<B, 1>::zeros([1], &forward.frame_patch_tokens.device());
        }
        let seed = forward
            .frame_patch_tokens
            .clone()
            .slice_dim(1, (forward.context_len - 1)..forward.context_len)
            .reshape([batch, patch_count, embed_dim]);
        let refine = self.rollout_conditioned_patch_step(
            seed.clone(),
            None,
            steps,
            backprop_steps,
            StructuredStepMode::Refine,
        );
        let predict_hidden = if forward.future_hidden_all.shape().dims::<3>()[1] > 0 {
            Some(
                forward
                    .future_hidden_all
                    .clone()
                    .slice_dim(1, 0..1)
                    .reshape([batch, embed_dim]),
            )
        } else {
            None
        };
        let predict = self.rollout_conditioned_patch_step(
            seed.clone(),
            predict_hidden,
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        );
        let refine_motion = (refine.patch_tokens - seed.clone()).abs().mean();
        let predict_motion = (predict.patch_tokens - seed).abs().mean();
        predict_motion.div(refine_motion.add_scalar(LEJEPA_EPS))
    }

    pub(super) fn mode_separation_ratio_structured(
        &self,
        state: StructuredTopologyState<B>,
        context_summary: Tensor<B, 2>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 1> {
        let seed = self.frame_model.pyramid_patch_tokens(&state);
        let refine_state = self.rollout_pyramid_step(
            state.clone(),
            None,
            steps,
            backprop_steps,
            StructuredStepMode::Refine,
        );
        let refine = self.frame_model.pyramid_patch_tokens(&refine_state);
        let predict_state = self.rollout_pyramid_step(
            state,
            Some(context_summary),
            steps,
            backprop_steps,
            StructuredStepMode::Predict,
        );
        let predict = self.frame_model.pyramid_patch_tokens(&predict_state);
        let refine_motion = (refine - seed.clone()).abs().mean();
        let predict_motion = (predict - seed).abs().mean();
        predict_motion.div(refine_motion.add_scalar(LEJEPA_EPS))
    }

    pub(super) fn effective_predict_backprop_frames(&self, future_len: usize) -> usize {
        let future_len = future_len.max(1);
        let configured = self.config.temporal.predict_backprop_frames;
        if configured == 0 {
            future_len
        } else {
            configured.min(future_len).max(1)
        }
    }

    pub(super) fn should_detach_predict_rollout(&self, step_idx: usize, future_len: usize) -> bool {
        let backprop_frames = self.effective_predict_backprop_frames(future_len);
        let detach_until = future_len.saturating_sub(backprop_frames);
        step_idx < detach_until
    }

    pub(super) fn frame_center_of_mass(&self, frames: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, frame_count, _channels, height, width] = frames.shape().dims::<5>();
        let device = frames.device();
        let weights =
            frames
                .clamp(0.0, 1.0)
                .mean_dim(2)
                .reshape([batch, frame_count, height, width]);
        let mass = weights
            .clone()
            .sum_dim(2)
            .sum_dim(3)
            .add_scalar(LEJEPA_EPS)
            .reshape([batch, frame_count, 1]);

        let x_scale = if width > 1 {
            1.0 / ((width - 1) as f64)
        } else {
            0.0
        };
        let y_scale = if height > 1 {
            1.0 / ((height - 1) as f64)
        } else {
            0.0
        };
        let x_coords = Tensor::<B, 1, Int>::arange(0..width as i64, &device)
            .float()
            .mul_scalar(x_scale)
            .reshape([1, 1, 1, width]);
        let y_coords = Tensor::<B, 1, Int>::arange(0..height as i64, &device)
            .float()
            .mul_scalar(y_scale)
            .reshape([1, 1, height, 1]);

        let com_x = weights
            .clone()
            .mul(x_coords)
            .sum_dim(2)
            .sum_dim(3)
            .reshape([batch, frame_count, 1])
            .div(mass.clone());
        let com_y = weights
            .mul(y_coords)
            .sum_dim(2)
            .sum_dim(3)
            .reshape([batch, frame_count, 1])
            .div(mass);

        Tensor::cat(vec![com_x, com_y], 2)
    }

    pub(super) fn rollout_kinematics_metrics(
        &self,
        forward: &VisionVideoForward<B>,
        predicted_clip: Option<Tensor<B, 5>>,
        future_horizon: usize,
    ) -> (Option<Tensor<B, 1>>, Option<Tensor<B, 1>>) {
        let Some(predicted_clip) = predicted_clip else {
            return (None, None);
        };
        let future_horizon = future_horizon.min(forward.future_len_all);
        if future_horizon == 0 {
            return (None, None);
        }

        let [batch, _clip_len, channels, height, width] = predicted_clip.shape().dims::<5>();
        let end = future_horizon
            .min(
                *VISION_ROLLOUT_HORIZON_CAPS
                    .last()
                    .unwrap_or(&future_horizon),
            )
            .max(1);
        let predicted_future = predicted_clip
            .slice_dim(1, forward.context_len..forward.context_len + end)
            .reshape([batch, end, channels, height, width]);
        let target_future = forward
            .clip_frames
            .clone()
            .slice_dim(0, 0..batch)
            .slice_dim(1, forward.context_len..forward.context_len + end)
            .reshape([batch, end, channels, height, width]);

        let predicted_com = self.frame_center_of_mass(predicted_future);
        let target_com = self.frame_center_of_mass(target_future);
        let com_error = (predicted_com.clone() - target_com.clone())
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .mean();
        let velocity_error = if end > 1 {
            let predicted_velocity = predicted_com.clone().slice_dim(1, 1..end)
                - predicted_com.slice_dim(1, 0..(end - 1));
            let target_velocity =
                target_com.clone().slice_dim(1, 1..end) - target_com.slice_dim(1, 0..(end - 1));
            (predicted_velocity - target_velocity)
                .powf_scalar(2.0)
                .sum_dim(2)
                .sqrt()
                .mean()
        } else {
            Tensor::<B, 1>::zeros([1], &forward.clip_frames.device())
        };

        (Some(com_error), Some(velocity_error))
    }

    pub(super) fn rollout_horizon_metrics(
        &self,
        forward: &VisionVideoForward<B>,
    ) -> RolloutHorizonMetricArrays<B> {
        self.rollout_horizon_metrics_from_tensors(
            forward.predicted_proj_all.clone(),
            forward.target_proj_all.clone(),
            forward.frame_patch_tokens.clone(),
            forward.future_patch_tokens.clone(),
            forward.context_len,
        )
    }

    pub(super) fn rollout_horizon_metrics_from_tensors(
        &self,
        predicted_proj_all: Tensor<B, 3>,
        target_proj_all: Tensor<B, 3>,
        frame_patch_tokens: Tensor<B, 4>,
        future_patch_tokens: Tensor<B, 4>,
        context_len: usize,
    ) -> RolloutHorizonMetricArrays<B> {
        let device = future_patch_tokens.device();
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let mut rollout_inv_to_horizon = core::array::from_fn(|_| zero.clone());
        let mut rollout_state_norm_ratio_to_horizon = core::array::from_fn(|_| zero.clone());
        let mut rollout_state_motion_to_horizon = core::array::from_fn(|_| zero.clone());

        let [batch, future_len_all, patch_count, embed_dim] =
            future_patch_tokens.shape().dims::<4>();
        if batch == 0 || future_len_all == 0 || patch_count == 0 || embed_dim == 0 {
            return (
                rollout_inv_to_horizon,
                rollout_state_norm_ratio_to_horizon,
                rollout_state_motion_to_horizon,
            );
        }

        let seed_patch_tokens = frame_patch_tokens
            .clone()
            .slice_dim(1, context_len.saturating_sub(1)..context_len)
            .reshape([batch, patch_count, embed_dim]);
        let seed_norm = seed_patch_tokens
            .powf_scalar(2.0)
            .mean()
            .sqrt()
            .add_scalar(LEJEPA_EPS);

        for (index, horizon) in VISION_ROLLOUT_HORIZON_CAPS.into_iter().enumerate() {
            let end = future_len_all.min(horizon).max(1);
            rollout_inv_to_horizon[index] = (predicted_proj_all.clone().slice_dim(1, 0..end)
                - target_proj_all.clone().slice_dim(1, 0..end))
            .powf_scalar(2.0)
            .mean();

            let future_prefix = future_patch_tokens.clone().slice_dim(1, 0..end);
            let future_norm = future_prefix.clone().powf_scalar(2.0).mean().sqrt();
            rollout_state_norm_ratio_to_horizon[index] = future_norm / seed_norm.clone();

            rollout_state_motion_to_horizon[index] = if end > 1 {
                (future_prefix.clone().slice_dim(1, 1..end)
                    - future_prefix.slice_dim(1, 0..(end - 1)))
                .abs()
                .mean()
            } else {
                zero.clone()
            };
        }

        (
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
        )
    }
}
