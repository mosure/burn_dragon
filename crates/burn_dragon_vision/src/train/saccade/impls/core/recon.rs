use super::*;

impl<B: BackendTrait> VisionSaccadeModel<B> {
    fn run_recon_rollout(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
        scratch: &mut SaccadeStepScratch<B>,
    ) {
        for step_idx in 0..ctx.rollout_steps {
            let pre_rollout = ctx.low_mem_pre_rollout && step_idx < ctx.detach_until;
            let in_backprop = step_idx >= ctx.detach_until;
            if ctx.tbptt_enabled && in_backprop && !state.started_backprop {
                state.started_backprop = true;
                state.reset_policy_accumulators(
                    ctx.gdpo_enabled,
                    ctx.info_reward_enabled,
                    ctx.batch,
                    &ctx.device,
                );
            }
            let anchor_traj = ctx.low_mem_pre_rollout
                && ctx.detach_until > 0
                && step_idx + 1 == ctx.detach_until + 1;
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state.state_levels.clone(),
                VisionPyramidMode::Laplacian => {
                    self.compose_pyramid(&state.state_levels, &ctx.grids)
                }
            };
            let collect_info = ctx.info_reward_enabled && (step_idx % ctx.info_stride == 0);
            scratch.reset_for_step(
                &state.state_levels,
                &ctx.device,
                ctx.num_eyes,
                ctx.capture_traj,
                ctx.capture_artifacts,
                collect_info,
            );
            let mut eye_trajs = Vec::with_capacity(ctx.num_eyes);
            let mut eye_weights = Vec::with_capacity(ctx.num_eyes);
            let mut tokens_in_multi = Vec::with_capacity(ctx.num_eyes);
            let mut tokens_in_null_multi = if collect_info {
                Some(Vec::with_capacity(ctx.num_eyes))
            } else {
                None
            };
            let traj_denom = ctx.traj_len.max(1) as f32;

            for eye_idx in 0..ctx.num_eyes {
                let mut traj = state.trajs[eye_idx].clone();
                if pre_rollout {
                    traj = traj.detach();
                } else if anchor_traj {
                    let anchor = self
                        .trajectory_token
                        .val()
                        .reshape([1, ctx.traj_len, ctx.embed_dim])
                        .repeat_dim(0, ctx.batch);
                    traj = traj + (anchor.clone() - anchor.detach());
                }
                let eye_embed = self
                    .eye_token
                    .val()
                    .slice_dim(0, eye_idx..eye_idx + 1)
                    .reshape([1, 1, ctx.embed_dim])
                    .repeat_dim(0, ctx.batch)
                    .repeat_dim(1, ctx.traj_len);
                let eye_embed = Self::detach_if(eye_embed, pre_rollout);
                let mut traj_with_eye = traj.clone() + eye_embed.clone();
                if let Some(view_embed) = ctx.view_embed.as_ref() {
                    let [_, eyes, _, _] = view_embed.shape().dims::<4>();
                    if eye_idx < eyes {
                        let view_bias = view_embed
                            .clone()
                            .slice_dim(1, eye_idx..eye_idx + 1)
                            .reshape([ctx.batch, 1, ctx.embed_dim])
                            .repeat_dim(1, ctx.traj_len);
                        let view_bias = Self::detach_if(view_bias, pre_rollout);
                        traj_with_eye = traj_with_eye + view_bias;
                    }
                }
                let traj_with_eye = Self::detach_if(traj_with_eye, pre_rollout);
                let traj_summary = traj_with_eye
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1, ctx.embed_dim]);
                let params = self.saccade_head.forward(traj_summary);
                let params = Self::detach_if(params, pre_rollout);
                let (mean_raw, sigma_raw) = self.decode_saccade_params(params);
                let mean_raw = Self::detach_if(mean_raw, pre_rollout);
                let sigma_raw = Self::detach_if(sigma_raw, pre_rollout);
                let (mean_action, sigma_action) = if ctx.gdpo_enabled {
                    let sample = self.sample_policy_action(mean_raw.clone(), sigma_raw.clone());
                    let log_prob_eye = sample.log_prob.sum_dim(1).reshape([ctx.batch, 1]);
                    if let Some(log_prob_sum) = state.log_prob_sum.as_mut() {
                        *log_prob_sum = log_prob_sum.clone() + log_prob_eye.clone();
                    }
                    if let Some(log_prob_sum_old) = state.log_prob_sum_old.as_mut() {
                        *log_prob_sum_old = log_prob_sum_old.clone() + log_prob_eye.detach();
                    }
                    if let Some(clamp_rate_sum) = state.clamp_rate_sum.as_mut() {
                        *clamp_rate_sum = clamp_rate_sum.clone() + sample.clamp_rate.clone();
                    }
                    state.clamp_rate_count += 1;
                    (sample.mean, sample.sigma)
                } else {
                    (mean_raw.clone(), sigma_raw.clone())
                };
                let mean = if ctx.detach_policy_from_recon {
                    mean_action.clone().detach()
                } else {
                    mean_action.clone()
                };
                let sigma = if ctx.detach_policy_from_recon {
                    sigma_action.clone().detach()
                } else {
                    sigma_action.clone()
                };
                let mean_step = mean
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 2]);
                let sigma_step = sigma
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1]);
                let mean_detached = mean_step.clone().detach();
                let sigma_detached = sigma_step.clone().detach();
                if ctx.capture_traj || ctx.capture_artifacts {
                    scratch
                        .step_capture
                        .traj
                        .push((mean_detached.clone(), sigma_detached.clone()));
                }
                let eye_levels = ctx
                    .mip_levels
                    .get(eye_idx)
                    .unwrap_or_else(|| ctx.mip_levels.first().expect("mip levels"));
                let weights_context =
                    self.mip_gaussian_weights(eye_levels, mean.clone(), sigma.clone());
                let weights_scatter = weights_context.clone();
                let patch_image = self.foveated_patch_image(
                    eye_levels,
                    &ctx.base_grid,
                    mean_step.clone(),
                    sigma_step.clone(),
                    ctx.laplacian_images
                        .as_ref()
                        .and_then(|images| images.get(eye_idx)),
                );
                let patch_tokens = self.model.patch_embed_raw(patch_image.clone()).tokens;
                let patch_tokens = Self::detach_if(patch_tokens, pre_rollout);
                if ctx.capture_artifacts {
                    scratch.step_capture.patches.push(patch_image);
                }
                let input_context = patch_tokens.clone();
                let state_context = {
                    let context = self.mip_weighted_sum(&state_composed, &weights_context);
                    self.project_pyramid_context(context)
                };
                let input_tokens = self.build_input_tokens(
                    input_context,
                    state_context.clone(),
                    mean.clone(),
                    sigma.clone(),
                );
                let input_tokens = Self::detach_if(input_tokens, pre_rollout);
                let input_tokens = input_tokens.repeat_dim(1, ctx.traj_len);
                let mut tokens_in = traj_with_eye.clone() + input_tokens;
                tokens_in = Self::detach_if(tokens_in, pre_rollout);
                tokens_in_multi.push(tokens_in.unsqueeze_dim::<4>(1));

                if let Some(tokens_in_null_multi) = tokens_in_null_multi.as_mut() {
                    let null_patch_tokens = self.null_patch_tokens(&patch_tokens);
                    let input_tokens_null = self.build_input_tokens(
                        null_patch_tokens,
                        state_context.clone(),
                        mean.clone(),
                        sigma.clone(),
                    );
                    let input_tokens_null = Self::detach_if(input_tokens_null, pre_rollout);
                    let input_tokens_null = input_tokens_null.repeat_dim(1, ctx.traj_len);
                    let mut tokens_in_null = traj_with_eye.clone() + input_tokens_null;
                    tokens_in_null = Self::detach_if(tokens_in_null, pre_rollout);
                    tokens_in_null_multi.push(tokens_in_null.unsqueeze_dim::<4>(1));
                }

                eye_trajs.push(traj);
                eye_weights.push(weights_scatter);
            }

            let tokens_in_multi = Tensor::cat(tokens_in_multi, 1);
            let mut out_tokens_multi = self
                .model
                .forward_tokens_embed_steps_rollout_multi(
                    tokens_in_multi,
                    ctx.inner_steps,
                    ctx.inner_steps,
                )
                .patch_tokens;
            out_tokens_multi = Self::detach_if(out_tokens_multi, pre_rollout);

            let out_tokens_null_multi = if let Some(tokens_in_null_multi) = tokens_in_null_multi {
                let tokens_in_null = Tensor::cat(tokens_in_null_multi, 1);
                let mut out_tokens_null = self
                    .model
                    .forward_tokens_embed_steps_rollout_multi(
                        tokens_in_null,
                        ctx.inner_steps,
                        ctx.inner_steps,
                    )
                    .patch_tokens;
                out_tokens_null = Self::detach_if(out_tokens_null, pre_rollout);
                Some(out_tokens_null)
            } else {
                None
            };

            for (eye_idx, weights_for_eye) in eye_weights.iter().enumerate().take(ctx.num_eyes) {
                let traj = eye_trajs
                    .get(eye_idx)
                    .cloned()
                    .unwrap_or_else(|| state.trajs[eye_idx].clone());
                let out_tokens = out_tokens_multi
                    .clone()
                    .slice_dim(1, eye_idx..eye_idx + 1)
                    .reshape([ctx.batch, ctx.traj_len, ctx.embed_dim]);
                let residual = self.residual_proj.forward(out_tokens.clone());
                let residual = Self::detach_if(residual, pre_rollout);
                let residual_pool = residual
                    .clone()
                    .sum_dim(1)
                    .mul_scalar(1.0 / traj_denom)
                    .reshape([ctx.batch, 1, self.pyramid_dim]);
                let next_traj = if ctx.traj_update_alpha >= 1.0 {
                    out_tokens
                } else {
                    let keep = 1.0 - ctx.traj_update_alpha;
                    traj.clone().mul_scalar(keep)
                        + out_tokens.clone().mul_scalar(ctx.traj_update_alpha)
                };
                for (update, weights) in scratch.updates.iter_mut().zip(weights_for_eye.iter()) {
                    let update_eye = self.weighted_sum_tokens(
                        weights.clone().swap_dims(1, 2),
                        residual_pool.clone(),
                    );
                    *update = update.clone() + update_eye;
                }
                if let Some(out_tokens_null_multi) = out_tokens_null_multi.as_ref() {
                    let out_tokens_null = out_tokens_null_multi
                        .clone()
                        .slice_dim(1, eye_idx..eye_idx + 1)
                        .reshape([ctx.batch, ctx.traj_len, ctx.embed_dim]);
                    let residual_null = self.residual_proj.forward(out_tokens_null);
                    let residual_null = Self::detach_if(residual_null, pre_rollout);
                    let residual_pool_null = residual_null
                        .sum_dim(1)
                        .mul_scalar(1.0 / traj_denom)
                        .reshape([ctx.batch, 1, self.pyramid_dim]);
                    for (update, weights) in
                        scratch.updates_null.iter_mut().zip(weights_for_eye.iter())
                    {
                        let update_eye_null = self.weighted_sum_tokens(
                            weights.clone().swap_dims(1, 2),
                            residual_pool_null.clone(),
                        );
                        *update = update.clone() + update_eye_null;
                    }
                }
                scratch.next_trajs.push(next_traj);
            }
            if ctx.gdpo_enabled {
                state.policy_steps += 1;
            }
            let state_real: Vec<Tensor<B, 3>> = state
                .state_levels
                .iter()
                .zip(scratch.updates.iter())
                .map(|(state, update)| state.clone() + update.clone())
                .collect();
            let state_null = if collect_info {
                Some(
                    state
                        .state_levels
                        .iter()
                        .zip(scratch.updates_null.iter())
                        .map(|(state, update)| state.clone() + update.clone())
                        .collect::<Vec<_>>(),
                )
            } else {
                None
            };
            state.state_levels = state_real;
            if collect_info
                && let (Some(hard_reward), Some(state_null)) =
                    (state.hard_reward.as_mut(), state_null)
            {
                let (real_sum, real_mask, _) = self.recon_loss_per_sample_from_state(
                    &state.state_levels,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    false,
                );
                let (null_sum, null_mask, _) = self.recon_loss_per_sample_from_state(
                    &state_null,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    false,
                );
                let real = real_sum / real_mask.add_scalar(LEJEPA_EPS);
                let null = null_sum / null_mask.add_scalar(LEJEPA_EPS);
                *hard_reward = hard_reward.clone() + (null - real);
            }
            if ctx.capture_traj || ctx.capture_artifacts {
                let step_traj = std::mem::take(&mut scratch.step_capture.traj);
                if ctx.capture_artifacts {
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state.state_levels.clone(),
                        VisionPyramidMode::Laplacian => {
                            self.compose_pyramid(&state.state_levels, &ctx.grids)
                        }
                    };
                    let pred_patches = self
                        .recon
                        .forward(self.project_pyramid_level(state_composed[0].clone()));
                    let recon_view = unpatchify(
                        pred_patches,
                        ctx.patch_size,
                        ctx.height,
                        ctx.width,
                        ctx.channels,
                    );
                    let mut combined_frame = ctx
                        .view_images
                        .first()
                        .cloned()
                        .unwrap_or_else(|| ctx.images.clone());
                    let mut appended_patch = false;
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        if let Some(overlay) = saccade_circle_overlay(
                            combined_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            combined_frame = overlay;
                        }
                    }
                    let mut frame_views = Vec::new();
                    let push_view = |views: &mut Vec<Tensor<B, 4>>, view: Tensor<B, 4>| {
                        if !views.is_empty() && SACCADE_VIEW_GAP > 0 {
                            views.push(view_separator_like(&view, SACCADE_VIEW_GAP));
                        }
                        views.push(view);
                    };
                    push_view(&mut frame_views, combined_frame);
                    for (eye_idx, (mean, sigma)) in step_traj.iter().enumerate() {
                        let mut eye_frame = ctx
                            .view_images
                            .get(eye_idx)
                            .cloned()
                            .unwrap_or_else(|| ctx.images.clone());
                        if let Some(overlay) = saccade_circle_overlay(
                            eye_frame.clone(),
                            mean.clone(),
                            sigma.clone(),
                            saccade_eye_color(eye_idx),
                        ) {
                            eye_frame = overlay;
                        }
                        push_view(&mut frame_views, eye_frame);
                    }
                    let step_patches = std::mem::take(&mut scratch.step_capture.patches);
                    if !step_patches.is_empty()
                        && let Some(patch_views) = saccade_patch_views(step_patches, ctx.height)
                    {
                        state.artifacts.last_patch_views = patch_views
                            .iter()
                            .map(|view| view.clone().detach())
                            .collect();
                        for patch_view in patch_views {
                            push_view(&mut frame_views, patch_view);
                        }
                        appended_patch = true;
                    }
                    if !appended_patch && !state.artifacts.last_patch_views.is_empty() {
                        for patch_view in &state.artifacts.last_patch_views {
                            push_view(&mut frame_views, patch_view.clone());
                        }
                    }
                    push_view(&mut frame_views, recon_view);
                    let frame = Tensor::cat(frame_views, 3);
                    state.artifacts.frame_steps.push(frame);
                }
                if ctx.capture_traj {
                    state.artifacts.traj_steps.push(step_traj);
                }
            }
            state.trajs.clear();
            state.trajs.append(&mut scratch.next_trajs);
            if step_idx < ctx.detach_until {
                for traj in &mut state.trajs {
                    *traj = traj.clone().detach();
                }
                for level in &mut state.state_levels {
                    *level = level.clone().detach();
                }
            }
            if ctx.tbptt_enabled && in_backprop {
                state.tbptt_step_idx += 1;
                let chunk_done = state.tbptt_step_idx >= ctx.tbptt_step_count
                    || step_idx + 1 == ctx.rollout_steps;
                if chunk_done {
                    state.tbptt_step_idx = 0;
                    state.tbptt_chunks += 1;
                    let state_composed = match self.config.pyramid_mode {
                        VisionPyramidMode::Stacked => state.state_levels.clone(),
                        VisionPyramidMode::Laplacian => {
                            self.compose_pyramid(&state.state_levels, &ctx.grids)
                        }
                    };
                    let state_composed_embed = self.project_pyramid_levels(&state_composed);
                    let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                        self.pyramid_lejepa_loss(&state_composed_embed)
                    } else {
                        let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
                        (zero.clone(), zero)
                    };
                    let (loss_per_sample, mask_per_sample, _) = self
                        .recon_loss_per_sample_from_projected_levels(
                            &state_composed_embed,
                            &ctx.target_patches,
                            Some(&ctx.loss_masks),
                            false,
                        );
                    let loss_sum = loss_per_sample.clone().sum();
                    let mask_sum = mask_per_sample.clone().sum();
                    state.tbptt_loss_sum = Some(match state.tbptt_loss_sum.take() {
                        Some(accum) => accum + loss_sum.clone(),
                        None => loss_sum,
                    });
                    state.tbptt_mask_sum = Some(match state.tbptt_mask_sum.take() {
                        Some(accum) => accum + mask_sum.clone(),
                        None => mask_sum,
                    });
                    state.tbptt_inv_sum = Some(match state.tbptt_inv_sum.take() {
                        Some(accum) => accum + inv.clone(),
                        None => inv,
                    });
                    state.tbptt_sigreg_sum = Some(match state.tbptt_sigreg_sum.take() {
                        Some(accum) => accum + sigreg.clone(),
                        None => sigreg,
                    });
                    if ctx.gdpo_policy_enabled {
                        let recon_per_sample =
                            loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
                        let hard_reward = state
                            .hard_reward
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 1>::zeros([ctx.batch], &ctx.device));
                        let log_prob_sum = state
                            .log_prob_sum
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                        let log_prob_sum_old = state
                            .log_prob_sum_old
                            .take()
                            .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                        let (log_prob_mean, entropy) = self.policy_log_prob_stats(
                            &log_prob_sum,
                            state.policy_steps,
                            ctx.num_eyes,
                            ctx.traj_len,
                        );
                        let action_clamp_rate = self.policy_action_clamp_rate(
                            state.clamp_rate_sum.take(),
                            state.clamp_rate_count,
                            ctx.batch,
                            &ctx.device,
                        );
                        state.tbptt_policy_inputs.push(GdpoPolicyInputs {
                            hard_reward,
                            recon_per_sample,
                            log_prob_sum,
                            log_prob_sum_old,
                            log_prob_mean,
                            entropy,
                            action_clamp_rate,
                            gdpo_group: ctx.gdpo_group,
                        });
                    }
                    if step_idx + 1 < ctx.rollout_steps {
                        state.reset_policy_accumulators(
                            ctx.gdpo_enabled,
                            ctx.info_reward_enabled,
                            ctx.batch,
                            &ctx.device,
                        );
                        for traj in &mut state.trajs {
                            *traj = traj.clone().detach();
                        }
                        for level in &mut state.state_levels {
                            *level = level.clone().detach();
                        }
                    }
                }
            }
        }
    }

    fn finalize_recon_loss(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
    ) -> SaccadeFinalizeOutput<B> {
        if ctx.tbptt_enabled {
            let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
            let chunk_count = state.tbptt_chunks.max(1) as f32;
            let inv_sum = state.tbptt_inv_sum.take().unwrap_or_else(|| zero.clone());
            let sigreg_sum = state
                .tbptt_sigreg_sum
                .take()
                .unwrap_or_else(|| zero.clone());
            let inv = inv_sum.mul_scalar(1.0 / chunk_count);
            let sigreg = sigreg_sum.mul_scalar(1.0 / chunk_count);
            let loss_sum = state.tbptt_loss_sum.take().unwrap_or_else(|| zero.clone());
            let mask_sum = state.tbptt_mask_sum.take().unwrap_or_else(|| zero.clone());
            let base_pair = if ctx.capture_artifacts {
                let (_, _, base_pair) = self.recon_loss_per_sample_from_state(
                    &state.state_levels,
                    &ctx.grids,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    true,
                );
                base_pair
            } else {
                None
            };
            let gdpo_inputs = if ctx.gdpo_policy_enabled {
                Some(std::mem::take(&mut state.tbptt_policy_inputs))
            } else {
                None
            };
            (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair)
        } else {
            let state_composed = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => state.state_levels.clone(),
                VisionPyramidMode::Laplacian => {
                    self.compose_pyramid(&state.state_levels, &ctx.grids)
                }
            };
            let state_composed_embed = self.project_pyramid_levels(&state_composed);
            let (inv, sigreg) = if self.config.loss.lejepa.enabled {
                self.pyramid_lejepa_loss(&state_composed_embed)
            } else {
                let zero = Tensor::<B, 1>::zeros([1], &ctx.device);
                (zero.clone(), zero)
            };
            let (loss_per_sample, mask_per_sample, base_pair) = self
                .recon_loss_per_sample_from_projected_levels(
                    &state_composed_embed,
                    &ctx.target_patches,
                    Some(&ctx.loss_masks),
                    ctx.capture_artifacts,
                );
            let loss_sum = loss_per_sample.clone().sum();
            let mask_sum = mask_per_sample.clone().sum();
            let recon_per_sample = loss_per_sample / mask_per_sample.add_scalar(LEJEPA_EPS);
            let gdpo_inputs = if ctx.gdpo_policy_enabled {
                let hard_reward = state
                    .hard_reward
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 1>::zeros([ctx.batch], &ctx.device));
                let log_prob_sum = state
                    .log_prob_sum
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                let log_prob_sum_old = state
                    .log_prob_sum_old
                    .take()
                    .unwrap_or_else(|| Tensor::<B, 2>::zeros([ctx.batch, 1], &ctx.device));
                let (log_prob_mean, entropy) = self.policy_log_prob_stats(
                    &log_prob_sum,
                    state.policy_steps,
                    ctx.num_eyes,
                    ctx.traj_len,
                );
                let action_clamp_rate = self.policy_action_clamp_rate(
                    state.clamp_rate_sum.take(),
                    state.clamp_rate_count,
                    ctx.batch,
                    &ctx.device,
                );
                Some(vec![GdpoPolicyInputs {
                    hard_reward,
                    recon_per_sample,
                    log_prob_sum,
                    log_prob_sum_old,
                    log_prob_mean,
                    entropy,
                    action_clamp_rate,
                    gdpo_group: ctx.gdpo_group,
                }])
            } else {
                None
            };
            (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair)
        }
    }

    fn build_recon_artifacts(
        &self,
        ctx: &SaccadeRolloutContext<B>,
        state: &mut SaccadeRolloutState<B>,
        base_pair: Option<(Tensor<B, 3>, Tensor<B, 3>)>,
    ) -> SaccadeArtifacts<B> {
        if !ctx.capture_artifacts || ctx.batch == 0 || ctx.tokens == 0 {
            return None;
        }
        let (pred_base, target_base) = if let Some((pred, target)) = base_pair {
            (Some(pred), Some(target))
        } else {
            (None, None)
        };
        let pred_first = pred_base.clone().unwrap_or_else(|| {
            Tensor::<B, 3>::zeros(
                [
                    ctx.batch,
                    ctx.tokens,
                    ctx.patch_size * ctx.patch_size * ctx.channels,
                ],
                &ctx.device,
            )
        });
        let target_first = target_base.clone().unwrap_or_else(|| {
            Tensor::<B, 3>::zeros(
                [
                    ctx.batch,
                    ctx.tokens,
                    ctx.patch_size * ctx.patch_size * ctx.channels,
                ],
                &ctx.device,
            )
        });
        let recon_view = unpatchify(
            pred_first.clone(),
            ctx.patch_size,
            ctx.height,
            ctx.width,
            ctx.channels,
        );
        let residual = pred_first - target_first;
        let mut target_width = ctx.width;
        for view in &ctx.view_images {
            target_width = target_width.max(view.shape().dims::<4>()[3]);
        }
        if !state.artifacts.last_patch_views.is_empty() {
            for patch_view in &state.artifacts.last_patch_views {
                target_width = target_width.max(patch_view.shape().dims::<4>()[3]);
            }
        }
        let mut combined_view = ctx
            .view_images
            .first()
            .cloned()
            .unwrap_or_else(|| ctx.images.clone());
        let mut per_eye_views = Vec::new();
        if let Some(last_step) = state.artifacts.traj_steps.last() {
            for (eye_idx, (mean, sigma)) in last_step.iter().enumerate() {
                if let Some(overlay) = saccade_circle_overlay(
                    combined_view.clone(),
                    mean.clone(),
                    sigma.clone(),
                    saccade_eye_color(eye_idx),
                ) {
                    combined_view = overlay;
                }
                let mut eye_view = ctx
                    .view_images
                    .get(eye_idx)
                    .cloned()
                    .unwrap_or_else(|| ctx.images.clone());
                if let Some(overlay) = saccade_circle_overlay(
                    eye_view.clone(),
                    mean.clone(),
                    sigma.clone(),
                    saccade_eye_color(eye_idx),
                ) {
                    eye_view = overlay;
                }
                per_eye_views.push((eye_idx, eye_view));
            }
        }
        let combined_view = pad_view_width(combined_view, target_width);
        let recon_view = pad_view_width(recon_view, target_width);
        let patch_views = if state.artifacts.last_patch_views.is_empty() {
            None
        } else {
            let patch_views = std::mem::take(&mut state.artifacts.last_patch_views);
            Some(
                patch_views
                    .into_iter()
                    .map(|patch_view| pad_view_width_centered(patch_view, target_width))
                    .collect::<Vec<_>>(),
            )
        };
        let mut views = Vec::new();
        let mut legend = Vec::new();
        views.push(combined_view);
        legend.push("input_with_fovea".to_string());
        for (eye_idx, eye_view) in per_eye_views {
            views.push(pad_view_width(eye_view, target_width));
            legend.push(format!("input_with_fovea_eye_{eye_idx}"));
        }
        if let Some(patch_views) = patch_views {
            for (eye_idx, patch_view) in patch_views.into_iter().enumerate() {
                views.push(patch_view);
                legend.push(format!("foveated_patch_eye_{eye_idx}"));
            }
        }
        views.push(recon_view);
        legend.push("reconstruction".to_string());
        if !state.artifacts.traj_steps.is_empty() {
            let steps = std::mem::take(&mut state.artifacts.traj_steps);
            let max_extra = self.config.artifact_max_views.saturating_sub(views.len());
            let mut remaining = max_extra;
            for idx in select_trajectory_indices(steps.len(), max_extra) {
                for (eye_idx, (mean, sigma)) in steps[idx].iter().enumerate() {
                    if remaining == 0 {
                        break;
                    }
                    let base_view = ctx
                        .view_images
                        .get(eye_idx)
                        .cloned()
                        .unwrap_or_else(|| ctx.images.clone());
                    if let Some(view) = saccade_circle_overlay(
                        base_view,
                        mean.clone(),
                        sigma.clone(),
                        saccade_eye_color(eye_idx),
                    ) {
                        views.push(pad_view_width(view, target_width));
                        legend.push(format!("trajectory_overlay_step_{idx}_eye_{eye_idx}"));
                        remaining = remaining.saturating_sub(1);
                    }
                }
                if remaining == 0 {
                    break;
                }
            }
        }
        let frames = if state.artifacts.frame_steps.is_empty() {
            None
        } else {
            let frames = std::mem::take(&mut state.artifacts.frame_steps);
            if frames.is_empty() {
                None
            } else {
                let mut max_width = 0;
                for frame in &frames {
                    let width = frame.shape().dims::<4>()[3];
                    max_width = max_width.max(width);
                }
                let mut stacked = Vec::with_capacity(frames.len());
                for frame in frames {
                    let frame = pad_view_width(frame, max_width);
                    stacked.push(frame.unsqueeze_dim::<5>(1));
                }
                Some(Tensor::cat(stacked, 1))
            }
        };
        Some((views, residual, frames, legend))
    }

    pub(crate) fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        view_images: Option<Tensor<B, 5>>,
        view_crops: Option<Tensor<B, 3>>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
        loss_on_all_patches: bool,
    ) -> SaccadeReconLossOutput<B> {
        let device = images.device();
        let [batch, channels, height, width] = images.shape().dims::<4>();
        let patch_size = self.model.patch_size().max(1);
        let num_eyes = self.config.num_eyes.max(1);
        let cross_view = self.config.cross_view.enabled && view_images.is_some() && num_eyes > 1;

        let (mut eye_views, view_crops) = if cross_view {
            let views = view_images.expect("view images");
            let [view_batch, view_count, _, _, _] = views.shape().dims::<5>();
            if view_batch != batch || view_count == 0 {
                let mut eye_views = Vec::with_capacity(num_eyes);
                for _ in 0..num_eyes {
                    eye_views.push(images.clone());
                }
                (eye_views, None)
            } else {
                let usable_views = view_count.min(num_eyes).max(1);
                let mut eye_views = Vec::with_capacity(num_eyes);
                for eye_idx in 0..num_eyes {
                    if eye_idx < usable_views {
                        let view = views
                            .clone()
                            .slice_dim(1, eye_idx..eye_idx + 1)
                            .reshape([batch, channels, height, width]);
                        eye_views.push(view);
                    } else {
                        eye_views.push(images.clone());
                    }
                }
                let view_crops = view_crops.and_then(|crops| {
                    let [crop_batch, crop_views, _] = crops.shape().dims::<3>();
                    if crop_batch != batch || crop_views == 0 {
                        return None;
                    }
                    let crops = if crop_views >= num_eyes {
                        crops.slice_dim(1, 0..num_eyes)
                    } else {
                        let pad = Tensor::<B, 3>::zeros([batch, num_eyes - crop_views, 4], &device);
                        Tensor::cat(vec![crops, pad], 1)
                    };
                    Some(crops)
                });
                (eye_views, view_crops)
            }
        } else {
            let mut eye_views = Vec::with_capacity(num_eyes);
            for _ in 0..num_eyes {
                eye_views.push(images.clone());
            }
            (eye_views, None)
        };

        let masked_eye = if cross_view {
            self.config
                .cross_view
                .masked_eye
                .min(num_eyes.saturating_sub(1))
        } else {
            0
        };
        let mask_ratio = if cross_view {
            self.config.loss.recon.mask_ratio
        } else {
            0.0
        };

        let (mip_levels, target_patches, loss_masks, grids, input_levels) = if cross_view {
            let (masked_levels, target_patches, masks) = self.build_masked_mip_pyramid(
                eye_views[masked_eye].clone(),
                patch_size,
                mask_ratio,
                randomize_mask,
                view_crops.clone(),
                masked_eye,
            );
            if masked_levels.is_empty() {
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
            }
            eye_views[masked_eye] = masked_levels[0].image.clone();
            let grids: Vec<PatchGrid> = masked_levels.iter().map(|level| level.grid).collect();
            let input_levels: Vec<Tensor<B, 3>> = masked_levels
                .iter()
                .map(|level| level.tokens.clone())
                .collect();
            let mut per_eye_levels = Vec::with_capacity(num_eyes);
            for (eye_idx, eye_view) in eye_views.iter().enumerate().take(num_eyes) {
                if eye_idx == masked_eye {
                    per_eye_levels.push(masked_levels.clone());
                } else {
                    let levels = self.build_mip_pyramid(eye_view.clone(), patch_size);
                    if levels.is_empty() {
                        let zero = Tensor::<B, 1>::zeros([1], &device);
                        return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
                    }
                    per_eye_levels.push(levels);
                }
            }
            let use_masks = !loss_on_all_patches && mask_ratio > 0.0;
            let loss_masks = if use_masks {
                masks.clone()
            } else {
                masks
                    .iter()
                    .map(|mask| {
                        let [mask_batch, mask_tokens] = mask.shape().dims::<2>();
                        Tensor::<B, 2>::ones([mask_batch, mask_tokens], &device)
                    })
                    .collect()
            };
            (
                per_eye_levels,
                target_patches,
                loss_masks,
                grids,
                input_levels,
            )
        } else {
            let base_levels = self.build_mip_pyramid(images.clone(), patch_size);
            if base_levels.is_empty() {
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero.clone(), zero.clone(), zero, None, None);
            }
            let grids: Vec<PatchGrid> = base_levels.iter().map(|level| level.grid).collect();
            let input_levels: Vec<Tensor<B, 3>> = base_levels
                .iter()
                .map(|level| level.tokens.clone())
                .collect();
            let target_patches: Vec<Tensor<B, 3>> = base_levels
                .iter()
                .map(|level| patchify(level.image.clone(), patch_size))
                .collect();
            let loss_masks: Vec<Tensor<B, 2>> = target_patches
                .iter()
                .map(|patches| {
                    let [mask_batch, mask_tokens, _] = patches.shape().dims::<3>();
                    Tensor::<B, 2>::ones([mask_batch, mask_tokens], &device)
                })
                .collect();
            (
                vec![base_levels; num_eyes],
                target_patches,
                loss_masks,
                grids,
                input_levels,
            )
        };

        let embed_dim = input_levels
            .first()
            .map(|level| level.shape().dims::<3>()[2])
            .unwrap_or(0)
            .max(1);
        let tokens = grids.first().map(|grid| grid.num_patches()).unwrap_or(0);
        let view_embed =
            if let (Some(view_crops), Some(view_embed)) = (view_crops, self.view_embed.as_ref()) {
                let [crop_batch, crop_eyes, _] = view_crops.shape().dims::<3>();
                if crop_batch == 0 || crop_eyes == 0 {
                    None
                } else {
                    let flat = view_crops.clone().reshape([crop_batch * crop_eyes, 4]);
                    let embed = view_embed
                        .forward(flat)
                        .reshape([crop_batch, crop_eyes, 1, embed_dim]);
                    Some(embed)
                }
            } else {
                None
            };

        let input_state_levels = if cross_view {
            let mut averaged = Vec::with_capacity(input_levels.len());
            for level_idx in 0..input_levels.len() {
                let mut sum: Option<Tensor<B, 3>> = None;
                let mut count = 0.0f32;
                for levels in &mip_levels {
                    if let Some(level) = levels.get(level_idx) {
                        sum = Some(match sum {
                            Some(accum) => accum + level.tokens.clone(),
                            None => level.tokens.clone(),
                        });
                        count += 1.0;
                    }
                }
                let fallback = input_levels.get(level_idx).cloned().unwrap_or_else(|| {
                    let [batch, tokens, dim] = input_levels
                        .first()
                        .map(|t| t.shape().dims::<3>())
                        .unwrap_or([0, 0, 0]);
                    Tensor::<B, 3>::zeros([batch, tokens, dim], &device)
                });
                let avg = sum
                    .map(|sum| sum.mul_scalar(1.0 / count.max(1.0)))
                    .unwrap_or(fallback);
                averaged.push(avg);
            }
            averaged
        } else {
            input_levels.clone()
        };
        let input_residuals = match self.config.pyramid_mode {
            VisionPyramidMode::Stacked => input_state_levels.clone(),
            VisionPyramidMode::Laplacian => self.decompose_pyramid(&input_state_levels, &grids),
        };
        let laplacian_images = if matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            let mut images = Vec::with_capacity(num_eyes);
            let mut ok = true;
            for levels in &mip_levels {
                if let Some(laplacian) = self.build_laplacian_images(levels) {
                    images.push(laplacian);
                } else {
                    ok = false;
                    break;
                }
            }
            if ok { Some(images) } else { None }
        } else {
            None
        };
        let base_grid = self.fovea_base_grid(patch_size, &device);
        let traj_len = self.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let inner_steps = self.config.inner_steps.max(1);
        let traj_update_alpha = self.config.traj_update_alpha;

        let base_traj = self
            .trajectory_token
            .val()
            .reshape([1, traj_len, embed_dim])
            .repeat_dim(0, batch);
        let trajs = vec![base_traj; num_eyes];
        let state_levels: Vec<Tensor<B, 3>> = if cross_view {
            input_residuals.clone()
        } else {
            input_residuals
                .iter()
                .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
                .collect()
        };
        let rollout_steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(rollout_steps);
        let detach_until = rollout_steps.saturating_sub(backprop_steps);
        let tbptt_step_count = self.config.tbptt.step_count;
        let tbptt_step_count = if tbptt_step_count == 0 {
            0
        } else {
            tbptt_step_count.max(1).min(backprop_steps)
        };
        let tbptt_enabled = tbptt_step_count > 0;
        let low_mem_pre_rollout = self.config.low_mem_pre_rollout;
        let capture_traj =
            capture_artifacts && self.config.artifact_max_views.saturating_sub(3) > 0;
        let gdpo = &self.config.policy.gdpo;
        let gdpo_enabled = gdpo.enabled && !capture_artifacts;
        let gdpo_group = gdpo.group_size.max(1);
        let gdpo_policy_enabled = gdpo_enabled && gdpo.policy_weight > 0.0;
        let info_reward_enabled = gdpo_enabled && self.config.policy.info_reward.enabled;
        let info_stride = self.config.policy.info_reward.stride.max(1);
        let detach_policy_from_recon = self.config.policy.detach_policy_from_recon;
        let log_prob_sum = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let log_prob_sum_old = if gdpo_enabled {
            Some(Tensor::<B, 2>::zeros([batch, 1], &device))
        } else {
            None
        };
        let hard_reward = if info_reward_enabled {
            Some(Tensor::<B, 1>::zeros([batch], &device))
        } else {
            None
        };
        // Heap-allocate rollout buffers to keep the stack frame small.
        let ctx = Box::new(SaccadeRolloutContext {
            device,
            images: eye_views.first().cloned().unwrap_or_else(|| images.clone()),
            view_images: eye_views,
            view_embed,
            batch,
            channels,
            height,
            width,
            patch_size,
            embed_dim,
            tokens,
            mip_levels,
            grids,
            target_patches,
            loss_masks,
            laplacian_images,
            base_grid,
            traj_len,
            num_eyes,
            inner_steps,
            traj_update_alpha,
            rollout_steps,
            detach_until,
            tbptt_step_count,
            tbptt_enabled,
            low_mem_pre_rollout,
            capture_traj,
            capture_artifacts,
            gdpo_enabled,
            gdpo_group,
            gdpo_policy_enabled,
            info_reward_enabled,
            info_stride,
            detach_policy_from_recon,
        });
        let mut state = Box::new(SaccadeRolloutState::new(
            trajs,
            state_levels,
            log_prob_sum,
            log_prob_sum_old,
            hard_reward,
            capture_traj,
            capture_artifacts,
            rollout_steps,
        ));
        let mut scratch = Box::new(SaccadeStepScratch::new());

        self.run_recon_rollout(&ctx, &mut state, &mut scratch);
        let (loss_sum, mask_sum, inv, sigreg, gdpo_inputs, base_pair) =
            self.finalize_recon_loss(&ctx, &mut state);
        let artifacts = self.build_recon_artifacts(&ctx, &mut state, base_pair);

        (loss_sum, mask_sum, inv, sigreg, artifacts, gdpo_inputs)
    }
}
