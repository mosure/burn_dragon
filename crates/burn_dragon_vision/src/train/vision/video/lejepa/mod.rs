//! Intermediate `video_lejepa` training surface.
//!
//! `VisionVideoLejepaModel` remains crate-private on purpose while the video stack is still in the
//! hybrid/intermediate phase. Shared-core TRM migration work should treat this module as the
//! current implementation seam, not as the final public video API.

use super::profile::{video_train_profile_enabled, video_train_profile_record};
use crate::train::prelude::*;
use crate::train::vision::video::dynamics::{
    SplitClipProjectionTrainConfig, VisionVideoContextForward, VisionVideoForward,
    VisionVideoObservationMerger, VisionVideoPredictor, VisionVideoRolloutOutput,
    encode_clip_frames_with_model, repeat_last_future_query,
    split_clip_observation_and_target_projections_train,
};
use burn::nn::loss::CrossEntropyLossConfig;
use burn_dragon_core::{
    BDH, BDHConfig, FusedKernelConfig, ModelState, RotaryEmbedding, StructuredStepMode,
    StructuredTopologyState,
};
use std::time::Instant;

mod rollout;
#[cfg(test)]
mod tests;

use rollout::VideoDebugReconInput;

#[derive(Debug, Clone)]
pub(crate) struct VisionVideoLejepaModel<B: BackendTrait> {
    pub(crate) frame_model: VisionDragon<B>,
    pub(crate) temporal_model: Option<BDH<B>>,
    pub(crate) predictor: VisionVideoPredictor<B>,
    pub(crate) patch_conditioner: VisionVideoPredictor<B>,
    pub(crate) observation_merger: VisionVideoObservationMerger<B>,
    pub(crate) step_mode_embeddings: Option<Param<Tensor<B, 2>>>,
    pub(crate) probe: VisionProbe<B>,
    pub(crate) recon: VisionReconstructionHead<B>,
    pub(crate) probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    pub(crate) future_queries: Option<Param<Tensor<B, 2>>>,
    pub(crate) config: VisionVideoLejepaConfig,
    pub(crate) teacher_frame_model: Option<VisionDragon<B>>,
    pub(crate) rollout: VisionRollout,
    pub(crate) embed_dim: usize,
    pub(crate) projection_dim: usize,
}

#[derive(burn::record::Record)]
pub(crate) struct VisionVideoLejepaModelRecord<B: BackendTrait> {
    pub(crate) frame_model: <VisionDragon<B> as Module<B>>::Record,
    pub(crate) temporal_model: <Option<BDH<B>> as Module<B>>::Record,
    pub(crate) predictor: <VisionVideoPredictor<B> as Module<B>>::Record,
    pub(crate) patch_conditioner: <VisionVideoPredictor<B> as Module<B>>::Record,
    pub(crate) observation_merger: <VisionVideoObservationMerger<B> as Module<B>>::Record,
    pub(crate) step_mode_embeddings: <Option<Param<Tensor<B, 2>>> as Module<B>>::Record,
    pub(crate) probe: <VisionProbe<B> as Module<B>>::Record,
    pub(crate) recon: <VisionReconstructionHead<B> as Module<B>>::Record,
    pub(crate) probe_loss: <burn::nn::loss::CrossEntropyLoss<B> as Module<B>>::Record,
    pub(crate) future_queries: <Option<Param<Tensor<B, 2>>> as Module<B>>::Record,
    pub(crate) config: <VisionVideoLejepaConfig as Module<B>>::Record,
}

pub(crate) struct VisionVideoLejepaLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) inv: Tensor<B, 1>,
    pub(crate) observe: Tensor<B, 1>,
    pub(crate) mode_separation_ratio: Tensor<B, 1>,
    pub(crate) sigreg: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) recon_psnr_masked: Tensor<B, 1>,
    pub(crate) recon_psnr_full: Tensor<B, 1>,
    pub(crate) probe_loss: Tensor<B, 1>,
    pub(crate) probe_acc: Tensor<B, 1>,
    pub(crate) rollout_inv_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    pub(crate) rollout_state_norm_ratio_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    pub(crate) rollout_state_motion_to_horizon: [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT],
    pub(crate) rollout_com_error_to_h24: Option<Tensor<B, 1>>,
    pub(crate) rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    pub(crate) long_rollout_com_error_to_h24: Option<Tensor<B, 1>>,
    pub(crate) long_rollout_velocity_error_to_h24: Option<Tensor<B, 1>>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

struct VisionVideoDebugRecon<B: BackendTrait> {
    loss: Tensor<B, 1>,
    psnr_short: Tensor<B, 1>,
    psnr_full: Tensor<B, 1>,
    clip: Option<Tensor<B, 5>>,
    pca_rgb_steps: Option<Tensor<B, 5>>,
}

type VideoArtifactMaps<B> = (
    Option<Tensor<B, 3>>,
    Option<Tensor<B, 4>>,
    Option<Tensor<B, 4>>,
    Option<Tensor<B, 5>>,
);

fn collect_video_feature_maps<B: BackendTrait>(
    frame_patch_tokens: Tensor<B, 4>,
    image_count: usize,
    include_heatmaps: bool,
) -> VideoArtifactMaps<B> {
    let [batch, frame_count, tokens, dim] = frame_patch_tokens.shape().dims::<4>();
    let image_count = image_count.min(batch);
    if image_count == 0 || frame_count == 0 || tokens == 0 || dim == 0 {
        return (None, None, None, None);
    }

    let patch_tokens = frame_patch_tokens.slice_dim(0, 0..image_count);
    let flat = patch_tokens
        .clone()
        .reshape([image_count * frame_count, tokens, dim]);
    let patch_norms_steps = if include_heatmaps {
        patch_heatmap_or_norm(flat.clone(), image_count * frame_count).map(|maps| {
            let [_, grid_h, grid_w] = maps.shape().dims::<3>();
            maps.reshape([image_count, frame_count, grid_h, grid_w])
        })
    } else {
        None
    };
    let pca_rgb_steps = pca_patch_rgb(&flat, image_count * frame_count).map(|maps| {
        let [_, channels, grid_h, grid_w] = maps.shape().dims::<4>();
        maps.reshape([image_count, frame_count, channels, grid_h, grid_w])
    });

    let last_frame = patch_tokens
        .slice_dim(1, (frame_count - 1)..frame_count)
        .reshape([image_count, tokens, dim]);
    let patch_norms = if include_heatmaps {
        patch_heatmap_or_norm(last_frame.clone(), image_count)
    } else {
        None
    };
    let pca_rgb = pca_patch_rgb(&last_frame, image_count);

    (patch_norms, pca_rgb, patch_norms_steps, pca_rgb_steps)
}

impl<B: BackendTrait> VisionVideoLejepaModel<B> {
    pub(crate) fn new(
        frame_model: VisionDragon<B>,
        config: VisionVideoLejepaConfig,
        vision: &VisionDragonConfig,
        rollout: VisionRollout,
        num_classes: usize,
        device: &B::Device,
    ) -> Self {
        let embed_dim = vision.embed_dim.max(1);
        let projection_dim = vision.projection_dim.max(1);
        let pyramid_backbone = frame_model.pyramid_backbone_enabled();
        let predictor_hidden_dim = if config.predictor_hidden_dim == 0 {
            vision.projection_hidden_dim.max(projection_dim)
        } else {
            config.predictor_hidden_dim
        };
        let temporal_model = if pyramid_backbone {
            None
        } else {
            Some(BDH::<B>::new(
                build_temporal_config(embed_dim, &config.temporal, &vision.normalization),
                device,
            ))
        };
        let predictor = VisionVideoPredictor::new(
            embed_dim,
            predictor_hidden_dim,
            projection_dim,
            &vision.normalization,
            device,
        );
        let patch_conditioner = VisionVideoPredictor::new(
            embed_dim,
            predictor_hidden_dim,
            embed_dim,
            &vision.normalization,
            device,
        );
        let observation_merger =
            VisionVideoObservationMerger::new(embed_dim, &vision.normalization, device);
        let step_mode_embeddings = if config.temporal.mode_embeddings {
            Some(Param::from_tensor(Tensor::<B, 2>::random(
                [StructuredStepMode::COUNT, embed_dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            )))
        } else {
            None
        };
        let probe = VisionProbe::new(embed_dim, num_classes, &vision.normalization, device);
        let patch_size = vision.patch_size.max(1);
        let patch_dim = vision.in_channels.max(1) * patch_size * patch_size;
        let recon_hidden_dim = if config.loss.debug_recon_hidden_dim == 0 {
            vision.projection_hidden_dim.max(embed_dim * 2).max(256)
        } else {
            config.loss.debug_recon_hidden_dim
        };
        let recon = VisionReconstructionHead::new(
            embed_dim,
            recon_hidden_dim,
            patch_dim,
            true,
            &vision.normalization,
            device,
        );
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let future_queries = if pyramid_backbone {
            None
        } else {
            Some(Param::from_tensor(Tensor::<B, 2>::random(
                [config.max_supervised_target_frames(), embed_dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            )))
        };
        let teacher_frame_model = init_momentum_teacher::<B, _>(&frame_model, &config.teacher_ema);

        Self {
            frame_model,
            temporal_model,
            predictor,
            patch_conditioner,
            observation_merger,
            step_mode_embeddings,
            probe,
            recon,
            probe_loss,
            future_queries,
            config,
            teacher_frame_model,
            rollout,
            embed_dim,
            projection_dim,
        }
    }

    pub(crate) fn sync_teacher_from_student(mut self) -> Self {
        self.teacher_frame_model = sync_optional_teacher_from_student::<B, _>(
            self.teacher_frame_model.take(),
            &self.frame_model,
            self.config.teacher_ema.enabled,
            self.config.teacher_ema.decay,
        );
        self
    }

    pub(crate) fn restore_teacher_from_student(mut self) -> Self {
        self.teacher_frame_model = restore_optional_teacher_from_student::<B, _>(
            &self.frame_model,
            self.config.teacher_ema.enabled,
        );
        self
    }

    pub(crate) fn uses_pyramid_backbone(&self) -> bool {
        self.frame_model.pyramid_backbone_enabled()
    }

    fn temporal_model_ref(&self) -> &BDH<B> {
        self.temporal_model
            .as_ref()
            .expect("video temporal tower missing for non-pyramid rollout path")
    }

    fn future_queries_ref(&self) -> Tensor<B, 2> {
        self.future_queries
            .as_ref()
            .expect("future queries missing for non-pyramid rollout path")
            .val()
    }

    fn weighted_loss_term(&self, term: Tensor<B, 1>, weight: f32) -> Option<Tensor<B, 1>> {
        let weight = weight.max(0.0);
        if weight <= 0.0 {
            None
        } else if (weight - 1.0).abs() <= f32::EPSILON {
            Some(term)
        } else {
            Some(term.mul_scalar(weight))
        }
    }

    fn accumulate_loss_terms(
        &self,
        zero: &Tensor<B, 1>,
        terms: Vec<Option<Tensor<B, 1>>>,
    ) -> Tensor<B, 1> {
        let mut iter = terms.into_iter().flatten();
        let Some(mut total) = iter.next() else {
            return zero.clone();
        };
        for term in iter {
            total = total + term;
        }
        total
    }

    pub(crate) fn forward_losses_train_pyramid(
        &self,
        batch: VideoClipBatch<B>,
        steps: usize,
        backprop_steps: usize,
        compute_rollout_metrics: bool,
        compute_mode_separation: bool,
    ) -> VisionVideoLejepaLosses<B> {
        video_train_profile_record(|state| {
            state.train_calls += 1;
        });
        let clip_frames = batch.clip_frames;
        let labels = batch.labels;
        let [batch_size, clip_len, _channels, height, width] = clip_frames.shape().dims::<5>();
        let context_len = batch.context_len.min(clip_len.saturating_sub(1)).max(1);
        let target_end = (context_len + batch.target_len).min(clip_len);
        let target_len = target_end.saturating_sub(context_len).max(1);
        let available_future_len = clip_len.saturating_sub(context_len).max(target_len);
        // Plain train steps only supervise `target_len`, so avoid rolling/projecting the
        // unsupervised tail unless rollout metrics explicitly request it.
        let future_len_all = if compute_rollout_metrics {
            available_future_len
        } else {
            target_len
        };
        let projection_dim = self.projection_dim.max(1);
        let patch_count = ((height / self.frame_model.patch_size().max(1))
            * (width / self.frame_model.patch_size().max(1)))
        .max(1);
        let device = clip_frames.device();
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let collect_patch_tokens =
            compute_rollout_metrics || self.config.loss.debug_recon_weight > 0.0;
        let collect_predicted_proj = compute_rollout_metrics || self.config.loss.sigreg.enabled;

        let split_start = video_train_profile_enabled().then(Instant::now);
        let include_context_target_proj = self.config.loss.observe_weight > 0.0;
        let (context_observation_patch_tokens, context_target_proj, target_proj_all) =
            split_clip_observation_and_target_projections_train(
                &self.frame_model,
                self.teacher_frame_model.as_ref(),
                clip_frames.clone(),
                SplitClipProjectionTrainConfig {
                    context_len,
                    future_len_all,
                    steps,
                    projection_dim,
                    include_context_target_proj,
                },
            );
        if let Some(start) = split_start {
            video_train_profile_record(|state| {
                state.split_projection_ns += start.elapsed().as_nanos() as u64;
            });
        }

        let mut observe = zero.clone();
        let mut previous_state: Option<StructuredTopologyState<B>> = None;
        let mut context_patch_tokens_detached = if collect_patch_tokens {
            Vec::with_capacity(context_len)
        } else {
            Vec::new()
        };
        let mut last_refined_cls: Option<Tensor<B, 2>> = None;

        let context_start = video_train_profile_enabled().then(Instant::now);
        for step_idx in 0..context_len {
            let observation = context_observation_patch_tokens
                .clone()
                .slice_dim(1, step_idx..step_idx + 1)
                .reshape([batch_size, patch_count, self.embed_dim]);

            let posterior_state = if let Some(prior_state) = previous_state.clone() {
                let prior_state = self.rollout_pyramid_step(
                    prior_state,
                    last_refined_cls.clone(),
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

            if include_context_target_proj {
                let posterior_cls = self.frame_model.pyramid_summary(&posterior_state);
                let observation_proj_step = self
                    .frame_model
                    .project_tokens(posterior_cls.unsqueeze_dim::<3>(1))
                    .reshape([batch_size, projection_dim]);
                let observation_target_step = context_target_proj
                    .as_ref()
                    .expect("context target projections should exist when observe loss is enabled")
                    .clone()
                    .slice_dim(1, step_idx..step_idx + 1)
                    .reshape([batch_size, projection_dim]);
                observe = observe
                    + (observation_proj_step - observation_target_step)
                        .powf_scalar(2.0)
                        .mean();
            }

            let refined_state =
                self.maybe_refine_pyramid_state(posterior_state, steps, backprop_steps);
            if collect_patch_tokens {
                let refined_patch = self.frame_model.pyramid_patch_tokens(&refined_state);
                context_patch_tokens_detached.push(refined_patch.detach().unsqueeze_dim::<4>(1));
            }
            let refined_cls = self.frame_model.pyramid_summary(&refined_state);
            last_refined_cls = Some(refined_cls);
            previous_state = Some(refined_state);
        }
        if let Some(start) = context_start {
            video_train_profile_record(|state| {
                state.context_rollout_ns += start.elapsed().as_nanos() as u64;
            });
        }

        if include_context_target_proj {
            observe = observe.div_scalar(context_len as f32);
        }

        let mut structured_state = previous_state.expect("pyramid context state missing");
        let context_summary = last_refined_cls.expect("pyramid context summary missing");
        let mut current_summary = context_summary.clone();
        let mode_separation_ratio = if compute_mode_separation {
            self.mode_separation_ratio_structured(
                structured_state.clone().detach(),
                context_summary.clone().detach(),
                steps,
                backprop_steps,
            )
        } else {
            zero.clone()
        };

        let mut prediction = zero.clone();
        let mut cosine = zero.clone();
        let mut predicted_proj_steps = if collect_predicted_proj {
            Vec::with_capacity(future_len_all)
        } else {
            Vec::new()
        };
        let mut future_patch_tokens_detached = if collect_patch_tokens {
            Vec::with_capacity(future_len_all)
        } else {
            Vec::new()
        };
        let predict_start = video_train_profile_enabled().then(Instant::now);
        for step_idx in 0..future_len_all {
            structured_state = self.rollout_pyramid_step(
                structured_state,
                Some(current_summary),
                steps,
                backprop_steps,
                StructuredStepMode::Predict,
            );
            let future_cls = self.frame_model.pyramid_summary(&structured_state);
            current_summary = future_cls.clone();
            let predicted_proj_step = self
                .predictor
                .forward(future_cls.clone().unsqueeze_dim::<3>(1));
            if step_idx < target_len {
                let target_proj_step = target_proj_all.clone().slice_dim(1, step_idx..step_idx + 1);
                prediction = prediction
                    + (predicted_proj_step.clone() - target_proj_step.clone())
                        .powf_scalar(2.0)
                        .mean();
                cosine =
                    cosine + temporal_cosine_loss(predicted_proj_step.clone(), target_proj_step);
            }
            if collect_predicted_proj {
                predicted_proj_steps.push(predicted_proj_step);
            }
            if collect_patch_tokens {
                future_patch_tokens_detached.push(
                    self.frame_model
                        .pyramid_patch_tokens(&structured_state)
                        .detach()
                        .unsqueeze_dim::<4>(1),
                );
            }
            if self.should_detach_predict_rollout(step_idx, future_len_all) {
                structured_state = structured_state.detach();
            }
        }
        if let Some(start) = predict_start {
            video_train_profile_record(|state| {
                state.predict_rollout_ns += start.elapsed().as_nanos() as u64;
            });
        }

        prediction = prediction.div_scalar(target_len as f32);
        cosine = cosine.div_scalar(target_len as f32);
        let predicted_proj_all = if collect_predicted_proj {
            Some(Tensor::cat(predicted_proj_steps, 1))
        } else {
            None
        };
        let sigreg = if self.config.loss.sigreg.enabled {
            let predicted_proj_all = predicted_proj_all
                .clone()
                .expect("predicted projections should be collected when sigreg is enabled");
            lejepa_sigreg_loss(predicted_proj_all.swap_dims(0, 1), &self.config.loss.sigreg)
        } else {
            zero.clone()
        };

        let loss_heads_start = video_train_profile_enabled().then(Instant::now);
        let debug_recon = if self.config.loss.debug_recon_weight > 0.0 {
            let context_patch_tokens = Tensor::cat(context_patch_tokens_detached.clone(), 1);
            let future_patch_tokens = Tensor::cat(future_patch_tokens_detached.clone(), 1);
            let frame_patch_tokens =
                Tensor::cat(vec![context_patch_tokens, future_patch_tokens.clone()], 1);
            self.debug_reconstruction_outputs_from_patch_tokens(VideoDebugReconInput {
                clip_frames: clip_frames.clone(),
                frame_patch_tokens,
                future_patch_tokens,
                context_len,
                target_len,
                future_len_all,
                steps,
                capture_artifacts: false,
            })
        } else {
            VisionVideoDebugRecon {
                loss: zero.clone(),
                psnr_short: zero.clone(),
                psnr_full: zero.clone(),
                clip: None,
                pca_rgb_steps: None,
            }
        };
        let (
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
        ) = if compute_rollout_metrics {
            let context_patch_tokens_for_metrics = Tensor::cat(context_patch_tokens_detached, 1);
            let future_patch_tokens_for_metrics = Tensor::cat(future_patch_tokens_detached, 1);
            let frame_patch_tokens_for_metrics = Tensor::cat(
                vec![
                    context_patch_tokens_for_metrics,
                    future_patch_tokens_for_metrics.clone(),
                ],
                1,
            );
            let predicted_proj_all_for_metrics = predicted_proj_all
                .expect("predicted projections should be collected for rollout metrics")
                .detach();
            let target_proj_all_for_metrics = target_proj_all.detach();
            let frame_patch_tokens_for_metrics = frame_patch_tokens_for_metrics.detach();
            let future_patch_tokens_for_metrics = future_patch_tokens_for_metrics.detach();
            self.rollout_horizon_metrics_from_tensors(
                predicted_proj_all_for_metrics,
                target_proj_all_for_metrics,
                frame_patch_tokens_for_metrics,
                future_patch_tokens_for_metrics,
                context_len,
            )
        } else {
            (
                core::array::from_fn(|_| zero.clone()),
                core::array::from_fn(|_| zero.clone()),
                core::array::from_fn(|_| zero.clone()),
            )
        };

        let total = self.accumulate_loss_terms(
            &zero,
            vec![
                self.weighted_loss_term(prediction.clone(), self.config.loss.prediction_weight),
                self.weighted_loss_term(observe.clone(), self.config.loss.observe_weight),
                self.weighted_loss_term(cosine.clone(), self.config.loss.cosine_weight),
                self.weighted_loss_term(sigreg.clone(), self.config.loss.sigreg.lambda),
                self.weighted_loss_term(
                    debug_recon.loss.clone(),
                    self.config.loss.debug_recon_weight,
                ),
            ],
        );

        let (probe_loss, probe_acc) = if self.config.loss.probe_weight > 0.0 {
            let probe_logits = self.probe.forward(context_summary.clone().detach());
            let probe_loss = self
                .probe_loss
                .forward(probe_logits.clone(), labels.clone());
            let probe_pred = probe_logits
                .argmax(1)
                .reshape([labels.shape().dims::<1>()[0]]);
            let probe_acc = probe_pred.equal(labels).float().mean();
            (probe_loss, probe_acc)
        } else {
            (zero.clone(), zero.clone())
        };
        if let Some(start) = loss_heads_start {
            video_train_profile_record(|state| {
                state.loss_heads_ns += start.elapsed().as_nanos() as u64;
            });
        }

        VisionVideoLejepaLosses {
            total,
            inv: prediction,
            observe,
            mode_separation_ratio,
            sigreg,
            recon: debug_recon.loss,
            recon_psnr_masked: debug_recon.psnr_short,
            recon_psnr_full: debug_recon.psnr_full,
            probe_loss,
            probe_acc,
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
            rollout_com_error_to_h24: None,
            rollout_velocity_error_to_h24: None,
            long_rollout_com_error_to_h24: None,
            long_rollout_velocity_error_to_h24: None,
            artifacts: None,
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: VideoClipBatch<B>,
        steps: usize,
        backprop_steps: usize,
        capture_artifacts: bool,
        compute_long_rollout_metrics: bool,
        compute_mode_separation: bool,
    ) -> VisionVideoLejepaLosses<B> {
        let requested_future_len =
            if compute_long_rollout_metrics && self.config.artifact_future_frames > 0 {
                self.config
                    .artifact_future_frames
                    .max(self.config.max_supervised_target_frames())
            } else {
                self.config.max_supervised_target_frames()
            };
        let forward = self.forward_video(batch, steps, backprop_steps, requested_future_len);
        let device = forward.predicted_proj_all.device();
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let [batch_size, future_len_all, _projection_dim] =
            forward.predicted_proj_all.shape().dims::<3>();
        let supervision_mask =
            supervised_time_mask(batch_size, future_len_all, forward.target_len, &device);
        let prediction = temporal_mse_loss_masked(
            forward.predicted_proj_all.clone(),
            forward.target_proj_all.clone(),
            supervision_mask.clone(),
        );
        let observe = (forward.observation_proj.clone() - forward.observation_target_proj.clone())
            .powf_scalar(2.0)
            .mean();
        let cosine = temporal_cosine_loss_masked(
            forward.predicted_proj_all.clone(),
            forward.target_proj_all.clone(),
            supervision_mask.clone(),
        );
        let mode_separation_ratio = if compute_mode_separation {
            self.mode_separation_ratio(&forward, steps, backprop_steps)
        } else {
            zero.clone()
        };
        let sigreg = if self.config.loss.sigreg.enabled {
            lejepa_sigreg_loss(
                (forward.predicted_proj_all.clone() * supervision_mask.clone()).swap_dims(0, 1),
                &self.config.loss.sigreg,
            )
        } else {
            zero.clone()
        };
        let (
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
        ) = self.rollout_horizon_metrics(&forward);
        let debug_recon = if capture_artifacts
            || compute_long_rollout_metrics
            || self.config.loss.debug_recon_weight > 0.0
        {
            let sample_batch_limit = if capture_artifacts || compute_long_rollout_metrics {
                Some(self.config.artifact_max_images.max(1))
            } else {
                None
            };
            self.debug_reconstruction_outputs(
                &forward,
                steps,
                capture_artifacts,
                compute_long_rollout_metrics,
                sample_batch_limit,
            )
        } else {
            VisionVideoDebugRecon {
                loss: zero.clone(),
                psnr_short: zero.clone(),
                psnr_full: zero.clone(),
                clip: None,
                pca_rgb_steps: None,
            }
        };
        let (rollout_com_error_to_h24, rollout_velocity_error_to_h24) =
            self.rollout_kinematics_metrics(&forward, debug_recon.clip.clone(), forward.target_len);
        let (long_rollout_com_error_to_h24, long_rollout_velocity_error_to_h24) =
            if compute_long_rollout_metrics && forward.future_len_all > forward.target_len {
                self.rollout_kinematics_metrics(
                    &forward,
                    debug_recon.clip.clone(),
                    forward.future_len_all,
                )
            } else {
                (None, None)
            };
        let total = self.accumulate_loss_terms(
            &zero,
            vec![
                self.weighted_loss_term(prediction.clone(), self.config.loss.prediction_weight),
                self.weighted_loss_term(observe.clone(), self.config.loss.observe_weight),
                self.weighted_loss_term(cosine.clone(), self.config.loss.cosine_weight),
                self.weighted_loss_term(sigreg.clone(), self.config.loss.sigreg.lambda),
                self.weighted_loss_term(
                    debug_recon.loss.clone(),
                    self.config.loss.debug_recon_weight,
                ),
            ],
        );
        let (probe_loss, probe_acc) = if self.config.loss.probe_weight > 0.0 {
            let probe_logits = forward
                .probe_logits
                .clone()
                .expect("probe logits should exist when probe loss is enabled");
            let probe_labels = forward
                .probe_labels
                .clone()
                .expect("probe labels should exist when probe loss is enabled");
            let probe_loss = self
                .probe_loss
                .forward(probe_logits.clone(), probe_labels.clone());
            let probe_pred = probe_logits
                .argmax(1)
                .reshape([probe_labels.shape().dims::<1>()[0]]);
            let probe_acc = probe_pred.equal(probe_labels).float().mean();
            (probe_loss, probe_acc)
        } else {
            (zero.clone(), zero.clone())
        };

        let artifacts = if capture_artifacts && self.config.artifact_every > 0 {
            let [batch_size, clip_len, channels, height, width] =
                forward.clip_frames.shape().dims::<5>();
            let image_count = self.config.artifact_max_images.min(batch_size);
            if image_count == 0 || clip_len == 0 {
                None
            } else {
                let (_, _, _reference_patch_norms_steps, reference_pca_rgb_steps) =
                    collect_video_feature_maps(
                        forward.frame_patch_tokens.clone(),
                        image_count,
                        false,
                    );
                let (_, _, _posterior_patch_norms_steps, posterior_pca_rgb_steps) =
                    collect_video_feature_maps(
                        forward
                            .context_posterior_patch_tokens
                            .clone()
                            .slice_dim(0, 0..image_count),
                        image_count,
                        false,
                    );
                let posterior_pca_rgb_steps = posterior_pca_rgb_steps.map(|maps| {
                    if forward.future_len_all == 0 {
                        maps
                    } else {
                        let [count, _context_len, channels_rgb, grid_h, grid_w] =
                            maps.shape().dims::<5>();
                        let zeros = Tensor::<B, 5>::zeros(
                            [count, forward.future_len_all, channels_rgb, grid_h, grid_w],
                            &maps.device(),
                        );
                        Tensor::cat(vec![maps, zeros], 1)
                    }
                });
                if self.config.artifact_output == VisionArtifactOutputMode::Images {
                    let reference_frame = forward
                        .clip_frames
                        .clone()
                        .slice_dim(0, 0..image_count)
                        .slice_dim(
                            1,
                            (forward.context_len + forward.future_len_all - 1)
                                ..(forward.context_len + forward.future_len_all),
                        )
                        .reshape([image_count, channels, height, width]);
                    let debug_clip = debug_recon.clip.clone().unwrap_or_else(|| {
                        forward.clip_frames.clone().slice_dim(0, 0..image_count)
                    });
                    let recon_frame = debug_clip
                        .clone()
                        .slice_dim(1, (clip_len - 1)..clip_len)
                        .reshape([image_count, channels, height, width]);
                    let views = Tensor::cat(
                        vec![
                            reference_frame.unsqueeze_dim::<5>(1),
                            recon_frame.unsqueeze_dim::<5>(1),
                        ],
                        1,
                    );
                    let pca_rgb = reference_pca_rgb_steps.as_ref().map(|maps| {
                        maps.clone()
                            .slice_dim(0, 0..image_count)
                            .slice_dim(1, (clip_len - 1)..clip_len)
                            .reshape([
                                image_count,
                                maps.shape().dims::<5>()[2],
                                maps.shape().dims::<5>()[3],
                                maps.shape().dims::<5>()[4],
                            ])
                    });
                    Some(VisionArtifactInput {
                        views: Some(views),
                        frames: None,
                        debug_recon_frames: None,
                        aux_frames: None,
                        patch_norms: None,
                        pca_rgb,
                        posterior_patch_norms_steps: None,
                        posterior_pca_rgb_steps: None,
                        patch_norms_steps: None,
                        pca_rgb_steps: None,
                        debug_patch_norms_steps: None,
                        debug_pca_rgb_steps: None,
                        probe_logits: forward
                            .probe_logits
                            .clone()
                            .map(|tensor| tensor.slice_dim(0, 0..image_count)),
                        labels: forward
                            .probe_labels
                            .clone()
                            .map(|tensor| tensor.slice_dim(0, 0..image_count)),
                        legend: Some(vec![
                            "reference_last".to_string(),
                            "decoded_spatiotemporal_latent_last".to_string(),
                            "state_pca_rgb_last".to_string(),
                        ]),
                        sidecar_json: None,
                        artifact_scale: self.config.artifact_upscale.max(1),
                        prediction_start: None,
                    })
                } else {
                    Some(VisionArtifactInput {
                        views: None,
                        frames: Some(forward.clip_frames.clone().slice_dim(0, 0..image_count)),
                        debug_recon_frames: debug_recon.clip.clone(),
                        aux_frames: None,
                        patch_norms: None,
                        pca_rgb: None,
                        posterior_patch_norms_steps: None,
                        posterior_pca_rgb_steps,
                        patch_norms_steps: None,
                        pca_rgb_steps: reference_pca_rgb_steps,
                        debug_patch_norms_steps: None,
                        debug_pca_rgb_steps: debug_recon.pca_rgb_steps.clone(),
                        probe_logits: forward
                            .probe_logits
                            .clone()
                            .map(|tensor| tensor.slice_dim(0, 0..image_count)),
                        labels: forward
                            .probe_labels
                            .clone()
                            .map(|tensor| tensor.slice_dim(0, 0..image_count)),
                        legend: Some(vec![
                            "reference_frame".to_string(),
                            "posterior_context_state_pca_rgb".to_string(),
                            "state_pca_rgb".to_string(),
                            "decoded_spatiotemporal_latent".to_string(),
                            "reencoded_decoded_latent_pca_rgb".to_string(),
                        ]),
                        sidecar_json: None,
                        artifact_scale: self.config.artifact_upscale.max(1),
                        prediction_start: Some(forward.context_len),
                    })
                }
            }
        } else {
            None
        };

        VisionVideoLejepaLosses {
            total,
            inv: prediction,
            observe,
            mode_separation_ratio,
            sigreg,
            recon: debug_recon.loss,
            recon_psnr_masked: debug_recon.psnr_short,
            recon_psnr_full: debug_recon.psnr_full,
            probe_loss,
            probe_acc,
            rollout_inv_to_horizon,
            rollout_state_norm_ratio_to_horizon,
            rollout_state_motion_to_horizon,
            rollout_com_error_to_h24,
            rollout_velocity_error_to_h24,
            long_rollout_com_error_to_h24,
            long_rollout_velocity_error_to_h24,
            artifacts,
        }
    }
}

impl<B: BackendTrait> Module<B> for VisionVideoLejepaModel<B> {
    type Record = VisionVideoLejepaModelRecord<B>;

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        let devices = Module::collect_devices(&self.frame_model, devices);
        let devices = Module::collect_devices(&self.temporal_model, devices);
        let devices = Module::collect_devices(&self.predictor, devices);
        let devices = Module::collect_devices(&self.patch_conditioner, devices);
        let devices = Module::collect_devices(&self.observation_merger, devices);
        let devices = Module::collect_devices(&self.step_mode_embeddings, devices);
        let devices = Module::collect_devices(&self.probe, devices);
        let devices = Module::collect_devices(&self.recon, devices);
        let devices = Module::collect_devices(&self.probe_loss, devices);
        let devices = Module::collect_devices(&self.future_queries, devices);
        let devices = Module::<B>::collect_devices(&self.config, devices);
        Module::collect_devices(&self.teacher_frame_model, devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            frame_model: Module::fork(self.frame_model, device),
            temporal_model: Module::fork(self.temporal_model, device),
            predictor: Module::fork(self.predictor, device),
            patch_conditioner: Module::fork(self.patch_conditioner, device),
            observation_merger: Module::fork(self.observation_merger, device),
            step_mode_embeddings: Module::fork(self.step_mode_embeddings, device),
            probe: Module::fork(self.probe, device),
            recon: Module::fork(self.recon, device),
            probe_loss: Module::fork(self.probe_loss, device),
            future_queries: Module::fork(self.future_queries, device),
            config: Module::<B>::fork(self.config, device),
            teacher_frame_model: Module::fork(self.teacher_frame_model, device),
            rollout: self.rollout,
            embed_dim: self.embed_dim,
            projection_dim: self.projection_dim,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            frame_model: Module::to_device(self.frame_model, device),
            temporal_model: Module::to_device(self.temporal_model, device),
            predictor: Module::to_device(self.predictor, device),
            patch_conditioner: Module::to_device(self.patch_conditioner, device),
            observation_merger: Module::to_device(self.observation_merger, device),
            step_mode_embeddings: Module::to_device(self.step_mode_embeddings, device),
            probe: Module::to_device(self.probe, device),
            recon: Module::to_device(self.recon, device),
            probe_loss: Module::to_device(self.probe_loss, device),
            future_queries: Module::to_device(self.future_queries, device),
            config: Module::<B>::to_device(self.config, device),
            teacher_frame_model: Module::to_device(self.teacher_frame_model, device),
            rollout: self.rollout,
            embed_dim: self.embed_dim,
            projection_dim: self.projection_dim,
        }
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        Module::visit(&self.frame_model, visitor);
        Module::visit(&self.temporal_model, visitor);
        Module::visit(&self.predictor, visitor);
        Module::visit(&self.patch_conditioner, visitor);
        Module::visit(&self.observation_merger, visitor);
        Module::visit(&self.step_mode_embeddings, visitor);
        Module::visit(&self.probe, visitor);
        Module::visit(&self.recon, visitor);
        Module::visit(&self.probe_loss, visitor);
        Module::visit(&self.future_queries, visitor);
        Module::visit(&self.config, visitor);
    }

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            frame_model: Module::map(self.frame_model, mapper),
            temporal_model: Module::map(self.temporal_model, mapper),
            predictor: Module::map(self.predictor, mapper),
            patch_conditioner: Module::map(self.patch_conditioner, mapper),
            observation_merger: Module::map(self.observation_merger, mapper),
            step_mode_embeddings: Module::map(self.step_mode_embeddings, mapper),
            probe: Module::map(self.probe, mapper),
            recon: Module::map(self.recon, mapper),
            probe_loss: Module::map(self.probe_loss, mapper),
            future_queries: Module::map(self.future_queries, mapper),
            config: Module::<B>::map(self.config, mapper),
            teacher_frame_model: self.teacher_frame_model,
            rollout: self.rollout,
            embed_dim: self.embed_dim,
            projection_dim: self.projection_dim,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            frame_model: Module::load_record(self.frame_model, record.frame_model),
            temporal_model: Module::load_record(self.temporal_model, record.temporal_model),
            predictor: Module::load_record(self.predictor, record.predictor),
            patch_conditioner: Module::load_record(
                self.patch_conditioner,
                record.patch_conditioner,
            ),
            observation_merger: Module::load_record(
                self.observation_merger,
                record.observation_merger,
            ),
            step_mode_embeddings: Module::load_record(
                self.step_mode_embeddings,
                record.step_mode_embeddings,
            ),
            probe: Module::load_record(self.probe, record.probe),
            recon: Module::load_record(self.recon, record.recon),
            probe_loss: Module::load_record(self.probe_loss, record.probe_loss),
            future_queries: Module::load_record(self.future_queries, record.future_queries),
            config: {
                let _: () = record.config;
                Module::<B>::load_record(self.config, ())
            },
            teacher_frame_model: None,
            rollout: self.rollout,
            embed_dim: self.embed_dim,
            projection_dim: self.projection_dim,
        }
        .restore_teacher_from_student()
    }

    fn into_record(self) -> Self::Record {
        VisionVideoLejepaModelRecord {
            frame_model: Module::into_record(self.frame_model),
            temporal_model: Module::into_record(self.temporal_model),
            predictor: Module::into_record(self.predictor),
            patch_conditioner: Module::into_record(self.patch_conditioner),
            observation_merger: Module::into_record(self.observation_merger),
            step_mode_embeddings: Module::into_record(self.step_mode_embeddings),
            probe: Module::into_record(self.probe),
            recon: Module::into_record(self.recon),
            probe_loss: Module::into_record(self.probe_loss),
            future_queries: Module::into_record(self.future_queries),
            config: Module::<B>::into_record(self.config),
        }
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoLejepaModel<B> {
    type InnerModule = VisionVideoLejepaModel<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        VisionVideoLejepaModel {
            frame_model: AutodiffModule::valid(&self.frame_model),
            temporal_model: AutodiffModule::valid(&self.temporal_model),
            predictor: AutodiffModule::valid(&self.predictor),
            patch_conditioner: AutodiffModule::valid(&self.patch_conditioner),
            observation_merger: AutodiffModule::valid(&self.observation_merger),
            step_mode_embeddings: AutodiffModule::valid(&self.step_mode_embeddings),
            probe: AutodiffModule::valid(&self.probe),
            recon: AutodiffModule::valid(&self.recon),
            probe_loss: AutodiffModule::valid(&self.probe_loss),
            future_queries: AutodiffModule::valid(&self.future_queries),
            config: AutodiffModule::<B>::valid(&self.config),
            teacher_frame_model: AutodiffModule::valid(&self.teacher_frame_model),
            rollout: self.rollout,
            embed_dim: self.embed_dim,
            projection_dim: self.projection_dim,
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        VisionVideoLejepaModel {
            frame_model: AutodiffModule::from_inner(module.frame_model),
            temporal_model: AutodiffModule::from_inner(module.temporal_model),
            predictor: AutodiffModule::from_inner(module.predictor),
            patch_conditioner: AutodiffModule::from_inner(module.patch_conditioner),
            observation_merger: AutodiffModule::from_inner(module.observation_merger),
            step_mode_embeddings: AutodiffModule::from_inner(module.step_mode_embeddings),
            probe: AutodiffModule::from_inner(module.probe),
            recon: AutodiffModule::from_inner(module.recon),
            probe_loss: AutodiffModule::from_inner(module.probe_loss),
            future_queries: AutodiffModule::from_inner(module.future_queries),
            config: AutodiffModule::<B>::from_inner(module.config),
            teacher_frame_model: AutodiffModule::from_inner(module.teacher_frame_model),
            rollout: module.rollout,
            embed_dim: module.embed_dim,
            projection_dim: module.projection_dim,
        }
    }
}

impl<B: BackendTrait> core::fmt::Display for VisionVideoLejepaModel<B> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(&burn::module::ModuleDisplay::format(
            self,
            burn::module::DisplaySettings::default(),
        ))
    }
}

impl<B: BackendTrait> ModuleDisplayDefault for VisionVideoLejepaModel<B> {
    fn content(&self, content: Content) -> Option<Content> {
        let content = content.add("frame_model", &self.frame_model);
        let content = if let Some(temporal_model) = &self.temporal_model {
            content.add("temporal_model", temporal_model)
        } else {
            content
        };
        content
            .add("predictor", &self.predictor)
            .add("patch_conditioner", &self.patch_conditioner)
            .add("observation_merger", &self.observation_merger)
            .add("step_mode_embeddings", &self.step_mode_embeddings)
            .add("recon", &self.recon)
            .add("config", &self.config)
            .optional()
    }

    fn num_params(&self) -> usize {
        Module::num_params(self)
    }
}

impl<B: BackendTrait> ModuleDisplay for VisionVideoLejepaModel<B> {}

struct GradientDetachVisitor<'a, B: AutodiffBackend> {
    grads: &'a mut GradientsParams,
    _phantom: core::marker::PhantomData<B>,
}

impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientDetachVisitor<'_, B> {
    fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
        let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) else {
            return;
        };
        self.grads
            .register::<B::InnerBackend, D>(param.id, grad.detach());
    }
}

fn detach_gradients_for_module<B, M>(module: &M, grads: &mut GradientsParams)
where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let mut visitor = GradientDetachVisitor::<B> {
        grads,
        _phantom: core::marker::PhantomData,
    };
    module.visit(&mut visitor);
}

impl<B: AutodiffBackend> TrainStep for VisionVideoLejepaModel<B> {
    type Input = VideoClipBatch<B>;
    type Output = VisionTrainItem<B>;

    fn step(&self, batch: VideoClipBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        crate::device::pin_stream_zero();
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = if self.uses_pyramid_backbone() {
            self.forward_losses_train_pyramid(batch, rollout_steps, backprop_steps, false, false)
        } else {
            self.forward_losses(batch, rollout_steps, backprop_steps, false, false, false)
        };
        let total_for_backprop = if let Some(weighted_probe) =
            self.weighted_loss_term(losses.probe_loss.clone(), self.config.loss.probe_weight)
        {
            losses.total.clone() + weighted_probe
        } else {
            losses.total.clone()
        };
        let mut grads = GradientsParams::from_grads(total_for_backprop.backward(), self);
        detach_gradients_for_module::<B, _>(self, &mut grads);
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());

        let item = VisionTrainItem::new(
            losses.total,
            losses.inv,
            losses.observe,
            losses.mode_separation_ratio,
            losses.sigreg,
            losses.recon,
            losses.recon_psnr_masked,
            losses.recon_psnr_full,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            losses.probe_loss,
            losses.probe_acc,
        )
        .with_rollout_horizon_metrics(
            losses.rollout_inv_to_horizon,
            losses.rollout_state_norm_ratio_to_horizon,
            losses.rollout_state_motion_to_horizon,
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

impl<B: BackendTrait> ValidStep for VisionVideoLejepaModel<B> {
    type Input = VideoClipBatch<B>;
    type Output = VisionOutput<B>;

    fn step(&self, mut batch: VideoClipBatch<B>) -> VisionOutput<B> {
        crate::device::pin_stream_zero();
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let compute_long_rollout_metrics =
            self.config.artifact_future_frames > self.config.max_supervised_target_frames();
        let subset_for_long_or_artifacts = compute_long_rollout_metrics
            || (batch.capture_artifacts
                && self.config.artifact_every > 0
                && self.config.artifact_max_images > 0);
        if subset_for_long_or_artifacts {
            batch = batch.take_prefix(self.config.artifact_max_images.max(1));
        }
        let capture_artifacts = batch.capture_artifacts
            && self.config.artifact_every > 0
            && self.config.artifact_max_images > 0;
        let losses = self.forward_losses(
            batch,
            self.rollout.max_steps,
            backprop_steps,
            capture_artifacts,
            compute_long_rollout_metrics,
            true,
        );
        let long_rollout_inv_to_horizon = if compute_long_rollout_metrics {
            losses.rollout_inv_to_horizon.clone().map(Some)
        } else {
            core::array::from_fn(|_| None)
        };
        let long_rollout_state_norm_ratio_to_horizon = if compute_long_rollout_metrics {
            losses.rollout_state_norm_ratio_to_horizon.clone().map(Some)
        } else {
            core::array::from_fn(|_| None)
        };
        let long_rollout_state_motion_to_horizon = if compute_long_rollout_metrics {
            losses.rollout_state_motion_to_horizon.clone().map(Some)
        } else {
            core::array::from_fn(|_| None)
        };
        let long_rollout_com_error_to_h24 = if compute_long_rollout_metrics {
            losses.long_rollout_com_error_to_h24.clone()
        } else {
            None
        };
        let long_rollout_velocity_error_to_h24 = if compute_long_rollout_metrics {
            losses.long_rollout_velocity_error_to_h24.clone()
        } else {
            None
        };
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());

        VisionOutput::new(
            losses.total,
            losses.inv,
            losses.observe,
            losses.mode_separation_ratio,
            losses.sigreg,
            losses.recon,
            losses.recon_psnr_masked,
            losses.recon_psnr_full,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            losses.probe_loss,
            losses.probe_acc,
            losses.artifacts,
        )
        .with_rollout_horizon_metrics(
            losses.rollout_inv_to_horizon,
            losses.rollout_state_norm_ratio_to_horizon,
            losses.rollout_state_motion_to_horizon,
        )
        .with_rollout_kinematics_metrics(
            losses.rollout_com_error_to_h24,
            losses.rollout_velocity_error_to_h24,
        )
        .with_long_rollout_horizon_metrics(
            long_rollout_inv_to_horizon,
            long_rollout_state_norm_ratio_to_horizon,
            long_rollout_state_motion_to_horizon,
        )
        .with_long_rollout_kinematics_metrics(
            long_rollout_com_error_to_h24,
            long_rollout_velocity_error_to_h24,
        )
    }
}

fn build_temporal_config(
    embed_dim: usize,
    temporal: &VisionVideoTemporalConfig,
    normalization: &burn_dragon_core::DragonNormConfig,
) -> BDHConfig {
    let mut config = BDHConfig {
        n_layer: temporal.n_layer,
        n_embd: embed_dim.max(1),
        dropout: 0.0,
        n_head: temporal.n_head.max(1),
        mlp_internal_dim_multiplier: temporal.mlp_internal_dim_multiplier.max(1),
        vocab_size: 1,
        fused_kernels: FusedKernelConfig {
            enabled: temporal.fused,
            wgpu_recurrent_kernel: temporal.wgpu_recurrent_kernel,
            wgpu_rollout_fused: temporal.wgpu_rollout_fused,
            ..FusedKernelConfig::default()
        },
        normalization: normalization.clone(),
        ..BDHConfig::default()
    };
    config
        .fused_kernels
        .set_block_sizes(temporal.latent_block_size, temporal.time_block_size);
    config
        .fused_kernels
        .set_rotary_embedding(RotaryEmbedding::Alibi);
    config.set_rollout_fast_steps_per_slow_step(temporal.rollout_fast_steps_per_slow_step);
    config
}

fn supervised_time_mask<B: BackendTrait>(
    batch: usize,
    time: usize,
    target_len: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    if batch == 0 || time == 0 {
        return Tensor::<B, 3>::zeros([batch, time, 1], device);
    }
    Tensor::<B, 1, Int>::arange(0..time as i64, device)
        .float()
        .lower_equal_elem(target_len.saturating_sub(1) as f32)
        .float()
        .reshape([1, time, 1])
        .repeat_dim(0, batch)
}

fn temporal_mse_loss_masked<B: BackendTrait>(
    predicted: Tensor<B, 3>,
    target: Tensor<B, 3>,
    time_mask: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, time, dim] = predicted.shape().dims::<3>();
    if batch == 0 || time == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &predicted.device());
    }
    let squared_error = (predicted - target).powf_scalar(2.0);
    let numerator = (squared_error * time_mask.clone()).sum();
    let denominator = time_mask
        .sum()
        .mul_scalar(dim as f32)
        .add_scalar(LEJEPA_EPS);
    numerator / denominator
}

fn temporal_cosine_loss_masked<B: BackendTrait>(
    predicted: Tensor<B, 3>,
    target: Tensor<B, 3>,
    time_mask: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, time, dim] = predicted.shape().dims::<3>();
    if batch == 0 || time == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &predicted.device());
    }
    let predicted_norm = predicted.clone()
        / predicted
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .reshape([batch, time, 1])
            .add_scalar(LEJEPA_EPS);
    let target_norm = target.clone()
        / target
            .clone()
            .powf_scalar(2.0)
            .sum_dim(2)
            .sqrt()
            .reshape([batch, time, 1])
            .add_scalar(LEJEPA_EPS);
    let cosine_similarity = (predicted_norm * target_norm)
        .sum_dim(2)
        .reshape([batch, time, 1]);
    let cosine_distance =
        Tensor::<B, 3>::ones([batch, time, 1], &predicted.device()) - cosine_similarity;
    let numerator = (cosine_distance * time_mask.clone()).sum();
    let denominator = time_mask.sum().add_scalar(LEJEPA_EPS);
    numerator / denominator
}

fn temporal_cosine_loss<B: BackendTrait>(
    predicted: Tensor<B, 3>,
    target: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let [batch, time, dim] = predicted.shape().dims::<3>();
    if batch == 0 || time == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &predicted.device());
    }
    let device = predicted.device();
    temporal_cosine_loss_masked(
        predicted,
        target,
        Tensor::<B, 3>::ones([batch, time, 1], &device),
    )
}
