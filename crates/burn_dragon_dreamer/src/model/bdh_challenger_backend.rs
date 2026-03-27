use super::{
    DragonDreamer, DreamerDebugOutput, DreamerForward, edge_mse_loss_frame, extract_crops,
    fixation_tensor_from_traces, motion_mse_loss_frame, mse_loss, mse_loss_tokens,
    occupancy_loss_frame, rollout_reconstruction_loss_frame, summarize_fixation_set,
    tokenizer_reconstruction_loss_frame,
};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn_autogaze::FrameFixationTrace;

impl<B: Backend> DragonDreamer<B> {
    pub(super) fn slots_from_summary(&self, summary: Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, _] = summary.shape().dims::<2>();
        let hidden =
            burn::tensor::activation::gelu(self.slot_state_from_summary_hidden.forward(summary));
        burn::tensor::activation::gelu(self.slot_state_from_summary_out.forward(hidden)).reshape([
            batch,
            self.slot_count,
            self.latent_dim,
        ])
    }

    pub(super) fn fuse_summary_slots_with_observation(
        &self,
        prior_slots: Tensor<B, 3>,
        summary_slots: Tensor<B, 3>,
        observed_slots: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, slot_count, _] = summary_slots.shape().dims::<3>();
        let gate = burn::tensor::activation::sigmoid(
            self.slot_state_mix_gate.forward(
                Tensor::cat(
                    vec![
                        prior_slots.clone(),
                        summary_slots.clone(),
                        observed_slots.clone(),
                    ],
                    2,
                )
                .reshape([batch * slot_count, self.latent_dim * 3]),
            ),
        )
        .reshape([batch, slot_count, self.latent_dim]);
        summary_slots.clone() + gate * (observed_slots - summary_slots)
    }

    pub(super) fn forward_internal_bdh_challenger(
        &self,
        clip_frames: Tensor<B, 5>,
        traces: &[FrameFixationTrace],
        teacher_features: Tensor<B, 3>,
        crop_teacher_features: Tensor<B, 3>,
        context_len: usize,
        target_len: usize,
        capture_debug: bool,
    ) -> (DreamerForward<B>, Option<DreamerDebugOutput<B>>) {
        let device = clip_frames.device();
        let [batch, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
        let context_len = context_len.clamp(1, clip_len.saturating_sub(1).max(1));
        let target_len = target_len.clamp(1, clip_len.saturating_sub(context_len).max(1));
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let mut summary = Tensor::<B, 2>::zeros([batch, self.latent_dim], &device);
        let mut slots = self.slots_from_summary(summary.clone());
        let mut current_total = zero.clone();
        let mut future_total = zero.clone();
        let mut prior_total = zero.clone();
        let mut prior_terms = 0usize;
        let mut gaze_total = zero.clone();
        let mut query_total = zero.clone();
        let mut tokenizer_recon_total = zero.clone();
        let mut slot_align_total = zero.clone();
        let mut recon_current_total = zero.clone();
        let mut recon_future_total = zero.clone();
        let mut recon_edge_total = zero.clone();
        let mut recon_motion_total = zero.clone();
        let mut bdh_state = Some(self.bdh.init_state());
        let mut context_reference_frames = Vec::new();
        let mut context_reconstruction_frames = Vec::new();
        let mut future_reference_frames = Vec::new();
        let mut future_reconstruction_frames = Vec::new();
        let mut context_latents = Vec::new();
        let mut future_latents = Vec::new();
        let mut teacher_fixations = Vec::new();
        let mut predicted_fixations = Vec::new();
        let mut previous_real_frame: Option<Tensor<B, 4>> = None;
        let mut previous_target_frame: Option<Tensor<B, 4>> = None;

        for step in 0..context_len {
            let frame = clip_frames
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, channels, height, width]);
            let peripheral = self.encode_peripheral(frame.clone());
            let predicted_fixation =
                self.predict_fixation_from_slots(slots.clone(), Some(peripheral.clone()));
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, step, self.k_fovea, &device);
            let fixation_points = trace_points
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let fixation_stop = trace_points
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let fixation_summary =
                summarize_fixation_set(fixation_points.clone(), fixation_stop.clone());
            gaze_total = gaze_total + mse_loss(predicted_fixation.clone(), trace_points.clone());

            let crops = extract_crops(frame.clone(), traces, step, self.crop_size, self.k_fovea);
            let fovea_tokens = self.encode_fovea_tokens(crops);
            let observed_slots = self.encode_frame_to_slot_tokens(frame.clone());
            let prior_summary = self.prior_step(summary.clone());
            let prior_slots = self.prior_step_slots(slots.clone());
            let (tokenizer_recon, tokenizer_occupancy) =
                self.decode_frame_from_slots_components(observed_slots.clone(), None);
            let slot_posterior = self.posterior_step_slots(
                prior_slots.clone(),
                observed_slots.clone(),
                peripheral,
                fovea_tokens,
                fixation_points,
                fixation_summary.clone(),
            );
            let slot_summary = self.summarize_slots(slot_posterior.clone());
            summary = self.posterior_step(prior_summary, slot_summary, bdh_state.as_mut());
            let summary_slots = self.slots_from_summary(summary.clone());
            slots = self.fuse_summary_slots_with_observation(
                prior_slots.clone(),
                summary_slots,
                slot_posterior.clone(),
            );
            let (current_recon, current_occupancy) =
                self.decode_frame_from_slots_components(slots.clone(), None);
            let current_pred = self.current_head.forward(summary.clone());
            let current_target = teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            current_total = current_total + mse_loss(current_pred, current_target);
            prior_total =
                prior_total + mse_loss_tokens(prior_slots, slot_posterior.clone().detach());
            prior_terms += 1;
            slot_align_total =
                slot_align_total + mse_loss_tokens(slot_posterior.clone(), observed_slots.detach());
            tokenizer_recon_total = tokenizer_recon_total
                + tokenizer_reconstruction_loss_frame(tokenizer_recon, frame.clone().detach())
                + occupancy_loss_frame(tokenizer_occupancy, frame.clone().detach())
                    .mul_scalar(0.35);
            recon_current_total = recon_current_total
                + rollout_reconstruction_loss_frame(current_recon.clone(), frame.clone().detach())
                + occupancy_loss_frame(current_occupancy, frame.clone().detach()).mul_scalar(0.20);
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(current_recon.clone(), frame.clone().detach());
            if let Some(previous_target) = previous_target_frame.clone() {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        current_recon.clone(),
                        previous_target.clone(),
                        frame.clone().detach(),
                        previous_target,
                    );
            }
            let query_pred = self
                .query_head
                .forward(Tensor::cat(vec![summary.clone(), fixation_summary], 1));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, step..step + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);

            if capture_debug {
                context_reference_frames.push(frame.clone().detach().unsqueeze_dim::<5>(1));
                context_reconstruction_frames.push(current_recon.detach().unsqueeze_dim::<5>(1));
                context_latents.push(summary.clone().detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.clone().detach().unsqueeze_dim::<3>(1));
                predicted_fixations.push(predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            let detached_frame = frame.detach();
            previous_real_frame = Some(detached_frame.clone());
            previous_target_frame = Some(detached_frame);
        }

        let mut rollout_summary = summary;
        let mut rollout_slots = slots;
        let mut previous_rollout_frame = previous_real_frame;
        for step in 0..target_len {
            let prior_summary = self.prior_step(rollout_summary.clone());
            let prior_slots = self.prior_step_slots(rollout_slots.clone());
            let summary_slots = self.slots_from_summary(prior_summary.clone());
            rollout_slots = self.fuse_summary_slots_with_observation(
                prior_slots.clone(),
                summary_slots,
                prior_slots.clone(),
            );
            rollout_summary = self.summarize_slots(rollout_slots.clone());
            let (future_recon, future_occupancy) =
                self.decode_frame_from_slots_components(rollout_slots.clone(), None);
            let future_pred = self.future_head.forward(rollout_summary.clone());
            let target_idx = context_len + step;
            let future_target = teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.teacher_dim])
                .detach();
            future_total = future_total + mse_loss(future_pred, future_target);
            let future_frame = clip_frames
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, channels, height, width]);
            let future_slot_target =
                self.encode_frame_to_slot_tokens(future_frame.clone().detach());
            let (future_tokenizer_recon, future_tokenizer_occupancy) =
                self.decode_frame_from_slots_components(future_slot_target.clone(), None);
            prior_total = prior_total + mse_loss_tokens(prior_slots, future_slot_target.detach());
            prior_terms += 1;
            tokenizer_recon_total = tokenizer_recon_total
                + tokenizer_reconstruction_loss_frame(
                    future_tokenizer_recon,
                    future_frame.clone().detach(),
                )
                + occupancy_loss_frame(future_tokenizer_occupancy, future_frame.clone().detach())
                    .mul_scalar(0.35);
            recon_future_total = recon_future_total
                + rollout_reconstruction_loss_frame(
                    future_recon.clone(),
                    future_frame.clone().detach(),
                )
                + occupancy_loss_frame(future_occupancy, future_frame.clone().detach())
                    .mul_scalar(0.20);
            recon_edge_total = recon_edge_total
                + edge_mse_loss_frame(future_recon.clone(), future_frame.clone().detach());
            if let (Some(previous_rollout), Some(previous_target)) = (
                previous_rollout_frame.clone(),
                previous_target_frame.clone(),
            ) {
                recon_motion_total = recon_motion_total
                    + motion_mse_loss_frame(
                        future_recon.clone(),
                        previous_rollout,
                        future_frame.clone().detach(),
                        previous_target,
                    );
            }
            let trace_points =
                fixation_tensor_from_traces::<B>(traces, target_idx, self.k_fovea, &device);
            let imagined_predicted_fixation =
                self.predict_fixation_from_slots(rollout_slots.clone(), None);
            let imagined_fixation_points = imagined_predicted_fixation
                .clone()
                .slice_dim(1, 0..self.k_fovea * 4)
                .reshape([batch, self.k_fovea, 4]);
            let imagined_fixation_stop = imagined_predicted_fixation
                .clone()
                .slice_dim(1, self.k_fovea * 4..self.k_fovea * 4 + 1);
            let imagined_fixation_summary =
                summarize_fixation_set(imagined_fixation_points, imagined_fixation_stop);
            gaze_total =
                gaze_total + mse_loss(imagined_predicted_fixation.clone(), trace_points.clone());
            let query_pred = self.query_head.forward(Tensor::cat(
                vec![rollout_summary.clone(), imagined_fixation_summary],
                1,
            ));
            let query_target = crop_teacher_features
                .clone()
                .slice_dim(1, target_idx..target_idx + 1)
                .reshape([batch, self.crop_teacher_dim])
                .detach();
            query_total = query_total + mse_loss(query_pred, query_target);
            if capture_debug {
                future_reference_frames.push(future_frame.clone().detach().unsqueeze_dim::<5>(1));
                future_reconstruction_frames
                    .push(future_recon.clone().detach().unsqueeze_dim::<5>(1));
                future_latents.push(rollout_summary.clone().detach().unsqueeze_dim::<3>(1));
                teacher_fixations.push(trace_points.detach().unsqueeze_dim::<3>(1));
                predicted_fixations
                    .push(imagined_predicted_fixation.detach().unsqueeze_dim::<3>(1));
            }
            previous_rollout_frame = Some(future_recon);
            previous_target_frame = Some(future_frame.clone().detach());
        }

        let context_denom = context_len as f32;
        let target_denom = target_len as f32;
        let fixation_denom = (context_len + target_len) as f32;
        let current = current_total.div_scalar(context_denom);
        let future = future_total.div_scalar(target_denom);
        let prior = prior_total.div_scalar(prior_terms.max(1) as f32);
        let gaze = gaze_total.div_scalar(fixation_denom);
        let query = query_total.div_scalar(fixation_denom);
        let tokenizer_recon = tokenizer_recon_total.div_scalar((context_len + target_len) as f32);
        let slot_align = slot_align_total.div_scalar(context_denom);
        let tokenizer = tokenizer_recon.clone()
            + slot_align
                .clone()
                .mul_scalar(self.tokenizer_slot_align_weight);
        let recon_current = recon_current_total.div_scalar(context_denom);
        let recon_future = recon_future_total.div_scalar(target_denom);
        let recon_edge = recon_edge_total.div_scalar((context_len + target_len) as f32);
        let recon_motion = recon_motion_total
            .div_scalar((context_len + target_len).saturating_sub(1).max(1) as f32);
        let recon = recon_current.clone().mul_scalar(self.recon_current_weight)
            + recon_future.clone().mul_scalar(self.recon_future_weight)
            + recon_edge.clone().mul_scalar(self.recon_edge_weight);
        let recon = recon + recon_motion.clone().mul_scalar(self.recon_motion_weight);
        let total = current.clone().mul_scalar(self.current_loss_weight)
            + future.clone().mul_scalar(self.future_loss_weight)
            + prior.clone().mul_scalar(self.prior_loss_weight)
            + gaze.clone().mul_scalar(self.gaze_loss_weight)
            + query.clone().mul_scalar(self.query_loss_weight)
            + tokenizer.clone().mul_scalar(self.tokenizer_loss_weight)
            + recon.clone().mul_scalar(self.recon_loss_weight);

        let forward = DreamerForward {
            total,
            current,
            future,
            prior,
            gaze,
            query,
            recon,
            tokenizer,
            tokenizer_recon,
            slot_align,
            recon_current,
            recon_future,
            recon_edge,
            recon_motion,
        };
        let debug = if capture_debug {
            Some(DreamerDebugOutput {
                context_reference_frames: Tensor::cat(context_reference_frames, 1),
                context_reconstruction_frames: Tensor::cat(context_reconstruction_frames, 1),
                future_reference_frames: Tensor::cat(future_reference_frames, 1),
                future_reconstruction_frames: Tensor::cat(future_reconstruction_frames, 1),
                context_latents: Tensor::cat(context_latents, 1),
                future_latents: Tensor::cat(future_latents, 1),
                teacher_fixations: Tensor::cat(teacher_fixations, 1),
                predicted_fixations: Tensor::cat(predicted_fixations, 1),
            })
        } else {
            None
        };
        (forward, debug)
    }
}
