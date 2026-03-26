use super::dynamics::embed_clip_frames_raw_with_model;
use crate::model::VisionRolloutState;
use crate::train::prelude::*;

mod masking;
mod predictor;
#[cfg(test)]
mod tests;

use masking::sample_mask_batch;
use predictor::VisionVideoVjepa21Predictor;

const VJEPA21_EPS: f32 = 1.0e-6;

#[derive(Module, Debug)]
pub(crate) struct VisionVideoVjepa21Model<B: BackendTrait> {
    pub(crate) frame_model: VisionDragon<B>,
    predictor: VisionVideoVjepa21Predictor<B>,
    probe: VisionProbe<B>,
    recon: VisionReconstructionHead<B>,
    probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    input_mask_token: Param<Tensor<B, 2>>,
    predictor_mask_bias: Option<Param<Tensor<B, 2>>>,
    pub(crate) teacher_frame_model: Option<VisionDragon<B>>,
    pub(crate) config: VisionVideoLejepaConfig,
}

#[derive(Clone)]
struct VisionVideoVjepa21Losses<B: BackendTrait> {
    total: Tensor<B, 1>,
    masked: Tensor<B, 1>,
    context: Tensor<B, 1>,
    recon: Tensor<B, 1>,
    recon_psnr_masked: Tensor<B, 1>,
    recon_psnr_full: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

struct VisionVideoVjepa21DebugRecon<B: BackendTrait> {
    loss: Tensor<B, 1>,
    psnr_masked: Tensor<B, 1>,
    psnr_full: Tensor<B, 1>,
    clip: Option<Tensor<B, 5>>,
}

impl<B: BackendTrait> VisionVideoVjepa21Model<B> {
    pub(crate) fn new(
        frame_model: VisionDragon<B>,
        config: VisionVideoLejepaConfig,
        vision: &VisionDragonConfig,
        num_classes: usize,
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
        let probe = VisionProbe::new(projection_dim, num_classes, &vision.normalization, device);
        let patch_size = vision.patch_size.max(1);
        let patch_dim = vision.in_channels.max(1) * patch_size * patch_size;
        let recon_hidden_dim = if config.loss.debug_recon_hidden_dim == 0 {
            vision
                .projection_hidden_dim
                .max(projection_dim * 2)
                .max(256)
        } else {
            config.loss.debug_recon_hidden_dim.max(1)
        };
        let recon = VisionReconstructionHead::new(
            projection_dim,
            recon_hidden_dim,
            patch_dim,
            true,
            &vision.normalization,
            device,
        );
        let probe_loss = CrossEntropyLossConfig::new().init(device);
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
            probe,
            recon,
            probe_loss,
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

    fn reconstruct_patch_values_from_patch_tokens(
        &self,
        patch_tokens: Tensor<B, 4>,
        channels: usize,
    ) -> Tensor<B, 4> {
        let [batch, frames, tokens, _] = patch_tokens.shape().dims::<4>();
        let patch_size = self.frame_model.patch_size().max(1);
        let patch_dim = channels.max(1) * patch_size * patch_size;
        self.recon
            .forward(patch_tokens)
            .reshape([batch, frames, tokens, patch_dim])
    }

    fn reconstruct_frames_from_patch_values(
        &self,
        patch_values: Tensor<B, 4>,
        height: usize,
        width: usize,
        channels: usize,
    ) -> Tensor<B, 5> {
        let [batch, frames, tokens, patch_dim] = patch_values.shape().dims::<4>();
        let patch_size = self.frame_model.patch_size().max(1);
        let patches = patch_values.reshape([batch * frames, tokens, patch_dim]);
        let images = unpatchify(patches, patch_size, height, width, channels);
        images.reshape([batch, frames, channels, height, width])
    }

    fn build_probe_summary(&self, patch_tokens: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, _frame_count, _patch_count, dim] = patch_tokens.shape().dims::<4>();
        patch_tokens.mean_dim(2).mean_dim(1).reshape([batch, dim])
    }

    fn debug_reconstruction_outputs(
        &self,
        clip_frames: Tensor<B, 5>,
        masks: &masking::Vjepa21MaskBatch<B>,
        teacher_levels: &[Tensor<B, 4>],
        predicted_levels: &[Tensor<B, 4>],
        capture_artifacts: bool,
        image_count: usize,
    ) -> VisionVideoVjepa21DebugRecon<B> {
        let device = clip_frames.device();
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let [batch, frame_count, channels, height, width] = clip_frames.shape().dims::<5>();
        if batch == 0 || frame_count == 0 || channels == 0 || height == 0 || width == 0 {
            return VisionVideoVjepa21DebugRecon {
                loss: zero.clone(),
                psnr_masked: zero.clone(),
                psnr_full: zero,
                clip: None,
            };
        }

        let patch_size = self.frame_model.patch_size().max(1);
        let target_patches = patchify(
            clip_frames
                .clone()
                .detach()
                .reshape([batch * frame_count, channels, height, width]),
            patch_size,
        );
        let [_, patch_count, patch_dim] = target_patches.shape().dims::<3>();
        let target_patches = target_patches.reshape([batch, frame_count, patch_count, patch_dim]);

        let teacher_patch_values = self.reconstruct_patch_values_from_patch_tokens(
            teacher_levels
                .last()
                .cloned()
                .expect("teacher levels populated")
                .detach(),
            channels,
        );
        let predicted_patch_values = self.reconstruct_patch_values_from_patch_tokens(
            predicted_levels
                .last()
                .cloned()
                .expect("predicted levels populated")
                .detach(),
            channels,
        );

        let teacher_full_mse = masked_patch_mse(
            teacher_patch_values,
            target_patches.clone(),
            None,
            patch_dim,
        );
        let predicted_full_mse = masked_patch_mse(
            predicted_patch_values.clone(),
            target_patches.clone(),
            None,
            patch_dim,
        );
        let predicted_masked_mse = masked_patch_mse(
            predicted_patch_values.clone(),
            target_patches,
            Some(masks.target.clone()),
            patch_dim,
        );
        let loss = (teacher_full_mse + predicted_full_mse.clone()).div_scalar(2.0);
        let clip = if capture_artifacts {
            let image_count = image_count.min(batch);
            (image_count > 0).then(|| {
                self.reconstruct_frames_from_patch_values(
                    predicted_patch_values.slice_dim(0, 0..image_count),
                    height,
                    width,
                    channels,
                )
            })
        } else {
            None
        };

        VisionVideoVjepa21DebugRecon {
            loss,
            psnr_masked: recon_psnr(predicted_masked_mse),
            psnr_full: recon_psnr(predicted_full_mse),
            clip,
        }
    }

    fn forward_losses(&self, batch: VideoClipBatch<B>) -> VisionVideoVjepa21Losses<B> {
        let capture_artifacts = batch.capture_artifacts
            && self.config.artifact_every > 0
            && self.config.artifact_max_images > 0;
        let labels = batch.labels;
        let clip_frames = batch.clip_frames;
        let recon_frames = clip_frames.clone();
        let artifact_frames = capture_artifacts.then(|| recon_frames.clone());
        let raw_tokens = embed_clip_frames_raw_with_model(&self.frame_model, clip_frames);
        let [batch_size, clip_len, patch_count, embed_dim] = raw_tokens.shape().dims::<4>();
        let projection_dim = self.frame_model.projection_dim().max(1);

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
        let visible_flat = masks
            .visible
            .clone()
            .reshape([batch_size, clip_len * patch_count]);
        let target_flat = masks
            .target
            .clone()
            .reshape([batch_size, clip_len * patch_count]);
        let distance_flat = masks
            .context_distance
            .clone()
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
        let zero = Tensor::<B, 1>::zeros([1], &masks.visible_ratio.device());
        let debug_recon = if capture_artifacts || self.config.loss.debug_recon_weight > 0.0 {
            self.debug_reconstruction_outputs(
                recon_frames,
                &masks,
                &teacher_levels,
                &predicted_levels,
                capture_artifacts,
                self.config.artifact_max_images.max(1),
            )
        } else {
            VisionVideoVjepa21DebugRecon {
                loss: zero.clone(),
                psnr_masked: zero.clone(),
                psnr_full: zero.clone(),
                clip: None,
            }
        };
        let (probe_loss, probe_acc, probe_logits) = if self.config.loss.probe_weight > 0.0 {
            let summary = self.build_probe_summary(
                teacher_levels
                    .last()
                    .cloned()
                    .expect("teacher levels populated")
                    .detach(),
            );
            let logits = self.probe.forward(summary);
            let probe_loss = self.probe_loss.forward(logits.clone(), labels.clone());
            let probe_pred = logits
                .clone()
                .argmax(1)
                .reshape([labels.shape().dims::<1>()[0]]);
            let probe_acc = probe_pred.equal(labels.clone()).float().mean();
            (probe_loss, probe_acc, Some(logits))
        } else {
            (zero.clone(), zero.clone(), None)
        };
        let total = masked_loss
            .clone()
            .mul_scalar(self.config.vjepa21.loss.masked_weight)
            + context_loss
                .clone()
                .mul_scalar(self.config.vjepa21.loss.context_weight)
            + debug_recon
                .loss
                .clone()
                .mul_scalar(self.config.loss.debug_recon_weight.max(0.0))
            + probe_loss
                .clone()
                .mul_scalar(self.config.loss.probe_weight.max(0.0));
        let artifacts = if capture_artifacts {
            build_vjepa21_artifacts(
                artifact_frames.expect("artifact frames available"),
                &masks,
                &predicted_levels,
                debug_recon.clip.clone(),
                probe_logits,
                Some(labels.clone()),
                self.config.artifact_max_images.max(1),
                self.config.artifact_upscale.max(1),
            )
        } else {
            None
        };

        VisionVideoVjepa21Losses {
            total,
            masked: masked_loss,
            context: context_loss,
            recon: debug_recon.loss,
            recon_psnr_masked: debug_recon.psnr_masked,
            recon_psnr_full: debug_recon.psnr_full,
            probe_loss,
            probe_acc,
            artifacts,
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
            zero.clone(),
            zero.clone(),
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
        let capture_artifacts = batch.capture_artifacts
            && self.config.artifact_every > 0
            && self.config.artifact_max_images > 0;
        let artifact_batch = capture_artifacts.then(|| {
            batch
                .clone()
                .take_prefix(self.config.artifact_max_images.max(1))
        });
        let mut losses = if capture_artifacts {
            self.forward_losses(batch.clone().with_capture_artifacts(false))
        } else {
            self.forward_losses(batch)
        };
        if let Some(artifact_batch) = artifact_batch {
            let artifact_losses = self.forward_losses(artifact_batch);
            losses.artifacts = artifact_losses.artifacts;
        }
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            losses.masked,
            losses.context,
            zero.clone(),
            zero.clone(),
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

fn collect_vjepa21_feature_steps<B: BackendTrait>(
    patch_tokens: Tensor<B, 4>,
    image_count: usize,
    include_heatmaps: bool,
) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 5>>) {
    let [batch, frame_count, tokens, dim] = patch_tokens.shape().dims::<4>();
    let image_count = image_count.min(batch);
    if image_count == 0 || frame_count == 0 || tokens == 0 || dim == 0 {
        return (None, None);
    }

    let patch_tokens = patch_tokens.slice_dim(0, 0..image_count);
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

    (patch_norms_steps, pca_rgb_steps)
}

fn collect_mask_heatmaps<B: BackendTrait>(
    mask: Tensor<B, 3>,
    image_count: usize,
) -> Option<Tensor<B, 4>> {
    let [batch, frame_count, patch_count] = mask.shape().dims::<3>();
    let image_count = image_count.min(batch);
    if image_count == 0 || frame_count == 0 || patch_count == 0 {
        return None;
    }

    let flat =
        mask.slice_dim(0, 0..image_count)
            .reshape([image_count * frame_count, patch_count, 1]);
    patch_heatmap_or_norm(flat, image_count * frame_count).map(|maps| {
        let [_, grid_h, grid_w] = maps.shape().dims::<3>();
        maps.reshape([image_count, frame_count, grid_h, grid_w])
    })
}

fn build_vjepa21_artifacts<B: BackendTrait>(
    clip_frames: Tensor<B, 5>,
    masks: &masking::Vjepa21MaskBatch<B>,
    predicted_levels: &[Tensor<B, 4>],
    debug_recon_frames: Option<Tensor<B, 5>>,
    probe_logits: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1, Int>>,
    image_count: usize,
    artifact_scale: usize,
) -> Option<VisionArtifactInput<B>> {
    let image_count = image_count.min(clip_frames.shape().dims::<5>()[0]);
    if image_count == 0 {
        return None;
    }

    let mask_steps = collect_mask_heatmaps(masks.target.clone(), image_count);
    let (_, predicted_pca_steps) = collect_vjepa21_feature_steps(
        predicted_levels
            .last()
            .cloned()
            .expect("predicted levels populated"),
        image_count,
        false,
    );

    if mask_steps.is_none() && predicted_pca_steps.is_none() && debug_recon_frames.is_none() {
        return None;
    }

    let mut legend = vec!["reference_frame".to_string()];
    if mask_steps.is_some() {
        legend.push("target_mask".to_string());
    }
    if predicted_pca_steps.is_some() {
        legend.push("predictor_state_pca_rgb".to_string());
    }
    if debug_recon_frames.is_some() {
        legend.push("decoded_predictor_output".to_string());
    }

    Some(VisionArtifactInput {
        views: None,
        frames: Some(clip_frames.slice_dim(0, 0..image_count)),
        debug_recon_frames: debug_recon_frames.map(|frames| frames.slice_dim(0, 0..image_count)),
        aux_frames: None,
        patch_norms: None,
        pca_rgb: None,
        posterior_patch_norms_steps: mask_steps,
        posterior_pca_rgb_steps: None,
        patch_norms_steps: None,
        pca_rgb_steps: predicted_pca_steps,
        debug_patch_norms_steps: None,
        debug_pca_rgb_steps: None,
        probe_logits: probe_logits.map(|tensor| tensor.slice_dim(0, 0..image_count)),
        labels: labels.map(|tensor| tensor.slice_dim(0, 0..image_count)),
        legend: Some(legend),
        sidecar_json: None,
        artifact_scale,
        prediction_start: None,
    })
}

fn masked_patch_mse<B: BackendTrait>(
    predicted: Tensor<B, 4>,
    target: Tensor<B, 4>,
    mask: Option<Tensor<B, 3>>,
    patch_dim: usize,
) -> Tensor<B, 1> {
    let sq_error = (predicted - target).powf_scalar(2.0);
    if let Some(mask) = mask {
        let mask = mask.unsqueeze_dim::<4>(3);
        let denom = mask
            .clone()
            .sum()
            .mul_scalar(patch_dim.max(1) as f32)
            .add_scalar(VJEPA21_EPS);
        sq_error.mul(mask).sum().div(denom).reshape([1])
    } else {
        sq_error.mean().reshape([1])
    }
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
