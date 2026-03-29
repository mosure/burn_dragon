use crate::artifacts::{
    ArtifactMetrics, DreamerArtifactSnapshot, FixationSequence, SequenceTensor,
};
use crate::data::{CachedSequenceSplit, sample_indices};
use crate::runtime::{
    Backend as RuntimeBackend, TrainBackend, train_to_runtime_tensor3, train_to_runtime_tensor5,
};
use crate::{DragonDreamer, DreamerDebugOutput, DreamerForward, MovingMnistDreamerTrainConfig};
use burn::module::AutodiffModule;
use burn_autogaze::FrameFixationTrace;
use std::path::PathBuf;

#[derive(Clone, Debug)]
pub struct SelectedValidationMetrics {
    pub step: usize,
    pub total: f32,
    pub future: f32,
    pub recon_current: f32,
    pub recon_future: f32,
    pub future_psnr: f32,
    pub future_fg_iou: f32,
    pub future_motion_ratio: f32,
    pub future_fixation_teacher_l1: f32,
    pub quality_score: f32,
    pub checkpoint_base: Option<PathBuf>,
}

#[derive(Clone, Debug)]
pub struct MovingMnistDreamerRunSummary {
    pub initial_valid_total: f32,
    pub final_valid_total: f32,
    pub initial_valid_future: f32,
    pub final_valid_future: f32,
    pub best_valid_total: f32,
    pub best_valid_future: f32,
    pub final_train_total: f32,
    pub final_train_future: f32,
    pub final_valid_recon_current: f32,
    pub final_valid_recon_future: f32,
    pub final_valid_future_psnr: f32,
    pub final_valid_future_fg_iou: f32,
    pub final_valid_future_motion_ratio: f32,
    pub final_valid_future_stop_mean: f32,
    pub final_valid_future_stop_std: f32,
    pub final_valid_tokenizer: f32,
    pub final_valid_tokenizer_recon: f32,
    pub final_valid_slot_align: f32,
    pub best_future_selection: SelectedValidationMetrics,
    pub best_quality_selection: SelectedValidationMetrics,
    pub latent_backend: String,
    pub steps: usize,
    pub run_dir: Option<PathBuf>,
    pub artifact_dir: Option<PathBuf>,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct LossSnapshot {
    pub(crate) total: f32,
    pub(crate) current: f32,
    pub(crate) future: f32,
    pub(crate) prior: f32,
    pub(crate) gaze: f32,
    pub(crate) query: f32,
    pub(crate) recon: f32,
    pub(crate) tokenizer: f32,
    pub(crate) tokenizer_recon: f32,
    pub(crate) slot_align: f32,
    pub(crate) recon_current: f32,
    pub(crate) recon_future: f32,
    pub(crate) recon_edge: f32,
    pub(crate) recon_motion: f32,
    pub(crate) current_mae: f32,
    pub(crate) future_mae: f32,
    pub(crate) current_psnr: f32,
    pub(crate) future_psnr: f32,
    pub(crate) current_fg_iou: f32,
    pub(crate) future_fg_iou: f32,
    pub(crate) future_frame_std: f32,
    pub(crate) future_latent_std: f32,
    pub(crate) future_motion_mse: f32,
    pub(crate) future_ref_motion_mse: f32,
    pub(crate) future_motion_ratio: f32,
    pub(crate) future_stop_mean: f32,
    pub(crate) future_stop_std: f32,
    pub(crate) future_fixation_motion: f32,
    pub(crate) future_confidence_mean: f32,
    pub(crate) context_fixation_teacher_l1: f32,
    pub(crate) future_fixation_teacher_l1: f32,
}

impl SelectedValidationMetrics {
    pub(crate) fn from_snapshot(
        step: usize,
        snapshot: &LossSnapshot,
        future_penalty_weight: f32,
        checkpoint_base: Option<PathBuf>,
    ) -> Self {
        Self {
            step,
            total: snapshot.total,
            future: snapshot.future,
            recon_current: snapshot.recon_current,
            recon_future: snapshot.recon_future,
            future_psnr: snapshot.future_psnr,
            future_fg_iou: snapshot.future_fg_iou,
            future_motion_ratio: snapshot.future_motion_ratio,
            future_fixation_teacher_l1: snapshot.future_fixation_teacher_l1,
            quality_score: dynamics_quality_score(snapshot, future_penalty_weight),
            checkpoint_base,
        }
    }
}

impl LossSnapshot {
    fn accumulate(&mut self, other: &Self) {
        self.total += other.total;
        self.current += other.current;
        self.future += other.future;
        self.prior += other.prior;
        self.gaze += other.gaze;
        self.query += other.query;
        self.recon += other.recon;
        self.tokenizer += other.tokenizer;
        self.tokenizer_recon += other.tokenizer_recon;
        self.slot_align += other.slot_align;
        self.recon_current += other.recon_current;
        self.recon_future += other.recon_future;
        self.recon_edge += other.recon_edge;
        self.recon_motion += other.recon_motion;
        self.current_mae += other.current_mae;
        self.future_mae += other.future_mae;
        self.current_psnr += other.current_psnr;
        self.future_psnr += other.future_psnr;
        self.current_fg_iou += other.current_fg_iou;
        self.future_fg_iou += other.future_fg_iou;
        self.future_frame_std += other.future_frame_std;
        self.future_latent_std += other.future_latent_std;
        self.future_motion_mse += other.future_motion_mse;
        self.future_ref_motion_mse += other.future_ref_motion_mse;
        self.future_motion_ratio += other.future_motion_ratio;
        self.future_stop_mean += other.future_stop_mean;
        self.future_stop_std += other.future_stop_std;
        self.future_fixation_motion += other.future_fixation_motion;
        self.future_confidence_mean += other.future_confidence_mean;
        self.context_fixation_teacher_l1 += other.context_fixation_teacher_l1;
        self.future_fixation_teacher_l1 += other.future_fixation_teacher_l1;
    }

    fn div_scalar(mut self, denom: f32) -> Self {
        self.total /= denom;
        self.current /= denom;
        self.future /= denom;
        self.prior /= denom;
        self.gaze /= denom;
        self.query /= denom;
        self.recon /= denom;
        self.tokenizer /= denom;
        self.tokenizer_recon /= denom;
        self.slot_align /= denom;
        self.recon_current /= denom;
        self.recon_future /= denom;
        self.recon_edge /= denom;
        self.recon_motion /= denom;
        self.current_mae /= denom;
        self.future_mae /= denom;
        self.current_psnr /= denom;
        self.future_psnr /= denom;
        self.current_fg_iou /= denom;
        self.future_fg_iou /= denom;
        self.future_frame_std /= denom;
        self.future_latent_std /= denom;
        self.future_motion_mse /= denom;
        self.future_ref_motion_mse /= denom;
        self.future_motion_ratio /= denom;
        self.future_stop_mean /= denom;
        self.future_stop_std /= denom;
        self.future_fixation_motion /= denom;
        self.future_confidence_mean /= denom;
        self.context_fixation_teacher_l1 /= denom;
        self.future_fixation_teacher_l1 /= denom;
        self
    }
}

#[derive(Clone, Debug)]
struct SequenceReconstructionMetricTensors<B: burn::tensor::backend::Backend> {
    mae: burn::tensor::Tensor<B, 1>,
    psnr: burn::tensor::Tensor<B, 1>,
    fg_iou: burn::tensor::Tensor<B, 1>,
    frame_std: burn::tensor::Tensor<B, 1>,
    motion_mse: burn::tensor::Tensor<B, 1>,
    ref_motion_mse: burn::tensor::Tensor<B, 1>,
    motion_ratio: burn::tensor::Tensor<B, 1>,
}

#[derive(Clone, Debug)]
struct FutureFixationMetricTensors<B: burn::tensor::backend::Backend> {
    stop_mean: burn::tensor::Tensor<B, 1>,
    stop_std: burn::tensor::Tensor<B, 1>,
    fixation_motion: burn::tensor::Tensor<B, 1>,
    confidence_mean: burn::tensor::Tensor<B, 1>,
}

#[derive(Clone, Debug)]
struct FixationAlignmentMetricTensors<B: burn::tensor::backend::Backend> {
    context_l1: burn::tensor::Tensor<B, 1>,
    future_l1: burn::tensor::Tensor<B, 1>,
}

#[derive(Clone, Debug, Default)]
struct SequenceReconstructionMetrics {
    mae: f32,
    psnr: f32,
    fg_iou: f32,
    frame_std: f32,
    motion_mse: f32,
    ref_motion_mse: f32,
    motion_ratio: f32,
}

#[derive(Clone, Debug, Default)]
struct FutureFixationMetrics {
    stop_mean: f32,
    stop_std: f32,
    fixation_motion: f32,
    confidence_mean: f32,
}

#[derive(Clone, Debug, Default)]
struct FixationAlignmentMetrics {
    context_l1: f32,
    future_l1: f32,
}

pub(crate) fn rollout_selection_score(snapshot: &LossSnapshot) -> f32 {
    snapshot.future_psnr
        + 24.0 * snapshot.future_fg_iou
        + 4.0 * snapshot.future_motion_ratio.min(1.0)
        - 1.5 * snapshot.recon_future
}

pub(crate) fn dynamics_quality_score(snapshot: &LossSnapshot, future_penalty_weight: f32) -> f32 {
    rollout_selection_score(snapshot)
        - future_penalty_weight * snapshot.future
        - 0.25 * snapshot.future_fixation_teacher_l1
}

fn unit_interval_sequence_tensor<B: burn::tensor::backend::Backend>(
    tensor: burn::tensor::Tensor<B, 5>,
) -> burn::tensor::Tensor<B, 5> {
    tensor.mul_scalar(0.5).add_scalar(0.5).clamp(0.0, 1.0)
}

fn zero_scalar<B: burn::tensor::backend::Backend>(
    device: &B::Device,
) -> burn::tensor::Tensor<B, 1> {
    burn::tensor::Tensor::<B, 1>::zeros([1], device)
}

fn tensor_std_all<B: burn::tensor::backend::Backend>(
    tensor: burn::tensor::Tensor<B, 3>,
) -> burn::tensor::Tensor<B, 1> {
    let mean = tensor.clone().mean().reshape([1]);
    let mean_sq = tensor.powf_scalar(2.0).mean().reshape([1]);
    mean_sq
        .sub(mean.clone().powf_scalar(2.0))
        .clamp(0.0, 1.0e6)
        .sqrt()
}

fn sequence_reconstruction_metrics_tensor<B: burn::tensor::backend::Backend>(
    reference: burn::tensor::Tensor<B, 5>,
    reconstruction: burn::tensor::Tensor<B, 5>,
) -> SequenceReconstructionMetricTensors<B> {
    let [_batch, steps, _channels, _height, _width] = reference.shape().dims::<5>();
    let pred_unit = unit_interval_sequence_tensor(reconstruction);
    let ref_unit = unit_interval_sequence_tensor(reference);
    let diff = pred_unit.clone() - ref_unit.clone();
    let mae = diff.clone().abs().mean().reshape([1]);
    let mse = diff.clone().powf_scalar(2.0).mean().reshape([1]);
    let psnr = mse
        .clone()
        .clamp(1.0e-8, 1.0e6)
        .log()
        .mul_scalar(-10.0 / std::f32::consts::LN_10)
        .reshape([1]);

    let ref_fg = ref_unit.clone().greater_elem(0.2).float();
    let pred_fg = pred_unit.clone().greater_elem(0.2).float();
    let intersection = ref_fg.clone().mul(pred_fg.clone()).sum().reshape([1]);
    let union = ref_fg
        .add(pred_fg)
        .greater_elem(0.0)
        .float()
        .sum()
        .reshape([1]);
    let fg_iou = intersection
        .add_scalar(1.0e-6)
        .div(union.add_scalar(1.0e-6));

    let pred_mean = pred_unit.clone().mean().reshape([1]);
    let pred_mean_sq = pred_unit.clone().powf_scalar(2.0).mean().reshape([1]);
    let frame_std = pred_mean_sq
        .sub(pred_mean.clone().powf_scalar(2.0))
        .clamp(0.0, 1.0e6)
        .sqrt();

    let device = pred_unit.device();
    if steps < 2 {
        let zero = zero_scalar(&device);
        return SequenceReconstructionMetricTensors {
            mae,
            psnr,
            fg_iou,
            frame_std,
            motion_mse: zero.clone(),
            ref_motion_mse: zero.clone(),
            motion_ratio: zero,
        };
    }

    let pred_delta =
        pred_unit.clone().slice_dim(1, 1..steps) - pred_unit.clone().slice_dim(1, 0..steps - 1);
    let ref_delta =
        ref_unit.clone().slice_dim(1, 1..steps) - ref_unit.clone().slice_dim(1, 0..steps - 1);
    let motion_mse = pred_delta.clone().powf_scalar(2.0).mean().reshape([1]);
    let ref_motion_mse = ref_delta.clone().powf_scalar(2.0).mean().reshape([1]);
    let motion_ratio = motion_mse
        .clone()
        .div(ref_motion_mse.clone().clamp(1.0e-8, 1.0e6));

    SequenceReconstructionMetricTensors {
        mae,
        psnr,
        fg_iou,
        frame_std,
        motion_mse,
        ref_motion_mse,
        motion_ratio,
    }
}

fn future_fixation_metrics_tensor<B: burn::tensor::backend::Backend>(
    fixations: burn::tensor::Tensor<B, 3>,
    context_steps: usize,
    future_steps: usize,
) -> FutureFixationMetricTensors<B> {
    let [batch, steps, features] = fixations.shape().dims::<3>();
    let device = fixations.device();
    let zero = zero_scalar(&device);
    let k = features.saturating_sub(1) / 4;
    if batch == 0 || steps == 0 || features == 0 || k == 0 {
        return FutureFixationMetricTensors {
            stop_mean: zero.clone(),
            stop_std: zero.clone(),
            fixation_motion: zero.clone(),
            confidence_mean: zero,
        };
    }

    let end = context_steps.saturating_add(future_steps).min(steps);
    if end <= context_steps {
        return FutureFixationMetricTensors {
            stop_mean: zero.clone(),
            stop_std: zero.clone(),
            fixation_motion: zero.clone(),
            confidence_mean: zero,
        };
    }

    let points = fixations
        .clone()
        .slice_dim(2, 0..k * 4)
        .reshape([batch, steps, k, 4]);
    let stop = fixations
        .slice_dim(2, k * 4..k * 4 + 1)
        .reshape([batch, steps]);
    let future_stop = stop.clone().slice_dim(1, context_steps..end);
    let stop_mean = future_stop.clone().mean().reshape([1]);
    let stop_std = future_stop
        .clone()
        .powf_scalar(2.0)
        .mean()
        .reshape([1])
        .sub(stop_mean.clone().powf_scalar(2.0))
        .clamp(0.0, 1.0e6)
        .sqrt();
    let confidence_mean = points
        .clone()
        .slice_dim(1, context_steps..end)
        .slice_dim(3, 3..4)
        .reshape([batch, end - context_steps, k])
        .mean()
        .reshape([1]);

    let motion_start = if context_steps == 0 { 1 } else { context_steps };
    let fixation_motion = if end <= motion_start {
        zero.clone()
    } else {
        let current = points
            .clone()
            .slice_dim(1, motion_start..end)
            .slice_dim(3, 0..2);
        let previous = points
            .slice_dim(1, motion_start - 1..end - 1)
            .slice_dim(3, 0..2);
        (current - previous)
            .powf_scalar(2.0)
            .sum_dim(3)
            .sqrt()
            .mean()
            .reshape([1])
    };

    FutureFixationMetricTensors {
        stop_mean,
        stop_std,
        fixation_motion,
        confidence_mean,
    }
}

fn fixation_teacher_alignment_metrics_tensor<B: burn::tensor::backend::Backend>(
    teacher: burn::tensor::Tensor<B, 3>,
    predicted: burn::tensor::Tensor<B, 3>,
    context_steps: usize,
    future_steps: usize,
) -> FixationAlignmentMetricTensors<B> {
    let [teacher_batch, teacher_steps, teacher_features] = teacher.shape().dims::<3>();
    let [pred_batch, pred_steps, pred_features] = predicted.shape().dims::<3>();
    let batch = teacher_batch.min(pred_batch);
    let steps = teacher_steps.min(pred_steps);
    let k = teacher_features
        .saturating_sub(1)
        .min(pred_features.saturating_sub(1))
        / 4;
    let device = teacher.device();
    let zero = zero_scalar(&device);
    if batch == 0 || steps == 0 || k == 0 {
        return FixationAlignmentMetricTensors {
            context_l1: zero.clone(),
            future_l1: zero,
        };
    }

    let end = context_steps.saturating_add(future_steps).min(steps);
    let teacher_points = teacher
        .clone()
        .slice_dim(0, 0..batch)
        .slice_dim(1, 0..end)
        .slice_dim(2, 0..k * 4)
        .reshape([batch, end, k, 4]);
    let predicted_points = predicted
        .clone()
        .slice_dim(0, 0..batch)
        .slice_dim(1, 0..end)
        .slice_dim(2, 0..k * 4)
        .reshape([batch, end, k, 4]);

    let context_l1 = if context_steps == 0 {
        zero.clone()
    } else {
        teacher_points
            .clone()
            .slice_dim(1, 0..context_steps.min(end))
            .sub(
                predicted_points
                    .clone()
                    .slice_dim(1, 0..context_steps.min(end)),
            )
            .abs()
            .mean()
            .reshape([1])
    };
    let future_l1 = if end <= context_steps {
        zero
    } else {
        teacher_points
            .slice_dim(1, context_steps..end)
            .sub(predicted_points.slice_dim(1, context_steps..end))
            .abs()
            .mean()
            .reshape([1])
    };

    FixationAlignmentMetricTensors {
        context_l1,
        future_l1,
    }
}

pub(crate) fn scalar_pack<B: burn::tensor::backend::Backend>(
    tensors: Vec<burn::tensor::Tensor<B, 1>>,
) -> Vec<f32> {
    burn::tensor::Tensor::cat(tensors, 0)
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("metric tensor pack")
}

fn loss_snapshot_from_forward_and_debug<B: burn::tensor::backend::Backend>(
    forward: &DreamerForward<B>,
    debug: &DreamerDebugOutput<B>,
) -> LossSnapshot {
    let context_steps = debug.context_reference_frames.shape().dims::<5>()[1];
    let future_steps = debug.future_reference_frames.shape().dims::<5>()[1];
    let current_stats = sequence_reconstruction_metrics_tensor(
        debug.context_reference_frames.clone(),
        debug.context_reconstruction_frames.clone(),
    );
    let future_stats = sequence_reconstruction_metrics_tensor(
        debug.future_reference_frames.clone(),
        debug.future_reconstruction_frames.clone(),
    );
    let fixation_stats = future_fixation_metrics_tensor(
        debug.predicted_fixations.clone(),
        context_steps,
        future_steps,
    );
    let fixation_alignment = fixation_teacher_alignment_metrics_tensor(
        debug.teacher_fixations.clone(),
        debug.predicted_fixations.clone(),
        context_steps,
        future_steps,
    );
    let future_latent_std = tensor_std_all(debug.future_latents.clone());

    let mut values = scalar_pack(vec![
        forward.total.clone(),
        forward.current.clone(),
        forward.future.clone(),
        forward.prior.clone(),
        forward.gaze.clone(),
        forward.query.clone(),
        forward.recon.clone(),
        forward.tokenizer.clone(),
        forward.tokenizer_recon.clone(),
        forward.slot_align.clone(),
        forward.recon_current.clone(),
        forward.recon_future.clone(),
        forward.recon_edge.clone(),
        forward.recon_motion.clone(),
        current_stats.mae,
        future_stats.mae,
        current_stats.psnr,
        future_stats.psnr,
        current_stats.fg_iou,
        future_stats.fg_iou,
        future_stats.frame_std,
        future_latent_std,
        future_stats.motion_mse,
        future_stats.ref_motion_mse,
        future_stats.motion_ratio,
        fixation_stats.stop_mean,
        fixation_stats.stop_std,
        fixation_stats.fixation_motion,
        fixation_stats.confidence_mean,
        fixation_alignment.context_l1,
        fixation_alignment.future_l1,
    ])
    .into_iter();

    let mut next = || values.next().expect("validation metric scalar");
    LossSnapshot {
        total: next(),
        current: next(),
        future: next(),
        prior: next(),
        gaze: next(),
        query: next(),
        recon: next(),
        tokenizer: next(),
        tokenizer_recon: next(),
        slot_align: next(),
        recon_current: next(),
        recon_future: next(),
        recon_edge: next(),
        recon_motion: next(),
        current_mae: next(),
        future_mae: next(),
        current_psnr: next(),
        future_psnr: next(),
        current_fg_iou: next(),
        future_fg_iou: next(),
        future_frame_std: next(),
        future_latent_std: next(),
        future_motion_mse: next(),
        future_ref_motion_mse: next(),
        future_motion_ratio: next(),
        future_stop_mean: next(),
        future_stop_std: next(),
        future_fixation_motion: next(),
        future_confidence_mean: next(),
        context_fixation_teacher_l1: next(),
        future_fixation_teacher_l1: next(),
    }
}

pub(crate) fn artifact_metrics_from_forward_and_snapshot<B: burn::tensor::backend::Backend>(
    forward: &DreamerForward<B>,
    snapshot: &DreamerArtifactSnapshot,
    latent_backend: &str,
) -> ArtifactMetrics {
    let current_stats = sequence_reconstruction_metrics(
        &snapshot.current_reference,
        &snapshot.current_reconstruction,
    );
    let future_stats = sequence_reconstruction_metrics(
        &snapshot.future_reference,
        &snapshot.future_reconstruction,
    );
    let fixation_stats = future_fixation_metrics(
        &snapshot.predicted_fixations,
        snapshot.current_reference.steps,
        snapshot.future_reference.steps,
    );
    let fixation_alignment = fixation_teacher_alignment_metrics(
        &snapshot.teacher_fixations,
        &snapshot.predicted_fixations,
        snapshot.current_reference.steps,
        snapshot.future_reference.steps,
    );
    ArtifactMetrics {
        latent_backend: latent_backend.to_string(),
        total: scalar(&forward.total),
        current: scalar(&forward.current),
        future: scalar(&forward.future),
        prior: scalar(&forward.prior),
        gaze: scalar(&forward.gaze),
        query: scalar(&forward.query),
        recon: scalar(&forward.recon),
        tokenizer: scalar(&forward.tokenizer),
        tokenizer_recon: scalar(&forward.tokenizer_recon),
        slot_align: scalar(&forward.slot_align),
        recon_current: scalar(&forward.recon_current),
        recon_future: scalar(&forward.recon_future),
        recon_edge: scalar(&forward.recon_edge),
        recon_motion: scalar(&forward.recon_motion),
        current_mae: current_stats.mae,
        future_mae: future_stats.mae,
        current_psnr: current_stats.psnr,
        future_psnr: future_stats.psnr,
        current_fg_iou: current_stats.fg_iou,
        future_fg_iou: future_stats.fg_iou,
        future_frame_std: future_stats.frame_std,
        future_latent_std: standard_deviation(&snapshot.future_latents.data),
        future_motion_mse: future_stats.motion_mse,
        future_ref_motion_mse: future_stats.ref_motion_mse,
        future_motion_ratio: future_stats.motion_ratio,
        future_stop_mean: fixation_stats.stop_mean,
        future_stop_std: fixation_stats.stop_std,
        future_fixation_motion: fixation_stats.fixation_motion,
        future_confidence_mean: fixation_stats.confidence_mean,
        context_fixation_teacher_l1: fixation_alignment.context_l1,
        future_fixation_teacher_l1: fixation_alignment.future_l1,
    }
}

fn sequence_reconstruction_metrics(
    reference: &SequenceTensor,
    reconstruction: &SequenceTensor,
) -> SequenceReconstructionMetrics {
    let batch = reference.batch.min(reconstruction.batch);
    let steps = reference.steps.min(reconstruction.steps);
    let channels = reference.channels.min(reconstruction.channels);
    let height = reference.height.min(reconstruction.height);
    let width = reference.width.min(reconstruction.width);
    if batch == 0 || steps == 0 || channels == 0 || height == 0 || width == 0 {
        return SequenceReconstructionMetrics::default();
    }

    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut pixel_count = 0usize;
    let mut intersection = 0usize;
    let mut union = 0usize;
    let mut pred_sum = 0.0f64;
    let mut pred_sq_sum = 0.0f64;
    let mut pred_motion_sq_sum = 0.0f64;
    let mut ref_motion_sq_sum = 0.0f64;
    let mut motion_count = 0usize;

    for batch_idx in 0..batch {
        for step_idx in 0..steps {
            for channel_idx in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let ref_value = unit_interval(sequence_value(
                            reference,
                            batch_idx,
                            step_idx,
                            channel_idx,
                            y,
                            x,
                        ));
                        let pred_value = unit_interval(sequence_value(
                            reconstruction,
                            batch_idx,
                            step_idx,
                            channel_idx,
                            y,
                            x,
                        ));
                        let diff = pred_value - ref_value;
                        abs_sum += diff.abs() as f64;
                        sq_sum += (diff * diff) as f64;
                        pred_sum += pred_value as f64;
                        pred_sq_sum += (pred_value * pred_value) as f64;
                        pixel_count += 1;

                        let ref_fg = ref_value > 0.2;
                        let pred_fg = pred_value > 0.2;
                        if ref_fg && pred_fg {
                            intersection += 1;
                        }
                        if ref_fg || pred_fg {
                            union += 1;
                        }

                        if step_idx > 0 {
                            let prev_ref = unit_interval(sequence_value(
                                reference,
                                batch_idx,
                                step_idx - 1,
                                channel_idx,
                                y,
                                x,
                            ));
                            let prev_pred = unit_interval(sequence_value(
                                reconstruction,
                                batch_idx,
                                step_idx - 1,
                                channel_idx,
                                y,
                                x,
                            ));
                            let pred_delta = pred_value - prev_pred;
                            let ref_delta = ref_value - prev_ref;
                            pred_motion_sq_sum += (pred_delta * pred_delta) as f64;
                            ref_motion_sq_sum += (ref_delta * ref_delta) as f64;
                            motion_count += 1;
                        }
                    }
                }
            }
        }
    }

    let pixel_denom = pixel_count.max(1) as f64;
    let mse = (sq_sum / pixel_denom) as f32;
    let pred_mean = pred_sum / pixel_denom;
    let pred_var = (pred_sq_sum / pixel_denom) - pred_mean * pred_mean;
    let motion_denom = motion_count.max(1) as f64;
    let motion_mse = (pred_motion_sq_sum / motion_denom) as f32;
    let ref_motion_mse = (ref_motion_sq_sum / motion_denom) as f32;
    SequenceReconstructionMetrics {
        mae: (abs_sum / pixel_denom) as f32,
        psnr: 10.0 * (1.0f32 / mse.max(1.0e-8)).log10(),
        fg_iou: if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        },
        frame_std: pred_var.max(0.0).sqrt() as f32,
        motion_mse,
        ref_motion_mse,
        motion_ratio: motion_mse / ref_motion_mse.max(1.0e-8),
    }
}

fn future_fixation_metrics(
    fixations: &FixationSequence,
    context_steps: usize,
    future_steps: usize,
) -> FutureFixationMetrics {
    let end = context_steps + future_steps;
    let mut stop_values = Vec::new();
    let mut confidence_sum = 0.0f32;
    let mut confidence_count = 0usize;
    let mut motion_sum = 0.0f32;
    let mut motion_count = 0usize;

    for (batch_idx, steps) in fixations.points.iter().enumerate() {
        let stop_steps = fixations
            .stop_probabilities
            .get(batch_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for step_idx in context_steps..end.min(steps.len()) {
            if let Some(stop) = stop_steps.get(step_idx) {
                stop_values.push(*stop);
            }
            for point in &steps[step_idx] {
                confidence_sum += point.confidence;
                confidence_count += 1;
            }
            if step_idx > 0 {
                let previous = &steps[step_idx - 1];
                let current = &steps[step_idx];
                for fixation_idx in 0..previous.len().min(current.len()) {
                    let dx = current[fixation_idx].x - previous[fixation_idx].x;
                    let dy = current[fixation_idx].y - previous[fixation_idx].y;
                    motion_sum += (dx * dx + dy * dy).sqrt();
                    motion_count += 1;
                }
            }
        }
    }

    let stop_mean = if stop_values.is_empty() {
        0.0
    } else {
        stop_values.iter().sum::<f32>() / stop_values.len() as f32
    };
    let stop_std = if stop_values.is_empty() {
        0.0
    } else {
        let mean = stop_mean;
        let variance = stop_values
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>()
            / stop_values.len() as f32;
        variance.max(0.0).sqrt()
    };

    FutureFixationMetrics {
        stop_mean,
        stop_std,
        fixation_motion: if motion_count == 0 {
            0.0
        } else {
            motion_sum / motion_count as f32
        },
        confidence_mean: if confidence_count == 0 {
            0.0
        } else {
            confidence_sum / confidence_count as f32
        },
    }
}

fn fixation_teacher_alignment_metrics(
    teacher: &FixationSequence,
    predicted: &FixationSequence,
    context_steps: usize,
    future_steps: usize,
) -> FixationAlignmentMetrics {
    let mut context_sum = 0.0f32;
    let mut context_count = 0usize;
    let mut future_sum = 0.0f32;
    let mut future_count = 0usize;
    let end = context_steps + future_steps;
    let batch = teacher.points.len().min(predicted.points.len());
    for batch_idx in 0..batch {
        let teacher_steps = &teacher.points[batch_idx];
        let predicted_steps = &predicted.points[batch_idx];
        let total_steps = teacher_steps.len().min(predicted_steps.len()).min(end);
        for step_idx in 0..total_steps {
            let teacher_points = &teacher_steps[step_idx];
            let predicted_points = &predicted_steps[step_idx];
            for point_idx in 0..teacher_points.len().min(predicted_points.len()) {
                let teacher_point = &teacher_points[point_idx];
                let predicted_point = &predicted_points[point_idx];
                let l1 = (teacher_point.x - predicted_point.x).abs()
                    + (teacher_point.y - predicted_point.y).abs()
                    + (teacher_point.scale - predicted_point.scale).abs()
                    + (teacher_point.confidence - predicted_point.confidence).abs();
                if step_idx < context_steps {
                    context_sum += l1;
                    context_count += 1;
                } else {
                    future_sum += l1;
                    future_count += 1;
                }
            }
        }
    }
    FixationAlignmentMetrics {
        context_l1: if context_count == 0 {
            0.0
        } else {
            context_sum / context_count as f32
        },
        future_l1: if future_count == 0 {
            0.0
        } else {
            future_sum / future_count as f32
        },
    }
}

fn unit_interval(value: f32) -> f32 {
    (value * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn sequence_value(
    sequence: &SequenceTensor,
    batch_idx: usize,
    step_idx: usize,
    channel_idx: usize,
    y: usize,
    x: usize,
) -> f32 {
    let index = ((((batch_idx * sequence.steps + step_idx) * sequence.channels + channel_idx)
        * sequence.height
        + y)
        * sequence.width)
        + x;
    sequence.data[index]
}

fn standard_deviation(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.max(0.0).sqrt()
}

pub(crate) fn evaluate_validation(
    model: &DragonDreamer<TrainBackend>,
    dataset: &CachedSequenceSplit<TrainBackend, FrameFixationTrace>,
    config: &MovingMnistDreamerTrainConfig,
    batches: usize,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> LossSnapshot {
    let mut totals = LossSnapshot::default();
    let count = batches.max(1);
    let runtime_model: DragonDreamer<RuntimeBackend> = model.valid();
    for batch_idx in 0..count {
        let indices = sample_indices(
            dataset.len(),
            config.batch_size,
            config
                .val_seed
                .wrapping_add((batch_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)),
        );
        let batch = dataset.batch(&indices);
        let clip_frames = train_to_runtime_tensor5(batch.clip_frames, device);
        let actions = batch
            .actions
            .map(|tensor| train_to_runtime_tensor3(tensor, device));
        let traces = batch.traces;
        let teacher_features = train_to_runtime_tensor3(batch.teacher_features, device);
        let crop_teacher_features = train_to_runtime_tensor3(batch.crop_teacher_features, device);
        let (forward, debug) = runtime_model.forward_with_debug(
            clip_frames,
            &traces,
            actions,
            teacher_features,
            crop_teacher_features,
            config.context_len,
            config.target_len,
        );
        let metrics = loss_snapshot_from_forward_and_debug(&forward, &debug);
        totals.accumulate(&metrics);
    }
    totals.div_scalar(count as f32)
}

pub(crate) fn scalar<B: burn::tensor::backend::Backend>(
    tensor: &burn::tensor::Tensor<B, 1>,
) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor")[0]
}
