use super::{
    AutoGazeSource, ClipTeacherSource, artifact_metrics_from_forward_and_snapshot,
    encode_crop_teacher, passive_full_frame_traces, zero_teacher_features,
};
use crate::artifacts::{
    ArtifactMetrics, DreamerArtifactSnapshot, FixationPointArtifact, FixationSequence,
    LatentTensor, SequenceTensor, StreamingArtifactWriter,
};
use crate::config::MovingMnistDreamerTrainConfig;
use crate::runtime::{Backend, TrainBackend, train_to_runtime_tensor3, train_to_runtime_tensor5};
use crate::{DragonDreamer, DreamerDebugOutput};
use anyhow::{Context, Result};
use burn::module::AutodiffModule;
use burn::tensor::backend::Backend as BurnBackend;
use burn_dragon_vision::MovingMnistVideoDataset;
use std::path::PathBuf;

pub(crate) fn should_write_artifacts(config: &MovingMnistDreamerTrainConfig, step: usize) -> bool {
    config.artifact_enabled
        && config.artifact_dir.is_some()
        && config.artifact_every > 0
        && step % config.artifact_every.max(1) == 0
}

pub(crate) fn write_validation_artifacts(
    model: &DragonDreamer<TrainBackend>,
    global_teacher: Option<&ClipTeacherSource>,
    crop_teacher: Option<&ClipTeacherSource>,
    teacher: &AutoGazeSource,
    dataset: &MovingMnistVideoDataset,
    config: &MovingMnistDreamerTrainConfig,
    artifact_context_len: usize,
    artifact_target_len: usize,
    step: usize,
    is_final: bool,
    output_label: Option<&str>,
    device: &<TrainBackend as BurnBackend>::Device,
) -> Result<PathBuf> {
    if config.model.passive_full_frame {
        return write_validation_artifacts_passive_runtime(
            model,
            teacher,
            dataset,
            config,
            artifact_context_len,
            artifact_target_len,
            step,
            is_final,
            output_label,
            device,
        );
    }
    let effective_artifact_target_len = [global_teacher, crop_teacher]
        .into_iter()
        .flatten()
        .filter_map(ClipTeacherSource::target_len_limit)
        .fold(artifact_target_len, |limit, teacher_limit| {
            limit.min(teacher_limit.max(1))
        })
        .max(1);
    let sample_count = config.artifact_samples.max(1).min(dataset.len().max(1));
    let indices: Vec<usize> = (0..sample_count).collect();
    let root = config
        .artifact_dir
        .as_ref()
        .expect("artifact dir should be present");
    let output_dir = if is_final {
        root.join(output_label.unwrap_or("final"))
    } else {
        root.join(format!("step_{step:06}"))
    };
    let runtime_model: DragonDreamer<Backend> = model.valid();
    let mut writer =
        StreamingArtifactWriter::new(&output_dir, config.model.latent_backend.as_str())?;
    let mut merged_metrics = None;
    let mut merged_weight = 0usize;

    for chunk in indices.chunks(1) {
        let batch = dataset.batch_from_indices::<TrainBackend>(chunk, device);
        let actions = Some(dataset.action_batch_from_indices::<Backend>(chunk, device));
        let traces = teacher.traces_for_batch(chunk, &batch, config.model.k_fovea);
        let total_steps = batch.clip_frames.shape().dims::<5>()[1];
        let teacher_features = if let Some(global_teacher) = global_teacher {
            global_teacher.encode_clip_from_batch(chunk, batch.clip_frames.clone().detach())
        } else {
            zero_teacher_features(chunk.len(), total_steps, config.model.teacher_dim, device)
        };
        let crop_teacher_features = if let Some(crop_teacher) = crop_teacher {
            encode_crop_teacher(
                crop_teacher,
                chunk,
                batch.clip_frames.clone().detach(),
                &traces,
                config,
            )
        } else {
            zero_teacher_features(
                chunk.len(),
                total_steps,
                config.model.crop_teacher_dim,
                device,
            )
        };
        let clip_frames = train_to_runtime_tensor5(batch.clip_frames, device);
        let teacher_features = train_to_runtime_tensor3(teacher_features, device);
        let crop_teacher_features = train_to_runtime_tensor3(crop_teacher_features, device);
        let (forward, debug) = runtime_model.forward_with_debug(
            clip_frames,
            &traces,
            actions,
            teacher_features,
            crop_teacher_features,
            artifact_context_len,
            effective_artifact_target_len,
        );
        let teacher_visibility = teacher
            .visibility_for_batch(chunk, artifact_context_len + effective_artifact_target_len);
        let snapshot = build_artifact_snapshot(
            debug,
            teacher_visibility,
            config.model.crop_size,
            config.model.passive_full_frame,
        )?;
        writer.push_snapshot(&snapshot)?;
        let metrics = artifact_metrics_from_forward_and_snapshot(
            &forward,
            &snapshot,
            config.model.latent_backend.as_str(),
        );
        merged_metrics = Some(match merged_metrics {
            Some(existing) => {
                merge_artifact_metrics(existing, metrics, merged_weight as f32, chunk.len() as f32)
            }
            None => metrics,
        });
        merged_weight += chunk.len();
    }

    writer.finish(&merged_metrics.expect("artifact metrics"))
}

fn write_validation_artifacts_passive_runtime(
    model: &DragonDreamer<TrainBackend>,
    teacher: &AutoGazeSource,
    dataset: &MovingMnistVideoDataset,
    config: &MovingMnistDreamerTrainConfig,
    artifact_context_len: usize,
    artifact_target_len: usize,
    step: usize,
    is_final: bool,
    output_label: Option<&str>,
    device: &<TrainBackend as BurnBackend>::Device,
) -> Result<PathBuf> {
    let sample_count = config.artifact_samples.max(1).min(dataset.len().max(1));
    let indices: Vec<usize> = (0..sample_count).collect();
    let root = config
        .artifact_dir
        .as_ref()
        .expect("artifact dir should be present");
    let output_dir = if is_final {
        root.join(output_label.unwrap_or("final"))
    } else {
        root.join(format!("step_{step:06}"))
    };
    let runtime_model: DragonDreamer<Backend> = model.valid();
    let mut writer =
        StreamingArtifactWriter::new(&output_dir, config.model.latent_backend.as_str())?;
    let mut merged_metrics = None;
    let mut merged_weight = 0usize;

    for chunk in indices.chunks(1) {
        let batch = dataset.batch_from_indices::<Backend>(chunk, device);
        let actions = Some(dataset.action_batch_from_indices::<Backend>(chunk, device));
        let total_steps = batch.clip_frames.shape().dims::<5>()[1];
        let traces = match teacher {
            AutoGazeSource::PassiveFullFrame { .. } => {
                passive_full_frame_traces(chunk.len(), total_steps, config.model.k_fovea)
            }
            _ => passive_full_frame_traces(chunk.len(), total_steps, config.model.k_fovea),
        };
        let teacher_features = burn::tensor::Tensor::<Backend, 3>::zeros(
            [chunk.len(), total_steps, config.model.teacher_dim.max(1)],
            device,
        );
        let crop_teacher_features = burn::tensor::Tensor::<Backend, 3>::zeros(
            [
                chunk.len(),
                total_steps,
                config.model.crop_teacher_dim.max(1),
            ],
            device,
        );
        let (forward, debug) = runtime_model.forward_with_debug(
            batch.clip_frames,
            &traces,
            actions,
            teacher_features,
            crop_teacher_features,
            artifact_context_len,
            artifact_target_len,
        );
        let teacher_visibility =
            teacher.visibility_for_batch(chunk, artifact_context_len + artifact_target_len);
        let snapshot = build_artifact_snapshot(
            debug,
            teacher_visibility,
            config.model.crop_size,
            config.model.passive_full_frame,
        )?;
        writer.push_snapshot(&snapshot)?;
        let metrics = artifact_metrics_from_forward_and_snapshot(
            &forward,
            &snapshot,
            config.model.latent_backend.as_str(),
        );
        merged_metrics = Some(match merged_metrics {
            Some(existing) => {
                merge_artifact_metrics(existing, metrics, merged_weight as f32, chunk.len() as f32)
            }
            None => metrics,
        });
        merged_weight += chunk.len();
    }

    writer.finish(&merged_metrics.expect("artifact metrics"))
}

fn merge_artifact_metrics(
    lhs: ArtifactMetrics,
    rhs: ArtifactMetrics,
    lhs_weight: f32,
    rhs_weight: f32,
) -> ArtifactMetrics {
    let denom = (lhs_weight + rhs_weight).max(1.0);
    macro_rules! avg {
        ($field:ident) => {
            (lhs.$field * lhs_weight + rhs.$field * rhs_weight) / denom
        };
    }
    ArtifactMetrics {
        latent_backend: lhs.latent_backend,
        total: avg!(total),
        current: avg!(current),
        future: avg!(future),
        prior: avg!(prior),
        gaze: avg!(gaze),
        query: avg!(query),
        recon: avg!(recon),
        tokenizer: avg!(tokenizer),
        tokenizer_recon: avg!(tokenizer_recon),
        slot_align: avg!(slot_align),
        recon_current: avg!(recon_current),
        recon_future: avg!(recon_future),
        recon_edge: avg!(recon_edge),
        recon_motion: avg!(recon_motion),
        current_mae: avg!(current_mae),
        future_mae: avg!(future_mae),
        current_psnr: avg!(current_psnr),
        future_psnr: avg!(future_psnr),
        current_fg_iou: avg!(current_fg_iou),
        future_fg_iou: avg!(future_fg_iou),
        future_frame_std: avg!(future_frame_std),
        future_latent_std: avg!(future_latent_std),
        future_motion_mse: avg!(future_motion_mse),
        future_ref_motion_mse: avg!(future_ref_motion_mse),
        future_motion_ratio: avg!(future_motion_ratio),
        future_stop_mean: avg!(future_stop_mean),
        future_stop_std: avg!(future_stop_std),
        future_fixation_motion: avg!(future_fixation_motion),
        future_confidence_mean: avg!(future_confidence_mean),
        context_fixation_teacher_l1: avg!(context_fixation_teacher_l1),
        future_fixation_teacher_l1: avg!(future_fixation_teacher_l1),
    }
}

fn build_artifact_snapshot<B: BurnBackend>(
    debug: DreamerDebugOutput<B>,
    teacher_visibility: Option<SequenceTensor>,
    crop_size: usize,
    passive_full_frame: bool,
) -> Result<DreamerArtifactSnapshot> {
    Ok(DreamerArtifactSnapshot {
        passive_full_frame,
        current_reference: sequence_from_tensor(debug.context_reference_frames)?,
        current_reconstruction: sequence_from_tensor(debug.context_reconstruction_frames)?,
        future_reference: sequence_from_tensor(debug.future_reference_frames)?,
        future_reconstruction: sequence_from_tensor(debug.future_reconstruction_frames)?,
        context_latents: latent_from_tensor(debug.context_latents)?,
        future_latents: latent_from_tensor(debug.future_latents)?,
        teacher_visibility,
        teacher_fixations: fixations_from_tensor(debug.teacher_fixations)?,
        predicted_fixations: fixations_from_tensor(debug.predicted_fixations)?,
        crop_size,
    })
}

fn sequence_from_tensor<B: BurnBackend>(
    tensor: burn::tensor::Tensor<B, 5>,
) -> Result<SequenceTensor> {
    let [batch, steps, channels, height, width] = tensor.shape().dims::<5>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("sequence tensor to host data")?;
    Ok(SequenceTensor {
        data,
        batch,
        steps,
        channels,
        height,
        width,
    })
}

fn latent_from_tensor<B: BurnBackend>(tensor: burn::tensor::Tensor<B, 3>) -> Result<LatentTensor> {
    let [batch, steps, dim] = tensor.shape().dims::<3>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("latent tensor to host data")?;
    Ok(LatentTensor {
        data,
        batch,
        steps,
        dim,
    })
}

fn fixations_from_tensor<B: BurnBackend>(
    tensor: burn::tensor::Tensor<B, 3>,
) -> Result<FixationSequence> {
    let [batch, steps, features] = tensor.shape().dims::<3>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("fixation tensor to host data")?;
    let k = features.saturating_sub(1) / 4;
    let mut points = Vec::with_capacity(batch);
    let mut stop_probabilities = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let mut per_step_points = Vec::with_capacity(steps);
        let mut per_step_stop = Vec::with_capacity(steps);
        for step_idx in 0..steps {
            let base = (batch_idx * steps + step_idx) * features;
            let mut per_fixation = Vec::with_capacity(k);
            for fixation_idx in 0..k {
                let offset = base + fixation_idx * 4;
                per_fixation.push(FixationPointArtifact {
                    x: data[offset].clamp(0.0, 1.0),
                    y: data[offset + 1].clamp(0.0, 1.0),
                    scale: data[offset + 2].clamp(0.01, 1.0),
                    confidence: data[offset + 3].clamp(0.0, 1.0),
                });
            }
            per_step_points.push(per_fixation);
            per_step_stop.push(data[base + k * 4].clamp(0.0, 1.0));
        }
        points.push(per_step_points);
        stop_probabilities.push(per_step_stop);
    }
    Ok(FixationSequence {
        points,
        stop_probabilities,
    })
}
