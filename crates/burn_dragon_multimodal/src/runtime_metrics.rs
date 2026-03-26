#![cfg(feature = "train")]

use anyhow::{Result, anyhow};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};

use crate::config::VlJepaDragonConfig;
use crate::runtime_config::{
    MultimodalTrainingConfig, MultimodalVideoTrainingConfig, RefineProbeMetric,
};
use crate::runtime_data::{
    PreparedTargetBank, batch_ranges, contiguous_batch_indices, target_bank_batch_for_targets,
    unique_target_batches,
};
use crate::train::{
    collate_video_language_segments, collate_vision_language_segments, multimodal_eval_step,
    multimodal_video_eval_step,
};

#[derive(Clone)]
pub(crate) struct EpochMetrics {
    pub(crate) steps: usize,
    pub(crate) mean_total_loss: f32,
    pub(crate) mean_diagonal_similarity: f32,
    pub(crate) mean_top1_accuracy: f32,
    pub(crate) refine_curve: Option<Vec<RefineProbeMetric>>,
}

pub(crate) fn scalar_from_tensor<B: BackendTrait>(tensor: burn::tensor::Tensor<B, 1>) -> f32 {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0]
}

pub(crate) fn diagonal_similarity_mean<B: BackendTrait>(
    similarities: burn::tensor::Tensor<B, 2>,
) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let diag = rows.min(cols).max(1);
    let mut sum = 0.0_f32;
    for index in 0..diag {
        sum += values[index * cols + index];
    }
    sum / diag as f32
}

pub(crate) fn bidirectional_top1_accuracy<B: BackendTrait>(
    similarities: burn::tensor::Tensor<B, 2>,
) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let diag = rows.min(cols).max(1);
    let mut row_hits = 0.0_f32;
    for row in 0..diag {
        let row_slice = &values[row * cols..(row + 1) * cols];
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in row_slice.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        if best_index == row {
            row_hits += 1.0;
        }
    }
    let mut col_hits = 0.0_f32;
    for col in 0..diag {
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;
        for row in 0..rows {
            let value = values[row * cols + col];
            if value > best_value {
                best_value = value;
                best_index = row;
            }
        }
        if best_index == col {
            col_hits += 1.0;
        }
    }
    (row_hits / diag as f32 + col_hits / diag as f32) / 2.0
}

pub(crate) fn labeled_top1_accuracy<B: BackendTrait>(
    similarities: burn::tensor::Tensor<B, 2>,
    target_indices: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let labels = target_indices
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("label vec");
    let mut hits = 0.0_f32;
    for row in 0..rows {
        let row_slice = &values[row * cols..(row + 1) * cols];
        let mut best_index = 0_usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in row_slice.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        if labels.get(row).copied() == Some(best_index as i64) {
            hits += 1.0;
        }
    }
    hits / rows.max(1) as f32
}

fn refine_probe_steps(max_steps: usize) -> Vec<usize> {
    if max_steps == 0 {
        return vec![0];
    }
    let mut steps = vec![0];
    let mut current = 1usize;
    while current < max_steps {
        steps.push(current);
        current = current.saturating_mul(2);
    }
    if steps.last().copied() != Some(max_steps) {
        steps.push(max_steps);
    }
    steps
}

#[allow(clippy::too_many_arguments)]
fn run_image_text_validation_refine_curve<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VisionLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
) -> Result<Vec<RefineProbeMetric>> {
    let max_refine_steps = model_config
        .eval_fusion_refine_steps
        .unwrap_or(model_config.fusion_refine_steps);
    let mut curve = Vec::new();
    for refine_steps in refine_probe_steps(max_refine_steps) {
        let mut eval_config = model_config.clone();
        eval_config.eval_fusion_refine_steps = Some(refine_steps);
        let metrics = run_image_text_validation_epoch(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            &eval_config,
            batched,
            false,
        )?;
        curve.push(RefineProbeMetric {
            refine_steps,
            mean_total_loss: metrics.mean_total_loss,
            mean_top1_accuracy: metrics.mean_top1_accuracy,
        });
    }
    Ok(curve)
}

#[allow(clippy::too_many_arguments)]
fn run_video_text_validation_refine_curve<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VideoLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalVideoTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
) -> Result<Vec<RefineProbeMetric>> {
    let max_refine_steps = model_config
        .eval_fusion_refine_steps
        .unwrap_or(model_config.fusion_refine_steps);
    let mut curve = Vec::new();
    for refine_steps in refine_probe_steps(max_refine_steps) {
        let mut eval_config = model_config.clone();
        eval_config.eval_fusion_refine_steps = Some(refine_steps);
        let metrics = run_video_text_validation_epoch(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            &eval_config,
            batched,
            false,
        )?;
        curve.push(RefineProbeMetric {
            refine_steps,
            mean_total_loss: metrics.mean_total_loss,
            mean_top1_accuracy: metrics.mean_top1_accuracy,
        });
    }
    Ok(curve)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_image_text_validation_epoch<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VisionLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
    collect_refine_curve: bool,
) -> Result<EpochMetrics> {
    let mut loss_sum = 0.0_f32;
    let mut diagonal_similarity_sum = 0.0_f32;
    let mut top1_accuracy_sum = 0.0_f32;
    let mut steps = 0_usize;
    if batched {
        let batch_indices =
            if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                contiguous_batch_indices(
                    validation.len(),
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                )
            } else {
                unique_target_batches(
                    validation,
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                    |segment| &segment.payload.target_y_tokens,
                )
            };
        for batch_indices in batch_indices {
            let segments = batch_indices
                .into_iter()
                .map(|index| validation[index].clone())
                .collect::<Vec<_>>();
            let collated = collate_vision_language_segments::<B>(&segments, device);
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                segments
                    .iter()
                    .map(|segment| segment.payload.target_y_tokens.as_slice()),
                device,
            );
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation image-text batch missing stream metadata")
                })?;
            let step = multimodal_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                model.init_state(),
                model_config,
            );
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    } else {
        let mut state = model.init_state();
        for range in batch_ranges(
            validation.len(),
            1,
            config.training.max_validation_steps_per_epoch,
        ) {
            let segment = validation[range.start].clone();
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                [segment.payload.target_y_tokens.as_slice()],
                device,
            );
            let collated = collate_vision_language_segments::<B>(&[segment], device);
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation image-text batch missing stream metadata")
                })?;
            let step = multimodal_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                state,
                model_config,
            );
            state = step.forward.state.detach();
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    }
    let mean_total_loss = loss_sum / steps.max(1) as f32;
    let mean_diagonal_similarity = diagonal_similarity_sum / steps.max(1) as f32;
    let mean_top1_accuracy = top1_accuracy_sum / steps.max(1) as f32;
    let refine_curve = if collect_refine_curve {
        Some(run_image_text_validation_refine_curve(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            model_config,
            batched,
        )?)
    } else {
        None
    };
    Ok(EpochMetrics {
        steps,
        mean_total_loss,
        mean_diagonal_similarity,
        mean_top1_accuracy,
        refine_curve,
    })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn run_video_text_validation_epoch<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VideoLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalVideoTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
    collect_refine_curve: bool,
) -> Result<EpochMetrics> {
    let mut loss_sum = 0.0_f32;
    let mut diagonal_similarity_sum = 0.0_f32;
    let mut top1_accuracy_sum = 0.0_f32;
    let mut steps = 0_usize;
    if batched {
        let batch_indices =
            if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                contiguous_batch_indices(
                    validation.len(),
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                )
            } else {
                unique_target_batches(
                    validation,
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                    |segment| &segment.payload.target_y_tokens,
                )
            };
        for batch_indices in batch_indices {
            let segments = batch_indices
                .into_iter()
                .map(|index| validation[index].clone())
                .collect::<Vec<_>>();
            let collated = collate_video_language_segments::<B>(&segments, device);
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                segments
                    .iter()
                    .map(|segment| segment.payload.target_y_tokens.as_slice()),
                device,
            );
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation video-text batch missing stream metadata")
                })?;
            let step = multimodal_video_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                model.init_state(),
                model_config,
            );
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    } else {
        let mut state = model.init_state();
        for range in batch_ranges(
            validation.len(),
            1,
            config.training.max_validation_steps_per_epoch,
        ) {
            let segment = validation[range.start].clone();
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                [segment.payload.target_y_tokens.as_slice()],
                device,
            );
            let collated = collate_video_language_segments::<B>(&[segment], device);
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation video-text batch missing stream metadata")
                })?;
            let step = multimodal_video_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                state,
                model_config,
            );
            state = step.forward.state.detach();
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    }
    let mean_total_loss = loss_sum / steps.max(1) as f32;
    let mean_diagonal_similarity = diagonal_similarity_sum / steps.max(1) as f32;
    let mean_top1_accuracy = top1_accuracy_sum / steps.max(1) as f32;
    let refine_curve = if collect_refine_curve {
        Some(run_video_text_validation_refine_curve(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            model_config,
            batched,
        )?)
    } else {
        None
    };
    Ok(EpochMetrics {
        steps,
        mean_total_loss,
        mean_diagonal_similarity,
        mean_top1_accuracy,
        refine_curve,
    })
}
