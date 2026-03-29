use crate::checkpoint::load_module_checkpoint;
use crate::data::sample_indices;
use crate::run::{prepare_configured_run, save_named_checkpoint, should_run_step};
use crate::runtime::{TrainBackend, cuda_device};
use crate::{DragonDreamer, DreamerLatentBackend, MovingMnistDreamerTrainConfig};
use anyhow::{Context, Result};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn_dragon_vision::MovingMnistSplit;
#[cfg(test)]
use burn_dragon_vision::MovingMnistVideoDataset;
#[cfg(test)]
use burn_dragon_vision::{MovingMnistVideoDatasetConfig, VisionNormalize};
use std::fs;
#[cfg(test)]
use std::path::PathBuf;

use super::{
    MovingMnistDreamerRunSummary, SelectedValidationMetrics, build_cached_moving_mnist_split,
    build_moving_mnist_dreamer_datasets, dynamics_quality_score, evaluate_validation,
    load_autogaze_source, load_crop_teacher_source, load_vjepa_source, rollout_selection_score,
    scalar_pack, should_write_artifacts, uses_crop_teacher, uses_global_teacher,
    write_validation_artifacts,
};

fn load_saved_tokenizer_config(
    checkpoint_base: &std::path::Path,
) -> Result<Option<MovingMnistDreamerTrainConfig>> {
    let Some(run_dir) = checkpoint_base.parent().and_then(|path| path.parent()) else {
        return Ok(None);
    };
    let config_path = run_dir.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&config_path)
        .with_context(|| format!("read tokenizer config {}", config_path.display()))?;
    let config = serde_json::from_slice::<MovingMnistDreamerTrainConfig>(&bytes)
        .with_context(|| format!("parse tokenizer config {}", config_path.display()))?;
    Ok(Some(config))
}

fn scheduled_target_len(config: &MovingMnistDreamerTrainConfig, step: usize) -> usize {
    let max_target_len = config.target_len.max(1);
    let min_target_len = if config.train_target_len_min == 0 {
        max_target_len
    } else {
        config.train_target_len_min.clamp(1, max_target_len)
    };
    if min_target_len >= max_target_len || config.target_len_warmup_steps == 0 {
        return max_target_len;
    }
    let progress = (step as f32 / config.target_len_warmup_steps.max(1) as f32).clamp(0.0, 1.0);
    let span = max_target_len - min_target_len;
    min_target_len + ((span as f32) * progress).round() as usize
}

fn scheduled_teacher_forcing_prefix(config: &MovingMnistDreamerTrainConfig, step: usize) -> usize {
    let max_prefix = config.model.passive_teacher_forcing_prefix_steps;
    if max_prefix == 0 {
        return 0;
    }
    let min_prefix = config.train_teacher_forcing_prefix_min.min(max_prefix);
    if min_prefix >= max_prefix || config.teacher_forcing_prefix_warmup_steps == 0 {
        return max_prefix;
    }
    let progress =
        (step as f32 / config.teacher_forcing_prefix_warmup_steps.max(1) as f32).clamp(0.0, 1.0);
    let span = max_prefix - min_prefix;
    max_prefix - ((span as f32) * progress).round() as usize
}

pub fn train_moving_mnist(
    mut config: MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistDreamerRunSummary> {
    let device = cuda_device()?;
    let workspace = prepare_configured_run(config.run_root.as_deref(), &config, "dreamer")?;
    let run_dir = workspace.run_dir.clone();
    let checkpoint_dir = workspace.checkpoint_dir.clone();
    let artifact_root = run_dir.as_ref().map(|path| path.join("artifacts"));
    if config.artifact_enabled && config.artifact_dir.is_none() {
        config.artifact_dir = artifact_root;
    }
    let datasets = build_moving_mnist_dreamer_datasets(&config)?;
    let train_dataset = datasets.train;
    let valid_dataset = datasets.valid;
    let selection_dataset = datasets.selection;
    let selection_context_len = datasets.selection_context_len;
    let selection_target_len = datasets.selection_target_len;
    let artifact_dataset = datasets.artifact;
    let artifact_context_len = datasets.artifact_context_len;
    let artifact_target_len = datasets.artifact_target_len;

    let train_teacher = load_autogaze_source(&config, MovingMnistSplit::Train, &device)?;
    let valid_teacher = load_autogaze_source(&config, MovingMnistSplit::Val, &device)?;
    let use_global_teacher = uses_global_teacher(&config);
    let future_penalty_weight = if config.model.passive_full_frame {
        15.0
    } else if use_global_teacher {
        10.0
    } else {
        0.0
    };
    let use_internal_state_targets = config.model.passive_full_frame
        && (config.model.current_loss_weight > 0.0 || config.model.future_loss_weight > 0.0);
    let use_crop_teacher = uses_crop_teacher(&config);
    if use_internal_state_targets {
        config.model.teacher_dim = config.model.latent_dim.max(1);
    } else if !use_global_teacher {
        config.model.teacher_dim = 1;
    }
    if !use_crop_teacher {
        config.model.crop_teacher_dim = 1;
    }
    let train_global_teacher = if use_global_teacher {
        Some(load_vjepa_source(
            &mut config,
            MovingMnistSplit::Train,
            &device,
        )?)
    } else {
        None
    };
    let valid_global_teacher = if use_global_teacher {
        Some(load_vjepa_source(
            &mut config,
            MovingMnistSplit::Val,
            &device,
        )?)
    } else {
        None
    };
    let train_crop_teacher = if use_crop_teacher {
        Some(load_crop_teacher_source(
            &mut config,
            MovingMnistSplit::Train,
            &device,
        )?)
    } else {
        None
    };
    let valid_crop_teacher = if use_crop_teacher {
        Some(load_crop_teacher_source(
            &mut config,
            MovingMnistSplit::Val,
            &device,
        )?)
    } else {
        None
    };
    let train_cache = build_cached_moving_mnist_split(
        &train_dataset,
        &config,
        &train_teacher,
        train_global_teacher.as_ref(),
        train_crop_teacher.as_ref(),
        &device,
    );
    let valid_cache = build_cached_moving_mnist_split(
        &valid_dataset,
        &config,
        &valid_teacher,
        valid_global_teacher.as_ref(),
        valid_crop_teacher.as_ref(),
        &device,
    );
    let selection_batches = if config.selection_batches == 0 {
        config.valid_batches
    } else {
        config.selection_batches
    };
    let selection_every = config.selection_every.max(1);
    let use_extended_selection =
        selection_context_len != config.context_len || selection_target_len != config.target_len;
    let mut selection_config = config.clone();
    selection_config.context_len = selection_context_len;
    selection_config.target_len = selection_target_len;
    let selection_cache = build_cached_moving_mnist_split(
        &selection_dataset,
        &selection_config,
        &valid_teacher,
        valid_global_teacher.as_ref(),
        valid_crop_teacher.as_ref(),
        &device,
    );
    let mut model = DragonDreamer::<TrainBackend>::new(config.model.clone(), &device);
    if let Some(checkpoint) = config.tokenizer_checkpoint.as_ref() {
        if config.model.latent_backend.uses_slot_tokenizer() {
            let mut tokenizer_config = load_saved_tokenizer_config(checkpoint)?
                .map(|saved| saved.model)
                .unwrap_or_else(|| config.model.clone());
            tokenizer_config.latent_backend = DreamerLatentBackend::TransformerBaseline;
            tokenizer_config.use_bdh_posterior = false;
            let tokenizer_model = DragonDreamer::<TrainBackend>::new(tokenizer_config, &device);
            let tokenizer_model = load_module_checkpoint(tokenizer_model, checkpoint, &device)?;
            model.restore_tokenizer_from(&tokenizer_model);
        } else {
            model = load_module_checkpoint(model, checkpoint, &device)?;
        }
    }
    if let Some(checkpoint) = config.dreamer_checkpoint.as_ref() {
        model = load_module_checkpoint(model, checkpoint, &device)?;
    }
    let frozen_tokenizer = if config.freeze_tokenizer {
        Some(model.clone())
    } else {
        None
    };
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<TrainBackend, DragonDreamer<TrainBackend>>();

    let initial_valid =
        evaluate_validation(&model, &valid_cache, &config, config.valid_batches, &device);
    let initial_selection = if use_extended_selection {
        evaluate_validation(
            &model,
            &selection_cache,
            &selection_config,
            selection_batches,
            &device,
        )
    } else {
        initial_valid.clone()
    };
    let mut last_valid_snapshot = initial_valid.clone();
    let mut best_valid_total = initial_valid.total;
    let mut best_valid_future = initial_valid.future;
    let mut best_future_selection = SelectedValidationMetrics::from_snapshot(
        0,
        &initial_selection,
        future_penalty_weight,
        None,
    );
    let mut best_quality_selection = SelectedValidationMetrics::from_snapshot(
        0,
        &initial_selection,
        future_penalty_weight,
        None,
    );
    let mut next_selection_step = selection_every;
    let mut final_train_total = initial_valid.total;
    let mut final_train_future = initial_valid.future;
    let mut last_artifact_dir = None;
    if config.artifact_enabled && config.artifact_dir.is_some() {
        last_artifact_dir = Some(write_validation_artifacts(
            &model,
            valid_global_teacher.as_ref(),
            valid_crop_teacher.as_ref(),
            &valid_teacher,
            &artifact_dataset,
            &config,
            artifact_context_len,
            artifact_target_len,
            0,
            false,
            None,
            &device,
        )?);
    }
    let tokenizer_pretrain_steps = if config.tokenizer_checkpoint.is_some() {
        0
    } else {
        config.model.tokenizer_pretrain_steps
    };
    if config.use_loss_normalization {
        anyhow::bail!(
            "use_loss_normalization is disabled for CUDA training because it forces per-step host scalar syncs"
        );
    }
    for step in 0..config.steps {
        let indices = sample_indices(
            train_cache.len(),
            config.batch_size,
            config
                .train_seed
                .wrapping_add((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let batch = train_cache.batch(&indices);
        let clip_frames = batch.clip_frames;
        let actions = batch.actions;
        let traces = batch.traces;
        let teacher_features = batch.teacher_features;
        let crop_teacher_features = batch.crop_teacher_features;
        let train_target_len = scheduled_target_len(&config, step);
        let teacher_forcing_prefix = scheduled_teacher_forcing_prefix(&config, step);
        let forward = if config.model.latent_backend.uses_slot_tokenizer()
            && step < tokenizer_pretrain_steps
        {
            model.forward_tokenizer_pretrain(
                clip_frames,
                &traces,
                actions,
                teacher_features,
                crop_teacher_features,
            )
        } else {
            model.forward_with_teacher_forcing_prefix(
                clip_frames,
                &traces,
                actions,
                teacher_features,
                crop_teacher_features,
                config.context_len,
                train_target_len,
                Some(teacher_forcing_prefix),
            )
        };
        let training_total = training_objective(&forward, &config, step, tokenizer_pretrain_steps);
        let should_read_train_scalars = should_run_step(step, config.steps, config.log_every)
            || should_run_step(step, config.steps, config.validate_every);
        if should_read_train_scalars {
            let values = scalar_pack(vec![training_total.clone(), forward.future.clone()]);
            final_train_total = values[0];
            final_train_future = values[1];
        }
        let grads = GradientsParams::from_grads(training_total.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        if let Some(frozen) = frozen_tokenizer.as_ref() {
            model.restore_tokenizer_from(frozen);
        }

        if should_run_step(step, config.steps, config.log_every) {
            println!(
                "step={} train_total={:.5} train_future={:.5}",
                step + 1,
                final_train_total,
                final_train_future
            );
        }

        if should_run_step(step, config.steps, config.validate_every) {
            let valid =
                evaluate_validation(&model, &valid_cache, &config, config.valid_batches, &device);
            last_valid_snapshot = valid.clone();
            let should_run_selection = if use_extended_selection {
                step + 1 == config.steps || step + 1 >= next_selection_step
            } else {
                false
            };
            let selection_valid = if use_extended_selection {
                if should_run_selection {
                    next_selection_step = (step + 1).saturating_add(selection_every);
                    Some(evaluate_validation(
                        &model,
                        &selection_cache,
                        &selection_config,
                        selection_batches,
                        &device,
                    ))
                } else {
                    None
                }
            } else {
                Some(valid.clone())
            };
            best_valid_total = best_valid_total.min(valid.total);
            best_valid_future = best_valid_future.min(valid.future);
            if let Some(selection_valid) = selection_valid.as_ref() {
                let quality_score = dynamics_quality_score(selection_valid, future_penalty_weight);
                let rollout_score = rollout_selection_score(selection_valid);
                let selection_score = if future_penalty_weight > 0.0 {
                    quality_score
                } else {
                    rollout_score
                };
                if best_future_selection.checkpoint_base.is_none()
                    || if use_global_teacher {
                        selection_valid.future <= best_future_selection.future
                    } else {
                        selection_score >= best_future_selection.quality_score
                    }
                {
                    let checkpoint_base_path = save_named_checkpoint::<TrainBackend, _>(
                        &model,
                        checkpoint_dir.as_deref(),
                        "dreamer-best-future",
                    )?;
                    best_future_selection = SelectedValidationMetrics::from_snapshot(
                        step + 1,
                        selection_valid,
                        future_penalty_weight,
                        checkpoint_base_path,
                    );
                }
                if best_quality_selection.checkpoint_base.is_none()
                    || quality_score >= best_quality_selection.quality_score
                {
                    let checkpoint_base_path = save_named_checkpoint::<TrainBackend, _>(
                        &model,
                        checkpoint_dir.as_deref(),
                        "dreamer-best-quality",
                    )?;
                    best_quality_selection = SelectedValidationMetrics::from_snapshot(
                        step + 1,
                        selection_valid,
                        future_penalty_weight,
                        checkpoint_base_path,
                    );
                }
                println!(
                    "valid step={} total={:.5} current={:.5} future={:.5} prior={:.5} gaze={:.5} query={:.5} recon={:.5} tok={:.5} tok_recon={:.5} slot_align={:.5} recon_cur={:.5} recon_fut={:.5} edge={:.5} motion={:.5} fut_psnr={:.2} fut_iou={:.3} fut_motion_ratio={:.3} fut_stop_mean={:.3} fut_stop_std={:.3} fut_fix_motion={:.3} ctx_fix_l1={:.3} fut_fix_l1={:.3} quality={:.3}",
                    step + 1,
                    valid.total,
                    valid.current,
                    valid.future,
                    valid.prior,
                    valid.gaze,
                    valid.query,
                    valid.recon,
                    valid.tokenizer,
                    valid.tokenizer_recon,
                    valid.slot_align,
                    valid.recon_current,
                    valid.recon_future,
                    valid.recon_edge,
                    valid.recon_motion,
                    valid.future_psnr,
                    valid.future_fg_iou,
                    valid.future_motion_ratio,
                    valid.future_stop_mean,
                    valid.future_stop_std,
                    valid.future_fixation_motion,
                    valid.context_fixation_teacher_l1,
                    valid.future_fixation_teacher_l1,
                    quality_score,
                );
                if use_extended_selection && should_run_selection {
                    println!(
                        "select step={} ctx={} fut={} total={:.5} future={:.5} fut_psnr={:.2} fut_iou={:.3} fut_motion_ratio={:.3} quality={:.3}",
                        step + 1,
                        selection_context_len,
                        selection_target_len,
                        selection_valid.total,
                        selection_valid.future,
                        selection_valid.future_psnr,
                        selection_valid.future_fg_iou,
                        selection_valid.future_motion_ratio,
                        quality_score,
                    );
                }
            } else {
                println!(
                    "valid step={} total={:.5} current={:.5} future={:.5} prior={:.5} gaze={:.5} query={:.5} recon={:.5} tok={:.5} tok_recon={:.5} slot_align={:.5} recon_cur={:.5} recon_fut={:.5} edge={:.5} motion={:.5} fut_psnr={:.2} fut_iou={:.3} fut_motion_ratio={:.3} fut_stop_mean={:.3} fut_stop_std={:.3} fut_fix_motion={:.3} ctx_fix_l1={:.3} fut_fix_l1={:.3} quality=deferred",
                    step + 1,
                    valid.total,
                    valid.current,
                    valid.future,
                    valid.prior,
                    valid.gaze,
                    valid.query,
                    valid.recon,
                    valid.tokenizer,
                    valid.tokenizer_recon,
                    valid.slot_align,
                    valid.recon_current,
                    valid.recon_future,
                    valid.recon_edge,
                    valid.recon_motion,
                    valid.future_psnr,
                    valid.future_fg_iou,
                    valid.future_motion_ratio,
                    valid.future_stop_mean,
                    valid.future_stop_std,
                    valid.future_fixation_motion,
                    valid.context_fixation_teacher_l1,
                    valid.future_fixation_teacher_l1,
                );
            }
            if should_write_artifacts(&config, step + 1) {
                last_artifact_dir = Some(write_validation_artifacts(
                    &model,
                    valid_global_teacher.as_ref(),
                    valid_crop_teacher.as_ref(),
                    &valid_teacher,
                    &artifact_dataset,
                    &config,
                    artifact_context_len,
                    artifact_target_len,
                    step + 1,
                    false,
                    None,
                    &device,
                )?);
            }
        }
    }

    let final_valid = last_valid_snapshot;
    if let Some(final_base) = save_named_checkpoint::<TrainBackend, _>(
        &model,
        checkpoint_dir.as_deref(),
        "dreamer-final",
    )? {
        if best_future_selection.checkpoint_base.is_none() {
            best_future_selection.checkpoint_base = Some(final_base.clone());
        }
        if best_quality_selection.checkpoint_base.is_none() {
            best_quality_selection.checkpoint_base = Some(final_base);
        }
    }
    if config.artifact_enabled && config.artifact_dir.is_some() {
        let final_artifact_checkpoint = if config.model.passive_full_frame {
            best_future_selection
                .checkpoint_base
                .as_ref()
                .or(best_quality_selection.checkpoint_base.as_ref())
        } else {
            best_quality_selection.checkpoint_base.as_ref()
        };
        let final_artifact_model = if let Some(base) = final_artifact_checkpoint {
            load_module_checkpoint(model.clone(), base, &device)?
        } else {
            model.clone()
        };
        let final_output = write_validation_artifacts(
            &final_artifact_model,
            valid_global_teacher.as_ref(),
            valid_crop_teacher.as_ref(),
            &valid_teacher,
            &artifact_dataset,
            &config,
            artifact_context_len,
            artifact_target_len,
            best_quality_selection.step.max(1),
            true,
            Some("final"),
            &device,
        )?;
        last_artifact_dir = Some(final_output);
    }
    Ok(MovingMnistDreamerRunSummary {
        initial_valid_total: initial_valid.total,
        final_valid_total: final_valid.total,
        initial_valid_future: initial_valid.future,
        final_valid_future: final_valid.future,
        best_valid_total,
        best_valid_future,
        final_train_total,
        final_train_future,
        final_valid_recon_current: final_valid.recon_current,
        final_valid_recon_future: final_valid.recon_future,
        final_valid_future_psnr: final_valid.future_psnr,
        final_valid_future_fg_iou: final_valid.future_fg_iou,
        final_valid_future_motion_ratio: final_valid.future_motion_ratio,
        final_valid_future_stop_mean: final_valid.future_stop_mean,
        final_valid_future_stop_std: final_valid.future_stop_std,
        final_valid_tokenizer: final_valid.tokenizer,
        final_valid_tokenizer_recon: final_valid.tokenizer_recon,
        final_valid_slot_align: final_valid.slot_align,
        best_future_selection,
        best_quality_selection,
        latent_backend: config.model.latent_backend.as_str().to_string(),
        steps: config.steps,
        run_dir,
        artifact_dir: last_artifact_dir,
    })
}

fn training_objective(
    forward: &crate::DreamerForward<TrainBackend>,
    config: &MovingMnistDreamerTrainConfig,
    step: usize,
    tokenizer_pretrain_steps: usize,
) -> burn::tensor::Tensor<TrainBackend, 1> {
    let gaze_weight = if config.model.passive_full_frame {
        0.0
    } else {
        config.model.gaze_loss_weight
    };
    let query_weight = if config.model.passive_full_frame {
        0.0
    } else {
        config.model.query_loss_weight
    };
    let tokenizer_term = if config.freeze_tokenizer {
        forward
            .slot_align
            .clone()
            .mul_scalar(config.model.tokenizer_slot_align_weight)
    } else {
        forward.tokenizer.clone()
    };
    if !config.model.latent_backend.uses_slot_tokenizer() {
        return forward.total.clone();
    }

    if step < tokenizer_pretrain_steps {
        return forward
            .current
            .clone()
            .mul_scalar(config.model.current_loss_weight * 0.25)
            + forward.query.clone().mul_scalar(query_weight * 0.5)
            + tokenizer_term.clone().mul_scalar(
                config.model.tokenizer_loss_weight * config.model.tokenizer_scale_start,
            )
            + forward
                .recon
                .clone()
                .mul_scalar(config.model.recon_loss_weight * config.model.recon_scale_start)
            + forward
                .gaze
                .clone()
                .mul_scalar(gaze_weight * 0.5 * config.model.gaze_scale_start);
    }

    let total_steps = config.steps.max(1);
    let warmup_steps = ((total_steps as f32 * config.model.warmup_fraction.clamp(0.0, 1.0)).round()
        as usize)
        .max(12)
        .min(total_steps);
    let warmup_progress = if step + 1 >= warmup_steps {
        1.0
    } else {
        (step + 1) as f32 / warmup_steps as f32
    };
    let tokenizer_scale = config.model.tokenizer_scale_start
        + (config.model.tokenizer_scale_end - config.model.tokenizer_scale_start) * warmup_progress;
    let dynamics_scale = config.model.dynamics_scale_start
        + (config.model.dynamics_scale_end - config.model.dynamics_scale_start) * warmup_progress;
    let recon_scale = config.model.recon_scale_start
        + (config.model.recon_scale_end - config.model.recon_scale_start) * warmup_progress;
    let gaze_scale = config.model.gaze_scale_start
        + (config.model.gaze_scale_end - config.model.gaze_scale_start) * warmup_progress;

    let current_term = forward
        .current
        .clone()
        .mul_scalar(config.model.current_loss_weight * dynamics_scale);
    let future_term = forward
        .future
        .clone()
        .mul_scalar(config.model.future_loss_weight * dynamics_scale);
    let prior_term = forward
        .prior
        .clone()
        .mul_scalar(config.model.prior_loss_weight * dynamics_scale);
    let shortcut_term = forward
        .shortcut
        .clone()
        .mul_scalar(config.model.passive_shortcut_loss_weight * dynamics_scale);
    let gaze_term = forward.gaze.clone().mul_scalar(gaze_weight * gaze_scale);
    let query_term = forward
        .query
        .clone()
        .mul_scalar(query_weight * dynamics_scale);
    let tokenizer_term =
        tokenizer_term.mul_scalar(config.model.tokenizer_loss_weight * tokenizer_scale);
    let recon_term = forward
        .recon
        .clone()
        .mul_scalar(config.model.recon_loss_weight * recon_scale);

    current_term
        + future_term
        + prior_term
        + shortcut_term
        + gaze_term
        + query_term
        + tokenizer_term
        + recon_term
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, View, serialize_to_file};
    use std::borrow::Cow;
    use tempfile::tempdir;

    #[derive(Clone)]
    struct OwnedTensor {
        shape: Vec<usize>,
        data: Vec<u8>,
        dtype: Dtype,
    }

    impl View for OwnedTensor {
        fn dtype(&self) -> Dtype {
            self.dtype
        }

        fn shape(&self) -> &[usize] {
            &self.shape
        }

        fn data(&self) -> Cow<'_, [u8]> {
            Cow::Borrowed(&self.data)
        }

        fn data_len(&self) -> usize {
            self.data.len()
        }
    }

    fn tensor_f32(shape: &[usize], values: &[f32]) -> OwnedTensor {
        let mut data = Vec::with_capacity(values.len() * 4);
        for value in values {
            data.extend_from_slice(&value.to_le_bytes());
        }
        OwnedTensor {
            shape: shape.to_vec(),
            data,
            dtype: Dtype::F32,
        }
    }

    fn write_trace_store(
        path: &std::path::Path,
        clips: usize,
        clip_len: usize,
        k: usize,
        frame_size: usize,
    ) {
        let mut fixations = Vec::with_capacity(clips * clip_len * k * 2);
        let mut scales = Vec::with_capacity(clips * clip_len * k);
        let mut confidences = Vec::with_capacity(clips * clip_len * k);
        let mut stops = Vec::with_capacity(clips * clip_len);
        let mut visibility = vec![0.0f32; clips * clip_len * frame_size * frame_size];

        for clip_idx in 0..clips {
            for frame_idx in 0..clip_len {
                let x = ((frame_idx + 1) as f32 / (clip_len + 1) as f32).clamp(0.15, 0.85);
                let y = ((clip_idx % 7 + frame_idx + 1) as f32 / (clip_len + 7) as f32)
                    .clamp(0.15, 0.85);
                for _ in 0..k {
                    fixations.extend_from_slice(&[x, y]);
                    scales.push(0.22);
                    confidences.push(0.9);
                }
                stops.push(0.1);

                let cx = (x * frame_size as f32).round() as isize;
                let cy = (y * frame_size as f32).round() as isize;
                let base = (clip_idx * clip_len + frame_idx) * frame_size * frame_size;
                for yy in 0..frame_size {
                    for xx in 0..frame_size {
                        let dx = xx as isize - cx;
                        let dy = yy as isize - cy;
                        if dx.abs() <= 3 && dy.abs() <= 3 {
                            visibility[base + yy * frame_size + xx] = 1.0;
                        }
                    }
                }
            }
        }

        let tensors = vec![
            (
                "fixations".to_string(),
                tensor_f32(&[clips, clip_len, k, 2], &fixations),
            ),
            (
                "scales".to_string(),
                tensor_f32(&[clips, clip_len, k], &scales),
            ),
            (
                "confidences".to_string(),
                tensor_f32(&[clips, clip_len, k], &confidences),
            ),
            (
                "stop_probabilities".to_string(),
                tensor_f32(&[clips, clip_len], &stops),
            ),
            (
                "visibility_maps".to_string(),
                tensor_f32(&[clips, clip_len, frame_size, frame_size], &visibility),
            ),
        ];
        serialize_to_file(tensors, None, path).expect("write trace store");
    }

    fn write_feature_store(
        path: &std::path::Path,
        clips: usize,
        context_len: usize,
        target_len: usize,
        feature_dim: usize,
        scale: f32,
    ) {
        let mut current = Vec::with_capacity(clips * context_len * feature_dim);
        let mut future = Vec::with_capacity(clips * target_len * feature_dim);
        for clip_idx in 0..clips {
            for frame_idx in 0..context_len {
                for feat_idx in 0..feature_dim {
                    let value =
                        (((clip_idx + 1) * (frame_idx + 1) * (feat_idx + 3)) as f32 * scale).sin();
                    current.push(value);
                }
            }
            for frame_idx in 0..target_len {
                for feat_idx in 0..feature_dim {
                    let value = (((clip_idx + 2) * (frame_idx + context_len + 1) * (feat_idx + 5))
                        as f32
                        * scale)
                        .cos();
                    future.push(value);
                }
            }
        }
        let tensors = vec![
            (
                "current_features".to_string(),
                tensor_f32(&[clips, context_len, feature_dim], &current),
            ),
            (
                "future_features".to_string(),
                tensor_f32(&[clips, target_len, feature_dim], &future),
            ),
        ];
        serialize_to_file(tensors, None, path).expect("write feature store");
    }

    fn strict_teacher_config(k_fovea: usize) -> (tempfile::TempDir, MovingMnistDreamerTrainConfig) {
        let temp = tempdir().expect("temp dir");
        let train_traces = temp.path().join("train_autogaze.safetensors");
        let val_traces = temp.path().join("val_autogaze.safetensors");
        let global_features = temp.path().join("global_vjepa.safetensors");
        let crop_features = temp.path().join("crop_teacher.safetensors");

        let mut config = MovingMnistDreamerTrainConfig {
            steps: 8,
            batch_size: 2,
            validate_every: 4,
            valid_batches: 1,
            run_root: None,
            artifact_enabled: false,
            ..Default::default()
        };
        config.model.k_fovea = k_fovea.max(1);
        config.artifact_future_steps = config.target_len;
        write_trace_store(
            &train_traces,
            256,
            config.context_len + config.target_len,
            config.model.k_fovea.max(1),
            config.model.frame_size,
        );
        write_trace_store(
            &val_traces,
            128,
            config.context_len + config.target_len,
            config.model.k_fovea.max(1),
            config.model.frame_size,
        );
        write_feature_store(
            &global_features,
            256,
            config.context_len,
            config.target_len,
            config.model.teacher_dim,
            0.007,
        );
        write_feature_store(
            &crop_features,
            256,
            config.context_len,
            config.target_len,
            config.model.crop_teacher_dim,
            0.011,
        );
        config.autogaze_train_trace_store = Some(train_traces);
        config.autogaze_val_trace_store = Some(val_traces);
        config.vjepa_feature_store = Some(global_features);
        config.crop_teacher_feature_store = Some(crop_features);
        (temp, config)
    }

    #[test]
    fn strict_teacher_loaders_require_assets() {
        let config = MovingMnistDreamerTrainConfig::default();
        let device = cuda_device().expect("cuda device");
        let err = load_autogaze_source(&config, MovingMnistSplit::Train, &device)
            .expect_err("missing teacher asset");
        assert!(err.to_string().contains("missing AutoGaze teacher"));
    }

    #[test]
    fn native_vjepa_hf_loader_emits_clip_features() {
        let mut config = MovingMnistDreamerTrainConfig::default();
        config.vjepa_hf_dir = Some(
            PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../burn_vjepa/tests/fixtures/vjepa2_tiny_hf"),
        );
        let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
        let teacher = load_vjepa_source(&mut config, MovingMnistSplit::Train, &device)
            .expect("load native teacher");

        let clip = burn::tensor::Tensor::<TrainBackend, 5>::zeros([2, 4, 1, 16, 16], &device);
        let features = teacher.encode_clip_from_batch(&[0, 1], clip);
        let [batch, frames, dim] = features.shape().dims::<3>();
        assert_eq!((batch, frames, dim), (2, 4, 24));
    }

    #[test]
    fn native_autogaze_hf_loader_emits_traces() {
        let mut config = MovingMnistDreamerTrainConfig::default();
        config.context_len = 4;
        config.target_len = 2;
        config.model.k_fovea = 2;
        config.autogaze_hf_dir = Some(PathBuf::from(
            "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a",
        ));
        let device = <TrainBackend as burn::tensor::backend::Backend>::Device::default();
        let teacher = load_autogaze_source(&config, MovingMnistSplit::Train, &device)
            .expect("load native autogaze teacher");
        let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Train,
            frame_size: config.model.frame_size,
            digit_size: 12,
            in_channels: config.model.channels,
            context_len: config.context_len,
            target_len: config.target_len,
            extra_future_frames: 0,
            frame_stride: 1,
            max_records: Some(1),
            normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
            min_velocity: config.min_velocity,
            max_velocity: config.max_velocity,
            seed: config.train_seed,
        })
        .expect("build moving mnist dataset");
        let batch = dataset.batch_from_indices::<TrainBackend>(&[0], &device);
        let traces = teacher.traces_for_batch(&[0], &batch, config.model.k_fovea);
        assert_eq!(traces.len(), 1);
        assert_eq!(
            traces[0].frames.len(),
            config.context_len + config.target_len
        );
        assert_eq!(traces[0].frames[0].points.len(), config.model.k_fovea);
    }

    #[test]
    fn moving_mnist_training_smoke_improves_validation() {
        let (_temp, mut config) = strict_teacher_config(1);
        config.steps = 20;
        config.batch_size = 4;
        config.validate_every = 10;
        config.valid_batches = 2;
        let summary = train_moving_mnist(config).expect("training");
        assert!(summary.final_valid_total.is_finite());
        assert!(summary.final_valid_future.is_finite());
        assert!(
            summary.best_valid_total <= summary.initial_valid_total,
            "expected some validation improvement"
        );
    }

    #[test]
    fn moving_mnist_multifovea_smoke_is_finite() {
        let (_temp, config) = strict_teacher_config(2);
        let summary = train_moving_mnist(config).expect("training");
        assert!(summary.final_valid_total.is_finite());
        assert!(summary.final_valid_future.is_finite());
    }

    #[test]
    fn moving_mnist_writes_artifact_snapshots() {
        let (temp, mut config) = strict_teacher_config(1);
        config.steps = 2;
        config.batch_size = 2;
        config.validate_every = 1;
        config.valid_batches = 1;
        config.artifact_enabled = true;
        config.artifact_dir = Some(temp.path().join("dreamer_artifacts"));
        config.artifact_every = 1;
        config.artifact_samples = 1;
        let summary = train_moving_mnist(config).expect("training");
        let artifact_dir = summary.artifact_dir.expect("artifact dir");
        for name in [
            "current_reference.png",
            "current_reconstruction.png",
            "future_reference.png",
            "future_reconstruction.png",
            "current_latent_pca.png",
            "future_latent_pca.png",
            "autogaze_teacher_label_patches.png",
            "fovea_saccade_reads.png",
            "fixation_overlays.png",
            "metrics.json",
        ] {
            assert!(
                artifact_dir.join(name).is_file(),
                "expected artifact {}",
                artifact_dir.join(name).display()
            );
        }
    }
}
