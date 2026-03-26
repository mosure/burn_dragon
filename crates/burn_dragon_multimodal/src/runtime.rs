#![cfg(feature = "train")]

use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_dragon_train::api::expert::train::pipeline::{activate_planned_run, plan_run_artifacts};
use burn_dragon_train::api::runtime::cleanup_device_memory;

use crate::checkpoint::write_training_snapshot as write_model_snapshot;
use crate::ema::{init_momentum_teacher, sync_optional_teacher_from_student};
use crate::runtime_config::resolve_multimodal_backend_run_root;
pub use crate::runtime_config::*;
use crate::runtime_data::{
    batch_ranges, contiguous_batch_indices, load_image_text_dataset_bundle,
    load_video_text_dataset_bundle, prepare_target_bank, target_bank_batch_for_targets,
    unique_target_batches,
};
use crate::runtime_init::{resolve_image_pretrained_init, resolve_video_pretrained_init};
use crate::runtime_metrics::{
    bidirectional_top1_accuracy, diagonal_similarity_mean, labeled_top1_accuracy,
    run_image_text_validation_epoch, run_video_text_validation_epoch, scalar_from_tensor,
};
use crate::train::{
    collate_video_language_segments, collate_vision_language_segments,
    multimodal_train_step_with_frozen_cores, multimodal_video_train_step_with_frozen_cores,
};

fn maybe_cleanup_multimodal_device<B: BackendTrait>(device: &B::Device, step_count: usize) {
    if step_count == 0 {
        return;
    }
    let _ = cleanup_device_memory::<B>(device, false);
}

pub fn train_backend<B, Init>(
    config: &MultimodalTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let run_root = resolve_multimodal_backend_run_root(
        &config.run_layout,
        &config.training.run_root,
        backend_name,
    );
    let planned_run = plan_run_artifacts(&run_root, None)?;
    activate_planned_run(&planned_run)?;
    let mut report = run_image_text_training_backend::<B, _>(config, &planned_run.run_dir, init)?;
    report.run_name = planned_run.run_name;
    Ok(report)
}

pub fn train_video_backend<B, Init>(
    config: &MultimodalVideoTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let run_root = resolve_multimodal_backend_run_root(
        &config.run_layout,
        &config.training.run_root,
        backend_name,
    );
    let planned_run = plan_run_artifacts(&run_root, None)?;
    activate_planned_run(&planned_run)?;
    let mut report = run_video_text_training_backend::<B, _>(config, &planned_run.run_dir, init)?;
    report.run_name = planned_run.run_name;
    Ok(report)
}

pub fn run_image_text_training_backend<B, Init>(
    config: &MultimodalTrainingConfig,
    run_dir: &Path,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    fs::create_dir_all(artifact_dir(run_dir)).with_context(|| {
        format!(
            "failed to create artifact directory {}",
            artifact_dir(run_dir).display()
        )
    })?;
    fs::create_dir_all(run_dir.join("checkpoint")).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            run_dir.join("checkpoint").display()
        )
    })?;

    write_runtime_snapshot(run_dir, config)?;
    let device = B::Device::default();
    init(&device);
    B::seed(&device, config.training.seed);
    let resolved_init = resolve_image_pretrained_init::<B>(config, &device)?;
    write_model_snapshot(run_dir, &resolved_init.model_config)?;
    let bundle = load_image_text_dataset_bundle(config, resolved_init.pretrained_vocab.as_ref())?;
    if !bundle.supports_batched_contrastive && config.training.batch_size != 1 {
        return Err(anyhow!(
            "multimodal image-text source {:?} requires training.batch_size = 1 because stream resets/detaches are explicit per segment",
            config.data.source
        ));
    }
    let vocab = bundle.vocab;
    let vocab_path = tokenizer_snapshot_path(run_dir);
    vocab.save(&vocab_path)?;
    let prepared_target_bank =
        prepare_target_bank::<B>(bundle.target_bank_texts.as_deref(), &vocab, &device);
    let mut model =
        crate::model::VlJepaDragon::<B>::new(resolved_init.model_config.clone(), &device);
    if let Some(vision_x_encoder) = resolved_init.vision_x_encoder {
        model.vision_x_encoder.replace_encoder(vision_x_encoder);
        model.vision_x_encoder.set_force_projection(true);
    }
    if let Some(text_core) = resolved_init.text_core {
        model.query_q_encoder.replace_encoder(text_core.clone());
        model.target_y_encoder.replace_encoder(text_core);
        model.query_q_encoder.set_force_projection(true);
        model.target_y_encoder.set_force_projection(true);
    }
    if let Some(fusion_core) = resolved_init.fusion_core {
        model.replace_fusion_core(fusion_core);
    }
    model.set_frozen_modalities(
        resolved_init.freeze_vision_x_encoder,
        resolved_init.freeze_query_q_encoder,
        resolved_init.freeze_target_y_encoder,
    );
    let frozen_cores = model.frozen_core_set();
    let mut target_teacher = init_momentum_teacher::<B, _>(
        &model.target_y_encoder,
        &resolved_init.model_config.target_teacher,
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, crate::model::VlJepaDragon<B>>();

    let mut checkpoint_paths = Vec::new();
    let mut artifact_paths = Vec::new();
    let mut epochs = Vec::new();

    for epoch in 0..config.training.epochs {
        let mut loss_sum = 0.0_f32;
        let mut diagonal_similarity_sum = 0.0_f32;
        let mut top1_accuracy_sum = 0.0_f32;
        let mut steps = 0_usize;
        if bundle.supports_batched_contrastive {
            let batch_indices =
                if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                    contiguous_batch_indices(
                        bundle.train.len(),
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                    )
                } else {
                    unique_target_batches(
                        &bundle.train,
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                        |segment| &segment.payload.target_y_tokens,
                    )
                };
            for batch_indices in batch_indices {
                let segments = batch_indices
                    .into_iter()
                    .map(|index| bundle.train[index].clone())
                    .collect::<Vec<_>>();
                let collated = collate_vision_language_segments::<B>(&segments, &device);
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    segments
                        .iter()
                        .map(|segment| segment.payload.target_y_tokens.as_slice()),
                    &device,
                );
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated batch missing stream metadata"))?;
                let step = multimodal_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    model.init_state(),
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = match target_bank {
                    Some(target_bank) => labeled_top1_accuracy(
                        step.loss.similarities.clone(),
                        target_bank.target_indices,
                    ),
                    None => bidirectional_top1_accuracy(step.loss.similarities.clone()),
                };
                let loss = step.loss;
                let forward = step.forward;
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        } else {
            let mut state = model.init_state();
            for range in batch_ranges(bundle.train.len(), 1, config.training.max_steps_per_epoch) {
                let segment = bundle.train[range.start].clone();
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    [segment.payload.target_y_tokens.as_slice()],
                    &device,
                );
                let collated =
                    collate_vision_language_segments::<B>(std::slice::from_ref(&segment), &device);
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated batch missing stream metadata"))?;
                let step = multimodal_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    state,
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = target_bank
                    .map(|target_bank| {
                        labeled_top1_accuracy(
                            step.loss.similarities.clone(),
                            target_bank.target_indices,
                        )
                    })
                    .unwrap_or_else(|| bidirectional_top1_accuracy(step.loss.similarities.clone()));
                let loss = step.loss;
                let forward = step.forward;
                let next_state = forward.state.detach();
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );
                state = next_state;

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        }

        let validation = if let Some(validation) = bundle.validation.as_ref() {
            Some(run_image_text_validation_epoch::<B>(
                &model,
                target_teacher.as_ref(),
                prepared_target_bank.as_ref(),
                validation,
                &device,
                config,
                &resolved_init.model_config,
                bundle.supports_batched_contrastive,
                true,
            )?)
        } else {
            None
        };

        let epoch_index = epoch + 1;
        let artifact = MultimodalEpochArtifact {
            epoch: epoch_index,
            steps,
            mean_total_loss: loss_sum / steps.max(1) as f32,
            mean_diagonal_similarity: diagonal_similarity_sum / steps.max(1) as f32,
            mean_top1_accuracy: top1_accuracy_sum / steps.max(1) as f32,
            validation_steps: validation
                .as_ref()
                .map(|metrics| metrics.steps)
                .unwrap_or(0),
            validation_mean_total_loss: validation.as_ref().map(|metrics| metrics.mean_total_loss),
            validation_mean_diagonal_similarity: validation
                .as_ref()
                .map(|metrics| metrics.mean_diagonal_similarity),
            validation_mean_top1_accuracy: validation
                .as_ref()
                .map(|metrics| metrics.mean_top1_accuracy),
            validation_refine_curve: validation
                .as_ref()
                .and_then(|metrics| metrics.refine_curve.clone()),
        };
        epochs.push(artifact.clone());

        if config.training.artifact_every_epochs > 0
            && epoch_index % config.training.artifact_every_epochs == 0
        {
            let artifact_path = artifact_dir(run_dir).join(format!("epoch-{epoch_index}.json"));
            let payload =
                serde_json::to_string_pretty(&artifact).context("serialize epoch artifact")?;
            fs::write(&artifact_path, payload)
                .with_context(|| format!("failed to write {}", artifact_path.display()))?;
            artifact_paths.push(artifact_path);
        }

        if config.training.checkpoint_every_epochs > 0
            && epoch_index % config.training.checkpoint_every_epochs == 0
        {
            let checkpoint_base = run_dir.join("checkpoint").join(format!("model-{epoch}"));
            BinFileRecorder::<FullPrecisionSettings>::new()
                .record(model.clone().into_record(), checkpoint_base.clone())
                .with_context(|| {
                    format!("failed to write checkpoint {}", checkpoint_base.display())
                })?;
            checkpoint_paths.push(checkpoint_base.with_extension("bin"));
        }
    }

    Ok(MultimodalTrainingReport {
        run_dir: run_dir.to_path_buf(),
        run_name: run_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "multimodal-run".to_string()),
        checkpoint_paths,
        artifact_paths,
        epochs,
        tokenizer_path: vocab_path,
    })
}

pub fn run_video_text_training_backend<B, Init>(
    config: &MultimodalVideoTrainingConfig,
    run_dir: &Path,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    fs::create_dir_all(artifact_dir(run_dir)).with_context(|| {
        format!(
            "failed to create artifact directory {}",
            artifact_dir(run_dir).display()
        )
    })?;
    fs::create_dir_all(run_dir.join("checkpoint")).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            run_dir.join("checkpoint").display()
        )
    })?;

    write_video_runtime_snapshot(run_dir, config)?;
    let device = B::Device::default();
    init(&device);
    B::seed(&device, config.training.seed);
    let resolved_init = resolve_video_pretrained_init::<B>(config, &device)?;
    write_model_snapshot(run_dir, &resolved_init.model_config)?;
    let bundle = load_video_text_dataset_bundle(config, resolved_init.pretrained_vocab.as_ref())?;
    if !bundle.supports_batched_contrastive && config.training.batch_size != 1 {
        return Err(anyhow!(
            "multimodal video-text source {:?} requires training.batch_size = 1 because stream resets/detaches are explicit per segment",
            config.data.source
        ));
    }
    let vocab = bundle.vocab;
    let vocab_path = tokenizer_snapshot_path(run_dir);
    vocab.save(&vocab_path)?;
    let prepared_target_bank =
        prepare_target_bank::<B>(bundle.target_bank_texts.as_deref(), &vocab, &device);
    let mut model =
        crate::model::VlJepaDragon::<B>::new(resolved_init.model_config.clone(), &device);
    if let Some(vision_x_encoder) = resolved_init.vision_x_encoder {
        model.vision_x_encoder.replace_encoder(vision_x_encoder);
        model.vision_x_encoder.set_force_projection(true);
    }
    if let Some(text_core) = resolved_init.text_core {
        model.query_q_encoder.replace_encoder(text_core.clone());
        model.target_y_encoder.replace_encoder(text_core);
        model.query_q_encoder.set_force_projection(true);
        model.target_y_encoder.set_force_projection(true);
    }
    if let Some(fusion_core) = resolved_init.fusion_core {
        model.replace_fusion_core(fusion_core);
    }
    model.set_frozen_modalities(
        resolved_init.freeze_vision_x_encoder,
        resolved_init.freeze_query_q_encoder,
        resolved_init.freeze_target_y_encoder,
    );
    let frozen_cores = model.frozen_core_set();
    let mut target_teacher = init_momentum_teacher::<B, _>(
        &model.target_y_encoder,
        &resolved_init.model_config.target_teacher,
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, crate::model::VlJepaDragon<B>>();

    let mut checkpoint_paths = Vec::new();
    let mut artifact_paths = Vec::new();
    let mut epochs = Vec::new();

    for epoch in 0..config.training.epochs {
        let mut loss_sum = 0.0_f32;
        let mut diagonal_similarity_sum = 0.0_f32;
        let mut top1_accuracy_sum = 0.0_f32;
        let mut steps = 0_usize;
        if bundle.supports_batched_contrastive {
            let batch_indices =
                if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                    contiguous_batch_indices(
                        bundle.train.len(),
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                    )
                } else {
                    unique_target_batches(
                        &bundle.train,
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                        |segment| &segment.payload.target_y_tokens,
                    )
                };
            for batch_indices in batch_indices {
                let segments = batch_indices
                    .into_iter()
                    .map(|index| bundle.train[index].clone())
                    .collect::<Vec<_>>();
                let collated = collate_video_language_segments::<B>(&segments, &device);
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    segments
                        .iter()
                        .map(|segment| segment.payload.target_y_tokens.as_slice()),
                    &device,
                );
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated video batch missing stream metadata"))?;
                let step = multimodal_video_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    model.init_state(),
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = match target_bank {
                    Some(target_bank) => labeled_top1_accuracy(
                        step.loss.similarities.clone(),
                        target_bank.target_indices,
                    ),
                    None => bidirectional_top1_accuracy(step.loss.similarities.clone()),
                };
                let loss = step.loss;
                let forward = step.forward;
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        } else {
            let mut state = model.init_state();
            for range in batch_ranges(bundle.train.len(), 1, config.training.max_steps_per_epoch) {
                let segment = bundle.train[range.start].clone();
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    [segment.payload.target_y_tokens.as_slice()],
                    &device,
                );
                let collated =
                    collate_video_language_segments::<B>(std::slice::from_ref(&segment), &device);
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated video batch missing stream metadata"))?;
                let step = multimodal_video_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    state,
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = target_bank
                    .map(|target_bank| {
                        labeled_top1_accuracy(
                            step.loss.similarities.clone(),
                            target_bank.target_indices,
                        )
                    })
                    .unwrap_or_else(|| bidirectional_top1_accuracy(step.loss.similarities.clone()));
                let loss = step.loss;
                let forward = step.forward;
                let next_state = forward.state.detach();
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );
                state = next_state;

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        }

        let validation = if let Some(validation) = bundle.validation.as_ref() {
            Some(run_video_text_validation_epoch::<B>(
                &model,
                target_teacher.as_ref(),
                prepared_target_bank.as_ref(),
                validation,
                &device,
                config,
                &resolved_init.model_config,
                bundle.supports_batched_contrastive,
                true,
            )?)
        } else {
            None
        };

        let epoch_index = epoch + 1;
        let artifact = MultimodalEpochArtifact {
            epoch: epoch_index,
            steps,
            mean_total_loss: loss_sum / steps.max(1) as f32,
            mean_diagonal_similarity: diagonal_similarity_sum / steps.max(1) as f32,
            mean_top1_accuracy: top1_accuracy_sum / steps.max(1) as f32,
            validation_steps: validation
                .as_ref()
                .map(|metrics| metrics.steps)
                .unwrap_or(0),
            validation_mean_total_loss: validation.as_ref().map(|metrics| metrics.mean_total_loss),
            validation_mean_diagonal_similarity: validation
                .as_ref()
                .map(|metrics| metrics.mean_diagonal_similarity),
            validation_mean_top1_accuracy: validation
                .as_ref()
                .map(|metrics| metrics.mean_top1_accuracy),
            validation_refine_curve: validation
                .as_ref()
                .and_then(|metrics| metrics.refine_curve.clone()),
        };
        epochs.push(artifact.clone());

        if config.training.artifact_every_epochs > 0
            && epoch_index % config.training.artifact_every_epochs == 0
        {
            let artifact_path = artifact_dir(run_dir).join(format!("epoch-{epoch_index}.json"));
            let payload =
                serde_json::to_string_pretty(&artifact).context("serialize epoch artifact")?;
            fs::write(&artifact_path, payload)
                .with_context(|| format!("failed to write {}", artifact_path.display()))?;
            artifact_paths.push(artifact_path);
        }

        if config.training.checkpoint_every_epochs > 0
            && epoch_index % config.training.checkpoint_every_epochs == 0
        {
            let checkpoint_base = run_dir.join("checkpoint").join(format!("model-{epoch}"));
            BinFileRecorder::<FullPrecisionSettings>::new()
                .record(model.clone().into_record(), checkpoint_base.clone())
                .with_context(|| {
                    format!("failed to write checkpoint {}", checkpoint_base.display())
                })?;
            checkpoint_paths.push(checkpoint_base.with_extension("bin"));
        }
    }

    Ok(MultimodalTrainingReport {
        run_dir: run_dir.to_path_buf(),
        run_name: run_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "multimodal-run".to_string()),
        checkpoint_paths,
        artifact_paths,
        epochs,
        tokenizer_path: vocab_path,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    use burn_autodiff::Autodiff;
    use burn_dragon_stream::TargetAlignmentPolicy;
    use burn_ndarray::NdArray;
    use image::RgbImage;
    use tempfile::tempdir;

    use crate::train::{VideoLanguageJsonlRecord, VisionLanguageJsonlRecord};

    type Backend = Autodiff<NdArray<f32>>;

    #[test]
    fn runs_image_text_training_and_writes_checkpoints_and_artifacts() {
        let dir = tempdir().expect("tempdir");
        let image_path = dir.path().join("sample.png");
        let mut image = RgbImage::new(8, 8);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 8) as u8, (y * 8) as u8, 127]);
        }
        image.save(&image_path).expect("save image");

        let manifest = dir.path().join("dataset.jsonl");
        fs::write(
            &manifest,
            [
                serde_json::to_string(&VisionLanguageJsonlRecord {
                    image_path: PathBuf::from("sample.png"),
                    query_q_text: "what".to_string(),
                    target_y_text: "color".to_string(),
                    source_id: 1,
                    episode_id: 1,
                    segment_id: 0,
                    step_index: 0,
                    absolute_time: 0,
                    boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
                })
                .expect("record 0"),
                serde_json::to_string(&VisionLanguageJsonlRecord {
                    image_path: PathBuf::from("sample.png"),
                    query_q_text: "what".to_string(),
                    target_y_text: "shape".to_string(),
                    source_id: 1,
                    episode_id: 1,
                    segment_id: 1,
                    step_index: 1,
                    absolute_time: 1,
                    boundary: burn_dragon_stream::StreamBoundary::Continue,
                })
                .expect("record 1"),
            ]
            .join("\n"),
        )
        .expect("write manifest");

        let mut config = MultimodalTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(2);

        let run_dir = dir.path().join("run");
        let report = run_image_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run multimodal training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
        assert!(run_dir.join("multimodal_vl_jepa_config.json").is_file());
    }

    #[test]
    fn train_backend_creates_named_run_under_backend_root() {
        let dir = tempdir().expect("tempdir");
        let image_path = dir.path().join("sample.png");
        let mut image = RgbImage::new(8, 8);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 8) as u8, (y * 8) as u8, 127]);
        }
        image.save(&image_path).expect("save image");

        let manifest = dir.path().join("dataset.jsonl");
        fs::write(
            &manifest,
            serde_json::to_string(&VisionLanguageJsonlRecord {
                image_path: PathBuf::from("sample.png"),
                query_q_text: "what".to_string(),
                target_y_text: "color".to_string(),
                source_id: 1,
                episode_id: 1,
                segment_id: 0,
                step_index: 0,
                absolute_time: 0,
                boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
            })
            .expect("record"),
        )
        .expect("write manifest");

        let mut config = MultimodalTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(1);
        config.training.run_root = dir.path().join("runs");

        let report = train_backend::<Backend, _>(&config, "cpu", |_| {}).expect("train backend");
        assert!(
            report
                .run_dir
                .starts_with(dir.path().join("runs").join("cpu"))
        );
        assert!(!report.run_name.is_empty());
        assert!(report.run_dir.join("checkpoint").is_dir());
    }

    #[test]
    fn run_layout_base_dir_overrides_legacy_run_root_for_backend_partition() {
        let dir = tempdir().expect("tempdir");
        let mut config = MultimodalTrainingConfig::default();
        config.training.run_root = dir.path().join("legacy-runs");
        config.run_layout.base_dir = Some(dir.path().join("shared-runs"));
        config.run_layout.mirror_config_path = false;

        assert_eq!(
            resolve_multimodal_backend_run_root(
                &config.run_layout,
                &config.training.run_root,
                "cuda"
            ),
            dir.path()
                .join("shared-runs")
                .join("multimodal")
                .join("cuda")
        );
    }

    #[test]
    fn runs_video_text_training_and_writes_checkpoints_and_artifacts() {
        let dir = tempdir().expect("tempdir");
        let mut records = Vec::new();
        for frame in 0..6 {
            let image_path = dir.path().join(format!("frame-{frame}.png"));
            let mut image = RgbImage::new(8, 8);
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                *pixel = image::Rgb([(x * 8 + frame) as u8, (y * 8) as u8, 127]);
            }
            image.save(&image_path).expect("save image");
        }
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-0.png"),
            query_q_text: "look".into(),
            target_y_text: "zero".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 0,
            step_index: 0,
            absolute_time: 0,
            boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-1.png"),
            query_q_text: "look".into(),
            target_y_text: "one".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 1,
            step_index: 1,
            absolute_time: 1,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-2.png"),
            query_q_text: "look".into(),
            target_y_text: "two".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 2,
            step_index: 2,
            absolute_time: 2,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-3.png"),
            query_q_text: "look".into(),
            target_y_text: "three".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 3,
            step_index: 3,
            absolute_time: 3,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-4.png"),
            query_q_text: "new".into(),
            target_y_text: "four".into(),
            source_id: 1,
            episode_id: 2,
            segment_id: 0,
            step_index: 0,
            absolute_time: 4,
            boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-5.png"),
            query_q_text: "new".into(),
            target_y_text: "five".into(),
            source_id: 1,
            episode_id: 2,
            segment_id: 1,
            step_index: 1,
            absolute_time: 5,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });

        let manifest = dir.path().join("video.jsonl");
        fs::write(
            &manifest,
            records
                .iter()
                .map(|record| serde_json::to_string(record).expect("record json"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("write manifest");

        let mut config = MultimodalVideoTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.model.tbptt.target_alignment_policy = TargetAlignmentPolicy::VariableFuture;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.data.clip_frames = 2;
        config.data.requested_horizons = vec![1, 2];
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(2);

        let run_dir = dir.path().join("video-run");
        let report = run_video_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run multimodal video training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "video-run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
    }

    #[test]
    fn runs_mnist_video_text_training_and_writes_checkpoints_and_artifacts() {
        type Backend = Autodiff<NdArray<f32>>;
        let dir = tempdir().expect("tempdir");
        let mut config = MultimodalVideoTrainingConfig::default();
        config.model.vision.image_size = 32;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.source = MultimodalVideoTextSource::MnistLabelText;
        config.data.clip_frames = 2;
        config.data.mnist.max_train_records = Some(16);
        config.data.mnist.max_validation_records = Some(8);
        config.training.epochs = 1;
        config.training.batch_size = 4;
        config.training.max_steps_per_epoch = Some(2);
        config.training.max_validation_steps_per_epoch = Some(1);

        let run_dir = dir.path().join("mnist-video-run");
        let report = run_video_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run MNIST video-text training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "mnist-video-run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
    }

    #[test]
    fn unique_target_batches_distribute_duplicate_targets_across_batches() {
        let items = vec![
            vec![1_i64],
            vec![2_i64],
            vec![1_i64],
            vec![2_i64],
            vec![3_i64],
            vec![3_i64],
        ];
        let batches = unique_target_batches(&items, 3, None, |tokens| tokens);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![0, 1, 4]);
        assert_eq!(batches[1], vec![2, 3, 5]);
    }
}
