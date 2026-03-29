use super::teacher::{
    AutoGazeSource, ClipTeacherSource, encode_crop_teacher, zero_teacher_features,
};
use crate::MovingMnistDreamerTrainConfig;
use crate::data::{CachedSequenceSplit, build_cached_sequence_split};
use crate::runtime::TrainBackend;
use anyhow::Result;
use burn_autogaze::FrameFixationTrace;
use burn_dragon_vision::{
    MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VisionNormalize,
};

pub(crate) struct MovingMnistDreamerDatasets {
    pub train: MovingMnistVideoDataset,
    pub valid: MovingMnistVideoDataset,
    pub selection: MovingMnistVideoDataset,
    pub selection_context_len: usize,
    pub selection_target_len: usize,
    pub artifact: MovingMnistVideoDataset,
    pub artifact_context_len: usize,
    pub artifact_target_len: usize,
}

pub(crate) struct MovingMnistTokenizerDatasets {
    pub train: MovingMnistVideoDataset,
    pub valid: MovingMnistVideoDataset,
}

pub(crate) fn build_moving_mnist_dreamer_datasets(
    config: &MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistDreamerDatasets> {
    let train = build_dataset(
        config,
        MovingMnistSplit::Train,
        config.context_len,
        config.target_len,
        config.train_max_records,
        config.train_seed,
    )?;
    let valid = build_dataset(
        config,
        MovingMnistSplit::Val,
        config.context_len,
        config.target_len,
        config.val_max_records,
        config.val_seed,
    )?;
    let selection_context_len = config.selection_context_len.max(config.context_len).max(1);
    let selection_target_len = config.selection_future_steps.max(config.target_len).max(1);
    let selection = build_dataset(
        config,
        MovingMnistSplit::Val,
        selection_context_len,
        selection_target_len,
        config.val_max_records,
        config.val_seed,
    )?;
    let artifact_context_len = config.artifact_context_len.max(config.context_len).max(1);
    let artifact_target_len = config.artifact_future_steps.max(config.target_len);
    let artifact = build_dataset(
        config,
        MovingMnistSplit::Val,
        artifact_context_len,
        artifact_target_len,
        config.artifact_max_records.or(config.val_max_records),
        config.val_seed,
    )?;
    Ok(MovingMnistDreamerDatasets {
        train,
        valid,
        selection,
        selection_context_len,
        selection_target_len,
        artifact,
        artifact_context_len,
        artifact_target_len,
    })
}

pub(crate) fn build_moving_mnist_tokenizer_datasets(
    config: &MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistTokenizerDatasets> {
    let sequence_len = (config.context_len + config.target_len).max(2);
    let train = build_dataset(
        config,
        MovingMnistSplit::Train,
        sequence_len.saturating_sub(1),
        1,
        config.train_max_records,
        config.train_seed,
    )?;
    let valid = build_dataset(
        config,
        MovingMnistSplit::Val,
        sequence_len.saturating_sub(1),
        1,
        config.val_max_records,
        config.val_seed,
    )?;
    Ok(MovingMnistTokenizerDatasets { train, valid })
}

pub(crate) fn build_cached_moving_mnist_split(
    dataset: &MovingMnistVideoDataset,
    config: &MovingMnistDreamerTrainConfig,
    teacher: &AutoGazeSource,
    global_teacher: Option<&ClipTeacherSource>,
    crop_teacher: Option<&ClipTeacherSource>,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> CachedSequenceSplit<TrainBackend, FrameFixationTrace> {
    build_cached_sequence_split(
        dataset,
        config.teacher_cache_batch_size,
        device,
        |indices, batch| teacher.traces_for_batch(indices, batch, config.model.k_fovea),
        |indices, clip_frames| {
            if let Some(global_teacher) = global_teacher {
                global_teacher.encode_clip_from_batch(indices, clip_frames)
            } else {
                zero_teacher_features(
                    indices.len(),
                    clip_frames.shape().dims::<5>()[1],
                    config.model.teacher_dim,
                    device,
                )
            }
        },
        |indices, clip_frames, traces| {
            if let Some(crop_teacher) = crop_teacher {
                encode_crop_teacher(crop_teacher, indices, clip_frames, traces, config)
            } else {
                zero_teacher_features(
                    indices.len(),
                    clip_frames.shape().dims::<5>()[1],
                    config.model.crop_teacher_dim,
                    device,
                )
            }
        },
    )
}

fn build_dataset(
    config: &MovingMnistDreamerTrainConfig,
    split: MovingMnistSplit,
    context_len: usize,
    target_len: usize,
    max_records: Option<usize>,
    seed: u64,
) -> Result<MovingMnistVideoDataset> {
    MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split,
        frame_size: config.model.frame_size,
        digit_size: config.digit_size.max(1),
        in_channels: config.model.channels,
        context_len,
        target_len,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records,
        normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
        min_velocity: config.min_velocity,
        max_velocity: config.max_velocity,
        seed,
    })
}
