#![cfg(feature = "train")]

use std::collections::VecDeque;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_language::api::inference::CharVocab;
use burn_dragon_stream::StreamDataset;

use crate::runtime_config::{
    MultimodalImageTextSource, MultimodalTrainingConfig, MultimodalVideoTextSource,
    MultimodalVideoTrainingConfig,
};
use crate::train::{
    ImagenetteVisionLanguageDataset, JsonlVideoLanguageDataset, JsonlVisionLanguageDataset,
    MnistVideoLanguageDataset, MnistVisionLanguageDataset, MovingMnistVideoLanguageDataset,
    MovingMnistVideoLanguageDatasetConfig, TargetTextBankBatch, VideoLanguageJsonlRecord,
    VisionLanguageJsonlRecord,
};

pub(crate) struct ImageTextDatasetBundle {
    pub(crate) train: Vec<crate::train::VisionLanguageCpuSegment>,
    pub(crate) validation: Option<Vec<crate::train::VisionLanguageCpuSegment>>,
    pub(crate) vocab: CharVocab,
    pub(crate) supports_batched_contrastive: bool,
    pub(crate) target_bank_texts: Option<Vec<String>>,
}

pub(crate) struct VideoTextDatasetBundle {
    pub(crate) train: Vec<crate::train::VideoLanguageCpuSegment>,
    pub(crate) validation: Option<Vec<crate::train::VideoLanguageCpuSegment>>,
    pub(crate) vocab: CharVocab,
    pub(crate) supports_batched_contrastive: bool,
    pub(crate) target_bank_texts: Option<Vec<String>>,
}

pub(crate) struct PreparedTargetBank<B: BackendTrait> {
    pub(crate) tokens: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    pub(crate) mask: Option<burn::tensor::Tensor<B, 2, burn::tensor::Bool>>,
    pub(crate) token_sequences: Vec<Vec<i64>>,
}

pub(crate) fn load_image_text_dataset_bundle(
    config: &MultimodalTrainingConfig,
    pretrained_vocab: Option<&CharVocab>,
) -> Result<ImageTextDatasetBundle> {
    match config.data.source {
        MultimodalImageTextSource::Jsonl => {
            let vocab = pretrained_vocab
                .cloned()
                .unwrap_or(build_vocab_from_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                    config.data.include_unknown_char,
                )?);
            let train = dataset_to_vec(JsonlVisionLanguageDataset::from_jsonl(
                &config.data.manifest,
                config.data.image_size,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = config
                .data
                .validation_manifest
                .as_ref()
                .map(|manifest| {
                    JsonlVisionLanguageDataset::from_jsonl(
                        manifest,
                        config.data.image_size,
                        &vocab,
                        config.data.normalize_mean,
                        config.data.normalize_std,
                    )
                    .map(dataset_to_vec)
                })
                .transpose()?;
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: false,
                target_bank_texts: Some(unique_target_texts_from_vision_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                )?),
            })
        }
        MultimodalImageTextSource::MnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char)
                    .context("failed to build character vocabulary from MNIST label-text dataset")?
            });
            let train = dataset_to_vec(MnistVisionLanguageDataset::from_mnist(
                true,
                config.data.image_size,
                config.data.mnist.max_train_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(MnistVisionLanguageDataset::from_mnist(
                false,
                config.data.image_size,
                config.data.mnist.max_validation_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?));
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
        MultimodalImageTextSource::ImagenetteLabelText => {
            let label_texts = imagenette_target_texts();
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.imagenette.query_q_text.as_str()]
                    .into_iter()
                    .chain(label_texts.iter().map(String::as_str))
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from Imagenette label-text dataset",
                )?
            });
            let train = dataset_to_vec(ImagenetteVisionLanguageDataset::from_imagenette(
                &config.data.imagenette.root,
                &config.data.imagenette.train_dir,
                config.data.image_size,
                config.data.imagenette.max_train_records,
                &config.data.imagenette.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(
                ImagenetteVisionLanguageDataset::from_imagenette(
                    &config.data.imagenette.root,
                    &config.data.imagenette.validation_dir,
                    config.data.image_size,
                    config.data.imagenette.max_validation_records,
                    &config.data.imagenette.query_q_text,
                    &vocab,
                    config.data.normalize_mean,
                    config.data.normalize_std,
                )?,
            ));
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(label_texts),
            })
        }
    }
}

pub(crate) fn load_video_text_dataset_bundle(
    config: &MultimodalVideoTrainingConfig,
    pretrained_vocab: Option<&CharVocab>,
) -> Result<VideoTextDatasetBundle> {
    match config.data.source {
        MultimodalVideoTextSource::Jsonl => {
            let vocab = pretrained_vocab
                .cloned()
                .unwrap_or(build_video_vocab_from_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                    config.data.include_unknown_char,
                )?);
            let train = dataset_to_vec(JsonlVideoLanguageDataset::from_jsonl(
                &config.data.manifest,
                config.data.image_size,
                config.data.clip_frames,
                config.model.tbptt.target_alignment_policy,
                &config.data.requested_horizons,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = config
                .data
                .validation_manifest
                .as_ref()
                .map(|manifest| {
                    JsonlVideoLanguageDataset::from_jsonl(
                        manifest,
                        config.data.image_size,
                        config.data.clip_frames,
                        config.model.tbptt.target_alignment_policy,
                        &config.data.requested_horizons,
                        &vocab,
                        config.data.normalize_mean,
                        config.data.normalize_std,
                    )
                    .map(dataset_to_vec)
                })
                .transpose()?;
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: false,
                target_bank_texts: Some(unique_target_texts_from_video_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                )?),
            })
        }
        MultimodalVideoTextSource::MnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from MNIST video label-text dataset",
                )?
            });
            let train = dataset_to_vec(MnistVideoLanguageDataset::from_mnist(
                true,
                config.data.image_size,
                config.data.clip_frames,
                config.data.mnist.max_train_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(MnistVideoLanguageDataset::from_mnist(
                false,
                config.data.image_size,
                config.data.clip_frames,
                config.data.mnist.max_validation_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?));
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
        MultimodalVideoTextSource::MovingMnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.moving_mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from moving-MNIST label-text dataset",
                )?
            });
            let train = dataset_to_vec(MovingMnistVideoLanguageDataset::from_moving_mnist(
                MovingMnistVideoLanguageDatasetConfig {
                    train_split: true,
                    frame_size: config.data.image_size,
                    clip_frames: config.data.clip_frames,
                    requested_horizons: &config.data.requested_horizons,
                    max_records: config.data.moving_mnist.max_train_records,
                    digit_size: config.data.moving_mnist.digit_size,
                    in_channels: config.data.moving_mnist.in_channels,
                    frame_stride: config.data.moving_mnist.frame_stride,
                    min_velocity: config.data.moving_mnist.min_velocity,
                    max_velocity: config.data.moving_mnist.max_velocity,
                    seed: config.data.moving_mnist.seed,
                    query_q_text: &config.data.moving_mnist.query_q_text,
                    vocab: &vocab,
                    normalize_mean: config.data.normalize_mean,
                    normalize_std: config.data.normalize_std,
                },
            )?);
            let validation = Some(dataset_to_vec(
                MovingMnistVideoLanguageDataset::from_moving_mnist(
                    MovingMnistVideoLanguageDatasetConfig {
                        train_split: false,
                        frame_size: config.data.image_size,
                        clip_frames: config.data.clip_frames,
                        requested_horizons: &config.data.requested_horizons,
                        max_records: config.data.moving_mnist.max_validation_records,
                        digit_size: config.data.moving_mnist.digit_size,
                        in_channels: config.data.moving_mnist.in_channels,
                        frame_stride: config.data.moving_mnist.frame_stride,
                        min_velocity: config.data.moving_mnist.min_velocity,
                        max_velocity: config.data.moving_mnist.max_velocity,
                        seed: config.data.moving_mnist.seed ^ 0xA5A5_A5A5_A5A5_A5A5,
                        query_q_text: &config.data.moving_mnist.query_q_text,
                        vocab: &vocab,
                        normalize_mean: config.data.normalize_mean,
                        normalize_std: config.data.normalize_std,
                    },
                )?,
            ));
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
    }
}

fn dataset_to_vec<D>(dataset: D) -> Vec<D::Item>
where
    D: StreamDataset,
{
    (0..dataset.len())
        .filter_map(|index| dataset.get(index))
        .collect()
}

pub(crate) fn batch_ranges(
    total: usize,
    batch_size: usize,
    max_steps: Option<usize>,
) -> Vec<std::ops::Range<usize>> {
    let batch_size = batch_size.max(1);
    let step_limit = max_steps.unwrap_or(usize::MAX);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < total && ranges.len() < step_limit {
        let end = (start + batch_size).min(total);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

pub(crate) fn unique_target_batches<T, F>(
    items: &[T],
    batch_size: usize,
    max_steps: Option<usize>,
    key_fn: F,
) -> Vec<Vec<usize>>
where
    F: Fn(&T) -> &[i64],
{
    let batch_size = batch_size.max(1);
    let step_limit = max_steps.unwrap_or(usize::MAX);
    let mut groups: Vec<(Vec<i64>, VecDeque<usize>)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let key = key_fn(item).to_vec();
        if let Some((_, indices)) = groups.iter_mut().find(|(existing, _)| *existing == key) {
            indices.push_back(index);
        } else {
            let mut indices = VecDeque::new();
            indices.push_back(index);
            groups.push((key, indices));
        }
    }

    let mut batches = Vec::new();
    while groups.iter().any(|(_, indices)| !indices.is_empty()) && batches.len() < step_limit {
        let mut batch = Vec::with_capacity(batch_size);
        for (_, indices) in groups.iter_mut() {
            if batch.len() >= batch_size {
                break;
            }
            if let Some(index) = indices.pop_front() {
                batch.push(index);
            }
        }
        if batch.is_empty() {
            break;
        }
        batches.push(batch);
    }
    batches
}

pub(crate) fn prepare_target_bank<B: BackendTrait>(
    target_bank_texts: Option<&[String]>,
    vocab: &CharVocab,
    device: &B::Device,
) -> Option<PreparedTargetBank<B>> {
    let texts = target_bank_texts?;
    if texts.is_empty() {
        return None;
    }
    let token_sequences = texts
        .iter()
        .map(|text| {
            vocab
                .encode(text, true, true)
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let max_len = token_sequences
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut token_data = vec![0_i64; token_sequences.len() * max_len];
    let mut mask_data = vec![0_i64; token_sequences.len() * max_len];
    for (row, tokens) in token_sequences.iter().enumerate() {
        for (col, token) in tokens.iter().copied().enumerate() {
            token_data[row * max_len + col] = token;
            mask_data[row * max_len + col] = 1;
        }
    }
    Some(PreparedTargetBank {
        tokens: burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(token_data, [token_sequences.len(), max_len]),
            device,
        ),
        mask: Some(
            burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(mask_data, [token_sequences.len(), max_len]),
                device,
            )
            .greater_elem(0),
        ),
        token_sequences,
    })
}

pub(crate) fn target_bank_batch_for_targets<B: BackendTrait, I>(
    bank: Option<&PreparedTargetBank<B>>,
    targets: I,
    device: &B::Device,
) -> Option<TargetTextBankBatch<B>>
where
    I: IntoIterator,
    I::Item: AsRef<[i64]>,
{
    let bank = bank?;
    let mut indices = Vec::new();
    for target in targets {
        let target = target.as_ref();
        let index = bank
            .token_sequences
            .iter()
            .position(|candidate| candidate.as_slice() == target)?;
        indices.push(index as i64);
    }
    Some(TargetTextBankBatch {
        tokens: bank.tokens.clone(),
        mask: bank.mask.clone(),
        target_indices: burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(indices.clone(), [indices.len()]),
            device,
        ),
    })
}

pub(crate) fn contiguous_batch_indices(
    total: usize,
    batch_size: usize,
    max_steps: Option<usize>,
) -> Vec<Vec<usize>> {
    batch_ranges(total, batch_size, max_steps)
        .into_iter()
        .map(|range| range.collect())
        .collect()
}

fn build_vocab_from_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
    include_unknown_char: bool,
) -> Result<CharVocab> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VisionLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        texts.push(record.query_q_text);
        texts.push(record.target_y_text);
    }
    CharVocab::fit(texts.iter().map(|text| text.as_str()), include_unknown_char)
        .context("failed to build character vocabulary from multimodal manifest")
}

fn build_video_vocab_from_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
    include_unknown_char: bool,
) -> Result<CharVocab> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VideoLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        texts.push(record.query_q_text);
        texts.push(record.target_y_text);
    }
    CharVocab::fit(texts.iter().map(|text| text.as_str()), include_unknown_char)
        .context("failed to build character vocabulary from multimodal video manifest")
}

fn unique_target_texts_from_vision_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VisionLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        if !texts.iter().any(|text| text == &record.target_y_text) {
            texts.push(record.target_y_text);
        }
    }
    Ok(texts)
}

fn unique_target_texts_from_video_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VideoLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        if !texts.iter().any(|text| text == &record.target_y_text) {
            texts.push(record.target_y_text);
        }
    }
    Ok(texts)
}

fn digit_target_texts() -> Vec<String> {
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn imagenette_target_texts() -> Vec<String> {
    [
        "tench",
        "english springer",
        "cassette player",
        "chainsaw",
        "church",
        "french horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}
