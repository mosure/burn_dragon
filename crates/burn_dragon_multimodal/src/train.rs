use anyhow::Context;
use crate::config::VlJepaDragonConfig;
use crate::data::{MultimodalStepMode, VideoLanguageTripletBatch, VisionLanguageTripletBatch};
use crate::loss::{
    VlJepaLossBreakdown, vl_jepa_bidirectional_info_nce_loss, vl_jepa_target_bank_loss,
    vl_jepa_teacher_student_info_nce_loss,
};
use crate::model::{FrozenMultimodalCoreSet, VlJepaDragon, VlJepaForwardOutput};
use crate::adapters::{TargetTextDragonEncoderAdapter, TargetTextEncoderAdapter};
use crate::state::MultimodalDragonState;
use burn_dataset::Dataset;
use burn_dataset::vision::{MnistDataset, MnistItem};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Bool, Int, Tensor};
use burn_dragon_language::api::inference::CharVocab;
use burn_dragon_stream::{
    CollatedStreamBatch, InMemoryStreamDataset, StreamBoundary, StreamDataset, StreamSampleId,
    StreamSegment, StreamStepMetadata, StreamWindowSelection, TargetAlignmentPolicy,
    resolve_stream_window_alignment,
};
use burn_dragon_vision::api::train::{
    MovingMnistRenderedClip, MovingMnistSplit, MovingMnistVideoDataset,
    MovingMnistVideoDatasetConfig, VisionNormalize,
};
use image::imageops::FilterType;
use image::{GrayImage, Luma};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLanguageJsonlRecord {
    pub image_path: PathBuf,
    pub query_q_text: String,
    pub target_y_text: String,
    pub source_id: u64,
    pub episode_id: u64,
    pub segment_id: u64,
    pub step_index: usize,
    pub absolute_time: usize,
    pub boundary: StreamBoundary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VideoLanguageJsonlRecord {
    pub frame_path: PathBuf,
    pub query_q_text: String,
    pub target_y_text: String,
    pub source_id: u64,
    pub episode_id: u64,
    pub segment_id: u64,
    pub step_index: usize,
    pub absolute_time: usize,
    pub boundary: StreamBoundary,
}

#[derive(Clone)]
pub struct VisionLanguageCpuSample {
    pub image_chw: Vec<f32>,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub query_q_tokens: Vec<i64>,
    pub target_y_tokens: Vec<i64>,
}

pub type VisionLanguageCpuSegment = StreamSegment<VisionLanguageCpuSample>;

#[derive(Clone)]
struct VideoLanguageFrameCpuSample {
    image_chw: Vec<f32>,
    channels: usize,
    height: usize,
    width: usize,
    query_q_tokens: Vec<i64>,
    target_y_tokens: Vec<i64>,
}

#[derive(Clone)]
pub struct VideoLanguageCpuSample {
    pub video_tchw: Vec<f32>,
    pub frames: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
    pub query_q_tokens: Vec<i64>,
    pub target_y_tokens: Vec<i64>,
    pub target_horizon: usize,
}

pub type VideoLanguageCpuSegment = StreamSegment<VideoLanguageCpuSample>;

pub struct JsonlVisionLanguageDataset {
    samples: InMemoryStreamDataset<VisionLanguageCpuSegment>,
}

pub struct JsonlVideoLanguageDataset {
    samples: InMemoryStreamDataset<VideoLanguageCpuSegment>,
}

pub struct MnistVisionLanguageDataset {
    samples: InMemoryStreamDataset<VisionLanguageCpuSegment>,
}

pub struct ImagenetteVisionLanguageDataset {
    samples: InMemoryStreamDataset<VisionLanguageCpuSegment>,
}

pub struct MnistVideoLanguageDataset {
    samples: InMemoryStreamDataset<VideoLanguageCpuSegment>,
}

pub struct MovingMnistVideoLanguageDataset {
    samples: InMemoryStreamDataset<VideoLanguageCpuSegment>,
}

pub struct MovingMnistVideoLanguageDatasetConfig<'a> {
    pub train_split: bool,
    pub frame_size: usize,
    pub clip_frames: usize,
    pub requested_horizons: &'a [usize],
    pub max_records: Option<usize>,
    pub digit_size: usize,
    pub in_channels: usize,
    pub frame_stride: usize,
    pub min_velocity: f32,
    pub max_velocity: f32,
    pub seed: u64,
    pub query_q_text: &'a str,
    pub vocab: &'a CharVocab,
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
}

const DIGIT_WORDS: [&str; 10] = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];

const IMAGENETTE_CLASS_LABELS: [(&str, &str); 10] = [
    ("n01440764", "tench"),
    ("n02102040", "english springer"),
    ("n02979186", "cassette player"),
    ("n03000684", "chainsaw"),
    ("n03028079", "church"),
    ("n03394916", "french horn"),
    ("n03417042", "garbage truck"),
    ("n03425413", "gas pump"),
    ("n03445777", "golf ball"),
    ("n03888257", "parachute"),
];

impl JsonlVisionLanguageDataset {
    pub fn from_jsonl(
        manifest: impl AsRef<Path>,
        image_size: usize,
        vocab: &CharVocab,
        normalize_mean: [f32; 3],
        normalize_std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let manifest = manifest.as_ref();
        let root = manifest.parent().unwrap_or_else(|| Path::new("."));
        let mut samples = Vec::new();
        for line in fs::read_to_string(manifest)?.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: VisionLanguageJsonlRecord = serde_json::from_str(line)?;
            let image = image::open(root.join(&record.image_path))?
                .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
                .to_rgb8();
            let mut image_chw = rgb_image_to_chw(&image);
            normalize_chw(&mut image_chw, image_size, image_size, normalize_mean, normalize_std);
            samples.push(StreamSegment {
                payload: VisionLanguageCpuSample {
                    image_chw,
                    channels: 3,
                    height: image_size,
                    width: image_size,
                    query_q_tokens: vocab
                        .encode(&record.query_q_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    target_y_tokens: vocab
                        .encode(&record.target_y_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                },
                stream: StreamStepMetadata {
                    sample_id: StreamSampleId {
                        source_id: record.source_id,
                        episode_id: record.episode_id,
                        segment_id: record.segment_id,
                    },
                    boundary: record.boundary,
                    step_index: record.step_index,
                    absolute_time: record.absolute_time,
                },
            });
        }
        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl MnistVisionLanguageDataset {
    pub fn from_mnist(
        train_split: bool,
        image_size: usize,
        max_records: Option<usize>,
        query_q_text: &str,
        vocab: &CharVocab,
        normalize_mean: [f32; 3],
        normalize_std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let dataset = if train_split {
            MnistDataset::train()
        } else {
            MnistDataset::test()
        };
        let limit = max_records.unwrap_or(dataset.len()).min(dataset.len());
        let mut samples = Vec::with_capacity(limit);
        for index in 0..limit {
            let Some(item) = dataset.get(index) else {
                continue;
            };
            let mut image_chw = resize_mnist_rgb(&item, image_size);
            normalize_chw(&mut image_chw, image_size.max(1), image_size.max(1), normalize_mean, normalize_std);
            let label_word = digit_word(item.label);
            samples.push(StreamSegment {
                payload: VisionLanguageCpuSample {
                    image_chw,
                    channels: 3,
                    height: image_size.max(1),
                    width: image_size.max(1),
                    query_q_tokens: vocab
                        .encode(query_q_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    target_y_tokens: vocab
                        .encode(label_word, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                },
                stream: independent_stream_step(index as u64),
            });
        }
        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl ImagenetteVisionLanguageDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn from_imagenette(
        root: impl AsRef<Path>,
        split_dir: &str,
        image_size: usize,
        max_records: Option<usize>,
        query_q_text: &str,
        vocab: &CharVocab,
        normalize_mean: [f32; 3],
        normalize_std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let split_root = root.as_ref().join(split_dir);
        let mut samples = Vec::new();
        for (synset, label_text) in IMAGENETTE_CLASS_LABELS {
            let class_dir = split_root.join(synset);
            if !class_dir.is_dir() {
                continue;
            }
            let mut entries = fs::read_dir(&class_dir)
                .with_context(|| format!("failed to read {}", class_dir.display()))?
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| {
                    path.extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| matches!(ext.to_ascii_lowercase().as_str(), "jpeg" | "jpg" | "png"))
                        .unwrap_or(false)
                })
                .collect::<Vec<_>>();
            entries.sort();
            for path in entries {
                if let Some(limit) = max_records && samples.len() >= limit {
                    return Ok(Self {
                        samples: InMemoryStreamDataset::new(samples),
                    });
                }
                let image = image::open(&path)
                    .with_context(|| format!("failed to open {}", path.display()))?
                    .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
                    .to_rgb8();
                let mut image_chw = rgb_image_to_chw(&image);
                normalize_chw(&mut image_chw, image_size.max(1), image_size.max(1), normalize_mean, normalize_std);
                let sample_index = samples.len() as u64;
                samples.push(StreamSegment {
                    payload: VisionLanguageCpuSample {
                        image_chw,
                        channels: 3,
                        height: image_size.max(1),
                        width: image_size.max(1),
                        query_q_tokens: vocab
                            .encode(query_q_text, true, true)
                            .into_iter()
                            .map(i64::from)
                            .collect(),
                        target_y_tokens: vocab
                            .encode(label_text, true, true)
                            .into_iter()
                            .map(i64::from)
                            .collect(),
                    },
                    stream: independent_stream_step(sample_index),
                });
            }
        }
        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl StreamDataset for JsonlVisionLanguageDataset {
    type Item = VisionLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

impl StreamDataset for MnistVisionLanguageDataset {
    type Item = VisionLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

impl StreamDataset for ImagenetteVisionLanguageDataset {
    type Item = VisionLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

impl MnistVideoLanguageDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn from_mnist(
        train_split: bool,
        image_size: usize,
        clip_frames: usize,
        max_records: Option<usize>,
        query_q_text: &str,
        vocab: &CharVocab,
        normalize_mean: [f32; 3],
        normalize_std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let dataset = if train_split {
            MnistDataset::train()
        } else {
            MnistDataset::test()
        };
        let limit = max_records.unwrap_or(dataset.len()).min(dataset.len());
        let frame_count = clip_frames.max(1);
        let mut samples = Vec::with_capacity(limit);
        for index in 0..limit {
            let Some(item) = dataset.get(index) else {
                continue;
            };
            let mut image_chw = resize_mnist_rgb(&item, image_size);
            normalize_chw(&mut image_chw, image_size.max(1), image_size.max(1), normalize_mean, normalize_std);
            let label_word = digit_word(item.label);
            let mut video_tchw = Vec::with_capacity(frame_count * image_chw.len());
            for _ in 0..frame_count {
                video_tchw.extend_from_slice(&image_chw);
            }
            samples.push(StreamSegment {
                payload: VideoLanguageCpuSample {
                    video_tchw,
                    frames: frame_count,
                    channels: 3,
                    height: image_size.max(1),
                    width: image_size.max(1),
                    query_q_tokens: vocab
                        .encode(query_q_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    target_y_tokens: vocab
                        .encode(label_word, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    target_horizon: 1,
                },
                stream: independent_stream_step(index as u64),
            });
        }
        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl JsonlVideoLanguageDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn from_jsonl(
        manifest: impl AsRef<Path>,
        image_size: usize,
        clip_frames: usize,
        target_alignment_policy: TargetAlignmentPolicy,
        requested_horizons: &[usize],
        vocab: &CharVocab,
        normalize_mean: [f32; 3],
        normalize_std: [f32; 3],
    ) -> anyhow::Result<Self> {
        let manifest = manifest.as_ref();
        let root = manifest.parent().unwrap_or_else(|| Path::new("."));
        let mut frames = Vec::new();
        for line in fs::read_to_string(manifest)?.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let record: VideoLanguageJsonlRecord = serde_json::from_str(line)?;
            let image = image::open(root.join(&record.frame_path))?
                .resize_exact(image_size as u32, image_size as u32, FilterType::Triangle)
                .to_rgb8();
            let mut image_chw = rgb_image_to_chw(&image);
            normalize_chw(&mut image_chw, image_size, image_size, normalize_mean, normalize_std);
            frames.push(StreamSegment {
                payload: VideoLanguageFrameCpuSample {
                    image_chw,
                    channels: 3,
                    height: image_size,
                    width: image_size,
                    query_q_tokens: vocab
                        .encode(&record.query_q_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                    target_y_tokens: vocab
                        .encode(&record.target_y_text, true, true)
                        .into_iter()
                        .map(i64::from)
                        .collect(),
                },
                stream: StreamStepMetadata {
                    sample_id: StreamSampleId {
                        source_id: record.source_id,
                        episode_id: record.episode_id,
                        segment_id: record.segment_id,
                    },
                    boundary: record.boundary,
                    step_index: record.step_index,
                    absolute_time: record.absolute_time,
                },
            });
        }

        let requested_horizons = if requested_horizons.is_empty() {
            vec![1]
        } else {
            requested_horizons.iter().copied().map(|h| h.max(1)).collect()
        };
        let metadata: Vec<_> = frames.iter().map(|segment| segment.stream).collect();
        let mut samples = Vec::new();
        for observation_index in 0..frames.len() {
            let requested_horizon = requested_horizons[observation_index % requested_horizons.len()];
            let Some(selection) = resolve_stream_window_alignment(
                target_alignment_policy,
                &metadata,
                observation_index,
                clip_frames,
                Some(requested_horizon),
            ) else {
                continue;
            };
            samples.push(build_video_cpu_segment(&frames, selection));
        }

        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl MovingMnistVideoLanguageDataset {
    pub fn from_moving_mnist(
        config: MovingMnistVideoLanguageDatasetConfig<'_>,
    ) -> anyhow::Result<Self> {
        let requested_horizons = if config.requested_horizons.is_empty() {
            vec![1]
        } else {
            config
                .requested_horizons
                .iter()
                .copied()
                .map(|h| h.max(1))
                .collect()
        };
        let split = if config.train_split {
            MovingMnistSplit::Train
        } else {
            MovingMnistSplit::Val
        };
        let target_len = requested_horizons.iter().copied().max().unwrap_or(1).max(1);
        let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split,
            frame_size: config.frame_size.max(1),
            digit_size: config.digit_size.max(1),
            in_channels: config.in_channels.max(1),
            context_len: config.clip_frames.max(1),
            target_len,
            extra_future_frames: 0,
            frame_stride: config.frame_stride.max(1),
            max_records: config.max_records,
            normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
            min_velocity: config.min_velocity,
            max_velocity: config.max_velocity,
            seed: config.seed,
        })?;
        let mut samples = Vec::with_capacity(dataset.len());
        for index in 0..dataset.len() {
            let Some(clip) = dataset.rendered_clip(index) else {
                continue;
            };
            let label_word = digit_word(clip.label as u8);
            samples.push(build_moving_mnist_video_segment(
                clip,
                requested_horizons[index % requested_horizons.len()],
                config.query_q_text,
                label_word,
                config.vocab,
                config.normalize_mean,
                config.normalize_std,
                index as u64,
            ));
        }
        Ok(Self {
            samples: InMemoryStreamDataset::new(samples),
        })
    }
}

impl StreamDataset for JsonlVideoLanguageDataset {
    type Item = VideoLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

impl StreamDataset for MnistVideoLanguageDataset {
    type Item = VideoLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

impl StreamDataset for MovingMnistVideoLanguageDataset {
    type Item = VideoLanguageCpuSegment;

    fn len(&self) -> usize {
        self.samples.len()
    }

    fn get(&self, index: usize) -> Option<Self::Item> {
        self.samples.get(index)
    }
}

fn build_video_cpu_segment(
    frames: &[StreamSegment<VideoLanguageFrameCpuSample>],
    selection: StreamWindowSelection,
) -> VideoLanguageCpuSegment {
    let observation = &frames[selection.observation_index];
    let target = &frames[selection.target_index];
    let channels = observation.payload.channels;
    let height = observation.payload.height;
    let width = observation.payload.width;
    let frame_count = selection.observation_index - selection.start_index + 1;
    let mut video_tchw = Vec::with_capacity(frame_count * channels * height * width);
    for frame in &frames[selection.start_index..=selection.observation_index] {
        video_tchw.extend_from_slice(&frame.payload.image_chw);
    }
    StreamSegment {
        payload: VideoLanguageCpuSample {
            video_tchw,
            frames: frame_count,
            channels,
            height,
            width,
            query_q_tokens: observation.payload.query_q_tokens.clone(),
            target_y_tokens: target.payload.target_y_tokens.clone(),
            target_horizon: selection.horizon,
        },
        stream: observation.stream,
    }
}

#[allow(clippy::too_many_arguments)]
fn build_moving_mnist_video_segment(
    clip: MovingMnistRenderedClip,
    target_horizon: usize,
    query_q_text: &str,
    target_y_text: &str,
    vocab: &CharVocab,
    normalize_mean: [f32; 3],
    normalize_std: [f32; 3],
    sample_index: u64,
) -> VideoLanguageCpuSegment {
    let mut video_tchw = clip.frames;
    let plane = clip.frame_size * clip.frame_size;
    let frame_stride = clip.channels * plane;
    for frame in 0..clip.clip_len {
        let frame_offset = frame * frame_stride;
        normalize_chw(
            &mut video_tchw[frame_offset..frame_offset + frame_stride],
            clip.frame_size,
            clip.frame_size,
            normalize_mean,
            normalize_std,
        );
    }
    StreamSegment {
        payload: VideoLanguageCpuSample {
            video_tchw,
            frames: clip.clip_len,
            channels: clip.channels,
            height: clip.frame_size,
            width: clip.frame_size,
            query_q_tokens: vocab
                .encode(query_q_text, true, true)
                .into_iter()
                .map(i64::from)
                .collect(),
            target_y_tokens: vocab
                .encode(target_y_text, true, true)
                .into_iter()
                .map(i64::from)
                .collect(),
            target_horizon: target_horizon.max(1),
        },
        stream: independent_stream_step(sample_index),
    }
}

fn independent_stream_step(sample_index: u64) -> StreamStepMetadata {
    StreamStepMetadata {
        sample_id: StreamSampleId {
            source_id: sample_index,
            episode_id: sample_index,
            segment_id: 0,
        },
        boundary: StreamBoundary::ResetEpisode,
        step_index: 0,
        absolute_time: sample_index as usize,
    }
}

fn digit_word(label: u8) -> &'static str {
    DIGIT_WORDS[label as usize % DIGIT_WORDS.len()]
}

fn rgb_image_to_chw(image: &image::RgbImage) -> Vec<f32> {
    let width = image.width() as usize;
    let height = image.height() as usize;
    let mut image_chw = Vec::with_capacity(3 * width * height);
    for channel in 0..3 {
        for y in 0..height {
            for x in 0..width {
                let pixel = image.get_pixel(x as u32, y as u32);
                image_chw.push(pixel[channel] as f32 / 255.0);
            }
        }
    }
    image_chw
}

fn normalize_chw(
    image_chw: &mut [f32],
    height: usize,
    width: usize,
    mean: [f32; 3],
    std: [f32; 3],
) {
    let plane = height.max(1) * width.max(1);
    for channel in 0..3 {
        let start = channel * plane;
        let end = start + plane;
        let denom = std[channel].max(1e-6);
        for value in &mut image_chw[start..end] {
            *value = (*value - mean[channel]) / denom;
        }
    }
}

fn resize_mnist_rgb(item: &MnistItem, image_size: usize) -> Vec<f32> {
    let image_size = image_size.max(1);
    let mut image = GrayImage::new(28, 28);
    for (y, row) in item.image.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            image.put_pixel(x as u32, y as u32, Luma([value.clamp(0.0, 255.0) as u8]));
        }
    }
    let resized = if image_size == 28 {
        image
    } else {
        image::imageops::resize(
            &image,
            image_size as u32,
            image_size as u32,
            FilterType::Triangle,
        )
    };
    let grayscale: Vec<f32> = resized
        .into_raw()
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect();
    let mut image_chw = Vec::with_capacity(3 * image_size * image_size);
    for _ in 0..3 {
        image_chw.extend_from_slice(&grayscale);
    }
    image_chw
}

pub fn collate_vision_language_segments<B: Backend>(
    segments: &[VisionLanguageCpuSegment],
    device: &B::Device,
) -> CollatedStreamBatch<VisionLanguageTripletBatch<B>> {
    let batch = segments.len();
    let max_query = segments
        .iter()
        .map(|segment| segment.payload.query_q_tokens.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let max_target = segments
        .iter()
        .map(|segment| segment.payload.target_y_tokens.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let channels = segments[0].payload.channels;
    let height = segments[0].payload.height;
    let width = segments[0].payload.width;

    let mut image = Vec::with_capacity(batch * channels * height * width);
    let mut query_tokens = vec![0_i64; batch * max_query];
    let mut target_tokens = vec![0_i64; batch * max_target];
    let mut query_mask = vec![0_i64; batch * max_query];
    let mut target_mask = vec![0_i64; batch * max_target];

    for (batch_index, segment) in segments.iter().enumerate() {
        image.extend_from_slice(&segment.payload.image_chw);
        for (token_index, token) in segment.payload.query_q_tokens.iter().enumerate() {
            query_tokens[batch_index * max_query + token_index] = *token;
            query_mask[batch_index * max_query + token_index] = 1;
        }
        for (token_index, token) in segment.payload.target_y_tokens.iter().enumerate() {
            target_tokens[batch_index * max_target + token_index] = *token;
            target_mask[batch_index * max_target + token_index] = 1;
        }
    }

    CollatedStreamBatch {
        payload: VisionLanguageTripletBatch {
            vision_x: Tensor::<B, 4>::from_data(
                burn::tensor::TensorData::new(image, [batch, channels, height, width]),
                device,
            ),
            query_q_tokens: Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(query_tokens, [batch, max_query]),
                device,
            ),
            query_q_mask: Some(
                Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(query_mask, [batch, max_query]),
                    device,
                )
                .greater_elem(0),
            ),
            target_y_tokens: Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(target_tokens, [batch, max_target]),
                device,
            ),
            target_y_mask: Some(
                Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(target_mask, [batch, max_target]),
                    device,
                )
                .greater_elem(0),
            ),
        },
        stream: segments.iter().map(|segment| segment.stream).collect(),
    }
}

pub fn collate_video_language_segments<B: Backend>(
    segments: &[VideoLanguageCpuSegment],
    device: &B::Device,
) -> CollatedStreamBatch<VideoLanguageTripletBatch<B>> {
    let batch = segments.len();
    let frames = segments[0].payload.frames;
    let channels = segments[0].payload.channels;
    let height = segments[0].payload.height;
    let width = segments[0].payload.width;
    let max_query = segments
        .iter()
        .map(|segment| segment.payload.query_q_tokens.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let max_target = segments
        .iter()
        .map(|segment| segment.payload.target_y_tokens.len())
        .max()
        .unwrap_or(1)
        .max(1);
    let mut video = Vec::with_capacity(batch * frames * channels * height * width);
    let mut query_tokens = vec![0_i64; batch * max_query];
    let mut target_tokens = vec![0_i64; batch * max_target];
    let mut query_mask = vec![0_i64; batch * max_query];
    let mut target_mask = vec![0_i64; batch * max_target];

    for (batch_index, segment) in segments.iter().enumerate() {
        video.extend_from_slice(&segment.payload.video_tchw);
        for (token_index, token) in segment.payload.query_q_tokens.iter().enumerate() {
            query_tokens[batch_index * max_query + token_index] = *token;
            query_mask[batch_index * max_query + token_index] = 1;
        }
        for (token_index, token) in segment.payload.target_y_tokens.iter().enumerate() {
            target_tokens[batch_index * max_target + token_index] = *token;
            target_mask[batch_index * max_target + token_index] = 1;
        }
    }

    CollatedStreamBatch {
        payload: VideoLanguageTripletBatch {
            video_x: Tensor::<B, 5>::from_data(
                burn::tensor::TensorData::new(video, [batch, frames, channels, height, width]),
                device,
            ),
            query_q_tokens: Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(query_tokens, [batch, max_query]),
                device,
            ),
            query_q_mask: Some(
                Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(query_mask, [batch, max_query]),
                    device,
                )
                .greater_elem(0),
            ),
            target_y_tokens: Tensor::<B, 2, Int>::from_data(
                burn::tensor::TensorData::new(target_tokens, [batch, max_target]),
                device,
            ),
            target_y_mask: Some(
                Tensor::<B, 2, Int>::from_data(
                    burn::tensor::TensorData::new(target_mask, [batch, max_target]),
                    device,
                )
                .greater_elem(0),
            ),
        },
        stream: segments.iter().map(|segment| segment.stream).collect(),
    }
}

#[derive(Clone)]
pub struct MultimodalTrainStepOutput<B: Backend> {
    pub loss: VlJepaLossBreakdown<B>,
    pub forward: VlJepaForwardOutput<B>,
}

#[derive(Clone)]
pub struct TargetTextBankBatch<B: Backend> {
    pub tokens: Tensor<B, 2, Int>,
    pub mask: Option<Tensor<B, 2, Bool>>,
    pub target_indices: Tensor<B, 1, Int>,
}

fn refine_step_weight(step_index: usize, power: f32) -> f32 {
    ((step_index + 1) as f32).powf(power.max(0.0))
}

fn loss_for_prediction<B: Backend>(
    predicted_target_embedding: Tensor<B, 2>,
    student_target_embedding_y: Tensor<B, 2>,
    teacher_target_embedding_y: Option<Tensor<B, 2>>,
    temperature: f32,
) -> VlJepaLossBreakdown<B> {
    match teacher_target_embedding_y {
        Some(teacher_target_embedding_y) => vl_jepa_teacher_student_info_nce_loss(
            predicted_target_embedding,
            student_target_embedding_y,
            teacher_target_embedding_y,
            temperature,
        ),
        None => vl_jepa_bidirectional_info_nce_loss(
            predicted_target_embedding,
            student_target_embedding_y,
            temperature,
        ),
    }
}

fn combine_with_target_bank_loss<B: Backend>(
    model: &VlJepaDragon<B>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    mut base_loss: VlJepaLossBreakdown<B>,
    predicted_target_embedding: Tensor<B, 2>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    config: &VlJepaDragonConfig,
) -> VlJepaLossBreakdown<B> {
    let pairwise_weight = config.pairwise_loss_weight.max(0.0);
    let bank_weight = config.target_bank_loss_weight.max(0.0);
    match (target_bank, bank_weight > 0.0) {
        (Some(target_bank), true) => {
            let target_bank_embeddings = target_teacher
                .map(|teacher| {
                    teacher
                        .encode_y((target_bank.tokens.clone(), target_bank.mask.clone()))
                        .target_embedding
                        .detach()
                })
                .unwrap_or_else(|| {
                    model
                        .encode_target_bank(target_bank.tokens.clone(), target_bank.mask.clone())
                });
            let bank_loss = vl_jepa_target_bank_loss(
                predicted_target_embedding,
                target_bank_embeddings,
                target_bank.target_indices.clone(),
                config.temperature,
            );
            let total_weight = (pairwise_weight + bank_weight).max(1e-6);
            base_loss.total =
                (base_loss.total * pairwise_weight + bank_loss.total.clone() * bank_weight)
                    / total_weight;
            base_loss.predictor_to_target = (base_loss.predictor_to_target * pairwise_weight
                + bank_loss.predictor_to_target.clone() * bank_weight)
                / total_weight;
            base_loss.target_to_predictor = (base_loss.target_to_predictor * pairwise_weight
                + bank_loss.target_to_predictor.clone() * bank_weight)
                / total_weight;
            base_loss.similarities = bank_loss.similarities;
            base_loss
        }
        _ if pairwise_weight <= 0.0 => {
            let zero = base_loss.total.clone() * 0.0;
            VlJepaLossBreakdown {
                total: zero.clone(),
                predictor_to_target: zero.clone(),
                target_to_predictor: zero,
                similarities: base_loss.similarities,
            }
        }
        _ => base_loss,
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn multimodal_train_step_with_frozen_cores<B: AutodiffBackend>(
    model: &VlJepaDragon<B>,
    frozen_cores: Option<&FrozenMultimodalCoreSet<B>>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VisionLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    mut state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    state.apply_stream_controls(
        stream,
        config.tbptt.window(),
        config.tbptt.state_carry_policy,
        config.tbptt.fusion_carry_policy,
    );
    let target_tokens = batch.target_y_tokens.clone();
    let target_mask = batch.target_y_mask.clone();
    let mut forward = model.forward_x_q_y_with_frozen_cores(
        frozen_cores,
        batch,
        state,
        MultimodalStepMode::Observe,
    );
    let student_target_embedding_y = forward.targets.target_embedding_y.clone();
    let teacher_target_embedding_y = target_teacher.map(|teacher| {
        teacher
            .encode_y((target_tokens, target_mask))
            .target_embedding
            .detach()
    });
    let mut loss = combine_with_target_bank_loss(
        model,
        target_teacher,
        loss_for_prediction(
        forward.fusion.predicted_target_embedding.clone(),
        student_target_embedding_y.clone(),
        teacher_target_embedding_y.clone(),
        config.temperature,
        ),
        forward.fusion.predicted_target_embedding.clone(),
        target_bank,
        config,
    );
    let mut weight_sum = refine_step_weight(0, config.refine_loss_power);
    let mut loss_total = loss.total.clone() * weight_sum;
    let mut predictor_to_target = loss.predictor_to_target.clone() * weight_sum;
    let mut target_to_predictor = loss.target_to_predictor.clone() * weight_sum;
    for refine_index in 0..config.fusion_refine_steps {
        let Some((fusion, next_state)) =
            model.refine_with_frozen_cores(frozen_cores, forward.state.clone())
        else {
            break;
        };
        let refined_loss = combine_with_target_bank_loss(
            model,
            target_teacher,
            loss_for_prediction(
                fusion.predicted_target_embedding.clone(),
                student_target_embedding_y.clone(),
                teacher_target_embedding_y.clone(),
                config.temperature,
            ),
            fusion.predicted_target_embedding.clone(),
            target_bank,
            config,
        );
        let weight = refine_step_weight(refine_index + 1, config.refine_loss_power);
        loss_total = loss_total + refined_loss.total.clone() * weight;
        predictor_to_target = predictor_to_target + refined_loss.predictor_to_target.clone() * weight;
        target_to_predictor = target_to_predictor + refined_loss.target_to_predictor.clone() * weight;
        weight_sum += weight;
        forward.fusion = fusion;
        forward.state = next_state;
        loss = refined_loss;
    }
    loss.total = loss_total / weight_sum;
    loss.predictor_to_target = predictor_to_target / weight_sum;
    loss.target_to_predictor = target_to_predictor / weight_sum;
    MultimodalTrainStepOutput { loss, forward }
}

pub fn multimodal_train_step<B: AutodiffBackend>(
    model: &VlJepaDragon<B>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VisionLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    let frozen_cores = model.frozen_core_set();
    multimodal_train_step_with_frozen_cores(
        model,
        Some(&frozen_cores),
        target_teacher,
        target_bank,
        batch,
        stream,
        state,
        config,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn multimodal_video_train_step_with_frozen_cores<B: AutodiffBackend>(
    model: &VlJepaDragon<B>,
    frozen_cores: Option<&FrozenMultimodalCoreSet<B>>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VideoLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    mut state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    state.apply_stream_controls(
        stream,
        config.tbptt.window(),
        config.tbptt.state_carry_policy,
        config.tbptt.fusion_carry_policy,
    );
    let target_tokens = batch.target_y_tokens.clone();
    let target_mask = batch.target_y_mask.clone();
    let mut forward = model.forward_video_x_q_y_with_frozen_cores(frozen_cores, batch, state);
    let student_target_embedding_y = forward.targets.target_embedding_y.clone();
    let teacher_target_embedding_y = target_teacher.map(|teacher| {
        teacher
            .encode_y((target_tokens, target_mask))
            .target_embedding
            .detach()
    });
    let mut loss = combine_with_target_bank_loss(
        model,
        target_teacher,
        loss_for_prediction(
        forward.fusion.predicted_target_embedding.clone(),
        student_target_embedding_y.clone(),
        teacher_target_embedding_y.clone(),
        config.temperature,
        ),
        forward.fusion.predicted_target_embedding.clone(),
        target_bank,
        config,
    );
    let mut weight_sum = refine_step_weight(0, config.refine_loss_power);
    let mut loss_total = loss.total.clone() * weight_sum;
    let mut predictor_to_target = loss.predictor_to_target.clone() * weight_sum;
    let mut target_to_predictor = loss.target_to_predictor.clone() * weight_sum;
    for refine_index in 0..config.fusion_refine_steps {
        let Some((fusion, next_state)) =
            model.refine_with_frozen_cores(frozen_cores, forward.state.clone())
        else {
            break;
        };
        let refined_loss = combine_with_target_bank_loss(
            model,
            target_teacher,
            loss_for_prediction(
                fusion.predicted_target_embedding.clone(),
                student_target_embedding_y.clone(),
                teacher_target_embedding_y.clone(),
                config.temperature,
            ),
            fusion.predicted_target_embedding.clone(),
            target_bank,
            config,
        );
        let weight = refine_step_weight(refine_index + 1, config.refine_loss_power);
        loss_total = loss_total + refined_loss.total.clone() * weight;
        predictor_to_target = predictor_to_target + refined_loss.predictor_to_target.clone() * weight;
        target_to_predictor = target_to_predictor + refined_loss.target_to_predictor.clone() * weight;
        weight_sum += weight;
        forward.fusion = fusion;
        forward.state = next_state;
        loss = refined_loss;
    }
    loss.total = loss_total / weight_sum;
    loss.predictor_to_target = predictor_to_target / weight_sum;
    loss.target_to_predictor = target_to_predictor / weight_sum;
    MultimodalTrainStepOutput { loss, forward }
}

pub fn multimodal_video_train_step<B: AutodiffBackend>(
    model: &VlJepaDragon<B>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VideoLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    let frozen_cores = model.frozen_core_set();
    multimodal_video_train_step_with_frozen_cores(
        model,
        Some(&frozen_cores),
        target_teacher,
        target_bank,
        batch,
        stream,
        state,
        config,
    )
}

pub fn multimodal_eval_step<B: Backend>(
    model: &VlJepaDragon<B>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VisionLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    mut state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    state.apply_stream_controls(
        stream,
        config.tbptt.window(),
        config.tbptt.state_carry_policy,
        config.tbptt.fusion_carry_policy,
    );
    let target_tokens = batch.target_y_tokens.clone();
    let target_mask = batch.target_y_mask.clone();
    let mut forward = model.forward_x_q_y(batch, state, MultimodalStepMode::Observe);
    let student_target_embedding_y = forward.targets.target_embedding_y.clone();
    let teacher_target_embedding_y = target_teacher.map(|teacher| {
        teacher
            .encode_y((target_tokens, target_mask))
            .target_embedding
            .detach()
    });
    let eval_refine_steps = config
        .eval_fusion_refine_steps
        .unwrap_or(config.fusion_refine_steps);
    let mut loss = combine_with_target_bank_loss(
        model,
        target_teacher,
        loss_for_prediction(
            forward.fusion.predicted_target_embedding.clone(),
            student_target_embedding_y.clone(),
            teacher_target_embedding_y.clone(),
            config.temperature,
        ),
        forward.fusion.predicted_target_embedding.clone(),
        target_bank,
        config,
    );
    for _ in 0..eval_refine_steps {
        let Some((fusion, next_state)) = model.refine(forward.state.clone()) else {
            break;
        };
        forward.fusion = fusion;
        forward.state = next_state;
        loss = combine_with_target_bank_loss(
            model,
            target_teacher,
            loss_for_prediction(
                forward.fusion.predicted_target_embedding.clone(),
                student_target_embedding_y.clone(),
                teacher_target_embedding_y.clone(),
                config.temperature,
            ),
            forward.fusion.predicted_target_embedding.clone(),
            target_bank,
            config,
        );
    }
    MultimodalTrainStepOutput { loss, forward }
}

pub fn multimodal_video_eval_step<B: Backend>(
    model: &VlJepaDragon<B>,
    target_teacher: Option<&TargetTextDragonEncoderAdapter<B>>,
    target_bank: Option<&TargetTextBankBatch<B>>,
    batch: VideoLanguageTripletBatch<B>,
    stream: &StreamStepMetadata,
    mut state: MultimodalDragonState<B>,
    config: &VlJepaDragonConfig,
) -> MultimodalTrainStepOutput<B> {
    state.apply_stream_controls(
        stream,
        config.tbptt.window(),
        config.tbptt.state_carry_policy,
        config.tbptt.fusion_carry_policy,
    );
    let target_tokens = batch.target_y_tokens.clone();
    let target_mask = batch.target_y_mask.clone();
    let mut forward = model.forward_video_x_q_y(batch, state);
    let student_target_embedding_y = forward.targets.target_embedding_y.clone();
    let teacher_target_embedding_y = target_teacher.map(|teacher| {
        teacher
            .encode_y((target_tokens, target_mask))
            .target_embedding
            .detach()
    });
    let eval_refine_steps = config
        .eval_fusion_refine_steps
        .unwrap_or(config.fusion_refine_steps);
    let mut loss = combine_with_target_bank_loss(
        model,
        target_teacher,
        loss_for_prediction(
            forward.fusion.predicted_target_embedding.clone(),
            student_target_embedding_y.clone(),
            teacher_target_embedding_y.clone(),
            config.temperature,
        ),
        forward.fusion.predicted_target_embedding.clone(),
        target_bank,
        config,
    );
    for _ in 0..eval_refine_steps {
        let Some((fusion, next_state)) = model.refine(forward.state.clone()) else {
            break;
        };
        forward.fusion = fusion;
        forward.state = next_state;
        loss = combine_with_target_bank_loss(
            model,
            target_teacher,
            loss_for_prediction(
                forward.fusion.predicted_target_embedding.clone(),
                student_target_embedding_y.clone(),
                teacher_target_embedding_y.clone(),
                config.temperature,
            ),
            forward.fusion.predicted_target_embedding.clone(),
            target_bank,
            config,
        );
    }
    MultimodalTrainStepOutput { loss, forward }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::VlJepaDragonConfig;
    use burn_ndarray::{NdArray, NdArrayDevice};
    use burn_autodiff::Autodiff;
    use burn::tensor::Bool;
    use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
    use burn_dragon_train::api::runtime::device_memory_usage_safe;
    use image::RgbImage;
    use tempfile::tempdir;

    #[cfg(not(target_arch = "wasm32"))]
    use burn_wgpu::{Wgpu, WgpuDevice};

    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Clone, Copy, Debug)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn init_wgpu_test_runtime(device: &WgpuDevice) {
        burn_dragon_train::api::wgpu::init_runtime(
            device,
            &burn_dragon_train::api::config::WgpuRuntimeConfig::default(),
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn wgpu_memory_snapshot(device: &WgpuDevice) -> MemorySnapshot {
        let usage = device_memory_usage_safe::<Autodiff<Wgpu<f32>>>(device)
            .expect("wgpu memory usage");
        MemorySnapshot {
            reserved: usage.reserved_bytes,
            in_use: usage.in_use_bytes,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn assert_memory_growth_bounded(
        label: &str,
        snapshots: &[MemorySnapshot],
        max_reserved_growth: u64,
        max_in_use_growth: u64,
    ) {
        assert!(!snapshots.is_empty(), "{label}: no memory snapshots collected");
        let min_reserved = snapshots.iter().map(|snapshot| snapshot.reserved).min().unwrap_or(0);
        let max_reserved = snapshots.iter().map(|snapshot| snapshot.reserved).max().unwrap_or(0);
        let min_in_use = snapshots.iter().map(|snapshot| snapshot.in_use).min().unwrap_or(0);
        let max_in_use = snapshots.iter().map(|snapshot| snapshot.in_use).max().unwrap_or(0);
        let growth_reserved = max_reserved.saturating_sub(min_reserved);
        let growth_in_use = max_in_use.saturating_sub(min_in_use);
        assert!(
            growth_reserved <= max_reserved_growth,
            "{label}: reserved bytes grew by {growth_reserved} (> {max_reserved_growth}); snapshots={snapshots:?}"
        );
        assert!(
            growth_in_use <= max_in_use_growth,
            "{label}: in-use bytes grew by {growth_in_use} (> {max_in_use_growth}); snapshots={snapshots:?}"
        );
    }

    #[test]
    fn collate_preserves_stream_metadata() {
        type Backend = NdArray<f32>;
        let segment = StreamSegment {
            payload: VisionLanguageCpuSample {
                image_chw: vec![0.0; 3 * 4 * 4],
                channels: 3,
                height: 4,
                width: 4,
                query_q_tokens: vec![1, 2],
                target_y_tokens: vec![3, 4, 5],
            },
            stream: StreamStepMetadata {
                sample_id: StreamSampleId {
                    source_id: 4,
                    episode_id: 5,
                    segment_id: 6,
                },
                boundary: StreamBoundary::Continue,
                step_index: 2,
                absolute_time: 7,
            },
        };
        let batch = collate_vision_language_segments::<Backend>(std::slice::from_ref(&segment), &Default::default());
        assert_eq!(batch.payload.vision_x.shape().dims(), [1, 3, 4, 4]);
        assert_eq!(batch.stream[0].sample_id.segment_id, 6);
    }

    #[test]
    fn jsonl_dataset_loads_real_image_text_records() {
        let dir = tempdir().expect("tempdir");
        let image_path = dir.path().join("sample.png");
        let mut image = RgbImage::new(4, 4);
        for pixel in image.pixels_mut() {
            *pixel = image::Rgb([32, 64, 128]);
        }
        image.save(&image_path).expect("save image");
        let manifest_path = dir.path().join("manifest.jsonl");
        let record = VisionLanguageJsonlRecord {
            image_path: PathBuf::from("sample.png"),
            query_q_text: "question".into(),
            target_y_text: "answer".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 0,
            step_index: 0,
            absolute_time: 0,
            boundary: StreamBoundary::ResetEpisode,
        };
        fs::write(
            &manifest_path,
            format!("{}\n", serde_json::to_string(&record).expect("record json")),
        )
        .expect("manifest write");
        let dataset =
            JsonlVisionLanguageDataset::from_jsonl(
                &manifest_path,
                4,
                &CharVocab::fit(["question", "answer"].into_iter(), true).expect("fit vocab"),
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            )
            .expect("dataset");
        assert_eq!(dataset.len(), 1);
        assert_eq!(
            dataset
                .get(0)
                .expect("item")
                .stream
                .boundary,
            StreamBoundary::ResetEpisode
        );
    }

    #[test]
    fn jsonl_video_dataset_builds_horizon_sampled_clips_without_cross_episode_leakage() {
        let dir = tempdir().expect("tempdir");
        for frame in 0..6 {
            let image_path = dir.path().join(format!("frame-{frame}.png"));
            let mut image = RgbImage::new(4, 4);
            for pixel in image.pixels_mut() {
                *pixel = image::Rgb([(frame * 32) as u8, 64, 128]);
            }
            image.save(&image_path).expect("save image");
        }

        let manifest_path = dir.path().join("video.jsonl");
        let records = [
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-0.png"),
                query_q_text: "look".into(),
                target_y_text: "frame0".into(),
                source_id: 1,
                episode_id: 1,
                segment_id: 0,
                step_index: 0,
                absolute_time: 0,
                boundary: StreamBoundary::ResetEpisode,
            },
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-1.png"),
                query_q_text: "look".into(),
                target_y_text: "frame1".into(),
                source_id: 1,
                episode_id: 1,
                segment_id: 1,
                step_index: 1,
                absolute_time: 1,
                boundary: StreamBoundary::Continue,
            },
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-2.png"),
                query_q_text: "look".into(),
                target_y_text: "frame2".into(),
                source_id: 1,
                episode_id: 1,
                segment_id: 2,
                step_index: 2,
                absolute_time: 2,
                boundary: StreamBoundary::Continue,
            },
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-3.png"),
                query_q_text: "look".into(),
                target_y_text: "frame3".into(),
                source_id: 1,
                episode_id: 1,
                segment_id: 3,
                step_index: 3,
                absolute_time: 3,
                boundary: StreamBoundary::Continue,
            },
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-4.png"),
                query_q_text: "look".into(),
                target_y_text: "frame4".into(),
                source_id: 1,
                episode_id: 2,
                segment_id: 0,
                step_index: 0,
                absolute_time: 4,
                boundary: StreamBoundary::ResetEpisode,
            },
            VideoLanguageJsonlRecord {
                frame_path: PathBuf::from("frame-5.png"),
                query_q_text: "new".into(),
                target_y_text: "frame5".into(),
                source_id: 1,
                episode_id: 2,
                segment_id: 1,
                step_index: 1,
                absolute_time: 5,
                boundary: StreamBoundary::Continue,
            },
        ];
        fs::write(
            &manifest_path,
            records
                .iter()
                .map(|record| serde_json::to_string(record).expect("record json"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("manifest write");
        let dataset = JsonlVideoLanguageDataset::from_jsonl(
            &manifest_path,
            4,
            2,
            TargetAlignmentPolicy::VariableFuture,
            &[1, 2],
            &CharVocab::fit(
                ["look", "new", "frame0", "frame1", "frame2", "frame3", "frame4", "frame5"]
                    .into_iter(),
                true,
            )
                .expect("fit vocab"),
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
        )
        .expect("video dataset");
        assert_eq!(dataset.len(), 2);
        let first = dataset.get(0).expect("first segment");
        assert_eq!(first.payload.frames, 2);
        assert_eq!(first.payload.target_horizon, 2);
        assert_eq!(first.stream.boundary, StreamBoundary::Continue);
        let second = dataset.get(1).expect("second segment");
        assert_eq!(second.payload.target_horizon, 1);
        assert_eq!(second.stream.sample_id.episode_id, 1);
    }

    #[test]
    fn collate_video_segments_preserves_stream_metadata() {
        type Backend = NdArray<f32>;
        let segment = StreamSegment {
            payload: VideoLanguageCpuSample {
                video_tchw: vec![0.0; 2 * 3 * 4 * 4],
                frames: 2,
                channels: 3,
                height: 4,
                width: 4,
                query_q_tokens: vec![1, 2],
                target_y_tokens: vec![3, 4],
                target_horizon: 1,
            },
            stream: StreamStepMetadata {
                sample_id: StreamSampleId {
                    source_id: 7,
                    episode_id: 8,
                    segment_id: 9,
                },
                boundary: StreamBoundary::Continue,
                step_index: 1,
                absolute_time: 3,
            },
        };
        let batch = collate_video_language_segments::<Backend>(&[segment], &Default::default());
        assert_eq!(batch.payload.video_x.shape().dims(), [1, 2, 3, 4, 4]);
        assert_eq!(batch.stream[0].sample_id.segment_id, 9);
    }

    #[test]
    fn multimodal_train_step_runs_on_synthetic_triplets() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.steps = 1;
        config.query_text.n_embd = 32;
        config.target_text.n_embd = 32;
        config.fusion.n_embd = 32;
        config.query_text.n_head = 4;
        config.target_text.n_head = 4;
        config.fusion.n_head = 4;
        config.fusion_dim = 32;
        config.target_dim = 32;
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([2, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([2, 4], &device)),
        };
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 1,
                episode_id: 2,
                segment_id: 3,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let output =
            multimodal_train_step(&model, None, None, batch, &stream, model.init_state(), &config);
        assert_eq!(output.loss.total.shape().dims(), [1]);
        assert_eq!(output.forward.targets.target_embedding_y.shape().dims(), [2, 32]);
    }

    #[test]
    fn multimodal_video_train_step_runs_on_synthetic_triplets() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.projection_hidden_dim = 32;
        config.vision.steps = 1;
        config.query_text.n_embd = 32;
        config.target_text.n_embd = 32;
        config.fusion.n_embd = 32;
        config.query_text.n_head = 4;
        config.target_text.n_head = 4;
        config.fusion.n_head = 4;
        config.fusion_dim = 32;
        config.target_dim = 32;
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let batch = VideoLanguageTripletBatch {
            video_x: Tensor::<Backend, 5>::zeros([2, 3, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([2, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([2, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([2, 4], &device)),
        };
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 4,
                episode_id: 8,
                segment_id: 9,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let output = multimodal_video_train_step(
            &model,
            None,
            None,
            batch,
            &stream,
            model.init_state(),
            &config,
        );
        assert_eq!(output.loss.total.shape().dims(), [1]);
        assert_eq!(output.forward.targets.target_embedding_y.shape().dims(), [2, 32]);
    }

    #[test]
    fn multimodal_eval_step_can_refine_beyond_training_steps() {
        type Backend = NdArray<f32>;
        let device = NdArrayDevice::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 32;
        config.vision.projection_dim = 32;
        config.vision.steps = 1;
        config.query_text.n_embd = 32;
        config.target_text.n_embd = 32;
        config.fusion.n_embd = 32;
        config.query_text.n_head = 4;
        config.target_text.n_head = 4;
        config.fusion.n_head = 4;
        config.fusion_dim = 32;
        config.target_dim = 32;
        config.fusion_refine_steps = 1;
        config.eval_fusion_refine_steps = Some(4);
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
        };
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 10,
                episode_id: 20,
                segment_id: 30,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let train = multimodal_train_step(
            &VlJepaDragon::<Autodiff<NdArray<f32>>>::new(config.clone(), &device),
            None,
            None,
            VisionLanguageTripletBatch {
                vision_x: Tensor::<Autodiff<NdArray<f32>>, 4>::zeros([1, 3, 8, 8], &device),
                query_q_tokens: Tensor::<Autodiff<NdArray<f32>>, 2, Int>::zeros([1, 4], &device),
                query_q_mask: Some(Tensor::<Autodiff<NdArray<f32>>, 2, Bool>::ones([1, 4], &device)),
                target_y_tokens: Tensor::<Autodiff<NdArray<f32>>, 2, Int>::zeros([1, 4], &device),
                target_y_mask: Some(Tensor::<Autodiff<NdArray<f32>>, 2, Bool>::ones([1, 4], &device)),
            },
            &stream,
            VlJepaDragon::<Autodiff<NdArray<f32>>>::new(config.clone(), &device).init_state(),
            &config,
        );
        let eval =
            multimodal_eval_step(&model, None, None, batch, &stream, model.init_state(), &config);
        assert!(eval.forward.state.fusion.position > train.forward.state.fusion.position);
    }

    #[test]
    fn reset_episode_prevents_cross_episode_leakage() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.vision.dropout = 0.0;
        config.query_text.n_embd = 16;
        config.target_text.n_embd = 16;
        config.fusion.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_head = 2;
        config.fusion.n_head = 2;
        config.query_text.dropout = 0.0;
        config.target_text.dropout = 0.0;
        config.fusion.dropout = 0.0;
        config.fusion_dim = 16;
        config.target_dim = 16;
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let batch = VisionLanguageTripletBatch {
            vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
        };
        let first_stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 1,
                episode_id: 1,
                segment_id: 0,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let carried = multimodal_train_step(
            &model,
            None,
            None,
            batch.clone(),
            &first_stream,
            model.init_state(),
            &config,
        )
        .forward
        .state;
        let reset_stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 1,
                episode_id: 2,
                segment_id: 0,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let reset_output = multimodal_train_step(
            &model,
            None,
            None,
            batch.clone(),
            &reset_stream,
            carried,
            &config,
        );
        let fresh_output = multimodal_train_step(
            &model,
            None,
            None,
            batch,
            &reset_stream,
            model.init_state(),
            &config,
        );
        let reset_values = reset_output
            .forward
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .expect("reset values");
        let fresh_values = fresh_output
            .forward
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .expect("fresh values");
        let max_abs_diff = reset_values
            .iter()
            .zip(fresh_values.iter())
            .map(|(left, right)| (left - right).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs_diff < 2e-2, "reset output drifted by {max_abs_diff}");
    }

    #[test]
    fn text_conditioning_changes_video_fusion_outputs() {
        type Backend = Autodiff<NdArray<f32>>;
        let device = NdArrayDevice::default();
        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_embd = 16;
        config.target_text.n_embd = 16;
        config.fusion.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_head = 2;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;
        let model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 7,
                episode_id: 3,
                segment_id: 1,
            },
            boundary: StreamBoundary::ResetEpisode,
            step_index: 0,
            absolute_time: 0,
        };
        let batch_a = VideoLanguageTripletBatch {
            video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
        };
        let batch_b = VideoLanguageTripletBatch {
            video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
            query_q_tokens: Tensor::<Backend, 2, Int>::ones([1, 4], &device),
            query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
            target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
        };
        let out_a = multimodal_video_train_step(
            &model,
            None,
            None,
            batch_a,
            &stream,
            model.init_state(),
            &config,
        );
        let out_b = multimodal_video_train_step(
            &model,
            None,
            None,
            batch_b,
            &stream,
            model.init_state(),
            &config,
        );
        let a = out_a
            .forward
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .expect("a values");
        let b = out_b
            .forward
            .fusion
            .predicted_target_embedding
            .to_data()
            .to_vec::<f32>()
            .expect("b values");
        assert_ne!(a, b);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wgpu_multimodal_image_text_train_step_memory_stays_bounded() {
        type Backend = Autodiff<Wgpu<f32>>;
        let device = WgpuDevice::default();
        init_wgpu_test_runtime(&device);

        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_layer = 2;
        config.query_text.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_layer = 2;
        config.target_text.n_embd = 16;
        config.target_text.n_head = 2;
        config.fusion.n_layer = 2;
        config.fusion.n_embd = 16;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;

        let mut model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<Backend, VlJepaDragon<Backend>>();
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId::new(1, 1, 0),
            boundary: StreamBoundary::Continue,
            step_index: 1,
            absolute_time: 1,
        };

        for _ in 0..2 {
            let batch = VisionLanguageTripletBatch {
                vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
                query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
                target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            };
            let step = multimodal_train_step(
                &model,
                None,
                None,
                batch,
                &stream,
                model.init_state(),
                &config,
            );
            let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
            model = optimizer.step(1.0e-3, model, grads);
        }
        let _ = Backend::sync(&device);

        let mut snapshots = Vec::with_capacity(12);
        for _ in 0..16 {
            let batch = VisionLanguageTripletBatch {
                vision_x: Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device),
                query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
                target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            };
            let step = multimodal_train_step(
                &model,
                None,
                None,
                batch,
                &stream,
                model.init_state(),
                &config,
            );
            let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
            model = optimizer.step(1.0e-3, model, grads);
            let _ = Backend::sync(&device);
            Backend::memory_cleanup(&device);
            let _ = Backend::sync(&device);
            snapshots.push(wgpu_memory_snapshot(&device));
        }

        assert_memory_growth_bounded(
            "multimodal_image_text_wgpu",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn wgpu_multimodal_video_text_train_step_memory_stays_bounded() {
        type Backend = Autodiff<Wgpu<f32>>;
        let device = WgpuDevice::default();
        init_wgpu_test_runtime(&device);

        let mut config = VlJepaDragonConfig::default();
        config.vision.embed_dim = 16;
        config.vision.projection_dim = 16;
        config.vision.projection_hidden_dim = 16;
        config.vision.steps = 1;
        config.query_text.n_layer = 2;
        config.query_text.n_embd = 16;
        config.query_text.n_head = 2;
        config.target_text.n_layer = 2;
        config.target_text.n_embd = 16;
        config.target_text.n_head = 2;
        config.fusion.n_layer = 2;
        config.fusion.n_embd = 16;
        config.fusion.n_head = 2;
        config.fusion_dim = 16;
        config.target_dim = 16;

        let mut model = VlJepaDragon::<Backend>::new(config.clone(), &device);
        let mut optimizer = AdamWConfig::new()
            .with_weight_decay(0.0)
            .init::<Backend, VlJepaDragon<Backend>>();
        let stream = StreamStepMetadata {
            sample_id: StreamSampleId::new(1, 1, 1),
            boundary: StreamBoundary::Continue,
            step_index: 1,
            absolute_time: 1,
        };

        for _ in 0..2 {
            let batch = VideoLanguageTripletBatch {
                video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
                query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
                target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            };
            let step = multimodal_video_train_step(
                &model,
                None,
                None,
                batch,
                &stream,
                model.init_state(),
                &config,
            );
            let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
            model = optimizer.step(1.0e-3, model, grads);
        }
        let _ = Backend::sync(&device);

        let mut snapshots = Vec::with_capacity(12);
        for _ in 0..16 {
            let batch = VideoLanguageTripletBatch {
                video_x: Tensor::<Backend, 5>::zeros([1, 2, 3, 8, 8], &device),
                query_q_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                query_q_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
                target_y_tokens: Tensor::<Backend, 2, Int>::zeros([1, 4], &device),
                target_y_mask: Some(Tensor::<Backend, 2, Bool>::ones([1, 4], &device)),
            };
            let step = multimodal_video_train_step(
                &model,
                None,
                None,
                batch,
                &stream,
                model.init_state(),
                &config,
            );
            let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
            model = optimizer.step(1.0e-3, model, grads);
            let _ = Backend::sync(&device);
            Backend::memory_cleanup(&device);
            let _ = Backend::sync(&device);
            snapshots.push(wgpu_memory_snapshot(&device));
        }

        assert_memory_growth_bounded(
            "multimodal_video_text_wgpu",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }
}
