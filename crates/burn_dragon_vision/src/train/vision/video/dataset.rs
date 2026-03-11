use crate::train::prelude::*;
use burn::data::dataloader::{DataLoaderIterator, Progress};
use burn_dataset::Dataset;
use burn_dataset::vision::{MnistDataset, MnistItem};
use image::imageops::FilterType;
use image::{GrayImage, Luma};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

#[derive(Clone)]
pub struct VideoClipBatch<B: BackendTrait> {
    pub clip_frames: Tensor<B, 5>,
    pub labels: Tensor<B, 1, Int>,
    pub context_len: usize,
    pub target_len: usize,
    pub capture_artifacts: bool,
}

impl<B: BackendTrait> VideoClipBatch<B> {
    pub fn new(
        clip_frames: Tensor<B, 5>,
        labels: Tensor<B, 1, Int>,
        context_len: usize,
        target_len: usize,
    ) -> Self {
        Self {
            clip_frames,
            labels,
            context_len,
            target_len,
            capture_artifacts: false,
        }
    }

    pub fn repeat_batch(&self, repeats: usize) -> Self {
        let repeats = repeats.max(1);
        if repeats == 1 {
            return self.clone();
        }
        Self {
            clip_frames: self.clip_frames.clone().repeat_dim(0, repeats),
            labels: self.labels.clone().repeat_dim(0, repeats),
            context_len: self.context_len,
            target_len: self.target_len,
            capture_artifacts: self.capture_artifacts,
        }
    }

    pub fn available_future_len(&self) -> usize {
        self.clip_frames.shape().dims::<5>()[1].saturating_sub(self.context_len)
    }

    pub fn with_target_len(mut self, target_len: usize) -> Self {
        let available = self.available_future_len().max(1);
        self.target_len = target_len.clamp(1, available);
        self
    }

    pub fn with_capture_artifacts(mut self, capture_artifacts: bool) -> Self {
        self.capture_artifacts = capture_artifacts;
        self
    }

    pub fn take_prefix(&self, limit: usize) -> Self {
        let batch = self.clip_frames.shape().dims::<5>()[0];
        let take = limit.max(1).min(batch);
        if take == batch {
            return self.clone();
        }
        Self {
            clip_frames: self.clip_frames.clone().slice_dim(0, 0..take),
            labels: self.labels.clone().slice_dim(0, 0..take),
            context_len: self.context_len,
            target_len: self.target_len,
            capture_artifacts: self.capture_artifacts,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MovingMnistSplit {
    Train,
    Val,
}

#[derive(Clone, Debug)]
pub struct MovingMnistVideoDatasetConfig {
    pub split: MovingMnistSplit,
    pub frame_size: usize,
    pub digit_size: usize,
    pub in_channels: usize,
    pub context_len: usize,
    pub target_len: usize,
    pub extra_future_frames: usize,
    pub frame_stride: usize,
    pub max_records: Option<usize>,
    pub normalize: VisionNormalize,
    pub min_velocity: f32,
    pub max_velocity: f32,
    pub seed: u64,
}

#[derive(Clone)]
pub struct MovingMnistDigit {
    pub pixels: Vec<f32>,
    pub label: u8,
    pub size: usize,
}

#[derive(Clone)]
pub(crate) struct MovingMnistVideoClip {
    frames: Vec<f32>,
    label: i64,
}

#[derive(Clone, Debug)]
struct MovingMnistVideoBatchData {
    frames: Vec<f32>,
    labels: Vec<i64>,
    batch_size: usize,
    clip_len: usize,
    channels: usize,
    frame_size: usize,
    context_len: usize,
    target_len: usize,
}

#[derive(Clone, Debug)]
pub struct VideoTargetHorizonCurriculum {
    pub min_target_len: usize,
    pub max_target_len: usize,
    pub warmup_steps: usize,
    pub seed: u64,
}

impl VideoTargetHorizonCurriculum {
    pub fn sample_target_len(&self, step: usize, available_future_len: usize) -> usize {
        let available_future_len = available_future_len.max(1);
        let min_target_len = self.min_target_len.clamp(1, available_future_len);
        let max_target_len = self
            .max_target_len
            .clamp(min_target_len, available_future_len);
        if min_target_len >= max_target_len {
            return min_target_len;
        }

        let ramp_steps = self.warmup_steps.max(1);
        let progress = if self.warmup_steps == 0 {
            1.0
        } else {
            (step as f32 / ramp_steps as f32).clamp(0.0, 1.0)
        };
        let span = max_target_len.saturating_sub(min_target_len);
        let current_max = min_target_len + ((span as f32) * progress).round() as usize;
        let current_max = current_max.clamp(min_target_len, max_target_len);
        if current_max <= min_target_len {
            return min_target_len;
        }

        let sample_span = current_max - min_target_len + 1;
        min_target_len + (splitmix64(self.seed ^ step as u64) as usize % sample_span)
    }
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

impl MovingMnistVideoBatchData {
    fn into_batch<B: BackendTrait>(self, device: &B::Device) -> VideoClipBatch<B> {
        let clip_frames = Tensor::<B, 5>::from_data(
            TensorData::new(
                self.frames,
                [
                    self.batch_size,
                    self.clip_len,
                    self.channels,
                    self.frame_size,
                    self.frame_size,
                ],
            ),
            device,
        );
        let labels =
            Tensor::<B, 1, Int>::from_data(TensorData::new(self.labels, [self.batch_size]), device);
        VideoClipBatch::new(clip_frames, labels, self.context_len, self.target_len)
    }
}

pub struct MovingMnistVideoDataset {
    digits: Vec<MovingMnistDigit>,
    config: MovingMnistVideoDatasetConfig,
}

impl MovingMnistVideoDataset {
    pub fn new_from_mnist(config: MovingMnistVideoDatasetConfig) -> Result<Self> {
        let digits = match config.split {
            MovingMnistSplit::Train => {
                load_digits(MnistDataset::train(), config.digit_size, config.max_records)
            }
            MovingMnistSplit::Val => {
                load_digits(MnistDataset::test(), config.digit_size, config.max_records)
            }
        };
        Self::from_digits(digits, config)
    }

    pub fn from_digits(
        digits: Vec<MovingMnistDigit>,
        config: MovingMnistVideoDatasetConfig,
    ) -> Result<Self> {
        if digits.is_empty() {
            return Err(anyhow!("moving mnist dataset is empty"));
        }
        Ok(Self { digits, config })
    }

    pub fn ensure_mnist_downloaded() {
        let _ = MnistDataset::train();
        let _ = MnistDataset::test();
    }

    pub fn len(&self) -> usize {
        self.config
            .max_records
            .unwrap_or(self.digits.len())
            .min(self.digits.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn steps_per_epoch(&self, batch_size: usize) -> usize {
        let batch_size = batch_size.max(1);
        self.len().div_ceil(batch_size).max(1)
    }

    pub fn num_classes(&self) -> usize {
        10
    }

    fn clip_len_with_extra_future(&self, extra_future_frames: usize) -> usize {
        self.config.context_len + self.config.target_len + extra_future_frames
    }

    fn render_clip_with_extra_future(
        &self,
        index: usize,
        extra_future_frames: usize,
    ) -> MovingMnistVideoClip {
        let len = self.len().max(1);
        let digit = &self.digits[index % len];
        let clip_len = self.clip_len_with_extra_future(extra_future_frames);
        let frame_size = self.config.frame_size.max(1);
        let digit_size = digit.size.max(1).min(frame_size);
        let max_offset = frame_size.saturating_sub(digit_size);
        let seed = self.config.seed ^ ((index as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let mut rng = StdRng::seed_from_u64(seed);
        let mut x = rng.gen_range(0.0..=(max_offset as f32));
        let mut y = rng.gen_range(0.0..=(max_offset as f32));
        let mut vx = rng.gen_range(self.config.min_velocity..=self.config.max_velocity);
        let mut vy = rng.gen_range(self.config.min_velocity..=self.config.max_velocity);
        if rng.gen_bool(0.5) {
            vx = -vx;
        }
        if rng.gen_bool(0.5) {
            vy = -vy;
        }

        let pixels_per_frame = frame_size * frame_size;
        let channels = self.config.in_channels.max(1);
        let mut frames = Vec::with_capacity(clip_len * channels * pixels_per_frame);

        for _ in 0..clip_len {
            let mut canvas = vec![0.0_f32; pixels_per_frame];
            let x0 = x.round().clamp(0.0, max_offset as f32) as usize;
            let y0 = y.round().clamp(0.0, max_offset as f32) as usize;
            for dy in 0..digit_size {
                let src_row = dy * digit_size;
                let dst_row = (y0 + dy) * frame_size;
                for dx in 0..digit_size {
                    let dst = dst_row + x0 + dx;
                    canvas[dst] = digit.pixels[src_row + dx];
                }
            }

            for channel in 0..channels {
                let mean = self.config.normalize.mean()[channel.min(2)];
                let std = self.config.normalize.std()[channel.min(2)];
                for value in canvas.iter().copied() {
                    frames.push((value - mean) / std);
                }
            }

            for _ in 0..self.config.frame_stride.max(1) {
                x += vx;
                y += vy;
                if x <= 0.0 {
                    x = 0.0;
                    vx = vx.abs();
                } else if x >= max_offset as f32 {
                    x = max_offset as f32;
                    vx = -vx.abs();
                }
                if y <= 0.0 {
                    y = 0.0;
                    vy = vy.abs();
                } else if y >= max_offset as f32 {
                    y = max_offset as f32;
                    vy = -vy.abs();
                }
            }
        }

        MovingMnistVideoClip {
            frames,
            label: i64::from(digit.label),
        }
    }

    fn render_clip(&self, index: usize) -> MovingMnistVideoClip {
        self.render_clip_with_extra_future(index, self.config.extra_future_frames)
    }

    pub(crate) fn get(&self, index: usize) -> Option<MovingMnistVideoClip> {
        if index >= self.len() {
            return None;
        }
        Some(self.render_clip(index))
    }

    pub fn sample_batch<B: BackendTrait>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> VideoClipBatch<B> {
        self.sample_batch_data(batch_size).into_batch(device)
    }

    pub fn sample_batch_with_extra_future<B: BackendTrait>(
        &self,
        batch_size: usize,
        device: &B::Device,
        extra_future_frames: usize,
    ) -> VideoClipBatch<B> {
        self.sample_batch_data_with_extra_future(batch_size, extra_future_frames)
            .into_batch(device)
    }

    fn sample_batch_data(&self, batch_size: usize) -> MovingMnistVideoBatchData {
        self.sample_batch_data_with_extra_future(batch_size, self.config.extra_future_frames)
    }

    fn sample_batch_data_with_extra_future(
        &self,
        batch_size: usize,
        extra_future_frames: usize,
    ) -> MovingMnistVideoBatchData {
        let batch_size = batch_size.max(1);
        let len = self.len().max(1);
        let mut rng = thread_rng();
        let indices = (0..batch_size)
            .map(|_| rng.gen_range(0..len))
            .collect::<Vec<_>>();
        self.batch_data_from_indices_with_extra_future(&indices, extra_future_frames)
    }

    pub fn batch_from_indices<B: BackendTrait>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> VideoClipBatch<B> {
        self.batch_data_from_indices(indices).into_batch(device)
    }

    fn batch_data_from_indices(&self, indices: &[usize]) -> MovingMnistVideoBatchData {
        self.batch_data_from_indices_with_extra_future(indices, self.config.extra_future_frames)
    }

    fn batch_data_from_indices_with_extra_future(
        &self,
        indices: &[usize],
        extra_future_frames: usize,
    ) -> MovingMnistVideoBatchData {
        let clip_len = self.clip_len_with_extra_future(extra_future_frames);
        let frame_size = self.config.frame_size.max(1);
        let channels = self.config.in_channels.max(1);
        let mut frames =
            Vec::with_capacity(indices.len() * clip_len * channels * frame_size * frame_size);
        let mut labels = Vec::with_capacity(indices.len());

        for &index in indices {
            let clip = if extra_future_frames == self.config.extra_future_frames {
                self.get(index)
                    .unwrap_or_else(|| panic!("moving mnist index {index} out of bounds"))
            } else {
                self.render_clip_with_extra_future(index, extra_future_frames)
            };
            frames.extend_from_slice(&clip.frames);
            labels.push(clip.label);
        }

        MovingMnistVideoBatchData {
            frames,
            labels,
            batch_size: indices.len(),
            clip_len,
            channels,
            frame_size,
            context_len: self.config.context_len,
            target_len: self.config.target_len,
        }
    }
}

pub struct MovingMnistVideoDataLoader<B: BackendTrait> {
    dataset: Arc<MovingMnistVideoDataset>,
    batch_size: usize,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    target_horizon_curriculum: Option<VideoTargetHorizonCurriculum>,
    device: B::Device,
    prefetch_batches: usize,
    prefetch_workers: usize,
    prefetch_to_device: bool,
    sequential: bool,
    artifact_capture_every: usize,
    artifact_capture_images: usize,
    artifact_extra_future_frames: usize,
}

impl<B: BackendTrait> Clone for MovingMnistVideoDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            target_horizon_curriculum: self.target_horizon_curriculum.clone(),
            device: self.device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
            sequential: self.sequential,
            artifact_capture_every: self.artifact_capture_every,
            artifact_capture_images: self.artifact_capture_images,
            artifact_extra_future_frames: self.artifact_extra_future_frames,
        }
    }
}

impl<B: BackendTrait> MovingMnistVideoDataLoader<B> {
    pub fn new(
        dataset: Arc<MovingMnistVideoDataset>,
        batch_size: usize,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        target_horizon_curriculum: Option<VideoTargetHorizonCurriculum>,
        prefetch_batches: usize,
        prefetch_workers: usize,
        prefetch_to_device: bool,
        sequential: bool,
        artifact_capture_every: usize,
        artifact_capture_images: usize,
        artifact_extra_future_frames: usize,
    ) -> Self {
        let steps_per_epoch = if steps_per_epoch == 0 {
            dataset.steps_per_epoch(batch_size)
        } else {
            steps_per_epoch
        }
        .max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let prefetch_batches = if sequential {
            0
        } else {
            prefetch_batches.min(steps_per_epoch)
        };
        let prefetch_to_device = prefetch_to_device && prefetch_batches > 0;
        let min_prefetch = if prefetch_to_device { 2 } else { 1 };
        let prefetch_workers = if prefetch_batches == 0 {
            0
        } else {
            prefetch_workers.max(1)
        };
        let prefetch_batches = if prefetch_batches == 0 {
            0
        } else {
            prefetch_batches
                .max(min_prefetch)
                .max(prefetch_workers)
                .min(steps_per_epoch)
        };
        Self {
            dataset,
            batch_size: batch_size.max(1),
            steps_per_epoch,
            total_steps,
            consumed_steps,
            target_horizon_curriculum,
            device: device.clone(),
            prefetch_batches,
            prefetch_workers,
            prefetch_to_device,
            sequential,
            artifact_capture_every,
            artifact_capture_images,
            artifact_extra_future_frames,
        }
    }
}

enum MovingMnistVideoPrefetchItem<B: BackendTrait> {
    Batch(VideoClipBatch<B>),
    Data(Box<MovingMnistVideoBatchData>),
}

struct MovingMnistVideoPrefetcher<B: BackendTrait> {
    rx: Option<mpsc::Receiver<Result<MovingMnistVideoPrefetchItem<B>>>>,
    stop: Arc<AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl<B> MovingMnistVideoPrefetcher<B>
where
    B: BackendTrait + 'static,
    B::Device: Clone + Send + Sync + 'static,
    VideoClipBatch<B>: Send,
{
    fn new(
        dataset: Arc<MovingMnistVideoDataset>,
        batch_size: usize,
        steps_total: usize,
        prefetch_batches: usize,
        prefetch_workers: usize,
        device: &B::Device,
        prefetch_to_device: bool,
    ) -> Self {
        let workers = prefetch_workers.max(1);
        let queue_len = prefetch_batches.max(1);
        let (tx, rx) = mpsc::sync_channel(queue_len);
        let remaining = Arc::new(AtomicUsize::new(steps_total));
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(workers + if prefetch_to_device { 1 } else { 0 });

        if prefetch_to_device {
            let (data_tx, data_rx) = mpsc::sync_channel(queue_len);
            for _ in 0..workers {
                let dataset = Arc::clone(&dataset);
                let tx = data_tx.clone();
                let remaining = Arc::clone(&remaining);
                let stop = Arc::clone(&stop);
                let handle = thread::spawn(move || {
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let decremented =
                            remaining.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                                value.checked_sub(1)
                            });
                        if decremented.is_err() {
                            break;
                        }
                        let result = Ok(dataset.sample_batch_data(batch_size));
                        if tx.send(result).is_err() {
                            break;
                        }
                    }
                });
                handles.push(handle);
            }
            drop(data_tx);

            let device = device.clone();
            let stop = Arc::clone(&stop);
            let handle = thread::spawn(move || {
                for data in data_rx {
                    if stop.load(Ordering::Relaxed) {
                        break;
                    }
                    crate::device::pin_stream_zero();
                    let item = match data {
                        Ok(data) => {
                            let _guard = crate::device::device_allocation_lock().lock().ok();
                            Ok(MovingMnistVideoPrefetchItem::Batch(
                                data.into_batch::<B>(&device),
                            ))
                        }
                        Err(err) => Err(err),
                    };
                    if tx.send(item).is_err() {
                        break;
                    }
                }
            });
            handles.push(handle);
        } else {
            for _ in 0..workers {
                let dataset = Arc::clone(&dataset);
                let tx = tx.clone();
                let remaining = Arc::clone(&remaining);
                let stop = Arc::clone(&stop);
                let handle = thread::spawn(move || {
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let decremented =
                            remaining.fetch_update(Ordering::AcqRel, Ordering::Acquire, |value| {
                                value.checked_sub(1)
                            });
                        if decremented.is_err() {
                            break;
                        }
                        let result = Ok(MovingMnistVideoPrefetchItem::Data(Box::new(
                            dataset.sample_batch_data(batch_size),
                        )));
                        if tx.send(result).is_err() {
                            break;
                        }
                    }
                });
                handles.push(handle);
            }
            drop(tx);
        }

        Self {
            rx: Some(rx),
            stop,
            handles,
        }
    }

    fn recv(&mut self) -> Option<Result<MovingMnistVideoPrefetchItem<B>>> {
        let rx = self.rx.as_ref()?;
        rx.recv().ok()
    }
}

impl<B> Drop for MovingMnistVideoPrefetcher<B>
where
    B: BackendTrait,
{
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.rx.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

impl<B> DataLoader<B, VideoClipBatch<B>> for MovingMnistVideoDataLoader<B>
where
    B: BackendTrait + 'static,
    B::Device: Clone + Send + Sync + 'static,
    VideoClipBatch<B>: Send,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<VideoClipBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        let prefetcher = if !self.sequential && self.prefetch_batches > 0 && steps_total > 0 {
            Some(MovingMnistVideoPrefetcher::new(
                Arc::clone(&self.dataset),
                self.batch_size,
                steps_total,
                self.prefetch_batches,
                self.prefetch_workers,
                &self.device,
                self.prefetch_to_device,
            ))
        } else {
            None
        };

        Box::new(MovingMnistVideoIterator {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            device: self.device.clone(),
            steps_total,
            step: 0,
            sequential: self.sequential,
            cursor: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            target_horizon_curriculum: self.target_horizon_curriculum.clone(),
            prefetcher,
            artifact_capture_every: self.artifact_capture_every,
            artifact_capture_images: self.artifact_capture_images,
            artifact_images_emitted: 0,
            artifact_extra_future_frames: self.artifact_extra_future_frames,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.batch_size
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, VideoClipBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            target_horizon_curriculum: self.target_horizon_curriculum.clone(),
            device: device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
            sequential: self.sequential,
            artifact_capture_every: self.artifact_capture_every,
            artifact_capture_images: self.artifact_capture_images,
            artifact_extra_future_frames: self.artifact_extra_future_frames,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, VideoClipBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: steps,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            target_horizon_curriculum: self.target_horizon_curriculum.clone(),
            device: self.device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
            sequential: self.sequential,
            artifact_capture_every: self.artifact_capture_every,
            artifact_capture_images: self.artifact_capture_images,
            artifact_extra_future_frames: self.artifact_extra_future_frames,
        })
    }
}

struct MovingMnistVideoIterator<B: BackendTrait> {
    dataset: Arc<MovingMnistVideoDataset>,
    batch_size: usize,
    device: B::Device,
    steps_total: usize,
    step: usize,
    sequential: bool,
    cursor: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    target_horizon_curriculum: Option<VideoTargetHorizonCurriculum>,
    prefetcher: Option<MovingMnistVideoPrefetcher<B>>,
    artifact_capture_every: usize,
    artifact_capture_images: usize,
    artifact_images_emitted: usize,
    artifact_extra_future_frames: usize,
}

impl<B: BackendTrait> Iterator for MovingMnistVideoIterator<B> {
    type Item = VideoClipBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;
        crate::device::pin_stream_zero();

        let global_step = if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
                previous
            } else {
                counter.fetch_add(1, Ordering::Relaxed)
            }
        } else {
            self.step - 1
        };

        let capture_artifacts = self.artifact_capture_every > 0
            && self.artifact_capture_images > self.artifact_images_emitted
            && self.step.is_multiple_of(self.artifact_capture_every);
        let artifact_extra_future_frames = if capture_artifacts {
            self.artifact_extra_future_frames
        } else {
            0
        };

        let mut batch = if let Some(prefetcher) = &mut self.prefetcher {
            match prefetcher.recv() {
                Some(Ok(MovingMnistVideoPrefetchItem::Batch(batch)))
                    if artifact_extra_future_frames == 0 =>
                {
                    batch
                }
                Some(Ok(MovingMnistVideoPrefetchItem::Data(data)))
                    if artifact_extra_future_frames == 0 =>
                {
                    (*data).into_batch(&self.device)
                }
                Some(Ok(_)) => self.dataset.sample_batch_with_extra_future(
                    self.batch_size,
                    &self.device,
                    artifact_extra_future_frames,
                ),
                Some(Err(err)) => panic!("moving mnist prefetch error: {err}"),
                None => panic!("moving mnist prefetch channel closed early"),
            }
        } else if self.sequential {
            let len = self.dataset.len().max(1);
            let indices = (0..self.batch_size)
                .map(|offset| (self.cursor + offset) % len)
                .collect::<Vec<_>>();
            self.cursor = (self.cursor + self.batch_size) % len;
            if artifact_extra_future_frames > 0 {
                self.dataset
                    .batch_data_from_indices_with_extra_future(
                        &indices,
                        artifact_extra_future_frames,
                    )
                    .into_batch(&self.device)
            } else {
                self.dataset.batch_from_indices(&indices, &self.device)
            }
        } else {
            if artifact_extra_future_frames > 0 {
                self.dataset.sample_batch_with_extra_future(
                    self.batch_size,
                    &self.device,
                    artifact_extra_future_frames,
                )
            } else {
                self.dataset.sample_batch(self.batch_size, &self.device)
            }
        };
        if let Some(curriculum) = &self.target_horizon_curriculum {
            let available_future_len = batch.available_future_len();
            let target_len = curriculum.sample_target_len(global_step, available_future_len);
            batch = batch.with_target_len(target_len);
        }
        if capture_artifacts {
            let remaining = self
                .artifact_capture_images
                .saturating_sub(self.artifact_images_emitted);
            self.artifact_images_emitted += remaining.min(self.batch_size);
            batch = batch.with_capture_artifacts(true);
        }
        Some(batch)
    }
}

impl<B: BackendTrait> DataLoaderIterator<VideoClipBatch<B>> for MovingMnistVideoIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.batch_size,
            self.steps_total * self.batch_size,
        )
    }
}

fn load_digits<D>(
    dataset: D,
    digit_size: usize,
    max_records: Option<usize>,
) -> Vec<MovingMnistDigit>
where
    D: Dataset<MnistItem>,
{
    let limit = max_records.unwrap_or(dataset.len()).min(dataset.len());
    (0..limit)
        .filter_map(|index| dataset.get(index))
        .map(|item| MovingMnistDigit {
            pixels: resize_mnist_digit(&item, digit_size),
            label: item.label,
            size: digit_size.max(1),
        })
        .collect()
}

fn resize_mnist_digit(item: &MnistItem, digit_size: usize) -> Vec<f32> {
    let digit_size = digit_size.max(1);
    let mut image = GrayImage::new(28, 28);
    for (y, row) in item.image.iter().enumerate() {
        for (x, value) in row.iter().enumerate() {
            image.put_pixel(x as u32, y as u32, Luma([value.clamp(0.0, 255.0) as u8]));
        }
    }
    let resized = if digit_size == 28 {
        image
    } else {
        image::imageops::resize(
            &image,
            digit_size as u32,
            digit_size as u32,
            FilterType::Triangle,
        )
    };
    resized
        .into_raw()
        .into_iter()
        .map(|value| value as f32 / 255.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn moving_mnist_batch_shapes_and_split_lengths() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let digit = MovingMnistDigit {
            pixels: vec![1.0; 4 * 4],
            label: 7,
            size: 4,
        };
        let dataset = MovingMnistVideoDataset::from_digits(
            vec![digit; 3],
            MovingMnistVideoDatasetConfig {
                split: MovingMnistSplit::Train,
                frame_size: 8,
                digit_size: 4,
                in_channels: 3,
                context_len: 3,
                target_len: 2,
                extra_future_frames: 3,
                frame_stride: 1,
                max_records: None,
                normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
                min_velocity: 1.0,
                max_velocity: 1.0,
                seed: 123,
            },
        )
        .expect("dataset");

        let batch = dataset.sample_batch::<Backend>(2, &device);
        assert_eq!(batch.context_len, 3);
        assert_eq!(batch.target_len, 2);
        assert_eq!(batch.clip_frames.shape().dims::<5>(), [2, 8, 3, 8, 8]);
        assert_eq!(batch.labels.shape().dims::<1>(), [2]);
    }

    #[test]
    fn moving_mnist_generation_is_deterministic_for_index() {
        let digit = MovingMnistDigit {
            pixels: vec![0.5; 4 * 4],
            label: 3,
            size: 4,
        };
        let dataset = MovingMnistVideoDataset::from_digits(
            vec![digit; 2],
            MovingMnistVideoDatasetConfig {
                split: MovingMnistSplit::Train,
                frame_size: 8,
                digit_size: 4,
                in_channels: 3,
                context_len: 2,
                target_len: 2,
                extra_future_frames: 0,
                frame_stride: 1,
                max_records: None,
                normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
                min_velocity: 1.0,
                max_velocity: 1.0,
                seed: 99,
            },
        )
        .expect("dataset");

        let a = dataset.get(1).expect("clip a");
        let b = dataset.get(1).expect("clip b");
        assert_eq!(a.label, b.label);
        assert_eq!(a.frames, b.frames);
    }

    #[test]
    fn moving_mnist_prefetch_loader_yields_batches() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let digit = MovingMnistDigit {
            pixels: vec![1.0; 4 * 4],
            label: 5,
            size: 4,
        };
        let dataset = Arc::new(
            MovingMnistVideoDataset::from_digits(
                vec![digit; 16],
                MovingMnistVideoDatasetConfig {
                    split: MovingMnistSplit::Train,
                    frame_size: 8,
                    digit_size: 4,
                    in_channels: 3,
                    context_len: 3,
                    target_len: 2,
                    extra_future_frames: 0,
                    frame_stride: 1,
                    max_records: None,
                    normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
                    min_velocity: 1.0,
                    max_velocity: 1.0,
                    seed: 7,
                },
            )
            .expect("dataset"),
        );
        let loader = MovingMnistVideoDataLoader::<Backend>::new(
            dataset, 4, &device, 3, None, None, 2, 2, true, false, 0, 0, 0,
        );

        let mut iter = loader.iter();
        let batch = iter.next().expect("first batch");
        assert_eq!(batch.clip_frames.shape().dims::<5>(), [4, 5, 3, 8, 8]);
        assert_eq!(batch.labels.shape().dims::<1>(), [4]);
    }

    #[test]
    fn moving_mnist_curriculum_loader_samples_target_horizon_range() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let digit = MovingMnistDigit {
            pixels: vec![1.0; 4 * 4],
            label: 5,
            size: 4,
        };
        let dataset = Arc::new(
            MovingMnistVideoDataset::from_digits(
                vec![digit; 16],
                MovingMnistVideoDatasetConfig {
                    split: MovingMnistSplit::Train,
                    frame_size: 8,
                    digit_size: 4,
                    in_channels: 3,
                    context_len: 3,
                    target_len: 6,
                    extra_future_frames: 0,
                    frame_stride: 1,
                    max_records: None,
                    normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
                    min_velocity: 1.0,
                    max_velocity: 1.0,
                    seed: 7,
                },
            )
            .expect("dataset"),
        );
        let loader = MovingMnistVideoDataLoader::<Backend>::new(
            dataset,
            4,
            &device,
            6,
            Some(6),
            Some(VideoTargetHorizonCurriculum {
                min_target_len: 2,
                max_target_len: 6,
                warmup_steps: 4,
                seed: 123,
            }),
            0,
            0,
            false,
            true,
            0,
            0,
            0,
        );

        let lengths = loader
            .iter()
            .map(|batch| batch.target_len)
            .collect::<Vec<_>>();
        assert_eq!(lengths.len(), 6);
        assert!(lengths[0] >= 2 && lengths[0] <= 2);
        assert!(lengths[1] >= 2 && lengths[1] <= 3);
        assert!(lengths[2] >= 2 && lengths[2] <= 4);
        assert!(lengths[3] >= 2 && lengths[3] <= 5);
        assert!(lengths[4] >= 2 && lengths[4] <= 6);
        assert!(lengths[5] >= 2 && lengths[5] <= 6);
    }

    #[test]
    fn moving_mnist_loader_only_marks_artifact_batches_within_budget() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let digit = MovingMnistDigit {
            pixels: vec![1.0; 4 * 4],
            label: 3,
            size: 4,
        };
        let dataset = Arc::new(
            MovingMnistVideoDataset::from_digits(
                vec![digit; 32],
                MovingMnistVideoDatasetConfig {
                    split: MovingMnistSplit::Val,
                    frame_size: 8,
                    digit_size: 4,
                    in_channels: 3,
                    context_len: 3,
                    target_len: 2,
                    extra_future_frames: 0,
                    frame_stride: 1,
                    max_records: None,
                    normalize: VisionNormalize::new([0.0; 3], [1.0; 3]),
                    min_velocity: 1.0,
                    max_velocity: 1.0,
                    seed: 11,
                },
            )
            .expect("dataset"),
        );
        let loader = MovingMnistVideoDataLoader::<Backend>::new(
            dataset, 8, &device, 4, None, None, 0, 0, false, true, 1, 8, 6,
        );

        let batches = loader.iter().collect::<Vec<_>>();
        assert_eq!(batches.len(), 4);
        assert!(batches[0].capture_artifacts);
        assert!(batches[1..].iter().all(|batch| !batch.capture_artifacts));
        assert_eq!(batches[0].clip_frames.shape().dims::<5>()[1], 11);
        assert!(
            batches[1..]
                .iter()
                .all(|batch| batch.clip_frames.shape().dims::<5>()[1] == 5)
        );
    }
}
