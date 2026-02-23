use anyhow::{Result, anyhow};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use rand::prelude::*;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;

const IMAGE_CHANNELS: usize = 3;
const BYTES_PER_F32: u64 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageNetSplit {
    Train,
    Val,
}

#[derive(Clone, Debug)]
pub struct ImageNetAugmentations {
    split: ImageNetSplit,
    image_size: u32,
    resize_short: u32,
    min_scale: f32,
    max_scale: f32,
    min_aspect_ratio: f32,
    max_aspect_ratio: f32,
    flip_prob: f32,
    color_jitter_prob: f32,
    brightness: f32,
    contrast: f32,
    saturation: f32,
    hue: f32,
    grayscale_prob: f32,
    blur_prob: f32,
    blur_sigma_min: f32,
    blur_sigma_max: f32,
    solarize_prob: f32,
    solarize_threshold: u8,
}

#[derive(Clone, Copy, Debug)]
struct CropParams {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn crop_overlap_ratio(a: CropParams, b: CropParams) -> f32 {
    let x1 = a.x.max(b.x);
    let y1 = a.y.max(b.y);
    let x2 = (a.x + a.width).min(b.x + b.width);
    let y2 = (a.y + a.height).min(b.y + b.height);
    let inter_w = x2.saturating_sub(x1);
    let inter_h = y2.saturating_sub(y1);
    let inter_area = (inter_w * inter_h) as f32;
    let area_a = (a.width * a.height) as f32;
    let area_b = (b.width * b.height) as f32;
    let denom = area_a.min(area_b).max(1.0);
    inter_area / denom
}

fn push_crop_normalized(buffer: &mut Vec<f32>, crop: CropParams, width: u32, height: u32) {
    let width = width.max(1) as f32;
    let height = height.max(1) as f32;
    buffer.extend_from_slice(&[
        crop.x as f32 / width,
        crop.y as f32 / height,
        crop.width as f32 / width,
        crop.height as f32 / height,
    ]);
}

impl ImageNetAugmentations {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        split: ImageNetSplit,
        image_size: usize,
        resize_short: usize,
        min_scale: f32,
        max_scale: f32,
        min_aspect_ratio: f32,
        max_aspect_ratio: f32,
        flip_prob: f32,
        color_jitter_prob: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
        grayscale_prob: f32,
        blur_prob: f32,
        blur_sigma_min: f32,
        blur_sigma_max: f32,
        solarize_prob: f32,
        solarize_threshold: u8,
    ) -> Self {
        Self {
            split,
            image_size: image_size as u32,
            resize_short: resize_short as u32,
            min_scale,
            max_scale,
            min_aspect_ratio,
            max_aspect_ratio,
            flip_prob,
            color_jitter_prob,
            brightness,
            contrast,
            saturation,
            hue,
            grayscale_prob,
            blur_prob,
            blur_sigma_min,
            blur_sigma_max,
            solarize_prob,
            solarize_threshold,
        }
    }

    pub fn image_size(&self) -> usize {
        self.image_size as usize
    }

    pub fn is_deterministic(&self) -> bool {
        match self.split {
            ImageNetSplit::Val => true,
            ImageNetSplit::Train => {
                self.flip_prob <= 0.0
                    && self.color_jitter_prob <= 0.0
                    && self.grayscale_prob <= 0.0
                    && self.blur_prob <= 0.0
                    && self.solarize_prob <= 0.0
                    && (self.min_scale - 1.0).abs() <= f32::EPSILON
                    && (self.max_scale - 1.0).abs() <= f32::EPSILON
                    && (self.min_aspect_ratio - 1.0).abs() <= f32::EPSILON
                    && (self.max_aspect_ratio - 1.0).abs() <= f32::EPSILON
            }
        }
    }

    pub fn apply(&self, image: &DynamicImage, rng: &mut impl Rng) -> RgbImage {
        match self.split {
            ImageNetSplit::Train => self.apply_train(image, rng),
            ImageNetSplit::Val => self.apply_val(image),
        }
    }

    fn apply_train(&self, image: &DynamicImage, rng: &mut impl Rng) -> RgbImage {
        if self.is_deterministic() {
            return self.apply_val(image);
        }

        let crop = self.random_resized_crop_params(image, rng);
        self.apply_train_with_crop(image, rng, crop)
    }

    fn apply_train_with_crop(
        &self,
        image: &DynamicImage,
        rng: &mut impl Rng,
        crop: CropParams,
    ) -> RgbImage {
        let mut image = self.apply_crop(image, crop);

        if self.flip_prob > 0.0 && rng.r#gen::<f32>() < self.flip_prob {
            image = image.fliph();
        }

        if self.color_jitter_prob > 0.0 && rng.r#gen::<f32>() < self.color_jitter_prob {
            if self.brightness > 0.0 {
                let delta = rng.gen_range(-self.brightness..self.brightness) * 255.0;
                image = image.brighten(delta as i32);
            }
            if self.contrast > 0.0 {
                let delta = rng.gen_range(-self.contrast..self.contrast) * 100.0;
                image = image.adjust_contrast(delta);
            }
            if self.saturation > 0.0 {
                let delta = rng.gen_range(-self.saturation..self.saturation);
                let factor = (1.0 + delta).max(0.0);
                image = self.adjust_saturation(image, factor);
            }
            if self.hue > 0.0 {
                let delta = rng.gen_range(-self.hue..self.hue) * 180.0;
                image = image.huerotate(delta as i32);
            }
        }

        if self.grayscale_prob > 0.0 && rng.r#gen::<f32>() < self.grayscale_prob {
            image = image.grayscale();
        }

        if self.blur_prob > 0.0 && rng.r#gen::<f32>() < self.blur_prob {
            let sigma_min = self.blur_sigma_min.max(0.0);
            let sigma_max = self.blur_sigma_max.max(sigma_min);
            if sigma_max > 0.0 {
                let sigma = rng.gen_range(sigma_min..=sigma_max);
                image = image.blur(sigma);
            }
        }

        if self.solarize_prob > 0.0 && rng.r#gen::<f32>() < self.solarize_prob {
            image = self.solarize(image, self.solarize_threshold);
        }

        image.to_rgb8()
    }

    fn apply_val(&self, image: &DynamicImage) -> RgbImage {
        let (width, height) = image.dimensions();
        let target = self.resize_short.max(1);

        let (new_width, new_height) = if width < height {
            (
                target,
                (height as f32 * (target as f32 / width as f32)).round() as u32,
            )
        } else {
            (
                (width as f32 * (target as f32 / height as f32)).round() as u32,
                target,
            )
        };

        let resized = image.resize_exact(new_width, new_height, FilterType::CatmullRom);
        let crop_x = (new_width.saturating_sub(self.image_size)) / 2;
        let crop_y = (new_height.saturating_sub(self.image_size)) / 2;
        let cropped = resized.crop_imm(crop_x, crop_y, self.image_size, self.image_size);
        cropped.to_rgb8()
    }

    fn val_crop_params(&self, image: &DynamicImage) -> CropParams {
        let (width, height) = image.dimensions();
        let target = self.resize_short.max(1);
        let (new_width, new_height) = if width < height {
            (
                target,
                (height as f32 * (target as f32 / width as f32)).round() as u32,
            )
        } else {
            (
                (width as f32 * (target as f32 / height as f32)).round() as u32,
                target,
            )
        };
        let scale = target as f32 / width.min(height).max(1) as f32;
        let crop_x = (new_width.saturating_sub(self.image_size)) / 2;
        let crop_y = (new_height.saturating_sub(self.image_size)) / 2;
        let crop_w = (self.image_size as f32 / scale).round().max(1.0);
        let crop_h = (self.image_size as f32 / scale).round().max(1.0);
        let x = (crop_x as f32 / scale).round().max(0.0);
        let y = (crop_y as f32 / scale).round().max(0.0);
        let max_w = width.max(1) as f32;
        let max_h = height.max(1) as f32;
        CropParams {
            x: x.min(max_w - 1.0).max(0.0) as u32,
            y: y.min(max_h - 1.0).max(0.0) as u32,
            width: crop_w.min(max_w).max(1.0) as u32,
            height: crop_h.min(max_h).max(1.0) as u32,
        }
    }

    fn apply_crop(&self, image: &DynamicImage, crop: CropParams) -> DynamicImage {
        let cropped = image.crop_imm(crop.x, crop.y, crop.width, crop.height);
        cropped.resize_exact(self.image_size, self.image_size, FilterType::CatmullRom)
    }

    fn random_resized_crop_params(&self, image: &DynamicImage, rng: &mut impl Rng) -> CropParams {
        let (width, height) = image.dimensions();
        let area = (width * height) as f32;
        let log_min = self.min_aspect_ratio.ln();
        let log_max = self.max_aspect_ratio.ln();

        for _ in 0..10 {
            let scale = if self.min_scale >= self.max_scale {
                self.min_scale
            } else {
                rng.gen_range(self.min_scale..self.max_scale)
            };
            let aspect = if self.min_aspect_ratio >= self.max_aspect_ratio {
                self.min_aspect_ratio
            } else {
                rng.gen_range(log_min..log_max).exp()
            };
            let target = scale * area;
            let new_width = (target * aspect).sqrt().round() as u32;
            let new_height = (target / aspect).sqrt().round() as u32;

            if new_width > 0 && new_height > 0 && new_width <= width && new_height <= height {
                let x = rng.gen_range(0..=width - new_width);
                let y = rng.gen_range(0..=height - new_height);
                return CropParams {
                    x,
                    y,
                    width: new_width,
                    height: new_height,
                };
            }
        }

        let side = width.min(height).max(1);
        let x = (width - side) / 2;
        let y = (height - side) / 2;
        CropParams {
            x,
            y,
            width: side,
            height: side,
        }
    }

    fn adjust_saturation(&self, image: DynamicImage, factor: f32) -> DynamicImage {
        let mut rgb = image.to_rgb8();
        for pixel in rgb.pixels_mut() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;
            let gray = 0.299 * r + 0.587 * g + 0.114 * b;
            let r = (gray + (r - gray) * factor).round().clamp(0.0, 255.0) as u8;
            let g = (gray + (g - gray) * factor).round().clamp(0.0, 255.0) as u8;
            let b = (gray + (b - gray) * factor).round().clamp(0.0, 255.0) as u8;
            *pixel = image::Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(rgb)
    }

    fn solarize(&self, image: DynamicImage, threshold: u8) -> DynamicImage {
        let mut rgb = image.to_rgb8();
        for pixel in rgb.pixels_mut() {
            for channel in pixel.0.iter_mut() {
                if *channel > threshold {
                    *channel = 255 - *channel;
                }
            }
        }
        DynamicImage::ImageRgb8(rgb)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct VisionNormalize {
    mean: [f32; IMAGE_CHANNELS],
    std: [f32; IMAGE_CHANNELS],
}

impl VisionNormalize {
    pub fn new(mean: [f32; IMAGE_CHANNELS], std: [f32; IMAGE_CHANNELS]) -> Self {
        Self { mean, std }
    }

    pub fn apply(&self, image: &RgbImage, buffer: &mut Vec<f32>) {
        let (width, height) = image.dimensions();
        let pixels = (width * height) as usize;
        if pixels == 0 {
            return;
        }
        let start = buffer.len();
        buffer.resize(start + pixels * IMAGE_CHANNELS, 0.0);
        let raw = image.as_raw();
        let stride = pixels;
        for idx in 0..pixels {
            let base = idx * IMAGE_CHANNELS;
            let r = raw[base] as f32 / 255.0;
            let g = raw[base + 1] as f32 / 255.0;
            let b = raw[base + 2] as f32 / 255.0;
            buffer[start + idx] = (r - self.mean[0]) / self.std[0];
            buffer[start + stride + idx] = (g - self.mean[1]) / self.std[1];
            buffer[start + 2 * stride + idx] = (b - self.mean[2]) / self.std[2];
        }
    }
}

#[derive(Debug)]
pub struct DinoFeatureStore {
    cls_file: Mutex<File>,
    patch_file: Mutex<File>,
    feature_dim: usize,
    patch_tokens: usize,
    records: usize,
}

#[derive(Debug)]
struct TeacherBatchData {
    cls: Vec<f32>,
    patch: Vec<f32>,
}

impl DinoFeatureStore {
    pub fn new(
        cls_path: &Path,
        patch_path: &Path,
        feature_dim: usize,
        patch_tokens: usize,
        expected_records: Option<usize>,
    ) -> Result<Self> {
        if feature_dim == 0 || patch_tokens == 0 {
            return Err(anyhow!("feature dimensions must be non-zero"));
        }

        let cls_file = File::open(cls_path)
            .map_err(|err| anyhow!("failed to open {}: {err}", cls_path.display()))?;
        let patch_file = File::open(patch_path)
            .map_err(|err| anyhow!("failed to open {}: {err}", patch_path.display()))?;

        let cls_len = cls_file
            .metadata()
            .map_err(|err| anyhow!("failed to read {} metadata: {err}", cls_path.display()))?
            .len();
        let patch_len = patch_file
            .metadata()
            .map_err(|err| anyhow!("failed to read {} metadata: {err}", patch_path.display()))?
            .len();

        let cls_stride = feature_dim as u64 * BYTES_PER_F32;
        let patch_stride = feature_dim as u64 * patch_tokens as u64 * BYTES_PER_F32;
        if cls_len % cls_stride != 0 {
            return Err(anyhow!(
                "cls feature file size mismatch: {} bytes not divisible by {}",
                cls_len,
                cls_stride
            ));
        }
        if patch_len % patch_stride != 0 {
            return Err(anyhow!(
                "patch feature file size mismatch: {} bytes not divisible by {}",
                patch_len,
                patch_stride
            ));
        }

        let cls_records = (cls_len / cls_stride) as usize;
        let patch_records = (patch_len / patch_stride) as usize;
        if cls_records != patch_records {
            return Err(anyhow!(
                "teacher records mismatch: cls={}, patch={}",
                cls_records,
                patch_records
            ));
        }
        if let Some(expected) = expected_records.filter(|expected| *expected > cls_records) {
            return Err(anyhow!(
                "teacher records fewer than expected: expected={}, available={}",
                expected,
                cls_records
            ));
        }

        Ok(Self {
            cls_file: Mutex::new(cls_file),
            patch_file: Mutex::new(patch_file),
            feature_dim,
            patch_tokens,
            records: cls_records,
        })
    }

    pub fn records(&self) -> usize {
        self.records
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn patch_tokens(&self) -> usize {
        self.patch_tokens
    }

    fn load_batch_data(&self, indices: &[usize]) -> Result<TeacherBatchData> {
        let batch = indices.len();
        if batch == 0 {
            return Err(anyhow!("teacher feature batch is empty"));
        }
        let mut cls_data = Vec::with_capacity(batch * self.feature_dim);
        let mut patch_data = Vec::with_capacity(batch * self.feature_dim * self.patch_tokens);

        {
            let mut cls_file = self.cls_file.lock().unwrap();
            for &index in indices {
                let offset = index as u64 * self.feature_dim as u64 * BYTES_PER_F32;
                let values = read_f32_block(&mut cls_file, offset, self.feature_dim)?;
                cls_data.extend_from_slice(&values);
            }
        }

        {
            let mut patch_file = self.patch_file.lock().unwrap();
            let block_len = self.feature_dim * self.patch_tokens;
            for &index in indices {
                let offset = index as u64 * block_len as u64 * BYTES_PER_F32;
                let values = read_f32_block(&mut patch_file, offset, block_len)?;
                patch_data.extend_from_slice(&values);
            }
        }

        Ok(TeacherBatchData {
            cls: cls_data,
            patch: patch_data,
        })
    }

    pub fn load_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<(Tensor<B, 2>, Tensor<B, 3>)> {
        let batch = indices.len();
        let data = self.load_batch_data(indices)?;
        let cls_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(data.cls, [batch, self.feature_dim]), device);
        let patch_tensor = Tensor::<B, 3>::from_data(
            TensorData::new(data.patch, [batch, self.patch_tokens, self.feature_dim]),
            device,
        );
        Ok((cls_tensor, patch_tensor))
    }
}

#[derive(Clone, Debug)]
pub struct ImageNetDatasetConfig {
    pub root: PathBuf,
    pub split: ImageNetSplit,
    pub max_records: Option<usize>,
    pub augmentations: ImageNetAugmentations,
    pub local_augmentations: Option<ImageNetAugmentations>,
    pub normalize: VisionNormalize,
    pub teacher: Option<Arc<DinoFeatureStore>>,
    pub views: usize,
    pub local_views: usize,
    pub min_view_overlap: f32,
    pub view_overlap_attempts: usize,
    pub cache_decoded: bool,
    pub cache_capacity: usize,
    pub cache_preprocessed: bool,
}

#[derive(Clone, Debug)]
struct ImageNetSample {
    path: PathBuf,
    label: usize,
}

#[derive(Clone)]
struct ImageCache {
    capacity: usize,
    inner: Arc<Mutex<ImageCacheState>>,
}

struct ImageCacheState {
    entries: HashMap<PathBuf, CacheEntry>,
    order: VecDeque<(PathBuf, u64)>,
    tick: u64,
}

struct CacheEntry {
    image: Arc<DynamicImage>,
    tick: u64,
}

impl ImageCache {
    const ORDER_GC_MULTIPLIER: usize = 4;

    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Arc::new(Mutex::new(ImageCacheState {
                entries: HashMap::new(),
                order: VecDeque::new(),
                tick: 0,
            })),
        }
    }

    fn prune_order(&self, state: &mut ImageCacheState) {
        let max_len = self
            .capacity
            .saturating_mul(Self::ORDER_GC_MULTIPLIER)
            .max(1);
        if state.order.len() <= max_len {
            return;
        }
        let mut pruned = VecDeque::with_capacity(state.entries.len());
        for (path, tick) in state.order.drain(..) {
            if let Some(entry) = state.entries.get(&path)
                && entry.tick == tick
            {
                pruned.push_back((path, tick));
            }
        }
        state.order = pruned;
    }

    fn get(&self, path: &Path) -> Option<Arc<DynamicImage>> {
        let mut state = self.inner.lock().unwrap();
        let image = state
            .entries
            .get(path)
            .map(|entry| Arc::clone(&entry.image))?;
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        if let Some(entry) = state.entries.get_mut(path) {
            entry.tick = tick;
        }
        state.order.push_back((path.to_path_buf(), tick));
        self.prune_order(&mut state);
        Some(image)
    }

    fn insert(&self, path: PathBuf, image: Arc<DynamicImage>) {
        let mut state = self.inner.lock().unwrap();
        state.tick = state.tick.wrapping_add(1);
        let tick = state.tick;
        state
            .entries
            .insert(path.clone(), CacheEntry { image, tick });
        state.order.push_back((path, tick));
        while state.entries.len() > self.capacity {
            while let Some((evict_path, evict_tick)) = state.order.pop_front() {
                if let Some(entry) = state.entries.get(&evict_path)
                    && entry.tick == evict_tick
                {
                    state.entries.remove(&evict_path);
                    break;
                }
            }
        }
        self.prune_order(&mut state);
    }
}

#[derive(Clone)]
pub struct ImageNetDataset {
    samples: Vec<ImageNetSample>,
    num_classes: usize,
    augmentations: ImageNetAugmentations,
    local_augmentations: Option<ImageNetAugmentations>,
    normalize: VisionNormalize,
    teacher: Option<Arc<DinoFeatureStore>>,
    global_views: usize,
    local_views: usize,
    min_view_overlap: f32,
    view_overlap_attempts: usize,
    cache: Option<ImageCache>,
    preprocessed_cache: Option<Vec<Arc<Vec<f32>>>>,
}

impl ImageNetDataset {
    pub fn new(config: ImageNetDatasetConfig) -> Result<Self> {
        let (mut samples, num_classes) = collect_samples(&config.root)?;
        if let Some(limit) = config
            .max_records
            .filter(|limit| *limit > 0 && *limit < samples.len())
        {
            samples.truncate(limit);
        }
        let global_views = config.views.max(1);
        let local_views = config.local_views;
        let min_view_overlap = config.min_view_overlap.max(0.0);
        let view_overlap_attempts = config.view_overlap_attempts.max(1);
        if local_views > 0 && config.local_augmentations.is_none() {
            return Err(anyhow!("local_augmentations required when local_views > 0"));
        }
        let cache = if config.cache_decoded && config.cache_capacity > 0 {
            Some(ImageCache::new(config.cache_capacity))
        } else {
            None
        };

        let allow_preprocessed = config.cache_preprocessed
            && global_views == 1
            && local_views == 0
            && config.augmentations.is_deterministic()
            && config.cache_capacity >= samples.len();

        let mut dataset = Self {
            samples,
            num_classes,
            augmentations: config.augmentations,
            local_augmentations: config.local_augmentations,
            normalize: config.normalize,
            teacher: config.teacher,
            global_views,
            local_views,
            min_view_overlap,
            view_overlap_attempts,
            cache,
            preprocessed_cache: None,
        };

        if allow_preprocessed {
            dataset.preprocessed_cache = Some(dataset.build_preprocessed_cache()?);
        }

        Ok(dataset)
    }

    pub fn with_teacher(mut self, teacher: Arc<DinoFeatureStore>) -> Self {
        self.teacher = Some(teacher);
        self
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn steps_per_epoch(&self, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 1;
        }
        self.len().div_ceil(batch_size).max(1)
    }

    fn load_image_cached(&self, path: &Path) -> Result<Arc<DynamicImage>> {
        if let Some(cache) = &self.cache
            && let Some(image) = cache.get(path)
        {
            return Ok(image);
        }

        let image = Arc::new(load_image(path)?);
        if let Some(cache) = &self.cache {
            cache.insert(path.to_path_buf(), Arc::clone(&image));
        }
        Ok(image)
    }

    fn build_preprocessed_cache(&self) -> Result<Vec<Arc<Vec<f32>>>> {
        let mut cache = Vec::with_capacity(self.samples.len());
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let image_size = self.augmentations.image_size();
        let buffer_len = image_size * image_size * IMAGE_CHANNELS;
        for sample in &self.samples {
            let image = self.load_image_cached(&sample.path)?;
            let aug = self.augmentations.apply(image.as_ref(), &mut rng);
            let mut buffer = Vec::with_capacity(buffer_len);
            self.normalize.apply(&aug, &mut buffer);
            cache.push(Arc::new(buffer));
        }
        Ok(cache)
    }

    pub fn sample_batch<B: Backend>(
        &self,
        batch_size: usize,
        device: &B::Device,
    ) -> ImageNetBatch<B> {
        self.sample_batch_data(batch_size)
            .unwrap_or_else(|err| panic!("imagenet batch failed: {err}"))
            .into_batch(device)
    }

    fn sample_batch_data(&self, batch_size: usize) -> Result<ImageNetBatchData> {
        if batch_size == 0 {
            return Err(anyhow!("imagenet batch size must be > 0"));
        }
        let len = self.len();
        if len == 0 {
            return Err(anyhow!("imagenet dataset is empty"));
        }
        let mut rng = thread_rng();
        let mut indices = Vec::with_capacity(batch_size);
        for _ in 0..batch_size {
            indices.push(rng.gen_range(0..len));
        }

        let global_image_size = self.augmentations.image_size();
        let mut images =
            Vec::with_capacity(batch_size * global_image_size * global_image_size * IMAGE_CHANNELS);
        let mut target_images = if self.global_views > 1 {
            Some(Vec::with_capacity(
                batch_size * global_image_size * global_image_size * IMAGE_CHANNELS,
            ))
        } else {
            None
        };
        let mut view_images = if self.local_views == 0 && self.global_views > 1 {
            Some(Vec::with_capacity(
                batch_size
                    * self.global_views
                    * global_image_size
                    * global_image_size
                    * IMAGE_CHANNELS,
            ))
        } else {
            None
        };
        let mut view_crops = if self.local_views == 0 && self.global_views > 1 {
            Some(Vec::with_capacity(batch_size * self.global_views * 4))
        } else {
            None
        };
        let mut global_view_images = if self.local_views > 0 {
            Some(Vec::with_capacity(
                batch_size
                    * self.global_views
                    * global_image_size
                    * global_image_size
                    * IMAGE_CHANNELS,
            ))
        } else {
            None
        };
        let local_image_size = self
            .local_augmentations
            .as_ref()
            .map(|aug| aug.image_size())
            .unwrap_or(0);
        let mut local_view_images = if self.local_views > 0 {
            Some(Vec::with_capacity(
                batch_size
                    * self.local_views
                    * local_image_size
                    * local_image_size
                    * IMAGE_CHANNELS,
            ))
        } else {
            None
        };
        let mut labels = Vec::with_capacity(batch_size);

        for &index in &indices {
            let sample = &self.samples[index];
            if self.global_views == 1
                && self.local_views == 0
                && let Some(preprocessed_cache) = &self.preprocessed_cache
                && let Some(cached) = preprocessed_cache.get(index)
            {
                images.extend_from_slice(cached.as_ref());
                labels.push(sample.label as i64);
                continue;
            }
            let image = self.load_image_cached(&sample.path)?;
            let (img_w, img_h) = image.dimensions();
            if self.global_views == 1 && self.local_views == 0 {
                let primary = self.augmentations.apply(image.as_ref(), &mut rng);
                self.normalize.apply(&primary, &mut images);
            } else {
                if self.min_view_overlap > 0.0
                    && matches!(self.augmentations.split, ImageNetSplit::Train)
                {
                    let primary_crop = self
                        .augmentations
                        .random_resized_crop_params(image.as_ref(), &mut rng);
                    let mut crops = Vec::with_capacity(self.global_views);
                    crops.push(primary_crop);
                    for _ in 1..self.global_views {
                        let mut selected = None;
                        for _ in 0..self.view_overlap_attempts {
                            let candidate = self
                                .augmentations
                                .random_resized_crop_params(image.as_ref(), &mut rng);
                            if crop_overlap_ratio(primary_crop, candidate) >= self.min_view_overlap
                            {
                                selected = Some(candidate);
                                break;
                            }
                        }
                        crops.push(selected.unwrap_or(primary_crop));
                    }

                    for (view_idx, crop) in crops.into_iter().enumerate() {
                        if let Some(buffer) = view_crops.as_mut() {
                            push_crop_normalized(buffer, crop, img_w, img_h);
                        }
                        let aug = self.augmentations.apply_train_with_crop(
                            image.as_ref(),
                            &mut rng,
                            crop,
                        );
                        if view_idx == 0 {
                            self.normalize.apply(&aug, &mut images);
                        }
                        if view_idx == 1
                            && let Some(buffer) = target_images.as_mut()
                        {
                            self.normalize.apply(&aug, buffer);
                        }
                        if let Some(buffer) = view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                        if let Some(buffer) = global_view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                    }
                } else {
                    for view_idx in 0..self.global_views {
                        let (aug, crop) =
                            if matches!(self.augmentations.split, ImageNetSplit::Train)
                                && !self.augmentations.is_deterministic()
                            {
                                let crop = self
                                    .augmentations
                                    .random_resized_crop_params(image.as_ref(), &mut rng);
                                let aug = self.augmentations.apply_train_with_crop(
                                    image.as_ref(),
                                    &mut rng,
                                    crop,
                                );
                                (aug, crop)
                            } else {
                                let crop = self.augmentations.val_crop_params(image.as_ref());
                                let aug = self.augmentations.apply_val(image.as_ref());
                                (aug, crop)
                            };
                        if view_idx == 0 {
                            self.normalize.apply(&aug, &mut images);
                        }
                        if view_idx == 1
                            && let Some(buffer) = target_images.as_mut()
                        {
                            self.normalize.apply(&aug, buffer);
                        }
                        if let Some(buffer) = view_crops.as_mut() {
                            push_crop_normalized(buffer, crop, img_w, img_h);
                        }
                        if let Some(buffer) = view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                        if let Some(buffer) = global_view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                    }
                }

                if self.local_views > 0 {
                    let local_aug = self
                        .local_augmentations
                        .as_ref()
                        .expect("local augmentations required");
                    for _ in 0..self.local_views {
                        let aug = local_aug.apply(image.as_ref(), &mut rng);
                        if let Some(buffer) = local_view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                    }
                }
            }
            labels.push(sample.label as i64);
        }

        let (teacher_cls, teacher_patch, teacher_dim, teacher_tokens) = match &self.teacher {
            Some(store) => {
                let batch = store.load_batch_data(&indices)?;
                (
                    Some(batch.cls),
                    Some(batch.patch),
                    Some(store.feature_dim()),
                    Some(store.patch_tokens()),
                )
            }
            None => (None, None, None, None),
        };

        Ok(ImageNetBatchData {
            images,
            target_images,
            view_images,
            view_crops,
            global_view_images,
            local_view_images,
            labels,
            teacher_patch,
            teacher_cls,
            batch_size,
            global_image_size,
            local_image_size,
            global_views: self.global_views.max(1),
            local_views: self.local_views,
            teacher_feature_dim: teacher_dim,
            teacher_patch_tokens: teacher_tokens,
        })
    }
}

struct ImageNetBatchData {
    images: Vec<f32>,
    target_images: Option<Vec<f32>>,
    view_images: Option<Vec<f32>>,
    view_crops: Option<Vec<f32>>,
    global_view_images: Option<Vec<f32>>,
    local_view_images: Option<Vec<f32>>,
    labels: Vec<i64>,
    teacher_patch: Option<Vec<f32>>,
    teacher_cls: Option<Vec<f32>>,
    batch_size: usize,
    global_image_size: usize,
    local_image_size: usize,
    global_views: usize,
    local_views: usize,
    teacher_feature_dim: Option<usize>,
    teacher_patch_tokens: Option<usize>,
}

impl ImageNetBatchData {
    fn into_batch<B: Backend>(self, device: &B::Device) -> ImageNetBatch<B> {
        let images_tensor = Tensor::<B, 4>::from_data(
            TensorData::new(
                self.images,
                [
                    self.batch_size,
                    IMAGE_CHANNELS,
                    self.global_image_size,
                    self.global_image_size,
                ],
            ),
            device,
        );
        let labels_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(self.labels, [self.batch_size]), device);
        let target_images_tensor = self.target_images.map(|buffer| {
            Tensor::<B, 4>::from_data(
                TensorData::new(
                    buffer,
                    [
                        self.batch_size,
                        IMAGE_CHANNELS,
                        self.global_image_size,
                        self.global_image_size,
                    ],
                ),
                device,
            )
        });
        let view_images_tensor = self.view_images.map(|buffer| {
            Tensor::<B, 5>::from_data(
                TensorData::new(
                    buffer,
                    [
                        self.batch_size,
                        self.global_views,
                        IMAGE_CHANNELS,
                        self.global_image_size,
                        self.global_image_size,
                    ],
                ),
                device,
            )
        });
        let view_crops_tensor = self.view_crops.map(|buffer| {
            Tensor::<B, 3>::from_data(
                TensorData::new(buffer, [self.batch_size, self.global_views, 4]),
                device,
            )
        });
        let global_view_images_tensor = self.global_view_images.map(|buffer| {
            Tensor::<B, 5>::from_data(
                TensorData::new(
                    buffer,
                    [
                        self.batch_size,
                        self.global_views,
                        IMAGE_CHANNELS,
                        self.global_image_size,
                        self.global_image_size,
                    ],
                ),
                device,
            )
        });
        let local_view_images_tensor = self.local_view_images.map(|buffer| {
            Tensor::<B, 5>::from_data(
                TensorData::new(
                    buffer,
                    [
                        self.batch_size,
                        self.local_views,
                        IMAGE_CHANNELS,
                        self.local_image_size,
                        self.local_image_size,
                    ],
                ),
                device,
            )
        });

        let teacher_cls = self.teacher_cls.map(|data| {
            let dim = self
                .teacher_feature_dim
                .expect("teacher feature dim required");
            Tensor::<B, 2>::from_data(TensorData::new(data, [self.batch_size, dim]), device)
        });
        let teacher_patch = self.teacher_patch.map(|data| {
            let dim = self
                .teacher_feature_dim
                .expect("teacher feature dim required");
            let tokens = self
                .teacher_patch_tokens
                .expect("teacher patch tokens required");
            Tensor::<B, 3>::from_data(
                TensorData::new(data, [self.batch_size, tokens, dim]),
                device,
            )
        });

        ImageNetBatch::new(
            images_tensor,
            target_images_tensor,
            view_images_tensor,
            view_crops_tensor,
            global_view_images_tensor,
            local_view_images_tensor,
            labels_tensor,
            teacher_patch,
            teacher_cls,
        )
    }
}

#[derive(Clone)]
pub struct ImageNetBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub target_images: Option<Tensor<B, 4>>,
    pub view_images: Option<Tensor<B, 5>>,
    pub view_crops: Option<Tensor<B, 3>>,
    pub global_view_images: Option<Tensor<B, 5>>,
    pub local_view_images: Option<Tensor<B, 5>>,
    pub labels: Tensor<B, 1, Int>,
    pub teacher_patch: Option<Tensor<B, 3>>,
    pub teacher_cls: Option<Tensor<B, 2>>,
}

impl<B: Backend> ImageNetBatch<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        images: Tensor<B, 4>,
        target_images: Option<Tensor<B, 4>>,
        view_images: Option<Tensor<B, 5>>,
        view_crops: Option<Tensor<B, 3>>,
        global_view_images: Option<Tensor<B, 5>>,
        local_view_images: Option<Tensor<B, 5>>,
        labels: Tensor<B, 1, Int>,
        teacher_patch: Option<Tensor<B, 3>>,
        teacher_cls: Option<Tensor<B, 2>>,
    ) -> Self {
        Self {
            images,
            target_images,
            view_images,
            view_crops,
            global_view_images,
            local_view_images,
            labels,
            teacher_patch,
            teacher_cls,
        }
    }

    pub fn repeat_batch(&self, repeats: usize) -> Self {
        let repeats = repeats.max(1);
        if repeats == 1 {
            return self.clone();
        }
        Self {
            images: self.images.clone().repeat_dim(0, repeats),
            target_images: self
                .target_images
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            view_images: self
                .view_images
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            view_crops: self
                .view_crops
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            global_view_images: self
                .global_view_images
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            local_view_images: self
                .local_view_images
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            labels: self.labels.clone().repeat_dim(0, repeats),
            teacher_patch: self
                .teacher_patch
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            teacher_cls: self
                .teacher_cls
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
        }
    }
}

pub struct ImageNetDataLoader<B: Backend> {
    dataset: Arc<ImageNetDataset>,
    batch_size: usize,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    device: B::Device,
    prefetch_batches: usize,
    prefetch_workers: usize,
    prefetch_to_device: bool,
}

impl<B: Backend> Clone for ImageNetDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: self.device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
        }
    }
}

impl<B: Backend> ImageNetDataLoader<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        dataset: Arc<ImageNetDataset>,
        batch_size: usize,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        prefetch_batches: usize,
        prefetch_workers: usize,
        prefetch_to_device: bool,
    ) -> Self {
        let steps_per_epoch = if steps_per_epoch == 0 {
            dataset.steps_per_epoch(batch_size)
        } else {
            steps_per_epoch
        };
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let prefetch_batches = prefetch_batches.min(steps_per_epoch);
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
            batch_size,
            steps_per_epoch,
            total_steps,
            consumed_steps,
            device: device.clone(),
            prefetch_batches,
            prefetch_workers,
            prefetch_to_device,
        }
    }
}

impl<B> DataLoader<B, ImageNetBatch<B>> for ImageNetDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone + Send + Sync + 'static,
    ImageNetBatch<B>: Send,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<ImageNetBatch<B>> + 'a> {
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

        let prefetcher = if self.prefetch_batches > 0 && steps_total > 0 {
            Some(ImageNetPrefetcher::new(
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

        Box::new(ImageNetIterator {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            prefetcher,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.batch_size
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, ImageNetBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, ImageNetBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: steps,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: self.device.clone(),
            prefetch_batches: self.prefetch_batches,
            prefetch_workers: self.prefetch_workers,
            prefetch_to_device: self.prefetch_to_device,
        })
    }
}

enum ImageNetPrefetchItem<B: Backend> {
    Data(Box<ImageNetBatchData>),
    Batch(ImageNetBatch<B>),
}

struct ImageNetPrefetcher<B: Backend> {
    rx: Option<mpsc::Receiver<Result<ImageNetPrefetchItem<B>>>>,
    stop: Arc<AtomicBool>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl<B> ImageNetPrefetcher<B>
where
    B: Backend + 'static,
    B::Device: Clone + Send + Sync + 'static,
    ImageNetBatch<B>: Send,
{
    fn new(
        dataset: Arc<ImageNetDataset>,
        batch_size: usize,
        steps_total: usize,
        prefetch_batches: usize,
        prefetch_workers: usize,
        device: &B::Device,
        prefetch_to_device: bool,
    ) -> Self {
        let workers = prefetch_workers.max(1);
        let (tx, rx) = mpsc::sync_channel(prefetch_batches.max(1));
        let remaining = Arc::new(AtomicUsize::new(steps_total));
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(workers + if prefetch_to_device { 1 } else { 0 });

        if prefetch_to_device {
            let (data_tx, data_rx) = mpsc::sync_channel(prefetch_batches.max(1));
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
                        let result = dataset.sample_batch_data(batch_size);
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
                    let item = match data {
                        Ok(data) => {
                            let _guard = crate::device::device_allocation_lock().lock().ok();
                            Ok(ImageNetPrefetchItem::Batch(data.into_batch::<B>(&device)))
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
                        let result = dataset
                            .sample_batch_data(batch_size)
                            .map(|data| ImageNetPrefetchItem::Data(Box::new(data)));
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

    fn recv(&mut self) -> Option<Result<ImageNetPrefetchItem<B>>> {
        let rx = self.rx.as_ref()?;
        rx.recv().ok()
    }
}

impl<B: Backend> Drop for ImageNetPrefetcher<B> {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.rx.take();
        for handle in self.handles.drain(..) {
            let _ = handle.join();
        }
    }
}

struct ImageNetIterator<B: Backend> {
    dataset: Arc<ImageNetDataset>,
    batch_size: usize,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    prefetcher: Option<ImageNetPrefetcher<B>>,
}

impl<B: Backend> Iterator for ImageNetIterator<B> {
    type Item = ImageNetBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }
        let batch = if let Some(prefetcher) = &mut self.prefetcher {
            match prefetcher.recv() {
                Some(Ok(ImageNetPrefetchItem::Batch(batch))) => batch,
                Some(Ok(ImageNetPrefetchItem::Data(data))) => (*data).into_batch(&self.device),
                Some(Err(err)) => panic!("imagenet prefetch error: {err}"),
                None => panic!("imagenet prefetch channel closed early"),
            }
        } else {
            self.dataset.sample_batch(self.batch_size, &self.device)
        };

        Some(batch)
    }
}

impl<B: Backend> DataLoaderIterator<ImageNetBatch<B>> for ImageNetIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.batch_size,
            self.steps_total * self.batch_size,
        )
    }
}

fn collect_samples(root: &Path) -> Result<(Vec<ImageNetSample>, usize)> {
    let mut class_dirs: Vec<PathBuf> = fs::read_dir(root)
        .map_err(|err| anyhow!("failed to read {}: {err}", root.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();

    class_dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

    if class_dirs.is_empty() {
        return Err(anyhow!("no class directories found in {}", root.display()));
    }

    let mut samples = Vec::new();
    for (label, class_dir) in class_dirs.iter().enumerate() {
        let mut images = collect_images(class_dir)?;
        images.sort();
        for image in images {
            samples.push(ImageNetSample { path: image, label });
        }
    }

    Ok((samples, class_dirs.len()))
}

fn collect_images(root: &Path) -> Result<Vec<PathBuf>> {
    let mut images = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in
            fs::read_dir(&dir).map_err(|err| anyhow!("failed to read {}: {err}", dir.display()))?
        {
            let entry = entry.map_err(|err| anyhow!("failed to read dir entry: {err}"))?;
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if is_image_file(&path) {
                images.push(path);
            }
        }
    }
    Ok(images)
}

fn is_image_file(path: &Path) -> bool {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"),
        None => false,
    }
}

fn load_image(path: &Path) -> Result<DynamicImage> {
    image::ImageReader::open(path)
        .map_err(|err| anyhow!("failed to open image {}: {err}", path.display()))?
        .decode()
        .map_err(|err| anyhow!("failed to decode {}: {err}", path.display()))
}

fn read_f32_block(file: &mut File, offset: u64, len: usize) -> Result<Vec<f32>> {
    let mut buf = vec![0u8; len * BYTES_PER_F32 as usize];
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| anyhow!("failed to seek teacher features: {err}"))?;
    file.read_exact(&mut buf)
        .map_err(|err| anyhow!("failed to read teacher features: {err}"))?;

    let mut out = Vec::with_capacity(len);
    for chunk in buf.chunks_exact(4) {
        let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        out.push(value);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_cache_prunes_order_growth() {
        let cache = ImageCache::new(2);
        let image = Arc::new(DynamicImage::ImageRgb8(RgbImage::new(1, 1)));
        let path_a = PathBuf::from("a.png");
        let path_b = PathBuf::from("b.png");
        cache.insert(path_a.clone(), Arc::clone(&image));
        cache.insert(path_b.clone(), Arc::clone(&image));

        for _ in 0..128 {
            let _ = cache.get(&path_a);
        }

        let order_len = cache.inner.lock().unwrap().order.len();
        let max_len = cache.capacity * ImageCache::ORDER_GC_MULTIPLIER;
        assert!(
            order_len <= max_len,
            "order len {order_len} exceeds {max_len}"
        );

        let path_c = PathBuf::from("c.png");
        cache.insert(path_c, image);
        let entries = cache.inner.lock().unwrap().entries.len();
        assert!(entries <= cache.capacity);
    }
}
