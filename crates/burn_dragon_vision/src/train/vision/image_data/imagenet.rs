use anyhow::{Result, anyhow};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use half::f16;
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView, RgbImage};
use rand::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, VecDeque};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, mpsc};
use std::thread;
use std::time::Instant;

use crate::config::VisionTeacherTargetKind;

const IMAGE_CHANNELS: usize = 3;
const BYTES_PER_F32: u64 = 4;
const BYTES_PER_F16: u64 = 2;

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

    pub fn mean(&self) -> [f32; IMAGE_CHANNELS] {
        self.mean
    }

    pub fn std(&self) -> [f32; IMAGE_CHANNELS] {
        self.std
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
    cls_backing: DinoFeatureBacking,
    patch_backing: Option<DinoFeatureBacking>,
    feature_dim: usize,
    patch_tokens: Option<usize>,
    records: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DinoFeatureDType {
    F32,
    F16,
}

impl DinoFeatureDType {
    fn bytes_per_scalar(self) -> u64 {
        match self {
            Self::F32 => BYTES_PER_F32,
            Self::F16 => BYTES_PER_F16,
        }
    }
}

#[derive(Debug)]
enum DinoFeatureBacking {
    File {
        file: Mutex<File>,
        dtype: DinoFeatureDType,
    },
    MemoryF32(Arc<Vec<f32>>),
    MemoryF16(Arc<Vec<f16>>),
}

#[derive(Debug)]
struct TeacherBatchData {
    cls: Vec<f32>,
    patch: Option<Vec<f32>>,
}

#[derive(Debug)]
struct ImageTensorBatchData {
    tensor: Vec<f32>,
}

impl DinoFeatureStore {
    pub fn new(
        cls_path: &Path,
        patch_path: &Path,
        feature_dim: usize,
        patch_tokens: usize,
        expected_records: Option<usize>,
    ) -> Result<Self> {
        Self::new_with_options(
            cls_path,
            patch_path,
            feature_dim,
            patch_tokens,
            expected_records,
            false,
        )
    }

    pub fn new_with_options(
        cls_path: &Path,
        patch_path: &Path,
        feature_dim: usize,
        patch_tokens: usize,
        expected_records: Option<usize>,
        cache_in_memory: bool,
    ) -> Result<Self> {
        Self::new_optional_patch_with_options(
            cls_path,
            Some(patch_path),
            feature_dim,
            Some(patch_tokens),
            expected_records,
            cache_in_memory,
        )
    }

    pub fn new_optional_patch_with_options(
        cls_path: &Path,
        patch_path: Option<&Path>,
        feature_dim: usize,
        patch_tokens: Option<usize>,
        expected_records: Option<usize>,
        cache_in_memory: bool,
    ) -> Result<Self> {
        if feature_dim == 0 {
            return Err(anyhow!("feature dimensions must be non-zero"));
        }
        if matches!(patch_tokens, Some(0)) {
            return Err(anyhow!("patch token count must be non-zero when provided"));
        }
        if patch_path.is_some() != patch_tokens.is_some() {
            return Err(anyhow!(
                "patch_path and patch_tokens must either both be set or both be omitted"
            ));
        }
        let storage_dtype = infer_feature_storage_dtype(cls_path)?;

        let cls_file = File::open(cls_path)
            .map_err(|err| anyhow!("failed to open {}: {err}", cls_path.display()))?;
        let cls_len = cls_file
            .metadata()
            .map_err(|err| anyhow!("failed to read {} metadata: {err}", cls_path.display()))?
            .len();
        let cls_stride = feature_dim as u64 * storage_dtype.bytes_per_scalar();
        if cls_len % cls_stride != 0 {
            return Err(anyhow!(
                "cls feature file size mismatch: {} bytes not divisible by {}",
                cls_len,
                cls_stride
            ));
        }
        let cls_records = (cls_len / cls_stride) as usize;

        let (patch_backing, patch_records) =
            if let (Some(path), Some(tokens)) = (patch_path, patch_tokens) {
                let patch_file = File::open(path)
                    .map_err(|err| anyhow!("failed to open {}: {err}", path.display()))?;
                let patch_len = patch_file
                    .metadata()
                    .map_err(|err| anyhow!("failed to read {} metadata: {err}", path.display()))?
                    .len();
                let patch_stride =
                    feature_dim as u64 * tokens as u64 * storage_dtype.bytes_per_scalar();
                if patch_len % patch_stride != 0 {
                    return Err(anyhow!(
                        "patch feature file size mismatch: {} bytes not divisible by {}",
                        patch_len,
                        patch_stride
                    ));
                }
                let patch_records = (patch_len / patch_stride) as usize;
                let patch_backing = if cache_in_memory {
                    let patch_scalars = patch_records
                        .saturating_mul(feature_dim)
                        .saturating_mul(tokens);
                    Some(match storage_dtype {
                        DinoFeatureDType::F32 => DinoFeatureBacking::MemoryF32(
                            load_f32_file_into_memory(path, patch_scalars)?,
                        ),
                        DinoFeatureDType::F16 => DinoFeatureBacking::MemoryF16(
                            load_f16_file_into_memory(path, patch_scalars)?,
                        ),
                    })
                } else {
                    Some(DinoFeatureBacking::File {
                        file: Mutex::new(patch_file),
                        dtype: storage_dtype,
                    })
                };
                (patch_backing, Some(patch_records))
            } else {
                (None, None)
            };

        if let Some(patch_records) = patch_records
            && cls_records != patch_records
        {
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

        let cls_backing = if cache_in_memory {
            match storage_dtype {
                DinoFeatureDType::F32 => DinoFeatureBacking::MemoryF32(load_f32_file_into_memory(
                    cls_path,
                    cls_records.saturating_mul(feature_dim),
                )?),
                DinoFeatureDType::F16 => DinoFeatureBacking::MemoryF16(load_f16_file_into_memory(
                    cls_path,
                    cls_records.saturating_mul(feature_dim),
                )?),
            }
        } else {
            DinoFeatureBacking::File {
                file: Mutex::new(cls_file),
                dtype: storage_dtype,
            }
        };

        Ok(Self {
            cls_backing,
            patch_backing,
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

    pub fn patch_tokens(&self) -> Option<usize> {
        self.patch_tokens
    }

    fn load_batch_data(&self, indices: &[usize]) -> Result<TeacherBatchData> {
        let batch = indices.len();
        if batch == 0 {
            return Err(anyhow!("teacher feature batch is empty"));
        }
        let mut ordered = indices.iter().copied().enumerate().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(_, index)| *index);

        let cls_stride = self.feature_dim;
        let mut cls_data = vec![0.0; batch * cls_stride];

        load_backing_batch(&self.cls_backing, &ordered, cls_stride, &mut cls_data)?;
        let patch = if let (Some(backing), Some(tokens)) = (&self.patch_backing, self.patch_tokens)
        {
            let patch_stride = self.feature_dim * tokens;
            let mut patch_data = vec![0.0; batch * patch_stride];
            load_backing_batch(backing, &ordered, patch_stride, &mut patch_data)?;
            Some(patch_data)
        } else {
            None
        };

        Ok(TeacherBatchData {
            cls: cls_data,
            patch,
        })
    }

    pub fn load_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<(Tensor<B, 2>, Option<Tensor<B, 3>>)> {
        let batch = indices.len();
        let data = self.load_batch_data(indices)?;
        let cls_tensor =
            Tensor::<B, 2>::from_data(TensorData::new(data.cls, [batch, self.feature_dim]), device);
        let patch_tensor = data.patch.map(|patch| {
            Tensor::<B, 3>::from_data(
                TensorData::new(
                    patch,
                    [
                        batch,
                        self.patch_tokens
                            .expect("patch tokens required for patch tensor"),
                        self.feature_dim,
                    ],
                ),
                device,
            )
        });
        Ok((cls_tensor, patch_tensor))
    }
}

#[derive(Debug)]
pub struct ImageTensorStore {
    backing: DinoFeatureBacking,
    channels: usize,
    height: usize,
    width: usize,
    records: usize,
}

impl ImageTensorStore {
    pub fn new_with_options(
        path: &Path,
        channels: usize,
        height: usize,
        width: usize,
        expected_records: Option<usize>,
        cache_in_memory: bool,
    ) -> Result<Self> {
        if channels == 0 || height == 0 || width == 0 {
            return Err(anyhow!(
                "image tensor store dimensions must be non-zero (got {channels}x{height}x{width})"
            ));
        }
        let storage_dtype = infer_feature_storage_dtype(path)?;
        let file =
            File::open(path).map_err(|err| anyhow!("failed to open {}: {err}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|err| anyhow!("failed to read {} metadata: {err}", path.display()))?
            .len();
        let stride =
            channels as u64 * height as u64 * width as u64 * storage_dtype.bytes_per_scalar();
        if len % stride != 0 {
            return Err(anyhow!(
                "image tensor file size mismatch: {} bytes not divisible by {}",
                len,
                stride
            ));
        }
        let records = (len / stride) as usize;
        if let Some(expected) = expected_records.filter(|expected| *expected > records) {
            return Err(anyhow!(
                "image tensor records fewer than expected: expected={}, available={}",
                expected,
                records
            ));
        }
        let scalars = records
            .saturating_mul(channels)
            .saturating_mul(height)
            .saturating_mul(width);
        let backing = if cache_in_memory {
            match storage_dtype {
                DinoFeatureDType::F32 => {
                    DinoFeatureBacking::MemoryF32(load_f32_file_into_memory(path, scalars)?)
                }
                DinoFeatureDType::F16 => {
                    DinoFeatureBacking::MemoryF16(load_f16_file_into_memory(path, scalars)?)
                }
            }
        } else {
            DinoFeatureBacking::File {
                file: Mutex::new(file),
                dtype: storage_dtype,
            }
        };
        Ok(Self {
            backing,
            channels,
            height,
            width,
            records,
        })
    }

    pub fn channels(&self) -> usize {
        self.channels
    }

    pub fn height(&self) -> usize {
        self.height
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn records(&self) -> usize {
        self.records
    }

    fn load_batch_data(&self, indices: &[usize]) -> Result<ImageTensorBatchData> {
        let batch = indices.len();
        if batch == 0 {
            return Err(anyhow!("image tensor batch is empty"));
        }
        let mut ordered = indices.iter().copied().enumerate().collect::<Vec<_>>();
        ordered.sort_unstable_by_key(|(_, index)| *index);
        let stride = self
            .channels
            .saturating_mul(self.height)
            .saturating_mul(self.width);
        let mut tensor = vec![0.0; batch.saturating_mul(stride)];
        load_backing_batch(&self.backing, &ordered, stride, &mut tensor)?;
        Ok(ImageTensorBatchData { tensor })
    }

    pub fn load_batch<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Result<Tensor<B, 4>> {
        let batch = indices.len();
        let data = self.load_batch_data(indices)?;
        Ok(Tensor::<B, 4>::from_data(
            TensorData::new(data.tensor, [batch, self.channels, self.height, self.width]),
            device,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct ImageTeacherTargetStore {
    pub name: String,
    pub weight: f32,
    pub target_kind: VisionTeacherTargetKind,
    pub store: Arc<DinoFeatureStore>,
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
    pub rac_teacher_latent: Option<Arc<ImageTensorStore>>,
    pub teacher_targets: Vec<ImageTeacherTargetStore>,
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
    teacher_index: usize,
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
    rac_teacher_latent: Option<Arc<ImageTensorStore>>,
    teacher_targets: Vec<ImageTeacherTargetStore>,
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
            samples = limit_samples_balanced(&samples, limit);
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
            rac_teacher_latent: config.rac_teacher_latent,
            teacher_targets: config.teacher_targets,
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

    pub fn with_rac_teacher_latent(mut self, teacher: Arc<ImageTensorStore>) -> Self {
        self.rac_teacher_latent = Some(teacher);
        self
    }

    pub fn with_teacher_targets(mut self, teacher_targets: Vec<ImageTeacherTargetStore>) -> Self {
        self.teacher_targets = teacher_targets;
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
        let prof_enabled = crate::train::profile::enabled();
        let cpu_start = prof_enabled.then(Instant::now);
        let (data, profile) = self
            .sample_batch_data_profiled(batch_size)
            .unwrap_or_else(|err| panic!("imagenet batch failed: {err}"));
        let cpu_ns = cpu_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        let host_to_device_copy_bytes = data.host_to_device_copy_bytes();
        let tensor_copy_start = prof_enabled.then(Instant::now);
        let batch = data.into_batch(device);
        let tensor_copy_ns = tensor_copy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_dataloader(
                cpu_ns,
                profile.image_load_ns,
                profile.image_transform_ns,
                profile.teacher_load_ns,
                tensor_copy_ns,
                host_to_device_copy_bytes,
                0,
            );
        }

        batch
    }

    fn sample_batch_data_profiled(
        &self,
        batch_size: usize,
    ) -> Result<(ImageNetBatchData, ImageNetBatchCpuProfile)> {
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
        let profile_enabled = crate::train::profile::enabled();
        let mut profile = ImageNetBatchCpuProfile::default();

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
            let image_load_start = profile_enabled.then(Instant::now);
            let image = self.load_image_cached(&sample.path)?;
            profile.image_load_ns = profile.image_load_ns.saturating_add(
                image_load_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default(),
            );
            let image_transform_start = profile_enabled.then(Instant::now);
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
            profile.image_transform_ns = profile.image_transform_ns.saturating_add(
                image_transform_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default(),
            );
        }

        let teacher_indices = indices
            .iter()
            .map(|index| self.samples[*index].teacher_index)
            .collect::<Vec<_>>();

        let (teacher_cls, teacher_patch, teacher_dim, teacher_tokens) = match &self.teacher {
            Some(store) => {
                let teacher_load_start = profile_enabled.then(Instant::now);
                let batch = store.load_batch_data(&teacher_indices)?;
                profile.teacher_load_ns = profile.teacher_load_ns.saturating_add(
                    teacher_load_start
                        .map(|start| start.elapsed().as_nanos())
                        .unwrap_or_default(),
                );
                (
                    Some(batch.cls),
                    batch.patch,
                    Some(store.feature_dim()),
                    store.patch_tokens(),
                )
            }
            None => (None, None, None, None),
        };
        let (rac_teacher_latent, rac_teacher_shape) = match &self.rac_teacher_latent {
            Some(store) => {
                let teacher_load_start = profile_enabled.then(Instant::now);
                let batch = store.load_batch_data(&teacher_indices)?;
                profile.teacher_load_ns = profile.teacher_load_ns.saturating_add(
                    teacher_load_start
                        .map(|start| start.elapsed().as_nanos())
                        .unwrap_or_default(),
                );
                (
                    Some(batch.tensor),
                    Some((store.channels(), store.height(), store.width())),
                )
            }
            None => (None, None),
        };
        let teacher_targets = if self.teacher_targets.is_empty() {
            Vec::new()
        } else {
            let teacher_load_start = profile_enabled.then(Instant::now);
            let mut targets = Vec::with_capacity(self.teacher_targets.len());
            for target in &self.teacher_targets {
                let batch = target.store.load_batch_data(&teacher_indices)?;
                targets.push(ImageNetTeacherTargetBatchData {
                    name: target.name.clone(),
                    weight: target.weight,
                    target_kind: target.target_kind,
                    patch: batch.patch,
                    cls: batch.cls,
                    feature_dim: target.store.feature_dim(),
                    patch_tokens: target.store.patch_tokens(),
                });
            }
            profile.teacher_load_ns = profile.teacher_load_ns.saturating_add(
                teacher_load_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default(),
            );
            targets
        };
        Ok((
            ImageNetBatchData {
                images,
                target_images,
                view_images,
                view_crops,
                global_view_images,
                local_view_images,
                labels,
                teacher_patch,
                teacher_cls,
                rac_teacher_latent,
                teacher_targets,
                batch_size,
                global_image_size,
                local_image_size,
                global_views: self.global_views.max(1),
                local_views: self.local_views,
                teacher_feature_dim: teacher_dim,
                teacher_patch_tokens: teacher_tokens,
                rac_teacher_shape,
            },
            profile,
        ))
    }
}

fn limit_samples_balanced(samples: &[ImageNetSample], max_samples: usize) -> Vec<ImageNetSample> {
    if max_samples >= samples.len() {
        return samples.to_vec();
    }
    let num_classes = samples
        .iter()
        .map(|sample| sample.label)
        .max()
        .map(|value| value + 1)
        .unwrap_or(0);
    if num_classes == 0 || max_samples == 0 {
        return Vec::new();
    }

    let mut buckets = vec![Vec::new(); num_classes];
    for sample in samples {
        buckets[sample.label].push(sample.clone());
    }
    let mut offsets = vec![0usize; num_classes];
    let mut limited = Vec::with_capacity(max_samples);
    while limited.len() < max_samples {
        let mut made_progress = false;
        for class in 0..num_classes {
            let offset = &mut offsets[class];
            if *offset < buckets[class].len() {
                limited.push(buckets[class][*offset].clone());
                *offset += 1;
                made_progress = true;
                if limited.len() >= max_samples {
                    break;
                }
            }
        }
        if !made_progress {
            break;
        }
    }
    limited
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
    rac_teacher_latent: Option<Vec<f32>>,
    teacher_targets: Vec<ImageNetTeacherTargetBatchData>,
    batch_size: usize,
    global_image_size: usize,
    local_image_size: usize,
    global_views: usize,
    local_views: usize,
    teacher_feature_dim: Option<usize>,
    teacher_patch_tokens: Option<usize>,
    rac_teacher_shape: Option<(usize, usize, usize)>,
}

struct ImageNetTeacherTargetBatchData {
    name: String,
    weight: f32,
    target_kind: VisionTeacherTargetKind,
    patch: Option<Vec<f32>>,
    cls: Vec<f32>,
    feature_dim: usize,
    patch_tokens: Option<usize>,
}

impl ImageNetBatchData {
    fn host_to_device_copy_bytes(&self) -> u128 {
        let mut bytes = 0usize;
        bytes = bytes.saturating_add(self.images.len().saturating_mul(size_of::<f32>()));
        bytes = bytes.saturating_add(self.labels.len().saturating_mul(size_of::<i64>()));
        if let Some(buffer) = self.target_images.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.view_images.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.view_crops.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.global_view_images.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.local_view_images.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.teacher_patch.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.teacher_cls.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        if let Some(buffer) = self.rac_teacher_latent.as_ref() {
            bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
        }
        for target in &self.teacher_targets {
            bytes = bytes.saturating_add(target.cls.len().saturating_mul(size_of::<f32>()));
            if let Some(buffer) = target.patch.as_ref() {
                bytes = bytes.saturating_add(buffer.len().saturating_mul(size_of::<f32>()));
            }
        }
        bytes as u128
    }

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

        let teacher_patch = self.teacher_patch.map(|data| {
            Tensor::<B, 3>::from_data(
                TensorData::new(
                    data,
                    [
                        self.batch_size,
                        self.teacher_patch_tokens
                            .expect("teacher patch tokens required"),
                        self.teacher_feature_dim
                            .expect("teacher feature dim required"),
                    ],
                ),
                device,
            )
        });
        let teacher_cls = self.teacher_cls.map(|data| {
            Tensor::<B, 2>::from_data(
                TensorData::new(
                    data,
                    [
                        self.batch_size,
                        self.teacher_feature_dim
                            .expect("teacher feature dim required"),
                    ],
                ),
                device,
            )
        });
        let rac_teacher_latent = self.rac_teacher_latent.map(|data| {
            let (channels, height, width) = self
                .rac_teacher_shape
                .expect("rac teacher latent shape required");
            Tensor::<B, 4>::from_data(
                TensorData::new(data, [self.batch_size, channels, height, width]),
                device,
            )
        });
        let teacher_targets = self
            .teacher_targets
            .into_iter()
            .map(|target| {
                let patch = target.patch.map(|data| {
                    Tensor::<B, 3>::from_data(
                        TensorData::new(
                            data,
                            [
                                self.batch_size,
                                target.patch_tokens.expect("teacher patch tokens required"),
                                target.feature_dim,
                            ],
                        ),
                        device,
                    )
                });
                let cls = Tensor::<B, 2>::from_data(
                    TensorData::new(target.cls, [self.batch_size, target.feature_dim]),
                    device,
                );
                ImageNetTeacherTargetBatch {
                    name: target.name,
                    weight: target.weight,
                    target_kind: target.target_kind,
                    patch,
                    cls,
                }
            })
            .collect();

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
        .with_rac_teacher_latent(rac_teacher_latent)
        .with_teacher_targets(teacher_targets)
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
    pub rac_teacher_latent: Option<Tensor<B, 4>>,
    pub teacher_targets: Vec<ImageNetTeacherTargetBatch<B>>,
}

#[derive(Clone)]
pub struct ImageNetTeacherTargetBatch<B: Backend> {
    pub name: String,
    pub weight: f32,
    pub target_kind: VisionTeacherTargetKind,
    pub patch: Option<Tensor<B, 3>>,
    pub cls: Tensor<B, 2>,
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
            rac_teacher_latent: None,
            teacher_targets: Vec::new(),
        }
    }

    pub fn with_rac_teacher_latent(mut self, rac_teacher_latent: Option<Tensor<B, 4>>) -> Self {
        self.rac_teacher_latent = rac_teacher_latent;
        self
    }

    pub fn with_teacher_targets(
        mut self,
        teacher_targets: Vec<ImageNetTeacherTargetBatch<B>>,
    ) -> Self {
        self.teacher_targets = teacher_targets;
        self
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
            rac_teacher_latent: self
                .rac_teacher_latent
                .as_ref()
                .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
            teacher_targets: self
                .teacher_targets
                .iter()
                .cloned()
                .map(|target| ImageNetTeacherTargetBatch {
                    name: target.name,
                    weight: target.weight,
                    target_kind: target.target_kind,
                    patch: target.patch.map(|tensor| tensor.repeat_dim(0, repeats)),
                    cls: target.cls.repeat_dim(0, repeats),
                })
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct ImageNetBatchCpuProfile {
    image_load_ns: u128,
    image_transform_ns: u128,
    teacher_load_ns: u128,
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

#[derive(Clone, Copy, Debug, Default)]
struct ImageNetPrefetchProfile {
    cpu_ns: u128,
    image_load_ns: u128,
    image_transform_ns: u128,
    teacher_load_ns: u128,
    tensor_copy_ns: u128,
    host_to_device_copy_bytes: u128,
    host_sync_points: u64,
}

enum ImageNetPrefetchItem<B: Backend> {
    Data {
        data: Box<ImageNetBatchData>,
        profile: ImageNetPrefetchProfile,
    },
    Batch {
        batch: ImageNetBatch<B>,
        profile: ImageNetPrefetchProfile,
    },
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
        let prof_enabled = crate::train::profile::enabled();
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
                        let cpu_start = prof_enabled.then(Instant::now);
                        let result = dataset.sample_batch_data_profiled(batch_size).map(
                            |(data, stage_profile)| {
                                let cpu_ns = cpu_start
                                    .map(|start| start.elapsed().as_nanos())
                                    .unwrap_or_default();
                                let host_to_device_copy_bytes = data.host_to_device_copy_bytes();
                                (
                                    data,
                                    ImageNetPrefetchProfile {
                                        cpu_ns,
                                        image_load_ns: stage_profile.image_load_ns,
                                        image_transform_ns: stage_profile.image_transform_ns,
                                        teacher_load_ns: stage_profile.teacher_load_ns,
                                        tensor_copy_ns: 0,
                                        host_to_device_copy_bytes,
                                        host_sync_points: 0,
                                    },
                                )
                            },
                        );
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
                        Ok((data, mut profile)) => {
                            let tensor_copy_start = prof_enabled.then(Instant::now);
                            let _guard = crate::device::device_allocation_lock().lock().ok();
                            let batch = data.into_batch::<B>(&device);
                            if crate::train::profile::sync_timing_enabled() {
                                let _ = B::sync(&device);
                                profile.host_sync_points =
                                    profile.host_sync_points.saturating_add(1);
                            }
                            profile.tensor_copy_ns = tensor_copy_start
                                .map(|start| start.elapsed().as_nanos())
                                .unwrap_or_default();
                            Ok(ImageNetPrefetchItem::Batch { batch, profile })
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
                        let cpu_start = prof_enabled.then(Instant::now);
                        let result = dataset.sample_batch_data_profiled(batch_size).map(
                            |(data, stage_profile)| {
                                let cpu_ns = cpu_start
                                    .map(|start| start.elapsed().as_nanos())
                                    .unwrap_or_default();
                                let host_to_device_copy_bytes = data.host_to_device_copy_bytes();
                                ImageNetPrefetchItem::Data {
                                    data: Box::new(data),
                                    profile: ImageNetPrefetchProfile {
                                        cpu_ns,
                                        image_load_ns: stage_profile.image_load_ns,
                                        image_transform_ns: stage_profile.image_transform_ns,
                                        teacher_load_ns: stage_profile.teacher_load_ns,
                                        tensor_copy_ns: 0,
                                        host_to_device_copy_bytes,
                                        host_sync_points: 0,
                                    },
                                }
                            },
                        );
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
        let prof_enabled = crate::train::profile::enabled();

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
                Some(Ok(ImageNetPrefetchItem::Batch { batch, profile })) => {
                    if prof_enabled {
                        crate::train::profile::record_dataloader(
                            profile.cpu_ns,
                            profile.image_load_ns,
                            profile.image_transform_ns,
                            profile.teacher_load_ns,
                            profile.tensor_copy_ns,
                            profile.host_to_device_copy_bytes,
                            profile.host_sync_points,
                        );
                    }
                    batch
                }
                Some(Ok(ImageNetPrefetchItem::Data { data, mut profile })) => {
                    let tensor_copy_start = prof_enabled.then(Instant::now);
                    let batch = (*data).into_batch(&self.device);
                    if crate::train::profile::sync_timing_enabled() {
                        let _ = B::sync(&self.device);
                        profile.host_sync_points = profile.host_sync_points.saturating_add(1);
                    }
                    profile.tensor_copy_ns = tensor_copy_start
                        .map(|start| start.elapsed().as_nanos())
                        .unwrap_or_default();
                    if prof_enabled {
                        crate::train::profile::record_dataloader(
                            profile.cpu_ns,
                            profile.image_load_ns,
                            profile.image_transform_ns,
                            profile.teacher_load_ns,
                            profile.tensor_copy_ns,
                            profile.host_to_device_copy_bytes,
                            profile.host_sync_points,
                        );
                    }
                    batch
                }
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
            samples.push(ImageNetSample {
                path: image,
                label,
                teacher_index: samples.len(),
            });
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

fn infer_feature_storage_dtype(cls_path: &Path) -> Result<DinoFeatureDType> {
    let feature_dir = cls_path
        .parent()
        .ok_or_else(|| anyhow!("teacher cls path has no parent: {}", cls_path.display()))?;
    let split_name = cls_path
        .file_stem()
        .and_then(|stem| stem.to_str())
        .and_then(|stem| {
            stem.strip_suffix("_cls")
                .or_else(|| stem.strip_suffix("_latent"))
        });
    if let Some(split_name) = split_name {
        let split_meta_path = feature_dir.join(format!("{split_name}_meta.json"));
        if split_meta_path.is_file() {
            let meta = read_json_value(&split_meta_path)?;
            if let Some(dtype) = feature_storage_dtype_from_meta(&meta)? {
                return Ok(dtype);
            }
        }
    }

    let meta_path = feature_dir.join("meta.json");
    if meta_path.is_file() {
        let meta = read_json_value(&meta_path)?;
        if let Some(split_name) = split_name
            && let Some(dtype) = meta
                .get("splits")
                .and_then(|splits| splits.get(split_name))
                .map(feature_storage_dtype_from_meta)
                .transpose()?
                .flatten()
        {
            return Ok(dtype);
        }
        if let Some(dtype) = feature_storage_dtype_from_meta(&meta)? {
            return Ok(dtype);
        }
    }

    Ok(DinoFeatureDType::F32)
}

fn read_json_value(path: &Path) -> Result<Value> {
    let file =
        File::open(path).map_err(|err| anyhow!("failed to open {}: {err}", path.display()))?;
    serde_json::from_reader(file)
        .map_err(|err| anyhow!("failed to parse {}: {err}", path.display()))
}

fn feature_storage_dtype_from_meta(meta: &Value) -> Result<Option<DinoFeatureDType>> {
    if let Some(storage_dtype) = meta.get("storage_dtype").and_then(Value::as_str) {
        return Ok(Some(match storage_dtype {
            "f32" => DinoFeatureDType::F32,
            "f16" => DinoFeatureDType::F16,
            other => {
                return Err(anyhow!(
                    "unsupported teacher feature storage_dtype '{other}'"
                ));
            }
        }));
    }
    if let Some(bytes_per_scalar) = meta.get("bytes_per_scalar").and_then(Value::as_u64) {
        return Ok(Some(match bytes_per_scalar {
            BYTES_PER_F32 => DinoFeatureDType::F32,
            BYTES_PER_F16 => DinoFeatureDType::F16,
            other => {
                return Err(anyhow!(
                    "unsupported teacher feature bytes_per_scalar '{other}'"
                ));
            }
        }));
    }
    Ok(None)
}

fn read_feature_block_into(
    file: &mut File,
    offset: u64,
    scratch: &mut Vec<u8>,
    out: &mut [f32],
    dtype: DinoFeatureDType,
) -> Result<()> {
    let bytes = out.len() * dtype.bytes_per_scalar() as usize;
    if scratch.len() != bytes {
        scratch.resize(bytes, 0);
    }
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| anyhow!("failed to seek teacher features: {err}"))?;
    file.read_exact(scratch)
        .map_err(|err| anyhow!("failed to read teacher features: {err}"))?;

    match dtype {
        DinoFeatureDType::F32 => {
            for (dst, chunk) in out.iter_mut().zip(scratch.chunks_exact(4)) {
                *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            }
        }
        DinoFeatureDType::F16 => {
            for (dst, chunk) in out.iter_mut().zip(scratch.chunks_exact(2)) {
                *dst = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32();
            }
        }
    }
    Ok(())
}

fn load_backing_batch(
    backing: &DinoFeatureBacking,
    ordered: &[(usize, usize)],
    stride: usize,
    out: &mut [f32],
) -> Result<()> {
    match backing {
        DinoFeatureBacking::File { file, dtype } => {
            let mut file = file.lock().unwrap();
            let mut scratch = Vec::new();
            for (slot, index) in ordered {
                let offset = *index as u64 * stride as u64 * dtype.bytes_per_scalar();
                let start = *slot * stride;
                let end = start + stride;
                read_feature_block_into(
                    &mut file,
                    offset,
                    &mut scratch,
                    &mut out[start..end],
                    *dtype,
                )?;
            }
        }
        DinoFeatureBacking::MemoryF32(data) => {
            for (slot, index) in ordered {
                let src_start = *index * stride;
                let src_end = src_start + stride;
                let dst_start = *slot * stride;
                let dst_end = dst_start + stride;
                out[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
            }
        }
        DinoFeatureBacking::MemoryF16(data) => {
            for (slot, index) in ordered {
                let src_start = *index * stride;
                let src_end = src_start + stride;
                let dst_start = *slot * stride;
                let dst_end = dst_start + stride;
                for (dst, src) in out[dst_start..dst_end]
                    .iter_mut()
                    .zip(data[src_start..src_end].iter())
                {
                    *dst = src.to_f32();
                }
            }
        }
    }
    Ok(())
}

fn load_f32_file_into_memory(path: &Path, floats: usize) -> Result<Arc<Vec<f32>>> {
    let bytes_len = floats.saturating_mul(BYTES_PER_F32 as usize);
    let mut file =
        File::open(path).map_err(|err| anyhow!("failed to open {}: {err}", path.display()))?;
    let mut bytes = vec![0u8; bytes_len];
    file.read_exact(&mut bytes)
        .map_err(|err| anyhow!("failed to read {} into memory: {err}", path.display()))?;
    let mut data = vec![0.0f32; floats];
    for (dst, chunk) in data.iter_mut().zip(bytes.chunks_exact(4)) {
        *dst = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    Ok(Arc::new(data))
}

fn load_f16_file_into_memory(path: &Path, floats: usize) -> Result<Arc<Vec<f16>>> {
    let bytes_len = floats.saturating_mul(BYTES_PER_F16 as usize);
    let mut file =
        File::open(path).map_err(|err| anyhow!("failed to open {}: {err}", path.display()))?;
    let mut bytes = vec![0u8; bytes_len];
    file.read_exact(&mut bytes)
        .map_err(|err| anyhow!("failed to read {} into memory: {err}", path.display()))?;
    let mut data = vec![f16::from_f32(0.0); floats];
    for (dst, chunk) in data.iter_mut().zip(bytes.chunks_exact(2)) {
        *dst = f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]]));
    }
    Ok(Arc::new(data))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    #[test]
    fn dino_feature_store_reads_f16_cls_features_from_metadata() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path();
        let cls_path = root.join("train_cls.bin");
        let meta_path = root.join("train_meta.json");
        fs::write(
            &meta_path,
            br#"{
  "storage_dtype": "f16",
  "bytes_per_scalar": 2
}"#,
        )
        .unwrap();
        let values = [1.0f32, -0.5, 0.25, 2.0];
        let mut file = File::create(&cls_path).unwrap();
        for value in values {
            file.write_all(&f16::from_f32(value).to_bits().to_le_bytes())
                .unwrap();
        }
        drop(file);

        let store = DinoFeatureStore::new_optional_patch_with_options(
            &cls_path,
            None,
            2,
            None,
            Some(2),
            false,
        )
        .unwrap();
        let batch = store.load_batch_data(&[1, 0]).unwrap();
        assert!(batch.patch.is_none());
        assert_eq!(batch.cls.len(), 4);
        let expected = [0.25f32, 2.0, 1.0, -0.5];
        for (got, want) in batch.cls.iter().zip(expected) {
            assert!((got - want).abs() < 1e-3, "got {got}, want {want}");
        }
    }
}
