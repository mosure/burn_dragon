use crate::CpuImageLevel;
use image::imageops::FilterType;
use image::{DynamicImage, RgbImage, Rgba, RgbaImage};
use std::path::Path;

#[derive(Clone)]
pub struct LoadedCpuArtifactSource {
    pub rgba: RgbaImage,
    pub cpu_level: CpuImageLevel,
}

impl LoadedCpuArtifactSource {
    pub fn width(&self) -> usize {
        self.cpu_level.width
    }

    pub fn height(&self) -> usize {
        self.cpu_level.height
    }
}

pub fn load_cpu_artifact_source(
    path: &Path,
    max_source_side: u32,
) -> image::ImageResult<LoadedCpuArtifactSource> {
    let mut image = image::open(path)?.to_rgba8();
    let (width, height) = image.dimensions();
    let max_side = width.max(height);
    if max_source_side > 0 && max_side > max_source_side {
        let scale = max_source_side as f32 / max_side as f32;
        let new_width = ((width as f32 * scale).round() as u32).max(1);
        let new_height = ((height as f32 * scale).round() as u32).max(1);
        image = DynamicImage::ImageRgba8(image)
            .resize(new_width, new_height, FilterType::Lanczos3)
            .to_rgba8();
    }
    let (width, height) = image.dimensions();
    let mut data = Vec::with_capacity(width as usize * height as usize * 3);
    for pixel in image.pixels() {
        let alpha = pixel[3] as f32 / 255.0;
        let inv_alpha = 1.0 - alpha;
        data.push((pixel[0] as f32 / 255.0) * alpha + inv_alpha);
        data.push((pixel[1] as f32 / 255.0) * alpha + inv_alpha);
        data.push((pixel[2] as f32 / 255.0) * alpha + inv_alpha);
    }
    Ok(LoadedCpuArtifactSource {
        rgba: image,
        cpu_level: CpuImageLevel {
            width: width as usize,
            height: height as usize,
            data,
        },
    })
}

pub fn rgb_f32_to_rgba_image(data: &[f32], width: u32, height: u32) -> RgbaImage {
    let expected = width as usize * height as usize * 3;
    assert_eq!(
        data.len(),
        expected,
        "expected {expected} rgb values, got {}",
        data.len()
    );
    let mut image = RgbaImage::new(width, height);
    for y in 0..height {
        for x in 0..width {
            let idx = (y as usize * width as usize + x as usize) * 3;
            image.put_pixel(
                x,
                y,
                Rgba([
                    to_u8(data[idx]),
                    to_u8(data[idx + 1]),
                    to_u8(data[idx + 2]),
                    255,
                ]),
            );
        }
    }
    image
}

pub fn save_rgb_f32_patch_image(
    path: &Path,
    patch: &[f32],
    patch_size: usize,
) -> image::ImageResult<()> {
    save_rgb_f32_image(path, patch_size, patch_size, patch)
}

pub fn save_rgb_f32_image(
    path: &Path,
    width: usize,
    height: usize,
    data: &[f32],
) -> image::ImageResult<()> {
    let expected = width * height * 3;
    assert_eq!(
        data.len(),
        expected,
        "expected {expected} rgb values, got {}",
        data.len()
    );
    let mut bytes = Vec::with_capacity(expected);
    for value in data.iter().copied() {
        bytes.push(to_u8(value));
    }
    let image = RgbImage::from_raw(width as u32, height as u32, bytes).expect("rgb image buffer");
    image.save(path)
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}
