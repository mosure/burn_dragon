use crate::VisionPyramidMode;

const FOVEA_PARAM_EPS: f32 = 1e-3;
const SIGMA_MIN: f32 = 0.03;
const SIGMA_MAX: f32 = 0.5;
const LOD_LOG2_MIN: f32 = -2.0;
const LOD_LOG2_MAX: f32 = 1.0;
const SQRT2: f32 = 1.41421356237;
const PI: f32 = 3.14159265359;
const ERF_A: f32 = 0.147;
const SQRT_PI_OVER_2: f32 = 0.88622692545;
const LOD_WINDOW: i32 = 3;

#[derive(Clone, Debug)]
pub struct CpuImageLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CpuPyramidCache {
    pub mode: VisionPyramidMode,
    pub gaussian: Vec<CpuImageLevel>,
    pub laplacian: Vec<CpuImageLevel>,
    pub coarse: CpuImageLevel,
}

pub fn image_from_nchw(
    data: &[f32],
    batch_idx: usize,
    channels: usize,
    height: usize,
    width: usize,
) -> Option<CpuImageLevel> {
    if channels < 3 || height == 0 || width == 0 {
        return None;
    }
    let frame_stride = channels * height * width;
    let start = batch_idx.checked_mul(frame_stride)?;
    let end = start.checked_add(frame_stride)?;
    if end > data.len() {
        return None;
    }
    let mut out = vec![0.0; width * height * 3];
    for y in 0..height {
        for x in 0..width {
            let base = start + y * width + x;
            let idx = (y * width + x) * 3;
            out[idx] = data[base];
            out[idx + 1] = data[base + height * width];
            out[idx + 2] = data[base + 2 * height * width];
        }
    }
    Some(CpuImageLevel {
        width,
        height,
        data: out,
    })
}

pub fn build_pyramid_cache(
    image: CpuImageLevel,
    depth: usize,
    mode: VisionPyramidMode,
) -> CpuPyramidCache {
    let gaussian = build_gaussian_pyramid(&image, depth);
    let (laplacian, coarse) = build_laplacian_pyramid(&gaussian);
    CpuPyramidCache {
        mode,
        gaussian,
        laplacian,
        coarse,
    }
}

pub fn render_foveated_patch(
    cache: &CpuPyramidCache,
    mean: [f32; 2],
    sigma: f32,
    patch_size: usize,
) -> Vec<f32> {
    render_foveated_patch_with_radius(cache, mean, sigma, sigma, patch_size)
}

pub fn render_foveated_patch_with_radius(
    cache: &CpuPyramidCache,
    mean: [f32; 2],
    sigma: f32,
    radius: f32,
    patch_size: usize,
) -> Vec<f32> {
    const SUBSAMPLES: usize = 4;
    let patch = patch_size.max(1);
    let base = cache.gaussian.first().unwrap_or(&cache.coarse);
    let width = base.width.max(1);
    let height = base.height.max(1);
    let center_x = mean[0].clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS) * width as f32;
    let center_y = mean[1].clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS) * height as f32;
    let min_side = width.min(height) as f32;
    let radius_norm = radius.clamp(FOVEA_PARAM_EPS, 1.0);
    let sigma_norm = sigma.clamp(FOVEA_PARAM_EPS, 1.0).min(radius_norm);
    let radius_px = (radius_norm * min_side).max(FOVEA_PARAM_EPS);
    let sigma_px = (sigma_norm * min_side).max(FOVEA_PARAM_EPS);
    let lod_sigma = lod_sigma_from_sigma(sigma_norm);
    let mut out = vec![0.0; patch * patch * 3];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half.max(1.0);
    for y in 0..patch {
        for x in 0..patch {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            for sy in 0..SUBSAMPLES {
                for sx in 0..SUBSAMPLES {
                    let jitter_x = (sx as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                    let jitter_y = (sy as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                    let ux = (base_dx + jitter_x) / half.max(1.0);
                    let uy = (base_dy + jitter_y) / half.max(1.0);
                    let warp_x = foveated_warp(ux, sigma_px, radius_px);
                    let warp_y = foveated_warp(uy, sigma_px, radius_px);
                    let offset_x = warp_x.offset;
                    let offset_y = warp_y.offset;
                    let local_scale = warp_x
                        .deriv
                        .abs()
                        .max(warp_y.deriv.abs())
                        * pixel_du;
                    let img_x = center_x + offset_x;
                    let img_y = center_y + offset_y;
                    let fx = img_x / width as f32;
                    let fy = img_y / height as f32;
                    let sample = match cache.mode {
                        VisionPyramidMode::Stacked => sample_gaussian_foveated(
                            &cache.gaussian,
                            offset_x,
                            offset_y,
                            sigma_px,
                            sigma_px,
                            local_scale,
                            lod_sigma,
                            fx,
                            fy,
                        ),
                        VisionPyramidMode::Laplacian => sample_laplacian_foveated(
                            &cache.laplacian,
                            &cache.coarse,
                            offset_x,
                            offset_y,
                            sigma_px,
                            sigma_px,
                            local_scale,
                            lod_sigma,
                            fx,
                            fy,
                        ),
                    };
                    color[0] += sample[0];
                    color[1] += sample[1];
                    color[2] += sample[2];
                    count += 1.0;
                }
            }
            if count > 0.0 {
                color[0] /= count;
                color[1] /= count;
                color[2] /= count;
            }
            let idx = (y * patch + x) * 3;
            out[idx] = color[0];
            out[idx + 1] = color[1];
            out[idx + 2] = color[2];
        }
    }
    out
}

pub fn sigma_from_unit(unit: f32) -> f32 {
    let t = unit.clamp(0.0, 1.0);
    SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * t
}

pub fn lod_sigma_from_sigma(sigma: f32) -> f32 {
    let range = (SIGMA_MAX - SIGMA_MIN).max(FOVEA_PARAM_EPS);
    let t = ((sigma - SIGMA_MIN) / range).clamp(0.0, 1.0);
    let log2 = t * (LOD_LOG2_MAX - LOD_LOG2_MIN) + LOD_LOG2_MIN;
    (log2 * 0.69314718056).exp()
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
    sign * y
}

fn erfinv_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let xx = x.clamp(-0.999, 0.999);
    let ln = (1.0 - xx * xx).ln();
    let term = 2.0 / (PI * ERF_A) + ln * 0.5;
    let inside = (term * term - ln / ERF_A).max(0.0);
    let result = (inside.sqrt() - term).max(0.0);
    sign * result.sqrt()
}

struct FoveaWarp {
    offset: f32,
    deriv: f32,
}

fn foveated_warp(u: f32, sigma: f32, radius: f32) -> FoveaWarp {
    let sigma = sigma.max(1e-3);
    let radius = radius.max(1e-3);
    let k = radius / sigma;
    let u_max = erf_approx(k / SQRT2).min(0.999);
    let u_scaled = u.clamp(-1.0, 1.0) * u_max;
    let erf_inv = erfinv_approx(u_scaled);
    let offset = sigma * SQRT2 * erf_inv;
    let deriv = sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * (erf_inv * erf_inv).exp();
    FoveaWarp { offset, deriv }
}

fn sample_gaussian_foveated(
    levels: &[CpuImageLevel],
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    local_scale: f32,
    lod_sigma: f32,
    fx: f32,
    fy: f32,
) -> [f32; 3] {
    if levels.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let max_level = (levels.len().saturating_sub(1)) as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    let mut color = [0.0; 3];
    let mut weight_sum = 0.0;
    let base = lod_center.floor() as i32;
    let start = (base - LOD_WINDOW).max(0);
    let end = (base + LOD_WINDOW).min(levels.len() as i32 - 1);
    for level_idx in start..=end {
        let level = &levels[level_idx as usize];
        let diff = (level_idx as f32 - lod_center) / lod_sigma.max(1e-3);
        let weight = (-0.5 * diff * diff).exp();
        let sample = sample_bilinear(level, fx, fy);
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        color[0] /= weight_sum;
        color[1] /= weight_sum;
        color[2] /= weight_sum;
    }
    color
}

fn sample_laplacian_foveated(
    residuals: &[CpuImageLevel],
    coarse: &CpuImageLevel,
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    local_scale: f32,
    lod_sigma: f32,
    fx: f32,
    fy: f32,
) -> [f32; 3] {
    let max_level = residuals.len() as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    let mut color = [0.0; 3];
    let mut weight_sum = 0.0;
    let base = lod_center.floor() as i32;
    let start = (base - LOD_WINDOW).max(0);
    let end = (base + LOD_WINDOW).min(residuals.len() as i32);
    for level_idx in start..=end {
        let diff = (level_idx as f32 - lod_center) / lod_sigma.max(1e-3);
        let weight = (-0.5 * diff * diff).exp();
        let sample = sample_laplacian_at(residuals, coarse, level_idx as usize, fx, fy);
        color[0] += sample[0] * weight;
        color[1] += sample[1] * weight;
        color[2] += sample[2] * weight;
        weight_sum += weight;
    }
    if weight_sum > 0.0 {
        color[0] /= weight_sum;
        color[1] /= weight_sum;
        color[2] /= weight_sum;
    }
    color
}

fn compute_lod(
    dx: f32,
    dy: f32,
    sigma_x: f32,
    sigma_y: f32,
    max_level: f32,
    local_scale: f32,
) -> f32 {
    if max_level <= 0.0 {
        return 0.0;
    }
    let sx = sigma_x.max(1e-3);
    let sy = sigma_y.max(1e-3);
    let dist = ((dx * dx) / (sx * sx) + (dy * dy) / (sy * sy)).sqrt();
    let lod_dist = if dist <= 1.0 { 0.0 } else { dist.log2() };
    let lod_scale = if local_scale <= 1.0 {
        0.0
    } else {
        local_scale.log2()
    };
    lod_dist.max(lod_scale).clamp(0.0, max_level)
}

fn sample_laplacian_at(
    residuals: &[CpuImageLevel],
    coarse: &CpuImageLevel,
    start_idx: usize,
    fx: f32,
    fy: f32,
) -> [f32; 3] {
    let mut color = sample_bilinear(coarse, fx, fy);
    for (idx, residual) in residuals.iter().enumerate() {
        if idx < start_idx {
            continue;
        }
        let sample = sample_bilinear(residual, fx, fy);
        color[0] += sample[0];
        color[1] += sample[1];
        color[2] += sample[2];
    }
    color
}

fn build_gaussian_pyramid(base: &CpuImageLevel, depth: usize) -> Vec<CpuImageLevel> {
    let mut out = Vec::with_capacity(depth.max(1));
    out.push(base.clone());
    for _ in 1..depth {
        let next = downsample(out.last().expect("pyramid level"));
        out.push(next);
    }
    out
}

fn build_laplacian_pyramid(gaussian: &[CpuImageLevel]) -> (Vec<CpuImageLevel>, CpuImageLevel) {
    if gaussian.is_empty() {
        return (
            Vec::new(),
            CpuImageLevel {
                width: 1,
                height: 1,
                data: vec![0.0; 3],
            },
        );
    }
    let mut residuals = Vec::with_capacity(gaussian.len().saturating_sub(1));
    for idx in 0..gaussian.len().saturating_sub(1) {
        let current = &gaussian[idx];
        let next = &gaussian[idx + 1];
        let up = resample(next, current.width, current.height);
        let mut data = vec![0.0; current.width * current.height * 3];
        for i in 0..data.len() {
            data[i] = current.data[i] - up.data[i];
        }
        residuals.push(CpuImageLevel {
            width: current.width,
            height: current.height,
            data,
        });
    }
    let coarse = gaussian.last().cloned().expect("coarse");
    (residuals, coarse)
}

fn downsample(level: &CpuImageLevel) -> CpuImageLevel {
    let new_w = (level.width / 2).max(1);
    let new_h = (level.height / 2).max(1);
    let mut data = vec![0.0; new_w * new_h * 3];
    let weights = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
    for y in 0..new_h {
        for x in 0..new_w {
            let mut accum = [0.0; 3];
            for ky in 0..5 {
                let wy = weights[ky];
                let sy = (y * 2).saturating_add(ky).saturating_sub(2);
                let sy = sy.min(level.height - 1);
                for kx in 0..5 {
                    let wx = weights[kx];
                    let sx = (x * 2).saturating_add(kx).saturating_sub(2);
                    let sx = sx.min(level.width - 1);
                    let weight = wx * wy;
                    let sample = get_pixel(level, sx, sy);
                    accum[0] += sample[0] * weight;
                    accum[1] += sample[1] * weight;
                    accum[2] += sample[2] * weight;
                }
            }
            let idx = (y * new_w + x) * 3;
            data[idx] = accum[0] / 256.0;
            data[idx + 1] = accum[1] / 256.0;
            data[idx + 2] = accum[2] / 256.0;
        }
    }
    CpuImageLevel {
        width: new_w,
        height: new_h,
        data,
    }
}

fn resample(level: &CpuImageLevel, width: usize, height: usize) -> CpuImageLevel {
    let mut data = vec![0.0; width * height * 3];
    for y in 0..height {
        let fy = (y as f32 + 0.5) / height as f32;
        for x in 0..width {
            let fx = (x as f32 + 0.5) / width as f32;
            let sample = sample_bilinear(level, fx, fy);
            let idx = (y * width + x) * 3;
            data[idx] = sample[0];
            data[idx + 1] = sample[1];
            data[idx + 2] = sample[2];
        }
    }
    CpuImageLevel { width, height, data }
}

fn sample_bilinear(level: &CpuImageLevel, fx: f32, fy: f32) -> [f32; 3] {
    // Match WGSL/texture sampling (0..1 maps to texel edges), and burn's grid mapping.
    let x = fx.clamp(0.0, 1.0) * level.width as f32 - 0.5;
    let y = fy.clamp(0.0, 1.0) * level.height as f32 - 0.5;
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let tx = x - x0;
    let ty = y - y0;
    let x0i = x0.clamp(0.0, (level.width - 1) as f32) as usize;
    let y0i = y0.clamp(0.0, (level.height - 1) as f32) as usize;
    let x1i = x1.clamp(0.0, (level.width - 1) as f32) as usize;
    let y1i = y1.clamp(0.0, (level.height - 1) as f32) as usize;

    let c00 = get_pixel(level, x0i, y0i);
    let c10 = get_pixel(level, x1i, y0i);
    let c01 = get_pixel(level, x0i, y1i);
    let c11 = get_pixel(level, x1i, y1i);

    let a = [
        lerp(c00[0], c10[0], tx),
        lerp(c00[1], c10[1], tx),
        lerp(c00[2], c10[2], tx),
    ];
    let b = [
        lerp(c01[0], c11[0], tx),
        lerp(c01[1], c11[1], tx),
        lerp(c01[2], c11[2], tx),
    ];
    [
        lerp(a[0], b[0], ty),
        lerp(a[1], b[1], ty),
        lerp(a[2], b[2], ty),
    ]
}

fn get_pixel(level: &CpuImageLevel, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * level.width + x) * 3;
    [
        level.data[idx],
        level.data[idx + 1],
        level.data[idx + 2],
    ]
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_image(width: usize, height: usize, value: f32) -> CpuImageLevel {
        let mut data = vec![0.0; width * height * 3];
        for idx in 0..(width * height) {
            let base = idx * 3;
            data[base] = value;
            data[base + 1] = value;
            data[base + 2] = value;
        }
        CpuImageLevel {
            width,
            height,
            data,
        }
    }

    #[test]
    fn gaussian_laplacian_roundtrip_constant() {
        let base = constant_image(8, 8, 0.5);
        let gaussian = build_gaussian_pyramid(&base, 3);
        let (laplacian, coarse) = build_laplacian_pyramid(&gaussian);
        let mut recon = coarse.clone();
        for residual in laplacian.iter().rev() {
            let up = resample(&recon, residual.width, residual.height);
            let mut data = vec![0.0; residual.width * residual.height * 3];
            for i in 0..data.len() {
                data[i] = residual.data[i] + up.data[i];
            }
            recon = CpuImageLevel {
                width: residual.width,
                height: residual.height,
                data,
            };
        }
        for value in recon.data {
            assert!((value - 0.5).abs() < 1e-4);
        }
    }

    #[test]
    fn foveated_patch_constant_image_is_constant() {
        let base = constant_image(16, 16, 0.25);
        let cache = build_pyramid_cache(base, 4, VisionPyramidMode::Stacked);
        let patch = render_foveated_patch(&cache, [0.5, 0.5], 0.1, 8);
        for value in patch {
            assert!((value - 0.25).abs() < 1e-3);
        }
    }
}
