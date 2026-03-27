use crate::constants::FOVEA_AA_THRESHOLD;

const FOVEA_PARAM_EPS: f32 = 1e-3;
const SIGMA_MIN: f32 = 0.03;
const SIGMA_MAX: f32 = 0.5;
const LOD_LOG2_MIN: f32 = -2.0;
const LOD_LOG2_MAX: f32 = 1.0;
const LN_2: f32 = std::f32::consts::LN_2;
const SQRT2: f32 = std::f32::consts::SQRT_2;
const PI: f32 = std::f32::consts::PI;
const ERF_A: f32 = 0.147;
const SQRT_PI_OVER_2: f32 = 0.886_226_95;
const LOD_WINDOW: i32 = 3;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum PyramidMode {
    Stacked,
    #[default]
    Laplacian,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum FoveaWarpMode {
    #[default]
    Warped,
    Conformal,
    Patched,
}

#[derive(Clone, Debug)]
pub struct CpuImageLevel {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

#[derive(Clone, Debug)]
pub struct CpuPyramidCache {
    pub mode: PyramidMode,
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
    mode: PyramidMode,
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
    warp_mode: FoveaWarpMode,
) -> Vec<f32> {
    render_foveated_patch_with_radius(cache, mean, sigma, sigma, patch_size, warp_mode)
}

pub fn render_foveated_patch_with_radius(
    cache: &CpuPyramidCache,
    mean: [f32; 2],
    sigma: f32,
    radius: f32,
    patch_size: usize,
    warp_mode: FoveaWarpMode,
) -> Vec<f32> {
    const SUBSAMPLES: usize = 4;
    let patch = patch_size.max(1);
    let base = cache.gaussian.first().unwrap_or(&cache.coarse);
    let width = base.width.max(1);
    let height = base.height.max(1);
    let mean_x = mean[0].clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS);
    let mean_y = mean[1].clamp(FOVEA_PARAM_EPS, 1.0 - FOVEA_PARAM_EPS);
    let min_side = width.min(height) as f32;
    let radius_norm = radius.clamp(FOVEA_PARAM_EPS, 1.0);
    let mut out = vec![0.0; patch * patch * 3];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half.max(1.0);
    if matches!(warp_mode, FoveaWarpMode::Patched) {
        let (level0, level1, level_t) = match cache.mode {
            PyramidMode::Stacked => {
                let max_level = cache.gaussian.len().saturating_sub(1);
                patched_levels_from_radius(radius_norm, max_level)
            }
            PyramidMode::Laplacian => {
                let max_level = cache.laplacian.len();
                patched_levels_from_radius(radius_norm, max_level)
            }
        };
        let level_dims = |level: usize| match cache.mode {
            PyramidMode::Stacked => {
                let level_img = cache.gaussian.get(level).unwrap_or(base);
                (level_img.width.max(1), level_img.height.max(1))
            }
            PyramidMode::Laplacian => {
                if level >= cache.laplacian.len() {
                    (cache.coarse.width.max(1), cache.coarse.height.max(1))
                } else {
                    let level_img = &cache.laplacian[level];
                    (level_img.width.max(1), level_img.height.max(1))
                }
            }
        };
        let sample_at = |level: usize, fx: f32, fy: f32| match cache.mode {
            PyramidMode::Stacked => {
                let level_img = cache.gaussian.get(level).unwrap_or(base);
                sample_bilinear(level_img, fx, fy)
            }
            PyramidMode::Laplacian => {
                sample_laplacian_at(&cache.laplacian, &cache.coarse, level, fx, fy)
            }
        };
        let (level0_w, level0_h) = level_dims(level0);
        let (level1_w, level1_h) = level_dims(level1);
        let center0_x = mean_x * level0_w as f32;
        let center0_y = mean_y * level0_h as f32;
        let center1_x = mean_x * level1_w as f32;
        let center1_y = mean_y * level1_h as f32;
        let blend = if level0 == level1 { 0.0 } else { level_t };
        for y in 0..patch {
            for x in 0..patch {
                let dx = x as f32 + 0.5 - half;
                let dy = y as f32 + 0.5 - half;
                let fx0 = (center0_x + dx) / level0_w as f32;
                let fy0 = (center0_y + dy) / level0_h as f32;
                let sample0 = sample_at(level0, fx0, fy0);
                let sample = if blend <= f32::EPSILON {
                    sample0
                } else {
                    let fx1 = (center1_x + dx) / level1_w as f32;
                    let fy1 = (center1_y + dy) / level1_h as f32;
                    let sample1 = sample_at(level1, fx1, fy1);
                    [
                        sample0[0] + (sample1[0] - sample0[0]) * blend,
                        sample0[1] + (sample1[1] - sample0[1]) * blend,
                        sample0[2] + (sample1[2] - sample0[2]) * blend,
                    ]
                };
                let idx = (y * patch + x) * 3;
                out[idx] = sample[0];
                out[idx + 1] = sample[1];
                out[idx + 2] = sample[2];
            }
        }
        return out;
    }
    let center_x = mean_x * width as f32;
    let center_y = mean_y * height as f32;
    let sigma_norm = sigma.clamp(FOVEA_PARAM_EPS, 1.0).min(radius_norm);
    let radius_px = (radius_norm * min_side).max(FOVEA_PARAM_EPS);
    let sigma_px = (sigma_norm * min_side).max(FOVEA_PARAM_EPS);
    let lod_sigma = lod_sigma_from_sigma(sigma_norm);
    for y in 0..patch {
        for x in 0..patch {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let ux_base = base_dx / half.max(1.0);
            let uy_base = base_dy / half.max(1.0);
            let warp_base = sample_foveated_warp(ux_base, uy_base, sigma_px, radius_px, warp_mode);
            let local_scale_base = warp_base.deriv.abs() * pixel_du;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            if local_scale_base <= FOVEA_AA_THRESHOLD {
                let offset_x = warp_base.dx;
                let offset_y = warp_base.dy;
                let img_x = center_x + offset_x;
                let img_y = center_y + offset_y;
                let fx = img_x / width as f32;
                let fy = img_y / height as f32;
                let sample = match cache.mode {
                    PyramidMode::Stacked => sample_gaussian_foveated(
                        &cache.gaussian,
                        offset_x,
                        offset_y,
                        sigma_px,
                        sigma_px,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        warp_mode,
                    ),
                    PyramidMode::Laplacian => sample_laplacian_foveated(
                        &cache.laplacian,
                        &cache.coarse,
                        offset_x,
                        offset_y,
                        sigma_px,
                        sigma_px,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        warp_mode,
                    ),
                };
                color = sample;
                count = 1.0;
            } else {
                for sy in 0..SUBSAMPLES {
                    for sx in 0..SUBSAMPLES {
                        let jitter_x = (sx as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let jitter_y = (sy as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let ux = (base_dx + jitter_x) / half.max(1.0);
                        let uy = (base_dy + jitter_y) / half.max(1.0);
                        let warp = sample_foveated_warp(ux, uy, sigma_px, radius_px, warp_mode);
                        let offset_x = warp.dx;
                        let offset_y = warp.dy;
                        let local_scale = warp.deriv.abs() * pixel_du;
                        let img_x = center_x + offset_x;
                        let img_y = center_y + offset_y;
                        let fx = img_x / width as f32;
                        let fy = img_y / height as f32;
                        let sample = match cache.mode {
                            PyramidMode::Stacked => sample_gaussian_foveated(
                                &cache.gaussian,
                                offset_x,
                                offset_y,
                                sigma_px,
                                sigma_px,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                warp_mode,
                            ),
                            PyramidMode::Laplacian => sample_laplacian_foveated(
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
                                warp_mode,
                            ),
                        };
                        color[0] += sample[0];
                        color[1] += sample[1];
                        color[2] += sample[2];
                        count += 1.0;
                    }
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
    (log2 * LN_2).exp()
}

fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let a1 = 0.254_829_6;
    let a2 = -0.284_496_72;
    let a3 = 1.421_413_8;
    let a4 = -1.453_152_1;
    let a5 = 1.061_405_4;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-ax * ax).exp();
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

struct FoveaWarp2d {
    dx: f32,
    dy: f32,
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

fn conformal_alpha_from_ratio(ratio: f32) -> f32 {
    let q = ratio.clamp(1e-4, 1.0);
    if q >= 1.0 - 1e-4 {
        return 0.0;
    }
    let mut alpha = if q > 0.75 {
        (6.0 * (1.0 - q)).max(0.0).sqrt()
    } else {
        (2.0 / q.max(1e-4)).ln().max(0.0)
    };
    for _ in 0..4 {
        let a = alpha.max(1e-4);
        let sinh_a = a.sinh();
        let cosh_a = a.cosh();
        let g = a / sinh_a - q;
        let gp = (sinh_a - a * cosh_a) / (sinh_a * sinh_a).max(1e-6);
        alpha = (a - g / gp).max(0.0);
    }
    alpha
}

fn conformal_warp(ux: f32, uy: f32, sigma: f32, radius: f32) -> FoveaWarp2d {
    let sigma_safe = sigma.max(1e-3);
    let radius_safe = radius.max(1e-3);
    let ratio = (sigma_safe / radius_safe).clamp(1e-4, 1.0);
    let alpha = conformal_alpha_from_ratio(ratio);
    if alpha <= 1e-3 {
        return FoveaWarp2d {
            dx: radius_safe * ux.clamp(-1.0, 1.0),
            dy: radius_safe * uy.clamp(-1.0, 1.0),
            deriv: radius_safe,
        };
    }

    let ux = ux.clamp(-1.0, 1.0);
    let uy = uy.clamp(-1.0, 1.0);
    let ax = alpha * ux;
    let ay = alpha * uy;
    let sinh_alpha = alpha.sinh().max(1e-6);
    let norm = radius_safe / sinh_alpha;
    let dx = norm * ax.sinh() * ay.cos();
    let dy = norm * ax.cosh() * ay.sin();
    let deriv_re = ax.cosh() * ay.cos();
    let deriv_im = ax.sinh() * ay.sin();
    let deriv = norm * alpha * (deriv_re * deriv_re + deriv_im * deriv_im).sqrt();
    FoveaWarp2d { dx, dy, deriv }
}

fn sample_foveated_warp(
    ux: f32,
    uy: f32,
    sigma: f32,
    radius: f32,
    warp_mode: FoveaWarpMode,
) -> FoveaWarp2d {
    match warp_mode {
        FoveaWarpMode::Warped | FoveaWarpMode::Patched => {
            let warp_x = foveated_warp(ux, sigma, radius);
            let warp_y = foveated_warp(uy, sigma, radius);
            FoveaWarp2d {
                dx: warp_x.offset,
                dy: warp_y.offset,
                deriv: warp_x.deriv.abs().max(warp_y.deriv.abs()),
            }
        }
        FoveaWarpMode::Conformal => conformal_warp(ux, uy, sigma, radius),
    }
}

#[allow(clippy::too_many_arguments)]
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
    warp_mode: FoveaWarpMode,
) -> [f32; 3] {
    if levels.is_empty() {
        return [0.0, 0.0, 0.0];
    }
    let max_level_idx = levels.len().saturating_sub(1);
    let max_level = max_level_idx as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    if matches!(warp_mode, FoveaWarpMode::Patched) {
        let (level0, level1, t) = patched_levels_from_lod(lod_center, max_level_idx);
        let sample0 = sample_bilinear(&levels[level0], fx, fy);
        if level0 == level1 || t <= f32::EPSILON {
            return sample0;
        }
        let sample1 = sample_bilinear(&levels[level1], fx, fy);
        return [
            sample0[0] + (sample1[0] - sample0[0]) * t,
            sample0[1] + (sample1[1] - sample0[1]) * t,
            sample0[2] + (sample1[2] - sample0[2]) * t,
        ];
    }
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

#[allow(clippy::too_many_arguments)]
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
    warp_mode: FoveaWarpMode,
) -> [f32; 3] {
    let max_level_idx = residuals.len();
    let max_level = max_level_idx as f32;
    let lod_center = compute_lod(dx, dy, sigma_x, sigma_y, max_level, local_scale);
    if matches!(warp_mode, FoveaWarpMode::Patched) {
        let (level0, level1, t) = patched_levels_from_lod(lod_center, max_level_idx);
        let sample0 = sample_laplacian_at(residuals, coarse, level0, fx, fy);
        if level0 == level1 || t <= f32::EPSILON {
            return sample0;
        }
        let sample1 = sample_laplacian_at(residuals, coarse, level1, fx, fy);
        return [
            sample0[0] + (sample1[0] - sample0[0]) * t,
            sample0[1] + (sample1[1] - sample0[1]) * t,
            sample0[2] + (sample1[2] - sample0[2]) * t,
        ];
    }
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
    let lod_dist = if dist <= 1.0 { 0.0 } else { dist.ln() / LN_2 };
    let lod_scale = if local_scale <= FOVEA_AA_THRESHOLD {
        0.0
    } else {
        (local_scale / FOVEA_AA_THRESHOLD).ln() / LN_2
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

fn patched_levels_from_radius(radius_norm: f32, max_level: usize) -> (usize, usize, f32) {
    if max_level == 0 {
        return (0, 0, 0.0);
    }
    let max_level_f = max_level as f32;
    let level_f = (radius_norm.clamp(0.0, 1.0) * max_level_f).clamp(0.0, max_level_f);
    let level0 = level_f.floor() as usize;
    let level1 = (level0 + 1).min(max_level);
    let t = (level_f - level0 as f32).clamp(0.0, 1.0);
    (level0, level1, t)
}

fn patched_levels_from_lod(lod_center: f32, max_level: usize) -> (usize, usize, f32) {
    if max_level == 0 {
        return (0, 0, 0.0);
    }
    let max_level_f = max_level as f32;
    let level_f = lod_center.clamp(0.0, max_level_f);
    let level0 = level_f.floor() as usize;
    let level1 = (level0 + 1).min(max_level);
    let t = (level_f - level0 as f32).clamp(0.0, 1.0);
    (level0, level1, t)
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
        for (idx, value) in data.iter_mut().enumerate() {
            *value = current.data[idx] - up.data[idx];
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
            for (ky, wy) in weights.iter().enumerate() {
                let wy = *wy;
                let sy = (y * 2).saturating_add(ky).saturating_sub(2);
                let sy = sy.min(level.height - 1);
                for (kx, wx) in weights.iter().enumerate() {
                    let wx = *wx;
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
    CpuImageLevel {
        width,
        height,
        data,
    }
}

fn sample_bilinear(level: &CpuImageLevel, fx: f32, fy: f32) -> [f32; 3] {
    // Match burn's grid_sample mapping and clamping.
    let grid_x = if level.width > 1 {
        (fx * level.width as f32 - 0.5) * (2.0 / (level.width - 1) as f32) - 1.0
    } else {
        0.0
    }
    .clamp(-1.0, 1.0);
    let grid_y = if level.height > 1 {
        (fy * level.height as f32 - 0.5) * (2.0 / (level.height - 1) as f32) - 1.0
    } else {
        0.0
    }
    .clamp(-1.0, 1.0);
    let x_half = (level.width - 1) as f32 * 0.5;
    let y_half = (level.height - 1) as f32 * 0.5;
    let x = grid_x * x_half + x_half;
    let y = grid_y * y_half + y_half;
    let x0 = x.floor();
    let y0 = y.floor();
    let x1 = (x + 1.0).floor();
    let y1 = (y + 1.0).floor();
    let x0i = x0.clamp(0.0, (level.width - 1) as f32) as usize;
    let y0i = y0.clamp(0.0, (level.height - 1) as f32) as usize;
    let x1i = x1.clamp(0.0, (level.width - 1) as f32) as usize;
    let y1i = y1.clamp(0.0, (level.height - 1) as f32) as usize;

    let c00 = get_pixel(level, x0i, y0i);
    let c10 = get_pixel(level, x1i, y0i);
    let c01 = get_pixel(level, x0i, y1i);
    let c11 = get_pixel(level, x1i, y1i);

    let weight_00 = (x1 - x) * (y1 - y);
    let weight_10 = (x - x0) * (y1 - y);
    let weight_01 = (x1 - x) * (y - y0);
    let weight_11 = (x - x0) * (y - y0);

    [
        c00[0] * weight_00 + c10[0] * weight_10 + c01[0] * weight_01 + c11[0] * weight_11,
        c00[1] * weight_00 + c10[1] * weight_10 + c01[1] * weight_01 + c11[1] * weight_11,
        c00[2] * weight_00 + c10[2] * weight_10 + c01[2] * weight_01 + c11[2] * weight_11,
    ]
}

fn get_pixel(level: &CpuImageLevel, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * level.width + x) * 3;
    [level.data[idx], level.data[idx + 1], level.data[idx + 2]]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn constant_image(width: usize, height: usize, value: f32) -> CpuImageLevel {
        let mut data = vec![0.0; width * height * 3];
        for pixel in data.chunks_exact_mut(3) {
            pixel.fill(value);
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
            for (out, (residual_val, up_val)) in data
                .iter_mut()
                .zip(residual.data.iter().zip(up.data.iter()))
            {
                *out = residual_val + up_val;
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
        let cache = build_pyramid_cache(base, 4, PyramidMode::Stacked);
        let patch = render_foveated_patch(&cache, [0.5, 0.5], 0.1, 8, FoveaWarpMode::Warped);
        for value in patch {
            assert!((value - 0.25).abs() < 1e-3);
        }
    }

    #[test]
    fn conformal_patch_constant_image_is_constant() {
        let base = constant_image(16, 16, 0.25);
        let cache = build_pyramid_cache(base, 4, PyramidMode::Stacked);
        let patch = render_foveated_patch_with_radius(
            &cache,
            [0.5, 0.5],
            0.1,
            0.35,
            8,
            FoveaWarpMode::Conformal,
        );
        for value in patch {
            assert!((value - 0.25).abs() < 1e-3);
        }
    }

    #[test]
    fn patched_blends_between_levels() {
        fn constant_level(width: usize, height: usize, value: f32) -> CpuImageLevel {
            let mut data = vec![0.0; width * height * 3];
            for pixel in data.chunks_exact_mut(3) {
                pixel.fill(value);
            }
            CpuImageLevel {
                width,
                height,
                data,
            }
        }

        let levels = vec![
            constant_level(8, 8, 0.1),
            constant_level(4, 4, 0.4),
            constant_level(2, 2, 0.7),
        ];
        let cache = CpuPyramidCache {
            mode: PyramidMode::Stacked,
            gaussian: levels,
            laplacian: Vec::new(),
            coarse: constant_level(1, 1, 0.0),
        };
        let mean = [0.5, 0.5];
        let sigma = 0.2;
        let patch_size = 4;
        let values: [f32; 3] = [0.1, 0.4, 0.7];
        let cases: [f32; 5] = [0.0, 0.25, 0.5, 0.75, 1.0];
        for radius in cases {
            let radius_norm = radius.clamp(FOVEA_PARAM_EPS, 1.0);
            let (level0, level1, t) =
                patched_levels_from_radius(radius_norm, cache.gaussian.len().saturating_sub(1));
            let expected = values[level0] + (values[level1] - values[level0]) * t;
            let patch = render_foveated_patch_with_radius(
                &cache,
                mean,
                sigma,
                radius,
                patch_size,
                FoveaWarpMode::Patched,
            );
            for value in patch {
                assert!(
                    (value - expected).abs() < 1e-4,
                    "expected {expected} got {value}"
                );
            }
        }
    }
}
