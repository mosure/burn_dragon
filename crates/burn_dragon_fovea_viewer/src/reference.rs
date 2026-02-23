use super::*;

#[cfg(test)]
#[allow(dead_code)]
pub(crate) fn render_patch(
    source: &SourceImage,
    cache: &PyramidCache,
    settings: &FoveationSettings,
    patch_size: usize,
) -> Vec<u8> {
    const SUBSAMPLES: usize = 4;
    let patch = patch_size.max(1);
    let width = patch;
    let height = patch;
    let sample = FoveationSample {
        mean_x: settings.mean_x,
        mean_y: settings.mean_y,
        radius_norm: settings.radius_norm,
    };
    let mean_x = sample.mean_x.clamp(0.0, 1.0);
    let mean_y = sample.mean_y.clamp(0.0, 1.0);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(settings, &sample);
    let sigma = sigma_px_from_norm(sigma_norm, source);
    let radius = radius_px_from_norm(radius_norm, source);
    let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
    let mut out = vec![0u8; width * height * 4];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half;
    if matches!(settings.warp_mode, FoveaWarpMode::Patched) {
        let (level0, level1, level_t) = match settings.mode {
            PyramidMode::Gaussian => {
                let max_level = cache.gaussian.len().saturating_sub(1);
                patched_levels_from_radius(radius_norm, max_level)
            }
            PyramidMode::Laplacian => {
                let max_level = cache.laplacian.len();
                patched_levels_from_radius(radius_norm, max_level)
            }
        };
        let level_dims = |level: usize| match settings.mode {
            PyramidMode::Gaussian => cache
                .gaussian
                .get(level)
                .map(|level| (level.width.max(1), level.height.max(1)))
                .unwrap_or((source.width.max(1), source.height.max(1))),
            PyramidMode::Laplacian => {
                if level >= cache.laplacian.len() {
                    cache
                        .coarse
                        .as_ref()
                        .or_else(|| cache.gaussian.last())
                        .map(|level| (level.width.max(1), level.height.max(1)))
                        .unwrap_or((source.width.max(1), source.height.max(1)))
                } else {
                    let level_img = &cache.laplacian[level];
                    (level_img.width.max(1), level_img.height.max(1))
                }
            }
        };
        let coarse = cache
            .coarse
            .as_ref()
            .unwrap_or_else(|| cache.gaussian.last().expect("coarse level"));
        let sample_at = |level: usize, fx: f32, fy: f32| match settings.mode {
            PyramidMode::Gaussian => {
                let level_img = cache
                    .gaussian
                    .get(level)
                    .unwrap_or_else(|| cache.gaussian.first().expect("gaussian level"));
                sample_bilinear(level_img, fx, fy)
            }
            PyramidMode::Laplacian => sample_laplacian_at(&cache.laplacian, coarse, level, fx, fy),
        };
        let (level0_w, level0_h) = level_dims(level0);
        let (level1_w, level1_h) = level_dims(level1);
        let center0_x = mean_x * level0_w as f32;
        let center0_y = mean_y * level0_h as f32;
        let center1_x = mean_x * level1_w as f32;
        let center1_y = mean_y * level1_h as f32;
        let blend = if level0 == level1 { 0.0 } else { level_t };
        for y in 0..height {
            for x in 0..width {
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
                let idx = (y * width + x) * 4;
                out[idx] = (sample[0].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 1] = (sample[1].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 2] = (sample[2].clamp(0.0, 1.0) * 255.0).round() as u8;
                out[idx + 3] = 255;
            }
        }
        return out;
    }
    let center_x = mean_x * source.width as f32;
    let center_y = mean_y * source.height as f32;
    for y in 0..height {
        for x in 0..width {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let ux_base = base_dx / half;
            let uy_base = base_dy / half;
            let warp_x_base = foveated_warp(ux_base, sigma, radius);
            let warp_y_base = foveated_warp(uy_base, sigma, radius);
            let local_scale_base = warp_x_base.deriv.abs().max(warp_y_base.deriv.abs()) * pixel_du;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            if local_scale_base <= FOVEA_AA_THRESHOLD {
                let offset_x = warp_x_base.offset;
                let offset_y = warp_y_base.offset;
                let img_x = center_x + offset_x;
                let img_y = center_y + offset_y;
                let fx = img_x / source.width as f32;
                let fy = img_y / source.height as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => sample_gaussian_foveated(
                        &cache.gaussian,
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                    PyramidMode::Laplacian => sample_laplacian_foveated(
                        &cache.laplacian,
                        cache.coarse.as_ref(),
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                };
                color = sample;
                count = 1.0;
            } else {
                for sy in 0..SUBSAMPLES {
                    for sx in 0..SUBSAMPLES {
                        let jitter_x = (sx as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let jitter_y = (sy as f32 + 0.5) / SUBSAMPLES as f32 - 0.5;
                        let ux = (base_dx + jitter_x) / half;
                        let uy = (base_dy + jitter_y) / half;
                        let warp_x = foveated_warp(ux, sigma, radius);
                        let warp_y = foveated_warp(uy, sigma, radius);
                        let offset_x = warp_x.offset;
                        let offset_y = warp_y.offset;
                        let local_scale = warp_x.deriv.abs().max(warp_y.deriv.abs()) * pixel_du;
                        let img_x = center_x + offset_x;
                        let img_y = center_y + offset_y;
                        let fx = img_x / source.width as f32;
                        let fy = img_y / source.height as f32;
                        let sample = match settings.mode {
                            PyramidMode::Gaussian => sample_gaussian_foveated(
                                &cache.gaussian,
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                            PyramidMode::Laplacian => sample_laplacian_foveated(
                                &cache.laplacian,
                                cache.coarse.as_ref(),
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
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
            let idx = (y * width + x) * 4;
            out[idx] = (color[0].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 1] = (color[1].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 2] = (color[2].clamp(0.0, 1.0) * 255.0).round() as u8;
            out[idx + 3] = 255;
        }
    }
    out
}

#[cfg(test)]
pub(crate) fn render_patch_f32(
    source: &SourceImage,
    cache: &PyramidCache,
    settings: &FoveationSettings,
    patch_size: usize,
) -> Vec<f32> {
    let subsamples = burn_dragon_vision::SACCADE_FOVEA_SUBSAMPLES.max(1);
    let patch = patch_size.max(1);
    let width = patch;
    let height = patch;
    let sample = FoveationSample {
        mean_x: settings.mean_x,
        mean_y: settings.mean_y,
        radius_norm: settings.radius_norm,
    };
    let mean_x = sample.mean_x.clamp(0.0, 1.0);
    let mean_y = sample.mean_y.clamp(0.0, 1.0);
    let radius_norm = radius_norm_from_sample(&sample);
    let sigma_norm = sigma_norm_from_settings(settings, &sample);
    let sigma = sigma_px_from_norm(sigma_norm, source);
    let radius = radius_px_from_norm(radius_norm, source);
    let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
    let mut out = vec![0.0f32; width * height * 3];

    let half = patch as f32 * 0.5;
    let pixel_du = 1.0 / half;
    if matches!(settings.warp_mode, FoveaWarpMode::Patched) {
        let (level0, level1, level_t) = match settings.mode {
            PyramidMode::Gaussian => {
                let max_level = cache.gaussian.len().saturating_sub(1);
                patched_levels_from_radius(radius_norm, max_level)
            }
            PyramidMode::Laplacian => {
                let max_level = cache.laplacian.len();
                patched_levels_from_radius(radius_norm, max_level)
            }
        };
        let level_dims = |level: usize| match settings.mode {
            PyramidMode::Gaussian => cache
                .gaussian
                .get(level)
                .map(|level| (level.width.max(1), level.height.max(1)))
                .unwrap_or((source.width.max(1), source.height.max(1))),
            PyramidMode::Laplacian => {
                if level >= cache.laplacian.len() {
                    cache
                        .coarse
                        .as_ref()
                        .or_else(|| cache.gaussian.last())
                        .map(|level| (level.width.max(1), level.height.max(1)))
                        .unwrap_or((source.width.max(1), source.height.max(1)))
                } else {
                    let level_img = &cache.laplacian[level];
                    (level_img.width.max(1), level_img.height.max(1))
                }
            }
        };
        let coarse = cache
            .coarse
            .as_ref()
            .unwrap_or_else(|| cache.gaussian.last().expect("coarse level"));
        let sample_at = |level: usize, fx: f32, fy: f32| match settings.mode {
            PyramidMode::Gaussian => {
                let level_img = cache
                    .gaussian
                    .get(level)
                    .unwrap_or_else(|| cache.gaussian.first().expect("gaussian level"));
                sample_bilinear(level_img, fx, fy)
            }
            PyramidMode::Laplacian => sample_laplacian_at(&cache.laplacian, coarse, level, fx, fy),
        };
        let (level0_w, level0_h) = level_dims(level0);
        let (level1_w, level1_h) = level_dims(level1);
        let center0_x = mean_x * level0_w as f32;
        let center0_y = mean_y * level0_h as f32;
        let center1_x = mean_x * level1_w as f32;
        let center1_y = mean_y * level1_h as f32;
        let blend = if level0 == level1 { 0.0 } else { level_t };
        for y in 0..height {
            for x in 0..width {
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
                let idx = (y * width + x) * 3;
                out[idx] = sample[0];
                out[idx + 1] = sample[1];
                out[idx + 2] = sample[2];
            }
        }
        return out;
    }
    let center_x = mean_x * source.width as f32;
    let center_y = mean_y * source.height as f32;
    for y in 0..height {
        for x in 0..width {
            let base_dx = x as f32 + 0.5 - half;
            let base_dy = y as f32 + 0.5 - half;
            let ux_base = base_dx / half;
            let uy_base = base_dy / half;
            let warp_x_base = foveated_warp(ux_base, sigma, radius);
            let warp_y_base = foveated_warp(uy_base, sigma, radius);
            let local_scale_base = warp_x_base.deriv.abs().max(warp_y_base.deriv.abs()) * pixel_du;
            let mut color = [0.0; 3];
            let mut count = 0.0;
            if local_scale_base <= FOVEA_AA_THRESHOLD {
                let offset_x = warp_x_base.offset;
                let offset_y = warp_y_base.offset;
                let img_x = center_x + offset_x;
                let img_y = center_y + offset_y;
                let fx = img_x / source.width as f32;
                let fy = img_y / source.height as f32;
                let sample = match settings.mode {
                    PyramidMode::Gaussian => sample_gaussian_foveated(
                        &cache.gaussian,
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                    PyramidMode::Laplacian => sample_laplacian_foveated(
                        &cache.laplacian,
                        cache.coarse.as_ref(),
                        offset_x,
                        offset_y,
                        sigma,
                        sigma,
                        local_scale_base,
                        lod_sigma,
                        fx,
                        fy,
                        settings.warp_mode,
                    ),
                };
                color = sample;
                count = 1.0;
            } else {
                for sy in 0..subsamples {
                    for sx in 0..subsamples {
                        let jitter_x = (sx as f32 + 0.5) / subsamples as f32 - 0.5;
                        let jitter_y = (sy as f32 + 0.5) / subsamples as f32 - 0.5;
                        let ux = (base_dx + jitter_x) / half;
                        let uy = (base_dy + jitter_y) / half;
                        let warp_x = foveated_warp(ux, sigma, radius);
                        let warp_y = foveated_warp(uy, sigma, radius);
                        let offset_x = warp_x.offset;
                        let offset_y = warp_y.offset;
                        let local_scale = warp_x.deriv.abs().max(warp_y.deriv.abs()) * pixel_du;
                        let img_x = center_x + offset_x;
                        let img_y = center_y + offset_y;
                        let fx = img_x / source.width as f32;
                        let fy = img_y / source.height as f32;
                        let sample = match settings.mode {
                            PyramidMode::Gaussian => sample_gaussian_foveated(
                                &cache.gaussian,
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
                            ),
                            PyramidMode::Laplacian => sample_laplacian_foveated(
                                &cache.laplacian,
                                cache.coarse.as_ref(),
                                offset_x,
                                offset_y,
                                sigma,
                                sigma,
                                local_scale,
                                lod_sigma,
                                fx,
                                fy,
                                settings.warp_mode,
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
            let idx = (y * width + x) * 3;
            out[idx] = color[0];
            out[idx + 1] = color[1];
            out[idx + 2] = color[2];
        }
    }
    out
}

#[cfg(test)]
const SQRT2: f32 = std::f32::consts::SQRT_2;

#[cfg(test)]
const PI: f32 = std::f32::consts::PI;

#[cfg(test)]
const ERF_A: f32 = 0.147;

#[cfg(test)]
const SQRT_PI_OVER_2: f32 = 0.886_226_95;

#[cfg(test)]
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

#[cfg(test)]
fn erfinv_approx(x: f32) -> f32 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let xx = x.clamp(-0.999, 0.999);
    let ln = (1.0 - xx * xx).ln();
    let term = 2.0 / (PI * ERF_A) + ln * 0.5;
    let inside = (term * term - ln / ERF_A).max(0.0);
    let result = (inside.sqrt() - term).max(0.0);
    sign * result.sqrt()
}

#[cfg(test)]
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

#[cfg(test)]
struct FoveaWarp {
    offset: f32,
    deriv: f32,
}

#[cfg(test)]
const LOD_WINDOW: i32 = 3;

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_gaussian_foveated(
    levels: &[ImageLevel],
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

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn sample_laplacian_foveated(
    residuals: &[ImageLevel],
    coarse: Option<&ImageLevel>,
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
    let Some(coarse) = coarse else {
        return [0.0, 0.0, 0.0];
    };
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

#[cfg(test)]
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

#[cfg(test)]
pub(crate) fn compute_lod(
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
    let lod_dist = if dist <= 1.0 {
        0.0
    } else {
        dist.ln() / std::f32::consts::LN_2
    };
    let lod_scale = if local_scale <= FOVEA_AA_THRESHOLD {
        0.0
    } else {
        (local_scale / FOVEA_AA_THRESHOLD).ln() / std::f32::consts::LN_2
    };
    lod_dist.max(lod_scale).clamp(0.0, max_level)
}

#[cfg(test)]
fn sample_laplacian_at(
    residuals: &[ImageLevel],
    coarse: &ImageLevel,
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

#[cfg(test)]
pub(crate) fn build_gaussian_pyramid(base: &ImageLevel, depth: usize) -> Vec<ImageLevel> {
    let mut out = Vec::with_capacity(depth.max(1));
    out.push(base.clone());
    for _ in 1..depth {
        let next = downsample(out.last().expect("pyramid level"));
        out.push(next);
    }
    out
}

#[cfg(test)]
pub(crate) fn build_laplacian_pyramid(gaussian: &[ImageLevel]) -> (Vec<ImageLevel>, ImageLevel) {
    if gaussian.is_empty() {
        return (
            Vec::new(),
            ImageLevel {
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
        for (out, (current_val, up_val)) in
            data.iter_mut().zip(current.data.iter().zip(up.data.iter()))
        {
            *out = current_val - up_val;
        }
        residuals.push(ImageLevel {
            width: current.width,
            height: current.height,
            data,
        });
    }
    let coarse = gaussian.last().cloned().expect("coarse");
    (residuals, coarse)
}

#[cfg(test)]
fn downsample(level: &ImageLevel) -> ImageLevel {
    let new_w = (level.width / 2).max(1);
    let new_h = (level.height / 2).max(1);
    let mut data = vec![0.0; new_w * new_h * 3];
    let weights = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
    for y in 0..new_h {
        for x in 0..new_w {
            let mut accum = [0.0; 3];
            for (ky, &wy) in weights.iter().enumerate() {
                let sy = (y * 2).saturating_add(ky).saturating_sub(2);
                let sy = sy.min(level.height - 1);
                for (kx, &wx) in weights.iter().enumerate() {
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
    ImageLevel {
        width: new_w,
        height: new_h,
        data,
    }
}

#[cfg(test)]
fn resample(level: &ImageLevel, width: usize, height: usize) -> ImageLevel {
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
    ImageLevel {
        width,
        height,
        data,
    }
}

#[cfg(test)]
fn sample_bilinear(level: &ImageLevel, fx: f32, fy: f32) -> [f32; 3] {
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

#[cfg(test)]
fn get_pixel(level: &ImageLevel, x: usize, y: usize) -> [f32; 3] {
    let idx = (y * level.width + x) * 3;
    [level.data[idx], level.data[idx + 1], level.data[idx + 2]]
}
