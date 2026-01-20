const SUBSAMPLE_AXIS: u32 = 4u;
const SUBSAMPLES: u32 = SUBSAMPLE_AXIS * SUBSAMPLE_AXIS;
const LOD_WINDOW: i32 = 3;
const AA_THRESHOLD: f32 = 1.25;
const SQRT2: f32 = 1.41421356237;
const PI: f32 = 3.14159265359;
const ERF_A: f32 = 0.147;
const SQRT_PI_OVER_2: f32 = 0.88622692545;
const INV_LN2: f32 = 1.4426950408889634;
const MAX_LEVELS: u32 = 8u;

const META_PATCH_W: u32 = 0u;
const META_PATCH_H: u32 = 1u;
const META_CHANNELS: u32 = 2u;
const META_LEVEL_COUNT: u32 = 3u;
const META_RESIDUAL_COUNT: u32 = 4u;
const META_MODE: u32 = 5u;
const META_WARP_MODE: u32 = 6u;
const META_BASE_W: u32 = 7u;
const META_BASE_H: u32 = 8u;
const META_BATCH: u32 = 9u;

const META_GAUSS_OFF: u32 = 10u;
const META_GAUSS_W: u32 = META_GAUSS_OFF + MAX_LEVELS;
const META_GAUSS_H: u32 = META_GAUSS_W + MAX_LEVELS;
const META_RESIDUAL_OFF: u32 = META_GAUSS_H + MAX_LEVELS;
const META_RESIDUAL_W: u32 = META_RESIDUAL_OFF + MAX_LEVELS;
const META_RESIDUAL_H: u32 = META_RESIDUAL_W + MAX_LEVELS;
const META_COARSE_OFF: u32 = META_RESIDUAL_H + MAX_LEVELS;
const META_COARSE_W: u32 = META_COARSE_OFF + 1u;
const META_COARSE_H: u32 = META_COARSE_W + 1u;

@group(0) @binding(0) var<storage, read_write> gaussian: array<f32>;
@group(0) @binding(1) var<storage, read_write> residual: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> params: array<f32>;
@group(0) @binding(4) var<storage, read_write> metadata: array<f32>;

struct FoveaWarp {
    offset: f32,
    deriv: f32,
};

fn meta_u32(idx: u32) -> u32 {
    return u32(metadata[idx]);
}

fn meta_f32(idx: u32) -> f32 {
    return metadata[idx];
}

fn erf_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let ax = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * ax);
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-ax * ax);
    return sign * y;
}

fn erfinv_approx(x: f32) -> f32 {
    let sign = select(-1.0, 1.0, x >= 0.0);
    let xx = clamp(x, -0.999, 0.999);
    let ln = log(1.0 - xx * xx);
    let term = 2.0 / (PI * ERF_A) + ln * 0.5;
    let inside = max(term * term - ln / ERF_A, 0.0);
    let result = max(sqrt(inside) - term, 0.0);
    return sign * sqrt(result);
}

fn foveated_warp(u: f32, sigma: f32, radius: f32) -> FoveaWarp {
    let sigma_safe = max(sigma, 1e-3);
    let radius_safe = max(radius, 1e-3);
    let k = radius_safe / sigma_safe;
    let u_max = min(erf_approx(k / SQRT2), 0.999);
    let u_scaled = clamp(u, -1.0, 1.0) * u_max;
    let erf_inv = erfinv_approx(u_scaled);
    let offset = sigma_safe * SQRT2 * erf_inv;
    let deriv = sigma_safe * SQRT2 * u_max * SQRT_PI_OVER_2 * exp(erf_inv * erf_inv);
    return FoveaWarp(offset, deriv);
}

fn compute_lod(dx: f32, dy: f32, sigma: f32, local_scale: f32, max_level: f32) -> f32 {
    if max_level <= 0.0 {
        return 0.0;
    }
    let sigma_safe = max(sigma, 1e-3);
    let dist = sqrt(((dx * dx) / (sigma_safe * sigma_safe)) + ((dy * dy) / (sigma_safe * sigma_safe)));
    let lod_dist = select(0.0, log(dist) * INV_LN2, dist > 1.0);
    let lod_scale = select(0.0, log(local_scale / AA_THRESHOLD) * INV_LN2, local_scale > AA_THRESHOLD);
    return clamp(max(lod_dist, lod_scale), 0.0, max_level);
}

fn gaussian_offset(level: u32) -> u32 {
    return meta_u32(META_GAUSS_OFF + level);
}

fn gaussian_width(level: u32) -> u32 {
    return meta_u32(META_GAUSS_W + level);
}

fn gaussian_height(level: u32) -> u32 {
    return meta_u32(META_GAUSS_H + level);
}

fn residual_offset(level: u32) -> u32 {
    return meta_u32(META_RESIDUAL_OFF + level);
}

fn residual_width(level: u32) -> u32 {
    return meta_u32(META_RESIDUAL_W + level);
}

fn residual_height(level: u32) -> u32 {
    return meta_u32(META_RESIDUAL_H + level);
}

fn coarse_offset() -> u32 {
    return meta_u32(META_COARSE_OFF);
}

fn coarse_width() -> u32 {
    return meta_u32(META_COARSE_W);
}

fn coarse_height() -> u32 {
    return meta_u32(META_COARSE_H);
}

fn sample_bilinear_gaussian(
    base_offset: u32,
    width: u32,
    height: u32,
    channels: u32,
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
) -> f32 {
    if width == 0u || height == 0u || channels == 0u {
        return 0.0;
    }
    let fx_clamp = clamp(fx, 0.0, 1.0);
    let fy_clamp = clamp(fy, 0.0, 1.0);
    let x = fx_clamp * f32(width) - 0.5;
    let y = fy_clamp * f32(height) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let tx = x - x0;
    let ty = y - y0;
    let x0i = u32(clamp(x0, 0.0, f32(width - 1u)));
    let y0i = u32(clamp(y0, 0.0, f32(height - 1u)));
    let x1i = u32(clamp(x1, 0.0, f32(width - 1u)));
    let y1i = u32(clamp(y1, 0.0, f32(height - 1u)));

    let base = base_offset + ((b * channels + c) * height + y0i) * width;
    let c00 = gaussian[base + x0i];
    let c10 = gaussian[base + x1i];
    let c01 = gaussian[base + width * (y1i - y0i) + x0i];
    let c11 = gaussian[base + width * (y1i - y0i) + x1i];

    let a = mix(c00, c10, tx);
    let b0 = mix(c01, c11, tx);
    return mix(a, b0, ty);
}

fn sample_bilinear_residual(
    base_offset: u32,
    width: u32,
    height: u32,
    channels: u32,
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
) -> f32 {
    if width == 0u || height == 0u || channels == 0u {
        return 0.0;
    }
    let fx_clamp = clamp(fx, 0.0, 1.0);
    let fy_clamp = clamp(fy, 0.0, 1.0);
    let x = fx_clamp * f32(width) - 0.5;
    let y = fy_clamp * f32(height) - 0.5;
    let x0 = floor(x);
    let y0 = floor(y);
    let x1 = x0 + 1.0;
    let y1 = y0 + 1.0;
    let tx = x - x0;
    let ty = y - y0;
    let x0i = u32(clamp(x0, 0.0, f32(width - 1u)));
    let y0i = u32(clamp(y0, 0.0, f32(height - 1u)));
    let x1i = u32(clamp(x1, 0.0, f32(width - 1u)));
    let y1i = u32(clamp(y1, 0.0, f32(height - 1u)));

    let base = base_offset + ((b * channels + c) * height + y0i) * width;
    let c00 = residual[base + x0i];
    let c10 = residual[base + x1i];
    let c01 = residual[base + width * (y1i - y0i) + x0i];
    let c11 = residual[base + width * (y1i - y0i) + x1i];

    let a = mix(c00, c10, tx);
    let b0 = mix(c01, c11, tx);
    return mix(a, b0, ty);
}

fn sample_gaussian_level(
    level: u32,
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    channels: u32,
) -> f32 {
    let offset = gaussian_offset(level);
    let width = gaussian_width(level);
    let height = gaussian_height(level);
    return sample_bilinear_gaussian(offset, width, height, channels, b, c, fx, fy);
}

fn sample_residual_level(
    level: u32,
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    channels: u32,
) -> f32 {
    let offset = residual_offset(level);
    let width = residual_width(level);
    let height = residual_height(level);
    return sample_bilinear_residual(offset, width, height, channels, b, c, fx, fy);
}

fn sample_coarse(
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    channels: u32,
) -> f32 {
    return sample_bilinear_residual(coarse_offset(), coarse_width(), coarse_height(), channels, b, c, fx, fy);
}

fn sample_gaussian(
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    lod_center: f32,
    lod_sigma: f32,
    level_count: u32,
    warp_mode: u32,
    channels: u32,
) -> f32 {
    if level_count == 0u {
        return 0.0;
    }
    let max_level = f32(level_count - 1u);
    if warp_mode == 1u {
        let level = u32(clamp(floor(lod_center + 0.5), 0.0, max_level));
        return sample_gaussian_level(level, b, c, fx, fy, channels);
    }
    var color = 0.0;
    var weight_sum = 0.0;
    let base = i32(floor(lod_center));
    let max_i = i32(level_count) - 1;
    let start = max(base - LOD_WINDOW, 0);
    let end = min(base + LOD_WINDOW, max_i);
    var level = start;
    loop {
        if level > end {
            break;
        }
        let level_f = f32(level);
        let diff = (level_f - lod_center) / max(lod_sigma, 1e-3);
        let weight = exp(-0.5 * diff * diff);
        color += sample_gaussian_level(u32(level), b, c, fx, fy, channels) * weight;
        weight_sum += weight;
        level += 1;
    }
    return color / max(weight_sum, 1e-6);
}

fn sample_laplacian_at(
    start_idx: u32,
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    residual_count: u32,
    channels: u32,
) -> f32 {
    var color = sample_coarse(b, c, fx, fy, channels);
    var level = start_idx;
    loop {
        if level >= residual_count {
            break;
        }
        color += sample_residual_level(level, b, c, fx, fy, channels);
        level += 1u;
    }
    return color;
}

fn sample_laplacian(
    b: u32,
    c: u32,
    fx: f32,
    fy: f32,
    lod_center: f32,
    lod_sigma: f32,
    residual_count: u32,
    warp_mode: u32,
    channels: u32,
) -> f32 {
    let max_level = f32(residual_count);
    if warp_mode == 1u {
        let level = u32(clamp(floor(lod_center + 0.5), 0.0, max_level));
        return sample_laplacian_at(level, b, c, fx, fy, residual_count, channels);
    }
    var color = 0.0;
    var weight_sum = 0.0;
    let base = i32(floor(lod_center));
    let max_i = i32(residual_count);
    let start = max(base - LOD_WINDOW, 0);
    let end = min(base + LOD_WINDOW, max_i);
    var level = start;
    loop {
        if level > end {
            break;
        }
        let level_f = f32(level);
        let diff = (level_f - lod_center) / max(lod_sigma, 1e-3);
        let weight = exp(-0.5 * diff * diff);
        color += sample_laplacian_at(u32(level), b, c, fx, fy, residual_count, channels) * weight;
        weight_sum += weight;
        level += 1;
    }
    return color / max(weight_sum, 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let b = gid.z;
    let patch_w = meta_u32(META_PATCH_W);
    let patch_h = meta_u32(META_PATCH_H);
    if x >= patch_w || y >= patch_h {
        return;
    }
    let batch = meta_u32(META_BATCH);
    if b >= batch {
        return;
    }
    let channels = meta_u32(META_CHANNELS);
    if channels == 0u {
        return;
    }

    let mode = meta_u32(META_MODE);
    let warp_mode = meta_u32(META_WARP_MODE);
    let level_count = meta_u32(META_LEVEL_COUNT);
    let residual_count = meta_u32(META_RESIDUAL_COUNT);
    let base_width = meta_f32(META_BASE_W);
    let base_height = meta_f32(META_BASE_H);

    let half = f32(patch_h) * 0.5;
    let half_safe = max(half, 1.0);
    let pixel_du = 1.0 / half_safe;
    let base_dx = f32(x) + 0.5 - half;
    let base_dy = f32(y) + 0.5 - half;
    let ux_base = base_dx / half_safe;
    let uy_base = base_dy / half_safe;

    let param_idx = b * 5u;
    let center_x = params[param_idx];
    let center_y = params[param_idx + 1u];
    let sigma = max(params[param_idx + 2u], 1e-6);
    let radius = max(params[param_idx + 3u], 1e-6);
    let lod_sigma = max(params[param_idx + 4u], 1e-6);

    let patched = warp_mode == 1u;
    let mean_x = center_x / base_width;
    let mean_y = center_y / base_height;
    if patched {
        let min_side = max(min(base_width, base_height), 1.0);
        let radius_norm = clamp(radius / min_side, 0.0, 1.0);
        var max_level_u = level_count - 1u;
        if mode == 1u {
            max_level_u = residual_count;
        }
        let max_level = f32(max_level_u);
        let level_f = clamp(radius_norm * max_level, 0.0, max_level);
        let level0 = u32(floor(level_f));
        let level1 = min(level0 + 1u, max_level_u);
        let t = clamp(level_f - f32(level0), 0.0, 1.0);
        var level0_w = gaussian_width(level0);
        var level0_h = gaussian_height(level0);
        var level1_w = gaussian_width(level1);
        var level1_h = gaussian_height(level1);
        if mode == 1u {
            if level0 >= residual_count {
                level0_w = coarse_width();
                level0_h = coarse_height();
            } else {
                level0_w = residual_width(level0);
                level0_h = residual_height(level0);
            }
            if level1 >= residual_count {
                level1_w = coarse_width();
                level1_h = coarse_height();
            } else {
                level1_w = residual_width(level1);
                level1_h = residual_height(level1);
            }
        }
        let level0_w_f = max(f32(level0_w), 1.0);
        let level0_h_f = max(f32(level0_h), 1.0);
        let level1_w_f = max(f32(level1_w), 1.0);
        let level1_h_f = max(f32(level1_h), 1.0);
        let dx = base_dx;
        let dy = base_dy;
        let fx0 = mean_x + dx / level0_w_f;
        let fy0 = mean_y + dy / level0_h_f;
        let fx1 = mean_x + dx / level1_w_f;
        let fy1 = mean_y + dy / level1_h_f;

        var channel = 0u;
        loop {
            if channel >= channels {
                break;
            }
            var sample0 = sample_gaussian_level(level0, b, channel, fx0, fy0, channels);
            var sample1 = sample_gaussian_level(level1, b, channel, fx1, fy1, channels);
            if mode == 1u {
                sample0 = sample_laplacian_at(level0, b, channel, fx0, fy0, residual_count, channels);
                sample1 = sample_laplacian_at(level1, b, channel, fx1, fy1, residual_count, channels);
            }
            let sample = sample0 + (sample1 - sample0) * t;
            let out_index = ((b * channels + channel) * patch_h + y) * patch_w + x;
            output[out_index] = sample;
            channel += 1u;
        }
        return;
    }
    var local_scale_base = 0.0;
    if !patched {
        let warp_x_base = foveated_warp(ux_base, sigma, radius);
        let warp_y_base = foveated_warp(uy_base, sigma, radius);
        local_scale_base = max(abs(warp_x_base.deriv), abs(warp_y_base.deriv)) * pixel_du;
    }

    var channel = 0u;
    loop {
        if channel >= channels {
            break;
        }
        var accum = 0.0;
        var count = 0.0;
        if patched || local_scale_base <= AA_THRESHOLD {
            var dx = 0.0;
            var dy = 0.0;
            var local_scale = 0.0;
            if patched {
                dx = ux_base * radius;
                dy = uy_base * radius;
            } else {
                let warp_x_base = foveated_warp(ux_base, sigma, radius);
                let warp_y_base = foveated_warp(uy_base, sigma, radius);
                dx = warp_x_base.offset;
                dy = warp_y_base.offset;
                local_scale = local_scale_base;
            }
            let img_x = center_x + dx;
            let img_y = center_y + dy;
            let fx = img_x / base_width;
            let fy = img_y / base_height;
            var max_level = f32(level_count - 1u);
            if mode == 1u {
                max_level = f32(residual_count);
            }
            let lod_center = compute_lod(dx, dy, sigma, local_scale, max_level);
            var sample = sample_gaussian(b, channel, fx, fy, lod_center, lod_sigma, level_count, warp_mode, channels);
            if mode == 1u {
                sample = sample_laplacian(b, channel, fx, fy, lod_center, lod_sigma, residual_count, warp_mode, channels);
            }
            accum = sample;
            count = 1.0;
        } else {
            for (var sy = 0u; sy < SUBSAMPLE_AXIS; sy = sy + 1u) {
                for (var sx = 0u; sx < SUBSAMPLE_AXIS; sx = sx + 1u) {
                    let jitter_x = (f32(sx) + 0.5) / f32(SUBSAMPLE_AXIS) - 0.5;
                    let jitter_y = (f32(sy) + 0.5) / f32(SUBSAMPLE_AXIS) - 0.5;
                    let ux = (base_dx + jitter_x) / half_safe;
                    let uy = (base_dy + jitter_y) / half_safe;
                    let warp_x = foveated_warp(ux, sigma, radius);
                    let warp_y = foveated_warp(uy, sigma, radius);
                    let dx = warp_x.offset;
                    let dy = warp_y.offset;
                    let local_scale = max(abs(warp_x.deriv), abs(warp_y.deriv)) * pixel_du;
                    let img_x = center_x + dx;
                    let img_y = center_y + dy;
                    let fx = img_x / base_width;
                    let fy = img_y / base_height;
                    var max_level = f32(level_count - 1u);
                    if mode == 1u {
                        max_level = f32(residual_count);
                    }
                    let lod_center = compute_lod(dx, dy, sigma, local_scale, max_level);
                    var sample = sample_gaussian(b, channel, fx, fy, lod_center, lod_sigma, level_count, warp_mode, channels);
                    if mode == 1u {
                        sample = sample_laplacian(b, channel, fx, fy, lod_center, lod_sigma, residual_count, warp_mode, channels);
                    }
                    accum += sample;
                    count += 1.0;
                }
            }
        }
        let denom = max(count, 1.0);
        let out_index = ((b * channels + channel) * patch_h + y) * patch_w + x;
        output[out_index] = accum / denom;
        channel += 1u;
    }
}
