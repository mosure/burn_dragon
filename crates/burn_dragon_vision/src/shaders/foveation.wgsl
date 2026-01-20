struct FoveationParams {
    image_size: vec2<f32>,
    inv_image_size: vec2<f32>,
    center: vec2<f32>,
    sigma: vec2<f32>,
    sample_scale: f32,
    lod_sigma: f32,
    patch_size: f32,
    pyramid_levels: u32,
    mode: u32,
    warp_mode: u32,
    _pad0: u32,
    _pad1: u32,
};

const SUBSAMPLES: u32 = 4u;
const LOD_WINDOW: i32 = 3;
const AA_THRESHOLD: f32 = 1.25;
const SQRT2: f32 = 1.41421356237;
const PI: f32 = 3.14159265359;
const ERF_A: f32 = 0.147;
const SQRT_PI_OVER_2: f32 = 0.88622692545;
const INV_LN2: f32 = 1.4426950408889634;

struct FoveaWarp {
    offset: f32,
    deriv: f32,
};

@group(0) @binding(0) var gaussian_tex: texture_2d<f32>;
@group(0) @binding(1) var gaussian_sampler: sampler;
@group(0) @binding(2) var residual_tex: texture_2d<f32>;
@group(0) @binding(3) var output_tex: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(4) var<uniform> params: FoveationParams;

fn compute_lod(dx: f32, dy: f32, max_level: f32) -> f32 {
    if max_level <= 0.0 {
        return 0.0;
    }
    let sx = max(params.sigma.x, 1e-3);
    let sy = max(params.sigma.y, 1e-3);
    let dist = sqrt((dx * dx) / (sx * sx) + (dy * dy) / (sy * sy));
    if dist <= 1.0 {
        return 0.0;
    }
    let lod = log(dist) * INV_LN2;
    return clamp(lod, 0.0, max_level);
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

fn sample_gaussian(uv: vec2<f32>, lod_center: f32, lod_sigma: f32, max_level: u32) -> vec3<f32> {
    if params.warp_mode == 1u {
        let level = u32(clamp(floor(lod_center + 0.5), 0.0, f32(max_level)));
        return textureSampleLevel(gaussian_tex, gaussian_sampler, uv, f32(level)).xyz;
    }
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let base = i32(floor(lod_center));
    let max_i = i32(max_level);
    let start = max(base - LOD_WINDOW, 0);
    let end = min(base + LOD_WINDOW, max_i);
    var level = start;
    loop {
        if level > end {
            break;
        }
        let level_u = u32(level);
        let level_f = f32(level_u);
        let diff = (level_f - lod_center) / lod_sigma;
        let weight = exp(-0.5 * diff * diff);
        color += textureSampleLevel(gaussian_tex, gaussian_sampler, uv, level_f).xyz * weight;
        weight_sum += weight;
        level += 1;
    }
    return color / max(weight_sum, 1e-6);
}

fn reconstruct_laplacian(uv: vec2<f32>, start: u32, max_level: u32) -> vec3<f32> {
    var color = textureSampleLevel(gaussian_tex, gaussian_sampler, uv, f32(max_level)).xyz;
    var level = start;
    loop {
        if level >= max_level {
            break;
        }
        let sample = textureSampleLevel(residual_tex, gaussian_sampler, uv, f32(level)).xyz;
        color += sample;
        level += 1u;
    }
    return color;
}

fn sample_laplacian(uv: vec2<f32>, lod_center: f32, lod_sigma: f32, max_level: u32) -> vec3<f32> {
    if params.warp_mode == 1u {
        let level = u32(clamp(floor(lod_center + 0.5), 0.0, f32(max_level)));
        return reconstruct_laplacian(uv, level, max_level);
    }
    var color = vec3<f32>(0.0);
    var weight_sum = 0.0;
    let base = i32(floor(lod_center));
    let max_i = i32(max_level);
    let start = max(base - LOD_WINDOW, 0);
    let end = min(base + LOD_WINDOW, max_i);
    var level = start;
    loop {
        if level > end {
            break;
        }
        let level_u = u32(level);
        let level_f = f32(level_u);
        let diff = (level_f - lod_center) / lod_sigma;
        let weight = exp(-0.5 * diff * diff);
        let sample = reconstruct_laplacian(uv, level_u, max_level);
        color += sample * weight;
        weight_sum += weight;
        level += 1;
    }
    return color / max(weight_sum, 1e-6);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if x >= u32(params.patch_size) || y >= u32(params.patch_size) {
        return;
    }
    let half = params.patch_size * 0.5;
    let pixel_du = 1.0 / half;
    let radius = params.sample_scale * half;
    let max_level = params.pyramid_levels - 1u;
    let lod_sigma = max(params.lod_sigma, 1e-3);
    let patched = params.warp_mode == 1u;
    let ux_base = ((f32(x) + 0.5) - half) / half;
    let uy_base = ((f32(y) + 0.5) - half) / half;
    var color = vec3<f32>(0.0);
    if patched {
        let min_side = max(min(params.image_size.x, params.image_size.y), 1.0);
        let radius_norm = clamp(radius / min_side, 0.0, 1.0);
        let max_level_f = f32(max_level);
        let level_f = clamp(radius_norm * max_level_f, 0.0, max_level_f);
        let level0 = u32(floor(level_f));
        let level1 = min(level0 + 1u, max_level);
        let t = clamp(level_f - f32(level0), 0.0, 1.0);
        let level0_dims = textureDimensions(gaussian_tex, level0);
        let level1_dims = textureDimensions(gaussian_tex, level1);
        let level0_w = max(f32(level0_dims.x), 1.0);
        let level0_h = max(f32(level0_dims.y), 1.0);
        let level1_w = max(f32(level1_dims.x), 1.0);
        let level1_h = max(f32(level1_dims.y), 1.0);
        let center_norm = params.center * params.inv_image_size;
        let dx = (f32(x) + 0.5) - half;
        let dy = (f32(y) + 0.5) - half;
        let uv0 = vec2<f32>(
            center_norm.x + dx / level0_w,
            center_norm.y + dy / level0_h,
        );
        let uv1 = vec2<f32>(
            center_norm.x + dx / level1_w,
            center_norm.y + dy / level1_h,
        );
        if params.mode == 0u {
            let sample0 = textureSampleLevel(gaussian_tex, gaussian_sampler, uv0, f32(level0)).xyz;
            let sample1 = textureSampleLevel(gaussian_tex, gaussian_sampler, uv1, f32(level1)).xyz;
            color = sample0 + (sample1 - sample0) * t;
        } else {
            let sample0 = reconstruct_laplacian(uv0, level0, max_level);
            let sample1 = reconstruct_laplacian(uv1, level1, max_level);
            color = sample0 + (sample1 - sample0) * t;
        }
        textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(color, vec3(0.0), vec3(1.0)), 1.0));
        return;
    }
    var local_scale_base = 0.0;
    let warp_x_base = foveated_warp(ux_base, params.sigma.x, radius);
    let warp_y_base = foveated_warp(uy_base, params.sigma.y, radius);
    local_scale_base = max(abs(warp_x_base.deriv), abs(warp_y_base.deriv)) * pixel_du;

    var count = 0.0;
    if local_scale_base <= AA_THRESHOLD {
        var dx = 0.0;
        var dy = 0.0;
        var local_scale = 0.0;
        dx = warp_x_base.offset;
        dy = warp_y_base.offset;
        local_scale = local_scale_base;
        var lod_scale = 0.0;
        if local_scale > AA_THRESHOLD {
            lod_scale = log(local_scale / AA_THRESHOLD) * INV_LN2;
        }
        let uv = vec2<f32>(
            (params.center.x + dx) * params.inv_image_size.x,
            (params.center.y + dy) * params.inv_image_size.y,
        );
        let lod_dist = compute_lod(dx, dy, f32(max_level));
        let lod = clamp(max(lod_dist, lod_scale), 0.0, f32(max_level));
        if params.mode == 0u {
            color = sample_gaussian(uv, lod, lod_sigma, max_level);
        } else {
            color = sample_laplacian(uv, lod, lod_sigma, max_level);
        }
        count = 1.0;
    } else {
        for (var sy = 0u; sy < SUBSAMPLES; sy = sy + 1u) {
            for (var sx = 0u; sx < SUBSAMPLES; sx = sx + 1u) {
                let jitter = (vec2<f32>(f32(sx) + 0.5, f32(sy) + 0.5) / f32(SUBSAMPLES)) - vec2<f32>(0.5, 0.5);
                let ux = (((f32(x) + 0.5) - half) + jitter.x) / half;
                let uy = (((f32(y) + 0.5) - half) + jitter.y) / half;
                let warp_x = foveated_warp(ux, params.sigma.x, radius);
                let warp_y = foveated_warp(uy, params.sigma.y, radius);
                let dx = warp_x.offset;
                let dy = warp_y.offset;
                let local_scale = max(abs(warp_x.deriv), abs(warp_y.deriv)) * pixel_du;
                var lod_scale = 0.0;
                if local_scale > AA_THRESHOLD {
                    lod_scale = log(local_scale / AA_THRESHOLD) * INV_LN2;
                }
                let uv = vec2<f32>(
                    (params.center.x + dx) * params.inv_image_size.x,
                    (params.center.y + dy) * params.inv_image_size.y,
                );
                let lod_dist = compute_lod(dx, dy, f32(max_level));
                let lod = clamp(max(lod_dist, lod_scale), 0.0, f32(max_level));
                if params.mode == 0u {
                    color += sample_gaussian(uv, lod, lod_sigma, max_level);
                } else {
                    color += sample_laplacian(uv, lod, lod_sigma, max_level);
                }
                count += 1.0;
            }
        }
        color /= max(count, 1.0);
    }

    textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(color, vec3(0.0), vec3(1.0)), 1.0));
}
