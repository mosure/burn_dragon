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
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const SUBSAMPLES: u32 = 2u;
const LOD_WINDOW: i32 = 2;
const SQRT2: f32 = 1.41421356237;
const PI: f32 = 3.14159265359;
const ERF_A: f32 = 0.147;

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
    let lod = log2(dist);
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

fn foveated_offset(u: f32, sigma: f32, radius: f32) -> f32 {
    let sigma_safe = max(sigma, 1e-3);
    let radius_safe = max(radius, 1e-3);
    let k = radius_safe / sigma_safe;
    let u_max = min(erf_approx(k / SQRT2), 0.999);
    let u_scaled = clamp(u, -1.0, 1.0) * u_max;
    return sigma_safe * SQRT2 * erfinv_approx(u_scaled);
}

fn sample_gaussian(uv: vec2<f32>, lod_center: f32, lod_sigma: f32, max_level: u32) -> vec3<f32> {
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
    let radius = params.sample_scale * half;
    let max_level = params.pyramid_levels - 1u;
    let lod_sigma = max(params.lod_sigma, 1e-3);
    var color = vec3<f32>(0.0);
    var count = 0.0;
    for (var sy = 0u; sy < SUBSAMPLES; sy = sy + 1u) {
        for (var sx = 0u; sx < SUBSAMPLES; sx = sx + 1u) {
            let jitter = (vec2<f32>(f32(sx) + 0.5, f32(sy) + 0.5) / f32(SUBSAMPLES)) - vec2<f32>(0.5, 0.5);
            let ux = (((f32(x) + 0.5) - half) + jitter.x) / half;
            let uy = (((f32(y) + 0.5) - half) + jitter.y) / half;
            let dx = foveated_offset(ux, params.sigma.x, radius);
            let dy = foveated_offset(uy, params.sigma.y, radius);
            let uv = vec2<f32>(
                (params.center.x + dx) * params.inv_image_size.x,
                (params.center.y + dy) * params.inv_image_size.y,
            );
            let lod = compute_lod(dx, dy, f32(max_level));
            if params.mode == 0u {
                color += sample_gaussian(uv, lod, lod_sigma, max_level);
            } else {
                color += sample_laplacian(uv, lod, lod_sigma, max_level);
            }
            count += 1.0;
        }
    }
    color /= max(count, 1.0);

    textureStore(output_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(clamp(color, vec3(0.0), vec3(1.0)), 1.0));
}
