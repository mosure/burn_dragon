struct PyramidParams {
    src_dst: vec4<u32>,
};

@group(0) @binding(0) var src_fine: texture_2d<f32>;
@group(0) @binding(1) var src_coarse: texture_2d<f32>;
@group(0) @binding(2) var src_sampler: sampler;
@group(0) @binding(3) var dst_tex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(4) var<uniform> params: PyramidParams;

fn clamp_coord(coord: vec2<i32>, size: vec2<i32>) -> vec2<i32> {
    return clamp(coord, vec2<i32>(0, 0), size - vec2<i32>(1, 1));
}

fn read_tex(tex: texture_2d<f32>, coord: vec2<i32>, size: vec2<i32>) -> vec3<f32> {
    let clamped = clamp_coord(coord, size);
    return textureLoad(tex, clamped, 0).xyz;
}

@compute @workgroup_size(8, 8, 1)
fn init_base(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let dst_size = params.src_dst.zw;
    if x >= dst_size.x || y >= dst_size.y {
        return;
    }
    let color = textureLoad(src_fine, vec2<i32>(i32(x), i32(y)), 0).xyz;
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn downsample(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let src_size = params.src_dst.xy;
    let dst_size = params.src_dst.zw;
    if x >= dst_size.x || y >= dst_size.y {
        return;
    }
    let src_size_i = vec2<i32>(i32(src_size.x), i32(src_size.y));
    let src_coord = vec2<i32>(i32(x) * 2, i32(y) * 2);
    var color = vec3<f32>(0.0);
    for (var ky: i32 = -2; ky <= 2; ky = ky + 1) {
        let wy = select(6.0, 4.0, ky == -1 || ky == 1);
        let wy2 = select(wy, 1.0, ky == -2 || ky == 2);
        for (var kx: i32 = -2; kx <= 2; kx = kx + 1) {
            let wx = select(6.0, 4.0, kx == -1 || kx == 1);
            let wx2 = select(wx, 1.0, kx == -2 || kx == 2);
            let weight = wx2 * wy2;
            let sample = read_tex(src_fine, src_coord + vec2<i32>(kx, ky), src_size_i);
            color += sample * weight;
        }
    }
    color *= 1.0 / 256.0;
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}

@compute @workgroup_size(8, 8, 1)
fn residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    let dst_size = params.src_dst.zw;
    if x >= dst_size.x || y >= dst_size.y {
        return;
    }
    let uv = (vec2<f32>(f32(x) + 0.5, f32(y) + 0.5)) / vec2<f32>(dst_size);
    let fine = textureLoad(src_fine, vec2<i32>(i32(x), i32(y)), 0).xyz;
    let coarse = textureSampleLevel(src_coarse, src_sampler, uv, 0.0).xyz;
    let color = fine - coarse;
    textureStore(dst_tex, vec2<i32>(i32(x), i32(y)), vec4<f32>(color, 1.0));
}
