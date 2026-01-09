struct CopyParams {
  stride_h: u32,
  stride_w: u32,
  stride_c: u32,
  _pad: u32,
};

@group(0) @binding(0)
var<storage, read> src: array<f32>;

@group(0) @binding(1)
var dst: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var<uniform> params: CopyParams;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let dims = textureDimensions(dst);
  if (gid.x >= dims.x || gid.y >= dims.y) { return; }
  let base = gid.y * params.stride_h + gid.x * params.stride_w;
  let r = src[base];
  let g = src[base + params.stride_c];
  let b = src[base + params.stride_c * 2u];
  let a = src[base + params.stride_c * 3u];
  textureStore(dst, vec2<i32>(i32(gid.x), i32(gid.y)), vec4<f32>(r, g, b, a));
}
