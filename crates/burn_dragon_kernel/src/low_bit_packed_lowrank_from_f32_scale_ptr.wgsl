enable packed_4x8_integer_dot_product;

@group(0) @binding(0)
var<storage, read> input_values: array<f32>;

@group(0) @binding(1)
var<storage, read> weight_packed: array<i32>;

@group(0) @binding(2)
var<storage, read_write> output_values: array<f32>;

@group(0) @binding(3)
var<storage, read> scale_values: array<f32>;

@group(0) @binding(4)
var<storage, read> meta_values: array<f32>;

fn meta_u(index: u32) -> u32 {
  return u32(meta_values[index]);
}

fn quantize_i8(value: f32, scale: f32, qmax: i32, positive_only: bool) -> i32 {
  var raw = value;
  if (positive_only && raw < 0.0) {
    raw = 0.0;
  }
  var shifted = raw / scale;
  if (shifted >= 0.0) {
    shifted = shifted + 0.5;
  } else {
    shifted = shifted - 0.5;
  }
  var quantized = i32(shifted);
  if (positive_only) {
    quantized = clamp(quantized, 0, qmax);
  } else {
    quantized = clamp(quantized, -qmax, qmax);
  }
  return quantized;
}

fn pack_i8x4(v0: i32, v1: i32, v2: i32, v3: i32) -> u32 {
  let b0 = bitcast<u32>(clamp(v0, -127, 127)) & 0xffu;
  let b1 = (bitcast<u32>(clamp(v1, -127, 127)) & 0xffu) << 8u;
  let b2 = (bitcast<u32>(clamp(v2, -127, 127)) & 0xffu) << 16u;
  let b3 = (bitcast<u32>(clamp(v3, -127, 127)) & 0xffu) << 24u;
  return b0 | b1 | b2 | b3;
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  let t = gid.y;
  let bh = gid.z;

  let batch = meta_u(0u);
  let input_heads = meta_u(1u);
  let heads = meta_u(2u);
  let time = meta_u(3u);
  let embd = meta_u(4u);
  let pack_len = meta_u(5u);
  let latent_out = meta_u(6u);
  let artifact_latent = meta_u(7u);
  let qmax = i32(meta_values[8u]);
  let positive_only = meta_u(9u) != 0u;
  let weight_scale = meta_values[10u];
  let activation_scale = max(scale_values[0u], 1.0e-8);

  if (l >= latent_out || t >= time || bh >= batch * heads) {
    return;
  }

  let h = bh % heads;
  let b = bh / heads;
  var input_head = h;
  if (input_heads == 1u) {
    input_head = 0u;
  }

  var acc: i32 = 0;
  let input_base = ((b * input_heads + input_head) * time + t) * embd;
  let weight_base = (h * pack_len) * artifact_latent + l;
  for (var p: u32 = 0u; p < pack_len; p += 1u) {
    let e = p * 4u;
    var v0: i32 = 0;
    if (e < embd) {
      v0 = quantize_i8(input_values[input_base + e], activation_scale, qmax, positive_only);
    }
    var v1: i32 = 0;
    if (e + 1u < embd) {
      v1 = quantize_i8(input_values[input_base + e + 1u], activation_scale, qmax, positive_only);
    }
    var v2: i32 = 0;
    if (e + 2u < embd) {
      v2 = quantize_i8(input_values[input_base + e + 2u], activation_scale, qmax, positive_only);
    }
    var v3: i32 = 0;
    if (e + 3u < embd) {
      v3 = quantize_i8(input_values[input_base + e + 3u], activation_scale, qmax, positive_only);
    }
    let packed_input = pack_i8x4(v0, v1, v2, v3);
    let packed_weight = bitcast<u32>(weight_packed[weight_base + p * artifact_latent]);
    acc = dot4I8Packed(packed_input, packed_weight) + acc;
  }

  let output_index = ((b * heads + h) * time + t) * latent_out + l;
  output_values[output_index] = f32(acc) * activation_scale * weight_scale;
}
