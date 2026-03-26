enable packed_4x8_integer_dot_product;

@group(0) @binding(0)
var<storage, read> y_codes: array<i32>;

@group(0) @binding(1)
var<storage, read> weight_codes: array<i32>;

@group(0) @binding(2)
var<storage, read_write> output_values: array<f32>;

@group(0) @binding(3)
var<storage, read> meta_values: array<f32>;

fn meta_u(index: u32) -> u32 {
  return u32(meta_values[index]);
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
  let d = gid.x;
  let t = gid.y;
  let b = gid.z;

  let batch = meta_u(0u);
  let heads = meta_u(1u);
  let time = meta_u(2u);
  let latent = meta_u(3u);
  let artifact_latent_per_head = meta_u(4u);
  let dim = meta_u(5u);
  let activation_scale = meta_values[6u];
  let weight_scale = meta_values[7u];

  if (d >= dim || t >= time || b >= batch) {
    return;
  }

  var acc: i32 = 0;
  var h: u32 = 0u;
  while (h < heads) {
    let input_base = ((b * heads + h) * time + t) * latent;
    let weight_base = h * artifact_latent_per_head * dim + d;
    var l: u32 = 0u;
    while (l + 4u <= latent) {
      let packed_input = pack_i8x4(
        y_codes[input_base + l],
        y_codes[input_base + l + 1u],
        y_codes[input_base + l + 2u],
        y_codes[input_base + l + 3u],
      );
      let packed_weight = pack_i8x4(
        weight_codes[weight_base + l * dim],
        weight_codes[weight_base + (l + 1u) * dim],
        weight_codes[weight_base + (l + 2u) * dim],
        weight_codes[weight_base + (l + 3u) * dim],
      );
      acc = dot4I8Packed(packed_input, packed_weight) + acc;
      l += 4u;
    }
    while (l < latent) {
      acc += y_codes[input_base + l] * weight_codes[weight_base + l * dim];
      l += 1u;
    }
    h += 1u;
  }

  output_values[(b * time + t) * dim + d] = f32(acc) * activation_scale * weight_scale;
}
