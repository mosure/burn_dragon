enable packed_4x8_integer_dot_product;

@group(0) @binding(0)
var<storage, read> input_codes: array<i32>;

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
  let l = gid.x;
  let t = gid.y;
  let bh = gid.z;

  let batch = meta_u(0u);
  let input_heads = meta_u(1u);
  let heads = meta_u(2u);
  let time = meta_u(3u);
  let embd = meta_u(4u);
  let latent_out = meta_u(5u);
  let artifact_latent = meta_u(6u);
  let activation_scale = meta_values[7u];
  let weight_scale = meta_values[8u];

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
  var e: u32 = 0u;
  let input_base = ((b * input_heads + input_head) * time + t) * embd;
  let weight_base = h * embd * artifact_latent + l;
  while (e + 4u <= embd) {
    let packed_input = pack_i8x4(
      input_codes[input_base + e],
      input_codes[input_base + e + 1u],
      input_codes[input_base + e + 2u],
      input_codes[input_base + e + 3u],
    );
    let packed_weight = pack_i8x4(
      weight_codes[weight_base + e * artifact_latent],
      weight_codes[weight_base + (e + 1u) * artifact_latent],
      weight_codes[weight_base + (e + 2u) * artifact_latent],
      weight_codes[weight_base + (e + 3u) * artifact_latent],
    );
    acc = dot4I8Packed(packed_input, packed_weight) + acc;
    e += 4u;
  }

  while (e < embd) {
    acc += input_codes[input_base + e] * weight_codes[weight_base + e * artifact_latent];
    e += 1u;
  }

  let output_index = ((b * heads + h) * time + t) * latent_out + l;
  output_values[output_index] = f32(acc) * activation_scale * weight_scale;
}
