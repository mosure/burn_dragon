enable packed_4x8_integer_dot_product;

@group(0) @binding(0)
var<storage, read> input_packed: array<i32>;

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

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  let t = gid.y;
  let bh = gid.z;

  let batch = meta_u(0u);
  let input_heads = meta_u(1u);
  let heads = meta_u(2u);
  let time = meta_u(3u);
  let pack_len = meta_u(4u);
  let latent_out = meta_u(5u);
  let artifact_latent = meta_u(6u);
  let activation_scale = scale_values[0u];
  let weight_scale = meta_values[7u];

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
  let input_base = ((b * input_heads + input_head) * time + t) * pack_len;
  let weight_base = (h * pack_len) * artifact_latent + l;
  for (var p: u32 = 0u; p < pack_len; p += 1u) {
    let packed_input = bitcast<u32>(input_packed[input_base + p]);
    let packed_weight = bitcast<u32>(weight_packed[weight_base + p * artifact_latent]);
    acc = dot4I8Packed(packed_input, packed_weight) + acc;
  }

  let output_index = ((b * heads + h) * time + t) * latent_out + l;
  output_values[output_index] = f32(acc) * activation_scale * weight_scale;
}
