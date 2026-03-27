enable packed_4x8_integer_dot_product;

@group(0) @binding(0)
var<storage, read> y_packed: array<i32>;

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
  let d = gid.x;
  let t = gid.y;
  let b = gid.z;

  let batch = meta_u(0u);
  let heads = meta_u(1u);
  let time = meta_u(2u);
  let latent_pack = meta_u(3u);
  let dim = meta_u(4u);
  let activation_scale = scale_values[0u];
  let weight_scale = meta_values[5u];

  if (d >= dim || t >= time || b >= batch) {
    return;
  }

  var acc: i32 = 0;
  for (var h: u32 = 0u; h < heads; h += 1u) {
    let y_base = ((b * heads + h) * time + t) * latent_pack;
    let weight_base = (h * latent_pack) * dim + d;
    for (var p: u32 = 0u; p < latent_pack; p += 1u) {
      let packed_y = bitcast<u32>(y_packed[y_base + p]);
      let packed_weight = bitcast<u32>(weight_packed[weight_base + p * dim]);
      acc = dot4I8Packed(packed_y, packed_weight) + acc;
    }
  }

  let output_index = (b * time + t) * dim + d;
  output_values[output_index] = f32(acc) * activation_scale * weight_scale;
}
