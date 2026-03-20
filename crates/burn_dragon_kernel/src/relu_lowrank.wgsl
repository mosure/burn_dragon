@group(0) @binding(0)
var<storage, read_write> input: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@group(0) @binding(3)
var<storage, read_write> params: array<f32>;

@group(0) @binding(4)
var<storage, read_write> sparse_mask: array<f32>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_input(
  b: u32,
  input_head: u32,
  t: u32,
  e: u32,
  input_heads: u32,
  time: u32,
  embd: u32,
) -> u32 {
  return (((b * input_heads + input_head) * time + t) * embd + e);
}

fn idx_weight(h: u32, e: u32, l: u32, embd: u32, latent: u32) -> u32 {
  return ((h * embd + e) * latent + l);
}

fn idx_output(b: u32, h: u32, t: u32, l: u32, heads: u32, time: u32, latent: u32) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = to_u32(params[0]);
  let input_heads = to_u32(params[1]);
  let heads = to_u32(params[2]);
  let time = to_u32(params[3]);
  let embd = to_u32(params[4]);
  let latent = to_u32(params[5]);
  let threshold = params[6];
  let has_mask = params[7] > 0.5;

  let l = gid.x;
  let t = gid.y;
  let bh = gid.z;
  if l >= latent || t >= time || bh >= batch * heads {
    return;
  }

  let h = bh % heads;
  let b = bh / heads;
  let input_head = select(h, 0u, input_heads == 1u);

  var sum = 0.0;
  var e = 0u;
  while e < embd {
    sum += input[idx_input(b, input_head, t, e, input_heads, time, embd)]
      * weight[idx_weight(h, e, l, embd, latent)];
    e += 1u;
  }

  sum -= threshold;
  if sum < 0.0 {
    sum = 0.0;
  }
  if has_mask {
    sum *= sparse_mask[l];
  }

  output[idx_output(b, h, t, l, heads, time, latent)] = sum;
}
