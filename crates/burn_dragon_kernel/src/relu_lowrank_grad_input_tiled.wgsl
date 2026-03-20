@group(0) @binding(0)
var<storage, read_write> grad_projected: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> grad_input: array<f32>;

@group(0) @binding(3)
var<storage, read_write> params: array<f32>;

var<workgroup> grad_tile: array<f32, 64>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_grad_projected(
  b: u32,
  h: u32,
  t: u32,
  l: u32,
  heads: u32,
  time: u32,
  latent: u32,
) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

fn idx_weight(h: u32, e: u32, l: u32, embd: u32, latent: u32) -> u32 {
  return ((h * embd + e) * latent + l);
}

fn idx_grad_input(
  b: u32,
  h: u32,
  t: u32,
  e: u32,
  heads: u32,
  time: u32,
  embd: u32,
) -> u32 {
  return (((b * heads + h) * time + t) * embd + e);
}

@compute @workgroup_size(64, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[2]);
  let time = to_u32(params[3]);
  let embd = to_u32(params[4]);
  let latent = to_u32(params[5]);

  let e = gid.x;
  let t = gid.y;
  let bh = gid.z;
  if e >= embd || t >= time || bh >= batch * heads {
    return;
  }

  let h = bh % heads;
  let b = bh / heads;
  let lane = lid.x;

  var sum = 0.0;
  var latent_base = 0u;
  while latent_base < latent {
    let tile_index = latent_base + lane;
    if tile_index < latent {
      grad_tile[lane] = grad_projected[idx_grad_projected(b, h, t, tile_index, heads, time, latent)];
    } else {
      grad_tile[lane] = 0.0;
    }
    workgroupBarrier();

    let tile_len = min(64u, latent - latent_base);
    var k = 0u;
    while k < tile_len {
      sum += grad_tile[k] * weight[idx_weight(h, e, latent_base + k, embd, latent)];
      k += 1u;
    }
    workgroupBarrier();
    latent_base += 64u;
  }

  grad_input[idx_grad_input(b, h, t, e, heads, time, embd)] = sum;
}
