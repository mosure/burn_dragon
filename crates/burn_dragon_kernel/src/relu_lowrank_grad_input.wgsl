@group(0) @binding(0)
var<storage, read_write> grad_projected: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> grad_input: array<f32>;

@group(0) @binding(3)
var<storage, read_write> params: array<f32>;

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
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
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

  var sum0 = 0.0;
  var sum1 = 0.0;
  var sum2 = 0.0;
  var sum3 = 0.0;
  var l = 0u;
  while l + 3u < latent {
    let l0 = l;
    let l1 = l + 1u;
    let l2 = l + 2u;
    let l3 = l + 3u;
    sum0 += grad_projected[idx_grad_projected(b, h, t, l0, heads, time, latent)]
      * weight[idx_weight(h, e, l0, embd, latent)];
    sum1 += grad_projected[idx_grad_projected(b, h, t, l1, heads, time, latent)]
      * weight[idx_weight(h, e, l1, embd, latent)];
    sum2 += grad_projected[idx_grad_projected(b, h, t, l2, heads, time, latent)]
      * weight[idx_weight(h, e, l2, embd, latent)];
    sum3 += grad_projected[idx_grad_projected(b, h, t, l3, heads, time, latent)]
      * weight[idx_weight(h, e, l3, embd, latent)];
    l += 4u;
  }

  var sum = (sum0 + sum1) + (sum2 + sum3);
  while l < latent {
    sum += grad_projected[idx_grad_projected(b, h, t, l, heads, time, latent)]
      * weight[idx_weight(h, e, l, embd, latent)];
    l += 1u;
  }

  grad_input[idx_grad_input(b, h, t, e, heads, time, embd)] = sum;
}
