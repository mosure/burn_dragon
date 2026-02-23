@group(0) @binding(0)
var<storage, read_write> query: array<f32>;

@group(0) @binding(1)
var<storage, read_write> value: array<f32>;

@group(0) @binding(2)
var<storage, read_write> rho_state: array<f32>;

@group(0) @binding(3)
var<storage, read_write> decay: array<f32>;

@group(0) @binding(4)
var<storage, read_write> context: array<f32>;

@group(0) @binding(5)
var<storage, read_write> params: array<f32>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_query(b: u32, h: u32, t: u32, l: u32, heads: u32, time: u32, latent: u32) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

fn idx_value(
  b: u32,
  h: u32,
  t: u32,
  e: u32,
  value_heads: u32,
  time: u32,
  embd: u32,
) -> u32 {
  return (((b * value_heads + h) * time + t) * embd + e);
}

fn idx_rho(b: u32, h: u32, l: u32, e: u32, heads: u32, latent: u32, embd: u32) -> u32 {
  return (((b * heads + h) * latent + l) * embd + e);
}

fn idx_context(b: u32, h: u32, t: u32, e: u32, heads: u32, time: u32, embd: u32) -> u32 {
  return (((b * heads + h) * time + t) * embd + e);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let value_heads = to_u32(params[2]);
  let time = to_u32(params[3]);
  let latent = to_u32(params[4]);
  let embd = to_u32(params[5]);

  let e = gid.x;
  let h = gid.y;
  let b = gid.z;

  if b >= batch || h >= heads || e >= embd {
    return;
  }

  let h_value = select(h, 0u, value_heads == 1u);
  let decay_value = decay[h];

  var t = 0u;
  while t < time {
    let value_index = idx_value(b, h_value, t, e, value_heads, time, embd);
    let value_t = value[value_index];

    var acc = 0.0;
    var l = 0u;
    while l < latent {
      let query_index = idx_query(b, h, t, l, heads, time, latent);
      let rho_index = idx_rho(b, h, l, e, heads, latent, embd);
      let q = query[query_index];
      let rho_prev = rho_state[rho_index];
      acc += rho_prev * q;
      rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
      l += 1u;
    }

    let out_index = idx_context(b, h, t, e, heads, time, embd);
    context[out_index] = acc;

    t += 1u;
  }
}
