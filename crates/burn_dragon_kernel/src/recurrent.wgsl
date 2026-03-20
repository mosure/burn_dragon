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

const LATENT_TILE: u32 = 32u;

var<workgroup> query_tile: array<f32, LATENT_TILE>;

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
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let value_heads = to_u32(params[2]);
  let time = to_u32(params[3]);
  let latent = to_u32(params[4]);
  let embd = to_u32(params[5]);

  let e = gid.x;
  let h = gid.y;
  let b = gid.z;
  let lane = lid.x;

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
    var latent_base = 0u;
    while latent_base < latent {
      let tile_len = min(LATENT_TILE, latent - latent_base);
      if lane < tile_len {
        let query_index = idx_query(b, h, t, latent_base + lane, heads, time, latent);
        query_tile[lane] = query[query_index];
      }
      workgroupBarrier();

      var tile_offset = 0u;
      while tile_offset < tile_len {
        let l = latent_base + tile_offset;
        let rho_index = idx_rho(b, h, l, e, heads, latent, embd);
        let q = query_tile[tile_offset];
        let rho_prev = rho_state[rho_index];
        acc += rho_prev * q;
        rho_state[rho_index] = (rho_prev + q * value_t) * decay_value;
        tile_offset += 1u;
      }
      workgroupBarrier();
      latent_base += LATENT_TILE;
    }

    let out_index = idx_context(b, h, t, e, heads, time, embd);
    context[out_index] = acc;

    t += 1u;
  }
}
