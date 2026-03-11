// Generic local-grid rho attention kernel for 2D token lattices.
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
var<storage, read_write> rho_next: array<f32>;

@group(0) @binding(6)
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

fn idx_rho(
  b: u32,
  h: u32,
  t: u32,
  l: u32,
  e: u32,
  heads: u32,
  time: u32,
  latent: u32,
  embd: u32,
) -> u32 {
  return ((((b * heads + h) * time + t) * latent + l) * embd + e);
}

fn idx_context(b: u32, h: u32, t: u32, e: u32, heads: u32, time: u32, embd: u32) -> u32 {
  return (((b * heads + h) * time + t) * embd + e);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let value_heads = to_u32(params[2]);
  let patch_tokens = to_u32(params[3]);
  let latent = to_u32(params[4]);
  let embd = to_u32(params[5]);
  let grid_h = to_u32(params[6]);
  let grid_w = to_u32(params[7]);
  let radius = i32(to_u32(params[8]));
  let local_diagonals = to_u32(params[9]) == 1u;
  let local_self = to_u32(params[10]) == 1u;

  let e = gid.x;
  let dst_token = gid.y;
  let bh = gid.z;

  if bh >= batch * heads || dst_token >= patch_tokens || e >= embd {
    return;
  }

  let b = bh / heads;
  let h = bh % heads;
  let h_value = select(h, 0u, value_heads == 1u);
  let decay_value = decay[h];
  let ty = dst_token / grid_w;
  let tx = dst_token % grid_w;

  var acc = 0.0;
  var dy = -radius;
  loop {
    if dy > radius {
      break;
    }
    var dx = -radius;
    loop {
      if dx > radius {
        break;
      }
      if !(dy == 0 && dx == 0 && !local_self) && !(!local_diagonals && dy != 0 && dx != 0) {
        let sy = i32(ty) + dy;
        let sx = i32(tx) + dx;
        if sy >= 0 && sy < i32(grid_h) && sx >= 0 && sx < i32(grid_w) {
          let source = u32(sy) * grid_w + u32(sx);
          if source < patch_tokens {
            var l = 0u;
            while l < latent {
              let query_index = idx_query(b, h, dst_token, l, heads, patch_tokens, latent);
              let rho_index = idx_rho(b, h, source, l, e, heads, patch_tokens, latent, embd);
              acc += rho_state[rho_index] * query[query_index];
              l += 1u;
            }
          }
        }
      }
      dx += 1;
    }
    dy += 1;
  }

  let out_index = idx_context(b, h, dst_token, e, heads, patch_tokens, embd);
  context[out_index] = acc;

  let value_index = idx_value(b, h_value, dst_token, e, value_heads, patch_tokens, embd);
  let value_t = value[value_index];
  var l = 0u;
  while l < latent {
    let query_index = idx_query(b, h, dst_token, l, heads, patch_tokens, latent);
    let rho_index = idx_rho(b, h, dst_token, l, e, heads, patch_tokens, latent, embd);
    let rho_prev = rho_state[rho_index];
    rho_next[rho_index] = rho_prev * decay_value + query[query_index] * value_t;
    l += 1u;
  }
}
