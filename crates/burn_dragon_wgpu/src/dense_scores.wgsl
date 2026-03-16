@group(0) @binding(0)
var<storage, read_write> query: array<f32>;

@group(0) @binding(1)
var<storage, read_write> scores: array<f32>;

@group(0) @binding(2)
var<storage, read_write> slopes: array<f32>;

@group(0) @binding(3)
var<storage, read_write> params: array<f32>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_query(b: u32, h: u32, t: u32, l: u32, heads: u32, time: u32, latent: u32) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

fn idx_score(b: u32, h: u32, row: u32, col: u32, heads: u32, time: u32) -> u32 {
  return (((b * heads + h) * time + row) * time + col);
}

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let time = to_u32(params[2]);
  let latent = to_u32(params[3]);
  let inv_scale = params[4];
  let eps = params[5];

  let row = gid.x;
  let h = gid.y;
  let b = gid.z;
  if row >= time || h >= heads || b >= batch {
    return;
  }

  let slope = slopes[h];
  var denom = eps;
  var col = 0u;
  while col < time {
    var sum = 0.0;
    var l = 0u;
    while l < latent {
      let q_row = query[idx_query(b, h, row, l, heads, time, latent)] * inv_scale;
      let q_col = query[idx_query(b, h, col, l, heads, time, latent)];
      sum += q_row * q_col;
      l += 1u;
    }
    sum += slope * (f32(col) - f32(row));
    scores[idx_score(b, h, row, col, heads, time)] = sum;
    denom += abs(sum);
    col += 1u;
  }

  col = 0u;
  while col < time {
    let index = idx_score(b, h, row, col, heads, time);
    scores[index] = scores[index] / denom;
    col += 1u;
  }
}
