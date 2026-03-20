const WORKGROUP_SIZE_X: u32 = 64u;
const MAX_FUSED_TIME: u32 = 1024u;

@group(0) @binding(0)
var<storage, read_write> query: array<f32>;

@group(0) @binding(1)
var<storage, read_write> value: array<f32>;

@group(0) @binding(2)
var<storage, read_write> context: array<f32>;

@group(0) @binding(3)
var<storage, read_write> decay: array<f32>;

@group(0) @binding(4)
var<storage, read_write> params: array<f32>;

var<workgroup> row_scores: array<f32, MAX_FUSED_TIME>;

fn to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn idx_query(b: u32, h: u32, t: u32, l: u32, heads: u32, time: u32, latent: u32) -> u32 {
  return (((b * heads + h) * time + t) * latent + l);
}

fn idx_value(
  b: u32,
  vh: u32,
  t: u32,
  e: u32,
  value_heads: u32,
  time: u32,
  value_dim: u32,
) -> u32 {
  return (((b * value_heads + vh) * time + t) * value_dim + e);
}

fn idx_context(
  b: u32,
  h: u32,
  row: u32,
  e: u32,
  heads: u32,
  time: u32,
  value_dim: u32,
) -> u32 {
  return (((b * heads + h) * time + row) * value_dim + e);
}

@compute @workgroup_size(WORKGROUP_SIZE_X, 1, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let batch = to_u32(params[0]);
  let heads = to_u32(params[1]);
  let value_heads = to_u32(params[2]);
  let time = to_u32(params[3]);
  let latent = to_u32(params[4]);
  let value_dim = to_u32(params[5]);

  let h = wid.y;
  let batch_row = wid.z;
  let b = batch_row / time;
  let row = batch_row % time;
  let e = gid.x;
  let lane = lid.x;

  if b >= batch || h >= heads || row >= time || time > MAX_FUSED_TIME {
    return;
  }

  let decay_value = decay[h];
  var col = lane;
  while col < row {
    var dot = 0.0;
    var l = 0u;
    while l < latent {
      let q_row = query[idx_query(b, h, row, l, heads, time, latent)];
      let q_col = query[idx_query(b, h, col, l, heads, time, latent)];
      dot += q_row * q_col;
      l += 1u;
    }
    row_scores[col] = dot * pow(decay_value, f32(row - col));
    col += WORKGROUP_SIZE_X;
  }
  workgroupBarrier();

  if e >= value_dim {
    return;
  }

  let value_head = select(h, 0u, value_heads == 1u);
  var acc = 0.0;
  col = 0u;
  while col < row {
    acc += row_scores[col] * value[idx_value(b, value_head, col, e, value_heads, time, value_dim)];
    col += 1u;
  }

  context[idx_context(b, h, row, e, heads, time, value_dim)] = acc;
}
