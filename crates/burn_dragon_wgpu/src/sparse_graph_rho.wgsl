// Generic sparse-graph rho route kernel over CSR routing.
@group(0) @binding(0)
var<storage, read_write> query: array<f32>;

@group(0) @binding(1)
var<storage, read_write> value: array<f32>;

@group(0) @binding(2)
var<storage, read_write> rho_state: array<f32>;

@group(0) @binding(3)
var<storage, read_write> source_offsets: array<i32>;

@group(0) @binding(4)
var<storage, read_write> source_indices: array<i32>;

@group(0) @binding(5)
var<storage, read_write> incoming_offsets: array<i32>;

@group(0) @binding(6)
var<storage, read_write> incoming_indices: array<i32>;

@group(0) @binding(7)
var<storage, read_write> decay: array<f32>;

@group(0) @binding(8)
var<storage, read_write> context: array<f32>;

@group(0) @binding(9)
var<storage, read_write> rho_next: array<f32>;

@group(0) @binding(10)
var<storage, read_write> params: array<f32>;

fn f32_to_u32(v: f32) -> u32 {
  return u32(v + 0.5);
}

fn i32_to_u32(v: i32) -> u32 {
  return u32(max(v, 0));
}

fn idx_query(b: u32, s: u32, l: u32, source_count: u32, latent: u32) -> u32 {
  return ((b * source_count + s) * latent + l);
}

fn idx_value(b: u32, s: u32, e: u32, source_count: u32, embd: u32) -> u32 {
  return ((b * source_count + s) * embd + e);
}

fn idx_rho(
  b: u32,
  t: u32,
  l: u32,
  e: u32,
  target_count: u32,
  latent: u32,
  embd: u32,
) -> u32 {
  return (((b * target_count + t) * latent + l) * embd + e);
}

fn idx_context(b: u32, s: u32, e: u32, source_count: u32, embd: u32) -> u32 {
  return ((b * source_count + s) * embd + e);
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let batch = f32_to_u32(params[0]);
  let source_count = f32_to_u32(params[1]);
  let target_count = f32_to_u32(params[2]);
  let latent = f32_to_u32(params[3]);
  let embd = f32_to_u32(params[4]);

  let e = gid.x;
  let item = gid.y;
  let b = gid.z;

  if b >= batch || e >= embd {
    return;
  }

  let decay_value = decay[0];

  if item < source_count {
    let start = i32_to_u32(source_offsets[item]);
    let end = i32_to_u32(source_offsets[item + 1u]);
    var acc = 0.0;
    var edge = start;
    loop {
      if edge >= end {
        break;
      }
      let dst_index = i32_to_u32(source_indices[edge]);
      if dst_index < target_count {
        var l = 0u;
        while l < latent {
          let q_index = idx_query(b, item, l, source_count, latent);
          let rho_index = idx_rho(b, dst_index, l, e, target_count, latent, embd);
          acc += rho_state[rho_index] * query[q_index];
          l += 1u;
        }
      }
      edge += 1u;
    }
    let out_index = idx_context(b, item, e, source_count, embd);
    context[out_index] = acc;
  }

  if item < target_count {
    let start = i32_to_u32(incoming_offsets[item]);
    let end = i32_to_u32(incoming_offsets[item + 1u]);
    var l = 0u;
    while l < latent {
      let rho_index = idx_rho(b, item, l, e, target_count, latent, embd);
      var acc = rho_state[rho_index] * decay_value;
      var edge = start;
      loop {
        if edge >= end {
          break;
        }
        let src_index = i32_to_u32(incoming_indices[edge]);
        if src_index < source_count {
          let q_index = idx_query(b, src_index, l, source_count, latent);
          let v_index = idx_value(b, src_index, e, source_count, embd);
          acc += query[q_index] * value[v_index];
        }
        edge += 1u;
      }
      rho_next[rho_index] = acc;
      l += 1u;
    }
  }
}
