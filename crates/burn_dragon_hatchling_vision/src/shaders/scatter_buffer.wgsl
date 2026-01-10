const META_BATCH: u32 = 0u;
const META_OUT_TOKENS: u32 = 1u;
const META_IN_TOKENS: u32 = 2u;
const META_DIM: u32 = 3u;

@group(0) @binding(0) var<storage, read_write> weights: array<f32>;
@group(0) @binding(1) var<storage, read_write> tokens: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;
@group(0) @binding(3) var<storage, read_write> metadata: array<f32>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = u32(metadata[META_BATCH]);
    let out_tokens = u32(metadata[META_OUT_TOKENS]);
    let in_tokens = u32(metadata[META_IN_TOKENS]);
    let dim = u32(metadata[META_DIM]);
    let o = gid.x;
    let d = gid.y;
    let b = gid.z;
    if o >= out_tokens || d >= dim || b >= batch {
        return;
    }

    let weight_base = (b * out_tokens + o) * in_tokens;
    let token_base = (b * in_tokens) * dim + d;
    var acc = 0.0;
    for (var i = 0u; i < in_tokens; i = i + 1u) {
        let weight = weights[weight_base + i];
        let token = tokens[token_base + i * dim];
        acc += weight * token;
    }
    let out_idx = (b * out_tokens + o) * dim + d;
    output[out_idx] = acc;
}
