use super::*;

#[cfg(feature = "cuda")]
const LOW_BIT_CUDA_RAW_DOT_FROM_CODES_SRC: &str = r#"
extern "C" __global__ void packed_lowrank_dp4a(
    const int* input_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int pack_len,
    int latent_out,
    float input_scale,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * pack_len;
    int weight_base = (head_idx * pack_len) * latent_out + latent_idx;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        acc = __dp4a(input_packed[input_base + p], weight_packed[weight_base + p * latent_out], acc);
    }
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_scale_ptr(
    const int* input_packed,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int pack_len,
    int latent_out,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * pack_len;
    int weight_base = (head_idx * pack_len) * latent_out + latent_idx;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        acc = __dp4a(input_packed[input_base + p], weight_packed[weight_base + p * latent_out], acc);
    }
    float input_scale = input_scale_ptr[0];
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_from_codes(
    const int* input_codes,
    const int* weight_packed,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int embd,
    int pack_len,
    int latent_out,
    float input_scale,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * embd;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        int e = p * 4;
        int v0 = (e < embd) ? input_codes[input_base + e] : 0;
        int v1 = (e + 1 < embd) ? input_codes[input_base + e + 1] : 0;
        int v2 = (e + 2 < embd) ? input_codes[input_base + e + 2] : 0;
        int v3 = (e + 3 < embd) ? input_codes[input_base + e + 3] : 0;
        int packed_input =
            (v0 & 0xff) |
            ((v1 & 0xff) << 8) |
            ((v2 & 0xff) << 16) |
            ((v3 & 0xff) << 24);
        int packed_weight = weight_packed[(head_idx * pack_len + p) * latent_out + latent_idx];
        acc = __dp4a(packed_input, packed_weight, acc);
    }
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_from_codes_scale_ptr(
    const int* input_codes,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int embd,
    int pack_len,
    int latent_out,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * embd;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        int e = p * 4;
        int v0 = (e < embd) ? input_codes[input_base + e] : 0;
        int v1 = (e + 1 < embd) ? input_codes[input_base + e + 1] : 0;
        int v2 = (e + 2 < embd) ? input_codes[input_base + e + 2] : 0;
        int v3 = (e + 3 < embd) ? input_codes[input_base + e + 3] : 0;
        int packed_input =
            (v0 & 0xff) |
            ((v1 & 0xff) << 8) |
            ((v2 & 0xff) << 16) |
            ((v3 & 0xff) << 24);
        int packed_weight = weight_packed[(head_idx * pack_len + p) * latent_out + latent_idx];
        acc = __dp4a(packed_input, packed_weight, acc);
    }
    float input_scale = input_scale_ptr[0];
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_lowrank_dp4a_from_f32_scale_ptr(
    const float* input,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int tokens,
    int embd,
    int pack_len,
    int latent_out,
    int qmax,
    int positive_only,
    float weight_scale
) {
    int latent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_head = blockIdx.z;
    if (latent_idx >= latent_out) return;
    int batch_idx = batch_head / heads;
    int head_idx = batch_head % heads;
    int input_head_idx = (input_heads == 1) ? 0 : head_idx;
    int input_base = ((batch_idx * input_heads + input_head_idx) * tokens + token_idx) * embd;
    float input_scale = input_scale_ptr[0];
    float inv_scale = input_scale > 0.0f ? (1.0f / input_scale) : 0.0f;
    int acc = 0;
    #pragma unroll 4
    for (int p = 0; p < pack_len; ++p) {
        int e = p * 4;
        int v0 = 0;
        int v1 = 0;
        int v2 = 0;
        int v3 = 0;
        if (e < embd) {
            float raw = input[input_base + e];
            if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
            v0 = (int)roundf(raw * inv_scale);
            if (positive_only) {
                if (v0 < 0) v0 = 0;
            } else if (v0 < -qmax) {
                v0 = -qmax;
            }
            if (v0 > qmax) v0 = qmax;
        }
        if (e + 1 < embd) {
            float raw = input[input_base + e + 1];
            if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
            v1 = (int)roundf(raw * inv_scale);
            if (positive_only) {
                if (v1 < 0) v1 = 0;
            } else if (v1 < -qmax) {
                v1 = -qmax;
            }
            if (v1 > qmax) v1 = qmax;
        }
        if (e + 2 < embd) {
            float raw = input[input_base + e + 2];
            if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
            v2 = (int)roundf(raw * inv_scale);
            if (positive_only) {
                if (v2 < 0) v2 = 0;
            } else if (v2 < -qmax) {
                v2 = -qmax;
            }
            if (v2 > qmax) v2 = qmax;
        }
        if (e + 3 < embd) {
            float raw = input[input_base + e + 3];
            if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
            v3 = (int)roundf(raw * inv_scale);
            if (positive_only) {
                if (v3 < 0) v3 = 0;
            } else if (v3 < -qmax) {
                v3 = -qmax;
            }
            if (v3 > qmax) v3 = qmax;
        }
        int packed_input =
            (v0 & 0xff) |
            ((v1 & 0xff) << 8) |
            ((v2 & 0xff) << 16) |
            ((v3 & 0xff) << 24);
        int packed_weight = weight_packed[(head_idx * pack_len + p) * latent_out + latent_idx];
        acc = __dp4a(packed_input, packed_weight, acc);
    }
    output[((batch_idx * heads + head_idx) * tokens + token_idx) * latent_out + latent_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a(
    const int* y_packed,
    const int* weight_packed,
    float* output,
    int batch,
    int heads,
    int tokens,
    int pack_len,
    int dim,
    float input_scale,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * pack_len;
        int weight_base = (head_idx * pack_len) * dim + dim_idx;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            acc = __dp4a(y_packed[input_base + p], weight_packed[weight_base + p * dim], acc);
        }
    }
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a_scale_ptr(
    const int* y_packed,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int heads,
    int tokens,
    int pack_len,
    int dim,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * pack_len;
        int weight_base = (head_idx * pack_len) * dim + dim_idx;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            acc = __dp4a(y_packed[input_base + p], weight_packed[weight_base + p * dim], acc);
        }
    }
    float input_scale = input_scale_ptr[0];
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a_from_codes(
    const int* y_codes,
    const int* weight_packed,
    float* output,
    int batch,
    int heads,
    int tokens,
    int latent,
    int pack_len,
    int dim,
    float input_scale,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * latent;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            int l = p * 4;
            int v0 = (l < latent) ? y_codes[input_base + l] : 0;
            int v1 = (l + 1 < latent) ? y_codes[input_base + l + 1] : 0;
            int v2 = (l + 2 < latent) ? y_codes[input_base + l + 2] : 0;
            int v3 = (l + 3 < latent) ? y_codes[input_base + l + 3] : 0;
            int packed_input =
                (v0 & 0xff) |
                ((v1 & 0xff) << 8) |
                ((v2 & 0xff) << 16) |
                ((v3 & 0xff) << 24);
            int packed_weight = weight_packed[(head_idx * pack_len + p) * dim + dim_idx];
            acc = __dp4a(packed_input, packed_weight, acc);
        }
    }
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a_from_codes_scale_ptr(
    const int* y_codes,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int heads,
    int tokens,
    int latent,
    int pack_len,
    int dim,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * latent;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            int l = p * 4;
            int v0 = (l < latent) ? y_codes[input_base + l] : 0;
            int v1 = (l + 1 < latent) ? y_codes[input_base + l + 1] : 0;
            int v2 = (l + 2 < latent) ? y_codes[input_base + l + 2] : 0;
            int v3 = (l + 3 < latent) ? y_codes[input_base + l + 3] : 0;
            int packed_input =
                (v0 & 0xff) |
                ((v1 & 0xff) << 8) |
                ((v2 & 0xff) << 16) |
                ((v3 & 0xff) << 24);
            int packed_weight = weight_packed[(head_idx * pack_len + p) * dim + dim_idx];
            acc = __dp4a(packed_input, packed_weight, acc);
        }
    }
    float input_scale = input_scale_ptr[0];
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_dp4a_from_f32_scale_ptr(
    const float* y,
    const int* weight_packed,
    const float* input_scale_ptr,
    float* output,
    int batch,
    int heads,
    int tokens,
    int latent,
    int pack_len,
    int dim,
    int qmax,
    int positive_only,
    float weight_scale
) {
    int dim_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int token_idx = blockIdx.y;
    int batch_idx = blockIdx.z;
    if (dim_idx >= dim) return;
    float input_scale = input_scale_ptr[0];
    float inv_scale = input_scale > 0.0f ? (1.0f / input_scale) : 0.0f;
    int acc = 0;
    for (int head_idx = 0; head_idx < heads; ++head_idx) {
        int input_base = ((batch_idx * heads + head_idx) * tokens + token_idx) * latent;
        #pragma unroll 4
        for (int p = 0; p < pack_len; ++p) {
            int l = p * 4;
            int v0 = 0;
            int v1 = 0;
            int v2 = 0;
            int v3 = 0;
            if (l < latent) {
                float raw = y[input_base + l];
                if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
                v0 = (int)roundf(raw * inv_scale);
                if (positive_only) {
                    if (v0 < 0) v0 = 0;
                } else if (v0 < -qmax) {
                    v0 = -qmax;
                }
                if (v0 > qmax) v0 = qmax;
            }
            if (l + 1 < latent) {
                float raw = y[input_base + l + 1];
                if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
                v1 = (int)roundf(raw * inv_scale);
                if (positive_only) {
                    if (v1 < 0) v1 = 0;
                } else if (v1 < -qmax) {
                    v1 = -qmax;
                }
                if (v1 > qmax) v1 = qmax;
            }
            if (l + 2 < latent) {
                float raw = y[input_base + l + 2];
                if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
                v2 = (int)roundf(raw * inv_scale);
                if (positive_only) {
                    if (v2 < 0) v2 = 0;
                } else if (v2 < -qmax) {
                    v2 = -qmax;
                }
                if (v2 > qmax) v2 = qmax;
            }
            if (l + 3 < latent) {
                float raw = y[input_base + l + 3];
                if (positive_only) raw = raw < 0.0f ? 0.0f : raw;
                v3 = (int)roundf(raw * inv_scale);
                if (positive_only) {
                    if (v3 < 0) v3 = 0;
                } else if (v3 < -qmax) {
                    v3 = -qmax;
                }
                if (v3 > qmax) v3 = qmax;
            }
            int packed_input =
                (v0 & 0xff) |
                ((v1 & 0xff) << 8) |
                ((v2 & 0xff) << 16) |
                ((v3 & 0xff) << 24);
            int packed_weight = weight_packed[(head_idx * pack_len + p) * dim + dim_idx];
            acc = __dp4a(packed_input, packed_weight, acc);
        }
    }
    output[(batch_idx * tokens + token_idx) * dim + dim_idx] =
        ((float)acc) * input_scale * weight_scale;
}

extern "C" __global__ void quantize_pack_i8x4_from_f32_scale_ptr(
    const float* input,
    const float* input_scale_ptr,
    int* output_packed,
    int outer,
    int inner,
    int pack_len,
    int qmax,
    int positive_only
) {
    int packed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (packed_idx >= outer * pack_len) {
        return;
    }
    int outer_idx = packed_idx / pack_len;
    int pack_offset = packed_idx % pack_len;
    int base = outer_idx * inner;
    int value_offset = pack_offset * 4;
    float scale = input_scale_ptr[0];
    float inv_scale = scale > 0.0f ? (1.0f / scale) : 0.0f;
    int v[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        int idx = value_offset + lane;
        if (idx < inner) {
            float raw = input[base + idx];
            if (positive_only) {
                raw = raw < 0.0f ? 0.0f : raw;
                v[lane] = __float2int_rn(raw * inv_scale);
                if (v[lane] < 0) v[lane] = 0;
                if (v[lane] > qmax) v[lane] = qmax;
            } else {
                v[lane] = __float2int_rn(raw * inv_scale);
                if (v[lane] < -qmax) v[lane] = -qmax;
                if (v[lane] > qmax) v[lane] = qmax;
            }
        }
    }
    output_packed[packed_idx] =
        (v[0] & 0xff) |
        ((v[1] & 0xff) << 8) |
        ((v[2] & 0xff) << 16) |
        ((v[3] & 0xff) << 24);
}

extern "C" __global__ void pack_i8x4_from_i32_codes(
    const int* input_codes,
    int* output_packed,
    int outer,
    int inner,
    int pack_len
) {
    int packed_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (packed_idx >= outer * pack_len) {
        return;
    }
    int outer_idx = packed_idx / pack_len;
    int pack_offset = packed_idx % pack_len;
    int base = outer_idx * inner;
    int value_offset = pack_offset * 4;
    int v[4] = {0, 0, 0, 0};
    #pragma unroll
    for (int lane = 0; lane < 4; ++lane) {
        int idx = value_offset + lane;
        if (idx < inner) {
            int raw = input_codes[base + idx];
            if (raw < -127) raw = -127;
            if (raw > 127) raw = 127;
            v[lane] = raw;
        }
    }
    output_packed[packed_idx] =
        (v[0] & 0xff) |
        ((v[1] & 0xff) << 8) |
        ((v[2] & 0xff) << 16) |
        ((v[3] & 0xff) << 24);
}

extern "C" __global__ void packed_lowrank_grad_input_raw(
    const float* grad,
    const int* weight_codes,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int time,
    int embd,
    int latent,
    float weight_scale
) {
    int e = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int bih = blockIdx.z;
    if (e >= embd || t >= time || bih >= batch * input_heads) {
        return;
    }
    int input_head = bih % input_heads;
    int b = bih / input_heads;
    float acc = 0.0f;
    if (input_heads == 1) {
        for (int h = 0; h < heads; ++h) {
            int grad_base = ((b * heads + h) * time + t) * latent;
            int weight_base = (h * embd + e) * latent;
            #pragma unroll 4
            for (int l = 0; l < latent; ++l) {
                acc += grad[grad_base + l] * (float)weight_codes[weight_base + l];
            }
        }
    } else {
        int h = input_head;
        int grad_base = ((b * heads + h) * time + t) * latent;
        int weight_base = (h * embd + e) * latent;
        #pragma unroll 4
        for (int l = 0; l < latent; ++l) {
            acc += grad[grad_base + l] * (float)weight_codes[weight_base + l];
        }
    }
    output[((b * input_heads + input_head) * time + t) * embd + e] = acc * weight_scale;
}

extern "C" __global__ void packed_lowrank_grad_weight_raw(
    const int* input_codes,
    const float* grad,
    float* output,
    int batch,
    int input_heads,
    int heads,
    int time,
    int embd,
    int latent,
    float activation_scale
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int e = blockIdx.y;
    int h = blockIdx.z;
    if (l >= latent || e >= embd || h >= heads) {
        return;
    }
    int input_head = input_heads == 1 ? 0 : h;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < time; ++t) {
            int input_index = ((b * input_heads + input_head) * time + t) * embd + e;
            int grad_index = ((b * heads + h) * time + t) * latent + l;
            acc += (float)input_codes[input_index] * grad[grad_index];
        }
    }
    output[(h * embd + e) * latent + l] = acc * activation_scale;
}

extern "C" __global__ void packed_decoder_tail_grad_input_raw(
    const float* grad,
    const int* weight_codes,
    float* output,
    int batch,
    int heads,
    int time,
    int latent,
    int dim,
    float weight_scale
) {
    int l = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    int bh = blockIdx.z;
    if (l >= latent || t >= time || bh >= batch * heads) {
        return;
    }
    int h = bh % heads;
    int b = bh / heads;
    int weight_row_base = (h * latent + l) * dim;
    int grad_base = (b * time + t) * dim;
    float acc = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < dim; ++d) {
        acc += grad[grad_base + d] * (float)weight_codes[weight_row_base + d];
    }
    output[((b * heads + h) * time + t) * latent + l] = acc * weight_scale;
}

extern "C" __global__ void packed_decoder_tail_grad_weight_raw(
    const int* y_codes,
    const float* grad,
    float* output,
    int batch,
    int heads,
    int time,
    int latent,
    int dim,
    float activation_scale
) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int hl = blockIdx.y;
    if (d >= dim || hl >= heads * latent) {
        return;
    }
    int h = hl / latent;
    int l = hl % latent;
    float acc = 0.0f;
    for (int b = 0; b < batch; ++b) {
        for (int t = 0; t < time; ++t) {
            int y_index = ((b * heads + h) * time + t) * latent + l;
            int grad_index = (b * time + t) * dim + d;
            acc += (float)y_codes[y_index] * grad[grad_index];
        }
    }
    output[hl * dim + d] = acc * activation_scale;
}
"#;

#[cfg(feature = "cuda")]
#[derive(Clone)]
struct RawCudaPackedDotKernels {
    #[allow(dead_code)]
    ctx: std::sync::Arc<CudaContext>,
    stream: std::sync::Arc<CudaStream>,
    lowrank: CudaFunction,
    lowrank_scale_ptr: CudaFunction,
    lowrank_from_codes: CudaFunction,
    lowrank_from_codes_scale_ptr: CudaFunction,
    decoder: CudaFunction,
    decoder_scale_ptr: CudaFunction,
    decoder_from_codes: CudaFunction,
    decoder_from_codes_scale_ptr: CudaFunction,
    quantize_pack_i8x4_from_f32_scale_ptr: CudaFunction,
    pack_i8x4_from_i32_codes: CudaFunction,
    lowrank_grad_input: CudaFunction,
    lowrank_grad_weight: CudaFunction,
    decoder_grad_input: CudaFunction,
    decoder_grad_weight: CudaFunction,
}

#[cfg(feature = "cuda")]
static RAW_CUDA_PACKED_DOT_KERNELS: OnceLock<Mutex<HashMap<usize, RawCudaPackedDotKernels>>> =
    OnceLock::new();

pub fn supports_packed_low_bit_device_backend<B: BackendTrait>() -> bool {
    let _ = core::any::type_name::<B>();
    true
}

pub fn supports_packed_rho_int8_block_device_backend<B: BackendTrait>() -> bool {
    supports_packed_low_bit_device_backend::<B>()
}

fn pack_i8x4_host(v0: i8, v1: i8, v2: i8, v3: i8) -> i32 {
    let to_byte = |value: i8| (i32::from(value).clamp(-127, 127) & 0xff) as u32;
    (to_byte(v0) | (to_byte(v1) << 8) | (to_byte(v2) << 16) | (to_byte(v3) << 24)) as i32
}

pub fn pack_lowrank_weight_codes_i8x4(
    codes: &[i8],
    heads: usize,
    embd: usize,
    latent: usize,
) -> Vec<i32> {
    let pack_len = embd.div_ceil(4);
    let mut packed = vec![0i32; heads * pack_len * latent];
    for h in 0..heads {
        for p in 0..pack_len {
            for l in 0..latent {
                let e = p * 4;
                let v0 = if e < embd {
                    codes[(h * embd + e) * latent + l]
                } else {
                    0
                };
                let v1 = if e + 1 < embd {
                    codes[(h * embd + e + 1) * latent + l]
                } else {
                    0
                };
                let v2 = if e + 2 < embd {
                    codes[(h * embd + e + 2) * latent + l]
                } else {
                    0
                };
                let v3 = if e + 3 < embd {
                    codes[(h * embd + e + 3) * latent + l]
                } else {
                    0
                };
                packed[(h * pack_len + p) * latent + l] = pack_i8x4_host(v0, v1, v2, v3);
            }
        }
    }
    packed
}

pub fn pack_decoder_weight_codes_i8x4(
    codes: &[i8],
    heads: usize,
    latent_per_head: usize,
    dim: usize,
) -> Vec<i32> {
    let pack_len = latent_per_head.div_ceil(4);
    let mut packed = vec![0i32; heads * pack_len * dim];
    for h in 0..heads {
        for p in 0..pack_len {
            for d in 0..dim {
                let l = p * 4;
                let row0 = h * latent_per_head + l;
                let row1 = h * latent_per_head + l + 1;
                let row2 = h * latent_per_head + l + 2;
                let row3 = h * latent_per_head + l + 3;
                let v0 = if l < latent_per_head {
                    codes[row0 * dim + d]
                } else {
                    0
                };
                let v1 = if l + 1 < latent_per_head {
                    codes[row1 * dim + d]
                } else {
                    0
                };
                let v2 = if l + 2 < latent_per_head {
                    codes[row2 * dim + d]
                } else {
                    0
                };
                let v3 = if l + 3 < latent_per_head {
                    codes[row3 * dim + d]
                } else {
                    0
                };
                packed[(h * pack_len + p) * dim + d] = pack_i8x4_host(v0, v1, v2, v3);
            }
        }
    }
    packed
}

pub fn pack_lowrank_input_codes_i8x4(
    codes: &[i8],
    batch: usize,
    input_heads: usize,
    tokens: usize,
    embd: usize,
) -> Vec<i32> {
    let pack_len = embd.div_ceil(4);
    let mut packed = vec![0i32; batch * input_heads * tokens * pack_len];
    for b in 0..batch {
        for h in 0..input_heads {
            for t in 0..tokens {
                let base = ((b * input_heads + h) * tokens + t) * embd;
                let out_base = ((b * input_heads + h) * tokens + t) * pack_len;
                for p in 0..pack_len {
                    let e = p * 4;
                    let v0 = *codes.get(base + e).unwrap_or(&0);
                    let v1 = *codes.get(base + e + 1).unwrap_or(&0);
                    let v2 = *codes.get(base + e + 2).unwrap_or(&0);
                    let v3 = *codes.get(base + e + 3).unwrap_or(&0);
                    packed[out_base + p] = pack_i8x4_host(v0, v1, v2, v3);
                }
            }
        }
    }
    packed
}

pub fn pack_decoder_input_codes_i8x4(
    codes: &[i8],
    batch: usize,
    heads: usize,
    tokens: usize,
    latent: usize,
) -> Vec<i32> {
    let pack_len = latent.div_ceil(4);
    let mut packed = vec![0i32; batch * heads * tokens * pack_len];
    for b in 0..batch {
        for h in 0..heads {
            for t in 0..tokens {
                let base = ((b * heads + h) * tokens + t) * latent;
                let out_base = ((b * heads + h) * tokens + t) * pack_len;
                for p in 0..pack_len {
                    let l = p * 4;
                    let v0 = *codes.get(base + l).unwrap_or(&0);
                    let v1 = *codes.get(base + l + 1).unwrap_or(&0);
                    let v2 = *codes.get(base + l + 2).unwrap_or(&0);
                    let v3 = *codes.get(base + l + 3).unwrap_or(&0);
                    packed[out_base + p] = pack_i8x4_host(v0, v1, v2, v3);
                }
            }
        }
    }
    packed
}

#[cfg(feature = "cuda")]
fn detect_cuda_arch_for_device(device_index: usize) -> Option<String> {
    if let Ok(value) = std::env::var("LOW_BIT_CUDA_NVRTC_ARCH") {
        if !value.trim().is_empty() {
            return Some(value);
        }
    }
    cudarc::driver::result::init().ok()?;
    let device_ptr = cudarc::driver::result::device::get(device_index as i32).ok()?;
    let (major, minor) = unsafe {
        (
            cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
            )
            .ok()?,
            cudarc::driver::result::device::get_attribute(
                device_ptr,
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
            )
            .ok()?,
        )
    };
    Some(format!("compute_{}{}", major, minor))
}

#[cfg(feature = "cuda")]
fn raw_cuda_packed_dot_kernels(device_index: usize) -> Option<RawCudaPackedDotKernels> {
    let cache = RAW_CUDA_PACKED_DOT_KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().ok()?;
    if let Some(existing) = cache.get(&device_index) {
        return Some(existing.clone());
    }
    let arch = Box::leak(detect_cuda_arch_for_device(device_index)?.into_boxed_str());
    let ctx = CudaContext::new(device_index).ok()?;
    let ptx = compile_ptx_with_opts(
        LOW_BIT_CUDA_RAW_DOT_FROM_CODES_SRC,
        CompileOptions {
            arch: Some(arch),
            fmad: Some(true),
            ..Default::default()
        },
    )
    .ok()?;
    let module = ctx.load_module(ptx).ok()?;
    let bundle = RawCudaPackedDotKernels {
        ctx: ctx.clone(),
        stream: ctx.default_stream(),
        lowrank: module.load_function("packed_lowrank_dp4a").ok()?,
        lowrank_scale_ptr: module.load_function("packed_lowrank_dp4a_scale_ptr").ok()?,
        lowrank_from_codes: module
            .load_function("packed_lowrank_dp4a_from_codes")
            .ok()?,
        lowrank_from_codes_scale_ptr: module
            .load_function("packed_lowrank_dp4a_from_codes_scale_ptr")
            .ok()?,
        decoder: module.load_function("packed_decoder_tail_dp4a").ok()?,
        decoder_scale_ptr: module
            .load_function("packed_decoder_tail_dp4a_scale_ptr")
            .ok()?,
        decoder_from_codes: module
            .load_function("packed_decoder_tail_dp4a_from_codes")
            .ok()?,
        decoder_from_codes_scale_ptr: module
            .load_function("packed_decoder_tail_dp4a_from_codes_scale_ptr")
            .ok()?,
        quantize_pack_i8x4_from_f32_scale_ptr: module
            .load_function("quantize_pack_i8x4_from_f32_scale_ptr")
            .ok()?,
        pack_i8x4_from_i32_codes: module.load_function("pack_i8x4_from_i32_codes").ok()?,
        lowrank_grad_input: module.load_function("packed_lowrank_grad_input_raw").ok()?,
        lowrank_grad_weight: module
            .load_function("packed_lowrank_grad_weight_raw")
            .ok()?,
        decoder_grad_input: module
            .load_function("packed_decoder_tail_grad_input_raw")
            .ok()?,
        decoder_grad_weight: module
            .load_function("packed_decoder_tail_grad_weight_raw")
            .ok()?,
    };
    cache.insert(device_index, bundle.clone());
    Some(bundle)
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input<B: BackendTrait>(
    input_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, pack_len] = input_packed.shape().dims::<4>();
    let [heads, weight_pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if weight_pack_len != pack_len || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale<B: BackendTrait>(
    input_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, pack_len] = input_packed.shape().dims::<4>();
    let [heads, weight_pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if weight_pack_len != pack_len || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_scale_ptr);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input<B: BackendTrait>(
    _input_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale<B: BackendTrait>(
    _input_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: &BurnTensor<B, 1>,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if pack_len != embd.div_ceil(4) || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_from_codes);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_projection_device_scale<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, pack_len, artifact_latent] = packed_weight_codes.shape().dims::<3>();
    if pack_len != embd.div_ceil(4) || !(input_heads == 1 || input_heads == heads) {
        return None;
    }
    if latent_out > artifact_latent {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([batch, heads, time, latent_out]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent_out.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let pack_len_i32 = pack_len as i32;
    let latent_out_i32 = latent_out as i32;
    let mut builder = kernels
        .stream
        .launch_builder(&kernels.lowrank_from_codes_scale_ptr);
    builder.arg(&input_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&latent_out_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_quantize_pack_activation_i8x4<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, inner] = input.shape().dims::<4>();
    let outer = batch * heads * time;
    let pack_len = inner.div_ceil(4);
    let input_tensor: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(input.clone().into_primitive().tensor())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if input_tensor.dtype != DType::F32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input_tensor.device.index)?;
    let output = empty_device::<CudaRuntime, i32>(
        input_tensor.client.clone(),
        input_tensor.device.clone(),
        Shape::new([batch, heads, time, pack_len]),
    );
    let input_ptr = input_tensor
        .client
        .get_resource(input_tensor.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            (outer * pack_len).div_ceil(block_size_x as usize) as u32,
            1,
            1,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let outer_i32 = outer as i32;
    let inner_i32 = inner as i32;
    let pack_len_i32 = pack_len as i32;
    let positive_only_i32 = if positive_only { 1 } else { 0 };
    let mut builder = kernels
        .stream
        .launch_builder(&kernels.quantize_pack_i8x4_from_f32_scale_ptr);
    builder.arg(&input_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer_i32);
    builder.arg(&inner_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&qmax);
    builder.arg(&positive_only_i32);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_int_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4, Int>::from_primitive(output_prim))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_quantize_pack_activation_i8x4<B: BackendTrait>(
    _input: &BurnTensor<B, 4>,
    _activation_scale: &BurnTensor<B, 1>,
    _qmax: i32,
    _positive_only: bool,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
#[allow(dead_code)]
pub fn try_raw_cuda_pack_activation_codes_i8x4<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::IntTensorPrimitive: 'static,
{
    let [batch, heads, time, inner] = input_codes.shape().dims::<4>();
    let outer = batch * heads * time;
    let pack_len = inner.div_ceil(4);
    let input_tensor: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    if input_tensor.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input_tensor.device.index)?;
    let output = empty_device::<CudaRuntime, i32>(
        input_tensor.client.clone(),
        input_tensor.device.clone(),
        Shape::new([batch, heads, time, pack_len]),
    );
    let input_ptr = input_tensor
        .client
        .get_resource(input_tensor.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            (outer * pack_len).div_ceil(block_size_x as usize) as u32,
            1,
            1,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let outer_i32 = outer as i32;
    let inner_i32 = inner as i32;
    let pack_len_i32 = pack_len as i32;
    let mut builder = kernels
        .stream
        .launch_builder(&kernels.pack_i8x4_from_i32_codes);
    builder.arg(&input_ptr);
    builder.arg(&output_ptr);
    builder.arg(&outer_i32);
    builder.arg(&inner_i32);
    builder.arg(&pack_len_i32);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_int_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4, Int>::from_primitive(output_prim))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_pack_activation_codes_i8x4<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::IntTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection_device_scale<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: &BurnTensor<B, 1>,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_projection<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 3, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
    _latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input<B: BackendTrait>(
    y_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, pack_len] = y_packed.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let weight_pack_len = packed_latent_total / heads;
    if weight_pack_len != pack_len {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input_device_scale<B: BackendTrait>(
    y_packed: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, pack_len] = y_packed.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let weight_pack_len = packed_latent_total / heads;
    if weight_pack_len != pack_len {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_packed.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_scale_ptr);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input<B: BackendTrait>(
    _y_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_prepacked_input_device_scale<B: BackendTrait>(
    _y_packed: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: &BurnTensor<B, 1>,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_grad_input<B: BackendTrait>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 3, Int>,
    weight_scale: f32,
    input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = grad_output.shape().dims::<4>();
    let [weight_heads, embd, weight_latent] = weight_codes.shape().dims::<3>();
    if heads != weight_heads || latent != weight_latent {
        return None;
    }
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(grad.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        grad.client.clone(),
        grad.device.clone(),
        Shape::new([batch, input_heads, time, embd]),
    );
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            embd.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * input_heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let latent_i32 = latent as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_grad_input);
    builder.arg(&grad_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&latent_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_grad_input<B: BackendTrait>(
    _grad_output: &BurnTensor<B, 4>,
    _weight_codes: &BurnTensor<B, 3, Int>,
    _weight_scale: f32,
    _input_heads: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_lowrank_grad_weight<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [grad_batch, heads, grad_time, latent] = grad_output.shape().dims::<4>();
    if batch != grad_batch || time != grad_time {
        return None;
    }
    let input: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(input_codes.clone().into_primitive())?;
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if input.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(input.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        input.client.clone(),
        input.device.clone(),
        Shape::new([1, heads, embd, latent]),
    );
    let input_ptr = input
        .client
        .get_resource(input.handle.clone().binding())
        .resource()
        .ptr;
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent.div_ceil(block_size_x as usize) as u32,
            embd as u32,
            heads as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let input_heads_i32 = input_heads as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let embd_i32 = embd as i32;
    let latent_i32 = latent as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.lowrank_grad_weight);
    builder.arg(&input_ptr);
    builder.arg(&grad_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&input_heads_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&embd_i32);
    builder.arg(&latent_i32);
    builder.arg(&activation_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_lowrank_grad_weight<B: BackendTrait>(
    _input_codes: &BurnTensor<B, 4, Int>,
    _grad_output: &BurnTensor<B, 4>,
    _activation_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_grad_input<B: BackendTrait>(
    grad_output: &BurnTensor<B, 4>,
    weight_codes: &BurnTensor<B, 2, Int>,
    weight_scale: f32,
    heads: usize,
    latent: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, grad_heads, time, dim] = grad_output.shape().dims::<4>();
    let [weight_rows, weight_dim] = weight_codes.shape().dims::<2>();
    if grad_heads != 1 || weight_rows != heads * latent || weight_dim != dim {
        return None;
    }
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(weight_codes.clone().into_primitive())?;
    if grad.dtype != DType::F32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(grad.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        grad.client.clone(),
        grad.device.clone(),
        Shape::new([batch, heads, time, latent]),
    );
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            latent.div_ceil(block_size_x as usize) as u32,
            time as u32,
            (batch * heads) as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_grad_input);
    builder.arg(&grad_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&dim_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_grad_input<B: BackendTrait>(
    _grad_output: &BurnTensor<B, 4>,
    _weight_codes: &BurnTensor<B, 2, Int>,
    _weight_scale: f32,
    _heads: usize,
    _latent: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_grad_weight<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    grad_output: &BurnTensor<B, 4>,
    activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [grad_batch, grad_heads, grad_time, dim] = grad_output.shape().dims::<4>();
    if grad_batch != batch || grad_heads != 1 || grad_time != time {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let grad: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(grad_output.clone().into_primitive().tensor())?;
    if y.dtype != DType::I32 || grad.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([heads * latent, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let grad_ptr = grad
        .client
        .get_resource(grad.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            (heads * latent) as u32,
            1,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_grad_weight);
    builder.arg(&y_ptr);
    builder.arg(&grad_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_grad_weight<B: BackendTrait>(
    _y_codes: &BurnTensor<B, 4, Int>,
    _grad_output: &BurnTensor<B, 4>,
    _activation_scale: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let pack_len = packed_latent_total / heads;
    if pack_len != latent.div_ceil(4) {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels.stream.launch_builder(&kernels.decoder_from_codes);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&activation_scale);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(feature = "cuda")]
pub fn try_raw_cuda_packed_decoder_tail_device_scale<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    packed_weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [packed_latent_total, dim] = packed_weight_codes.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return None;
    }
    let pack_len = packed_latent_total / heads;
    if pack_len != latent.div_ceil(4) {
        return None;
    }
    let y: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(y_codes.clone().into_primitive())?;
    let weight: CubeTensor<CudaRuntime> =
        try_cast_int_primitive::<B, _>(packed_weight_codes.clone().into_primitive())?;
    let scale: CubeTensor<CudaRuntime> =
        try_cast_float_primitive::<B, _>(activation_scale.clone().into_primitive().tensor())?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return None;
    }
    let kernels = raw_cuda_packed_dot_kernels(y.device.index)?;
    let output = empty_device::<CudaRuntime, f32>(
        y.client.clone(),
        y.device.clone(),
        Shape::new([batch, 1, time, dim]),
    );
    let y_ptr = y
        .client
        .get_resource(y.handle.clone().binding())
        .resource()
        .ptr;
    let weight_ptr = weight
        .client
        .get_resource(weight.handle.clone().binding())
        .resource()
        .ptr;
    let scale_ptr = scale
        .client
        .get_resource(scale.handle.clone().binding())
        .resource()
        .ptr;
    let output_ptr = output
        .client
        .get_resource(output.handle.clone().binding())
        .resource()
        .ptr;
    let block_size_x = raw_cuda_workgroup_size_x();
    let launch_cfg = LaunchConfig {
        grid_dim: (
            dim.div_ceil(block_size_x as usize) as u32,
            time as u32,
            batch as u32,
        ),
        block_dim: (block_size_x, 1, 1),
        shared_mem_bytes: 0,
    };
    let batch_i32 = batch as i32;
    let heads_i32 = heads as i32;
    let time_i32 = time as i32;
    let latent_i32 = latent as i32;
    let pack_len_i32 = pack_len as i32;
    let dim_i32 = dim as i32;
    let mut builder = kernels
        .stream
        .launch_builder(&kernels.decoder_from_codes_scale_ptr);
    builder.arg(&y_ptr);
    builder.arg(&weight_ptr);
    builder.arg(&scale_ptr);
    builder.arg(&output_ptr);
    builder.arg(&batch_i32);
    builder.arg(&heads_i32);
    builder.arg(&time_i32);
    builder.arg(&latent_i32);
    builder.arg(&pack_len_i32);
    builder.arg(&dim_i32);
    builder.arg(&weight_scale);
    unsafe { builder.launch(launch_cfg) }.ok()?;
    let output_prim = try_cast_float_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail<B: BackendTrait>(
    _y_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: f32,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}

#[cfg(not(feature = "cuda"))]
pub fn try_raw_cuda_packed_decoder_tail_device_scale<B: BackendTrait>(
    _y_codes: &BurnTensor<B, 4, Int>,
    _packed_weight_codes: &BurnTensor<B, 2, Int>,
    _activation_scale: &BurnTensor<B, 1>,
    _weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    None
}
