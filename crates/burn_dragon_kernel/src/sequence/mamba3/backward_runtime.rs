use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Shape, TensorData, TensorPrimitive};
use burn_cubecl::CubeRuntime;
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::prelude::*;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;

const REVERSE_CUMSUM_BHL_PARAM_LEN: usize = 3;
const REVERSE_CUMSUM_BLHR_PARAM_LEN: usize = 4;
const CARRY_BACKWARD_PARAM_LEN: usize = 5;
const SCORE_BACKWARD_PARAM_LEN: usize = 5;
const FUSED_SCORE_CARRY_BACKWARD_PARAM_LEN: usize = 5;
const REVERSE_CUMSUM_WORKGROUP_X: u32 = 64;
const FUSED_SCORE_CARRY_WGPU_MAX_TIME: usize = 128;

#[cube]
fn reduce_partials_wgpu(
    partials: &mut SharedMemory<f32>,
    lane: usize,
    #[comptime] workgroup_size: usize,
) {
    if comptime!(workgroup_size >= 128usize) {
        if lane < 64usize {
            partials[lane] = partials[lane] + partials[lane + 64usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 64usize) {
        if lane < 32usize {
            partials[lane] = partials[lane] + partials[lane + 32usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 32usize) {
        if lane < 16usize {
            partials[lane] = partials[lane] + partials[lane + 16usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 16usize) {
        if lane < 8usize {
            partials[lane] = partials[lane] + partials[lane + 8usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 8usize) {
        if lane < 4usize {
            partials[lane] = partials[lane] + partials[lane + 4usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 4usize) {
        if lane < 2usize {
            partials[lane] = partials[lane] + partials[lane + 2usize];
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 2usize) {
        if lane < 1usize {
            partials[lane] = partials[lane] + partials[lane + 1usize];
        }
        sync_cube();
    }
}

#[cube(launch)]
fn reverse_cumsum_bhl_kernel(values: &Tensor<f32>, output: &mut Tensor<f32>, params: &Tensor<f32>) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let t = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);
    let mut tau = t;
    while tau < time {
        let index = b * values.stride(0) + h * values.stride(1) + tau * values.stride(2);
        acc += values[index];
        tau += 1usize;
    }

    let out_index = b * output.stride(0) + h * output.stride(1) + t * output.stride(2);
    output[out_index] = acc;
}

#[cube(launch)]
fn reverse_cumsum_blhr_kernel(
    values: &Tensor<f32>,
    output: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let heads = u32::cast_from(params[2]) as usize;
    let width = u32::cast_from(params[3]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let w = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || t >= time || h >= heads || w >= width {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);
    let mut tau = t;
    while tau < time {
        let index = b * values.stride(0)
            + tau * values.stride(1)
            + h * values.stride(2)
            + w * values.stride(3);
        acc += values[index];
        tau += 1usize;
    }

    let out_index =
        b * output.stride(0) + t * output.stride(1) + h * output.stride(2) + w * output.stride(3);
    output[out_index] = acc;
}

#[cube(launch)]
fn carry_backward_v_da_kernel(
    grad_ssm_carry: &Tensor<f32>,
    k_head: &Tensor<f32>,
    v_head: &Tensor<f32>,
    weighted_scale: &Tensor<f32>,
    grad_v_add: &mut Tensor<f32>,
    grad_da_terms: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let headdim = u32::cast_from(params[3]) as usize;
    let d_state = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let t = CUBE_POS_Y as usize;
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || d >= headdim {
        terminate!();
    }

    let scale = weighted_scale[b * weighted_scale.stride(0)
        + h * weighted_scale.stride(1)
        + t * weighted_scale.stride(2)];

    let mut acc = f32::cast_from(0u32);
    let mut s = 0usize;
    while s < d_state {
        let grad_index = b * grad_ssm_carry.stride(0)
            + h * grad_ssm_carry.stride(1)
            + d * grad_ssm_carry.stride(2)
            + s * grad_ssm_carry.stride(3);
        let k_index = b * k_head.stride(0)
            + h * k_head.stride(1)
            + t * k_head.stride(2)
            + s * k_head.stride(3);
        acc += grad_ssm_carry[grad_index] * k_head[k_index];
        s += 1usize;
    }

    let v_index =
        b * v_head.stride(0) + h * v_head.stride(1) + t * v_head.stride(2) + d * v_head.stride(3);
    let out_index = b * grad_v_add.stride(0)
        + h * grad_v_add.stride(1)
        + t * grad_v_add.stride(2)
        + d * grad_v_add.stride(3);
    let scaled = acc * scale;
    grad_v_add[out_index] = scaled;
    grad_da_terms[out_index] = acc * v_head[v_index];
}

#[cube(launch)]
fn carry_backward_k_kernel(
    grad_ssm_carry: &Tensor<f32>,
    v_head: &Tensor<f32>,
    weighted_scale: &Tensor<f32>,
    grad_k_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let headdim = u32::cast_from(params[3]) as usize;
    let d_state = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let t = CUBE_POS_Y as usize;
    let s = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || s >= d_state {
        terminate!();
    }

    let scale = weighted_scale[b * weighted_scale.stride(0)
        + h * weighted_scale.stride(1)
        + t * weighted_scale.stride(2)];
    let mut acc = f32::cast_from(0u32);
    let mut d = 0usize;
    while d < headdim {
        let grad_index = b * grad_ssm_carry.stride(0)
            + h * grad_ssm_carry.stride(1)
            + d * grad_ssm_carry.stride(2)
            + s * grad_ssm_carry.stride(3);
        let v_index = b * v_head.stride(0)
            + h * v_head.stride(1)
            + t * v_head.stride(2)
            + d * v_head.stride(3);
        acc += (v_head[v_index] * scale) * grad_ssm_carry[grad_index];
        d += 1usize;
    }

    let out_index = b * grad_k_add.stride(0)
        + h * grad_k_add.stride(1)
        + t * grad_k_add.stride(2)
        + s * grad_k_add.stride(3);
    grad_k_add[out_index] = acc;
}

#[cube(launch)]
fn score_backward_grad_q_kernel(
    grad_current_out: &Tensor<f32>,
    v_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_q_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let d_state = u32::cast_from(params[3]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let t = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= d_state {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);
    let mut u = 0usize;
    while u < t {
        let decay_index =
            b * decay.stride(0) + h * decay.stride(1) + t * decay.stride(2) + u * decay.stride(3);
        let decay_value = decay[decay_index];

        let mut score_grad = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + t * grad_current_out.stride(2)
                + d * grad_current_out.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + u * v_head.stride(2)
                + d * v_head.stride(3);
            score_grad += grad_current_out[grad_index] * v_head[v_index];
            d += 1usize;
        }

        let k_index = b * k_head.stride(0)
            + h * k_head.stride(1)
            + u * k_head.stride(2)
            + l * k_head.stride(3);
        acc += (score_grad * decay_value) * k_head[k_index];
        u += 1usize;
    }

    let out_index = b * grad_q_add.stride(0)
        + h * grad_q_add.stride(1)
        + t * grad_q_add.stride(2)
        + l * grad_q_add.stride(3);
    grad_q_add[out_index] = acc;
}

#[cube(launch)]
fn score_backward_grad_k_kernel(
    grad_current_out: &Tensor<f32>,
    v_head: &Tensor<f32>,
    q_head: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_k_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let d_state = u32::cast_from(params[3]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let u = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || u >= time || l >= d_state {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);
    let mut t = u + 1usize;
    while t < time {
        let decay_index =
            b * decay.stride(0) + h * decay.stride(1) + t * decay.stride(2) + u * decay.stride(3);
        let decay_value = decay[decay_index];

        let mut score_grad = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + t * grad_current_out.stride(2)
                + d * grad_current_out.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + u * v_head.stride(2)
                + d * v_head.stride(3);
            score_grad += grad_current_out[grad_index] * v_head[v_index];
            d += 1usize;
        }

        let q_index = b * q_head.stride(0)
            + h * q_head.stride(1)
            + t * q_head.stride(2)
            + l * q_head.stride(3);
        acc += (score_grad * decay_value) * q_head[q_index];
        t += 1usize;
    }

    let out_index = b * grad_k_add.stride(0)
        + h * grad_k_add.stride(1)
        + u * grad_k_add.stride(2)
        + l * grad_k_add.stride(3);
    grad_k_add[out_index] = acc;
}

#[cube(launch)]
fn score_backward_grad_v_kernel(
    grad_current_out: &Tensor<f32>,
    raw_scores: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_v_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let u = CUBE_POS_Y as usize;
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || u >= time || d >= headdim {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);
    let mut t = u + 1usize;
    while t < time {
        let raw_index = b * raw_scores.stride(0)
            + h * raw_scores.stride(1)
            + t * raw_scores.stride(2)
            + u * raw_scores.stride(3);
        let decay_index =
            b * decay.stride(0) + h * decay.stride(1) + t * decay.stride(2) + u * decay.stride(3);
        let grad_index = b * grad_current_out.stride(0)
            + h * grad_current_out.stride(1)
            + t * grad_current_out.stride(2)
            + d * grad_current_out.stride(3);
        acc += (raw_scores[raw_index] * decay[decay_index]) * grad_current_out[grad_index];
        t += 1usize;
    }

    let out_index = b * grad_v_add.stride(0)
        + h * grad_v_add.stride(1)
        + u * grad_v_add.stride(2)
        + d * grad_v_add.stride(3);
    grad_v_add[out_index] = acc;
}

#[cube(launch)]
fn score_backward_grad_da_kernel(
    grad_current_out: &Tensor<f32>,
    v_head: &Tensor<f32>,
    raw_scores: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_da_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let i = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || i >= time {
        terminate!();
    }

    let mut acc = f32::cast_from(0u32);

    let mut u = 0usize;
    while u < i {
        let mut score_grad = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + i * grad_current_out.stride(2)
                + d * grad_current_out.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + u * v_head.stride(2)
                + d * v_head.stride(3);
            score_grad += grad_current_out[grad_index] * v_head[v_index];
            d += 1usize;
        }
        let raw_index = b * raw_scores.stride(0)
            + h * raw_scores.stride(1)
            + i * raw_scores.stride(2)
            + u * raw_scores.stride(3);
        let decay_index =
            b * decay.stride(0) + h * decay.stride(1) + i * decay.stride(2) + u * decay.stride(3);
        acc += score_grad * raw_scores[raw_index] * decay[decay_index];
        u += 1usize;
    }

    let mut t = i + 1usize;
    while t < time {
        let mut score_grad = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + t * grad_current_out.stride(2)
                + d * grad_current_out.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + i * v_head.stride(2)
                + d * v_head.stride(3);
            score_grad += grad_current_out[grad_index] * v_head[v_index];
            d += 1usize;
        }
        let raw_index = b * raw_scores.stride(0)
            + h * raw_scores.stride(1)
            + t * raw_scores.stride(2)
            + i * raw_scores.stride(3);
        let decay_index =
            b * decay.stride(0) + h * decay.stride(1) + t * decay.stride(2) + i * decay.stride(3);
        acc -= score_grad * raw_scores[raw_index] * decay[decay_index];
        t += 1usize;
    }

    let out_index =
        b * grad_da_add.stride(0) + h * grad_da_add.stride(1) + i * grad_da_add.stride(2);
    grad_da_add[out_index] = acc;
}

#[cube(launch)]
fn fused_score_carry_backward_kernel(
    grad_current_out: &Tensor<f32>,
    v_head: &Tensor<f32>,
    q_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    raw_scores: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_ssm_carry: &Tensor<f32>,
    weighted_scale: &Tensor<f32>,
    grad_q_add: &mut Tensor<f32>,
    grad_k_add: &mut Tensor<f32>,
    grad_v_add: &mut Tensor<f32>,
    grad_da_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let d_state = u32::cast_from(params[3]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let t = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time {
        terminate!();
    }

    let scale_t = weighted_scale[b * weighted_scale.stride(0)
        + h * weighted_scale.stride(1)
        + t * weighted_scale.stride(2)];

    if l < d_state {
        let mut q_acc = f32::cast_from(0u32);
        let mut u = 0usize;
        while u < t {
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + t * decay.stride(2)
                + u * decay.stride(3);
            let decay_value = decay[decay_index];

            let mut score_grad = f32::cast_from(0u32);
            let mut d = 0usize;
            while d < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + t * grad_current_out.stride(2)
                    + d * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + u * v_head.stride(2)
                    + d * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d += 1usize;
            }

            let k_index = b * k_head.stride(0)
                + h * k_head.stride(1)
                + u * k_head.stride(2)
                + l * k_head.stride(3);
            q_acc += (score_grad * decay_value) * k_head[k_index];
            u += 1usize;
        }

        let mut k_acc = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_ssm_carry.stride(0)
                + h * grad_ssm_carry.stride(1)
                + d * grad_ssm_carry.stride(2)
                + l * grad_ssm_carry.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + t * v_head.stride(2)
                + d * v_head.stride(3);
            k_acc += (v_head[v_index] * scale_t) * grad_ssm_carry[grad_index];
            d += 1usize;
        }

        let mut i = t + 1usize;
        while i < time {
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + i * decay.stride(2)
                + t * decay.stride(3);
            let decay_value = decay[decay_index];

            let mut score_grad = f32::cast_from(0u32);
            let mut d2 = 0usize;
            while d2 < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + i * grad_current_out.stride(2)
                    + d2 * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + t * v_head.stride(2)
                    + d2 * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d2 += 1usize;
            }

            let q_index = b * q_head.stride(0)
                + h * q_head.stride(1)
                + i * q_head.stride(2)
                + l * q_head.stride(3);
            k_acc += (score_grad * decay_value) * q_head[q_index];
            i += 1usize;
        }

        let q_out = b * grad_q_add.stride(0)
            + h * grad_q_add.stride(1)
            + t * grad_q_add.stride(2)
            + l * grad_q_add.stride(3);
        grad_q_add[q_out] = q_acc;
        let k_out = b * grad_k_add.stride(0)
            + h * grad_k_add.stride(1)
            + t * grad_k_add.stride(2)
            + l * grad_k_add.stride(3);
        grad_k_add[k_out] = k_acc;
    }

    if l < headdim {
        let mut carry_inner = f32::cast_from(0u32);
        let mut s = 0usize;
        while s < d_state {
            let grad_index = b * grad_ssm_carry.stride(0)
                + h * grad_ssm_carry.stride(1)
                + l * grad_ssm_carry.stride(2)
                + s * grad_ssm_carry.stride(3);
            let k_index = b * k_head.stride(0)
                + h * k_head.stride(1)
                + t * k_head.stride(2)
                + s * k_head.stride(3);
            carry_inner += grad_ssm_carry[grad_index] * k_head[k_index];
            s += 1usize;
        }

        let mut v_acc = carry_inner * scale_t;
        let mut i = t + 1usize;
        while i < time {
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + i * raw_scores.stride(2)
                + t * raw_scores.stride(3);
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + i * decay.stride(2)
                + t * decay.stride(3);
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + i * grad_current_out.stride(2)
                + l * grad_current_out.stride(3);
            v_acc += (raw_scores[raw_index] * decay[decay_index]) * grad_current_out[grad_index];
            i += 1usize;
        }

        let v_out = b * grad_v_add.stride(0)
            + h * grad_v_add.stride(1)
            + t * grad_v_add.stride(2)
            + l * grad_v_add.stride(3);
        grad_v_add[v_out] = v_acc;
    }

    if l == 0usize {
        let mut da_acc = f32::cast_from(0u32);

        let mut u = 0usize;
        while u < t {
            let mut score_grad = f32::cast_from(0u32);
            let mut d = 0usize;
            while d < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + t * grad_current_out.stride(2)
                    + d * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + u * v_head.stride(2)
                    + d * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d += 1usize;
            }
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + t * raw_scores.stride(2)
                + u * raw_scores.stride(3);
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + t * decay.stride(2)
                + u * decay.stride(3);
            da_acc += score_grad * raw_scores[raw_index] * decay[decay_index];
            u += 1usize;
        }

        let mut i = t + 1usize;
        while i < time {
            let mut score_grad = f32::cast_from(0u32);
            let mut d = 0usize;
            while d < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + i * grad_current_out.stride(2)
                    + d * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + t * v_head.stride(2)
                    + d * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d += 1usize;
            }
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + i * raw_scores.stride(2)
                + t * raw_scores.stride(3);
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + i * decay.stride(2)
                + t * decay.stride(3);
            da_acc -= score_grad * raw_scores[raw_index] * decay[decay_index];
            i += 1usize;
        }

        let mut carry_inner_t = f32::cast_from(0u32);
        let mut d = 0usize;
        while d < headdim {
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + t * v_head.stride(2)
                + d * v_head.stride(3);
            let mut s = 0usize;
            while s < d_state {
                let grad_index = b * grad_ssm_carry.stride(0)
                    + h * grad_ssm_carry.stride(1)
                    + d * grad_ssm_carry.stride(2)
                    + s * grad_ssm_carry.stride(3);
                let k_index = b * k_head.stride(0)
                    + h * k_head.stride(1)
                    + t * k_head.stride(2)
                    + s * k_head.stride(3);
                carry_inner_t += grad_ssm_carry[grad_index] * k_head[k_index] * v_head[v_index];
                s += 1usize;
            }
            d += 1usize;
        }
        let carry_term_t = carry_inner_t * scale_t;
        da_acc -= carry_term_t;

        if t + 1usize == time {
            let mut total_carry = f32::cast_from(0u32);
            let mut tau = 0usize;
            while tau < time {
                let scale_tau = weighted_scale[b * weighted_scale.stride(0)
                    + h * weighted_scale.stride(1)
                    + tau * weighted_scale.stride(2)];
                let mut carry_inner_tau = f32::cast_from(0u32);
                let mut d2 = 0usize;
                while d2 < headdim {
                    let v_index = b * v_head.stride(0)
                        + h * v_head.stride(1)
                        + tau * v_head.stride(2)
                        + d2 * v_head.stride(3);
                    let mut s2 = 0usize;
                    while s2 < d_state {
                        let grad_index = b * grad_ssm_carry.stride(0)
                            + h * grad_ssm_carry.stride(1)
                            + d2 * grad_ssm_carry.stride(2)
                            + s2 * grad_ssm_carry.stride(3);
                        let k_index = b * k_head.stride(0)
                            + h * k_head.stride(1)
                            + tau * k_head.stride(2)
                            + s2 * k_head.stride(3);
                        carry_inner_tau +=
                            grad_ssm_carry[grad_index] * k_head[k_index] * v_head[v_index];
                        s2 += 1usize;
                    }
                    d2 += 1usize;
                }
                total_carry += carry_inner_tau * scale_tau;
                tau += 1usize;
            }
            da_acc += total_carry;
        }

        let da_out =
            b * grad_da_add.stride(0) + h * grad_da_add.stride(1) + t * grad_da_add.stride(2);
        grad_da_add[da_out] = da_acc;
    }
}

#[cube(launch)]
fn fused_score_carry_backward_tiled_wgpu_kernel(
    grad_current_out: &Tensor<f32>,
    v_head: &Tensor<f32>,
    q_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    raw_scores: &Tensor<f32>,
    decay: &Tensor<f32>,
    grad_ssm_carry: &Tensor<f32>,
    weighted_scale: &Tensor<f32>,
    grad_q_add: &mut Tensor<f32>,
    grad_k_add: &mut Tensor<f32>,
    grad_v_add: &mut Tensor<f32>,
    grad_da_add: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_time: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let d_state = u32::cast_from(params[3]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / heads.max(1);
    let h = z % heads.max(1);
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || h >= heads || t >= time || time > max_time {
        terminate!();
    }

    let zero = f32::cast_from(0u32);
    let scale_t = weighted_scale[b * weighted_scale.stride(0)
        + h * weighted_scale.stride(1)
        + t * weighted_scale.stride(2)];

    let mut weighted_past = SharedMemory::<f32>::new_aligned(max_time, 1usize);
    let mut weighted_future = SharedMemory::<f32>::new_aligned(max_time, 1usize);
    let mut carry_partials =
        SharedMemory::<f32>::new_aligned(REVERSE_CUMSUM_WORKGROUP_X as usize, 1usize);

    let mut idx = lane;
    while idx < time {
        if idx < t {
            let mut score_grad = zero;
            let mut d = 0usize;
            while d < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + t * grad_current_out.stride(2)
                    + d * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + idx * v_head.stride(2)
                    + d * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d += 1usize;
            }
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + t * decay.stride(2)
                + idx * decay.stride(3);
            weighted_past[idx] = score_grad * decay[decay_index];
        } else {
            weighted_past[idx] = zero;
        }

        if idx > t {
            let mut score_grad = zero;
            let mut d = 0usize;
            while d < headdim {
                let grad_index = b * grad_current_out.stride(0)
                    + h * grad_current_out.stride(1)
                    + idx * grad_current_out.stride(2)
                    + d * grad_current_out.stride(3);
                let v_index = b * v_head.stride(0)
                    + h * v_head.stride(1)
                    + t * v_head.stride(2)
                    + d * v_head.stride(3);
                score_grad += grad_current_out[grad_index] * v_head[v_index];
                d += 1usize;
            }
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + idx * decay.stride(2)
                + t * decay.stride(3);
            weighted_future[idx] = score_grad * decay[decay_index];
        } else {
            weighted_future[idx] = zero;
        }
        idx += CUBE_DIM_X as usize;
    }

    sync_cube();

    let mut l = lane;
    while l < d_state {
        let mut q_acc = zero;
        let mut u = 0usize;
        while u < t {
            let k_index = b * k_head.stride(0)
                + h * k_head.stride(1)
                + u * k_head.stride(2)
                + l * k_head.stride(3);
            q_acc += weighted_past[u] * k_head[k_index];
            u += 1usize;
        }

        let mut k_acc = zero;
        let mut d = 0usize;
        while d < headdim {
            let grad_index = b * grad_ssm_carry.stride(0)
                + h * grad_ssm_carry.stride(1)
                + d * grad_ssm_carry.stride(2)
                + l * grad_ssm_carry.stride(3);
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + t * v_head.stride(2)
                + d * v_head.stride(3);
            k_acc += (v_head[v_index] * scale_t) * grad_ssm_carry[grad_index];
            d += 1usize;
        }

        let mut i = t + 1usize;
        while i < time {
            let q_index = b * q_head.stride(0)
                + h * q_head.stride(1)
                + i * q_head.stride(2)
                + l * q_head.stride(3);
            k_acc += weighted_future[i] * q_head[q_index];
            i += 1usize;
        }

        let q_out = b * grad_q_add.stride(0)
            + h * grad_q_add.stride(1)
            + t * grad_q_add.stride(2)
            + l * grad_q_add.stride(3);
        grad_q_add[q_out] = q_acc;
        let k_out = b * grad_k_add.stride(0)
            + h * grad_k_add.stride(1)
            + t * grad_k_add.stride(2)
            + l * grad_k_add.stride(3);
        grad_k_add[k_out] = k_acc;
        l += CUBE_DIM_X as usize;
    }

    let mut local_carry_da = zero;
    let mut d = lane;
    while d < headdim {
        let mut carry_inner = zero;
        let mut s = 0usize;
        while s < d_state {
            let grad_index = b * grad_ssm_carry.stride(0)
                + h * grad_ssm_carry.stride(1)
                + d * grad_ssm_carry.stride(2)
                + s * grad_ssm_carry.stride(3);
            let k_index = b * k_head.stride(0)
                + h * k_head.stride(1)
                + t * k_head.stride(2)
                + s * k_head.stride(3);
            carry_inner += grad_ssm_carry[grad_index] * k_head[k_index];
            s += 1usize;
        }

        let mut v_acc = carry_inner * scale_t;
        let mut i = t + 1usize;
        while i < time {
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + i * raw_scores.stride(2)
                + t * raw_scores.stride(3);
            let decay_index = b * decay.stride(0)
                + h * decay.stride(1)
                + i * decay.stride(2)
                + t * decay.stride(3);
            let grad_index = b * grad_current_out.stride(0)
                + h * grad_current_out.stride(1)
                + i * grad_current_out.stride(2)
                + d * grad_current_out.stride(3);
            v_acc += (raw_scores[raw_index] * decay[decay_index]) * grad_current_out[grad_index];
            i += 1usize;
        }

        let v_out = b * grad_v_add.stride(0)
            + h * grad_v_add.stride(1)
            + t * grad_v_add.stride(2)
            + d * grad_v_add.stride(3);
        grad_v_add[v_out] = v_acc;

        let v_index = b * v_head.stride(0)
            + h * v_head.stride(1)
            + t * v_head.stride(2)
            + d * v_head.stride(3);
        local_carry_da += (carry_inner * scale_t) * v_head[v_index];
        d += CUBE_DIM_X as usize;
    }

    carry_partials[lane] = local_carry_da;
    sync_cube();
    reduce_partials_wgpu(
        &mut carry_partials,
        lane,
        REVERSE_CUMSUM_WORKGROUP_X as usize,
    );

    if lane == 0usize {
        let mut da_acc = zero;

        let mut u = 0usize;
        while u < t {
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + t * raw_scores.stride(2)
                + u * raw_scores.stride(3);
            da_acc += weighted_past[u] * raw_scores[raw_index];
            u += 1usize;
        }

        let mut i = t + 1usize;
        while i < time {
            let raw_index = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + i * raw_scores.stride(2)
                + t * raw_scores.stride(3);
            da_acc -= weighted_future[i] * raw_scores[raw_index];
            i += 1usize;
        }

        da_acc -= carry_partials[0usize];

        if t + 1usize == time {
            let mut total_carry = zero;
            let mut tau = 0usize;
            while tau < time {
                let scale_tau = weighted_scale[b * weighted_scale.stride(0)
                    + h * weighted_scale.stride(1)
                    + tau * weighted_scale.stride(2)];
                let mut carry_tau = zero;
                let mut d2 = 0usize;
                while d2 < headdim {
                    let mut carry_inner_tau = zero;
                    let mut s2 = 0usize;
                    while s2 < d_state {
                        let grad_index = b * grad_ssm_carry.stride(0)
                            + h * grad_ssm_carry.stride(1)
                            + d2 * grad_ssm_carry.stride(2)
                            + s2 * grad_ssm_carry.stride(3);
                        let k_index = b * k_head.stride(0)
                            + h * k_head.stride(1)
                            + tau * k_head.stride(2)
                            + s2 * k_head.stride(3);
                        carry_inner_tau += grad_ssm_carry[grad_index] * k_head[k_index];
                        s2 += 1usize;
                    }
                    let v_index = b * v_head.stride(0)
                        + h * v_head.stride(1)
                        + tau * v_head.stride(2)
                        + d2 * v_head.stride(3);
                    carry_tau += carry_inner_tau * v_head[v_index];
                    d2 += 1usize;
                }
                total_carry += carry_tau * scale_tau;
                tau += 1usize;
            }
            da_acc += total_carry;
        }

        let da_out =
            b * grad_da_add.stride(0) + h * grad_da_add.stride(1) + t * grad_da_add.stride(2);
        grad_da_add[da_out] = da_acc;
    }
}

fn reverse_cumsum_bhl_params<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![batch as f32, heads as f32, time as f32],
            [REVERSE_CUMSUM_BHL_PARAM_LEN],
        ),
        device,
    )
}

fn reverse_cumsum_blhr_params<B: BackendTrait>(
    batch: usize,
    time: usize,
    heads: usize,
    width: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![batch as f32, time as f32, heads as f32, width as f32],
            [REVERSE_CUMSUM_BLHR_PARAM_LEN],
        ),
        device,
    )
}

fn carry_backward_params<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    headdim: usize,
    d_state: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                headdim as f32,
                d_state as f32,
            ],
            [CARRY_BACKWARD_PARAM_LEN],
        ),
        device,
    )
}

fn score_backward_params<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    d_state: usize,
    headdim: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                d_state as f32,
                headdim as f32,
            ],
            [SCORE_BACKWARD_PARAM_LEN],
        ),
        device,
    )
}

fn fused_score_carry_backward_params<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    d_state: usize,
    headdim: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                d_state as f32,
                headdim as f32,
            ],
            [FUSED_SCORE_CARRY_BACKWARD_PARAM_LEN],
        ),
        device,
    )
}

fn reverse_cumsum_bhl_runtime<R: CubeRuntime>(
    values: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let values = into_contiguous(values);
    let params = into_contiguous(params);
    let [batch, heads, time] = values.meta.shape.dims::<3>();
    let client = values.client.clone();
    let device = values.device.clone();
    let output = empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        time.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        heads as u32,
        batch as u32,
    );
    let _ = reverse_cumsum_bhl_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        values.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    output
}

fn reverse_cumsum_blhr_runtime<R: CubeRuntime>(
    values: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let values = into_contiguous(values);
    let params = into_contiguous(params);
    let [batch, time, heads, width] = values.meta.shape.dims::<4>();
    let client = values.client.clone();
    let device = values.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, time, heads, width]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        width.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        heads as u32,
        (batch * time) as u32,
    );
    let _ = reverse_cumsum_blhr_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        values.into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    output
}

fn carry_backward_v_da_runtime<R: CubeRuntime>(
    grad_ssm_carry: CubeTensor<R>,
    k_head: CubeTensor<R>,
    v_head: CubeTensor<R>,
    weighted_scale: CubeTensor<R>,
    params: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let grad_ssm_carry = into_contiguous(grad_ssm_carry);
    let k_head = into_contiguous(k_head);
    let v_head = into_contiguous(v_head);
    let weighted_scale = into_contiguous(weighted_scale);
    let params = into_contiguous(params);
    let [batch, heads, time, headdim] = v_head.meta.shape.dims::<4>();
    let client = v_head.client.clone();
    let device = v_head.device.clone();
    let grad_v_add = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, headdim]),
    );
    let grad_da_terms = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, headdim]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        headdim.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = carry_backward_v_da_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_ssm_carry.into_tensor_arg(),
        k_head.into_tensor_arg(),
        v_head.into_tensor_arg(),
        weighted_scale.into_tensor_arg(),
        grad_v_add.clone().into_tensor_arg(),
        grad_da_terms.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    (grad_v_add, grad_da_terms)
}

fn carry_backward_k_runtime<R: CubeRuntime>(
    grad_ssm_carry: CubeTensor<R>,
    v_head: CubeTensor<R>,
    weighted_scale: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let grad_ssm_carry = into_contiguous(grad_ssm_carry);
    let v_head = into_contiguous(v_head);
    let weighted_scale = into_contiguous(weighted_scale);
    let params = into_contiguous(params);
    let [batch, heads, time] = weighted_scale.meta.shape.dims::<3>();
    let d_state = grad_ssm_carry.meta.shape.dims::<4>()[3];
    let client = grad_ssm_carry.client.clone();
    let device = grad_ssm_carry.device.clone();
    let grad_k_add = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, d_state]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        d_state.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = carry_backward_k_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_ssm_carry.into_tensor_arg(),
        v_head.into_tensor_arg(),
        weighted_scale.into_tensor_arg(),
        grad_k_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    grad_k_add
}

fn score_backward_grad_q_runtime<R: CubeRuntime>(
    grad_current_out: CubeTensor<R>,
    v_head: CubeTensor<R>,
    k_head: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let grad_current_out = into_contiguous(grad_current_out);
    let v_head = into_contiguous(v_head);
    let k_head = into_contiguous(k_head);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);
    let [batch, heads, time, d_state] = k_head.meta.shape.dims::<4>();
    let client = k_head.client.clone();
    let device = k_head.device.clone();
    let grad_q_add = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, d_state]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        d_state.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = score_backward_grad_q_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        v_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_q_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    grad_q_add
}

fn score_backward_grad_k_runtime<R: CubeRuntime>(
    grad_current_out: CubeTensor<R>,
    v_head: CubeTensor<R>,
    q_head: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let grad_current_out = into_contiguous(grad_current_out);
    let v_head = into_contiguous(v_head);
    let q_head = into_contiguous(q_head);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);
    let [batch, heads, time, d_state] = q_head.meta.shape.dims::<4>();
    let client = q_head.client.clone();
    let device = q_head.device.clone();
    let grad_k_add = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, d_state]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        d_state.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = score_backward_grad_k_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        v_head.into_tensor_arg(),
        q_head.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_k_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    grad_k_add
}

fn score_backward_grad_v_runtime<R: CubeRuntime>(
    grad_current_out: CubeTensor<R>,
    raw_scores: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let grad_current_out = into_contiguous(grad_current_out);
    let raw_scores = into_contiguous(raw_scores);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);
    let [batch, heads, time, headdim] = grad_current_out.meta.shape.dims::<4>();
    let client = grad_current_out.client.clone();
    let device = grad_current_out.device.clone();
    let grad_v_add = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, headdim]),
    );
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        headdim.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = score_backward_grad_v_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        raw_scores.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_v_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    grad_v_add
}

fn score_backward_grad_da_runtime<R: CubeRuntime>(
    grad_current_out: CubeTensor<R>,
    v_head: CubeTensor<R>,
    raw_scores: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let grad_current_out = into_contiguous(grad_current_out);
    let v_head = into_contiguous(v_head);
    let raw_scores = into_contiguous(raw_scores);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);
    let [batch, heads, time, _headdim] = grad_current_out.meta.shape.dims::<4>();
    let client = grad_current_out.client.clone();
    let device = grad_current_out.device.clone();
    let grad_da_add =
        empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        time.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        1,
        (batch * heads) as u32,
    );
    let _ = score_backward_grad_da_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        v_head.into_tensor_arg(),
        raw_scores.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_da_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    grad_da_add
}

fn fused_score_carry_backward_runtime<R: CubeRuntime>(
    grad_current_out: CubeTensor<R>,
    v_head: CubeTensor<R>,
    q_head: CubeTensor<R>,
    k_head: CubeTensor<R>,
    raw_scores: CubeTensor<R>,
    decay: CubeTensor<R>,
    grad_ssm_carry: CubeTensor<R>,
    weighted_scale: CubeTensor<R>,
    params: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>, CubeTensor<R>, CubeTensor<R>) {
    let grad_current_out = into_contiguous(grad_current_out);
    let v_head = into_contiguous(v_head);
    let q_head = into_contiguous(q_head);
    let k_head = into_contiguous(k_head);
    let raw_scores = into_contiguous(raw_scores);
    let decay = into_contiguous(decay);
    let grad_ssm_carry = into_contiguous(grad_ssm_carry);
    let weighted_scale = into_contiguous(weighted_scale);
    let params = into_contiguous(params);

    let [batch, heads, time, headdim] = grad_current_out.meta.shape.dims::<4>();
    let d_state = q_head.meta.shape.dims::<4>()[3];
    let active_width = d_state.max(headdim);
    let client = grad_current_out.client.clone();
    let device = grad_current_out.device.clone();
    let grad_q_add = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, d_state]),
    );
    let grad_k_add = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, d_state]),
    );
    let grad_v_add = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, headdim]),
    );
    let grad_da_add =
        empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        active_width.div_ceil(REVERSE_CUMSUM_WORKGROUP_X as usize) as u32,
        time as u32,
        (batch * heads) as u32,
    );
    let _ = fused_score_carry_backward_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        v_head.into_tensor_arg(),
        q_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        raw_scores.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_ssm_carry.into_tensor_arg(),
        weighted_scale.into_tensor_arg(),
        grad_q_add.clone().into_tensor_arg(),
        grad_k_add.clone().into_tensor_arg(),
        grad_v_add.clone().into_tensor_arg(),
        grad_da_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    (grad_q_add, grad_k_add, grad_v_add, grad_da_add)
}

fn fused_score_carry_backward_tiled_wgpu_runtime(
    grad_current_out: CubeTensor<WgpuRuntime>,
    v_head: CubeTensor<WgpuRuntime>,
    q_head: CubeTensor<WgpuRuntime>,
    k_head: CubeTensor<WgpuRuntime>,
    raw_scores: CubeTensor<WgpuRuntime>,
    decay: CubeTensor<WgpuRuntime>,
    grad_ssm_carry: CubeTensor<WgpuRuntime>,
    weighted_scale: CubeTensor<WgpuRuntime>,
    params: CubeTensor<WgpuRuntime>,
) -> (
    CubeTensor<WgpuRuntime>,
    CubeTensor<WgpuRuntime>,
    CubeTensor<WgpuRuntime>,
    CubeTensor<WgpuRuntime>,
) {
    let grad_current_out = into_contiguous(grad_current_out);
    let v_head = into_contiguous(v_head);
    let q_head = into_contiguous(q_head);
    let k_head = into_contiguous(k_head);
    let raw_scores = into_contiguous(raw_scores);
    let decay = into_contiguous(decay);
    let grad_ssm_carry = into_contiguous(grad_ssm_carry);
    let weighted_scale = into_contiguous(weighted_scale);
    let params = into_contiguous(params);

    let [batch, heads, time, headdim] = grad_current_out.meta.shape.dims::<4>();
    let d_state = q_head.meta.shape.dims::<4>()[3];
    let client = grad_current_out.client.clone();
    let device = grad_current_out.device.clone();
    let grad_q_add = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, d_state]),
    );
    let grad_k_add = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, d_state]),
    );
    let grad_v_add = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, headdim]),
    );
    let grad_da_add =
        empty_device::<WgpuRuntime, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(REVERSE_CUMSUM_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, time as u32, (batch * heads) as u32);
    let _ = fused_score_carry_backward_tiled_wgpu_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        grad_current_out.into_tensor_arg(),
        v_head.into_tensor_arg(),
        q_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        raw_scores.into_tensor_arg(),
        decay.into_tensor_arg(),
        grad_ssm_carry.into_tensor_arg(),
        weighted_scale.into_tensor_arg(),
        grad_q_add.clone().into_tensor_arg(),
        grad_k_add.clone().into_tensor_arg(),
        grad_v_add.clone().into_tensor_arg(),
        grad_da_add.clone().into_tensor_arg(),
        params.into_tensor_arg(),
        FUSED_SCORE_CARRY_WGPU_MAX_TIME,
    );
    (grad_q_add, grad_k_add, grad_v_add, grad_da_add)
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn std::any::Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}

pub(crate) fn try_reverse_cumsum_bhl<B: BackendTrait>(
    values: BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time] = values.shape().dims::<3>();
    let params = reverse_cumsum_bhl_params::<B>(batch, heads, time, &values.device());
    let values_raw = values.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (Some(values_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(values_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let output = reverse_cumsum_bhl_runtime::<WgpuRuntime>(values_cube, params_cube);
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let (Some(values_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(values_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let output = reverse_cumsum_bhl_runtime::<CudaRuntime>(values_cube, params_cube);
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output)?,
        )));
    }

    None
}

pub(crate) fn try_reverse_cumsum_blhr<B: BackendTrait>(
    values: BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, time, heads, width] = values.shape().dims::<4>();
    let params = reverse_cumsum_blhr_params::<B>(batch, time, heads, width, &values.device());
    let values_raw = values.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (Some(values_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(values_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let output = reverse_cumsum_blhr_runtime::<WgpuRuntime>(values_cube, params_cube);
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let (Some(values_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(values_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let output = reverse_cumsum_blhr_runtime::<CudaRuntime>(values_cube, params_cube);
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(output)?,
        )));
    }

    None
}

pub(crate) fn try_carry_backward<B: BackendTrait>(
    grad_ssm_carry: BurnTensor<B, 4>,
    k_head: BurnTensor<B, 4>,
    v_head: BurnTensor<B, 4>,
    weighted_scale: BurnTensor<B, 3>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>, BurnTensor<B, 3>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, headdim] = v_head.shape().dims::<4>();
    let d_state = grad_ssm_carry.shape().dims::<4>()[3];
    let params = carry_backward_params::<B>(batch, heads, time, headdim, d_state, &v_head.device());
    let grad_raw = grad_ssm_carry.into_primitive().tensor();
    let k_raw = k_head.into_primitive().tensor();
    let v_raw = v_head.into_primitive().tensor();
    let scale_raw = weighted_scale.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (Some(grad_cube), Some(k_cube), Some(v_cube), Some(scale_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(scale_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let (grad_v_add, grad_da_terms) = carry_backward_v_da_runtime::<WgpuRuntime>(
            grad_cube.clone(),
            k_cube,
            v_cube.clone(),
            scale_cube.clone(),
            params_cube.clone(),
        );
        let grad_k_add =
            carry_backward_k_runtime::<WgpuRuntime>(grad_cube, v_cube, scale_cube, params_cube);
        let grad_v_add = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_v_add)?,
        ));
        let grad_da_terms = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_da_terms)?,
        ));
        let grad_k_add = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_k_add)?,
        ));
        let grad_da_scale = grad_da_terms.sum_dim(3).reshape([batch, heads, time]);
        return Some((grad_k_add, grad_v_add, grad_da_scale));
    }

    #[cfg(feature = "cuda")]
    if let (Some(grad_cube), Some(k_cube), Some(v_cube), Some(scale_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(scale_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let (grad_v_add, grad_da_terms) = carry_backward_v_da_runtime::<CudaRuntime>(
            grad_cube.clone(),
            k_cube,
            v_cube.clone(),
            scale_cube.clone(),
            params_cube.clone(),
        );
        let grad_k_add =
            carry_backward_k_runtime::<CudaRuntime>(grad_cube, v_cube, scale_cube, params_cube);
        let grad_v_add = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_v_add)?,
        ));
        let grad_da_terms = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_da_terms)?,
        ));
        let grad_k_add = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(grad_k_add)?,
        ));
        let grad_da_scale = grad_da_terms.sum_dim(3).reshape([batch, heads, time]);
        return Some((grad_k_add, grad_v_add, grad_da_scale));
    }

    None
}

pub(crate) fn try_current_score_backward<B: BackendTrait>(
    grad_current_out: BurnTensor<B, 4>,
    v_head: BurnTensor<B, 4>,
    q_head: BurnTensor<B, 4>,
    k_head: BurnTensor<B, 4>,
    raw_scores: BurnTensor<B, 4>,
    decay: BurnTensor<B, 4>,
) -> Option<(
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, headdim] = grad_current_out.shape().dims::<4>();
    let d_state = q_head.shape().dims::<4>()[3];
    let params = score_backward_params::<B>(
        batch,
        heads,
        time,
        d_state,
        headdim,
        &grad_current_out.device(),
    );
    let grad_current_out_raw = grad_current_out.into_primitive().tensor();
    let v_raw = v_head.into_primitive().tensor();
    let q_raw = q_head.into_primitive().tensor();
    let k_raw = k_head.into_primitive().tensor();
    let raw_scores_raw = raw_scores.into_primitive().tensor();
    let decay_raw = decay.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (
        Some(grad_cube),
        Some(v_cube),
        Some(q_cube),
        Some(k_cube),
        Some(raw_scores_cube),
        Some(decay_cube),
        Some(params_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_current_out_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(raw_scores_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(decay_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let grad_q_add = score_backward_grad_q_runtime::<WgpuRuntime>(
            grad_cube.clone(),
            v_cube.clone(),
            k_cube,
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_k_add = score_backward_grad_k_runtime::<WgpuRuntime>(
            grad_cube.clone(),
            v_cube.clone(),
            q_cube,
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_v_add = score_backward_grad_v_runtime::<WgpuRuntime>(
            grad_cube.clone(),
            raw_scores_cube.clone(),
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_da_add = score_backward_grad_da_runtime::<WgpuRuntime>(
            grad_cube,
            v_cube,
            raw_scores_cube,
            decay_cube,
            params_cube,
        );
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_q_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_k_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_v_add,
            )?)),
            BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_da_add,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (
        Some(grad_cube),
        Some(v_cube),
        Some(q_cube),
        Some(k_cube),
        Some(raw_scores_cube),
        Some(decay_cube),
        Some(params_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_current_out_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(raw_scores_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(decay_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let grad_q_add = score_backward_grad_q_runtime::<CudaRuntime>(
            grad_cube.clone(),
            v_cube.clone(),
            k_cube,
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_k_add = score_backward_grad_k_runtime::<CudaRuntime>(
            grad_cube.clone(),
            v_cube.clone(),
            q_cube,
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_v_add = score_backward_grad_v_runtime::<CudaRuntime>(
            grad_cube.clone(),
            raw_scores_cube.clone(),
            decay_cube.clone(),
            params_cube.clone(),
        );
        let grad_da_add = score_backward_grad_da_runtime::<CudaRuntime>(
            grad_cube,
            v_cube,
            raw_scores_cube,
            decay_cube,
            params_cube,
        );
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_q_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_k_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_v_add,
            )?)),
            BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_da_add,
            )?)),
        ));
    }

    None
}

pub(crate) fn try_fused_score_carry_backward<B: BackendTrait>(
    grad_current_out: BurnTensor<B, 4>,
    v_head: BurnTensor<B, 4>,
    q_head: BurnTensor<B, 4>,
    k_head: BurnTensor<B, 4>,
    raw_scores: BurnTensor<B, 4>,
    decay: BurnTensor<B, 4>,
    grad_ssm_carry: BurnTensor<B, 4>,
    weighted_scale: BurnTensor<B, 3>,
) -> Option<(
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 4>,
    BurnTensor<B, 3>,
)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, headdim] = grad_current_out.shape().dims::<4>();
    let d_state = q_head.shape().dims::<4>()[3];
    let params = fused_score_carry_backward_params::<B>(
        batch,
        heads,
        time,
        d_state,
        headdim,
        &grad_current_out.device(),
    );
    let grad_current_out_raw = grad_current_out.into_primitive().tensor();
    let v_raw = v_head.into_primitive().tensor();
    let q_raw = q_head.into_primitive().tensor();
    let k_raw = k_head.into_primitive().tensor();
    let raw_scores_raw = raw_scores.into_primitive().tensor();
    let decay_raw = decay.into_primitive().tensor();
    let grad_ssm_carry_raw = grad_ssm_carry.into_primitive().tensor();
    let weighted_scale_raw = weighted_scale.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (
        Some(grad_cube),
        Some(v_cube),
        Some(q_cube),
        Some(k_cube),
        Some(raw_scores_cube),
        Some(decay_cube),
        Some(grad_ssm_carry_cube),
        Some(weighted_scale_cube),
        Some(params_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_current_out_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(raw_scores_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(decay_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(grad_ssm_carry_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(weighted_scale_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let time = grad_cube.meta.shape.dims::<4>()[2];
        let (grad_q_add, grad_k_add, grad_v_add, grad_da_add) =
            if time <= FUSED_SCORE_CARRY_WGPU_MAX_TIME {
                fused_score_carry_backward_tiled_wgpu_runtime(
                    grad_cube,
                    v_cube,
                    q_cube,
                    k_cube,
                    raw_scores_cube,
                    decay_cube,
                    grad_ssm_carry_cube,
                    weighted_scale_cube,
                    params_cube,
                )
            } else {
                fused_score_carry_backward_runtime::<WgpuRuntime>(
                    grad_cube,
                    v_cube,
                    q_cube,
                    k_cube,
                    raw_scores_cube,
                    decay_cube,
                    grad_ssm_carry_cube,
                    weighted_scale_cube,
                    params_cube,
                )
            };
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_q_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_k_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_v_add,
            )?)),
            BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_da_add,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (
        Some(grad_cube),
        Some(v_cube),
        Some(q_cube),
        Some(k_cube),
        Some(raw_scores_cube),
        Some(decay_cube),
        Some(grad_ssm_carry_cube),
        Some(weighted_scale_cube),
        Some(params_cube),
    ) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_current_out_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(raw_scores_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(decay_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(grad_ssm_carry_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(weighted_scale_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let (grad_q_add, grad_k_add, grad_v_add, grad_da_add) =
            fused_score_carry_backward_runtime::<CudaRuntime>(
                grad_cube,
                v_cube,
                q_cube,
                k_cube,
                raw_scores_cube,
                decay_cube,
                grad_ssm_carry_cube,
                weighted_scale_cube,
                params_cube,
            );
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_q_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_k_add,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_v_add,
            )?)),
            BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                grad_da_add,
            )?)),
        ));
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_wgpu::CubeBackend;

    type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn assert_close<const D: usize>(
        lhs: BurnTensor<WgpuBackend, D>,
        rhs: BurnTensor<WgpuBackend, D>,
    ) {
        let lhs = lhs.into_data().to_vec::<f32>().expect("lhs vec");
        let rhs = rhs.into_data().to_vec::<f32>().expect("rhs vec");
        assert_eq!(lhs.len(), rhs.len());
        for (left, right) in lhs.into_iter().zip(rhs) {
            assert!((left - right).abs() < 1e-4, "left={left} right={right}");
        }
    }

    fn add_last_time_grad_bhl(
        values: BurnTensor<WgpuBackend, 3>,
        last_grad: BurnTensor<WgpuBackend, 2>,
    ) -> BurnTensor<WgpuBackend, 3> {
        let [batch, heads, time] = values.shape().dims::<3>();
        let updated_last = values
            .clone()
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads])
            + last_grad;
        values.slice_assign(
            [0..batch, 0..heads, time - 1..time],
            updated_last.reshape([batch, heads, 1]),
        )
    }

    #[test]
    fn reverse_cumsum_bhl_runtime_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        let values = BurnTensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..(2 * 3 * 5))
                    .map(|idx| idx as f32 * 0.1 - 0.3)
                    .collect::<Vec<_>>(),
                [2, 3, 5],
            ),
            &device,
        );
        let runtime = try_reverse_cumsum_bhl(values.clone()).expect("wgpu runtime");
        let reference = {
            let [batch, heads, time] = values.shape().dims::<3>();
            let device = values.device();
            let reverse_index =
                BurnTensor::<WgpuBackend, 1, burn::tensor::Int>::arange(0..time as i64, &device)
                    .mul_scalar(-1)
                    .add_scalar(time as i64 - 1)
                    .reshape([1, 1, time])
                    .repeat_dim(0, batch)
                    .repeat_dim(1, heads);
            let reversed = values.clone().gather(2, reverse_index.clone());
            reversed.cumsum(2).gather(2, reverse_index)
        };
        assert_close(runtime, reference);
    }

    #[test]
    fn reverse_cumsum_blhr_runtime_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        let values = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(2 * 4 * 3 * 6))
                    .map(|idx| idx as f32 * 0.05 - 0.2)
                    .collect::<Vec<_>>(),
                [2, 4, 3, 6],
            ),
            &device,
        );
        let runtime = try_reverse_cumsum_blhr(values.clone()).expect("wgpu runtime");
        let reference = {
            let [batch, time, heads, width] = values.shape().dims::<4>();
            let device = values.device();
            let reverse_index =
                BurnTensor::<WgpuBackend, 1, burn::tensor::Int>::arange(0..time as i64, &device)
                    .mul_scalar(-1)
                    .add_scalar(time as i64 - 1)
                    .reshape([1, time, 1, 1])
                    .repeat_dim(0, batch)
                    .repeat_dim(2, heads)
                    .repeat_dim(3, width);
            let reversed = values.clone().gather(1, reverse_index.clone());
            reversed.cumsum(1).gather(1, reverse_index)
        };
        assert_close(runtime, reference);
    }

    #[test]
    fn carry_backward_runtime_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        let batch = 2;
        let heads = 3;
        let time = 5;
        let headdim = 4;
        let d_state = 2;
        let grad_ssm_carry = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * headdim * d_state))
                    .map(|idx| idx as f32 * 0.03 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, heads, headdim, d_state],
            ),
            &device,
        );
        let k_head = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * d_state))
                    .map(|idx| idx as f32 * 0.02 - 0.1)
                    .collect::<Vec<_>>(),
                [batch, heads, time, d_state],
            ),
            &device,
        );
        let v_head = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * headdim))
                    .map(|idx| idx as f32 * 0.01 - 0.15)
                    .collect::<Vec<_>>(),
                [batch, heads, time, headdim],
            ),
            &device,
        );
        let weighted_scale = BurnTensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * heads * time))
                    .map(|idx| idx as f32 * 0.015 + 0.5)
                    .collect::<Vec<_>>(),
                [batch, heads, time],
            ),
            &device,
        );
        let (grad_k_add, grad_v_add, grad_da_scale) = try_carry_backward(
            grad_ssm_carry.clone(),
            k_head.clone(),
            v_head.clone(),
            weighted_scale.clone(),
        )
        .expect("wgpu carry runtime");

        let reference_grad_weighted_v_t = grad_ssm_carry
            .clone()
            .matmul(k_head.clone().swap_dims(2, 3));
        let reference_grad_weighted_v = reference_grad_weighted_v_t.swap_dims(2, 3);
        let reference_grad_k_add = (v_head.clone()
            * weighted_scale.clone().reshape([batch, heads, time, 1]))
        .matmul(grad_ssm_carry.clone());
        let reference_grad_v_add = reference_grad_weighted_v.clone()
            * weighted_scale.clone().reshape([batch, heads, time, 1]);
        let reference_grad_da_scale = (reference_grad_weighted_v * v_head.clone())
            .sum_dim(3)
            .reshape([batch, heads, time]);

        assert_close(grad_k_add, reference_grad_k_add);
        assert_close(grad_v_add, reference_grad_v_add);
        assert_close(grad_da_scale, reference_grad_da_scale);
    }

    #[test]
    fn fused_score_carry_backward_runtime_matches_reference_on_wgpu() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        let batch = 1;
        let heads = 2;
        let time = 4;
        let headdim = 4;
        let d_state = 3;

        let grad_current_out = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * headdim))
                    .map(|idx| idx as f32 * 0.02 - 0.2)
                    .collect::<Vec<_>>(),
                [batch, heads, time, headdim],
            ),
            &device,
        );
        let v_head = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * headdim))
                    .map(|idx| idx as f32 * 0.015 - 0.15)
                    .collect::<Vec<_>>(),
                [batch, heads, time, headdim],
            ),
            &device,
        );
        let q_head = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * d_state))
                    .map(|idx| idx as f32 * 0.017 - 0.1)
                    .collect::<Vec<_>>(),
                [batch, heads, time, d_state],
            ),
            &device,
        );
        let k_head = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * d_state))
                    .map(|idx| idx as f32 * 0.019 - 0.12)
                    .collect::<Vec<_>>(),
                [batch, heads, time, d_state],
            ),
            &device,
        );
        let raw_scores = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * time))
                    .map(|idx| idx as f32 * 0.01 - 0.18)
                    .collect::<Vec<_>>(),
                [batch, heads, time, time],
            ),
            &device,
        )
        .tril(-1);
        let decay = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * time * time))
                    .map(|idx| 0.5 + (idx % 7) as f32 * 0.03)
                    .collect::<Vec<_>>(),
                [batch, heads, time, time],
            ),
            &device,
        )
        .tril(-1);
        let grad_ssm_carry = BurnTensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..(batch * heads * headdim * d_state))
                    .map(|idx| idx as f32 * 0.011 - 0.08)
                    .collect::<Vec<_>>(),
                [batch, heads, headdim, d_state],
            ),
            &device,
        );
        let weighted_scale = BurnTensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..(batch * heads * time))
                    .map(|idx| 0.55 + idx as f32 * 0.01)
                    .collect::<Vec<_>>(),
                [batch, heads, time],
            ),
            &device,
        );

        let (grad_q_add, grad_k_add, grad_v_add, grad_da_add) = try_fused_score_carry_backward(
            grad_current_out.clone(),
            v_head.clone(),
            q_head.clone(),
            k_head.clone(),
            raw_scores.clone(),
            decay.clone(),
            grad_ssm_carry.clone(),
            weighted_scale.clone(),
        )
        .expect("wgpu fused score/carry runtime");

        let grad_tril_scores = grad_current_out
            .clone()
            .matmul(v_head.clone().swap_dims(2, 3));
        let grad_v_current_add = (raw_scores.clone() * decay.clone())
            .tril(-1)
            .swap_dims(2, 3)
            .matmul(grad_current_out.clone());
        let grad_current_scores = grad_tril_scores.tril(-1);
        let grad_raw_scores = grad_current_scores.clone() * decay.clone();
        let grad_decay = grad_current_scores.clone() * raw_scores.clone();

        let reference_grad_q_add = grad_raw_scores.clone().matmul(k_head.clone());
        let mut reference_grad_k_add = grad_raw_scores
            .clone()
            .swap_dims(2, 3)
            .matmul(q_head.clone());
        let grad_diff = grad_decay.clone() * decay.clone();
        let reference_grad_da_score = grad_diff.clone().sum_dim(3).reshape([batch, heads, time])
            - grad_diff.sum_dim(2).reshape([batch, heads, time]);

        let weighted_v = v_head.clone() * weighted_scale.clone().reshape([batch, heads, time, 1]);
        let grad_weighted_v_t = grad_ssm_carry
            .clone()
            .matmul(k_head.clone().swap_dims(2, 3));
        let grad_weighted_v = grad_weighted_v_t.swap_dims(2, 3);
        reference_grad_k_add = reference_grad_k_add + weighted_v.matmul(grad_ssm_carry.clone());
        let reference_grad_v_add = grad_v_current_add
            + grad_weighted_v.clone() * weighted_scale.clone().reshape([batch, heads, time, 1]);
        let grad_weighted_scale = (grad_weighted_v * v_head.clone())
            .sum_dim(3)
            .reshape([batch, heads, time]);
        let reference_grad_da_add = add_last_time_grad_bhl(
            reference_grad_da_score - grad_weighted_scale.clone() * weighted_scale.clone(),
            (grad_weighted_scale * weighted_scale.clone())
                .sum_dim(2)
                .reshape([batch, heads]),
        );

        assert_close(grad_q_add, reference_grad_q_add);
        assert_close(grad_k_add, reference_grad_k_add);
        assert_close(grad_v_add, reference_grad_v_add);
        assert_close(grad_da_add, reference_grad_da_add);
    }
}
