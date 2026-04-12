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
#[cfg(test)]
use std::sync::atomic::{AtomicI8, Ordering};

const CURRENT_SCORE_FORWARD_PARAM_LEN: usize = 5;
const STATE_UPDATE_FORWARD_PARAM_LEN: usize = 5;
const CURRENT_SCORE_FORWARD_WORKGROUP_X: u32 = 64;
const STATE_UPDATE_FORWARD_WORKGROUP_X: u32 = 64;
const CURRENT_SCORE_FORWARD_WGPU_MAX_TIME: usize = 128;

#[cfg(test)]
static WGPU_TILED_CURRENT_SCORE_RUNTIME_OVERRIDE: AtomicI8 = AtomicI8::new(-1);

#[cube(launch)]
fn current_score_forward_kernel(
    q_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    v_head: &Tensor<f32>,
    da_prefix: &Tensor<f32>,
    current_out: &mut Tensor<f32>,
    raw_scores: &mut Tensor<f32>,
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
    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || d >= headdim {
        terminate!();
    }

    let da_t =
        da_prefix[b * da_prefix.stride(0) + h * da_prefix.stride(1) + t * da_prefix.stride(2)];
    let mut out_acc = f32::cast_from(0u32);

    let mut u = 0usize;
    while u < t {
        let mut raw_score = f32::cast_from(0u32);
        let mut l = 0usize;
        while l < d_state {
            let q_index = b * q_head.stride(0)
                + h * q_head.stride(1)
                + t * q_head.stride(2)
                + l * q_head.stride(3);
            let k_index = b * k_head.stride(0)
                + h * k_head.stride(1)
                + u * k_head.stride(2)
                + l * k_head.stride(3);
            raw_score += q_head[q_index] * k_head[k_index];
            l += 1usize;
        }

        if d == 0usize {
            let raw_out = b * raw_scores.stride(0)
                + h * raw_scores.stride(1)
                + t * raw_scores.stride(2)
                + u * raw_scores.stride(3);
            raw_scores[raw_out] = raw_score;
        }

        let da_u =
            da_prefix[b * da_prefix.stride(0) + h * da_prefix.stride(1) + u * da_prefix.stride(2)];
        let zero = f32::cast_from(0u32);
        let diff = da_t - da_u;
        let clamped = if diff > zero { zero } else { diff };
        let decay = f32::exp(clamped);
        let v_index = b * v_head.stride(0)
            + h * v_head.stride(1)
            + u * v_head.stride(2)
            + d * v_head.stride(3);
        out_acc += raw_score * decay * v_head[v_index];
        u += 1usize;
    }

    let out_index = b * current_out.stride(0)
        + h * current_out.stride(1)
        + t * current_out.stride(2)
        + d * current_out.stride(3);
    current_out[out_index] = out_acc;
}

#[cube(launch)]
fn current_score_forward_tiled_wgpu_kernel(
    q_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    v_head: &Tensor<f32>,
    da_prefix: &Tensor<f32>,
    current_out: &mut Tensor<f32>,
    raw_scores: &mut Tensor<f32>,
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
    let mut row_scores = SharedMemory::<f32>::new_aligned(max_time, 1usize);
    let da_t =
        da_prefix[b * da_prefix.stride(0) + h * da_prefix.stride(1) + t * da_prefix.stride(2)];

    let mut u = lane;
    while u < time {
        let raw_out = b * raw_scores.stride(0)
            + h * raw_scores.stride(1)
            + t * raw_scores.stride(2)
            + u * raw_scores.stride(3);
        if u < t {
            let mut raw_score = zero;
            let mut l = 0usize;
            while l < d_state {
                let q_index = b * q_head.stride(0)
                    + h * q_head.stride(1)
                    + t * q_head.stride(2)
                    + l * q_head.stride(3);
                let k_index = b * k_head.stride(0)
                    + h * k_head.stride(1)
                    + u * k_head.stride(2)
                    + l * k_head.stride(3);
                raw_score += q_head[q_index] * k_head[k_index];
                l += 1usize;
            }
            let da_u = da_prefix
                [b * da_prefix.stride(0) + h * da_prefix.stride(1) + u * da_prefix.stride(2)];
            let diff = da_t - da_u;
            let clamped = if diff > zero { zero } else { diff };
            row_scores[u] = raw_score * f32::exp(clamped);
            raw_scores[raw_out] = raw_score;
        } else {
            row_scores[u] = zero;
            raw_scores[raw_out] = zero;
        }
        u += CUBE_DIM_X as usize;
    }

    sync_cube();

    let mut d = lane;
    while d < headdim {
        let mut out_acc = zero;
        let mut col = 0usize;
        while col < t {
            let v_index = b * v_head.stride(0)
                + h * v_head.stride(1)
                + col * v_head.stride(2)
                + d * v_head.stride(3);
            out_acc += row_scores[col] * v_head[v_index];
            col += 1usize;
        }
        let out_index = b * current_out.stride(0)
            + h * current_out.stride(1)
            + t * current_out.stride(2)
            + d * current_out.stride(3);
        current_out[out_index] = out_acc;
        d += CUBE_DIM_X as usize;
    }
}

fn current_score_forward_params<B: BackendTrait>(
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
            [CURRENT_SCORE_FORWARD_PARAM_LEN],
        ),
        device,
    )
}

#[cube(launch)]
fn state_update_forward_kernel(
    state_tilde: &Tensor<f32>,
    da_prefix: &Tensor<f32>,
    v_head: &Tensor<f32>,
    k_head: &Tensor<f32>,
    ssm_state: &mut Tensor<f32>,
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
    let d = CUBE_POS_Y as usize;
    let s = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || d >= headdim || s >= d_state {
        terminate!();
    }

    let last_index =
        b * da_prefix.stride(0) + h * da_prefix.stride(1) + (time - 1usize) * da_prefix.stride(2);
    let da_last = da_prefix[last_index];
    let scale_last = f32::exp(da_last);

    let state_index = b * state_tilde.stride(0)
        + h * state_tilde.stride(1)
        + d * state_tilde.stride(2)
        + s * state_tilde.stride(3);
    let mut acc = state_tilde[state_index] * scale_last;

    let mut t = 0usize;
    while t < time {
        let da_index = b * da_prefix.stride(0) + h * da_prefix.stride(1) + t * da_prefix.stride(2);
        let scale = f32::exp(da_last - da_prefix[da_index]);
        let v_index = b * v_head.stride(0)
            + h * v_head.stride(1)
            + t * v_head.stride(2)
            + d * v_head.stride(3);
        let k_index = b * k_head.stride(0)
            + h * k_head.stride(1)
            + t * k_head.stride(2)
            + s * k_head.stride(3);
        acc += (v_head[v_index] * scale) * k_head[k_index];
        t += 1usize;
    }

    let out_index = b * ssm_state.stride(0)
        + h * ssm_state.stride(1)
        + d * ssm_state.stride(2)
        + s * ssm_state.stride(3);
    ssm_state[out_index] = acc;
}

fn state_update_forward_params<B: BackendTrait>(
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
            [STATE_UPDATE_FORWARD_PARAM_LEN],
        ),
        device,
    )
}

fn current_score_forward_runtime<R: CubeRuntime>(
    q_head: CubeTensor<R>,
    k_head: CubeTensor<R>,
    v_head: CubeTensor<R>,
    da_prefix: CubeTensor<R>,
    params: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let q_head = into_contiguous(q_head);
    let k_head = into_contiguous(k_head);
    let v_head = into_contiguous(v_head);
    let da_prefix = into_contiguous(da_prefix);
    let params = into_contiguous(params);

    let [batch, heads, time, _d_state] = q_head.meta.shape.dims::<4>();
    let headdim = v_head.meta.shape.dims::<4>()[3];
    let client = q_head.client.clone();
    let device = q_head.device.clone();
    let current_out = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, headdim]),
    );
    let raw_scores = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, time]),
    );
    let cube_dim = CubeDim::new_1d(CURRENT_SCORE_FORWARD_WORKGROUP_X);
    let count_x = headdim.div_ceil(CURRENT_SCORE_FORWARD_WORKGROUP_X as usize) as u32;
    let count_y = time as u32;
    let count_z = (batch * heads) as u32;
    let _ = current_score_forward_kernel::launch::<R>(
        &client,
        CubeCount::Static(count_x, count_y, count_z),
        cube_dim,
        q_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        v_head.into_tensor_arg(),
        da_prefix.into_tensor_arg(),
        current_out.clone().into_tensor_arg(),
        raw_scores.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    (current_out, raw_scores)
}

fn current_score_forward_tiled_wgpu_runtime(
    q_head: CubeTensor<WgpuRuntime>,
    k_head: CubeTensor<WgpuRuntime>,
    v_head: CubeTensor<WgpuRuntime>,
    da_prefix: CubeTensor<WgpuRuntime>,
    params: CubeTensor<WgpuRuntime>,
) -> (CubeTensor<WgpuRuntime>, CubeTensor<WgpuRuntime>) {
    let q_head = into_contiguous(q_head);
    let k_head = into_contiguous(k_head);
    let v_head = into_contiguous(v_head);
    let da_prefix = into_contiguous(da_prefix);
    let params = into_contiguous(params);

    let [batch, heads, time, _d_state] = q_head.meta.shape.dims::<4>();
    let headdim = v_head.meta.shape.dims::<4>()[3];
    let client = q_head.client.clone();
    let device = q_head.device.clone();
    let current_out = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, headdim]),
    );
    let raw_scores = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, time]),
    );
    let cube_dim = CubeDim::new_1d(CURRENT_SCORE_FORWARD_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, time as u32, (batch * heads) as u32);
    let _ = current_score_forward_tiled_wgpu_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        q_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        v_head.into_tensor_arg(),
        da_prefix.into_tensor_arg(),
        current_out.clone().into_tensor_arg(),
        raw_scores.clone().into_tensor_arg(),
        params.into_tensor_arg(),
        CURRENT_SCORE_FORWARD_WGPU_MAX_TIME,
    );
    (current_out, raw_scores)
}

fn use_wgpu_tiled_current_score_runtime(time: usize, d_state: usize, headdim: usize) -> bool {
    #[cfg(test)]
    match WGPU_TILED_CURRENT_SCORE_RUNTIME_OVERRIDE.load(Ordering::Relaxed) {
        0 => return false,
        1 => return true,
        _ => {}
    }

    match std::env::var("BURN_DRAGON_MAMBA3_WGPU_TILED_CURRENT_SCORE_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("0") | Some("false") | Some("FALSE") | Some("off") | Some("OFF") => false,
        Some(_) => true,
        None => time <= CURRENT_SCORE_FORWARD_WGPU_MAX_TIME && d_state >= 32 && headdim <= 128,
    }
}

fn state_update_forward_runtime<R: CubeRuntime>(
    state_tilde: CubeTensor<R>,
    da_prefix: CubeTensor<R>,
    v_head: CubeTensor<R>,
    k_head: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let state_tilde = into_contiguous(state_tilde);
    let da_prefix = into_contiguous(da_prefix);
    let v_head = into_contiguous(v_head);
    let k_head = into_contiguous(k_head);
    let params = into_contiguous(params);

    let [batch, heads, headdim, d_state] = state_tilde.meta.shape.dims::<4>();
    let client = state_tilde.client.clone();
    let device = state_tilde.device.clone();
    let ssm_state = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, headdim, d_state]),
    );
    let cube_dim = CubeDim::new_1d(STATE_UPDATE_FORWARD_WORKGROUP_X);
    let count_x = d_state.div_ceil(STATE_UPDATE_FORWARD_WORKGROUP_X as usize) as u32;
    let count_y = headdim as u32;
    let count_z = (batch * heads) as u32;
    let _ = state_update_forward_kernel::launch::<R>(
        &client,
        CubeCount::Static(count_x, count_y, count_z),
        cube_dim,
        state_tilde.into_tensor_arg(),
        da_prefix.into_tensor_arg(),
        v_head.into_tensor_arg(),
        k_head.into_tensor_arg(),
        ssm_state.clone().into_tensor_arg(),
        params.into_tensor_arg(),
    );
    ssm_state
}

pub(crate) fn try_current_score_forward<B: BackendTrait>(
    q_head: BurnTensor<B, 4>,
    k_head: BurnTensor<B, 4>,
    v_head: BurnTensor<B, 4>,
    da_prefix: BurnTensor<B, 3>,
) -> Option<(BurnTensor<B, 4>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, d_state] = q_head.shape().dims::<4>();
    let headdim = v_head.shape().dims::<4>()[3];
    let params =
        current_score_forward_params::<B>(batch, heads, time, d_state, headdim, &q_head.device());
    let q_raw = q_head.into_primitive().tensor();
    let k_raw = k_head.into_primitive().tensor();
    let v_raw = v_head.into_primitive().tensor();
    let da_raw = da_prefix.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (Some(q_cube), Some(k_cube), Some(v_cube), Some(da_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(da_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let [_, _, time, d_state] = q_cube.meta.shape.dims::<4>();
        let headdim = v_cube.meta.shape.dims::<4>()[3];
        let (current_out, raw_scores) =
            if use_wgpu_tiled_current_score_runtime(time, d_state, headdim) {
                current_score_forward_tiled_wgpu_runtime(
                    q_cube,
                    k_cube,
                    v_cube,
                    da_cube,
                    params_cube,
                )
            } else {
                current_score_forward_runtime::<WgpuRuntime>(
                    q_cube,
                    k_cube,
                    v_cube,
                    da_cube,
                    params_cube,
                )
            };
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                current_out,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                raw_scores,
            )?)),
        ));
    }

    #[cfg(feature = "cuda")]
    if let (Some(q_cube), Some(k_cube), Some(v_cube), Some(da_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(q_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(da_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let (current_out, raw_scores) = current_score_forward_runtime::<CudaRuntime>(
            q_cube,
            k_cube,
            v_cube,
            da_cube,
            params_cube,
        );
        return Some((
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                current_out,
            )?)),
            BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
                raw_scores,
            )?)),
        ));
    }

    None
}

pub(crate) fn try_state_update_forward<B: BackendTrait>(
    state_tilde: BurnTensor<B, 4>,
    da_prefix: BurnTensor<B, 3>,
    v_head: BurnTensor<B, 4>,
    k_head: BurnTensor<B, 4>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, headdim, d_state] = state_tilde.shape().dims::<4>();
    let time = da_prefix.shape().dims::<3>()[2];
    let params = state_update_forward_params::<B>(
        batch,
        heads,
        time,
        headdim,
        d_state,
        &state_tilde.device(),
    );
    let state_tilde_raw = state_tilde.into_primitive().tensor();
    let da_raw = da_prefix.into_primitive().tensor();
    let v_raw = v_head.into_primitive().tensor();
    let k_raw = k_head.into_primitive().tensor();
    let params_raw = params.into_primitive().tensor();

    if let (Some(state_tilde_cube), Some(da_cube), Some(v_cube), Some(k_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(state_tilde_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(da_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(params_raw.clone()),
    ) {
        let ssm_state = state_update_forward_runtime::<WgpuRuntime>(
            state_tilde_cube,
            da_cube,
            v_cube,
            k_cube,
            params_cube,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(ssm_state)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let (Some(state_tilde_cube), Some(da_cube), Some(v_cube), Some(k_cube), Some(params_cube)) = (
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(state_tilde_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(da_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(v_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(k_raw.clone()),
        try_cast_primitive::<B, CubeTensor<CudaRuntime>>(params_raw.clone()),
    ) {
        let ssm_state = state_update_forward_runtime::<CudaRuntime>(
            state_tilde_cube,
            da_cube,
            v_cube,
            k_cube,
            params_cube,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(ssm_state)?,
        )));
    }

    None
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Tensor, TensorData};
    use burn_wgpu::CubeBackend;
    use burn_wgpu::{RuntimeOptions, graphics};
    use std::sync::Once;

    type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    fn init_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn reference_current_score<B: BackendTrait>(
        q_head: BurnTensor<B, 4>,
        k_head: BurnTensor<B, 4>,
        v_head: BurnTensor<B, 4>,
        da_prefix: BurnTensor<B, 3>,
    ) -> (BurnTensor<B, 4>, BurnTensor<B, 4>) {
        let [batch, heads, time, _] = q_head.shape().dims::<4>();
        let raw_scores = q_head.clone().matmul(k_head.clone().swap_dims(2, 3));
        let decay = (da_prefix.clone().unsqueeze_dim::<4>(3)
            - da_prefix.clone().unsqueeze_dim::<4>(2))
        .clamp_max(0.0)
        .exp();
        let current_out = (raw_scores.clone() * decay).tril(-1).matmul(v_head);
        let lower_mask =
            Tensor::<B, 4>::ones([batch, heads, time, time], &q_head.device()).tril(-1);
        (current_out, raw_scores * lower_mask)
    }

    fn reference_state_update<B: BackendTrait>(
        state_tilde: BurnTensor<B, 4>,
        da_prefix: BurnTensor<B, 3>,
        v_head: BurnTensor<B, 4>,
        k_head: BurnTensor<B, 4>,
    ) -> BurnTensor<B, 4> {
        let [batch, heads, _headdim, _d_state] = state_tilde.shape().dims::<4>();
        let time = da_prefix.shape().dims::<3>()[2];
        let da_last = da_prefix
            .clone()
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads]);
        let weighted_v = v_head
            * (da_last.clone().unsqueeze_dim::<3>(2) - da_prefix.clone())
                .exp()
                .unsqueeze_dim::<4>(3);
        state_tilde * da_last.reshape([batch, heads, 1, 1]).exp()
            + weighted_v.swap_dims(2, 3).matmul(k_head)
    }

    fn assert_close<B: BackendTrait, const D: usize>(
        lhs: BurnTensor<B, D>,
        rhs: BurnTensor<B, D>,
        tol: f32,
    ) {
        let lhs = lhs.into_data().to_vec::<f32>().expect("lhs f32");
        let rhs = rhs.into_data().to_vec::<f32>().expect("rhs f32");
        for (l, r) in lhs.into_iter().zip(rhs.into_iter()) {
            assert!(
                (l - r).abs() <= tol,
                "expected |{l} - {r}| <= {tol}, got {}",
                (l - r).abs()
            );
        }
    }

    #[test]
    fn current_score_forward_runtime_matches_reference_on_wgpu() {
        let device = Default::default();
        init_runtime(&device);
        WGPU_TILED_CURRENT_SCORE_RUNTIME_OVERRIDE.store(1, Ordering::Relaxed);
        let q_head = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                [
                    1.0f32, 0.5, -0.2, 0.3, -0.4, 0.1, 0.6, -0.7, 0.9, -0.8, 0.2, 0.4,
                ]
                .to_vec(),
                [1, 1, 3, 4],
            ),
            &device,
        )
        .reshape([1, 1, 3, 4]);
        let k_head = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                [
                    0.3f32, -0.2, 0.7, 0.5, 0.1, 0.8, -0.6, 0.2, -0.5, 0.4, 0.9, -0.1,
                ]
                .to_vec(),
                [1, 1, 3, 4],
            ),
            &device,
        )
        .reshape([1, 1, 3, 4]);
        let v_head = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new([0.2f32, -0.3, 0.4, 0.6, -0.5, 0.8].to_vec(), [1, 1, 3, 2]),
            &device,
        )
        .reshape([1, 1, 3, 2]);
        let da_prefix = Tensor::<WgpuBackend, 3>::from_data(
            TensorData::new([0.1f32, -0.2, -0.6].to_vec(), [1, 1, 3]),
            &device,
        )
        .reshape([1, 1, 3]);

        let (current_out, raw_scores) = try_current_score_forward(
            q_head.clone(),
            k_head.clone(),
            v_head.clone(),
            da_prefix.clone(),
        )
        .expect("current score runtime");
        let (reference_out, reference_raw) =
            reference_current_score(q_head, k_head, v_head, da_prefix);

        assert_close(current_out, reference_out, 1.0e-4);
        assert_close(raw_scores, reference_raw, 1.0e-4);
        WGPU_TILED_CURRENT_SCORE_RUNTIME_OVERRIDE.store(-1, Ordering::Relaxed);
    }

    #[test]
    fn state_update_forward_runtime_matches_reference_on_wgpu() {
        let device = Default::default();
        init_runtime(&device);
        let state_tilde = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..24)
                    .map(|idx| ((idx % 41) as f32) / 41.0 - 0.25)
                    .collect::<Vec<_>>(),
                [1, 2, 4, 3],
            ),
            &device,
        );
        let da_prefix = Tensor::<WgpuBackend, 3>::from_data(
            TensorData::new(
                (0..12)
                    .map(|idx| ((idx % 17) as f32) / 17.0 - 0.3)
                    .collect::<Vec<_>>(),
                [1, 2, 6],
            ),
            &device,
        );
        let v_head = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..48)
                    .map(|idx| ((idx % 29) as f32) / 29.0 - 0.2)
                    .collect::<Vec<_>>(),
                [1, 2, 6, 4],
            ),
            &device,
        );
        let k_head = Tensor::<WgpuBackend, 4>::from_data(
            TensorData::new(
                (0..36)
                    .map(|idx| ((idx % 31) as f32) / 31.0 - 0.1)
                    .collect::<Vec<_>>(),
                [1, 2, 6, 3],
            ),
            &device,
        );

        let runtime = try_state_update_forward(
            state_tilde.clone(),
            da_prefix.clone(),
            v_head.clone(),
            k_head.clone(),
        )
        .expect("state update runtime");
        let reference = reference_state_update(state_tilde, da_prefix, v_head, k_head);

        assert_close(runtime, reference, 1.0e-4);
    }
}
