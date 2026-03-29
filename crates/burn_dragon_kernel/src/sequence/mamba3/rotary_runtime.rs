use burn::tensor::Tensor as BurnTensor;
use burn::tensor::{Shape, TensorData};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::cubecl::{self, prelude::*};
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_wgpu::CubeBackend;

type WgpuCubeBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

const ROTARY_PARAMS_LEN: usize = 5;
const ROTARY_WGPU_WORKGROUP_X: u32 = 64;
#[cfg(feature = "cuda")]
const ROTARY_CUDA_WORKGROUP_X: u32 = 128;

pub(crate) struct Mamba3RotaryWgpuForwardOutput {
    pub(crate) q_rot: CubeTensor<WgpuRuntime>,
    pub(crate) k_rot: CubeTensor<WgpuRuntime>,
}

pub(crate) struct Mamba3RotaryWgpuBackwardOutput {
    pub(crate) grad_q: CubeTensor<WgpuRuntime>,
    pub(crate) grad_k: CubeTensor<WgpuRuntime>,
    pub(crate) grad_angle: CubeTensor<WgpuRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba3RotaryCudaForwardOutput {
    pub(crate) q_rot: CubeTensor<CudaRuntime>,
    pub(crate) k_rot: CubeTensor<CudaRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba3RotaryCudaBackwardOutput {
    pub(crate) grad_q: CubeTensor<CudaRuntime>,
    pub(crate) grad_k: CubeTensor<CudaRuntime>,
    pub(crate) grad_angle: CubeTensor<CudaRuntime>,
}

pub(crate) fn fused_mamba3_rotary_forward_wgpu(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    angles: CubeTensor<WgpuRuntime>,
    num_rope_angles: usize,
) -> Mamba3RotaryWgpuForwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let client = q.client.clone();
    let device = q.device.clone();

    let q_rot = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let k_rot = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(ROTARY_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(width as u32, ROTARY_WGPU_WORKGROUP_X),
        nheads as u32,
        (batch * time) as u32,
    );

    unsafe {
        let _ = mamba3_rotary_forward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            angles.as_tensor_arg(1),
            q_rot.as_tensor_arg(1),
            k_rot.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    Mamba3RotaryWgpuForwardOutput { q_rot, k_rot }
}

pub(crate) fn fused_mamba3_rotary_backward_wgpu(
    q: CubeTensor<WgpuRuntime>,
    k: CubeTensor<WgpuRuntime>,
    angles: CubeTensor<WgpuRuntime>,
    grad_q_rot: CubeTensor<WgpuRuntime>,
    grad_k_rot: CubeTensor<WgpuRuntime>,
    num_rope_angles: usize,
) -> Mamba3RotaryWgpuBackwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let grad_q_rot = into_contiguous(grad_q_rot);
    let grad_k_rot = into_contiguous(grad_k_rot);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let client = q.client.clone();
    let device = q.device.clone();

    let grad_q = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_k = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_angle =
        BurnTensor::<WgpuCubeBackend, 4>::zeros([batch, time, nheads, num_rope_angles], &device)
            .into_primitive()
            .tensor();
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(ROTARY_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(width as u32, ROTARY_WGPU_WORKGROUP_X),
        nheads as u32,
        (batch * time) as u32,
    );

    unsafe {
        let _ = mamba3_rotary_backward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            angles.as_tensor_arg(1),
            grad_q_rot.as_tensor_arg(1),
            grad_k_rot.as_tensor_arg(1),
            grad_q.as_tensor_arg(1),
            grad_k.as_tensor_arg(1),
            grad_angle.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    Mamba3RotaryWgpuBackwardOutput {
        grad_q,
        grad_k,
        grad_angle,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba3_rotary_forward_cuda(
    q: CubeTensor<CudaRuntime>,
    k: CubeTensor<CudaRuntime>,
    angles: CubeTensor<CudaRuntime>,
    num_rope_angles: usize,
) -> Mamba3RotaryCudaForwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let client = q.client.clone();
    let device = q.device.clone();

    let q_rot = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let k_rot = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(ROTARY_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(width as u32, ROTARY_CUDA_WORKGROUP_X),
        nheads as u32,
        (batch * time) as u32,
    );

    unsafe {
        let _ = mamba3_rotary_forward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            angles.as_tensor_arg(1),
            q_rot.as_tensor_arg(1),
            k_rot.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    Mamba3RotaryCudaForwardOutput { q_rot, k_rot }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba3_rotary_backward_cuda(
    q: CubeTensor<CudaRuntime>,
    k: CubeTensor<CudaRuntime>,
    angles: CubeTensor<CudaRuntime>,
    grad_q_rot: CubeTensor<CudaRuntime>,
    grad_k_rot: CubeTensor<CudaRuntime>,
    num_rope_angles: usize,
) -> Mamba3RotaryCudaBackwardOutput {
    let q = into_contiguous(q);
    let k = into_contiguous(k);
    let angles = into_contiguous(angles);
    let grad_q_rot = into_contiguous(grad_q_rot);
    let grad_k_rot = into_contiguous(grad_k_rot);
    let [batch, time, nheads, width] = q.meta.shape.dims::<4>();
    let client = q.client.clone();
    let device = q.device.clone();

    let grad_q = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_k = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, width]),
    );
    let grad_angle =
        BurnTensor::<CudaCubeBackend, 4>::zeros([batch, time, nheads, num_rope_angles], &device)
            .into_primitive()
            .tensor();
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            nheads as f32,
            width as f32,
            num_rope_angles as f32,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(ROTARY_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(width as u32, ROTARY_CUDA_WORKGROUP_X),
        nheads as u32,
        (batch * time) as u32,
    );

    unsafe {
        let _ = mamba3_rotary_backward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            q.as_tensor_arg(1),
            k.as_tensor_arg(1),
            angles.as_tensor_arg(1),
            grad_q_rot.as_tensor_arg(1),
            grad_k_rot.as_tensor_arg(1),
            grad_q.as_tensor_arg(1),
            grad_k.as_tensor_arg(1),
            grad_angle.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    Mamba3RotaryCudaBackwardOutput {
        grad_q,
        grad_k,
        grad_angle,
    }
}

#[cfg(feature = "cuda")]
fn params_tensor(
    device: &<CudaCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; ROTARY_PARAMS_LEN],
) -> BurnTensor<CudaCubeBackend, 1> {
    BurnTensor::<CudaCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [ROTARY_PARAMS_LEN]),
        device,
    )
}

fn params_tensor_wgpu(
    device: &<WgpuCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; ROTARY_PARAMS_LEN],
) -> BurnTensor<WgpuCubeBackend, 1> {
    BurnTensor::<WgpuCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [ROTARY_PARAMS_LEN]),
        device,
    )
}

#[cube(launch_unchecked)]
fn mamba3_rotary_forward_wgpu_kernel(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    q_rot: &mut Tensor<Line<f32>>,
    k_rot: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    rotary_forward_impl(q, k, angles, q_rot, k_rot, params);
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba3_rotary_forward_cuda_kernel(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    q_rot: &mut Tensor<Line<f32>>,
    k_rot: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    rotary_forward_impl(q, k, angles, q_rot, k_rot, params);
}

#[cube]
fn rotary_forward_impl(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    q_rot: &mut Tensor<Line<f32>>,
    k_rot: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let nheads = u32::cast_from(params[2]) as usize;
    let width = u32::cast_from(params[3]) as usize;
    let num_rope_angles = u32::cast_from(params[4]) as usize;
    let rotary_dim = num_rope_angles * 2usize;

    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let h = CUBE_POS_Y as usize;
    let bt = CUBE_POS_Z as usize;
    if d >= width || h >= nheads || bt >= batch * time {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let q_idx = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + d * q.stride(3);
    let k_idx = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + d * k.stride(3);
    let q_out_idx =
        b * q_rot.stride(0) + t * q_rot.stride(1) + h * q_rot.stride(2) + d * q_rot.stride(3);
    let k_out_idx =
        b * k_rot.stride(0) + t * k_rot.stride(1) + h * k_rot.stride(2) + d * k_rot.stride(3);

    if d >= rotary_dim {
        q_rot[q_out_idx] = q[q_idx];
        k_rot[k_out_idx] = k[k_idx];
    } else {
        let pair_base = (d / 2usize) * 2usize;
        let angle_idx = d / 2usize;
        let angle_tensor_idx = b * angles.stride(0)
            + t * angles.stride(1)
            + h * angles.stride(2)
            + angle_idx * angles.stride(3);
        let angle = angles[angle_tensor_idx];
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        let q0_idx = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + pair_base * q.stride(3);
        let q1_idx = q0_idx + q.stride(3);
        let k0_idx = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + pair_base * k.stride(3);
        let k1_idx = k0_idx + k.stride(3);
        let q0 = q[q0_idx];
        let q1 = q[q1_idx];
        let k0 = k[k0_idx];
        let k1 = k[k1_idx];

        if d % 2usize == 0usize {
            q_rot[q_out_idx] = q0 * cos_val - q1 * sin_val;
            k_rot[k_out_idx] = k0 * cos_val - k1 * sin_val;
        } else {
            q_rot[q_out_idx] = q0 * sin_val + q1 * cos_val;
            k_rot[k_out_idx] = k0 * sin_val + k1 * cos_val;
        }
    }
}

#[cube(launch_unchecked)]
fn mamba3_rotary_backward_wgpu_kernel(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    grad_q_rot: &Tensor<Line<f32>>,
    grad_k_rot: &Tensor<Line<f32>>,
    grad_q: &mut Tensor<Line<f32>>,
    grad_k: &mut Tensor<Line<f32>>,
    grad_angle: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    rotary_backward_impl(
        q, k, angles, grad_q_rot, grad_k_rot, grad_q, grad_k, grad_angle, params,
    );
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba3_rotary_backward_cuda_kernel(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    grad_q_rot: &Tensor<Line<f32>>,
    grad_k_rot: &Tensor<Line<f32>>,
    grad_q: &mut Tensor<Line<f32>>,
    grad_k: &mut Tensor<Line<f32>>,
    grad_angle: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    rotary_backward_impl(
        q, k, angles, grad_q_rot, grad_k_rot, grad_q, grad_k, grad_angle, params,
    );
}

#[cube]
fn rotary_backward_impl(
    q: &Tensor<Line<f32>>,
    k: &Tensor<Line<f32>>,
    angles: &Tensor<Line<f32>>,
    grad_q_rot: &Tensor<Line<f32>>,
    grad_k_rot: &Tensor<Line<f32>>,
    grad_q: &mut Tensor<Line<f32>>,
    grad_k: &mut Tensor<Line<f32>>,
    grad_angle: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let nheads = u32::cast_from(params[2]) as usize;
    let width = u32::cast_from(params[3]) as usize;
    let num_rope_angles = u32::cast_from(params[4]) as usize;
    let rotary_dim = num_rope_angles * 2usize;

    let d = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    let h = CUBE_POS_Y as usize;
    let bt = CUBE_POS_Z as usize;
    if d >= width || h >= nheads || bt >= batch * time {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let gq_rot_idx = b * grad_q_rot.stride(0)
        + t * grad_q_rot.stride(1)
        + h * grad_q_rot.stride(2)
        + d * grad_q_rot.stride(3);
    let gk_rot_idx = b * grad_k_rot.stride(0)
        + t * grad_k_rot.stride(1)
        + h * grad_k_rot.stride(2)
        + d * grad_k_rot.stride(3);
    let gq_idx =
        b * grad_q.stride(0) + t * grad_q.stride(1) + h * grad_q.stride(2) + d * grad_q.stride(3);
    let gk_idx =
        b * grad_k.stride(0) + t * grad_k.stride(1) + h * grad_k.stride(2) + d * grad_k.stride(3);

    if d >= rotary_dim {
        grad_q[gq_idx] = grad_q_rot[gq_rot_idx];
        grad_k[gk_idx] = grad_k_rot[gk_rot_idx];
    } else {
        let pair_base = (d / 2usize) * 2usize;
        let angle_idx = d / 2usize;
        let angle_tensor_idx = b * angles.stride(0)
            + t * angles.stride(1)
            + h * angles.stride(2)
            + angle_idx * angles.stride(3);
        let grad_angle_idx = b * grad_angle.stride(0)
            + t * grad_angle.stride(1)
            + h * grad_angle.stride(2)
            + angle_idx * grad_angle.stride(3);
        let angle = angles[angle_tensor_idx];
        let cos_val = angle.cos();
        let sin_val = angle.sin();

        let q0_idx = b * q.stride(0) + t * q.stride(1) + h * q.stride(2) + pair_base * q.stride(3);
        let q1_idx = q0_idx + q.stride(3);
        let k0_idx = b * k.stride(0) + t * k.stride(1) + h * k.stride(2) + pair_base * k.stride(3);
        let k1_idx = k0_idx + k.stride(3);
        let gq0_idx = b * grad_q_rot.stride(0)
            + t * grad_q_rot.stride(1)
            + h * grad_q_rot.stride(2)
            + pair_base * grad_q_rot.stride(3);
        let gq1_idx = gq0_idx + grad_q_rot.stride(3);
        let gk0_idx = b * grad_k_rot.stride(0)
            + t * grad_k_rot.stride(1)
            + h * grad_k_rot.stride(2)
            + pair_base * grad_k_rot.stride(3);
        let gk1_idx = gk0_idx + grad_k_rot.stride(3);

        let q0 = q[q0_idx];
        let q1 = q[q1_idx];
        let k0 = k[k0_idx];
        let k1 = k[k1_idx];
        let gq0 = grad_q_rot[gq0_idx];
        let gq1 = grad_q_rot[gq1_idx];
        let gk0 = grad_k_rot[gk0_idx];
        let gk1 = grad_k_rot[gk1_idx];

        if d % 2usize == 0usize {
            grad_q[gq_idx] = gq0 * cos_val + gq1 * sin_val;
            grad_k[gk_idx] = gk0 * cos_val + gk1 * sin_val;
            grad_angle[grad_angle_idx] = (-(gq0 * q0 + gq1 * q1) * sin_val
                + (gq1 * q0 - gq0 * q1) * cos_val)
                + (-(gk0 * k0 + gk1 * k1) * sin_val + (gk1 * k0 - gk0 * k1) * cos_val);
        } else {
            grad_q[gq_idx] = gq1 * cos_val - gq0 * sin_val;
            grad_k[gk_idx] = gk1 * cos_val - gk0 * sin_val;
        }
    }
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor.max(1))
}
