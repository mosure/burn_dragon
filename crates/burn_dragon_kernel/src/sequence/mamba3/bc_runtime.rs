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
const BC_PARAMS_LEN: usize = 6;
const BC_WGPU_WORKGROUP_X: u32 = 64;
#[cfg(feature = "cuda")]
const BC_CUDA_WORKGROUP_X: u32 = 128;

pub(crate) struct Mamba3BcWgpuForwardOutput {
    pub(crate) expanded: CubeTensor<WgpuRuntime>,
    pub(crate) inv_rms: CubeTensor<WgpuRuntime>,
}

pub(crate) struct Mamba3BcWgpuBackwardOutput {
    pub(crate) grad_input: CubeTensor<WgpuRuntime>,
    pub(crate) grad_weight_contrib: CubeTensor<WgpuRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba3BcCudaForwardOutput {
    pub(crate) expanded: CubeTensor<CudaRuntime>,
    pub(crate) inv_rms: CubeTensor<CudaRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba3BcCudaBackwardOutput {
    pub(crate) grad_input: CubeTensor<CudaRuntime>,
    pub(crate) grad_weight_contrib: CubeTensor<CudaRuntime>,
}

pub(crate) fn fused_mamba3_bc_forward_wgpu(
    grouped: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    bias: CubeTensor<WgpuRuntime>,
    nheads: usize,
    eps: f32,
) -> Mamba3BcWgpuForwardOutput {
    let grouped = into_contiguous(grouped);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);
    let [batch, time, ngroups, d_state] = grouped.meta.shape.dims::<4>();
    assert_eq!(nheads % ngroups, 0);
    let heads_per_group = nheads / ngroups;
    let client = grouped.client.clone();
    let device = grouped.device.clone();

    let expanded = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, d_state]),
    );
    let inv_rms = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups]),
    );
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            d_state as f32,
            eps,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(BC_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, ngroups as u32, (batch * time) as u32);

    unsafe {
        let _ = mamba3_bc_forward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            grouped.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            bias.clone().into_tensor_arg(),
            expanded.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            BC_WGPU_WORKGROUP_X as usize,
        );
    }

    Mamba3BcWgpuForwardOutput { expanded, inv_rms }
}

pub(crate) fn fused_mamba3_bc_backward_wgpu(
    grouped: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    grad_expanded: CubeTensor<WgpuRuntime>,
    inv_rms: CubeTensor<WgpuRuntime>,
    nheads: usize,
) -> Mamba3BcWgpuBackwardOutput {
    let grouped = into_contiguous(grouped);
    let weight = into_contiguous(weight);
    let grad_expanded = into_contiguous(grad_expanded);
    let inv_rms = into_contiguous(inv_rms);
    let [batch, time, ngroups, d_state] = grouped.meta.shape.dims::<4>();
    assert_eq!(nheads % ngroups, 0);
    let heads_per_group = nheads / ngroups;
    let client = grouped.client.clone();
    let device = grouped.device.clone();

    let grad_input = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups, d_state]),
    );
    let grad_weight_contrib = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups, d_state]),
    );
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            d_state as f32,
            0.0,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(BC_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, ngroups as u32, (batch * time) as u32);

    unsafe {
        let _ = mamba3_bc_backward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            grouped.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            grad_expanded.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            grad_input.clone().into_tensor_arg(),
            grad_weight_contrib.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            BC_WGPU_WORKGROUP_X as usize,
        );
    }

    Mamba3BcWgpuBackwardOutput {
        grad_input,
        grad_weight_contrib,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba3_bc_forward_cuda(
    grouped: CubeTensor<CudaRuntime>,
    weight: CubeTensor<CudaRuntime>,
    bias: CubeTensor<CudaRuntime>,
    nheads: usize,
    eps: f32,
) -> Mamba3BcCudaForwardOutput {
    let grouped = into_contiguous(grouped);
    let weight = into_contiguous(weight);
    let bias = into_contiguous(bias);
    let [batch, time, ngroups, d_state] = grouped.meta.shape.dims::<4>();
    assert_eq!(nheads % ngroups, 0);
    let heads_per_group = nheads / ngroups;
    let client = grouped.client.clone();
    let device = grouped.device.clone();

    let expanded = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, nheads, d_state]),
    );
    let inv_rms = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups]),
    );
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            d_state as f32,
            eps,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(BC_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, ngroups as u32, (batch * time) as u32);

    unsafe {
        let _ = mamba3_bc_forward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            grouped.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            bias.clone().into_tensor_arg(),
            expanded.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            BC_CUDA_WORKGROUP_X as usize,
        );
    }

    Mamba3BcCudaForwardOutput { expanded, inv_rms }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba3_bc_backward_cuda(
    grouped: CubeTensor<CudaRuntime>,
    weight: CubeTensor<CudaRuntime>,
    grad_expanded: CubeTensor<CudaRuntime>,
    inv_rms: CubeTensor<CudaRuntime>,
    nheads: usize,
) -> Mamba3BcCudaBackwardOutput {
    let grouped = into_contiguous(grouped);
    let weight = into_contiguous(weight);
    let grad_expanded = into_contiguous(grad_expanded);
    let inv_rms = into_contiguous(inv_rms);
    let [batch, time, ngroups, d_state] = grouped.meta.shape.dims::<4>();
    assert_eq!(nheads % ngroups, 0);
    let heads_per_group = nheads / ngroups;
    let client = grouped.client.clone();
    let device = grouped.device.clone();

    let grad_input = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups, d_state]),
    );
    let grad_weight_contrib = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups, d_state]),
    );
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            d_state as f32,
            0.0,
        ],
    )
    .into_primitive()
    .tensor();
    let cube_dim = CubeDim::new_1d(BC_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, ngroups as u32, (batch * time) as u32);

    unsafe {
        let _ = mamba3_bc_backward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            grouped.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            grad_expanded.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            grad_input.clone().into_tensor_arg(),
            grad_weight_contrib.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            BC_CUDA_WORKGROUP_X as usize,
        );
    }

    Mamba3BcCudaBackwardOutput {
        grad_input,
        grad_weight_contrib,
    }
}

#[cfg(feature = "cuda")]
fn params_tensor(
    device: &<CudaCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; BC_PARAMS_LEN],
) -> BurnTensor<CudaCubeBackend, 1> {
    BurnTensor::<CudaCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [BC_PARAMS_LEN]),
        device,
    )
}

fn params_tensor_wgpu(
    device: &<WgpuCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; BC_PARAMS_LEN],
) -> BurnTensor<WgpuCubeBackend, 1> {
    BurnTensor::<WgpuCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [BC_PARAMS_LEN]),
        device,
    )
}

#[cube(launch_unchecked)]
fn mamba3_bc_forward_wgpu_kernel(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
    expanded: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    bc_forward_wgpu_impl(
        grouped,
        weight,
        bias,
        expanded,
        inv_rms,
        params,
        workgroup_size,
    );
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba3_bc_forward_cuda_kernel(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
    expanded: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    bc_forward_cuda_impl(
        grouped,
        weight,
        bias,
        expanded,
        inv_rms,
        params,
        workgroup_size,
    );
}

#[cube]
fn bc_forward_wgpu_impl(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
    expanded: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let ngroups = u32::cast_from(params[2]) as usize;
    let heads_per_group = u32::cast_from(params[3]) as usize;
    let d_state = u32::cast_from(params[4]) as usize;
    let eps = params[5];

    let bt = CUBE_POS_Z as usize;
    let g = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if bt >= batch * time || g >= ngroups {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let mut partials = SharedMemory::<f32>::new_aligned(workgroup_size, 1usize);
    let zero = f32::cast_from(0u32);
    let local_sum = if lane < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + lane * grouped.stride(3);
        let value = grouped[input_idx];
        value * value
    } else {
        zero
    };
    partials[lane] = local_sum;
    sync_cube();
    reduce_partials_wgpu(&mut partials, lane, workgroup_size);

    let one = f32::cast_from(1u32);
    let inv_rms_row = one / (partials[0] / f32::cast_from(d_state as u32) + eps).sqrt();
    if lane == 0usize {
        let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1) + g * inv_rms.stride(2);
        inv_rms[inv_idx] = inv_rms_row;
    }
    if lane < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + lane * grouped.stride(3);
        let base = grouped[input_idx] * inv_rms_row * weight[lane * weight.stride(0)];
        for head_offset in 0..heads_per_group {
            let h = g * heads_per_group + head_offset;
            let out_idx = b * expanded.stride(0)
                + t * expanded.stride(1)
                + h * expanded.stride(2)
                + lane * expanded.stride(3);
            let bias_idx = h * bias.stride(0) + lane * bias.stride(1);
            expanded[out_idx] = base + bias[bias_idx];
        }
    }
}

#[cfg(feature = "cuda")]
#[cube]
fn bc_forward_cuda_impl(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    bias: &Tensor<f32>,
    expanded: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let ngroups = params[2] as usize;
    let heads_per_group = params[3] as usize;
    let d_state = params[4] as usize;
    let eps = params[5];

    let bt = CUBE_POS_Z as usize;
    let g = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if bt >= batch * time || g >= ngroups {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let mut partials = SharedMemory::<f32>::new(workgroup_size);
    let mut local_sum = 0.0;
    let mut idx = lane;
    while idx < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + idx * grouped.stride(3);
        let value = grouped[input_idx];
        local_sum += value * value;
        idx += workgroup_size;
    }
    partials[lane] = local_sum;
    sync_cube();
    reduce_partials_cuda(&mut partials, lane, workgroup_size);

    let inv_rms_row = 1.0 / f32::sqrt(partials[0] / d_state as f32 + eps);
    if lane == 0usize {
        let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1) + g * inv_rms.stride(2);
        inv_rms[inv_idx] = inv_rms_row;
    }
    sync_cube();

    let mut out_lane = lane;
    while out_lane < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + out_lane * grouped.stride(3);
        let base = grouped[input_idx] * inv_rms_row * weight[out_lane * weight.stride(0)];
        let mut head_offset = 0usize;
        while head_offset < heads_per_group {
            let h = g * heads_per_group + head_offset;
            let out_idx = b * expanded.stride(0)
                + t * expanded.stride(1)
                + h * expanded.stride(2)
                + out_lane * expanded.stride(3);
            let bias_idx = h * bias.stride(0) + out_lane * bias.stride(1);
            expanded[out_idx] = base + bias[bias_idx];
            head_offset += 1usize;
        }
        out_lane += workgroup_size;
    }
}

#[cube(launch_unchecked)]
fn mamba3_bc_backward_wgpu_kernel(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    grad_expanded: &Tensor<f32>,
    inv_rms: &Tensor<f32>,
    grad_input: &mut Tensor<f32>,
    grad_weight_contrib: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    bc_backward_wgpu_impl(
        grouped,
        weight,
        grad_expanded,
        inv_rms,
        grad_input,
        grad_weight_contrib,
        params,
        workgroup_size,
    );
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba3_bc_backward_cuda_kernel(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    grad_expanded: &Tensor<f32>,
    inv_rms: &Tensor<f32>,
    grad_input: &mut Tensor<f32>,
    grad_weight_contrib: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    bc_backward_cuda_impl(
        grouped,
        weight,
        grad_expanded,
        inv_rms,
        grad_input,
        grad_weight_contrib,
        params,
        workgroup_size,
    );
}

#[cube]
fn bc_backward_wgpu_impl(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    grad_expanded: &Tensor<f32>,
    inv_rms: &Tensor<f32>,
    grad_input: &mut Tensor<f32>,
    grad_weight_contrib: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let ngroups = u32::cast_from(params[2]) as usize;
    let heads_per_group = u32::cast_from(params[3]) as usize;
    let d_state = u32::cast_from(params[4]) as usize;

    let bt = CUBE_POS_Z as usize;
    let g = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if bt >= batch * time || g >= ngroups {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1) + g * inv_rms.stride(2);
    let inv_rms_row = inv_rms[inv_idx];
    let mut partials = SharedMemory::<f32>::new_aligned(workgroup_size, 1usize);
    let zero = f32::cast_from(0u32);
    let mut dot_local = zero;

    let mut grad_sum = zero;
    if lane < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + lane * grouped.stride(3);
        let value = grouped[input_idx];
        let normalized = value * inv_rms_row;
        for head_offset in 0..heads_per_group {
            let h = g * heads_per_group + head_offset;
            let grad_idx = b * grad_expanded.stride(0)
                + t * grad_expanded.stride(1)
                + h * grad_expanded.stride(2)
                + lane * grad_expanded.stride(3);
            grad_sum += grad_expanded[grad_idx];
        }
        let grad_normalized = grad_sum * weight[lane * weight.stride(0)];
        dot_local = grad_normalized * value;
        let grad_weight_idx = b * grad_weight_contrib.stride(0)
            + t * grad_weight_contrib.stride(1)
            + g * grad_weight_contrib.stride(2)
            + lane * grad_weight_contrib.stride(3);
        grad_weight_contrib[grad_weight_idx] = grad_sum * normalized;
    }

    partials[lane] = dot_local;
    sync_cube();
    reduce_partials_wgpu(&mut partials, lane, workgroup_size);

    if lane < d_state {
        let input_idx = b * grad_input.stride(0)
            + t * grad_input.stride(1)
            + g * grad_input.stride(2)
            + lane * grad_input.stride(3);
        let value = grouped[b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + lane * grouped.stride(3)];
        let grad_normalized = grad_sum * weight[lane * weight.stride(0)];
        grad_input[input_idx] = grad_normalized * inv_rms_row
            - value * partials[0] * inv_rms_row * inv_rms_row * inv_rms_row
                / f32::cast_from(d_state as u32);
    }
}

#[cfg(feature = "cuda")]
#[cube]
fn bc_backward_cuda_impl(
    grouped: &Tensor<f32>,
    weight: &Tensor<f32>,
    grad_expanded: &Tensor<f32>,
    inv_rms: &Tensor<f32>,
    grad_input: &mut Tensor<f32>,
    grad_weight_contrib: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let ngroups = params[2] as usize;
    let heads_per_group = params[3] as usize;
    let d_state = params[4] as usize;

    let bt = CUBE_POS_Z as usize;
    let g = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if bt >= batch * time || g >= ngroups {
        terminate!();
    }

    let b = bt / time;
    let t = bt % time;
    let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1) + g * inv_rms.stride(2);
    let inv_rms_row = inv_rms[inv_idx];
    let mut partials = SharedMemory::<f32>::new(workgroup_size);
    let mut dot_local = 0.0;

    let mut idx = lane;
    while idx < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + idx * grouped.stride(3);
        let value = grouped[input_idx];
        let normalized = value * inv_rms_row;
        let mut grad_sum = 0.0;
        let mut head_offset = 0usize;
        while head_offset < heads_per_group {
            let h = g * heads_per_group + head_offset;
            let grad_idx = b * grad_expanded.stride(0)
                + t * grad_expanded.stride(1)
                + h * grad_expanded.stride(2)
                + idx * grad_expanded.stride(3);
            grad_sum += grad_expanded[grad_idx];
            head_offset += 1usize;
        }
        let grad_normalized = grad_sum * weight[idx * weight.stride(0)];
        dot_local += grad_normalized * value;
        let grad_weight_idx = b * grad_weight_contrib.stride(0)
            + t * grad_weight_contrib.stride(1)
            + g * grad_weight_contrib.stride(2)
            + idx * grad_weight_contrib.stride(3);
        grad_weight_contrib[grad_weight_idx] = grad_sum * normalized;
        idx += workgroup_size;
    }

    partials[lane] = dot_local;
    sync_cube();
    reduce_partials_cuda(&mut partials, lane, workgroup_size);
    let dot = partials[0];
    sync_cube();

    let mut out_lane = lane;
    while out_lane < d_state {
        let input_idx = b * grouped.stride(0)
            + t * grouped.stride(1)
            + g * grouped.stride(2)
            + out_lane * grouped.stride(3);
        let value = grouped[input_idx];
        let mut grad_sum = 0.0;
        let mut head_offset = 0usize;
        while head_offset < heads_per_group {
            let h = g * heads_per_group + head_offset;
            let grad_idx = b * grad_expanded.stride(0)
                + t * grad_expanded.stride(1)
                + h * grad_expanded.stride(2)
                + out_lane * grad_expanded.stride(3);
            grad_sum += grad_expanded[grad_idx];
            head_offset += 1usize;
        }
        let grad_normalized = grad_sum * weight[out_lane * weight.stride(0)];
        let grad_idx = b * grad_input.stride(0)
            + t * grad_input.stride(1)
            + g * grad_input.stride(2)
            + out_lane * grad_input.stride(3);
        grad_input[grad_idx] = grad_normalized * inv_rms_row
            - value * dot * inv_rms_row * inv_rms_row * inv_rms_row / d_state as f32;
        out_lane += workgroup_size;
    }
}

#[cfg(feature = "cuda")]
#[cube]
fn reduce_partials_cuda(
    partials: &mut SharedMemory<f32>,
    lane: usize,
    #[comptime] workgroup_size: usize,
) {
    if comptime!(workgroup_size >= 128usize) {
        if lane < 64usize {
            let rhs = partials[lane + 64usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 64usize) {
        if lane < 32usize {
            let rhs = partials[lane + 32usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 32usize) {
        if lane < 16usize {
            let rhs = partials[lane + 16usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 16usize) {
        if lane < 8usize {
            let rhs = partials[lane + 8usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 8usize) {
        if lane < 4usize {
            let rhs = partials[lane + 4usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 4usize) {
        if lane < 2usize {
            let rhs = partials[lane + 2usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 2usize) {
        if lane < 1usize {
            let rhs = partials[lane + 1usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
}

#[cube]
fn reduce_partials_wgpu(
    partials: &mut SharedMemory<f32>,
    lane: usize,
    #[comptime] workgroup_size: usize,
) {
    if comptime!(workgroup_size >= 128usize) {
        if lane < 64usize {
            let rhs = partials[lane + 64usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 64usize) {
        if lane < 32usize {
            let rhs = partials[lane + 32usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 32usize) {
        if lane < 16usize {
            let rhs = partials[lane + 16usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 16usize) {
        if lane < 8usize {
            let rhs = partials[lane + 8usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 8usize) {
        if lane < 4usize {
            let rhs = partials[lane + 4usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 4usize) {
        if lane < 2usize {
            let rhs = partials[lane + 2usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
    if comptime!(workgroup_size >= 2usize) {
        if lane < 1usize {
            let rhs = partials[lane + 1usize];
            let lhs = partials[lane];
            partials[lane] = lhs + rhs;
        }
        sync_cube();
    }
}
