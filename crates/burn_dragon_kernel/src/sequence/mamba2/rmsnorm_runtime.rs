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
const RMSNORM_GATED_PARAMS_LEN: usize = 4;
const RMSNORM_GATED_WGPU_WORKGROUP_X: u32 = 64;
#[cfg(feature = "cuda")]
const RMSNORM_GATED_CUDA_WORKGROUP_X: u32 = 128;

pub(crate) struct Mamba2RmsnormGatedWgpuForwardOutput {
    pub(crate) gated: CubeTensor<WgpuRuntime>,
    pub(crate) inv_rms: CubeTensor<WgpuRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba2RmsnormGatedCudaForwardOutput {
    pub(crate) gated: CubeTensor<CudaRuntime>,
    pub(crate) inv_rms: CubeTensor<CudaRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba2RmsnormGatedCudaBackwardOutput {
    pub(crate) grad_y: CubeTensor<CudaRuntime>,
    pub(crate) grad_z: CubeTensor<CudaRuntime>,
    pub(crate) grad_weight: CubeTensor<CudaRuntime>,
}

pub(crate) fn fused_mamba2_rmsnorm_gated_forward_wgpu(
    y: CubeTensor<WgpuRuntime>,
    z: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    eps: f32,
) -> Mamba2RmsnormGatedWgpuForwardOutput {
    let y = into_contiguous(y);
    let z = into_contiguous(z);
    let weight = into_contiguous(weight);
    let [batch, time, width] = y.meta.shape.dims::<3>();
    let client = y.client.clone();
    let device = y.device.clone();

    let gated = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, width]),
    );
    let inv_rms =
        empty_device::<WgpuRuntime, f32>(client.clone(), device.clone(), Shape::new([batch, time]));
    let params = params_tensor_wgpu(&device, [batch as f32, time as f32, width as f32, eps])
        .into_primitive()
        .tensor();
    let cube_dim = CubeDim::new_1d(RMSNORM_GATED_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, time as u32, batch as u32);

    unsafe {
        let _ = mamba2_rmsnorm_gated_forward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            y.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            gated.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            RMSNORM_GATED_WGPU_WORKGROUP_X as usize,
        );
    }

    Mamba2RmsnormGatedWgpuForwardOutput { gated, inv_rms }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba2_rmsnorm_gated_forward_cuda(
    y: CubeTensor<CudaRuntime>,
    z: CubeTensor<CudaRuntime>,
    weight: CubeTensor<CudaRuntime>,
    eps: f32,
) -> Mamba2RmsnormGatedCudaForwardOutput {
    let y = into_contiguous(y);
    let z = into_contiguous(z);
    let weight = into_contiguous(weight);
    let [batch, time, width] = y.meta.shape.dims::<3>();
    let client = y.client.clone();
    let device = y.device.clone();

    let gated = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, width]),
    );
    let inv_rms =
        empty_device::<CudaRuntime, f32>(client.clone(), device.clone(), Shape::new([batch, time]));
    let params = params_tensor(&device, [batch as f32, time as f32, width as f32, eps])
        .into_primitive()
        .tensor();
    let cube_dim = CubeDim::new_1d(RMSNORM_GATED_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, time as u32, batch as u32);

    unsafe {
        let _ = mamba2_rmsnorm_gated_forward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            y.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            gated.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            RMSNORM_GATED_CUDA_WORKGROUP_X as usize,
        );
    }

    Mamba2RmsnormGatedCudaForwardOutput { gated, inv_rms }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba2_rmsnorm_gated_backward_cuda(
    y: CubeTensor<CudaRuntime>,
    z: CubeTensor<CudaRuntime>,
    weight: CubeTensor<CudaRuntime>,
    grad_output: CubeTensor<CudaRuntime>,
    inv_rms: CubeTensor<CudaRuntime>,
) -> Mamba2RmsnormGatedCudaBackwardOutput {
    let y = into_contiguous(y);
    let z = into_contiguous(z);
    let weight = into_contiguous(weight);
    let grad_output = into_contiguous(grad_output);
    let inv_rms = into_contiguous(inv_rms);
    let [batch, time, width] = y.meta.shape.dims::<3>();
    let client = y.client.clone();
    let device = y.device.clone();

    let grad_y = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, width]),
    );
    let grad_z = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, width]),
    );
    let grad_weight = BurnTensor::<CudaCubeBackend, 1>::zeros([width], &device)
        .into_primitive()
        .tensor();
    let params = params_tensor(&device, [batch as f32, time as f32, width as f32, 0.0])
        .into_primitive()
        .tensor();
    let cube_dim = CubeDim::new_1d(RMSNORM_GATED_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(1, time as u32, batch as u32);

    unsafe {
        let _ = mamba2_rmsnorm_gated_backward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            y.clone().into_tensor_arg(),
            z.clone().into_tensor_arg(),
            weight.clone().into_tensor_arg(),
            grad_output.clone().into_tensor_arg(),
            inv_rms.clone().into_tensor_arg(),
            grad_y.clone().into_tensor_arg(),
            grad_z.clone().into_tensor_arg(),
            grad_weight.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            RMSNORM_GATED_CUDA_WORKGROUP_X as usize,
        );
    }

    Mamba2RmsnormGatedCudaBackwardOutput {
        grad_y,
        grad_z,
        grad_weight,
    }
}

#[cfg(feature = "cuda")]
fn params_tensor(
    device: &<CudaCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; RMSNORM_GATED_PARAMS_LEN],
) -> BurnTensor<CudaCubeBackend, 1> {
    BurnTensor::<CudaCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [RMSNORM_GATED_PARAMS_LEN]),
        device,
    )
}

fn params_tensor_wgpu(
    device: &<WgpuCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; RMSNORM_GATED_PARAMS_LEN],
) -> BurnTensor<WgpuCubeBackend, 1> {
    BurnTensor::<WgpuCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [RMSNORM_GATED_PARAMS_LEN]),
        device,
    )
}

#[cube(launch_unchecked)]
fn mamba2_rmsnorm_gated_forward_wgpu_kernel(
    y: &Tensor<f32>,
    z: &Tensor<f32>,
    weight: &Tensor<f32>,
    gated: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let width = u32::cast_from(params[2]) as usize;
    let eps = params[3];

    let b = CUBE_POS_Z as usize;
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || t >= time {
        terminate!();
    }

    let mut partials = SharedMemory::<f32>::new_aligned(workgroup_size, 1usize);
    let mut local_sum = f32::cast_from(0u32);
    let mut idx = lane;
    while idx < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + idx * y.stride(2);
        let y_val = y[y_idx];
        local_sum += y_val * y_val;
        idx += workgroup_size;
    }
    partials[lane] = local_sum;
    sync_cube();

    reduce_partials_wgpu(&mut partials, lane, workgroup_size);

    let one = f32::cast_from(1u32);
    let inv_rms_row = one / (partials[0] / f32::cast_from(width as u32) + eps).sqrt();
    if lane == 0usize {
        let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1);
        inv_rms[inv_idx] = inv_rms_row;
    }
    sync_cube();

    let mut out_lane = lane;
    while out_lane < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + out_lane * y.stride(2);
        let z_idx = b * z.stride(0) + t * z.stride(1) + out_lane * z.stride(2);
        let w_idx = out_lane * weight.stride(0);
        let sigmoid = one / (one + (f32::cast_from(0u32) - z[z_idx]).exp());
        let out_idx = b * gated.stride(0) + t * gated.stride(1) + out_lane * gated.stride(2);
        gated[out_idx] = (y[y_idx] * inv_rms_row) * weight[w_idx] * (z[z_idx] * sigmoid);
        out_lane += workgroup_size;
    }
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba2_rmsnorm_gated_forward_cuda_kernel(
    y: &Tensor<f32>,
    z: &Tensor<f32>,
    weight: &Tensor<f32>,
    gated: &mut Tensor<f32>,
    inv_rms: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let width = params[2] as usize;
    let eps = params[3];

    let b = CUBE_POS_Z as usize;
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || t >= time {
        terminate!();
    }

    let mut partials = SharedMemory::<f32>::new(workgroup_size);
    let mut local_sum = 0.0;
    let mut idx = lane;
    while idx < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + idx * y.stride(2);
        let y_val = y[y_idx];
        local_sum += y_val * y_val;
        idx += workgroup_size;
    }
    partials[lane] = local_sum;
    sync_cube();

    reduce_partials_cuda(&mut partials, lane, workgroup_size);

    let inv_rms_row = 1.0 / f32::sqrt(partials[0] / width as f32 + eps);
    if lane == 0usize {
        let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1);
        inv_rms[inv_idx] = inv_rms_row;
    }
    sync_cube();

    let mut out_lane = lane;
    while out_lane < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + out_lane * y.stride(2);
        let z_idx = b * z.stride(0) + t * z.stride(1) + out_lane * z.stride(2);
        let w_idx = out_lane * weight.stride(0);
        let sigmoid = 1.0 / (1.0 + f32::exp(-z[z_idx]));
        let out_idx = b * gated.stride(0) + t * gated.stride(1) + out_lane * gated.stride(2);
        gated[out_idx] = (y[y_idx] * inv_rms_row) * weight[w_idx] * (z[z_idx] * sigmoid);
        out_lane += workgroup_size;
    }
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba2_rmsnorm_gated_backward_cuda_kernel(
    y: &Tensor<f32>,
    z: &Tensor<f32>,
    weight: &Tensor<f32>,
    grad_output: &Tensor<f32>,
    inv_rms: &Tensor<f32>,
    grad_y: &mut Tensor<f32>,
    grad_z: &mut Tensor<f32>,
    grad_weight: &mut Tensor<Atomic<f32>>,
    params: &Tensor<f32>,
    #[comptime] workgroup_size: usize,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let width = params[2] as usize;

    let b = CUBE_POS_Z as usize;
    let t = CUBE_POS_Y as usize;
    let lane = UNIT_POS_X as usize;
    if b >= batch || t >= time {
        terminate!();
    }

    let inv_idx = b * inv_rms.stride(0) + t * inv_rms.stride(1);
    let inv_rms_row = inv_rms[inv_idx];
    let inv_rms_cubed_over_width = inv_rms_row * inv_rms_row * inv_rms_row / width as f32;

    let mut partials = SharedMemory::<f32>::new(workgroup_size);
    let mut local_dot = 0.0;
    let mut idx = lane;
    while idx < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + idx * y.stride(2);
        let z_idx = b * z.stride(0) + t * z.stride(1) + idx * z.stride(2);
        let go_idx =
            b * grad_output.stride(0) + t * grad_output.stride(1) + idx * grad_output.stride(2);
        let sigmoid = 1.0 / (1.0 + f32::exp(-z[z_idx]));
        let gate = z[z_idx] * sigmoid;
        let grad_normalized = grad_output[go_idx] * gate * weight[idx * weight.stride(0)];
        local_dot += grad_normalized * y[y_idx];
        idx += workgroup_size;
    }
    partials[lane] = local_dot;
    sync_cube();

    reduce_partials_cuda(&mut partials, lane, workgroup_size);

    let correction = partials[0] * inv_rms_cubed_over_width;
    sync_cube();

    let mut out_lane = lane;
    while out_lane < width {
        let y_idx = b * y.stride(0) + t * y.stride(1) + out_lane * y.stride(2);
        let z_idx = b * z.stride(0) + t * z.stride(1) + out_lane * z.stride(2);
        let go_idx = b * grad_output.stride(0)
            + t * grad_output.stride(1)
            + out_lane * grad_output.stride(2);
        let w_idx = out_lane * weight.stride(0);

        let y_val = y[y_idx];
        let z_val = z[z_idx];
        let grad_out = grad_output[go_idx];
        let weight_val = weight[w_idx];
        let sigmoid = 1.0 / (1.0 + f32::exp(-z_val));
        let gate = z_val * sigmoid;
        let normalized = y_val * inv_rms_row;
        let weighted = normalized * weight_val;
        let grad_normalized = grad_out * gate * weight_val;
        grad_y[y_idx] = grad_normalized * inv_rms_row - y_val * correction;

        let silu_grad = sigmoid * (1.0 + z_val * (1.0 - sigmoid));
        grad_z[z_idx] = grad_out * weighted * silu_grad;
        grad_weight[w_idx].fetch_add(grad_out * gate * normalized);
        out_lane += workgroup_size;
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
