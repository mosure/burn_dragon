#[cfg(feature = "cuda")]
use burn::tensor::Tensor as BurnTensor;
#[cfg(feature = "cuda")]
use burn::tensor::{Shape, TensorData};
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::{self, prelude::*};
#[cfg(feature = "cuda")]
use burn_cubecl::kernel::into_contiguous;
#[cfg(feature = "cuda")]
use burn_cubecl::ops::numeric::empty_device;
#[cfg(feature = "cuda")]
use burn_cubecl::tensor::CubeTensor;
#[cfg(feature = "cuda")]
use burn_wgpu::CubeBackend;

#[cfg(feature = "cuda")]
type CudaCubeBackend = CubeBackend<CudaRuntime, f32, i32, u8>;

#[cfg(feature = "cuda")]
const PARAMS_LEN: usize = 5;
#[cfg(feature = "cuda")]
const MAMBA_CONV_CUDA_WORKGROUP_X: u32 = 128;

#[cfg(feature = "cuda")]
pub(crate) struct MambaDepthwiseConvCudaForwardOutput {
    pub(crate) preact: CubeTensor<CudaRuntime>,
    pub(crate) activated: CubeTensor<CudaRuntime>,
    pub(crate) next_state: CubeTensor<CudaRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) struct MambaDepthwiseConvCudaBackwardOutput {
    pub(crate) grad_x: CubeTensor<CudaRuntime>,
    pub(crate) grad_weight: CubeTensor<CudaRuntime>,
    pub(crate) grad_bias: CubeTensor<CudaRuntime>,
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba_depthwise_conv_forward_cuda(
    x: CubeTensor<CudaRuntime>,
    conv_weight: CubeTensor<CudaRuntime>,
    conv_bias: CubeTensor<CudaRuntime>,
    state: CubeTensor<CudaRuntime>,
) -> MambaDepthwiseConvCudaForwardOutput {
    let x = into_contiguous(x);
    let conv_weight = into_contiguous(conv_weight);
    let conv_bias = into_contiguous(conv_bias);
    let state = into_contiguous(state);

    let [batch, views, channels, time] = x.meta.shape.dims::<4>();
    let d_conv = conv_weight.meta.shape.dims::<2>()[1];
    let client = x.client.clone();
    let device = x.device.clone();
    let max_pos = time.max(d_conv);

    let preact = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, views, channels, time]),
    );
    let activated = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, views, channels, time]),
    );
    let next_state = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, views, channels, d_conv]),
    );
    let params = params_tensor(
        &device,
        [
            batch as f32,
            views as f32,
            channels as f32,
            time as f32,
            d_conv as f32,
        ],
    )
    .into_primitive()
    .tensor();

    let cube_dim = CubeDim::new_1d(MAMBA_CONV_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(max_pos as u32, MAMBA_CONV_CUDA_WORKGROUP_X),
        channels as u32,
        (batch * views) as u32,
    );

    unsafe {
        let _ = mamba_depthwise_conv_forward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            x.as_tensor_arg(1),
            conv_weight.as_tensor_arg(1),
            conv_bias.as_tensor_arg(1),
            state.as_tensor_arg(1),
            preact.as_tensor_arg(1),
            activated.as_tensor_arg(1),
            next_state.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    MambaDepthwiseConvCudaForwardOutput {
        preact,
        activated,
        next_state,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba_depthwise_conv_backward_cuda(
    x: CubeTensor<CudaRuntime>,
    conv_weight: CubeTensor<CudaRuntime>,
    state: CubeTensor<CudaRuntime>,
    grad_preact: CubeTensor<CudaRuntime>,
) -> MambaDepthwiseConvCudaBackwardOutput {
    let x = into_contiguous(x);
    let conv_weight = into_contiguous(conv_weight);
    let state = into_contiguous(state);
    let grad_preact = into_contiguous(grad_preact);

    let [batch, views, channels, time] = x.meta.shape.dims::<4>();
    let d_conv = conv_weight.meta.shape.dims::<2>()[1];
    let client = x.client.clone();
    let device = x.device.clone();
    let max_pos = time.max(d_conv);

    let grad_x = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, views, channels, time]),
    );
    let grad_weight = BurnTensor::<CudaCubeBackend, 2>::zeros([channels, d_conv], &device)
        .into_primitive()
        .tensor();
    let grad_bias = BurnTensor::<CudaCubeBackend, 1>::zeros([channels], &device)
        .into_primitive()
        .tensor();
    let params = params_tensor(
        &device,
        [
            batch as f32,
            views as f32,
            channels as f32,
            time as f32,
            d_conv as f32,
        ],
    )
    .into_primitive()
    .tensor();

    let cube_dim = CubeDim::new_1d(MAMBA_CONV_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(max_pos as u32, MAMBA_CONV_CUDA_WORKGROUP_X),
        channels as u32,
        (batch * views) as u32,
    );

    unsafe {
        let _ = mamba_depthwise_conv_backward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            x.as_tensor_arg(1),
            conv_weight.as_tensor_arg(1),
            state.as_tensor_arg(1),
            grad_preact.as_tensor_arg(1),
            grad_x.as_tensor_arg(1),
            grad_weight.as_tensor_arg(1),
            grad_bias.as_tensor_arg(1),
            params.as_tensor_arg(1),
        );
    }

    MambaDepthwiseConvCudaBackwardOutput {
        grad_x,
        grad_weight,
        grad_bias,
    }
}

#[cfg(feature = "cuda")]
fn params_tensor(
    device: &<CudaCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; PARAMS_LEN],
) -> BurnTensor<CudaCubeBackend, 1> {
    BurnTensor::<CudaCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [PARAMS_LEN]),
        device,
    )
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba_depthwise_conv_forward_cuda_kernel(
    x: &Tensor<f32>,
    conv_weight: &Tensor<f32>,
    conv_bias: &Tensor<f32>,
    state: &Tensor<f32>,
    preact: &mut Tensor<f32>,
    activated: &mut Tensor<f32>,
    next_state: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = params[0] as usize;
    let views = params[1] as usize;
    let channels = params[2] as usize;
    let time = params[3] as usize;
    let d_conv = params[4] as usize;

    let bv = CUBE_POS_Z as usize;
    let channel = CUBE_POS_Y as usize;
    let pos = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if bv >= batch * views || channel >= channels {
        terminate!();
    }

    let batch_idx = bv / views;
    let view_idx = bv % views;

    if pos < time {
        let mut acc = conv_bias[channel * conv_bias.stride(0)];
        let mut tap = 0usize;
        while tap < d_conv {
            let hist_idx = pos + 1usize + tap;
            let value = if hist_idx < d_conv {
                let idx = batch_idx * state.stride(0)
                    + view_idx * state.stride(1)
                    + channel * state.stride(2)
                    + hist_idx * state.stride(3);
                state[idx]
            } else {
                let x_idx = hist_idx - d_conv;
                let idx = batch_idx * x.stride(0)
                    + view_idx * x.stride(1)
                    + channel * x.stride(2)
                    + x_idx * x.stride(3);
                x[idx]
            };
            let weight_idx = channel * conv_weight.stride(0) + tap * conv_weight.stride(1);
            acc += value * conv_weight[weight_idx];
            tap += 1usize;
        }
        let out_idx = batch_idx * preact.stride(0)
            + view_idx * preact.stride(1)
            + channel * preact.stride(2)
            + pos * preact.stride(3);
        preact[out_idx] = acc;
        let sigmoid = 1.0 / (1.0 + f32::exp(-acc));
        activated[batch_idx * activated.stride(0)
            + view_idx * activated.stride(1)
            + channel * activated.stride(2)
            + pos * activated.stride(3)] = acc * sigmoid;
    }

    if pos < d_conv {
        let hist_idx = time + pos;
        let value = if hist_idx < d_conv {
            let idx = batch_idx * state.stride(0)
                + view_idx * state.stride(1)
                + channel * state.stride(2)
                + hist_idx * state.stride(3);
            state[idx]
        } else {
            let x_idx = hist_idx - d_conv;
            let idx = batch_idx * x.stride(0)
                + view_idx * x.stride(1)
                + channel * x.stride(2)
                + x_idx * x.stride(3);
            x[idx]
        };
        let out_idx = batch_idx * next_state.stride(0)
            + view_idx * next_state.stride(1)
            + channel * next_state.stride(2)
            + pos * next_state.stride(3);
        next_state[out_idx] = value;
    }
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba_depthwise_conv_backward_cuda_kernel(
    x: &Tensor<f32>,
    conv_weight: &Tensor<f32>,
    state: &Tensor<f32>,
    grad_preact: &Tensor<f32>,
    grad_x: &mut Tensor<f32>,
    grad_weight: &mut Tensor<Atomic<f32>>,
    grad_bias: &mut Tensor<Atomic<f32>>,
    params: &Tensor<f32>,
) {
    let batch = params[0] as usize;
    let views = params[1] as usize;
    let channels = params[2] as usize;
    let time = params[3] as usize;
    let d_conv = params[4] as usize;

    let bv = CUBE_POS_Z as usize;
    let channel = CUBE_POS_Y as usize;
    let pos = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if bv >= batch * views || channel >= channels {
        terminate!();
    }

    let batch_idx = bv / views;
    let view_idx = bv % views;

    if pos < time {
        let grad_idx = batch_idx * grad_preact.stride(0)
            + view_idx * grad_preact.stride(1)
            + channel * grad_preact.stride(2)
            + pos * grad_preact.stride(3);
        let grad_val = grad_preact[grad_idx];
        grad_bias[channel * grad_bias.stride(0)].fetch_add(grad_val);

        let mut tap = 0usize;
        while tap < d_conv {
            let hist_idx = pos + 1usize + tap;
            let value = if hist_idx < d_conv {
                let idx = batch_idx * state.stride(0)
                    + view_idx * state.stride(1)
                    + channel * state.stride(2)
                    + hist_idx * state.stride(3);
                state[idx]
            } else {
                let x_idx = hist_idx - d_conv;
                let idx = batch_idx * x.stride(0)
                    + view_idx * x.stride(1)
                    + channel * x.stride(2)
                    + x_idx * x.stride(3);
                x[idx]
            };
            let weight_idx = channel * grad_weight.stride(0) + tap * grad_weight.stride(1);
            grad_weight[weight_idx].fetch_add(grad_val * value);
            tap += 1usize;
        }
    }

    if pos < time {
        let mut acc = 0.0;
        let mut tap = 0usize;
        while tap < d_conv {
            let out_t = pos + d_conv - 1usize;
            if out_t >= tap {
                let target_t = out_t - tap;
                if target_t < time {
                    let grad_idx = batch_idx * grad_preact.stride(0)
                        + view_idx * grad_preact.stride(1)
                        + channel * grad_preact.stride(2)
                        + target_t * grad_preact.stride(3);
                    let weight_idx = channel * conv_weight.stride(0) + tap * conv_weight.stride(1);
                    acc += grad_preact[grad_idx] * conv_weight[weight_idx];
                }
            }
            tap += 1usize;
        }
        let out_idx = batch_idx * grad_x.stride(0)
            + view_idx * grad_x.stride(1)
            + channel * grad_x.stride(2)
            + pos * grad_x.stride(3);
        grad_x[out_idx] = acc;
    }
}

#[cfg(feature = "cuda")]
fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}
