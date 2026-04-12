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
const MAMBA2_SSD_WGPU_WORKGROUP_X: u32 = 64;
const MAMBA2_SSD_WGPU_MAX_DSTATE: usize = 64;
#[cfg(feature = "cuda")]
const MAMBA2_SSD_CUDA_WORKGROUP_X: u32 = 128;
const PARAMS_LEN: usize = 8;

pub(crate) struct Mamba2SsdWgpuForwardOutput {
    pub(crate) y_grouped: CubeTensor<WgpuRuntime>,
    pub(crate) final_ssm: CubeTensor<WgpuRuntime>,
    pub(crate) state_history: Option<CubeTensor<WgpuRuntime>>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba2SsdCudaForwardOutput {
    pub(crate) y_grouped: CubeTensor<CudaRuntime>,
    pub(crate) final_ssm: CubeTensor<CudaRuntime>,
    pub(crate) state_history: Option<CubeTensor<CudaRuntime>>,
}

#[cfg(feature = "cuda")]
pub(crate) struct Mamba2SsdCudaBackwardOutput {
    pub(crate) grad_x_grouped: CubeTensor<CudaRuntime>,
    pub(crate) grad_b_group: CubeTensor<CudaRuntime>,
    pub(crate) grad_c_group: CubeTensor<CudaRuntime>,
    pub(crate) grad_dt_grouped: CubeTensor<CudaRuntime>,
    pub(crate) grad_a_log: CubeTensor<CudaRuntime>,
    pub(crate) grad_d_skip: CubeTensor<CudaRuntime>,
}

pub(crate) fn fused_mamba2_ssd_forward_wgpu(
    x_grouped: CubeTensor<WgpuRuntime>,
    b_group: CubeTensor<WgpuRuntime>,
    c_group: CubeTensor<WgpuRuntime>,
    dt_grouped: CubeTensor<WgpuRuntime>,
    a_log: CubeTensor<WgpuRuntime>,
    d_skip: CubeTensor<WgpuRuntime>,
    initial_ssm: Option<CubeTensor<WgpuRuntime>>,
    capture_state_history: bool,
) -> Mamba2SsdWgpuForwardOutput {
    let x_grouped = into_contiguous(x_grouped);
    let b_group = into_contiguous(b_group);
    let c_group = into_contiguous(c_group);
    let dt_grouped = into_contiguous(dt_grouped);
    let a_log = into_contiguous(a_log);
    let d_skip = into_contiguous(d_skip);

    let [batch, time, ngroups, heads_per_group, headdim] = x_grouped.meta.shape.dims::<5>();
    let d_state = b_group.meta.shape.dims::<4>()[3];
    assert!(
        d_state <= MAMBA2_SSD_WGPU_MAX_DSTATE,
        "wgpu fused mamba2 ssd forward requires d_state <= {} (got {d_state})",
        MAMBA2_SSD_WGPU_MAX_DSTATE
    );
    let client = x_grouped.client.clone();
    let device = x_grouped.device.clone();
    let has_initial = initial_ssm.is_some();
    let initial_ssm = initial_ssm.unwrap_or_else(|| {
        BurnTensor::<WgpuCubeBackend, 5>::zeros(
            [batch, ngroups, heads_per_group, headdim, d_state],
            &device,
        )
        .into_primitive()
        .tensor()
    });
    let initial_ssm = into_contiguous(initial_ssm);

    let y_grouped = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, time, ngroups, heads_per_group, headdim]),
    );
    let final_ssm = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, ngroups, heads_per_group, headdim, d_state]),
    );
    let state_history = if capture_state_history {
        Some(empty_device::<WgpuRuntime, f32>(
            client.clone(),
            device.clone(),
            Shape::new([batch, time, ngroups, heads_per_group, headdim, d_state]),
        ))
    } else {
        None
    };
    let state_history_arg = state_history.clone().unwrap_or_else(|| {
        empty_device::<WgpuRuntime, f32>(client.clone(), device.clone(), Shape::new([1]))
    });
    let params = params_tensor_wgpu(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            headdim as f32,
            d_state as f32,
            if has_initial { 1.0 } else { 0.0 },
            if capture_state_history { 1.0 } else { 0.0 },
        ],
    );
    let params = params.into_primitive().tensor();

    let cube_dim = CubeDim::new_1d(MAMBA2_SSD_WGPU_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        1,
        (ngroups * heads_per_group * headdim) as u32,
        batch as u32,
    );

    unsafe {
        let _ = mamba2_ssd_forward_wgpu_kernel::launch_unchecked::<WgpuRuntime>(
            &client,
            cube_count,
            cube_dim,
            x_grouped.clone().into_tensor_arg(),
            b_group.clone().into_tensor_arg(),
            c_group.clone().into_tensor_arg(),
            dt_grouped.clone().into_tensor_arg(),
            a_log.clone().into_tensor_arg(),
            d_skip.clone().into_tensor_arg(),
            initial_ssm.clone().into_tensor_arg(),
            y_grouped.clone().into_tensor_arg(),
            final_ssm.clone().into_tensor_arg(),
            state_history_arg.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
            MAMBA2_SSD_WGPU_MAX_DSTATE,
        );
    }

    Mamba2SsdWgpuForwardOutput {
        y_grouped,
        final_ssm,
        state_history,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba2_ssd_forward_cuda(
    x_grouped: CubeTensor<CudaRuntime>,
    b_group: CubeTensor<CudaRuntime>,
    c_group: CubeTensor<CudaRuntime>,
    dt_grouped: CubeTensor<CudaRuntime>,
    a_log: CubeTensor<CudaRuntime>,
    d_skip: CubeTensor<CudaRuntime>,
    initial_ssm: Option<CubeTensor<CudaRuntime>>,
    capture_state_history: bool,
) -> Mamba2SsdCudaForwardOutput {
    let x_grouped = into_contiguous(x_grouped);
    let b_group = into_contiguous(b_group);
    let c_group = into_contiguous(c_group);
    let dt_grouped = into_contiguous(dt_grouped);
    let a_log = into_contiguous(a_log);
    let d_skip = into_contiguous(d_skip);

    let [batch, time, ngroups, heads_per_group, headdim] = x_grouped.meta.shape.dims::<5>();
    let d_state = b_group.meta.shape.dims::<4>()[3];
    let client = x_grouped.client.clone();
    let device = x_grouped.device.clone();
    let has_initial = initial_ssm.is_some();
    let initial_ssm = initial_ssm.unwrap_or_else(|| {
        BurnTensor::<CudaCubeBackend, 5>::zeros(
            [batch, ngroups, heads_per_group, headdim, d_state],
            &device,
        )
        .into_primitive()
        .tensor()
    });
    let initial_ssm = into_contiguous(initial_ssm);

    let y_grouped = BurnTensor::<CudaCubeBackend, 5>::zeros(
        [batch, time, ngroups, heads_per_group, headdim],
        &device,
    )
    .into_primitive()
    .tensor();
    let final_ssm = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, ngroups, heads_per_group, headdim, d_state]),
    );
    let state_history = if capture_state_history {
        Some(empty_device::<CudaRuntime, f32>(
            client.clone(),
            device.clone(),
            Shape::new([batch, time, ngroups, heads_per_group, headdim, d_state]),
        ))
    } else {
        None
    };
    let state_history_arg = state_history.clone().unwrap_or_else(|| {
        empty_device::<CudaRuntime, f32>(client.clone(), device.clone(), Shape::new([1]))
    });
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            headdim as f32,
            d_state as f32,
            if has_initial { 1.0 } else { 0.0 },
            if capture_state_history { 1.0 } else { 0.0 },
        ],
    );
    let params = params.into_primitive().tensor();

    let cube_dim = CubeDim::new_1d(MAMBA2_SSD_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32((headdim * d_state) as u32, MAMBA2_SSD_CUDA_WORKGROUP_X),
        (ngroups * heads_per_group) as u32,
        batch as u32,
    );

    unsafe {
        let _ = mamba2_ssd_forward_cuda_kernel::launch_unchecked::<CudaRuntime>(
            &client,
            cube_count,
            cube_dim,
            x_grouped.clone().into_tensor_arg(),
            b_group.clone().into_tensor_arg(),
            c_group.clone().into_tensor_arg(),
            dt_grouped.clone().into_tensor_arg(),
            a_log.clone().into_tensor_arg(),
            d_skip.clone().into_tensor_arg(),
            initial_ssm.clone().into_tensor_arg(),
            y_grouped.clone().into_tensor_arg(),
            final_ssm.clone().into_tensor_arg(),
            state_history_arg.clone().into_tensor_arg(),
            params.clone().into_tensor_arg(),
        );
    }

    Mamba2SsdCudaForwardOutput {
        y_grouped,
        final_ssm,
        state_history,
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn fused_mamba2_ssd_backward_cuda(
    x_grouped: CubeTensor<CudaRuntime>,
    b_group: CubeTensor<CudaRuntime>,
    c_group: CubeTensor<CudaRuntime>,
    dt_grouped: CubeTensor<CudaRuntime>,
    a_log: CubeTensor<CudaRuntime>,
    d_skip: CubeTensor<CudaRuntime>,
    initial_ssm: Option<CubeTensor<CudaRuntime>>,
    grad_y_grouped: CubeTensor<CudaRuntime>,
    state_history: Option<CubeTensor<CudaRuntime>>,
) -> Mamba2SsdCudaBackwardOutput {
    let x_grouped = into_contiguous(x_grouped);
    let b_group = into_contiguous(b_group);
    let c_group = into_contiguous(c_group);
    let dt_grouped = into_contiguous(dt_grouped);
    let a_log = into_contiguous(a_log);
    let d_skip = into_contiguous(d_skip);
    let grad_y_grouped = into_contiguous(grad_y_grouped);

    let [batch, time, ngroups, heads_per_group, headdim] = x_grouped.meta.shape.dims::<5>();
    let d_state = b_group.meta.shape.dims::<4>()[3];
    let nheads = ngroups * heads_per_group;
    let client = x_grouped.client.clone();
    let device = x_grouped.device.clone();
    let has_initial = initial_ssm.is_some();
    let initial_ssm = initial_ssm.unwrap_or_else(|| {
        BurnTensor::<CudaCubeBackend, 5>::zeros(
            [batch, ngroups, heads_per_group, headdim, d_state],
            &device,
        )
        .into_primitive()
        .tensor()
    });
    let initial_ssm = into_contiguous(initial_ssm);

    let provided_state_history = state_history.map(into_contiguous);
    let recompute_state_history = provided_state_history.is_none();
    let state_history = provided_state_history.clone().unwrap_or_else(|| {
        empty_device::<CudaRuntime, f32>(
            client.clone(),
            device.clone(),
            Shape::new([batch, time, ngroups, heads_per_group, headdim, d_state]),
        )
    });
    let grad_x_grouped = BurnTensor::<CudaCubeBackend, 5>::zeros(
        [batch, time, ngroups, heads_per_group, headdim],
        &device,
    )
    .into_primitive()
    .tensor();
    let grad_b_group =
        BurnTensor::<CudaCubeBackend, 4>::zeros([batch, time, ngroups, d_state], &device)
            .into_primitive()
            .tensor();
    let grad_c_group =
        BurnTensor::<CudaCubeBackend, 4>::zeros([batch, time, ngroups, d_state], &device)
            .into_primitive()
            .tensor();
    let grad_dt_grouped =
        BurnTensor::<CudaCubeBackend, 4>::zeros([batch, time, ngroups, heads_per_group], &device)
            .into_primitive()
            .tensor();
    let grad_a_log = BurnTensor::<CudaCubeBackend, 1>::zeros([nheads], &device)
        .into_primitive()
        .tensor();
    let grad_d_skip = BurnTensor::<CudaCubeBackend, 1>::zeros([nheads], &device)
        .into_primitive()
        .tensor();
    let grad_initial_ssm = empty_device::<CudaRuntime, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, ngroups, heads_per_group, headdim, d_state]),
    );
    let params = params_tensor(
        &device,
        [
            batch as f32,
            time as f32,
            ngroups as f32,
            heads_per_group as f32,
            headdim as f32,
            d_state as f32,
            if has_initial { 1.0 } else { 0.0 },
            0.0,
        ],
    );
    let params = params.into_primitive().tensor();

    let cube_dim = CubeDim::new_1d(MAMBA2_SSD_CUDA_WORKGROUP_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32((headdim * d_state) as u32, MAMBA2_SSD_CUDA_WORKGROUP_X),
        (ngroups * heads_per_group) as u32,
        batch as u32,
    );

    unsafe {
        if recompute_state_history {
            let _ = mamba2_ssd_backward_cuda_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                x_grouped.clone().into_tensor_arg(),
                b_group.clone().into_tensor_arg(),
                c_group.clone().into_tensor_arg(),
                dt_grouped.clone().into_tensor_arg(),
                a_log.clone().into_tensor_arg(),
                d_skip.clone().into_tensor_arg(),
                initial_ssm.clone().into_tensor_arg(),
                grad_y_grouped.clone().into_tensor_arg(),
                state_history.clone().into_tensor_arg(),
                grad_x_grouped.clone().into_tensor_arg(),
                grad_b_group.clone().into_tensor_arg(),
                grad_c_group.clone().into_tensor_arg(),
                grad_dt_grouped.clone().into_tensor_arg(),
                grad_a_log.clone().into_tensor_arg(),
                grad_d_skip.clone().into_tensor_arg(),
                grad_initial_ssm.clone().into_tensor_arg(),
                params.clone().into_tensor_arg(),
            );
        } else {
            let _ = mamba2_ssd_backward_from_history_cuda_kernel::launch_unchecked::<CudaRuntime>(
                &client,
                cube_count,
                cube_dim,
                x_grouped.clone().into_tensor_arg(),
                b_group.clone().into_tensor_arg(),
                c_group.clone().into_tensor_arg(),
                dt_grouped.clone().into_tensor_arg(),
                a_log.clone().into_tensor_arg(),
                d_skip.clone().into_tensor_arg(),
                initial_ssm.clone().into_tensor_arg(),
                grad_y_grouped.clone().into_tensor_arg(),
                state_history.clone().into_tensor_arg(),
                grad_x_grouped.clone().into_tensor_arg(),
                grad_b_group.clone().into_tensor_arg(),
                grad_c_group.clone().into_tensor_arg(),
                grad_dt_grouped.clone().into_tensor_arg(),
                grad_a_log.clone().into_tensor_arg(),
                grad_d_skip.clone().into_tensor_arg(),
                params.clone().into_tensor_arg(),
            );
        }
    }

    Mamba2SsdCudaBackwardOutput {
        grad_x_grouped,
        grad_b_group,
        grad_c_group,
        grad_dt_grouped,
        grad_a_log,
        grad_d_skip,
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

fn params_tensor_wgpu(
    device: &<WgpuCubeBackend as burn::tensor::backend::Backend>::Device,
    values: [f32; PARAMS_LEN],
) -> BurnTensor<WgpuCubeBackend, 1> {
    BurnTensor::<WgpuCubeBackend, 1>::from_data(
        TensorData::new(values.to_vec(), [PARAMS_LEN]),
        device,
    )
}

#[cfg(feature = "cuda")]
fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

#[cube(launch_unchecked)]
fn mamba2_ssd_forward_wgpu_kernel(
    x_grouped: &Tensor<f32>,
    b_group: &Tensor<f32>,
    c_group: &Tensor<f32>,
    dt_grouped: &Tensor<f32>,
    a_log: &Tensor<f32>,
    d_skip: &Tensor<f32>,
    initial_ssm: &Tensor<f32>,
    y_grouped: &mut Tensor<f32>,
    final_ssm: &mut Tensor<f32>,
    state_history: &mut Tensor<f32>,
    params: &Tensor<f32>,
    #[comptime] max_d_state: usize,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let time = u32::cast_from(params[1]) as usize;
    let ngroups = u32::cast_from(params[2]) as usize;
    let heads_per_group = u32::cast_from(params[3]) as usize;
    let headdim = u32::cast_from(params[4]) as usize;
    let d_state = u32::cast_from(params[5]) as usize;
    let has_initial = params[6] > f32::cast_from(0u32);
    let capture_state_history = params[7] > f32::cast_from(0u32);

    let b = CUBE_POS_Z as usize;
    let ghm = CUBE_POS_Y as usize;
    let s = UNIT_POS_X as usize;
    if b >= batch || ghm >= ngroups * heads_per_group * headdim || s >= max_d_state {
        terminate!();
    }

    let gh = ghm / headdim;
    let m = ghm % headdim;
    let g = gh / heads_per_group;
    let h = gh % heads_per_group;
    let head_flat = g * heads_per_group + h;
    let active_s = s < d_state;

    let mut contrib_tile = SharedMemory::<f32>::new_aligned(max_d_state, 1usize);
    let a = f32::cast_from(0u32) - a_log[head_flat * a_log.stride(0)].exp();
    let d = d_skip[head_flat * d_skip.stride(0)];
    let init_idx = b * initial_ssm.stride(0)
        + g * initial_ssm.stride(1)
        + h * initial_ssm.stride(2)
        + m * initial_ssm.stride(3)
        + s * initial_ssm.stride(4);
    let mut state = if active_s && has_initial {
        initial_ssm[init_idx]
    } else {
        f32::cast_from(0u32)
    };

    let mut t = 0usize;
    while t < time {
        let x_idx = b * x_grouped.stride(0)
            + t * x_grouped.stride(1)
            + g * x_grouped.stride(2)
            + h * x_grouped.stride(3)
            + m * x_grouped.stride(4);
        let dt_idx = b * dt_grouped.stride(0)
            + t * dt_grouped.stride(1)
            + g * dt_grouped.stride(2)
            + h * dt_grouped.stride(3);
        let x_t = x_grouped[x_idx];
        let dt_t = dt_grouped[dt_idx];

        let mut contrib = f32::cast_from(0u32);
        if active_s {
            let b_idx = b * b_group.stride(0)
                + t * b_group.stride(1)
                + g * b_group.stride(2)
                + s * b_group.stride(3);
            let c_idx = b * c_group.stride(0)
                + t * c_group.stride(1)
                + g * c_group.stride(2)
                + s * c_group.stride(3);
            let b_t = b_group[b_idx];
            let c_t = c_group[c_idx];
            let d_a = (dt_t * a).exp();
            state = state * d_a + dt_t * b_t * x_t;
            contrib = state * c_t;
            if capture_state_history {
                let hist_idx = b * state_history.stride(0)
                    + t * state_history.stride(1)
                    + g * state_history.stride(2)
                    + h * state_history.stride(3)
                    + m * state_history.stride(4)
                    + s * state_history.stride(5);
                state_history[hist_idx] = state;
            }
        }
        contrib_tile[s] = contrib;
        sync_cube();

        if s == 0usize {
            let mut y_t = d * x_t;
            let mut acc_s = 0usize;
            while acc_s < d_state {
                y_t += contrib_tile[acc_s];
                acc_s += 1usize;
            }
            let y_idx = b * y_grouped.stride(0)
                + t * y_grouped.stride(1)
                + g * y_grouped.stride(2)
                + h * y_grouped.stride(3)
                + m * y_grouped.stride(4);
            y_grouped[y_idx] = y_t;
        }
        sync_cube();
        t += 1usize;
    }

    if active_s {
        let final_idx = b * final_ssm.stride(0)
            + g * final_ssm.stride(1)
            + h * final_ssm.stride(2)
            + m * final_ssm.stride(3)
            + s * final_ssm.stride(4);
        final_ssm[final_idx] = state;
    }
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba2_ssd_forward_cuda_kernel(
    x_grouped: &Tensor<f32>,
    b_group: &Tensor<f32>,
    c_group: &Tensor<f32>,
    dt_grouped: &Tensor<f32>,
    a_log: &Tensor<f32>,
    d_skip: &Tensor<f32>,
    initial_ssm: &Tensor<f32>,
    y_grouped: &mut Tensor<Atomic<f32>>,
    final_ssm: &mut Tensor<f32>,
    state_history: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let ngroups = params[2] as usize;
    let heads_per_group = params[3] as usize;
    let headdim = params[4] as usize;
    let d_state = params[5] as usize;
    let has_initial = params[6] > 0.5;
    let capture_state_history = params[7] > 0.5;

    let b = CUBE_POS_Z as usize;
    let gh = CUBE_POS_Y as usize;
    let ms = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || gh >= ngroups * heads_per_group || ms >= headdim * d_state {
        terminate!();
    }

    let g = gh / heads_per_group;
    let h = gh % heads_per_group;
    let m = ms / d_state;
    let s = ms % d_state;
    let head_flat = g * heads_per_group + h;

    let a = -f32::exp(a_log[head_flat * a_log.stride(0)]);
    let d = d_skip[head_flat * d_skip.stride(0)];
    let init_idx = b * initial_ssm.stride(0)
        + g * initial_ssm.stride(1)
        + h * initial_ssm.stride(2)
        + m * initial_ssm.stride(3)
        + s * initial_ssm.stride(4);
    let mut state = if has_initial {
        initial_ssm[init_idx]
    } else {
        0.0.into()
    };

    let mut t = 0usize;
    while t < time {
        let x_idx = b * x_grouped.stride(0)
            + t * x_grouped.stride(1)
            + g * x_grouped.stride(2)
            + h * x_grouped.stride(3)
            + m * x_grouped.stride(4);
        let dt_idx = b * dt_grouped.stride(0)
            + t * dt_grouped.stride(1)
            + g * dt_grouped.stride(2)
            + h * dt_grouped.stride(3);
        let bc_idx = b * b_group.stride(0)
            + t * b_group.stride(1)
            + g * b_group.stride(2)
            + s * b_group.stride(3);
        let x_t = x_grouped[x_idx];
        let dt_t = dt_grouped[dt_idx];
        let b_t = b_group[bc_idx];
        let c_idx = b * c_group.stride(0)
            + t * c_group.stride(1)
            + g * c_group.stride(2)
            + s * c_group.stride(3);
        let c_t = c_group[c_idx];
        let d_a = f32::exp(dt_t * a);
        state = state * d_a + dt_t * b_t * x_t;

        let y_idx = b * y_grouped.stride(0)
            + t * y_grouped.stride(1)
            + g * y_grouped.stride(2)
            + h * y_grouped.stride(3)
            + m * y_grouped.stride(4);
        y_grouped[y_idx].fetch_add(state * c_t);
        if s == 0usize {
            y_grouped[y_idx].fetch_add(d * x_t);
        }
        if capture_state_history {
            let hist_idx = b * state_history.stride(0)
                + t * state_history.stride(1)
                + g * state_history.stride(2)
                + h * state_history.stride(3)
                + m * state_history.stride(4)
                + s * state_history.stride(5);
            state_history[hist_idx] = state;
        }
        t += 1usize;
    }

    let final_idx = b * final_ssm.stride(0)
        + g * final_ssm.stride(1)
        + h * final_ssm.stride(2)
        + m * final_ssm.stride(3)
        + s * final_ssm.stride(4);
    final_ssm[final_idx] = state;
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba2_ssd_backward_cuda_kernel(
    x_grouped: &Tensor<f32>,
    b_group: &Tensor<f32>,
    c_group: &Tensor<f32>,
    dt_grouped: &Tensor<f32>,
    a_log: &Tensor<f32>,
    d_skip: &Tensor<f32>,
    initial_ssm: &Tensor<f32>,
    grad_y_grouped: &Tensor<f32>,
    state_history: &mut Tensor<f32>,
    grad_x_grouped: &mut Tensor<Atomic<f32>>,
    grad_b_group: &mut Tensor<Atomic<f32>>,
    grad_c_group: &mut Tensor<Atomic<f32>>,
    grad_dt_grouped: &mut Tensor<Atomic<f32>>,
    grad_a_log: &mut Tensor<Atomic<f32>>,
    grad_d_skip: &mut Tensor<Atomic<f32>>,
    grad_initial_ssm: &mut Tensor<f32>,
    params: &Tensor<f32>,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let ngroups = params[2] as usize;
    let heads_per_group = params[3] as usize;
    let headdim = params[4] as usize;
    let d_state = params[5] as usize;
    let has_initial = params[6] > 0.5;

    let b = CUBE_POS_Z as usize;
    let gh = CUBE_POS_Y as usize;
    let ms = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || gh >= ngroups * heads_per_group || ms >= headdim * d_state {
        terminate!();
    }

    let g = gh / heads_per_group;
    let h = gh % heads_per_group;
    let m = ms / d_state;
    let s = ms % d_state;
    let head_flat = g * heads_per_group + h;

    let a = -f32::exp(a_log[head_flat * a_log.stride(0)]);
    let d = d_skip[head_flat * d_skip.stride(0)];
    let init_idx = b * initial_ssm.stride(0)
        + g * initial_ssm.stride(1)
        + h * initial_ssm.stride(2)
        + m * initial_ssm.stride(3)
        + s * initial_ssm.stride(4);
    let mut state = if has_initial {
        initial_ssm[init_idx]
    } else {
        0.0.into()
    };

    let mut t = 0usize;
    while t < time {
        let x_idx = b * x_grouped.stride(0)
            + t * x_grouped.stride(1)
            + g * x_grouped.stride(2)
            + h * x_grouped.stride(3)
            + m * x_grouped.stride(4);
        let dt_idx = b * dt_grouped.stride(0)
            + t * dt_grouped.stride(1)
            + g * dt_grouped.stride(2)
            + h * dt_grouped.stride(3);
        let bc_idx = b * b_group.stride(0)
            + t * b_group.stride(1)
            + g * b_group.stride(2)
            + s * b_group.stride(3);
        let x_t = x_grouped[x_idx];
        let dt_t = dt_grouped[dt_idx];
        let b_t = b_group[bc_idx];
        let d_a = f32::exp(dt_t * a);
        state = state * d_a + dt_t * b_t * x_t;
        let hist_idx = b * state_history.stride(0)
            + t * state_history.stride(1)
            + g * state_history.stride(2)
            + h * state_history.stride(3)
            + m * state_history.stride(4)
            + s * state_history.stride(5);
        state_history[hist_idx] = state;
        t += 1usize;
    }

    let mut grad_state_carry = 0.0;
    let mut grad_a_total = 0.0;
    let mut grad_d_skip_total = 0.0;
    let mut rev = time;
    while rev > 0usize {
        let t = rev - 1usize;
        let x_idx = b * x_grouped.stride(0)
            + t * x_grouped.stride(1)
            + g * x_grouped.stride(2)
            + h * x_grouped.stride(3)
            + m * x_grouped.stride(4);
        let dt_idx = b * dt_grouped.stride(0)
            + t * dt_grouped.stride(1)
            + g * dt_grouped.stride(2)
            + h * dt_grouped.stride(3);
        let b_idx = b * b_group.stride(0)
            + t * b_group.stride(1)
            + g * b_group.stride(2)
            + s * b_group.stride(3);
        let c_idx = b * c_group.stride(0)
            + t * c_group.stride(1)
            + g * c_group.stride(2)
            + s * c_group.stride(3);
        let y_idx = b * grad_y_grouped.stride(0)
            + t * grad_y_grouped.stride(1)
            + g * grad_y_grouped.stride(2)
            + h * grad_y_grouped.stride(3)
            + m * grad_y_grouped.stride(4);
        let hist_idx = b * state_history.stride(0)
            + t * state_history.stride(1)
            + g * state_history.stride(2)
            + h * state_history.stride(3)
            + m * state_history.stride(4)
            + s * state_history.stride(5);
        let x_t = x_grouped[x_idx];
        let dt_t = dt_grouped[dt_idx];
        let b_t = b_group[b_idx];
        let c_t = c_group[c_idx];
        let grad_y_t = grad_y_grouped[y_idx];
        let state_t = state_history[hist_idx];
        let prev_state = if t > 0usize {
            let prev_hist_idx = b * state_history.stride(0)
                + (t - 1usize) * state_history.stride(1)
                + g * state_history.stride(2)
                + h * state_history.stride(3)
                + m * state_history.stride(4)
                + s * state_history.stride(5);
            state_history[prev_hist_idx]
        } else if has_initial {
            initial_ssm[init_idx]
        } else {
            0.0.into()
        };
        let d_a = f32::exp(dt_t * a);
        let grad_state = grad_y_t * c_t + grad_state_carry;
        grad_c_group[c_idx].fetch_add(grad_y_t * state_t);
        let grad_d_a = grad_state * prev_state;
        grad_b_group[b_idx].fetch_add(grad_state * dt_t * x_t);
        grad_x_grouped[x_idx].fetch_add(grad_state * dt_t * b_t);
        grad_dt_grouped[dt_idx].fetch_add(grad_state * b_t * x_t + grad_d_a * d_a * a);
        grad_a_total += grad_d_a * d_a * dt_t * a;
        grad_state_carry = grad_state * d_a;

        if s == 0usize {
            grad_x_grouped[x_idx].fetch_add(grad_y_t * d);
            grad_d_skip_total += grad_y_t * x_t;
        }
        rev -= 1usize;
    }

    grad_a_log[head_flat * grad_a_log.stride(0)].fetch_add(grad_a_total);
    if s == 0usize {
        grad_d_skip[head_flat * grad_d_skip.stride(0)].fetch_add(grad_d_skip_total);
    }
    grad_initial_ssm[init_idx] = grad_state_carry;
}

#[cfg(feature = "cuda")]
#[cube(launch_unchecked)]
fn mamba2_ssd_backward_from_history_cuda_kernel(
    x_grouped: &Tensor<f32>,
    b_group: &Tensor<f32>,
    c_group: &Tensor<f32>,
    dt_grouped: &Tensor<f32>,
    a_log: &Tensor<f32>,
    d_skip: &Tensor<f32>,
    initial_ssm: &Tensor<f32>,
    grad_y_grouped: &Tensor<f32>,
    state_history: &Tensor<f32>,
    grad_x_grouped: &mut Tensor<Atomic<f32>>,
    grad_b_group: &mut Tensor<Atomic<f32>>,
    grad_c_group: &mut Tensor<Atomic<f32>>,
    grad_dt_grouped: &mut Tensor<Atomic<f32>>,
    grad_a_log: &mut Tensor<Atomic<f32>>,
    grad_d_skip: &mut Tensor<Atomic<f32>>,
    params: &Tensor<f32>,
) {
    let batch = params[0] as usize;
    let time = params[1] as usize;
    let ngroups = params[2] as usize;
    let heads_per_group = params[3] as usize;
    let headdim = params[4] as usize;
    let d_state = params[5] as usize;
    let has_initial = params[6] > 0.5;

    let b = CUBE_POS_Z as usize;
    let gh = CUBE_POS_Y as usize;
    let ms = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || gh >= ngroups * heads_per_group || ms >= headdim * d_state {
        terminate!();
    }

    let g = gh / heads_per_group;
    let h = gh % heads_per_group;
    let m = ms / d_state;
    let s = ms % d_state;
    let head_flat = g * heads_per_group + h;

    let a = -f32::exp(a_log[head_flat * a_log.stride(0)]);
    let d = d_skip[head_flat * d_skip.stride(0)];
    let init_idx = b * initial_ssm.stride(0)
        + g * initial_ssm.stride(1)
        + h * initial_ssm.stride(2)
        + m * initial_ssm.stride(3)
        + s * initial_ssm.stride(4);

    let mut grad_state_carry = 0.0;
    let mut grad_a_total = 0.0;
    let mut grad_d_skip_total = 0.0;
    let mut rev = time;
    while rev > 0usize {
        let t = rev - 1usize;
        let x_idx = b * x_grouped.stride(0)
            + t * x_grouped.stride(1)
            + g * x_grouped.stride(2)
            + h * x_grouped.stride(3)
            + m * x_grouped.stride(4);
        let dt_idx = b * dt_grouped.stride(0)
            + t * dt_grouped.stride(1)
            + g * dt_grouped.stride(2)
            + h * dt_grouped.stride(3);
        let b_idx = b * b_group.stride(0)
            + t * b_group.stride(1)
            + g * b_group.stride(2)
            + s * b_group.stride(3);
        let c_idx = b * c_group.stride(0)
            + t * c_group.stride(1)
            + g * c_group.stride(2)
            + s * c_group.stride(3);
        let y_idx = b * grad_y_grouped.stride(0)
            + t * grad_y_grouped.stride(1)
            + g * grad_y_grouped.stride(2)
            + h * grad_y_grouped.stride(3)
            + m * grad_y_grouped.stride(4);
        let hist_idx = b * state_history.stride(0)
            + t * state_history.stride(1)
            + g * state_history.stride(2)
            + h * state_history.stride(3)
            + m * state_history.stride(4)
            + s * state_history.stride(5);
        let x_t = x_grouped[x_idx];
        let dt_t = dt_grouped[dt_idx];
        let b_t = b_group[b_idx];
        let c_t = c_group[c_idx];
        let grad_y_t = grad_y_grouped[y_idx];
        let state_t = state_history[hist_idx];
        let prev_state = if t > 0usize {
            let prev_hist_idx = b * state_history.stride(0)
                + (t - 1usize) * state_history.stride(1)
                + g * state_history.stride(2)
                + h * state_history.stride(3)
                + m * state_history.stride(4)
                + s * state_history.stride(5);
            state_history[prev_hist_idx]
        } else if has_initial {
            initial_ssm[init_idx]
        } else {
            0.0.into()
        };
        let d_a = f32::exp(dt_t * a);
        let grad_state = grad_y_t * c_t + grad_state_carry;
        grad_c_group[c_idx].fetch_add(grad_y_t * state_t);
        let grad_d_a = grad_state * prev_state;
        grad_b_group[b_idx].fetch_add(grad_state * dt_t * x_t);
        grad_x_grouped[x_idx].fetch_add(grad_state * dt_t * b_t);
        grad_dt_grouped[dt_idx].fetch_add(grad_state * b_t * x_t + grad_d_a * d_a * a);
        grad_a_total += grad_d_a * d_a * dt_t * a;
        grad_state_carry = grad_state * d_a;

        if s == 0usize {
            grad_x_grouped[x_idx].fetch_add(grad_y_t * d);
            grad_d_skip_total += grad_y_t * x_t;
        }
        rev -= 1usize;
    }

    grad_a_log[head_flat * grad_a_log.stride(0)].fetch_add(grad_a_total);
    if s == 0usize {
        grad_d_skip[head_flat * grad_d_skip.stride(0)].fetch_add(grad_d_skip_total);
    }
}
