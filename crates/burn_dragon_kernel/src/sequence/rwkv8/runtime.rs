use std::any::{Any, TypeId};

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Shape, TensorData, TensorPrimitive};
use burn_cubecl::cubecl::{self, prelude::*};
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::CubeRuntime;

const RWKV8_META_LEN: usize = 6;
const RWKV8_EPS: f32 = 1.0e-6;
const RWKV8_WGPU_WORKGROUP_X: u32 = 64;
#[cfg(feature = "cuda")]
const RWKV8_CUDA_WORKGROUP_X: u32 = 128;

pub(crate) struct Rwkv8RuntimeForwardOutput<B: BackendTrait> {
    pub(crate) context: BurnTensor<B, 4>,
    pub(crate) rho: BurnTensor<B, 4>,
    pub(crate) rho_norm: BurnTensor<B, 3>,
    pub(crate) rho_before: Option<BurnTensor<B, 5>>,
    pub(crate) rho_norm_before: Option<BurnTensor<B, 4>>,
}

pub(crate) struct Rwkv8RuntimeStateOutput<B: BackendTrait> {
    pub(crate) rho: BurnTensor<B, 4>,
    pub(crate) rho_norm: BurnTensor<B, 3>,
}

pub(crate) struct Rwkv8RuntimeBackwardOutput<B: BackendTrait> {
    pub(crate) grad_query: BurnTensor<B, 4>,
    pub(crate) grad_value: BurnTensor<B, 4>,
    pub(crate) grad_decay: BurnTensor<B, 3>,
    pub(crate) prev_boundary_grad_rho: BurnTensor<B, 4>,
    pub(crate) prev_boundary_grad_rho_norm: BurnTensor<B, 3>,
}

struct Rwkv8RuntimeForwardCubeOutput<R: CubeRuntime> {
    context: CubeTensor<R>,
    rho: CubeTensor<R>,
    rho_norm: CubeTensor<R>,
    rho_before: CubeTensor<R>,
    rho_norm_before: CubeTensor<R>,
}

struct Rwkv8RuntimeStateCubeOutput<R: CubeRuntime> {
    rho: CubeTensor<R>,
    rho_norm: CubeTensor<R>,
}

struct Rwkv8RuntimeBackwardCubeOutput<R: CubeRuntime> {
    grad_query: CubeTensor<R>,
    grad_value: CubeTensor<R>,
    grad_decay: CubeTensor<R>,
    prev_boundary_grad_rho: CubeTensor<R>,
    prev_boundary_grad_rho_norm: CubeTensor<R>,
}

pub(crate) fn try_rwkv8_runtime_forward<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: BurnTensor<B, 4>,
    rho_norm_state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
    capture_history: bool,
) -> Option<Rwkv8RuntimeForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_rwkv8_runtime_forward_with_runtime::<B, WgpuRuntime>(
        query.clone(),
        value.clone(),
        rho_state.clone(),
        rho_norm_state.clone(),
        decay.clone(),
        capture_history,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_rwkv8_runtime_forward_with_runtime::<B, CudaRuntime>(
                query,
                value,
                rho_state,
                rho_norm_state,
                decay,
                capture_history,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub(crate) fn try_rwkv8_runtime_advance_state<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: BurnTensor<B, 4>,
    rho_norm_state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
) -> Option<Rwkv8RuntimeStateOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_rwkv8_runtime_advance_state_with_runtime::<B, WgpuRuntime>(
        query.clone(),
        value.clone(),
        rho_state.clone(),
        rho_norm_state.clone(),
        decay.clone(),
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_rwkv8_runtime_advance_state_with_runtime::<B, CudaRuntime>(
                query,
                value,
                rho_state,
                rho_norm_state,
                decay,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

pub(crate) fn try_rwkv8_runtime_backward_chunk<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_before: BurnTensor<B, 5>,
    rho_norm_before: BurnTensor<B, 4>,
    decay: BurnTensor<B, 3>,
    grad_output: BurnTensor<B, 4>,
    boundary_grad_rho_after: BurnTensor<B, 4>,
    boundary_grad_rho_norm_after: BurnTensor<B, 3>,
) -> Option<Rwkv8RuntimeBackwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    try_rwkv8_runtime_backward_chunk_with_runtime::<B, WgpuRuntime>(
        query.clone(),
        value.clone(),
        rho_before.clone(),
        rho_norm_before.clone(),
        decay.clone(),
        grad_output.clone(),
        boundary_grad_rho_after.clone(),
        boundary_grad_rho_norm_after.clone(),
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_rwkv8_runtime_backward_chunk_with_runtime::<B, CudaRuntime>(
                query,
                value,
                rho_before,
                rho_norm_before,
                decay,
                grad_output,
                boundary_grad_rho_after,
                boundary_grad_rho_norm_after,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })
}

fn try_rwkv8_runtime_forward_with_runtime<B: BackendTrait, R: CubeRuntime>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: BurnTensor<B, 4>,
    rho_norm_state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
    capture_history: bool,
) -> Option<Rwkv8RuntimeForwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let params = rwkv8_meta_tensor::<B>(&query, value_heads, embd);

    let query = cast_cube_tensor::<B, R, 4>(query)?;
    let value = cast_cube_tensor::<B, R, 4>(value)?;
    let rho_state = cast_cube_tensor::<B, R, 4>(rho_state)?;
    let rho_norm_state = cast_cube_tensor::<B, R, 3>(rho_norm_state)?;
    let decay = cast_cube_tensor::<B, R, 3>(decay)?;
    let params = cast_cube_tensor::<B, R, 1>(params)?;

    let output = rwkv8_runtime_forward::<R>(
        query,
        value,
        rho_state,
        rho_norm_state,
        decay,
        params,
    );

    Some(Rwkv8RuntimeForwardOutput {
        context: cast_burn_tensor::<B, R, 4>(output.context)?,
        rho: cast_burn_tensor::<B, R, 4>(output.rho)?,
        rho_norm: cast_burn_tensor::<B, R, 3>(output.rho_norm)?,
        rho_before: if capture_history {
            Some(cast_burn_tensor::<B, R, 5>(output.rho_before)?)
        } else {
            None
        },
        rho_norm_before: if capture_history {
            Some(cast_burn_tensor::<B, R, 4>(output.rho_norm_before)?)
        } else {
            None
        },
    })
}

fn try_rwkv8_runtime_advance_state_with_runtime<B: BackendTrait, R: CubeRuntime>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_state: BurnTensor<B, 4>,
    rho_norm_state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
) -> Option<Rwkv8RuntimeStateOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let params = rwkv8_meta_tensor::<B>(&query, value_heads, embd);

    let query = cast_cube_tensor::<B, R, 4>(query)?;
    let value = cast_cube_tensor::<B, R, 4>(value)?;
    let rho_state = cast_cube_tensor::<B, R, 4>(rho_state)?;
    let rho_norm_state = cast_cube_tensor::<B, R, 3>(rho_norm_state)?;
    let decay = cast_cube_tensor::<B, R, 3>(decay)?;
    let params = cast_cube_tensor::<B, R, 1>(params)?;

    let output =
        rwkv8_runtime_advance_state::<R>(query, value, rho_state, rho_norm_state, decay, params);

    Some(Rwkv8RuntimeStateOutput {
        rho: cast_burn_tensor::<B, R, 4>(output.rho)?,
        rho_norm: cast_burn_tensor::<B, R, 3>(output.rho_norm)?,
    })
}

fn try_rwkv8_runtime_backward_chunk_with_runtime<B: BackendTrait, R: CubeRuntime>(
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    rho_before: BurnTensor<B, 5>,
    rho_norm_before: BurnTensor<B, 4>,
    decay: BurnTensor<B, 3>,
    grad_output: BurnTensor<B, 4>,
    boundary_grad_rho_after: BurnTensor<B, 4>,
    boundary_grad_rho_norm_after: BurnTensor<B, 3>,
) -> Option<Rwkv8RuntimeBackwardOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let params = rwkv8_meta_tensor::<B>(&query, value_heads, embd);

    let query_cube = cast_cube_tensor::<B, R, 4>(query.clone())?;
    let value_cube = cast_cube_tensor::<B, R, 4>(value.clone())?;
    let rho_before_cube = cast_cube_tensor::<B, R, 5>(rho_before.clone())?;
    let rho_norm_before_cube = cast_cube_tensor::<B, R, 4>(rho_norm_before.clone())?;
    let _decay_cube = cast_cube_tensor::<B, R, 3>(decay.clone())?;
    let grad_output_cube = cast_cube_tensor::<B, R, 4>(grad_output)?;
    let params_cube = cast_cube_tensor::<B, R, 1>(params)?;

    let grad_query_weights = cast_burn_tensor::<B, R, 4>(rwkv8_runtime_grad_query_weights::<R>(
        rho_before_cube.clone(),
        rho_norm_before_cube.clone(),
        grad_output_cube.clone(),
        params_cube.clone(),
    ))?;
    let grad_rho_from_context =
        cast_burn_tensor::<B, R, 5>(rwkv8_runtime_grad_rho_from_context::<R>(
            query_cube.clone(),
            rho_norm_before_cube.clone(),
            grad_output_cube.clone(),
            params_cube.clone(),
        ))?;
    let grad_rho_norm_from_context = cast_burn_tensor::<B, R, 4>(
        rwkv8_runtime_grad_rho_norm_from_context::<R>(
            query_cube.clone(),
            rho_norm_before_cube,
            cast_cube_tensor::<B, R, 4>(grad_query_weights.clone())?,
            params_cube.clone(),
        ),
    )?;

    let reversed_grad_rho_from_context = reverse_time_tensor5(grad_rho_from_context);
    let reversed_grad_rho_norm_from_context = reverse_time_tensor4(grad_rho_norm_from_context);

    let grad_rho_carry = try_rwkv8_runtime_state_recurrence_with_runtime::<B, R>(
        reversed_grad_rho_from_context,
        boundary_grad_rho_after,
        decay.clone(),
    )?;
    let grad_rho_norm_carry = try_rwkv8_runtime_norm_recurrence_with_runtime::<B, R>(
        reversed_grad_rho_norm_from_context,
        boundary_grad_rho_norm_after,
        decay.clone(),
    )?;

    let grad_rho_carry_forward = reverse_time_tensor5(grad_rho_carry.history);
    let grad_rho_norm_carry_forward = reverse_time_tensor4(grad_rho_norm_carry.history);

    let grad_query = cast_burn_tensor::<B, R, 4>(rwkv8_runtime_grad_query::<R>(
        query_cube.clone(),
        value_cube.clone(),
        cast_cube_tensor::<B, R, 4>(grad_query_weights)?,
        cast_cube_tensor::<B, R, 5>(grad_rho_carry_forward.clone())?,
        cast_cube_tensor::<B, R, 4>(grad_rho_norm_carry_forward.clone())?,
        params_cube.clone(),
    ))?;
    let grad_value_heads = cast_burn_tensor::<B, R, 4>(rwkv8_runtime_grad_value::<R>(
        query_cube,
        cast_cube_tensor::<B, R, 5>(grad_rho_carry_forward.clone())?,
        params_cube.clone(),
    ))?;
    let grad_decay_partial = cast_burn_tensor::<B, R, 4>(rwkv8_runtime_grad_decay_partial::<R>(
        cast_cube_tensor::<B, R, 5>(rho_before)?,
        cast_cube_tensor::<B, R, 4>(rho_norm_before)?,
        cast_cube_tensor::<B, R, 5>(grad_rho_carry_forward)?,
        cast_cube_tensor::<B, R, 4>(grad_rho_norm_carry_forward)?,
        params_cube,
    ))?;

    let grad_value = if value_heads == 1 {
        grad_value_heads
            .sum_dim(1)
            .reshape([batch, 1, time, embd])
    } else {
        grad_value_heads
    };
    let grad_decay = grad_decay_partial.sum_dim(2).sum_dim(0).reshape([1, heads, latent]);

    Some(Rwkv8RuntimeBackwardOutput {
        grad_query,
        grad_value,
        grad_decay,
        prev_boundary_grad_rho: grad_rho_carry.final_state,
        prev_boundary_grad_rho_norm: grad_rho_norm_carry.final_state,
    })
}

struct Rwkv8StateRecurrenceOutput<B: BackendTrait> {
    history: BurnTensor<B, 5>,
    final_state: BurnTensor<B, 4>,
}

struct Rwkv8NormRecurrenceOutput<B: BackendTrait> {
    history: BurnTensor<B, 4>,
    final_state: BurnTensor<B, 3>,
}

fn try_rwkv8_runtime_state_recurrence_with_runtime<B: BackendTrait, R: CubeRuntime>(
    delta: BurnTensor<B, 5>,
    state: BurnTensor<B, 4>,
    decay: BurnTensor<B, 3>,
) -> Option<Rwkv8StateRecurrenceOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let value_heads = 1;
    let embd = delta.shape().dims::<5>()[4];
    let query_stub = BurnTensor::<B, 4>::zeros(
        [
            delta.shape().dims::<5>()[0],
            delta.shape().dims::<5>()[1],
            delta.shape().dims::<5>()[2],
            delta.shape().dims::<5>()[3],
        ],
        &delta.device(),
    );
    let params = rwkv8_meta_tensor::<B>(&query_stub, value_heads, embd);

    let delta_cube = cast_cube_tensor::<B, R, 5>(delta)?;
    let state_cube = cast_cube_tensor::<B, R, 4>(state)?;
    let decay_cube = cast_cube_tensor::<B, R, 3>(decay)?;
    let params_cube = cast_cube_tensor::<B, R, 1>(params)?;

    let (history, final_state) =
        rwkv8_runtime_state_recurrence::<R>(delta_cube, state_cube, decay_cube, params_cube);
    Some(Rwkv8StateRecurrenceOutput {
        history: cast_burn_tensor::<B, R, 5>(history)?,
        final_state: cast_burn_tensor::<B, R, 4>(final_state)?,
    })
}

fn try_rwkv8_runtime_norm_recurrence_with_runtime<B: BackendTrait, R: CubeRuntime>(
    delta: BurnTensor<B, 4>,
    state: BurnTensor<B, 3>,
    decay: BurnTensor<B, 3>,
) -> Option<Rwkv8NormRecurrenceOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let value_heads = 1;
    let query_stub = BurnTensor::<B, 4>::zeros(delta.shape().dims::<4>(), &delta.device());
    let params = rwkv8_meta_tensor::<B>(&query_stub, value_heads, 1);

    let delta_cube = cast_cube_tensor::<B, R, 4>(delta)?;
    let state_cube = cast_cube_tensor::<B, R, 3>(state)?;
    let decay_cube = cast_cube_tensor::<B, R, 3>(decay)?;
    let params_cube = cast_cube_tensor::<B, R, 1>(params)?;

    let (history, final_state) =
        rwkv8_runtime_norm_recurrence::<R>(delta_cube, state_cube, decay_cube, params_cube);
    Some(Rwkv8NormRecurrenceOutput {
        history: cast_burn_tensor::<B, R, 4>(history)?,
        final_state: cast_burn_tensor::<B, R, 3>(final_state)?,
    })
}

fn rwkv8_runtime_forward<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho_state: CubeTensor<R>,
    rho_norm_state: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> Rwkv8RuntimeForwardCubeOutput<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho_state = into_contiguous(rho_state);
    let rho_norm_state = into_contiguous(rho_norm_state);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();

    let rho_before = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent, embd]),
    );
    let rho = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, latent, embd]),
    );
    let rho_norm_before = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, time, latent]),
    );
    let rho_norm = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, latent]),
    );
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );

    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);

    let _ = rwkv8_state_qv_history_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(embd as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        query.as_tensor_arg(1),
        value.as_tensor_arg(1),
        rho_state.as_tensor_arg(1),
        rho_before.as_tensor_arg(1),
        rho.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    let _ = rwkv8_norm_query_history_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        query.as_tensor_arg(1),
        rho_norm_state.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        rho_norm.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    let _ = rwkv8_context_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(embd as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        query.as_tensor_arg(1),
        rho_before.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        context.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );

    Rwkv8RuntimeForwardCubeOutput {
        context,
        rho,
        rho_norm,
        rho_before,
        rho_norm_before,
    }
}

fn rwkv8_runtime_advance_state<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho_state: CubeTensor<R>,
    rho_norm_state: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> Rwkv8RuntimeStateCubeOutput<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho_state = into_contiguous(rho_state);
    let rho_norm_state = into_contiguous(rho_norm_state);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);

    let [batch, heads, _time, latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let rho = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([batch, heads, latent, embd]),
    );
    let rho_norm = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, latent]),
    );

    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);

    let _ = rwkv8_state_qv_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(embd as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        query.as_tensor_arg(1),
        value.as_tensor_arg(1),
        rho_state.as_tensor_arg(1),
        rho.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    let _ = rwkv8_norm_query_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        query.as_tensor_arg(1),
        rho_norm_state.as_tensor_arg(1),
        rho_norm.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );

    Rwkv8RuntimeStateCubeOutput {
        rho,
        rho_norm,
    }
}

fn rwkv8_runtime_state_recurrence<R: CubeRuntime>(
    delta: CubeTensor<R>,
    state: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let delta = into_contiguous(delta);
    let state = into_contiguous(state);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);

    let [batch, heads, time, latent, embd] = delta.meta.shape.dims::<5>();
    let client = delta.client.clone();
    let device = delta.device.clone();
    let history = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent, embd]),
    );
    let final_state = empty_device::<R, f32>(
        client.clone(),
        delta.device.clone(),
        Shape::new([batch, heads, latent, embd]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_state_recurrence_history_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(embd as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        delta.as_tensor_arg(1),
        state.as_tensor_arg(1),
        history.as_tensor_arg(1),
        final_state.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    (history, final_state)
}

fn rwkv8_runtime_norm_recurrence<R: CubeRuntime>(
    delta: CubeTensor<R>,
    state: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> (CubeTensor<R>, CubeTensor<R>) {
    let delta = into_contiguous(delta);
    let state = into_contiguous(state);
    let decay = into_contiguous(decay);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = delta.meta.shape.dims::<4>();
    let client = delta.client.clone();
    let device = delta.device.clone();
    let history = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let final_state = empty_device::<R, f32>(
        client.clone(),
        delta.device.clone(),
        Shape::new([batch, heads, latent]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_norm_recurrence_history_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, batch as u32),
        cube_dim,
        delta.as_tensor_arg(1),
        state.as_tensor_arg(1),
        history.as_tensor_arg(1),
        final_state.as_tensor_arg(1),
        decay.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    (history, final_state)
}

fn rwkv8_runtime_grad_query_weights<R: CubeRuntime>(
    rho_before: CubeTensor<R>,
    rho_norm_before: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let rho_before = into_contiguous(rho_before);
    let rho_norm_before = into_contiguous(rho_norm_before);
    let grad_output = into_contiguous(grad_output);
    let params = into_contiguous(params);

    let [batch, heads, time, latent, _embd] = rho_before.meta.shape.dims::<5>();
    let client = rho_before.client.clone();
    let device = rho_before.device.clone();
    let grad_query_weights = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_query_weights_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, (batch * time) as u32),
        cube_dim,
        rho_before.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        grad_output.as_tensor_arg(1),
        grad_query_weights.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_query_weights
}

fn rwkv8_runtime_grad_rho_from_context<R: CubeRuntime>(
    query: CubeTensor<R>,
    rho_norm_before: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let rho_norm_before = into_contiguous(rho_norm_before);
    let grad_output = into_contiguous(grad_output);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let embd = grad_output.meta.shape.dims::<4>()[3];
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_rho = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent, embd]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_rho_from_context_kernel::launch::<R>(
        &client,
        CubeCount::Static(
            div_ceil_u32(embd as u32, workgroup_x),
            heads as u32,
            (batch * time * latent) as u32,
        ),
        cube_dim,
        query.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        grad_output.as_tensor_arg(1),
        grad_rho.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_rho
}

fn rwkv8_runtime_grad_rho_norm_from_context<R: CubeRuntime>(
    query: CubeTensor<R>,
    rho_norm_before: CubeTensor<R>,
    grad_query_weights: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let rho_norm_before = into_contiguous(rho_norm_before);
    let grad_query_weights = into_contiguous(grad_query_weights);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_rho_norm = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_rho_norm_from_context_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, (batch * time) as u32),
        cube_dim,
        query.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        grad_query_weights.as_tensor_arg(1),
        grad_rho_norm.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_rho_norm
}

fn rwkv8_runtime_grad_query<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    grad_query_weights: CubeTensor<R>,
    grad_rho_carry: CubeTensor<R>,
    grad_rho_norm_carry: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let grad_query_weights = into_contiguous(grad_query_weights);
    let grad_rho_carry = into_contiguous(grad_rho_carry);
    let grad_rho_norm_carry = into_contiguous(grad_rho_norm_carry);
    let params = into_contiguous(params);

    let [batch, heads, time, latent] = query.meta.shape.dims::<4>();
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_query = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_query_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, (batch * time) as u32),
        cube_dim,
        query.as_tensor_arg(1),
        value.as_tensor_arg(1),
        grad_query_weights.as_tensor_arg(1),
        grad_rho_carry.as_tensor_arg(1),
        grad_rho_norm_carry.as_tensor_arg(1),
        grad_query.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_query
}

fn rwkv8_runtime_grad_value<R: CubeRuntime>(
    query: CubeTensor<R>,
    grad_rho_carry: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let grad_rho_carry = into_contiguous(grad_rho_carry);
    let params = into_contiguous(params);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let embd = grad_rho_carry.meta.shape.dims::<5>()[4];
    let client = query.client.clone();
    let device = query.device.clone();
    let grad_value = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_value_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(embd as u32, workgroup_x), heads as u32, (batch * time) as u32),
        cube_dim,
        query.as_tensor_arg(1),
        grad_rho_carry.as_tensor_arg(1),
        grad_value.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_value
}

fn rwkv8_runtime_grad_decay_partial<R: CubeRuntime>(
    rho_before: CubeTensor<R>,
    rho_norm_before: CubeTensor<R>,
    grad_rho_carry: CubeTensor<R>,
    grad_rho_norm_carry: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let rho_before = into_contiguous(rho_before);
    let rho_norm_before = into_contiguous(rho_norm_before);
    let grad_rho_carry = into_contiguous(grad_rho_carry);
    let grad_rho_norm_carry = into_contiguous(grad_rho_norm_carry);
    let params = into_contiguous(params);

    let [batch, heads, time, latent, _embd] = rho_before.meta.shape.dims::<5>();
    let client = rho_before.client.clone();
    let device = rho_before.device.clone();
    let grad_decay_partial = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let workgroup_x = rwkv8_workgroup_x::<R>();
    let cube_dim = CubeDim::new_1d(workgroup_x);
    let _ = rwkv8_grad_decay_partial_kernel::launch::<R>(
        &client,
        CubeCount::Static(div_ceil_u32(latent as u32, workgroup_x), heads as u32, (batch * time) as u32),
        cube_dim,
        rho_before.as_tensor_arg(1),
        rho_norm_before.as_tensor_arg(1),
        grad_rho_carry.as_tensor_arg(1),
        grad_rho_norm_carry.as_tensor_arg(1),
        grad_decay_partial.as_tensor_arg(1),
        params.as_tensor_arg(1),
    );
    grad_decay_partial
}

#[cube(launch)]
fn rwkv8_state_qv_kernel(
    query: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    rho_state_in: &Tensor<Line<f32>>,
    rho_state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut l = 0usize;
    while l < latent {
        let decay_index = h * decay.stride(1) + l * decay.stride(2);
        let rho_index = b * rho_state_in.stride(0)
            + h * rho_state_in.stride(1)
            + l * rho_state_in.stride(2)
            + e * rho_state_in.stride(3);
        let mut rho_prev = rho_state_in[rho_index];
        let mut t = 0usize;
        while t < time {
            let value_index = b * value.stride(0)
                + value_head * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            let query_index =
                b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
            rho_prev = rho_prev * decay[decay_index] + query[query_index] * value[value_index];
            t += 1usize;
        }
        let out_index = b * rho_state_out.stride(0)
            + h * rho_state_out.stride(1)
            + l * rho_state_out.stride(2)
            + e * rho_state_out.stride(3);
        rho_state_out[out_index] = rho_prev;
        l += 1usize;
    }
}

#[cube(launch)]
fn rwkv8_state_qv_history_kernel(
    query: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    rho_state_in: &Tensor<Line<f32>>,
    rho_before: &mut Tensor<Line<f32>>,
    rho_state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut l = 0usize;
    while l < latent {
        let decay_index = h * decay.stride(1) + l * decay.stride(2);
        let state_index = b * rho_state_in.stride(0)
            + h * rho_state_in.stride(1)
            + l * rho_state_in.stride(2)
            + e * rho_state_in.stride(3);
        let mut rho_prev = rho_state_in[state_index];
        let mut t = 0usize;
        while t < time {
            let value_index = b * value.stride(0)
                + value_head * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            let query_index =
                b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
            let history_index = b * rho_before.stride(0)
                + h * rho_before.stride(1)
                + t * rho_before.stride(2)
                + l * rho_before.stride(3)
                + e * rho_before.stride(4);
            rho_before[history_index] = rho_prev;
            rho_prev = rho_prev * decay[decay_index] + query[query_index] * value[value_index];
            t += 1usize;
        }
        let out_index = b * rho_state_out.stride(0)
            + h * rho_state_out.stride(1)
            + l * rho_state_out.stride(2)
            + e * rho_state_out.stride(3);
        rho_state_out[out_index] = rho_prev;
        l += 1usize;
    }
}

#[cube(launch)]
fn rwkv8_norm_query_kernel(
    query: &Tensor<Line<f32>>,
    rho_norm_state_in: &Tensor<Line<f32>>,
    rho_norm_state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || l >= latent {
        terminate!();
    }

    let decay_index = h * decay.stride(1) + l * decay.stride(2);
    let state_index = b * rho_norm_state_in.stride(0)
        + h * rho_norm_state_in.stride(1)
        + l * rho_norm_state_in.stride(2);
    let mut rho_prev = rho_norm_state_in[state_index];
    let mut t = 0usize;
    while t < time {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        rho_prev = rho_prev * decay[decay_index] + query[query_index];
        t += 1usize;
    }
    let out_index = b * rho_norm_state_out.stride(0)
        + h * rho_norm_state_out.stride(1)
        + l * rho_norm_state_out.stride(2);
    rho_norm_state_out[out_index] = rho_prev;
}

#[cube(launch)]
fn rwkv8_norm_query_history_kernel(
    query: &Tensor<Line<f32>>,
    rho_norm_state_in: &Tensor<Line<f32>>,
    rho_norm_before: &mut Tensor<Line<f32>>,
    rho_norm_state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || l >= latent {
        terminate!();
    }

    let decay_index = h * decay.stride(1) + l * decay.stride(2);
    let state_index = b * rho_norm_state_in.stride(0)
        + h * rho_norm_state_in.stride(1)
        + l * rho_norm_state_in.stride(2);
    let mut rho_prev = rho_norm_state_in[state_index];
    let mut t = 0usize;
    while t < time {
        let history_index = b * rho_norm_before.stride(0)
            + h * rho_norm_before.stride(1)
            + t * rho_norm_before.stride(2)
            + l * rho_norm_before.stride(3);
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        rho_norm_before[history_index] = rho_prev;
        rho_prev = rho_prev * decay[decay_index] + query[query_index];
        t += 1usize;
    }
    let out_index = b * rho_norm_state_out.stride(0)
        + h * rho_norm_state_out.stride(1)
        + l * rho_norm_state_out.stride(2);
    rho_norm_state_out[out_index] = rho_prev;
}

#[cube(launch)]
fn rwkv8_context_kernel(
    query: &Tensor<Line<f32>>,
    rho_before: &Tensor<Line<f32>>,
    rho_norm_before: &Tensor<Line<f32>>,
    context: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let eps = Line::cast_from(RWKV8_EPS);
    let mut t = 0usize;
    while t < time {
        let mut q_sum = Line::cast_from(0u32);
        let mut l = 0usize;
        while l < latent {
            let query_index =
                b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
            q_sum += query[query_index];
            l += 1usize;
        }
        q_sum += eps;

        let mut acc = Line::cast_from(0u32);
        let mut l2 = 0usize;
        while l2 < latent {
            let query_index =
                b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l2 * query.stride(3);
            let rho_index = b * rho_before.stride(0)
                + h * rho_before.stride(1)
                + t * rho_before.stride(2)
                + l2 * rho_before.stride(3)
                + e * rho_before.stride(4);
            let rho_norm_index = b * rho_norm_before.stride(0)
                + h * rho_norm_before.stride(1)
                + t * rho_norm_before.stride(2)
                + l2 * rho_norm_before.stride(3);
            acc += (query[query_index] / q_sum) * (rho_before[rho_index] / (rho_norm_before[rho_norm_index] + eps));
            l2 += 1usize;
        }

        let out_index = b * context.stride(0)
            + h * context.stride(1)
            + t * context.stride(2)
            + e * context.stride(3);
        context[out_index] = acc;
        t += 1usize;
    }
}

#[cube(launch)]
fn rwkv8_state_recurrence_history_kernel(
    delta: &Tensor<Line<f32>>,
    state_in: &Tensor<Line<f32>>,
    state_before: &mut Tensor<Line<f32>>,
    state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || e >= embd {
        terminate!();
    }

    let mut l = 0usize;
    while l < latent {
        let decay_index = h * decay.stride(1) + l * decay.stride(2);
        let state_index = b * state_in.stride(0)
            + h * state_in.stride(1)
            + l * state_in.stride(2)
            + e * state_in.stride(3);
        let mut prev = state_in[state_index];
        let mut t = 0usize;
        while t < time {
            let history_index = b * state_before.stride(0)
                + h * state_before.stride(1)
                + t * state_before.stride(2)
                + l * state_before.stride(3)
                + e * state_before.stride(4);
            let delta_index = b * delta.stride(0)
                + h * delta.stride(1)
                + t * delta.stride(2)
                + l * delta.stride(3)
                + e * delta.stride(4);
            state_before[history_index] = prev;
            prev = prev * decay[decay_index] + delta[delta_index];
            t += 1usize;
        }
        let out_index = b * state_out.stride(0)
            + h * state_out.stride(1)
            + l * state_out.stride(2)
            + e * state_out.stride(3);
        state_out[out_index] = prev;
        l += 1usize;
    }
}

#[cube(launch)]
fn rwkv8_norm_recurrence_history_kernel(
    delta: &Tensor<Line<f32>>,
    state_in: &Tensor<Line<f32>>,
    state_before: &mut Tensor<Line<f32>>,
    state_out: &mut Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || l >= latent {
        terminate!();
    }

    let decay_index = h * decay.stride(1) + l * decay.stride(2);
    let state_index = b * state_in.stride(0) + h * state_in.stride(1) + l * state_in.stride(2);
    let mut prev = state_in[state_index];
    let mut t = 0usize;
    while t < time {
        let history_index = b * state_before.stride(0)
            + h * state_before.stride(1)
            + t * state_before.stride(2)
            + l * state_before.stride(3);
        let delta_index =
            b * delta.stride(0) + h * delta.stride(1) + t * delta.stride(2) + l * delta.stride(3);
        state_before[history_index] = prev;
        prev = prev * decay[decay_index] + delta[delta_index];
        t += 1usize;
    }
    let out_index = b * state_out.stride(0) + h * state_out.stride(1) + l * state_out.stride(2);
    state_out[out_index] = prev;
}

#[cube(launch)]
fn rwkv8_grad_query_weights_kernel(
    rho_before: &Tensor<Line<f32>>,
    rho_norm_before: &Tensor<Line<f32>>,
    grad_output: &Tensor<Line<f32>>,
    grad_query_weights: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = (CUBE_POS_Z as usize) / time.max(1);
    let t = (CUBE_POS_Z as usize) % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let eps = Line::cast_from(RWKV8_EPS);
    let rho_norm_index = b * rho_norm_before.stride(0)
        + h * rho_norm_before.stride(1)
        + t * rho_norm_before.stride(2)
        + l * rho_norm_before.stride(3);
    let denom = rho_norm_before[rho_norm_index] + eps;
    let mut acc = Line::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let rho_index = b * rho_before.stride(0)
            + h * rho_before.stride(1)
            + t * rho_before.stride(2)
            + l * rho_before.stride(3)
            + e * rho_before.stride(4);
        let grad_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + t * grad_output.stride(2)
            + e * grad_output.stride(3);
        acc += grad_output[grad_index] * (rho_before[rho_index] / denom);
        e += 1usize;
    }
    let out_index = b * grad_query_weights.stride(0)
        + h * grad_query_weights.stride(1)
        + t * grad_query_weights.stride(2)
        + l * grad_query_weights.stride(3);
    grad_query_weights[out_index] = acc;
}

#[cube(launch)]
fn rwkv8_grad_rho_from_context_kernel(
    query: &Tensor<Line<f32>>,
    rho_norm_before: &Tensor<Line<f32>>,
    grad_output: &Tensor<Line<f32>>,
    grad_rho: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let block = CUBE_POS_Z as usize;
    let b = block / (time.max(1) * latent.max(1));
    let rem = block % (time.max(1) * latent.max(1));
    let t = rem / latent.max(1);
    let l = rem % latent.max(1);
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent || e >= embd {
        terminate!();
    }

    let eps = Line::cast_from(RWKV8_EPS);
    let mut q_sum = Line::cast_from(0u32);
    let mut l2 = 0usize;
    while l2 < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l2 * query.stride(3);
        q_sum += query[query_index];
        l2 += 1usize;
    }
    q_sum += eps;

    let query_index =
        b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
    let rho_norm_index = b * rho_norm_before.stride(0)
        + h * rho_norm_before.stride(1)
        + t * rho_norm_before.stride(2)
        + l * rho_norm_before.stride(3);
    let grad_index = b * grad_output.stride(0)
        + h * grad_output.stride(1)
        + t * grad_output.stride(2)
        + e * grad_output.stride(3);
    let out_index = b * grad_rho.stride(0)
        + h * grad_rho.stride(1)
        + t * grad_rho.stride(2)
        + l * grad_rho.stride(3)
        + e * grad_rho.stride(4);
    grad_rho[out_index] =
        grad_output[grad_index] * (query[query_index] / q_sum) / (rho_norm_before[rho_norm_index] + eps);
}

#[cube(launch)]
fn rwkv8_grad_rho_norm_from_context_kernel(
    query: &Tensor<Line<f32>>,
    rho_norm_before: &Tensor<Line<f32>>,
    grad_query_weights: &Tensor<Line<f32>>,
    grad_rho_norm: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;

    let b = (CUBE_POS_Z as usize) / time.max(1);
    let t = (CUBE_POS_Z as usize) % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let eps = Line::cast_from(RWKV8_EPS);
    let mut q_sum = Line::cast_from(0u32);
    let mut l2 = 0usize;
    while l2 < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l2 * query.stride(3);
        q_sum += query[query_index];
        l2 += 1usize;
    }
    q_sum += eps;

    let query_index =
        b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
    let rho_norm_index = b * rho_norm_before.stride(0)
        + h * rho_norm_before.stride(1)
        + t * rho_norm_before.stride(2)
        + l * rho_norm_before.stride(3);
    let grad_qw_index = b * grad_query_weights.stride(0)
        + h * grad_query_weights.stride(1)
        + t * grad_query_weights.stride(2)
        + l * grad_query_weights.stride(3);
    let out_index = b * grad_rho_norm.stride(0)
        + h * grad_rho_norm.stride(1)
        + t * grad_rho_norm.stride(2)
        + l * grad_rho_norm.stride(3);
    grad_rho_norm[out_index] =
        -(query[query_index] / q_sum) * grad_query_weights[grad_qw_index] / (rho_norm_before[rho_norm_index] + eps);
}

#[cube(launch)]
fn rwkv8_grad_query_kernel(
    query: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    grad_query_weights: &Tensor<Line<f32>>,
    grad_rho_carry: &Tensor<Line<f32>>,
    grad_rho_norm_carry: &Tensor<Line<f32>>,
    grad_query: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let value_heads = u32::cast_from(params[2]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = (CUBE_POS_Z as usize) / time.max(1);
    let t = (CUBE_POS_Z as usize) % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let eps = Line::cast_from(RWKV8_EPS);
    let mut value_head = h;
    if value_heads == 1usize {
        value_head = 0usize;
    }
    let mut q_sum = Line::cast_from(0u32);
    let mut weighted_dot = Line::cast_from(0u32);
    let mut l2 = 0usize;
    while l2 < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l2 * query.stride(3);
        let grad_qw_index = b * grad_query_weights.stride(0)
            + h * grad_query_weights.stride(1)
            + t * grad_query_weights.stride(2)
            + l2 * grad_query_weights.stride(3);
        let q = query[query_index];
        q_sum += q;
        weighted_dot += grad_query_weights[grad_qw_index] * q;
        l2 += 1usize;
    }
    q_sum += eps;
    weighted_dot = weighted_dot / q_sum;

    let mut state_term = Line::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let value_index = b * value.stride(0)
            + value_head * value.stride(1)
            + t * value.stride(2)
            + e * value.stride(3);
        let grad_rho_index = b * grad_rho_carry.stride(0)
            + h * grad_rho_carry.stride(1)
            + t * grad_rho_carry.stride(2)
            + l * grad_rho_carry.stride(3)
            + e * grad_rho_carry.stride(4);
        state_term += grad_rho_carry[grad_rho_index] * value[value_index];
        e += 1usize;
    }

    let grad_qw_index =
        b * grad_query_weights.stride(0) + h * grad_query_weights.stride(1) + t * grad_query_weights.stride(2) + l * grad_query_weights.stride(3);
    let grad_rho_norm_index = b * grad_rho_norm_carry.stride(0)
        + h * grad_rho_norm_carry.stride(1)
        + t * grad_rho_norm_carry.stride(2)
        + l * grad_rho_norm_carry.stride(3);
    let out_index =
        b * grad_query.stride(0) + h * grad_query.stride(1) + t * grad_query.stride(2) + l * grad_query.stride(3);
    grad_query[out_index] = state_term + grad_rho_norm_carry[grad_rho_norm_index]
        + (grad_query_weights[grad_qw_index] - weighted_dot) / q_sum;
}

#[cube(launch)]
fn rwkv8_grad_value_kernel(
    query: &Tensor<Line<f32>>,
    grad_rho_carry: &Tensor<Line<f32>>,
    grad_value: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = (CUBE_POS_Z as usize) / time.max(1);
    let t = (CUBE_POS_Z as usize) % time.max(1);
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || e >= embd {
        terminate!();
    }

    let mut grad = Line::cast_from(0u32);
    let mut l = 0usize;
    while l < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        let grad_rho_index = b * grad_rho_carry.stride(0)
            + h * grad_rho_carry.stride(1)
            + t * grad_rho_carry.stride(2)
            + l * grad_rho_carry.stride(3)
            + e * grad_rho_carry.stride(4);
        grad += grad_rho_carry[grad_rho_index] * query[query_index];
        l += 1usize;
    }

    let out_index =
        b * grad_value.stride(0) + h * grad_value.stride(1) + t * grad_value.stride(2) + e * grad_value.stride(3);
    grad_value[out_index] = grad;
}

#[cube(launch)]
fn rwkv8_grad_decay_partial_kernel(
    rho_before: &Tensor<Line<f32>>,
    rho_norm_before: &Tensor<Line<f32>>,
    grad_rho_carry: &Tensor<Line<f32>>,
    grad_rho_norm_carry: &Tensor<Line<f32>>,
    grad_decay_partial: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[3]) as usize;
    let latent = u32::cast_from(params[4]) as usize;
    let embd = u32::cast_from(params[5]) as usize;

    let b = (CUBE_POS_Z as usize) / time.max(1);
    let t = (CUBE_POS_Z as usize) % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let mut grad = Line::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let rho_index = b * rho_before.stride(0)
            + h * rho_before.stride(1)
            + t * rho_before.stride(2)
            + l * rho_before.stride(3)
            + e * rho_before.stride(4);
        let grad_rho_index = b * grad_rho_carry.stride(0)
            + h * grad_rho_carry.stride(1)
            + t * grad_rho_carry.stride(2)
            + l * grad_rho_carry.stride(3)
            + e * grad_rho_carry.stride(4);
        grad += grad_rho_carry[grad_rho_index] * rho_before[rho_index];
        e += 1usize;
    }
    let rho_norm_index = b * rho_norm_before.stride(0)
        + h * rho_norm_before.stride(1)
        + t * rho_norm_before.stride(2)
        + l * rho_norm_before.stride(3);
    let grad_rho_norm_index = b * grad_rho_norm_carry.stride(0)
        + h * grad_rho_norm_carry.stride(1)
        + t * grad_rho_norm_carry.stride(2)
        + l * grad_rho_norm_carry.stride(3);
    grad += grad_rho_norm_carry[grad_rho_norm_index] * rho_norm_before[rho_norm_index];

    let out_index = b * grad_decay_partial.stride(0)
        + h * grad_decay_partial.stride(1)
        + t * grad_decay_partial.stride(2)
        + l * grad_decay_partial.stride(3);
    grad_decay_partial[out_index] = grad;
}

fn rwkv8_meta_tensor<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value_heads: usize,
    embd: usize,
) -> BurnTensor<B, 1> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                value_heads as f32,
                time as f32,
                latent as f32,
                embd as f32,
            ],
            [RWKV8_META_LEN],
        ),
        &query.device(),
    )
}

fn reverse_time_indices<B: BackendTrait>(time: usize, device: &B::Device) -> BurnTensor<B, 1, Int> {
    BurnTensor::<B, 1, Int>::arange(0..time as i64, device).flip([0])
}

fn reverse_time_tensor5<B: BackendTrait>(tensor: BurnTensor<B, 5>) -> BurnTensor<B, 5> {
    let time = tensor.shape().dims::<5>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn reverse_time_tensor4<B: BackendTrait>(tensor: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
    let time = tensor.shape().dims::<4>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn rwkv8_workgroup_x<R: CubeRuntime>() -> u32 {
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return RWKV8_CUDA_WORKGROUP_X;
    }
    RWKV8_WGPU_WORKGROUP_X
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    value.div_ceil(divisor)
}

fn cast_cube_tensor<B: BackendTrait, R: CubeRuntime, const D: usize>(
    tensor: BurnTensor<B, D>,
) -> Option<CubeTensor<R>>
where
    B::FloatTensorPrimitive: 'static,
{
    let primitive = tensor.into_primitive().tensor();
    let cube_tensor: CubeTensor<R> = try_cast_primitive::<B, _>(primitive)?;
    if cube_tensor.dtype != burn::tensor::DType::F32 {
        return None;
    }
    Some(cube_tensor)
}

fn cast_burn_tensor<B: BackendTrait, R: CubeRuntime, const D: usize>(
    tensor: CubeTensor<R>,
) -> Option<BurnTensor<B, D>>
where
    B::FloatTensorPrimitive: 'static,
{
    let primitive = try_cast_backend::<B, _>(tensor)?;
    Some(BurnTensor::<B, D>::from_primitive(TensorPrimitive::Float(
        primitive,
    )))
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(value: B::FloatTensorPrimitive) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}
