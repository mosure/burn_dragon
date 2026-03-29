use super::*;
use burn::tensor::Int;
use burn::tensor::backend::AutodiffBackend;
use burn_cubecl::cubecl::prelude::Tensor;

const BACKWARD_META_LEN: usize = 5;
const BACKWARD_WORKGROUP_SIZE_X: u32 = 64;

#[cube(launch)]
fn recurrent_attention_grad_query_kernel(
    forward_state: &Tensor<Line<f32>>,
    reverse_state_rev: &Tensor<Line<f32>>,
    grad_output: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    grad_query: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let l = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || l >= latent {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut grad = Line::cast_from(0u32);
    let mut e = 0usize;
    while e < embd {
        let forward_index = b * forward_state.stride(0)
            + h * forward_state.stride(1)
            + t * forward_state.stride(2)
            + l * forward_state.stride(3)
            + e * forward_state.stride(4);
        let reverse_index = b * reverse_state_rev.stride(0)
            + h * reverse_state_rev.stride(1)
            + tau * reverse_state_rev.stride(2)
            + l * reverse_state_rev.stride(3)
            + e * reverse_state_rev.stride(4);
        let grad_output_index = b * grad_output.stride(0)
            + h * grad_output.stride(1)
            + t * grad_output.stride(2)
            + e * grad_output.stride(3);
        let value_index =
            b * value.stride(0) + h * value.stride(1) + t * value.stride(2) + e * value.stride(3);
        grad += forward_state[forward_index] * grad_output[grad_output_index]
            + reverse_state_rev[reverse_index] * value[value_index];
        e += 1usize;
    }

    let out_index = b * grad_query.stride(0)
        + h * grad_query.stride(1)
        + t * grad_query.stride(2)
        + l * grad_query.stride(3);
    grad_query[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_value_kernel(
    reverse_state_rev: &Tensor<Line<f32>>,
    query: &Tensor<Line<f32>>,
    grad_value: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let z = CUBE_POS_Z as usize;
    let b = z / time.max(1);
    let t = z % time.max(1);
    let h = CUBE_POS_Y as usize;
    let e = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time || e >= embd {
        terminate!();
    }

    let tau = time - 1usize - t;
    let mut grad = Line::cast_from(0u32);
    let mut l = 0usize;
    while l < latent {
        let reverse_index = b * reverse_state_rev.stride(0)
            + h * reverse_state_rev.stride(1)
            + tau * reverse_state_rev.stride(2)
            + l * reverse_state_rev.stride(3)
            + e * reverse_state_rev.stride(4);
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        grad += reverse_state_rev[reverse_index] * query[query_index];
        l += 1usize;
    }

    let out_index = b * grad_value.stride(0)
        + h * grad_value.stride(1)
        + t * grad_value.stride(2)
        + e * grad_value.stride(3);
    grad_value[out_index] = grad;
}

#[cube(launch)]
fn recurrent_attention_grad_decay_kernel(
    forward_state: &Tensor<Line<f32>>,
    reverse_state_rev: &Tensor<Line<f32>>,
    query: &Tensor<Line<f32>>,
    value: &Tensor<Line<f32>>,
    decay: &Tensor<Line<f32>>,
    grad_decay_partial: &mut Tensor<Line<f32>>,
    params: &Tensor<Line<f32>>,
) {
    let batch = u32::cast_from(params[0]) as usize;
    let heads = u32::cast_from(params[1]) as usize;
    let time = u32::cast_from(params[2]) as usize;
    let latent = u32::cast_from(params[3]) as usize;
    let embd = u32::cast_from(params[4]) as usize;

    let b = CUBE_POS_Z as usize;
    let h = CUBE_POS_Y as usize;
    let t = (CUBE_POS_X * CUBE_DIM_X + UNIT_POS_X) as usize;
    if b >= batch || h >= heads || t >= time {
        terminate!();
    }

    let tau = time - 1usize - t;
    let decay_value = decay[h * decay.stride(0)];
    let eps = Line::cast_from(1.0e-8f32);
    let neg_eps = Line::cast_from(-1.0e-8f32);
    let safe_decay = if decay_value < eps && decay_value > neg_eps {
        eps
    } else {
        decay_value
    };

    let mut grad = Line::cast_from(0u32);
    let mut l = 0usize;
    while l < latent {
        let query_index =
            b * query.stride(0) + h * query.stride(1) + t * query.stride(2) + l * query.stride(3);
        let q = query[query_index];
        let mut e = 0usize;
        while e < embd {
            let forward_index = b * forward_state.stride(0)
                + h * forward_state.stride(1)
                + t * forward_state.stride(2)
                + l * forward_state.stride(3)
                + e * forward_state.stride(4);
            let reverse_index = b * reverse_state_rev.stride(0)
                + h * reverse_state_rev.stride(1)
                + tau * reverse_state_rev.stride(2)
                + l * reverse_state_rev.stride(3)
                + e * reverse_state_rev.stride(4);
            let value_index = b * value.stride(0)
                + h * value.stride(1)
                + t * value.stride(2)
                + e * value.stride(3);
            grad += (reverse_state_rev[reverse_index] / safe_decay)
                * (forward_state[forward_index] + q * value[value_index]);
            e += 1usize;
        }
        l += 1usize;
    }

    let out_index = b * grad_decay_partial.stride(0)
        + h * grad_decay_partial.stride(1)
        + t * grad_decay_partial.stride(2);
    grad_decay_partial[out_index] = grad;
}

fn backward_params_tensor<B: BackendTrait>(
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    embd: usize,
    device: &B::Device,
) -> BurnTensor<B, 1> {
    BurnTensor::<B, 1>::from_data(
        TensorData::new(
            vec![
                batch as f32,
                heads as f32,
                time as f32,
                latent as f32,
                embd as f32,
            ],
            [BACKWARD_META_LEN],
        ),
        device,
    )
}

fn reverse_time_indices<B: BackendTrait>(time: usize, device: &B::Device) -> BurnTensor<B, 1, Int> {
    BurnTensor::<B, 1, Int>::from_data(
        TensorData::new((0..time as i64).rev().collect::<Vec<_>>(), [time]),
        device,
    )
}

fn reverse_time_tensor4<B: BackendTrait>(tensor: BurnTensor<B, 4>) -> BurnTensor<B, 4> {
    let time = tensor.shape().dims::<4>()[2];
    let device = tensor.device();
    tensor.select(2, reverse_time_indices::<B>(time, &device))
}

fn recurrent_meta_tensor<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value_heads: usize,
    embd: usize,
) -> BurnTensor<B, 1> {
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    CompiledRecurrentAttentionPlan::new(
        batch,
        heads,
        value_heads,
        time,
        latent,
        embd,
        &query.device(),
    )
    .meta()
}

fn recurrent_attention_grad_query_runtime<R: CubeRuntime>(
    forward_state: CubeTensor<R>,
    reverse_state_rev: CubeTensor<R>,
    grad_output: CubeTensor<R>,
    value: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, latent, _] = forward_state.meta.shape.dims::<5>();
    let client = forward_state.client.clone();
    let device = forward_state.device.clone();
    let grad_query = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(latent as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );
    let _ = recurrent_attention_grad_query_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(forward_state).as_tensor_arg(1),
        into_contiguous(reverse_state_rev).as_tensor_arg(1),
        into_contiguous(grad_output).as_tensor_arg(1),
        into_contiguous(value).as_tensor_arg(1),
        grad_query.as_tensor_arg(1),
        into_contiguous(params).as_tensor_arg(1),
    );
    grad_query
}

fn recurrent_attention_grad_value_runtime<R: CubeRuntime>(
    reverse_state_rev: CubeTensor<R>,
    query: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, _, embd] = reverse_state_rev.meta.shape.dims::<5>();
    let client = reverse_state_rev.client.clone();
    let device = reverse_state_rev.device.clone();
    let grad_value = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        (batch * time) as u32,
    );
    let _ = recurrent_attention_grad_value_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(reverse_state_rev).as_tensor_arg(1),
        into_contiguous(query).as_tensor_arg(1),
        grad_value.as_tensor_arg(1),
        into_contiguous(params).as_tensor_arg(1),
    );
    grad_value
}

fn recurrent_attention_grad_decay_runtime<R: CubeRuntime>(
    forward_state: CubeTensor<R>,
    reverse_state_rev: CubeTensor<R>,
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    params: CubeTensor<R>,
) -> CubeTensor<R> {
    let [batch, heads, time, _, _] = forward_state.meta.shape.dims::<5>();
    let client = forward_state.client.clone();
    let device = forward_state.device.clone();
    let grad_decay_partial =
        empty_device::<R, f32>(client.clone(), device, Shape::new([batch, heads, time]));
    let cube_dim = CubeDim::new_1d(BACKWARD_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(time as u32, BACKWARD_WORKGROUP_SIZE_X),
        heads as u32,
        batch as u32,
    );
    let _ = recurrent_attention_grad_decay_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        into_contiguous(forward_state).as_tensor_arg(1),
        into_contiguous(reverse_state_rev).as_tensor_arg(1),
        into_contiguous(query).as_tensor_arg(1),
        into_contiguous(value).as_tensor_arg(1),
        into_contiguous(decay).as_tensor_arg(1),
        grad_decay_partial.as_tensor_arg(1),
        into_contiguous(params).as_tensor_arg(1),
    );
    grad_decay_partial
}

pub(super) fn recurrent_attention_reverse_state_history<B: BackendTrait>(
    query: BurnTensor<B, 4>,
    grad_output: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
) -> Option<(BurnTensor<B, 5>, BurnTensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, _time, latent] = query.shape().dims::<4>();
    let embd = grad_output.shape().dims::<4>()[3];
    let query_rev = reverse_time_tensor4(query);
    let grad_output_rev = reverse_time_tensor4(grad_output);
    let zero_rho = BurnTensor::<B, 4>::zeros([batch, heads, latent, embd], &query_rev.device());
    let meta = recurrent_meta_tensor(&query_rev, heads, embd);

    let captured = try_direct_path_runtime_with_state_history::<B, WgpuRuntime>(
        &query_rev,
        &grad_output_rev,
        &zero_rho,
        &decay,
        &meta,
    )
    .or_else(|| {
        #[cfg(feature = "cuda")]
        {
            try_direct_path_runtime_with_state_history::<B, CudaRuntime>(
                &query_rev,
                &grad_output_rev,
                &zero_rho,
                &decay,
                &meta,
            )
        }
        #[cfg(not(feature = "cuda"))]
        {
            None
        }
    })?;

    Some((captured.state_history, captured.rho))
}

pub(super) fn recurrent_attention_backward_impl<B: BackendTrait>(
    ops: Ops<RecurrentAttentionBackwardState<B::FloatTensorPrimitive>, 4>,
    grads: &mut Gradients,
) where
    B::FloatTensorPrimitive: 'static,
{
    let grad_output =
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
    let RecurrentAttentionBackwardState {
        query,
        value,
        rho,
        decay,
        state_history,
    } = ops.state;
    let parents = ops.parents;

    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value));
    let rho = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho));
    let decay = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(decay));
    let state_history = BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(state_history));

    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let embd = value.shape().dims::<4>()[3];
    let value_per_head = if value_heads == heads {
        value.clone()
    } else {
        value.clone().repeat_dim(1, heads)
    };

    let (reverse_state_rev, reverse_final_rho) = recurrent_attention_reverse_state_history(
        query.clone(),
        grad_output.clone(),
        decay.clone(),
    )
    .expect("recurrent custom backward reverse state history");
    let params = backward_params_tensor(batch, heads, time, latent, embd, &query.device());

    if let Some(parent) = &parents[0] {
        let grad_query = try_direct_recurrent_grad_query::<B>(
            state_history.clone(),
            reverse_state_rev.clone(),
            grad_output.clone(),
            value_per_head.clone(),
            params.clone(),
        )
        .expect("recurrent grad_query runtime");
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }

    if let Some(parent) = &parents[1] {
        let grad_value_heads = try_direct_recurrent_grad_value::<B>(
            reverse_state_rev.clone(),
            query.clone(),
            params.clone(),
        )
        .expect("recurrent grad_value runtime");
        let grad_value = if value_heads == heads {
            grad_value_heads
        } else {
            grad_value_heads.sum_dim(1).reshape([batch, 1, time, embd])
        };
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }

    if let Some(parent) = &parents[2] {
        let decay_safe = decay.clone().add_scalar(1.0e-8).reshape([1, heads, 1, 1]);
        let grad_rho = reverse_final_rho.div(decay_safe);
        let _ = rho;
        grads.register::<B>(parent.id, grad_rho.into_primitive().tensor());
    }

    if let Some(parent) = &parents[3] {
        let grad_decay_partial = try_direct_recurrent_grad_decay::<B>(
            state_history,
            reverse_state_rev,
            query,
            value_per_head,
            decay.clone(),
            params,
        )
        .expect("recurrent grad_decay runtime");
        let grad_decay = grad_decay_partial.sum_dim(2).sum_dim(0);
        grads.register::<B>(parent.id, grad_decay.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 4> for FusedRecurrentAttentionBackward<WgpuCubeBackend> {
    type State = RecurrentAttentionBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        recurrent_attention_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 4> for FusedRecurrentAttentionBackward<CudaCubeBackend> {
    type State = RecurrentAttentionBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 4>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        recurrent_attention_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

fn try_direct_recurrent_grad_query<B: BackendTrait>(
    forward_state: BurnTensor<B, 5>,
    reverse_state_rev: BurnTensor<B, 5>,
    grad_output: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_forward = forward_state.into_primitive().tensor();
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_grad = grad_output.into_primitive().tensor();
    let prim_value = value.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(forward_state) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_forward.clone())
    {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse.clone())?;
        let grad_output = try_cast_primitive::<B, _>(prim_grad.clone())?;
        let value = try_cast_primitive::<B, _>(prim_value.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor = recurrent_attention_grad_query_runtime::<WgpuRuntime>(
            forward_state,
            reverse_state_rev,
            grad_output,
            value,
            params,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(forward_state) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_forward) {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse)?;
        let grad_output = try_cast_primitive::<B, _>(prim_grad)?;
        let value = try_cast_primitive::<B, _>(prim_value)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor = recurrent_attention_grad_query_runtime::<CudaRuntime>(
            forward_state,
            reverse_state_rev,
            grad_output,
            value,
            params,
        );
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn try_direct_recurrent_grad_value<B: BackendTrait>(
    reverse_state_rev: BurnTensor<B, 5>,
    query: BurnTensor<B, 4>,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_query = query.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(reverse_state_rev) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_reverse.clone())
    {
        let query = try_cast_primitive::<B, _>(prim_query.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor =
            recurrent_attention_grad_value_runtime::<WgpuRuntime>(reverse_state_rev, query, params);
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(reverse_state_rev) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_reverse)
    {
        let query = try_cast_primitive::<B, _>(prim_query)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor =
            recurrent_attention_grad_value_runtime::<CudaRuntime>(reverse_state_rev, query, params);
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn try_direct_recurrent_grad_decay<B: BackendTrait>(
    forward_state: BurnTensor<B, 5>,
    reverse_state_rev: BurnTensor<B, 5>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
    params: BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_forward = forward_state.into_primitive().tensor();
    let prim_reverse = reverse_state_rev.into_primitive().tensor();
    let prim_query = query.into_primitive().tensor();
    let prim_value = value.into_primitive().tensor();
    let prim_decay = decay.into_primitive().tensor();
    let prim_params = params.into_primitive().tensor();
    if let Some(forward_state) =
        try_cast_primitive::<B, CubeTensor<WgpuRuntime>>(prim_forward.clone())
    {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse.clone())?;
        let query = try_cast_primitive::<B, _>(prim_query.clone())?;
        let value = try_cast_primitive::<B, _>(prim_value.clone())?;
        let decay = try_cast_primitive::<B, _>(prim_decay.clone())?;
        let params = try_cast_primitive::<B, _>(prim_params.clone())?;
        let tensor = recurrent_attention_grad_decay_runtime::<WgpuRuntime>(
            forward_state,
            reverse_state_rev,
            query,
            value,
            decay,
            params,
        );
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    #[cfg(feature = "cuda")]
    if let Some(forward_state) = try_cast_primitive::<B, CubeTensor<CudaRuntime>>(prim_forward) {
        let reverse_state_rev = try_cast_primitive::<B, _>(prim_reverse)?;
        let query = try_cast_primitive::<B, _>(prim_query)?;
        let value = try_cast_primitive::<B, _>(prim_value)?;
        let decay = try_cast_primitive::<B, _>(prim_decay)?;
        let params = try_cast_primitive::<B, _>(prim_params)?;
        let tensor = recurrent_attention_grad_decay_runtime::<CudaRuntime>(
            forward_state,
            reverse_state_rev,
            query,
            value,
            decay,
            params,
        );
        return Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
            try_cast_backend::<B, _>(tensor)?,
        )));
    }

    None
}

fn recurrent_attention_autodiff_custom_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let rho_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(rho.clone().into_primitive().tensor())?;
    let decay_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let rho_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(rho_ad.clone());
    let decay_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let captured = try_direct_path_runtime_with_state_history::<WgpuCubeBackend, WgpuRuntime>(
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            rho_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        &BurnTensor::<WgpuCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            meta_inner.clone(),
        )),
    )?;

    let context_inner = captured.context.into_primitive().tensor();
    let rho_inner_out = captured.rho.into_primitive().tensor();
    let state_history_inner = captured.state_history.into_primitive().tensor();

    let context_ad = match FusedRecurrentAttentionBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            rho_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            RecurrentAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                rho: rho_inner,
                decay: decay_inner,
                state_history: state_history_inner,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <WgpuCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner_out),
        )?)),
    })
}

#[cfg(feature = "cuda")]
fn recurrent_attention_autodiff_custom_cuda<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let rho_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(rho.clone().into_primitive().tensor())?;
    let decay_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let rho_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(rho_ad.clone());
    let decay_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let captured = try_direct_path_runtime_with_state_history::<CudaCubeBackend, CudaRuntime>(
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            query_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            value_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 4>::from_primitive(TensorPrimitive::Float(
            rho_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            decay_inner.clone(),
        )),
        &BurnTensor::<CudaCubeBackend, 1>::from_primitive(TensorPrimitive::Float(
            meta_inner.clone(),
        )),
    )?;

    let context_inner = captured.context.into_primitive().tensor();
    let rho_inner_out = captured.rho.into_primitive().tensor();
    let state_history_inner = captured.state_history.into_primitive().tensor();

    let context_ad = match FusedRecurrentAttentionBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            rho_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            RecurrentAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                rho: rho_inner,
                decay: decay_inner,
                state_history: state_history_inner,
            },
            context_inner,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context_inner),
    };

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<
            B,
            _,
        >(context_ad)?)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(try_cast_backend::<B, _>(
            <CudaCubeAutodiffBackend as AutodiffBackend>::from_inner(rho_inner_out),
        )?)),
    })
}

pub(super) fn recurrent_attention_autodiff_custom<B: BackendTrait, R: CubeRuntime + 'static>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        return recurrent_attention_autodiff_custom_wgpu(query, value, rho, decay, meta);
    }
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return recurrent_attention_autodiff_custom_cuda(query, value, rho, decay, meta);
    }
    None
}
