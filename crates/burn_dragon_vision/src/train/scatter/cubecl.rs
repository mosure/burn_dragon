use std::any::{Any, TypeId};

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{DType, Shape, TensorPrimitive};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_fusion::FusionTensor;
use burn_fusion::stream::StreamId;
use burn_wgpu::WgpuRuntime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

pub(crate) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    #[cfg(feature = "cuda")]
    {
        matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>>()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
            >()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<CudaRuntime, u32>>,
            >()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<CudaRuntime, u8>>,
            >()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>()
    }
    #[cfg(not(feature = "cuda"))]
    {
        matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>>()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
            >()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
    }
}

pub(crate) fn try_weighted_sum_tokens_cubecl<B: BackendTrait>(
    weights: &BurnTensor<B, 3>,
    tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_backend::<B>() {
        return None;
    }
    let [batch, out_tokens, in_tokens] = weights.shape().dims::<3>();
    let [tok_batch, tok_in, dim] = tokens.shape().dims::<3>();
    if batch == 0 || out_tokens == 0 || in_tokens == 0 || dim == 0 {
        return None;
    }
    if batch != tok_batch || in_tokens != tok_in {
        return None;
    }

    if let Some(result) =
        try_weighted_sum_tokens_cubecl_fusion::<B, u32, WgpuRuntime>(weights, tokens)
    {
        return Some(result);
    }
    if let Some(result) =
        try_weighted_sum_tokens_cubecl_fusion::<B, u8, WgpuRuntime>(weights, tokens)
    {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    if let Some(result) =
        try_weighted_sum_tokens_cubecl_fusion::<B, u32, CudaRuntime>(weights, tokens)
    {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    if let Some(result) =
        try_weighted_sum_tokens_cubecl_fusion::<B, u8, CudaRuntime>(weights, tokens)
    {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(result) =
            try_weighted_sum_tokens_cubecl_direct::<B, CudaRuntime>(weights, tokens)
        {
            return Some(result);
        }
    }
    try_weighted_sum_tokens_cubecl_direct::<B, WgpuRuntime>(weights, tokens)
}

fn try_weighted_sum_tokens_cubecl_fusion<B, BT, R>(
    weights: &BurnTensor<B, 3>,
    tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R, BT>>>() {
        return None;
    }
    let prim_weights = weights.clone().into_primitive().tensor();
    let fusion_weights: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim_weights)?;
    let fusion_client = fusion_weights.client.clone();
    let weights =
        fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_weights);
    if weights.dtype != DType::F32 {
        return None;
    }

    let prim_tokens = tokens.clone().into_primitive().tensor();
    let fusion_tokens: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim_tokens)?;
    let tokens = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_tokens);
    if tokens.dtype != DType::F32 {
        return None;
    }

    let output = weighted_sum_tokens_cubecl_runtime::<R>(weights, tokens);
    let shape = output.shape.clone();
    let dtype = output.dtype;
    let handle = output.into();
    let fusion_out = fusion_client.register_tensor(handle, shape, StreamId::current(), dtype);
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
        out_prim,
    )))
}

fn try_weighted_sum_tokens_cubecl_direct<B, R>(
    weights: &BurnTensor<B, 3>,
    tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<R>>() {
        return None;
    }
    let prim_weights = weights.clone().into_primitive().tensor();
    let weights: CubeTensor<R> = try_cast_primitive::<B, _>(prim_weights)?;
    if weights.dtype != DType::F32 {
        return None;
    }
    let prim_tokens = tokens.clone().into_primitive().tensor();
    let tokens: CubeTensor<R> = try_cast_primitive::<B, _>(prim_tokens)?;
    if tokens.dtype != DType::F32 {
        return None;
    }

    let output = weighted_sum_tokens_cubecl_runtime::<R>(weights, tokens);
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(
        out_prim,
    )))
}

fn weighted_sum_tokens_cubecl_runtime<R: CubeRuntime>(
    weights: CubeTensor<R>,
    tokens: CubeTensor<R>,
) -> CubeTensor<R> {
    let weights = into_contiguous(weights);
    let tokens = into_contiguous(tokens);
    let [batch, out_tokens, _] = weights.shape.dims::<3>();
    let dim = tokens.shape.dims::<3>()[2];

    let client = weights.client.clone();
    let device = weights.device.clone();
    let output =
        empty_device::<R, f32>(client.clone(), device, Shape::new([batch, out_tokens, dim]));
    let out_elems = output.shape.num_elements();
    let cube_dim = CubeDim::new(256, 1, 1);
    let cube_count = calculate_cube_count_elemwise(out_elems, cube_dim);

    weighted_sum_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        weights.as_tensor_arg::<f32>(1),
        tokens.as_tensor_arg::<f32>(1),
        output.as_tensor_arg::<f32>(1),
    );
    output
}

#[cube(launch)]
fn weighted_sum_kernel(weights: &Tensor<f32>, tokens: &Tensor<f32>, output: &mut Tensor<f32>) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let dim = output.shape(2);
    let out_tokens = output.shape(1);
    if dim == 0 || out_tokens == 0 {
        terminate!();
    }
    let d = ABSOLUTE_POS % dim;
    let pos = ABSOLUTE_POS / dim;
    let o = pos % out_tokens;
    let b = pos / out_tokens;
    let in_tokens = weights.shape(2);

    let weight_base = b * weights.stride(0) + o * weights.stride(1);
    let token_base = b * tokens.stride(0) + d * tokens.stride(2);

    let mut acc = 0.0f32;
    let mut i = 0u32;
    while i < in_tokens {
        let w_idx = weight_base + i * weights.stride(2);
        let t_idx = token_base + i * tokens.stride(1);
        acc += weights[w_idx] * tokens[t_idx];
        i += 1u32;
    }

    let out_idx = b * output.stride(0) + o * output.stride(1) + d * output.stride(2);
    output[out_idx] = acc;
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
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
