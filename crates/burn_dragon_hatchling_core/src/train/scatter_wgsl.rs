use std::any::{Any, TypeId};

use burn::tensor::{DType, Shape, TensorData, TensorPrimitive};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::Tensor as BurnTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::FusionTensor;
use burn_fusion::stream::StreamId;
use burn_wgpu::{KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use cubecl::prelude::*;
use cubecl_runtime::server::Bindings;

use burn_dragon_hatchling_vision::SCATTER_BUFFER_SHADER;

const META_LEN: usize = 4;
const WORKGROUP_SIZE: u32 = 8;

pub(super) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<
        B::FloatTensorPrimitive,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>,
    >() || matches_type::<
        B::FloatTensorPrimitive,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
    >() || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
}

pub(super) fn try_weighted_sum_tokens_wgsl<B: BackendTrait>(
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

    if let Some(result) = try_weighted_sum_tokens_wgsl_fusion::<B, u32>(weights, tokens) {
        return Some(result);
    }
    if let Some(result) = try_weighted_sum_tokens_wgsl_fusion::<B, u8>(weights, tokens) {
        return Some(result);
    }
    try_weighted_sum_tokens_wgsl_direct::<B>(weights, tokens)
}

fn try_weighted_sum_tokens_wgsl_fusion<B: BackendTrait, BT: BoolElement>(
    weights: &BurnTensor<B, 3>,
    tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>>() {
        return None;
    }
    let device = weights.device();
    let prim_weights = weights.clone().into_primitive().tensor();
    let fusion_weights: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim_weights)?;
    let fusion_client = fusion_weights.client.clone();
    let weights = fusion_client
        .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_weights);
    if weights.dtype != DType::F32 {
        return None;
    }

    let prim_tokens = tokens.clone().into_primitive().tensor();
    let fusion_tokens: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim_tokens)?;
    let tokens = fusion_client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_tokens);
    if tokens.dtype != DType::F32 {
        return None;
    }

    let meta =
        build_meta::<B>(weights.shape.dims::<3>(), tokens.shape.dims::<3>(), device);
    let meta = resolve_fusion_tensor::<B, BT>(&meta)?;
    let output = weighted_sum_tokens_wgsl_runtime::<WgpuRuntime>(weights, tokens, meta);
    let shape = output.shape.clone();
    let dtype = output.dtype;
    let handle = output.into();
    let fusion_out = fusion_client.register_tensor(handle, shape, StreamId::current(), dtype);
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn try_weighted_sum_tokens_wgsl_direct<B: BackendTrait>(
    weights: &BurnTensor<B, 3>,
    tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>() {
        return None;
    }
    let device = weights.device();
    let prim_weights = weights.clone().into_primitive().tensor();
    let weights: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_weights)?;
    if weights.dtype != DType::F32 {
        return None;
    }
    let prim_tokens = tokens.clone().into_primitive().tensor();
    let tokens: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim_tokens)?;
    if tokens.dtype != DType::F32 {
        return None;
    }

    let meta =
        build_meta::<B>(weights.shape.dims::<3>(), tokens.shape.dims::<3>(), device);
    let meta = resolve_direct_tensor::<B>(&meta)?;
    let output = weighted_sum_tokens_wgsl_runtime::<WgpuRuntime>(weights, tokens, meta);
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 3>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn build_meta<B: BackendTrait>(
    weights_shape: [usize; 3],
    tokens_shape: [usize; 3],
    device: B::Device,
) -> BurnTensor<B, 1> {
    let meta = [
        weights_shape[0] as f32,
        weights_shape[1] as f32,
        weights_shape[2] as f32,
        tokens_shape[2] as f32,
    ];
    BurnTensor::<B, 1>::from_data(TensorData::new(meta.to_vec(), [META_LEN]), &device)
}

fn weighted_sum_tokens_wgsl_runtime<R: CubeRuntime>(
    weights: CubeTensor<R>,
    tokens: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let weights = into_contiguous(weights);
    let tokens = into_contiguous(tokens);
    let meta = into_contiguous(meta);

    let [batch, out_tokens, _] = weights.shape.dims::<3>();
    let dim = tokens.shape.dims::<3>()[2];
    let client = weights.client.clone();
    let device = weights.device.clone();
    let output = empty_device::<R, f32>(client.clone(), device, Shape::new([batch, out_tokens, dim]));

    let workgroups_x = div_ceil_u32(out_tokens as u32, WORKGROUP_SIZE);
    let workgroups_y = div_ceil_u32(dim as u32, WORKGROUP_SIZE);
    let count = CubeCount::Static(workgroups_x, workgroups_y, batch as u32);

    let kernel = SourceKernel::new(ScatterBufferKernel, CubeDim::new(WORKGROUP_SIZE, WORKGROUP_SIZE, 1));
    let bindings = Bindings::new().with_buffers(vec![
        weights.handle.clone().binding(),
        tokens.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.execute(Box::new(kernel), count, bindings);
    output
}

fn resolve_fusion_tensor<B: BackendTrait, BT: BoolElement>(
    tensor: &BurnTensor<B, 1>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn resolve_direct_tensor<B: BackendTrait>(
    tensor: &BurnTensor<B, 1>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let cube: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim)?;
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    (value + divisor - 1) / divisor
}

#[derive(Clone)]
struct ScatterBufferKernel;

impl KernelSource for ScatterBufferKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(SCATTER_BUFFER_SHADER)
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(
    value: B::FloatTensorPrimitive,
) -> Option<T>
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
