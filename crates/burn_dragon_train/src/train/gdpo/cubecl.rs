use std::any::{Any, TypeId};

use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{DType, Shape, TensorPrimitive};
use burn_cubecl::cubecl;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
use burn_cubecl::cubecl::{calculate_cube_count_elemwise, prelude::*};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_fusion::{FusionTensor, NoOp, stream::OperationStreams};
use burn_ir::{InitOperationIr, OperationIr, OperationOutput};
use burn_wgpu::WgpuRuntime;

pub const MAX_GROUP: usize = 8;

pub fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    #[cfg(feature = "cuda")]
    {
        matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
            || matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<CudaRuntime>>>(
            )
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>()
    }
    #[cfg(not(feature = "cuda"))]
    {
        matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime>>>()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
    }
}

pub fn try_percentile_thresholds_cubecl<B: BackendTrait>(
    values: &BurnTensor<B, 2>,
    quantile: f32,
) -> Option<BurnTensor<B, 2>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_backend::<B>() {
        return None;
    }
    let [batch, group] = values.shape().dims::<2>();
    if batch == 0 || group == 0 {
        return Some(BurnTensor::<B, 2>::zeros(
            [batch.max(1), 1],
            &values.device(),
        ));
    }
    if group > MAX_GROUP {
        return None;
    }
    let quantile = if quantile.is_nan() {
        0.0
    } else {
        quantile.clamp(0.0, 1.0)
    };

    if let Some(result) =
        try_percentile_thresholds_cubecl_fusion::<B, u32, WgpuRuntime>(values, quantile)
    {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    if let Some(result) =
        try_percentile_thresholds_cubecl_fusion::<B, u8, CudaRuntime>(values, quantile)
    {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(result) =
            try_percentile_thresholds_cubecl_direct::<B, CudaRuntime>(values, quantile)
        {
            return Some(result);
        }
    }
    try_percentile_thresholds_cubecl_direct::<B, WgpuRuntime>(values, quantile)
}

fn try_percentile_thresholds_cubecl_fusion<B, BT, R>(
    values: &BurnTensor<B, 2>,
    quantile: f32,
) -> Option<BurnTensor<B, 2>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R>>>() {
        return None;
    }
    let prim_values = values.clone().into_primitive().tensor();
    let fusion_values: FusionTensor<FusionCubeRuntime<R>> =
        try_cast_primitive::<B, _>(prim_values)?;
    let fusion_client = fusion_values.client.clone();
    let values = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_values);
    if values.dtype != DType::F32 {
        return None;
    }

    let output = percentile_thresholds_cubecl_runtime::<R>(values, quantile);
    let shape = output.meta.shape().clone();
    let dtype = output.dtype;
    let handle = output.into();
    let desc = InitOperationIr::create(shape, dtype, || {
        fusion_client.register_tensor_handle(handle)
    });
    let fusion_out = fusion_client
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<CubeBackend<R, f32, i32, BT>>::new(),
        )
        .output();
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
        out_prim,
    )))
}

fn try_percentile_thresholds_cubecl_direct<B, R>(
    values: &BurnTensor<B, 2>,
    quantile: f32,
) -> Option<BurnTensor<B, 2>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<R>>() {
        return None;
    }
    let prim_values = values.clone().into_primitive().tensor();
    let values: CubeTensor<R> = try_cast_primitive::<B, _>(prim_values)?;
    if values.dtype != DType::F32 {
        return None;
    }

    let output = percentile_thresholds_cubecl_runtime::<R>(values, quantile);
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 2>::from_primitive(TensorPrimitive::Float(
        out_prim,
    )))
}

fn percentile_thresholds_cubecl_runtime<R: CubeRuntime>(
    values: CubeTensor<R>,
    quantile: f32,
) -> CubeTensor<R> {
    let values = into_contiguous(values);
    let [batch, _group] = values.meta.shape.dims::<2>();

    let client = values.client.clone();
    let device = values.device.clone();
    let output = empty_device::<R, f32>(client.clone(), device, Shape::new([batch, 1]));
    let out_elems = output.meta.shape.num_elements();
    let cube_dim = CubeDim::new_3d(1, 1, 1);
    let cube_count = calculate_cube_count_elemwise(&client, out_elems, cube_dim);

    let _ = percentile_thresholds_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        values.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        quantile,
    );

    output
}

#[cube(launch)]
fn percentile_thresholds_kernel(values: &Tensor<f32>, output: &mut Tensor<f32>, quantile: f32) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let batch = values.shape(0);
    let group = values.shape(1);
    if batch == 0 || group == 0 {
        terminate!();
    }
    if group > MAX_GROUP {
        let out_idx = ABSOLUTE_POS * output.stride(0);
        output[out_idx] = 0.0f32;
        terminate!();
    }

    let mut scratch = SharedMemory::<f32>::new(MAX_GROUP);
    let base = ABSOLUTE_POS * values.stride(0);
    let mut i = 0usize;
    while i < group {
        let idx = base + i * values.stride(1);
        let mut value = values[idx];
        if <f32 as IsNan>::is_nan(value) {
            value = 0.0f32;
        } else if <f32 as IsInf>::is_inf(value) {
            value = if value < 0.0f32 {
                f32::new(-3.4028235e38f32)
            } else {
                f32::new(3.4028235e38f32)
            };
        }
        scratch[i] = value;
        i += 1usize;
    }

    let out_idx = ABSOLUTE_POS * output.stride(0);
    if group == 1usize {
        output[out_idx] = scratch[0];
        terminate!();
    }
    if group == 2usize {
        let a = scratch[0];
        let b = scratch[1];
        let lo = if a < b { a } else { b };
        let hi = if a < b { b } else { a };
        output[out_idx] = lo + (hi - lo) * quantile;
        terminate!();
    }

    let mut i = 1usize;
    while i < group {
        let mut j = i;
        while j > 0usize {
            let prev = scratch[j - 1usize];
            let curr = scratch[j];
            if curr < prev {
                scratch[j - 1usize] = curr;
                scratch[j] = prev;
            }
            j -= 1usize;
        }
        i += 1usize;
    }

    let pos = (group - 1usize) as f32 * quantile;
    let lower = pos as usize;
    let upper = if lower + 1usize < group {
        lower + 1usize
    } else {
        lower
    };
    let weight = pos - lower as f32;
    let lower_val = scratch[lower];
    let upper_val = scratch[upper];
    output[out_idx] = lower_val + (upper_val - lower_val) * weight;
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
