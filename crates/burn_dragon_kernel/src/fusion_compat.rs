use std::any::TypeId;

#[cfg(not(feature = "cuda"))]
use burn_cubecl::cubecl::wgpu::WgpuRuntime;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_cubecl::{BoolElement, CubeRuntime};
#[cfg(feature = "cuda")]
use burn_cubecl::{cubecl::cuda::CudaRuntime, cubecl::wgpu::WgpuRuntime};
use burn_fusion::{Client, FusionTensor, NoOp, stream::OperationStreams};
use burn_ir::{InitOperationIr, OperationIr, OperationOutput};
use burn_wgpu::CubeBackend;

fn register_fusion_float_tensor_with_bool<R: CubeRuntime, BT: BoolElement + 'static>(
    client: &Client<FusionCubeRuntime<R>>,
    tensor: CubeTensor<R>,
) -> FusionTensor<FusionCubeRuntime<R>> {
    let shape = tensor.meta.shape().clone();
    let dtype = tensor.dtype;
    let handle = tensor.into();
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

    client
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<CubeBackend<R, f32, i32, BT>>::new(),
        )
        .output()
}

fn register_fusion_int_tensor_with_bool<R: CubeRuntime, BT: BoolElement + 'static>(
    client: &Client<FusionCubeRuntime<R>>,
    tensor: CubeTensor<R>,
) -> FusionTensor<FusionCubeRuntime<R>> {
    let shape = tensor.meta.shape().clone();
    let dtype = tensor.dtype;
    let handle = tensor.into();
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

    client
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<CubeBackend<R, f32, i32, BT>>::new(),
        )
        .output()
}

pub(crate) fn register_fusion_float_tensor<R: CubeRuntime + 'static>(
    client: &Client<FusionCubeRuntime<R>>,
    tensor: CubeTensor<R>,
) -> FusionTensor<FusionCubeRuntime<R>> {
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        return register_fusion_float_tensor_with_bool::<R, u32>(client, tensor);
    }
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return register_fusion_float_tensor_with_bool::<R, u8>(client, tensor);
    }
    panic!(
        "unsupported fusion runtime for float tensor registration: {}",
        std::any::type_name::<R>()
    );
}

pub(crate) fn register_fusion_int_tensor<R: CubeRuntime + 'static>(
    client: &Client<FusionCubeRuntime<R>>,
    tensor: CubeTensor<R>,
) -> FusionTensor<FusionCubeRuntime<R>> {
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        return register_fusion_int_tensor_with_bool::<R, u32>(client, tensor);
    }
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return register_fusion_int_tensor_with_bool::<R, u8>(client, tensor);
    }
    panic!(
        "unsupported fusion runtime for int tensor registration: {}",
        std::any::type_name::<R>()
    );
}
