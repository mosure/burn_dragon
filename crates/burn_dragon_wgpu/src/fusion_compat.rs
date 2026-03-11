use burn_cubecl::BoolElement;
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::{Client, FusionTensor, NoOp, stream::OperationStreams};
use burn_ir::{InitOperationIr, OperationIr, OperationOutput};
use burn_wgpu::{CubeBackend, WgpuRuntime};

pub(crate) fn register_fusion_float_tensor<BT: BoolElement + 'static>(
    client: &Client<FusionCubeRuntime<WgpuRuntime, BT>>,
    tensor: CubeTensor<WgpuRuntime>,
) -> FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> {
    let shape = tensor.meta.shape().clone();
    let dtype = tensor.dtype;
    let handle = tensor.into();
    let desc = InitOperationIr::create(shape, dtype, || client.register_tensor_handle(handle));

    client
        .register(
            OperationStreams::default(),
            OperationIr::Init(desc),
            NoOp::<CubeBackend<WgpuRuntime, f32, i32, BT>>::new(),
        )
        .output()
}
