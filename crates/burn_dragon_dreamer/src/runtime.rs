use anyhow::Result;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend as BurnBackend;
use burn_autodiff::Autodiff;
use burn_cuda::{Cuda, CudaDevice};

pub type Backend = Cuda<f32>;
pub type TrainBackend = Autodiff<Backend>;

pub fn cuda_device() -> Result<CudaDevice> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::default))
        .map_err(|_| anyhow::anyhow!("CUDA device initialization failed"))
}

pub fn train_to_runtime_tensor5(
    tensor: Tensor<TrainBackend, 5>,
    _device: &<Backend as BurnBackend>::Device,
) -> Tensor<Backend, 5> {
    tensor.inner()
}

pub fn train_to_runtime_tensor3(
    tensor: Tensor<TrainBackend, 3>,
    _device: &<Backend as BurnBackend>::Device,
) -> Tensor<Backend, 3> {
    tensor.inner()
}

pub fn runtime_to_train_tensor3(
    tensor: Tensor<Backend, 3>,
    _device: &<TrainBackend as BurnBackend>::Device,
) -> Tensor<TrainBackend, 3> {
    Tensor::<TrainBackend, 3>::from_inner(tensor)
}
