use std::any::Any;

use burn::tensor::backend::Backend as BackendTrait;
use burn_cubecl::cubecl::Runtime;

#[cfg(all(feature = "cuda", any(feature = "cli", feature = "train")))]
use burn_cuda::CudaDevice;
#[cfg(any(feature = "cli", feature = "train"))]
use burn_wgpu::WgpuDevice;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceMemoryUsage {
    pub reserved_bytes: u64,
    pub in_use_bytes: u64,
}

impl DeviceMemoryUsage {
    pub fn reserved_mb(self) -> f64 {
        bytes_to_mb(self.reserved_bytes)
    }

    pub fn in_use_mb(self) -> f64 {
        bytes_to_mb(self.in_use_bytes)
    }
}

pub fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

pub fn device_memory_usage<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        let usage =
            <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }

    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        let usage = <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }

    None
}

pub fn device_memory_usage_safe<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| device_memory_usage::<B>(device)))
        .ok()
        .flatten()
}

pub fn cleanup_device_memory<B: BackendTrait>(
    device: &B::Device,
    allow_cuda_cleanup: bool,
) -> bool
where
    B::Device: 'static,
{
    if !cleanup_device_memory_allowed::<B>(device, allow_cuda_cleanup) {
        return false;
    }

    let _guard = crate::device::device_allocation_lock().lock().ok();
    let _ = B::sync(device);
    B::memory_cleanup(device);
    extra_memory_cleanup::<B>(device);
    let _ = B::sync(device);
    true
}

pub fn cleanup_device_memory_allowed<B: BackendTrait>(
    device: &B::Device,
    allow_cuda_cleanup: bool,
) -> bool
where
    B::Device: 'static,
{
    allow_memory_cleanup::<B>(device, allow_cuda_cleanup)
}

fn extra_memory_cleanup<B: BackendTrait>(device: &B::Device)
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_cleanup();
    }

    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_cleanup();
    }
}

fn allow_memory_cleanup<B: BackendTrait>(_device: &B::Device, _allow_cuda_cleanup: bool) -> bool
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if (_device as &dyn Any).downcast_ref::<CudaDevice>().is_some() {
        return _allow_cuda_cleanup;
    }

    true
}
