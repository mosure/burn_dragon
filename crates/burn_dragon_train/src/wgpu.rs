use burn::tensor::backend::Backend as BackendTrait;
use burn_wgpu::{self, MemoryConfiguration, RuntimeOptions, Wgpu, graphics};

use crate::config::{WgpuBackend, WgpuMemoryConfig, WgpuRuntimeConfig};

/// The concrete device type used by the `Wgpu<f32>` backend.
pub type WgpuDevice = <Wgpu<f32> as BackendTrait>::Device;

/// Initialize the global wgpu runtime using config-driven overrides.
pub fn init_runtime(device: &WgpuDevice, config: &WgpuRuntimeConfig) {
    if matches!(device, WgpuDevice::Existing(_)) {
        return;
    }

    let options = runtime_options(config);
    match config.backend {
        WgpuBackend::Auto => {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, options);
        }
        WgpuBackend::Vulkan => {
            burn_wgpu::init_setup::<graphics::Vulkan>(device, options);
        }
        WgpuBackend::Dx12 => {
            burn_wgpu::init_setup::<graphics::Dx12>(device, options);
        }
        WgpuBackend::Metal => {
            burn_wgpu::init_setup::<graphics::Metal>(device, options);
        }
        WgpuBackend::OpenGl => {
            burn_wgpu::init_setup::<graphics::OpenGl>(device, options);
        }
    }
}

fn runtime_options(config: &WgpuRuntimeConfig) -> RuntimeOptions {
    let memory_config = match config.memory {
        WgpuMemoryConfig::SubSlices => MemoryConfiguration::SubSlices,
        WgpuMemoryConfig::Exclusive => MemoryConfiguration::ExclusivePages,
    };
    RuntimeOptions {
        tasks_max: config
            .tasks_max
            .unwrap_or(RuntimeOptions::default().tasks_max),
        memory_config,
    }
}
