use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_core::BDHConfig;
use burn_wgpu::{self, MemoryConfiguration, RuntimeOptions, Wgpu, graphics};

use crate::config::{WgpuBackend, WgpuMemoryConfig, WgpuRuntimeConfig};

/// The concrete device type used by the `Wgpu<f32>` backend.
pub type WgpuDevice = <Wgpu<f32> as BackendTrait>::Device;

pub fn is_wgpu_backend_name(backend_name: &str) -> bool {
    backend_name.eq_ignore_ascii_case("wgpu")
        || backend_name
            .get(..5)
            .is_some_and(|prefix| prefix.eq_ignore_ascii_case("wgpu-"))
}

pub fn apply_wgpu_fused_core_override(
    model_config: &mut BDHConfig,
    backend_name: &str,
    fused_core_recurrent: Option<bool>,
    fused_core_rollout: Option<bool>,
) {
    if !is_wgpu_backend_name(backend_name) {
        return;
    }

    if let Some(enabled) = fused_core_recurrent {
        model_config
            .fused_kernels
            .set_wgpu_recurrent_kernel(enabled);
        if enabled {
            model_config.fused_kernels.enabled = true;
        }
    }

    let rollout_override = fused_core_rollout.or(fused_core_recurrent);
    if let Some(enabled) = rollout_override {
        model_config.fused_kernels.set_wgpu_rollout_fused(enabled);
    }
}

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

#[cfg(test)]
mod tests {
    use super::{apply_wgpu_fused_core_override, is_wgpu_backend_name};
    use burn_dragon_core::BDHConfig;

    #[test]
    fn backend_name_detection_accepts_wgpu_variants() {
        assert!(is_wgpu_backend_name("wgpu"));
        assert!(is_wgpu_backend_name("WGPU"));
        assert!(is_wgpu_backend_name("wgpu-fused-core"));
        assert!(is_wgpu_backend_name("wgpu-nofusion"));
        assert!(!is_wgpu_backend_name("cuda"));
    }

    #[test]
    fn override_enables_fused_recurrent_and_rollout_by_default() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(&mut model_config, "wgpu", Some(true), None);

        assert!(model_config.fused_kernels.enabled);
        assert!(model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn explicit_rollout_override_does_not_follow_recurrent() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = true;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(true);
        model_config.fused_kernels.set_wgpu_rollout_fused(true);

        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu-fused-core",
            Some(true),
            Some(false),
        );

        assert!(model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn non_wgpu_backend_ignores_override() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(&mut model_config, "cuda", Some(true), Some(true));

        assert!(!model_config.fused_kernels.enabled);
        assert!(!model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }
}
