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

/// Optional fused-core overrides applied only when the active backend is WGPU.
///
/// `rollout` falls back to `recurrent` when it is not provided so callers can express
/// "set both" without repeating themselves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WgpuFusedCoreOverride {
    pub recurrent: Option<bool>,
    pub rollout: Option<bool>,
}

pub fn apply_wgpu_fused_core_override(
    model_config: &mut BDHConfig,
    backend_name: &str,
    override_config: WgpuFusedCoreOverride,
) {
    if !is_wgpu_backend_name(backend_name) {
        return;
    }

    if let Some(enabled) = override_config.recurrent {
        model_config
            .fused_kernels
            .set_wgpu_recurrent_kernel(enabled);
        if enabled {
            model_config.fused_kernels.enabled = true;
        }
    }

    let rollout_override = override_config.rollout.or(override_config.recurrent);
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
    use super::{WgpuFusedCoreOverride, apply_wgpu_fused_core_override, is_wgpu_backend_name};
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

        apply_wgpu_fused_core_override(
            &mut model_config,
            "wgpu",
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: None,
            },
        );

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
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: Some(false),
            },
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

        apply_wgpu_fused_core_override(
            &mut model_config,
            "cuda",
            WgpuFusedCoreOverride {
                recurrent: Some(true),
                rollout: Some(true),
            },
        );

        assert!(!model_config.fused_kernels.enabled);
        assert!(!model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }
}
