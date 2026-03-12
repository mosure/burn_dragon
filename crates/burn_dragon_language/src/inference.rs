use burn_dragon_core::BDHConfig;
#[cfg(feature = "train")]
use burn_dragon_train::wgpu as shared_wgpu;

use crate::ModelOverrides;

/// Build a model configuration by applying training overrides.
pub fn build_model_config(overrides: &ModelOverrides, training_block_size: usize) -> BDHConfig {
    let mut model_config = BDHConfig::default();

    if let Some(n_layer) = overrides.n_layer {
        model_config.n_layer = n_layer;
    }
    if let Some(n_embd) = overrides.n_embd {
        model_config.n_embd = n_embd;
    }
    if let Some(n_head) = overrides.n_head {
        model_config.n_head = n_head;
    }
    if let Some(multiplier) = overrides.mlp_internal_dim_multiplier {
        model_config.mlp_internal_dim_multiplier = multiplier;
    }
    if let Some(relu_threshold) = overrides.relu_threshold {
        model_config.fused_kernels.relu_threshold = relu_threshold;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(enabled) = overrides.fused_kernels {
        model_config.fused_kernels.enabled = enabled;
    }
    let block = overrides.block_size.unwrap_or(training_block_size).max(1);
    model_config.fused_kernels.set_block_sizes(block, block);
    if let Some(rollout_fast_steps) = overrides.rollout_fast_steps_per_slow_step {
        model_config.set_rollout_fast_steps_per_slow_step(rollout_fast_steps);
    }
    if let Some(rotary_embedding) = overrides.rotary_embedding {
        model_config
            .fused_kernels
            .set_rotary_embedding(rotary_embedding);
    }
    if let Some(y_neuron_recurrence) = &overrides.y_neuron_recurrence {
        model_config.y_neuron_recurrence = y_neuron_recurrence.clone();
    }
    if let Some(mhc) = &overrides.mhc {
        model_config.mhc = mhc.clone();
    }

    model_config
}

pub fn is_wgpu_backend_name(backend_name: &str) -> bool {
    #[cfg(feature = "train")]
    {
        shared_wgpu::is_wgpu_backend_name(backend_name)
    }
    #[cfg(not(feature = "train"))]
    {
        backend_name.eq_ignore_ascii_case("wgpu")
            || backend_name
                .get(..5)
                .is_some_and(|prefix| prefix.eq_ignore_ascii_case("wgpu-"))
    }
}

pub fn apply_wgpu_fused_core_override(
    model_config: &mut BDHConfig,
    backend_name: &str,
    fused_core_recurrent: Option<bool>,
    fused_core_rollout: Option<bool>,
) {
    #[cfg(feature = "train")]
    {
        shared_wgpu::apply_wgpu_fused_core_override(
            model_config,
            backend_name,
            fused_core_recurrent,
            fused_core_rollout,
        );
    }

    #[cfg(not(feature = "train"))]
    {
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
}

#[cfg(test)]
mod tests {
    use super::{apply_wgpu_fused_core_override, build_model_config, is_wgpu_backend_name};
    use crate::ModelOverrides;
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
    fn wgpu_override_wrapper_delegates_to_shared_behavior() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(&mut model_config, "wgpu", Some(true), None);

        assert!(
            model_config.fused_kernels.enabled,
            "wgpu backend override should enable fused kernels for recurrent path selection"
        );
        assert!(
            model_config.fused_kernels.wgpu_recurrent_kernel,
            "wgpu recurrent kernel should be enabled by override"
        );
        assert!(
            model_config.fused_kernels.wgpu_rollout_fused,
            "wgpu rollout fused path should default to recurrent override when unspecified"
        );
    }

    #[test]
    fn model_override_applies_rollout_fast_steps() {
        let overrides = ModelOverrides {
            rollout_fast_steps_per_slow_step: Some(8),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.rollout_fast_steps_per_slow_step, 8);
    }
}
