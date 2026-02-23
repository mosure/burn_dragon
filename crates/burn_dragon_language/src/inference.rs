use burn_dragon_core::BDHConfig;

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

    model_config
}

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
            // Selecting fused recurrent via backend config should work without requiring
            // an additional model-level fused toggle change.
            model_config.fused_kernels.enabled = true;
        }
    }

    let rollout_override = fused_core_rollout.or(match fused_core_recurrent {
        Some(enabled) => Some(enabled),
        None => None,
    });
    if let Some(enabled) = rollout_override {
        model_config.fused_kernels.set_wgpu_rollout_fused(enabled);
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
    fn wgpu_backend_override_enables_fused_recurrent_path() {
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
    fn wgpu_backend_override_can_disable_recurrent_kernel_without_disabling_other_fusion() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = true;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(true);
        model_config.fused_kernels.set_wgpu_rollout_fused(true);

        apply_wgpu_fused_core_override(&mut model_config, "wgpu-fused-core", Some(false), None);

        assert!(
            model_config.fused_kernels.enabled,
            "disabling recurrent override should preserve other fused kernel settings"
        );
        assert!(
            !model_config.fused_kernels.wgpu_recurrent_kernel,
            "wgpu recurrent kernel should be disabled by override"
        );
        assert!(
            !model_config.fused_kernels.wgpu_rollout_fused,
            "rollout override should follow recurrent override when unspecified"
        );
    }

    #[test]
    fn rollout_override_can_be_set_independently() {
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
    fn non_wgpu_backends_ignore_override() {
        let mut model_config = BDHConfig::default();
        model_config.fused_kernels.enabled = false;
        model_config.fused_kernels.set_wgpu_recurrent_kernel(false);
        model_config.fused_kernels.set_wgpu_rollout_fused(false);

        apply_wgpu_fused_core_override(&mut model_config, "cuda", Some(true), Some(true));

        assert!(!model_config.fused_kernels.enabled);
        assert!(!model_config.fused_kernels.wgpu_recurrent_kernel);
        assert!(!model_config.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn model_override_applies_rollout_fast_steps() {
        let mut overrides = ModelOverrides::default();
        overrides.rollout_fast_steps_per_slow_step = Some(8);

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.rollout_fast_steps_per_slow_step, 8);
    }
}
