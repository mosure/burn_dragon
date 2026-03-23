use burn_dragon_core::BDHConfig;
#[cfg(feature = "train")]
use burn_dragon_train::wgpu as shared_wgpu;

use crate::ModelOverrides;
use crate::summary_events::resolve_summary_memory_write_triggers;
use crate::tokenizer::Tokenizer;

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
    if let Some(latent_total) = overrides.latent_total {
        assert!(
            latent_total % model_config.n_embd == 0,
            "model.latent_total must be divisible by n_embd (got latent_total={} n_embd={})",
            latent_total,
            model_config.n_embd
        );
        model_config.mlp_internal_dim_multiplier = latent_total / model_config.n_embd;
    }
    if let Some(sequence_kernel) = overrides.sequence_kernel {
        model_config.sequence_kernel = sequence_kernel;
    }
    if let Some(mamba) = &overrides.mamba {
        model_config.mamba = mamba.clone();
    }
    if let Some(residual_connector) = overrides.residual_connector {
        model_config.residual_connector = residual_connector;
    }
    if let Some(attention_residual) = &overrides.attention_residual {
        model_config.attention_residual = attention_residual.clone();
    }
    if let Some(block_attention_residual) = &overrides.block_attention_residual {
        model_config.block_attention_residual = block_attention_residual.clone();
    }
    if let Some(schedule) = &overrides.latent_fanout_schedule {
        if let Err(message) = model_config.validate_latent_fanout_schedule(schedule) {
            panic!("{message}");
        }
        model_config.latent_fanout_schedule = Some(schedule.clone());
    }
    if let Some(relu_threshold) = overrides.relu_threshold {
        model_config.fused_kernels.relu_threshold = relu_threshold;
    }
    if let Some(dropout) = overrides.dropout {
        model_config.dropout = dropout;
    }
    if let Some(normalization) = &overrides.normalization {
        model_config.normalization = normalization.clone();
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
    if let Some(clocked_slow_memory) = &overrides.clocked_slow_memory {
        model_config.clocked_slow_memory = clocked_slow_memory.clone();
    }
    if let Some(summary_memory) = &overrides.summary_memory {
        model_config.summary_memory = summary_memory.clone();
    }
    if let Some(mhc) = &overrides.mhc {
        model_config.mhc = mhc.clone();
    }

    match overrides.residual_connector {
        Some(burn_dragon_core::ResidualConnectorKind::Vanilla) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::Mhc) => {
            model_config.mhc.enabled = true;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::AttentionResidual) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = true;
            model_config.block_attention_residual.enabled = false;
        }
        Some(burn_dragon_core::ResidualConnectorKind::BlockAttentionResidual) => {
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = true;
        }
        None => {
            model_config.residual_connector = burn_dragon_core::ResidualConnectorKind::Vanilla;
            model_config.mhc.enabled = false;
            model_config.attention_residual.enabled = false;
            model_config.block_attention_residual.enabled = false;
        }
    }

    model_config
}

pub fn build_model_config_with_tokenizer(
    overrides: &ModelOverrides,
    training_block_size: usize,
    tokenizer: &dyn Tokenizer,
) -> anyhow::Result<BDHConfig> {
    let mut model_config = build_model_config(overrides, training_block_size);
    resolve_summary_memory_write_triggers(&mut model_config, tokenizer)?;
    model_config.vocab_size = tokenizer.len();
    Ok(model_config)
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
    use burn_dragon_core::{BDHConfig, ResidualConnectorKind};

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

    #[test]
    fn model_override_applies_explicit_latent_total() {
        let overrides = ModelOverrides {
            n_embd: Some(256),
            latent_total: Some(32768),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.latent_total(), 32768);
        assert_eq!(config.mlp_internal_dim_multiplier, 128);
    }

    #[test]
    fn model_override_applies_latent_fanout_schedule() {
        let overrides = ModelOverrides {
            n_layer: Some(8),
            n_embd: Some(256),
            n_head: Some(4),
            latent_total: Some(32768),
            latent_fanout_schedule: Some(burn_dragon_core::LatentFanoutScheduleConfig::LateLayer {
                base_latent_total: 8192,
                last_layers: 4,
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.latent_total_for_layer(0), 8192);
        assert_eq!(config.latent_total_for_layer(7), 32768);
    }

    #[test]
    fn model_override_applies_sequence_kernel() {
        let overrides = ModelOverrides {
            sequence_kernel: Some(
                burn_dragon_core::SequenceKernelKind::Rwkv8StateSpaceExperimental,
            ),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(
            config.sequence_kernel,
            burn_dragon_core::SequenceKernelKind::Rwkv8StateSpaceExperimental
        );
    }

    #[test]
    fn model_override_without_explicit_connector_defaults_to_vanilla() {
        let overrides = ModelOverrides {
            mhc: Some(burn_dragon_core::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.residual_connector, ResidualConnectorKind::Vanilla);
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(!config.block_attention_residual.enabled);
    }

    #[test]
    fn model_override_explicit_vanilla_disables_other_connectors() {
        let overrides = ModelOverrides {
            residual_connector: Some(ResidualConnectorKind::Vanilla),
            mhc: Some(burn_dragon_core::ManifoldHyperConnectionsConfig {
                enabled: true,
                num_streams: 2,
                num_views: 1,
                mhc_iters: 4,
                mhc_tau: 0.1,
                add_branch_out_to_residual: true,
                dropout: 0.0,
                ..Default::default()
            }),
            attention_residual: Some(burn_dragon_core::AttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                ..Default::default()
            }),
            block_attention_residual: Some(burn_dragon_core::BlockAttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                layers_per_block: 2,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(config.residual_connector, ResidualConnectorKind::Vanilla);
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(!config.block_attention_residual.enabled);
    }

    #[test]
    fn model_override_explicit_block_attention_residual_enables_block_connector() {
        let overrides = ModelOverrides {
            residual_connector: Some(ResidualConnectorKind::BlockAttentionResidual),
            block_attention_residual: Some(burn_dragon_core::BlockAttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                layers_per_block: 2,
                block_history_window: Some(3),
                intra_block_history_window: Some(1),
                ..Default::default()
            }),
            attention_residual: Some(burn_dragon_core::AttentionResidualConfig {
                enabled: true,
                num_heads: 4,
                ..Default::default()
            }),
            ..ModelOverrides::default()
        };

        let config = build_model_config(&overrides, 32);
        assert_eq!(
            config.residual_connector,
            ResidualConnectorKind::BlockAttentionResidual
        );
        assert!(!config.mhc.enabled);
        assert!(!config.attention_residual.enabled);
        assert!(config.block_attention_residual.enabled);
        assert_eq!(config.block_attention_residual.layers_per_block, 2);
    }
}
