use crate::train::prelude::*;

pub fn adamw_config_from_optimizer(optimizer_cfg: &OptimizerConfig) -> AdamWConfig {
    let mut config = AdamWConfig::new().with_weight_decay(optimizer_cfg.weight_decay);
    if let Some(clip) = optimizer_cfg.grad_clip_norm {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Norm(clip)));
    } else if let Some(clip) = optimizer_cfg.grad_clip_value {
        config = config.with_grad_clipping(Some(GradientClippingConfig::Value(clip)));
    }
    config
}
