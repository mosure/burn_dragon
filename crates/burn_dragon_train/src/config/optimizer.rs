use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerKind {
    #[default]
    Adamw,
    BitnetPublished,
    MuonHybridExp,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum OptimizerScheduleMode {
    #[default]
    BdhReference,
    BitnetB158Reference,
    Hybrid,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum MuonAdjustLrFn {
    #[default]
    MatchRmsAdamw,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct MuonHybridConfig {
    pub enabled: bool,
    pub momentum: f32,
    pub nesterov: bool,
    pub ns_steps: usize,
    pub adjust_lr_fn: MuonAdjustLrFn,
    pub split_decoder_heads: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_modules: Option<Vec<String>>,
}

impl Default for MuonHybridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
            adjust_lr_fn: MuonAdjustLrFn::default(),
            split_decoder_heads: true,
            target_modules: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OptimizerConfig {
    #[serde(default)]
    pub name: OptimizerKind,
    pub learning_rate: f64,
    pub weight_decay: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_decay_final: Option<f32>,
    #[serde(default)]
    pub lr_schedule: Option<LearningRateScheduleConfig>,
    #[serde(default)]
    pub schedule_mode: OptimizerScheduleMode,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_value: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub muon: Option<MuonHybridConfig>,
}

impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }
        if let Some(weight_decay_final) = self.weight_decay_final
            && weight_decay_final < 0.0
        {
            return Err(anyhow!("optimizer.weight_decay_final must be >= 0"));
        }
        if let Some(clip) = self.grad_clip_norm
            && clip <= 0.0
        {
            return Err(anyhow!("optimizer.grad_clip_norm must be > 0"));
        }
        if let Some(clip) = self.grad_clip_value
            && clip <= 0.0
        {
            return Err(anyhow!("optimizer.grad_clip_value must be > 0"));
        }
        if self.grad_clip_norm.is_some() && self.grad_clip_value.is_some() {
            return Err(anyhow!(
                "optimizer.grad_clip_norm and optimizer.grad_clip_value are mutually exclusive"
            ));
        }
        if matches!(self.name, OptimizerKind::MuonHybridExp) {
            let muon = self.muon.as_ref().ok_or_else(|| {
                anyhow!("optimizer.muon must be set when optimizer.name = \"muon_hybrid_exp\"")
            })?;
            if !muon.enabled {
                return Err(anyhow!(
                    "optimizer.muon.enabled must be true when optimizer.name = \"muon_hybrid_exp\""
                ));
            }
            if muon.momentum <= 0.0 || muon.momentum >= 1.0 {
                return Err(anyhow!("optimizer.muon.momentum must be in (0, 1)"));
            }
            if muon.ns_steps == 0 {
                return Err(anyhow!("optimizer.muon.ns_steps must be > 0"));
            }
            if let Some(target_modules) = muon.target_modules.as_ref()
                && target_modules.is_empty()
            {
                return Err(anyhow!(
                    "optimizer.muon.target_modules must be non-empty when set"
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LearningRateScheduleConfig {
    Constant {
        #[serde(default)]
        initial_lr: Option<f64>,
    },
    Cosine {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        min_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Linear {
        #[serde(default)]
        initial_lr: Option<f64>,
        final_lr: f64,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Exponential {
        #[serde(default)]
        initial_lr: Option<f64>,
        gamma: f64,
    },
    Step {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default = "default_step_gamma")]
        gamma: f64,
        #[serde(default)]
        step_size: Option<usize>,
    },
    Noam {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        model_size: Option<usize>,
    },
}

fn default_step_gamma() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_optimizer() -> OptimizerConfig {
        OptimizerConfig {
            name: OptimizerKind::default(),
            learning_rate: 1.0e-3,
            weight_decay: 0.0,
            weight_decay_final: None,
            lr_schedule: None,
            schedule_mode: OptimizerScheduleMode::default(),
            grad_clip_norm: None,
            grad_clip_value: None,
            muon: None,
        }
    }

    #[test]
    fn muon_hybrid_requires_enabled_muon_config() {
        let config = OptimizerConfig {
            name: OptimizerKind::MuonHybridExp,
            muon: Some(MuonHybridConfig {
                enabled: false,
                ..Default::default()
            }),
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("optimizer.muon.enabled"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn weight_decay_final_must_be_non_negative() {
        let config = OptimizerConfig {
            weight_decay_final: Some(-0.1),
            ..base_optimizer()
        };
        let err = config.validate().expect_err("expected validation failure");
        assert!(
            err.to_string().contains("weight_decay_final"),
            "unexpected error: {err}"
        );
    }
}
