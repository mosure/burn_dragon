use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
    #[serde(default)]
    pub lr_schedule: Option<LearningRateScheduleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_value: Option<f32>,
}

impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
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
