use anyhow::{Result, anyhow};

use burn_dragon_core::BDHConfig;
use burn_dragon_train::{GdpoHardGate, LearningRateScheduleConfig};

use super::{DatasetSourceConfig, TrainingConfig};

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.block_size == 0 {
            return Err(anyhow!("training.block_size must be > 0"));
        }
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        self.optimizer.validate()?;
        if !(0.0 < self.dataset.train_split_ratio && self.dataset.train_split_ratio <= 1.0) {
            return Err(anyhow!(
                "dataset.train_split_ratio must be in (0, 1] (got {})",
                self.dataset.train_split_ratio
            ));
        }
        if let Some(max_tokens) = self.generation.max_tokens
            && max_tokens <= 0
        {
            return Err(anyhow!("generation.max_tokens must be > 0"));
        }
        if self.generation.temperature <= 0.0 {
            return Err(anyhow!("generation.temperature must be > 0"));
        }
        if let Some(top_k) = self.generation.top_k
            && top_k == 0
        {
            return Err(anyhow!("generation.top_k must be > 0"));
        }

        match &self.dataset.source {
            DatasetSourceConfig::HuggingFace(config) => {
                if config.repo_id.trim().is_empty() {
                    return Err(anyhow!("dataset.repo_id must not be empty"));
                }
                if config.train_files.is_empty() {
                    return Err(anyhow!("dataset.train_files must not be empty"));
                }
                if config.text_fields.is_empty() {
                    return Err(anyhow!("dataset.text_fields must not be empty"));
                }
            }
            DatasetSourceConfig::DeepMath { max_records, .. }
            | DatasetSourceConfig::TinyChat { max_records, .. }
            | DatasetSourceConfig::WebscaleRl { max_records, .. }
            | DatasetSourceConfig::PoetryFoundation { max_records, .. } => {
                if matches!(max_records, Some(0)) {
                    return Err(anyhow!("dataset.max_records must be > 0 when set"));
                }
            }
            DatasetSourceConfig::Shakespeare { .. } => {}
        }

        if let Some(gdpo) = &self.training.gdpo
            && gdpo.enabled
        {
            if gdpo.group_size == 0 {
                return Err(anyhow!("training.gdpo.group_size must be > 0"));
            }
            if gdpo.hard_weight < 0.0 {
                return Err(anyhow!("training.gdpo.hard_weight must be >= 0"));
            }
            if gdpo.easy_weight < 0.0 {
                return Err(anyhow!("training.gdpo.easy_weight must be >= 0"));
            }
            if gdpo.policy_weight < 0.0 {
                return Err(anyhow!("training.gdpo.policy_weight must be >= 0"));
            }
            if gdpo.policy_clip_range < 0.0 {
                return Err(anyhow!("training.gdpo.policy_clip_range must be >= 0"));
            }
            if gdpo.advantage_clip < 0.0 {
                return Err(anyhow!(
                    "training.gdpo.advantage_clip must be >= 0 (got {})",
                    gdpo.advantage_clip
                ));
            }
            if !(0.0..1.0).contains(&gdpo.advantage_ema_decay) {
                return Err(anyhow!(
                    "training.gdpo.advantage_ema_decay must be in [0, 1) (got {})",
                    gdpo.advantage_ema_decay
                ));
            }
            if let GdpoHardGate::Percentile { quantile } = gdpo.hard_gate
                && !(0.0..=1.0).contains(&quantile)
            {
                return Err(anyhow!(
                    "training.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                    quantile
                ));
            }
        }

        if let Some(n_layer) = self.model.n_layer
            && n_layer == 0
        {
            return Err(anyhow!("model.n_layer must be > 0 when set"));
        }
        if let Some(n_embd) = self.model.n_embd
            && n_embd == 0
        {
            return Err(anyhow!("model.n_embd must be > 0 when set"));
        }
        if let Some(n_head) = self.model.n_head
            && n_head == 0
        {
            return Err(anyhow!("model.n_head must be > 0 when set"));
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier
            && multiplier == 0
        {
            return Err(anyhow!(
                "model.mlp_internal_dim_multiplier must be > 0 when set"
            ));
        }
        if let Some(dropout) = self.model.dropout
            && dropout < 0.0
        {
            return Err(anyhow!("model.dropout must be >= 0"));
        }
        if let Some(block_size) = self.model.block_size
            && block_size == 0
        {
            return Err(anyhow!("model.block_size must be > 0 when set"));
        }
        if let Some(rollout_fast_steps) = self.model.rollout_fast_steps_per_slow_step
            && !BDHConfig::is_valid_rollout_fast_steps(rollout_fast_steps)
        {
            return Err(anyhow!(
                "model.rollout_fast_steps_per_slow_step must be one of {:?} when set (got {})",
                BDHConfig::SUPPORTED_ROLLOUT_FAST_STEPS,
                rollout_fast_steps
            ));
        }

        if let Some(schedule) = &self.optimizer.lr_schedule {
            match schedule {
                LearningRateScheduleConfig::Constant { initial_lr }
                | LearningRateScheduleConfig::Cosine { initial_lr, .. }
                | LearningRateScheduleConfig::Linear { initial_lr, .. }
                | LearningRateScheduleConfig::Exponential { initial_lr, .. }
                | LearningRateScheduleConfig::Step { initial_lr, .. }
                | LearningRateScheduleConfig::Noam { initial_lr, .. } => {
                    if matches!(initial_lr.as_ref(), Some(value) if *value <= 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.initial_lr must be > 0"));
                    }
                }
            }

            match schedule {
                LearningRateScheduleConfig::Cosine {
                    min_lr, num_iters, ..
                } => {
                    if matches!(min_lr.as_ref(), Some(value) if *value < 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.min_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Linear {
                    final_lr,
                    num_iters,
                    ..
                } => {
                    if *final_lr < 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.final_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Exponential { gamma, .. } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Step {
                    gamma, step_size, ..
                } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                    if matches!(step_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.step_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Noam {
                    warmup_steps,
                    model_size,
                    ..
                } => {
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(model_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.model_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Constant { .. } => {}
            }
        }

        Ok(())
    }
}
