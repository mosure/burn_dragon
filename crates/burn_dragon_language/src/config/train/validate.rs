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
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
        }
        if matches!(self.training.target_effective_batch_size, Some(0)) {
            return Err(anyhow!(
                "training.target_effective_batch_size must be > 0 when set"
            ));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.wgpu.training.startup_autotune.enabled {
            let autotune = &self.wgpu.training.startup_autotune;
            if autotune.target_device_memory_mb == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.target_device_memory_mb must be > 0 when enabled"
                ));
            }
            if autotune.min_batch_size == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.min_batch_size must be > 0 when enabled"
                ));
            }
            if matches!(autotune.max_batch_size, Some(0)) {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be > 0 when set"
                ));
            }
            if autotune.probe_steps == 0 {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.probe_steps must be > 0 when enabled"
                ));
            }
            if let Some(max_batch_size) = autotune.max_batch_size
                && max_batch_size < autotune.min_batch_size
            {
                return Err(anyhow!(
                    "wgpu.training.startup_autotune.max_batch_size must be >= min_batch_size"
                ));
            }
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
        if let Some(y_neuron_recurrence) = &self.model.y_neuron_recurrence
            && y_neuron_recurrence.enabled
        {
            if y_neuron_recurrence.carry_in_scale < 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.carry_in_scale must be >= 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.last_layers must be > 0 when set"
                ));
            }
            if y_neuron_recurrence.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.chunk_tokens must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&y_neuron_recurrence.state_decay) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_decay must be in [0, 1] when enabled"
                ));
            }
            if y_neuron_recurrence.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_update_scale must be > 0 when enabled"
                ));
            }
            if matches!(y_neuron_recurrence.state_rms_cap, Some(value) if value <= 0.0) {
                return Err(anyhow!(
                    "model.y_neuron_recurrence.state_rms_cap must be > 0 when set"
                ));
            }
        }
        if let Some(clocked_slow_memory) = &self.model.clocked_slow_memory
            && clocked_slow_memory.enabled
        {
            if matches!(clocked_slow_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.clocked_slow_memory.last_layers must be > 0 when set"
                ));
            }
            if clocked_slow_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if clocked_slow_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.clocked_slow_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.clocked_slow_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(summary_memory) = &self.model.summary_memory
            && summary_memory.enabled
        {
            if matches!(summary_memory.last_layers, Some(0)) {
                return Err(anyhow!(
                    "model.summary_memory.last_layers must be > 0 when set"
                ));
            }
            if summary_memory.chunk_tokens == 0 {
                return Err(anyhow!(
                    "model.summary_memory.chunk_tokens must be > 0 when enabled"
                ));
            }
            if summary_memory.residual_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.residual_scale must be > 0 when enabled"
                ));
            }
            if !(0.0..=1.0).contains(&summary_memory.state_decay) {
                return Err(anyhow!(
                    "model.summary_memory.state_decay must be in [0, 1] when enabled"
                ));
            }
            if summary_memory.state_update_scale <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.state_update_scale must be > 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_threshold < 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_threshold must be >= 0 when enabled"
                ));
            }
            if summary_memory.surprise_gate_sharpness <= 0.0 {
                return Err(anyhow!(
                    "model.summary_memory.surprise_gate_sharpness must be > 0 when enabled"
                ));
            }
            if matches!(
                summary_memory.write_trigger_text.as_ref(),
                Some(value) if value.trim().is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_text must not be empty when set"
                ));
            }
            if matches!(
                summary_memory.write_trigger_token_ids.as_ref(),
                Some(value) if value.is_empty()
            ) {
                return Err(anyhow!(
                    "model.summary_memory.write_trigger_token_ids must not be empty when set"
                ));
            }
            if matches!(self.model.y_neuron_recurrence.as_ref(), Some(value) if value.enabled) {
                return Err(anyhow!(
                    "model.summary_memory is not yet supported together with model.y_neuron_recurrence"
                ));
            }
        }
        if let Some(mhc) = &self.model.mhc
            && mhc.enabled
        {
            if mhc.num_streams == 0 {
                return Err(anyhow!("model.mhc.num_streams must be > 0 when enabled"));
            }
            if mhc.num_views == 0 {
                return Err(anyhow!("model.mhc.num_views must be > 0 when enabled"));
            }
            if matches!(mhc.last_layers, Some(0)) {
                return Err(anyhow!("model.mhc.last_layers must be > 0 when set"));
            }
            if mhc.mhc_tau <= 0.0 {
                return Err(anyhow!("model.mhc.mhc_tau must be > 0 when enabled"));
            }
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
