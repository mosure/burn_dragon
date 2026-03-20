use anyhow::{Result, anyhow};

use burn_dragon_train::{GdpoHardGate, LearningRateScheduleConfig};

use super::*;

impl SudokuTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.epochs.is_none() && self.training.max_iters == 0 {
            return Err(anyhow!(
                "training.max_iters must be > 0 when epochs is not set"
            ));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.rollout.steps == 0 {
            return Err(anyhow!("training.rollout.steps must be > 0"));
        }
        let max_rollout_steps = if self.training.rollout.max_steps > 0 {
            self.training.rollout.max_steps
        } else {
            self.training.rollout.steps
        };
        let min_rollout_steps = if self.training.rollout.min_steps > 0 {
            self.training.rollout.min_steps
        } else {
            max_rollout_steps
        };
        if min_rollout_steps == 0 || max_rollout_steps == 0 {
            return Err(anyhow!(
                "training.rollout.min_steps/rollout_max_steps must be > 0"
            ));
        }
        if min_rollout_steps > max_rollout_steps {
            return Err(anyhow!(
                "training.rollout.min_steps ({}) must be <= rollout_max_steps ({})",
                min_rollout_steps,
                max_rollout_steps
            ));
        }

        if self.training.rollout.pre_steps_max < self.training.rollout.pre_steps_min {
            return Err(anyhow!(
                "training.rollout.pre_steps_max ({}) must be >= pre_steps_min ({})",
                self.training.rollout.pre_steps_max,
                self.training.rollout.pre_steps_min
            ));
        }
        if self.training.rollout.max_steps_warmup_iters > 0 {
            if self.training.rollout.max_steps_warmup_cap == 0 {
                return Err(anyhow!(
                    "training.rollout.max_steps_warmup_cap must be > 0 when warmup is enabled"
                ));
            }
            if self.training.rollout.max_steps_warmup_cap > max_rollout_steps {
                return Err(anyhow!(
                    "training.rollout.max_steps_warmup_cap ({}) must be <= rollout_max_steps ({})",
                    self.training.rollout.max_steps_warmup_cap,
                    max_rollout_steps
                ));
            }
        }
        if let Some(backprop_steps) = self.training.rollout.backprop_steps
            && backprop_steps > 0
            && backprop_steps > max_rollout_steps
        {
            return Err(anyhow!(
                "training.rollout.backprop_steps ({}) must be <= rollout_max_steps ({})",
                backprop_steps,
                max_rollout_steps
            ));
        }
        if self.training.halt.weight < 0.0 {
            return Err(anyhow!("training.halt.weight must be >= 0"));
        }
        if !(0.0..=1.0).contains(&self.training.halt.exploration_prob) {
            return Err(anyhow!(
                "training.halt.exploration_prob must be in [0, 1] (got {})",
                self.training.halt.exploration_prob
            ));
        }
        if self.training.halt.min_steps == 0 {
            return Err(anyhow!("training.halt.min_steps must be > 0"));
        }
        if self.training.halt.min_steps > max_rollout_steps {
            return Err(anyhow!(
                "training.halt.min_steps ({}) must be <= rollout_max_steps ({})",
                self.training.halt.min_steps,
                max_rollout_steps
            ));
        }
        if self.training.rollout.saccade_step_cells == 0 {
            return Err(anyhow!("training.rollout.saccade_step_cells must be > 0"));
        }
        if matches!(self.training.rollout.traversal, SudokuTraversal::L2rT2b)
            && matches!(self.training.rollout.trm_mode, SudokuTrmMode::Chunk)
        {
            let chunk = self.training.rollout.trm_chunk_size;
            let max_chunk = default_trm_chunk_size();
            if chunk == 0 {
                return Err(anyhow!("training.rollout.trm_chunk_size must be > 0"));
            }
            if chunk > max_chunk {
                return Err(anyhow!(
                    "training.rollout.trm_chunk_size ({}) must be <= {}",
                    chunk,
                    max_chunk
                ));
            }
        }
        if matches!(self.training.rollout.traversal, SudokuTraversal::L2rT2b)
            && matches!(self.training.rollout.trm_mode, SudokuTrmMode::ConstraintCa)
        {
            if !(0.0..=1.0).contains(&self.training.rollout.trm_ca_decay) {
                return Err(anyhow!(
                    "training.rollout.trm_ca_decay must be in [0, 1] (got {})",
                    self.training.rollout.trm_ca_decay
                ));
            }
            if self.model.n_head == 0 {
                return Err(anyhow!(
                    "model.n_head must be > 0 for trm_mode=constraint_ca"
                ));
            }
            if self.model.n_embd % self.model.n_head != 0 {
                return Err(anyhow!(
                    "model.n_embd ({}) must be divisible by model.n_head ({}) for trm_mode=constraint_ca",
                    self.model.n_embd,
                    self.model.n_head
                ));
            }
        }
        if let Some(schedule) = &self.training.rollout.schedule
            && (schedule.start_steps == 0 || schedule.final_steps == 0)
        {
            return Err(anyhow!(
                "training.rollout.schedule.start_steps/final_steps must be > 0"
            ));
        }

        if !(0.0..=1.0).contains(&self.training.policy.teacher_forcing_prob) {
            return Err(anyhow!(
                "training.policy.teacher_forcing_prob must be in [0, 1] (got {})",
                self.training.policy.teacher_forcing_prob
            ));
        }
        if !(0.0..=1.0).contains(&self.training.policy.teacher_forcing_final) {
            return Err(anyhow!(
                "training.policy.teacher_forcing_final must be in [0, 1] (got {})",
                self.training.policy.teacher_forcing_final
            ));
        }
        if !(0.0..=1.0).contains(&self.training.policy.epsilon) {
            return Err(anyhow!(
                "training.policy.epsilon must be in [0, 1] (got {})",
                self.training.policy.epsilon
            ));
        }
        if !(0.0..=1.0).contains(&self.training.policy.epsilon_final) {
            return Err(anyhow!(
                "training.policy.epsilon_final must be in [0, 1] (got {})",
                self.training.policy.epsilon_final
            ));
        }
        if self.training.policy.temperature <= 0.0 {
            return Err(anyhow!(
                "training.policy.temperature must be > 0 (got {})",
                self.training.policy.temperature
            ));
        }
        if self.training.policy.temperature_final <= 0.0 {
            return Err(anyhow!(
                "training.policy.temperature_final must be > 0 (got {})",
                self.training.policy.temperature_final
            ));
        }
        if !self.training.policy.entropy_weight.is_finite()
            || self.training.policy.entropy_weight < 0.0
        {
            return Err(anyhow!(
                "training.policy.entropy_weight must be >= 0 (got {})",
                self.training.policy.entropy_weight
            ));
        }
        if !self.training.policy.entropy_weight_final.is_finite()
            || self.training.policy.entropy_weight_final < 0.0
        {
            return Err(anyhow!(
                "training.policy.entropy_weight_final must be >= 0 (got {})",
                self.training.policy.entropy_weight_final
            ));
        }
        if !self.training.policy.entropy_target_scale.is_finite()
            || self.training.policy.entropy_target_scale < 0.0
        {
            return Err(anyhow!(
                "training.policy.entropy_target_scale must be >= 0 (got {})",
                self.training.policy.entropy_target_scale
            ));
        }
        if !(0.0..1.0).contains(&self.training.policy.entropy_target_ema_decay) {
            return Err(anyhow!(
                "training.policy.entropy_target_ema_decay must be in [0, 1) (got {})",
                self.training.policy.entropy_target_ema_decay
            ));
        }
        if !self.training.policy.entropy_alpha.is_finite()
            || self.training.policy.entropy_alpha < 0.0
        {
            return Err(anyhow!(
                "training.policy.entropy_alpha must be >= 0 (got {})",
                self.training.policy.entropy_alpha
            ));
        }
        if !self.training.policy.entropy_alpha_lr.is_finite()
            || self.training.policy.entropy_alpha_lr < 0.0
        {
            return Err(anyhow!(
                "training.policy.entropy_alpha_lr must be >= 0 (got {})",
                self.training.policy.entropy_alpha_lr
            ));
        }
        if !self.training.policy.visit_penalty.is_finite()
            || self.training.policy.visit_penalty < 0.0
        {
            return Err(anyhow!(
                "training.policy.visit_penalty must be >= 0 (got {})",
                self.training.policy.visit_penalty
            ));
        }
        if !self.training.policy.revisit_penalty.is_finite()
            || self.training.policy.revisit_penalty < 0.0
        {
            return Err(anyhow!(
                "training.policy.revisit_penalty must be >= 0 (got {})",
                self.training.policy.revisit_penalty
            ));
        }
        if !self.training.policy.recon_weight.is_finite() || self.training.policy.recon_weight < 0.0
        {
            return Err(anyhow!(
                "training.policy.recon_weight must be >= 0 (got {})",
                self.training.policy.recon_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.training.revisit.min_filled_frac) {
            return Err(anyhow!(
                "training.revisit.min_filled_frac must be in [0, 1] (got {})",
                self.training.revisit.min_filled_frac
            ));
        }
        if !(0.0..=1.0).contains(&self.training.revisit.min_filled_final) {
            return Err(anyhow!(
                "training.revisit.min_filled_final must be in [0, 1] (got {})",
                self.training.revisit.min_filled_final
            ));
        }
        if self.training.reward.unknown_power < 0.0 {
            return Err(anyhow!(
                "training.reward.unknown_power must be >= 0 (got {})",
                self.training.reward.unknown_power
            ));
        }
        if self.training.reward.no_op_penalty < 0.0 {
            return Err(anyhow!(
                "training.reward.no_op_penalty must be >= 0 (got {})",
                self.training.reward.no_op_penalty
            ));
        }
        if self.training.reward.shaping.enabled {
            if self.training.reward.shaping.weight < 0.0 {
                return Err(anyhow!(
                    "training.reward.shaping.weight must be >= 0 (got {})",
                    self.training.reward.shaping.weight
                ));
            }
            if !(0.0..=1.0).contains(&self.training.reward.shaping.gamma) {
                return Err(anyhow!(
                    "training.reward.shaping.gamma must be in [0, 1] (got {})",
                    self.training.reward.shaping.gamma
                ));
            }
            if self.training.reward.shaping.unknown_weight < 0.0 {
                return Err(anyhow!(
                    "training.reward.shaping.unknown_weight must be >= 0 (got {})",
                    self.training.reward.shaping.unknown_weight
                ));
            }
            if self.training.reward.shaping.accuracy_weight < 0.0 {
                return Err(anyhow!(
                    "training.reward.shaping.accuracy_weight must be >= 0 (got {})",
                    self.training.reward.shaping.accuracy_weight
                ));
            }
            if self.training.reward.shaping.incorrect_penalty < 0.0 {
                return Err(anyhow!(
                    "training.reward.shaping.incorrect_penalty must be >= 0 (got {})",
                    self.training.reward.shaping.incorrect_penalty
                ));
            }
        }
        if self.training.reward.baseline.enabled {
            if !(0.0..=1.0).contains(&self.training.reward.baseline.gamma) {
                return Err(anyhow!(
                    "training.reward.baseline.gamma must be in [0, 1] (got {})",
                    self.training.reward.baseline.gamma
                ));
            }
            if !(0.0..=1.0).contains(&self.training.reward.baseline.lambda) {
                return Err(anyhow!(
                    "training.reward.baseline.lambda must be in [0, 1] (got {})",
                    self.training.reward.baseline.lambda
                ));
            }
            if self.training.reward.baseline.value_loss_weight < 0.0 {
                return Err(anyhow!(
                    "training.reward.baseline.value_loss_weight must be >= 0 (got {})",
                    self.training.reward.baseline.value_loss_weight
                ));
            }
        }
        if self.training.reward.info_reward.enabled && self.training.reward.info_reward.stride == 0
        {
            return Err(anyhow!(
                "training.reward.info_reward.stride must be > 0 (got 0)"
            ));
        }
        if self.training.gdpo.enabled {
            if !self.training.policy.noise.is_finite() || self.training.policy.noise <= 0.0 {
                return Err(anyhow!(
                    "training.policy.noise must be > 0 when GDPO is enabled (got {})",
                    self.training.policy.noise
                ));
            }
            if !matches!(
                self.training.reward.easy_mode,
                SudokuEasyRewardMode::Recon | SudokuEasyRewardMode::AccuracyDelta
            ) {
                return Err(anyhow!(
                    "training.reward.easy_mode must be recon or accuracy_delta when GDPO is enabled"
                ));
            }
            if self.training.reward.hard_mode != SudokuHardRewardMode::InfoReward
                && self.training.reward.hard_mode != SudokuHardRewardMode::Accuracy
            {
                return Err(anyhow!(
                    "training.reward.hard_mode must be info_reward or accuracy when GDPO is enabled"
                ));
            }
        }
        if self.model.cache_mhc.enabled {
            if self.model.cache_mhc.num_streams == 0 {
                return Err(anyhow!("model.cache_mhc.num_streams must be > 0"));
            }
            if self.model.cache_mhc.num_views == 0 {
                return Err(anyhow!("model.cache_mhc.num_views must be > 0"));
            }
            if self.model.cache_mhc.mhc_iters == 0 {
                return Err(anyhow!("model.cache_mhc.mhc_iters must be > 0"));
            }
            if !self.model.cache_mhc.mhc_tau.is_finite() || self.model.cache_mhc.mhc_tau <= 0.0 {
                return Err(anyhow!(
                    "model.cache_mhc.mhc_tau must be > 0 (got {})",
                    self.model.cache_mhc.mhc_tau
                ));
            }
            if self.model.cache_mhc.dropout < 0.0 {
                return Err(anyhow!(
                    "model.cache_mhc.dropout must be >= 0 (got {})",
                    self.model.cache_mhc.dropout
                ));
            }
        }
        if self.training.recon.global_loss_weight < 0.0 {
            return Err(anyhow!(
                "training.recon.global_loss_weight must be >= 0 (got {})",
                self.training.recon.global_loss_weight
            ));
        }
        if self.training.recon.loss_weight < 0.0 {
            return Err(anyhow!(
                "training.recon.loss_weight must be >= 0 (got {})",
                self.training.recon.loss_weight
            ));
        }
        if self.training.recon.loss_weight_final < 0.0 {
            return Err(anyhow!(
                "training.recon.loss_weight_final must be >= 0 (got {})",
                self.training.recon.loss_weight_final
            ));
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        if !(0.0 < self.dataset.train_split_ratio && self.dataset.train_split_ratio <= 1.0) {
            return Err(anyhow!(
                "dataset.train_split_ratio must be in (0, 1] (got {})",
                self.dataset.train_split_ratio
            ));
        }
        if !(0.0..=1.0).contains(&self.dataset.augment_prob) {
            return Err(anyhow!(
                "dataset.augment_prob must be in [0, 1] (got {})",
                self.dataset.augment_prob
            ));
        }
        if self.training.gdpo.group_size == 0 {
            return Err(anyhow!("training.gdpo.group_size must be > 0"));
        }
        if self.training.gdpo.hard_weight < 0.0 {
            return Err(anyhow!("training.gdpo.hard_weight must be >= 0"));
        }
        if self.training.gdpo.easy_weight < 0.0 {
            return Err(anyhow!("training.gdpo.easy_weight must be >= 0"));
        }
        if self.training.gdpo.policy_weight < 0.0 {
            return Err(anyhow!("training.gdpo.policy_weight must be >= 0"));
        }
        if self.training.gdpo.policy_clip_range < 0.0 {
            return Err(anyhow!("training.gdpo.policy_clip_range must be >= 0"));
        }
        if self.training.gdpo.advantage_clip < 0.0 {
            return Err(anyhow!(
                "training.gdpo.advantage_clip must be >= 0 (got {})",
                self.training.gdpo.advantage_clip
            ));
        }
        if !(0.0..1.0).contains(&self.training.gdpo.advantage_ema_decay) {
            return Err(anyhow!(
                "training.gdpo.advantage_ema_decay must be in [0, 1) (got {})",
                self.training.gdpo.advantage_ema_decay
            ));
        }
        if let GdpoHardGate::Percentile { quantile } = self.training.gdpo.hard_gate
            && !(0.0..=1.0).contains(&quantile)
        {
            return Err(anyhow!(
                "training.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                quantile
            ));
        }

        match &self.dataset.source {
            SudokuDatasetSourceConfig::HuggingFace(cfg) => {
                if cfg.repo_id.trim().is_empty() {
                    return Err(anyhow!("dataset.repo_id must not be empty"));
                }
                if cfg.train_files.is_empty() {
                    return Err(anyhow!("dataset.train_files must not be empty"));
                }
                if cfg.puzzle_field.trim().is_empty() {
                    return Err(anyhow!("dataset.puzzle_field must not be empty"));
                }
                if cfg.solution_field.trim().is_empty() {
                    return Err(anyhow!("dataset.solution_field must not be empty"));
                }
            }
            SudokuDatasetSourceConfig::Local(cfg) => {
                if cfg.train_files.is_empty() {
                    return Err(anyhow!("dataset.train_files must not be empty"));
                }
                if cfg.puzzle_field.trim().is_empty() {
                    return Err(anyhow!("dataset.puzzle_field must not be empty"));
                }
                if cfg.solution_field.trim().is_empty() {
                    return Err(anyhow!("dataset.solution_field must not be empty"));
                }
            }
        }

        if self.model.summary_tokens == 0 {
            return Err(anyhow!("model.summary_tokens must be > 0"));
        }
        if self.model.policy_heads == 0 {
            return Err(anyhow!("model.policy_heads must be > 0"));
        }
        if self.model.policy_mlp_hidden_mult == 0 {
            return Err(anyhow!("model.policy_mlp_hidden_mult must be > 0"));
        }
        if self.model.n_embd % self.model.policy_heads != 0 {
            return Err(anyhow!(
                "model.policy_heads ({}) must divide model.n_embd ({})",
                self.model.policy_heads,
                self.model.n_embd
            ));
        }

        if matches!(self.model.grid_positional, SudokuGridPositional::Rope2d) {
            if self.model.n_embd % 4 != 0 {
                return Err(anyhow!(
                    "model.n_embd ({}) must be divisible by 4 for model.grid_positional = rope_2d",
                    self.model.n_embd
                ));
            }
            if self.model.grid_rope_theta <= 0.0 {
                return Err(anyhow!(
                    "model.grid_rope_theta must be > 0 for model.grid_positional = rope_2d"
                ));
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
