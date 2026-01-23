use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use toml::Value;

use burn_dragon_train::{
    GdpoConfig, GdpoHardGate, LearningRateScheduleConfig, OptimizerConfig, WgpuRuntimeConfig,
};

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuDatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(default)]
    pub augment: bool,
    #[serde(default = "default_augment_prob")]
    pub augment_prob: f32,
    #[serde(flatten)]
    pub source: SudokuDatasetSourceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum SudokuDatasetSourceConfig {
    HuggingFace(SudokuHuggingFaceConfig),
    Local(SudokuLocalConfig),
}

impl Default for SudokuDatasetSourceConfig {
    fn default() -> Self {
        Self::HuggingFace(SudokuHuggingFaceConfig::default())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuHuggingFaceConfig {
    pub repo_id: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub format: SudokuRecordFormat,
    #[serde(default = "default_hf_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_puzzle_field")]
    pub puzzle_field: String,
    #[serde(default = "default_solution_field")]
    pub solution_field: String,
    #[serde(default)]
    pub max_records: Option<usize>,
}

impl Default for SudokuHuggingFaceConfig {
    fn default() -> Self {
        Self {
            repo_id: "Ritvik19/Sudoku-Dataset".to_string(),
            token: None,
            revision: None,
            format: SudokuRecordFormat::Parquet,
            train_files: default_hf_train_files(),
            validation_files: vec!["valid_0.parquet".to_string()],
            puzzle_field: default_puzzle_field(),
            solution_field: default_solution_field(),
            max_records: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuLocalConfig {
    pub root: PathBuf,
    #[serde(default)]
    pub format: SudokuRecordFormat,
    #[serde(default = "default_local_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_puzzle_field")]
    pub puzzle_field: String,
    #[serde(default = "default_solution_field")]
    pub solution_field: String,
    #[serde(default)]
    pub max_records: Option<usize>,
}

impl Default for SudokuLocalConfig {
    fn default() -> Self {
        Self {
            root: PathBuf::from("data/sudoku"),
            format: SudokuRecordFormat::Jsonl,
            train_files: default_local_train_files(),
            validation_files: Vec::new(),
            puzzle_field: default_puzzle_field(),
            solution_field: default_solution_field(),
            max_records: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuRecordFormat {
    #[default]
    Jsonl,
    Csv,
    Parquet,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SudokuReconLoss {
    Softmax,
    Stablemax,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum SudokuLossMask {
    All,
    Unknown,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuTrainingHyperparameters {
    pub batch_size: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    pub rollout_steps: usize,
    #[serde(default)]
    pub rollout_min_steps: usize,
    #[serde(default)]
    pub rollout_max_steps: usize,
    #[serde(default)]
    pub rollout_max_steps_warmup_iters: usize,
    #[serde(default)]
    pub rollout_max_steps_warmup_cap: usize,
    #[serde(default)]
    pub rollout_backprop_steps: Option<usize>,
    #[serde(default = "default_halt_weight")]
    pub halt_weight: f32,
    #[serde(default = "default_halt_exploration_prob")]
    pub halt_exploration_prob: f32,
    #[serde(default = "default_halt_min_steps")]
    pub halt_min_steps: usize,
    #[serde(default)]
    pub policy_noise: f32,
    #[serde(default = "default_teacher_forcing_prob")]
    pub teacher_forcing_prob: f32,
    #[serde(default = "default_teacher_forcing_final")]
    pub teacher_forcing_final: f32,
    #[serde(default)]
    pub teacher_forcing_anneal_steps: usize,
    #[serde(default = "default_policy_temperature")]
    pub policy_temperature: f32,
    #[serde(default = "default_policy_temperature_final")]
    pub policy_temperature_final: f32,
    #[serde(default)]
    pub policy_temperature_anneal_steps: usize,
    #[serde(default = "default_policy_entropy_weight")]
    pub policy_entropy_weight: f32,
    #[serde(default = "default_policy_entropy_weight_final")]
    pub policy_entropy_weight_final: f32,
    #[serde(default)]
    pub policy_entropy_anneal_steps: usize,
    #[serde(default = "default_policy_recon_weight")]
    pub policy_recon_weight: f32,
    #[serde(default = "default_revisit_min_filled_frac")]
    pub revisit_min_filled_frac: f32,
    #[serde(default = "default_revisit_min_filled_final")]
    pub revisit_min_filled_final: f32,
    #[serde(default)]
    pub revisit_min_filled_anneal_steps: usize,
    #[serde(default = "default_reward_unknown_power")]
    pub reward_unknown_power: f32,
    #[serde(default = "default_saccade_step_cells")]
    pub saccade_step_cells: usize,
    #[serde(default = "default_recon_loss")]
    pub recon_loss: SudokuReconLoss,
    #[serde(default = "default_loss_mask")]
    pub loss_mask: SudokuLossMask,
    #[serde(default = "default_recon_loss_interval_steps")]
    pub recon_loss_interval_steps: usize,
    #[serde(default = "default_global_loss_samples")]
    pub global_loss_samples: usize,
    #[serde(default = "default_global_loss_weight")]
    pub global_loss_weight: f32,
    #[serde(default)]
    pub gdpo: GdpoConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuModelConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    #[serde(default = "default_summary_tokens")]
    pub summary_tokens: usize,
    #[serde(default = "default_policy_heads")]
    pub policy_heads: usize,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default)]
    pub fused_kernels: bool,
    #[serde(default)]
    pub relu_threshold: f32,
}

impl Default for SudokuModelConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_embd: 256,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            summary_tokens: default_summary_tokens(),
            policy_heads: default_policy_heads(),
            dropout: default_dropout(),
            fused_kernels: false,
            relu_threshold: 0.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuArtifactConfig {
    #[serde(default)]
    pub output: burn_dragon_train::VisionArtifactOutputMode,
    #[serde(default = "default_artifact_fps")]
    pub fps: u32,
    #[serde(default = "default_artifact_samples")]
    pub max_samples: usize,
    #[serde(default)]
    pub overwrite: bool,
}

impl Default for SudokuArtifactConfig {
    fn default() -> Self {
        Self {
            output: burn_dragon_train::VisionArtifactOutputMode::Mp4,
            fps: default_artifact_fps(),
            max_samples: default_artifact_samples(),
            overwrite: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuTrainingConfig {
    pub dataset: SudokuDatasetConfig,
    pub training: SudokuTrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub artifacts: SudokuArtifactConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    #[serde(default)]
    pub model: SudokuModelConfig,
}

impl SudokuTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.epochs.is_none() && self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0 when epochs is not set"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.rollout_steps == 0 {
            return Err(anyhow!("training.rollout_steps must be > 0"));
        }
        let max_rollout_steps = if self.training.rollout_max_steps > 0 {
            self.training.rollout_max_steps
        } else {
            self.training.rollout_steps
        };
        let min_rollout_steps = if self.training.rollout_min_steps > 0 {
            self.training.rollout_min_steps
        } else {
            max_rollout_steps
        };
        if min_rollout_steps == 0 || max_rollout_steps == 0 {
            return Err(anyhow!(
                "training.rollout_min_steps/rollout_max_steps must be > 0"
            ));
        }
        if min_rollout_steps > max_rollout_steps {
            return Err(anyhow!(
                "training.rollout_min_steps ({}) must be <= rollout_max_steps ({})",
                min_rollout_steps,
                max_rollout_steps
            ));
        }
        if self.training.rollout_max_steps_warmup_iters > 0 {
            if self.training.rollout_max_steps_warmup_cap == 0 {
                return Err(anyhow!(
                    "training.rollout_max_steps_warmup_cap must be > 0 when warmup is enabled"
                ));
            }
            if self.training.rollout_max_steps_warmup_cap > max_rollout_steps {
                return Err(anyhow!(
                    "training.rollout_max_steps_warmup_cap ({}) must be <= rollout_max_steps ({})",
                    self.training.rollout_max_steps_warmup_cap,
                    max_rollout_steps
                ));
            }
        }
        if let Some(backprop_steps) = self.training.rollout_backprop_steps
            && backprop_steps > 0
            && backprop_steps > max_rollout_steps
        {
            return Err(anyhow!(
                "training.rollout_backprop_steps ({}) must be <= rollout_max_steps ({})",
                backprop_steps, max_rollout_steps
            ));
        }
        if self.training.halt_weight < 0.0 {
            return Err(anyhow!("training.halt_weight must be >= 0"));
        }
        if !(0.0..=1.0).contains(&self.training.halt_exploration_prob) {
            return Err(anyhow!(
                "training.halt_exploration_prob must be in [0, 1] (got {})",
                self.training.halt_exploration_prob
            ));
        }
        if self.training.halt_min_steps == 0 {
            return Err(anyhow!("training.halt_min_steps must be > 0"));
        }
        if self.training.halt_min_steps > max_rollout_steps {
            return Err(anyhow!(
                "training.halt_min_steps ({}) must be <= rollout_max_steps ({})",
                self.training.halt_min_steps, max_rollout_steps
            ));
        }
        if self.training.saccade_step_cells == 0 {
            return Err(anyhow!("training.saccade_step_cells must be > 0"));
        }
        if !(0.0..=1.0).contains(&self.training.teacher_forcing_prob) {
            return Err(anyhow!(
                "training.teacher_forcing_prob must be in [0, 1] (got {})",
                self.training.teacher_forcing_prob
            ));
        }
        if !(0.0..=1.0).contains(&self.training.teacher_forcing_final) {
            return Err(anyhow!(
                "training.teacher_forcing_final must be in [0, 1] (got {})",
                self.training.teacher_forcing_final
            ));
        }
        if self.training.policy_temperature <= 0.0 {
            return Err(anyhow!(
                "training.policy_temperature must be > 0 (got {})",
                self.training.policy_temperature
            ));
        }
        if self.training.policy_temperature_final <= 0.0 {
            return Err(anyhow!(
                "training.policy_temperature_final must be > 0 (got {})",
                self.training.policy_temperature_final
            ));
        }
        if !self.training.policy_entropy_weight.is_finite()
            || self.training.policy_entropy_weight < 0.0
        {
            return Err(anyhow!(
                "training.policy_entropy_weight must be >= 0 (got {})",
                self.training.policy_entropy_weight
            ));
        }
        if !self.training.policy_entropy_weight_final.is_finite()
            || self.training.policy_entropy_weight_final < 0.0
        {
            return Err(anyhow!(
                "training.policy_entropy_weight_final must be >= 0 (got {})",
                self.training.policy_entropy_weight_final
            ));
        }
        if !self.training.policy_recon_weight.is_finite()
            || self.training.policy_recon_weight < 0.0
        {
            return Err(anyhow!(
                "training.policy_recon_weight must be >= 0 (got {})",
                self.training.policy_recon_weight
            ));
        }
        if !(0.0..=1.0).contains(&self.training.revisit_min_filled_frac) {
            return Err(anyhow!(
                "training.revisit_min_filled_frac must be in [0, 1] (got {})",
                self.training.revisit_min_filled_frac
            ));
        }
        if !(0.0..=1.0).contains(&self.training.revisit_min_filled_final) {
            return Err(anyhow!(
                "training.revisit_min_filled_final must be in [0, 1] (got {})",
                self.training.revisit_min_filled_final
            ));
        }
        if self.training.reward_unknown_power < 0.0 {
            return Err(anyhow!(
                "training.reward_unknown_power must be >= 0 (got {})",
                self.training.reward_unknown_power
            ));
        }
        if self.training.global_loss_weight < 0.0 {
            return Err(anyhow!(
                "training.global_loss_weight must be >= 0 (got {})",
                self.training.global_loss_weight
            ));
        }
        if let Some(epochs) = self.training.epochs && epochs == 0 {
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
        if !self.model.n_embd.is_multiple_of(self.model.policy_heads) {
            return Err(anyhow!(
                "model.policy_heads ({}) must divide model.n_embd ({})",
                self.model.policy_heads,
                self.model.n_embd
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

pub fn load_training_config(paths: &[PathBuf]) -> Result<SudokuTrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<SudokuTrainingConfig>()
        .map_err(|err| anyhow!(err))
}

fn load_value(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration file {}", path.display()))?;
    let table: toml::value::Table = toml::from_str(&content)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
    Ok(Value::Table(table))
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
}

fn default_train_split_ratio() -> f32 {
    0.9
}

fn default_hf_train_files() -> Vec<String> {
    vec!["train_0.parquet".to_string()]
}

fn default_local_train_files() -> Vec<String> {
    vec!["train.jsonl".to_string()]
}

fn default_puzzle_field() -> String {
    "puzzle".to_string()
}

fn default_solution_field() -> String {
    "solution".to_string()
}

fn default_artifact_fps() -> u32 {
    8
}

fn default_artifact_samples() -> usize {
    8
}

fn default_augment_prob() -> f32 {
    0.0
}

fn default_dropout() -> f64 {
    0.1
}

fn default_halt_weight() -> f32 {
    0.1
}

fn default_halt_exploration_prob() -> f32 {
    0.0
}

fn default_halt_min_steps() -> usize {
    1
}

fn default_recon_loss() -> SudokuReconLoss {
    SudokuReconLoss::Softmax
}

fn default_loss_mask() -> SudokuLossMask {
    SudokuLossMask::All
}

fn default_recon_loss_interval_steps() -> usize {
    1
}

fn default_teacher_forcing_prob() -> f32 {
    0.0
}

fn default_teacher_forcing_final() -> f32 {
    0.0
}

fn default_policy_temperature() -> f32 {
    1.0
}

fn default_policy_temperature_final() -> f32 {
    1.0
}

fn default_policy_entropy_weight() -> f32 {
    0.0
}

fn default_policy_entropy_weight_final() -> f32 {
    0.0
}

fn default_policy_recon_weight() -> f32 {
    0.05
}

fn default_revisit_min_filled_frac() -> f32 {
    0.0
}

fn default_revisit_min_filled_final() -> f32 {
    0.0
}

fn default_reward_unknown_power() -> f32 {
    0.0
}

fn default_saccade_step_cells() -> usize {
    1
}

fn default_summary_tokens() -> usize {
    1
}

fn default_policy_heads() -> usize {
    1
}

fn default_global_loss_samples() -> usize {
    16
}

fn default_global_loss_weight() -> f32 {
    0.2
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_config(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        let trimmed_lines: Vec<&str> = contents.lines().map(|line| line.trim_start()).collect();
        let mut formatted = trimmed_lines.join("\n");
        if formatted.starts_with('\n') {
            formatted = formatted.trim_start_matches('\n').to_string();
        }
        fs::write(&path, formatted).expect("write config");
        path
    }

    #[test]
    fn load_merges_in_order() {
        let dir = tempdir().expect("tempdir");

        let base_contents = [
            "[dataset]",
            "cache_dir = \"data\"",
            "train_split_ratio = 0.8",
            "type = \"hugging_face\"",
            "repo_id = \"Ritvik19/Sudoku-Dataset\"",
            "train_files = [\"train_0.parquet\"]",
            "puzzle_field = \"puzzle\"",
            "solution_field = \"solution\"",
            "",
            "[training]",
            "batch_size = 8",
            "max_iters = 1000",
            "log_frequency = 50",
            "rollout_steps = 4",
            "policy_noise = 0.5",
            "",
            "[optimizer]",
            "learning_rate = 0.001",
            "weight_decay = 0.05",
        ]
        .join("\n");
        let base = write_config(dir.path(), "base.toml", &base_contents);

        let override_contents = [
            "[training]",
            "max_iters = 2000",
            "",
            "[optimizer]",
            "learning_rate = 0.0005",
        ]
        .join("\n");
        let override_cfg = write_config(dir.path(), "override.toml", &override_contents);

        let config = load_training_config(&[base, override_cfg]).expect("load config");

        assert_eq!(config.training.batch_size, 8);
        assert_eq!(config.training.max_iters, 2000);
        assert_eq!(config.training.rollout_steps, 4);
        assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
    }
}
