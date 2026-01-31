use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use toml::Value;

use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor,
};
use burn::tensor::backend::{AutodiffBackend, Backend};

use burn_dragon_core::{ManifoldHyperConnectionsConfig, RotaryEmbedding};

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

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuPolicyHead {
    #[default]
    Cache,
    SummaryPos,
    SummaryMlp,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuGridPositional {
    #[default]
    Additive,
    #[serde(rename = "rope_2d")]
    Rope2d,
}

impl SudokuGridPositional {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Rope2d => "rope_2d",
        }
    }
}

impl std::fmt::Display for SudokuGridPositional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<B: Backend> Module<B> for SudokuGridPositional {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SudokuGridPositional {
    type InnerModule = SudokuGridPositional;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for SudokuGridPositional {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!("grid_positional={self}");
        content
            .set_top_level_type("SudokuGridPositional")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for SudokuGridPositional {}



#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuTraversal {
    #[default]
    Saccade,
    L2rT2b,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuTrmMode {
    Recurrent,
    #[default]
    Chunk,
}
#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuRolloutConfig {
    pub steps: usize,
    #[serde(default)]
    pub min_steps: usize,
    #[serde(default)]
    pub max_steps: usize,
    #[serde(default)]
    pub max_steps_warmup_iters: usize,
    #[serde(default)]
    pub max_steps_warmup_cap: usize,
    #[serde(default)]
    pub backprop_steps: Option<usize>,
    #[serde(default)]
    pub traversal: SudokuTraversal,
    #[serde(default)]
    pub trm_mode: SudokuTrmMode,
    #[serde(default = "default_trm_chunk_size")]
    pub trm_chunk_size: usize,
    #[serde(default)]
    pub pre_steps_min: usize,
    #[serde(default)]
    pub pre_steps_max: usize,
    #[serde(default = "default_saccade_step_cells")]
    pub saccade_step_cells: usize,
    #[serde(default)]
    pub schedule: Option<SudokuRolloutSchedule>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct SudokuRolloutSchedule {
    pub start_steps: usize,
    pub final_steps: usize,
    #[serde(default)]
    pub anneal_iters: usize,
}


impl Default for SudokuRolloutConfig {
    fn default() -> Self {
        Self {
            steps: 0,
            min_steps: 0,
            max_steps: 0,
            max_steps_warmup_iters: 0,
            max_steps_warmup_cap: 0,
            backprop_steps: None,
            traversal: SudokuTraversal::default(),
            trm_mode: SudokuTrmMode::default(),
            trm_chunk_size: default_trm_chunk_size(),
            pre_steps_min: 0,
            pre_steps_max: 0,
            saccade_step_cells: default_saccade_step_cells(),
            schedule: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuHaltConfig {
    #[serde(default = "default_halt_weight")]
    pub weight: f32,
    #[serde(default = "default_halt_exploration_prob")]
    pub exploration_prob: f32,
    #[serde(default = "default_halt_min_steps")]
    pub min_steps: usize,
}

impl Default for SudokuHaltConfig {
    fn default() -> Self {
        Self {
            weight: default_halt_weight(),
            exploration_prob: default_halt_exploration_prob(),
            min_steps: default_halt_min_steps(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuPolicyConfig {
    #[serde(default)]
    pub noise: f32,
    #[serde(default = "default_policy_epsilon")]
    pub epsilon: f32,
    #[serde(default = "default_policy_epsilon_final")]
    pub epsilon_final: f32,
    #[serde(default)]
    pub epsilon_anneal_steps: usize,
    #[serde(default = "default_teacher_forcing_prob")]
    pub teacher_forcing_prob: f32,
    #[serde(default = "default_teacher_forcing_final")]
    pub teacher_forcing_final: f32,
    #[serde(default)]
    pub teacher_forcing_anneal_steps: usize,
    #[serde(default = "default_policy_temperature")]
    pub temperature: f32,
    #[serde(default = "default_policy_temperature_final")]
    pub temperature_final: f32,
    #[serde(default)]
    pub temperature_anneal_steps: usize,
    #[serde(default = "default_policy_entropy_weight")]
    pub entropy_weight: f32,
    #[serde(default = "default_policy_entropy_weight_final")]
    pub entropy_weight_final: f32,
    #[serde(default)]
    pub entropy_anneal_steps: usize,
    #[serde(default = "default_policy_entropy_adaptive")]
    pub entropy_adaptive: bool,
    #[serde(default = "default_policy_entropy_target_scale")]
    pub entropy_target_scale: f32,
    #[serde(default = "default_policy_entropy_target_ema_decay")]
    pub entropy_target_ema_decay: f32,
    #[serde(default = "default_policy_entropy_alpha")]
    pub entropy_alpha: f32,
    #[serde(default = "default_policy_entropy_alpha_lr")]
    pub entropy_alpha_lr: f32,
    #[serde(default = "default_policy_visit_penalty")]
    pub visit_penalty: f32,
    #[serde(default)]
    pub revisit_penalty: f32,
    #[serde(default = "default_policy_recon_weight")]
    pub recon_weight: f32,
    #[serde(default)]
    pub cache_update_clues: bool,
}

impl Default for SudokuPolicyConfig {
    fn default() -> Self {
        Self {
            noise: 0.0,
            epsilon: default_policy_epsilon(),
            epsilon_final: default_policy_epsilon_final(),
            epsilon_anneal_steps: 0,
            teacher_forcing_prob: default_teacher_forcing_prob(),
            teacher_forcing_final: default_teacher_forcing_final(),
            teacher_forcing_anneal_steps: 0,
            temperature: default_policy_temperature(),
            temperature_final: default_policy_temperature_final(),
            temperature_anneal_steps: 0,
            entropy_weight: default_policy_entropy_weight(),
            entropy_weight_final: default_policy_entropy_weight_final(),
            entropy_anneal_steps: 0,
            entropy_adaptive: default_policy_entropy_adaptive(),
            entropy_target_scale: default_policy_entropy_target_scale(),
            entropy_target_ema_decay: default_policy_entropy_target_ema_decay(),
            entropy_alpha: default_policy_entropy_alpha(),
            entropy_alpha_lr: default_policy_entropy_alpha_lr(),
            visit_penalty: default_policy_visit_penalty(),
            revisit_penalty: 0.0,
            recon_weight: default_policy_recon_weight(),
            cache_update_clues: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuRevisitConfig {
    #[serde(default = "default_revisit_min_filled_frac")]
    pub min_filled_frac: f32,
    #[serde(default = "default_revisit_min_filled_final")]
    pub min_filled_final: f32,
    #[serde(default)]
    pub min_filled_anneal_steps: usize,
}

impl Default for SudokuRevisitConfig {
    fn default() -> Self {
        Self {
            min_filled_frac: default_revisit_min_filled_frac(),
            min_filled_final: default_revisit_min_filled_final(),
            min_filled_anneal_steps: 0,
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuEasyRewardMode {
    #[default]
    Recon,
    AccuracyDelta,
    Gae,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuHardRewardMode {
    #[default]
    InfoReward,
    Accuracy,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuInfoRewardConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_info_reward_stride")]
    pub stride: usize,
}

impl Default for SudokuInfoRewardConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stride: default_info_reward_stride(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuRewardConfig {
    #[serde(default = "default_reward_unknown_power")]
    pub unknown_power: f32,
    #[serde(default = "default_reward_no_op_penalty")]
    pub no_op_penalty: f32,
    #[serde(default)]
    pub easy_mode: SudokuEasyRewardMode,
    #[serde(default)]
    pub hard_mode: SudokuHardRewardMode,
    #[serde(default)]
    pub info_reward: SudokuInfoRewardConfig,
    #[serde(default)]
    pub shaping: SudokuRewardShapingConfig,
    #[serde(default)]
    pub baseline: SudokuRewardBaselineConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuRewardShapingConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default)]
    pub metric: SudokuRewardShapingMetric,
    #[serde(default = "default_reward_shaping_weight")]
    pub weight: f32,
    #[serde(default = "default_reward_shaping_gamma")]
    pub gamma: f32,
}

impl Default for SudokuRewardShapingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metric: SudokuRewardShapingMetric::default(),
            weight: default_reward_shaping_weight(),
            gamma: default_reward_shaping_gamma(),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuRewardShapingMetric {
    #[default]
    Conflict,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuRewardBaselineConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_reward_baseline_gamma")]
    pub gamma: f32,
    #[serde(default = "default_reward_baseline_lambda")]
    pub lambda: f32,
    #[serde(default = "default_reward_baseline_value_loss_weight")]
    pub value_loss_weight: f32,
}

impl Default for SudokuRewardBaselineConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            gamma: default_reward_baseline_gamma(),
            lambda: default_reward_baseline_lambda(),
            value_loss_weight: default_reward_baseline_value_loss_weight(),
        }
    }
}

impl Default for SudokuRewardConfig {
    fn default() -> Self {
        Self {
            unknown_power: default_reward_unknown_power(),
            no_op_penalty: default_reward_no_op_penalty(),
            easy_mode: SudokuEasyRewardMode::default(),
            hard_mode: SudokuHardRewardMode::default(),
            info_reward: SudokuInfoRewardConfig::default(),
            shaping: SudokuRewardShapingConfig::default(),
            baseline: SudokuRewardBaselineConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuReconConfig {
    #[serde(default = "default_recon_loss")]
    pub loss: SudokuReconLoss,
    #[serde(default = "default_loss_mask")]
    pub loss_mask: SudokuLossMask,
    #[serde(default = "default_recon_loss_interval_steps")]
    pub loss_interval_steps: usize,
    #[serde(default = "default_global_loss_samples")]
    pub global_loss_samples: usize,
    #[serde(default = "default_global_loss_weight")]
    pub global_loss_weight: f32,
}

impl Default for SudokuReconConfig {
    fn default() -> Self {
        Self {
            loss: default_recon_loss(),
            loss_mask: default_loss_mask(),
            loss_interval_steps: default_recon_loss_interval_steps(),
            global_loss_samples: default_global_loss_samples(),
            global_loss_weight: default_global_loss_weight(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuValidationConfig {
    #[serde(default)]
    pub rollout_steps: Option<usize>,
    #[serde(default = "default_validation_sample_policy")]
    pub sample_policy: bool,
}

impl Default for SudokuValidationConfig {
    fn default() -> Self {
        Self {
            rollout_steps: None,
            sample_policy: default_validation_sample_policy(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
struct SudokuTrainingLegacy {
    pub rollout_steps: Option<usize>,
    pub rollout_min_steps: Option<usize>,
    pub rollout_max_steps: Option<usize>,
    pub rollout_max_steps_warmup_iters: Option<usize>,
    pub rollout_max_steps_warmup_cap: Option<usize>,
    pub rollout_backprop_steps: Option<Option<usize>>,
    pub halt_weight: Option<f32>,
    pub halt_exploration_prob: Option<f32>,
    pub halt_min_steps: Option<usize>,
    pub policy_noise: Option<f32>,
    pub policy_epsilon: Option<f32>,
    pub policy_epsilon_final: Option<f32>,
    pub policy_epsilon_anneal_steps: Option<usize>,
    pub teacher_forcing_prob: Option<f32>,
    pub teacher_forcing_final: Option<f32>,
    pub teacher_forcing_anneal_steps: Option<usize>,
    pub policy_temperature: Option<f32>,
    pub policy_temperature_final: Option<f32>,
    pub policy_temperature_anneal_steps: Option<usize>,
    pub policy_entropy_weight: Option<f32>,
    pub policy_entropy_weight_final: Option<f32>,
    pub policy_entropy_anneal_steps: Option<usize>,
    pub policy_entropy_adaptive: Option<bool>,
    pub policy_entropy_target_scale: Option<f32>,
    pub policy_entropy_alpha: Option<f32>,
    pub policy_entropy_alpha_lr: Option<f32>,
    pub policy_visit_penalty: Option<f32>,
    pub policy_revisit_penalty: Option<f32>,
    pub policy_recon_weight: Option<f32>,
    pub revisit_min_filled_frac: Option<f32>,
    pub revisit_min_filled_final: Option<f32>,
    pub revisit_min_filled_anneal_steps: Option<usize>,
    pub reward_unknown_power: Option<f32>,
    pub saccade_step_cells: Option<usize>,
    pub recon_loss: Option<SudokuReconLoss>,
    pub loss_mask: Option<SudokuLossMask>,
    pub recon_loss_interval_steps: Option<usize>,
    pub global_loss_samples: Option<usize>,
    pub global_loss_weight: Option<f32>,
    pub gdpo: Option<GdpoConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(default)]
struct SudokuTrainingHyperparametersRaw {
    pub batch_size: usize,
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    pub rollout: Option<SudokuRolloutConfig>,
    pub halt: Option<SudokuHaltConfig>,
    pub policy: Option<SudokuPolicyConfig>,
    pub revisit: Option<SudokuRevisitConfig>,
    pub reward: Option<SudokuRewardConfig>,
    pub recon: Option<SudokuReconConfig>,
    pub validation: Option<SudokuValidationConfig>,
    pub gdpo: Option<GdpoConfig>,
    #[serde(flatten)]
    pub legacy: SudokuTrainingLegacy,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(from = "SudokuTrainingHyperparametersRaw")]
pub struct SudokuTrainingHyperparameters {
    pub batch_size: usize,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    pub rollout: SudokuRolloutConfig,
    pub halt: SudokuHaltConfig,
    pub policy: SudokuPolicyConfig,
    pub revisit: SudokuRevisitConfig,
    pub reward: SudokuRewardConfig,
    pub recon: SudokuReconConfig,
    pub validation: SudokuValidationConfig,
    pub gdpo: GdpoConfig,
}

impl From<SudokuTrainingHyperparametersRaw> for SudokuTrainingHyperparameters {
    fn from(raw: SudokuTrainingHyperparametersRaw) -> Self {
        let mut rollout = raw.rollout.unwrap_or_default();
        let mut halt = raw.halt.unwrap_or_default();
        let mut policy = raw.policy.unwrap_or_default();
        let mut revisit = raw.revisit.unwrap_or_default();
        let mut reward = raw.reward.unwrap_or_default();
        let mut recon = raw.recon.unwrap_or_default();
        let validation = raw.validation.unwrap_or_default();
        let mut gdpo = raw.gdpo.unwrap_or_default();

        if let Some(value) = raw.legacy.rollout_steps {
            rollout.steps = value;
        }
        if let Some(value) = raw.legacy.rollout_min_steps {
            rollout.min_steps = value;
        }
        if let Some(value) = raw.legacy.rollout_max_steps {
            rollout.max_steps = value;
        }
        if let Some(value) = raw.legacy.rollout_max_steps_warmup_iters {
            rollout.max_steps_warmup_iters = value;
        }
        if let Some(value) = raw.legacy.rollout_max_steps_warmup_cap {
            rollout.max_steps_warmup_cap = value;
        }
        if let Some(value) = raw.legacy.rollout_backprop_steps {
            rollout.backprop_steps = value;
        }
        if let Some(value) = raw.legacy.saccade_step_cells {
            rollout.saccade_step_cells = value;
        }

        if let Some(value) = raw.legacy.halt_weight {
            halt.weight = value;
        }
        if let Some(value) = raw.legacy.halt_exploration_prob {
            halt.exploration_prob = value;
        }
        if let Some(value) = raw.legacy.halt_min_steps {
            halt.min_steps = value;
        }

        if let Some(value) = raw.legacy.policy_noise {
            policy.noise = value;
        }
        if let Some(value) = raw.legacy.policy_epsilon {
            policy.epsilon = value;
        }
        if let Some(value) = raw.legacy.policy_epsilon_final {
            policy.epsilon_final = value;
        }
        if let Some(value) = raw.legacy.policy_epsilon_anneal_steps {
            policy.epsilon_anneal_steps = value;
        }
        if let Some(value) = raw.legacy.teacher_forcing_prob {
            policy.teacher_forcing_prob = value;
        }
        if let Some(value) = raw.legacy.teacher_forcing_final {
            policy.teacher_forcing_final = value;
        }
        if let Some(value) = raw.legacy.teacher_forcing_anneal_steps {
            policy.teacher_forcing_anneal_steps = value;
        }
        if let Some(value) = raw.legacy.policy_temperature {
            policy.temperature = value;
        }
        if let Some(value) = raw.legacy.policy_temperature_final {
            policy.temperature_final = value;
        }
        if let Some(value) = raw.legacy.policy_temperature_anneal_steps {
            policy.temperature_anneal_steps = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_weight {
            policy.entropy_weight = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_weight_final {
            policy.entropy_weight_final = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_anneal_steps {
            policy.entropy_anneal_steps = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_adaptive {
            policy.entropy_adaptive = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_target_scale {
            policy.entropy_target_scale = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_alpha {
            policy.entropy_alpha = value;
        }
        if let Some(value) = raw.legacy.policy_entropy_alpha_lr {
            policy.entropy_alpha_lr = value;
        }
        if let Some(value) = raw.legacy.policy_visit_penalty {
            policy.visit_penalty = value;
        }
        if let Some(value) = raw.legacy.policy_revisit_penalty {
            policy.revisit_penalty = value;
        }
        if let Some(value) = raw.legacy.policy_recon_weight {
            policy.recon_weight = value;
        }

        if let Some(value) = raw.legacy.revisit_min_filled_frac {
            revisit.min_filled_frac = value;
        }
        if let Some(value) = raw.legacy.revisit_min_filled_final {
            revisit.min_filled_final = value;
        }
        if let Some(value) = raw.legacy.revisit_min_filled_anneal_steps {
            revisit.min_filled_anneal_steps = value;
        }

        if let Some(value) = raw.legacy.reward_unknown_power {
            reward.unknown_power = value;
        }

        if let Some(value) = raw.legacy.recon_loss {
            recon.loss = value;
        }
        if let Some(value) = raw.legacy.loss_mask {
            recon.loss_mask = value;
        }
        if let Some(value) = raw.legacy.recon_loss_interval_steps {
            recon.loss_interval_steps = value;
        }
        if let Some(value) = raw.legacy.global_loss_samples {
            recon.global_loss_samples = value;
        }
        if let Some(value) = raw.legacy.global_loss_weight {
            recon.global_loss_weight = value;
        }
        if let Some(value) = raw.legacy.gdpo {
            gdpo = value;
        }

        Self {
            batch_size: raw.batch_size,
            epochs: raw.epochs,
            max_iters: raw.max_iters,
            log_frequency: raw.log_frequency,
            rollout,
            halt,
            policy,
            revisit,
            reward,
            recon,
            validation,
            gdpo,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuCacheMhcConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cache_mhc_num_streams")]
    pub num_streams: usize,
    #[serde(default = "default_cache_mhc_num_views")]
    pub num_views: usize,
    #[serde(default = "default_cache_mhc_iters")]
    pub mhc_iters: usize,
    #[serde(default = "default_cache_mhc_tau")]
    pub mhc_tau: f32,
    #[serde(default = "default_cache_mhc_add_branch_out_to_residual")]
    pub add_branch_out_to_residual: bool,
    #[serde(default = "default_cache_mhc_dropout")]
    pub dropout: f64,
}

impl Default for SudokuCacheMhcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_streams: default_cache_mhc_num_streams(),
            num_views: default_cache_mhc_num_views(),
            mhc_iters: default_cache_mhc_iters(),
            mhc_tau: default_cache_mhc_tau(),
            add_branch_out_to_residual: default_cache_mhc_add_branch_out_to_residual(),
            dropout: default_cache_mhc_dropout(),
        }
    }
}

impl SudokuCacheMhcConfig {
    pub fn to_core(&self) -> ManifoldHyperConnectionsConfig {
        ManifoldHyperConnectionsConfig {
            enabled: self.enabled,
            num_streams: self.num_streams,
            num_views: self.num_views,
            mhc_iters: self.mhc_iters,
            mhc_tau: self.mhc_tau,
            add_branch_out_to_residual: self.add_branch_out_to_residual,
            dropout: self.dropout,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuCacheUpdateMode {
    #[default]
    Overwrite,
    GatedResidual,
}


#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct SudokuCacheUpdateConfig {
    #[serde(default)]
    pub mode: SudokuCacheUpdateMode,
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
    #[serde(default)]
    pub policy_head: SudokuPolicyHead,
    #[serde(default = "default_policy_mlp_hidden_mult")]
    pub policy_mlp_hidden_mult: usize,
    #[serde(default)]
    pub rotary_embedding: RotaryEmbedding,
    #[serde(default)]
    pub grid_positional: SudokuGridPositional,
    #[serde(default = "default_grid_rope_theta")]
    pub grid_rope_theta: f32,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default)]
    pub fused_kernels: bool,
    #[serde(default)]
    pub relu_threshold: f32,
    #[serde(default)]
    pub cache_mhc: SudokuCacheMhcConfig,
    #[serde(default)]
    pub cache_update: SudokuCacheUpdateConfig,
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
            policy_head: SudokuPolicyHead::default(),
            policy_mlp_hidden_mult: default_policy_mlp_hidden_mult(),
            rotary_embedding: RotaryEmbedding::default(),
            grid_positional: SudokuGridPositional::default(),
            grid_rope_theta: default_grid_rope_theta(),
            dropout: default_dropout(),
            fused_kernels: false,
            relu_threshold: 0.0,
            cache_mhc: SudokuCacheMhcConfig::default(),
            cache_update: SudokuCacheUpdateConfig::default(),
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
    #[serde(default = "default_artifact_sample_policy")]
    pub sample_policy: bool,
    pub overwrite: bool,
}

impl Default for SudokuArtifactConfig {
    fn default() -> Self {
        Self {
            output: burn_dragon_train::VisionArtifactOutputMode::Mp4,
            fps: default_artifact_fps(),
            max_samples: default_artifact_samples(),
            sample_policy: default_artifact_sample_policy(),
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
                backprop_steps, max_rollout_steps
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
                self.training.halt.min_steps, max_rollout_steps
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
                    chunk, max_chunk
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
        if !self.training.policy.recon_weight.is_finite()
            || self.training.policy.recon_weight < 0.0
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
        if self.training.reward.info_reward.enabled && self.training.reward.info_reward.stride == 0 {
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
            if self.training.reward.shaping.enabled {
                return Err(anyhow!(
                    "training.reward.shaping must be disabled when GDPO is enabled"
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
        if self.model.policy_mlp_hidden_mult == 0 {
            return Err(anyhow!("model.policy_mlp_hidden_mult must be > 0"));
        }
        if !self.model.n_embd.is_multiple_of(self.model.policy_heads) {
            return Err(anyhow!(
                "model.policy_heads ({}) must divide model.n_embd ({})",
                self.model.policy_heads,
                self.model.n_embd
            ));
        }

        if matches!(self.model.grid_positional, SudokuGridPositional::Rope2d) {
            if !self.model.n_embd.is_multiple_of(4) {
                return Err(anyhow!(
                    "model.n_embd ({}) must be divisible by 4 for model.grid_positional = rope_2d",
                    self.model.n_embd
                ));
            }
            if self.model.grid_rope_theta <= 0.0 {
                return Err(anyhow!("model.grid_rope_theta must be > 0 for model.grid_positional = rope_2d"));
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

fn default_artifact_sample_policy() -> bool {
    false
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

fn default_validation_sample_policy() -> bool {
    false
}

fn default_teacher_forcing_prob() -> f32 {
    0.0
}

fn default_teacher_forcing_final() -> f32 {
    0.0
}

fn default_policy_epsilon() -> f32 {
    0.0
}

fn default_policy_epsilon_final() -> f32 {
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


fn default_policy_entropy_adaptive() -> bool {
    false
}

fn default_policy_entropy_target_scale() -> f32 {
    1.0
}

fn default_policy_entropy_alpha() -> f32 {
    0.01
}

fn default_policy_entropy_alpha_lr() -> f32 {
    0.001
}

fn default_policy_visit_penalty() -> f32 {
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

fn default_reward_no_op_penalty() -> f32 {
    0.0
}

fn default_reward_shaping_weight() -> f32 {
    0.1
}

fn default_reward_shaping_gamma() -> f32 {
    1.0
}

fn default_reward_baseline_gamma() -> f32 {
    0.99
}

fn default_reward_baseline_lambda() -> f32 {
    0.95
}

fn default_reward_baseline_value_loss_weight() -> f32 {
    0.5
}

fn default_info_reward_stride() -> usize {
    1
}

fn default_policy_entropy_target_ema_decay() -> f32 {
    0.99
}

fn default_saccade_step_cells() -> usize {
    1
}

fn default_trm_chunk_size() -> usize {
    81
}

fn default_summary_tokens() -> usize {
    1
}

fn default_policy_heads() -> usize {
    1
}

fn default_policy_mlp_hidden_mult() -> usize {
    2
}

fn default_grid_rope_theta() -> f32 {
    65_536.0
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
            "",
            "[training.rollout]",
            "steps = 4",
            "",
            "[training.policy]",
            "noise = 0.5",
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
        assert_eq!(config.training.rollout.steps, 4);
        assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
    }
}









fn default_cache_mhc_num_streams() -> usize {
    1
}

fn default_cache_mhc_num_views() -> usize {
    1
}

fn default_cache_mhc_iters() -> usize {
    10
}

fn default_cache_mhc_tau() -> f32 {
    0.05
}

fn default_cache_mhc_add_branch_out_to_residual() -> bool {
    true
}

fn default_cache_mhc_dropout() -> f64 {
    0.0
}























