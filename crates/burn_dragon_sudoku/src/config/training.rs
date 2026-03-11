use super::*;

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
pub enum SudokuWriteGateMode {
    #[default]
    StraightThrough,
    Bernoulli,
}

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
    ConstraintCa,
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
    #[serde(default = "default_trm_ca_decay")]
    pub trm_ca_decay: f32,
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
            trm_ca_decay: default_trm_ca_decay(),
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
    pub write_gate_mode: SudokuWriteGateMode,
    #[serde(default)]
    pub write_gate_warmup_steps: usize,
    #[serde(default)]
    pub write_gate_warmup_floor: f32,
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
            write_gate_mode: SudokuWriteGateMode::default(),
            write_gate_warmup_steps: 0,
            write_gate_warmup_floor: 0.0,
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
    #[serde(default = "default_reward_shaping_unknown_weight")]
    pub unknown_weight: f32,
    #[serde(default = "default_reward_shaping_accuracy_weight")]
    pub accuracy_weight: f32,
    #[serde(default = "default_reward_shaping_incorrect_penalty")]
    pub incorrect_penalty: f32,
}

impl Default for SudokuRewardShapingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            metric: SudokuRewardShapingMetric::default(),
            weight: default_reward_shaping_weight(),
            gamma: default_reward_shaping_gamma(),
            unknown_weight: default_reward_shaping_unknown_weight(),
            accuracy_weight: default_reward_shaping_accuracy_weight(),
            incorrect_penalty: default_reward_shaping_incorrect_penalty(),
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
    #[serde(default = "default_recon_loss_weight")]
    pub loss_weight: f32,
    #[serde(default = "default_recon_loss_weight_final")]
    pub loss_weight_final: f32,
    #[serde(default = "default_recon_loss_weight_anneal_steps")]
    pub loss_weight_anneal_steps: usize,
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
            loss_weight: default_recon_loss_weight(),
            loss_weight_final: default_recon_loss_weight_final(),
            loss_weight_anneal_steps: default_recon_loss_weight_anneal_steps(),
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
