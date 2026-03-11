use super::*;

pub(crate) fn default_train_split_ratio() -> f32 {
    0.9
}

pub(crate) fn default_hf_train_files() -> Vec<String> {
    vec!["train_0.parquet".to_string()]
}

pub(crate) fn default_local_train_files() -> Vec<String> {
    vec!["train.jsonl".to_string()]
}

pub(crate) fn default_puzzle_field() -> String {
    "puzzle".to_string()
}

pub(crate) fn default_solution_field() -> String {
    "solution".to_string()
}

pub(crate) fn default_artifact_fps() -> u32 {
    8
}

pub(crate) fn default_artifact_sample_policy() -> bool {
    false
}

pub(crate) fn default_artifact_samples() -> usize {
    8
}

pub(crate) fn default_augment_prob() -> f32 {
    0.0
}

pub(crate) fn default_dropout() -> f64 {
    0.1
}

pub(crate) fn default_halt_weight() -> f32 {
    0.1
}

pub(crate) fn default_halt_exploration_prob() -> f32 {
    0.0
}

pub(crate) fn default_halt_min_steps() -> usize {
    1
}

pub(crate) fn default_recon_loss() -> SudokuReconLoss {
    SudokuReconLoss::Softmax
}

pub(crate) fn default_loss_mask() -> SudokuLossMask {
    SudokuLossMask::All
}

pub(crate) fn default_recon_loss_interval_steps() -> usize {
    1
}

pub(crate) fn default_recon_loss_weight() -> f32 {
    1.0
}

pub(crate) fn default_recon_loss_weight_final() -> f32 {
    default_recon_loss_weight()
}

pub(crate) fn default_recon_loss_weight_anneal_steps() -> usize {
    0
}

pub(crate) fn default_validation_sample_policy() -> bool {
    false
}

pub(crate) fn default_teacher_forcing_prob() -> f32 {
    0.0
}

pub(crate) fn default_teacher_forcing_final() -> f32 {
    0.0
}

pub(crate) fn default_policy_epsilon() -> f32 {
    0.0
}

pub(crate) fn default_policy_epsilon_final() -> f32 {
    0.0
}

pub(crate) fn default_policy_temperature() -> f32 {
    1.0
}

pub(crate) fn default_policy_temperature_final() -> f32 {
    1.0
}

pub(crate) fn default_policy_entropy_weight() -> f32 {
    0.0
}

pub(crate) fn default_policy_entropy_weight_final() -> f32 {
    0.0
}

pub(crate) fn default_policy_entropy_adaptive() -> bool {
    false
}

pub(crate) fn default_policy_entropy_target_scale() -> f32 {
    1.0
}

pub(crate) fn default_policy_entropy_alpha() -> f32 {
    0.01
}

pub(crate) fn default_policy_entropy_alpha_lr() -> f32 {
    0.001
}

pub(crate) fn default_policy_visit_penalty() -> f32 {
    0.0
}

pub(crate) fn default_policy_recon_weight() -> f32 {
    0.05
}

pub(crate) fn default_revisit_min_filled_frac() -> f32 {
    0.0
}

pub(crate) fn default_revisit_min_filled_final() -> f32 {
    0.0
}

pub(crate) fn default_reward_unknown_power() -> f32 {
    0.0
}

pub(crate) fn default_reward_no_op_penalty() -> f32 {
    0.0
}

pub(crate) fn default_reward_shaping_weight() -> f32 {
    0.1
}

pub(crate) fn default_reward_shaping_gamma() -> f32 {
    1.0
}

pub(crate) fn default_reward_shaping_unknown_weight() -> f32 {
    0.0
}

pub(crate) fn default_reward_shaping_accuracy_weight() -> f32 {
    0.0
}

pub(crate) fn default_reward_shaping_incorrect_penalty() -> f32 {
    0.0
}

pub(crate) fn default_reward_baseline_gamma() -> f32 {
    0.99
}

pub(crate) fn default_reward_baseline_lambda() -> f32 {
    0.95
}

pub(crate) fn default_reward_baseline_value_loss_weight() -> f32 {
    0.5
}

pub(crate) fn default_info_reward_stride() -> usize {
    1
}

pub(crate) fn default_policy_entropy_target_ema_decay() -> f32 {
    0.99
}

pub(crate) fn default_saccade_step_cells() -> usize {
    1
}

pub(crate) fn default_trm_chunk_size() -> usize {
    81
}

pub(crate) fn default_trm_ca_decay() -> f32 {
    0.9
}

pub(crate) fn default_summary_tokens() -> usize {
    1
}

pub(crate) fn default_policy_heads() -> usize {
    1
}

pub(crate) fn default_policy_mlp_hidden_mult() -> usize {
    2
}

pub(crate) fn default_grid_rope_theta() -> f32 {
    65_536.0
}

pub(crate) fn default_global_loss_samples() -> usize {
    16
}

pub(crate) fn default_global_loss_weight() -> f32 {
    0.2
}

pub(crate) fn default_cache_mhc_num_streams() -> usize {
    1
}

pub(crate) fn default_cache_mhc_num_views() -> usize {
    1
}

pub(crate) fn default_cache_mhc_iters() -> usize {
    10
}

pub(crate) fn default_cache_mhc_tau() -> f32 {
    0.05
}

pub(crate) fn default_cache_mhc_add_branch_out_to_residual() -> bool {
    true
}

pub(crate) fn default_cache_mhc_dropout() -> f64 {
    0.0
}
