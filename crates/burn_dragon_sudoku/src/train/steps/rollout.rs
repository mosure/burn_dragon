use super::*;

pub(super) struct RolloutBase<B: BackendTrait> {
    pub(super) recon_loss: Tensor<B, 1>,
    pub(super) acc: Tensor<B, 1>,
    pub(super) exact_acc: Tensor<B, 1>,
    pub(super) solve_rate: Tensor<B, 1>,
    pub(super) hard_reward: Tensor<B, 1>,
    pub(super) easy_reward: Tensor<B, 1>,
    pub(super) saccade_revisit_rate: Tensor<B, 1>,
    pub(super) saccade_repeat_rate: Tensor<B, 1>,
    pub(super) saccade_unknown_frac: Tensor<B, 1>,
    pub(super) saccade_unique_frac: Tensor<B, 1>,
    pub(super) write_gate_mean: Tensor<B, 1>,
    pub(super) write_rate: Tensor<B, 1>,
    pub(super) halt_loss: Tensor<B, 1>,
    pub(super) halt_prob_mean: Tensor<B, 1>,
    pub(super) halt_target_mean: Tensor<B, 1>,
    #[allow(dead_code)]
    pub(super) log_prob_sum: Option<Tensor<B, 2>>,
    #[allow(dead_code)]
    pub(super) policy_entropy_sum: Tensor<B, 2>,
    #[allow(dead_code)]
    pub(super) policy_steps: usize,
    #[allow(dead_code)]
    pub(super) selectable_count_sum: Tensor<B, 1>,
    #[allow(dead_code)]
    pub(super) selectable_steps: usize,
    #[allow(dead_code)]
    pub(super) batch: usize,
}

#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn rollout_base<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    policy_epsilon: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    _force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = rollout_max_steps(training, 0);
    rollout_base_impl(
        model,
        batch,
        training,
        policy_noise,
        policy_epsilon,
        repeat_for_gdpo,
        track_policy,
        train_mode,
        teacher_forcing_prob,
        policy_temperature,
        write_gate_floor_for_step(training, 0),
        revisit_min_filled,
        rollout_steps,
    )
}

#[cfg(test)]
#[allow(dead_code, clippy::too_many_arguments)]
pub(super) fn rollout_base_autodiff<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    policy_epsilon: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    _force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = rollout_max_steps(training, 0);
    rollout_base_impl(
        model,
        batch,
        training,
        policy_noise,
        policy_epsilon,
        repeat_for_gdpo,
        track_policy,
        train_mode,
        teacher_forcing_prob,
        policy_temperature,
        write_gate_floor_for_step(training, 0),
        revisit_min_filled,
        rollout_steps,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn rollout_base_impl<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    policy_epsilon: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    write_gate_floor: f32,
    revisit_min_filled: f32,
    rollout_steps: usize,
) -> RolloutBase<B> {
    let rollout_steps = rollout_steps.max(1);
    let recon_loss_interval = training.recon.loss_interval_steps;
    let global_weight = training.recon.global_loss_weight.max(0.0);
    let global_samples = training.recon.global_loss_samples.min(GRID_LEN);
    let visit_penalty = training.policy.visit_penalty.max(0.0);
    let revisit_penalty = training.policy.revisit_penalty.max(0.0);
    let write_gate_floor = write_gate_floor.clamp(0.0, 1.0);
    let mut puzzles = batch.puzzles;
    let mut solutions = batch.solutions;
    let device = puzzles.device();

    if let Some(group) = repeat_for_gdpo
        && group > 1
    {
        puzzles = puzzles.repeat_dim(0, group);
        solutions = solutions.repeat_dim(0, group);
    }
    let [batch_size, _] = puzzles.shape().dims::<2>();
    let ones_grid = Tensor::<B, 2>::ones([batch_size.max(1), GRID_LEN], &device);
    let action_index = build_action_index(batch_size, &device);
    let solution_one_hot = build_solution_one_hot(&solutions, batch_size, &device);
    let (row_ids, col_ids) = model.grid_row_col_ids(batch_size, &device);
    let row_ids_f = row_ids.clone().float();
    let col_ids_f = col_ids.clone().float();

    let clue_mask = puzzles.clone().greater_elem(0.0).float();
    let editable_mask = ones_grid.clone().sub(clue_mask.clone());
    let editable_counts = editable_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1), 1])
        .clamp_min(1.0);

    let mut tokens = puzzles;
    let mut unknown_mask = tokens.clone().equal_elem(0).float();
    let loss_unknown_mask = unknown_mask.clone();
    let mut tokens_reward = tokens.clone();
    let initial_unknown_counts = unknown_mask.clone().sum_dim(1).reshape([batch_size.max(1)]);
    let difficulty_scale = difficulty_scale_from_unknowns(
        initial_unknown_counts.clone(),
        training.reward.unknown_power,
    );
    let easy_mode = training.reward.easy_mode;
    let hard_mode = training.reward.hard_mode;
    let info_enabled = training.reward.info_reward.enabled
        && matches!(hard_mode, SudokuHardRewardMode::InfoReward);
    let info_stride = training.reward.info_reward.stride.max(1);
    let conflict_weight = training.reward.shaping.weight.max(0.0);
    let unknown_weight = training.reward.shaping.unknown_weight.max(0.0);
    let accuracy_weight = training.reward.shaping.accuracy_weight.max(0.0);
    let incorrect_penalty = training.reward.shaping.incorrect_penalty.max(0.0);
    let shaping_enabled = training.reward.shaping.enabled
        && (conflict_weight > 0.0
            || unknown_weight > 0.0
            || accuracy_weight > 0.0
            || incorrect_penalty > 0.0);
    let shaping_metric = training.reward.shaping.metric;
    let shaping_gamma = training.reward.shaping.gamma.clamp(0.0, 1.0);
    let baseline_enabled = training.reward.baseline.enabled;
    let baseline_gamma = training.reward.baseline.gamma.clamp(0.0, 1.0);
    let baseline_lambda = training.reward.baseline.lambda.clamp(0.0, 1.0);
    let _baseline_value_weight = training.reward.baseline.value_loss_weight.max(0.0);
    let no_op_penalty = training.reward.no_op_penalty.max(0.0);
    let mut rollout_shaping_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_shaping_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut conflict_prev = if shaping_enabled && conflict_weight > 0.0 {
        shaping_potential(&tokens, shaping_metric)
    } else {
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device)
    };
    let mut unknown_prev = if shaping_enabled && unknown_weight > 0.0 {
        unknown_potential(&tokens)
    } else {
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device)
    };
    let mut rollout_step_rewards: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_values: Vec<Tensor<B, 1>> = Vec::new();

    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

    let mut state = model.init_state();
    let input_cache =
        model.cell_embeddings_with_positions(tokens.clone(), row_ids.clone(), col_ids.clone());
    let [_, _, embd] = input_cache.shape().dims();
    let cache_streams = model.cache_streams();
    let input_cache = input_cache.unsqueeze_dim::<4>(1).expand([
        batch_size.max(1),
        cache_streams,
        GRID_LEN,
        embd,
    ]);
    let input_cache_read =
        input_cache
            .clone()
            .mean_dim(1)
            .reshape([batch_size.max(1), GRID_LEN, embd]);
    let mut cache = input_cache.clone();
    let mut summary_tokens = model.init_summary_tokens(batch_size);
    let summary_len = model.summary_token_count();
    let (_initial_acc_per_sample, initial_acc_mean, initial_exact, initial_solve_rate) =
        compute_grid_accuracy(&tokens_reward, &solutions);
    let (reward_initial_acc_per_sample, _reward_initial_acc_mean) =
        compute_grid_accuracy_masked(&tokens_reward, &solutions, &editable_mask);
    let reward_initial_acc = reward_initial_acc_per_sample.clone();
    let mut last_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut last_acc = initial_acc_mean;
    let mut last_exact = initial_exact;
    let mut last_solve_rate = initial_solve_rate;
    let mut last_reward_acc_per_sample = reward_initial_acc.clone();
    let mut prev_reward_acc_per_sample = reward_initial_acc.clone();

    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let mut recon_loss_sum = zeros.clone();
    let mut recon_steps = 0usize;
    let mut chunk_recon_delta_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_recon_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_delta_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut info_reward_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut halt_loss_sum = zeros.clone();
    let mut halt_prob_sum = zeros.clone();
    let mut halt_target_sum = zeros.clone();

    let mut policy_entropy_sum = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut log_prob_sum = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut policy_steps = 0usize;
    let mut action_count_sum = zeros.clone();
    let mut revisit_count_sum = zeros.clone();
    let mut repeat_count_sum = zeros.clone();
    let mut unknown_select_sum = zeros.clone();
    let mut write_gate_sum = zeros.clone();
    let mut write_rate_sum = zeros.clone();
    let mut visit_counts = Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut prev_action_one_hot = Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut selectable_count_sum = zeros.clone();
    let mut selectable_steps = 0usize;

    let pre_steps = if matches!(training.rollout.traversal, SudokuTraversal::L2rT2b) {
        sample_pre_rollout_steps(training)
    } else {
        0
    };

    if pre_steps > 0 {
        for pre_idx in 0..pre_steps {
            let cache_read = cache
                .clone()
                .mean_dim(1)
                .reshape([batch_size.max(1), GRID_LEN, embd]);
            let actions = static_traversal_actions(batch_size, pre_idx, &device);
            let action_one_hot = build_action_one_hot(&actions, &action_index);
            prev_action_one_hot = action_one_hot.clone();
            visit_counts = visit_counts + action_one_hot.clone();

            let step_input_base = input_cache_read
                .clone()
                .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
                .sum_dim(1)
                .reshape([batch_size.max(1), 1, embd]);
            let step_residual = cache_read
                .clone()
                .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
                .sum_dim(1)
                .reshape([batch_size.max(1), 1, embd]);
            let step_input = model.project_input_tokens(step_input_base) + step_residual;
            let step_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
            let (step_hidden, _step_logits_full) =
                model.forward_with_hidden_and_state_embedded(step_input, &mut state);
            let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
            let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
            summary_tokens = model.normalize_summary_tokens(summary_hidden);
            let step_logits = model.value_logits_from_hidden(step_hidden);

            let teacher_force = if teacher_forcing_prob > 0.0 {
                Tensor::<B, 2>::random(
                    [batch_size.max(1), 1],
                    TensorDistribution::Uniform(0.0, 1.0),
                    &device,
                )
                .lower_equal_elem(teacher_forcing_prob)
                .float()
            } else {
                Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device)
            };
            let teacher_mask = teacher_force.clone().greater_equal_elem(0.5);
            let pred_values = step_logits.argmax(2).reshape([batch_size.max(1), 1]);
            let mut update_values = pred_values.clone();
            let selected_solution = solutions
                .clone()
                .float()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .int();
            update_values = update_values.mask_where(teacher_mask, selected_solution.clone());

            let update_mask = action_one_hot
                .clone()
                .mul(editable_mask.clone())
                .greater_equal_elem(0.5);
            let update_values_grid = update_values.clone().repeat_dim(1, GRID_LEN);
            tokens = tokens.mask_where(update_mask.clone(), update_values_grid);
            unknown_mask = (unknown_mask - action_one_hot.clone()).clamp_min(0.0);
            let update_values_pred_grid = pred_values.clone().repeat_dim(1, GRID_LEN);
            tokens_reward = tokens_reward.mask_where(update_mask.clone(), update_values_pred_grid);

            let update_mask_cache = if training.policy.cache_update_clues {
                action_one_hot.clone()
            } else {
                action_one_hot.clone().mul(editable_mask.clone())
            };
            let update_mask_f = update_mask_cache.clone().unsqueeze_dim::<3>(2);
            let selected_row = row_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .int();
            let selected_col = col_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .int();
            let action_mask = action_one_hot
                .clone()
                .unsqueeze_dim::<3>(2)
                .unsqueeze_dim::<4>(1);
            let cache_cell = cache.clone().mul(action_mask.clone()).sum_dim(2).reshape([
                batch_size.max(1) * cache_streams,
                1,
                embd,
            ]);
            let summary_streams = summary_tokens
                .clone()
                .unsqueeze_dim::<4>(1)
                .expand([batch_size.max(1), cache_streams, summary_len, embd])
                .reshape([batch_size.max(1) * cache_streams, summary_len, embd]);
            let token_emb = model.cell_embeddings_with_positions(
                update_values.clone(),
                selected_row,
                selected_col,
            );
            let token_emb = token_emb
                .unsqueeze_dim::<4>(1)
                .expand([batch_size.max(1), cache_streams, 1, embd])
                .reshape([batch_size.max(1) * cache_streams, 1, embd]);
            let update_emb = model.update_cell_embedding(summary_streams, cache_cell, token_emb);
            let update_emb = update_emb
                .reshape([batch_size.max(1), cache_streams, 1, embd])
                .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
            let update_mask_stream = update_mask_f.clone().unsqueeze_dim::<4>(1);
            let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
            cache = cache * keep + update_emb.mul(update_mask_stream);
            cache = mhc_passthrough(model.cache_mhc.as_ref(), cache);
        }
    }

    for step_offset in 0..rollout_steps {
        let step_idx = step_offset + pre_steps;
        let cache_read = cache
            .clone()
            .mean_dim(1)
            .reshape([batch_size.max(1), GRID_LEN, embd]);
        let policy_logits =
            model.policy_logits_from_cache(summary_tokens.clone(), cache_read.clone());

        let step_value = if baseline_enabled {
            Some(
                model
                    .value_baseline_from_summary_tokens(summary_tokens.clone())
                    .reshape([batch_size.max(1)]),
            )
        } else {
            None
        };

        let tokens_solved_before = tokens
            .clone()
            .equal(solutions.clone())
            .float()
            .sum_dim(1)
            .reshape([batch_size.max(1)])
            .equal_elem(GRID_LEN as f32)
            .float()
            .reshape([batch_size.max(1), 1]);
        let use_select_mask =
            revisit_min_filled > 0.0 || visit_penalty > 0.0 || revisit_penalty > 0.0;
        let (select_mask, selectable_counts, masked_logits) = if use_select_mask {
            let unknown_counts = unknown_mask
                .clone()
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            let filled_frac = editable_counts
                .clone()
                .sub(unknown_counts.clone())
                .div(editable_counts.clone())
                .clamp_min(0.0)
                .clamp_max(1.0);
            let revisit_threshold = revisit_min_filled.clamp(0.0, 1.0);
            let allow_revisit = filled_frac.greater_equal_elem(revisit_threshold).float();
            let allow_revisit_grid = allow_revisit.clone().repeat_dim(1, GRID_LEN);
            let select_mask = clue_mask.clone().add(
                unknown_mask
                    .clone()
                    .mul(ones_grid.clone().sub(allow_revisit_grid.clone()))
                    .add(editable_mask.clone().mul(allow_revisit_grid)),
            );
            let mut masked_logits = policy_logits
                - ones_grid
                    .clone()
                    .sub(select_mask.clone())
                    .mul_scalar(POLICY_MASK_PENALTY);
            if visit_penalty > 0.0 {
                let visit_log = visit_counts.clone().add_scalar(1.0).log();
                masked_logits = masked_logits - visit_log.mul_scalar(visit_penalty);
            }
            if revisit_penalty > 0.0 {
                let revisits = visit_counts.clone().greater_elem(0.0).float();
                masked_logits = masked_logits - revisits.mul_scalar(revisit_penalty);
            }

            let selectable_counts = select_mask
                .clone()
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            (select_mask, selectable_counts, masked_logits)
        } else {
            (
                ones_grid.clone(),
                ones_step.clone().mul_scalar(GRID_LEN as f32),
                policy_logits.clone(),
            )
        };
        if track_policy {
            let step_selectable_mean = selectable_counts.clone().mean();
            selectable_count_sum = selectable_count_sum + step_selectable_mean.detach();
            selectable_steps += 1;
        }
        let active_mask = selectable_counts
            .greater_elem(0.0)
            .float()
            .mul(ones_step.clone().sub(halted.clone()))
            .mul(ones_step.clone().sub(tokens_solved_before.clone()));
        let policy_logits = if (policy_temperature - 1.0).abs() > f32::EPSILON {
            masked_logits.clone().div_scalar(policy_temperature)
        } else {
            masked_logits.clone()
        };
        let sampled_logits = if policy_noise > 0.0 {
            let noise = Tensor::<B, 2>::random(
                [batch_size, GRID_LEN],
                TensorDistribution::Normal(0.0, f64::from(policy_noise)),
                &device,
            );
            policy_logits.clone() + noise
        } else {
            policy_logits.clone()
        };
        let actions = match training.rollout.traversal {
            SudokuTraversal::Saccade => sample_actions(
                sampled_logits,
                select_mask.clone(),
                train_mode,
                policy_epsilon,
            ),
            SudokuTraversal::L2rT2b => static_traversal_actions(batch_size, step_idx, &device),
        };
        let mut action_one_hot = build_action_one_hot(&actions, &action_index);
        let active_mask_grid = active_mask.clone().repeat_dim(1, GRID_LEN);
        action_one_hot = action_one_hot * active_mask_grid.clone() * select_mask.clone();
        let first_visit_mask = visit_counts.clone().equal_elem(0.0).float();
        let selected_first_visit = action_one_hot
            .clone()
            .mul(first_visit_mask.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let action_any = action_one_hot
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let step_revisit = action_any
            .clone()
            .sub(selected_first_visit.clone())
            .clamp_min(0.0);
        let step_repeat = action_one_hot
            .clone()
            .mul(prev_action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let step_unknown = unknown_mask
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        action_count_sum = action_count_sum + action_any.clone().mean().detach();
        revisit_count_sum = revisit_count_sum + step_revisit.mean().detach();
        repeat_count_sum = repeat_count_sum + step_repeat.mean().detach();
        unknown_select_sum = unknown_select_sum + step_unknown.mean().detach();
        prev_action_one_hot = action_one_hot.clone();
        visit_counts = visit_counts + action_one_hot.clone();

        let mut selected_log_prob = None;
        let mut step_entropy = None;
        if track_policy {
            let log_probs = activation::log_softmax(policy_logits, 1);
            let selected = log_probs
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .mul(active_mask.clone());
            let entropy = log_probs
                .clone()
                .exp()
                .mul(log_probs)
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .mul_scalar(-1.0)
                .mul(active_mask.clone());
            selected_log_prob = Some(selected);
            step_entropy = Some(entropy);
        }
        let [_, _, embd] = cache_read.shape().dims();
        let step_input_base = input_cache_read
            .clone()
            .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch_size.max(1), 1, embd]);
        let step_residual = cache_read
            .clone()
            .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch_size.max(1), 1, embd]);
        let step_input = model.project_input_tokens(step_input_base) + step_residual;
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
        let (step_hidden, _step_logits_full) =
            model.forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = model.normalize_summary_tokens(summary_hidden);
        let step_logits = model.value_logits_from_hidden(step_hidden.clone());
        let halt_logit = model.halt_logit_from_summary_tokens(summary_tokens.clone());
        let halt_prob = activation::sigmoid(halt_logit.clone());

        let selected_solution_one_hot = solution_one_hot
            .clone()
            .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch_size.max(1), 1, VOCAB_SIZE]);
        let selected_solution = solutions
            .clone()
            .float()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .int();
        let selected_editable = editable_mask
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let mut reward_mask = selected_editable.clone().reshape([batch_size.max(1)]);
        let mut local_loss_mask = active_mask.clone();
        if matches!(training.recon.loss_mask, SudokuLossMask::Unknown) {
            let selected_unknown = loss_unknown_mask
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            local_loss_mask = local_loss_mask.mul(selected_unknown);
        }
        local_loss_mask = ensure_non_empty_mask(local_loss_mask, active_mask.clone());
        let (local_loss, _local_acc, _local_exact, local_loss_per_sample, ..) =
            compute_loss_and_acc(
                &step_logits,
                &selected_solution,
                &selected_solution_one_hot,
                &local_loss_mask,
                &training.recon.loss,
            );

        let reward_local_mask = active_mask.clone().mul(selected_editable.clone());
        let (
            _reward_local_loss,
            _reward_local_acc,
            _reward_local_exact,
            reward_local_loss_per_sample,
            ..,
        ) = compute_loss_and_acc(
            &step_logits,
            &selected_solution,
            &selected_solution_one_hot,
            &reward_local_mask,
            &training.recon.loss,
        );

        let teacher_force = if teacher_forcing_prob > 0.0 {
            Tensor::<B, 2>::random(
                [batch_size.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(teacher_forcing_prob)
            .float()
        } else {
            Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device)
        };
        let teacher_mask = teacher_force.clone().greater_equal_elem(0.5);
        let pred_values = step_logits.argmax(2).reshape([batch_size.max(1), 1]);
        let mut update_values = pred_values.clone();
        update_values = update_values.mask_where(teacher_mask, selected_solution.clone());

        let selected_row = row_ids_f
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .int();
        let selected_col = col_ids_f
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .int();
        let action_mask = action_one_hot
            .clone()
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(1);
        let cache_cell = cache.clone().mul(action_mask.clone()).sum_dim(2).reshape([
            batch_size.max(1) * cache_streams,
            1,
            embd,
        ]);
        let summary_streams = summary_tokens
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batch_size.max(1), cache_streams, summary_len, embd])
            .reshape([batch_size.max(1) * cache_streams, summary_len, embd]);
        let token_emb =
            model.cell_embeddings_with_positions(update_values.clone(), selected_row, selected_col);
        let token_emb = token_emb
            .unsqueeze_dim::<4>(1)
            .expand([batch_size.max(1), cache_streams, 1, embd])
            .reshape([batch_size.max(1) * cache_streams, 1, embd]);
        let (update_emb, write_gate) =
            model.update_cell_embedding_with_gate(summary_streams, cache_cell, token_emb);
        let write_gate = write_gate
            .reshape([batch_size.max(1), cache_streams.max(1), 1])
            .mean_dim(1)
            .reshape([batch_size.max(1), 1]);
        let write_gate = if write_gate_floor > 0.0 {
            write_gate
                .mul_scalar(1.0 - write_gate_floor)
                .add_scalar(write_gate_floor)
        } else {
            write_gate
        };
        let (write_mask_f, gate_log_prob, gate_entropy) = resolve_write_gate(
            write_gate.clone(),
            training.policy.write_gate_mode,
            action_any.clone(),
        );
        let write_mask_grid = write_mask_f.clone().repeat_dim(1, GRID_LEN);
        if track_policy {
            let mut selected_log_prob = selected_log_prob.expect("selected_log_prob");
            let mut step_entropy = step_entropy.expect("step_entropy");
            if let Some(gate_log_prob) = gate_log_prob {
                selected_log_prob = selected_log_prob + gate_log_prob;
            }
            if let Some(gate_entropy) = gate_entropy {
                step_entropy = step_entropy + gate_entropy;
            }
            log_prob_sum = log_prob_sum + selected_log_prob;
            policy_entropy_sum = policy_entropy_sum + step_entropy;
            policy_steps += 1;
        }

        let update_mask_base = action_one_hot.clone().mul(editable_mask.clone());
        let update_mask = update_mask_base
            .clone()
            .mul(write_mask_grid.clone())
            .greater_equal_elem(0.5);
        let update_any = update_mask
            .clone()
            .float()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .greater_elem(0.0)
            .float();
        let no_update = ones_step.clone().sub(update_any).mul(active_mask.clone());
        let no_op_penalty_per_sample = no_update
            .clone()
            .reshape([batch_size.max(1)])
            .mul_scalar(no_op_penalty);
        let update_values_grid = update_values.clone().repeat_dim(1, GRID_LEN);
        tokens = tokens.mask_where(update_mask.clone(), update_values_grid);
        unknown_mask = (unknown_mask - update_mask.clone().float()).clamp_min(0.0);
        let update_values_pred_grid = pred_values.clone().repeat_dim(1, GRID_LEN);
        tokens_reward = tokens_reward.mask_where(update_mask.clone(), update_values_pred_grid);

        reward_mask = reward_mask.mul(write_mask_f.clone().reshape([batch_size.max(1)]));

        let update_mask_cache = if training.policy.cache_update_clues {
            action_one_hot.clone()
        } else {
            action_one_hot.clone().mul(editable_mask.clone())
        };
        let update_mask_f = update_mask_cache.clone().unsqueeze_dim::<3>(2);
        let update_emb = update_emb
            .reshape([batch_size.max(1), cache_streams, 1, embd])
            .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
        let update_mask_stream = update_mask_f.clone().unsqueeze_dim::<4>(1);
        let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_stream);
        cache = mhc_passthrough(model.cache_mhc.as_ref(), cache);

        let tokens_solved_after = tokens_reward
            .clone()
            .equal(solutions.clone())
            .float()
            .sum_dim(1)
            .reshape([batch_size.max(1)])
            .equal_elem(GRID_LEN as f32)
            .float()
            .reshape([batch_size.max(1), 1]);

        let halt_target = tokens_solved_after.clone();
        let step_halt_loss = halt_bce_loss(halt_logit, halt_target.clone());
        halt_loss_sum = halt_loss_sum + step_halt_loss.clone();
        let halt_prob_masked = halt_prob.clone().mul(halt_target.clone());
        halt_prob_sum = halt_prob_sum + halt_prob_masked.mean();
        halt_target_sum = halt_target_sum + halt_target.mean();

        let mut step_halt = halt_prob.clone().greater_equal_elem(0.5).float();
        if step_idx + 1 < training.halt.min_steps {
            step_halt = step_halt.mul_scalar(0.0);
        }
        if training.halt.exploration_prob > 0.0 {
            let explore = Tensor::<B, 2>::random(
                [batch_size.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(training.halt.exploration_prob)
            .float();
            step_halt = step_halt.mul(ones_step.clone().sub(explore));
        }
        halted = halted.max_pair(step_halt.clone());

        let (reward_acc_per_sample, _reward_acc_mean) =
            compute_grid_accuracy_masked(&tokens_reward, &solutions, &editable_mask);
        let (_acc_per_sample, acc, exact_acc, solve_rate) =
            compute_grid_accuracy(&tokens_reward, &solutions);
        let step_acc_delta = reward_acc_per_sample
            .clone()
            .sub(prev_reward_acc_per_sample.clone());
        prev_reward_acc_per_sample = reward_acc_per_sample.detach();
        last_reward_acc_per_sample = prev_reward_acc_per_sample.clone();
        last_acc = acc;
        last_exact = exact_acc;
        last_solve_rate = solve_rate;

        let mut global_loss = zeros.clone();
        let mut global_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        let mut reward_global_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        if recon_loss_interval > 0
            && (step_idx + 1) % recon_loss_interval == 0
            && global_samples > 0
        {
            let sample_prob = (global_samples as f32 / GRID_LEN as f32).clamp(0.0, 1.0);
            let random = Tensor::<B, 2>::random(
                [batch_size.max(1), GRID_LEN],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            );
            let mut reward_global_mask = random.clone().lower_equal_elem(sample_prob).float();
            let mut global_mask = random.lower_equal_elem(sample_prob).float();
            global_mask = (global_mask + action_one_hot.clone()).clamp_max(1.0);
            reward_global_mask = (reward_global_mask + action_one_hot.clone()).clamp_max(1.0);
            global_mask = global_mask.mul(active_mask_grid.clone());
            reward_global_mask = reward_global_mask.mul(active_mask_grid.clone());
            reward_global_mask = reward_global_mask.mul(editable_mask.clone());
            if matches!(training.recon.loss_mask, SudokuLossMask::Unknown) {
                global_mask = global_mask.mul(loss_unknown_mask.clone());
                reward_global_mask =
                    reward_global_mask.mul(tokens_reward.clone().equal_elem(0).float());
            }
            global_mask = ensure_non_empty_mask(global_mask, action_one_hot.clone());
            let global_logits = model.value_logits_from_cache(cache_read.clone());
            let (step_loss, _step_acc, _step_exact, step_loss_per_sample, ..) =
                compute_loss_and_acc(
                    &global_logits,
                    &solutions,
                    &solution_one_hot,
                    &global_mask,
                    &training.recon.loss,
                );
            global_loss = step_loss;
            global_loss_per_sample = step_loss_per_sample;

            let (
                _reward_step_loss,
                _reward_step_acc,
                _reward_step_exact,
                reward_step_loss_per_sample,
                ..,
            ) = compute_loss_and_acc(
                &global_logits,
                &solutions,
                &solution_one_hot,
                &reward_global_mask,
                &training.recon.loss,
            );
            reward_global_loss_per_sample = reward_step_loss_per_sample;
        }

        let _step_recon_per_sample =
            local_loss_per_sample + global_loss_per_sample.mul_scalar(global_weight);
        let step_recon_per_sample_reward =
            reward_local_loss_per_sample + reward_global_loss_per_sample.mul_scalar(global_weight);
        let step_recon_per_sample_reward_detached = step_recon_per_sample_reward.clone().detach();
        let mut baseline_loss = last_loss_per_sample.clone();
        let baseline_mask = baseline_loss.clone().equal_elem(0.0);
        baseline_loss =
            baseline_loss.mask_where(baseline_mask, step_recon_per_sample_reward_detached.clone());
        let step_recon_delta = baseline_loss
            .sub(step_recon_per_sample_reward_detached.clone())
            .mul(reward_mask.clone());
        let step_reward_mask = reward_mask
            .clone()
            .add(no_update.clone().reshape([batch_size.max(1)]))
            .clamp_max(1.0);
        let step_recon_delta_reward = step_recon_delta
            .clone()
            .sub(no_op_penalty_per_sample.clone());
        chunk_recon_delta_sum = chunk_recon_delta_sum + step_recon_delta_reward.clone();
        chunk_recon_mask_sum = chunk_recon_mask_sum + step_reward_mask.clone();
        rollout_recon_delta_sum = rollout_recon_delta_sum + step_recon_delta_reward.clone();
        rollout_recon_mask_sum = rollout_recon_mask_sum + step_reward_mask.clone();
        let step_active_mask = active_mask.clone().reshape([batch_size.max(1)]);
        let write_gate_step = write_gate
            .clone()
            .reshape([batch_size.max(1)])
            .mul(step_active_mask.clone());
        let write_rate_step = write_mask_f
            .clone()
            .reshape([batch_size.max(1)])
            .mul(step_active_mask.clone());
        write_gate_sum = write_gate_sum + write_gate_step.mean();
        write_rate_sum = write_rate_sum + write_rate_step.mean();
        let step_acc_delta_reward = step_acc_delta
            .clone()
            .mul(step_active_mask.clone())
            .sub(no_op_penalty_per_sample.clone());
        chunk_acc_delta_sum = chunk_acc_delta_sum + step_acc_delta_reward.clone();
        chunk_acc_delta_mask_sum = chunk_acc_delta_mask_sum + step_active_mask.clone();
        rollout_acc_delta_sum = rollout_acc_delta_sum + step_acc_delta_reward.clone();
        rollout_acc_delta_mask_sum = rollout_acc_delta_mask_sum + step_active_mask.clone();
        if info_enabled && step_idx % info_stride == 0 {
            info_reward_sum = info_reward_sum + step_recon_delta.clone();
        }

        let mut step_reward = match easy_mode {
            SudokuEasyRewardMode::AccuracyDelta => step_acc_delta_reward.clone(),
            _ => step_recon_delta_reward.clone(),
        };
        if shaping_enabled {
            let mut shaping_step = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            let keep = Tensor::<B, 1>::ones([batch_size.max(1)], &device).sub(reward_mask.clone());
            let step_correct = pred_values
                .clone()
                .equal(selected_solution.clone())
                .float()
                .reshape([batch_size.max(1)])
                .mul(reward_mask.clone())
                .mul(step_active_mask.clone());
            if conflict_weight > 0.0 {
                let conflict_next = shaping_potential(&tokens, shaping_metric);
                let mut shaping_delta =
                    conflict_next.clone().mul_scalar(shaping_gamma) - conflict_prev.clone();
                shaping_delta = shaping_delta.mul(reward_mask.clone());
                shaping_step = shaping_step + shaping_delta.mul_scalar(conflict_weight);
                conflict_prev =
                    conflict_prev.mul(keep.clone()) + conflict_next.mul(reward_mask.clone());
            }
            if unknown_weight > 0.0 {
                let unknown_next = unknown_potential(&tokens);
                let mut shaping_delta =
                    unknown_next.clone().mul_scalar(shaping_gamma) - unknown_prev.clone();
                shaping_delta = shaping_delta
                    .mul(reward_mask.clone())
                    .mul(step_correct.clone());
                shaping_step = shaping_step + shaping_delta.mul_scalar(unknown_weight);
                unknown_prev =
                    unknown_prev.mul(keep.clone()) + unknown_next.mul(reward_mask.clone());
            }
            if accuracy_weight > 0.0 {
                let acc_delta = step_acc_delta.clone().mul(step_active_mask.clone());
                shaping_step = shaping_step + acc_delta.mul_scalar(accuracy_weight);
            }
            if incorrect_penalty > 0.0 {
                let incorrect = reward_mask
                    .clone()
                    .mul(step_active_mask.clone())
                    .sub(step_correct)
                    .clamp_min(0.0);
                shaping_step = shaping_step - incorrect.mul_scalar(incorrect_penalty);
            }
            let shaping_step_masked = shaping_step.clone().mul(step_reward_mask.clone());
            rollout_shaping_sum = rollout_shaping_sum + shaping_step_masked;
            rollout_shaping_mask_sum = rollout_shaping_mask_sum + step_reward_mask.clone();
            step_reward = step_reward + shaping_step;
        }
        step_reward = step_reward.mul(step_reward_mask.clone());
        if baseline_enabled && let Some(value) = step_value.clone() {
            rollout_step_rewards.push(step_reward.clone());
            rollout_step_values.push(value);
        }

        let reward_keep =
            Tensor::<B, 1>::ones([batch_size.max(1)], &device).sub(reward_mask.clone());
        last_loss_per_sample = last_loss_per_sample.mul(reward_keep)
            + step_recon_per_sample_reward_detached.mul(reward_mask.clone());
        let step_recon = local_loss.clone() + global_loss.clone().mul_scalar(global_weight);
        recon_loss_sum = recon_loss_sum + step_recon;
        recon_steps += 1;
    }

    let recon_loss = if recon_steps > 0 {
        recon_loss_sum.div_scalar(recon_steps as f32)
    } else {
        Tensor::<B, 1>::zeros([1], &device)
    };
    let halt_loss = halt_loss_sum.div_scalar(rollout_steps as f32);
    let halt_target_mean = halt_target_sum.clone().div_scalar(rollout_steps as f32);
    let halt_prob_mean = halt_prob_sum.div(halt_target_sum.clamp_min(HALT_EPS));

    let log_prob_sum = if track_policy {
        Some(log_prob_sum)
    } else {
        None
    };
    let policy_entropy_sum = if track_policy {
        policy_entropy_sum
    } else {
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device)
    };
    let policy_steps = if track_policy { policy_steps } else { 0 };
    let selectable_count_sum = if track_policy {
        selectable_count_sum
    } else {
        zeros.clone()
    };
    let selectable_steps = if track_policy { selectable_steps } else { 0 };

    let hard_reward = match hard_mode {
        SudokuHardRewardMode::InfoReward => info_reward_sum.clone(),
        SudokuHardRewardMode::Accuracy => last_reward_acc_per_sample
            .clone()
            .sub(reward_initial_acc.clone()),
    };
    let rollout_shaping_mean = if shaping_enabled {
        rollout_shaping_sum
            .clone()
            .div(rollout_shaping_mask_sum.clone().clamp_min(1.0))
    } else {
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device)
    };
    let mut easy_reward = match easy_mode {
        SudokuEasyRewardMode::Recon => rollout_recon_delta_sum
            .clone()
            .div(rollout_recon_mask_sum.clone().clamp_min(1.0)),
        SudokuEasyRewardMode::AccuracyDelta => rollout_acc_delta_sum
            .clone()
            .div(rollout_acc_delta_mask_sum.clone().clamp_min(1.0)),
        SudokuEasyRewardMode::Gae => {
            if baseline_enabled && !rollout_step_rewards.is_empty() {
                let values_detached: Vec<_> = rollout_step_values
                    .iter()
                    .map(|value| value.clone().detach())
                    .collect();
                gae_advantage(
                    &rollout_step_rewards,
                    &values_detached,
                    baseline_gamma,
                    baseline_lambda,
                )
            } else {
                let mut easy = last_loss_per_sample.clone().mul_scalar(-1.0);
                if shaping_enabled {
                    easy = easy + rollout_shaping_sum.clone();
                }
                easy
            }
        }
    };
    if shaping_enabled
        && matches!(
            easy_mode,
            SudokuEasyRewardMode::Recon | SudokuEasyRewardMode::AccuracyDelta
        )
    {
        easy_reward = easy_reward + rollout_shaping_mean.clone();
    }
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale.clone());

    let action_count = action_count_sum.clone().clamp_min(SUDOKU_EPS);
    let saccade_revisit_rate = revisit_count_sum.clone().div(action_count.clone());
    let saccade_repeat_rate = repeat_count_sum.clone().div(action_count.clone());
    let saccade_unknown_frac = unknown_select_sum.clone().div(action_count.clone());
    let unique_cells = visit_counts.clone().greater_elem(0.0).float().sum_dim(1);
    let total_visits = visit_counts.clone().sum_dim(1).clamp_min(1.0);
    let saccade_unique_frac = unique_cells.div(total_visits).mean();
    let write_gate_mean = if rollout_steps > 0 {
        write_gate_sum.clone().div_scalar(rollout_steps as f32)
    } else {
        zeros.clone()
    };
    let write_rate = if rollout_steps > 0 {
        write_rate_sum.clone().div_scalar(rollout_steps as f32)
    } else {
        zeros.clone()
    };

    RolloutBase {
        recon_loss,
        acc: last_acc,
        exact_acc: last_exact,
        solve_rate: last_solve_rate,
        hard_reward,
        easy_reward,
        saccade_revisit_rate,
        saccade_repeat_rate,
        saccade_unknown_frac,
        saccade_unique_frac,
        write_gate_mean,
        write_rate,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        log_prob_sum,
        policy_entropy_sum,
        policy_steps,
        selectable_count_sum,
        selectable_steps,
        batch: batch_size,
    }
}
