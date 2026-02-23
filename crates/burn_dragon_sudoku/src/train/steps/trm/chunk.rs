use super::*;

pub(crate) fn rollout_losses_train_trm_chunk<B: AutodiffBackend>(
    trainer: &SudokuTrainer<B>,
    batch: SudokuBatch<B>,
    teacher_forcing_prob: f32,
    recon_loss_weight: f32,
    rollout_steps: usize,
) -> (SudokuLosses<B>, GradientsParams) {
    let training = &trainer.training;
    let rollout_steps = rollout_steps.max(1);
    let backprop_steps_cfg = training.rollout.backprop_steps.unwrap_or(rollout_steps);
    let chunk_steps = if backprop_steps_cfg == 0 {
        rollout_steps
    } else {
        backprop_steps_cfg.min(rollout_steps).max(1)
    };
    let trm_chunk_size = training.rollout.trm_chunk_size.clamp(1, GRID_LEN);
    let recon_interval = training.recon.loss_interval_steps;
    let global_weight = training.recon.global_loss_weight.max(0.0);
    let global_samples = training.recon.global_loss_samples.min(GRID_LEN);

    let puzzles = batch.puzzles;
    let solutions = batch.solutions;
    let device = puzzles.device();
    let [batch_size, _] = puzzles.shape().dims::<2>();

    let ones_grid = Tensor::<B, 2>::ones([batch_size.max(1), GRID_LEN], &device);
    let solution_one_hot = build_solution_one_hot(&solutions, batch_size, &device);
    let (row_ids, col_ids) = trainer.model.grid_row_col_ids(batch_size, &device);

    let clue_mask = puzzles.clone().greater_elem(0.0).float();
    let editable_mask = ones_grid.clone().sub(clue_mask.clone());

    let mut tokens = puzzles;
    let loss_unknown_mask = tokens.clone().equal_elem(0).float();
    let mut unknown_mask = loss_unknown_mask.clone();
    let mut tokens_reward = tokens.clone();

    let mut state = trainer.model.init_state();
    let input_cache = trainer.model.cell_embeddings_with_positions(
        tokens.clone(),
        row_ids.clone(),
        col_ids.clone(),
    );
    let [_, _, embd] = input_cache.shape().dims();
    let cache_streams = trainer.model.cache_streams();
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
    let mut summary_tokens = trainer.model.init_summary_tokens(batch_size);
    let summary_len = trainer.model.summary_token_count();

    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let mut recon_loss_sum_metrics = zeros.clone();
    let mut recon_steps_metrics = 0usize;
    let mut chunk_local_loss_sum = zeros.clone();
    let mut chunk_global_loss_sum = zeros.clone();
    let mut chunk_global_steps = 0usize;
    let mut steps_done = 0usize;

    let pre_steps = if matches!(training.rollout.traversal, SudokuTraversal::L2rT2b) {
        sample_pre_rollout_steps(training)
    } else {
        0
    };

    if pre_steps > 0 {
        let input_cache_read_pre = input_cache_read.clone().detach();
        summary_tokens = summary_tokens.detach();
        cache = cache.detach();
        unknown_mask = unknown_mask.detach();
        detach_state(&mut state);
        let mut pre_done = 0usize;
        while pre_done < pre_steps {
            let remaining = pre_steps - pre_done;
            let chunk_len = remaining.min(trm_chunk_size).max(1);
            let grid_start = pre_done % GRID_LEN;

            let row_ids_chunk = slice_grid_2d_int(&row_ids, grid_start, chunk_len);
            let col_ids_chunk = slice_grid_2d_int(&col_ids, grid_start, chunk_len);
            let input_cache_read_chunk =
                slice_grid_3d(&input_cache_read_pre, grid_start, chunk_len);
            let cache_read = cache
                .clone()
                .mean_dim(1)
                .reshape([batch_size.max(1), GRID_LEN, embd]);
            let cache_read_chunk = slice_grid_3d(&cache_read, grid_start, chunk_len);

            let step_input =
                trainer.model.project_input_tokens(input_cache_read_chunk) + cache_read_chunk;
            let chunk_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
            let (hidden, _logits_full) = trainer
                .model
                .forward_with_hidden_and_state_embedded(chunk_input, &mut state);
            let hidden = hidden.detach();
            let summary_hidden = hidden.clone().slice_dim(1, 0..summary_len);
            summary_tokens = trainer.model.normalize_summary_tokens(summary_hidden);
            let step_hidden = hidden.slice_dim(1, summary_len..summary_len + chunk_len);
            let step_logits = trainer
                .model
                .value_logits_from_hidden(step_hidden.clone())
                .detach();

            let solutions_chunk = slice_grid_2d_int(&solutions, grid_start, chunk_len);
            let teacher_force = if teacher_forcing_prob > 0.0 {
                Tensor::<B, 2>::random(
                    [batch_size.max(1), chunk_len],
                    TensorDistribution::Uniform(0.0, 1.0),
                    &device,
                )
                .lower_equal_elem(teacher_forcing_prob)
                .float()
            } else {
                Tensor::<B, 2>::zeros([batch_size.max(1), chunk_len], &device)
            };
            let teacher_mask = teacher_force.clone().greater_equal_elem(0.5);
            let pred_values = step_logits
                .argmax(2)
                .reshape([batch_size.max(1), chunk_len]);
            let mut update_values = pred_values.clone();
            update_values = update_values.mask_where(teacher_mask, solutions_chunk.clone());

            let editable_chunk = slice_grid_2d(&editable_mask, grid_start, chunk_len);
            let update_mask_chunk = editable_chunk.clone().greater_equal_elem(0.5);
            let tokens_chunk = slice_grid_2d_int(&tokens, grid_start, chunk_len)
                .mask_where(update_mask_chunk.clone(), update_values.clone());
            tokens = replace_grid_2d_int(&tokens, grid_start, tokens_chunk);

            let reward_chunk = slice_grid_2d_int(&tokens_reward, grid_start, chunk_len)
                .mask_where(update_mask_chunk.clone(), pred_values.clone());
            tokens_reward = replace_grid_2d_int(&tokens_reward, grid_start, reward_chunk);

            let update_mask_cache = if training.policy.cache_update_clues {
                Tensor::<B, 2>::ones([batch_size.max(1), chunk_len], &device)
            } else {
                editable_chunk.clone()
            };
            let update_mask_stream = update_mask_cache
                .clone()
                .unsqueeze_dim::<3>(2)
                .unsqueeze_dim::<4>(1)
                .expand([batch_size.max(1), cache_streams, chunk_len, 1]);
            let cache_chunk = slice_grid_4d(&cache, grid_start, chunk_len);
            let cache_cell = cache_chunk.clone().swap_dims(1, 2).reshape([
                batch_size.max(1) * chunk_len * cache_streams,
                1,
                embd,
            ]);
            let summary_streams = step_hidden
                .clone()
                .unsqueeze_dim::<4>(2)
                .expand([batch_size.max(1), chunk_len, cache_streams, embd])
                .reshape([batch_size.max(1) * chunk_len * cache_streams, 1, embd]);
            let token_emb = trainer.model.cell_embeddings_with_positions(
                update_values.clone(),
                row_ids_chunk.clone(),
                col_ids_chunk.clone(),
            );
            let token_emb = token_emb
                .unsqueeze_dim::<4>(2)
                .expand([batch_size.max(1), chunk_len, cache_streams, embd])
                .reshape([batch_size.max(1) * chunk_len * cache_streams, 1, embd]);
            let update_emb = trainer
                .model
                .update_cell_embedding(summary_streams, cache_cell, token_emb)
                .detach();
            let update_emb = update_emb
                .reshape([batch_size.max(1), chunk_len, cache_streams, embd])
                .swap_dims(1, 2);
            let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
            let cache_chunk = cache_chunk * keep + update_emb.mul(update_mask_stream);
            cache = replace_grid_4d(&cache, grid_start, cache_chunk);
            cache = mhc_passthrough(trainer.model.cache_mhc.as_ref(), cache);

            summary_tokens = summary_tokens.detach();
            cache = cache.detach();
            unknown_mask = unknown_mask.detach();
            detach_state(&mut state);
            pre_done += chunk_len;
        }
    }

    let total_local_steps = rollout_steps.max(1);

    let mut grads_accum = GradientsAccumulator::<SudokuTrainer<B>>::new();

    while steps_done < rollout_steps {
        let remaining = rollout_steps - steps_done;
        let chunk_len = remaining.min(trm_chunk_size).max(1);
        let grid_start = (steps_done + pre_steps) % GRID_LEN;

        let row_ids_chunk = slice_grid_2d_int(&row_ids, grid_start, chunk_len);
        let col_ids_chunk = slice_grid_2d_int(&col_ids, grid_start, chunk_len);
        let input_cache_read_chunk = slice_grid_3d(&input_cache_read, grid_start, chunk_len);
        let cache_read = cache
            .clone()
            .mean_dim(1)
            .reshape([batch_size.max(1), GRID_LEN, embd]);
        let cache_read_chunk = slice_grid_3d(&cache_read, grid_start, chunk_len);

        let step_input =
            trainer.model.project_input_tokens(input_cache_read_chunk) + cache_read_chunk;
        let chunk_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
        let (hidden, _logits_full) = trainer
            .model
            .forward_with_hidden_and_state_embedded(chunk_input, &mut state);
        let summary_hidden = hidden.clone().slice_dim(1, 0..summary_len);
        summary_tokens = trainer.model.normalize_summary_tokens(summary_hidden);
        let step_hidden = hidden.slice_dim(1, summary_len..summary_len + chunk_len);
        let step_logits = trainer.model.value_logits_from_hidden(step_hidden.clone());

        let solutions_chunk = slice_grid_2d_int(&solutions, grid_start, chunk_len);
        let solution_one_hot_chunk = slice_grid_3d(&solution_one_hot, grid_start, chunk_len);
        let mut loss_mask = Tensor::<B, 2>::ones([batch_size.max(1), chunk_len], &device);
        if matches!(training.recon.loss_mask, SudokuLossMask::Unknown) {
            let loss_unknown_chunk = slice_grid_2d(&loss_unknown_mask, grid_start, chunk_len);
            loss_mask = loss_mask.mul(loss_unknown_chunk);
        }
        loss_mask = ensure_non_empty_mask(
            loss_mask,
            Tensor::<B, 2>::ones([batch_size.max(1), chunk_len], &device),
        );
        let (chunk_loss, _chunk_acc, _chunk_exact, ..) = compute_loss_and_acc(
            &step_logits,
            &solutions_chunk,
            &solution_one_hot_chunk,
            &loss_mask,
            &training.recon.loss,
        );

        let chunk_weight = chunk_len as f32;
        chunk_local_loss_sum = chunk_local_loss_sum + chunk_loss.clone().mul_scalar(chunk_weight);
        recon_loss_sum_metrics =
            recon_loss_sum_metrics + chunk_loss.clone().detach().mul_scalar(chunk_weight);
        recon_steps_metrics += chunk_len;

        let teacher_force = if teacher_forcing_prob > 0.0 {
            Tensor::<B, 2>::random(
                [batch_size.max(1), chunk_len],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(teacher_forcing_prob)
            .float()
        } else {
            Tensor::<B, 2>::zeros([batch_size.max(1), chunk_len], &device)
        };
        let teacher_mask = teacher_force.clone().greater_equal_elem(0.5);
        let pred_values = step_logits
            .argmax(2)
            .reshape([batch_size.max(1), chunk_len]);
        let mut update_values = pred_values.clone();
        update_values = update_values.mask_where(teacher_mask, solutions_chunk.clone());

        let editable_chunk = slice_grid_2d(&editable_mask, grid_start, chunk_len);
        let update_mask_chunk = editable_chunk.clone().greater_equal_elem(0.5);
        let tokens_chunk = slice_grid_2d_int(&tokens, grid_start, chunk_len)
            .mask_where(update_mask_chunk.clone(), update_values.clone());
        tokens = replace_grid_2d_int(&tokens, grid_start, tokens_chunk);

        let reward_chunk = slice_grid_2d_int(&tokens_reward, grid_start, chunk_len)
            .mask_where(update_mask_chunk.clone(), pred_values.clone());
        tokens_reward = replace_grid_2d_int(&tokens_reward, grid_start, reward_chunk);

        let unknown_chunk = Tensor::<B, 2>::zeros([batch_size.max(1), chunk_len], &device);
        unknown_mask = replace_grid_2d(&unknown_mask, grid_start, unknown_chunk);

        let update_mask_cache = if training.policy.cache_update_clues {
            Tensor::<B, 2>::ones([batch_size.max(1), chunk_len], &device)
        } else {
            editable_chunk.clone()
        };
        let update_mask_stream = update_mask_cache
            .clone()
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(1)
            .expand([batch_size.max(1), cache_streams, chunk_len, 1]);
        let cache_chunk = slice_grid_4d(&cache, grid_start, chunk_len);
        let cache_cell = cache_chunk.clone().swap_dims(1, 2).reshape([
            batch_size.max(1) * chunk_len * cache_streams,
            1,
            embd,
        ]);
        let summary_streams = step_hidden
            .clone()
            .unsqueeze_dim::<4>(2)
            .expand([batch_size.max(1), chunk_len, cache_streams, embd])
            .reshape([batch_size.max(1) * chunk_len * cache_streams, 1, embd]);
        let token_emb = trainer.model.cell_embeddings_with_positions(
            update_values.clone(),
            row_ids_chunk.clone(),
            col_ids_chunk.clone(),
        );
        let token_emb = token_emb
            .unsqueeze_dim::<4>(2)
            .expand([batch_size.max(1), chunk_len, cache_streams, embd])
            .reshape([batch_size.max(1) * chunk_len * cache_streams, 1, embd]);
        let update_emb =
            trainer
                .model
                .update_cell_embedding(summary_streams, cache_cell, token_emb);
        let update_emb = update_emb
            .reshape([batch_size.max(1), chunk_len, cache_streams, embd])
            .swap_dims(1, 2);
        let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
        let cache_chunk = cache_chunk * keep + update_emb.mul(update_mask_stream);
        cache = replace_grid_4d(&cache, grid_start, cache_chunk);
        cache = mhc_passthrough(trainer.model.cache_mhc.as_ref(), cache);

        if recon_interval > 0 && global_samples > 0 {
            let chunk_idx = steps_done / trm_chunk_size;
            if (chunk_idx + 1).is_multiple_of(recon_interval) {
                let cache_read =
                    cache
                        .clone()
                        .mean_dim(1)
                        .reshape([batch_size.max(1), GRID_LEN, embd]);
                let global_logits = trainer.model.value_logits_from_cache(cache_read);
                let sample_prob = (global_samples as f32 / GRID_LEN as f32).clamp(0.0, 1.0);
                let random = Tensor::<B, 2>::random(
                    [batch_size.max(1), GRID_LEN],
                    TensorDistribution::Uniform(0.0, 1.0),
                    &device,
                );
                let mut global_mask = random.lower_equal_elem(sample_prob).float();
                let chunk_action_mask = replace_grid_2d(
                    &Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device),
                    grid_start,
                    Tensor::<B, 2>::ones([batch_size.max(1), chunk_len], &device),
                );
                global_mask = (global_mask + chunk_action_mask.clone()).clamp_max(1.0);
                if matches!(training.recon.loss_mask, SudokuLossMask::Unknown) {
                    global_mask = global_mask.mul(loss_unknown_mask.clone());
                }
                global_mask = ensure_non_empty_mask(global_mask, chunk_action_mask);
                let (global_loss, ..) = compute_loss_and_acc(
                    &global_logits,
                    &solutions,
                    &solution_one_hot,
                    &global_mask,
                    &training.recon.loss,
                );
                chunk_global_loss_sum =
                    chunk_global_loss_sum + global_loss.clone().mul_scalar(chunk_weight);
                recon_loss_sum_metrics = recon_loss_sum_metrics
                    + global_loss
                        .detach()
                        .mul_scalar(chunk_weight * global_weight);
                chunk_global_steps += chunk_len;
            }
        }

        steps_done += chunk_len;
        let chunk_end = steps_done.is_multiple_of(chunk_steps) || steps_done == rollout_steps;
        if chunk_end {
            let local_term = chunk_local_loss_sum
                .clone()
                .div_scalar(total_local_steps as f32);
            let global_term = if chunk_global_steps > 0 {
                chunk_global_loss_sum
                    .clone()
                    .div_scalar(chunk_global_steps as f32)
            } else {
                zeros.clone()
            };
            let recon_term = local_term + global_term.mul_scalar(global_weight);
            let recon_term = recon_term.mul_scalar(recon_loss_weight);
            let chunk_loss = recon_term;
            let grads = GradientsParams::from_grads(chunk_loss.backward(), trainer);
            grads_accum.accumulate(trainer, grads);

            chunk_local_loss_sum = zeros.clone();
            chunk_global_loss_sum = zeros.clone();
            chunk_global_steps = 0;

            detach_state(&mut state);
            summary_tokens = summary_tokens.detach();
            cache = cache.detach();
        }
    }

    let grads = grads_accum.grads();

    let recon_loss = if recon_steps_metrics > 0 {
        recon_loss_sum_metrics.div_scalar(recon_steps_metrics as f32)
    } else {
        zeros.clone()
    };
    let (_acc_per_sample, acc, exact_acc, solve_rate) =
        compute_grid_accuracy(&tokens_reward, &solutions);

    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let loss = recon_loss.clone().mul_scalar(recon_loss_weight);
    let policy_entropy_alpha = zeros.clone();
    let policy_entropy_target = zeros.clone();

    (
        SudokuLosses {
            loss,
            recon_loss,
            acc,
            exact_acc,
            solve_rate,
            policy_loss: zeros.clone(),
            halt_loss: zeros.clone(),
            halt_prob_mean: zeros.clone(),
            halt_target_mean: zeros.clone(),
            advantage_abs_mean: zeros.clone(),
            advantage_std: zeros.clone(),
            log_prob_mean: zeros.clone(),
            policy_entropy: zeros.clone(),
            policy_entropy_alpha,
            policy_entropy_target,
            hard_reward_mean: zeros.clone(),
            easy_reward_mean: zeros.clone(),
            shaping_conflict_mean: zeros.clone(),
            shaping_unknown_mean: zeros.clone(),
            shaping_accuracy_mean: zeros.clone(),
            shaping_incorrect_mean: zeros.clone(),
            saccade_revisit_rate: zeros.clone(),
            saccade_repeat_rate: zeros.clone(),
            saccade_unknown_frac: zeros.clone(),
            saccade_unique_frac: zeros.clone(),
            write_gate_mean: zeros.clone(),
            write_rate: zeros,
        },
        grads,
    )
}
