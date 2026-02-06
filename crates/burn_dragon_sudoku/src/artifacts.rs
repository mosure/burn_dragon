use std::path::Path;

use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData};

use burn_dragon_core::mhc_passthrough;
use burn_dragon_train::VisionArtifactOutputMode;
use burn_dragon_train::train::artifacts::{ArtifactFrame, write_video};

use crate::config::{SudokuArtifactConfig, SudokuTrainingHyperparameters};
use crate::dataset::{SudokuBatch, SudokuDataset, SudokuSplit};
use crate::model::SudokuSaccadeModel;
use crate::train::{sample_actions, static_traversal_actions};
use crate::vocab::{GRID_LEN, VOCAB_SIZE};

const GRID_SIZE: usize = 9;
const CELL_SIZE: usize = 32;
const GRID_PAD: usize = 16;
const LINE_THIN: usize = 1;
const LINE_THICK: usize = 3;
const DIGIT_SCALE: usize = 4;
const DIGIT_PAD: usize = 4;
const HUD_SCALE: usize = 2;

const COLOR_BG: [u8; 3] = [245, 242, 235];
const COLOR_CLUE_BG: [u8; 3] = [220, 230, 242];
const COLOR_LINE: [u8; 3] = [30, 30, 30];
const COLOR_DIGIT: [u8; 3] = [25, 25, 25];
const COLOR_HIGHLIGHT: [u8; 3] = [255, 187, 0];
const COLOR_WRITE_BG: [u8; 3] = [210, 210, 210];
const COLOR_WRITE_FILL: [u8; 3] = [78, 158, 105];
const WRITE_BAR_HEIGHT: usize = 6;
const ARTIFACT_EPS_FLOOR: f32 = 0.05;
const POLICY_MASK_PENALTY: f32 = 1e9;

pub fn write_validation_artifacts<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    dataset: &SudokuDataset,
    config: &SudokuArtifactConfig,
    training: &SudokuTrainingHyperparameters,
    run_dir: &Path,
    device: &B::Device,
) -> Result<()> {
    if config.max_samples == 0 {
        return Ok(());
    }

    let batch = dataset.sample_batch::<B>(SudokuSplit::Val, device);
    write_validation_artifacts_from_batch(model, &batch, config, training, run_dir, 0)
}

pub fn write_validation_artifacts_from_batch<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: &SudokuBatch<B>,
    config: &SudokuArtifactConfig,
    training: &SudokuTrainingHyperparameters,
    run_dir: &Path,
    epoch: usize,
) -> Result<()> {
    if config.max_samples == 0 {
        return Ok(());
    }
    let device = batch.puzzles.device();
    let puzzles = batch
        .puzzles
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("puzzle to vec: {err:?}"))?;
    let solutions = batch
        .solutions
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .map_err(|err| anyhow!("solution to vec: {err:?}"))?;

    let batch_size = puzzles.len() / GRID_LEN;
    if batch_size == 0 {
        return Ok(());
    }
    let sample_count = config.max_samples.min(batch_size).max(1);
    let mut puzzle_grids = Vec::with_capacity(sample_count);
    let mut solution_grids = Vec::with_capacity(sample_count);
    for idx in 0..sample_count {
        let mut grid = Vec::with_capacity(GRID_LEN);
        let mut sol = Vec::with_capacity(GRID_LEN);
        for cell in 0..GRID_LEN {
            grid.push(puzzles[idx * GRID_LEN + cell] as u8);
            sol.push(solutions[idx * GRID_LEN + cell] as u8);
        }
        puzzle_grids.push(grid);
        solution_grids.push(sol);
    }

    write_validation_artifacts_with_grids(
        model,
        puzzle_grids,
        solution_grids,
        config,
        training,
        run_dir,
        &device,
        epoch,
    )
}

#[allow(clippy::too_many_arguments)]
fn write_validation_artifacts_with_grids<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    mut puzzle_grids: Vec<Vec<u8>>,
    mut solution_grids: Vec<Vec<u8>>,
    config: &SudokuArtifactConfig,
    training: &SudokuTrainingHyperparameters,
    run_dir: &Path,
    device: &B::Device,
    epoch: usize,
) -> Result<()> {
    if config.max_samples == 0 {
        return Ok(());
    }
    if puzzle_grids.is_empty() || solution_grids.is_empty() {
        return Ok(());
    }

    let available = puzzle_grids.len().min(solution_grids.len());
    let sample_count = config.max_samples.min(available).max(1);
    puzzle_grids.truncate(sample_count);
    solution_grids.truncate(sample_count);

    let output_dir = run_dir.join("artifacts");
    let output_mode = match config.output {
        VisionArtifactOutputMode::Images => VisionArtifactOutputMode::Avi,
        other => other,
    };

    let rollout_steps = if training.rollout.max_steps > 0 {
        training.rollout.max_steps
    } else {
        training.rollout.steps.max(1)
    };
    let frames = generate_rollout_frames(
        model,
        puzzle_grids,
        solution_grids,
        config,
        training,
        rollout_steps,
        device,
    )?;

    for (sample_idx, sample_frames) in frames.into_iter().enumerate() {
        if sample_frames.is_empty() {
            continue;
        }
        let _outcome = write_video(
            &output_dir,
            output_mode,
            config.overwrite,
            epoch,
            sample_idx,
            &sample_frames,
            config.fps,
            None,
        )?;
    }

    Ok(())
}

fn generate_rollout_frames<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    mut puzzles: Vec<Vec<u8>>,
    solutions: Vec<Vec<u8>>,
    config: &SudokuArtifactConfig,
    training: &SudokuTrainingHyperparameters,
    steps: usize,
    device: &B::Device,
) -> Result<Vec<Vec<ArtifactFrame>>> {
    let batch = puzzles.len();
    let steps = steps.max(1);
    let mut frames = vec![Vec::with_capacity(steps); batch];
    if batch == 0 {
        return Ok(frames);
    }

    let mut unknown_masks = Vec::with_capacity(batch);
    let mut editable_masks = Vec::with_capacity(batch);
    let mut editable_counts = Vec::with_capacity(batch);
    let mut clue_masks = Vec::with_capacity(batch);
    for grid in puzzles.iter() {
        let mut unknown = Vec::with_capacity(GRID_LEN);
        let mut editable = Vec::with_capacity(GRID_LEN);
        let mut clue = Vec::with_capacity(GRID_LEN);
        for &value in grid.iter().take(GRID_LEN) {
            let is_clue = value > 0;
            editable.push(if is_clue { 0.0 } else { 1.0 });
            unknown.push(if value == 0 { 1.0 } else { 0.0 });
            clue.push(if is_clue { 1 } else { 0 });
        }
        let editable_count = editable.iter().copied().sum::<f32>().max(1.0);
        unknown_masks.push(unknown);
        editable_masks.push(editable);
        editable_counts.push(editable_count);
        clue_masks.push(clue);
    }

    let ones_grid = Tensor::<B, 2>::ones([batch.max(1), GRID_LEN], device);
    let ones_step = Tensor::<B, 2>::ones([batch.max(1), 1], device);
    let revisit_min_filled = training.revisit.min_filled_final.clamp(0.0, 1.0);
    let visit_penalty = training.policy.visit_penalty.max(0.0);
    let policy_temperature = training.policy.temperature.max(1e-4);
    let policy_epsilon = training
        .policy
        .epsilon
        .clamp(0.0, 1.0)
        .max(ARTIFACT_EPS_FLOOR);
    let policy_noise = training.policy.noise.max(0.0);
    let action_index = build_action_index(batch, device);
    let (row_ids, col_ids) = model.grid_row_col_ids(batch, device);
    let row_ids_f = row_ids.clone().float();
    let col_ids_f = col_ids.clone().float();

    let solution_tokens = build_tokens_tensor::<B>(&solutions, device);
    let init_tokens = build_tokens_tensor::<B>(&puzzles, device);
    let editable_mask_data: Vec<f32> = editable_masks
        .iter()
        .flat_map(|mask| mask.iter().copied())
        .collect();
    let editable_mask_tensor = Tensor::<B, 2>::from_data(
        TensorData::new(editable_mask_data, [batch.max(1), GRID_LEN]),
        device,
    );
    let input_cache =
        model.cell_embeddings_with_positions(init_tokens.clone(), row_ids.clone(), col_ids.clone());
    let [_, _, embd] = input_cache.shape().dims();
    let cache_streams = model.cache_streams();
    let input_cache = input_cache
        .unsqueeze_dim::<4>(1)
        .expand([batch.max(1), cache_streams, GRID_LEN, embd]);
    let input_cache_read = input_cache
        .clone()
        .mean_dim(1)
        .reshape([batch.max(1), GRID_LEN, embd]);
    let mut summary_tokens = model.init_summary_tokens(batch);
    let mut cache = input_cache.clone();
    let summary_len = model.summary_token_count();
    let mut state = model.init_state();
    let mut visit_counts =
        Tensor::<B, 2>::zeros([batch.max(1), GRID_LEN], device);

    for step_idx in 0..steps {
        let tokens = build_tokens_tensor::<B>(&puzzles, device);
        let cache_read = cache
            .clone()
            .mean_dim(1)
            .reshape([batch.max(1), GRID_LEN, embd]);
        let policy_logits =
            model.policy_logits_from_cache(summary_tokens.clone(), cache_read.clone());

        let (select_mask, selectable_counts) = build_select_mask(
            &unknown_masks,
            &editable_masks,
            &editable_counts,
            revisit_min_filled,
        );
        let select_mask = Tensor::<B, 2>::from_data(
            TensorData::new(select_mask, [batch.max(1), GRID_LEN]),
            device,
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
        let actions = match training.rollout.traversal {
            crate::config::SudokuTraversal::Saccade => {
                if config.sample_policy {
                    let policy_logits = if (policy_temperature - 1.0).abs() > f32::EPSILON {
                        masked_logits.clone().div_scalar(policy_temperature)
                    } else {
                        masked_logits.clone()
                    };
                    let sampled_logits = if policy_noise > 0.0 {
                        let noise = Tensor::<B, 2>::random(
                            [batch.max(1), GRID_LEN],
                            TensorDistribution::Normal(0.0, f64::from(policy_noise)),
                            device,
                        );
                        policy_logits + noise
                    } else {
                        policy_logits
                    };
                    sample_actions(sampled_logits, select_mask.clone(), true, policy_epsilon)
                } else {
                    masked_logits.argmax(1)
                }
            }
            crate::config::SudokuTraversal::L2rT2b => {
                static_traversal_actions(batch, step_idx, device)
            }
        };

        let tokens_solved_before = tokens
            .clone()
            .equal(solution_tokens.clone())
            .float()
            .sum_dim(1)
            .reshape([batch.max(1)])
            .equal_elem(GRID_LEN as f32)
            .float()
            .reshape([batch.max(1), 1]);
        let selectable_counts_t = select_mask
            .clone()
            .sum_dim(1)
            .reshape([batch.max(1), 1]);
        let active_mask = selectable_counts_t
            .greater_elem(0.0)
            .float()
            .mul(ones_step.clone().sub(tokens_solved_before.clone()));
        let mut action_one_hot = build_action_one_hot(&actions, &action_index);
        let active_mask_grid = active_mask.clone().repeat_dim(1, GRID_LEN);
        action_one_hot = action_one_hot * active_mask_grid.clone() * select_mask.clone();
        visit_counts = visit_counts + action_one_hot.clone();
        let [_, _, embd] = cache_read.shape().dims();
        let step_input_base = input_cache_read
            .clone()
            .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch.max(1), 1, embd]);
        let step_residual = cache_read
            .clone()
            .mul(action_one_hot.clone().unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .reshape([batch.max(1), 1, embd]);
        let step_input = model
            .project_input_tokens(step_input_base)
            + step_residual;
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
        let (step_hidden, _step_logits_full) =
            model.forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = model.normalize_summary_tokens(summary_hidden);
        let step_logits = model.value_logits_from_hidden(step_hidden.clone());

        let action_data = actions
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("actions to vec: {err:?}"))?;
        let pred_values = step_logits.argmax(2).reshape([batch.max(1), 1]);
        let update_mask_f = action_one_hot
            .clone()
            .mul(editable_mask_tensor.clone())
            .unsqueeze_dim::<3>(2);
        let selected_row = row_ids_f
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch.max(1), 1])
            .int();
        let selected_col = col_ids_f
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch.max(1), 1])
            .int();
                let action_mask = action_one_hot
            .clone()
            .unsqueeze_dim::<3>(2)
            .unsqueeze_dim::<4>(1);
        let cache_cell = cache
            .clone()
            .mul(action_mask.clone())
            .sum_dim(2)
            .reshape([batch.max(1) * cache_streams, 1, embd]);
        let summary_streams = summary_tokens
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batch.max(1), cache_streams, summary_len, embd])
            .reshape([batch.max(1) * cache_streams, summary_len, embd]);
        let token_emb = model.cell_embeddings_with_positions(
            pred_values.clone(),
            selected_row,
            selected_col,
        );
        let token_emb = token_emb
            .unsqueeze_dim::<4>(1)
            .expand([batch.max(1), cache_streams, 1, embd])
            .reshape([batch.max(1) * cache_streams, 1, embd]);
        let (update_emb, write_gate) = model.update_cell_embedding_with_gate(
            summary_streams,
            cache_cell,
            token_emb,
        );
        let write_gate = write_gate
            .reshape([batch.max(1), cache_streams.max(1), 1])
            .mean_dim(1)
            .reshape([batch.max(1), 1])
            .clamp_min(0.0)
            .clamp_max(1.0);
        let write_mask = if matches!(training.policy.write_gate_mode, crate::config::SudokuWriteGateMode::Bernoulli)
            && config.sample_policy
        {
            Tensor::<B, 2>::random(
                [batch.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                device,
            )
            .sub(write_gate.clone())
            .lower_equal_elem(0.0)
            .float()
        } else {
            write_gate.clone().greater_equal_elem(0.5).float()
        };
        let update_emb = update_emb
            .reshape([batch.max(1), cache_streams, 1, embd])
            .expand([batch.max(1), cache_streams, GRID_LEN, embd]);
        let update_mask_stream = update_mask_f.clone().unsqueeze_dim::<4>(1);
        let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_stream);
        cache = mhc_passthrough(model.cache_mhc.as_ref(), cache);
        let write_gate_data = write_gate
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("write gate to vec: {err:?}"))?;
        let write_mask_data = write_mask
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .map_err(|err| anyhow!("write mask to vec: {err:?}"))?;
        let pred_data = pred_values
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .map_err(|err| anyhow!("preds to vec: {err:?}"))?;

        for (sample_idx, grid) in puzzles.iter_mut().enumerate() {
            let mut focus = None;
            let selectable = selectable_counts
                .get(sample_idx)
                .copied()
                .unwrap_or(0.0)
                > 0.0;
            let solved = solutions
                .get(sample_idx)
                .is_some_and(|sol| is_grid_solved(grid, sol));
            if selectable && !solved {
                let action = action_data
                    .get(sample_idx)
                    .copied()
                    .unwrap_or(0)
                    .clamp(0, (GRID_LEN - 1) as i64) as usize;
                let editable = editable_masks
                    .get(sample_idx)
                    .and_then(|mask| mask.get(action))
                    .copied()
                    .unwrap_or(0.0);
                let write_mask = write_mask_data
                    .get(sample_idx)
                    .copied()
                    .unwrap_or(0.0);
                if editable > 0.5 && write_mask > 0.5 {
                    let pred = pred_data
                        .get(sample_idx)
                        .copied()
                        .unwrap_or(0)
                        .clamp(0, (VOCAB_SIZE - 1) as i64) as u8;
                    grid[action] = pred;
                    if let Some(mask) = unknown_masks.get_mut(sample_idx)
                        && let Some(entry) = mask.get_mut(action)
                    {
                        *entry = 0.0;
                    }
                }
                focus = Some(action);
            }
            let clue_mask = clue_masks.get(sample_idx).map(|mask| mask.as_slice());
            let write_prob = write_gate_data.get(sample_idx).copied();
            frames[sample_idx].push(render_sudoku_frame(
                grid,
                focus,
                clue_mask,
                write_prob,
            ));
        }
    }

    Ok(frames)
}
fn build_select_mask(
    unknown_masks: &[Vec<f32>],
    editable_masks: &[Vec<f32>],
    editable_counts: &[f32],
    revisit_min_filled: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut select_mask = Vec::new();
    let mut selectable_counts = Vec::new();
    for (idx, unknown) in unknown_masks.iter().enumerate() {
        let editable = editable_masks.get(idx).unwrap_or(unknown);
        let editable_count = editable_counts.get(idx).copied().unwrap_or(1.0).max(1.0);
        let unknown_count = unknown.iter().copied().sum::<f32>();
        let filled_frac = (editable_count - unknown_count) / editable_count;
        let allow_revisit = filled_frac >= revisit_min_filled;
        let mut mask = Vec::with_capacity(GRID_LEN);
        for (&unknown_cell, &editable_cell) in unknown.iter().zip(editable.iter()) {
            let clue = 1.0 - editable_cell;
            let allowed = if allow_revisit {
                editable_cell + clue
            } else {
                unknown_cell + clue
            };
            mask.push(allowed.min(1.0));
        }
        selectable_counts.push(mask.iter().copied().sum::<f32>());
        select_mask.extend(mask.into_iter());
    }
    (select_mask, selectable_counts)
}

fn build_action_index<B: BackendTrait>(batch: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let batch = batch.max(1);
    Tensor::<B, 1, Int>::arange(0..GRID_LEN as i64, device)
        .unsqueeze_dim::<2>(0)
        .expand([batch, GRID_LEN])
}

fn build_action_one_hot<B: BackendTrait>(
    actions: &Tensor<B, 2, Int>,
    action_index: &Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, grid] = action_index.shape().dims::<2>();
    let expanded = actions.clone().expand([batch, grid]);
    expanded.equal(action_index.clone()).float()
}

fn is_grid_solved(grid: &[u8], solution: &[u8]) -> bool {
    if grid.len() < GRID_LEN || solution.len() < GRID_LEN {
        return false;
    }
    grid.iter()
        .take(GRID_LEN)
        .zip(solution.iter().take(GRID_LEN))
        .all(|(a, b)| a == b)
}

fn build_tokens_tensor<B: BackendTrait>(grids: &[Vec<u8>], device: &B::Device) -> Tensor<B, 2, Int> {
    let batch = grids.len().max(1);
    let mut data = Vec::with_capacity(batch * GRID_LEN);
    for grid in grids {
        for &value in grid.iter().take(GRID_LEN) {
            data.push(value as i64);
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(data, [batch, GRID_LEN]), device)
}

fn render_sudoku_frame(
    grid: &[u8],
    focus: Option<usize>,
    clue_mask: Option<&[u8]>,
    write_prob: Option<f32>,
) -> ArtifactFrame {
    let width = GRID_PAD * 2 + CELL_SIZE * GRID_SIZE;
    let height = GRID_PAD * 2 + CELL_SIZE * GRID_SIZE;
    let mut rgb = vec![0u8; width * height * 3];
    fill_rect(&mut rgb, width, 0, 0, width, height, COLOR_BG);
    if let Some(prob) = write_prob {
        let prob = prob.clamp(0.0, 1.0);
        let bar_w = CELL_SIZE * GRID_SIZE;
        let bar_x = GRID_PAD;
        let bar_y = GRID_PAD.saturating_sub(WRITE_BAR_HEIGHT + 2);
        fill_rect(&mut rgb, width, bar_x, bar_y, bar_w, WRITE_BAR_HEIGHT, COLOR_WRITE_BG);
        let fill_w = ((bar_w as f32) * prob).round() as usize;
        if fill_w > 0 {
            fill_rect(
                &mut rgb,
                width,
                bar_x,
                bar_y,
                fill_w.min(bar_w),
                WRITE_BAR_HEIGHT,
                COLOR_WRITE_FILL,
            );
        }
        let percent = (prob * 100.0).round().clamp(0.0, 100.0) as usize;
        let digits = if percent >= 100 {
            3
        } else if percent >= 10 {
            2
        } else {
            1
        };
        let glyph_w = 5 * HUD_SCALE;
        let glyph_h = 7 * HUD_SCALE;
        let total_w = digits * glyph_w + digits.saturating_sub(1) * HUD_SCALE;
        let text_x = GRID_PAD + CELL_SIZE * GRID_SIZE - total_w;
        let text_y = GRID_PAD + CELL_SIZE * GRID_SIZE + (GRID_PAD - glyph_h) / 2;
        draw_number_at(
            &mut rgb,
            width,
            text_x,
            text_y,
            percent,
            HUD_SCALE,
            COLOR_LINE,
        );
    }

    if let Some(mask) = clue_mask {
        for row in 0..GRID_SIZE {
            for col in 0..GRID_SIZE {
                let idx = row * GRID_SIZE + col;
                if mask.get(idx).copied().unwrap_or(0) > 0 {
                    let cell_x = GRID_PAD + col * CELL_SIZE;
                    let cell_y = GRID_PAD + row * CELL_SIZE;
                    fill_rect(&mut rgb, width, cell_x, cell_y, CELL_SIZE, CELL_SIZE, COLOR_CLUE_BG);
                }
            }
        }
    }

    draw_grid_lines(&mut rgb, width, height);

    for row in 0..GRID_SIZE {
        for col in 0..GRID_SIZE {
            let idx = row * GRID_SIZE + col;
            let value = grid.get(idx).copied().unwrap_or(0);
            if value > 0 {
                draw_digit(
                    &mut rgb,
                    width,
                    row,
                    col,
                    value as usize,
                    COLOR_DIGIT,
                );
            }
        }
    }

    if let Some(focus_idx) = focus {
        let row = focus_idx / GRID_SIZE;
        let col = focus_idx % GRID_SIZE;
        highlight_cell(&mut rgb, width, row, col);
    }

    ArtifactFrame { width, height, rgb }
}

fn draw_grid_lines(rgb: &mut [u8], width: usize, height: usize) {
    for i in 0..=GRID_SIZE {
        let thickness = if i % 3 == 0 { LINE_THICK } else { LINE_THIN };
        let x = GRID_PAD + i * CELL_SIZE;
        fill_rect(
            rgb,
            width,
            x,
            GRID_PAD,
            thickness,
            CELL_SIZE * GRID_SIZE,
            COLOR_LINE,
        );
        let y = GRID_PAD + i * CELL_SIZE;
        fill_rect(
            rgb,
            width,
            GRID_PAD,
            y,
            CELL_SIZE * GRID_SIZE,
            thickness,
            COLOR_LINE,
        );
    }

    let _ = height;
}

fn draw_digit(
    rgb: &mut [u8],
    width: usize,
    row: usize,
    col: usize,
    digit: usize,
    color: [u8; 3],
) {
    if digit == 0 || digit > 9 {
        return;
    }
    let glyph = DIGITS[digit];
    let cell_x = GRID_PAD + col * CELL_SIZE;
    let cell_y = GRID_PAD + row * CELL_SIZE;
    let glyph_w = 5 * DIGIT_SCALE;
    let glyph_h = 7 * DIGIT_SCALE;
    let offset_x = cell_x + (CELL_SIZE - glyph_w) / 2;
    let offset_y = cell_y + (CELL_SIZE - glyph_h) / 2;

    for (gy, row_bits) in glyph.iter().enumerate() {
        for (gx, bit) in row_bits.as_bytes().iter().enumerate() {
            if *bit == b'1' {
                let x = offset_x + gx * DIGIT_SCALE + DIGIT_PAD / 2;
                let y = offset_y + gy * DIGIT_SCALE + DIGIT_PAD / 2;
                fill_rect(rgb, width, x, y, DIGIT_SCALE, DIGIT_SCALE, color);
            }
        }
    }
}

fn draw_number_at(
    rgb: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    value: usize,
    scale: usize,
    color: [u8; 3],
) {
    let value = value.min(100);
    let digits: Vec<usize> = if value >= 100 {
        vec![1, 0, 0]
    } else if value >= 10 {
        vec![value / 10, value % 10]
    } else {
        vec![value]
    };
    let mut cursor_x = x;
    for (idx, digit) in digits.iter().enumerate() {
        if idx > 0 {
            cursor_x = cursor_x.saturating_add(scale);
        }
        draw_digit_at(rgb, width, cursor_x, y, *digit, scale, color);
        cursor_x = cursor_x.saturating_add(5 * scale);
    }
}

fn draw_digit_at(
    rgb: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    digit: usize,
    scale: usize,
    color: [u8; 3],
) {
    if digit > 9 || scale == 0 {
        return;
    }
    let glyph = DIGITS[digit];
    for (gy, row_bits) in glyph.iter().enumerate() {
        for (gx, bit) in row_bits.as_bytes().iter().enumerate() {
            if *bit == b'1' {
                let px = x + gx * scale;
                let py = y + gy * scale;
                fill_rect(rgb, width, px, py, scale, scale, color);
            }
        }
    }
}

fn highlight_cell(rgb: &mut [u8], width: usize, row: usize, col: usize) {
    let cell_x = GRID_PAD + col * CELL_SIZE;
    let cell_y = GRID_PAD + row * CELL_SIZE;
    let border = LINE_THICK.max(2);
    fill_rect(rgb, width, cell_x, cell_y, CELL_SIZE, border, COLOR_HIGHLIGHT);
    fill_rect(
        rgb,
        width,
        cell_x,
        cell_y + CELL_SIZE - border,
        CELL_SIZE,
        border,
        COLOR_HIGHLIGHT,
    );
    fill_rect(rgb, width, cell_x, cell_y, border, CELL_SIZE, COLOR_HIGHLIGHT);
    fill_rect(
        rgb,
        width,
        cell_x + CELL_SIZE - border,
        cell_y,
        border,
        CELL_SIZE,
        COLOR_HIGHLIGHT,
    );
}

fn fill_rect(
    rgb: &mut [u8],
    width: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: [u8; 3],
) {
    for yy in y..(y + h) {
        for xx in x..(x + w) {
            let idx = (yy * width + xx) * 3;
            if idx + 2 < rgb.len() {
                rgb[idx] = color[0];
                rgb[idx + 1] = color[1];
                rgb[idx + 2] = color[2];
            }
        }
    }
}

const DIGITS: [&[&str; 7]; 10] = [
    &[
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
        "00000",
    ],
    &[
        "00100",
        "01100",
        "00100",
        "00100",
        "00100",
        "00100",
        "01110",
    ],
    &[
        "01110",
        "10001",
        "00001",
        "00010",
        "00100",
        "01000",
        "11111",
    ],
    &[
        "11110",
        "00001",
        "00001",
        "01110",
        "00001",
        "00001",
        "11110",
    ],
    &[
        "00010",
        "00110",
        "01010",
        "10010",
        "11111",
        "00010",
        "00010",
    ],
    &[
        "11111",
        "10000",
        "11110",
        "00001",
        "00001",
        "10001",
        "01110",
    ],
    &[
        "00110",
        "01000",
        "10000",
        "11110",
        "10001",
        "10001",
        "01110",
    ],
    &[
        "11111",
        "00001",
        "00010",
        "00100",
        "01000",
        "01000",
        "01000",
    ],
    &[
        "01110",
        "10001",
        "10001",
        "01110",
        "10001",
        "10001",
        "01110",
    ],
    &[
        "01110",
        "10001",
        "10001",
        "01111",
        "00001",
        "00010",
        "01100",
    ],
];













