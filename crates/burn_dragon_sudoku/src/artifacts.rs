use std::path::Path;

use anyhow::{Result, anyhow};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Int, Tensor, TensorData};

use burn_dragon_train::VisionArtifactOutputMode;
use burn_dragon_train::train::artifacts::{ArtifactFrame, write_video};

use crate::config::{SudokuArtifactConfig, SudokuTrainingHyperparameters};
use crate::dataset::{SudokuBatch, SudokuDataset, SudokuSplit};
use crate::model::SudokuSaccadeModel;
use crate::vocab::{GRID_LEN, VOCAB_SIZE};

const GRID_SIZE: usize = 9;
const CELL_SIZE: usize = 32;
const GRID_PAD: usize = 16;
const LINE_THIN: usize = 1;
const LINE_THICK: usize = 3;
const DIGIT_SCALE: usize = 4;
const DIGIT_PAD: usize = 4;

const COLOR_BG: [u8; 3] = [245, 242, 235];
const COLOR_LINE: [u8; 3] = [30, 30, 30];
const COLOR_DIGIT: [u8; 3] = [25, 25, 25];
const COLOR_HIGHLIGHT: [u8; 3] = [255, 187, 0];
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

    let rollout_steps = if training.rollout_max_steps > 0 {
        training.rollout_max_steps
    } else {
        training.rollout_steps.max(1)
    };
    let frames = generate_rollout_frames(
        model,
        puzzle_grids,
        solution_grids,
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
    for grid in puzzles.iter() {
        let mut unknown = Vec::with_capacity(GRID_LEN);
        let mut editable = Vec::with_capacity(GRID_LEN);
        for &value in grid.iter().take(GRID_LEN) {
            let clue = value > 0;
            editable.push(if clue { 0.0 } else { 1.0 });
            unknown.push(if value == 0 { 1.0 } else { 0.0 });
        }
        let editable_count = editable.iter().copied().sum::<f32>().max(1.0);
        unknown_masks.push(unknown);
        editable_masks.push(editable);
        editable_counts.push(editable_count);
    }

    let ones_grid = Tensor::<B, 2>::ones([batch.max(1), GRID_LEN], device);
    let ones_step = Tensor::<B, 2>::ones([batch.max(1), 1], device);
    let revisit_min_filled = training.revisit_min_filled_final.clamp(0.0, 1.0);
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
    let mut cache =
        model.cell_embeddings_with_positions(init_tokens.clone(), row_ids.clone(), col_ids.clone());
    let mut summary_tokens = model.init_summary_tokens(batch);
    let summary_len = model.summary_token_count();
    let mut state = model.init_state();

    for _step in 0..steps {
        let tokens = build_tokens_tensor::<B>(&puzzles, device);
        let policy_logits =
            model.policy_logits_from_cache(summary_tokens.clone(), cache.clone());

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
        let masked_logits = policy_logits
            - ones_grid
                .clone()
                .sub(select_mask.clone())
                .mul_scalar(POLICY_MASK_PENALTY);
        let actions = masked_logits.argmax(1);

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

        let step_mask = if training.saccade_step_cells > 1 {
            let selected_row = row_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch.max(1), 1]);
            let selected_col = col_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch.max(1), 1]);
            let row_match = row_ids_f
                .clone()
                .equal(selected_row.expand([batch.max(1), GRID_LEN]))
                .float();
            let col_match = col_ids_f
                .clone()
                .equal(selected_col.expand([batch.max(1), GRID_LEN]))
                .float();
            (row_match + col_match)
                .clamp_max(1.0)
                .mul(active_mask_grid.clone())
        } else {
            action_one_hot.clone()
        };
        let step_mask_sum = step_mask
            .clone()
            .sum_dim(1)
            .reshape([batch.max(1), 1])
            .clamp_min(1.0);
        let [_, _, embd] = cache.shape().dims();
        let step_emb = cache
            .clone()
            .mul(step_mask.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .div(step_mask_sum.clone().reshape([batch.max(1), 1, 1]))
            .reshape([batch.max(1), 1, embd]);
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_emb], 1);
        let (step_hidden, _step_logits_full) =
            model.forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = summary_hidden;
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
        let update_emb = model.cell_embeddings_with_positions(
            pred_values.clone(),
            selected_row,
            selected_col,
        );
        let update_emb = update_emb.expand([batch.max(1), GRID_LEN, embd]);
        let keep = update_mask_f.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_f);
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
                if editable > 0.5 {
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
            frames[sample_idx].push(render_sudoku_frame(grid, focus));
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

fn render_sudoku_frame(grid: &[u8], focus: Option<usize>) -> ArtifactFrame {
    let width = GRID_PAD * 2 + CELL_SIZE * GRID_SIZE;
    let height = GRID_PAD * 2 + CELL_SIZE * GRID_SIZE;
    let mut rgb = vec![0u8; width * height * 3];
    fill_rect(&mut rgb, width, 0, 0, width, height, COLOR_BG);

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
