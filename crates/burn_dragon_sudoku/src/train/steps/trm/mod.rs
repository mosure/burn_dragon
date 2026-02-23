use super::*;

mod chunk;
mod constraint_ca;

pub(super) use chunk::rollout_losses_train_trm_chunk;
pub(super) use constraint_ca::{
    rollout_losses_train_trm_constraint_ca, rollout_losses_valid_trm_chunk,
    rollout_losses_valid_trm_constraint_ca,
};

pub(super) fn use_trm_chunk(training: &SudokuTrainingHyperparameters) -> bool {
    matches!(training.rollout.traversal, SudokuTraversal::L2rT2b)
        && matches!(training.rollout.trm_mode, SudokuTrmMode::Chunk)
}

pub(super) fn use_trm_constraint_ca(training: &SudokuTrainingHyperparameters) -> bool {
    matches!(training.rollout.traversal, SudokuTraversal::L2rT2b)
        && matches!(training.rollout.trm_mode, SudokuTrmMode::ConstraintCa)
}

fn slice_grid_2d<B: BackendTrait>(tensor: &Tensor<B, 2>, start: usize, len: usize) -> Tensor<B, 2> {
    if len == 0 {
        return tensor.clone().slice_dim(1, 0..0);
    }
    let end = start + len;
    if end <= GRID_LEN {
        tensor.clone().slice_dim(1, start..end)
    } else {
        let first = tensor.clone().slice_dim(1, start..GRID_LEN);
        let second = tensor.clone().slice_dim(1, 0..(end - GRID_LEN));
        Tensor::cat(vec![first, second], 1)
    }
}

fn slice_grid_2d_int<B: BackendTrait>(
    tensor: &Tensor<B, 2, Int>,
    start: usize,
    len: usize,
) -> Tensor<B, 2, Int> {
    if len == 0 {
        return tensor.clone().slice_dim(1, 0..0);
    }
    let end = start + len;
    if end <= GRID_LEN {
        tensor.clone().slice_dim(1, start..end)
    } else {
        let first = tensor.clone().slice_dim(1, start..GRID_LEN);
        let second = tensor.clone().slice_dim(1, 0..(end - GRID_LEN));
        Tensor::cat(vec![first, second], 1)
    }
}

fn slice_grid_3d<B: BackendTrait>(tensor: &Tensor<B, 3>, start: usize, len: usize) -> Tensor<B, 3> {
    if len == 0 {
        return tensor.clone().slice_dim(1, 0..0);
    }
    let end = start + len;
    if end <= GRID_LEN {
        tensor.clone().slice_dim(1, start..end)
    } else {
        let first = tensor.clone().slice_dim(1, start..GRID_LEN);
        let second = tensor.clone().slice_dim(1, 0..(end - GRID_LEN));
        Tensor::cat(vec![first, second], 1)
    }
}

fn slice_grid_4d<B: BackendTrait>(tensor: &Tensor<B, 4>, start: usize, len: usize) -> Tensor<B, 4> {
    if len == 0 {
        return tensor.clone().slice_dim(2, 0..0);
    }
    let end = start + len;
    if end <= GRID_LEN {
        tensor.clone().slice_dim(2, start..end)
    } else {
        let first = tensor.clone().slice_dim(2, start..GRID_LEN);
        let second = tensor.clone().slice_dim(2, 0..(end - GRID_LEN));
        Tensor::cat(vec![first, second], 2)
    }
}

fn replace_grid_2d<B: BackendTrait>(
    tensor: &Tensor<B, 2>,
    start: usize,
    segment: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [_batch, len] = segment.shape().dims::<2>();
    if len == 0 {
        return tensor.clone();
    }
    let end = start + len;
    if end <= GRID_LEN {
        let mut parts: Vec<Tensor<B, 2>> = Vec::new();
        if start > 0 {
            parts.push(tensor.clone().slice_dim(1, 0..start));
        }
        parts.push(segment);
        if end < GRID_LEN {
            parts.push(tensor.clone().slice_dim(1, end..GRID_LEN));
        }
        return Tensor::cat(parts, 1);
    }
    let first_len = GRID_LEN - start;
    let seg_first = segment.clone().slice_dim(1, 0..first_len);
    let seg_second = segment.slice_dim(1, first_len..len);
    let temp = replace_grid_2d(tensor, start, seg_first);
    replace_grid_2d(&temp, 0, seg_second)
}

fn replace_grid_2d_int<B: BackendTrait>(
    tensor: &Tensor<B, 2, Int>,
    start: usize,
    segment: Tensor<B, 2, Int>,
) -> Tensor<B, 2, Int> {
    let [_batch, len] = segment.shape().dims::<2>();
    if len == 0 {
        return tensor.clone();
    }
    let end = start + len;
    if end <= GRID_LEN {
        let mut parts: Vec<Tensor<B, 2, Int>> = Vec::new();
        if start > 0 {
            parts.push(tensor.clone().slice_dim(1, 0..start));
        }
        parts.push(segment);
        if end < GRID_LEN {
            parts.push(tensor.clone().slice_dim(1, end..GRID_LEN));
        }
        return Tensor::cat(parts, 1);
    }
    let first_len = GRID_LEN - start;
    let seg_first = segment.clone().slice_dim(1, 0..first_len);
    let seg_second = segment.slice_dim(1, first_len..len);
    let temp = replace_grid_2d_int(tensor, start, seg_first);
    replace_grid_2d_int(&temp, 0, seg_second)
}

fn replace_grid_4d<B: BackendTrait>(
    tensor: &Tensor<B, 4>,
    start: usize,
    segment: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_batch, _streams, len, _dim] = segment.shape().dims::<4>();
    if len == 0 {
        return tensor.clone();
    }
    let end = start + len;
    if end <= GRID_LEN {
        let mut parts: Vec<Tensor<B, 4>> = Vec::new();
        if start > 0 {
            parts.push(tensor.clone().slice_dim(2, 0..start));
        }
        parts.push(segment);
        if end < GRID_LEN {
            parts.push(tensor.clone().slice_dim(2, end..GRID_LEN));
        }
        return Tensor::cat(parts, 2);
    }
    let first_len = GRID_LEN - start;
    let seg_first = segment.clone().slice_dim(2, 0..first_len);
    let seg_second = segment.slice_dim(2, first_len..len);
    let temp = replace_grid_4d(tensor, start, seg_first);
    replace_grid_4d(&temp, 0, seg_second)
}
