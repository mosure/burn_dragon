use super::*;

pub(super) fn compute_grid_accuracy<B: BackendTrait>(
    tokens: &Tensor<B, 2, Int>,
    solutions: &Tensor<B, 2, Int>,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let [batch, time] = tokens.shape().dims();
    let device = tokens.device();
    if batch == 0 || time == 0 {
        let zeros = Tensor::<B, 1>::zeros([1], &device);
        return (zeros.clone(), zeros.clone(), zeros.clone(), zeros);
    }

    let correct = tokens.clone().equal(solutions.clone()).float();
    let acc_per_sample = correct
        .clone()
        .sum_dim(1)
        .reshape([batch])
        .div_scalar(time.max(1) as f32);
    let acc = acc_per_sample.clone().mean();

    let exact_per_sample = correct
        .sum_dim(1)
        .reshape([batch])
        .equal_elem(time as f32)
        .float();
    let exact_acc = exact_per_sample.clone().mean();
    let solve_rate = exact_acc.clone();

    (acc_per_sample, acc, exact_acc, solve_rate)
}

pub(super) fn compute_grid_accuracy_masked<B: BackendTrait>(
    tokens: &Tensor<B, 2, Int>,
    solutions: &Tensor<B, 2, Int>,
    mask: &Tensor<B, 2>,
) -> (Tensor<B, 1>, Tensor<B, 1>) {
    let [batch, time] = tokens.shape().dims();
    let device = tokens.device();
    if batch == 0 || time == 0 {
        let zeros = Tensor::<B, 1>::zeros([1], &device);
        return (zeros.clone(), zeros);
    }

    let correct = tokens.clone().equal(solutions.clone()).float();
    let mask_sum = mask.clone().sum_dim(1).reshape([batch]).clamp_min(1.0);
    let acc_per_sample = correct
        .mul(mask.clone())
        .sum_dim(1)
        .reshape([batch])
        .div(mask_sum);
    let acc = acc_per_sample.clone().mean();

    (acc_per_sample, acc)
}

#[allow(clippy::type_complexity)]
pub(super) fn compute_loss_and_acc<B: BackendTrait>(
    logits: &Tensor<B, 3>,
    solutions: &Tensor<B, 2, Int>,
    solution_one_hot: &Tensor<B, 3>,
    loss_mask: &Tensor<B, 2>,
    recon_loss: &SudokuReconLoss,
) -> (
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
    Tensor<B, 1>,
) {
    let [batch, time, _] = logits.shape().dims();
    if batch == 0 || time == 0 {
        let device = logits.device();
        let zeros = Tensor::<B, 1>::zeros([1], &device);
        return (
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            zeros,
        );
    }

    let log_probs = match recon_loss {
        SudokuReconLoss::Softmax => activation::log_softmax(logits.clone(), 2),
        SudokuReconLoss::Stablemax => stablemax_log_probs(logits.clone()),
    };
    let target_log_prob = log_probs
        .mul(solution_one_hot.clone())
        .sum_dim(2)
        .reshape([batch, time]);
    let nll = target_log_prob.mul_scalar(-1.0);
    let mask_sum = loss_mask
        .clone()
        .sum_dim(1)
        .reshape([batch])
        .add_scalar(SUDOKU_EPS);
    let masked_nll = nll.clone().mul(loss_mask.clone());
    let loss_per_sample = masked_nll.sum_dim(1).reshape([batch]) / mask_sum.clone();
    let loss = loss_per_sample.clone().mean();

    let preds = logits.clone().argmax(2).reshape([batch, time]);
    let correct = preds.equal(solutions.clone()).float();
    let acc_per_sample = correct
        .clone()
        .sum_dim(1)
        .reshape([batch])
        .div_scalar(time.max(1) as f32);
    let acc = acc_per_sample.clone().mean();

    let exact_per_sample = correct
        .sum_dim(1)
        .reshape([batch])
        .equal_elem(time as f32)
        .float();
    let exact_acc = exact_per_sample.clone().mean();

    (
        loss,
        acc,
        exact_acc,
        loss_per_sample,
        acc_per_sample,
        exact_per_sample,
    )
}

fn stablemax_log_probs<B: BackendTrait>(logits: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, time, vocab] = logits.shape().dims();
    if batch == 0 || time == 0 || vocab == 0 {
        return logits;
    }
    let device = logits.device();
    let shape = [batch, time, vocab];
    let ones = Tensor::<B, 3>::ones(shape, &device);
    let neg_mask = logits.clone().lower_elem(0.0);
    let pos = logits.clone().add_scalar(1.0);
    let denom = ones.clone().sub(logits).add_scalar(STABLEMAX_EPS);
    let neg = ones.div(denom);
    let s = pos.mask_where(neg_mask, neg);
    let sum = s
        .clone()
        .sum_dim(2)
        .reshape([batch, time, 1])
        .clamp_min(STABLEMAX_EPS);
    s.div(sum).clamp_min(STABLEMAX_EPS).log()
}

pub(super) fn halt_bce_loss<B: BackendTrait>(
    logits: Tensor<B, 2>,
    targets: Tensor<B, 2>,
) -> Tensor<B, 1> {
    let device = logits.device();
    let [batch, time] = logits.shape().dims();
    let probs = activation::sigmoid(logits)
        .clamp_min(HALT_EPS)
        .clamp_max(1.0 - HALT_EPS);
    let ones = Tensor::<B, 2>::ones([batch.max(1), time.max(1)], &device);
    let log_prob = probs.clone().log();
    let log_not = (ones.clone() - probs).clamp_min(HALT_EPS).log();
    let loss = targets.clone() * log_prob + (ones - targets) * log_not;
    loss.mul_scalar(-1.0).mean()
}

pub(crate) fn sample_actions<B: BackendTrait>(
    logits: Tensor<B, 2>,
    select_mask: Tensor<B, 2>,
    train_mode: bool,
    policy_epsilon: f32,
) -> Tensor<B, 2, Int> {
    if !train_mode {
        return logits.argmax(1);
    }

    let device = logits.device();
    let [batch, time] = logits.shape().dims();
    if batch == 0 || time == 0 {
        return Tensor::<B, 2, Int>::zeros([batch.max(1), 1], &device);
    }

    let policy_epsilon = policy_epsilon.clamp(0.0, 1.0);
    let masked_logits = logits
        + select_mask
            .clone()
            .sub_scalar(1.0)
            .mul_scalar(POLICY_MASK_PENALTY);

    let uniform = Tensor::<B, 2>::random(
        [batch, time],
        TensorDistribution::Uniform(0.0, 1.0),
        &device,
    )
    .clamp_min(GUMBEL_EPS)
    .clamp_max(1.0 - GUMBEL_EPS);
    let gumbel = uniform
        .clone()
        .log()
        .mul_scalar(-1.0)
        .log()
        .mul_scalar(-1.0);
    let gumbel_logits = masked_logits + gumbel;
    let gumbel_actions = gumbel_logits.argmax(1);

    if policy_epsilon <= 0.0 {
        return gumbel_actions;
    }

    let random_scores = Tensor::<B, 2>::random(
        [batch, time],
        TensorDistribution::Uniform(0.0, 1.0),
        &device,
    );
    let random_scores =
        random_scores * select_mask.clone() + select_mask.clone().sub_scalar(1.0).mul_scalar(2.0);
    let random_actions = random_scores.argmax(1);
    let gate = Tensor::<B, 2>::random([batch, 1], TensorDistribution::Uniform(0.0, 1.0), &device)
        .lower_equal_elem(policy_epsilon);

    gumbel_actions.mask_where(gate, random_actions)
}

pub(super) fn build_solution_one_hot<B: BackendTrait>(
    solutions: &Tensor<B, 2, Int>,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let batch = batch.max(1);
    let vocab = Tensor::<B, 1, Int>::arange(0..VOCAB_SIZE as i64, device)
        .unsqueeze_dim::<2>(0)
        .unsqueeze_dim::<3>(0)
        .expand([batch, GRID_LEN, VOCAB_SIZE]);
    let solutions = solutions
        .clone()
        .unsqueeze_dim::<3>(2)
        .expand([batch, GRID_LEN, VOCAB_SIZE]);
    solutions.equal(vocab).float()
}

pub(super) fn build_action_one_hot<B: BackendTrait>(
    actions: &Tensor<B, 2, Int>,
    action_index: &Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, grid] = action_index.shape().dims::<2>();
    let expanded = actions.clone().expand([batch, grid]);
    expanded.equal(action_index.clone()).float()
}
