use crate::train::metrics::{SudokuOutput, SudokuTrainItem};
use crate::train::prelude::*;
use crate::vocab::{GRID_LEN, VOCAB_SIZE};
use burn_dragon_train::train::gdpo::{gdpo_advantage_autodiff, gdpo_policy_loss};

const SUDOKU_EPS: f32 = 1e-6;
const POLICY_MASK_PENALTY: f32 = 1e9;
const HALT_EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct SudokuTrainer<B: BackendTrait> {
    pub model: SudokuSaccadeModel<B>,
    pub training: SudokuTrainingHyperparameters,
}

impl<B: BackendTrait> SudokuTrainer<B> {
    pub fn new(model: SudokuSaccadeModel<B>, training: SudokuTrainingHyperparameters) -> Self {
        Self { model, training }
    }
}

impl<B: BackendTrait> std::fmt::Display for SudokuTrainer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SudokuTrainer")
    }
}

impl<B: BackendTrait> Module<B> for SudokuTrainer<B> {
    type Record = <SudokuSaccadeModel<B> as Module<B>>::Record;

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.model.collect_devices(devices)
    }

    fn fork(self, device: &B::Device) -> Self {
        Self {
            model: self.model.fork(device),
            training: self.training,
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: self.model.to_device(device),
            training: self.training,
        }
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: self.model.map(mapper),
            training: self.training,
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: self.model.load_record(record),
            training: self.training,
        }
    }

    fn into_record(self) -> Self::Record {
        self.model.into_record()
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for SudokuTrainer<B> {
    type InnerModule = SudokuTrainer<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        SudokuTrainer {
            model: self.model.valid(),
            training: self.training.clone(),
        }
    }
}

impl<B: AutodiffBackend> TrainStep<SudokuBatch<B>, SudokuTrainItem<B>> for SudokuTrainer<B> {
    fn step(&self, batch: SudokuBatch<B>) -> TrainOutput<SudokuTrainItem<B>> {
        let losses = rollout_losses::<B>(
            &self.model,
            batch,
            &self.training,
            self.training.policy_noise,
            self.training.gdpo.enabled,
        );

        let grads = losses.loss.backward();
        let item = SudokuTrainItem::new(
            losses.loss,
            losses.recon_loss,
            losses.acc,
            losses.policy_loss,
            losses.halt_loss,
            losses.halt_prob_mean,
            losses.halt_target_mean,
            losses.advantage_abs_mean,
            losses.advantage_std,
            losses.log_prob_mean,
            losses.policy_entropy,
            losses.hard_reward_mean,
            losses.easy_reward_mean,
        );
        TrainOutput::new(self, grads, item)
    }
}

impl<B: BackendTrait> ValidStep<SudokuBatch<B>, SudokuOutput<B>> for SudokuTrainer<B> {
    fn step(&self, batch: SudokuBatch<B>) -> SudokuOutput<B> {
        let losses = rollout_losses_valid::<B>(&self.model, batch, &self.training);
        SudokuOutput::new(
            losses.loss,
            losses.recon_loss,
            losses.acc,
            losses.policy_loss,
            losses.halt_loss,
            losses.halt_prob_mean,
            losses.halt_target_mean,
            losses.advantage_abs_mean,
            losses.advantage_std,
            losses.log_prob_mean,
            losses.policy_entropy,
            losses.hard_reward_mean,
            losses.easy_reward_mean,
        )
    }
}

pub struct SudokuLosses<B: BackendTrait> {
    pub loss: Tensor<B, 1>,
    pub recon_loss: Tensor<B, 1>,
    pub acc: Tensor<B, 1>,
    pub policy_loss: Tensor<B, 1>,
    pub halt_loss: Tensor<B, 1>,
    pub halt_prob_mean: Tensor<B, 1>,
    pub halt_target_mean: Tensor<B, 1>,
    pub advantage_abs_mean: Tensor<B, 1>,
    pub advantage_std: Tensor<B, 1>,
    pub log_prob_mean: Tensor<B, 1>,
    pub policy_entropy: Tensor<B, 1>,
    pub hard_reward_mean: Tensor<B, 1>,
    pub easy_reward_mean: Tensor<B, 1>,
}

fn rollout_losses<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    gdpo_active: bool,
) -> SudokuLosses<B> {
    let gdpo_group = training.gdpo.group_size.max(1);
    let repeat_for_gdpo = gdpo_active && gdpo_group > 1;
    let rollout = rollout_base(
        model,
        batch,
        training.rollout_steps.max(1),
        policy_noise,
        repeat_for_gdpo.then_some(gdpo_group),
        gdpo_active,
    );

    let device = rollout.recon_loss.device();
    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let log_prob_sum = match rollout.log_prob_sum {
        Some(sum) => sum,
        None => Tensor::<B, 2>::zeros([rollout.batch.max(1), 1], &device),
    };

    let policy_entropy = if rollout.policy_steps > 0 {
        rollout
            .policy_entropy_sum
            .clone()
            .div_scalar(rollout.policy_steps as f32)
            .mean()
    } else {
        zeros.clone()
    };

    let log_prob_mean = if rollout.policy_steps > 0 && gdpo_active {
        log_prob_sum
            .clone()
            .div_scalar(rollout.policy_steps as f32)
            .mean()
    } else {
        zeros.clone()
    };

    let hard_reward_mean = rollout.hard_reward.clone().mean();
    let easy_reward_mean = rollout.easy_reward.clone().mean();
    let halt_loss = rollout.halt_loss.clone();
    let halt_prob_mean = rollout.halt_prob_mean.clone();
    let halt_target_mean = rollout.halt_target_mean.clone();

    let (policy_loss, advantage_abs_mean, advantage_std) = if gdpo_active {
        let scene_batch = if gdpo_group == 0 {
            rollout.batch
        } else {
            rollout.batch / gdpo_group
        };
        let hard = rollout
            .hard_reward
            .clone()
            .detach()
            .reshape([scene_batch.max(1), gdpo_group.max(1)]);
        let easy = rollout
            .easy_reward
            .clone()
            .detach()
            .reshape([scene_batch.max(1), gdpo_group.max(1)]);
        let advantage = gdpo_advantage_autodiff::<B>(hard, easy, &training.gdpo)
            .reshape([rollout.batch.max(1), 1])
            .detach();

        let log_prob_old = log_prob_sum.clone().detach();
        let policy_loss = gdpo_policy_loss(
            log_prob_sum.clone(),
            log_prob_old,
            advantage.clone(),
            &training.gdpo,
        );

        let adv_abs = advantage.clone().abs().mean();
        let adv_mean = advantage.clone().mean();
        let adv_sq_mean = advantage.clone().powf_scalar(2.0).mean();
        let adv_var = adv_sq_mean - adv_mean.clone().powf_scalar(2.0);
        let adv_std = adv_var.add_scalar(SUDOKU_EPS).sqrt();
        (policy_loss, adv_abs, adv_std)
    } else {
        (zeros.clone(), zeros.clone(), zeros.clone())
    };

    let loss = rollout.recon_loss.clone()
        + policy_loss.clone()
        + halt_loss.clone().mul_scalar(training.halt_weight);

    SudokuLosses {
        loss,
        recon_loss: rollout.recon_loss,
        acc: rollout.acc,
        policy_loss,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        advantage_abs_mean,
        advantage_std,
        log_prob_mean,
        policy_entropy,
        hard_reward_mean,
        easy_reward_mean,
    }
}

fn rollout_losses_valid<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
) -> SudokuLosses<B> {
    let rollout = rollout_base(
        model,
        batch,
        training.rollout_steps.max(1),
        0.0,
        None,
        false,
    );
    let device = rollout.recon_loss.device();
    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let loss = rollout
        .recon_loss
        .clone()
        .add(rollout.halt_loss.clone().mul_scalar(training.halt_weight));
    SudokuLosses {
        loss,
        recon_loss: rollout.recon_loss,
        acc: rollout.acc,
        policy_loss: zeros.clone(),
        halt_loss: rollout.halt_loss,
        halt_prob_mean: rollout.halt_prob_mean,
        halt_target_mean: rollout.halt_target_mean,
        advantage_abs_mean: zeros.clone(),
        advantage_std: zeros.clone(),
        log_prob_mean: zeros.clone(),
        policy_entropy: zeros.clone(),
        hard_reward_mean: rollout.hard_reward.mean(),
        easy_reward_mean: rollout.easy_reward.mean(),
    }
}

struct RolloutBase<B: BackendTrait> {
    recon_loss: Tensor<B, 1>,
    acc: Tensor<B, 1>,
    hard_reward: Tensor<B, 1>,
    easy_reward: Tensor<B, 1>,
    halt_loss: Tensor<B, 1>,
    halt_prob_mean: Tensor<B, 1>,
    halt_target_mean: Tensor<B, 1>,
    log_prob_sum: Option<Tensor<B, 2>>,
    policy_entropy_sum: Tensor<B, 2>,
    policy_steps: usize,
    batch: usize,
}

fn rollout_base<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    rollout_steps: usize,
    policy_noise: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
) -> RolloutBase<B> {
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
    let solution_one_hot = build_solution_one_hot(&solutions, batch_size, &device);

    let mut tokens = puzzles;
    let mut unknown_mask = tokens.clone().equal_elem(0).float();
    let mut seen_mask = tokens.clone().greater_equal_elem(1).float();

    let mut initial_acc = None;

    let mut log_prob_sum = if track_policy {
        Some(Tensor::<B, 2>::zeros([batch_size, 1], &device))
    } else {
        None
    };
    let mut policy_entropy_sum = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut halt_loss_sum = Tensor::<B, 1>::zeros([1], &device);
    let mut halt_prob_sum = Tensor::<B, 1>::zeros([1], &device);
    let mut halt_target_sum = Tensor::<B, 1>::zeros([1], &device);
    let mut halt_steps = 0usize;

    for step_idx in 0..rollout_steps {
        let (hidden, logits) = model.forward_with_hidden(tokens.clone());

        if step_idx == 0 {
            let (_loss, _acc, _loss_per_sample, acc_per_sample) =
                compute_loss_and_acc(&logits, &solutions, &solution_one_hot, &unknown_mask);
            initial_acc = Some(acc_per_sample);
        }

        let policy_logits = model.policy_logits(hidden.clone());
        let halt_logit = model.halt_logit(hidden);
        let masked_logits = policy_logits - seen_mask.clone().mul_scalar(POLICY_MASK_PENALTY);
        let unknown_counts = unknown_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let halt_target = unknown_counts.clone().equal_elem(0.0).float();
        let halt_prob = activation::sigmoid(halt_logit.clone());
        let step_halt_loss = halt_bce_loss(halt_logit, halt_target.clone());
        halt_loss_sum = halt_loss_sum + step_halt_loss;
        halt_prob_sum = halt_prob_sum + halt_prob.mean();
        halt_target_sum = halt_target_sum + halt_target.mean();
        halt_steps += 1;
        let sampled_logits = if policy_noise > 0.0 {
            let noise = Tensor::<B, 2>::random(
                [batch_size, GRID_LEN],
                TensorDistribution::Normal(0.0, f64::from(policy_noise)),
                &device,
            );
            masked_logits + noise
        } else {
            masked_logits
        };

        let actions = sampled_logits.clone().argmax(1);
        let mut action_one_hot = build_action_one_hot(&actions, batch_size, &device);
        let active_mask = unknown_counts.greater_elem(0.0).float();
        let active_mask_grid = active_mask.clone().repeat_dim(1, GRID_LEN);
        action_one_hot = action_one_hot * active_mask_grid;

        let log_probs = activation::log_softmax(sampled_logits, 1);
        let selected_log_prob = log_probs
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .mul(active_mask.clone());
        if let Some(sum) = log_prob_sum.as_mut() {
            *sum = sum.clone() + selected_log_prob.reshape([batch_size, 1]);
        }

        let step_entropy = log_probs
            .clone()
            .exp()
            .mul(log_probs)
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .mul_scalar(-1.0)
            .mul(active_mask.clone());
        policy_entropy_sum = policy_entropy_sum + step_entropy;

        let update_mask = action_one_hot.clone().greater_equal_elem(0.5);
        tokens = tokens.mask_where(update_mask, solutions.clone());
        seen_mask = (seen_mask + action_one_hot.clone()).clamp_max(1.0);
        unknown_mask = (unknown_mask - action_one_hot).clamp_min(0.0);
    }

    let (_hidden, logits) = model.forward_with_hidden(tokens);
    let (recon_loss, acc, loss_per_sample, acc_per_sample) =
        compute_loss_and_acc(&logits, &solutions, &solution_one_hot, &unknown_mask);

    let initial_acc = initial_acc.unwrap_or_else(|| Tensor::<B, 1>::zeros([batch_size], &device));
    let hard_reward = acc_per_sample.clone() - initial_acc;
    let easy_reward = loss_per_sample.clone().mul_scalar(-1.0);
    let halt_steps = halt_steps.max(1) as f32;
    let halt_loss = halt_loss_sum.div_scalar(halt_steps);
    let halt_prob_mean = halt_prob_sum.div_scalar(halt_steps);
    let halt_target_mean = halt_target_sum.div_scalar(halt_steps);

    RolloutBase {
        recon_loss,
        acc,
        hard_reward,
        easy_reward,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        log_prob_sum,
        policy_entropy_sum,
        policy_steps: rollout_steps,
        batch: batch_size,
    }
}

fn compute_loss_and_acc<B: BackendTrait>(
    logits: &Tensor<B, 3>,
    solutions: &Tensor<B, 2, Int>,
    solution_one_hot: &Tensor<B, 3>,
    mask: &Tensor<B, 2>,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let [batch, time, _] = logits.shape().dims();
    if batch == 0 || time == 0 {
        let device = logits.device();
        let zeros = Tensor::<B, 1>::zeros([1], &device);
        return (zeros.clone(), zeros.clone(), zeros.clone(), zeros);
    }

    let log_probs = activation::log_softmax(logits.clone(), 2);
    let target_log_prob = log_probs
        .mul(solution_one_hot.clone())
        .sum_dim(2)
        .reshape([batch, GRID_LEN]);
    let nll = target_log_prob.mul_scalar(-1.0);
    let mask_sum = mask
        .clone()
        .sum_dim(1)
        .reshape([batch])
        .add_scalar(SUDOKU_EPS);
    let masked_nll = nll.clone().mul(mask.clone());
    let loss_per_sample = masked_nll.sum_dim(1).reshape([batch]) / mask_sum.clone();
    let loss = loss_per_sample.clone().mean();

    let preds = logits.clone().argmax(2).reshape([batch, GRID_LEN]);
    let correct = preds.equal(solutions.clone()).float();
    let acc_per_sample =
        correct.mul(mask.clone()).sum_dim(1).reshape([batch]) / mask_sum;
    let acc = acc_per_sample.clone().mean();

    (loss, acc, loss_per_sample, acc_per_sample)
}

fn halt_bce_loss<B: BackendTrait>(logits: Tensor<B, 2>, targets: Tensor<B, 2>) -> Tensor<B, 1> {
    let device = logits.device();
    let [batch, time] = logits.shape().dims();
    let probs = activation::sigmoid(logits)
        .clamp_min(HALT_EPS)
        .clamp_max(1.0 - HALT_EPS);
    let ones = Tensor::<B, 2>::ones([batch.max(1), time.max(1)], &device);
    let log_prob = probs.clone().log();
    let log_not = (ones.clone() - probs)
        .clamp_min(HALT_EPS)
        .log();
    let loss = targets.clone() * log_prob + (ones - targets) * log_not;
    loss.mul_scalar(-1.0).mean()
}

fn build_solution_one_hot<B: BackendTrait>(
    solutions: &Tensor<B, 2, Int>,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 3> {
    let data = solutions
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("solutions to vec");
    let mut one_hot = vec![0f32; batch * GRID_LEN * VOCAB_SIZE];
    for b in 0..batch {
        for idx in 0..GRID_LEN {
            let value = data[b * GRID_LEN + idx] as usize;
            if value < VOCAB_SIZE {
                let offset = (b * GRID_LEN + idx) * VOCAB_SIZE + value;
                one_hot[offset] = 1.0;
            }
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(one_hot, [batch, GRID_LEN, VOCAB_SIZE]),
        device,
    )
}

fn build_action_one_hot<B: BackendTrait>(
    actions: &Tensor<B, 2, Int>,
    batch: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let data = actions
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("actions to vec");
    let action_dim = actions.shape().dims::<2>()[1].max(1);
    let mut one_hot = vec![0f32; batch * GRID_LEN];
    for b in 0..batch.min(data.len() / action_dim) {
        let action = data[b * action_dim] as usize;
        let idx = action.min(GRID_LEN.saturating_sub(1));
        one_hot[b * GRID_LEN + idx] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(one_hot, [batch, GRID_LEN]), device)
}
