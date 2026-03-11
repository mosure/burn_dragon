use crate::artifacts::write_validation_artifacts_from_batch;
use crate::train::metrics::{SudokuOutput, SudokuTrainItem};
use crate::train::prelude::*;
use crate::vocab::{GRID_LEN, VOCAB_SIZE};
use burn_dragon_core::{ModelState, mhc_passthrough};
use burn_dragon_train::train::gdpo::{gdpo_advantage_autodiff, gdpo_policy_loss};
use std::sync::Mutex;
use tracing::warn;

const SUDOKU_EPS: f32 = 1e-6;
const POLICY_MASK_PENALTY: f32 = 1e9;
const HALT_EPS: f32 = 1e-6;
const STABLEMAX_EPS: f32 = 1e-6;
const GUMBEL_EPS: f32 = 1e-6;
const GRID_SIDE: usize = 9;

fn resolve_write_gate<B: BackendTrait>(
    write_gate: Tensor<B, 2>,
    gate_mode: SudokuWriteGateMode,
    action_any: Tensor<B, 2>,
) -> (Tensor<B, 2>, Option<Tensor<B, 2>>, Option<Tensor<B, 2>>) {
    let [batch, _] = write_gate.shape().dims();
    let device = write_gate.device();
    let gate_mask = action_any.clone().clamp_max(1.0);
    match gate_mode {
        SudokuWriteGateMode::StraightThrough => {
            let write_mask = write_gate.clone().greater_equal_elem(0.5);
            let write_mask_hard = write_mask.clone().float();
            let write_mask_f = write_mask_hard
                .clone()
                .add(write_gate.clone().sub(write_gate.detach()));
            (write_mask_f, None, None)
        }
        SudokuWriteGateMode::Bernoulli => {
            let write_prob = write_gate
                .clone()
                .clamp_min(SUDOKU_EPS)
                .clamp_max(1.0 - SUDOKU_EPS);
            let write_mask_f = Tensor::<B, 2>::random(
                [batch.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .sub(write_prob.clone())
            .lower_equal_elem(0.0)
            .float();
            let ones = Tensor::<B, 2>::ones([batch.max(1), 1], &device);
            let log_prob = write_mask_f.clone().mul(write_prob.clone().log())
                + ones
                    .clone()
                    .sub(write_mask_f.clone())
                    .mul(ones.clone().sub(write_prob.clone()).log());
            let gate_log_prob = log_prob.mul(gate_mask.clone());
            let entropy = write_prob.clone().mul(write_prob.clone().log())
                + ones
                    .clone()
                    .sub(write_prob.clone())
                    .mul(ones.clone().sub(write_prob.clone()).log());
            let gate_entropy = entropy.mul_scalar(-1.0).mul(gate_mask);
            (write_mask_f, Some(gate_log_prob), Some(gate_entropy))
        }
    }
}

fn rollout_scheduled_steps(training: &SudokuTrainingHyperparameters, step: usize) -> usize {
    let mut steps = training.rollout.steps.max(1);
    if let Some(schedule) = &training.rollout.schedule {
        let start = schedule.start_steps.max(1);
        let end = schedule.final_steps.max(1);
        let anneal = schedule.anneal_iters;
        if anneal == 0 {
            steps = end;
        } else {
            steps = schedule_linear(start as f32, end as f32, anneal, step).round() as usize;
        }
    }
    steps.max(1)
}

fn rollout_max_steps(training: &SudokuTrainingHyperparameters, step: usize) -> usize {
    let base_steps = rollout_scheduled_steps(training, step);
    if training.rollout.max_steps > 0 {
        training.rollout.max_steps
    } else {
        base_steps
    }
}

fn rollout_min_steps(training: &SudokuTrainingHyperparameters, step: usize) -> usize {
    let max_steps = rollout_max_steps(training, step);
    let min_steps = if training.rollout.min_steps > 0 {
        training.rollout.min_steps
    } else {
        max_steps
    };
    min_steps.min(max_steps).max(1)
}

fn rollout_bounds(training: &SudokuTrainingHyperparameters, step: usize) -> (usize, usize) {
    let min_steps = rollout_min_steps(training, step);
    let mut max_steps = rollout_max_steps(training, step);
    if training.rollout.max_steps_warmup_iters > 0
        && step < training.rollout.max_steps_warmup_iters
        && training.rollout.max_steps_warmup_cap > 0
    {
        max_steps = max_steps.min(training.rollout.max_steps_warmup_cap);
    }
    max_steps = max_steps.max(1);
    let min_steps = min_steps.min(max_steps).max(1);
    (min_steps, max_steps)
}

fn sample_rollout_steps(training: &SudokuTrainingHyperparameters, step: usize) -> usize {
    let (min_steps, max_steps) = rollout_bounds(training, step);
    if min_steps >= max_steps {
        max_steps
    } else {
        thread_rng().gen_range(min_steps..=max_steps)
    }
}

fn sample_pre_rollout_steps(training: &SudokuTrainingHyperparameters) -> usize {
    let min_steps = training.rollout.pre_steps_min;
    let max_steps = training.rollout.pre_steps_max.max(min_steps);
    if max_steps == 0 {
        return 0;
    }
    if min_steps >= max_steps {
        max_steps
    } else {
        thread_rng().gen_range(min_steps..=max_steps)
    }
}

fn validation_rollout_steps(training: &SudokuTrainingHyperparameters) -> usize {
    match training.validation.rollout_steps {
        Some(steps) if steps > 0 => steps,
        _ => rollout_max_steps(training, 0),
    }
}

fn write_gate_floor_for_step(training: &SudokuTrainingHyperparameters, step: usize) -> f32 {
    if training.policy.write_gate_warmup_steps == 0 {
        0.0
    } else {
        schedule_linear(
            training.policy.write_gate_warmup_floor,
            0.0,
            training.policy.write_gate_warmup_steps,
            step,
        )
        .clamp(0.0, 1.0)
    }
}

fn recon_step_count(rollout_steps: usize, recon_interval: usize) -> usize {
    if recon_interval == 0 {
        return 0;
    }
    let rollout_steps = rollout_steps.max(1);
    rollout_steps.div_ceil(recon_interval)
}

fn detach_state<B: AutodiffBackend>(state: &mut ModelState<B>) {
    for layer in &mut state.layers {
        if let Some(rho) = layer.rho.take() {
            layer.rho = Some(rho.detach());
        }
    }
}

fn scalar_from_tensor<B: BackendTrait>(value: &Tensor<B, 1>) -> f32 {
    value
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .ok()
        .and_then(|mut vec| vec.pop())
        .unwrap_or(0.0)
}

#[derive(Clone, Debug)]
pub struct SudokuTrainer<B: BackendTrait> {
    pub model: SudokuSaccadeModel<B>,
    pub training: SudokuTrainingHyperparameters,
    pub total_steps: usize,
    step_counter: Arc<AtomicUsize>,
    gdpo_stats: Arc<Mutex<GdpoAdvantageStats<B>>>,
    entropy_alpha: Arc<Mutex<f32>>,
    entropy_target_ema: Arc<Mutex<f32>>,
    valid_epoch_counter: Arc<AtomicUsize>,
    valid_epoch: usize,
    valid_step_counter: Arc<AtomicUsize>,
    artifacts: Option<SudokuArtifactConfig>,
    artifact_run_dir: Option<Arc<PathBuf>>,
}

impl<B: BackendTrait> SudokuTrainer<B> {
    pub fn new(
        model: SudokuSaccadeModel<B>,
        training: SudokuTrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        let entropy_alpha = training.policy.entropy_alpha.max(0.0);
        Self {
            model,
            training,
            total_steps: total_steps.max(1),
            step_counter: Arc::new(AtomicUsize::new(0)),
            gdpo_stats: Arc::new(Mutex::new(GdpoAdvantageStats::default())),
            entropy_alpha: Arc::new(Mutex::new(entropy_alpha)),
            entropy_target_ema: Arc::new(Mutex::new(0.0)),
            valid_epoch_counter: Arc::new(AtomicUsize::new(0)),
            valid_epoch: 0,
            valid_step_counter: Arc::new(AtomicUsize::new(0)),
            artifacts: None,
            artifact_run_dir: None,
        }
    }

    pub fn with_validation_artifacts(
        mut self,
        artifacts: SudokuArtifactConfig,
        run_dir: PathBuf,
    ) -> Self {
        self.artifacts = Some(artifacts);
        self.artifact_run_dir = Some(Arc::new(run_dir));
        self
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
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
            entropy_alpha: Arc::clone(&self.entropy_alpha),
            entropy_target_ema: Arc::clone(&self.entropy_target_ema),
            valid_epoch_counter: Arc::clone(&self.valid_epoch_counter),
            valid_epoch: self.valid_epoch,
            valid_step_counter: Arc::clone(&self.valid_step_counter),
            artifacts: self.artifacts.clone(),
            artifact_run_dir: self.artifact_run_dir.clone(),
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: self.model.to_device(device),
            training: self.training,
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
            entropy_alpha: Arc::clone(&self.entropy_alpha),
            entropy_target_ema: Arc::clone(&self.entropy_target_ema),
            valid_epoch_counter: Arc::clone(&self.valid_epoch_counter),
            valid_epoch: self.valid_epoch,
            valid_step_counter: Arc::clone(&self.valid_step_counter),
            artifacts: self.artifacts.clone(),
            artifact_run_dir: self.artifact_run_dir.clone(),
        }
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, visitor: &mut Visitor) {
        self.model.visit(visitor);
    }

    fn map<Mapper: ModuleMapper<B>>(self, mapper: &mut Mapper) -> Self {
        Self {
            model: self.model.map(mapper),
            training: self.training,
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
            entropy_alpha: Arc::clone(&self.entropy_alpha),
            entropy_target_ema: Arc::clone(&self.entropy_target_ema),
            valid_epoch_counter: Arc::clone(&self.valid_epoch_counter),
            valid_epoch: self.valid_epoch,
            valid_step_counter: Arc::clone(&self.valid_step_counter),
            artifacts: self.artifacts.clone(),
            artifact_run_dir: self.artifact_run_dir.clone(),
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: self.model.load_record(record),
            training: self.training,
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
            entropy_alpha: Arc::clone(&self.entropy_alpha),
            entropy_target_ema: Arc::clone(&self.entropy_target_ema),
            valid_epoch_counter: Arc::clone(&self.valid_epoch_counter),
            valid_epoch: self.valid_epoch,
            valid_step_counter: Arc::clone(&self.valid_step_counter),
            artifacts: self.artifacts.clone(),
            artifact_run_dir: self.artifact_run_dir.clone(),
        }
    }

    fn into_record(self) -> Self::Record {
        self.model.into_record()
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for SudokuTrainer<B> {
    type InnerModule = SudokuTrainer<B::InnerBackend>;

    fn valid(&self) -> Self::InnerModule {
        let valid_epoch = self.valid_epoch_counter.fetch_add(1, Ordering::Relaxed) + 1;
        SudokuTrainer {
            model: self.model.valid(),
            training: self.training.clone(),
            total_steps: self.total_steps,
            step_counter: Arc::new(AtomicUsize::new(0)),
            gdpo_stats: Arc::new(Mutex::new(GdpoAdvantageStats::default())),
            entropy_alpha: Arc::clone(&self.entropy_alpha),
            entropy_target_ema: Arc::clone(&self.entropy_target_ema),
            valid_epoch_counter: Arc::clone(&self.valid_epoch_counter),
            valid_epoch,
            valid_step_counter: Arc::new(AtomicUsize::new(0)),
            artifacts: self.artifacts.clone(),
            artifact_run_dir: self.artifact_run_dir.clone(),
        }
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        SudokuTrainer {
            model: AutodiffModule::from_inner(module.model),
            training: module.training,
            total_steps: module.total_steps,
            step_counter: module.step_counter,
            gdpo_stats: Arc::new(Mutex::new(GdpoAdvantageStats::default())),
            entropy_alpha: module.entropy_alpha,
            entropy_target_ema: module.entropy_target_ema,
            valid_epoch_counter: module.valid_epoch_counter,
            valid_epoch: module.valid_epoch,
            valid_step_counter: module.valid_step_counter,
            artifacts: module.artifacts,
            artifact_run_dir: module.artifact_run_dir,
        }
    }
}

impl<B: AutodiffBackend> TrainStep for SudokuTrainer<B> {
    type Input = SudokuBatch<B>;
    type Output = SudokuTrainItem<B>;

    fn step(&self, batch: SudokuBatch<B>) -> TrainOutput<SudokuTrainItem<B>> {
        let step = self.step_counter.fetch_add(1, Ordering::Relaxed);
        let teacher_forcing_prob = schedule_linear(
            self.training.policy.teacher_forcing_prob,
            self.training.policy.teacher_forcing_final,
            self.training.policy.teacher_forcing_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);
        let policy_epsilon = schedule_linear(
            self.training.policy.epsilon,
            self.training.policy.epsilon_final,
            self.training.policy.epsilon_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);

        let policy_temperature = schedule_linear(
            self.training.policy.temperature,
            self.training.policy.temperature_final,
            self.training.policy.temperature_anneal_steps,
            step,
        )
        .max(1e-4);
        let scheduled_entropy_weight = schedule_linear(
            self.training.policy.entropy_weight,
            self.training.policy.entropy_weight_final,
            self.training.policy.entropy_anneal_steps,
            step,
        )
        .max(0.0);
        let mut policy_entropy_weight = if self.training.policy.entropy_adaptive {
            let entropy_alpha = *self.entropy_alpha.lock().unwrap();
            entropy_alpha.max(0.0)
        } else {
            scheduled_entropy_weight
        };
        if self.training.gdpo.enabled {
            policy_entropy_weight = 0.0;
        }
        let policy_recon_weight = self.training.policy.recon_weight.max(0.0);
        let write_gate_floor = write_gate_floor_for_step(&self.training, step);
        let recon_loss_weight = schedule_linear(
            self.training.recon.loss_weight,
            self.training.recon.loss_weight_final,
            self.training.recon.loss_weight_anneal_steps,
            step,
        )
        .max(0.0);
        let revisit_min_filled = schedule_linear(
            self.training.revisit.min_filled_frac,
            self.training.revisit.min_filled_final,
            self.training.revisit.min_filled_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);
        let rollout_steps = sample_rollout_steps(&self.training, step);
        let (losses, grads) = rollout_losses_train::<B>(
            self,
            batch,
            self.training.policy.noise,
            policy_epsilon,
            self.training.gdpo.enabled,
            teacher_forcing_prob,
            policy_temperature,
            policy_entropy_weight,
            policy_recon_weight,
            recon_loss_weight,
            write_gate_floor,
            revisit_min_filled,
            rollout_steps,
            Some(Arc::clone(&self.gdpo_stats)),
        );
        let mut target_tensor = losses.policy_entropy_target.clone();
        let update_entropy_stats = !self.training.gdpo.enabled
            && (self.training.policy.entropy_adaptive
                || self.training.policy.entropy_target_ema_decay > 0.0);
        if update_entropy_stats {
            let target_value = scalar_from_tensor(&losses.policy_entropy_target);
            let ema_decay = self
                .training
                .policy
                .entropy_target_ema_decay
                .clamp(0.0, 0.9999);
            let mut target_ema = self.entropy_target_ema.lock().unwrap();
            let ema_value = if ema_decay > 0.0 {
                if *target_ema <= 0.0 {
                    target_value
                } else {
                    ema_decay * *target_ema + (1.0 - ema_decay) * target_value
                }
            } else {
                target_value
            };
            *target_ema = ema_value;
            target_tensor = Tensor::<B, 1>::from_data(
                TensorData::new(vec![ema_value], [1]),
                &losses.policy_entropy_target.device(),
            );
            if self.training.policy.entropy_adaptive && self.training.policy.entropy_alpha_lr > 0.0
            {
                let entropy_value = scalar_from_tensor(&losses.policy_entropy);
                let mut entropy_alpha = self.entropy_alpha.lock().unwrap();
                let next = (*entropy_alpha
                    + self.training.policy.entropy_alpha_lr * (ema_value - entropy_value))
                    .clamp(1e-4, 10.0);
                *entropy_alpha = next;
            }
        }
        let item = SudokuTrainItem::new(
            losses.loss,
            losses.recon_loss,
            losses.acc,
            losses.exact_acc,
            losses.solve_rate,
            losses.policy_loss,
            losses.halt_loss,
            losses.halt_prob_mean,
            losses.halt_target_mean,
            losses.advantage_abs_mean,
            losses.advantage_std,
            losses.log_prob_mean,
            losses.policy_entropy,
            losses.policy_entropy_alpha,
            target_tensor,
            losses.hard_reward_mean,
            losses.easy_reward_mean,
            losses.shaping_conflict_mean,
            losses.shaping_unknown_mean,
            losses.shaping_accuracy_mean,
            losses.shaping_incorrect_mean,
            losses.saccade_revisit_rate,
            losses.saccade_repeat_rate,
            losses.saccade_unknown_frac,
            losses.saccade_unique_frac,
            losses.write_gate_mean,
            losses.write_rate,
        );
        TrainOutput { grads, item }
    }
}

impl<B: BackendTrait> ValidStep for SudokuTrainer<B> {
    type Input = SudokuBatch<B>;
    type Output = SudokuOutput<B>;

    fn step(&self, batch: SudokuBatch<B>) -> SudokuOutput<B> {
        let step_idx = self.valid_step_counter.fetch_add(1, Ordering::Relaxed);
        let train_step = self.step_counter.load(Ordering::Relaxed);
        if step_idx == 0
            && let (Some(artifacts), Some(run_dir)) = (&self.artifacts, &self.artifact_run_dir)
            && artifacts.max_samples > 0
            && let Err(err) = write_validation_artifacts_from_batch(
                &self.model,
                &batch,
                artifacts,
                &self.training,
                run_dir.as_ref(),
                self.valid_epoch,
            )
        {
            warn!("validation artifacts failed: {err}");
        }

        let write_gate_floor = write_gate_floor_for_step(&self.training, train_step);
        let losses =
            rollout_losses_valid::<B>(&self.model, batch, &self.training, write_gate_floor);
        let target_tensor = losses.policy_entropy_target.clone();
        SudokuOutput::new(
            losses.loss,
            losses.recon_loss,
            losses.acc,
            losses.exact_acc,
            losses.solve_rate,
            losses.policy_loss,
            losses.halt_loss,
            losses.halt_prob_mean,
            losses.halt_target_mean,
            losses.advantage_abs_mean,
            losses.advantage_std,
            losses.log_prob_mean,
            losses.policy_entropy,
            losses.policy_entropy_alpha,
            target_tensor,
            losses.hard_reward_mean,
            losses.easy_reward_mean,
            losses.shaping_conflict_mean,
            losses.shaping_unknown_mean,
            losses.shaping_accuracy_mean,
            losses.shaping_incorrect_mean,
            losses.saccade_revisit_rate,
            losses.saccade_repeat_rate,
            losses.saccade_unknown_frac,
            losses.saccade_unique_frac,
            losses.write_gate_mean,
            losses.write_rate,
        )
    }
}

pub struct SudokuLosses<B: BackendTrait> {
    pub loss: Tensor<B, 1>,
    pub recon_loss: Tensor<B, 1>,
    pub acc: Tensor<B, 1>,
    pub exact_acc: Tensor<B, 1>,
    pub solve_rate: Tensor<B, 1>,
    pub policy_loss: Tensor<B, 1>,
    pub halt_loss: Tensor<B, 1>,
    pub halt_prob_mean: Tensor<B, 1>,
    pub halt_target_mean: Tensor<B, 1>,
    pub advantage_abs_mean: Tensor<B, 1>,
    pub advantage_std: Tensor<B, 1>,
    pub log_prob_mean: Tensor<B, 1>,
    pub policy_entropy: Tensor<B, 1>,
    pub policy_entropy_alpha: Tensor<B, 1>,
    pub policy_entropy_target: Tensor<B, 1>,
    pub hard_reward_mean: Tensor<B, 1>,
    pub easy_reward_mean: Tensor<B, 1>,
    pub shaping_conflict_mean: Tensor<B, 1>,
    pub shaping_unknown_mean: Tensor<B, 1>,
    pub shaping_accuracy_mean: Tensor<B, 1>,
    pub shaping_incorrect_mean: Tensor<B, 1>,
    pub saccade_revisit_rate: Tensor<B, 1>,
    pub saccade_repeat_rate: Tensor<B, 1>,
    pub saccade_unknown_frac: Tensor<B, 1>,
    pub saccade_unique_frac: Tensor<B, 1>,
    pub write_gate_mean: Tensor<B, 1>,
    pub write_rate: Tensor<B, 1>,
}

#[derive(Debug, Default)]
struct GdpoAdvantageStats<B: BackendTrait> {
    mean: Option<Tensor<B, 2>>,
    std: Option<Tensor<B, 2>>,
}

impl<B: BackendTrait> GdpoAdvantageStats<B> {
    fn update(
        &mut self,
        batch_mean: Tensor<B, 1>,
        batch_std: Tensor<B, 1>,
        decay: f32,
    ) -> (Tensor<B, 2>, Tensor<B, 2>) {
        let batch_mean = batch_mean.reshape([1, 1]);
        let batch_std = batch_std.reshape([1, 1]);
        if self.mean.is_none() || self.std.is_none() {
            self.mean = Some(batch_mean.clone());
            self.std = Some(batch_std.clone());
            return (batch_mean, batch_std);
        }
        let decay = decay.clamp(0.0, 0.9999);
        let keep = decay;
        let add = 1.0 - decay;
        let mean = self.mean.take().expect("gdpo mean");
        let std = self.std.take().expect("gdpo std");
        let mean = mean
            .mul_scalar(keep)
            .add(batch_mean.clone().mul_scalar(add));
        let std = std.mul_scalar(keep).add(batch_std.clone().mul_scalar(add));
        self.mean = Some(mean.clone());
        self.std = Some(std.clone());
        (mean, std)
    }
}

fn schedule_linear(start: f32, end: f32, steps: usize, step: usize) -> f32 {
    if steps == 0 {
        return start;
    }
    let t = (step as f32 / steps.max(1) as f32).clamp(0.0, 1.0);
    start + (end - start) * t
}

mod loss;
mod metrics;
mod reward;
mod rollout;
mod trm;

use self::loss::*;
pub(crate) use self::metrics::sample_actions;
use self::metrics::*;
pub(crate) use self::reward::static_traversal_actions;
use self::reward::*;
use self::rollout::*;
use self::trm::*;

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    #[test]
    fn compute_loss_and_acc_perfect_logits() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let batch = 1;
        let time = GRID_LEN;
        let mut solutions = vec![0i64; batch * time];
        for (idx, value) in solutions.iter_mut().enumerate().take(time) {
            *value = ((idx % 9) + 1) as i64;
        }
        let solutions =
            Tensor::<B, 2, Int>::from_data(TensorData::new(solutions, [batch, time]), &device);
        let solution_one_hot = build_solution_one_hot(&solutions, batch, &device);

        let mut logits = vec![0f32; batch * time * VOCAB_SIZE];
        for idx in 0..time {
            let target = (idx % 9) + 1;
            logits[idx * VOCAB_SIZE + target] = 10.0;
        }
        let logits =
            Tensor::<B, 3>::from_data(TensorData::new(logits, [batch, time, VOCAB_SIZE]), &device);

        let loss_mask = Tensor::<B, 2>::ones([batch, time], &device);
        let (loss, acc, exact_acc, ..) = compute_loss_and_acc(
            &logits,
            &solutions,
            &solution_one_hot,
            &loss_mask,
            &SudokuReconLoss::Softmax,
        );

        let loss_value = loss.into_data().iter::<f32>().next().unwrap_or(0.0);
        let acc_value = acc.into_data().iter::<f32>().next().unwrap_or(0.0);
        let exact_value = exact_acc.into_data().iter::<f32>().next().unwrap_or(0.0);

        assert!(loss_value < 1e-2, "loss too high: {}", loss_value);
        assert!(
            (acc_value - 1.0).abs() < 1e-6,
            "acc not perfect: {}",
            acc_value
        );
        assert!(
            (exact_value - 1.0).abs() < 1e-6,
            "exact acc not perfect: {}",
            exact_value
        );
    }

    #[test]
    fn policy_epsilon_sampling_respects_mask() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let logits = Tensor::<B, 2>::zeros([2, 4], &device);
        let mask_data = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let select_mask = Tensor::<B, 2>::from_data(TensorData::new(mask_data, [2, 4]), &device);
        let actions = sample_actions(logits, select_mask, true, 1.0);
        let actions = actions
            .to_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("actions to vec");
        for action in actions {
            assert!(action == 1 || action == 3, "unexpected action {}", action);
        }
    }

    #[test]
    fn gdpo_disabled_policy_loss_is_zero() {
        type B = Autodiff<NdArray<f32>>;
        let device = <B as BackendTrait>::Device::default();
        let model = SudokuSaccadeModel::new(
            &SudokuModelConfig {
                n_layer: 1,
                n_embd: 32,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                summary_tokens: 1,
                policy_heads: 1,
                policy_head: SudokuPolicyHead::Cache,
                policy_mlp_hidden_mult: 2,
                rotary_embedding: Default::default(),
                grid_positional: SudokuGridPositional::Additive,
                grid_rope_theta: 65_536.0,
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
                cache_update: SudokuCacheUpdateConfig::default(),
            },
            &device,
        );

        let training = SudokuTrainingHyperparameters {
            batch_size: 2,
            epochs: None,
            max_iters: 1,
            log_frequency: 1,
            rollout: SudokuRolloutConfig {
                steps: 2,
                ..SudokuRolloutConfig::default()
            },
            halt: SudokuHaltConfig {
                weight: 0.1,
                exploration_prob: 0.0,
                min_steps: 1,
            },
            policy: SudokuPolicyConfig {
                noise: 0.0,
                epsilon: 0.0,
                epsilon_final: 0.0,
                epsilon_anneal_steps: 0,
                teacher_forcing_prob: 0.0,
                teacher_forcing_final: 0.0,
                teacher_forcing_anneal_steps: 0,
                temperature: 1.0,
                temperature_final: 1.0,
                temperature_anneal_steps: 0,
                entropy_weight: 0.0,
                entropy_weight_final: 0.0,
                entropy_anneal_steps: 0,
                entropy_adaptive: false,
                entropy_target_scale: 1.0,
                entropy_target_ema_decay: 0.0,
                entropy_alpha: 0.0,
                entropy_alpha_lr: 0.0,
                visit_penalty: 0.0,
                revisit_penalty: 0.0,
                recon_weight: 0.0,
                write_gate_mode: SudokuWriteGateMode::StraightThrough,
                write_gate_warmup_steps: 0,
                write_gate_warmup_floor: 0.0,
                cache_update_clues: false,
            },
            revisit: SudokuRevisitConfig {
                min_filled_frac: 0.0,
                min_filled_final: 0.0,
                min_filled_anneal_steps: 0,
            },
            reward: SudokuRewardConfig {
                unknown_power: 0.0,
                ..SudokuRewardConfig::default()
            },
            recon: SudokuReconConfig {
                loss: SudokuReconLoss::Softmax,
                loss_mask: SudokuLossMask::All,
                loss_interval_steps: 1,
                global_loss_samples: 8,
                global_loss_weight: 0.2,
                ..SudokuReconConfig::default()
            },
            validation: SudokuValidationConfig::default(),
            gdpo: GdpoConfig {
                enabled: false,
                group_size: 1,
                ..GdpoConfig::default()
            },
        };

        let puzzles = Tensor::<B, 2, Int>::zeros([2, GRID_LEN], &device);
        let solution_values = vec![1i64; 2 * GRID_LEN];
        let solutions = Tensor::<B, 2, Int>::from_data(
            TensorData::new(solution_values, [2, GRID_LEN]),
            &device,
        );
        let batch = SudokuBatch::new(puzzles, solutions);

        let losses = rollout_losses(
            &model, batch, &training, 0.0, 0.0, false, 0.0, 1.0, 0.0, 0.0, None,
        );
        let policy_loss = losses
            .policy_loss
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(0.0);

        assert!(
            policy_loss.abs() < 1e-6,
            "expected zero policy loss when gdpo disabled, got {}",
            policy_loss
        );
    }

    #[test]
    fn recon_weight_scales_loss() {
        type B = Autodiff<NdArray<f32>>;
        let device = <B as BackendTrait>::Device::default();
        let model = SudokuSaccadeModel::new(
            &SudokuModelConfig {
                n_layer: 1,
                n_embd: 32,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                summary_tokens: 1,
                policy_heads: 1,
                policy_head: SudokuPolicyHead::Cache,
                policy_mlp_hidden_mult: 2,
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
                cache_update: SudokuCacheUpdateConfig::default(),
                ..SudokuModelConfig::default()
            },
            &device,
        );

        let recon_weight = 2.5;
        let training = SudokuTrainingHyperparameters {
            batch_size: 1,
            epochs: None,
            max_iters: 1,
            log_frequency: 1,
            rollout: SudokuRolloutConfig {
                steps: 1,
                ..SudokuRolloutConfig::default()
            },
            halt: SudokuHaltConfig {
                weight: 0.0,
                exploration_prob: 0.0,
                min_steps: 1,
            },
            policy: SudokuPolicyConfig {
                noise: 0.0,
                epsilon: 0.0,
                epsilon_final: 0.0,
                epsilon_anneal_steps: 0,
                teacher_forcing_prob: 1.0,
                teacher_forcing_final: 1.0,
                teacher_forcing_anneal_steps: 0,
                temperature: 1.0,
                temperature_final: 1.0,
                temperature_anneal_steps: 0,
                entropy_weight: 0.0,
                entropy_weight_final: 0.0,
                entropy_anneal_steps: 0,
                entropy_adaptive: false,
                entropy_target_scale: 1.0,
                entropy_target_ema_decay: 0.0,
                entropy_alpha: 0.0,
                entropy_alpha_lr: 0.0,
                visit_penalty: 0.0,
                revisit_penalty: 0.0,
                recon_weight: 0.0,
                write_gate_mode: SudokuWriteGateMode::StraightThrough,
                write_gate_warmup_steps: 0,
                write_gate_warmup_floor: 0.0,
                cache_update_clues: false,
            },
            revisit: SudokuRevisitConfig {
                min_filled_frac: 0.0,
                min_filled_final: 0.0,
                min_filled_anneal_steps: 0,
            },
            reward: SudokuRewardConfig {
                unknown_power: 0.0,
                ..SudokuRewardConfig::default()
            },
            recon: SudokuReconConfig {
                loss: SudokuReconLoss::Softmax,
                loss_mask: SudokuLossMask::All,
                loss_weight: recon_weight,
                loss_weight_final: recon_weight,
                loss_weight_anneal_steps: 0,
                loss_interval_steps: 1,
                global_loss_samples: 0,
                global_loss_weight: 0.0,
                ..SudokuReconConfig::default()
            },
            validation: SudokuValidationConfig::default(),
            gdpo: GdpoConfig {
                enabled: false,
                group_size: 1,
                ..GdpoConfig::default()
            },
        };

        let trainer = SudokuTrainer::new(model, training, 1);
        let puzzles = Tensor::<B, 2, Int>::zeros([1, GRID_LEN], &device);
        let solutions = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![1i64; GRID_LEN], [1, GRID_LEN]),
            &device,
        );
        let batch = SudokuBatch::new(puzzles, solutions);

        let (losses, _grads) = rollout_losses_train(
            &trainer,
            batch,
            0.0,
            0.0,
            false,
            1.0,
            1.0,
            0.0,
            0.0,
            recon_weight,
            0.0,
            0.0,
            1,
            None,
        );

        let loss = losses.loss.into_data().iter::<f32>().next().unwrap_or(0.0);
        let recon = losses
            .recon_loss
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(0.0);

        assert!(
            (loss - recon * recon_weight).abs() < 1e-4,
            "expected loss to scale with recon weight (loss={}, recon={}, weight={})",
            loss,
            recon,
            recon_weight
        );
    }

    #[test]
    fn unknown_potential_matches_unknown_count() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let tokens = Tensor::<B, 2, Int>::from_data(
            TensorData::new(vec![0i64, 3, 0, 5, 1, 0], [2, 3]),
            &device,
        );
        let potential = unknown_potential(&tokens);
        let values = potential
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .unwrap_or_default();

        assert_eq!(values.len(), 2, "expected one potential per batch");
        assert!(
            (values[0] + 2.0).abs() < 1e-6,
            "expected -2, got {}",
            values[0]
        );
        assert!(
            (values[1] + 1.0).abs() < 1e-6,
            "expected -1, got {}",
            values[1]
        );
    }

    #[test]
    fn solve_rate_is_one_for_complete_puzzle() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let model = SudokuSaccadeModel::new(
            &SudokuModelConfig {
                n_layer: 1,
                n_embd: 32,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                summary_tokens: 1,
                policy_heads: 1,
                policy_head: SudokuPolicyHead::Cache,
                policy_mlp_hidden_mult: 2,
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
                cache_update: SudokuCacheUpdateConfig::default(),
                ..SudokuModelConfig::default()
            },
            &device,
        );

        let training = SudokuTrainingHyperparameters {
            batch_size: 2,
            epochs: None,
            max_iters: 1,
            log_frequency: 1,
            rollout: SudokuRolloutConfig {
                steps: 2,
                ..SudokuRolloutConfig::default()
            },
            halt: SudokuHaltConfig {
                weight: 0.1,
                exploration_prob: 0.0,
                min_steps: 1,
            },
            policy: SudokuPolicyConfig {
                noise: 0.0,
                epsilon: 0.0,
                epsilon_final: 0.0,
                epsilon_anneal_steps: 0,
                teacher_forcing_prob: 0.0,
                teacher_forcing_final: 0.0,
                teacher_forcing_anneal_steps: 0,
                temperature: 1.0,
                temperature_final: 1.0,
                temperature_anneal_steps: 0,
                entropy_weight: 0.0,
                entropy_weight_final: 0.0,
                entropy_anneal_steps: 0,
                entropy_adaptive: false,
                entropy_target_scale: 1.0,
                entropy_target_ema_decay: 0.0,
                entropy_alpha: 0.0,
                entropy_alpha_lr: 0.0,
                visit_penalty: 0.0,
                revisit_penalty: 0.0,
                recon_weight: 0.0,
                write_gate_mode: SudokuWriteGateMode::StraightThrough,
                write_gate_warmup_steps: 0,
                write_gate_warmup_floor: 0.0,
                cache_update_clues: false,
            },
            revisit: SudokuRevisitConfig {
                min_filled_frac: 0.0,
                min_filled_final: 0.0,
                min_filled_anneal_steps: 0,
            },
            reward: SudokuRewardConfig {
                unknown_power: 0.0,
                ..SudokuRewardConfig::default()
            },
            recon: SudokuReconConfig {
                loss: SudokuReconLoss::Softmax,
                loss_mask: SudokuLossMask::All,
                loss_interval_steps: 1,
                global_loss_samples: 8,
                global_loss_weight: 0.2,
                ..SudokuReconConfig::default()
            },
            validation: SudokuValidationConfig::default(),
            gdpo: GdpoConfig {
                enabled: false,
                group_size: 1,
                ..GdpoConfig::default()
            },
        };

        let filled = vec![1i64; 2 * GRID_LEN];
        let puzzles =
            Tensor::<B, 2, Int>::from_data(TensorData::new(filled.clone(), [2, GRID_LEN]), &device);
        let solutions =
            Tensor::<B, 2, Int>::from_data(TensorData::new(filled, [2, GRID_LEN]), &device);
        let batch = SudokuBatch::new(puzzles, solutions);

        let write_gate_floor = write_gate_floor_for_step(&training, 0);
        let losses = rollout_losses_valid(&model, batch, &training, write_gate_floor);
        let solve_rate = losses
            .solve_rate
            .into_data()
            .iter::<f32>()
            .next()
            .unwrap_or(0.0);

        assert!(
            (solve_rate - 1.0).abs() < 1e-6,
            "expected solve rate 1.0, got {}",
            solve_rate
        );
    }

    #[test]
    fn difficulty_scale_tracks_unknown_counts() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let counts = Tensor::<B, 1>::from_data(
            TensorData::new(vec![0.0, 40.0, GRID_LEN as f32], [3]),
            &device,
        );
        let scale = difficulty_scale_from_unknowns(counts, 1.0);
        let values: Vec<f32> = scale.into_data().iter::<f32>().collect();
        let expected_mid = 40.0 / GRID_LEN as f32;

        assert!(values.len() >= 3, "expected three scale values");
        assert!(values[0].abs() < 1e-6, "zero unknowns should yield 0");
        assert!(
            (values[1] - expected_mid).abs() < 1e-3,
            "mid unknowns scaling off: {} vs {}",
            values[1],
            expected_mid
        );
        assert!(
            (values[2] - 1.0).abs() < 1e-6,
            "full unknowns should yield 1.0, got {}",
            values[2]
        );
    }
}
