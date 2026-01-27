use crate::artifacts::write_validation_artifacts_from_batch;
use crate::train::metrics::{SudokuOutput, SudokuTrainItem};
use crate::train::prelude::*;
use crate::vocab::{GRID_LEN, VOCAB_SIZE};
use burn_dragon_core::ModelState;
use burn_dragon_train::train::gdpo::{gdpo_advantage_autodiff, gdpo_policy_loss};
use std::sync::Mutex;
use tracing::warn;

const SUDOKU_EPS: f32 = 1e-6;
const POLICY_MASK_PENALTY: f32 = 1e9;
const HALT_EPS: f32 = 1e-6;
const STABLEMAX_EPS: f32 = 1e-6;
const GUMBEL_EPS: f32 = 1e-6;

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
}

impl<B: AutodiffBackend> TrainStep<SudokuBatch<B>, SudokuTrainItem<B>> for SudokuTrainer<B> {
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
            let ema_decay = self.training.policy.entropy_target_ema_decay.clamp(0.0, 0.9999);
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
            if self.training.policy.entropy_adaptive
                && self.training.policy.entropy_alpha_lr > 0.0
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
            losses.saccade_revisit_rate,
            losses.saccade_repeat_rate,
            losses.saccade_unknown_frac,
            losses.saccade_unique_frac,
        );
        TrainOutput { grads, item }
    }
}

impl<B: BackendTrait> ValidStep<SudokuBatch<B>, SudokuOutput<B>> for SudokuTrainer<B> {
    fn step(&self, batch: SudokuBatch<B>) -> SudokuOutput<B> {
        let step_idx = self.valid_step_counter.fetch_add(1, Ordering::Relaxed);
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

        let losses = rollout_losses_valid::<B>(&self.model, batch, &self.training);
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
            losses.saccade_revisit_rate,
            losses.saccade_repeat_rate,
            losses.saccade_unknown_frac,
            losses.saccade_unique_frac,
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
    pub saccade_revisit_rate: Tensor<B, 1>,
    pub saccade_repeat_rate: Tensor<B, 1>,
    pub saccade_unknown_frac: Tensor<B, 1>,
    pub saccade_unique_frac: Tensor<B, 1>,
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
        let mean = mean.mul_scalar(keep).add(batch_mean.clone().mul_scalar(add));
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

fn apply_advantage_guardrails<B: BackendTrait>(
    advantage: Tensor<B, 2>,
    config: &GdpoConfig,
    stats: Option<&Arc<Mutex<GdpoAdvantageStats<B>>>>,
) -> Tensor<B, 2> {
    let mut advantage = advantage;
    let decay = config.advantage_ema_decay;
    if decay > 0.0
        && let Some(stats) = stats
        && let Ok(mut stats) = stats.lock()
    {
        let batch_mean = advantage.clone().mean();
        let batch_sq_mean = advantage.clone().powf_scalar(2.0).mean();
        let batch_var = batch_sq_mean
            .clone()
            .sub(batch_mean.clone().powf_scalar(2.0))
            .clamp_min(0.0);
        let batch_std = (batch_var + config.norm_epsilon.max(1e-12)).sqrt();
        let (ema_mean, ema_std) = stats.update(batch_mean, batch_std, decay);
        let ema_std = ema_std.clamp_min(config.norm_epsilon.max(1e-12));
        let [batch, _] = advantage.shape().dims::<2>();
        let ema_mean = ema_mean.expand([batch.max(1), 1]);
        let ema_std = ema_std.expand([batch.max(1), 1]);
        advantage = advantage.sub(ema_mean).div(ema_std);
    }
    if config.advantage_clip > 0.0 {
        advantage = advantage
            .clamp_min(-config.advantage_clip)
            .clamp_max(config.advantage_clip);
    }
    advantage
}

fn difficulty_scale_from_unknowns<B: BackendTrait>(
    unknown_counts: Tensor<B, 1>,
    power: f32,
) -> Tensor<B, 1> {
    let device = unknown_counts.device();
    let [batch] = unknown_counts.shape().dims();
    if power <= 0.0 {
        return Tensor::<B, 1>::ones([batch.max(1)], &device);
    }
    unknown_counts
        .div_scalar(GRID_LEN as f32)
        .clamp_min(0.0)
        .clamp_max(1.0)
        .powf_scalar(power)
}

fn build_action_index<B: BackendTrait>(batch: usize, device: &B::Device) -> Tensor<B, 2, Int> {
    let batch = batch.max(1);
    Tensor::<B, 1, Int>::arange(0..GRID_LEN as i64, device)
        .unsqueeze_dim::<2>(0)
        .expand([batch, GRID_LEN])
}

fn conflict_count<B: BackendTrait>(tokens: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
    let device = tokens.device();
    let [batch, time] = tokens.shape().dims();
    if batch == 0 || time == 0 {
        return Tensor::<B, 1>::zeros([batch.max(1)], &device);
    }
    let data = tokens.to_data().convert::<i64>();
    let values = data.into_vec::<i64>().unwrap_or_default();
    let mut counts = Vec::with_capacity(batch);
    let stride = time;
    for sample in 0..batch {
        let slice = &values[sample * stride..(sample + 1) * stride];
        let mut conflicts = 0i32;
        for row in 0..9 {
            let mut seen = [0i32; 10];
            for col in 0..9 {
                let v = slice[row * 9 + col] as usize;
                if v > 0 && v <= 9 {
                    seen[v] += 1;
                }
            }
            for &count in seen.iter().skip(1) {
                if count > 1 {
                    conflicts += count - 1;
                }
            }
        }
        for col in 0..9 {
            let mut seen = [0i32; 10];
            for row in 0..9 {
                let v = slice[row * 9 + col] as usize;
                if v > 0 && v <= 9 {
                    seen[v] += 1;
                }
            }
            for &count in seen.iter().skip(1) {
                if count > 1 {
                    conflicts += count - 1;
                }
            }
        }
        for box_row in 0..3 {
            for box_col in 0..3 {
                let mut seen = [0i32; 10];
                for row in 0..3 {
                    for col in 0..3 {
                        let idx = (box_row * 3 + row) * 9 + (box_col * 3 + col);
                        let v = slice[idx] as usize;
                        if v > 0 && v <= 9 {
                            seen[v] += 1;
                        }
                    }
                }
                for &count in seen.iter().skip(1) {
                    if count > 1 {
                        conflicts += count - 1;
                    }
                }
            }
        }
        counts.append(&mut vec![conflicts as f32]);
    }
    Tensor::<B, 1>::from_data(TensorData::new(counts, [batch]), &device)
}

fn shaping_potential<B: BackendTrait>(
    tokens: &Tensor<B, 2, Int>,
    metric: SudokuRewardShapingMetric,
) -> Tensor<B, 1> {
    match metric {
        SudokuRewardShapingMetric::Conflict => conflict_potential(tokens),
    }
}

fn conflict_potential<B: BackendTrait>(tokens: &Tensor<B, 2, Int>) -> Tensor<B, 1> {
    conflict_count(tokens).mul_scalar(-1.0)
}

fn gae_advantage<B: BackendTrait>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    gamma: f32,
    lambda: f32,
) -> Tensor<B, 1> {
    if rewards.is_empty() {
        let device = values
            .first()
            .map(|v| v.device())
            .unwrap_or_default();
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let device = rewards[0].device();
    let [batch] = rewards[0].shape().dims();
    let mut adv = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut adv_sum = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut next_value = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    for (reward, value) in rewards.iter().zip(values.iter()).rev() {
        let delta = reward.clone() + next_value.clone().mul_scalar(gamma) - value.clone();
        adv = delta + adv.mul_scalar(gamma * lambda);
        adv_sum = adv_sum + adv.clone();
        next_value = value.clone();
    }
    adv_sum
}

fn gae_advantages<B: BackendTrait>(
    rewards: &[Tensor<B, 1>],
    values: &[Tensor<B, 1>],
    dones: &[Tensor<B, 1>],
    next_value: Tensor<B, 1>,
    gamma: f32,
    lambda: f32,
) -> Vec<Tensor<B, 1>> {
    if rewards.is_empty() || values.is_empty() || dones.is_empty() {
        return Vec::new();
    }
    let device = rewards[0].device();
    let [batch] = rewards[0].shape().dims();
    let mut adv = Tensor::<B, 1>::zeros([batch.max(1)], &device);
    let mut next_value = next_value;
    let mut advantages_rev: Vec<Tensor<B, 1>> = Vec::with_capacity(rewards.len());
    for idx in (0..rewards.len()).rev() {
        let reward = &rewards[idx];
        let value = &values[idx];
        let done = &dones[idx];
        let not_done = done.clone().mul_scalar(-1.0).add_scalar(1.0);
        let delta = reward.clone()
            + next_value.clone().mul_scalar(gamma).mul(not_done.clone())
            - value.clone();
        adv = delta + adv.mul_scalar(gamma * lambda).mul(not_done);
        advantages_rev.push(adv.clone());
        next_value = value.clone();
    }
    advantages_rev.reverse();
    advantages_rev
}


fn ensure_non_empty_mask<B: BackendTrait>(
    mask: Tensor<B, 2>,
    fallback: Tensor<B, 2>,
) -> Tensor<B, 2> {
    let [batch, time] = mask.shape().dims();
    if batch == 0 || time == 0 {
        return mask;
    }
    let device = mask.device();
    let mask_sum = mask.clone().sum_dim(1).reshape([batch, 1]);
    let has_any = mask_sum.greater_elem(0.0).float();
    let has_any_grid = has_any.clone().repeat_dim(1, time);
    let keep = has_any_grid.clone();
    let use_fallback = Tensor::<B, 2>::ones([batch, time], &device) - has_any_grid;
    mask * keep + fallback * use_fallback
}

#[cfg(test)]
#[allow(dead_code, clippy::too_many_arguments)]
fn rollout_losses<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    policy_epsilon: f32,
    gdpo_active: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    policy_entropy_weight: f32,
    revisit_min_filled: f32,
    gdpo_stats: Option<Arc<Mutex<GdpoAdvantageStats<B>>>>,
) -> SudokuLosses<B> {
    let gdpo_group = training.gdpo.group_size.max(1);
    let repeat_for_gdpo = gdpo_active && gdpo_group > 1;
    let rollout = rollout_base_autodiff(
        model,
        batch,
        training,
        policy_noise,
        policy_epsilon,
        repeat_for_gdpo.then_some(gdpo_group),
        gdpo_active,
        true,
        false,
        teacher_forcing_prob,
        policy_temperature,
        revisit_min_filled,
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
    let policy_entropy_target = if rollout.selectable_steps > 0 {
        rollout
            .selectable_count_sum
            .clone()
            .div_scalar(rollout.selectable_steps as f32)
            .clamp_min(1.0)
            .log()
            .mul_scalar(training.policy.entropy_target_scale.max(0.0))
    } else {
        zeros.clone()
    };
    let policy_entropy_alpha = Tensor::<B, 1>::from_data(
        TensorData::new(vec![policy_entropy_weight], [1]),
        &device,
    );

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
        let advantage = apply_advantage_guardrails(
            advantage,
            &training.gdpo,
            gdpo_stats.as_ref(),
        );

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

    let entropy_bonus = if policy_entropy_weight > 0.0 {
        policy_entropy.clone().mul_scalar(policy_entropy_weight)
    } else {
        zeros.clone()
    };
    let loss = rollout.recon_loss.clone()
        + policy_loss.clone()
        + halt_loss.clone().mul_scalar(training.halt.weight)
        - entropy_bonus;

    SudokuLosses {
        loss,
        recon_loss: rollout.recon_loss,
        acc: rollout.acc,
        exact_acc: rollout.exact_acc,
        solve_rate: rollout.solve_rate,
        policy_loss,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        advantage_abs_mean,
        advantage_std,
        log_prob_mean,
        policy_entropy,
        policy_entropy_alpha,
        policy_entropy_target,
        hard_reward_mean,
        easy_reward_mean,
        saccade_revisit_rate: rollout.saccade_revisit_rate,
        saccade_repeat_rate: rollout.saccade_repeat_rate,
        saccade_unknown_frac: rollout.saccade_unknown_frac,
        saccade_unique_frac: rollout.saccade_unique_frac,
    }
}

fn rollout_losses_valid<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
) -> SudokuLosses<B> {
    let sample_policy = training.validation.sample_policy;
    let policy_noise = if sample_policy {
        training.policy.noise
    } else {
        0.0
    };
    let policy_epsilon = if sample_policy {
        training.policy.epsilon
    } else {
        0.0
    };
    let policy_temperature = if sample_policy {
        training.policy.temperature
    } else {
        1.0
    };
    let teacher_forcing_prob = if sample_policy {
        training.policy.teacher_forcing_prob
    } else {
        0.0
    };

    let rollout = rollout_base(
        model,
        batch,
        training,
        policy_noise,
        policy_epsilon,
        None,
        false,
        sample_policy,
        false,
        teacher_forcing_prob,
        policy_temperature,
        training.revisit.min_filled_final,
    );
    let device = rollout.recon_loss.device();
    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let loss = rollout
        .recon_loss
        .clone()
        .add(rollout.halt_loss.clone().mul_scalar(training.halt.weight));
    let policy_entropy_alpha = zeros.clone();
    let policy_entropy_target = zeros.clone();
    SudokuLosses {
        loss,
        recon_loss: rollout.recon_loss,
        acc: rollout.acc,
        exact_acc: rollout.exact_acc,
        solve_rate: rollout.solve_rate,
        policy_loss: zeros.clone(),
        halt_loss: rollout.halt_loss,
        halt_prob_mean: rollout.halt_prob_mean,
        halt_target_mean: rollout.halt_target_mean,
        advantage_abs_mean: zeros.clone(),
        advantage_std: zeros.clone(),
        log_prob_mean: zeros.clone(),
        policy_entropy: zeros.clone(),
        policy_entropy_alpha,
        policy_entropy_target,
        hard_reward_mean: rollout.hard_reward.mean(),
        easy_reward_mean: rollout.easy_reward.mean(),
        saccade_revisit_rate: rollout.saccade_revisit_rate,
        saccade_repeat_rate: rollout.saccade_repeat_rate,
        saccade_unknown_frac: rollout.saccade_unknown_frac,
        saccade_unique_frac: rollout.saccade_unique_frac,
    }
}

#[allow(clippy::too_many_arguments)]
fn rollout_losses_train<B: AutodiffBackend>(
    trainer: &SudokuTrainer<B>,
    batch: SudokuBatch<B>,
    policy_noise: f32,
    policy_epsilon: f32,
    gdpo_active: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    policy_entropy_weight: f32,
    policy_recon_weight: f32,
    revisit_min_filled: f32,
    rollout_steps: usize,
    gdpo_stats: Option<Arc<Mutex<GdpoAdvantageStats<B>>>>,
) -> (SudokuLosses<B>, GradientsParams) {
    let training = &trainer.training;
    let rollout_steps = rollout_steps.max(1);
    let backprop_steps_cfg = training.rollout.backprop_steps.unwrap_or(rollout_steps);
    let chunk_steps = if backprop_steps_cfg == 0 {
        rollout_steps
    } else {
        backprop_steps_cfg.min(rollout_steps).max(1)
    };
    let recon_interval = training.recon.loss_interval_steps;
    let total_local_steps = rollout_steps.max(1);
    let total_global_steps = recon_step_count(rollout_steps, recon_interval);
    let total_halt_steps = rollout_steps.max(1);
    let total_policy_steps = rollout_steps.max(1);
    let global_weight = training.recon.global_loss_weight.max(0.0);
    let visit_penalty = training.policy.visit_penalty.max(0.0);

    let gdpo_group = training.gdpo.group_size.max(1);
    let repeat_for_gdpo = gdpo_active && gdpo_group > 1;
    let global_samples = training.recon.global_loss_samples.min(GRID_LEN);

    let mut puzzles = batch.puzzles;
    let mut solutions = batch.solutions;
    let device = puzzles.device();

    if let Some(group) = repeat_for_gdpo.then_some(gdpo_group)
        && group > 1
    {
        puzzles = puzzles.repeat_dim(0, group);
        solutions = solutions.repeat_dim(0, group);
    }
        let [batch_size, _] = puzzles.shape().dims::<2>();
    let ones_grid = Tensor::<B, 2>::ones([batch_size.max(1), GRID_LEN], &device);
    let action_index = build_action_index(batch_size, &device);
    let solution_one_hot = build_solution_one_hot(&solutions, batch_size, &device);
    let (row_ids, col_ids) = trainer.model.grid_row_col_ids(batch_size, &device);
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
    let initial_unknown_counts = unknown_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1)]);
    let difficulty_scale =
        difficulty_scale_from_unknowns(initial_unknown_counts.clone(), training.reward.unknown_power);
    let easy_mode = training.reward.easy_mode;
    let hard_mode = training.reward.hard_mode;
    let info_enabled = training.reward.info_reward.enabled
        && matches!(hard_mode, SudokuHardRewardMode::InfoReward);
    let info_stride = training.reward.info_reward.stride.max(1);
    let shaping_enabled = training.reward.shaping.enabled && !gdpo_active;
    let shaping_metric = training.reward.shaping.metric;
    let shaping_weight = training.reward.shaping.weight.max(0.0);
    let shaping_gamma = training.reward.shaping.gamma.clamp(0.0, 1.0);
    let baseline_enabled = training.reward.baseline.enabled;
    let baseline_gamma = training.reward.baseline.gamma.clamp(0.0, 1.0);
    let baseline_lambda = training.reward.baseline.lambda.clamp(0.0, 1.0);
    let baseline_value_weight = training.reward.baseline.value_loss_weight.max(0.0);
    let no_op_penalty = training.reward.no_op_penalty.max(0.0);
    let mut shaping_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut conflict_prev = if shaping_enabled {
        shaping_potential(&tokens, shaping_metric)
    } else {
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device)
    };
    let mut chunk_step_rewards: Vec<Tensor<B, 1>> = Vec::new();
    let mut chunk_step_values: Vec<Tensor<B, 1>> = Vec::new();
    let mut chunk_step_log_probs: Vec<Tensor<B, 2>> = Vec::new();
    let mut chunk_step_masks: Vec<Tensor<B, 1>> = Vec::new();
    let mut chunk_step_dones: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_rewards: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_values: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_log_probs: Vec<Tensor<B, 2>> = Vec::new();
    let mut rollout_step_masks: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_dones: Vec<Tensor<B, 1>> = Vec::new();

    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

    let mut state = trainer.model.init_state();
    let input_cache = trainer.model.cell_embeddings_with_positions(
        tokens.clone(),
        row_ids.clone(),
        col_ids.clone(),
    );
    let [_, _, embd] = input_cache.shape().dims();
    let cache_streams = trainer.model.cache_streams();
    let input_cache = input_cache
        .unsqueeze_dim::<4>(1)
        .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
    let input_cache_read = input_cache
        .clone()
        .mean_dim(1)
        .reshape([batch_size.max(1), GRID_LEN, embd]);
    let mut cache = input_cache.clone();
    let mut summary_tokens = trainer.model.init_summary_tokens(batch_size);
    let summary_len = trainer.model.summary_token_count();
    let (reward_initial_acc_per_sample, initial_acc_mean, initial_exact, _initial_solve) =
        compute_grid_accuracy(&tokens_reward, &solutions);
    let reward_initial_acc = reward_initial_acc_per_sample.clone();
    let mut last_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut last_acc = initial_acc_mean.detach();
    let mut last_exact = initial_exact.detach();
    let mut last_reward_acc_per_sample = reward_initial_acc.clone();
    let mut prev_reward_acc_per_sample = reward_initial_acc.clone();

    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let mut recon_loss_sum_metrics = zeros.clone();
    let mut recon_steps_metrics = 0usize;
    let mut halt_loss_sum_metrics = zeros.clone();
    let mut halt_prob_sum_metrics = zeros.clone();
    let mut halt_target_sum_metrics = zeros.clone();
    let mut policy_entropy_sum_metrics =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut log_prob_sum_metrics =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut policy_recon_sum_metrics = zeros.clone();
    let mut action_count_sum_metrics = zeros.clone();
    let mut revisit_count_sum_metrics = zeros.clone();
    let mut repeat_count_sum_metrics = zeros.clone();
    let mut unknown_select_sum_metrics = zeros.clone();

    let mut chunk_local_loss_sum = zeros.clone();
    let mut chunk_global_loss_sum = zeros.clone();
    let mut chunk_halt_loss_sum = zeros.clone();
    let mut chunk_halt_steps = 0usize;
    let mut chunk_policy_entropy_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut chunk_log_prob_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut chunk_policy_recon_sum = zeros.clone();
    let mut chunk_recon_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_recon_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut info_reward_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);

    let mut steps_done = 0usize;

    let mut grads_accum = GradientsAccumulator::<SudokuTrainer<B>>::new();
    let mut visit_counts =
        Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut prev_action_one_hot =
        Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut selectable_count_sum = zeros.clone();
    let mut selectable_steps = 0usize;

    for step_idx in 0..rollout_steps {
        let cache_read = cache
            .clone()
            .mean_dim(1)
            .reshape([batch_size.max(1), GRID_LEN, embd]);
        let policy_logits =
            trainer
                .model
                .policy_logits_from_cache(summary_tokens.clone(), cache_read.clone());

        let step_value = if baseline_enabled {
            Some(
                trainer
                    .model
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

        let selectable_counts = select_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let step_selectable_mean = selectable_counts.clone().mean();
        selectable_count_sum = selectable_count_sum + step_selectable_mean.detach();
        selectable_steps += 1;
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
        let actions = sample_actions(sampled_logits, select_mask.clone(), true, policy_epsilon);
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
        action_count_sum_metrics =
            action_count_sum_metrics + action_any.mean().detach();
        revisit_count_sum_metrics =
            revisit_count_sum_metrics + step_revisit.mean().detach();
        repeat_count_sum_metrics =
            repeat_count_sum_metrics + step_repeat.mean().detach();
        unknown_select_sum_metrics =
            unknown_select_sum_metrics + step_unknown.mean().detach();
        prev_action_one_hot = action_one_hot.clone();
        visit_counts = visit_counts + action_one_hot.clone();

        let log_probs = activation::log_softmax(policy_logits, 1);
        let selected_log_prob = log_probs
            .clone()
            .mul(action_one_hot.clone())
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .mul(active_mask.clone());
        chunk_log_prob_sum = chunk_log_prob_sum + selected_log_prob.clone();
        log_prob_sum_metrics = log_prob_sum_metrics + selected_log_prob.clone().detach();

        let step_entropy = log_probs
            .clone()
            .exp()
            .mul(log_probs)
            .sum_dim(1)
            .reshape([batch_size.max(1), 1])
            .mul_scalar(-1.0)
            .mul(active_mask.clone());
        chunk_policy_entropy_sum = chunk_policy_entropy_sum + step_entropy.clone();
        policy_entropy_sum_metrics = policy_entropy_sum_metrics + step_entropy.detach();
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
        let step_input = trainer
            .model
            .project_input_tokens(step_input_base)
            + step_residual;
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
        let (step_hidden, _step_logits_full) = trainer
            .model
            .forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = summary_hidden.clone();
        let step_logits = trainer.model.value_logits_from_hidden(step_hidden.clone());
        let halt_logit = trainer
            .model
            .halt_logit_from_summary_tokens(summary_tokens.clone());
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
        let reward_mask = selected_editable
            .clone()
            .reshape([batch_size.max(1)]);
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

        let reward_local_mask = active_mask
            .clone()
            .mul(selected_editable.clone());
        let (
            _reward_local_loss,
            _reward_local_acc,
            _reward_local_exact,
            reward_local_loss_per_sample,
            ..
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

        let update_mask = action_one_hot
            .clone()
            .mul(editable_mask.clone())
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
        let cache_cell = cache
            .clone()
            .mul(action_mask.clone())
            .sum_dim(2)
            .reshape([batch_size.max(1) * cache_streams, 1, embd]);
        let summary_streams = summary_hidden
            .clone()
            .unsqueeze_dim::<4>(1)
            .expand([batch_size.max(1), cache_streams, summary_len, embd])
            .reshape([batch_size.max(1) * cache_streams, summary_len, embd]);
        let token_emb = trainer.model.cell_embeddings_with_positions(
            update_values.clone(),
            selected_row,
            selected_col,
        );
        let token_emb = token_emb
            .unsqueeze_dim::<4>(1)
            .expand([batch_size.max(1), cache_streams, 1, embd])
            .reshape([batch_size.max(1) * cache_streams, 1, embd]);
        let update_emb = trainer.model.update_cell_embedding(
            summary_streams,
            cache_cell,
            token_emb,
        );
        let update_emb = update_emb
            .reshape([batch_size.max(1), cache_streams, 1, embd])
            .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
        let update_mask_stream = update_mask_f.clone().unsqueeze_dim::<4>(1);
        let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_stream);
        if let Some(mhc) = trainer.model.cache_mhc.as_ref() {
            let (branch_input, residuals_out, beta) = mhc.width_connection(cache.clone());
            cache = mhc.depth_connection(branch_input, residuals_out, beta);
        }

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
        chunk_halt_loss_sum = chunk_halt_loss_sum + step_halt_loss.clone();
        chunk_halt_steps += 1;
        halt_loss_sum_metrics = halt_loss_sum_metrics + step_halt_loss.detach();
        let halt_prob_masked = halt_prob.clone().mul(halt_target.clone());
        halt_prob_sum_metrics = halt_prob_sum_metrics + halt_prob_masked.detach().mean();
        halt_target_sum_metrics = halt_target_sum_metrics + halt_target.detach().mean();

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

        let mut step_done = active_mask.clone().mul_scalar(-1.0).add_scalar(1.0);
        step_done = step_done.max_pair(tokens_solved_after.clone());
        step_done = step_done.max_pair(halted.clone());
        let step_done = step_done.reshape([batch_size.max(1)]);

        steps_done += 1;

        let (reward_acc_per_sample, acc, exact_acc, _solve_rate) =
            compute_grid_accuracy(&tokens_reward, &solutions);
        let step_acc_delta = reward_acc_per_sample.clone().sub(prev_reward_acc_per_sample.clone());
        prev_reward_acc_per_sample = reward_acc_per_sample.detach();
        last_reward_acc_per_sample = prev_reward_acc_per_sample.clone();
        last_acc = acc.detach();
        last_exact = exact_acc.detach();

        let mut global_loss = zeros.clone();
        let mut global_loss_per_sample =
            Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        let mut reward_global_loss_per_sample =
            Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        let mut global_active = false;
        if recon_interval > 0 && (step_idx + 1) % recon_interval == 0 && global_samples > 0 {
            global_active = true;
            let sample_prob =
                (global_samples as f32 / GRID_LEN as f32).clamp(0.0, 1.0);
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
                reward_global_mask = reward_global_mask.mul(tokens_reward.clone().equal_elem(0).float());
            }
            global_mask = ensure_non_empty_mask(global_mask, action_one_hot.clone());
            let global_logits = trainer.model.value_logits_from_cache(cache_read.clone());
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
                ..
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
        let step_recon_per_sample_reward = reward_local_loss_per_sample
            + reward_global_loss_per_sample.mul_scalar(global_weight);
        let step_recon_per_sample_reward_detached =
            step_recon_per_sample_reward.clone().detach();
        let mut baseline_loss = last_loss_per_sample.clone();
        let baseline_mask = baseline_loss.clone().equal_elem(0.0);
        baseline_loss = baseline_loss.mask_where(
            baseline_mask,
            step_recon_per_sample_reward_detached.clone(),
        );
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
        chunk_recon_delta_sum =
            chunk_recon_delta_sum + step_recon_delta_reward.clone();
        chunk_recon_mask_sum = chunk_recon_mask_sum + step_reward_mask.clone();
        rollout_recon_delta_sum =
            rollout_recon_delta_sum + step_recon_delta_reward.clone();
        rollout_recon_mask_sum = rollout_recon_mask_sum + step_reward_mask.clone();
        let step_active_mask = active_mask.clone().reshape([batch_size.max(1)]);
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
            let conflict_next = shaping_potential(&tokens, shaping_metric);
            let mut shaping_delta =
                conflict_next.clone().mul_scalar(shaping_gamma) - conflict_prev.clone();
            shaping_delta = shaping_delta.mul(reward_mask.clone());
            shaping_sum = shaping_sum + shaping_delta.clone();
            step_reward = step_reward + shaping_delta.mul_scalar(shaping_weight);
            let keep = Tensor::<B, 1>::ones([batch_size.max(1)], &device)
                .sub(reward_mask.clone());
            conflict_prev = conflict_prev.mul(keep) + conflict_next.mul(reward_mask.clone());
        }
        step_reward = step_reward.mul(step_reward_mask.clone());
        if baseline_enabled && let Some(value) = step_value.clone() {
            chunk_step_rewards.push(step_reward.clone());
            chunk_step_values.push(value.clone());
            chunk_step_log_probs.push(selected_log_prob.clone());
            chunk_step_masks.push(step_active_mask.clone());
            chunk_step_dones.push(step_done.clone());
            rollout_step_rewards.push(step_reward.clone());
            rollout_step_values.push(value);
            rollout_step_log_probs.push(selected_log_prob.clone());
            rollout_step_masks.push(step_active_mask.clone());
            rollout_step_dones.push(step_done.clone());
        }

        if policy_recon_weight > 0.0 {
            let reward = step_recon_per_sample_reward
                .clone()
                .mul(reward_mask.clone())
                .mul_scalar(-1.0)
                .detach();
            let reward_mean = reward.clone().mean();
            let advantage = reward
                .sub(reward_mean)
                .reshape([batch_size.max(1), 1]);
            let step_policy_recon = selected_log_prob
                .clone()
                .mul(advantage)
                .mul_scalar(-1.0)
                .mean();
            let step_policy_recon_detached = step_policy_recon.clone().detach();
            chunk_policy_recon_sum = chunk_policy_recon_sum + step_policy_recon;
            policy_recon_sum_metrics =
                policy_recon_sum_metrics + step_policy_recon_detached;
        }

        let reward_keep = Tensor::<B, 1>::ones([batch_size.max(1)], &device)
            .sub(reward_mask.clone());
        last_loss_per_sample = last_loss_per_sample.mul(reward_keep)
            + step_recon_per_sample_reward_detached.mul(reward_mask.clone());
        let step_recon = local_loss.clone() + global_loss.clone().mul_scalar(global_weight);
        recon_loss_sum_metrics = recon_loss_sum_metrics + step_recon.detach();
        recon_steps_metrics += 1;
        chunk_local_loss_sum = chunk_local_loss_sum + local_loss;
        if global_active {
            chunk_global_loss_sum = chunk_global_loss_sum + global_loss;
        }

        let chunk_end = (step_idx + 1) % chunk_steps == 0 || step_idx + 1 == rollout_steps;
        if chunk_end {
            let hard_reward = match hard_mode {
                SudokuHardRewardMode::InfoReward => info_reward_sum.clone(),
                SudokuHardRewardMode::Accuracy => {
                    last_reward_acc_per_sample.clone().sub(reward_initial_acc.clone())
                }
            };
            let easy_reward = match easy_mode {
                SudokuEasyRewardMode::Recon => chunk_recon_delta_sum
                    .clone()
                    .div(chunk_recon_mask_sum.clone().clamp_min(1.0)),
                SudokuEasyRewardMode::AccuracyDelta => chunk_acc_delta_sum
                    .clone()
                    .div(chunk_acc_delta_mask_sum.clone().clamp_min(1.0)),
                SudokuEasyRewardMode::Gae => {
                    if baseline_enabled && !chunk_step_rewards.is_empty() {
                        let values_detached: Vec<_> = chunk_step_values
                            .iter()
                            .map(|value| value.clone().detach())
                            .collect();
                        gae_advantage(
                            &chunk_step_rewards,
                            &values_detached,
                            baseline_gamma,
                            baseline_lambda,
                        )
                    } else {
                        let mut easy = last_loss_per_sample.clone().mul_scalar(-1.0);
                        if shaping_enabled {
                            easy = easy + shaping_sum.clone().mul_scalar(shaping_weight);
                        }
                        easy
                    }
                }
            };
            let hard_reward = hard_reward.mul(difficulty_scale.clone());
            let easy_reward = easy_reward.mul(difficulty_scale.clone());

            let mut advantages: Vec<Tensor<B, 1>> = Vec::new();
            let mut value_loss = zeros.clone();
            if baseline_enabled && !chunk_step_rewards.is_empty() {
                let values_detached: Vec<_> = chunk_step_values
                    .iter()
                    .map(|value| value.clone().detach())
                    .collect();
                let rewards_detached: Vec<_> = chunk_step_rewards
                    .iter()
                    .map(|reward| reward.clone().detach())
                    .collect();
                let dones_detached: Vec<_> = chunk_step_dones
                    .iter()
                    .map(|done| done.clone().detach())
                    .collect();
                let next_value = trainer
                    .model
                    .value_baseline_from_summary_tokens(summary_tokens.clone())
                    .reshape([batch_size.max(1)])
                    .detach();
                advantages = gae_advantages(
                    &rewards_detached,
                    &values_detached,
                    &dones_detached,
                    next_value,
                    baseline_gamma,
                    baseline_lambda,
                );

                if baseline_value_weight > 0.0 {
                    let mut value_loss_sum = zeros.clone();
                    let mut value_loss_steps = 0usize;
                    for ((advantage, value), step_mask) in advantages
                        .iter()
                        .zip(chunk_step_values.iter())
                        .zip(chunk_step_masks.iter())
                    {
                        let target = advantage.clone().detach() + value.clone().detach();
                        let diff = value.clone() - target;
                        let mask_sum = step_mask.clone().sum_dim(0).clamp_min(1.0);
                        let step_loss = diff
                            .powf_scalar(2.0)
                            .mul(step_mask.clone())
                            .sum_dim(0)
                            .div(mask_sum)
                            .reshape([1]);
                        value_loss_sum = value_loss_sum + step_loss;
                        value_loss_steps += 1;
                    }
                    if value_loss_steps > 0 {
                        value_loss = value_loss_sum.div_scalar(value_loss_steps as f32);
                    }
                }
            }

            let policy_loss = if gdpo_active {
                if baseline_enabled && !advantages.is_empty() {
                    let step_count = advantages.len();
                    let batch = batch_size.max(1);
                    let mut adv_stack: Vec<Tensor<B, 2>> = Vec::with_capacity(step_count);
                    let mut log_prob_stack: Vec<Tensor<B, 3>> = Vec::with_capacity(step_count);
                    for idx in 0..step_count {
                        adv_stack.push(
                            advantages[idx]
                                .clone()
                                .mul(chunk_step_masks[idx].clone())
                                .reshape([1, batch]),
                        );
                        log_prob_stack.push(
                            chunk_step_log_probs[idx]
                                .clone()
                                .reshape([1, batch, 1]),
                        );
                    }
                    let mut advantage = Tensor::cat(adv_stack, 0)
                        .reshape([step_count * batch, 1]);
                    if gdpo_group > 1 {
                        let scene_batch = if gdpo_group == 0 {
                            batch
                        } else {
                            batch / gdpo_group
                        };
                        let hard = advantage
                            .clone()
                            .reshape([step_count * scene_batch.max(1), gdpo_group.max(1)]);
                        let easy = Tensor::<B, 2>::zeros(
                            [step_count * scene_batch.max(1), gdpo_group.max(1)],
                            &device,
                        );
                        advantage = gdpo_advantage_autodiff::<B>(hard, easy, &training.gdpo)
                            .reshape([step_count * batch, 1]);
                    }
                    let advantage = apply_advantage_guardrails(
                        advantage,
                        &training.gdpo,
                        gdpo_stats.as_ref(),
                    );
                    let log_prob = Tensor::cat(log_prob_stack, 0)
                        .reshape([step_count * batch, 1]);
                    let log_prob_old = log_prob.clone().detach();
                    gdpo_policy_loss(log_prob, log_prob_old, advantage, &training.gdpo)
                } else {
                    let scene_batch = if gdpo_group == 0 {
                        batch_size
                    } else {
                        batch_size / gdpo_group
                    };
                    let hard = hard_reward
                        .clone()
                        .detach()
                        .reshape([scene_batch.max(1), gdpo_group.max(1)]);
                    let easy = easy_reward
                        .clone()
                        .detach()
                        .reshape([scene_batch.max(1), gdpo_group.max(1)]);
                    let advantage = gdpo_advantage_autodiff::<B>(hard, easy, &training.gdpo)
                        .reshape([batch_size.max(1), 1])
                        .detach();
                    let advantage = apply_advantage_guardrails(
                        advantage,
                        &training.gdpo,
                        gdpo_stats.as_ref(),
                    );
                    let log_prob_old = chunk_log_prob_sum.clone().detach();
                    gdpo_policy_loss(
                        chunk_log_prob_sum.clone(),
                        log_prob_old,
                        advantage,
                        &training.gdpo,
                    )
                }
            } else {
                zeros.clone()
            };

            let entropy_bonus = if policy_entropy_weight > 0.0 {
                chunk_policy_entropy_sum
                    .clone()
                    .div_scalar(total_policy_steps as f32)
                    .mean()
                    .mul_scalar(policy_entropy_weight)
            } else {
                zeros.clone()
            };
            let policy_recon_term = if policy_recon_weight > 0.0 && total_policy_steps > 0 {
                chunk_policy_recon_sum
                    .clone()
                    .div_scalar(total_policy_steps as f32)
            } else {
                zeros.clone()
            };

            let local_term = chunk_local_loss_sum
                .clone()
                .div_scalar(total_local_steps as f32);
            let global_term = if total_global_steps > 0 {
                chunk_global_loss_sum
                    .clone()
                    .div_scalar(total_global_steps as f32)
            } else {
                zeros.clone()
            };
            let recon_term = local_term + global_term.mul_scalar(global_weight);
            let halt_term = if chunk_halt_steps > 0 {
                chunk_halt_loss_sum
                    .clone()
                    .div_scalar(total_halt_steps as f32)
            } else {
                zeros.clone()
            };

            let chunk_loss = recon_term
                + policy_loss
                + policy_recon_term.mul_scalar(policy_recon_weight)
                + value_loss.mul_scalar(baseline_value_weight)
                + halt_term.mul_scalar(training.halt.weight)
                - entropy_bonus;

            let grads = GradientsParams::from_grads(chunk_loss.backward(), trainer);
            grads_accum.accumulate(trainer, grads);

            chunk_local_loss_sum = zeros.clone();
            chunk_global_loss_sum = zeros.clone();
            chunk_halt_loss_sum = zeros.clone();
            chunk_halt_steps = 0;
            chunk_policy_entropy_sum =
                Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
            chunk_log_prob_sum = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
            chunk_policy_recon_sum = zeros.clone();
            chunk_recon_delta_sum =
                Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            chunk_recon_mask_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            chunk_acc_delta_sum =
                Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            chunk_acc_delta_mask_sum =
                Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            info_reward_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
            chunk_step_rewards.clear();
            chunk_step_values.clear();
            chunk_step_log_probs.clear();
            chunk_step_masks.clear();
            chunk_step_dones.clear();

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
    let halt_loss = halt_loss_sum_metrics.div_scalar(total_halt_steps as f32);
    let halt_target_mean = halt_target_sum_metrics
        .clone()
        .div_scalar(total_halt_steps as f32);
    let halt_prob_mean = halt_prob_sum_metrics.div(halt_target_sum_metrics.clamp_min(HALT_EPS));

    let policy_entropy = if steps_done > 0 {
        policy_entropy_sum_metrics
            .clone()
            .div_scalar(steps_done as f32)
            .mean()
    } else {
        zeros.clone()
    };

    let policy_entropy_target = if selectable_steps > 0 {
        selectable_count_sum
            .clone()
            .div_scalar(selectable_steps as f32)
            .clamp_min(1.0)
            .log()
            .mul_scalar(training.policy.entropy_target_scale.max(0.0))
    } else {
        zeros.clone()
    };
    let policy_entropy_alpha = Tensor::<B, 1>::from_data(
        TensorData::new(vec![policy_entropy_weight], [1]),
        &device,
    );

    let policy_recon_term = if policy_recon_weight > 0.0 && steps_done > 0 {

        policy_recon_sum_metrics
            .clone()
            .div_scalar(steps_done as f32)
    } else {
        zeros.clone()
    };

    let log_prob_mean = if steps_done > 0 && gdpo_active {
        log_prob_sum_metrics
            .clone()
            .div_scalar(steps_done as f32)
            .mean()
    } else {
        zeros.clone()
    };

    let action_count = action_count_sum_metrics.clone().clamp_min(SUDOKU_EPS);
    let saccade_revisit_rate = revisit_count_sum_metrics.clone().div(action_count.clone());
    let saccade_repeat_rate = repeat_count_sum_metrics.clone().div(action_count.clone());
    let saccade_unknown_frac = unknown_select_sum_metrics.clone().div(action_count.clone());
    let unique_cells = visit_counts.clone().greater_elem(0.0).float().sum_dim(1);
    let total_visits = visit_counts.clone().sum_dim(1).clamp_min(1.0);
    let saccade_unique_frac = unique_cells.div(total_visits).mean();

    let hard_reward = match hard_mode {
        SudokuHardRewardMode::InfoReward => info_reward_sum.clone(),
        SudokuHardRewardMode::Accuracy => last_reward_acc_per_sample.clone().sub(reward_initial_acc.clone()),
    };
    let easy_reward = match easy_mode {
        SudokuEasyRewardMode::Recon => {
            rollout_recon_delta_sum
                .clone()
                .div(rollout_recon_mask_sum.clone().clamp_min(1.0))
        }
        SudokuEasyRewardMode::AccuracyDelta => {
            rollout_acc_delta_sum
                .clone()
                .div(rollout_acc_delta_mask_sum.clone().clamp_min(1.0))
        }
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
                    easy = easy + shaping_sum.clone().mul_scalar(shaping_weight);
                }
                easy
            }
        }
    };
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale.clone());

    let hard_reward_mean = hard_reward.clone().mean();
    let easy_reward_mean = easy_reward.clone().mean();

    let (policy_loss, advantage_abs_mean, advantage_std) = if gdpo_active {
        if baseline_enabled && !rollout_step_rewards.is_empty() {
            let values_detached: Vec<_> = rollout_step_values
                .iter()
                .map(|value| value.clone().detach())
                .collect();
            let rewards_detached: Vec<_> = rollout_step_rewards
                .iter()
                .map(|reward| reward.clone().detach())
                .collect();
            let dones_detached: Vec<_> = rollout_step_dones
                .iter()
                .map(|done| done.clone().detach())
                .collect();
            let next_value = trainer
                .model
                .value_baseline_from_summary_tokens(summary_tokens.clone())
                .reshape([batch_size.max(1)])
                .detach();
            let advantages = gae_advantages(
                &rewards_detached,
                &values_detached,
                &dones_detached,
                next_value,
                baseline_gamma,
                baseline_lambda,
            );

            let step_count = advantages.len();
            let batch = batch_size.max(1);
            let mut adv_stack: Vec<Tensor<B, 2>> = Vec::with_capacity(step_count);
            let mut log_prob_stack: Vec<Tensor<B, 3>> = Vec::with_capacity(step_count);
            for idx in 0..step_count {
                adv_stack.push(
                    advantages[idx]
                        .clone()
                        .mul(rollout_step_masks[idx].clone())
                        .reshape([1, batch]),
                );
                log_prob_stack.push(
                    rollout_step_log_probs[idx]
                        .clone()
                        .reshape([1, batch, 1]),
                );
            }
            let mut advantage = Tensor::cat(adv_stack, 0)
                .reshape([step_count * batch, 1]);
            if gdpo_group > 1 {
                let scene_batch = if gdpo_group == 0 {
                    batch
                } else {
                    batch / gdpo_group
                };
                let hard = advantage
                    .clone()
                    .reshape([step_count * scene_batch.max(1), gdpo_group.max(1)]);
                let easy = Tensor::<B, 2>::zeros(
                    [step_count * scene_batch.max(1), gdpo_group.max(1)],
                    &device,
                );
                advantage = gdpo_advantage_autodiff::<B>(hard, easy, &training.gdpo)
                    .reshape([step_count * batch, 1]);
            }
            let advantage = apply_advantage_guardrails(advantage, &training.gdpo, None);
            let log_prob = Tensor::cat(log_prob_stack, 0)
                .reshape([step_count * batch, 1]);
            let log_prob_old = log_prob.clone().detach();
            let policy_loss = gdpo_policy_loss(
                log_prob,
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
            let scene_batch = if gdpo_group == 0 {
                batch_size
            } else {
                batch_size / gdpo_group
            };
            let hard = hard_reward
                .clone()
                .reshape([scene_batch.max(1), gdpo_group.max(1)]);
            let easy = easy_reward
                .clone()
                .reshape([scene_batch.max(1), gdpo_group.max(1)]);
            let advantage = gdpo_advantage_autodiff::<B>(hard, easy, &training.gdpo)
                .reshape([batch_size.max(1), 1])
                .detach();
            let advantage = apply_advantage_guardrails(advantage, &training.gdpo, None);
            let log_prob_old = log_prob_sum_metrics.clone();
            let policy_loss = gdpo_policy_loss(
                log_prob_sum_metrics.clone(),
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
        }
    } else {
        (zeros.clone(), zeros.clone(), zeros.clone())
    };

    let entropy_bonus = if policy_entropy_weight > 0.0 {
        policy_entropy.clone().mul_scalar(policy_entropy_weight)
    } else {
        zeros.clone()
    };
    let loss = recon_loss.clone()
        + policy_loss.clone()
        + policy_recon_term.mul_scalar(policy_recon_weight)
        + halt_loss.clone().mul_scalar(training.halt.weight)
        - entropy_bonus;

    let solve_exact = tokens
        .equal(solutions)
        .float()
        .sum_dim(1)
        .reshape([batch_size.max(1)])
        .equal_elem(GRID_LEN as f32)
        .float()
        .mean();

    let losses = SudokuLosses {
        loss: loss.clone(),
        recon_loss,
        acc: last_acc,
        exact_acc: last_exact,
        solve_rate: solve_exact,
        policy_loss,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        advantage_abs_mean,
        advantage_std,
        log_prob_mean,
        policy_entropy,
        policy_entropy_alpha,
        policy_entropy_target,
        hard_reward_mean,
        easy_reward_mean,
        saccade_revisit_rate,
        saccade_repeat_rate,
        saccade_unknown_frac,
        saccade_unique_frac,
    };

    (losses, grads)
}


struct RolloutBase<B: BackendTrait> {
    recon_loss: Tensor<B, 1>,
    acc: Tensor<B, 1>,
    exact_acc: Tensor<B, 1>,
    solve_rate: Tensor<B, 1>,
    hard_reward: Tensor<B, 1>,
    easy_reward: Tensor<B, 1>,
    saccade_revisit_rate: Tensor<B, 1>,
    saccade_repeat_rate: Tensor<B, 1>,
    saccade_unknown_frac: Tensor<B, 1>,
    saccade_unique_frac: Tensor<B, 1>,
    halt_loss: Tensor<B, 1>,
    halt_prob_mean: Tensor<B, 1>,
    halt_target_mean: Tensor<B, 1>,
    #[allow(dead_code)]
    log_prob_sum: Option<Tensor<B, 2>>,
    #[allow(dead_code)]
    policy_entropy_sum: Tensor<B, 2>,
    #[allow(dead_code)]
    policy_steps: usize,
    #[allow(dead_code)]
    selectable_count_sum: Tensor<B, 1>,
    #[allow(dead_code)]
    selectable_steps: usize,
    #[allow(dead_code)]
    batch: usize,
}

#[allow(clippy::too_many_arguments)]
fn rollout_base<B: BackendTrait>(
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
        revisit_min_filled,
        rollout_steps,
    )
}

#[cfg(test)]
#[allow(dead_code, clippy::too_many_arguments)]
fn rollout_base_autodiff<B: AutodiffBackend>(
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
        revisit_min_filled,
        rollout_steps,
    )
}

#[allow(clippy::too_many_arguments)]
fn rollout_base_impl<B: BackendTrait>(
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
    revisit_min_filled: f32,
    rollout_steps: usize,
) -> RolloutBase<B> {
    let rollout_steps = rollout_steps.max(1);
    let recon_loss_interval = training.recon.loss_interval_steps;
    let global_weight = training.recon.global_loss_weight.max(0.0);
    let global_samples = training.recon.global_loss_samples.min(GRID_LEN);
    let visit_penalty = training.policy.visit_penalty.max(0.0);
    let gdpo_active = training.gdpo.enabled && track_policy;

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
    let initial_unknown_counts = unknown_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1)]);
    let difficulty_scale =
        difficulty_scale_from_unknowns(initial_unknown_counts.clone(), training.reward.unknown_power);
    let easy_mode = training.reward.easy_mode;
    let hard_mode = training.reward.hard_mode;
    let info_enabled = training.reward.info_reward.enabled
        && matches!(hard_mode, SudokuHardRewardMode::InfoReward);
    let info_stride = training.reward.info_reward.stride.max(1);
    let shaping_enabled = training.reward.shaping.enabled && !gdpo_active;
    let shaping_metric = training.reward.shaping.metric;
    let shaping_weight = training.reward.shaping.weight.max(0.0);
    let shaping_gamma = training.reward.shaping.gamma.clamp(0.0, 1.0);
    let baseline_enabled = training.reward.baseline.enabled;
    let baseline_gamma = training.reward.baseline.gamma.clamp(0.0, 1.0);
    let baseline_lambda = training.reward.baseline.lambda.clamp(0.0, 1.0);
    let _baseline_value_weight = training.reward.baseline.value_loss_weight.max(0.0);
    let no_op_penalty = training.reward.no_op_penalty.max(0.0);
    let mut shaping_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut conflict_prev = if shaping_enabled {
        shaping_potential(&tokens, shaping_metric)
    } else {
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device)
    };
    let mut rollout_step_rewards: Vec<Tensor<B, 1>> = Vec::new();
    let mut rollout_step_values: Vec<Tensor<B, 1>> = Vec::new();

    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

    let mut state = model.init_state();
    let input_cache = model.cell_embeddings_with_positions(
        tokens.clone(),
        row_ids.clone(),
        col_ids.clone(),
    );
    let [_, _, embd] = input_cache.shape().dims();
    let cache_streams = model.cache_streams();
    let input_cache = input_cache
        .unsqueeze_dim::<4>(1)
        .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
    let input_cache_read = input_cache
        .clone()
        .mean_dim(1)
        .reshape([batch_size.max(1), GRID_LEN, embd]);
    let mut cache = input_cache.clone();
    let mut summary_tokens = model.init_summary_tokens(batch_size);
    let summary_len = model.summary_token_count();
    let (reward_initial_acc_per_sample, initial_acc_mean, initial_exact, initial_solve_rate) =
        compute_grid_accuracy(&tokens_reward, &solutions);
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
    let mut chunk_recon_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_recon_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_recon_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut chunk_acc_delta_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut rollout_acc_delta_mask_sum =
        Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut info_reward_sum = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut halt_loss_sum = zeros.clone();
    let mut halt_prob_sum = zeros.clone();
    let mut halt_target_sum = zeros.clone();

    let mut policy_entropy_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut log_prob_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut policy_steps = 0usize;
    let mut action_count_sum = zeros.clone();
    let mut revisit_count_sum = zeros.clone();
    let mut repeat_count_sum = zeros.clone();
    let mut unknown_select_sum = zeros.clone();
    let mut visit_counts =
        Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut prev_action_one_hot =
        Tensor::<B, 2>::zeros([batch_size.max(1), GRID_LEN], &device);
    let mut selectable_count_sum = zeros.clone();
    let mut selectable_steps = 0usize;

    for step_idx in 0..rollout_steps {
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

        let selectable_counts = select_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
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
        let actions = sample_actions(sampled_logits, select_mask.clone(), train_mode, policy_epsilon);
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
        action_count_sum = action_count_sum + action_any.mean().detach();
        revisit_count_sum = revisit_count_sum + step_revisit.mean().detach();
        repeat_count_sum = repeat_count_sum + step_repeat.mean().detach();
        unknown_select_sum = unknown_select_sum + step_unknown.mean().detach();
        prev_action_one_hot = action_one_hot.clone();
        visit_counts = visit_counts + action_one_hot.clone();

        if track_policy {
            let log_probs = activation::log_softmax(policy_logits, 1);
            let selected_log_prob = log_probs
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .mul(active_mask.clone());
            log_prob_sum = log_prob_sum + selected_log_prob;

            let step_entropy = log_probs
                .clone()
                .exp()
                .mul(log_probs)
                .sum_dim(1)
                .reshape([batch_size.max(1), 1])
                .mul_scalar(-1.0)
                .mul(active_mask.clone());
            policy_entropy_sum = policy_entropy_sum + step_entropy;
            policy_steps += 1;
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
        let step_input = model
            .project_input_tokens(step_input_base)
            + step_residual;
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_input], 1);
        let (step_hidden, _step_logits_full) =
            model.forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = summary_hidden.clone();
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
        let reward_mask = selected_editable
            .clone()
            .reshape([batch_size.max(1)]);
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

        let reward_local_mask = active_mask
            .clone()
            .mul(selected_editable.clone());
        let (
            _reward_local_loss,
            _reward_local_acc,
            _reward_local_exact,
            reward_local_loss_per_sample,
            ..
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

        let update_mask = action_one_hot
            .clone()
            .mul(editable_mask.clone())
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
        let cache_cell = cache
            .clone()
            .mul(action_mask.clone())
            .sum_dim(2)
            .reshape([batch_size.max(1) * cache_streams, 1, embd]);
        let summary_streams = summary_hidden
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
        let update_emb = model.update_cell_embedding(
            summary_streams,
            cache_cell,
            token_emb,
        );
        let update_emb = update_emb
            .reshape([batch_size.max(1), cache_streams, 1, embd])
            .expand([batch_size.max(1), cache_streams, GRID_LEN, embd]);
        let update_mask_stream = update_mask_f.clone().unsqueeze_dim::<4>(1);
        let keep = update_mask_stream.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_stream);
        if let Some(mhc) = model.cache_mhc.as_ref() {
            let (branch_input, residuals_out, beta) = mhc.width_connection(cache.clone());
            cache = mhc.depth_connection(branch_input, residuals_out, beta);
        }

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

        let (reward_acc_per_sample, acc, exact_acc, solve_rate) =
            compute_grid_accuracy(&tokens_reward, &solutions);
        let step_acc_delta = reward_acc_per_sample.clone().sub(prev_reward_acc_per_sample.clone());
        prev_reward_acc_per_sample = reward_acc_per_sample.detach();
        last_reward_acc_per_sample = prev_reward_acc_per_sample.clone();
        last_acc = acc;
        last_exact = exact_acc;
        last_solve_rate = solve_rate;

        let mut global_loss = zeros.clone();
        let mut global_loss_per_sample =
            Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        let mut reward_global_loss_per_sample =
            Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
        if recon_loss_interval > 0
            && (step_idx + 1) % recon_loss_interval == 0
            && global_samples > 0
        {
            let sample_prob =
                (global_samples as f32 / GRID_LEN as f32).clamp(0.0, 1.0);
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
                reward_global_mask = reward_global_mask.mul(tokens_reward.clone().equal_elem(0).float());
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
                ..
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
        let step_recon_per_sample_reward = reward_local_loss_per_sample
            + reward_global_loss_per_sample.mul_scalar(global_weight);
        let step_recon_per_sample_reward_detached =
            step_recon_per_sample_reward.clone().detach();
        let mut baseline_loss = last_loss_per_sample.clone();
        let baseline_mask = baseline_loss.clone().equal_elem(0.0);
        baseline_loss = baseline_loss.mask_where(
            baseline_mask,
            step_recon_per_sample_reward_detached.clone(),
        );
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
        chunk_recon_delta_sum =
            chunk_recon_delta_sum + step_recon_delta_reward.clone();
        chunk_recon_mask_sum = chunk_recon_mask_sum + step_reward_mask.clone();
        rollout_recon_delta_sum =
            rollout_recon_delta_sum + step_recon_delta_reward.clone();
        rollout_recon_mask_sum = rollout_recon_mask_sum + step_reward_mask.clone();
        let step_active_mask = active_mask.clone().reshape([batch_size.max(1)]);
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
            let conflict_next = shaping_potential(&tokens, shaping_metric);
            let mut shaping_delta =
                conflict_next.clone().mul_scalar(shaping_gamma) - conflict_prev.clone();
            shaping_delta = shaping_delta.mul(reward_mask.clone());
            shaping_sum = shaping_sum + shaping_delta.clone();
            step_reward = step_reward + shaping_delta.mul_scalar(shaping_weight);
            let keep = Tensor::<B, 1>::ones([batch_size.max(1)], &device)
                .sub(reward_mask.clone());
            conflict_prev = conflict_prev.mul(keep) + conflict_next.mul(reward_mask.clone());
        }
        step_reward = step_reward.mul(step_reward_mask.clone());
        if baseline_enabled && let Some(value) = step_value.clone() {
            rollout_step_rewards.push(step_reward.clone());
            rollout_step_values.push(value);
        }

        let reward_keep = Tensor::<B, 1>::ones([batch_size.max(1)], &device)
            .sub(reward_mask.clone());
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

    let log_prob_sum = if track_policy { Some(log_prob_sum) } else { None };
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
        SudokuHardRewardMode::Accuracy => last_reward_acc_per_sample.clone().sub(reward_initial_acc.clone()),
    };
    let easy_reward = match easy_mode {
        SudokuEasyRewardMode::Recon => {
            rollout_recon_delta_sum
                .clone()
                .div(rollout_recon_mask_sum.clone().clamp_min(1.0))
        }
        SudokuEasyRewardMode::AccuracyDelta => {
            rollout_acc_delta_sum
                .clone()
                .div(rollout_acc_delta_mask_sum.clone().clamp_min(1.0))
        }
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
                    easy = easy + shaping_sum.clone().mul_scalar(shaping_weight);
                }
                easy
            }
        }
    };
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale.clone());

    let action_count = action_count_sum.clone().clamp_min(SUDOKU_EPS);
    let saccade_revisit_rate = revisit_count_sum.clone().div(action_count.clone());
    let saccade_repeat_rate = repeat_count_sum.clone().div(action_count.clone());
    let saccade_unknown_frac = unknown_select_sum.clone().div(action_count.clone());
    let unique_cells = visit_counts.clone().greater_elem(0.0).float().sum_dim(1);
    let total_visits = visit_counts.clone().sum_dim(1).clamp_min(1.0);
    let saccade_unique_frac = unique_cells.div(total_visits).mean();

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

fn compute_grid_accuracy<B: BackendTrait>(
    tokens: &Tensor<B, 2, Int>,
    solutions: &Tensor<B, 2, Int>,
) -> (Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>, Tensor<B, 1>) {
    let [batch, time] = tokens.shape().dims();
    let device = tokens.device();
    if batch == 0 || time == 0 {
        let zeros = Tensor::<B, 1>::zeros([1], &device);
        return (
            zeros.clone(),
            zeros.clone(),
            zeros.clone(),
            zeros,
        );
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

#[allow(clippy::type_complexity)]
fn compute_loss_and_acc<B: BackendTrait>(
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
    let random_scores = random_scores * select_mask.clone()
        + select_mask.clone().sub_scalar(1.0).mul_scalar(2.0);
    let random_actions = random_scores.argmax(1);
    let gate = Tensor::<B, 2>::random(
        [batch, 1],
        TensorDistribution::Uniform(0.0, 1.0),
        &device,
    )
    .lower_equal_elem(policy_epsilon);

    gumbel_actions.mask_where(gate, random_actions)
}

fn build_solution_one_hot<B: BackendTrait>(
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

fn build_action_one_hot<B: BackendTrait>(
    actions: &Tensor<B, 2, Int>,
    action_index: &Tensor<B, 2, Int>,
) -> Tensor<B, 2> {
    let [batch, grid] = action_index.shape().dims::<2>();
    let expanded = actions.clone().expand([batch, grid]);
    expanded.equal(action_index.clone()).float()
}

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
        let solutions = Tensor::<B, 2, Int>::from_data(
            TensorData::new(solutions, [batch, time]),
            &device,
        );
        let solution_one_hot = build_solution_one_hot(&solutions, batch, &device);

        let mut logits = vec![0f32; batch * time * VOCAB_SIZE];
        for idx in 0..time {
            let target = (idx % 9) + 1;
            logits[idx * VOCAB_SIZE + target] = 10.0;
        }
        let logits = Tensor::<B, 3>::from_data(
            TensorData::new(logits, [batch, time, VOCAB_SIZE]),
            &device,
        );

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
        assert!((acc_value - 1.0).abs() < 1e-6, "acc not perfect: {}", acc_value);
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
        let select_mask = Tensor::<B, 2>::from_data(
            TensorData::new(mask_data, [2, 4]),
            &device,
        );
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
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
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
            &model,
            batch,
            &training,
            0.0,
            0.0,
            false,
            0.0,
            1.0,
            0.0,
            0.0,
            None,
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
            },
            validation: SudokuValidationConfig::default(),
            gdpo: GdpoConfig {
                enabled: false,
                group_size: 1,
                ..GdpoConfig::default()
            },
        };

        let filled = vec![1i64; 2 * GRID_LEN];
        let puzzles = Tensor::<B, 2, Int>::from_data(
            TensorData::new(filled.clone(), [2, GRID_LEN]),
            &device,
        );
        let solutions = Tensor::<B, 2, Int>::from_data(
            TensorData::new(filled, [2, GRID_LEN]),
            &device,
        );
        let batch = SudokuBatch::new(puzzles, solutions);

        let losses = rollout_losses_valid(&model, batch, &training);
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
            TensorData::new(
                vec![0.0, 40.0, GRID_LEN as f32],
                [3],
            ),
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


























































