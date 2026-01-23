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

fn rollout_max_steps(training: &SudokuTrainingHyperparameters) -> usize {
    if training.rollout_max_steps > 0 {
        training.rollout_max_steps
    } else {
        training.rollout_steps.max(1)
    }
}

fn rollout_min_steps(training: &SudokuTrainingHyperparameters) -> usize {
    let max_steps = rollout_max_steps(training);
    let min_steps = if training.rollout_min_steps > 0 {
        training.rollout_min_steps
    } else {
        max_steps
    };
    min_steps.min(max_steps).max(1)
}

fn rollout_bounds(training: &SudokuTrainingHyperparameters, step: usize) -> (usize, usize) {
    let min_steps = rollout_min_steps(training);
    let mut max_steps = rollout_max_steps(training);
    if training.rollout_max_steps_warmup_iters > 0
        && step < training.rollout_max_steps_warmup_iters
        && training.rollout_max_steps_warmup_cap > 0
    {
        max_steps = max_steps.min(training.rollout_max_steps_warmup_cap);
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

#[derive(Clone, Debug)]
pub struct SudokuTrainer<B: BackendTrait> {
    pub model: SudokuSaccadeModel<B>,
    pub training: SudokuTrainingHyperparameters,
    pub total_steps: usize,
    step_counter: Arc<AtomicUsize>,
    gdpo_stats: Arc<Mutex<GdpoAdvantageStats<B>>>,
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
        Self {
            model,
            training,
            total_steps: total_steps.max(1),
            step_counter: Arc::new(AtomicUsize::new(0)),
            gdpo_stats: Arc::new(Mutex::new(GdpoAdvantageStats::default())),
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
            self.training.teacher_forcing_prob,
            self.training.teacher_forcing_final,
            self.training.teacher_forcing_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);
        let policy_temperature = schedule_linear(
            self.training.policy_temperature,
            self.training.policy_temperature_final,
            self.training.policy_temperature_anneal_steps,
            step,
        )
        .max(1e-4);
        let policy_entropy_weight = schedule_linear(
            self.training.policy_entropy_weight,
            self.training.policy_entropy_weight_final,
            self.training.policy_entropy_anneal_steps,
            step,
        )
        .max(0.0);
        let policy_recon_weight = self.training.policy_recon_weight.max(0.0);
        let revisit_min_filled = schedule_linear(
            self.training.revisit_min_filled_frac,
            self.training.revisit_min_filled_final,
            self.training.revisit_min_filled_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);
        let rollout_steps = sample_rollout_steps(&self.training, step);
        let (losses, grads) = rollout_losses_train::<B>(
            self,
            batch,
            self.training.policy_noise,
            self.training.gdpo.enabled,
            teacher_forcing_prob,
            policy_temperature,
            policy_entropy_weight,
            policy_recon_weight,
            revisit_min_filled,
            rollout_steps,
            Some(Arc::clone(&self.gdpo_stats)),
        );
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
            losses.hard_reward_mean,
            losses.easy_reward_mean,
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
            losses.hard_reward_mean,
            losses.easy_reward_mean,
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
    pub hard_reward_mean: Tensor<B, 1>,
    pub easy_reward_mean: Tensor<B, 1>,
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

fn tensor_to_f32<B: BackendTrait>(value: Tensor<B, 1>) -> f32 {
    value
        .into_data()
        .iter::<f32>()
        .next()
        .unwrap_or(0.0)
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

#[cfg(test)]
fn rollout_losses<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
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
        + halt_loss.clone().mul_scalar(training.halt_weight)
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
        training,
        0.0,
        None,
        false,
        false,
        false,
        0.0,
        1.0,
        training.revisit_min_filled_final,
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
        hard_reward_mean: rollout.hard_reward.mean(),
        easy_reward_mean: rollout.easy_reward.mean(),
    }
}

#[allow(clippy::too_many_arguments)]
fn rollout_losses_train<B: AutodiffBackend>(
    trainer: &SudokuTrainer<B>,
    batch: SudokuBatch<B>,
    policy_noise: f32,
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
    let backprop_steps_cfg = training.rollout_backprop_steps.unwrap_or(rollout_steps);
    let chunk_steps = if backprop_steps_cfg == 0 {
        rollout_steps
    } else {
        backprop_steps_cfg.min(rollout_steps).max(1)
    };
    let recon_interval = training.recon_loss_interval_steps;
    let total_local_steps = rollout_steps.max(1);
    let total_global_steps = recon_step_count(rollout_steps, recon_interval);
    let total_halt_steps = rollout_steps.max(1);
    let total_policy_steps = rollout_steps.max(1);
    let global_weight = training.global_loss_weight.max(0.0);

    let gdpo_group = training.gdpo.group_size.max(1);
    let repeat_for_gdpo = gdpo_active && gdpo_group > 1;
    let global_samples = training.global_loss_samples.min(GRID_LEN);

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
    let initial_unknown_counts = unknown_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1)]);
    let difficulty_scale =
        difficulty_scale_from_unknowns(initial_unknown_counts.clone(), training.reward_unknown_power);

    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

    let mut state = trainer.model.init_state();
    let mut cache = trainer.model.cell_embeddings_with_positions(
        tokens.clone(),
        row_ids.clone(),
        col_ids.clone(),
    );
    let mut summary_tokens = trainer.model.init_summary_tokens(batch_size);
    let summary_len = trainer.model.summary_token_count();
    let (initial_acc_per_sample, initial_acc_mean, initial_exact, _initial_solve) =
        compute_grid_accuracy(&tokens, &solutions);
    let initial_acc_per_sample = initial_acc_per_sample.detach();
    let initial_acc = initial_acc_per_sample.clone();
    let mut last_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut last_acc = initial_acc_mean.detach();
    let mut last_exact = initial_exact.detach();
    let mut last_acc_per_sample = initial_acc_per_sample.clone();

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

    let mut chunk_local_loss_sum = zeros.clone();
    let mut chunk_global_loss_sum = zeros.clone();
    let mut chunk_halt_loss_sum = zeros.clone();
    let mut chunk_halt_steps = 0usize;
    let mut chunk_policy_entropy_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut chunk_log_prob_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut chunk_policy_recon_sum = zeros.clone();

    let mut steps_done = 0usize;

    let mut grads_accum = GradientsAccumulator::<SudokuTrainer<B>>::new();

    for step_idx in 0..rollout_steps {
        let policy_logits =
            trainer
                .model
                .policy_logits_from_cache(summary_tokens.clone(), cache.clone());

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
        let masked_logits = policy_logits
            - ones_grid
                .clone()
                .sub(select_mask.clone())
                .mul_scalar(POLICY_MASK_PENALTY);

        let selectable_counts = select_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
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
        let actions = sample_actions(sampled_logits, true);
        let mut action_one_hot = build_action_one_hot(&actions, &action_index);
        let active_mask_grid = active_mask.clone().repeat_dim(1, GRID_LEN);
        action_one_hot = action_one_hot * active_mask_grid.clone() * select_mask.clone();

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

        let step_mask = if training.saccade_step_cells > 1 {
            let selected_row = row_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            let selected_col = col_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            let row_match = row_ids_f
                .clone()
                .equal(selected_row.expand([batch_size.max(1), GRID_LEN]))
                .float();
            let col_match = col_ids_f
                .clone()
                .equal(selected_col.expand([batch_size.max(1), GRID_LEN]))
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
            .reshape([batch_size.max(1), 1])
            .clamp_min(1.0);
        let [_, _, embd] = cache.shape().dims();
        let step_emb = cache
            .clone()
            .mul(step_mask.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .div(step_mask_sum.clone().reshape([batch_size.max(1), 1, 1]))
            .reshape([batch_size.max(1), 1, embd]);
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_emb], 1);
        let (step_hidden, _step_logits_full) = trainer
            .model
            .forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = summary_hidden;
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
        let mut local_loss_mask = active_mask.clone();
        if matches!(training.loss_mask, SudokuLossMask::Unknown) {
            let selected_unknown = loss_unknown_mask
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            local_loss_mask = local_loss_mask.mul(selected_unknown);
        }
        let (local_loss, _local_acc, _local_exact, local_loss_per_sample, ..) =
            compute_loss_and_acc(
                &step_logits,
                &selected_solution,
                &selected_solution_one_hot,
                &local_loss_mask,
                &training.recon_loss,
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
        let update_values_grid = update_values.clone().repeat_dim(1, GRID_LEN);
        tokens = tokens.mask_where(update_mask.clone(), update_values_grid);
        unknown_mask = (unknown_mask - action_one_hot.clone()).clamp_min(0.0);

        let update_mask_f = action_one_hot
            .clone()
            .mul(editable_mask.clone())
            .unsqueeze_dim::<3>(2);
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
        let update_emb = trainer.model.cell_embeddings_with_positions(
            update_values.clone(),
            selected_row,
            selected_col,
        );
        let update_emb = update_emb.expand([batch_size.max(1), GRID_LEN, embd]);
        let keep = update_mask_f.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_f);

        let tokens_solved_after = tokens
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
        if step_idx + 1 < training.halt_min_steps {
            step_halt = step_halt.mul_scalar(0.0);
        }
        if training.halt_exploration_prob > 0.0 {
            let explore = Tensor::<B, 2>::random(
                [batch_size.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(training.halt_exploration_prob)
            .float();
            step_halt = step_halt.mul(ones_step.clone().sub(explore));
        }
        halted = halted.max_pair(step_halt.clone());

        steps_done += 1;

        let (acc_per_sample, acc, exact_acc, _solve_rate) =
            compute_grid_accuracy(&tokens, &solutions);
        last_acc_per_sample = acc_per_sample.detach();
        last_acc = acc.detach();
        last_exact = exact_acc.detach();

        let mut global_loss = zeros.clone();
        let mut global_loss_per_sample =
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
            let mut global_mask = random.lower_equal_elem(sample_prob).float();
            global_mask = (global_mask + action_one_hot.clone()).clamp_max(1.0);
            global_mask = global_mask.mul(active_mask_grid.clone());
            if matches!(training.loss_mask, SudokuLossMask::Unknown) {
                global_mask = global_mask.mul(loss_unknown_mask.clone());
            }
            let global_logits = trainer.model.value_logits_from_cache(cache.clone());
            let (step_loss, _step_acc, _step_exact, step_loss_per_sample, ..) =
                compute_loss_and_acc(
                    &global_logits,
                    &solutions,
                    &solution_one_hot,
                    &global_mask,
                    &training.recon_loss,
                );
            global_loss = step_loss;
            global_loss_per_sample = step_loss_per_sample;
        }

        let step_recon_per_sample =
            local_loss_per_sample + global_loss_per_sample.mul_scalar(global_weight);
        if policy_recon_weight > 0.0 {
            let reward = step_recon_per_sample.clone().mul_scalar(-1.0).detach();
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

        last_loss_per_sample = step_recon_per_sample.detach();

        let step_recon = local_loss.clone() + global_loss.clone().mul_scalar(global_weight);
        recon_loss_sum_metrics = recon_loss_sum_metrics + step_recon.detach();
        recon_steps_metrics += 1;
        chunk_local_loss_sum = chunk_local_loss_sum + local_loss;
        if global_active {
            chunk_global_loss_sum = chunk_global_loss_sum + global_loss;
        }

        let chunk_end = (step_idx + 1) % chunk_steps == 0 || step_idx + 1 == rollout_steps;
        if chunk_end {
            let hard_reward = last_acc_per_sample.clone().sub(initial_acc.clone());
            let easy_reward = last_loss_per_sample.clone().mul_scalar(-1.0);
            let hard_reward = hard_reward.mul(difficulty_scale.clone());
            let easy_reward = easy_reward.mul(difficulty_scale.clone());

            let policy_loss = if gdpo_active {
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
                let advantage =
                    apply_advantage_guardrails(advantage, &training.gdpo, gdpo_stats.as_ref());
                let log_prob_old = chunk_log_prob_sum.clone().detach();
                gdpo_policy_loss(
                    chunk_log_prob_sum.clone(),
                    log_prob_old,
                    advantage,
                    &training.gdpo,
                )
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
                + halt_term.mul_scalar(training.halt_weight)
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

    let hard_reward = last_acc_per_sample.clone().sub(initial_acc);
    let easy_reward = last_loss_per_sample.clone().mul_scalar(-1.0);
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale.clone());

    let hard_reward_mean = hard_reward.clone().mean();
    let easy_reward_mean = easy_reward.clone().mean();

    let (policy_loss, advantage_abs_mean, advantage_std) = if gdpo_active {
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
        + halt_loss.clone().mul_scalar(training.halt_weight)
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
        hard_reward_mean,
        easy_reward_mean,
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
    batch: usize,
}

#[allow(clippy::too_many_arguments)]
fn rollout_base<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    _force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = rollout_max_steps(training);
    rollout_base_impl(
        model,
        batch,
        training,
        policy_noise,
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
fn rollout_base_autodiff<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    _force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = rollout_max_steps(training);
    rollout_base_impl(
        model,
        batch,
        training,
        policy_noise,
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
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
    rollout_steps: usize,
) -> RolloutBase<B> {
    let rollout_steps = rollout_steps.max(1);
    let recon_loss_interval = training.recon_loss_interval_steps;
    let global_weight = training.global_loss_weight.max(0.0);
    let global_samples = training.global_loss_samples.min(GRID_LEN);

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
    let initial_unknown_counts = unknown_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1)]);
    let difficulty_scale =
        difficulty_scale_from_unknowns(initial_unknown_counts.clone(), training.reward_unknown_power);

    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

    let mut state = model.init_state();
    let mut cache = model.cell_embeddings_with_positions(
        tokens.clone(),
        row_ids.clone(),
        col_ids.clone(),
    );
    let mut summary_tokens = model.init_summary_tokens(batch_size);
    let summary_len = model.summary_token_count();
    let (initial_acc_per_sample, initial_acc_mean, initial_exact, initial_solve_rate) =
        compute_grid_accuracy(&tokens, &solutions);
    let initial_acc = initial_acc_per_sample.clone();
    let mut last_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut last_acc = initial_acc_mean;
    let mut last_exact = initial_exact;
    let mut last_solve_rate = initial_solve_rate;
    let mut last_acc_per_sample = initial_acc.clone();

    let zeros = Tensor::<B, 1>::zeros([1], &device);
    let mut recon_loss_sum = zeros.clone();
    let mut recon_steps = 0usize;
    let mut halt_loss_sum = zeros.clone();
    let mut halt_prob_sum = zeros.clone();
    let mut halt_target_sum = zeros.clone();

    let mut policy_entropy_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut log_prob_sum =
        Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);
    let mut policy_steps = 0usize;

    for step_idx in 0..rollout_steps {
        let policy_logits =
            model.policy_logits_from_cache(summary_tokens.clone(), cache.clone());

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
        let masked_logits = policy_logits
            - ones_grid
                .clone()
                .sub(select_mask.clone())
                .mul_scalar(POLICY_MASK_PENALTY);

        let selectable_counts = select_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
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
        let actions = sample_actions(sampled_logits, train_mode);
        let mut action_one_hot = build_action_one_hot(&actions, &action_index);
        let active_mask_grid = active_mask.clone().repeat_dim(1, GRID_LEN);
        action_one_hot = action_one_hot * active_mask_grid.clone() * select_mask.clone();

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

        let step_mask = if training.saccade_step_cells > 1 {
            let selected_row = row_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            let selected_col = col_ids_f
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            let row_match = row_ids_f
                .clone()
                .equal(selected_row.expand([batch_size.max(1), GRID_LEN]))
                .float();
            let col_match = col_ids_f
                .clone()
                .equal(selected_col.expand([batch_size.max(1), GRID_LEN]))
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
            .reshape([batch_size.max(1), 1])
            .clamp_min(1.0);
        let [_, _, embd] = cache.shape().dims();
        let step_emb = cache
            .clone()
            .mul(step_mask.unsqueeze_dim::<3>(2))
            .sum_dim(1)
            .div(step_mask_sum.clone().reshape([batch_size.max(1), 1, 1]))
            .reshape([batch_size.max(1), 1, embd]);
        let step_input = Tensor::cat(vec![summary_tokens.clone(), step_emb], 1);
        let (step_hidden, _step_logits_full) =
            model.forward_with_hidden_and_state_embedded(step_input, &mut state);
        let summary_hidden = step_hidden.clone().slice_dim(1, 0..summary_len);
        let step_hidden = step_hidden.slice_dim(1, summary_len..summary_len + 1);
        summary_tokens = summary_hidden;
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
        let mut local_loss_mask = active_mask.clone();
        if matches!(training.loss_mask, SudokuLossMask::Unknown) {
            let selected_unknown = loss_unknown_mask
                .clone()
                .mul(action_one_hot.clone())
                .sum_dim(1)
                .reshape([batch_size.max(1), 1]);
            local_loss_mask = local_loss_mask.mul(selected_unknown);
        }
        let (local_loss, _local_acc, _local_exact, local_loss_per_sample, ..) =
            compute_loss_and_acc(
                &step_logits,
                &selected_solution,
                &selected_solution_one_hot,
                &local_loss_mask,
                &training.recon_loss,
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
        let update_values_grid = update_values.clone().repeat_dim(1, GRID_LEN);
        tokens = tokens.mask_where(update_mask.clone(), update_values_grid);
        unknown_mask = (unknown_mask - action_one_hot.clone()).clamp_min(0.0);

        let update_mask_f = action_one_hot
            .clone()
            .mul(editable_mask.clone())
            .unsqueeze_dim::<3>(2);
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
        let update_emb =
            model.cell_embeddings_with_positions(update_values.clone(), selected_row, selected_col);
        let update_emb = update_emb.expand([batch_size.max(1), GRID_LEN, embd]);
        let keep = update_mask_f.clone().mul_scalar(-1.0).add_scalar(1.0);
        cache = cache * keep + update_emb.mul(update_mask_f);

        let tokens_solved_after = tokens
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
        if step_idx + 1 < training.halt_min_steps {
            step_halt = step_halt.mul_scalar(0.0);
        }
        if training.halt_exploration_prob > 0.0 {
            let explore = Tensor::<B, 2>::random(
                [batch_size.max(1), 1],
                TensorDistribution::Uniform(0.0, 1.0),
                &device,
            )
            .lower_equal_elem(training.halt_exploration_prob)
            .float();
            step_halt = step_halt.mul(ones_step.clone().sub(explore));
        }
        halted = halted.max_pair(step_halt.clone());

        let (acc_per_sample, acc, exact_acc, solve_rate) =
            compute_grid_accuracy(&tokens, &solutions);
        last_acc_per_sample = acc_per_sample;
        last_acc = acc;
        last_exact = exact_acc;
        last_solve_rate = solve_rate;

        let mut global_loss = zeros.clone();
        let mut global_loss_per_sample =
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
            let mut global_mask = random.lower_equal_elem(sample_prob).float();
            global_mask = (global_mask + action_one_hot.clone()).clamp_max(1.0);
            global_mask = global_mask.mul(active_mask_grid.clone());
            if matches!(training.loss_mask, SudokuLossMask::Unknown) {
                global_mask = global_mask.mul(loss_unknown_mask.clone());
            }
            let global_logits = model.value_logits_from_cache(cache.clone());
            let (step_loss, _step_acc, _step_exact, step_loss_per_sample, ..) =
                compute_loss_and_acc(
                    &global_logits,
                    &solutions,
                    &solution_one_hot,
                    &global_mask,
                    &training.recon_loss,
                );
            global_loss = step_loss;
            global_loss_per_sample = step_loss_per_sample;
        }

        last_loss_per_sample =
            local_loss_per_sample + global_loss_per_sample.mul_scalar(global_weight);

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

    let hard_reward = last_acc_per_sample.clone().sub(initial_acc);
    let easy_reward = last_loss_per_sample.clone().mul_scalar(-1.0);
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale.clone());

    RolloutBase {
        recon_loss,
        acc: last_acc,
        exact_acc: last_exact,
        solve_rate: last_solve_rate,
        hard_reward,
        easy_reward,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        log_prob_sum,
        policy_entropy_sum,
        policy_steps,
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

fn sample_actions<B: BackendTrait>(
    logits: Tensor<B, 2>,
    train_mode: bool,
) -> Tensor<B, 2, Int> {
    if !train_mode {
        return logits.argmax(1);
    }

    let device = logits.device();
    let [batch, time] = logits.shape().dims();
    if batch == 0 || time == 0 {
        return Tensor::<B, 2, Int>::zeros([batch.max(1), 1], &device);
    }

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
    let gumbel_logits = logits + gumbel;
    gumbel_logits.argmax(1)
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
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
            },
            &device,
        );

        let training = SudokuTrainingHyperparameters {
            batch_size: 2,
            epochs: None,
            max_iters: 1,
            log_frequency: 1,
            rollout_steps: 2,
            rollout_min_steps: 0,
            rollout_max_steps: 0,
            rollout_max_steps_warmup_iters: 0,
            rollout_max_steps_warmup_cap: 0,
            rollout_backprop_steps: None,
            halt_weight: 0.1,
            halt_exploration_prob: 0.0,
            halt_min_steps: 1,
            policy_noise: 0.0,
            teacher_forcing_prob: 0.0,
            teacher_forcing_final: 0.0,
            teacher_forcing_anneal_steps: 0,
            policy_temperature: 1.0,
            policy_temperature_final: 1.0,
            policy_temperature_anneal_steps: 0,
            policy_entropy_weight: 0.0,
            policy_entropy_weight_final: 0.0,
            policy_entropy_anneal_steps: 0,
            policy_recon_weight: 0.0,
            revisit_min_filled_frac: 0.0,
            revisit_min_filled_final: 0.0,
            revisit_min_filled_anneal_steps: 0,
            reward_unknown_power: 0.0,
            saccade_step_cells: 1,
            recon_loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            recon_loss_interval_steps: 1,
            global_loss_samples: 8,
            global_loss_weight: 0.2,
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
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
            },
            &device,
        );

        let training = SudokuTrainingHyperparameters {
            batch_size: 2,
            epochs: None,
            max_iters: 1,
            log_frequency: 1,
            rollout_steps: 2,
            rollout_min_steps: 0,
            rollout_max_steps: 0,
            rollout_max_steps_warmup_iters: 0,
            rollout_max_steps_warmup_cap: 0,
            rollout_backprop_steps: None,
            halt_weight: 0.1,
            halt_exploration_prob: 0.0,
            halt_min_steps: 1,
            policy_noise: 0.0,
            teacher_forcing_prob: 0.0,
            teacher_forcing_final: 0.0,
            teacher_forcing_anneal_steps: 0,
            policy_temperature: 1.0,
            policy_temperature_final: 1.0,
            policy_temperature_anneal_steps: 0,
            policy_entropy_weight: 0.0,
            policy_entropy_weight_final: 0.0,
            policy_entropy_anneal_steps: 0,
            policy_recon_weight: 0.0,
            revisit_min_filled_frac: 0.0,
            revisit_min_filled_final: 0.0,
            revisit_min_filled_anneal_steps: 0,
            reward_unknown_power: 0.0,
            saccade_step_cells: 1,
            recon_loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            recon_loss_interval_steps: 1,
            global_loss_samples: 8,
            global_loss_weight: 0.2,
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
