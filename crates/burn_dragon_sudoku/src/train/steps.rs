use crate::train::metrics::{SudokuOutput, SudokuTrainItem};
use crate::train::prelude::*;
use crate::vocab::{GRID_LEN, VOCAB_SIZE};
use burn_dragon_train::train::gdpo::{gdpo_advantage_autodiff, gdpo_policy_loss};
use std::sync::Mutex;

const SUDOKU_EPS: f32 = 1e-6;
const POLICY_MASK_PENALTY: f32 = 1e9;
const HALT_EPS: f32 = 1e-6;
const STABLEMAX_EPS: f32 = 1e-6;
const GUMBEL_EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct SudokuTrainer<B: BackendTrait> {
    pub model: SudokuSaccadeModel<B>,
    pub training: SudokuTrainingHyperparameters,
    pub total_steps: usize,
    step_counter: Arc<AtomicUsize>,
    gdpo_stats: Arc<Mutex<GdpoAdvantageStats<B>>>,
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
        }
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
        }
    }

    fn to_device(self, device: &B::Device) -> Self {
        Self {
            model: self.model.to_device(device),
            training: self.training,
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
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
        }
    }

    fn load_record(self, record: Self::Record) -> Self {
        Self {
            model: self.model.load_record(record),
            training: self.training,
            total_steps: self.total_steps,
            step_counter: Arc::clone(&self.step_counter),
            gdpo_stats: Arc::clone(&self.gdpo_stats),
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
            total_steps: self.total_steps,
            step_counter: Arc::new(AtomicUsize::new(0)),
            gdpo_stats: Arc::new(Mutex::new(GdpoAdvantageStats::default())),
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
        let revisit_min_filled = schedule_linear(
            self.training.revisit_min_filled_frac,
            self.training.revisit_min_filled_final,
            self.training.revisit_min_filled_anneal_steps,
            step,
        )
        .clamp(0.0, 1.0);
        let (losses, grads) = rollout_losses_train::<B>(
            self,
            batch,
            self.training.policy_noise,
            self.training.gdpo.enabled,
            teacher_forcing_prob,
            policy_temperature,
            policy_entropy_weight,
            revisit_min_filled,
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
    if decay > 0.0 {
        if let Some(stats) = stats {
            if let Ok(mut stats) = stats.lock() {
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
        }
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

fn rollout_losses_train<B: AutodiffBackend>(
    trainer: &SudokuTrainer<B>,
    batch: SudokuBatch<B>,
    policy_noise: f32,
    gdpo_active: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    policy_entropy_weight: f32,
    revisit_min_filled: f32,
    gdpo_stats: Option<Arc<Mutex<GdpoAdvantageStats<B>>>>,
) -> (SudokuLosses<B>, GradientsParams) {
    let training = &trainer.training;
    let gdpo_group = training.gdpo.group_size.max(1);
    let repeat_for_gdpo = gdpo_active && gdpo_group > 1;
    let rollout = rollout_base_autodiff(
        &trainer.model,
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

    let losses = SudokuLosses {
        loss: loss.clone(),
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
    };

    let grads = GradientsParams::from_grads(loss.backward(), trainer);

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
    log_prob_sum: Option<Tensor<B, 2>>,
    policy_entropy_sum: Tensor<B, 2>,
    policy_steps: usize,
    batch: usize,
}

struct RolloutForward<B: BackendTrait> {
    logits: Tensor<B, 3>,
    policy_logits: Tensor<B, 2>,
    halt_logit: Tensor<B, 2>,
}

fn rollout_base<B: BackendTrait>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = training.rollout_steps.max(1);
    let backprop_steps_cfg = training.rollout_backprop_steps.unwrap_or(rollout_steps);
    let backprop_steps = if backprop_steps_cfg == 0 {
        rollout_steps
    } else {
        backprop_steps_cfg.min(rollout_steps).max(1)
    };
    let detach_until = if force_detach {
        rollout_steps
    } else {
        rollout_steps.saturating_sub(backprop_steps)
    };

    rollout_base_impl(
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
        |tokens, step_idx| {
            let (hidden, logits) = model.forward_with_hidden(tokens);
            let (hidden, logits) = if train_mode && step_idx < detach_until {
                (hidden.detach(), logits.detach())
            } else {
                (hidden, logits)
            };
            let policy_logits = model.policy_logits(hidden.clone());
            let policy_logits = if train_mode && step_idx < detach_until {
                policy_logits.detach()
            } else {
                policy_logits
            };
            let halt_logit = model.halt_logit(hidden);
            let halt_logit = if train_mode && step_idx < detach_until {
                halt_logit.detach()
            } else {
                halt_logit
            };

            RolloutForward {
                logits,
                policy_logits,
                halt_logit,
            }
        },
    )
}

fn rollout_base_autodiff<B: AutodiffBackend>(
    model: &SudokuSaccadeModel<B>,
    batch: SudokuBatch<B>,
    training: &SudokuTrainingHyperparameters,
    policy_noise: f32,
    repeat_for_gdpo: Option<usize>,
    track_policy: bool,
    train_mode: bool,
    force_detach: bool,
    teacher_forcing_prob: f32,
    policy_temperature: f32,
    revisit_min_filled: f32,
) -> RolloutBase<B> {
    let rollout_steps = training.rollout_steps.max(1);
    let backprop_steps_cfg = training.rollout_backprop_steps.unwrap_or(rollout_steps);
    let backprop_steps = if backprop_steps_cfg == 0 {
        rollout_steps
    } else {
        backprop_steps_cfg.min(rollout_steps).max(1)
    };
    let detach_until = if force_detach {
        rollout_steps
    } else {
        rollout_steps.saturating_sub(backprop_steps)
    };
    let model_inner = if train_mode && detach_until > 0 {
        Some(model.valid())
    } else {
        None
    };

    rollout_base_impl(
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
        |tokens, step_idx| {
            if train_mode && step_idx < detach_until {
                let inner = model_inner.as_ref().expect("inner model");
                let (hidden_inner, logits_inner) =
                    inner.forward_with_hidden(tokens.inner());
                let policy_logits_inner = inner.policy_logits(hidden_inner.clone());
                let halt_logit_inner = inner.halt_logit(hidden_inner);
                RolloutForward {
                    logits: Tensor::<B, 3>::from_inner(logits_inner),
                    policy_logits: Tensor::<B, 2>::from_inner(policy_logits_inner),
                    halt_logit: Tensor::<B, 2>::from_inner(halt_logit_inner),
                }
            } else {
                let (hidden, logits) = model.forward_with_hidden(tokens);
                let policy_logits = model.policy_logits(hidden.clone());
                let halt_logit = model.halt_logit(hidden);
                RolloutForward {
                    logits,
                    policy_logits,
                    halt_logit,
                }
            }
        },
    )
}

fn rollout_base_impl<B, F>(
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
    mut forward: F,
) -> RolloutBase<B>
where
    B: BackendTrait,
    F: FnMut(Tensor<B, 2, Int>, usize) -> RolloutForward<B>,
{
    let rollout_steps = rollout_steps.max(1);
    let recon_loss_interval = training.recon_loss_interval_steps;
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

    let clue_mask = puzzles.clone().greater_elem(0.0).float();
    let editable_mask = ones_grid.clone().sub(clue_mask.clone());
    let editable_counts = editable_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1), 1])
        .clamp_min(1.0);

    let mut tokens = puzzles;
    let mut unknown_mask = tokens.clone().equal_elem(0).float();
    let mut loss_unknown_mask = unknown_mask.clone();
    let initial_unknown_counts = unknown_mask
        .clone()
        .sum_dim(1)
        .reshape([batch_size.max(1)]);

    let loss_mask_all = match training.loss_mask {
        SudokuLossMask::All => Some(ones_grid.clone()),
        SudokuLossMask::Unknown => None,
    };
    let ones_step = Tensor::<B, 2>::ones([batch_size.max(1), 1], &device);
    let mut halted = Tensor::<B, 2>::zeros([batch_size.max(1), 1], &device);

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
    let mut recon_loss_sum = Tensor::<B, 1>::zeros([1], &device);
    let mut recon_steps = 0usize;
    let mut steps_done = 0usize;

    let mut last_loss_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);
    let mut last_acc = Tensor::<B, 1>::zeros([1], &device);
    let mut last_exact = Tensor::<B, 1>::zeros([1], &device);
    let mut last_acc_per_sample = Tensor::<B, 1>::zeros([batch_size.max(1)], &device);

    for step_idx in 0..rollout_steps {
        let RolloutForward {
            logits,
            policy_logits,
            halt_logit,
        } = forward(tokens.clone(), step_idx);

        let loss_mask = match training.loss_mask {
            SudokuLossMask::All => loss_mask_all
                .as_ref()
                .expect("all loss mask")
                .clone(),
            SudokuLossMask::Unknown => loss_unknown_mask.clone(),
        };

        let mut evaluated_this_step = false;
        if recon_loss_interval > 0 && (step_idx + 1) % recon_loss_interval == 0 {
            let (
                step_loss,
                step_acc,
                step_exact,
                step_loss_per_sample,
                step_acc_per_sample,
                _step_exact_per_sample,
            ) = compute_loss_and_acc(
                &logits,
                &solutions,
                &solution_one_hot,
                &loss_mask,
                &training.recon_loss,
            );

            recon_loss_sum = recon_loss_sum + step_loss.clone();
            recon_steps += 1;

            last_loss_per_sample = step_loss_per_sample;
            last_acc = step_acc;
            last_exact = step_exact;
            last_acc_per_sample = step_acc_per_sample.clone();

            if step_idx == 0 {
                initial_acc = Some(step_acc_per_sample);
            }
            evaluated_this_step = true;
        }

        let halt_prob = activation::sigmoid(halt_logit.clone());
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
        let select_mask = unknown_mask
            .clone()
            .mul(ones_grid.clone().sub(allow_revisit_grid.clone()))
            .add(editable_mask.clone().mul(allow_revisit_grid));
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
        action_one_hot = action_one_hot * active_mask_grid * select_mask.clone();

        let log_probs = activation::log_softmax(policy_logits, 1);
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

        let preds = logits
            .clone()
            .argmax(2)
            .reshape([batch_size.max(1), GRID_LEN]);
        if step_idx == 0 && initial_acc.is_none() {
            let acc_per_sample = preds
                .clone()
                .equal(solutions.clone())
                .float()
                .sum_dim(1)
                .reshape([batch_size.max(1)])
                .div_scalar(GRID_LEN as f32);
            initial_acc = Some(acc_per_sample);
        }
        let teacher_force = if train_mode && teacher_forcing_prob > 0.0 {
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
        let teacher_force_grid = teacher_force.clone().repeat_dim(1, GRID_LEN);
        let teacher_mask = teacher_force_grid.clone().greater_equal_elem(0.5);
        let mut update_values = preds;
        update_values = update_values.mask_where(teacher_mask, solutions.clone());

        let update_mask = action_one_hot.clone().greater_equal_elem(0.5);
        tokens = tokens.mask_where(update_mask, update_values);
        unknown_mask = (unknown_mask - action_one_hot.clone()).clamp_min(0.0);
        let teacher_action = action_one_hot.mul(teacher_force_grid);
        loss_unknown_mask = (loss_unknown_mask - teacher_action).clamp_min(0.0);

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
        halt_loss_sum = halt_loss_sum + step_halt_loss;
        let halt_prob_masked = halt_prob.clone().mul(halt_target.clone());
        halt_prob_sum = halt_prob_sum + halt_prob_masked.mean();
        halt_target_sum = halt_target_sum + halt_target.mean();
        halt_steps += 1;

        let mut step_halt = halt_prob.clone().greater_equal_elem(0.5).float();
        if !train_mode {
            step_halt = step_halt.mul_scalar(0.0);
        } else {
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
        }
        halted = halted.max_pair(step_halt.clone());

        steps_done += 1;

        let unknown_counts_next = unknown_mask
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let filled_frac_next = editable_counts
            .clone()
            .sub(unknown_counts_next.clone())
            .div(editable_counts.clone())
            .clamp_min(0.0)
            .clamp_max(1.0);
        let allow_revisit_next = filled_frac_next.greater_equal_elem(revisit_threshold).float();
        let allow_revisit_grid_next = allow_revisit_next.clone().repeat_dim(1, GRID_LEN);
        let select_mask_next = unknown_mask
            .clone()
            .mul(ones_grid.clone().sub(allow_revisit_grid_next.clone()))
            .add(editable_mask.clone().mul(allow_revisit_grid_next));
        let selectable_counts_next = select_mask_next
            .clone()
            .sum_dim(1)
            .reshape([batch_size.max(1), 1]);
        let active_next = selectable_counts_next
            .greater_elem(0.0)
            .float()
            .mul(ones_step.clone().sub(halted.clone()))
            .mul(ones_step.clone().sub(tokens_solved_after.clone()));
        let should_stop = if train_mode {
            false
        } else {
            tensor_to_f32(active_next.clone().mean()) <= 0.0
        };

        let final_eval_needed = should_stop || step_idx + 1 == rollout_steps;
        if final_eval_needed && !evaluated_this_step {
            let (
                step_loss,
                step_acc,
                step_exact,
                step_loss_per_sample,
                step_acc_per_sample,
                _step_exact_per_sample,
            ) = compute_loss_and_acc(
                &logits,
                &solutions,
                &solution_one_hot,
                &loss_mask,
                &training.recon_loss,
            );

            recon_loss_sum = recon_loss_sum + step_loss.clone();
            recon_steps += 1;

            last_loss_per_sample = step_loss_per_sample;
            last_acc = step_acc;
            last_exact = step_exact;
            last_acc_per_sample = step_acc_per_sample.clone();

            if step_idx == 0 && initial_acc.is_none() {
                initial_acc = Some(step_acc_per_sample);
            }
        }

        if should_stop {
            break;
        }
    }

    let recon_loss = if recon_steps > 0 {
        recon_loss_sum.div_scalar(recon_steps as f32)
    } else {
        Tensor::<B, 1>::zeros([1], &device)
    };

    let initial_acc = initial_acc
        .unwrap_or_else(|| Tensor::<B, 1>::zeros([batch_size.max(1)], &device));
    let hard_reward = last_acc_per_sample.clone() - initial_acc;
    let easy_reward = last_loss_per_sample.clone().mul_scalar(-1.0);
    let difficulty_scale =
        difficulty_scale_from_unknowns(initial_unknown_counts, training.reward_unknown_power);
    let hard_reward = hard_reward.mul(difficulty_scale.clone());
    let easy_reward = easy_reward.mul(difficulty_scale);
    let halt_steps = halt_steps.max(1) as f32;
    let halt_loss = halt_loss_sum.div_scalar(halt_steps);
    let halt_target_mean = halt_target_sum.clone().div_scalar(halt_steps);
    let halt_prob_mean = halt_prob_sum.div(halt_target_sum.clamp_min(HALT_EPS));
    let solve_exact = tokens
        .equal(solutions)
        .float()
        .sum_dim(1)
        .reshape([batch_size.max(1)])
        .equal_elem(GRID_LEN as f32)
        .float()
        .mean();

    RolloutBase {
        recon_loss,
        acc: last_acc,
        exact_acc: last_exact,
        solve_rate: solve_exact,
        hard_reward,
        easy_reward,
        halt_loss,
        halt_prob_mean,
        halt_target_mean,
        log_prob_sum,
        policy_entropy_sum,
        policy_steps: steps_done,
        batch: batch_size,
    }
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
            revisit_min_filled_frac: 0.0,
            revisit_min_filled_final: 0.0,
            revisit_min_filled_anneal_steps: 0,
            reward_unknown_power: 0.0,
            recon_loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            recon_loss_interval_steps: 1,
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
            revisit_min_filled_frac: 0.0,
            revisit_min_filled_final: 0.0,
            revisit_min_filled_anneal_steps: 0,
            reward_unknown_power: 0.0,
            recon_loss: SudokuReconLoss::Softmax,
            loss_mask: SudokuLossMask::All,
            recon_loss_interval_steps: 1,
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
