use crate::config::VisionAugmentationConfig;
use crate::train::gdpo;
use crate::train::prelude::*;
use std::time::Instant;
use burn::optim::{GradientsAccumulator, GradientsParams, Optimizer};
use burn::tensor::Distribution as TensorDistribution;
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_train::{GdpoConfig, GdpoHardGate};

pub struct VisionSaccadeBench<B: AutodiffBackend> {
    model: VisionSaccadeModel<B>,
    images: Tensor<B, 4>,
    labels: Tensor<B, 1, Int>,
    steps: usize,
    backprop_steps: usize,
    patch_size: usize,
    levels: Vec<SaccadeMipLevel<B>>,
    input_sample_levels: Vec<Tensor<B, 3>>,
    state_composed: Vec<Tensor<B, 3>>,
    mean: Tensor<B, 3>,
    sigma: Tensor<B, 3>,
    mean_step: Tensor<B, 2>,
    sigma_step: Tensor<B, 2>,
    cached_weights: Vec<Tensor<B, 3>>,
    tokens_in: Tensor<B, 3>,
    residual_pool: Tensor<B, 3>,
    base_grid: Tensor<B, 4>,
    laplacian_images: Option<SaccadeLaplacianImages<B>>,
    gdpo_hard: Tensor<B, 2>,
    gdpo_easy: Tensor<B, 2>,
    gdpo_log_prob_new: Tensor<B, 2>,
    gdpo_log_prob_old: Tensor<B, 2>,
    gdpo_config: GdpoConfig,
}

pub struct VisionScatterBench<B: BackendTrait> {
    model: VisionSaccadeModel<B>,
    weights: Tensor<B, 3>,
    tokens: Tensor<B, 3>,
}

pub struct VisionInputProjectionBench<B: BackendTrait> {
    projection: VisionSaccadeInputProjection<B>,
    tokens: Tensor<B, 3>,
}

pub struct VisionSaccadeTrainStepBench<B: AutodiffBackend> {
    model: Option<VisionSaccadeModel<B>>,
    optimizer: OptimizerAdaptor<AdamW, VisionSaccadeModel<B>, B>,
    lr: LearningRate,
    rollout_steps: usize,
    backprop_steps: usize,
}

pub struct VisionMaeTrainStepBench<B: AutodiffBackend> {
    model: Option<VisionMaeModel<B>>,
    optimizer: OptimizerAdaptor<AdamW, VisionMaeModel<B>, B>,
    lr: LearningRate,
    rollout_steps: usize,
    backprop_steps: usize,
}

pub struct VisionLejepaTrainStepBench<B: AutodiffBackend> {
    model: Option<VisionLejepaModel<B>>,
    optimizer: OptimizerAdaptor<AdamW, VisionLejepaModel<B>, B>,
    lr: LearningRate,
    rollout_steps: usize,
    backprop_steps: usize,
}

pub struct VisionVideoLejepaTrainStepBench<B: AutodiffBackend> {
    model: Option<VisionVideoLejepaModel<B>>,
    optimizer: OptimizerAdaptor<AdamW, VisionVideoLejepaModel<B>, B>,
    lr: LearningRate,
    rollout_steps: usize,
    backprop_steps: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct VisionVideoLejepaTrainStepPhaseTimes {
    pub forward_ns: f64,
    pub backward_ns: f64,
    pub optimize_ns: f64,
}

pub struct VisionVideoLejepaTrainStepProfile<B: AutodiffBackend> {
    pub loss: Tensor<B, 1>,
    pub phases: VisionVideoLejepaTrainStepPhaseTimes,
}

#[derive(Clone, Copy, Debug)]
pub struct FoveaKernelEstimate {
    pub levels: usize,
    pub subsamples: usize,
    pub grid_sample_calls: usize,
    pub unfused_grid_sample_calls: usize,
}

impl<B: AutodiffBackend> VisionSaccadeBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        saccade: VisionSaccadeConfig,
        batch_size: usize,
        steps: usize,
        backprop_steps: usize,
        device: &B::Device,
    ) -> Self {
        let model = VisionDragon::<B>::new(vision.clone(), device);
        let recon_patch_dim = vision.patch_size * vision.patch_size * vision.in_channels;
        let rollout = VisionRollout {
            min_steps: steps,
            max_steps: steps,
            backprop_steps: backprop_steps.min(steps).max(1),
        };
        let saccade = VisionSaccadeModel::new(
            model,
            saccade,
            vision.embed_dim,
            vision.patch_size,
            rollout,
            recon_patch_dim,
            1,
            0,
            device,
        );

        let images = Tensor::<B, 4>::random(
            [
                batch_size,
                vision.in_channels,
                vision.image_size,
                vision.image_size,
            ],
            TensorDistribution::Default,
            device,
        );
        let labels = Tensor::<B, 1, Int>::from_data(
            TensorData::new(vec![0i64; batch_size], [batch_size]),
            device,
        );

        let levels = saccade.build_mip_pyramid(images.clone(), vision.patch_size);
        let input_levels: Vec<Tensor<B, 3>> =
            levels.iter().map(|level| level.tokens.clone()).collect();
        let grids: Vec<PatchGrid> = levels.iter().map(|level| level.grid).collect();
        let (input_residuals, input_sample_levels) = match saccade.config.pyramid_mode {
            VisionPyramidMode::Stacked => (input_levels.clone(), input_levels.clone()),
            VisionPyramidMode::Laplacian => {
                let residuals = saccade.decompose_pyramid(&input_levels, &grids);
                (residuals.clone(), residuals)
            }
        };
        let state_levels: Vec<Tensor<B, 3>> = input_residuals
            .iter()
            .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), device))
            .collect();
        let state_composed = match saccade.config.pyramid_mode {
            VisionPyramidMode::Stacked => state_levels.clone(),
            VisionPyramidMode::Laplacian => saccade.compose_pyramid(&state_levels, &grids),
        };

        let [batch, _channels, _height, _width] = images.shape().dims::<4>();
        let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
        let base_traj = saccade
            .trajectory_token
            .val()
            .reshape([1, traj_len, vision.embed_dim])
            .repeat_dim(0, batch);
        let eye_embed = saccade
            .eye_token
            .val()
            .slice_dim(0, 0..1)
            .reshape([1, 1, vision.embed_dim])
            .repeat_dim(0, batch)
            .repeat_dim(1, traj_len);
        let eye_embed = if saccade.config.num_eyes > 1 {
            eye_embed
        } else {
            Tensor::<B, 3>::zeros([batch, traj_len, vision.embed_dim], device)
        };
        let traj_with_eye = base_traj + eye_embed;
        let traj_summary = traj_with_eye
            .clone()
            .mean_dim(1)
            .reshape([batch, 1, vision.embed_dim]);
        let params = saccade.saccade_head.forward(traj_summary);
        let (mean, sigma) = saccade.decode_saccade_params(params);
        let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
        let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
        let cached_weights = saccade.mip_gaussian_weights(&levels, mean.clone(), sigma.clone());

        let input_context = saccade.mip_weighted_sum(&input_sample_levels, &cached_weights);
        let input_context = saccade.project_pyramid_context(input_context);
        let state_context = saccade.mip_weighted_sum(&state_composed, &cached_weights);
        let state_context = saccade.project_pyramid_context(state_context);
        let input_tokens =
            saccade.build_input_tokens(input_context, state_context, mean.clone(), sigma.clone());
        let tokens_in = traj_with_eye.clone() + input_tokens.repeat_dim(1, traj_len);
        let inner_steps = saccade.config.inner_steps.max(1);
        let tokens_out = saccade
            .model
            .forward_tokens_embed_steps(tokens_in.clone(), inner_steps)
            .patch_tokens;
        let residual = saccade.residual_proj.forward(tokens_out);
        let residual_pool = residual
            .mean_dim(1)
            .reshape([batch, 1, saccade.pyramid_feature_dim()]);

        let base_grid = build_foveated_base_grid::<B>(vision.patch_size, device);
        let laplacian_images =
            if matches!(saccade.config.pyramid_mode, VisionPyramidMode::Laplacian) {
                saccade.build_laplacian_images(&levels)
            } else {
                None
            };
        let gdpo_group = 4usize;
        let gdpo_hard = Tensor::<B, 2>::random(
            [batch_size, gdpo_group],
            TensorDistribution::Default,
            device,
        );
        let gdpo_easy = Tensor::<B, 2>::random(
            [batch_size, gdpo_group],
            TensorDistribution::Default,
            device,
        );
        let gdpo_batch = batch_size * gdpo_group;
        let gdpo_log_prob_new = Tensor::<B, 2>::random(
            [gdpo_batch, 1],
            TensorDistribution::Normal(0.0, 1.0),
            device,
        );
        let gdpo_log_prob_old = Tensor::<B, 2>::random(
            [gdpo_batch, 1],
            TensorDistribution::Normal(0.0, 1.0),
            device,
        );
        let gdpo_config = GdpoConfig {
            enabled: true,
            group_size: gdpo_group,
            hard_gate: GdpoHardGate::Off,
            ..GdpoConfig::default()
        };

        Self {
            model: saccade,
            images,
            labels,
            steps,
            backprop_steps,
            patch_size: vision.patch_size,
            levels,
            input_sample_levels,
            state_composed,
            mean,
            sigma,
            mean_step,
            sigma_step,
            cached_weights,
            tokens_in,
            residual_pool,
            base_grid,
            laplacian_images,
            gdpo_hard,
            gdpo_easy,
            gdpo_log_prob_new,
            gdpo_log_prob_old,
            gdpo_config,
        }
    }

    pub fn fovea_patch_kernel_estimate(&self) -> FoveaKernelEstimate {
        let levels = self
            .model
            .build_mip_pyramid(self.images.clone(), self.patch_size);
        let level_count = levels.len();
        let subsamples_axis = self.model.config.fovea_subsamples.max(1);
        let subsamples = subsamples_axis * subsamples_axis;
        let grid_sample_calls = level_count;
        let unfused_grid_sample_calls = level_count.saturating_mul(subsamples);
        FoveaKernelEstimate {
            levels: level_count,
            subsamples,
            grid_sample_calls,
            unfused_grid_sample_calls,
        }
    }

    pub fn stage_patch_embed(&self) -> Tensor<B, 1> {
        let patch = self.model.model.patch_embed_raw(self.images.clone());
        patch.tokens.sum()
    }

    pub fn stage_mip_pyramid(&self) -> Tensor<B, 1> {
        let levels = self
            .model
            .build_mip_pyramid(self.images.clone(), self.patch_size);
        let device = self.images.device();
        let mut total = Tensor::<B, 1>::zeros([1], &device);
        for level in levels {
            total = total + level.tokens.sum();
        }
        total
    }

    pub fn stage_fovea_weights(&self) -> Tensor<B, 1> {
        let weights =
            self.model
                .mip_gaussian_weights(&self.levels, self.mean.clone(), self.sigma.clone());
        let device = self.images.device();
        let mut total = Tensor::<B, 1>::zeros([1], &device);
        for weight in weights {
            total = total + weight.sum();
        }
        total
    }

    pub fn stage_fovea_context(&self) -> Tensor<B, 1> {
        let input_context = self
            .model
            .mip_weighted_sum(&self.input_sample_levels, &self.cached_weights);
        let input_context = self.model.project_pyramid_context(input_context);
        let state_context = self
            .model
            .mip_weighted_sum(&self.state_composed, &self.cached_weights);
        let state_context = self.model.project_pyramid_context(state_context);
        let input_tokens = self.model.build_input_tokens(
            input_context,
            state_context,
            self.mean.clone(),
            self.sigma.clone(),
        );
        input_tokens.sum()
    }

    pub fn stage_gdpo_advantage(&self) -> Tensor<B, 1> {
        let advantage = gdpo::gdpo_advantage_autodiff::<B>(
            self.gdpo_hard.clone(),
            self.gdpo_easy.clone(),
            &self.gdpo_config,
        );
        advantage.sum()
    }

    pub fn stage_gdpo_policy_loss(&self) -> Tensor<B, 1> {
        let [scene_batch, group] = self.gdpo_hard.shape().dims::<2>();
        let batch = scene_batch * group;
        let advantage = gdpo::gdpo_advantage_autodiff::<B>(
            self.gdpo_hard.clone(),
            self.gdpo_easy.clone(),
            &self.gdpo_config,
        )
        .reshape([batch, 1])
        .detach();
        gdpo::gdpo_policy_loss(
            self.gdpo_log_prob_new.clone(),
            self.gdpo_log_prob_old.clone(),
            advantage,
            &self.gdpo_config,
        )
    }

    pub fn stage_fovea_patch(&self) -> Tensor<B, 1> {
        let patch = self.model.foveated_patch_image(
            &self.levels,
            &self.base_grid,
            self.mean_step.clone(),
            self.sigma_step.clone(),
            self.laplacian_images.as_ref(),
        );
        patch.sum()
    }

    pub fn stage_token_forward(&self) -> Tensor<B, 1> {
        let inner_steps = self.model.config.inner_steps.max(1);
        let out = self
            .model
            .model
            .forward_tokens_embed_steps(self.tokens_in.clone(), inner_steps)
            .patch_tokens;
        out.sum()
    }

    pub fn stage_residual_scatter(&self) -> Tensor<B, 1> {
        let device = self.images.device();
        let mut total = Tensor::<B, 1>::zeros([1], &device);
        for weights in &self.cached_weights {
            let update = self
                .model
                .weighted_sum_tokens(weights.clone().swap_dims(1, 2), self.residual_pool.clone());
            total = total + update.sum();
        }
        total
    }

    pub fn stage_full_forward(&self) -> Tensor<B, 1> {
        let batch = self.make_batch();
        let losses =
            self.model
                .forward_losses_train(batch, self.steps, self.backprop_steps, true, false);
        losses.total
    }

    pub fn stage_full_backward(&self) -> Tensor<B, 1> {
        let batch = self.make_batch();
        let losses =
            self.model
                .forward_losses_train(batch, self.steps, self.backprop_steps, true, false);
        let total = losses.total.clone();
        let grads = GradientsParams::from_grads(total.backward(), &self.model);
        let grad = grads
            .get::<ValidBackend<B>, 2>(self.model.trajectory_token.id)
            .expect("trajectory_token grad");
        let _ = grad.sum().to_data();
        total.detach()
    }

    fn make_batch(&self) -> ImageNetBatch<B> {
        ImageNetBatch::new(
            self.images.clone(),
            None,
            None,
            None,
            None,
            None,
            self.labels.clone(),
            None,
            None,
        )
    }
}

impl<B: AutodiffBackend> VisionSaccadeTrainStepBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        saccade: VisionSaccadeConfig,
        training: &VisionTrainingHyperparameters,
        optimizer_cfg: &OptimizerConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let rollout = resolve_vision_rollout(training, vision.steps)?;
        let recon_patch_dim = vision.patch_size * vision.patch_size * vision.in_channels;
        let model = VisionDragon::<B>::new(vision.clone(), device);
        let saccade = VisionSaccadeModel::new(
            model,
            saccade,
            vision.embed_dim,
            vision.patch_size,
            rollout,
            recon_patch_dim,
            training.batch_repeats,
            training.train_repeat_chunk,
            device,
        );
        let rollout_steps = saccade.rollout.max_steps;
        let backprop_steps = saccade.rollout.backprop_steps(rollout_steps);
        let optimizer =
            adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionSaccadeModel<B>>();
        let lr = optimizer_cfg.learning_rate;
        Ok(Self {
            model: Some(saccade),
            optimizer,
            lr,
            rollout_steps,
            backprop_steps,
        })
    }

    pub fn train_step(&mut self, batch: ImageNetBatch<B>) -> Tensor<B, 1> {
        let mut model = self.model.take().expect("saccade model");
        let repeats = model.train_repeats.max(1);
        if repeats == 1 {
            let losses = model.forward_losses_train(
                batch,
                self.rollout_steps,
                self.backprop_steps,
                true,
                false,
            );
            let grads = losses.total.clone().backward();
            let grads = GradientsParams::from_grads(grads, &model);
            let loss = losses.total.detach();
            model = self.optimizer.step(self.lr, model, grads);
            self.model = Some(model);
            return loss;
        }

        let scale = 1.0 / repeats as f32;
        let repeat_chunk = train_repeat_chunk(repeats, model.train_repeat_chunk);
        let mut grads = GradientsAccumulator::new();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut consumed = 0;
        while consumed < repeats {
            let chunk = (repeats - consumed).min(repeat_chunk);
            let batch_chunk = if chunk == 1 {
                batch.clone()
            } else {
                batch.repeat_batch(chunk)
            };
            let losses = model.forward_losses_train(
                batch_chunk,
                self.rollout_steps,
                self.backprop_steps,
                true,
                false,
            );
            let loss_scaled = losses.total.clone().mul_scalar(scale * chunk as f32);
            let grads_step = GradientsParams::from_grads(loss_scaled.backward(), &model);
            grads.accumulate(&model, grads_step);
            let weight = chunk as f32;
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + losses.total.clone().mul_scalar(weight),
                None => losses.total.clone().mul_scalar(weight),
            });
            consumed += chunk;
        }
        let grads = grads.grads();
        let loss = loss_sum.expect("repeat loss").mul_scalar(scale).detach();
        model = self.optimizer.step(self.lr, model, grads);
        self.model = Some(model);
        loss
    }
}

impl<B: AutodiffBackend> VisionMaeTrainStepBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        mae: VisionMaeConfig,
        training: &VisionTrainingHyperparameters,
        optimizer_cfg: &OptimizerConfig,
        device: &B::Device,
    ) -> Result<Self> {
        let rollout = resolve_vision_rollout(training, vision.steps)?;
        let embed_dim = vision.embed_dim;
        let num_eyes = vision.num_eyes;
        let patch_size = vision.patch_size;
        let in_channels = vision.in_channels;
        let recon_patch_dim = patch_size * patch_size * in_channels;
        let normalize_std = VisionAugmentationConfig::default().normalize_std;
        let normalization = vision.normalization.clone();
        let model = VisionDragon::<B>::new(vision, device);
        let mae = VisionMaeModel::new(
            model,
            mae,
            VisionMaeInit {
                num_eyes,
                embed_dim,
                rollout,
                recon: VisionReconstructionInit {
                    patch_dim: recon_patch_dim,
                    normalize_std,
                    patch_size,
                    in_channels,
                },
                normalization,
            },
            device,
        );
        let rollout_steps = mae.rollout.max_steps;
        let backprop_steps = mae.rollout.backprop_steps(rollout_steps);
        let optimizer = adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionMaeModel<B>>();
        let lr = optimizer_cfg.learning_rate;
        Ok(Self {
            model: Some(mae),
            optimizer,
            lr,
            rollout_steps,
            backprop_steps,
        })
    }

    pub fn train_step(&mut self, batch: ImageNetBatch<B>) -> Tensor<B, 1> {
        let mut model = self.model.take().expect("mae model");
        let losses = model.forward_losses(
            batch,
            self.rollout_steps,
            self.backprop_steps,
            true,
            false,
            false,
        );
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
        let loss = losses.total.detach();
        model = model.optimize::<B, _>(&mut self.optimizer, self.lr, grads);
        self.model = Some(model);
        loss
    }
}

impl<B: AutodiffBackend> VisionLejepaTrainStepBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        lejepa: VisionLejepaConfig,
        training: &VisionTrainingHyperparameters,
        optimizer_cfg: &OptimizerConfig,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let rollout = resolve_vision_rollout(training, vision.steps)?;
        let embed_dim = vision.embed_dim;
        let patch_size = vision.patch_size;
        let in_channels = vision.in_channels;
        let recon_patch_dim = patch_size * patch_size * in_channels;
        let normalize_std = VisionAugmentationConfig::default().normalize_std;
        let normalization = vision.normalization.clone();
        let model = VisionDragon::<B>::new(vision, device);
        let lejepa = VisionLejepaModel::new(
            model,
            lejepa,
            VisionLejepaInit {
                embed_dim,
                num_classes,
                rollout,
                recon: VisionReconstructionInit {
                    patch_dim: recon_patch_dim,
                    normalize_std,
                    patch_size,
                    in_channels,
                },
                normalization,
            },
            device,
        );
        let rollout_steps = lejepa.rollout.max_steps;
        let backprop_steps = lejepa.rollout.backprop_steps(rollout_steps);
        let optimizer =
            adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionLejepaModel<B>>();
        let lr = optimizer_cfg.learning_rate;
        Ok(Self {
            model: Some(lejepa),
            optimizer,
            lr,
            rollout_steps,
            backprop_steps,
        })
    }

    pub fn train_step(&mut self, batch: ImageNetBatch<B>) -> Tensor<B, 1> {
        let mut model = self.model.take().expect("lejepa model");
        let losses = model.forward_losses(
            batch,
            self.rollout_steps,
            self.backprop_steps,
            true,
            false,
            false,
        );
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
        let loss = losses.total.detach();
        model = model.optimize::<B, _>(&mut self.optimizer, self.lr, grads);
        self.model = Some(model);
        loss
    }
}

impl<B: AutodiffBackend> VisionVideoLejepaTrainStepBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        video: VisionVideoLejepaConfig,
        training: &VisionTrainingHyperparameters,
        optimizer_cfg: &OptimizerConfig,
        num_classes: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let rollout = resolve_vision_rollout(training, vision.steps)?;
        let model = VisionDragon::<B>::new(vision.clone(), device);
        let video =
            VisionVideoLejepaModel::new(model, video, &vision, rollout, num_classes, device);
        let rollout_steps = video.rollout.max_steps;
        let backprop_steps = video.rollout.backprop_steps(rollout_steps);
        let optimizer =
            adamw_config_from_optimizer(optimizer_cfg).init::<B, VisionVideoLejepaModel<B>>();
        let lr = optimizer_cfg.learning_rate;
        Ok(Self {
            model: Some(video),
            optimizer,
            lr,
            rollout_steps,
            backprop_steps,
        })
    }

    pub fn train_step(&mut self, batch: VideoClipBatch<B>) -> Tensor<B, 1> {
        self.train_step_profile(batch).loss
    }

    pub fn train_step_profile(
        &mut self,
        batch: VideoClipBatch<B>,
    ) -> VisionVideoLejepaTrainStepProfile<B> {
        let mut model = self.model.take().expect("video lejepa model");
        let forward_start = Instant::now();
        let losses = if model.uses_pyramid_backbone() {
            model.forward_losses_train_pyramid(
                batch,
                self.rollout_steps,
                self.backprop_steps,
                false,
                false,
            )
        } else {
            model.forward_losses(
                batch,
                self.rollout_steps,
                self.backprop_steps,
                false,
                false,
                false,
            )
        };
        let forward_ns = forward_start.elapsed().as_nanos() as f64;
        let total = losses.total.clone()
            + losses
                .probe_loss
                .clone()
                .mul_scalar(model.config.loss.probe_weight.max(0.0));
        let backward_start = Instant::now();
        let grads = GradientsParams::from_grads(total.backward(), &model);
        let backward_ns = backward_start.elapsed().as_nanos() as f64;
        let loss = losses.total.detach();
        let optimize_start = Instant::now();
        model = model.optimize::<B, _>(&mut self.optimizer, self.lr, grads);
        let optimize_ns = optimize_start.elapsed().as_nanos() as f64;
        self.model = Some(model);
        VisionVideoLejepaTrainStepProfile {
            loss,
            phases: VisionVideoLejepaTrainStepPhaseTimes {
                forward_ns,
                backward_ns,
                optimize_ns,
            },
        }
    }

    pub fn forward_loss(&self, batch: VideoClipBatch<B>) -> Tensor<B, 1> {
        let model = self.model.as_ref().expect("video lejepa model");
        if model.uses_pyramid_backbone() {
            model.forward_losses_train_pyramid(
                batch,
                self.rollout_steps,
                self.backprop_steps,
                false,
                false,
            )
                .total
                .detach()
        } else {
            model.forward_losses(
                batch,
                self.rollout_steps,
                self.backprop_steps,
                false,
                false,
                false,
            )
                .total
                .detach()
        }
    }
}

impl<B: BackendTrait> VisionInputProjectionBench<B> {
    pub fn new(
        embed_dim: usize,
        patch_size: usize,
        token_count: usize,
        batch: usize,
        config: VisionSaccadeInputProjectionConfig,
        device: &B::Device,
    ) -> Self {
        let projection = VisionSaccadeInputProjection::new(embed_dim, patch_size, &config, device);
        let tokens = Tensor::<B, 3>::random(
            [batch, token_count, embed_dim],
            TensorDistribution::Default,
            device,
        );
        Self { projection, tokens }
    }

    pub fn param_count(&self) -> usize {
        self.projection.param_count()
    }

    pub fn forward(&self) -> Tensor<B, 1> {
        self.projection.forward(self.tokens.clone()).sum()
    }
}

impl<B: BackendTrait> VisionScatterBench<B> {
    pub fn new(
        vision: VisionDragonConfig,
        saccade: VisionSaccadeConfig,
        batch_size: usize,
        out_tokens: usize,
        in_tokens: usize,
        feature_dim: usize,
        device: &B::Device,
    ) -> Self {
        let model = VisionDragon::<B>::new(vision.clone(), device);
        let recon_patch_dim = vision.patch_size * vision.patch_size * vision.in_channels;
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 1,
            backprop_steps: 1,
        };
        let saccade = VisionSaccadeModel::new(
            model,
            saccade,
            vision.embed_dim,
            vision.patch_size,
            rollout,
            recon_patch_dim,
            1,
            0,
            device,
        );

        let weights = Tensor::<B, 3>::random(
            [batch_size, out_tokens.max(1), in_tokens.max(1)],
            TensorDistribution::Default,
            device,
        );
        let tokens = Tensor::<B, 3>::random(
            [batch_size, in_tokens.max(1), feature_dim.max(1)],
            TensorDistribution::Default,
            device,
        );

        Self {
            model: saccade,
            weights,
            tokens,
        }
    }

    pub fn stage_scatter(&self) -> Tensor<B, 1> {
        self.model
            .weighted_sum_tokens(self.weights.clone(), self.tokens.clone())
            .sum()
    }
}
