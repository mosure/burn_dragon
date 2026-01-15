use crate::train::prelude::*;
use crate::train::gdpo;

/// GPU foveation sampler that reuses the saccade pipeline's mip + patch sampling.
pub struct SaccadeFoveationSampler<B: BackendTrait> {
    saccade: VisionSaccadeModel<B>,
    base_grid: Tensor<B, 4>,
    patch_size: usize,
    levels: Vec<SaccadeMipLevel<B>>,
    laplacian: Option<SaccadeLaplacianImages<B>>,
}

impl<B: BackendTrait> SaccadeFoveationSampler<B> {
    pub fn new(
        vision: VisionDragonHatchlingConfig,
        saccade: VisionSaccadeConfig,
        device: &B::Device,
    ) -> Self {
        let model = VisionDragonHatchling::<B>::new(vision.clone(), device);
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
        let base_grid = build_foveated_base_grid::<B>(vision.patch_size, device);
        Self {
            saccade,
            base_grid,
            patch_size: vision.patch_size,
            levels: Vec::new(),
            laplacian: None,
        }
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }

    pub fn mip_levels(&self) -> usize {
        self.levels.len()
    }

    pub fn update_image(&mut self, images: Tensor<B, 4>) {
        self.levels = build_sampling_pyramid::<B>(images, self.saccade.config.mip_levels);
        self.laplacian = if matches!(self.saccade.config.pyramid_mode, VisionPyramidMode::Laplacian) {
            self.saccade.build_laplacian_images(&self.levels)
        } else {
            None
        };
    }

    pub fn sample_patch(
        &self,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        self.saccade.foveated_patch_image(
            &self.levels,
            &self.base_grid,
            mean,
            sigma,
            self.laplacian.as_ref(),
        )
    }

    pub fn sample_patch_with_radius(
        &self,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        radius: Tensor<B, 2>,
    ) -> Tensor<B, 4> {
        self.saccade.foveated_patch_image_with_radius(
            &self.levels,
            &self.base_grid,
            mean,
            sigma,
            radius,
            self.laplacian.as_ref(),
        )
    }
}

pub(crate) fn build_sampling_pyramid<B: BackendTrait>(
    images: Tensor<B, 4>,
    mip_levels: usize,
) -> Vec<SaccadeMipLevel<B>> {
    let max_levels = mip_levels.max(1);
    let device = images.device();
    let [batch, _channels, height, width] = images.shape().dims::<4>();
    if height == 0 || width == 0 {
        return Vec::new();
    }
    let mut levels = Vec::with_capacity(max_levels);
    let mut current = images;
    for level in 0..max_levels {
        let tokens = Tensor::<B, 3>::zeros([batch.max(1), 1, 1], &device);
        levels.push(SaccadeMipLevel {
            tokens,
            grid: PatchGrid { height: 1, width: 1 },
            image: current.clone(),
        });
        if level + 1 == max_levels {
            break;
        }
        let Some(next) = downsample_image(current) else {
            break;
        };
        current = next;
    }
    levels
}

impl<B: AutodiffBackend> TrainStep<SequenceBatch<B>, LanguageModelTrainItem<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let logits = if fast_train_enabled() {
            self.forward_fast(batch.inputs)
        } else {
            self.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        let grads = loss.backward();

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep<SequenceBatch<B>, LanguageModelOutput<B>> for BDH<B> {
    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let logits = if fast_train_enabled() {
            self.forward_fast(batch.inputs)
        } else {
            self.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>>
    for VisionDistillModel<B>
{
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let (teacher_patch, teacher_cls) = if let Some(teacher) = &self.teacher {
            let output = teacher.forward(images.clone(), None);
            (output.x_norm_patchtokens, output.x_norm_clstoken)
        } else {
            let teacher_patch = teacher_patch.expect("teacher patch features required");
            let teacher_cls = teacher_cls.expect("teacher cls features required");
            (teacher_patch, teacher_cls)
        };

        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let output = self
            .model
            .forward_images_steps_rollout(images, rollout_steps, backprop_steps);
        let loss = vision_distillation_loss(
            output.patch_tokens,
            teacher_patch,
            output.cls_token,
            teacher_cls,
            &self.loss,
        );
        let zero = Tensor::<B, 1>::zeros([1], &loss.device());
        let grads = loss.backward();

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                loss,
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero.clone(),
                zero,
            ),
        )
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionDistillModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let (teacher_patch, teacher_cls) = if let Some(teacher) = &self.teacher {
            let output = teacher.forward(images.clone(), None);
            (output.x_norm_patchtokens, output.x_norm_clstoken)
        } else {
            let teacher_patch = teacher_patch.expect("teacher patch features required");
            let teacher_cls = teacher_cls.expect("teacher cls features required");
            (teacher_patch, teacher_cls)
        };

        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let output = self
            .model
            .forward_images_steps_rollout(images, self.rollout.max_steps, backprop_steps);
        let loss = vision_distillation_loss(
            output.patch_tokens,
            teacher_patch,
            output.cls_token,
            teacher_cls,
            &self.loss,
        );
        let zero = Tensor::<B, 1>::zeros([1], &loss.device());
        VisionOutput::new(
            loss,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            None,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionLejepaModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = self.forward_losses(batch, rollout_steps, backprop_steps, true);
        let total_for_backprop = losses.total.clone() + losses.probe_loss.clone();
        let grads = total_for_backprop.backward();

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                losses.total,
                losses.inv,
                losses.sigreg,
                losses.recon,
                losses.probe_loss,
                losses.probe_acc,
            ),
        )
    }

    fn optimize<BB, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: burn::optim::Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        optim.step(lr, self, grads)
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionLejepaModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let losses = self.forward_losses(batch, self.rollout.max_steps, backprop_steps, false);

        VisionOutput::new(
            losses.total,
            losses.inv,
            losses.sigreg,
            losses.recon,
            losses.probe_loss,
            losses.probe_acc,
            losses.artifacts,
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionMaeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let losses = self.forward_losses(batch, rollout_steps, backprop_steps, true, false);
        let grads = losses.total.clone().backward();
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());

        TrainOutput::new(
            self,
            grads,
            VisionTrainItem::new(
                losses.total,
                zero.clone(),
                zero.clone(),
                losses.recon,
                zero.clone(),
                zero,
            ),
        )
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionMaeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let capture_artifacts = self.config.artifact_every > 0;
        let losses = self.forward_losses(
            batch,
            self.rollout.max_steps,
            backprop_steps,
            false,
            capture_artifacts,
        );
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            zero.clone(),
            zero.clone(),
            losses.recon,
            zero.clone(),
            zero,
            losses.artifacts,
        )
    }
}

impl<B: AutodiffBackend> VisionSaccadeModel<B> {
    pub(crate) fn forward_losses_train(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionSaccadeLosses<B> {
        let gdpo = &self.config.policy.gdpo;
        self.forward_losses_with_policy(
            batch,
            steps,
            backprop_steps,
            randomize_mask,
            capture_artifacts,
            |inputs| {
                self.build_gdpo_policy_loss(gdpo, inputs, |hard, easy, gdpo| {
                    gdpo::gdpo_advantage_autodiff::<B>(hard, easy, gdpo)
                })
            },
        )
    }
}

impl<B: AutodiffBackend> TrainStep<ImageNetBatch<B>, VisionTrainItem<B>> for VisionSaccadeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let rollout_steps = self.rollout.sample_steps();
        let backprop_steps = self.rollout.backprop_steps(rollout_steps);
        let repeats = self.train_repeats.max(1);
        if repeats == 1 {
            let losses =
                self.forward_losses_train(batch, rollout_steps, backprop_steps, true, false);
            let grads = losses.total.clone().backward();
            let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
            return TrainOutput::new(
                self,
                grads,
                VisionTrainItem::new(
                    losses.total,
                    losses.inv,
                    losses.sigreg,
                    losses.recon,
                    zero.clone(),
                    zero,
                ),
            );
        }

        let scale = 1.0 / repeats as f32;
        let repeat_chunk = train_repeat_chunk(repeats, self.train_repeat_chunk);
        let mut grads = GradientsAccumulator::new();
        let mut total_sum: Option<Tensor<B, 1>> = None;
        let mut inv_sum: Option<Tensor<B, 1>> = None;
        let mut sigreg_sum: Option<Tensor<B, 1>> = None;
        let mut recon_sum: Option<Tensor<B, 1>> = None;
        let mut consumed = 0;

        while consumed < repeats {
            let chunk = (repeats - consumed).min(repeat_chunk);
            let batch_chunk = if chunk == 1 {
                batch.clone()
            } else {
                batch.repeat_batch(chunk)
            };
            let losses = self.forward_losses_train(
                batch_chunk,
                rollout_steps,
                backprop_steps,
                true,
                false,
            );
            let chunk_scale = scale * chunk as f32;
            let loss_scaled = losses.total.clone().mul_scalar(chunk_scale);
            let grads_step = GradientsParams::from_grads(loss_scaled.backward(), self);
            grads.accumulate(self, grads_step);
            let weight = chunk as f32;
            total_sum = Some(match total_sum {
                Some(accum) => accum + losses.total.clone().mul_scalar(weight),
                None => losses.total.clone().mul_scalar(weight),
            });
            inv_sum = Some(match inv_sum {
                Some(accum) => accum + losses.inv.clone().mul_scalar(weight),
                None => losses.inv.clone().mul_scalar(weight),
            });
            sigreg_sum = Some(match sigreg_sum {
                Some(accum) => accum + losses.sigreg.clone().mul_scalar(weight),
                None => losses.sigreg.clone().mul_scalar(weight),
            });
            recon_sum = Some(match recon_sum {
                Some(accum) => accum + losses.recon.clone().mul_scalar(weight),
                None => losses.recon.clone().mul_scalar(weight),
            });
            consumed += chunk;
        }

        let total = total_sum.expect("repeat loss").mul_scalar(scale);
        let inv = inv_sum.expect("repeat inv").mul_scalar(scale);
        let sigreg = sigreg_sum.expect("repeat sigreg").mul_scalar(scale);
        let recon = recon_sum.expect("repeat recon").mul_scalar(scale);
        let grads = grads.grads();
        let zero = Tensor::<B, 1>::zeros([1], &total.device());
        TrainOutput {
            grads,
            item: VisionTrainItem::new(total, inv, sigreg, recon, zero.clone(), zero),
        }
    }
}

impl<B: BackendTrait> ValidStep<ImageNetBatch<B>, VisionOutput<B>> for VisionSaccadeModel<B> {
    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let backprop_steps = self.rollout.backprop_steps(self.rollout.max_steps);
        let capture_artifacts = self.config.artifact_every > 0;
        let losses = self.forward_losses(
            batch,
            self.rollout.max_steps,
            backprop_steps,
            false,
            capture_artifacts,
        );
        let zero = Tensor::<B, 1>::zeros([1], &losses.total.device());
        VisionOutput::new(
            losses.total,
            losses.inv,
            losses.sigreg,
            losses.recon,
            zero.clone(),
            zero,
            losses.artifacts,
        )
    }
}


