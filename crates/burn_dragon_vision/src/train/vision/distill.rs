use crate::train::prelude::*;

use super::models::{DistillTeacherModel, VisionDistillModel};

type RolloutMetricTensorArray<B> = [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT];
type RolloutMetricArrays<B> = (
    RolloutMetricTensorArray<B>,
    RolloutMetricTensorArray<B>,
    RolloutMetricTensorArray<B>,
);

fn rollout_supervision_steps(total_steps: usize, frames: usize) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }
    let mut steps = select_trajectory_indices(total_steps, frames.max(1))
        .into_iter()
        .map(|index| index + 1)
        .collect::<Vec<_>>();
    steps.push(total_steps);
    steps.sort_unstable();
    steps.dedup();
    steps
}

fn rollout_metric_caps_unbounded() -> Vec<usize> {
    VISION_ROLLOUT_HORIZON_CAPS.into_iter().collect()
}

fn merge_rollout_steps(primary: &[usize], metric_caps: &[usize], final_step: usize) -> Vec<usize> {
    let mut steps = primary.to_vec();
    steps.extend(metric_caps.iter().copied());
    steps.push(final_step.max(1));
    steps.sort_unstable();
    steps.dedup();
    steps
}

fn step_weight(step: usize, power: f32) -> f32 {
    if power == 0.0 {
        1.0
    } else {
        (step.max(1) as f32).powf(power)
    }
}

fn sample_rollout_steps<R: Rng + ?Sized>(
    rollout: &VisionRollout,
    sampling_power: f32,
    rng: &mut R,
) -> usize {
    if rollout.min_steps >= rollout.max_steps {
        return rollout.max_steps.max(1);
    }
    if sampling_power <= 0.0 {
        return rng.gen_range(rollout.min_steps..=rollout.max_steps);
    }

    let mut total_weight = 0.0f32;
    for step in rollout.min_steps..=rollout.max_steps {
        total_weight += (step.max(1) as f32).powf(sampling_power);
    }
    let mut draw = rng.gen_range(0.0..total_weight.max(f32::EPSILON));
    for step in rollout.min_steps..=rollout.max_steps {
        draw -= (step.max(1) as f32).powf(sampling_power);
        if draw <= 0.0 {
            return step;
        }
    }

    rollout.max_steps.max(1)
}

fn zero_rollout_metric_arrays<B: BackendTrait>(device: &B::Device) -> RolloutMetricArrays<B> {
    let total = core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], device));
    let patch = core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], device));
    let cls = core::array::from_fn(|_| Tensor::<B, 1>::zeros([1], device));
    (total, patch, cls)
}

fn rollout_metric_arrays<B: BackendTrait>(
    device: &B::Device,
    terms_by_step: &[(usize, VisionDistillationLossTerms<B>)],
) -> RolloutMetricArrays<B> {
    let (mut total, mut patch, mut cls) = zero_rollout_metric_arrays(device);
    for (index, step) in VISION_ROLLOUT_HORIZON_CAPS.into_iter().enumerate() {
        if let Some((_, terms)) = terms_by_step
            .iter()
            .find(|(candidate, _)| *candidate == step)
        {
            total[index] = terms.total.clone();
            patch[index] = terms.patch.clone();
            cls[index] = terms.cls.clone();
        }
    }
    (total, patch, cls)
}

fn aggregate_rollout_terms<B: BackendTrait>(
    supervision_steps: &[usize],
    terms_by_step: &[(usize, VisionDistillationLossTerms<B>)],
    power: f32,
) -> VisionDistillationLossTerms<B> {
    let device = terms_by_step
        .first()
        .map(|(_, terms)| terms.total.device())
        .expect("distill aggregation requires at least one evaluated step");
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let mut patch = Tensor::<B, 1>::zeros([1], &device);
    let mut cls = Tensor::<B, 1>::zeros([1], &device);
    let mut relational = Tensor::<B, 1>::zeros([1], &device);
    let mut weight_sum = 0.0f32;

    for step in supervision_steps {
        let terms = terms_by_step
            .iter()
            .find(|(candidate, _)| candidate == step)
            .map(|(_, terms)| terms)
            .expect("supervision step should be evaluated");
        let weight = step_weight(*step, power);
        weight_sum += weight;
        total = total + terms.total.clone().mul_scalar(weight);
        patch = patch + terms.patch.clone().mul_scalar(weight);
        cls = cls + terms.cls.clone().mul_scalar(weight);
        relational = relational + terms.relational.clone().mul_scalar(weight);
    }

    let inv_weight = 1.0f32 / weight_sum.max(1e-6);
    VisionDistillationLossTerms {
        total: total.mul_scalar(inv_weight),
        patch: patch.mul_scalar(inv_weight),
        cls: cls.mul_scalar(inv_weight),
        relational: relational.mul_scalar(inv_weight),
    }
}

fn teacher_targets_train<B: AutodiffBackend>(
    teacher: &Option<DistillTeacherModel<B>>,
    images: Tensor<B, 4>,
    teacher_patch: Option<Tensor<B, 3>>,
    teacher_cls: Option<Tensor<B, 2>>,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    #[cfg(feature = "burn_dino")]
    if let Some(teacher) = teacher {
        let output = teacher.forward(images, None);
        return (output.x_norm_patchtokens, output.x_norm_clstoken);
    }

    #[cfg(not(feature = "burn_dino"))]
    {
        let _ = teacher;
        let _ = images;
    }

    let teacher_patch = teacher_patch.expect("teacher patch features required");
    let teacher_cls = teacher_cls.expect("teacher cls features required");
    (teacher_patch, teacher_cls)
}

fn teacher_targets_valid<B: BackendTrait>(
    teacher_patch: Option<Tensor<B, 3>>,
    teacher_cls: Option<Tensor<B, 2>>,
) -> (Tensor<B, 3>, Tensor<B, 2>) {
    let teacher_patch = teacher_patch.expect("teacher patch features required");
    let teacher_cls = teacher_cls.expect("teacher cls features required");
    (teacher_patch, teacher_cls)
}

impl<B: BackendTrait> VisionDistillModel<B> {
    fn evaluate_distill_steps_bounded(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        steps: &[usize],
    ) -> Vec<(usize, VisionDistillationLossTerms<B>)> {
        let mut outputs = Vec::with_capacity(steps.len());
        for step in steps {
            let backprop_steps = self.rollout.backprop_steps(*step);
            let output =
                self.model
                    .forward_images_steps_rollout(images.clone(), *step, backprop_steps);
            let terms = vision_distillation_loss_terms(
                output.patch_tokens,
                teacher_patch.clone(),
                output.cls_token,
                teacher_cls.clone(),
                &self.loss,
            );
            outputs.push((*step, terms));
        }
        outputs
    }

    fn evaluate_distill_steps_unbounded(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        steps: &[usize],
    ) -> Vec<(usize, VisionDistillationLossTerms<B>)> {
        let mut outputs = Vec::with_capacity(steps.len());
        for step in steps {
            let backprop_steps = self.rollout.backprop_steps(*step);
            let output = self.model.forward_images_steps_rollout_unbounded(
                images.clone(),
                *step,
                backprop_steps,
            );
            let terms = vision_distillation_loss_terms(
                output.patch_tokens,
                teacher_patch.clone(),
                output.cls_token,
                teacher_cls.clone(),
                &self.loss,
            );
            outputs.push((*step, terms));
        }
        outputs
    }
}

impl<B: AutodiffBackend> TrainStep for VisionDistillModel<B> {
    type Input = ImageNetBatch<B>;
    type Output = VisionTrainItem<B>;

    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let teacher_images = images.clone();
        let (teacher_patch, teacher_cls) =
            teacher_targets_train(&self.teacher, teacher_images, teacher_patch, teacher_cls);

        let mut rng = thread_rng();
        let sampled_steps =
            sample_rollout_steps(&self.rollout, self.rollout_sampling_power, &mut rng);
        let supervision_steps =
            rollout_supervision_steps(sampled_steps, self.rollout_supervision_frames);
        let evaluated_steps = supervision_steps.clone();
        let terms_by_step = self.evaluate_distill_steps_bounded(
            images,
            teacher_patch,
            teacher_cls,
            &evaluated_steps,
        );
        let aggregated = aggregate_rollout_terms(
            &supervision_steps,
            &terms_by_step,
            self.rollout_supervision_power,
        );
        let (rollout_total, rollout_patch, rollout_cls) =
            rollout_metric_arrays(&aggregated.total.device(), &terms_by_step);
        let grads = aggregated.total.clone().backward();
        let zero = Tensor::<B, 1>::zeros([1], &aggregated.total.device());

        let item = VisionTrainItem::new(
            aggregated.total,
            aggregated.patch,
            aggregated.cls,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        )
        .with_rollout_horizon_metrics(rollout_total, rollout_patch, rollout_cls);

        TrainOutput::new(self, grads, item)
    }
}

impl<B: BackendTrait> ValidStep for VisionDistillModel<B> {
    type Input = ImageNetBatch<B>;
    type Output = VisionOutput<B>;

    fn step(&self, batch: ImageNetBatch<B>) -> VisionOutput<B> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            ..
        } = batch;

        let (teacher_patch, teacher_cls) = teacher_targets_valid(teacher_patch, teacher_cls);
        let metric_caps = rollout_metric_caps_unbounded();
        let evaluated_steps =
            merge_rollout_steps(&metric_caps, &metric_caps, self.rollout.max_steps);
        let terms_by_step = self.evaluate_distill_steps_unbounded(
            images,
            teacher_patch,
            teacher_cls,
            &evaluated_steps,
        );
        let final_terms = terms_by_step
            .iter()
            .find(|(step, _)| *step == self.rollout.max_steps)
            .map(|(_, terms)| terms.clone())
            .expect("final rollout step should be evaluated");
        let (rollout_total, rollout_patch, rollout_cls) =
            rollout_metric_arrays(&final_terms.total.device(), &terms_by_step);
        let zero = Tensor::<B, 1>::zeros([1], &final_terms.total.device());

        VisionOutput::new(
            final_terms.total,
            final_terms.patch,
            final_terms.cls,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
            None,
        )
        .with_rollout_horizon_metrics(rollout_total, rollout_patch, rollout_cls)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VisionBackboneKind;
    use burn::optim::{AdamWConfig, Optimizer};
    use burn::tensor::Distribution;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::FusedKernelConfig;
    use burn_ndarray::NdArray;

    type Backend = Autodiff<NdArray<f32>>;

    fn make_distill_model(
        device: &<Backend as BackendTrait>::Device,
        steps: usize,
    ) -> VisionDistillModel<Backend> {
        let vision = VisionDragonConfig {
            image_size: 8,
            patch_size: 4,
            backbone: VisionBackboneKind::Dense,
            in_channels: 3,
            embed_dim: 16,
            steps,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            dropout: 0.0,
            projection_dim: 12,
            projection_hidden_dim: 24,
            use_cls_token: true,
            pos_encoding: SpatialPositionalEncodingKind::Rope,
            pos_max_height: 2,
            pos_max_width: 2,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: FusedKernelConfig::default(),
            trm_graph: Default::default(),
            rho_stream: Default::default(),
            ..VisionDragonConfig::default()
        };
        VisionDistillModel::new(
            VisionDragon::<Backend>::new(vision, device),
            VisionDistillConfig {
                rollout_supervision_frames: 3,
                rollout_supervision_power: 1.0,
                rollout_sampling_power: 0.0,
                ..VisionDistillConfig::default()
            },
            None,
            VisionRollout {
                min_steps: steps,
                max_steps: steps,
                backprop_steps: steps,
            },
        )
    }

    fn teacher_batch(
        teacher: &VisionDragon<Backend>,
        images: Tensor<Backend, 4>,
        steps: usize,
    ) -> ImageNetBatch<Backend> {
        let teacher_output = teacher.forward_images_steps_rollout(images.clone(), steps, steps);
        let batch_size = images.shape().dims::<4>()[0];
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &images.device());
        ImageNetBatch::new(
            images,
            None,
            None,
            None,
            None,
            None,
            labels,
            Some(teacher_output.patch_tokens),
            Some(teacher_output.cls_token),
        )
    }

    #[test]
    fn distill_valid_metrics_improve_with_more_rollout_steps_when_teacher_matches_final_step() {
        let device = Default::default();
        let model = make_distill_model(&device, 4);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch(&model.model, images.clone(), 4);
        let (teacher_patch, teacher_cls) =
            teacher_targets_valid(batch.teacher_patch, batch.teacher_cls);
        let metric_caps = VISION_ROLLOUT_HORIZON_CAPS
            .into_iter()
            .filter(|step| *step <= model.rollout.max_steps)
            .collect::<Vec<_>>();
        let evaluated_steps =
            merge_rollout_steps(&metric_caps, &metric_caps, model.rollout.max_steps);
        let terms = model.evaluate_distill_steps_bounded(
            images,
            teacher_patch,
            teacher_cls,
            &evaluated_steps,
        );
        let final_loss = terms
            .iter()
            .find(|(step, _)| *step == 4)
            .map(|(_, terms)| terms.total.clone())
            .expect("final loss")
            .into_data()
            .to_vec::<f32>()
            .expect("final")[0];
        let step1 = terms
            .iter()
            .find(|(step, _)| *step == 1)
            .map(|(_, terms)| terms.total.clone())
            .expect("step1")
            .into_data()
            .to_vec::<f32>()
            .expect("step1")[0];
        let step2 = terms
            .iter()
            .find(|(step, _)| *step == 2)
            .map(|(_, terms)| terms.total.clone())
            .expect("step2")
            .into_data()
            .to_vec::<f32>()
            .expect("step2")[0];
        let step4 = terms
            .iter()
            .find(|(step, _)| *step == 4)
            .map(|(_, terms)| terms.total.clone())
            .expect("step4")
            .into_data()
            .to_vec::<f32>()
            .expect("step4")[0];

        assert!(final_loss <= step1 + 1e-6);
        assert!(step4 <= step2 + 1e-6);
    }

    #[test]
    fn distill_train_step_reduces_loss_on_repeated_teacher_batch() {
        let device = Default::default();
        let teacher_model = make_distill_model(&device, 4).model;
        let mut model = make_distill_model(&device, 4);
        let mut optimizer = AdamWConfig::new().init();
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch(&teacher_model, images, 4);

        let before = {
            let (teacher_patch, teacher_cls) =
                teacher_targets_valid(batch.teacher_patch.clone(), batch.teacher_cls.clone());
            model
                .evaluate_distill_steps_bounded(
                    batch.images.clone(),
                    teacher_patch,
                    teacher_cls,
                    &[4],
                )
                .remove(0)
                .1
                .total
                .into_data()
                .to_vec::<f32>()
                .expect("before")[0]
        };
        for _ in 0..3 {
            let train = TrainStep::step(&model, batch.clone());
            model = optimizer.step(0.02, model, train.grads);
        }
        let after = {
            let (teacher_patch, teacher_cls) =
                teacher_targets_valid(batch.teacher_patch, batch.teacher_cls);
            model
                .evaluate_distill_steps_bounded(batch.images, teacher_patch, teacher_cls, &[4])
                .remove(0)
                .1
                .total
                .into_data()
                .to_vec::<f32>()
                .expect("after")[0]
        };
        assert!(after <= before);
    }

    #[test]
    fn rollout_sampling_power_biases_toward_deeper_steps() {
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 4,
            backprop_steps: 2,
        };
        let mut uniform_rng = StdRng::seed_from_u64(7);
        let mut biased_rng = StdRng::seed_from_u64(7);
        let trials = 4096usize;

        let uniform_mean = (0..trials)
            .map(|_| sample_rollout_steps(&rollout, 0.0, &mut uniform_rng) as f32)
            .sum::<f32>()
            / trials as f32;
        let biased_mean = (0..trials)
            .map(|_| sample_rollout_steps(&rollout, 1.5, &mut biased_rng) as f32)
            .sum::<f32>()
            / trials as f32;

        assert!(biased_mean > uniform_mean + 0.35);
    }

    #[test]
    fn distill_unbounded_eval_emits_metrics_beyond_train_horizon() {
        let device = Default::default();
        let model = make_distill_model(&device, 4);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch(&model.model, images.clone(), 4);
        let (teacher_patch, teacher_cls) =
            teacher_targets_valid(batch.teacher_patch, batch.teacher_cls);

        let terms = model.evaluate_distill_steps_unbounded(
            images,
            teacher_patch,
            teacher_cls,
            &[1, 2, 4, 8],
        );

        let steps = terms.into_iter().map(|(step, _)| step).collect::<Vec<_>>();
        assert_eq!(steps, vec![1, 2, 4, 8]);
    }
}
