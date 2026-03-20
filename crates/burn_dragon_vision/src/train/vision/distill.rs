use crate::loss::{
    WeightedClsDistillTarget, WeightedPatchDistillTarget, weighted_cls_cosine_loss,
    weighted_cls_mse_loss, weighted_patch_mse_loss,
};
use crate::model::vision::VisionDragonOutput;
use crate::train::prelude::*;
use burn::tensor::activation;
use burn::tensor::module::{adaptive_avg_pool2d, interpolate};
use burn::tensor::ops::InterpolateOptions;
use rand::prelude::SliceRandom;
use std::time::Instant;

use super::models::{DistillTeacherModel, VisionDistillModel};

type RolloutMetricTensorArray<B> = [Tensor<B, 1>; VISION_ROLLOUT_HORIZON_COUNT];
type RolloutMetricArrays<B> = (
    RolloutMetricTensorArray<B>,
    RolloutMetricTensorArray<B>,
    RolloutMetricTensorArray<B>,
);
type DistillTermsByStep<B> = Vec<(usize, VisionDistillationLossTerms<B>)>;
type ForwardTrainTermsOutput<B> = (
    VisionDistillationLossTerms<B>,
    Tensor<B, 1>,
    DistillTermsByStep<B>,
);

struct RolloutSupervisionSamplingConfig<'a> {
    frames: usize,
    stride: usize,
    groups: usize,
    explicit_steps: &'a [usize],
    explicit_groups: &'a [Vec<usize>],
    include_step1: bool,
    sampling_power: f32,
}

fn rollout_supervision_steps(
    total_steps: usize,
    frames: usize,
    stride: usize,
    include_step1: bool,
) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }
    let stride = stride.max(1);
    let candidates = (1..=total_steps)
        .filter(|step| {
            *step == total_steps
                || (include_step1 && *step == 1)
                || (*step > 1 && ((*step - 1) % stride) == 0)
        })
        .collect::<Vec<_>>();
    let mut steps = if candidates.len() <= frames.max(1) {
        candidates
    } else {
        select_trajectory_indices(candidates.len(), frames.max(1))
            .into_iter()
            .map(|index| candidates[index])
            .collect::<Vec<_>>()
    };
    steps.push(total_steps);
    steps.sort_unstable();
    steps.dedup();
    steps
}

fn rollout_supervision_explicit_steps(total_steps: usize, steps: &[usize]) -> Vec<usize> {
    if total_steps == 0 {
        return Vec::new();
    }
    let mut filtered = steps
        .iter()
        .copied()
        .filter(|step| *step > 0 && *step <= total_steps)
        .collect::<Vec<_>>();
    if filtered.is_empty() {
        filtered.push(total_steps);
    }
    filtered.sort_unstable();
    filtered.dedup();
    filtered
}

fn sample_rollout_supervision_groups<R: Rng + ?Sized>(
    rollout: &VisionRollout,
    config: RolloutSupervisionSamplingConfig<'_>,
    rng: &mut R,
) -> Vec<Vec<usize>> {
    if !config.explicit_groups.is_empty() {
        let normalized_groups = config
            .explicit_groups
            .iter()
            .map(|steps| rollout_supervision_explicit_steps(rollout.max_steps.max(1), steps))
            .filter(|steps| !steps.is_empty())
            .collect::<Vec<_>>();
        let mut supervision_groups = Vec::with_capacity(config.groups.max(1));
        for _ in 0..config.groups.max(1) {
            let chosen = normalized_groups
                .choose(rng)
                .expect("explicit rollout supervision groups")
                .clone();
            supervision_groups.push(chosen);
        }
        return supervision_groups;
    }
    if !config.explicit_steps.is_empty() {
        let normalized =
            rollout_supervision_explicit_steps(rollout.max_steps.max(1), config.explicit_steps);
        return vec![normalized; config.groups.max(1)];
    }
    let mut supervision_groups = Vec::with_capacity(config.groups.max(1));
    for _ in 0..config.groups.max(1) {
        let sampled_steps = sample_rollout_steps(rollout, config.sampling_power, rng);
        supervision_groups.push(rollout_supervision_steps(
            sampled_steps,
            config.frames,
            config.stride,
            config.include_step1,
        ));
    }
    supervision_groups
}

fn merge_rollout_supervision_groups(groups: &[Vec<usize>]) -> Vec<usize> {
    let mut steps = groups
        .iter()
        .flat_map(|group| group.iter().copied())
        .collect::<Vec<_>>();
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

fn rollout_improvement_penalty<B: BackendTrait>(
    supervision_steps: &[usize],
    terms_by_step: &[(usize, VisionDistillationLossTerms<B>)],
    margin: f32,
) -> Tensor<B, 1> {
    let device = terms_by_step
        .first()
        .map(|(_, terms)| terms.total.device())
        .expect("distill improvement penalty requires at least one evaluated step");
    if supervision_steps.len() < 2 {
        return Tensor::<B, 1>::zeros([1], &device);
    }

    let mut penalty = Tensor::<B, 1>::zeros([1], &device);
    let mut pair_count = 0.0f32;
    let mut previous: Option<&VisionDistillationLossTerms<B>> = None;
    for step in supervision_steps {
        let terms = terms_by_step
            .iter()
            .find(|(candidate, _)| candidate == step)
            .map(|(_, terms)| terms)
            .expect("improvement step should be evaluated");
        if let Some(prev_terms) = previous {
            let step_penalty = activation::relu(
                terms
                    .total
                    .clone()
                    .sub(prev_terms.total.clone())
                    .add_scalar(margin),
            );
            penalty = penalty + step_penalty;
            pair_count += 1.0;
        }
        previous = Some(terms);
    }

    if pair_count == 0.0 {
        Tensor::<B, 1>::zeros([1], &device)
    } else {
        penalty.mul_scalar(1.0 / pair_count)
    }
}

fn aggregate_rollout_supervision_groups<B: BackendTrait>(
    supervision_groups: &[Vec<usize>],
    terms_by_step: &[(usize, VisionDistillationLossTerms<B>)],
    power: f32,
    improvement_weight: f32,
    improvement_margin: f32,
) -> (VisionDistillationLossTerms<B>, Tensor<B, 1>) {
    let device = terms_by_step
        .first()
        .map(|(_, terms)| terms.total.device())
        .expect("distill aggregation requires at least one evaluated step");
    let mut total = Tensor::<B, 1>::zeros([1], &device);
    let mut patch = Tensor::<B, 1>::zeros([1], &device);
    let mut cls = Tensor::<B, 1>::zeros([1], &device);
    let mut relational = Tensor::<B, 1>::zeros([1], &device);
    let mut improvement_penalty = Tensor::<B, 1>::zeros([1], &device);
    let groups = supervision_groups.len().max(1) as f32;

    for supervision_steps in supervision_groups {
        let aggregated = aggregate_rollout_terms(supervision_steps, terms_by_step, power);
        total = total + aggregated.total;
        patch = patch + aggregated.patch;
        cls = cls + aggregated.cls;
        relational = relational + aggregated.relational;
        if improvement_weight > 0.0 {
            improvement_penalty = improvement_penalty
                + rollout_improvement_penalty(supervision_steps, terms_by_step, improvement_margin)
                    .mul_scalar(improvement_weight);
        }
    }

    let inv_groups = 1.0 / groups.max(1.0);
    (
        VisionDistillationLossTerms {
            total: total.mul_scalar(inv_groups),
            patch: patch.mul_scalar(inv_groups),
            cls: cls.mul_scalar(inv_groups),
            relational: relational.mul_scalar(inv_groups),
        },
        improvement_penalty.mul_scalar(inv_groups),
    )
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

fn distill_config_for_teacher_target(
    config: &VisionDistillationLossConfig,
    target_kind: VisionTeacherTargetKind,
    has_patch: bool,
) -> VisionDistillationLossConfig {
    let mut config = config.clone();
    if !matches!(target_kind, VisionTeacherTargetKind::PatchAndCls) || !has_patch {
        config.patch_mse_weight = 0.0;
        config.rel_weight = 0.0;
    }
    config
}

#[cfg(test)]
fn scale_distillation_terms<B: BackendTrait>(
    terms: VisionDistillationLossTerms<B>,
    weight: f32,
) -> VisionDistillationLossTerms<B> {
    VisionDistillationLossTerms {
        total: terms.total.mul_scalar(weight),
        patch: terms.patch.mul_scalar(weight),
        cls: terms.cls.mul_scalar(weight),
        relational: terms.relational.mul_scalar(weight),
    }
}

#[cfg(test)]
fn accumulate_distillation_terms<B: BackendTrait>(
    base: VisionDistillationLossTerms<B>,
    extra: VisionDistillationLossTerms<B>,
) -> VisionDistillationLossTerms<B> {
    VisionDistillationLossTerms {
        total: base.total + extra.total,
        patch: base.patch + extra.patch,
        cls: base.cls + extra.cls,
        relational: base.relational + extra.relational,
    }
}

#[derive(Clone)]
struct PreparedTeacherTarget<B: BackendTrait> {
    student_patch: Option<Tensor<B, 3>>,
    teacher_patch: Option<Tensor<B, 3>>,
    student_cls: Tensor<B, 2>,
    teacher_cls: Tensor<B, 2>,
    patch_weight: f32,
    cls_mse_weight: f32,
    cls_cosine_weight: f32,
    relational_weight: f32,
    rel_tau: f32,
    rel_sample_tokens: Option<usize>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct PatchLossShapeKey {
    tokens: usize,
    dim: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ClsLossShapeKey {
    dim: usize,
}

fn patch_grid_from_tokens(tokens: usize, label: &str) -> PatchGrid {
    let side = (tokens as f64).sqrt().round() as usize;
    assert_eq!(
        side.saturating_mul(side),
        tokens,
        "{label} token count {tokens} must form a square patch grid"
    );
    PatchGrid {
        height: side,
        width: side,
    }
}

fn resample_patch_tokens_to_grid<B: BackendTrait>(
    patch_tokens: Tensor<B, 3>,
    target_grid: PatchGrid,
) -> Tensor<B, 3> {
    let [batch, token_count, dim] = patch_tokens.shape().dims::<3>();
    let source_grid = patch_grid_from_tokens(token_count, "student patch");
    if source_grid == target_grid {
        return patch_tokens;
    }
    let patch_state =
        patch_tokens
            .swap_dims(1, 2)
            .reshape([batch, dim, source_grid.height, source_grid.width]);
    let resized =
        if target_grid.height <= source_grid.height && target_grid.width <= source_grid.width {
            adaptive_avg_pool2d(patch_state, [target_grid.height, target_grid.width])
        } else {
            interpolate(
                patch_state,
                [target_grid.height, target_grid.width],
                InterpolateOptions::new(InterpolateMode::Nearest),
            )
        };
    resized
        .reshape([batch, dim, target_grid.num_patches()])
        .swap_dims(1, 2)
}

#[cfg(test)]
fn cached_resampled_patch_tokens<B: BackendTrait>(
    student_patch: &Tensor<B, 3>,
    target_tokens: usize,
    cache: &mut Vec<(usize, Tensor<B, 3>)>,
) -> Tensor<B, 3> {
    if let Some((_, cached)) = cache.iter().find(|(tokens, _)| *tokens == target_tokens) {
        return cached.clone();
    }
    let resampled = resample_patch_tokens_to_grid(
        student_patch.clone(),
        patch_grid_from_tokens(target_tokens, "teacher patch"),
    );
    cache.push((target_tokens, resampled.clone()));
    resampled
}

fn split_batched_tensor3<B: BackendTrait>(
    tensor: Tensor<B, 3>,
    batch: usize,
    chunks: usize,
) -> Vec<Tensor<B, 3>> {
    let [total_batch, _, _] = tensor.shape().dims::<3>();
    assert_eq!(
        total_batch,
        batch.saturating_mul(chunks),
        "batched patch tensor should split evenly into {chunks} chunks of batch {batch}"
    );
    let mut outputs = Vec::with_capacity(chunks);
    for index in 0..chunks {
        let start = index.saturating_mul(batch);
        let end = start + batch;
        outputs.push(tensor.clone().slice_dim(0, start..end));
    }
    outputs
}

fn split_batched_tensor2<B: BackendTrait>(
    tensor: Tensor<B, 2>,
    batch: usize,
    chunks: usize,
) -> Vec<Tensor<B, 2>> {
    let [total_batch, _] = tensor.shape().dims::<2>();
    assert_eq!(
        total_batch,
        batch.saturating_mul(chunks),
        "batched cls tensor should split evenly into {chunks} chunks of batch {batch}"
    );
    let mut outputs = Vec::with_capacity(chunks);
    for index in 0..chunks {
        let start = index.saturating_mul(batch);
        let end = start + batch;
        outputs.push(tensor.clone().slice_dim(0, start..end));
    }
    outputs
}

#[cfg(test)]
fn prepare_teacher_targets<B: BackendTrait>(
    model: &VisionDistillModel<B>,
    student_patch: Tensor<B, 3>,
    student_cls: Tensor<B, 2>,
    teacher_patch: Tensor<B, 3>,
    teacher_cls: Tensor<B, 2>,
    teacher_targets: &[ImageNetTeacherTargetBatch<B>],
    config: &VisionDistillationLossConfig,
) -> Vec<PreparedTeacherTarget<B>> {
    let mut prepared = Vec::with_capacity(teacher_targets.len() + 1);
    let mut resampled_patch_cache = Vec::new();
    prepared.push(PreparedTeacherTarget {
        student_patch: Some(student_patch.clone()),
        teacher_patch: Some(teacher_patch),
        student_cls: student_cls.clone(),
        teacher_cls,
        patch_weight: config.patch_mse_weight,
        cls_mse_weight: config.cls_mse_weight,
        cls_cosine_weight: config.cls_cosine_weight,
        relational_weight: config.rel_weight,
        rel_tau: config.rel_tau,
        rel_sample_tokens: config.rel_sample_tokens,
    });

    for target in teacher_targets {
        if target.weight <= 0.0 {
            continue;
        }
        let (patch_head, cls_head, decoder_mode, target_patch_tokens) = model
            .auxiliary_teacher_head(&target.name)
            .map(|(decoder, decoder_mode, target_patch_tokens)| {
                (
                    decoder.patch_head.as_ref(),
                    Some(&decoder.cls_head),
                    decoder_mode,
                    target_patch_tokens,
                )
            })
            .unwrap_or((None, None, VisionTeacherDecoderMode::SharedProjection, None));
        let target_config =
            distill_config_for_teacher_target(config, target.target_kind, target.patch.is_some());
        let patch_enabled = target_config.patch_mse_weight > 0.0 || target_config.rel_weight > 0.0;
        let student_patch_target = patch_enabled.then(|| {
            let patch_input = if decoder_mode.supports_spatial_resampling() {
                let target_tokens = target_patch_tokens.unwrap_or_else(|| {
                    target
                        .patch
                        .as_ref()
                        .map(|patch| patch.shape().dims::<3>()[1])
                        .unwrap_or(student_patch.shape().dims::<3>()[1])
                });
                cached_resampled_patch_tokens(
                    &student_patch,
                    target_tokens,
                    &mut resampled_patch_cache,
                )
            } else {
                student_patch.clone()
            };
            patch_head
                .map(|head| head.forward(patch_input.clone()))
                .unwrap_or(patch_input)
        });
        let student_cls_target = cls_head
            .map(|head| head.forward(student_cls.clone()))
            .unwrap_or_else(|| student_cls.clone());
        prepared.push(PreparedTeacherTarget {
            student_patch: student_patch_target,
            teacher_patch: patch_enabled.then(|| {
                target
                    .patch
                    .clone()
                    .expect("auxiliary spatial teacher patch targets required")
            }),
            student_cls: student_cls_target,
            teacher_cls: target.cls.clone(),
            patch_weight: target.weight * target_config.patch_mse_weight,
            cls_mse_weight: target.weight * target_config.cls_mse_weight,
            cls_cosine_weight: target.weight * target_config.cls_cosine_weight,
            relational_weight: target.weight * target_config.rel_weight,
            rel_tau: target_config.rel_tau,
            rel_sample_tokens: target_config.rel_sample_tokens,
        });
    }
    prepared
}

fn prepare_teacher_targets_many<B: BackendTrait>(
    model: &VisionDistillModel<B>,
    student_outputs: Vec<(usize, VisionDragonOutput<B>)>,
    teacher_patch: Tensor<B, 3>,
    teacher_cls: Tensor<B, 2>,
    teacher_targets: &[ImageNetTeacherTargetBatch<B>],
    config: &VisionDistillationLossConfig,
) -> Vec<(usize, Vec<PreparedTeacherTarget<B>>)> {
    if student_outputs.is_empty() {
        return Vec::new();
    }

    let chunk_batch = student_outputs[0].1.patch_tokens.shape().dims::<3>()[0];
    let output_count = student_outputs.len();
    let mut prepared_by_output = student_outputs
        .iter()
        .map(|(step, output)| {
            (
                *step,
                vec![PreparedTeacherTarget {
                    student_patch: Some(output.patch_tokens.clone()),
                    teacher_patch: Some(teacher_patch.clone()),
                    student_cls: output.cls_token.clone(),
                    teacher_cls: teacher_cls.clone(),
                    patch_weight: config.patch_mse_weight,
                    cls_mse_weight: config.cls_mse_weight,
                    cls_cosine_weight: config.cls_cosine_weight,
                    relational_weight: config.rel_weight,
                    rel_tau: config.rel_tau,
                    rel_sample_tokens: config.rel_sample_tokens,
                }],
            )
        })
        .collect::<Vec<_>>();

    for target in teacher_targets {
        if target.weight <= 0.0 {
            continue;
        }
        let (patch_head, cls_head, decoder_mode, target_patch_tokens) = model
            .auxiliary_teacher_head(&target.name)
            .map(|(decoder, decoder_mode, target_patch_tokens)| {
                (
                    decoder.patch_head.as_ref(),
                    Some(&decoder.cls_head),
                    decoder_mode,
                    target_patch_tokens,
                )
            })
            .unwrap_or((None, None, VisionTeacherDecoderMode::SharedProjection, None));
        let target_config =
            distill_config_for_teacher_target(config, target.target_kind, target.patch.is_some());
        let patch_enabled = target_config.patch_mse_weight > 0.0 || target_config.rel_weight > 0.0;

        let student_cls_inputs = Tensor::cat(
            student_outputs
                .iter()
                .map(|(_, output)| output.cls_token.clone())
                .collect(),
            0,
        );
        let student_cls_projected = cls_head
            .map(|head| head.forward(student_cls_inputs.clone()))
            .unwrap_or(student_cls_inputs);
        let student_cls_chunks =
            split_batched_tensor2(student_cls_projected, chunk_batch, output_count);

        let student_patch_chunks = if patch_enabled {
            let patch_inputs = student_outputs
                .iter()
                .map(|(_, output)| {
                    if decoder_mode.supports_spatial_resampling() {
                        let target_tokens = target_patch_tokens.unwrap_or_else(|| {
                            target
                                .patch
                                .as_ref()
                                .map(|patch| patch.shape().dims::<3>()[1])
                                .unwrap_or(output.patch_tokens.shape().dims::<3>()[1])
                        });
                        resample_patch_tokens_to_grid(
                            output.patch_tokens.clone(),
                            patch_grid_from_tokens(target_tokens, "teacher patch"),
                        )
                    } else {
                        output.patch_tokens.clone()
                    }
                })
                .collect::<Vec<_>>();
            let packed_patch_inputs = Tensor::cat(patch_inputs, 0);
            let packed_patch_projected = patch_head
                .map(|head| head.forward(packed_patch_inputs.clone()))
                .unwrap_or(packed_patch_inputs);
            Some(split_batched_tensor3(
                packed_patch_projected,
                chunk_batch,
                output_count,
            ))
        } else {
            None
        };

        for index in 0..output_count {
            prepared_by_output[index].1.push(PreparedTeacherTarget {
                student_patch: student_patch_chunks
                    .as_ref()
                    .map(|chunks| chunks[index].clone()),
                teacher_patch: patch_enabled.then(|| {
                    target
                        .patch
                        .clone()
                        .expect("auxiliary spatial teacher patch targets required")
                }),
                student_cls: student_cls_chunks[index].clone(),
                teacher_cls: target.cls.clone(),
                patch_weight: target.weight * target_config.patch_mse_weight,
                cls_mse_weight: target.weight * target_config.cls_mse_weight,
                cls_cosine_weight: target.weight * target_config.cls_cosine_weight,
                relational_weight: target.weight * target_config.rel_weight,
                rel_tau: target_config.rel_tau,
                rel_sample_tokens: target_config.rel_sample_tokens,
            });
        }
    }

    prepared_by_output
}

fn grouped_patch_terms<B: BackendTrait>(
    prepared_targets: &[PreparedTeacherTarget<B>],
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut groups: Vec<(PatchLossShapeKey, Vec<WeightedPatchDistillTarget<B>>)> = Vec::new();
    for target in prepared_targets {
        if target.patch_weight <= 0.0 {
            continue;
        }
        let student_patch = target
            .student_patch
            .as_ref()
            .expect("patch loss requires student patch targets")
            .clone();
        let teacher_patch = target
            .teacher_patch
            .as_ref()
            .expect("patch loss requires teacher patch targets")
            .clone();
        let [_, tokens, dim] = student_patch.shape().dims::<3>();
        let key = PatchLossShapeKey { tokens, dim };
        if let Some((_, grouped)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
            grouped.push(WeightedPatchDistillTarget {
                student: student_patch,
                teacher: teacher_patch,
                weight: target.patch_weight,
            });
        } else {
            groups.push((
                key,
                vec![WeightedPatchDistillTarget {
                    student: student_patch,
                    teacher: teacher_patch,
                    weight: target.patch_weight,
                }],
            ));
        }
    }

    groups
        .into_iter()
        .fold(Tensor::<B, 1>::zeros([1], device), |total, (_, grouped)| {
            total + weighted_patch_mse_loss(&grouped)
        })
}

fn grouped_cls_mse_terms<B: BackendTrait>(
    prepared_targets: &[PreparedTeacherTarget<B>],
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut groups: Vec<(ClsLossShapeKey, Vec<WeightedClsDistillTarget<B>>)> = Vec::new();
    for target in prepared_targets {
        if target.cls_mse_weight <= 0.0 {
            continue;
        }
        let [_, dim] = target.student_cls.shape().dims::<2>();
        let key = ClsLossShapeKey { dim };
        if let Some((_, grouped)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
            grouped.push(WeightedClsDistillTarget {
                student: target.student_cls.clone(),
                teacher: target.teacher_cls.clone(),
                weight: target.cls_mse_weight,
            });
        } else {
            groups.push((
                key,
                vec![WeightedClsDistillTarget {
                    student: target.student_cls.clone(),
                    teacher: target.teacher_cls.clone(),
                    weight: target.cls_mse_weight,
                }],
            ));
        }
    }

    groups
        .into_iter()
        .fold(Tensor::<B, 1>::zeros([1], device), |total, (_, grouped)| {
            total + weighted_cls_mse_loss(&grouped)
        })
}

fn grouped_cls_cosine_terms<B: BackendTrait>(
    prepared_targets: &[PreparedTeacherTarget<B>],
    device: &B::Device,
) -> Tensor<B, 1> {
    let mut groups: Vec<(ClsLossShapeKey, Vec<WeightedClsDistillTarget<B>>)> = Vec::new();
    for target in prepared_targets {
        if target.cls_cosine_weight <= 0.0 {
            continue;
        }
        let [_, dim] = target.student_cls.shape().dims::<2>();
        let key = ClsLossShapeKey { dim };
        if let Some((_, grouped)) = groups.iter_mut().find(|(candidate, _)| *candidate == key) {
            grouped.push(WeightedClsDistillTarget {
                student: target.student_cls.clone(),
                teacher: target.teacher_cls.clone(),
                weight: target.cls_cosine_weight,
            });
        } else {
            groups.push((
                key,
                vec![WeightedClsDistillTarget {
                    student: target.student_cls.clone(),
                    teacher: target.teacher_cls.clone(),
                    weight: target.cls_cosine_weight,
                }],
            ));
        }
    }

    groups
        .into_iter()
        .fold(Tensor::<B, 1>::zeros([1], device), |total, (_, grouped)| {
            total + weighted_cls_cosine_loss(&grouped)
        })
}

fn relational_terms<B: BackendTrait>(
    prepared_targets: &[PreparedTeacherTarget<B>],
    device: &B::Device,
) -> Tensor<B, 1> {
    prepared_targets
        .iter()
        .filter(|target| target.relational_weight > 0.0)
        .fold(Tensor::<B, 1>::zeros([1], device), |total, target| {
            let rel_config = VisionDistillationLossConfig {
                patch_mse_weight: 0.0,
                cls_mse_weight: 0.0,
                cls_cosine_weight: 0.0,
                rel_weight: target.relational_weight,
                rel_tau: target.rel_tau,
                rel_sample_tokens: target.rel_sample_tokens,
            };
            let rel_terms = vision_distillation_loss_terms(
                target
                    .student_patch
                    .as_ref()
                    .expect("relational loss requires student patch targets")
                    .clone(),
                target
                    .teacher_patch
                    .as_ref()
                    .expect("relational loss requires teacher patch targets")
                    .clone(),
                target.student_cls.clone(),
                target.teacher_cls.clone(),
                &rel_config,
            );
            total + rel_terms.relational
        })
}

fn aggregate_prepared_teacher_targets<B: BackendTrait>(
    prepared_targets: Vec<PreparedTeacherTarget<B>>,
) -> VisionDistillationLossTerms<B> {
    let device = prepared_targets
        .first()
        .map(|target| target.student_cls.device())
        .expect("distill aggregation requires at least one prepared target");
    let patch = grouped_patch_terms(&prepared_targets, &device);
    let cls_mse = grouped_cls_mse_terms(&prepared_targets, &device);
    let cls_cosine = grouped_cls_cosine_terms(&prepared_targets, &device);
    let cls = cls_mse + cls_cosine;
    let relational = relational_terms(&prepared_targets, &device);
    VisionDistillationLossTerms {
        total: patch.clone() + cls.clone() + relational.clone(),
        patch,
        cls,
        relational,
    }
}

impl<B: BackendTrait> VisionDistillModel<B> {
    fn evaluate_distill_steps_bounded(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        teacher_targets: &[ImageNetTeacherTargetBatch<B>],
        steps: &[usize],
    ) -> Vec<(usize, VisionDistillationLossTerms<B>)> {
        let schedule = steps
            .iter()
            .map(|step| (*step, self.rollout.backprop_steps(*step)))
            .collect::<Vec<_>>();
        let rollout_outputs = self
            .model
            .forward_images_steps_rollout_schedule(images, &schedule);
        prepare_teacher_targets_many(
            self,
            rollout_outputs,
            teacher_patch,
            teacher_cls,
            teacher_targets,
            &self.loss,
        )
        .into_iter()
        .map(|(step, prepared)| (step, aggregate_prepared_teacher_targets(prepared)))
        .collect()
    }

    fn evaluate_distill_steps_unbounded(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        teacher_targets: &[ImageNetTeacherTargetBatch<B>],
        steps: &[usize],
    ) -> Vec<(usize, VisionDistillationLossTerms<B>)> {
        let schedule = steps
            .iter()
            .map(|step| (*step, self.rollout.backprop_steps(*step)))
            .collect::<Vec<_>>();
        let rollout_outputs = self
            .model
            .forward_images_steps_rollout_schedule_unbounded(images, &schedule);
        prepare_teacher_targets_many(
            self,
            rollout_outputs,
            teacher_patch,
            teacher_cls,
            teacher_targets,
            &self.loss,
        )
        .into_iter()
        .map(|(step, prepared)| (step, aggregate_prepared_teacher_targets(prepared)))
        .collect()
    }
}

impl<B: AutodiffBackend> VisionDistillModel<B> {
    fn forward_train_terms(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        teacher_targets: &[ImageNetTeacherTargetBatch<B>],
    ) -> ForwardTrainTermsOutput<B> {
        let mut rng = thread_rng();
        let supervision_groups = sample_rollout_supervision_groups(
            &self.rollout,
            RolloutSupervisionSamplingConfig {
                frames: self.rollout_supervision_frames,
                stride: self.rollout_supervision_stride,
                groups: self.rollout_supervision_groups,
                explicit_steps: &self.rollout_supervision_explicit_steps,
                explicit_groups: &self.rollout_supervision_explicit_groups,
                include_step1: self.rollout_supervision_include_step1,
                sampling_power: self.rollout_sampling_power,
            },
            &mut rng,
        );
        let evaluated_steps = merge_rollout_supervision_groups(&supervision_groups);
        let terms_by_step = self.evaluate_distill_steps_bounded(
            images,
            teacher_patch,
            teacher_cls,
            teacher_targets,
            &evaluated_steps,
        );
        let (aggregated, improvement_penalty) = aggregate_rollout_supervision_groups(
            &supervision_groups,
            &terms_by_step,
            self.rollout_supervision_power,
            self.rollout_improvement_weight,
            self.rollout_improvement_margin,
        );
        (aggregated, improvement_penalty, terms_by_step)
    }

    #[cfg_attr(
        not(any(test, feature = "benchmark")),
        expect(dead_code, reason = "benchmark-only helper")
    )]
    pub(crate) fn forward_train_total_loss(&self, batch: ImageNetBatch<B>) -> Tensor<B, 1> {
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            teacher_targets,
            ..
        } = batch;

        let teacher_images = images.clone();
        let (teacher_patch, teacher_cls) =
            teacher_targets_train(&self.teacher, teacher_images, teacher_patch, teacher_cls);
        let (aggregated, improvement_penalty, _) =
            self.forward_train_terms(images, teacher_patch, teacher_cls, &teacher_targets);

        aggregated.total + improvement_penalty
    }
}

impl<B: AutodiffBackend> TrainStep for VisionDistillModel<B> {
    type Input = ImageNetBatch<B>;
    type Output = VisionTrainItem<B>;

    fn step(&self, batch: ImageNetBatch<B>) -> TrainOutput<VisionTrainItem<B>> {
        let prof_enabled = crate::train::profile::enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let ImageNetBatch {
            images,
            teacher_patch,
            teacher_cls,
            teacher_targets,
            ..
        } = batch;

        let teacher_images = images.clone();
        let (teacher_patch, teacher_cls) =
            teacher_targets_train(&self.teacher, teacher_images, teacher_patch, teacher_cls);
        let (aggregated, improvement_penalty, terms_by_step) =
            self.forward_train_terms(images, teacher_patch, teacher_cls, &teacher_targets);
        let total = aggregated.total.clone() + improvement_penalty;
        let (rollout_total, rollout_patch, rollout_cls) =
            rollout_metric_arrays(&aggregated.total.device(), &terms_by_step);
        if crate::train::profile::sync_timing_enabled() {
            let _ = B::sync(&aggregated.total.device());
        }
        let forward_ns = forward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        let loss_backward_start = prof_enabled.then(Instant::now);
        let grads = total.clone().backward();
        if crate::train::profile::sync_timing_enabled() {
            let _ = B::sync(&total.device());
        }
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        if prof_enabled {
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
        }
        let zero = Tensor::<B, 1>::zeros([1], &aggregated.total.device());

        let item = VisionTrainItem::new(
            total,
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

    fn optimize<BB, O>(self, optim: &mut O, lr: f64, grads: GradientsParams) -> Self
    where
        BB: AutodiffBackend,
        O: burn::optim::Optimizer<Self, BB>,
        Self: AutodiffModule<BB>,
    {
        let prof_enabled = crate::train::profile::enabled();
        let optimizer_start = prof_enabled.then(Instant::now);
        let updated = optim.step(lr, self, grads);
        if crate::train::profile::sync_timing_enabled() {
            let _ = BB::sync(&BB::Device::default());
        }
        let optimizer_ns = optimizer_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();
        if prof_enabled {
            crate::train::profile::record_optimizer_step(optimizer_ns);
        }
        updated
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
            teacher_targets,
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
            &teacher_targets,
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
    use burn_dragon_core::{FusedAttentionExecutor, FusedKernelConfig};
    use burn_ndarray::NdArray;

    type Backend = Autodiff<NdArray<f32>>;

    fn make_distill_model(
        device: &<Backend as BackendTrait>::Device,
        steps: usize,
    ) -> VisionDistillModel<Backend> {
        make_distill_model_with_targets_and_executor(
            device,
            steps,
            Vec::new(),
            FusedAttentionExecutor::AttentionContext,
        )
    }

    fn make_distill_model_with_executor(
        device: &<Backend as BackendTrait>::Device,
        steps: usize,
        attention_executor: FusedAttentionExecutor,
    ) -> VisionDistillModel<Backend> {
        make_distill_model_with_targets_and_executor(device, steps, Vec::new(), attention_executor)
    }

    fn make_distill_model_with_targets(
        device: &<Backend as BackendTrait>::Device,
        steps: usize,
        teacher_targets: Vec<VisionTeacherTargetConfig>,
    ) -> VisionDistillModel<Backend> {
        make_distill_model_with_targets_and_executor(
            device,
            steps,
            teacher_targets,
            FusedAttentionExecutor::AttentionContext,
        )
    }

    fn make_distill_model_with_targets_and_executor(
        device: &<Backend as BackendTrait>::Device,
        steps: usize,
        teacher_targets: Vec<VisionTeacherTargetConfig>,
        attention_executor: FusedAttentionExecutor,
    ) -> VisionDistillModel<Backend> {
        let mut fused_kernels = FusedKernelConfig::default();
        fused_kernels.attention_executor = attention_executor;
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
            fused_kernels,
            trm_graph: Default::default(),
            rho_stream: Default::default(),
            ..VisionDragonConfig::default()
        };
        VisionDistillModel::new(
            VisionDragon::<Backend>::new(vision, device),
            VisionDistillConfig {
                teacher_targets,
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
            device,
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

    fn teacher_batch_with_auxiliary_cls_target(
        teacher: &VisionDragon<Backend>,
        images: Tensor<Backend, 4>,
        steps: usize,
        weight: f32,
    ) -> ImageNetBatch<Backend> {
        let teacher_output = teacher.forward_images_steps_rollout(images.clone(), steps, steps);
        let batch = teacher_batch(teacher, images, steps);
        batch.with_teacher_targets(vec![ImageNetTeacherTargetBatch {
            name: "siglip2_global".to_string(),
            weight,
            target_kind: VisionTeacherTargetKind::GlobalOnly,
            patch: None,
            cls: teacher_output.cls_token.mul_scalar(0.75),
        }])
    }

    fn teacher_batch_with_auxiliary_spatial_target(
        teacher: &VisionDragon<Backend>,
        images: Tensor<Backend, 4>,
        steps: usize,
        weight: f32,
    ) -> ImageNetBatch<Backend> {
        let teacher_output = teacher.forward_images_steps_rollout(images.clone(), steps, steps);
        let batch = teacher_batch(teacher, images, steps);
        let teacher_patch = teacher_output
            .patch_tokens
            .mean_dim(1)
            .reshape([teacher_output.cls_token.shape().dims::<2>()[0], 1, 12])
            .slice_dim(2, 0..7);
        let teacher_cls = teacher_output.cls_token.slice_dim(1, 0..7);
        batch.with_teacher_targets(vec![ImageNetTeacherTargetBatch {
            name: "siglip2_spatial".to_string(),
            weight,
            target_kind: VisionTeacherTargetKind::PatchAndCls,
            patch: Some(teacher_patch),
            cls: teacher_cls,
        }])
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
            &[],
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
    fn distill_supports_auxiliary_global_teacher_targets() {
        let device = Default::default();
        let model = make_distill_model(&device, 4);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch_with_auxiliary_cls_target(&model.model, images, 4, 0.5);

        let loss = model
            .forward_train_total_loss(batch)
            .into_data()
            .to_vec::<f32>()
            .expect("train loss")[0];
        assert!(
            loss.is_finite(),
            "multi-target distill loss should stay finite"
        );
    }

    #[test]
    fn distill_supports_auxiliary_spatial_teacher_targets_with_dedicated_decoder() {
        let device = Default::default();
        let model = make_distill_model_with_targets(
            &device,
            4,
            vec![VisionTeacherTargetConfig {
                name: "siglip2_spatial".to_string(),
                weight: 0.5,
                target_kind: VisionTeacherTargetKind::PatchAndCls,
                decoder_mode: VisionTeacherDecoderMode::DedicatedSpatialProjection,
                decoder_hidden_dim: Some(16),
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: "unused/train_cls.bin".into(),
                    train_patch_path: Some("unused/train_patch.bin".into()),
                    val_cls_path: "unused/val_cls.bin".into(),
                    val_patch_path: Some("unused/val_patch.bin".into()),
                    feature_dim: 7,
                    patch_tokens: Some(1),
                }),
            }],
        );
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch_with_auxiliary_spatial_target(&model.model, images, 4, 0.5);

        let loss = model
            .forward_train_total_loss(batch)
            .into_data()
            .to_vec::<f32>()
            .expect("train loss")[0];
        assert!(
            loss.is_finite(),
            "dedicated spatial auxiliary distill loss should stay finite"
        );
        assert_eq!(model.auxiliary_teacher_heads.len(), 1);
        assert!(model.auxiliary_teacher_heads[0].patch_head.is_some());
    }

    #[test]
    fn grouped_teacher_target_pipeline_matches_sequential_reference() {
        let device = Default::default();
        let model = make_distill_model_with_targets(
            &device,
            4,
            vec![VisionTeacherTargetConfig {
                name: "siglip2_spatial".to_string(),
                weight: 0.5,
                target_kind: VisionTeacherTargetKind::PatchAndCls,
                decoder_mode: VisionTeacherDecoderMode::DedicatedSpatialProjection,
                decoder_hidden_dim: Some(16),
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: "unused/train_cls.bin".into(),
                    train_patch_path: Some("unused/train_patch.bin".into()),
                    val_cls_path: "unused/val_cls.bin".into(),
                    val_patch_path: Some("unused/val_patch.bin".into()),
                    feature_dim: 7,
                    patch_tokens: Some(1),
                }),
            }],
        );
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch_with_auxiliary_spatial_target(&model.model, images, 4, 0.5);
        let (teacher_patch, teacher_cls) =
            teacher_targets_valid(batch.teacher_patch.clone(), batch.teacher_cls.clone());
        let output = model
            .model
            .forward_images_steps_rollout_schedule(
                batch.images.clone(),
                &[(4, model.rollout.backprop_steps(4))],
            )
            .into_iter()
            .next()
            .expect("step 4 output")
            .1;

        let grouped = aggregate_prepared_teacher_targets(prepare_teacher_targets(
            &model,
            output.patch_tokens.clone(),
            output.cls_token.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &batch.teacher_targets,
            &model.loss,
        ));

        let mut reference = vision_distillation_loss_terms(
            output.patch_tokens.clone(),
            teacher_patch,
            output.cls_token.clone(),
            teacher_cls,
            &model.loss,
        );
        for target in &batch.teacher_targets {
            let (patch_head, cls_head, decoder_mode, target_patch_tokens) = model
                .auxiliary_teacher_head(&target.name)
                .map(|(decoder, decoder_mode, target_patch_tokens)| {
                    (
                        decoder.patch_head.as_ref(),
                        Some(&decoder.cls_head),
                        decoder_mode,
                        target_patch_tokens,
                    )
                })
                .unwrap_or((None, None, VisionTeacherDecoderMode::SharedProjection, None));
            let patch_enabled = matches!(target.target_kind, VisionTeacherTargetKind::PatchAndCls)
                && target.patch.is_some();
            let student_patch_target = patch_enabled.then(|| {
                let patch_input = if decoder_mode.supports_spatial_resampling() {
                    let target_tokens = target_patch_tokens
                        .or_else(|| {
                            target
                                .patch
                                .as_ref()
                                .map(|patch| patch.shape().dims::<3>()[1])
                        })
                        .unwrap_or(output.patch_tokens.shape().dims::<3>()[1]);
                    resample_patch_tokens_to_grid(
                        output.patch_tokens.clone(),
                        patch_grid_from_tokens(target_tokens, "teacher patch"),
                    )
                } else {
                    output.patch_tokens.clone()
                };
                patch_head
                    .map(|head| head.forward(patch_input.clone()))
                    .unwrap_or(patch_input)
            });
            let student_cls_target = cls_head
                .map(|head| head.forward(output.cls_token.clone()))
                .unwrap_or_else(|| output.cls_token.clone());
            let target_config = distill_config_for_teacher_target(
                &model.loss,
                target.target_kind,
                target.patch.is_some(),
            );
            let extra_terms = vision_distillation_loss_terms(
                student_patch_target
                    .clone()
                    .unwrap_or_else(|| Tensor::<Backend, 3>::zeros([2, 1, 7], &device)),
                target.patch.clone().unwrap_or_else(|| {
                    student_patch_target.expect("teacher patch target").detach()
                }),
                student_cls_target,
                target.cls.clone(),
                &target_config,
            );
            reference = accumulate_distillation_terms(
                reference,
                scale_distillation_terms(extra_terms, target.weight),
            );
        }

        for (lhs, rhs) in [
            (grouped.total, reference.total),
            (grouped.patch, reference.patch),
            (grouped.cls, reference.cls),
            (grouped.relational, reference.relational),
        ] {
            let lhs = lhs
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("lhs")[0];
            let rhs = rhs
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("rhs")[0];
            assert!((lhs - rhs).abs() < 1e-5, "{lhs} vs {rhs}");
        }
    }

    #[test]
    fn batched_rollout_teacher_target_pipeline_matches_sequential_reference() {
        let device = Default::default();
        let model = make_distill_model_with_targets(
            &device,
            4,
            vec![VisionTeacherTargetConfig {
                name: "siglip2_spatial".to_string(),
                weight: 0.5,
                target_kind: VisionTeacherTargetKind::PatchAndCls,
                decoder_mode: VisionTeacherDecoderMode::DedicatedSpatialProjection,
                decoder_hidden_dim: Some(16),
                teacher: VisionTeacherConfig::Features(VisionTeacherFeatureConfig {
                    train_cls_path: "unused/train_cls.bin".into(),
                    train_patch_path: Some("unused/train_patch.bin".into()),
                    val_cls_path: "unused/val_cls.bin".into(),
                    val_patch_path: Some("unused/val_patch.bin".into()),
                    feature_dim: 7,
                    patch_tokens: Some(1),
                }),
            }],
        );
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch_with_auxiliary_spatial_target(&model.model, images, 4, 0.5);
        let (teacher_patch, teacher_cls) =
            teacher_targets_valid(batch.teacher_patch.clone(), batch.teacher_cls.clone());
        let rollout_outputs = model.model.forward_images_steps_rollout_schedule(
            batch.images.clone(),
            &[
                (2, model.rollout.backprop_steps(2)),
                (4, model.rollout.backprop_steps(4)),
            ],
        );

        let grouped = prepare_teacher_targets_many(
            &model,
            rollout_outputs.clone(),
            teacher_patch.clone(),
            teacher_cls.clone(),
            &batch.teacher_targets,
            &model.loss,
        )
        .into_iter()
        .map(|(step, prepared)| (step, aggregate_prepared_teacher_targets(prepared)))
        .collect::<Vec<_>>();

        let sequential = rollout_outputs
            .into_iter()
            .map(|(step, output)| {
                let prepared = prepare_teacher_targets(
                    &model,
                    output.patch_tokens,
                    output.cls_token,
                    teacher_patch.clone(),
                    teacher_cls.clone(),
                    &batch.teacher_targets,
                    &model.loss,
                );
                (step, aggregate_prepared_teacher_targets(prepared))
            })
            .collect::<Vec<_>>();

        assert_eq!(grouped.len(), sequential.len());
        for ((grouped_step, grouped_terms), (sequential_step, sequential_terms)) in
            grouped.into_iter().zip(sequential.into_iter())
        {
            assert_eq!(grouped_step, sequential_step);
            for (lhs, rhs) in [
                (grouped_terms.total, sequential_terms.total),
                (grouped_terms.patch, sequential_terms.patch),
                (grouped_terms.cls, sequential_terms.cls),
                (grouped_terms.relational, sequential_terms.relational),
            ] {
                let lhs = lhs
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("lhs")[0];
                let rhs = rhs
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("rhs")[0];
                assert!((lhs - rhs).abs() < 1e-5, "{lhs} vs {rhs}");
            }
        }
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
                    &[],
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
                .evaluate_distill_steps_bounded(batch.images, teacher_patch, teacher_cls, &[], &[4])
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
    fn rollout_supervision_stride_selects_sparse_blocks_and_keeps_final_step() {
        assert_eq!(
            rollout_supervision_steps(8, 8, 1, true),
            vec![1, 2, 3, 4, 5, 6, 7, 8]
        );
        assert_eq!(
            rollout_supervision_steps(8, 8, 2, true),
            vec![1, 3, 5, 7, 8]
        );
        assert_eq!(rollout_supervision_steps(8, 3, 2, true), vec![1, 5, 8]);
        assert_eq!(rollout_supervision_steps(8, 2, 4, true), vec![1, 8]);
    }

    #[test]
    fn rollout_supervision_groups_merge_and_dedup_steps() {
        let groups = vec![vec![1, 3, 4], vec![1, 2, 4], vec![2, 4]];
        assert_eq!(merge_rollout_supervision_groups(&groups), vec![1, 2, 3, 4]);
    }

    #[test]
    fn rollout_supervision_stride_has_no_effect_when_frames_is_one() {
        assert_eq!(rollout_supervision_steps(8, 1, 1, true), vec![8]);
        assert_eq!(rollout_supervision_steps(8, 1, 2, true), vec![8]);
        assert_eq!(rollout_supervision_steps(8, 1, 3, true), vec![8]);
        assert_eq!(rollout_supervision_steps(8, 1, 4, true), vec![8]);
        assert_eq!(rollout_supervision_steps(8, 1, 4, false), vec![8]);
    }

    #[test]
    fn rollout_supervision_can_skip_step_one() {
        assert_eq!(rollout_supervision_steps(8, 4, 3, false), vec![4, 7, 8]);
        assert_eq!(rollout_supervision_steps(8, 2, 3, false), vec![4, 8]);
        assert_eq!(rollout_supervision_steps(4, 4, 3, false), vec![4]);
    }

    #[test]
    fn rollout_supervision_explicit_steps_clip_and_dedup() {
        assert_eq!(
            rollout_supervision_explicit_steps(8, &[8, 4, 4, 12, 0, 2]),
            vec![2, 4, 8]
        );
        assert_eq!(rollout_supervision_explicit_steps(8, &[0, 9]), vec![8]);
        assert_eq!(
            rollout_supervision_explicit_steps(0, &[1, 2, 3]),
            Vec::<usize>::new()
        );
    }

    #[test]
    fn rollout_supervision_explicit_steps_override_sampling_groups() {
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 12,
            backprop_steps: 4,
        };
        let mut rng = StdRng::seed_from_u64(11);
        let groups = sample_rollout_supervision_groups(
            &rollout,
            RolloutSupervisionSamplingConfig {
                frames: 4,
                stride: 3,
                groups: 2,
                explicit_steps: &[4, 8, 12],
                explicit_groups: &[],
                include_step1: true,
                sampling_power: 1.25,
            },
            &mut rng,
        );
        assert_eq!(groups, vec![vec![4, 8, 12], vec![4, 8, 12]]);
    }

    #[test]
    fn rollout_supervision_explicit_groups_sample_candidates() {
        let rollout = VisionRollout {
            min_steps: 1,
            max_steps: 12,
            backprop_steps: 4,
        };
        let mut rng = StdRng::seed_from_u64(7);
        let groups = sample_rollout_supervision_groups(
            &rollout,
            RolloutSupervisionSamplingConfig {
                frames: 4,
                stride: 3,
                groups: 4,
                explicit_steps: &[],
                explicit_groups: &[vec![4, 8], vec![2, 4, 8]],
                include_step1: false,
                sampling_power: 1.25,
            },
            &mut rng,
        );
        assert_eq!(groups.len(), 4);
        for group in groups {
            assert!(group == vec![4, 8] || group == vec![2, 4, 8]);
        }
    }

    #[test]
    fn grouped_rollout_aggregation_matches_single_group_contract() {
        let device = Default::default();
        let model = make_distill_model(&device, 4);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let batch = teacher_batch(&model.model, images.clone(), 4);
        let (teacher_patch, teacher_cls) =
            teacher_targets_valid(batch.teacher_patch, batch.teacher_cls);
        let supervision_steps = vec![1, 4];
        let terms_by_step = model.evaluate_distill_steps_bounded(
            images,
            teacher_patch,
            teacher_cls,
            &[],
            &supervision_steps,
        );

        let expected_terms = aggregate_rollout_terms(
            &supervision_steps,
            &terms_by_step,
            model.rollout_supervision_power,
        );
        let expected_penalty = rollout_improvement_penalty(
            &supervision_steps,
            &terms_by_step,
            model.rollout_improvement_margin,
        )
        .mul_scalar(model.rollout_improvement_weight);
        let (grouped_terms, grouped_penalty) = aggregate_rollout_supervision_groups(
            &[supervision_steps],
            &terms_by_step,
            model.rollout_supervision_power,
            model.rollout_improvement_weight,
            model.rollout_improvement_margin,
        );

        let expected_total = expected_terms
            .total
            .into_data()
            .to_vec::<f32>()
            .expect("expected total")[0];
        let grouped_total = grouped_terms
            .total
            .into_data()
            .to_vec::<f32>()
            .expect("grouped total")[0];
        let expected_patch = expected_terms
            .patch
            .into_data()
            .to_vec::<f32>()
            .expect("expected patch")[0];
        let grouped_patch = grouped_terms
            .patch
            .into_data()
            .to_vec::<f32>()
            .expect("grouped patch")[0];
        let expected_penalty = expected_penalty
            .into_data()
            .to_vec::<f32>()
            .expect("expected penalty")[0];
        let grouped_penalty = grouped_penalty
            .into_data()
            .to_vec::<f32>()
            .expect("grouped penalty")[0];

        assert!((expected_total - grouped_total).abs() <= 1e-6);
        assert!((expected_patch - grouped_patch).abs() <= 1e-6);
        assert!((expected_penalty - grouped_penalty).abs() <= 1e-6);
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
            &[],
            &[1, 2, 4, 8],
        );

        let steps = terms.into_iter().map(|(step, _)| step).collect::<Vec<_>>();
        assert_eq!(steps, vec![1, 2, 4, 8]);
    }

    fn assert_tensor_close_vec(actual: Tensor<Backend, 3>, expected: Tensor<Backend, 3>, tol: f32) {
        let actual = actual
            .into_data()
            .to_vec::<f32>()
            .expect("actual tensor data");
        let expected = expected
            .into_data()
            .to_vec::<f32>()
            .expect("expected tensor data");
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "tensor mismatch at index {index}: actual={a}, expected={b}, tol={tol}"
            );
        }
    }

    fn assert_tensor_close_cls(actual: Tensor<Backend, 2>, expected: Tensor<Backend, 2>, tol: f32) {
        let actual = actual.into_data().to_vec::<f32>().expect("actual cls data");
        let expected = expected
            .into_data()
            .to_vec::<f32>()
            .expect("expected cls data");
        assert_eq!(actual.len(), expected.len());
        for (index, (a, b)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (a - b).abs() <= tol,
                "cls mismatch at index {index}: actual={a}, expected={b}, tol={tol}"
            );
        }
    }

    #[test]
    fn dense_rollout_schedule_matches_repeated_public_rollout() {
        let device = Default::default();
        let model = make_distill_model(&device, 8);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let schedule = vec![(2usize, 2usize), (4usize, 3usize), (8usize, 3usize)];

        let scheduled = model
            .model
            .forward_images_steps_rollout_schedule(images.clone(), &schedule);

        assert_eq!(scheduled.len(), schedule.len());
        for ((scheduled_step, scheduled_output), (step, backprop_steps)) in
            scheduled.into_iter().zip(schedule.into_iter())
        {
            assert_eq!(scheduled_step, step);
            let repeated =
                model
                    .model
                    .forward_images_steps_rollout(images.clone(), step, backprop_steps);
            assert_tensor_close_vec(scheduled_output.patch_tokens, repeated.patch_tokens, 1e-6);
            assert_tensor_close_cls(scheduled_output.cls_token, repeated.cls_token, 1e-6);
        }
    }

    #[test]
    fn dense_scores_only_schedule_matches_repeated_public_rollout() {
        let device = Default::default();
        let model =
            make_distill_model_with_executor(&device, 8, FusedAttentionExecutor::ScoresOnly);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let schedule = vec![(2usize, 2usize), (4usize, 3usize), (8usize, 3usize)];

        let scheduled = model
            .model
            .forward_images_steps_rollout_schedule(images.clone(), &schedule);

        assert_eq!(scheduled.len(), schedule.len());
        for ((scheduled_step, scheduled_output), (step, backprop_steps)) in
            scheduled.into_iter().zip(schedule.into_iter())
        {
            assert_eq!(scheduled_step, step);
            let repeated =
                model
                    .model
                    .forward_images_steps_rollout(images.clone(), step, backprop_steps);
            assert_tensor_close_vec(scheduled_output.patch_tokens, repeated.patch_tokens, 1e-6);
            assert_tensor_close_cls(scheduled_output.cls_token, repeated.cls_token, 1e-6);
        }
    }

    #[test]
    fn dense_scores_only_schedule_matches_repeated_public_rollout_broader_mixed_backprop() {
        let device = Default::default();
        let model =
            make_distill_model_with_executor(&device, 8, FusedAttentionExecutor::ScoresOnly);
        let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
        let schedule = vec![
            (1usize, 1usize),
            (2usize, 2usize),
            (4usize, 3usize),
            (8usize, 3usize),
        ];

        let scheduled = model
            .model
            .forward_images_steps_rollout_schedule(images.clone(), &schedule);

        assert_eq!(scheduled.len(), schedule.len());
        for ((scheduled_step, scheduled_output), (step, backprop_steps)) in
            scheduled.into_iter().zip(schedule.into_iter())
        {
            assert_eq!(scheduled_step, step);
            let repeated =
                model
                    .model
                    .forward_images_steps_rollout(images.clone(), step, backprop_steps);
            assert_tensor_close_vec(scheduled_output.patch_tokens, repeated.patch_tokens, 1e-6);
            assert_tensor_close_cls(scheduled_output.cls_token, repeated.cls_token, 1e-6);
        }
    }

    fn make_distill_model_for_resolution(
        device: &<Backend as BackendTrait>::Device,
        image_size: usize,
        patch_size: usize,
        steps: usize,
    ) -> VisionDistillModel<Backend> {
        let grid = image_size.div_ceil(patch_size).max(1);
        let vision = VisionDragonConfig {
            image_size,
            patch_size,
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
            pos_max_height: grid,
            pos_max_width: grid,
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
            device,
        )
    }

    #[test]
    fn distill_supports_multiple_square_resolutions() {
        let device = Default::default();

        for &image_size in &[8usize, 12, 16] {
            let patch_size = 4usize;
            let steps = 4usize;
            let grid = image_size.div_ceil(patch_size).max(1);
            let expected_tokens = grid * grid;
            let model = make_distill_model_for_resolution(&device, image_size, patch_size, steps);
            let images = Tensor::<Backend, 4>::random(
                [2, 3, image_size, image_size],
                Distribution::Default,
                &device,
            );
            let batch = teacher_batch(&model.model, images.clone(), steps);
            let patch_shape = batch
                .teacher_patch
                .as_ref()
                .expect("teacher patch tokens")
                .shape()
                .dims::<3>();
            assert_eq!(patch_shape[1], expected_tokens);

            let loss = model
                .forward_train_total_loss(batch.clone())
                .into_data()
                .to_vec::<f32>()
                .expect("train loss")[0];
            assert!(
                loss.is_finite(),
                "loss should be finite at image_size={image_size}"
            );

            let (teacher_patch, teacher_cls) =
                teacher_targets_valid(batch.teacher_patch, batch.teacher_cls);
            let terms = model.evaluate_distill_steps_unbounded(
                images,
                teacher_patch,
                teacher_cls,
                &[],
                &[1, steps],
            );
            let final_total = terms
                .iter()
                .find(|(step, _)| *step == steps)
                .map(|(_, terms)| terms.total.clone())
                .expect("final step total")
                .into_data()
                .to_vec::<f32>()
                .expect("final total")[0];
            assert!(
                final_total.is_finite(),
                "final rollout total should be finite at image_size={image_size}"
            );
        }
    }
}
