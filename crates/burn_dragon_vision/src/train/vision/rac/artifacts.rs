use serde_json::json;

use crate::model::patchify;
use crate::train::prelude::*;

use super::VisionRacTrajectory;

const RAC_ACTIVITY_EPS: f32 = 1.0e-3;

#[derive(Clone, Copy, Default)]
struct ActivityStats {
    active_fraction: f32,
    positive_fraction: f32,
    effective_rank_proxy: f32,
}

fn straight_path_state<B: BackendTrait>(
    start_state: Tensor<B, 4>,
    target_state: Tensor<B, 4>,
    time_value: f32,
) -> Tensor<B, 4> {
    start_state.mul_scalar(1.0 - time_value) + target_state.mul_scalar(time_value)
}

fn stack_frames<B: BackendTrait>(frames: &[Tensor<B, 4>]) -> Option<Tensor<B, 5>> {
    if frames.is_empty() {
        return None;
    }
    Some(Tensor::cat(
        frames
            .iter()
            .map(|frame| frame.clone().unsqueeze_dim::<5>(1))
            .collect::<Vec<_>>(),
        1,
    ))
}

fn stack_patch_maps<B: BackendTrait>(maps: &[Tensor<B, 3>]) -> Option<Tensor<B, 4>> {
    if maps.is_empty() {
        return None;
    }
    Some(Tensor::cat(
        maps.iter()
            .map(|map| map.clone().unsqueeze_dim::<4>(1))
            .collect::<Vec<_>>(),
        1,
    ))
}

fn stack_pca_maps<B: BackendTrait>(maps: &[Tensor<B, 4>]) -> Option<Tensor<B, 5>> {
    if maps.is_empty() {
        return None;
    }
    Some(Tensor::cat(
        maps.iter()
            .map(|map| map.clone().unsqueeze_dim::<5>(1))
            .collect::<Vec<_>>(),
        1,
    ))
}

fn per_sample_mse4<B: BackendTrait>(
    prediction: Tensor<B, 4>,
    target: Tensor<B, 4>,
) -> Tensor<B, 1> {
    let [batch, ..] = prediction.shape().dims::<4>();
    (prediction - target)
        .powf_scalar(2.0)
        .mean_dim(3)
        .mean_dim(2)
        .mean_dim(1)
        .reshape([batch])
}

fn tensor1_to_f32_vec<B: BackendTrait>(tensor: Tensor<B, 1>) -> Vec<f32> {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default()
}

fn tensor1_to_i64_vec<B: BackendTrait>(tensor: Tensor<B, 1, Int>) -> Vec<i64> {
    tensor
        .to_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .unwrap_or_default()
}

fn tensor2_to_rows<B: BackendTrait>(tensor: Tensor<B, 2>) -> Vec<Vec<f32>> {
    let [batch, width] = tensor.shape().dims::<2>();
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default();
    if width == 0 {
        return vec![Vec::new(); batch];
    }
    (0..batch)
        .map(|sample_idx| {
            let start = sample_idx * width;
            values[start..start + width].to_vec()
        })
        .collect()
}

fn participation_ratio(values: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;
    for value in values {
        let v = value.max(0.0);
        sum += v;
        sum_sq += v * v;
    }
    if sum_sq <= f32::EPSILON {
        0.0
    } else {
        (sum * sum) / sum_sq
    }
}

fn tensor3_activity_rows<B: BackendTrait>(tensor: Tensor<B, 3>) -> Vec<ActivityStats> {
    let [batch, tokens, dims] = tensor.shape().dims::<3>();
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default();
    let stride = tokens.saturating_mul(dims);
    (0..batch)
        .map(|sample_idx| {
            let start = sample_idx.saturating_mul(stride);
            let end = start.saturating_add(stride).min(values.len());
            let slice = &values[start..end];
            if slice.is_empty() || dims == 0 {
                return ActivityStats::default();
            }
            let mut active = 0usize;
            let mut positive = 0usize;
            let mut energies = vec![0.0f32; dims];
            for token_idx in 0..tokens {
                let token_start = token_idx.saturating_mul(dims);
                let token_end = token_start.saturating_add(dims).min(slice.len());
                let token_slice = &slice[token_start..token_end];
                for (dim_idx, value) in token_slice.iter().enumerate() {
                    if value.abs() > RAC_ACTIVITY_EPS {
                        active += 1;
                    }
                    if *value > RAC_ACTIVITY_EPS {
                        positive += 1;
                    }
                    energies[dim_idx] += value * value;
                }
            }
            let denom = slice.len().max(1) as f32;
            ActivityStats {
                active_fraction: active as f32 / denom,
                positive_fraction: positive as f32 / denom,
                effective_rank_proxy: participation_ratio(&energies),
            }
        })
        .collect()
}

fn tensor2_activity_rows<B: BackendTrait>(tensor: Tensor<B, 2>) -> Vec<ActivityStats> {
    let [batch, dims] = tensor.shape().dims::<2>();
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .unwrap_or_default();
    (0..batch)
        .map(|sample_idx| {
            let start = sample_idx.saturating_mul(dims);
            let end = start.saturating_add(dims).min(values.len());
            let slice = &values[start..end];
            if slice.is_empty() {
                return ActivityStats::default();
            }
            let mut active = 0usize;
            let mut positive = 0usize;
            let mut energies = vec![0.0f32; slice.len()];
            for (idx, value) in slice.iter().enumerate() {
                if value.abs() > RAC_ACTIVITY_EPS {
                    active += 1;
                }
                if *value > RAC_ACTIVITY_EPS {
                    positive += 1;
                }
                energies[idx] = value * value;
            }
            let denom = slice.len().max(1) as f32;
            ActivityStats {
                active_fraction: active as f32 / denom,
                positive_fraction: positive as f32 / denom,
                effective_rank_proxy: participation_ratio(&energies),
            }
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn build_sidecar_json<B: BackendTrait>(
    target_state: Tensor<B, 4>,
    init_state: Tensor<B, 4>,
    forward: &VisionRacTrajectory<B>,
    reverse: &VisionRacTrajectory<B>,
    roundtrip: &VisionRacTrajectory<B>,
    labels: Tensor<B, 1, Int>,
    probe_logits: Tensor<B, 2>,
    memory_components: &[String],
    artifact_max_images: usize,
    reset_each_step: bool,
    detach_each_step: bool,
    flow_backprop_steps: Option<usize>,
    disable_writes: bool,
    eval_wipe_after_step: Option<usize>,
) -> String {
    let image_count = artifact_max_images.max(1);
    let sample_count = target_state.shape().dims::<4>()[0].min(image_count);
    let labels_vec = tensor1_to_i64_vec(labels);
    let probe_preds = tensor1_to_i64_vec(
        probe_logits
            .clone()
            .argmax(1)
            .reshape([probe_logits.shape().dims::<2>()[0]]),
    );
    let forward_final = tensor1_to_f32_vec(per_sample_mse4(
        forward.final_state.clone(),
        target_state.clone(),
    ));
    let reverse_to_init = tensor1_to_f32_vec(per_sample_mse4(
        reverse.final_state.clone(),
        init_state.clone(),
    ));
    let roundtrip_final = tensor1_to_f32_vec(per_sample_mse4(
        roundtrip.final_state.clone(),
        target_state.clone(),
    ));

    let mut step_reference_mse = Vec::with_capacity(forward.step_tokens.len());
    let mut step_reverse_reference_mse = Vec::with_capacity(forward.step_tokens.len());
    let mut step_roundtrip_target_mse = Vec::with_capacity(forward.step_tokens.len());
    let mut step_memory_read = Vec::with_capacity(forward.step_tokens.len());
    let mut step_memory_write = Vec::with_capacity(forward.step_tokens.len());
    let mut step_token_activity = Vec::with_capacity(forward.step_tokens.len());
    let mut step_summary_activity = Vec::with_capacity(forward.step_tokens.len());
    for step_idx in 0..forward.step_tokens.len() {
        let state = forward
            .states
            .get(step_idx)
            .cloned()
            .unwrap_or_else(|| forward.final_state.clone());
        let time_value = *forward.state_times.get(step_idx).unwrap_or(&0.0);
        let reference = straight_path_state(init_state.clone(), target_state.clone(), time_value);
        step_reference_mse.push(tensor1_to_f32_vec(per_sample_mse4(
            state,
            reference.clone(),
        )));
        let reverse_idx = reverse.states.len().saturating_sub(step_idx + 1);
        let reverse_state = reverse
            .states
            .get(reverse_idx)
            .cloned()
            .unwrap_or_else(|| reverse.final_state.clone());
        step_reverse_reference_mse.push(tensor1_to_f32_vec(per_sample_mse4(
            reverse_state,
            reference,
        )));
        let roundtrip_state = roundtrip
            .states
            .get(step_idx)
            .cloned()
            .unwrap_or_else(|| roundtrip.final_state.clone());
        step_roundtrip_target_mse.push(tensor1_to_f32_vec(per_sample_mse4(
            roundtrip_state,
            target_state.clone(),
        )));
        step_memory_read.push(tensor2_to_rows(
            forward.step_memory_read_norms[step_idx].clone(),
        ));
        step_memory_write.push(tensor2_to_rows(
            forward.step_memory_write_norms[step_idx].clone(),
        ));
        step_token_activity.push(tensor3_activity_rows(forward.step_tokens[step_idx].clone()));
        step_summary_activity.push(tensor2_activity_rows(
            forward.step_summaries[step_idx].clone(),
        ));
    }

    let samples = (0..sample_count)
        .map(|sample_idx| {
            let steps = (0..forward.step_tokens.len())
                .map(|step_idx| {
                    json!({
                        "step": step_idx,
                        "time": *forward.state_times.get(step_idx).unwrap_or(&0.0),
                        "reference_mse": step_reference_mse
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .copied()
                            .unwrap_or_default(),
                        "reverse_reference_mse": step_reverse_reference_mse
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .copied()
                            .unwrap_or_default(),
                        "roundtrip_target_mse": step_roundtrip_target_mse
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .copied()
                            .unwrap_or_default(),
                        "memory_read_norm": step_memory_read
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .cloned()
                            .unwrap_or_default(),
                        "memory_write_norm": step_memory_write
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .cloned()
                            .unwrap_or_default(),
                        "token_active_fraction": step_token_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.active_fraction)
                            .unwrap_or_default(),
                        "token_positive_fraction": step_token_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.positive_fraction)
                            .unwrap_or_default(),
                        "token_effective_rank_proxy": step_token_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.effective_rank_proxy)
                            .unwrap_or_default(),
                        "summary_active_fraction": step_summary_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.active_fraction)
                            .unwrap_or_default(),
                        "summary_positive_fraction": step_summary_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.positive_fraction)
                            .unwrap_or_default(),
                        "summary_effective_rank_proxy": step_summary_activity
                            .get(step_idx)
                            .and_then(|values| values.get(sample_idx))
                            .map(|stats| stats.effective_rank_proxy)
                            .unwrap_or_default(),
                    })
                })
                .collect::<Vec<_>>();

            let pred = probe_preds.get(sample_idx).copied().unwrap_or(-1);
            let label = labels_vec.get(sample_idx).copied().unwrap_or(-1);
            json!({
                "sample_index": sample_idx,
                "label": label,
                "probe_pred": pred,
                "probe_correct": pred == label,
                "forward_final_mse": forward_final.get(sample_idx).copied().unwrap_or_default(),
                "reverse_to_init_mse": reverse_to_init.get(sample_idx).copied().unwrap_or_default(),
                "roundtrip_mse": roundtrip_final.get(sample_idx).copied().unwrap_or_default(),
                "steps": steps,
            })
        })
        .collect::<Vec<_>>();

    serde_json::to_string_pretty(&json!({
        "schema": "vision_rac_artifacts_v2",
        "memory_components": memory_components,
        "memory_mode": {
            "reset_each_step": reset_each_step,
            "detach_each_step": detach_each_step,
            "flow_backprop_steps": flow_backprop_steps,
            "disable_writes": disable_writes,
            "eval_wipe_after_step": eval_wipe_after_step,
        },
        "samples": samples,
    }))
    .unwrap_or_else(|_| "{}".to_string())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn build_rac_artifacts<B: BackendTrait>(
    target_state: Tensor<B, 4>,
    init_state: Tensor<B, 4>,
    forward: &VisionRacTrajectory<B>,
    reverse: &VisionRacTrajectory<B>,
    roundtrip: &VisionRacTrajectory<B>,
    labels: Tensor<B, 1, Int>,
    probe_logits: Tensor<B, 2>,
    memory_components: &[String],
    patch_size: usize,
    artifact_max_images: usize,
    artifact_upscale: usize,
    reset_each_step: bool,
    detach_each_step: bool,
    flow_backprop_steps: Option<usize>,
    disable_writes: bool,
    eval_wipe_after_step: Option<usize>,
) -> Option<VisionArtifactInput<B>> {
    let image_count = artifact_max_images.max(1);
    let steps = forward.step_tokens.len();
    if steps == 0 {
        return None;
    }

    let mut solver_frames = Vec::with_capacity(steps);
    let mut reverse_frames = Vec::with_capacity(steps);
    let mut roundtrip_frames = Vec::with_capacity(steps);
    let mut forward_velocity_maps = Vec::with_capacity(steps);
    let mut forward_pca_maps = Vec::with_capacity(steps);
    let mut reverse_velocity_maps = Vec::with_capacity(steps);
    let mut reverse_pca_maps = Vec::with_capacity(steps);
    let mut error_maps = Vec::with_capacity(steps);

    for step_idx in 0..steps {
        let state = forward
            .states
            .get(step_idx)
            .cloned()
            .unwrap_or_else(|| forward.final_state.clone());
        let time_value = *forward.state_times.get(step_idx).unwrap_or(&0.0);
        let reference = straight_path_state(init_state.clone(), target_state.clone(), time_value);
        let error = (state.clone() - reference.clone()).abs();
        let error_patches = patchify(error, patch_size);
        let reverse_idx = reverse.states.len().saturating_sub(step_idx + 1);
        let reverse_state = reverse
            .states
            .get(reverse_idx)
            .cloned()
            .unwrap_or_else(|| reverse.final_state.clone());
        let roundtrip_state = roundtrip
            .states
            .get(step_idx)
            .cloned()
            .unwrap_or_else(|| roundtrip.final_state.clone());
        solver_frames.push(state);
        reverse_frames.push(reverse_state);
        roundtrip_frames.push(roundtrip_state);
        if let Some(map) =
            patch_heatmap_or_norm(forward.step_velocity_patches[step_idx].clone(), image_count)
        {
            forward_velocity_maps.push(map);
        }
        if let Some(map) = pca_patch_rgb(&forward.step_tokens[step_idx], image_count) {
            forward_pca_maps.push(map);
        }
        let reverse_step_idx = reverse
            .step_velocity_patches
            .len()
            .saturating_sub(step_idx + 1);
        let reverse_velocity = reverse
            .step_velocity_patches
            .get(reverse_step_idx)
            .cloned()
            .unwrap_or_else(|| {
                let [batch, tokens, dims] =
                    forward.step_velocity_patches[step_idx].shape().dims::<3>();
                Tensor::<B, 3>::zeros(
                    [batch.min(image_count), tokens, dims],
                    &forward.step_velocity_patches[step_idx].device(),
                )
            });
        if let Some(map) = patch_heatmap_or_norm(reverse_velocity, image_count) {
            reverse_velocity_maps.push(map);
        }
        let reverse_tokens = reverse
            .step_tokens
            .get(reverse_step_idx)
            .cloned()
            .unwrap_or_else(|| {
                let [batch, tokens, dims] = forward.step_tokens[step_idx].shape().dims::<3>();
                Tensor::<B, 3>::zeros(
                    [batch.min(image_count), tokens, dims],
                    &forward.step_tokens[step_idx].device(),
                )
            });
        if let Some(map) = pca_patch_rgb(&reverse_tokens, image_count) {
            reverse_pca_maps.push(map);
        }
        if let Some(map) = patch_heatmap_or_norm(error_patches, image_count) {
            error_maps.push(map);
        }
    }

    let frames = stack_frames(&solver_frames);
    let debug_recon_frames = stack_frames(&reverse_frames);
    let aux_frames = stack_frames(&roundtrip_frames);
    let posterior_patch_norms_steps = stack_patch_maps(&forward_velocity_maps);
    let posterior_pca_rgb_steps = stack_pca_maps(&forward_pca_maps);
    let patch_norms_steps = stack_patch_maps(&reverse_velocity_maps);
    let pca_rgb_steps = stack_pca_maps(&reverse_pca_maps);
    let debug_patch_norms_steps = stack_patch_maps(&error_maps);

    let mut take = image_count;
    if let Some(tensor) = &frames {
        take = take.min(tensor.shape().dims::<5>()[0]);
    }
    if let Some(tensor) = &debug_recon_frames {
        take = take.min(tensor.shape().dims::<5>()[0]);
    }
    if let Some(tensor) = &aux_frames {
        take = take.min(tensor.shape().dims::<5>()[0]);
    }
    if let Some(tensor) = &posterior_patch_norms_steps {
        take = take.min(tensor.shape().dims::<4>()[0]);
    }
    if let Some(tensor) = &posterior_pca_rgb_steps {
        take = take.min(tensor.shape().dims::<5>()[0]);
    }
    if let Some(tensor) = &patch_norms_steps {
        take = take.min(tensor.shape().dims::<4>()[0]);
    }
    if let Some(tensor) = &pca_rgb_steps {
        take = take.min(tensor.shape().dims::<5>()[0]);
    }
    if let Some(tensor) = &debug_patch_norms_steps {
        take = take.min(tensor.shape().dims::<4>()[0]);
    }
    take = take.min(probe_logits.shape().dims::<2>()[0]);
    take = take.min(labels.shape().dims::<1>()[0]);
    if take == 0 {
        return None;
    }

    let sidecar_json = build_sidecar_json(
        target_state.clone(),
        init_state.clone(),
        forward,
        reverse,
        roundtrip,
        labels.clone(),
        probe_logits.clone(),
        memory_components,
        take,
        reset_each_step,
        detach_each_step,
        flow_backprop_steps,
        disable_writes,
        eval_wipe_after_step,
    );

    Some(VisionArtifactInput {
        views: None,
        frames: frames.map(|tensor| tensor.slice_dim(0, 0..take)),
        debug_recon_frames: debug_recon_frames.map(|tensor| tensor.slice_dim(0, 0..take)),
        aux_frames: aux_frames.map(|tensor| tensor.slice_dim(0, 0..take)),
        patch_norms: None,
        pca_rgb: None,
        posterior_patch_norms_steps: posterior_patch_norms_steps
            .map(|tensor| tensor.slice_dim(0, 0..take)),
        posterior_pca_rgb_steps: posterior_pca_rgb_steps.map(|tensor| tensor.slice_dim(0, 0..take)),
        patch_norms_steps: patch_norms_steps.map(|tensor| tensor.slice_dim(0, 0..take)),
        pca_rgb_steps: pca_rgb_steps.map(|tensor| tensor.slice_dim(0, 0..take)),
        debug_patch_norms_steps: debug_patch_norms_steps.map(|tensor| tensor.slice_dim(0, 0..take)),
        debug_pca_rgb_steps: None,
        probe_logits: Some(probe_logits.slice_dim(0, 0..take)),
        labels: Some(labels.slice_dim(0, 0..take)),
        legend: Some(vec![
            "forward_state_x_t".to_string(),
            "forward_velocity_patch_norm_v_t".to_string(),
            "forward_patch_pca_z_t".to_string(),
            "reverse_velocity_patch_norm_v_t".to_string(),
            "reverse_patch_pca_z_t".to_string(),
            "reverse_state_x_t_matched".to_string(),
            "roundtrip_state_x_t".to_string(),
            "forward_reference_error_patch_norm".to_string(),
        ]),
        sidecar_json: Some(sidecar_json),
        artifact_scale: artifact_upscale.max(1),
        prediction_start: Some(0),
    })
}
