use super::*;
use crate::train::vision::video::dynamics::embed_clip_frames_raw_with_model;

#[test]
fn video_lejepa_forward_losses_are_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let losses = model.forward_losses(batch, 2, 2, false, false, true);
    for value in [
        losses.total,
        losses.inv,
        losses.observe,
        losses.mode_separation_ratio,
        losses.sigreg,
        losses.recon,
        losses.recon_psnr_masked,
        losses.recon_psnr_full,
        losses.probe_loss,
        losses.probe_acc,
    ] {
        let scalar = value
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scalar")[0];
        assert!(scalar.is_finite());
    }
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn video_lejepa_trm_backbone_forward_losses_are_finite() {
    type Backend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let losses = model.forward_losses(batch, 2, 2, false, false, true);
    for value in [
        losses.total,
        losses.inv,
        losses.observe,
        losses.mode_separation_ratio,
        losses.sigreg,
        losses.recon,
        losses.recon_psnr_masked,
        losses.recon_psnr_full,
        losses.probe_loss,
        losses.probe_acc,
    ] {
        let scalar = value
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("scalar")[0];
        assert!(scalar.is_finite());
    }
}

#[test]
fn video_lejepa_trm_backbone_does_not_construct_temporal_tower_or_future_queries() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    assert!(model.temporal_model.is_none());
    assert!(model.future_queries.is_none());
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 2);

    let reference = model.forward_video(batch.clone(), 2, 2, 6);

    let mut perturbed = model.clone();
    perturbed.temporal_model = Some(BDH::<Backend>::new(
        build_temporal_config(
            perturbed.embed_dim,
            &perturbed.config.temporal,
            &vision.normalization,
        ),
        &device,
    ));
    perturbed.future_queries = Some(Param::from_tensor(Tensor::<Backend, 2>::random(
        [
            perturbed.config.max_supervised_target_frames(),
            perturbed.embed_dim,
        ],
        TensorDistribution::Normal(5.0, 1.0),
        &device,
    )));
    let changed = perturbed.forward_video(batch, 2, 2, 6);

    assert_close(reference.predicted_proj, changed.predicted_proj, 1e-6, 1e-6);
    assert_close(
        reference.future_patch_tokens,
        changed.future_patch_tokens,
        1e-6,
        1e-6,
    );
    assert_close(
        reference.future_cls_embed,
        changed.future_cls_embed,
        1e-6,
        1e-6,
    );
}

#[test]
fn video_lejepa_non_trm_backbone_keeps_temporal_tower_and_future_queries() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    assert!(model.temporal_model.is_some());
    assert!(model.future_queries.is_some());
}

#[test]
fn video_lejepa_trm_backbone_refine_passes_change_context_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.temporal.refine_passes = 1;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 2);
    let forward = model.forward_video(batch, 2, 2, 6);
    let delta = (forward
        .frame_patch_tokens
        .clone()
        .slice_dim(1, 0..forward.context_len)
        - forward.context_posterior_patch_tokens.clone())
    .abs()
    .sum()
    .to_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .expect("delta")[0];
    assert!(
        delta > 1e-4,
        "refine passes did not move the pyramid context state"
    );
}

#[test]
fn video_lejepa_trm_backbone_predict_advances_temporal_age_and_observe_resets_it() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 0);
    let context_frames = batch.clip_frames.slice_dim(1, 0..batch.context_len);
    let observation_patch_tokens =
        embed_clip_frames_raw_with_model(&model.frame_model, context_frames);
    let seed = observation_patch_tokens
        .slice_dim(1, 0..1)
        .reshape([2, 4, 16]);
    let state = model
        .frame_model
        .pyramid_state_from_patch_tokens(seed.clone());
    let refined = model
        .frame_model
        .forward_pyramid_state_rollout_mode_unbounded(
            state.clone(),
            2,
            2,
            StructuredStepMode::Refine,
        );
    assert_eq!(refined.temporal_position, 0);
    assert_eq!(refined.prediction_age, 0);

    let predicted = model
        .frame_model
        .forward_pyramid_state_rollout_mode_unbounded(
            state.clone(),
            2,
            2,
            StructuredStepMode::Predict,
        );
    assert_eq!(predicted.temporal_position, 1);
    assert_eq!(predicted.prediction_age, 1);

    let observed_state = model.frame_model.pyramid_state_with_patch_tokens(
        predicted,
        model.apply_step_mode(seed, StructuredStepMode::Observe),
    );
    let observed = model
        .frame_model
        .forward_pyramid_state_rollout_mode_unbounded(
            observed_state,
            2,
            2,
            StructuredStepMode::Observe,
        );
    assert_eq!(observed.temporal_position, 1);
    assert_eq!(observed.prediction_age, 0);
}

#[test]
fn video_lejepa_mode_embeddings_make_refine_and_predict_distinct() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let forward = model.forward_video(batch, 2, 2, 2);
    let seed = forward
        .frame_patch_tokens
        .clone()
        .slice_dim(1, 2..3)
        .reshape([2, 4, 16]);
    let refine =
        model.rollout_conditioned_patch_step(seed.clone(), None, 2, 2, StructuredStepMode::Refine);
    let predict =
        model.rollout_conditioned_patch_step(seed, None, 2, 2, StructuredStepMode::Predict);
    let delta = (refine.patch_tokens - predict.patch_tokens)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("delta")[0];
    assert!(
        delta > 1e-4,
        "mode embeddings did not separate refine and predict"
    );
}

#[test]
fn video_lejepa_without_mode_embeddings_refine_and_predict_match() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.temporal.mode_embeddings = false;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let forward = model.forward_video(batch, 2, 2, 2);
    let seed = forward
        .frame_patch_tokens
        .clone()
        .slice_dim(1, 2..3)
        .reshape([2, 4, 16]);
    let refine =
        model.rollout_conditioned_patch_step(seed.clone(), None, 2, 2, StructuredStepMode::Refine);
    let predict =
        model.rollout_conditioned_patch_step(seed, None, 2, 2, StructuredStepMode::Predict);
    assert_close(refine.patch_tokens, predict.patch_tokens, 1e-6, 1e-6);
    assert_close(refine.cls_token, predict.cls_token, 1e-6, 1e-6);
}
