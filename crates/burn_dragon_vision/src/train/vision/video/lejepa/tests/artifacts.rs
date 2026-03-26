use super::*;

#[test]
fn video_lejepa_artifacts_include_full_clip_encoder_pca() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 6;
    video.artifact_upscale = 4;
    video.artifact_output = VisionArtifactOutputMode::Avi;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 1);
    let losses = model.forward_losses(batch, 2, 2, true, true, true);
    let artifacts = losses.artifacts.expect("artifacts");

    assert!(artifacts.views.is_none());
    let frames = artifacts.frames.expect("clip frames");
    assert_eq!(frames.shape().dims::<5>(), [1, 6, 3, 8, 8]);
    let debug_recon = artifacts.debug_recon_frames.expect("debug recon clip");
    assert_eq!(debug_recon.shape().dims::<5>(), [1, 6, 3, 8, 8]);

    let pca_rgb_steps = artifacts.pca_rgb_steps.expect("pca rgb steps");
    assert_eq!(pca_rgb_steps.shape().dims::<5>(), [1, 6, 3, 2, 2]);
    let posterior_pca_rgb_steps = artifacts
        .posterior_pca_rgb_steps
        .expect("posterior pca rgb steps");
    assert_eq!(posterior_pca_rgb_steps.shape().dims::<5>(), [1, 6, 3, 2, 2]);
    let debug_pca_rgb_steps = artifacts.debug_pca_rgb_steps.expect("debug pca rgb steps");
    assert_eq!(debug_pca_rgb_steps.shape().dims::<5>(), [1, 6, 3, 2, 2]);
    assert!(artifacts.patch_norms_steps.is_none());
    assert!(artifacts.posterior_patch_norms_steps.is_none());
    assert!(artifacts.debug_patch_norms_steps.is_none());
    assert_eq!(artifacts.artifact_scale, 4);
    assert_eq!(artifacts.prediction_start, Some(3));

    let legend = artifacts.legend.expect("legend");
    assert_eq!(
        legend,
        vec![
            "reference_frame".to_string(),
            "posterior_context_state_pca_rgb".to_string(),
            "state_pca_rgb".to_string(),
            "decoded_spatiotemporal_latent".to_string(),
            "reencoded_decoded_latent_pca_rgb".to_string(),
        ]
    );
}

#[test]
fn video_lejepa_artifacts_reference_maps_match_reference_encoder() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 6;
    video.artifact_output = VisionArtifactOutputMode::Avi;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 1);

    let forward = model.forward_video(batch.clone(), 2, 2, 6);
    let (_, _, expected_patch_steps, expected_pca_steps) =
        collect_video_feature_maps(forward.frame_patch_tokens, 1, false);
    let losses = model.forward_losses(batch, 2, 2, true, true, true);
    let artifacts = losses.artifacts.expect("artifacts");

    let (_, _, expected_posterior_patch_steps, expected_posterior_pca_steps) =
        collect_video_feature_maps(forward.context_posterior_patch_tokens.clone(), 1, false);
    let expected_posterior_pca_steps = expected_posterior_pca_steps.map(|maps| {
        Tensor::cat(
            vec![maps, Tensor::<Backend, 5>::zeros([1, 3, 3, 2, 2], &device)],
            1,
        )
    });
    assert_close(
        artifacts
            .posterior_pca_rgb_steps
            .expect("artifact posterior pca steps"),
        expected_posterior_pca_steps.expect("expected posterior pca steps"),
        1e-6,
        1e-6,
    );
    assert_close(
        artifacts.pca_rgb_steps.expect("artifact pca steps"),
        expected_pca_steps.expect("expected pca steps"),
        1e-6,
        1e-6,
    );
    assert!(artifacts.patch_norms_steps.is_none());
    assert!(artifacts.posterior_patch_norms_steps.is_none());
    assert!(expected_patch_steps.is_none());
    assert!(expected_posterior_patch_steps.is_none());
}

#[test]
fn video_lejepa_context_filter_uses_temporal_prior() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_same_observation_different_history::<Backend>(&device);
    let forward = model.forward_video(batch, 2, 2, 2);

    let sample0 = forward
        .frame_patch_tokens
        .clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 1..2);
    let sample1 = forward
        .frame_patch_tokens
        .clone()
        .slice_dim(0, 1..2)
        .slice_dim(1, 1..2);
    let delta = (sample0 - sample1)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("delta")[0];

    assert!(
        delta > 1e-4,
        "posterior context state ignored prior history despite matched observation"
    );
}

#[test]
fn video_lejepa_decoded_clip_uses_live_context_then_predictive_future() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 6;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 1);
    let forward = model.forward_video(batch, 2, 2, 6);
    let [_, _, channels, height, width] = forward.clip_frames.shape().dims::<5>();
    let debug_recon = model.debug_reconstruction_outputs(&forward, 2, true, true, None);
    let clip = debug_recon.clip.expect("decoded clip");

    let expected_context = model
        .reconstruct_frames_from_patch_tokens(
            forward
                .frame_patch_tokens
                .clone()
                .slice_dim(0, 0..1)
                .slice_dim(1, 0..forward.context_len)
                .detach(),
            height,
            width,
            channels,
        )
        .reshape([1, forward.context_len, channels, height, width]);
    let expected_future = model
        .reconstruct_frames_from_patch_tokens(
            forward
                .future_patch_tokens
                .clone()
                .slice_dim(0, 0..1)
                .detach(),
            height,
            width,
            channels,
        )
        .reshape([1, forward.future_len_all, channels, height, width]);

    assert_close(
        clip.clone().slice_dim(1, 0..forward.context_len),
        expected_context,
        1e-6,
        1e-6,
    );
    assert_close(
        clip.slice_dim(
            1,
            forward.context_len..forward.context_len + forward.future_len_all,
        ),
        expected_future,
        1e-6,
        1e-6,
    );
}

#[test]
fn video_lejepa_long_horizon_predictions_extend_beyond_training_target() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 7;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 2);
    let forward = model.forward_video(batch.clone(), 2, 2, 7);
    assert_eq!(forward.predicted_proj.shape().dims::<3>(), [2, 4, 12]);
    assert_eq!(forward.future_hidden_all.shape().dims::<3>(), [2, 4, 16]);
    assert_eq!(forward.future_cls_embed.shape().dims::<3>(), [2, 4, 16]);
    assert_eq!(
        forward.future_patch_tokens.shape().dims::<4>(),
        [2, 4, 4, 16]
    );

    let losses = model.forward_losses(batch, 2, 2, true, true, true);
    let artifacts = losses.artifacts.expect("artifacts");
    let frames = artifacts.frames.expect("reference frames");
    let recon = artifacts.debug_recon_frames.expect("debug recon frames");
    assert_eq!(frames.shape().dims::<5>(), [1, 7, 3, 8, 8]);
    assert_eq!(recon.shape().dims::<5>(), [1, 7, 3, 8, 8]);
    assert_eq!(
        losses.rollout_inv_to_horizon.len(),
        VISION_ROLLOUT_HORIZON_COUNT
    );
    assert_eq!(
        losses.rollout_state_norm_ratio_to_horizon.len(),
        VISION_ROLLOUT_HORIZON_COUNT
    );
    assert_eq!(
        losses.rollout_state_motion_to_horizon.len(),
        VISION_ROLLOUT_HORIZON_COUNT
    );
    for value in losses
        .rollout_inv_to_horizon
        .iter()
        .chain(losses.rollout_state_norm_ratio_to_horizon.iter())
        .chain(losses.rollout_state_motion_to_horizon.iter())
    {
        let scalar = value
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rollout metric")[0];
        assert!(scalar.is_finite());
    }
    let psnr_full = losses
        .recon_psnr_full
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("psnr")[0];
    assert!(psnr_full.is_finite());
}

#[test]
fn video_lejepa_valid_step_long_rollout_metrics_do_not_require_artifact_capture() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 12;
    let model = make_video_model::<Backend>(&vision, &video, &device);

    let plain_batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 12);
    let plain_output = burn_train::InferenceStep::step(&model, plain_batch);
    let plain_long_h1 = <VisionOutput<Backend> as Adaptor<
        LongRolloutInvToHorizonInput<Backend, 0>,
    >>::adapt(&plain_output)
    .value();
    let plain_long_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutInvToHorizonInput<Backend, 5>,
    >>::adapt(&plain_output)
    .value();
    let plain_motion_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutStateMotionToHorizonInput<Backend, 5>,
    >>::adapt(&plain_output)
    .value();
    let plain_com_h24 =
        <VisionOutput<Backend> as Adaptor<RolloutComErrorToH24Input<Backend>>>::adapt(
            &plain_output,
        )
        .value();
    let plain_velocity_h24 = <VisionOutput<Backend> as Adaptor<
        RolloutVelocityErrorToH24Input<Backend>,
    >>::adapt(&plain_output)
    .value();
    let plain_long_com_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutComErrorToH24Input<Backend>,
    >>::adapt(&plain_output)
    .value();
    let plain_long_velocity_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutVelocityErrorToH24Input<Backend>,
    >>::adapt(&plain_output)
    .value();
    for value in [
        plain_long_h1,
        plain_long_h24,
        plain_motion_h24,
        plain_com_h24,
        plain_velocity_h24,
        plain_long_com_h24,
        plain_long_velocity_h24,
    ] {
        let scalar = value
            .expect("plain valid batch should report long-rollout metrics")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("plain rollout metric")[0];
        assert!(scalar.is_finite());
    }

    let artifact_batch =
        toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 12).with_capture_artifacts(true);
    let artifact_output = burn_train::InferenceStep::step(&model, artifact_batch);
    let artifact_long_h1 = <VisionOutput<Backend> as Adaptor<
        LongRolloutInvToHorizonInput<Backend, 0>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout h1 metric");
    let artifact_long_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutInvToHorizonInput<Backend, 5>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout h24 metric");
    let artifact_norm_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutStateNormRatioToHorizonInput<Backend, 5>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout norm metric");
    let artifact_motion_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutStateMotionToHorizonInput<Backend, 5>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout motion metric");
    let artifact_com_h24 =
        <VisionOutput<Backend> as Adaptor<RolloutComErrorToH24Input<Backend>>>::adapt(
            &artifact_output,
        )
        .value()
        .expect("artifact batch should report rollout COM metric");
    let artifact_velocity_h24 = <VisionOutput<Backend> as Adaptor<
        RolloutVelocityErrorToH24Input<Backend>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report rollout velocity metric");
    let artifact_long_com_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutComErrorToH24Input<Backend>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout COM metric");
    let artifact_long_velocity_h24 = <VisionOutput<Backend> as Adaptor<
        LongRolloutVelocityErrorToH24Input<Backend>,
    >>::adapt(&artifact_output)
    .value()
    .expect("artifact batch should report long-rollout velocity metric");

    for value in [
        artifact_long_h1,
        artifact_long_h24,
        artifact_norm_h24,
        artifact_motion_h24,
        artifact_com_h24,
        artifact_velocity_h24,
        artifact_long_com_h24,
        artifact_long_velocity_h24,
    ] {
        let scalar = value
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("artifact rollout metric")[0];
        assert!(scalar.is_finite());
    }
}
