use super::*;

#[test]
fn video_lejepa_predict_backprop_frames_are_clamped_to_future_horizon() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.temporal.predict_backprop_frames = 3;
    let model = make_video_model::<Backend>(&vision, &video, &device);

    assert_eq!(model.effective_predict_backprop_frames(1), 1);
    assert_eq!(model.effective_predict_backprop_frames(2), 2);
    assert_eq!(model.effective_predict_backprop_frames(6), 3);
    assert!(model.should_detach_predict_rollout(0, 6));
    assert!(model.should_detach_predict_rollout(1, 6));
    assert!(model.should_detach_predict_rollout(2, 6));
    assert!(!model.should_detach_predict_rollout(3, 6));
    assert!(!model.should_detach_predict_rollout(4, 6));
    assert!(!model.should_detach_predict_rollout(5, 6));
}

#[test]
fn video_lejepa_long_rollout_kinematics_extend_beyond_supervised_horizon() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_future_frames = 6;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 6);
    let forward = model.forward_video(batch, 2, 2, 6);
    let [_, _clip_len, channels, height, width] = forward.clip_frames.shape().dims::<5>();
    let prefix_len = forward.context_len + forward.target_len;
    let long_tail_len = forward.future_len_all.saturating_sub(forward.target_len);
    assert!(long_tail_len > 0, "test requires extra future frames");

    let prefix = forward
        .clip_frames
        .clone()
        .slice_dim(0, 0..1)
        .slice_dim(1, 0..prefix_len);
    let corrupt_tail =
        Tensor::<Backend, 5>::zeros([1, long_tail_len, channels, height, width], &device)
            .add_scalar(1.0);
    let predicted_clip = Tensor::cat(vec![prefix, corrupt_tail], 1);
    assert_eq!(
        predicted_clip.shape().dims::<5>(),
        [
            1,
            forward.context_len + forward.future_len_all,
            channels,
            height,
            width,
        ]
    );

    let (supervised_com, supervised_velocity) = model.rollout_kinematics_metrics(
        &forward,
        Some(predicted_clip.clone()),
        forward.target_len,
    );
    let (long_com, long_velocity) =
        model.rollout_kinematics_metrics(&forward, Some(predicted_clip), forward.future_len_all);

    let supervised_com = supervised_com
        .expect("supervised COM")
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("supervised COM scalar")[0];
    let supervised_velocity = supervised_velocity
        .expect("supervised velocity")
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("supervised velocity scalar")[0];
    let long_com = long_com
        .expect("long COM")
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("long COM scalar")[0];
    let long_velocity = long_velocity
        .expect("long velocity")
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("long velocity scalar")[0];

    assert!(
        long_com > supervised_com + 1e-4,
        "long COM error should exceed supervised COM when only long-tail frames are corrupted (supervised={supervised_com}, long={long_com})"
    );
    assert!(
        long_velocity > supervised_velocity + 1e-4,
        "long velocity error should exceed supervised velocity when only long-tail frames are corrupted (supervised={supervised_velocity}, long={long_velocity})"
    );
}

#[test]
fn video_lejepa_train_horizon_can_exceed_default_target_frames() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.target_frames = 2;
    video.train_target_frames_max = 5;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 5, 0);
    let forward = model.forward_video(batch, 2, 2, 5);
    assert_eq!(forward.predicted_proj.shape().dims::<3>(), [2, 5, 12]);
    assert_eq!(forward.target_proj.shape().dims::<3>(), [2, 5, 12]);
    assert_eq!(forward.future_cls_embed.shape().dims::<3>(), [2, 5, 16]);
}

#[test]
fn video_lejepa_train_pyramid_reports_rollout_metrics() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 0);
    let losses = model.forward_losses_train_pyramid(batch, 2, 2, true, true);

    let inv_h2 = losses.rollout_inv_to_horizon[1]
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("train rollout inv")[0];
    let norm_h2 = losses.rollout_state_norm_ratio_to_horizon[1]
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("train rollout norm")[0];
    let motion_h2 = losses.rollout_state_motion_to_horizon[1]
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("train rollout motion")[0];

    assert!(
        inv_h2.is_finite() && inv_h2 > 0.0,
        "expected nonzero finite train rollout inv metric, got {inv_h2}"
    );
    assert!(
        norm_h2.is_finite() && norm_h2 > 0.0,
        "expected positive finite train rollout norm metric, got {norm_h2}"
    );
    assert!(
        motion_h2.is_finite(),
        "expected finite train rollout motion metric, got {motion_h2}"
    );
}

#[test]
fn video_lejepa_future_patch_latent_evolves_over_time() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.artifact_future_frames = 6;
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 2);
    let forward = model.forward_video(batch, 2, 2, 6);

    let step0 = forward.future_patch_tokens.clone().slice_dim(1, 0..1);
    let step1 = forward.future_patch_tokens.clone().slice_dim(1, 1..2);
    let delta = (step1 - step0)
        .abs()
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("delta")[0];

    assert!(
        delta > 1e-4,
        "future patch rollout stayed effectively static"
    );
}

#[test]
fn video_lejepa_prediction_improves_on_toy_batch() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);

    let initial = model.forward_losses(batch.clone(), 2, 2, false, false, true);
    let initial_pred = initial
        .inv
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("initial")[0];

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 2e-2;
    for _ in 0..40 {
        let losses = model.forward_losses(batch.clone(), 2, 2, false, false, true);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);
    }

    let final_losses = model.forward_losses(batch, 2, 2, false, false, true);
    let final_pred = final_losses
        .inv
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("final")[0];
    assert!(final_pred.is_finite());
    assert!(final_pred < initial_pred);
}

#[test]
fn debug_reconstruction_head_does_not_move_video_jepa_core() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.loss.prediction_weight = 0.0;
    video.loss.cosine_weight = 0.0;
    video.loss.sigreg.enabled = false;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 1.0;
    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);

    let before = model.forward_video(batch.clone(), 2, 2, 2);
    let initial_losses = model.forward_losses(batch.clone(), 2, 2, false, false, true);
    let initial_recon = initial_losses
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("initial recon")[0];

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 2e-2;
    for _ in 0..20 {
        let losses = model.forward_losses(batch.clone(), 2, 2, false, false, true);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);
    }

    let after = model.forward_video(batch.clone(), 2, 2, 2);
    let final_losses = model.forward_losses(batch, 2, 2, false, false, true);
    let final_recon = final_losses
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("final recon")[0];

    assert!(final_recon < initial_recon);
    assert_close(before.predicted_proj, after.predicted_proj, 1e-6, 1e-6);
    assert_close(before.target_proj, after.target_proj, 1e-6, 1e-6);
    assert_close(before.cls_embed, after.cls_embed, 1e-6, 1e-6);
    assert_close(
        before.future_hidden_all,
        after.future_hidden_all,
        1e-6,
        1e-6,
    );
    assert_close(before.future_cls_embed, after.future_cls_embed, 1e-6, 1e-6);
    assert_close(
        before.frame_patch_tokens,
        after.frame_patch_tokens,
        1e-6,
        1e-6,
    );
    assert_close(
        before.future_patch_tokens,
        after.future_patch_tokens,
        1e-6,
        1e-6,
    );
}

#[test]
fn video_lejepa_targets_come_from_teacher_when_ema_enabled() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let forward = model.forward_video(batch.clone(), 2, 2, 2);
    let teacher = model
        .teacher_frame_model
        .as_ref()
        .expect("momentum teacher should be initialized");
    let expected =
        projected_future_frames(teacher, batch.clip_frames, 3, 2, 2, model.projection_dim);
    assert_close(forward.target_proj, expected, 1e-6, 1e-6);
}

#[test]
fn video_lejepa_teacher_ema_lags_student_after_one_step() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_video_configs(false, false, 1);
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let batch = toy_video_batch::<Backend>(&device);
    let initial_teacher = model
        .teacher_frame_model
        .as_ref()
        .expect("teacher should exist");
    let initial_teacher_proj = projected_future_frames(
        initial_teacher,
        batch.clone().clip_frames,
        3,
        2,
        2,
        model.projection_dim,
    );
    let initial_student_proj = projected_future_frames(
        &model.frame_model,
        batch.clone().clip_frames,
        3,
        2,
        2,
        model.projection_dim,
    );
    assert_close(initial_teacher_proj, initial_student_proj, 1e-6, 1e-6);

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 2e-2;
    let losses = model.forward_losses(batch.clone(), 2, 2, false, false, true);
    let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
    model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);

    let teacher = model
        .teacher_frame_model
        .as_ref()
        .expect("teacher should persist");
    let teacher_proj = projected_future_frames(
        teacher,
        batch.clone().clip_frames,
        3,
        2,
        2,
        model.projection_dim,
    );
    let student_proj = projected_future_frames(
        &model.frame_model,
        batch.clip_frames,
        3,
        2,
        2,
        model.projection_dim,
    );
    let mean_abs = (student_proj - teacher_proj)
        .abs()
        .mean()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("teacher/student drift")[0];
    assert!(mean_abs.is_finite());
    assert!(mean_abs > 0.0);
}

#[test]
fn video_lejepa_load_record_rebuilds_teacher_from_student() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs(false, false, 1);
    let reference = make_video_model::<Backend>(&vision, &video, &device);
    let restored =
        make_video_model::<Backend>(&vision, &video, &device).load_record(reference.into_record());
    assert!(restored.teacher_frame_model.is_some());

    let batch = toy_video_batch::<Backend>(&device);
    let teacher = restored
        .teacher_frame_model
        .as_ref()
        .expect("teacher should be restored");
    let teacher_proj = projected_future_frames(
        teacher,
        batch.clone().clip_frames,
        3,
        2,
        2,
        restored.projection_dim,
    );
    let student_proj = projected_future_frames(
        &restored.frame_model,
        batch.clip_frames,
        3,
        2,
        2,
        restored.projection_dim,
    );
    assert_close(teacher_proj, student_proj, 1e-6, 1e-6);
}
