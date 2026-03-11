use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_temporal_forward_matches_reference_with_alibi() {
    type Backend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, baseline_video) = make_video_configs(false, false, 4);
    let (_, fused_video) = make_video_configs(true, true, 4);
    <Backend as BackendTrait>::seed(&device, 4242);
    let reference = make_video_model::<Backend>(&vision, &baseline_video, &device);
    let fused = make_video_model::<Backend>(&vision, &fused_video, &device)
        .load_record(reference.clone().into_record());
    let batch = toy_video_batch::<Backend>(&device);

    let reference_forward = reference.forward_video(batch.clone(), 2, 2, 2);
    let fused_forward = fused.forward_video(batch, 2, 2, 2);

    assert_close(
        reference_forward.predicted_proj,
        fused_forward.predicted_proj,
        8e-2,
        8e-2,
    );
    assert_close(
        reference_forward.target_proj,
        fused_forward.target_proj,
        1e-6,
        1e-6,
    );
    assert_close(
        reference_forward.context_summary,
        fused_forward.context_summary,
        1e-1,
        1e-1,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_temporal_autodiff_matches_reference_after_one_step() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, baseline_video) = make_video_configs(false, false, 4);
    let (_, fused_video) = make_video_configs(true, true, 4);
    <Backend as BackendTrait>::seed(&device, 5151);
    let reference = make_video_model::<Backend>(&vision, &baseline_video, &device);
    let fused = make_video_model::<Backend>(&vision, &fused_video, &device)
        .load_record(reference.clone().into_record());
    let batch = toy_video_batch::<Backend>(&device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 1e-3;

    let reference_losses = reference.forward_losses(batch.clone(), 2, 2, false, false);
    let reference_total = reference_losses.total.clone()
        + reference_losses
            .probe_loss
            .clone()
            .mul_scalar(reference.config.loss.probe_weight.max(0.0));
    let reference_loss_value = reference_total
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let reference_grads = GradientsParams::from_grads(reference_total.backward(), &reference);
    let reference = reference.optimize::<Backend, _>(&mut optimizer, lr, reference_grads);

    let fused_losses = fused.forward_losses(batch.clone(), 2, 2, false, false);
    let fused_total = fused_losses.total.clone()
        + fused_losses
            .probe_loss
            .clone()
            .mul_scalar(fused.config.loss.probe_weight.max(0.0));
    let fused_loss_value = fused_total
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused loss")[0];
    let fused_grads = GradientsParams::from_grads(fused_total.backward(), &fused);
    let fused = fused.optimize::<Backend, _>(&mut optimizer, lr, fused_grads);

    assert!((reference_loss_value - fused_loss_value).abs() <= 6e-2);

    let reference_forward = reference.forward_video(batch.clone(), 2, 2, 2);
    let fused_forward = fused.forward_video(batch, 2, 2, 2);
    assert_close(
        reference_forward.predicted_proj.inner(),
        fused_forward.predicted_proj.inner(),
        2e-1,
        2e-1,
    );
    assert_close(
        reference_forward.context_summary.inner(),
        fused_forward.context_summary.inner(),
        2e-1,
        2e-1,
    );
}
