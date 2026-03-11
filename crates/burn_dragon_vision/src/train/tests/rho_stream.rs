use super::*;
#[cfg(not(target_arch = "wasm32"))]
use burn::tensor::Distribution;

#[cfg(not(target_arch = "wasm32"))]
type WgpuBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
#[cfg(not(target_arch = "wasm32"))]
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

#[cfg(not(target_arch = "wasm32"))]
fn build_public_rho_model<B: BackendTrait>(
    device: &B::Device,
    wgpu_forward_kernel: bool,
    wgpu_rollout_fused: bool,
) -> VisionDragon<B> {
    let config = make_rho_stream_contract_config(wgpu_forward_kernel, wgpu_rollout_fused);
    VisionDragon::<B>::new(config, device)
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_rollout_output_finite<B: BackendTrait>(output: &crate::VisionDragonOutput<B>) {
    assert_tensor_finite(output.patch_tokens.clone());
    assert_tensor_finite(output.cls_token.clone());
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_forward_kernel_executes_public_rollout_path() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let model = build_public_rho_model::<WgpuBackend>(&device, true, false);
    let tokens = rho_stream_contract_tokens::<WgpuBackend>(&device);
    let output = model.forward_tokens_embed_steps_rollout(tokens, 3, 3);

    assert_eq!(output.patch_tokens.shape().dims::<3>(), [2, 4, 16]);
    assert_eq!(output.cls_token.shape().dims::<3>(), [2, 1, 16]);
    assert_rollout_output_finite(&output);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_forward_kernel_matches_reference_public_rollout_path() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 2026);

    let reference = build_public_rho_model::<WgpuBackend>(&device, false, false);
    let fused = build_public_rho_model::<WgpuBackend>(&device, true, false)
        .load_record(reference.clone().into_record());
    let tokens =
        Tensor::<WgpuBackend, 3>::random([2, 4, 16], Distribution::Normal(0.0, 1.0), &device);

    let reference_output = reference.forward_tokens_embed_steps_rollout(tokens.clone(), 3, 3);
    let fused_output = fused.forward_tokens_embed_steps_rollout(tokens, 3, 3);

    assert_tensor_close(
        reference_output.patch_tokens,
        fused_output.patch_tokens,
        8e-2,
        8e-2,
    );
    assert_tensor_close(
        reference_output.cls_token,
        fused_output.cls_token,
        8e-2,
        8e-2,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_rollout_fused_executes_public_rollout_path() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let model = build_public_rho_model::<WgpuBackend>(&device, true, true);
    let tokens = rho_stream_contract_tokens::<WgpuBackend>(&device);
    let output = model.forward_tokens_embed_steps_rollout(tokens, 3, 3);

    assert_eq!(output.patch_tokens.shape().dims::<3>(), [2, 4, 16]);
    assert_eq!(output.cls_token.shape().dims::<3>(), [2, 1, 16]);
    assert_rollout_output_finite(&output);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_rollout_fused_matches_reference_public_rollout_path() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 4242);

    let reference = build_public_rho_model::<WgpuBackend>(&device, false, false);
    let fused = build_public_rho_model::<WgpuBackend>(&device, true, true)
        .load_record(reference.clone().into_record());
    let tokens =
        Tensor::<WgpuBackend, 3>::random([2, 4, 16], Distribution::Normal(0.0, 1.0), &device);

    let reference_output = reference.forward_tokens_embed_steps_rollout(tokens.clone(), 3, 3);
    let fused_output = fused.forward_tokens_embed_steps_rollout(tokens, 3, 3);

    assert_tensor_close(
        reference_output.patch_tokens,
        fused_output.patch_tokens,
        8e-2,
        8e-2,
    );
    assert_tensor_close(
        reference_output.cls_token,
        fused_output.cls_token,
        8e-2,
        8e-2,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_autodiff_executes_public_rollout_train_step() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let model = build_public_rho_model::<WgpuAutodiffBackend>(&device, true, true);
    let tokens = rho_stream_contract_tokens::<WgpuAutodiffBackend>(&device);
    let loss = rho_stream_contract_loss(&model, tokens);
    let loss_value = loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    let grads = GradientsParams::from_grads(loss.backward(), &model);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let lr: LearningRate = 1e-3;
    let updated = optimizer.step(lr, model, grads);
    let output = updated.forward_tokens_embed_steps_rollout(
        rho_stream_contract_tokens::<WgpuAutodiffBackend>(&device),
        3,
        3,
    );

    assert!(loss_value.is_finite());
    assert_rollout_output_finite(&output);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_rho_stream_autodiff_matches_reference_after_one_step() {
    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 5151);

    let reference = build_public_rho_model::<WgpuAutodiffBackend>(&device, false, false);
    let fused = build_public_rho_model::<WgpuAutodiffBackend>(&device, true, true)
        .load_record(reference.clone().into_record());
    let tokens = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 4, 16],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let mut reference_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let mut fused_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let lr: LearningRate = 1e-3;

    let reference_loss = rho_stream_contract_loss(&reference, tokens.clone());
    let reference_loss_value = reference_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);
    let reference = reference_optimizer.step(lr, reference, reference_grads);

    let fused_loss = rho_stream_contract_loss(&fused, tokens.clone());
    let fused_loss_value = fused_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused loss")[0];
    let fused_grads = GradientsParams::from_grads(fused_loss.backward(), &fused);
    let fused = fused_optimizer.step(lr, fused, fused_grads);

    assert!((reference_loss_value - fused_loss_value).abs() <= 8e-2);

    let reference_output = reference.forward_tokens_embed_steps_rollout(tokens.clone(), 3, 3);
    let fused_output = fused.forward_tokens_embed_steps_rollout(tokens, 3, 3);

    assert_tensor_close(
        reference_output.patch_tokens.inner(),
        fused_output.patch_tokens.inner(),
        2e-1,
        2e-1,
    );
    assert_tensor_close(
        reference_output.cls_token.inner(),
        fused_output.cls_token.inner(),
        2e-1,
        2e-1,
    );
}
