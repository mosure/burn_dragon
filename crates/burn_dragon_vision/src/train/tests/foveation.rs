use super::*;

#[test]
fn fovea_warped_checkerboard_moire_is_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let batch = 1;
    let channels = 3;
    let width = 64;
    let height = 64;
    let patch_size = 32;
    let data = make_checkerboard_image(channels, height, width, 1);
    let images = Tensor::<Backend, 4>::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        &device,
    );

    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<Backend>(&device, 1, width, height, patch_size);
    saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
    let sampling_modes = [
        VisionFoveaSamplingMode::Sequential,
        VisionFoveaSamplingMode::Batched,
    ];

    for sampling_mode in sampling_modes {
        saccade.config.fovea_sampling_mode = sampling_mode;
        saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
        saccade.config.mip_levels = 2;

        let levels = saccade.build_mip_pyramid(images.clone(), patch_size);
        let base_grid = build_foveated_base_grid::<Backend>(patch_size, &device);
        let mean =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.5, 0.5], [1, 2]), &device);
        let sigma = Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.08], [1, 1]), &device);
        let radius = Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.3], [1, 1]), &device);
        let patch = saccade
            .foveated_patch_image_with_radius(&levels, &base_grid, mean, sigma, radius, None);
        let patch_vec = patch
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("patch vec");
        let (mean_diff, range) =
            checkerboard_center_metrics(&patch_vec, channels, patch_size, patch_size);
        let normalized = if range > 0.0 { mean_diff / range } else { 0.0 };
        assert!(
            normalized >= 0.2 && range >= 0.55,
            "foveated checkerboard lost high-frequency detail (normalized {normalized:.3} range {range:.3})"
        );
        let block_std = block_mean_std(&patch_vec, channels, patch_size, patch_size, 8);
        assert!(
            block_std < 0.12,
            "foveated checkerboard shows low-frequency artifacts (block std {block_std:.3})"
        );
    }
}

#[test]
fn fovea_warped_image_gradients_focus_center() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<Backend>(&device, 1, 32, 32, 8);
    let sampling_modes = [
        VisionFoveaSamplingMode::Sequential,
        VisionFoveaSamplingMode::Batched,
    ];
    let data = make_test_image(3, 32, 32);

    for sampling_mode in sampling_modes {
        saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
        saccade.config.fovea_sampling_mode = sampling_mode;
        saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
        saccade.config.mip_levels = 2;

        let images =
            Tensor::<Backend, 4>::from_data(TensorData::new(data.clone(), [1, 3, 32, 32]), &device)
                .require_grad();
        let patch_size = saccade.model.patch_size().max(1);
        let levels = saccade.build_mip_pyramid(images.clone(), patch_size);
        let base_grid = build_foveated_base_grid::<Backend>(patch_size, &device);
        let mean_raw =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.5, 0.5], [1, 2]), &device);
        let sigma_raw =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.08], [1, 1]), &device);
        let mean = activation::sigmoid(mean_raw);
        let sigma = activation::sigmoid(sigma_raw)
            .mul_scalar(0.25)
            .add_scalar(0.05);
        let patch = saccade.foveated_patch_image(&levels, &base_grid, mean, sigma, None);
        let grads = patch.mean().backward();
        let image_grad = images.grad(&grads).expect("image grad");
        assert_tensor_finite(image_grad.clone());
        assert_tensor_nonzero(image_grad.clone(), 1e-6);

        let grad_vec = image_grad
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("grad vec");
        let (inner_mean, outer_mean) = gradient_focus_stats(&grad_vec, 3, 32, 32, 0.5);
        if outer_mean > 0.0 {
            assert!(
                inner_mean > outer_mean * 1.2,
                "fovea gradients not focused (inner {inner_mean:.6}, outer {outer_mean:.6})"
            );
        } else {
            assert!(inner_mean > 0.0, "inner gradient mean is zero");
        }
    }
}

#[test]
fn fovea_warped_feature_gradients_focus_center() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<Backend>(&device, 1, 32, 32, 8);
    let sampling_modes = [
        VisionFoveaSamplingMode::Sequential,
        VisionFoveaSamplingMode::Batched,
    ];
    let feature_channels = 8;
    let mut feature_data = Vec::with_capacity(feature_channels * 32 * 32);
    for c in 0..feature_channels {
        for y in 0..32 {
            for x in 0..32 {
                feature_data.push((x as f32 + y as f32) * 0.01 + c as f32 * 0.02);
            }
        }
    }

    for sampling_mode in sampling_modes {
        saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
        saccade.config.fovea_sampling_mode = sampling_mode;
        saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
        saccade.config.mip_levels = 1;

        let features = Tensor::<Backend, 4>::from_data(
            TensorData::new(feature_data.clone(), [1, 8, 32, 32]),
            &device,
        )
        .require_grad();
        let base_grid = build_foveated_base_grid::<Backend>(8, &device);
        let mean_raw =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.5, 0.5], [1, 2]), &device);
        let sigma_raw =
            Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.06], [1, 1]), &device);
        let mean = activation::sigmoid(mean_raw);
        let sigma = activation::sigmoid(sigma_raw)
            .mul_scalar(0.2)
            .add_scalar(0.05);
        let level = SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([1, 1, 1], &device),
            grid: PatchGrid {
                height: 4,
                width: 4,
            },
            image: features.clone(),
        };
        let patch = saccade.foveated_patch_image(&[level], &base_grid, mean, sigma, None);
        let grads = patch.mean().backward();
        let feature_grad = features.grad(&grads).expect("feature grad");
        assert_tensor_finite(feature_grad.clone());
        assert_tensor_nonzero(feature_grad.clone(), 1e-6);

        let grad_vec = feature_grad
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("feature grad vec");
        let (inner_mean, outer_mean) =
            gradient_focus_stats(&grad_vec, feature_channels, 32, 32, 0.5);
        if outer_mean > 0.0 {
            assert!(
                inner_mean > outer_mean * 1.2,
                "feature gradients not focused (inner {inner_mean:.6}, outer {outer_mean:.6})"
            );
        } else {
            assert!(inner_mean > 0.0, "inner feature gradient mean is zero");
        }
    }
}

#[test]
fn saccade_patch_view_matches_cpu_foveation() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    run_foveation_equivalence::<Backend>(&device, "ndarray");
}

#[test]
fn saccade_fovea_snellen_checkerboard_resolves_cpu() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    run_foveation_snellen::<Backend>(&device, "ndarray");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn saccade_patch_view_matches_cpu_foveation_wgpu() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    run_foveation_equivalence::<Backend>(&device, "wgpu");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn saccade_fovea_snellen_checkerboard_resolves_wgpu() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    run_foveation_snellen::<Backend>(&device, "wgpu");
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn saccade_fovea_snellen_checkerboard_resolves_cuda() {
    type Backend = Cuda<f32>;
    if !cuda_memory_pool_stable() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = burn_cuda::CudaDevice::default();
        run_foveation_snellen::<Backend>(&device, "cuda");
    }));
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn saccade_scatter_matches_tensor_wgpu() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    run_scatter_equivalence::<Backend>(&device, "wgpu");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_foveation_custom_backward_produces_grads() {
    type Backend = Autodiff<Wgpu<f32>>;
    if !crate::train::foveation::wgsl::supports_backend::<Wgpu<f32>>() {
        return;
    }

    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<Backend>(&device, 1, 16, 16, 4);
    saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
    saccade.config.fovea_sampling_mode = VisionFoveaSamplingMode::Wgsl;
    saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
    saccade.config.mip_levels = 2;

    let data = make_test_image(3, 16, 16);
    let images = Tensor::<Backend, 4>::from_data(TensorData::new(data, [1, 3, 16, 16]), &device)
        .require_grad();
    let patch_size = saccade.model.patch_size().max(1);
    let levels = saccade.build_mip_pyramid(images.clone(), patch_size);
    let base_grid = build_foveated_base_grid::<Backend>(patch_size, &device);

    let mean_raw =
        Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.25, -0.1], [1, 2]), &device)
            .require_grad();
    let sigma_raw = Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.05], [1, 1]), &device)
        .require_grad();
    let mean = activation::sigmoid(mean_raw.clone());
    let sigma = activation::sigmoid(sigma_raw.clone())
        .mul_scalar(0.3)
        .add_scalar(0.05);

    let patch = saccade.foveated_patch_image(&levels, &base_grid, mean, sigma, None);
    let grads = patch.mean().backward();

    let mean_grad = mean_raw.grad(&grads).expect("mean_raw grad");
    let sigma_grad = sigma_raw.grad(&grads).expect("sigma_raw grad");
    let image_grad = images.grad(&grads).expect("image grad");
    assert_tensor_finite(mean_grad.clone());
    assert_tensor_finite(sigma_grad.clone());
    assert_tensor_finite(image_grad.clone());
    assert_tensor_nonzero(mean_grad, 1e-6);
    assert_tensor_nonzero(sigma_grad, 1e-6);
    assert_tensor_nonzero(image_grad, 1e-6);
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_foveation_custom_backward_produces_grads() {
    type Backend = Autodiff<Cuda<f32>>;
    if !crate::train::foveation::cubecl::supports_backend::<Cuda<f32>>() {
        return;
    }
    if !cuda_memory_pool_stable() || !cuda_random_kernel_stable() {
        return;
    }

    let device = burn_cuda::CudaDevice::default();

    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<Backend>(&device, 1, 16, 16, 4);
    saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
    saccade.config.fovea_sampling_mode = VisionFoveaSamplingMode::Cubecl;
    saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
    saccade.config.mip_levels = 2;

    let data = make_test_image(3, 16, 16);
    let images = Tensor::<Backend, 4>::from_data(TensorData::new(data, [1, 3, 16, 16]), &device)
        .require_grad();
    let patch_size = saccade.model.patch_size().max(1);
    let levels = saccade.build_mip_pyramid(images.clone(), patch_size);
    let base_grid = build_foveated_base_grid::<Backend>(patch_size, &device);

    let mean_raw =
        Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.25, -0.1], [1, 2]), &device)
            .require_grad();
    let sigma_raw = Tensor::<Backend, 2>::from_data(TensorData::new(vec![0.05], [1, 1]), &device)
        .require_grad();
    let mean = activation::sigmoid(mean_raw.clone());
    let sigma = activation::sigmoid(sigma_raw.clone())
        .mul_scalar(0.3)
        .add_scalar(0.05);

    let patch = saccade.foveated_patch_image(&levels, &base_grid, mean, sigma, None);
    let grads = patch.mean().backward();

    let mean_grad = mean_raw.grad(&grads).expect("mean_raw grad");
    let sigma_grad = sigma_raw.grad(&grads).expect("sigma_raw grad");
    let image_grad = images.grad(&grads).expect("image grad");
    assert_tensor_finite(mean_grad.clone());
    assert_tensor_finite(sigma_grad.clone());
    assert_tensor_finite(image_grad.clone());
    assert_tensor_nonzero(mean_grad, 1e-6);
    assert_tensor_nonzero(sigma_grad, 1e-6);
    assert_tensor_nonzero(image_grad, 1e-6);
}
