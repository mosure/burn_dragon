use crate::train::prelude::*;
use crate::train::SaccadeFoveationSampler;
use crate::ContextStrategyConfig;
use crate::{
    FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode, VisionLossConfig,
    VisionPyramidMode, VisionReconLossConfig, VisionSaccadeCacheConfig,
    VisionSaccadePolicyConfig,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::{VisionTrainingModeConfig, load_vision_training_config};
use burn_autodiff::Autodiff;
#[cfg(not(target_arch = "wasm32"))]
use burn_cubecl::CubeBackend;
use burn_ndarray::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use burn_wgpu::Wgpu;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;
use std::sync::Once;

fn make_training(max_iters: usize, epochs: Option<usize>) -> TrainingHyperparameters {
    TrainingHyperparameters {
        block_size: 16,
        batch_size: 2,
        epochs,
        max_iters,
        log_frequency: 10,
        fast_train: false,
        context_strategy: ContextStrategyConfig::Infinite,
        gdpo: None,
    }
}

fn make_saccade_model<B: BackendTrait>(
    device: &B::Device,
    num_eyes: usize,
) -> (VisionSaccadeModel<B>, VisionDragonHatchlingConfig) {
    make_saccade_model_with_dims(device, num_eyes, 8, 8, 4)
}

fn make_saccade_model_with_dims<B: BackendTrait>(
    device: &B::Device,
    num_eyes: usize,
    image_width: usize,
    image_height: usize,
    patch_size: usize,
) -> (VisionSaccadeModel<B>, VisionDragonHatchlingConfig) {
    let patch_size = patch_size.max(1);
    let image_size = image_width.max(image_height).max(patch_size);
    let grid_w = (image_width / patch_size).max(1);
    let grid_h = (image_height / patch_size).max(1);
    let vision_config = VisionDragonHatchlingConfig {
        image_size,
        patch_size,
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: grid_h,
        pos_max_width: grid_w,
        attention_mode: VisionAttentionMode::RowL1,
        fused_kernels: FusedKernelConfig::default(),
    };
    let model = VisionDragonHatchling::<B>::new(vision_config.clone(), device);
    let saccade_config = VisionSaccadeConfig {
        num_eyes,
        mip_levels: 3,
        pyramid_mode: VisionPyramidMode::Laplacian,
        fovea_sampling_mode: VisionFoveaSamplingMode::Sequential,
        fovea_warp_mode: VisionFoveaWarpMode::Warped,
        fovea_subpatch_size: 0,
        fovea_scatter_mode: VisionFoveaScatterMode::Tensor,
        pyramid_feature_dim: None,
        inner_steps: 1,
        low_mem_pre_rollout: true,
        policy: VisionSaccadePolicyConfig::default(),
        cache: VisionSaccadeCacheConfig::default(),
        loss: VisionLossConfig {
            lejepa: VisionLejepaLossConfig {
                enabled: true,
                lambda: 0.02,
                sigreg_knots: 5,
                sigreg_t_max: 1.0,
                sigreg_proj_dim: 8,
            },
            recon: VisionReconLossConfig {
                weight: 0.0,
                mask_ratio: 0.0,
                hidden_dim: 16,
            },
        },
        artifact_output: VisionArtifactOutputMode::Images,
        artifact_fps: 4,
        artifact_every: 0,
        artifact_max_images: 0,
        artifact_max_views: 0,
        artifact_overwrite: true,
    };
    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 2,
        backprop_steps: 2,
    };
    let recon_patch_dim =
        vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    let saccade = VisionSaccadeModel::new(
        model,
        saccade_config,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        device,
    );
    (saccade, vision_config)
}

fn make_test_image(channels: usize, height: usize, width: usize) -> Vec<f32> {
    let mut data = Vec::with_capacity(channels * height * width);
    let denom_w = (width - 1).max(1) as f32;
    let denom_h = (height - 1).max(1) as f32;
    for c in 0..channels {
        for y in 0..height {
            let gy = y as f32 / denom_h;
            for x in 0..width {
                let gx = x as f32 / denom_w;
                let checker = ((x / 2 + y / 3 + c) % 2) as f32;
                let value = match c {
                    0 => gx,
                    1 => gy,
                    _ => 0.55 * gx + 0.35 * gy + 0.1 * checker,
                };
                data.push(value);
            }
        }
    }
    data
}

#[cfg(not(target_arch = "wasm32"))]
fn init_wgpu_test_runtime(device: &burn_wgpu::WgpuDevice) {
    use burn_wgpu::{RuntimeOptions, graphics};
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

#[cfg(not(target_arch = "wasm32"))]
fn vision_saccade_tiny_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir
            .join("..")
            .join("..")
            .join("config")
            .join("vision_saccade_tiny.toml"),
        manifest_dir
            .join("..")
            .join("config")
            .join("vision_saccade_tiny.toml"),
        manifest_dir.join("config").join("vision_saccade_tiny.toml"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    candidates[0].clone()
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
    allocs: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_memory_snapshot(device: &burn_wgpu::WgpuDevice) -> MemorySnapshot {
    let usage = <burn_wgpu::WgpuRuntime as cubecl::Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
        allocs: usage.number_allocs,
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_memory_snapshot(device: &burn_cuda::CudaDevice) -> MemorySnapshot {
    let usage = <cubecl::cuda::CudaRuntime as cubecl::Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
        allocs: usage.number_allocs,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_memory_growth_bounded(
    label: &str,
    snapshots: &[MemorySnapshot],
    max_reserved_growth: u64,
    max_in_use_growth: u64,
) {
    assert!(!snapshots.is_empty(), "{label}: no memory snapshots collected");
    let min_reserved = snapshots
        .iter()
        .map(|snapshot| snapshot.reserved)
        .min()
        .unwrap_or(0);
    let max_reserved = snapshots
        .iter()
        .map(|snapshot| snapshot.reserved)
        .max()
        .unwrap_or(0);
    let min_in_use = snapshots
        .iter()
        .map(|snapshot| snapshot.in_use)
        .min()
        .unwrap_or(0);
    let max_in_use = snapshots
        .iter()
        .map(|snapshot| snapshot.in_use)
        .max()
        .unwrap_or(0);
    let growth_reserved = max_reserved.saturating_sub(min_reserved);
    let growth_in_use = max_in_use.saturating_sub(min_in_use);
    if max_reserved_growth > 0 {
        assert!(
            growth_reserved <= max_reserved_growth,
            "{label}: reserved bytes grew by {growth_reserved} (> {max_reserved_growth}); snapshots={snapshots:?}"
        );
    }
    if max_in_use_growth > 0 {
        assert!(
            growth_in_use <= max_in_use_growth,
            "{label}: in-use bytes grew by {growth_in_use} (> {max_in_use_growth}); snapshots={snapshots:?}"
        );
    }
}

fn make_text_config(vocab_size: usize) -> BDHConfig {
    let mut config = BDHConfig::default();
    config.n_layer = 2;
    config.n_embd = 64;
    config.n_head = 4;
    config.mlp_internal_dim_multiplier = 2;
    config.n_expert = 1;
    config.dropout = 0.0;
    config.vocab_size = vocab_size.max(8);
    config
}

fn make_text_batch<B: BackendTrait>(
    device: &B::Device,
    batch: usize,
    block: usize,
    vocab: usize,
) -> SequenceBatch<B> {
    let vocab = vocab.max(2);
    let mut inputs = Vec::with_capacity(batch * block);
    let mut targets = Vec::with_capacity(batch * block);
    for idx in 0..(batch * block) {
        let token = (idx % vocab) as i64;
        inputs.push(token);
        targets.push(((idx + 1) % vocab) as i64);
    }
    let inputs = Tensor::<B, 2, Int>::from_data(
        TensorData::new(inputs, [batch, block]),
        device,
    );
    let targets = Tensor::<B, 2, Int>::from_data(
        TensorData::new(targets, [batch, block]),
        device,
    );
    SequenceBatch::new(inputs, targets)
}

fn run_foveation_equivalence<B: BackendTrait>(
    device: &B::Device,
    backend_label: &str,
) {
    let batch = 1;
    let channels = 3;
    let cases = [
        ([0.5, 0.5], 0.2, 0.2),
        ([0.2, 0.8], 0.1, 0.3),
        ([0.8, 0.2], 0.35, 0.4),
        ([0.6, 0.4], 0.4, 0.2),
        ([0.02, 0.98], 0.05, 0.15),
        ([0.98, 0.02], 0.25, 0.1),
        ([0.1, 0.1], 0.02, 0.05),
        ([0.9, 0.9], 0.45, 0.45),
    ];
    let size_configs = [
        (8, 8, 4, 2),
        (12, 8, 4, 2),
        (12, 12, 6, 3),
        (16, 16, 8, 4),
    ];
    let pyramid_configs = [
        (VisionPyramidMode::Stacked, 2),
        (VisionPyramidMode::Laplacian, 3),
    ];
    let mut sampling_modes = vec![
        VisionFoveaSamplingMode::Batched,
        VisionFoveaSamplingMode::Sequential,
        VisionFoveaSamplingMode::Subpatch,
    ];
    if crate::train::foveation::cubecl::supports_backend::<B>() {
        sampling_modes.push(VisionFoveaSamplingMode::Cubecl);
    }
    if crate::train::foveation::wgsl::supports_backend::<B>() {
        sampling_modes.push(VisionFoveaSamplingMode::Wgsl);
    }
    let warp_modes = [
        VisionFoveaWarpMode::Warped,
        VisionFoveaWarpMode::Patched,
    ];

    for (size_idx, (width, height, patch_size, subpatch_size)) in
        size_configs.iter().copied().enumerate()
    {
        let data = make_test_image(channels, height, width);
        let (min_val, max_val) = data.iter().fold(
            (f32::INFINITY, f32::NEG_INFINITY),
            |(min_v, max_v), &val| (min_v.min(val), max_v.max(val)),
        );
        assert!(max_val - min_val > 0.25, "test image too flat");
        let images = Tensor::<B, 4>::from_data(
            TensorData::new(data.clone(), [batch, channels, height, width]),
            device,
        );

        for (pyramid_mode, mip_levels) in pyramid_configs.iter().copied() {
            for sampling_mode in sampling_modes.iter().copied() {
                for warp_mode in warp_modes.iter().copied() {
                    let (mut saccade, vision_config) =
                        make_saccade_model_with_dims::<B>(
                            device,
                            1,
                            width,
                            height,
                            patch_size,
                        );
                    saccade.config.pyramid_mode = pyramid_mode;
                    saccade.config.fovea_sampling_mode = sampling_mode;
                    saccade.config.fovea_warp_mode = warp_mode;
                    saccade.config.fovea_subpatch_size = if matches!(
                        sampling_mode,
                        VisionFoveaSamplingMode::Subpatch
                    ) {
                        subpatch_size
                    } else {
                        0
                    };
                    saccade.config.mip_levels = mip_levels;

                    let mut sampler = SaccadeFoveationSampler::<B>::new(
                        vision_config,
                        saccade.config.clone(),
                        device,
                    );
                    sampler.update_image(images.clone());
                    let patch_size = sampler.patch_size();
                    let cpu_depth = sampler.mip_levels().max(1);
                    let cpu_mode = match saccade.config.pyramid_mode {
                        VisionPyramidMode::Stacked => foveation::PyramidMode::Stacked,
                        VisionPyramidMode::Laplacian => foveation::PyramidMode::Laplacian,
                    };
                    let cache = foveation::build_pyramid_cache(
                        foveation::image_from_nchw(&data, 0, channels, height, width)
                            .expect("cpu image"),
                        cpu_depth,
                        cpu_mode,
                    );

                    for (case_idx, (mean_vals, sigma_val, radius_val)) in
                        cases.iter().enumerate()
                    {
                        let mean_vals = *mean_vals;
                        let sigma_val = *sigma_val;
                        let radius_val = *radius_val;
                        let mean = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![mean_vals[0], mean_vals[1]], [batch, 2]),
                            device,
                        );
                        let sigma = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![sigma_val], [batch, 1]),
                            device,
                        );
                        let radius = Tensor::<B, 2>::from_data(
                            TensorData::new(vec![radius_val], [batch, 1]),
                            device,
                        );
                        let patch_view =
                            sampler.sample_patch_with_radius(mean, sigma, radius);
                        let patch_vec = patch_view
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("patch vec");

                        let expected_patch = foveation::render_foveated_patch_with_radius(
                            &cache,
                            mean_vals,
                            sigma_val,
                            radius_val,
                            patch_size,
                            match warp_mode {
                                VisionFoveaWarpMode::Warped => {
                                    foveation::FoveaWarpMode::Warped
                                }
                                VisionFoveaWarpMode::Patched => {
                                    foveation::FoveaWarpMode::Patched
                                }
                            },
                        );
                        let mut expected =
                            vec![0.0f32; channels * patch_size * patch_size];
                        let channel_stride = patch_size * patch_size;
                        for y in 0..patch_size {
                            for x in 0..patch_size {
                                let src = (y * patch_size + x) * 3;
                                let dst = y * patch_size + x;
                                expected[dst] = expected_patch[src];
                                expected[dst + channel_stride] =
                                    expected_patch[src + 1];
                                expected[dst + 2 * channel_stride] =
                                    expected_patch[src + 2];
                            }
                        }
                        let mut mse = 0.0f32;
                        let mut max_abs = 0.0f32;
                        for (lhs, rhs) in patch_vec.iter().zip(expected.iter()) {
                            let diff = lhs - rhs;
                            let abs = diff.abs();
                            if abs > max_abs {
                                max_abs = abs;
                            }
                            mse += diff * diff;
                        }
                        mse /= patch_vec.len().max(1) as f32;
                        let max_abs_threshold = 1e-3;
                        let mse_threshold = 1e-6;
                        assert!(
                            max_abs < max_abs_threshold && mse < mse_threshold,
                            "backend {backend_label} size {size_idx} {width}x{height} patch {patch_size} pyramid {pyramid_mode:?} sampling {sampling_mode:?} warp {warp_mode:?} case {case_idx} max_abs {max_abs} mse {mse}"
                        );
                    }
                }
            }
        }
    }
}

fn run_scatter_equivalence<B: BackendTrait>(
    device: &B::Device,
    backend_label: &str,
) {
    let batch = 2;
    let channels = 3;
    let width = 12;
    let height = 8;
    let patch_size = 4;
    let (mut saccade, _vision_config) =
        make_saccade_model_with_dims::<B>(device, 1, width, height, patch_size);
    let feature_dim = saccade.pyramid_feature_dim();

    let base = make_test_image(channels, height, width);
    let mut data = Vec::with_capacity(batch * channels * height * width);
    for b in 0..batch {
        let offset = b as f32 * 0.07;
        data.extend(base.iter().map(|value| value + offset));
    }
    let images = Tensor::<B, 4>::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        device,
    );
    let levels = saccade.build_mip_pyramid(images, patch_size);

    let mean = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.3, 0.7, 0.6, 0.4], [batch, 1, 2]),
        device,
    );
    let sigma = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.2, 0.35], [batch, 1, 1]),
        device,
    );
    let mut residual_values = Vec::with_capacity(batch * feature_dim);
    for b in 0..batch {
        for d in 0..feature_dim {
            residual_values.push(0.15 + b as f32 * 0.03 + d as f32 * 0.01);
        }
    }
    let residual_pool = Tensor::<B, 3>::from_data(
        TensorData::new(residual_values, [batch, 1, feature_dim]),
        device,
    );

    let mut scatter_modes = Vec::new();
    if crate::train::scatter::cubecl::supports_backend::<B>() {
        scatter_modes.push(VisionFoveaScatterMode::Cubecl);
    }
    if crate::train::scatter::wgsl::supports_backend::<B>() {
        scatter_modes.push(VisionFoveaScatterMode::Wgsl);
    }
    if scatter_modes.is_empty() {
        return;
    }

    let pyramid_modes = [VisionPyramidMode::Stacked, VisionPyramidMode::Laplacian];
    let warp_modes = [VisionFoveaWarpMode::Warped, VisionFoveaWarpMode::Patched];
    for pyramid_mode in pyramid_modes {
        saccade.config.pyramid_mode = pyramid_mode;
        for warp_mode in warp_modes {
            saccade.config.fovea_warp_mode = warp_mode;
            let weights = saccade.mip_gaussian_weights(&levels, mean.clone(), sigma.clone());

            saccade.config.fovea_scatter_mode = VisionFoveaScatterMode::Tensor;
            let baseline: Vec<Tensor<B, 3>> = weights
                .iter()
                .map(|weight| {
                    saccade.weighted_sum_tokens(
                        weight.clone().swap_dims(1, 2),
                        residual_pool.clone(),
                    )
                })
                .collect();

            for mode in scatter_modes.iter().copied() {
                saccade.config.fovea_scatter_mode = mode;
                for (level_idx, (weight, base)) in
                    weights.iter().zip(baseline.iter()).enumerate()
                {
                    let output = saccade.weighted_sum_tokens(
                        weight.clone().swap_dims(1, 2),
                        residual_pool.clone(),
                    );
                    let mse_tensor = (output - base.clone()).powf_scalar(2.0).mean();
                    let mse = mse_tensor
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("mse vec")[0];
                    assert!(
                        mse < 1e-6,
                        "backend {backend_label} pyramid {pyramid_mode:?} warp {warp_mode:?} scatter {mode:?} level {level_idx} mse {mse}"
                    );
                }
            }
        }
    }
}

fn make_level<B: BackendTrait>(
    device: &B::Device,
    tokens: usize,
    dim: usize,
    offset: f32,
) -> Tensor<B, 3> {
    let mut data = Vec::with_capacity(tokens * dim);
    for token in 0..tokens {
        for d in 0..dim {
            data.push(offset + token as f32 + d as f32 * 0.1);
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [tokens, dim]), device)
        .reshape([1, tokens, dim])
}

fn assert_mse_below<B: BackendTrait>(tensor: Tensor<B, 1>, threshold: f32) {
    let value = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("mse vec")[0];
    assert!(value < threshold, "mse {value} >= {threshold}");
}

fn assert_mse_above<B: BackendTrait>(tensor: Tensor<B, 1>, threshold: f32) {
    let value = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("mse vec")[0];
    assert!(value > threshold, "mse {value} <= {threshold}");
}

fn assert_tensor_finite<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>) {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor vec");
    assert!(!values.is_empty(), "tensor has no values");
    assert!(
        values.iter().all(|value| value.is_finite()),
        "tensor contains non-finite values"
    );
}

fn saccade_eye_step<B: BackendTrait>(
    saccade: &VisionSaccadeModel<B>,
    images: Tensor<B, 4>,
    eye_idx: usize,
) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
    let device = images.device();
    let [batch, _channels, height, _width] = images.shape().dims::<4>();
    let patch = saccade.model.patch_embed_raw(images.clone());
    let grid_h = patch.grid.height;
    let grid_w = patch.grid.width;
    assert!(grid_h > 0 && grid_w > 0);
    let patch_size = height / grid_h;
    assert!(patch_size > 0);

    let mip_levels = saccade.build_mip_pyramid(images, patch_size);
    assert!(!mip_levels.is_empty());
    let input_levels: Vec<Tensor<B, 3>> =
        mip_levels.iter().map(|level| level.tokens.clone()).collect();
    let grids: Vec<PatchGrid> = mip_levels.iter().map(|level| level.grid).collect();
    let input_residuals = match saccade.config.pyramid_mode {
        VisionPyramidMode::Stacked => input_levels.clone(),
        VisionPyramidMode::Laplacian => saccade.decompose_pyramid(&input_levels, &grids),
    };
    let laplacian_images = if matches!(saccade.config.pyramid_mode, VisionPyramidMode::Laplacian) {
        saccade.build_laplacian_images(&mip_levels)
    } else {
        None
    };
    let embed_dim = patch.tokens.shape().dims::<3>()[2];
    let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
    let base_traj = saccade
        .trajectory_token
        .val()
        .reshape([1, traj_len, embed_dim])
        .repeat_dim(0, batch);
    let state_levels: Vec<Tensor<B, 3>> = input_residuals
        .iter()
        .map(|level| Tensor::<B, 3>::zeros(level.shape().dims::<3>(), &device))
        .collect();
    let state_composed = match saccade.config.pyramid_mode {
        VisionPyramidMode::Stacked => state_levels.clone(),
        VisionPyramidMode::Laplacian => saccade.compose_pyramid(&state_levels, &grids),
    };
    let eye_embed = saccade
        .eye_token
        .val()
        .slice_dim(0, eye_idx..eye_idx + 1)
        .reshape([1, 1, embed_dim])
        .repeat_dim(0, batch)
        .repeat_dim(1, traj_len);
    let traj_with_eye = base_traj + eye_embed;
    let traj_summary = traj_with_eye
        .clone()
        .mean_dim(1)
        .reshape([batch, 1, embed_dim]);
    let params = saccade.saccade_head.forward(traj_summary);
    let (mean, sigma) = saccade.decode_saccade_params(params);
    let weights = saccade.mip_gaussian_weights(&mip_levels, mean.clone(), sigma.clone());
    let mean_step = mean.clone().mean_dim(1).reshape([batch, 2]);
    let sigma_step = sigma.clone().mean_dim(1).reshape([batch, 1]);
    let base_grid = build_foveated_base_grid::<B>(patch_size, &device);
    let patch_image = saccade.foveated_patch_image(
        &mip_levels,
        &base_grid,
        mean_step,
        sigma_step,
        laplacian_images.as_ref(),
    );
    let patch_tokens = saccade.model.patch_embed_raw(patch_image).tokens;
    let input_context = patch_tokens;
    let state_context = saccade.mip_weighted_sum(&state_composed, &weights);
    let state_context = saccade.project_pyramid_context(state_context);
    let input_tokens =
        saccade.build_input_tokens(input_context, state_context, mean.clone(), sigma.clone());
    let input_tokens = input_tokens.repeat_dim(1, traj_len);
    let tokens_in = traj_with_eye + input_tokens;
    let inner_steps = saccade.config.inner_steps.max(1);
    let out_tokens = saccade
        .model
        .forward_tokens_embed_steps(tokens_in, inner_steps)
        .patch_tokens;
    let residual = saccade.residual_proj.forward(out_tokens.clone());
    let residual_pool = residual
        .clone()
        .mean_dim(1)
        .reshape([batch, 1, saccade.pyramid_feature_dim()]);
    let next_traj = out_tokens;
    let mut updates = Vec::with_capacity(weights.len());
    for weights in &weights {
        updates.push(
            saccade.weighted_sum_tokens(weights.clone().swap_dims(1, 2), residual_pool.clone()),
        );
    }
    (next_traj, updates)
}

fn saccade_fovea_params<B: BackendTrait>(
    saccade: &VisionSaccadeModel<B>,
    traj_with_eye: Tensor<B, 3>,
    embed_dim: usize,
) -> Tensor<B, 3> {
    let [batch, _, _] = traj_with_eye.shape().dims::<3>();
    let traj_summary = traj_with_eye.mean_dim(1).reshape([batch, 1, embed_dim]);
    let params = saccade.saccade_head.forward(traj_summary);
    let (mean, sigma) = saccade.decode_saccade_params(params);
    Tensor::cat(vec![mean, sigma], 2)
}

fn saccade_weights_for_eye<B: BackendTrait>(
    saccade: &VisionSaccadeModel<B>,
    traj_with_eye: Tensor<B, 3>,
    mip_levels: &[SaccadeMipLevel<B>],
    embed_dim: usize,
) -> Vec<Tensor<B, 3>> {
    let [batch, _, _] = traj_with_eye.shape().dims::<3>();
    let traj_summary = traj_with_eye.mean_dim(1).reshape([batch, 1, embed_dim]);
    let params = saccade.saccade_head.forward(traj_summary);
    let (mean, sigma) = saccade.decode_saccade_params(params);
    saccade.mip_gaussian_weights(mip_levels, mean, sigma)
}

#[test]
fn epochs_schedule_overrides_max_iters() {
    let training = make_training(5, Some(3));
    let schedule = resolve_train_schedule(&training, 4).expect("schedule");

    assert_eq!(schedule.source, ScheduleSource::Epochs);
    assert_eq!(schedule.steps_per_epoch, 4);
    assert_eq!(schedule.total_epochs, 3);
    assert_eq!(schedule.total_steps, 12);
    assert_eq!(schedule.total_steps % schedule.steps_per_epoch, 0);
}

#[test]
fn max_iters_schedule_uses_step_limit() {
    let training = make_training(12, None);
    let schedule = resolve_train_schedule(&training, 5).expect("schedule");

    assert_eq!(schedule.source, ScheduleSource::MaxIters);
    assert_eq!(schedule.steps_per_epoch, 5);
    assert_eq!(schedule.total_steps, 12);
    assert_eq!(schedule.total_epochs, 3);
}

#[test]
fn saccade_recon_loss_smoke() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 2, 1, true, false);
    let value = losses
        .total
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn saccade_multi_eye_loss_smoke() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 2, 1, true, false);
    let value = losses
        .total
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn saccade_multi_eye_trajectory_states_diverge() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let mut eye_values = vec![0.0; embed_dim];
    eye_values.extend(std::iter::repeat(1.0).take(embed_dim));
    let eye_token =
        Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
    saccade.eye_token = Param::from_tensor(eye_token);

    let mut image_values = Vec::with_capacity(1 * 3 * 8 * 8);
    for idx in 0..(1 * 3 * 8 * 8) {
        image_values.push(idx as f32 / 255.0);
    }
    let images =
        Tensor::<Backend, 4>::from_data(TensorData::new(image_values, [1, 3, 8, 8]), &device);

    let (traj0, _) = saccade_eye_step(&saccade, images.clone(), 0);
    let (traj1, _) = saccade_eye_step(&saccade, images, 1);
    let mse = (traj0 - traj1).powf_scalar(2.0).mean();
    assert_mse_above(mse, 0.0);
}

#[test]
fn saccade_patch_view_matches_cpu_foveation() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    run_foveation_equivalence::<Backend>(&device, "ndarray");
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
fn saccade_scatter_matches_tensor_wgpu() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);
    run_scatter_equivalence::<Backend>(&device, "wgpu");
}

#[test]
fn saccade_multi_eye_updates_are_additive() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let mut eye_values = vec![0.0; embed_dim];
    eye_values.extend(std::iter::repeat(1.0).take(embed_dim));
    let eye_token =
        Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
    saccade.eye_token = Param::from_tensor(eye_token);

    let images = Tensor::<Backend, 4>::random([1, 3, 8, 8], TensorDistribution::Default, &device);
    let (_, updates0) = saccade_eye_step(&saccade, images.clone(), 0);
    let (_, updates1) = saccade_eye_step(&saccade, images, 1);

    let mut state_sum: Vec<Tensor<Backend, 3>> = updates0
        .iter()
        .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
        .collect();
    for (state, update0, update1) in
        state_sum.iter_mut().zip(updates0.iter()).zip(updates1.iter()).map(
            |((state, update0), update1)| (state, update0, update1),
        )
    {
        *state = state.clone() + update0.clone() + update1.clone();
    }

    let mut state_seq: Vec<Tensor<Backend, 3>> = updates0
        .iter()
        .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
        .collect();
    for (state, update) in state_seq.iter_mut().zip(updates0.iter()) {
        *state = state.clone() + update.clone();
    }
    for (state, update) in state_seq.iter_mut().zip(updates1.iter()) {
        *state = state.clone() + update.clone();
    }

    for (sum, seq) in state_sum.iter().zip(state_seq.iter()) {
        let mse = (sum.clone() - seq.clone()).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }
}

#[test]
fn saccade_multi_eye_step_produces_finite_grads() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let images =
        Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 1, 1, true, false);
    let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

    let eye_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
        .expect("eye_token grad");
    let traj_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
        .expect("trajectory_token grad");
    assert_tensor_finite(eye_grad);
    assert_tensor_finite(traj_grad);
}

#[test]
fn saccade_artifact_frames_match_steps() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    saccade.config.artifact_max_images = 1;
    saccade.config.artifact_max_views = 4;
    saccade.config.artifact_every = 1;

    let images = Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([1], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);

    let steps = 3;
    let losses = saccade.forward_losses(batch, steps, 1, false, true);
    let artifacts = losses.artifacts.expect("artifacts");
    let frames = artifacts.frames.expect("frames");
    let views = artifacts.views.expect("views");
    let [batch, frame_count, _, _, _] = frames.shape().dims::<5>();
    let [view_batch, view_count, _, _, _] = views.shape().dims::<5>();
    assert_eq!(batch, 1);
    assert_eq!(frame_count, steps);
    assert_eq!(view_batch, 1);
    assert_eq!(view_count, 4);
}

#[test]
fn saccade_laplacian_roundtrip_is_exact() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grids = vec![
        PatchGrid { height: 4, width: 4 },
        PatchGrid { height: 2, width: 2 },
        PatchGrid { height: 1, width: 1 },
    ];
    let levels = vec![
        make_level::<Backend>(&device, 16, 2, 0.0),
        make_level::<Backend>(&device, 4, 2, 10.0),
        make_level::<Backend>(&device, 1, 2, 20.0),
    ];
    let residuals = saccade.decompose_pyramid(&levels, &grids);
    let composed = saccade.compose_pyramid(&residuals, &grids);
    for (orig, recon) in levels.iter().zip(composed.iter()) {
        let mse = (orig.clone() - recon.clone()).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }
}

#[test]
fn saccade_laplacian_drop_residual_matches_upsample() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grids = vec![
        PatchGrid { height: 4, width: 4 },
        PatchGrid { height: 2, width: 2 },
        PatchGrid { height: 1, width: 1 },
    ];
    let levels = vec![
        make_level::<Backend>(&device, 16, 2, 0.0),
        make_level::<Backend>(&device, 4, 2, 10.0),
        make_level::<Backend>(&device, 1, 2, 20.0),
    ];
    let residuals = saccade.decompose_pyramid(&levels, &grids);
    let mut truncated = residuals.clone();
    truncated[0] = Tensor::<Backend, 3>::zeros([1, 16, 2], &device);
    let composed = saccade.compose_pyramid(&truncated, &grids);
    let upsampled = saccade.upsample_tokens(levels[1].clone(), grids[1], grids[0]);
    let mse = (composed[0].clone() - upsampled).powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_mip_gaussian_weights_normalize() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let levels = vec![
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
            grid: PatchGrid { height: 2, width: 2 },
            image: Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device),
        },
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
            grid: PatchGrid { height: 1, width: 1 },
            image: Tensor::<Backend, 4>::zeros([2, 3, 4, 4], &device),
        },
    ];
    let mean = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.4, 0.7, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.3, 0.5], [2, 1, 1]),
        &device,
    );
    let weights = saccade.mip_gaussian_weights(&levels, mean, sigma);
    let mut total = weights[0].clone().sum_dim(2);
    for weight in weights.iter().skip(1) {
        total = total + weight.clone().sum_dim(2);
    }
    let ones = Tensor::<Backend, 3>::ones([2, 1, 1], &device);
    let diff = total.add(ones.mul_scalar(-1.0));
    let mse = diff.powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_fovea_params_use_single_trajectory_token() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
    assert_eq!(traj_len, SACCADE_TRAJ_TOKENS);

    let base_traj = saccade
        .trajectory_token
        .val()
        .reshape([1, traj_len, embed_dim]);
    let eye_embed = saccade
        .eye_token
        .val()
        .reshape([1, 1, embed_dim])
        .repeat_dim(1, traj_len);
    let traj_with_eye = base_traj + eye_embed;
    let fovea_params = saccade_fovea_params(&saccade, traj_with_eye.clone(), embed_dim);
    assert_eq!(fovea_params.shape().dims::<3>(), [1, 1, 3]);

    let levels = vec![
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([1, 4, 3], &device),
            grid: PatchGrid { height: 2, width: 2 },
            image: Tensor::<Backend, 4>::zeros([1, 3, 4, 4], &device),
        },
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([1, 1, 3], &device),
            grid: PatchGrid { height: 1, width: 1 },
            image: Tensor::<Backend, 4>::zeros([1, 3, 2, 2], &device),
        },
    ];
    let weights = saccade_weights_for_eye(&saccade, traj_with_eye, &levels, embed_dim);
    for weight in weights {
        let shape = weight.shape().dims::<3>();
        assert_eq!(shape[1], 1, "fovea weights should use a single token");
    }
}

#[test]
fn saccade_mip_scatter_gather_one_hot() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let residual = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 1, 3]),
        &device,
    );
    let mut state_levels = vec![
        Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
        Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
    ];
    let weights_level0 = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 1, 4]),
        &device,
    );
    let weights_level1 = Tensor::<Backend, 3>::zeros([2, 1, 1], &device);
    let weights = vec![weights_level0, weights_level1];

    saccade.apply_mip_residual(&mut state_levels, &weights, residual.clone());
    let gathered = saccade.mip_weighted_sum(&state_levels, &weights);
    let mse = (gathered - residual).powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_upsample_tokens_mismatch_returns_zero() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);

    let tokens = Tensor::<Backend, 3>::zeros([1, 1, 2], &device);
    let from = PatchGrid { height: 2, width: 2 };
    let to = PatchGrid { height: 3, width: 3 };
    let upsampled = saccade.upsample_tokens(tokens, from, to);

    assert_eq!(upsampled.shape().dims(), [1, 9, 2]);
    let value = upsampled
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("sum vec")[0];
    assert_eq!(value, 0.0);
}

#[test]
fn saccade_level_coords_cache_is_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grid = PatchGrid { height: 2, width: 2 };

    let _ = saccade.level_coords_cached(grid, &device);
    let _ = saccade.level_coords_cached(grid, &device);

    let len = saccade
        .level_coords_cache
        .inner
        .lock()
        .expect("level coords cache lock")
        .map
        .len();
    assert_eq!(len, 1);
}

#[test]
fn saccade_upsample_weights_cache_is_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let from = PatchGrid { height: 2, width: 2 };
    let to = PatchGrid { height: 4, width: 4 };

    let _ = saccade.upsample_weights_cached(from, to, &device);
    let _ = saccade.upsample_weights_cached(from, to, &device);

    let len = saccade
        .upsample_weights_cache
        .inner
        .lock()
        .expect("upsample weights cache lock")
        .map
        .len();
    assert_eq!(len, 1);
}

#[test]
fn saccade_step_produces_finite_grads() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let images =
        Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 1, 1, true, false);
    let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

    let token_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
        .expect("trajectory_token grad");
    let eye_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
        .expect("eye_token grad");
    assert_tensor_finite(token_grad);
    assert_tensor_finite(eye_grad);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_text_memory_stays_bounded_across_epochs() {
    type Backend = Autodiff<Wgpu<f32>>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let vocab = 64;
    let model = BDH::<Backend>::new(make_text_config(vocab), &device);
    let make_batch = || make_text_batch::<Backend>(&device, 2, 16, vocab);

    for _ in 0..2 {
        let output = burn_train::TrainStep::step(&model, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::TrainStep::step(&model, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "text",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_vision_saccade_memory_stays_bounded_across_epochs() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let config_path = vision_saccade_tiny_path();
    let mut config =
        load_vision_training_config(&[config_path]).expect("load vision_saccade_tiny");
    config.training.memory_cleanup_every = 0;

    let vision_config = config.vision.build();
    let saccade_config = match config.mode {
        VisionTrainingModeConfig::Saccade(config) => config,
        other => panic!("expected saccade config, got {other:?}"),
    };
    let fixed_steps = config
        .training
        .rollout_max_steps
        .unwrap_or(vision_config.steps)
        .min(4)
        .max(1);
    let rollout = VisionRollout {
        min_steps: fixed_steps,
        max_steps: fixed_steps,
        backprop_steps: fixed_steps,
    };

    let model = VisionDragonHatchling::<Backend>::new(vision_config.clone(), &device);
    let recon_patch_dim = vision_config
        .patch_size
        .saturating_mul(vision_config.patch_size)
        .saturating_mul(vision_config.in_channels);
    let saccade = VisionSaccadeModel::new(
        model,
        saccade_config,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [
                batch_size,
                vision_config.in_channels,
                vision_config.image_size,
                vision_config.image_size,
            ],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = burn_train::ValidStep::step(&saccade, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::ValidStep::step(&saccade, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "vision",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_vision_saccade_train_memory_stays_bounded_small_config() {
    type Backend = Autodiff<CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (saccade, vision_config) = make_saccade_model_with_dims::<Backend>(&device, 1, 64, 64, 16);
    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [
                batch_size,
                vision_config.in_channels,
                64,
                64,
            ],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = burn_train::TrainStep::step(&saccade, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::TrainStep::step(&saccade, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "vision_train_small",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_text_memory_stays_bounded_across_epochs() {
    type Backend = Autodiff<Cuda<f32>>;
    let device = burn_cuda::CudaDevice::default();

    let vocab = 64;
    let model = BDH::<Backend>::new(make_text_config(vocab), &device);
    let make_batch = || make_text_batch::<Backend>(&device, 2, 16, vocab);

    for _ in 0..2 {
        let output = burn_train::TrainStep::step(&model, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::TrainStep::step(&model, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(cuda_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "cuda_text",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_vision_saccade_memory_stays_bounded_across_epochs() {
    type Backend = Cuda<f32>;
    let device = burn_cuda::CudaDevice::default();

    let config_path = vision_saccade_tiny_path();
    let mut config =
        load_vision_training_config(&[config_path]).expect("load vision_saccade_tiny");
    config.training.memory_cleanup_every = 0;

    let vision_config = config.vision.build();
    let saccade_config = match config.mode {
        VisionTrainingModeConfig::Saccade(config) => config,
        other => panic!("expected saccade config, got {other:?}"),
    };
    let fixed_steps = config
        .training
        .rollout_max_steps
        .unwrap_or(vision_config.steps)
        .min(4)
        .max(1);
    let rollout = VisionRollout {
        min_steps: fixed_steps,
        max_steps: fixed_steps,
        backprop_steps: fixed_steps,
    };

    let model = VisionDragonHatchling::<Backend>::new(vision_config.clone(), &device);
    let recon_patch_dim = vision_config
        .patch_size
        .saturating_mul(vision_config.patch_size)
        .saturating_mul(vision_config.in_channels);
    let saccade = VisionSaccadeModel::new(
        model,
        saccade_config,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [
                batch_size,
                vision_config.in_channels,
                vision_config.image_size,
                vision_config.image_size,
            ],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = burn_train::ValidStep::step(&saccade, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::ValidStep::step(&saccade, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(cuda_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "cuda_vision",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_vision_saccade_train_memory_stays_bounded_small_config() {
    type Backend = Autodiff<Cuda<f32>>;
    let device = burn_cuda::CudaDevice::default();

    let (saccade, vision_config) = make_saccade_model_with_dims::<Backend>(&device, 1, 64, 64, 16);
    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [
                batch_size,
                vision_config.in_channels,
                64,
                64,
            ],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = burn_train::TrainStep::step(&saccade, make_batch());
        drop(output);
    }
    Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::TrainStep::step(&saccade, make_batch());
            drop(output);
        }
        Backend::sync(&device);
        Backend::memory_cleanup(&device);
        Backend::sync(&device);
        snapshots.push(cuda_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "cuda_vision_train_small",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}


