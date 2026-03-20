#[cfg(not(target_arch = "wasm32"))]
use super::{init_wgpu_test_runtime, wgpu_test_guard};
use crate::config::{
    VisionLossConfig, VisionPyramidMode, VisionReconLossConfig, VisionSaccadeCacheConfig,
    VisionSaccadeCrossViewConfig, VisionSaccadeInputProjectionConfig, VisionSaccadePolicyConfig,
    VisionTbpttConfig,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::config::{VisionTrainingModeConfig, load_vision_training_config};
use crate::foveation as foveation_cpu;
use crate::train::SaccadeFoveationSampler;
use crate::train::prelude::*;
use crate::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionBackboneKind, VisionLatentActivation,
};
use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
use burn_autodiff::Autodiff;
#[cfg(not(target_arch = "wasm32"))]
use burn_cubecl::CubeBackend;
use burn_dragon_core::{BDH, BDHConfig, FusedKernelConfig, ManifoldHyperConnectionsConfig};
use burn_dragon_language::ContextStrategyConfig;
use burn_dragon_language::TrainingHyperparameters;
use burn_dragon_language::dataset::SequenceBatch;
use burn_dragon_language::loss::language_model_loss;
use burn_dragon_language::train::resolve_train_schedule;
use burn_dragon_train::train::pipeline::ScheduleSource as TrainScheduleSource;
use burn_ndarray::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use burn_wgpu::Wgpu;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

fn make_training(max_iters: usize, epochs: Option<usize>) -> TrainingHyperparameters {
    TrainingHyperparameters {
        block_size: 16,
        batch_size: 2,
        gradient_accumulation_steps: 1,
        target_effective_batch_size: None,
        epochs,
        max_iters,
        log_frequency: 10,
        fast_train: false,
        context_strategy: ContextStrategyConfig::Infinite,
        gdpo: None,
    }
}

fn run_text_train_step<B: AutodiffBackend>(model: &BDH<B>, batch: SequenceBatch<B>) {
    let logits = model.forward(batch.inputs);
    let loss = language_model_loss::<B>(logits, batch.targets);
    let _ = loss.backward();
}

fn make_saccade_model<B: BackendTrait>(
    device: &B::Device,
    num_eyes: usize,
) -> (VisionSaccadeModel<B>, VisionDragonConfig) {
    make_saccade_model_with_dims(device, num_eyes, 8, 8, 4)
}

fn make_saccade_model_with_dims<B: BackendTrait>(
    device: &B::Device,
    num_eyes: usize,
    image_width: usize,
    image_height: usize,
    patch_size: usize,
) -> (VisionSaccadeModel<B>, VisionDragonConfig) {
    let patch_size = patch_size.max(1);
    let image_size = image_width.max(image_height).max(patch_size);
    let grid_w = (image_width / patch_size).max(1);
    let grid_h = (image_height / patch_size).max(1);
    let vision_config = VisionDragonConfig {
        image_size,
        patch_size,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Dense,
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: num_eyes.max(1),
        cross_eye_steps: 0,
        token_state_norm: true,
        normalization: burn_dragon_core::DragonNormConfig::default(),
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: grid_h,
        pos_max_width: grid_w,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: Default::default(),
    };
    let model = VisionDragon::<B>::new(vision_config.clone(), device);
    let saccade_config = VisionSaccadeConfig {
        num_eyes,
        traj_tokens: 1,
        traj_update_alpha: 1.0,
        mip_levels: 3,
        pyramid_mode: VisionPyramidMode::Laplacian,
        fovea_sampling_mode: VisionFoveaSamplingMode::Sequential,
        fovea_warp_mode: VisionFoveaWarpMode::Warped,
        fovea_subsamples: 4,
        fovea_radius_scale: 1.0,
        fovea_subpatch_size: 0,
        fovea_scatter_mode: VisionFoveaScatterMode::Tensor,
        grid_sample_max_mb: 512,
        mip_concat_max_mb: 512,
        pyramid_feature_dim: None,
        inner_steps: 1,
        low_mem_pre_rollout: true,
        recon_batch_chunk: 0,
        recon_max_elems: 50_000_000,
        tbptt: VisionTbpttConfig::default(),
        policy: VisionSaccadePolicyConfig::default(),
        cache: VisionSaccadeCacheConfig::default(),
        cross_view: VisionSaccadeCrossViewConfig::default(),
        input_projection: VisionSaccadeInputProjectionConfig::default(),
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
                ..VisionReconLossConfig::default()
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
        vision_config.patch_size,
        rollout,
        recon_patch_dim,
        1,
        0,
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
fn make_rho_stream_contract_config(
    wgpu_forward_kernel: bool,
    wgpu_rollout_fused: bool,
) -> VisionDragonConfig {
    let mut config = VisionDragonConfig {
        image_size: 8,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Cellular,
        in_channels: 3,
        embed_dim: 16,
        steps: 3,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 16,
        projection_hidden_dim: 32,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        normalization: burn_dragon_core::DragonNormConfig::default(),
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: Default::default(),
    };
    config.rho_stream.enabled = true;
    config.rho_stream.local_radius = 1;
    config.rho_stream.local_diagonals = false;
    config.rho_stream.local_self = true;
    config.rho_stream.decay = 0.9;
    config.rho_stream.wgpu_forward_kernel = wgpu_forward_kernel;
    config.rho_stream.wgpu_rollout_fused = wgpu_rollout_fused;
    config
}

#[cfg(not(target_arch = "wasm32"))]
fn rho_stream_contract_tokens<B: BackendTrait>(device: &B::Device) -> Tensor<B, 3> {
    let data: Vec<f32> = (0..(2 * 4 * 16))
        .map(|idx| ((idx % 19) as f32 - 9.0) / 10.0)
        .collect();
    Tensor::<B, 3>::from_data(TensorData::new(data, [2, 4, 16]), device)
}

#[cfg(not(target_arch = "wasm32"))]
fn rho_stream_contract_loss<B: BackendTrait>(
    model: &VisionDragon<B>,
    tokens: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let output = model.forward_tokens_embed_steps_rollout(tokens, 3, 3);
    output.patch_tokens.tanh().powf_scalar(2.0).mean()
        + output.cls_token.tanh().powf_scalar(2.0).mean()
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_tensor_close<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
    atol: f32,
    rtol: f32,
) {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs vec");
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");

    let mut max_diff = 0.0_f32;
    let mut max_tol = 0.0_f32;
    let mut max_lhs = 0.0_f32;
    let mut max_rhs = 0.0_f32;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        let tol = atol + rtol * b.abs();
        if diff > max_diff {
            max_diff = diff;
            max_tol = tol;
            max_lhs = *a;
            max_rhs = *b;
        }
    }

    assert!(
        max_diff <= max_tol,
        "max difference {max_diff} exceeds tolerance {max_tol} (lhs={max_lhs}, rhs={max_rhs})"
    );
}

fn make_checkerboard_image(
    channels: usize,
    height: usize,
    width: usize,
    cell_size: usize,
) -> Vec<f32> {
    let mut data = Vec::with_capacity(channels * height * width);
    let cell = cell_size.max(1);
    for _c in 0..channels {
        for y in 0..height {
            let cy = y / cell;
            for x in 0..width {
                let cx = x / cell;
                let value = if (cx + cy) % 2 == 0 { 0.0 } else { 1.0 };
                data.push(value);
            }
        }
    }
    data
}

fn checkerboard_center_metrics(
    patch: &[f32],
    channels: usize,
    height: usize,
    width: usize,
) -> (f32, f32) {
    let start_x = width / 4;
    let start_y = height / 4;
    let mut end_x = (width * 3) / 4;
    let mut end_y = (height * 3) / 4;
    if end_x <= start_x + 1 {
        end_x = width.max(start_x + 2);
    }
    if end_y <= start_y + 1 {
        end_y = height.max(start_y + 2);
    }
    let end_x = end_x.min(width);
    let end_y = end_y.min(height);

    let mut min_val = f32::INFINITY;
    let mut max_val = f32::NEG_INFINITY;
    let mut diff_sum = 0.0;
    let mut diff_count = 0usize;

    for c in 0..channels {
        let base = c * height * width;
        for y in start_y..end_y {
            let row = base + y * width;
            for x in start_x..end_x {
                let value = patch[row + x];
                min_val = min_val.min(value);
                max_val = max_val.max(value);
            }
        }
        for y in start_y..(end_y - 1) {
            let row = base + y * width;
            let next_row = base + (y + 1) * width;
            for x in start_x..(end_x - 1) {
                let value = patch[row + x];
                diff_sum += (value - patch[row + x + 1]).abs();
                diff_sum += (value - patch[next_row + x]).abs();
                diff_count += 2;
            }
        }
    }

    let range = if min_val.is_finite() {
        max_val - min_val
    } else {
        0.0
    };
    let mean_diff = if diff_count == 0 {
        0.0
    } else {
        diff_sum / diff_count as f32
    };
    (mean_diff, range)
}

fn run_foveation_snellen<B: BackendTrait>(device: &B::Device, backend_label: &str) {
    let batch = 1;
    let channels = 3;
    let width = 32;
    let height = 32;
    let patch_size = 16;
    let data = make_checkerboard_image(channels, height, width, 1);
    let images = Tensor::<B, 4>::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        device,
    );

    let mut sampling_modes = vec![
        VisionFoveaSamplingMode::Batched,
        VisionFoveaSamplingMode::Sequential,
    ];
    if crate::train::foveation::cubecl::supports_backend::<B>() {
        sampling_modes.push(VisionFoveaSamplingMode::Cubecl);
    }
    if crate::train::foveation::wgsl::supports_backend::<B>() {
        sampling_modes.push(VisionFoveaSamplingMode::Wgsl);
    }

    let cases = [(0.06, 0.2), (0.08, 0.25), (0.1, 0.35), (0.12, 0.45)];
    let normalized_threshold = 0.3;
    let range_threshold = 0.6;

    for sampling_mode in sampling_modes {
        let (mut saccade, vision_config) =
            make_saccade_model_with_dims::<B>(device, 1, width, height, patch_size);
        saccade.config.pyramid_mode = VisionPyramidMode::Stacked;
        saccade.config.fovea_sampling_mode = sampling_mode;
        saccade.config.fovea_warp_mode = VisionFoveaWarpMode::Warped;
        saccade.config.fovea_subpatch_size = 0;
        saccade.config.mip_levels = 3;

        let mut sampler =
            SaccadeFoveationSampler::<B>::new(vision_config, saccade.config.clone(), device);
        sampler.update_image(images.clone());
        let patch_size = sampler.patch_size();

        let mut best_norm = 0.0;
        let mut best_diff = 0.0;
        let mut best_range = 0.0;
        let mut best_case = None;
        let mut passed = false;
        for (case_idx, (sigma_val, radius_val)) in cases.iter().copied().enumerate() {
            let mean =
                Tensor::<B, 2>::from_data(TensorData::new(vec![0.5, 0.5], [batch, 2]), device);
            let sigma =
                Tensor::<B, 2>::from_data(TensorData::new(vec![sigma_val], [batch, 1]), device);
            let radius =
                Tensor::<B, 2>::from_data(TensorData::new(vec![radius_val], [batch, 1]), device);
            let patch_view = sampler.sample_patch_with_radius(mean, sigma, radius);
            let patch_vec = patch_view
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("patch vec");
            let (mean_diff, range) =
                checkerboard_center_metrics(&patch_vec, channels, patch_size, patch_size);
            let normalized = if range > 0.0 { mean_diff / range } else { 0.0 };
            if normalized > best_norm {
                best_norm = normalized;
                best_diff = mean_diff;
                best_range = range;
                best_case = Some((case_idx, sigma_val, radius_val));
            }
            if normalized >= normalized_threshold && range >= range_threshold {
                passed = true;
                break;
            }
        }
        assert!(
            passed,
            "backend {backend_label} sampling {sampling_mode:?} best_norm {best_norm:.3} best_diff {best_diff:.3} best_range {best_range:.3} best_case {best_case:?} thresholds norm {normalized_threshold:.3} range {range_threshold:.3}"
        );
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn vision_saccade_tiny_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir
            .join("..")
            .join("..")
            .join("config")
            .join("vision")
            .join("saccade")
            .join("tiny.toml"),
        manifest_dir
            .join("..")
            .join("config")
            .join("vision")
            .join("saccade")
            .join("tiny.toml"),
        manifest_dir
            .join("config")
            .join("vision")
            .join("saccade")
            .join("tiny.toml"),
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
    _allocs: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_memory_snapshot(device: &burn_wgpu::WgpuDevice) -> MemorySnapshot {
    let usage = <burn_wgpu::WgpuRuntime as cubecl::Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
        _allocs: usage.number_allocs,
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_memory_snapshot(device: &burn_cuda::CudaDevice) -> MemorySnapshot {
    let usage = <cubecl::cuda::CudaRuntime as cubecl::Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
        _allocs: usage.number_allocs,
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_memory_snapshot_safe(device: &burn_cuda::CudaDevice) -> Option<MemorySnapshot> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cuda_memory_snapshot(device)
    }))
    .ok()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_memory_cleanup_safe<B: BackendTrait>(device: &B::Device) -> bool {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| B::memory_cleanup(device))).is_ok()
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_memory_pool_stable() -> bool {
    static STABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STABLE.get_or_init(|| {
        if !crate::train::foveation::cubecl::supports_backend::<Cuda<f32>>() {
            return false;
        }
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let device = burn_cuda::CudaDevice::default();
            let _ = Tensor::<Cuda<f32>, 2>::zeros([1, 1], &device);
            let _ = Cuda::<f32>::sync(&device);
            let _ = cuda_memory_snapshot(&device);
        }))
        .is_ok()
    })
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
fn cuda_random_kernel_stable() -> bool {
    static STABLE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *STABLE.get_or_init(|| {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let device = burn_cuda::CudaDevice::default();
            let _ = Tensor::<Cuda<f32>, 2>::random([1, 1], TensorDistribution::Default, &device);
            let _ = Cuda::<f32>::sync(&device);
        }))
        .is_ok()
    })
}

#[cfg(not(target_arch = "wasm32"))]
fn assert_memory_growth_bounded(
    label: &str,
    snapshots: &[MemorySnapshot],
    max_reserved_growth: u64,
    max_in_use_growth: u64,
) {
    assert!(
        !snapshots.is_empty(),
        "{label}: no memory snapshots collected"
    );
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
    BDHConfig {
        n_layer: 2,
        n_embd: 64,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        n_expert: 1,
        dropout: 0.0,
        vocab_size: vocab_size.max(8),
        ..Default::default()
    }
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
    let inputs = Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [batch, block]), device);
    let targets = Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [batch, block]), device);
    SequenceBatch::new(inputs, targets, None)
}

fn run_foveation_equivalence<B: BackendTrait>(device: &B::Device, backend_label: &str) {
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
    let size_configs = [(8, 8, 4, 2), (12, 8, 4, 2), (12, 12, 6, 3), (16, 16, 8, 4)];
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
    let warp_modes = [VisionFoveaWarpMode::Warped, VisionFoveaWarpMode::Patched];

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
                        make_saccade_model_with_dims::<B>(device, 1, width, height, patch_size);
                    saccade.config.pyramid_mode = pyramid_mode;
                    saccade.config.fovea_sampling_mode = sampling_mode;
                    saccade.config.fovea_warp_mode = warp_mode;
                    saccade.config.fovea_subpatch_size =
                        if matches!(sampling_mode, VisionFoveaSamplingMode::Subpatch) {
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
                        VisionPyramidMode::Stacked => foveation_cpu::PyramidMode::Stacked,
                        VisionPyramidMode::Laplacian => foveation_cpu::PyramidMode::Laplacian,
                    };
                    let cache = foveation_cpu::build_pyramid_cache(
                        foveation_cpu::image_from_nchw(&data, 0, channels, height, width)
                            .expect("cpu image"),
                        cpu_depth,
                        cpu_mode,
                    );

                    for (case_idx, (mean_vals, sigma_val, radius_val)) in cases.iter().enumerate() {
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
                        let patch_view = sampler.sample_patch_with_radius(mean, sigma, radius);
                        let patch_vec = patch_view
                            .to_data()
                            .convert::<f32>()
                            .into_vec::<f32>()
                            .expect("patch vec");

                        let expected_patch = foveation_cpu::render_foveated_patch_with_radius(
                            &cache,
                            mean_vals,
                            sigma_val,
                            radius_val,
                            patch_size,
                            match warp_mode {
                                VisionFoveaWarpMode::Warped => foveation_cpu::FoveaWarpMode::Warped,
                                VisionFoveaWarpMode::Patched => {
                                    foveation_cpu::FoveaWarpMode::Patched
                                }
                            },
                        );
                        let mut expected = vec![0.0f32; channels * patch_size * patch_size];
                        let channel_stride = patch_size * patch_size;
                        for y in 0..patch_size {
                            for x in 0..patch_size {
                                let src = (y * patch_size + x) * 3;
                                let dst = y * patch_size + x;
                                expected[dst] = expected_patch[src];
                                expected[dst + channel_stride] = expected_patch[src + 1];
                                expected[dst + 2 * channel_stride] = expected_patch[src + 2];
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

fn run_scatter_equivalence<B: BackendTrait>(device: &B::Device, backend_label: &str) {
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
    let levels = saccade.build_mip_pyramid(images.clone(), patch_size);

    let mean = Tensor::<B, 3>::from_data(
        TensorData::new(vec![0.3, 0.7, 0.6, 0.4], [batch, 1, 2]),
        device,
    );
    let sigma = Tensor::<B, 3>::from_data(TensorData::new(vec![0.2, 0.35], [batch, 1, 1]), device);
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
                    saccade
                        .weighted_sum_tokens(weight.clone().swap_dims(1, 2), residual_pool.clone())
                })
                .collect();

            for mode in scatter_modes.iter().copied() {
                saccade.config.fovea_scatter_mode = mode;
                for (level_idx, (weight, base)) in weights.iter().zip(baseline.iter()).enumerate() {
                    let output = saccade
                        .weighted_sum_tokens(weight.clone().swap_dims(1, 2), residual_pool.clone());
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

fn assert_tensor_nonzero<B: BackendTrait, const D: usize>(tensor: Tensor<B, D>, eps: f32) {
    let values = tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor vec");
    assert!(!values.is_empty(), "tensor has no values");
    let mut max_abs = 0.0f32;
    for value in values {
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }
    assert!(max_abs > eps, "tensor max abs {max_abs} <= {eps}");
}

fn gradient_focus_stats(
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    inner_frac: f32,
) -> (f32, f32) {
    let inner_frac = inner_frac.clamp(0.05, 1.0);
    let min_side = width.min(height).max(1) as f32;
    let radius = 0.5 * inner_frac * min_side;
    let radius_sq = radius * radius;
    let cx = (width as f32 - 1.0) * 0.5;
    let cy = (height as f32 - 1.0) * 0.5;
    let mut inner_sum = 0.0f32;
    let mut inner_count = 0usize;
    let mut outer_sum = 0.0f32;
    let mut outer_count = 0usize;

    for c in 0..channels {
        let base = c * height * width;
        for y in 0..height {
            let dy = y as f32 - cy;
            let row = base + y * width;
            for x in 0..width {
                let dx = x as f32 - cx;
                let value = values[row + x].abs();
                if dx * dx + dy * dy <= radius_sq {
                    inner_sum += value;
                    inner_count += 1;
                } else {
                    outer_sum += value;
                    outer_count += 1;
                }
            }
        }
    }

    let inner_mean = if inner_count == 0 {
        0.0
    } else {
        inner_sum / inner_count as f32
    };
    let outer_mean = if outer_count == 0 {
        0.0
    } else {
        outer_sum / outer_count as f32
    };
    (inner_mean, outer_mean)
}

fn block_mean_std(
    values: &[f32],
    channels: usize,
    height: usize,
    width: usize,
    blocks: usize,
) -> f32 {
    let blocks = blocks.max(1);
    let block_h = height / blocks;
    let block_w = width / blocks;
    if block_h == 0 || block_w == 0 {
        return 0.0;
    }
    let mut means = Vec::with_capacity(blocks * blocks);
    for by in 0..blocks {
        for bx in 0..blocks {
            let mut sum = 0.0f32;
            let mut count = 0usize;
            let start_y = by * block_h;
            let start_x = bx * block_w;
            let end_y = (start_y + block_h).min(height);
            let end_x = (start_x + block_w).min(width);
            for c in 0..channels {
                let base = c * height * width;
                for y in start_y..end_y {
                    let row = base + y * width;
                    for x in start_x..end_x {
                        sum += values[row + x];
                        count += 1;
                    }
                }
            }
            if count > 0 {
                means.push(sum / count as f32);
            }
        }
    }
    if means.is_empty() {
        return 0.0;
    }
    let mean = means.iter().sum::<f32>() / means.len() as f32;
    let var = means
        .iter()
        .map(|value| {
            let diff = value - mean;
            diff * diff
        })
        .sum::<f32>()
        / means.len() as f32;
    var.sqrt()
}

fn saccade_eye_step<B: BackendTrait>(
    saccade: &VisionSaccadeModel<B>,
    images: Tensor<B, 4>,
    eye_idx: usize,
) -> (Tensor<B, 3>, Vec<Tensor<B, 3>>) {
    let device = images.device();
    let [batch, _channels, _height, _width] = images.shape().dims::<4>();
    let patch = saccade.model.patch_embed_raw(images.clone());
    let grid_h = patch.grid.height;
    let grid_w = patch.grid.width;
    assert!(grid_h > 0 && grid_w > 0);
    let patch_size = saccade.model.patch_size().max(1);
    assert!(patch_size > 0);

    let mip_levels = saccade.build_mip_pyramid(images, patch_size);
    assert!(!mip_levels.is_empty());
    let input_levels: Vec<Tensor<B, 3>> = mip_levels
        .iter()
        .map(|level| level.tokens.clone())
        .collect();
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
    let residual_pool =
        residual
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

mod foveation;
mod memory;
mod model_smoke;
mod rho_stream;
mod saccade;
mod schedule;
mod video;
