use super::*;
use crate::VisionBackboneKind;
use crate::train::metrics::{
    LongRolloutComErrorToH24Input, LongRolloutInvToHorizonInput,
    LongRolloutStateMotionToHorizonInput, LongRolloutStateNormRatioToHorizonInput,
    LongRolloutVelocityErrorToH24Input, RolloutComErrorToH24Input, RolloutVelocityErrorToH24Input,
    VisionOutput,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::train::{init_wgpu_test_runtime, wgpu_test_guard};
use burn::optim::{AdamWConfig, GradientsParams, LearningRate};
use burn_autodiff::Autodiff;
#[cfg(not(target_arch = "wasm32"))]
use burn_cubecl::CubeBackend;
use burn_dragon_train::train::metrics::OptionalScalarValue;
use burn_ndarray::NdArray;
use burn_train::metric::Adaptor;

#[cfg(not(target_arch = "wasm32"))]
#[derive(Clone, Copy, Debug)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

#[cfg(not(target_arch = "wasm32"))]
fn wgpu_memory_snapshot(device: &burn_wgpu::WgpuDevice) -> MemorySnapshot {
    let usage = <burn_wgpu::WgpuRuntime as cubecl::Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
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
    assert!(
        growth_reserved <= max_reserved_growth,
        "{label}: reserved bytes grew by {growth_reserved} (> {max_reserved_growth}); snapshots={snapshots:?}"
    );
    assert!(
        growth_in_use <= max_in_use_growth,
        "{label}: in-use bytes grew by {growth_in_use} (> {max_in_use_growth}); snapshots={snapshots:?}"
    );
}

fn make_video_configs(
    temporal_wgpu_recurrent: bool,
    temporal_rollout_fused: bool,
    fast_steps: usize,
) -> (VisionDragonConfig, VisionVideoLejepaConfig) {
    let vision = VisionDragonConfig {
        image_size: 8,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Cellular,
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 12,
        projection_hidden_dim: 24,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: Default::default(),
        trm_graph: Default::default(),
        rho_stream: crate::VisionRhoStreamConfig {
            enabled: true,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            decay: 0.9,
            mode_embeddings: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
        },
    };
    let mut video = VisionVideoLejepaConfig {
        context_frames: 3,
        target_frames: 2,
        ..VisionVideoLejepaConfig::default()
    };
    video.loss.sigreg.enabled = true;
    video.loss.sigreg.lambda = 0.02;
    video.loss.cosine_weight = 0.1;
    video.loss.probe_weight = 0.25;
    video.temporal.n_layer = 2;
    video.temporal.n_head = 4;
    video.temporal.mlp_internal_dim_multiplier = 2;
    video.temporal.rollout_fast_steps_per_slow_step = fast_steps;
    video.temporal.mode_embeddings = true;
    video.temporal.refine_passes = 0;
    video.temporal.fused = true;
    video.temporal.wgpu_recurrent_kernel = temporal_wgpu_recurrent;
    video.temporal.wgpu_rollout_fused = temporal_rollout_fused;
    video.temporal.latent_block_size = 8;
    video.temporal.time_block_size = 8;
    (vision, video)
}

fn make_video_configs_runlike(
    temporal_wgpu_recurrent: bool,
    temporal_rollout_fused: bool,
    fast_steps: usize,
) -> (VisionDragonConfig, VisionVideoLejepaConfig) {
    let mut vision = VisionDragonConfig {
        image_size: 32,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Dense,
        in_channels: 3,
        embed_dim: 64,
        steps: 2,
        n_head: 8,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 32,
        projection_hidden_dim: 64,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Rope,
        pos_max_height: 8,
        pos_max_width: 8,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: Default::default(),
        trm_graph: Default::default(),
        rho_stream: crate::VisionRhoStreamConfig {
            enabled: false,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            decay: 0.9,
            mode_embeddings: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
        },
    };
    vision = enable_pyramid_backbone(vision);
    let mut video = VisionVideoLejepaConfig {
        context_frames: 4,
        target_frames: 6,
        train_target_frames_min: 6,
        train_target_frames_max: 6,
        ..VisionVideoLejepaConfig::default()
    };
    video.teacher_ema.enabled = false;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;
    video.temporal.n_layer = 2;
    video.temporal.n_head = 8;
    video.temporal.mlp_internal_dim_multiplier = 2;
    video.temporal.rollout_fast_steps_per_slow_step = fast_steps;
    video.temporal.mode_embeddings = true;
    video.temporal.refine_passes = 1;
    video.temporal.fused = true;
    video.temporal.wgpu_recurrent_kernel = temporal_wgpu_recurrent;
    video.temporal.wgpu_rollout_fused = temporal_rollout_fused;
    video.temporal.latent_block_size = 8;
    video.temporal.time_block_size = 8;
    (vision, video)
}

fn enable_pyramid_backbone(mut vision: VisionDragonConfig) -> VisionDragonConfig {
    vision.backbone = VisionBackboneKind::Pyramid;
    vision.rho_stream.enabled = false;
    vision.trm_graph.enabled = true;
    vision.trm_graph.coarse_stride = 2;
    vision.trm_graph.hub_count = 2;
    vision.trm_graph.rank = 4;
    vision.trm_graph.value_dim = 16;
    vision.trm_graph.local_radius = 1;
    vision.trm_graph.local_diagonals = true;
    vision.trm_graph.local_self = true;
    vision.trm_graph.decay = 0.9;
    vision
}

fn make_video_model<B: BackendTrait>(
    vision: &VisionDragonConfig,
    video: &VisionVideoLejepaConfig,
    device: &B::Device,
) -> VisionVideoLejepaModel<B> {
    let frame_model = VisionDragon::<B>::new(vision.clone(), device);
    VisionVideoLejepaModel::new(
        frame_model,
        video.clone(),
        vision,
        VisionRollout {
            min_steps: 2,
            max_steps: 2,
            backprop_steps: 2,
        },
        10,
        device,
    )
}

fn toy_video_batch_with_lengths<B: BackendTrait>(
    device: &B::Device,
    context_len: usize,
    target_len: usize,
    extra_future: usize,
) -> VideoClipBatch<B> {
    toy_video_batch_with_lengths_custom(device, context_len, target_len, extra_future, 2, 8)
}

fn toy_video_batch_with_lengths_custom<B: BackendTrait>(
    device: &B::Device,
    context_len: usize,
    target_len: usize,
    extra_future: usize,
    batch: usize,
    size: usize,
) -> VideoClipBatch<B> {
    let batch = batch.max(1);
    let size = size.max(4);
    let frames = context_len + target_len + extra_future;
    let channels = 3;
    let mut data = Vec::with_capacity(batch * frames * channels * size * size);
    for b in 0..batch {
        for t in 0..frames {
            for _channel in 0..channels {
                for y in 0..size {
                    for x in 0..size {
                        let center_x = (t + b) % (size - 2);
                        let center_y = (t + 2 * b) % (size - 2);
                        let dx = x.abs_diff(center_x);
                        let dy = y.abs_diff(center_y);
                        let value = if dx <= 1 && dy <= 1 { 1.0 } else { 0.0 };
                        data.push(value);
                    }
                }
            }
        }
    }
    let clip = Tensor::<B, 5>::from_data(
        TensorData::new(data, [batch, frames, channels, size, size]),
        device,
    );
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new((0..batch).map(|idx| (idx % 10) as i64).collect(), [batch]),
        device,
    );
    VideoClipBatch::new(clip, labels, context_len, target_len)
}

fn toy_video_batch<B: BackendTrait>(device: &B::Device) -> VideoClipBatch<B> {
    toy_video_batch_with_lengths(device, 3, 2, 0)
}

fn toy_video_batch_same_observation_different_history<B: BackendTrait>(
    device: &B::Device,
) -> VideoClipBatch<B> {
    let batch = 2;
    let frames = 5;
    let channels = 3;
    let size = 8;
    let mut data = Vec::with_capacity(batch * frames * channels * size * size);
    let trajectories = [
        [(0, 0), (2, 2), (3, 2), (4, 2), (5, 2)],
        [(5, 0), (2, 2), (3, 2), (4, 2), (5, 2)],
    ];
    for sample in trajectories {
        for (center_x, center_y) in sample {
            for _channel in 0..channels {
                for y in 0..size {
                    for x in 0..size {
                        let dx = x.abs_diff(center_x);
                        let dy = y.abs_diff(center_y);
                        let value = if dx <= 1 && dy <= 1 { 1.0 } else { 0.0 };
                        data.push(value);
                    }
                }
            }
        }
    }
    let clip = Tensor::<B, 5>::from_data(
        TensorData::new(data, [batch, frames, channels, size, size]),
        device,
    );
    let labels =
        Tensor::<B, 1, Int>::from_data(TensorData::new(vec![1_i64, 2_i64], [batch]), device);
    VideoClipBatch::new(clip, labels, 3, 2)
}

fn projected_future_frames<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_frames: Tensor<B, 5>,
    context_len: usize,
    target_len: usize,
    steps: usize,
    projection_dim: usize,
) -> Tensor<B, 3> {
    let [batch_size, _, channels, height, width] = clip_frames.shape().dims::<5>();
    let target_end = (context_len + target_len).min(clip_frames.shape().dims::<5>()[1]);
    let target_len = target_end.saturating_sub(context_len).max(1);
    let target_frames = clip_frames.slice_dim(1, context_len..target_end).reshape([
        batch_size * target_len,
        channels,
        height,
        width,
    ]);
    let patch = model.patch_embed(target_frames);
    let encoded = model.forward_tokens_embed_steps_rollout(patch.tokens, steps, steps);
    model
        .project_tokens(encoded.cls_token.unsqueeze_dim::<3>(1))
        .reshape([batch_size, target_len, projection_dim])
}

fn assert_close<B: BackendTrait, const D: usize>(
    lhs: Tensor<B, D>,
    rhs: Tensor<B, D>,
    atol: f32,
    rtol: f32,
) {
    let lhs = lhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("lhs");
    let rhs = rhs
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rhs");
    assert_eq!(lhs.len(), rhs.len(), "length mismatch");
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        let diff = (*a - *b).abs();
        let tol = atol + rtol * b.abs();
        assert!(
            diff <= tol,
            "difference {diff} exceeds tolerance {tol} (lhs={a}, rhs={b})"
        );
    }
}
mod artifacts;
mod memory;
mod parity;
mod smoke;
mod training;
