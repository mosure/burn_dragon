use crate::train::prelude::*;
use burn::optim::Optimizer;
use burn::tensor::Distribution;
use burn_autodiff::Autodiff;
use burn_dragon_core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
use crate::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionLatentActivation,
    VisionPatchEmbedMode,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::config::{VisionTrainingModeConfig, load_vision_training_config};
#[cfg(not(target_arch = "wasm32"))]
use crate::train::pipeline::resolve_vision_rollout;
use crate::config::{VisionMaeCrossViewConfig, VisionMaeLossConfig, VisionReconLossConfig};
use burn_ndarray::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[test]
fn mae_pyramid_recon_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let vision_config = VisionDragonConfig {
        image_size: 8,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 8,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
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
        mhc: ManifoldHyperConnectionsConfig::default(),
    };
    let recon_patch_dim =
        vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    let num_eyes = vision_config.num_eyes;
    let mae_config = VisionMaeConfig {
        loss: VisionMaeLossConfig {
            recon: VisionReconLossConfig {
                weight: 1.0,
                mask_ratio: 0.5,
                hidden_dim: 8,
                ..VisionReconLossConfig::default()
            },
        },
        pyramid_levels: 2,
        ..VisionMaeConfig::default()
    };
    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 1,
        backprop_steps: 1,
    };
    let model = VisionDragon::<Backend>::new(vision_config, &device);
    let mae = VisionMaeModel::new(
        model,
        mae_config,
        num_eyes,
        8,
        rollout,
        recon_patch_dim,
        &device,
    );

    let images = Tensor::<Backend, 4>::random([1, 3, 8, 8], Distribution::Default, &device);
    let (loss_sum, mask_sum, _) = mae.recon_loss(images, 1, 1, false, false);
    let loss = loss_sum / mask_sum.add_scalar(LEJEPA_EPS);
    let value = loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn mae_cross_view_forward_is_finite() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();

    let vision_config = VisionDragonConfig {
        image_size: 8,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 16,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 16,
        projection_hidden_dim: 32,
        use_cls_token: false,
        cls_sync_alpha: 0.0,
        num_eyes: 2,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::Softmax,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig {
            enabled: true,
            num_streams: 2,
            num_views: 2,
            ..ManifoldHyperConnectionsConfig::default()
        },
    };
    let recon_patch_dim =
        vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    let num_eyes = vision_config.num_eyes;
    let mae_config = VisionMaeConfig {
        loss: VisionMaeLossConfig {
            recon: VisionReconLossConfig {
                weight: 1.0,
                mask_ratio: 0.5,
                hidden_dim: 32,
                ..VisionReconLossConfig::default()
            },
        },
        cross_view: VisionMaeCrossViewConfig {
            enabled: true,
            min_overlap: 0.3,
            max_attempts: 1,
            masked_eye: 1,
            fuse_alpha: 0.0,
            visible_weight: 0.0,
        },
        pyramid_levels: 1,
        artifact_every: 0,
        artifact_max_images: 0,
        artifact_max_views: 0,
        ..VisionMaeConfig::default()
    };
    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 1,
        backprop_steps: 1,
    };
    let model = VisionDragon::<Backend>::new(vision_config, &device);
    let mae = VisionMaeModel::new(
        model,
        mae_config,
        num_eyes,
        16,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 2;
    let images =
        Tensor::<Backend, 4>::random([batch_size, 3, 8, 8], Distribution::Default, &device);
    let target =
        Tensor::<Backend, 4>::random([batch_size, 3, 8, 8], Distribution::Default, &device);
    let views = Tensor::cat(
        vec![
            images.clone().unsqueeze_dim::<5>(1),
            target.clone().unsqueeze_dim::<5>(1),
        ],
        1,
    );
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = ImageNetBatch::new(images, None, Some(views), None, None, None, labels, None, None);

    let losses = mae.forward_losses(batch, 1, 1, false, false);
    let value = losses
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec")[0];
    assert!(value.is_finite());
}

fn toy_images<B: BackendTrait>(
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let mut data = Vec::with_capacity(batch * channels * height * width);
    let denom_w = (width - 1).max(1) as f32;
    let denom_h = (height - 1).max(1) as f32;
    for _ in 0..batch {
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
    }
    Tensor::<B, 4>::from_data(
        TensorData::new(data, [batch, channels, height, width]),
        device,
    )
}

#[cfg(not(target_arch = "wasm32"))]
fn vision_identity_tiny_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir
            .join("..")
            .join("..")
            .join("config")
            .join("vision")
            .join("identity")
            .join("tiny.toml"),
        manifest_dir
            .join("..")
            .join("config")
            .join("vision")
            .join("identity")
            .join("tiny.toml"),
        manifest_dir.join("config").join("vision").join("identity").join("tiny.toml"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    candidates[0].clone()
}

#[cfg(not(target_arch = "wasm32"))]
fn resolve_config_path(raw: &str) -> PathBuf {
    let path = PathBuf::from(raw);
    if path.is_absolute() || path.exists() {
        return path;
    }
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir.join(raw),
        manifest_dir.join("..").join(raw),
        manifest_dir.join("..").join("..").join(raw),
    ];
    for candidate in candidates {
        if candidate.exists() {
            return candidate;
        }
    }
    path
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn mae_config_smoke_from_env() {
    let Ok(raw) = std::env::var("VISION_MAE_CONFIG_SMOKE") else {
        return;
    };
    let mut paths = Vec::new();
    for part in raw.split(&[',', ';'][..]) {
        let trimmed = part.trim();
        if !trimmed.is_empty() {
            paths.push(resolve_config_path(trimmed));
        }
    }
    if paths.is_empty() {
        return;
    }
    let config =
        load_vision_training_config(&paths).expect("load VISION_MAE_CONFIG_SMOKE config");
    let mae_cfg = match &config.mode {
        VisionTrainingModeConfig::Mae(mae) => mae.clone(),
        _ => panic!("VISION_MAE_CONFIG_SMOKE config is not mae mode"),
    };

    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let rollout = resolve_vision_rollout(&config.training, config.vision.steps)
        .expect("resolve rollout");
    let vision_cfg = config.vision.build();
    let recon_patch_dim =
        vision_cfg.patch_size * vision_cfg.patch_size * vision_cfg.in_channels;
    let model = VisionDragon::<Backend>::new(vision_cfg.clone(), &device);
    let mae = VisionMaeModel::new(
        model,
        mae_cfg,
        vision_cfg.num_eyes,
        vision_cfg.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 2;
    let images = Tensor::<Backend, 4>::random(
        [batch_size, vision_cfg.in_channels, vision_cfg.image_size, vision_cfg.image_size],
        Distribution::Default,
        &device,
    );
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = if mae.config.cross_view.enabled {
        let eyes = mae.num_eyes.max(1);
        let views = images
            .clone()
            .unsqueeze_dim::<5>(1)
            .repeat_dim(1, eyes);
        ImageNetBatch::new(images, None, Some(views), None, None, None, labels, None, None)
    } else {
        ImageNetBatch::new(images, None, None, None, None, None, labels, None, None)
    };

    let losses = mae.forward_losses(batch, 1, 1, false, false);
    let value = losses
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec")[0];
    assert!(value.is_finite(), "recon loss not finite: {value}");
}

#[test]
fn mae_recon_psnr_improves_on_toy_batch() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();

    let image_size: usize = 8;
    let patch_size: usize = 4;
    let grid = image_size.div_ceil(patch_size);
    let vision_config = VisionDragonConfig {
        image_size,
        patch_size,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        in_channels: 3,
        embed_dim: 32,
        steps: 1,
        n_head: 4,
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
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: grid,
        pos_max_width: grid,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
    };
    let mae_config = VisionMaeConfig {
        loss: VisionMaeLossConfig {
            recon: VisionReconLossConfig {
                weight: 1.0,
                mask_ratio: 0.5,
                hidden_dim: 64,
                ..VisionReconLossConfig::default()
            },
        },
        pyramid_levels: 1,
        artifact_every: 0,
        artifact_max_images: 0,
        artifact_max_views: 0,
        ..VisionMaeConfig::default()
    };
    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 1,
        backprop_steps: 1,
    };
    let recon_patch_dim =
        vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
    let mut mae = VisionMaeModel::new(
        model,
        mae_config,
        vision_config.num_eyes,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 2;
    let images = toy_images::<Backend>(batch_size, 3, image_size, image_size, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let steps = 1;
    let backprop_steps = 1;

    let initial_psnr = mae
        .forward_losses(batch.clone(), steps, backprop_steps, false, false)
        .recon_psnr
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("psnr vec")[0];

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionMaeModel<Backend>>();
    let lr = 0.02;
    for _ in 0..40 {
        let losses = mae.forward_losses(batch.clone(), steps, backprop_steps, false, false);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &mae);
        mae = optimizer.step(lr, mae, grads);
    }

    let final_psnr = mae
        .forward_losses(batch, steps, backprop_steps, false, false)
        .recon_psnr
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("psnr vec")[0];

    assert!(final_psnr.is_finite());
    assert!(final_psnr > initial_psnr);
    assert!(final_psnr > 24.0);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn identity_config_recon_loss_decreases() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    Backend::seed(&device, 1337);

    let config_path = vision_identity_tiny_path();
    let config = load_vision_training_config(&[config_path]).expect("load vision_identity_tiny");
    let vision_config = config.vision.build();
    let mae_config = match config.mode {
        VisionTrainingModeConfig::Mae(config) => config,
        other => panic!("expected mae config, got {other:?}"),
    };
    let rollout = resolve_vision_rollout(&config.training, vision_config.steps).expect("rollout");
    let recon_patch_dim = vision_config
        .patch_size
        .saturating_mul(vision_config.patch_size)
        .saturating_mul(vision_config.in_channels);
    let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
    let mut mae = VisionMaeModel::new(
        model,
        mae_config,
        vision_config.num_eyes,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = config.training.batch_size.max(1);
    let images = toy_images::<Backend>(
        batch_size,
        vision_config.in_channels,
        vision_config.image_size,
        vision_config.image_size,
        &device,
    );
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let steps = rollout.max_steps.max(1);
    let backprop_steps = rollout.backprop_steps.max(1);

    let initial_recon = mae
        .forward_losses(batch.clone(), steps, backprop_steps, false, false)
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec")[0];

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionMaeModel<Backend>>();
    let lr = config.optimizer.learning_rate;
    for _ in 0..30 {
        let losses = mae.forward_losses(batch.clone(), steps, backprop_steps, false, false);
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &mae);
        mae = optimizer.step(lr, mae, grads);
    }

    let final_recon = mae
        .forward_losses(batch, steps, backprop_steps, false, false)
        .recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec")[0];

    assert!(final_recon.is_finite());
    assert!(final_recon < initial_recon);
}

