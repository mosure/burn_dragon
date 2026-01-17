use crate::train::prelude::*;
use burn_dragon_hatchling_core::{
    FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode, VisionMaeLossConfig,
    VisionReconLossConfig,
};
#[cfg(not(target_arch = "wasm32"))]
use burn_dragon_hatchling_core::{VisionTrainingModeConfig, load_vision_training_config};
use burn::optim::Optimizer;
use burn::tensor::Distribution;
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
#[cfg(not(target_arch = "wasm32"))]
use std::path::PathBuf;

#[test]
fn mae_pyramid_recon_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let vision_config = VisionDragonHatchlingConfig {
        image_size: 8,
        patch_size: 4,
        in_channels: 3,
        embed_dim: 8,
        steps: 2,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 8,
        projection_hidden_dim: 16,
        use_cls_token: true,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        fused_kernels: FusedKernelConfig::default(),
    };
    let recon_patch_dim = vision_config.patch_size
        * vision_config.patch_size
        * vision_config.in_channels;
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
    let model = VisionDragonHatchling::<Backend>::new(vision_config, &device);
    let mae = VisionMaeModel::new(model, mae_config, 8, rollout, recon_patch_dim, &device);

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
    Tensor::<B, 4>::from_data(TensorData::new(data, [batch, channels, height, width]), device)
}

#[cfg(not(target_arch = "wasm32"))]
fn vision_identity_tiny_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest_dir
            .join("..")
            .join("..")
            .join("config")
            .join("vision_identity_tiny.toml"),
        manifest_dir
            .join("..")
            .join("config")
            .join("vision_identity_tiny.toml"),
        manifest_dir.join("config").join("vision_identity_tiny.toml"),
    ];
    for candidate in &candidates {
        if candidate.exists() {
            return candidate.clone();
        }
    }
    candidates[0].clone()
}

#[test]
fn mae_recon_psnr_improves_on_toy_batch() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();

    let image_size = 8;
    let patch_size = 4;
    let grid = (image_size + patch_size - 1) / patch_size;
    let vision_config = VisionDragonHatchlingConfig {
        image_size,
        patch_size,
        in_channels: 3,
        embed_dim: 32,
        steps: 1,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 32,
        projection_hidden_dim: 64,
        use_cls_token: true,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: grid,
        pos_max_width: grid,
        attention_mode: VisionAttentionMode::RowL1,
        fused_kernels: FusedKernelConfig::default(),
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
    let model = VisionDragonHatchling::<Backend>::new(vision_config.clone(), &device);
    let mut mae = VisionMaeModel::new(
        model,
        mae_config,
        vision_config.embed_dim,
        rollout,
        recon_patch_dim,
        &device,
    );

    let batch_size = 2;
    let images = toy_images::<Backend>(batch_size, 3, image_size, image_size, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
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
    let config =
        load_vision_training_config(&[config_path]).expect("load vision_identity_tiny");
    let vision_config = config.vision.build();
    let mae_config = match config.mode {
        VisionTrainingModeConfig::Mae(config) => config,
        other => panic!("expected mae config, got {other:?}"),
    };
    let rollout =
        resolve_vision_rollout(&config.training, vision_config.steps).expect("rollout");
    let recon_patch_dim = vision_config
        .patch_size
        .saturating_mul(vision_config.patch_size)
        .saturating_mul(vision_config.in_channels);
    let model = VisionDragonHatchling::<Backend>::new(vision_config.clone(), &device);
    let mut mae = VisionMaeModel::new(
        model,
        mae_config,
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
    let batch = ImageNetBatch::new(images, None, None, None, None, labels, None, None);
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
