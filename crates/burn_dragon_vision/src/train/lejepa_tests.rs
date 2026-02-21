use crate::config::VisionAugmentationConfig;
use crate::train::prelude::*;
use crate::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionLatentActivation,
    VisionPatchEmbedMode,
};
use burn::optim::Optimizer;
use burn::tensor::Distribution;
use burn_autodiff::Autodiff;
use burn_dragon_core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
use burn_ndarray::NdArray;

#[test]
fn lejepa_invariance_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
    let loss = lejepa_invariance_loss(proj);
    let value = loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn lejepa_sigreg_loss_is_finite() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let config = VisionLejepaConfig::default();

    let proj = Tensor::<Backend, 3>::random([2, 4, 8], Distribution::Default, &device);
    let loss = lejepa_sigreg_loss(proj, &config.loss.lejepa);
    let value = loss
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn patchify_roundtrip() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let batch = 1;
    let channels = 3;
    let height = 4;
    let width = 4;
    let patch_size = 2;
    let total = batch * channels * height * width;
    let data: Vec<f32> = (0..total).map(|v| v as f32).collect();

    let images = Tensor::<Backend, 4>::from_data(
        TensorData::new(data.clone(), [batch, channels, height, width]),
        &device,
    );
    let patches = patchify(images.clone(), patch_size);
    let [patch_batch, tokens, patch_dim] = patches.shape().dims::<3>();
    assert_eq!(patch_batch, batch);
    assert_eq!(tokens, (height / patch_size) * (width / patch_size));
    assert_eq!(patch_dim, channels * patch_size * patch_size);

    let recon = unpatchify(patches, patch_size, height, width, channels);
    let out = recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec");
    assert_eq!(data, out);
}

#[test]
fn patchify_roundtrip_with_padding() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();

    let batch = 1;
    let channels = 3;
    let height = 5;
    let width = 6;
    let patch_size = 4;
    let total = batch * channels * height * width;
    let data: Vec<f32> = (0..total).map(|v| v as f32).collect();

    let images = Tensor::<Backend, 4>::from_data(
        TensorData::new(data.clone(), [batch, channels, height, width]),
        &device,
    );
    let patches = patchify(images.clone(), patch_size);
    let [patch_batch, tokens, patch_dim] = patches.shape().dims::<3>();
    let grid_h = height.div_ceil(patch_size);
    let grid_w = width.div_ceil(patch_size);
    assert_eq!(patch_batch, batch);
    assert_eq!(tokens, grid_h * grid_w);
    assert_eq!(patch_dim, channels * patch_size * patch_size);

    let recon = unpatchify(patches, patch_size, height, width, channels);
    let out = recon
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("recon vec");
    assert_eq!(data, out);
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

#[test]
fn lejepa_recon_psnr_improves_on_toy_batch() {
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
        trm_graph: Default::default(),
    };
    let mut lejepa_config = VisionLejepaConfig {
        views: 1,
        global_views: 0,
        local_views: 0,
        artifact_every: 0,
        artifact_max_images: 0,
        artifact_max_views: 0,
        ..Default::default()
    };
    lejepa_config.loss.lejepa.enabled = false;
    lejepa_config.loss.recon.weight = 1.0;
    lejepa_config.loss.recon.mask_ratio = 0.5;
    lejepa_config.loss.recon.hidden_dim = 64;

    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 1,
        backprop_steps: 1,
    };
    let recon_patch_dim =
        vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    let normalize_std = VisionAugmentationConfig::default().normalize_std;
    let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
    let mut lejepa = VisionLejepaModel::new(
        model,
        lejepa_config,
        VisionLejepaInit {
            embed_dim: vision_config.embed_dim,
            num_classes: 1,
            rollout,
            recon: VisionReconstructionInit {
                patch_dim: recon_patch_dim,
                normalize_std,
                patch_size: vision_config.patch_size,
                in_channels: vision_config.in_channels,
            },
        },
        &device,
    );

    let batch_size = 2;
    let images = toy_images::<Backend>(batch_size, 3, image_size, image_size, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let steps = 1;
    let backprop_steps = 1;

    let initial_psnr = lejepa
        .forward_losses(batch.clone(), steps, backprop_steps, false, false, false)
        .recon_psnr_full
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("psnr vec")[0];

    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionLejepaModel<Backend>>();
    let lr = 0.02;
    for _ in 0..40 {
        let losses =
            lejepa.forward_losses(batch.clone(), steps, backprop_steps, false, false, false);
        let total = losses.total.clone() + losses.probe_loss.clone();
        let grads = GradientsParams::from_grads(total.backward(), &lejepa);
        lejepa = optimizer.step(lr, lejepa, grads);
    }

    let final_psnr = lejepa
        .forward_losses(batch, steps, backprop_steps, false, false, false)
        .recon_psnr_full
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("psnr vec")[0];

    assert!(final_psnr.is_finite());
    assert!(final_psnr > initial_psnr);
    assert!(final_psnr > 24.0);
}
