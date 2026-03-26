use super::*;
use crate::{
    SpatialPositionalEncodingKind, VisionAttentionMode, VisionBackboneKind, VisionDragonConfig,
    VisionLatentActivation, VisionPatchEmbedMode, VisionRhoStreamConfig,
};
use burn_ndarray::NdArray;

fn toy_cifar_batch<B: BackendTrait>(batch: usize, device: &B::Device) -> VisionRacBatch<B> {
    let height = 32usize;
    let width = 32usize;
    let channels = 3usize;
    let mut data = Vec::with_capacity(batch * channels * height * width);
    let denom_w = (width - 1).max(1) as f32;
    let denom_h = (height - 1).max(1) as f32;
    for sample in 0..batch {
        for channel in 0..channels {
            for y in 0..height {
                let gy = y as f32 / denom_h;
                for x in 0..width {
                    let gx = x as f32 / denom_w;
                    let checker = ((x / 2 + y / 3 + sample + channel) % 2) as f32;
                    let value = match channel {
                        0 => gx,
                        1 => gy,
                        _ => 0.45 * gx + 0.35 * gy + 0.2 * checker,
                    };
                    data.push(value);
                }
            }
        }
    }
    let images = Tensor::<B, 4>::from_data(TensorData::new(data, [batch, 3, 32, 32]), device);
    let labels = Tensor::<B, 1, Int>::from_data(
        TensorData::new((0..batch).map(|idx| (idx % 10) as i64).collect(), [batch]),
        device,
    );
    VisionRacBatch::new(images, labels, None, None, None)
}

fn make_rac_model<B: BackendTrait>(device: &B::Device) -> VisionRacModel<B> {
    let mut rho_stream = VisionRhoStreamConfig::default();
    rho_stream.enabled = true;
    rho_stream.local_radius = 1;
    rho_stream.local_diagonals = true;
    rho_stream.local_self = true;
    rho_stream.decay = 0.985;
    rho_stream.mode_embeddings = true;
    rho_stream.wgpu_forward_kernel = false;
    rho_stream.wgpu_rollout_fused = false;

    let vision = VisionDragonConfig {
        image_size: 32,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::Conv,
        backbone: VisionBackboneKind::Cellular,
        in_channels: 3,
        embed_dim: 48,
        steps: 3,
        n_head: 3,
        mlp_internal_dim_multiplier: 3,
        dropout: 0.0,
        projection_dim: 32,
        projection_hidden_dim: 64,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        normalization: burn_dragon_core::DragonNormConfig::default(),
        latent_activation: VisionLatentActivation::Relu,
        pos_encoding: SpatialPositionalEncodingKind::Rope,
        pos_max_height: 8,
        pos_max_width: 8,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: Default::default(),
        mhc: Default::default(),
        trm_graph: Default::default(),
        rho_stream,
    };
    let mut rac = VisionRacConfig::default();
    rac.sample_steps = 3;
    rac.random_time_grid = false;
    rac.velocity_hidden_dim = 64;
    rac.artifact_output = VisionArtifactOutputMode::Images;
    rac.artifact_every = 1;
    rac.artifact_max_images = 2;
    rac.artifact_upscale = 2;
    rac.memory.observe_steps = 2;
    rac.memory.backprop_steps = 2;
    rac.memory.flow_backprop_steps = Some(2);

    let model = VisionDragon::<B>::new(vision.clone(), device);
    VisionRacModel::new(model, rac, &vision, 10, device)
}

#[test]
fn rac_forward_losses_emit_artifacts_with_expected_panels() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rac_model::<Backend>(&device);
    let batch = toy_cifar_batch::<Backend>(2, &device);

    let losses = model.forward_losses(batch, true);
    let artifacts = losses
        .artifacts
        .expect("artifact capture should be enabled for RAC");
    let legend = artifacts.legend.expect("legend should be present");

    assert_eq!(
        legend,
        vec![
            "forward_state_x_t".to_string(),
            "forward_velocity_patch_norm_v_t".to_string(),
            "forward_patch_pca_z_t".to_string(),
            "reverse_velocity_patch_norm_v_t".to_string(),
            "reverse_patch_pca_z_t".to_string(),
            "reverse_state_x_t_matched".to_string(),
            "roundtrip_state_x_t".to_string(),
            "forward_reference_error_patch_norm".to_string(),
        ]
    );
    assert!(
        artifacts.frames.is_some(),
        "expected solver trajectory frames"
    );
    assert!(
        artifacts.posterior_patch_norms_steps.is_some(),
        "expected velocity heatmaps"
    );
    assert!(
        artifacts.posterior_pca_rgb_steps.is_some(),
        "expected solver PCA maps"
    );
    assert!(
        artifacts.patch_norms_steps.is_some(),
        "expected reverse velocity heatmaps"
    );
    assert!(
        artifacts.pca_rgb_steps.is_some(),
        "expected reverse PCA maps"
    );
    assert!(
        artifacts.debug_recon_frames.is_some(),
        "expected reverse matched frames"
    );
    assert!(
        artifacts.aux_frames.is_some(),
        "expected roundtrip trajectory frames"
    );
    assert!(
        artifacts.debug_patch_norms_steps.is_some(),
        "expected trajectory error maps"
    );
    let sidecar = artifacts
        .sidecar_json
        .expect("expected RAC artifact sidecar json");
    assert!(
        sidecar.contains("\"flow_backprop_steps\": 2"),
        "expected sidecar memory mode to record flow_backprop_steps"
    );
}
