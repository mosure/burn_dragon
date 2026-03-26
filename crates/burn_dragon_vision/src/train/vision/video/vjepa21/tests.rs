use super::*;
use burn::optim::{AdamWConfig, GradientsParams, LearningRate};
use burn_autodiff::Autodiff;
use burn_dragon_core::FusedKernelConfig;
use burn_ndarray::NdArray;

fn make_config(
    backbone: crate::VisionBackboneKind,
) -> (VisionDragonConfig, VisionVideoLejepaConfig) {
    let mut vision = VisionDragonConfig {
        image_size: 16,
        patch_size: 4,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone,
        in_channels: 3,
        embed_dim: 24,
        steps: 2,
        n_head: 4,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 12,
        projection_hidden_dim: 24,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        normalization: DragonNormConfig::default(),
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Rope,
        pos_max_height: 4,
        pos_max_width: 4,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: false,
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
    if matches!(backbone, crate::VisionBackboneKind::Pyramid) {
        vision.trm_graph.enabled = true;
        vision.trm_graph.coarse_stride = 2;
        vision.trm_graph.hub_count = 2;
        vision.trm_graph.rank = 4;
        vision.trm_graph.value_dim = 12;
        vision.trm_graph.local_radius = 1;
        vision.trm_graph.local_diagonals = true;
        vision.trm_graph.local_self = true;
        vision.trm_graph.decay = 0.9;
    }
    if matches!(backbone, crate::VisionBackboneKind::Cellular) {
        vision.rho_stream.enabled = true;
    }
    let mut video = VisionVideoLejepaConfig::default();
    video.paradigm = VisionVideoParadigmKind::Vjepa21;
    video.frame_stride = 1;
    video.vjepa21.clip_frames = 6;
    video.vjepa21.observe_steps = 1;
    video.vjepa21.observe_backprop_steps = 1;
    video.vjepa21.predictor_hidden_dim = 32;
    video.vjepa21.checkpoint_depths = vec![1, 2];
    video.vjepa21.loss.context_weight = 0.5;
    video.vjepa21.loss.predict_all = true;
    video.vjepa21.loss.weight_distance_loss = true;
    video.vjepa21.mask.num_blocks = 2;
    video.vjepa21.mask.temporal_scale_min = 0.5;
    video.vjepa21.mask.temporal_scale_max = 1.0;
    (vision, video)
}

const TOY_NUM_CLASSES: usize = 10;

fn toy_batch<B: BackendTrait>(device: &B::Device) -> VideoClipBatch<B> {
    let batch = 2;
    let clip_len = 6;
    let channels = 3;
    let size = 16;
    let mut data = vec![0.0_f32; batch * clip_len * channels * size * size];
    for b in 0..batch {
        for t in 0..clip_len {
            let x = (t + b) % (size - 4);
            let y = (2 * t + b) % (size - 4);
            for c in 0..channels {
                for dy in 0..4 {
                    for dx in 0..4 {
                        let idx = ((((b * clip_len + t) * channels + c) * size + (y + dy)) * size)
                            + x
                            + dx;
                        data[idx] = 1.0;
                    }
                }
            }
        }
    }
    let clip_frames = Tensor::<B, 5>::from_data(
        TensorData::new(data, [batch, clip_len, channels, size, size]),
        device,
    );
    let labels = Tensor::<B, 1, Int>::zeros([batch], device);
    VideoClipBatch::new(clip_frames, labels, clip_len, 0)
}

#[test]
fn vjepa21_mask_sampler_keeps_targets_and_context() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (_, video) = make_config(crate::VisionBackboneKind::Dense);
    let masks = sample_mask_batch::<Backend>(2, 6, 16, &video.vjepa21, &device);
    let visible = masks
        .visible
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("visible sum")[0];
    let target = masks
        .target
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("target sum")[0];
    assert!(visible > 0.0);
    assert!(target > 0.0);
}

#[test]
fn vjepa21_forward_losses_are_finite_for_all_backbones() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    for backbone in [
        crate::VisionBackboneKind::Dense,
        crate::VisionBackboneKind::Pyramid,
        crate::VisionBackboneKind::Cellular,
    ] {
        let (vision, video) = make_config(backbone);
        let model = VisionVideoVjepa21Model::<Backend>::new(
            VisionDragon::<Backend>::new(vision.clone(), &device),
            video,
            &vision,
            TOY_NUM_CLASSES,
            &device,
        );
        let losses = model.forward_losses(toy_batch::<Backend>(&device));
        let total = losses
            .total
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("total")[0];
        assert!(total.is_finite());
        assert!(total > 0.0);
    }
}

#[test]
fn vjepa21_validation_artifacts_include_mask_and_state_maps() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, mut video) = make_config(crate::VisionBackboneKind::Dense);
    video.artifact_every = 1;
    video.artifact_max_images = 1;
    video.artifact_upscale = 2;
    video.artifact_output = VisionArtifactOutputMode::Avi;
    video.loss.debug_recon_weight = 1.0;
    let model = VisionVideoVjepa21Model::<Backend>::new(
        VisionDragon::<Backend>::new(vision.clone(), &device),
        video,
        &vision,
        TOY_NUM_CLASSES,
        &device,
    );
    let losses = model.forward_losses(toy_batch::<Backend>(&device).with_capture_artifacts(true));
    let artifacts = losses.artifacts.expect("artifacts");

    let frames = artifacts.frames.expect("frames");
    assert_eq!(frames.shape().dims::<5>(), [1, 6, 3, 16, 16]);
    assert!(artifacts.posterior_patch_norms_steps.is_some());
    assert!(artifacts.pca_rgb_steps.is_some());
    assert!(artifacts.debug_recon_frames.is_some());
    assert!(artifacts.posterior_pca_rgb_steps.is_none());
    assert!(artifacts.debug_pca_rgb_steps.is_none());
    assert_eq!(artifacts.artifact_scale, 2);
    assert_eq!(
        artifacts.legend.expect("legend"),
        vec![
            "reference_frame".to_string(),
            "target_mask".to_string(),
            "predictor_state_pca_rgb".to_string(),
            "decoded_predictor_output".to_string(),
        ]
    );
}

#[test]
fn vjepa21_dense_prediction_improves_on_toy_batch() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_config(crate::VisionBackboneKind::Dense);
    let mut model = VisionVideoVjepa21Model::<Backend>::new(
        VisionDragon::<Backend>::new(vision.clone(), &device),
        video,
        &vision,
        TOY_NUM_CLASSES,
        &device,
    );
    let batch = toy_batch::<Backend>(&device);
    let initial = model
        .forward_losses(batch.clone())
        .masked
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("initial")[0];
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoVjepa21Model<Backend>>();
    let lr: LearningRate = 1.0e-2;
    for _ in 0..30 {
        let losses = model.forward_losses(batch.clone());
        let grads = GradientsParams::from_grads(losses.total.clone().backward(), &model);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);
    }
    let final_loss = model
        .forward_losses(batch)
        .masked
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("final")[0];
    assert!(final_loss.is_finite());
    assert!(final_loss < initial);
}
