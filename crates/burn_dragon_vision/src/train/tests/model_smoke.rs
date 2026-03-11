use super::*;

#[test]
fn patch_embed_supports_large_patches() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let vision_config = VisionDragonConfig {
        image_size: 160,
        patch_size: 64,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Dense,
        in_channels: 3,
        embed_dim: 16,
        steps: 1,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 16,
        projection_hidden_dim: 16,
        use_cls_token: true,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: true,
        latent_activation: VisionLatentActivation::default(),
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 3,
        pos_max_width: 3,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: Default::default(),
    };
    let model = VisionDragon::<Backend>::new(vision_config, &device);
    let images =
        Tensor::<Backend, 4>::random([1, 3, 160, 160], TensorDistribution::Default, &device);
    let patch = model.patch_embed_raw(images);
    assert_eq!(patch.grid.height, 3);
    assert_eq!(patch.grid.width, 3);
    assert_eq!(patch.tokens.shape().dims::<3>()[1], 9);
}
