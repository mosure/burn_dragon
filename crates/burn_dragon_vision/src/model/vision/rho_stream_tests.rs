use super::*;
#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
use burn::tensor::Distribution;
use burn::tensor::backend::Backend as BackendTrait;
#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
use burn_autodiff::Autodiff;
#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
use burn_cubecl::CubeBackend;
use burn_dragon_core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
use burn_ndarray::NdArray;
#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
use burn_wgpu::WgpuRuntime;

fn make_rho_stream_model<B: BackendTrait>(
    device: &B::Device,
    use_cls_token: bool,
    local_diagonals: bool,
    local_self: bool,
    wgpu_forward_kernel: bool,
    wgpu_rollout_fused: bool,
    mode_embeddings: bool,
) -> VisionDragon<B> {
    make_rho_stream_model_with_decay(
        device,
        use_cls_token,
        local_diagonals,
        local_self,
        wgpu_forward_kernel,
        wgpu_rollout_fused,
        mode_embeddings,
        0.9,
    )
}

fn make_rho_stream_model_with_decay<B: BackendTrait>(
    device: &B::Device,
    use_cls_token: bool,
    local_diagonals: bool,
    local_self: bool,
    wgpu_forward_kernel: bool,
    wgpu_rollout_fused: bool,
    mode_embeddings: bool,
    decay: f32,
) -> VisionDragon<B> {
    let config = VisionDragonConfig {
        image_size: 4,
        patch_size: 2,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Cellular,
        in_channels: 3,
        embed_dim: 4,
        steps: 1,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        projection_dim: 4,
        projection_hidden_dim: 8,
        use_cls_token,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: false,
        latent_activation: VisionLatentActivation::Identity,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: false,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: VisionRhoStreamConfig {
            enabled: true,
            local_radius: 1,
            local_diagonals,
            local_self,
            decay,
            mode_embeddings,
            wgpu_forward_kernel,
            wgpu_rollout_fused,
        },
    };
    VisionDragon::<B>::new(config, device)
}

fn make_pyramid_model<B: BackendTrait>(device: &B::Device) -> VisionDragon<B> {
    let config = VisionDragonConfig {
        image_size: 4,
        patch_size: 2,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Pyramid,
        in_channels: 3,
        embed_dim: 4,
        steps: 1,
        n_head: 1,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        projection_dim: 4,
        projection_hidden_dim: 8,
        use_cls_token: false,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: false,
        latent_activation: VisionLatentActivation::Identity,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: false,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: VisionTrmGraphConfig {
            enabled: true,
            coarse_stride: 2,
            hub_count: 2,
            rank: 2,
            value_dim: 4,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            decay: 0.9,
            hub_gates: true,
            grid_mismatch_policy: VisionTrmGridMismatchPolicy::Error,
        },
        rho_stream: Default::default(),
    };
    VisionDragon::<B>::new(config, device)
}

#[test]
fn cellular_routing_spec_reports_local_recurrent_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, true, true, false, false, true);
    let spec = model.backbone_routing_spec();

    assert!(spec.contains(StructuredRouteSpec::new(
        StructuredBankRole::Primary,
        StructuredBankRole::Primary,
        StructuredRouteOperation::Read,
        StructuredRoutePattern::Local,
    )));
    assert!(spec.contains(StructuredRouteSpec::new(
        StructuredBankRole::Primary,
        StructuredBankRole::Primary,
        StructuredRouteOperation::Write,
        StructuredRoutePattern::Identity,
    )));
}

#[test]
fn pyramid_routing_spec_reports_cross_bank_routes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let spec = model.backbone_routing_spec();

    assert!(spec.contains(StructuredRouteSpec::new(
        StructuredBankRole::Context,
        StructuredBankRole::Primary,
        StructuredRouteOperation::Read,
        StructuredRoutePattern::Broadcast,
    )));
    assert!(spec.contains(StructuredRouteSpec::new(
        StructuredBankRole::Primary,
        StructuredBankRole::Context,
        StructuredRouteOperation::Write,
        StructuredRoutePattern::Pool,
    )));
    assert!(spec.contains(StructuredRouteSpec::new(
        StructuredBankRole::Global,
        StructuredBankRole::Context,
        StructuredRouteOperation::Read,
        StructuredRoutePattern::Broadcast,
    )));
}

#[test]
fn rho_stream_attention_updates_only_active_token_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, true, true, false, false, false);
    let query = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![0.0, 0.0, 2.0, 0.0], [1, 1, 4, 1]),
        &device,
    );
    let value = Tensor::<Backend, 4>::from_data(
        TensorData::new(vec![0.0, 0.0, 3.0, 0.0], [1, 1, 4, 1]),
        &device,
    );
    let mut rho_state = None;

    let output = model.rho_stream_attention(query, value, &mut rho_state);
    let output_vec = output
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("output vec");
    assert!(output_vec.iter().all(|value| value.abs() < 1e-6));

    let rho_state = rho_state.expect("rho state should be initialized");
    assert_eq!(rho_state.shape().dims(), [1, 1, 4, 1, 1]);
    let state_vec = rho_state
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("state vec");
    assert_eq!(state_vec, vec![0.0, 0.0, 6.0, 0.0]);
}

#[test]
fn cellular_state_rollout_matches_existing_embed_rollout_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, true, false, true, false, false, true);
    let tokens = Tensor::<Backend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.5, -0.5, 0.0, 1.0, -0.5, 0.5, 0.25, -0.25, 0.75, -0.75, -0.5, 0.5, 1.0,
                0.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );

    let expected = model.forward_tokens_embed_steps_rollout_unbounded(tokens.clone(), 3, 3);
    let state = model.cellular_state_from_tokens(tokens);
    let state = model.forward_cellular_state_rollout_unbounded(state, 3, 3);
    let actual = model.forward_cellular_state_embed(&state);

    assert_eq!(
        actual
            .patch_tokens
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual patch vec"),
        expected
            .patch_tokens
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected patch vec")
    );
    assert_eq!(
        actual
            .cls_token
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual cls vec"),
        expected
            .cls_token
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected cls vec")
    );
}

#[test]
fn cellular_state_with_tokens_preserves_rho_and_replaces_dense_token_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, true, true, true, false, false, true);
    let tokens_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 0.0, 1.0, 0.5, -0.5, 1.0, 0.0, 0.25, 0.75, -0.25, -0.75, -1.0, 0.5, 0.25,
                0.0,
            ],
            [1, 4, 4],
        ),
        &device,
    );
    let tokens_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(
            vec![
                -0.25, 0.25, 0.5, -0.5, 0.75, 0.0, -0.75, 0.25, 1.0, -1.0, 0.5, -0.5, 0.0, 0.5,
                -0.25, 0.75,
            ],
            [1, 4, 4],
        ),
        &device,
    );

    let state = model.cellular_state_from_tokens(tokens_a);
    let state = model.forward_cellular_state_rollout_unbounded(state, 1, 1);
    let rho_before = state
        .rho
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho before");

    let observed = model.cellular_state_with_tokens(state, tokens_b.clone());
    let rho_after = observed
        .rho
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho after");
    assert_eq!(rho_after, rho_before);

    let expected = model.cellular_state_from_tokens(tokens_b);
    assert_eq!(
        observed
            .token_state
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("observed token state"),
        expected
            .token_state
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected token state")
    );
}

#[test]
fn cellular_state_rollout_mode_tracks_temporal_counters() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, true, true, false, false, true);
    let tokens = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);

    let state = model.cellular_state_from_tokens(tokens.clone());
    let refined = model.forward_cellular_state_rollout_mode_unbounded(
        state.clone(),
        2,
        2,
        StructuredStepMode::Refine,
    );
    assert_eq!(refined.temporal_position, 0);
    assert_eq!(refined.prediction_age, 0);

    let predicted = model.forward_cellular_state_rollout_mode_unbounded(
        state,
        2,
        2,
        StructuredStepMode::Predict,
    );
    assert_eq!(predicted.temporal_position, 1);
    assert_eq!(predicted.prediction_age, 1);

    let observed_state = model.cellular_state_with_tokens(predicted, tokens);
    let observed = model.forward_cellular_state_rollout_mode_unbounded(
        observed_state,
        2,
        2,
        StructuredStepMode::Observe,
    );
    assert_eq!(observed.temporal_position, 1);
    assert_eq!(observed.prediction_age, 0);
}

#[test]
fn cellular_state_rollout_mode_distinguishes_refine_from_predict_decay() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, true, true, false, false, false);
    let tokens = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let state = VisionCellularState {
        token_state: model.cellular_state_from_tokens(tokens).token_state,
        rho: Tensor::<Backend, 5>::ones([1, 1, 4, 4, 4], &device),
        temporal_position: 0,
        prediction_age: 0,
    };

    let refined = model.forward_cellular_state_rollout_mode_unbounded(
        state.clone(),
        1,
        1,
        StructuredStepMode::Refine,
    );
    let predicted = model.forward_cellular_state_rollout_mode_unbounded(
        state,
        1,
        1,
        StructuredStepMode::Predict,
    );

    let refine_vec = refined
        .rho
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("refine rho");
    let predict_vec = predicted
        .rho
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("predict rho");

    assert!(refine_vec.iter().all(|value| (*value - 1.0).abs() < 1e-6));
    assert!(predict_vec.iter().all(|value| (*value - 0.9).abs() < 1e-6));
}

#[test]
fn cellular_mode_embeddings_make_refine_and_predict_distinct_when_decay_is_one() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model_with_decay::<Backend>(
        &device, false, true, true, false, false, true, 1.0,
    );
    let tokens = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let state = model.cellular_state_from_tokens(tokens);

    let refined = model.forward_cellular_state_rollout_mode_unbounded(
        state.clone(),
        1,
        1,
        StructuredStepMode::Refine,
    );
    let predicted = model.forward_cellular_state_rollout_mode_unbounded(
        state,
        1,
        1,
        StructuredStepMode::Predict,
    );

    let refine_vec = refined
        .token_state
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("refine token state");
    let predict_vec = predicted
        .token_state
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("predict token state");
    assert_ne!(refine_vec, predict_vec);
}

#[test]
fn cellular_recurrent_mode_gates_make_reference_attention_mode_dependent_when_decay_is_one() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model_with_decay::<Backend>(
        &device, false, true, true, false, false, true, 1.0,
    );
    let query = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);
    let value = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);
    let state = Tensor::<Backend, 5>::ones([1, 1, 4, 4, 4], &device);
    let decay = model.scalar_cellular_decay(1.0, &device);

    let (refine_context, refine_rho) = model.rho_stream_attention_reference_with_state_decay_mode(
        query.clone(),
        value.clone(),
        state.clone(),
        decay.clone(),
        StructuredStepMode::Refine,
    );
    let (predict_context, predict_rho) = model
        .rho_stream_attention_reference_with_state_decay_mode(
            query,
            value,
            state,
            decay,
            StructuredStepMode::Predict,
        );

    assert_ne!(
        refine_context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("refine context"),
        predict_context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predict context")
    );
    assert_ne!(
        refine_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("refine rho"),
        predict_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predict rho")
    );
}

#[test]
fn cellular_without_mode_embeddings_refine_and_predict_match_when_decay_is_one() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model_with_decay::<Backend>(
        &device, false, true, true, false, false, false, 1.0,
    );
    let tokens = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let state = model.cellular_state_from_tokens(tokens);

    let refined = model.forward_cellular_state_rollout_mode_unbounded(
        state.clone(),
        1,
        1,
        StructuredStepMode::Refine,
    );
    let predicted = model.forward_cellular_state_rollout_mode_unbounded(
        state,
        1,
        1,
        StructuredStepMode::Predict,
    );

    assert_eq!(
        refined
            .token_state
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("refine token state"),
        predicted
            .token_state
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predict token state")
    );
}

#[test]
fn cellular_without_mode_embeddings_reference_attention_matches_across_modes_when_decay_is_one() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model_with_decay::<Backend>(
        &device, false, true, true, false, false, false, 1.0,
    );
    let query = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);
    let value = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);
    let state = Tensor::<Backend, 5>::ones([1, 1, 4, 4, 4], &device);
    let decay = model.scalar_cellular_decay(1.0, &device);

    let (refine_context, refine_rho) = model.rho_stream_attention_reference_with_state_decay_mode(
        query.clone(),
        value.clone(),
        state.clone(),
        decay.clone(),
        StructuredStepMode::Refine,
    );
    let (predict_context, predict_rho) = model
        .rho_stream_attention_reference_with_state_decay_mode(
            query,
            value,
            state,
            decay,
            StructuredStepMode::Predict,
        );

    assert_eq!(
        refine_context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("refine context"),
        predict_context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predict context")
    );
    assert_eq!(
        refine_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("refine rho"),
        predict_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("predict rho")
    );
}

#[test]
fn cellular_predict_decay_uses_alibi_head_slopes() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let config = VisionDragonConfig {
        image_size: 4,
        patch_size: 2,
        patch_embed_mode: VisionPatchEmbedMode::default(),
        backbone: VisionBackboneKind::Cellular,
        in_channels: 3,
        embed_dim: 4,
        steps: 1,
        n_head: 2,
        mlp_internal_dim_multiplier: 1,
        dropout: 0.0,
        projection_dim: 4,
        projection_hidden_dim: 8,
        use_cls_token: false,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: false,
        latent_activation: VisionLatentActivation::Identity,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        use_alibi: true,
        fused_kernels: FusedKernelConfig::default(),
        mhc: ManifoldHyperConnectionsConfig::default(),
        trm_graph: Default::default(),
        rho_stream: VisionRhoStreamConfig {
            enabled: true,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            decay: 0.9,
            mode_embeddings: false,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
        },
    };
    let model = VisionDragon::<Backend>::new(config, &device);

    let refine_decay = model
        .cellular_decay_by_mode(StructuredStepMode::Refine, &device)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("refine decay");
    let predict_decay = model
        .cellular_decay_by_mode(StructuredStepMode::Predict, &device)
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("predict decay");
    let expected: Vec<f32> = burn_dragon_core::kernel::linear_attention::default_alibi_slopes(2)
        .into_iter()
        .map(|s| (-s).exp())
        .collect();

    assert_eq!(refine_decay, vec![1.0, 1.0]);
    assert_eq!(predict_decay.len(), expected.len());
    for (actual, expected) in predict_decay.iter().zip(expected.iter()) {
        assert!((actual - expected).abs() < 1e-6);
    }
    assert_ne!(predict_decay[0], predict_decay[1]);
}

#[test]
fn pyramid_state_uses_banked_rho_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    assert_eq!(state.patch_rho().shape().dims(), [1, 2, 4, 2, 2]);
    assert_eq!(state.coarse_rho().shape().dims(), [1, 2, 4, 1, 1]);
    assert_eq!(state.hub_rho().shape().dims(), [1, 2, 2, 4]);

    let predicted = model.forward_pyramid_state_rollout_mode_unbounded(
        state.clone(),
        1,
        1,
        StructuredStepMode::Predict,
    );
    assert_eq!(predicted.temporal_position, 1);
    assert_eq!(predicted.prediction_age, 1);
    assert_eq!(predicted.patch_rho().shape().dims(), [1, 2, 4, 2, 2]);
    assert_eq!(predicted.coarse_rho().shape().dims(), [1, 2, 4, 1, 1]);
    assert_eq!(predicted.hub_rho().shape().dims(), [1, 2, 2, 4]);

    let observed = model.forward_pyramid_state_rollout_mode_unbounded(
        predicted,
        1,
        1,
        StructuredStepMode::Observe,
    );
    assert_eq!(observed.temporal_position, 1);
    assert_eq!(observed.prediction_age, 0);
}

#[test]
#[should_panic(expected = "cellular rho state shape")]
fn cellular_state_with_tokens_rejects_patch_count_mismatch() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, true, true, false, false, true);
    let tokens_a = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let tokens_b = Tensor::<Backend, 3>::zeros([1, 1, 4], &device);

    let state = model.cellular_state_from_tokens(tokens_a);
    let _ = model.cellular_state_with_tokens(state, tokens_b);
}

#[test]
fn rope_positional_encoding_rotates_patch_tokens() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let encoding = SpatialPositionalEncoding::<Backend>::new(
        SpatialPositionalEncodingKind::Rope,
        2,
        2,
        8,
        &device,
    );
    let tokens = Tensor::<Backend, 3>::from_data(
        TensorData::new(
            vec![
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
            ],
            [1, 4, 8],
        ),
        &device,
    );

    let rotated = encoding.add_position(
        tokens.clone(),
        PatchGrid {
            height: 2,
            width: 2,
        },
    );
    let input = tokens
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("input vec");
    let output = rotated
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rotated vec");
    assert_eq!(input.len(), output.len());
    let total_delta: f32 = input
        .iter()
        .zip(output.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    assert!(total_delta > 1e-3, "rope should rotate at least one token");
    let first_token_delta: f32 = input[..8]
        .iter()
        .zip(output[..8].iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .sum();
    assert!(
        first_token_delta < 1e-6,
        "origin token should stay unchanged"
    );
}

#[test]
fn rho_stream_local_read_is_neighborhood_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, false, false, true, false, false, true);
    let rho_state = Tensor::<Backend, 5>::from_data(
        TensorData::new(vec![1.0, 0.0, 0.0, 0.0], [1, 1, 4, 1, 1]),
        &device,
    );
    let query = Tensor::<Backend, 4>::ones([1, 1, 4, 1], &device);

    let context = model.rho_stream_local_read(
        rho_state,
        query,
        PatchGrid {
            height: 2,
            width: 2,
        },
    );
    let context_vec = context
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("context vec");
    assert_eq!(context_vec, vec![1.0, 1.0, 1.0, 0.0]);
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
fn assert_close<const D: usize>(
    lhs: Tensor<WgpuBackend, D>,
    rhs: Tensor<WgpuBackend, D>,
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
    let mut max_diff = 0.0_f32;
    let mut max_tol = 0.0_f32;

    for (lhs_value, rhs_value) in lhs.iter().zip(rhs.iter()) {
        let diff = (*lhs_value - *rhs_value).abs();
        let tol = atol + rtol * rhs_value.abs();
        if diff > max_diff {
            max_diff = diff;
            max_tol = tol;
        }
    }

    assert!(
        max_diff <= max_tol,
        "max difference {max_diff} exceeds tolerance {max_tol}"
    );
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn rho_stream_wgpu_forward_matches_reference_contract_with_cls() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 2026);

    let reference_model =
        make_rho_stream_model::<WgpuBackend>(&device, true, false, true, false, false, true);
    let fused_model =
        make_rho_stream_model::<WgpuBackend>(&device, true, false, true, true, true, true);
    assert!(supports_local_grid_rho_backend::<WgpuBackend>());

    let query =
        Tensor::<WgpuBackend, 4>::random([2, 1, 5, 4], Distribution::Normal(0.0, 1.0), &device);
    let value =
        Tensor::<WgpuBackend, 4>::random([2, 1, 5, 4], Distribution::Normal(0.0, 1.0), &device);
    let state =
        Tensor::<WgpuBackend, 5>::random([2, 1, 4, 4, 4], Distribution::Normal(0.0, 1.0), &device);

    let (reference_context, reference_rho) = reference_model
        .rho_stream_attention_reference_with_state(query.clone(), value.clone(), state.clone());
    let mut fused_rho = Some(state);
    let fused_context = fused_model.rho_stream_attention_fused(query, value, &mut fused_rho);
    let fused_rho = fused_rho.expect("fused rho state");

    assert_close(fused_context, reference_context, 4e-4, 4e-4);
    assert_close(fused_rho, reference_rho, 4e-4, 4e-4);
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn rho_stream_wgpu_forward_matches_reference_over_multiple_calls_with_cls() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 4_242);

    let reference_model =
        make_rho_stream_model::<WgpuBackend>(&device, true, false, true, false, false, true);
    let fused_model =
        make_rho_stream_model::<WgpuBackend>(&device, true, false, true, true, true, true);

    let mut reference_rho = Some(Tensor::<WgpuBackend, 5>::zeros([2, 1, 4, 4, 4], &device));
    let mut fused_rho = reference_rho.clone();

    for _ in 0..3 {
        let query =
            Tensor::<WgpuBackend, 4>::random([2, 1, 5, 4], Distribution::Normal(0.0, 1.0), &device);
        let value =
            Tensor::<WgpuBackend, 4>::random([2, 1, 5, 4], Distribution::Normal(0.0, 1.0), &device);

        let reference_context = reference_model.rho_stream_attention_reference(
            query.clone(),
            value.clone(),
            &mut reference_rho,
        );
        let fused_context = fused_model.rho_stream_attention_fused(query, value, &mut fused_rho);

        assert_close(fused_context, reference_context, 4e-4, 4e-4);
        assert_close(
            fused_rho.clone().expect("fused rho"),
            reference_rho.clone().expect("reference rho"),
            4e-4,
            4e-4,
        );
    }
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn rho_stream_wgpu_autodiff_matches_reference_backward_contract() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 9_901);

    let reference_model = make_rho_stream_model::<WgpuAutodiffBackend>(
        &device, true, false, true, false, false, true,
    );
    let fused_model =
        make_rho_stream_model::<WgpuAutodiffBackend>(&device, true, false, true, true, true, true);

    let reference_query = Tensor::<WgpuAutodiffBackend, 4>::random(
        [2, 1, 5, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();
    let reference_value = Tensor::<WgpuAutodiffBackend, 4>::random(
        [2, 1, 5, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();
    let reference_state = Tensor::<WgpuAutodiffBackend, 5>::random(
        [2, 1, 4, 4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();

    let (reference_context, reference_rho) = reference_model
        .rho_stream_attention_reference_with_state(
            reference_query.clone(),
            reference_value.clone(),
            reference_state.clone(),
        );
    let reference_loss = reference_context.tanh().powf_scalar(2.0).mean()
        + reference_rho.tanh().powf_scalar(2.0).mean();
    let reference_loss_value = reference_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let reference_grads = reference_loss.backward();
    let reference_query_grad = reference_query
        .grad(&reference_grads)
        .expect("reference query grad");
    let reference_value_grad = reference_value
        .grad(&reference_grads)
        .expect("reference value grad");
    let reference_state_grad = reference_state
        .grad(&reference_grads)
        .expect("reference state grad");

    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 9_901);
    let fused_query = Tensor::<WgpuAutodiffBackend, 4>::random(
        [2, 1, 5, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();
    let fused_value = Tensor::<WgpuAutodiffBackend, 4>::random(
        [2, 1, 5, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();
    let fused_state_input = Tensor::<WgpuAutodiffBackend, 5>::random(
        [2, 1, 4, 4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    )
    .require_grad();
    let mut fused_state = Some(fused_state_input.clone());
    let fused_context = fused_model.rho_stream_attention_fused(
        fused_query.clone(),
        fused_value.clone(),
        &mut fused_state,
    );
    let fused_rho = fused_state.expect("fused rho");
    let fused_loss =
        fused_context.tanh().powf_scalar(2.0).mean() + fused_rho.tanh().powf_scalar(2.0).mean();
    let fused_loss_value = fused_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused loss")[0];
    let fused_grads = fused_loss.backward();
    let fused_query_grad = fused_query.grad(&fused_grads).expect("fused query grad");
    let fused_value_grad = fused_value.grad(&fused_grads).expect("fused value grad");
    let fused_state_grad = fused_state_input
        .grad(&fused_grads)
        .expect("fused state grad");

    assert!(reference_loss_value.is_finite());
    assert!(fused_loss_value.is_finite());
    assert_close(reference_query_grad, fused_query_grad, 1e-1, 1e-1);
    assert_close(reference_value_grad, fused_value_grad, 1e-1, 1e-1);
    assert_close(reference_state_grad, fused_state_grad, 1e-1, 1e-1);
}
