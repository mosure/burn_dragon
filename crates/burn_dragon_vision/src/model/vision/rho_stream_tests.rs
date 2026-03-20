use super::pyramid_ops::{
    CompiledLocalBridgeProjectionPairPlan, CompiledStructuredDenseUpdatePairPlan,
};
use super::*;
#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
use burn::optim::{AdamWConfig, GradientsParams, LearningRate, Optimizer};
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
#[derive(Clone, Copy)]
struct RhoStreamTestConfig {
    use_cls_token: bool,
    local_diagonals: bool,
    local_self: bool,
    wgpu_forward_kernel: bool,
    wgpu_rollout_fused: bool,
    mode_embeddings: bool,
    decay: f32,
}

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
        RhoStreamTestConfig {
            use_cls_token,
            local_diagonals,
            local_self,
            wgpu_forward_kernel,
            wgpu_rollout_fused,
            mode_embeddings,
            decay: 0.9,
        },
    )
}

fn make_rho_stream_model_with_decay<B: BackendTrait>(
    device: &B::Device,
    config: RhoStreamTestConfig,
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
        use_cls_token: config.use_cls_token,
        cls_sync_alpha: 0.0,
        num_eyes: 1,
        cross_eye_steps: 0,
        token_state_norm: false,
        normalization: burn_dragon_core::DragonNormConfig::default(),
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
            local_diagonals: config.local_diagonals,
            local_self: config.local_self,
            decay: config.decay,
            mode_embeddings: config.mode_embeddings,
            wgpu_forward_kernel: config.wgpu_forward_kernel,
            wgpu_rollout_fused: config.wgpu_rollout_fused,
        },
    };
    VisionDragon::<B>::new(config, device)
}

fn make_pyramid_config() -> VisionDragonConfig {
    VisionDragonConfig {
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
        normalization: burn_dragon_core::DragonNormConfig::default(),
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
            patch_rank: None,
            coarse_rank: None,
            global_rank: None,
            value_dim: 4,
            local_radius: 1,
            local_diagonals: true,
            local_self: true,
            coarse_local_radius: None,
            coarse_local_diagonals: None,
            coarse_local_self: None,
            predict_coarse_substeps: 1,
            predict_substep_kind: VisionTrmPredictSubstepKind::CoarseOnly,
            cls_readout: VisionTrmClsReadoutKind::PatchMean,
            decay: 0.9,
            hub_gates: true,
            bank_schedule: VisionTrmGraphBankScheduleConfig::default(),
            grid_mismatch_policy: VisionTrmGridMismatchPolicy::Error,
        },
        rho_stream: Default::default(),
    }
}

fn make_pyramid_model<B: BackendTrait>(device: &B::Device) -> VisionDragon<B> {
    make_pyramid_model_with_kernel(device, false)
}

fn make_pyramid_model_with_kernel<B: BackendTrait>(
    device: &B::Device,
    kernel_enabled: bool,
) -> VisionDragon<B> {
    let mut config = make_pyramid_config();
    config.fused_kernels = FusedKernelConfig {
        enabled: kernel_enabled,
        ..FusedKernelConfig::default()
    };
    VisionDragon::<B>::new(config, device)
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
fn make_stage_aware_pyramid_config() -> VisionDragonConfig {
    let mut config = make_pyramid_config();
    config.image_size = 8;
    config.patch_size = 2;
    config.embed_dim = 8;
    config.projection_dim = 8;
    config.projection_hidden_dim = 16;
    config.pos_max_height = 4;
    config.pos_max_width = 4;
    config.trm_graph.patch_rank = Some(1);
    config.trm_graph.coarse_rank = Some(4);
    config.trm_graph.global_rank = Some(2);
    config.trm_graph.predict_coarse_substeps = 3;
    config.trm_graph.bank_schedule = VisionTrmGraphBankScheduleConfig {
        observe: VisionTrmGraphBankModeConfig::default(),
        refine: VisionTrmGraphBankModeConfig::default(),
        predict: VisionTrmGraphBankModeConfig::scene_slot_predict_preset(),
    };
    config
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
fn make_stage_aware_pyramid_model_with_kernel<B: BackendTrait>(
    device: &B::Device,
    kernel_enabled: bool,
) -> VisionDragon<B> {
    let mut config = make_stage_aware_pyramid_config();
    config.fused_kernels = FusedKernelConfig {
        enabled: kernel_enabled,
        ..FusedKernelConfig::default()
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
        &device,
        RhoStreamTestConfig {
            use_cls_token: false,
            local_diagonals: true,
            local_self: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
            mode_embeddings: true,
            decay: 1.0,
        },
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
        &device,
        RhoStreamTestConfig {
            use_cls_token: false,
            local_diagonals: true,
            local_self: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
            mode_embeddings: true,
            decay: 1.0,
        },
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
        &device,
        RhoStreamTestConfig {
            use_cls_token: false,
            local_diagonals: true,
            local_self: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
            mode_embeddings: false,
            decay: 1.0,
        },
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
        &device,
        RhoStreamTestConfig {
            use_cls_token: false,
            local_diagonals: true,
            local_self: true,
            wgpu_forward_kernel: false,
            wgpu_rollout_fused: false,
            mode_embeddings: false,
            decay: 1.0,
        },
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
        normalization: burn_dragon_core::DragonNormConfig::default(),
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
fn cellular_mode_helpers_share_observe_refine_predict_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_rho_stream_model::<Backend>(&device, true, true, true, false, false, true);
    let tokens_a = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let tokens_b = Tensor::<Backend, 3>::ones([1, 4, 4], &device);

    let state = model.cellular_state_from_tokens(tokens_a);
    let predicted = model.predict_cellular_state(state, 1, 1);
    assert_eq!(predicted.temporal_position, 1);
    assert_eq!(predicted.prediction_age, 1);

    let refined = model.refine_cellular_state(predicted.clone(), 1, 1);
    assert_eq!(refined.temporal_position, 1);
    assert_eq!(refined.prediction_age, 1);

    let observed = model.observe_cellular_state(refined, tokens_b, 1, 1);
    assert_eq!(observed.temporal_position, 1);
    assert_eq!(observed.prediction_age, 0);
    assert_eq!(observed.rho.shape().dims(), [1, 1, 4, 4, 4]);
}

#[test]
fn pyramid_mode_helpers_share_observe_refine_predict_contract() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens_a = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let tokens_b = Tensor::<Backend, 3>::ones([1, 4, 4], &device);

    let state = model.pyramid_state_from_patch_tokens(tokens_a);
    let predicted = model.predict_pyramid_state(state, 1, 1);
    assert_eq!(predicted.temporal_position, 1);
    assert_eq!(predicted.prediction_age, 1);

    let refined = model.refine_pyramid_state(predicted.clone(), 1, 1);
    assert_eq!(refined.temporal_position, 1);
    assert_eq!(refined.prediction_age, 1);

    let observed = model.observe_pyramid_state(refined, tokens_b, 1, 1);
    assert_eq!(observed.temporal_position, 1);
    assert_eq!(observed.prediction_age, 0);
    assert_eq!(observed.patch_rho().shape().dims(), [1, 2, 4, 2, 2]);
    assert_eq!(observed.coarse_rho().shape().dims(), [1, 2, 4, 1, 1]);
    assert_eq!(observed.hub_rho().shape().dims(), [1, 2, 2, 4]);
}

#[test]
fn pyramid_observe_preserves_local_and_global_rho_banks() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens_a = Tensor::<Backend, 3>::zeros([1, 4, 4], &device);
    let tokens_b = Tensor::<Backend, 3>::ones([1, 4, 4], &device);

    let mut state = model.pyramid_state_from_patch_tokens(tokens_a);
    *state.patch_rho_mut() =
        Tensor::<Backend, 5>::from_data(TensorData::new(vec![1.0; 16], [1, 2, 4, 2, 2]), &device);
    *state.coarse_rho_mut() =
        Tensor::<Backend, 5>::from_data(TensorData::new(vec![2.0; 8], [1, 2, 4, 1, 1]), &device);
    *state.hub_rho_mut() =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![3.0; 16], [1, 2, 2, 4]), &device);
    state.temporal_position = 5;
    state.prediction_age = 2;

    let observed = model.pyramid_state_with_patch_tokens(state.clone(), tokens_b);
    assert_eq!(observed.temporal_position, 5);
    assert_eq!(observed.prediction_age, 2);
    assert_eq!(
        observed
            .patch_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("patch rho"),
        state
            .patch_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("patch rho reference")
    );
    assert_eq!(
        observed
            .coarse_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("coarse rho"),
        state
            .coarse_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("coarse rho reference")
    );
    assert_eq!(
        observed
            .hub_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hub rho"),
        state
            .hub_rho()
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hub rho reference")
    );
}

#[test]
fn pyramid_hub_bank_produces_nonzero_patch_readout() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let h8 = state.patch_state().clone();
    let h32 = state.coarse_state().clone();
    let x8 = activation::relu(
        model.project_spatial(
            h8.clone(),
            model
                .pyramid_patch_to_global_query_proj
                .as_ref()
                .expect("pyramid global query projection"),
        ),
    );
    let (hub_w8, _) = model.pyramid_hub_weights(h8, h32, model.trm_graph.hub_count.max(1));
    let zero_hub = Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device);
    let ones_hub =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![1.0; 16], [1, 2, 2, 4]), &device);
    let zero_vec = model
        .pyramid_hub_read(zero_hub, x8.clone(), hub_w8.clone())
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("zero hub read");
    let hub_vec = model
        .pyramid_hub_read(ones_hub, x8, hub_w8)
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hub read");
    let max_diff = zero_vec
        .iter()
        .zip(hub_vec.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0_f32, f32::max);

    assert!(max_diff > 1e-6, "hub bank should influence patch readout");
}

#[test]
fn pyramid_patch_mean_summary_ignores_hub_rho() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut config = make_pyramid_config();
    config.trm_graph.cls_readout = VisionTrmClsReadoutKind::PatchMean;
    let model = VisionDragon::<Backend>::new(config, &device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let mut state = model.pyramid_state_from_patch_tokens(tokens);
    let baseline = model.pyramid_summary(&state);
    *state.hub_rho_mut() =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![5.0; 16], [1, 2, 2, 4]), &device);
    let changed = model.pyramid_summary(&state);

    assert!(
        max_abs_diff(baseline, changed) < 1e-6,
        "patch-mean summary should ignore hub rho changes"
    );
}

#[test]
fn pyramid_hub_cls_readout_responds_to_hub_rho() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut config = make_pyramid_config();
    config.trm_graph.cls_readout = VisionTrmClsReadoutKind::Hub;
    let model = VisionDragon::<Backend>::new(config, &device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let mut state = model.pyramid_state_from_patch_tokens(tokens);
    *state.hub_rho_mut() = Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device);
    let baseline = model.pyramid_summary(&state);
    *state.hub_rho_mut() = Tensor::<Backend, 4>::from_data(
        TensorData::new(
            (0..16).map(|value| value as f32).collect::<Vec<_>>(),
            [1, 2, 2, 4],
        ),
        &device,
    );
    let changed = model.pyramid_summary(&state);

    assert!(
        max_abs_diff(baseline, changed) > 1e-6,
        "hub cls readout should respond to hub rho changes"
    );
}

#[test]
fn pyramid_hub_and_coarse_cls_readout_responds_to_coarse_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut config = make_pyramid_config();
    config.trm_graph.cls_readout = VisionTrmClsReadoutKind::HubAndCoarse;
    let model = VisionDragon::<Backend>::new(config, &device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let mut state = model.pyramid_state_from_patch_tokens(tokens);
    let baseline = model.pyramid_summary(&state);
    *state.context_state_mut() =
        Tensor::<Backend, 4>::from_data(TensorData::new(vec![7.0; 4], [1, 4, 1, 1]), &device);
    let changed = model.pyramid_summary(&state);

    assert!(
        max_abs_diff(baseline, changed) > 1e-6,
        "hub+coarse cls readout should respond to coarse state changes"
    );
}

#[test]
fn pyramid_project_spatial_pair_matches_individual_projection() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let h8 = state.patch_state().clone();
    let h32 = state.coarse_state().clone();
    let layer = model
        .pyramid_write_value_proj
        .as_ref()
        .expect("pyramid value projection");

    let single_h8 = model.project_spatial(h8.clone(), layer);
    let single_h32 = model.project_spatial(h32.clone(), layer);
    let (pair_h8, pair_h32) = model.project_spatial_pair(h8, h32, layer);

    assert_eq!(
        single_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single patch projection"),
        pair_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair patch projection")
    );
    assert_eq!(
        single_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single coarse projection"),
        pair_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair coarse projection")
    );
}

#[test]
fn pyramid_local_bridge_projection_pair_matches_individual_projection() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let h8 = state.patch_state().clone();
    let h32 = state.coarse_state().clone();
    let patch_x_layer = model
        .pyramid_patch_x_neuron_proj
        .as_ref()
        .expect("pyramid patch x projection");
    let coarse_x_layer = model
        .pyramid_coarse_x_neuron_proj
        .as_ref()
        .expect("pyramid coarse x projection");
    let value_layer = model
        .pyramid_write_value_proj
        .as_ref()
        .expect("pyramid value projection");
    let plan =
        CompiledLocalBridgeProjectionPairPlan::new(patch_x_layer, coarse_x_layer, value_layer)
            .expect("local bridge projection pair plan");

    let patch_x = model.project_spatial(h8.clone(), patch_x_layer);
    let patch_v = model.project_spatial(h8.clone(), value_layer);
    let coarse_x = model.project_spatial(h32.clone(), coarse_x_layer);
    let coarse_v = model.project_spatial(h32.clone(), value_layer);
    let (pair_patch_x, pair_patch_v, pair_coarse_x, pair_coarse_v) =
        model.project_local_bridge_pair_with_plan(h8, h32, &plan);

    assert_eq!(
        patch_x
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single patch x projection"),
        pair_patch_x
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair patch x projection")
    );
    assert_eq!(
        patch_v
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single patch value projection"),
        pair_patch_v
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair patch value projection")
    );
    assert_eq!(
        coarse_x
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single coarse x projection"),
        pair_coarse_x
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair coarse x projection")
    );
    assert_eq!(
        coarse_v
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single coarse value projection"),
        pair_coarse_v
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair coarse value projection")
    );
}

#[test]
fn pyramid_update_states_matches_individual_updates() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let h8 = state.patch_state().clone();
    let h32 = state.coarse_state().clone();
    let x_proj = model
        .pyramid_patch_x_neuron_proj
        .as_ref()
        .expect("pyramid patch x projection");
    let v_proj = model
        .pyramid_write_value_proj
        .as_ref()
        .expect("pyramid value projection");
    let y_gate_proj = model
        .pyramid_patch_y_gate_proj
        .as_ref()
        .expect("pyramid patch y gate projection");
    let delta_proj = model
        .pyramid_patch_delta_proj
        .as_ref()
        .expect("pyramid patch delta projection");
    let value_norm = model
        .pyramid_value_norm
        .as_ref()
        .expect("pyramid value norm");

    let x8 = activation::relu(model.project_spatial(h8.clone(), x_proj));
    let msg8 = model.project_spatial(h8.clone(), v_proj);
    let x32 = activation::relu(model.project_spatial(h32.clone(), x_proj));
    let msg32 = model.project_spatial(h32.clone(), v_proj);

    let single_h8 = model.pyramid_update_state(
        h8.clone(),
        x8.clone(),
        msg8.clone(),
        y_gate_proj,
        delta_proj,
        value_norm,
    );
    let single_h32 = model.pyramid_update_state(
        h32.clone(),
        x32.clone(),
        msg32.clone(),
        y_gate_proj,
        delta_proj,
        value_norm,
    );
    let (pair_h8, pair_h32) = model.pyramid_update_states(
        h8,
        x8,
        msg8,
        h32,
        x32,
        msg32,
        y_gate_proj,
        delta_proj,
        value_norm,
    );

    assert_eq!(
        single_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single patch update"),
        pair_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair patch update")
    );
    assert_eq!(
        single_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single coarse update"),
        pair_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair coarse update")
    );
}

#[test]
fn pyramid_update_states_separate_with_plan_matches_individual_updates() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let h8 = state.patch_state().clone();
    let h32 = state.coarse_state().clone();
    let x8_proj = model
        .pyramid_patch_x_neuron_proj
        .as_ref()
        .expect("pyramid patch x projection");
    let x32_proj = model
        .pyramid_coarse_x_neuron_proj
        .as_ref()
        .expect("pyramid coarse x projection");
    let v_proj = model
        .pyramid_write_value_proj
        .as_ref()
        .expect("pyramid value projection");
    let patch_y_gate_proj = model
        .pyramid_patch_y_gate_proj
        .as_ref()
        .expect("pyramid patch y gate projection");
    let patch_delta_proj = model
        .pyramid_patch_delta_proj
        .as_ref()
        .expect("pyramid patch delta projection");
    let coarse_y_gate_proj = model
        .pyramid_coarse_y_gate_proj
        .as_ref()
        .expect("pyramid coarse y gate projection");
    let coarse_delta_proj = model
        .pyramid_coarse_delta_proj
        .as_ref()
        .expect("pyramid coarse delta projection");
    let value_norm = model
        .pyramid_value_norm
        .as_ref()
        .expect("pyramid value norm");
    let plan = CompiledStructuredDenseUpdatePairPlan::new(
        patch_y_gate_proj,
        patch_delta_proj,
        coarse_y_gate_proj,
        coarse_delta_proj,
    )
    .expect("matching patch/coarse update plan");

    let x8 = activation::relu(model.project_spatial(h8.clone(), x8_proj));
    let msg8 = model.project_spatial(h8.clone(), v_proj);
    let x32 = activation::relu(model.project_spatial(h32.clone(), x32_proj));
    let msg32 = model.project_spatial(h32.clone(), v_proj);

    let single_h8 = model.pyramid_update_state(
        h8.clone(),
        x8.clone(),
        msg8.clone(),
        patch_y_gate_proj,
        patch_delta_proj,
        value_norm,
    );
    let single_h32 = model.pyramid_update_state(
        h32.clone(),
        x32.clone(),
        msg32.clone(),
        coarse_y_gate_proj,
        coarse_delta_proj,
        value_norm,
    );
    let (pair_h8, pair_h32) = model
        .pyramid_update_states_separate_with_plan(h8, x8, msg8, h32, x32, msg32, value_norm, &plan);

    assert_eq!(
        single_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single patch update"),
        pair_h8
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair patch update")
    );
    assert_eq!(
        single_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("single coarse update"),
        pair_h32
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pair coarse update")
    );
}

#[test]
fn pyramid_state_uses_scale_aware_rank_overrides() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut config = make_pyramid_config();
    config.trm_graph.patch_rank = Some(3);
    config.trm_graph.coarse_rank = Some(5);
    config.trm_graph.global_rank = Some(2);
    let model = VisionDragon::<Backend>::new(config, &device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);

    let patch_shape = state.patch_rho().shape().dims::<5>();
    let coarse_shape = state.coarse_rho().shape().dims::<5>();
    let global_shape = state.hub_rho().shape().dims::<4>();

    assert_eq!(patch_shape[1], 3);
    assert_eq!(coarse_shape[1], 5);
    assert_eq!(global_shape[2], 2);
}

#[test]
fn pyramid_predict_writes_patch_activity_into_hub_bank() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_pyramid_model::<Backend>(&device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let state = model.pyramid_state_from_patch_tokens(tokens);
    let predicted = model.predict_pyramid_state(state, 1, 1);
    let hub_vec = predicted
        .hub_rho()
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("hub rho");
    let max_abs = hub_vec
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_abs > 1e-6,
        "predict should write patch/coarse activity into hub rho"
    );
}

#[test]
fn pyramid_predict_schedule_can_disable_patch_local_bank() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let default_model = make_pyramid_model::<Backend>(&device);
    let mut scheduled_config = make_pyramid_config();
    scheduled_config
        .trm_graph
        .bank_schedule
        .predict
        .patch_local_read = false;
    scheduled_config
        .trm_graph
        .bank_schedule
        .predict
        .patch_local_write = false;
    let scheduled_model = VisionDragon::<Backend>::new(scheduled_config, &device);
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16], [1, 4, 4]), &device);

    let default_state = default_model.predict_pyramid_state(
        default_model.pyramid_state_from_patch_tokens(tokens.clone()),
        2,
        2,
    );
    let scheduled_state = scheduled_model.predict_pyramid_state(
        scheduled_model.pyramid_state_from_patch_tokens(tokens),
        2,
        2,
    );

    let primary_diff = max_abs_diff(
        scheduled_state.primary_state().clone(),
        default_state.primary_state().clone(),
    );
    let patch_rho_diff = max_abs_diff(
        scheduled_state.patch_rho().clone(),
        default_state.patch_rho().clone(),
    );

    assert!(
        primary_diff > 1e-5,
        "predict schedule should change primary state"
    );
    assert!(
        patch_rho_diff > 1e-5,
        "predict schedule should change patch rho"
    );
}

#[cfg(feature = "train")]
#[test]
fn scene_slot_graph_bridge_preset_changes_predict_rollout_state() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let tokens =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![1.0; 16 * 8], [1, 16, 8]), &device);

    let make_model = |trm_graph: VisionTrmGraphConfig| {
        let mut config = make_stage_aware_pyramid_config();
        config.steps = 2;
        config.trm_graph = trm_graph;
        config.trm_graph.patch_rank = Some(1);
        config.trm_graph.coarse_rank = Some(4);
        config.trm_graph.global_rank = Some(2);
        VisionDragon::<Backend>::new(config, &device)
    };

    let bridge_model = make_model(VisionTrmGraphConfig::scene_slot_graph_bridge_preset());
    let scene_slot_model = make_model(VisionTrmGraphConfig::scene_slot_graph_preset())
        .load_record(bridge_model.clone().into_record());

    let bridge_state = bridge_model.forward_pyramid_state_rollout_mode_unbounded(
        bridge_model.pyramid_state_from_patch_tokens(tokens.clone()),
        2,
        2,
        StructuredStepMode::Predict,
    );
    let scene_slot_state = scene_slot_model.forward_pyramid_state_rollout_mode_unbounded(
        scene_slot_model.pyramid_state_from_patch_tokens(tokens),
        2,
        2,
        StructuredStepMode::Predict,
    );

    assert!(
        max_abs_diff(
            bridge_state.context_state().clone(),
            scene_slot_state.context_state().clone(),
        ) > 1e-5,
        "bridge preset should change coarse context state during predict rollout"
    );
    assert!(
        max_abs_diff(
            bridge_state.hub_rho().clone(),
            scene_slot_state.hub_rho().clone()
        ) > 1e-5,
        "bridge preset should change hub rho during predict rollout"
    );
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

fn max_abs_diff<B: BackendTrait, const D: usize>(lhs: Tensor<B, D>, rhs: Tensor<B, D>) -> f32 {
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

    lhs.iter()
        .zip(rhs.iter())
        .map(|(lhs_value, rhs_value)| (*lhs_value - *rhs_value).abs())
        .fold(0.0_f32, f32::max)
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
fn pyramid_rollout_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &VisionDragon<B>,
    tokens: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let state = model.pyramid_state_from_patch_tokens(tokens);
    let state = model.forward_pyramid_state_rollout_mode_unbounded(
        state,
        3,
        3,
        StructuredStepMode::Predict,
    );

    state.primary_state().clone().tanh().powf_scalar(2.0).mean()
        + state.context_state().clone().tanh().powf_scalar(2.0).mean()
        + state.patch_rho().clone().tanh().powf_scalar(2.0).mean()
        + state.coarse_rho().clone().tanh().powf_scalar(2.0).mean()
        + state.hub_rho().clone().tanh().powf_scalar(2.0).mean()
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
fn pyramid_mode_sequence_loss<B: burn::tensor::backend::AutodiffBackend>(
    model: &VisionDragon<B>,
    tokens_a: Tensor<B, 3>,
    tokens_b: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let state = model.pyramid_state_from_patch_tokens(tokens_a);
    let state = model.predict_pyramid_state(state, 2, 2);
    let state = model.observe_pyramid_state(state, tokens_b, 1, 1);
    let state = model.refine_pyramid_state(state, 1, 1);

    state.primary_state().clone().tanh().powf_scalar(2.0).mean()
        + state.context_state().clone().tanh().powf_scalar(2.0).mean()
        + state.patch_rho().clone().tanh().powf_scalar(2.0).mean()
        + state.coarse_rho().clone().tanh().powf_scalar(2.0).mean()
        + state.hub_rho().clone().tanh().powf_scalar(2.0).mean()
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
#[test]
fn pyramid_wgpu_rollout_matches_reference_public_state_path_with_hub_gates() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 6_060);

    let reference = make_pyramid_model_with_kernel::<WgpuBackend>(&device, false);
    let fused = make_pyramid_model_with_kernel::<WgpuBackend>(&device, true)
        .load_record(reference.clone().into_record());
    assert!(supports_structured_pyramid_rho_backend::<WgpuBackend>());

    let tokens =
        Tensor::<WgpuBackend, 3>::random([2, 4, 4], Distribution::Normal(0.0, 1.0), &device);
    let reference_state = reference.pyramid_state_from_patch_tokens(tokens.clone());
    let fused_state = fused.pyramid_state_from_patch_tokens(tokens);

    let reference_state = reference.forward_pyramid_state_rollout_mode_unbounded(
        reference_state,
        3,
        3,
        StructuredStepMode::Predict,
    );
    let fused_state = fused.forward_pyramid_state_rollout_mode_unbounded(
        fused_state,
        3,
        3,
        StructuredStepMode::Predict,
    );

    let primary_diff = max_abs_diff(
        fused_state.primary_state().clone(),
        reference_state.primary_state().clone(),
    );
    let context_diff = max_abs_diff(
        fused_state.context_state().clone(),
        reference_state.context_state().clone(),
    );
    let patch_rho_diff = max_abs_diff(
        fused_state.patch_rho().clone(),
        reference_state.patch_rho().clone(),
    );
    let coarse_rho_diff = max_abs_diff(
        fused_state.coarse_rho().clone(),
        reference_state.coarse_rho().clone(),
    );
    let hub_rho_diff = max_abs_diff(
        fused_state.hub_rho().clone(),
        reference_state.hub_rho().clone(),
    );

    assert!(
        primary_diff <= 5e-3,
        "primary dense-state drift {primary_diff} exceeds 5e-3"
    );
    assert!(
        context_diff <= 5e-3,
        "context dense-state drift {context_diff} exceeds 5e-3"
    );
    assert!(
        patch_rho_diff <= 5e-4,
        "patch rho drift {patch_rho_diff} exceeds 5e-4"
    );
    assert!(
        coarse_rho_diff <= 5e-4,
        "coarse rho drift {coarse_rho_diff} exceeds 5e-4"
    );
    assert!(
        hub_rho_diff <= 5e-4,
        "hub rho drift {hub_rho_diff} exceeds 5e-4"
    );
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
type WgpuAutodiffBackend = Autodiff<WgpuBackend>;

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn pyramid_wgpu_autodiff_matches_reference_after_one_step() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 7_171);

    let reference = make_pyramid_model_with_kernel::<WgpuAutodiffBackend>(&device, false);
    let fused = make_pyramid_model_with_kernel::<WgpuAutodiffBackend>(&device, true)
        .load_record(reference.clone().into_record());
    let tokens = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let mut reference_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let mut fused_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let lr: LearningRate = 1e-3;

    let reference_loss = pyramid_rollout_loss(&reference, tokens.clone());
    let reference_loss_value = reference_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference loss")[0];
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);
    let reference = reference_optimizer.step(lr, reference, reference_grads);

    let fused_loss = pyramid_rollout_loss(&fused, tokens.clone());
    let fused_loss_value = fused_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused loss")[0];
    let fused_grads = GradientsParams::from_grads(fused_loss.backward(), &fused);
    let fused = fused_optimizer.step(lr, fused, fused_grads);

    assert!((reference_loss_value - fused_loss_value).abs() <= 8e-2);

    let reference_state = reference.forward_pyramid_state_rollout_mode_unbounded(
        reference.pyramid_state_from_patch_tokens(tokens.clone()),
        3,
        3,
        StructuredStepMode::Predict,
    );
    let fused_state = fused.forward_pyramid_state_rollout_mode_unbounded(
        fused.pyramid_state_from_patch_tokens(tokens),
        3,
        3,
        StructuredStepMode::Predict,
    );

    assert!(
        max_abs_diff(
            fused_state.primary_state().clone(),
            reference_state.primary_state().clone(),
        ) <= 1.5e-1
    );
    assert!(
        max_abs_diff(
            fused_state.context_state().clone(),
            reference_state.context_state().clone(),
        ) <= 1.5e-1
    );
    assert!(
        max_abs_diff(
            fused_state.patch_rho().clone(),
            reference_state.patch_rho().clone()
        ) <= 5e-2
    );
    assert!(
        max_abs_diff(
            fused_state.coarse_rho().clone(),
            reference_state.coarse_rho().clone(),
        ) <= 5e-2
    );
    assert!(
        max_abs_diff(
            fused_state.hub_rho().clone(),
            reference_state.hub_rho().clone()
        ) <= 5e-2
    );
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn pyramid_wgpu_autodiff_matches_reference_on_observe_refine_predict_sequence() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuAutodiffBackend as BackendTrait>::seed(&device, 8_181);

    let reference = make_pyramid_model_with_kernel::<WgpuAutodiffBackend>(&device, false);
    let fused = make_pyramid_model_with_kernel::<WgpuAutodiffBackend>(&device, true)
        .load_record(reference.clone().into_record());
    let tokens_a = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let tokens_b = Tensor::<WgpuAutodiffBackend, 3>::random(
        [2, 4, 4],
        Distribution::Normal(0.0, 1.0),
        &device,
    );
    let mut reference_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let mut fused_optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<WgpuAutodiffBackend, VisionDragon<WgpuAutodiffBackend>>();
    let lr: LearningRate = 1e-3;

    let reference_loss = pyramid_mode_sequence_loss(&reference, tokens_a.clone(), tokens_b.clone());
    let reference_loss_value = reference_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("reference mode-sequence loss")[0];
    let reference_grads = GradientsParams::from_grads(reference_loss.backward(), &reference);
    let reference = reference_optimizer.step(lr, reference, reference_grads);

    let fused_loss = pyramid_mode_sequence_loss(&fused, tokens_a.clone(), tokens_b.clone());
    let fused_loss_value = fused_loss
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fused mode-sequence loss")[0];
    let fused_grads = GradientsParams::from_grads(fused_loss.backward(), &fused);
    let fused = fused_optimizer.step(lr, fused, fused_grads);

    assert!((reference_loss_value - fused_loss_value).abs() <= 1e-1);

    let reference_state = reference.refine_pyramid_state(
        reference.observe_pyramid_state(
            reference.predict_pyramid_state(
                reference.pyramid_state_from_patch_tokens(tokens_a.clone()),
                2,
                2,
            ),
            tokens_b.clone(),
            1,
            1,
        ),
        1,
        1,
    );
    let fused_state = fused.refine_pyramid_state(
        fused.observe_pyramid_state(
            fused.predict_pyramid_state(fused.pyramid_state_from_patch_tokens(tokens_a), 2, 2),
            tokens_b,
            1,
            1,
        ),
        1,
        1,
    );

    assert!(
        max_abs_diff(
            fused_state.primary_state().clone(),
            reference_state.primary_state().clone(),
        ) <= 2e-1
    );
    assert!(
        max_abs_diff(
            fused_state.context_state().clone(),
            reference_state.context_state().clone(),
        ) <= 2e-1
    );
    assert!(
        max_abs_diff(
            fused_state.patch_rho().clone(),
            reference_state.patch_rho().clone()
        ) <= 8e-2
    );
    assert!(
        max_abs_diff(
            fused_state.coarse_rho().clone(),
            reference_state.coarse_rho().clone(),
        ) <= 8e-2
    );
    assert!(
        max_abs_diff(
            fused_state.hub_rho().clone(),
            reference_state.hub_rho().clone()
        ) <= 8e-2
    );
}

#[cfg(all(feature = "train", not(target_arch = "wasm32")))]
#[test]
fn pyramid_wgpu_stage_aware_rollout_matches_reference_on_custom_schedule_and_ranks() {
    let _guard = crate::train::wgpu_test_guard();
    let device = <WgpuBackend as BackendTrait>::Device::default();
    crate::train::init_wgpu_test_runtime(&device);
    <WgpuBackend as BackendTrait>::seed(&device, 4_242);

    let reference = make_stage_aware_pyramid_model_with_kernel::<WgpuBackend>(&device, false);
    let fused = make_stage_aware_pyramid_model_with_kernel::<WgpuBackend>(&device, true)
        .load_record(reference.clone().into_record());
    assert!(supports_local_grid_rho_backend::<WgpuBackend>());

    let tokens =
        Tensor::<WgpuBackend, 3>::random([2, 16, 8], Distribution::Normal(0.0, 1.0), &device);
    let reference_state = reference.pyramid_state_from_patch_tokens(tokens.clone());
    let fused_state = fused.pyramid_state_from_patch_tokens(tokens);

    let reference_state = reference.forward_pyramid_state_rollout_mode_unbounded(
        reference_state,
        3,
        3,
        StructuredStepMode::Predict,
    );
    let fused_state = fused.forward_pyramid_state_rollout_mode_unbounded(
        fused_state,
        3,
        3,
        StructuredStepMode::Predict,
    );

    assert!(
        max_abs_diff(
            fused_state.primary_state().clone(),
            reference_state.primary_state().clone(),
        ) <= 6e-3
    );
    assert!(
        max_abs_diff(
            fused_state.context_state().clone(),
            reference_state.context_state().clone(),
        ) <= 6e-3
    );
    assert!(
        max_abs_diff(
            fused_state.patch_rho().clone(),
            reference_state.patch_rho().clone()
        ) <= 8e-4
    );
    assert!(
        max_abs_diff(
            fused_state.coarse_rho().clone(),
            reference_state.coarse_rho().clone(),
        ) <= 8e-4
    );
    assert!(
        max_abs_diff(
            fused_state.hub_rho().clone(),
            reference_state.hub_rho().clone()
        ) <= 8e-4
    );
}

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
