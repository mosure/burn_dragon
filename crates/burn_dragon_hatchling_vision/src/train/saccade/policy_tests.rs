use crate::train::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::TensorData;
use burn::tensor::backend::Backend as BackendTrait;
use burn_ndarray::NdArray;
use burn_dragon_hatchling_core::{
    FusedKernelConfig, SpatialPositionalEncodingKind, VisionAttentionMode,
    VisionLocationEmbeddingMode,
};

fn make_saccade_model<B: BackendTrait>(
    device: &B::Device,
    mode: VisionLocationEmbeddingMode,
) -> VisionSaccadeModel<B> {
    let vision_config = VisionDragonHatchlingConfig {
        image_size: 32,
        patch_size: 16,
        in_channels: 3,
        embed_dim: 16,
        steps: 1,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 16,
        projection_hidden_dim: 16,
        use_cls_token: true,
        pos_encoding: SpatialPositionalEncodingKind::Learned2d,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        fused_kernels: FusedKernelConfig::default(),
    };
    let model = VisionDragonHatchling::<B>::new(vision_config.clone(), device);
    let mut saccade_config = VisionSaccadeConfig::default();
    saccade_config.mip_levels = 2;
    saccade_config.loss.recon.weight = 0.0;
    saccade_config.policy.location_embedding.mode = mode;
    saccade_config.policy.location_embedding.embed_dim = 8;
    let rollout = VisionRollout {
        min_steps: 1,
        max_steps: 1,
        backprop_steps: 1,
    };
    let recon_patch_dim = vision_config.patch_size * vision_config.patch_size * vision_config.in_channels;
    VisionSaccadeModel::new(
        model,
        saccade_config,
        vision_config.embed_dim,
        vision_config.patch_size,
        rollout,
        recon_patch_dim,
        1,
        0,
        device,
    )
}

fn tensor_scalar<B: BackendTrait>(tensor: Tensor<B, 1>) -> f32 {
    tensor
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("tensor vec")[0]
}

#[test]
fn location_embedding_none_is_invariant() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let saccade = make_saccade_model::<Backend>(&device, VisionLocationEmbeddingMode::None);

    let input_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let state_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let mean_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.3, 0.7, 0.8], [2, 1, 2]),
        &device,
    );
    let sigma_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.15, 0.35], [2, 1, 1]),
        &device,
    );
    let mean_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.6, 0.4, 0.1, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.25, 0.45], [2, 1, 1]),
        &device,
    );

    let tokens_a = saccade.build_input_tokens(
        input_context.clone(),
        state_context.clone(),
        mean_a,
        sigma_a,
    );
    let tokens_b =
        saccade.build_input_tokens(input_context, state_context, mean_b, sigma_b);
    let mse = (tokens_a - tokens_b).powf_scalar(2.0).mean();
    assert!(tensor_scalar(mse) < 1e-6);
}

#[test]
fn location_embedding_learned_changes_tokens() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let saccade = make_saccade_model::<Backend>(&device, VisionLocationEmbeddingMode::Learned);

    let input_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let state_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let mean_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.3, 0.7, 0.8], [2, 1, 2]),
        &device,
    );
    let sigma_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.15, 0.35], [2, 1, 1]),
        &device,
    );
    let mean_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.6, 0.4, 0.1, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.25, 0.45], [2, 1, 1]),
        &device,
    );

    let tokens_a = saccade.build_input_tokens(
        input_context.clone(),
        state_context.clone(),
        mean_a,
        sigma_a,
    );
    let tokens_b =
        saccade.build_input_tokens(input_context, state_context, mean_b, sigma_b);
    let mse = (tokens_a - tokens_b).powf_scalar(2.0).mean();
    assert!(tensor_scalar(mse) > 1e-6);
}

#[test]
fn location_embedding_rope_changes_tokens() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let saccade = make_saccade_model::<Backend>(&device, VisionLocationEmbeddingMode::Rope);

    let input_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let state_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let mean_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.3, 0.7, 0.8], [2, 1, 2]),
        &device,
    );
    let sigma_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.15, 0.35], [2, 1, 1]),
        &device,
    );
    let mean_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.6, 0.4, 0.1, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.25, 0.45], [2, 1, 1]),
        &device,
    );

    let tokens_a = saccade.build_input_tokens(
        input_context.clone(),
        state_context.clone(),
        mean_a,
        sigma_a,
    );
    let tokens_b =
        saccade.build_input_tokens(input_context, state_context, mean_b, sigma_b);
    let mse = (tokens_a - tokens_b).powf_scalar(2.0).mean();
    assert!(tensor_scalar(mse) > 1e-6);
}

#[test]
fn location_embedding_pope_changes_tokens() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let saccade = make_saccade_model::<Backend>(&device, VisionLocationEmbeddingMode::Pope);

    let input_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let state_context =
        Tensor::<Backend, 3>::random([2, 1, 16], Distribution::Default, &device);
    let mean_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.3, 0.7, 0.8], [2, 1, 2]),
        &device,
    );
    let sigma_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.15, 0.35], [2, 1, 1]),
        &device,
    );
    let mean_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.6, 0.4, 0.1, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.25, 0.45], [2, 1, 1]),
        &device,
    );

    let tokens_a = saccade.build_input_tokens(
        input_context.clone(),
        state_context.clone(),
        mean_a,
        sigma_a,
    );
    let tokens_b =
        saccade.build_input_tokens(input_context, state_context, mean_b, sigma_b);
    let mse = (tokens_a - tokens_b).powf_scalar(2.0).mean();
    assert!(tensor_scalar(mse) > 1e-6);
}

#[test]
fn location_embedding_pope_fills_double_budget() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let mut saccade = make_saccade_model::<Backend>(&device, VisionLocationEmbeddingMode::Pope);
    saccade.config.policy.location_embedding.embed_dim = 4;

    let input_context = Tensor::<Backend, 3>::zeros([2, 1, 16], &device);
    let state_context = Tensor::<Backend, 3>::zeros([2, 1, 16], &device);
    let mean_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.3, 0.7, 0.8], [2, 1, 2]),
        &device,
    );
    let sigma_a = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.15, 0.35], [2, 1, 1]),
        &device,
    );
    let mean_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.6, 0.4, 0.1, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma_b = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.25, 0.45], [2, 1, 1]),
        &device,
    );

    let tokens_a = saccade.build_input_tokens(
        input_context.clone(),
        state_context.clone(),
        mean_a,
        sigma_a,
    );
    let tokens_b = saccade.build_input_tokens(input_context, state_context, mean_b, sigma_b);
    let delta = tokens_a - tokens_b;

    let embed_dim = delta.shape().dims::<3>()[2];
    let base_dim = saccade
        .config
        .policy
        .location_embedding
        .embed_dim
        .min(embed_dim / 2);
    let pope_dim = base_dim * 2;

    let mid = delta.clone().slice_dim(2, base_dim..pope_dim);
    let mid_norm = mid.abs().sum();
    assert!(tensor_scalar(mid_norm) > 1e-6);

    if pope_dim < embed_dim {
        let tail = delta.slice_dim(2, pope_dim..embed_dim);
        let tail_norm = tail.abs().sum();
        assert!(tensor_scalar(tail_norm) < 1e-6);
    }
}
