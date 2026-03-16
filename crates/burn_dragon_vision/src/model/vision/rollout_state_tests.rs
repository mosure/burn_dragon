use super::*;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Distribution, Tensor};
use burn_ndarray::NdArray;

type Backend = NdArray<f32>;

fn make_model(backbone: VisionBackboneKind) -> VisionDragon<Backend> {
    let device = <Backend as BackendTrait>::Device::default();
    let vision = VisionDragonConfig {
        image_size: 8,
        patch_size: 4,
        backbone,
        in_channels: 3,
        embed_dim: 16,
        steps: 4,
        n_head: 2,
        mlp_internal_dim_multiplier: 2,
        dropout: 0.0,
        projection_dim: 12,
        projection_hidden_dim: 24,
        use_cls_token: true,
        pos_encoding: SpatialPositionalEncodingKind::Rope,
        pos_max_height: 2,
        pos_max_width: 2,
        attention_mode: VisionAttentionMode::RowL1,
        fused_kernels: FusedKernelConfig::default(),
        trm_graph: Default::default(),
        rho_stream: Default::default(),
        ..VisionDragonConfig::default()
    };
    VisionDragon::<Backend>::new(vision, &device)
}

fn assert_close(actual: Tensor<Backend, 3>, expected: Tensor<Backend, 3>, tol: f32) {
    let actual = actual
        .into_data()
        .to_vec::<f32>()
        .expect("actual tensor data");
    let expected = expected
        .into_data()
        .to_vec::<f32>()
        .expect("expected tensor data");
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "tensor mismatch at index {index}: actual={a}, expected={b}, tol={tol}"
        );
    }
}

fn assert_close_cls(actual: Tensor<Backend, 2>, expected: Tensor<Backend, 2>, tol: f32) {
    let actual = actual.into_data().to_vec::<f32>().expect("actual cls data");
    let expected = expected
        .into_data()
        .to_vec::<f32>()
        .expect("expected cls data");
    assert_eq!(actual.len(), expected.len());
    for (index, (a, b)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
        assert!(
            (a - b).abs() <= tol,
            "cls mismatch at index {index}: actual={a}, expected={b}, tol={tol}"
        );
    }
}

#[test]
fn dense_rollout_state_matches_public_rollout() {
    let device = <Backend as BackendTrait>::Device::default();
    let model = make_model(VisionBackboneKind::Dense);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);

    let state = model.rollout_state_from_images(images.clone());
    let state = model.refine_rollout_state_unbounded(state, 4, 2);
    let generic = model.forward_rollout_state(&state);
    let direct = model.forward_images_steps_rollout_unbounded(images, 4, 2);

    assert_close(generic.patch_tokens, direct.patch_tokens, 1e-6);
    assert_close_cls(generic.cls_token, direct.cls_token, 1e-6);
}

#[test]
fn stateful_rollout_state_variants_smoke() {
    let device = <Backend as BackendTrait>::Device::default();
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);

    let pyramid = make_model(VisionBackboneKind::Pyramid);
    let pyramid_state = pyramid.rollout_state_from_images(images.clone());
    assert!(matches!(pyramid_state, VisionRolloutState::Pyramid(_)));
    let pyramid_state = pyramid.refine_rollout_state_unbounded(pyramid_state, 2, 1);
    let pyramid_output = pyramid.forward_rollout_state(&pyramid_state);
    assert_eq!(pyramid_output.patch_tokens.shape().dims::<3>(), [2, 4, 12]);
    assert_eq!(pyramid_output.cls_token.shape().dims::<2>(), [2, 12]);

    let cellular = make_model(VisionBackboneKind::Cellular);
    let cellular_state = cellular.rollout_state_from_images(images);
    assert!(matches!(cellular_state, VisionRolloutState::Cellular(_)));
    let cellular_state = cellular.refine_rollout_state_unbounded(cellular_state, 2, 1);
    let cellular_output = cellular.forward_rollout_state(&cellular_state);
    assert_eq!(cellular_output.patch_tokens.shape().dims::<3>(), [2, 4, 12]);
    assert_eq!(cellular_output.cls_token.shape().dims::<2>(), [2, 12]);
}

#[test]
fn stateful_rollout_schedule_matches_repeated_public_rollout() {
    let device = <Backend as BackendTrait>::Device::default();
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], Distribution::Default, &device);
    let schedule = vec![(2usize, 2usize), (4usize, 3usize)];

    for backbone in [VisionBackboneKind::Pyramid, VisionBackboneKind::Cellular] {
        let model = make_model(backbone);
        let scheduled = model.predict_rollout_state_schedule_unbounded(
            model.rollout_state_from_images(images.clone()),
            &schedule,
        );
        assert_eq!(scheduled.len(), schedule.len());

        for ((scheduled_step, state), (step, backprop_steps)) in
            scheduled.into_iter().zip(schedule.iter().copied())
        {
            assert_eq!(scheduled_step, step);
            let scheduled_output = model.forward_rollout_state(&state);
            let repeated =
                model.forward_images_steps_rollout_unbounded(images.clone(), step, backprop_steps);
            assert_close(scheduled_output.patch_tokens, repeated.patch_tokens, 1e-3);
            assert_close_cls(scheduled_output.cls_token, repeated.cls_token, 1e-3);
        }
    }
}
