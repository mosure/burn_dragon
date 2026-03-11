use super::*;

#[test]
fn saccade_recon_loss_smoke() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 2, 1, true, false, false);
    let value = losses
        .total
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn saccade_multi_eye_loss_smoke() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 2, 1, true, false, false);
    let value = losses
        .total
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0];
    assert!(value.is_finite());
}

#[test]
fn saccade_multi_eye_trajectory_states_diverge() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let mut eye_values = vec![0.0; embed_dim];
    eye_values.extend(std::iter::repeat_n(1.0, embed_dim));
    let eye_token =
        Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
    saccade.eye_token = Param::from_tensor(eye_token);

    let image_len = 3 * 8 * 8;
    let mut image_values = Vec::with_capacity(image_len);
    for idx in 0..image_len {
        image_values.push(idx as f32 / 255.0);
    }
    let images =
        Tensor::<Backend, 4>::from_data(TensorData::new(image_values, [1, 3, 8, 8]), &device);

    let (traj0, _) = saccade_eye_step(&saccade, images.clone(), 0);
    let (traj1, _) = saccade_eye_step(&saccade, images, 1);
    let mse = (traj0 - traj1).powf_scalar(2.0).mean();
    assert_mse_above(mse, 0.0);
}

#[test]
fn saccade_multi_eye_updates_are_additive() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let mut eye_values = vec![0.0; embed_dim];
    eye_values.extend(std::iter::repeat_n(1.0, embed_dim));
    let eye_token =
        Tensor::<Backend, 2>::from_data(TensorData::new(eye_values, [2, embed_dim]), &device);
    saccade.eye_token = Param::from_tensor(eye_token);

    let images = Tensor::<Backend, 4>::random([1, 3, 8, 8], TensorDistribution::Default, &device);
    let (_, updates0) = saccade_eye_step(&saccade, images.clone(), 0);
    let (_, updates1) = saccade_eye_step(&saccade, images, 1);

    let mut state_sum: Vec<Tensor<Backend, 3>> = updates0
        .iter()
        .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
        .collect();
    for (state, update0, update1) in state_sum
        .iter_mut()
        .zip(updates0.iter())
        .zip(updates1.iter())
        .map(|((state, update0), update1)| (state, update0, update1))
    {
        *state = state.clone() + update0.clone() + update1.clone();
    }

    let mut state_seq: Vec<Tensor<Backend, 3>> = updates0
        .iter()
        .map(|update| Tensor::<Backend, 3>::zeros(update.shape().dims::<3>(), &device))
        .collect();
    for (state, update) in state_seq.iter_mut().zip(updates0.iter()) {
        *state = state.clone() + update.clone();
    }
    for (state, update) in state_seq.iter_mut().zip(updates1.iter()) {
        *state = state.clone() + update.clone();
    }

    for (sum, seq) in state_sum.iter().zip(state_seq.iter()) {
        let mse = (sum.clone() - seq.clone()).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }
}

#[test]
fn saccade_multi_eye_step_produces_finite_grads() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 1, 1, true, false, false);
    let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

    let eye_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
        .expect("eye_token grad");
    let traj_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
        .expect("trajectory_token grad");
    assert_tensor_finite(eye_grad);
    assert_tensor_finite(traj_grad);
}

#[test]
fn saccade_artifact_frames_match_steps() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (mut saccade, _vision_config) = make_saccade_model::<Backend>(&device, 2);
    saccade.config.artifact_max_images = 1;
    saccade.config.artifact_max_views = 4;
    saccade.config.artifact_every = 1;

    let images = Tensor::<Backend, 4>::zeros([1, 3, 8, 8], &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([1], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);

    let steps = 3;
    let losses = saccade.forward_losses(batch, steps, 1, false, true, true);
    let artifacts = losses.artifacts.expect("artifacts");
    let frames = artifacts.frames.expect("frames");
    let views = artifacts.views.expect("views");
    let [batch, frame_count, _, _, _] = frames.shape().dims::<5>();
    let [view_batch, view_count, _, _, _] = views.shape().dims::<5>();
    assert_eq!(batch, 1);
    assert_eq!(frame_count, steps);
    assert_eq!(view_batch, 1);
    assert_eq!(view_count, 4);
}

#[test]
fn saccade_laplacian_roundtrip_is_exact() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grids = vec![
        PatchGrid {
            height: 4,
            width: 4,
        },
        PatchGrid {
            height: 2,
            width: 2,
        },
        PatchGrid {
            height: 1,
            width: 1,
        },
    ];
    let levels = vec![
        make_level::<Backend>(&device, 16, 2, 0.0),
        make_level::<Backend>(&device, 4, 2, 10.0),
        make_level::<Backend>(&device, 1, 2, 20.0),
    ];
    let residuals = saccade.decompose_pyramid(&levels, &grids);
    let composed = saccade.compose_pyramid(&residuals, &grids);
    for (orig, recon) in levels.iter().zip(composed.iter()) {
        let mse = (orig.clone() - recon.clone()).powf_scalar(2.0).mean();
        assert_mse_below(mse, 1e-6);
    }
}

#[test]
fn saccade_laplacian_drop_residual_matches_upsample() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grids = vec![
        PatchGrid {
            height: 4,
            width: 4,
        },
        PatchGrid {
            height: 2,
            width: 2,
        },
        PatchGrid {
            height: 1,
            width: 1,
        },
    ];
    let levels = vec![
        make_level::<Backend>(&device, 16, 2, 0.0),
        make_level::<Backend>(&device, 4, 2, 10.0),
        make_level::<Backend>(&device, 1, 2, 20.0),
    ];
    let residuals = saccade.decompose_pyramid(&levels, &grids);
    let mut truncated = residuals.clone();
    truncated[0] = Tensor::<Backend, 3>::zeros([1, 16, 2], &device);
    let composed = saccade.compose_pyramid(&truncated, &grids);
    let upsampled = saccade.upsample_tokens(levels[1].clone(), grids[1], grids[0]);
    let mse = (composed[0].clone() - upsampled).powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_mip_gaussian_weights_normalize() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let levels = vec![
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
            grid: PatchGrid {
                height: 2,
                width: 2,
            },
            image: Tensor::<Backend, 4>::zeros([2, 3, 8, 8], &device),
        },
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
            grid: PatchGrid {
                height: 1,
                width: 1,
            },
            image: Tensor::<Backend, 4>::zeros([2, 3, 4, 4], &device),
        },
    ];
    let mean = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.2, 0.4, 0.7, 0.9], [2, 1, 2]),
        &device,
    );
    let sigma =
        Tensor::<Backend, 3>::from_data(TensorData::new(vec![0.3, 0.5], [2, 1, 1]), &device);
    let weights = saccade.mip_gaussian_weights(&levels, mean, sigma);
    let mut total = weights[0].clone().sum_dim(2);
    for weight in weights.iter().skip(1) {
        total = total + weight.clone().sum_dim(2);
    }
    let ones = Tensor::<Backend, 3>::ones([2, 1, 1], &device);
    let diff = total.add(ones.mul_scalar(-1.0));
    let mse = diff.powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_fovea_params_use_configured_trajectory_tokens() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let embed_dim = saccade.trajectory_token.val().shape().dims::<2>()[1];
    let traj_len = saccade.trajectory_token.val().shape().dims::<2>()[0].max(1);
    assert_eq!(traj_len, saccade.config.traj_tokens.max(1));

    let base_traj = saccade
        .trajectory_token
        .val()
        .reshape([1, traj_len, embed_dim]);
    let eye_embed = saccade
        .eye_token
        .val()
        .reshape([1, 1, embed_dim])
        .repeat_dim(1, traj_len);
    let traj_with_eye = base_traj + eye_embed;
    let fovea_params = saccade_fovea_params(&saccade, traj_with_eye.clone(), embed_dim);
    assert_eq!(fovea_params.shape().dims::<3>(), [1, 1, 3]);

    let levels = vec![
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([1, 4, 3], &device),
            grid: PatchGrid {
                height: 2,
                width: 2,
            },
            image: Tensor::<Backend, 4>::zeros([1, 3, 4, 4], &device),
        },
        SaccadeMipLevel {
            tokens: Tensor::<Backend, 3>::zeros([1, 1, 3], &device),
            grid: PatchGrid {
                height: 1,
                width: 1,
            },
            image: Tensor::<Backend, 4>::zeros([1, 3, 2, 2], &device),
        },
    ];
    let weights = saccade_weights_for_eye(&saccade, traj_with_eye, &levels, embed_dim);
    for weight in weights {
        let shape = weight.shape().dims::<3>();
        assert_eq!(
            shape[1], traj_len,
            "fovea weights should track the configured trajectory token count"
        );
    }
}

#[test]
fn saccade_mip_scatter_gather_one_hot() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let residual = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 1, 3]),
        &device,
    );
    let mut state_levels = vec![
        Tensor::<Backend, 3>::zeros([2, 4, 3], &device),
        Tensor::<Backend, 3>::zeros([2, 1, 3], &device),
    ];
    let weights_level0 = Tensor::<Backend, 3>::from_data(
        TensorData::new(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [2, 1, 4]),
        &device,
    );
    let weights_level1 = Tensor::<Backend, 3>::zeros([2, 1, 1], &device);
    let weights = vec![weights_level0, weights_level1];

    saccade.apply_mip_residual(&mut state_levels, &weights, residual.clone());
    let gathered = saccade.mip_weighted_sum(&state_levels, &weights);
    let mse = (gathered - residual).powf_scalar(2.0).mean();
    assert_mse_below(mse, 1e-6);
}

#[test]
fn saccade_upsample_tokens_mismatch_returns_zero() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);

    let tokens = Tensor::<Backend, 3>::zeros([1, 1, 2], &device);
    let from = PatchGrid {
        height: 2,
        width: 2,
    };
    let to = PatchGrid {
        height: 3,
        width: 3,
    };
    let upsampled = saccade.upsample_tokens(tokens, from, to);

    assert_eq!(upsampled.shape().dims(), [1, 9, 2]);
    let value = upsampled
        .sum()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("sum vec")[0];
    assert_eq!(value, 0.0);
}

#[test]
fn saccade_level_coords_cache_is_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let grid = PatchGrid {
        height: 2,
        width: 2,
    };

    let _ = saccade.level_coords_cached(grid, &device);
    let _ = saccade.level_coords_cached(grid, &device);

    let len = saccade
        .level_coords_cache
        .inner
        .lock()
        .expect("level coords cache lock")
        .map
        .len();
    assert_eq!(len, 1);
}

#[test]
fn saccade_upsample_weights_cache_is_bounded() {
    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let from = PatchGrid {
        height: 2,
        width: 2,
    };
    let to = PatchGrid {
        height: 4,
        width: 4,
    };

    let _ = saccade.upsample_weights_cached(from, to, &device);
    let _ = saccade.upsample_weights_cached(from, to, &device);

    let len = saccade
        .upsample_weights_cache
        .inner
        .lock()
        .expect("upsample weights cache lock")
        .map
        .len();
    assert_eq!(len, 1);
}

#[test]
fn saccade_step_produces_finite_grads() {
    type Backend = Autodiff<NdArray<f32>>;
    let device = <Backend as BackendTrait>::Device::default();
    let (saccade, _vision_config) = make_saccade_model::<Backend>(&device, 1);
    let images = Tensor::<Backend, 4>::random([2, 3, 8, 8], TensorDistribution::Default, &device);
    let labels = Tensor::<Backend, 1, Int>::zeros([2], &device);
    let batch = ImageNetBatch::new(images, None, None, None, None, None, labels, None, None);
    let losses = saccade.forward_losses(batch, 1, 1, true, false, false);
    let grads = GradientsParams::from_grads(losses.total.backward(), &saccade);

    let token_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.trajectory_token.id)
        .expect("trajectory_token grad");
    let eye_grad = grads
        .get::<ValidBackend<Backend>, 2>(saccade.eye_token.id)
        .expect("eye_token grad");
    assert_tensor_finite(token_grad);
    assert_tensor_finite(eye_grad);
}
