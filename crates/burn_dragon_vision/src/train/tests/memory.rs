use super::*;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_text_memory_stays_bounded_across_epochs() {
    type Backend = Autodiff<Wgpu<f32>>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let vocab = 64;
    let model = BDH::<Backend>::new(make_text_config(vocab), &device);
    let make_batch = || make_text_batch::<Backend>(&device, 2, 16, vocab);

    for _ in 0..2 {
        run_text_train_step(&model, make_batch());
    }
    let _ = Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            run_text_train_step(&model, make_batch());
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded("text", &snapshots, 1024 * 1024 * 1024, 256 * 1024 * 1024);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_vision_saccade_memory_stays_bounded_across_epochs() {
    type Backend = Wgpu<f32>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let config_path = vision_saccade_tiny_path();
    let mut config = load_vision_training_config(&[config_path]).expect("load vision_saccade_tiny");
    config.training.memory_cleanup_every = 0;

    let vision_config = config.vision.build();
    let saccade_config = match config.mode {
        VisionTrainingModeConfig::Saccade(config) => *config,
        other => panic!("expected saccade config, got {other:?}"),
    };
    let fixed_steps = config
        .training
        .rollout_max_steps
        .unwrap_or(vision_config.steps)
        .clamp(1, 4);
    let rollout = VisionRollout {
        min_steps: fixed_steps,
        max_steps: fixed_steps,
        backprop_steps: fixed_steps,
    };

    let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
    let recon_patch_dim = vision_config
        .patch_size
        .saturating_mul(vision_config.patch_size)
        .saturating_mul(vision_config.in_channels);
    let saccade = VisionSaccadeModel::new(
        model,
        saccade_config,
        vision_config.embed_dim,
        vision_config.patch_size,
        rollout,
        recon_patch_dim,
        config.training.batch_repeats,
        config.training.train_repeat_chunk,
        &device,
    );

    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [
                batch_size,
                vision_config.in_channels,
                vision_config.image_size,
                vision_config.image_size,
            ],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = ValidStep::step(&saccade, make_batch());
        drop(output);
    }
    let _ = Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = ValidStep::step(&saccade, make_batch());
            drop(output);
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded("vision", &snapshots, 1024 * 1024 * 1024, 256 * 1024 * 1024);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_vision_saccade_train_memory_stays_bounded_small_config() {
    type Backend = Autodiff<CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>>;
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (saccade, vision_config) = make_saccade_model_with_dims::<Backend>(&device, 1, 64, 64, 16);
    let batch_size = 1usize;
    let make_batch = || {
        let images = Tensor::<Backend, 4>::random(
            [batch_size, vision_config.in_channels, 64, 64],
            TensorDistribution::Default,
            &device,
        );
        let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
        ImageNetBatch::new(images, None, None, None, None, None, labels, None, None)
    };

    for _ in 0..2 {
        let output = burn_train::TrainStep::step(&saccade, make_batch());
        drop(output);
    }
    let _ = Backend::sync(&device);

    let epochs = 3;
    let steps_per_epoch = 2;
    let mut snapshots = Vec::with_capacity(epochs);
    for _ in 0..epochs {
        for _ in 0..steps_per_epoch {
            let output = burn_train::TrainStep::step(&saccade, make_batch());
            drop(output);
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "vision_train_small",
        &snapshots,
        1024 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_text_memory_stays_bounded_across_epochs() {
    type Backend = Autodiff<Cuda<f32>>;
    if !cuda_memory_pool_stable() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = burn_cuda::CudaDevice::default();

        let vocab = 64;
        let model = BDH::<Backend>::new(make_text_config(vocab), &device);
        let make_batch = || make_text_batch::<Backend>(&device, 2, 16, vocab);

        for _ in 0..2 {
            run_text_train_step(&model, make_batch());
        }
        let _ = Backend::sync(&device);

        let epochs = 3;
        let steps_per_epoch = 2;
        let mut snapshots = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            for _ in 0..steps_per_epoch {
                run_text_train_step(&model, make_batch());
            }
            let _ = Backend::sync(&device);
            if !cuda_memory_cleanup_safe::<Backend>(&device) {
                return;
            }
            let _ = Backend::sync(&device);
            let Some(snapshot) = cuda_memory_snapshot_safe(&device) else {
                return;
            };
            snapshots.push(snapshot);
        }

        assert_memory_growth_bounded(
            "cuda_text",
            &snapshots,
            1024 * 1024 * 1024,
            256 * 1024 * 1024,
        );
    }));
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_vision_saccade_memory_stays_bounded_across_epochs() {
    type Backend = Cuda<f32>;
    if !cuda_memory_pool_stable() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = burn_cuda::CudaDevice::default();

        let config_path = vision_saccade_tiny_path();
        let mut config =
            load_vision_training_config(&[config_path]).expect("load vision_saccade_tiny");
        config.training.memory_cleanup_every = 0;

        let vision_config = config.vision.build();
        let saccade_config = match config.mode {
            VisionTrainingModeConfig::Saccade(config) => *config,
            other => panic!("expected saccade config, got {other:?}"),
        };
        let fixed_steps = config
            .training
            .rollout_max_steps
            .unwrap_or(vision_config.steps)
            .clamp(1, 4);
        let rollout = VisionRollout {
            min_steps: fixed_steps,
            max_steps: fixed_steps,
            backprop_steps: fixed_steps,
        };

        let model = VisionDragon::<Backend>::new(vision_config.clone(), &device);
        let recon_patch_dim = vision_config
            .patch_size
            .saturating_mul(vision_config.patch_size)
            .saturating_mul(vision_config.in_channels);
        let saccade = VisionSaccadeModel::new(
            model,
            saccade_config,
            vision_config.embed_dim,
            vision_config.patch_size,
            rollout,
            recon_patch_dim,
            config.training.batch_repeats,
            config.training.train_repeat_chunk,
            &device,
        );

        let batch_size = 1usize;
        let make_batch = || {
            let images = Tensor::<Backend, 4>::random(
                [
                    batch_size,
                    vision_config.in_channels,
                    vision_config.image_size,
                    vision_config.image_size,
                ],
                TensorDistribution::Default,
                &device,
            );
            let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
            ImageNetBatch::new(images, None, None, None, None, None, labels, None, None)
        };

        for _ in 0..2 {
            let output = ValidStep::step(&saccade, make_batch());
            drop(output);
        }
        let _ = Backend::sync(&device);

        let epochs = 3;
        let steps_per_epoch = 2;
        let mut snapshots = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            for _ in 0..steps_per_epoch {
                let output = ValidStep::step(&saccade, make_batch());
                drop(output);
            }
            let _ = Backend::sync(&device);
            if !cuda_memory_cleanup_safe::<Backend>(&device) {
                return;
            }
            let _ = Backend::sync(&device);
            let Some(snapshot) = cuda_memory_snapshot_safe(&device) else {
                return;
            };
            snapshots.push(snapshot);
        }

        assert_memory_growth_bounded(
            "cuda_vision",
            &snapshots,
            1024 * 1024 * 1024,
            256 * 1024 * 1024,
        );
    }));
}

#[cfg(all(not(target_arch = "wasm32"), feature = "cuda"))]
#[test]
fn cuda_vision_saccade_train_memory_stays_bounded_small_config() {
    type Backend = Autodiff<Cuda<f32>>;
    if !cuda_memory_pool_stable() {
        return;
    }
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let device = burn_cuda::CudaDevice::default();

        let (saccade, vision_config) =
            make_saccade_model_with_dims::<Backend>(&device, 1, 64, 64, 16);
        let batch_size = 1usize;
        let make_batch = || {
            let images = Tensor::<Backend, 4>::random(
                [batch_size, vision_config.in_channels, 64, 64],
                TensorDistribution::Default,
                &device,
            );
            let labels = Tensor::<Backend, 1, Int>::zeros([batch_size], &device);
            ImageNetBatch::new(images, None, None, None, None, None, labels, None, None)
        };

        for _ in 0..2 {
            let output = burn_train::TrainStep::step(&saccade, make_batch());
            drop(output);
        }
        let _ = Backend::sync(&device);

        let epochs = 3;
        let steps_per_epoch = 2;
        let mut snapshots = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            for _ in 0..steps_per_epoch {
                let output = burn_train::TrainStep::step(&saccade, make_batch());
                drop(output);
            }
            let _ = Backend::sync(&device);
            if !cuda_memory_cleanup_safe::<Backend>(&device) {
                return;
            }
            let _ = Backend::sync(&device);
            let Some(snapshot) = cuda_memory_snapshot_safe(&device) else {
                return;
            };
            snapshots.push(snapshot);
        }

        assert_memory_growth_bounded(
            "cuda_vision_train_small",
            &snapshots,
            1024 * 1024 * 1024,
            256 * 1024 * 1024,
        );
    }));
}
