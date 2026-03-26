use super::*;
use crate::train::vision::video::dataset::MovingMnistVideoLoaderConfig;

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_trm_artifact_validation_memory_stays_bounded() {
    type Backend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.artifact_every = 1;
    video.artifact_max_images = 2;
    video.artifact_future_frames = 12;
    video.artifact_upscale = 2;
    video.loss.debug_recon_weight = 1.0;

    let model = make_video_model::<Backend>(&vision, &video, &device);

    for _ in 0..2 {
        let batch =
            toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 12).with_capture_artifacts(true);
        let output = burn_train::InferenceStep::step(&model, batch);
        drop(output);
    }
    let _ = Backend::sync(&device);
    Backend::memory_cleanup(&device);
    let _ = Backend::sync(&device);

    let mut snapshots = Vec::with_capacity(3);
    for _epoch in 0..3 {
        for step in 0..2 {
            let batch = toy_video_batch_with_lengths::<Backend>(&device, 3, 2, 12)
                .with_capture_artifacts(step == 0);
            let output = burn_train::InferenceStep::step(&model, batch);
            drop(output);
        }
        let _ = Backend::sync(&device);
        Backend::memory_cleanup(&device);
        let _ = Backend::sync(&device);
        snapshots.push(wgpu_memory_snapshot(&device));
    }

    assert_memory_growth_bounded(
        "video_trm_artifact_validation_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        128 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_non_trm_train_horizon_curriculum_memory_stays_bounded() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    video.target_frames = 2;
    video.train_target_frames_min = 2;
    video.train_target_frames_max = 6;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;

    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 1e-3;

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let target_len = 2 + (step % 5);
        let batch =
            toy_video_batch_with_lengths::<Backend>(&device, 3, 6, 0).with_target_len(target_len);
        let losses = model.forward_losses(batch, 2, 2, false, false, true);
        let total = losses.total.clone();
        let grads = GradientsParams::from_grads(total.backward(), &model);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);

        let _ = InnerBackend::sync(&device);
        InnerBackend::memory_cleanup(&device);
        let _ = InnerBackend::sync(&device);

        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "video_non_trm_train_horizon_curriculum_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_trm_train_memory_stays_bounded_fixed_horizon() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.target_frames = 6;
    video.train_target_frames_min = 6;
    video.train_target_frames_max = 6;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;
    video.teacher_ema.enabled = false;

    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 1e-3;

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let batch = toy_video_batch_with_lengths::<Backend>(&device, 4, 6, 0);
        let losses = model.forward_losses_train_pyramid(batch, 2, 2, false, true);
        let total = losses.total.clone();
        let grads = GradientsParams::from_grads(total.backward(), &model);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, grads);

        let _ = InnerBackend::sync(&device);
        InnerBackend::memory_cleanup(&device);
        let _ = InnerBackend::sync(&device);

        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "video_trm_train_fixed_horizon_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_trm_train_step_output_memory_stays_bounded() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.target_frames = 6;
    video.train_target_frames_min = 6;
    video.train_target_frames_max = 6;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;
    video.teacher_ema.enabled = false;

    let model = make_video_model::<Backend>(&vision, &video, &device);
    let mut snapshots = Vec::with_capacity(24);

    for step in 0..32 {
        let batch = toy_video_batch_with_lengths::<Backend>(&device, 4, 6, 0);
        let output = burn_train::TrainStep::step(&model, batch);
        drop(output);

        let _ = InnerBackend::sync(&device);
        InnerBackend::memory_cleanup(&device);
        let _ = InnerBackend::sync(&device);

        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "video_trm_train_step_output_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_trm_step_optimize_loop_memory_stays_bounded() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.target_frames = 6;
    video.train_target_frames_min = 6;
    video.train_target_frames_max = 6;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;
    video.teacher_ema.enabled = false;

    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 1e-3;

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let batch = toy_video_batch_with_lengths::<Backend>(&device, 4, 6, 0);
        let output = burn_train::TrainStep::step(&model, batch);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, output.grads);
        drop(output.item);

        let _ = InnerBackend::sync(&device);
        InnerBackend::memory_cleanup(&device);
        let _ = InnerBackend::sync(&device);

        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "video_trm_step_optimize_loop_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn wgpu_video_trm_dataloader_step_optimize_memory_stays_bounded() {
    type InnerBackend = CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>;
    type Backend = Autodiff<InnerBackend>;

    let _guard = wgpu_test_guard();
    let device = burn_wgpu::WgpuDevice::default();
    init_wgpu_test_runtime(&device);

    let (vision, mut video) = make_video_configs(false, false, 1);
    let vision = enable_pyramid_backbone(vision);
    video.target_frames = 6;
    video.train_target_frames_min = 6;
    video.train_target_frames_max = 6;
    video.loss.sigreg.enabled = false;
    video.loss.cosine_weight = 0.0;
    video.loss.probe_weight = 0.0;
    video.loss.debug_recon_weight = 0.0;
    video.teacher_ema.enabled = false;

    let mut model = make_video_model::<Backend>(&vision, &video, &device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<Backend, VisionVideoLejepaModel<Backend>>();
    let lr: LearningRate = 1e-3;

    let dataset = Arc::new(
        MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Train,
            frame_size: 8,
            digit_size: 4,
            in_channels: 3,
            context_len: 4,
            target_len: 6,
            extra_future_frames: 0,
            frame_stride: 1,
            max_records: Some(1024),
            normalize: VisionNormalize::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            min_velocity: 0.8,
            max_velocity: 2.0,
            seed: 1337,
        })
        .expect("moving mnist dataset"),
    );
    let loader = MovingMnistVideoDataLoader::<Backend>::new(
        dataset,
        &device,
        MovingMnistVideoLoaderConfig {
            batch_size: 64,
            steps_per_epoch: 32,
            total_steps: Some(32),
            target_horizon_curriculum: None,
            prefetch_batches: 0,
            prefetch_workers: 0,
            prefetch_to_device: false,
            sequential: false,
            artifact_capture_every: 0,
            artifact_capture_images: 0,
            artifact_extra_future_frames: 0,
        },
    );
    let mut iterator = loader.iter();

    let mut snapshots = Vec::with_capacity(24);
    for step in 0..32 {
        let batch = iterator.next().expect("train batch");
        let output = burn_train::TrainStep::step(&model, batch);
        model = model.optimize::<Backend, _>(&mut optimizer, lr, output.grads);
        drop(output.item);

        let _ = InnerBackend::sync(&device);
        InnerBackend::memory_cleanup(&device);
        let _ = InnerBackend::sync(&device);

        if step >= 8 {
            snapshots.push(wgpu_memory_snapshot(&device));
        }
    }

    assert_memory_growth_bounded(
        "video_trm_dataloader_step_optimize_wgpu",
        &snapshots,
        512 * 1024 * 1024,
        256 * 1024 * 1024,
    );
}

#[test]
fn video_trm_runlike_reports_parameter_bytes() {
    type Backend = NdArray<f32>;

    struct ParamBytesVisitor {
        elements: usize,
    }

    impl<B: BackendTrait> burn::module::ModuleVisitor<B> for ParamBytesVisitor {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let numel = param.shape().dims::<D>().iter().product::<usize>();
            self.elements += numel;
        }
    }

    let device = <Backend as BackendTrait>::Device::default();
    let (vision, video) = make_video_configs_runlike(false, false, 4);
    let model = make_video_model::<Backend>(&vision, &video, &device);
    let mut visitor = ParamBytesVisitor { elements: 0 };
    model.visit(&mut visitor);
    let bytes = visitor.elements * core::mem::size_of::<f32>();

    println!(
        "video_trm_runlike_param_elements={}, bytes={}",
        visitor.elements, bytes
    );
}
