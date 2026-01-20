#![recursion_limit = "256"]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::tensor::backend::AutodiffBackend;
use burn_autodiff::Autodiff;
use burn_dragon::train::{
    VisionTrainingConfig, VisionTrainingModeConfig, WgpuRuntimeConfig,
    load_vision_training_config,
};
use burn_dragon::{
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
    vision::train::bench::VisionLejepaTrainStepBench,
};
use burn_dragon::train::wgpu::init_runtime;
use burn_wgpu::{CubeBackend, WgpuDevice, WgpuRuntime};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

#[derive(Debug, Default, Deserialize)]
struct BenchSettings {
    #[serde(default)]
    include_cuda: bool,
}

#[derive(Debug, Default, Deserialize)]
struct BenchConfig {
    #[serde(default)]
    bench: BenchSettings,
}

fn load_bench_settings(path: &Path) -> BenchSettings {
    let contents = fs::read_to_string(path).expect("read bench config");
    let config: BenchConfig = toml::from_str(&contents).expect("parse bench config");
    config.bench
}

fn load_bench_config() -> (VisionTrainingConfig, BenchSettings) {
    let base_path = PathBuf::from("config").join("vision/lejepa/tiny.toml");
    let bench_path = PathBuf::from("config").join("vision/lejepa/bench.toml");
    let config =
        load_vision_training_config(&[base_path, bench_path.clone()]).expect("load bench config");
    let bench = load_bench_settings(&bench_path);
    (config, bench)
}

fn build_train_dataset(config: &VisionTrainingConfig) -> Option<Arc<ImageNetDataset>> {
    let VisionTrainingModeConfig::Lejepa(lejepa) = &config.mode else {
        eprintln!("vision_lejepa_train_step bench skipped: config is not lejepa mode");
        return None;
    };
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_aug = ImageNetAugmentations::new(
        ImageNetSplit::Train,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let multi_crop = lejepa.global_views > 0 || lejepa.local_views > 0;
    let global_views = if multi_crop {
        lejepa.global_views.max(1)
    } else {
        lejepa.views.max(1)
    };
    let local_views = if multi_crop { lejepa.local_views } else { 0 };
    let local_train_aug = if local_views > 0 {
        if lejepa.local_image_size == 0 {
            eprintln!("vision_lejepa_train_step bench skipped: local_image_size is 0");
            return None;
        }
        Some(ImageNetAugmentations::new(
            ImageNetSplit::Train,
            lejepa.local_image_size,
            lejepa.local_image_size,
            lejepa.local_min_scale,
            lejepa.local_max_scale,
            config.augment.min_aspect_ratio,
            config.augment.max_aspect_ratio,
            config.augment.flip_prob,
            config.augment.color_jitter_prob,
            config.augment.brightness,
            config.augment.contrast,
            config.augment.saturation,
            config.augment.hue,
            config.augment.grayscale_prob,
            config.augment.blur_prob,
            config.augment.blur_sigma_min,
            config.augment.blur_sigma_max,
            config.augment.solarize_prob,
            config.augment.solarize_threshold,
        ))
    } else {
        None
    };
    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let dataset = ImageNetDataset::new(ImageNetDatasetConfig {
        root: train_root,
        split: ImageNetSplit::Train,
        max_records: config.dataset.max_records,
        augmentations: train_aug,
        local_augmentations: local_train_aug,
        normalize,
        teacher: None,
        views: global_views,
        local_views,
        min_view_overlap: 0.0,
        view_overlap_attempts: 1,
        cache_decoded: config.dataset.cache_decoded,
        cache_capacity: config.dataset.cache_capacity,
        cache_preprocessed: config.dataset.cache_preprocessed,
    });
    match dataset {
        Ok(dataset) => Some(Arc::new(dataset)),
        Err(err) => {
            eprintln!("vision_lejepa_train_step bench skipped: {err}");
            None
        }
    }
}

fn init_wgpu_runtime(device: &WgpuDevice, config: &WgpuRuntimeConfig) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        init_runtime(device, config);
    });
}

fn vision_lejepa_train_step_bench(c: &mut Criterion) {
    let (config, bench) = load_bench_config();
    let Some(dataset) = build_train_dataset(&config) else {
        return;
    };

    type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    run_backend::<Autodiff<WgpuBackend>, _>(c, "wgpu", &config, Arc::clone(&dataset), |device| {
        init_wgpu_runtime(device, &config.wgpu);
    });

    #[cfg(feature = "cuda")]
    if bench.include_cuda {
        run_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", &config, dataset, |_| {});
    }
}

fn run_backend<B, Init>(
    c: &mut Criterion,
    name: &'static str,
    config: &VisionTrainingConfig,
    dataset: Arc<ImageNetDataset>,
    init_backend: Init,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + Send + Sync + 'static,
    Init: Fn(&B::Device),
{
    let VisionTrainingModeConfig::Lejepa(lejepa_cfg) = &config.mode else {
        eprintln!("vision_lejepa_train_step bench skipped: config is not lejepa mode");
        return;
    };

    let device = B::Device::default();
    B::seed(&device, 7);
    init_backend(&device);

    let training = config.training.clone();
    let optimizer_cfg = config.optimizer.clone();
    let vision_cfg = config.vision.build();
    let num_classes = dataset.num_classes();

    let steps_per_epoch = dataset.steps_per_epoch(training.batch_size);
    let loader: Arc<dyn burn::data::dataloader::DataLoader<B, ImageNetBatch<B>>> =
        Arc::new(ImageNetDataLoader::<B>::new(
            Arc::clone(&dataset),
            training.batch_size,
            &device,
            steps_per_epoch,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            config.dataset.prefetch_to_device,
        ));

    {
        let mut warm_bench = VisionLejepaTrainStepBench::<B>::new(
            vision_cfg.clone(),
            lejepa_cfg.clone(),
            &training,
            &optimizer_cfg,
            num_classes,
            &device,
        )
        .expect("warm bench");
        let mut warm_iter = loader.iter();
        if let Some(batch) = warm_iter.next() {
            let _ = warm_bench.train_step(batch);
        }
    }

    let mut group = c.benchmark_group(format!("vision_lejepa_train_step/{name}"));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::from_parameter("vision_lejepa_tiny"),
        &vision_cfg,
        |b, _| {
            let loader = Arc::clone(&loader);
            let vision_cfg = vision_cfg.clone();
            let lejepa_cfg = lejepa_cfg.clone();
            let training = training.clone();
            let optimizer_cfg = optimizer_cfg.clone();
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                let mut bench = VisionLejepaTrainStepBench::<B>::new(
                    vision_cfg.clone(),
                    lejepa_cfg.clone(),
                    &training,
                    &optimizer_cfg,
                    num_classes,
                    &device,
                )
                .expect("bench init");
                let mut iter = loader.iter();
                for _ in 0..iters {
                    let batch = match iter.next() {
                        Some(batch) => batch,
                        None => {
                            iter = loader.iter();
                            iter.next().expect("batch restart")
                        }
                    };
                    let start = Instant::now();
                    let _ = bench.train_step(batch);
                    total += start.elapsed();
                }
                total
            });
        },
    );
    group.finish();
}

criterion_group!(benches, vision_lejepa_train_step_bench);
criterion_main!(benches);
