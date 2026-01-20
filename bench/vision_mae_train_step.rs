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
    vision::train::bench::VisionMaeTrainStepBench,
};
use burn_ndarray::NdArray;
#[cfg(feature = "cli")]
use burn_wgpu::{CubeBackend, WgpuDevice, WgpuRuntime};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use serde::Deserialize;

#[cfg(feature = "cuda")]
use burn_cuda::Cuda;
#[cfg(feature = "cli")]
use burn_dragon::train::wgpu::init_runtime;

#[derive(Debug, Default, Deserialize)]
struct BenchSettings {
    #[serde(default)]
    include_cuda: bool,
    #[serde(default)]
    include_wgpu: bool,
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

fn load_bench_config(name: &str) -> Option<(VisionTrainingConfig, BenchSettings)> {
    let mut base_path = PathBuf::from("config").join(name);
    if base_path.extension().is_none() {
        base_path.set_extension("toml");
    }
    if !base_path.is_file() {
        return None;
    }
    let bench_path = base_path
        .parent()
        .map(|dir| dir.join("bench.toml"))
        .unwrap_or_else(|| PathBuf::from("config").join("bench.toml"));
    let config_paths = if bench_path.is_file() {
        vec![base_path, bench_path.clone()]
    } else {
        vec![base_path]
    };
    let config = load_vision_training_config(&config_paths).expect("load bench config");
    let bench = if bench_path.is_file() {
        load_bench_settings(&bench_path)
    } else {
        BenchSettings::default()
    };
    Some((config, bench))
}

fn bench_config_names() -> Vec<String> {
    if let Ok(names) = std::env::var("VISION_BENCH_CONFIGS") {
        let mut configs = Vec::new();
        for name in names.split(&[',', ';'][..]) {
            let trimmed = name.trim();
            if !trimmed.is_empty() {
                configs.push(trimmed.to_string());
            }
        }
        if !configs.is_empty() {
            return configs;
        }
    }
    vec!["vision/mae/tiny".to_string(), "vision/croco/tiny".to_string()]
}

fn build_train_dataset(config: &VisionTrainingConfig) -> Option<Arc<ImageNetDataset>> {
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
    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let (views, min_view_overlap, view_overlap_attempts) = match &config.mode {
        VisionTrainingModeConfig::Mae(mae) if mae.cross_view.enabled => (
            config.vision.num_eyes.max(1),
            mae.cross_view.min_overlap.max(0.0),
            mae.cross_view.max_attempts.max(1),
        ),
        _ => (1, 0.0, 1),
    };
    let dataset = ImageNetDataset::new(ImageNetDatasetConfig {
        root: train_root,
        split: ImageNetSplit::Train,
        max_records: config.dataset.max_records,
        augmentations: train_aug,
        local_augmentations: None,
        normalize,
        teacher: None,
        views,
        local_views: 0,
        min_view_overlap,
        view_overlap_attempts,
        cache_decoded: config.dataset.cache_decoded,
        cache_capacity: config.dataset.cache_capacity,
        cache_preprocessed: config.dataset.cache_preprocessed,
    });
    match dataset {
        Ok(dataset) => Some(Arc::new(dataset)),
        Err(err) => {
            eprintln!("vision_mae_train_step bench skipped: {err}");
            None
        }
    }
}

#[cfg(feature = "cli")]
fn init_wgpu_runtime(device: &WgpuDevice, config: &WgpuRuntimeConfig) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        init_runtime(device, config);
    });
}

fn vision_mae_train_step_bench(c: &mut Criterion) {
    let configs = bench_config_names();
    for name in configs {
        let Some((config, bench)) = load_bench_config(&name) else {
            continue;
        };
        let include_cuda = bench.include_cuda;
        let include_wgpu = bench.include_wgpu;
        let _ = (include_cuda, include_wgpu);
        let Some(dataset) = build_train_dataset(&config) else {
            continue;
        };

        run_backend::<Autodiff<NdArray<f32>>, _>(
            c,
            "cpu",
            &name,
            &config,
            Arc::clone(&dataset),
            |_| {},
        );

        #[cfg(feature = "cli")]
        if include_wgpu {
            type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
            run_backend::<Autodiff<WgpuBackend>, _>(
                c,
                "wgpu",
                &name,
                &config,
                Arc::clone(&dataset),
                |device| {
                    init_wgpu_runtime(device, &config.wgpu);
                },
            );
        }

        #[cfg(feature = "cuda")]
        if include_cuda {
            run_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", &name, &config, dataset, |_| {});
        }
    }
}

fn run_backend<B, Init>(
    c: &mut Criterion,
    name: &'static str,
    config_name: &str,
    config: &VisionTrainingConfig,
    dataset: Arc<ImageNetDataset>,
    init_backend: Init,
) where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone + Send + Sync + 'static,
    Init: Fn(&B::Device),
{
    let VisionTrainingModeConfig::Mae(mae_cfg) = &config.mode else {
        eprintln!("vision_mae_train_step bench skipped: config is not mae mode");
        return;
    };

    let device = B::Device::default();
    B::seed(&device, 7);
    init_backend(&device);

    let training = config.training.clone();
    let optimizer_cfg = config.optimizer.clone();
    let vision_cfg = config.vision.build();

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
        let mut warm_bench = VisionMaeTrainStepBench::<B>::new(
            vision_cfg.clone(),
            mae_cfg.clone(),
            &training,
            &optimizer_cfg,
            &device,
        )
        .expect("warm bench");
        let mut warm_iter = loader.iter();
        if let Some(batch) = warm_iter.next() {
            let _ = warm_bench.train_step(batch);
        }
    }

    let mut group = c.benchmark_group(format!("vision_mae_train_step/{name}"));
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(10);
    group.bench_with_input(
        BenchmarkId::from_parameter(config_name),
        &vision_cfg,
        |b, _| {
            let loader = Arc::clone(&loader);
            let vision_cfg = vision_cfg.clone();
            let mae_cfg = mae_cfg.clone();
            let training = training.clone();
            let optimizer_cfg = optimizer_cfg.clone();
            b.iter_custom(|iters| {
                let mut total = Duration::ZERO;
                let mut bench = VisionMaeTrainStepBench::<B>::new(
                    vision_cfg.clone(),
                    mae_cfg.clone(),
                    &training,
                    &optimizer_cfg,
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

criterion_group!(benches, vision_mae_train_step_bench);
criterion_main!(benches);
