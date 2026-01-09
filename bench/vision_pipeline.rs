#![recursion_limit = "256"]

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "train", feature = "benchmark"))]
mod vision_bench {
    use super::*;
    use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
    use burn::tensor::Tensor;
    use burn_autodiff::Autodiff;
    use burn_dragon_hatchling::{
        ImageNetAugmentations, ImageNetSplit, VisionAugmentationConfig, VisionDragonHatchlingConfig,
        VisionFoveaSamplingMode, VisionPyramidMode, VisionSaccadeConfig, VisionNormalize,
    };
    use burn_dragon_hatchling::train::bench::VisionSaccadeBench;
    use burn_wgpu::Wgpu;
    use image::{DynamicImage, RgbImage};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use std::hint::black_box;

    #[cfg(feature = "cuda")]
    use burn_cuda::Cuda;

    #[derive(Clone, Copy)]
    struct VisionBenchConfig {
        name: &'static str,
        batch: usize,
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        steps: usize,
        mip_levels: usize,
    }

    const VISION_CONFIGS: &[VisionBenchConfig] = &[
        VisionBenchConfig {
            name: "b16_img160_p32_d128",
            batch: 16,
            image_size: 160,
            patch_size: 32,
            embed_dim: 128,
            steps: 4,
            mip_levels: 3,
        },
        VisionBenchConfig {
            name: "b32_img160_p32_d128",
            batch: 32,
            image_size: 160,
            patch_size: 32,
            embed_dim: 128,
            steps: 4,
            mip_levels: 3,
        },
    ];

    pub fn vision_pipeline_bench(c: &mut Criterion) {
        run_vision_backend::<Autodiff<Wgpu<f32>>, _>(c, "wgpu", |device| {
            #[cfg(feature = "cli")]
            {
                burn_dragon_hatchling::wgpu::init_runtime(device);
            }
        });

        #[cfg(feature = "cuda")]
        run_vision_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", |_| {});
    }

    fn run_vision_backend<B, Init>(c: &mut Criterion, name: &'static str, init: Init)
    where
        B: AutodiffBackend + Clone + 'static,
        Init: Fn(&<B as BackendTrait>::Device),
    {
        let device = <B as BackendTrait>::Device::default();
        <B as BackendTrait>::seed(&device, 7);
        init(&device);

        let mut group = c.benchmark_group(format!("vision_saccade_pipeline/{name}"));
        for cfg in VISION_CONFIGS {
            let vision = VisionDragonHatchlingConfig {
                image_size: cfg.image_size,
                patch_size: cfg.patch_size,
                in_channels: 3,
                embed_dim: cfg.embed_dim,
                steps: cfg.steps,
                n_head: 4,
                mlp_internal_dim_multiplier: 4,
                dropout: 0.1,
                projection_dim: 128,
                projection_hidden_dim: 256,
                use_cls_token: true,
                pos_encoding: burn_dragon_hatchling::SpatialPositionalEncodingKind::Learned2d,
                pos_max_height: cfg.image_size / cfg.patch_size,
                pos_max_width: cfg.image_size / cfg.patch_size,
                attention_mode: burn_dragon_hatchling::VisionAttentionMode::RowL1,
                fused_kernels: burn_dragon_hatchling::FusedKernelConfig::default(),
            };
            let saccade_base = VisionSaccadeConfig {
                num_eyes: 2,
                mip_levels: cfg.mip_levels,
                pyramid_mode: VisionPyramidMode::Laplacian,
                low_mem_pre_rollout: true,
                ..VisionSaccadeConfig::default()
            };
            let bench =
                VisionSaccadeBench::<B>::new(vision.clone(), saccade_base.clone(), cfg.batch, cfg.steps, 1, &device);
            let estimate = bench.fovea_patch_kernel_estimate();
            eprintln!(
                "[{name}:{cfg_name}] fovea_patch grid_sample calls={calls} (unfused={unfused}) levels={levels} subsamples={subsamples}",
                name = name,
                cfg_name = cfg.name,
                calls = estimate.grid_sample_calls,
                unfused = estimate.unfused_grid_sample_calls,
                levels = estimate.levels,
                subsamples = estimate.subsamples
            );

            bench_stage(&mut group, cfg, "patch_embed", &bench, VisionSaccadeBench::stage_patch_embed);
            bench_stage(&mut group, cfg, "mip_pyramid", &bench, VisionSaccadeBench::stage_mip_pyramid);
            bench_stage(&mut group, cfg, "fovea_weights", &bench, VisionSaccadeBench::stage_fovea_weights);
            bench_stage(&mut group, cfg, "fovea_context", &bench, VisionSaccadeBench::stage_fovea_context);
            bench_stage(&mut group, cfg, "token_forward", &bench, VisionSaccadeBench::stage_token_forward);
            bench_stage(&mut group, cfg, "residual_scatter", &bench, VisionSaccadeBench::stage_residual_scatter);
            bench_stage(&mut group, cfg, "full_forward", &bench, VisionSaccadeBench::stage_full_forward);
            if name != "wgpu" {
                bench_stage(&mut group, cfg, "full_backward", &bench, VisionSaccadeBench::stage_full_backward);
            }

            let sampling_modes = [
                ("batched", VisionFoveaSamplingMode::Batched),
                ("sequential", VisionFoveaSamplingMode::Sequential),
                ("subpatch", VisionFoveaSamplingMode::Subpatch),
                ("cubecl", VisionFoveaSamplingMode::Cubecl),
            ];
            for (sampling_name, sampling_mode) in sampling_modes {
                let mut saccade = saccade_base.clone();
                saccade.fovea_sampling_mode = sampling_mode;
                saccade.fovea_subpatch_size = if matches!(
                    sampling_mode,
                    VisionFoveaSamplingMode::Subpatch
                ) {
                    (cfg.patch_size / 2).max(1)
                } else {
                    0
                };
                let bench = VisionSaccadeBench::<B>::new(
                    vision.clone(),
                    saccade,
                    cfg.batch,
                    cfg.steps,
                    1,
                    &device,
                );
                let stage = format!("fovea_patch/{sampling_name}");
                bench_stage(&mut group, cfg, &stage, &bench, VisionSaccadeBench::stage_fovea_patch);
            }
        }
        group.finish();
    }

    fn bench_stage<B, F>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        cfg: &VisionBenchConfig,
        stage: &str,
        bench: &VisionSaccadeBench<B>,
        mut func: F,
    ) where
        B: AutodiffBackend + Clone + 'static,
        F: FnMut(&VisionSaccadeBench<B>) -> Tensor<B, 1>,
    {
        group.bench_with_input(
            BenchmarkId::new(stage, cfg.name),
            cfg,
            |b, _cfg| {
                b.iter_custom(|iters| {
                    let mut total = Duration::ZERO;
                    for _ in 0..iters {
                        let start = Instant::now();
                        let value = func(bench);
                        force_sync(value);
                        total += start.elapsed();
                    }
                    total
                });
            },
        );
    }

    fn force_sync<B: BackendTrait>(value: Tensor<B, 1>) {
        // Force lazy execution to finish before timing ends.
        let _ = value.to_data();
    }


    pub fn vision_augment_bench(c: &mut Criterion) {
        let config = VisionAugmentationConfig::default();
        let augment = ImageNetAugmentations::new(
            ImageNetSplit::Train,
            config.image_size,
            config.resize_short,
            config.min_scale,
            config.max_scale,
            config.min_aspect_ratio,
            config.max_aspect_ratio,
            config.flip_prob,
            config.color_jitter_prob,
            config.brightness,
            config.contrast,
            config.saturation,
            config.hue,
            config.grayscale_prob,
            config.blur_prob,
            config.blur_sigma_min,
            config.blur_sigma_max,
            config.solarize_prob,
            config.solarize_threshold,
        );
        let normalize = VisionNormalize::new(config.normalize_mean, config.normalize_std);
        let base = RgbImage::from_fn(config.image_size as u32, config.image_size as u32, |x, y| {
            let r = ((x + y) % 255) as u8;
            let g = (x % 255) as u8;
            let b = (y % 255) as u8;
            image::Rgb([r, g, b])
        });
        let mut group = c.benchmark_group("vision_augment_pipeline");
        group.bench_function("augment_train", |b| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                let view = augment.apply(DynamicImage::ImageRgb8(base.clone()), &mut rng);
                black_box(view);
            });
        });
        group.bench_function("normalize", |b| {
            b.iter(|| {
                let mut buffer = Vec::with_capacity(config.image_size * config.image_size * 3);
                normalize.apply(&base, &mut buffer);
                black_box(buffer);
            });
        });
        group.finish();
    }
}

#[cfg(all(feature = "train", feature = "benchmark"))]
criterion_group!(
    benches,
    vision_bench::vision_pipeline_bench,
    vision_bench::vision_augment_bench
);
#[cfg(all(feature = "train", feature = "benchmark"))]
criterion_main!(benches);

#[cfg(not(all(feature = "train", feature = "benchmark")))]
fn main() {
    eprintln!("vision_pipeline benchmarks require --features train,benchmark");
}
