#![recursion_limit = "256"]

#[cfg(all(feature = "train", feature = "benchmark"))]
use std::time::{Duration, Instant};

#[cfg(all(feature = "train", feature = "benchmark"))]
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};

#[cfg(all(feature = "train", feature = "benchmark"))]
mod vision_bench {
    use super::*;
    use burn::tensor::Tensor;
    use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
    use burn_autodiff::Autodiff;
    use burn_dragon_hatchling::vision::train::bench::{VisionSaccadeBench, VisionScatterBench};
    use burn_dragon_hatchling::{
        ImageNetAugmentations, ImageNetSplit, ManifoldHyperConnectionsConfig,
        VisionAugmentationConfig, VisionDragonHatchlingConfig, VisionFoveaSamplingMode,
        VisionFoveaScatterMode, VisionFoveaWarpMode, VisionLatentActivation, VisionNormalize,
        VisionPyramidMode, VisionSaccadeConfig, WgpuRuntimeConfig, wgpu::init_runtime,
    };
    use burn_dragon_hatchling_vision::FOVEATION_SHADER;
    use burn_dragon_hatchling_vision::foveation;
    use burn_wgpu::{Wgpu, WgpuDevice};
    use bytemuck::{Pod, Zeroable};
    use half::f16;
    use image::{DynamicImage, RgbImage};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use serde::Deserialize;
    use std::fs;
    use std::hint::black_box;
    use std::path::PathBuf;
    use std::sync::Once;
    use wgpu::util::DeviceExt;

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

    const SAMPLING_MODES_FULL: &[(&str, VisionFoveaSamplingMode)] = &[
        ("batched", VisionFoveaSamplingMode::Batched),
        ("sequential", VisionFoveaSamplingMode::Sequential),
        ("subpatch", VisionFoveaSamplingMode::Subpatch),
        ("cubecl", VisionFoveaSamplingMode::Cubecl),
        ("wgsl", VisionFoveaSamplingMode::Wgsl),
    ];
    const SAMPLING_MODES_QUICK: &[(&str, VisionFoveaSamplingMode)] =
        &[("batched", VisionFoveaSamplingMode::Batched)];
    const WARP_MODES_FULL: &[(&str, VisionFoveaWarpMode)] = &[
        ("warped", VisionFoveaWarpMode::Warped),
        ("patched", VisionFoveaWarpMode::Patched),
    ];
    const WARP_MODES_QUICK: &[(&str, VisionFoveaWarpMode)] = &[
        ("warped", VisionFoveaWarpMode::Warped),
        ("patched", VisionFoveaWarpMode::Patched),
    ];
    const BASELINE_WARP_MODES_FULL: &[(&str, foveation::FoveaWarpMode)] = &[
        ("warped", foveation::FoveaWarpMode::Warped),
        ("patched", foveation::FoveaWarpMode::Patched),
    ];
    const BASELINE_WARP_MODES_QUICK: &[(&str, foveation::FoveaWarpMode)] = &[
        ("warped", foveation::FoveaWarpMode::Warped),
        ("patched", foveation::FoveaWarpMode::Patched),
    ];

    #[derive(Debug, Default, Deserialize)]
    struct BenchSettings {
        #[serde(default)]
        full: bool,
        #[serde(default)]
        include_cuda: bool,
    }

    #[derive(Debug, Default, Deserialize)]
    struct BenchConfig {
        #[serde(default)]
        bench: BenchSettings,
        #[serde(default)]
        wgpu: WgpuRuntimeConfig,
    }

    fn load_bench_config() -> BenchConfig {
        let path = PathBuf::from("config").join("vision_pipeline_bench.toml");
        let contents = fs::read_to_string(&path).expect("read bench config");
        toml::from_str(&contents).expect("parse bench config")
    }

    struct BenchProfile {
        configs: &'static [VisionBenchConfig],
        sampling_modes: &'static [(&'static str, VisionFoveaSamplingMode)],
        warp_modes: &'static [(&'static str, VisionFoveaWarpMode)],
        baseline_warp_modes: &'static [(&'static str, foveation::FoveaWarpMode)],
        warm_up: Duration,
        measurement: Duration,
        sample_size: usize,
        include_full: bool,
        include_cuda: bool,
    }

    fn bench_profile(settings: &BenchSettings) -> BenchProfile {
        let full = settings.full;
        let include_cuda = settings.include_cuda;
        BenchProfile {
            configs: if full {
                VISION_CONFIGS
            } else {
                &VISION_CONFIGS[..1]
            },
            sampling_modes: if full {
                SAMPLING_MODES_FULL
            } else {
                SAMPLING_MODES_QUICK
            },
            warp_modes: if full {
                WARP_MODES_FULL
            } else {
                WARP_MODES_QUICK
            },
            baseline_warp_modes: if full {
                BASELINE_WARP_MODES_FULL
            } else {
                BASELINE_WARP_MODES_QUICK
            },
            warm_up: if full {
                Duration::from_secs(3)
            } else {
                Duration::from_secs(1)
            },
            measurement: if full {
                Duration::from_secs(5)
            } else {
                Duration::from_secs(2)
            },
            sample_size: if full { 50 } else { 10 },
            include_full: full,
            include_cuda,
        }
    }

    const FOVEATION_SHADER_SOURCE: &str = FOVEATION_SHADER;
    const FOVEATION_WORKGROUP_SIZE: u32 = 8;

    #[repr(C, align(8))]
    #[derive(Clone, Copy, Default, Pod, Zeroable)]
    struct AlignedVec2 {
        x: f32,
        y: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Default, Pod, Zeroable)]
    struct FoveationUniform {
        image_size: AlignedVec2,
        inv_image_size: AlignedVec2,
        center: AlignedVec2,
        sigma: AlignedVec2,
        sample_scale: f32,
        lod_sigma: f32,
        patch_size: f32,
        pyramid_levels: u32,
        mode: u32,
        warp_mode: u32,
        _pad0: u32,
        _pad1: u32,
    }

    pub fn vision_pipeline_bench(c: &mut Criterion) {
        let bench_config = load_bench_config();
        let profile = bench_profile(&bench_config.bench);
        let wgpu_config = bench_config.wgpu;
        bench_foveation_baselines(c, &profile);
        bench_scatter_modes(c, &profile, &wgpu_config);
        run_vision_backend::<Autodiff<Wgpu<f32>>, _>(c, "wgpu", &profile, |device| {
            init_wgpu_runtime(device, &wgpu_config);
        });

        #[cfg(feature = "cuda")]
        if profile.include_cuda {
            run_vision_backend::<Autodiff<Cuda<f32>>, _>(c, "cuda", &profile, |_| {});
        }
    }

    fn init_wgpu_runtime(device: &WgpuDevice, config: &WgpuRuntimeConfig) {
        static INIT: Once = Once::new();
        INIT.call_once(|| {
            init_runtime(device, config);
        });
    }

    fn run_vision_backend<B, Init>(
        c: &mut Criterion,
        name: &'static str,
        profile: &BenchProfile,
        init: Init,
    ) where
        B: AutodiffBackend + Clone + 'static,
        Init: Fn(&<B as BackendTrait>::Device),
    {
        let device = <B as BackendTrait>::Device::default();
        <B as BackendTrait>::seed(&device, 7);
        init(&device);

        let mut group = c.benchmark_group(format!("vision_saccade_pipeline/{name}"));
        group.warm_up_time(profile.warm_up);
        group.measurement_time(profile.measurement);
        group.sample_size(profile.sample_size);
        for cfg in profile.configs {
            let bench_steps = if profile.include_full {
                cfg.steps
            } else {
                cfg.steps.min(2).max(1)
            };
            let vision = VisionDragonHatchlingConfig {
                image_size: cfg.image_size,
                patch_size: cfg.patch_size,
                patch_embed_mode: burn_dragon_hatchling::VisionPatchEmbedMode::default(),
                in_channels: 3,
                embed_dim: cfg.embed_dim,
                steps: bench_steps,
                n_head: 4,
                mlp_internal_dim_multiplier: 4,
                dropout: 0.1,
                projection_dim: 128,
                projection_hidden_dim: 256,
                use_cls_token: true,
                cls_sync_alpha: 0.0,
                num_eyes: 2,
                cross_eye_steps: 0,
                token_state_norm: true,
                latent_activation: VisionLatentActivation::default(),
                pos_encoding: burn_dragon_hatchling::SpatialPositionalEncodingKind::Learned2d,
                pos_max_height: cfg.image_size / cfg.patch_size,
                pos_max_width: cfg.image_size / cfg.patch_size,
                attention_mode: burn_dragon_hatchling::VisionAttentionMode::RowL1,
                fused_kernels: burn_dragon_hatchling::FusedKernelConfig::default(),
                mhc: ManifoldHyperConnectionsConfig::default(),
            };
            let saccade_base = VisionSaccadeConfig {
                num_eyes: 2,
                mip_levels: cfg.mip_levels,
                pyramid_mode: VisionPyramidMode::Laplacian,
                low_mem_pre_rollout: true,
                ..VisionSaccadeConfig::default()
            };
            let bench = VisionSaccadeBench::<B>::new(
                vision.clone(),
                saccade_base.clone(),
                cfg.batch,
                bench_steps,
                1,
                &device,
            );
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

            bench_stage(
                &mut group,
                cfg,
                "patch_embed",
                &bench,
                VisionSaccadeBench::stage_patch_embed,
            );
            bench_stage(
                &mut group,
                cfg,
                "mip_pyramid",
                &bench,
                VisionSaccadeBench::stage_mip_pyramid,
            );
            bench_stage(
                &mut group,
                cfg,
                "fovea_weights",
                &bench,
                VisionSaccadeBench::stage_fovea_weights,
            );
            bench_stage(
                &mut group,
                cfg,
                "fovea_context",
                &bench,
                VisionSaccadeBench::stage_fovea_context,
            );
            bench_stage(
                &mut group,
                cfg,
                "gdpo_advantage",
                &bench,
                VisionSaccadeBench::stage_gdpo_advantage,
            );
            bench_stage(
                &mut group,
                cfg,
                "gdpo_policy_loss",
                &bench,
                VisionSaccadeBench::stage_gdpo_policy_loss,
            );
            bench_stage(
                &mut group,
                cfg,
                "token_forward",
                &bench,
                VisionSaccadeBench::stage_token_forward,
            );
            bench_stage(
                &mut group,
                cfg,
                "residual_scatter",
                &bench,
                VisionSaccadeBench::stage_residual_scatter,
            );
            if profile.include_full {
                bench_stage(
                    &mut group,
                    cfg,
                    "full_forward",
                    &bench,
                    VisionSaccadeBench::stage_full_forward,
                );
                if name != "wgpu" {
                    bench_stage(
                        &mut group,
                        cfg,
                        "full_backward",
                        &bench,
                        VisionSaccadeBench::stage_full_backward,
                    );
                }
            }

            for &(sampling_name, sampling_mode) in profile.sampling_modes {
                for &(warp_label, warp_mode) in profile.warp_modes {
                    let mut saccade = saccade_base.clone();
                    saccade.fovea_sampling_mode = sampling_mode;
                    saccade.fovea_warp_mode = warp_mode;
                    saccade.fovea_subpatch_size =
                        if matches!(sampling_mode, VisionFoveaSamplingMode::Subpatch) {
                            (cfg.patch_size / 2).max(1)
                        } else {
                            0
                        };
                    let bench = VisionSaccadeBench::<B>::new(
                        vision.clone(),
                        saccade,
                        cfg.batch,
                        bench_steps,
                        1,
                        &device,
                    );
                    let stage = format!("fovea_patch/{sampling_name}/{warp_label}");
                    bench_stage(
                        &mut group,
                        cfg,
                        &stage,
                        &bench,
                        VisionSaccadeBench::stage_fovea_patch,
                    );
                }
            }
        }
        group.finish();
    }

    fn bench_scatter_modes(
        c: &mut Criterion,
        profile: &BenchProfile,
        wgpu_config: &WgpuRuntimeConfig,
    ) {
        let device = WgpuDevice::default();
        init_wgpu_runtime(&device, wgpu_config);
        <Wgpu<f32> as BackendTrait>::seed(&device, 7);

        let mut group = c.benchmark_group("vision_scatter/wgpu");
        group.warm_up_time(profile.warm_up);
        group.measurement_time(profile.measurement);
        group.sample_size(profile.sample_size);
        for cfg in profile.configs {
            let vision = VisionDragonHatchlingConfig {
                image_size: cfg.image_size,
                patch_size: cfg.patch_size,
                patch_embed_mode: burn_dragon_hatchling::VisionPatchEmbedMode::default(),
                in_channels: 3,
                embed_dim: cfg.embed_dim,
                steps: cfg.steps,
                n_head: 4,
                mlp_internal_dim_multiplier: 4,
                dropout: 0.1,
                projection_dim: 128,
                projection_hidden_dim: 256,
                use_cls_token: true,
                cls_sync_alpha: 0.0,
                num_eyes: 1,
                cross_eye_steps: 0,
                token_state_norm: true,
                latent_activation: VisionLatentActivation::default(),
                pos_encoding: burn_dragon_hatchling::SpatialPositionalEncodingKind::Learned2d,
                pos_max_height: cfg.image_size / cfg.patch_size,
                pos_max_width: cfg.image_size / cfg.patch_size,
                attention_mode: burn_dragon_hatchling::VisionAttentionMode::RowL1,
                fused_kernels: burn_dragon_hatchling::FusedKernelConfig::default(),
                mhc: ManifoldHyperConnectionsConfig::default(),
            };
            let grid = (cfg.image_size / cfg.patch_size).max(1);
            let out_tokens = grid * grid;
            let in_tokens = 1;

            let scatter_modes = [
                ("tensor", VisionFoveaScatterMode::Tensor),
                ("cubecl", VisionFoveaScatterMode::Cubecl),
                ("wgsl", VisionFoveaScatterMode::Wgsl),
            ];
            for (label, mode) in scatter_modes {
                let mut saccade = VisionSaccadeConfig {
                    num_eyes: 2,
                    mip_levels: cfg.mip_levels,
                    pyramid_mode: VisionPyramidMode::Laplacian,
                    low_mem_pre_rollout: true,
                    ..VisionSaccadeConfig::default()
                };
                saccade.fovea_scatter_mode = mode;
                let feature_dim = saccade
                    .pyramid_feature_dim
                    .filter(|&value| value > 0)
                    .unwrap_or(cfg.embed_dim);
                let bench = VisionScatterBench::<Wgpu<f32>>::new(
                    vision.clone(),
                    saccade,
                    cfg.batch,
                    out_tokens,
                    in_tokens,
                    feature_dim,
                    &device,
                );
                let stage = format!("scatter/{label}");
                bench_scatter_stage(
                    &mut group,
                    cfg,
                    &stage,
                    &bench,
                    VisionScatterBench::stage_scatter,
                );
            }
        }
        group.finish();
    }

    fn bench_foveation_baselines(c: &mut Criterion, profile: &BenchProfile) {
        let mut group = c.benchmark_group("foveation_baseline");
        group.warm_up_time(profile.warm_up);
        group.measurement_time(profile.measurement);
        group.sample_size(profile.sample_size);
        for cfg in profile.configs {
            let base = make_cpu_image(cfg.image_size, cfg.image_size);
            let cache = foveation::build_pyramid_cache(
                base,
                cfg.mip_levels.max(1),
                foveation::PyramidMode::Laplacian,
            );

            let mean = [0.5f32, 0.5f32];
            let radius_norm = foveation::sigma_from_unit(0.6);
            let sigma_norm = (radius_norm * 0.6).clamp(1e-3, radius_norm);

            for &(warp_label, warp_mode) in profile.baseline_warp_modes {
                let uniform = make_foveation_uniform(
                    cfg,
                    &cache,
                    foveation::PyramidMode::Laplacian,
                    warp_mode,
                );
                let wgsl = WgslFoveationBench::new(cfg, &cache, uniform);

                group.bench_with_input(
                    BenchmarkId::new(format!("cpu_{warp_label}"), cfg.name),
                    cfg,
                    |b, cfg| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let start = Instant::now();
                                let mut acc = 0.0f32;
                                for _ in 0..cfg.batch {
                                    let patch = foveation::render_foveated_patch_with_radius(
                                        &cache,
                                        mean,
                                        sigma_norm,
                                        radius_norm,
                                        cfg.patch_size,
                                        warp_mode,
                                    );
                                    acc += patch.get(0).copied().unwrap_or(0.0);
                                }
                                black_box(acc);
                                total += start.elapsed();
                            }
                            total
                        });
                    },
                );

                group.bench_with_input(
                    BenchmarkId::new(format!("wgsl_{warp_label}"), cfg.name),
                    cfg,
                    |b, cfg| {
                        b.iter_custom(|iters| {
                            let mut total = Duration::ZERO;
                            for _ in 0..iters {
                                let start = Instant::now();
                                for _ in 0..cfg.batch {
                                    wgsl.run_once();
                                }
                                total += start.elapsed();
                            }
                            total
                        });
                    },
                );
            }
        }
        group.finish();
    }

    fn make_cpu_image(width: usize, height: usize) -> foveation::CpuImageLevel {
        let mut data = Vec::with_capacity(width * height * 3);
        let denom_w = (width - 1).max(1) as f32;
        let denom_h = (height - 1).max(1) as f32;
        for y in 0..height {
            for x in 0..width {
                let fx = x as f32 / denom_w;
                let fy = y as f32 / denom_h;
                let checker = ((x / 4 + y / 3) % 2) as f32;
                data.push(fx);
                data.push(fy);
                data.push(0.55 * fx + 0.35 * fy + 0.1 * checker);
            }
        }
        foveation::CpuImageLevel {
            width,
            height,
            data,
        }
    }

    fn make_foveation_uniform(
        cfg: &VisionBenchConfig,
        cache: &foveation::CpuPyramidCache,
        mode: foveation::PyramidMode,
        warp_mode: foveation::FoveaWarpMode,
    ) -> FoveationUniform {
        let width = cache.gaussian.first().map(|level| level.width).unwrap_or(1);
        let height = cache
            .gaussian
            .first()
            .map(|level| level.height)
            .unwrap_or(1);
        let min_dim = width.min(height).max(1) as f32;
        let radius_norm = foveation::sigma_from_unit(0.6);
        let sigma_norm = (radius_norm * 0.6).clamp(1e-3, radius_norm);
        let sigma_px = sigma_norm * min_dim;
        let radius_px = radius_norm * min_dim;
        let patch = cfg.patch_size.max(1) as f32;
        let sample_scale = (radius_px * 2.0) / patch;
        let lod_sigma = foveation::lod_sigma_from_sigma(sigma_norm);
        FoveationUniform {
            image_size: AlignedVec2 {
                x: width as f32,
                y: height as f32,
            },
            inv_image_size: AlignedVec2 {
                x: 1.0 / width.max(1) as f32,
                y: 1.0 / height.max(1) as f32,
            },
            center: AlignedVec2 {
                x: 0.5 * width as f32,
                y: 0.5 * height as f32,
            },
            sigma: AlignedVec2 {
                x: sigma_px.max(1e-3),
                y: sigma_px.max(1e-3),
            },
            sample_scale,
            lod_sigma,
            patch_size: patch,
            pyramid_levels: cfg.mip_levels.max(1) as u32,
            mode: match mode {
                foveation::PyramidMode::Stacked => 0,
                foveation::PyramidMode::Laplacian => 1,
            },
            warp_mode: match warp_mode {
                foveation::FoveaWarpMode::Warped => 0,
                foveation::FoveaWarpMode::Patched => 1,
            },
            _pad0: 0,
            _pad1: 0,
        }
    }

    struct WgslFoveationBench {
        device: wgpu::Device,
        queue: wgpu::Queue,
        pipeline: wgpu::ComputePipeline,
        bind_group: wgpu::BindGroup,
        output_texture: wgpu::Texture,
        output_buffer: wgpu::Buffer,
        output_size: u32,
        aligned_bytes_per_row: u32,
        _gaussian: wgpu::Texture,
        _residual: wgpu::Texture,
        _gaussian_view: wgpu::TextureView,
        _residual_view: wgpu::TextureView,
        _output_view: wgpu::TextureView,
        _uniform: wgpu::Buffer,
        _sampler: wgpu::Sampler,
    }

    impl WgslFoveationBench {
        fn new(
            cfg: &VisionBenchConfig,
            cache: &foveation::CpuPyramidCache,
            uniform: FoveationUniform,
        ) -> Self {
            let instance = wgpu::Instance::default();
            let adapter =
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
                .expect("wgpu adapter");
            let (device, queue) =
                pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::default(),
                    trace: wgpu::Trace::Off,
                }))
                .expect("wgpu device");

            let gaussian = create_mip_texture(&device, &queue, &cache.gaussian);
            let residual_levels = build_residual_levels(cache);
            let residual = create_mip_texture(&device, &queue, &residual_levels);
            let gaussian_view = gaussian.create_view(&wgpu::TextureViewDescriptor::default());
            let residual_view = residual.create_view(&wgpu::TextureViewDescriptor::default());

            let output_size = cfg.patch_size.max(1) as u32;
            let output_texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("foveation_output"),
                size: wgpu::Extent3d {
                    width: output_size,
                    height: output_size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba16Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            });
            let output_view = output_texture.create_view(&wgpu::TextureViewDescriptor::default());

            let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Linear,
                ..Default::default()
            });

            let shader_source = FOVEATION_SHADER_SOURCE.replace("rgba8unorm", "rgba16float");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("foveation_shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("foveation_uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

            let bind_group_layout =
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("foveation_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: wgpu::TextureFormat::Rgba16Float,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

            let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("foveation_bind_group"),
                layout: &bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&gaussian_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&residual_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::TextureView(&output_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: uniform_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("foveation_pipeline_layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("foveation_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

            let bytes_per_row = output_size * 8;
            let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
            let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("foveation_output_buffer"),
                size: (aligned_bytes_per_row as u64) * output_size as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            Self {
                device,
                queue,
                pipeline,
                bind_group,
                output_texture,
                output_buffer,
                output_size,
                aligned_bytes_per_row,
                _gaussian: gaussian,
                _residual: residual,
                _gaussian_view: gaussian_view,
                _residual_view: residual_view,
                _output_view: output_view,
                _uniform: uniform_buffer,
                _sampler: sampler,
            }
        }

        fn run_once(&self) {
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("foveation_encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("foveation_pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&self.pipeline);
                pass.set_bind_group(0, &self.bind_group, &[]);
                let groups_x =
                    (self.output_size + FOVEATION_WORKGROUP_SIZE - 1) / FOVEATION_WORKGROUP_SIZE;
                let groups_y =
                    (self.output_size + FOVEATION_WORKGROUP_SIZE - 1) / FOVEATION_WORKGROUP_SIZE;
                pass.dispatch_workgroups(groups_x.max(1), groups_y.max(1), 1);
            }
            encoder.copy_texture_to_buffer(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.output_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyBufferInfo {
                    buffer: &self.output_buffer,
                    layout: wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(self.aligned_bytes_per_row),
                        rows_per_image: Some(self.output_size),
                    },
                },
                wgpu::Extent3d {
                    width: self.output_size,
                    height: self.output_size,
                    depth_or_array_layers: 1,
                },
            );
            self.queue.submit(Some(encoder.finish()));
            let buffer_slice = self.output_buffer.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
                let _ = tx.send(res);
            });
            let _ = self.device.poll(wgpu::PollType::Wait);
            let _ = rx.recv();
            self.output_buffer.unmap();
        }
    }

    fn build_residual_levels(cache: &foveation::CpuPyramidCache) -> Vec<foveation::CpuImageLevel> {
        let mut levels = cache.laplacian.clone();
        let coarse = &cache.coarse;
        let data = vec![0.0; coarse.width * coarse.height * 3];
        levels.push(foveation::CpuImageLevel {
            width: coarse.width,
            height: coarse.height,
            data,
        });
        levels
    }

    fn create_mip_texture(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        levels: &[foveation::CpuImageLevel],
    ) -> wgpu::Texture {
        let base = levels.first().expect("mip levels");
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("foveation_mip_texture"),
            size: wgpu::Extent3d {
                width: base.width as u32,
                height: base.height as u32,
                depth_or_array_layers: 1,
            },
            mip_level_count: levels.len() as u32,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        for (level_idx, level) in levels.iter().enumerate() {
            let bytes = level_to_f16_bytes(level);
            let bytes_per_row = 8 * level.width as u32;
            let aligned_bytes_per_row = align_bytes_per_row(bytes_per_row);
            let padded = if aligned_bytes_per_row == bytes_per_row {
                bytes
            } else {
                pad_rows(
                    &bytes,
                    level.width as u32,
                    level.height as u32,
                    8,
                    aligned_bytes_per_row,
                )
            };
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &texture,
                    mip_level: level_idx as u32,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &padded,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(aligned_bytes_per_row),
                    rows_per_image: Some(level.height as u32),
                },
                wgpu::Extent3d {
                    width: level.width as u32,
                    height: level.height as u32,
                    depth_or_array_layers: 1,
                },
            );
        }
        texture
    }

    fn level_to_f16_bytes(level: &foveation::CpuImageLevel) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(level.width * level.height * 8);
        for idx in 0..(level.width * level.height) {
            let base = idx * 3;
            push_f16(&mut bytes, level.data[base]);
            push_f16(&mut bytes, level.data[base + 1]);
            push_f16(&mut bytes, level.data[base + 2]);
            push_f16(&mut bytes, 1.0);
        }
        bytes
    }

    fn push_f16(bytes: &mut Vec<u8>, value: f32) {
        let bits = f16::from_f32(value).to_bits();
        bytes.extend_from_slice(&bits.to_le_bytes());
    }

    fn align_bytes_per_row(bytes_per_row: u32) -> u32 {
        let align = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        ((bytes_per_row + align - 1) / align) * align
    }

    fn pad_rows(
        data: &[u8],
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        aligned_bytes_per_row: u32,
    ) -> Vec<u8> {
        let bytes_per_row = bytes_per_pixel * width;
        let mut padded = vec![0u8; (aligned_bytes_per_row * height) as usize];
        for row in 0..height {
            let src_start = (row * bytes_per_row) as usize;
            let src_end = src_start + bytes_per_row as usize;
            let dst_start = (row * aligned_bytes_per_row) as usize;
            let dst_end = dst_start + bytes_per_row as usize;
            padded[dst_start..dst_end].copy_from_slice(&data[src_start..src_end]);
        }
        padded
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
        group.bench_with_input(BenchmarkId::new(stage, cfg.name), cfg, |b, _cfg| {
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
        });
    }

    fn bench_scatter_stage<B, F>(
        group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
        cfg: &VisionBenchConfig,
        stage: &str,
        bench: &VisionScatterBench<B>,
        mut func: F,
    ) where
        B: BackendTrait + Clone + 'static,
        F: FnMut(&VisionScatterBench<B>) -> Tensor<B, 1>,
    {
        group.bench_with_input(BenchmarkId::new(stage, cfg.name), cfg, |b, _cfg| {
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
        });
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
        let base = RgbImage::from_fn(
            config.image_size as u32,
            config.image_size as u32,
            |x, y| {
                let r = ((x + y) % 255) as u8;
                let g = (x % 255) as u8;
                let b = (y % 255) as u8;
                image::Rgb([r, g, b])
            },
        );
        let mut group = c.benchmark_group("vision_augment_pipeline");
        group.bench_function("augment_train", |b| {
            let mut rng = StdRng::seed_from_u64(42);
            b.iter(|| {
                let image = DynamicImage::ImageRgb8(base.clone());
                let view = augment.apply(&image, &mut rng);
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
