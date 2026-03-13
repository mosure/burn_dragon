#[cfg(not(feature = "benchmark"))]
fn main() {
    panic!("video_lejepa_bench requires --features benchmark");
}

#[cfg(feature = "benchmark")]
mod real {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn_autodiff::Autodiff;
    use burn_dragon::core::{FusedKernelConfig, ManifoldHyperConnectionsConfig};
    use burn_dragon::vision::train::bench::VisionVideoLejepaTrainStepBench;
    use burn_dragon::vision::{
        MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
        SpatialPositionalEncodingKind, VisionAttentionMode, VisionBackboneKind, VisionDragonConfig,
        VisionLatentActivation, VisionNormalize, VisionPatchEmbedMode, VisionVideoLejepaConfig,
    };
    use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
    use clap::Parser;
    use serde::Serialize;

    type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type TrainBackend = Autodiff<InnerBackend>;
    type Device = <TrainBackend as BackendTrait>::Device;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Copy, Serialize)]
    struct BenchCase {
        name: &'static str,
        batch_size: usize,
        image_size: usize,
        patch_size: usize,
        embed_dim: usize,
        projection_dim: usize,
        projection_hidden_dim: usize,
        spatial_steps: usize,
        n_head: usize,
        temporal_layers: usize,
        temporal_heads: usize,
        temporal_mlp_multiplier: usize,
        rollout_fast_steps: usize,
        context_frames: usize,
        target_frames: usize,
        max_records: usize,
    }

    #[derive(Clone, Serialize)]
    struct CaseResult {
        case: BenchCase,
        warmup: usize,
        iterations: usize,
        baseline_forward_ms: f64,
        fused_forward_ms: f64,
        forward_speedup_x: f64,
        baseline_train_step_ms: f64,
        fused_train_step_ms: f64,
        train_step_speedup_x: f64,
        baseline_forward_frames_per_sec: f64,
        fused_forward_frames_per_sec: f64,
        baseline_train_frames_per_sec: f64,
        fused_train_frames_per_sec: f64,
        forward_loss_abs_diff: f32,
        train_loss_abs_diff: f32,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        benchmark: &'static str,
        adapter: String,
        warmup: usize,
        iterations: usize,
        cases: Vec<CaseResult>,
    }

    const CASES: &[BenchCase] = &[
        BenchCase {
            name: "tiny_fs1",
            batch_size: 8,
            image_size: 32,
            patch_size: 4,
            embed_dim: 32,
            projection_dim: 16,
            projection_hidden_dim: 32,
            spatial_steps: 2,
            n_head: 4,
            temporal_layers: 2,
            temporal_heads: 4,
            temporal_mlp_multiplier: 2,
            rollout_fast_steps: 1,
            context_frames: 4,
            target_frames: 2,
            max_records: 64,
        },
        BenchCase {
            name: "tiny_fs4",
            batch_size: 8,
            image_size: 32,
            patch_size: 4,
            embed_dim: 32,
            projection_dim: 16,
            projection_hidden_dim: 32,
            spatial_steps: 2,
            n_head: 4,
            temporal_layers: 2,
            temporal_heads: 4,
            temporal_mlp_multiplier: 2,
            rollout_fast_steps: 4,
            context_frames: 4,
            target_frames: 2,
            max_records: 64,
        },
        BenchCase {
            name: "small_fs4",
            batch_size: 8,
            image_size: 32,
            patch_size: 4,
            embed_dim: 64,
            projection_dim: 32,
            projection_hidden_dim: 64,
            spatial_steps: 2,
            n_head: 8,
            temporal_layers: 3,
            temporal_heads: 8,
            temporal_mlp_multiplier: 2,
            rollout_fast_steps: 4,
            context_frames: 4,
            target_frames: 2,
            max_records: 64,
        },
    ];

    pub fn main() {
        let args = Args::parse();
        let device = Device::default();
        init_runtime(&device);

        let report = Report {
            benchmark: "burn_dragon video LEJEPA temporal fused benchmark",
            adapter: adapter_info(),
            warmup: args.warmup,
            iterations: args.iterations,
            cases: run_all_cases(&device, &args),
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).expect("serialize video bench report");
        println!("{markdown}");

        if let Some(path) = args.markdown_path.as_ref() {
            write_text_artifact(path, &markdown, "markdown artifact");
        }
        if let Some(path) = args.json_path.as_ref() {
            write_text_artifact(path, &json, "json artifact");
        }
    }

    fn init_runtime(device: &Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    fn adapter_info() -> String {
        let instance = wgpu::Instance::default();
        let adapter =
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
                .expect("wgpu adapter");
        let info = adapter.get_info();
        format!("{} ({:?})", info.name, info.device_type)
    }

    fn run_all_cases(device: &Device, args: &Args) -> Vec<CaseResult> {
        CASES
            .iter()
            .copied()
            .map(|case| run_case(case, device, args))
            .collect()
    }

    fn run_case(case: BenchCase, device: &Device, args: &Args) -> CaseResult {
        let batch = sample_batch(case, device);
        let batch_frames = (case.batch_size * (case.context_frames + case.target_frames)) as f64;

        let (forward_loss_abs_diff, train_loss_abs_diff) =
            parity_snapshot(case, batch.clone(), device);

        let baseline_forward_bench = build_bench(case, false, device);
        let fused_forward_bench = build_bench(case, true, device);
        let mut baseline_train_bench = build_bench(case, false, device);
        let mut fused_train_bench = build_bench(case, true, device);

        for _ in 0..args.warmup {
            let _ = baseline_forward_bench.forward_loss(batch.clone());
            let _ = fused_forward_bench.forward_loss(batch.clone());
            let _ = baseline_train_bench.train_step(batch.clone());
            let _ = fused_train_bench.train_step(batch.clone());
        }

        let baseline_forward_ns = (0..args.iterations)
            .map(|_| {
                time_ns(|| {
                    let _ = baseline_forward_bench.forward_loss(batch.clone());
                })
            })
            .collect::<Vec<_>>();
        let fused_forward_ns = (0..args.iterations)
            .map(|_| {
                time_ns(|| {
                    let _ = fused_forward_bench.forward_loss(batch.clone());
                })
            })
            .collect::<Vec<_>>();
        let baseline_train_ns = (0..args.iterations)
            .map(|_| {
                time_ns(|| {
                    let _ = baseline_train_bench.train_step(batch.clone());
                })
            })
            .collect::<Vec<_>>();
        let fused_train_ns = (0..args.iterations)
            .map(|_| {
                time_ns(|| {
                    let _ = fused_train_bench.train_step(batch.clone());
                })
            })
            .collect::<Vec<_>>();

        let baseline_forward_avg = mean_u128(&baseline_forward_ns);
        let fused_forward_avg = mean_u128(&fused_forward_ns);
        let baseline_train_avg = mean_u128(&baseline_train_ns);
        let fused_train_avg = mean_u128(&fused_train_ns);

        CaseResult {
            case,
            warmup: args.warmup,
            iterations: args.iterations,
            baseline_forward_ms: ns_to_ms(baseline_forward_avg),
            fused_forward_ms: ns_to_ms(fused_forward_avg),
            forward_speedup_x: baseline_forward_avg / fused_forward_avg,
            baseline_train_step_ms: ns_to_ms(baseline_train_avg),
            fused_train_step_ms: ns_to_ms(fused_train_avg),
            train_step_speedup_x: baseline_train_avg / fused_train_avg,
            baseline_forward_frames_per_sec: batch_frames / (baseline_forward_avg / 1e9),
            fused_forward_frames_per_sec: batch_frames / (fused_forward_avg / 1e9),
            baseline_train_frames_per_sec: batch_frames / (baseline_train_avg / 1e9),
            fused_train_frames_per_sec: batch_frames / (fused_train_avg / 1e9),
            forward_loss_abs_diff,
            train_loss_abs_diff,
        }
    }

    fn sample_batch(
        case: BenchCase,
        device: &Device,
    ) -> burn_dragon::vision::VideoClipBatch<TrainBackend> {
        let normalize = VisionNormalize::new([0.5; 3], [0.5; 3]);
        let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Train,
            frame_size: case.image_size,
            digit_size: case.image_size.saturating_sub(12).max(12),
            in_channels: 3,
            context_len: case.context_frames,
            target_len: case.target_frames,
            extra_future_frames: 0,
            frame_stride: 1,
            max_records: Some(case.max_records),
            normalize,
            min_velocity: 0.8,
            max_velocity: 2.0,
            seed: 2026,
        })
        .expect("moving mnist dataset");
        dataset.sample_batch::<TrainBackend>(case.batch_size, device)
    }

    fn build_bench(
        case: BenchCase,
        fused_temporal: bool,
        device: &Device,
    ) -> VisionVideoLejepaTrainStepBench<TrainBackend> {
        let vision = build_vision_config(case);
        let video = build_video_config(case, fused_temporal);
        let training = burn_dragon::vision::VisionTrainingHyperparameters {
            batch_size: case.batch_size,
            max_iters: 1,
            log_frequency: 1,
            rollout_min_steps: Some(case.spatial_steps),
            rollout_max_steps: Some(case.spatial_steps),
            rollout_backprop_steps: Some(case.spatial_steps),
            ..Default::default()
        };
        let optimizer = burn_dragon::train::OptimizerConfig {
            learning_rate: 1e-3,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        };
        <TrainBackend as BackendTrait>::seed(device, 4_242);
        VisionVideoLejepaTrainStepBench::<TrainBackend>::new(
            vision, video, &training, &optimizer, 10, device,
        )
        .expect("video bench")
    }

    fn build_vision_config(case: BenchCase) -> VisionDragonConfig {
        VisionDragonConfig {
            image_size: case.image_size,
            patch_size: case.patch_size,
            patch_embed_mode: VisionPatchEmbedMode::default(),
            backbone: VisionBackboneKind::Dense,
            in_channels: 3,
            embed_dim: case.embed_dim,
            steps: case.spatial_steps,
            n_head: case.n_head,
            mlp_internal_dim_multiplier: 2,
            dropout: 0.0,
            projection_dim: case.projection_dim,
            projection_hidden_dim: case.projection_hidden_dim,
            use_cls_token: true,
            cls_sync_alpha: 0.0,
            num_eyes: 1,
            cross_eye_steps: 0,
            token_state_norm: true,
            normalization: burn_dragon::core::DragonNormConfig::default(),
            latent_activation: VisionLatentActivation::default(),
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: case.image_size.div_ceil(case.patch_size),
            pos_max_width: case.image_size.div_ceil(case.patch_size),
            attention_mode: VisionAttentionMode::RowL1,
            use_alibi: true,
            fused_kernels: FusedKernelConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            trm_graph: Default::default(),
            rho_stream: Default::default(),
        }
    }

    fn build_video_config(case: BenchCase, fused_temporal: bool) -> VisionVideoLejepaConfig {
        let mut video = VisionVideoLejepaConfig {
            context_frames: case.context_frames,
            target_frames: case.target_frames,
            ..VisionVideoLejepaConfig::default()
        };
        video.teacher_ema.enabled = true;
        video.teacher_ema.decay = 0.996;
        video.loss.probe_weight = 0.25;
        video.loss.cosine_weight = 0.1;
        video.loss.sigreg.enabled = true;
        video.loss.sigreg.lambda = 0.02;
        video.temporal.n_layer = case.temporal_layers;
        video.temporal.n_head = case.temporal_heads;
        video.temporal.mlp_internal_dim_multiplier = case.temporal_mlp_multiplier;
        video.temporal.rollout_fast_steps_per_slow_step = case.rollout_fast_steps;
        video.temporal.fused = true;
        video.temporal.wgpu_recurrent_kernel = fused_temporal;
        video.temporal.wgpu_rollout_fused = fused_temporal;
        video.temporal.latent_block_size = 8;
        video.temporal.time_block_size = 8;
        video
    }

    fn parity_snapshot(
        case: BenchCase,
        batch: burn_dragon::vision::VideoClipBatch<TrainBackend>,
        device: &Device,
    ) -> (f32, f32) {
        let mut baseline = build_bench(case, false, device);
        let mut fused = build_bench(case, true, device);

        let baseline_forward = baseline
            .forward_loss(batch.clone())
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("baseline forward")[0];
        let fused_forward = fused
            .forward_loss(batch.clone())
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused forward")[0];

        let baseline_train = baseline
            .train_step(batch.clone())
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("baseline train")[0];
        let fused_train = fused
            .train_step(batch)
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused train")[0];

        (
            (baseline_forward - fused_forward).abs(),
            (baseline_train - fused_train).abs(),
        )
    }

    fn time_ns<F>(mut f: F) -> u128
    where
        F: FnMut(),
    {
        let start = Instant::now();
        f();
        start.elapsed().as_nanos()
    }

    fn mean_u128(values: &[u128]) -> f64 {
        let total = values.iter().copied().sum::<u128>() as f64;
        total / values.len().max(1) as f64
    }

    fn ns_to_ms(value: f64) -> f64 {
        value / 1_000_000.0
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        let _ = writeln!(&mut out, "# Video LEJEPA Fused Benchmark");
        let _ = writeln!(&mut out);
        let _ = writeln!(&mut out, "- adapter: {}", report.adapter);
        let _ = writeln!(&mut out, "- warmup: {}", report.warmup);
        let _ = writeln!(&mut out, "- iterations: {}", report.iterations);
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| case | forward ms (base) | forward ms (fused) | forward speedup | train ms (base) | train ms (fused) | train speedup | fused train fps | forward loss drift | train loss drift |"
        );
        let _ = writeln!(
            &mut out,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for case in &report.cases {
            let _ = writeln!(
                &mut out,
                "| {} | {:.2} | {:.2} | {:.2}x | {:.2} | {:.2} | {:.2}x | {:.1} | {:.5} | {:.5} |",
                case.case.name,
                case.baseline_forward_ms,
                case.fused_forward_ms,
                case.forward_speedup_x,
                case.baseline_train_step_ms,
                case.fused_train_step_ms,
                case.train_step_speedup_x,
                case.fused_train_frames_per_sec,
                case.forward_loss_abs_diff,
                case.train_loss_abs_diff,
            );
        }
        out
    }

    fn write_text_artifact(path: &Path, contents: &str, label: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create artifact parent");
        }
        fs::write(path, contents)
            .unwrap_or_else(|err| panic!("failed to write {label} {}: {err}", path.display()));
    }
}

#[cfg(feature = "benchmark")]
fn main() {
    real::main();
}
