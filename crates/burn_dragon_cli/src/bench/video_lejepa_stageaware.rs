#[cfg(feature = "benchmark")]
pub mod real {
    use std::fmt::Write as _;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use burn::tensor::backend::Backend as BackendTrait;
    use burn_autodiff::Autodiff;
    use burn_dragon::vision::train::bench::{
        VisionVideoLejepaTrainStepBench, VisionVideoLejepaTrainStepPhaseTimes,
        VisionVideoLejepaTrainStepProfile,
    };
    use burn_dragon::vision::{
        MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig,
        StageAwareHostProfileSnapshot, VideoClipBatch, VisionDatasetSource, VisionNormalize,
        VisionArtifactHeader, VisionTrainingConfig, VisionTrainingModeConfig,
        VisionVideoTrainProfileSnapshot, push_vision_artifact_markdown_prelude,
        stage_aware_host_profile_reset, stage_aware_host_profile_snapshot,
        video_train_profile_reset, video_train_profile_snapshot,
    };
    use burn_dragon_wgpu::api::recurrent::{
        RecurrentProfileSnapshot, recurrent_profile_reset, recurrent_profile_snapshot,
    };
    use burn_dragon_wgpu::api::spatial::{
        LocalGridRhoProfileSnapshot, StructuredPyramidProfileSnapshot,
        local_grid_rho_profile_reset, local_grid_rho_profile_snapshot,
        structured_pyramid_profile_reset, structured_pyramid_profile_snapshot,
    };
    use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};
    use clap::{Parser, ValueEnum};
    use serde::Serialize;

    type InnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
    type TrainBackend = Autodiff<InnerBackend>;
    type Device = <TrainBackend as BackendTrait>::Device;

    #[derive(Parser, Debug)]
    struct Args {
        #[arg(long)]
        config_json: PathBuf,
        #[arg(long, default_value_t = 1)]
        warmup: usize,
        #[arg(long, default_value_t = 5)]
        iterations: usize,
        #[arg(long)]
        batch_size: Option<usize>,
        #[arg(long, value_enum, default_value_t = BenchmarkMode::Both)]
        mode: BenchmarkMode,
        #[arg(long)]
        markdown_path: Option<PathBuf>,
        #[arg(long)]
        json_path: Option<PathBuf>,
    }

    #[derive(Clone, Copy, Debug, Serialize, ValueEnum, PartialEq, Eq)]
    #[serde(rename_all = "snake_case")]
    enum BenchmarkMode {
        Both,
        Host,
        Hybrid,
    }

    #[derive(Clone, Serialize)]
    struct KernelProfileResult {
        calls: f64,
        launches: f64,
        dispatch_ms: f64,
        transient_allocations: f64,
        metadata_upload_bytes: f64,
        metadata_reuse_bytes: f64,
    }

    #[derive(Clone, Serialize)]
    struct HostStageProfileResult {
        step_calls: f64,
        coarse_only_step_calls: f64,
        patch_local_ms: f64,
        coarse_local_ms: f64,
        patch_from_coarse_ms: f64,
        hub_read_ms: f64,
        patch_to_coarse_ms: f64,
        hub_update_ms: f64,
    }

    #[derive(Clone, Serialize)]
    struct ModeResult {
        mode: &'static str,
        forward_ms: f64,
        train_step_ms: f64,
        train_forward_ms: f64,
        train_backward_ms: f64,
        train_optimize_ms: f64,
        frames_per_sec: f64,
        train_frames_per_sec: f64,
        forward_structured: KernelProfileResult,
        forward_local_grid: KernelProfileResult,
        forward_recurrent: KernelProfileResult,
        forward_stage_host: HostStageProfileResult,
        train_structured: KernelProfileResult,
        train_local_grid: KernelProfileResult,
        train_recurrent: KernelProfileResult,
        train_stage_host: HostStageProfileResult,
        train_video_forward: VideoTrainProfileResult,
    }

    #[derive(Clone, Serialize)]
    struct VideoTrainProfileResult {
        train_calls: f64,
        split_projection_ms: f64,
        context_rollout_ms: f64,
        predict_rollout_ms: f64,
        loss_heads_ms: f64,
    }

    #[derive(Clone, Serialize)]
    struct Report {
        artifact: VisionArtifactHeader,
        benchmark: &'static str,
        adapter: String,
        config_json: PathBuf,
        warmup: usize,
        iterations: usize,
        batch_size: usize,
        mode: BenchmarkMode,
        forward_loss_abs_diff: Option<f32>,
        train_loss_abs_diff: Option<f32>,
        host: Option<ModeResult>,
        hybrid: Option<ModeResult>,
        forward_speedup_x: Option<f64>,
        train_step_speedup_x: Option<f64>,
    }

    struct BenchmarkResults {
        host: Option<ModeResult>,
        hybrid: Option<ModeResult>,
        forward_loss_abs_diff: Option<f32>,
        train_loss_abs_diff: Option<f32>,
        forward_speedup_x: Option<f64>,
        train_step_speedup_x: Option<f64>,
    }

    pub fn main() {
        // Force stage-level profiling on for this benchmark so the markdown report captures
        // the remaining host-side patch/coarse/hub breakdowns instead of zeroing them out.
        unsafe {
            std::env::set_var("BDH_STAGE_PROFILE", "1");
        }
        let args = Args::parse();
        let device = Device::default();
        init_runtime(&device);

        let config = load_config(&args.config_json);
        let batch_size = args.batch_size.unwrap_or(config.training.batch_size).max(1);
        let batch = sample_batch(&config, batch_size, &device);
        let results = run_benchmark(&config, batch, batch_size, &device, &args);

        let report = Report {
            artifact: VisionArtifactHeader::new("video_lejepa_stageaware_bench"),
            benchmark: "burn_dragon video LEJEPA stage-aware host-vs-hybrid benchmark",
            adapter: adapter_info(),
            config_json: args.config_json.clone(),
            warmup: args.warmup,
            iterations: args.iterations,
            batch_size,
            mode: args.mode,
            forward_loss_abs_diff: results.forward_loss_abs_diff,
            train_loss_abs_diff: results.train_loss_abs_diff,
            forward_speedup_x: results.forward_speedup_x,
            train_step_speedup_x: results.train_step_speedup_x,
            host: results.host,
            hybrid: results.hybrid,
        };

        let markdown = format_markdown(&report);
        let json = serde_json::to_string_pretty(&report).expect("serialize benchmark report");
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

    fn load_config(path: &Path) -> VisionTrainingConfig {
        let text = fs::read_to_string(path)
            .unwrap_or_else(|err| panic!("failed to read config json {}: {err}", path.display()));
        let config: VisionTrainingConfig = serde_json::from_str(&text)
            .unwrap_or_else(|err| panic!("failed to parse config json {}: {err}", path.display()));
        config
            .validate()
            .unwrap_or_else(|err| panic!("invalid config json {}: {err}", path.display()));
        config
    }

    fn sample_batch(
        config: &VisionTrainingConfig,
        batch_size: usize,
        device: &Device,
    ) -> VideoClipBatch<TrainBackend> {
        if !matches!(config.dataset.source, VisionDatasetSource::MovingMnist) {
            panic!(
                "video_lejepa_stageaware_bench currently requires dataset.source = \"moving_mnist\""
            );
        }
        let video = match &config.mode {
            VisionTrainingModeConfig::VideoLejepa(video) => video,
            other => panic!("expected mode.type = video_lejepa, got {other:?}"),
        };
        let vision = config.vision.build();
        let normalize =
            VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
        let train_target_frames_max = video.effective_train_target_frames_max();
        let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Train,
            frame_size: vision.image_size,
            digit_size: config.dataset.moving_mnist.digit_size,
            in_channels: vision.in_channels,
            context_len: video.context_frames,
            target_len: train_target_frames_max,
            extra_future_frames: 0,
            frame_stride: video.frame_stride,
            max_records: config.dataset.max_records,
            normalize,
            min_velocity: config.dataset.moving_mnist.min_velocity,
            max_velocity: config.dataset.moving_mnist.max_velocity,
            seed: config.dataset.moving_mnist.train_seed,
        })
        .expect("moving mnist dataset");
        dataset.sample_batch(batch_size, device)
    }

    fn run_benchmark(
        config: &VisionTrainingConfig,
        batch: VideoClipBatch<TrainBackend>,
        batch_size: usize,
        device: &Device,
        args: &Args,
    ) -> BenchmarkResults {
        let mut host_cfg = config.clone();
        host_cfg.training.batch_size = batch_size;
        host_cfg.vision.fused_kernels = false;

        let mut hybrid_cfg = config.clone();
        hybrid_cfg.training.batch_size = batch_size;
        hybrid_cfg.vision.fused_kernels = true;

        let video = match &config.mode {
            VisionTrainingModeConfig::VideoLejepa(video) => video,
            _ => unreachable!(),
        };
        let frames_per_batch = (batch_size
            * (video.context_frames + video.effective_train_target_frames_max()))
            as f64;
        let mut host: Option<ModeResult> = None;
        let mut hybrid: Option<ModeResult> = None;
        let mut host_forward_loss: Option<f32> = None;
        let mut host_train_loss: Option<f32> = None;
        let mut hybrid_forward_loss: Option<f32> = None;
        let mut hybrid_train_loss: Option<f32> = None;

        if matches!(args.mode, BenchmarkMode::Both | BenchmarkMode::Host) {
            let mut host_bench = build_bench(&host_cfg, device);
            for _ in 0..args.warmup {
                let _ = host_bench.forward_loss(batch.clone());
                let _ = host_bench.train_step(batch.clone());
            }

            let host_forward = measure_with_profiles(args.iterations, || {
                let _ = host_bench.forward_loss(batch.clone());
            });
            let host_train = measure_train_step_with_profiles(args.iterations, || {
                host_bench.train_step_profile(batch.clone())
            });

            host_forward_loss = Some(
                host_bench
                    .forward_loss(batch.clone())
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("host forward loss")[0],
            );
            host_train_loss = Some(
                host_bench
                    .train_step(batch.clone())
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("host train loss")[0],
            );
            host = Some(ModeResult {
                mode: "host",
                forward_ms: ns_to_ms(host_forward.elapsed_ns),
                train_step_ms: ns_to_ms(host_train.elapsed_ns),
                train_forward_ms: ns_to_ms(host_train.train_phases.forward_ns),
                train_backward_ms: ns_to_ms(host_train.train_phases.backward_ns),
                train_optimize_ms: ns_to_ms(host_train.train_phases.optimize_ns),
                frames_per_sec: frames_per_batch / (host_forward.elapsed_ns / 1e9),
                train_frames_per_sec: frames_per_batch / (host_train.elapsed_ns / 1e9),
                forward_structured: host_forward.structured,
                forward_local_grid: host_forward.local_grid,
                forward_recurrent: host_forward.recurrent,
                forward_stage_host: host_forward.stage_host,
                train_structured: host_train.structured,
                train_local_grid: host_train.local_grid,
                train_recurrent: host_train.recurrent,
                train_stage_host: host_train.stage_host,
                train_video_forward: host_train.video_forward,
            });
            drop(host_bench);
            cleanup_device(device);
        }

        if matches!(args.mode, BenchmarkMode::Both | BenchmarkMode::Hybrid) {
            let mut hybrid_bench = build_bench(&hybrid_cfg, device);
            for _ in 0..args.warmup {
                let _ = hybrid_bench.forward_loss(batch.clone());
                let _ = hybrid_bench.train_step(batch.clone());
            }

            let hybrid_forward = measure_with_profiles(args.iterations, || {
                let _ = hybrid_bench.forward_loss(batch.clone());
            });
            let hybrid_train = measure_train_step_with_profiles(args.iterations, || {
                hybrid_bench.train_step_profile(batch.clone())
            });

            hybrid_forward_loss = Some(
                hybrid_bench
                    .forward_loss(batch.clone())
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("hybrid forward loss")[0],
            );
            hybrid_train_loss = Some(
                hybrid_bench
                    .train_step(batch)
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("hybrid train loss")[0],
            );
            hybrid = Some(ModeResult {
                mode: "hybrid",
                forward_ms: ns_to_ms(hybrid_forward.elapsed_ns),
                train_step_ms: ns_to_ms(hybrid_train.elapsed_ns),
                train_forward_ms: ns_to_ms(hybrid_train.train_phases.forward_ns),
                train_backward_ms: ns_to_ms(hybrid_train.train_phases.backward_ns),
                train_optimize_ms: ns_to_ms(hybrid_train.train_phases.optimize_ns),
                frames_per_sec: frames_per_batch / (hybrid_forward.elapsed_ns / 1e9),
                train_frames_per_sec: frames_per_batch / (hybrid_train.elapsed_ns / 1e9),
                forward_structured: hybrid_forward.structured,
                forward_local_grid: hybrid_forward.local_grid,
                forward_recurrent: hybrid_forward.recurrent,
                forward_stage_host: hybrid_forward.stage_host,
                train_structured: hybrid_train.structured,
                train_local_grid: hybrid_train.local_grid,
                train_recurrent: hybrid_train.recurrent,
                train_stage_host: hybrid_train.stage_host,
                train_video_forward: hybrid_train.video_forward,
            });
        }

        let forward_loss_abs_diff = host_forward_loss
            .zip(hybrid_forward_loss)
            .map(|(host, hybrid)| (host - hybrid).abs());
        let train_loss_abs_diff = host_train_loss
            .zip(hybrid_train_loss)
            .map(|(host, hybrid)| (host - hybrid).abs());
        let forward_speedup_x = host
            .as_ref()
            .zip(hybrid.as_ref())
            .map(|(host, hybrid)| host.forward_ms / hybrid.forward_ms);
        let train_step_speedup_x = host
            .as_ref()
            .zip(hybrid.as_ref())
            .map(|(host, hybrid)| host.train_step_ms / hybrid.train_step_ms);

        BenchmarkResults {
            host,
            hybrid,
            forward_loss_abs_diff,
            train_loss_abs_diff,
            forward_speedup_x,
            train_step_speedup_x,
        }
    }

    fn cleanup_device(device: &Device) {
        let _ = <TrainBackend as BackendTrait>::sync(device);
        <TrainBackend as BackendTrait>::memory_cleanup(device);
        let _ = <TrainBackend as BackendTrait>::sync(device);
    }

    #[derive(Clone)]
    struct TimedProfileResult {
        elapsed_ns: f64,
        train_phases: VisionVideoLejepaTrainStepPhaseTimes,
        structured: KernelProfileResult,
        local_grid: KernelProfileResult,
        recurrent: KernelProfileResult,
        stage_host: HostStageProfileResult,
        video_forward: VideoTrainProfileResult,
    }

    fn build_bench(
        config: &VisionTrainingConfig,
        device: &Device,
    ) -> VisionVideoLejepaTrainStepBench<TrainBackend> {
        let vision = config.vision.build();
        let video = match &config.mode {
            VisionTrainingModeConfig::VideoLejepa(video) => video.clone(),
            other => panic!("expected video_lejepa mode, got {other:?}"),
        };
        VisionVideoLejepaTrainStepBench::<TrainBackend>::new(
            vision,
            video,
            &config.training,
            &config.optimizer,
            10,
            device,
        )
        .expect("video stage-aware bench")
    }

    fn measure_with_profiles<F>(iterations: usize, mut f: F) -> TimedProfileResult
    where
        F: FnMut(),
    {
        let mut elapsed = Vec::with_capacity(iterations.max(1));
        let mut structured = Vec::with_capacity(iterations.max(1));
        let mut local_grid = Vec::with_capacity(iterations.max(1));
        let mut recurrent = Vec::with_capacity(iterations.max(1));
        let mut stage_host = Vec::with_capacity(iterations.max(1));
        let mut video_forward = Vec::with_capacity(iterations.max(1));
        for _ in 0..iterations.max(1) {
            structured_pyramid_profile_reset();
            local_grid_rho_profile_reset();
            recurrent_profile_reset();
            stage_aware_host_profile_reset();
            video_train_profile_reset();
            let start = Instant::now();
            f();
            elapsed.push(start.elapsed().as_nanos() as f64);
            structured.push(structured_pyramid_profile_snapshot());
            local_grid.push(local_grid_rho_profile_snapshot());
            recurrent.push(recurrent_profile_snapshot());
            stage_host.push(stage_aware_host_profile_snapshot());
            video_forward.push(video_train_profile_snapshot());
        }
        TimedProfileResult {
            elapsed_ns: mean_f64(&elapsed),
            train_phases: VisionVideoLejepaTrainStepPhaseTimes {
                forward_ns: 0.0,
                backward_ns: 0.0,
                optimize_ns: 0.0,
            },
            structured: average_structured_profile(&structured),
            local_grid: average_local_grid_profile(&local_grid),
            recurrent: average_recurrent_profile(&recurrent),
            stage_host: average_stage_host_profile(&stage_host),
            video_forward: average_video_train_profile(&video_forward),
        }
    }

    fn measure_train_step_with_profiles<F>(iterations: usize, mut f: F) -> TimedProfileResult
    where
        F: FnMut() -> VisionVideoLejepaTrainStepProfile<TrainBackend>,
    {
        let mut elapsed = Vec::with_capacity(iterations.max(1));
        let mut forward = Vec::with_capacity(iterations.max(1));
        let mut backward = Vec::with_capacity(iterations.max(1));
        let mut optimize = Vec::with_capacity(iterations.max(1));
        let mut structured = Vec::with_capacity(iterations.max(1));
        let mut local_grid = Vec::with_capacity(iterations.max(1));
        let mut recurrent = Vec::with_capacity(iterations.max(1));
        let mut stage_host = Vec::with_capacity(iterations.max(1));
        let mut video_forward = Vec::with_capacity(iterations.max(1));
        for _ in 0..iterations.max(1) {
            structured_pyramid_profile_reset();
            local_grid_rho_profile_reset();
            recurrent_profile_reset();
            stage_aware_host_profile_reset();
            video_train_profile_reset();
            let start = Instant::now();
            let profile = f();
            elapsed.push(start.elapsed().as_nanos() as f64);
            forward.push(profile.phases.forward_ns);
            backward.push(profile.phases.backward_ns);
            optimize.push(profile.phases.optimize_ns);
            structured.push(structured_pyramid_profile_snapshot());
            local_grid.push(local_grid_rho_profile_snapshot());
            recurrent.push(recurrent_profile_snapshot());
            stage_host.push(stage_aware_host_profile_snapshot());
            video_forward.push(video_train_profile_snapshot());
        }
        TimedProfileResult {
            elapsed_ns: mean_f64(&elapsed),
            train_phases: VisionVideoLejepaTrainStepPhaseTimes {
                forward_ns: mean_f64(&forward),
                backward_ns: mean_f64(&backward),
                optimize_ns: mean_f64(&optimize),
            },
            structured: average_structured_profile(&structured),
            local_grid: average_local_grid_profile(&local_grid),
            recurrent: average_recurrent_profile(&recurrent),
            stage_host: average_stage_host_profile(&stage_host),
            video_forward: average_video_train_profile(&video_forward),
        }
    }

    fn ns_to_ms(value: f64) -> f64 {
        value / 1_000_000.0
    }

    fn mean_f64(values: &[f64]) -> f64 {
        if values.is_empty() {
            0.0
        } else {
            values.iter().sum::<f64>() / values.len() as f64
        }
    }

    fn average_structured_profile(
        values: &[StructuredPyramidProfileSnapshot],
    ) -> KernelProfileResult {
        KernelProfileResult {
            calls: mean_f64(&values.iter().map(|v| v.calls as f64).collect::<Vec<_>>()),
            launches: mean_f64(&values.iter().map(|v| v.launches as f64).collect::<Vec<_>>()),
            dispatch_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.dispatch_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            transient_allocations: mean_f64(
                &values
                    .iter()
                    .map(|v| v.transient_allocations as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_upload_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_upload_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_reuse_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_reuse_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn average_recurrent_profile(values: &[RecurrentProfileSnapshot]) -> KernelProfileResult {
        KernelProfileResult {
            calls: mean_f64(&values.iter().map(|v| v.calls as f64).collect::<Vec<_>>()),
            launches: mean_f64(&values.iter().map(|v| v.launches as f64).collect::<Vec<_>>()),
            dispatch_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.dispatch_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            transient_allocations: mean_f64(
                &values
                    .iter()
                    .map(|v| v.transient_allocations as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_upload_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_upload_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_reuse_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_reuse_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn average_stage_host_profile(
        values: &[StageAwareHostProfileSnapshot],
    ) -> HostStageProfileResult {
        HostStageProfileResult {
            step_calls: mean_f64(
                &values
                    .iter()
                    .map(|v| v.step_calls as f64)
                    .collect::<Vec<_>>(),
            ),
            coarse_only_step_calls: mean_f64(
                &values
                    .iter()
                    .map(|v| v.coarse_only_step_calls as f64)
                    .collect::<Vec<_>>(),
            ),
            patch_local_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.patch_local_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            coarse_local_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.coarse_local_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            patch_from_coarse_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.patch_from_coarse_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            hub_read_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.hub_read_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            patch_to_coarse_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.patch_to_coarse_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            hub_update_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.hub_update_ns as f64))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn average_local_grid_profile(values: &[LocalGridRhoProfileSnapshot]) -> KernelProfileResult {
        KernelProfileResult {
            calls: mean_f64(&values.iter().map(|v| v.calls as f64).collect::<Vec<_>>()),
            launches: mean_f64(&values.iter().map(|v| v.launches as f64).collect::<Vec<_>>()),
            dispatch_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.dispatch_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            transient_allocations: mean_f64(
                &values
                    .iter()
                    .map(|v| v.transient_allocations as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_upload_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_upload_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
            metadata_reuse_bytes: mean_f64(
                &values
                    .iter()
                    .map(|v| v.metadata_reuse_bytes as f64)
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn average_video_train_profile(
        values: &[VisionVideoTrainProfileSnapshot],
    ) -> VideoTrainProfileResult {
        VideoTrainProfileResult {
            train_calls: mean_f64(
                &values
                    .iter()
                    .map(|v| v.train_calls as f64)
                    .collect::<Vec<_>>(),
            ),
            split_projection_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.split_projection_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            context_rollout_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.context_rollout_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            predict_rollout_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.predict_rollout_ns as f64))
                    .collect::<Vec<_>>(),
            ),
            loss_heads_ms: mean_f64(
                &values
                    .iter()
                    .map(|v| ns_to_ms(v.loss_heads_ns as f64))
                    .collect::<Vec<_>>(),
            ),
        }
    }

    fn format_markdown(report: &Report) -> String {
        let mut out = String::new();
        push_vision_artifact_markdown_prelude(
            &mut out,
            "Video LEJEPA Stage-Aware Host vs Hybrid Benchmark",
            &report.artifact,
        );
        let _ = writeln!(&mut out, "- adapter: {}", report.adapter);
        let _ = writeln!(&mut out, "- config_json: {}", report.config_json.display());
        let _ = writeln!(&mut out, "- warmup: {}", report.warmup);
        let _ = writeln!(&mut out, "- iterations: {}", report.iterations);
        let _ = writeln!(&mut out, "- batch_size: {}", report.batch_size);
        let _ = writeln!(&mut out, "- mode: {:?}", report.mode);
        let _ = writeln!(&mut out);
        let mode_results = [report.host.as_ref(), report.hybrid.as_ref()];
        let _ = writeln!(
            &mut out,
            "| mode | forward ms | train-step ms | forward fps | train fps | train local-grid launches | train local-grid dispatch ms | train structured launches | train structured dispatch ms | train recurrent launches | train recurrent dispatch ms |"
        );
        let _ = writeln!(
            &mut out,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for result in mode_results.iter().flatten() {
            let _ = writeln!(
                &mut out,
                "| {} | {:.2} | {:.2} | {:.1} | {:.1} | {:.1} | {:.3} | {:.1} | {:.3} | {:.1} | {:.3} |",
                result.mode,
                result.forward_ms,
                result.train_step_ms,
                result.frames_per_sec,
                result.train_frames_per_sec,
                result.train_local_grid.launches,
                result.train_local_grid.dispatch_ms,
                result.train_structured.launches,
                result.train_structured.dispatch_ms,
                result.train_recurrent.launches,
                result.train_recurrent.dispatch_ms
            );
        }
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| mode | train forward ms | train backward ms | train optimize ms |"
        );
        let _ = writeln!(&mut out, "| --- | ---: | ---: | ---: |");
        for result in mode_results.iter().flatten() {
            let _ = writeln!(
                &mut out,
                "| {} | {:.2} | {:.2} | {:.2} |",
                result.mode,
                result.train_forward_ms,
                result.train_backward_ms,
                result.train_optimize_ms
            );
        }
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| mode | train host patch-local ms | train host coarse-local ms | train host patch<-coarse ms | train host hub-read ms | train host patch->coarse ms | train host hub-update ms |"
        );
        let _ = writeln!(
            &mut out,
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"
        );
        for result in mode_results.iter().flatten() {
            let _ = writeln!(
                &mut out,
                "| {} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} | {:.3} |",
                result.mode,
                result.train_stage_host.patch_local_ms,
                result.train_stage_host.coarse_local_ms,
                result.train_stage_host.patch_from_coarse_ms,
                result.train_stage_host.hub_read_ms,
                result.train_stage_host.patch_to_coarse_ms,
                result.train_stage_host.hub_update_ms
            );
        }
        let _ = writeln!(&mut out);
        let _ = writeln!(
            &mut out,
            "| mode | train split/proj ms | train context rollout ms | train predict rollout ms | train loss-heads ms |"
        );
        let _ = writeln!(&mut out, "| --- | ---: | ---: | ---: | ---: |");
        for result in mode_results.iter().flatten() {
            let _ = writeln!(
                &mut out,
                "| {} | {:.2} | {:.2} | {:.2} | {:.2} |",
                result.mode,
                result.train_video_forward.split_projection_ms,
                result.train_video_forward.context_rollout_ms,
                result.train_video_forward.predict_rollout_ms,
                result.train_video_forward.loss_heads_ms,
            );
        }
        let _ = writeln!(&mut out);
        if let Some(speedup) = report.forward_speedup_x {
            let _ = writeln!(&mut out, "- forward speedup: {:.2}x", speedup);
        }
        if let Some(speedup) = report.train_step_speedup_x {
            let _ = writeln!(&mut out, "- train-step speedup: {:.2}x", speedup);
        }
        if let Some(diff) = report.forward_loss_abs_diff {
            let _ = writeln!(&mut out, "- forward loss drift: {:.6}", diff);
        }
        if let Some(diff) = report.train_loss_abs_diff {
            let _ = writeln!(&mut out, "- train loss drift: {:.6}", diff);
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
