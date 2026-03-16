#![recursion_limit = "256"]
#![cfg(feature = "benchmark")]

use std::time::{Duration, Instant};

use anyhow::Result;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::backend::Backend;
use burn_autodiff::Autodiff;
use burn_dragon::api::multimodal::model::{
    TargetTextEncoderAdapter, TextFusionAdapter, VisionFusionAdapter,
};
use burn_dragon::api::{multimodal, stream, train};
use burn_ndarray::{NdArray, NdArrayDevice};
use burn_wgpu::{Wgpu, WgpuDevice};
use clap::Parser;

type ForwardBackend = Wgpu<f32>;
type TrainBackend = Autodiff<NdArray<f32>>;

#[derive(Parser, Debug)]
#[command(
    author,
    version,
    about = "Benchmark multimodal VL-JEPA staging and train-step timings"
)]
struct Args {
    #[arg(long, default_value_t = 20)]
    iterations: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct TimingStats {
    segment_assembly_ms: f64,
    reset_ms: f64,
    vision_encode_ms: f64,
    query_encode_ms: f64,
    target_encode_ms: f64,
    packing_ms: f64,
    fusion_ms: f64,
    total_forward_ms: f64,
    total_train_step_ms: f64,
}

#[derive(Clone, Copy, Debug, Default)]
struct LoaderStats {
    collate_throughput_steps_per_s: f64,
    host_to_device_bytes_per_step: usize,
    reset_bookkeeping_ms: f64,
}

fn main() -> Result<()> {
    let args = Args::parse();
    let image_config = small_config();
    let video_config = small_config();

    let image_segments = image_cpu_segments(64);
    let video_segments = video_cpu_segments(64);

    let train_device = NdArrayDevice::default();
    let image_train = benchmark_image_train_step(
        &image_config,
        &image_segments,
        &train_device,
        args.iterations,
    );
    let video_train = benchmark_video_train_step(
        &video_config,
        &video_segments,
        &train_device,
        args.iterations,
    );

    let forward_device = init_bench_device();
    let image_loader = benchmark_image_loader(&image_segments, &forward_device, args.iterations);
    let video_loader = benchmark_video_loader(&video_segments, &forward_device, args.iterations);
    let image_forward = benchmark_image_forward(
        &image_config,
        &image_segments,
        &forward_device,
        args.iterations,
    );
    let video_forward = benchmark_video_forward(
        &video_config,
        &video_segments,
        &forward_device,
        args.iterations,
    );
    ForwardBackend::memory_cleanup(&forward_device);

    println!("multimodal_vl_jepa_bench");
    println!("iterations={}", args.iterations);
    print_loader("image_text_loader", image_loader);
    print_loader("video_text_loader", video_loader);
    print_timing("image_text_forward", image_forward);
    print_timing("video_text_forward", video_forward);
    print_train("image_text_forward_backward_reference", image_train);
    print_train("video_text_forward_backward_reference", video_train);

    Ok(())
}

fn init_bench_device() -> WgpuDevice {
    let device = WgpuDevice::default();
    train::wgpu::init_runtime(&device, &train::config::WgpuRuntimeConfig::default());
    device
}

fn small_config() -> multimodal::config::VlJepaDragonConfig {
    let mut config = multimodal::config::VlJepaDragonConfig::default();
    config.vision.image_size = 16;
    config.vision.patch_size = 4;
    config.vision.embed_dim = 16;
    config.vision.projection_dim = 16;
    config.vision.projection_hidden_dim = 16;
    config.vision.steps = 1;
    config.query_text.n_layer = 2;
    config.query_text.n_embd = 16;
    config.query_text.n_head = 2;
    config.target_text.n_layer = 2;
    config.target_text.n_embd = 16;
    config.target_text.n_head = 2;
    config.fusion.n_layer = 2;
    config.fusion.n_embd = 16;
    config.fusion.n_head = 2;
    config.fusion_dim = 16;
    config.target_dim = 16;
    config
}

fn image_cpu_segments(
    count: usize,
) -> Vec<stream::segment::StreamSegment<multimodal::train::VisionLanguageCpuSample>> {
    (0..count)
        .map(|index| {
            stream::segment::StreamSegment::new(
                multimodal::train::VisionLanguageCpuSample {
                    image_chw: vec![0.0; 3 * 16 * 16],
                    channels: 3,
                    height: 16,
                    width: 16,
                    query_q_tokens: vec![1, 2, 3, 4],
                    target_y_tokens: vec![5, 6, 7, 8],
                },
                stream::segment::StreamStepMetadata::new(
                    stream::ids::StreamSampleId::new(1, 1, index as u64),
                    if index == 0 {
                        stream::boundary::StreamBoundary::ResetEpisode
                    } else {
                        stream::boundary::StreamBoundary::Continue
                    },
                    index,
                    index,
                ),
            )
        })
        .collect()
}

fn video_cpu_segments(
    count: usize,
) -> Vec<stream::segment::StreamSegment<multimodal::train::VideoLanguageCpuSample>> {
    (0..count)
        .map(|index| {
            stream::segment::StreamSegment::new(
                multimodal::train::VideoLanguageCpuSample {
                    video_tchw: vec![0.0; 2 * 3 * 16 * 16],
                    frames: 2,
                    channels: 3,
                    height: 16,
                    width: 16,
                    query_q_tokens: vec![1, 2, 3, 4],
                    target_y_tokens: vec![5, 6, 7, 8],
                    target_horizon: 1,
                },
                stream::segment::StreamStepMetadata::new(
                    stream::ids::StreamSampleId::new(2, 3, index as u64),
                    if index == 0 {
                        stream::boundary::StreamBoundary::ResetEpisode
                    } else {
                        stream::boundary::StreamBoundary::Continue
                    },
                    index,
                    index,
                ),
            )
        })
        .collect()
}

fn host_to_device_bytes_for_image(
    segment: &stream::segment::StreamSegment<multimodal::train::VisionLanguageCpuSample>,
) -> usize {
    let payload = &segment.payload;
    payload.image_chw.len() * std::mem::size_of::<f32>()
        + (payload.query_q_tokens.len() * 2 + payload.target_y_tokens.len() * 2)
            * std::mem::size_of::<i64>()
}

fn host_to_device_bytes_for_video(
    segment: &stream::segment::StreamSegment<multimodal::train::VideoLanguageCpuSample>,
) -> usize {
    let payload = &segment.payload;
    payload.video_tchw.len() * std::mem::size_of::<f32>()
        + (payload.query_q_tokens.len() * 2 + payload.target_y_tokens.len() * 2)
            * std::mem::size_of::<i64>()
}

fn benchmark_image_loader(
    segments: &[stream::segment::StreamSegment<multimodal::train::VisionLanguageCpuSample>],
    device: &WgpuDevice,
    iterations: usize,
) -> LoaderStats {
    let config = small_config();
    let model = multimodal::model::VlJepaDragon::<ForwardBackend>::new(config.clone(), device);
    let start = Instant::now();
    let mut reset_duration = Duration::default();

    for index in 0..iterations {
        let segment = segments[index % segments.len()].clone();
        let collated = multimodal::train::collate_vision_language_segments::<ForwardBackend>(
            &[segment],
            device,
        );
        let mut state = model.init_state();
        let reset_start = Instant::now();
        state.apply_stream_controls(
            &collated.stream[0],
            config.tbptt.window(),
            config.tbptt.state_carry_policy,
            config.tbptt.fusion_carry_policy,
        );
        reset_duration += reset_start.elapsed();
    }

    let total = start.elapsed();
    LoaderStats {
        collate_throughput_steps_per_s: iterations as f64 / total.as_secs_f64().max(1e-9),
        host_to_device_bytes_per_step: host_to_device_bytes_for_image(&segments[0]),
        reset_bookkeeping_ms: reset_duration.as_secs_f64() * 1000.0 / iterations.max(1) as f64,
    }
}

fn benchmark_video_loader(
    segments: &[stream::segment::StreamSegment<multimodal::train::VideoLanguageCpuSample>],
    device: &WgpuDevice,
    iterations: usize,
) -> LoaderStats {
    let config = small_config();
    let model = multimodal::model::VlJepaDragon::<ForwardBackend>::new(config.clone(), device);
    let start = Instant::now();
    let mut reset_duration = Duration::default();

    for index in 0..iterations {
        let segment = segments[index % segments.len()].clone();
        let collated = multimodal::train::collate_video_language_segments::<ForwardBackend>(
            &[segment],
            device,
        );
        let mut state = model.init_state();
        let reset_start = Instant::now();
        state.apply_stream_controls(
            &collated.stream[0],
            config.tbptt.window(),
            config.tbptt.state_carry_policy,
            config.tbptt.fusion_carry_policy,
        );
        reset_duration += reset_start.elapsed();
    }

    let total = start.elapsed();
    LoaderStats {
        collate_throughput_steps_per_s: iterations as f64 / total.as_secs_f64().max(1e-9),
        host_to_device_bytes_per_step: host_to_device_bytes_for_video(&segments[0]),
        reset_bookkeeping_ms: reset_duration.as_secs_f64() * 1000.0 / iterations.max(1) as f64,
    }
}

fn benchmark_image_forward(
    config: &multimodal::config::VlJepaDragonConfig,
    segments: &[stream::segment::StreamSegment<multimodal::train::VisionLanguageCpuSample>],
    device: &WgpuDevice,
    iterations: usize,
) -> TimingStats {
    let model = multimodal::model::VlJepaDragon::<ForwardBackend>::new(config.clone(), device);
    let mut stats = TimingStats::default();

    for index in 0..iterations {
        let assembly_start = Instant::now();
        let segment = segments[index % segments.len()].clone();
        let collated = multimodal::train::collate_vision_language_segments::<ForwardBackend>(
            &[segment],
            device,
        );
        stats.segment_assembly_ms += assembly_start.elapsed().as_secs_f64() * 1000.0;

        let mut state = model.init_state();
        let reset_start = Instant::now();
        state.apply_stream_controls(
            &collated.stream[0],
            config.tbptt.window(),
            config.tbptt.state_carry_policy,
            config.tbptt.fusion_carry_policy,
        );
        stats.reset_ms += reset_start.elapsed().as_secs_f64() * 1000.0;

        let batch = collated.payload.clone();
        let total_start = Instant::now();

        let vision_start = Instant::now();
        let vision = model.vision_x_encoder.observe_x(
            batch.vision_x.clone(),
            state.vision.take(),
            multimodal::data::MultimodalStepMode::Observe,
        );
        stats.vision_encode_ms += vision_start.elapsed().as_secs_f64() * 1000.0;
        state.vision = vision.state.clone();

        let query_start = Instant::now();
        let query = model.query_q_encoder.observe_q(
            (batch.query_q_tokens.clone(), batch.query_q_mask.clone()),
            state.query_text.take(),
        );
        stats.query_encode_ms += query_start.elapsed().as_secs_f64() * 1000.0;
        state.query_text = query.state.clone();

        let target_start = Instant::now();
        let _target = model
            .target_y_encoder
            .encode_y((batch.target_y_tokens.clone(), batch.target_y_mask.clone()));
        stats.target_encode_ms += target_start.elapsed().as_secs_f64() * 1000.0;

        let pack_start = Instant::now();
        let packed = model.pack_fusion_inputs(&vision, &query, &state);
        stats.packing_ms += pack_start.elapsed().as_secs_f64() * 1000.0;

        let fusion_start = Instant::now();
        let _ = model.predict_fusion_from_packed(packed, state);
        let _ = ForwardBackend::sync(device);
        stats.fusion_ms += fusion_start.elapsed().as_secs_f64() * 1000.0;
        stats.total_forward_ms += total_start.elapsed().as_secs_f64() * 1000.0;
    }

    divide_timing(stats, iterations)
}

fn benchmark_video_forward(
    config: &multimodal::config::VlJepaDragonConfig,
    segments: &[stream::segment::StreamSegment<multimodal::train::VideoLanguageCpuSample>],
    device: &WgpuDevice,
    iterations: usize,
) -> TimingStats {
    let model = multimodal::model::VlJepaDragon::<ForwardBackend>::new(config.clone(), device);
    let mut stats = TimingStats::default();

    for index in 0..iterations {
        let assembly_start = Instant::now();
        let segment = segments[index % segments.len()].clone();
        let collated = multimodal::train::collate_video_language_segments::<ForwardBackend>(
            &[segment],
            device,
        );
        stats.segment_assembly_ms += assembly_start.elapsed().as_secs_f64() * 1000.0;

        let mut state = model.init_state();
        let reset_start = Instant::now();
        state.apply_stream_controls(
            &collated.stream[0],
            config.tbptt.window(),
            config.tbptt.state_carry_policy,
            config.tbptt.fusion_carry_policy,
        );
        stats.reset_ms += reset_start.elapsed().as_secs_f64() * 1000.0;

        let batch = collated.payload.clone();
        let total_start = Instant::now();

        let vision_start = Instant::now();
        let vision = model
            .vision_x_encoder
            .observe_video_x(batch.video_x.clone(), state.vision.take());
        stats.vision_encode_ms += vision_start.elapsed().as_secs_f64() * 1000.0;
        state.vision = vision.state.clone();

        let query_start = Instant::now();
        let query = model.query_q_encoder.observe_q(
            (batch.query_q_tokens.clone(), batch.query_q_mask.clone()),
            state.query_text.take(),
        );
        stats.query_encode_ms += query_start.elapsed().as_secs_f64() * 1000.0;
        state.query_text = query.state.clone();

        let target_start = Instant::now();
        let _target = model
            .target_y_encoder
            .encode_y((batch.target_y_tokens.clone(), batch.target_y_mask.clone()));
        stats.target_encode_ms += target_start.elapsed().as_secs_f64() * 1000.0;

        let pack_start = Instant::now();
        let packed = model.pack_fusion_inputs(&vision, &query, &state);
        stats.packing_ms += pack_start.elapsed().as_secs_f64() * 1000.0;

        let fusion_start = Instant::now();
        let _ = model.predict_fusion_from_packed(packed, state);
        let _ = ForwardBackend::sync(device);
        stats.fusion_ms += fusion_start.elapsed().as_secs_f64() * 1000.0;
        stats.total_forward_ms += total_start.elapsed().as_secs_f64() * 1000.0;
    }

    divide_timing(stats, iterations)
}

fn benchmark_image_train_step(
    config: &multimodal::config::VlJepaDragonConfig,
    segments: &[stream::segment::StreamSegment<multimodal::train::VisionLanguageCpuSample>],
    device: &NdArrayDevice,
    iterations: usize,
) -> TimingStats {
    let mut model = multimodal::model::VlJepaDragon::<TrainBackend>::new(config.clone(), device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TrainBackend, multimodal::model::VlJepaDragon<TrainBackend>>();
    let mut stats = TimingStats::default();

    for index in 0..iterations {
        let segment = segments[index % segments.len()].clone();
        let collated =
            multimodal::train::collate_vision_language_segments::<TrainBackend>(&[segment], device);
        let start = Instant::now();
        let step = multimodal::train::multimodal_train_step(
            &model,
            None,
            None,
            collated.payload,
            &collated.stream[0],
            model.init_state(),
            config,
        );
        let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
        model = optimizer.step(1.0e-3, model, grads);
        let _ = TrainBackend::sync(device);
        stats.total_train_step_ms += start.elapsed().as_secs_f64() * 1000.0;
    }

    divide_timing(stats, iterations)
}

fn benchmark_video_train_step(
    config: &multimodal::config::VlJepaDragonConfig,
    segments: &[stream::segment::StreamSegment<multimodal::train::VideoLanguageCpuSample>],
    device: &NdArrayDevice,
    iterations: usize,
) -> TimingStats {
    let mut model = multimodal::model::VlJepaDragon::<TrainBackend>::new(config.clone(), device);
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(0.0)
        .init::<TrainBackend, multimodal::model::VlJepaDragon<TrainBackend>>();
    let mut stats = TimingStats::default();

    for index in 0..iterations {
        let segment = segments[index % segments.len()].clone();
        let collated =
            multimodal::train::collate_video_language_segments::<TrainBackend>(&[segment], device);
        let start = Instant::now();
        let step = multimodal::train::multimodal_video_train_step(
            &model,
            None,
            None,
            collated.payload,
            &collated.stream[0],
            model.init_state(),
            config,
        );
        let grads = GradientsParams::from_grads(step.loss.total.backward(), &model);
        model = optimizer.step(1.0e-3, model, grads);
        let _ = TrainBackend::sync(device);
        stats.total_train_step_ms += start.elapsed().as_secs_f64() * 1000.0;
    }

    divide_timing(stats, iterations)
}

fn divide_timing(mut stats: TimingStats, iterations: usize) -> TimingStats {
    let denom = iterations.max(1) as f64;
    stats.segment_assembly_ms /= denom;
    stats.reset_ms /= denom;
    stats.vision_encode_ms /= denom;
    stats.query_encode_ms /= denom;
    stats.target_encode_ms /= denom;
    stats.packing_ms /= denom;
    stats.fusion_ms /= denom;
    stats.total_forward_ms /= denom;
    stats.total_train_step_ms /= denom;
    stats
}

fn print_loader(label: &str, stats: LoaderStats) {
    println!(
        "{label}: collate_throughput_steps_per_s={:.2} host_to_device_bytes_per_step={} reset_bookkeeping_ms={:.4}",
        stats.collate_throughput_steps_per_s,
        stats.host_to_device_bytes_per_step,
        stats.reset_bookkeeping_ms,
    );
}

fn print_timing(label: &str, stats: TimingStats) {
    println!(
        "{label}: segment_assembly_ms={:.4} reset_ms={:.4} vision_encode_ms={:.4} query_encode_ms={:.4} target_encode_ms={:.4} packing_ms={:.4} fusion_ms={:.4} total_forward_ms={:.4}",
        stats.segment_assembly_ms,
        stats.reset_ms,
        stats.vision_encode_ms,
        stats.query_encode_ms,
        stats.target_encode_ms,
        stats.packing_ms,
        stats.fusion_ms,
        stats.total_forward_ms,
    );
}

fn print_train(label: &str, stats: TimingStats) {
    println!(
        "{label}: total_train_step_ms={:.4}",
        stats.total_train_step_ms
    );
}
