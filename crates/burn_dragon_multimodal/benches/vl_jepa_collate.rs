use burn_dragon_multimodal::train::{VisionLanguageCpuSample, collate_vision_language_segments};
use burn_dragon_stream::{StreamBoundary, StreamSampleId, StreamSegment, StreamStepMetadata};
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};

type Backend = NdArray<f32>;

fn synthetic_segments(batch: usize) -> Vec<StreamSegment<VisionLanguageCpuSample>> {
    (0..batch)
        .map(|index| StreamSegment {
            payload: VisionLanguageCpuSample {
                image_chw: vec![0.0; 3 * 32 * 32],
                channels: 3,
                height: 32,
                width: 32,
                query_q_tokens: vec![1, 2, 3, 4, 5, 6, 7, 8],
                target_y_tokens: vec![8, 7, 6, 5, 4, 3, 2, 1],
            },
            stream: StreamStepMetadata {
                sample_id: StreamSampleId {
                    source_id: 1,
                    episode_id: 1,
                    segment_id: index as u64,
                },
                boundary: StreamBoundary::Continue,
                step_index: index,
                absolute_time: index,
            },
        })
        .collect()
}

fn benchmark_vl_jepa_collate(c: &mut Criterion) {
    let device = <Backend as burn::tensor::backend::Backend>::Device::default();
    let segments_32 = synthetic_segments(32);
    let segments_128 = synthetic_segments(128);

    c.bench_function("vl_jepa_collate_32", |b| {
        b.iter(|| {
            let _ = collate_vision_language_segments::<Backend>(&segments_32, &device);
        });
    });

    c.bench_function("vl_jepa_collate_128", |b| {
        b.iter(|| {
            let _ = collate_vision_language_segments::<Backend>(&segments_128, &device);
        });
    });
}

criterion_group!(benches, benchmark_vl_jepa_collate);
criterion_main!(benches);
