use burn_dragon_dreamer::CachedSequenceSplit;
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_cached_sequence_batch(c: &mut Criterion) {
    type BenchBackend = NdArray<f32>;
    let split = CachedSequenceSplit::<BenchBackend, usize>::new(
        Default::default(),
        vec![0.0; 256 * 6 * 1 * 28 * 28],
        [6, 1, 28, 28],
        vec![0.0; 256 * 6 * 64],
        [6, 64],
        vec![0.0; 256 * 6 * 32],
        [6, 32],
        (0..256).collect(),
    );
    let indices: Vec<usize> = (0..32).map(|i| (i * 7) % 256).collect();
    c.bench_function("cached_sequence_batch_32", |b| {
        b.iter(|| {
            let batch = split.batch(&indices);
            criterion::black_box(batch.clip_frames.shape());
            criterion::black_box(batch.teacher_features.shape());
            criterion::black_box(batch.crop_teacher_features.shape());
            criterion::black_box(batch.traces.len());
        })
    });
}

criterion_group!(benches, bench_cached_sequence_batch);
criterion_main!(benches);
