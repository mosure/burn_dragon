use burn::tensor::{Tensor, TensorData};
use burn_autogaze::{FixationPoint, FixationSet, FrameFixationTrace};
use burn_dragon_dreamer::extract_crops;
use burn_ndarray::NdArray;
use criterion::{Criterion, criterion_group, criterion_main};

fn bench_extract_crops(c: &mut Criterion) {
    type BenchBackend = NdArray<f32>;
    let device = Default::default();
    let frame = Tensor::<BenchBackend, 4>::from_data(
        TensorData::new(vec![0.0; 32 * 1 * 28 * 28], [32, 1, 28, 28]),
        &device,
    );
    let traces = (0..32)
        .map(|_| {
            FrameFixationTrace::new(vec![FixationSet::new(
                vec![
                    FixationPoint::new(0.35, 0.35, 0.4, 1.0),
                    FixationPoint::new(0.65, 0.65, 0.3, 0.9),
                ],
                0.0,
                2,
            )])
        })
        .collect::<Vec<_>>();
    c.bench_function("extract_crops_batch32_k2_crop12", |b| {
        b.iter(|| {
            let crops = extract_crops(frame.clone(), &traces, 0, 12, 2);
            criterion::black_box(crops.shape());
        })
    });
}

criterion_group!(benches, bench_extract_crops);
criterion_main!(benches);
