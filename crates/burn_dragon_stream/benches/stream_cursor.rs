use burn_dragon_stream::{
    StreamBoundary, StreamSampleId, StreamSegment, StreamStepMetadata, VecStreamCursor,
    collate_stream_segments,
};
use burn_dragon_stream::{StreamCollatable, StreamCursor};
use criterion::{Criterion, criterion_group, criterion_main};

#[derive(Clone)]
struct DummyPayload {
    value: u32,
}

impl StreamCollatable for DummyPayload {
    type Batch = Vec<u32>;

    fn collate(items: &[Self]) -> Self::Batch {
        items.iter().map(|item| item.value).collect()
    }
}

fn make_segments(len: usize) -> Vec<StreamSegment<DummyPayload>> {
    (0..len)
        .map(|idx| {
            StreamSegment::new(
                DummyPayload { value: idx as u32 },
                StreamStepMetadata::new(
                    StreamSampleId::new(0, 0, idx as u64),
                    if idx == 0 {
                        StreamBoundary::ResetEpisode
                    } else {
                        StreamBoundary::Continue
                    },
                    idx,
                    idx,
                ),
            )
        })
        .collect()
}

fn bench_stream_cursor(c: &mut Criterion) {
    let segments = make_segments(1024);
    c.bench_function("stream_cursor_iterate_1024", |b| {
        b.iter(|| {
            let mut cursor = VecStreamCursor::new(segments.clone());
            let mut count = 0usize;
            while cursor.next_segment().is_some() {
                count = count.saturating_add(1);
            }
            count
        })
    });
    c.bench_function("stream_collate_1024", |b| {
        b.iter(|| collate_stream_segments(&segments))
    });
}

criterion_group!(benches, bench_stream_cursor);
criterion_main!(benches);
