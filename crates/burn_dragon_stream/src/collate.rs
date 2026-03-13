use crate::{CollatedStreamBatch, StreamSegment};

pub trait StreamCollatable {
    type Batch;

    fn collate(items: &[Self]) -> Self::Batch
    where
        Self: Sized;
}

pub fn collate_stream_segments<T>(items: &[StreamSegment<T>]) -> CollatedStreamBatch<T::Batch>
where
    T: StreamCollatable + Clone,
{
    let payloads: Vec<T> = items.iter().map(|item| item.payload.clone()).collect();
    let stream = items.iter().map(|item| item.stream).collect();
    CollatedStreamBatch::new(T::collate(&payloads), stream)
}

#[cfg(test)]
mod tests {
    use crate::{StreamBoundary, StreamSampleId, StreamSegment, StreamStepMetadata};

    use super::{StreamCollatable, collate_stream_segments};

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct DummyPayload {
        value: u32,
    }

    impl StreamCollatable for DummyPayload {
        type Batch = Vec<u32>;

        fn collate(items: &[Self]) -> Self::Batch {
            items.iter().map(|item| item.value).collect()
        }
    }

    #[test]
    fn collate_stream_segments_preserves_metadata_and_payload_order() {
        let items = vec![
            StreamSegment::new(
                DummyPayload { value: 7 },
                StreamStepMetadata::new(
                    StreamSampleId::new(1, 2, 0),
                    StreamBoundary::ResetEpisode,
                    0,
                    0,
                ),
            ),
            StreamSegment::new(
                DummyPayload { value: 9 },
                StreamStepMetadata::new(
                    StreamSampleId::new(1, 2, 1),
                    StreamBoundary::Continue,
                    1,
                    1,
                ),
            ),
        ];
        let batch = collate_stream_segments(&items);
        assert_eq!(batch.payload, vec![7, 9]);
        assert_eq!(batch.stream.len(), 2);
        assert!(batch.stream[0].should_reset_state());
        assert!(!batch.stream[1].should_reset_state());
    }
}
