use crate::StreamSegment;

pub trait StreamCursor {
    type Item;

    fn next_segment(&mut self) -> Option<StreamSegment<Self::Item>>;
}

#[derive(Clone, Debug)]
pub struct VecStreamCursor<T> {
    items: Vec<StreamSegment<T>>,
    index: usize,
}

impl<T> VecStreamCursor<T> {
    pub fn new(items: Vec<StreamSegment<T>>) -> Self {
        Self { items, index: 0 }
    }
}

impl<T: Clone> StreamCursor for VecStreamCursor<T> {
    type Item = T;

    fn next_segment(&mut self) -> Option<StreamSegment<Self::Item>> {
        let next = self.items.get(self.index).cloned();
        if next.is_some() {
            self.index = self.index.saturating_add(1);
        }
        next
    }
}

#[cfg(test)]
mod tests {
    use crate::{StreamBoundary, StreamSampleId, StreamSegment, StreamStepMetadata};

    use super::{StreamCursor, VecStreamCursor};

    #[test]
    fn vec_stream_cursor_iterates_dummy_segments_in_order() {
        let items = vec![
            StreamSegment::new(
                3_u32,
                StreamStepMetadata::new(
                    StreamSampleId::new(1, 2, 0),
                    StreamBoundary::ResetEpisode,
                    0,
                    0,
                ),
            ),
            StreamSegment::new(
                5_u32,
                StreamStepMetadata::new(
                    StreamSampleId::new(1, 2, 1),
                    StreamBoundary::Continue,
                    1,
                    1,
                ),
            ),
        ];
        let mut cursor = VecStreamCursor::new(items);
        assert_eq!(cursor.next_segment().map(|seg| seg.payload), Some(3));
        assert_eq!(cursor.next_segment().map(|seg| seg.payload), Some(5));
        assert!(cursor.next_segment().is_none());
    }
}
