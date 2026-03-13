use serde::{Deserialize, Serialize};

use crate::{StreamBoundary, StreamSampleId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamStepMetadata {
    pub sample_id: StreamSampleId,
    pub boundary: StreamBoundary,
    pub step_index: usize,
    pub absolute_time: usize,
}

impl StreamStepMetadata {
    pub fn new(
        sample_id: StreamSampleId,
        boundary: StreamBoundary,
        step_index: usize,
        absolute_time: usize,
    ) -> Self {
        Self {
            sample_id,
            boundary,
            step_index,
            absolute_time,
        }
    }

    pub fn should_reset_state(self) -> bool {
        self.boundary.resets_episode()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamSegment<T> {
    pub payload: T,
    pub stream: StreamStepMetadata,
}

impl<T> StreamSegment<T> {
    pub fn new(payload: T, stream: StreamStepMetadata) -> Self {
        Self { payload, stream }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CollatedStreamBatch<T> {
    pub payload: T,
    pub stream: Vec<StreamStepMetadata>,
}

impl<T> CollatedStreamBatch<T> {
    pub fn new(payload: T, stream: Vec<StreamStepMetadata>) -> Self {
        Self { payload, stream }
    }
}
