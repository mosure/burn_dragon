use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StreamSampleId {
    pub source_id: u64,
    pub episode_id: u64,
    pub segment_id: u64,
}

impl StreamSampleId {
    pub fn new(source_id: u64, episode_id: u64, segment_id: u64) -> Self {
        Self {
            source_id,
            episode_id,
            segment_id,
        }
    }
}
