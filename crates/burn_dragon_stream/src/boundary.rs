use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamBoundary {
    #[default]
    Continue,
    ResetEpisode,
    ResetAll,
}

impl StreamBoundary {
    pub fn resets_episode(self) -> bool {
        matches!(self, Self::ResetEpisode | Self::ResetAll)
    }

    pub fn resets_all(self) -> bool {
        matches!(self, Self::ResetAll)
    }
}

#[cfg(test)]
mod tests {
    use super::StreamBoundary;

    #[test]
    fn boundary_helpers_report_reset_semantics() {
        assert!(!StreamBoundary::Continue.resets_episode());
        assert!(!StreamBoundary::Continue.resets_all());
        assert!(StreamBoundary::ResetEpisode.resets_episode());
        assert!(!StreamBoundary::ResetEpisode.resets_all());
        assert!(StreamBoundary::ResetAll.resets_episode());
        assert!(StreamBoundary::ResetAll.resets_all());
    }
}
