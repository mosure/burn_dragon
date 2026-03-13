use crate::{StreamBoundary, StreamStepMetadata, policy::TargetAlignmentPolicy};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TargetAlignmentSelection {
    pub observation_index: usize,
    pub target_index: usize,
    pub horizon: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamWindowSelection {
    pub start_index: usize,
    pub observation_index: usize,
    pub target_index: usize,
    pub horizon: usize,
}

pub fn resolve_target_alignment(
    policy: TargetAlignmentPolicy,
    observation_index: usize,
    max_index: usize,
    requested_horizon: Option<usize>,
) -> Option<TargetAlignmentSelection> {
    if observation_index > max_index {
        return None;
    }

    let horizon = match policy {
        TargetAlignmentPolicy::SameStep => 0,
        TargetAlignmentPolicy::FixedFuture => requested_horizon.unwrap_or(1).max(1),
        TargetAlignmentPolicy::VariableFuture => requested_horizon.unwrap_or(1).max(1),
    };
    let target_index = observation_index.checked_add(horizon)?;
    if target_index > max_index {
        return None;
    }
    Some(TargetAlignmentSelection {
        observation_index,
        target_index,
        horizon,
    })
}

pub fn resolve_stream_target_alignment(
    policy: TargetAlignmentPolicy,
    stream: &[StreamStepMetadata],
    observation_index: usize,
    requested_horizon: Option<usize>,
) -> Option<TargetAlignmentSelection> {
    let selection = resolve_target_alignment(
        policy,
        observation_index,
        stream.len().checked_sub(1)?,
        requested_horizon,
    )?;
    if stream_alignment_is_valid(stream, selection.observation_index, selection.target_index) {
        Some(selection)
    } else {
        None
    }
}

pub fn resolve_stream_window_alignment(
    policy: TargetAlignmentPolicy,
    stream: &[StreamStepMetadata],
    observation_index: usize,
    window_len: usize,
    requested_horizon: Option<usize>,
) -> Option<StreamWindowSelection> {
    let selection =
        resolve_stream_target_alignment(policy, stream, observation_index, requested_horizon)?;
    let window_len = window_len.max(1);
    let start_index = selection
        .observation_index
        .checked_add(1)?
        .checked_sub(window_len)?;
    if stream_window_is_valid(stream, start_index, selection.observation_index) {
        Some(StreamWindowSelection {
            start_index,
            observation_index: selection.observation_index,
            target_index: selection.target_index,
            horizon: selection.horizon,
        })
    } else {
        None
    }
}

fn stream_alignment_is_valid(
    stream: &[StreamStepMetadata],
    observation_index: usize,
    target_index: usize,
) -> bool {
    if observation_index >= stream.len() || target_index >= stream.len() || observation_index > target_index {
        return false;
    }
    let observation = stream[observation_index];
    let source_id = observation.sample_id.source_id;
    let episode_id = observation.sample_id.episode_id;
    let mut previous_time = observation.absolute_time;
    for metadata in stream.iter().take(target_index + 1).skip(observation_index + 1) {
        if metadata.sample_id.source_id != source_id || metadata.sample_id.episode_id != episode_id {
            return false;
        }
        if metadata.absolute_time < previous_time {
            return false;
        }
        if metadata.boundary.resets_episode() {
            return false;
        }
        previous_time = metadata.absolute_time;
    }
    true
}

fn stream_window_is_valid(stream: &[StreamStepMetadata], start_index: usize, end_index: usize) -> bool {
    if start_index >= stream.len() || end_index >= stream.len() || start_index > end_index {
        return false;
    }
    let start = stream[start_index];
    let source_id = start.sample_id.source_id;
    let episode_id = start.sample_id.episode_id;
    let mut previous_time = start.absolute_time;
    for (relative_index, metadata) in stream
        .iter()
        .take(end_index + 1)
        .skip(start_index)
        .enumerate()
    {
        if metadata.sample_id.source_id != source_id || metadata.sample_id.episode_id != episode_id {
            return false;
        }
        if relative_index > 0 && metadata.absolute_time < previous_time {
            return false;
        }
        if relative_index > 0 && metadata.boundary != StreamBoundary::Continue {
            return false;
        }
        previous_time = metadata.absolute_time;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{StreamSampleId, StreamStepMetadata};

    fn metadata(
        segment_id: u64,
        step_index: usize,
        absolute_time: usize,
        boundary: StreamBoundary,
        episode_id: u64,
    ) -> StreamStepMetadata {
        StreamStepMetadata {
            sample_id: StreamSampleId {
                source_id: 1,
                episode_id,
                segment_id,
            },
            boundary,
            step_index,
            absolute_time,
        }
    }

    #[test]
    fn target_alignment_same_step_keeps_observation_index() {
        let selection = resolve_target_alignment(TargetAlignmentPolicy::SameStep, 3, 8, None)
            .expect("selection");
        assert_eq!(selection.target_index, 3);
        assert_eq!(selection.horizon, 0);
    }

    #[test]
    fn target_alignment_fixed_future_requires_available_horizon() {
        let selection =
            resolve_target_alignment(TargetAlignmentPolicy::FixedFuture, 2, 8, Some(3))
                .expect("selection");
        assert_eq!(selection.target_index, 5);
        assert!(resolve_target_alignment(TargetAlignmentPolicy::FixedFuture, 7, 8, Some(3)).is_none());
    }

    #[test]
    fn target_alignment_variable_future_accepts_runtime_horizon() {
        let selection =
            resolve_target_alignment(TargetAlignmentPolicy::VariableFuture, 1, 6, Some(4))
                .expect("selection");
        assert_eq!(selection.target_index, 5);
    }

    #[test]
    fn stream_target_alignment_rejects_cross_episode_future() {
        let stream = vec![
            metadata(0, 0, 0, StreamBoundary::ResetEpisode, 1),
            metadata(1, 1, 1, StreamBoundary::Continue, 1),
            metadata(0, 0, 2, StreamBoundary::ResetEpisode, 2),
        ];
        assert!(
            resolve_stream_target_alignment(TargetAlignmentPolicy::FixedFuture, &stream, 1, Some(1))
                .is_none()
        );
    }

    #[test]
    fn stream_window_alignment_requires_contiguous_same_episode_window() {
        let stream = vec![
            metadata(0, 0, 0, StreamBoundary::ResetEpisode, 1),
            metadata(1, 1, 1, StreamBoundary::Continue, 1),
            metadata(2, 2, 2, StreamBoundary::Continue, 1),
        ];
        let selection = resolve_stream_window_alignment(
            TargetAlignmentPolicy::FixedFuture,
            &stream,
            1,
            2,
            Some(1),
        )
        .expect("window selection");
        assert_eq!(selection.start_index, 0);
        assert_eq!(selection.target_index, 2);
        assert_eq!(selection.horizon, 1);
    }

    #[test]
    fn stream_window_alignment_rejects_reset_inside_window() {
        let stream = vec![
            metadata(0, 0, 0, StreamBoundary::ResetEpisode, 1),
            metadata(1, 1, 1, StreamBoundary::Continue, 1),
            metadata(0, 0, 2, StreamBoundary::ResetEpisode, 2),
        ];
        assert!(
            resolve_stream_window_alignment(
                TargetAlignmentPolicy::SameStep,
                &stream,
                2,
                2,
                None,
            )
            .is_none()
        );
    }
}
