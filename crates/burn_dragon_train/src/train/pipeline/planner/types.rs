use std::ops::Range;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PipelineEventKind {
    Forward,
    Backward,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineStageAssignment {
    pub virtual_stage_id: usize,
    pub physical_stage_id: usize,
    pub local_stage_index: usize,
    pub layer_range: Range<usize>,
}

impl PipelineStageAssignment {
    pub fn layer_count(&self) -> usize {
        self.layer_range.end.saturating_sub(self.layer_range.start)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineScheduleEvent {
    pub tick: usize,
    pub physical_stage_id: usize,
    pub virtual_stage_id: usize,
    pub local_stage_index: usize,
    pub microbatch_id: usize,
    pub kind: PipelineEventKind,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelinePlanMetrics {
    pub total_ticks: usize,
    pub total_events: usize,
    pub stage_busy_ticks: Vec<usize>,
    pub stage_idle_ticks: Vec<usize>,
    pub bubble_fraction: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PipelinePlan {
    pub physical_stage_count: usize,
    pub virtual_stages_per_rank: usize,
    pub total_virtual_stages: usize,
    pub microbatches: usize,
    pub schedule: crate::PipelineScheduleKind,
    pub partition: crate::PipelinePartitionKind,
    pub stage_assignments: Vec<PipelineStageAssignment>,
    pub events: Vec<PipelineScheduleEvent>,
    pub metrics: PipelinePlanMetrics,
}

impl PipelinePlan {
    pub fn summary(&self) -> String {
        format!(
            "schedule={:?} physical_stages={} virtual_stages_per_rank={} total_virtual_stages={} microbatches={} total_ticks={} bubble_fraction={:.3}",
            self.schedule,
            self.physical_stage_count,
            self.virtual_stages_per_rank,
            self.total_virtual_stages,
            self.microbatches,
            self.metrics.total_ticks,
            self.metrics.bubble_fraction,
        )
    }

    pub fn assignment(&self, virtual_stage_id: usize) -> &PipelineStageAssignment {
        &self.stage_assignments[virtual_stage_id]
    }

    pub fn stage_assignments_for_physical_stage(
        &self,
        physical_stage_id: usize,
    ) -> Vec<&PipelineStageAssignment> {
        self.stage_assignments
            .iter()
            .filter(|assignment| assignment.physical_stage_id == physical_stage_id)
            .collect()
    }

    pub fn events_for_physical_stage(
        &self,
        physical_stage_id: usize,
    ) -> Vec<&PipelineScheduleEvent> {
        self.events
            .iter()
            .filter(|event| event.physical_stage_id == physical_stage_id)
            .collect()
    }

    pub fn forward_events_for_physical_stage(
        &self,
        physical_stage_id: usize,
    ) -> Vec<&PipelineScheduleEvent> {
        self.events_for_physical_stage(physical_stage_id)
            .into_iter()
            .filter(|event| matches!(event.kind, PipelineEventKind::Forward))
            .collect()
    }

    pub fn backward_events_for_physical_stage(
        &self,
        physical_stage_id: usize,
    ) -> Vec<&PipelineScheduleEvent> {
        self.events_for_physical_stage(physical_stage_id)
            .into_iter()
            .filter(|event| matches!(event.kind, PipelineEventKind::Backward))
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PipelineRankWorkload {
    pub global_rank: usize,
    pub pipeline_stage_id: usize,
    pub data_parallel_rank: usize,
    pub stage_assignments: Vec<PipelineStageAssignment>,
    pub forward_events: Vec<PipelineScheduleEvent>,
    pub backward_events: Vec<PipelineScheduleEvent>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SharedWeightGradientReport {
    pub reference_gradient: f32,
    pub merged_gradient: f32,
    pub stage_local_gradients: Vec<f32>,
    pub updated_weight: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CrossStageCacheKey {
    pub source_stage_id: usize,
    pub destination_stage_id: usize,
    pub logical_block_id: usize,
    pub microbatch_id: usize,
    pub freshness_marker: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CrossStageCacheAccessKind {
    Hit,
    Miss,
    Bypass,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossStageCacheAccess {
    pub kind: CrossStageCacheAccessKind,
    pub transmitted_bytes: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CrossStageCacheStats {
    pub raw_payload_bytes_requested: usize,
    pub payload_bytes_transmitted: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub resend_count_avoided: usize,
    pub backward_reuse_hits: usize,
    pub invalidated_entries: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PipelineCommunicationReport {
    pub raw_payload_bytes_requested: usize,
    pub payload_bytes_transmitted: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub resend_count_avoided: usize,
    pub backward_reuse_hits: usize,
    pub invalidated_entries: usize,
    pub forward_transfer_requests: usize,
    pub backward_transfer_requests: usize,
    pub stage_transmitted_bytes: Vec<usize>,
}

impl PipelineCommunicationReport {
    pub fn bytes_saved(&self) -> usize {
        self.raw_payload_bytes_requested
            .saturating_sub(self.payload_bytes_transmitted)
    }

    pub fn cache_hit_rate(&self) -> f32 {
        let accesses = self.cache_hits + self.cache_misses;
        if accesses == 0 {
            0.0
        } else {
            self.cache_hits as f32 / accesses as f32
        }
    }
}
