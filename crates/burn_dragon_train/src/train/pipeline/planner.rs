use std::cmp::Reverse;
use std::collections::{HashSet, VecDeque};
use std::ops::Range;

use anyhow::{Result, anyhow};

use crate::{
    ParallelPipelineCacheConfig, ParallelPipelineConfig, PipelineCacheEvictionKind,
    PipelineCachePolicy, PipelineCommunicationKind, PipelinePartitionKind, PipelineScheduleKind,
};

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
    pub schedule: PipelineScheduleKind,
    pub partition: PipelinePartitionKind,
    pub stage_assignments: Vec<PipelineStageAssignment>,
    pub events: Vec<PipelineScheduleEvent>,
    pub metrics: PipelinePlanMetrics,
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

pub fn build_pipeline_rank_workload(
    plan: &PipelinePlan,
    global_rank: usize,
    pipeline_stage_id: usize,
    data_parallel_rank: usize,
) -> PipelineRankWorkload {
    PipelineRankWorkload {
        global_rank,
        pipeline_stage_id,
        data_parallel_rank,
        stage_assignments: plan
            .stage_assignments_for_physical_stage(pipeline_stage_id)
            .into_iter()
            .cloned()
            .collect(),
        forward_events: plan
            .forward_events_for_physical_stage(pipeline_stage_id)
            .into_iter()
            .cloned()
            .collect(),
        backward_events: plan
            .backward_events_for_physical_stage(pipeline_stage_id)
            .into_iter()
            .cloned()
            .collect(),
    }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReadyTask {
    kind: PipelineEventKind,
    microbatch_id: usize,
    virtual_stage_id: usize,
    physical_stage_id: usize,
    local_stage_index: usize,
}

pub fn build_pipeline_plan(
    n_layer: usize,
    pipeline: &ParallelPipelineConfig,
) -> Result<PipelinePlan> {
    if !pipeline.enabled {
        return Err(anyhow!(
            "pipeline planning requires parallel.pipeline.enabled = true"
        ));
    }
    if n_layer == 0 {
        return Err(anyhow!("pipeline planning requires n_layer > 0"));
    }
    if pipeline.stage_count == 0 {
        return Err(anyhow!(
            "pipeline planning requires parallel.pipeline.stage_count > 0"
        ));
    }
    if pipeline.virtual_stages_per_rank == 0 {
        return Err(anyhow!(
            "pipeline planning requires parallel.pipeline.virtual_stages_per_rank > 0"
        ));
    }
    if pipeline.virtual_stages_per_rank > pipeline.stage_count {
        return Err(anyhow!(
            "parallel.pipeline.virtual_stages_per_rank must be <= parallel.pipeline.stage_count (got {} > {})",
            pipeline.virtual_stages_per_rank,
            pipeline.stage_count
        ));
    }
    if pipeline.microbatches == 0 {
        return Err(anyhow!(
            "pipeline planning requires parallel.pipeline.microbatches > 0"
        ));
    }
    if matches!(pipeline.schedule, PipelineScheduleKind::Interleaved1f1b)
        && pipeline.microbatches < pipeline.stage_count
    {
        return Err(anyhow!(
            "parallel.pipeline.microbatches must be >= parallel.pipeline.stage_count for interleaved_1f1b (got {} < {})",
            pipeline.microbatches,
            pipeline.stage_count
        ));
    }

    let total_virtual_stages = pipeline
        .stage_count
        .checked_mul(pipeline.virtual_stages_per_rank)
        .ok_or_else(|| anyhow!("pipeline virtual-stage count overflow"))?;
    if total_virtual_stages > n_layer {
        return Err(anyhow!(
            "pipeline planning requires total virtual stages <= n_layer (got {} > {})",
            total_virtual_stages,
            n_layer
        ));
    }
    if !matches!(pipeline.partition, PipelinePartitionKind::LayerContiguous) {
        return Err(anyhow!(
            "unsupported pipeline partition {:?}",
            pipeline.partition
        ));
    }

    let stage_assignments = partition_layers(
        n_layer,
        pipeline.stage_count,
        pipeline.virtual_stages_per_rank,
    )?;
    let events = build_pipeline_events(
        pipeline.stage_count,
        &stage_assignments,
        pipeline.microbatches,
        pipeline.schedule,
    )?;
    let metrics = build_pipeline_metrics(pipeline.stage_count, &events);

    Ok(PipelinePlan {
        physical_stage_count: pipeline.stage_count,
        virtual_stages_per_rank: pipeline.virtual_stages_per_rank,
        total_virtual_stages,
        microbatches: pipeline.microbatches,
        schedule: pipeline.schedule,
        partition: pipeline.partition,
        stage_assignments,
        events,
        metrics,
    })
}

pub fn split_microbatch_ranges(
    total_items: usize,
    microbatches: usize,
) -> Result<Vec<Range<usize>>> {
    if microbatches == 0 {
        return Err(anyhow!("microbatch split requires microbatches > 0"));
    }
    if total_items < microbatches {
        return Err(anyhow!(
            "microbatch split requires total_items >= microbatches (got {} < {})",
            total_items,
            microbatches
        ));
    }

    let base = total_items / microbatches;
    let remainder = total_items % microbatches;
    let mut start = 0;
    let mut ranges = Vec::with_capacity(microbatches);
    for index in 0..microbatches {
        let len = base + usize::from(index < remainder);
        let end = start + len;
        ranges.push(start..end);
        start = end;
    }
    Ok(ranges)
}

pub fn simulate_shared_weight_gradient_merge<F>(
    plan: &PipelinePlan,
    learning_rate: f32,
    initial_weight: f32,
    mut gradient_fn: F,
) -> SharedWeightGradientReport
where
    F: FnMut(usize, usize) -> f32,
{
    let mut reference_gradient = 0.0f32;
    let mut stage_local_gradients = vec![0.0f32; plan.physical_stage_count];

    for assignment in &plan.stage_assignments {
        for layer_index in assignment.layer_range.clone() {
            for microbatch_id in 0..plan.microbatches {
                let gradient = gradient_fn(layer_index, microbatch_id);
                reference_gradient += gradient;
                stage_local_gradients[assignment.physical_stage_id] += gradient;
            }
        }
    }

    let merged_gradient = stage_local_gradients.iter().copied().sum::<f32>();
    let updated_weight = initial_weight - learning_rate * merged_gradient;

    SharedWeightGradientReport {
        reference_gradient,
        merged_gradient,
        stage_local_gradients,
        updated_weight,
    }
}

pub fn simulate_pipeline_communication(
    plan: &PipelinePlan,
    communication: PipelineCommunicationKind,
    cache: &ParallelPipelineCacheConfig,
    layers_per_block: usize,
    payload_bytes: usize,
) -> Result<PipelineCommunicationReport> {
    if layers_per_block == 0 {
        return Err(anyhow!(
            "communication simulation requires layers_per_block > 0"
        ));
    }

    let mut manager = CrossStageCacheManager::new(cache);
    let mut report = PipelineCommunicationReport {
        stage_transmitted_bytes: vec![0; plan.physical_stage_count],
        ..PipelineCommunicationReport::default()
    };

    for event in &plan.events {
        let key = match (communication, event.kind) {
            (PipelineCommunicationKind::ActivationTensor, PipelineEventKind::Forward) => {
                forward_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::ActivationTensor, PipelineEventKind::Backward) => {
                backward_activation_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::BlockResidualCache, PipelineEventKind::Forward) => {
                forward_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::BlockResidualCache, PipelineEventKind::Backward) => {
                backward_cache_reuse_key(plan, event, layers_per_block)
            }
        };
        let Some(key) = key else {
            continue;
        };

        match event.kind {
            PipelineEventKind::Forward => report.forward_transfer_requests += 1,
            PipelineEventKind::Backward => report.backward_transfer_requests += 1,
        }

        let access = match (communication, event.kind) {
            (PipelineCommunicationKind::ActivationTensor, _) => {
                manager.access_bypass(key, payload_bytes)
            }
            (PipelineCommunicationKind::BlockResidualCache, PipelineEventKind::Forward) => {
                manager.access_forward(key, payload_bytes)
            }
            (PipelineCommunicationKind::BlockResidualCache, PipelineEventKind::Backward) => {
                manager.access_backward(key, payload_bytes)
            }
        };

        if access.transmitted_bytes > 0 {
            report.stage_transmitted_bytes[key.source_stage_id] += access.transmitted_bytes;
        }
    }

    let stats = manager.stats().clone();
    report.raw_payload_bytes_requested = stats.raw_payload_bytes_requested;
    report.payload_bytes_transmitted = stats.payload_bytes_transmitted;
    report.cache_hits = stats.cache_hits;
    report.cache_misses = stats.cache_misses;
    report.resend_count_avoided = stats.resend_count_avoided;
    report.backward_reuse_hits = stats.backward_reuse_hits;
    report.invalidated_entries = stats.invalidated_entries;

    Ok(report)
}

fn partition_layers(
    n_layer: usize,
    physical_stage_count: usize,
    virtual_stages_per_rank: usize,
) -> Result<Vec<PipelineStageAssignment>> {
    let total_virtual_stages = physical_stage_count
        .checked_mul(virtual_stages_per_rank)
        .ok_or_else(|| anyhow!("pipeline partition overflow"))?;
    let base = n_layer / total_virtual_stages;
    let remainder = n_layer % total_virtual_stages;
    let mut layer_cursor = 0;
    let mut assignments = Vec::with_capacity(total_virtual_stages);

    for virtual_stage_id in 0..total_virtual_stages {
        let layer_count = base + usize::from(virtual_stage_id < remainder);
        let next_cursor = layer_cursor + layer_count;
        assignments.push(PipelineStageAssignment {
            virtual_stage_id,
            physical_stage_id: virtual_stage_id % physical_stage_count,
            local_stage_index: virtual_stage_id / physical_stage_count,
            layer_range: layer_cursor..next_cursor,
        });
        layer_cursor = next_cursor;
    }

    Ok(assignments)
}

fn build_pipeline_events(
    physical_stage_count: usize,
    stage_assignments: &[PipelineStageAssignment],
    microbatches: usize,
    schedule: PipelineScheduleKind,
) -> Result<Vec<PipelineScheduleEvent>> {
    let total_virtual_stages = stage_assignments.len();
    let total_events = microbatches
        .checked_mul(total_virtual_stages)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| anyhow!("pipeline event count overflow"))?;
    let assignments_by_physical =
        assignments_by_physical_stage(physical_stage_count, stage_assignments);
    let mut forward_done = vec![vec![false; total_virtual_stages]; microbatches];
    let mut backward_done = vec![vec![false; total_virtual_stages]; microbatches];
    let mut events = Vec::with_capacity(total_events);
    let mut tick = 0usize;

    while events.len() < total_events {
        let all_forwards_complete = forward_done
            .iter()
            .flatten()
            .copied()
            .all(|completed| completed);
        let mut ready = Vec::with_capacity(physical_stage_count);

        for (physical_stage_id, assignments) in assignments_by_physical.iter().enumerate() {
            if let Some(task) = choose_ready_task(
                assignments,
                &forward_done,
                &backward_done,
                schedule,
                all_forwards_complete,
            ) {
                ready.push((physical_stage_id, task));
            }
        }

        if ready.is_empty() {
            return Err(anyhow!(
                "pipeline schedule deadlocked at tick {tick} with schedule {:?}",
                schedule
            ));
        }

        for (physical_stage_id, task) in ready {
            match task.kind {
                PipelineEventKind::Forward => {
                    forward_done[task.microbatch_id][task.virtual_stage_id] = true;
                }
                PipelineEventKind::Backward => {
                    backward_done[task.microbatch_id][task.virtual_stage_id] = true;
                }
            }
            events.push(PipelineScheduleEvent {
                tick,
                physical_stage_id,
                virtual_stage_id: task.virtual_stage_id,
                local_stage_index: task.local_stage_index,
                microbatch_id: task.microbatch_id,
                kind: task.kind,
            });
        }
        tick += 1;
    }

    Ok(events)
}

fn assignments_by_physical_stage<'a>(
    physical_stage_count: usize,
    stage_assignments: &'a [PipelineStageAssignment],
) -> Vec<Vec<&'a PipelineStageAssignment>> {
    let mut assignments = vec![Vec::new(); physical_stage_count];
    for assignment in stage_assignments {
        assignments[assignment.physical_stage_id].push(assignment);
    }
    assignments
}

fn choose_ready_task(
    stage_assignments: &[&PipelineStageAssignment],
    forward_done: &[Vec<bool>],
    backward_done: &[Vec<bool>],
    schedule: PipelineScheduleKind,
    all_forwards_complete: bool,
) -> Option<ReadyTask> {
    let mut ready_forwards = Vec::new();
    let mut ready_backwards = Vec::new();

    for assignment in stage_assignments {
        let virtual_stage_id = assignment.virtual_stage_id;
        for microbatch_id in 0..forward_done.len() {
            if !forward_done[microbatch_id][virtual_stage_id]
                && (virtual_stage_id == 0 || forward_done[microbatch_id][virtual_stage_id - 1])
            {
                ready_forwards.push(ReadyTask {
                    kind: PipelineEventKind::Forward,
                    microbatch_id,
                    virtual_stage_id,
                    physical_stage_id: assignment.physical_stage_id,
                    local_stage_index: assignment.local_stage_index,
                });
            }
            if !backward_done[microbatch_id][virtual_stage_id]
                && forward_done[microbatch_id][virtual_stage_id]
                && (virtual_stage_id + 1 == backward_done[microbatch_id].len()
                    || backward_done[microbatch_id][virtual_stage_id + 1])
            {
                ready_backwards.push(ReadyTask {
                    kind: PipelineEventKind::Backward,
                    microbatch_id,
                    virtual_stage_id,
                    physical_stage_id: assignment.physical_stage_id,
                    local_stage_index: assignment.local_stage_index,
                });
            }
        }
    }

    match schedule {
        PipelineScheduleKind::Gpipe => {
            if !all_forwards_complete {
                ready_forwards
                    .into_iter()
                    .min_by_key(|task| (task.microbatch_id, task.virtual_stage_id))
                    .or_else(|| {
                        ready_backwards
                            .into_iter()
                            .min_by_key(|task| (task.microbatch_id, Reverse(task.virtual_stage_id)))
                    })
            } else {
                ready_backwards
                    .into_iter()
                    .min_by_key(|task| (task.microbatch_id, Reverse(task.virtual_stage_id)))
            }
        }
        PipelineScheduleKind::Interleaved1f1b => ready_backwards
            .into_iter()
            .min_by_key(|task| {
                (
                    task.microbatch_id,
                    Reverse(task.local_stage_index),
                    Reverse(task.virtual_stage_id),
                )
            })
            .or_else(|| {
                ready_forwards.into_iter().min_by_key(|task| {
                    (
                        task.microbatch_id,
                        task.local_stage_index,
                        task.virtual_stage_id,
                    )
                })
            }),
    }
}

fn build_pipeline_metrics(
    physical_stage_count: usize,
    events: &[PipelineScheduleEvent],
) -> PipelinePlanMetrics {
    let total_ticks = events
        .iter()
        .map(|event| event.tick)
        .max()
        .map(|tick| tick + 1)
        .unwrap_or(0);
    let mut stage_busy_ticks = vec![0usize; physical_stage_count];
    for event in events {
        stage_busy_ticks[event.physical_stage_id] += 1;
    }
    let stage_idle_ticks = stage_busy_ticks
        .iter()
        .map(|busy| total_ticks.saturating_sub(*busy))
        .collect::<Vec<_>>();
    let total_slots = total_ticks.saturating_mul(physical_stage_count).max(1);
    let total_idle = stage_idle_ticks.iter().copied().sum::<usize>();

    PipelinePlanMetrics {
        total_ticks,
        total_events: events.len(),
        stage_busy_ticks,
        stage_idle_ticks,
        bubble_fraction: total_idle as f32 / total_slots as f32,
    }
}

fn forward_transfer_key(
    plan: &PipelinePlan,
    event: &PipelineScheduleEvent,
    layers_per_block: usize,
) -> Option<CrossStageCacheKey> {
    let next_virtual_stage_id = event.virtual_stage_id + 1;
    if next_virtual_stage_id >= plan.total_virtual_stages {
        return None;
    }
    let source = plan.assignment(event.virtual_stage_id);
    let destination = plan.assignment(next_virtual_stage_id);
    if source.physical_stage_id == destination.physical_stage_id {
        return None;
    }
    Some(CrossStageCacheKey {
        source_stage_id: source.physical_stage_id,
        destination_stage_id: destination.physical_stage_id,
        logical_block_id: block_id_for_assignment(source, layers_per_block),
        microbatch_id: event.microbatch_id,
        freshness_marker: 0,
    })
}

fn backward_cache_reuse_key(
    plan: &PipelinePlan,
    event: &PipelineScheduleEvent,
    layers_per_block: usize,
) -> Option<CrossStageCacheKey> {
    if event.virtual_stage_id == 0 {
        return None;
    }
    let source = plan.assignment(event.virtual_stage_id - 1);
    let destination = plan.assignment(event.virtual_stage_id);
    if source.physical_stage_id == destination.physical_stage_id {
        return None;
    }
    Some(CrossStageCacheKey {
        source_stage_id: source.physical_stage_id,
        destination_stage_id: destination.physical_stage_id,
        logical_block_id: block_id_for_assignment(source, layers_per_block),
        microbatch_id: event.microbatch_id,
        freshness_marker: 0,
    })
}

fn backward_activation_transfer_key(
    plan: &PipelinePlan,
    event: &PipelineScheduleEvent,
    layers_per_block: usize,
) -> Option<CrossStageCacheKey> {
    if event.virtual_stage_id == 0 {
        return None;
    }
    let source = plan.assignment(event.virtual_stage_id);
    let destination = plan.assignment(event.virtual_stage_id - 1);
    if source.physical_stage_id == destination.physical_stage_id {
        return None;
    }
    Some(CrossStageCacheKey {
        source_stage_id: source.physical_stage_id,
        destination_stage_id: destination.physical_stage_id,
        logical_block_id: block_id_for_assignment(destination, layers_per_block),
        microbatch_id: event.microbatch_id,
        freshness_marker: 0,
    })
}

fn block_id_for_assignment(assignment: &PipelineStageAssignment, layers_per_block: usize) -> usize {
    assignment
        .layer_range
        .end
        .saturating_sub(1)
        .checked_div(layers_per_block.max(1))
        .unwrap_or(0)
}

#[derive(Clone, Debug)]
pub struct CrossStageCacheManager {
    enabled: bool,
    policy: PipelineCachePolicy,
    reuse_across_backward: bool,
    max_inflight_microbatches: usize,
    eviction: PipelineCacheEvictionKind,
    current_freshness: Option<u64>,
    entries: HashSet<CrossStageCacheKey>,
    resident_microbatches: VecDeque<(u64, usize)>,
    stats: CrossStageCacheStats,
}

impl CrossStageCacheManager {
    pub fn new(config: &ParallelPipelineCacheConfig) -> Self {
        Self {
            enabled: config.enabled && !matches!(config.policy, PipelineCachePolicy::Disabled),
            policy: config.policy,
            reuse_across_backward: config.reuse_across_backward,
            max_inflight_microbatches: config.max_inflight_microbatches.max(1),
            eviction: config.eviction,
            current_freshness: None,
            entries: HashSet::new(),
            resident_microbatches: VecDeque::new(),
            stats: CrossStageCacheStats::default(),
        }
    }

    pub fn stats(&self) -> &CrossStageCacheStats {
        &self.stats
    }

    pub fn access_forward(
        &mut self,
        key: CrossStageCacheKey,
        payload_bytes: usize,
    ) -> CrossStageCacheAccess {
        self.access(key, payload_bytes, false)
    }

    pub fn access_backward(
        &mut self,
        key: CrossStageCacheKey,
        payload_bytes: usize,
    ) -> CrossStageCacheAccess {
        if self.enabled && self.reuse_across_backward {
            self.access(key, payload_bytes, true)
        } else {
            self.access_bypass(key, payload_bytes)
        }
    }

    pub fn access_bypass(
        &mut self,
        key: CrossStageCacheKey,
        payload_bytes: usize,
    ) -> CrossStageCacheAccess {
        self.begin_freshness(key.freshness_marker);
        self.stats.raw_payload_bytes_requested += payload_bytes;
        self.stats.payload_bytes_transmitted += payload_bytes;
        self.stats.cache_misses += 1;
        CrossStageCacheAccess {
            kind: CrossStageCacheAccessKind::Bypass,
            transmitted_bytes: payload_bytes,
        }
    }

    fn access(
        &mut self,
        key: CrossStageCacheKey,
        payload_bytes: usize,
        is_backward_reuse: bool,
    ) -> CrossStageCacheAccess {
        self.begin_freshness(key.freshness_marker);
        self.stats.raw_payload_bytes_requested += payload_bytes;

        if !self.enabled || matches!(self.policy, PipelineCachePolicy::Disabled) {
            self.stats.payload_bytes_transmitted += payload_bytes;
            self.stats.cache_misses += 1;
            return CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Bypass,
                transmitted_bytes: payload_bytes,
            };
        }

        if self.entries.contains(&key) {
            self.stats.cache_hits += 1;
            self.stats.resend_count_avoided += 1;
            if is_backward_reuse {
                self.stats.backward_reuse_hits += 1;
            }
            return CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Hit,
                transmitted_bytes: 0,
            };
        }

        self.stats.cache_misses += 1;
        self.stats.payload_bytes_transmitted += payload_bytes;
        self.insert_entry(key);
        CrossStageCacheAccess {
            kind: CrossStageCacheAccessKind::Miss,
            transmitted_bytes: payload_bytes,
        }
    }

    fn begin_freshness(&mut self, freshness_marker: u64) {
        if self.current_freshness == Some(freshness_marker) {
            return;
        }
        self.current_freshness = Some(freshness_marker);
        if matches!(self.eviction, PipelineCacheEvictionKind::StepBoundary) {
            self.clear_entries();
        }
    }

    fn clear_entries(&mut self) {
        self.stats.invalidated_entries += self.entries.len();
        self.entries.clear();
        self.resident_microbatches.clear();
    }

    fn insert_entry(&mut self, key: CrossStageCacheKey) {
        let resident = (key.freshness_marker, key.microbatch_id);
        if !self.resident_microbatches.contains(&resident) {
            while self.resident_microbatches.len() >= self.max_inflight_microbatches {
                if let Some((freshness_marker, microbatch_id)) =
                    self.resident_microbatches.pop_front()
                {
                    let before = self.entries.len();
                    self.entries.retain(|entry| {
                        !(entry.freshness_marker == freshness_marker
                            && entry.microbatch_id == microbatch_id)
                    });
                    self.stats.invalidated_entries += before.saturating_sub(self.entries.len());
                }
            }
            self.resident_microbatches.push_back(resident);
        }
        self.entries.insert(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn pipeline_config() -> ParallelPipelineConfig {
        ParallelPipelineConfig {
            enabled: true,
            stage_count: 2,
            virtual_stages_per_rank: 1,
            schedule: PipelineScheduleKind::Interleaved1f1b,
            microbatches: 4,
            partition: PipelinePartitionKind::LayerContiguous,
            activation_checkpointing: false,
            shared_weight_sync: Default::default(),
            communication: PipelineCommunicationKind::ActivationTensor,
            cache: ParallelPipelineCacheConfig::default(),
        }
    }

    #[test]
    fn split_microbatch_ranges_preserves_total_items() {
        let ranges = split_microbatch_ranges(10, 4).expect("ranges");
        assert_eq!(ranges, vec![0..3, 3..6, 6..8, 8..10]);
    }

    #[test]
    fn build_pipeline_plan_partitions_layers_contiguously_across_virtual_stages() {
        let mut config = pipeline_config();
        config.stage_count = 3;
        config.virtual_stages_per_rank = 2;
        let plan = build_pipeline_plan(10, &config).expect("plan");

        let spans = plan
            .stage_assignments
            .iter()
            .map(|assignment| {
                (
                    assignment.virtual_stage_id,
                    assignment.physical_stage_id,
                    assignment.local_stage_index,
                    assignment.layer_range.clone(),
                )
            })
            .collect::<Vec<_>>();
        assert_eq!(
            spans,
            vec![
                (0, 0, 0, 0..2),
                (1, 1, 0, 2..4),
                (2, 2, 0, 4..6),
                (3, 0, 1, 6..8),
                (4, 1, 1, 8..9),
                (5, 2, 1, 9..10),
            ]
        );
    }

    #[test]
    fn build_pipeline_plan_rejects_virtual_stages_exceeding_stage_count() {
        let mut config = pipeline_config();
        config.virtual_stages_per_rank = 3;
        let err = build_pipeline_plan(6, &config).expect_err("invalid plan should fail");
        assert!(
            err.to_string()
                .contains("virtual_stages_per_rank must be <= parallel.pipeline.stage_count"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn build_pipeline_plan_rejects_more_virtual_stages_than_layers() {
        let mut config = pipeline_config();
        config.virtual_stages_per_rank = 2;
        let err = build_pipeline_plan(3, &config).expect_err("plan should fail");
        assert!(
            err.to_string().contains("total virtual stages <= n_layer"),
            "unexpected error: {err:#}"
        );
    }

    #[test]
    fn gpipe_schedule_flushes_all_forwards_before_backwards() {
        let mut config = pipeline_config();
        config.schedule = PipelineScheduleKind::Gpipe;
        let plan = build_pipeline_plan(2, &config).expect("plan");

        let stage_one = plan
            .events
            .iter()
            .filter(|event| event.physical_stage_id == 1)
            .map(|event| (event.kind, event.microbatch_id))
            .collect::<Vec<_>>();
        assert_eq!(
            stage_one,
            vec![
                (PipelineEventKind::Forward, 0),
                (PipelineEventKind::Forward, 1),
                (PipelineEventKind::Forward, 2),
                (PipelineEventKind::Forward, 3),
                (PipelineEventKind::Backward, 0),
                (PipelineEventKind::Backward, 1),
                (PipelineEventKind::Backward, 2),
                (PipelineEventKind::Backward, 3),
            ]
        );
    }

    #[test]
    fn interleaved_1f1b_schedule_starts_backward_before_forward_phase_finishes() {
        let plan = build_pipeline_plan(2, &pipeline_config()).expect("plan");

        let stage_one = plan
            .events
            .iter()
            .filter(|event| event.physical_stage_id == 1)
            .map(|event| (event.kind, event.microbatch_id))
            .collect::<Vec<_>>();
        assert_eq!(
            stage_one,
            vec![
                (PipelineEventKind::Forward, 0),
                (PipelineEventKind::Backward, 0),
                (PipelineEventKind::Forward, 1),
                (PipelineEventKind::Backward, 1),
                (PipelineEventKind::Forward, 2),
                (PipelineEventKind::Backward, 2),
                (PipelineEventKind::Forward, 3),
                (PipelineEventKind::Backward, 3),
            ]
        );
    }

    #[test]
    fn interleaved_plan_uses_multiple_virtual_chunks_per_physical_stage() {
        let mut config = pipeline_config();
        config.virtual_stages_per_rank = 2;
        let plan = build_pipeline_plan(8, &config).expect("plan");

        let stage_zero_virtuals = plan
            .events
            .iter()
            .filter(|event| event.physical_stage_id == 0)
            .map(|event| event.virtual_stage_id)
            .collect::<HashSet<_>>();
        let stage_one_virtuals = plan
            .events
            .iter()
            .filter(|event| event.physical_stage_id == 1)
            .map(|event| event.virtual_stage_id)
            .collect::<HashSet<_>>();

        assert_eq!(stage_zero_virtuals, HashSet::from([0usize, 2usize]));
        assert_eq!(stage_one_virtuals, HashSet::from([1usize, 3usize]));
    }

    #[test]
    fn build_pipeline_rank_workload_collects_stage_owned_assignments_and_events() {
        let mut config = pipeline_config();
        config.virtual_stages_per_rank = 2;
        let plan = build_pipeline_plan(8, &config).expect("plan");
        let workload = build_pipeline_rank_workload(&plan, 2, 0, 1);

        assert_eq!(workload.global_rank, 2);
        assert_eq!(workload.pipeline_stage_id, 0);
        assert_eq!(workload.data_parallel_rank, 1);
        assert_eq!(
            workload
                .stage_assignments
                .iter()
                .map(|assignment| assignment.virtual_stage_id)
                .collect::<Vec<_>>(),
            vec![0, 2]
        );
        assert!(
            workload
                .forward_events
                .iter()
                .all(|event| event.physical_stage_id == 0)
        );
        assert!(
            workload
                .backward_events
                .iter()
                .all(|event| event.physical_stage_id == 0)
        );
        assert_eq!(
            workload.forward_events.len() + workload.backward_events.len(),
            plan.events_for_physical_stage(0).len()
        );
    }

    #[test]
    fn shared_weight_gradient_merge_matches_reference_sum() {
        let mut config = pipeline_config();
        config.stage_count = 3;
        config.virtual_stages_per_rank = 1;
        config.microbatches = 5;
        let plan = build_pipeline_plan(7, &config).expect("plan");
        let report = simulate_shared_weight_gradient_merge(&plan, 0.1, 1.0, |layer, microbatch| {
            layer as f32 + microbatch as f32 * 0.25
        });

        assert!((report.reference_gradient - report.merged_gradient).abs() < 1e-6);
        assert_eq!(report.stage_local_gradients.len(), 3);
        assert!((report.updated_weight - (1.0 - 0.1 * report.reference_gradient)).abs() < 1e-6);
    }

    #[test]
    fn cross_stage_cache_hits_on_backward_reuse() {
        let mut config = pipeline_config();
        config.communication = PipelineCommunicationKind::BlockResidualCache;
        config.cache = ParallelPipelineCacheConfig {
            enabled: true,
            policy: PipelineCachePolicy::ResidentBlockSummaries,
            reuse_across_backward: true,
            max_inflight_microbatches: 4,
            eviction: PipelineCacheEvictionKind::StepBoundary,
            transport_dtype: Default::default(),
        };
        let plan = build_pipeline_plan(4, &config).expect("plan");
        let report =
            simulate_pipeline_communication(&plan, config.communication, &config.cache, 2, 128)
                .expect("report");

        assert!(report.cache_hits > 0);
        assert!(report.backward_reuse_hits > 0);
        assert!(report.bytes_saved() > 0);
        assert!(report.payload_bytes_transmitted < report.raw_payload_bytes_requested);
    }

    #[test]
    fn cross_stage_cache_step_boundary_invalidation_forces_resend() {
        let cache = ParallelPipelineCacheConfig {
            enabled: true,
            policy: PipelineCachePolicy::ResidentBlockSummaries,
            reuse_across_backward: true,
            max_inflight_microbatches: 2,
            eviction: PipelineCacheEvictionKind::StepBoundary,
            transport_dtype: Default::default(),
        };
        let mut manager = CrossStageCacheManager::new(&cache);
        let key = CrossStageCacheKey {
            source_stage_id: 0,
            destination_stage_id: 1,
            logical_block_id: 2,
            microbatch_id: 0,
            freshness_marker: 0,
        };
        assert_eq!(
            manager.access_forward(key, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Miss,
                transmitted_bytes: 64,
            }
        );
        assert_eq!(
            manager.access_forward(key, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Hit,
                transmitted_bytes: 0,
            }
        );

        let next_step_key = CrossStageCacheKey {
            freshness_marker: 1,
            ..key
        };
        assert_eq!(
            manager.access_forward(next_step_key, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Miss,
                transmitted_bytes: 64,
            }
        );
        assert!(manager.stats().invalidated_entries > 0);
    }

    #[test]
    fn cross_stage_cache_max_inflight_eviction_evicts_oldest_microbatch() {
        let cache = ParallelPipelineCacheConfig {
            enabled: true,
            policy: PipelineCachePolicy::ResidentBlockSummaries,
            reuse_across_backward: true,
            max_inflight_microbatches: 1,
            eviction: PipelineCacheEvictionKind::StepBoundary,
            transport_dtype: Default::default(),
        };
        let mut manager = CrossStageCacheManager::new(&cache);
        let key_a = CrossStageCacheKey {
            source_stage_id: 0,
            destination_stage_id: 1,
            logical_block_id: 2,
            microbatch_id: 0,
            freshness_marker: 0,
        };
        let key_b = CrossStageCacheKey {
            microbatch_id: 1,
            ..key_a
        };

        assert_eq!(
            manager.access_forward(key_a, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Miss,
                transmitted_bytes: 64,
            }
        );
        assert_eq!(
            manager.access_forward(key_b, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Miss,
                transmitted_bytes: 64,
            }
        );
        assert_eq!(
            manager.access_backward(key_a, 64),
            CrossStageCacheAccess {
                kind: CrossStageCacheAccessKind::Miss,
                transmitted_bytes: 64,
            }
        );
        assert!(manager.stats().invalidated_entries > 0);
    }

    #[test]
    fn block_residual_cache_without_backward_reuse_sends_all_payloads() {
        let mut config = pipeline_config();
        config.communication = PipelineCommunicationKind::BlockResidualCache;
        config.cache = ParallelPipelineCacheConfig {
            enabled: true,
            policy: PipelineCachePolicy::ResidentBlockSummaries,
            reuse_across_backward: false,
            max_inflight_microbatches: 4,
            eviction: PipelineCacheEvictionKind::StepBoundary,
            transport_dtype: Default::default(),
        };
        let plan = build_pipeline_plan(4, &config).expect("plan");
        let report =
            simulate_pipeline_communication(&plan, config.communication, &config.cache, 2, 32)
                .expect("report");

        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.backward_reuse_hits, 0);
        assert_eq!(report.bytes_saved(), 0);
        assert_eq!(
            report.raw_payload_bytes_requested,
            report.payload_bytes_transmitted
        );
    }

    #[test]
    fn activation_tensor_simulation_never_uses_cache() {
        let mut config = pipeline_config();
        config.communication = PipelineCommunicationKind::ActivationTensor;
        config.cache = ParallelPipelineCacheConfig {
            enabled: true,
            policy: PipelineCachePolicy::ResidentBlockSummaries,
            reuse_across_backward: true,
            max_inflight_microbatches: 4,
            eviction: PipelineCacheEvictionKind::StepBoundary,
            transport_dtype: Default::default(),
        };
        let plan = build_pipeline_plan(4, &config).expect("plan");
        let report =
            simulate_pipeline_communication(&plan, config.communication, &config.cache, 2, 32)
                .expect("report");

        assert_eq!(report.cache_hits, 0);
        assert_eq!(report.bytes_saved(), 0);
        assert_eq!(
            report.raw_payload_bytes_requested,
            report.payload_bytes_transmitted
        );
    }
}
