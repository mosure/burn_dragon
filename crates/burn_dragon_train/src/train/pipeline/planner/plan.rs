use std::cmp::Reverse;
use std::ops::Range;

use anyhow::{Result, anyhow};

use crate::{ParallelPipelineConfig, PipelinePartitionKind, PipelineScheduleKind};

use super::types::{
    PipelineEventKind, PipelinePlan, PipelinePlanMetrics, PipelineRankWorkload,
    PipelineScheduleEvent, PipelineStageAssignment, SharedWeightGradientReport,
};

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
