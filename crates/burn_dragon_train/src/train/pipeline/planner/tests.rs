use std::collections::HashSet;

use super::*;
use crate::{
    ParallelPipelineCacheConfig, ParallelPipelineConfig, PipelineCacheEvictionKind,
    PipelineCachePolicy, PipelineCommunicationKind, PipelinePartitionKind, PipelineScheduleKind,
};

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
    let report = simulate_pipeline_communication(&plan, config.communication, &config.cache, 2, 32)
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
    let report = simulate_pipeline_communication(&plan, config.communication, &config.cache, 2, 32)
        .expect("report");

    assert_eq!(report.cache_hits, 0);
    assert_eq!(report.bytes_saved(), 0);
    assert_eq!(
        report.raw_payload_bytes_requested,
        report.payload_bytes_transmitted
    );
}
