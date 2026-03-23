use burn_dragon_train::train::pipeline::{build_pipeline_plan, simulate_pipeline_communication};
use burn_dragon_train::{
    ParallelPipelineCacheConfig, ParallelPipelineConfig, PipelineCacheEvictionKind,
    PipelineCachePolicy, PipelineCommunicationKind, PipelinePartitionKind, PipelineScheduleKind,
};
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

#[derive(Clone)]
struct PipelineCacheBenchCase {
    name: &'static str,
    n_layer: usize,
    layers_per_block: usize,
    payload_bytes: usize,
    pipeline: ParallelPipelineConfig,
}

fn pipeline_cache_config() -> ParallelPipelineCacheConfig {
    ParallelPipelineCacheConfig {
        enabled: true,
        policy: PipelineCachePolicy::ResidentBlockSummaries,
        reuse_across_backward: true,
        max_inflight_microbatches: 8,
        eviction: PipelineCacheEvictionKind::StepBoundary,
        transport_dtype: Default::default(),
    }
}

fn benchmark_cases() -> Vec<PipelineCacheBenchCase> {
    vec![
        PipelineCacheBenchCase {
            name: "small",
            n_layer: 16,
            layers_per_block: 2,
            payload_bytes: 128 * 1024,
            pipeline: ParallelPipelineConfig {
                enabled: true,
                stage_count: 4,
                virtual_stages_per_rank: 1,
                schedule: PipelineScheduleKind::Interleaved1f1b,
                microbatches: 8,
                partition: PipelinePartitionKind::LayerContiguous,
                activation_checkpointing: false,
                shared_weight_sync: Default::default(),
                communication: PipelineCommunicationKind::BlockResidualCache,
                cache: pipeline_cache_config(),
            },
        },
        PipelineCacheBenchCase {
            name: "medium",
            n_layer: 32,
            layers_per_block: 4,
            payload_bytes: 512 * 1024,
            pipeline: ParallelPipelineConfig {
                enabled: true,
                stage_count: 4,
                virtual_stages_per_rank: 2,
                schedule: PipelineScheduleKind::Interleaved1f1b,
                microbatches: 16,
                partition: PipelinePartitionKind::LayerContiguous,
                activation_checkpointing: false,
                shared_weight_sync: Default::default(),
                communication: PipelineCommunicationKind::BlockResidualCache,
                cache: pipeline_cache_config(),
            },
        },
    ]
}

fn bench_pipeline_cache(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_communication");

    for case in benchmark_cases() {
        let plan = build_pipeline_plan(case.n_layer, &case.pipeline).expect("pipeline plan");
        let throughput_bytes = (case.payload_bytes * plan.events.len()) as u64;
        group.throughput(criterion::Throughput::Bytes(throughput_bytes));

        group.bench_with_input(
            BenchmarkId::new("activation_tensor", case.name),
            &case,
            |b, case| {
                b.iter(|| {
                    let report = simulate_pipeline_communication(
                        &plan,
                        PipelineCommunicationKind::ActivationTensor,
                        &case.pipeline.cache,
                        case.layers_per_block,
                        case.payload_bytes,
                    )
                    .expect("activation report");
                    black_box(report.payload_bytes_transmitted);
                    black_box(report.forward_transfer_requests);
                    black_box(report.backward_transfer_requests);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("block_residual_cache", case.name),
            &case,
            |b, case| {
                b.iter(|| {
                    let report = simulate_pipeline_communication(
                        &plan,
                        PipelineCommunicationKind::BlockResidualCache,
                        &case.pipeline.cache,
                        case.layers_per_block,
                        case.payload_bytes,
                    )
                    .expect("block cache report");
                    black_box(report.payload_bytes_transmitted);
                    black_box(report.bytes_saved());
                    black_box(report.cache_hit_rate());
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_pipeline_cache);
criterion_main!(benches);
