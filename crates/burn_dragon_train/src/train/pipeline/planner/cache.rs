use std::collections::{HashSet, VecDeque};

use anyhow::{Result, anyhow};

use crate::{
    ParallelPipelineCacheConfig, PipelineCacheEvictionKind, PipelineCachePolicy,
    PipelineCommunicationKind,
};

use super::types::{
    CrossStageCacheAccess, CrossStageCacheAccessKind, CrossStageCacheKey, CrossStageCacheStats,
    PipelineCommunicationReport, PipelinePlan, PipelineStageAssignment,
};

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
            (PipelineCommunicationKind::ActivationTensor, super::PipelineEventKind::Forward) => {
                forward_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::ActivationTensor, super::PipelineEventKind::Backward) => {
                backward_activation_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::BlockResidualCache, super::PipelineEventKind::Forward) => {
                forward_transfer_key(plan, event, layers_per_block)
            }
            (PipelineCommunicationKind::BlockResidualCache, super::PipelineEventKind::Backward) => {
                backward_cache_reuse_key(plan, event, layers_per_block)
            }
        };
        let Some(key) = key else {
            continue;
        };

        match event.kind {
            super::PipelineEventKind::Forward => report.forward_transfer_requests += 1,
            super::PipelineEventKind::Backward => report.backward_transfer_requests += 1,
        }

        let access = match (communication, event.kind) {
            (PipelineCommunicationKind::ActivationTensor, _) => {
                manager.access_bypass(key, payload_bytes)
            }
            (PipelineCommunicationKind::BlockResidualCache, super::PipelineEventKind::Forward) => {
                manager.access_forward(key, payload_bytes)
            }
            (PipelineCommunicationKind::BlockResidualCache, super::PipelineEventKind::Backward) => {
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

fn forward_transfer_key(
    plan: &PipelinePlan,
    event: &super::PipelineScheduleEvent,
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
    event: &super::PipelineScheduleEvent,
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
    event: &super::PipelineScheduleEvent,
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
