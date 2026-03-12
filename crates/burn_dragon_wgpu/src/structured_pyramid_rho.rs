use burn::prelude::*;
use std::time::Instant;

use crate::local_grid_rho::{
    CompiledLocalGridRhoPlan, LocalGridNeighborhood, LocalGridRhoPlanSpec, LocalGridShape2d,
    local_grid_rho_profile_snapshot, supports_local_grid_rho_backend,
    try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan,
};
use crate::profiling::{
    KernelProfileSite, KernelProfileSnapshot, profile_enabled, profile_record, profile_reset,
    profile_snapshot,
};

static STRUCTURED_PYRAMID_PROFILE: KernelProfileSite = KernelProfileSite::new();

pub type StructuredPyramidProfileSnapshot = KernelProfileSnapshot;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StructuredPyramidShape {
    pub patch: LocalGridShape2d,
    pub coarse: LocalGridShape2d,
    pub coarse_stride: usize,
    pub hub_count: usize,
}

#[derive(Debug, Clone)]
pub struct StructuredPyramidRhoStepInput<B: Backend> {
    pub patch_query: Tensor<B, 4>,
    pub patch_value: Tensor<B, 4>,
    pub coarse_query: Tensor<B, 4>,
    pub coarse_value: Tensor<B, 4>,
    pub patch_rho: Tensor<B, 5>,
    pub coarse_rho: Tensor<B, 5>,
    pub hub_rho: Tensor<B, 4>,
    pub patch_hub_weights: Option<Tensor<B, 4>>,
    pub coarse_hub_weights: Option<Tensor<B, 4>>,
    pub neighborhood: LocalGridNeighborhood,
    pub decay: Tensor<B, 1>,
}

#[derive(Debug, Clone)]
pub struct StructuredPyramidRhoStepOutput<B: Backend> {
    pub patch_local_context: Tensor<B, 4>,
    pub coarse_local_context: Tensor<B, 4>,
    pub patch_from_coarse_context: Tensor<B, 4>,
    pub patch_from_hub_context: Tensor<B, 4>,
    pub coarse_from_hub_context: Tensor<B, 4>,
    pub next_patch_rho: Tensor<B, 5>,
    pub next_coarse_rho: Tensor<B, 5>,
    pub next_hub_rho: Tensor<B, 4>,
}

#[derive(Debug, Clone)]
pub struct CompiledStructuredPyramidRhoPlan<B: Backend> {
    patch_plan: Option<CompiledLocalGridRhoPlan<B>>,
    coarse_plan: Option<CompiledLocalGridRhoPlan<B>>,
    patch_from_coarse_route: Option<Tensor<B, 3>>,
    shape: StructuredPyramidShape,
    neighborhood: LocalGridNeighborhood,
    rank: usize,
    value_dim: usize,
}

impl<B: Backend> CompiledStructuredPyramidRhoPlan<B> {
    pub fn new(
        batch: usize,
        rank: usize,
        value_dim: usize,
        shape: StructuredPyramidShape,
        neighborhood: LocalGridNeighborhood,
        device: &B::Device,
    ) -> Self {
        Self {
            patch_plan: (shape.patch.token_count() > 1).then(|| {
                CompiledLocalGridRhoPlan::new(
                    LocalGridRhoPlanSpec {
                        batch,
                        heads: rank,
                        value_heads: 1,
                        patch_tokens: shape.patch.token_count(),
                        latent: 1,
                        embd: value_dim,
                        grid: shape.patch,
                        neighborhood,
                    },
                    device,
                )
            }),
            coarse_plan: (shape.coarse.token_count() > 1).then(|| {
                CompiledLocalGridRhoPlan::new(
                    LocalGridRhoPlanSpec {
                        batch,
                        heads: rank,
                        value_heads: 1,
                        patch_tokens: shape.coarse.token_count(),
                        latent: 1,
                        embd: value_dim,
                        grid: shape.coarse,
                        neighborhood,
                    },
                    device,
                )
            }),
            patch_from_coarse_route: (shape.coarse_stride > 1
                && shape.coarse_stride <= 2
                && shape.patch.height == shape.coarse.height * shape.coarse_stride
                && shape.patch.width == shape.coarse.width * shape.coarse_stride)
                .then(|| patch_from_coarse_route(batch, shape.patch, shape.coarse, device)),
            shape,
            neighborhood,
            rank,
            value_dim,
        }
    }

    fn matches(&self, input: &StructuredPyramidRhoStepInput<B>) -> bool {
        let [batch, rank, patch_h, patch_w] = input.patch_query.shape().dims::<4>();
        let [patch_value_batch, value_dim, patch_value_h, patch_value_w] =
            input.patch_value.shape().dims::<4>();
        let [coarse_batch, coarse_rank, coarse_h, coarse_w] = input.coarse_query.shape().dims::<4>();
        let [coarse_value_batch, coarse_value_dim, coarse_value_h, coarse_value_w] =
            input.coarse_value.shape().dims::<4>();
        let patch_kernel_ok = if self.shape.patch.token_count() > 1 {
            self.patch_plan.is_some()
        } else {
            self.patch_plan.is_none()
        };
        let coarse_kernel_ok = if self.shape.coarse.token_count() > 1 {
            self.coarse_plan.is_some()
        } else {
            self.coarse_plan.is_none()
        };
        batch == patch_value_batch
            && batch == coarse_batch
            && batch == coarse_value_batch
            && rank == self.rank
            && coarse_rank == self.rank
            && value_dim == self.value_dim
            && coarse_value_dim == self.value_dim
            && patch_h == self.shape.patch.height
            && patch_w == self.shape.patch.width
            && patch_value_h == self.shape.patch.height
            && patch_value_w == self.shape.patch.width
            && coarse_h == self.shape.coarse.height
            && coarse_w == self.shape.coarse.width
            && coarse_value_h == self.shape.coarse.height
            && coarse_value_w == self.shape.coarse.width
            && input.patch_rho.shape().dims::<5>()
                == [
                    batch,
                    self.rank,
                    self.value_dim,
                    self.shape.patch.height,
                    self.shape.patch.width,
                ]
            && input.coarse_rho.shape().dims::<5>()
                == [
                    batch,
                    self.rank,
                    self.value_dim,
                    self.shape.coarse.height,
                    self.shape.coarse.width,
                ]
            && input.hub_rho.shape().dims::<4>()
                == [batch, self.shape.hub_count.max(1), self.rank, self.value_dim]
            && input.neighborhood == self.neighborhood
            && patch_kernel_ok
            && coarse_kernel_ok
    }
}

pub fn structured_pyramid_profile_reset() {
    profile_reset(&STRUCTURED_PYRAMID_PROFILE);
}

pub fn structured_pyramid_profile_snapshot() -> StructuredPyramidProfileSnapshot {
    profile_snapshot(&STRUCTURED_PYRAMID_PROFILE)
}

/// Reference recurrent step for the structured pyramid patch/coarse/hub bank family.
///
/// This module deliberately stops at recurrent read/write outputs. Dense-state merge remains the
/// responsibility of the adapter crate, while topology-specific recurrent routing lives here so a
/// future fused kernel can share the same input/output contract.
pub fn reference_structured_pyramid_rho_step<B: Backend>(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<B>,
) -> StructuredPyramidRhoStepOutput<B> {
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let patch_local_context = local_read(
        input.patch_rho.clone(),
        input.patch_query.clone(),
        shape.patch,
        input.neighborhood,
    );
    let coarse_local_context = local_read(
        input.coarse_rho.clone(),
        input.coarse_query.clone(),
        shape.coarse,
        input.neighborhood,
    );
    let patch_from_coarse_context = cross_scale_read(
        input.coarse_rho.clone(),
        input.patch_query.clone(),
        shape.coarse,
        shape.patch,
        shape.coarse_stride.max(1),
    );
    let patch_from_hub_context = hub_read(
        input.hub_rho.clone(),
        input.patch_query.clone(),
        input.patch_hub_weights,
    );
    let coarse_from_hub_context = hub_read(
        input.hub_rho.clone(),
        input.coarse_query.clone(),
        input.coarse_hub_weights,
    );

    let patch_update = spatial_outer_target_major(input.patch_query, input.patch_value);
    let coarse_update = spatial_outer_target_major(input.coarse_query, input.coarse_value);
    let pooled_patch_update =
        pool_target_major_outer(patch_update.clone(), shape.patch, shape.coarse_stride.max(1));

    let next_patch_rho = rho_from_target_major(
        target_major_decay_add(
            rho_to_target_major(input.patch_rho),
            patch_update.clone(),
            input.decay.clone(),
        ),
        shape.patch,
    );
    let next_coarse_rho = rho_from_target_major(
        target_major_decay_add(
            rho_to_target_major(input.coarse_rho),
            coarse_update.clone().add(pooled_patch_update),
            input.decay.clone(),
        ),
        shape.coarse,
    );
    let next_hub_rho = update_hub_from_deltas(
        input.hub_rho,
        patch_update.sum_dims_squeeze::<3, usize>(&[1]),
        coarse_update.sum_dims_squeeze::<3, usize>(&[1]),
        shape.hub_count.max(1),
        input.decay,
    );

    let output = StructuredPyramidRhoStepOutput {
        patch_local_context,
        coarse_local_context,
        patch_from_coarse_context,
        patch_from_hub_context,
        coarse_from_hub_context,
        next_patch_rho,
        next_coarse_rho,
        next_hub_rho,
    };

    if let Some(start) = total_start {
        profile_record(&STRUCTURED_PYRAMID_PROFILE, |state| {
            state.calls = state.calls.saturating_add(1);
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            state.transient_allocations = state.transient_allocations.saturating_add(6);
            state.resident_rollout_steps = state.resident_rollout_steps.saturating_add(1);
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_add(1);
        });
    }

    output
}

/// Fused forward boundary for the structured pyramid recurrent kernel family.
///
/// This is intentionally a topology-specific hook owned by `burn_dragon_wgpu`. The current
/// implementation is reference-only; a future fused path MUST preserve this contract so the vision
/// adapter keeps dense-state merge logic outside the kernel layer.
pub fn try_fused_structured_pyramid_rho_step_wgpu<B: Backend>(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<B>,
) -> Option<StructuredPyramidRhoStepOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, rank, _, _] = input.patch_query.shape().dims::<4>();
    let value_dim = input.patch_value.shape().dims::<4>()[1];
    let plan = CompiledStructuredPyramidRhoPlan::new(
        batch,
        rank,
        value_dim,
        shape,
        input.neighborhood,
        &input.patch_query.device(),
    );
    let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan);
    if output.is_some() {
        profile_record(&STRUCTURED_PYRAMID_PROFILE, |state| {
            let reused_bytes =
                (2 * 11 * core::mem::size_of::<f32>()) as u64;
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_sub(2);
            state.metadata_reuse_bytes = state.metadata_reuse_bytes.saturating_sub(reused_bytes);
            state.metadata_upload_bytes = state.metadata_upload_bytes.saturating_add(reused_bytes);
        });
    }
    output
}

pub fn supports_structured_pyramid_rho_backend<B: Backend>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    supports_local_grid_rho_backend::<B>()
}

pub fn try_fused_structured_pyramid_rho_step_wgpu_with_plan<B: Backend>(
    shape: StructuredPyramidShape,
    input: StructuredPyramidRhoStepInput<B>,
    plan: &CompiledStructuredPyramidRhoPlan<B>,
) -> Option<StructuredPyramidRhoStepOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !supports_structured_pyramid_rho_backend::<B>()
        || plan.shape != shape
        || !plan.matches(&input)
    {
        return None;
    }

    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let local_before = local_grid_rho_profile_snapshot();

    let patch_local = if shape.patch.token_count() == 1 {
        reference_local_step(
            input.patch_query.clone(),
            input.patch_value.clone(),
            input.patch_rho.clone(),
            shape.patch,
            input.neighborhood,
            input.decay.clone(),
        )
    } else {
        fused_local_step(
            input.patch_query.clone(),
            input.patch_value.clone(),
            input.patch_rho.clone(),
            shape.patch,
            input.decay.clone(),
            plan.patch_plan
                .as_ref()
                .expect("non-degenerate patch bank requires compiled local-grid plan"),
        )?
    };
    // A single-token bank has no meaningful neighborhood fanout and is not worth pushing
    // through the local-grid kernel. Use the exact reference update there to preserve parity.
    let coarse_local = if shape.coarse.token_count() == 1 {
        let reference = reference_local_step(
            input.coarse_query.clone(),
            input.coarse_value.clone(),
            input.coarse_rho.clone(),
            shape.coarse,
            input.neighborhood,
            input.decay.clone(),
        );
        (reference.0, rho_to_target_major(reference.1))
    } else {
        fused_local_step_target_major(
            input.coarse_query.clone(),
            input.coarse_value.clone(),
            input.coarse_rho.clone(),
            shape.coarse,
            input.decay.clone(),
            plan.coarse_plan
                .as_ref()
                .expect("non-degenerate coarse bank requires compiled local-grid plan"),
        )?
    };

    let patch_update = spatial_outer_target_major(input.patch_query.clone(), input.patch_value.clone());
    let coarse_update =
        spatial_outer_target_major(input.coarse_query.clone(), input.coarse_value.clone());
    let pooled_patch_update =
        pool_target_major_outer(patch_update.clone(), shape.patch, shape.coarse_stride.max(1));

    let next_coarse_rho =
        rho_from_target_major(coarse_local.1.clone().add(pooled_patch_update), shape.coarse);
    let next_hub_rho = update_hub_from_deltas(
        input.hub_rho.clone(),
        patch_update.sum_dims_squeeze::<3, usize>(&[1]),
        coarse_update.sum_dims_squeeze::<3, usize>(&[1]),
        shape.hub_count.max(1),
        input.decay.clone(),
    );

        let output = StructuredPyramidRhoStepOutput {
            patch_local_context: patch_local.0,
            coarse_local_context: coarse_local.0,
            patch_from_coarse_context: cross_scale_read_fast(
                input.coarse_rho.clone(),
                input.patch_query.clone(),
                shape.coarse,
                shape.patch,
                shape.coarse_stride.max(1),
                plan.patch_from_coarse_route.clone(),
            )
            .unwrap_or_else(|| {
                cross_scale_read(
                    input.coarse_rho.clone(),
                    input.patch_query.clone(),
                    shape.coarse,
                    shape.patch,
                    shape.coarse_stride.max(1),
                )
            }),
            patch_from_hub_context: hub_read(
                input.hub_rho.clone(),
                input.patch_query.clone(),
                input.patch_hub_weights.clone(),
        ),
        coarse_from_hub_context: hub_read(
            input.hub_rho,
            input.coarse_query.clone(),
            input.coarse_hub_weights,
        ),
        next_patch_rho: patch_local.1,
        next_coarse_rho,
        next_hub_rho,
    };

    if let Some(start) = total_start {
        let local_after = local_grid_rho_profile_snapshot();
        profile_record(&STRUCTURED_PYRAMID_PROFILE, |state| {
            state.calls = state.calls.saturating_add(1);
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
            state.launches = state
                .launches
                .saturating_add(local_after.launches.saturating_sub(local_before.launches));
            state.dispatch_ns = state.dispatch_ns.saturating_add(
                local_after
                    .dispatch_ns
                    .saturating_sub(local_before.dispatch_ns),
            );
            state.transient_allocations = state.transient_allocations.saturating_add(
                local_after
                    .transient_allocations
                    .saturating_sub(local_before.transient_allocations)
                    .saturating_add(4),
            );
            state.metadata_upload_bytes = state.metadata_upload_bytes.saturating_add(
                local_after
                    .metadata_upload_bytes
                    .saturating_sub(local_before.metadata_upload_bytes),
            );
            state.metadata_reuse_hits = state.metadata_reuse_hits.saturating_add(
                local_after
                    .metadata_reuse_hits
                    .saturating_sub(local_before.metadata_reuse_hits),
            );
            state.metadata_reuse_bytes = state.metadata_reuse_bytes.saturating_add(
                local_after
                    .metadata_reuse_bytes
                    .saturating_sub(local_before.metadata_reuse_bytes),
            );
            state.resident_rollout_steps = state.resident_rollout_steps.saturating_add(1);
        });
    }

    Some(output)
}

fn tokens_to_target_major<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, channels, height, width] = input.shape().dims::<4>();
    input
        .swap_dims(1, 3)
        .swap_dims(1, 2)
        .reshape([batch, height * width, channels])
}

fn tokens_from_target_major<B: Backend>(
    input: Tensor<B, 3>,
    shape: LocalGridShape2d,
) -> Tensor<B, 4> {
    let [batch, tokens, channels] = input.shape().dims::<3>();
    assert_eq!(tokens, shape.token_count());
    input
        .reshape([batch, shape.height, shape.width, channels])
        .swap_dims(1, 3)
        .swap_dims(2, 3)
}

fn rho_to_target_major<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 4> {
    let [batch, rank, value_dim, height, width] = input.shape().dims::<5>();
    input
        .swap_dims(1, 3)
        .swap_dims(2, 4)
        .reshape([batch, height * width, rank, value_dim])
}

fn rho_from_target_major<B: Backend>(input: Tensor<B, 4>, shape: LocalGridShape2d) -> Tensor<B, 5> {
    let [batch, tokens, rank, value_dim] = input.shape().dims::<4>();
    assert_eq!(tokens, shape.token_count());
    input
        .reshape([batch, shape.height, shape.width, rank, value_dim])
        .swap_dims(2, 4)
        .swap_dims(1, 3)
}

fn contract<B: Backend>(rho: Tensor<B, 5>, query: Tensor<B, 4>, shape: LocalGridShape2d) -> Tensor<B, 4> {
    let read = target_major_identity_read(tokens_to_target_major(query), rho_to_target_major(rho));
    tokens_from_target_major(read, shape)
}

fn spatial_tokens_to_local_grid<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.shape().dims::<4>();
    tokens_to_target_major(input)
        .swap_dims(1, 2)
        .reshape([batch, channels, height * width, 1])
}

fn spatial_values_to_local_grid<B: Backend>(input: Tensor<B, 4>) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.shape().dims::<4>();
    tokens_to_target_major(input).reshape([batch, 1, height * width, channels])
}

fn spatial_rho_to_local_grid<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 5> {
    let [batch, rank, value_dim, height, width] = input.shape().dims::<5>();
    rho_to_target_major(input)
        .swap_dims(1, 2)
        .reshape([batch, rank, height * width, 1, value_dim])
}

fn local_grid_context_to_spatial<B: Backend>(
    input: Tensor<B, 4>,
    shape: LocalGridShape2d,
) -> Tensor<B, 4> {
    let [batch, _, tokens, value_dim] = input.shape().dims::<4>();
    let target_major = input.sum_dim(1).reshape([batch, tokens, value_dim]);
    tokens_from_target_major(target_major, shape)
}

fn local_grid_rho_to_spatial<B: Backend>(input: Tensor<B, 5>, shape: LocalGridShape2d) -> Tensor<B, 5> {
    rho_from_target_major(local_grid_rho_to_target_major(input), shape)
}

fn local_grid_rho_to_target_major<B: Backend>(input: Tensor<B, 5>) -> Tensor<B, 4> {
    let [batch, rank, tokens, _, value_dim] = input.shape().dims::<5>();
    input.reshape([batch, rank, tokens, value_dim]).swap_dims(1, 2)
}

fn fused_local_step<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho: Tensor<B, 5>,
    shape: LocalGridShape2d,
    decay: Tensor<B, 1>,
    plan: &CompiledLocalGridRhoPlan<B>,
) -> Option<(Tensor<B, 4>, Tensor<B, 5>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let output = try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan(
        &spatial_tokens_to_local_grid(query),
        &spatial_values_to_local_grid(value),
        Some(&spatial_rho_to_local_grid(rho)),
        &decay,
        plan,
    )?;
    Some((
        local_grid_context_to_spatial(output.context, shape),
        local_grid_rho_to_spatial(output.rho, shape),
    ))
}

fn fused_local_step_target_major<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho: Tensor<B, 5>,
    shape: LocalGridShape2d,
    decay: Tensor<B, 1>,
    plan: &CompiledLocalGridRhoPlan<B>,
) -> Option<(Tensor<B, 4>, Tensor<B, 4>)>
where
    B::FloatTensorPrimitive: 'static,
{
    let output = try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan(
        &spatial_tokens_to_local_grid(query),
        &spatial_values_to_local_grid(value),
        Some(&spatial_rho_to_local_grid(rho)),
        &decay,
        plan,
    )?;
    Some((
        local_grid_context_to_spatial(output.context, shape),
        local_grid_rho_to_target_major(output.rho),
    ))
}

fn reference_local_step<B: Backend>(
    query: Tensor<B, 4>,
    value: Tensor<B, 4>,
    rho: Tensor<B, 5>,
    shape: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
    decay: Tensor<B, 1>,
) -> (Tensor<B, 4>, Tensor<B, 5>) {
    let context = local_read(rho.clone(), query.clone(), shape, neighborhood);
    let next_rho = rho_from_target_major(
        target_major_decay_add(
            rho_to_target_major(rho),
            spatial_outer_target_major(query, value),
            decay,
        ),
        shape,
    );
    (context, next_rho)
}

fn spatial_outer_target_major<B: Backend>(query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
    target_major_outer_product(tokens_to_target_major(query), tokens_to_target_major(value))
}

fn shift_spatial<B: Backend>(input: Tensor<B, 4>, dy: isize, dx: isize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = input.shape().dims::<4>();
    let device = input.device();
    let mut output = input;

    if dy != 0 {
        let shift = dy.unsigned_abs();
        if shift >= height {
            output = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
        } else if dy > 0 {
            let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
            output = Tensor::cat(vec![pad, output.slice_dim(2, 0..height - shift)], 2);
        } else {
            let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
            output = Tensor::cat(vec![output.slice_dim(2, shift..height), pad], 2);
        }
    }

    if dx != 0 {
        let shift = dx.unsigned_abs();
        if shift >= width {
            output = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
        } else if dx > 0 {
            let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
            output = Tensor::cat(vec![pad, output.slice_dim(3, 0..width - shift)], 3);
        } else {
            let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
            output = Tensor::cat(vec![output.slice_dim(3, shift..width), pad], 3);
        }
    }

    output
}

fn local_read<B: Backend>(
    rho: Tensor<B, 5>,
    query: Tensor<B, 4>,
    shape: LocalGridShape2d,
    neighborhood: LocalGridNeighborhood,
) -> Tensor<B, 4> {
    let [batch, _, value_dim, _, _] = rho.shape().dims::<5>();
    let mut acc = Tensor::<B, 4>::zeros(
        [batch, value_dim, shape.height.max(1), shape.width.max(1)],
        &rho.device(),
    );
    let radius = neighborhood.radius as isize;
    if neighborhood.self_edges {
        acc = acc.add(contract(rho.clone(), query.clone(), shape));
    }
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            if dy == 0 && dx == 0 {
                continue;
            }
            if !neighborhood.diagonals && dy != 0 && dx != 0 {
                continue;
            }
            let shifted_query = shift_spatial(query.clone(), dy, dx);
            let msg = contract(rho.clone(), shifted_query, shape);
            acc = acc.add(shift_spatial(msg, -dy, -dx));
        }
    }
    acc
}

fn cross_scale_read<B: Backend>(
    coarse_rho: Tensor<B, 5>,
    patch_query: Tensor<B, 4>,
    coarse_shape: LocalGridShape2d,
    patch_shape: LocalGridShape2d,
    coarse_stride: usize,
) -> Tensor<B, 4> {
    if coarse_stride <= 1 {
        return contract(coarse_rho, patch_query, patch_shape);
    }
    let mut up = coarse_rho
        .repeat_dim(3, coarse_stride)
        .repeat_dim(4, coarse_stride);
    let [_, _, _, up_h, up_w] = up.shape().dims::<5>();
    if up_h != patch_shape.height {
        up = up.slice_dim(3, 0..patch_shape.height.min(up_h));
    }
    if up_w != patch_shape.width {
        up = up.slice_dim(4, 0..patch_shape.width.min(up_w));
    }
    let effective = LocalGridShape2d::new(
        patch_shape.height.min(coarse_shape.height * coarse_stride),
        patch_shape.width.min(coarse_shape.width * coarse_stride),
    );
    contract(up, patch_query, effective)
}

fn cross_scale_read_fast<B: Backend>(
    coarse_rho: Tensor<B, 5>,
    patch_query: Tensor<B, 4>,
    coarse_shape: LocalGridShape2d,
    patch_shape: LocalGridShape2d,
    coarse_stride: usize,
    route: Option<Tensor<B, 3>>,
) -> Option<Tensor<B, 4>> {
    if let Some(context) = cross_scale_read_with_route(
        coarse_rho.clone(),
        patch_query.clone(),
        coarse_shape,
        patch_shape,
        coarse_stride,
        route,
    ) {
        return Some(context);
    }
    cross_scale_read_tiled(coarse_rho, patch_query, coarse_shape, patch_shape, coarse_stride)
}

fn cross_scale_read_with_route<B: Backend>(
    coarse_rho: Tensor<B, 5>,
    patch_query: Tensor<B, 4>,
    coarse_shape: LocalGridShape2d,
    patch_shape: LocalGridShape2d,
    coarse_stride: usize,
    route: Option<Tensor<B, 3>>,
) -> Option<Tensor<B, 4>> {
    if coarse_stride <= 1 {
        return Some(contract(coarse_rho, patch_query, patch_shape));
    }
    let route = route?;
    let [batch, patch_tokens, coarse_tokens] = route.shape().dims::<3>();
    let [rho_batch, rank, value_dim, coarse_h, coarse_w] = coarse_rho.shape().dims::<5>();
    let [query_batch, query_rank, patch_h, patch_w] = patch_query.shape().dims::<4>();
    if rho_batch != batch
        || query_batch != batch
        || query_rank != rank
        || coarse_h != coarse_shape.height
        || coarse_w != coarse_shape.width
        || patch_h != patch_shape.height
        || patch_w != patch_shape.width
        || coarse_tokens != coarse_shape.token_count()
        || patch_tokens != patch_shape.token_count()
    {
        return None;
    }

    let routed_rho = rho_to_target_major(coarse_rho).reshape([batch, coarse_tokens, rank * value_dim]);
    let routed_rho = route.matmul(routed_rho).reshape([batch, patch_tokens, rank, value_dim]);
    let patch_query = tokens_to_target_major(patch_query);
    let context = routed_rho
        .mul(patch_query.unsqueeze_dim::<4>(3))
        .sum_dims_squeeze::<3, usize>(&[2]);
    Some(tokens_from_target_major(context, patch_shape))
}

fn patch_from_coarse_route<B: Backend>(
    batch: usize,
    patch_shape: LocalGridShape2d,
    coarse_shape: LocalGridShape2d,
    device: &B::Device,
) -> Tensor<B, 3> {
    let patch_tokens = patch_shape.token_count();
    let coarse_tokens = coarse_shape.token_count();
    let mut data = vec![0.0_f32; batch * patch_tokens * coarse_tokens];
    for b in 0..batch {
        let batch_offset = b * patch_tokens * coarse_tokens;
        for py in 0..patch_shape.height {
            for px in 0..patch_shape.width {
                let patch_idx = py * patch_shape.width + px;
                let coarse_idx =
                    (py % coarse_shape.height) * coarse_shape.width + (px % coarse_shape.width);
                data[batch_offset + patch_idx * coarse_tokens + coarse_idx] = 1.0;
            }
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(data, [batch, patch_tokens, coarse_tokens]),
        device,
    )
}

fn cross_scale_read_tiled<B: Backend>(
    coarse_rho: Tensor<B, 5>,
    patch_query: Tensor<B, 4>,
    coarse_shape: LocalGridShape2d,
    patch_shape: LocalGridShape2d,
    coarse_stride: usize,
) -> Option<Tensor<B, 4>> {
    if coarse_stride <= 1 {
        return Some(contract(coarse_rho, patch_query, patch_shape));
    }
    let [batch, rank, value_dim, coarse_h, coarse_w] = coarse_rho.shape().dims::<5>();
    let [query_batch, query_rank, patch_h, patch_w] = patch_query.shape().dims::<4>();
    if query_batch != batch
        || query_rank != rank
        || coarse_h != coarse_shape.height
        || coarse_w != coarse_shape.width
        || patch_h != patch_shape.height
        || patch_w != patch_shape.width
        || patch_h != coarse_h * coarse_stride
        || patch_w != coarse_w * coarse_stride
    {
        return None;
    }

    let coarse_rho = coarse_rho
        .unsqueeze_dim::<6>(3)
        .unsqueeze_dim::<7>(5);
    let patch_query = patch_query
        .reshape([
            batch,
            rank,
            coarse_stride,
            coarse_h,
            coarse_stride,
            coarse_w,
        ])
        .unsqueeze_dim::<7>(2);
    let context = coarse_rho
        .mul(patch_query)
        .sum_dims_squeeze::<6, usize>(&[1]);
    Some(context.reshape([batch, value_dim, patch_h, patch_w]))
}

fn hub_read<B: Backend>(
    hub_rho: Tensor<B, 4>,
    query: Tensor<B, 4>,
    weights: Option<Tensor<B, 4>>,
) -> Tensor<B, 4> {
    let [_, hubs, _, _] = hub_rho.shape().dims::<4>();
    let [_, _, height, width] = query.shape().dims::<4>();
    let query = tokens_to_target_major(query).unsqueeze_dim::<4>(1);
    let hub_context = query.matmul(hub_rho);
    let reduced = if let Some(weights) = weights {
        let weights = tokens_to_target_major(weights).swap_dims(1, 2).unsqueeze_dim::<4>(3);
        hub_context.mul(weights).sum_dims_squeeze::<3, usize>(&[1])
    } else if hubs > 1 {
        hub_context
            .sum_dims_squeeze::<3, usize>(&[1])
            .div_scalar(hubs as f32)
    } else {
        hub_context.sum_dims_squeeze::<3, usize>(&[1])
    };
    tokens_from_target_major(reduced, LocalGridShape2d::new(height, width))
}

fn pool_target_major_outer<B: Backend>(
    update: Tensor<B, 4>,
    patch_shape: LocalGridShape2d,
    stride: usize,
) -> Tensor<B, 4> {
    if stride <= 1 {
        return update;
    }
    let [batch, tokens, rank, value_dim] = update.shape().dims::<4>();
    assert_eq!(tokens, patch_shape.token_count());
    let pooled_height = patch_shape.height / stride;
    let pooled_width = patch_shape.width / stride;
    update
        .reshape([batch, patch_shape.height, patch_shape.width, rank * value_dim])
        .reshape([
            batch,
            pooled_height,
            stride,
            pooled_width,
            stride,
            rank * value_dim,
        ])
        .sum_dims_squeeze::<4, usize>(&[2, 4])
        .reshape([batch, pooled_height * pooled_width, rank, value_dim])
}

fn update_hub_from_deltas<B: Backend>(
    hub_rho: Tensor<B, 4>,
    patch_delta: Tensor<B, 3>,
    coarse_delta: Tensor<B, 3>,
    hub_count: usize,
    decay: Tensor<B, 1>,
) -> Tensor<B, 4> {
    let delta = if hub_count > 1 {
        patch_delta
            .add(coarse_delta)
            .unsqueeze_dim::<4>(1)
            .div_scalar(hub_count as f32)
    } else {
        patch_delta.add(coarse_delta).unsqueeze_dim::<4>(1)
    };
    target_major_decay_add(hub_rho, delta, decay)
}

fn target_major_identity_read<B: Backend>(query: Tensor<B, 3>, rho: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, targets, rank] = query.shape().dims::<3>();
    let [rho_batch, rho_targets, rho_rank, value_dim] = rho.shape().dims::<4>();
    assert_eq!(rho_batch, batch);
    assert_eq!(rho_targets, targets);
    assert_eq!(rho_rank, rank);
    rho.mul(query.unsqueeze_dim::<4>(3))
        .sum_dims_squeeze::<3, usize>(&[2])
        .reshape([batch, targets, value_dim])
}

fn target_major_outer_product<B: Backend>(query: Tensor<B, 3>, value: Tensor<B, 3>) -> Tensor<B, 4> {
    let [batch, targets, rank] = query.shape().dims::<3>();
    let [value_batch, value_targets, value_dim] = value.shape().dims::<3>();
    assert_eq!(value_batch, batch);
    assert_eq!(value_targets, targets);
    query
        .unsqueeze_dim::<4>(3)
        .mul(value.unsqueeze_dim::<4>(2))
        .reshape([batch, targets, rank, value_dim])
}

fn target_major_decay_add<B: Backend>(
    rho: Tensor<B, 4>,
    update: Tensor<B, 4>,
    decay: Tensor<B, 1>,
) -> Tensor<B, 4> {
    let [_, _, rank, _] = rho.shape().dims::<4>();
    let [decay_len] = decay.shape().dims::<1>();
    let decay = match decay_len {
        1 => decay.repeat_dim(0, rank.max(1)),
        len if len == rank => decay,
        _ => panic!("structured pyramid decay length {decay_len} must be 1 or {rank}"),
    };
    rho.mul(decay.reshape([1, 1, rank, 1])).add(update)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::TensorData;
    use burn_cubecl::cubecl::Runtime;
    use burn_ndarray::NdArray;
    use burn_wgpu::{CubeBackend, RuntimeOptions, WgpuRuntime, graphics};

    #[test]
    fn structured_pyramid_reference_step_preserves_shapes() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(2, 2),
            coarse: LocalGridShape2d::new(1, 1),
            coarse_stride: 2,
            hub_count: 2,
        };
        let input = StructuredPyramidRhoStepInput {
            patch_query: Tensor::<Backend, 4>::ones([1, 2, 2, 2], &device),
            patch_value: Tensor::<Backend, 4>::ones([1, 4, 2, 2], &device),
            coarse_query: Tensor::<Backend, 4>::ones([1, 2, 1, 1], &device),
            coarse_value: Tensor::<Backend, 4>::ones([1, 4, 1, 1], &device),
            patch_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 2, 2], &device),
            coarse_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 1, 1], &device),
            hub_rho: Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device),
            patch_hub_weights: None,
            coarse_hub_weights: None,
            neighborhood: LocalGridNeighborhood::moore(1),
            decay: Tensor::<Backend, 1>::ones([2], &device),
        };

        let output = reference_structured_pyramid_rho_step(shape, input);
        assert_eq!(output.patch_local_context.shape().dims(), [1, 4, 2, 2]);
        assert_eq!(output.coarse_local_context.shape().dims(), [1, 4, 1, 1]);
        assert_eq!(output.patch_from_coarse_context.shape().dims(), [1, 4, 2, 2]);
        assert_eq!(output.patch_from_hub_context.shape().dims(), [1, 4, 2, 2]);
        assert_eq!(output.coarse_from_hub_context.shape().dims(), [1, 4, 1, 1]);
        assert_eq!(output.next_patch_rho.shape().dims(), [1, 2, 4, 2, 2]);
        assert_eq!(output.next_coarse_rho.shape().dims(), [1, 2, 4, 1, 1]);
        assert_eq!(output.next_hub_rho.shape().dims(), [1, 2, 2, 4]);
    }

    #[test]
    fn structured_pyramid_reference_step_writes_patch_and_coarse_activity_into_hub_bank() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(2, 2),
            coarse: LocalGridShape2d::new(1, 1),
            coarse_stride: 2,
            hub_count: 2,
        };
        let input = StructuredPyramidRhoStepInput {
            patch_query: Tensor::<Backend, 4>::ones([1, 2, 2, 2], &device),
            patch_value: Tensor::<Backend, 4>::ones([1, 4, 2, 2], &device),
            coarse_query: Tensor::<Backend, 4>::ones([1, 2, 1, 1], &device),
            coarse_value: Tensor::<Backend, 4>::ones([1, 4, 1, 1], &device),
            patch_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 2, 2], &device),
            coarse_rho: Tensor::<Backend, 5>::zeros([1, 2, 4, 1, 1], &device),
            hub_rho: Tensor::<Backend, 4>::zeros([1, 2, 2, 4], &device),
            patch_hub_weights: None,
            coarse_hub_weights: None,
            neighborhood: LocalGridNeighborhood::moore(1),
            decay: Tensor::<Backend, 1>::ones([2], &device),
        };

        let output = reference_structured_pyramid_rho_step(shape, input);
        let max_abs = output
            .next_hub_rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hub rho")
            .into_iter()
            .map(f32::abs)
            .fold(0.0_f32, f32::max);
        assert!(max_abs > 0.0);
    }

    #[test]
    fn cross_scale_read_tiles_the_coarse_grid_across_patch_space() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let coarse_shape = LocalGridShape2d::new(2, 2);
        let patch_shape = LocalGridShape2d::new(4, 4);
        let coarse_rho = Tensor::<Backend, 5>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, //
                    3.0, 4.0,
                ],
                [1, 1, 1, 2, 2],
            ),
            &device,
        );
        let patch_query = Tensor::<Backend, 4>::ones([1, 1, 4, 4], &device);

        let context = cross_scale_read(coarse_rho, patch_query, coarse_shape, patch_shape, 2);
        assert_eq!(context.shape().dims(), [1, 1, 4, 4]);
        assert_eq!(
            context
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("cross scale context"),
            vec![
                1.0, 2.0, 1.0, 2.0, //
                3.0, 4.0, 3.0, 4.0, //
                1.0, 2.0, 1.0, 2.0, //
                3.0, 4.0, 3.0, 4.0,
            ]
        );
    }

    #[test]
    fn pool_target_major_outer_sums_patch_blocks_without_spatial_roundtrip() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let update = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, //
                    3.0, 4.0,
                ],
                [1, 4, 1, 1],
            ),
            &device,
        );

        let pooled = pool_target_major_outer(update, LocalGridShape2d::new(2, 2), 2);
        assert_eq!(pooled.shape().dims(), [1, 1, 1, 1]);
        let values = pooled
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("pooled update");
        assert!((values[0] - 10.0).abs() <= 1.0e-6);
    }

    #[test]
    fn hub_read_matches_manual_weighted_projection() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hub_rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, //
                    3.0, 4.0,
                ],
                [1, 2, 2, 1],
            ),
            &device,
        );
        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 0.0, //
                    0.0, 1.0,
                ],
                [1, 2, 2, 1],
            ),
            &device,
        );
        let weights = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.75, 0.20, //
                    0.25, 0.80,
                ],
                [1, 2, 2, 1],
            ),
            &device,
        );

        let output = hub_read(hub_rho, query, Some(weights));
        assert_eq!(output.shape().dims(), [1, 1, 2, 1]);
        let values = output
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hub read output");
        assert!((values[0] - 1.5).abs() <= 1.0e-6);
        assert!((values[1] - 3.6).abs() <= 1.0e-6);
    }

    #[test]
    fn update_hub_from_deltas_distributes_multi_hub_delta_uniformly() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let hub_rho = Tensor::<Backend, 4>::zeros([1, 2, 2, 1], &device);
        let patch_delta = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![2.0, 4.0], [1, 2, 1]),
            &device,
        );
        let coarse_delta = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![1.0, 3.0], [1, 2, 1]),
            &device,
        );
        let decay = Tensor::<Backend, 1>::ones([2], &device);

        let output = update_hub_from_deltas(hub_rho, patch_delta, coarse_delta, 2, decay);
        assert_eq!(output.shape().dims(), [1, 2, 2, 1]);
        assert_eq!(
            output
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("hub rho output"),
            vec![
                1.5, 3.5, //
                1.5, 3.5,
            ]
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    type WgpuBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;

    #[cfg(not(target_arch = "wasm32"))]
    fn init_wgpu_runtime(device: &<WgpuBackend as BackendTrait>::Device) {
        static INIT: std::sync::Once = std::sync::Once::new();
        INIT.call_once(|| {
            burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[derive(Clone, Copy)]
    struct MemorySnapshot {
        reserved: u64,
        in_use: u64,
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn memory_snapshot(device: &<WgpuBackend as BackendTrait>::Device) -> MemorySnapshot {
        let usage = <WgpuRuntime as Runtime>::client(device).memory_usage();
        MemorySnapshot {
            reserved: usage.bytes_reserved,
            in_use: usage.bytes_in_use,
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn assert_memory_growth_bounded(
        label: &str,
        snapshots: &[MemorySnapshot],
        max_reserved_growth: u64,
        max_in_use_growth: u64,
    ) {
        assert!(!snapshots.is_empty(), "{label}: no memory snapshots");
        let first = snapshots[0];
        let last = snapshots[snapshots.len() - 1];
        let reserved_growth = last.reserved.saturating_sub(first.reserved);
        let in_use_growth = last.in_use.saturating_sub(first.in_use);
        assert!(
            reserved_growth <= max_reserved_growth,
            "{label}: reserved growth {} exceeded {}",
            reserved_growth,
            max_reserved_growth
        );
        assert!(
            in_use_growth <= max_in_use_growth,
            "{label}: in_use growth {} exceeded {}",
            in_use_growth,
            max_in_use_growth
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn max_abs_diff<const D: usize>(
        lhs: Tensor<WgpuBackend, D>,
        rhs: Tensor<WgpuBackend, D>,
    ) -> f32 {
        lhs.sub(rhs)
            .abs()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("diff vec")
            .into_iter()
            .fold(0.0_f32, f32::max)
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_structured_pyramid_matches_reference() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_wgpu_runtime(&device);

        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(4, 4),
            coarse: LocalGridShape2d::new(2, 2),
            coarse_stride: 2,
            hub_count: 2,
        };
        let input = StructuredPyramidRhoStepInput {
            patch_query: Tensor::<WgpuBackend, 4>::random(
                [2, 4, 4, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_value: Tensor::<WgpuBackend, 4>::random(
                [2, 6, 4, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_query: Tensor::<WgpuBackend, 4>::random(
                [2, 4, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_value: Tensor::<WgpuBackend, 4>::random(
                [2, 6, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_rho: Tensor::<WgpuBackend, 5>::random(
                [2, 4, 6, 4, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_rho: Tensor::<WgpuBackend, 5>::random(
                [2, 4, 6, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            hub_rho: Tensor::<WgpuBackend, 4>::random(
                [2, 2, 4, 6],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_hub_weights: None,
            coarse_hub_weights: None,
            neighborhood: LocalGridNeighborhood::moore(1),
            decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.9, 0.95, 0.975], &device),
        };

        let reference = reference_structured_pyramid_rho_step(shape, input.clone());
        let plan = CompiledStructuredPyramidRhoPlan::new(2, 4, 6, shape, input.neighborhood, &device);
        let fused =
            try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan).expect(
                "structured pyramid fused output",
            );

        assert!(max_abs_diff(fused.patch_local_context, reference.patch_local_context) <= 1.0e-5);
        assert!(
            max_abs_diff(fused.coarse_local_context, reference.coarse_local_context) <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.patch_from_coarse_context, reference.patch_from_coarse_context)
                <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.patch_from_hub_context, reference.patch_from_hub_context) <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.coarse_from_hub_context, reference.coarse_from_hub_context)
                <= 1.0e-5
        );
        assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_structured_pyramid_matches_reference_on_compact_hub_gated_shape() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_wgpu_runtime(&device);
        <WgpuBackend as BackendTrait>::seed(&device, 6_060);

        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(2, 2),
            coarse: LocalGridShape2d::new(1, 1),
            coarse_stride: 2,
            hub_count: 2,
        };
        let patch_hub_weights = Tensor::<WgpuBackend, 4>::random(
            [2, 2, 2, 2],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        )
        .abs();
        let patch_hub_weights = patch_hub_weights.clone()
            / patch_hub_weights
                .clone()
                .sum_dim(1)
                .add_scalar(1.0e-6);
        let coarse_hub_weights = Tensor::<WgpuBackend, 4>::random(
            [2, 2, 1, 1],
            burn::tensor::Distribution::Normal(0.0, 1.0),
            &device,
        )
        .abs();
        let coarse_hub_weights = coarse_hub_weights.clone()
            / coarse_hub_weights
                .clone()
                .sum_dim(1)
                .add_scalar(1.0e-6);
        let input = StructuredPyramidRhoStepInput {
            patch_query: Tensor::<WgpuBackend, 4>::random(
                [2, 2, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_value: Tensor::<WgpuBackend, 4>::random(
                [2, 4, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_query: Tensor::<WgpuBackend, 4>::random(
                [2, 2, 1, 1],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_value: Tensor::<WgpuBackend, 4>::random(
                [2, 4, 1, 1],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_rho: Tensor::<WgpuBackend, 5>::random(
                [2, 2, 4, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_rho: Tensor::<WgpuBackend, 5>::random(
                [2, 2, 4, 1, 1],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            hub_rho: Tensor::<WgpuBackend, 4>::random(
                [2, 2, 2, 4],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_hub_weights: Some(patch_hub_weights),
            coarse_hub_weights: Some(coarse_hub_weights),
            neighborhood: LocalGridNeighborhood::moore(1),
            decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.95], &device),
        };

        let reference = reference_structured_pyramid_rho_step(shape, input.clone());
        let plan = CompiledStructuredPyramidRhoPlan::new(2, 2, 4, shape, input.neighborhood, &device);
        let fused =
            try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan).expect(
                "structured pyramid fused output",
            );

        assert!(max_abs_diff(fused.patch_local_context, reference.patch_local_context) <= 1.0e-5);
        assert!(
            max_abs_diff(fused.coarse_local_context, reference.coarse_local_context) <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.patch_from_coarse_context, reference.patch_from_coarse_context)
                <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.patch_from_hub_context, reference.patch_from_hub_context) <= 1.0e-5
        );
        assert!(
            max_abs_diff(fused.coarse_from_hub_context, reference.coarse_from_hub_context)
                <= 1.0e-5
        );
        assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_structured_pyramid_matches_reference_on_stride3_tiled_cross_scale_shape() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_wgpu_runtime(&device);
        <WgpuBackend as BackendTrait>::seed(&device, 7_070);

        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(6, 6),
            coarse: LocalGridShape2d::new(2, 2),
            coarse_stride: 3,
            hub_count: 2,
        };
        let input = StructuredPyramidRhoStepInput {
            patch_query: Tensor::<WgpuBackend, 4>::random(
                [1, 3, 6, 6],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_value: Tensor::<WgpuBackend, 4>::random(
                [1, 5, 6, 6],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_query: Tensor::<WgpuBackend, 4>::random(
                [1, 3, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_value: Tensor::<WgpuBackend, 4>::random(
                [1, 5, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_rho: Tensor::<WgpuBackend, 5>::random(
                [1, 3, 5, 6, 6],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            coarse_rho: Tensor::<WgpuBackend, 5>::random(
                [1, 3, 5, 2, 2],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            hub_rho: Tensor::<WgpuBackend, 4>::random(
                [1, 2, 3, 5],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                &device,
            ),
            patch_hub_weights: None,
            coarse_hub_weights: None,
            neighborhood: LocalGridNeighborhood::moore(1),
            decay: Tensor::<WgpuBackend, 1>::from_floats([0.85, 0.925, 0.975], &device),
        };

        let reference = reference_structured_pyramid_rho_step(shape, input.clone());
        let plan = CompiledStructuredPyramidRhoPlan::new(1, 3, 5, shape, input.neighborhood, &device);
        let fused =
            try_fused_structured_pyramid_rho_step_wgpu_with_plan(shape, input, &plan).expect(
                "structured pyramid fused output",
            );

        assert!(
            max_abs_diff(fused.patch_from_coarse_context, reference.patch_from_coarse_context)
                <= 1.0e-5
        );
        assert!(max_abs_diff(fused.next_patch_rho, reference.next_patch_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_coarse_rho, reference.next_coarse_rho) <= 1.0e-5);
        assert!(max_abs_diff(fused.next_hub_rho, reference.next_hub_rho) <= 1.0e-5);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn structured_pyramid_reference_memory_stays_bounded_across_repeated_calls() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_wgpu_runtime(&device);

        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(4, 4),
            coarse: LocalGridShape2d::new(2, 2),
            coarse_stride: 2,
            hub_count: 2,
        };
        let patch_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 4, 4], &device);
        let patch_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 4, 4], &device);
        let coarse_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 2, 2], &device);
        let coarse_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 2, 2], &device);
        let decay = Tensor::<WgpuBackend, 1>::ones([4], &device);

        let mut patch_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 4, 4], &device);
        let mut coarse_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 2, 2], &device);
        let mut hub_rho = Tensor::<WgpuBackend, 4>::zeros([1, 2, 4, 6], &device);

        let mut snapshots = Vec::with_capacity(24);
        for step in 0..32 {
            let output = reference_structured_pyramid_rho_step(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: patch_query.clone(),
                    patch_value: patch_value.clone(),
                    coarse_query: coarse_query.clone(),
                    coarse_value: coarse_value.clone(),
                    patch_rho: patch_rho.clone(),
                    coarse_rho: coarse_rho.clone(),
                    hub_rho: hub_rho.clone(),
                    patch_hub_weights: None,
                    coarse_hub_weights: None,
                    neighborhood: LocalGridNeighborhood::moore(1),
                    decay: decay.clone(),
                },
            );
            patch_rho = output.next_patch_rho;
            coarse_rho = output.next_coarse_rho;
            hub_rho = output.next_hub_rho;
            let _ = WgpuBackend::sync(&device);
            WgpuBackend::memory_cleanup(&device);
            let _ = WgpuBackend::sync(&device);
            if step >= 8 {
                snapshots.push(memory_snapshot(&device));
            }
        }

        assert_memory_growth_bounded(
            "structured_pyramid_reference",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn fused_structured_pyramid_memory_stays_bounded_across_repeated_calls() {
        let device = <WgpuBackend as BackendTrait>::Device::default();
        init_wgpu_runtime(&device);

        let shape = StructuredPyramidShape {
            patch: LocalGridShape2d::new(4, 4),
            coarse: LocalGridShape2d::new(2, 2),
            coarse_stride: 2,
            hub_count: 2,
        };
        let patch_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 4, 4], &device);
        let patch_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 4, 4], &device);
        let coarse_query = Tensor::<WgpuBackend, 4>::ones([1, 4, 2, 2], &device);
        let coarse_value = Tensor::<WgpuBackend, 4>::ones([1, 6, 2, 2], &device);
        let decay = Tensor::<WgpuBackend, 1>::ones([4], &device);
        let plan = CompiledStructuredPyramidRhoPlan::new(
            1,
            4,
            6,
            shape,
            LocalGridNeighborhood::moore(1),
            &device,
        );

        let mut patch_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 4, 4], &device);
        let mut coarse_rho = Tensor::<WgpuBackend, 5>::zeros([1, 4, 6, 2, 2], &device);
        let mut hub_rho = Tensor::<WgpuBackend, 4>::zeros([1, 2, 4, 6], &device);

        let mut snapshots = Vec::with_capacity(24);
        for step in 0..32 {
            let output = try_fused_structured_pyramid_rho_step_wgpu_with_plan(
                shape,
                StructuredPyramidRhoStepInput {
                    patch_query: patch_query.clone(),
                    patch_value: patch_value.clone(),
                    coarse_query: coarse_query.clone(),
                    coarse_value: coarse_value.clone(),
                    patch_rho: patch_rho.clone(),
                    coarse_rho: coarse_rho.clone(),
                    hub_rho: hub_rho.clone(),
                    patch_hub_weights: None,
                    coarse_hub_weights: None,
                    neighborhood: LocalGridNeighborhood::moore(1),
                    decay: decay.clone(),
                },
                &plan,
            )
            .expect("fused structured pyramid output");
            patch_rho = output.next_patch_rho;
            coarse_rho = output.next_coarse_rho;
            hub_rho = output.next_hub_rho;
            let _ = WgpuBackend::sync(&device);
            WgpuBackend::memory_cleanup(&device);
            let _ = WgpuBackend::sync(&device);
            if step >= 8 {
                snapshots.push(memory_snapshot(&device));
            }
        }

        assert_memory_growth_bounded(
            "structured_pyramid_fused",
            &snapshots,
            256 * 1024 * 1024,
            64 * 1024 * 1024,
        );
    }
}
