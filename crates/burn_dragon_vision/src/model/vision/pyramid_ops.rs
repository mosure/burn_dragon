use super::*;
use burn_dragon_core::{
    target_major_decay_add, target_major_identity_read, target_major_outer_product,
};
use std::sync::{LazyLock, Mutex};
use std::time::Instant;

#[derive(Clone, Copy, Debug, Default)]
pub struct StageAwareHostProfileSnapshot {
    pub step_calls: u64,
    pub coarse_only_step_calls: u64,
    pub patch_local_ns: u64,
    pub coarse_local_ns: u64,
    pub patch_from_coarse_ns: u64,
    pub hub_read_ns: u64,
    pub patch_to_coarse_ns: u64,
    pub hub_update_ns: u64,
}

static STAGE_AWARE_HOST_PROFILE: LazyLock<Mutex<StageAwareHostProfileSnapshot>> =
    LazyLock::new(|| Mutex::new(StageAwareHostProfileSnapshot::default()));

#[inline]
fn stage_aware_host_profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

#[inline]
fn stage_aware_profile_record(f: impl FnOnce(&mut StageAwareHostProfileSnapshot)) {
    if !stage_aware_host_profile_enabled() {
        return;
    }
    if let Ok(mut state) = STAGE_AWARE_HOST_PROFILE.lock() {
        f(&mut state);
    }
}

pub fn stage_aware_host_profile_reset() {
    if let Ok(mut state) = STAGE_AWARE_HOST_PROFILE.lock() {
        *state = StageAwareHostProfileSnapshot::default();
    }
}

pub fn stage_aware_host_profile_snapshot() -> StageAwareHostProfileSnapshot {
    STAGE_AWARE_HOST_PROFILE
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

#[derive(Clone)]
pub(super) struct CompiledStageAwarePyramidLocalPlan<B: Backend> {
    patch_plan: Option<CompiledLocalGridRhoPlan<B>>,
    coarse_plan: Option<CompiledLocalGridRhoPlan<B>>,
    patch_shape: LocalGridShape2d,
    coarse_shape: LocalGridShape2d,
    patch_neighborhood: LocalGridNeighborhood,
    coarse_neighborhood: LocalGridNeighborhood,
}

#[derive(Clone, Copy)]
pub(super) struct CompiledStageAwarePyramidLocalPlanSpec<'a, B: Backend> {
    pub batch: usize,
    pub patch_rank: usize,
    pub coarse_rank: usize,
    pub value_dim: usize,
    pub patch_shape: LocalGridShape2d,
    pub coarse_shape: LocalGridShape2d,
    pub patch_neighborhood: LocalGridNeighborhood,
    pub coarse_neighborhood: LocalGridNeighborhood,
    pub device: &'a B::Device,
}

impl<B: Backend> CompiledStageAwarePyramidLocalPlan<B> {
    pub(super) fn new(spec: CompiledStageAwarePyramidLocalPlanSpec<'_, B>) -> Self {
        let patch_plan = (spec.patch_shape.token_count() > 1).then(|| {
            CompiledLocalGridRhoPlan::new(
                LocalGridRhoPlanSpec {
                    batch: spec.batch,
                    heads: spec.patch_rank.max(1),
                    value_heads: 1,
                    patch_tokens: spec.patch_shape.token_count(),
                    latent: 1,
                    embd: spec.value_dim.max(1),
                    grid: spec.patch_shape,
                    neighborhood: spec.patch_neighborhood,
                },
                spec.device,
            )
        });
        let coarse_plan = (spec.coarse_shape.token_count() > 1).then(|| {
            CompiledLocalGridRhoPlan::new(
                LocalGridRhoPlanSpec {
                    batch: spec.batch,
                    heads: spec.coarse_rank.max(1),
                    value_heads: 1,
                    patch_tokens: spec.coarse_shape.token_count(),
                    latent: 1,
                    embd: spec.value_dim.max(1),
                    grid: spec.coarse_shape,
                    neighborhood: spec.coarse_neighborhood,
                },
                spec.device,
            )
        });

        Self {
            patch_plan,
            coarse_plan,
            patch_shape: spec.patch_shape,
            coarse_shape: spec.coarse_shape,
            patch_neighborhood: spec.patch_neighborhood,
            coarse_neighborhood: spec.coarse_neighborhood,
        }
    }

    pub(super) fn patch_plan(&self) -> Option<&CompiledLocalGridRhoPlan<B>> {
        self.patch_plan.as_ref()
    }

    pub(super) fn coarse_plan(&self) -> Option<&CompiledLocalGridRhoPlan<B>> {
        self.coarse_plan.as_ref()
    }

    pub(super) fn patch_shape(&self) -> LocalGridShape2d {
        self.patch_shape
    }

    pub(super) fn coarse_shape(&self) -> LocalGridShape2d {
        self.coarse_shape
    }

    pub(super) fn patch_neighborhood(&self) -> LocalGridNeighborhood {
        self.patch_neighborhood
    }

    pub(super) fn coarse_neighborhood(&self) -> LocalGridNeighborhood {
        self.coarse_neighborhood
    }
}

#[derive(Clone)]
pub(super) struct CompiledSpatialProjectionPlan<B: Backend> {
    fused_weight: Tensor<B, 2>,
    fused_bias: Option<Tensor<B, 1>>,
    out_dims: Vec<usize>,
}

impl<B: Backend> CompiledSpatialProjectionPlan<B> {
    pub(super) fn new(layers: &[&Linear<B>]) -> Option<Self> {
        if layers.len() <= 1 {
            return None;
        }
        let out_dims = layers
            .iter()
            .map(|layer| layer.weight.val().shape().dims::<2>()[1])
            .collect::<Vec<_>>();
        let fused_weight = Tensor::cat(
            layers
                .iter()
                .map(|layer| layer.weight.val())
                .collect::<Vec<_>>(),
            1,
        );
        let fused_bias = if layers.iter().any(|layer| layer.bias.is_some()) {
            Some(Tensor::cat(
                layers
                    .iter()
                    .zip(out_dims.iter())
                    .map(|(layer, &out_dim)| {
                        layer
                            .bias
                            .as_ref()
                            .map(|bias| bias.val())
                            .unwrap_or_else(|| {
                                Tensor::<B, 1>::zeros([out_dim], &fused_weight.device())
                            })
                    })
                    .collect::<Vec<_>>(),
                0,
            ))
        } else {
            None
        };
        Some(Self {
            fused_weight,
            fused_bias,
            out_dims,
        })
    }
}

#[derive(Clone)]
pub(super) struct CompiledLocalBridgeProjectionPairPlan<B: Backend> {
    fused_weight: Tensor<B, 2>,
    fused_bias: Option<Tensor<B, 1>>,
    patch_x_dim: usize,
    coarse_x_dim: usize,
    value_dim: usize,
}

impl<B: Backend> CompiledLocalBridgeProjectionPairPlan<B> {
    pub(super) fn new(
        patch_x_proj: &Linear<B>,
        coarse_x_proj: &Linear<B>,
        value_proj: &Linear<B>,
    ) -> Option<Self> {
        let [patch_in, patch_out] = patch_x_proj.weight.val().shape().dims::<2>();
        let [coarse_in, coarse_out] = coarse_x_proj.weight.val().shape().dims::<2>();
        let [value_in, value_out] = value_proj.weight.val().shape().dims::<2>();
        if patch_in != coarse_in || patch_in != value_in {
            return None;
        }
        let fused_weight = Tensor::cat(
            vec![
                patch_x_proj.weight.val(),
                coarse_x_proj.weight.val(),
                value_proj.weight.val(),
            ],
            1,
        );
        let fused_bias = if patch_x_proj.bias.is_some()
            || coarse_x_proj.bias.is_some()
            || value_proj.bias.is_some()
        {
            Some(Tensor::cat(
                vec![
                    patch_x_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([patch_out], &fused_weight.device())
                        }),
                    coarse_x_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([coarse_out], &fused_weight.device())
                        }),
                    value_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([value_out], &fused_weight.device())
                        }),
                ],
                0,
            ))
        } else {
            None
        };
        Some(Self {
            fused_weight,
            fused_bias,
            patch_x_dim: patch_out,
            coarse_x_dim: coarse_out,
            value_dim: value_out,
        })
    }
}

#[derive(Clone)]
pub(super) struct CompiledStructuredDenseUpdatePairPlan<B: Backend> {
    fused_y_gate_weight: Tensor<B, 2>,
    fused_y_gate_bias: Option<Tensor<B, 1>>,
    fused_delta_weight: Tensor<B, 2>,
    fused_delta_bias: Option<Tensor<B, 1>>,
    gate_dim: usize,
    patch_delta_dim: usize,
    coarse_delta_dim: usize,
}

impl<B: Backend> CompiledStructuredDenseUpdatePairPlan<B> {
    pub(super) fn new(
        patch_y_gate_proj: &Linear<B>,
        patch_delta_proj: &Linear<B>,
        coarse_y_gate_proj: &Linear<B>,
        coarse_delta_proj: &Linear<B>,
    ) -> Option<Self> {
        let [patch_gate_in, patch_gate_out] = patch_y_gate_proj.weight.val().shape().dims::<2>();
        let [coarse_gate_in, coarse_gate_out] = coarse_y_gate_proj.weight.val().shape().dims::<2>();
        let [patch_delta_in, patch_delta_out] = patch_delta_proj.weight.val().shape().dims::<2>();
        let [coarse_delta_in, coarse_delta_out] =
            coarse_delta_proj.weight.val().shape().dims::<2>();
        if patch_gate_in != coarse_gate_in
            || patch_gate_out != patch_delta_in
            || coarse_gate_out != coarse_delta_in
            || patch_delta_in != coarse_delta_in
        {
            return None;
        }
        let fused_y_gate_weight = Tensor::cat(
            vec![
                patch_y_gate_proj.weight.val(),
                coarse_y_gate_proj.weight.val(),
            ],
            1,
        );
        let fused_y_gate_bias = if patch_y_gate_proj.bias.is_some()
            || coarse_y_gate_proj.bias.is_some()
        {
            Some(Tensor::cat(
                vec![
                    patch_y_gate_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([patch_gate_out], &fused_y_gate_weight.device())
                        }),
                    coarse_y_gate_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([coarse_gate_out], &fused_y_gate_weight.device())
                        }),
                ],
                0,
            ))
        } else {
            None
        };
        let fused_delta_weight = Tensor::cat(
            vec![
                patch_delta_proj.weight.val(),
                coarse_delta_proj.weight.val(),
            ],
            1,
        );
        let fused_delta_bias = if patch_delta_proj.bias.is_some()
            || coarse_delta_proj.bias.is_some()
        {
            Some(Tensor::cat(
                vec![
                    patch_delta_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([patch_delta_out], &fused_delta_weight.device())
                        }),
                    coarse_delta_proj
                        .bias
                        .as_ref()
                        .map(|bias| bias.val())
                        .unwrap_or_else(|| {
                            Tensor::<B, 1>::zeros([coarse_delta_out], &fused_delta_weight.device())
                        }),
                ],
                0,
            ))
        } else {
            None
        };
        Some(Self {
            fused_y_gate_weight,
            fused_y_gate_bias,
            fused_delta_weight,
            fused_delta_bias,
            gate_dim: patch_gate_out,
            patch_delta_dim: patch_delta_out,
            coarse_delta_dim: coarse_delta_out,
        })
    }
}

// Retain the older pyramid helper surface for debug/reference use while the
// active recurrent path migrates onto the shared structured-pyramid executor.
#[allow(dead_code)]
impl<B: Backend> VisionDragon<B> {
    pub(super) fn pyramid_tokens_target_major(input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, channels, height, width] = input.shape().dims::<4>();
        input
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch, height * width, channels])
    }

    pub(super) fn pyramid_tokens_from_target_major(
        input: Tensor<B, 3>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let [batch, tokens, channels] = input.shape().dims::<3>();
        assert_eq!(
            tokens,
            height * width,
            "target-major token count {} does not match spatial grid {}x{}",
            tokens,
            height,
            width
        );
        input
            .reshape([batch, height, width, channels])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
    }

    pub(super) fn pyramid_rho_to_target_major(memory: Tensor<B, 5>) -> Tensor<B, 4> {
        let [batch, rank, value_dim, height, width] = memory.shape().dims::<5>();
        memory
            .swap_dims(1, 3)
            .swap_dims(2, 4)
            .reshape([batch, height * width, rank, value_dim])
    }

    pub(super) fn pyramid_rho_from_target_major(
        memory: Tensor<B, 4>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 5> {
        let [batch, tokens, rank, value_dim] = memory.shape().dims::<4>();
        assert_eq!(
            tokens,
            height * width,
            "target-major rho token count {} does not match spatial grid {}x{}",
            tokens,
            height,
            width
        );
        memory
            .reshape([batch, height, width, rank, value_dim])
            .swap_dims(2, 4)
            .swap_dims(1, 3)
    }

    pub(super) fn apply_embed_norm_spatial(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.token_norm {
            Some(norm) => {
                let [batch, dim, height, width] = input.shape().dims::<4>();
                if batch == 0 || dim == 0 || height == 0 || width == 0 {
                    return input;
                }
                let flat = input.swap_dims(1, 3).swap_dims(1, 2);
                let flat = norm.forward(flat);
                flat.swap_dims(1, 2).swap_dims(1, 3)
            }
            None => input,
        }
    }

    pub(super) fn apply_embed_norm_spatial_pair(
        &self,
        left: Tensor<B, 4>,
        right: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        match &self.token_norm {
            Some(norm) => {
                let [left_batch, left_dim, left_height, left_width] = left.shape().dims::<4>();
                let [right_batch, right_dim, right_height, right_width] = right.shape().dims::<4>();
                if left_batch == 0
                    || left_dim == 0
                    || left_height == 0
                    || left_width == 0
                    || right_batch == 0
                    || right_dim == 0
                    || right_height == 0
                    || right_width == 0
                {
                    return (left, right);
                }
                assert_eq!(
                    left_batch, right_batch,
                    "paired embed norm requires matching batch dimensions"
                );
                assert_eq!(
                    left_dim, right_dim,
                    "paired embed norm requires matching channel dimensions"
                );
                let left_tokens = left_height * left_width;
                let right_tokens = right_height * right_width;
                let left = left.swap_dims(1, 3).swap_dims(1, 2).reshape([
                    left_batch,
                    left_tokens,
                    left_dim,
                ]);
                let right = right.swap_dims(1, 3).swap_dims(1, 2).reshape([
                    right_batch,
                    right_tokens,
                    right_dim,
                ]);
                let tokens = Tensor::cat(vec![left, right], 1);
                let tokens = norm.forward(tokens);
                let left = tokens
                    .clone()
                    .slice_dim(1, 0..left_tokens)
                    .reshape([left_batch, left_height, left_width, left_dim])
                    .swap_dims(1, 3)
                    .swap_dims(2, 3);
                let right = tokens
                    .slice_dim(1, left_tokens..left_tokens + right_tokens)
                    .reshape([right_batch, right_height, right_width, right_dim])
                    .swap_dims(1, 3)
                    .swap_dims(2, 3);
                (left, right)
            }
            None => (left, right),
        }
    }

    pub(super) fn project_spatial(&self, input: Tensor<B, 4>, layer: &Linear<B>) -> Tensor<B, 4> {
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return input;
        }
        let flat = input
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch * height * width, dim]);
        let flat = layer.forward(flat);
        let out_dim = flat.shape().dims::<2>()[1];
        flat.reshape([batch, height, width, out_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
    }

    pub(super) fn project_spatial_many(
        &self,
        input: Tensor<B, 4>,
        layers: &[&Linear<B>],
    ) -> Vec<Tensor<B, 4>> {
        if layers.is_empty() {
            return Vec::new();
        }
        // In the recurrent pyramid path, these "many" calls are usually just 2-4
        // projections. Rebuilding concatenated weights/biases every step is more
        // expensive than issuing the individual projections, so keep the hot small
        // cases on the direct path and reserve fused concatenation for larger fans.
        if layers.len() <= 4 {
            return layers
                .iter()
                .map(|layer| self.project_spatial(input.clone(), layer))
                .collect();
        }
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return layers
                .iter()
                .map(|layer| self.project_spatial(input.clone(), layer))
                .collect();
        }
        let flat = input
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch * height * width, dim]);
        let out_dims: Vec<usize> = layers
            .iter()
            .map(|layer| layer.weight.val().shape().dims::<2>()[1])
            .collect();
        let fused_weight = Tensor::cat(layers.iter().map(|layer| layer.weight.val()).collect(), 1);
        let fused_bias = if layers.iter().any(|layer| layer.bias.is_some()) {
            Some(Tensor::cat(
                layers
                    .iter()
                    .zip(out_dims.iter())
                    .map(|(layer, &out_dim)| {
                        layer
                            .bias
                            .as_ref()
                            .map(|bias| bias.val())
                            .unwrap_or_else(|| Tensor::<B, 1>::zeros([out_dim], &flat.device()))
                    })
                    .collect(),
                0,
            ))
        } else {
            None
        };
        let fused = burn::tensor::module::linear(flat, fused_weight, fused_bias);
        let mut outputs = Vec::with_capacity(layers.len());
        let mut start = 0;
        for &out_dim in &out_dims {
            let projected = fused.clone().slice_dim(1, start..start + out_dim);
            outputs.push(
                projected
                    .reshape([batch, height, width, out_dim])
                    .swap_dims(1, 3)
                    .swap_dims(2, 3),
            );
            start += out_dim;
        }
        outputs
    }

    pub(super) fn project_spatial_many_with_plan(
        &self,
        input: Tensor<B, 4>,
        plan: &CompiledSpatialProjectionPlan<B>,
    ) -> Vec<Tensor<B, 4>> {
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return plan
                .out_dims
                .iter()
                .map(|&out_dim| {
                    Tensor::<B, 4>::zeros([batch, out_dim, height, width], &input.device())
                })
                .collect();
        }
        let flat = input
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch * height * width, dim]);
        let fused =
            burn::tensor::module::linear(flat, plan.fused_weight.clone(), plan.fused_bias.clone());
        let mut outputs = Vec::with_capacity(plan.out_dims.len());
        let mut start = 0;
        for &out_dim in &plan.out_dims {
            let projected = fused.clone().slice_dim(1, start..start + out_dim);
            outputs.push(
                projected
                    .reshape([batch, height, width, out_dim])
                    .swap_dims(1, 3)
                    .swap_dims(2, 3),
            );
            start += out_dim;
        }
        outputs
    }

    pub(super) fn project_spatial_pair(
        &self,
        left: Tensor<B, 4>,
        right: Tensor<B, 4>,
        layer: &Linear<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [left_batch, left_dim, left_height, left_width] = left.shape().dims::<4>();
        let [right_batch, right_dim, right_height, right_width] = right.shape().dims::<4>();
        if left_batch == 0
            || left_dim == 0
            || left_height == 0
            || left_width == 0
            || right_batch == 0
            || right_dim == 0
            || right_height == 0
            || right_width == 0
        {
            return (
                self.project_spatial(left, layer),
                self.project_spatial(right, layer),
            );
        }
        assert_eq!(
            left_batch, right_batch,
            "paired spatial projection requires matching batch dimensions"
        );
        assert_eq!(
            left_dim, right_dim,
            "paired spatial projection requires matching channel dimensions"
        );
        let left_tokens = left_height * left_width;
        let right_tokens = right_height * right_width;
        let left =
            left.swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([left_batch, left_tokens, left_dim]);
        let right =
            right
                .swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([right_batch, right_tokens, right_dim]);
        let tokens = Tensor::cat(vec![left, right], 1);
        let flat = tokens.reshape([left_batch * (left_tokens + right_tokens), left_dim]);
        let flat = layer.forward(flat);
        let out_dim = flat.shape().dims::<2>()[1];
        let tokens = flat.reshape([left_batch, left_tokens + right_tokens, out_dim]);
        let left = tokens
            .clone()
            .slice_dim(1, 0..left_tokens)
            .reshape([left_batch, left_height, left_width, out_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let right = tokens
            .slice_dim(1, left_tokens..left_tokens + right_tokens)
            .reshape([right_batch, right_height, right_width, out_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        (left, right)
    }

    pub(super) fn project_local_bridge_pair_with_plan(
        &self,
        patch_state: Tensor<B, 4>,
        coarse_state: Tensor<B, 4>,
        plan: &CompiledLocalBridgeProjectionPairPlan<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>) {
        let [patch_batch, patch_dim, patch_height, patch_width] = patch_state.shape().dims::<4>();
        let [coarse_batch, coarse_dim, coarse_height, coarse_width] =
            coarse_state.shape().dims::<4>();
        if patch_batch == 0
            || patch_dim == 0
            || patch_height == 0
            || patch_width == 0
            || coarse_batch == 0
            || coarse_dim == 0
            || coarse_height == 0
            || coarse_width == 0
        {
            return (
                Tensor::<B, 4>::zeros(
                    [
                        patch_batch.max(1),
                        plan.patch_x_dim.max(1),
                        patch_height.max(1),
                        patch_width.max(1),
                    ],
                    &patch_state.device(),
                ),
                Tensor::<B, 4>::zeros(
                    [
                        patch_batch.max(1),
                        plan.value_dim.max(1),
                        patch_height.max(1),
                        patch_width.max(1),
                    ],
                    &patch_state.device(),
                ),
                Tensor::<B, 4>::zeros(
                    [
                        coarse_batch.max(1),
                        plan.coarse_x_dim.max(1),
                        coarse_height.max(1),
                        coarse_width.max(1),
                    ],
                    &coarse_state.device(),
                ),
                Tensor::<B, 4>::zeros(
                    [
                        coarse_batch.max(1),
                        plan.value_dim.max(1),
                        coarse_height.max(1),
                        coarse_width.max(1),
                    ],
                    &coarse_state.device(),
                ),
            );
        }
        assert_eq!(
            patch_batch, coarse_batch,
            "bridge pair projection requires matching batch dimensions"
        );
        assert_eq!(
            patch_dim, coarse_dim,
            "bridge pair projection requires matching channel dimensions"
        );
        let patch_tokens = patch_height * patch_width;
        let coarse_tokens = coarse_height * coarse_width;
        let patch = patch_state.swap_dims(1, 3).swap_dims(1, 2).reshape([
            patch_batch,
            patch_tokens,
            patch_dim,
        ]);
        let coarse = coarse_state.swap_dims(1, 3).swap_dims(1, 2).reshape([
            coarse_batch,
            coarse_tokens,
            coarse_dim,
        ]);
        let tokens = Tensor::cat(vec![patch, coarse], 1);
        let flat = tokens.reshape([patch_batch * (patch_tokens + coarse_tokens), patch_dim]);
        let fused =
            burn::tensor::module::linear(flat, plan.fused_weight.clone(), plan.fused_bias.clone())
                .reshape([
                    patch_batch,
                    patch_tokens + coarse_tokens,
                    plan.patch_x_dim + plan.coarse_x_dim + plan.value_dim,
                ]);
        let patch_all = fused.clone().slice_dim(1, 0..patch_tokens);
        let coarse_all = fused.slice_dim(1, patch_tokens..patch_tokens + coarse_tokens);
        let patch_x = patch_all
            .clone()
            .slice_dim(2, 0..plan.patch_x_dim)
            .reshape([patch_batch, patch_height, patch_width, plan.patch_x_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let patch_value = patch_all
            .slice_dim(
                2,
                plan.patch_x_dim + plan.coarse_x_dim
                    ..plan.patch_x_dim + plan.coarse_x_dim + plan.value_dim,
            )
            .reshape([patch_batch, patch_height, patch_width, plan.value_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let coarse_x = coarse_all
            .clone()
            .slice_dim(2, plan.patch_x_dim..plan.patch_x_dim + plan.coarse_x_dim)
            .reshape([coarse_batch, coarse_height, coarse_width, plan.coarse_x_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let coarse_value = coarse_all
            .slice_dim(
                2,
                plan.patch_x_dim + plan.coarse_x_dim
                    ..plan.patch_x_dim + plan.coarse_x_dim + plan.value_dim,
            )
            .reshape([coarse_batch, coarse_height, coarse_width, plan.value_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        (patch_x, patch_value, coarse_x, coarse_value)
    }

    pub(super) fn pyramid_patch_tokens_to_spatial(
        &self,
        patch_tokens: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, patch_count, dim] = patch_tokens.shape().dims::<3>();
        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        assert_eq!(
            patch_count,
            grid_height * grid_width,
            "pyramid patch tokens require grid {}x{} (got {})",
            grid_height,
            grid_width,
            patch_count
        );
        let h8 = patch_tokens
            .reshape([batch, grid_height, grid_width, dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        self.apply_embed_norm_spatial(h8)
    }

    pub(super) fn pyramid_spatial_to_patch_tokens(&self, h8: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, dim, height, width] = h8.shape().dims::<4>();
        h8.swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch, height * width, dim])
    }

    pub(super) fn pyramid_pool_patch_state(&self, h8: Tensor<B, 4>) -> Tensor<B, 4> {
        let coarse_stride = self.trm_graph.coarse_stride.max(1);
        if coarse_stride <= 1 {
            return h8;
        }
        let [batch, dim, grid_height, grid_width] = h8.shape().dims::<4>();
        let h32_height = grid_height / coarse_stride.max(1);
        let h32_width = grid_width / coarse_stride.max(1);
        if h32_height == 0 || h32_width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch, dim, h32_height.max(1), h32_width.max(1)],
                &h8.device(),
            );
        }
        let pooled = h8
            .reshape([
                batch,
                dim,
                h32_height.max(1),
                coarse_stride,
                h32_width.max(1),
                coarse_stride,
            ])
            .sum_dims_squeeze::<4, usize>(&[3, 5])
            .div_scalar((coarse_stride * coarse_stride) as f32);
        self.apply_embed_norm_spatial(pooled)
    }

    pub(super) fn trm_shift(input: Tensor<B, 4>, dy: isize, dx: isize) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.shape().dims::<4>();
        if height == 0 || width == 0 {
            return input;
        }
        let device = input.device();
        let mut out = input;

        if dy != 0 {
            let shift = dy.unsigned_abs();
            if shift >= height {
                out = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
            } else if dy > 0 {
                let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
                let cropped = out.slice_dim(2, 0..(height - shift));
                out = Tensor::cat(vec![pad, cropped], 2);
            } else {
                let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
                let cropped = out.slice_dim(2, shift..height);
                out = Tensor::cat(vec![cropped, pad], 2);
            }
        }

        if dx != 0 {
            let shift = dx.unsigned_abs();
            if shift >= width {
                out = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
            } else if dx > 0 {
                let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
                let cropped = out.slice_dim(3, 0..(width - shift));
                out = Tensor::cat(vec![pad, cropped], 3);
            } else {
                let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
                let cropped = out.slice_dim(3, shift..width);
                out = Tensor::cat(vec![cropped, pad], 3);
            }
        }

        out
    }

    pub(super) fn pyramid_contract(
        &self,
        memory: Tensor<B, 5>,
        query: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, rank, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || rank == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &memory.device(),
            );
        }
        let memory = Self::pyramid_rho_to_target_major(memory);
        let query = Self::pyramid_tokens_target_major(query);
        let read = target_major_identity_read(query, memory);
        Self::pyramid_tokens_from_target_major(read, height, width)
            .reshape([batch, value_dim, height, width])
    }

    pub(super) fn pyramid_local_read(
        &self,
        memory: Tensor<B, 5>,
        query: Tensor<B, 4>,
        coarse_bank: bool,
    ) -> Tensor<B, 4> {
        let [batch, _, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &memory.device(),
            );
        }
        let mut acc = Tensor::<B, 4>::zeros([batch, value_dim, height, width], &memory.device());
        let radius = if coarse_bank {
            self.trm_graph.coarse_local_radius_resolved()
        } else {
            self.trm_graph.local_radius
        }
        .max(1) as isize;
        let allow_diagonals = if coarse_bank {
            self.trm_graph.coarse_local_diagonals_resolved()
        } else {
            self.trm_graph.local_diagonals
        };
        let allow_self = if coarse_bank {
            self.trm_graph.coarse_local_self_resolved()
        } else {
            self.trm_graph.local_self
        };
        if allow_self {
            acc = acc + self.pyramid_contract(memory.clone(), query.clone());
        }
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dy == 0 && dx == 0 {
                    continue;
                }
                if !allow_diagonals && dy != 0 && dx != 0 {
                    continue;
                }
                let shifted = Self::trm_shift(query.clone(), dy, dx);
                let msg = self.pyramid_contract(memory.clone(), shifted);
                let msg = Self::trm_shift(msg, -dy, -dx);
                acc = acc + msg;
            }
        }
        acc
    }

    pub(super) fn pyramid_patch_neighborhood(&self) -> LocalGridNeighborhood {
        LocalGridNeighborhood {
            radius: self.trm_graph.local_radius,
            diagonals: self.trm_graph.local_diagonals,
            self_edges: self.trm_graph.local_self,
        }
    }

    pub(super) fn pyramid_coarse_neighborhood(&self) -> LocalGridNeighborhood {
        LocalGridNeighborhood {
            radius: self.trm_graph.coarse_local_radius_resolved(),
            diagonals: self.trm_graph.coarse_local_diagonals_resolved(),
            self_edges: self.trm_graph.coarse_local_self_resolved(),
        }
    }

    pub(super) fn pyramid_spatial_tokens_to_local_grid(input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.shape().dims::<4>();
        Self::pyramid_tokens_target_major(input)
            .swap_dims(1, 2)
            .reshape([batch, channels, height * width, 1])
    }

    pub(super) fn pyramid_spatial_values_to_local_grid(input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.shape().dims::<4>();
        Self::pyramid_tokens_target_major(input).reshape([batch, 1, height * width, channels])
    }

    pub(super) fn pyramid_spatial_rho_to_local_grid(input: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, rank, value_dim, height, width] = input.shape().dims::<5>();
        Self::pyramid_rho_to_target_major(input)
            .swap_dims(1, 2)
            .reshape([batch, rank, height * width, 1, value_dim])
    }

    pub(super) fn pyramid_local_grid_context_to_spatial(
        input: Tensor<B, 4>,
        shape: LocalGridShape2d,
    ) -> Tensor<B, 4> {
        let [batch, _, tokens, value_dim] = input.shape().dims::<4>();
        let target_major = input.sum_dim(1).reshape([batch, tokens, value_dim]);
        Self::pyramid_tokens_from_target_major(target_major, shape.height, shape.width)
    }

    pub(super) fn pyramid_local_grid_rho_to_target_major(input: Tensor<B, 5>) -> Tensor<B, 4> {
        let [batch, rank, tokens, _, value_dim] = input.shape().dims::<5>();
        input
            .reshape([batch, rank, tokens, value_dim])
            .swap_dims(1, 2)
    }

    pub(super) fn pyramid_local_grid_rho_to_spatial(
        input: Tensor<B, 5>,
        shape: LocalGridShape2d,
    ) -> Tensor<B, 5> {
        Self::pyramid_rho_from_target_major(
            Self::pyramid_local_grid_rho_to_target_major(input),
            shape.height,
            shape.width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_local_step_with_plan(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho: Tensor<B, 5>,
        shape: LocalGridShape2d,
        decay: Tensor<B, 1>,
        plan: Option<&CompiledLocalGridRhoPlan<B>>,
        _neighborhood: LocalGridNeighborhood,
        coarse_bank: bool,
        read_enabled: bool,
        write_enabled: bool,
    ) -> (Tensor<B, 4>, Tensor<B, 5>) {
        let [batch, _, value_dim, height, width] = rho.shape().dims::<5>();
        let zero_context = || {
            Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &rho.device(),
            )
        };

        if !read_enabled && !write_enabled {
            let next_rho = Self::pyramid_rho_from_target_major(
                target_major_decay_add(
                    Self::pyramid_rho_to_target_major(rho.clone()),
                    Tensor::<B, 4>::zeros(
                        [
                            batch.max(1),
                            shape.token_count().max(1),
                            rho.shape().dims::<5>()[1].max(1),
                            value_dim.max(1),
                        ],
                        &rho.device(),
                    ),
                    decay,
                ),
                shape.height,
                shape.width,
            );
            return (zero_context(), next_rho);
        }

        let fused_value = if write_enabled {
            value.clone()
        } else {
            let [value_batch, value_channels, value_height, value_width] =
                value.shape().dims::<4>();
            Tensor::<B, 4>::zeros(
                [
                    value_batch.max(1),
                    value_channels.max(1),
                    value_height.max(1),
                    value_width.max(1),
                ],
                &value.device(),
            )
        };

        if let Some(plan) = plan
            && let Some(output) = try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan(
                &Self::pyramid_spatial_tokens_to_local_grid(query.clone()),
                &Self::pyramid_spatial_values_to_local_grid(fused_value),
                Some(&Self::pyramid_spatial_rho_to_local_grid(rho.clone())),
                &decay,
                plan,
            )
        {
            let context = if read_enabled {
                Self::pyramid_local_grid_context_to_spatial(output.context, shape)
            } else {
                zero_context()
            };
            let next_rho = Self::pyramid_local_grid_rho_to_spatial(output.rho, shape);
            return (context, next_rho);
        }

        let context = if read_enabled {
            self.pyramid_local_read(rho.clone(), query.clone(), coarse_bank)
        } else {
            zero_context()
        };
        let update = if write_enabled {
            self.pyramid_outer_product(query, value)
        } else {
            Tensor::<B, 5>::zeros(rho.shape().dims::<5>(), &rho.device())
        };
        let next_rho = Self::pyramid_rho_from_target_major(
            target_major_decay_add(
                Self::pyramid_rho_to_target_major(rho),
                Self::pyramid_rho_to_target_major(update),
                decay,
            ),
            shape.height,
            shape.width,
        );
        (context, next_rho)
    }

    pub(super) fn pyramid_cross_scale_read(
        &self,
        memory: Tensor<B, 5>,
        query: Tensor<B, 4>,
        scale: usize,
    ) -> Tensor<B, 4> {
        let scale = scale.max(1);
        if scale == 1 {
            return self.pyramid_contract(memory, query);
        }
        let mut up = memory.repeat_dim(3, scale).repeat_dim(4, scale);
        let [_, _, height, width] = query.shape().dims::<4>();
        let [_, _, _, up_h, up_w] = up.shape().dims::<5>();
        if up_h != height {
            up = up.slice_dim(3, 0..height.min(up_h));
        }
        if up_w != width {
            up = up.slice_dim(4, 0..width.min(up_w));
        }
        self.pyramid_contract(up, query)
    }

    pub(super) fn pyramid_outer_product(&self, x: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 5> {
        let [_, _, height, width] = x.shape().dims::<4>();
        let x = Self::pyramid_tokens_target_major(x);
        let v = Self::pyramid_tokens_target_major(v);
        let update = target_major_outer_product(x, v);
        Self::pyramid_rho_from_target_major(update, height, width)
    }

    pub(super) fn pyramid_pool_outer(&self, u: Tensor<B, 5>, scale: usize) -> Tensor<B, 5> {
        let scale = scale.max(1);
        if scale == 1 {
            return u;
        }
        let [batch, rank, value_dim, height, width] = u.shape().dims::<5>();
        let pooled_height = height / scale;
        let pooled_width = width / scale;
        if pooled_height == 0 || pooled_width == 0 {
            return Tensor::<B, 5>::zeros(
                [
                    batch,
                    rank,
                    value_dim,
                    pooled_height.max(1),
                    pooled_width.max(1),
                ],
                &u.device(),
            );
        }
        u.reshape([batch, rank * value_dim, height, width])
            .reshape([
                batch,
                rank * value_dim,
                pooled_height,
                scale,
                pooled_width,
                scale,
            ])
            .sum_dims_squeeze::<4, usize>(&[3, 5])
            .reshape([batch, rank, value_dim, pooled_height, pooled_width])
    }

    pub(super) fn pyramid_update_state(
        &self,
        state: Tensor<B, 4>,
        x: Tensor<B, 4>,
        msg: Tensor<B, 4>,
        pyramid_y_gate_proj: &Linear<B>,
        pyramid_delta_proj: &Linear<B>,
        pyramid_value_norm: &DragonNorm<B>,
    ) -> Tensor<B, 4> {
        let [batch, dense_dim, height, width] = state.shape().dims::<4>();
        let [x_batch, rank, x_height, x_width] = x.shape().dims::<4>();
        let [msg_batch, value_dim, msg_height, msg_width] = msg.shape().dims::<4>();
        assert_eq!(x_batch, batch, "pyramid x batch must match state batch");
        assert_eq!(msg_batch, batch, "pyramid msg batch must match state batch");
        assert_eq!(x_height, height, "pyramid x height must match state height");
        assert_eq!(x_width, width, "pyramid x width must match state width");
        assert_eq!(
            msg_height, height,
            "pyramid msg height must match state height"
        );
        assert_eq!(msg_width, width, "pyramid msg width must match state width");

        let x_tokens = x
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch, height * width, rank]);
        let msg_tokens =
            msg.swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([batch, height * width, value_dim]);
        let delta = structured_dense_update_tokens(
            x_tokens,
            msg_tokens,
            pyramid_y_gate_proj,
            pyramid_delta_proj,
            Some(pyramid_value_norm),
        )
        .delta_dense
        .reshape([batch, height, width, dense_dim])
        .swap_dims(1, 3)
        .swap_dims(2, 3);
        let next = state + delta;
        self.apply_embed_norm_spatial(next)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_update_states_separate_with_plan(
        &self,
        patch_state: Tensor<B, 4>,
        patch_x: Tensor<B, 4>,
        patch_msg: Tensor<B, 4>,
        coarse_state: Tensor<B, 4>,
        coarse_x: Tensor<B, 4>,
        coarse_msg: Tensor<B, 4>,
        pyramid_value_norm: &DragonNorm<B>,
        plan: &CompiledStructuredDenseUpdatePairPlan<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, patch_dense_dim, patch_height, patch_width] = patch_state.shape().dims::<4>();
        let [coarse_batch, coarse_dense_dim, coarse_height, coarse_width] =
            coarse_state.shape().dims::<4>();
        let [patch_x_batch, patch_rank, patch_x_height, patch_x_width] =
            patch_x.shape().dims::<4>();
        let [coarse_x_batch, coarse_rank, coarse_x_height, coarse_x_width] =
            coarse_x.shape().dims::<4>();
        let [
            patch_msg_batch,
            value_dim,
            patch_msg_height,
            patch_msg_width,
        ] = patch_msg.shape().dims::<4>();
        let [
            coarse_msg_batch,
            coarse_value_dim,
            coarse_msg_height,
            coarse_msg_width,
        ] = coarse_msg.shape().dims::<4>();
        assert_eq!(coarse_batch, batch, "coarse batch must match patch batch");
        assert_eq!(patch_x_batch, batch, "patch x batch must match state batch");
        assert_eq!(
            coarse_x_batch, batch,
            "coarse x batch must match state batch"
        );
        assert_eq!(
            patch_msg_batch, batch,
            "patch msg batch must match state batch"
        );
        assert_eq!(
            coarse_msg_batch, batch,
            "coarse msg batch must match state batch"
        );
        assert_eq!(
            patch_x_height, patch_height,
            "patch x height must match patch state height"
        );
        assert_eq!(
            patch_x_width, patch_width,
            "patch x width must match patch state width"
        );
        assert_eq!(
            coarse_x_height, coarse_height,
            "coarse x height must match coarse state height"
        );
        assert_eq!(
            coarse_x_width, coarse_width,
            "coarse x width must match coarse state width"
        );
        assert_eq!(
            patch_msg_height, patch_height,
            "patch msg height must match patch state height"
        );
        assert_eq!(
            patch_msg_width, patch_width,
            "patch msg width must match patch state width"
        );
        assert_eq!(
            coarse_msg_height, coarse_height,
            "coarse msg height must match coarse state height"
        );
        assert_eq!(
            coarse_msg_width, coarse_width,
            "coarse msg width must match coarse state width"
        );
        assert_eq!(
            coarse_value_dim, value_dim,
            "coarse value dim must match patch value dim"
        );
        assert_eq!(
            patch_rank, coarse_rank,
            "paired fused update requires matching ranks"
        );
        assert_eq!(
            patch_rank, plan.gate_dim,
            "compiled update plan gate dim must match patch/coarse rank"
        );

        let patch_tokens = patch_height * patch_width;
        let coarse_tokens = coarse_height * coarse_width;
        let total_tokens = patch_tokens + coarse_tokens;

        let patch_x_tokens =
            patch_x
                .swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([batch, patch_tokens, patch_rank]);
        let coarse_x_tokens =
            coarse_x
                .swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([batch, coarse_tokens, coarse_rank]);
        let patch_msg_tokens =
            patch_msg
                .swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([batch, patch_tokens, value_dim]);
        let coarse_msg_tokens =
            coarse_msg
                .swap_dims(1, 3)
                .swap_dims(1, 2)
                .reshape([batch, coarse_tokens, value_dim]);
        let msg_tokens = Tensor::cat(vec![patch_msg_tokens, coarse_msg_tokens], 1);
        let msg_tokens = pyramid_value_norm.forward(msg_tokens);
        let fused_y_gate = burn::tensor::module::linear(
            msg_tokens.reshape([batch * total_tokens, value_dim]),
            plan.fused_y_gate_weight.clone(),
            plan.fused_y_gate_bias.clone(),
        )
        .reshape([batch, total_tokens, patch_rank * 2]);
        let patch_y_gate = activation::relu(
            fused_y_gate
                .clone()
                .slice_dim(1, 0..patch_tokens)
                .slice_dim(2, 0..patch_rank),
        );
        let coarse_y_gate = activation::relu(
            fused_y_gate
                .slice_dim(1, patch_tokens..patch_tokens + coarse_tokens)
                .slice_dim(2, patch_rank..patch_rank * 2),
        );
        let patch_y_neuron = patch_y_gate.mul(patch_x_tokens);
        let coarse_y_neuron = coarse_y_gate.mul(coarse_x_tokens);
        let y_neuron = Tensor::cat(vec![patch_y_neuron, coarse_y_neuron], 1);
        let fused_delta = burn::tensor::module::linear(
            y_neuron.reshape([batch * total_tokens, patch_rank]),
            plan.fused_delta_weight.clone(),
            plan.fused_delta_bias.clone(),
        )
        .reshape([
            batch,
            total_tokens,
            plan.patch_delta_dim + plan.coarse_delta_dim,
        ]);
        let patch_delta = fused_delta
            .clone()
            .slice_dim(1, 0..patch_tokens)
            .slice_dim(2, 0..plan.patch_delta_dim)
            .reshape([batch, patch_height, patch_width, patch_dense_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let coarse_delta = fused_delta
            .slice_dim(1, patch_tokens..patch_tokens + coarse_tokens)
            .slice_dim(
                2,
                plan.patch_delta_dim..plan.patch_delta_dim + plan.coarse_delta_dim,
            )
            .reshape([batch, coarse_height, coarse_width, coarse_dense_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        self.apply_embed_norm_spatial_pair(patch_state + patch_delta, coarse_state + coarse_delta)
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub(super) fn pyramid_update_states(
        &self,
        patch_state: Tensor<B, 4>,
        patch_x: Tensor<B, 4>,
        patch_msg: Tensor<B, 4>,
        coarse_state: Tensor<B, 4>,
        coarse_x: Tensor<B, 4>,
        coarse_msg: Tensor<B, 4>,
        pyramid_y_gate_proj: &Linear<B>,
        pyramid_delta_proj: &Linear<B>,
        pyramid_value_norm: &DragonNorm<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, dense_dim, patch_height, patch_width] = patch_state.shape().dims::<4>();
        let [coarse_batch, coarse_dense_dim, coarse_height, coarse_width] =
            coarse_state.shape().dims::<4>();
        let [patch_x_batch, rank, patch_x_height, patch_x_width] = patch_x.shape().dims::<4>();
        let [coarse_x_batch, coarse_rank, coarse_x_height, coarse_x_width] =
            coarse_x.shape().dims::<4>();
        let [
            patch_msg_batch,
            value_dim,
            patch_msg_height,
            patch_msg_width,
        ] = patch_msg.shape().dims::<4>();
        let [
            coarse_msg_batch,
            coarse_value_dim,
            coarse_msg_height,
            coarse_msg_width,
        ] = coarse_msg.shape().dims::<4>();
        assert_eq!(coarse_batch, batch, "coarse batch must match patch batch");
        assert_eq!(
            coarse_dense_dim, dense_dim,
            "coarse dense dim must match patch dense dim"
        );
        assert_eq!(patch_x_batch, batch, "patch x batch must match state batch");
        assert_eq!(
            coarse_x_batch, batch,
            "coarse x batch must match state batch"
        );
        assert_eq!(
            patch_msg_batch, batch,
            "patch msg batch must match state batch"
        );
        assert_eq!(
            coarse_msg_batch, batch,
            "coarse msg batch must match state batch"
        );
        assert_eq!(coarse_rank, rank, "coarse rank must match patch rank");
        assert_eq!(
            coarse_value_dim, value_dim,
            "coarse value dim must match patch value dim"
        );
        assert_eq!(
            patch_x_height, patch_height,
            "patch x height must match patch state height"
        );
        assert_eq!(
            patch_x_width, patch_width,
            "patch x width must match patch state width"
        );
        assert_eq!(
            patch_msg_height, patch_height,
            "patch msg height must match patch state height"
        );
        assert_eq!(
            patch_msg_width, patch_width,
            "patch msg width must match patch state width"
        );
        assert_eq!(
            coarse_x_height, coarse_height,
            "coarse x height must match coarse state height"
        );
        assert_eq!(
            coarse_x_width, coarse_width,
            "coarse x width must match coarse state width"
        );
        assert_eq!(
            coarse_msg_height, coarse_height,
            "coarse msg height must match coarse state height"
        );
        assert_eq!(
            coarse_msg_width, coarse_width,
            "coarse msg width must match coarse state width"
        );

        let patch_tokens = patch_height * patch_width;
        let coarse_tokens = coarse_height * coarse_width;
        let x_tokens = Tensor::cat(
            vec![
                patch_x
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([batch, patch_tokens, rank]),
                coarse_x
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([batch, coarse_tokens, rank]),
            ],
            1,
        );
        let msg_tokens = Tensor::cat(
            vec![
                patch_msg
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([batch, patch_tokens, value_dim]),
                coarse_msg.swap_dims(1, 3).swap_dims(1, 2).reshape([
                    batch,
                    coarse_tokens,
                    value_dim,
                ]),
            ],
            1,
        );

        let delta_tokens = structured_dense_update_tokens(
            x_tokens,
            msg_tokens,
            pyramid_y_gate_proj,
            pyramid_delta_proj,
            Some(pyramid_value_norm),
        )
        .delta_dense;

        let patch_delta = delta_tokens
            .clone()
            .slice_dim(1, 0..patch_tokens)
            .reshape([batch, patch_height, patch_width, dense_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        let coarse_delta = delta_tokens
            .slice_dim(1, patch_tokens..patch_tokens + coarse_tokens)
            .reshape([batch, coarse_height, coarse_width, coarse_dense_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);

        self.apply_embed_norm_spatial_pair(patch_state + patch_delta, coarse_state + coarse_delta)
    }

    pub(super) fn pyramid_hub_weights(
        &self,
        h8: Tensor<B, 4>,
        h32: Tensor<B, 4>,
        hub_count: usize,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        if hub_count <= 1 {
            return (None, None);
        }
        let hub_gate = self.pyramid_hub_gate.as_ref();
        if let Some(gate) = hub_gate {
            let (w8, w32) = self.project_spatial_pair(h8, h32, gate);
            let w8 = self.normalize_hub_weights(w8);
            let w32 = self.normalize_hub_weights(w32);
            (Some(w8), Some(w32))
        } else {
            let w8 = self.pyramid_hub_weights_single(h8, hub_count, None);
            let w32 = self.pyramid_hub_weights_single(h32, hub_count, None);
            (Some(w8), Some(w32))
        }
    }

    pub(super) fn normalize_hub_weights(&self, weights: Tensor<B, 4>) -> Tensor<B, 4> {
        let weights = activation::relu(weights);
        let denom = weights.clone().sum_dim(1).add_scalar(ROW_NORM_EPS);
        weights / denom
    }

    pub(super) fn pyramid_hub_weights_single(
        &self,
        h: Tensor<B, 4>,
        hub_count: usize,
        hub_gate: Option<&Linear<B>>,
    ) -> Tensor<B, 4> {
        let [batch, _, height, width] = h.shape().dims::<4>();
        let device = h.device();
        if let Some(gate) = hub_gate {
            let weights = self.project_spatial(h, gate);
            self.normalize_hub_weights(weights)
        } else {
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &device)
                .div_scalar(hub_count as f32)
        }
    }

    pub(super) fn pyramid_hub_read(
        &self,
        hub: Tensor<B, 4>,
        query: Tensor<B, 4>,
        weights: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [batch, hubs, rank, value_dim] = hub.shape().dims::<4>();
        let [_, _, height, width] = query.shape().dims::<4>();
        if batch == 0 || hubs == 0 || rank == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &hub.device(),
            );
        }
        let tokens = height * width;
        let hub_mat = hub.swap_dims(2, 3);
        let query_mat = query.reshape([batch, 1, rank, tokens]);
        let reduced = hub_mat.matmul(query_mat);
        self.pyramid_reduce_hub_values(reduced, weights, height, width)
    }

    pub(super) fn pyramid_hub_read_pair(
        &self,
        hub: Tensor<B, 4>,
        patch_query: Option<Tensor<B, 4>>,
        coarse_query: Option<Tensor<B, 4>>,
        patch_weights: Option<Tensor<B, 4>>,
        coarse_weights: Option<Tensor<B, 4>>,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        match (patch_query, coarse_query) {
            (None, None) => (None, None),
            (Some(patch_query), None) => (
                Some(self.pyramid_hub_read(hub, patch_query, patch_weights)),
                None,
            ),
            (None, Some(coarse_query)) => (
                None,
                Some(self.pyramid_hub_read(hub, coarse_query, coarse_weights)),
            ),
            (Some(patch_query), Some(coarse_query)) => {
                let [batch, hubs, rank, value_dim] = hub.shape().dims::<4>();
                let [_, _, patch_height, patch_width] = patch_query.shape().dims::<4>();
                let [_, _, coarse_height, coarse_width] = coarse_query.shape().dims::<4>();
                let patch_tokens = patch_height * patch_width;
                let coarse_tokens = coarse_height * coarse_width;
                let total_tokens = patch_tokens + coarse_tokens;
                let hub_mat = hub.swap_dims(2, 3);
                let query = Tensor::cat(
                    vec![
                        patch_query.reshape([batch, rank, patch_tokens]),
                        coarse_query.reshape([batch, rank, coarse_tokens]),
                    ],
                    2,
                )
                .reshape([batch, 1, rank, total_tokens]);
                let reduced = hub_mat.matmul(query);
                let patch_reduced = reduced.clone().slice_dim(3, 0..patch_tokens).reshape([
                    batch,
                    hubs,
                    value_dim,
                    patch_tokens,
                ]);
                let coarse_reduced = reduced.slice_dim(3, patch_tokens..total_tokens).reshape([
                    batch,
                    hubs,
                    value_dim,
                    coarse_tokens,
                ]);
                (
                    Some(self.pyramid_reduce_hub_values(
                        patch_reduced,
                        patch_weights,
                        patch_height,
                        patch_width,
                    )),
                    Some(self.pyramid_reduce_hub_values(
                        coarse_reduced,
                        coarse_weights,
                        coarse_height,
                        coarse_width,
                    )),
                )
            }
        }
    }

    fn pyramid_reduce_hub_values(
        &self,
        values: Tensor<B, 4>,
        weights: Option<Tensor<B, 4>>,
        height: usize,
        width: usize,
    ) -> Tensor<B, 4> {
        let [batch, hubs, value_dim, tokens] = values.shape().dims::<4>();
        let mut reduced = values;
        let has_weights = weights.is_some();
        if let Some(weights) = weights {
            let hub_weights = weights.reshape([batch, hubs, 1, tokens]);
            reduced = reduced.mul(hub_weights);
        }
        let reduced = reduced.sum_dim(1);
        let reduced = if !has_weights && hubs > 1 {
            reduced.div_scalar(hubs as f32)
        } else {
            reduced
        };
        reduced.reshape([batch, value_dim, height, width])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_update_hub(
        &self,
        hub: Tensor<B, 4>,
        u8: Option<Tensor<B, 5>>,
        u32: Option<Tensor<B, 5>>,
        hub_w8: Option<Tensor<B, 4>>,
        hub_w32: Option<Tensor<B, 4>>,
        hub_count: usize,
        decay: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        let zero_delta = || {
            let [batch, hubs, rank, value_dim] = hub.shape().dims::<4>();
            Tensor::<B, 4>::zeros([batch, hubs, rank, value_dim], &hub.device())
        };
        if hub_count <= 1 {
            let sum8 = u8
                .map(|u8| u8.sum_dims_squeeze::<3, usize>(&[3, 4]))
                .unwrap_or_else(|| {
                    let [batch, _, rank, value_dim] = hub.shape().dims::<4>();
                    Tensor::<B, 3>::zeros([batch, rank, value_dim], &hub.device())
                });
            let sum32 = u32
                .map(|u32| u32.sum_dims_squeeze::<3, usize>(&[3, 4]))
                .unwrap_or_else(|| {
                    let [batch, _, rank, value_dim] = hub.shape().dims::<4>();
                    Tensor::<B, 3>::zeros([batch, rank, value_dim], &hub.device())
                });
            let delta = sum8.add(sum32).unsqueeze_dim::<4>(1);
            return target_major_decay_add(hub, delta, decay);
        }

        let delta8 = u8.map(|u8| {
            let w8 = hub_w8.unwrap_or_else(|| {
                let [batch, _, _, height, width] = u8.shape().dims::<5>();
                Tensor::<B, 4>::ones([batch, hub_count, height, width], &u8.device())
                    .div_scalar(hub_count as f32)
            });
            self.trm_weighted_global_sum(u8, w8)
        });
        let delta32 = u32.map(|u32| {
            let w32 = hub_w32.unwrap_or_else(|| {
                let [batch, _, _, height, width] = u32.shape().dims::<5>();
                Tensor::<B, 4>::ones([batch, hub_count, height, width], &u32.device())
                    .div_scalar(hub_count as f32)
            });
            self.trm_weighted_global_sum(u32, w32)
        });
        let delta8 = delta8.unwrap_or_else(zero_delta);
        let delta32 = delta32.unwrap_or_else(zero_delta);
        target_major_decay_add(hub, delta8 + delta32, decay)
    }

    pub(super) fn trm_weighted_global_sum(&self, u: Tensor<B, 5>, w: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, rank, value_dim, height, width] = u.shape().dims::<5>();
        let [_, hubs, _, _] = w.shape().dims::<4>();
        if batch == 0 || hubs == 0 || rank == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), hubs.max(1), rank.max(1), value_dim.max(1)],
                &u.device(),
            );
        }
        let tokens = height * width;
        u.reshape([batch, 1, rank, value_dim, tokens])
            .mul(w.reshape([batch, hubs, 1, 1, tokens]))
            .sum_dims_squeeze::<4, usize>(&[4])
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_stage_aware_step_split_with_plan(
        &self,
        patch_query: Tensor<B, 4>,
        patch_query_for_coarse: Tensor<B, 4>,
        patch_query_for_global: Tensor<B, 4>,
        patch_value: Tensor<B, 4>,
        coarse_query: Tensor<B, 4>,
        coarse_query_for_global: Tensor<B, 4>,
        coarse_value: Tensor<B, 4>,
        patch_rho: Tensor<B, 5>,
        coarse_rho: Tensor<B, 5>,
        global_rho: Tensor<B, 4>,
        patch_hub_weights: Option<Tensor<B, 4>>,
        coarse_hub_weights: Option<Tensor<B, 4>>,
        patch_decay: Tensor<B, 1>,
        coarse_decay: Tensor<B, 1>,
        global_decay: Tensor<B, 1>,
        bank_mode: &VisionTrmGraphBankModeConfig,
        plan: &CompiledStageAwarePyramidLocalPlan<B>,
    ) -> StructuredPyramidRhoStepOutput<B> {
        stage_aware_profile_record(|state| {
            state.step_calls += 1;
        });
        let [patch_batch, patch_value_dim, patch_height, patch_width] =
            patch_value.shape().dims::<4>();
        let [coarse_batch, coarse_value_dim, coarse_height, coarse_width] =
            coarse_value.shape().dims::<4>();
        let patch_zero = || {
            Tensor::<B, 4>::zeros(
                [patch_batch, patch_value_dim, patch_height, patch_width],
                &patch_value.device(),
            )
        };
        let coarse_zero = || {
            Tensor::<B, 4>::zeros(
                [coarse_batch, coarse_value_dim, coarse_height, coarse_width],
                &coarse_value.device(),
            )
        };
        let coarse_rho_for_cross_scale = coarse_rho.clone();

        let (patch_local_context, next_patch_rho) = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_local_step_with_plan(
                patch_query.clone(),
                patch_value.clone(),
                patch_rho,
                plan.patch_shape(),
                patch_decay,
                plan.patch_plan(),
                plan.patch_neighborhood(),
                false,
                bank_mode.patch_local_read,
                bank_mode.patch_local_write,
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.patch_local_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };
        let (coarse_local_context, next_coarse_local_rho) = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_local_step_with_plan(
                coarse_query.clone(),
                coarse_value.clone(),
                coarse_rho,
                plan.coarse_shape(),
                coarse_decay.clone(),
                plan.coarse_plan(),
                plan.coarse_neighborhood(),
                true,
                bank_mode.coarse_local_read,
                bank_mode.coarse_local_write,
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.coarse_local_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };
        let patch_from_coarse_context = if bank_mode.patch_from_coarse_read {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_cross_scale_read(
                coarse_rho_for_cross_scale,
                patch_query_for_coarse.clone(),
                self.trm_graph.coarse_stride.max(1),
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.patch_from_coarse_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        } else {
            patch_zero()
        };
        let (patch_from_hub_context, coarse_from_hub_context) = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_hub_read_pair(
                global_rho.clone(),
                bank_mode
                    .patch_from_hub_read
                    .then(|| patch_query_for_global.clone()),
                bank_mode
                    .coarse_from_hub_read
                    .then(|| coarse_query_for_global.clone()),
                patch_hub_weights.clone(),
                coarse_hub_weights.clone(),
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.hub_read_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };
        let patch_from_hub_context = patch_from_hub_context.unwrap_or_else(patch_zero);
        let coarse_from_hub_context = coarse_from_hub_context.unwrap_or_else(coarse_zero);

        let next_coarse_rho = if bank_mode.patch_to_coarse_write {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let patch_to_coarse_update = self.pyramid_pool_outer(
                self.pyramid_outer_product(patch_query_for_coarse, patch_value.clone()),
                self.trm_graph.coarse_stride.max(1),
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.patch_to_coarse_ns += start.elapsed().as_nanos() as u64;
                });
            }
            Self::pyramid_rho_from_target_major(
                Self::pyramid_rho_to_target_major(next_coarse_local_rho.clone())
                    .add(Self::pyramid_rho_to_target_major(patch_to_coarse_update)),
                plan.coarse_shape().height,
                plan.coarse_shape().width,
            )
        } else {
            next_coarse_local_rho.clone()
        };
        let patch_to_global_update = bank_mode
            .patch_to_global_write
            .then(|| self.pyramid_outer_product(patch_query_for_global, patch_value));
        let coarse_to_global_update = bank_mode
            .coarse_to_global_write
            .then(|| self.pyramid_outer_product(coarse_query_for_global, coarse_value));
        let next_hub_rho = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_update_hub(
                global_rho,
                patch_to_global_update,
                coarse_to_global_update,
                patch_hub_weights,
                coarse_hub_weights,
                self.trm_graph.hub_count.max(1),
                global_decay,
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.hub_update_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };

        StructuredPyramidRhoStepOutput {
            patch_local_context,
            coarse_local_context,
            patch_from_coarse_context,
            patch_from_hub_context,
            coarse_from_hub_context,
            next_patch_rho,
            next_coarse_rho,
            next_hub_rho,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_stage_aware_coarse_only_step_with_plan(
        &self,
        coarse_query: Tensor<B, 4>,
        coarse_query_for_global: Tensor<B, 4>,
        coarse_value: Tensor<B, 4>,
        coarse_rho: Tensor<B, 5>,
        global_rho: Tensor<B, 4>,
        coarse_hub_weights: Option<Tensor<B, 4>>,
        coarse_decay: Tensor<B, 1>,
        global_decay: Tensor<B, 1>,
        bank_mode: &VisionTrmGraphBankModeConfig,
        plan: &CompiledStageAwarePyramidLocalPlan<B>,
    ) -> StructuredPyramidCoarseOnlyStepOutput<B> {
        stage_aware_profile_record(|state| {
            state.coarse_only_step_calls += 1;
        });
        let [coarse_batch, coarse_value_dim, coarse_height, coarse_width] =
            coarse_value.shape().dims::<4>();
        let coarse_zero = || {
            Tensor::<B, 4>::zeros(
                [
                    coarse_batch.max(1),
                    coarse_value_dim.max(1),
                    coarse_height.max(1),
                    coarse_width.max(1),
                ],
                &coarse_value.device(),
            )
        };
        let (coarse_local_context, next_coarse_rho) = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_local_step_with_plan(
                coarse_query.clone(),
                coarse_value.clone(),
                coarse_rho,
                plan.coarse_shape(),
                coarse_decay,
                plan.coarse_plan(),
                plan.coarse_neighborhood(),
                true,
                bank_mode.coarse_local_read,
                bank_mode.coarse_local_write,
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.coarse_local_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };
        let coarse_from_hub_context = if bank_mode.coarse_from_hub_read {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_hub_read(
                global_rho.clone(),
                coarse_query_for_global.clone(),
                coarse_hub_weights.clone(),
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.hub_read_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        } else {
            coarse_zero()
        };
        let coarse_to_global_update = bank_mode
            .coarse_to_global_write
            .then(|| self.pyramid_outer_product(coarse_query_for_global, coarse_value));
        let next_hub_rho = {
            let start = stage_aware_host_profile_enabled().then(Instant::now);
            let output = self.pyramid_update_hub(
                global_rho,
                None,
                coarse_to_global_update,
                None,
                coarse_hub_weights,
                self.trm_graph.hub_count.max(1),
                global_decay,
            );
            if let Some(start) = start {
                stage_aware_profile_record(|state| {
                    state.hub_update_ns += start.elapsed().as_nanos() as u64;
                });
            }
            output
        };

        StructuredPyramidCoarseOnlyStepOutput {
            coarse_local_context,
            coarse_from_hub_context,
            next_coarse_rho,
            next_hub_rho,
        }
    }
}
