use super::*;
use burn_dragon_core::{
    target_major_decay_add, target_major_identity_read, target_major_outer_product,
};

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
                let left = left
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([left_batch, left_tokens, left_dim]);
                let right = right
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([right_batch, right_tokens, right_dim]);
                let tokens = Tensor::cat(
                    vec![left, right],
                    1,
                );
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
        let left = left
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([left_batch, left_tokens, left_dim]);
        let right = right
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([right_batch, right_tokens, right_dim]);
        let tokens = Tensor::cat(
            vec![left, right],
            1,
        );
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
        Self::pyramid_tokens_from_target_major(read, height, width).reshape([
            batch, value_dim, height, width,
        ])
    }

    pub(super) fn pyramid_local_read(
        &self,
        memory: Tensor<B, 5>,
        query: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, _, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &memory.device(),
            );
        }
        let mut acc = Tensor::<B, 4>::zeros([batch, value_dim, height, width], &memory.device());
        let radius = self.trm_graph.local_radius.max(1) as isize;
        let allow_diagonals = self.trm_graph.local_diagonals;
        if self.trm_graph.local_self {
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
        pyramid_value_norm: &LayerNorm<B>,
    ) -> Tensor<B, 4> {
        let [batch, dense_dim, height, width] = state.shape().dims::<4>();
        let [x_batch, rank, x_height, x_width] = x.shape().dims::<4>();
        let [msg_batch, value_dim, msg_height, msg_width] = msg.shape().dims::<4>();
        assert_eq!(x_batch, batch, "pyramid x batch must match state batch");
        assert_eq!(msg_batch, batch, "pyramid msg batch must match state batch");
        assert_eq!(x_height, height, "pyramid x height must match state height");
        assert_eq!(x_width, width, "pyramid x width must match state width");
        assert_eq!(msg_height, height, "pyramid msg height must match state height");
        assert_eq!(msg_width, width, "pyramid msg width must match state width");

        let x_tokens = x
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch, height * width, rank]);
        let msg_tokens = msg
            .swap_dims(1, 3)
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
        pyramid_value_norm: &LayerNorm<B>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        let [batch, dense_dim, patch_height, patch_width] = patch_state.shape().dims::<4>();
        let [coarse_batch, coarse_dense_dim, coarse_height, coarse_width] =
            coarse_state.shape().dims::<4>();
        let [patch_x_batch, rank, patch_x_height, patch_x_width] = patch_x.shape().dims::<4>();
        let [coarse_x_batch, coarse_rank, coarse_x_height, coarse_x_width] =
            coarse_x.shape().dims::<4>();
        let [patch_msg_batch, value_dim, patch_msg_height, patch_msg_width] =
            patch_msg.shape().dims::<4>();
        let [coarse_msg_batch, coarse_value_dim, coarse_msg_height, coarse_msg_width] =
            coarse_msg.shape().dims::<4>();
        assert_eq!(coarse_batch, batch, "coarse batch must match patch batch");
        assert_eq!(
            coarse_dense_dim, dense_dim,
            "coarse dense dim must match patch dense dim"
        );
        assert_eq!(patch_x_batch, batch, "patch x batch must match state batch");
        assert_eq!(coarse_x_batch, batch, "coarse x batch must match state batch");
        assert_eq!(patch_msg_batch, batch, "patch msg batch must match state batch");
        assert_eq!(coarse_msg_batch, batch, "coarse msg batch must match state batch");
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
                coarse_msg
                    .swap_dims(1, 3)
                    .swap_dims(1, 2)
                    .reshape([batch, coarse_tokens, value_dim]),
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

        self.apply_embed_norm_spatial_pair(
            patch_state + patch_delta,
            coarse_state + coarse_delta,
        )
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
        let w8 = self.pyramid_hub_weights_single(h8, hub_count, hub_gate);
        let w32 = self.pyramid_hub_weights_single(h32, hub_count, hub_gate);
        (Some(w8), Some(w32))
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
            let weights = activation::relu(weights);
            let denom = weights.clone().sum_dim(1).add_scalar(ROW_NORM_EPS);
            weights / denom
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
        // Avoid the 6D broadcasted hub/query multiply here. On WGPU that fused path can
        // materialize an oversized intermediate during training. A per-hub 5D multiply/sum
        // keeps the working set bounded while preserving the same contraction.
        let query_exp = query.unsqueeze_dim::<5>(2);
        let mut reduced = Tensor::<B, 4>::zeros([batch, value_dim, height, width], &hub.device());

        for hub_idx in 0..hubs {
            let hub_slice = hub
                .clone()
                .slice_dim(1, hub_idx..hub_idx + 1)
                .reshape([batch, rank, value_dim, 1, 1]);
            let mut msg = hub_slice
                .mul(query_exp.clone())
                .sum_dims_squeeze::<4, usize>(&[1]);
            if let Some(ref weights) = weights {
                let hub_weight = weights
                    .clone()
                    .slice_dim(1, hub_idx..hub_idx + 1)
                    .reshape([batch, 1, height, width]);
                msg = msg.mul(hub_weight);
            }
            reduced = reduced.add(msg);
        }

        if weights.is_none() && hubs > 1 {
            reduced.div_scalar(hubs as f32)
        } else {
            reduced
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn pyramid_update_hub(
        &self,
        hub: Tensor<B, 4>,
        u8: Tensor<B, 5>,
        u32: Tensor<B, 5>,
        hub_w8: Option<Tensor<B, 4>>,
        hub_w32: Option<Tensor<B, 4>>,
        hub_count: usize,
        decay: Tensor<B, 1>,
    ) -> Tensor<B, 4> {
        if hub_count <= 1 {
            let sum8 = u8.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let sum32 = u32.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let delta = (sum8 + sum32).unsqueeze_dim::<4>(1);
            return target_major_decay_add(hub, delta, decay);
        }

        let w8 = hub_w8.unwrap_or_else(|| {
            let [batch, _, _, height, width] = u8.shape().dims::<5>();
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &u8.device())
                .div_scalar(hub_count as f32)
        });
        let w32 = hub_w32.unwrap_or_else(|| {
            let [batch, _, _, height, width] = u32.shape().dims::<5>();
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &u32.device())
                .div_scalar(hub_count as f32)
        });

        let delta8 = self.trm_weighted_global_sum(u8, w8);
        let delta32 = self.trm_weighted_global_sum(u32, w32);
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
        let mut outputs = Vec::with_capacity(hubs);
        for hub_idx in 0..hubs {
            let hub_weight = w
                .clone()
                .slice_dim(1, hub_idx..hub_idx + 1)
                .reshape([batch, 1, 1, height, width]);
            let weighted = u
                .clone()
                .mul(hub_weight)
                .sum_dims_squeeze::<3, usize>(&[3, 4])
                .unsqueeze_dim::<4>(1);
            outputs.push(weighted);
        }

        Tensor::cat(outputs, 1)
    }
}
