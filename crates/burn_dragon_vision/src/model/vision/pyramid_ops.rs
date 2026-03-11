use super::*;

impl<B: Backend> VisionDragon<B> {
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

    pub(super) fn apply_value_norm_spatial(
        &self,
        norm: &LayerNorm<B>,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return input;
        }
        let flat = input.swap_dims(1, 3).swap_dims(1, 2);
        let flat = norm.forward(flat);
        flat.swap_dims(1, 2).swap_dims(1, 3)
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
        let query = query.unsqueeze_dim::<5>(2);
        memory.mul(query).sum_dims_squeeze::<4, usize>(&[1])
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
        let x = x.unsqueeze_dim::<5>(2);
        let v = v.unsqueeze_dim::<5>(1);
        x.mul(v)
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
        let msg = self.apply_value_norm_spatial(pyramid_value_norm, msg);
        let y = activation::relu(self.project_spatial(msg, pyramid_y_gate_proj));
        let u = y.mul(x);
        let delta = self.project_spatial(u, pyramid_delta_proj);
        let next = state + delta;
        self.apply_embed_norm_spatial(next)
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
        let decayed_hub = self.pyramid_apply_decay_4d(hub, decay);
        if hub_count <= 1 {
            let sum8 = u8.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let sum32 = u32.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let delta = (sum8 + sum32).unsqueeze_dim::<4>(1);
            return decayed_hub.add(delta);
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
        let delta = delta8 + delta32;
        decayed_hub.add(delta)
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
