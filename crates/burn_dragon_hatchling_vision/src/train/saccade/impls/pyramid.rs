use crate::train::prelude::*;

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn decode_saccade_params(
        &self,
        params: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mean = activation::sigmoid(params.clone().slice_dim(2, 0..2))
            .mul_scalar(1.0 - 2.0 * SACCADE_EPS)
            .add_scalar(SACCADE_EPS);
        let sigma = activation::sigmoid(params.slice_dim(2, 2..3))
            .mul_scalar(SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN)
            .add_scalar(SACCADE_SIGMA_MIN);
        (mean, sigma)
    }

    pub(crate) fn build_mip_pyramid(
        &self,
        images: Tensor<B, 4>,
        patch_size: usize,
    ) -> Vec<SaccadeMipLevel<B>> {
        let max_levels = self.config.mip_levels.max(1);
        let mut levels = Vec::new();
        let mut current = images;
        for level in 0..max_levels {
            let [_, _, height, width] = current.shape().dims::<4>();
            if height < patch_size || width < patch_size {
                break;
            }
            let patch = self.model.patch_embed_raw(current.clone());
            let grid = patch.grid;
            if grid.height == 0 || grid.width == 0 {
                break;
            }
            let tokens = self.project_pyramid_tokens(patch.tokens);
            levels.push(SaccadeMipLevel {
                tokens,
                grid,
                image: current.clone(),
            });

            if level + 1 == max_levels {
                break;
            }
            let next = downsample_image(current.clone());
            if let Some(next) = next {
                current = next;
            } else {
                break;
            }
        }
        levels
    }

    pub(crate) fn build_laplacian_images(
        &self,
        levels: &[SaccadeMipLevel<B>],
    ) -> Option<SaccadeLaplacianImages<B>> {
        if levels.len() < 2 {
            return None;
        }
        let grid_sample_max_bytes = limit_bytes_from_mb(self.config.grid_sample_max_mb);
        let device = levels
            .first()
            .map(|level| level.image.device())
            .unwrap_or_default();
        let mut residuals = Vec::with_capacity(levels.len().saturating_sub(1));
        for idx in 0..levels.len().saturating_sub(1) {
            let current = &levels[idx].image;
            let next = &levels[idx + 1].image;
            let [batch, _, level_h, level_w] = current.shape().dims::<4>();
            let [_, _, next_h, next_w] = next.shape().dims::<4>();
            if level_h == 0 || level_w == 0 {
                return None;
            }
            let grid = build_image_grid::<B>(level_h, level_w, next_h, next_w, &device);
            let grid = if grid.shape().dims::<4>()[0] == batch {
                grid
            } else {
                grid.repeat_dim(0, batch)
            };
            let upsampled =
                grid_sample_2d_bilinear::<B>(next.clone(), grid, grid_sample_max_bytes);
            residuals.push(current.clone() - upsampled);
        }
        let coarse = levels
            .last()
            .expect("levels not empty")
            .image
            .clone();
        Some(SaccadeLaplacianImages { residuals, coarse })
    }

    pub(crate) fn decompose_pyramid(
        &self,
        levels: &[Tensor<B, 3>],
        grids: &[PatchGrid],
    ) -> Vec<Tensor<B, 3>> {
        let mut residuals = Vec::with_capacity(levels.len());
        for idx in 0..levels.len() {
            if idx + 1 < levels.len() {
                let upsampled =
                    self.upsample_tokens(levels[idx + 1].clone(), grids[idx + 1], grids[idx]);
                residuals.push(levels[idx].clone() - upsampled);
            } else {
                residuals.push(levels[idx].clone());
            }
        }
        residuals
    }

    pub(crate) fn compose_pyramid(
        &self,
        residuals: &[Tensor<B, 3>],
        grids: &[PatchGrid],
    ) -> Vec<Tensor<B, 3>> {
        if residuals.is_empty() {
            return Vec::new();
        }
        let mut composed_rev = Vec::with_capacity(residuals.len());
        let mut current = residuals
            .last()
            .expect("residuals not empty")
            .clone();
        composed_rev.push(current.clone());
        if residuals.len() > 1 {
            for idx in (0..residuals.len() - 1).rev() {
                let upsampled = self.upsample_tokens(current, grids[idx + 1], grids[idx]);
                current = residuals[idx].clone() + upsampled;
                composed_rev.push(current.clone());
            }
        }
        composed_rev.reverse();
        composed_rev
    }

    pub(crate) fn level_coords_cached(&self, grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
        self.level_coords_cache.get_or_build(grid, device)
    }

    pub(crate) fn upsample_weights_cached(
        &self,
        from: PatchGrid,
        to: PatchGrid,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        self.upsample_weights_cache.get_or_build(from, to, device)
    }

    pub(crate) fn upsample_tokens(
        &self,
        tokens: Tensor<B, 3>,
        from: PatchGrid,
        to: PatchGrid,
    ) -> Tensor<B, 3> {
        let [batch, tokens_len, dim] = tokens.shape().dims::<3>();
        if from.height == 0 || from.width == 0 || to.height == 0 || to.width == 0 {
            return Tensor::<B, 3>::zeros([batch, to.num_patches().max(1), dim], &tokens.device());
        }
        if tokens_len == 0 || tokens_len != from.num_patches() {
            return Tensor::<B, 3>::zeros([batch, to.num_patches().max(1), dim], &tokens.device());
        }
        if from.height == to.height && from.width == to.width {
            return tokens;
        }
        let device = tokens.device();
        let weights = self.upsample_weights_cached(from, to, &device);
        let weights = weights.unsqueeze_dim::<3>(0).repeat_dim(0, batch);
        self.weighted_sum_tokens(weights, tokens)
    }

    pub(crate) fn pyramid_lejepa_loss(
        &self,
        levels: &[Tensor<B, 3>],
    ) -> (Tensor<B, 1>, Tensor<B, 1>) {
        let Some(first) = levels.first() else {
            let device = self.trajectory_token.val().device();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero);
        };
        let device = first.device();
        let mut views = Vec::with_capacity(levels.len());
        for level in levels {
            let [batch, tokens, dim] = level.shape().dims::<3>();
            if batch == 0 || tokens == 0 || dim == 0 {
                continue;
            }
            let pooled = level.clone().mean_dim(1).reshape([batch, 1, dim]);
            let proj = self.model.project_tokens(pooled);
            let proj_dim = proj.shape().dims::<3>()[2];
            views.push(proj.reshape([1, batch, proj_dim]));
        }
        if views.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero);
        }
        let proj = if views.len() == 1 {
            views.pop().expect("single view")
        } else {
            Tensor::cat(views, 0)
        };
        let inv = lejepa_invariance_loss(proj.clone());
        let sigreg = lejepa_sigreg_loss_params(
            proj,
            self.config.loss.lejepa.sigreg_knots,
            self.config.loss.lejepa.sigreg_t_max,
            self.config.loss.lejepa.sigreg_proj_dim,
        );
        (inv, sigreg)
    }

    pub(crate) fn mip_gaussian_weights(
        &self,
        levels: &[SaccadeMipLevel<B>],
        mean: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
    ) -> Vec<Tensor<B, 3>> {
        let [batch, traj_tokens, _] = mean.shape().dims::<3>();
        let device = mean.device();
        let level_count = levels.len();
        if level_count == 0 {
            return Vec::new();
        }

        let mean_flat = mean.reshape([batch * traj_tokens, 2]);
        let sigma_scaled = sigma
            .mul_scalar(self.config.fovea_radius_scale)
            .clamp_min(SACCADE_EPS);
        let sigma_flat = sigma_scaled.reshape([batch * traj_tokens, 1]);
        let lod_sigma = self
            .lod_sigma_from_sigma(sigma_flat.clone())
            .clamp_min(SACCADE_EPS);
        let max_level = level_count.saturating_sub(1) as f32;
        let patched = matches!(self.config.fovea_warp_mode, VisionFoveaWarpMode::Patched);

        let mut total = Tensor::<B, 2>::zeros([batch * traj_tokens, 1], &device);
        let mut raw_weights = Vec::with_capacity(level_count);
        for (level_idx, level) in levels.iter().enumerate() {
            let coords = self.level_coords_cached(level.grid, &device);
            let tokens_len = level.tokens.shape().dims::<3>()[1].max(1);
            let coords = coords.reshape([1, tokens_len, 2]);
            let diff = mean_flat.clone().unsqueeze_dim::<3>(1) - coords;
            let dist2 = diff
                .powf_scalar(2.0)
                .sum_dim(2)
                .reshape([batch * traj_tokens, tokens_len]);
            let sigma2 = sigma_flat
                .clone()
                .powf_scalar(2.0)
                .add_scalar(SACCADE_EPS)
                .repeat_dim(1, tokens_len);
            let spatial = (dist2.clone() / sigma2.mul_scalar(2.0))
                .mul_scalar(-1.0)
                .exp();

            let sigma_tokens = sigma_flat.clone().repeat_dim(1, tokens_len);
            let dist = dist2.add_scalar(SACCADE_EPS).sqrt();
            let dist_norm = dist / sigma_tokens;
            let lod_center = dist_norm
                .clamp_min(1.0)
                .log()
                .div_scalar(SACCADE_LN_2)
                .clamp_max(max_level);
            let lod_weight = if patched {
                let lod_round = lod_center
                    .clone()
                    .add_scalar(0.5)
                    .floor()
                    .clamp_min(0.0)
                    .clamp_max(max_level);
                lod_round.equal_elem(level_idx as f32).float()
            } else {
                let lod_sigma = lod_sigma.clone().repeat_dim(1, tokens_len);
                let diff = lod_center.clone().sub_scalar(level_idx as f32) / lod_sigma;
                let lod_weight = diff.powf_scalar(2.0).mul_scalar(-0.5).exp();
                let lod_window = lod_center
                    .sub_scalar(level_idx as f32)
                    .abs()
                    .lower_equal_elem(SACCADE_FOVEA_LOD_WINDOW);
                Tensor::<B, 2>::zeros(lod_weight.shape().dims::<2>(), &device)
                    .mask_where(lod_window, lod_weight)
            };

            let weights = spatial * lod_weight;
            let sum = weights.clone().sum_dim(1).reshape([batch * traj_tokens, 1]);
            total = total + sum;
            raw_weights.push(weights);
        }

        let total = total.add_scalar(SACCADE_EPS);
        let mut weights_out = Vec::with_capacity(level_count);
        for (level, weights) in levels.iter().zip(raw_weights.into_iter()) {
            let tokens_len = level.tokens.shape().dims::<3>()[1].max(1);
            let denom = total.clone().repeat_dim(1, tokens_len);
            let weights = (weights / denom).reshape([batch, traj_tokens, tokens_len]);
            weights_out.push(weights);
        }
        weights_out
    }

}

