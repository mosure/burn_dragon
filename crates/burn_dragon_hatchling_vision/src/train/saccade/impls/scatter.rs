#![allow(clippy::too_many_arguments)]

use crate::train::foveation::wgsl as foveation_wgsl;
use crate::train::prelude::*;
use crate::train::scatter::cubecl as scatter_cubecl;
use crate::train::scatter::wgsl as scatter_wgsl;

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn foveated_patch_sample_patched(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        let device = base_grid.device();
        let Some(first) = levels.first() else {
            let [batch, _, _, _] = base_grid.shape().dims::<4>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let grid_sample_max_bytes = limit_bytes_from_mb(self.config.grid_sample_max_mb);
        if batch == 0 || channels == 0 || patch_h == 0 || patch_w == 0 {
            return Tensor::<B, 4>::zeros(
                [
                    batch.max(1),
                    channels.max(1),
                    patch_h.max(1),
                    patch_w.max(1),
                ],
                &device,
            );
        }
        let full_patch_h = full_patch_h.max(1);
        #[cfg(feature = "integration_test")]
        if !should_fix_grid::<B>() {
            if channels < 3 {
                return Tensor::<B, 4>::zeros(
                    [
                        batch.max(1),
                        channels.max(1),
                        patch_h.max(1),
                        patch_w.max(1),
                    ],
                    &device,
                );
            }
            let data = first
                .image
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("foveation image data");
            let mean_x = center_x.clone().div_scalar(width as f32);
            let mean_y = center_y.clone().div_scalar(height as f32);
            let min_side = width.min(height).max(1) as f32;
            let sigma_norm = sigma_px.clone().div_scalar(min_side);
            let radius_norm = radius_px.clone().div_scalar(min_side);
            let mean_x_vals = mean_x
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("foveation mean_x");
            let mean_y_vals = mean_y
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("foveation mean_y");
            let sigma_vals = sigma_norm
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("foveation sigma");
            let radius_vals = radius_norm
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("foveation radius");
            let mode = match self.config.pyramid_mode {
                VisionPyramidMode::Stacked => crate::foveation::PyramidMode::Stacked,
                VisionPyramidMode::Laplacian => crate::foveation::PyramidMode::Laplacian,
            };
            let depth = levels.len().max(1);
            let channel_stride = patch_h * patch_w;
            let mut out = vec![0.0f32; batch * channels * channel_stride];
            for b in 0..batch {
                let Some(image) =
                    crate::foveation::image_from_nchw(&data, b, channels, height, width)
                else {
                    continue;
                };
                let cache = crate::foveation::build_pyramid_cache(image, depth, mode);
                let mean = [
                    *mean_x_vals.get(b).unwrap_or(&0.5),
                    *mean_y_vals.get(b).unwrap_or(&0.5),
                ];
                let sigma = *sigma_vals.get(b).unwrap_or(&0.1);
                let radius = *radius_vals.get(b).unwrap_or(&sigma);
                let patch = crate::foveation::render_foveated_patch_with_radius(
                    &cache,
                    mean,
                    sigma,
                    radius,
                    full_patch_h,
                    crate::foveation::FoveaWarpMode::Patched,
                );
                for y in 0..patch_h {
                    for x in 0..patch_w {
                        let src = (y * patch_w + x) * 3;
                        let dst = b * channels * channel_stride + y * patch_w + x;
                        if src + 2 < patch.len() && dst + 2 * channel_stride < out.len() {
                            out[dst] = patch[src];
                            out[dst + channel_stride] = patch[src + 1];
                            out[dst + 2 * channel_stride] = patch[src + 2];
                        }
                    }
                }
            }
            return Tensor::<B, 4>::from_data(
                TensorData::new(out, [batch, channels, patch_h, patch_w]),
                &device,
            );
        }
        if let Some(patch) = foveation_wgsl::try_foveated_patch_wgsl(
            levels,
            &base_grid,
            &center_x,
            &center_y,
            &sigma_px,
            &radius_px,
            &lod_sigma,
            laplacian_images,
            VisionFoveaWarpMode::Patched,
        ) {
            return patch;
        }

        let ux = base_grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let uy = base_grid.slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let half = full_patch_h as f32 * 0.5;
        let dx = ux.mul_scalar(half);
        let dy = uy.mul_scalar(half);
        let mean_x = center_x.clone().div_scalar(width as f32);
        let mean_y = center_y.clone().div_scalar(height as f32);
        let min_side = width.min(height).max(1) as f32;
        let max_level = levels.len().saturating_sub(1) as f32;
        let level_f = radius_px
            .clone()
            .div_scalar(min_side)
            .clamp_min(0.0)
            .clamp_max(1.0)
            .mul_scalar(max_level)
            .clamp_min(0.0)
            .clamp_max(max_level);
        let level0 = level_f.clone().floor();
        let level1 = level0.clone().add_scalar(1.0).clamp_max(max_level);
        let t = level_f
            .clone()
            .sub(level0.clone())
            .clamp_min(0.0)
            .clamp_max(1.0);
        let level0_map = level0.clone().repeat_dim(1, patch_h).repeat_dim(2, patch_w);
        let level1_map = level1.clone().repeat_dim(1, patch_h).repeat_dim(2, patch_w);
        let t_map = t.repeat_dim(1, patch_h).repeat_dim(2, patch_w);
        let inv_t_map = t_map.clone().mul_scalar(-1.0).add_scalar(1.0);

        let make_grid = |fx: &Tensor<B, 3>, fy: &Tensor<B, 3>, level_w: usize, level_h: usize| {
            grid_from_fx_fy::<B>(fx, fy, level_w, level_h, &device)
        };

        let sample_laplacian = |start_idx: usize, fx: &Tensor<B, 3>, fy: &Tensor<B, 3>| {
            let laplacian = laplacian_images.expect("laplacian images");
            let [_, _, coarse_h, coarse_w] = laplacian.coarse.shape().dims::<4>();
            let coarse_grid = make_grid(fx, fy, coarse_w, coarse_h);
            let mut sample = grid_sample_2d_bilinear::<B>(
                laplacian.coarse.clone(),
                coarse_grid,
                grid_sample_max_bytes,
            );
            for (idx, residual) in laplacian.residuals.iter().enumerate() {
                if idx < start_idx {
                    continue;
                }
                let [_, _, res_h, res_w] = residual.shape().dims::<4>();
                let residual_grid = make_grid(fx, fy, res_w, res_h);
                sample = sample
                    + grid_sample_2d_bilinear::<B>(
                        residual.clone(),
                        residual_grid,
                        grid_sample_max_bytes,
                    );
            }
            sample
        };

        let mut color = Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
        for (level_idx, level) in levels.iter().enumerate() {
            let level_f = level_idx as f32;
            let [_, _, level_h, level_w] = level.image.shape().dims::<4>();
            let fx = mean_x.clone().add(dx.clone().div_scalar(level_w as f32));
            let fy = mean_y.clone().add(dy.clone().div_scalar(level_h as f32));
            let weight0 = level0_map.clone().equal_elem(level_f).float();
            let weight1 = level1_map.clone().equal_elem(level_f).float();
            let weight =
                weight0.clone().mul(inv_t_map.clone()) + weight1.clone().mul(t_map.clone());
            let sample = if laplacian_images.is_some() {
                sample_laplacian(level_idx, &fx, &fy)
            } else {
                let level_grid = make_grid(&fx, &fy, level_w, level_h);
                grid_sample_2d_bilinear::<B>(level.image.clone(), level_grid, grid_sample_max_bytes)
            };
            color = color + sample * weight.clone().unsqueeze_dim::<4>(1);
        }
        color
    }

    pub(crate) fn foveated_patch_sample_sequential(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        if matches!(self.config.fovea_warp_mode, VisionFoveaWarpMode::Patched) {
            return self.foveated_patch_sample_patched(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            );
        }
        let device = base_grid.device();
        let Some(first) = levels.first() else {
            let [batch, _, _, _] = base_grid.shape().dims::<4>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let grid_sample_max_bytes = limit_bytes_from_mb(self.config.grid_sample_max_mb);
        let full_half = full_patch_h as f32 * 0.5;
        let pixel_du = 1.0 / full_half.max(1.0);
        let subsamples_axis = self.config.fovea_subsamples.max(1);
        let jitter_samples = self
            .fovea_jitter(full_patch_h, subsamples_axis, &device)
            .sequential;
        let subsample_count = jitter_samples.len().max(1) as f32;
        let mut accum = Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
        let ux_base = base_grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let uy_base = base_grid.clone().slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let (_, dx_deriv_base) = self.foveated_warp(ux_base, sigma_px.clone(), radius_px.clone());
        let (_, dy_deriv_base) = self.foveated_warp(uy_base, sigma_px.clone(), radius_px.clone());
        let local_scale_base = dx_deriv_base
            .abs()
            .max_pair(dy_deriv_base.abs())
            .mul_scalar(pixel_du);
        let use_subsamples = local_scale_base
            .greater_elem(SACCADE_FOVEA_AA_THRESHOLD)
            .unsqueeze_dim::<4>(3)
            .repeat_dim(3, 2);

        for jitter in jitter_samples {
            let grid = base_grid.clone() + jitter;
            let grid = base_grid.clone().mask_where(use_subsamples.clone(), grid);

            let ux = grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
            let uy = grid.slice_dim(3, 1..2).squeeze_dim::<3>(3);
            let (dx, dx_deriv) = self.foveated_warp(ux, sigma_px.clone(), radius_px.clone());
            let (dy, dy_deriv) = self.foveated_warp(uy, sigma_px.clone(), radius_px.clone());
            let local_scale = dx_deriv.abs().max_pair(dy_deriv.abs()).mul_scalar(pixel_du);
            let img_x = center_x.clone() + dx.clone();
            let img_y = center_y.clone() + dy.clone();
            let fx = img_x.div_scalar(width as f32);
            let fy = img_y.div_scalar(height as f32);

            let sigma_sq = sigma_px.clone().powf_scalar(2.0);
            let dist = dx
                .clone()
                .powf_scalar(2.0)
                .div(sigma_sq.clone())
                .add(dy.clone().powf_scalar(2.0).div(sigma_sq))
                .sqrt();
            let zeros = Tensor::<B, 3>::zeros(dist.shape().dims::<3>(), &device);
            let dist_safe = dist.clone().clamp_min(1.0);
            let lod_dist = dist_safe
                .log()
                .div_scalar(SACCADE_LN_2)
                .mask_where(dist.lower_equal_elem(1.0), zeros.clone());
            let scale_safe = local_scale.clone().clamp_min(SACCADE_FOVEA_AA_THRESHOLD);
            let lod_scale = scale_safe
                .div_scalar(SACCADE_FOVEA_AA_THRESHOLD)
                .log()
                .div_scalar(SACCADE_LN_2)
                .mask_where(
                    local_scale.lower_equal_elem(SACCADE_FOVEA_AA_THRESHOLD),
                    zeros,
                );
            let max_level = levels.len().saturating_sub(1) as f32;
            let lod = lod_dist
                .max_pair(lod_scale)
                .clamp_min(0.0)
                .clamp_max(max_level);
            let patched = matches!(self.config.fovea_warp_mode, VisionFoveaWarpMode::Patched);
            let lod_round = lod
                .clone()
                .add_scalar(0.5)
                .floor()
                .clamp_min(0.0)
                .clamp_max(max_level);

            let make_grid =
                |fx: &Tensor<B, 3>, fy: &Tensor<B, 3>, level_w: usize, level_h: usize| {
                    grid_from_fx_fy::<B>(fx, fy, level_w, level_h, &device)
                };

            let laplacian_samples = if let Some(laplacian) = laplacian_images {
                let [_, _, coarse_h, coarse_w] = laplacian.coarse.shape().dims::<4>();
                let coarse_grid = make_grid(&fx, &fy, coarse_w, coarse_h);
                let coarse_sample = grid_sample_2d_bilinear::<B>(
                    laplacian.coarse.clone(),
                    coarse_grid,
                    grid_sample_max_bytes,
                );
                let mut residual_samples = Vec::with_capacity(laplacian.residuals.len());
                for residual in laplacian.residuals.iter() {
                    let [_, _, res_h, res_w] = residual.shape().dims::<4>();
                    let residual_grid = make_grid(&fx, &fy, res_w, res_h);
                    residual_samples.push(grid_sample_2d_bilinear::<B>(
                        residual.clone(),
                        residual_grid,
                        grid_sample_max_bytes,
                    ));
                }
                let mut recon_samples = Vec::with_capacity(levels.len());
                let mut current = coarse_sample;
                recon_samples.push(current.clone());
                for residual in residual_samples.iter().rev() {
                    current = current + residual.clone();
                    recon_samples.push(current.clone());
                }
                recon_samples.reverse();
                Some(recon_samples)
            } else {
                None
            };

            let mut color = Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
            let mut weight_sum = Tensor::<B, 3>::zeros([batch, patch_h, patch_w], &device);
            for (level_idx, level) in levels.iter().enumerate() {
                let level_f = level_idx as f32;
                let weight = if patched {
                    lod_round.clone().equal_elem(level_f).float()
                } else {
                    let diff = lod.clone().sub_scalar(level_f).div(lod_sigma.clone());
                    let weight = diff.powf_scalar(2.0).mul_scalar(-0.5).exp();
                    let window_mask = lod
                        .clone()
                        .sub_scalar(level_f)
                        .abs()
                        .lower_equal_elem(SACCADE_FOVEA_LOD_WINDOW);
                    Tensor::<B, 3>::zeros(weight.shape().dims::<3>(), &device)
                        .mask_where(window_mask, weight)
                };
                let sample = if let Some(laplacian_samples) = laplacian_samples.as_ref() {
                    laplacian_samples[level_idx].clone()
                } else {
                    let [_, _, level_h, level_w] = level.image.shape().dims::<4>();
                    let level_grid = make_grid(&fx, &fy, level_w, level_h);
                    grid_sample_2d_bilinear::<B>(
                        level.image.clone(),
                        level_grid,
                        grid_sample_max_bytes,
                    )
                };
                color = color + sample * weight.clone().unsqueeze_dim::<4>(1);
                weight_sum = weight_sum + weight;
            }
            let weight_sum = weight_sum.clamp_min(SACCADE_EPS);
            let sample = color / weight_sum.unsqueeze_dim::<4>(1);
            accum = accum + sample;
        }
        accum.mul_scalar(1.0 / subsample_count)
    }

    pub(crate) fn foveated_patch_sample_subpatch(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: Tensor<B, 4>,
        center_x: Tensor<B, 3>,
        center_y: Tensor<B, 3>,
        sigma_px: Tensor<B, 3>,
        radius_px: Tensor<B, 3>,
        lod_sigma: Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        subpatch_size: usize,
        full_patch_h: usize,
    ) -> Tensor<B, 4> {
        if matches!(self.config.fovea_warp_mode, VisionFoveaWarpMode::Patched)
            && foveation_wgsl::supports_backend::<B>()
        {
            return self.foveated_patch_sample_patched(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            );
        }
        #[cfg(feature = "integration_test")]
        if !should_fix_grid::<B>()
            && matches!(self.config.fovea_warp_mode, VisionFoveaWarpMode::Patched)
        {
            return self.foveated_patch_sample_patched(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            );
        }
        let device = base_grid.device();
        let [batch, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        if patch_h == 0 || patch_w == 0 {
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        }
        let mut tile = subpatch_size.min(patch_h).min(patch_w);
        while tile > 1 && (patch_h % tile != 0 || patch_w % tile != 0) {
            tile -= 1;
        }
        if tile >= patch_h && tile >= patch_w {
            return self.foveated_patch_sample_sequential(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            );
        }

        let mut rows = Vec::new();
        let mut y = 0;
        while y < patch_h {
            let y_end = (y + tile).min(patch_h);
            let mut row_tiles = Vec::new();
            let mut x = 0;
            while x < patch_w {
                let x_end = (x + tile).min(patch_w);
                let tile_grid = base_grid
                    .clone()
                    .slice_dim(1, y..y_end)
                    .slice_dim(2, x..x_end);
                let tile_patch = self.foveated_patch_sample_sequential(
                    levels,
                    tile_grid,
                    center_x.clone(),
                    center_y.clone(),
                    sigma_px.clone(),
                    radius_px.clone(),
                    lod_sigma.clone(),
                    laplacian_images,
                    full_patch_h,
                );
                row_tiles.push(tile_patch);
                x += tile;
            }
            let row = Tensor::cat(row_tiles, 3);
            rows.push(row);
            y += tile;
        }
        Tensor::cat(rows, 2)
    }

    pub(crate) fn lod_sigma_from_sigma(&self, sigma: Tensor<B, 2>) -> Tensor<B, 2> {
        let range = (SACCADE_SIGMA_MAX - SACCADE_SIGMA_MIN).max(SACCADE_EPS);
        let t = sigma
            .sub_scalar(SACCADE_SIGMA_MIN)
            .div_scalar(range)
            .clamp_min(0.0)
            .clamp_max(1.0);
        let log2 = t
            .mul_scalar(SACCADE_LOD_LOG2_MAX - SACCADE_LOD_LOG2_MIN)
            .add_scalar(SACCADE_LOD_LOG2_MIN);
        (log2.mul_scalar(SACCADE_LN_2)).exp()
    }

    pub(crate) fn mip_weighted_sum(
        &self,
        levels: &[Tensor<B, 3>],
        weights: &[Tensor<B, 3>],
    ) -> Tensor<B, 3> {
        let device = if let Some(weight) = weights.first() {
            weight.device()
        } else if let Some(level) = levels.first() {
            level.device()
        } else {
            return Tensor::<B, 3>::zeros([0, 0, 0], &B::Device::default());
        };
        let Some(weight) = weights.first() else {
            return Tensor::<B, 3>::zeros([0, 0, 0], &device);
        };
        let [batch, traj_tokens, _] = weight.shape().dims::<3>();
        let embed_dim = levels
            .first()
            .map(|tokens| tokens.shape().dims::<3>()[2])
            .unwrap_or(0);
        if batch == 0 || traj_tokens == 0 || embed_dim == 0 {
            return Tensor::<B, 3>::zeros([batch, traj_tokens.max(1), embed_dim.max(1)], &device);
        }
        let prefer_concat = {
            #[cfg(any(feature = "train", feature = "cli"))]
            {
                TypeId::of::<B::Device>() != TypeId::of::<WgpuDevice>()
            }
            #[cfg(not(any(feature = "train", feature = "cli")))]
            {
                true
            }
        };
        if !prefer_concat {
            let mut context =
                Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim.max(1)], &device);
            for (tokens, weights) in levels.iter().zip(weights.iter()) {
                context = context + self.weighted_sum_tokens(weights.clone(), tokens.clone());
            }
            return context;
        }
        let mut tokens_chunks = Vec::with_capacity(levels.len());
        let mut weight_chunks = Vec::with_capacity(weights.len());
        let mut concat_tokens = 0usize;
        for (tokens, weights_level) in levels.iter().zip(weights.iter()) {
            let [batch_t, in_tokens, dim_t] = tokens.shape().dims::<3>();
            let [batch_w, out_tokens, in_tokens_w] = weights_level.shape().dims::<3>();
            if batch_t == 0 || in_tokens == 0 || dim_t == 0 {
                continue;
            }
            if batch_w == 0 || out_tokens == 0 || in_tokens_w == 0 {
                continue;
            }
            if batch_t != batch
                || batch_w != batch
                || out_tokens != traj_tokens
                || dim_t != embed_dim
                || in_tokens_w != in_tokens
            {
                let mut context =
                    Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim.max(1)], &device);
                for (tokens, weights) in levels.iter().zip(weights.iter()) {
                    context = context + self.weighted_sum_tokens(weights.clone(), tokens.clone());
                }
                return context;
            }
            tokens_chunks.push(tokens.clone());
            weight_chunks.push(weights_level.clone());
            concat_tokens = concat_tokens.saturating_add(in_tokens);
        }
        if tokens_chunks.is_empty() {
            return Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim.max(1)], &device);
        }
        if tokens_chunks.len() == 1 {
            return self.weighted_sum_tokens(
                weight_chunks.pop().expect("single weight"),
                tokens_chunks.pop().expect("single tokens"),
            );
        }
        let mut use_concat = prefer_concat;
        if use_concat {
            let mip_concat_max_bytes = limit_bytes_from_mb(self.config.mip_concat_max_mb);
            let elem_bytes = std::mem::size_of::<B::FloatElem>() as u64;
            let matmul_bytes = (batch as u64)
                .saturating_mul(traj_tokens as u64)
                .saturating_mul(concat_tokens as u64)
                .saturating_mul(embed_dim as u64)
                .saturating_mul(elem_bytes);
            if matmul_bytes > mip_concat_max_bytes {
                use_concat = false;
            }
        }
        if !use_concat {
            let mut context =
                Tensor::<B, 3>::zeros([batch, traj_tokens, embed_dim.max(1)], &device);
            for (tokens, weights) in levels.iter().zip(weights.iter()) {
                context = context + self.weighted_sum_tokens(weights.clone(), tokens.clone());
            }
            return context;
        }
        let tokens_cat = Tensor::cat(tokens_chunks, 1);
        let weights_cat = Tensor::cat(weight_chunks, 2);
        self.weighted_sum_tokens(weights_cat, tokens_cat)
    }

    pub(crate) fn weighted_sum_tokens(
        &self,
        weights: Tensor<B, 3>,
        tokens: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let device = weights.device();
        let [batch, out_tokens, in_tokens] = weights.shape().dims::<3>();
        let dim = tokens.shape().dims::<3>()[2];
        if batch == 0 || out_tokens == 0 || in_tokens == 0 || dim == 0 {
            return Tensor::<B, 3>::zeros([batch, out_tokens.max(1), dim.max(1)], &device);
        }
        if in_tokens == 1 {
            return weights * tokens;
        }
        if out_tokens == 1 {
            let weights = weights.squeeze_dim::<2>(1).unsqueeze_dim::<3>(2);
            let weighted = tokens * weights;
            return weighted.sum_dim(1).reshape([batch, 1, dim]);
        }
        let scatter_mode = if matches!(
            self.config.fovea_scatter_mode,
            VisionFoveaScatterMode::Tensor
        ) {
            if scatter_cubecl::supports_backend::<B>() {
                VisionFoveaScatterMode::Cubecl
            } else if scatter_wgsl::supports_backend::<B>() {
                VisionFoveaScatterMode::Wgsl
            } else {
                VisionFoveaScatterMode::Tensor
            }
        } else {
            self.config.fovea_scatter_mode
        };
        match scatter_mode {
            VisionFoveaScatterMode::Cubecl => {
                if let Some(result) =
                    scatter_cubecl::try_weighted_sum_tokens_cubecl(&weights, &tokens)
                {
                    return result;
                }
            }
            VisionFoveaScatterMode::Wgsl => {
                if let Some(result) = scatter_wgsl::try_weighted_sum_tokens_wgsl(&weights, &tokens)
                {
                    return result;
                }
            }
            VisionFoveaScatterMode::Tensor => {}
        }
        weights.matmul(tokens)
    }

    #[cfg(test)]
    pub(crate) fn apply_mip_residual(
        &self,
        state_levels: &mut [Tensor<B, 3>],
        weights: &[Tensor<B, 3>],
        residual: Tensor<B, 3>,
    ) {
        for (state, weights) in state_levels.iter_mut().zip(weights.iter()) {
            let update =
                self.weighted_sum_tokens(weights.clone().swap_dims(1, 2), residual.clone());
            *state = state.clone() + update;
        }
    }
}
