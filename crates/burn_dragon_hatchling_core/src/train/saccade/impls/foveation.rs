use crate::train::prelude::*;
use crate::train::foveation::cubecl as foveation_cubecl;
use crate::train::foveation::wgsl as foveation_wgsl;
use burn_autodiff::Autodiff;
use burn_wgpu::Wgpu;
use std::any::Any;
#[cfg(feature = "cuda")]
use burn_cuda::Cuda;

impl<B: BackendTrait> VisionSaccadeModel<B> {
    pub(crate) fn erfinv_approx(&self, values: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = values.device();
        let shape = values.shape().dims::<3>();
        let ones = Tensor::<B, 3>::ones(shape, &device);
        let sign = ones
            .clone()
            .mul_scalar(-1.0)
            .mask_where(values.clone().greater_equal_elem(0.0), ones.clone());
        let xx = values.clamp_min(-0.999).clamp_max(0.999);
        let ln = ones.clone().sub(xx.clone().powf_scalar(2.0)).log();
        let term = ln
            .clone()
            .mul_scalar(0.5)
            .add_scalar(2.0 / (SACCADE_FOVEA_PI * SACCADE_FOVEA_ERF_A));
        let inside = term
            .clone()
            .powf_scalar(2.0)
            .sub(ln.div_scalar(SACCADE_FOVEA_ERF_A))
            .clamp_min(0.0);
        let result = inside.sqrt().sub(term).clamp_min(0.0).sqrt();
        sign * result
    }

    pub(crate) fn erf_approx(&self, values: Tensor<B, 3>) -> Tensor<B, 3> {
        let device = values.device();
        let shape = values.shape().dims::<3>();
        let ones = Tensor::<B, 3>::ones(shape, &device);
        let sign = ones
            .clone()
            .mul_scalar(-1.0)
            .mask_where(values.clone().greater_equal_elem(0.0), ones.clone());
        let ax = values.abs();
        let t = ones.clone().div(ax.clone().mul_scalar(0.3275911).add_scalar(1.0));
        let a1 = 0.254829592;
        let a2 = -0.284496736;
        let a3 = 1.421413741;
        let a4 = -1.453152027;
        let a5 = 1.061405429;
        let poly = t
            .clone()
            .mul_scalar(a5)
            .add_scalar(a4)
            .mul(t.clone())
            .add_scalar(a3)
            .mul(t.clone())
            .add_scalar(a2)
            .mul(t.clone())
            .add_scalar(a1)
            .mul(t);
        let y = ones.clone().sub(poly.mul((ax.clone().mul(ax).mul_scalar(-1.0)).exp()));
        sign * y
    }

    pub(crate) fn foveated_warp(
        &self,
        u: Tensor<B, 3>,
        sigma: Tensor<B, 3>,
        radius: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let sigma_safe = sigma.clamp_min(SACCADE_EPS);
        let radius_safe = radius.clamp_min(SACCADE_EPS);
        let k = radius_safe / sigma_safe.clone();
        let u_max = self
            .erf_approx(k.div_scalar(SACCADE_FOVEA_SQRT2))
            .clamp_max(0.999);
        let u_scaled = u.clamp_min(-1.0).clamp_max(1.0) * u_max.clone();
        let erf_inv = self.erfinv_approx(u_scaled);
        let offset = erf_inv.clone().mul(sigma_safe.clone()).mul_scalar(SACCADE_FOVEA_SQRT2);
        let deriv = sigma_safe
            .mul_scalar(SACCADE_FOVEA_SQRT2)
            .mul(u_max)
            .mul_scalar(SACCADE_FOVEA_SQRT_PI_OVER_2)
            .mul(erf_inv.clone().powf_scalar(2.0).exp());
        (offset, deriv)
    }

    // Foveated patch sampling on the image pyramid (GPU tensor path).
    pub(crate) fn foveated_patch_image(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    ) -> Tensor<B, 4> {
        let radius = sigma
            .clone()
            .mul_scalar(self.config.fovea_radius_scale)
            .clamp_min(SACCADE_EPS)
            .clamp_max(1.0 - SACCADE_EPS);
        self.foveated_patch_image_with_radius(
            levels,
            base_grid,
            mean,
            sigma.clone(),
            radius,
            laplacian_images,
        )
    }

    pub(crate) fn foveated_patch_image_with_radius(
        &self,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        mean: Tensor<B, 2>,
        sigma: Tensor<B, 2>,
        radius: Tensor<B, 2>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    ) -> Tensor<B, 4> {
        let device = mean.device();
        let Some(first) = levels.first() else {
            let [batch, _] = mean.shape().dims::<2>();
            return Tensor::<B, 4>::zeros([batch.max(1), 3, 1, 1], &device);
        };
        let [batch, channels, height, width] = first.image.shape().dims::<4>();
        let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
        let full_patch_h = patch_h;
        if batch == 0 || channels == 0 || patch_h == 0 || patch_w == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), channels.max(1), patch_h.max(1), patch_w.max(1)],
                &device,
            );
        }
        let base_grid = if base_grid.shape().dims::<4>()[0] == batch {
            base_grid.clone()
        } else {
            base_grid.clone().repeat_dim(0, batch)
        };

        let use_laplacian = matches!(self.config.pyramid_mode, VisionPyramidMode::Laplacian);
        let laplacian_fallback = if use_laplacian && laplacian_images.is_none() {
            self.build_laplacian_images(levels)
        } else {
            None
        };
        let laplacian_images = if use_laplacian {
            laplacian_images.or(laplacian_fallback.as_ref())
        } else {
            None
        };

        let min_side = width.min(height) as f32;
        let mean = mean.clamp_min(SACCADE_EPS).clamp_max(1.0 - SACCADE_EPS);
        let mean_x = mean.clone().slice_dim(1, 0..1).reshape([batch, 1, 1]);
        let mean_y = mean.slice_dim(1, 1..2).reshape([batch, 1, 1]);
        let radius_norm = radius.clamp_min(SACCADE_EPS).reshape([batch, 1, 1]);
        let sigma_norm = sigma
            .clamp_min(SACCADE_EPS)
            .reshape([batch, 1, 1])
            .min_pair(radius_norm.clone());
        let sigma_px = sigma_norm.clone().mul_scalar(min_side);
        let radius_px = radius_norm.clone().mul_scalar(min_side);
        let lod_sigma = self
            .lod_sigma_from_sigma(sigma_norm.clone().reshape([batch, 1]))
            .reshape([batch, 1, 1])
            .clamp_min(SACCADE_EPS);
        let center_x = mean_x.mul_scalar(width as f32);
        let center_y = mean_y.mul_scalar(height as f32);
        let warp_mode = self.config.fovea_warp_mode;
        let subsamples_match = self.config.fovea_subsamples == SACCADE_FOVEA_SUBSAMPLES;
        let sampling_mode = if subsamples_match
            && matches!(self.config.fovea_sampling_mode, VisionFoveaSamplingMode::Batched)
        {
            if foveation_wgsl::supports_backend::<B>() {
                VisionFoveaSamplingMode::Wgsl
            } else if matches!(warp_mode, VisionFoveaWarpMode::Warped)
                && foveation_cubecl::supports_backend::<B>()
            {
                VisionFoveaSamplingMode::Cubecl
            } else {
                VisionFoveaSamplingMode::Batched
            }
        } else if subsamples_match {
            self.config.fovea_sampling_mode
        } else {
            match self.config.fovea_sampling_mode {
                VisionFoveaSamplingMode::Wgsl | VisionFoveaSamplingMode::Cubecl => {
                    VisionFoveaSamplingMode::Batched
                }
                _ => self.config.fovea_sampling_mode,
            }
        };
        let grid_sample_max_bytes = limit_bytes_from_mb(self.config.grid_sample_max_mb);
        if B::ad_enabled() {
            if let Some(patch) = self.try_foveated_patch_custom_backward(
                sampling_mode,
                warp_mode,
                levels,
                &base_grid,
                &center_x,
                &center_y,
                &sigma_px,
                &radius_px,
                &lod_sigma,
                laplacian_images,
                full_patch_h,
            ) {
                return patch;
            }
        }
        match sampling_mode {
            VisionFoveaSamplingMode::Batched => {
                self.foveated_patch_sample_batched(
                    levels,
                    base_grid,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    laplacian_images,
                    full_patch_h,
                )
            }
            VisionFoveaSamplingMode::Sequential => self.foveated_patch_sample_sequential(
                levels,
                base_grid,
                center_x,
                center_y,
                sigma_px,
                radius_px,
                lod_sigma,
                laplacian_images,
                full_patch_h,
            ),
            VisionFoveaSamplingMode::Cubecl => {
                if B::ad_enabled() {
                    self.foveated_patch_sample_batched(
                        levels,
                        base_grid,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        laplacian_images,
                        full_patch_h,
                    )
                } else if matches!(warp_mode, VisionFoveaWarpMode::Warped) {
                    if let Some(patch) = foveation_cubecl::try_foveated_patch_cubecl(
                        levels,
                        &base_grid,
                        &center_x,
                        &center_y,
                        &sigma_px,
                        &radius_px,
                        &lod_sigma,
                        laplacian_images,
                        grid_sample_max_bytes,
                    ) {
                        patch
                    } else {
                        self.foveated_patch_sample_sequential(
                            levels,
                            base_grid,
                            center_x,
                            center_y,
                            sigma_px,
                            radius_px,
                            lod_sigma,
                            laplacian_images,
                            full_patch_h,
                        )
                    }
                } else {
                    self.foveated_patch_sample_sequential(
                        levels,
                        base_grid,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        laplacian_images,
                        full_patch_h,
                    )
                }
            }
            VisionFoveaSamplingMode::Wgsl => {
                if B::ad_enabled() {
                    self.foveated_patch_sample_batched(
                        levels,
                        base_grid,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        laplacian_images,
                        full_patch_h,
                    )
                } else {
                    if let Some(patch) = foveation_wgsl::try_foveated_patch_wgsl(
                        levels,
                        &base_grid,
                        &center_x,
                        &center_y,
                        &sigma_px,
                        &radius_px,
                        &lod_sigma,
                        laplacian_images,
                        warp_mode,
                    ) {
                        patch
                    } else {
                        self.foveated_patch_sample_sequential(
                            levels,
                            base_grid,
                            center_x,
                            center_y,
                            sigma_px,
                            radius_px,
                            lod_sigma,
                            laplacian_images,
                            full_patch_h,
                        )
                    }
                }
            }
            VisionFoveaSamplingMode::Subpatch => {
                let subpatch = self.config.fovea_subpatch_size;
                if subpatch == 0 {
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
                self.foveated_patch_sample_subpatch(
                    levels,
                    base_grid,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    laplacian_images,
                    subpatch,
                    full_patch_h,
                )
            }
        }
    }

    fn try_foveated_patch_custom_backward(
        &self,
        sampling_mode: VisionFoveaSamplingMode,
        warp_mode: VisionFoveaWarpMode,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        center_x: &Tensor<B, 3>,
        center_y: &Tensor<B, 3>,
        sigma_px: &Tensor<B, 3>,
        radius_px: &Tensor<B, 3>,
        lod_sigma: &Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Option<Tensor<B, 4>> {
        if !B::ad_enabled() {
            return None;
        }
        if self.config.fovea_subsamples != SACCADE_FOVEA_SUBSAMPLES {
            return None;
        }
        if let Some(result) = self.try_foveated_patch_custom_backward_for::<Autodiff<Wgpu<f32>>>(
            sampling_mode,
            warp_mode,
            levels,
            base_grid,
            center_x,
            center_y,
            sigma_px,
            radius_px,
            lod_sigma,
            laplacian_images,
            full_patch_h,
        ) {
            return Some(result);
        }
        #[cfg(feature = "cuda")]
        if let Some(result) = self.try_foveated_patch_custom_backward_for::<Autodiff<Cuda<f32>>>(
            sampling_mode,
            warp_mode,
            levels,
            base_grid,
            center_x,
            center_y,
            sigma_px,
            radius_px,
            lod_sigma,
            laplacian_images,
            full_patch_h,
        ) {
            return Some(result);
        }
        None
    }

    fn try_foveated_patch_custom_backward_for<AD>(
        &self,
        sampling_mode: VisionFoveaSamplingMode,
        warp_mode: VisionFoveaWarpMode,
        levels: &[SaccadeMipLevel<B>],
        base_grid: &Tensor<B, 4>,
        center_x: &Tensor<B, 3>,
        center_y: &Tensor<B, 3>,
        sigma_px: &Tensor<B, 3>,
        radius_px: &Tensor<B, 3>,
        lod_sigma: &Tensor<B, 3>,
        laplacian_images: Option<&SaccadeLaplacianImages<B>>,
        full_patch_h: usize,
    ) -> Option<Tensor<B, 4>>
    where
        AD: AutodiffBackend,
    {
        let model = (self as &dyn Any).downcast_ref::<VisionSaccadeModel<AD>>()?;
        let levels = downcast_levels::<B, AD>(levels)?;
        let base_grid = downcast_tensor::<B, AD, 4>(base_grid)?;
        let center_x = downcast_tensor::<B, AD, 3>(center_x)?;
        let center_y = downcast_tensor::<B, AD, 3>(center_y)?;
        let sigma_px = downcast_tensor::<B, AD, 3>(sigma_px)?;
        let radius_px = downcast_tensor::<B, AD, 3>(radius_px)?;
        let lod_sigma = downcast_tensor::<B, AD, 3>(lod_sigma)?;
        let laplacian_images = match laplacian_images {
            Some(laplacian) => Some(downcast_laplacian::<B, AD>(laplacian)?),
            None => None,
        };
        let patch = try_foveated_patch_custom_backward_autodiff(
            model,
            sampling_mode,
            warp_mode,
            &levels,
            &base_grid,
            &center_x,
            &center_y,
            &sigma_px,
            &radius_px,
            &lod_sigma,
            laplacian_images.as_ref(),
            full_patch_h,
        )?;
        let boxed: Box<dyn Any> = Box::new(patch);
        boxed
            .downcast::<Tensor<B, 4>>()
            .ok()
            .map(|boxed| *boxed)
    }

    pub(crate) fn foveated_patch_sample_batched(
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
        let subsamples = subsamples_axis * subsamples_axis;
        let jitter = self
            .fovea_jitter(full_patch_h, subsamples_axis, &device)
            .batched;
        let ux_base = base_grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let uy_base = base_grid.clone().slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let sigma_base = sigma_px
            .clone()
            .repeat_dim(1, patch_h)
            .repeat_dim(2, patch_w);
        let radius_base = radius_px
            .clone()
            .repeat_dim(1, patch_h)
            .repeat_dim(2, patch_w);
        let (_, dx_deriv_base) =
            self.foveated_warp(ux_base, sigma_base.clone(), radius_base.clone());
        let (_, dy_deriv_base) = self.foveated_warp(uy_base, sigma_base, radius_base);
        let local_scale_base = dx_deriv_base
            .abs()
            .max_pair(dy_deriv_base.abs())
            .mul_scalar(pixel_du);
        let use_subsamples = local_scale_base.greater_elem(SACCADE_FOVEA_AA_THRESHOLD);

        let base_grid = base_grid
            .unsqueeze_dim::<5>(0)
            .repeat_dim(0, subsamples);
        let base_grid_flat = base_grid
            .clone()
            .reshape([subsamples * batch, patch_h, patch_w, 2]);
        let grid = (base_grid + jitter).reshape([subsamples * batch, patch_h, patch_w, 2]);
        let use_subsamples = use_subsamples
            .unsqueeze_dim::<4>(3)
            .unsqueeze_dim::<5>(0)
            .repeat_dim(0, subsamples)
            .reshape([subsamples * batch, patch_h, patch_w, 1])
            .repeat_dim(3, 2);
        let grid = base_grid_flat.mask_where(use_subsamples, grid);

        let expand_3d = |tensor: Tensor<B, 3>| -> Tensor<B, 3> {
            let [_, h, w] = tensor.shape().dims::<3>();
            tensor
                .unsqueeze_dim::<4>(0)
                .repeat_dim(0, subsamples)
                .reshape([subsamples * batch, h, w])
        };
        let expand_4d = |tensor: Tensor<B, 4>| -> Tensor<B, 4> {
            let [_, ch, h, w] = tensor.shape().dims::<4>();
            tensor
                .unsqueeze_dim::<5>(0)
                .repeat_dim(0, subsamples)
                .reshape([subsamples * batch, ch, h, w])
        };

        let sigma_px = expand_3d(sigma_px);
        let radius_px = expand_3d(radius_px);
        let center_x = expand_3d(center_x);
        let center_y = expand_3d(center_y);
        let lod_sigma = expand_3d(lod_sigma);

        let ux = grid.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let uy = grid.slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let (dx, dx_deriv) = self.foveated_warp(ux, sigma_px.clone(), radius_px.clone());
        let (dy, dy_deriv) = self.foveated_warp(uy, sigma_px.clone(), radius_px.clone());
        let local_scale = dx_deriv.abs().max_pair(dy_deriv.abs()).mul_scalar(pixel_du);
        let img_x = center_x + dx.clone();
        let img_y = center_y + dy.clone();
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
        let scale_safe = local_scale
            .clone()
            .clamp_min(SACCADE_FOVEA_AA_THRESHOLD);
        let lod_scale = scale_safe
            .div_scalar(SACCADE_FOVEA_AA_THRESHOLD)
            .log()
            .div_scalar(SACCADE_LN_2)
            .mask_where(local_scale.lower_equal_elem(SACCADE_FOVEA_AA_THRESHOLD), zeros);
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

        let make_grid = |fx: &Tensor<B, 3>, fy: &Tensor<B, 3>, level_w: usize, level_h: usize| {
            grid_from_fx_fy::<B>(fx, fy, level_w, level_h, &device)
        };

        let laplacian_samples = if let Some(laplacian) = laplacian_images {
            let [_, _, coarse_h, coarse_w] = laplacian.coarse.shape().dims::<4>();
            let coarse_grid = make_grid(&fx, &fy, coarse_w, coarse_h);
            let coarse_sample = grid_sample_2d_bilinear::<B>(
                expand_4d(laplacian.coarse.clone()),
                coarse_grid,
                grid_sample_max_bytes,
            );
            let mut residual_samples = Vec::with_capacity(laplacian.residuals.len());
            for residual in laplacian.residuals.iter() {
                let [_, _, res_h, res_w] = residual.shape().dims::<4>();
                let residual_grid = make_grid(&fx, &fy, res_w, res_h);
                residual_samples.push(grid_sample_2d_bilinear::<B>(
                    expand_4d(residual.clone()),
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

        let mut color =
            Tensor::<B, 4>::zeros([subsamples * batch, channels, patch_h, patch_w], &device);
        let mut weight_sum =
            Tensor::<B, 3>::zeros([subsamples * batch, patch_h, patch_w], &device);
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
                    expand_4d(level.image.clone()),
                    level_grid,
                    grid_sample_max_bytes,
                )
            };
            color = color + sample * weight.clone().unsqueeze_dim::<4>(1);
            weight_sum = weight_sum + weight;
        }
        let weight_sum = weight_sum.clamp_min(SACCADE_EPS);
        let sample = color / weight_sum.unsqueeze_dim::<4>(1);
        let sample = sample.reshape([subsamples, batch, channels, patch_h, patch_w]);
        let mut accum =
            Tensor::<B, 4>::zeros([batch, channels, patch_h, patch_w], &device);
        for idx in 0..subsamples {
            let slice = sample
                .clone()
                .slice_dim(0, idx..idx + 1)
                .squeeze_dim::<4>(0);
            accum = accum + slice;
        }
        accum.mul_scalar(1.0 / subsamples as f32)
    }

}

fn downcast_tensor<B: BackendTrait, AD: BackendTrait, const D: usize>(
    tensor: &Tensor<B, D>,
) -> Option<Tensor<AD, D>> {
    let any = tensor as &dyn Any;
    any.downcast_ref::<Tensor<AD, D>>().cloned()
}

fn downcast_levels<B: BackendTrait, AD: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
) -> Option<Vec<SaccadeMipLevel<AD>>> {
    let mut out = Vec::with_capacity(levels.len());
    for level in levels {
        let tokens = downcast_tensor::<B, AD, 3>(&level.tokens)?;
        let image = downcast_tensor::<B, AD, 4>(&level.image)?;
        out.push(SaccadeMipLevel {
            tokens,
            grid: level.grid,
            image,
        });
    }
    Some(out)
}

fn downcast_laplacian<B: BackendTrait, AD: BackendTrait>(
    laplacian: &SaccadeLaplacianImages<B>,
) -> Option<SaccadeLaplacianImages<AD>> {
    let mut residuals = Vec::with_capacity(laplacian.residuals.len());
    for residual in &laplacian.residuals {
        residuals.push(downcast_tensor::<B, AD, 4>(residual)?);
    }
    let coarse = downcast_tensor::<B, AD, 4>(&laplacian.coarse)?;
    Some(SaccadeLaplacianImages { residuals, coarse })
}

fn try_foveated_patch_custom_backward_autodiff<AD: AutodiffBackend>(
    model: &VisionSaccadeModel<AD>,
    sampling_mode: VisionFoveaSamplingMode,
    warp_mode: VisionFoveaWarpMode,
    levels: &[SaccadeMipLevel<AD>],
    base_grid: &Tensor<AD, 4>,
    center_x: &Tensor<AD, 3>,
    center_y: &Tensor<AD, 3>,
    sigma_px: &Tensor<AD, 3>,
    radius_px: &Tensor<AD, 3>,
    lod_sigma: &Tensor<AD, 3>,
    laplacian_images: Option<&SaccadeLaplacianImages<AD>>,
    full_patch_h: usize,
) -> Option<Tensor<AD, 4>> {
    let inner_levels: Vec<SaccadeMipLevel<AD::InnerBackend>> = levels
        .iter()
        .map(|level| SaccadeMipLevel {
            tokens: level.tokens.clone().inner(),
            grid: level.grid,
            image: level.image.clone().inner(),
        })
        .collect();
    let inner_base_grid = base_grid.clone().inner();
    let inner_center_x = center_x.clone().inner();
    let inner_center_y = center_y.clone().inner();
    let inner_sigma_px = sigma_px.clone().inner();
    let inner_radius_px = radius_px.clone().inner();
    let inner_lod_sigma = lod_sigma.clone().inner();
    let inner_laplacian = laplacian_images.map(|laplacian| SaccadeLaplacianImages {
        residuals: laplacian
            .residuals
            .iter()
            .map(|residual| residual.clone().inner())
            .collect(),
        coarse: laplacian.coarse.clone().inner(),
    });
    let inner_laplacian_ref = inner_laplacian.as_ref();
    let grid_sample_max_bytes = limit_bytes_from_mb(model.config.grid_sample_max_mb);

    let fused_inner = match sampling_mode {
        VisionFoveaSamplingMode::Wgsl => foveation_wgsl::try_foveated_patch_wgsl::<
            AD::InnerBackend,
        >(
            &inner_levels,
            &inner_base_grid,
            &inner_center_x,
            &inner_center_y,
            &inner_sigma_px,
            &inner_radius_px,
            &inner_lod_sigma,
            inner_laplacian_ref,
            warp_mode,
        ),
        VisionFoveaSamplingMode::Cubecl => {
            if matches!(warp_mode, VisionFoveaWarpMode::Warped) {
                foveation_cubecl::try_foveated_patch_cubecl::<AD::InnerBackend>(
                    &inner_levels,
                    &inner_base_grid,
                    &inner_center_x,
                    &inner_center_y,
                    &inner_sigma_px,
                    &inner_radius_px,
                    &inner_lod_sigma,
                    inner_laplacian_ref,
                    grid_sample_max_bytes,
                )
            } else {
                None
            }
        }
        VisionFoveaSamplingMode::Batched => foveation_wgsl::try_foveated_patch_wgsl::<
            AD::InnerBackend,
        >(
            &inner_levels,
            &inner_base_grid,
            &inner_center_x,
            &inner_center_y,
            &inner_sigma_px,
            &inner_radius_px,
            &inner_lod_sigma,
            inner_laplacian_ref,
            warp_mode,
        )
        .or_else(|| {
            if matches!(warp_mode, VisionFoveaWarpMode::Warped) {
                foveation_cubecl::try_foveated_patch_cubecl::<AD::InnerBackend>(
                    &inner_levels,
                    &inner_base_grid,
                    &inner_center_x,
                    &inner_center_y,
                    &inner_sigma_px,
                    &inner_radius_px,
                    &inner_lod_sigma,
                    inner_laplacian_ref,
                    grid_sample_max_bytes,
                )
            } else {
                None
            }
        }),
        _ => None,
    }?;

    let fused = Tensor::<AD, 4>::from_inner(fused_inner);
    let surrogate = model.foveated_patch_sample_batched(
        levels,
        base_grid.clone(),
        center_x.clone(),
        center_y.clone(),
        sigma_px.clone(),
        radius_px.clone(),
        lod_sigma.clone(),
        laplacian_images,
        full_patch_h,
    );
    Some(fused + (surrogate.clone() - surrogate.detach()))
}

