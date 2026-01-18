use crate::train::prelude::*;

pub(crate) struct CollectedViews<B: BackendTrait> {
    pub(crate) global: Vec<Tensor<B, 4>>,
    pub(crate) local: Vec<Tensor<B, 4>>,
    pub(crate) all: Vec<Tensor<B, 4>>,
}

impl<B: BackendTrait> CollectedViews<B> {
    pub(crate) fn artifact_views(&self) -> Vec<Tensor<B, 4>> {
        if !self.global.is_empty() {
            self.global.clone()
        } else if !self.all.is_empty() {
            self.all.clone()
        } else {
            Vec::new()
        }
    }
}

pub(crate) fn split_view_tensor<B: BackendTrait>(views: &Tensor<B, 5>) -> Vec<Tensor<B, 4>> {
    let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
    let mut out = Vec::with_capacity(view_count);
    for view_idx in 0..view_count {
        let view = views
            .clone()
            .slice_dim(1, view_idx..view_idx + 1)
            .reshape([batch, channels, height, width]);
        out.push(view);
    }
    out
}

pub(crate) fn collect_views<B: BackendTrait>(
    images: Tensor<B, 4>,
    target_images: Option<Tensor<B, 4>>,
    view_images: Option<Tensor<B, 5>>,
    global_view_images: Option<Tensor<B, 5>>,
    local_view_images: Option<Tensor<B, 5>>,
) -> CollectedViews<B> {
    let mut global = Vec::new();
    let mut local = Vec::new();
    let mut all = Vec::new();

    if let Some(global_views) = global_view_images {
        let views = split_view_tensor(&global_views);
        global.extend(views.clone());
        all.extend(views);
    }
    if let Some(local_views) = local_view_images {
        let views = split_view_tensor(&local_views);
        local.extend(views.clone());
        all.extend(views);
    }
    if all.is_empty() {
        if let Some(view_images) = view_images {
            let views = split_view_tensor(&view_images);
            global.extend(views.clone());
            all.extend(views);
        } else if let Some(target) = target_images {
            global.push(images.clone());
            global.push(target.clone());
            all.push(images);
            all.push(target);
        } else {
            global.push(images.clone());
            all.push(images);
        }
    }

    CollectedViews { global, local, all }
}

pub(crate) fn stack_views<B: BackendTrait>(views: &[Tensor<B, 4>]) -> Tensor<B, 4> {
    let view_count = views.len();
    if view_count == 1 {
        views[0].clone()
    } else {
        Tensor::cat(views.iter().cloned().collect(), 0)
    }
}

pub(crate) fn sample_patch_mask<B: BackendTrait>(
    device: &B::Device,
    batch: usize,
    tokens: usize,
    mask_ratio: f32,
    randomize_mask: bool,
) -> Tensor<B, 2> {
    if batch == 0 || tokens == 0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device);
    }
    let mask_ratio = mask_ratio.clamp(0.0, 1.0);
    if mask_ratio <= 0.0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device);
    }
    if mask_ratio >= 1.0 {
        return Tensor::<B, 2>::zeros([batch, tokens], device).add_scalar(1.0);
    }
    if randomize_mask {
        return Tensor::<B, 2>::random(
            [batch, tokens],
            TensorDistribution::Uniform(0.0, 1.0),
            device,
        )
        .lower_elem(mask_ratio)
        .float();
    }

    let total = batch * tokens;
    let mut rng = StdRng::seed_from_u64(0);
    let mut data = Vec::with_capacity(total);
    for _ in 0..total {
        let value = if rng.r#gen::<f32>() < mask_ratio {
            1.0
        } else {
            0.0
        };
        data.push(value);
    }
    Tensor::<B, 2>::from_data(TensorData::new(data, [batch, tokens]), device)
}

pub(crate) fn recon_psnr<B: BackendTrait>(mse: Tensor<B, 1>) -> Tensor<B, 1> {
    let denom = mse.add_scalar(LEJEPA_EPS);
    let scale = -10.0 / std::f32::consts::LN_10;
    denom.log().mul_scalar(scale)
}

pub(crate) fn lejepa_invariance_loss<B: BackendTrait>(proj: Tensor<B, 3>) -> Tensor<B, 1> {
    let device = proj.device();
    let [views, batch, dim] = proj.shape().dims::<3>();
    if views == 0 || batch == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let mean = proj.clone().mean_dim(0);
    (proj - mean).powf_scalar(2.0).mean()
}

pub(crate) fn normalize_columns<B: BackendTrait>(matrix: Tensor<B, 2>) -> Tensor<B, 2> {
    let norm = matrix
        .clone()
        .powf_scalar(2.0)
        .sum_dim(0)
        .sqrt()
        .add_scalar(LEJEPA_EPS);
    matrix / norm
}

pub(crate) fn lejepa_sigreg_loss<B: BackendTrait>(
    proj: Tensor<B, 3>,
    config: &VisionLejepaLossConfig,
) -> Tensor<B, 1> {
    lejepa_sigreg_loss_params(
        proj,
        config.sigreg_knots,
        config.sigreg_t_max,
        config.sigreg_proj_dim,
    )
}

pub(crate) fn lejepa_sigreg_loss_params<B: BackendTrait>(
    proj: Tensor<B, 3>,
    sigreg_knots: usize,
    sigreg_t_max: f32,
    sigreg_proj_dim: usize,
) -> Tensor<B, 1> {
    let device = proj.device();
    let [views, batch, dim] = proj.shape().dims::<3>();
    if views == 0 || batch == 0 || dim == 0 {
        return Tensor::<B, 1>::zeros([1], &device);
    }

    let knots = sigreg_knots.max(2);
    let t_max = sigreg_t_max.max(LEJEPA_EPS);
    let dt = t_max / (knots as f32 - 1.0);
    let mut t = Vec::with_capacity(knots);
    let mut phi = Vec::with_capacity(knots);
    let mut weights = Vec::with_capacity(knots);
    for i in 0..knots {
        let value = i as f32 * dt;
        let window = (-0.5 * value * value).exp();
        let weight = if i == 0 || i + 1 == knots {
            dt
        } else {
            2.0 * dt
        };
        t.push(value);
        phi.push(window);
        weights.push(weight * window);
    }

    let t =
        Tensor::<B, 1>::from_data(TensorData::new(t, [knots]), &device).reshape([1, 1, 1, knots]);
    let phi =
        Tensor::<B, 1>::from_data(TensorData::new(phi, [knots]), &device).reshape([1, 1, knots]);
    let weights = Tensor::<B, 1>::from_data(TensorData::new(weights, [knots]), &device)
        .reshape([1, 1, knots]);

    let sketch_dim = sigreg_proj_dim.max(1);
    let a = Tensor::<B, 2>::random(
        [dim, sketch_dim],
        TensorDistribution::Normal(0.0, 1.0),
        &device,
    );
    let a = normalize_columns(a);

    let proj_flat = proj.reshape([views * batch, dim]);
    let sketched = proj_flat.matmul(a).reshape([views, batch, sketch_dim]);
    let x_t = sketched.unsqueeze_dim::<4>(3).mul(t);
    let cos = x_t
        .clone()
        .cos()
        .mean_dim(1)
        .reshape([views, sketch_dim, knots]);
    let sin = x_t.sin().mean_dim(1).reshape([views, sketch_dim, knots]);
    let phi = phi.repeat_dim(0, views).repeat_dim(1, sketch_dim);
    let weights = weights.repeat_dim(0, views).repeat_dim(1, sketch_dim);
    let err = (cos - phi).powf_scalar(2.0) + sin.powf_scalar(2.0);
    let statistic = err.mul(weights).sum_dim(2).mul_scalar(batch as f32);
    statistic.mean()
}

pub(crate) fn normalize_artifact_legend(
    legend: Option<Vec<String>>,
    view_count: usize,
) -> Option<Vec<String>> {
    if view_count == 0 {
        return None;
    }
    let mut legend =
        legend.unwrap_or_else(|| (0..view_count).map(|idx| format!("view_{idx}")).collect());
    if legend.len() < view_count {
        for idx in legend.len()..view_count {
            legend.push(format!("view_{idx}"));
        }
    } else if legend.len() > view_count {
        legend.truncate(view_count);
    }
    Some(legend)
}

pub(crate) fn build_lejepa_artifacts<B: BackendTrait>(
    config: &VisionLejepaConfig,
    views: &[Tensor<B, 4>],
    frames: Option<Tensor<B, 5>>,
    first_patch: Option<Tensor<B, 3>>,
    probe_logits: Option<Tensor<B, 2>>,
    labels: Option<Tensor<B, 1, Int>>,
    legend: Option<Vec<String>>,
) -> Option<VisionArtifactInput<B>> {
    let max_images = config.artifact_max_images;
    let max_views = config.artifact_max_views;
    if max_images == 0 || max_views == 0 || views.is_empty() {
        return None;
    }
    let [batch, _, _, _] = views[0].shape().dims::<4>();
    let image_count = max_images.min(batch);
    if image_count == 0 {
        return None;
    }
    let view_count = max_views.min(views.len()).max(1);
    let legend = normalize_artifact_legend(legend, view_count);
    let mut stacked = Vec::with_capacity(view_count);
    for view in views.iter().take(view_count) {
        let view = view.clone().slice_dim(0, 0..image_count);
        stacked.push(view.unsqueeze_dim::<5>(1));
    }
    let views_tensor = Tensor::cat(stacked, 1);

    let patch_norms = first_patch.and_then(|patch| {
        let [batch, tokens, _] = patch.shape().dims::<3>();
        if batch == 0 || tokens == 0 {
            return None;
        }
        let grid = (tokens as f64).sqrt().round() as usize;
        if grid * grid != tokens {
            return None;
        }
        let norms = patch.powf_scalar(2.0).sum_dim(2).sqrt();
        let norms = norms.reshape([batch, grid, grid]);
        Some(norms.slice_dim(0, 0..image_count))
    });

    let probe_logits = probe_logits.map(|logits| logits.slice_dim(0, 0..image_count));
    let labels = labels.map(|labels| labels.slice_dim(0, 0..image_count));
    let frames = frames.map(|frames| frames.slice_dim(0, 0..image_count));

    Some(VisionArtifactInput {
        views: Some(views_tensor),
        frames,
        patch_norms,
        probe_logits,
        labels,
        legend,
    })
}

pub(crate) fn select_trajectory_indices(total: usize, max: usize) -> Vec<usize> {
    if total == 0 || max == 0 {
        return Vec::new();
    }
    if max >= total {
        return (0..total).collect();
    }
    if max == 1 {
        return vec![total - 1];
    }
    let last = (total - 1) as f32;
    let denom = (max - 1) as f32;
    let mut indices = Vec::with_capacity(max);
    for i in 0..max {
        let idx = ((i as f32) * last / denom).round() as usize;
        indices.push(idx.min(total - 1));
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}
