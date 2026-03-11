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
        Tensor::cat(views.to_vec(), 0)
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

pub(crate) fn lejepa_teacher_invariance_loss<B: BackendTrait>(
    student_proj: Tensor<B, 3>,
    teacher_proj: Tensor<B, 3>,
) -> Tensor<B, 1> {
    let device = student_proj.device();
    let [student_views, student_batch, student_dim] = student_proj.shape().dims::<3>();
    let [teacher_views, teacher_batch, teacher_dim] = teacher_proj.shape().dims::<3>();
    if student_views == 0
        || student_batch == 0
        || student_dim == 0
        || teacher_views == 0
        || teacher_batch != student_batch
        || teacher_dim != student_dim
    {
        return Tensor::<B, 1>::zeros([1], &device);
    }
    let teacher_target = if teacher_views == 1 {
        teacher_proj
    } else {
        teacher_proj
            .mean_dim(0)
            .reshape([1, student_batch, student_dim])
    };
    let teacher_target = teacher_target.repeat_dim(0, student_views);
    (student_proj - teacher_target).powf_scalar(2.0).mean()
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

const PCA_HEATMAP_ITERS: usize = 8;
const PCA_HEATMAP_EPS: f32 = 1e-6;

fn normalize_vec(vec: &mut [f32]) -> f32 {
    let mut sum = 0.0f32;
    for value in vec.iter() {
        sum += value * value;
    }
    let norm = sum.sqrt();
    if norm > 0.0 {
        let inv = 1.0 / norm;
        for value in vec.iter_mut() {
            *value *= inv;
        }
    }
    norm
}

pub(crate) fn pca_patch_heatmap<B: BackendTrait>(
    patch: &Tensor<B, 3>,
    image_count: usize,
) -> Option<Tensor<B, 3>> {
    let [batch, tokens, dim] = patch.shape().dims::<3>();
    if batch == 0 || tokens == 0 || dim == 0 {
        return None;
    }
    let grid = (tokens as f64).sqrt().round() as usize;
    if grid * grid != tokens {
        return None;
    }
    let images = image_count.min(batch);
    if images == 0 {
        return None;
    }
    let data = patch.to_data().convert::<f32>().into_vec::<f32>().ok()?;

    let mut out = vec![0.0f32; images * tokens];
    let mut mean = vec![0.0f32; dim];
    let mut cov = vec![0.0f32; dim * dim];
    let mut vec = vec![0.0f32; dim];
    let mut work = vec![0.0f32; dim];
    let stride = tokens * dim;
    let inv_tokens = 1.0 / tokens as f32;

    for image_idx in 0..images {
        mean.fill(0.0);
        cov.fill(0.0);

        let base = image_idx * stride;
        for token_idx in 0..tokens {
            let offset = base + token_idx * dim;
            for d in 0..dim {
                mean[d] += data[offset + d];
            }
        }
        for mean_value in mean.iter_mut().take(dim) {
            *mean_value *= inv_tokens;
        }

        for token_idx in 0..tokens {
            let offset = base + token_idx * dim;
            for i in 0..dim {
                let xi = data[offset + i] - mean[i];
                let row = i * dim;
                for j in 0..dim {
                    cov[row + j] += xi * (data[offset + j] - mean[j]);
                }
            }
        }
        for value in cov.iter_mut() {
            *value *= inv_tokens;
        }

        vec.fill(0.0);
        for d in 0..dim {
            vec[d] = data[base + d] - mean[d];
        }
        if normalize_vec(&mut vec) < PCA_HEATMAP_EPS {
            vec.fill(0.0);
            vec[0] = 1.0;
        }

        for _ in 0..PCA_HEATMAP_ITERS {
            for (i, work_item) in work.iter_mut().enumerate().take(dim) {
                let row = i * dim;
                let mut sum = 0.0f32;
                for j in 0..dim {
                    sum += cov[row + j] * vec[j];
                }
                *work_item = sum;
            }
            if normalize_vec(&mut work) < PCA_HEATMAP_EPS {
                break;
            }
            std::mem::swap(&mut vec, &mut work);
        }

        for token_idx in 0..tokens {
            let offset = base + token_idx * dim;
            let mut score = 0.0f32;
            for d in 0..dim {
                score += (data[offset + d] - mean[d]) * vec[d];
            }
            out[image_idx * tokens + token_idx] = score;
        }
    }

    let device = patch.device();
    Some(Tensor::<B, 3>::from_data(
        TensorData::new(out, [images, grid, grid]),
        &device,
    ))
}

pub(crate) fn pca_patch_rgb<B: BackendTrait>(
    patch: &Tensor<B, 3>,
    image_count: usize,
) -> Option<Tensor<B, 4>> {
    let [batch, tokens, dim] = patch.shape().dims::<3>();
    if batch == 0 || tokens == 0 || dim == 0 {
        return None;
    }
    let grid = (tokens as f64).sqrt().round() as usize;
    if grid * grid != tokens {
        return None;
    }
    let images = image_count.min(batch);
    if images == 0 {
        return None;
    }
    let data = patch.to_data().convert::<f32>().into_vec::<f32>().ok()?;

    let mut out = vec![0.0f32; images * 3 * tokens];
    let mut mean = vec![0.0f32; dim];
    let mut cov = vec![0.0f32; dim * dim];
    let mut vec = vec![0.0f32; dim];
    let mut work = vec![0.0f32; dim];
    let mut vecs = vec![0.0f32; 3 * dim];
    let stride = tokens * dim;
    let inv_tokens = 1.0 / tokens as f32;

    for image_idx in 0..images {
        mean.fill(0.0);
        cov.fill(0.0);

        let base = image_idx * stride;
        for token_idx in 0..tokens {
            let offset = base + token_idx * dim;
            for d in 0..dim {
                mean[d] += data[offset + d];
            }
        }
        for mean_value in mean.iter_mut().take(dim) {
            *mean_value *= inv_tokens;
        }

        for token_idx in 0..tokens {
            let offset = base + token_idx * dim;
            for i in 0..dim {
                let xi = data[offset + i] - mean[i];
                let row = i * dim;
                for j in 0..dim {
                    cov[row + j] += xi * (data[offset + j] - mean[j]);
                }
            }
        }
        for value in cov.iter_mut() {
            *value *= inv_tokens;
        }

        for comp in 0..3 {
            vec.fill(0.0);
            for d in 0..dim {
                vec[d] = data[base + d] - mean[d];
            }
            if normalize_vec(&mut vec) < PCA_HEATMAP_EPS {
                vec.fill(0.0);
                vec[comp.min(dim.saturating_sub(1))] = 1.0;
            }

            for _ in 0..PCA_HEATMAP_ITERS {
                for (i, work_item) in work.iter_mut().enumerate().take(dim) {
                    let row = i * dim;
                    let mut sum = 0.0f32;
                    for j in 0..dim {
                        sum += cov[row + j] * vec[j];
                    }
                    *work_item = sum;
                }
                if normalize_vec(&mut work) < PCA_HEATMAP_EPS {
                    break;
                }
                std::mem::swap(&mut vec, &mut work);
            }

            let comp_offset = comp * dim;
            vecs[comp_offset..comp_offset + dim].copy_from_slice(&vec);

            let mut lambda = 0.0f32;
            for i in 0..dim {
                let row = i * dim;
                let mut sum = 0.0f32;
                for j in 0..dim {
                    sum += cov[row + j] * vec[j];
                }
                lambda += vec[i] * sum;
            }
            for i in 0..dim {
                let row = i * dim;
                for j in 0..dim {
                    cov[row + j] -= lambda * vec[i] * vec[j];
                }
            }
        }

        for comp in 0..3 {
            let comp_offset = comp * dim;
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for token_idx in 0..tokens {
                let offset = base + token_idx * dim;
                let mut score = 0.0f32;
                for d in 0..dim {
                    score += (data[offset + d] - mean[d]) * vecs[comp_offset + d];
                }
                min_val = min_val.min(score);
                max_val = max_val.max(score);
            }
            let denom = (max_val - min_val).max(PCA_HEATMAP_EPS);
            for token_idx in 0..tokens {
                let offset = base + token_idx * dim;
                let mut score = 0.0f32;
                for d in 0..dim {
                    score += (data[offset + d] - mean[d]) * vecs[comp_offset + d];
                }
                let value = (score - min_val) / denom;
                out[((image_idx * 3 + comp) * tokens) + token_idx] = value.clamp(0.0, 1.0);
            }
        }
    }

    let device = patch.device();
    Some(Tensor::<B, 4>::from_data(
        TensorData::new(out, [images, 3, grid, grid]),
        &device,
    ))
}

pub(crate) fn patch_heatmap_or_norm<B: BackendTrait>(
    patch: Tensor<B, 3>,
    image_count: usize,
) -> Option<Tensor<B, 3>> {
    let pca = pca_patch_heatmap(&patch, image_count);
    if pca.is_some() {
        return pca;
    }
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
}

pub(crate) struct LejepaArtifactBuildInput<B: BackendTrait> {
    pub(crate) frames: Option<Tensor<B, 5>>,
    pub(crate) first_patch: Option<Tensor<B, 3>>,
    pub(crate) pca_source: Option<Tensor<B, 3>>,
    pub(crate) patch_norms_steps: Option<Tensor<B, 4>>,
    pub(crate) pca_rgb_steps: Option<Tensor<B, 5>>,
    pub(crate) probe_logits: Option<Tensor<B, 2>>,
    pub(crate) labels: Option<Tensor<B, 1, Int>>,
    pub(crate) legend: Option<Vec<String>>,
}

pub(crate) fn build_lejepa_artifacts<B: BackendTrait>(
    config: &VisionLejepaConfig,
    views: &[Tensor<B, 4>],
    input: LejepaArtifactBuildInput<B>,
) -> Option<VisionArtifactInput<B>> {
    let LejepaArtifactBuildInput {
        frames,
        first_patch,
        pca_source,
        patch_norms_steps,
        pca_rgb_steps,
        probe_logits,
        labels,
        legend,
    } = input;
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
    let mut stacked = Vec::with_capacity(view_count);
    for view in views.iter().take(view_count) {
        let view = view.clone().slice_dim(0, 0..image_count);
        stacked.push(view.unsqueeze_dim::<5>(1));
    }
    let views_tensor = Tensor::cat(stacked, 1);

    let patch_norms = first_patch.and_then(|patch| patch_heatmap_or_norm(patch, image_count));

    let pca_rgb = pca_source.and_then(|patch| pca_patch_rgb(&patch, image_count));

    let mut legend = normalize_artifact_legend(legend, view_count);
    if let Some(ref mut legend) = legend {
        if patch_norms.is_some() {
            legend.push("heatmap".to_string());
        }
        if pca_rgb.is_some() {
            legend.push("pca_rgb".to_string());
        }
    }

    let probe_logits = probe_logits.map(|logits| logits.slice_dim(0, 0..image_count));
    let labels = labels.map(|labels| labels.slice_dim(0, 0..image_count));
    let frames = frames.map(|frames| frames.slice_dim(0, 0..image_count));
    let patch_norms_steps = patch_norms_steps.map(|maps| maps.slice_dim(0, 0..image_count));
    let pca_rgb_steps = pca_rgb_steps.map(|maps| maps.slice_dim(0, 0..image_count));

    Some(VisionArtifactInput {
        views: Some(views_tensor),
        frames,
        debug_recon_frames: None,
        patch_norms,
        pca_rgb,
        posterior_patch_norms_steps: None,
        posterior_pca_rgb_steps: None,
        patch_norms_steps,
        pca_rgb_steps,
        debug_patch_norms_steps: None,
        debug_pca_rgb_steps: None,
        probe_logits,
        labels,
        legend,
        artifact_scale: 1,
        prediction_start: None,
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
