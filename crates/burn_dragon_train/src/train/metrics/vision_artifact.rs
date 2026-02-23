use super::*;

#[derive(Clone)]
pub struct VisionArtifactMetric<B: BackendTrait> {
    name: Arc<String>,
    output_dir: PathBuf,
    every: usize,
    output_mode: VisionArtifactOutputMode,
    max_images: usize,
    remaining_images: usize,
    last_epoch: Option<usize>,
    fps: u32,
    mean: [f32; 3],
    std: [f32; 3],
    overwrite: bool,
    ffmpeg_path: Option<PathBuf>,
    _marker: std::marker::PhantomData<B>,
}

impl<B: BackendTrait> VisionArtifactMetric<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        output_dir: PathBuf,
        every: usize,
        output_mode: VisionArtifactOutputMode,
        max_images: usize,
        fps: u32,
        mean: [f32; 3],
        std: [f32; 3],
        overwrite: bool,
        ffmpeg_path: Option<PathBuf>,
    ) -> Self {
        Self {
            name: Arc::new("vision_artifacts".to_string()),
            output_dir,
            every,
            output_mode,
            max_images,
            remaining_images: max_images,
            last_epoch: None,
            fps,
            mean,
            std,
            overwrite,
            ffmpeg_path,
            _marker: std::marker::PhantomData,
        }
    }

    fn denormalize_channel(&self, value: f32, channel: usize) -> u8 {
        let mut value = value * self.std[channel] + self.mean[channel];
        value = value.clamp(0.0, 1.0);
        (value * 255.0).round() as u8
    }

    fn write_legend(&self, legend: &[String]) {
        if legend.is_empty() {
            return;
        }
        if fs::create_dir_all(&self.output_dir).is_err() {
            return;
        }
        let mut contents = String::new();
        for (idx, label) in legend.iter().enumerate() {
            if idx > 0 {
                contents.push('\n');
            }
            contents.push_str(&format!("Column {}: {}", idx + 1, label));
        }
        let path = self.output_dir.join("vision_artifacts_key.txt");
        let _ = fs::write(path, contents);
    }

    #[allow(clippy::too_many_arguments)]
    fn build_lejepa_frame(
        &self,
        views_vec: &[f32],
        patch_vec: Option<&[f32]>,
        pca_vec: Option<&[f32]>,
        batch_idx: usize,
        view_count: usize,
        channels: usize,
        height: usize,
        width: usize,
        grid_h: usize,
        grid_w: usize,
        probe_preds: Option<(&[i64], &[i64])>,
    ) -> Option<ArtifactFrame> {
        if channels < 3 || height == 0 || width == 0 || grid_h == 0 || grid_w == 0 {
            return None;
        }
        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
        if heat_patch_h == 0 || heat_patch_w == 0 {
            return None;
        }
        let mut extra_cols = 0usize;
        if patch_vec.is_some() {
            extra_cols += 1;
        }
        if pca_vec.is_some() {
            extra_cols += 1;
        }
        if extra_cols == 0 {
            return None;
        }
        let width_total = width * (view_count + extra_cols);
        let mut canvas = vec![0u8; width_total * height * 3];

        for view_idx in 0..view_count {
            for y in 0..height {
                for x in 0..width {
                    let base =
                        ((batch_idx * view_count + view_idx) * channels * height + y) * width + x;
                    let r = self.denormalize_channel(views_vec[base], 0);
                    let g = self.denormalize_channel(views_vec[base + height * width], 1);
                    let b = self.denormalize_channel(views_vec[base + 2 * height * width], 2);
                    let out_x = view_idx * width + x;
                    let offset = (y * width_total + out_x) * 3;
                    canvas[offset] = r;
                    canvas[offset + 1] = g;
                    canvas[offset + 2] = b;
                }
            }
        }

        let mut column_idx = view_count;
        if let Some(patch_vec) = patch_vec {
            let patch_offset = batch_idx * grid_h * grid_w;
            let patch_slice = &patch_vec[patch_offset..patch_offset + grid_h * grid_w];
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            for value in patch_slice {
                min_val = min_val.min(*value);
                max_val = max_val.max(*value);
            }
            let denom = (max_val - min_val).max(LEJEPA_EPS);
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let value = (patch_slice[gy * grid_w + gx] - min_val) / denom;
                    let pix = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                    for y in (gy * heat_patch_h)..((gy + 1) * heat_patch_h) {
                        for x in (gx * heat_patch_w)..((gx + 1) * heat_patch_w) {
                            let out_x = column_idx * width + x;
                            let offset = (y * width_total + out_x) * 3;
                            canvas[offset] = pix;
                            canvas[offset + 1] = pix;
                            canvas[offset + 2] = pix;
                        }
                    }
                }
            }
            column_idx += 1;
        }

        if let Some(pca_vec) = pca_vec {
            let pca_offset = batch_idx * 3 * grid_h * grid_w;
            for gy in 0..grid_h {
                for gx in 0..grid_w {
                    let base = pca_offset + gy * grid_w + gx;
                    let r = (pca_vec[base].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let g = (pca_vec[base + grid_h * grid_w].clamp(0.0, 1.0) * 255.0).round() as u8;
                    let b =
                        (pca_vec[base + 2 * grid_h * grid_w].clamp(0.0, 1.0) * 255.0).round() as u8;
                    for y in (gy * heat_patch_h)..((gy + 1) * heat_patch_h) {
                        for x in (gx * heat_patch_w)..((gx + 1) * heat_patch_w) {
                            let out_x = column_idx * width + x;
                            let offset = (y * width_total + out_x) * 3;
                            canvas[offset] = r;
                            canvas[offset + 1] = g;
                            canvas[offset + 2] = b;
                        }
                    }
                }
            }
        }

        let is_correct = probe_preds.and_then(|(preds, labels)| {
            let pred = preds.get(batch_idx)?;
            let label = labels.get(batch_idx)?;
            Some(pred == label)
        });
        if let Some(is_correct) = is_correct {
            let (r, g, b) = if is_correct {
                (0u8, 200u8, 0u8)
            } else {
                (200u8, 0u8, 0u8)
            };
            for x in 0..width_total {
                let top = x * 3;
                canvas[top] = r;
                canvas[top + 1] = g;
                canvas[top + 2] = b;
                let bottom = ((height - 1) * width_total + x) * 3;
                canvas[bottom] = r;
                canvas[bottom + 1] = g;
                canvas[bottom + 2] = b;
            }
            for y in 0..height {
                let left = (y * width_total) * 3;
                canvas[left] = r;
                canvas[left + 1] = g;
                canvas[left + 2] = b;
                let right = (y * width_total + (width_total - 1)) * 3;
                canvas[right] = r;
                canvas[right + 1] = g;
                canvas[right + 2] = b;
            }
        }

        Some(ArtifactFrame {
            width: width_total,
            height,
            rgb: canvas,
        })
    }
}

impl<B: BackendTrait> burn_train::metric::Metric for VisionArtifactMetric<B> {
    type Input = VisionArtifactInput<B>;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        if self.every == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "disabled".to_string(),
                "0".to_string(),
            );
        }
        if !metadata.iteration.is_multiple_of(self.every) {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "skip".to_string(),
                "0".to_string(),
            );
        }
        if self.last_epoch != Some(metadata.epoch) {
            self.last_epoch = Some(metadata.epoch);
            self.remaining_images = self.max_images;
        }
        if self.remaining_images == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "budget_exhausted".to_string(),
                "0".to_string(),
            );
        }
        if self.output_mode != VisionArtifactOutputMode::Images {
            if item.frames.is_none()
                && let Some(views) = &item.views
                && (item.patch_norms.is_some()
                    || item.pca_rgb.is_some()
                    || item.patch_norms_steps.is_some()
                    || item.pca_rgb_steps.is_some())
            {
                if let Some(legend) = item.legend.as_ref() {
                    self.write_legend(legend);
                }
                let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
                if batch == 0 || view_count == 0 || channels == 0 || height == 0 || width == 0 {
                    return burn_train::metric::MetricEntry::new(
                        Arc::clone(&self.name),
                        "empty_views".to_string(),
                        "0".to_string(),
                    );
                }
                let (grid_h, grid_w, _norm_batch) =
                    if let Some(patch_steps) = &item.patch_norms_steps {
                        let [norm_batch, _frame_count, grid_h, grid_w] =
                            patch_steps.shape().dims::<4>();
                        if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "empty_norms_steps".to_string(),
                                "0".to_string(),
                            );
                        }
                        (grid_h, grid_w, norm_batch)
                    } else if let Some(pca_steps) = &item.pca_rgb_steps {
                        let [pca_batch, _frame_count, pca_channels, grid_h, grid_w] =
                            pca_steps.shape().dims::<5>();
                        if pca_batch == 0 || grid_h == 0 || grid_w == 0 || pca_channels < 3 {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "empty_pca_steps".to_string(),
                                "0".to_string(),
                            );
                        }
                        (grid_h, grid_w, pca_batch)
                    } else if let Some(patch_norms) = &item.patch_norms {
                        let [norm_batch, grid_h, grid_w] = patch_norms.shape().dims::<3>();
                        if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "empty_norms".to_string(),
                                "0".to_string(),
                            );
                        }
                        (grid_h, grid_w, norm_batch)
                    } else if let Some(pca_rgb) = &item.pca_rgb {
                        let [pca_batch, pca_channels, grid_h, grid_w] = pca_rgb.shape().dims::<4>();
                        if pca_batch == 0 || grid_h == 0 || grid_w == 0 || pca_channels < 3 {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "empty_pca".to_string(),
                                "0".to_string(),
                            );
                        }
                        (grid_h, grid_w, pca_batch)
                    } else {
                        return burn_train::metric::MetricEntry::new(
                            Arc::clone(&self.name),
                            "no_patch_data".to_string(),
                            "0".to_string(),
                        );
                    };
                let views_vec = match views.to_data().convert::<f32>().into_vec::<f32>() {
                    Ok(vec) => vec,
                    Err(_) => {
                        return burn_train::metric::MetricEntry::new(
                            Arc::clone(&self.name),
                            "view_copy_failed".to_string(),
                            "0".to_string(),
                        );
                    }
                };
                let patch_vec = if let Some(patch_norms) = &item.patch_norms {
                    match patch_norms.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "patch_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let pca_dims = item.pca_rgb.as_ref().map(|pca| pca.shape().dims::<4>());
                let mut pca_vec = if let Some(pca_rgb) = &item.pca_rgb {
                    match pca_rgb.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "pca_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                if let (Some([_, pca_channels, pca_h, pca_w]), Some(vec)) =
                    (pca_dims, pca_vec.as_ref())
                {
                    if pca_channels < 3 || pca_h != grid_h || pca_w != grid_w {
                        pca_vec = None;
                    } else {
                        let channel_stride = grid_h * grid_w;
                        let sample_stride = pca_channels * channel_stride;
                        let expected = batch * sample_stride;
                        if vec.len() < expected {
                            pca_vec = None;
                        } else if pca_channels > 3 {
                            let mut packed = vec![0.0f32; batch * 3 * channel_stride];
                            for sample_idx in 0..batch {
                                let src_base = sample_idx * sample_stride;
                                let dst_base = sample_idx * 3 * channel_stride;
                                for channel in 0..3 {
                                    let src = src_base + channel * channel_stride;
                                    let dst = dst_base + channel * channel_stride;
                                    packed[dst..dst + channel_stride]
                                        .copy_from_slice(&vec[src..src + channel_stride]);
                                }
                            }
                            pca_vec = Some(packed);
                        }
                    }
                }

                let patch_steps_dims = item
                    .patch_norms_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<4>());
                let patch_steps_vec = if let Some(maps) = &item.patch_norms_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "patch_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let pca_steps_dims = item
                    .pca_rgb_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<5>());
                let pca_steps_vec = if let Some(maps) = &item.pca_rgb_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "pca_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };

                let patch_steps_frames =
                    if let (Some([norm_batch, frame_count, p_h, p_w]), Some(vec)) =
                        (patch_steps_dims, patch_steps_vec.as_ref())
                    {
                        let expected = norm_batch
                            .saturating_mul(frame_count)
                            .saturating_mul(p_h)
                            .saturating_mul(p_w);
                        if norm_batch == batch
                            && p_h == grid_h
                            && p_w == grid_w
                            && frame_count > 0
                            && vec.len() >= expected
                        {
                            Some(frame_count)
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                let pca_steps_meta =
                    if let (Some([pca_batch, frame_count, pca_channels, p_h, p_w]), Some(vec)) =
                        (pca_steps_dims, pca_steps_vec.as_ref())
                    {
                        let expected = pca_batch
                            .saturating_mul(frame_count)
                            .saturating_mul(pca_channels)
                            .saturating_mul(p_h)
                            .saturating_mul(p_w);
                        if pca_batch == batch
                            && pca_channels >= 3
                            && p_h == grid_h
                            && p_w == grid_w
                            && frame_count > 0
                            && vec.len() >= expected
                        {
                            Some((frame_count, pca_channels))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                let temporal_frames = match (patch_steps_frames, pca_steps_meta) {
                    (Some(patch_frames), Some((pca_frames, _))) => patch_frames.min(pca_frames),
                    (Some(patch_frames), None) => patch_frames,
                    (None, Some((pca_frames, _))) => pca_frames,
                    (None, None) => 0,
                };

                let probe_preds =
                    if let (Some(logits), Some(labels)) = (&item.probe_logits, &item.labels) {
                        let preds = logits
                            .clone()
                            .argmax(1)
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .ok();
                        let labels = labels
                            .clone()
                            .to_data()
                            .convert::<i64>()
                            .into_vec::<i64>()
                            .ok();
                        preds.zip(labels)
                    } else {
                        None
                    };

                let mut saved = 0usize;
                let mut last_mode = self.output_mode;
                let batch_limit = batch.min(self.remaining_images);
                for batch_idx in 0..batch_limit {
                    let probe_slices = probe_preds
                        .as_ref()
                        .map(|(preds, labels)| (preds.as_slice(), labels.as_slice()));
                    let mut frames = Vec::new();
                    if temporal_frames > 0 {
                        for frame_idx in 0..temporal_frames {
                            let patch_frame_vec = if let (Some(vec), Some(frame_count)) =
                                (patch_steps_vec.as_ref(), patch_steps_frames)
                            {
                                let frame_stride = grid_h * grid_w;
                                let mut out = vec![0.0f32; batch * frame_stride];
                                for batch_step in 0..batch {
                                    let src = (batch_step * frame_count + frame_idx) * frame_stride;
                                    let dst = batch_step * frame_stride;
                                    out[dst..dst + frame_stride]
                                        .copy_from_slice(&vec[src..src + frame_stride]);
                                }
                                Some(out)
                            } else {
                                None
                            };
                            let pca_frame_vec =
                                if let (Some(vec), Some((frame_count, pca_channels))) =
                                    (pca_steps_vec.as_ref(), pca_steps_meta)
                                {
                                    let channel_stride = grid_h * grid_w;
                                    let src_frame_stride = pca_channels * channel_stride;
                                    let mut out = vec![0.0f32; batch * 3 * channel_stride];
                                    for batch_step in 0..batch {
                                        let src_base = (batch_step * frame_count + frame_idx)
                                            * src_frame_stride;
                                        let dst_base = batch_step * 3 * channel_stride;
                                        for channel in 0..3 {
                                            let src = src_base + channel * channel_stride;
                                            let dst = dst_base + channel * channel_stride;
                                            out[dst..dst + channel_stride]
                                                .copy_from_slice(&vec[src..src + channel_stride]);
                                        }
                                    }
                                    Some(out)
                                } else {
                                    None
                                };
                            let patch_ref = patch_frame_vec.as_deref().or(patch_vec.as_deref());
                            let pca_ref = pca_frame_vec.as_deref().or(pca_vec.as_deref());
                            if let Some(frame) = self.build_lejepa_frame(
                                &views_vec,
                                patch_ref,
                                pca_ref,
                                batch_idx,
                                view_count,
                                channels,
                                height,
                                width,
                                grid_h,
                                grid_w,
                                probe_slices,
                            ) {
                                frames.push(frame);
                            }
                        }
                    } else if let Some(frame) = self.build_lejepa_frame(
                        &views_vec,
                        patch_vec.as_deref(),
                        pca_vec.as_deref(),
                        batch_idx,
                        view_count,
                        channels,
                        height,
                        width,
                        grid_h,
                        grid_w,
                        probe_slices,
                    ) {
                        frames.push(frame);
                    }
                    if frames.is_empty() {
                        continue;
                    }
                    let outcome = match write_video(
                        &self.output_dir,
                        self.output_mode,
                        self.overwrite,
                        metadata.iteration,
                        batch_idx,
                        &frames,
                        self.fps,
                        self.ffmpeg_path.as_deref(),
                    ) {
                        Ok(outcome) => outcome,
                        Err(_) => {
                            return burn_train::metric::MetricEntry::new(
                                Arc::clone(&self.name),
                                "video_write_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    };
                    saved += outcome.saved;
                    last_mode = outcome.mode;
                }
                self.remaining_images = self.remaining_images.saturating_sub(batch_limit);
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    format!("saved={saved} mode={last_mode}"),
                    saved.to_string(),
                );
            }

            let frames_tensor = item.frames.as_ref().or(item.views.as_ref());
            let Some(frames_tensor) = frames_tensor else {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "no_frames".to_string(),
                    "0".to_string(),
                );
            };
            let [batch, frame_count, channels, height, width] = frames_tensor.shape().dims::<5>();
            if batch == 0 || frame_count == 0 || channels == 0 || height == 0 || width == 0 {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "empty_frames".to_string(),
                    "0".to_string(),
                );
            }
            if let Some(legend) = item.legend.as_ref() {
                self.write_legend(legend);
            }
            let frames_vec = match frames_tensor.to_data().convert::<f32>().into_vec::<f32>() {
                Ok(vec) => vec,
                Err(_) => {
                    return burn_train::metric::MetricEntry::new(
                        Arc::clone(&self.name),
                        "frame_copy_failed".to_string(),
                        "0".to_string(),
                    );
                }
            };
            let mut saved = 0usize;
            let mut last_mode = self.output_mode;
            let batch_limit = batch.min(self.remaining_images);
            for batch_idx in 0..batch_limit {
                let frames = collect_frames(
                    &frames_vec,
                    batch,
                    frame_count,
                    channels,
                    height,
                    width,
                    batch_idx,
                    self.mean,
                    self.std,
                );
                if frames.is_empty() {
                    continue;
                }
                let outcome = match write_video(
                    &self.output_dir,
                    self.output_mode,
                    self.overwrite,
                    metadata.iteration,
                    batch_idx,
                    &frames,
                    self.fps,
                    self.ffmpeg_path.as_deref(),
                ) {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        return burn_train::metric::MetricEntry::new(
                            Arc::clone(&self.name),
                            "video_write_failed".to_string(),
                            "0".to_string(),
                        );
                    }
                };
                saved += outcome.saved;
                last_mode = outcome.mode;
            }
            self.remaining_images = self.remaining_images.saturating_sub(batch_limit);
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                format!("saved={saved} mode={last_mode}"),
                saved.to_string(),
            );
        }

        let Some(views) = &item.views else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_views".to_string(),
                "0".to_string(),
            );
        };
        if item.patch_norms.is_none() && item.pca_rgb.is_none() {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_patch_data".to_string(),
                "0".to_string(),
            );
        }
        if let Some(legend) = item.legend.as_ref() {
            self.write_legend(legend);
        }

        let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
        if batch == 0 || view_count == 0 || channels == 0 || height == 0 || width == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "empty".to_string(),
                "0".to_string(),
            );
        }

        let (grid_h, grid_w, _norm_batch) = if let Some(patch_norms) = &item.patch_norms {
            let [norm_batch, grid_h, grid_w] = patch_norms.shape().dims::<3>();
            if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "empty_norms".to_string(),
                    "0".to_string(),
                );
            }
            (grid_h, grid_w, norm_batch)
        } else if let Some(pca_rgb) = &item.pca_rgb {
            let [pca_batch, pca_channels, grid_h, grid_w] = pca_rgb.shape().dims::<4>();
            if pca_batch == 0 || grid_h == 0 || grid_w == 0 || pca_channels < 3 {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "empty_pca".to_string(),
                    "0".to_string(),
                );
            }
            (grid_h, grid_w, pca_batch)
        } else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_patch_data".to_string(),
                "0".to_string(),
            );
        };

        let views_vec = match views.to_data().convert::<f32>().into_vec::<f32>() {
            Ok(vec) => vec,
            Err(_) => {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "view_copy_failed".to_string(),
                    "0".to_string(),
                );
            }
        };
        let patch_vec = if let Some(patch_norms) = &item.patch_norms {
            match patch_norms.to_data().convert::<f32>().into_vec::<f32>() {
                Ok(vec) => Some(vec),
                Err(_) => {
                    return burn_train::metric::MetricEntry::new(
                        Arc::clone(&self.name),
                        "patch_copy_failed".to_string(),
                        "0".to_string(),
                    );
                }
            }
        } else {
            None
        };
        let pca_dims = item.pca_rgb.as_ref().map(|pca| pca.shape().dims::<4>());
        let mut pca_vec = if let Some(pca_rgb) = &item.pca_rgb {
            match pca_rgb.to_data().convert::<f32>().into_vec::<f32>() {
                Ok(vec) => Some(vec),
                Err(_) => {
                    return burn_train::metric::MetricEntry::new(
                        Arc::clone(&self.name),
                        "pca_copy_failed".to_string(),
                        "0".to_string(),
                    );
                }
            }
        } else {
            None
        };
        if let (Some([_, pca_channels, pca_h, pca_w]), Some(vec)) = (pca_dims, pca_vec.as_ref()) {
            if pca_channels < 3 || pca_h != grid_h || pca_w != grid_w {
                pca_vec = None;
            } else {
                let expected = batch * 3 * grid_h * grid_w;
                if vec.len() < expected {
                    pca_vec = None;
                }
            }
        }
        let probe_preds = if let (Some(logits), Some(labels)) = (&item.probe_logits, &item.labels) {
            let preds = logits
                .clone()
                .argmax(1)
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .ok();
            let labels = labels
                .clone()
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .ok();
            preds.zip(labels)
        } else {
            None
        };

        if let Err(err) = fs::create_dir_all(&self.output_dir) {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                format!("mkdir_failed: {err}"),
                "0".to_string(),
            );
        }

        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
        if heat_patch_h == 0 || heat_patch_w == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "heatmap_scale_invalid".to_string(),
                "0".to_string(),
            );
        }

        let mut saved = 0usize;
        let mut log_lines = Vec::new();
        let batch_limit = batch.min(self.remaining_images);
        let probe_slices = probe_preds
            .as_ref()
            .map(|(preds, labels)| (preds.as_slice(), labels.as_slice()));
        for batch_idx in 0..batch_limit {
            let Some(frame) = self.build_lejepa_frame(
                &views_vec,
                patch_vec.as_deref(),
                pca_vec.as_deref(),
                batch_idx,
                view_count,
                channels,
                height,
                width,
                grid_h,
                grid_w,
                probe_slices,
            ) else {
                continue;
            };
            if let Some(image) =
                image::RgbImage::from_vec(frame.width as u32, frame.height as u32, frame.rgb)
            {
                let filename = if self.overwrite {
                    format!("sample_{:02}.png", batch_idx)
                } else if let Some((preds, labels)) = &probe_preds {
                    let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                    let label = labels.get(batch_idx).copied().unwrap_or(-1);
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}_pred_{pred}_label_{label}.png",
                        metadata.iteration, batch_idx
                    )
                } else {
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}.png",
                        metadata.iteration, batch_idx
                    )
                };
                let path = self.output_dir.join(filename);
                if image.save(path).is_ok() {
                    saved += 1;
                }
            }

            if let Some((preds, labels)) = &probe_preds {
                let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                let label = labels.get(batch_idx).copied().unwrap_or(-1);
                let correct = if pred == label { "1" } else { "0" };
                log_lines.push(format!(
                    "{},{},{},{},{}",
                    metadata.iteration, batch_idx, pred, label, correct
                ));
            }
        }

        if !log_lines.is_empty() {
            let log_path = self.output_dir.join("vision_artifacts.log");
            let mut contents = String::new();
            contents.push_str("iteration,batch_idx,pred,label,correct\n");
            contents.push_str(&log_lines.join("\n"));
            if self.overwrite {
                let _ = fs::write(log_path, contents);
            } else if let Ok(mut file) = fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(log_path)
            {
                let _ = writeln!(file, "{contents}");
            }
        }

        self.remaining_images = self.remaining_images.saturating_sub(batch_limit);

        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            format!("saved={saved}"),
            saved.to_string(),
        )
    }

    fn clear(&mut self) {}
}
