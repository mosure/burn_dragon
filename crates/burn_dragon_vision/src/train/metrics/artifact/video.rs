use burn_dragon_train::train::artifacts::{collect_frames, write_video};

use super::*;

impl<B: BackendTrait> VisionArtifactMetric<B> {
    #[allow(clippy::too_many_arguments)]
    fn build_video_feature_frame(
        &self,
        frames_vec: &[f32],
        debug_recon_vec: Option<&[f32]>,
        posterior_patch_steps_vec: Option<&[f32]>,
        posterior_pca_steps_vec: Option<&[f32]>,
        reference_patch_steps_vec: Option<&[f32]>,
        reference_pca_steps_vec: Option<&[f32]>,
        debug_patch_steps_vec: Option<&[f32]>,
        debug_pca_steps_vec: Option<&[f32]>,
        batch_idx: usize,
        frame_idx: usize,
        frame_count: usize,
        channels: usize,
        height: usize,
        width: usize,
        grid_h: usize,
        grid_w: usize,
        pca_channels: usize,
        prediction_start: Option<usize>,
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
        if posterior_patch_steps_vec.is_some() {
            extra_cols += 1;
        }
        if posterior_pca_steps_vec.is_some() {
            extra_cols += 1;
        }
        if reference_patch_steps_vec.is_some() {
            extra_cols += 1;
        }
        if reference_pca_steps_vec.is_some() {
            extra_cols += 1;
        }
        if debug_recon_vec.is_some() {
            extra_cols += 1;
        }
        if debug_patch_steps_vec.is_some() {
            extra_cols += 1;
        }
        if debug_pca_steps_vec.is_some() {
            extra_cols += 1;
        }
        if extra_cols == 0 || frame_idx >= frame_count {
            return None;
        }

        let width_total = width * (1 + extra_cols);
        let mut canvas = vec![0u8; width_total * height * 3];
        let channel_stride = height * width;
        let frame_stride = channels * channel_stride;
        let frame_base = (batch_idx * frame_count + frame_idx) * frame_stride;
        for y in 0..height {
            for x in 0..width {
                let idx = frame_base + y * width + x;
                let offset = (y * width_total + x) * 3;
                canvas[offset] = self.denormalize_channel(frames_vec[idx], 0);
                canvas[offset + 1] = self.denormalize_channel(frames_vec[idx + channel_stride], 1);
                canvas[offset + 2] =
                    self.denormalize_channel(frames_vec[idx + 2 * channel_stride], 2);
            }
        }

        let mut column_idx = 1usize;
        if let Some(posterior_patch_steps_vec) = posterior_patch_steps_vec {
            let frame_stride = grid_h * grid_w;
            let patch_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_patch_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &posterior_patch_steps_vec[patch_base..patch_base + frame_stride],
            );
            column_idx += 1;
        }
        if let Some(posterior_pca_steps_vec) = posterior_pca_steps_vec {
            let channel_stride = grid_h * grid_w;
            let frame_stride = pca_channels * channel_stride;
            let pca_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_pca_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &posterior_pca_steps_vec[pca_base..pca_base + 3 * channel_stride],
            );
            column_idx += 1;
        }
        if let Some(reference_patch_steps_vec) = reference_patch_steps_vec {
            let frame_stride = grid_h * grid_w;
            let patch_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_patch_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &reference_patch_steps_vec[patch_base..patch_base + frame_stride],
            );
            column_idx += 1;
        }
        if let Some(reference_pca_steps_vec) = reference_pca_steps_vec {
            let channel_stride = grid_h * grid_w;
            let frame_stride = pca_channels * channel_stride;
            let pca_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_pca_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &reference_pca_steps_vec[pca_base..pca_base + 3 * channel_stride],
            );
            column_idx += 1;
        }
        if let Some(debug_recon_vec) = debug_recon_vec {
            for y in 0..height {
                for x in 0..width {
                    let idx = frame_base + y * width + x;
                    let out_x = column_idx * width + x;
                    let offset = (y * width_total + out_x) * 3;
                    canvas[offset] = self.denormalize_channel(debug_recon_vec[idx], 0);
                    canvas[offset + 1] =
                        self.denormalize_channel(debug_recon_vec[idx + channel_stride], 1);
                    canvas[offset + 2] =
                        self.denormalize_channel(debug_recon_vec[idx + 2 * channel_stride], 2);
                }
            }
            column_idx += 1;
        }
        if let Some(debug_patch_steps_vec) = debug_patch_steps_vec {
            let frame_stride = grid_h * grid_w;
            let patch_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_patch_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &debug_patch_steps_vec[patch_base..patch_base + frame_stride],
            );
            column_idx += 1;
        }
        if let Some(debug_pca_steps_vec) = debug_pca_steps_vec {
            let channel_stride = grid_h * grid_w;
            let frame_stride = pca_channels * channel_stride;
            let pca_base = (batch_idx * frame_count + frame_idx) * frame_stride;
            self.paint_pca_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &debug_pca_steps_vec[pca_base..pca_base + 3 * channel_stride],
            );
        }

        let _ = probe_preds;
        let _ = prediction_start;

        Some(ArtifactFrame {
            width: width_total,
            height,
            rgb: canvas,
        })
    }

    pub(super) fn update_video_artifacts(
        &mut self,
        item: &VisionArtifactInput<B>,
        iteration: usize,
    ) -> SerializedEntry {
        if let Some(frames_tensor) = &item.frames {
            let [batch, frame_count, channels, height, width] = frames_tensor.shape().dims::<5>();
            if batch > 0 && frame_count > 0 && channels > 0 && height > 0 && width > 0 {
                let artifact_scale = item.artifact_scale.max(1);
                let debug_recon_vec = if let Some(debug_frames) = &item.debug_recon_frames {
                    let dims = debug_frames.shape().dims::<5>();
                    let expected = batch
                        .saturating_mul(frame_count)
                        .saturating_mul(channels)
                        .saturating_mul(height)
                        .saturating_mul(width);
                    if dims == [batch, frame_count, channels, height, width] {
                        match debug_frames.to_data().convert::<f32>().into_vec::<f32>() {
                            Ok(vec) if vec.len() >= expected => Some(vec),
                            _ => None,
                        }
                    } else {
                        None
                    }
                } else {
                    None
                };
                let patch_steps_dims = item
                    .patch_norms_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<4>());
                let posterior_patch_steps_dims = item
                    .posterior_patch_norms_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<4>());
                let patch_steps_vec = if let Some(maps) = &item.patch_norms_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "patch_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let posterior_patch_steps_vec = if let Some(maps) = &item.posterior_patch_norms_steps
                {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "posterior_patch_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let pca_steps_dims = item.pca_rgb_steps.as_ref().map(|maps| maps.shape().dims::<5>());
                let posterior_pca_steps_dims = item
                    .posterior_pca_rgb_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<5>());
                let pca_steps_vec = if let Some(maps) = &item.pca_rgb_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "pca_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let posterior_pca_steps_vec = if let Some(maps) = &item.posterior_pca_rgb_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "posterior_pca_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let debug_patch_steps_dims = item
                    .debug_patch_norms_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<4>());
                let debug_patch_steps_vec = if let Some(maps) = &item.debug_patch_norms_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "debug_patch_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };
                let debug_pca_steps_dims = item
                    .debug_pca_rgb_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<5>());
                let debug_pca_steps_vec = if let Some(maps) = &item.debug_pca_rgb_steps {
                    match maps.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => Some(vec),
                        Err(_) => {
                            return serialized_entry(
                                "debug_pca_steps_copy_failed".to_string(),
                                "0".to_string(),
                            );
                        }
                    }
                } else {
                    None
                };

                let patch_meta = if let (Some([norm_batch, patch_frames, grid_h, grid_w]), Some(vec)) =
                    (patch_steps_dims, patch_steps_vec.as_ref())
                {
                    let expected = norm_batch
                        .saturating_mul(patch_frames)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if norm_batch == batch
                        && patch_frames == frame_count
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let posterior_patch_meta = if let (
                    Some([norm_batch, patch_frames, grid_h, grid_w]),
                    Some(vec),
                ) = (posterior_patch_steps_dims, posterior_patch_steps_vec.as_ref())
                {
                    let expected = norm_batch
                        .saturating_mul(patch_frames)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if norm_batch == batch
                        && patch_frames == frame_count
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let pca_meta = if let (
                    Some([pca_batch, pca_frames, pca_channels, grid_h, grid_w]),
                    Some(vec),
                ) = (pca_steps_dims, pca_steps_vec.as_ref())
                {
                    let expected = pca_batch
                        .saturating_mul(pca_frames)
                        .saturating_mul(pca_channels)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if pca_batch == batch
                        && pca_frames == frame_count
                        && pca_channels >= 3
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w, pca_channels))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let posterior_pca_meta = if let (
                    Some([pca_batch, pca_frames, pca_channels, grid_h, grid_w]),
                    Some(vec),
                ) = (posterior_pca_steps_dims, posterior_pca_steps_vec.as_ref())
                {
                    let expected = pca_batch
                        .saturating_mul(pca_frames)
                        .saturating_mul(pca_channels)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if pca_batch == batch
                        && pca_frames == frame_count
                        && pca_channels >= 3
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w, pca_channels))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let debug_patch_meta = if let (Some([norm_batch, patch_frames, grid_h, grid_w]), Some(vec)) =
                    (debug_patch_steps_dims, debug_patch_steps_vec.as_ref())
                {
                    let expected = norm_batch
                        .saturating_mul(patch_frames)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if norm_batch == batch
                        && patch_frames == frame_count
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w))
                    } else {
                        None
                    }
                } else {
                    None
                };
                let debug_pca_meta = if let (
                    Some([pca_batch, pca_frames, pca_channels, grid_h, grid_w]),
                    Some(vec),
                ) = (debug_pca_steps_dims, debug_pca_steps_vec.as_ref())
                {
                    let expected = pca_batch
                        .saturating_mul(pca_frames)
                        .saturating_mul(pca_channels)
                        .saturating_mul(grid_h)
                        .saturating_mul(grid_w);
                    if pca_batch == batch
                        && pca_frames == frame_count
                        && pca_channels >= 3
                        && grid_h > 0
                        && grid_w > 0
                        && vec.len() >= expected
                    {
                        Some((grid_h, grid_w, pca_channels))
                    } else {
                        None
                    }
                } else {
                    None
                };

                let mut grid = patch_meta
                    .or_else(|| pca_meta.map(|(h, w, _)| (h, w)))
                    .or(posterior_patch_meta)
                    .or_else(|| posterior_pca_meta.map(|(h, w, _)| (h, w)));
                if grid.is_none() {
                    grid = debug_patch_meta.or_else(|| debug_pca_meta.map(|(h, w, _)| (h, w)));
                }
                let pca_channels = pca_meta
                    .map(|(_, _, channels)| channels)
                    .or_else(|| posterior_pca_meta.map(|(_, _, channels)| channels))
                    .or_else(|| debug_pca_meta.map(|(_, _, channels)| channels))
                    .unwrap_or(0);
                let grid = grid.and_then(|(grid_h, grid_w)| {
                    let refs_match = patch_meta.is_none_or(|(h, w)| h == grid_h && w == grid_w)
                        && posterior_patch_meta.is_none_or(|(h, w)| h == grid_h && w == grid_w)
                        && pca_meta.is_none_or(|(h, w, _)| h == grid_h && w == grid_w)
                        && posterior_pca_meta.is_none_or(|(h, w, _)| h == grid_h && w == grid_w)
                        && debug_patch_meta.is_none_or(|(h, w)| h == grid_h && w == grid_w)
                        && debug_pca_meta.is_none_or(|(h, w, _)| h == grid_h && w == grid_w);
                    refs_match.then_some((grid_h, grid_w, pca_channels))
                });

                if let Some((grid_h, grid_w, pca_channels)) = grid {
                    if let Some(legend) = item.legend.as_ref() {
                        let mut notes = Vec::new();
                        if let Some(start) = item.prediction_start {
                            if start > 0 {
                                notes.push(format!(
                                    "Frames 0-{}: observed context; posterior columns show the post-merge state, while state columns show the refined recurrent state.",
                                    start.saturating_sub(1)
                                ));
                            }
                            if frame_count > start {
                                notes.push(format!(
                                    "Frames {}-{}: predictive future; posterior columns are not applicable, and state columns show open-loop recurrent rollout.",
                                    start,
                                    frame_count - 1
                                ));
                            }
                        }
                        if item.debug_recon_frames.is_some()
                            && (item.debug_patch_norms_steps.is_some()
                                || item.debug_pca_rgb_steps.is_some())
                        {
                            notes.push(
                                "Final columns are encoder maps of the decoded clip, not direct latent-state maps."
                                    .to_string(),
                            );
                        }
                        self.write_legend_with_notes(legend, &notes);
                    }
                    let frames_vec = match frames_tensor.to_data().convert::<f32>().into_vec::<f32>() {
                        Ok(vec) => vec,
                        Err(_) => {
                            return serialized_entry("frame_copy_failed".to_string(), "0".to_string());
                        }
                    };
                    let probe_preds = if let (Some(logits), Some(labels)) =
                        (&item.probe_logits, &item.labels)
                    {
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
                    let probe_slices = probe_preds
                        .as_ref()
                        .map(|(preds, labels)| (preds.as_slice(), labels.as_slice()));
                    for batch_idx in 0..batch_limit {
                        let mut frames = Vec::with_capacity(frame_count);
                        for frame_idx in 0..frame_count {
                            if let Some(frame) = self.build_video_feature_frame(
                                &frames_vec,
                                debug_recon_vec.as_deref(),
                                posterior_patch_steps_vec.as_deref(),
                                posterior_pca_steps_vec.as_deref(),
                                patch_steps_vec.as_deref(),
                                pca_steps_vec.as_deref(),
                                debug_patch_steps_vec.as_deref(),
                                debug_pca_steps_vec.as_deref(),
                                batch_idx,
                                frame_idx,
                                frame_count,
                                channels,
                                height,
                                width,
                                grid_h,
                                grid_w,
                                pca_channels,
                                item.prediction_start,
                                probe_slices,
                            ) {
                                frames.push(self.maybe_upscale_frame(frame, artifact_scale));
                            }
                        }
                        if frames.is_empty() {
                            continue;
                        }
                        let outcome = match write_video(
                            &self.output_dir,
                            self.output_mode,
                            self.overwrite,
                            iteration,
                            batch_idx,
                            &frames,
                            self.fps,
                            self.ffmpeg_path.as_deref(),
                        ) {
                            Ok(outcome) => outcome,
                            Err(_) => {
                                return serialized_entry(
                                    "video_write_failed".to_string(),
                                    "0".to_string(),
                                );
                            }
                        };
                        saved += outcome.saved;
                        last_mode = outcome.mode;
                    }
                    self.remaining_images = self.remaining_images.saturating_sub(batch_limit);
                    return serialized_entry(
                        format!("saved={saved} mode={last_mode}"),
                        saved.to_string(),
                    );
                }
            }
        }

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
                return serialized_entry("empty_views".to_string(), "0".to_string());
            }
            let (grid_h, grid_w, _norm_batch) = if let Some(patch_steps) = &item.patch_norms_steps {
                let [norm_batch, _frame_count, grid_h, grid_w] = patch_steps.shape().dims::<4>();
                if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
                    return serialized_entry("empty_norms_steps".to_string(), "0".to_string());
                }
                (grid_h, grid_w, norm_batch)
            } else if let Some(pca_steps) = &item.pca_rgb_steps {
                let [pca_batch, _frame_count, pca_channels, grid_h, grid_w] =
                    pca_steps.shape().dims::<5>();
                if pca_batch == 0 || grid_h == 0 || grid_w == 0 || pca_channels < 3 {
                    return serialized_entry("empty_pca_steps".to_string(), "0".to_string());
                }
                (grid_h, grid_w, pca_batch)
            } else if let Some(patch_norms) = &item.patch_norms {
                let [norm_batch, grid_h, grid_w] = patch_norms.shape().dims::<3>();
                if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
                    return serialized_entry("empty_norms".to_string(), "0".to_string());
                }
                (grid_h, grid_w, norm_batch)
            } else if let Some(pca_rgb) = &item.pca_rgb {
                let [pca_batch, pca_channels, grid_h, grid_w] = pca_rgb.shape().dims::<4>();
                if pca_batch == 0 || grid_h == 0 || grid_w == 0 || pca_channels < 3 {
                    return serialized_entry("empty_pca".to_string(), "0".to_string());
                }
                (grid_h, grid_w, pca_batch)
            } else {
                return serialized_entry("no_patch_data".to_string(), "0".to_string());
            };
            let views_vec = match views.to_data().convert::<f32>().into_vec::<f32>() {
                Ok(vec) => vec,
                Err(_) => {
                    return serialized_entry("view_copy_failed".to_string(), "0".to_string());
                }
            };
            let patch_vec = if let Some(patch_norms) = &item.patch_norms {
                match patch_norms.to_data().convert::<f32>().into_vec::<f32>() {
                    Ok(vec) => Some(vec),
                    Err(_) => {
                        return serialized_entry("patch_copy_failed".to_string(), "0".to_string());
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
                        return serialized_entry("pca_copy_failed".to_string(), "0".to_string());
                    }
                }
            } else {
                None
            };
            if let (Some([_, pca_channels, pca_h, pca_w]), Some(vec)) = (pca_dims, pca_vec.as_ref()) {
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
                        return serialized_entry(
                            "patch_steps_copy_failed".to_string(),
                            "0".to_string(),
                        );
                    }
                }
            } else {
                None
            };
            let pca_steps_dims = item.pca_rgb_steps.as_ref().map(|maps| maps.shape().dims::<5>());
            let pca_steps_vec = if let Some(maps) = &item.pca_rgb_steps {
                match maps.to_data().convert::<f32>().into_vec::<f32>() {
                    Ok(vec) => Some(vec),
                    Err(_) => {
                        return serialized_entry("pca_steps_copy_failed".to_string(), "0".to_string());
                    }
                }
            } else {
                None
            };

            let patch_steps_frames = if let (Some([norm_batch, frame_count, p_h, p_w]), Some(vec)) =
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

            let probe_preds = if let (Some(logits), Some(labels)) = (&item.probe_logits, &item.labels)
            {
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
            let artifact_scale = item.artifact_scale.max(1);
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
                        let pca_frame_vec = if let (Some(vec), Some((frame_count, pca_channels))) =
                            (pca_steps_vec.as_ref(), pca_steps_meta)
                        {
                            let channel_stride = grid_h * grid_w;
                            let src_frame_stride = pca_channels * channel_stride;
                            let mut out = vec![0.0f32; batch * 3 * channel_stride];
                            for batch_step in 0..batch {
                                let src_base = (batch_step * frame_count + frame_idx) * src_frame_stride;
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
                            frames.push(self.maybe_upscale_frame(frame, artifact_scale));
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
                    frames.push(self.maybe_upscale_frame(frame, artifact_scale));
                }
                if frames.is_empty() {
                    continue;
                }
                let outcome = match write_video(
                    &self.output_dir,
                    self.output_mode,
                    self.overwrite,
                    iteration,
                    batch_idx,
                    &frames,
                    self.fps,
                    self.ffmpeg_path.as_deref(),
                ) {
                    Ok(outcome) => outcome,
                    Err(_) => {
                        return serialized_entry("video_write_failed".to_string(), "0".to_string());
                    }
                };
                saved += outcome.saved;
                last_mode = outcome.mode;
            }
            self.remaining_images = self.remaining_images.saturating_sub(batch_limit);
            return serialized_entry(format!("saved={saved} mode={last_mode}"), saved.to_string());
        }

        let frames_tensor = item.frames.as_ref().or(item.views.as_ref());
        let Some(frames_tensor) = frames_tensor else {
            return serialized_entry("no_frames".to_string(), "0".to_string());
        };
        let [batch, frame_count, channels, height, width] = frames_tensor.shape().dims::<5>();
        if batch == 0 || frame_count == 0 || channels == 0 || height == 0 || width == 0 {
            return serialized_entry("empty_frames".to_string(), "0".to_string());
        }
        if let Some(legend) = item.legend.as_ref() {
            self.write_legend(legend);
        }
        let frames_vec = match frames_tensor.to_data().convert::<f32>().into_vec::<f32>() {
            Ok(vec) => vec,
            Err(_) => {
                return serialized_entry("frame_copy_failed".to_string(), "0".to_string());
            }
        };
        let mut saved = 0usize;
        let mut last_mode = self.output_mode;
        let batch_limit = batch.min(self.remaining_images);
        let artifact_scale = item.artifact_scale.max(1);
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
            )
            .into_iter()
            .map(|frame| self.maybe_upscale_frame(frame, artifact_scale))
            .collect::<Vec<_>>();
            if frames.is_empty() {
                continue;
            }
            let outcome = match write_video(
                &self.output_dir,
                self.output_mode,
                self.overwrite,
                iteration,
                batch_idx,
                &frames,
                self.fps,
                self.ffmpeg_path.as_deref(),
            ) {
                Ok(outcome) => outcome,
                Err(_) => {
                    return serialized_entry("video_write_failed".to_string(), "0".to_string());
                }
            };
            saved += outcome.saved;
            last_mode = outcome.mode;
        }
        self.remaining_images = self.remaining_images.saturating_sub(batch_limit);
        serialized_entry(format!("saved={saved} mode={last_mode}"), saved.to_string())
    }
}
