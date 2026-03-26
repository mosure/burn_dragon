use std::io::Write;

use ::image::RgbImage;
use burn_dragon_train::train::constants::LEJEPA_EPS;

use super::*;

impl<B: BackendTrait> VisionArtifactMetric<B> {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn paint_patch_column(
        &self,
        canvas: &mut [u8],
        width_total: usize,
        column_idx: usize,
        width: usize,
        height: usize,
        grid_h: usize,
        grid_w: usize,
        patch_slice: &[f32],
    ) {
        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
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
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn paint_pca_column(
        &self,
        canvas: &mut [u8],
        width_total: usize,
        column_idx: usize,
        width: usize,
        height: usize,
        grid_h: usize,
        grid_w: usize,
        pca_slice: &[f32],
    ) {
        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
        let channel_stride = grid_h * grid_w;
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let base = gy * grid_w + gx;
                let r = (pca_slice[base].clamp(0.0, 1.0) * 255.0).round() as u8;
                let g = (pca_slice[base + channel_stride].clamp(0.0, 1.0) * 255.0).round() as u8;
                let b =
                    (pca_slice[base + 2 * channel_stride].clamp(0.0, 1.0) * 255.0).round() as u8;
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

    fn paint_probe_border(
        &self,
        canvas: &mut [u8],
        width_total: usize,
        height: usize,
        is_correct: Option<bool>,
    ) {
        let Some(is_correct) = is_correct else {
            return;
        };
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_lejepa_frame(
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
            self.paint_patch_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                patch_slice,
            );
            column_idx += 1;
        }

        if let Some(pca_vec) = pca_vec {
            let pca_offset = batch_idx * 3 * grid_h * grid_w;
            self.paint_pca_column(
                &mut canvas,
                width_total,
                column_idx,
                width,
                height,
                grid_h,
                grid_w,
                &pca_vec[pca_offset..pca_offset + 3 * grid_h * grid_w],
            );
        }

        let is_correct = probe_preds.and_then(|(preds, labels)| {
            let pred = preds.get(batch_idx)?;
            let label = labels.get(batch_idx)?;
            Some(pred == label)
        });
        self.paint_probe_border(&mut canvas, width_total, height, is_correct);

        Some(ArtifactFrame {
            width: width_total,
            height,
            rgb: canvas,
        })
    }

    fn stack_temporal_rows(&self, rows: &[ArtifactFrame]) -> Option<ArtifactFrame> {
        let first = rows.first()?;
        if rows
            .iter()
            .any(|row| row.width != first.width || row.height != first.height)
        {
            return None;
        }
        let width = first.width;
        let row_height = first.height;
        let separator = 1usize;
        let height = rows.len().saturating_mul(row_height)
            + rows.len().saturating_sub(1).saturating_mul(separator);
        let mut canvas = vec![0u8; width.saturating_mul(height).saturating_mul(3)];
        for (row_idx, row) in rows.iter().enumerate() {
            let dst_y = row_idx.saturating_mul(row_height + separator);
            for y in 0..row_height {
                let dst_offset = (dst_y + y).saturating_mul(width).saturating_mul(3);
                let src_offset = y.saturating_mul(width).saturating_mul(3);
                canvas[dst_offset..dst_offset + width * 3]
                    .copy_from_slice(&row.rgb[src_offset..src_offset + width * 3]);
            }
            if row_idx + 1 < rows.len() {
                let sep_y = dst_y + row_height;
                for x in 0..width {
                    let offset = (sep_y * width + x) * 3;
                    canvas[offset] = 24;
                    canvas[offset + 1] = 24;
                    canvas[offset + 2] = 24;
                }
            }
        }

        Some(ArtifactFrame {
            width,
            height,
            rgb: canvas,
        })
    }

    pub(super) fn update_image_artifacts(
        &mut self,
        item: &VisionArtifactInput<B>,
        epoch: usize,
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
                let aux_frames_vec = if let Some(aux_frames) = &item.aux_frames {
                    let dims = aux_frames.shape().dims::<5>();
                    let expected = batch
                        .saturating_mul(frame_count)
                        .saturating_mul(channels)
                        .saturating_mul(height)
                        .saturating_mul(width);
                    if dims == [batch, frame_count, channels, height, width] {
                        match aux_frames.to_data().convert::<f32>().into_vec::<f32>() {
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
                let posterior_patch_steps_vec =
                    if let Some(maps) = &item.posterior_patch_norms_steps {
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
                let pca_steps_dims = item
                    .pca_rgb_steps
                    .as_ref()
                    .map(|maps| maps.shape().dims::<5>());
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

                let patch_meta =
                    if let (Some([norm_batch, patch_frames, grid_h, grid_w]), Some(vec)) =
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
                let posterior_patch_meta =
                    if let (Some([norm_batch, patch_frames, grid_h, grid_w]), Some(vec)) = (
                        posterior_patch_steps_dims,
                        posterior_patch_steps_vec.as_ref(),
                    ) {
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
                ) =
                    (posterior_pca_steps_dims, posterior_pca_steps_vec.as_ref())
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
                let debug_patch_meta =
                    if let (Some([norm_batch, patch_frames, grid_h, grid_w]), Some(vec)) =
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
                    let mut notes = vec![format!(
                        "Rows correspond to solver steps 0-{}, top to bottom.",
                        frame_count.saturating_sub(1)
                    )];
                    if let Some(start) = item.prediction_start {
                        if start > 0 {
                            notes.push(format!(
                                "Rows 0-{}: observed context; posterior columns show the post-merge state, while state columns show the refined recurrent state.",
                                start.saturating_sub(1)
                            ));
                        }
                        if frame_count > start {
                            notes.push(format!(
                                "Rows {}-{}: predictive future; posterior columns are not applicable, and state columns show open-loop recurrent rollout.",
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
                    if let Some(legend) = item.legend.as_ref() {
                        self.write_legend_with_notes(legend, &notes);
                    }
                    if let Some(sidecar_json) = item.sidecar_json.as_deref() {
                        self.write_sidecar_json(sidecar_json, epoch, iteration);
                    }
                    let frames_vec =
                        match frames_tensor.to_data().convert::<f32>().into_vec::<f32>() {
                            Ok(vec) => vec,
                            Err(_) => {
                                return serialized_entry(
                                    "frame_copy_failed".to_string(),
                                    "0".to_string(),
                                );
                            }
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
                    if let Err(err) = fs::create_dir_all(&self.output_dir) {
                        return serialized_entry(format!("mkdir_failed: {err}"), "0".to_string());
                    }
                    let mut saved = 0usize;
                    let mut log_lines = Vec::new();
                    let batch_limit = batch.min(self.remaining_images);
                    let probe_slices = probe_preds
                        .as_ref()
                        .map(|(preds, labels)| (preds.as_slice(), labels.as_slice()));
                    let debug_recon_frames_ref = debug_recon_vec.as_deref();
                    let aux_frames_ref = aux_frames_vec.as_deref();
                    let posterior_patch_steps_ref = posterior_patch_meta
                        .is_some()
                        .then(|| posterior_patch_steps_vec.as_deref())
                        .flatten();
                    let posterior_pca_steps_ref = posterior_pca_meta
                        .is_some()
                        .then(|| posterior_pca_steps_vec.as_deref())
                        .flatten();
                    let patch_steps_ref = patch_meta
                        .is_some()
                        .then(|| patch_steps_vec.as_deref())
                        .flatten();
                    let pca_steps_ref = pca_meta
                        .is_some()
                        .then(|| pca_steps_vec.as_deref())
                        .flatten();
                    let debug_patch_steps_ref = debug_patch_meta
                        .is_some()
                        .then(|| debug_patch_steps_vec.as_deref())
                        .flatten();
                    let debug_pca_steps_ref = debug_pca_meta
                        .is_some()
                        .then(|| debug_pca_steps_vec.as_deref())
                        .flatten();
                    for batch_idx in 0..batch_limit {
                        let mut rows = Vec::with_capacity(frame_count);
                        for frame_idx in 0..frame_count {
                            if let Some(frame) = self.build_video_feature_frame(
                                &frames_vec,
                                debug_recon_frames_ref,
                                aux_frames_ref,
                                posterior_patch_steps_ref,
                                posterior_pca_steps_ref,
                                patch_steps_ref,
                                pca_steps_ref,
                                debug_patch_steps_ref,
                                debug_pca_steps_ref,
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
                                rows.push(self.maybe_upscale_frame(frame, artifact_scale));
                            }
                        }
                        let Some(sheet) = self.stack_temporal_rows(&rows) else {
                            continue;
                        };
                        if let Some(image) =
                            RgbImage::from_vec(sheet.width as u32, sheet.height as u32, sheet.rgb)
                        {
                            let filename = if self.overwrite {
                                format!("sample_{:02}.png", batch_idx)
                            } else if let Some((preds, labels)) = &probe_preds {
                                let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                                let label = labels.get(batch_idx).copied().unwrap_or(-1);
                                format!(
                                    "epoch_{epoch:03}_iter_{iteration:06}_sample_{batch_idx:02}_pred_{pred}_label_{label}.png",
                                )
                            } else {
                                format!(
                                    "epoch_{epoch:03}_iter_{iteration:06}_sample_{batch_idx:02}.png"
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
                                iteration, batch_idx, pred, label, correct
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
                    return serialized_entry(format!("saved={saved}"), saved.to_string());
                }
            }
        }

        let Some(views) = &item.views else {
            return serialized_entry("no_views".to_string(), "0".to_string());
        };
        if item.patch_norms.is_none() && item.pca_rgb.is_none() {
            return serialized_entry("no_patch_data".to_string(), "0".to_string());
        }
        if let Some(legend) = item.legend.as_ref() {
            self.write_legend(legend);
        }
        if let Some(sidecar_json) = item.sidecar_json.as_deref() {
            self.write_sidecar_json(sidecar_json, epoch, iteration);
        }

        let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
        if batch == 0 || view_count == 0 || channels == 0 || height == 0 || width == 0 {
            return serialized_entry("empty".to_string(), "0".to_string());
        }

        let (grid_h, grid_w, _norm_batch) = if let Some(patch_norms) = &item.patch_norms {
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
            return serialized_entry(format!("mkdir_failed: {err}"), "0".to_string());
        }

        let heat_patch_h = height / grid_h;
        let heat_patch_w = width / grid_w;
        if heat_patch_h == 0 || heat_patch_w == 0 {
            return serialized_entry("heatmap_scale_invalid".to_string(), "0".to_string());
        }

        let mut saved = 0usize;
        let mut log_lines = Vec::new();
        let batch_limit = batch.min(self.remaining_images);
        let artifact_scale = item.artifact_scale.max(1);
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
            let frame = self.maybe_upscale_frame(frame, artifact_scale);
            if let Some(image) =
                RgbImage::from_vec(frame.width as u32, frame.height as u32, frame.rgb)
            {
                let filename = if self.overwrite {
                    format!("sample_{:02}.png", batch_idx)
                } else if let Some((preds, labels)) = &probe_preds {
                    let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                    let label = labels.get(batch_idx).copied().unwrap_or(-1);
                    format!(
                        "epoch_{epoch:03}_lejepa_iter_{iteration:06}_sample_{batch_idx:02}_pred_{pred}_label_{label}.png",
                    )
                } else {
                    format!("epoch_{epoch:03}_lejepa_iter_{iteration:06}_sample_{batch_idx:02}.png")
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
                    iteration, batch_idx, pred, label, correct
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

        serialized_entry(format!("saved={saved}"), saved.to_string())
    }
}
