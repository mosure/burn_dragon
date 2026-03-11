use std::io::Write;

use burn_dragon_train::train::constants::LEJEPA_EPS;
use ::image::RgbImage;

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

    pub(super) fn update_image_artifacts(
        &mut self,
        item: &VisionArtifactInput<B>,
        iteration: usize,
    ) -> SerializedEntry {
        let Some(views) = &item.views else {
            return serialized_entry("no_views".to_string(), "0".to_string());
        };
        if item.patch_norms.is_none() && item.pca_rgb.is_none() {
            return serialized_entry("no_patch_data".to_string(), "0".to_string());
        }
        if let Some(legend) = item.legend.as_ref() {
            self.write_legend(legend);
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
            if let Some(image) = RgbImage::from_vec(frame.width as u32, frame.height as u32, frame.rgb)
            {
                let filename = if self.overwrite {
                    format!("sample_{:02}.png", batch_idx)
                } else if let Some((preds, labels)) = &probe_preds {
                    let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                    let label = labels.get(batch_idx).copied().unwrap_or(-1);
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}_pred_{pred}_label_{label}.png",
                        iteration, batch_idx
                    )
                } else {
                    format!("lejepa_iter_{:06}_sample_{:02}.png", iteration, batch_idx)
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
