use anyhow::{Context, Result};
use burn_dragon_train::VisionArtifactOutputMode;
use burn_dragon_train::train::artifacts::{ARTIFACT_DEFAULT_FPS, ArtifactFrame, write_video};
use font8x8::{BASIC_FONTS, UnicodeFonts};
use image::{Rgb, RgbImage, imageops::FilterType};
use serde::Serialize;
use std::fs;
use std::path::{Path, PathBuf};

const TILE_SCALE: usize = 4;
const TILE_GAP: usize = 2;
const CAPTION_HEIGHT: usize = 12;
const LEGEND_HEIGHT: usize = 72;
const FRAME_LEGEND_HEIGHT: usize = 28;

#[derive(Clone, Debug, Serialize)]
pub(crate) struct ArtifactMetrics {
    pub latent_backend: String,
    pub total: f32,
    pub current: f32,
    pub future: f32,
    pub prior: f32,
    pub gaze: f32,
    pub query: f32,
    pub recon: f32,
    pub tokenizer: f32,
    pub tokenizer_recon: f32,
    pub slot_align: f32,
    pub recon_current: f32,
    pub recon_future: f32,
    pub recon_edge: f32,
    pub recon_motion: f32,
    pub current_mae: f32,
    pub future_mae: f32,
    pub current_psnr: f32,
    pub future_psnr: f32,
    pub current_fg_iou: f32,
    pub future_fg_iou: f32,
    pub future_frame_std: f32,
    pub future_latent_std: f32,
    pub future_motion_mse: f32,
    pub future_ref_motion_mse: f32,
    pub future_motion_ratio: f32,
    pub future_stop_mean: f32,
    pub future_stop_std: f32,
    pub future_fixation_motion: f32,
    pub future_confidence_mean: f32,
    pub context_fixation_teacher_l1: f32,
    pub future_fixation_teacher_l1: f32,
}

#[derive(Clone, Debug)]
pub(crate) struct SequenceTensor {
    pub data: Vec<f32>,
    pub batch: usize,
    pub steps: usize,
    pub channels: usize,
    pub height: usize,
    pub width: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct LatentTensor {
    pub data: Vec<f32>,
    pub batch: usize,
    pub steps: usize,
    pub dim: usize,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FixationPointArtifact {
    pub x: f32,
    pub y: f32,
    pub scale: f32,
    pub confidence: f32,
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct FixationSequence {
    pub points: Vec<Vec<Vec<FixationPointArtifact>>>,
    pub stop_probabilities: Vec<Vec<f32>>,
}

#[derive(Clone, Debug)]
pub(crate) struct DreamerArtifactSnapshot {
    pub current_reference: SequenceTensor,
    pub current_reconstruction: SequenceTensor,
    pub future_reference: SequenceTensor,
    pub future_reconstruction: SequenceTensor,
    pub context_latents: LatentTensor,
    pub future_latents: LatentTensor,
    pub teacher_visibility: Option<SequenceTensor>,
    pub teacher_fixations: FixationSequence,
    pub predicted_fixations: FixationSequence,
    pub crop_size: usize,
}

#[derive(Clone, Debug, Serialize)]
struct ArtifactManifest {
    metrics: ArtifactMetrics,
    latent_backend: String,
    crop_size: usize,
    tile_scale: usize,
    current_steps: usize,
    future_steps: usize,
    files: Vec<String>,
    teacher_fixations: FixationSequence,
    predicted_fixations: FixationSequence,
}

pub(crate) fn write_moving_mnist_artifacts(
    output_dir: &Path,
    snapshot: &DreamerArtifactSnapshot,
    metrics: &ArtifactMetrics,
    latent_backend: &str,
) -> Result<PathBuf> {
    fs::create_dir_all(output_dir).context("create dreamer artifact output dir")?;
    let mut files = Vec::new();

    save_sheet(
        output_dir,
        "current_reference.png",
        &sequence_to_image(&snapshot.current_reference, None, None),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "current_reconstruction.png",
        &sequence_to_image(&snapshot.current_reconstruction, None, None),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "future_reference.png",
        &sequence_to_image(&snapshot.future_reference, None, None),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "future_reconstruction.png",
        &sequence_to_image(&snapshot.future_reconstruction, None, None),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "fixation_overlays.png",
        &sequence_to_image(
            &snapshot.current_reference,
            Some((&snapshot.teacher_fixations, [32, 220, 96])),
            Some((&snapshot.predicted_fixations, [235, 92, 70])),
        ),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "autogaze_teacher_label_patches.png",
        &snapshot
            .teacher_visibility
            .as_ref()
            .map(|visibility| {
                dense_visibility_sheet(&snapshot.current_reference, visibility, [32, 220, 96])
            })
            .unwrap_or_else(|| {
                visibility_sheet(
                    &snapshot.current_reference,
                    &snapshot.teacher_fixations,
                    snapshot.crop_size,
                    [32, 220, 96],
                )
            }),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "fovea_saccade_reads.png",
        &visibility_sheet(
            &snapshot.current_reference,
            &snapshot.predicted_fixations,
            snapshot.crop_size,
            [235, 92, 70],
        ),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "current_latent_pca.png",
        &latent_pca_sheet(&snapshot.context_latents, snapshot.current_reference.height),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "future_latent_pca.png",
        &latent_pca_sheet(&snapshot.future_latents, snapshot.future_reference.height),
        &mut files,
    )?;
    save_sheet(
        output_dir,
        "artifact_legend.png",
        &artifact_legend_image(),
        &mut files,
    )?;
    write_rollout_videos(output_dir, snapshot, &mut files)?;

    let manifest = ArtifactManifest {
        metrics: metrics.clone(),
        latent_backend: latent_backend.to_string(),
        crop_size: snapshot.crop_size,
        tile_scale: TILE_SCALE,
        current_steps: snapshot.current_reference.steps,
        future_steps: snapshot.future_reference.steps,
        files,
        teacher_fixations: snapshot.teacher_fixations.clone(),
        predicted_fixations: snapshot.predicted_fixations.clone(),
    };
    let manifest_path = output_dir.join("metrics.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).context("serialize dreamer artifact manifest")?,
    )
    .context("write dreamer artifact manifest")?;
    Ok(output_dir.to_path_buf())
}

fn save_sheet(
    output_dir: &Path,
    name: &str,
    image: &RgbImage,
    files: &mut Vec<String>,
) -> Result<()> {
    image
        .save(output_dir.join(name))
        .with_context(|| format!("save {name}"))?;
    files.push(name.to_string());
    Ok(())
}

fn write_rollout_videos(
    output_dir: &Path,
    snapshot: &DreamerArtifactSnapshot,
    files: &mut Vec<String>,
) -> Result<()> {
    for sample_idx in 0..snapshot.current_reference.batch {
        let frames = dream_rollout_frames(snapshot, sample_idx);
        if frames.is_empty() {
            continue;
        }
        let outcome = write_video(
            output_dir,
            VisionArtifactOutputMode::Mp4,
            true,
            0,
            0,
            sample_idx,
            &frames,
            ARTIFACT_DEFAULT_FPS,
            None,
        )?;
        let extension = outcome
            .path
            .extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("avi");
        let target = output_dir.join(format!("dream_rollout_sample_{sample_idx:02}.{extension}"));
        if outcome.path != target {
            fs::rename(&outcome.path, &target).with_context(|| {
                format!(
                    "rename rollout video {} -> {}",
                    outcome.path.display(),
                    target.display()
                )
            })?;
        }
        if let Some(name) = target.file_name().and_then(|name| name.to_str()) {
            files.push(name.to_string());
        }
    }
    Ok(())
}

fn sequence_to_image(
    sequence: &SequenceTensor,
    overlay_a: Option<(&FixationSequence, [u8; 3])>,
    overlay_b: Option<(&FixationSequence, [u8; 3])>,
) -> RgbImage {
    let cell_w = sequence.width * TILE_SCALE;
    let cell_h = sequence.height * TILE_SCALE;
    let width = sequence.steps * cell_w + sequence.steps.saturating_sub(1) * TILE_GAP;
    let height = sequence.batch * cell_h + sequence.batch.saturating_sub(1) * TILE_GAP;
    let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([12, 12, 12]));
    for batch_idx in 0..sequence.batch {
        for step_idx in 0..sequence.steps {
            let mut tile = frame_rgb(sequence, batch_idx, step_idx);
            if let Some((fixations, color)) = overlay_a {
                draw_fixations(
                    &mut tile,
                    fixations,
                    batch_idx,
                    step_idx,
                    sequence.width,
                    sequence.height,
                    color,
                );
            }
            if let Some((fixations, color)) = overlay_b {
                draw_fixations(
                    &mut tile,
                    fixations,
                    batch_idx,
                    step_idx,
                    sequence.width,
                    sequence.height,
                    color,
                );
            }
            paste(
                &mut canvas,
                &upscale(&tile, sequence.width, sequence.height, TILE_SCALE),
                step_idx * (cell_w + TILE_GAP),
                batch_idx * (cell_h + TILE_GAP),
                cell_w,
                cell_h,
            );
        }
    }
    canvas
}

fn dream_rollout_frames(
    snapshot: &DreamerArtifactSnapshot,
    batch_idx: usize,
) -> Vec<ArtifactFrame> {
    let context_steps = snapshot.current_reference.steps;
    let future_steps = snapshot.future_reference.steps;
    let total_steps = context_steps + future_steps;
    let tile_w = snapshot.current_reference.width * TILE_SCALE;
    let tile_h = snapshot.current_reference.height * TILE_SCALE;
    let gap = TILE_GAP * 2;
    let phase_h = CAPTION_HEIGHT + 6;
    let panel_w = tile_w;
    let panel_h = tile_h + CAPTION_HEIGHT;
    let width = panel_w * 3 + gap * 4;
    let height = phase_h + panel_h * 2 + gap * 4 + FRAME_LEGEND_HEIGHT;
    let mut frames = Vec::with_capacity(total_steps.max(1));
    for step_idx in 0..total_steps {
        let in_context = step_idx < context_steps;
        let phase_color = if in_context {
            [32, 220, 96]
        } else {
            [255, 178, 64]
        };
        let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([10, 10, 12]));
        fill_rect(&mut canvas, 0, 0, width, phase_h, phase_color);
        let phase_label = if in_context {
            format!(
                "OBSERVED STEP {:02}/{:02}  REAL FRAME + TEACHER SIGNALS",
                step_idx + 1,
                total_steps
            )
        } else {
            format!(
                "DREAMED STEP {:02}/{:02}  ROLLOUT ONLY",
                step_idx + 1,
                total_steps
            )
        };
        draw_text(&mut canvas, 6, 5, &phase_label, [12, 12, 12], 1);

        let reference = reference_frame_for_step(snapshot, batch_idx, step_idx);
        let dream = dream_frame_for_step(snapshot, batch_idx, step_idx);
        let teacher_gaze = reference_with_fixations(snapshot, batch_idx, step_idx, [32, 220, 96]);
        let predicted_gaze = reference_with_fixations(snapshot, batch_idx, step_idx, [235, 92, 70]);
        let teacher_visible = reference_with_visibility(snapshot, batch_idx, step_idx);
        let error = error_panel(
            &reference,
            &dream,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
        );

        let top_y = phase_h + gap;
        let bottom_y = phase_h + gap * 2 + panel_h;
        paste_labeled_panel(
            &mut canvas,
            "REFERENCE",
            &reference,
            gap,
            top_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [210, 210, 220],
        );
        paste_labeled_panel(
            &mut canvas,
            "AUTOGAZE LABELS",
            &teacher_gaze,
            gap * 2 + panel_w,
            top_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [32, 220, 96],
        );
        paste_labeled_panel(
            &mut canvas,
            "DREAMER READS",
            &predicted_gaze,
            gap * 3 + panel_w * 2,
            top_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [235, 92, 70],
        );
        paste_labeled_panel(
            &mut canvas,
            "AUTOGAZE VISIBLE",
            &teacher_visible,
            gap,
            bottom_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [32, 220, 96],
        );
        let dream_label = if in_context {
            "FILTER RECON"
        } else {
            "DREAM ROLLOUT"
        };
        paste_labeled_panel(
            &mut canvas,
            dream_label,
            &dream,
            gap * 2 + panel_w,
            bottom_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            if in_context {
                [80, 180, 255]
            } else {
                [255, 178, 64]
            },
        );
        paste_labeled_panel(
            &mut canvas,
            "PIXEL ERROR",
            &error,
            gap * 3 + panel_w * 2,
            bottom_y,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [255, 208, 64],
        );
        draw_frame_legend(&mut canvas, phase_h + panel_h * 2 + gap * 3, width);
        frames.push(ArtifactFrame {
            width,
            height,
            rgb: canvas.into_raw(),
        });
    }
    frames
}

fn paste_labeled_panel(
    canvas: &mut RgbImage,
    label: &str,
    rgb: &[u8],
    x: usize,
    y: usize,
    width: usize,
    height: usize,
    accent: [u8; 3],
) {
    fill_rect(
        canvas,
        x,
        y,
        width * TILE_SCALE,
        CAPTION_HEIGHT,
        [18, 18, 22],
    );
    draw_text(canvas, x + 4, y + 2, label, accent, 1);
    let upscaled = upscale(rgb, width, height, TILE_SCALE);
    paste(
        canvas,
        &upscaled,
        x,
        y + CAPTION_HEIGHT,
        width * TILE_SCALE,
        height * TILE_SCALE,
    );
    draw_rect_outline(
        canvas,
        x,
        y + CAPTION_HEIGHT,
        width * TILE_SCALE,
        height * TILE_SCALE,
        accent,
    );
}

fn reference_frame_for_step(
    snapshot: &DreamerArtifactSnapshot,
    batch_idx: usize,
    step_idx: usize,
) -> Vec<u8> {
    if step_idx < snapshot.current_reference.steps {
        frame_rgb(&snapshot.current_reference, batch_idx, step_idx)
    } else {
        frame_rgb(
            &snapshot.future_reference,
            batch_idx,
            step_idx - snapshot.current_reference.steps,
        )
    }
}

fn dream_frame_for_step(
    snapshot: &DreamerArtifactSnapshot,
    batch_idx: usize,
    step_idx: usize,
) -> Vec<u8> {
    if step_idx < snapshot.current_reconstruction.steps {
        frame_rgb(&snapshot.current_reconstruction, batch_idx, step_idx)
    } else {
        frame_rgb(
            &snapshot.future_reconstruction,
            batch_idx,
            step_idx - snapshot.current_reconstruction.steps,
        )
    }
}

fn reference_with_fixations(
    snapshot: &DreamerArtifactSnapshot,
    batch_idx: usize,
    step_idx: usize,
    color: [u8; 3],
) -> Vec<u8> {
    let mut panel = reference_frame_for_step(snapshot, batch_idx, step_idx);
    let fixations = if color == [32, 220, 96] {
        &snapshot.teacher_fixations
    } else {
        &snapshot.predicted_fixations
    };
    draw_fixations(
        &mut panel,
        fixations,
        batch_idx,
        step_idx,
        snapshot.current_reference.width,
        snapshot.current_reference.height,
        color,
    );
    panel
}

fn reference_with_visibility(
    snapshot: &DreamerArtifactSnapshot,
    batch_idx: usize,
    step_idx: usize,
) -> Vec<u8> {
    let mut panel = reference_frame_for_step(snapshot, batch_idx, step_idx);
    if let Some(teacher_visibility) = snapshot.teacher_visibility.as_ref()
        && step_idx < teacher_visibility.steps
    {
        draw_dense_visibility(
            &mut panel,
            teacher_visibility,
            batch_idx,
            step_idx,
            snapshot.current_reference.width,
            snapshot.current_reference.height,
            [32, 220, 96],
        );
        return panel;
    }
    draw_visible_regions(
        &mut panel,
        &snapshot.teacher_fixations,
        batch_idx,
        step_idx,
        snapshot.current_reference.width,
        snapshot.current_reference.height,
        snapshot.crop_size,
        [32, 220, 96],
    );
    panel
}

fn error_panel(reference: &[u8], dream: &[u8], width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; width * height * 3];
    for pixel in 0..(width * height) {
        let base = pixel * 3;
        let dr = (reference[base] as f32 - dream[base] as f32).abs() / 255.0;
        let dg = (reference[base + 1] as f32 - dream[base + 1] as f32).abs() / 255.0;
        let db = (reference[base + 2] as f32 - dream[base + 2] as f32).abs() / 255.0;
        let diff = ((dr + dg + db) / 3.0).clamp(0.0, 1.0);
        rgb[base] = (255.0 * diff).round() as u8;
        rgb[base + 1] = (200.0 * diff.sqrt()).round() as u8;
        rgb[base + 2] = (40.0 * diff.powf(0.35)).round() as u8;
    }
    rgb
}

fn fill_rect(
    canvas: &mut RgbImage,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    color: [u8; 3],
) {
    let max_x = (x0 + width).min(canvas.width() as usize);
    let max_y = (y0 + height).min(canvas.height() as usize);
    for y in y0..max_y {
        for x in x0..max_x {
            canvas.put_pixel(x as u32, y as u32, Rgb(color));
        }
    }
}

fn draw_rect_outline(
    canvas: &mut RgbImage,
    x0: usize,
    y0: usize,
    width: usize,
    height: usize,
    color: [u8; 3],
) {
    if width == 0 || height == 0 {
        return;
    }
    let x1 = x0 + width.saturating_sub(1);
    let y1 = y0 + height.saturating_sub(1);
    for x in x0..=x1.min(canvas.width() as usize - 1) {
        if y0 < canvas.height() as usize {
            canvas.put_pixel(x as u32, y0 as u32, Rgb(color));
        }
        if y1 < canvas.height() as usize {
            canvas.put_pixel(x as u32, y1 as u32, Rgb(color));
        }
    }
    for y in y0..=y1.min(canvas.height() as usize - 1) {
        if x0 < canvas.width() as usize {
            canvas.put_pixel(x0 as u32, y as u32, Rgb(color));
        }
        if x1 < canvas.width() as usize {
            canvas.put_pixel(x1 as u32, y as u32, Rgb(color));
        }
    }
}

fn draw_text(canvas: &mut RgbImage, x: usize, y: usize, text: &str, color: [u8; 3], scale: usize) {
    let mut cursor_x = x;
    for ch in text.chars() {
        if ch == ' ' {
            cursor_x += 4 * scale;
            continue;
        }
        if let Some(glyph) = BASIC_FONTS.get(ch.to_ascii_uppercase()) {
            for (row_idx, row) in glyph.iter().enumerate() {
                for col_idx in 0..8 {
                    if (row >> col_idx) & 1 == 0 {
                        continue;
                    }
                    for dy in 0..scale {
                        for dx in 0..scale {
                            let px = cursor_x + col_idx * scale + dx;
                            let py = y + row_idx * scale + dy;
                            if px < canvas.width() as usize && py < canvas.height() as usize {
                                canvas.put_pixel(px as u32, py as u32, Rgb(color));
                            }
                        }
                    }
                }
            }
        }
        cursor_x += 8 * scale;
    }
}

fn artifact_legend_image() -> RgbImage {
    let width = 520usize;
    let height = LEGEND_HEIGHT + 12;
    let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([10, 10, 12]));
    draw_text(
        &mut canvas,
        8,
        8,
        "GREEN BOXES = OFFICIAL AUTOGAZE FIXATIONS",
        [32, 220, 96],
        1,
    );
    draw_text(
        &mut canvas,
        8,
        24,
        "RED BOXES = DREAMER PREDICTED FIXATIONS",
        [235, 92, 70],
        1,
    );
    draw_text(
        &mut canvas,
        8,
        40,
        "GREEN MASK = REGIONS AUTOGAZE EXPOSES TO THE ENCODER",
        [32, 220, 96],
        1,
    );
    draw_text(
        &mut canvas,
        8,
        56,
        "TOP BAR: GREEN = OBSERVED CONTEXT, ORANGE = DREAMED FUTURE",
        [255, 178, 64],
        1,
    );
    draw_text(
        &mut canvas,
        8,
        72,
        "PCA PLOTS USE DOTS ONLY: BLUE = EARLY, ORANGE = LATE",
        [120, 200, 255],
        1,
    );
    canvas
}

fn visibility_sheet(
    sequence: &SequenceTensor,
    fixations: &FixationSequence,
    crop_size: usize,
    color: [u8; 3],
) -> RgbImage {
    let cell_w = sequence.width * TILE_SCALE;
    let cell_h = sequence.height * TILE_SCALE;
    let width = sequence.steps * cell_w + sequence.steps.saturating_sub(1) * TILE_GAP;
    let height = sequence.batch * cell_h + sequence.batch.saturating_sub(1) * TILE_GAP;
    let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([12, 12, 12]));
    for batch_idx in 0..sequence.batch {
        for step_idx in 0..sequence.steps {
            let mut tile = frame_rgb(sequence, batch_idx, step_idx);
            draw_visible_regions(
                &mut tile,
                fixations,
                batch_idx,
                step_idx,
                sequence.width,
                sequence.height,
                crop_size,
                color,
            );
            paste(
                &mut canvas,
                &upscale(&tile, sequence.width, sequence.height, TILE_SCALE),
                step_idx * (cell_w + TILE_GAP),
                batch_idx * (cell_h + TILE_GAP),
                cell_w,
                cell_h,
            );
        }
    }
    canvas
}

fn dense_visibility_sheet(
    reference: &SequenceTensor,
    visibility: &SequenceTensor,
    color: [u8; 3],
) -> RgbImage {
    let cell_w = reference.width * TILE_SCALE;
    let cell_h = reference.height * TILE_SCALE;
    let width = reference.steps * cell_w + reference.steps.saturating_sub(1) * TILE_GAP;
    let height = reference.batch * cell_h + reference.batch.saturating_sub(1) * TILE_GAP;
    let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([12, 12, 12]));
    for batch_idx in 0..reference.batch {
        for step_idx in 0..reference.steps {
            let mut tile = frame_rgb(reference, batch_idx, step_idx);
            draw_dense_visibility(
                &mut tile,
                visibility,
                batch_idx,
                step_idx,
                reference.width,
                reference.height,
                color,
            );
            paste(
                &mut canvas,
                &upscale(&tile, reference.width, reference.height, TILE_SCALE),
                step_idx * (cell_w + TILE_GAP),
                batch_idx * (cell_h + TILE_GAP),
                cell_w,
                cell_h,
            );
        }
    }
    canvas
}

fn latent_pca_sheet(latents: &LatentTensor, tile_size: usize) -> RgbImage {
    let panel = tile_size.max(24) * TILE_SCALE;
    let width = panel;
    let header = CAPTION_HEIGHT + 18;
    let height = header + latents.batch * panel + latents.batch.saturating_sub(1) * TILE_GAP;
    let mut canvas = RgbImage::from_pixel(width as u32, height as u32, Rgb([10, 10, 12]));
    draw_text(&mut canvas, 4, 4, "LATENT PCA", [210, 210, 220], 1);
    draw_text(
        &mut canvas,
        4,
        16,
        "DOTS: EARLY BLUE -> LATE ORANGE",
        [120, 200, 255],
        1,
    );
    for batch_idx in 0..latents.batch {
        let tile = latent_trajectory_tile(latents, batch_idx, panel);
        paste(
            &mut canvas,
            &tile,
            0,
            header + batch_idx * (panel + TILE_GAP),
            panel,
            panel,
        );
    }
    canvas
}

fn draw_visible_regions(
    rgb: &mut [u8],
    fixations: &FixationSequence,
    batch_idx: usize,
    step_idx: usize,
    width: usize,
    height: usize,
    crop_size: usize,
    color: [u8; 3],
) {
    if batch_idx >= fixations.points.len() || step_idx >= fixations.points[batch_idx].len() {
        return;
    }
    for point in &fixations.points[batch_idx][step_idx] {
        let window = ((crop_size as f32) * point.scale.clamp(0.25, 1.0)).round() as isize;
        let window = window.max(2).min(width.max(height) as isize);
        let center_x = (point.x.clamp(0.0, 1.0) * width as f32).round() as isize;
        let center_y = (point.y.clamp(0.0, 1.0) * height as f32).round() as isize;
        let x0 = (center_x - window / 2).clamp(0, width.saturating_sub(1) as isize) as usize;
        let x1 = (center_x + window / 2).clamp(0, width.saturating_sub(1) as isize) as usize;
        let y0 = (center_y - window / 2).clamp(0, height.saturating_sub(1) as isize) as usize;
        let y1 = (center_y + window / 2).clamp(0, height.saturating_sub(1) as isize) as usize;
        let fill_alpha = (0.12 + 0.25 * point.confidence.clamp(0.0, 1.0)).clamp(0.08, 0.45);
        for y in y0..=y1 {
            for x in x0..=x1 {
                blend_pixel(rgb, width, x, y, color, fill_alpha);
            }
        }
        for x in x0..=x1 {
            paint_pixel(rgb, width, x, y0, color);
            paint_pixel(rgb, width, x, y1, color);
        }
        for y in y0..=y1 {
            paint_pixel(rgb, width, x0, y, color);
            paint_pixel(rgb, width, x1, y, color);
        }
        let cx = center_x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let cy = center_y.clamp(0, height.saturating_sub(1) as isize) as usize;
        paint_pixel(rgb, width, cx, cy, [255, 255, 255]);
    }
}

fn draw_dense_visibility(
    rgb: &mut [u8],
    visibility: &SequenceTensor,
    batch_idx: usize,
    step_idx: usize,
    width: usize,
    height: usize,
    color: [u8; 3],
) {
    if batch_idx >= visibility.batch || step_idx >= visibility.steps {
        return;
    }
    let channels = visibility.channels.max(1);
    let frame_area = visibility.height * visibility.width;
    let base = (batch_idx * visibility.steps + step_idx) * channels * frame_area;
    for y in 0..height.min(visibility.height) {
        for x in 0..width.min(visibility.width) {
            let idx = base + y * visibility.width + x;
            let alpha = visibility.data[idx].clamp(0.0, 1.0) * 0.65;
            if alpha > 1.0e-4 {
                blend_pixel(rgb, width, x, y, color, alpha);
            }
        }
    }
}

fn blend_pixel(rgb: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 3], alpha: f32) {
    let offset = (y * width + x) * 3;
    for channel in 0..3 {
        let base = rgb[offset + channel] as f32;
        let mixed = base * (1.0 - alpha) + color[channel] as f32 * alpha;
        rgb[offset + channel] = mixed.round().clamp(0.0, 255.0) as u8;
    }
}

fn latent_trajectory_tile(latents: &LatentTensor, batch_idx: usize, panel: usize) -> Vec<u8> {
    let mut tile = vec![18u8; panel * panel * 3];
    let margin = (panel / 12).max(8);
    let inner = panel.saturating_sub(2 * margin).max(1);
    draw_panel_grid(&mut tile, panel, margin);
    let coords = latent_sequence_projection(latents, batch_idx);
    if coords.is_empty() {
        return tile;
    }
    let (mut min_x, mut max_x, mut min_y, mut max_y) = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    for (x, y) in &coords {
        min_x = min_x.min(*x);
        max_x = max_x.max(*x);
        min_y = min_y.min(*y);
        max_y = max_y.max(*y);
    }
    let span_x = (max_x - min_x).max(1.0e-5);
    let span_y = (max_y - min_y).max(1.0e-5);
    let mapped: Vec<(usize, usize)> = coords
        .iter()
        .enumerate()
        .map(|(step_idx, (x, y))| {
            if coords.len() == 1 {
                return (panel / 2, panel / 2);
            }
            let x01 = (*x - min_x) / span_x;
            let y01 = (*y - min_y) / span_y;
            let px = margin + ((x01 * inner as f32).round() as usize).min(inner);
            let py = margin + (((1.0 - y01) * inner as f32).round() as usize).min(inner);
            if step_idx == 0 && span_x <= 1.0e-5 && span_y <= 1.0e-5 {
                (panel / 2, panel / 2)
            } else {
                (
                    px.min(panel.saturating_sub(1)),
                    py.min(panel.saturating_sub(1)),
                )
            }
        })
        .collect();
    for (step_idx, (x, y)) in mapped.into_iter().enumerate() {
        let color = step_color(step_idx, coords.len());
        let radius = if step_idx == 0 || step_idx + 1 == coords.len() {
            3
        } else {
            2
        };
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let px = x as isize + dx;
                let py = y as isize + dy;
                if px >= 0 && py >= 0 && (px as usize) < panel && (py as usize) < panel {
                    let alpha = if dx.abs() + dy.abs() <= radius {
                        1.0
                    } else {
                        0.35
                    };
                    blend_pixel(&mut tile, panel, px as usize, py as usize, color, alpha);
                }
            }
        }
    }
    tile
}

fn draw_panel_grid(tile: &mut [u8], panel: usize, margin: usize) {
    let grid = [42, 42, 50];
    let axis = [84, 84, 96];
    for y in [margin, panel / 2, panel.saturating_sub(margin + 1)] {
        for x in margin..panel.saturating_sub(margin) {
            paint_pixel(tile, panel, x, y, if y == panel / 2 { axis } else { grid });
        }
    }
    for x in [margin, panel / 2, panel.saturating_sub(margin + 1)] {
        for y in margin..panel.saturating_sub(margin) {
            paint_pixel(tile, panel, x, y, if x == panel / 2 { axis } else { grid });
        }
    }
}

fn latent_sequence_projection(latents: &LatentTensor, batch_idx: usize) -> Vec<(f32, f32)> {
    if latents.steps == 0 || latents.dim == 0 || batch_idx >= latents.batch {
        return Vec::new();
    }
    let base = batch_idx * latents.steps * latents.dim;
    let slice = &latents.data[base..base + latents.steps * latents.dim];
    let mut mean = vec![0.0f32; latents.dim];
    for step_idx in 0..latents.steps {
        let offset = step_idx * latents.dim;
        for dim_idx in 0..latents.dim {
            mean[dim_idx] += slice[offset + dim_idx];
        }
    }
    for value in &mut mean {
        *value /= latents.steps.max(1) as f32;
    }

    let mut cov = vec![0.0f32; latents.dim * latents.dim];
    for step_idx in 0..latents.steps {
        let offset = step_idx * latents.dim;
        for i in 0..latents.dim {
            let xi = slice[offset + i] - mean[i];
            for j in 0..latents.dim {
                cov[i * latents.dim + j] += xi * (slice[offset + j] - mean[j]);
            }
        }
    }
    let inv_steps = 1.0 / latents.steps.max(1) as f32;
    for value in &mut cov {
        *value *= inv_steps;
    }

    let mut components = vec![vec![0.0f32; latents.dim]; 2];
    let mut vec = vec![0.0f32; latents.dim];
    let mut work = vec![0.0f32; latents.dim];
    for comp in 0..2 {
        vec.fill(0.0);
        vec[comp.min(latents.dim.saturating_sub(1))] = 1.0;
        for _ in 0..32 {
            for i in 0..latents.dim {
                let mut sum = 0.0f32;
                for j in 0..latents.dim {
                    sum += cov[i * latents.dim + j] * vec[j];
                }
                work[i] = sum;
            }
            if normalize(&mut work) < 1.0e-6 {
                break;
            }
            std::mem::swap(&mut vec, &mut work);
        }
        components[comp].copy_from_slice(&vec);
        let mut lambda = 0.0f32;
        for i in 0..latents.dim {
            let mut sum = 0.0f32;
            for j in 0..latents.dim {
                sum += cov[i * latents.dim + j] * vec[j];
            }
            lambda += vec[i] * sum;
        }
        for i in 0..latents.dim {
            for j in 0..latents.dim {
                cov[i * latents.dim + j] -= lambda * vec[i] * vec[j];
            }
        }
    }

    (0..latents.steps)
        .map(|step_idx| {
            let offset = step_idx * latents.dim;
            let x = (0..latents.dim)
                .map(|dim_idx| (slice[offset + dim_idx] - mean[dim_idx]) * components[0][dim_idx])
                .sum::<f32>();
            let y = (0..latents.dim)
                .map(|dim_idx| (slice[offset + dim_idx] - mean[dim_idx]) * components[1][dim_idx])
                .sum::<f32>();
            (x, y)
        })
        .collect()
}

fn step_color(step_idx: usize, steps: usize) -> [u8; 3] {
    let t = if steps <= 1 {
        0.0
    } else {
        step_idx as f32 / (steps - 1) as f32
    };
    [
        (56.0 + 180.0 * t).round() as u8,
        (220.0 - 120.0 * t).round() as u8,
        (240.0 - 170.0 * t).round() as u8,
    ]
}

fn frame_rgb(sequence: &SequenceTensor, batch_idx: usize, step_idx: usize) -> Vec<u8> {
    let frame_stride = sequence.channels * sequence.height * sequence.width;
    let base = (batch_idx * sequence.steps + step_idx) * frame_stride;
    let mut rgb = vec![0u8; sequence.height * sequence.width * 3];
    for y in 0..sequence.height {
        for x in 0..sequence.width {
            let pixel = y * sequence.width + x;
            let dst = pixel * 3;
            if sequence.channels == 1 {
                let value = denormalize(sequence.data[base + pixel]);
                rgb[dst] = value;
                rgb[dst + 1] = value;
                rgb[dst + 2] = value;
            } else {
                for channel in 0..3 {
                    let offset = base + channel * sequence.height * sequence.width + pixel;
                    rgb[dst + channel] = denormalize(sequence.data[offset]);
                }
            }
        }
    }
    rgb
}

fn draw_fixations(
    rgb: &mut [u8],
    fixations: &FixationSequence,
    batch_idx: usize,
    step_idx: usize,
    width: usize,
    height: usize,
    color: [u8; 3],
) {
    if batch_idx >= fixations.points.len() || step_idx >= fixations.points[batch_idx].len() {
        return;
    }
    for point in &fixations.points[batch_idx][step_idx] {
        let size = ((width.min(height) as f32) * point.scale.clamp(0.08, 1.0)).round() as isize;
        let size = size.max(2);
        let center_x = (point.x.clamp(0.0, 1.0) * width as f32).round() as isize;
        let center_y = (point.y.clamp(0.0, 1.0) * height as f32).round() as isize;
        let x0 = (center_x - size / 2).clamp(0, width.saturating_sub(1) as isize) as usize;
        let x1 = (center_x + size / 2).clamp(0, width.saturating_sub(1) as isize) as usize;
        let y0 = (center_y - size / 2).clamp(0, height.saturating_sub(1) as isize) as usize;
        let y1 = (center_y + size / 2).clamp(0, height.saturating_sub(1) as isize) as usize;
        for x in x0..=x1 {
            paint_pixel(rgb, width, x, y0, color);
            paint_pixel(rgb, width, x, y1, color);
        }
        for y in y0..=y1 {
            paint_pixel(rgb, width, x0, y, color);
            paint_pixel(rgb, width, x1, y, color);
        }
        let cx = center_x.clamp(0, width.saturating_sub(1) as isize) as usize;
        let cy = center_y.clamp(0, height.saturating_sub(1) as isize) as usize;
        paint_pixel(rgb, width, cx, cy, [255, 255, 255]);
    }
}

fn draw_frame_legend(canvas: &mut RgbImage, y: usize, width: usize) {
    fill_rect(canvas, 0, y, width, FRAME_LEGEND_HEIGHT, [14, 14, 18]);
    fill_rect(canvas, 8, y + 8, 10, 10, [32, 220, 96]);
    draw_text(
        canvas,
        24,
        y + 8,
        "GREEN BOX = OFFICIAL AUTOGAZE",
        [210, 210, 220],
        1,
    );
    fill_rect(canvas, 210, y + 8, 10, 10, [235, 92, 70]);
    draw_text(canvas, 226, y + 8, "RED BOX = DREAMER", [210, 210, 220], 1);
    fill_rect(canvas, 356, y + 8, 10, 10, [255, 178, 64]);
    draw_text(
        canvas,
        372,
        y + 8,
        "ORANGE FRAMES = DREAMED FUTURE",
        [210, 210, 220],
        1,
    );
}

fn paint_pixel(rgb: &mut [u8], width: usize, x: usize, y: usize, color: [u8; 3]) {
    let offset = (y * width + x) * 3;
    rgb[offset] = color[0];
    rgb[offset + 1] = color[1];
    rgb[offset + 2] = color[2];
}

fn upscale(rgb: &[u8], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let image =
        RgbImage::from_vec(width as u32, height as u32, rgb.to_vec()).expect("valid rgb tile");
    image::imageops::resize(
        &image,
        (width * scale) as u32,
        (height * scale) as u32,
        FilterType::Nearest,
    )
    .into_raw()
}

fn paste(
    canvas: &mut RgbImage,
    tile: &[u8],
    offset_x: usize,
    offset_y: usize,
    width: usize,
    height: usize,
) {
    for y in 0..height {
        for x in 0..width {
            let src = (y * width + x) * 3;
            canvas.put_pixel(
                (offset_x + x) as u32,
                (offset_y + y) as u32,
                Rgb([tile[src], tile[src + 1], tile[src + 2]]),
            );
        }
    }
}

fn denormalize(value: f32) -> u8 {
    (((value * 0.5) + 0.5).clamp(0.0, 1.0) * 255.0).round() as u8
}

fn normalize(values: &mut [f32]) -> f32 {
    let norm = values.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 1.0e-6 {
        for value in values {
            *value /= norm;
        }
    }
    norm
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn moving_mnist_artifacts_write_expected_files() {
        let temp = tempdir().expect("temp dir");
        let sequence = SequenceTensor {
            data: vec![0.0; 2 * 2 * 1 * 8 * 8],
            batch: 2,
            steps: 2,
            channels: 1,
            height: 8,
            width: 8,
        };
        let latents = LatentTensor {
            data: (0..(2 * 2 * 8)).map(|idx| idx as f32 * 0.01).collect(),
            batch: 2,
            steps: 2,
            dim: 8,
        };
        let fixations = FixationSequence {
            points: vec![
                vec![
                    vec![FixationPointArtifact {
                        x: 0.5,
                        y: 0.5,
                        scale: 0.25,
                        confidence: 1.0,
                    }],
                    vec![FixationPointArtifact {
                        x: 0.4,
                        y: 0.6,
                        scale: 0.3,
                        confidence: 0.8,
                    }],
                ],
                vec![
                    vec![FixationPointArtifact {
                        x: 0.5,
                        y: 0.5,
                        scale: 0.25,
                        confidence: 1.0,
                    }],
                    vec![FixationPointArtifact {
                        x: 0.6,
                        y: 0.4,
                        scale: 0.3,
                        confidence: 0.8,
                    }],
                ],
            ],
            stop_probabilities: vec![vec![0.1, 0.2], vec![0.3, 0.4]],
        };
        let snapshot = DreamerArtifactSnapshot {
            current_reference: sequence.clone(),
            current_reconstruction: sequence.clone(),
            future_reference: sequence.clone(),
            future_reconstruction: sequence,
            context_latents: latents.clone(),
            future_latents: latents,
            teacher_visibility: None,
            teacher_fixations: fixations.clone(),
            predicted_fixations: fixations,
            crop_size: 6,
        };
        let metrics = ArtifactMetrics {
            latent_backend: "pooled".to_string(),
            total: 1.0,
            current: 0.8,
            future: 0.7,
            prior: 0.2,
            gaze: 0.1,
            query: 0.05,
            recon: 0.3,
            tokenizer: 0.15,
            tokenizer_recon: 0.12,
            slot_align: 0.09,
            recon_current: 0.2,
            recon_future: 0.4,
            recon_edge: 0.1,
            recon_motion: 0.05,
            current_mae: 0.05,
            future_mae: 0.08,
            current_psnr: 24.0,
            future_psnr: 18.0,
            current_fg_iou: 0.7,
            future_fg_iou: 0.5,
            future_frame_std: 0.12,
            future_latent_std: 0.42,
            future_motion_mse: 0.06,
            future_ref_motion_mse: 0.08,
            future_motion_ratio: 0.75,
            future_stop_mean: 0.2,
            future_stop_std: 0.04,
            future_fixation_motion: 0.07,
            future_confidence_mean: 0.8,
            context_fixation_teacher_l1: 0.02,
            future_fixation_teacher_l1: 0.05,
        };
        write_moving_mnist_artifacts(temp.path(), &snapshot, &metrics, "pooled")
            .expect("write artifacts");
        for name in [
            "current_reference.png",
            "current_reconstruction.png",
            "future_reference.png",
            "future_reconstruction.png",
            "fixation_overlays.png",
            "autogaze_teacher_label_patches.png",
            "fovea_saccade_reads.png",
            "current_latent_pca.png",
            "future_latent_pca.png",
            "artifact_legend.png",
            "metrics.json",
        ] {
            assert!(
                temp.path().join(name).is_file(),
                "expected artifact file {}",
                temp.path().join(name).display()
            );
        }
        let video_count = fs::read_dir(temp.path())
            .expect("read artifact dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                matches!(
                    entry.path().extension().and_then(|ext| ext.to_str()),
                    Some("mp4") | Some("avi")
                )
            })
            .count();
        assert!(
            video_count >= 2,
            "expected rollout videos for each sample in {}",
            temp.path().display()
        );
    }
}
