use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon_train::VisionArtifactOutputMode;
use burn_dragon_train::train::artifacts::ArtifactFrame;
use burn_train::metric::{Metric, MetricMetadata, MetricName, SerializedEntry};

use super::VisionArtifactInput;

mod image;
#[cfg(test)]
mod tests;
mod video;

fn serialized_entry(
    formatted: impl Into<String>,
    serialized: impl Into<String>,
) -> SerializedEntry {
    SerializedEntry::new(formatted.into(), serialized.into())
}

fn metric_iteration(metadata: &MetricMetadata) -> usize {
    metadata.iteration.unwrap_or(0)
}

fn should_emit_metric(metadata: &MetricMetadata, every: usize) -> bool {
    every <= 1
        || metadata
            .iteration
            .is_some_and(|iteration| iteration.is_multiple_of(every))
}

fn metric_epoch(metadata: &MetricMetadata) -> usize {
    metadata.global_progress.items_processed
}

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

    fn write_legend_with_notes(&self, legend: &[String], notes: &[String]) {
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
        for note in notes {
            if !note.is_empty() {
                contents.push('\n');
                contents.push_str(note);
            }
        }
        let path = self.output_dir.join("vision_artifacts_key.txt");
        let _ = fs::write(path, contents);
    }

    fn write_legend(&self, legend: &[String]) {
        self.write_legend_with_notes(legend, &[]);
    }

    fn maybe_upscale_frame(&self, frame: ArtifactFrame, scale: usize) -> ArtifactFrame {
        frame.upscale_nearest(scale.max(1))
    }
}

impl<B: BackendTrait> Metric for VisionArtifactMetric<B> {
    type Input = VisionArtifactInput<B>;

    fn name(&self) -> MetricName {
        Arc::clone(&self.name)
    }

    fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> SerializedEntry {
        let iteration = metric_iteration(metadata);
        let epoch = metric_epoch(metadata);
        if self.every == 0 {
            return serialized_entry("disabled".to_string(), "0".to_string());
        }
        if !should_emit_metric(metadata, self.every) {
            return serialized_entry("skip".to_string(), "0".to_string());
        }
        if self.last_epoch != Some(epoch) {
            self.last_epoch = Some(epoch);
            self.remaining_images = self.max_images;
        }
        if self.remaining_images == 0 {
            return serialized_entry("budget_exhausted".to_string(), "0".to_string());
        }

        if self.output_mode != VisionArtifactOutputMode::Images {
            return self.update_video_artifacts(item, iteration);
        }

        self.update_image_artifacts(item, iteration)
    }

    fn clear(&mut self) {}
}
