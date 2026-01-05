use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;

use burn::tensor::{Int, Tensor};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_train::metric::{Adaptor, ItemLazy, LossInput};

use super::LEJEPA_EPS;
use super::artifacts::{collect_frames, write_video};
use crate::VisionArtifactOutputMode;

pub(crate) struct LanguageModelOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
}

impl<B: BackendTrait> LanguageModelOutput<B> {
    pub(crate) fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: BackendTrait> ItemLazy for LanguageModelOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

pub(crate) struct LanguageModelTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LanguageModelTrainItem<B> {
    pub(crate) fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: AutodiffBackend> ItemLazy for LanguageModelTrainItem<B> {
    type ItemSync = LanguageModelOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        LanguageModelOutput::new(self.loss.inner())
    }
}

#[derive(Clone)]
pub(crate) struct VisionArtifactInput<B: BackendTrait> {
    pub(crate) views: Option<Tensor<B, 5>>,
    pub(crate) frames: Option<Tensor<B, 5>>,
    pub(crate) patch_norms: Option<Tensor<B, 3>>,
    pub(crate) probe_logits: Option<Tensor<B, 2>>,
    pub(crate) labels: Option<Tensor<B, 1, Int>>,
}

impl<B: BackendTrait> VisionArtifactInput<B> {
    pub(crate) fn empty() -> Self {
        Self {
            views: None,
            frames: None,
            patch_norms: None,
            probe_logits: None,
            labels: None,
        }
    }
}

#[derive(Clone)]
pub(crate) struct VisionOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionOutput<B> {
    pub(crate) fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
        artifacts: Option<VisionArtifactInput<B>>,
    ) -> Self {
        Self {
            loss,
            inv_loss,
            sigreg_loss,
            recon_loss,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }
}

impl<B: BackendTrait> ItemLazy for VisionOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

#[derive(Clone)]
pub(crate) struct InvLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> InvLossInput<B> {
    pub(crate) fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub(crate) struct SigRegLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SigRegLossInput<B> {
    pub(crate) fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub(crate) struct ReconLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ReconLossInput<B> {
    pub(crate) fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub(crate) struct ProbeLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeLossInput<B> {
    pub(crate) fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub(crate) struct ProbeAccInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeAccInput<B> {
    pub(crate) fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

impl<B: BackendTrait> Adaptor<InvLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> InvLossInput<B> {
        InvLossInput::new(self.inv_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SigRegLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> SigRegLossInput<B> {
        SigRegLossInput::new(self.sigreg_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconLossInput<B> {
        ReconLossInput::new(self.recon_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeLossInput<B> {
        ProbeLossInput::new(self.probe_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeAccInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeAccInput<B> {
        ProbeAccInput::new(self.probe_acc.clone())
    }
}

impl<B: BackendTrait> Adaptor<VisionArtifactInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> VisionArtifactInput<B> {
        self.artifacts.clone().unwrap_or_else(VisionArtifactInput::empty)
    }
}

pub(crate) struct VisionTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
}

impl<B: AutodiffBackend> VisionTrainItem<B> {
    pub(crate) fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
    ) -> Self {
        Self {
            loss,
            inv_loss,
            sigreg_loss,
            recon_loss,
            probe_loss,
            probe_acc,
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for VisionTrainItem<B> {
    type ItemSync = VisionOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        VisionOutput::new(
            self.loss.inner(),
            self.inv_loss.inner(),
            self.sigreg_loss.inner(),
            self.recon_loss.inner(),
            self.probe_loss.inner(),
            self.probe_acc.inner(),
            None,
        )
    }
}

pub(crate) trait ScalarValue<B: BackendTrait> {
    fn value(&self) -> Tensor<B, 1>;
}

impl<B: BackendTrait> ScalarValue<B> for InvLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SigRegLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ReconLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeAccInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

pub(crate) struct ScalarMetric<B: BackendTrait, I: ScalarValue<B>> {
    name: Arc<String>,
    last: f64,
    _marker: std::marker::PhantomData<(B, I)>,
}

impl<B: BackendTrait, I: ScalarValue<B>> Clone for ScalarMetric<B, I> {
    fn clone(&self) -> Self {
        Self {
            name: Arc::clone(&self.name),
            last: self.last,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B>> ScalarMetric<B, I> {
    pub(crate) fn new(name: &str) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            last: 0.0,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Metric
    for ScalarMetric<B, I>
{
    type Input = I;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        _metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        let value = item
            .value()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("metric value");
        let value = value.first().copied().unwrap_or(0.0) as f64;
        self.last = value;
        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            burn_train::metric::format_float(value, 4),
            value.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last = 0.0;
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Numeric
    for ScalarMetric<B, I>
{
    fn value(&self) -> burn_train::metric::NumericEntry {
        burn_train::metric::NumericEntry::Value(self.last)
    }
}

#[derive(Clone)]
pub(crate) struct VisionArtifactMetric<B: BackendTrait> {
    name: Arc<String>,
    output_dir: PathBuf,
    every: usize,
    output_mode: VisionArtifactOutputMode,
    mean: [f32; 3],
    std: [f32; 3],
    overwrite: bool,
    _marker: std::marker::PhantomData<B>,
}

impl<B: BackendTrait> VisionArtifactMetric<B> {
    pub(crate) fn new(
        output_dir: PathBuf,
        every: usize,
        output_mode: VisionArtifactOutputMode,
        mean: [f32; 3],
        std: [f32; 3],
        overwrite: bool,
    ) -> Self {
        Self {
            name: Arc::new("vision_artifacts".to_string()),
            output_dir,
            every,
            output_mode,
            mean,
            std,
            overwrite,
            _marker: std::marker::PhantomData,
        }
    }

    fn denormalize_channel(&self, value: f32, channel: usize) -> u8 {
        let mut value = value * self.std[channel] + self.mean[channel];
        value = value.clamp(0.0, 1.0);
        (value * 255.0).round() as u8
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
        if metadata.iteration % self.every != 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "skip".to_string(),
                "0".to_string(),
            );
        }
        if self.output_mode != VisionArtifactOutputMode::Images {
            let frames_tensor = item.frames.as_ref().or(item.views.as_ref());
            let Some(frames_tensor) = frames_tensor else {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "no_frames".to_string(),
                    "0".to_string(),
                );
            };
            let [batch, frame_count, channels, height, width] =
                frames_tensor.shape().dims::<5>();
            if batch == 0 || frame_count == 0 || channels == 0 || height == 0 || width == 0 {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "empty_frames".to_string(),
                    "0".to_string(),
                );
            }
            let frames_vec = match frames_tensor
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
            {
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
            for batch_idx in 0..batch {
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
        let Some(patch_norms) = &item.patch_norms else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "no_patch_norms".to_string(),
                "0".to_string(),
            );
        };

        let [batch, view_count, channels, height, width] = views.shape().dims::<5>();
        if batch == 0 || view_count == 0 || channels == 0 || height == 0 || width == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "empty".to_string(),
                "0".to_string(),
            );
        }

        let [norm_batch, grid_h, grid_w] = patch_norms.shape().dims::<3>();
        if norm_batch == 0 || grid_h == 0 || grid_w == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "empty_norms".to_string(),
                "0".to_string(),
            );
        }

        let views_vec = match views
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        {
            Ok(vec) => vec,
            Err(_) => {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "view_copy_failed".to_string(),
                    "0".to_string(),
                );
            }
        };
        let patch_vec = match patch_norms
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
        {
            Ok(vec) => vec,
            Err(_) => {
                return burn_train::metric::MetricEntry::new(
                    Arc::clone(&self.name),
                    "patch_copy_failed".to_string(),
                    "0".to_string(),
                );
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
        let width_total = width * (view_count + 1);
        for batch_idx in 0..batch {
            let mut canvas = vec![0u8; width_total * height * 3];
            for view_idx in 0..view_count {
                for y in 0..height {
                    for x in 0..width {
                        let base = (((batch_idx * view_count + view_idx) * channels + 0) * height
                            + y)
                            * width
                            + x;
                        let r = self.denormalize_channel(views_vec[base], 0);
                        let g = self.denormalize_channel(
                            views_vec[base + height * width],
                            1,
                        );
                        let b = self.denormalize_channel(
                            views_vec[base + 2 * height * width],
                            2,
                        );
                        let out_x = view_idx * width + x;
                        let offset = (y * width_total + out_x) * 3;
                        canvas[offset] = r;
                        canvas[offset + 1] = g;
                        canvas[offset + 2] = b;
                    }
                }
            }

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
                            let out_x = view_count * width + x;
                            let offset = (y * width_total + out_x) * 3;
                            canvas[offset] = pix;
                            canvas[offset + 1] = pix;
                            canvas[offset + 2] = pix;
                        }
                    }
                }
            }

            let is_correct = probe_preds.as_ref().and_then(|(preds, labels)| {
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

            if let Some(image) = image::RgbImage::from_vec(
                width_total as u32,
                height as u32,
                canvas,
            ) {
                let filename = if self.overwrite {
                    format!("sample_{:02}.png", batch_idx)
                } else if let Some((preds, labels)) = &probe_preds {
                    let pred = preds.get(batch_idx).copied().unwrap_or(-1);
                    let label = labels.get(batch_idx).copied().unwrap_or(-1);
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}_pred_{pred}_label_{label}.png",
                        metadata.iteration,
                        batch_idx
                    )
                } else {
                    format!(
                        "lejepa_iter_{:06}_sample_{:02}.png",
                        metadata.iteration,
                        batch_idx
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

        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            format!("saved={saved}"),
            saved.to_string(),
        )
    }

    fn clear(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::data::dataloader::Progress;
    use burn_ndarray::NdArray;
    use burn_train::metric::{Metric, MetricMetadata};
    use std::env;
    use tempfile::tempdir;

    fn test_metadata(iteration: usize) -> MetricMetadata {
        MetricMetadata {
            progress: Progress::new(1, 1),
            epoch: 0,
            epoch_total: 1,
            iteration,
            lr: None,
        }
    }

    #[test]
    fn artifact_images_write_png() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Images,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
        );
        let views = Tensor::<Backend, 5>::zeros([1, 1, 3, 4, 4], &device);
        let patch_norms = Tensor::<Backend, 3>::zeros([1, 2, 2], &device);
        let input = VisionArtifactInput {
            views: Some(views),
            frames: None,
            patch_norms: Some(patch_norms),
            probe_logits: None,
            labels: None,
        };
        let _ = metric.update(&input, &test_metadata(0));
        assert!(output_dir.path().join("sample_00.png").is_file());
    }

    #[test]
    fn artifact_avi_write_video() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Avi,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
        );
        let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &device);
        let input = VisionArtifactInput {
            views: None,
            frames: Some(frames),
            patch_norms: None,
            probe_logits: None,
            labels: None,
        };
        let _ = metric.update(&input, &test_metadata(0));
        let path = output_dir.path().join("sample_00.avi");
        assert!(path.is_file());
        let bytes = fs::read(path).expect("read avi");
        assert!(bytes.starts_with(b"RIFF"));
    }

    #[test]
    fn artifact_mp4_write_video() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let bin_dir = output_dir.path().join("bin");
        fs::create_dir_all(&bin_dir).expect("bin dir");
        let script_path = bin_dir.join("ffmpeg.cmd");
        let script = r#"@echo off
set OUT=
for %%A in (%*) do set OUT=%%A
type nul > "%OUT%"
exit /b 0
"#;
        fs::write(&script_path, script).expect("write stub");
        let original = env::var("FFMPEG").ok();
        unsafe { env::set_var("FFMPEG", &script_path) };

        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Mp4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
        );
        let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &device);
        let input = VisionArtifactInput {
            views: None,
            frames: Some(frames),
            patch_norms: None,
            probe_logits: None,
            labels: None,
        };
        let _ = metric.update(&input, &test_metadata(0));
        let path = output_dir.path().join("sample_00.mp4");
        assert!(path.is_file());

        if let Some(original) = original {
            unsafe { env::set_var("FFMPEG", original) };
        } else {
            unsafe { env::remove_var("FFMPEG") };
        }
    }
}
