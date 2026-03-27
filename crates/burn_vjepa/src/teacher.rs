use anyhow::Result;
use burn::module::Module;
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

#[cfg(test)]
use burn::tensor::Distribution as TensorDistribution;
use burn_dragon_vision::{VisionDragon, VisionDragonConfig, load_vision_encoder_from_checkpoint};
use std::path::{Path, PathBuf};

pub trait ClipFeatureTeacher<B: Backend> {
    fn feature_dim(&self) -> usize;
    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3>;
}

#[cfg(test)]
#[derive(Module, Debug)]
pub(crate) struct FixedProjectionTeacher<B: Backend> {
    projection: Tensor<B, 2>,
    #[module(ignore)]
    input_dim: usize,
    #[module(ignore)]
    feature_dim: usize,
}

#[cfg(test)]
impl<B: Backend> FixedProjectionTeacher<B> {
    pub fn new(input_dim: usize, feature_dim: usize, device: &B::Device) -> Self {
        let projection = Tensor::<B, 2>::random(
            [input_dim.max(1), feature_dim.max(1)],
            TensorDistribution::Normal(0.0, f64::from((1.0 / input_dim.max(1) as f32).sqrt())),
            device,
        );
        Self {
            projection,
            input_dim: input_dim.max(1),
            feature_dim: feature_dim.max(1),
        }
    }
}

#[cfg(test)]
impl<B: Backend> ClipFeatureTeacher<B> for FixedProjectionTeacher<B> {
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3> {
        let [batch, frames, channels, height, width] = clip_frames.shape().dims::<5>();
        let flat = clip_frames.reshape([batch * frames, channels * height * width]);
        let projected = flat.matmul(self.projection.clone());
        projected.reshape([batch, frames, self.feature_dim])
    }
}

#[derive(Module, Debug)]
pub struct VisionDragonTeacher<B: Backend> {
    model: VisionDragon<B>,
    #[module(ignore)]
    projection_dim: usize,
    #[module(ignore)]
    image_size: usize,
    #[module(ignore)]
    in_channels: usize,
    #[module(ignore)]
    rollout_steps: usize,
    #[module(ignore)]
    backprop_steps: usize,
}

impl<B: Backend> VisionDragonTeacher<B> {
    pub fn new(
        config: VisionDragonConfig,
        rollout_steps: usize,
        backprop_steps: usize,
        device: &B::Device,
    ) -> Self {
        let projection_dim = config.projection_dim.max(1);
        let image_size = config.image_size.max(1);
        let in_channels = config.in_channels.max(1);
        let model = VisionDragon::new(config, device);
        Self {
            model,
            projection_dim,
            image_size,
            in_channels,
            rollout_steps: rollout_steps.max(1),
            backprop_steps: backprop_steps.max(1),
        }
    }
}

#[derive(Debug)]
pub struct CheckpointVisionDragonTeacher<B: Backend> {
    model: VisionDragon<B>,
    projection_dim: usize,
    image_size: usize,
    in_channels: usize,
    rollout_steps: usize,
    backprop_steps: usize,
    checkpoint: PathBuf,
}

impl<B: Backend> CheckpointVisionDragonTeacher<B> {
    pub fn from_checkpoint(
        checkpoint: impl AsRef<Path>,
        config_paths: &[PathBuf],
        rollout_steps: usize,
        backprop_steps: usize,
        device: &B::Device,
    ) -> Result<Self> {
        let checkpoint = checkpoint.as_ref().to_path_buf();
        let model = load_vision_encoder_from_checkpoint(&checkpoint, None, config_paths, device)?;
        let projection_dim = model.projection_dim().max(1);
        let image_size = model.image_size().max(1);
        let in_channels = model.in_channels().max(1);
        Ok(Self {
            model,
            projection_dim,
            image_size,
            in_channels,
            rollout_steps: rollout_steps.max(1),
            backprop_steps: backprop_steps.max(1),
            checkpoint,
        })
    }

    pub fn checkpoint_path(&self) -> &Path {
        &self.checkpoint
    }
}

impl<B: Backend> ClipFeatureTeacher<B> for VisionDragonTeacher<B> {
    fn feature_dim(&self) -> usize {
        self.projection_dim
    }

    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3> {
        let adapted = adapt_clip_frames(clip_frames, self.in_channels, self.image_size);
        let [batch, frames, channels, height, width] = adapted.shape().dims::<5>();
        let flat = adapted.reshape([batch * frames, channels, height, width]);
        let encoded = self.model.forward_images_embed_steps_rollout_unbounded(
            flat,
            self.rollout_steps,
            self.backprop_steps,
        );
        let projected = self
            .model
            .project_tokens(encoded.cls_token.unsqueeze_dim::<3>(1))
            .reshape([batch * frames, self.projection_dim]);
        projected.reshape([batch, frames, self.projection_dim])
    }
}

impl<B: Backend> ClipFeatureTeacher<B> for CheckpointVisionDragonTeacher<B> {
    fn feature_dim(&self) -> usize {
        self.projection_dim
    }

    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3> {
        let adapted = adapt_clip_frames(clip_frames, self.in_channels, self.image_size);
        let [batch, frames, channels, height, width] = adapted.shape().dims::<5>();
        let flat = adapted.reshape([batch * frames, channels, height, width]);
        let encoded = self.model.forward_images_embed_steps_rollout_unbounded(
            flat,
            self.rollout_steps,
            self.backprop_steps,
        );
        let projected = self
            .model
            .project_tokens(encoded.cls_token.unsqueeze_dim::<3>(1))
            .reshape([batch * frames, self.projection_dim]);
        projected.reshape([batch, frames, self.projection_dim])
    }
}

fn adapt_clip_frames<B: Backend>(
    clip_frames: Tensor<B, 5>,
    target_channels: usize,
    target_size: usize,
) -> Tensor<B, 5> {
    let [batch, frames, channels, height, width] = clip_frames.shape().dims::<5>();
    if channels == target_channels && height == target_size && width == target_size {
        return clip_frames;
    }

    let device = clip_frames.device();
    let source = clip_frames
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("teacher clip frames to host");
    let mut adapted = vec![0.0f32; batch * frames * target_channels * target_size * target_size];

    for batch_idx in 0..batch {
        for frame_idx in 0..frames {
            for out_channel in 0..target_channels {
                for out_y in 0..target_size {
                    let src_y = ((out_y as f32 + 0.5) * height as f32 / target_size as f32 - 0.5)
                        .round()
                        .clamp(0.0, height.saturating_sub(1) as f32)
                        as usize;
                    for out_x in 0..target_size {
                        let src_x = ((out_x as f32 + 0.5) * width as f32 / target_size as f32 - 0.5)
                            .round()
                            .clamp(0.0, width.saturating_sub(1) as f32)
                            as usize;
                        let dst_offset = (((batch_idx * frames + frame_idx) * target_channels
                            + out_channel)
                            * target_size
                            + out_y)
                            * target_size
                            + out_x;
                        adapted[dst_offset] = if channels == target_channels {
                            let src_offset = (((batch_idx * frames + frame_idx) * channels
                                + out_channel)
                                * height
                                + src_y)
                                * width
                                + src_x;
                            source[src_offset]
                        } else if channels == 1 {
                            let src_offset =
                                (((batch_idx * frames + frame_idx) * channels) * height + src_y)
                                    * width
                                    + src_x;
                            source[src_offset]
                        } else if target_channels == 1 {
                            let mut sum = 0.0f32;
                            for src_channel in 0..channels {
                                let src_offset = (((batch_idx * frames + frame_idx) * channels
                                    + src_channel)
                                    * height
                                    + src_y)
                                    * width
                                    + src_x;
                                sum += source[src_offset];
                            }
                            sum / channels.max(1) as f32
                        } else {
                            let src_channel = out_channel.min(channels.saturating_sub(1));
                            let src_offset = (((batch_idx * frames + frame_idx) * channels
                                + src_channel)
                                * height
                                + src_y)
                                * width
                                + src_x;
                            source[src_offset]
                        };
                    }
                }
            }
        }
    }

    Tensor::<B, 5>::from_data(
        TensorData::new(
            adapted,
            [batch, frames, target_channels, target_size, target_size],
        ),
        &device,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type B = NdArray<f32>;

    #[test]
    fn fixed_projection_teacher_has_expected_shape() {
        let device = Default::default();
        let teacher = FixedProjectionTeacher::<B>::new(16, 6, &device);
        let clip = Tensor::<B, 5>::zeros([2, 3, 1, 4, 4], &device);
        let encoded = teacher.encode_clip(clip);
        assert_eq!(encoded.shape().dims::<3>(), [2, 3, 6]);
    }

    #[test]
    fn vision_dragon_teacher_emits_clip_features() {
        let device = Default::default();
        let config = VisionDragonConfig::default();
        let teacher = VisionDragonTeacher::<B>::new(config, 1, 1, &device);
        let clip = Tensor::<B, 5>::zeros([1, 2, 3, 32, 32], &device);
        let encoded = teacher.encode_clip(clip);
        assert_eq!(encoded.shape().dims::<3>()[0], 1);
        assert_eq!(encoded.shape().dims::<3>()[1], 2);
        assert_eq!(encoded.shape().dims::<3>()[2], teacher.feature_dim());
    }
}
