use crate::{Vjepa2Config, Vjepa2Model};
use anyhow::Result;
use burn::module::Module;
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions};

#[cfg(test)]
use burn::tensor::Distribution as TensorDistribution;
#[cfg(feature = "train")]
use burn_dragon_vision::{VisionDragon, VisionDragonConfig, load_vision_encoder_from_checkpoint};
use std::path::{Path, PathBuf};

pub trait ClipFeatureTeacher<B: Backend> {
    fn feature_dim(&self) -> usize;
    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3>;
}

#[derive(Module, Debug)]
pub struct NativeVjepa2Teacher<B: Backend> {
    pub model: Vjepa2Model<B>,
    #[module(ignore)]
    feature_dim: usize,
    #[module(ignore)]
    image_size: usize,
    #[module(ignore)]
    in_channels: usize,
    #[module(ignore)]
    tubelet_size: usize,
}

impl<B: Backend> NativeVjepa2Teacher<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        let feature_dim = config.hidden_size.max(1);
        let image_size = config.crop_size.max(1);
        let in_channels = config.in_chans.max(1);
        let tubelet_size = config.tubelet_size.max(1);
        Self {
            model: Vjepa2Model::new(config, device),
            feature_dim,
            image_size,
            in_channels,
            tubelet_size,
        }
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let config = Vjepa2Config::from_json_file(dir.as_ref().join("config.json"))?;
        let feature_dim = config.hidden_size.max(1);
        let image_size = config.crop_size.max(1);
        let in_channels = config.in_chans.max(1);
        let tubelet_size = config.tubelet_size.max(1);
        let model = Vjepa2Model::from_hf_dir(dir, device)?;
        Ok(Self {
            model,
            feature_dim,
            image_size,
            in_channels,
            tubelet_size,
        })
    }
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

#[cfg(feature = "train")]
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

impl<B: Backend> ClipFeatureTeacher<B> for NativeVjepa2Teacher<B> {
    fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    fn encode_clip(&self, clip_frames: Tensor<B, 5>) -> Tensor<B, 3> {
        let original_frames = clip_frames.shape().dims::<5>()[1].max(1);
        let adapted = adapt_clip_frames(clip_frames, self.in_channels, self.image_size);
        let encoded = self.model.get_vision_features(adapted);
        let [batch, tokens, dim] = encoded.shape().dims::<3>();
        let spatial = (self.image_size / self.model.config.0.patch_size.max(1))
            .pow(2)
            .max(1);
        let tubelets = (tokens / spatial).max(1);
        let per_tubelet = encoded
            .reshape([batch, tubelets, spatial, dim])
            .mean_dim(2)
            .reshape([batch, tubelets, dim]);
        let repeated = per_tubelet.repeat_dim(1, self.tubelet_size.max(1));
        repeated.slice_dim(1, 0..original_frames)
    }
}

fn adapt_clip_frames<B: Backend>(
    clip_frames: Tensor<B, 5>,
    target_channels: usize,
    target_size: usize,
) -> Tensor<B, 5> {
    let [batch, frames, channels, height, width] = clip_frames.shape().dims::<5>();
    let clip_frames = match (channels, target_channels) {
        (c, t) if c == t => clip_frames,
        (1, 3) => clip_frames.repeat_dim(2, 3),
        (3, 1) => clip_frames.mean_dim(2).unsqueeze_dim::<5>(2),
        _ => panic!(
            "unsupported clip channel adaptation {} -> {} for native V-JEPA teacher",
            channels, target_channels
        ),
    };
    if height == target_size && width == target_size {
        return clip_frames;
    }

    let clip_frames = clip_frames.reshape([batch * frames, target_channels, height, width]);
    let clip_frames = interpolate(
        clip_frames,
        [target_size, target_size],
        InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
    );
    clip_frames.reshape([batch, frames, target_channels, target_size, target_size])
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

    #[cfg(feature = "train")]
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
