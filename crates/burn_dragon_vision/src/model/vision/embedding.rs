use super::{PatchGrid, SpatialPositionalEncodingKind, VisionDragonConfig, VisionPatchEmbedMode};
use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData, activation};
use burn_dragon_core::{DragonNorm, DragonNormConfig};
use core::f32::consts::PI;

#[derive(Clone)]
pub struct PatchEmbedOutput<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub grid: PatchGrid,
}

const PATCH_EMBED_EXPANSION: usize = 4;
const PATCH_EMBED_BLOCKS_PER_STAGE: usize = 1;
const PATCH_EMBED_CONVNEXT_STEM_BLOCKS: usize = 2;

#[derive(Module, Debug)]
struct PatchConvNeXtBlock<B: Backend> {
    depthwise: Conv2d<B>,
    pointwise_in: Conv2d<B>,
    pointwise_out: Conv2d<B>,
}

impl<B: Backend> PatchConvNeXtBlock<B> {
    fn new(channels: usize, expansion: usize, device: &B::Device) -> Self {
        let expansion = expansion.max(1);
        let depthwise = Conv2dConfig::new([channels, channels], [3, 3])
            .with_padding(PaddingConfig2d::Same)
            .with_groups(channels.max(1))
            .init(device);
        let pointwise_in =
            Conv2dConfig::new([channels, channels.saturating_mul(expansion)], [1, 1]).init(device);
        let pointwise_out =
            Conv2dConfig::new([channels.saturating_mul(expansion), channels], [1, 1]).init(device);
        Self {
            depthwise,
            pointwise_in,
            pointwise_out,
        }
    }

    fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let residual = x.clone();
        let x = self.depthwise.forward(x);
        let x = activation::gelu(x);
        let x = self.pointwise_in.forward(x);
        let x = activation::gelu(x);
        let x = self.pointwise_out.forward(x);
        x + residual
    }
}

#[derive(Module, Debug)]
struct PatchEmbedStage<B: Backend> {
    downsample: Conv2d<B>,
    blocks: Vec<PatchConvNeXtBlock<B>>,
}

impl<B: Backend> PatchEmbedStage<B> {
    fn new(
        in_channels: usize,
        out_channels: usize,
        stride: usize,
        blocks: usize,
        expansion: usize,
        device: &B::Device,
    ) -> Self {
        let stride = stride.max(1);
        let kernel = patch_kernel_for_stride(stride);
        let padding = if stride > 1 {
            PaddingConfig2d::Valid
        } else {
            PaddingConfig2d::Same
        };
        let downsample =
            Conv2dConfig::new([in_channels.max(1), out_channels.max(1)], [kernel, kernel])
                .with_stride([stride, stride])
                .with_padding(padding)
                .init(device);
        let blocks = (0..blocks.max(1))
            .map(|_| PatchConvNeXtBlock::new(out_channels.max(1), expansion, device))
            .collect();
        Self { downsample, blocks }
    }

    fn forward(&self, mut x: Tensor<B, 4>) -> Tensor<B, 4> {
        x = self.downsample.forward(x);
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    #[module(ignore)]
    mode: VisionPatchEmbedMode,
    stages: Vec<PatchEmbedStage<B>>,
    proj: Option<Conv2d<B>>,
    linear: Option<Linear<B>>,
    pos_encoding: SpatialPositionalEncoding<B>,
    patch_size: usize,
    embed_dim: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(config: &VisionDragonConfig, device: &B::Device) -> Self {
        let patch_size = config.patch_size.max(1);
        let patch_dim = patch_size
            .saturating_mul(patch_size)
            .saturating_mul(config.in_channels.max(1));
        let (stages, proj, linear) = match config.patch_embed_mode {
            VisionPatchEmbedMode::Conv => {
                let mut strides = patch_downsample_strides(patch_size);
                if strides.is_empty() {
                    strides.push(1);
                }
                let hidden_dim = (config.embed_dim / 2).max(16).min(config.embed_dim.max(1));
                let mut stages = Vec::with_capacity(strides.len());
                let mut in_channels = config.in_channels.max(1);
                for stride in strides {
                    stages.push(PatchEmbedStage::new(
                        in_channels,
                        hidden_dim,
                        stride,
                        PATCH_EMBED_BLOCKS_PER_STAGE,
                        PATCH_EMBED_EXPANSION,
                        device,
                    ));
                    in_channels = hidden_dim;
                }
                let proj = Conv2dConfig::new([in_channels.max(1), config.embed_dim.max(1)], [1, 1])
                    .init(device);
                (stages, Some(proj), None)
            }
            VisionPatchEmbedMode::ConvNext => {
                let mut strides = patch_downsample_strides(patch_size);
                if strides.is_empty() {
                    strides.push(1);
                }
                let hidden_dim = (config.embed_dim / 2).max(16).min(config.embed_dim.max(1));
                let mut stages = Vec::with_capacity(strides.len().saturating_add(1));
                let mut in_channels = config.in_channels.max(1);
                // Real ConvNeXt-style local stem: local residual mixing before patchify/downsample.
                stages.push(PatchEmbedStage::new(
                    in_channels,
                    hidden_dim,
                    1,
                    PATCH_EMBED_CONVNEXT_STEM_BLOCKS,
                    PATCH_EMBED_EXPANSION,
                    device,
                ));
                in_channels = hidden_dim;
                for stride in strides {
                    stages.push(PatchEmbedStage::new(
                        in_channels,
                        hidden_dim,
                        stride,
                        PATCH_EMBED_BLOCKS_PER_STAGE,
                        PATCH_EMBED_EXPANSION,
                        device,
                    ));
                    in_channels = hidden_dim;
                }
                let proj = Conv2dConfig::new([in_channels.max(1), config.embed_dim.max(1)], [1, 1])
                    .init(device);
                (stages, Some(proj), None)
            }
            VisionPatchEmbedMode::Linear => {
                let linear =
                    LinearConfig::new(patch_dim.max(1), config.embed_dim.max(1)).init(device);
                (Vec::new(), None, Some(linear))
            }
            VisionPatchEmbedMode::Identity => {
                assert!(
                    patch_dim == config.embed_dim,
                    "identity patch embed requires embed_dim ({}) to match patch_dim ({})",
                    config.embed_dim,
                    patch_dim
                );
                (Vec::new(), None, None)
            }
        };
        let pos_encoding = SpatialPositionalEncoding::new(
            config.pos_encoding,
            config.pos_max_height,
            config.pos_max_width,
            config.embed_dim,
            device,
        );
        Self {
            mode: config.patch_embed_mode,
            stages,
            proj,
            linear,
            pos_encoding,
            patch_size,
            embed_dim: config.embed_dim,
        }
    }

    pub fn forward(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        let output = self.forward_raw(images);
        let tokens = self.pos_encoding.add_position(output.tokens, output.grid);
        PatchEmbedOutput {
            tokens,
            grid: output.grid,
        }
    }

    pub fn forward_raw(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        match self.mode {
            VisionPatchEmbedMode::Conv | VisionPatchEmbedMode::ConvNext => {
                let [batch, channels, height, width] = images.shape().dims::<4>();
                let device = images.device();
                let patch_size = self.patch_size.max(1);
                let padded_h = height.div_ceil(patch_size) * patch_size;
                let padded_w = width.div_ceil(patch_size) * patch_size;
                let mut patches = images;
                let pad_w = padded_w.saturating_sub(width);
                let pad_h = padded_h.saturating_sub(height);
                if pad_w > 0 {
                    let pad = Tensor::<B, 4>::zeros([batch, channels, height, pad_w], &device);
                    patches = Tensor::cat(vec![patches, pad], 3);
                }
                if pad_h > 0 {
                    let pad = Tensor::<B, 4>::zeros([batch, channels, pad_h, padded_w], &device);
                    patches = Tensor::cat(vec![patches, pad], 2);
                }
                for stage in &self.stages {
                    patches = stage.forward(patches);
                }
                let proj = self.proj.as_ref().expect("patch embed conv projection");
                let patches = proj.forward(patches);
                let [_, _, grid_h, grid_w] = patches.shape().dims::<4>();
                let tokens = patches
                    .reshape([batch, self.embed_dim, grid_h * grid_w])
                    .swap_dims(1, 2);

                PatchEmbedOutput {
                    tokens,
                    grid: PatchGrid {
                        height: grid_h,
                        width: grid_w,
                    },
                }
            }
            VisionPatchEmbedMode::Linear => {
                let [_batch, _, height, width] = images.shape().dims::<4>();
                let patch_size = self.patch_size.max(1);
                let grid_h = height.div_ceil(patch_size);
                let grid_w = width.div_ceil(patch_size);
                let patches = patchify(images, patch_size);
                let linear = self.linear.as_ref().expect("patch embed linear projection");
                let tokens = linear.forward(patches);
                PatchEmbedOutput {
                    tokens,
                    grid: PatchGrid {
                        height: grid_h,
                        width: grid_w,
                    },
                }
            }
            VisionPatchEmbedMode::Identity => {
                let [_batch, _, height, width] = images.shape().dims::<4>();
                let patch_size = self.patch_size.max(1);
                let grid_h = height.div_ceil(patch_size);
                let grid_w = width.div_ceil(patch_size);
                let tokens = patchify(images, patch_size);
                PatchEmbedOutput {
                    tokens,
                    grid: PatchGrid {
                        height: grid_h,
                        width: grid_w,
                    },
                }
            }
        }
    }

    pub fn add_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        self.pos_encoding.add_position(tokens, grid)
    }

    pub fn patch_size(&self) -> usize {
        self.patch_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::vision::{
        FusedKernelConfig, VisionAttentionMode, VisionBackboneKind, VisionDragonConfig,
        VisionLatentActivation,
    };
    use burn::backend::NdArray;
    use burn_dragon_core::ManifoldHyperConnectionsConfig;

    fn test_config(mode: VisionPatchEmbedMode) -> VisionDragonConfig {
        VisionDragonConfig {
            image_size: 196,
            patch_size: 14,
            patch_embed_mode: mode,
            backbone: VisionBackboneKind::Dense,
            in_channels: 3,
            embed_dim: 128,
            steps: 4,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            dropout: 0.0,
            projection_dim: 384,
            projection_hidden_dim: 384,
            use_cls_token: true,
            cls_sync_alpha: 0.1,
            num_eyes: 1,
            cross_eye_steps: 0,
            token_state_norm: true,
            normalization: DragonNormConfig::default(),
            latent_activation: VisionLatentActivation::default(),
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: 14,
            pos_max_width: 14,
            attention_mode: VisionAttentionMode::RowL1,
            use_alibi: true,
            fused_kernels: FusedKernelConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            trm_graph: Default::default(),
            rho_stream: Default::default(),
        }
    }

    #[test]
    fn convnext_patch_embed_builds_distinct_local_stem() {
        type BackendImpl = NdArray<f32>;
        let device = <BackendImpl as Backend>::Device::default();
        let conv =
            PatchEmbed::<BackendImpl>::new(&test_config(VisionPatchEmbedMode::Conv), &device);
        let convnext =
            PatchEmbed::<BackendImpl>::new(&test_config(VisionPatchEmbedMode::ConvNext), &device);

        assert!(convnext.stages.len() > conv.stages.len());
        assert_eq!(convnext.stages.first().expect("stem").blocks.len(), 2);
        assert_eq!(conv.stages.first().expect("conv stage").blocks.len(), 1);
    }
}

fn patch_downsample_strides(patch_size: usize) -> Vec<usize> {
    let mut remaining = patch_size.max(1);
    let mut strides = Vec::new();
    while remaining > 1 {
        if remaining.is_multiple_of(4) {
            strides.push(4);
            remaining /= 4;
        } else if remaining.is_multiple_of(2) {
            strides.push(2);
            remaining /= 2;
        } else {
            strides.push(remaining);
            remaining = 1;
        }
    }
    strides
}

fn patch_kernel_for_stride(stride: usize) -> usize {
    if stride <= 1 { 3 } else { stride }
}

pub fn pool_patch_tokens<B: Backend>(
    tokens: Tensor<B, 3>,
    grid: PatchGrid,
) -> (Tensor<B, 3>, PatchGrid) {
    let [batch, tokens_len, dim] = tokens.shape().dims::<3>();
    let grid_h = grid.height;
    let grid_w = grid.width;
    if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens_len {
        return (tokens, grid);
    }
    let even_h = grid_h - (grid_h % 2);
    let even_w = grid_w - (grid_w % 2);
    if even_h == 0 || even_w == 0 {
        return (tokens, grid);
    }
    let tokens = tokens.reshape([batch, grid_h, grid_w, dim]);
    let tokens = tokens.slice_dim(1, 0..even_h).slice_dim(2, 0..even_w);
    let next_h = even_h / 2;
    let next_w = even_w / 2;
    let tokens = tokens
        .reshape([batch, next_h, 2, next_w, 2, dim])
        .mean_dim(2)
        .mean_dim(4)
        .reshape([batch, next_h * next_w, dim]);
    (
        tokens,
        PatchGrid {
            height: next_h,
            width: next_w,
        },
    )
}

pub fn patchify<B: Backend>(images: Tensor<B, 4>, patch_size: usize) -> Tensor<B, 3> {
    let [batch, channels, height, width] = images.shape().dims::<4>();
    let patch_size = patch_size.max(1);
    let grid_h = height.div_ceil(patch_size);
    let grid_w = width.div_ceil(patch_size);
    let padded_h = grid_h * patch_size;
    let padded_w = grid_w * patch_size;
    let device = images.device();
    let mut images = images;
    let pad_w = padded_w.saturating_sub(width);
    let pad_h = padded_h.saturating_sub(height);
    if pad_w > 0 {
        let pad = Tensor::<B, 4>::zeros([batch, channels, height, pad_w], &device);
        images = Tensor::cat(vec![images, pad], 3);
    }
    if pad_h > 0 {
        let pad = Tensor::<B, 4>::zeros([batch, channels, pad_h, padded_w], &device);
        images = Tensor::cat(vec![images, pad], 2);
    }
    images
        .reshape([batch, channels, grid_h, patch_size, grid_w, patch_size])
        .swap_dims(1, 2)
        .swap_dims(2, 4)
        .swap_dims(3, 4)
        .reshape([batch, grid_h * grid_w, channels * patch_size * patch_size])
}

pub fn unpatchify<B: Backend>(
    patches: Tensor<B, 3>,
    patch_size: usize,
    height: usize,
    width: usize,
    channels: usize,
) -> Tensor<B, 4> {
    let [batch, tokens, patch_dim] = patches.shape().dims::<3>();
    assert!(patch_dim > 0, "unpatchify expects non-empty patch dim");
    let patch_size = patch_size.max(1);
    let grid_h = height.div_ceil(patch_size);
    let grid_w = width.div_ceil(patch_size);
    assert!(
        grid_h * grid_w == tokens,
        "unpatchify expects token count to match grid"
    );
    let padded_h = grid_h * patch_size;
    let padded_w = grid_w * patch_size;
    let image = patches
        .reshape([batch, grid_h, grid_w, channels, patch_size, patch_size])
        .swap_dims(3, 4)
        .swap_dims(2, 4)
        .swap_dims(1, 2)
        .reshape([batch, channels, padded_h, padded_w]);
    if padded_h == height && padded_w == width {
        image
    } else {
        image.slice_dim(2, 0..height).slice_dim(3, 0..width)
    }
}

#[derive(Module, Debug)]
pub struct SpatialPositionalEncoding<B: Backend> {
    kind: SpatialPositionalEncodingKind,
    row_embed: Option<Param<Tensor<B, 2>>>,
    col_embed: Option<Param<Tensor<B, 2>>>,
    max_height: usize,
    max_width: usize,
    dim: usize,
}

impl<B: Backend> SpatialPositionalEncoding<B> {
    pub fn new(
        kind: SpatialPositionalEncodingKind,
        max_height: usize,
        max_width: usize,
        dim: usize,
        device: &B::Device,
    ) -> Self {
        let max_height = max_height.max(1);
        let max_width = max_width.max(1);
        let (row_embed, col_embed) = if kind == SpatialPositionalEncodingKind::Learned2d {
            let row = Tensor::<B, 2>::random(
                [max_height, dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            let col = Tensor::<B, 2>::random(
                [max_width, dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            (Some(Param::from_tensor(row)), Some(Param::from_tensor(col)))
        } else {
            (None, None)
        };

        Self {
            kind,
            row_embed,
            col_embed,
            max_height,
            max_width,
            dim,
        }
    }

    pub fn add_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        match self.kind {
            SpatialPositionalEncodingKind::None => tokens,
            SpatialPositionalEncodingKind::Learned2d => tokens + self.learned_positions(grid),
            SpatialPositionalEncodingKind::SineCosine2d => {
                let device = tokens.device();
                tokens + self.sincos_positions(grid, &device)
            }
            SpatialPositionalEncodingKind::Rope => self.rope_positions(tokens, grid),
        }
    }

    fn learned_positions(&self, grid: PatchGrid) -> Tensor<B, 3> {
        assert!(
            grid.height <= self.max_height && grid.width <= self.max_width,
            "positional grid exceeds configured max size"
        );
        let row = self
            .row_embed
            .as_ref()
            .expect("row embedding required")
            .val()
            .slice_dim(0, 0..grid.height);
        let col = self
            .col_embed
            .as_ref()
            .expect("col embedding required")
            .val()
            .slice_dim(0, 0..grid.width);
        let row = row.unsqueeze_dim::<3>(1);
        let col = col.unsqueeze_dim::<3>(0);
        let pos = row + col;
        let pos = pos.reshape([grid.height * grid.width, self.dim]);
        pos.unsqueeze_dim::<3>(0)
    }

    fn sincos_positions(&self, grid: PatchGrid, device: &B::Device) -> Tensor<B, 3> {
        assert!(
            self.dim.is_multiple_of(4),
            "sine-cosine positional encoding requires dim divisible by 4"
        );
        let quarter = self.dim / 4;
        let mut omega = Vec::with_capacity(quarter);
        for idx in 0..quarter {
            let value = 1.0 / 10000.0f32.powf(idx as f32 / quarter as f32);
            omega.push(value);
        }

        let mut data = Vec::with_capacity(grid.num_patches() * self.dim);
        for y in 0..grid.height {
            for x in 0..grid.width {
                for omega_value in &omega {
                    let wy = y as f32 * *omega_value;
                    let wx = x as f32 * *omega_value;
                    data.push(wy.sin());
                    data.push(wy.cos());
                    data.push(wx.sin());
                    data.push(wx.cos());
                }
            }
        }

        Tensor::<B, 3>::from_data(
            TensorData::new(data, [1, grid.num_patches(), self.dim]),
            device,
        )
    }

    fn rope_positions(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        let patch_count = grid.num_patches();
        if batch == 0 || time == 0 || dim < 4 || patch_count == 0 || time != patch_count {
            return tokens;
        }

        let rope_dim = dim - (dim % 4);
        if rope_dim == 0 {
            return tokens;
        }

        let quarter = rope_dim / 4;
        if quarter == 0 {
            return tokens;
        }

        let mut omega = Vec::with_capacity(quarter);
        for idx in 0..quarter {
            let value = 1.0 / 10000.0f32.powf(idx as f32 / quarter as f32);
            omega.push(value);
        }

        let mut phase_data = Vec::with_capacity(time * rope_dim);
        for y in 0..grid.height {
            for x in 0..grid.width {
                for omega_value in &omega {
                    let phase = ((y as f32) * *omega_value).fract() * (2.0 * PI);
                    phase_data.push(phase);
                    phase_data.push(phase);
                }
                for omega_value in &omega {
                    let phase = ((x as f32) * *omega_value).fract() * (2.0 * PI);
                    phase_data.push(phase);
                    phase_data.push(phase);
                }
            }
        }

        let phases = Tensor::<B, 3>::from_data(
            TensorData::new(phase_data, [1, time, rope_dim]),
            &tokens.device(),
        );
        let rotated = Self::apply_rope_3d(tokens.clone().slice_dim(2, 0..rope_dim), phases);
        if rope_dim == dim {
            rotated
        } else {
            Tensor::cat(vec![rotated, tokens.slice_dim(2, rope_dim..dim)], 2)
        }
    }

    fn apply_rope_3d(values: Tensor<B, 3>, phases: Tensor<B, 3>) -> Tensor<B, 3> {
        let cos = phases.clone().cos();
        let sin = phases.sin();
        let [batch, time, dim] = values.shape().dims::<3>();
        let pairs = values.clone().reshape([batch, time, dim / 2, 2]);
        let even = pairs.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let odd = pairs.slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let rotated =
            Tensor::stack::<4>(vec![odd.clone().neg(), even], 3).reshape([batch, time, dim]);
        values * cos + rotated * sin
    }
}

#[derive(Module, Debug)]
pub struct VisionProjectionHead<B: Backend> {
    norm: DragonNorm<B>,
    fc1: Linear<B>,
    fc2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> VisionProjectionHead<B> {
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        dropout: f64,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, input_dim, device);
        let fc1 = LinearConfig::new(input_dim, hidden_dim).init(device);
        let fc2 = LinearConfig::new(hidden_dim, output_dim).init(device);
        let dropout = DropoutConfig::new(dropout).init();
        Self {
            norm,
            fc1,
            fc2,
            dropout,
        }
    }

    pub fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = self.norm.forward(tokens);
        let tokens = self.fc1.forward(tokens);
        let tokens = activation::gelu(tokens);
        let tokens = self.dropout.forward(tokens);
        self.fc2.forward(tokens)
    }
}
