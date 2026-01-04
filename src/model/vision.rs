use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData, activation};
use serde::{Deserialize, Serialize};

use crate::kernel::{BlockPattern1d, relu_lowrank};

use super::config::FusedKernelConfig;

const ROW_NORM_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum SpatialPositionalEncodingKind {
    None,
    #[default]
    Learned2d,
    SineCosine2d,
}

impl core::fmt::Display for SpatialPositionalEncodingKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for SpatialPositionalEncodingKind {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SpatialPositionalEncodingKind {
    type InnerModule = SpatialPositionalEncodingKind;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for SpatialPositionalEncodingKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for SpatialPositionalEncodingKind {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionAttentionMode {
    #[default]
    RowL1,
    Softmax,
}

impl core::fmt::Display for VisionAttentionMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl<B: Backend> Module<B> for VisionAttentionMode {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionAttentionMode {
    type InnerModule = VisionAttentionMode;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for VisionAttentionMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionAttentionMode {}

#[derive(Clone, Debug)]
pub struct VisionDragonHatchlingConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub steps: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub dropout: f64,
    pub projection_dim: usize,
    pub projection_hidden_dim: usize,
    pub use_cls_token: bool,
    pub pos_encoding: SpatialPositionalEncodingKind,
    pub pos_max_height: usize,
    pub pos_max_width: usize,
    pub attention_mode: VisionAttentionMode,
    pub fused_kernels: FusedKernelConfig,
}

impl Default for VisionDragonHatchlingConfig {
    fn default() -> Self {
        let image_size = 224;
        let patch_size = 16;
        let grid = (image_size / patch_size).max(1);
        Self {
            image_size,
            patch_size,
            in_channels: 3,
            embed_dim: 256,
            steps: 6,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            dropout: 0.1,
            projection_dim: 384,
            projection_hidden_dim: 512,
            use_cls_token: true,
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: grid,
            pos_max_width: grid,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: FusedKernelConfig::default(),
        }
    }
}

impl VisionDragonHatchlingConfig {
    pub fn latent_per_head(&self) -> usize {
        let total = self.mlp_internal_dim_multiplier * self.embed_dim;
        assert!(
            total.is_multiple_of(self.n_head),
            "latent size must be divisible by the number of heads"
        );
        total / self.n_head
    }

    pub fn latent_total(&self) -> usize {
        self.latent_per_head() * self.n_head
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PatchGrid {
    pub height: usize,
    pub width: usize,
}

impl PatchGrid {
    pub fn num_patches(&self) -> usize {
        self.height * self.width
    }
}

#[derive(Clone)]
pub struct PatchEmbedOutput<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub grid: PatchGrid,
}

#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    conv: Conv2d<B>,
    pos_encoding: SpatialPositionalEncoding<B>,
    patch_size: usize,
    embed_dim: usize,
}

impl<B: Backend> PatchEmbed<B> {
    pub fn new(config: &VisionDragonHatchlingConfig, device: &B::Device) -> Self {
        let conv = Conv2dConfig::new(
            [config.in_channels, config.embed_dim],
            [config.patch_size, config.patch_size],
        )
        .with_stride([config.patch_size, config.patch_size])
        .init(device);
        let pos_encoding = SpatialPositionalEncoding::new(
            config.pos_encoding,
            config.pos_max_height,
            config.pos_max_width,
            config.embed_dim,
            device,
        );
        Self {
            conv,
            pos_encoding,
            patch_size: config.patch_size,
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
        let [batch, _, height, width] = images.shape().dims::<4>();
        assert!(
            height.is_multiple_of(self.patch_size) && width.is_multiple_of(self.patch_size),
            "image size must be divisible by patch size"
        );

        let patches = self.conv.forward(images);
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

    pub fn add_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        self.pos_encoding.add_position(tokens, grid)
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
                for omega_value in omega.iter() {
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
}

#[derive(Module, Debug)]
pub struct VisionProjectionHead<B: Backend> {
    norm: LayerNorm<B>,
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
        device: &B::Device,
    ) -> Self {
        let norm = LayerNormConfig::new(input_dim).init(device);
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

#[derive(Clone)]
pub struct VisionDragonHatchlingOutput<B: Backend> {
    pub patch_tokens: Tensor<B, 3>,
    pub cls_token: Tensor<B, 2>,
}

#[derive(Module, Debug)]
pub struct VisionDragonHatchling<B: Backend> {
    steps: usize,
    n_head: usize,
    embed_dim: usize,
    mlp_internal_dim_multiplier: usize,
    use_cls_token: bool,
    attention_mode: VisionAttentionMode,
    kernel: FusedKernelConfig,
    patch_embed: PatchEmbed<B>,
    dropout: Dropout,
    token_norm: LayerNorm<B>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    projection: VisionProjectionHead<B>,
    cls_token: Option<Param<Tensor<B, 2>>>,
    cls_pos: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> VisionDragonHatchling<B> {
    pub fn new(config: VisionDragonHatchlingConfig, device: &B::Device) -> Self {
        let patch_embed = PatchEmbed::new(&config, device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let token_norm = LayerNormConfig::new(config.embed_dim).init(device);

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();

        let encoder = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.embed_dim, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let encoder_v = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.embed_dim, latent_per_head],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let decoder = Param::from_tensor(Tensor::<B, 2>::random(
            [latent_total, config.embed_dim],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));

        let projection = VisionProjectionHead::new(
            config.embed_dim,
            config.projection_hidden_dim.max(1),
            config.projection_dim.max(1),
            config.dropout,
            device,
        );

        let (cls_token, cls_pos) = if config.use_cls_token {
            let cls_token = Tensor::<B, 2>::random(
                [1, config.embed_dim],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            let cls_pos = if config.pos_encoding == SpatialPositionalEncodingKind::Learned2d {
                Some(Param::from_tensor(Tensor::<B, 2>::random(
                    [1, config.embed_dim],
                    TensorDistribution::Normal(0.0, 0.02),
                    device,
                )))
            } else {
                None
            };
            (Some(Param::from_tensor(cls_token)), cls_pos)
        } else {
            (None, None)
        };

        Self {
            steps: config.steps.max(1),
            n_head: config.n_head,
            embed_dim: config.embed_dim,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            use_cls_token: config.use_cls_token,
            attention_mode: config.attention_mode,
            kernel: config.fused_kernels,
            patch_embed,
            dropout,
            token_norm,
            encoder,
            encoder_v,
            decoder,
            projection,
            cls_token,
            cls_pos,
        }
    }

    pub fn patch_embed(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward(images)
    }

    pub fn add_patch_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        self.patch_embed.add_position(tokens, grid)
    }

    pub fn project_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.projection.forward(tokens)
    }

    pub fn forward_images(&self, images: Tensor<B, 4>) -> VisionDragonHatchlingOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens(patch.tokens)
    }

    pub fn forward_images_steps(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
    ) -> VisionDragonHatchlingOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps(patch.tokens, steps)
    }

    pub fn forward_patches(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
    ) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens(tokens)
    }

    pub fn forward_patches_steps(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
        steps: usize,
    ) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens_steps(tokens, steps)
    }

    pub fn forward_tokens(&self, tokens: Tensor<B, 3>) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.encode_tokens(tokens);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_embed(&self, tokens: Tensor<B, 3>) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.encode_tokens(tokens);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonHatchlingOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        self.split_output(tokens)
    }

    fn encode_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encode_tokens_steps(tokens, self.steps)
    }

    fn encode_tokens_steps(&self, tokens: Tensor<B, 3>, steps: usize) -> Tensor<B, 3> {
        let steps = steps.max(1).min(self.steps);
        let tokens = if self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, _] = tokens.shape().dims::<3>();
        let mut current = tokens.reshape([batch, 1, time, self.embed_dim]);
        current = self.token_norm.forward(current);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern: &BlockPattern1d = &self.kernel.block_sparse.latent;

        for _ in 0..steps {
            let x_sparse = if fused {
                relu_lowrank::fused_forward(
                    current.clone(),
                    encoder.clone(),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut x_latent = current.clone().matmul(encoder.clone());
                if self.kernel.relu_threshold != 0.0 {
                    x_latent = x_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(x_latent)
            };

            let attn = self.full_attention(x_sparse.clone(), current.clone());
            let attn = self.token_norm.forward(attn);

            let y_sparse = if fused {
                relu_lowrank::fused_forward(
                    attn.clone(),
                    encoder_v.clone(),
                    None,
                    self.kernel.relu_threshold,
                    latent_pattern,
                )
            } else {
                let mut y_latent = attn.matmul(encoder_v.clone());
                if self.kernel.relu_threshold != 0.0 {
                    y_latent = y_latent.sub_scalar(self.kernel.relu_threshold);
                }
                activation::relu(y_latent)
            };

            let xy_sparse = x_sparse * y_sparse;
            let xy_sparse = self.dropout.forward(xy_sparse);

            let mixed = xy_sparse.clone().swap_dims(1, 2);
            let [batch, time, heads, latent] = mixed.shape().dims();
            let mixed_flat = mixed.reshape([batch * time, heads * latent]);
            let mlp_flat = mixed_flat.matmul(decoder.clone());
            let mlp_out = mlp_flat.reshape([batch, 1, time, self.embed_dim]);
            let mlp_out = self.token_norm.forward(mlp_out);
            current = self.token_norm.forward(current + mlp_out);
        }

        current.reshape([batch, time, self.embed_dim])
    }

    fn full_attention(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let k = query.clone();
        let mut scores = query.matmul(k.swap_dims(2, 3));
        match self.attention_mode {
            VisionAttentionMode::Softmax => {
                let scale = latent.sqrt().max(1.0);
                scores = scores.div_scalar(scale);
                scores = activation::softmax(scores, 3);
            }
            VisionAttentionMode::RowL1 => {
                let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
                scores = scores / denom;
            }
        }
        let value = value.repeat_dim(1, self.n_head);
        scores.matmul(value)
    }

    fn prepend_cls(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _time, dim] = tokens.shape().dims::<3>();
        let cls = self
            .cls_token
            .as_ref()
            .expect("cls token enabled")
            .val()
            .reshape([1, 1, dim])
            .repeat_dim(0, batch);
        let cls = if let Some(cls_pos) = &self.cls_pos {
            cls + cls_pos.val().reshape([1, 1, dim])
        } else {
            cls
        };
        Tensor::cat(vec![cls, tokens], 1)
    }

    fn split_output(&self, tokens: Tensor<B, 3>) -> VisionDragonHatchlingOutput<B> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens
                .clone()
                .slice_dim(1, 0..1)
                .reshape([batch, dim]);
            let patch_tokens = tokens.slice_dim(1, 1..time);
            VisionDragonHatchlingOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(1).reshape([batch, dim]);
            VisionDragonHatchlingOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }
}

#[cfg(feature = "train")]
mod cifar {
    use anyhow::{Result, anyhow};
    use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Int, Tensor, TensorData};
    use rand::prelude::*;
    use std::fs;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const CIFAR_HEIGHT: usize = 32;
    const CIFAR_WIDTH: usize = 32;
    const CIFAR_CHANNELS: usize = 3;
    const CIFAR_IMAGE_BYTES: usize = CIFAR_HEIGHT * CIFAR_WIDTH * CIFAR_CHANNELS;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CifarType {
        Cifar10,
        Cifar100,
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum CifarSplit {
        Train,
        Test,
    }

    pub struct CifarDataset {
        images: Vec<u8>,
        labels: Vec<u8>,
    }

    impl CifarDataset {
        pub fn new<P: AsRef<Path>>(
            root: P,
            cifar_type: CifarType,
            split: CifarSplit,
        ) -> Result<Self> {
            let root = resolve_root(root.as_ref(), cifar_type);
            let (files, record_len, label_offset) = match (cifar_type, split) {
                (CifarType::Cifar10, CifarSplit::Train) => (
                    vec![
                        "data_batch_1.bin",
                        "data_batch_2.bin",
                        "data_batch_3.bin",
                        "data_batch_4.bin",
                        "data_batch_5.bin",
                    ],
                    1 + CIFAR_IMAGE_BYTES,
                    0,
                ),
                (CifarType::Cifar10, CifarSplit::Test) => {
                    (vec!["test_batch.bin"], 1 + CIFAR_IMAGE_BYTES, 0)
                }
                (CifarType::Cifar100, CifarSplit::Train) => {
                    (vec!["train.bin"], 2 + CIFAR_IMAGE_BYTES, 1)
                }
                (CifarType::Cifar100, CifarSplit::Test) => {
                    (vec!["test.bin"], 2 + CIFAR_IMAGE_BYTES, 1)
                }
            };

            let mut images = Vec::new();
            let mut labels = Vec::new();
            for file in files {
                let path = root.join(file);
                let (file_images, file_labels) = read_records(&path, record_len, label_offset)?;
                images.extend_from_slice(&file_images);
                labels.extend_from_slice(&file_labels);
            }

            Ok(Self {
                images,
                labels,
            })
        }

        pub fn len(&self) -> usize {
            self.labels.len()
        }

        pub fn is_empty(&self) -> bool {
            self.labels.is_empty()
        }

        pub fn steps_per_epoch(&self, batch_size: usize) -> usize {
            if batch_size == 0 {
                return 1;
            }
            self.len().div_ceil(batch_size).max(1)
        }

        pub fn sample_batch<B: Backend>(
            &self,
            batch_size: usize,
            device: &B::Device,
        ) -> CifarBatch<B> {
            let len = self.len();
            assert!(len > 0, "cifar dataset is empty");
            let mut rng = thread_rng();

            let mut images = vec![0.0f32; batch_size * CIFAR_IMAGE_BYTES];
            let mut labels = vec![0i64; batch_size];

            for (batch_idx, label) in labels.iter_mut().enumerate() {
                let idx = rng.gen_range(0..len);
                let src_offset = idx * CIFAR_IMAGE_BYTES;
                let dst_offset = batch_idx * CIFAR_IMAGE_BYTES;
                for i in 0..CIFAR_IMAGE_BYTES {
                    images[dst_offset + i] = self.images[src_offset + i] as f32 / 255.0;
                }
                *label = self.labels[idx] as i64;
            }

            let images_tensor = Tensor::<B, 4>::from_data(
                TensorData::new(
                    images,
                    [batch_size, CIFAR_CHANNELS, CIFAR_HEIGHT, CIFAR_WIDTH],
                ),
                device,
            );
            let labels_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(labels, [batch_size]),
                device,
            );
            CifarBatch::new(images_tensor, labels_tensor)
        }
    }

    #[derive(Clone)]
    pub struct CifarBatch<B: Backend> {
        pub images: Tensor<B, 4>,
        pub labels: Tensor<B, 1, Int>,
    }

    impl<B: Backend> CifarBatch<B> {
        pub fn new(images: Tensor<B, 4>, labels: Tensor<B, 1, Int>) -> Self {
            Self { images, labels }
        }
    }

    pub struct CifarDataLoader<B: Backend> {
        dataset: Arc<CifarDataset>,
        batch_size: usize,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        consumed_steps: Option<Arc<AtomicUsize>>,
        device: B::Device,
    }

    impl<B: Backend> Clone for CifarDataLoader<B> {
        fn clone(&self) -> Self {
            Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: self.steps_per_epoch,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: self.device.clone(),
            }
        }
    }

    impl<B: Backend> CifarDataLoader<B> {
        pub fn new(
            dataset: Arc<CifarDataset>,
            batch_size: usize,
            device: &B::Device,
            steps_per_epoch: usize,
            total_steps: Option<usize>,
        ) -> Self {
            let steps_per_epoch = if steps_per_epoch == 0 {
                dataset.steps_per_epoch(batch_size)
            } else {
                steps_per_epoch
            };
            let steps_per_epoch = steps_per_epoch.max(1);
            let total_steps = total_steps.filter(|value| *value > 0);
            let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

            Self {
                dataset,
                batch_size,
                steps_per_epoch,
                total_steps,
                consumed_steps,
                device: device.clone(),
            }
        }
    }

    impl<B> DataLoader<B, CifarBatch<B>> for CifarDataLoader<B>
    where
        B: Backend + 'static,
        B::Device: Clone,
    {
        fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CifarBatch<B>> + 'a> {
            let steps_total =
                if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                    let used = consumed.load(Ordering::Relaxed);
                    if used >= limit {
                        0
                    } else {
                        (limit - used).min(self.steps_per_epoch)
                    }
                } else {
                    self.steps_per_epoch
                };

            Box::new(CifarIterator {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                device: self.device.clone(),
                steps_total,
                step: 0,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.clone(),
            })
        }

        fn num_items(&self) -> usize {
            self.steps_per_epoch * self.batch_size
        }

        fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, CifarBatch<B>>> {
            Arc::new(Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: self.steps_per_epoch,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: device.clone(),
            })
        }

        fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, CifarBatch<B>>> {
            let end = end.min(self.steps_per_epoch);
            let start = start.min(end);
            let steps = (end - start).max(1);

            Arc::new(Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: steps,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: self.device.clone(),
            })
        }
    }

    struct CifarIterator<B: Backend> {
        dataset: Arc<CifarDataset>,
        batch_size: usize,
        device: B::Device,
        steps_total: usize,
        step: usize,
        total_steps: Option<usize>,
        consumed_steps: Option<Arc<AtomicUsize>>,
    }

    impl<B: Backend> Iterator for CifarIterator<B> {
        type Item = CifarBatch<B>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.step >= self.steps_total {
                return None;
            }
            self.step += 1;

            if let Some(counter) = &self.consumed_steps {
                if let Some(limit) = self.total_steps {
                    let previous = counter.fetch_add(1, Ordering::Relaxed);
                    if previous >= limit {
                        return None;
                    }
                } else {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
            Some(self.dataset.sample_batch::<B>(self.batch_size, &self.device))
        }
    }

    impl<B: Backend> DataLoaderIterator<CifarBatch<B>> for CifarIterator<B> {
        fn progress(&self) -> Progress {
            Progress::new(
                self.step * self.batch_size,
                self.steps_total * self.batch_size,
            )
        }
    }

    fn resolve_root(root: &Path, cifar_type: CifarType) -> PathBuf {
        let subdir = match cifar_type {
            CifarType::Cifar10 => "cifar-10-batches-bin",
            CifarType::Cifar100 => "cifar-100-binary",
        };
        let candidate = root.join(subdir);
        if candidate.is_dir() {
            candidate
        } else {
            root.to_path_buf()
        }
    }

    fn read_records(
        path: &Path,
        record_len: usize,
        label_offset: usize,
    ) -> Result<(Vec<u8>, Vec<u8>)> {
        let bytes = fs::read(path)
            .map_err(|err| anyhow!("failed to read {}: {err}", path.display()))?;
        if bytes.len() % record_len != 0 {
            return Err(anyhow!(
                "invalid CIFAR record size in {}: {} bytes (record_len={})",
                path.display(),
                bytes.len(),
                record_len
            ));
        }
        if label_offset >= record_len {
            return Err(anyhow!(
                "label offset out of range for {}",
                path.display()
            ));
        }

        let records = bytes.len() / record_len;
        let mut labels = Vec::with_capacity(records);
        let mut images = Vec::with_capacity(records * CIFAR_IMAGE_BYTES);
        let image_offset = record_len - CIFAR_IMAGE_BYTES;

        for idx in 0..records {
            let start = idx * record_len;
            labels.push(bytes[start + label_offset]);
            let image_start = start + image_offset;
            images.extend_from_slice(&bytes[image_start..image_start + CIFAR_IMAGE_BYTES]);
        }

        Ok((images, labels))
    }
}

#[cfg(feature = "train")]
mod imagenet {
    use anyhow::{Result, anyhow};
    use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
    use burn::tensor::backend::Backend;
    use burn::tensor::{Int, Tensor, TensorData};
    use image::{DynamicImage, GenericImageView, RgbImage};
    use image::imageops::FilterType;
    use rand::prelude::*;
    use std::collections::{HashMap, VecDeque};
    use std::fs::{self, File};
    use std::io::{Read, Seek, SeekFrom};
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::{Mutex, mpsc};
    use std::thread;

    const IMAGE_CHANNELS: usize = 3;
    const BYTES_PER_F32: u64 = 4;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum ImageNetSplit {
        Train,
        Val,
    }

    #[derive(Clone, Debug)]
    pub struct ImageNetAugmentations {
        split: ImageNetSplit,
        image_size: u32,
        resize_short: u32,
        min_scale: f32,
        max_scale: f32,
        min_aspect_ratio: f32,
        max_aspect_ratio: f32,
        flip_prob: f32,
        color_jitter_prob: f32,
        brightness: f32,
        contrast: f32,
        saturation: f32,
        hue: f32,
        grayscale_prob: f32,
        blur_prob: f32,
        blur_sigma_min: f32,
        blur_sigma_max: f32,
        solarize_prob: f32,
        solarize_threshold: u8,
    }

    impl ImageNetAugmentations {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            split: ImageNetSplit,
            image_size: usize,
            resize_short: usize,
            min_scale: f32,
            max_scale: f32,
            min_aspect_ratio: f32,
            max_aspect_ratio: f32,
            flip_prob: f32,
            color_jitter_prob: f32,
            brightness: f32,
            contrast: f32,
            saturation: f32,
            hue: f32,
            grayscale_prob: f32,
            blur_prob: f32,
            blur_sigma_min: f32,
            blur_sigma_max: f32,
            solarize_prob: f32,
            solarize_threshold: u8,
        ) -> Self {
            Self {
                split,
                image_size: image_size as u32,
                resize_short: resize_short as u32,
                min_scale,
                max_scale,
                min_aspect_ratio,
                max_aspect_ratio,
                flip_prob,
                color_jitter_prob,
                brightness,
                contrast,
                saturation,
                hue,
                grayscale_prob,
                blur_prob,
                blur_sigma_min,
                blur_sigma_max,
                solarize_prob,
                solarize_threshold,
            }
        }

        pub fn image_size(&self) -> usize {
            self.image_size as usize
        }

        pub fn apply(&self, image: DynamicImage, rng: &mut impl Rng) -> RgbImage {
            match self.split {
                ImageNetSplit::Train => self.apply_train(image, rng),
                ImageNetSplit::Val => self.apply_val(image),
            }
        }

        fn apply_train(&self, image: DynamicImage, rng: &mut impl Rng) -> RgbImage {
            let mut image = self.random_resized_crop(image, rng);

            if self.flip_prob > 0.0 && rng.r#gen::<f32>() < self.flip_prob {
                image = image.fliph();
            }

            if self.color_jitter_prob > 0.0 && rng.r#gen::<f32>() < self.color_jitter_prob {
                if self.brightness > 0.0 {
                    let delta = rng.gen_range(-self.brightness..self.brightness) * 255.0;
                    image = image.brighten(delta as i32);
                }
                if self.contrast > 0.0 {
                    let delta = rng.gen_range(-self.contrast..self.contrast) * 100.0;
                    image = image.adjust_contrast(delta);
                }
                if self.saturation > 0.0 {
                    let delta = rng.gen_range(-self.saturation..self.saturation);
                    let factor = (1.0 + delta).max(0.0);
                    image = self.adjust_saturation(image, factor);
                }
                if self.hue > 0.0 {
                    let delta = rng.gen_range(-self.hue..self.hue) * 180.0;
                    image = image.huerotate(delta as i32);
                }
            }

            if self.grayscale_prob > 0.0 && rng.r#gen::<f32>() < self.grayscale_prob {
                image = image.grayscale();
            }

            if self.blur_prob > 0.0 && rng.r#gen::<f32>() < self.blur_prob {
                let sigma_min = self.blur_sigma_min.max(0.0);
                let sigma_max = self.blur_sigma_max.max(sigma_min);
                if sigma_max > 0.0 {
                    let sigma = rng.gen_range(sigma_min..=sigma_max);
                    image = image.blur(sigma);
                }
            }

            if self.solarize_prob > 0.0 && rng.r#gen::<f32>() < self.solarize_prob {
                image = self.solarize(image, self.solarize_threshold);
            }

            image.to_rgb8()
        }

        fn apply_val(&self, image: DynamicImage) -> RgbImage {
            let (width, height) = image.dimensions();
            let target = self.resize_short.max(1);

            let (new_width, new_height) = if width < height {
                (target, (height as f32 * (target as f32 / width as f32)).round() as u32)
            } else {
                ((width as f32 * (target as f32 / height as f32)).round() as u32, target)
            };

            let resized = image.resize_exact(new_width, new_height, FilterType::CatmullRom);
            let crop_x = (new_width.saturating_sub(self.image_size)) / 2;
            let crop_y = (new_height.saturating_sub(self.image_size)) / 2;
            let cropped = resized.crop_imm(crop_x, crop_y, self.image_size, self.image_size);
            cropped.to_rgb8()
        }

        fn random_resized_crop(&self, image: DynamicImage, rng: &mut impl Rng) -> DynamicImage {
            let (width, height) = image.dimensions();
            let area = (width * height) as f32;
            let log_min = self.min_aspect_ratio.ln();
            let log_max = self.max_aspect_ratio.ln();

            for _ in 0..10 {
                let target = rng.gen_range(self.min_scale..self.max_scale) * area;
                let aspect = rng.gen_range(log_min..log_max).exp();
                let new_width = (target * aspect).sqrt().round() as u32;
                let new_height = (target / aspect).sqrt().round() as u32;

                if new_width > 0 && new_height > 0 && new_width <= width && new_height <= height {
                    let x = rng.gen_range(0..=width - new_width);
                    let y = rng.gen_range(0..=height - new_height);
                    let cropped = image.crop_imm(x, y, new_width, new_height);
                    return cropped.resize_exact(
                        self.image_size,
                        self.image_size,
                        FilterType::CatmullRom,
                    );
                }
            }

            let side = width.min(height).max(1);
            let x = (width - side) / 2;
            let y = (height - side) / 2;
            let cropped = image.crop_imm(x, y, side, side);
            cropped.resize_exact(self.image_size, self.image_size, FilterType::CatmullRom)
        }

        fn adjust_saturation(&self, image: DynamicImage, factor: f32) -> DynamicImage {
            let mut rgb = image.to_rgb8();
            for pixel in rgb.pixels_mut() {
                let r = pixel[0] as f32;
                let g = pixel[1] as f32;
                let b = pixel[2] as f32;
                let gray = 0.299 * r + 0.587 * g + 0.114 * b;
                let r = (gray + (r - gray) * factor).round().clamp(0.0, 255.0) as u8;
                let g = (gray + (g - gray) * factor).round().clamp(0.0, 255.0) as u8;
                let b = (gray + (b - gray) * factor).round().clamp(0.0, 255.0) as u8;
                *pixel = image::Rgb([r, g, b]);
            }
            DynamicImage::ImageRgb8(rgb)
        }

        fn solarize(&self, image: DynamicImage, threshold: u8) -> DynamicImage {
            let mut rgb = image.to_rgb8();
            for pixel in rgb.pixels_mut() {
                for channel in pixel.0.iter_mut() {
                    if *channel > threshold {
                        *channel = 255 - *channel;
                    }
                }
            }
            DynamicImage::ImageRgb8(rgb)
        }
    }

    #[derive(Clone, Copy, Debug)]
    pub struct VisionNormalize {
        mean: [f32; IMAGE_CHANNELS],
        std: [f32; IMAGE_CHANNELS],
    }

    impl VisionNormalize {
        pub fn new(mean: [f32; IMAGE_CHANNELS], std: [f32; IMAGE_CHANNELS]) -> Self {
            Self { mean, std }
        }

        pub fn apply(&self, image: &RgbImage, buffer: &mut Vec<f32>) {
            let (width, height) = image.dimensions();
            for channel in 0..IMAGE_CHANNELS {
                for y in 0..height {
                    for x in 0..width {
                        let pixel = image.get_pixel(x, y).0[channel] as f32 / 255.0;
                        let value = (pixel - self.mean[channel]) / self.std[channel];
                        buffer.push(value);
                    }
                }
            }
        }
    }

    #[derive(Debug)]
    pub struct DinoFeatureStore {
        cls_file: Mutex<File>,
        patch_file: Mutex<File>,
        feature_dim: usize,
        patch_tokens: usize,
        records: usize,
    }

    #[derive(Debug)]
    struct TeacherBatchData {
        cls: Vec<f32>,
        patch: Vec<f32>,
    }

    impl DinoFeatureStore {
        pub fn new(
            cls_path: &Path,
            patch_path: &Path,
            feature_dim: usize,
            patch_tokens: usize,
            expected_records: Option<usize>,
        ) -> Result<Self> {
            if feature_dim == 0 || patch_tokens == 0 {
                return Err(anyhow!("feature dimensions must be non-zero"));
            }

            let cls_file = File::open(cls_path)
                .map_err(|err| anyhow!("failed to open {}: {err}", cls_path.display()))?;
            let patch_file = File::open(patch_path)
                .map_err(|err| anyhow!("failed to open {}: {err}", patch_path.display()))?;

            let cls_len = cls_file
                .metadata()
                .map_err(|err| anyhow!("failed to read {} metadata: {err}", cls_path.display()))?
                .len();
            let patch_len = patch_file
                .metadata()
                .map_err(|err| anyhow!("failed to read {} metadata: {err}", patch_path.display()))?
                .len();

            let cls_stride = feature_dim as u64 * BYTES_PER_F32;
            let patch_stride = feature_dim as u64 * patch_tokens as u64 * BYTES_PER_F32;
            if cls_len % cls_stride != 0 {
                return Err(anyhow!(
                    "cls feature file size mismatch: {} bytes not divisible by {}",
                    cls_len,
                    cls_stride
                ));
            }
            if patch_len % patch_stride != 0 {
                return Err(anyhow!(
                    "patch feature file size mismatch: {} bytes not divisible by {}",
                    patch_len,
                    patch_stride
                ));
            }

            let cls_records = (cls_len / cls_stride) as usize;
            let patch_records = (patch_len / patch_stride) as usize;
            if cls_records != patch_records {
                return Err(anyhow!(
                    "teacher records mismatch: cls={}, patch={}",
                    cls_records,
                    patch_records
                ));
            }
            if let Some(expected) =
                expected_records.filter(|expected| *expected > cls_records)
            {
                return Err(anyhow!(
                    "teacher records fewer than expected: expected={}, available={}",
                    expected,
                    cls_records
                ));
            }

            Ok(Self {
                cls_file: Mutex::new(cls_file),
                patch_file: Mutex::new(patch_file),
                feature_dim,
                patch_tokens,
                records: cls_records,
            })
        }

        pub fn records(&self) -> usize {
            self.records
        }

        pub fn feature_dim(&self) -> usize {
            self.feature_dim
        }

        pub fn patch_tokens(&self) -> usize {
            self.patch_tokens
        }

        fn load_batch_data(&self, indices: &[usize]) -> Result<TeacherBatchData> {
            let batch = indices.len();
            if batch == 0 {
                return Err(anyhow!("teacher feature batch is empty"));
            }
            let mut cls_data = Vec::with_capacity(batch * self.feature_dim);
            let mut patch_data = Vec::with_capacity(batch * self.feature_dim * self.patch_tokens);

            {
                let mut cls_file = self.cls_file.lock().unwrap();
                for &index in indices {
                    let offset = index as u64 * self.feature_dim as u64 * BYTES_PER_F32;
                    let values = read_f32_block(&mut cls_file, offset, self.feature_dim)?;
                    cls_data.extend_from_slice(&values);
                }
            }

            {
                let mut patch_file = self.patch_file.lock().unwrap();
                let block_len = self.feature_dim * self.patch_tokens;
                for &index in indices {
                    let offset = index as u64 * block_len as u64 * BYTES_PER_F32;
                    let values = read_f32_block(&mut patch_file, offset, block_len)?;
                    patch_data.extend_from_slice(&values);
                }
            }

            Ok(TeacherBatchData {
                cls: cls_data,
                patch: patch_data,
            })
        }

        pub fn load_batch<B: Backend>(
            &self,
            indices: &[usize],
            device: &B::Device,
        ) -> Result<(Tensor<B, 2>, Tensor<B, 3>)> {
            let batch = indices.len();
            let data = self.load_batch_data(indices)?;
            let cls_tensor = Tensor::<B, 2>::from_data(
                TensorData::new(data.cls, [batch, self.feature_dim]),
                device,
            );
            let patch_tensor = Tensor::<B, 3>::from_data(
                TensorData::new(data.patch, [batch, self.patch_tokens, self.feature_dim]),
                device,
            );
            Ok((cls_tensor, patch_tensor))
        }
    }

    #[derive(Clone, Debug)]
    pub struct ImageNetDatasetConfig {
        pub root: PathBuf,
        pub split: ImageNetSplit,
        pub max_records: Option<usize>,
        pub augmentations: ImageNetAugmentations,
        pub local_augmentations: Option<ImageNetAugmentations>,
        pub normalize: VisionNormalize,
        pub teacher: Option<Arc<DinoFeatureStore>>,
        pub views: usize,
        pub local_views: usize,
        pub cache_decoded: bool,
        pub cache_capacity: usize,
    }

    #[derive(Clone, Debug)]
    struct ImageNetSample {
        path: PathBuf,
        label: usize,
    }

    #[derive(Clone)]
    struct ImageCache {
        capacity: usize,
        inner: Arc<Mutex<ImageCacheState>>,
    }

    struct ImageCacheState {
        entries: HashMap<PathBuf, CacheEntry>,
        order: VecDeque<(PathBuf, u64)>,
        tick: u64,
    }

    struct CacheEntry {
        image: Arc<DynamicImage>,
        tick: u64,
    }

    impl ImageCache {
        fn new(capacity: usize) -> Self {
            Self {
                capacity: capacity.max(1),
                inner: Arc::new(Mutex::new(ImageCacheState {
                    entries: HashMap::new(),
                    order: VecDeque::new(),
                    tick: 0,
                })),
            }
        }

        fn get(&self, path: &Path) -> Option<Arc<DynamicImage>> {
            let mut state = self.inner.lock().unwrap();
            let image = state
                .entries
                .get(path)
                .map(|entry| Arc::clone(&entry.image))?;
            state.tick = state.tick.wrapping_add(1);
            let tick = state.tick;
            if let Some(entry) = state.entries.get_mut(path) {
                entry.tick = tick;
            }
            state.order.push_back((path.to_path_buf(), tick));
            Some(image)
        }

        fn insert(&self, path: PathBuf, image: Arc<DynamicImage>) {
            let mut state = self.inner.lock().unwrap();
            state.tick = state.tick.wrapping_add(1);
            let tick = state.tick;
            state
                .entries
                .insert(path.clone(), CacheEntry { image, tick });
            state.order.push_back((path, tick));
            while state.entries.len() > self.capacity {
                while let Some((evict_path, evict_tick)) = state.order.pop_front() {
                    if let Some(entry) = state.entries.get(&evict_path) {
                        if entry.tick == evict_tick {
                            state.entries.remove(&evict_path);
                            break;
                        }
                    }
                }
            }
        }
    }

    #[derive(Clone)]
    pub struct ImageNetDataset {
        samples: Vec<ImageNetSample>,
        num_classes: usize,
        augmentations: ImageNetAugmentations,
        local_augmentations: Option<ImageNetAugmentations>,
        normalize: VisionNormalize,
        teacher: Option<Arc<DinoFeatureStore>>,
        global_views: usize,
        local_views: usize,
        cache: Option<ImageCache>,
    }

    impl ImageNetDataset {
        pub fn new(config: ImageNetDatasetConfig) -> Result<Self> {
            let (mut samples, num_classes) = collect_samples(&config.root)?;
            if let Some(limit) = config
                .max_records
                .filter(|limit| *limit > 0 && *limit < samples.len())
            {
                samples.truncate(limit);
            }
            let global_views = config.views.max(1);
            let local_views = config.local_views;
            if local_views > 0 && config.local_augmentations.is_none() {
                return Err(anyhow!(
                    "local_augmentations required when local_views > 0"
                ));
            }
            let cache = if config.cache_decoded && config.cache_capacity > 0 {
                Some(ImageCache::new(config.cache_capacity))
            } else {
                None
            };

            Ok(Self {
                samples,
                num_classes,
                augmentations: config.augmentations,
                local_augmentations: config.local_augmentations,
                normalize: config.normalize,
                teacher: config.teacher,
                global_views,
                local_views,
                cache,
            })
        }

        pub fn with_teacher(mut self, teacher: Arc<DinoFeatureStore>) -> Self {
            self.teacher = Some(teacher);
            self
        }

        pub fn len(&self) -> usize {
            self.samples.len()
        }

        pub fn is_empty(&self) -> bool {
            self.samples.is_empty()
        }

        pub fn num_classes(&self) -> usize {
            self.num_classes
        }

        pub fn steps_per_epoch(&self, batch_size: usize) -> usize {
            if batch_size == 0 {
                return 1;
            }
            self.len().div_ceil(batch_size).max(1)
        }

        fn load_image_cached(&self, path: &Path) -> Result<DynamicImage> {
            if let Some(cache) = &self.cache {
                if let Some(image) = cache.get(path) {
                    return Ok((*image).clone());
                }
            }

            let image = load_image(path)?;
            if let Some(cache) = &self.cache {
                cache.insert(path.to_path_buf(), Arc::new(image.clone()));
            }
            Ok(image)
        }

        pub fn sample_batch<B: Backend>(
            &self,
            batch_size: usize,
            device: &B::Device,
        ) -> ImageNetBatch<B> {
            self.sample_batch_data(batch_size)
                .unwrap_or_else(|err| panic!("imagenet batch failed: {err}"))
                .into_batch(device)
        }

        fn sample_batch_data(&self, batch_size: usize) -> Result<ImageNetBatchData> {
            if batch_size == 0 {
                return Err(anyhow!("imagenet batch size must be > 0"));
            }
            let len = self.len();
            if len == 0 {
                return Err(anyhow!("imagenet dataset is empty"));
            }
            let mut rng = thread_rng();
            let mut indices = Vec::with_capacity(batch_size);
            for _ in 0..batch_size {
                indices.push(rng.gen_range(0..len));
            }

            let global_image_size = self.augmentations.image_size();
            let mut images =
                Vec::with_capacity(batch_size * global_image_size * global_image_size * IMAGE_CHANNELS);
            let mut target_images = if self.global_views > 1 {
                Some(Vec::with_capacity(
                    batch_size * global_image_size * global_image_size * IMAGE_CHANNELS,
                ))
            } else {
                None
            };
            let mut view_images = if self.local_views == 0 && self.global_views > 1 {
                Some(Vec::with_capacity(
                    batch_size
                        * self.global_views
                        * global_image_size
                        * global_image_size
                        * IMAGE_CHANNELS,
                ))
            } else {
                None
            };
            let mut global_view_images = if self.local_views > 0 {
                Some(Vec::with_capacity(
                    batch_size
                        * self.global_views
                        * global_image_size
                        * global_image_size
                        * IMAGE_CHANNELS,
                ))
            } else {
                None
            };
            let local_image_size = self
                .local_augmentations
                .as_ref()
                .map(|aug| aug.image_size())
                .unwrap_or(0);
            let mut local_view_images = if self.local_views > 0 {
                Some(Vec::with_capacity(
                    batch_size
                        * self.local_views
                        * local_image_size
                        * local_image_size
                        * IMAGE_CHANNELS,
                ))
            } else {
                None
            };
            let mut labels = Vec::with_capacity(batch_size);

            for &index in &indices {
                let sample = &self.samples[index];
                let image = self.load_image_cached(&sample.path)?;
                if self.global_views == 1 && self.local_views == 0 {
                    let primary = self.augmentations.apply(image, &mut rng);
                    self.normalize.apply(&primary, &mut images);
                } else {
                    for view_idx in 0..self.global_views {
                        let view_image = image.clone();
                        let aug = self.augmentations.apply(view_image, &mut rng);
                        if view_idx == 0 {
                            self.normalize.apply(&aug, &mut images);
                        }
                        if view_idx == 1 {
                            if let Some(buffer) = target_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                        }
                        if let Some(buffer) = view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                        if let Some(buffer) = global_view_images.as_mut() {
                            self.normalize.apply(&aug, buffer);
                        }
                    }

                    if self.local_views > 0 {
                        let local_aug = self
                            .local_augmentations
                            .as_ref()
                            .expect("local augmentations required");
                        for _ in 0..self.local_views {
                            let view_image = image.clone();
                            let aug = local_aug.apply(view_image, &mut rng);
                            if let Some(buffer) = local_view_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                        }
                    }
                }
                labels.push(sample.label as i64);
            }

            let (teacher_cls, teacher_patch, teacher_dim, teacher_tokens) = match &self.teacher {
                Some(store) => {
                    let batch = store.load_batch_data(&indices)?;
                    (
                        Some(batch.cls),
                        Some(batch.patch),
                        Some(store.feature_dim()),
                        Some(store.patch_tokens()),
                    )
                }
                None => (None, None, None, None),
            };

            Ok(ImageNetBatchData {
                images,
                target_images,
                view_images,
                global_view_images,
                local_view_images,
                labels,
                teacher_patch,
                teacher_cls,
                batch_size,
                global_image_size,
                local_image_size,
                global_views: self.global_views.max(1),
                local_views: self.local_views,
                teacher_feature_dim: teacher_dim,
                teacher_patch_tokens: teacher_tokens,
            })
        }
    }

    struct ImageNetBatchData {
        images: Vec<f32>,
        target_images: Option<Vec<f32>>,
        view_images: Option<Vec<f32>>,
        global_view_images: Option<Vec<f32>>,
        local_view_images: Option<Vec<f32>>,
        labels: Vec<i64>,
        teacher_patch: Option<Vec<f32>>,
        teacher_cls: Option<Vec<f32>>,
        batch_size: usize,
        global_image_size: usize,
        local_image_size: usize,
        global_views: usize,
        local_views: usize,
        teacher_feature_dim: Option<usize>,
        teacher_patch_tokens: Option<usize>,
    }

    impl ImageNetBatchData {
        fn into_batch<B: Backend>(self, device: &B::Device) -> ImageNetBatch<B> {
            let images_tensor = Tensor::<B, 4>::from_data(
                TensorData::new(
                    self.images,
                    [
                        self.batch_size,
                        IMAGE_CHANNELS,
                        self.global_image_size,
                        self.global_image_size,
                    ],
                ),
                device,
            );
            let labels_tensor = Tensor::<B, 1, Int>::from_data(
                TensorData::new(self.labels, [self.batch_size]),
                device,
            );
            let target_images_tensor = self.target_images.map(|buffer| {
                Tensor::<B, 4>::from_data(
                    TensorData::new(
                        buffer,
                        [
                            self.batch_size,
                            IMAGE_CHANNELS,
                            self.global_image_size,
                            self.global_image_size,
                        ],
                    ),
                    device,
                )
            });
            let view_images_tensor = self.view_images.map(|buffer| {
                Tensor::<B, 5>::from_data(
                    TensorData::new(
                        buffer,
                        [
                            self.batch_size,
                            self.global_views,
                            IMAGE_CHANNELS,
                            self.global_image_size,
                            self.global_image_size,
                        ],
                    ),
                    device,
                )
            });
            let global_view_images_tensor = self.global_view_images.map(|buffer| {
                Tensor::<B, 5>::from_data(
                    TensorData::new(
                        buffer,
                        [
                            self.batch_size,
                            self.global_views,
                            IMAGE_CHANNELS,
                            self.global_image_size,
                            self.global_image_size,
                        ],
                    ),
                    device,
                )
            });
            let local_view_images_tensor = self.local_view_images.map(|buffer| {
                Tensor::<B, 5>::from_data(
                    TensorData::new(
                        buffer,
                        [
                            self.batch_size,
                            self.local_views,
                            IMAGE_CHANNELS,
                            self.local_image_size,
                            self.local_image_size,
                        ],
                    ),
                    device,
                )
            });

            let teacher_cls = self.teacher_cls.map(|data| {
                let dim = self
                    .teacher_feature_dim
                    .expect("teacher feature dim required");
                Tensor::<B, 2>::from_data(
                    TensorData::new(data, [self.batch_size, dim]),
                    device,
                )
            });
            let teacher_patch = self.teacher_patch.map(|data| {
                let dim = self
                    .teacher_feature_dim
                    .expect("teacher feature dim required");
                let tokens = self
                    .teacher_patch_tokens
                    .expect("teacher patch tokens required");
                Tensor::<B, 3>::from_data(
                    TensorData::new(data, [self.batch_size, tokens, dim]),
                    device,
                )
            });

            ImageNetBatch::new(
                images_tensor,
                target_images_tensor,
                view_images_tensor,
                global_view_images_tensor,
                local_view_images_tensor,
                labels_tensor,
                teacher_patch,
                teacher_cls,
            )
        }
    }

    #[derive(Clone)]
    pub struct ImageNetBatch<B: Backend> {
        pub images: Tensor<B, 4>,
        pub target_images: Option<Tensor<B, 4>>,
        pub view_images: Option<Tensor<B, 5>>,
        pub global_view_images: Option<Tensor<B, 5>>,
        pub local_view_images: Option<Tensor<B, 5>>,
        pub labels: Tensor<B, 1, Int>,
        pub teacher_patch: Option<Tensor<B, 3>>,
        pub teacher_cls: Option<Tensor<B, 2>>,
    }

    impl<B: Backend> ImageNetBatch<B> {
        pub fn new(
            images: Tensor<B, 4>,
            target_images: Option<Tensor<B, 4>>,
            view_images: Option<Tensor<B, 5>>,
            global_view_images: Option<Tensor<B, 5>>,
            local_view_images: Option<Tensor<B, 5>>,
            labels: Tensor<B, 1, Int>,
            teacher_patch: Option<Tensor<B, 3>>,
            teacher_cls: Option<Tensor<B, 2>>,
        ) -> Self {
            Self {
                images,
                target_images,
                view_images,
                global_view_images,
                local_view_images,
                labels,
                teacher_patch,
                teacher_cls,
            }
        }
    }

    pub struct ImageNetDataLoader<B: Backend> {
        dataset: Arc<ImageNetDataset>,
        batch_size: usize,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        consumed_steps: Option<Arc<AtomicUsize>>,
        device: B::Device,
        prefetch_batches: usize,
        prefetch_workers: usize,
    }

    impl<B: Backend> Clone for ImageNetDataLoader<B> {
        fn clone(&self) -> Self {
            Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: self.steps_per_epoch,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: self.device.clone(),
                prefetch_batches: self.prefetch_batches,
                prefetch_workers: self.prefetch_workers,
            }
        }
    }

    impl<B: Backend> ImageNetDataLoader<B> {
        pub fn new(
            dataset: Arc<ImageNetDataset>,
            batch_size: usize,
            device: &B::Device,
            steps_per_epoch: usize,
            total_steps: Option<usize>,
            prefetch_batches: usize,
            prefetch_workers: usize,
        ) -> Self {
            let steps_per_epoch = if steps_per_epoch == 0 {
                dataset.steps_per_epoch(batch_size)
            } else {
                steps_per_epoch
            };
            let steps_per_epoch = steps_per_epoch.max(1);
            let total_steps = total_steps.filter(|value| *value > 0);
            let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
            let prefetch_batches = prefetch_batches.min(steps_per_epoch);
            let prefetch_workers = if prefetch_batches == 0 {
                0
            } else {
                prefetch_workers.max(1)
            };

            Self {
                dataset,
                batch_size,
                steps_per_epoch,
                total_steps,
                consumed_steps,
                device: device.clone(),
                prefetch_batches,
                prefetch_workers,
            }
        }
    }

    impl<B> DataLoader<B, ImageNetBatch<B>> for ImageNetDataLoader<B>
    where
        B: Backend + 'static,
        B::Device: Clone,
    {
        fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<ImageNetBatch<B>> + 'a> {
            let steps_total =
                if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                    let used = consumed.load(Ordering::Relaxed);
                    if used >= limit {
                        0
                    } else {
                        (limit - used).min(self.steps_per_epoch)
                    }
                } else {
                    self.steps_per_epoch
                };

            let prefetcher = if self.prefetch_batches > 0 && steps_total > 0 {
                Some(ImageNetPrefetcher::new(
                    Arc::clone(&self.dataset),
                    self.batch_size,
                    steps_total,
                    self.prefetch_batches,
                    self.prefetch_workers,
                ))
            } else {
                None
            };

            Box::new(ImageNetIterator {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                device: self.device.clone(),
                steps_total,
                step: 0,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.clone(),
                prefetcher,
            })
        }

        fn num_items(&self) -> usize {
            self.steps_per_epoch * self.batch_size
        }

        fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, ImageNetBatch<B>>> {
            Arc::new(Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: self.steps_per_epoch,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: device.clone(),
                prefetch_batches: self.prefetch_batches,
                prefetch_workers: self.prefetch_workers,
            })
        }

        fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, ImageNetBatch<B>>> {
            let end = end.min(self.steps_per_epoch);
            let start = start.min(end);
            let steps = (end - start).max(1);

            Arc::new(Self {
                dataset: Arc::clone(&self.dataset),
                batch_size: self.batch_size,
                steps_per_epoch: steps,
                total_steps: self.total_steps,
                consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
                device: self.device.clone(),
                prefetch_batches: self.prefetch_batches,
                prefetch_workers: self.prefetch_workers,
            })
        }
    }

    struct ImageNetPrefetcher {
        rx: Option<mpsc::Receiver<Result<ImageNetBatchData>>>,
        stop: Arc<AtomicBool>,
        handles: Vec<thread::JoinHandle<()>>,
    }

    impl ImageNetPrefetcher {
        fn new(
            dataset: Arc<ImageNetDataset>,
            batch_size: usize,
            steps_total: usize,
            prefetch_batches: usize,
            prefetch_workers: usize,
        ) -> Self {
            let workers = prefetch_workers.max(1);
            let (tx, rx) = mpsc::sync_channel(prefetch_batches.max(1));
            let remaining = Arc::new(AtomicUsize::new(steps_total));
            let stop = Arc::new(AtomicBool::new(false));
            let mut handles = Vec::with_capacity(workers);

            for _ in 0..workers {
                let dataset = Arc::clone(&dataset);
                let tx = tx.clone();
                let remaining = Arc::clone(&remaining);
                let stop = Arc::clone(&stop);
                let handle = thread::spawn(move || {
                    loop {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let decremented = remaining.fetch_update(
                            Ordering::AcqRel,
                            Ordering::Acquire,
                            |value| value.checked_sub(1),
                        );
                        if decremented.is_err() {
                            break;
                        }
                        let result = dataset.sample_batch_data(batch_size);
                        if tx.send(result).is_err() {
                            break;
                        }
                    }
                });
                handles.push(handle);
            }
            drop(tx);

            Self {
                rx: Some(rx),
                stop,
                handles,
            }
        }

        fn recv(&mut self) -> Option<Result<ImageNetBatchData>> {
            let rx = self.rx.as_ref()?;
            rx.recv().ok()
        }
    }

    impl Drop for ImageNetPrefetcher {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Relaxed);
            self.rx.take();
            for handle in self.handles.drain(..) {
                let _ = handle.join();
            }
        }
    }

    struct ImageNetIterator<B: Backend> {
        dataset: Arc<ImageNetDataset>,
        batch_size: usize,
        device: B::Device,
        steps_total: usize,
        step: usize,
        total_steps: Option<usize>,
        consumed_steps: Option<Arc<AtomicUsize>>,
        prefetcher: Option<ImageNetPrefetcher>,
    }

    impl<B: Backend> Iterator for ImageNetIterator<B> {
        type Item = ImageNetBatch<B>;

        fn next(&mut self) -> Option<Self::Item> {
            if self.step >= self.steps_total {
                return None;
            }
            self.step += 1;

            if let Some(counter) = &self.consumed_steps {
                if let Some(limit) = self.total_steps {
                    let previous = counter.fetch_add(1, Ordering::Relaxed);
                    if previous >= limit {
                        return None;
                    }
                } else {
                    counter.fetch_add(1, Ordering::Relaxed);
                }
            }
            let batch_data = if let Some(prefetcher) = &mut self.prefetcher {
                match prefetcher.recv() {
                    Some(Ok(data)) => data,
                    Some(Err(err)) => panic!("imagenet prefetch error: {err}"),
                    None => panic!("imagenet prefetch channel closed early"),
                }
            } else {
                self.dataset
                    .sample_batch_data(self.batch_size)
                    .unwrap_or_else(|err| panic!("imagenet batch failed: {err}"))
            };

            Some(batch_data.into_batch(&self.device))
        }
    }

    impl<B: Backend> DataLoaderIterator<ImageNetBatch<B>> for ImageNetIterator<B> {
        fn progress(&self) -> Progress {
            Progress::new(
                self.step * self.batch_size,
                self.steps_total * self.batch_size,
            )
        }
    }

    fn collect_samples(root: &Path) -> Result<(Vec<ImageNetSample>, usize)> {
        let mut class_dirs: Vec<PathBuf> = fs::read_dir(root)
            .map_err(|err| anyhow!("failed to read {}: {err}", root.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_dir())
            .collect();

        class_dirs.sort_by(|a, b| a.file_name().cmp(&b.file_name()));

        if class_dirs.is_empty() {
            return Err(anyhow!(
                "no class directories found in {}",
                root.display()
            ));
        }

        let mut samples = Vec::new();
        for (label, class_dir) in class_dirs.iter().enumerate() {
            let mut images = collect_images(class_dir)?;
            images.sort();
            for image in images {
                samples.push(ImageNetSample {
                    path: image,
                    label,
                });
            }
        }

        Ok((samples, class_dirs.len()))
    }

    fn collect_images(root: &Path) -> Result<Vec<PathBuf>> {
        let mut images = Vec::new();
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            for entry in fs::read_dir(&dir)
                .map_err(|err| anyhow!("failed to read {}: {err}", dir.display()))?
            {
                let entry = entry.map_err(|err| anyhow!("failed to read dir entry: {err}"))?;
                let path = entry.path();
                if path.is_dir() {
                    stack.push(path);
                } else if is_image_file(&path) {
                    images.push(path);
                }
            }
        }
        Ok(images)
    }

    fn is_image_file(path: &Path) -> bool {
        match path.extension().and_then(|ext| ext.to_str()) {
            Some(ext) => matches!(
                ext.to_ascii_lowercase().as_str(),
                "jpg" | "jpeg" | "png"
            ),
            None => false,
        }
    }

    fn load_image(path: &Path) -> Result<DynamicImage> {
        image::ImageReader::open(path)
            .map_err(|err| anyhow!("failed to open image {}: {err}", path.display()))?
            .decode()
            .map_err(|err| anyhow!("failed to decode {}: {err}", path.display()))
    }

    fn read_f32_block(file: &mut File, offset: u64, len: usize) -> Result<Vec<f32>> {
        let mut buf = vec![0u8; len * BYTES_PER_F32 as usize];
        file.seek(SeekFrom::Start(offset))
            .map_err(|err| anyhow!("failed to seek teacher features: {err}"))?;
        file.read_exact(&mut buf)
            .map_err(|err| anyhow!("failed to read teacher features: {err}"))?;

        let mut out = Vec::with_capacity(len);
        for chunk in buf.chunks_exact(4) {
            let value = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            out.push(value);
        }
        Ok(out)
    }
}

#[cfg(feature = "train")]
pub use cifar::{CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType};
#[cfg(feature = "train")]
pub use imagenet::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};
