use burn::module::{Module, Param};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};

use burn_dragon_core::{
    FusedKernelConfig, ManifoldHyperConnections, lowrank_residual_step, mhc_merge, mhc_split,
};

const ROW_NORM_EPS: f32 = 1e-6;

mod config;
#[cfg(feature = "train")]
mod datasets;

pub use config::*;
#[cfg(feature = "train")]
pub use datasets::{
    CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType, DinoFeatureStore,
    ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};

#[derive(Clone)]
pub struct PatchEmbedOutput<B: Backend> {
    pub tokens: Tensor<B, 3>,
    pub grid: PatchGrid,
}

const PATCH_EMBED_EXPANSION: usize = 4;
const PATCH_EMBED_BLOCKS_PER_STAGE: usize = 1;

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
            VisionPatchEmbedMode::Conv => {
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
pub struct VisionDragonOutput<B: Backend> {
    pub patch_tokens: Tensor<B, 3>,
    pub cls_token: Tensor<B, 2>,
}

#[derive(Clone)]
pub struct VisionDragonMultiOutput<B: Backend> {
    pub patch_tokens: Tensor<B, 4>,
    pub cls_token: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct VisionDragon<B: Backend> {
    steps: usize,
    n_head: usize,
    embed_dim: usize,
    mlp_internal_dim_multiplier: usize,
    use_cls_token: bool,
    attention_mode: VisionAttentionMode,
    use_alibi: bool,
    alibi_slopes: Option<Tensor<B, 1>>,
    latent_activation: VisionLatentActivation,
    kernel: FusedKernelConfig,
    #[module(ignore)]
    trm_graph: VisionTrmGraphConfig,
    trm_x: Option<Linear<B>>,
    trm_v: Option<Linear<B>>,
    trm_y: Option<Linear<B>>,
    trm_enc: Option<Linear<B>>,
    trm_norm: Option<LayerNorm<B>>,
    trm_hub_gate: Option<Linear<B>>,
    grid_height: usize,
    grid_width: usize,
    patch_embed: PatchEmbed<B>,
    dropout: Dropout,
    token_norm: Option<LayerNorm<B>>,
    mhc_layers: Option<Vec<ManifoldHyperConnections<B>>>,
    eye_token: Option<Param<Tensor<B, 2>>>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    projection: VisionProjectionHead<B>,
    cls_token: Option<Param<Tensor<B, 2>>>,
    cls_pos: Option<Param<Tensor<B, 2>>>,
    cls_sync_alpha: f32,
    cross_eye_steps: usize,
}

impl<B: Backend> VisionDragon<B> {
    pub fn new(config: VisionDragonConfig, device: &B::Device) -> Self {
        let patch_embed = PatchEmbed::new(&config, device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let token_norm = if config.token_state_norm {
            Some(LayerNormConfig::new(config.embed_dim).init(device))
        } else {
            None
        };
        let mhc_layers = if config.mhc.enabled && config.mhc.num_streams > 1 {
            let mut layers = Vec::with_capacity(config.steps.max(1));
            for layer_idx in 0..config.steps.max(1) {
                layers.push(ManifoldHyperConnections::new(
                    &config.mhc,
                    layer_idx,
                    device,
                ));
            }
            Some(layers)
        } else {
            None
        };
        let eye_token = if config.num_eyes.max(1) > 1 {
            let eye = Tensor::<B, 2>::random(
                [config.num_eyes.max(1), config.embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            Some(Param::from_tensor(eye))
        } else {
            None
        };

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

        let trm_graph = config.trm_graph.clone();
        let (trm_x, trm_v, trm_y, trm_enc, trm_norm, trm_hub_gate) = if trm_graph.enabled {
            let trm_x = LinearConfig::new(config.embed_dim, trm_graph.rank.max(1)).init(device);
            let trm_v =
                LinearConfig::new(config.embed_dim, trm_graph.value_dim.max(1)).init(device);
            let trm_y =
                LinearConfig::new(trm_graph.value_dim.max(1), trm_graph.rank.max(1)).init(device);
            let trm_enc = LinearConfig::new(trm_graph.rank.max(1), config.embed_dim).init(device);
            let trm_norm = LayerNormConfig::new(trm_graph.value_dim.max(1)).init(device);
            let trm_hub_gate = if trm_graph.hub_count > 1 && trm_graph.hub_gates {
                Some(LinearConfig::new(config.embed_dim, trm_graph.hub_count).init(device))
            } else {
                None
            };
            (
                Some(trm_x),
                Some(trm_v),
                Some(trm_y),
                Some(trm_enc),
                Some(trm_norm),
                trm_hub_gate,
            )
        } else {
            (None, None, None, None, None, None)
        };

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
        let (use_alibi, alibi_slopes) = if config.use_alibi {
            let slopes = burn_dragon_core::kernel::linear_attention::default_alibi_slopes(
                config.n_head.max(1),
            );
            let slopes =
                Tensor::<B, 1>::from_data(TensorData::new(slopes, [config.n_head.max(1)]), device);
            (true, Some(slopes))
        } else {
            (false, None)
        };

        Self {
            steps: config.steps.max(1),
            n_head: config.n_head,
            embed_dim: config.embed_dim,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            use_cls_token: config.use_cls_token,
            attention_mode: config.attention_mode,
            use_alibi,
            alibi_slopes,
            latent_activation: config.latent_activation,
            kernel: config.fused_kernels,
            patch_embed,
            dropout,
            token_norm,
            mhc_layers,
            eye_token,
            encoder,
            encoder_v,
            decoder,
            projection,
            cls_token,
            cls_pos,
            cls_sync_alpha: config.cls_sync_alpha,
            cross_eye_steps: config.cross_eye_steps,
            trm_graph,
            trm_x,
            trm_v,
            trm_y,
            trm_enc,
            trm_norm,
            trm_hub_gate,
            grid_height: config.pos_max_height.max(1),
            grid_width: config.pos_max_width.max(1),
        }
    }

    pub fn patch_embed(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward(images)
    }

    pub fn patch_embed_raw(&self, images: Tensor<B, 4>) -> PatchEmbedOutput<B> {
        self.patch_embed.forward_raw(images)
    }

    pub fn patch_size(&self) -> usize {
        self.patch_embed.patch_size()
    }

    pub fn add_patch_position(&self, tokens: Tensor<B, 3>, grid: PatchGrid) -> Tensor<B, 3> {
        self.patch_embed.add_position(tokens, grid)
    }

    pub fn add_patch_position_multi(&self, tokens: Tensor<B, 4>, grid: PatchGrid) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        let flat = tokens.reshape([batch * streams, time, dim]);
        let flat = self.patch_embed.add_position(flat, grid);
        flat.reshape([batch, streams, time, dim])
    }

    pub fn project_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.projection.forward(tokens)
    }

    pub fn forward_images(&self, images: Tensor<B, 4>) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens(patch.tokens)
    }

    pub fn forward_images_steps(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps(patch.tokens, steps)
    }

    pub fn forward_images_steps_rollout(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let patch = self.patch_embed.forward(images);
        self.forward_tokens_steps_rollout(patch.tokens, steps, backprop_steps)
    }

    pub fn forward_patches(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
    ) -> VisionDragonOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens(tokens)
    }

    pub fn forward_patches_steps(
        &self,
        patch_tokens: Tensor<B, 3>,
        grid: PatchGrid,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.patch_embed.add_position(patch_tokens, grid);
        self.forward_tokens_steps(tokens, steps)
    }

    pub fn forward_tokens(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens(tokens);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout(tokens, steps, backprop_steps);
        let projected = self.projection.forward(tokens);
        self.split_output(projected)
    }

    pub fn forward_tokens_embed(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens(tokens);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps(tokens, steps);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout(tokens, steps, backprop_steps);
        self.split_output(tokens)
    }

    /// Same as `forward_tokens_embed_steps_rollout`, but does not clamp `steps` to `self.steps`.
    /// Useful for validation-time extrapolation beyond the training rollout depth.
    pub fn forward_tokens_embed_steps_rollout_unbounded(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonOutput<B> {
        let tokens = self.encode_tokens_steps_rollout_unbounded(tokens, steps, backprop_steps);
        self.split_output(tokens)
    }

    pub fn forward_tokens_embed_steps_rollout_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> VisionDragonMultiOutput<B> {
        let tokens = self.encode_tokens_steps_rollout_multi(tokens, steps, backprop_steps);
        self.split_output_multi(tokens)
    }

    fn encode_tokens(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        self.encode_tokens_steps(tokens, self.steps)
    }

    fn encode_tokens_steps(&self, tokens: Tensor<B, 3>, steps: usize) -> Tensor<B, 3> {
        let steps = steps.max(1).min(self.steps);
        self.encode_tokens_steps_inner(tokens, steps, 0, true)
    }

    fn encode_tokens_steps_rollout(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 3> {
        let steps = steps.max(1).min(self.steps);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner(tokens, steps, detach_until, true)
    }

    fn encode_tokens_steps_rollout_unbounded(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 3> {
        let steps = steps.max(1);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner(tokens, steps, detach_until, true)
    }

    fn encode_tokens_steps_rollout_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
    ) -> Tensor<B, 4> {
        let steps = steps.max(1).min(self.steps);
        let backprop_steps = backprop_steps.max(1).min(steps);
        let detach_until = steps.saturating_sub(backprop_steps);
        self.encode_tokens_steps_inner_multi(tokens, steps, detach_until)
    }

    fn encode_tokens_steps_inner(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        if self.trm_graph.enabled {
            return self.encode_tokens_steps_inner_graph(tokens, steps, detach_until, add_cls);
        }

        self.encode_tokens_steps_inner_default(tokens, steps, detach_until, add_cls)
    }

    fn trm_graph_fallback(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        reason: &str,
    ) -> Tensor<B, 3> {
        match self.trm_graph.grid_mismatch_policy {
            VisionTrmGridMismatchPolicy::FallbackDefault => {
                self.encode_tokens_steps_inner_default(tokens, steps, detach_until, false)
            }
            VisionTrmGridMismatchPolicy::Error => {
                panic!(
                    "TRM graph path unavailable: {reason}. Set `vision.trm_graph.grid_mismatch_policy = \"fallback_default\"` to allow explicit fallback."
                )
            }
        }
    }

    fn encode_tokens_steps_inner_default(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, _] = tokens.shape().dims::<3>();
        let mut current = tokens.reshape([batch, 1, time, self.embed_dim]);
        current = self.apply_token_norm(current);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;

        for step_idx in 0..steps {
            let output = lowrank_residual_step(
                current,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                |query, value| self.full_attention(query, value),
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );
            current = output.next;
            if step_idx < detach_until {
                current = current.detach();
            }
        }

        current.reshape([batch, time, self.embed_dim])
    }

    fn encode_tokens_steps_inner_graph(
        &self,
        tokens: Tensor<B, 3>,
        steps: usize,
        detach_until: usize,
        add_cls: bool,
    ) -> Tensor<B, 3> {
        let tokens = if add_cls && self.use_cls_token {
            self.prepend_cls(tokens)
        } else {
            tokens
        };

        let [batch, time, dim] = tokens.shape().dims::<3>();
        if batch == 0 || time == 0 || dim == 0 {
            return tokens;
        }

        let grid_height = self.grid_height.max(1);
        let grid_width = self.grid_width.max(1);
        let patch_count = grid_height * grid_width;
        let (patch_tokens, has_cls) = if self.use_cls_token && time == patch_count + 1 {
            let patch = tokens.clone().slice_dim(1, 1..time);
            (patch, true)
        } else {
            (tokens.clone(), false)
        };

        let patch_len = patch_tokens.shape().dims::<3>()[1];
        if patch_len != patch_count {
            return self.trm_graph_fallback(
                tokens,
                steps,
                detach_until,
                &format!(
                    "token count mismatch (got {patch_len}, expected {patch_count} from grid {}x{})",
                    grid_height, grid_width
                ),
            );
        }

        let trm_x = match self.trm_x.as_ref() {
            Some(layer) => layer,
            None => {
                return self.trm_graph_fallback(
                    tokens,
                    steps,
                    detach_until,
                    "missing TRM graph projection layer `trm_x`",
                );
            }
        };
        let trm_v = match self.trm_v.as_ref() {
            Some(layer) => layer,
            None => {
                return self.trm_graph_fallback(
                    tokens,
                    steps,
                    detach_until,
                    "missing TRM graph projection layer `trm_v`",
                );
            }
        };
        let trm_y = match self.trm_y.as_ref() {
            Some(layer) => layer,
            None => {
                return self.trm_graph_fallback(
                    tokens,
                    steps,
                    detach_until,
                    "missing TRM graph projection layer `trm_y`",
                );
            }
        };
        let trm_enc = match self.trm_enc.as_ref() {
            Some(layer) => layer,
            None => {
                return self.trm_graph_fallback(
                    tokens,
                    steps,
                    detach_until,
                    "missing TRM graph projection layer `trm_enc`",
                );
            }
        };
        let trm_norm = match self.trm_norm.as_ref() {
            Some(layer) => layer,
            None => {
                return self.trm_graph_fallback(
                    tokens,
                    steps,
                    detach_until,
                    "missing TRM graph normalization layer `trm_norm`",
                );
            }
        };

        let device = tokens.device();
        let rank = self.trm_graph.rank.max(1);
        let value_dim = self.trm_graph.value_dim.max(1);
        let hub_count = self.trm_graph.hub_count.max(1);
        let decay = self.trm_graph.decay.clamp(0.0, 1.0);
        let coarse_stride = self.trm_graph.coarse_stride.max(1);

        let mut h8 = patch_tokens
            .reshape([batch, grid_height, grid_width, dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3);
        h8 = self.apply_embed_norm_spatial(h8);

        let h32_height = grid_height / coarse_stride.max(1);
        let h32_width = grid_width / coarse_stride.max(1);
        let mut h32 = if coarse_stride > 1 {
            let pooled = h8
                .clone()
                .reshape([
                    batch,
                    dim,
                    h32_height.max(1),
                    coarse_stride,
                    h32_width.max(1),
                    coarse_stride,
                ])
                .sum_dims_squeeze::<4, usize>(&[3, 5])
                .div_scalar((coarse_stride * coarse_stride) as f32);
            self.apply_embed_norm_spatial(pooled)
        } else {
            h8.clone()
        };

        let mut mem8 =
            Tensor::<B, 5>::zeros([batch, rank, value_dim, grid_height, grid_width], &device);
        let mut mem32 = Tensor::<B, 5>::zeros(
            [batch, rank, value_dim, h32_height.max(1), h32_width.max(1)],
            &device,
        );
        let mut mem_hub = Tensor::<B, 4>::zeros([batch, hub_count, rank, value_dim], &device);

        for step_idx in 0..steps {
            let x8 = activation::relu(self.project_spatial(h8.clone(), trm_x));
            let v8 = self.project_spatial(h8.clone(), trm_v);
            let x32 = activation::relu(self.project_spatial(h32.clone(), trm_x));
            let v32 = self.project_spatial(h32.clone(), trm_v);

            let msg8_local = self.trm_local_read(mem8.clone(), x8.clone());
            let msg32_local = self.trm_local_read(mem32.clone(), x32.clone());
            let msg8_down = if coarse_stride > 1 {
                self.trm_cross_scale_read(mem32.clone(), x8.clone(), coarse_stride)
            } else {
                self.trm_contract(mem32.clone(), x8.clone())
            };

            let (hub_w8, hub_w32) = self.trm_hub_weights(h8.clone(), h32.clone(), hub_count);
            let msg8_hub = self.trm_hub_read(mem_hub.clone(), x8.clone(), hub_w8.clone());
            let msg32_hub = self.trm_hub_read(mem_hub.clone(), x32.clone(), hub_w32.clone());

            let msg8 = msg8_local + msg8_down + msg8_hub;
            let msg32 = msg32_local + msg32_hub;

            h8 = self.trm_update_state(h8, x8.clone(), msg8, trm_y, trm_enc, trm_norm);
            h32 = self.trm_update_state(h32, x32.clone(), msg32, trm_y, trm_enc, trm_norm);

            let u8 = self.trm_outer_product(x8, v8);
            let u32 = self.trm_outer_product(x32, v32);
            let u8_pool = if coarse_stride > 1 {
                self.trm_pool_outer(u8.clone(), coarse_stride)
            } else {
                u8.clone()
            };
            mem8 = mem8.mul_scalar(decay).add(u8.clone());
            mem32 = mem32.mul_scalar(decay).add(u32.clone()).add(u8_pool);

            mem_hub =
                self.trm_update_hub(mem_hub, u8.clone(), u32, hub_w8, hub_w32, hub_count, decay);

            if step_idx < detach_until {
                h8 = h8.detach();
                h32 = h32.detach();
                mem8 = mem8.detach();
                mem32 = mem32.detach();
                mem_hub = mem_hub.detach();
            }
        }

        let patch_tokens = h8
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch, patch_count, dim]);
        if has_cls {
            let cls = patch_tokens.clone().mean_dim(1).reshape([batch, 1, dim]);
            Tensor::cat(vec![cls, patch_tokens], 1)
        } else {
            patch_tokens
        }
    }

    fn encode_tokens_steps_inner_multi(
        &self,
        tokens: Tensor<B, 4>,
        steps: usize,
        detach_until: usize,
    ) -> Tensor<B, 4> {
        let tokens = if self.use_cls_token {
            let [batch, streams, time, dim] = tokens.shape().dims::<4>();
            let flat = tokens.reshape([batch * streams, time, dim]);
            let flat = self.prepend_cls(flat);
            let [flat_batch, time, dim] = flat.shape().dims::<3>();
            let streams = (flat_batch / batch).max(1);
            flat.reshape([batch, streams, time, dim])
        } else {
            tokens
        };

        let [batch, streams, time, _] = tokens.shape().dims::<4>();
        let mut current = tokens.reshape([batch, streams, time, self.embed_dim]);
        current = self.apply_token_norm(current);

        if let Some(eye_token) = &self.eye_token {
            let eye = eye_token
                .val()
                .reshape([1, streams, 1, self.embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(2, time);
            current = current + eye;
        }
        current = self.sync_cls_tokens_multi(current);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let decoder = self.decoder.val();
        let fused =
            self.kernel.enabled && matches!(self.latent_activation, VisionLatentActivation::Relu);
        let apply_threshold = matches!(self.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.kernel.block_sparse.latent;

        for step_idx in 0..steps {
            let mhc = self
                .mhc_layers
                .as_ref()
                .map(|layers| &layers[step_idx.min(layers.len().saturating_sub(1))]);
            let (branch_input, residuals_base, beta) = mhc_split(mhc, current);

            let [batch, views, time, dim] = branch_input.shape().dims::<4>();
            let branch_flat = branch_input.reshape([batch * views, 1, time, dim]);

            let output = lowrank_residual_step(
                branch_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                apply_threshold,
                latent_pattern,
                |query, value| self.full_attention(query, value),
                |values| self.apply_latent_activation(values),
                |values| self.apply_token_norm(values),
            );

            let branch_out = output.next.reshape([batch, views, time, dim]);
            let next = mhc_merge(mhc, branch_out, residuals_base, beta);

            current = self.sync_cls_tokens_multi(self.apply_token_norm(next));
            if step_idx < detach_until {
                current = current.detach();
            }
        }

        if self.cross_eye_steps > 0 && streams > 1 && time > 0 {
            let cross_steps = self.cross_eye_steps.min(self.steps);
            if cross_steps > 0 {
                let flat = current.reshape([batch, streams * time, self.embed_dim]);
                let mixed = self.encode_tokens_steps_inner(flat, cross_steps, 0, false);
                current = self.sync_cls_tokens_multi(self.apply_token_norm(mixed.reshape([
                    batch,
                    streams,
                    time,
                    self.embed_dim,
                ])));
            }
        }

        current.reshape([batch, streams, time, self.embed_dim])
    }

    fn apply_embed_norm_spatial(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        match &self.token_norm {
            Some(norm) => {
                let [batch, dim, height, width] = input.shape().dims::<4>();
                if batch == 0 || dim == 0 || height == 0 || width == 0 {
                    return input;
                }
                let flat = input.swap_dims(1, 3).swap_dims(1, 2);
                let flat = norm.forward(flat);
                flat.swap_dims(1, 2).swap_dims(1, 3)
            }
            None => input,
        }
    }

    fn apply_value_norm_spatial(&self, norm: &LayerNorm<B>, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return input;
        }
        let flat = input.swap_dims(1, 3).swap_dims(1, 2);
        let flat = norm.forward(flat);
        flat.swap_dims(1, 2).swap_dims(1, 3)
    }

    fn project_spatial(&self, input: Tensor<B, 4>, layer: &Linear<B>) -> Tensor<B, 4> {
        let [batch, dim, height, width] = input.shape().dims::<4>();
        if batch == 0 || dim == 0 || height == 0 || width == 0 {
            return input;
        }
        let flat = input
            .swap_dims(1, 3)
            .swap_dims(1, 2)
            .reshape([batch * height * width, dim]);
        let flat = layer.forward(flat);
        let out_dim = flat.shape().dims::<2>()[1];
        flat.reshape([batch, height, width, out_dim])
            .swap_dims(1, 3)
            .swap_dims(2, 3)
    }

    fn trm_shift(input: Tensor<B, 4>, dy: isize, dx: isize) -> Tensor<B, 4> {
        let [batch, channels, height, width] = input.shape().dims::<4>();
        if height == 0 || width == 0 {
            return input;
        }
        let device = input.device();
        let mut out = input;

        if dy != 0 {
            let shift = dy.unsigned_abs();
            if shift >= height {
                out = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
            } else if dy > 0 {
                let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
                let cropped = out.slice_dim(2, 0..(height - shift));
                out = Tensor::cat(vec![pad, cropped], 2);
            } else {
                let pad = Tensor::<B, 4>::zeros([batch, channels, shift, width], &device);
                let cropped = out.slice_dim(2, shift..height);
                out = Tensor::cat(vec![cropped, pad], 2);
            }
        }

        if dx != 0 {
            let shift = dx.unsigned_abs();
            if shift >= width {
                out = Tensor::<B, 4>::zeros([batch, channels, height, width], &device);
            } else if dx > 0 {
                let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
                let cropped = out.slice_dim(3, 0..(width - shift));
                out = Tensor::cat(vec![pad, cropped], 3);
            } else {
                let pad = Tensor::<B, 4>::zeros([batch, channels, height, shift], &device);
                let cropped = out.slice_dim(3, shift..width);
                out = Tensor::cat(vec![cropped, pad], 3);
            }
        }

        out
    }

    fn trm_contract(&self, memory: Tensor<B, 5>, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, rank, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || rank == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &memory.device(),
            );
        }
        let query = query.unsqueeze_dim::<5>(2);
        memory.mul(query).sum_dims_squeeze::<4, usize>(&[1])
    }

    fn trm_local_read(&self, memory: Tensor<B, 5>, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &memory.device(),
            );
        }
        let mut acc = Tensor::<B, 4>::zeros([batch, value_dim, height, width], &memory.device());
        let radius = self.trm_graph.local_radius.max(1) as isize;
        let allow_diagonals = self.trm_graph.local_diagonals;
        if self.trm_graph.local_self {
            acc = acc + self.trm_contract(memory.clone(), query.clone());
        }
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dy == 0 && dx == 0 {
                    continue;
                }
                if !allow_diagonals && dy != 0 && dx != 0 {
                    continue;
                }
                let shifted = Self::trm_shift(query.clone(), dy, dx);
                let msg = self.trm_contract(memory.clone(), shifted);
                let msg = Self::trm_shift(msg, -dy, -dx);
                acc = acc + msg;
            }
        }
        acc
    }

    fn trm_cross_scale_read(
        &self,
        memory: Tensor<B, 5>,
        query: Tensor<B, 4>,
        scale: usize,
    ) -> Tensor<B, 4> {
        let scale = scale.max(1);
        if scale == 1 {
            return self.trm_contract(memory, query);
        }
        let mut up = memory.repeat_dim(3, scale).repeat_dim(4, scale);
        let [_, _, height, width] = query.shape().dims::<4>();
        let [_, _, _, up_h, up_w] = up.shape().dims::<5>();
        if up_h != height {
            up = up.slice_dim(3, 0..height.min(up_h));
        }
        if up_w != width {
            up = up.slice_dim(4, 0..width.min(up_w));
        }
        self.trm_contract(up, query)
    }

    fn trm_outer_product(&self, x: Tensor<B, 4>, v: Tensor<B, 4>) -> Tensor<B, 5> {
        let x = x.unsqueeze_dim::<5>(2);
        let v = v.unsqueeze_dim::<5>(1);
        x.mul(v)
    }

    fn trm_pool_outer(&self, u: Tensor<B, 5>, scale: usize) -> Tensor<B, 5> {
        let scale = scale.max(1);
        if scale == 1 {
            return u;
        }
        let [batch, rank, value_dim, height, width] = u.shape().dims::<5>();
        let pooled_height = height / scale;
        let pooled_width = width / scale;
        if pooled_height == 0 || pooled_width == 0 {
            return Tensor::<B, 5>::zeros(
                [
                    batch,
                    rank,
                    value_dim,
                    pooled_height.max(1),
                    pooled_width.max(1),
                ],
                &u.device(),
            );
        }
        u.reshape([
            batch,
            rank,
            value_dim,
            pooled_height,
            scale,
            pooled_width,
            scale,
        ])
        .sum_dims_squeeze::<5, usize>(&[4, 6])
    }

    fn trm_update_state(
        &self,
        state: Tensor<B, 4>,
        x: Tensor<B, 4>,
        msg: Tensor<B, 4>,
        trm_y: &Linear<B>,
        trm_enc: &Linear<B>,
        trm_norm: &LayerNorm<B>,
    ) -> Tensor<B, 4> {
        let msg = self.apply_value_norm_spatial(trm_norm, msg);
        let y = activation::relu(self.project_spatial(msg, trm_y));
        let u = y.mul(x);
        let delta = self.project_spatial(u, trm_enc);
        let next = state + delta;
        self.apply_embed_norm_spatial(next)
    }

    fn trm_hub_weights(
        &self,
        h8: Tensor<B, 4>,
        h32: Tensor<B, 4>,
        hub_count: usize,
    ) -> (Option<Tensor<B, 4>>, Option<Tensor<B, 4>>) {
        if hub_count <= 1 {
            return (None, None);
        }
        let hub_gate = self.trm_hub_gate.as_ref();
        let w8 = self.trm_hub_weights_single(h8, hub_count, hub_gate);
        let w32 = self.trm_hub_weights_single(h32, hub_count, hub_gate);
        (Some(w8), Some(w32))
    }

    fn trm_hub_weights_single(
        &self,
        h: Tensor<B, 4>,
        hub_count: usize,
        hub_gate: Option<&Linear<B>>,
    ) -> Tensor<B, 4> {
        let [batch, _, height, width] = h.shape().dims::<4>();
        let device = h.device();
        if let Some(gate) = hub_gate {
            let weights = self.project_spatial(h, gate);
            let weights = activation::relu(weights);
            let denom = weights.clone().sum_dim(1).add_scalar(ROW_NORM_EPS);
            weights / denom
        } else {
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &device)
                .div_scalar(hub_count as f32)
        }
    }

    fn trm_hub_read(
        &self,
        hub: Tensor<B, 4>,
        query: Tensor<B, 4>,
        weights: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        let [batch, hubs, rank, value_dim] = hub.shape().dims::<4>();
        let [_, _, height, width] = query.shape().dims::<4>();
        if batch == 0 || hubs == 0 || rank == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros(
                [batch.max(1), value_dim.max(1), height.max(1), width.max(1)],
                &hub.device(),
            );
        }
        let hub_exp = hub.unsqueeze_dim::<5>(4).unsqueeze_dim::<6>(5);
        let query_exp = query.unsqueeze_dim::<5>(1).unsqueeze_dim::<6>(3);
        let msg = hub_exp.mul(query_exp).sum_dims_squeeze::<5, usize>(&[2]);
        match weights {
            Some(w) => {
                let w = w.unsqueeze_dim::<5>(2);
                msg.mul(w).sum_dims_squeeze::<4, usize>(&[1])
            }
            None => {
                let mut reduced = msg.sum_dims_squeeze::<4, usize>(&[1]);
                if hubs > 1 {
                    reduced = reduced.div_scalar(hubs as f32);
                }
                reduced
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trm_update_hub(
        &self,
        hub: Tensor<B, 4>,
        u8: Tensor<B, 5>,
        u32: Tensor<B, 5>,
        hub_w8: Option<Tensor<B, 4>>,
        hub_w32: Option<Tensor<B, 4>>,
        hub_count: usize,
        decay: f32,
    ) -> Tensor<B, 4> {
        if hub_count <= 1 {
            let sum8 = u8.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let sum32 = u32.sum_dims_squeeze::<3, usize>(&[3, 4]);
            let delta = (sum8 + sum32).unsqueeze_dim::<4>(1);
            return hub.mul_scalar(decay).add(delta);
        }

        let w8 = hub_w8.unwrap_or_else(|| {
            let [batch, _, _, height, width] = u8.shape().dims::<5>();
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &u8.device())
                .div_scalar(hub_count as f32)
        });
        let w32 = hub_w32.unwrap_or_else(|| {
            let [batch, _, _, height, width] = u32.shape().dims::<5>();
            Tensor::<B, 4>::ones([batch, hub_count, height, width], &u32.device())
                .div_scalar(hub_count as f32)
        });

        let delta8 = self.trm_weighted_global_sum(u8, w8);
        let delta32 = self.trm_weighted_global_sum(u32, w32);
        let delta = delta8 + delta32;
        hub.mul_scalar(decay).add(delta)
    }

    fn trm_weighted_global_sum(&self, u: Tensor<B, 5>, w: Tensor<B, 4>) -> Tensor<B, 4> {
        let u = u.unsqueeze_dim::<6>(1);
        let w = w.unsqueeze_dim::<5>(2).unsqueeze_dim::<6>(3);
        u.mul(w).sum_dims_squeeze::<4, usize>(&[4, 5])
    }

    fn apply_token_norm<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        match &self.token_norm {
            Some(norm) => norm.forward(tokens),
            None => tokens,
        }
    }

    fn apply_latent_activation<const D: usize>(&self, values: Tensor<B, D>) -> Tensor<B, D> {
        match self.latent_activation {
            VisionLatentActivation::Relu => activation::relu(values),
            VisionLatentActivation::Gelu => activation::gelu(values),
            VisionLatentActivation::Identity => values,
        }
    }

    fn sync_cls_tokens_multi(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        if !(self.use_cls_token && self.cls_sync_alpha > 0.0) {
            return tokens;
        }
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if streams <= 1 || time == 0 {
            return tokens;
        }
        let alpha = self.cls_sync_alpha.clamp(0.0, 1.0);
        let cls = tokens.clone().slice_dim(2, 0..1);
        let shared = cls
            .clone()
            .sum_dim(1)
            .mul_scalar(1.0 / streams as f32)
            .reshape([batch, 1, 1, dim])
            .repeat_dim(1, streams);
        let blended = cls.mul_scalar(1.0 - alpha) + shared.mul_scalar(alpha);
        if time <= 1 {
            return blended;
        }
        let rest = tokens.slice_dim(2, 1..time);
        Tensor::cat(vec![blended, rest], 2)
    }

    fn full_attention(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        let k = query.clone();
        let query_scaled = query.clone().div_scalar(scale);
        let mut scores = query_scaled.matmul(k.swap_dims(2, 3));
        if self.use_alibi
            && let Some(slopes) = self.alibi_slopes.as_ref()
        {
            let device = query.device();
            let [_, heads, time, _] = query.shape().dims::<4>();
            let slopes = slopes.clone().reshape([1, heads, 1, 1]);
            let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, time, 1]);
            let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, 1, time]);
            let alibi = slopes * (pos_col - pos_row);
            scores = scores + alibi;
        }
        match self.attention_mode {
            VisionAttentionMode::Softmax => {
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

    fn split_output(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens.clone().slice_dim(1, 0..1).reshape([batch, dim]);
            let patch_tokens = tokens.slice_dim(1, 1..time);
            VisionDragonOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(1).reshape([batch, dim]);
            VisionDragonOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }

    fn split_output_multi(&self, tokens: Tensor<B, 4>) -> VisionDragonMultiOutput<B> {
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens
                .clone()
                .slice_dim(2, 0..1)
                .reshape([batch, streams, dim]);
            let patch_tokens = tokens.slice_dim(2, 1..time);
            VisionDragonMultiOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(2).reshape([batch, streams, dim]);
            VisionDragonMultiOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }
}
