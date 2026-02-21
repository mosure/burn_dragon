use burn::module::{
    AutodiffModule, Content, Devices, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper,
    ModuleVisitor, Param,
};
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{
    Dropout, DropoutConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig2d,
};
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Distribution as TensorDistribution, Tensor, TensorData, activation, Int};
use serde::{Deserialize, Serialize};

use burn_dragon_core::{
    FusedKernelConfig, ManifoldHyperConnections, ManifoldHyperConnectionsConfig,
    lowrank_residual_step, mhc_merge, mhc_split,
};

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionPatchEmbedMode {
    #[default]
    Conv,
    Linear,
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionLatentActivation {
    #[default]
    Relu,
    Gelu,
    Identity,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionTrmGridMismatchPolicy {
    #[default]
    Error,
    FallbackDefault,
}

impl core::fmt::Display for VisionAttentionMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionPatchEmbedMode {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionLatentActivation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl core::fmt::Display for VisionTrmGridMismatchPolicy {
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

impl<B: Backend> Module<B> for VisionPatchEmbedMode {
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

impl<B: Backend> Module<B> for VisionLatentActivation {
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

impl<B: Backend> Module<B> for VisionTrmGridMismatchPolicy {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionPatchEmbedMode {
    type InnerModule = VisionPatchEmbedMode;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionLatentActivation {
    type InnerModule = VisionLatentActivation;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmGridMismatchPolicy {
    type InnerModule = VisionTrmGridMismatchPolicy;

    fn valid(&self) -> Self::InnerModule {
        *self
    }
}

impl ModuleDisplayDefault for VisionAttentionMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionPatchEmbedMode {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionLatentActivation {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplayDefault for VisionTrmGridMismatchPolicy {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionAttentionMode {}

impl ModuleDisplay for VisionPatchEmbedMode {}

impl ModuleDisplay for VisionLatentActivation {}

impl ModuleDisplay for VisionTrmGridMismatchPolicy {}

impl ModuleDisplayDefault for VisionTrmGraphConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("enabled", &self.enabled)
            .add("coarse_stride", &self.coarse_stride)
            .add("hub_count", &self.hub_count)
            .add("rank", &self.rank)
            .add("value_dim", &self.value_dim)
            .add("local_radius", &self.local_radius)
            .add("local_diagonals", &self.local_diagonals)
            .add("local_self", &self.local_self)
            .add("decay", &self.decay)
            .add("hub_gates", &self.hub_gates)
            .add("grid_mismatch_policy", &self.grid_mismatch_policy)
            .optional()
    }
}

impl ModuleDisplay for VisionTrmGraphConfig {}

#[derive(Clone, Debug, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct VisionTrmGraphConfig {
    pub enabled: bool,
    pub coarse_stride: usize,
    pub hub_count: usize,
    pub rank: usize,
    pub value_dim: usize,
    pub local_radius: usize,
    pub local_diagonals: bool,
    pub local_self: bool,
    pub decay: f32,
    pub hub_gates: bool,
    pub grid_mismatch_policy: VisionTrmGridMismatchPolicy,
}

impl Default for VisionTrmGraphConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            coarse_stride: 4,
            hub_count: 1,
            rank: 4,
            value_dim: 32,
            local_radius: 1,
            local_diagonals: false,
            local_self: false,
            decay: 0.9,
            hub_gates: true,
            grid_mismatch_policy: VisionTrmGridMismatchPolicy::Error,
        }
    }
}

impl<B: Backend> Module<B> for VisionTrmGraphConfig {
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

impl<B: AutodiffBackend> AutodiffModule<B> for VisionTrmGraphConfig {
    type InnerModule = VisionTrmGraphConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }
}

#[derive(Clone, Debug)]
pub struct VisionDragonConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub patch_embed_mode: VisionPatchEmbedMode,
    pub in_channels: usize,
    pub embed_dim: usize,
    pub steps: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub dropout: f64,
    pub projection_dim: usize,
    pub projection_hidden_dim: usize,
    pub use_cls_token: bool,
    pub cls_sync_alpha: f32,
    pub num_eyes: usize,
    pub cross_eye_steps: usize,
    pub token_state_norm: bool,
    pub latent_activation: VisionLatentActivation,
    pub pos_encoding: SpatialPositionalEncodingKind,
    pub pos_max_height: usize,
    pub pos_max_width: usize,
    pub attention_mode: VisionAttentionMode,
    pub use_alibi: bool,
    pub fused_kernels: FusedKernelConfig,
    pub mhc: ManifoldHyperConnectionsConfig,
    pub trm_graph: VisionTrmGraphConfig,
}

impl Default for VisionDragonConfig {
    fn default() -> Self {
        let image_size: usize = 224;
        let patch_size: usize = 16;
        let grid = image_size.div_ceil(patch_size).max(1);
        Self {
            image_size,
            patch_size,
            patch_embed_mode: VisionPatchEmbedMode::default(),
            in_channels: 3,
            embed_dim: 256,
            steps: 6,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            dropout: 0.1,
            projection_dim: 384,
            projection_hidden_dim: 512,
            use_cls_token: true,
            cls_sync_alpha: 0.0,
            num_eyes: 1,
            cross_eye_steps: 0,
            token_state_norm: true,
            latent_activation: VisionLatentActivation::default(),
            pos_encoding: SpatialPositionalEncodingKind::Learned2d,
            pos_max_height: grid,
            pos_max_width: grid,
            attention_mode: VisionAttentionMode::RowL1,
            use_alibi: true,
            fused_kernels: FusedKernelConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            trm_graph: VisionTrmGraphConfig::default(),
        }
    }
}

impl VisionDragonConfig {
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
            let trm_v = LinearConfig::new(config.embed_dim, trm_graph.value_dim.max(1)).init(device);
            let trm_y = LinearConfig::new(trm_graph.value_dim.max(1), trm_graph.rank.max(1)).init(device);
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
            [
                batch,
                rank,
                value_dim,
                h32_height.max(1),
                h32_width.max(1),
            ],
            &device,
        );
        let mut mem_hub =
            Tensor::<B, 4>::zeros([batch, hub_count, rank, value_dim], &device);

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

            mem_hub = self.trm_update_hub(
                mem_hub,
                u8.clone(),
                u32,
                hub_w8,
                hub_w32,
                hub_count,
                decay,
            );

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
            let cls = patch_tokens
                .clone()
                .mean_dim(1)
                .reshape([batch, 1, dim]);
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
                current = self
                    .sync_cls_tokens_multi(self.apply_token_norm(mixed.reshape([
                        batch, streams, time, self.embed_dim,
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
            return Tensor::<B, 4>::zeros([batch.max(1), value_dim.max(1), height.max(1), width.max(1)], &memory.device());
        }
        let query = query.unsqueeze_dim::<5>(2);
        memory.mul(query).sum_dims_squeeze::<4, usize>(&[1])
    }

    fn trm_local_read(&self, memory: Tensor<B, 5>, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, _, value_dim, height, width] = memory.shape().dims::<5>();
        if batch == 0 || value_dim == 0 || height == 0 || width == 0 {
            return Tensor::<B, 4>::zeros([batch.max(1), value_dim.max(1), height.max(1), width.max(1)], &memory.device());
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
                [batch, rank, value_dim, pooled_height.max(1), pooled_width.max(1)],
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
            let denom = weights
                .clone()
                .sum_dim(1)
                .add_scalar(ROW_NORM_EPS);
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
        if self.use_alibi && let Some(slopes) = self.alibi_slopes.as_ref() {
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

            Ok(Self { images, labels })
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
            let labels_tensor =
                Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [batch_size]), device);
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
            Some(
                self.dataset
                    .sample_batch::<B>(self.batch_size, &self.device),
            )
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
        let bytes =
            fs::read(path).map_err(|err| anyhow!("failed to read {}: {err}", path.display()))?;
        if bytes.len() % record_len != 0 {
            return Err(anyhow!(
                "invalid CIFAR record size in {}: {} bytes (record_len={})",
                path.display(),
                bytes.len(),
                record_len
            ));
        }
        if label_offset >= record_len {
            return Err(anyhow!("label offset out of range for {}", path.display()));
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
    use image::imageops::FilterType;
    use image::{DynamicImage, GenericImageView, RgbImage};
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

    #[derive(Clone, Copy, Debug)]
    struct CropParams {
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    }

    fn crop_overlap_ratio(a: CropParams, b: CropParams) -> f32 {
        let x1 = a.x.max(b.x);
        let y1 = a.y.max(b.y);
        let x2 = (a.x + a.width).min(b.x + b.width);
        let y2 = (a.y + a.height).min(b.y + b.height);
        let inter_w = x2.saturating_sub(x1);
        let inter_h = y2.saturating_sub(y1);
        let inter_area = (inter_w * inter_h) as f32;
        let area_a = (a.width * a.height) as f32;
        let area_b = (b.width * b.height) as f32;
        let denom = area_a.min(area_b).max(1.0);
        inter_area / denom
    }

    fn push_crop_normalized(buffer: &mut Vec<f32>, crop: CropParams, width: u32, height: u32) {
        let width = width.max(1) as f32;
        let height = height.max(1) as f32;
        buffer.extend_from_slice(&[
            crop.x as f32 / width,
            crop.y as f32 / height,
            crop.width as f32 / width,
            crop.height as f32 / height,
        ]);
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

        pub fn is_deterministic(&self) -> bool {
            match self.split {
                ImageNetSplit::Val => true,
                ImageNetSplit::Train => {
                    self.flip_prob <= 0.0
                        && self.color_jitter_prob <= 0.0
                        && self.grayscale_prob <= 0.0
                        && self.blur_prob <= 0.0
                        && self.solarize_prob <= 0.0
                        && (self.min_scale - 1.0).abs() <= f32::EPSILON
                        && (self.max_scale - 1.0).abs() <= f32::EPSILON
                        && (self.min_aspect_ratio - 1.0).abs() <= f32::EPSILON
                        && (self.max_aspect_ratio - 1.0).abs() <= f32::EPSILON
                }
            }
        }

        pub fn apply(&self, image: &DynamicImage, rng: &mut impl Rng) -> RgbImage {
            match self.split {
                ImageNetSplit::Train => self.apply_train(image, rng),
                ImageNetSplit::Val => self.apply_val(image),
            }
        }

        fn apply_train(&self, image: &DynamicImage, rng: &mut impl Rng) -> RgbImage {
            if self.is_deterministic() {
                return self.apply_val(image);
            }

            let crop = self.random_resized_crop_params(image, rng);
            self.apply_train_with_crop(image, rng, crop)
        }

        fn apply_train_with_crop(
            &self,
            image: &DynamicImage,
            rng: &mut impl Rng,
            crop: CropParams,
        ) -> RgbImage {
            let mut image = self.apply_crop(image, crop);

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

        fn apply_val(&self, image: &DynamicImage) -> RgbImage {
            let (width, height) = image.dimensions();
            let target = self.resize_short.max(1);

            let (new_width, new_height) = if width < height {
                (
                    target,
                    (height as f32 * (target as f32 / width as f32)).round() as u32,
                )
            } else {
                (
                    (width as f32 * (target as f32 / height as f32)).round() as u32,
                    target,
                )
            };

            let resized = image.resize_exact(new_width, new_height, FilterType::CatmullRom);
            let crop_x = (new_width.saturating_sub(self.image_size)) / 2;
            let crop_y = (new_height.saturating_sub(self.image_size)) / 2;
            let cropped = resized.crop_imm(crop_x, crop_y, self.image_size, self.image_size);
            cropped.to_rgb8()
        }

        fn val_crop_params(&self, image: &DynamicImage) -> CropParams {
            let (width, height) = image.dimensions();
            let target = self.resize_short.max(1);
            let (new_width, new_height) = if width < height {
                (
                    target,
                    (height as f32 * (target as f32 / width as f32)).round() as u32,
                )
            } else {
                (
                    (width as f32 * (target as f32 / height as f32)).round() as u32,
                    target,
                )
            };
            let scale = target as f32 / width.min(height).max(1) as f32;
            let crop_x = (new_width.saturating_sub(self.image_size)) / 2;
            let crop_y = (new_height.saturating_sub(self.image_size)) / 2;
            let crop_w = (self.image_size as f32 / scale).round().max(1.0);
            let crop_h = (self.image_size as f32 / scale).round().max(1.0);
            let x = (crop_x as f32 / scale).round().max(0.0);
            let y = (crop_y as f32 / scale).round().max(0.0);
            let max_w = width.max(1) as f32;
            let max_h = height.max(1) as f32;
            CropParams {
                x: x.min(max_w - 1.0).max(0.0) as u32,
                y: y.min(max_h - 1.0).max(0.0) as u32,
                width: crop_w.min(max_w).max(1.0) as u32,
                height: crop_h.min(max_h).max(1.0) as u32,
            }
        }

        fn apply_crop(&self, image: &DynamicImage, crop: CropParams) -> DynamicImage {
            let cropped = image.crop_imm(crop.x, crop.y, crop.width, crop.height);
            cropped.resize_exact(self.image_size, self.image_size, FilterType::CatmullRom)
        }

        fn random_resized_crop_params(
            &self,
            image: &DynamicImage,
            rng: &mut impl Rng,
        ) -> CropParams {
            let (width, height) = image.dimensions();
            let area = (width * height) as f32;
            let log_min = self.min_aspect_ratio.ln();
            let log_max = self.max_aspect_ratio.ln();

            for _ in 0..10 {
                let scale = if self.min_scale >= self.max_scale {
                    self.min_scale
                } else {
                    rng.gen_range(self.min_scale..self.max_scale)
                };
                let aspect = if self.min_aspect_ratio >= self.max_aspect_ratio {
                    self.min_aspect_ratio
                } else {
                    rng.gen_range(log_min..log_max).exp()
                };
                let target = scale * area;
                let new_width = (target * aspect).sqrt().round() as u32;
                let new_height = (target / aspect).sqrt().round() as u32;

                if new_width > 0 && new_height > 0 && new_width <= width && new_height <= height {
                    let x = rng.gen_range(0..=width - new_width);
                    let y = rng.gen_range(0..=height - new_height);
                    return CropParams {
                        x,
                        y,
                        width: new_width,
                        height: new_height,
                    };
                }
            }

            let side = width.min(height).max(1);
            let x = (width - side) / 2;
            let y = (height - side) / 2;
            CropParams {
                x,
                y,
                width: side,
                height: side,
            }
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
            let pixels = (width * height) as usize;
            if pixels == 0 {
                return;
            }
            let start = buffer.len();
            buffer.resize(start + pixels * IMAGE_CHANNELS, 0.0);
            let raw = image.as_raw();
            let stride = pixels;
            for idx in 0..pixels {
                let base = idx * IMAGE_CHANNELS;
                let r = raw[base] as f32 / 255.0;
                let g = raw[base + 1] as f32 / 255.0;
                let b = raw[base + 2] as f32 / 255.0;
                buffer[start + idx] = (r - self.mean[0]) / self.std[0];
                buffer[start + stride + idx] = (g - self.mean[1]) / self.std[1];
                buffer[start + 2 * stride + idx] = (b - self.mean[2]) / self.std[2];
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
            if let Some(expected) = expected_records.filter(|expected| *expected > cls_records) {
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
        pub min_view_overlap: f32,
        pub view_overlap_attempts: usize,
        pub cache_decoded: bool,
        pub cache_capacity: usize,
        pub cache_preprocessed: bool,
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
        const ORDER_GC_MULTIPLIER: usize = 4;

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

        fn prune_order(&self, state: &mut ImageCacheState) {
            let max_len = self
                .capacity
                .saturating_mul(Self::ORDER_GC_MULTIPLIER)
                .max(1);
            if state.order.len() <= max_len {
                return;
            }
            let mut pruned = VecDeque::with_capacity(state.entries.len());
            for (path, tick) in state.order.drain(..) {
                if let Some(entry) = state.entries.get(&path) && entry.tick == tick {
                    pruned.push_back((path, tick));
                }
            }
            state.order = pruned;
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
            self.prune_order(&mut state);
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
                    if let Some(entry) = state.entries.get(&evict_path)
                        && entry.tick == evict_tick
                    {
                        state.entries.remove(&evict_path);
                        break;
                    }
                }
            }
            self.prune_order(&mut state);
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
        min_view_overlap: f32,
        view_overlap_attempts: usize,
        cache: Option<ImageCache>,
        preprocessed_cache: Option<Vec<Arc<Vec<f32>>>>,
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
            let min_view_overlap = config.min_view_overlap.max(0.0);
            let view_overlap_attempts = config.view_overlap_attempts.max(1);
            if local_views > 0 && config.local_augmentations.is_none() {
                return Err(anyhow!("local_augmentations required when local_views > 0"));
            }
            let cache = if config.cache_decoded && config.cache_capacity > 0 {
                Some(ImageCache::new(config.cache_capacity))
            } else {
                None
            };

            let allow_preprocessed = config.cache_preprocessed
                && global_views == 1
                && local_views == 0
                && config.augmentations.is_deterministic()
                && config.cache_capacity >= samples.len();

            let mut dataset = Self {
                samples,
                num_classes,
                augmentations: config.augmentations,
                local_augmentations: config.local_augmentations,
                normalize: config.normalize,
                teacher: config.teacher,
                global_views,
                local_views,
                min_view_overlap,
                view_overlap_attempts,
                cache,
                preprocessed_cache: None,
            };

            if allow_preprocessed {
                dataset.preprocessed_cache = Some(dataset.build_preprocessed_cache()?);
            }

            Ok(dataset)
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

        fn load_image_cached(&self, path: &Path) -> Result<Arc<DynamicImage>> {
            if let Some(cache) = &self.cache && let Some(image) = cache.get(path) {
                return Ok(image);
            }

            let image = Arc::new(load_image(path)?);
            if let Some(cache) = &self.cache {
                cache.insert(path.to_path_buf(), Arc::clone(&image));
            }
            Ok(image)
        }

        fn build_preprocessed_cache(&self) -> Result<Vec<Arc<Vec<f32>>>> {
            let mut cache = Vec::with_capacity(self.samples.len());
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            let image_size = self.augmentations.image_size();
            let buffer_len = image_size * image_size * IMAGE_CHANNELS;
            for sample in &self.samples {
                let image = self.load_image_cached(&sample.path)?;
                let aug = self.augmentations.apply(image.as_ref(), &mut rng);
                let mut buffer = Vec::with_capacity(buffer_len);
                self.normalize.apply(&aug, &mut buffer);
                cache.push(Arc::new(buffer));
            }
            Ok(cache)
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
            let mut images = Vec::with_capacity(
                batch_size * global_image_size * global_image_size * IMAGE_CHANNELS,
            );
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
            let mut view_crops = if self.local_views == 0 && self.global_views > 1 {
                Some(Vec::with_capacity(batch_size * self.global_views * 4))
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
                if self.global_views == 1
                    && self.local_views == 0
                    && let Some(preprocessed_cache) = &self.preprocessed_cache
                    && let Some(cached) = preprocessed_cache.get(index)
                {
                    images.extend_from_slice(cached.as_ref());
                    labels.push(sample.label as i64);
                    continue;
                }
                let image = self.load_image_cached(&sample.path)?;
                let (img_w, img_h) = image.dimensions();
                if self.global_views == 1 && self.local_views == 0 {
                    let primary = self.augmentations.apply(image.as_ref(), &mut rng);
                    self.normalize.apply(&primary, &mut images);
                } else {
                    if self.min_view_overlap > 0.0
                        && matches!(self.augmentations.split, ImageNetSplit::Train)
                    {
                        let primary_crop = self
                            .augmentations
                            .random_resized_crop_params(image.as_ref(), &mut rng);
                        let mut crops = Vec::with_capacity(self.global_views);
                        crops.push(primary_crop);
                        for _ in 1..self.global_views {
                            let mut selected = None;
                            for _ in 0..self.view_overlap_attempts {
                                let candidate = self
                                    .augmentations
                                    .random_resized_crop_params(image.as_ref(), &mut rng);
                                if crop_overlap_ratio(primary_crop, candidate)
                                    >= self.min_view_overlap
                                {
                                    selected = Some(candidate);
                                    break;
                                }
                            }
                            crops.push(selected.unwrap_or(primary_crop));
                        }

                        for (view_idx, crop) in crops.into_iter().enumerate() {
                            if let Some(buffer) = view_crops.as_mut() {
                                push_crop_normalized(buffer, crop, img_w, img_h);
                            }
                            let aug = self.augmentations.apply_train_with_crop(
                                image.as_ref(),
                                &mut rng,
                                crop,
                            );
                            if view_idx == 0 {
                                self.normalize.apply(&aug, &mut images);
                            }
                            if view_idx == 1 && let Some(buffer) = target_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                            if let Some(buffer) = view_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                            if let Some(buffer) = global_view_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                        }
                    } else {
                        for view_idx in 0..self.global_views {
                            let (aug, crop) = if matches!(self.augmentations.split, ImageNetSplit::Train)
                                && !self.augmentations.is_deterministic()
                            {
                                let crop = self
                                    .augmentations
                                    .random_resized_crop_params(image.as_ref(), &mut rng);
                                let aug = self
                                    .augmentations
                                    .apply_train_with_crop(image.as_ref(), &mut rng, crop);
                                (aug, crop)
                            } else {
                                let crop = self.augmentations.val_crop_params(image.as_ref());
                                let aug = self.augmentations.apply_val(image.as_ref());
                                (aug, crop)
                            };
                            if view_idx == 0 {
                                self.normalize.apply(&aug, &mut images);
                            }
                            if view_idx == 1 && let Some(buffer) = target_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                            if let Some(buffer) = view_crops.as_mut() {
                                push_crop_normalized(buffer, crop, img_w, img_h);
                            }
                            if let Some(buffer) = view_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                            if let Some(buffer) = global_view_images.as_mut() {
                                self.normalize.apply(&aug, buffer);
                            }
                        }
                    }

                    if self.local_views > 0 {
                        let local_aug = self
                            .local_augmentations
                            .as_ref()
                            .expect("local augmentations required");
                        for _ in 0..self.local_views {
                            let aug = local_aug.apply(image.as_ref(), &mut rng);
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
                view_crops,
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
        view_crops: Option<Vec<f32>>,
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
            let view_crops_tensor = self.view_crops.map(|buffer| {
                Tensor::<B, 3>::from_data(
                    TensorData::new(buffer, [self.batch_size, self.global_views, 4]),
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
                Tensor::<B, 2>::from_data(TensorData::new(data, [self.batch_size, dim]), device)
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
                view_crops_tensor,
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
        pub view_crops: Option<Tensor<B, 3>>,
        pub global_view_images: Option<Tensor<B, 5>>,
        pub local_view_images: Option<Tensor<B, 5>>,
        pub labels: Tensor<B, 1, Int>,
        pub teacher_patch: Option<Tensor<B, 3>>,
        pub teacher_cls: Option<Tensor<B, 2>>,
    }

    impl<B: Backend> ImageNetBatch<B> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            images: Tensor<B, 4>,
            target_images: Option<Tensor<B, 4>>,
            view_images: Option<Tensor<B, 5>>,
            view_crops: Option<Tensor<B, 3>>,
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
                view_crops,
                global_view_images,
                local_view_images,
                labels,
                teacher_patch,
                teacher_cls,
            }
        }

        pub fn repeat_batch(&self, repeats: usize) -> Self {
            let repeats = repeats.max(1);
            if repeats == 1 {
                return self.clone();
            }
            Self {
                images: self.images.clone().repeat_dim(0, repeats),
                target_images: self
                    .target_images
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                view_images: self
                    .view_images
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                view_crops: self
                    .view_crops
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                global_view_images: self
                    .global_view_images
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                local_view_images: self
                    .local_view_images
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                labels: self.labels.clone().repeat_dim(0, repeats),
                teacher_patch: self
                    .teacher_patch
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
                teacher_cls: self
                    .teacher_cls
                    .as_ref()
                    .map(|tensor| tensor.clone().repeat_dim(0, repeats)),
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
        prefetch_to_device: bool,
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
                prefetch_to_device: self.prefetch_to_device,
            }
        }
    }

    impl<B: Backend> ImageNetDataLoader<B> {
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            dataset: Arc<ImageNetDataset>,
            batch_size: usize,
            device: &B::Device,
            steps_per_epoch: usize,
            total_steps: Option<usize>,
            prefetch_batches: usize,
            prefetch_workers: usize,
            prefetch_to_device: bool,
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
            let prefetch_to_device = prefetch_to_device && prefetch_batches > 0;
            let min_prefetch = if prefetch_to_device { 2 } else { 1 };
            let prefetch_workers = if prefetch_batches == 0 {
                0
            } else {
                prefetch_workers.max(1)
            };
            let prefetch_batches = if prefetch_batches == 0 {
                0
            } else {
                prefetch_batches
                    .max(min_prefetch)
                    .max(prefetch_workers)
                    .min(steps_per_epoch)
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
                prefetch_to_device,
            }
        }
    }

    impl<B> DataLoader<B, ImageNetBatch<B>> for ImageNetDataLoader<B>
    where
        B: Backend + 'static,
        B::Device: Clone + Send + Sync + 'static,
        ImageNetBatch<B>: Send,
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
                    &self.device,
                    self.prefetch_to_device,
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
                prefetch_to_device: self.prefetch_to_device,
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
                prefetch_to_device: self.prefetch_to_device,
            })
        }
    }

    enum ImageNetPrefetchItem<B: Backend> {
        Data(Box<ImageNetBatchData>),
        Batch(ImageNetBatch<B>),
    }

    struct ImageNetPrefetcher<B: Backend> {
        rx: Option<mpsc::Receiver<Result<ImageNetPrefetchItem<B>>>>,
        stop: Arc<AtomicBool>,
        handles: Vec<thread::JoinHandle<()>>,
    }

    impl<B> ImageNetPrefetcher<B>
    where
        B: Backend + 'static,
        B::Device: Clone + Send + Sync + 'static,
        ImageNetBatch<B>: Send,
    {
        fn new(
            dataset: Arc<ImageNetDataset>,
            batch_size: usize,
            steps_total: usize,
            prefetch_batches: usize,
            prefetch_workers: usize,
            device: &B::Device,
            prefetch_to_device: bool,
        ) -> Self {
            let workers = prefetch_workers.max(1);
            let (tx, rx) = mpsc::sync_channel(prefetch_batches.max(1));
            let remaining = Arc::new(AtomicUsize::new(steps_total));
            let stop = Arc::new(AtomicBool::new(false));
            let mut handles = Vec::with_capacity(workers + if prefetch_to_device { 1 } else { 0 });

            if prefetch_to_device {
                let (data_tx, data_rx) = mpsc::sync_channel(prefetch_batches.max(1));
                for _ in 0..workers {
                    let dataset = Arc::clone(&dataset);
                    let tx = data_tx.clone();
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
                drop(data_tx);

                let device = device.clone();
                let stop = Arc::clone(&stop);
                let handle = thread::spawn(move || {
                    for data in data_rx {
                        if stop.load(Ordering::Relaxed) {
                            break;
                        }
                        let item = match data {
                            Ok(data) => {
                                let _guard = crate::device::device_allocation_lock().lock().ok();
                                Ok(ImageNetPrefetchItem::Batch(data.into_batch::<B>(&device)))
                            }
                            Err(err) => Err(err),
                        };
                        if tx.send(item).is_err() {
                            break;
                        }
                    }
                });
                handles.push(handle);
            } else {
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
                            let result = dataset
                                .sample_batch_data(batch_size)
                                .map(|data| ImageNetPrefetchItem::Data(Box::new(data)));
                            if tx.send(result).is_err() {
                                break;
                            }
                        }
                    });
                    handles.push(handle);
                }
                drop(tx);
            }

            Self {
                rx: Some(rx),
                stop,
                handles,
            }
        }

        fn recv(&mut self) -> Option<Result<ImageNetPrefetchItem<B>>> {
            let rx = self.rx.as_ref()?;
            rx.recv().ok()
        }
    }

    impl<B: Backend> Drop for ImageNetPrefetcher<B> {
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
        prefetcher: Option<ImageNetPrefetcher<B>>,
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
            let batch = if let Some(prefetcher) = &mut self.prefetcher {
                match prefetcher.recv() {
                    Some(Ok(ImageNetPrefetchItem::Batch(batch))) => batch,
                    Some(Ok(ImageNetPrefetchItem::Data(data))) => {
                        (*data).into_batch(&self.device)
                    }
                    Some(Err(err)) => panic!("imagenet prefetch error: {err}"),
                    None => panic!("imagenet prefetch channel closed early"),
                }
            } else {
                self.dataset.sample_batch(self.batch_size, &self.device)
            };

            Some(batch)
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
            return Err(anyhow!("no class directories found in {}", root.display()));
        }

        let mut samples = Vec::new();
        for (label, class_dir) in class_dirs.iter().enumerate() {
            let mut images = collect_images(class_dir)?;
            images.sort();
            for image in images {
                samples.push(ImageNetSample { path: image, label });
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
            Some(ext) => matches!(ext.to_ascii_lowercase().as_str(), "jpg" | "jpeg" | "png"),
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn image_cache_prunes_order_growth() {
            let cache = ImageCache::new(2);
            let image = Arc::new(DynamicImage::ImageRgb8(RgbImage::new(1, 1)));
            let path_a = PathBuf::from("a.png");
            let path_b = PathBuf::from("b.png");
            cache.insert(path_a.clone(), Arc::clone(&image));
            cache.insert(path_b.clone(), Arc::clone(&image));

            for _ in 0..128 {
                let _ = cache.get(&path_a);
            }

            let order_len = cache.inner.lock().unwrap().order.len();
            let max_len = cache.capacity * ImageCache::ORDER_GC_MULTIPLIER;
            assert!(
                order_len <= max_len,
                "order len {order_len} exceeds {max_len}"
            );

            let path_c = PathBuf::from("c.png");
            cache.insert(path_c, image);
            let entries = cache.inner.lock().unwrap().entries.len();
            assert!(entries <= cache.capacity);
        }
    }
}

#[cfg(feature = "train")]
pub use cifar::{CifarBatch, CifarDataLoader, CifarDataset, CifarSplit, CifarType};
#[cfg(feature = "train")]
pub use imagenet::{
    DinoFeatureStore, ImageNetAugmentations, ImageNetBatch, ImageNetDataLoader, ImageNetDataset,
    ImageNetDatasetConfig, ImageNetSplit, VisionNormalize,
};

