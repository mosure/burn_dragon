use crate::train::prelude::*;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::PaddingConfig2d;

#[derive(Module, Debug)]
pub(crate) struct VisionDistillModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) loss: VisionDistillationLossConfig,
    pub(crate) teacher: Option<DinoVisionTransformer<B>>,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

impl<B: BackendTrait> VisionDistillModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
        loss: VisionDistillationLossConfig,
        teacher: Option<DinoVisionTransformer<B>>,
        rollout: VisionRollout,
    ) -> Self {
        Self {
            model,
            loss,
            teacher,
            rollout,
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionProbe<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) head: Linear<B>,
}

impl<B: BackendTrait> VisionProbe<B> {
    pub(crate) fn new(embed_dim: usize, num_classes: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let head = LinearConfig::new(embed_dim, num_classes.max(1)).init(device);
        Self { norm, head }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 2>) -> Tensor<B, 2> {
        let tokens = self.norm.forward(tokens);
        self.head.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionReconstructionHead<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) hidden: Option<Linear<B>>,
    pub(crate) out: Linear<B>,
}

impl<B: BackendTrait> VisionReconstructionHead<B> {
    pub(crate) fn new(embed_dim: usize, hidden_dim: usize, patch_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(embed_dim, hidden_dim).init(device))
        } else {
            None
        };
        let out_dim = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            embed_dim.max(1)
        };
        let out = LinearConfig::new(out_dim, patch_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    pub(crate) fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeHead<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeHead<B> {
    pub(crate) fn new(embed_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let proj = LinearConfig::new(embed_dim, 3).init(device);
        Self { norm, proj }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeProjection<B: BackendTrait> {
    pub(crate) norm: LayerNorm<B>,
    pub(crate) proj: Linear<B>,
}

impl<B: BackendTrait> VisionSaccadeProjection<B> {
    pub(crate) fn new(embed_dim: usize, out_dim: usize, device: &B::Device) -> Self {
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let proj = LinearConfig::new(embed_dim, out_dim).init(device);
        Self { norm, proj }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let tokens = self.norm.forward(tokens);
        self.proj.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionSaccadeInputProjection<B: BackendTrait> {
    linear: Option<VisionSaccadeProjection<B>>,
    cnn: Option<VisionInputProjectionCnn<B>>,
    micro_vit: Option<VisionInputProjectionMicroVit<B>>,
    #[module(ignore)]
    param_count: usize,
}

impl<B: BackendTrait> VisionSaccadeInputProjection<B> {
    pub(crate) fn new(
        embed_dim: usize,
        patch_size: usize,
        config: &VisionSaccadeInputProjectionConfig,
        device: &B::Device,
    ) -> Self {
        match config {
            VisionSaccadeInputProjectionConfig::Linear => {
                let linear = VisionSaccadeProjection::new(embed_dim, embed_dim, device);
                let param_count = linear_params(embed_dim, embed_dim) + layer_norm_params(embed_dim);
                Self {
                    linear: Some(linear),
                    cnn: None,
                    micro_vit: None,
                    param_count,
                }
            }
            VisionSaccadeInputProjectionConfig::Cnn(cfg) => {
                let cnn = VisionInputProjectionCnn::new(embed_dim, patch_size, cfg, device);
                let param_count = cnn.param_count();
                Self {
                    linear: None,
                    cnn: Some(cnn),
                    micro_vit: None,
                    param_count,
                }
            }
            VisionSaccadeInputProjectionConfig::RadialMicroVit(cfg) => {
                let micro_vit =
                    VisionInputProjectionMicroVit::new(embed_dim, patch_size, cfg, device);
                let param_count = micro_vit.param_count();
                Self {
                    linear: None,
                    cnn: None,
                    micro_vit: Some(micro_vit),
                    param_count,
                }
            }
        }
    }

    pub(crate) fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        if let Some(linear) = &self.linear {
            return linear.forward(tokens);
        }
        if let Some(cnn) = &self.cnn {
            return cnn.forward(tokens);
        }
        self.micro_vit
            .as_ref()
            .expect("micro vit projection")
            .forward(tokens)
    }

    #[cfg(any(test, feature = "benchmark"))]
    pub(crate) fn param_count(&self) -> usize {
        self.param_count
    }
}

#[derive(Module, Debug)]
struct VisionInputProjectionCnn<B: BackendTrait> {
    norm: LayerNorm<B>,
    in_proj: Linear<B>,
    blocks: Vec<VisionInputProjectionCnnBlock<B>>,
    out_proj: Linear<B>,
    #[module(ignore)]
    embed_dim: usize,
    #[module(ignore)]
    channels: usize,
    #[module(ignore)]
    kernel: usize,
    #[module(ignore)]
    expansion: usize,
    #[module(ignore)]
    param_count: usize,
}

impl<B: BackendTrait> VisionInputProjectionCnn<B> {
    fn new(
        embed_dim: usize,
        patch_size: usize,
        config: &VisionSaccadeInputProjectionCnnConfig,
        device: &B::Device,
    ) -> Self {
        let channels = config
            .channels
            .filter(|&value| value > 0)
            .unwrap_or(embed_dim.max(1));
        let kernel = resolve_cnn_kernel(patch_size, config.kernel);
        let expansion = config.expansion.max(1);
        let blocks = resolve_cnn_blocks(patch_size, config.blocks);
        let norm = LayerNormConfig::new(embed_dim).init(device);
        let in_proj = LinearConfig::new(embed_dim, channels).init(device);
        let out_proj = LinearConfig::new(channels, embed_dim).init(device);
        let mut block_list = Vec::with_capacity(blocks);
        for _ in 0..blocks {
            block_list.push(VisionInputProjectionCnnBlock::new(
                channels,
                kernel,
                expansion,
                device,
            ));
        }
        let mut param_count = layer_norm_params(embed_dim)
            + linear_params(embed_dim, channels)
            + linear_params(channels, embed_dim);
        param_count = param_count.saturating_add(
            cnn_block_params(channels, kernel, expansion).saturating_mul(blocks),
        );
        Self {
            norm,
            in_proj,
            blocks: block_list,
            out_proj,
            embed_dim,
            channels,
            kernel,
            expansion,
            param_count,
        }
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, token_count, _] = tokens.shape().dims::<3>();
        if batch == 0 || token_count == 0 {
            return tokens;
        }
        let (grid_h, grid_w) = token_grid(token_count);
        let tokens = self.norm.forward(tokens);
        let tokens = self.in_proj.forward(tokens);
        let mut image = tokens_to_image(tokens, grid_h, grid_w);
        for block in &self.blocks {
            let update = block.forward(image.clone());
            image = image + update;
        }
        let tokens = image_to_tokens(image);
        self.out_proj.forward(tokens)
    }

    fn param_count(&self) -> usize {
        self.param_count
    }
}

#[derive(Module, Debug)]
struct VisionInputProjectionCnnBlock<B: BackendTrait> {
    depthwise: Conv2d<B>,
    pointwise_in: Conv2d<B>,
    pointwise_out: Conv2d<B>,
}

impl<B: BackendTrait> VisionInputProjectionCnnBlock<B> {
    fn new(channels: usize, kernel: usize, expansion: usize, device: &B::Device) -> Self {
        let depthwise = Conv2dConfig::new([channels, channels], [kernel, kernel])
            .with_padding(PaddingConfig2d::Same)
            .with_groups(channels.max(1))
            .init(device);
        let pointwise_in = Conv2dConfig::new(
            [channels, channels.saturating_mul(expansion)],
            [1, 1],
        )
        .init(device);
        let pointwise_out = Conv2dConfig::new(
            [channels.saturating_mul(expansion), channels],
            [1, 1],
        )
        .init(device);
        Self {
            depthwise,
            pointwise_in,
            pointwise_out,
        }
    }

    fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.depthwise.forward(input);
        let x = activation::gelu(x);
        let x = self.pointwise_in.forward(x);
        let x = activation::gelu(x);
        self.pointwise_out.forward(x)
    }
}

#[derive(Module, Debug)]
struct VisionInputProjectionMicroVit<B: BackendTrait> {
    radial: VisionInputProjectionRadial<B>,
    blocks: Vec<VisionInputProjectionMicroVitBlock<B>>,
    #[module(ignore)]
    embed_dim: usize,
    #[module(ignore)]
    heads: usize,
    #[module(ignore)]
    mlp_ratio: usize,
    #[module(ignore)]
    radial_hidden_dim: usize,
}

impl<B: BackendTrait> VisionInputProjectionMicroVit<B> {
    fn new(
        embed_dim: usize,
        patch_size: usize,
        config: &VisionSaccadeInputProjectionMicroVitConfig,
        device: &B::Device,
    ) -> Self {
        let layers = resolve_micro_vit_layers(patch_size, config.layers);
        let heads = resolve_micro_vit_heads(embed_dim, patch_size, config.heads);
        let mlp_ratio = config.mlp_ratio.max(1);
        let radial_hidden_dim = resolve_radial_hidden_dim(embed_dim, config.radial_hidden_dim);
        let radial = VisionInputProjectionRadial::new(
            embed_dim,
            radial_hidden_dim,
            config.radial_scale,
            device,
        );
        let mut blocks = Vec::with_capacity(layers);
        for _ in 0..layers {
            blocks.push(VisionInputProjectionMicroVitBlock::new(
                embed_dim,
                heads,
                mlp_ratio,
                device,
            ));
        }
        Self {
            radial,
            blocks,
            embed_dim,
            heads,
            mlp_ratio,
            radial_hidden_dim,
        }
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, token_count, _] = tokens.shape().dims::<3>();
        if batch == 0 || token_count == 0 {
            return tokens;
        }
        let (grid_h, grid_w) = token_grid(token_count);
        let device = tokens.device();
        let mut x = tokens + self.radial.forward(batch, token_count, grid_h, grid_w, &device);
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }

    fn param_count(&self) -> usize {
        let mut count = radial_params(self.embed_dim, self.radial_hidden_dim);
        let block_params = micro_vit_block_params(self.embed_dim, self.mlp_ratio);
        count = count.saturating_add(block_params.saturating_mul(self.blocks.len()));
        count
    }
}

#[derive(Module, Debug)]
struct VisionInputProjectionRadial<B: BackendTrait> {
    in_proj: Linear<B>,
    out_proj: Linear<B>,
    #[module(ignore)]
    scale: f32,
}

impl<B: BackendTrait> VisionInputProjectionRadial<B> {
    fn new(embed_dim: usize, hidden_dim: usize, scale: f32, device: &B::Device) -> Self {
        let in_proj = LinearConfig::new(1, hidden_dim).init(device);
        let out_proj = LinearConfig::new(hidden_dim, embed_dim).init(device);
        Self {
            in_proj,
            out_proj,
            scale,
        }
    }

    fn forward(
        &self,
        batch: usize,
        tokens: usize,
        grid_h: usize,
        grid_w: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        let positions = radial_positions(grid_h, grid_w);
        let r = Tensor::<B, 1>::from_data(TensorData::new(positions, [tokens]), &device)
            .reshape([1, tokens, 1])
            .repeat_dim(0, batch)
            .mul_scalar(self.scale);
        let x = self.in_proj.forward(r);
        let x = activation::gelu(x);
        self.out_proj.forward(x)
    }
}

#[derive(Module, Debug)]
struct VisionInputProjectionMicroVitBlock<B: BackendTrait> {
    norm_attn: LayerNorm<B>,
    qkv: Linear<B>,
    proj: Linear<B>,
    norm_mlp: LayerNorm<B>,
    mlp_in: Linear<B>,
    mlp_out: Linear<B>,
    #[module(ignore)]
    heads: usize,
    #[module(ignore)]
    head_dim: usize,
}

impl<B: BackendTrait> VisionInputProjectionMicroVitBlock<B> {
    fn new(embed_dim: usize, heads: usize, mlp_ratio: usize, device: &B::Device) -> Self {
        let heads = heads.max(1);
        let head_dim = (embed_dim / heads).max(1);
        let norm_attn = LayerNormConfig::new(embed_dim).init(device);
        let qkv = LinearConfig::new(embed_dim, embed_dim * 3).init(device);
        let proj = LinearConfig::new(embed_dim, embed_dim).init(device);
        let norm_mlp = LayerNormConfig::new(embed_dim).init(device);
        let mlp_dim = embed_dim.saturating_mul(mlp_ratio).max(1);
        let mlp_in = LinearConfig::new(embed_dim, mlp_dim).init(device);
        let mlp_out = LinearConfig::new(mlp_dim, embed_dim).init(device);
        Self {
            norm_attn,
            qkv,
            proj,
            norm_mlp,
            mlp_in,
            mlp_out,
            heads,
            head_dim,
        }
    }

    fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = tokens.clone();
        let attn_in = self.norm_attn.forward(tokens);
        let attn = self.attention(attn_in);
        let tokens = residual + attn;
        let residual = tokens.clone();
        let mlp_in = self.norm_mlp.forward(tokens);
        let mlp = activation::gelu(self.mlp_in.forward(mlp_in));
        let mlp = self.mlp_out.forward(mlp);
        residual + mlp
    }

    fn attention(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        if batch == 0 || time == 0 || dim == 0 {
            return tokens;
        }
        let qkv = self.qkv.forward(tokens);
        let q = qkv.clone().slice_dim(2, 0..dim);
        let k = qkv.clone().slice_dim(2, dim..(2 * dim));
        let v = qkv.slice_dim(2, (2 * dim)..(3 * dim));
        let heads = self.heads.max(1);
        let head_dim = self.head_dim.max(1);
        let q = split_heads(q, heads, head_dim);
        let k = split_heads(k, heads, head_dim);
        let v = split_heads(v, heads, head_dim);
        let scale = (head_dim as f32).sqrt().max(1.0);
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn = activation::softmax(scores, 3);
        let out = attn.matmul(v);
        let out = merge_heads(out);
        self.proj.forward(out)
    }
}

fn token_grid(tokens: usize) -> (usize, usize) {
    let tokens = tokens.max(1);
    let grid = (tokens as f64).sqrt().round() as usize;
    if grid * grid == tokens {
        (grid.max(1), grid.max(1))
    } else {
        (1, tokens.max(1))
    }
}

fn tokens_to_image<B: BackendTrait>(
    tokens: Tensor<B, 3>,
    grid_h: usize,
    grid_w: usize,
) -> Tensor<B, 4> {
    let [batch, _tokens, channels] = tokens.shape().dims::<3>();
    let grid_h = grid_h.max(1);
    let grid_w = grid_w.max(1);
    let reshaped = tokens.reshape([batch, grid_h, grid_w, channels]);
    let reshaped = reshaped.swap_dims(1, 3).swap_dims(2, 3);
    reshaped
}

fn image_to_tokens<B: BackendTrait>(image: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, channels, grid_h, grid_w] = image.shape().dims::<4>();
    let reshaped = image.swap_dims(1, 3).swap_dims(1, 2);
    reshaped.reshape([batch, grid_h * grid_w, channels])
}

fn split_heads<B: BackendTrait>(
    tokens: Tensor<B, 3>,
    heads: usize,
    head_dim: usize,
) -> Tensor<B, 4> {
    let [batch, time, _] = tokens.shape().dims::<3>();
    tokens
        .reshape([batch, time, heads, head_dim])
        .swap_dims(1, 2)
}

fn merge_heads<B: BackendTrait>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, time, head_dim] = tokens.shape().dims::<4>();
    tokens
        .swap_dims(1, 2)
        .reshape([batch, time, heads * head_dim])
}

fn radial_positions(grid_h: usize, grid_w: usize) -> Vec<f32> {
    let grid_h = grid_h.max(1);
    let grid_w = grid_w.max(1);
    let cx = (grid_w.saturating_sub(1) as f32) * 0.5;
    let cy = (grid_h.saturating_sub(1) as f32) * 0.5;
    let max_r = (cx * cx + cy * cy).sqrt().max(1.0);
    let mut out = Vec::with_capacity(grid_h * grid_w);
    for y in 0..grid_h {
        let dy = y as f32 - cy;
        for x in 0..grid_w {
            let dx = x as f32 - cx;
            let r = (dx * dx + dy * dy).sqrt() / max_r;
            out.push(r);
        }
    }
    out
}

fn resolve_cnn_blocks(patch_size: usize, blocks: usize) -> usize {
    if blocks > 0 {
        return blocks;
    }
    let scale = (patch_size / 16).max(1);
    1 + scale.ilog2() as usize
}

fn resolve_cnn_kernel(patch_size: usize, kernel: usize) -> usize {
    if kernel > 0 {
        return kernel;
    }
    let scale = (patch_size / 16).max(1);
    let extra = scale.ilog2() as usize;
    let value = 3 + 2 * extra;
    value.min(7).max(3)
}

fn resolve_micro_vit_layers(patch_size: usize, layers: usize) -> usize {
    if layers > 0 {
        return layers;
    }
    let scale = (patch_size / 16).max(1);
    1 + scale.ilog2() as usize
}

fn resolve_micro_vit_heads(embed_dim: usize, patch_size: usize, heads: usize) -> usize {
    let mut heads = if heads > 0 {
        heads
    } else {
        let scale = (patch_size / 16).max(1);
        (2 * scale).min(8)
    };
    heads = heads.max(1).min(embed_dim.max(1));
    while heads > 1 && embed_dim % heads != 0 {
        heads -= 1;
    }
    heads.max(1)
}

fn resolve_radial_hidden_dim(embed_dim: usize, hidden_dim: usize) -> usize {
    if hidden_dim > 0 {
        hidden_dim
    } else {
        (embed_dim / 2).max(8)
    }
}

fn linear_params(in_dim: usize, out_dim: usize) -> usize {
    in_dim.saturating_mul(out_dim).saturating_add(out_dim)
}

fn layer_norm_params(dim: usize) -> usize {
    dim.saturating_mul(2)
}

fn conv_params(in_ch: usize, out_ch: usize, kernel: usize, groups: usize) -> usize {
    let groups = groups.max(1);
    let per_group = in_ch / groups.max(1);
    out_ch
        .saturating_mul(per_group)
        .saturating_mul(kernel.saturating_mul(kernel))
        .saturating_add(out_ch)
}

fn cnn_block_params(channels: usize, kernel: usize, expansion: usize) -> usize {
    let depthwise = conv_params(channels, channels, kernel, channels);
    let expanded = channels.saturating_mul(expansion);
    let point_in = conv_params(channels, expanded, 1, 1);
    let point_out = conv_params(expanded, channels, 1, 1);
    depthwise + point_in + point_out
}

fn micro_vit_block_params(embed_dim: usize, mlp_ratio: usize) -> usize {
    let mlp_dim = embed_dim.saturating_mul(mlp_ratio).max(1);
    let ln = layer_norm_params(embed_dim) * 2;
    let qkv = linear_params(embed_dim, embed_dim * 3);
    let proj = linear_params(embed_dim, embed_dim);
    let mlp = linear_params(embed_dim, mlp_dim) + linear_params(mlp_dim, embed_dim);
    ln + qkv + proj + mlp
}

fn radial_params(embed_dim: usize, hidden_dim: usize) -> usize {
    linear_params(1, hidden_dim) + linear_params(hidden_dim, embed_dim)
}

#[derive(Module, Debug)]
pub(crate) struct VisionLejepaModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) probe: VisionProbe<B>,
    pub(crate) probe_loss: burn::nn::loss::CrossEntropyLoss<B>,
    pub(crate) recon: Option<VisionReconstructionHead<B>>,
    pub(crate) mask_token: Option<Param<Tensor<B, 2>>>,
    pub(crate) config: VisionLejepaConfig,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

pub(crate) struct VisionLejepaLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) inv: Tensor<B, 1>,
    pub(crate) sigreg: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) recon_psnr: Tensor<B, 1>,
    pub(crate) probe_loss: Tensor<B, 1>,
    pub(crate) probe_acc: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

pub(crate) struct ViewGroupOutput<B: BackendTrait> {
    pub(crate) proj: Tensor<B, 3>,
    pub(crate) embed: Tensor<B, 3>,
    pub(crate) patch_tokens: Tensor<B, 3>,
}

impl<B: BackendTrait> VisionLejepaModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
        config: VisionLejepaConfig,
        embed_dim: usize,
        num_classes: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let probe = VisionProbe::new(embed_dim, num_classes, device);
        let probe_loss = CrossEntropyLossConfig::new().init(device);
        let recon_weight = config.loss.recon.weight;
        let recon = if recon_weight > 0.0 {
            if recon_patch_dim == 0 {
                None
            } else {
                Some(VisionReconstructionHead::new(
                    embed_dim,
                    config.loss.recon.hidden_dim,
                    recon_patch_dim,
                    device,
                ))
            }
        } else {
            None
        };
        let mask_token = recon.as_ref().map(|_| {
            let token = Tensor::<B, 2>::random(
                [1, embed_dim.max(1)],
                TensorDistribution::Normal(0.0, 0.02),
                device,
            );
            Param::from_tensor(token)
        });
        Self {
            model,
            probe,
            probe_loss,
            recon,
            mask_token,
            config,
            rollout,
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
    ) -> VisionLejepaLosses<B> {
        let ImageNetBatch {
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
            labels,
            ..
        } = batch;

        let device = labels.device();
        let collected = collect_views(
            images,
            target_images,
            view_images,
            global_view_images,
            local_view_images,
        );
        let mut proj_groups = Vec::new();
        let mut embed_groups = Vec::new();
        let mut heatmap_source = None;
        let mut artifact_views = None;
        let mut probe_primary = None;
        let mut probe_embed = None;
        let mut recon_loss_sum = Tensor::<B, 1>::zeros([1], &device);
        let mut recon_mask_sum = Tensor::<B, 1>::zeros([1], &device);
        let recon_enabled = self.recon.is_some();

        if !collected.global.is_empty() {
            let output = self.forward_view_group(&collected.global, steps, backprop_steps);
            probe_embed = Some(output.embed.clone());
            let [view_count, batch, dim] = output.embed.shape().dims::<3>();
            if view_count > 0 {
                probe_primary = Some(
                    output
                        .embed
                        .clone()
                        .slice_dim(0, 0..1)
                        .reshape([batch, dim]),
                );
                if heatmap_source.is_none() {
                    heatmap_source =
                        Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
                }
            }

            if recon_enabled {
                let (loss_sum, mask_sum, artifacts) =
                    self.recon_group_loss(
                        &collected.global,
                        steps,
                        backprop_steps,
                        true,
                        randomize_mask,
                    );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
                if let Some((views, residual)) = artifacts {
                    artifact_views = Some(views);
                    heatmap_source = Some(residual);
                }
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if !collected.local.is_empty() {
            let output = self.forward_view_group(&collected.local, steps, backprop_steps);
            if heatmap_source.is_none() {
                let [_, batch, _] = output.embed.shape().dims::<3>();
                heatmap_source =
                    Some(output.patch_tokens.clone().slice_dim(0, 0..batch));
            }
            if recon_enabled {
                let (loss_sum, mask_sum, _) = self.recon_group_loss(
                    &collected.local,
                    steps,
                    backprop_steps,
                    false,
                    randomize_mask,
                );
                recon_loss_sum = recon_loss_sum + loss_sum;
                recon_mask_sum = recon_mask_sum + mask_sum;
            }
            proj_groups.push(output.proj);
            embed_groups.push(output.embed);
        }
        if proj_groups.is_empty() {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return VisionLejepaLosses {
                total: zero.clone(),
                inv: zero.clone(),
                sigreg: zero.clone(),
                recon: zero.clone(),
                recon_psnr: zero.clone(),
                probe_loss: zero.clone(),
                probe_acc: zero,
                artifacts: None,
            };
        }

        let proj = if proj_groups.len() == 1 {
            proj_groups.pop().expect("proj group")
        } else {
            Tensor::cat(proj_groups, 0)
        };
        let embed = if embed_groups.len() == 1 {
            embed_groups.pop().expect("embed group")
        } else {
            Tensor::cat(embed_groups, 0)
        };
        let zero = Tensor::<B, 1>::zeros([1], &device);
        let (inv, sigreg, mut total) = if self.config.loss.lejepa.enabled {
            let inv = lejepa_invariance_loss(proj.clone());
            let sigreg = lejepa_sigreg_loss(proj.clone(), &self.config.loss.lejepa);
            let lambda = self.config.loss.lejepa.lambda.clamp(0.0, 1.0);
            let total = inv.clone().mul_scalar(1.0 - lambda) + sigreg.clone().mul_scalar(lambda);
            (inv, sigreg, total)
        } else {
            (zero.clone(), zero.clone(), zero.clone())
        };
        let (recon, recon_psnr) = if recon_enabled {
            let denom = recon_mask_sum.clone().add_scalar(LEJEPA_EPS);
            let recon = recon_loss_sum / denom;
            let weight = self.config.loss.recon.weight.max(0.0);
            total = total + recon.clone().mul_scalar(weight);
            let psnr = recon_psnr(recon.clone());
            (recon, psnr)
        } else {
            (zero.clone(), zero.clone())
        };

        let probe_source = probe_embed.as_ref().unwrap_or(&embed);
        let [view_count, batch, embed_dim] = probe_source.shape().dims::<3>();
        let embed_flat = probe_source
            .clone()
            .reshape([view_count * batch, embed_dim])
            .detach();
        let labels_flat = labels.clone().repeat_dim(0, view_count);
        let probe_logits = self.probe.forward(embed_flat);
        let probe_loss = self.probe_loss.forward(probe_logits.clone(), labels_flat.clone());
        let probe_pred = probe_logits
            .clone()
            .argmax(1)
            .reshape([view_count * batch]);
        let probe_acc = probe_pred
            .equal(labels_flat)
            .float()
            .mean();

        let probe_primary = probe_primary.map(|embed| self.probe.forward(embed.detach()));
        let artifact_views = artifact_views.unwrap_or_else(|| collected.artifact_views());
        let legend = if artifact_views.is_empty() {
            None
        } else if recon_enabled && artifact_views.len() == 3 {
            Some(vec![
                "input".to_string(),
                "masked_input".to_string(),
                "reconstruction".to_string(),
            ])
        } else if artifact_views.len() == 1 {
            Some(vec!["input".to_string()])
        } else {
            Some(
                (0..artifact_views.len())
                    .map(|idx| format!("view_{idx}"))
                    .collect(),
            )
        };
        let artifacts = build_lejepa_artifacts(
            &self.config,
            &artifact_views,
            None,
            heatmap_source,
            probe_primary,
            Some(labels),
            legend,
        );

        VisionLejepaLosses {
            total,
            inv,
            sigreg,
            recon,
            recon_psnr,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }

    pub(crate) fn recon_group_loss(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
        capture_artifacts: bool,
        randomize_mask: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let recon = match &self.recon {
            Some(recon) => recon,
            None => {
                let device = views
                    .get(0)
                    .map(|view| view.device())
                    .unwrap_or_default();
                let zero = Tensor::<B, 1>::zeros([1], &device);
                return (zero.clone(), zero, None);
            }
        };
        if views.is_empty() {
            let device = <B as BackendTrait>::Device::default();
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let device = views[0].device();
        let [batch, channels, height, width] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed_raw(stacked.clone());
        let [total, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
        let grid_h = patch.grid.height;
        let grid_w = patch.grid.width;
        if grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }
        let patch_size = self.model.patch_size().max(1);

        let target_patches = patchify(stacked, patch_size);
        let mask_ratio = self.config.loss.recon.mask_ratio;
        let mask = sample_patch_mask(&device, total, tokens, mask_ratio, randomize_mask);
        let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
        let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
        let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
        if let Some(mask_token) = &self.mask_token {
            let token = mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, total)
                .repeat_dim(1, tokens);
            masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
        }
        let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
        let embed_out =
            self.model
                .forward_tokens_embed_steps_rollout(masked_tokens, steps, backprop_steps);

        let pred_patches = recon.forward(embed_out.patch_tokens);
        let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
        debug_assert_eq!(
            target_patches.shape().dims::<3>(),
            pred_patches.shape().dims::<3>(),
            "recon patches shape mismatch"
        );
        if total == 0 || tokens == 0 || patch_dim == 0 {
            let zero = Tensor::<B, 1>::zeros([1], &device);
            return (zero.clone(), zero, None);
        }

        let diff = pred_patches.clone() - target_patches.clone();
        let loss_sum = diff
            .powf_scalar(2.0)
            .mul(mask_expanded.clone())
            .sum();
        let mask_sum = mask.clone().sum().mul_scalar(patch_dim as f32);

        let artifacts = if capture_artifacts && batch > 0 {
            let pred_first = pred_patches.slice_dim(0, 0..batch);
            let target_first = target_patches.slice_dim(0, 0..batch);
            let mask_first = mask.slice_dim(0, 0..batch);
            let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let masked_patches = target_first.clone().mul(keep.clone());
            let recon_patches =
                pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
            let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
            let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
            let residual = (pred_first - target_first).mul(mask_expanded);
            Some((vec![views[0].clone(), masked_view, recon_view], residual))
        } else {
            None
        };

        (loss_sum, mask_sum, artifacts)
    }

    pub(crate) fn forward_view_group(
        &self,
        views: &[Tensor<B, 4>],
        steps: usize,
        backprop_steps: usize,
    ) -> ViewGroupOutput<B> {
        let view_count = views.len();
        let [batch, _, _, _] = views[0].shape().dims::<4>();
        let stacked = stack_views(views);
        let patch = self.model.patch_embed(stacked);
        let embed_out = self
            .model
            .forward_tokens_embed_steps_rollout(patch.tokens, steps, backprop_steps);
        let cls_embed = embed_out.cls_token;
        let patch_tokens = embed_out.patch_tokens;
        let [total, embed_dim] = cls_embed.shape().dims::<2>();
        debug_assert_eq!(total, view_count * batch, "lejepa embed mismatch");
        let tokens = Tensor::cat(
            vec![cls_embed.clone().unsqueeze_dim::<3>(1), patch_tokens.clone()],
            1,
        );
        let proj_tokens = self.model.project_tokens(tokens);
        let proj_dim = proj_tokens.shape().dims::<3>()[2];
        let proj_cls = proj_tokens
            .slice_dim(1, 0..1)
            .reshape([view_count, batch, proj_dim]);
        let embed_cls = cls_embed.reshape([view_count, batch, embed_dim]);
        ViewGroupOutput {
            proj: proj_cls,
            embed: embed_cls,
            patch_tokens,
        }
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionMaeModel<B: BackendTrait> {
    pub(crate) model: VisionDragonHatchling<B>,
    pub(crate) recon: VisionReconstructionHead<B>,
    pub(crate) mask_token: Param<Tensor<B, 2>>,
    pub(crate) config: VisionMaeConfig,
    #[module(ignore)]
    pub(crate) rollout: VisionRollout,
}

pub(crate) struct VisionMaeLosses<B: BackendTrait> {
    pub(crate) total: Tensor<B, 1>,
    pub(crate) recon: Tensor<B, 1>,
    pub(crate) recon_psnr: Tensor<B, 1>,
    pub(crate) artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionMaeModel<B> {
    pub(crate) fn new(
        model: VisionDragonHatchling<B>,
        config: VisionMaeConfig,
        embed_dim: usize,
        rollout: VisionRollout,
        recon_patch_dim: usize,
        device: &B::Device,
    ) -> Self {
        let recon = VisionReconstructionHead::new(
            embed_dim,
            config.loss.recon.hidden_dim,
            recon_patch_dim,
            device,
        );
        let token = Tensor::<B, 2>::random(
            [1, embed_dim.max(1)],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        );
        let mask_token = Param::from_tensor(token);
        Self {
            model,
            recon,
            mask_token,
            config,
            rollout,
        }
    }

    pub(crate) fn forward_losses(
        &self,
        batch: ImageNetBatch<B>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> VisionMaeLosses<B> {
        let ImageNetBatch { images, labels, .. } = batch;
        let (loss_sum, mask_sum, artifacts) =
            self.recon_loss(images, steps, backprop_steps, randomize_mask, capture_artifacts);
        let denom = mask_sum.clone().add_scalar(LEJEPA_EPS);
        let recon = loss_sum / denom;
        let recon_psnr = recon_psnr(recon.clone());
        let total = recon
            .clone()
            .mul_scalar(self.config.loss.recon.weight.max(0.0));

        let artifacts = artifacts.and_then(|(views, residual)| {
            build_lejepa_artifacts(
                &VisionLejepaConfig {
                    artifact_every: self.config.artifact_every,
                    artifact_max_images: self.config.artifact_max_images,
                    artifact_max_views: self.config.artifact_max_views,
                    ..VisionLejepaConfig::default()
                },
                &views,
                None,
                Some(residual),
                None,
                Some(labels),
                Some(vec![
                    "input".to_string(),
                    "masked_input".to_string(),
                    "reconstruction".to_string(),
                ]),
            )
        });

        VisionMaeLosses {
            total,
            recon,
            recon_psnr,
            artifacts,
        }
    }

    pub(crate) fn recon_loss(
        &self,
        images: Tensor<B, 4>,
        steps: usize,
        backprop_steps: usize,
        randomize_mask: bool,
        capture_artifacts: bool,
    ) -> (
        Tensor<B, 1>,
        Tensor<B, 1>,
        Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)>,
    ) {
        let device = images.device();
        let patch_size = self.model.patch_size().max(1);
        let pyramid_levels = self.config.pyramid_levels.max(1);
        let mut pyramid = Vec::with_capacity(pyramid_levels);
        let mut current = images;
        pyramid.push(current.clone());
        for _ in 1..pyramid_levels {
            let Some(next) = downsample_image(current.clone()) else {
                break;
            };
            pyramid.push(next.clone());
            current = next;
        }
        pyramid.reverse();

        let level_count = pyramid.len();
        let mut loss_sum: Option<Tensor<B, 1>> = None;
        let mut mask_sum: Option<Tensor<B, 1>> = None;
        let mut artifacts: Option<(Vec<Tensor<B, 4>>, Tensor<B, 3>)> = None;

        for (level_idx, level_images) in pyramid.into_iter().enumerate() {
            let [batch, channels, height, width] = level_images.shape().dims::<4>();
            let patch = self.model.patch_embed_raw(level_images.clone());
            let [_, tokens, embed_dim] = patch.tokens.shape().dims::<3>();
            let grid_h = patch.grid.height;
            let grid_w = patch.grid.width;
            if batch == 0 || grid_h == 0 || grid_w == 0 || grid_h * grid_w != tokens {
                continue;
            }

            let target_patches = patchify(level_images.clone(), patch_size);
            let mask_ratio = self.config.loss.recon.mask_ratio;
            let mask = sample_patch_mask(&device, batch, tokens, mask_ratio, randomize_mask);

            let mask_expanded = mask.clone().unsqueeze_dim::<3>(2);
            let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
            let mut masked_tokens = patch.tokens.clone().mul(keep.clone());
            let token = self
                .mask_token
                .val()
                .reshape([1, 1, embed_dim])
                .repeat_dim(0, batch)
                .repeat_dim(1, tokens);
            masked_tokens = masked_tokens + token.mul(mask_expanded.clone());
            let masked_tokens = self.model.add_patch_position(masked_tokens, patch.grid);
            let embed_out = self.model.forward_tokens_embed_steps_rollout(
                masked_tokens,
                steps,
                backprop_steps,
            );

            let pred_patches = self.recon.forward(embed_out.patch_tokens);
            let [total, tokens, patch_dim] = pred_patches.shape().dims::<3>();
            if total == 0 || tokens == 0 || patch_dim == 0 {
                continue;
            }

            let diff = pred_patches.clone() - target_patches.clone();
            let loss_sum_level = diff
                .powf_scalar(2.0)
                .mul(mask_expanded.clone())
                .sum();
            let mask_sum_level = mask.clone().sum().mul_scalar(patch_dim as f32);
            loss_sum = Some(match loss_sum {
                Some(accum) => accum + loss_sum_level,
                None => loss_sum_level,
            });
            mask_sum = Some(match mask_sum {
                Some(accum) => accum + mask_sum_level,
                None => mask_sum_level,
            });

            if capture_artifacts && level_idx + 1 == level_count && batch > 0 {
                let pred_first = pred_patches.slice_dim(0, 0..batch);
                let target_first = target_patches.slice_dim(0, 0..batch);
                let mask_first = mask.slice_dim(0, 0..batch);
                let mask_expanded = mask_first.clone().unsqueeze_dim::<3>(2);
                let keep = mask_expanded.clone().mul_scalar(-1.0).add_scalar(1.0);
                let masked_patches = target_first.clone().mul(keep.clone());
                let recon_patches =
                    pred_first.clone().mul(mask_expanded.clone()) + target_first.clone().mul(keep);
                let masked_view = unpatchify(masked_patches, patch_size, height, width, channels);
                let recon_view = unpatchify(recon_patches, patch_size, height, width, channels);
                let residual = (pred_first - target_first).mul(mask_expanded);
                artifacts = Some((vec![level_images.clone(), masked_view, recon_view], residual));
            }
        }

        let zero = Tensor::<B, 1>::zeros([1], &device);
        let loss_sum = loss_sum.unwrap_or_else(|| zero.clone());
        let mask_sum = mask_sum.unwrap_or_else(|| zero);
        (loss_sum, mask_sum, artifacts)
    }
}


