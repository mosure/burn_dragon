use super::*;

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
                let linear = VisionSaccadeProjection::new(
                    embed_dim,
                    embed_dim,
                    &DragonNormConfig::default(),
                    device,
                );
                let param_count =
                    linear_params(embed_dim, embed_dim) + layer_norm_params(embed_dim);
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
                channels, kernel, expansion, device,
            ));
        }
        let mut param_count = layer_norm_params(embed_dim)
            + linear_params(embed_dim, channels)
            + linear_params(channels, embed_dim);
        param_count = param_count
            .saturating_add(cnn_block_params(channels, kernel, expansion).saturating_mul(blocks));
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
                embed_dim, heads, mlp_ratio, device,
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
        let mut x = tokens
            + self
                .radial
                .forward(batch, token_count, grid_h, grid_w, &device);
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
        let r = Tensor::<B, 1>::from_data(TensorData::new(positions, [tokens]), device)
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
    tokens
        .reshape([batch, grid_h, grid_w, channels])
        .swap_dims(1, 3)
        .swap_dims(2, 3)
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
    value.clamp(3, 7)
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
    while heads > 1 && !embed_dim.is_multiple_of(heads) {
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
