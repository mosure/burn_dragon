use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::activation;
use burn::tensor::backend::Backend;

#[derive(Module, Debug)]
pub struct MicroTransformerBlock<B: Backend> {
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

impl<B: Backend> MicroTransformerBlock<B> {
    pub fn new(embed_dim: usize, heads: usize, mlp_ratio: usize, device: &B::Device) -> Self {
        let heads = heads.max(1);
        let head_dim = (embed_dim / heads).max(1);
        let mlp_dim = embed_dim.saturating_mul(mlp_ratio).max(1);
        Self {
            norm_attn: LayerNormConfig::new(embed_dim).init(device),
            qkv: LinearConfig::new(embed_dim, embed_dim * 3).init(device),
            proj: LinearConfig::new(embed_dim, embed_dim).init(device),
            norm_mlp: LayerNormConfig::new(embed_dim).init(device),
            mlp_in: LinearConfig::new(embed_dim, mlp_dim).init(device),
            mlp_out: LinearConfig::new(mlp_dim, embed_dim).init(device),
            heads,
            head_dim,
        }
    }

    pub fn forward(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
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
        let q = split_heads(q, self.heads, self.head_dim);
        let k = split_heads(k, self.heads, self.head_dim);
        let v = split_heads(v, self.heads, self.head_dim);
        let scale = (self.head_dim as f32).sqrt().max(1.0);
        let scores = q.matmul(k.swap_dims(2, 3)).div_scalar(scale);
        let attn = activation::softmax(scores, 3);
        let out = attn.matmul(v);
        self.proj.forward(merge_heads(out))
    }
}

fn split_heads<B: Backend>(tokens: Tensor<B, 3>, heads: usize, head_dim: usize) -> Tensor<B, 4> {
    let [batch, time, _] = tokens.shape().dims::<3>();
    tokens
        .reshape([batch, time, heads.max(1), head_dim.max(1)])
        .swap_dims(1, 2)
}

fn merge_heads<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, time, head_dim] = tokens.shape().dims::<4>();
    tokens
        .swap_dims(1, 2)
        .reshape([batch, time, heads * head_dim])
}
