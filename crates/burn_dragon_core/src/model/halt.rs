use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Tensor;
use burn::tensor::backend::Backend;

use super::{DragonNorm, DragonNormConfig};

#[derive(Module, Debug)]
pub struct HaltHead<B: Backend> {
    norm: DragonNorm<B>,
    proj: Linear<B>,
}

impl<B: Backend> HaltHead<B> {
    pub fn new(embed_dim: usize, norm_config: &DragonNormConfig, device: &B::Device) -> Self {
        let norm = DragonNorm::new(norm_config, embed_dim, device);
        let proj = LinearConfig::new(embed_dim, 1).init(device);
        Self { norm, proj }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, time, dim] = hidden.shape().dims();
        let flat = hidden.reshape([batch * time, dim]);
        let flat = self.norm.forward(flat);
        let logits = self.proj.forward(flat);
        logits.reshape([batch, time])
    }

    pub fn forward_pooled(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let logits = self.forward(hidden);
        let [batch, _time] = logits.shape().dims();
        logits.mean_dim(1).reshape([batch, 1])
    }
}
