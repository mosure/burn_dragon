use crate::train::prelude::*;

#[derive(Module, Debug)]
pub(super) struct VisionVideoVjepa21Predictor<B: BackendTrait> {
    norm: DragonNorm<B>,
    hidden: Option<Linear<B>>,
    out: Linear<B>,
}

impl<B: BackendTrait> VisionVideoVjepa21Predictor<B> {
    pub(super) fn new(
        input_dim: usize,
        hidden_dim: usize,
        output_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, input_dim.max(1), device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(input_dim.max(1), hidden_dim.max(1)).init(device))
        } else {
            None
        };
        let out_in = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            input_dim.max(1)
        };
        let out = LinearConfig::new(out_in, output_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    pub(super) fn forward(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}
