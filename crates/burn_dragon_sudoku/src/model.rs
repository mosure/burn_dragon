use burn::module::Module;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};

use burn_dragon_core::{BDH, BDHConfig, FusedKernelConfig, HaltHead};

use crate::config::SudokuModelConfig;
use crate::vocab::VOCAB_SIZE;

#[derive(Module, Debug)]
pub struct SudokuSaccadeModel<B: Backend> {
    pub core: BDH<B>,
    pub policy_head: Linear<B>,
    pub halt_head: HaltHead<B>,
}

impl SudokuModelConfig {
    pub fn to_bdh_config(&self) -> BDHConfig {
        let fused = FusedKernelConfig {
            enabled: self.fused_kernels,
            relu_threshold: self.relu_threshold,
            ..Default::default()
        };

        BDHConfig {
            n_layer: self.n_layer,
            n_embd: self.n_embd,
            dropout: self.dropout,
            n_head: self.n_head,
            mlp_internal_dim_multiplier: self.mlp_internal_dim_multiplier,
            n_expert: 1,
            vocab_size: VOCAB_SIZE,
            fused_kernels: fused,
        }
    }
}

impl<B: Backend> SudokuSaccadeModel<B> {
    pub fn new(config: &SudokuModelConfig, device: &B::Device) -> Self {
        let model_config = config.to_bdh_config();
        let core = BDH::new(model_config.clone(), device);
        let policy_head = LinearConfig::new(model_config.n_embd, 1).init(device);
        let halt_head = HaltHead::new(model_config.n_embd, device);
        Self {
            core,
            policy_head,
            halt_head,
        }
    }

    pub fn forward_with_hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.core.forward_with_hidden(tokens)
    }

    pub fn policy_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, time, dim] = hidden.shape().dims();
        let flat = hidden.reshape([batch * time, dim]);
        let logits = self.policy_head.forward(flat);
        logits.reshape([batch, time])
    }

    pub fn halt_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        self.halt_head.forward(hidden)
    }

    pub fn halt_logit(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        self.halt_head.forward_pooled(hidden)
    }
}
