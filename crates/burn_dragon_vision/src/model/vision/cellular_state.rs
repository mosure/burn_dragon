use burn::prelude::*;

/// Explicit recurrent state for the VisionDragon cellular backbone.
///
/// Persistence contract:
/// - `token_state` is the dense-space token stream carried across recurrent calls.
/// - `rho` is the only persistent associative memory bank. Its shape is
///   `[batch, heads, patch_tokens, neuron_space, dense_space]`.
/// - Per-step activations such as `y_gate` and `y_neuron` are intentionally not stored here.
///
/// There is no per-layer stack in the cellular backbone. A single recurrent block is reused
/// across rollout steps, so this state is shared across calls to that block.
#[derive(Clone)]
pub struct VisionCellularState<B: Backend> {
    pub token_state: Tensor<B, 3>,
    pub rho: Tensor<B, 5>,
    pub temporal_position: usize,
    pub prediction_age: usize,
}

impl<B: Backend> VisionCellularState<B> {
    pub fn detach(&self) -> Self {
        Self {
            token_state: self.token_state.clone().detach(),
            rho: self.rho.clone().detach(),
            temporal_position: self.temporal_position,
            prediction_age: self.prediction_age,
        }
    }
}
