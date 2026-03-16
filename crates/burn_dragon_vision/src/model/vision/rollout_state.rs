use burn::prelude::*;
use burn_dragon_core::StructuredTopologyState;

use super::VisionCellularState;

/// Topology-agnostic recurrent state for VisionDragon rollout APIs.
///
/// Dense backbones only carry the current dense token stream. Pyramid and cellular variants also
/// carry their explicit persistent recurrent banks.
#[derive(Clone)]
pub enum VisionRolloutState<B: Backend> {
    Dense { token_state: Tensor<B, 3> },
    Pyramid(StructuredTopologyState<B>),
    Cellular(VisionCellularState<B>),
}

impl<B: Backend> VisionRolloutState<B> {
    pub fn detach(&self) -> Self {
        match self {
            Self::Dense { token_state } => Self::Dense {
                token_state: token_state.clone().detach(),
            },
            Self::Pyramid(state) => Self::Pyramid(state.detach()),
            Self::Cellular(state) => Self::Cellular(state.detach()),
        }
    }
}
