mod block;
mod config;
mod reference;

pub use block::BlockAttentionResidual;
pub use config::{
    AttentionResidualConfig, BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode,
    ResidualConnectorKind,
};
pub use reference::AttentionResidual;

#[cfg(test)]
mod tests;
