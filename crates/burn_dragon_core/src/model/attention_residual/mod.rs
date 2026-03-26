mod block;
mod config;
mod history;
mod reference;

pub use block::BlockAttentionResidual;
pub use config::{
    AttentionResidualConfig, BlockAttentionResidualConfig, BlockAttentionResidualSummaryMode,
    ResidualConnectorKind,
};
pub(crate) use history::ResidualHistory;
pub use reference::AttentionResidual;

#[cfg(test)]
mod tests;
