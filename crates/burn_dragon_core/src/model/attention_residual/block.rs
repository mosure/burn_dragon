use burn::module::{Module, Param};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData};

use super::{
    AttentionResidual, AttentionResidualConfig, BlockAttentionResidualConfig,
    BlockAttentionResidualSummaryMode,
};

#[derive(Module, Debug)]
pub struct BlockAttentionResidual<B: Backend> {
    layers_per_block: usize,
    block_history_window: Option<usize>,
    intra_block_history_window: usize,
    summary_mode: BlockAttentionResidualSummaryMode,
    cache_block_summaries: bool,
    two_phase_compute: bool,
    summary_proj: Param<Tensor<B, 2>>,
    attention: AttentionResidual<B>,
}

impl<B: Backend> BlockAttentionResidual<B> {
    pub fn new(
        config: &BlockAttentionResidualConfig,
        dense_dim: usize,
        device: &B::Device,
    ) -> Self {
        let attention = AttentionResidual::new(
            &AttentionResidualConfig {
                enabled: config.enabled,
                last_layers: config.last_layers,
                num_heads: config.resolved_num_heads(dense_dim),
                history_window: None,
                dropout: config.dropout,
                recency_bias: config.recency_bias,
            },
            dense_dim,
            device,
        );

        Self {
            layers_per_block: config.resolved_layers_per_block(),
            block_history_window: config.block_history_window,
            intra_block_history_window: config.resolved_intra_block_history_window(),
            summary_mode: config.summary_mode,
            cache_block_summaries: config.cache_block_summaries,
            two_phase_compute: config.two_phase_compute,
            summary_proj: Param::from_tensor(identity_matrix(dense_dim, device)),
            attention,
        }
    }

    pub fn branch_input(
        &self,
        current: Tensor<B, 4>,
        residual_history: &[Tensor<B, 4>],
    ) -> Tensor<B, 4> {
        let candidates = self.build_candidates(current.clone(), residual_history);
        self.attention.branch_input(current, &candidates)
    }

    fn build_candidates(
        &self,
        current: Tensor<B, 4>,
        residual_history: &[Tensor<B, 4>],
    ) -> Vec<Tensor<B, 4>> {
        let mut candidates = residual_history.to_vec();
        if candidates.is_empty() {
            candidates.push(current);
        }

        let total = candidates.len();
        let local_window = self.intra_block_history_window.max(1).min(total);
        let raw_start = total.saturating_sub(local_window);
        let raw_recent = candidates[raw_start..].to_vec();

        let mut block_summaries = Vec::new();
        if raw_start > 0 {
            let block_count = raw_start.div_ceil(self.layers_per_block);
            let keep_blocks = self
                .block_history_window
                .unwrap_or(block_count)
                .max(1)
                .min(block_count);
            let first_block = block_count.saturating_sub(keep_blocks);
            for block_index in first_block..block_count {
                let start = block_index * self.layers_per_block;
                let end = ((block_index + 1) * self.layers_per_block).min(raw_start);
                block_summaries.push(self.summarize_block(&candidates[start..end]));
            }
        }

        block_summaries.extend(raw_recent);
        block_summaries
    }

    fn summarize_block(&self, layers: &[Tensor<B, 4>]) -> Tensor<B, 4> {
        let history = Tensor::cat(layers.to_vec(), 1);
        let [batch, _, time, dim] = history.shape().dims::<4>();
        let summary = history.mean_dim(1).reshape([batch, 1, time, dim]);
        match self.summary_mode {
            BlockAttentionResidualSummaryMode::MeanPool => summary,
            BlockAttentionResidualSummaryMode::LearnedProjection => summary
                .reshape([batch * time, dim])
                .matmul(self.summary_proj.val())
                .reshape([batch, 1, time, dim]),
        }
    }

    pub fn cache_block_summaries(&self) -> bool {
        self.cache_block_summaries
    }

    pub fn two_phase_compute(&self) -> bool {
        self.two_phase_compute
    }

    #[cfg(test)]
    pub(crate) fn debug_candidate_count(
        &self,
        current: Tensor<B, 4>,
        residual_history: &[Tensor<B, 4>],
    ) -> usize {
        self.build_candidates(current, residual_history).len()
    }
}

fn identity_matrix<B: Backend>(dim: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut values = vec![0.0f32; dim * dim];
    for idx in 0..dim {
        values[idx * dim + idx] = 1.0;
    }
    Tensor::<B, 2>::from_data(TensorData::new(values, [dim, dim]), device)
}
