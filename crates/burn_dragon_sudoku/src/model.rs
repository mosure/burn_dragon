use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{activation, Distribution as TensorDistribution, Int, Tensor, TensorData};

use burn_dragon_core::{BDH, BDHConfig, FusedKernelConfig, HaltHead, ManifoldHyperConnections, ModelState};

use crate::config::{SudokuModelConfig, SudokuPolicyHead};
use crate::vocab::{GRID_LEN, VOCAB_SIZE};

const GRID_SIDE: usize = 9;
const POLICY_HEAD_CACHE: u8 = 0;
const POLICY_HEAD_SUMMARY_POS: u8 = 1;
const POLICY_HEAD_SUMMARY_MLP: u8 = 2;

#[derive(Module, Debug)]
pub struct SudokuSaccadeModel<B: Backend> {
    pub core: BDH<B>,
    pub row_embed: Embedding<B>,
    pub col_embed: Embedding<B>,
    pub input_proj: Linear<B>,
    pub policy_q: Linear<B>,
    pub policy_k: Linear<B>,
    pub policy_pos_keys: Param<Tensor<B, 2>>,
    pub policy_mlp_fc1: Linear<B>,
    pub policy_mlp_fc2: Linear<B>,
    pub value_head: Linear<B>,
    pub value_baseline: Linear<B>,
    pub update_mlp_fc1: Linear<B>,
    pub update_mlp_fc2: Linear<B>,
    pub cache_mhc: Option<ManifoldHyperConnections<B>>,
    pub summary_tokens: Param<Tensor<B, 2>>,
    pub summary_norm: LayerNorm<B>,
    pub halt_head: HaltHead<B>,
    #[module(ignore)]
    summary_token_count: usize,
    #[module(ignore)]
    policy_heads: usize,
    #[module(ignore)]
    policy_head_dim: usize,
    #[module(ignore)]
    policy_head_kind: u8,
    #[module(ignore)]
    policy_mlp_hidden: usize,
    #[module(ignore)]
    cache_streams: usize,
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
        let row_embed = EmbeddingConfig::new(GRID_SIDE, model_config.n_embd).init(device);
        let col_embed = EmbeddingConfig::new(GRID_SIDE, model_config.n_embd).init(device);
        let input_proj = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let policy_q = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let policy_k = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let policy_pos_keys = Param::from_tensor(Tensor::<B, 2>::random(
            [GRID_LEN, model_config.n_embd],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let policy_mlp_hidden = model_config
            .n_embd
            .saturating_mul(config.policy_mlp_hidden_mult.max(1));
        let policy_mlp_fc1 =
            LinearConfig::new(model_config.n_embd, policy_mlp_hidden).init(device);
        let policy_mlp_fc2 = LinearConfig::new(policy_mlp_hidden, GRID_LEN).init(device);
        let value_head = LinearConfig::new(model_config.n_embd, VOCAB_SIZE).init(device);
        let value_baseline = LinearConfig::new(model_config.n_embd, 1).init(device);
        let update_mlp_fc1 =
            LinearConfig::new(model_config.n_embd * 3, model_config.n_embd * 2).init(device);
        let update_mlp_fc2 =
            LinearConfig::new(model_config.n_embd * 2, model_config.n_embd).init(device);
        let cache_streams = if config.cache_mhc.enabled {
            config.cache_mhc.num_streams.max(1)
        } else {
            1
        };
        let cache_mhc = if config.cache_mhc.enabled {
            Some(ManifoldHyperConnections::new(&config.cache_mhc.to_core(), 0, device))
        } else {
            None
        };
        let summary_tokens = Param::from_tensor(Tensor::<B, 2>::random(
            [config.summary_tokens.max(1), model_config.n_embd],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let summary_norm = LayerNormConfig::new(model_config.n_embd).init(device);
        let halt_head = HaltHead::new(model_config.n_embd, device);
        let policy_heads = config.policy_heads.max(1);
        let policy_head_dim = model_config.n_embd / policy_heads;
        let policy_head_kind = match config.policy_head {
            SudokuPolicyHead::Cache => POLICY_HEAD_CACHE,
            SudokuPolicyHead::SummaryPos => POLICY_HEAD_SUMMARY_POS,
            SudokuPolicyHead::SummaryMlp => POLICY_HEAD_SUMMARY_MLP,
        };
        Self {
            core,
            row_embed,
            col_embed,
            input_proj,
            policy_q,
            policy_k,
            policy_pos_keys,
            policy_mlp_fc1,
            policy_mlp_fc2,
            value_head,
            value_baseline,
            update_mlp_fc1,
            update_mlp_fc2,
            cache_mhc,
            summary_tokens,
            summary_norm,
            halt_head,
            summary_token_count: config.summary_tokens.max(1),
            policy_heads,
            policy_head_dim,
            policy_head_kind,
            policy_mlp_hidden,
            cache_streams,
        }
    }

    pub fn grid_row_col_ids(&self, batch: usize, device: &B::Device) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        let mut rows = Vec::with_capacity(GRID_LEN);
        let mut cols = Vec::with_capacity(GRID_LEN);
        for idx in 0..GRID_LEN {
            rows.push((idx / GRID_SIDE) as i64);
            cols.push((idx % GRID_SIDE) as i64);
        }
        let row_ids = Tensor::<B, 1, Int>::from_data(TensorData::new(rows, [GRID_LEN]), device)
            .unsqueeze_dim::<2>(0)
            .expand([batch.max(1), GRID_LEN]);
        let col_ids = Tensor::<B, 1, Int>::from_data(TensorData::new(cols, [GRID_LEN]), device)
            .unsqueeze_dim::<2>(0)
            .expand([batch.max(1), GRID_LEN]);
        (row_ids, col_ids)
    }

    pub fn cell_embeddings_with_positions(
        &self,
        tokens: Tensor<B, 2, Int>,
        row_ids: Tensor<B, 2, Int>,
        col_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let token_emb = self.core.embed_tokens(tokens);
        let row_emb = self.row_embed.forward(row_ids);
        let col_emb = self.col_embed.forward(col_ids);
        token_emb + row_emb + col_emb
    }

    pub fn project_input_tokens(&self, embedded: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = embedded.shape().dims();
        let flat = embedded.reshape([batch * time, dim]);
        let projected = self.input_proj.forward(flat);
        projected.reshape([batch, time, dim])
    }

    pub fn cell_embeddings(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let device = tokens.device();
        let [batch, _] = tokens.shape().dims::<2>();
        let (row_ids, col_ids) = self.grid_row_col_ids(batch, &device);
        self.cell_embeddings_with_positions(tokens, row_ids, col_ids)
    }

    pub fn init_summary_tokens(&self, batch: usize) -> Tensor<B, 3> {
        let tokens = self.summary_tokens.val();
        let [count, dim] = tokens.shape().dims();
        let summary = tokens
            .reshape([1, count.max(1), dim])
            .expand([batch.max(1), count.max(1), dim]);
        self.summary_norm.forward(summary)
    }

    pub fn cache_streams(&self) -> usize {
        self.cache_streams.max(1)
    }
    pub fn summary_token_count(&self) -> usize {
        self.summary_token_count
    }

    pub fn policy_logits_from_cache(
        &self,
        summary_tokens: Tensor<B, 3>,
        cache: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [batch, summary_len, dim] = summary_tokens.shape().dims();
        let device = summary_tokens.device();
        if batch == 0 || summary_len == 0 {
            return Tensor::<B, 2>::zeros([batch.max(1), GRID_LEN], &device);
        }

        match self.policy_head_kind {
            POLICY_HEAD_CACHE => {
                let [_batch_cache, time, _dim_cache] = cache.shape().dims();
                if time == 0 {
                    return Tensor::<B, 2>::zeros([batch.max(1), time.max(1)], &device);
                }

                let q = self
                    .policy_q
                    .forward(summary_tokens.reshape([batch * summary_len, dim]))
                    .reshape([batch, summary_len, dim]);
                let k = self
                    .policy_k
                    .forward(cache.reshape([batch * time, dim]))
                    .reshape([batch, time, dim]);
                let heads = self.policy_heads.max(1);
                let head_dim = self.policy_head_dim.max(1);

                let q = q
                    .reshape([batch, summary_len, heads, head_dim])
                    .swap_dims(1, 2);
                let k = k
                    .reshape([batch, time, heads, head_dim])
                    .swap_dims(1, 2);
                let q = q.unsqueeze_dim::<5>(3);
                let k = k.unsqueeze_dim::<5>(2);
                let scores = q.mul(k).sum_dim(4);
                let mut scores = scores.reshape([batch, heads, summary_len, time]);
                let scale = (head_dim as f32).sqrt().max(1.0);
                scores = scores.div_scalar(scale);
                let scores = scores.mean_dim(2).reshape([batch, heads, time]);
                scores.mean_dim(1).reshape([batch, time])
            }
            POLICY_HEAD_SUMMARY_POS => {
                let summary = summary_tokens.mean_dim(1).reshape([batch, dim]);
                let q = self.policy_q.forward(summary);
                let keys = self
                    .policy_pos_keys
                    .val()
                    .reshape([1, GRID_LEN, dim]);
                let q = q.unsqueeze_dim::<3>(1);
                let logits = q.mul(keys).sum_dim(2).reshape([batch, GRID_LEN]);
                let scale = (dim as f32).sqrt().max(1.0);
                logits.div_scalar(scale)
            }
            POLICY_HEAD_SUMMARY_MLP => {
                let summary = summary_tokens.mean_dim(1).reshape([batch, dim]);
                let hidden = self.policy_mlp_fc1.forward(summary);
                let hidden = activation::gelu(hidden);
                self.policy_mlp_fc2.forward(hidden)
            }
            _ => Tensor::<B, 2>::zeros([batch.max(1), GRID_LEN], &device),
        }
    }

    pub fn value_baseline_from_summary_tokens(
        &self,
        summary_tokens: Tensor<B, 3>,
    ) -> Tensor<B, 2> {
        let [batch, count, dim] = summary_tokens.shape().dims();
        if batch == 0 || count == 0 {
            return Tensor::<B, 2>::zeros([batch.max(1), 1], &summary_tokens.device());
        }
        let pooled = summary_tokens.mean_dim(1);
        let flat = pooled.reshape([batch, dim]);
        self.value_baseline.forward(flat).reshape([batch, 1])
    }


    pub fn update_cell_embedding(
        &self,
        summary_tokens: Tensor<B, 3>,
        cache_cell: Tensor<B, 3>,
        token_emb: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [batch, _, dim] = summary_tokens.shape().dims();
        if batch == 0 || dim == 0 {
            return Tensor::<B, 3>::zeros([batch.max(1), 1, dim.max(1)], &summary_tokens.device());
        }
        let summary = summary_tokens.mean_dim(1).reshape([batch, 1, dim]);
        let stacked = Tensor::cat(vec![summary, cache_cell, token_emb], 2);
        let flat = stacked.reshape([batch, dim * 3]);
        let hidden = activation::gelu(self.update_mlp_fc1.forward(flat));
        let updated = self.update_mlp_fc2.forward(hidden);
        updated.reshape([batch, 1, dim])
    }
    pub fn value_logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = hidden.shape().dims();
        let flat = hidden.reshape([batch * time, dim]);
        let logits = self.value_head.forward(flat);
        logits.reshape([batch, time, VOCAB_SIZE])
    }

    pub fn value_logits_from_cache(&self, cache: Tensor<B, 3>) -> Tensor<B, 3> {
        self.value_logits_from_hidden(cache)
    }

    pub fn init_state(&self) -> ModelState<B> {
        self.core.init_state()
    }

    pub fn forward_with_hidden(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.core.forward_with_hidden(tokens)
    }

    pub fn forward_with_hidden_and_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.core.forward_with_hidden_and_state(tokens, state)
    }

    pub fn forward_with_hidden_and_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.core.forward_with_hidden_and_state_embedded(embedded, state)
    }

    pub fn halt_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        self.halt_head.forward(hidden)
    }

    pub fn halt_logit(&self, hidden: Tensor<B, 3>) -> Tensor<B, 2> {
        self.halt_head.forward_pooled(hidden)
    }

    pub fn halt_logit_from_summary_tokens(&self, summary_tokens: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch, _count, dim] = summary_tokens.shape().dims();
        let pooled = summary_tokens.mean_dim(1).reshape([batch, 1, dim]);
        self.halt_head.forward_pooled(pooled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SudokuCacheMhcConfig;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

    #[test]
    fn policy_head_variants_output_grid_logits() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let heads = [
            SudokuPolicyHead::SummaryPos,
            SudokuPolicyHead::SummaryMlp,
        ];

        for head in heads {
            let config = SudokuModelConfig {
                n_layer: 1,
                n_embd: 32,
                n_head: 1,
                mlp_internal_dim_multiplier: 2,
                summary_tokens: 1,
                policy_heads: 1,
                policy_head: head,
                policy_mlp_hidden_mult: 2,
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
            };
            let model = SudokuSaccadeModel::new(&config, &device);
            let batch = 2;
            let summary = model.init_summary_tokens(batch);
            let cache = Tensor::<B, 3>::zeros([batch, GRID_LEN, config.n_embd], &device);
            let logits = model.policy_logits_from_cache(summary, cache);
            let [out_batch, out_grid] = logits.shape().dims();
            assert_eq!(out_batch, batch);
            assert_eq!(out_grid, GRID_LEN);
        }
    }
}





