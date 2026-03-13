use burn::module::{Module, Param};
use burn::nn::{Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use std::f32::consts::PI;

use burn_dragon_core::{
    BDH, BDHConfig, FusedKernelConfig, HaltHead, ManifoldHyperConnections,
    ManifoldHyperConnectionsConfig, ModelState,
};
use burn_dragon_train::WgpuRuntimeConfig;
use burn_dragon_train::wgpu::apply_wgpu_fused_core_override;

use crate::config::{
    SudokuCacheUpdateMode, SudokuGridPositional, SudokuModelConfig, SudokuPolicyHead,
};
use crate::vocab::{GRID_LEN, VOCAB_SIZE};

const GRID_SIDE: usize = 9;
const POLICY_HEAD_CACHE: u8 = 0;
const POLICY_HEAD_SUMMARY_POS: u8 = 1;
const POLICY_HEAD_SUMMARY_MLP: u8 = 2;
const WRITE_GATE_INIT: f32 = -6.0;

/// Sudoku task adapter over the shared BDH Dragon core.
///
/// Paper mapping:
/// - the shared `core` owns the paper-faithful `x_neuron`, `y_gate`, `y_neuron`, and per-layer
///   `rho` contract
/// - the remaining heads, cache updates, and summary tokens are task-specific dense-space adapters
///   for Sudoku policy/value/reconstruction behavior
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
    pub ca_q: Linear<B>,
    pub ca_k: Linear<B>,
    pub ca_v: Linear<B>,
    pub ca_update_fc1: Linear<B>,
    pub ca_update_fc2: Linear<B>,
    pub ca_norm: LayerNorm<B>,
    pub cache_mhc: Option<ManifoldHyperConnections<B>>,
    pub summary_tokens: Param<Tensor<B, 2>>,
    pub summary_norm: LayerNorm<B>,
    pub cache_norm: LayerNorm<B>,
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
    ca_heads: usize,
    #[module(ignore)]
    ca_head_dim: usize,
    #[module(ignore)]
    grid_positional: SudokuGridPositional,
    #[module(ignore)]
    grid_rope_theta: f32,
    #[module(ignore)]
    grid_rope_freqs: Option<Tensor<B, 2>>,
    #[module(ignore)]
    grid_rope_row_cos: Option<Tensor<B, 3>>,
    #[module(ignore)]
    grid_rope_row_sin: Option<Tensor<B, 3>>,
    #[module(ignore)]
    grid_rope_col_cos: Option<Tensor<B, 3>>,
    #[module(ignore)]
    grid_rope_col_sin: Option<Tensor<B, 3>>,
    #[module(ignore)]
    cache_streams: usize,
}

impl SudokuModelConfig {
    /// Dragon Hatchling paper terminology: dense/value-space dimension carried by the Sudoku
    /// adapter before and after the shared BDH core.
    pub fn dense_space_dim(&self) -> usize {
        self.n_embd.max(1)
    }

    /// Dragon Hatchling paper terminology: total neuron-space width used inside the shared BDH
    /// core.
    pub fn neuron_space_dim(&self) -> usize {
        self.dense_space_dim() * self.mlp_internal_dim_multiplier.max(1)
    }

    /// Dragon Hatchling paper terminology: per-head neuron-space width used by the shared BDH
    /// core.
    pub fn neuron_space_dim_per_head(&self) -> usize {
        let total = self.neuron_space_dim();
        let heads = self.n_head.max(1);
        assert!(
            total.is_multiple_of(heads),
            "Sudoku neuron space must be divisible by the number of heads"
        );
        total / heads
    }

    pub fn to_bdh_config(&self) -> BDHConfig {
        let mut fused = FusedKernelConfig {
            enabled: self.fused_kernels,
            relu_threshold: self.relu_threshold,
            ..Default::default()
        };
        fused.set_rotary_embedding(self.rotary_embedding);

        BDHConfig {
            n_layer: self.n_layer,
            n_embd: self.n_embd,
            dropout: self.dropout,
            n_head: self.n_head,
            mlp_internal_dim_multiplier: self.mlp_internal_dim_multiplier,
            n_expert: 1,
            vocab_size: VOCAB_SIZE,
            rollout_fast_steps_per_slow_step: 1,
            fused_kernels: fused,
            normalization: burn_dragon_core::DragonNormConfig::default(),
            mhc: ManifoldHyperConnectionsConfig::default(),
            y_neuron_recurrence: Default::default(),
        }
    }

    pub fn to_bdh_config_for_backend(
        &self,
        backend_name: &str,
        wgpu: &WgpuRuntimeConfig,
    ) -> BDHConfig {
        let mut config = self.to_bdh_config();
        apply_wgpu_fused_core_override(
            &mut config,
            backend_name,
            wgpu.training.fused_core_recurrent,
            wgpu.training.fused_core_rollout,
        );
        config
    }
}

impl<B: Backend> SudokuSaccadeModel<B> {
    pub fn new(config: &SudokuModelConfig, device: &B::Device) -> Self {
        Self::new_with_bdh_config(config, config.to_bdh_config(), device)
    }

    pub fn new_with_bdh_config(
        config: &SudokuModelConfig,
        model_config: BDHConfig,
        device: &B::Device,
    ) -> Self {
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
        let policy_mlp_fc1 = LinearConfig::new(model_config.n_embd, policy_mlp_hidden).init(device);
        let policy_mlp_fc2 = LinearConfig::new(policy_mlp_hidden, GRID_LEN).init(device);
        let value_head = LinearConfig::new(model_config.n_embd, VOCAB_SIZE).init(device);
        let value_baseline = LinearConfig::new(model_config.n_embd, 1).init(device);
        let update_mlp_fc1 =
            LinearConfig::new(model_config.n_embd * 3, model_config.n_embd * 2).init(device);
        let cache_update_mode = config.cache_update.mode.clone();
        let update_out = match cache_update_mode {
            SudokuCacheUpdateMode::Overwrite => model_config.n_embd,
            SudokuCacheUpdateMode::GatedResidual => model_config.n_embd * 2,
        };
        let mut update_mlp_fc2 =
            LinearConfig::new(model_config.n_embd * 2, update_out).init(device);
        if matches!(cache_update_mode, SudokuCacheUpdateMode::GatedResidual)
            && let Some(bias) = update_mlp_fc2.bias.take()
        {
            let bias = bias.map(|tensor| {
                let device = tensor.device();
                let [out_dim] = tensor.shape().dims();
                if out_dim <= 1 {
                    return tensor;
                }
                let split = out_dim / 2;
                let delta_bias = tensor.clone().slice_dim(0, 0..split);
                let gate_bias =
                    Tensor::<B, 1>::zeros([out_dim - split], &device).add_scalar(WRITE_GATE_INIT);
                Tensor::cat(vec![delta_bias, gate_bias], 0)
            });
            update_mlp_fc2.bias = Some(bias);
        }
        let cache_streams = if config.cache_mhc.enabled {
            config.cache_mhc.num_streams.max(1)
        } else {
            1
        };
        let ca_heads = model_config.n_head.max(1);
        let ca_head_dim = (model_config.n_embd / ca_heads).max(1);
        let ca_q = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let ca_k = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let ca_v = LinearConfig::new(model_config.n_embd, model_config.n_embd).init(device);
        let ca_update_fc1 =
            LinearConfig::new(model_config.n_embd * 2 + 1, model_config.n_embd * 2).init(device);
        let ca_update_fc2 =
            LinearConfig::new(model_config.n_embd * 2, model_config.n_embd).init(device);
        let ca_norm = LayerNormConfig::new(model_config.n_embd).init(device);
        let cache_mhc = if config.cache_mhc.enabled {
            Some(ManifoldHyperConnections::new(
                &config.cache_mhc.to_core(),
                0,
                device,
            ))
        } else {
            None
        };
        let summary_tokens = Param::from_tensor(Tensor::<B, 2>::random(
            [config.summary_tokens.max(1), model_config.n_embd],
            TensorDistribution::Normal(0.0, 0.02),
            device,
        ));
        let summary_norm = LayerNormConfig::new(model_config.n_embd).init(device);
        let cache_norm = LayerNormConfig::new(model_config.n_embd).init(device);
        let halt_head = HaltHead::new(model_config.n_embd, &model_config.normalization, device);
        let policy_heads = config.policy_heads.max(1);
        let policy_head_dim = model_config.n_embd / policy_heads;
        let grid_positional = config.grid_positional;
        let grid_rope_theta = config.grid_rope_theta;
        let (
            grid_rope_freqs,
            grid_rope_row_cos,
            grid_rope_row_sin,
            grid_rope_col_cos,
            grid_rope_col_sin,
        ) = if matches!(grid_positional, SudokuGridPositional::Rope2d) {
            let half = model_config.n_embd / 2;
            let freqs = Self::build_rope_freqs(half, grid_rope_theta, device);
            let freqs_3 = freqs.clone().reshape([1, 1, half]);
            let (row_ids, col_ids) = Self::build_grid_row_col_ids(1, device);
            let (row_cos, row_sin) = Self::rope_cos_sin(row_ids, freqs_3.clone());
            let (col_cos, col_sin) = Self::rope_cos_sin(col_ids, freqs_3);
            (
                Some(freqs),
                Some(row_cos),
                Some(row_sin),
                Some(col_cos),
                Some(col_sin),
            )
        } else {
            (None, None, None, None, None)
        };
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
            cache_norm,
            halt_head,
            summary_token_count: config.summary_tokens.max(1),
            policy_heads,
            policy_head_dim,
            policy_head_kind,
            policy_mlp_hidden,
            ca_heads,
            ca_head_dim,
            grid_positional,
            grid_rope_theta,
            grid_rope_freqs,
            grid_rope_row_cos,
            grid_rope_row_sin,
            grid_rope_col_cos,
            grid_rope_col_sin,
            cache_streams,
            ca_q,
            ca_k,
            ca_v,
            ca_update_fc1,
            ca_update_fc2,
            ca_norm,
        }
    }

    fn build_grid_row_col_ids(
        batch: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
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

    pub fn grid_row_col_ids(
        &self,
        batch: usize,
        device: &B::Device,
    ) -> (Tensor<B, 2, Int>, Tensor<B, 2, Int>) {
        Self::build_grid_row_col_ids(batch, device)
    }

    fn build_rope_freqs(dim: usize, theta: f32, device: &B::Device) -> Tensor<B, 2> {
        if dim == 0 {
            return Tensor::<B, 2>::zeros([1, 1], device);
        }
        let mut data = Vec::with_capacity(dim);
        for idx in 0..dim {
            let exponent = ((idx as f32 / 2.0).floor() * 2.0) / dim as f32;
            let value = 1.0 / theta.powf(exponent) / (2.0 * PI);
            data.push(value);
        }
        Tensor::<B, 1>::from_floats(data.as_slice(), device).reshape([1, dim])
    }

    fn rope_cos_sin(
        positions: Tensor<B, 2, Int>,
        freqs: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let [batch, time] = positions.shape().dims();
        let pos = positions.float().reshape([batch, time, 1]);
        let raw = pos * freqs;
        let phases = (raw.clone() - raw.clone().detach().floor()) * (2.0 * PI);
        let cos = phases.clone().cos();
        let sin = phases.sin();
        (cos, sin)
    }

    fn rope_apply(values: Tensor<B, 3>, cos: Tensor<B, 3>, sin: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = values.shape().dims();
        if dim == 0 || dim % 2 != 0 {
            return values;
        }
        let pairs = values.clone().reshape([batch, time, dim / 2, 2]);
        let even = pairs.clone().slice_dim(3, 0..1).squeeze_dim::<3>(3);
        let odd = pairs.slice_dim(3, 1..2).squeeze_dim::<3>(3);
        let rotated =
            Tensor::stack::<4>(vec![odd.clone().neg(), even], 3).reshape([batch, time, dim]);
        values * cos + rotated * sin
    }

    fn rope_with_positions(
        &self,
        values: Tensor<B, 3>,
        positions: Tensor<B, 2, Int>,
        freqs: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [_batch, _time, dim] = values.shape().dims();
        if dim == 0 || dim % 2 != 0 {
            return values;
        }
        let (cos, sin) = Self::rope_cos_sin(positions, freqs);
        Self::rope_apply(values, cos, sin)
    }

    fn apply_rope_2d(
        &self,
        token_emb: Tensor<B, 3>,
        row_ids: Tensor<B, 2, Int>,
        col_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, time, dim] = token_emb.shape().dims();
        if dim == 0 {
            return token_emb;
        }
        let half = dim / 2;
        if half == 0 || half * 2 != dim || half % 2 != 0 {
            return token_emb;
        }
        if time == GRID_LEN
            && let (Some(row_cos), Some(row_sin), Some(col_cos), Some(col_sin)) = (
                self.grid_rope_row_cos.as_ref(),
                self.grid_rope_row_sin.as_ref(),
                self.grid_rope_col_cos.as_ref(),
                self.grid_rope_col_sin.as_ref(),
            )
        {
            let row_cos = row_cos.clone().expand([batch, time, half]);
            let row_sin = row_sin.clone().expand([batch, time, half]);
            let col_cos = col_cos.clone().expand([batch, time, half]);
            let col_sin = col_sin.clone().expand([batch, time, half]);
            let row_part = token_emb.clone().slice_dim(2, 0..half);
            let col_part = token_emb.clone().slice_dim(2, half..(half * 2));
            let row_rot = Self::rope_apply(row_part, row_cos, row_sin);
            let col_rot = Self::rope_apply(col_part, col_cos, col_sin);
            return Tensor::cat(vec![row_rot, col_rot], 2);
        }
        let device = token_emb.device();
        let freqs = self
            .grid_rope_freqs
            .clone()
            .unwrap_or_else(|| Self::build_rope_freqs(half, self.grid_rope_theta, &device))
            .reshape([1, 1, half]);
        let row_part = token_emb.clone().slice_dim(2, 0..half);
        let col_part = token_emb.clone().slice_dim(2, half..(half * 2));
        let row_rot = self.rope_with_positions(row_part, row_ids, freqs.clone());
        let col_rot = self.rope_with_positions(col_part, col_ids, freqs);
        Tensor::cat(vec![row_rot, col_rot], 2)
    }

    pub fn cell_embeddings_with_positions(
        &self,
        tokens: Tensor<B, 2, Int>,
        row_ids: Tensor<B, 2, Int>,
        col_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let token_emb = self.core.embed_tokens(tokens);
        match self.grid_positional {
            SudokuGridPositional::Additive => {
                let row_emb = self.row_embed.forward(row_ids);
                let col_emb = self.col_embed.forward(col_ids);
                token_emb + row_emb + col_emb
            }
            SudokuGridPositional::Rope2d => self.apply_rope_2d(token_emb, row_ids, col_ids),
        }
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
        let summary =
            tokens
                .reshape([1, count.max(1), dim])
                .expand([batch.max(1), count.max(1), dim]);
        self.summary_norm.forward(summary)
    }

    pub fn normalize_summary_tokens(&self, summary: Tensor<B, 3>) -> Tensor<B, 3> {
        self.summary_norm.forward(summary)
    }

    pub fn cache_streams(&self) -> usize {
        self.cache_streams.max(1)
    }
    pub fn summary_token_count(&self) -> usize {
        self.summary_token_count
    }
    pub fn ca_heads(&self) -> usize {
        self.ca_heads.max(1)
    }
    pub fn ca_head_dim(&self) -> usize {
        self.ca_head_dim.max(1)
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
                let k = k.reshape([batch, time, heads, head_dim]).swap_dims(1, 2);
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
                let keys = self.policy_pos_keys.val().reshape([1, GRID_LEN, dim]);
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

    pub fn value_baseline_from_summary_tokens(&self, summary_tokens: Tensor<B, 3>) -> Tensor<B, 2> {
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
        self.update_cell_embedding_with_gate(summary_tokens, cache_cell, token_emb)
            .0
    }

    fn fast_weight_group_update(
        q: Tensor<B, 5>,
        k: Tensor<B, 5>,
        v: Tensor<B, 5>,
        mem: Tensor<B, 5>,
        decay: f32,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, heads, groups, len, dim] = q.shape().dims();
        let device = q.device();
        if batch == 0 || heads == 0 || groups == 0 || len == 0 || dim == 0 {
            let zeros = Tensor::<B, 5>::zeros(
                [
                    batch.max(1),
                    heads.max(1),
                    groups.max(1),
                    len.max(1),
                    dim.max(1),
                ],
                &device,
            );
            return (zeros, mem);
        }
        let flat_groups = batch * heads * groups;
        let q_flat = q.reshape([flat_groups, len, dim]);
        let k_flat = k.reshape([flat_groups, len, dim]);
        let v_flat = v.reshape([flat_groups, len, dim]);
        let mem_flat = mem.reshape([flat_groups, dim, dim]);
        let k_exp = k_flat.unsqueeze_dim::<4>(3);
        let v_exp = v_flat.unsqueeze_dim::<4>(2);
        let delta = k_exp.mul(v_exp).sum_dims_squeeze::<3, usize>(&[1]);
        let mem_flat = mem_flat.mul_scalar(decay).add(delta);
        let q_exp = q_flat.unsqueeze_dim::<4>(3);
        let mem_exp = mem_flat.clone().unsqueeze_dim::<4>(1);
        let msg_flat = q_exp.mul(mem_exp).sum_dims_squeeze::<3, usize>(&[2]);
        let msg = msg_flat.reshape([batch, heads, groups, len, dim]);
        let mem = mem_flat.reshape([batch, heads, groups, dim, dim]);
        (msg, mem)
    }

    pub fn constraint_ca_step(
        &self,
        cell_state: Tensor<B, 3>,
        clue_mask: Tensor<B, 2>,
        mem_row: Tensor<B, 5>,
        mem_col: Tensor<B, 5>,
        mem_box: Tensor<B, 5>,
        decay: f32,
    ) -> (Tensor<B, 3>, Tensor<B, 5>, Tensor<B, 5>, Tensor<B, 5>) {
        let [batch, time, dim] = cell_state.shape().dims();
        if batch == 0 || time == 0 || dim == 0 || time != GRID_LEN {
            return (cell_state, mem_row, mem_col, mem_box);
        }
        let heads = self.ca_heads.max(1);
        let head_dim = self.ca_head_dim.max(1);
        if heads * head_dim != dim {
            return (cell_state, mem_row, mem_col, mem_box);
        }
        let decay = decay.clamp(0.0, 1.0);
        let flat = cell_state.clone().reshape([batch * time, dim]);
        let q = self.ca_q.forward(flat.clone()).reshape([batch, time, dim]);
        let k = self.ca_k.forward(flat.clone()).reshape([batch, time, dim]);
        let v = self.ca_v.forward(flat).reshape([batch, time, dim]);
        let q = q.reshape([batch, time, heads, head_dim]).swap_dims(1, 2);
        let k = k.reshape([batch, time, heads, head_dim]).swap_dims(1, 2);
        let v = v.reshape([batch, time, heads, head_dim]).swap_dims(1, 2);

        let q_grid = q.reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);
        let k_grid = k.reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);
        let v_grid = v.reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);

        let (msg_row, mem_row) = Self::fast_weight_group_update(
            q_grid.clone(),
            k_grid.clone(),
            v_grid.clone(),
            mem_row,
            decay,
        );

        let q_col = q_grid.clone().swap_dims(2, 3);
        let k_col = k_grid.clone().swap_dims(2, 3);
        let v_col = v_grid.clone().swap_dims(2, 3);
        let (msg_col, mem_col) =
            Self::fast_weight_group_update(q_col, k_col, v_col, mem_col, decay);
        let msg_col = msg_col.swap_dims(2, 3);

        let q_box = q_grid
            .clone()
            .reshape([batch, heads, 3, 3, 3, 3, head_dim])
            .swap_dims(3, 4)
            .reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);
        let k_box = k_grid
            .clone()
            .reshape([batch, heads, 3, 3, 3, 3, head_dim])
            .swap_dims(3, 4)
            .reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);
        let v_box = v_grid
            .clone()
            .reshape([batch, heads, 3, 3, 3, 3, head_dim])
            .swap_dims(3, 4)
            .reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);
        let (msg_box, mem_box) =
            Self::fast_weight_group_update(q_box, k_box, v_box, mem_box, decay);
        let msg_box = msg_box
            .reshape([batch, heads, 3, 3, 3, 3, head_dim])
            .swap_dims(3, 4)
            .reshape([batch, heads, GRID_SIDE, GRID_SIDE, head_dim]);

        let mut msg = msg_row + msg_col + msg_box;
        msg = msg.div_scalar(3.0);
        let msg = msg
            .reshape([batch, heads, GRID_LEN, head_dim])
            .swap_dims(1, 2)
            .reshape([batch, GRID_LEN, dim]);

        let givens = clue_mask.reshape([batch, GRID_LEN, 1]);
        let update_input = Tensor::cat(vec![cell_state.clone(), msg, givens], 2);
        let update_flat = update_input.reshape([batch * GRID_LEN, dim * 2 + 1]);
        let hidden = activation::gelu(self.ca_update_fc1.forward(update_flat));
        let delta = self
            .ca_update_fc2
            .forward(hidden)
            .reshape([batch, GRID_LEN, dim]);
        let next = self.ca_norm.forward(cell_state + delta);
        (next, mem_row, mem_col, mem_box)
    }

    pub fn update_cell_embedding_with_gate(
        &self,
        summary_tokens: Tensor<B, 3>,
        cache_cell: Tensor<B, 3>,
        token_emb: Tensor<B, 3>,
    ) -> (Tensor<B, 3>, Tensor<B, 2>) {
        let [batch, _, dim] = summary_tokens.shape().dims();
        let device = summary_tokens.device();
        if batch == 0 || dim == 0 {
            let zeros = Tensor::<B, 3>::zeros([batch.max(1), 1, dim.max(1)], &device);
            let write_gate = Tensor::<B, 2>::zeros([batch.max(1), 1], &device);
            return (zeros, write_gate);
        }
        let summary = self
            .summary_norm
            .forward(summary_tokens)
            .mean_dim(1)
            .reshape([batch, 1, dim]);
        let stacked = Tensor::cat(vec![summary, cache_cell.clone(), token_emb], 2);
        let flat = stacked.reshape([batch, dim * 3]);
        let hidden = activation::gelu(self.update_mlp_fc1.forward(flat));
        let updated = self.update_mlp_fc2.forward(hidden);
        let [_, out_dim] = updated.shape().dims();
        let updated = updated.reshape([batch, 1, out_dim]);
        let (updated, write_gate) = if out_dim > dim {
            let split = out_dim / 2;
            let delta = updated.clone().slice_dim(2, 0..split);
            let gate = activation::sigmoid(updated.slice_dim(2, split..out_dim));
            let write_gate = gate.clone().mean_dim(2).reshape([batch, 1]);
            (cache_cell + delta.mul(gate), write_gate)
        } else {
            let write_gate = Tensor::<B, 2>::ones([batch.max(1), 1], &device);
            (updated, write_gate)
        };
        (self.cache_norm.forward(updated), write_gate)
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

    pub fn forward_with_hidden(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 3>) {
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
        self.core
            .forward_with_hidden_and_state_embedded(embedded, state)
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
    use crate::config::{SudokuCacheMhcConfig, SudokuCacheUpdateConfig};
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_dragon_train::WgpuRuntimeConfig;
    use burn_ndarray::NdArray;

    #[test]
    fn policy_head_variants_output_grid_logits() {
        type B = NdArray<f32>;
        let device = <B as BackendTrait>::Device::default();
        let heads = [SudokuPolicyHead::SummaryPos, SudokuPolicyHead::SummaryMlp];

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
                rotary_embedding: Default::default(),
                grid_positional: SudokuGridPositional::Additive,
                grid_rope_theta: 65_536.0,
                dropout: 0.0,
                fused_kernels: false,
                relu_threshold: 0.0,
                cache_mhc: SudokuCacheMhcConfig::default(),
                cache_update: SudokuCacheUpdateConfig::default(),
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

    #[test]
    fn wgpu_backend_override_enables_fused_core_contract() {
        let config = SudokuModelConfig {
            fused_kernels: false,
            ..SudokuModelConfig::default()
        };
        let mut wgpu = WgpuRuntimeConfig::default();
        wgpu.training.fused_core_recurrent = Some(true);
        wgpu.training.fused_core_rollout = None;

        let resolved = config.to_bdh_config_for_backend("wgpu", &wgpu);

        assert!(resolved.fused_kernels.enabled);
        assert!(resolved.fused_kernels.wgpu_recurrent_kernel);
        assert!(resolved.fused_kernels.wgpu_rollout_fused);
    }

    #[test]
    fn sudoku_model_config_exposes_paper_dimension_aliases() {
        let config = SudokuModelConfig {
            n_layer: 1,
            n_embd: 48,
            n_head: 3,
            mlp_internal_dim_multiplier: 4,
            ..SudokuModelConfig::default()
        };

        assert_eq!(config.dense_space_dim(), 48);
        assert_eq!(config.neuron_space_dim(), 192);
        assert_eq!(config.neuron_space_dim_per_head(), 64);
    }
}
