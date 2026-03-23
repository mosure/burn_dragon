use burn::module::{Ignored, Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use burn_dragon_kernel::api::attention::{
    supports_dense_causal_attention_backend, try_fused_dense_causal_attention_wgpu,
};
use burn_dragon_kernel::api::recurrent::{
    CompiledRecurrentAttentionPlan, supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
    try_fused_recurrent_attention_wgpu_with_plan,
};
use burn_dragon_kernel::kernels::sequence::mamba::selective_scan_forward::{
    MambaTensorizedState, tensorized_mamba_forward, use_tensorized_mamba_forward_experimental,
};
use burn_dragon_kernel::kernels::sequence::rwkv8::forward::{
    tensorized_rwkv8_forward, use_tensorized_rwkv8_forward_experimental,
};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use serde::Serialize;
use std::cmp::Ordering;
use std::ops::Range;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use super::attention::Attention;
use super::attention_residual::{AttentionResidual, BlockAttentionResidual, ResidualConnectorKind};
use super::config::{
    BDHConfig, ClockedSlowMemoryConfig, FusedKernelConfig, SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};
use super::init::{
    near_critical_embedding_initializer, near_critical_projection_std,
    near_critical_residual_output_std,
};
use super::norm::DragonNorm;
use super::residual_stream::{lowrank_residual_step, lowrank_residual_step_next};
use super::sequence::linear::{
    recurrent_attention_dense_score_final_rho_reference,
    recurrent_attention_dense_score_initial_context_reference,
    recurrent_attention_dense_score_reference, recurrent_attention_reference,
};
use super::sequence::mamba::{
    MambaReferenceState, MambaSequenceParameters, ResolvedMambaSequenceConfig, mamba_reference,
};
use super::sequence::rwkv8::recurrent_rwkv8_state_space_reference;
use super::sequence::state::{
    linear_attention_state, mamba_state, rwkv8_state, write_linear_attention_state,
    write_mamba_state, write_rwkv8_state,
};
use super::sequence::{SequenceKernelConfig, SequenceKernelFamily, SequenceTrainingExecutor};
#[cfg(feature = "viz")]
use super::state::LayerVizState;
use super::state::{LayerState, ModelState};
use super::{ManifoldHyperConnections, mhc_merge_with_coefficients, mhc_split_with_coefficients};
#[cfg(test)]
use crate::model::config::SequenceKernelKind;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecurrentPositionMode {
    Sequential,
    Fixed,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RolloutExecutorMode {
    HostLoop,
    WgpuFused,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct LogitsProjectionProfileSnapshot {
    pub calls: u64,
    pub total_ns: u128,
}

#[derive(Default)]
struct LogitsProjectionProfileState {
    calls: u64,
    total_ns: u128,
}

static LOGITS_PROJECTION_PROFILE: OnceLock<Mutex<LogitsProjectionProfileState>> = OnceLock::new();

fn logits_projection_profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

pub fn logits_projection_profile_reset() {
    if let Ok(mut state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        *state = LogitsProjectionProfileState::default();
    }
}

pub fn logits_projection_profile_snapshot() -> LogitsProjectionProfileSnapshot {
    if let Ok(state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        return LogitsProjectionProfileSnapshot {
            calls: state.calls,
            total_ns: state.total_ns,
        };
    }
    LogitsProjectionProfileSnapshot::default()
}

fn logits_projection_profile_record(elapsed_ns: u128) {
    if let Ok(mut state) = LOGITS_PROJECTION_PROFILE
        .get_or_init(|| Mutex::new(LogitsProjectionProfileState::default()))
        .lock()
    {
        state.calls = state.calls.saturating_add(1);
        state.total_ns = state.total_ns.saturating_add(elapsed_ns);
    }
}

struct LanguageMhcLayerBindings<B: Backend> {
    branch_input: Tensor<B, 4>,
    residuals_base: Tensor<B, 4>,
    legacy_beta: Option<Tensor<B, 2>>,
    stream_coefficients: Option<super::ManifoldHyperConnectionStreamCoefficients<B>>,
}

#[derive(Clone, Debug)]
pub struct LanguagePipelineState<B: Backend> {
    current: Tensor<B, 4>,
    residual_history: Vec<Tensor<B, 4>>,
}

impl<B: Backend> LanguagePipelineState<B> {
    pub fn from_parts(current: Tensor<B, 4>, residual_history: Vec<Tensor<B, 4>>) -> Self {
        Self {
            current,
            residual_history,
        }
    }

    pub fn into_parts(self) -> (Tensor<B, 4>, Vec<Tensor<B, 4>>) {
        (self.current, self.residual_history)
    }

    pub fn current(&self) -> &Tensor<B, 4> {
        &self.current
    }

    pub fn residual_history(&self) -> &[Tensor<B, 4>] {
        &self.residual_history
    }
}

#[derive(Clone, Copy)]
enum ResidualConnectorRef<'a, B: Backend> {
    Vanilla,
    Mhc(&'a ManifoldHyperConnections<B>),
    AttentionResidual(&'a AttentionResidual<B>),
    BlockAttentionResidual(&'a BlockAttentionResidual<B>),
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct LanguageMhcLayerDiagnostics {
    pub layer_index: usize,
    pub num_streams: usize,
    pub stream_norm_mean: f64,
    pub stream_norm_variance: f64,
    pub pairwise_stream_cosine_mean: Option<f64>,
    pub alpha_entropy_mean: f64,
    pub alpha_entropy_normalized_mean: f64,
    pub beta_entropy_mean: Option<f64>,
    pub beta_entropy_normalized_mean: Option<f64>,
    pub residual_distance_identity_l1_mean: f64,
    pub residual_distance_uniform_l1_mean: f64,
}

#[derive(Clone, Debug, Default)]
struct LanguageMhcLayerDiagnosticsAccumulator {
    count: usize,
    num_streams: usize,
    stream_norm_mean_sum: f64,
    stream_norm_variance_sum: f64,
    pairwise_stream_cosine_mean_sum: f64,
    pairwise_stream_cosine_mean_count: usize,
    alpha_entropy_mean_sum: f64,
    alpha_entropy_normalized_mean_sum: f64,
    beta_entropy_mean_sum: f64,
    beta_entropy_mean_count: usize,
    beta_entropy_normalized_mean_sum: f64,
    beta_entropy_normalized_mean_count: usize,
    residual_distance_identity_l1_mean_sum: f64,
    residual_distance_uniform_l1_mean_sum: f64,
}

#[derive(Module, Debug)]
pub struct BDH<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    sequence_kernel: SequenceKernelConfig,
    rollout_fast_steps_per_slow_step: usize,
    kernel: FusedKernelConfig,
    y_neuron_recurrence: YNeuronRecurrenceConfig,
    clocked_slow_memory: ClockedSlowMemoryConfig,
    summary_memory: SummaryMemoryConfig,
    layer_latent_totals: Ignored<Vec<usize>>,
    embed: Embedding<B>,
    dropout: Dropout,
    norm: DragonNorm<B>,
    attention: Attention<B>,
    residual_connector: ResidualConnectorKind,
    mhc_first_layer: usize,
    mhc_shared: Option<ManifoldHyperConnections<B>>,
    attention_residual_first_layer: usize,
    attention_residual_shared: Option<AttentionResidual<B>>,
    block_attention_residual_first_layer: usize,
    block_attention_residual_shared: Option<BlockAttentionResidual<B>>,
    rwkv_time_decay: Param<Tensor<B, 2>>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
    mamba_config: Ignored<ResolvedMambaSequenceConfig>,
    mamba: Option<MambaSequenceParameters<B>>,
    lm_head: Param<Tensor<B, 2>>,
}

impl<B: Backend> BDH<B> {
    pub fn new(config: BDHConfig, device: &B::Device) -> Self {
        let embed = EmbeddingConfig::new(config.vocab_size, config.n_embd)
            .with_initializer(near_critical_embedding_initializer(config.n_embd))
            .init(device);
        let dropout = DropoutConfig::new(config.dropout).init();
        let norm = DragonNorm::new(&config.normalization, config.n_embd, device);

        let latent_per_head = config.latent_per_head();
        let latent_total = config.latent_total();
        let attention = Attention::new(
            latent_per_head,
            config.n_head,
            device,
            &config.fused_kernels,
        );
        let rwkv_time_decay = Param::from_tensor(Self::init_rwkv_time_decay(
            config.n_head,
            latent_per_head,
            device,
        ));
        let residual_depth = config.n_layer.max(1) * config.rollout_fast_steps_per_slow_step.max(1);
        let encoder_std =
            near_critical_residual_output_std(config.n_embd, latent_per_head, residual_depth);
        let decoder_std =
            near_critical_residual_output_std(latent_total, config.n_embd, residual_depth);
        let lm_head_std = near_critical_projection_std(config.n_embd, config.vocab_size);

        let encoder = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, encoder_std),
            device,
        ));

        let encoder_v = Param::from_tensor(Tensor::<B, 3>::random(
            [config.n_head, config.n_embd, latent_per_head],
            TensorDistribution::Normal(0.0, encoder_std),
            device,
        ));

        let decoder = Param::from_tensor(Tensor::<B, 2>::random(
            [latent_total, config.n_embd],
            TensorDistribution::Normal(0.0, decoder_std),
            device,
        ));
        let residual_connector = config.resolved_residual_connector_kind();
        let mhc_first_layer = config
            .mhc
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let mhc_shared = if residual_connector == ResidualConnectorKind::Mhc
            && config.mhc.enabled
            && (config.mhc.resolved_num_streams() > 1 || config.mhc.resolved_num_views() > 1)
        {
            Some(ManifoldHyperConnections::new_with_dense_dim(
                &config.mhc,
                mhc_first_layer,
                Some(config.n_embd),
                device,
            ))
        } else {
            None
        };
        let attention_residual_first_layer = config
            .attention_residual
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let attention_residual_shared = (residual_connector
            == ResidualConnectorKind::AttentionResidual
            && config.attention_residual.enabled)
            .then(|| AttentionResidual::new(&config.attention_residual, config.n_embd, device));
        let block_attention_residual_first_layer = config
            .block_attention_residual
            .last_layers
            .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
            .unwrap_or(0);
        let block_attention_residual_shared = (residual_connector
            == ResidualConnectorKind::BlockAttentionResidual
            && config.block_attention_residual.enabled)
            .then(|| {
                BlockAttentionResidual::new(&config.block_attention_residual, config.n_embd, device)
            });
        let sequence_kernel = config.resolved_sequence_kernel_config();
        let mamba_config = config.mamba.resolve(config.n_embd);
        let mamba = (sequence_kernel.family == SequenceKernelFamily::Mamba1SelectiveSsm)
            .then(|| MambaSequenceParameters::new(mamba_config, device));
        let lm_head = Param::from_tensor(Tensor::<B, 2>::random(
            [config.n_embd, config.vocab_size],
            TensorDistribution::Normal(0.0, lm_head_std),
            device,
        ));
        let layer_latent_totals = Ignored(
            (0..config.n_layer)
                .map(|layer_idx| config.latent_total_for_layer(layer_idx))
                .collect(),
        );

        Self {
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_head: config.n_head,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            vocab_size: config.vocab_size,
            sequence_kernel,
            rollout_fast_steps_per_slow_step: config.rollout_fast_steps_per_slow_step,
            kernel: config.fused_kernels,
            y_neuron_recurrence: config.y_neuron_recurrence,
            clocked_slow_memory: config.clocked_slow_memory,
            summary_memory: config.summary_memory,
            layer_latent_totals,
            embed,
            dropout,
            norm,
            attention,
            residual_connector,
            mhc_first_layer,
            mhc_shared,
            attention_residual_first_layer,
            attention_residual_shared,
            block_attention_residual_first_layer,
            block_attention_residual_shared,
            rwkv_time_decay,
            encoder,
            encoder_v,
            decoder,
            mamba_config: Ignored(mamba_config),
            mamba,
            lm_head,
        }
    }

    fn init_rwkv_time_decay(
        n_head: usize,
        latent_per_head: usize,
        device: &B::Device,
    ) -> Tensor<B, 2> {
        let mut values = Vec::with_capacity(n_head * latent_per_head);
        for head_idx in 0..n_head {
            let head_ratio = if n_head <= 1 {
                0.0
            } else {
                head_idx as f32 / (n_head - 1) as f32
            };
            for latent_idx in 0..latent_per_head {
                let latent_ratio = if latent_per_head <= 1 {
                    0.0
                } else {
                    latent_idx as f32 / (latent_per_head - 1) as f32
                };
                let target_decay =
                    (0.995 - 0.22 * latent_ratio - 0.03 * head_ratio).clamp(0.55, 0.995);
                let raw = (target_decay / (1.0 - target_decay)).ln();
                values.push(raw);
            }
        }
        Tensor::<B, 1>::from_floats(values.as_slice(), device).reshape([n_head, latent_per_head])
    }

    fn rwkv_decay(&self, latent: usize) -> Tensor<B, 3> {
        activation::sigmoid(
            self.rwkv_time_decay
                .val()
                .slice([0..self.n_head, 0..latent])
                .reshape([1, self.n_head, latent]),
        )
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_with_state(tokens, &mut state)
    }

    pub fn forward_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_with_state_and_summary_event_mask(tokens, summary_event_mask, &mut state)
    }

    pub fn forward_with_hidden(&self, tokens: Tensor<B, 2, Int>) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let mut state = self.init_state();
        self.forward_with_hidden_and_state(tokens, &mut state)
    }

    pub fn embed_tokens(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.embed.forward(tokens)
    }

    pub fn begin_language_pipeline_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
    ) -> LanguagePipelineState<B> {
        assert_eq!(
            self.rollout_fast_steps_per_slow_step, 1,
            "language pipeline execution currently requires rollout_fast_steps_per_slow_step = 1"
        );
        assert!(
            !self.y_neuron_recurrence.enabled,
            "language pipeline execution is not supported with y-neuron recurrence enabled"
        );
        self.initialize_language_pipeline_state(embedded)
    }

    pub fn begin_language_pipeline(&self, tokens: Tensor<B, 2, Int>) -> LanguagePipelineState<B> {
        self.begin_language_pipeline_from_embedded(self.embed.forward(tokens))
    }

    pub fn forward_language_pipeline_stage_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
        layer_range: Range<usize>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> LanguagePipelineState<B> {
        self.forward_language_pipeline_state_layer_range(
            pipeline_state,
            state,
            state.position,
            RecurrentPositionMode::Sequential,
            summary_event_mask,
            layer_range,
        )
    }

    pub fn finish_language_pipeline_hidden_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let hidden = self.collapse_language_streams(pipeline_state.current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        state.position = state.position.saturating_add(time);
        hidden
    }

    pub fn finish_language_pipeline_with_state(
        &self,
        pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = self.finish_language_pipeline_hidden_with_state(pipeline_state, state);
        let logits = self.project_hidden_to_logits(hidden.clone());
        (hidden, logits)
    }

    pub fn rollout_fast_steps_per_slow_step(&self) -> usize {
        self.rollout_fast_steps_per_slow_step
    }

    pub fn forward_fast(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        self.forward(tokens)
    }

    pub fn forward_fast_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        self.forward_with_summary_event_mask(tokens, summary_event_mask)
    }

    pub fn collect_language_mhc_diagnostics(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        let mut state = self.init_state();
        self.collect_language_mhc_diagnostics_with_state(tokens, &mut state)
    }

    pub fn collect_language_mhc_diagnostics_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        let mut state = self.init_state();
        self.collect_language_mhc_diagnostics_with_state_and_summary_event_mask(
            tokens,
            summary_event_mask,
            &mut state,
        )
    }

    pub fn generate(
        &self,
        mut indices: Tensor<B, 2, Int>,
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
    ) -> Tensor<B, 2, Int> {
        let [batch, _] = indices.shape().dims();
        assert_eq!(batch, 1, "generation currently supports batch size 1");

        let mut state = self.init_state();
        let mut logits = self.forward_with_state(indices.clone(), &mut state);
        let [_, mut time, vocab] = logits.shape().dims();
        assert_eq!(time, indices.shape().dims::<2>()[1]);

        let mut last_logits = logits
            .slice_dim(1, (time - 1)..time)
            .reshape([vocab])
            .div_scalar(temperature);

        for _ in 0..max_new_tokens {
            let mut logits_values = last_logits
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("logits to vec");

            if let Some(k) = top_k
                && k > 0
                && k < vocab
            {
                let mut sorted = logits_values.clone();
                sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
                let threshold = sorted[k - 1];
                for value in logits_values.iter_mut() {
                    if *value < threshold {
                        *value = f32::NEG_INFINITY;
                    }
                }
            }

            let max_logit = logits_values
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut probs: Vec<f32> = logits_values
                .iter()
                .map(|value| (value - max_logit).exp())
                .collect();
            let sum: f32 = probs.iter().sum();
            if sum == 0.0 || sum.is_nan() {
                let uniform = 1.0 / vocab as f32;
                for p in probs.iter_mut() {
                    *p = uniform;
                }
            } else {
                for p in probs.iter_mut() {
                    *p /= sum;
                }
            }

            let dist = WeightedIndex::new(&probs).expect("valid probability distribution");
            let mut rng = thread_rng();
            let next = dist.sample(&mut rng) as i64;

            let next_token = Tensor::<B, 2, Int>::from_data(
                TensorData::new(vec![next], [1, 1]),
                &indices.device(),
            );
            indices = Tensor::cat(vec![indices, next_token.clone()], 1);

            logits = self.forward_with_state(next_token, &mut state);
            let [_, new_time, _] = logits.shape().dims();
            time = new_time;
            last_logits = logits
                .slice_dim(1, (time - 1)..time)
                .reshape([vocab])
                .div_scalar(temperature);
        }

        indices
    }

    pub fn init_state(&self) -> ModelState<B> {
        ModelState::new(self.n_layer)
    }

    fn layer_latent_total(&self, layer_idx: usize) -> usize {
        self.layer_latent_totals
            .0
            .get(layer_idx)
            .copied()
            .unwrap_or(self.mlp_internal_dim_multiplier * self.n_embd)
    }

    fn layer_latent_per_head(&self, layer_idx: usize) -> usize {
        let total = self.layer_latent_total(layer_idx);
        assert_eq!(
            total % self.n_head,
            0,
            "layer latent total must divide evenly across heads"
        );
        total / self.n_head
    }

    fn layer_lowrank_weights(
        &self,
        layer_idx: usize,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 2>, usize) {
        let latent_per_head = self.layer_latent_per_head(layer_idx);
        let latent_total = self.layer_latent_total(layer_idx);
        let encoder = self
            .encoder
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let encoder_v = self
            .encoder_v
            .val()
            .slice([0..self.n_head, 0..self.n_embd, 0..latent_per_head])
            .reshape([1, self.n_head, self.n_embd, latent_per_head]);
        let decoder = self.decoder.val().slice([0..latent_total, 0..self.n_embd]);
        (encoder, encoder_v, decoder, latent_per_head)
    }

    fn rollout_executor_mode(&self) -> RolloutExecutorMode {
        if self.sequence_kernel.family == SequenceKernelFamily::LinearAttention
            && self.sequence_kernel.executor == SequenceTrainingExecutor::Reference
            && self.kernel.enabled
            && self.kernel.wgpu_recurrent_kernel
            && self.kernel.wgpu_rollout_fused
            && supports_recurrent_backend::<B>()
        {
            return RolloutExecutorMode::WgpuFused;
        }
        RolloutExecutorMode::HostLoop
    }

    fn recurrent_attention_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_reference(query, value, rho_state, decay)
    }

    fn recurrent_attention_dense_score_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_dense_score_reference(query, value, rho_state, decay)
    }

    fn recurrent_attention_dense_score_final_rho_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_final_rho_reference(query, value, rho_state, decay)
    }

    fn recurrent_attention_dense_score_initial_context_reference(
        &self,
        query: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
        n_embd: usize,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_initial_context_reference(query, rho_state, decay, n_embd)
    }

    fn recurrent_rwkv8_state_space_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        rho_norm_state: Option<Tensor<B, 3>>,
        decay: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
        recurrent_rwkv8_state_space_reference(query, value, rho_state, rho_norm_state, decay)
    }

    fn project_lowrank_positive(
        &self,
        dense: Tensor<B, 4>,
        projector: Tensor<B, 4>,
        use_fused: bool,
        latent_pattern: &crate::kernel::BlockPattern1d,
        sparse_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        if use_fused {
            crate::kernel::relu_lowrank::fused_forward_with_executor(
                dense,
                projector,
                None,
                self.kernel.relu_threshold,
                latent_pattern,
                sparse_mask,
                self.kernel.lowrank_grad_input_executor,
            )
        } else {
            let mut latent = dense.matmul(projector);
            if self.kernel.relu_threshold != 0.0 {
                latent = latent.sub_scalar(self.kernel.relu_threshold);
            }
            activation::relu(latent)
        }
    }

    fn resolve_y_neuron_state(
        &self,
        layer_state: &LayerState<B>,
        batch: usize,
        heads: usize,
        latent: usize,
        device: &B::Device,
    ) -> Tensor<B, 3> {
        match layer_state.y_neuron_state.as_ref() {
            Some(state) if state.shape().dims::<3>() == [batch, heads, latent] => {
                self.stabilize_y_neuron_state(state.clone())
            }
            _ => Tensor::<B, 3>::zeros([batch, heads, latent], device),
        }
    }

    fn y_neuron_recurrence_applies_to_layer(&self, layer_idx: usize) -> bool {
        if !self.y_neuron_recurrence.enabled {
            return false;
        }
        match self.y_neuron_recurrence.last_layers {
            Some(last_layers) => {
                if last_layers == 0 {
                    return false;
                }
                let first_recurrent_layer = self.n_layer.saturating_sub(last_layers);
                layer_idx >= first_recurrent_layer
            }
            None => true,
        }
    }

    fn clocked_slow_memory_applies_to_layer(&self, layer_idx: usize) -> bool {
        if !self.clocked_slow_memory.enabled {
            return false;
        }
        match self.clocked_slow_memory.last_layers {
            Some(last_layers) => {
                if last_layers == 0 {
                    return false;
                }
                let first_slow_layer = self.n_layer.saturating_sub(last_layers);
                layer_idx >= first_slow_layer
            }
            None => true,
        }
    }

    fn summary_memory_applies_to_layer(&self, layer_idx: usize) -> bool {
        if !self.summary_memory.enabled {
            return false;
        }
        match self.summary_memory.last_layers {
            Some(last_layers) => {
                if last_layers == 0 {
                    return false;
                }
                let first_summary_layer = self.n_layer.saturating_sub(last_layers);
                layer_idx >= first_summary_layer
            }
            None => true,
        }
    }

    fn summary_memory_uses_write_trigger(&self) -> bool {
        self.summary_memory
            .write_trigger_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
    }

    fn resolve_clocked_slow_hidden(
        &self,
        layer_state: &LayerState<B>,
        batch: usize,
        views: usize,
        dim: usize,
    ) -> Option<Tensor<B, 4>> {
        match layer_state.clocked_slow_hidden.as_ref() {
            Some(hidden) if hidden.shape().dims::<4>() == [batch, views, 1, dim] => {
                Some(hidden.clone())
            }
            _ => None,
        }
    }

    fn resolve_summary_memory_hidden(
        &self,
        layer_state: &LayerState<B>,
        batch: usize,
        views: usize,
        dim: usize,
    ) -> Option<Tensor<B, 4>> {
        match layer_state.summary_memory_hidden.as_ref() {
            Some(hidden) if hidden.shape().dims::<4>() == [batch, views, 1, dim] => {
                Some(hidden.clone())
            }
            _ => None,
        }
    }

    fn stabilize_y_neuron_state(&self, y_neuron_state: Tensor<B, 3>) -> Tensor<B, 3> {
        let Some(state_rms_cap) = self.y_neuron_recurrence.state_rms_cap else {
            return y_neuron_state;
        };
        let rms = y_neuron_state
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .clamp_min(1.0e-6);
        let scale = rms.div_scalar(state_rms_cap).clamp_min(1.0);
        y_neuron_state.div(scale)
    }

    fn inject_y_neuron_state(
        &self,
        x_neuron: Tensor<B, 4>,
        y_neuron_state: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        if self.y_neuron_recurrence.carry_in_scale == 0.0 {
            return x_neuron;
        }
        x_neuron
            + y_neuron_state
                .unsqueeze_dim::<4>(2)
                .mul_scalar(self.y_neuron_recurrence.carry_in_scale)
    }

    fn update_y_neuron_state(
        &self,
        previous_state: Tensor<B, 3>,
        y_neuron: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
        debug_assert_eq!(time, 1, "token-wise y_neuron recurrence expects time=1");
        let current_state = y_neuron.reshape([batch, heads, latent]);
        let next_state = previous_state
            .mul_scalar(self.y_neuron_recurrence.state_decay)
            .add(current_state.mul_scalar(self.y_neuron_recurrence.state_update_scale));
        self.stabilize_y_neuron_state(next_state)
    }

    fn recurrent_attention_with_plan(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        position: usize,
        position_mode: RecurrentPositionMode,
        fused_plan: Option<&CompiledRecurrentAttentionPlan<B>>,
    ) -> Tensor<B, 4> {
        match (self.sequence_kernel.family, self.sequence_kernel.executor) {
            (SequenceKernelFamily::LinearAttention, SequenceTrainingExecutor::Reference) => {
                let query = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(query, position)
                    }
                    RecurrentPositionMode::Fixed => {
                        self.attention.rotate_positions_fixed(query, position)
                    }
                };
                let decay = self.attention.alibi_decay();
                let initial_rho = linear_attention_state(layer_state).rho;
                let device = query.device();

                if self.kernel.enabled && self.kernel.wgpu_recurrent_kernel {
                    let fused = if let Some(plan) = fused_plan {
                        try_fused_recurrent_attention_wgpu_with_plan(
                            &query,
                            &value,
                            initial_rho.as_ref(),
                            decay.as_ref(),
                            plan,
                        )
                    } else {
                        try_fused_recurrent_attention_wgpu(
                            &query,
                            &value,
                            initial_rho.as_ref(),
                            decay.as_ref(),
                        )
                    };
                    if let Some(output) = fused {
                        if B::ad_enabled(&device) {
                            let (reference_context, reference_rho) = self
                                .recurrent_attention_reference(
                                    query.clone(),
                                    value.clone(),
                                    initial_rho,
                                    decay,
                                );
                            let context = reference_context.clone() + output.context
                                - reference_context.detach();
                            let rho = reference_rho.clone() + output.rho - reference_rho.detach();
                            write_linear_attention_state(layer_state, rho);
                            return context;
                        }
                        write_linear_attention_state(layer_state, output.rho);
                        return output.context;
                    }
                }

                let (context, rho) =
                    self.recurrent_attention_reference(query, value, initial_rho, decay);
                write_linear_attention_state(layer_state, rho);
                context
            }
            (
                SequenceKernelFamily::LinearAttention,
                SequenceTrainingExecutor::DenseScoreShortContext,
            ) => {
                let query = match position_mode {
                    RecurrentPositionMode::Sequential => {
                        self.attention.rotate_positions(query, position)
                    }
                    RecurrentPositionMode::Fixed => {
                        self.attention.rotate_positions_fixed(query, position)
                    }
                };
                let decay = self.attention.alibi_decay();
                let initial_rho = linear_attention_state(layer_state).rho;
                let device = query.device();
                if self.kernel.enabled
                    && self.kernel.wgpu_rollout_fused
                    && supports_dense_causal_attention_backend::<B>()
                {
                    let decay_tensor = decay
                        .clone()
                        .unwrap_or_else(|| Tensor::<B, 1>::ones([self.n_head], &device));
                    if let Some(fused_context) =
                        try_fused_dense_causal_attention_wgpu(&query, &value, &decay_tensor)
                    {
                        let initial_context = self
                            .recurrent_attention_dense_score_initial_context_reference(
                                query.clone(),
                                initial_rho.clone(),
                                decay.clone(),
                                value.shape().dims::<4>()[3],
                            );
                        let rho = self.recurrent_attention_dense_score_final_rho_reference(
                            query.clone(),
                            value.clone(),
                            initial_rho.clone(),
                            decay.clone(),
                        );
                        if B::ad_enabled(&device) {
                            let (reference_context, reference_rho) = self
                                .recurrent_attention_dense_score_reference(
                                    query.clone(),
                                    value.clone(),
                                    initial_rho,
                                    decay,
                                );
                            let context = reference_context.clone()
                                + (initial_context.clone() + fused_context)
                                - reference_context.detach();
                            write_linear_attention_state(
                                layer_state,
                                reference_rho.clone() + rho - reference_rho.detach(),
                            );
                            return context;
                        }
                        write_linear_attention_state(layer_state, rho);
                        return initial_context + fused_context;
                    }
                }
                let (context, rho) = self.recurrent_attention_dense_score_reference(
                    query,
                    value,
                    initial_rho,
                    decay,
                );
                write_linear_attention_state(layer_state, rho);
                context
            }
            (SequenceKernelFamily::Rwkv8, SequenceTrainingExecutor::Reference) => {
                let [batch, heads, _time, latent] = query.shape().dims::<4>();
                let device = query.device();
                let initial_state = rwkv8_state(layer_state, batch, heads, latent, &device);
                let decay = self.rwkv_decay(latent);
                if self.kernel.enabled && use_tensorized_rwkv8_forward_experimental() {
                    let output = tensorized_rwkv8_forward(
                        query,
                        value,
                        initial_state.rho,
                        Some(initial_state.rho_norm),
                        decay,
                    );
                    write_rwkv8_state(layer_state, output.rho, output.rho_norm);
                    return output.context;
                }
                let (context, rho, rho_norm) = self.recurrent_rwkv8_state_space_reference(
                    query,
                    value,
                    initial_state.rho,
                    Some(initial_state.rho_norm),
                    decay,
                );
                write_rwkv8_state(layer_state, rho, rho_norm);
                context
            }
            (SequenceKernelFamily::Mamba1SelectiveSsm, SequenceTrainingExecutor::Reference) => {
                let params = self
                    .mamba
                    .as_ref()
                    .expect("mamba sequence family requires initialized mamba params");
                let [batch, views, _time, dim] = value.shape().dims::<4>();
                assert_eq!(
                    views, 1,
                    "Mamba sequence family expects a single dense stream view"
                );
                assert_eq!(
                    dim, self.n_embd,
                    "Mamba dense stream dim {} must match model dim {}",
                    dim, self.n_embd
                );
                let config = self.mamba_config.0;
                let device = value.device();
                let initial_state = mamba_state(
                    layer_state,
                    batch,
                    config.d_inner,
                    config.d_state,
                    config.d_conv,
                    &device,
                );
                if self.kernel.enabled
                    && config.use_fast_path
                    && use_tensorized_mamba_forward_experimental()
                {
                    let output = tensorized_mamba_forward(
                        value,
                        config.d_inner,
                        config.d_state,
                        config.d_conv,
                        config.dt_rank,
                        params.in_proj_tensor(),
                        params.conv_weight_tensor(),
                        params.conv_bias_tensor(),
                        params.x_proj_tensor(),
                        params.dt_proj_weight_tensor(),
                        params.dt_proj_bias_tensor(),
                        params.a_log_tensor(),
                        params.d_skip_tensor(),
                        params.out_proj_tensor(),
                        Some(MambaTensorizedState {
                            conv: initial_state.conv,
                            ssm: initial_state.ssm,
                        }),
                    );
                    write_mamba_state(layer_state, output.state.ssm, output.state.conv);
                    return output.context;
                }
                let (context, next_state) = mamba_reference(
                    value,
                    params,
                    Some(MambaReferenceState {
                        conv: initial_state.conv,
                        ssm: initial_state.ssm,
                    }),
                );
                write_mamba_state(layer_state, next_state.ssm, next_state.conv);
                context
            }
            (family, executor) => panic!(
                "sequence kernel family {:?} with executor {:?} is not implemented in BDH yet",
                family, executor
            ),
        }
    }

    fn forward_branch_clocked_slow_layer(
        &self,
        layer_idx: usize,
        branch_input: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        start_pos: usize,
        position_mode: RecurrentPositionMode,
    ) -> Tensor<B, 4> {
        let [branch_batch, branch_views, branch_time, branch_dim] =
            branch_input.shape().dims::<4>();
        if branch_time == 0 {
            layer_state.clocked_slow_hidden = None;
            return branch_input;
        }

        let chunk_tokens = self
            .clocked_slow_memory
            .chunk_tokens
            .max(1)
            .min(branch_time.max(1));

        if branch_time == 1
            && chunk_tokens > 1
            && start_pos % chunk_tokens != 0
            && let Some(cached) = self.resolve_clocked_slow_hidden(
                layer_state,
                branch_batch,
                branch_views,
                branch_dim,
            )
        {
            return branch_input + cached.mul_scalar(self.clocked_slow_memory.residual_scale);
        }

        let flat_batch = branch_batch * branch_views;
        let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
        let fused = self.kernel.enabled;
        let fused_x = fused && self.kernel.projection_executor.use_x();
        let fused_y = fused && self.kernel.projection_executor.use_y();
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if (fused_x || fused_y) && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &branch_input.device()))
        } else {
            None
        };

        let mut next_chunks = Vec::with_capacity(branch_time.div_ceil(chunk_tokens));
        let mut last_slow_hidden = None;
        for (slow_idx, chunk_start) in (0..branch_time).step_by(chunk_tokens).enumerate() {
            let chunk_end = (chunk_start + chunk_tokens).min(branch_time);
            let chunk_len = chunk_end - chunk_start;
            let chunk = branch_input.clone().slice_dim(2, chunk_start..chunk_end);
            let summary = chunk.clone().mean_dim(2);
            let summary_flat = summary.reshape([flat_batch, 1, 1, branch_dim]);
            let slow_pos = match position_mode {
                RecurrentPositionMode::Sequential => start_pos / chunk_tokens + slow_idx,
                RecurrentPositionMode::Fixed => start_pos / chunk_tokens,
            };
            let next = lowrank_residual_step_next(
                summary_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused_x,
                fused_y,
                self.kernel.relu_threshold,
                true,
                latent_pattern,
                self.kernel.lowrank_grad_input_executor,
                sparse_mask.clone(),
                |query, value| {
                    self.recurrent_attention_with_plan(
                        query,
                        value,
                        layer_state,
                        slow_pos,
                        position_mode,
                        None,
                    )
                },
                activation::relu,
                |values| self.norm.forward(values),
            );
            let slow_hidden = next.reshape([branch_batch, branch_views, 1, branch_dim]);
            last_slow_hidden = Some(slow_hidden.clone());
            let broadcast = slow_hidden.repeat_dim(2, chunk_len);
            next_chunks.push(chunk + broadcast.mul_scalar(self.clocked_slow_memory.residual_scale));
        }
        layer_state.clocked_slow_hidden = last_slow_hidden;
        Tensor::cat(next_chunks, 2)
    }

    fn forward_branch_summary_memory(
        &self,
        branch_input: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        start_pos: usize,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 4> {
        let [branch_batch, branch_views, branch_time, branch_dim] =
            branch_input.shape().dims::<4>();
        if branch_time == 0 {
            layer_state.summary_memory_hidden = None;
            return branch_input;
        }

        let chunk_tokens = self.summary_memory.chunk_tokens.max(1);
        if branch_time == 1
            && chunk_tokens > 1
            && start_pos % chunk_tokens != 0
            && let Some(cached) = self.resolve_summary_memory_hidden(
                layer_state,
                branch_batch,
                branch_views,
                branch_dim,
            )
        {
            return branch_input + cached.mul_scalar(self.summary_memory.residual_scale);
        }

        let mut next_chunks = Vec::with_capacity(branch_time.div_ceil(chunk_tokens));
        let mut carry =
            self.resolve_summary_memory_hidden(layer_state, branch_batch, branch_views, branch_dim);
        for chunk_start in (0..branch_time).step_by(chunk_tokens) {
            let chunk_end = (chunk_start + chunk_tokens).min(branch_time);
            let chunk_len = chunk_end - chunk_start;
            let chunk = branch_input.clone().slice_dim(2, chunk_start..chunk_end);
            let summary = chunk.clone().mean_dim(2);
            let branch_out = match carry.as_ref() {
                Some(previous) => {
                    let broadcast = previous.clone().repeat_dim(2, chunk_len);
                    chunk + broadcast.mul_scalar(self.summary_memory.residual_scale)
                }
                None => chunk,
            };
            let next_carry = match carry {
                Some(previous) => {
                    let updated = self.update_summary_memory_hidden(previous.clone(), summary);
                    if self.summary_memory_uses_write_trigger() {
                        let gate = self.summary_memory_event_gate(
                            summary_event_mask.as_ref(),
                            chunk_start,
                            chunk_end,
                            branch_batch,
                            &branch_input.device(),
                        );
                        previous
                            .mul(gate.clone().neg().add_scalar(1.0))
                            .add(updated.mul(gate))
                    } else {
                        updated
                    }
                }
                None => {
                    let updated = summary.mul_scalar(self.summary_memory.state_update_scale);
                    if self.summary_memory_uses_write_trigger() {
                        let gate = self.summary_memory_event_gate(
                            summary_event_mask.as_ref(),
                            chunk_start,
                            chunk_end,
                            branch_batch,
                            &branch_input.device(),
                        );
                        updated.mul(gate)
                    } else {
                        updated
                    }
                }
            };
            next_chunks.push(branch_out);
            carry = Some(next_carry);
        }
        layer_state.summary_memory_hidden = carry;
        Tensor::cat(next_chunks, 2)
    }

    fn summary_memory_event_gate(
        &self,
        summary_event_mask: Option<&Tensor<B, 2, Int>>,
        chunk_start: usize,
        chunk_end: usize,
        branch_batch: usize,
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let Some(summary_event_mask) = summary_event_mask else {
            return Tensor::<B, 4>::zeros([branch_batch, 1, 1, 1], device);
        };
        let hits = summary_event_mask
            .clone()
            .slice_dim(1, chunk_start..chunk_end)
            .float()
            .sum_dim(1)
            .reshape([branch_batch, 1, 1, 1]);
        hits.clone().div(hits.add_scalar(1.0e-6))
    }

    fn update_summary_memory_hidden(
        &self,
        previous: Tensor<B, 4>,
        summary: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let ungated = previous
            .clone()
            .mul_scalar(self.summary_memory.state_decay)
            .add(
                summary
                    .clone()
                    .mul_scalar(self.summary_memory.state_update_scale),
            );
        let threshold = self.summary_memory.surprise_gate_threshold;
        if threshold <= 0.0 {
            return ungated;
        }

        let gate_logits = activation::relu(
            (summary.clone() - previous.clone())
                .abs()
                .mean_dim(3)
                .mean_dim(2)
                .sub_scalar(threshold)
                .mul_scalar(self.summary_memory.surprise_gate_sharpness),
        );
        let gate = gate_logits.clone().div(gate_logits.add_scalar(1.0));
        previous
            .mul(gate.clone().neg().add_scalar(1.0))
            .add(ungated.mul(gate))
    }

    fn residual_connector_for_layer(&self, layer_idx: usize) -> ResidualConnectorRef<'_, B> {
        match self.residual_connector {
            ResidualConnectorKind::Vanilla => ResidualConnectorRef::Vanilla,
            ResidualConnectorKind::Mhc => {
                if layer_idx < self.mhc_first_layer {
                    ResidualConnectorRef::Vanilla
                } else if let Some(mhc) = self.mhc_shared.as_ref() {
                    ResidualConnectorRef::Mhc(mhc)
                } else {
                    ResidualConnectorRef::Vanilla
                }
            }
            ResidualConnectorKind::AttentionResidual => {
                if layer_idx < self.attention_residual_first_layer {
                    ResidualConnectorRef::Vanilla
                } else if let Some(attention_residual) = self.attention_residual_shared.as_ref() {
                    ResidualConnectorRef::AttentionResidual(attention_residual)
                } else {
                    ResidualConnectorRef::Vanilla
                }
            }
            ResidualConnectorKind::BlockAttentionResidual => {
                if layer_idx < self.block_attention_residual_first_layer {
                    ResidualConnectorRef::Vanilla
                } else if let Some(block_attention_residual) =
                    self.block_attention_residual_shared.as_ref()
                {
                    ResidualConnectorRef::BlockAttentionResidual(block_attention_residual)
                } else {
                    ResidualConnectorRef::Vanilla
                }
            }
        }
    }

    #[cfg(test)]
    fn mhc_for_layer(&self, layer_idx: usize) -> Option<&ManifoldHyperConnections<B>> {
        match self.residual_connector_for_layer(layer_idx) {
            ResidualConnectorRef::Mhc(mhc) => Some(mhc),
            _ => None,
        }
    }

    fn prepare_language_residuals(
        &self,
        residuals: Tensor<B, 4>,
        connector: &ResidualConnectorRef<'_, B>,
    ) -> Tensor<B, 4> {
        match connector {
            ResidualConnectorRef::Mhc(mhc) => mhc.bootstrap_streams(residuals),
            ResidualConnectorRef::Vanilla
            | ResidualConnectorRef::AttentionResidual(_)
            | ResidualConnectorRef::BlockAttentionResidual(_) => residuals,
        }
    }

    fn collapse_language_streams(&self, current: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, streams, time, dim] = current.shape().dims();
        if streams == 1 {
            current.reshape([batch, time, dim])
        } else {
            current.mean_dim(1).reshape([batch, time, dim])
        }
    }

    fn residual_connector_needs_post_merge_norm(
        &self,
        connector: &ResidualConnectorRef<'_, B>,
    ) -> bool {
        !matches!(connector, ResidualConnectorRef::Vanilla)
    }

    fn split_language_residuals_for_layer(
        &self,
        current: Tensor<B, 4>,
        connector: &ResidualConnectorRef<'_, B>,
        residual_history: &[Tensor<B, 4>],
        mhc_coefficients: Option<&super::ManifoldHyperConnectionCoefficients<B>>,
    ) -> LanguageMhcLayerBindings<B> {
        let current_residuals = self.prepare_language_residuals(current.clone(), &connector);
        match connector {
            ResidualConnectorRef::Mhc(mhc)
                if mhc.coefficient_policy().uses_dynamic_stream_controller() =>
            {
                let output = mhc.stream_width_connection(current_residuals);
                LanguageMhcLayerBindings {
                    branch_input: output.branch_input,
                    residuals_base: output.residuals_out,
                    legacy_beta: None,
                    stream_coefficients: Some(output.coefficients),
                }
            }
            ResidualConnectorRef::AttentionResidual(attention_residual) => {
                let branch_input =
                    attention_residual.branch_input(current_residuals.clone(), residual_history);
                LanguageMhcLayerBindings {
                    branch_input,
                    residuals_base: current_residuals,
                    legacy_beta: None,
                    stream_coefficients: None,
                }
            }
            ResidualConnectorRef::BlockAttentionResidual(block_attention_residual) => {
                let branch_input = block_attention_residual
                    .branch_input(current_residuals.clone(), residual_history);
                LanguageMhcLayerBindings {
                    branch_input,
                    residuals_base: current_residuals,
                    legacy_beta: None,
                    stream_coefficients: None,
                }
            }
            _ => {
                let mhc = match connector {
                    ResidualConnectorRef::Mhc(mhc) => Some(*mhc),
                    ResidualConnectorRef::Vanilla
                    | ResidualConnectorRef::AttentionResidual(_)
                    | ResidualConnectorRef::BlockAttentionResidual(_) => None,
                };
                let (branch_input, residuals_base, legacy_beta) =
                    mhc_split_with_coefficients(mhc, current_residuals, mhc_coefficients);
                LanguageMhcLayerBindings {
                    branch_input,
                    residuals_base,
                    legacy_beta,
                    stream_coefficients: None,
                }
            }
        }
    }

    fn merge_language_residuals_for_layer(
        &self,
        branch_out: Tensor<B, 4>,
        bindings: LanguageMhcLayerBindings<B>,
        connector: &ResidualConnectorRef<'_, B>,
        mhc_coefficients: Option<&super::ManifoldHyperConnectionCoefficients<B>>,
    ) -> Tensor<B, 4> {
        match connector {
            ResidualConnectorRef::Mhc(mhc)
                if mhc.coefficient_policy().uses_dynamic_stream_controller() =>
            {
                mhc.stream_depth_connection(
                    branch_out,
                    bindings.residuals_base,
                    &bindings
                        .stream_coefficients
                        .expect("dynamic stream coefficients"),
                )
            }
            ResidualConnectorRef::AttentionResidual(_)
            | ResidualConnectorRef::BlockAttentionResidual(_) => {
                bindings.residuals_base + branch_out
            }
            _ => mhc_merge_with_coefficients(
                match connector {
                    ResidualConnectorRef::Mhc(mhc) => Some(*mhc),
                    ResidualConnectorRef::Vanilla
                    | ResidualConnectorRef::AttentionResidual(_)
                    | ResidualConnectorRef::BlockAttentionResidual(_) => None,
                },
                branch_out,
                bindings.residuals_base,
                mhc_coefficients,
                bindings.legacy_beta,
            ),
        }
    }

    fn summarize_language_mhc_layer_diagnostics(
        &self,
        layer_index: usize,
        current_residuals: Tensor<B, 4>,
        connector: &ResidualConnectorRef<'_, B>,
    ) -> Option<LanguageMhcLayerDiagnostics> {
        let ResidualConnectorRef::Mhc(mhc) = connector else {
            return None;
        };

        let [batch, streams, time, dim] = current_residuals.shape().dims::<4>();
        if streams <= 1 || time == 0 || dim == 0 {
            return None;
        }

        let residual_values = current_residuals
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("residual diagnostics values");
        let coeffs = mhc.stream_coefficients(current_residuals);
        let alpha_values = coeffs
            .branch_input_weights
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("alpha diagnostics values");
        let beta_values = coeffs.branch_output_weights.as_ref().map(|weights| {
            weights
                .clone()
                .to_data()
                .convert::<f32>()
                .into_vec::<f32>()
                .expect("beta diagnostics values")
        });
        let residual_weight_values = coeffs
            .residual_weights
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("residual weight diagnostics values");

        let mut stream_norm_sum = 0.0f64;
        let mut stream_norm_sq_sum = 0.0f64;
        let mut stream_norm_count = 0usize;
        let mut pairwise_cosine_sum = 0.0f64;
        let mut pairwise_cosine_count = 0usize;
        let mut alpha_entropy_sum = 0.0f64;
        let mut alpha_entropy_norm_sum = 0.0f64;
        let mut beta_entropy_sum = 0.0f64;
        let mut beta_entropy_norm_sum = 0.0f64;
        let mut beta_entropy_count = 0usize;
        let mut residual_identity_l1_sum = 0.0f64;
        let mut residual_uniform_l1_sum = 0.0f64;

        let entropy_norm = (streams as f64).ln().max(1.0e-12);
        let uniform_weight = 1.0f64 / streams as f64;

        for batch_idx in 0..batch {
            for time_idx in 0..time {
                let mut stream_norms = vec![0.0f64; streams];
                for stream_idx in 0..streams {
                    let mut sum_sq = 0.0f64;
                    let base = ((batch_idx * streams + stream_idx) * time + time_idx) * dim;
                    for dim_idx in 0..dim {
                        let value = residual_values[base + dim_idx] as f64;
                        sum_sq += value * value;
                    }
                    let norm = sum_sq.sqrt();
                    stream_norms[stream_idx] = norm;
                    stream_norm_sum += norm;
                    stream_norm_sq_sum += norm * norm;
                    stream_norm_count += 1;
                }

                if streams > 1 {
                    for left in 0..streams {
                        for right in (left + 1)..streams {
                            let left_base = ((batch_idx * streams + left) * time + time_idx) * dim;
                            let right_base =
                                ((batch_idx * streams + right) * time + time_idx) * dim;
                            let mut dot = 0.0f64;
                            for dim_idx in 0..dim {
                                dot += residual_values[left_base + dim_idx] as f64
                                    * residual_values[right_base + dim_idx] as f64;
                            }
                            let denom = (stream_norms[left] * stream_norms[right]).max(1.0e-12);
                            pairwise_cosine_sum += dot / denom;
                            pairwise_cosine_count += 1;
                        }
                    }
                }

                let alpha_base = (batch_idx * time + time_idx) * streams;
                let alpha_slice = &alpha_values[alpha_base..alpha_base + streams];
                let alpha_entropy = shannon_entropy(alpha_slice);
                alpha_entropy_sum += alpha_entropy;
                alpha_entropy_norm_sum += alpha_entropy / entropy_norm;

                if let Some(beta_values) = beta_values.as_ref() {
                    let beta_slice = &beta_values[alpha_base..alpha_base + streams];
                    let beta_entropy = shannon_entropy(beta_slice);
                    beta_entropy_sum += beta_entropy;
                    beta_entropy_norm_sum += beta_entropy / entropy_norm;
                    beta_entropy_count += 1;
                }

                let weight_base = (batch_idx * time + time_idx) * streams * streams;
                let matrix = &residual_weight_values[weight_base..weight_base + streams * streams];
                let mut identity_l1 = 0.0f64;
                let mut uniform_l1 = 0.0f64;
                for row in 0..streams {
                    for col in 0..streams {
                        let value = matrix[row * streams + col] as f64;
                        let identity = if row == col { 1.0 } else { 0.0 };
                        identity_l1 += (value - identity).abs();
                        uniform_l1 += (value - uniform_weight).abs();
                    }
                }
                residual_identity_l1_sum += identity_l1;
                residual_uniform_l1_sum += uniform_l1;
            }
        }

        let stream_norm_count_f = stream_norm_count.max(1) as f64;
        let token_count = (batch * time).max(1) as f64;
        let stream_norm_mean = stream_norm_sum / stream_norm_count_f;
        let stream_norm_variance =
            (stream_norm_sq_sum / stream_norm_count_f) - stream_norm_mean * stream_norm_mean;

        Some(LanguageMhcLayerDiagnostics {
            layer_index,
            num_streams: streams,
            stream_norm_mean,
            stream_norm_variance: stream_norm_variance.max(0.0),
            pairwise_stream_cosine_mean: (pairwise_cosine_count > 0)
                .then_some(pairwise_cosine_sum / pairwise_cosine_count as f64),
            alpha_entropy_mean: alpha_entropy_sum / token_count,
            alpha_entropy_normalized_mean: alpha_entropy_norm_sum / token_count,
            beta_entropy_mean: (beta_entropy_count > 0)
                .then_some(beta_entropy_sum / beta_entropy_count as f64),
            beta_entropy_normalized_mean: (beta_entropy_count > 0)
                .then_some(beta_entropy_norm_sum / beta_entropy_count as f64),
            residual_distance_identity_l1_mean: residual_identity_l1_sum / token_count,
            residual_distance_uniform_l1_mean: residual_uniform_l1_sum / token_count,
        })
    }

    fn collect_language_mhc_diagnostics_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.collect_language_mhc_diagnostics_from_embedded_single_pass(
                embedded,
                state,
                start_pos,
                true,
                RecurrentPositionMode::Sequential,
                summary_event_mask,
            );
        }

        let [_batch, slow_steps, _embd] = embedded.shape().dims::<3>();
        if slow_steps == 0 {
            return Vec::new();
        }

        let mut per_pass = Vec::new();
        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            for _ in 0..self.rollout_fast_steps_per_slow_step {
                per_pass.push(
                    self.collect_language_mhc_diagnostics_from_embedded_single_pass(
                        token_embedded.clone(),
                        state,
                        start_pos,
                        false,
                        RecurrentPositionMode::Sequential,
                        token_summary_event_mask.clone(),
                    ),
                );
            }
            state.position = state.position.saturating_add(1);
        }

        average_language_mhc_diagnostics(per_pass)
    }

    fn collect_language_mhc_diagnostics_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        if self.y_neuron_recurrence.enabled {
            return self
                .collect_language_mhc_diagnostics_from_embedded_single_pass_y_neuron_recurrence(
                    embedded,
                    state,
                    start_pos,
                    advance_position,
                    position_mode,
                );
        }

        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = vec![current.clone()];

        let mut diagnostics = Vec::new();
        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_residuals = self.prepare_language_residuals(current.clone(), &connector);
            if let Some(layer_diag) = self.summarize_language_mhc_layer_diagnostics(
                layer_idx,
                current_residuals,
                &connector,
            ) {
                diagnostics.push(layer_diag);
            }
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                &residual_history,
                mhc_coefficients,
            );
            let branch_input = if self.summary_memory_applies_to_layer(layer_idx) {
                self.forward_branch_summary_memory(
                    bindings.branch_input.clone(),
                    layer_state,
                    start_pos,
                    summary_event_mask.clone(),
                )
            } else {
                layer_state.summary_memory_hidden = None;
                bindings.branch_input.clone()
            };

            if self.clocked_slow_memory_applies_to_layer(layer_idx) {
                let branch_out = self.forward_branch_clocked_slow_layer(
                    layer_idx,
                    branch_input,
                    layer_state,
                    start_pos,
                    position_mode,
                );
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    bindings,
                    &connector,
                    mhc_coefficients,
                );
                current = if self.residual_connector_needs_post_merge_norm(&connector) {
                    self.norm.forward(next)
                } else {
                    next
                };
                residual_history.push(current.clone());
                continue;
            }
            layer_state.clocked_slow_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let branch_flat =
                branch_input.reshape([branch_batch * branch_views, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            let fused_recurrent_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    branch_batch * branch_views,
                    self.n_head,
                    1,
                    branch_time,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };
            let next = lowrank_residual_step_next(
                branch_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused && self.kernel.projection_executor.use_x(),
                fused && self.kernel.projection_executor.use_y(),
                self.kernel.relu_threshold,
                true,
                latent_pattern,
                self.kernel.lowrank_grad_input_executor,
                sparse_mask.clone(),
                |query, value| {
                    self.recurrent_attention_with_plan(
                        query,
                        value,
                        layer_state,
                        start_pos,
                        position_mode,
                        fused_recurrent_plan.as_ref(),
                    )
                },
                |values| activation::relu(values),
                |values| self.norm.forward(values),
            );
            let branch_out = next.reshape([branch_batch, branch_views, branch_time, branch_dim]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            residual_history.push(current.clone());
        }

        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        diagnostics
    }

    fn collect_language_mhc_diagnostics_from_embedded_single_pass_y_neuron_recurrence(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = vec![current.clone()];

        let mut diagnostics = Vec::new();
        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_residuals = self.prepare_language_residuals(current.clone(), &connector);
            if let Some(layer_diag) = self.summarize_language_mhc_layer_diagnostics(
                layer_idx,
                current_residuals,
                &connector,
            ) {
                diagnostics.push(layer_diag);
            }
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                &residual_history,
                mhc_coefficients,
            );
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                bindings.branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat =
                bindings
                    .branch_input
                    .clone()
                    .reshape([flat_batch, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let heads = self.n_head;
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            if !self.y_neuron_recurrence_applies_to_layer(layer_idx) {
                layer_state.y_neuron_state = None;
                let fused_recurrent_plan = if self.kernel.enabled
                    && self.kernel.wgpu_recurrent_kernel
                    && supports_recurrent_backend::<B>()
                {
                    Some(CompiledRecurrentAttentionPlan::new(
                        flat_batch,
                        self.n_head,
                        1,
                        branch_time,
                        latent,
                        branch_dim,
                        &branch_flat.device(),
                    ))
                } else {
                    None
                };
                let next = lowrank_residual_step_next(
                    branch_flat,
                    encoder.clone(),
                    encoder_v.clone(),
                    decoder.clone(),
                    &self.dropout,
                    fused && self.kernel.projection_executor.use_x(),
                    fused && self.kernel.projection_executor.use_y(),
                    self.kernel.relu_threshold,
                    true,
                    latent_pattern,
                    self.kernel.lowrank_grad_input_executor,
                    sparse_mask.clone(),
                    |query, value| {
                        self.recurrent_attention_with_plan(
                            query,
                            value,
                            layer_state,
                            start_pos,
                            position_mode,
                            fused_recurrent_plan.as_ref(),
                        )
                    },
                    |values| activation::relu(values),
                    |values| self.norm.forward(values),
                );
                let branch_out =
                    next.reshape([branch_batch, branch_views, branch_time, branch_dim]);
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    bindings,
                    &connector,
                    mhc_coefficients,
                );
                current = if self.residual_connector_needs_post_merge_norm(&connector) {
                    self.norm.forward(next)
                } else {
                    next
                };
                residual_history.push(current.clone());
                continue;
            }
            let x_base = self.project_lowrank_positive(
                branch_flat.clone(),
                encoder.clone(),
                fused,
                latent_pattern,
                sparse_mask.clone(),
            );
            let mut next_tokens = Vec::with_capacity(branch_time);
            let mut y_neuron_state = self.resolve_y_neuron_state(
                layer_state,
                flat_batch,
                self.n_head,
                latent,
                &branch_flat.device(),
            );
            let chunk_tokens = self
                .y_neuron_recurrence
                .chunk_tokens
                .max(1)
                .min(branch_time.max(1));
            let fused_recurrent_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    self.n_head,
                    1,
                    chunk_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };
            let tail_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
                && branch_time % chunk_tokens != 0
            {
                let tail_tokens = branch_time % chunk_tokens;
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    self.n_head,
                    1,
                    tail_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };

            for chunk_start in (0..branch_time).step_by(chunk_tokens) {
                let chunk_end = (chunk_start + chunk_tokens).min(branch_time);
                let chunk_len = chunk_end - chunk_start;
                let x_neuron_base = x_base.clone().slice_dim(2, chunk_start..chunk_end);
                let x_neuron = self.inject_y_neuron_state(x_neuron_base, y_neuron_state.clone());
                let current_token = branch_flat.clone().slice_dim(2, chunk_start..chunk_end);
                let token_position = match position_mode {
                    RecurrentPositionMode::Sequential => start_pos + chunk_start,
                    RecurrentPositionMode::Fixed => start_pos,
                };
                let a_dense = self.recurrent_attention_with_plan(
                    x_neuron.clone(),
                    current_token.clone(),
                    layer_state,
                    token_position,
                    position_mode,
                    if chunk_len == chunk_tokens {
                        fused_recurrent_plan.as_ref()
                    } else {
                        tail_plan.as_ref()
                    },
                );
                let a_dense = self.norm.forward(a_dense);
                let y_gate = self.project_lowrank_positive(
                    a_dense,
                    encoder_v.clone(),
                    fused,
                    latent_pattern,
                    sparse_mask.clone(),
                );
                let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                let mixed = y_neuron.clone().swap_dims(1, 2);
                let mixed_flat = mixed.reshape([flat_batch * chunk_len, heads * latent]);
                let mlp_flat = mixed_flat.matmul(decoder.clone());
                let mlp_out = self
                    .norm
                    .forward(mlp_flat.reshape([flat_batch, 1, chunk_len, branch_dim]));
                next_tokens.push(self.norm.forward(current_token + mlp_out));
                let y_neuron_last = y_neuron.clone().slice_dim(2, (chunk_len - 1)..chunk_len);
                y_neuron_state = self.update_y_neuron_state(y_neuron_state, y_neuron_last);
            }

            layer_state.y_neuron_state = Some(y_neuron_state);

            let branch_out = Tensor::cat(next_tokens, 2).reshape([
                branch_batch,
                branch_views,
                branch_time,
                branch_dim,
            ]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            residual_history.push(current.clone());
        }

        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        diagnostics
    }

    fn forward_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let embedded = self.embed.forward(tokens);
        self.forward_with_state_from_embedded(embedded, state, summary_event_mask)
    }

    fn forward_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.forward_with_state_from_embedded_single_pass(
                embedded,
                state,
                start_pos,
                true,
                RecurrentPositionMode::Sequential,
                summary_event_mask,
            );
        }

        match self.rollout_executor_mode() {
            RolloutExecutorMode::HostLoop => self
                .forward_with_state_from_embedded_rollout_host_loop(
                    embedded,
                    state,
                    summary_event_mask,
                ),
            RolloutExecutorMode::WgpuFused => self.forward_with_state_from_embedded_rollout_fused(
                embedded,
                state,
                summary_event_mask,
            ),
        }
    }

    fn forward_with_state_from_embedded_rollout_host_loop(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            let hidden = Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
            let logits = Tensor::<B, 3>::zeros([batch, 0, self.vocab_size], &device);
            return (hidden, logits);
        }

        let mut hidden_slow = Vec::with_capacity(slow_steps);
        let mut logits_slow = Vec::with_capacity(slow_steps);
        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let mut hidden_last = None;
            let mut logits_last = None;
            for _ in 0..self.rollout_fast_steps_per_slow_step {
                let (hidden, logits) = self.forward_with_state_from_embedded_single_pass(
                    token_embedded.clone(),
                    state,
                    start_pos,
                    false,
                    RecurrentPositionMode::Sequential,
                    token_summary_event_mask.clone(),
                );
                hidden_last = Some(hidden);
                logits_last = Some(logits);
            }
            hidden_slow.push(hidden_last.expect("rollout hidden output"));
            logits_slow.push(logits_last.expect("rollout logits output"));
            state.position = state.position.saturating_add(1);
        }

        (Tensor::cat(hidden_slow, 1), Tensor::cat(logits_slow, 1))
    }

    fn forward_with_state_from_embedded_rollout_fused(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, slow_steps, _embd] = embedded.shape().dims::<3>();

        if slow_steps == 0 {
            let device = embedded.device();
            let hidden = Tensor::<B, 3>::zeros([batch, 0, self.n_embd], &device);
            let logits = Tensor::<B, 3>::zeros([batch, 0, self.vocab_size], &device);
            return (hidden, logits);
        }

        let fast_steps = self.rollout_fast_steps_per_slow_step;
        let mut hidden_slow = Vec::with_capacity(slow_steps);
        let mut logits_slow = Vec::with_capacity(slow_steps);

        for slow_idx in 0..slow_steps {
            let token_embedded = embedded.clone().slice_dim(1, slow_idx..slow_idx + 1);
            let rollout_embedded = token_embedded.repeat_dim(1, fast_steps);
            let token_summary_event_mask = summary_event_mask
                .as_ref()
                .map(|mask| mask.clone().slice_dim(1, slow_idx..slow_idx + 1));
            let start_pos = state.position;
            let hidden_rollout = self.forward_hidden_with_state_from_embedded_single_pass(
                rollout_embedded,
                state,
                start_pos,
                false,
                RecurrentPositionMode::Fixed,
                token_summary_event_mask,
            );
            let last = fast_steps - 1;
            let hidden_last = hidden_rollout.slice_dim(1, last..fast_steps);
            let logits_last = self.project_hidden_to_logits(hidden_last.clone());
            hidden_slow.push(hidden_last);
            logits_slow.push(logits_last);
            state.position = state.position.saturating_add(1);
        }

        (Tensor::cat(hidden_slow, 1), Tensor::cat(logits_slow, 1))
    }

    fn forward_with_state_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = self.forward_hidden_with_state_from_embedded_single_pass(
            embedded,
            state,
            start_pos,
            advance_position,
            position_mode,
            summary_event_mask,
        );
        let logits = self.project_hidden_to_logits(hidden.clone());
        (hidden, logits)
    }

    fn forward_hidden_with_state_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        self.forward_hidden_with_state_from_embedded_single_pass_layer_limit(
            embedded,
            state,
            start_pos,
            advance_position,
            position_mode,
            summary_event_mask,
            self.n_layer,
        )
    }

    fn initialize_language_pipeline_state(
        &self,
        embedded: Tensor<B, 3>,
    ) -> LanguagePipelineState<B> {
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        LanguagePipelineState {
            current: current.clone(),
            residual_history: vec![current],
        }
    }

    fn forward_language_pipeline_state_layer_range(
        &self,
        mut pipeline_state: LanguagePipelineState<B>,
        state: &mut ModelState<B>,
        start_pos: usize,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        layer_range: Range<usize>,
    ) -> LanguagePipelineState<B> {
        assert!(
            !self.y_neuron_recurrence.enabled,
            "layer-range pipeline execution is not supported with y-neuron recurrence enabled"
        );

        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let layer_end = layer_range.end.min(self.n_layer);

        for layer_idx in layer_range.start.min(layer_end)..layer_end {
            let layer_state = &mut state.layers[layer_idx];
            let connector = self.residual_connector_for_layer(layer_idx);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                pipeline_state.current,
                &connector,
                &pipeline_state.residual_history,
                mhc_coefficients,
            );
            let branch_input = if self.summary_memory_applies_to_layer(layer_idx) {
                self.forward_branch_summary_memory(
                    bindings.branch_input.clone(),
                    layer_state,
                    start_pos,
                    summary_event_mask.clone(),
                )
            } else {
                layer_state.summary_memory_hidden = None;
                bindings.branch_input.clone()
            };

            if self.clocked_slow_memory_applies_to_layer(layer_idx) {
                let branch_out = self.forward_branch_clocked_slow_layer(
                    layer_idx,
                    branch_input,
                    layer_state,
                    start_pos,
                    position_mode,
                );
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    bindings,
                    &connector,
                    mhc_coefficients,
                );
                pipeline_state.current =
                    if self.residual_connector_needs_post_merge_norm(&connector) {
                        self.norm.forward(next)
                    } else {
                        next
                    };
                pipeline_state
                    .residual_history
                    .push(pipeline_state.current.clone());
                continue;
            }
            layer_state.clocked_slow_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let branch_flat =
                branch_input.reshape([branch_batch * branch_views, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            let fused_recurrent_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    branch_batch * branch_views,
                    self.n_head,
                    1,
                    branch_time,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };
            let output = lowrank_residual_step(
                branch_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused && self.kernel.projection_executor.use_x(),
                fused && self.kernel.projection_executor.use_y(),
                self.kernel.relu_threshold,
                true,
                latent_pattern,
                self.kernel.lowrank_grad_input_executor,
                sparse_mask.clone(),
                |query, value| {
                    self.recurrent_attention_with_plan(
                        query,
                        value,
                        layer_state,
                        start_pos,
                        position_mode,
                        fused_recurrent_plan.as_ref(),
                    )
                },
                |values| activation::relu(values),
                |values| self.norm.forward(values),
            );

            #[cfg(feature = "viz")]
            let mixed = output.y_neuron.clone().swap_dims(1, 2);
            #[cfg(feature = "viz")]
            let [flat_batch, time, heads, latent] = mixed.shape().dims();

            #[cfg(feature = "viz")]
            if time > 0 {
                let last = time - 1;
                let viz_batch = branch_batch.max(1);
                let viz_views = branch_views.max(1);
                let x_neuron_last = output
                    .x_neuron
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_gate_last = output
                    .y_gate
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_neuron_last = output
                    .y_neuron
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let device = x_neuron_last.device();
                let rho_last = match layer_state.rho.as_ref() {
                    Some(rho) => {
                        let dims = rho.shape().dims::<4>();
                        if dims == [flat_batch, heads, latent, self.n_embd] {
                            let rho_energy =
                                rho.clone().abs().sum_dim(3).div_scalar(self.n_embd as f32);
                            let rho_energy = rho_energy
                                .reshape([viz_batch, viz_views, heads, latent])
                                .mean_dim(1)
                                .sum_dim(0)
                                .div_scalar(viz_batch as f32);
                            rho_energy.reshape([heads, latent])
                        } else {
                            Tensor::<B, 2>::zeros([heads, latent], &device)
                        }
                    }
                    None => Tensor::<B, 2>::zeros([heads, latent], &device),
                };

                layer_state.viz = Some(LayerVizState {
                    x_neuron_last,
                    y_gate_last,
                    y_neuron_last,
                    rho_last,
                });
            }

            let branch_out =
                output
                    .next
                    .reshape([branch_batch, branch_views, branch_time, branch_dim]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                bindings,
                &connector,
                mhc_coefficients,
            );
            pipeline_state.current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            pipeline_state
                .residual_history
                .push(pipeline_state.current.clone());
        }

        pipeline_state
    }

    fn forward_hidden_with_state_from_embedded_single_pass_layer_limit(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        layer_limit: usize,
    ) -> Tensor<B, 3> {
        if self.y_neuron_recurrence.enabled {
            assert_eq!(
                layer_limit, self.n_layer,
                "layer-limited profiling is not supported with y-neuron recurrence enabled"
            );
            return self.forward_hidden_with_state_from_embedded_single_pass_y_neuron_recurrence(
                embedded,
                state,
                start_pos,
                advance_position,
                position_mode,
            );
        }
        let pipeline_state = self.initialize_language_pipeline_state(embedded);
        let pipeline_state = self.forward_language_pipeline_state_layer_range(
            pipeline_state,
            state,
            start_pos,
            position_mode,
            summary_event_mask,
            0..layer_limit.min(self.n_layer),
        );
        let hidden = self.collapse_language_streams(pipeline_state.current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        hidden
    }

    fn forward_hidden_with_state_from_embedded_single_pass_y_neuron_recurrence(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let mut current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        let fused = self.kernel.enabled;
        let static_mhc_coefficients = self.mhc_shared.as_ref().and_then(|mhc| {
            (!mhc.coefficient_policy().uses_dynamic_stream_controller()).then(|| mhc.coefficients())
        });
        let mut residual_history = vec![current.clone()];

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                &residual_history,
                mhc_coefficients,
            );
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                bindings.branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat =
                bindings
                    .branch_input
                    .clone()
                    .reshape([flat_batch, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let heads = self.n_head;
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };
            if !self.y_neuron_recurrence_applies_to_layer(layer_idx) {
                layer_state.y_neuron_state = None;
                let fused_recurrent_plan = if self.kernel.enabled
                    && self.kernel.wgpu_recurrent_kernel
                    && supports_recurrent_backend::<B>()
                {
                    Some(CompiledRecurrentAttentionPlan::new(
                        flat_batch,
                        heads,
                        1,
                        branch_time,
                        latent,
                        branch_dim,
                        &branch_flat.device(),
                    ))
                } else {
                    None
                };
                let output = lowrank_residual_step(
                    branch_flat,
                    encoder.clone(),
                    encoder_v.clone(),
                    decoder.clone(),
                    &self.dropout,
                    fused && self.kernel.projection_executor.use_x(),
                    fused && self.kernel.projection_executor.use_y(),
                    self.kernel.relu_threshold,
                    true,
                    latent_pattern,
                    self.kernel.lowrank_grad_input_executor,
                    sparse_mask.clone(),
                    |query, value| {
                        self.recurrent_attention_with_plan(
                            query,
                            value,
                            layer_state,
                            start_pos,
                            position_mode,
                            fused_recurrent_plan.as_ref(),
                        )
                    },
                    |values| activation::relu(values),
                    |values| self.norm.forward(values),
                );

                #[cfg(feature = "viz")]
                if branch_time > 0 {
                    let last = branch_time - 1;
                    let viz_batch = branch_batch.max(1);
                    let viz_views = branch_views.max(1);
                    let x_neuron_last = output
                        .x_neuron
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let y_gate_last = output
                        .y_gate
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let y_neuron_last = output
                        .y_neuron
                        .clone()
                        .slice_dim(2, last..branch_time)
                        .reshape([viz_batch, viz_views, heads, latent])
                        .mean_dim(1)
                        .slice_dim(0, 0..1)
                        .reshape([heads, latent]);
                    let device = x_neuron_last.device();
                    let rho_last = match layer_state.rho.as_ref() {
                        Some(rho) => {
                            let dims = rho.shape().dims::<4>();
                            if dims == [flat_batch, heads, latent, self.n_embd] {
                                let rho_energy =
                                    rho.clone().abs().sum_dim(3).div_scalar(self.n_embd as f32);
                                let rho_energy = rho_energy
                                    .reshape([viz_batch, viz_views, heads, latent])
                                    .mean_dim(1)
                                    .sum_dim(0)
                                    .div_scalar(viz_batch as f32);
                                rho_energy.reshape([heads, latent])
                            } else {
                                Tensor::<B, 2>::zeros([heads, latent], &device)
                            }
                        }
                        None => Tensor::<B, 2>::zeros([heads, latent], &device),
                    };

                    layer_state.viz = Some(LayerVizState {
                        x_neuron_last,
                        y_gate_last,
                        y_neuron_last,
                        rho_last,
                    });
                }

                let branch_out =
                    output
                        .next
                        .reshape([branch_batch, branch_views, branch_time, branch_dim]);
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    bindings,
                    &connector,
                    mhc_coefficients,
                );
                current = if self.residual_connector_needs_post_merge_norm(&connector) {
                    self.norm.forward(next)
                } else {
                    next
                };
                residual_history.push(current.clone());
                continue;
            }
            let x_base = self.project_lowrank_positive(
                branch_flat.clone(),
                encoder.clone(),
                fused,
                latent_pattern,
                sparse_mask.clone(),
            );
            let mut next_tokens = Vec::with_capacity(branch_time);
            let mut y_neuron_state = self.resolve_y_neuron_state(
                layer_state,
                flat_batch,
                heads,
                latent,
                &branch_flat.device(),
            );
            let chunk_tokens = self
                .y_neuron_recurrence
                .chunk_tokens
                .max(1)
                .min(branch_time.max(1));
            let fused_recurrent_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    chunk_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };
            let tail_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
                && branch_time % chunk_tokens != 0
            {
                let tail_tokens = branch_time % chunk_tokens;
                Some(CompiledRecurrentAttentionPlan::new(
                    flat_batch,
                    heads,
                    1,
                    tail_tokens,
                    latent,
                    branch_dim,
                    &branch_flat.device(),
                ))
            } else {
                None
            };

            #[cfg(feature = "viz")]
            let mut viz_last: Option<(Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 4>)> = None;

            for chunk_start in (0..branch_time).step_by(chunk_tokens) {
                let chunk_end = (chunk_start + chunk_tokens).min(branch_time);
                let chunk_len = chunk_end - chunk_start;
                let x_neuron_base = x_base.clone().slice_dim(2, chunk_start..chunk_end);
                let x_neuron = self.inject_y_neuron_state(x_neuron_base, y_neuron_state.clone());
                let current_token = branch_flat.clone().slice_dim(2, chunk_start..chunk_end);
                let token_position = match position_mode {
                    RecurrentPositionMode::Sequential => start_pos + chunk_start,
                    RecurrentPositionMode::Fixed => start_pos,
                };
                let a_dense = self.recurrent_attention_with_plan(
                    x_neuron.clone(),
                    current_token.clone(),
                    layer_state,
                    token_position,
                    position_mode,
                    if chunk_len == chunk_tokens {
                        fused_recurrent_plan.as_ref()
                    } else {
                        tail_plan.as_ref()
                    },
                );
                let a_dense = self.norm.forward(a_dense);
                let y_gate = self.project_lowrank_positive(
                    a_dense,
                    encoder_v.clone(),
                    fused,
                    latent_pattern,
                    sparse_mask.clone(),
                );
                let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                let mixed = y_neuron.clone().swap_dims(1, 2);
                let mixed_flat = mixed.reshape([flat_batch * chunk_len, heads * latent]);
                let mlp_flat = mixed_flat.matmul(decoder.clone());
                let mlp_out = self
                    .norm
                    .forward(mlp_flat.reshape([flat_batch, 1, chunk_len, branch_dim]));
                next_tokens.push(self.norm.forward(current_token + mlp_out));
                let y_neuron_last = y_neuron.clone().slice_dim(2, (chunk_len - 1)..chunk_len);
                y_neuron_state = self.update_y_neuron_state(y_neuron_state, y_neuron_last);

                #[cfg(feature = "viz")]
                if chunk_end == branch_time {
                    let last_start = chunk_len - 1;
                    viz_last = Some((
                        x_neuron.slice_dim(2, last_start..chunk_len),
                        y_gate.slice_dim(2, last_start..chunk_len),
                        y_neuron.slice_dim(2, last_start..chunk_len),
                    ));
                }
            }

            layer_state.y_neuron_state = Some(y_neuron_state);

            #[cfg(feature = "viz")]
            if let Some((x_neuron_last_raw, y_gate_last_raw, y_neuron_last_raw)) = viz_last {
                let viz_batch = branch_batch.max(1);
                let viz_views = branch_views.max(1);
                let x_neuron_last = x_neuron_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_gate_last = y_gate_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_neuron_last = y_neuron_last_raw
                    .reshape([viz_batch, viz_views, heads, latent])
                    .mean_dim(1)
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let device = x_neuron_last.device();
                let rho_last = match layer_state.rho.as_ref() {
                    Some(rho) => {
                        let dims = rho.shape().dims::<4>();
                        if dims == [flat_batch, heads, latent, self.n_embd] {
                            let rho_energy =
                                rho.clone().abs().sum_dim(3).div_scalar(self.n_embd as f32);
                            let rho_energy = rho_energy
                                .reshape([viz_batch, viz_views, heads, latent])
                                .mean_dim(1)
                                .sum_dim(0)
                                .div_scalar(viz_batch as f32);
                            rho_energy.reshape([heads, latent])
                        } else {
                            Tensor::<B, 2>::zeros([heads, latent], &device)
                        }
                    }
                    None => Tensor::<B, 2>::zeros([heads, latent], &device),
                };

                layer_state.viz = Some(LayerVizState {
                    x_neuron_last,
                    y_gate_last,
                    y_neuron_last,
                    rho_last,
                });
            }

            let branch_out = Tensor::cat(next_tokens, 2).reshape([
                branch_batch,
                branch_views,
                branch_time,
                branch_dim,
            ]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                bindings,
                &connector,
                mhc_coefficients,
            );
            current = if self.residual_connector_needs_post_merge_norm(&connector) {
                self.norm.forward(next)
            } else {
                next
            };
            residual_history.push(current.clone());
        }

        let hidden = self.collapse_language_streams(current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        hidden
    }

    fn project_hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let prof_enabled = logits_projection_profile_enabled();
        let start = prof_enabled.then(Instant::now);
        let [batch, time, dim] = hidden.shape().dims();
        let logits = hidden
            .reshape([batch * time, dim])
            .matmul(self.lm_head.val())
            .reshape([batch, time, self.vocab_size]);
        if let Some(start) = start {
            logits_projection_profile_record(start.elapsed().as_nanos());
        }
        logits
    }

    pub fn logits_from_hidden(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        self.project_hidden_to_logits(hidden)
    }

    pub fn forward_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) = self.forward_with_state_impl(tokens, state, None);
        logits
    }

    pub fn forward_with_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) =
            self.forward_with_state_impl(tokens, state, Some(summary_event_mask));
        logits
    }

    pub fn forward_with_hidden_and_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_impl(tokens, state, None)
    }

    pub fn forward_with_hidden_and_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_impl(tokens, state, Some(summary_event_mask))
    }

    pub fn forward_with_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) = self.forward_with_state_from_embedded(embedded, state, None);
        logits
    }

    pub fn forward_with_hidden_and_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_from_embedded(embedded, state, None)
    }

    #[doc(hidden)]
    pub fn forward_hidden_prefix_layers_from_embedded_for_profile(
        &self,
        embedded: Tensor<B, 3>,
        layer_limit: usize,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_hidden_with_state_from_embedded_single_pass_layer_limit(
            embedded,
            &mut state,
            0,
            true,
            RecurrentPositionMode::Sequential,
            summary_event_mask,
            layer_limit.min(self.n_layer),
        )
    }

    pub fn collect_language_mhc_diagnostics_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        let embedded = self.embed.forward(tokens);
        self.collect_language_mhc_diagnostics_from_embedded(embedded, state, None)
    }

    pub fn collect_language_mhc_diagnostics_with_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Vec<LanguageMhcLayerDiagnostics> {
        let embedded = self.embed.forward(tokens);
        self.collect_language_mhc_diagnostics_from_embedded(
            embedded,
            state,
            Some(summary_event_mask),
        )
    }

    pub fn summary_memory_write_trigger_token_ids(&self) -> Option<&[u32]> {
        self.summary_memory.write_trigger_token_ids.as_deref()
    }
}

fn shannon_entropy(probabilities: &[f32]) -> f64 {
    let total = probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(f64::from)
        .sum::<f64>();
    if total <= 0.0 {
        return 0.0;
    }

    probabilities
        .iter()
        .copied()
        .filter(|value| *value > 0.0)
        .map(|value| {
            let value = value as f64 / total;
            -value * value.ln()
        })
        .sum()
}

fn average_language_mhc_diagnostics(
    diagnostics_runs: Vec<Vec<LanguageMhcLayerDiagnostics>>,
) -> Vec<LanguageMhcLayerDiagnostics> {
    use std::collections::BTreeMap;

    let mut accumulators = BTreeMap::<usize, LanguageMhcLayerDiagnosticsAccumulator>::new();
    for run in diagnostics_runs {
        for diag in run {
            let accumulator = accumulators.entry(diag.layer_index).or_default();
            accumulator.count += 1;
            accumulator.num_streams = diag.num_streams;
            accumulator.stream_norm_mean_sum += diag.stream_norm_mean;
            accumulator.stream_norm_variance_sum += diag.stream_norm_variance;
            if let Some(value) = diag.pairwise_stream_cosine_mean {
                accumulator.pairwise_stream_cosine_mean_sum += value;
                accumulator.pairwise_stream_cosine_mean_count += 1;
            }
            accumulator.alpha_entropy_mean_sum += diag.alpha_entropy_mean;
            accumulator.alpha_entropy_normalized_mean_sum += diag.alpha_entropy_normalized_mean;
            if let Some(value) = diag.beta_entropy_mean {
                accumulator.beta_entropy_mean_sum += value;
                accumulator.beta_entropy_mean_count += 1;
            }
            if let Some(value) = diag.beta_entropy_normalized_mean {
                accumulator.beta_entropy_normalized_mean_sum += value;
                accumulator.beta_entropy_normalized_mean_count += 1;
            }
            accumulator.residual_distance_identity_l1_mean_sum +=
                diag.residual_distance_identity_l1_mean;
            accumulator.residual_distance_uniform_l1_mean_sum +=
                diag.residual_distance_uniform_l1_mean;
        }
    }

    accumulators
        .into_iter()
        .map(|(layer_index, accumulator)| {
            let count = accumulator.count.max(1) as f64;
            LanguageMhcLayerDiagnostics {
                layer_index,
                num_streams: accumulator.num_streams,
                stream_norm_mean: accumulator.stream_norm_mean_sum / count,
                stream_norm_variance: accumulator.stream_norm_variance_sum / count,
                pairwise_stream_cosine_mean: (accumulator.pairwise_stream_cosine_mean_count > 0)
                    .then_some(
                        accumulator.pairwise_stream_cosine_mean_sum
                            / accumulator.pairwise_stream_cosine_mean_count as f64,
                    ),
                alpha_entropy_mean: accumulator.alpha_entropy_mean_sum / count,
                alpha_entropy_normalized_mean: accumulator.alpha_entropy_normalized_mean_sum
                    / count,
                beta_entropy_mean: (accumulator.beta_entropy_mean_count > 0).then_some(
                    accumulator.beta_entropy_mean_sum / accumulator.beta_entropy_mean_count as f64,
                ),
                beta_entropy_normalized_mean: (accumulator.beta_entropy_normalized_mean_count > 0)
                    .then_some(
                        accumulator.beta_entropy_normalized_mean_sum
                            / accumulator.beta_entropy_normalized_mean_count as f64,
                    ),
                residual_distance_identity_l1_mean: accumulator
                    .residual_distance_identity_l1_mean_sum
                    / count,
                residual_distance_uniform_l1_mean: accumulator
                    .residual_distance_uniform_l1_mean_sum
                    / count,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LatentFanoutScheduleConfig;
    use crate::model::sequence::mamba::MambaSequenceConfig;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn::tensor::{Int, TensorData};
    use burn_ndarray::NdArray;

    type RecurrenceBackend = NdArray<f32>;

    fn recurrence_test_model_with_shape(
        kernel: SequenceKernelKind,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        latent_total: usize,
        vocab_size: usize,
    ) -> BDH<RecurrenceBackend> {
        let device = <RecurrenceBackend as BackendTrait>::Device::default();
        <RecurrenceBackend as BackendTrait>::seed(&device, 2026);
        assert_eq!(
            latent_total % n_embd,
            0,
            "latent_total must be divisible by n_embd in recurrence tests"
        );
        BDH::<RecurrenceBackend>::new(
            BDHConfig {
                n_layer,
                n_embd,
                n_head,
                mlp_internal_dim_multiplier: latent_total / n_embd,
                vocab_size,
                dropout: 0.0,
                sequence_kernel: kernel,
                fused_kernels: FusedKernelConfig {
                    enabled: false,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        )
    }

    fn recurrence_test_tokens_with_shape(
        device: &<RecurrenceBackend as BackendTrait>::Device,
        values: Vec<i64>,
        shape: [usize; 2],
    ) -> Tensor<RecurrenceBackend, 2, Int> {
        Tensor::<RecurrenceBackend, 2, Int>::from_data(TensorData::new(values, shape), device)
    }

    fn tensor_max_abs_diff<const D: usize>(
        lhs: Tensor<RecurrenceBackend, D>,
        rhs: Tensor<RecurrenceBackend, D>,
    ) -> f32 {
        let lhs_vec = lhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("lhs vec");
        let rhs_vec = rhs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rhs vec");
        lhs_vec
            .iter()
            .zip(rhs_vec.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max)
    }

    fn option_tensor_max_abs_diff<const D: usize>(
        lhs: &Option<Tensor<RecurrenceBackend, D>>,
        rhs: &Option<Tensor<RecurrenceBackend, D>>,
    ) -> f32 {
        match (lhs, rhs) {
            (None, None) => 0.0,
            (Some(lhs), Some(rhs)) => tensor_max_abs_diff(lhs.clone(), rhs.clone()),
            _ => f32::INFINITY,
        }
    }

    fn model_state_max_abs_diff(
        lhs: &ModelState<RecurrenceBackend>,
        rhs: &ModelState<RecurrenceBackend>,
    ) -> f32 {
        if lhs.position != rhs.position || lhs.layers.len() != rhs.layers.len() {
            return f32::INFINITY;
        }

        let mut max_diff = 0.0f32;
        for (lhs_layer, rhs_layer) in lhs.layers.iter().zip(rhs.layers.iter()) {
            max_diff = max_diff.max(option_tensor_max_abs_diff(&lhs_layer.rho, &rhs_layer.rho));
            max_diff = max_diff.max(option_tensor_max_abs_diff(
                &lhs_layer.rho_norm,
                &rhs_layer.rho_norm,
            ));
            max_diff = max_diff.max(option_tensor_max_abs_diff(
                &lhs_layer.sequence_aux,
                &rhs_layer.sequence_aux,
            ));
            max_diff = max_diff.max(option_tensor_max_abs_diff(
                &lhs_layer.y_neuron_state,
                &rhs_layer.y_neuron_state,
            ));
            max_diff = max_diff.max(option_tensor_max_abs_diff(
                &lhs_layer.clocked_slow_hidden,
                &rhs_layer.clocked_slow_hidden,
            ));
            max_diff = max_diff.max(option_tensor_max_abs_diff(
                &lhs_layer.summary_memory_hidden,
                &rhs_layer.summary_memory_hidden,
            ));
        }
        max_diff
    }

    fn assert_full_forward_matches_token_step_recurrence(kernel: SequenceKernelKind) {
        let tokens = vec![1i64, 2, 3, 4, 5, 6];
        assert_full_forward_matches_token_step_recurrence_with_shape(
            kernel,
            2,
            8,
            2,
            32,
            32,
            tokens,
            [1, 6],
        );
    }

    fn assert_full_forward_matches_token_step_recurrence_with_shape(
        kernel: SequenceKernelKind,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        latent_total: usize,
        vocab_size: usize,
        token_values: Vec<i64>,
        token_shape: [usize; 2],
    ) {
        let device = <RecurrenceBackend as BackendTrait>::Device::default();
        let model = recurrence_test_model_with_shape(
            kernel,
            n_layer,
            n_embd,
            n_head,
            latent_total,
            vocab_size,
        );
        let tokens = recurrence_test_tokens_with_shape(&device, token_values, token_shape);

        let logits_full = model.forward(tokens.clone());
        let mut recurrent_state = model.init_state();
        let mut logits_steps = Vec::new();
        for step in 0..tokens.shape().dims::<2>()[1] {
            let step_tokens = tokens.clone().slice_dim(1, step..step + 1);
            logits_steps.push(model.forward_with_state(step_tokens, &mut recurrent_state));
        }
        let logits_stepwise = Tensor::cat(logits_steps, 1);
        let max_diff = tensor_max_abs_diff(logits_full, logits_stepwise);

        assert!(
            max_diff <= 1.0e-4,
            "expected full forward and token-step recurrence to match for {kernel:?}, max diff {max_diff}"
        );
    }

    fn assert_chunked_recurrence_matches_uninterrupted_state(kernel: SequenceKernelKind) {
        let tokens = vec![1i64, 2, 3, 4, 5, 6];
        assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
            kernel,
            2,
            8,
            2,
            32,
            32,
            tokens,
            [1, 6],
            &[1, 2, 3],
        );
    }

    fn assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
        kernel: SequenceKernelKind,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        latent_total: usize,
        vocab_size: usize,
        token_values: Vec<i64>,
        token_shape: [usize; 2],
        chunk_sizes: &[usize],
    ) {
        let device = <RecurrenceBackend as BackendTrait>::Device::default();
        let model = recurrence_test_model_with_shape(
            kernel,
            n_layer,
            n_embd,
            n_head,
            latent_total,
            vocab_size,
        );
        let tokens = recurrence_test_tokens_with_shape(&device, token_values, token_shape);
        let logits_full = model.forward(tokens.clone());

        let mut uninterrupted_state = model.init_state();
        let logits_uninterrupted =
            model.forward_with_state(tokens.clone(), &mut uninterrupted_state);
        let uninterrupted_diff = tensor_max_abs_diff(logits_full.clone(), logits_uninterrupted);
        assert!(
            uninterrupted_diff <= 1.0e-4,
            "expected uninterrupted stateful forward to match full forward for {kernel:?}, max diff {uninterrupted_diff}"
        );

        for &chunk_tokens in chunk_sizes {
            let mut chunked_state = model.init_state();
            let mut logits_chunks = Vec::new();
            let seq_len = tokens.shape().dims::<2>()[1];
            for chunk_start in (0..seq_len).step_by(chunk_tokens) {
                let chunk_end = (chunk_start + chunk_tokens).min(seq_len);
                let chunk = tokens.clone().slice_dim(1, chunk_start..chunk_end);
                logits_chunks.push(model.forward_with_state(chunk, &mut chunked_state));
            }
            let logits_chunked = Tensor::cat(logits_chunks, 1);
            let logits_diff = tensor_max_abs_diff(logits_full.clone(), logits_chunked);
            let state_diff = model_state_max_abs_diff(&uninterrupted_state, &chunked_state);

            assert!(
                logits_diff <= 1.0e-4,
                "expected chunked recurrence to match full forward for {kernel:?} chunk_tokens={chunk_tokens}, max diff {logits_diff}"
            );
            assert!(
                state_diff <= 1.0e-4,
                "expected chunked recurrence state to match uninterrupted state for {kernel:?} chunk_tokens={chunk_tokens}, max diff {state_diff}"
            );
        }
    }

    fn repeat_value_heads_for_recurrence(
        value: Tensor<RecurrenceBackend, 4>,
        heads: usize,
    ) -> Tensor<RecurrenceBackend, 4> {
        match value.shape().dims::<4>()[1] {
            1 => value.repeat_dim(1, heads),
            existing if existing == heads => value,
            existing => panic!("value heads {existing} must be 1 or {heads}"),
        }
    }

    fn exclusive_prefix_sum_time_5d(
        tensor: Tensor<RecurrenceBackend, 5>,
    ) -> Tensor<RecurrenceBackend, 5> {
        let [batch, heads, time, latent, embd] = tensor.shape().dims::<5>();
        let prefix = tensor.cumsum(2);
        let zero = Tensor::<RecurrenceBackend, 5>::zeros(
            [batch, heads, 1, latent, embd],
            &prefix.device(),
        );
        if time == 1 {
            zero
        } else {
            Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
        }
    }

    fn exclusive_prefix_sum_time_4d(
        tensor: Tensor<RecurrenceBackend, 4>,
    ) -> Tensor<RecurrenceBackend, 4> {
        let [batch, heads, time, latent] = tensor.shape().dims::<4>();
        let prefix = tensor.cumsum(2);
        let zero =
            Tensor::<RecurrenceBackend, 4>::zeros([batch, heads, 1, latent], &prefix.device());
        if time == 1 {
            zero
        } else {
            Tensor::cat(vec![zero, prefix.slice_dim(2, 0..time - 1)], 2)
        }
    }

    fn recurrent_attention_tensorized_no_decay_reference(
        query: Tensor<RecurrenceBackend, 4>,
        value: Tensor<RecurrenceBackend, 4>,
    ) -> (Tensor<RecurrenceBackend, 4>, Tensor<RecurrenceBackend, 4>) {
        let [batch, heads, time, latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let value = repeat_value_heads_for_recurrence(value, heads);

        let delta = query.clone().unsqueeze_dim::<5>(4) * value.clone().unsqueeze_dim::<5>(3);
        let rho_before = exclusive_prefix_sum_time_5d(delta.clone());
        let context = (rho_before.clone() * query.clone().unsqueeze_dim::<5>(4))
            .sum_dim(3)
            .reshape([batch, heads, time, embd]);
        let rho = delta
            .cumsum(2)
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads, latent, embd]);

        (context, rho)
    }

    fn recurrent_attention_dense_score_reference(
        query: Tensor<RecurrenceBackend, 4>,
        value: Tensor<RecurrenceBackend, 4>,
        decay: Option<Tensor<RecurrenceBackend, 1>>,
    ) -> (Tensor<RecurrenceBackend, 4>, Tensor<RecurrenceBackend, 4>) {
        let [batch, heads, time, latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let device = query.device();
        let value = repeat_value_heads_for_recurrence(value, heads);

        let pos_row = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);
        let pos_col = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, 1, time]);

        let mut scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
        let rho = if let Some(decay) = decay {
            let diff = (pos_row.clone() - pos_col.clone())
                .tril(-1)
                .repeat_dim(1, heads);
            let decay_score = decay
                .clone()
                .reshape([1, heads, 1, 1])
                .repeat_dim(2, time)
                .repeat_dim(3, time);
            scores = scores * decay_score.powf(diff);

            let final_exponents = pos_row
                .clone()
                .mul_scalar(-1.0)
                .add_scalar(time as f32)
                .repeat_dim(1, heads);
            let decay_final = decay
                .reshape([1, heads, 1, 1])
                .repeat_dim(2, time)
                .powf(final_exponents);
            query.mul(decay_final).swap_dims(2, 3).matmul(value.clone())
        } else {
            query.swap_dims(2, 3).matmul(value.clone())
        };

        let context = scores.matmul(value).reshape([batch, heads, time, embd]);
        assert_eq!(rho.shape().dims::<4>(), [batch, heads, latent, embd]);

        (context, rho)
    }

    fn recurrent_rwkv8_tensorized_reference(
        query: Tensor<RecurrenceBackend, 4>,
        value: Tensor<RecurrenceBackend, 4>,
        decay: Tensor<RecurrenceBackend, 3>,
    ) -> (
        Tensor<RecurrenceBackend, 4>,
        Tensor<RecurrenceBackend, 4>,
        Tensor<RecurrenceBackend, 3>,
    ) {
        let [batch, heads, time, latent] = query.shape().dims::<4>();
        let embd = value.shape().dims::<4>()[3];
        let device = query.device();
        let value = repeat_value_heads_for_recurrence(value, heads);
        let time_idx = Tensor::<RecurrenceBackend, 1, Int>::arange(0..time as i64, &device).float();

        let delta = query.clone().unsqueeze_dim::<5>(4) * value.clone().unsqueeze_dim::<5>(3);

        let decay5 = decay
            .clone()
            .reshape([1, heads, 1, latent, 1])
            .repeat_dim(2, time);
        let state_exp5 = time_idx
            .clone()
            .reshape([1, 1, time, 1, 1])
            .repeat_dim(1, heads)
            .repeat_dim(3, latent);
        let inv_exp5 = time_idx
            .clone()
            .add_scalar(1.0)
            .mul_scalar(-1.0)
            .reshape([1, 1, time, 1, 1])
            .repeat_dim(1, heads)
            .repeat_dim(3, latent);
        let rho_before =
            exclusive_prefix_sum_time_5d(delta.clone() * decay5.clone().powf(inv_exp5))
                * decay5.clone().powf(state_exp5);

        let decay4 = decay
            .clone()
            .reshape([1, heads, 1, latent])
            .repeat_dim(2, time);
        let state_exp4 = time_idx
            .clone()
            .reshape([1, 1, time, 1])
            .repeat_dim(1, heads)
            .repeat_dim(3, latent);
        let inv_exp4 = time_idx
            .clone()
            .add_scalar(1.0)
            .mul_scalar(-1.0)
            .reshape([1, 1, time, 1])
            .repeat_dim(1, heads)
            .repeat_dim(3, latent);
        let rho_norm_before =
            exclusive_prefix_sum_time_4d(query.clone() * decay4.clone().powf(inv_exp4))
                * decay4.clone().powf(state_exp4);

        let q_weights = query.clone().div(
            query
                .clone()
                .sum_dim(3)
                .add_scalar(1.0e-6)
                .reshape([batch, heads, time, 1]),
        );
        let context = rho_before
            .clone()
            .div(
                rho_norm_before
                    .clone()
                    .add_scalar(1.0e-6)
                    .unsqueeze_dim::<5>(4),
            )
            .mul(q_weights.unsqueeze_dim::<5>(4))
            .sum_dim(3)
            .reshape([batch, heads, time, embd]);

        let last_rho_before = rho_before
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads, latent, embd]);
        let last_rho_norm_before = rho_norm_before
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads, latent]);
        let last_delta = delta
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads, latent, embd]);
        let last_query = query
            .slice_dim(2, time - 1..time)
            .reshape([batch, heads, latent]);

        let rho = last_rho_before
            .mul(decay.clone().reshape([1, heads, latent, 1]))
            .add(last_delta);
        let rho_norm = last_rho_norm_before
            .mul(decay.reshape([1, heads, latent]))
            .add(last_query);

        (context, rho, rho_norm)
    }

    fn deterministic_y_neuron_recurrence_model_with_layers(
        recurrence: YNeuronRecurrenceConfig,
        n_layer: usize,
    ) -> BDH<NdArray<f32>> {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let mut model = BDH::<Backend>::new(
            BDHConfig {
                n_layer,
                n_embd: 2,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 8,
                dropout: 0.0,
                y_neuron_recurrence: recurrence,
                ..Default::default()
            },
            &device,
        );

        model.encoder = Param::from_tensor(Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 2, 2]),
            &device,
        ));
        model.encoder_v = Param::from_tensor(Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![1.0, 0.0, 0.0, 1.0], [1, 2, 2]),
            &device,
        ));
        model.decoder = Param::from_tensor(Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![0.0, 0.0, 1.0, 0.5], [2, 2]),
            &device,
        ));

        model
    }

    fn deterministic_y_neuron_recurrence_model(
        recurrence: YNeuronRecurrenceConfig,
    ) -> BDH<NdArray<f32>> {
        deterministic_y_neuron_recurrence_model_with_layers(recurrence, 1)
    }

    #[test]
    fn recurrent_attention_reference_matches_outer_product_state_space_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 8,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![2.0, 3.0, 5.0, 7.0], [1, 1, 2, 2]),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![11.0, 13.0, 17.0, 19.0], [1, 1, 2, 2]),
            &device,
        );

        let (context, rho) = model.recurrent_attention_reference(query, value, None, None);

        let context_vec = context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("context vec");
        let rho_vec = rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rho vec");

        assert_eq!(context_vec, vec![0.0, 0.0, 341.0, 403.0]);
        assert_eq!(rho_vec, vec![107.0, 121.0, 152.0, 172.0]);
    }

    #[test]
    fn recurrent_rwkv8_state_space_reference_matches_decayed_normalized_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 8,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![2.0, 0.0, 1.0, 3.0], [1, 1, 2, 2]),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![10.0, 20.0, 30.0, 40.0], [1, 1, 2, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.5, 0.5], [2]), &device)
            .reshape([1, 1, 2]);

        let (context, rho, rho_norm) =
            model.recurrent_rwkv8_state_space_reference(query, value, None, None, decay);

        let context_vec = context
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("context vec");
        let rho_vec = rho
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rho vec");
        let rho_norm_vec = rho_norm
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rho norm vec");

        for (actual, expected) in context_vec.iter().zip([0.0, 0.0, 2.5, 5.0]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        for (actual, expected) in rho_vec.iter().zip([40.0, 60.0, 90.0, 120.0]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
        for (actual, expected) in rho_norm_vec.iter().zip([2.0, 3.0]) {
            assert!((actual - expected).abs() < 1.0e-4);
        }
    }

    #[test]
    fn rwkv8_forward_with_state_populates_rho_norm() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 2,
                n_embd: 8,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let mut state = model.init_state();
        let _ = model.forward_with_state(tokens, &mut state);

        assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
        assert!(state.layers.iter().all(|layer| layer.rho_norm.is_some()));
    }

    #[test]
    fn mamba_forward_with_state_populates_sequence_aux() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 2,
                n_embd: 8,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 32,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::MambaSelectiveSsmExperimental,
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let mut state = model.init_state();
        let _ = model.forward_with_state(tokens, &mut state);

        assert!(state.layers.iter().all(|layer| layer.rho.is_some()));
        assert!(
            state
                .layers
                .iter()
                .all(|layer| layer.sequence_aux.is_some())
        );
        assert!(state.layers.iter().all(|layer| layer.rho_norm.is_none()));
    }

    #[test]
    fn linear_full_forward_matches_token_step_recurrence() {
        assert_full_forward_matches_token_step_recurrence(SequenceKernelKind::BdhLinearAttention);
    }

    #[test]
    fn rwkv8_full_forward_matches_token_step_recurrence() {
        assert_full_forward_matches_token_step_recurrence(
            SequenceKernelKind::Rwkv8StateSpaceExperimental,
        );
    }

    #[test]
    fn mamba_full_forward_matches_token_step_recurrence() {
        assert_full_forward_matches_token_step_recurrence(
            SequenceKernelKind::MambaSelectiveSsmExperimental,
        );
    }

    #[test]
    fn linear_chunked_recurrence_matches_uninterrupted_state_and_logits() {
        assert_chunked_recurrence_matches_uninterrupted_state(
            SequenceKernelKind::BdhLinearAttention,
        );
    }

    #[test]
    fn linear_dense_score_full_forward_matches_token_step_recurrence() {
        assert_full_forward_matches_token_step_recurrence(
            SequenceKernelKind::BdhLinearDenseScoreExperimental,
        );
    }

    #[test]
    fn linear_dense_score_chunked_recurrence_matches_uninterrupted_state_and_logits() {
        assert_chunked_recurrence_matches_uninterrupted_state(
            SequenceKernelKind::BdhLinearDenseScoreExperimental,
        );
    }

    #[test]
    fn rwkv8_chunked_recurrence_matches_uninterrupted_state_and_logits() {
        assert_chunked_recurrence_matches_uninterrupted_state(
            SequenceKernelKind::Rwkv8StateSpaceExperimental,
        );
    }

    #[test]
    fn mamba_chunked_recurrence_matches_uninterrupted_state_and_logits() {
        assert_chunked_recurrence_matches_uninterrupted_state(
            SequenceKernelKind::MambaSelectiveSsmExperimental,
        );
    }

    #[test]
    fn linear_multi_head_recurrence_matches_token_step_and_chunked_state() {
        let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_full_forward_matches_token_step_recurrence_with_shape(
            SequenceKernelKind::BdhLinearAttention,
            3,
            12,
            3,
            72,
            64,
            token_values.clone(),
            [2, 5],
        );
        assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
            SequenceKernelKind::BdhLinearAttention,
            3,
            12,
            3,
            72,
            64,
            token_values,
            [2, 5],
            &[1, 2, 4],
        );
    }

    #[test]
    fn linear_dense_score_multi_head_recurrence_matches_token_step_and_chunked_state() {
        let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_full_forward_matches_token_step_recurrence_with_shape(
            SequenceKernelKind::BdhLinearDenseScoreExperimental,
            3,
            12,
            3,
            72,
            64,
            token_values.clone(),
            [2, 5],
        );
        assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
            SequenceKernelKind::BdhLinearDenseScoreExperimental,
            3,
            12,
            3,
            72,
            64,
            token_values,
            [2, 5],
            &[1, 2, 4],
        );
    }

    #[test]
    fn rwkv8_multi_head_recurrence_matches_token_step_and_chunked_state() {
        let token_values = vec![1i64, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_full_forward_matches_token_step_recurrence_with_shape(
            SequenceKernelKind::Rwkv8StateSpaceExperimental,
            3,
            12,
            3,
            72,
            64,
            token_values.clone(),
            [2, 5],
        );
        assert_chunked_recurrence_matches_uninterrupted_state_with_shape(
            SequenceKernelKind::Rwkv8StateSpaceExperimental,
            3,
            12,
            3,
            72,
            64,
            token_values,
            [2, 5],
            &[1, 2, 4],
        );
    }

    #[test]
    fn linear_tensorized_parallel_reference_matches_host_loop_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 3.0, 2.0, 1.0, 0.5,
                    1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5, 2.0,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );

        let (context_host, rho_host) =
            model.recurrent_attention_reference(query.clone(), value.clone(), None, None);
        let (context_tensorized, rho_tensorized) =
            recurrent_attention_tensorized_no_decay_reference(query, value);

        assert!(tensor_max_abs_diff(context_host, context_tensorized) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, rho_tensorized) <= 1.0e-4);
    }

    #[test]
    fn rwkv8_tensorized_parallel_reference_matches_host_loop_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                    2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
            &device,
        );

        let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );
        let (context_tensorized, rho_tensorized, rho_norm_tensorized) =
            recurrent_rwkv8_tensorized_reference(query, value, decay);

        assert!(tensor_max_abs_diff(context_host, context_tensorized) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, rho_tensorized) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_norm_host, rho_norm_tensorized) <= 1.0e-4);
    }

    #[test]
    fn rwkv8_kernel_tensorized_forward_matches_host_loop_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                    2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
            &device,
        );

        let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );
        let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);

        assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
    }

    #[test]
    fn rwkv8_kernel_scan_fallback_matches_host_loop_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                    2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
            &device,
        );

        let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );

        unsafe {
            std::env::set_var(
                "BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES",
                "1",
            )
        };
        let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);
        unsafe {
            std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES")
        };

        assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
    }

    #[test]
    fn rwkv8_kernel_matmul_fallback_matches_host_loop_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                sequence_kernel: SequenceKernelKind::Rwkv8StateSpaceExperimental,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 1.5, 0.5, 2.0, 1.0, 0.5, 1.5, 1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5,
                    2.0, 0.75, 1.25, 1.0, 2.0, 2.5, 1.5, 0.75, 1.25,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.95, 0.9, 0.85, 0.8, 0.9, 0.85, 0.8, 0.75], [1, 2, 4]),
            &device,
        );

        let (context_host, rho_host, rho_norm_host) = model.recurrent_rwkv8_state_space_reference(
            query.clone(),
            value.clone(),
            None,
            None,
            decay.clone(),
        );

        unsafe {
            std::env::set_var(
                "BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES",
                "1",
            );
            std::env::set_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_MATMUL_MAX_CHUNK", "8");
        };
        let tensorized = tensorized_rwkv8_forward(query, value, None, None, decay);
        unsafe {
            std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_SCAN_THRESHOLD_BYTES");
            std::env::remove_var("BURN_DRAGON_RWKV8_TENSORIZED_FORWARD_MATMUL_MAX_CHUNK");
        };

        assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, tensorized.rho) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_norm_host, tensorized.rho_norm) <= 1.0e-4);
    }

    #[test]
    fn mamba_kernel_tensorized_forward_matches_reference() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let config = MambaSequenceConfig::default().resolve(8);
        let params = MambaSequenceParameters::<Backend>::new(config, &device);
        let hidden = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..(2 * 1 * 5 * 8))
                    .map(|idx| ((idx % 19) as f32) / 19.0 - 0.3)
                    .collect::<Vec<_>>(),
                [2, 1, 5, 8],
            ),
            &device,
        );

        let (context_host, state_host) = mamba_reference(hidden.clone(), &params, None);
        let tensorized = tensorized_mamba_forward(
            hidden,
            config.d_inner,
            config.d_state,
            config.d_conv,
            config.dt_rank,
            params.in_proj_tensor(),
            params.conv_weight_tensor(),
            params.conv_bias_tensor(),
            params.x_proj_tensor(),
            params.dt_proj_weight_tensor(),
            params.dt_proj_bias_tensor(),
            params.a_log_tensor(),
            params.d_skip_tensor(),
            params.out_proj_tensor(),
            None,
        );

        assert!(tensor_max_abs_diff(context_host, tensorized.context) <= 1.0e-4);
        assert!(tensor_max_abs_diff(state_host.conv, tensorized.state.conv) <= 1.0e-4);
        assert!(tensor_max_abs_diff(state_host.ssm, tensorized.state.ssm) <= 1.0e-4);
    }

    #[test]
    fn linear_dense_score_reference_matches_host_loop_reference_with_decay() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 2,
                n_head: 2,
                mlp_internal_dim_multiplier: 2,
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );

        let query = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    1.0, 2.0, 3.0, 4.0, 2.0, 1.0, 1.5, 0.5, 0.5, 1.5, 2.5, 3.5, 3.0, 2.0, 1.0, 0.5,
                    1.25, 0.75, 2.25, 1.75, 0.25, 1.0, 1.5, 2.0,
                ],
                [1, 2, 3, 4],
            ),
            &device,
        );
        let value = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![1.0, 0.5, 0.25, 1.5, 2.0, 1.0], [1, 1, 3, 2]),
            &device,
        );
        let decay = Tensor::<Backend, 1>::from_data(TensorData::new(vec![0.9, 0.8], [2]), &device);

        let (context_host, rho_host) = model.recurrent_attention_reference(
            query.clone(),
            value.clone(),
            None,
            Some(decay.clone()),
        );
        let (context_dense, rho_dense) =
            recurrent_attention_dense_score_reference(query, value, Some(decay));

        assert!(tensor_max_abs_diff(context_host, context_dense) <= 1.0e-4);
        assert!(tensor_max_abs_diff(rho_host, rho_dense) <= 1.0e-4);
    }

    #[test]
    fn bdh_mhc_two_view_wrapper_matches_manual_layer_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 1,
                    num_views: 2,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let mut state = model.init_state();
        let (hidden, _logits) = model.forward_with_hidden_and_state(tokens.clone(), &mut state);

        let embedded = model.embed.forward(tokens);
        let [batch, time, dim] = embedded.shape().dims::<3>();
        let current = model.norm.forward(embedded.reshape([batch, 1, time, dim]));
        let mhc = model.mhc_for_layer(0).expect("mhc");
        let coeffs = mhc.coefficients();
        let (branch_input, residuals_base, beta) =
            mhc_split_with_coefficients(Some(mhc), current, Some(&coeffs));
        let [branch_batch, branch_views, branch_time, branch_dim] =
            branch_input.shape().dims::<4>();
        let branch_flat =
            branch_input.reshape([branch_batch * branch_views, 1, branch_time, branch_dim]);

        let encoder =
            model
                .encoder
                .val()
                .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
        let encoder_v =
            model
                .encoder_v
                .val()
                .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
        let decoder = model.decoder.val();
        let mut layer_state = LayerState {
            rho: None,
            rho_norm: None,
            sequence_aux: None,
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
            #[cfg(feature = "viz")]
            viz: None,
        };
        let output = lowrank_residual_step(
            branch_flat,
            encoder,
            encoder_v,
            decoder,
            &model.dropout,
            false,
            false,
            0.0,
            true,
            &model.kernel.block_sparse.latent,
            model.kernel.lowrank_grad_input_executor,
            None,
            |query, value| {
                model.recurrent_attention_with_plan(
                    query,
                    value,
                    &mut layer_state,
                    0,
                    RecurrentPositionMode::Sequential,
                    None,
                )
            },
            activation::relu,
            |values| model.norm.forward(values),
        );
        let branch_out = output
            .next
            .reshape([branch_batch, branch_views, branch_time, branch_dim]);
        let manual = model
            .norm
            .forward(mhc_merge_with_coefficients(
                Some(mhc),
                branch_out,
                residuals_base,
                Some(&coeffs),
                beta,
            ))
            .reshape([batch, time, dim]);

        let hidden_vec = hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hidden vec");
        let manual_vec = manual
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("manual vec");

        assert_eq!(hidden_vec, manual_vec);
    }

    #[test]
    fn bdh_mhc_dynamic_stream_wrapper_matches_manual_layer_contract() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 2,
                    num_views: 1,
                    coefficient_policy:
                        super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let mut state = model.init_state();
        let (hidden, _logits) = model.forward_with_hidden_and_state(tokens.clone(), &mut state);

        let embedded = model.embed.forward(tokens);
        let [batch, time, dim] = embedded.shape().dims::<3>();
        let current = model.norm.forward(embedded.reshape([batch, 1, time, dim]));
        let mhc = model.mhc_for_layer(0).expect("mhc");
        let connector = model.residual_connector_for_layer(0);
        let current_residuals = model.prepare_language_residuals(current, &connector);
        let stream_output = mhc.stream_width_connection(current_residuals);
        let branch_input = stream_output.branch_input.clone();
        let encoder =
            model
                .encoder
                .val()
                .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
        let encoder_v =
            model
                .encoder_v
                .val()
                .reshape([1, 1, dim, model.mlp_internal_dim_multiplier * dim]);
        let decoder = model.decoder.val();
        let mut layer_state = LayerState {
            rho: None,
            rho_norm: None,
            sequence_aux: None,
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
            #[cfg(feature = "viz")]
            viz: None,
        };
        let output = lowrank_residual_step(
            branch_input.clone(),
            encoder,
            encoder_v,
            decoder,
            &model.dropout,
            false,
            false,
            0.0,
            true,
            &model.kernel.block_sparse.latent,
            model.kernel.lowrank_grad_input_executor,
            None,
            |query, value| {
                model.recurrent_attention_with_plan(
                    query,
                    value,
                    &mut layer_state,
                    0,
                    RecurrentPositionMode::Sequential,
                    None,
                )
            },
            activation::relu,
            |values| model.norm.forward(values),
        );
        let manual =
            model.collapse_language_streams(model.norm.forward(mhc.stream_depth_connection(
                output.next,
                stream_output.residuals_out,
                &stream_output.coefficients,
            )));

        let hidden_vec = hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("hidden vec");
        let manual_vec = manual
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("manual vec");

        assert_eq!(hidden_vec, manual_vec);
    }

    #[test]
    fn bdh_language_mhc_diagnostics_report_non_uniform_stream_behavior() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 3,
                    num_views: 1,
                    coefficient_policy:
                        super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);

        let diagnostics = model.collect_language_mhc_diagnostics(tokens);
        let layer0 = diagnostics.first().expect("layer diagnostics");

        assert_eq!(layer0.layer_index, 0);
        assert_eq!(layer0.num_streams, 3);
        assert!(layer0.stream_norm_mean.is_finite());
        assert!(layer0.alpha_entropy_normalized_mean < 1.0);
        assert!(
            layer0
                .pairwise_stream_cosine_mean
                .is_some_and(|value| value.is_finite()),
            "pairwise cosine should be finite for multi-stream diagnostics"
        );
        assert!(layer0.residual_distance_identity_l1_mean > 0.0);
    }

    #[test]
    fn bdh_mhc_single_stream_single_view_skips_per_layer_mhc_allocation() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 2,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 1,
                    num_views: 1,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let output = model.forward(tokens);
        let [batch, time, vocab] = output.shape().dims::<3>();

        assert!(model.mhc_shared.is_none());
        assert_eq!([batch, time, vocab], [1, 3, 16]);
    }

    #[test]
    fn bdh_mhc_multi_stream_language_contract_collapses_back_to_hidden() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 2,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 2,
                    num_views: 1,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        let tokens =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![1, 2, 3], [1, 3]), &device);
        let output = model.forward(tokens);
        let [batch, time, vocab] = output.shape().dims::<3>();

        assert!(model.mhc_shared.is_some());
        assert_eq!([batch, time, vocab], [1, 3, 16]);
    }

    #[test]
    fn bdh_mhc_shared_wrapper_respects_last_layers_gate() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 4,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 2,
                    num_views: 1,
                    last_layers: Some(1),
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );
        assert!(model.mhc_shared.is_some());
        assert_eq!(model.mhc_first_layer, 3);
        assert!(model.mhc_for_layer(0).is_none());
        assert!(model.mhc_for_layer(1).is_none());
        assert!(model.mhc_for_layer(2).is_none());
        assert!(model.mhc_for_layer(3).is_some());
    }

    #[test]
    fn bdh_mhc_reuses_same_wrapper_weights_across_active_layers() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 3,
                n_embd: 8,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                mhc: super::super::mhc::ManifoldHyperConnectionsConfig {
                    enabled: true,
                    num_streams: 2,
                    num_views: 1,
                    coefficient_policy:
                        super::super::mhc::ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
                    mhc_iters: 4,
                    mhc_tau: 0.1,
                    add_branch_out_to_residual: true,
                    dropout: 0.0,
                    ..Default::default()
                },
                ..Default::default()
            },
            &device,
        );

        let layer0 = model.mhc_for_layer(0).expect("layer 0 mhc");
        let layer1 = model.mhc_for_layer(1).expect("layer 1 mhc");
        let layer2 = model.mhc_for_layer(2).expect("layer 2 mhc");

        assert!(std::ptr::eq(layer0, layer1));
        assert!(std::ptr::eq(layer1, layer2));
    }

    #[test]
    fn bdh_layer_lowrank_weights_follow_latent_fanout_schedule() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 4,
                n_embd: 8,
                n_head: 2,
                mlp_internal_dim_multiplier: 8,
                latent_fanout_schedule: Some(LatentFanoutScheduleConfig::LateLayer {
                    base_latent_total: 16,
                    last_layers: 2,
                }),
                vocab_size: 16,
                dropout: 0.0,
                ..Default::default()
            },
            &device,
        );

        let (encoder0, encoder_v0, decoder0, latent0) = model.layer_lowrank_weights(0);
        let (encoder3, encoder_v3, decoder3, latent3) = model.layer_lowrank_weights(3);

        assert_eq!(latent0, 8);
        assert_eq!(encoder0.shape().dims::<4>(), [1, 2, 8, 8]);
        assert_eq!(encoder_v0.shape().dims::<4>(), [1, 2, 8, 8]);
        assert_eq!(decoder0.shape().dims::<2>(), [16, 8]);

        assert_eq!(latent3, 32);
        assert_eq!(encoder3.shape().dims::<4>(), [1, 2, 8, 32]);
        assert_eq!(encoder_v3.shape().dims::<4>(), [1, 2, 8, 32]);
        assert_eq!(decoder3.shape().dims::<2>(), [64, 8]);
    }

    #[test]
    fn y_neuron_recurrence_persists_across_calls_and_changes_next_token() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let prefix = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
            &device,
        );
        let suffix =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![0.0, 6.0], [1, 1, 2]), &device);

        let mut baseline =
            deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig::default());
        let mut baseline_state = baseline.init_state();
        let _ =
            baseline.forward_with_hidden_and_state_embedded(prefix.clone(), &mut baseline_state);
        assert!(baseline_state.layers[0].y_neuron_state.is_none());
        let (baseline_hidden, _) =
            baseline.forward_with_hidden_and_state_embedded(suffix.clone(), &mut baseline_state);

        baseline.y_neuron_recurrence = YNeuronRecurrenceConfig {
            enabled: true,
            carry_in_scale: 0.5,
            last_layers: None,
            chunk_tokens: 1,
            state_decay: 1.0,
            state_update_scale: 1.0,
            state_rms_cap: None,
        };
        let mut recurrent_state = baseline.init_state();
        let _ = baseline.forward_with_hidden_and_state_embedded(prefix, &mut recurrent_state);
        let carried_state = recurrent_state.layers[0]
            .y_neuron_state
            .as_ref()
            .expect("recurrent y_neuron state");
        let carried_vec = carried_state
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("carried y_neuron state vec");
        assert!(
            carried_vec.iter().any(|value| value.abs() > 1.0e-5),
            "expected non-zero carried y_neuron state"
        );
        let (recurrent_hidden, _) =
            baseline.forward_with_hidden_and_state_embedded(suffix, &mut recurrent_state);

        let baseline_vec = baseline_hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("baseline suffix hidden vec");
        let recurrent_vec = recurrent_hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("recurrent suffix hidden vec");
        let diff: f32 = baseline_vec
            .iter()
            .zip(recurrent_vec.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum();
        assert!(
            diff > 1.0e-4,
            "expected y_neuron recurrence to change the next-token hidden state, got diff {diff}"
        );
    }

    #[test]
    fn y_neuron_recurrence_state_rms_cap_bounds_carried_state() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let embedded = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.0, 2.0, 0.0, 4.0, 0.0, 6.0], [1, 3, 2]),
            &device,
        );
        let cap = 0.25f32;
        let model = deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig {
            enabled: true,
            carry_in_scale: 0.5,
            last_layers: None,
            chunk_tokens: 1,
            state_decay: 1.0,
            state_update_scale: 4.0,
            state_rms_cap: Some(cap),
        });
        let mut state = model.init_state();
        let _ = model.forward_with_hidden_and_state_embedded(embedded, &mut state);
        let carried_state = state.layers[0]
            .y_neuron_state
            .as_ref()
            .expect("bounded carried y_neuron state")
            .clone();
        let rms = carried_state
            .clone()
            .powf_scalar(2.0)
            .mean_dim(2)
            .sqrt()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rms vec");
        assert!(
            rms.iter().all(|value| *value <= cap + 1.0e-4),
            "expected carried y_neuron state rms <= {cap}, got {rms:?}"
        );
    }

    #[test]
    fn y_neuron_recurrence_chunked_mode_persists_across_calls_and_changes_next_token() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let prefix = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
            &device,
        );
        let suffix =
            Tensor::<Backend, 3>::from_data(TensorData::new(vec![0.0, 6.0], [1, 1, 2]), &device);

        let mut baseline =
            deterministic_y_neuron_recurrence_model(YNeuronRecurrenceConfig::default());
        let mut baseline_state = baseline.init_state();
        let _ =
            baseline.forward_with_hidden_and_state_embedded(prefix.clone(), &mut baseline_state);
        let (baseline_hidden, _) =
            baseline.forward_with_hidden_and_state_embedded(suffix.clone(), &mut baseline_state);

        baseline.y_neuron_recurrence = YNeuronRecurrenceConfig {
            enabled: true,
            carry_in_scale: 0.5,
            last_layers: None,
            chunk_tokens: 2,
            state_decay: 1.0,
            state_update_scale: 1.0,
            state_rms_cap: None,
        };
        let mut recurrent_state = baseline.init_state();
        let _ = baseline.forward_with_hidden_and_state_embedded(prefix, &mut recurrent_state);
        let carried_state = recurrent_state.layers[0]
            .y_neuron_state
            .as_ref()
            .expect("chunked recurrent y_neuron state");
        let carried_vec = carried_state
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("chunked carried state vec");
        assert!(
            carried_vec.iter().any(|value| value.abs() > 1.0e-5),
            "expected non-zero carried y_neuron state in chunked mode"
        );
        let (recurrent_hidden, _) =
            baseline.forward_with_hidden_and_state_embedded(suffix, &mut recurrent_state);

        let baseline_vec = baseline_hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("baseline suffix hidden vec");
        let recurrent_vec = recurrent_hidden
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("chunked recurrent suffix hidden vec");
        let diff: f32 = baseline_vec
            .iter()
            .zip(recurrent_vec.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum();
        assert!(
            diff > 1.0e-4,
            "expected chunked y_neuron recurrence to change the next-token hidden state, got diff {diff}"
        );
    }

    #[test]
    fn y_neuron_recurrence_last_layers_only_updates_top_layers() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let embedded = Tensor::<Backend, 3>::from_data(
            TensorData::new(vec![0.0, 2.0, 0.0, 4.0], [1, 2, 2]),
            &device,
        );
        let model = deterministic_y_neuron_recurrence_model_with_layers(
            YNeuronRecurrenceConfig {
                enabled: true,
                carry_in_scale: 0.5,
                last_layers: Some(1),
                chunk_tokens: 1,
                state_decay: 1.0,
                state_update_scale: 1.0,
                state_rms_cap: None,
            },
            2,
        );
        let mut state = model.init_state();
        let _ = model.forward_with_hidden_and_state_embedded(embedded, &mut state);
        assert!(
            state.layers[0].y_neuron_state.is_none(),
            "non-recurrent lower layers should not carry y_neuron state"
        );
        let top_state = state.layers[1]
            .y_neuron_state
            .as_ref()
            .expect("top recurrent layer should carry y_neuron state");
        let top_vec = top_state
            .clone()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("top layer carried state vec");
        assert!(
            top_vec.iter().any(|value| value.abs() > 1.0e-5),
            "expected non-zero carried y_neuron state on the recurrent top layer"
        );
    }

    #[test]
    fn summary_memory_reads_previous_chunk_instead_of_self_summary() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 4,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                summary_memory: SummaryMemoryConfig {
                    enabled: true,
                    last_layers: Some(1),
                    chunk_tokens: 2,
                    residual_scale: 0.5,
                    state_decay: 1.0,
                    state_update_scale: 1.0,
                    surprise_gate_threshold: 0.0,
                    surprise_gate_sharpness: 8.0,
                    write_trigger_text: None,
                    write_trigger_token_ids: None,
                },
                ..Default::default()
            },
            &device,
        );

        let mut layer_state = LayerState {
            rho: None,
            rho_norm: None,
            sequence_aux: None,
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
            #[cfg(feature = "viz")]
            viz: None,
        };
        let first_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0, 4.0], [1, 1, 2, 1]), &device);
        let first_out = model.forward_branch_summary_memory(first_chunk, &mut layer_state, 0, None);
        let first_vec = first_out
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("first chunk vec");
        assert_eq!(first_vec, vec![2.0, 4.0]);

        let second_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
        let second_out =
            model.forward_branch_summary_memory(second_chunk, &mut layer_state, 2, None);
        let second_vec = second_out
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("second chunk vec");
        assert_eq!(second_vec, vec![11.5]);
    }

    #[test]
    fn summary_memory_surprise_gate_preserves_prior_carry_when_chunk_is_unsurprising() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 4,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                summary_memory: SummaryMemoryConfig {
                    enabled: true,
                    last_layers: Some(1),
                    chunk_tokens: 1,
                    residual_scale: 0.5,
                    state_decay: 1.0,
                    state_update_scale: 1.0,
                    surprise_gate_threshold: 0.5,
                    surprise_gate_sharpness: 8.0,
                    write_trigger_text: None,
                    write_trigger_token_ids: None,
                },
                ..Default::default()
            },
            &device,
        );

        let mut layer_state = LayerState {
            rho: None,
            rho_norm: None,
            sequence_aux: None,
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
            #[cfg(feature = "viz")]
            viz: None,
        };

        let first_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0], [1, 1, 1, 1]), &device);
        let _ = model.forward_branch_summary_memory(first_chunk, &mut layer_state, 0, None);

        let nearly_same_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.01], [1, 1, 1, 1]), &device);
        let _ = model.forward_branch_summary_memory(nearly_same_chunk, &mut layer_state, 1, None);

        let probe_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
        let probe_out = model.forward_branch_summary_memory(probe_chunk, &mut layer_state, 2, None);
        let probe_vec = probe_out
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("probe vec");
        assert!(
            (probe_vec[0] - 11.0).abs() < 1.0e-3,
            "expected prior carry to remain near 2.0, got {:?}",
            probe_vec
        );
    }

    #[test]
    fn summary_memory_write_trigger_updates_only_on_event_chunks() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let model = BDH::<Backend>::new(
            BDHConfig {
                n_layer: 1,
                n_embd: 4,
                n_head: 1,
                mlp_internal_dim_multiplier: 1,
                vocab_size: 16,
                dropout: 0.0,
                summary_memory: SummaryMemoryConfig {
                    enabled: true,
                    last_layers: Some(1),
                    chunk_tokens: 2,
                    residual_scale: 0.5,
                    state_decay: 1.0,
                    state_update_scale: 1.0,
                    surprise_gate_threshold: 0.0,
                    surprise_gate_sharpness: 8.0,
                    write_trigger_text: None,
                    write_trigger_token_ids: Some(vec![7]),
                },
                ..Default::default()
            },
            &device,
        );

        let mut layer_state = LayerState {
            rho: None,
            rho_norm: None,
            sequence_aux: None,
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
            #[cfg(feature = "viz")]
            viz: None,
        };
        let first_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![2.0, 4.0], [1, 1, 2, 1]), &device);
        let no_event_mask =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![0i64, 0], [1, 2]), &device);
        let _ = model.forward_branch_summary_memory(
            first_chunk,
            &mut layer_state,
            0,
            Some(no_event_mask),
        );

        let second_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![10.0], [1, 1, 1, 1]), &device);
        let no_update_out =
            model.forward_branch_summary_memory(second_chunk.clone(), &mut layer_state, 2, None);
        let no_update_vec = no_update_out
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("no update vec");
        assert!(
            (no_update_vec[0] - 10.0).abs() < 1.0e-4,
            "expected no summary carry before a trigger, got {:?}",
            no_update_vec
        );

        let event_chunk =
            Tensor::<Backend, 4>::from_data(TensorData::new(vec![6.0, 8.0], [1, 1, 2, 1]), &device);
        let event_mask =
            Tensor::<Backend, 2, Int>::from_data(TensorData::new(vec![0i64, 1], [1, 2]), &device);
        let _ =
            model.forward_branch_summary_memory(event_chunk, &mut layer_state, 2, Some(event_mask));

        let probe_out =
            model.forward_branch_summary_memory(second_chunk, &mut layer_state, 4, None);
        let probe_vec = probe_out
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("probe vec");
        assert!(
            probe_vec[0] > 10.0,
            "expected trigger-gated summary carry to affect the next chunk, got {:?}",
            probe_vec
        );
    }
}
