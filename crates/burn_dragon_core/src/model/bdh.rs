mod auxiliary_memory;
mod connector;
mod diagnostics;
mod language_pipeline;
mod low_bit_export;
mod sequence_dispatch;

use burn::module::{Ignored, Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData, activation};
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
use std::cmp::Ordering;
use std::ops::Range;
use std::time::Instant;

use super::attention::Attention;
use super::attention_residual::{
    AttentionResidual, BlockAttentionResidual, ResidualConnectorKind, ResidualHistory,
};
#[cfg(any(feature = "probe", test))]
use super::bdh_support::{
    LanguageBdhInitLayerDiagnostics, average_language_bdh_init_diagnostics, positive_fraction,
    rms_from_values, tensor_values_f32, values_are_finite,
};
use super::bdh_support::{
    LanguageMhcLayerBindings, LanguageMhcLayerDiagnostics, LanguagePipelineState,
    RecurrentPositionMode, ResidualConnectorRef, RolloutExecutorMode,
    average_language_mhc_diagnostics, logits_projection_profile_enabled,
    logits_projection_profile_record, shannon_entropy,
};
use super::config::{
    BDHConfig, ClockedSlowMemoryConfig, FusedKernelConfig, SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};
use super::init::{BdhFiringTargetKind, BdhInitializer, BdhProjectionRole};
use super::low_bit::{
    LowBitActivationFormat, LowBitInferenceMode, LowBitQuantizationConfig, LowBitRhoConfig,
    LowBitWeightFormat, RhoCompressionConfig,
};
use super::low_bit_runtime::{
    LowBitKernelRuntimeKind, LowBitProjectionPlan, PackedLowBitProjectionArtifacts,
    fake_quantize_activation_ste, fake_quantize_weight_ste, pack_rho_int8_block_state,
    pack_rho_int8_block_state_device, packed_decoder_tail_native, packed_decoder_tail_reference,
    packed_decoder_tail_training_native, packed_lowrank_projection_native,
    packed_lowrank_projection_reference, packed_lowrank_projection_training_native,
    resolve_low_bit_kernel_plan, unpack_rho_int8_block_state, unpack_rho_int8_block_state_device,
};
use super::norm::DragonNorm;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::LowRankResidualOutput;
#[cfg(test)]
use super::residual_stream::lowrank_residual_step;
#[cfg(any(feature = "probe", test))]
use super::residual_stream::lowrank_residual_step_with_metrics_branch_thresholds;
use super::residual_stream::{
    lowrank_residual_step_branch_thresholds_relu_native,
    lowrank_residual_step_next_branch_thresholds,
    lowrank_residual_step_next_branch_thresholds_relu_native,
};
use super::sequence::linear::{
    recurrent_attention_dense_score_final_rho_reference,
    recurrent_attention_dense_score_initial_context_reference,
    recurrent_attention_dense_score_reference, recurrent_attention_reference,
};
use super::sequence::mamba::{
    MambaReferenceState, MambaSequenceParameters, ResolvedMambaSequenceConfig, mamba_reference,
};
use super::sequence::rwkv8::recurrent_rwkv8_state_space_reference;
use super::sequence::state::{mamba_state, write_mamba_state};
use super::sequence::{SequenceKernelConfig, SequenceKernelFamily, SequenceTrainingExecutor};
#[cfg(any(feature = "viz", feature = "probe"))]
use super::state::LayerVizState;
use super::state::{LayerState, ModelState};
use super::{ManifoldHyperConnections, mhc_merge_with_coefficients, mhc_split_with_coefficients};
use crate::experimental::bitnet_reference::PackedWeightArtifact;
#[cfg(test)]
use crate::model::config::SequenceKernelKind;

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
    x_relu_threshold: f32,
    y_relu_threshold: f32,
    y_neuron_recurrence: YNeuronRecurrenceConfig,
    clocked_slow_memory: ClockedSlowMemoryConfig,
    summary_memory: SummaryMemoryConfig,
    #[module(ignore)]
    low_bit_quant: Ignored<LowBitQuantizationConfig>,
    #[module(ignore)]
    low_bit_rho: Ignored<LowBitRhoConfig>,
    #[module(ignore)]
    packed_decoder_x: Ignored<Option<PackedWeightArtifact>>,
    #[module(ignore)]
    packed_decoder_y: Ignored<Option<PackedWeightArtifact>>,
    #[module(ignore)]
    packed_encoder: Ignored<Option<PackedWeightArtifact>>,
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
        let initializer = BdhInitializer::new(&config.initialization);
        let embed = EmbeddingConfig::new(config.vocab_size, config.n_embd)
            .with_initializer(initializer.embedding_initializer(config.n_embd))
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
        let activation_thresholds =
            initializer.activation_thresholds(config.n_embd, latent_per_head, residual_depth);
        let use_shared_relu_threshold = matches!(
            config.initialization.firing_targets.kind,
            BdhFiringTargetKind::Disabled
        );
        let shared_relu_threshold = config.fused_kernels.relu_threshold;
        let encoder = Param::from_tensor(initializer.headwise_projection_tensor::<B>(
            BdhProjectionRole::Encoder,
            config.n_head,
            config.n_embd,
            latent_per_head,
            residual_depth,
            device,
        ));

        let encoder_v = Param::from_tensor(initializer.headwise_projection_tensor::<B>(
            BdhProjectionRole::EncoderValue,
            config.n_head,
            config.n_embd,
            latent_per_head,
            residual_depth,
            device,
        ));

        let decoder = Param::from_tensor(initializer.projection_tensor::<B>(
            BdhProjectionRole::Decoder,
            latent_total,
            config.n_embd,
            residual_depth,
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
        let lm_head = Param::from_tensor(initializer.projection_tensor::<B>(
            BdhProjectionRole::LmHead,
            config.n_embd,
            config.vocab_size,
            residual_depth,
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
            x_relu_threshold: if use_shared_relu_threshold {
                shared_relu_threshold
            } else {
                activation_thresholds.x
            },
            y_relu_threshold: if use_shared_relu_threshold {
                shared_relu_threshold
            } else {
                activation_thresholds.y
            },
            y_neuron_recurrence: config.y_neuron_recurrence,
            clocked_slow_memory: config.clocked_slow_memory,
            summary_memory: config.summary_memory,
            low_bit_quant: Ignored(config.quant),
            low_bit_rho: Ignored(config.rho),
            packed_decoder_x: Ignored(None),
            packed_decoder_y: Ignored(None),
            packed_encoder: Ignored(None),
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

    #[cfg(any(feature = "probe", test))]
    pub fn collect_language_bdh_init_diagnostics(
        &self,
        tokens: Tensor<B, 2, Int>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        let mut state = self.init_state();
        self.collect_language_bdh_init_diagnostics_with_state(tokens, &mut state)
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

    #[cfg(any(feature = "probe", test))]
    pub fn collect_language_bdh_init_diagnostics_with_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        let mut state = self.init_state();
        self.collect_language_bdh_init_diagnostics_with_state_and_summary_event_mask(
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

    fn available_packed_low_bit_projection_artifacts(
        &self,
    ) -> PackedLowBitProjectionArtifacts<'_, B> {
        if !matches!(
            self.low_bit_quant.0.inference_mode,
            LowBitInferenceMode::OfflinePack
        ) {
            return PackedLowBitProjectionArtifacts::default();
        }

        PackedLowBitProjectionArtifacts {
            runtime: LowBitKernelRuntimeKind::FakeQuantReference,
            x: self.packed_decoder_x.0.as_ref(),
            y: self.packed_decoder_y.0.as_ref(),
            residual: self.packed_encoder.0.as_ref(),
            _marker: core::marker::PhantomData,
        }
    }

    fn packed_low_bit_projection_artifacts(&self) -> PackedLowBitProjectionArtifacts<'_, B> {
        let artifacts = self.available_packed_low_bit_projection_artifacts();
        let kernel_plan =
            resolve_low_bit_kernel_plan::<B>(&self.low_bit_quant.0, artifacts.clone());
        PackedLowBitProjectionArtifacts {
            runtime: kernel_plan.runtime,
            ..if matches!(
                kernel_plan.runtime,
                LowBitKernelRuntimeKind::PackedReference
                    | LowBitKernelRuntimeKind::PackedNativeInference
            ) {
                artifacts
            } else {
                PackedLowBitProjectionArtifacts::default()
            }
        }
    }

    fn low_bit_projection_plan(&self) -> LowBitProjectionPlan {
        let mut plan = LowBitProjectionPlan::from_config(&self.low_bit_quant.0);
        let kernel_plan = resolve_low_bit_kernel_plan::<B>(
            &self.low_bit_quant.0,
            self.available_packed_low_bit_projection_artifacts(),
        );
        if matches!(
            kernel_plan.runtime,
            LowBitKernelRuntimeKind::PackedReference
                | LowBitKernelRuntimeKind::PackedNativeInference
        ) {
            if self.packed_decoder_x.0.is_some() {
                plan.x_weight_format = None;
            }
            if self.packed_decoder_y.0.is_some() {
                plan.y_weight_format = None;
            }
            if self.packed_encoder.0.is_some() {
                plan.residual_weight_format = None;
            }
        }
        plan
    }

    fn rho_chunk_compression_enabled(&self) -> bool {
        matches!(
            self.low_bit_rho.0.compression,
            RhoCompressionConfig::Int8BlockExp
        )
    }

    fn resolve_linear_attention_rho_state(
        &self,
        layer_state: &LayerState<B>,
        device: &B::Device,
    ) -> Option<Tensor<B, 4>> {
        if let Some(rho) = layer_state.rho.as_ref() {
            return Some(rho.clone());
        }
        if self.rho_chunk_compression_enabled() {
            if let Some(packed) = layer_state.packed_rho_int8_device.as_ref() {
                return Some(unpack_rho_int8_block_state_device(packed));
            }
            return layer_state
                .packed_rho_int8
                .as_ref()
                .map(|packed| unpack_rho_int8_block_state::<B>(packed, device));
        }
        None
    }

    fn write_linear_attention_rho_state(&self, layer_state: &mut LayerState<B>, rho: Tensor<B, 4>) {
        if self.rho_chunk_compression_enabled() {
            if resolve_low_bit_kernel_plan::<B>(
                &self.low_bit_quant.0,
                self.available_packed_low_bit_projection_artifacts(),
            )
            .capabilities
            .native_rho_int8_block_supported
            {
                layer_state.packed_rho_int8_device = Some(pack_rho_int8_block_state_device(&rho));
                layer_state.packed_rho_int8 = None;
            } else {
                layer_state.packed_rho_int8 = Some(pack_rho_int8_block_state(&rho));
                layer_state.packed_rho_int8_device = None;
            }
            layer_state.rho = None;
        } else {
            layer_state.rho = Some(rho);
            layer_state.packed_rho_int8 = None;
            layer_state.packed_rho_int8_device = None;
        }
        layer_state.rho_norm = None;
        layer_state.sequence_aux = None;
    }

    fn resolve_rwkv8_state(
        &self,
        layer_state: &LayerState<B>,
        batch: usize,
        heads: usize,
        latent: usize,
        device: &B::Device,
    ) -> super::sequence::state::Rwkv8State<B> {
        let rho_norm = match layer_state.rho_norm.as_ref() {
            Some(state) if state.shape().dims::<3>() == [batch, heads, latent] => state.clone(),
            _ => Tensor::<B, 3>::zeros([batch, heads, latent], device),
        };

        super::sequence::state::Rwkv8State {
            rho: self.resolve_linear_attention_rho_state(layer_state, device),
            rho_norm,
        }
    }

    fn write_rwkv8_sequence_state(
        &self,
        layer_state: &mut LayerState<B>,
        rho: Tensor<B, 4>,
        rho_norm: Tensor<B, 3>,
    ) {
        if self.rho_chunk_compression_enabled() {
            if resolve_low_bit_kernel_plan::<B>(
                &self.low_bit_quant.0,
                self.available_packed_low_bit_projection_artifacts(),
            )
            .capabilities
            .native_rho_int8_block_supported
            {
                layer_state.packed_rho_int8_device = Some(pack_rho_int8_block_state_device(&rho));
                layer_state.packed_rho_int8 = None;
            } else {
                layer_state.packed_rho_int8 = Some(pack_rho_int8_block_state(&rho));
                layer_state.packed_rho_int8_device = None;
            }
            layer_state.rho = None;
        } else {
            layer_state.rho = Some(rho);
            layer_state.packed_rho_int8 = None;
            layer_state.packed_rho_int8_device = None;
        }
        layer_state.rho_norm = Some(rho_norm);
        layer_state.sequence_aux = None;
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

    fn project_lowrank_positive(
        &self,
        dense: Tensor<B, 4>,
        projector: Tensor<B, 4>,
        packed_weight_artifact: Option<&PackedWeightArtifact>,
        weight_format: Option<LowBitWeightFormat>,
        activation_format: Option<LowBitActivationFormat>,
        relu_threshold: f32,
        use_fused: bool,
        latent_pattern: &crate::kernel::BlockPattern1d,
        sparse_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        if matches!(
            self.packed_low_bit_projection_artifacts().runtime,
            LowBitKernelRuntimeKind::PackedNativeTrainingForward
        ) && weight_format.is_some()
        {
            let latent_out = projector.shape().dims::<4>()[3];
            let fused_relu_threshold = (!self.low_bit_quant.0.strict_bitnet_reference).then_some(
                if relu_threshold != 0.0 {
                    relu_threshold
                } else {
                    0.0
                },
            );
            let mut projected = packed_lowrank_projection_training_native(
                dense,
                projector,
                weight_format.expect("native training forward requires low-bit weight format"),
                activation_format,
                latent_out,
                self.low_bit_quant.0.saved_activations.mode,
                fused_relu_threshold,
            );
            let activated = if fused_relu_threshold.is_some() {
                projected
            } else {
                if relu_threshold != 0.0 {
                    projected = projected.sub_scalar(relu_threshold);
                }
                activation::relu(projected)
            };
            return if !self.low_bit_quant.0.strict_bitnet_reference {
                activated
            } else if let Some(format) = activation_format {
                fake_quantize_activation_ste(activated, format)
            } else {
                activated
            };
        }

        if let Some(artifact) = packed_weight_artifact {
            let mut projected = match self.packed_low_bit_projection_artifacts().runtime {
                LowBitKernelRuntimeKind::PackedNativeInference => packed_lowrank_projection_native(
                    dense,
                    artifact,
                    activation_format,
                    projector.shape().dims::<4>()[3],
                ),
                _ => packed_lowrank_projection_reference(
                    dense,
                    artifact,
                    activation_format,
                    projector.shape().dims::<4>()[3],
                ),
            };
            if relu_threshold != 0.0 {
                projected = projected.sub_scalar(relu_threshold);
            }
            let activated = activation::relu(projected);
            return if let Some(format) = activation_format {
                fake_quantize_activation_ste(activated, format)
            } else {
                activated
            };
        }

        let dense = if let Some(format) = activation_format {
            fake_quantize_activation_ste(dense, format)
        } else {
            dense
        };
        let projector = if let Some(format) = weight_format {
            fake_quantize_weight_ste(projector, format)
        } else {
            projector
        };
        if use_fused {
            let projected = crate::kernel::relu_lowrank::fused_forward_with_executor(
                dense,
                projector,
                None,
                relu_threshold,
                latent_pattern,
                sparse_mask,
                self.kernel.lowrank_grad_input_executor,
            );
            if let Some(format) = activation_format {
                fake_quantize_activation_ste(projected, format)
            } else {
                projected
            }
        } else {
            let mut latent = dense.matmul(projector);
            if relu_threshold != 0.0 {
                latent = latent.sub_scalar(relu_threshold);
            }
            let activated = activation::relu(latent);
            if let Some(format) = activation_format {
                fake_quantize_activation_ste(activated, format)
            } else {
                activated
            }
        }
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
        let mut residual_history = self.initialize_language_residual_history(&current);

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                current,
                &connector,
                residual_history.as_slice(),
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
                let output = lowrank_residual_step_branch_thresholds_relu_native(
                    branch_flat,
                    encoder.clone(),
                    encoder_v.clone(),
                    decoder.clone(),
                    &self.dropout,
                    fused && self.kernel.projection_executor.use_x(),
                    fused && self.kernel.projection_executor.use_y(),
                    self.x_relu_threshold,
                    self.y_relu_threshold,
                    true,
                    self.low_bit_projection_plan(),
                    self.low_bit_quant.0.saved_activations.clone(),
                    self.packed_low_bit_projection_artifacts(),
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

                #[cfg(any(feature = "viz", feature = "probe"))]
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
                    let rho_last =
                        match self.resolve_linear_attention_rho_state(layer_state, &device) {
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
                self.update_language_residual_history(
                    &mut residual_history,
                    current_before,
                    &current,
                );
                continue;
            }
            let low_bit_plan = self.low_bit_projection_plan();
            let packed_artifacts = self.packed_low_bit_projection_artifacts();
            let x_base = self.project_lowrank_positive(
                branch_flat.clone(),
                encoder.clone(),
                packed_artifacts.x,
                low_bit_plan.x_weight_format,
                low_bit_plan.x_activation_format,
                self.x_relu_threshold,
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

            #[cfg(any(feature = "viz", feature = "probe"))]
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
                    packed_artifacts.y,
                    low_bit_plan.y_weight_format,
                    low_bit_plan.y_activation_format,
                    self.y_relu_threshold,
                    fused,
                    latent_pattern,
                    sparse_mask.clone(),
                );
                let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                let y_neuron = if let Some(format) = low_bit_plan.residual_activation_format {
                    fake_quantize_activation_ste(y_neuron, format)
                } else {
                    y_neuron
                };
                let mlp_out = if matches!(
                    packed_artifacts.runtime,
                    LowBitKernelRuntimeKind::PackedNativeTrainingForward
                ) && low_bit_plan.residual_weight_format.is_some()
                {
                    packed_decoder_tail_training_native(
                        y_neuron.clone(),
                        decoder.clone(),
                        low_bit_plan
                            .residual_weight_format
                            .expect("native training decoder tail requires low-bit weight format"),
                        None,
                        self.low_bit_quant.0.saved_activations.mode,
                    )
                } else if let Some(artifact) = packed_artifacts.residual {
                    match packed_artifacts.runtime {
                        LowBitKernelRuntimeKind::PackedNativeInference => {
                            packed_decoder_tail_native(y_neuron.clone(), artifact, None)
                        }
                        _ => packed_decoder_tail_reference(y_neuron.clone(), artifact, None),
                    }
                } else {
                    let mixed = y_neuron.clone().swap_dims(1, 2);
                    let mixed_flat = mixed.reshape([flat_batch * chunk_len, heads * latent]);
                    let decoder = if let Some(format) = low_bit_plan.residual_weight_format {
                        fake_quantize_weight_ste(decoder.clone(), format)
                    } else {
                        decoder.clone()
                    };
                    let mlp_flat = mixed_flat.matmul(decoder);
                    mlp_flat.reshape([flat_batch, 1, chunk_len, branch_dim])
                };
                let mlp_out = self.norm.forward(mlp_out);
                next_tokens.push(self.norm.forward(current_token + mlp_out));
                let y_neuron_last = y_neuron.clone().slice_dim(2, (chunk_len - 1)..chunk_len);
                y_neuron_state = self.update_y_neuron_state(y_neuron_state, y_neuron_last);

                #[cfg(any(feature = "viz", feature = "probe"))]
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

            #[cfg(any(feature = "viz", feature = "probe"))]
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
                let rho_last = match self.resolve_linear_attention_rho_state(layer_state, &device) {
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
            self.update_language_residual_history(&mut residual_history, current_before, &current);
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

    pub fn summary_memory_write_trigger_token_ids(&self) -> Option<&[u32]> {
        self.summary_memory.write_trigger_token_ids.as_deref()
    }
}

#[cfg(test)]
mod tests;
