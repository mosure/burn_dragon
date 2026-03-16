use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use burn_dragon_wgpu::api::recurrent::{
    CompiledRecurrentAttentionPlan, supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
    try_fused_recurrent_attention_wgpu_with_plan,
};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::cmp::Ordering;

use super::attention::Attention;
use super::config::{
    BDHConfig, ClockedSlowMemoryConfig, FusedKernelConfig, SummaryMemoryConfig,
    YNeuronRecurrenceConfig,
};
use super::init::{
    near_critical_embedding_initializer, near_critical_projection_std,
    near_critical_residual_output_std,
};
use super::norm::DragonNorm;
use super::residual_stream::lowrank_residual_step;
#[cfg(feature = "viz")]
use super::state::LayerVizState;
use super::state::{LayerState, ModelState};
use super::{ManifoldHyperConnections, mhc_merge_with_coefficients, mhc_split_with_coefficients};

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

#[derive(Module, Debug)]
pub struct BDH<B: Backend> {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    mlp_internal_dim_multiplier: usize,
    vocab_size: usize,
    rollout_fast_steps_per_slow_step: usize,
    kernel: FusedKernelConfig,
    y_neuron_recurrence: YNeuronRecurrenceConfig,
    clocked_slow_memory: ClockedSlowMemoryConfig,
    summary_memory: SummaryMemoryConfig,
    embed: Embedding<B>,
    dropout: Dropout,
    norm: DragonNorm<B>,
    attention: Attention<B>,
    mhc_layers: Option<Vec<Option<ManifoldHyperConnections<B>>>>,
    encoder: Param<Tensor<B, 3>>,
    encoder_v: Param<Tensor<B, 3>>,
    decoder: Param<Tensor<B, 2>>,
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
        let mhc_layers = if config.mhc.enabled
            && (config.mhc.resolved_num_streams() > 1 || config.mhc.resolved_num_views() > 1)
        {
            let first_mhc_layer = config
                .mhc
                .last_layers
                .map(|last_layers| config.n_layer.max(1).saturating_sub(last_layers))
                .unwrap_or(0);
            Some(
                (0..config.n_layer.max(1))
                    .map(|layer_index| {
                        if layer_index >= first_mhc_layer {
                            Some(ManifoldHyperConnections::new(
                                &config.mhc,
                                layer_index,
                                device,
                            ))
                        } else {
                            None
                        }
                    })
                    .collect(),
            )
        } else {
            None
        };
        let lm_head = Param::from_tensor(Tensor::<B, 2>::random(
            [config.n_embd, config.vocab_size],
            TensorDistribution::Normal(0.0, lm_head_std),
            device,
        ));

        Self {
            n_layer: config.n_layer,
            n_embd: config.n_embd,
            n_head: config.n_head,
            mlp_internal_dim_multiplier: config.mlp_internal_dim_multiplier,
            vocab_size: config.vocab_size,
            rollout_fast_steps_per_slow_step: config.rollout_fast_steps_per_slow_step,
            kernel: config.fused_kernels,
            y_neuron_recurrence: config.y_neuron_recurrence,
            clocked_slow_memory: config.clocked_slow_memory,
            summary_memory: config.summary_memory,
            embed,
            dropout,
            norm,
            attention,
            mhc_layers,
            encoder,
            encoder_v,
            decoder,
            lm_head,
        }
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

    fn rollout_executor_mode(&self) -> RolloutExecutorMode {
        if self.kernel.enabled
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
        let [batch, heads, time, latent] = query.shape().dims();
        let n_embd = value.shape().dims::<4>()[3];
        let device = value.device();
        let decay = decay.map(|tensor| tensor.reshape([1, heads, 1, 1]));

        let mut rho = match rho_state {
            Some(existing) => {
                let dims = existing.shape().dims::<4>();
                if dims == [batch, heads, latent, n_embd] {
                    existing
                } else {
                    Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device)
                }
            }
            None => Tensor::<B, 4>::zeros([batch, heads, latent, n_embd], &device),
        };

        let mut outputs: Vec<Tensor<B, 4>> = Vec::with_capacity(time);

        for t in 0..time {
            let x_t = query.clone().slice_dim(2, t..t + 1);
            let v_t = value.clone().slice_dim(2, t..t + 1).repeat_dim(1, heads);
            let x_t_latent = x_t.swap_dims(2, 3);

            let attn_t = (rho.clone() * x_t_latent.clone())
                .sum_dim(2)
                .reshape([batch, heads, 1, n_embd]);
            outputs.push(attn_t);

            rho = rho + x_t_latent * v_t;
            if let Some(decay) = &decay {
                rho = rho * decay.clone();
            }
        }

        (Tensor::cat(outputs, 2), rho)
    }

    fn project_lowrank_positive(
        &self,
        dense: Tensor<B, 4>,
        projector: Tensor<B, 4>,
        use_fused: bool,
        latent_pattern: &crate::kernel::BlockPattern1d,
        sparse_mask: Option<Tensor<B, 4>>,
    ) -> Tensor<B, 4> {
        if use_fused {
            crate::kernel::relu_lowrank::fused_forward(
                dense,
                projector,
                None,
                self.kernel.relu_threshold,
                latent_pattern,
                sparse_mask,
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
        let query = match position_mode {
            RecurrentPositionMode::Sequential => self.attention.rotate_positions(query, position),
            RecurrentPositionMode::Fixed => self.attention.rotate_positions_fixed(query, position),
        };
        let decay = self.attention.alibi_decay();
        let initial_rho = layer_state.rho.as_ref().cloned();
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
                    // Keep fused forward values while reusing tensor-core backward semantics.
                    let (reference_context, reference_rho) = self.recurrent_attention_reference(
                        query.clone(),
                        value.clone(),
                        initial_rho,
                        decay,
                    );
                    let context =
                        reference_context.clone() + output.context - reference_context.detach();
                    let rho = reference_rho.clone() + output.rho - reference_rho.detach();
                    layer_state.rho = Some(rho);
                    return context;
                }
                layer_state.rho = Some(output.rho);
                return output.context;
            }
        }

        let (context, rho) = self.recurrent_attention_reference(query, value, initial_rho, decay);
        layer_state.rho = Some(rho);
        context
    }

    fn forward_branch_clocked_slow_layer(
        &self,
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
        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);
        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
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
            let output = lowrank_residual_step(
                summary_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused,
                self.kernel.relu_threshold,
                true,
                latent_pattern,
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
            let slow_hidden = output
                .next
                .reshape([branch_batch, branch_views, 1, branch_dim]);
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

    fn prepare_language_mhc_residuals(
        &self,
        residuals: Tensor<B, 4>,
        mhc: Option<&ManifoldHyperConnections<B>>,
    ) -> Tensor<B, 4> {
        let Some(mhc) = mhc else {
            return residuals;
        };
        let target_streams = mhc.num_streams().max(1);
        let [_, streams, _, _] = residuals.shape().dims::<4>();
        if streams == target_streams {
            residuals
        } else if streams == 1 && target_streams > 1 {
            residuals.repeat_dim(1, target_streams)
        } else {
            residuals
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
        if self.y_neuron_recurrence.enabled {
            return self.forward_hidden_with_state_from_embedded_single_pass_y_neuron_recurrence(
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
        let mut current = embedded.reshape([batch, 1, time, embd]);
        current = self.norm.forward(current);

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);
        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        let mhc_coefficients = self.mhc_layers.as_ref().map(|layers| {
            layers
                .iter()
                .map(|mhc| mhc.as_ref().map(|mhc| mhc.coefficients()))
                .collect::<Vec<_>>()
        });

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let mhc = self
                .mhc_layers
                .as_ref()
                .and_then(|layers| layers.get(layer_idx))
                .and_then(|mhc| mhc.as_ref());
            let mhc_coefficients = mhc_coefficients
                .as_ref()
                .and_then(|coefficients| coefficients.get(layer_idx))
                .and_then(|coefficients| coefficients.as_ref());
            let current_residuals = self.prepare_language_mhc_residuals(current, mhc);
            let (branch_input, residuals_base, beta) =
                mhc_split_with_coefficients(mhc, current_residuals, mhc_coefficients);
            let branch_input = if self.summary_memory_applies_to_layer(layer_idx) {
                self.forward_branch_summary_memory(
                    branch_input,
                    layer_state,
                    start_pos,
                    summary_event_mask.clone(),
                )
            } else {
                layer_state.summary_memory_hidden = None;
                branch_input
            };

            if self.clocked_slow_memory_applies_to_layer(layer_idx) {
                let branch_out = self.forward_branch_clocked_slow_layer(
                    branch_input,
                    layer_state,
                    start_pos,
                    position_mode,
                );
                let next = mhc_merge_with_coefficients(
                    mhc,
                    branch_out,
                    residuals_base,
                    mhc_coefficients,
                    beta,
                );
                current = if mhc.is_some() {
                    self.norm.forward(next)
                } else {
                    next
                };
                continue;
            }
            layer_state.clocked_slow_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let branch_flat =
                branch_input.reshape([branch_batch * branch_views, 1, branch_time, branch_dim]);
            let fused_recurrent_plan = if self.kernel.enabled
                && self.kernel.wgpu_recurrent_kernel
                && supports_recurrent_backend::<B>()
            {
                Some(CompiledRecurrentAttentionPlan::new(
                    branch_batch * branch_views,
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
                fused,
                self.kernel.relu_threshold,
                true,
                latent_pattern,
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
            let next = mhc_merge_with_coefficients(
                mhc,
                branch_out,
                residuals_base,
                mhc_coefficients,
                beta,
            );
            current = if mhc.is_some() {
                self.norm.forward(next)
            } else {
                next
            };
        }

        let hidden = self.collapse_language_streams(current);
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

        let encoder_raw = self.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);
        let decoder = self.decoder.val();
        let fused = self.kernel.enabled;
        let latent_pattern = &self.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        let mhc_coefficients = self.mhc_layers.as_ref().map(|layers| {
            layers
                .iter()
                .map(|mhc| mhc.as_ref().map(|mhc| mhc.coefficients()))
                .collect::<Vec<_>>()
        });

        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let mhc = self
                .mhc_layers
                .as_ref()
                .and_then(|layers| layers.get(layer_idx))
                .and_then(|mhc| mhc.as_ref());
            let mhc_coefficients = mhc_coefficients
                .as_ref()
                .and_then(|coefficients| coefficients.get(layer_idx))
                .and_then(|coefficients| coefficients.as_ref());
            let current_residuals = self.prepare_language_mhc_residuals(current, mhc);
            let (branch_input, residuals_base, beta) =
                mhc_split_with_coefficients(mhc, current_residuals, mhc_coefficients);
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat = branch_input.reshape([flat_batch, 1, branch_time, branch_dim]);
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
                    fused,
                    self.kernel.relu_threshold,
                    true,
                    latent_pattern,
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
                let next = mhc_merge_with_coefficients(
                    mhc,
                    branch_out,
                    residuals_base,
                    mhc_coefficients,
                    beta,
                );
                current = if mhc.is_some() {
                    self.norm.forward(next)
                } else {
                    next
                };
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
            let next = mhc_merge_with_coefficients(
                mhc,
                branch_out,
                residuals_base,
                mhc_coefficients,
                beta,
            );
            current = if mhc.is_some() {
                self.norm.forward(next)
            } else {
                next
            };
        }

        let hidden = self.collapse_language_streams(current);
        let [_batch, time, _dim] = hidden.shape().dims::<3>();
        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        hidden
    }

    fn project_hidden_to_logits(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, time, dim] = hidden.shape().dims();
        hidden
            .reshape([batch * time, dim])
            .matmul(self.lm_head.val())
            .reshape([batch, time, self.vocab_size])
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

    pub fn summary_memory_write_trigger_token_ids(&self) -> Option<&[u32]> {
        self.summary_memory.write_trigger_token_ids.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

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
        let mhc = model
            .mhc_layers
            .as_ref()
            .expect("mhc layers")
            .first()
            .and_then(|mhc| mhc.as_ref())
            .expect("first mhc");
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
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
        };
        let output = lowrank_residual_step(
            branch_flat,
            encoder,
            encoder_v,
            decoder,
            &model.dropout,
            false,
            0.0,
            true,
            &model.kernel.block_sparse.latent,
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

        assert!(model.mhc_layers.is_none());
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

        assert!(model.mhc_layers.is_some());
        assert_eq!([batch, time, vocab], [1, 3, 16]);
    }

    #[test]
    fn bdh_mhc_last_layers_only_allocates_top_layers() {
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
        let layers = model.mhc_layers.as_ref().expect("mhc layers");
        assert!(layers[0].is_none());
        assert!(layers[1].is_none());
        assert!(layers[2].is_none());
        assert!(layers[3].is_some());
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
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
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
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
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
            y_neuron_state: None,
            clocked_slow_hidden: None,
            summary_memory_hidden: None,
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
