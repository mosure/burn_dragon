use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig, Embedding, EmbeddingConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Distribution as TensorDistribution, Int, Tensor, TensorData, activation};
use rand::distributions::{Distribution, WeightedIndex};
use rand::prelude::*;
use std::cmp::Ordering;

use super::attention::Attention;
use super::config::{BDHConfig, FusedKernelConfig};
use super::init::{
    near_critical_embedding_initializer, near_critical_projection_std,
    near_critical_residual_output_std,
};
use super::residual_stream::lowrank_residual_step;
#[cfg(feature = "viz")]
use super::state::LayerVizState;
use super::state::{LayerState, ModelState};

const LAYER_NORM_EPS: f32 = 1e-5;

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
    embed: Embedding<B>,
    dropout: Dropout,
    attention: Attention<B>,
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
            embed,
            dropout,
            attention,
            encoder,
            encoder_v,
            decoder,
            lm_head,
        }
    }

    fn layer_norm<const D: usize>(&self, tensor: Tensor<B, D>) -> Tensor<B, D> {
        let (var, mean) = tensor.clone().var_mean_bias(D - 1);
        tensor.sub(mean).div(var.add_scalar(LAYER_NORM_EPS).sqrt())
    }

    pub fn forward(&self, tokens: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let mut state = self.init_state();
        self.forward_with_state(tokens, &mut state)
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
            && burn_dragon_wgpu::supports_recurrent_backend::<B>()
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

    fn recurrent_attention(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        layer_state: &mut LayerState<B>,
        position: usize,
        position_mode: RecurrentPositionMode,
    ) -> Tensor<B, 4> {
        let query = match position_mode {
            RecurrentPositionMode::Sequential => self.attention.rotate_positions(query, position),
            RecurrentPositionMode::Fixed => self.attention.rotate_positions_fixed(query, position),
        };
        let decay = self.attention.alibi_decay();
        let initial_rho = layer_state.rho.as_ref().cloned();
        let device = query.device();

        if self.kernel.enabled && self.kernel.wgpu_recurrent_kernel {
            if let Some(output) = burn_dragon_wgpu::try_fused_recurrent_attention_wgpu(
                &query,
                &value,
                initial_rho.as_ref(),
                decay.as_ref(),
            ) {
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

    fn forward_with_state_impl(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let embedded = self.embed.forward(tokens);
        self.forward_with_state_from_embedded(embedded, state)
    }

    fn forward_with_state_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.forward_with_state_from_embedded_single_pass(
                embedded,
                state,
                start_pos,
                true,
                RecurrentPositionMode::Sequential,
            );
        }

        match self.rollout_executor_mode() {
            RolloutExecutorMode::HostLoop => {
                self.forward_with_state_from_embedded_rollout_host_loop(embedded, state)
            }
            RolloutExecutorMode::WgpuFused => {
                self.forward_with_state_from_embedded_rollout_fused(embedded, state)
            }
        }
    }

    fn forward_with_state_from_embedded_rollout_host_loop(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
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
            let start_pos = state.position;
            let hidden_rollout = self.forward_hidden_with_state_from_embedded_single_pass(
                rollout_embedded,
                state,
                start_pos,
                false,
                RecurrentPositionMode::Fixed,
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
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        let hidden = self.forward_hidden_with_state_from_embedded_single_pass(
            embedded,
            state,
            start_pos,
            advance_position,
            position_mode,
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
    ) -> Tensor<B, 3> {
        assert_eq!(
            state.layers.len(),
            self.n_layer,
            "model state layers mismatch"
        );
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let mut current = embedded.reshape([batch, 1, time, embd]);
        current = self.layer_norm(current);

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

        for layer_state in &mut state.layers {
            let output = lowrank_residual_step(
                current,
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
                    self.recurrent_attention(query, value, layer_state, start_pos, position_mode)
                },
                |values| activation::relu(values),
                |values| self.layer_norm(values),
            );

            #[cfg(feature = "viz")]
            let mixed = output.y_neuron.clone().swap_dims(1, 2);
            #[cfg(feature = "viz")]
            let [batch, time, heads, latent] = mixed.shape().dims();

            #[cfg(feature = "viz")]
            if time > 0 {
                let last = time - 1;
                let x_neuron_last = output
                    .x_neuron
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([batch, heads, latent])
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_gate_last = output
                    .y_gate
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([batch, heads, latent])
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let y_neuron_last = output
                    .y_neuron
                    .clone()
                    .slice_dim(2, last..time)
                    .reshape([batch, heads, latent])
                    .slice_dim(0, 0..1)
                    .reshape([heads, latent]);
                let device = x_neuron_last.device();
                let rho_last = match layer_state.rho.as_ref() {
                    Some(rho) => {
                        let dims = rho.shape().dims::<4>();
                        if dims == [batch, heads, latent, self.n_embd] {
                            let rho_energy = rho
                                .clone()
                                .abs()
                                .sum_dim(3)
                                .div_scalar(self.n_embd as f32)
                                .reshape([batch, heads, latent])
                                .sum_dim(0)
                                .div_scalar(batch as f32);
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

            current = output.next;
        }

        let [batch, _, time, dim] = current.shape().dims();
        let hidden = current.reshape([batch, time, dim]);
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
        let (_hidden, logits) = self.forward_with_state_impl(tokens, state);
        logits
    }

    pub fn forward_with_hidden_and_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_impl(tokens, state)
    }

    pub fn forward_with_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> Tensor<B, 3> {
        let (_hidden, logits) = self.forward_with_state_from_embedded(embedded, state);
        logits
    }

    pub fn forward_with_hidden_and_state_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 3>, Tensor<B, 3>) {
        self.forward_with_state_from_embedded(embedded, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::backend::Backend as BackendTrait;
    use burn_ndarray::NdArray;

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
}
