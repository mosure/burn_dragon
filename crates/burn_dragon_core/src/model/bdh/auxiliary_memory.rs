use super::*;

impl<B: Backend> BDH<B> {
    pub(super) fn resolve_y_neuron_state(
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

    pub(super) fn y_neuron_recurrence_applies_to_layer(&self, layer_idx: usize) -> bool {
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

    pub(super) fn clocked_slow_memory_applies_to_layer(&self, layer_idx: usize) -> bool {
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

    pub(super) fn summary_memory_applies_to_layer(&self, layer_idx: usize) -> bool {
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

    pub(super) fn summary_memory_uses_write_trigger(&self) -> bool {
        self.summary_memory
            .write_trigger_token_ids
            .as_ref()
            .is_some_and(|ids| !ids.is_empty())
    }

    pub(super) fn resolve_clocked_slow_hidden(
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

    pub(super) fn resolve_summary_memory_hidden(
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

    pub(super) fn stabilize_y_neuron_state(&self, y_neuron_state: Tensor<B, 3>) -> Tensor<B, 3> {
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

    pub(super) fn inject_y_neuron_state(
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

    pub(super) fn update_y_neuron_state(
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

    pub(super) fn forward_branch_clocked_slow_layer(
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
            let next = lowrank_residual_step_next_branch_thresholds(
                summary_flat,
                encoder.clone(),
                encoder_v.clone(),
                decoder.clone(),
                &self.dropout,
                fused_x,
                fused_y,
                self.x_relu_threshold,
                self.y_relu_threshold,
                true,
                self.low_bit_projection_plan(),
                self.low_bit_quant.saved_activations.clone(),
                self.packed_low_bit_projection_artifacts(),
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

    pub(super) fn forward_branch_summary_memory(
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

    pub(super) fn summary_memory_event_gate(
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

    pub(super) fn update_summary_memory_hidden(
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
}
