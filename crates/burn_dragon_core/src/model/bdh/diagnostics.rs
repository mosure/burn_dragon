use super::*;
use crate::model::bdh_support::LanguageMhcSplitBindings;

impl<B: Backend> BDH<B> {
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

    #[cfg(any(feature = "probe", test))]
    pub fn collect_language_bdh_init_diagnostics_with_state(
        &self,
        tokens: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        let embedded = self.embed.forward(tokens);
        self.collect_language_bdh_init_diagnostics_from_embedded(embedded, state, None)
    }

    #[cfg(any(feature = "probe", test))]
    pub fn collect_language_bdh_init_diagnostics_with_state_and_summary_event_mask(
        &self,
        tokens: Tensor<B, 2, Int>,
        summary_event_mask: Tensor<B, 2, Int>,
        state: &mut ModelState<B>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        let embedded = self.embed.forward(tokens);
        self.collect_language_bdh_init_diagnostics_from_embedded(
            embedded,
            state,
            Some(summary_event_mask),
        )
    }

    pub(super) fn collect_language_mhc_diagnostics_from_embedded(
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

    #[cfg(any(feature = "probe", test))]
    pub(super) fn collect_language_bdh_init_diagnostics_from_embedded(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
        if self.rollout_fast_steps_per_slow_step <= 1 {
            let start_pos = state.position;
            return self.collect_language_bdh_init_diagnostics_from_embedded_single_pass(
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
                    self.collect_language_bdh_init_diagnostics_from_embedded_single_pass(
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

        average_language_bdh_init_diagnostics(per_pass)
    }

    #[cfg(any(feature = "probe", test))]
    pub(super) fn collect_language_bdh_init_diagnostics_from_embedded_single_pass(
        &self,
        embedded: Tensor<B, 3>,
        state: &mut ModelState<B>,
        start_pos: usize,
        advance_position: bool,
        position_mode: RecurrentPositionMode,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Vec<LanguageBdhInitLayerDiagnostics> {
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

        let mut diagnostics = Vec::with_capacity(self.n_layer);
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
            let LanguageMhcSplitBindings {
                branch_input: split_branch_input,
                merge: merge_bindings,
            } = bindings;
            let branch_input = if !self.y_neuron_recurrence.enabled
                && self.summary_memory_applies_to_layer(layer_idx)
            {
                self.forward_branch_summary_memory(
                    split_branch_input,
                    layer_state,
                    start_pos,
                    summary_event_mask.clone(),
                )
            } else {
                layer_state.summary_memory_hidden = None;
                split_branch_input
            };

            if self.clocked_slow_memory_applies_to_layer(layer_idx) {
                diagnostics.push(
                    self.unsupported_bdh_init_layer_diagnostics(layer_idx, branch_input.clone()),
                );
                let branch_out = self.forward_branch_clocked_slow_layer(
                    layer_idx,
                    branch_input,
                    layer_state,
                    start_pos,
                    position_mode,
                );
                let next = self.merge_language_residuals_for_layer(
                    branch_out,
                    merge_bindings,
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
            layer_state.clocked_slow_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat =
                branch_input
                    .clone()
                    .reshape([flat_batch, 1, branch_time, branch_dim]);
            let (encoder, encoder_v, decoder, latent) = self.layer_lowrank_weights(layer_idx);
            let latent_pattern = &self.kernel.block_sparse.latent;
            let sparse_mask = if fused && latent_pattern.is_sparse() {
                Some(latent_pattern.mask::<B>(latent, &branch_flat.device()))
            } else {
                None
            };

            if self.y_neuron_recurrence.enabled
                && self.y_neuron_recurrence_applies_to_layer(layer_idx)
            {
                diagnostics.push(
                    self.unsupported_bdh_init_layer_diagnostics(layer_idx, branch_flat.clone()),
                );

                let heads = self.n_head;
                let low_bit_plan = self.low_bit_projection_plan();
                let packed_artifacts = self.packed_low_bit_projection_artifacts();
                let x_base = self.project_lowrank_positive(LowrankProjectionRequest {
                    dense: branch_flat.clone(),
                    projector: encoder.clone(),
                    scale_cache_kind: "decoder_x",
                    packed_weight_artifact: packed_artifacts.x,
                    weight_format: low_bit_plan.x_weight_format,
                    activation_format: low_bit_plan.x_activation_format,
                    relu_threshold: self.x_relu_threshold,
                    use_fused: fused,
                    latent_pattern,
                    sparse_mask: sparse_mask.clone(),
                });
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
                    let x_neuron =
                        self.inject_y_neuron_state(x_neuron_base, y_neuron_state.clone());
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
                    let y_gate = self.project_lowrank_positive(LowrankProjectionRequest {
                        dense: a_dense,
                        projector: encoder_v.clone(),
                        scale_cache_kind: "decoder_y",
                        packed_weight_artifact: packed_artifacts.y,
                        weight_format: low_bit_plan.y_weight_format,
                        activation_format: low_bit_plan.y_activation_format,
                        relu_threshold: self.y_relu_threshold,
                        use_fused: fused,
                        latent_pattern,
                        sparse_mask: sparse_mask.clone(),
                    });
                    let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                    let y_neuron = if let Some(format) = low_bit_plan.residual_activation_format {
                        fake_quantize_activation_ste(y_neuron, format)
                    } else {
                        y_neuron
                    };
                    let mlp_out = if let Some(artifact) = packed_artifacts.residual {
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
                    merge_bindings,
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
            let output = lowrank_residual_step_with_metrics_branch_thresholds(
                branch_flat.clone(),
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
                        start_pos,
                        position_mode,
                        fused_recurrent_plan.as_ref(),
                    )
                },
                |values| activation::relu(values),
                |values| self.norm.forward(values),
            );
            diagnostics.push(self.summarize_language_bdh_init_layer_diagnostics(
                layer_idx,
                branch_flat,
                &output,
            ));
            let branch_out =
                output
                    .next
                    .reshape([branch_batch, branch_views, branch_time, branch_dim]);
            let next = self.merge_language_residuals_for_layer(
                branch_out,
                merge_bindings,
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

        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        diagnostics
    }

    pub(super) fn collect_language_mhc_diagnostics_from_embedded_single_pass(
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
        let mut residual_history = self.initialize_language_residual_history(&current);

        let mut diagnostics = Vec::new();
        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
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
                residual_history.as_slice(),
                mhc_coefficients,
            );
            let LanguageMhcSplitBindings {
                branch_input: split_branch_input,
                merge: merge_bindings,
            } = bindings;
            let branch_input = if self.summary_memory_applies_to_layer(layer_idx) {
                self.forward_branch_summary_memory(
                    split_branch_input,
                    layer_state,
                    start_pos,
                    summary_event_mask.clone(),
                )
            } else {
                layer_state.summary_memory_hidden = None;
                split_branch_input
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
                    merge_bindings,
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
            let next = lowrank_residual_step_next_branch_thresholds(
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
                merge_bindings,
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

        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        diagnostics
    }

    pub(super) fn collect_language_mhc_diagnostics_from_embedded_single_pass_y_neuron_recurrence(
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
        let mut residual_history = self.initialize_language_residual_history(&current);

        let mut diagnostics = Vec::new();
        for (layer_idx, layer_state) in state.layers.iter_mut().enumerate() {
            let connector = self.residual_connector_for_layer(layer_idx);
            let current_before = residual_history.capture_previous(&current);
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
                residual_history.as_slice(),
                mhc_coefficients,
            );
            let LanguageMhcSplitBindings {
                branch_input,
                merge: merge_bindings,
            } = bindings;
            layer_state.clocked_slow_hidden = None;
            layer_state.summary_memory_hidden = None;

            let [branch_batch, branch_views, branch_time, branch_dim] =
                branch_input.shape().dims::<4>();
            let flat_batch = branch_batch * branch_views;
            let branch_flat = branch_input.reshape([flat_batch, 1, branch_time, branch_dim]);
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
                let next = lowrank_residual_step_next_branch_thresholds(
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
                    merge_bindings,
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
            let x_base = self.project_lowrank_positive(LowrankProjectionRequest {
                dense: branch_flat.clone(),
                projector: encoder.clone(),
                scale_cache_kind: "decoder_x",
                packed_weight_artifact: packed_artifacts.x,
                weight_format: low_bit_plan.x_weight_format,
                activation_format: low_bit_plan.x_activation_format,
                relu_threshold: self.x_relu_threshold,
                use_fused: fused,
                latent_pattern,
                sparse_mask: sparse_mask.clone(),
            });
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
                let y_gate = self.project_lowrank_positive(LowrankProjectionRequest {
                    dense: a_dense,
                    projector: encoder_v.clone(),
                    scale_cache_kind: "decoder_y",
                    packed_weight_artifact: packed_artifacts.y,
                    weight_format: low_bit_plan.y_weight_format,
                    activation_format: low_bit_plan.y_activation_format,
                    relu_threshold: self.y_relu_threshold,
                    use_fused: fused,
                    latent_pattern,
                    sparse_mask: sparse_mask.clone(),
                });
                let y_neuron = self.dropout.forward(x_neuron.clone() * y_gate.clone());
                let y_neuron = if let Some(format) = low_bit_plan.residual_activation_format {
                    fake_quantize_activation_ste(y_neuron, format)
                } else {
                    y_neuron
                };
                let mlp_out = if let Some(artifact) = packed_artifacts.residual {
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
                merge_bindings,
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

        if advance_position {
            state.position = state.position.saturating_add(time);
        }

        diagnostics
    }
}
