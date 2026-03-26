use super::*;

impl<B: Backend> BDH<B> {
    pub(super) fn forward_with_state_from_embedded_single_pass(
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

    pub(super) fn forward_hidden_with_state_from_embedded_single_pass(
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

    pub(super) fn initialize_language_pipeline_state(
        &self,
        embedded: Tensor<B, 3>,
    ) -> LanguagePipelineState<B> {
        let [batch, time, embd] = embedded.shape().dims::<3>();
        let current = self.norm.forward(embedded.reshape([batch, 1, time, embd]));
        LanguagePipelineState {
            current: current.clone(),
            residual_history: self.initialize_language_residual_history(&current),
        }
    }

    pub(super) fn forward_language_pipeline_state_layer_range(
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
            let current_before = pipeline_state
                .residual_history
                .capture_previous(&pipeline_state.current);
            let mhc_coefficients = match connector {
                ResidualConnectorRef::Mhc(_) => static_mhc_coefficients.as_ref(),
                ResidualConnectorRef::Vanilla
                | ResidualConnectorRef::AttentionResidual(_)
                | ResidualConnectorRef::BlockAttentionResidual(_) => None,
            };
            let bindings = self.split_language_residuals_for_layer(
                pipeline_state.current,
                &connector,
                pipeline_state.residual_history.as_slice(),
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
                self.update_language_residual_history(
                    &mut pipeline_state.residual_history,
                    current_before,
                    &pipeline_state.current,
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
            #[cfg(any(feature = "viz", feature = "probe"))]
            let output = lowrank_residual_step_branch_thresholds_relu_native(
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
            #[cfg(not(any(feature = "viz", feature = "probe")))]
            let branch_out = lowrank_residual_step_next_branch_thresholds_relu_native(
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
            )
            .reshape([branch_batch, branch_views, branch_time, branch_dim]);

            #[cfg(any(feature = "viz", feature = "probe"))]
            let mixed = output.y_neuron.clone().swap_dims(1, 2);
            #[cfg(any(feature = "viz", feature = "probe"))]
            let [flat_batch, time, heads, latent] = mixed.shape().dims();

            #[cfg(any(feature = "viz", feature = "probe"))]
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

            #[cfg(any(feature = "viz", feature = "probe"))]
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
            self.update_language_residual_history(
                &mut pipeline_state.residual_history,
                current_before,
                &pipeline_state.current,
            );
        }

        pipeline_state
    }

    pub(super) fn forward_hidden_with_state_from_embedded_single_pass_layer_limit(
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
}
