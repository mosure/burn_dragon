use super::*;

impl<B: Backend> BDH<B> {
    pub(super) fn rollout_executor_mode(&self) -> RolloutExecutorMode {
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

    pub(super) fn recurrent_attention_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>) {
        recurrent_attention_dense_score_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_final_rho_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_final_rho_reference(query, value, rho_state, decay)
    }

    pub(super) fn recurrent_attention_dense_score_initial_context_reference(
        &self,
        query: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        decay: Option<Tensor<B, 1>>,
        n_embd: usize,
    ) -> Tensor<B, 4> {
        recurrent_attention_dense_score_initial_context_reference(query, rho_state, decay, n_embd)
    }

    pub(super) fn recurrent_rwkv8_state_space_reference(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        rho_state: Option<Tensor<B, 4>>,
        rho_norm_state: Option<Tensor<B, 3>>,
        decay: Tensor<B, 3>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Tensor<B, 3>) {
        recurrent_rwkv8_state_space_reference(query, value, rho_state, rho_norm_state, decay)
    }

    pub(super) fn recurrent_attention_with_plan(
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
                let device = query.device();
                let initial_rho = self.resolve_linear_attention_rho_state(layer_state, &device);

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
                            self.write_linear_attention_rho_state(layer_state, rho);
                            return context;
                        }
                        self.write_linear_attention_rho_state(layer_state, output.rho);
                        return output.context;
                    }
                }

                let (context, rho) =
                    self.recurrent_attention_reference(query, value, initial_rho, decay);
                self.write_linear_attention_rho_state(layer_state, rho);
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
                let device = query.device();
                let initial_rho = self.resolve_linear_attention_rho_state(layer_state, &device);
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
                            self.write_linear_attention_rho_state(
                                layer_state,
                                reference_rho.clone() + rho - reference_rho.detach(),
                            );
                            return context;
                        }
                        self.write_linear_attention_rho_state(layer_state, rho);
                        return initial_context + fused_context;
                    }
                }
                let (context, rho) = self.recurrent_attention_dense_score_reference(
                    query,
                    value,
                    initial_rho,
                    decay,
                );
                self.write_linear_attention_rho_state(layer_state, rho);
                context
            }
            (SequenceKernelFamily::Rwkv8, SequenceTrainingExecutor::Reference) => {
                let [batch, heads, _time, latent] = query.shape().dims::<4>();
                let device = query.device();
                let initial_state =
                    self.resolve_rwkv8_state(layer_state, batch, heads, latent, &device);
                let decay = self.rwkv_decay(latent);
                if self.kernel.enabled && use_tensorized_rwkv8_forward_experimental() {
                    let output = tensorized_rwkv8_forward(
                        query,
                        value,
                        initial_state.rho,
                        Some(initial_state.rho_norm),
                        decay,
                    );
                    self.write_rwkv8_sequence_state(layer_state, output.rho, output.rho_norm);
                    return output.context;
                }
                let (context, rho, rho_norm) = self.recurrent_rwkv8_state_space_reference(
                    query,
                    value,
                    initial_state.rho,
                    Some(initial_state.rho_norm),
                    decay,
                );
                self.write_rwkv8_sequence_state(layer_state, rho, rho_norm);
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
}
