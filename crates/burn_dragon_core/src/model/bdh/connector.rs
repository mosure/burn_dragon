use super::super::ManifoldHyperConnectionCoefficients;
use super::*;

impl<B: Backend> BDH<B> {
    pub(super) fn residual_connector_for_layer(
        &self,
        layer_idx: usize,
    ) -> ResidualConnectorRef<'_, B> {
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
    pub(super) fn mhc_for_layer(&self, layer_idx: usize) -> Option<&ManifoldHyperConnections<B>> {
        match self.residual_connector_for_layer(layer_idx) {
            ResidualConnectorRef::Mhc(mhc) => Some(mhc),
            _ => None,
        }
    }

    pub(super) fn prepare_language_residuals(
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

    pub(super) fn collapse_language_streams(&self, current: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch, streams, time, dim] = current.shape().dims();
        if streams == 1 {
            current.reshape([batch, time, dim])
        } else {
            current.mean_dim(1).reshape([batch, time, dim])
        }
    }

    pub(super) fn residual_connector_needs_post_merge_norm(
        &self,
        connector: &ResidualConnectorRef<'_, B>,
    ) -> bool {
        !matches!(connector, ResidualConnectorRef::Vanilla)
    }

    pub(super) fn residual_connector_uses_history(&self) -> bool {
        match self.residual_connector {
            ResidualConnectorKind::AttentionResidual => self.attention_residual_shared.is_some(),
            ResidualConnectorKind::BlockAttentionResidual => {
                self.block_attention_residual_shared.is_some()
            }
            ResidualConnectorKind::Vanilla | ResidualConnectorKind::Mhc => false,
        }
    }

    pub(super) fn initialize_language_residual_history(
        &self,
        current: &Tensor<B, 4>,
    ) -> ResidualHistory<B> {
        ResidualHistory::from_anchor_if_enabled(self.residual_connector_uses_history(), current)
    }

    pub(super) fn update_language_residual_history(
        &self,
        residual_history: &mut ResidualHistory<B>,
        previous: Option<Tensor<B, 4>>,
        current: &Tensor<B, 4>,
    ) {
        residual_history.push_delta_from(previous, current);
    }

    pub(super) fn split_language_residuals_for_layer(
        &self,
        current: Tensor<B, 4>,
        connector: &ResidualConnectorRef<'_, B>,
        residual_history: &[Tensor<B, 4>],
        mhc_coefficients: Option<&ManifoldHyperConnectionCoefficients<B>>,
    ) -> LanguageMhcLayerBindings<B> {
        let current_residuals = self.prepare_language_residuals(current.clone(), connector);
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

    pub(super) fn merge_language_residuals_for_layer(
        &self,
        branch_out: Tensor<B, 4>,
        bindings: LanguageMhcLayerBindings<B>,
        connector: &ResidualConnectorRef<'_, B>,
        mhc_coefficients: Option<&ManifoldHyperConnectionCoefficients<B>>,
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

    pub(super) fn summarize_language_mhc_layer_diagnostics(
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

    #[cfg(any(feature = "probe", test))]
    pub(super) fn unsupported_bdh_init_layer_diagnostics(
        &self,
        layer_index: usize,
        current: Tensor<B, 4>,
    ) -> LanguageBdhInitLayerDiagnostics {
        let current_values = tensor_values_f32(current);
        LanguageBdhInitLayerDiagnostics {
            layer_index,
            lowrank_path_active: false,
            finite: values_are_finite(&current_values),
            current_rms: Some(rms_from_values(&current_values)),
            ..Default::default()
        }
    }

    #[cfg(any(feature = "probe", test))]
    pub(super) fn summarize_language_bdh_init_layer_diagnostics(
        &self,
        layer_index: usize,
        current: Tensor<B, 4>,
        output: &LowRankResidualOutput<B>,
    ) -> LanguageBdhInitLayerDiagnostics {
        let current_values = tensor_values_f32(current);
        let x_values = tensor_values_f32(output.x_neuron.clone());
        let y_values = tensor_values_f32(output.y_gate.clone());
        let readout_values = tensor_values_f32(
            output
                .attention_readout
                .clone()
                .expect("attention_readout for init diagnostics"),
        );
        let residual_values = tensor_values_f32(
            output
                .residual_delta
                .clone()
                .expect("residual_delta for init diagnostics"),
        );

        let current_rms = rms_from_values(&current_values);
        let recurrent_readout_rms = rms_from_values(&readout_values);
        let residual_delta_rms = rms_from_values(&residual_values);
        let denom = current_rms.max(1.0e-12);

        LanguageBdhInitLayerDiagnostics {
            layer_index,
            lowrank_path_active: true,
            finite: values_are_finite(&current_values)
                && values_are_finite(&x_values)
                && values_are_finite(&y_values)
                && values_are_finite(&readout_values)
                && values_are_finite(&residual_values),
            p_x: Some(positive_fraction(&x_values)),
            p_y: Some(positive_fraction(&y_values)),
            current_rms: Some(current_rms),
            recurrent_readout_rms: Some(recurrent_readout_rms),
            recurrent_readout_ratio: Some(recurrent_readout_rms / denom),
            residual_delta_rms: Some(residual_delta_rms),
            r_res: Some(residual_delta_rms / denom),
        }
    }
}
