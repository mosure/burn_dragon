use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};

use super::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionStreamCoefficients, ManifoldHyperConnectionStreamOutput,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnectionsConfig,
};

const MHC_EPS: f32 = 1e-6;
const MHC_DYNAMIC_GATE_INIT_LOGIT: f32 = -4.59512;
const MHC_SELECTED_STREAM_LOGIT: f32 = 2.0;

/// Manifold-constrained hyper-connections operating on residual streams.
#[derive(Module, Debug)]
pub struct ManifoldHyperConnections<B: Backend> {
    coefficient_policy_id: usize,
    num_streams: usize,
    num_views: usize,
    mhc_iters: usize,
    mhc_tau: f32,
    add_branch_out_to_residual: bool,
    dense_dim: Option<usize>,
    dropout: Dropout,
    h_res_logits: Param<Tensor<B, 2>>,
    h_pre_logits: Param<Tensor<B, 2>>,
    h_post_logits: Option<Param<Tensor<B, 2>>>,
    alpha_weight: Option<Param<Tensor<B, 2>>>,
    alpha_bias: Option<Param<Tensor<B, 1>>>,
    alpha_dynamic_gate: Option<Param<Tensor<B, 1>>>,
    beta_weight: Option<Param<Tensor<B, 2>>>,
    beta_bias: Option<Param<Tensor<B, 1>>>,
    beta_dynamic_gate: Option<Param<Tensor<B, 1>>>,
    carry_weight: Option<Param<Tensor<B, 2>>>,
    carry_bias: Option<Param<Tensor<B, 1>>>,
    carry_dynamic_gate: Option<Param<Tensor<B, 1>>>,
    stream_bootstrap_bias: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> ManifoldHyperConnections<B> {
    pub fn new(
        config: &ManifoldHyperConnectionsConfig,
        layer_index: usize,
        device: &B::Device,
    ) -> Self {
        Self::new_with_dense_dim(config, layer_index, None, device)
    }

    pub fn new_with_dense_dim(
        config: &ManifoldHyperConnectionsConfig,
        layer_index: usize,
        dense_dim: Option<usize>,
        device: &B::Device,
    ) -> Self {
        let num_streams = config.resolved_num_streams();
        let num_views = config.resolved_num_views();
        let mut h_res = vec![-8.0f32; num_streams * num_streams];
        for idx in 0..num_streams {
            h_res[idx * num_streams + idx] = 0.0;
        }
        let h_res_logits = Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(h_res.clone(), [num_streams, num_streams]),
            device,
        ));

        let init_idx = layer_index % num_streams;
        let mut h_pre = vec![-8.0f32; num_views * num_streams];
        for view_idx in 0..num_views {
            h_pre[view_idx * num_streams + init_idx] = MHC_SELECTED_STREAM_LOGIT;
        }
        let h_pre_logits = Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(h_pre, [num_views, num_streams]),
            device,
        ));

        let h_post_logits = if config.add_branch_out_to_residual {
            let mut h_post = vec![-8.0f32; num_views * num_streams];
            for view_idx in 0..num_views {
                h_post[view_idx * num_streams + init_idx] = MHC_SELECTED_STREAM_LOGIT;
            }
            Some(Param::from_tensor(Tensor::<B, 2>::from_data(
                TensorData::new(h_post, [num_views, num_streams]),
                device,
            )))
        } else {
            None
        };

        let controller_dim = dense_dim.unwrap_or(0).max(1) * num_streams.max(1);
        let dynamic_policy = config.coefficient_policy.uses_dynamic_stream_controller();
        let controller_dense_dim = if dynamic_policy {
            Some(
                dense_dim
                    .expect("dynamic_positive mHC requires the BDH dense dimension at init time")
                    .max(1),
            )
        } else {
            dense_dim
        };
        let zero_weight = || Tensor::<B, 2>::zeros([controller_dim, num_streams], device);
        let zero_bias = || Tensor::<B, 1>::zeros([num_streams], device);

        let alpha_weight = dynamic_policy.then(|| Param::from_tensor(zero_weight()));
        let alpha_bias = dynamic_policy.then(|| Param::from_tensor(zero_bias()));
        let alpha_dynamic_gate = dynamic_policy.then(|| {
            Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(vec![MHC_DYNAMIC_GATE_INIT_LOGIT], [1]),
                device,
            ))
        });
        let beta_weight = (dynamic_policy && config.add_branch_out_to_residual)
            .then(|| Param::from_tensor(zero_weight()));
        let beta_bias = (dynamic_policy && config.add_branch_out_to_residual)
            .then(|| Param::from_tensor(zero_bias()));
        let beta_dynamic_gate = (dynamic_policy && config.add_branch_out_to_residual).then(|| {
            Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(vec![MHC_DYNAMIC_GATE_INIT_LOGIT], [1]),
                device,
            ))
        });
        let carry_weight = dynamic_policy.then(|| {
            Param::from_tensor(Tensor::<B, 2>::zeros(
                [controller_dim, num_streams * num_streams],
                device,
            ))
        });
        let carry_bias = dynamic_policy.then(|| {
            Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(h_res, [num_streams * num_streams]),
                device,
            ))
        });
        let carry_dynamic_gate = dynamic_policy.then(|| {
            Param::from_tensor(Tensor::<B, 1>::from_data(
                TensorData::new(vec![MHC_DYNAMIC_GATE_INIT_LOGIT], [1]),
                device,
            ))
        });
        let stream_bootstrap_bias = if dynamic_policy && num_streams > 1 {
            controller_dense_dim.map(|dense_dim| {
                let center = (num_streams.saturating_sub(1) as f32) * 0.5;
                let values = (0..num_streams)
                    .flat_map(|stream_idx| {
                        (0..dense_dim).map(move |dim_idx| {
                            let stream_term = (stream_idx as f32 - center) / num_streams as f32;
                            let dim_phase = (dim_idx % 7) as f32 / 6.0 - 0.5;
                            1e-3 * stream_term * dim_phase
                        })
                    })
                    .collect::<Vec<_>>();
                Param::from_tensor(Tensor::<B, 2>::from_data(
                    TensorData::new(values, [num_streams, dense_dim]),
                    device,
                ))
            })
        } else {
            None
        };

        Self {
            coefficient_policy_id: match config.coefficient_policy {
                ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn => 0,
                ManifoldHyperConnectionCoefficientPolicy::DynamicPositive => 1,
            },
            num_streams,
            num_views,
            mhc_iters: config.resolved_iters(),
            mhc_tau: config.resolved_tau(),
            add_branch_out_to_residual: config.add_branch_out_to_residual,
            dense_dim: controller_dense_dim,
            dropout: DropoutConfig::new(config.dropout).init(),
            h_res_logits,
            h_pre_logits,
            h_post_logits,
            alpha_weight,
            alpha_bias,
            alpha_dynamic_gate,
            beta_weight,
            beta_bias,
            beta_dynamic_gate,
            carry_weight,
            carry_bias,
            carry_dynamic_gate,
            stream_bootstrap_bias,
        }
    }

    pub fn coefficient_policy(&self) -> ManifoldHyperConnectionCoefficientPolicy {
        match self.coefficient_policy_id {
            0 => ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
            1 => ManifoldHyperConnectionCoefficientPolicy::DynamicPositive,
            _ => ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
        }
    }

    pub fn num_streams(&self) -> usize {
        self.num_streams
    }

    pub fn num_views(&self) -> usize {
        self.num_views
    }

    pub fn dense_dim(&self) -> Option<usize> {
        self.dense_dim
    }

    fn sinkhorn(&self, logits: Tensor<B, 2>) -> Tensor<B, 2> {
        let [rows, cols] = logits.shape().dims::<2>();
        debug_assert_eq!(rows, cols);
        let mut z = logits.div_scalar(self.mhc_tau.max(MHC_EPS));
        for _ in 0..self.mhc_iters {
            z = activation::log_softmax(z, 1);
            z = activation::log_softmax(z, 0);
        }
        z.exp()
    }

    fn sinkhorn_batched(&self, logits: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, rows, cols] = logits.shape().dims::<3>();
        debug_assert_eq!(rows, cols);
        let mut z = logits
            .reshape([batch, rows, cols])
            .div_scalar(self.mhc_tau.max(MHC_EPS));
        for _ in 0..self.mhc_iters {
            z = activation::log_softmax(z, 2);
            z = activation::log_softmax(z, 1);
        }
        z.exp()
    }

    fn coefficients_static_sinkhorn(&self) -> ManifoldHyperConnectionCoefficients<B> {
        let residual_weights = self.sinkhorn(self.h_res_logits.val());
        let branch_input_weights = activation::softmax(self.h_pre_logits.val(), 1).swap_dims(0, 1);
        let branch_output_weights = self
            .h_post_logits
            .as_ref()
            .map(|param| activation::softmax(param.val(), 1));
        ManifoldHyperConnectionCoefficients {
            residual_weights,
            branch_input_weights,
            branch_output_weights,
        }
    }

    pub fn coefficients(&self) -> ManifoldHyperConnectionCoefficients<B> {
        self.coefficients_static_sinkhorn()
    }

    fn controller_features(&self, residuals: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch, streams, time, dim] = residuals.shape().dims::<4>();
        debug_assert_eq!(streams, self.num_streams);
        if let Some(expected_dim) = self.dense_dim {
            debug_assert_eq!(dim, expected_dim);
        }
        let flat = residuals
            .swap_dims(1, 2)
            .reshape([batch * time, streams * dim.max(1)]);
        let rms = flat
            .clone()
            .powf_scalar(2.0)
            .mean_dim(1)
            .sqrt()
            .clamp_min(MHC_EPS)
            .reshape([batch * time, 1]);
        flat / rms
    }

    fn positive_sigmoid(&self, values: Tensor<B, 2>) -> Tensor<B, 2> {
        activation::sigmoid(values)
    }

    fn dynamic_gate_value(&self, gate: &Param<Tensor<B, 1>>) -> Tensor<B, 2> {
        activation::sigmoid(gate.val()).reshape([1, 1])
    }

    fn static_stream_logits(&self, weights: Tensor<B, 2>) -> Tensor<B, 2> {
        weights.slice_dim(0, 0..1).reshape([1, self.num_streams])
    }

    fn stream_coefficients_static(
        &self,
        residuals: Tensor<B, 4>,
    ) -> ManifoldHyperConnectionStreamCoefficients<B> {
        let [batch, _streams, time, _dim] = residuals.shape().dims::<4>();
        let coefficients = self.coefficients_static_sinkhorn();
        let branch_input = coefficients
            .branch_input_weights
            .clone()
            .slice_dim(1, 0..1)
            .reshape([1, 1, self.num_streams])
            .repeat_dim(0, batch)
            .repeat_dim(1, time);
        let branch_output = coefficients.branch_output_weights.map(|weights| {
            weights
                .slice_dim(0, 0..1)
                .reshape([1, 1, self.num_streams])
                .repeat_dim(0, batch)
                .repeat_dim(1, time)
        });
        let residual_weights = coefficients
            .residual_weights
            .reshape([1, 1, self.num_streams, self.num_streams])
            .repeat_dim(0, batch)
            .repeat_dim(1, time);
        ManifoldHyperConnectionStreamCoefficients {
            residual_weights,
            branch_input_weights: branch_input,
            branch_output_weights: branch_output,
        }
    }

    fn stream_coefficients_dynamic(
        &self,
        residuals: Tensor<B, 4>,
    ) -> ManifoldHyperConnectionStreamCoefficients<B> {
        let [batch, _streams, time, _dim] = residuals.shape().dims::<4>();
        let features = self.controller_features(residuals);
        let bt = batch * time;
        let alpha_dynamic = features.clone().matmul(
            self.alpha_weight
                .as_ref()
                .expect("dynamic_positive alpha_weight")
                .val(),
        ) + self
            .alpha_bias
            .as_ref()
            .expect("dynamic_positive alpha_bias")
            .val()
            .reshape([1, self.num_streams]);
        let alpha_logits = self
            .static_stream_logits(self.h_pre_logits.val())
            .repeat_dim(0, bt)
            + alpha_dynamic
                * self.dynamic_gate_value(
                    self.alpha_dynamic_gate
                        .as_ref()
                        .expect("dynamic_positive alpha_dynamic_gate"),
                );
        let branch_input_weights =
            self.positive_sigmoid(alpha_logits)
                .reshape([batch, time, self.num_streams]);

        let branch_output_weights = if self.add_branch_out_to_residual {
            let beta_dynamic = features.clone().matmul(
                self.beta_weight
                    .as_ref()
                    .expect("dynamic_positive beta_weight")
                    .val(),
            ) + self
                .beta_bias
                .as_ref()
                .expect("dynamic_positive beta_bias")
                .val()
                .reshape([1, self.num_streams]);
            let beta_logits = self
                .static_stream_logits(
                    self.h_post_logits
                        .as_ref()
                        .expect("dynamic_positive h_post_logits")
                        .val(),
                )
                .repeat_dim(0, bt)
                + beta_dynamic
                    * self.dynamic_gate_value(
                        self.beta_dynamic_gate
                            .as_ref()
                            .expect("dynamic_positive beta_dynamic_gate"),
                    );
            Some(
                self.positive_sigmoid(beta_logits)
                    .reshape([batch, time, self.num_streams]),
            )
        } else {
            None
        };

        let carry_dynamic = features.matmul(
            self.carry_weight
                .as_ref()
                .expect("dynamic_positive carry_weight")
                .val(),
        );
        let carry_logits = self
            .carry_bias
            .as_ref()
            .expect("dynamic_positive carry_bias")
            .val()
            .reshape([1, self.num_streams * self.num_streams])
            .repeat_dim(0, bt)
            + carry_dynamic
                * self.dynamic_gate_value(
                    self.carry_dynamic_gate
                        .as_ref()
                        .expect("dynamic_positive carry_dynamic_gate"),
                );
        let residual_weights = self
            .sinkhorn_batched(carry_logits.reshape([bt, self.num_streams, self.num_streams]))
            .reshape([batch, time, self.num_streams, self.num_streams]);

        ManifoldHyperConnectionStreamCoefficients {
            residual_weights,
            branch_input_weights,
            branch_output_weights,
        }
    }

    pub fn stream_coefficients(
        &self,
        residuals: Tensor<B, 4>,
    ) -> ManifoldHyperConnectionStreamCoefficients<B> {
        match self.coefficient_policy() {
            ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn => {
                self.stream_coefficients_static(residuals)
            }
            ManifoldHyperConnectionCoefficientPolicy::DynamicPositive => {
                self.stream_coefficients_dynamic(residuals)
            }
        }
    }

    pub fn bootstrap_streams(&self, residuals: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, streams, _, dim] = residuals.shape().dims::<4>();
        if streams == self.num_streams {
            return residuals;
        }
        if streams != 1 || self.num_streams <= 1 {
            return residuals;
        }

        let expanded = residuals.repeat_dim(1, self.num_streams);
        match &self.stream_bootstrap_bias {
            Some(bias) => expanded + bias.val().reshape([1, self.num_streams, 1, dim]),
            None => expanded,
        }
    }

    fn mix_streams_generic(&self, residuals: Tensor<B, 4>, weights: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = residuals.shape().dims::<4>();
        let [in_streams, out_streams] = weights.shape().dims::<2>();
        debug_assert_eq!(streams, in_streams);
        let flat = residuals
            .swap_dims(1, 2)
            .swap_dims(2, 3)
            .reshape([batch * time * dim, streams]);
        let mixed = flat.matmul(weights);
        mixed
            .reshape([batch, time, dim, out_streams])
            .swap_dims(2, 3)
            .swap_dims(1, 2)
    }

    fn mix_streams(&self, residuals: Tensor<B, 4>, weights: Tensor<B, 2>) -> Tensor<B, 4> {
        let [_, streams, _, _] = residuals.shape().dims::<4>();
        let [in_streams, out_streams] = weights.shape().dims::<2>();
        debug_assert_eq!(streams, in_streams);

        if streams == 1 && out_streams == 1 {
            return residuals * weights.reshape([1, 1, 1, 1]);
        }

        if streams == 1 {
            let expanded =
                residuals.swap_dims(1, 2).swap_dims(2, 3) * weights.reshape([1, 1, 1, out_streams]);
            return expanded.swap_dims(2, 3).swap_dims(1, 2);
        }

        if out_streams == 1 {
            let scaled = residuals * weights.reshape([1, streams, 1, 1]);
            return scaled.sum_dim(1);
        }

        self.mix_streams_generic(residuals, weights)
    }

    fn mix_streams_dynamic(&self, residuals: Tensor<B, 4>, weights: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = residuals.shape().dims::<4>();
        let [wb, wt, in_streams, out_streams] = weights.shape().dims::<4>();
        debug_assert_eq!([batch, time, streams], [wb, wt, in_streams]);

        let mut outputs = Vec::with_capacity(out_streams);
        for out_idx in 0..out_streams {
            let mut combined: Option<Tensor<B, 4>> = None;
            for in_idx in 0..in_streams {
                let stream = residuals.clone().slice_dim(1, in_idx..in_idx + 1);
                let weight = weights
                    .clone()
                    .slice_dim(2, in_idx..in_idx + 1)
                    .slice_dim(3, out_idx..out_idx + 1)
                    .reshape([batch, 1, time, 1]);
                let contribution = stream * weight;
                combined = Some(match combined {
                    Some(current) => current + contribution,
                    None => contribution,
                });
            }
            outputs.push(combined.expect("at least one stream"));
        }

        if outputs.is_empty() {
            Tensor::<B, 4>::zeros([batch, 0, time, dim], &residuals.device())
        } else {
            Tensor::cat(outputs, 1)
        }
    }

    fn aggregate_streams_dynamic(
        &self,
        residuals: Tensor<B, 4>,
        weights: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = residuals.shape().dims::<4>();
        let [wb, wt, ws] = weights.shape().dims::<3>();
        debug_assert_eq!([batch, time, streams], [wb, wt, ws]);

        let mut combined: Option<Tensor<B, 4>> = None;
        for in_idx in 0..streams {
            let stream = residuals.clone().slice_dim(1, in_idx..in_idx + 1);
            let weight = weights
                .clone()
                .slice_dim(2, in_idx..in_idx + 1)
                .reshape([batch, 1, time, 1]);
            let contribution = stream * weight;
            combined = Some(match combined {
                Some(current) => current + contribution,
                None => contribution,
            });
        }

        combined
            .unwrap_or_else(|| Tensor::<B, 4>::zeros([batch, 1, time, dim], &residuals.device()))
    }

    fn distribute_branch_dynamic(
        &self,
        branch_output: Tensor<B, 4>,
        weights: Tensor<B, 3>,
    ) -> Tensor<B, 4> {
        let [batch, branch_views, time, dim] = branch_output.shape().dims::<4>();
        debug_assert_eq!(
            branch_views, 1,
            "stream-wrapper mHC expects a single BDH branch view"
        );
        let [wb, wt, streams] = weights.shape().dims::<3>();
        debug_assert_eq!([batch, time], [wb, wt]);

        let mut outputs = Vec::with_capacity(streams);
        for stream_idx in 0..streams {
            let weight = weights
                .clone()
                .slice_dim(2, stream_idx..stream_idx + 1)
                .reshape([batch, 1, time, 1]);
            outputs.push(branch_output.clone() * weight);
        }

        if outputs.is_empty() {
            Tensor::<B, 4>::zeros([batch, 0, time, dim], &branch_output.device())
        } else {
            Tensor::cat(outputs, 1)
        }
    }

    pub fn width_connection_with_coefficients(
        &self,
        residuals: Tensor<B, 4>,
        coefficients: &ManifoldHyperConnectionCoefficients<B>,
    ) -> ManifoldHyperConnectionWidthOutput<B> {
        debug_assert_eq!(residuals.shape().dims::<4>()[1], self.num_streams);
        let residuals_out =
            self.mix_streams(residuals.clone(), coefficients.residual_weights.clone());
        let branch_input = self.mix_streams(residuals, coefficients.branch_input_weights.clone());

        ManifoldHyperConnectionWidthOutput {
            branch_input,
            residuals_out,
            coefficients: coefficients.clone(),
        }
    }

    pub fn width_connection(
        &self,
        residuals: Tensor<B, 4>,
    ) -> ManifoldHyperConnectionWidthOutput<B> {
        let coefficients = self.coefficients();
        self.width_connection_with_coefficients(residuals, &coefficients)
    }

    pub fn stream_width_connection(
        &self,
        residuals: Tensor<B, 4>,
    ) -> ManifoldHyperConnectionStreamOutput<B> {
        let coefficients = self.stream_coefficients(residuals.clone());
        let residuals_out =
            self.mix_streams_dynamic(residuals.clone(), coefficients.residual_weights.clone());
        let branch_input =
            self.aggregate_streams_dynamic(residuals, coefficients.branch_input_weights.clone());
        ManifoldHyperConnectionStreamOutput {
            branch_input,
            residuals_out,
            coefficients,
        }
    }

    pub fn depth_connection_with_coefficients(
        &self,
        branch_output: Tensor<B, 4>,
        residuals: Tensor<B, 4>,
        coefficients: &ManifoldHyperConnectionCoefficients<B>,
    ) -> Tensor<B, 4> {
        if !self.add_branch_out_to_residual {
            return branch_output;
        }
        let Some(beta) = coefficients.branch_output_weights.clone() else {
            return residuals;
        };
        let updates = self.mix_streams(branch_output, beta);
        self.dropout.forward(residuals + updates)
    }

    pub fn stream_depth_connection(
        &self,
        branch_output: Tensor<B, 4>,
        residuals: Tensor<B, 4>,
        coefficients: &ManifoldHyperConnectionStreamCoefficients<B>,
    ) -> Tensor<B, 4> {
        if !self.add_branch_out_to_residual {
            return residuals;
        }
        let Some(beta) = coefficients.branch_output_weights.clone() else {
            return residuals;
        };
        let updates = self.distribute_branch_dynamic(branch_output, beta);
        self.dropout.forward(residuals + updates)
    }

    pub fn depth_connection(
        &self,
        branch_output: Tensor<B, 4>,
        residuals: Tensor<B, 4>,
        branch_output_weights: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        let coefficients = ManifoldHyperConnectionCoefficients {
            residual_weights: Tensor::<B, 2>::zeros(
                [self.num_streams, self.num_streams],
                &branch_output.device(),
            ),
            branch_input_weights: Tensor::<B, 2>::zeros(
                [self.num_streams, self.num_views],
                &branch_output.device(),
            ),
            branch_output_weights,
        };
        self.depth_connection_with_coefficients(branch_output, residuals, &coefficients)
    }

    pub fn passthrough(&self, residuals: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.width_connection(residuals);
        self.depth_connection_with_coefficients(
            output.branch_input,
            output.residuals_out,
            &output.coefficients,
        )
    }

    pub fn stream_passthrough(&self, residuals: Tensor<B, 4>) -> Tensor<B, 4> {
        let output = self.stream_width_connection(residuals);
        self.stream_depth_connection(
            output.branch_input,
            output.residuals_out,
            &output.coefficients,
        )
    }
}
