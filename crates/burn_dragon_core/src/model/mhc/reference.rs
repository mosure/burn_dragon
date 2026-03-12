use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};

use super::{
    ManifoldHyperConnectionCoefficientPolicy, ManifoldHyperConnectionCoefficients,
    ManifoldHyperConnectionWidthOutput, ManifoldHyperConnectionsConfig,
};

const MHC_EPS: f32 = 1e-6;

/// Manifold-constrained hyper-connections operating on residual streams.
#[derive(Module, Debug)]
pub struct ManifoldHyperConnections<B: Backend> {
    coefficient_policy_id: usize,
    num_streams: usize,
    num_views: usize,
    mhc_iters: usize,
    mhc_tau: f32,
    add_branch_out_to_residual: bool,
    dropout: Dropout,
    h_res_logits: Param<Tensor<B, 2>>,
    h_pre_logits: Param<Tensor<B, 2>>,
    h_post_logits: Option<Param<Tensor<B, 2>>>,
}

impl<B: Backend> ManifoldHyperConnections<B> {
    pub fn new(
        config: &ManifoldHyperConnectionsConfig,
        layer_index: usize,
        device: &B::Device,
    ) -> Self {
        let num_streams = config.resolved_num_streams();
        let num_views = config.resolved_num_views();
        let mut h_res = vec![-8.0f32; num_streams * num_streams];
        for idx in 0..num_streams {
            h_res[idx * num_streams + idx] = 0.0;
        }
        let h_res_logits = Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(h_res, [num_streams, num_streams]),
            device,
        ));

        let init_idx = layer_index % num_streams;
        let mut h_pre = vec![-8.0f32; num_views * num_streams];
        for view_idx in 0..num_views {
            h_pre[view_idx * num_streams + init_idx] = 0.0;
        }
        let h_pre_logits = Param::from_tensor(Tensor::<B, 2>::from_data(
            TensorData::new(h_pre, [num_views, num_streams]),
            device,
        ));

        let h_post_logits = if config.add_branch_out_to_residual {
            Some(Param::from_tensor(Tensor::<B, 2>::zeros(
                [num_views, num_streams],
                device,
            )))
        } else {
            None
        };

        Self {
            coefficient_policy_id: match config.coefficient_policy {
                ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn => 0,
            },
            num_streams,
            num_views,
            mhc_iters: config.resolved_iters(),
            mhc_tau: config.resolved_tau(),
            add_branch_out_to_residual: config.add_branch_out_to_residual,
            dropout: DropoutConfig::new(config.dropout).init(),
            h_res_logits,
            h_pre_logits,
            h_post_logits,
        }
    }

    pub fn coefficient_policy(&self) -> ManifoldHyperConnectionCoefficientPolicy {
        match self.coefficient_policy_id {
            0 => ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
            _ => ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn,
        }
    }

    pub fn num_streams(&self) -> usize {
        self.num_streams
    }

    pub fn num_views(&self) -> usize {
        self.num_views
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
        match self.coefficient_policy() {
            ManifoldHyperConnectionCoefficientPolicy::StaticSinkhorn => {
                self.coefficients_static_sinkhorn()
            }
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
            let expanded = residuals.swap_dims(1, 2).swap_dims(2, 3)
                * weights.reshape([1, 1, 1, out_streams]);
            return expanded.swap_dims(2, 3).swap_dims(1, 2);
        }

        if out_streams == 1 {
            let scaled = residuals * weights.reshape([1, streams, 1, 1]);
            return scaled.sum_dim(1);
        }

        self.mix_streams_generic(residuals, weights)
    }

    pub fn width_connection_with_coefficients(
        &self,
        residuals: Tensor<B, 4>,
        coefficients: &ManifoldHyperConnectionCoefficients<B>,
    ) -> ManifoldHyperConnectionWidthOutput<B> {
        debug_assert_eq!(residuals.shape().dims::<4>()[1], self.num_streams);
        let residuals_out =
            self.mix_streams(residuals.clone(), coefficients.residual_weights.clone());
        let branch_input =
            self.mix_streams(residuals, coefficients.branch_input_weights.clone());

        ManifoldHyperConnectionWidthOutput {
            branch_input,
            residuals_out,
            coefficients: coefficients.clone(),
        }
    }

    pub fn width_connection(&self, residuals: Tensor<B, 4>) -> ManifoldHyperConnectionWidthOutput<B> {
        let coefficients = self.coefficients();
        self.width_connection_with_coefficients(residuals, &coefficients)
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
}
