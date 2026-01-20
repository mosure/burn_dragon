use burn::module::{Module, Param};
use burn::nn::{Dropout, DropoutConfig};
use burn::tensor::backend::Backend;
use burn::tensor::{Tensor, TensorData, activation};

const MHC_EPS: f32 = 1e-6;

/// Configuration for manifold-constrained hyper-connections (mHC).
#[derive(Clone, Debug)]
pub struct ManifoldHyperConnectionsConfig {
    pub enabled: bool,
    pub num_streams: usize,
    pub num_views: usize,
    pub mhc_iters: usize,
    pub mhc_tau: f32,
    pub add_branch_out_to_residual: bool,
    pub dropout: f64,
}

impl Default for ManifoldHyperConnectionsConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_streams: 1,
            num_views: 1,
            mhc_iters: 10,
            mhc_tau: 0.05,
            add_branch_out_to_residual: true,
            dropout: 0.0,
        }
    }
}

/// Manifold-constrained hyper-connections operating on residual streams.
#[derive(Module, Debug)]
pub struct ManifoldHyperConnections<B: Backend> {
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
        let num_streams = config.num_streams.max(1);
        let num_views = config.num_views.max(1);
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
            num_streams,
            num_views,
            mhc_iters: config.mhc_iters.max(1),
            mhc_tau: config.mhc_tau.max(MHC_EPS),
            add_branch_out_to_residual: config.add_branch_out_to_residual,
            dropout: DropoutConfig::new(config.dropout).init(),
            h_res_logits,
            h_pre_logits,
            h_post_logits,
        }
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

    fn mix_streams(&self, residuals: Tensor<B, 4>, weights: Tensor<B, 2>) -> Tensor<B, 4> {
        let [batch, streams, time, dim] = residuals.shape().dims::<4>();
        let [in_streams, out_streams] = weights.shape().dims::<2>();
        debug_assert_eq!(streams, in_streams);
        let flat = residuals
            .swap_dims(1, 2)
            .reshape([batch * time * dim, streams]);
        let mixed = flat.matmul(weights);
        mixed
            .reshape([batch, time, dim, out_streams])
            .swap_dims(2, 3)
            .swap_dims(1, 2)
    }

    pub fn width_connection(
        &self,
        residuals: Tensor<B, 4>,
    ) -> (Tensor<B, 4>, Tensor<B, 4>, Option<Tensor<B, 2>>) {
        debug_assert_eq!(residuals.shape().dims::<4>()[1], self.num_streams);
        let h_res = self.sinkhorn(self.h_res_logits.val());
        let residuals_out = self.mix_streams(residuals.clone(), h_res);

        let h_pre = activation::softmax(self.h_pre_logits.val(), 1).swap_dims(0, 1);
        let branch_input = self.mix_streams(residuals, h_pre);

        let h_post = self
            .h_post_logits
            .as_ref()
            .map(|param| activation::softmax(param.val(), 1));

        (branch_input, residuals_out, h_post)
    }

    pub fn depth_connection(
        &self,
        branch_output: Tensor<B, 4>,
        residuals: Tensor<B, 4>,
        beta: Option<Tensor<B, 2>>,
    ) -> Tensor<B, 4> {
        if !self.add_branch_out_to_residual {
            return branch_output;
        }
        let Some(beta) = beta else {
            return residuals;
        };
        let updates = self.mix_streams(branch_output, beta);
        self.dropout.forward(residuals + updates)
    }
}

#[cfg(test)]
mod mhc_tests {
    use super::*;
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn mhc_sinkhorn_rows_cols_sum_close_to_one() {
        let device = <TestBackend as Backend>::Device::default();
        let config = ManifoldHyperConnectionsConfig {
            enabled: true,
            num_streams: 3,
            num_views: 2,
            ..Default::default()
        };
        let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 0, &device);
        let h_res = mhc.sinkhorn(mhc.h_res_logits.val());
        let row_sums = h_res
            .clone()
            .sum_dim(1)
            .to_data()
            .iter::<f32>()
            .collect::<Vec<_>>();
        let col_sums = h_res
            .sum_dim(0)
            .to_data()
            .iter::<f32>()
            .collect::<Vec<_>>();
        for sum in row_sums.into_iter().chain(col_sums.into_iter()) {
            assert!((sum - 1.0).abs() < 1e-3, "sum not close to 1: {sum}");
        }
    }

    #[test]
    fn mhc_width_connection_shapes_match_streams_and_views() {
        let device = <TestBackend as Backend>::Device::default();
        let config = ManifoldHyperConnectionsConfig {
            enabled: true,
            num_streams: 2,
            num_views: 2,
            ..Default::default()
        };
        let mhc = ManifoldHyperConnections::<TestBackend>::new(&config, 1, &device);
        let residuals = Tensor::<TestBackend, 4>::zeros([4, config.num_streams, 6, 8], &device);
        let (branch_input, residuals_out, beta) = mhc.width_connection(residuals);
        assert_eq!(
            branch_input.shape().dims::<4>(),
            [4, config.num_views, 6, 8]
        );
        assert_eq!(
            residuals_out.shape().dims::<4>(),
            [4, config.num_streams, 6, 8]
        );
        let beta = beta.expect("expected beta");
        assert_eq!(beta.shape().dims::<2>(), [config.num_views, config.num_streams]);
    }
}
