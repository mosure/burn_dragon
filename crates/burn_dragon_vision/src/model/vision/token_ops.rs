use super::*;
use burn_dragon_core::FusedAttentionExecutor;
use burn_dragon_wgpu::api::attention::{
    CompiledDenseAttentionPlan, CompiledDenseScoresPlan, try_fused_dense_row_l1_attention_wgpu,
    try_fused_dense_row_l1_attention_wgpu_with_plan, try_fused_dense_row_l1_scores_wgpu,
    try_fused_dense_row_l1_scores_wgpu_with_plan,
};

impl<B: Backend> VisionDragon<B> {
    pub(super) fn apply_token_norm<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        match &self.token_norm {
            Some(norm) => norm.forward(tokens),
            None => tokens,
        }
    }

    pub(super) fn apply_latent_activation<const D: usize>(
        &self,
        values: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self.latent_activation {
            VisionLatentActivation::Relu => activation::relu(values),
            VisionLatentActivation::Gelu => activation::gelu(values),
            VisionLatentActivation::Identity => values,
        }
    }

    pub(super) fn sync_cls_tokens_multi(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        if !(self.use_cls_token && self.cls_sync_alpha > 0.0) {
            return tokens;
        }
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if streams <= 1 || time == 0 {
            return tokens;
        }
        let alpha = self.cls_sync_alpha.clamp(0.0, 1.0);
        let cls = tokens.clone().slice_dim(2, 0..1);
        let shared = cls
            .clone()
            .sum_dim(1)
            .mul_scalar(1.0 / streams as f32)
            .reshape([batch, 1, 1, dim])
            .repeat_dim(1, streams);
        let blended = cls.mul_scalar(1.0 - alpha) + shared.mul_scalar(alpha);
        if time <= 1 {
            return blended;
        }
        let rest = tokens.slice_dim(2, 1..time);
        Tensor::cat(vec![blended, rest], 2)
    }

    pub(super) fn full_attention(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        self.full_attention_with_plans(query, value, None, None)
    }

    pub(super) fn full_attention_with_plans(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        scores_plan: Option<&CompiledDenseScoresPlan<B>>,
        attention_plan: Option<&CompiledDenseAttentionPlan<B>>,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, _latent_dim] = query.shape().dims::<4>();
        if self.kernel.enabled
            && matches!(
                self.kernel.attention_executor,
                FusedAttentionExecutor::AttentionContext
            )
            && matches!(self.attention_mode, VisionAttentionMode::RowL1)
            && self.use_alibi
            && let Some(slopes) = self.alibi_slopes.as_ref()
            && let Some(context) = attention_plan
                .and_then(|plan| {
                    try_fused_dense_row_l1_attention_wgpu_with_plan(&query, &value, slopes, plan)
                })
                .or_else(|| try_fused_dense_row_l1_attention_wgpu(&query, &value, slopes))
        {
            return context;
        }

        let scores = if self.kernel.enabled
            && matches!(self.attention_mode, VisionAttentionMode::RowL1)
            && self.use_alibi
        {
            self.alibi_slopes
                .as_ref()
                .and_then(|slopes| {
                    scores_plan
                        .and_then(|plan| {
                            try_fused_dense_row_l1_scores_wgpu_with_plan(&query, slopes, plan)
                        })
                        .or_else(|| try_fused_dense_row_l1_scores_wgpu(&query, slopes))
                })
                .unwrap_or_else(|| self.full_attention_scores_reference(query.clone()))
        } else {
            self.full_attention_scores_reference(query.clone())
        };
        let value_dims = value.shape().dims::<4>();
        let value_time = value_dims[2];
        let value_dim = value_dims[3];
        let value_flat = if value_dims[1] == heads {
            value.reshape([batch * heads, value_time, value_dim])
        } else {
            value
                .reshape([batch, 1, value_time, value_dim])
                .repeat_dim(1, heads)
                .reshape([batch * heads, value_time, value_dim])
        };
        scores
            .reshape([batch * heads, time, time])
            .matmul(value_flat)
            .reshape([batch, heads, time, value_dim])
    }

    fn full_attention_scores_reference(&self, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        let [batch, heads, time, latent_dim] = query.shape().dims::<4>();
        let query_scaled_flat =
            query
                .clone()
                .div_scalar(scale)
                .reshape([batch * heads, time, latent_dim]);
        let key_flat = query
            .reshape([batch * heads, time, latent_dim])
            .swap_dims(1, 2);
        let mut scores = query_scaled_flat
            .matmul(key_flat)
            .reshape([batch, heads, time, time]);
        if self.use_alibi
            && let Some(slopes) = self.alibi_slopes.as_ref()
        {
            let device = scores.device();
            let slopes = slopes.clone().reshape([1, heads, 1, 1]);
            let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, time, 1]);
            let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, 1, time]);
            scores = scores + slopes * (pos_col - pos_row);
        }
        match self.attention_mode {
            VisionAttentionMode::Softmax => activation::softmax(scores, 3),
            VisionAttentionMode::RowL1 => {
                let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
                scores / denom
            }
        }
    }

    pub(super) fn prepend_cls(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _time, dim] = tokens.shape().dims::<3>();
        let cls = self
            .cls_token
            .as_ref()
            .expect("cls token enabled")
            .val()
            .reshape([1, 1, dim])
            .repeat_dim(0, batch);
        let cls = if let Some(cls_pos) = &self.cls_pos {
            cls + cls_pos.val().reshape([1, 1, dim])
        } else {
            cls
        };
        Tensor::cat(vec![cls, tokens], 1)
    }

    pub(super) fn split_output(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens.clone().slice_dim(1, 0..1).reshape([batch, dim]);
            let patch_tokens = tokens.slice_dim(1, 1..time);
            VisionDragonOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(1).reshape([batch, dim]);
            VisionDragonOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }

    pub(super) fn split_output_multi(&self, tokens: Tensor<B, 4>) -> VisionDragonMultiOutput<B> {
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens
                .clone()
                .slice_dim(2, 0..1)
                .reshape([batch, streams, dim]);
            let patch_tokens = tokens.slice_dim(2, 1..time);
            VisionDragonMultiOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(2).reshape([batch, streams, dim]);
            VisionDragonMultiOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Distribution;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn make_test_model(n_head: usize) -> VisionDragon<TestBackend> {
        let device = <TestBackend as Backend>::Device::default();
        let config = make_test_config(n_head);
        VisionDragon::<TestBackend>::new(config, &device)
    }

    fn make_test_config(n_head: usize) -> VisionDragonConfig {
        VisionDragonConfig {
            image_size: 8,
            patch_size: 4,
            in_channels: 3,
            embed_dim: 8,
            steps: 2,
            n_head,
            mlp_internal_dim_multiplier: 2,
            projection_dim: 8,
            projection_hidden_dim: 16,
            use_cls_token: true,
            pos_encoding: SpatialPositionalEncodingKind::Rope,
            pos_max_height: 2,
            pos_max_width: 2,
            attention_mode: VisionAttentionMode::RowL1,
            fused_kernels: FusedKernelConfig::default(),
            ..VisionDragonConfig::default()
        }
    }

    fn full_attention_reference(
        model: &VisionDragon<TestBackend>,
        query: Tensor<TestBackend, 4>,
        value: Tensor<TestBackend, 4>,
    ) -> Tensor<TestBackend, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        let mut scores = query
            .clone()
            .div_scalar(scale)
            .matmul(query.clone().swap_dims(2, 3));
        if model.use_alibi
            && let Some(slopes) = model.alibi_slopes.as_ref()
        {
            let device = query.device();
            let [_, heads, time, _] = query.shape().dims::<4>();
            let slopes = slopes.clone().reshape([1, heads, 1, 1]);
            let pos_row = Tensor::<TestBackend, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, time, 1]);
            let pos_col = Tensor::<TestBackend, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, 1, time]);
            scores = scores + slopes * (pos_col - pos_row);
        }
        let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
        let scores = scores / denom;
        let value = if value.shape().dims::<4>()[1] == model.n_head {
            value
        } else {
            value.repeat_dim(1, model.n_head)
        };
        scores.matmul(value)
    }

    #[test]
    fn flattened_attention_matches_reference_repeat_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = make_test_model(2);
        let query = Tensor::<TestBackend, 4>::random([3, 2, 5, 4], Distribution::Default, &device);
        let value = Tensor::<TestBackend, 4>::random([3, 1, 5, 8], Distribution::Default, &device);

        let actual = model.full_attention(query.clone(), value.clone());
        let expected = full_attention_reference(&model, query, value);
        let actual = actual
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual attention vec");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected attention vec");

        assert_eq!(actual.len(), expected.len());
        for (index, (lhs, rhs)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "attention mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn flattened_attention_matches_reference_per_head_value_path() {
        let device = <TestBackend as Backend>::Device::default();
        let model = make_test_model(2);
        let query = Tensor::<TestBackend, 4>::random([3, 2, 5, 4], Distribution::Default, &device);
        let value = Tensor::<TestBackend, 4>::random([3, 2, 5, 8], Distribution::Default, &device);

        let actual = model.full_attention(query.clone(), value.clone());
        let expected = full_attention_reference(&model, query, value);
        let actual = actual
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual attention vec");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected attention vec");

        assert_eq!(actual.len(), expected.len());
        for (index, (lhs, rhs)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "attention mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[cfg(feature = "benchmark")]
    #[test]
    fn benchmark_dense_step_parts_match_full_step() {
        let device = <TestBackend as Backend>::Device::default();
        let model = make_test_model(2);
        let dense = super::VisionDenseBenchAdapter::new(&model);
        let tokens = Tensor::<TestBackend, 3>::random([2, 5, 8], Distribution::Default, &device);
        let state = model.rollout_state_from_tokens(tokens);
        let VisionRolloutState::Dense { token_state } = state else {
            panic!("expected dense rollout state");
        };
        let [batch, time, dim] = token_state.shape().dims::<3>();
        let current = token_state.reshape([batch, 1, time, dim]);

        let x_neuron = dense.x_projection(current.clone());
        let attn = dense.attention_context(x_neuron.clone(), current.clone());
        let y_gate = dense.y_projection(attn);
        let tail = dense.tail(current.clone(), x_neuron, y_gate);
        let full = dense.step(current);

        let tail = tail
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("tail vec");
        let full = full
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("full vec");

        assert_eq!(tail.len(), full.len());
        for (index, (lhs, rhs)) in tail.into_iter().zip(full.into_iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "dense step mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[cfg(feature = "benchmark")]
    #[test]
    fn benchmark_dense_attention_adapter_matches_model_full_attention() {
        let device = <TestBackend as Backend>::Device::default();
        let model = make_test_model(2);
        let adapter = super::VisionDenseAttentionBenchAdapter::from_model(&model);
        let query = Tensor::<TestBackend, 4>::random([2, 2, 5, 4], Distribution::Default, &device);
        let value = Tensor::<TestBackend, 4>::random([2, 1, 5, 8], Distribution::Default, &device);

        let actual = adapter.full_attention_current(query.clone(), value.clone());
        let expected = model.full_attention(query, value);

        let actual = actual
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("adapter attention vec");
        let expected = expected
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("model attention vec");

        assert_eq!(actual.len(), expected.len());
        for (index, (lhs, rhs)) in actual.into_iter().zip(expected.into_iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1e-5,
                "attention adapter mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[cfg(all(feature = "train", not(target_arch = "wasm32")))]
    #[test]
    fn fused_score_kernel_matches_reference_gradients_on_wgpu_autodiff() {
        use burn::tensor::{TensorData, backend::Backend as BackendTrait};
        use burn_autodiff::Autodiff;
        use burn_wgpu::Wgpu;

        type WgpuAutodiffBackend = Autodiff<Wgpu<f32>>;

        let _guard = crate::train::wgpu_test_guard();
        let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
        crate::train::init_wgpu_test_runtime(&device);
        <WgpuAutodiffBackend as BackendTrait>::seed(&device, 3_151);

        let reference_model =
            VisionDragon::<WgpuAutodiffBackend>::new(make_test_config(2), &device);
        let mut fused_config = make_test_config(2);
        fused_config.fused_kernels = FusedKernelConfig {
            enabled: true,
            ..FusedKernelConfig::default()
        };
        let mut fused_model = VisionDragon::<WgpuAutodiffBackend>::new(fused_config, &device);
        fused_model = fused_model.load_record(reference_model.clone().into_record());

        let query = Tensor::<WgpuAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..3 * 2 * 5 * 4).map(|v| v as f32 * 0.01).collect(),
                [3, 2, 5, 4],
            ),
            &device,
        )
        .require_grad();
        let value = Tensor::<WgpuAutodiffBackend, 4>::from_data(
            TensorData::new(
                (0..3 * 2 * 5 * 8).map(|v| v as f32 * 0.005).collect(),
                [3, 2, 5, 8],
            ),
            &device,
        )
        .require_grad();

        let reference_loss = reference_model
            .full_attention(query.clone(), value.clone())
            .mean();
        let fused_attention = fused_model
            .alibi_slopes
            .as_ref()
            .and_then(|slopes| try_fused_dense_row_l1_attention_wgpu(&query, &value, slopes));
        assert!(
            fused_attention.is_some(),
            "fused dense attention should stay available on wgpu autodiff once a backward rule exists"
        );
        let fused_loss = fused_model
            .full_attention(query.clone(), value.clone())
            .mean();

        let reference_grads = reference_loss.backward();
        let fused_grads = fused_loss.backward();

        let reference_query_grad = query
            .grad(&reference_grads)
            .expect("reference query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference query grad vec");
        let fused_query_grad = query
            .grad(&fused_grads)
            .expect("fused query grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused query grad vec");
        let reference_value_grad = value
            .grad(&reference_grads)
            .expect("reference value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("reference value grad vec");
        let fused_value_grad = value
            .grad(&fused_grads)
            .expect("fused value grad")
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("fused value grad vec");

        assert_eq!(reference_query_grad.len(), fused_query_grad.len());
        assert_eq!(reference_value_grad.len(), fused_value_grad.len());
        for (index, (lhs, rhs)) in reference_query_grad
            .into_iter()
            .zip(fused_query_grad.into_iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() <= 1e-4,
                "query grad mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
        for (index, (lhs, rhs)) in reference_value_grad
            .into_iter()
            .zip(fused_value_grad.into_iter())
            .enumerate()
        {
            assert!(
                (lhs - rhs).abs() <= 1e-4,
                "value grad mismatch at index {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[cfg(all(feature = "train", not(target_arch = "wasm32")))]
    #[test]
    fn fused_attention_matches_reference_on_wgpu_large_shape() {
        use burn::tensor::backend::Backend as BackendTrait;
        use burn_autodiff::Autodiff;
        use burn_wgpu::Wgpu;

        type WgpuBackend = Autodiff<Wgpu<f32>>;

        let _guard = crate::train::wgpu_test_guard();
        let device = <WgpuBackend as BackendTrait>::Device::default();
        crate::train::init_wgpu_test_runtime(&device);
        <WgpuBackend as BackendTrait>::seed(&device, 9_271);

        let mut reference_config = make_test_config(8);
        reference_config.embed_dim = 384;
        reference_config.projection_dim = 384;
        reference_config.projection_hidden_dim = 768;
        reference_config.image_size = 280;
        reference_config.patch_size = 14;
        let reference_model = VisionDragon::<WgpuBackend>::new(reference_config.clone(), &device);

        let mut fused_config = reference_config.clone();
        fused_config.fused_kernels = FusedKernelConfig {
            enabled: true,
            ..FusedKernelConfig::default()
        };
        let mut fused_model = VisionDragon::<WgpuBackend>::new(fused_config, &device);
        fused_model = fused_model.load_record(reference_model.clone().into_record());

        let query =
            Tensor::<WgpuBackend, 4>::random([1, 8, 401, 48], Distribution::Default, &device);
        let value =
            Tensor::<WgpuBackend, 4>::random([1, 1, 401, 384], Distribution::Default, &device);

        let fused_attention = fused_model
            .alibi_slopes
            .as_ref()
            .and_then(|slopes| try_fused_dense_row_l1_attention_wgpu(&query, &value, slopes));
        assert!(
            fused_attention.is_some(),
            "fused dense attention should be available on long-sequence wgpu shapes"
        );

        let actual = fused_model
            .full_attention(query.clone(), value.clone())
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual attention vec");
        let expected = reference_model
            .full_attention(query, value)
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected attention vec");

        assert_eq!(actual.len(), expected.len());
        let len = expected.len().max(1);
        let mut max_abs = 0.0_f32;
        let mut mean_abs = 0.0_f32;
        for (lhs, rhs) in actual.into_iter().zip(expected.into_iter()) {
            let diff = (lhs - rhs).abs();
            max_abs = max_abs.max(diff);
            mean_abs += diff;
        }
        mean_abs /= len as f32;
        assert!(
            max_abs <= 1e-2,
            "large-shape fused attention max abs drift too high: {max_abs}"
        );
        assert!(
            mean_abs <= 5e-4,
            "large-shape fused attention mean abs drift too high: {mean_abs}"
        );
    }

    #[cfg(all(feature = "train", not(target_arch = "wasm32")))]
    #[test]
    fn fused_scores_only_executor_matches_reference_on_wgpu_large_shape() {
        use burn::tensor::backend::Backend as BackendTrait;
        use burn_autodiff::Autodiff;
        use burn_wgpu::Wgpu;

        type WgpuBackend = Autodiff<Wgpu<f32>>;

        let _guard = crate::train::wgpu_test_guard();
        let device = <WgpuBackend as BackendTrait>::Device::default();
        crate::train::init_wgpu_test_runtime(&device);
        <WgpuBackend as BackendTrait>::seed(&device, 1_927);

        let mut reference_config = make_test_config(8);
        reference_config.embed_dim = 384;
        reference_config.projection_dim = 384;
        reference_config.projection_hidden_dim = 768;
        reference_config.image_size = 280;
        reference_config.patch_size = 14;
        let reference_model = VisionDragon::<WgpuBackend>::new(reference_config.clone(), &device);

        let mut fused_config = reference_config.clone();
        fused_config.fused_kernels = FusedKernelConfig {
            enabled: true,
            attention_executor: FusedAttentionExecutor::ScoresOnly,
            ..FusedKernelConfig::default()
        };
        let mut fused_model = VisionDragon::<WgpuBackend>::new(fused_config, &device);
        fused_model = fused_model.load_record(reference_model.clone().into_record());

        let query =
            Tensor::<WgpuBackend, 4>::random([1, 8, 401, 48], Distribution::Default, &device);
        let value =
            Tensor::<WgpuBackend, 4>::random([1, 1, 401, 384], Distribution::Default, &device);

        let actual = fused_model
            .full_attention(query.clone(), value.clone())
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual attention vec");
        let expected = reference_model
            .full_attention(query, value)
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected attention vec");

        assert_eq!(actual.len(), expected.len());
        let len = expected.len().max(1);
        let mut max_abs = 0.0_f32;
        let mut mean_abs = 0.0_f32;
        for (lhs, rhs) in actual.into_iter().zip(expected.into_iter()) {
            let diff = (lhs - rhs).abs();
            max_abs = max_abs.max(diff);
            mean_abs += diff;
        }
        mean_abs /= len as f32;
        assert!(
            max_abs <= 1e-2,
            "large-shape fused scores-only max abs drift too high: {max_abs}"
        );
        assert!(
            mean_abs <= 5e-4,
            "large-shape fused scores-only mean abs drift too high: {mean_abs}"
        );
    }

    #[cfg(all(feature = "train", not(target_arch = "wasm32")))]
    #[test]
    fn fused_scores_only_executor_with_plan_matches_reference_on_wgpu_large_shape() {
        use burn::tensor::backend::Backend as BackendTrait;
        use burn_autodiff::Autodiff;
        use burn_wgpu::Wgpu;

        type WgpuBackend = Autodiff<Wgpu<f32>>;

        let _guard = crate::train::wgpu_test_guard();
        let device = <WgpuBackend as BackendTrait>::Device::default();
        crate::train::init_wgpu_test_runtime(&device);
        <WgpuBackend as BackendTrait>::seed(&device, 1_927);

        let mut reference_config = make_test_config(8);
        reference_config.embed_dim = 384;
        reference_config.projection_dim = 384;
        reference_config.projection_hidden_dim = 768;
        reference_config.image_size = 280;
        reference_config.patch_size = 14;
        let reference_model = VisionDragon::<WgpuBackend>::new(reference_config.clone(), &device);

        let mut fused_config = reference_config.clone();
        fused_config.fused_kernels = FusedKernelConfig {
            enabled: true,
            attention_executor: FusedAttentionExecutor::ScoresOnly,
            ..FusedKernelConfig::default()
        };
        let mut fused_model = VisionDragon::<WgpuBackend>::new(fused_config, &device);
        fused_model = fused_model.load_record(reference_model.clone().into_record());

        let query =
            Tensor::<WgpuBackend, 4>::random([1, 8, 401, 48], Distribution::Default, &device);
        let value =
            Tensor::<WgpuBackend, 4>::random([1, 1, 401, 384], Distribution::Default, &device);
        let scores_plan = CompiledDenseScoresPlan::new(1, 8, 401, 48, &device);

        let actual = fused_model
            .full_attention_with_plans(query.clone(), value.clone(), Some(&scores_plan), None)
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("actual attention vec");
        let expected = reference_model
            .full_attention(query, value)
            .inner()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("expected attention vec");

        assert_eq!(actual.len(), expected.len());
        let len = expected.len().max(1);
        let mut max_abs = 0.0_f32;
        let mut mean_abs = 0.0_f32;
        for (lhs, rhs) in actual.into_iter().zip(expected.into_iter()) {
            let diff = (lhs - rhs).abs();
            max_abs = max_abs.max(diff);
            mean_abs += diff;
        }
        mean_abs /= len as f32;
        assert!(
            max_abs <= 1e-2,
            "large-shape fused scores-only with plan max abs drift too high: {max_abs}"
        );
        assert!(
            mean_abs <= 5e-4,
            "large-shape fused scores-only with plan mean abs drift too high: {mean_abs}"
        );
    }
}
