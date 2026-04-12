use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct SharedLowrankContinualBackpropRuntime {
    optimizer_step: Arc<AtomicUsize>,
    sample_interval_steps: usize,
    batch_stats: Arc<Mutex<Option<SharedLowrankActivationBatchStats>>>,
}

impl SharedLowrankContinualBackpropRuntime {
    pub fn new(sample_interval_steps: usize) -> Self {
        Self {
            optimizer_step: Arc::new(AtomicUsize::new(0)),
            sample_interval_steps: sample_interval_steps.max(1),
            batch_stats: Arc::new(Mutex::new(None)),
        }
    }

    pub fn record_y_neuron_stats<B: Backend>(&self, y_neuron: Tensor<B, 4>) {
        if !self.should_sample_step() {
            return;
        }

        let [batch, heads, time, latent_per_head] = y_neuron.shape().dims::<4>();
        let denom = (batch * heads * time) as f64;
        let mean = y_neuron
            .clone()
            .sum_dims_squeeze::<1, _>(&[0, 1, 2])
            .div_scalar(denom)
            .detach();
        let mean_abs = y_neuron
            .abs()
            .sum_dims_squeeze::<1, _>(&[0, 1, 2])
            .div_scalar(denom)
            .detach();
        debug_assert_eq!(mean.shape().dims::<1>()[0], latent_per_head);
        debug_assert_eq!(mean_abs.shape().dims::<1>()[0], latent_per_head);
        let mean_values = mean
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("shared lowrank cbp mean activation vec");
        let abs_values = mean_abs
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("shared lowrank cbp abs activation vec");

        let mut stats = self
            .batch_stats
            .lock()
            .expect("shared lowrank cbp batch stats lock poisoned");
        let entry = stats.get_or_insert_with(|| SharedLowrankActivationBatchStats {
            mean_sum: vec![0.0; mean_values.len()],
            mean_abs_sum: vec![0.0; abs_values.len()],
            samples: 0,
        });
        if entry.mean_sum.len() != mean_values.len() || entry.mean_abs_sum.len() != abs_values.len()
        {
            *entry = SharedLowrankActivationBatchStats {
                mean_sum: vec![0.0; mean_values.len()],
                mean_abs_sum: vec![0.0; abs_values.len()],
                samples: 0,
            };
        }
        for (dst, value) in entry.mean_sum.iter_mut().zip(mean_values.iter()) {
            *dst += *value;
        }
        for (dst, value) in entry.mean_abs_sum.iter_mut().zip(abs_values.iter()) {
            *dst += *value;
        }
        entry.samples = entry.samples.saturating_add(1);
    }

    pub fn should_sample_step(&self) -> bool {
        let next_optimizer_step = self.optimizer_step.load(Ordering::Relaxed) + 1;
        next_optimizer_step % self.sample_interval_steps == 0
    }

    pub fn optimizer_step(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.optimizer_step)
    }

    pub fn take_batch_stats(&self) -> Option<SharedLowrankActivationBatchStats> {
        self.batch_stats
            .lock()
            .expect("shared lowrank cbp batch stats lock poisoned")
            .take()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SharedLowrankActivationBatchStats {
    pub mean_sum: Vec<f32>,
    pub mean_abs_sum: Vec<f32>,
    pub samples: usize,
}

impl SharedLowrankActivationBatchStats {
    pub fn mean(&self) -> Vec<f32> {
        if self.samples == 0 {
            return vec![0.0; self.mean_sum.len()];
        }
        let scale = 1.0 / self.samples as f32;
        self.mean_sum.iter().map(|value| value * scale).collect()
    }

    pub fn mean_abs(&self) -> Vec<f32> {
        if self.samples == 0 {
            return vec![0.0; self.mean_abs_sum.len()];
        }
        let scale = 1.0 / self.samples as f32;
        self.mean_abs_sum
            .iter()
            .map(|value| value * scale)
            .collect()
    }
}

#[derive(Clone, Debug, Default)]
pub struct SharedLowrankFeatureMetrics {
    pub incoming_l1: Vec<f32>,
    pub outgoing_l1: Vec<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SharedLowrankParamIds {
    pub rwkv_time_decay: burn::module::ParamId,
    pub encoder: burn::module::ParamId,
    pub encoder_v: burn::module::ParamId,
    pub decoder: burn::module::ParamId,
}

impl<B: Backend> BDH<B> {
    pub fn with_shared_lowrank_continual_backprop_runtime(
        mut self,
        runtime: Option<SharedLowrankContinualBackpropRuntime>,
    ) -> Self {
        self.shared_lowrank_continual_backprop = runtime;
        self
    }

    pub fn shared_lowrank_continual_backprop_runtime(
        &self,
    ) -> Option<&SharedLowrankContinualBackpropRuntime> {
        self.shared_lowrank_continual_backprop.as_ref()
    }

    pub fn take_shared_lowrank_continual_backprop_batch_stats(
        &self,
    ) -> Option<SharedLowrankActivationBatchStats> {
        self.shared_lowrank_continual_backprop
            .as_ref()
            .and_then(|runtime| runtime.take_batch_stats())
    }

    pub fn supports_shared_lowrank_continual_backprop(&self) -> bool {
        !self.y_neuron_recurrence.enabled && self.rollout_fast_steps_per_slow_step == 1
    }

    pub fn shared_lowrank_param_ids(&self) -> SharedLowrankParamIds {
        SharedLowrankParamIds {
            rwkv_time_decay: self.rwkv_time_decay.id,
            encoder: self.encoder.id,
            encoder_v: self.encoder_v.id,
            decoder: self.decoder.id,
        }
    }

    pub fn shared_lowrank_feature_count(&self) -> usize {
        self.encoder.val().shape().dims::<3>()[2]
    }

    pub fn shared_lowrank_device(&self) -> B::Device {
        self.decoder.val().device()
    }

    pub fn shared_lowrank_feature_metrics(&self) -> SharedLowrankFeatureMetrics {
        let [heads, embd, latent_per_head] = self.encoder.val().shape().dims::<3>();
        let decoder = self
            .decoder
            .val()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("shared lowrank decoder weights to vec");
        let encoder = self
            .encoder
            .val()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("shared lowrank encoder weights to vec");
        let encoder_v = self
            .encoder_v
            .val()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("shared lowrank encoder_v weights to vec");
        let mut incoming_l1 = vec![0.0; latent_per_head];
        let mut outgoing_l1 = vec![0.0; latent_per_head];

        for head in 0..heads {
            for local_idx in 0..latent_per_head {
                let mut incoming = 0.0f32;
                for embd_idx in 0..embd {
                    let base = (head * embd + embd_idx) * latent_per_head + local_idx;
                    incoming += encoder[base].abs();
                    incoming += encoder_v[base].abs();
                }
                incoming_l1[local_idx] += incoming;
            }
        }

        let decoder_width = self.decoder.val().shape().dims::<2>()[1];
        for head in 0..heads {
            for local_idx in 0..latent_per_head {
                let row = head * latent_per_head + local_idx;
                let row_start = row * decoder_width;
                let row_end = row_start + decoder_width;
                outgoing_l1[local_idx] += decoder[row_start..row_end]
                    .iter()
                    .map(|value| value.abs())
                    .sum::<f32>();
            }
        }

        SharedLowrankFeatureMetrics {
            incoming_l1,
            outgoing_l1,
        }
    }

    pub fn with_reinitialized_shared_lowrank_features(
        &self,
        fresh: &Self,
        feature_indices: &[usize],
    ) -> Self {
        if feature_indices.is_empty() {
            return self.clone();
        }

        let mut updated = self.clone();
        let [heads, _embd, latent_per_head] = self.encoder.val().shape().dims::<3>();
        let selected = feature_indices
            .iter()
            .copied()
            .filter(|idx| *idx < latent_per_head)
            .collect::<Vec<_>>();
        if selected.is_empty() {
            return updated;
        }

        updated.encoder = Param::from_tensor(replace_selected_3d_features_from_fresh(
            updated.encoder.val(),
            fresh.encoder.val(),
            &selected,
        ));
        updated.encoder_v = Param::from_tensor(replace_selected_3d_features_from_fresh(
            updated.encoder_v.val(),
            fresh.encoder_v.val(),
            &selected,
        ));
        updated.decoder = Param::from_tensor(zero_selected_2d_rows(
            updated.decoder.val(),
            &selected,
            latent_per_head,
        ));

        let rwkv_shape = updated.rwkv_time_decay.val().shape().dims::<2>();
        if rwkv_shape == [heads, latent_per_head] {
            updated.rwkv_time_decay = Param::from_tensor(replace_selected_2d_features_from_fresh(
                updated.rwkv_time_decay.val(),
                fresh.rwkv_time_decay.val(),
                &selected,
            ));
        }

        updated
    }
}

fn replace_selected_3d_features_from_fresh<B: Backend>(
    current: Tensor<B, 3>,
    fresh: Tensor<B, 3>,
    selected: &[usize],
) -> Tensor<B, 3> {
    let device = current.device();
    let [heads, embd, latent_per_head] = current.shape().dims::<3>();
    let mut current_values = current
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("3d tensor to vec");
    let fresh_values = fresh
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fresh 3d tensor to vec");
    for local_idx in selected
        .iter()
        .copied()
        .filter(|idx| *idx < latent_per_head)
    {
        for head in 0..heads {
            for embd_idx in 0..embd {
                let flat = (head * embd + embd_idx) * latent_per_head + local_idx;
                current_values[flat] = fresh_values[flat];
            }
        }
    }
    Tensor::<B, 3>::from_data(
        TensorData::new(current_values, [heads, embd, latent_per_head]),
        &device,
    )
}

fn replace_selected_2d_features_from_fresh<B: Backend>(
    current: Tensor<B, 2>,
    fresh: Tensor<B, 2>,
    selected: &[usize],
) -> Tensor<B, 2> {
    let device = current.device();
    let [rows, cols] = current.shape().dims::<2>();
    let mut current_values = current
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("2d tensor to vec");
    let fresh_values = fresh
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("fresh 2d tensor to vec");
    for local_idx in selected.iter().copied().filter(|idx| *idx < cols) {
        for row in 0..rows {
            let flat = row * cols + local_idx;
            current_values[flat] = fresh_values[flat];
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(current_values, [rows, cols]), &device)
}

fn zero_selected_2d_rows<B: Backend>(
    current: Tensor<B, 2>,
    selected: &[usize],
    latent_per_head: usize,
) -> Tensor<B, 2> {
    let device = current.device();
    let [rows, cols] = current.shape().dims::<2>();
    let mut current_values = current
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("2d tensor to vec");
    if latent_per_head == 0 {
        return Tensor::<B, 2>::from_data(TensorData::new(current_values, [rows, cols]), &device);
    }
    for local_idx in selected
        .iter()
        .copied()
        .filter(|idx| *idx < latent_per_head)
    {
        let mut row = local_idx;
        while row < rows {
            let start = row * cols;
            let end = start + cols;
            current_values[start..end].fill(0.0);
            row += latent_per_head;
        }
    }
    Tensor::<B, 2>::from_data(TensorData::new(current_values, [rows, cols]), &device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    fn test_model() -> BDH<TestBackend> {
        let device = Default::default();
        let config = BDHConfig {
            n_layer: 2,
            n_embd: 8,
            dropout: 0.0,
            n_head: 2,
            mlp_internal_dim_multiplier: 2,
            vocab_size: 32,
            rollout_fast_steps_per_slow_step: 1,
            ..Default::default()
        };
        BDH::new(config, &device)
    }

    #[test]
    fn shared_lowrank_runtime_accumulates_and_takes_batch_stats() {
        let runtime = SharedLowrankContinualBackpropRuntime::new(1);
        let device = Default::default();
        let y_neuron = Tensor::<TestBackend, 4>::from_floats([[[[1.0, -1.0]]]], &device);
        runtime.record_y_neuron_stats(y_neuron);
        let stats = runtime.take_batch_stats().expect("stats should exist");
        assert_eq!(stats.samples, 1);
        assert_eq!(stats.mean().len(), 2);
        assert!(
            runtime.take_batch_stats().is_none(),
            "take should clear stats"
        );
    }

    #[test]
    fn reinitialized_shared_lowrank_features_replace_inputs_and_zero_decoder_rows() {
        let mut source = test_model();
        let fresh = test_model();
        let device = Default::default();
        source.encoder = Param::from_tensor(Tensor::<TestBackend, 3>::ones(
            source.encoder.val().shape().dims::<3>(),
            &device,
        ));
        source.encoder_v = Param::from_tensor(Tensor::<TestBackend, 3>::ones(
            source.encoder_v.val().shape().dims::<3>(),
            &device,
        ));
        source.decoder = Param::from_tensor(Tensor::<TestBackend, 2>::ones(
            source.decoder.val().shape().dims::<2>(),
            &device,
        ));
        let updated = source.with_reinitialized_shared_lowrank_features(&fresh, &[0, 3]);
        let decoder = updated
            .decoder
            .val()
            .to_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("decoder vec");
        let width = updated.decoder.val().shape().dims::<2>()[1];
        assert!(decoder[0..width].iter().all(|value| *value == 0.0));
        let row3 = 3 * width;
        assert!(
            decoder[row3..row3 + width]
                .iter()
                .all(|value| *value == 0.0)
        );
        let second_head_row0 = 8 * width;
        assert!(
            decoder[second_head_row0..second_head_row0 + width]
                .iter()
                .all(|value| *value == 0.0)
        );
    }
}
