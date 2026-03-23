use crate::train::prelude::*;
use burn::module::Ignored;
use burn_dragon_core::ModelState;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

type StreamingStateStore = HashMap<(usize, TypeId), Box<dyn Any + Send>>;

fn streaming_state_store() -> &'static Mutex<StreamingStateStore> {
    static STORE: OnceLock<Mutex<StreamingStateStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

fn next_streaming_runtime_key() -> usize {
    static NEXT_KEY: AtomicUsize = AtomicUsize::new(1);
    NEXT_KEY.fetch_add(1, Ordering::Relaxed)
}

#[derive(Module, Debug)]
pub(crate) struct LanguageTrainModel<B: BackendTrait> {
    pub(crate) model: BDH<B>,
    pub(crate) tbptt_chunk_size: Option<usize>,
    pub(crate) pipeline_plan: Ignored<Option<PipelinePlan>>,
    #[module(ignore)]
    pub(crate) tbptt_persist_across_steps: bool,
    #[module(ignore)]
    streaming_runtime_key: usize,
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(crate) fn new(model: BDH<B>) -> Self {
        Self {
            model,
            tbptt_chunk_size: None,
            pipeline_plan: Ignored(None),
            tbptt_persist_across_steps: false,
            streaming_runtime_key: next_streaming_runtime_key(),
        }
    }

    pub(crate) fn with_tbptt_chunk_size(mut self, tbptt_chunk_size: Option<usize>) -> Self {
        self.tbptt_chunk_size = tbptt_chunk_size;
        self
    }

    pub(crate) fn with_pipeline_plan(mut self, pipeline_plan: Option<PipelinePlan>) -> Self {
        self.pipeline_plan = Ignored(pipeline_plan);
        self
    }

    pub(crate) fn with_tbptt_persist_across_steps(mut self, enabled: bool) -> Self {
        self.tbptt_persist_across_steps = enabled;
        self
    }

    fn effective_tbptt_chunk_size(&self, block_size: usize) -> Option<usize> {
        self.tbptt_chunk_size
            .filter(|chunk_size| *chunk_size > 0 && *chunk_size < block_size)
    }

    fn load_step_state(&self, reset_stream_state: bool) -> ModelState<B> {
        if !self.tbptt_persist_across_steps {
            return self.model.init_state();
        }
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut runtime = streaming_state_store()
            .lock()
            .expect("streaming tbptt runtime lock poisoned");
        if reset_stream_state {
            runtime.remove(&key);
        }
        runtime
            .remove(&key)
            .and_then(|state| state.downcast::<ModelState<B>>().ok().map(|state| *state))
            .unwrap_or_else(|| self.model.init_state())
    }

    fn store_step_state(&self, mut state: ModelState<B>) {
        if !self.tbptt_persist_across_steps {
            return;
        }
        state.detach_in_place();
        let key = (self.streaming_runtime_key, TypeId::of::<B>());
        let mut runtime = streaming_state_store()
            .lock()
            .expect("streaming tbptt runtime lock poisoned");
        runtime.insert(key, Box::new(state));
    }

    #[cfg(test)]
    fn peek_step_state_for_test(&self) -> Option<ModelState<B>> {
        streaming_state_store()
            .lock()
            .expect("streaming tbptt runtime lock poisoned")
            .get(&(self.streaming_runtime_key, TypeId::of::<B>()))
            .and_then(|state| {
                state
                    .downcast_ref::<ModelState<B>>()
                    .map(|state| state.clone())
            })
    }

    fn slice_tokens(
        tensor: Tensor<B, 2, Int>,
        batch_size: usize,
        start: usize,
        end: usize,
    ) -> Tensor<B, 2, Int> {
        tensor.slice([0..batch_size, start..end])
    }

    fn slice_batch(
        tensor: Tensor<B, 2, Int>,
        batch_start: usize,
        batch_end: usize,
    ) -> Tensor<B, 2, Int> {
        let [_batch_size, block_size] = tensor.shape().dims();
        tensor.slice([batch_start..batch_end, 0..block_size])
    }

    fn pipeline_enabled(&self) -> bool {
        self.pipeline_plan.is_some()
    }

    fn forward_loss_with_pipeline(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> (Tensor<B, 1>, Tensor<B, 3>, Tensor<B, 3>) {
        let plan = self
            .pipeline_plan
            .as_ref()
            .expect("forward_loss_with_pipeline requires a pipeline plan");
        assert!(
            !self.tbptt_persist_across_steps,
            "pipeline execution does not support tbptt_persist_across_steps"
        );
        assert!(
            self.tbptt_chunk_size.is_none(),
            "pipeline execution does not support tbptt chunking"
        );

        let [batch_size, _block_size] = inputs.shape().dims();
        let ranges = split_microbatch_ranges(batch_size, plan.microbatches)
            .expect("pipeline execution requires batch_size >= microbatches");
        let chunk_inputs = ranges
            .iter()
            .map(|range| Self::slice_batch(inputs.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_targets = ranges
            .iter()
            .map(|range| Self::slice_batch(targets.clone(), range.start, range.end))
            .collect::<Vec<_>>();
        let chunk_masks = ranges
            .iter()
            .map(|range| {
                summary_event_mask
                    .clone()
                    .map(|mask| Self::slice_batch(mask, range.start, range.end))
            })
            .collect::<Vec<_>>();

        let mut chunk_states = (0..plan.microbatches)
            .map(|_| self.model.init_state())
            .collect::<Vec<_>>();
        let mut pipeline_states = vec![None; plan.microbatches];

        for event in plan.events.iter().filter(|event| {
            matches!(
                event.kind,
                burn_dragon_train::train::pipeline::PipelineEventKind::Forward
            )
        }) {
            let microbatch_id = event.microbatch_id;
            if pipeline_states[microbatch_id].is_none() {
                pipeline_states[microbatch_id] = Some(
                    self.model
                        .begin_language_pipeline(chunk_inputs[microbatch_id].clone()),
                );
            }
            let assignment = plan.assignment(event.virtual_stage_id).clone();
            let state = &mut chunk_states[microbatch_id];
            let stage_state = pipeline_states[microbatch_id]
                .take()
                .expect("microbatch stage state");
            pipeline_states[microbatch_id] =
                Some(self.model.forward_language_pipeline_stage_with_state(
                    stage_state,
                    state,
                    assignment.layer_range.clone(),
                    chunk_masks[microbatch_id].clone(),
                ));
        }

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut hidden_chunks = Vec::with_capacity(plan.microbatches);
        let mut logits_chunks = Vec::with_capacity(plan.microbatches);
        for microbatch_id in 0..plan.microbatches {
            let (hidden, logits) = self.model.finish_language_pipeline_with_state(
                pipeline_states[microbatch_id]
                    .take()
                    .expect("pipeline state after scheduled forward"),
                &mut chunk_states[microbatch_id],
            );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let chunk_loss =
                language_model_loss::<B>(logits.clone(), chunk_targets[microbatch_id].clone())
                    .mul_scalar(weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });
            hidden_chunks.push(hidden);
            logits_chunks.push(logits);
        }

        (
            total_loss.expect("pipeline forward should produce at least one microbatch loss"),
            Tensor::cat(hidden_chunks, 0),
            Tensor::cat(logits_chunks, 0),
        )
    }

    fn forward_loss_with_tbptt(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
        chunk_size: usize,
        state: &mut ModelState<B>,
    ) -> (Tensor<B, 1>, u128) {
        let [batch_size, block_size] = inputs.shape().dims();
        debug_assert!(chunk_size > 0 && chunk_size < block_size);

        let mut total_loss: Option<Tensor<B, 1>> = None;
        let mut total_forward_ns = 0u128;

        for start in (0..block_size).step_by(chunk_size) {
            let end = (start + chunk_size).min(block_size);
            let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
            let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
            let chunk_summary_event_mask = summary_event_mask
                .clone()
                .map(|mask| Self::slice_tokens(mask, batch_size, start, end));

            let chunk_forward_start = Instant::now();
            let logits = if let Some(mask) = chunk_summary_event_mask {
                self.model
                    .forward_with_state_and_summary_event_mask(chunk_inputs, mask, state)
            } else {
                self.model.forward_with_state(chunk_inputs, state)
            };
            total_forward_ns += chunk_forward_start.elapsed().as_nanos();

            let chunk_weight = (end - start) as f32 / block_size as f32;
            let chunk_loss =
                language_model_loss::<B>(logits, chunk_targets).mul_scalar(chunk_weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });

            if end < block_size {
                state.detach_in_place();
            }
        }

        (
            total_loss.expect("tbptt forward should produce at least one loss chunk"),
            total_forward_ns,
        )
    }
}

impl<B: AutodiffBackend> TrainStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelTrainItem<B>;

    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let prof_enabled = crate::train::profile::enabled();
        let detail_prof_enabled = crate::train::profile::detail_enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let inputs = batch.inputs;
        let targets = batch.targets;
        let summary_event_mask = batch.summary_event_mask;
        let reset_stream_state = batch.reset_stream_state;
        let [_batch_size, block_size] = inputs.shape().dims();
        let tbptt_chunk_size = self.effective_tbptt_chunk_size(block_size);
        let probe_inputs = detail_prof_enabled.then(|| inputs.clone());
        let probe_summary_event_mask = detail_prof_enabled
            .then(|| summary_event_mask.clone())
            .flatten();
        let mut step_state = self.load_step_state(reset_stream_state);
        let (loss, probe_hidden, probe_logits, forward_ns) = if self.pipeline_enabled() {
            let forward_start = Instant::now();
            let (loss, hidden, logits) =
                self.forward_loss_with_pipeline(inputs, targets.clone(), summary_event_mask);
            step_state = self.model.init_state();
            (
                loss,
                Some(hidden),
                Some(logits),
                forward_start.elapsed().as_nanos(),
            )
        } else if let Some(chunk_size) = tbptt_chunk_size {
            if detail_prof_enabled {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut hidden_chunks = Vec::new();
                let mut logits_chunks = Vec::new();
                let mut total_forward_ns = 0u128;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));
                    let chunk_forward_start = Instant::now();
                    let (hidden, logits) = if let Some(mask) = chunk_summary_event_mask {
                        self.model
                            .forward_with_hidden_and_state_and_summary_event_mask(
                                chunk_inputs,
                                mask,
                                &mut step_state,
                            )
                    } else {
                        self.model
                            .forward_with_hidden_and_state(chunk_inputs, &mut step_state)
                    };
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();
                    hidden_chunks.push(hidden);
                    logits_chunks.push(logits);
                    if end < block_size {
                        step_state.detach_in_place();
                    }
                }
                let hidden = Tensor::cat(hidden_chunks, 1);
                let logits = Tensor::cat(logits_chunks, 1);
                let loss = language_model_loss::<B>(logits.clone(), targets.clone());
                (loss, Some(hidden), Some(logits), total_forward_ns)
            } else {
                let (loss, total_forward_ns) = self.forward_loss_with_tbptt(
                    inputs,
                    targets.clone(),
                    summary_event_mask,
                    chunk_size,
                    &mut step_state,
                );
                (loss, None, None, total_forward_ns)
            }
        } else if detail_prof_enabled {
            if let Some(summary_event_mask) = summary_event_mask {
                let (hidden, logits) = self
                    .model
                    .forward_with_hidden_and_state_and_summary_event_mask(
                        inputs,
                        summary_event_mask,
                        &mut step_state,
                    );
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = language_model_loss::<B>(logits.clone(), targets.clone());
                (loss, Some(hidden), Some(logits), forward_ns)
            } else {
                let (hidden, logits) = self
                    .model
                    .forward_with_hidden_and_state(inputs, &mut step_state);
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = language_model_loss::<B>(logits.clone(), targets.clone());
                (loss, Some(hidden), Some(logits), forward_ns)
            }
        } else {
            let logits = if let Some(summary_event_mask) = summary_event_mask {
                self.model.forward_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                )
            } else {
                self.model.forward_with_state(inputs, &mut step_state)
            };
            let forward_ns = forward_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();
            let loss = language_model_loss::<B>(logits.clone(), targets.clone());
            (loss, None, Some(logits), forward_ns)
        };
        self.store_step_state(step_state);

        let probe_targets = (prof_enabled && detail_prof_enabled).then(|| targets.clone());
        let probe_logits = (prof_enabled && detail_prof_enabled)
            .then(|| probe_logits.clone().expect("probe logits").detach());
        let probe_hidden = probe_hidden.map(|hidden| hidden.detach());

        let loss_backward_start = prof_enabled.then(Instant::now);
        let grads = loss.backward();
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
            if detail_prof_enabled {
                let mut embed_probe_ns = 0;
                let mut first_layer_forward_probe_ns = 0;
                let mut first_layer_probe_ns = 0;
                let mut logits_loss_probe_ns = 0;
                let mut hidden_logits_loss_probe_ns = 0;
                let mut hidden_model_forward_probe_ns = 0;
                let mut hidden_model_probe_ns = 0;
                if let Some(probe_inputs) = probe_inputs.clone() {
                    let embed_start = Instant::now();
                    let probe_embedded = self.model.embed_tokens(probe_inputs);
                    let embed_loss = probe_embedded.clone().tanh().powf_scalar(2.0).mean();
                    let _embed_grads = embed_loss.backward();
                    let _ = B::sync(&probe_embedded.device());
                    embed_probe_ns = embed_start.elapsed().as_nanos();

                    let first_layer_forward_start = Instant::now();
                    let first_layer_forward_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded.clone().detach(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let _ = B::sync(&first_layer_forward_hidden.device());
                    first_layer_forward_probe_ns = first_layer_forward_start.elapsed().as_nanos();

                    let first_layer_start = Instant::now();
                    let probe_embedded_leaf = probe_embedded.detach().require_grad();
                    let first_layer_hidden = self
                        .model
                        .forward_hidden_prefix_layers_from_embedded_for_profile(
                            probe_embedded_leaf.clone(),
                            1,
                            probe_summary_event_mask.clone(),
                        );
                    let first_layer_loss =
                        first_layer_hidden.clone().tanh().powf_scalar(2.0).mean();
                    let _first_layer_grads = first_layer_loss.backward();
                    let _ = B::sync(&probe_embedded_leaf.device());
                    first_layer_probe_ns = first_layer_start.elapsed().as_nanos();
                }
                if let (Some(probe_targets), Some(probe_logits), Some(probe_hidden)) =
                    (probe_targets, probe_logits, probe_hidden)
                {
                    let hidden_model_forward_start = Instant::now();
                    let probe_hidden_forward = if let Some(mask) = probe_summary_event_mask.clone()
                    {
                        let mut probe_hidden_forward_state = self.model.init_state();
                        self.model
                            .forward_with_hidden_and_state_and_summary_event_mask(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                                mask,
                                &mut probe_hidden_forward_state,
                            )
                            .0
                    } else {
                        self.model
                            .forward_with_hidden(
                                probe_inputs
                                    .clone()
                                    .expect("probe inputs for hidden forward probe"),
                            )
                            .0
                    };
                    let _ = B::sync(&probe_hidden_forward.device());
                    hidden_model_forward_probe_ns = hidden_model_forward_start.elapsed().as_nanos();

                    let logits_only_start = Instant::now();
                    let probe_logits_leaf = probe_logits.require_grad();
                    let logits_only_loss =
                        language_model_loss::<B>(probe_logits_leaf.clone(), probe_targets.clone());
                    let logits_only_grads = logits_only_loss.backward();
                    let _ = probe_logits_leaf
                        .grad(&logits_only_grads)
                        .expect("probe logits grad")
                        .sum()
                        .into_data();
                    logits_loss_probe_ns = logits_only_start.elapsed().as_nanos();

                    let hidden_logits_start = Instant::now();
                    let probe_hidden_leaf = probe_hidden.require_grad();
                    let hidden_logits_loss = language_model_loss::<B>(
                        self.model.logits_from_hidden(probe_hidden_leaf.clone()),
                        probe_targets,
                    );
                    let hidden_logits_grads = hidden_logits_loss.backward();
                    let _ = probe_hidden_leaf
                        .grad(&hidden_logits_grads)
                        .expect("probe hidden grad")
                        .sum()
                        .into_data();
                    hidden_logits_loss_probe_ns = hidden_logits_start.elapsed().as_nanos();
                }
                if let Some(probe_inputs) = probe_inputs {
                    let hidden_model_start = Instant::now();
                    let probe_hidden_model =
                        if let Some(summary_event_mask) = probe_summary_event_mask {
                            let mut probe_state = self.model.init_state();
                            self.model
                                .forward_with_hidden_and_state_and_summary_event_mask(
                                    probe_inputs,
                                    summary_event_mask,
                                    &mut probe_state,
                                )
                                .0
                        } else {
                            self.model.forward_with_hidden(probe_inputs).0
                        };
                    let hidden_model_loss =
                        probe_hidden_model.clone().tanh().powf_scalar(2.0).mean();
                    let _hidden_model_grads = hidden_model_loss.backward();
                    let _ = B::sync(&probe_hidden_model.device());
                    hidden_model_probe_ns = hidden_model_start.elapsed().as_nanos();
                }
                crate::train::profile::record_detail_probe(
                    embed_probe_ns,
                    first_layer_forward_probe_ns,
                    first_layer_probe_ns,
                    logits_loss_probe_ns,
                    hidden_logits_loss_probe_ns,
                    hidden_model_forward_probe_ns,
                    hidden_model_probe_ns,
                );
            }
        }

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelOutput<B>;

    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        if self.pipeline_enabled() {
            let (loss, _hidden, _logits) = self.forward_loss_with_pipeline(
                batch.inputs,
                batch.targets,
                batch.summary_event_mask,
            );
            return LanguageModelOutput::new(loss);
        }
        let logits = if let Some(summary_event_mask) = batch.summary_event_mask {
            if let Some(chunk_size) =
                self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
            {
                let [batch_size, block_size] = batch.inputs.shape().dims();
                let mut state = self.model.init_state();
                let mut loss: Option<Tensor<B, 1>> = None;
                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs =
                        Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                    let chunk_targets =
                        Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                    let chunk_mask =
                        Self::slice_tokens(summary_event_mask.clone(), batch_size, start, end);
                    let logits = self.model.forward_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        &mut state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss =
                        language_model_loss::<B>(logits, chunk_targets).mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                return LanguageModelOutput::new(
                    loss.expect("tbptt valid step should produce at least one loss chunk"),
                );
            } else if fast_train_enabled() {
                self.model
                    .forward_fast_with_summary_event_mask(batch.inputs, summary_event_mask)
            } else {
                self.model
                    .forward_with_summary_event_mask(batch.inputs, summary_event_mask)
            }
        } else if let Some(chunk_size) =
            self.effective_tbptt_chunk_size(batch.inputs.shape().dims::<2>()[1])
        {
            let [batch_size, block_size] = batch.inputs.shape().dims();
            let mut state = self.model.init_state();
            let mut loss: Option<Tensor<B, 1>> = None;
            for start in (0..block_size).step_by(chunk_size) {
                let end = (start + chunk_size).min(block_size);
                let chunk_inputs = Self::slice_tokens(batch.inputs.clone(), batch_size, start, end);
                let chunk_targets =
                    Self::slice_tokens(batch.targets.clone(), batch_size, start, end);
                let logits = self.model.forward_with_state(chunk_inputs, &mut state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss =
                    language_model_loss::<B>(logits, chunk_targets).mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            return LanguageModelOutput::new(
                loss.expect("tbptt valid step should produce at least one loss chunk"),
            );
        } else if fast_train_enabled() {
            self.model.forward_fast(batch.inputs)
        } else {
            self.model.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_ndarray::NdArray;

    type TestBackend = Autodiff<NdArray<f32>>;
    type TestValidBackend = ValidBackend<TestBackend>;

    fn make_batch<B: BackendTrait>(
        device: &B::Device,
        inputs: &[i64],
        targets: &[i64],
        shape: [usize; 2],
    ) -> SequenceBatch<B> {
        SequenceBatch::new(
            Tensor::<B, 2, Int>::from_data(TensorData::new(inputs.to_vec(), shape), device),
            Tensor::<B, 2, Int>::from_data(TensorData::new(targets.to_vec(), shape), device),
            None,
        )
    }

    fn tiny_model_config() -> BDHConfig {
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 16,
            ..Default::default()
        }
    }

    fn pipeline_plan_for_tiny_model() -> PipelinePlan {
        build_pipeline_plan(
            tiny_model_config().n_layer,
            &burn_dragon_train::ParallelPipelineConfig {
                enabled: true,
                stage_count: 2,
                virtual_stages_per_rank: 1,
                schedule: burn_dragon_train::PipelineScheduleKind::Interleaved1f1b,
                microbatches: 2,
                ..Default::default()
            },
        )
        .expect("pipeline plan")
    }

    fn loss_scalar<B: BackendTrait>(output: LanguageModelOutput<B>) -> f32 {
        <LanguageModelOutput<B> as Adaptor<LossValue<B>>>::adapt(&output)
            .value()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("loss vec")[0]
    }

    #[test]
    fn tbptt_valid_step_matches_full_loss_value() {
        let device = <TestValidBackend as BackendTrait>::Device::default();
        let model = BDH::<TestValidBackend>::new(tiny_model_config(), &device);
        let baseline = LanguageTrainModel::new(model.clone());
        let tbptt = LanguageTrainModel::new(model).with_tbptt_chunk_size(Some(2));
        let batch = make_batch::<TestValidBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let baseline_loss = loss_scalar(ValidStep::step(&baseline, batch.clone()));
        let tbptt_loss = loss_scalar(ValidStep::step(&tbptt, batch));
        assert!(
            (baseline_loss - tbptt_loss).abs() < 1.0e-5,
            "expected tbptt loss to match full loss value, got baseline={baseline_loss} tbptt={tbptt_loss}"
        );
    }

    #[test]
    fn tbptt_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(tiny_model_config(), &device))
            .with_tbptt_chunk_size(Some(2));
        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let output = TrainStep::step(&model, batch);
        let synced = output.item.sync();
        let loss = loss_scalar(synced);
        assert!(loss.is_finite(), "tbptt train loss must be finite");
    }

    #[test]
    fn pipeline_valid_step_matches_full_loss_value() {
        let device = <TestValidBackend as BackendTrait>::Device::default();
        let model = BDH::<TestValidBackend>::new(tiny_model_config(), &device);
        let baseline = LanguageTrainModel::new(model.clone());
        let pipelined =
            LanguageTrainModel::new(model).with_pipeline_plan(Some(pipeline_plan_for_tiny_model()));
        let batch = make_batch::<TestValidBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );

        let baseline_loss = loss_scalar(ValidStep::step(&baseline, batch.clone()));
        let pipeline_loss = loss_scalar(ValidStep::step(&pipelined, batch));
        assert!(
            (baseline_loss - pipeline_loss).abs() < 1.0e-5,
            "expected pipeline loss to match full loss value, got baseline={baseline_loss} pipeline={pipeline_loss}"
        );
    }

    #[test]
    fn pipeline_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(tiny_model_config(), &device))
            .with_pipeline_plan(Some(pipeline_plan_for_tiny_model()));
        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );

        let output = TrainStep::step(&model, batch);
        let synced = output.item.sync();
        let loss = loss_scalar(synced);
        assert!(loss.is_finite(), "pipeline train loss must be finite");
    }

    #[test]
    fn streaming_tbptt_persists_state_across_steps_with_minimal_block() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(tiny_model_config(), &device))
            .with_tbptt_persist_across_steps(true);

        let batch_a = make_batch::<TestBackend>(&device, &[0, 1], &[1, 2], [1, 2])
            .with_reset_stream_state(true);
        let batch_b = make_batch::<TestBackend>(&device, &[2, 3], &[3, 4], [1, 2])
            .with_reset_stream_state(false);

        let _ = TrainStep::step(&model, batch_a);
        let persisted_state = model
            .peek_step_state_for_test()
            .expect("persisted state after first chunk");
        assert_eq!(persisted_state.position, 2);

        let expected_loss = {
            let mut state = persisted_state.clone();
            let logits = model
                .model
                .forward_with_state(batch_b.inputs.clone(), &mut state);
            loss_scalar(LanguageModelOutput::new(
                language_model_loss::<TestBackend>(logits, batch_b.targets.clone()),
            ))
        };

        let loss_b = loss_scalar(TrainStep::step(&model, batch_b).item.sync());
        assert!(
            (loss_b - expected_loss).abs() < 1.0e-5,
            "expected persisted-stream second chunk loss to match direct carried-state loss, got loss_b={loss_b} expected_loss={expected_loss}"
        );
    }

    #[test]
    fn streaming_tbptt_reset_starts_fresh_sequence() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(tiny_model_config(), &device))
            .with_tbptt_persist_across_steps(true);

        let first = make_batch::<TestBackend>(&device, &[0, 1], &[1, 2], [1, 2])
            .with_reset_stream_state(true);
        let second = make_batch::<TestBackend>(&device, &[2, 3], &[3, 4], [1, 2])
            .with_reset_stream_state(true);

        let _ = TrainStep::step(&model, first);
        let persisted_state = model
            .peek_step_state_for_test()
            .expect("persisted state after first chunk");
        assert_eq!(persisted_state.position, 2);

        let expected_loss = {
            let mut fresh_state = model.model.init_state();
            let logits = model
                .model
                .forward_with_state(second.inputs.clone(), &mut fresh_state);
            loss_scalar(LanguageModelOutput::new(
                language_model_loss::<TestBackend>(logits, second.targets.clone()),
            ))
        };
        let second_loss = loss_scalar(TrainStep::step(&model, second).item.sync());
        let reset_state = model
            .peek_step_state_for_test()
            .expect("persisted state after reset step");
        assert_eq!(reset_state.position, 2);
        assert!(
            (second_loss - expected_loss).abs() < 1.0e-5,
            "expected reset streaming step to match direct fresh-state loss, got second_loss={second_loss} expected_loss={expected_loss}"
        );
    }
}
