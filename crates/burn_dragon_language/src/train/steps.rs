use crate::train::prelude::*;
use burn::module::Ignored;
use burn_dragon_core::ModelState;
use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
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

#[derive(Clone, Debug, Default)]
struct GradientScaleSchedule {
    param_scale_rules: Arc<HashMap<burn::module::ParamId, ParamScaleScheduleRule>>,
    shared_lowrank_param_ids: Arc<Vec<burn::module::ParamId>>,
    backbone_grad_scale: Option<f32>,
    backbone_grad_scale_steps: usize,
    backbone_param_ids: Arc<HashSet<burn::module::ParamId>>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ParamScaleScheduleRule {
    initial_scale: f32,
    final_scale: f32,
    start_step_index: usize,
    end_step_index: usize,
}

impl ParamScaleScheduleRule {
    fn constant(scale: f32) -> Self {
        Self {
            initial_scale: scale,
            final_scale: scale,
            start_step_index: 0,
            end_step_index: 0,
        }
    }

    fn for_total_steps(
        initial_scale: f32,
        schedule: Option<&crate::config::train::ModuleLrScaleScheduleConfig>,
        total_steps: usize,
    ) -> Self {
        let Some(schedule) = schedule else {
            return Self::constant(initial_scale);
        };
        let total_steps = total_steps.max(1);
        let last_step_index = total_steps.saturating_sub(1);
        let start_step_index =
            ((last_step_index as f32) * schedule.start_fraction.clamp(0.0, 1.0)).round() as usize;
        let end_step_index =
            ((last_step_index as f32) * schedule.end_fraction.clamp(0.0, 1.0)).round() as usize;
        Self {
            initial_scale,
            final_scale: schedule.final_scale,
            start_step_index,
            end_step_index,
        }
    }

    fn scale_for_step_index(self, step_index: usize) -> f32 {
        if step_index <= self.start_step_index {
            return self.initial_scale;
        }
        if step_index >= self.end_step_index {
            return self.final_scale;
        }
        if self.end_step_index <= self.start_step_index {
            return self.final_scale;
        }
        let progress = (step_index - self.start_step_index) as f32
            / (self.end_step_index - self.start_step_index) as f32;
        self.initial_scale + (self.final_scale - self.initial_scale) * progress
    }
}

impl GradientScaleSchedule {
    fn from_training<B: BackendTrait>(
        model: &BDH<B>,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        let param_scale_rules =
            Self::build_param_scale_rules(model, &training.module_lr_scales, total_steps);
        let shared_lowrank_param_ids = vec![
            model.shared_lowrank_param_ids().rwkv_time_decay,
            model.shared_lowrank_param_ids().encoder,
            model.shared_lowrank_param_ids().encoder_v,
            model.shared_lowrank_param_ids().decoder,
        ];
        let Some(backbone_grad_scale) = training.init_transfer.backbone_grad_scale else {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        };
        let Some(backbone_grad_scale_steps) = training.init_transfer.backbone_grad_scale_steps
        else {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        };
        if backbone_grad_scale_steps == 0 || (backbone_grad_scale - 1.0).abs() <= f32::EPSILON {
            return Self {
                param_scale_rules: Arc::new(param_scale_rules),
                shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
                ..Self::default()
            };
        }
        let backbone_param_ids = model
            .transferred_backbone_param_ids(
                training.init_transfer.preserve_fresh_decoder,
                training.init_transfer.preserve_fresh_norm,
            )
            .into_iter()
            .collect::<HashSet<_>>();
        Self {
            param_scale_rules: Arc::new(param_scale_rules),
            shared_lowrank_param_ids: Arc::new(shared_lowrank_param_ids),
            backbone_grad_scale: Some(backbone_grad_scale),
            backbone_grad_scale_steps,
            backbone_param_ids: Arc::new(backbone_param_ids),
        }
    }

    fn build_param_scale_rules<B: BackendTrait>(
        model: &BDH<B>,
        entries: &[crate::config::train::ModuleLrScaleEntry],
        total_steps: usize,
    ) -> HashMap<burn::module::ParamId, ParamScaleScheduleRule> {
        let mut scales = HashMap::new();
        for entry in entries {
            for param_id in model.language_module_lr_scale_param_ids(entry.target) {
                scales.insert(
                    param_id,
                    ParamScaleScheduleRule::for_total_steps(
                        entry.scale,
                        entry.schedule.as_ref(),
                        total_steps,
                    ),
                );
            }
        }
        scales
    }

    fn mean_scale_for_param_ids(
        rules: &HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
        param_ids: &[burn::module::ParamId],
        step_index: usize,
    ) -> f32 {
        if param_ids.is_empty() {
            return 1.0;
        }
        let sum = param_ids
            .iter()
            .map(|param_id| {
                rules
                    .get(param_id)
                    .copied()
                    .unwrap_or_else(|| ParamScaleScheduleRule::constant(1.0))
                    .scale_for_step_index(step_index)
            })
            .sum::<f32>();
        sum / param_ids.len() as f32
    }

    fn shared_lowrank_target_lr_scale_for_step_index(&self, step_index: usize) -> f32 {
        Self::mean_scale_for_param_ids(
            self.param_scale_rules.as_ref(),
            self.shared_lowrank_param_ids.as_ref(),
            step_index,
        )
    }
}

fn scale_gradients_by_schedule<B, M>(
    module: &M,
    grads: &mut GradientsParams,
    param_scale_rules: &HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
    step_index: usize,
    extra_param_ids: &HashSet<burn::module::ParamId>,
    extra_scale: Option<f32>,
) where
    B: AutodiffBackend,
    M: AutodiffModule<B>,
{
    let has_static_scales = param_scale_rules
        .values()
        .any(|rule| (rule.scale_for_step_index(step_index) - 1.0).abs() > f32::EPSILON);
    let has_extra_scale = extra_scale
        .is_some_and(|scale| (scale - 1.0).abs() > f32::EPSILON && !extra_param_ids.is_empty());
    if !has_static_scales && !has_extra_scale {
        return;
    }

    struct GradientScaleVisitor<'a, B: AutodiffBackend> {
        grads: &'a mut GradientsParams,
        param_scale_rules: &'a HashMap<burn::module::ParamId, ParamScaleScheduleRule>,
        step_index: usize,
        extra_param_ids: &'a HashSet<burn::module::ParamId>,
        extra_scale: Option<f32>,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: AutodiffBackend> burn::module::ModuleVisitor<B> for GradientScaleVisitor<'_, B> {
        fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<B, D>>) {
            let mut scale = self
                .param_scale_rules
                .get(&param.id)
                .copied()
                .unwrap_or_else(|| ParamScaleScheduleRule::constant(1.0))
                .scale_for_step_index(self.step_index);
            if let Some(extra_scale) = self.extra_scale
                && self.extra_param_ids.contains(&param.id)
            {
                scale *= extra_scale;
            }
            if (scale - 1.0).abs() <= f32::EPSILON {
                return;
            }
            if let Some(grad) = self.grads.remove::<B::InnerBackend, D>(param.id) {
                self.grads.register(param.id, grad.mul_scalar(scale));
            }
        }
    }

    let mut visitor = GradientScaleVisitor::<B> {
        grads,
        param_scale_rules,
        step_index,
        extra_param_ids,
        extra_scale,
        _marker: std::marker::PhantomData,
    };
    module.visit(&mut visitor);
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
    gradient_scale_schedule: Ignored<GradientScaleSchedule>,
    gradient_scale_step: Ignored<Arc<AtomicUsize>>,
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(crate) fn new(model: BDH<B>) -> Self {
        Self {
            model,
            tbptt_chunk_size: None,
            pipeline_plan: Ignored(None),
            tbptt_persist_across_steps: false,
            streaming_runtime_key: next_streaming_runtime_key(),
            gradient_scale_schedule: Ignored(GradientScaleSchedule::default()),
            gradient_scale_step: Ignored(Arc::new(AtomicUsize::new(0))),
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

    pub(crate) fn with_gradient_scale_schedule(
        mut self,
        training: &TrainingHyperparameters,
        total_steps: usize,
    ) -> Self {
        self.gradient_scale_schedule = Ignored(GradientScaleSchedule::from_training(
            &self.model,
            training,
            total_steps,
        ));
        self
    }

    pub(crate) fn continual_backprop_target_lr_scale(&self) -> f32 {
        let step_index = self
            .gradient_scale_step
            .0
            .load(Ordering::Relaxed)
            .saturating_sub(1);
        self.gradient_scale_schedule
            .0
            .shared_lowrank_target_lr_scale_for_step_index(step_index)
    }

    fn apply_gradient_scale_schedule(&self, mut grads: GradientsParams) -> GradientsParams
    where
        B: AutodiffBackend,
    {
        let step = self.gradient_scale_step.0.fetch_add(1, Ordering::Relaxed) + 1;
        let step_index = step.saturating_sub(1);
        let extra_scale = self
            .gradient_scale_schedule
            .0
            .backbone_grad_scale
            .filter(|_| step <= self.gradient_scale_schedule.0.backbone_grad_scale_steps);
        scale_gradients_by_schedule::<B, _>(
            self,
            &mut grads,
            self.gradient_scale_schedule.0.param_scale_rules.as_ref(),
            step_index,
            self.gradient_scale_schedule.0.backbone_param_ids.as_ref(),
            extra_scale,
        );
        grads
    }

    fn effective_tbptt_chunk_size(&self, block_size: usize) -> Option<usize> {
        self.tbptt_chunk_size
            .filter(|chunk_size| *chunk_size > 0 && *chunk_size < block_size)
    }

    fn load_step_state(&self, reset_stream_state: bool) -> ModelState<B> {
        if !self.tbptt_persist_across_steps {
            return self.model.init_state_ephemeral();
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

    fn language_loss_from_hidden(
        &self,
        hidden: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        self.model.language_loss_from_hidden(hidden, targets)
    }

    fn language_loss_from_logits(
        &self,
        logits: Tensor<B, 3>,
        targets: Tensor<B, 2, Int>,
    ) -> Tensor<B, 1> {
        self.model.language_loss_from_logits(logits, targets)
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
        let factorized_head = self.model.uses_factorized_language_head();

        let mut chunk_states = (0..plan.microbatches)
            .map(|_| self.model.init_state_ephemeral())
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
            let hidden = self.model.finish_language_pipeline_hidden_with_state(
                pipeline_states[microbatch_id]
                    .take()
                    .expect("pipeline state after scheduled forward"),
                &mut chunk_states[microbatch_id],
            );
            let weight = ranges[microbatch_id].len() as f32 / batch_size as f32;
            let chunk_loss = self
                .language_loss_from_hidden(hidden.clone(), chunk_targets[microbatch_id].clone())
                .mul_scalar(weight);
            total_loss = Some(match total_loss {
                Some(accumulated) => accumulated + chunk_loss,
                None => chunk_loss,
            });
            if !factorized_head {
                logits_chunks.push(self.model.logits_from_hidden(hidden.clone()));
            }
            hidden_chunks.push(hidden);
        }

        (
            total_loss.expect("pipeline forward should produce at least one microbatch loss"),
            Tensor::cat(hidden_chunks, 0),
            if logits_chunks.is_empty() {
                let device = inputs.device();
                Tensor::<B, 3>::zeros([batch_size, 0, 1], &device)
            } else {
                Tensor::cat(logits_chunks, 0)
            },
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
        let memory_prof_enabled = prof_enabled && crate::train::profile::memory_enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let inputs = batch.inputs;
        let targets = batch.targets;
        let summary_event_mask = batch.summary_event_mask;
        let reset_stream_state = batch.reset_stream_state;
        let step_device = memory_prof_enabled.then(|| inputs.device());
        let step_memory_before = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));
        let [_batch_size, block_size] = inputs.shape().dims();
        let tbptt_chunk_size = self.effective_tbptt_chunk_size(block_size);
        let factorized_head = self.model.uses_factorized_language_head();
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
                (!factorized_head).then_some(logits),
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
                    let hidden = if let Some(mask) = chunk_summary_event_mask {
                        self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        )
                    } else {
                        self.model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state)
                    };
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();
                    hidden_chunks.push(hidden);
                    if !factorized_head {
                        logits_chunks.push(
                            self.model
                                .logits_from_hidden(hidden_chunks.last().expect("hidden").clone()),
                        );
                    }
                    if end < block_size {
                        step_state.detach_in_place();
                    }
                }
                let hidden = Tensor::cat(hidden_chunks, 1);
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits = (!factorized_head).then(|| Tensor::cat(logits_chunks, 1));
                (loss, Some(hidden), logits, total_forward_ns)
            } else {
                let [batch_size, block_size] = inputs.shape().dims();
                let mut total_forward_ns = 0u128;
                let mut total_backward_ns = 0u128;
                let mut total_loss: Option<Tensor<B, 1>> = None;
                let mut accumulator = GradientsAccumulator::new();

                for start in (0..block_size).step_by(chunk_size) {
                    let end = (start + chunk_size).min(block_size);
                    let chunk_inputs = Self::slice_tokens(inputs.clone(), batch_size, start, end);
                    let chunk_targets = Self::slice_tokens(targets.clone(), batch_size, start, end);
                    let chunk_summary_event_mask = summary_event_mask
                        .clone()
                        .map(|mask| Self::slice_tokens(mask, batch_size, start, end));

                    let chunk_forward_start = Instant::now();
                    let chunk_loss = if let Some(mask) = chunk_summary_event_mask {
                        let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                            chunk_inputs,
                            mask,
                            &mut step_state,
                        );
                        self.language_loss_from_hidden(hidden, chunk_targets.clone())
                    } else {
                        let hidden = self
                            .model
                            .forward_hidden_with_state(chunk_inputs, &mut step_state);
                        self.language_loss_from_hidden(hidden, chunk_targets.clone())
                    };
                    total_forward_ns += chunk_forward_start.elapsed().as_nanos();

                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = chunk_loss.mul_scalar(chunk_weight);
                    total_loss = Some(match total_loss {
                        Some(accumulated) => accumulated + chunk_loss.clone().detach(),
                        None => chunk_loss.clone().detach(),
                    });

                    let chunk_backward_start = Instant::now();
                    let chunk_grads = chunk_loss.backward();
                    total_backward_ns += chunk_backward_start.elapsed().as_nanos();
                    accumulator.accumulate(self, GradientsParams::from_grads(chunk_grads, self));

                    if end < block_size {
                        step_state.detach_in_place();
                    }
                }

                self.store_step_state(step_state);

                let step_memory_after_forward = step_device
                    .as_ref()
                    .and_then(|device| device_memory_usage_safe::<B>(device));
                if prof_enabled {
                    crate::train::profile::record_train_step(total_forward_ns, total_backward_ns);
                    if let (Some(before), Some(after_forward), Some(device)) = (
                        step_memory_before,
                        step_memory_after_forward,
                        step_device.as_ref(),
                    ) {
                        let after_backward =
                            device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                        crate::train::profile::record_train_step_memory(
                            before.reserved_bytes,
                            before.in_use_bytes,
                            after_forward.reserved_bytes,
                            after_forward.in_use_bytes,
                            after_backward.reserved_bytes,
                            after_backward.in_use_bytes,
                        );
                    }
                }

                return TrainOutput {
                    grads: self.apply_gradient_scale_schedule(accumulator.grads()),
                    item: LanguageModelTrainItem::new(
                        total_loss
                            .expect("tbptt train step should produce at least one loss chunk"),
                    ),
                };
            }
        } else if detail_prof_enabled {
            if let Some(summary_event_mask) = summary_event_mask {
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                );
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            } else {
                let hidden = self
                    .model
                    .forward_hidden_with_state(inputs, &mut step_state);
                let forward_ns = forward_start
                    .map(|start| start.elapsed().as_nanos())
                    .unwrap_or_default();
                let loss = self.language_loss_from_hidden(hidden.clone(), targets.clone());
                let logits =
                    (!factorized_head).then(|| self.model.logits_from_hidden(hidden.clone()));
                (loss, Some(hidden), logits, forward_ns)
            }
        } else {
            let hidden = if let Some(summary_event_mask) = summary_event_mask {
                self.model.forward_hidden_with_state_and_summary_event_mask(
                    inputs,
                    summary_event_mask,
                    &mut step_state,
                )
            } else {
                self.model
                    .forward_hidden_with_state(inputs, &mut step_state)
            };
            let forward_ns = forward_start
                .map(|start| start.elapsed().as_nanos())
                .unwrap_or_default();
            let loss = self.language_loss_from_hidden(hidden, targets.clone());
            (loss, None, None, forward_ns)
        };
        self.store_step_state(step_state);
        let step_memory_after_forward = step_device
            .as_ref()
            .and_then(|device| device_memory_usage_safe::<B>(device));

        let probe_targets = (prof_enabled && detail_prof_enabled).then(|| targets.clone());
        let probe_logits = if prof_enabled && detail_prof_enabled {
            probe_logits.clone().map(|logits| logits.detach())
        } else {
            None
        };
        let probe_hidden = probe_hidden.map(|hidden| hidden.detach());

        let loss_backward_start = prof_enabled.then(Instant::now);
        let grads = loss.backward();
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
            if let (Some(before), Some(after_forward), Some(device)) = (
                step_memory_before,
                step_memory_after_forward,
                step_device.as_ref(),
            ) {
                let after_backward = device_memory_usage_safe::<B>(device).unwrap_or(after_forward);
                crate::train::profile::record_train_step_memory(
                    before.reserved_bytes,
                    before.in_use_bytes,
                    after_forward.reserved_bytes,
                    after_forward.in_use_bytes,
                    after_backward.reserved_bytes,
                    after_backward.in_use_bytes,
                );
            }
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

        TrainOutput {
            grads: self.apply_gradient_scale_schedule(GradientsParams::from_grads(grads, self)),
            item: LanguageModelTrainItem::new(loss),
        }
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
        if let Some(summary_event_mask) = batch.summary_event_mask {
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
                    let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                        chunk_inputs,
                        chunk_mask,
                        &mut state,
                    );
                    let chunk_weight = (end - start) as f32 / block_size as f32;
                    let chunk_loss = self
                        .language_loss_from_hidden(hidden, chunk_targets)
                        .mul_scalar(chunk_weight);
                    loss = Some(match loss {
                        Some(accumulated) => accumulated + chunk_loss,
                        None => chunk_loss,
                    });
                }
                return LanguageModelOutput::new(
                    loss.expect("tbptt valid step should produce at least one loss chunk"),
                );
            } else {
                let mut state = self.model.init_state();
                let hidden = self.model.forward_hidden_with_state_and_summary_event_mask(
                    batch.inputs,
                    summary_event_mask,
                    &mut state,
                );
                let loss = self.language_loss_from_hidden(hidden, batch.targets);
                return LanguageModelOutput::new(loss);
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
                let hidden = self
                    .model
                    .forward_hidden_with_state(chunk_inputs, &mut state);
                let chunk_weight = (end - start) as f32 / block_size as f32;
                let chunk_loss = self
                    .language_loss_from_hidden(hidden, chunk_targets)
                    .mul_scalar(chunk_weight);
                loss = Some(match loss {
                    Some(accumulated) => accumulated + chunk_loss,
                    None => chunk_loss,
                });
            }
            return LanguageModelOutput::new(
                loss.expect("tbptt valid step should produce at least one loss chunk"),
            );
        } else {
            let hidden = self.model.forward_hidden(batch.inputs);
            let loss = self.language_loss_from_hidden(hidden, batch.targets);
            return LanguageModelOutput::new(loss);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::ParamId;
    use burn::tensor::TensorData;
    use burn_autodiff::Autodiff;
    use burn_dragon_core::{
        BitNetLowBitProtocol, LanguageHeadConfig, LanguageModuleLrScaleTarget,
        LowBitActivationFormat, LowBitQuantizationConfig, LowBitSavedActivationConfig,
        LowBitSavedActivationMode, LowBitTargetModule, LowBitTrainingMode, LowBitWeightFormat,
    };
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

    fn tiny_low_bit_model_config() -> BDHConfig {
        BDHConfig {
            quant: LowBitQuantizationConfig {
                enable: true,
                protocol: BitNetLowBitProtocol::BitnetB158,
                weight_format: LowBitWeightFormat::Ternary158,
                act_format: LowBitActivationFormat::Int8,
                target_modules: vec![LowBitTargetModule::Encoder, LowBitTargetModule::DecoderY],
                decoder_x_mode: LowBitWeightFormat::Fp16,
                ..Default::default()
            },
            ..tiny_model_config()
        }
    }

    fn tiny_low_bit_train_kernel_model_config() -> BDHConfig {
        let mut config = tiny_low_bit_model_config();
        config.quant.training_mode = LowBitTrainingMode::TrainKernelExp;
        config
    }

    fn tiny_low_bit_train_kernel_saved_activation_model_config() -> BDHConfig {
        let mut config = tiny_low_bit_train_kernel_model_config();
        config.quant.saved_activations = LowBitSavedActivationConfig {
            mode: LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
            format: LowBitActivationFormat::Int8,
        };
        config
    }

    fn tiny_nca_factorized_model_config() -> BDHConfig {
        BDHConfig {
            n_layer: 2,
            n_embd: 8,
            n_head: 1,
            mlp_internal_dim_multiplier: 1,
            dropout: 0.0,
            vocab_size: 19,
            language_head: LanguageHeadConfig::NcaFactorizedPatch {
                state_count: 2,
                patch_size: 2,
                frame_special_tokens: true,
                eos_id: Some(18),
            },
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

    fn grad_l2_by_param_ids<M>(
        module: &M,
        mut grads: GradientsParams,
        param_ids: &HashSet<ParamId>,
    ) -> f32
    where
        M: AutodiffModule<TestBackend>,
    {
        struct GradNormVisitor<'a> {
            grads: &'a mut GradientsParams,
            param_ids: &'a HashSet<ParamId>,
            sum_sq: f32,
        }

        impl burn::module::ModuleVisitor<TestBackend> for GradNormVisitor<'_> {
            fn visit_float<const D: usize>(&mut self, param: &Param<Tensor<TestBackend, D>>) {
                if !self.param_ids.contains(&param.id) {
                    return;
                }
                if let Some(grad) = self
                    .grads
                    .remove::<<TestBackend as AutodiffBackend>::InnerBackend, D>(param.id)
                {
                    let values = grad
                        .to_data()
                        .convert::<f32>()
                        .into_vec::<f32>()
                        .expect("grad vec");
                    self.sum_sq += values.iter().map(|value| value * value).sum::<f32>();
                }
            }
        }

        let mut visitor = GradNormVisitor {
            grads: &mut grads,
            param_ids,
            sum_sq: 0.0,
        };
        module.visit(&mut visitor);
        visitor.sum_sq.sqrt()
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
    fn nca_factorized_tbptt_valid_step_matches_full_loss_value() {
        let device = <TestValidBackend as BackendTrait>::Device::default();
        let model = BDH::<TestValidBackend>::new(tiny_nca_factorized_model_config(), &device);
        let baseline = LanguageTrainModel::new(model.clone());
        let tbptt = LanguageTrainModel::new(model).with_tbptt_chunk_size(Some(2));
        let batch = make_batch::<TestValidBackend>(
            &device,
            &[16, 0, 1, 17, 16, 2, 3, 18],
            &[0, 1, 17, 18, 2, 3, 16, 18],
            [2, 4],
        );
        let baseline_loss = loss_scalar(ValidStep::step(&baseline, batch.clone()));
        let tbptt_loss = loss_scalar(ValidStep::step(&tbptt, batch));
        assert!(
            (baseline_loss - tbptt_loss).abs() < 1.0e-5,
            "expected factorized tbptt loss to match full loss value, got baseline={baseline_loss} tbptt={tbptt_loss}"
        );
    }

    #[test]
    fn nca_factorized_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            tiny_nca_factorized_model_config(),
            &device,
        ))
        .with_tbptt_chunk_size(Some(2));
        let batch = make_batch::<TestBackend>(
            &device,
            &[16, 0, 1, 17, 16, 2, 3, 18],
            &[0, 1, 17, 18, 2, 3, 16, 18],
            [2, 4],
        );
        let output = TrainStep::step(&model, batch);
        let synced = output.item.sync();
        let loss = loss_scalar(synced);
        assert!(loss.is_finite(), "factorized NCA train loss must be finite");
    }

    #[test]
    fn low_bit_partial_safe_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            tiny_low_bit_model_config(),
            &device,
        ))
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
        assert!(
            loss.is_finite(),
            "low-bit partial-safe train loss must be finite"
        );
    }

    #[test]
    fn low_bit_train_kernel_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            tiny_low_bit_train_kernel_model_config(),
            &device,
        ))
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
        assert!(loss.is_finite(), "low-bit train-kernel loss must be finite");
    }

    #[test]
    fn low_bit_train_kernel_saved_activation_recompute_train_step_runs_and_emits_finite_loss() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let model = LanguageTrainModel::new(BDH::<TestBackend>::new(
            tiny_low_bit_train_kernel_saved_activation_model_config(),
            &device,
        ))
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
        assert!(
            loss.is_finite(),
            "low-bit train-kernel saved-activation recompute loss must be finite"
        );
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

    #[test]
    fn init_transfer_grad_schedule_can_freeze_transferred_backbone_without_zeroing_surfaces() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let mut training = TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 1,
            checkpoint_interval_iters: 1,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::InitFromCheckpoint,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: Some(PathBuf::from("dummy.bin")),
            init_checkpoint_epoch: Some(1),
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            module_lr_scales: Vec::new(),
            context_strategy: Default::default(),
            sequence_kernel_override: None,
            gdpo: None,
        };
        training.init_transfer.backbone_grad_scale = Some(0.0);
        training.init_transfer.backbone_grad_scale_steps = Some(1);
        training.init_transfer.preserve_fresh_decoder = true;
        training.init_transfer.preserve_fresh_norm = true;

        let base_model = BDH::<TestBackend>::new(tiny_model_config(), &device);
        let backbone_ids = base_model
            .transferred_backbone_param_ids(
                training.init_transfer.preserve_fresh_decoder,
                training.init_transfer.preserve_fresh_norm,
            )
            .into_iter()
            .collect::<HashSet<_>>();
        let surface_ids = base_model
            .transfer_interface_surface_param_ids(
                training.init_transfer.preserve_fresh_decoder,
                training.init_transfer.preserve_fresh_norm,
            )
            .into_iter()
            .collect::<HashSet<_>>();

        let baseline = LanguageTrainModel::new(base_model.clone());
        let scheduled =
            LanguageTrainModel::new(base_model.clone()).with_gradient_scale_schedule(&training, 1);
        let scheduled_surface_model =
            LanguageTrainModel::new(base_model).with_gradient_scale_schedule(&training, 1);

        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let baseline_backbone = grad_l2_by_param_ids(
            &baseline,
            TrainStep::step(&baseline, batch.clone()).grads,
            &backbone_ids,
        );
        let scheduled_backbone = grad_l2_by_param_ids(
            &scheduled,
            TrainStep::step(&scheduled, batch.clone()).grads,
            &backbone_ids,
        );
        let baseline_surface = grad_l2_by_param_ids(
            &baseline,
            TrainStep::step(&baseline, batch.clone()).grads,
            &surface_ids,
        );
        let scheduled_surface = grad_l2_by_param_ids(
            &scheduled_surface_model,
            TrainStep::step(&scheduled_surface_model, batch).grads,
            &surface_ids,
        );

        assert!(
            baseline_backbone > 0.0,
            "baseline backbone grads should be nonzero"
        );
        assert!(
            scheduled_backbone <= 1.0e-8,
            "scheduled backbone grads should be zeroed, got {scheduled_backbone}"
        );
        assert!(
            baseline_surface > 0.0,
            "baseline surface grads should be nonzero"
        );
        assert!(
            scheduled_surface > 0.0,
            "scheduled surface grads should stay nonzero"
        );
    }

    #[test]
    fn module_lr_scales_scale_matching_parameter_gradients() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let mut training = TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 1,
            checkpoint_interval_iters: 1,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            module_lr_scales: Vec::new(),
            context_strategy: Default::default(),
            sequence_kernel_override: None,
            gdpo: None,
        };
        training
            .module_lr_scales
            .push(crate::config::train::ModuleLrScaleEntry {
                target: LanguageModuleLrScaleTarget::SharedLowrankDecoder,
                scale: 0.5,
                schedule: None,
            });

        let base_model = BDH::<TestBackend>::new(tiny_model_config(), &device);
        let decoder_ids = base_model
            .language_module_lr_scale_param_ids(LanguageModuleLrScaleTarget::SharedLowrankDecoder)
            .into_iter()
            .collect::<HashSet<_>>();
        let baseline = LanguageTrainModel::new(base_model.clone());
        let scheduled =
            LanguageTrainModel::new(base_model).with_gradient_scale_schedule(&training, 1);

        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let baseline_norm = grad_l2_by_param_ids(
            &baseline,
            TrainStep::step(&baseline, batch.clone()).grads,
            &decoder_ids,
        );
        let scaled_norm = grad_l2_by_param_ids(
            &scheduled,
            TrainStep::step(&scheduled, batch).grads,
            &decoder_ids,
        );

        assert!(
            baseline_norm > 0.0,
            "baseline decoder grads should be nonzero"
        );
        assert!(
            (scaled_norm / baseline_norm - 0.5).abs() < 0.05,
            "expected module lr scale to halve decoder grad norm, got baseline={baseline_norm} scaled={scaled_norm}"
        );
    }

    #[test]
    fn module_lr_scale_schedule_interpolates_across_training_steps() {
        let device = <TestBackend as BackendTrait>::Device::default();
        let training = TrainingHyperparameters {
            block_size: 4,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: 2,
            seed: 1,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: Some(1),
            max_iters: 2,
            checkpoint_interval_iters: 1,
            log_frequency: 1,
            launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode::Fresh,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: None,
            init_checkpoint_epoch: None,
            init_transfer: Default::default(),
            continual_backprop: Default::default(),
            module_lr_scales: vec![
                crate::config::train::ModuleLrScaleEntry {
                    target: LanguageModuleLrScaleTarget::SharedLowrankEncoder,
                    scale: 0.5,
                    schedule: Some(crate::config::train::ModuleLrScaleScheduleConfig {
                        final_scale: 1.0,
                        start_fraction: 0.0,
                        end_fraction: 1.0,
                    }),
                },
                crate::config::train::ModuleLrScaleEntry {
                    target: LanguageModuleLrScaleTarget::SharedLowrankDecoder,
                    scale: 0.5,
                    schedule: Some(crate::config::train::ModuleLrScaleScheduleConfig {
                        final_scale: 1.0,
                        start_fraction: 0.0,
                        end_fraction: 1.0,
                    }),
                },
                crate::config::train::ModuleLrScaleEntry {
                    target: LanguageModuleLrScaleTarget::SharedLowrankDecay,
                    scale: 0.5,
                    schedule: Some(crate::config::train::ModuleLrScaleScheduleConfig {
                        final_scale: 1.0,
                        start_fraction: 0.0,
                        end_fraction: 1.0,
                    }),
                },
            ],
            context_strategy: Default::default(),
            sequence_kernel_override: None,
            gdpo: None,
        };
        let base_model = BDH::<TestBackend>::new(tiny_model_config(), &device);
        let decoder_ids = base_model
            .language_module_lr_scale_param_ids(LanguageModuleLrScaleTarget::SharedLowrankDecoder)
            .into_iter()
            .collect::<HashSet<_>>();
        let scheduled =
            LanguageTrainModel::new(base_model).with_gradient_scale_schedule(&training, 2);
        assert!(
            (scheduled.continual_backprop_target_lr_scale() - 0.5).abs() < 1.0e-6,
            "pre-step target lr scale should start at initial module scale"
        );
        let batch = make_batch::<TestBackend>(
            &device,
            &[0, 1, 2, 3, 7, 6, 5, 4],
            &[1, 2, 3, 4, 6, 5, 4, 3],
            [2, 4],
        );
        let first_norm = grad_l2_by_param_ids(
            &scheduled,
            TrainStep::step(&scheduled, batch.clone()).grads,
            &decoder_ids,
        );
        let second_norm = grad_l2_by_param_ids(
            &scheduled,
            TrainStep::step(&scheduled, batch).grads,
            &decoder_ids,
        );
        assert!(
            (scheduled.continual_backprop_target_lr_scale() - 1.0).abs() < 1.0e-6,
            "post-step target lr scale should advance to the scheduled final scale"
        );

        assert!(first_norm > 0.0, "first-step grad norm should be nonzero");
        assert!(
            second_norm > first_norm * 1.75,
            "scheduled scale should grow toward 1.0 across steps, got first={first_norm} second={second_norm}"
        );
    }
}
