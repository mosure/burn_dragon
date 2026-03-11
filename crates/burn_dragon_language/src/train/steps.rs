use crate::train::prelude::*;
use std::time::Instant;

#[derive(Module, Debug)]
pub(crate) struct LanguageTrainModel<B: BackendTrait> {
    pub(crate) model: BDH<B>,
}

impl<B: BackendTrait> LanguageTrainModel<B> {
    pub(crate) fn new(model: BDH<B>) -> Self {
        Self { model }
    }
}

impl<B: AutodiffBackend> TrainStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelTrainItem<B>;

    fn step(&self, batch: SequenceBatch<B>) -> TrainOutput<LanguageModelTrainItem<B>> {
        let prof_enabled = crate::train::profile::enabled();
        let forward_start = prof_enabled.then(Instant::now);
        let logits = if fast_train_enabled() {
            self.model.forward_fast(batch.inputs)
        } else {
            self.model.forward(batch.inputs)
        };
        let forward_ns = forward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        let loss_backward_start = prof_enabled.then(Instant::now);
        let loss = language_model_loss::<B>(logits, batch.targets);
        let grads = loss.backward();
        let loss_backward_ns = loss_backward_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            crate::train::profile::record_train_step(forward_ns, loss_backward_ns);
        }

        TrainOutput::new(self, grads, LanguageModelTrainItem::new(loss))
    }
}

impl<B: BackendTrait> ValidStep for LanguageTrainModel<B> {
    type Input = SequenceBatch<B>;
    type Output = LanguageModelOutput<B>;

    fn step(&self, batch: SequenceBatch<B>) -> LanguageModelOutput<B> {
        let logits = if fast_train_enabled() {
            self.model.forward_fast(batch.inputs)
        } else {
            self.model.forward(batch.inputs)
        };
        let loss = language_model_loss::<B>(logits, batch.targets);
        LanguageModelOutput::new(loss)
    }
}
