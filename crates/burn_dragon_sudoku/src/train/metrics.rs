use crate::train::prelude::*;
use burn_train::metric::{Adaptor, ItemLazy, LossInput};

#[derive(Clone)]
pub struct SudokuOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    acc: Tensor<B, 1>,
    exact_acc: Tensor<B, 1>,
    solve_rate: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    halt_loss: Tensor<B, 1>,
    halt_prob_mean: Tensor<B, 1>,
    halt_target_mean: Tensor<B, 1>,
    advantage_abs_mean: Tensor<B, 1>,
    advantage_std: Tensor<B, 1>,
    log_prob_mean: Tensor<B, 1>,
    policy_entropy: Tensor<B, 1>,
    hard_reward_mean: Tensor<B, 1>,
    easy_reward_mean: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuOutput<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        acc: Tensor<B, 1>,
        exact_acc: Tensor<B, 1>,
        solve_rate: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        halt_loss: Tensor<B, 1>,
        halt_prob_mean: Tensor<B, 1>,
        halt_target_mean: Tensor<B, 1>,
        advantage_abs_mean: Tensor<B, 1>,
        advantage_std: Tensor<B, 1>,
        log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        hard_reward_mean: Tensor<B, 1>,
        easy_reward_mean: Tensor<B, 1>,
    ) -> Self {
        Self {
            loss,
            recon_loss,
            acc,
            exact_acc,
            solve_rate,
            policy_loss,
            halt_loss,
            halt_prob_mean,
            halt_target_mean,
            advantage_abs_mean,
            advantage_std,
            log_prob_mean,
            policy_entropy,
            hard_reward_mean,
            easy_reward_mean,
        }
    }
}

impl<B: BackendTrait> ItemLazy for SudokuOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<LossValue<B>> for SudokuOutput<B> {
    fn adapt(&self) -> LossValue<B> {
        LossValue::new(self.loss.clone())
    }
}

#[derive(Clone)]
pub struct SudokuReconLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuReconLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuAccInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuAccInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuExactAccInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuExactAccInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuSolveRateInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuSolveRateInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuPolicyLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuPolicyLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuHaltLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuHaltLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuHaltProbInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuHaltProbInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuHaltTargetInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuHaltTargetInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuAdvantageAbsMeanInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuAdvantageAbsMeanInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuAdvantageStdInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuAdvantageStdInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuLogProbMeanInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuLogProbMeanInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuPolicyEntropyInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuPolicyEntropyInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuHardRewardInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuHardRewardInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SudokuEasyRewardInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SudokuEasyRewardInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

impl<B: BackendTrait> Adaptor<SudokuReconLossInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuReconLossInput<B> {
        SudokuReconLossInput::new(self.recon_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuAccInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuAccInput<B> {
        SudokuAccInput::new(self.acc.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuExactAccInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuExactAccInput<B> {
        SudokuExactAccInput::new(self.exact_acc.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuSolveRateInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuSolveRateInput<B> {
        SudokuSolveRateInput::new(self.solve_rate.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuPolicyLossInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuPolicyLossInput<B> {
        SudokuPolicyLossInput::new(self.policy_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuHaltLossInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuHaltLossInput<B> {
        SudokuHaltLossInput::new(self.halt_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuHaltProbInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuHaltProbInput<B> {
        SudokuHaltProbInput::new(self.halt_prob_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuHaltTargetInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuHaltTargetInput<B> {
        SudokuHaltTargetInput::new(self.halt_target_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuAdvantageAbsMeanInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuAdvantageAbsMeanInput<B> {
        SudokuAdvantageAbsMeanInput::new(self.advantage_abs_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuAdvantageStdInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuAdvantageStdInput<B> {
        SudokuAdvantageStdInput::new(self.advantage_std.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuLogProbMeanInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuLogProbMeanInput<B> {
        SudokuLogProbMeanInput::new(self.log_prob_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuPolicyEntropyInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuPolicyEntropyInput<B> {
        SudokuPolicyEntropyInput::new(self.policy_entropy.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuHardRewardInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuHardRewardInput<B> {
        SudokuHardRewardInput::new(self.hard_reward_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<SudokuEasyRewardInput<B>> for SudokuOutput<B> {
    fn adapt(&self) -> SudokuEasyRewardInput<B> {
        SudokuEasyRewardInput::new(self.easy_reward_mean.clone())
    }
}

pub struct SudokuTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    acc: Tensor<B, 1>,
    exact_acc: Tensor<B, 1>,
    solve_rate: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    halt_loss: Tensor<B, 1>,
    halt_prob_mean: Tensor<B, 1>,
    halt_target_mean: Tensor<B, 1>,
    advantage_abs_mean: Tensor<B, 1>,
    advantage_std: Tensor<B, 1>,
    log_prob_mean: Tensor<B, 1>,
    policy_entropy: Tensor<B, 1>,
    hard_reward_mean: Tensor<B, 1>,
    easy_reward_mean: Tensor<B, 1>,
}

impl<B: AutodiffBackend> SudokuTrainItem<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        acc: Tensor<B, 1>,
        exact_acc: Tensor<B, 1>,
        solve_rate: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        halt_loss: Tensor<B, 1>,
        halt_prob_mean: Tensor<B, 1>,
        halt_target_mean: Tensor<B, 1>,
        advantage_abs_mean: Tensor<B, 1>,
        advantage_std: Tensor<B, 1>,
        log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        hard_reward_mean: Tensor<B, 1>,
        easy_reward_mean: Tensor<B, 1>,
    ) -> Self {
        Self {
            loss: loss.detach(),
            recon_loss: recon_loss.detach(),
            acc: acc.detach(),
            exact_acc: exact_acc.detach(),
            solve_rate: solve_rate.detach(),
            policy_loss: policy_loss.detach(),
            halt_loss: halt_loss.detach(),
            halt_prob_mean: halt_prob_mean.detach(),
            halt_target_mean: halt_target_mean.detach(),
            advantage_abs_mean: advantage_abs_mean.detach(),
            advantage_std: advantage_std.detach(),
            log_prob_mean: log_prob_mean.detach(),
            policy_entropy: policy_entropy.detach(),
            hard_reward_mean: hard_reward_mean.detach(),
            easy_reward_mean: easy_reward_mean.detach(),
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for SudokuTrainItem<B> {
    type ItemSync = SudokuOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        SudokuOutput::new(
            self.loss.detach().inner(),
            self.recon_loss.detach().inner(),
            self.acc.detach().inner(),
            self.exact_acc.detach().inner(),
            self.solve_rate.detach().inner(),
            self.policy_loss.detach().inner(),
            self.halt_loss.detach().inner(),
            self.halt_prob_mean.detach().inner(),
            self.halt_target_mean.detach().inner(),
            self.advantage_abs_mean.detach().inner(),
            self.advantage_std.detach().inner(),
            self.log_prob_mean.detach().inner(),
            self.policy_entropy.detach().inner(),
            self.hard_reward_mean.detach().inner(),
            self.easy_reward_mean.detach().inner(),
        )
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuReconLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuAccInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuExactAccInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuSolveRateInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuPolicyLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuHaltLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuHaltProbInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuHaltTargetInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuAdvantageAbsMeanInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuAdvantageStdInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuLogProbMeanInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuPolicyEntropyInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuHardRewardInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SudokuEasyRewardInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

#[cfg(feature = "integration_test")]
mod loss_trace {
    use super::*;
    use burn_train::metric::{Metric, MetricEntry, MetricMetadata, format_float};
    use std::sync::{Mutex, OnceLock};

    fn storage() -> &'static Mutex<Vec<f32>> {
        static TRACE: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();
        TRACE.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn reset() {
        if let Ok(mut trace) = storage().lock() {
            trace.clear();
        }
    }

    pub fn take() -> Vec<f32> {
        if let Ok(mut trace) = storage().lock() {
            let mut out = Vec::new();
            std::mem::swap(&mut *trace, &mut out);
            out
        } else {
            Vec::new()
        }
    }

    pub fn len() -> usize {
        if let Ok(trace) = storage().lock() {
            trace.len()
        } else {
            0
        }
    }

    #[derive(Clone)]
    pub struct LossTraceMetric<B: BackendTrait> {
        name: Arc<String>,
        every: usize,
        last: f64,
        initialized: bool,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> LossTraceMetric<B> {
        pub fn new(name: &str, every: usize) -> Self {
            Self {
                name: Arc::new(name.to_string()),
                every: every.max(1),
                last: 0.0,
                initialized: false,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<B: BackendTrait> Metric for LossTraceMetric<B> {
        type Input = LossValue<B>;

        fn name(&self) -> burn_train::metric::MetricName {
            Arc::clone(&self.name)
        }

        fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
            if self.every > 1
                && !metadata.iteration.is_multiple_of(self.every)
                && self.initialized
            {
                return MetricEntry::new(
                    Arc::clone(&self.name),
                    format_float(self.last, 4),
                    self.last.to_string(),
                );
            }
            let value = item
                .value()
                .mean()
                .into_data()
                .iter::<f64>()
                .next()
                .unwrap_or(0.0);
            self.last = value;
            self.initialized = true;
            if let Ok(mut trace) = storage().lock() {
                trace.push(value as f32);
            }
            MetricEntry::new(
                Arc::clone(&self.name),
                format_float(value, 4),
                value.to_string(),
            )
        }

        fn clear(&mut self) {
            self.last = 0.0;
            self.initialized = false;
        }
    }

}

#[cfg(feature = "integration_test")]
pub use loss_trace::{len as loss_trace_len, reset as loss_trace_reset, take as loss_trace_take};
#[cfg(feature = "integration_test")]
pub use loss_trace::LossTraceMetric;

#[cfg(feature = "integration_test")]
mod solve_rate_trace {
    use super::*;
    use burn_train::metric::{Metric, MetricEntry, MetricMetadata, format_float};
    use std::sync::{Mutex, OnceLock};

    fn storage() -> &'static Mutex<Vec<f32>> {
        static TRACE: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();
        TRACE.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn reset() {
        if let Ok(mut trace) = storage().lock() {
            trace.clear();
        }
    }

    pub fn take() -> Vec<f32> {
        if let Ok(mut trace) = storage().lock() {
            let mut out = Vec::new();
            std::mem::swap(&mut *trace, &mut out);
            out
        } else {
            Vec::new()
        }
    }

    pub fn len() -> usize {
        if let Ok(trace) = storage().lock() {
            trace.len()
        } else {
            0
        }
    }

    #[derive(Clone)]
    pub struct SolveRateTraceMetric<B: BackendTrait> {
        name: Arc<String>,
        every: usize,
        last: f64,
        initialized: bool,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> SolveRateTraceMetric<B> {
        pub fn new(name: &str, every: usize) -> Self {
            Self {
                name: Arc::new(name.to_string()),
                every: every.max(1),
                last: 0.0,
                initialized: false,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<B: BackendTrait> Metric for SolveRateTraceMetric<B> {
        type Input = SudokuSolveRateInput<B>;

        fn name(&self) -> burn_train::metric::MetricName {
            Arc::clone(&self.name)
        }

        fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
            if self.every > 1
                && !metadata.iteration.is_multiple_of(self.every)
                && self.initialized
            {
                return MetricEntry::new(
                    Arc::clone(&self.name),
                    format_float(self.last, 4),
                    self.last.to_string(),
                );
            }
            let value = item
                .value()
                .mean()
                .into_data()
                .iter::<f64>()
                .next()
                .unwrap_or(0.0);
            self.last = value;
            self.initialized = true;
            if let Ok(mut trace) = storage().lock() {
                trace.push(value as f32);
            }
            MetricEntry::new(
                Arc::clone(&self.name),
                format_float(value, 4),
                value.to_string(),
            )
        }

        fn clear(&mut self) {
            self.last = 0.0;
            self.initialized = false;
        }
    }
}

#[cfg(feature = "integration_test")]
pub use solve_rate_trace::{
    len as solve_rate_trace_len, reset as solve_rate_trace_reset, take as solve_rate_trace_take,
};
#[cfg(feature = "integration_test")]
pub use solve_rate_trace::SolveRateTraceMetric;

#[cfg(feature = "integration_test")]
mod halt_prob_trace {
    use super::*;
    use burn_train::metric::{Metric, MetricEntry, MetricMetadata, format_float};
    use std::sync::{Mutex, OnceLock};

    fn storage() -> &'static Mutex<Vec<f32>> {
        static TRACE: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();
        TRACE.get_or_init(|| Mutex::new(Vec::new()))
    }

    pub fn reset() {
        if let Ok(mut trace) = storage().lock() {
            trace.clear();
        }
    }

    pub fn take() -> Vec<f32> {
        if let Ok(mut trace) = storage().lock() {
            let mut out = Vec::new();
            std::mem::swap(&mut *trace, &mut out);
            out
        } else {
            Vec::new()
        }
    }

    pub fn len() -> usize {
        if let Ok(trace) = storage().lock() {
            trace.len()
        } else {
            0
        }
    }

    #[derive(Clone)]
    pub struct HaltProbTraceMetric<B: BackendTrait> {
        name: Arc<String>,
        every: usize,
        last: f64,
        initialized: bool,
        _marker: std::marker::PhantomData<B>,
    }

    impl<B: BackendTrait> HaltProbTraceMetric<B> {
        pub fn new(name: &str, every: usize) -> Self {
            Self {
                name: Arc::new(name.to_string()),
                every: every.max(1),
                last: 0.0,
                initialized: false,
                _marker: std::marker::PhantomData,
            }
        }
    }

    impl<B: BackendTrait> Metric for HaltProbTraceMetric<B> {
        type Input = SudokuHaltProbInput<B>;

        fn name(&self) -> burn_train::metric::MetricName {
            Arc::clone(&self.name)
        }

        fn update(&mut self, item: &Self::Input, metadata: &MetricMetadata) -> MetricEntry {
            if self.every > 1
                && !metadata.iteration.is_multiple_of(self.every)
                && self.initialized
            {
                return MetricEntry::new(
                    Arc::clone(&self.name),
                    format_float(self.last, 4),
                    self.last.to_string(),
                );
            }
            let value = item
                .value()
                .mean()
                .into_data()
                .iter::<f64>()
                .next()
                .unwrap_or(0.0);
            self.last = value;
            self.initialized = true;
            if let Ok(mut trace) = storage().lock() {
                trace.push(value as f32);
            }
            MetricEntry::new(
                Arc::clone(&self.name),
                format_float(value, 4),
                value.to_string(),
            )
        }

        fn clear(&mut self) {
            self.last = 0.0;
            self.initialized = false;
        }
    }
}

#[cfg(feature = "integration_test")]
pub use halt_prob_trace::{
    len as halt_prob_trace_len, reset as halt_prob_trace_reset, take as halt_prob_trace_take,
};
#[cfg(feature = "integration_test")]
pub use halt_prob_trace::HaltProbTraceMetric;
