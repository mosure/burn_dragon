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
    policy_entropy_alpha: Tensor<B, 1>,
    policy_entropy_target: Tensor<B, 1>,
    hard_reward_mean: Tensor<B, 1>,
    easy_reward_mean: Tensor<B, 1>,
    shaping_conflict_mean: Tensor<B, 1>,
    shaping_unknown_mean: Tensor<B, 1>,
    shaping_accuracy_mean: Tensor<B, 1>,
    shaping_incorrect_mean: Tensor<B, 1>,
    saccade_revisit_rate: Tensor<B, 1>,
    saccade_repeat_rate: Tensor<B, 1>,
    saccade_unknown_frac: Tensor<B, 1>,
    saccade_unique_frac: Tensor<B, 1>,
    write_gate_mean: Tensor<B, 1>,
    write_rate: Tensor<B, 1>,
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
        policy_entropy_alpha: Tensor<B, 1>,
        policy_entropy_target: Tensor<B, 1>,
        hard_reward_mean: Tensor<B, 1>,
        easy_reward_mean: Tensor<B, 1>,
        shaping_conflict_mean: Tensor<B, 1>,
        shaping_unknown_mean: Tensor<B, 1>,
        shaping_accuracy_mean: Tensor<B, 1>,
        shaping_incorrect_mean: Tensor<B, 1>,
        saccade_revisit_rate: Tensor<B, 1>,
        saccade_repeat_rate: Tensor<B, 1>,
        saccade_unknown_frac: Tensor<B, 1>,
        saccade_unique_frac: Tensor<B, 1>,
        write_gate_mean: Tensor<B, 1>,
        write_rate: Tensor<B, 1>,
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
            policy_entropy_alpha,
            policy_entropy_target,
            hard_reward_mean,
            easy_reward_mean,
            shaping_conflict_mean,
            shaping_unknown_mean,
            shaping_accuracy_mean,
            shaping_incorrect_mean,
            saccade_revisit_rate,
            saccade_repeat_rate,
            saccade_unknown_frac,
            saccade_unique_frac,
            write_gate_mean,
            write_rate,
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

macro_rules! define_sudoku_scalar_input {
    ($name:ident, $field:ident) => {
        #[derive(Clone)]
        pub struct $name<B: BackendTrait> {
            value: Tensor<B, 1>,
        }

        impl<B: BackendTrait> $name<B> {
            pub fn new(value: Tensor<B, 1>) -> Self {
                Self { value }
            }
        }

        impl<B: BackendTrait> Adaptor<$name<B>> for SudokuOutput<B> {
            fn adapt(&self) -> $name<B> {
                $name::new(self.$field.clone())
            }
        }

        impl<B: BackendTrait> ScalarValue<B> for $name<B> {
            fn value(&self) -> Tensor<B, 1> {
                self.value.clone()
            }
        }
    };
}

define_sudoku_scalar_input!(SudokuReconLossInput, recon_loss);
define_sudoku_scalar_input!(SudokuAccInput, acc);
define_sudoku_scalar_input!(SudokuExactAccInput, exact_acc);
define_sudoku_scalar_input!(SudokuSolveRateInput, solve_rate);
define_sudoku_scalar_input!(SudokuPolicyLossInput, policy_loss);
define_sudoku_scalar_input!(SudokuHaltLossInput, halt_loss);
define_sudoku_scalar_input!(SudokuHaltProbInput, halt_prob_mean);
define_sudoku_scalar_input!(SudokuHaltTargetInput, halt_target_mean);
define_sudoku_scalar_input!(SudokuAdvantageAbsMeanInput, advantage_abs_mean);
define_sudoku_scalar_input!(SudokuAdvantageStdInput, advantage_std);
define_sudoku_scalar_input!(SudokuLogProbMeanInput, log_prob_mean);
define_sudoku_scalar_input!(SudokuPolicyEntropyInput, policy_entropy);
define_sudoku_scalar_input!(SudokuPolicyEntropyAlphaInput, policy_entropy_alpha);
define_sudoku_scalar_input!(SudokuPolicyEntropyTargetInput, policy_entropy_target);
define_sudoku_scalar_input!(SudokuHardRewardInput, hard_reward_mean);
define_sudoku_scalar_input!(SudokuEasyRewardInput, easy_reward_mean);
define_sudoku_scalar_input!(SudokuShapingConflictInput, shaping_conflict_mean);
define_sudoku_scalar_input!(SudokuShapingUnknownInput, shaping_unknown_mean);
define_sudoku_scalar_input!(SudokuShapingAccuracyInput, shaping_accuracy_mean);
define_sudoku_scalar_input!(SudokuShapingIncorrectInput, shaping_incorrect_mean);
define_sudoku_scalar_input!(SudokuSaccadeRevisitRateInput, saccade_revisit_rate);
define_sudoku_scalar_input!(SudokuSaccadeRepeatRateInput, saccade_repeat_rate);
define_sudoku_scalar_input!(SudokuSaccadeUnknownFracInput, saccade_unknown_frac);
define_sudoku_scalar_input!(SudokuSaccadeUniqueFracInput, saccade_unique_frac);
define_sudoku_scalar_input!(SudokuWriteGateMeanInput, write_gate_mean);
define_sudoku_scalar_input!(SudokuWriteRateInput, write_rate);

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
    policy_entropy_alpha: Tensor<B, 1>,
    policy_entropy_target: Tensor<B, 1>,
    hard_reward_mean: Tensor<B, 1>,
    easy_reward_mean: Tensor<B, 1>,
    shaping_conflict_mean: Tensor<B, 1>,
    shaping_unknown_mean: Tensor<B, 1>,
    shaping_accuracy_mean: Tensor<B, 1>,
    shaping_incorrect_mean: Tensor<B, 1>,
    saccade_revisit_rate: Tensor<B, 1>,
    saccade_repeat_rate: Tensor<B, 1>,
    saccade_unknown_frac: Tensor<B, 1>,
    saccade_unique_frac: Tensor<B, 1>,
    write_gate_mean: Tensor<B, 1>,
    write_rate: Tensor<B, 1>,
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
        policy_entropy_alpha: Tensor<B, 1>,
        policy_entropy_target: Tensor<B, 1>,
        hard_reward_mean: Tensor<B, 1>,
        easy_reward_mean: Tensor<B, 1>,
        shaping_conflict_mean: Tensor<B, 1>,
        shaping_unknown_mean: Tensor<B, 1>,
        shaping_accuracy_mean: Tensor<B, 1>,
        shaping_incorrect_mean: Tensor<B, 1>,
        saccade_revisit_rate: Tensor<B, 1>,
        saccade_repeat_rate: Tensor<B, 1>,
        saccade_unknown_frac: Tensor<B, 1>,
        saccade_unique_frac: Tensor<B, 1>,
        write_gate_mean: Tensor<B, 1>,
        write_rate: Tensor<B, 1>,
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
            policy_entropy_alpha: policy_entropy_alpha.detach(),
            policy_entropy_target: policy_entropy_target.detach(),
            hard_reward_mean: hard_reward_mean.detach(),
            easy_reward_mean: easy_reward_mean.detach(),
            shaping_conflict_mean: shaping_conflict_mean.detach(),
            shaping_unknown_mean: shaping_unknown_mean.detach(),
            shaping_accuracy_mean: shaping_accuracy_mean.detach(),
            shaping_incorrect_mean: shaping_incorrect_mean.detach(),
            saccade_revisit_rate: saccade_revisit_rate.detach(),
            saccade_repeat_rate: saccade_repeat_rate.detach(),
            saccade_unknown_frac: saccade_unknown_frac.detach(),
            saccade_unique_frac: saccade_unique_frac.detach(),
            write_gate_mean: write_gate_mean.detach(),
            write_rate: write_rate.detach(),
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
            self.policy_entropy_alpha.detach().inner(),
            self.policy_entropy_target.detach().inner(),
            self.hard_reward_mean.detach().inner(),
            self.easy_reward_mean.detach().inner(),
            self.shaping_conflict_mean.detach().inner(),
            self.shaping_unknown_mean.detach().inner(),
            self.shaping_accuracy_mean.detach().inner(),
            self.shaping_incorrect_mean.detach().inner(),
            self.saccade_revisit_rate.detach().inner(),
            self.saccade_repeat_rate.detach().inner(),
            self.saccade_unknown_frac.detach().inner(),
            self.saccade_unique_frac.detach().inner(),
            self.write_gate_mean.detach().inner(),
            self.write_rate.detach().inner(),
        )
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
            if self.every > 1 && !metadata.iteration.is_multiple_of(self.every) && self.initialized
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
pub use loss_trace::LossTraceMetric;
#[cfg(feature = "integration_test")]
pub use loss_trace::{len as loss_trace_len, reset as loss_trace_reset, take as loss_trace_take};

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
            if self.every > 1 && !metadata.iteration.is_multiple_of(self.every) && self.initialized
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
pub use solve_rate_trace::SolveRateTraceMetric;
#[cfg(feature = "integration_test")]
pub use solve_rate_trace::{
    len as solve_rate_trace_len, reset as solve_rate_trace_reset, take as solve_rate_trace_take,
};

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
            if self.every > 1 && !metadata.iteration.is_multiple_of(self.every) && self.initialized
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
pub use halt_prob_trace::HaltProbTraceMetric;
#[cfg(feature = "integration_test")]
pub use halt_prob_trace::{
    len as halt_prob_trace_len, reset as halt_prob_trace_reset, take as halt_prob_trace_take,
};
