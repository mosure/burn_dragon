#![cfg_attr(not(feature = "cli"), allow(dead_code))]

use std::any::Any;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
#[cfg(feature = "integration_test")]
use std::sync::Mutex;
#[cfg(feature = "integration_test")]
use std::sync::OnceLock;

use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::{Int, Tensor};
use burn_train::metric::{Adaptor, ItemLazy, LossInput};

#[cfg(any(feature = "train", feature = "cli"))]
use cubecl::Runtime;

use crate::VisionArtifactOutputMode;
use crate::train::artifacts::{ArtifactFrame, collect_frames, write_video};
use crate::train::constants::LEJEPA_EPS;

pub struct LanguageModelOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
}

impl<B: BackendTrait> LanguageModelOutput<B> {
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self { loss }
    }
}

impl<B: BackendTrait> ItemLazy for LanguageModelOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<LossValue<B>> for LanguageModelOutput<B> {
    fn adapt(&self) -> LossValue<B> {
        LossValue::new(self.loss.clone())
    }
}

#[derive(Clone)]
pub struct LossValue<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> LossValue<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

pub struct LanguageModelTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
}

impl<B: AutodiffBackend> LanguageModelTrainItem<B> {
    pub fn new(loss: Tensor<B, 1>) -> Self {
        Self {
            loss: loss.detach(),
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for LanguageModelTrainItem<B> {
    type ItemSync = LanguageModelOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        LanguageModelOutput::new(self.loss.detach().inner())
    }
}

#[derive(Clone)]
pub struct VisionArtifactInput<B: BackendTrait> {
    pub views: Option<Tensor<B, 5>>,
    pub frames: Option<Tensor<B, 5>>,
    pub patch_norms: Option<Tensor<B, 3>>,
    pub pca_rgb: Option<Tensor<B, 4>>,
    pub patch_norms_steps: Option<Tensor<B, 4>>,
    pub pca_rgb_steps: Option<Tensor<B, 5>>,
    pub probe_logits: Option<Tensor<B, 2>>,
    pub labels: Option<Tensor<B, 1, Int>>,
    pub legend: Option<Vec<String>>,
}

impl<B: BackendTrait> VisionArtifactInput<B> {
    pub fn empty() -> Self {
        Self {
            views: None,
            frames: None,
            patch_norms: None,
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: None,
        }
    }
}

#[derive(Clone)]
pub struct VisionOutput<B: BackendTrait> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    recon_psnr_masked: Tensor<B, 1>,
    recon_psnr_full: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    policy_advantage_abs_mean: Tensor<B, 1>,
    policy_advantage_std: Tensor<B, 1>,
    policy_log_prob_mean: Tensor<B, 1>,
    policy_entropy: Tensor<B, 1>,
    policy_action_clamp_rate: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
    artifacts: Option<VisionArtifactInput<B>>,
}

impl<B: BackendTrait> VisionOutput<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        recon_psnr_masked: Tensor<B, 1>,
        recon_psnr_full: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        policy_advantage_abs_mean: Tensor<B, 1>,
        policy_advantage_std: Tensor<B, 1>,
        policy_log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        policy_action_clamp_rate: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
        artifacts: Option<VisionArtifactInput<B>>,
    ) -> Self {
        Self {
            loss,
            inv_loss,
            sigreg_loss,
            recon_loss,
            recon_psnr_masked,
            recon_psnr_full,
            policy_loss,
            policy_advantage_abs_mean,
            policy_advantage_std,
            policy_log_prob_mean,
            policy_entropy,
            policy_action_clamp_rate,
            probe_loss,
            probe_acc,
            artifacts,
        }
    }
}

impl<B: BackendTrait> ItemLazy for VisionOutput<B> {
    type ItemSync = Self;

    fn sync(self) -> Self::ItemSync {
        self
    }
}

impl<B: BackendTrait> Adaptor<LossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossInput<B> {
        LossInput::new(self.loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<LossValue<B>> for VisionOutput<B> {
    fn adapt(&self) -> LossValue<B> {
        LossValue::new(self.loss.clone())
    }
}

#[derive(Clone)]
pub struct InvLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> InvLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct SigRegLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> SigRegLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ReconLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ReconLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ReconPsnrMaskedInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ReconPsnrMaskedInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ReconPsnrFullInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ReconPsnrFullInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct PolicyLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> PolicyLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct AdvantageAbsMeanInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> AdvantageAbsMeanInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct AdvantageStdInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> AdvantageStdInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct LogProbMeanInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> LogProbMeanInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct PolicyEntropyInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> PolicyEntropyInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ActionClampRateInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ActionClampRateInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ProbeLossInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeLossInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

#[derive(Clone)]
pub struct ProbeAccInput<B: BackendTrait> {
    value: Tensor<B, 1>,
}

impl<B: BackendTrait> ProbeAccInput<B> {
    pub fn new(value: Tensor<B, 1>) -> Self {
        Self { value }
    }
}

impl<B: BackendTrait> Adaptor<InvLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> InvLossInput<B> {
        InvLossInput::new(self.inv_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<SigRegLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> SigRegLossInput<B> {
        SigRegLossInput::new(self.sigreg_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconLossInput<B> {
        ReconLossInput::new(self.recon_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconPsnrMaskedInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconPsnrMaskedInput<B> {
        ReconPsnrMaskedInput::new(self.recon_psnr_masked.clone())
    }
}

impl<B: BackendTrait> Adaptor<ReconPsnrFullInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ReconPsnrFullInput<B> {
        ReconPsnrFullInput::new(self.recon_psnr_full.clone())
    }
}

impl<B: BackendTrait> Adaptor<PolicyLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> PolicyLossInput<B> {
        PolicyLossInput::new(self.policy_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<AdvantageAbsMeanInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> AdvantageAbsMeanInput<B> {
        AdvantageAbsMeanInput::new(self.policy_advantage_abs_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<AdvantageStdInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> AdvantageStdInput<B> {
        AdvantageStdInput::new(self.policy_advantage_std.clone())
    }
}

impl<B: BackendTrait> Adaptor<LogProbMeanInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> LogProbMeanInput<B> {
        LogProbMeanInput::new(self.policy_log_prob_mean.clone())
    }
}

impl<B: BackendTrait> Adaptor<PolicyEntropyInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> PolicyEntropyInput<B> {
        PolicyEntropyInput::new(self.policy_entropy.clone())
    }
}

impl<B: BackendTrait> Adaptor<ActionClampRateInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ActionClampRateInput<B> {
        ActionClampRateInput::new(self.policy_action_clamp_rate.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeLossInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeLossInput<B> {
        ProbeLossInput::new(self.probe_loss.clone())
    }
}

impl<B: BackendTrait> Adaptor<ProbeAccInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> ProbeAccInput<B> {
        ProbeAccInput::new(self.probe_acc.clone())
    }
}

impl<B: BackendTrait> Adaptor<VisionArtifactInput<B>> for VisionOutput<B> {
    fn adapt(&self) -> VisionArtifactInput<B> {
        self.artifacts
            .clone()
            .unwrap_or_else(VisionArtifactInput::empty)
    }
}

pub struct VisionTrainItem<B: AutodiffBackend> {
    loss: Tensor<B, 1>,
    inv_loss: Tensor<B, 1>,
    sigreg_loss: Tensor<B, 1>,
    recon_loss: Tensor<B, 1>,
    recon_psnr_masked: Tensor<B, 1>,
    recon_psnr_full: Tensor<B, 1>,
    policy_loss: Tensor<B, 1>,
    policy_advantage_abs_mean: Tensor<B, 1>,
    policy_advantage_std: Tensor<B, 1>,
    policy_log_prob_mean: Tensor<B, 1>,
    policy_entropy: Tensor<B, 1>,
    policy_action_clamp_rate: Tensor<B, 1>,
    probe_loss: Tensor<B, 1>,
    probe_acc: Tensor<B, 1>,
}

impl<B: AutodiffBackend> VisionTrainItem<B> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        loss: Tensor<B, 1>,
        inv_loss: Tensor<B, 1>,
        sigreg_loss: Tensor<B, 1>,
        recon_loss: Tensor<B, 1>,
        recon_psnr_masked: Tensor<B, 1>,
        recon_psnr_full: Tensor<B, 1>,
        policy_loss: Tensor<B, 1>,
        policy_advantage_abs_mean: Tensor<B, 1>,
        policy_advantage_std: Tensor<B, 1>,
        policy_log_prob_mean: Tensor<B, 1>,
        policy_entropy: Tensor<B, 1>,
        policy_action_clamp_rate: Tensor<B, 1>,
        probe_loss: Tensor<B, 1>,
        probe_acc: Tensor<B, 1>,
    ) -> Self {
        Self {
            loss: loss.detach(),
            inv_loss: inv_loss.detach(),
            sigreg_loss: sigreg_loss.detach(),
            recon_loss: recon_loss.detach(),
            recon_psnr_masked: recon_psnr_masked.detach(),
            recon_psnr_full: recon_psnr_full.detach(),
            policy_loss: policy_loss.detach(),
            policy_advantage_abs_mean: policy_advantage_abs_mean.detach(),
            policy_advantage_std: policy_advantage_std.detach(),
            policy_log_prob_mean: policy_log_prob_mean.detach(),
            policy_entropy: policy_entropy.detach(),
            policy_action_clamp_rate: policy_action_clamp_rate.detach(),
            probe_loss: probe_loss.detach(),
            probe_acc: probe_acc.detach(),
        }
    }
}

impl<B: AutodiffBackend> ItemLazy for VisionTrainItem<B> {
    type ItemSync = VisionOutput<B::InnerBackend>;

    fn sync(self) -> Self::ItemSync {
        VisionOutput::new(
            self.loss.detach().inner(),
            self.inv_loss.detach().inner(),
            self.sigreg_loss.detach().inner(),
            self.recon_loss.detach().inner(),
            self.recon_psnr_masked.detach().inner(),
            self.recon_psnr_full.detach().inner(),
            self.policy_loss.detach().inner(),
            self.policy_advantage_abs_mean.detach().inner(),
            self.policy_advantage_std.detach().inner(),
            self.policy_log_prob_mean.detach().inner(),
            self.policy_entropy.detach().inner(),
            self.policy_action_clamp_rate.detach().inner(),
            self.probe_loss.detach().inner(),
            self.probe_acc.detach().inner(),
            None,
        )
    }
}

pub trait ScalarValue<B: BackendTrait> {
    fn value(&self) -> Tensor<B, 1>;
}

impl<B: BackendTrait> ScalarValue<B> for InvLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for SigRegLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ReconLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ReconPsnrMaskedInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ReconPsnrFullInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for PolicyLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for AdvantageAbsMeanInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for AdvantageStdInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for LogProbMeanInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for PolicyEntropyInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ActionClampRateInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeLossInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for ProbeAccInput<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

impl<B: BackendTrait> ScalarValue<B> for LossValue<B> {
    fn value(&self) -> Tensor<B, 1> {
        self.value.clone()
    }
}

pub struct ScalarMetric<B: BackendTrait, I: ScalarValue<B>> {
    name: Arc<String>,
    last: f64,
    every: usize,
    initialized: bool,
    _marker: std::marker::PhantomData<(B, I)>,
}

impl<B: BackendTrait, I: ScalarValue<B>> Clone for ScalarMetric<B, I> {
    fn clone(&self) -> Self {
        Self {
            name: Arc::clone(&self.name),
            last: self.last,
            every: self.every,
            initialized: self.initialized,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B>> ScalarMetric<B, I> {
    pub fn new_every(name: &str, every: usize) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            last: 0.0,
            every: every.max(1),
            initialized: false,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Metric
    for ScalarMetric<B, I>
{
    type Input = I;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        if self.every > 1 && !metadata.iteration.is_multiple_of(self.every) && self.initialized {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                burn_train::metric::format_float(self.last, 4),
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
        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            burn_train::metric::format_float(value, 4),
            value.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last = 0.0;
        self.initialized = false;
    }
}

impl<B: BackendTrait, I: ScalarValue<B> + Send + Sync> burn_train::metric::Numeric
    for ScalarMetric<B, I>
{
    fn value(&self) -> burn_train::metric::NumericEntry {
        burn_train::metric::NumericEntry::Value(self.last)
    }
}

#[cfg(feature = "integration_test")]
fn loss_trace_storage() -> &'static Mutex<Vec<f32>> {
    static TRACE: OnceLock<Mutex<Vec<f32>>> = OnceLock::new();
    TRACE.get_or_init(|| Mutex::new(Vec::new()))
}

#[cfg(feature = "integration_test")]
pub fn loss_trace_reset() {
    if let Ok(mut trace) = loss_trace_storage().lock() {
        trace.clear();
    }
}

#[cfg(feature = "integration_test")]
pub fn loss_trace_take() -> Vec<f32> {
    if let Ok(mut trace) = loss_trace_storage().lock() {
        let mut out = Vec::new();
        std::mem::swap(&mut *trace, &mut out);
        out
    } else {
        Vec::new()
    }
}

#[cfg(feature = "integration_test")]
pub fn loss_trace_len() -> usize {
    if let Ok(trace) = loss_trace_storage().lock() {
        trace.len()
    } else {
        0
    }
}

#[cfg(feature = "integration_test")]
#[derive(Clone)]
pub struct LossTraceMetric<B: BackendTrait> {
    name: Arc<String>,
    every: usize,
    last: f64,
    initialized: bool,
    _marker: std::marker::PhantomData<B>,
}

#[cfg(feature = "integration_test")]
impl<B: BackendTrait> LossTraceMetric<B> {
    pub fn new(name: &str, every: usize) -> Self {
        let every = every.max(1);
        Self {
            name: Arc::new(name.to_string()),
            every,
            last: 0.0,
            initialized: false,
            _marker: std::marker::PhantomData,
        }
    }
}

#[cfg(feature = "integration_test")]
impl<B: BackendTrait> burn_train::metric::Metric for LossTraceMetric<B> {
    type Input = LossValue<B>;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        if self.every > 1 && !metadata.iteration.is_multiple_of(self.every) && self.initialized {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                burn_train::metric::format_float(self.last, 4),
                self.last.to_string(),
            );
        }
        let value = item
            .value()
            .mean()
            .into_data()
            .iter::<f64>()
            .next()
            .unwrap_or(0.0) as f32;
        self.last = value as f64;
        self.initialized = true;
        if let Ok(mut trace) = loss_trace_storage().lock() {
            trace.push(value);
        }
        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            burn_train::metric::format_float(value as f64, 4),
            value.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last = 0.0;
        self.initialized = false;
    }
}

#[derive(Clone)]
pub struct DeviceMetric {
    name: Arc<String>,
    value: Arc<String>,
}

impl DeviceMetric {
    pub fn new(name: &str, value: &str) -> Self {
        Self {
            name: Arc::new(name.to_string()),
            value: Arc::new(value.to_string()),
        }
    }
}

impl burn_train::metric::Metric for DeviceMetric {
    type Input = ();

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        _item: &Self::Input,
        _metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            self.value.to_string(),
            self.value.to_string(),
        )
    }

    fn clear(&mut self) {}
}

#[cfg(any(feature = "train", feature = "cli"))]
fn extra_memory_cleanup<B: BackendTrait>(device: &B::Device)
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    {
        if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<burn_cuda::CudaDevice>() {
            <cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_cleanup();
        }
    }
    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<burn_wgpu::WgpuDevice>() {
        <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_cleanup();
    }
}

fn allow_memory_cleanup<B: BackendTrait>(_device: &B::Device, _allow_cuda_cleanup: bool) -> bool
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if (_device as &dyn Any)
        .downcast_ref::<burn_cuda::CudaDevice>()
        .is_some()
    {
        return _allow_cuda_cleanup;
    }
    true
}

#[derive(Clone, Copy, Debug)]
struct DeviceMemoryUsage {
    reserved_bytes: u64,
    in_use_bytes: u64,
}

impl DeviceMemoryUsage {
    fn reserved_mb(self) -> f64 {
        bytes_to_mb(self.reserved_bytes)
    }

    fn in_use_mb(self) -> f64 {
        bytes_to_mb(self.in_use_bytes)
    }
}

fn bytes_to_mb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

#[cfg(any(feature = "train", feature = "cli"))]
fn device_memory_usage<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<burn_cuda::CudaDevice>() {
        let usage = <cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }
    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<burn_wgpu::WgpuDevice>() {
        let usage = <burn_wgpu::WgpuRuntime as Runtime>::client(wgpu_device).memory_usage();
        return Some(DeviceMemoryUsage {
            reserved_bytes: usage.bytes_reserved,
            in_use_bytes: usage.bytes_in_use,
        });
    }
    None
}

fn device_memory_usage_safe<B: BackendTrait>(device: &B::Device) -> Option<DeviceMemoryUsage>
where
    B::Device: 'static,
{
    #[cfg(any(feature = "train", feature = "cli"))]
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            device_memory_usage::<B>(device)
        }))
        .ok()
        .flatten()
    }
    #[cfg(not(any(feature = "train", feature = "cli")))]
    {
        let _ = device;
        None
    }
}

#[derive(Clone)]
pub struct MemoryCleanupMetric<B: BackendTrait> {
    name: Arc<String>,
    device: B::Device,
    every_epochs: usize,
    every_iters: usize,
    last_epoch: Option<usize>,
    allow_cuda_cleanup: bool,
}

impl<B: BackendTrait> MemoryCleanupMetric<B> {
    pub fn new(
        device: &B::Device,
        every_epochs: usize,
        every_iters: usize,
        allow_cuda_cleanup: bool,
    ) -> Self {
        Self {
            name: Arc::new("memory_cleanup".to_string()),
            device: device.clone(),
            every_epochs,
            every_iters,
            last_epoch: None,
            allow_cuda_cleanup,
        }
    }
}

impl<B: BackendTrait> burn_train::metric::Metric for MemoryCleanupMetric<B>
where
    B::Device: 'static,
{
    type Input = ();

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        _item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        let allow_cleanup = allow_memory_cleanup::<B>(&self.device, self.allow_cuda_cleanup);
        if self.every_epochs == 0 && self.every_iters == 0 {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "disabled".to_string(),
                "0".to_string(),
            );
        }

        let epoch = metadata.epoch;
        let mut cleaned = false;
        if allow_cleanup
            && self.every_iters > 0
            && metadata.iteration.is_multiple_of(self.every_iters)
        {
            let _guard = crate::device::device_allocation_lock().lock().ok();
            B::sync(&self.device);
            B::memory_cleanup(&self.device);
            extra_memory_cleanup::<B>(&self.device);
            cleaned = true;
        }
        if let Some(last_epoch) = self.last_epoch
            && allow_cleanup
            && self.every_epochs > 0
            && epoch != last_epoch
            && epoch.is_multiple_of(self.every_epochs)
        {
            let _guard = crate::device::device_allocation_lock().lock().ok();
            B::sync(&self.device);
            B::memory_cleanup(&self.device);
            extra_memory_cleanup::<B>(&self.device);
            cleaned = true;
        }
        self.last_epoch = Some(epoch);

        burn_train::metric::MetricEntry::new(
            Arc::clone(&self.name),
            if cleaned {
                "cleaned".to_string()
            } else if !allow_cleanup {
                "disabled".to_string()
            } else {
                "skip".to_string()
            },
            if cleaned {
                "1".to_string()
            } else {
                "0".to_string()
            },
        )
    }

    fn clear(&mut self) {}
}

#[derive(Clone)]
pub struct DeviceMemoryMetric<B: BackendTrait> {
    name: Arc<String>,
    device: B::Device,
    every_iters: usize,
    max_bytes: u64,
    allow_cuda_cleanup: bool,
    last: Option<DeviceMemoryUsage>,
}

impl<B: BackendTrait> DeviceMemoryMetric<B> {
    pub fn new(
        device: &B::Device,
        every_iters: usize,
        max_device_memory_mb: usize,
        allow_cuda_cleanup: bool,
    ) -> Self {
        Self {
            name: Arc::new("device_memory_mb".to_string()),
            device: device.clone(),
            every_iters: every_iters.max(1),
            max_bytes: (max_device_memory_mb as u64).saturating_mul(1024 * 1024),
            allow_cuda_cleanup,
            last: None,
        }
    }
}

impl<B: BackendTrait> burn_train::metric::Metric for DeviceMemoryMetric<B>
where
    B::Device: 'static,
{
    type Input = ();

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        _item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::MetricEntry {
        if self.every_iters > 1
            && !metadata.iteration.is_multiple_of(self.every_iters)
            && let Some(last) = self.last
        {
            let value = format!("{:.1}/{:.1} MiB", last.reserved_mb(), last.in_use_mb());
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                value.clone(),
                value,
            );
        }

        let Some(mut usage) = device_memory_usage_safe::<B>(&self.device) else {
            return burn_train::metric::MetricEntry::new(
                Arc::clone(&self.name),
                "unsupported".to_string(),
                "0".to_string(),
            );
        };

        if self.max_bytes > 0 {
            let mut current = usage.reserved_bytes.max(usage.in_use_bytes);
            if current > self.max_bytes {
                let allow_cleanup =
                    allow_memory_cleanup::<B>(&self.device, self.allow_cuda_cleanup);
                if allow_cleanup {
                    let _guard = crate::device::device_allocation_lock().lock().ok();
                    B::sync(&self.device);
                    B::memory_cleanup(&self.device);
                    extra_memory_cleanup::<B>(&self.device);
                    B::sync(&self.device);
                    if let Some(cleaned) = device_memory_usage_safe::<B>(&self.device) {
                        usage = cleaned;
                        current = usage.reserved_bytes.max(usage.in_use_bytes);
                    }
                }
                if current > self.max_bytes {
                    let max_mb = bytes_to_mb(self.max_bytes);
                    panic!(
                        "device memory usage exceeded cap: reserved={:.1} MiB in_use={:.1} MiB cap={:.1} MiB",
                        usage.reserved_mb(),
                        usage.in_use_mb(),
                        max_mb
                    );
                }
            }
        }

        self.last = Some(usage);
        let value = format!("{:.1}/{:.1} MiB", usage.reserved_mb(), usage.in_use_mb());
        burn_train::metric::MetricEntry::new(Arc::clone(&self.name), value.clone(), value)
    }

    fn clear(&mut self) {
        self.last = None;
    }
}

mod vision_artifact;
pub use vision_artifact::VisionArtifactMetric;

#[cfg(test)]
mod tests {
    use crate::train::metrics::*;
    use crate::train::test_support::create_stub_ffmpeg;
    use burn::data::dataloader::Progress;
    use burn_ndarray::NdArray;
    use burn_train::metric::{Metric, MetricMetadata};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::tempdir;

    #[derive(Clone)]
    struct CountValue {
        counter: Arc<AtomicUsize>,
    }

    impl<B: BackendTrait> ScalarValue<B> for CountValue {
        fn value(&self) -> Tensor<B, 1> {
            self.counter.fetch_add(1, Ordering::SeqCst);
            let device = <B as BackendTrait>::Device::default();
            Tensor::<B, 1>::zeros([1], &device)
        }
    }

    fn test_metadata(iteration: usize) -> MetricMetadata {
        MetricMetadata {
            progress: Progress::new(1, 1),
            epoch: 0,
            epoch_total: 1,
            iteration,
            lr: None,
        }
    }

    fn test_metadata_epoch(iteration: usize, epoch: usize) -> MetricMetadata {
        MetricMetadata {
            progress: Progress::new(1, 1),
            epoch,
            epoch_total: 1,
            iteration,
            lr: None,
        }
    }

    #[test]
    fn scalar_metric_respects_every() {
        type Backend = NdArray<f32>;
        let counter = Arc::new(AtomicUsize::new(0));
        let mut metric = ScalarMetric::<Backend, CountValue>::new_every("test_scalar", 2);
        for iteration in 0..4 {
            let input = CountValue {
                counter: Arc::clone(&counter),
            };
            metric.update(&input, &test_metadata(iteration));
        }
        assert_eq!(counter.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn artifact_images_write_png() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Images,
            4,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            None,
        );
        let views = Tensor::<Backend, 5>::zeros([1, 1, 3, 4, 4], &device);
        let patch_norms = Tensor::<Backend, 3>::zeros([1, 2, 2], &device);
        let input = VisionArtifactInput {
            views: Some(views),
            frames: None,
            patch_norms: Some(patch_norms),
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: Some(vec!["input".to_string()]),
        };
        let _ = metric.update(&input, &test_metadata(0));
        assert!(output_dir.path().join("sample_00.png").is_file());
        let key_path = output_dir.path().join("vision_artifacts_key.txt");
        let key_contents = fs::read_to_string(key_path).expect("legend");
        assert!(key_contents.contains("Column 1"));
    }

    #[test]
    fn artifact_images_overwrite_reuses_filename() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Images,
            4,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            None,
        );
        let views = Tensor::<Backend, 5>::zeros([1, 1, 3, 4, 4], &device);
        let patch_norms = Tensor::<Backend, 3>::zeros([1, 2, 2], &device);
        let input = VisionArtifactInput {
            views: Some(views),
            frames: None,
            patch_norms: Some(patch_norms),
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: None,
        };
        let _ = metric.update(&input, &test_metadata(0));
        let _ = metric.update(&input, &test_metadata(1));
        let png_count = fs::read_dir(output_dir.path())
            .expect("read dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(png_count, 1);
    }

    #[test]
    fn artifact_images_non_overwrite_uses_iteration_name() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Images,
            4,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            false,
            None,
        );
        let views = Tensor::<Backend, 5>::zeros([1, 1, 3, 4, 4], &device);
        let patch_norms = Tensor::<Backend, 3>::zeros([1, 2, 2], &device);
        let input = VisionArtifactInput {
            views: Some(views),
            frames: None,
            patch_norms: Some(patch_norms),
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: None,
        };
        let _ = metric.update(&input, &test_metadata(5));
        let expected = output_dir.path().join("lejepa_iter_000005_sample_00.png");
        assert!(expected.is_file());
    }

    #[test]
    fn artifact_images_respects_epoch_budget() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Images,
            1,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            None,
        );
        let views = Tensor::<Backend, 5>::zeros([2, 1, 3, 4, 4], &device);
        let patch_norms = Tensor::<Backend, 3>::zeros([2, 2, 2], &device);
        let input = VisionArtifactInput {
            views: Some(views),
            frames: None,
            patch_norms: Some(patch_norms),
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: None,
        };
        let _ = metric.update(&input, &test_metadata_epoch(0, 0));
        let _ = metric.update(&input, &test_metadata_epoch(1, 0));
        let png_count = fs::read_dir(output_dir.path())
            .expect("read dir")
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .path()
                    .extension()
                    .and_then(|ext| ext.to_str())
                    .map(|ext| ext.eq_ignore_ascii_case("png"))
                    .unwrap_or(false)
            })
            .count();
        assert_eq!(png_count, 1);
    }

    #[test]
    fn artifact_avi_write_video() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Avi,
            4,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            None,
        );
        let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &device);
        let input = VisionArtifactInput {
            views: None,
            frames: Some(frames),
            patch_norms: None,
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: Some(vec!["frame".to_string()]),
        };
        let _ = metric.update(&input, &test_metadata(0));
        let path = output_dir.path().join("sample_00.avi");
        assert!(path.is_file());
        let bytes = fs::read(path).expect("read avi");
        assert!(bytes.starts_with(b"RIFF"));
        let key_path = output_dir.path().join("vision_artifacts_key.txt");
        assert!(key_path.is_file());
    }

    #[test]
    fn artifact_mp4_write_video() {
        type Backend = NdArray<f32>;
        let device = <Backend as BackendTrait>::Device::default();
        let output_dir = tempdir().expect("tempdir");
        let bin_dir = output_dir.path().join("bin");
        let script_path = create_stub_ffmpeg(&bin_dir).expect("ffmpeg stub");
        let mut metric = VisionArtifactMetric::<Backend>::new(
            output_dir.path().to_path_buf(),
            1,
            VisionArtifactOutputMode::Mp4,
            4,
            4,
            [0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            true,
            Some(script_path.clone()),
        );
        let frames = Tensor::<Backend, 5>::zeros([1, 2, 3, 4, 4], &device);
        let input = VisionArtifactInput {
            views: None,
            frames: Some(frames),
            patch_norms: None,
            pca_rgb: None,
            patch_norms_steps: None,
            pca_rgb_steps: None,
            probe_logits: None,
            labels: None,
            legend: Some(vec!["frame".to_string()]),
        };
        let _ = metric.update(&input, &test_metadata(0));
        let path = output_dir.path().join("sample_00.mp4");
        assert!(path.is_file());
        let key_path = output_dir.path().join("vision_artifacts_key.txt");
        assert!(key_path.is_file());
    }
}
