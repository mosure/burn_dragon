#![cfg_attr(not(feature = "cli"), allow(dead_code))]

use std::any::Any;
use std::sync::Arc;
#[cfg(feature = "integration_test")]
use std::sync::Mutex;
#[cfg(feature = "integration_test")]
use std::sync::OnceLock;

use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn::tensor::Tensor;
use burn_cubecl::cubecl::Runtime;
use burn_ndarray::NdArray;
use burn_train::metric::{Adaptor, ItemLazy, LossInput};
pub type MetricsBackend = NdArray<f32>;

fn serialized_entry(
    formatted: impl Into<String>,
    serialized: impl Into<String>,
) -> burn_train::metric::SerializedEntry {
    burn_train::metric::SerializedEntry::new(formatted.into(), serialized.into())
}

fn metric_iteration(metadata: &burn_train::metric::MetricMetadata) -> usize {
    metadata.iteration.unwrap_or(0)
}

fn should_emit_metric(metadata: &burn_train::metric::MetricMetadata, every: usize) -> bool {
    every <= 1
        || metadata
            .iteration
            .is_some_and(|iteration| iteration.is_multiple_of(every))
}

fn metric_epoch(metadata: &burn_train::metric::MetricMetadata) -> usize {
    metadata.global_progress.items_processed
}

fn sync_float_tensor<B: BackendTrait, const D: usize>(
    tensor: Tensor<B, D>,
) -> Tensor<MetricsBackend, D> {
    Tensor::<MetricsBackend, D>::from_data(tensor.into_data(), &Default::default())
}

fn sync_optional_float_tensor<B: BackendTrait, const D: usize>(
    tensor: Option<Tensor<B, D>>,
) -> Option<Tensor<MetricsBackend, D>> {
    tensor.map(sync_float_tensor)
}

mod language;

pub use language::{LanguageModelOutput, LanguageModelTrainItem, LossValue};

pub trait ScalarValue<B: BackendTrait> {
    fn value(&self) -> Tensor<B, 1>;
}

pub trait OptionalScalarValue<B: BackendTrait> {
    fn value(&self) -> Option<Tensor<B, 1>>;
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
    ) -> burn_train::metric::SerializedEntry {
        if !should_emit_metric(metadata, self.every) && self.initialized {
            return serialized_entry(
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
        serialized_entry(
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

    fn running_value(&self) -> burn_train::metric::NumericEntry {
        self.value()
    }
}

pub struct OptionalScalarMetric<B: BackendTrait, I: OptionalScalarValue<B>> {
    name: Arc<String>,
    last: f64,
    every: usize,
    initialized: bool,
    _marker: std::marker::PhantomData<(B, I)>,
}

impl<B: BackendTrait, I: OptionalScalarValue<B>> Clone for OptionalScalarMetric<B, I> {
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

impl<B: BackendTrait, I: OptionalScalarValue<B>> OptionalScalarMetric<B, I> {
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

impl<B: BackendTrait, I: OptionalScalarValue<B> + Send + Sync> burn_train::metric::Metric
    for OptionalScalarMetric<B, I>
{
    type Input = I;

    fn name(&self) -> burn_train::metric::MetricName {
        Arc::clone(&self.name)
    }

    fn update(
        &mut self,
        item: &Self::Input,
        metadata: &burn_train::metric::MetricMetadata,
    ) -> burn_train::metric::SerializedEntry {
        if !should_emit_metric(metadata, self.every) && self.initialized {
            return serialized_entry(
                burn_train::metric::format_float(self.last, 4),
                self.last.to_string(),
            );
        }
        if let Some(value) = item.value() {
            self.last = value
                .mean()
                .into_data()
                .iter::<f64>()
                .next()
                .unwrap_or(self.last);
            self.initialized = true;
        }
        serialized_entry(
            burn_train::metric::format_float(self.last, 4),
            self.last.to_string(),
        )
    }

    fn clear(&mut self) {
        self.last = 0.0;
        self.initialized = false;
    }
}

impl<B: BackendTrait, I: OptionalScalarValue<B> + Send + Sync> burn_train::metric::Numeric
    for OptionalScalarMetric<B, I>
{
    fn value(&self) -> burn_train::metric::NumericEntry {
        burn_train::metric::NumericEntry::Value(self.last)
    }

    fn running_value(&self) -> burn_train::metric::NumericEntry {
        self.value()
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
    ) -> burn_train::metric::SerializedEntry {
        if !should_emit_metric(metadata, self.every) && self.initialized {
            return serialized_entry(
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
        serialized_entry(
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
    ) -> burn_train::metric::SerializedEntry {
        serialized_entry(self.value.to_string(), self.value.to_string())
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
            <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device)
                .memory_cleanup();
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
        let usage =
            <burn_cubecl::cubecl::cuda::CudaRuntime as Runtime>::client(cuda_device).memory_usage();
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
    ) -> burn_train::metric::SerializedEntry {
        let allow_cleanup = allow_memory_cleanup::<B>(&self.device, self.allow_cuda_cleanup);
        if self.every_epochs == 0 && self.every_iters == 0 {
            return serialized_entry("disabled", "0");
        }

        let epoch = metric_epoch(metadata);
        let mut cleaned = false;
        if allow_cleanup
            && self.every_iters > 0
            && metadata
                .iteration
                .is_some_and(|iteration| iteration.is_multiple_of(self.every_iters))
        {
            let _guard = crate::device::device_allocation_lock().lock().ok();
            let _ = B::sync(&self.device);
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
            let _ = B::sync(&self.device);
            B::memory_cleanup(&self.device);
            extra_memory_cleanup::<B>(&self.device);
            cleaned = true;
        }
        self.last_epoch = Some(epoch);

        serialized_entry(
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
    ) -> burn_train::metric::SerializedEntry {
        if !should_emit_metric(metadata, self.every_iters)
            && let Some(last) = self.last
        {
            let value = format!("{:.1}/{:.1} MiB", last.reserved_mb(), last.in_use_mb());
            return serialized_entry(value.clone(), value);
        }

        let Some(mut usage) = device_memory_usage_safe::<B>(&self.device) else {
            return serialized_entry("unsupported", "0");
        };

        if self.max_bytes > 0 {
            let mut current = usage.reserved_bytes.max(usage.in_use_bytes);
            if current > self.max_bytes {
                let allow_cleanup =
                    allow_memory_cleanup::<B>(&self.device, self.allow_cuda_cleanup);
                if allow_cleanup {
                    let _guard = crate::device::device_allocation_lock().lock().ok();
                    let _ = B::sync(&self.device);
                    B::memory_cleanup(&self.device);
                    extra_memory_cleanup::<B>(&self.device);
                    let _ = B::sync(&self.device);
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
        serialized_entry(value.clone(), value)
    }

    fn clear(&mut self) {
        self.last = None;
    }
}

#[cfg(test)]
mod tests {
    use crate::train::metrics::*;
    use burn::data::dataloader::Progress;
    use burn_ndarray::NdArray;
    use burn_train::metric::{Metric, MetricMetadata};
    use std::sync::atomic::{AtomicUsize, Ordering};

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
            global_progress: Progress::new(0, 1),
            iteration: Some(iteration),
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
}
