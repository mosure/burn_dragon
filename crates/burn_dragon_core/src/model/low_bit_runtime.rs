use burn::tensor::backend::Backend;
use burn::tensor::{ElementConversion, Int, Tensor, TensorData};
#[cfg(any(feature = "benchmark", feature = "train", feature = "cuda"))]
use burn_cubecl::cubecl::Runtime;
#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;
#[cfg(feature = "cuda")]
use burn_cuda::CudaDevice;
use burn_dragon_kernel::api::low_bit::{
    cached_wgpu_packed_dot_decoder_tail_support, cached_wgpu_packed_dot_lowrank_support,
    pack_decoder_input_codes_i8x4, pack_decoder_weight_codes_i8x4, pack_lowrank_input_codes_i8x4,
    pack_lowrank_weight_codes_i8x4, pack_rho_int8_block_device_reference,
    packed_decoder_tail_device_reference, packed_lowrank_projection_device_reference,
    supports_packed_low_bit_device_backend, supports_packed_rho_int8_block_device_backend,
    try_fused_packed_decoder_tail, try_fused_packed_decoder_tail_training_autodiff,
    try_fused_packed_lowrank_projection, try_fused_packed_lowrank_training_autodiff,
    try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale,
    try_raw_cuda_packed_decoder_tail, try_raw_cuda_packed_decoder_tail_device_scale,
    try_raw_cuda_packed_decoder_tail_prepacked_input,
    try_raw_cuda_packed_decoder_tail_prepacked_input_device_scale,
    try_raw_cuda_packed_lowrank_projection, try_raw_cuda_packed_lowrank_projection_device_scale,
    try_raw_cuda_packed_lowrank_projection_prepacked_input,
    try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale,
    try_raw_cuda_quantize_pack_activation_i8x4, try_wgpu_packed_dot_decoder_tail_device_scale,
    try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale,
    try_wgpu_packed_dot_lowrank_projection_device_scale,
    try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale,
    try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale,
    try_wgpu_quantize_activation_codes_i32, try_wgpu_quantize_pack_activation_i8x4,
    unpack_rho_int8_block_device_reference,
};
#[cfg(any(feature = "benchmark", feature = "train"))]
use burn_wgpu::{WgpuDevice, WgpuRuntime};
use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use crate::experimental::bitnet_reference::{
    PackedTernaryBuffer, PackedWeightArtifact, pack_binary_1bit, pack_ternary_2bit,
    quantize_binary_sign, quantize_ternary_absmean, unpack_binary_1bit, unpack_ternary_2bit,
    unpack_weight_artifact_to_i8_codes,
};
use crate::model::low_bit::{
    LowBitActivationFormat, LowBitInferenceMode, LowBitQuantizationConfig,
    LowBitSavedActivationMode, LowBitTargetModule, LowBitWeightFormat, RhoCompressionConfig,
};

const QUANT_EPSILON: f32 = 1.0e-8;
const RHO_BLOCK_SIZE: usize = 32;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitProjectionPlan {
    pub strict_bitnet_reference: bool,
    pub x_weight_format: Option<LowBitWeightFormat>,
    pub x_activation_format: Option<LowBitActivationFormat>,
    pub y_weight_format: Option<LowBitWeightFormat>,
    pub y_activation_format: Option<LowBitActivationFormat>,
    pub residual_weight_format: Option<LowBitWeightFormat>,
    pub residual_activation_format: Option<LowBitActivationFormat>,
}

#[derive(Clone, Debug)]
pub struct CachedLowrankProjectionArtifact<B: Backend> {
    pub artifact: PackedWeightArtifact,
    pub codes: Tensor<B, 3, Int>,
    pub packed_weight: Tensor<B, 3, Int>,
    pub latent_out: usize,
}

#[derive(Clone, Debug)]
pub struct CachedDecoderTailArtifact<B: Backend> {
    pub artifact: PackedWeightArtifact,
    pub codes: Tensor<B, 2, Int>,
    pub packed_weight: Tensor<B, 2, Int>,
}

#[derive(Clone, Debug)]
struct CachedDecoderTailRuntimeView<B: Backend> {
    codes: Tensor<B, 2, Int>,
    packed_weight: Tensor<B, 2, Int>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct PackedLowBitProjectionArtifacts<'a, B: Backend> {
    pub runtime: LowBitKernelRuntimeKind,
    pub x: Option<&'a PackedWeightArtifact>,
    pub y: Option<&'a PackedWeightArtifact>,
    pub residual: Option<&'a PackedWeightArtifact>,
    pub _marker: PhantomData<B>,
}

impl<B: Backend> PackedLowBitProjectionArtifacts<'_, B> {
    pub fn has_any(&self) -> bool {
        self.x.is_some() || self.y.is_some() || self.residual.is_some()
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowBitKernelRuntimeKind {
    #[default]
    FakeQuantReference,
    PackedReference,
    PackedNativeInference,
    PackedNativeTrainingForward,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowBitKernelFallbackReason {
    QuantDisabled,
    MissingPackedArtifacts,
    #[default]
    NativeLowBitUnavailable,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitKernelCapabilities {
    pub packed_reference_supported: bool,
    pub native_low_bit_supported: bool,
    pub native_projection_supported: bool,
    pub native_decoder_tail_supported: bool,
    pub native_rho_int8_block_supported: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitKernelPlan {
    pub runtime: LowBitKernelRuntimeKind,
    pub capabilities: LowBitKernelCapabilities,
    pub packed_static_ready: bool,
    pub fallback_reason: Option<LowBitKernelFallbackReason>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitMemoryEstimateInput {
    pub batch_size: usize,
    pub time_steps: usize,
    pub n_layer: usize,
    pub n_head: usize,
    pub n_embd: usize,
    pub latent_total: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LowBitSavedActivationRecomputePolicy {
    #[default]
    Disabled,
    QuantizedCacheOnly,
    RecomputeSafeWithinWindow,
}

impl LowBitSavedActivationRecomputePolicy {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Disabled => "disabled",
            Self::QuantizedCacheOnly => "quantized_cache_only",
            Self::RecomputeSafeWithinWindow => "recompute_safe_within_window",
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LowBitSavedActivationTensorInventoryEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub element_count: u64,
    pub estimated_bytes: u64,
    pub recompute_policy: LowBitSavedActivationRecomputePolicy,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LowBitSavedActivationInventory {
    pub mode: LowBitSavedActivationMode,
    pub format: LowBitActivationFormat,
    pub requires_rho_window_anchor: bool,
    pub tensors: Vec<LowBitSavedActivationTensorInventoryEntry>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitMemoryBucketEstimate {
    pub master_weight_bytes: u64,
    pub execution_weight_bytes: u64,
    pub activation_shell_bytes: u64,
    pub saved_activation_bytes: u64,
    pub rho_state_bytes: u64,
    pub workspace_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitNativeProjectionProfileSnapshot {
    pub calls: u64,
    pub total_ns: u128,
    pub quantize_ns: u128,
    pub prepacked_quantize_ns: u128,
    pub raw_cuda_ns: u128,
    pub fused_ns: u128,
    pub reference_ns: u128,
    pub dynamic_scale_calls: u64,
    pub cached_scale_hits: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitTrainingProjectionMemoryStageSnapshot {
    pub reserved_bytes: u64,
    pub in_use_bytes: u64,
    pub tracked_tensor_bytes: u64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitTrainingQuantizeProfileSnapshot {
    pub lowrank_direct_calls: u64,
    pub lowrank_direct_total_ns: u128,
    pub lowrank_fallback_calls: u64,
    pub lowrank_fallback_total_ns: u128,
    pub decoder_tail_calls: u64,
    pub decoder_tail_total_ns: u128,
}

impl LowBitTrainingProjectionMemoryStageSnapshot {
    fn should_replace(self, observed: Self) -> bool {
        (
            observed.in_use_bytes,
            observed.reserved_bytes,
            observed.tracked_tensor_bytes,
        ) > (
            self.in_use_bytes,
            self.reserved_bytes,
            self.tracked_tensor_bytes,
        )
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LowBitTrainingProjectionMemoryProfileSnapshot {
    pub calls: u64,
    pub after_weight_codes: LowBitTrainingProjectionMemoryStageSnapshot,
    pub after_activation_codes: LowBitTrainingProjectionMemoryStageSnapshot,
    pub after_output: LowBitTrainingProjectionMemoryStageSnapshot,
}

impl LowBitMemoryBucketEstimate {
    pub fn estimated_total_bytes(&self) -> u64 {
        self.master_weight_bytes
            .saturating_add(self.execution_weight_bytes)
            .saturating_add(self.activation_shell_bytes)
            .saturating_add(self.saved_activation_bytes)
            .saturating_add(self.rho_state_bytes)
            .saturating_add(self.workspace_bytes)
    }
}

/// Host-side rho block encoding used for chunk-compressed recurrent state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PackedRhoBlockEncoding {
    Int8,
    Ternary2,
    Binary1,
}

impl PackedRhoBlockEncoding {
    fn from_compression(compression: RhoCompressionConfig) -> Option<Self> {
        match compression {
            RhoCompressionConfig::Int8BlockExp => Some(Self::Int8),
            RhoCompressionConfig::TernaryBlockExp => Some(Self::Ternary2),
            RhoCompressionConfig::BinaryBlockExp => Some(Self::Binary1),
            _ => None,
        }
    }
}

/// Host-side rho block state.
///
/// This is the portable fallback representation for chunk-compressed rho when no
/// backend-native device path exists. Each block stores one scale plus packed codes.
#[derive(Clone, Debug, PartialEq)]
pub struct PackedRhoBlockState {
    pub logical_shape: [usize; 4],
    pub block_size: usize,
    pub encoding: PackedRhoBlockEncoding,
    pub scales: Vec<f32>,
    pub packed: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PackedSavedActivationBuffer {
    Float(Vec<f32>),
    Signed { values: Vec<i8>, scale: f32 },
    UnsignedPositive { values: Vec<u8>, scale: f32 },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PackedSavedActivationState {
    pub logical_shape: Vec<usize>,
    pub format: LowBitActivationFormat,
    pub estimated_bytes: u64,
    pub buffer: PackedSavedActivationBuffer,
}

#[derive(Clone, Debug)]
pub struct PackedRhoInt8DeviceState<B: Backend> {
    pub logical_shape: [usize; 4],
    pub block_size: usize,
    pub scales: Tensor<B, 1>,
    pub packed: Tensor<B, 1, Int>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RhoCompressionStatsSnapshot {
    pub calls: u64,
    pub total_original_rms: f64,
    pub total_reconstructed_rms: f64,
    pub total_mean_abs_error: f64,
    pub max_abs_error: f32,
    pub total_original_bytes: u64,
    pub total_compressed_bytes: u64,
}

impl RhoCompressionStatsSnapshot {
    pub fn mean_abs_error(&self) -> Option<f32> {
        (self.calls > 0).then(|| (self.total_mean_abs_error / self.calls as f64) as f32)
    }

    pub fn mean_rms_ratio(&self) -> Option<f32> {
        if self.calls == 0 || self.total_original_rms <= 0.0 {
            return None;
        }
        Some((self.total_reconstructed_rms / self.total_original_rms) as f32)
    }

    pub fn compression_ratio(&self) -> Option<f32> {
        if self.total_original_bytes == 0 {
            return None;
        }
        Some(self.total_compressed_bytes as f32 / self.total_original_bytes as f32)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RhoCompressionQualityGate {
    pub max_mean_abs_error: f32,
    pub max_max_abs_error: f32,
    pub min_mean_rms_ratio: f32,
    pub max_mean_rms_ratio: f32,
    pub max_compression_ratio: f32,
}

impl Default for RhoCompressionQualityGate {
    fn default() -> Self {
        Self {
            max_mean_abs_error: 0.02,
            max_max_abs_error: 0.05,
            min_mean_rms_ratio: 0.95,
            max_mean_rms_ratio: 1.05,
            max_compression_ratio: 0.5,
        }
    }
}

#[derive(Default)]
struct RhoCompressionStatsState {
    calls: u64,
    total_original_rms: f64,
    total_reconstructed_rms: f64,
    total_mean_abs_error: f64,
    max_abs_error: f32,
    total_original_bytes: u64,
    total_compressed_bytes: u64,
}

static RHO_COMPRESSION_PROFILE: OnceLock<Mutex<RhoCompressionStatsState>> = OnceLock::new();

impl<B: Backend> PackedRhoInt8DeviceState<B> {
    pub fn detach(&self) -> Self {
        Self {
            logical_shape: self.logical_shape,
            block_size: self.block_size,
            scales: self.scales.clone().detach(),
            packed: self.packed.clone(),
        }
    }
}

impl LowBitProjectionPlan {
    pub fn from_config(config: &LowBitQuantizationConfig) -> Self {
        if !config.enable {
            return Self::default();
        }

        let x_enabled = config
            .target_modules
            .iter()
            .any(|module| matches!(module, LowBitTargetModule::DecoderX));
        let y_enabled = config
            .target_modules
            .iter()
            .any(|module| matches!(module, LowBitTargetModule::DecoderY));
        let residual_enabled = config
            .target_modules
            .iter()
            .any(|module| matches!(module, LowBitTargetModule::Encoder));

        Self {
            strict_bitnet_reference: config.strict_bitnet_reference,
            x_weight_format: x_enabled.then_some(config.decoder_x_mode),
            x_activation_format: x_enabled.then_some(config.act_format),
            y_weight_format: y_enabled.then_some(config.weight_format),
            y_activation_format: y_enabled.then_some(config.act_format),
            residual_weight_format: residual_enabled
                .then_some(config.encoder_mode.unwrap_or(config.weight_format)),
            residual_activation_format: residual_enabled.then_some(config.act_format),
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.x_weight_format.is_some()
            || self.y_weight_format.is_some()
            || self.residual_weight_format.is_some()
    }
}

impl LowBitKernelRuntimeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::FakeQuantReference => "fake_quant_reference",
            Self::PackedReference => "packed_reference",
            Self::PackedNativeInference => "packed_native_inference",
            Self::PackedNativeTrainingForward => "packed_native_training_forward",
        }
    }
}

impl LowBitKernelCapabilities {
    pub fn any_native_supported(&self) -> bool {
        self.native_low_bit_supported
            || self.native_projection_supported
            || self.native_decoder_tail_supported
            || self.native_rho_int8_block_supported
    }
}

fn backend_name_prefers_training_low_bit_runtime(backend_name: &str) -> bool {
    backend_name.contains("Autodiff")
}

fn native_training_supports_decoder_x_format(format: LowBitWeightFormat) -> bool {
    matches!(format, LowBitWeightFormat::Int8 | LowBitWeightFormat::Sign1)
}

fn native_training_supports_weight_format(format: LowBitWeightFormat) -> bool {
    !matches!(format, LowBitWeightFormat::Fp16)
}

fn config_prefers_native_training_runtime(config: &LowBitQuantizationConfig) -> bool {
    for module in &config.target_modules {
        match module {
            LowBitTargetModule::DecoderX => {
                if !native_training_supports_decoder_x_format(config.decoder_x_mode) {
                    return false;
                }
            }
            LowBitTargetModule::DecoderY => {
                if !native_training_supports_weight_format(config.weight_format) {
                    return false;
                }
            }
            LowBitTargetModule::Encoder => {
                if !native_training_supports_weight_format(
                    config.encoder_mode.unwrap_or(config.weight_format),
                ) {
                    return false;
                }
            }
        }
    }
    true
}

fn backend_name_prefers_inference_cached_scale(backend_name: &str) -> bool {
    let normalized = backend_name.trim().to_ascii_lowercase();
    normalized.contains("wgpu") || normalized.contains("burn_wgpu")
}

pub fn low_bit_kernel_capabilities<B: Backend>() -> LowBitKernelCapabilities {
    let _ = core::any::type_name::<B>();
    LowBitKernelCapabilities {
        packed_reference_supported: true,
        native_low_bit_supported: supports_packed_low_bit_device_backend::<B>(),
        native_projection_supported: supports_packed_low_bit_device_backend::<B>(),
        native_decoder_tail_supported: supports_packed_low_bit_device_backend::<B>(),
        native_rho_int8_block_supported: supports_packed_rho_int8_block_device_backend::<B>(),
    }
}

pub fn low_bit_kernel_capabilities_for_backend_name(
    backend_name: &str,
) -> LowBitKernelCapabilities {
    let normalized = backend_name.trim().to_ascii_lowercase();
    let _ = normalized;
    LowBitKernelCapabilities {
        packed_reference_supported: true,
        native_low_bit_supported: true,
        native_projection_supported: true,
        native_decoder_tail_supported: true,
        native_rho_int8_block_supported: true,
    }
}

pub fn resolve_low_bit_kernel_plan_from_capabilities(
    config: &LowBitQuantizationConfig,
    packed_artifacts: PackedLowBitProjectionArtifacts<'_, impl Backend>,
    capabilities: LowBitKernelCapabilities,
) -> LowBitKernelPlan {
    resolve_low_bit_kernel_plan_from_packed_static_ready(
        config,
        packed_artifacts.has_any(),
        capabilities,
    )
}

fn resolve_low_bit_kernel_plan_from_packed_static_ready(
    config: &LowBitQuantizationConfig,
    packed_static_ready: bool,
    capabilities: LowBitKernelCapabilities,
) -> LowBitKernelPlan {
    if !config.enable {
        return LowBitKernelPlan {
            runtime: LowBitKernelRuntimeKind::FakeQuantReference,
            capabilities,
            packed_static_ready,
            fallback_reason: Some(LowBitKernelFallbackReason::QuantDisabled),
        };
    }

    if matches!(config.inference_mode, LowBitInferenceMode::OfflinePack) {
        if packed_static_ready && capabilities.native_projection_supported {
            return LowBitKernelPlan {
                runtime: LowBitKernelRuntimeKind::PackedNativeInference,
                capabilities,
                packed_static_ready,
                fallback_reason: None,
            };
        }

        if packed_static_ready && capabilities.packed_reference_supported {
            return LowBitKernelPlan {
                runtime: LowBitKernelRuntimeKind::PackedReference,
                capabilities,
                packed_static_ready,
                fallback_reason: None,
            };
        }

        return LowBitKernelPlan {
            runtime: LowBitKernelRuntimeKind::FakeQuantReference,
            capabilities,
            packed_static_ready,
            fallback_reason: Some(if packed_static_ready {
                LowBitKernelFallbackReason::NativeLowBitUnavailable
            } else {
                LowBitKernelFallbackReason::MissingPackedArtifacts
            }),
        };
    }

    if matches!(
        config.training_mode,
        crate::model::low_bit::LowBitTrainingMode::TrainKernelExp
    ) && capabilities.native_projection_supported
        && config_prefers_native_training_runtime(config)
    {
        return LowBitKernelPlan {
            runtime: LowBitKernelRuntimeKind::PackedNativeTrainingForward,
            capabilities,
            packed_static_ready,
            fallback_reason: None,
        };
    }

    LowBitKernelPlan {
        runtime: LowBitKernelRuntimeKind::FakeQuantReference,
        capabilities,
        packed_static_ready,
        fallback_reason: None,
    }
}

pub fn resolve_low_bit_kernel_plan<B: Backend>(
    config: &LowBitQuantizationConfig,
    packed_artifacts: PackedLowBitProjectionArtifacts<'_, B>,
) -> LowBitKernelPlan {
    let capabilities = low_bit_kernel_capabilities::<B>();
    if config.enable
        && matches!(
            config.training_mode,
            crate::model::low_bit::LowBitTrainingMode::TrainKernelExp
        )
        && capabilities.native_projection_supported
        && config_prefers_native_training_runtime(config)
        && backend_name_prefers_training_low_bit_runtime(core::any::type_name::<B>())
    {
        return LowBitKernelPlan {
            runtime: LowBitKernelRuntimeKind::PackedNativeTrainingForward,
            capabilities,
            packed_static_ready: packed_artifacts.has_any(),
            fallback_reason: None,
        };
    }

    resolve_low_bit_kernel_plan_from_capabilities(config, packed_artifacts, capabilities)
}

pub fn resolve_low_bit_kernel_plan_for_backend_name(
    backend_name: &str,
    config: &LowBitQuantizationConfig,
    packed_static_ready: bool,
) -> LowBitKernelPlan {
    resolve_low_bit_kernel_plan_from_packed_static_ready(
        config,
        packed_static_ready,
        low_bit_kernel_capabilities_for_backend_name(backend_name),
    )
}

fn bytes_per_weight_element(format: LowBitWeightFormat) -> (u64, u64) {
    match format {
        LowBitWeightFormat::Fp16 => (2, 1),
        LowBitWeightFormat::Int8 => (1, 1),
        LowBitWeightFormat::Sign1 => (1, 8),
        LowBitWeightFormat::Ternary158 | LowBitWeightFormat::Packed2 => (1, 4),
    }
}

fn bytes_per_activation_element(format: LowBitActivationFormat) -> (u64, u64) {
    match format {
        LowBitActivationFormat::Fp16 => (2, 1),
        LowBitActivationFormat::Int8 | LowBitActivationFormat::Uint8PosExp => (1, 1),
        LowBitActivationFormat::Int4Exp => (1, 2),
    }
}

fn estimate_packed_tensor_bytes(elements: u64, numerator: u64, denominator: u64) -> u64 {
    elements
        .saturating_mul(numerator)
        .div_ceil(denominator)
        .saturating_add(core::mem::size_of::<f32>() as u64)
}

fn rho_bytes_per_element(config: &crate::model::low_bit::LowBitRhoConfig) -> (u64, u64) {
    use crate::model::low_bit::{RhoCompressionConfig, RhoPrecisionConfig};

    match config.compression {
        RhoCompressionConfig::Int8BlockExp => return (9, 8),
        RhoCompressionConfig::TernaryBlockExp => return (3, 8),
        RhoCompressionConfig::BinaryBlockExp => return (1, 4),
        _ => {}
    }

    match config.precision {
        RhoPrecisionConfig::Fp32 => (4, 1),
        RhoPrecisionConfig::Bf16 => (2, 1),
        RhoPrecisionConfig::Fp8Exp
        | RhoPrecisionConfig::Int8BlockExp
        | RhoPrecisionConfig::Blockfp8Exp
        | RhoPrecisionConfig::SparseTileExp => (1, 1),
    }
}

fn saved_activation_bytes_per_element(format: LowBitActivationFormat) -> (u64, u64) {
    match format {
        LowBitActivationFormat::Fp16 => (2, 1),
        LowBitActivationFormat::Int8 | LowBitActivationFormat::Uint8PosExp => (1, 1),
        LowBitActivationFormat::Int4Exp => (1, 2),
    }
}

fn saved_activation_recompute_factor(mode: LowBitSavedActivationMode) -> (u64, u64) {
    match mode {
        LowBitSavedActivationMode::Disabled => (0, 1),
        LowBitSavedActivationMode::QuantizedCacheExp => (1, 1),
        // Approximate that quantized checkpoints plus recompute keep about half as many
        // resident saved activations live as the pure quantized-cache path.
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp => (1, 2),
    }
}

pub fn estimate_low_bit_memory_buckets(
    quant: &LowBitQuantizationConfig,
    rho: &crate::model::low_bit::LowBitRhoConfig,
    input: LowBitMemoryEstimateInput,
) -> LowBitMemoryBucketEstimate {
    if !quant.enable {
        return LowBitMemoryBucketEstimate::default();
    }

    let plan = LowBitProjectionPlan::from_config(quant);
    let latent_per_head = input.latent_total / input.n_head.max(1);
    let headwise_projection_elements =
        input.n_head as u64 * input.n_embd as u64 * latent_per_head as u64;
    let decoder_tail_elements = input.latent_total as u64 * input.n_embd as u64;

    let mut master_weight_bytes = 0u64;
    let mut execution_weight_bytes = 0u64;
    let mut activation_shell_bytes = 0u64;
    let mut saved_activation_bytes = 0u64;

    if let Some(format) = plan.x_weight_format {
        master_weight_bytes =
            master_weight_bytes.saturating_add(headwise_projection_elements.saturating_mul(4));
        let (num, den) = bytes_per_weight_element(format);
        execution_weight_bytes = execution_weight_bytes.saturating_add(
            estimate_packed_tensor_bytes(headwise_projection_elements, num, den),
        );
    }
    if let Some(format) = plan.y_weight_format {
        master_weight_bytes =
            master_weight_bytes.saturating_add(headwise_projection_elements.saturating_mul(4));
        let (num, den) = bytes_per_weight_element(format);
        execution_weight_bytes = execution_weight_bytes.saturating_add(
            estimate_packed_tensor_bytes(headwise_projection_elements, num, den),
        );
    }
    if let Some(format) = plan.residual_weight_format {
        master_weight_bytes =
            master_weight_bytes.saturating_add(decoder_tail_elements.saturating_mul(4));
        let (num, den) = bytes_per_weight_element(format);
        execution_weight_bytes = execution_weight_bytes.saturating_add(
            estimate_packed_tensor_bytes(decoder_tail_elements, num, den),
        );
    }

    let activation_tokens = input.batch_size as u64 * input.time_steps as u64;
    if let Some(format) = plan.x_activation_format {
        let (num, den) = bytes_per_activation_element(format);
        activation_shell_bytes =
            activation_shell_bytes.saturating_add(estimate_packed_tensor_bytes(
                activation_tokens.saturating_mul(input.n_embd as u64),
                num,
                den,
            ));
    }
    if let Some(format) = plan.y_activation_format {
        let (num, den) = bytes_per_activation_element(format);
        activation_shell_bytes =
            activation_shell_bytes.saturating_add(estimate_packed_tensor_bytes(
                activation_tokens
                    .saturating_mul(input.n_head as u64)
                    .saturating_mul(input.n_embd as u64),
                num,
                den,
            ));
    }
    if let Some(format) = plan.residual_activation_format {
        let (num, den) = bytes_per_activation_element(format);
        activation_shell_bytes =
            activation_shell_bytes.saturating_add(estimate_packed_tensor_bytes(
                activation_tokens
                    .saturating_mul(input.n_head as u64)
                    .saturating_mul(latent_per_head as u64),
                num,
                den,
            ));
    }

    if !matches!(
        quant.saved_activations.mode,
        LowBitSavedActivationMode::Disabled
    ) {
        let saved_activation_elements = activation_tokens
            .saturating_mul(input.n_layer as u64)
            .saturating_mul(input.n_embd as u64)
            .saturating_mul(3);
        let (bytes_num, bytes_den) =
            saved_activation_bytes_per_element(quant.saved_activations.format);
        let raw_saved_activation_bytes =
            estimate_packed_tensor_bytes(saved_activation_elements, bytes_num, bytes_den);
        let (factor_num, factor_den) =
            saved_activation_recompute_factor(quant.saved_activations.mode);
        saved_activation_bytes = raw_saved_activation_bytes
            .saturating_mul(factor_num)
            .saturating_div(factor_den.max(1));
    }

    let rho_elements = input
        .batch_size
        .saturating_mul(input.n_layer)
        .saturating_mul(input.n_head)
        .saturating_mul(latent_per_head)
        .saturating_mul(input.n_embd) as u64;
    let (rho_num, rho_den) = rho_bytes_per_element(rho);
    let rho_state_bytes = estimate_packed_tensor_bytes(rho_elements, rho_num, rho_den);

    let mut workspace_bytes = activation_shell_bytes / 2;
    if matches!(
        quant.saved_activations.mode,
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp
    ) {
        workspace_bytes = workspace_bytes.saturating_add(activation_shell_bytes / 4);
    }

    LowBitMemoryBucketEstimate {
        master_weight_bytes,
        execution_weight_bytes,
        activation_shell_bytes,
        saved_activation_bytes,
        rho_state_bytes,
        workspace_bytes,
    }
}

pub fn build_low_bit_saved_activation_inventory(
    quant: &LowBitQuantizationConfig,
    input: LowBitMemoryEstimateInput,
) -> Option<LowBitSavedActivationInventory> {
    if !quant.enable
        || matches!(
            quant.saved_activations.mode,
            LowBitSavedActivationMode::Disabled
        )
    {
        return None;
    }

    let plan = LowBitProjectionPlan::from_config(quant);
    let latent_per_head = input.latent_total / input.n_head.max(1);
    let (bytes_num, bytes_den) = saved_activation_bytes_per_element(quant.saved_activations.format);
    let recompute_policy = match quant.saved_activations.mode {
        LowBitSavedActivationMode::Disabled => LowBitSavedActivationRecomputePolicy::Disabled,
        LowBitSavedActivationMode::QuantizedCacheExp => {
            LowBitSavedActivationRecomputePolicy::QuantizedCacheOnly
        }
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp => {
            LowBitSavedActivationRecomputePolicy::RecomputeSafeWithinWindow
        }
    };

    let mut tensors = Vec::new();
    let mut push_tensor = |name: &str, shape: Vec<usize>| {
        let element_count = shape
            .iter()
            .fold(1u64, |acc, dim| acc.saturating_mul(*dim as u64));
        let estimated_bytes = estimate_packed_tensor_bytes(element_count, bytes_num, bytes_den);
        tensors.push(LowBitSavedActivationTensorInventoryEntry {
            name: name.to_string(),
            shape,
            element_count,
            estimated_bytes,
            recompute_policy,
        });
    };

    if plan.x_weight_format.is_some() {
        push_tensor(
            "x_projection_input",
            vec![
                input.batch_size,
                input.n_head,
                input.time_steps,
                input.n_embd,
            ],
        );
    }
    if plan.y_weight_format.is_some() {
        push_tensor(
            "y_projection_input",
            vec![
                input.batch_size,
                input.n_head,
                input.time_steps,
                input.n_embd,
            ],
        );
    }
    if plan.residual_weight_format.is_some() {
        push_tensor(
            "residual_tail_input",
            vec![
                input.batch_size,
                input.n_head,
                input.time_steps,
                latent_per_head,
            ],
        );
    }

    Some(LowBitSavedActivationInventory {
        mode: quant.saved_activations.mode,
        format: quant.saved_activations.format,
        requires_rho_window_anchor: matches!(
            quant.saved_activations.mode,
            LowBitSavedActivationMode::QuantizedCacheRecomputeExp
        ),
        tensors,
    })
}

pub fn rho_compression_profile_reset() {
    if let Ok(mut state) = RHO_COMPRESSION_PROFILE
        .get_or_init(|| Mutex::new(RhoCompressionStatsState::default()))
        .lock()
    {
        *state = RhoCompressionStatsState::default();
    }
}

pub fn rho_compression_profile_snapshot() -> RhoCompressionStatsSnapshot {
    if let Ok(state) = RHO_COMPRESSION_PROFILE
        .get_or_init(|| Mutex::new(RhoCompressionStatsState::default()))
        .lock()
    {
        return RhoCompressionStatsSnapshot {
            calls: state.calls,
            total_original_rms: state.total_original_rms,
            total_reconstructed_rms: state.total_reconstructed_rms,
            total_mean_abs_error: state.total_mean_abs_error,
            max_abs_error: state.max_abs_error,
            total_original_bytes: state.total_original_bytes,
            total_compressed_bytes: state.total_compressed_bytes,
        };
    }
    RhoCompressionStatsSnapshot::default()
}

pub fn rho_compression_snapshot_passes_gate(
    snapshot: &RhoCompressionStatsSnapshot,
    gate: &RhoCompressionQualityGate,
) -> bool {
    let Some(mean_abs_error) = snapshot.mean_abs_error() else {
        return false;
    };
    let Some(mean_rms_ratio) = snapshot.mean_rms_ratio() else {
        return false;
    };
    let Some(compression_ratio) = snapshot.compression_ratio() else {
        return false;
    };

    mean_abs_error <= gate.max_mean_abs_error
        && snapshot.max_abs_error <= gate.max_max_abs_error
        && mean_rms_ratio >= gate.min_mean_rms_ratio
        && mean_rms_ratio <= gate.max_mean_rms_ratio
        && compression_ratio <= gate.max_compression_ratio
}

fn total_rho_elements(logical_shape: [usize; 4]) -> usize {
    logical_shape.iter().product()
}

fn pack_rho_block_values(
    values: &[f32],
    compression: RhoCompressionConfig,
) -> (PackedRhoBlockEncoding, Vec<f32>, Vec<u8>, Vec<f32>) {
    let encoding = PackedRhoBlockEncoding::from_compression(compression)
        .expect("rho block packing requires a block compression mode");
    let mut packed = Vec::new();
    let mut scales = Vec::with_capacity(values.len().div_ceil(RHO_BLOCK_SIZE));
    let mut reconstructed = Vec::with_capacity(values.len());

    for block in values.chunks(RHO_BLOCK_SIZE) {
        match encoding {
            PackedRhoBlockEncoding::Int8 => {
                let max_abs = block.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
                let scale = (max_abs / 127.0).max(QUANT_EPSILON);
                scales.push(scale);
                for value in block {
                    let quantized = (value / scale).round().clamp(-127.0, 127.0) as i8;
                    packed.push(quantized as u8);
                    reconstructed.push(quantized as f32 * scale);
                }
            }
            PackedRhoBlockEncoding::Ternary2 => {
                let quantized = quantize_ternary_absmean(block);
                scales.push(quantized.scale);
                reconstructed.extend(
                    quantized
                        .values
                        .iter()
                        .map(|value| *value as f32 * quantized.scale),
                );
                packed.extend(pack_ternary_2bit(&quantized.values).packed);
            }
            PackedRhoBlockEncoding::Binary1 => {
                let quantized = quantize_binary_sign(block);
                scales.push(quantized.scale);
                reconstructed.extend(
                    quantized
                        .values
                        .iter()
                        .map(|value| *value as f32 * quantized.scale),
                );
                packed.extend(pack_binary_1bit(&quantized.values));
            }
        }
    }

    (encoding, scales, packed, reconstructed)
}

fn unpack_rho_block_values(packed_state: &PackedRhoBlockState) -> Vec<f32> {
    let total_elements = total_rho_elements(packed_state.logical_shape);
    let mut values = Vec::with_capacity(total_elements);
    let mut packed_offset = 0usize;

    for (block_index, scale) in packed_state.scales.iter().copied().enumerate() {
        let remaining = total_elements.saturating_sub(block_index * packed_state.block_size);
        if remaining == 0 {
            break;
        }
        let block_len = remaining.min(packed_state.block_size);
        match packed_state.encoding {
            PackedRhoBlockEncoding::Int8 => {
                let next_offset = packed_offset + block_len;
                for value in &packed_state.packed[packed_offset..next_offset] {
                    values.push((*value as i8) as f32 * scale);
                }
                packed_offset = next_offset;
            }
            PackedRhoBlockEncoding::Ternary2 => {
                let byte_len = block_len.div_ceil(4);
                let next_offset = packed_offset + byte_len;
                let ternary = unpack_ternary_2bit(&PackedTernaryBuffer {
                    packed: packed_state.packed[packed_offset..next_offset].to_vec(),
                    len: block_len,
                });
                values.extend(ternary.into_iter().map(|value| value as f32 * scale));
                packed_offset = next_offset;
            }
            PackedRhoBlockEncoding::Binary1 => {
                let byte_len = block_len.div_ceil(8);
                let next_offset = packed_offset + byte_len;
                let binary =
                    unpack_binary_1bit(&packed_state.packed[packed_offset..next_offset], block_len);
                values.extend(binary.into_iter().map(|value| value as f32 * scale));
                packed_offset = next_offset;
            }
        }
    }

    values
}

/// Pack rho into a host-side block-compressed representation using the requested encoding.
pub fn pack_rho_block_state<B: Backend>(
    rho: &Tensor<B, 4>,
    compression: RhoCompressionConfig,
) -> PackedRhoBlockState {
    let logical_shape = rho.shape().dims::<4>();
    let values = rho
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho values");
    let (encoding, scales, packed, reconstructed) = pack_rho_block_values(&values, compression);

    let compressed_bytes =
        packed.len() as u64 + (scales.len() * core::mem::size_of::<f32>()) as u64;
    record_rho_compression_profile(&values, &reconstructed, compressed_bytes);

    PackedRhoBlockState {
        logical_shape,
        block_size: RHO_BLOCK_SIZE,
        encoding,
        scales,
        packed,
    }
}

pub fn pack_rho_int8_block_state_device<B: Backend>(
    rho: &Tensor<B, 4>,
) -> PackedRhoInt8DeviceState<B> {
    let logical_shape = rho.shape().dims::<4>();
    let compressed = pack_rho_int8_block_device_reference(rho.clone(), RHO_BLOCK_SIZE);
    let packed_values = compressed
        .packed
        .clone()
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("packed rho values");
    let scale_values = compressed
        .scales
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho scales");
    let reconstructed = unpack_rho_int8_block_device_reference(
        compressed.packed.clone(),
        compressed.scales.clone(),
        logical_shape,
        RHO_BLOCK_SIZE,
    )
    .into_data()
    .convert::<f32>()
    .into_vec::<f32>()
    .expect("reconstructed rho");
    let original = rho
        .clone()
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("rho values");
    let compressed_bytes =
        packed_values.len() as u64 + (scale_values.len() * core::mem::size_of::<f32>()) as u64;
    record_rho_compression_profile(&original, &reconstructed, compressed_bytes);
    PackedRhoInt8DeviceState {
        logical_shape,
        block_size: RHO_BLOCK_SIZE,
        scales: compressed.scales,
        packed: compressed.packed,
    }
}

/// Unpack a host-side block-compressed rho state back into a dense tensor.
pub fn unpack_rho_block_state<B: Backend>(
    packed_state: &PackedRhoBlockState,
    device: &B::Device,
) -> Tensor<B, 4> {
    let values = unpack_rho_block_values(packed_state);
    Tensor::<B, 4>::from_data(TensorData::new(values, packed_state.logical_shape), device)
}

pub fn unpack_rho_int8_block_state_device<B: Backend>(
    packed_state: &PackedRhoInt8DeviceState<B>,
) -> Tensor<B, 4> {
    unpack_rho_int8_block_device_reference(
        packed_state.packed.clone(),
        packed_state.scales.clone(),
        packed_state.logical_shape,
        packed_state.block_size,
    )
}

fn record_rho_compression_profile(original: &[f32], reconstructed: &[f32], compressed_bytes: u64) {
    if original.is_empty() || original.len() != reconstructed.len() {
        return;
    }

    let original_rms = (original
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        / original.len() as f64)
        .sqrt();
    let reconstructed_rms = (reconstructed
        .iter()
        .map(|value| {
            let value = f64::from(*value);
            value * value
        })
        .sum::<f64>()
        / reconstructed.len() as f64)
        .sqrt();
    let mean_abs_error = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| f64::from((lhs - rhs).abs()))
        .sum::<f64>()
        / original.len() as f64;
    let max_abs_error = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(lhs, rhs)| (lhs - rhs).abs())
        .fold(0.0f32, f32::max);

    if let Ok(mut state) = RHO_COMPRESSION_PROFILE
        .get_or_init(|| Mutex::new(RhoCompressionStatsState::default()))
        .lock()
    {
        state.calls = state.calls.saturating_add(1);
        state.total_original_rms += original_rms;
        state.total_reconstructed_rms += reconstructed_rms;
        state.total_mean_abs_error += mean_abs_error;
        state.max_abs_error = state.max_abs_error.max(max_abs_error);
        state.total_original_bytes = state
            .total_original_bytes
            .saturating_add((original.len() * core::mem::size_of::<f32>()) as u64);
        state.total_compressed_bytes = state
            .total_compressed_bytes
            .saturating_add(compressed_bytes);
    }
}

enum ReferenceActivationBuffer {
    Float(Vec<f32>),
    Signed { values: Vec<i8>, scale: f32 },
    UnsignedPositive { values: Vec<u8>, scale: f32 },
}

impl ReferenceActivationBuffer {
    fn estimated_bytes(&self) -> u64 {
        match self {
            Self::Float(values) => (values.len() * core::mem::size_of::<f32>()) as u64,
            Self::Signed { values, .. } => values.len() as u64 + core::mem::size_of::<f32>() as u64,
            Self::UnsignedPositive { values, .. } => {
                values.len() as u64 + core::mem::size_of::<f32>() as u64
            }
        }
    }

    fn into_packed_saved_activation(
        self,
        logical_shape: Vec<usize>,
        format: LowBitActivationFormat,
    ) -> PackedSavedActivationState {
        let estimated_bytes = self.estimated_bytes();
        let buffer = match self {
            Self::Float(values) => PackedSavedActivationBuffer::Float(values),
            Self::Signed { values, scale } => PackedSavedActivationBuffer::Signed { values, scale },
            Self::UnsignedPositive { values, scale } => {
                PackedSavedActivationBuffer::UnsignedPositive { values, scale }
            }
        };
        PackedSavedActivationState {
            logical_shape,
            format,
            estimated_bytes,
            buffer,
        }
    }
}

pub fn pack_saved_activation_state<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    format: LowBitActivationFormat,
) -> PackedSavedActivationState {
    let logical_shape = tensor.shape().dims::<D>().into_iter().collect::<Vec<_>>();
    let values = tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("saved activation values");
    quantize_reference_activations(&values, Some(format))
        .into_packed_saved_activation(logical_shape, format)
}

pub fn unpack_saved_activation_state<B: Backend, const D: usize>(
    packed: &PackedSavedActivationState,
    device: &B::Device,
) -> Tensor<B, D> {
    assert_eq!(
        packed.logical_shape.len(),
        D,
        "saved activation rank mismatch: state={} requested={}",
        packed.logical_shape.len(),
        D
    );
    let shape: [usize; D] = packed
        .logical_shape
        .clone()
        .try_into()
        .expect("saved activation logical shape");
    let values = match &packed.buffer {
        PackedSavedActivationBuffer::Float(values) => values.clone(),
        PackedSavedActivationBuffer::Signed { values, scale } => {
            values.iter().map(|value| *value as f32 * *scale).collect()
        }
        PackedSavedActivationBuffer::UnsignedPositive { values, scale } => {
            values.iter().map(|value| *value as f32 * *scale).collect()
        }
    };
    Tensor::<B, D>::from_data(TensorData::new(values, shape), device)
}

fn quantize_activation_codes_tensor<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    format: Option<LowBitActivationFormat>,
    scale_cache_key: Option<&ActivationScaleCacheKey>,
) -> Option<(Tensor<B, D, Int>, f32, bool)> {
    let cached_scale = if low_bit_inference_cached_scale_enabled_for_backend::<B>() {
        scale_cache_key
            .and_then(|key| ACTIVATION_SCALE_CACHE.with(|cache| cache.borrow().get(key).copied()))
    } else {
        None
    };
    quantize_activation_codes_tensor_with_cached_scale(
        tensor,
        format,
        cached_scale,
        |scale, quant_format| {
            if low_bit_inference_cached_scale_enabled_for_backend::<B>() {
                if let Some(key) = scale_cache_key {
                    ACTIVATION_SCALE_CACHE.with(|cache| {
                        cache
                            .borrow_mut()
                            .insert(key.clone(), scale.max(QUANT_EPSILON));
                    });
                }
            }
            let _ = quant_format;
        },
    )
}

fn quantize_activation_codes_tensor_with_cached_scale<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    format: Option<LowBitActivationFormat>,
    cached_scale: Option<f32>,
    mut store_scale: impl FnMut(f32, LowBitActivationFormat),
) -> Option<(Tensor<B, D, Int>, f32, bool)> {
    let quant_format = format?;
    match quant_format {
        LowBitActivationFormat::Fp16 => None,
        LowBitActivationFormat::Int8 | LowBitActivationFormat::Int4Exp => {
            let qmax = if matches!(quant_format, LowBitActivationFormat::Int8) {
                127.0
            } else {
                7.0
            };
            let (scale, used_cached_scale) = if let Some(scale) = cached_scale {
                (scale.max(QUANT_EPSILON), true)
            } else {
                let dynamic_range = tensor
                    .clone()
                    .abs()
                    .mean()
                    .mul_scalar(2.0)
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                let scale = (dynamic_range / qmax).max(QUANT_EPSILON);
                store_scale(scale, quant_format);
                (scale, false)
            };
            let codes = round_nearest(tensor.clone().div_scalar(scale))
                .clamp_min(-qmax)
                .clamp_max(qmax)
                .int();
            Some((codes, scale, used_cached_scale))
        }
        LowBitActivationFormat::Uint8PosExp => {
            let qmax = 255.0;
            let clamped = tensor.clone().clamp_min(0.0);
            let (scale, used_cached_scale) = if let Some(scale) = cached_scale {
                (scale.max(QUANT_EPSILON), true)
            } else {
                let dynamic_range = clamped
                    .clone()
                    .mean()
                    .mul_scalar(2.0)
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                let scale = (dynamic_range / qmax).max(QUANT_EPSILON);
                store_scale(scale, quant_format);
                (scale, false)
            };
            let codes = round_nearest(clamped.div_scalar(scale))
                .clamp_min(0.0)
                .clamp_max(qmax)
                .int();
            Some((codes, scale, used_cached_scale))
        }
    }
}

fn training_activation_scale_cache_key<B: Backend>(
    kind: &'static str,
    format: LowBitActivationFormat,
    shape: [usize; 4],
) -> TrainingActivationScaleCacheKey {
    TrainingActivationScaleCacheKey {
        backend_name: std::any::type_name::<B>(),
        kind,
        format,
        shape,
    }
}

fn training_weight_scale_cache_key<B: Backend, const D: usize>(
    kind: &'static str,
    format: LowBitWeightFormat,
    shape: [usize; D],
) -> TrainingWeightScaleCacheKey {
    TrainingWeightScaleCacheKey {
        backend_name: std::any::type_name::<B>(),
        kind,
        format,
        shape: shape.into_iter().collect(),
    }
}

fn quantize_training_activation_codes_tensor_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
    cache_kind: Option<&'static str>,
) -> Option<(Tensor<B, 4, Int>, f32, bool)> {
    let quant_format = format?;
    let cache_key = if low_bit_training_cached_scale_enabled() {
        cache_kind.map(|kind| {
            training_activation_scale_cache_key::<B>(kind, quant_format, tensor.shape().dims::<4>())
        })
    } else {
        None
    };
    let cached_scale = cache_key.as_ref().and_then(|key| {
        TRAINING_ACTIVATION_SCALE_CACHE.with(|cache| cache.borrow().get(key).copied())
    });
    quantize_activation_codes_tensor_with_cached_scale(
        tensor,
        Some(quant_format),
        cached_scale,
        |scale, _| {
            if let Some(key) = cache_key.as_ref() {
                TRAINING_ACTIVATION_SCALE_CACHE.with(|cache| {
                    cache
                        .borrow_mut()
                        .insert(key.clone(), scale.max(QUANT_EPSILON));
                });
            }
        },
    )
}

fn activation_scale_tensor_device_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
) -> Option<(LowBitActivationFormat, Tensor<B, 1>)> {
    let quant_format = format?;
    match quant_format {
        LowBitActivationFormat::Fp16 => None,
        LowBitActivationFormat::Int8 | LowBitActivationFormat::Int4Exp => {
            let qmax = if matches!(quant_format, LowBitActivationFormat::Int8) {
                127.0
            } else {
                7.0
            };
            let scale = tensor
                .clone()
                .abs()
                .mean()
                .mul_scalar(2.0 / qmax)
                .clamp_min(QUANT_EPSILON)
                .reshape([1]);
            Some((quant_format, scale))
        }
        LowBitActivationFormat::Uint8PosExp => {
            let qmax = 255.0;
            let clamped = tensor.clone().clamp_min(0.0);
            let scale = clamped
                .clone()
                .mean()
                .mul_scalar(2.0 / qmax)
                .clamp_min(QUANT_EPSILON)
                .reshape([1]);
            Some((quant_format, scale))
        }
    }
}

fn quantize_activation_codes_tensor_device_scale_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
) -> Option<(Tensor<B, 4, Int>, Tensor<B, 1>)> {
    let (quant_format, scale) = activation_scale_tensor_device_4d(tensor, format)?;
    let wgpu_codes = match quant_format {
        LowBitActivationFormat::Int8 => {
            try_wgpu_quantize_activation_codes_i32(tensor, &scale, 127, false)
        }
        LowBitActivationFormat::Int4Exp => {
            try_wgpu_quantize_activation_codes_i32(tensor, &scale, 7, false)
        }
        LowBitActivationFormat::Uint8PosExp => {
            try_wgpu_quantize_activation_codes_i32(tensor, &scale, 255, true)
        }
        LowBitActivationFormat::Fp16 => None,
    };
    if let Some(codes) = wgpu_codes {
        return Some((codes, scale));
    }
    let codes = match quant_format {
        LowBitActivationFormat::Int8 => {
            round_nearest(tensor.clone().div(scale.clone().reshape([1, 1, 1, 1])))
                .clamp_min(-127.0)
                .clamp_max(127.0)
                .int()
        }
        LowBitActivationFormat::Int4Exp => {
            round_nearest(tensor.clone().div(scale.clone().reshape([1, 1, 1, 1])))
                .clamp_min(-7.0)
                .clamp_max(7.0)
                .int()
        }
        LowBitActivationFormat::Uint8PosExp => round_nearest(
            tensor
                .clone()
                .clamp_min(0.0)
                .div(scale.clone().reshape([1, 1, 1, 1])),
        )
        .clamp_min(0.0)
        .clamp_max(255.0)
        .int(),
        LowBitActivationFormat::Fp16 => return None,
    };
    Some((codes, scale))
}

fn activation_scale_tensor_cached_device_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
    scale_cache_key: Option<&ActivationScaleCacheKey>,
) -> Option<(LowBitActivationFormat, Tensor<B, 1>, bool)> {
    let quant_format = format?;
    if matches!(
        quant_format,
        LowBitActivationFormat::Fp16 | LowBitActivationFormat::Uint8PosExp
    ) {
        return None;
    }
    let cached_scale = if low_bit_inference_cached_scale_enabled_for_backend::<B>() {
        scale_cache_key
            .and_then(|key| ACTIVATION_SCALE_CACHE.with(|cache| cache.borrow().get(key).copied()))
    } else {
        None
    };
    if let Some(scale) = cached_scale {
        return Some((
            quant_format,
            Tensor::<B, 1>::from_data(
                TensorData::new(vec![scale.max(QUANT_EPSILON)], [1]),
                &tensor.device(),
            ),
            true,
        ));
    }

    let (_, scale) = activation_scale_tensor_device_4d(tensor, Some(quant_format))?;
    if low_bit_inference_cached_scale_enabled_for_backend::<B>() {
        if let Some(key) = scale_cache_key {
            let scale_value = scale.clone().into_scalar().elem::<f32>().max(QUANT_EPSILON);
            ACTIVATION_SCALE_CACHE.with(|cache| {
                cache.borrow_mut().insert(key.clone(), scale_value);
            });
        }
    }
    Some((quant_format, scale, false))
}

fn quantize_activation_packed_codes_tensor_device_scale_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
    scale_cache_key: Option<&ActivationScaleCacheKey>,
) -> Option<(Tensor<B, 4, Int>, Tensor<B, 1>, bool)> {
    let (quant_format, scale, used_cached_scale) =
        activation_scale_tensor_cached_device_4d(tensor, format, scale_cache_key)?;
    let (qmax, positive_only) = match quant_format {
        LowBitActivationFormat::Int8 => (127, false),
        LowBitActivationFormat::Int4Exp => (7, false),
        LowBitActivationFormat::Fp16 | LowBitActivationFormat::Uint8PosExp => return None,
    };
    let packed = try_raw_cuda_quantize_pack_activation_i8x4(tensor, &scale, qmax, positive_only)
        .or_else(|| try_wgpu_quantize_pack_activation_i8x4(tensor, &scale, qmax, positive_only))?;
    Some((packed, scale, used_cached_scale))
}

fn quantize_activation_packed_codes_tensor_4d<B: Backend>(
    tensor: &Tensor<B, 4>,
    format: Option<LowBitActivationFormat>,
    pack_last_dim: impl FnOnce(&[i8], [usize; 4]) -> Vec<i32>,
) -> Option<(Tensor<B, 4, Int>, f32)> {
    let quant_format = format?;
    if matches!(quant_format, LowBitActivationFormat::Fp16) {
        return None;
    }
    let shape = tensor.shape().dims::<4>();
    let values = tensor
        .clone()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("activation values");
    let quantized = quantize_reference_activations(&values, Some(quant_format));
    match quantized {
        ReferenceActivationBuffer::Float(_) => None,
        ReferenceActivationBuffer::Signed { values, scale } => {
            let signed = values.iter().map(|value| *value as i8).collect::<Vec<_>>();
            let packed = pack_last_dim(&signed, shape);
            Some((
                Tensor::<B, 4, Int>::from_data(
                    TensorData::new(
                        packed.into_iter().map(i64::from).collect::<Vec<_>>(),
                        [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                    ),
                    &tensor.device(),
                ),
                scale,
            ))
        }
        ReferenceActivationBuffer::UnsignedPositive { values, scale } => {
            let signed = values.iter().map(|value| *value as i8).collect::<Vec<_>>();
            let packed = pack_last_dim(&signed, shape);
            Some((
                Tensor::<B, 4, Int>::from_data(
                    TensorData::new(
                        packed.into_iter().map(i64::from).collect::<Vec<_>>(),
                        [shape[0], shape[1], shape[2], shape[3].div_ceil(4)],
                    ),
                    &tensor.device(),
                ),
                scale,
            ))
        }
    }
}

pub(crate) fn packed_lowrank_projection_reference<B: Backend>(
    input: Tensor<B, 4>,
    artifact: &PackedWeightArtifact,
    activation_format: Option<LowBitActivationFormat>,
    latent_out: usize,
) -> Tensor<B, 4> {
    let device = input.device();
    let [batch, input_heads, time, embd] = input.shape().dims::<4>();
    let [artifact_heads, artifact_embd, artifact_latent] =
        logical_shape_3d(artifact, "packed low-rank projection");
    assert!(
        input_heads == 1 || input_heads == artifact_heads,
        "packed low-rank projection head mismatch: artifact={} input={}",
        artifact_heads,
        input_heads
    );
    assert_eq!(
        artifact_embd, embd,
        "packed low-rank projection embd mismatch: artifact={} input={}",
        artifact_embd, embd
    );
    assert!(
        latent_out <= artifact_latent,
        "packed low-rank projection latent mismatch: requested {} > artifact {}",
        latent_out,
        artifact_latent
    );

    let input_values = input
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("packed projection input values");
    let weight_codes = unpack_weight_artifact_to_i8_codes(artifact);
    let weight_scale = artifact.scale.max(QUANT_EPSILON);
    let mut output = vec![0.0f32; batch * artifact_heads * time * latent_out];

    match quantize_reference_activations(&input_values, activation_format) {
        ReferenceActivationBuffer::Float(values) => {
            for batch_idx in 0..batch {
                for head_idx in 0..artifact_heads {
                    let input_head_idx = if input_heads == 1 { 0 } else { head_idx };
                    for time_idx in 0..time {
                        let input_base =
                            ((batch_idx * input_heads + input_head_idx) * time + time_idx) * embd;
                        for latent_idx in 0..latent_out {
                            let mut acc = 0.0f32;
                            for embd_idx in 0..embd {
                                let weight_index =
                                    ((head_idx * embd + embd_idx) * artifact_latent) + latent_idx;
                                acc += (weight_codes[weight_index] as f32 * weight_scale)
                                    * values[input_base + embd_idx];
                            }
                            let output_index = (((batch_idx * artifact_heads + head_idx) * time
                                + time_idx)
                                * latent_out)
                                + latent_idx;
                            output[output_index] = acc;
                        }
                    }
                }
            }
        }
        ReferenceActivationBuffer::Signed { values, scale } => {
            for batch_idx in 0..batch {
                for head_idx in 0..artifact_heads {
                    let input_head_idx = if input_heads == 1 { 0 } else { head_idx };
                    for time_idx in 0..time {
                        let input_base =
                            ((batch_idx * input_heads + input_head_idx) * time + time_idx) * embd;
                        for latent_idx in 0..latent_out {
                            let mut acc = 0i32;
                            for embd_idx in 0..embd {
                                let weight_index =
                                    ((head_idx * embd + embd_idx) * artifact_latent) + latent_idx;
                                acc += (weight_codes[weight_index] as i32)
                                    * (values[input_base + embd_idx] as i32);
                            }
                            let output_index = (((batch_idx * artifact_heads + head_idx) * time
                                + time_idx)
                                * latent_out)
                                + latent_idx;
                            output[output_index] = acc as f32 * weight_scale * scale;
                        }
                    }
                }
            }
        }
        ReferenceActivationBuffer::UnsignedPositive { values, scale } => {
            for batch_idx in 0..batch {
                for head_idx in 0..artifact_heads {
                    let input_head_idx = if input_heads == 1 { 0 } else { head_idx };
                    for time_idx in 0..time {
                        let input_base =
                            ((batch_idx * input_heads + input_head_idx) * time + time_idx) * embd;
                        for latent_idx in 0..latent_out {
                            let mut acc = 0i32;
                            for embd_idx in 0..embd {
                                let weight_index =
                                    ((head_idx * embd + embd_idx) * artifact_latent) + latent_idx;
                                acc += (weight_codes[weight_index] as i32)
                                    * (values[input_base + embd_idx] as i32);
                            }
                            let output_index = (((batch_idx * artifact_heads + head_idx) * time
                                + time_idx)
                                * latent_out)
                                + latent_idx;
                            output[output_index] = acc as f32 * weight_scale * scale;
                        }
                    }
                }
            }
        }
    }

    Tensor::<B, 4>::from_data(
        TensorData::new(output, [batch, artifact_heads, time, latent_out]),
        &device,
    )
}

pub(crate) fn packed_decoder_tail_reference<B: Backend>(
    y_neuron: Tensor<B, 4>,
    artifact: &PackedWeightArtifact,
    activation_format: Option<LowBitActivationFormat>,
) -> Tensor<B, 4> {
    let device = y_neuron.device();
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let [artifact_latent_total, dim] = logical_shape_2d(artifact, "packed decoder tail");
    assert_eq!(
        artifact_latent_total % heads,
        0,
        "packed decoder tail latent_total must divide across heads"
    );
    let artifact_latent_per_head = artifact_latent_total / heads;
    assert!(
        latent <= artifact_latent_per_head,
        "packed decoder tail latent mismatch: requested {} > artifact {}",
        latent,
        artifact_latent_per_head
    );

    let y_values = y_neuron
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("packed decoder tail values");
    let weight_codes = unpack_weight_artifact_to_i8_codes(artifact);
    let weight_scale = artifact.scale.max(QUANT_EPSILON);
    let mut output = vec![0.0f32; batch * time * dim];

    match quantize_reference_activations(&y_values, activation_format) {
        ReferenceActivationBuffer::Float(values) => {
            for batch_idx in 0..batch {
                for time_idx in 0..time {
                    for dim_idx in 0..dim {
                        let mut acc = 0.0f32;
                        for head_idx in 0..heads {
                            let input_base =
                                ((batch_idx * heads + head_idx) * time + time_idx) * latent;
                            let weight_base = (head_idx * artifact_latent_per_head) * dim;
                            for latent_idx in 0..latent {
                                let weight_index = weight_base + latent_idx * dim + dim_idx;
                                acc += (weight_codes[weight_index] as f32 * weight_scale)
                                    * values[input_base + latent_idx];
                            }
                        }
                        output[(batch_idx * time + time_idx) * dim + dim_idx] = acc;
                    }
                }
            }
        }
        ReferenceActivationBuffer::Signed { values, scale } => {
            for batch_idx in 0..batch {
                for time_idx in 0..time {
                    for dim_idx in 0..dim {
                        let mut acc = 0i32;
                        for head_idx in 0..heads {
                            let input_base =
                                ((batch_idx * heads + head_idx) * time + time_idx) * latent;
                            let weight_base = (head_idx * artifact_latent_per_head) * dim;
                            for latent_idx in 0..latent {
                                let weight_index = weight_base + latent_idx * dim + dim_idx;
                                acc += (weight_codes[weight_index] as i32)
                                    * (values[input_base + latent_idx] as i32);
                            }
                        }
                        output[(batch_idx * time + time_idx) * dim + dim_idx] =
                            acc as f32 * weight_scale * scale;
                    }
                }
            }
        }
        ReferenceActivationBuffer::UnsignedPositive { values, scale } => {
            for batch_idx in 0..batch {
                for time_idx in 0..time {
                    for dim_idx in 0..dim {
                        let mut acc = 0i32;
                        for head_idx in 0..heads {
                            let input_base =
                                ((batch_idx * heads + head_idx) * time + time_idx) * latent;
                            let weight_base = (head_idx * artifact_latent_per_head) * dim;
                            for latent_idx in 0..latent {
                                let weight_index = weight_base + latent_idx * dim + dim_idx;
                                acc += (weight_codes[weight_index] as i32)
                                    * (values[input_base + latent_idx] as i32);
                            }
                        }
                        output[(batch_idx * time + time_idx) * dim + dim_idx] =
                            acc as f32 * weight_scale * scale;
                    }
                }
            }
        }
    }

    Tensor::<B, 4>::from_data(TensorData::new(output, [batch, 1, time, dim]), &device)
}

fn logical_shape_3d(artifact: &PackedWeightArtifact, context: &str) -> [usize; 3] {
    artifact
        .logical_shape
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("{context} expects rank-3 logical shape"))
}

fn logical_shape_2d(artifact: &PackedWeightArtifact, context: &str) -> [usize; 2] {
    artifact
        .logical_shape
        .clone()
        .try_into()
        .unwrap_or_else(|_| panic!("{context} expects rank-2 logical shape"))
}

fn weight_codes_tensor_from_float_values<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    format: LowBitWeightFormat,
    cache_kind: Option<&'static str>,
) -> Option<(Tensor<B, D, Int>, f32, bool)> {
    let cache_key = if low_bit_training_cached_scale_enabled() {
        cache_kind.map(|kind| {
            training_weight_scale_cache_key::<B, D>(kind, format, tensor.shape().dims())
        })
    } else {
        None
    };
    let cached_scale = cache_key
        .as_ref()
        .and_then(|key| TRAINING_WEIGHT_SCALE_CACHE.with(|cache| cache.borrow().get(key).copied()));
    match format {
        LowBitWeightFormat::Fp16 => None,
        LowBitWeightFormat::Int8 => {
            let qmax = 127.0;
            let (scale, used_cached_scale) = if let Some(scale) = cached_scale {
                (scale.max(QUANT_EPSILON), true)
            } else {
                let dynamic_range = tensor
                    .clone()
                    .abs()
                    .mean()
                    .mul_scalar(2.0)
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                let scale = (dynamic_range / qmax).max(QUANT_EPSILON);
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale);
                    });
                }
                (scale, false)
            };
            let codes = round_nearest(tensor.clone().div_scalar(scale))
                .clamp_min(-qmax)
                .clamp_max(qmax)
                .int();
            Some((codes, scale, used_cached_scale))
        }
        LowBitWeightFormat::Sign1 => {
            let (scale, used_cached_scale) = if let Some(scale) = cached_scale {
                (scale.max(QUANT_EPSILON), true)
            } else {
                let scale = tensor
                    .clone()
                    .abs()
                    .mean()
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale);
                    });
                }
                (scale, false)
            };
            let codes = tensor
                .clone()
                .greater_equal_elem(0.0)
                .float()
                .mul_scalar(2.0)
                .sub_scalar(1.0)
                .int();
            Some((codes, scale, used_cached_scale))
        }
        LowBitWeightFormat::Ternary158 | LowBitWeightFormat::Packed2 => {
            let (scale, used_cached_scale) = if let Some(scale) = cached_scale {
                (scale.max(QUANT_EPSILON), true)
            } else {
                let scale = tensor
                    .clone()
                    .abs()
                    .mean()
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale);
                    });
                }
                (scale, false)
            };
            let active = tensor.clone().abs().greater_equal_elem(scale).float();
            let sign = tensor
                .clone()
                .greater_equal_elem(0.0)
                .float()
                .mul_scalar(2.0)
                .sub_scalar(1.0);
            Some((sign.mul(active).int(), scale, used_cached_scale))
        }
    }
}

fn weight_codes_tensor_device_scale_from_float_values<B: Backend, const D: usize>(
    tensor: &Tensor<B, D>,
    format: LowBitWeightFormat,
    cache_kind: Option<&'static str>,
) -> Option<(Tensor<B, D, Int>, Tensor<B, 1>, f32, bool)> {
    let cache_key = if low_bit_training_cached_scale_enabled() {
        cache_kind.map(|kind| {
            training_weight_scale_cache_key::<B, D>(kind, format, tensor.shape().dims())
        })
    } else {
        None
    };
    let cached_scale = cache_key
        .as_ref()
        .and_then(|key| TRAINING_WEIGHT_SCALE_CACHE.with(|cache| cache.borrow().get(key).copied()));
    match format {
        LowBitWeightFormat::Fp16 => None,
        LowBitWeightFormat::Int8 => {
            let qmax = 127.0;
            let (scale, scale_scalar, used_cached_scale) = if let Some(scale) = cached_scale {
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale.max(QUANT_EPSILON)], [1]),
                        &tensor.device(),
                    ),
                    scale.max(QUANT_EPSILON),
                    true,
                )
            } else {
                let scale_scalar = tensor
                    .clone()
                    .abs()
                    .mean()
                    .mul_scalar(2.0 / qmax)
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale_scalar);
                    });
                }
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale_scalar], [1]),
                        &tensor.device(),
                    ),
                    scale_scalar,
                    false,
                )
            };
            let codes = round_nearest(tensor.clone().div(scale.clone().reshape([1; D])))
                .clamp_min(-qmax)
                .clamp_max(qmax)
                .int();
            Some((codes, scale, scale_scalar, used_cached_scale))
        }
        LowBitWeightFormat::Sign1 => {
            let (scale, scale_scalar, used_cached_scale) = if let Some(scale) = cached_scale {
                let scale_scalar = scale.max(QUANT_EPSILON);
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale_scalar], [1]),
                        &tensor.device(),
                    ),
                    scale_scalar,
                    true,
                )
            } else {
                let scale_scalar = tensor
                    .clone()
                    .abs()
                    .mean()
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale_scalar);
                    });
                }
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale_scalar], [1]),
                        &tensor.device(),
                    ),
                    scale_scalar,
                    false,
                )
            };
            let codes = tensor
                .clone()
                .greater_equal_elem(0.0)
                .float()
                .mul_scalar(2.0)
                .sub_scalar(1.0)
                .int();
            Some((codes, scale, scale_scalar, used_cached_scale))
        }
        LowBitWeightFormat::Ternary158 | LowBitWeightFormat::Packed2 => {
            let (scale, scale_scalar, used_cached_scale) = if let Some(scale) = cached_scale {
                let scale_scalar = scale.max(QUANT_EPSILON);
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale_scalar], [1]),
                        &tensor.device(),
                    ),
                    scale_scalar,
                    true,
                )
            } else {
                let scale_scalar = tensor
                    .clone()
                    .abs()
                    .mean()
                    .clamp_min(QUANT_EPSILON)
                    .into_scalar()
                    .elem::<f32>();
                if let Some(key) = cache_key.as_ref() {
                    TRAINING_WEIGHT_SCALE_CACHE.with(|cache| {
                        cache.borrow_mut().insert(key.clone(), scale_scalar);
                    });
                }
                (
                    Tensor::<B, 1>::from_data(
                        TensorData::new(vec![scale_scalar], [1]),
                        &tensor.device(),
                    ),
                    scale_scalar,
                    false,
                )
            };
            let active = tensor
                .clone()
                .abs()
                .div(scale.clone().reshape([1; D]))
                .greater_equal_elem(1.0)
                .float();
            let sign = tensor
                .clone()
                .greater_equal_elem(0.0)
                .float()
                .mul_scalar(2.0)
                .sub_scalar(1.0);
            Some((
                sign.mul(active).int(),
                scale,
                scale_scalar,
                used_cached_scale,
            ))
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ArtifactDeviceCacheKey {
    artifact_ptr: usize,
    device_tag: String,
    backend_name: &'static str,
    kind: &'static str,
    heads: usize,
    latent_per_head: usize,
}

thread_local! {
    static PACKED_ARTIFACT_DEVICE_CACHE: RefCell<HashMap<ArtifactDeviceCacheKey, Box<dyn Any>>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct DecoderTailRuntimeViewCacheKey {
    artifact_ptr: usize,
    device_tag: String,
    backend_name: &'static str,
    heads: usize,
    latent: usize,
}

thread_local! {
    static DECODER_TAIL_RUNTIME_VIEW_CACHE: RefCell<HashMap<DecoderTailRuntimeViewCacheKey, Box<dyn Any>>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ActivationScaleCacheKey {
    artifact_ptr: usize,
    device_tag: String,
    backend_name: &'static str,
    kind: &'static str,
    format: LowBitActivationFormat,
}

thread_local! {
    static ACTIVATION_SCALE_CACHE: RefCell<HashMap<ActivationScaleCacheKey, f32>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TrainingActivationScaleCacheKey {
    backend_name: &'static str,
    kind: &'static str,
    format: LowBitActivationFormat,
    shape: [usize; 4],
}

thread_local! {
    static TRAINING_ACTIVATION_SCALE_CACHE: RefCell<HashMap<TrainingActivationScaleCacheKey, f32>> =
        RefCell::new(HashMap::new());
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct TrainingWeightScaleCacheKey {
    backend_name: &'static str,
    kind: &'static str,
    format: LowBitWeightFormat,
    shape: Vec<usize>,
}

thread_local! {
    static TRAINING_WEIGHT_SCALE_CACHE: RefCell<HashMap<TrainingWeightScaleCacheKey, f32>> =
        RefCell::new(HashMap::new());
}

static LOW_BIT_NATIVE_PROJECTION_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LOW_BIT_NATIVE_LOWRANK_PROFILE: OnceLock<Mutex<LowBitNativeProjectionProfileSnapshot>> =
    OnceLock::new();
static LOW_BIT_NATIVE_DECODER_TAIL_PROFILE: OnceLock<Mutex<LowBitNativeProjectionProfileSnapshot>> =
    OnceLock::new();
static LOW_BIT_TRAINING_PROJECTION_MEMORY_PROFILE_ENABLED: OnceLock<bool> = OnceLock::new();
static LOW_BIT_TRAINING_PROJECTION_MEMORY_PROFILE_SYNC_ENABLED: OnceLock<bool> = OnceLock::new();
static LOW_BIT_TRAINING_LOWRANK_MEMORY_PROFILE: OnceLock<
    Mutex<LowBitTrainingProjectionMemoryProfileSnapshot>,
> = OnceLock::new();
static LOW_BIT_TRAINING_QUANTIZE_PROFILE: OnceLock<Mutex<LowBitTrainingQuantizeProfileSnapshot>> =
    OnceLock::new();
static LOW_BIT_TRAINING_CACHED_SCALE_ENABLED: OnceLock<bool> = OnceLock::new();

fn low_bit_native_projection_profile_enabled() -> bool {
    *LOW_BIT_NATIVE_PROJECTION_PROFILE_ENABLED
        .get_or_init(|| std::env::var_os("BDH_STAGE_PROFILE").is_some())
}

fn low_bit_training_projection_memory_profile_enabled() -> bool {
    *LOW_BIT_TRAINING_PROJECTION_MEMORY_PROFILE_ENABLED
        .get_or_init(|| std::env::var_os("BDH_STAGE_PROFILE_MEMORY").is_some())
}

fn low_bit_training_projection_memory_profile_sync_enabled() -> bool {
    *LOW_BIT_TRAINING_PROJECTION_MEMORY_PROFILE_SYNC_ENABLED
        .get_or_init(|| std::env::var_os("BDH_STAGE_PROFILE_MEMORY_SYNC").is_some())
}

fn low_bit_inference_cached_scale_enabled_for_backend<B: Backend>() -> bool {
    if std::env::var_os("BURN_DRAGON_LOWBIT_INFERENCE_DYNAMIC_SCALE_ONLY").is_some() {
        return false;
    }
    if std::env::var_os("BURN_DRAGON_LOWBIT_INFERENCE_CACHED_SCALE").is_some() {
        return true;
    }
    backend_name_prefers_inference_cached_scale(core::any::type_name::<B>())
}

fn low_bit_training_cached_scale_enabled() -> bool {
    *LOW_BIT_TRAINING_CACHED_SCALE_ENABLED.get_or_init(|| {
        std::env::var_os("BURN_DRAGON_LOWBIT_TRAINING_DYNAMIC_SCALE_ONLY").is_none()
    })
}

fn low_bit_native_lowrank_profile_state() -> &'static Mutex<LowBitNativeProjectionProfileSnapshot> {
    LOW_BIT_NATIVE_LOWRANK_PROFILE
        .get_or_init(|| Mutex::new(LowBitNativeProjectionProfileSnapshot::default()))
}

fn low_bit_native_decoder_tail_profile_state()
-> &'static Mutex<LowBitNativeProjectionProfileSnapshot> {
    LOW_BIT_NATIVE_DECODER_TAIL_PROFILE
        .get_or_init(|| Mutex::new(LowBitNativeProjectionProfileSnapshot::default()))
}

fn low_bit_training_lowrank_memory_profile_state()
-> &'static Mutex<LowBitTrainingProjectionMemoryProfileSnapshot> {
    LOW_BIT_TRAINING_LOWRANK_MEMORY_PROFILE
        .get_or_init(|| Mutex::new(LowBitTrainingProjectionMemoryProfileSnapshot::default()))
}

fn low_bit_training_quantize_profile_state() -> &'static Mutex<LowBitTrainingQuantizeProfileSnapshot>
{
    LOW_BIT_TRAINING_QUANTIZE_PROFILE
        .get_or_init(|| Mutex::new(LowBitTrainingQuantizeProfileSnapshot::default()))
}

pub fn low_bit_native_projection_profile_reset() {
    if let Ok(mut state) = low_bit_native_lowrank_profile_state().lock() {
        *state = LowBitNativeProjectionProfileSnapshot::default();
    }
    if let Ok(mut state) = low_bit_native_decoder_tail_profile_state().lock() {
        *state = LowBitNativeProjectionProfileSnapshot::default();
    }
    if let Ok(mut state) = low_bit_training_lowrank_memory_profile_state().lock() {
        *state = LowBitTrainingProjectionMemoryProfileSnapshot::default();
    }
    if let Ok(mut state) = low_bit_training_quantize_profile_state().lock() {
        *state = LowBitTrainingQuantizeProfileSnapshot::default();
    }
}

pub fn low_bit_native_lowrank_profile_snapshot() -> LowBitNativeProjectionProfileSnapshot {
    low_bit_native_lowrank_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

pub fn low_bit_native_decoder_tail_profile_snapshot() -> LowBitNativeProjectionProfileSnapshot {
    low_bit_native_decoder_tail_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

pub fn low_bit_training_lowrank_memory_profile_snapshot()
-> LowBitTrainingProjectionMemoryProfileSnapshot {
    low_bit_training_lowrank_memory_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

pub fn low_bit_training_quantize_profile_snapshot() -> LowBitTrainingQuantizeProfileSnapshot {
    low_bit_training_quantize_profile_state()
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}

fn low_bit_training_projection_memory_usage<B: Backend>(device: &B::Device) -> Option<(u64, u64)>
where
    B::Device: 'static,
{
    if low_bit_training_projection_memory_profile_sync_enabled() {
        let _ = B::sync(device);
    }

    #[cfg(feature = "cuda")]
    if let Some(cuda_device) = (device as &dyn Any).downcast_ref::<CudaDevice>() {
        let usage = <CudaRuntime as Runtime>::client(cuda_device)
            .memory_usage()
            .expect("cuda memory usage");
        return Some((usage.bytes_reserved, usage.bytes_in_use));
    }

    #[cfg(any(feature = "benchmark", feature = "train"))]
    if let Some(wgpu_device) = (device as &dyn Any).downcast_ref::<WgpuDevice>() {
        let usage = <WgpuRuntime as Runtime>::client(wgpu_device)
            .memory_usage()
            .expect("wgpu memory usage");
        return Some((usage.bytes_reserved, usage.bytes_in_use));
    }

    None
}

fn tensor_bytes<const D: usize>(shape: [usize; D], element_bytes: u64) -> u64 {
    shape
        .into_iter()
        .fold(1u64, |acc, dim| acc.saturating_mul(dim as u64))
        .saturating_mul(element_bytes)
}

fn record_low_bit_training_lowrank_memory_stage<B: Backend>(
    stage: &'static str,
    device: &B::Device,
    tracked_tensor_bytes: u64,
) where
    B::Device: 'static,
{
    if !low_bit_training_projection_memory_profile_enabled() {
        return;
    }
    let Some((reserved_bytes, in_use_bytes)) =
        low_bit_training_projection_memory_usage::<B>(device)
    else {
        return;
    };
    let observed = LowBitTrainingProjectionMemoryStageSnapshot {
        reserved_bytes,
        in_use_bytes,
        tracked_tensor_bytes,
    };
    if let Ok(mut profile) = low_bit_training_lowrank_memory_profile_state().lock() {
        profile.calls = profile.calls.saturating_add(1);
        let slot = match stage {
            "after_weight_codes" => &mut profile.after_weight_codes,
            "after_activation_codes" => &mut profile.after_activation_codes,
            "after_output" => &mut profile.after_output,
            _ => return,
        };
        if slot.should_replace(observed) {
            *slot = observed;
        }
    }
}

fn record_low_bit_training_quantize_profile(stage: &'static str, elapsed_ns: u128) {
    if !low_bit_native_projection_profile_enabled() {
        return;
    }
    if let Ok(mut profile) = low_bit_training_quantize_profile_state().lock() {
        match stage {
            "lowrank_direct" => {
                profile.lowrank_direct_calls = profile.lowrank_direct_calls.saturating_add(1);
                profile.lowrank_direct_total_ns =
                    profile.lowrank_direct_total_ns.saturating_add(elapsed_ns);
            }
            "lowrank_fallback" => {
                profile.lowrank_fallback_calls = profile.lowrank_fallback_calls.saturating_add(1);
                profile.lowrank_fallback_total_ns =
                    profile.lowrank_fallback_total_ns.saturating_add(elapsed_ns);
            }
            "decoder_tail" => {
                profile.decoder_tail_calls = profile.decoder_tail_calls.saturating_add(1);
                profile.decoder_tail_total_ns =
                    profile.decoder_tail_total_ns.saturating_add(elapsed_ns);
            }
            _ => {}
        }
    }
}

fn record_low_bit_native_projection_profile(
    kind: &'static str,
    total_ns: u128,
    quantize_ns: u128,
    prepacked_quantize_ns: u128,
    raw_cuda_ns: u128,
    fused_ns: u128,
    reference_ns: u128,
    used_cached_scale: bool,
) {
    let state = match kind {
        "lowrank" => low_bit_native_lowrank_profile_state(),
        "decoder_tail" => low_bit_native_decoder_tail_profile_state(),
        _ => return,
    };
    if let Ok(mut snapshot) = state.lock() {
        snapshot.calls = snapshot.calls.saturating_add(1);
        snapshot.total_ns = snapshot.total_ns.saturating_add(total_ns);
        snapshot.quantize_ns = snapshot.quantize_ns.saturating_add(quantize_ns);
        snapshot.prepacked_quantize_ns = snapshot
            .prepacked_quantize_ns
            .saturating_add(prepacked_quantize_ns);
        snapshot.raw_cuda_ns = snapshot.raw_cuda_ns.saturating_add(raw_cuda_ns);
        snapshot.fused_ns = snapshot.fused_ns.saturating_add(fused_ns);
        snapshot.reference_ns = snapshot.reference_ns.saturating_add(reference_ns);
        if used_cached_scale {
            snapshot.cached_scale_hits = snapshot.cached_scale_hits.saturating_add(1);
        } else {
            snapshot.dynamic_scale_calls = snapshot.dynamic_scale_calls.saturating_add(1);
        }
    }
}

fn cached_device_artifact<T, F>(key: ArtifactDeviceCacheKey, build: F) -> T
where
    T: Clone + 'static,
    F: FnOnce() -> T,
{
    PACKED_ARTIFACT_DEVICE_CACHE.with(|cache| {
        let mut cache = cache.borrow_mut();
        if let Some(existing) = cache.get(&key) {
            return existing
                .downcast_ref::<T>()
                .unwrap_or_else(|| panic!("packed artifact device cache type mismatch"))
                .clone();
        }
        let value = build();
        cache.insert(key, Box::new(value.clone()));
        value
    })
}

fn activation_scale_cache_key<B: Backend>(
    artifact: &PackedWeightArtifact,
    device: &B::Device,
    kind: &'static str,
    format: LowBitActivationFormat,
) -> ActivationScaleCacheKey
where
    B::Device: core::fmt::Debug,
{
    ActivationScaleCacheKey {
        artifact_ptr: artifact as *const PackedWeightArtifact as usize,
        device_tag: format!("{device:?}"),
        backend_name: std::any::type_name::<B>(),
        kind,
        format,
    }
}

fn artifact_codes_tensor_3d<B: Backend>(
    artifact: &PackedWeightArtifact,
    context: &str,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let shape = logical_shape_3d(artifact, context);
    let codes = unpack_weight_artifact_to_i8_codes(artifact)
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    Tensor::<B, 3, Int>::from_data(TensorData::new(codes, shape), device)
}

fn artifact_codes_tensor_2d<B: Backend>(
    artifact: &PackedWeightArtifact,
    context: &str,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let shape = logical_shape_2d(artifact, context);
    let codes = unpack_weight_artifact_to_i8_codes(artifact)
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    Tensor::<B, 2, Int>::from_data(TensorData::new(codes, shape), device)
}

fn artifact_packed_lowrank_weight_tensor_3d<B: Backend>(
    artifact: &PackedWeightArtifact,
    context: &str,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let [heads, embd, latent] = logical_shape_3d(artifact, context);
    let packed = pack_lowrank_weight_codes_i8x4(
        &unpack_weight_artifact_to_i8_codes(artifact),
        heads,
        embd,
        latent,
    )
    .into_iter()
    .map(i64::from)
    .collect::<Vec<_>>();
    Tensor::<B, 3, Int>::from_data(
        TensorData::new(packed, [heads, embd.div_ceil(4), latent]),
        device,
    )
}

fn artifact_packed_decoder_weight_tensor_2d<B: Backend>(
    artifact: &PackedWeightArtifact,
    context: &str,
    heads: usize,
    latent_per_head: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let [artifact_latent_total, dim] = logical_shape_2d(artifact, context);
    assert_eq!(
        artifact_latent_total,
        heads * latent_per_head,
        "{context} latent_total mismatch: artifact={} expected={}",
        artifact_latent_total,
        heads * latent_per_head
    );
    let packed = pack_decoder_weight_codes_i8x4(
        &unpack_weight_artifact_to_i8_codes(artifact),
        heads,
        latent_per_head,
        dim,
    )
    .into_iter()
    .map(i64::from)
    .collect::<Vec<_>>();
    Tensor::<B, 2, Int>::from_data(
        TensorData::new(packed, [heads * latent_per_head.div_ceil(4), dim]),
        device,
    )
}

fn decoder_tail_runtime_views<B: Backend>(
    cache: &CachedDecoderTailArtifact<B>,
    heads: usize,
    latent: usize,
) -> CachedDecoderTailRuntimeView<B>
where
    B: 'static,
    B::Device: core::fmt::Debug,
{
    let [artifact_latent_total, dim] =
        logical_shape_2d(&cache.artifact, "packed decoder tail runtime view");
    assert_eq!(
        artifact_latent_total % heads,
        0,
        "packed decoder tail runtime view head mismatch: latent_total={} heads={}",
        artifact_latent_total,
        heads
    );
    let artifact_latent_per_head = artifact_latent_total / heads;
    assert!(
        latent <= artifact_latent_per_head,
        "packed decoder tail runtime latent mismatch: requested {} > artifact {}",
        latent,
        artifact_latent_per_head
    );
    if latent == artifact_latent_per_head {
        return CachedDecoderTailRuntimeView {
            codes: cache.codes.clone(),
            packed_weight: cache.packed_weight.clone(),
        };
    }
    let key = DecoderTailRuntimeViewCacheKey {
        artifact_ptr: &cache.artifact as *const PackedWeightArtifact as usize,
        device_tag: format!("{:?}", cache.codes.device()),
        backend_name: std::any::type_name::<B>(),
        heads,
        latent,
    };
    DECODER_TAIL_RUNTIME_VIEW_CACHE.with(|runtime_cache| {
        let mut runtime_cache = runtime_cache.borrow_mut();
        if let Some(existing) = runtime_cache.get(&key) {
            return existing
                .downcast_ref::<CachedDecoderTailRuntimeView<B>>()
                .unwrap_or_else(|| panic!("decoder-tail runtime view cache type mismatch"))
                .clone();
        }
        let full_codes = unpack_weight_artifact_to_i8_codes(&cache.artifact);
        let mut active_codes = Vec::with_capacity(heads * latent * dim);
        for head_idx in 0..heads {
            let head_offset = head_idx * artifact_latent_per_head * dim;
            for latent_idx in 0..latent {
                let row_offset = head_offset + latent_idx * dim;
                active_codes.extend_from_slice(&full_codes[row_offset..row_offset + dim]);
            }
        }
        let packed = pack_decoder_weight_codes_i8x4(&active_codes, heads, latent, dim);
        let device = cache.codes.device();
        let view = CachedDecoderTailRuntimeView {
            codes: Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    active_codes
                        .iter()
                        .copied()
                        .map(i64::from)
                        .collect::<Vec<_>>(),
                    [heads * latent, dim],
                ),
                &device,
            ),
            packed_weight: Tensor::<B, 2, Int>::from_data(
                TensorData::new(
                    packed.into_iter().map(i64::from).collect::<Vec<_>>(),
                    [heads * latent.div_ceil(4), dim],
                ),
                &device,
            ),
        };
        runtime_cache.insert(key, Box::new(view.clone()));
        view
    })
}

pub(crate) fn cache_lowrank_projection_artifact<B: Backend>(
    artifact: &PackedWeightArtifact,
    device: &B::Device,
    context: &str,
) -> CachedLowrankProjectionArtifact<B>
where
    B: 'static,
    B::Device: core::fmt::Debug,
{
    let [_, _, latent_out] = logical_shape_3d(artifact, context);
    let key = ArtifactDeviceCacheKey {
        artifact_ptr: artifact as *const PackedWeightArtifact as usize,
        device_tag: format!("{device:?}"),
        backend_name: std::any::type_name::<B>(),
        kind: "lowrank",
        heads: 0,
        latent_per_head: 0,
    };
    cached_device_artifact(key, || CachedLowrankProjectionArtifact {
        artifact: artifact.clone(),
        codes: artifact_codes_tensor_3d(artifact, context, device),
        packed_weight: artifact_packed_lowrank_weight_tensor_3d(artifact, context, device),
        latent_out,
    })
}

pub(crate) fn cache_decoder_tail_artifact<B: Backend>(
    artifact: &PackedWeightArtifact,
    heads: usize,
    latent_per_head: usize,
    device: &B::Device,
    context: &str,
) -> CachedDecoderTailArtifact<B>
where
    B: 'static,
    B::Device: core::fmt::Debug,
{
    let key = ArtifactDeviceCacheKey {
        artifact_ptr: artifact as *const PackedWeightArtifact as usize,
        device_tag: format!("{device:?}"),
        backend_name: std::any::type_name::<B>(),
        kind: "decoder_tail",
        heads,
        latent_per_head,
    };
    cached_device_artifact(key, || CachedDecoderTailArtifact {
        artifact: artifact.clone(),
        codes: artifact_codes_tensor_2d(artifact, context, device),
        packed_weight: artifact_packed_decoder_weight_tensor_2d(
            artifact,
            context,
            heads,
            latent_per_head,
            device,
        ),
    })
}

pub(crate) fn packed_lowrank_projection_native_cached<B: Backend>(
    input: Tensor<B, 4>,
    cache: &CachedLowrankProjectionArtifact<B>,
    activation_format: Option<LowBitActivationFormat>,
) -> Tensor<B, 4> {
    let profile_enabled = low_bit_native_projection_profile_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let mut quantize_ns = 0;
    let mut prepacked_quantize_ns = 0;
    let mut raw_cuda_ns = 0;
    let mut fused_ns = 0;
    let mut reference_ns = 0;
    let mut used_cached_scale = false;
    let scale_cache_key = activation_format
        .filter(|format| !matches!(format, LowBitActivationFormat::Fp16))
        .map(|format| {
            activation_scale_cache_key::<B>(&cache.artifact, &input.device(), "lowrank", format)
        });
    let fused_start = profile_enabled.then(Instant::now);
    if let Some((quant_format, activation_scale, cached_scale_hit)) =
        activation_scale_tensor_cached_device_4d(
            &input,
            activation_format,
            scale_cache_key.as_ref(),
        )
    {
        let (qmax, positive_only) = match quant_format {
            LowBitActivationFormat::Int8 => (127, false),
            LowBitActivationFormat::Int4Exp => (7, false),
            LowBitActivationFormat::Fp16 | LowBitActivationFormat::Uint8PosExp => (0, false),
        };
        used_cached_scale = cached_scale_hit;
        if let Some(fused) = try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale(
            &input,
            &cache.packed_weight,
            &activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
            cache.latent_out,
            qmax,
            positive_only,
        ) {
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "lowrank",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return fused;
        }
    }
    let skip_wgpu_prepacked_lowrank =
        matches!(cached_wgpu_packed_dot_lowrank_support(&input), Some(false));
    let quantize_start = profile_enabled.then(Instant::now);
    if !skip_wgpu_prepacked_lowrank {
        if let Some((input_packed, activation_scale, cached_scale_hit)) =
            quantize_activation_packed_codes_tensor_device_scale_4d(
                &input,
                activation_format,
                scale_cache_key.as_ref(),
            )
        {
            used_cached_scale = cached_scale_hit;
            if let Some(start) = quantize_start {
                prepacked_quantize_ns = start.elapsed().as_nanos();
            }
            let raw_start = profile_enabled.then(Instant::now);
            if let Some(raw_cuda) =
                try_raw_cuda_packed_lowrank_projection_prepacked_input_device_scale(
                    &input_packed,
                    &cache.packed_weight,
                    &activation_scale,
                    cache.artifact.scale.max(QUANT_EPSILON),
                    cache.latent_out,
                )
            {
                if let Some(start) = raw_start {
                    raw_cuda_ns = start.elapsed().as_nanos();
                }
                if let Some(start) = total_start {
                    record_low_bit_native_projection_profile(
                        "lowrank",
                        start.elapsed().as_nanos(),
                        quantize_ns,
                        prepacked_quantize_ns,
                        raw_cuda_ns,
                        fused_ns,
                        reference_ns,
                        used_cached_scale,
                    );
                }
                return raw_cuda;
            }
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            let fused_start = profile_enabled.then(Instant::now);
            if let Some(fused) = try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale(
                &input_packed,
                &cache.packed_weight,
                &activation_scale,
                cache.artifact.scale.max(QUANT_EPSILON),
                cache.latent_out,
            ) {
                if let Some(start) = fused_start {
                    fused_ns = start.elapsed().as_nanos();
                }
                if let Some(start) = total_start {
                    record_low_bit_native_projection_profile(
                        "lowrank",
                        start.elapsed().as_nanos(),
                        quantize_ns,
                        prepacked_quantize_ns,
                        raw_cuda_ns,
                        fused_ns,
                        reference_ns,
                        used_cached_scale,
                    );
                }
                return fused;
            }
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
        }
        let quantize_start = profile_enabled.then(Instant::now);
        if let Some((input_codes, activation_scale)) =
            quantize_activation_codes_tensor_device_scale_4d(&input, activation_format)
        {
            if let Some(start) = quantize_start {
                quantize_ns = start.elapsed().as_nanos();
            }
            let raw_start = profile_enabled.then(Instant::now);
            if let Some(raw_cuda) = try_raw_cuda_packed_lowrank_projection_device_scale(
                &input_codes,
                &cache.packed_weight,
                &activation_scale,
                cache.artifact.scale.max(QUANT_EPSILON),
                cache.latent_out,
            ) {
                if let Some(start) = raw_start {
                    raw_cuda_ns = start.elapsed().as_nanos();
                }
                if let Some(start) = total_start {
                    record_low_bit_native_projection_profile(
                        "lowrank",
                        start.elapsed().as_nanos(),
                        quantize_ns,
                        prepacked_quantize_ns,
                        raw_cuda_ns,
                        fused_ns,
                        reference_ns,
                        used_cached_scale,
                    );
                }
                return raw_cuda;
            }
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            let fused_start = profile_enabled.then(Instant::now);
            if let Some(fused) = try_wgpu_packed_dot_lowrank_projection_device_scale(
                &input_codes,
                &cache.packed_weight,
                &activation_scale,
                cache.artifact.scale.max(QUANT_EPSILON),
                cache.latent_out,
            ) {
                if let Some(start) = fused_start {
                    fused_ns = start.elapsed().as_nanos();
                }
                if let Some(start) = total_start {
                    record_low_bit_native_projection_profile(
                        "lowrank",
                        start.elapsed().as_nanos(),
                        quantize_ns,
                        prepacked_quantize_ns,
                        raw_cuda_ns,
                        fused_ns,
                        reference_ns,
                        used_cached_scale,
                    );
                }
                return fused;
            }
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
        }
    } else if let Some(start) = quantize_start {
        quantize_ns = start.elapsed().as_nanos();
    }
    let quantize_start = profile_enabled.then(Instant::now);
    if let Some((input_codes, activation_scale, cached_scale_hit)) =
        quantize_activation_codes_tensor(&input, activation_format, scale_cache_key.as_ref())
    {
        used_cached_scale = cached_scale_hit;
        if let Some(start) = quantize_start {
            quantize_ns = start.elapsed().as_nanos();
        }
        let raw_start = profile_enabled.then(Instant::now);
        if let Some(raw_cuda) = try_raw_cuda_packed_lowrank_projection(
            &input_codes,
            &cache.packed_weight,
            activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
            cache.latent_out,
        ) {
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "lowrank",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return raw_cuda;
        }
        if let Some(start) = raw_start {
            raw_cuda_ns = start.elapsed().as_nanos();
        }
        let fused_start = profile_enabled.then(Instant::now);
        if let Some(fused) = try_fused_packed_lowrank_projection(
            &input_codes,
            &cache.codes,
            activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
            cache.latent_out,
        ) {
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "lowrank",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return fused;
        }
        if let Some(start) = fused_start {
            fused_ns = start.elapsed().as_nanos();
        }
    } else if let Some(start) = quantize_start {
        quantize_ns = start.elapsed().as_nanos();
    }
    if !skip_wgpu_prepacked_lowrank {
        let prepacked_start = profile_enabled.then(Instant::now);
        if let Some((input_packed, activation_scale)) =
            quantize_activation_packed_codes_tensor_4d(&input, activation_format, |codes, shape| {
                pack_lowrank_input_codes_i8x4(codes, shape[0], shape[1], shape[2], shape[3])
            })
        {
            if let Some(start) = prepacked_start {
                prepacked_quantize_ns = start.elapsed().as_nanos();
            }
            let raw_start = profile_enabled.then(Instant::now);
            if let Some(raw_cuda) = try_raw_cuda_packed_lowrank_projection_prepacked_input(
                &input_packed,
                &cache.packed_weight,
                activation_scale,
                cache.artifact.scale.max(QUANT_EPSILON),
                cache.latent_out,
            ) {
                if let Some(start) = raw_start {
                    raw_cuda_ns = raw_cuda_ns.saturating_add(start.elapsed().as_nanos());
                }
                if let Some(start) = total_start {
                    record_low_bit_native_projection_profile(
                        "lowrank",
                        start.elapsed().as_nanos(),
                        quantize_ns,
                        prepacked_quantize_ns,
                        raw_cuda_ns,
                        fused_ns,
                        reference_ns,
                        used_cached_scale,
                    );
                }
                return raw_cuda;
            }
            if let Some(start) = raw_start {
                raw_cuda_ns = raw_cuda_ns.saturating_add(start.elapsed().as_nanos());
            }
        } else if let Some(start) = prepacked_start {
            prepacked_quantize_ns = start.elapsed().as_nanos();
        }
    }
    let input = if let Some(format) = activation_format {
        fake_quantize_activation_ste(input, format)
    } else {
        input
    };
    let reference_start = profile_enabled.then(Instant::now);
    let output = packed_lowrank_projection_device_reference(
        input,
        cache.codes.clone(),
        cache.artifact.scale.max(QUANT_EPSILON),
        cache.latent_out,
    );
    if let Some(start) = reference_start {
        reference_ns = start.elapsed().as_nanos();
    }
    if let Some(start) = total_start {
        record_low_bit_native_projection_profile(
            "lowrank",
            start.elapsed().as_nanos(),
            quantize_ns,
            prepacked_quantize_ns,
            raw_cuda_ns,
            fused_ns,
            reference_ns,
            used_cached_scale,
        );
    }
    output
}

pub(crate) fn packed_decoder_tail_native_cached<B: Backend>(
    y_neuron: Tensor<B, 4>,
    cache: &CachedDecoderTailArtifact<B>,
    activation_format: Option<LowBitActivationFormat>,
) -> Tensor<B, 4> {
    let [_, heads, _, latent] = y_neuron.shape().dims::<4>();
    let runtime_view = decoder_tail_runtime_views(cache, heads, latent);
    let profile_enabled = low_bit_native_projection_profile_enabled();
    let total_start = profile_enabled.then(Instant::now);
    let mut quantize_ns = 0;
    let mut prepacked_quantize_ns = 0;
    let mut raw_cuda_ns = 0;
    let mut fused_ns = 0;
    let mut reference_ns = 0;
    let mut used_cached_scale = false;
    let scale_cache_key = activation_format
        .filter(|format| !matches!(format, LowBitActivationFormat::Fp16))
        .map(|format| {
            activation_scale_cache_key::<B>(
                &cache.artifact,
                &y_neuron.device(),
                "decoder_tail",
                format,
            )
        });
    let skip_wgpu_prepacked_decoder_tail = matches!(
        cached_wgpu_packed_dot_decoder_tail_support(&y_neuron),
        Some(false)
    );
    let quantize_start = profile_enabled.then(Instant::now);
    if !skip_wgpu_prepacked_decoder_tail
        && let Some((y_packed, activation_scale, cached_scale_hit)) =
            quantize_activation_packed_codes_tensor_device_scale_4d(
                &y_neuron,
                activation_format,
                scale_cache_key.as_ref(),
            )
    {
        used_cached_scale = cached_scale_hit;
        if let Some(start) = quantize_start {
            prepacked_quantize_ns = start.elapsed().as_nanos();
        }
        let raw_start = profile_enabled.then(Instant::now);
        if let Some(raw_cuda) = try_raw_cuda_packed_decoder_tail_prepacked_input_device_scale(
            &y_packed,
            &runtime_view.packed_weight,
            &activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return raw_cuda;
        }
        if let Some(start) = raw_start {
            raw_cuda_ns = start.elapsed().as_nanos();
        }
        let fused_start = profile_enabled.then(Instant::now);
        if let Some(fused) = try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale(
            &y_packed,
            &runtime_view.packed_weight,
            &activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return fused;
        }
        if let Some(start) = fused_start {
            fused_ns = start.elapsed().as_nanos();
        }
    }
    let quantize_start = profile_enabled.then(Instant::now);
    if let Some((y_codes, activation_scale)) =
        quantize_activation_codes_tensor_device_scale_4d(&y_neuron, activation_format)
    {
        if let Some(start) = quantize_start {
            quantize_ns = start.elapsed().as_nanos();
        }
        let raw_start = profile_enabled.then(Instant::now);
        if let Some(raw_cuda) = try_raw_cuda_packed_decoder_tail_device_scale(
            &y_codes,
            &runtime_view.packed_weight,
            &activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return raw_cuda;
        }
        if let Some(start) = raw_start {
            raw_cuda_ns = start.elapsed().as_nanos();
        }
        let fused_start = profile_enabled.then(Instant::now);
        if let Some(fused) = try_wgpu_packed_dot_decoder_tail_device_scale(
            &y_codes,
            &runtime_view.packed_weight,
            &activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return fused;
        }
        if let Some(start) = fused_start {
            fused_ns = start.elapsed().as_nanos();
        }
    }
    let quantize_start = profile_enabled.then(Instant::now);
    if let Some((y_codes, activation_scale, cached_scale_hit)) =
        quantize_activation_codes_tensor(&y_neuron, activation_format, scale_cache_key.as_ref())
    {
        used_cached_scale = cached_scale_hit;
        if let Some(start) = quantize_start {
            quantize_ns = start.elapsed().as_nanos();
        }
        let raw_start = profile_enabled.then(Instant::now);
        if let Some(raw_cuda) = try_raw_cuda_packed_decoder_tail(
            &y_codes,
            &runtime_view.packed_weight,
            activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = raw_start {
                raw_cuda_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return raw_cuda;
        }
        if let Some(start) = raw_start {
            raw_cuda_ns = start.elapsed().as_nanos();
        }
        let fused_start = profile_enabled.then(Instant::now);
        if let Some(fused) = try_fused_packed_decoder_tail(
            &y_codes,
            &runtime_view.codes,
            activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = fused_start {
                fused_ns = start.elapsed().as_nanos();
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return fused;
        }
        if let Some(start) = fused_start {
            fused_ns = start.elapsed().as_nanos();
        }
    } else if let Some(start) = quantize_start {
        quantize_ns = start.elapsed().as_nanos();
    }
    let prepacked_start = profile_enabled.then(Instant::now);
    if let Some((y_packed, activation_scale)) =
        quantize_activation_packed_codes_tensor_4d(&y_neuron, activation_format, |codes, shape| {
            pack_decoder_input_codes_i8x4(codes, shape[0], shape[1], shape[2], shape[3])
        })
    {
        if let Some(start) = prepacked_start {
            prepacked_quantize_ns = start.elapsed().as_nanos();
        }
        let raw_start = profile_enabled.then(Instant::now);
        if let Some(raw_cuda) = try_raw_cuda_packed_decoder_tail_prepacked_input(
            &y_packed,
            &runtime_view.packed_weight,
            activation_scale,
            cache.artifact.scale.max(QUANT_EPSILON),
        ) {
            if let Some(start) = raw_start {
                raw_cuda_ns = raw_cuda_ns.saturating_add(start.elapsed().as_nanos());
            }
            if let Some(start) = total_start {
                record_low_bit_native_projection_profile(
                    "decoder_tail",
                    start.elapsed().as_nanos(),
                    quantize_ns,
                    prepacked_quantize_ns,
                    raw_cuda_ns,
                    fused_ns,
                    reference_ns,
                    used_cached_scale,
                );
            }
            return raw_cuda;
        }
        if let Some(start) = raw_start {
            raw_cuda_ns = raw_cuda_ns.saturating_add(start.elapsed().as_nanos());
        }
    } else if let Some(start) = prepacked_start {
        prepacked_quantize_ns = start.elapsed().as_nanos();
    }
    let y_neuron = if let Some(format) = activation_format {
        fake_quantize_activation_ste(y_neuron, format)
    } else {
        y_neuron
    };
    let reference_start = profile_enabled.then(Instant::now);
    let output = packed_decoder_tail_device_reference(
        y_neuron,
        runtime_view.codes,
        cache.artifact.scale.max(QUANT_EPSILON),
    );
    if let Some(start) = reference_start {
        reference_ns = start.elapsed().as_nanos();
    }
    if let Some(start) = total_start {
        record_low_bit_native_projection_profile(
            "decoder_tail",
            start.elapsed().as_nanos(),
            quantize_ns,
            prepacked_quantize_ns,
            raw_cuda_ns,
            fused_ns,
            reference_ns,
            used_cached_scale,
        );
    }
    output
}

pub(crate) fn packed_lowrank_projection_native<B: Backend>(
    input: Tensor<B, 4>,
    artifact: &PackedWeightArtifact,
    activation_format: Option<LowBitActivationFormat>,
    latent_out: usize,
) -> Tensor<B, 4>
where
    B: 'static,
    B::Device: core::fmt::Debug,
{
    let device = input.device();
    let cache =
        cache_lowrank_projection_artifact(artifact, &device, "packed low-rank native projection");
    debug_assert_eq!(cache.latent_out, latent_out);
    packed_lowrank_projection_native_cached(input, &cache, activation_format)
}

pub(crate) fn packed_decoder_tail_native<B: Backend>(
    y_neuron: Tensor<B, 4>,
    artifact: &PackedWeightArtifact,
    activation_format: Option<LowBitActivationFormat>,
) -> Tensor<B, 4>
where
    B: 'static,
    B::Device: core::fmt::Debug,
{
    let device = y_neuron.device();
    let [_, heads, _, latent] = y_neuron.shape().dims::<4>();
    let cache = cache_decoder_tail_artifact(
        artifact,
        heads,
        latent,
        &device,
        "packed decoder tail native projection",
    );
    packed_decoder_tail_native_cached(y_neuron, &cache, activation_format)
}

pub(crate) fn packed_lowrank_projection_training_native<B: Backend>(
    input: Tensor<B, 4>,
    projector: Tensor<B, 4>,
    weight_format: LowBitWeightFormat,
    activation_format: Option<LowBitActivationFormat>,
    latent_out: usize,
    saved_activation_mode: LowBitSavedActivationMode,
    relu_threshold: Option<f32>,
    scale_cache_kind: Option<&'static str>,
) -> Tensor<B, 4> {
    let input_detached = input.clone().detach();
    let input_device = input.device();
    let pack_activation_state_to_host = matches!(
        saved_activation_mode,
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp
    );
    let projector_detached = projector.clone().detach();
    if let Some((direct_codes, _weight_scale_tensor, weight_scale_scalar, _)) =
        weight_codes_tensor_device_scale_from_float_values(
            &projector_detached,
            weight_format,
            scale_cache_kind,
        )
    {
        let direct_shape = direct_codes.shape().dims::<4>();
        assert_eq!(
            direct_shape[0], 1,
            "packed low-rank training projection expects singleton outer axis for direct codes"
        );
        let codes = direct_codes
            .slice([
                0..1,
                0..direct_shape[1],
                0..direct_shape[2],
                0..direct_shape[3],
            ])
            .reshape([direct_shape[1], direct_shape[2], direct_shape[3]]);
        record_low_bit_training_lowrank_memory_stage::<B>(
            "after_weight_codes",
            &input_device,
            tensor_bytes(
                projector_detached.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            )
            .saturating_add(tensor_bytes(
                codes.shape().dims::<3>(),
                core::mem::size_of::<i32>() as u64,
            )),
        );
        let quantize_start = low_bit_native_projection_profile_enabled().then(Instant::now);
        if let Some((input_codes, activation_scale, _)) =
            quantize_training_activation_codes_tensor_4d(
                &input_detached,
                activation_format,
                scale_cache_kind,
            )
        {
            if let Some(start) = quantize_start {
                record_low_bit_training_quantize_profile(
                    "lowrank_direct",
                    start.elapsed().as_nanos(),
                );
            }
            record_low_bit_training_lowrank_memory_stage::<B>(
                "after_activation_codes",
                &input_device,
                tensor_bytes(
                    projector_detached.shape().dims::<4>(),
                    core::mem::size_of::<f32>() as u64,
                )
                .saturating_add(tensor_bytes(
                    codes.shape().dims::<3>(),
                    core::mem::size_of::<i32>() as u64,
                ))
                .saturating_add(tensor_bytes(
                    input_detached.shape().dims::<4>(),
                    core::mem::size_of::<f32>() as u64,
                ))
                .saturating_add(tensor_bytes(
                    input_codes.shape().dims::<4>(),
                    core::mem::size_of::<i32>() as u64,
                )),
            );
            let projection_scale = Tensor::<B, 1>::from_data(
                TensorData::new(vec![activation_scale * weight_scale_scalar], [1]),
                &input_device,
            );
            if let Some(fused_autodiff) = try_fused_packed_lowrank_training_autodiff(
                &input,
                &projector,
                &input_codes,
                &codes,
                activation_scale,
                weight_scale_scalar,
                latent_out,
                pack_activation_state_to_host,
                relu_threshold,
            )
            .or_else(|| {
                try_fused_packed_lowrank_training_autodiff_cuda_device_projection_scale(
                    &input,
                    &projector,
                    &input_codes,
                    &codes,
                    activation_scale,
                    &projection_scale,
                    latent_out,
                    pack_activation_state_to_host,
                    relu_threshold,
                )
            }) {
                record_low_bit_training_lowrank_memory_stage::<B>(
                    "after_output",
                    &input_device,
                    tensor_bytes(
                        projector_detached.shape().dims::<4>(),
                        core::mem::size_of::<f32>() as u64,
                    )
                    .saturating_add(tensor_bytes(
                        codes.shape().dims::<3>(),
                        core::mem::size_of::<i32>() as u64,
                    ))
                    .saturating_add(tensor_bytes(
                        input_detached.shape().dims::<4>(),
                        core::mem::size_of::<f32>() as u64,
                    ))
                    .saturating_add(tensor_bytes(
                        input_codes.shape().dims::<4>(),
                        core::mem::size_of::<i32>() as u64,
                    ))
                    .saturating_add(tensor_bytes(
                        fused_autodiff.shape().dims::<4>(),
                        core::mem::size_of::<f32>() as u64,
                    )),
                );
                return fused_autodiff;
            }
        } else if let Some(start) = quantize_start {
            record_low_bit_training_quantize_profile("lowrank_direct", start.elapsed().as_nanos());
        }
    }

    let (direct_codes, weight_scale, _) =
        weight_codes_tensor_from_float_values(&projector_detached, weight_format, scale_cache_kind)
            .expect("native low-rank training projection requires non-fp16 low-bit weights");
    let direct_shape = direct_codes.shape().dims::<4>();
    assert_eq!(
        direct_shape[0], 1,
        "packed low-rank training projection expects singleton outer axis for direct codes"
    );
    let codes = direct_codes
        .slice([
            0..1,
            0..direct_shape[1],
            0..direct_shape[2],
            0..direct_shape[3],
        ])
        .reshape([direct_shape[1], direct_shape[2], direct_shape[3]]);
    let quantize_start = low_bit_native_projection_profile_enabled().then(Instant::now);
    if let Some((input_codes, activation_scale, _)) = quantize_training_activation_codes_tensor_4d(
        &input_detached,
        activation_format,
        scale_cache_kind,
    ) {
        if let Some(start) = quantize_start {
            record_low_bit_training_quantize_profile(
                "lowrank_fallback",
                start.elapsed().as_nanos(),
            );
        }
        let quantized_weight = fake_quantize_weight_ste(projector.clone(), weight_format);
        let quantized_input = if let Some(format) = activation_format {
            fake_quantize_activation_ste(input.clone(), format)
        } else {
            input.clone()
        };
        let native = try_fused_packed_lowrank_projection(
            &input_codes,
            &codes,
            activation_scale,
            weight_scale,
            latent_out,
        )
        .unwrap_or_else(|| {
            packed_lowrank_projection_device_reference(
                quantized_input.clone(),
                codes.clone(),
                weight_scale,
                latent_out,
            )
        });
        let reference = quantized_input.clone().matmul(quantized_weight.clone());
        let output = reference + native.clone() - native.detach();
        record_low_bit_training_lowrank_memory_stage::<B>(
            "after_output",
            &input_device,
            tensor_bytes(
                projector_detached.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            )
            .saturating_add(tensor_bytes(
                codes.shape().dims::<3>(),
                core::mem::size_of::<i32>() as u64,
            ))
            .saturating_add(tensor_bytes(
                input_detached.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            ))
            .saturating_add(tensor_bytes(
                input_codes.shape().dims::<4>(),
                core::mem::size_of::<i32>() as u64,
            ))
            .saturating_add(tensor_bytes(
                output.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            )),
        );
        output
    } else {
        if let Some(start) = quantize_start {
            record_low_bit_training_quantize_profile(
                "lowrank_fallback",
                start.elapsed().as_nanos(),
            );
        }
        let quantized_weight = fake_quantize_weight_ste(projector.clone(), weight_format);
        let quantized_input = if let Some(format) = activation_format {
            fake_quantize_activation_ste(input.clone(), format)
        } else {
            input.clone()
        };
        let native = packed_lowrank_projection_device_reference(
            quantized_input.clone(),
            codes,
            weight_scale,
            latent_out,
        );
        let reference = quantized_input.clone().matmul(quantized_weight.clone());
        let output = reference + native.clone() - native.detach();
        record_low_bit_training_lowrank_memory_stage::<B>(
            "after_output",
            &input_device,
            tensor_bytes(
                projector_detached.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            )
            .saturating_add(tensor_bytes(
                input_detached.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            ))
            .saturating_add(tensor_bytes(
                output.shape().dims::<4>(),
                core::mem::size_of::<f32>() as u64,
            )),
        );
        output
    }
}

pub(crate) fn packed_decoder_tail_training_native<B: Backend>(
    y_neuron: Tensor<B, 4>,
    decoder: Tensor<B, 2>,
    weight_format: LowBitWeightFormat,
    activation_format: Option<LowBitActivationFormat>,
    saved_activation_mode: LowBitSavedActivationMode,
    scale_cache_kind: Option<&'static str>,
) -> Tensor<B, 4> {
    let shape = decoder.shape().dims::<2>();
    let y_neuron_detached = y_neuron.clone().detach();
    let pack_activation_state_to_host = matches!(
        saved_activation_mode,
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp
    );
    let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
    let decoder_detached = decoder.clone().detach();
    let (codes, weight_scale, _) =
        weight_codes_tensor_from_float_values(&decoder_detached, weight_format, scale_cache_kind)
            .expect("native decoder-tail training projection requires non-fp16 low-bit weights");
    let quantize_start = low_bit_native_projection_profile_enabled().then(Instant::now);
    if let Some((y_codes, activation_scale, _)) =
        quantize_activation_codes_tensor(&y_neuron_detached, activation_format, None)
    {
        if let Some(start) = quantize_start {
            record_low_bit_training_quantize_profile("decoder_tail", start.elapsed().as_nanos());
        }
        if let Some(fused_autodiff) = try_fused_packed_decoder_tail_training_autodiff(
            &y_neuron,
            &decoder,
            &y_codes,
            &codes,
            activation_scale,
            weight_scale,
            pack_activation_state_to_host,
        ) {
            return fused_autodiff;
        }
        let quantized_decoder = fake_quantize_weight_ste(decoder.clone(), weight_format);
        let quantized_y_neuron = if let Some(format) = activation_format {
            fake_quantize_activation_ste(y_neuron.clone(), format)
        } else {
            y_neuron.clone()
        };
        let native =
            try_fused_packed_decoder_tail(&y_codes, &codes, activation_scale, weight_scale)
                .unwrap_or_else(|| {
                    packed_decoder_tail_device_reference(
                        quantized_y_neuron.clone(),
                        codes.clone(),
                        weight_scale,
                    )
                });
        let decoder_by_head = quantized_decoder.clone().reshape([heads, latent, shape[1]]);
        let mixed_by_head =
            quantized_y_neuron
                .clone()
                .swap_dims(0, 1)
                .reshape([heads, batch * time, latent]);
        let reference = mixed_by_head
            .matmul(decoder_by_head)
            .sum_dim(0)
            .reshape([batch, 1, time, shape[1]]);
        return reference + native.clone() - native.detach();
    } else {
        if let Some(start) = quantize_start {
            record_low_bit_training_quantize_profile("decoder_tail", start.elapsed().as_nanos());
        }
        let quantized_decoder = fake_quantize_weight_ste(decoder.clone(), weight_format);
        let quantized_y_neuron = if let Some(format) = activation_format {
            fake_quantize_activation_ste(y_neuron.clone(), format)
        } else {
            y_neuron.clone()
        };
        let native =
            packed_decoder_tail_device_reference(quantized_y_neuron.clone(), codes, weight_scale);
        let decoder_by_head = quantized_decoder.clone().reshape([heads, latent, shape[1]]);
        let mixed_by_head =
            quantized_y_neuron
                .clone()
                .swap_dims(0, 1)
                .reshape([heads, batch * time, latent]);
        let reference = mixed_by_head
            .matmul(decoder_by_head)
            .sum_dim(0)
            .reshape([batch, 1, time, shape[1]]);
        return reference + native.clone() - native.detach();
    }
}

fn quantize_reference_activations(
    values: &[f32],
    format: Option<LowBitActivationFormat>,
) -> ReferenceActivationBuffer {
    match format.unwrap_or(LowBitActivationFormat::Fp16) {
        LowBitActivationFormat::Fp16 => ReferenceActivationBuffer::Float(values.to_vec()),
        LowBitActivationFormat::Int8 => quantize_reference_signed(values, 127.0),
        LowBitActivationFormat::Int4Exp => quantize_reference_signed(values, 7.0),
        LowBitActivationFormat::Uint8PosExp => quantize_reference_positive(values, 255.0),
    }
}

fn quantize_reference_signed(values: &[f32], qmax: f32) -> ReferenceActivationBuffer {
    let mean_abs = if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| value.abs()).sum::<f32>() / values.len() as f32
    };
    let dynamic_range = (mean_abs * 2.0).max(QUANT_EPSILON);
    let scale = (dynamic_range / qmax).max(QUANT_EPSILON);
    let values = values
        .iter()
        .map(|value| (value / scale).round().clamp(-qmax, qmax) as i8)
        .collect();
    ReferenceActivationBuffer::Signed { values, scale }
}

fn quantize_reference_positive(values: &[f32], qmax: f32) -> ReferenceActivationBuffer {
    let clamped = values
        .iter()
        .map(|value| value.max(0.0))
        .collect::<Vec<_>>();
    let mean_value = if clamped.is_empty() {
        0.0
    } else {
        clamped.iter().sum::<f32>() / clamped.len() as f32
    };
    let dynamic_range = (mean_value * 2.0).max(QUANT_EPSILON);
    let scale = (dynamic_range / qmax).max(QUANT_EPSILON);
    let values = clamped
        .into_iter()
        .map(|value| (value / scale).round().clamp(0.0, qmax) as u8)
        .collect();
    ReferenceActivationBuffer::UnsignedPositive { values, scale }
}

pub fn fake_quantize_weight_ste<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    format: LowBitWeightFormat,
) -> Tensor<B, D> {
    match format {
        LowBitWeightFormat::Fp16 => tensor,
        LowBitWeightFormat::Int8 => symmetric_fake_quant_ste(tensor, 127.0),
        LowBitWeightFormat::Sign1 => binary_fake_quant_ste(tensor),
        LowBitWeightFormat::Ternary158 | LowBitWeightFormat::Packed2 => {
            ternary_absmean_fake_quant_ste(tensor)
        }
    }
}

pub fn fake_quantize_activation_ste<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    format: LowBitActivationFormat,
) -> Tensor<B, D> {
    match format {
        LowBitActivationFormat::Fp16 => tensor,
        LowBitActivationFormat::Int8 => symmetric_fake_quant_ste(tensor, 127.0),
        LowBitActivationFormat::Int4Exp => symmetric_fake_quant_ste(tensor, 7.0),
        LowBitActivationFormat::Uint8PosExp => positive_fake_quant_ste(tensor, 255.0),
    }
}

pub fn fraction_nonzero<B: Backend, const D: usize>(tensor: &Tensor<B, D>) -> f32 {
    tensor
        .clone()
        .abs()
        .greater_elem(0.0)
        .float()
        .mean()
        .into_scalar()
        .elem::<f32>()
}

fn ste_passthrough<B: Backend, const D: usize>(
    original: Tensor<B, D>,
    quantized: Tensor<B, D>,
) -> Tensor<B, D> {
    original.clone() + (quantized - original).detach()
}

fn broadcast_scalar_tensor<B: Backend, const D: usize>(scalar: Tensor<B, 1>) -> Tensor<B, D> {
    scalar.reshape([1; D])
}

fn symmetric_fake_quant_ste<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    qmax: f32,
) -> Tensor<B, D> {
    let step = tensor
        .clone()
        .abs()
        .mean()
        .mul_scalar(2.0 / qmax)
        .clamp_min(QUANT_EPSILON);
    let step_broadcast = broadcast_scalar_tensor(step.clone());
    let quantized = round_nearest(tensor.clone().div(step_broadcast.clone()))
        .clamp_min(-qmax)
        .clamp_max(qmax)
        .mul(step_broadcast);
    ste_passthrough(tensor, quantized)
}

fn positive_fake_quant_ste<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    qmax: f32,
) -> Tensor<B, D> {
    let clamped = tensor.clone().clamp_min(0.0);
    let step = clamped
        .clone()
        .mean()
        .mul_scalar(2.0 / qmax)
        .clamp_min(QUANT_EPSILON);
    let step_broadcast = broadcast_scalar_tensor(step.clone());
    let quantized = round_nearest(clamped.div(step_broadcast.clone()))
        .clamp_min(0.0)
        .clamp_max(qmax)
        .mul(step_broadcast);
    ste_passthrough(tensor, quantized)
}

fn binary_fake_quant_ste<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let scale = tensor.clone().abs().mean().clamp_min(QUANT_EPSILON);
    let scale_broadcast = broadcast_scalar_tensor(scale);
    let sign = tensor
        .clone()
        .greater_equal_elem(0.0)
        .float()
        .mul_scalar(2.0)
        .sub_scalar(1.0);
    ste_passthrough(tensor, sign.mul(scale_broadcast))
}

fn ternary_absmean_fake_quant_ste<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
) -> Tensor<B, D> {
    let scale = tensor.clone().abs().mean().clamp_min(QUANT_EPSILON);
    let scale_broadcast = broadcast_scalar_tensor(scale);
    let active = tensor
        .clone()
        .abs()
        .div(scale_broadcast.clone())
        .greater_equal_elem(1.0)
        .float();
    let sign = tensor
        .clone()
        .greater_equal_elem(0.0)
        .float()
        .mul_scalar(2.0)
        .sub_scalar(1.0);
    ste_passthrough(tensor, sign.mul(active).mul(scale_broadcast))
}

fn round_nearest<B: Backend, const D: usize>(tensor: Tensor<B, D>) -> Tensor<B, D> {
    let sign = tensor.clone().sign();
    let magnitude = tensor.abs().add_scalar(0.5).floor();
    sign.mul(magnitude)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::experimental::bitnet_reference::pack_weight_artifact_from_format;
    use burn::tensor::TensorData;
    use burn_ndarray::NdArray;

    type Backend = NdArray<f32>;

    #[derive(Clone, Copy, Debug)]
    struct RhoRoundTripMetrics {
        mean_abs_error: f32,
        max_abs_error: f32,
        compression_ratio: f32,
    }

    fn synthetic_rho() -> Tensor<Backend, 4> {
        let device = Default::default();
        Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..192)
                    .map(|index| ((index as f32 * 0.17).sin() * 3.0) + (index % 7) as f32 * 0.1)
                    .collect::<Vec<_>>(),
                [1, 2, 8, 12],
            ),
            &device,
        )
    }

    fn rho_round_trip_metrics(compression: RhoCompressionConfig) -> RhoRoundTripMetrics {
        let device = Default::default();
        let rho = synthetic_rho();
        let packed = pack_rho_block_state(&rho, compression);
        let restored = unpack_rho_block_state::<Backend>(&packed, &device);
        let original = rho.into_data().to_vec::<f32>().expect("original vec");
        let restored = restored.into_data().to_vec::<f32>().expect("restored vec");
        let max_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let mean_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / original.len() as f32;
        RhoRoundTripMetrics {
            mean_abs_error,
            max_abs_error,
            compression_ratio: (packed.packed.len() as f32
                + (packed.scales.len() * core::mem::size_of::<f32>()) as f32)
                / (original.len() * core::mem::size_of::<f32>()) as f32,
        }
    }

    fn rho_local_snapshot(
        original: &[f32],
        reconstructed: &[f32],
        compressed_bytes: u64,
    ) -> RhoCompressionStatsSnapshot {
        let mean_abs_error = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs() as f64)
            .sum::<f64>()
            / original.len().max(1) as f64;
        let max_abs_error = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let original_rms = (original
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum::<f64>()
            / original.len().max(1) as f64)
            .sqrt();
        let reconstructed_rms = (reconstructed
            .iter()
            .map(|value| {
                let value = f64::from(*value);
                value * value
            })
            .sum::<f64>()
            / reconstructed.len().max(1) as f64)
            .sqrt();
        RhoCompressionStatsSnapshot {
            calls: 1,
            total_original_rms: original_rms,
            total_reconstructed_rms: reconstructed_rms,
            total_mean_abs_error: mean_abs_error,
            max_abs_error,
            total_original_bytes: (original.len() * core::mem::size_of::<f32>()) as u64,
            total_compressed_bytes: compressed_bytes,
        }
    }

    #[test]
    fn projection_plan_maps_repo_targets_to_runtime_roles() {
        let plan = LowBitProjectionPlan::from_config(&LowBitQuantizationConfig {
            enable: true,
            weight_format: LowBitWeightFormat::Ternary158,
            act_format: LowBitActivationFormat::Int8,
            target_modules: vec![LowBitTargetModule::Encoder, LowBitTargetModule::DecoderY],
            decoder_x_mode: LowBitWeightFormat::Int8,
            encoder_mode: Some(LowBitWeightFormat::Int8),
            ..Default::default()
        });

        assert_eq!(plan.x_weight_format, None);
        assert_eq!(plan.y_weight_format, Some(LowBitWeightFormat::Ternary158));
        assert_eq!(plan.residual_weight_format, Some(LowBitWeightFormat::Int8));
    }

    #[test]
    fn ternary_weight_fake_quant_preserves_finite_output() {
        let device = Default::default();
        let weights = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![0.1, -0.4, 0.8, -1.2], [2, 2]),
            &device,
        );
        let quantized = fake_quantize_weight_ste(weights, LowBitWeightFormat::Ternary158);
        let values = quantized.into_data().to_vec::<f32>().expect("f32 data");
        assert!(values.iter().all(|value| value.is_finite()));
    }

    #[test]
    fn positive_activation_fake_quant_stays_nonnegative() {
        let device = Default::default();
        let activations = Tensor::<Backend, 2>::from_data(
            TensorData::new(vec![0.0, 0.2, 1.3, 4.7], [2, 2]),
            &device,
        );
        let quantized =
            fake_quantize_activation_ste(activations, LowBitActivationFormat::Uint8PosExp);
        let values = quantized.into_data().to_vec::<f32>().expect("f32 data");
        assert!(values.iter().all(|value| *value >= 0.0));
    }

    #[test]
    fn training_activation_quantization_reuses_cached_scale_after_warmup() {
        let device = Default::default();
        let activations = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.0, 0.2, 1.3, 4.7, -0.4, 0.9, -1.1, 0.3, 0.7, -0.8, 0.5, -0.2, 1.0, 0.6, -0.3,
                    0.4,
                ],
                [1, 1, 4, 4],
            ),
            &device,
        );
        let format = Some(LowBitActivationFormat::Int8);
        let cache_kind = Some("unit_test_training_cache");

        let (_, first_scale, first_hit) =
            quantize_training_activation_codes_tensor_4d(&activations, format, cache_kind)
                .expect("first quantization");
        let (_, second_scale, second_hit) =
            quantize_training_activation_codes_tensor_4d(&activations, format, cache_kind)
                .expect("second quantization");

        assert!(
            !first_hit,
            "first training quantization should warm the cache"
        );
        assert!(
            second_hit,
            "second training quantization should hit the cache"
        );
        assert!(
            (first_scale - second_scale).abs() <= 1.0e-8,
            "cached training scale drifted: first={first_scale} second={second_scale}"
        );
    }

    #[test]
    fn training_weight_quantization_reuses_cached_scale_after_warmup() {
        let device = Default::default();
        let weights = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                vec![
                    -0.4, 0.1, 0.9, -1.2, 0.3, -0.7, 1.1, -0.2, 0.5, -0.6, 0.8, -0.9,
                ],
                [3, 4],
            ),
            &device,
        );
        let cache_kind = Some("unit_test_weight_cache");

        let (_, first_scale, first_hit) = weight_codes_tensor_from_float_values(
            &weights,
            LowBitWeightFormat::Ternary158,
            cache_kind,
        )
        .expect("first weight quantization");
        let (_, second_scale, second_hit) = weight_codes_tensor_from_float_values(
            &weights,
            LowBitWeightFormat::Ternary158,
            cache_kind,
        )
        .expect("second weight quantization");

        assert!(
            !first_hit,
            "first weight quantization should warm the cache"
        );
        assert!(
            second_hit,
            "second weight quantization should hit the cache"
        );
        assert!(
            (first_scale - second_scale).abs() <= 1.0e-8,
            "cached weight scale drifted: first={first_scale} second={second_scale}"
        );
    }

    #[test]
    fn packed_lowrank_projection_reference_matches_fake_quant_matmul() {
        let device = Default::default();
        let input = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.25, -0.75, 1.0, 0.5, -0.1, 0.2, 0.4, -0.6, 0.8, 0.3, -0.9, 0.7,
                ],
                [1, 1, 3, 4],
            ),
            &device,
        );
        let weight = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    0.2, -0.7, 0.0, 1.1, -0.8, 0.6, 0.5, -0.3, 0.9, -1.2, 0.4, 0.1,
                ],
                [1, 1, 4, 3],
            ),
            &device,
        );
        let artifact = pack_weight_artifact_from_format(
            &weight
                .clone()
                .reshape([1, 4, 3])
                .into_data()
                .to_vec::<f32>()
                .expect("artifact weight vec"),
            &[1, 4, 3],
            LowBitWeightFormat::Ternary158,
        )
        .expect("packed artifact");

        let expected =
            fake_quantize_activation_ste(input.clone(), LowBitActivationFormat::Int8).matmul(
                fake_quantize_weight_ste(weight, LowBitWeightFormat::Ternary158),
            );
        let actual = packed_lowrank_projection_reference(
            input,
            &artifact,
            Some(LowBitActivationFormat::Int8),
            3,
        );

        let expected = expected.into_data().to_vec::<f32>().expect("expected vec");
        let actual = actual.into_data().to_vec::<f32>().expect("actual vec");
        assert_eq!(expected.len(), actual.len());
        for (index, (lhs, rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1.0e-6,
                "packed low-rank projection mismatch at {index}: lhs={lhs} rhs={rhs}"
            );
        }
    }

    #[test]
    fn packed_decoder_tail_reference_matches_dequantized_tail_projection() {
        let device = Default::default();
        let y_neuron = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![0.5, 0.0, 1.0, 0.0, 0.2, 0.4, 0.8, 0.6, 0.3, 0.1, 0.0, 0.7],
                [1, 2, 2, 3],
            ),
            &device,
        );
        let decoder = Tensor::<Backend, 2>::from_data(
            TensorData::new(
                vec![
                    0.3, -0.1, 0.2, 0.7, -0.4, 0.6, 0.5, -0.2, 0.1, 0.8, -0.5, 0.4, 0.9, 0.2, -0.3,
                    0.0, 0.6, -0.7,
                ],
                [6, 3],
            ),
            &device,
        );
        let artifact = pack_weight_artifact_from_format(
            &decoder
                .clone()
                .into_data()
                .to_vec::<f32>()
                .expect("decoder vec"),
            &[6, 3],
            LowBitWeightFormat::Ternary158,
        )
        .expect("packed decoder");
        let quantized_decoder = fake_quantize_weight_ste(decoder, LowBitWeightFormat::Ternary158);
        let expected = y_neuron
            .clone()
            .swap_dims(1, 2)
            .reshape([2, 6])
            .matmul(quantized_decoder)
            .reshape([1, 1, 2, 3]);
        let actual = packed_decoder_tail_reference(y_neuron, &artifact, None);
        let expected = expected.into_data().to_vec::<f32>().expect("expected vec");
        let actual = actual.into_data().to_vec::<f32>().expect("actual vec");
        assert_eq!(expected.len(), actual.len());
        for (index, (lhs, rhs)) in expected.iter().zip(actual.iter()).enumerate() {
            assert!(
                (lhs - rhs).abs() <= 1.0e-6,
                "packed decoder tail mismatch at {index}: lhs={lhs} rhs={rhs}"
            );
        }
    }

    #[test]
    fn rho_int8_block_round_trip_keeps_error_bounded() {
        let device = Default::default();
        let rho = synthetic_rho();

        let packed = pack_rho_block_state(&rho, RhoCompressionConfig::Int8BlockExp);
        let restored = unpack_rho_block_state::<Backend>(&packed, &device);
        let original = rho.into_data().to_vec::<f32>().expect("original vec");
        let restored = restored.into_data().to_vec::<f32>().expect("restored vec");
        let max_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let mean_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / original.len() as f32;
        let snapshot = rho_local_snapshot(
            &original,
            &restored,
            packed.packed.len() as u64 + (packed.scales.len() * core::mem::size_of::<f32>()) as u64,
        );

        assert_eq!(packed.logical_shape, [1, 2, 8, 12]);
        assert!(
            rho_compression_snapshot_passes_gate(&snapshot, &RhoCompressionQualityGate::default()),
            "expected rho int8-block compression snapshot to pass default quality gate: {snapshot:?}"
        );
        assert!(
            mean_abs_error <= 0.02,
            "mean abs error too large: {mean_abs_error}"
        );
        assert!(
            max_abs_error <= 0.05,
            "max abs error too large: {max_abs_error}"
        );
        assert!(
            snapshot.compression_ratio().expect("compression ratio") < 0.5,
            "expected int8 block rho compression to beat dense fp32 storage"
        );
    }

    #[test]
    fn rho_int8_block_device_round_trip_keeps_error_bounded() {
        let device = Default::default();
        let rho = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                (0..192)
                    .map(|index| ((index as f32 * 0.13).sin() * 2.5) + (index % 11) as f32 * 0.07)
                    .collect::<Vec<_>>(),
                [1, 2, 8, 12],
            ),
            &device,
        );

        let packed = pack_rho_int8_block_state_device(&rho);
        let restored = unpack_rho_int8_block_state_device(&packed);
        let original = rho.into_data().to_vec::<f32>().expect("original vec");
        let restored = restored.into_data().to_vec::<f32>().expect("restored vec");
        let max_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        let mean_abs_error = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .sum::<f32>()
            / original.len() as f32;
        let packed_values = packed
            .packed
            .clone()
            .into_data()
            .convert::<i64>()
            .into_vec::<i64>()
            .expect("packed rho values");
        let scale_values = packed
            .scales
            .clone()
            .into_data()
            .convert::<f32>()
            .into_vec::<f32>()
            .expect("rho scales");
        let snapshot = rho_local_snapshot(
            &original,
            &restored,
            packed_values.len() as u64 + (scale_values.len() * core::mem::size_of::<f32>()) as u64,
        );

        assert_eq!(packed.logical_shape, [1, 2, 8, 12]);
        assert!(mean_abs_error <= 0.02);
        assert!(max_abs_error <= 0.05);
        assert!(rho_compression_snapshot_passes_gate(
            &snapshot,
            &RhoCompressionQualityGate::default()
        ));
    }

    #[test]
    fn rho_binary_and_ternary_block_trade_accuracy_for_size() {
        let int8 = rho_round_trip_metrics(RhoCompressionConfig::Int8BlockExp);
        let ternary = rho_round_trip_metrics(RhoCompressionConfig::TernaryBlockExp);
        let binary = rho_round_trip_metrics(RhoCompressionConfig::BinaryBlockExp);

        println!(
            "rho block tradeoff: int8 mean={:.5} max={:.5} ratio={:.5}; ternary mean={:.5} max={:.5} ratio={:.5}; binary mean={:.5} max={:.5} ratio={:.5}",
            int8.mean_abs_error,
            int8.max_abs_error,
            int8.compression_ratio,
            ternary.mean_abs_error,
            ternary.max_abs_error,
            ternary.compression_ratio,
            binary.mean_abs_error,
            binary.max_abs_error,
            binary.compression_ratio,
        );

        assert!(int8.mean_abs_error <= ternary.mean_abs_error);
        assert!(int8.mean_abs_error <= binary.mean_abs_error);
        assert!(int8.max_abs_error <= ternary.max_abs_error);
        assert!(int8.max_abs_error <= binary.max_abs_error);
        assert!(int8.compression_ratio >= ternary.compression_ratio);
        assert!(ternary.compression_ratio >= binary.compression_ratio);
    }

    #[test]
    fn low_bit_kernel_plan_uses_packed_reference_when_artifacts_exist() {
        let artifact = PackedWeightArtifact {
            encoding: crate::experimental::bitnet_reference::PackedWeightEncoding::Ternary2,
            logical_shape: vec![1, 4, 3],
            scale: 0.25,
            packed: vec![0u8; 6],
            len: 12,
        };
        let config = LowBitQuantizationConfig {
            enable: true,
            inference_mode: LowBitInferenceMode::OfflinePack,
            ..Default::default()
        };
        let plan = resolve_low_bit_kernel_plan::<Backend>(
            &config,
            PackedLowBitProjectionArtifacts {
                x: Some(&artifact),
                ..Default::default()
            },
        );
        assert_eq!(plan.runtime, LowBitKernelRuntimeKind::PackedNativeInference);
        assert_eq!(plan.fallback_reason, None);
        assert!(plan.capabilities.packed_reference_supported);
        assert!(plan.capabilities.native_projection_supported);
    }

    #[test]
    fn low_bit_kernel_runtime_kind_has_stable_strings() {
        assert_eq!(
            LowBitKernelRuntimeKind::FakeQuantReference.as_str(),
            "fake_quant_reference"
        );
        assert_eq!(
            LowBitKernelRuntimeKind::PackedReference.as_str(),
            "packed_reference"
        );
        assert_eq!(
            LowBitKernelRuntimeKind::PackedNativeInference.as_str(),
            "packed_native_inference"
        );
        assert_eq!(
            LowBitKernelRuntimeKind::PackedNativeTrainingForward.as_str(),
            "packed_native_training_forward"
        );
    }

    #[test]
    fn low_bit_backend_name_capabilities_default_to_reference_only() {
        let caps = low_bit_kernel_capabilities_for_backend_name("cuda");
        assert!(caps.packed_reference_supported);
        assert!(caps.any_native_supported());
    }

    #[test]
    fn low_bit_kernel_plan_falls_back_safely_without_packed_artifacts() {
        let config = LowBitQuantizationConfig {
            enable: true,
            inference_mode: LowBitInferenceMode::OfflinePack,
            ..Default::default()
        };
        let plan = resolve_low_bit_kernel_plan::<Backend>(
            &config,
            PackedLowBitProjectionArtifacts::default(),
        );
        assert_eq!(plan.runtime, LowBitKernelRuntimeKind::FakeQuantReference);
        assert_eq!(
            plan.fallback_reason,
            Some(LowBitKernelFallbackReason::MissingPackedArtifacts)
        );
    }

    #[test]
    fn low_bit_kernel_plan_uses_native_training_forward_for_train_kernel_mode() {
        let config = LowBitQuantizationConfig {
            enable: true,
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            inference_mode: LowBitInferenceMode::RuntimeFakeQuant,
            target_modules: vec![LowBitTargetModule::DecoderY],
            ..Default::default()
        };
        let plan = resolve_low_bit_kernel_plan::<Backend>(
            &config,
            PackedLowBitProjectionArtifacts::default(),
        );
        assert_eq!(
            plan.runtime,
            LowBitKernelRuntimeKind::PackedNativeTrainingForward
        );
        assert!(plan.capabilities.native_projection_supported);
    }

    #[test]
    fn low_bit_kernel_plan_uses_native_training_forward_for_tri_matrix_hybrid_recipe() {
        let config = LowBitQuantizationConfig {
            enable: true,
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            inference_mode: LowBitInferenceMode::RuntimeFakeQuant,
            weight_format: LowBitWeightFormat::Ternary158,
            target_modules: vec![
                LowBitTargetModule::Encoder,
                LowBitTargetModule::DecoderX,
                LowBitTargetModule::DecoderY,
            ],
            decoder_x_mode: LowBitWeightFormat::Sign1,
            encoder_mode: Some(LowBitWeightFormat::Int8),
            ..Default::default()
        };
        let plan = resolve_low_bit_kernel_plan::<Backend>(
            &config,
            PackedLowBitProjectionArtifacts::default(),
        );
        assert_eq!(
            plan.runtime,
            LowBitKernelRuntimeKind::PackedNativeTrainingForward
        );
    }

    #[test]
    fn low_bit_kernel_plan_avoids_native_training_forward_when_decoder_x_ternary_is_targeted() {
        let config = LowBitQuantizationConfig {
            enable: true,
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            inference_mode: LowBitInferenceMode::RuntimeFakeQuant,
            weight_format: LowBitWeightFormat::Ternary158,
            target_modules: vec![LowBitTargetModule::DecoderX, LowBitTargetModule::DecoderY],
            decoder_x_mode: LowBitWeightFormat::Ternary158,
            ..Default::default()
        };
        let plan = resolve_low_bit_kernel_plan::<Backend>(
            &config,
            PackedLowBitProjectionArtifacts::default(),
        );
        assert_eq!(plan.runtime, LowBitKernelRuntimeKind::FakeQuantReference);
    }

    #[test]
    fn autodiff_backend_name_prefers_training_runtime() {
        assert!(backend_name_prefers_training_low_bit_runtime(
            "burn_autodiff::backend::Autodiff<burn_ndarray::backend::NdArray<f32>>"
        ));
        assert!(!backend_name_prefers_training_low_bit_runtime(
            "burn_ndarray::backend::NdArray<f32>"
        ));
    }

    #[test]
    fn wgpu_backend_name_prefers_inference_cached_scale() {
        assert!(backend_name_prefers_inference_cached_scale(
            "burn_wgpu::backend::Wgpu<f32>"
        ));
        assert!(backend_name_prefers_inference_cached_scale(
            "burn_cubecl::backend::CubeBackend<burn_wgpu::WgpuRuntime, f32, i32, u32>"
        ));
        assert!(!backend_name_prefers_inference_cached_scale(
            "burn_ndarray::backend::NdArray<f32>"
        ));
    }

    #[test]
    fn low_bit_memory_estimate_is_nonzero_for_enabled_quantized_bdh() {
        let estimate = estimate_low_bit_memory_buckets(
            &LowBitQuantizationConfig {
                enable: true,
                protocol: crate::BitNetLowBitProtocol::BitnetB158,
                training_mode: crate::LowBitTrainingMode::QatSte,
                inference_mode: crate::LowBitInferenceMode::OfflinePack,
                weight_format: LowBitWeightFormat::Ternary158,
                act_format: LowBitActivationFormat::Int8,
                target_modules: vec![
                    LowBitTargetModule::Encoder,
                    LowBitTargetModule::DecoderX,
                    LowBitTargetModule::DecoderY,
                ],
                decoder_x_mode: LowBitWeightFormat::Ternary158,
                ..Default::default()
            },
            &crate::LowBitRhoConfig::default(),
            LowBitMemoryEstimateInput {
                batch_size: 4,
                time_steps: 128,
                n_layer: 4,
                n_head: 4,
                n_embd: 256,
                latent_total: 32768,
            },
        );

        assert!(estimate.master_weight_bytes > 0);
        assert!(estimate.execution_weight_bytes > 0);
        assert!(estimate.activation_shell_bytes > 0);
        assert!(estimate.rho_state_bytes > 0);
        assert!(estimate.estimated_total_bytes() >= estimate.master_weight_bytes);
    }

    #[test]
    fn rho_int8_block_estimate_reduces_rho_bucket_vs_bf16() {
        let input = LowBitMemoryEstimateInput {
            batch_size: 2,
            time_steps: 64,
            n_layer: 2,
            n_head: 4,
            n_embd: 128,
            latent_total: 8192,
        };
        let quant = LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            target_modules: vec![LowBitTargetModule::Encoder],
            ..Default::default()
        };
        let bf16 =
            estimate_low_bit_memory_buckets(&quant, &crate::LowBitRhoConfig::default(), input);
        let compressed = estimate_low_bit_memory_buckets(
            &quant,
            &crate::LowBitRhoConfig {
                compression: crate::RhoCompressionConfig::Int8BlockExp,
                ..Default::default()
            },
            input,
        );
        assert!(compressed.rho_state_bytes < bf16.rho_state_bytes);
    }

    #[test]
    fn rho_binary_and_ternary_block_estimates_reduce_rho_bucket_beyond_int8() {
        let input = LowBitMemoryEstimateInput {
            batch_size: 2,
            time_steps: 64,
            n_layer: 2,
            n_head: 4,
            n_embd: 128,
            latent_total: 8192,
        };
        let quant = LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            target_modules: vec![LowBitTargetModule::Encoder],
            ..Default::default()
        };
        let int8 = estimate_low_bit_memory_buckets(
            &quant,
            &crate::LowBitRhoConfig {
                compression: crate::RhoCompressionConfig::Int8BlockExp,
                ..Default::default()
            },
            input,
        );
        let ternary = estimate_low_bit_memory_buckets(
            &quant,
            &crate::LowBitRhoConfig {
                compression: crate::RhoCompressionConfig::TernaryBlockExp,
                ..Default::default()
            },
            input,
        );
        let binary = estimate_low_bit_memory_buckets(
            &quant,
            &crate::LowBitRhoConfig {
                compression: crate::RhoCompressionConfig::BinaryBlockExp,
                ..Default::default()
            },
            input,
        );

        assert!(ternary.rho_state_bytes < int8.rho_state_bytes);
        assert!(binary.rho_state_bytes < ternary.rho_state_bytes);
    }

    #[test]
    fn saved_activation_estimate_is_nonzero_when_enabled() {
        let estimate = estimate_low_bit_memory_buckets(
            &LowBitQuantizationConfig {
                enable: true,
                protocol: crate::BitNetLowBitProtocol::BitnetB158,
                training_mode: crate::LowBitTrainingMode::TrainKernelExp,
                target_modules: vec![LowBitTargetModule::Encoder],
                saved_activations: crate::LowBitSavedActivationConfig {
                    mode: crate::LowBitSavedActivationMode::QuantizedCacheExp,
                    format: LowBitActivationFormat::Int8,
                },
                ..Default::default()
            },
            &crate::LowBitRhoConfig::default(),
            LowBitMemoryEstimateInput {
                batch_size: 4,
                time_steps: 128,
                n_layer: 6,
                n_head: 4,
                n_embd: 192,
                latent_total: 12288,
            },
        );
        assert!(estimate.saved_activation_bytes > 0);
    }

    #[test]
    fn saved_activation_recompute_estimate_reduces_resident_bytes() {
        let input = LowBitMemoryEstimateInput {
            batch_size: 4,
            time_steps: 256,
            n_layer: 8,
            n_head: 4,
            n_embd: 256,
            latent_total: 32768,
        };
        let mut quant = LowBitQuantizationConfig {
            enable: true,
            protocol: crate::BitNetLowBitProtocol::BitnetB158,
            training_mode: crate::LowBitTrainingMode::TrainKernelExp,
            target_modules: vec![LowBitTargetModule::Encoder, LowBitTargetModule::DecoderY],
            ..Default::default()
        };
        quant.saved_activations = crate::LowBitSavedActivationConfig {
            mode: crate::LowBitSavedActivationMode::QuantizedCacheExp,
            format: LowBitActivationFormat::Int8,
        };
        let cached =
            estimate_low_bit_memory_buckets(&quant, &crate::LowBitRhoConfig::default(), input);
        quant.saved_activations.mode = crate::LowBitSavedActivationMode::QuantizedCacheRecomputeExp;
        let recompute =
            estimate_low_bit_memory_buckets(&quant, &crate::LowBitRhoConfig::default(), input);

        assert!(cached.saved_activation_bytes > recompute.saved_activation_bytes);
        assert!(recompute.workspace_bytes > cached.workspace_bytes);
    }

    #[test]
    fn saved_activation_inventory_reports_expected_training_tensors() {
        let inventory = build_low_bit_saved_activation_inventory(
            &LowBitQuantizationConfig {
                enable: true,
                protocol: crate::BitNetLowBitProtocol::BitnetB158,
                training_mode: crate::LowBitTrainingMode::TrainKernelExp,
                target_modules: vec![
                    LowBitTargetModule::Encoder,
                    LowBitTargetModule::DecoderX,
                    LowBitTargetModule::DecoderY,
                ],
                saved_activations: crate::LowBitSavedActivationConfig {
                    mode: crate::LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
                    format: LowBitActivationFormat::Int8,
                },
                ..Default::default()
            },
            LowBitMemoryEstimateInput {
                batch_size: 2,
                time_steps: 64,
                n_layer: 4,
                n_head: 4,
                n_embd: 128,
                latent_total: 8192,
            },
        )
        .expect("expected saved activation inventory");

        assert!(inventory.requires_rho_window_anchor);
        assert_eq!(inventory.tensors.len(), 3);
        assert_eq!(inventory.tensors[0].name, "x_projection_input");
        assert_eq!(inventory.tensors[1].name, "y_projection_input");
        assert_eq!(inventory.tensors[2].name, "residual_tail_input");
        assert!(
            inventory
                .tensors
                .iter()
                .all(|entry| entry.estimated_bytes > 0)
        );
    }

    #[test]
    fn saved_activation_cache_int8_round_trip_keeps_error_bounded() {
        let device = Default::default();
        let tensor = Tensor::<Backend, 4>::from_data(
            TensorData::new(
                vec![
                    -1.5, -0.25, 0.0, 0.4, 0.9, 1.2, -0.8, 0.15, 0.33, -0.6, 0.75, 1.8,
                ],
                [1, 2, 2, 3],
            ),
            &device,
        );
        let packed = pack_saved_activation_state(&tensor, LowBitActivationFormat::Int8);
        let restored = unpack_saved_activation_state::<Backend, 4>(&packed, &device);
        let original = tensor.into_data().to_vec::<f32>().expect("original vec");
        let restored = restored.into_data().to_vec::<f32>().expect("restored vec");
        let max_abs = original
            .iter()
            .zip(restored.iter())
            .map(|(lhs, rhs)| (lhs - rhs).abs())
            .fold(0.0f32, f32::max);
        assert!(packed.estimated_bytes > 0);
        assert!(max_abs <= 0.4, "max_abs={max_abs}");
    }

    #[test]
    fn saved_activation_cache_uint8_positive_stays_nonnegative() {
        let device = Default::default();
        let tensor = Tensor::<Backend, 4>::from_data(
            TensorData::new(vec![-1.0, 0.0, 0.5, 1.0], [1, 1, 2, 2]),
            &device,
        );
        let packed = pack_saved_activation_state(&tensor, LowBitActivationFormat::Uint8PosExp);
        let restored = unpack_saved_activation_state::<Backend, 4>(&packed, &device);
        let restored = restored.into_data().to_vec::<f32>().expect("restored vec");
        assert!(restored.iter().all(|value| *value >= 0.0));
    }
}
