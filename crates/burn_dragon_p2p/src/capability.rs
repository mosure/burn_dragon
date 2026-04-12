#[cfg(all(not(feature = "native"), feature = "wasm-peer"))]
use burn_dragon_core::{BDHConfig, SequenceMemorySystem, SequenceTrainingExecutor};
#[cfg(feature = "native")]
use burn_dragon_language::{BDHConfig, SequenceMemorySystem, SequenceTrainingExecutor};
use burn_p2p::WorkloadTrainingBudget;
#[cfg(feature = "native")]
use burn_p2p::burn::BurnTarget;
#[cfg(feature = "wasm-ui")]
use burn_p2p_browser::{
    BrowserAppTarget, BrowserCapabilityReport, BrowserGpuSupport, BrowserRuntimeRole,
    BrowserWorkerSupport,
};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use js_sys::Reflect;
use serde::{Deserialize, Serialize};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use wasm_bindgen::JsValue;

#[cfg(feature = "wasm-peer")]
use crate::config::DragonBrowserTrainingConfig;
use crate::config::{DragonCapabilityPolicy, DragonNativeTarget};

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const GIB: u64 = 1024 * 1024 * 1024;
const MIB: u64 = 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DragonCapabilityClass {
    NativeCpu,
    NativeWgpu,
    NativeCuda,
    BrowserCpu,
    BrowserWgpu,
}

impl DragonCapabilityClass {
    pub fn from_backend_label(backend_label: &str) -> Self {
        match backend_label {
            "cpu" | "ndarray" => Self::NativeCpu,
            "cuda" => Self::NativeCuda,
            "wgpu" => Self::NativeWgpu,
            _ => Self::NativeWgpu,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonTrainingFootprint {
    pub estimated_parameter_bytes: u64,
    pub estimated_optimizer_state_bytes: u64,
    pub estimated_activation_bytes: u64,
    pub estimated_training_bytes: u64,
    pub estimated_checkpoint_bytes: u64,
    pub estimated_shard_bytes: u64,
    pub estimated_tokens_per_second: f64,
}

impl DragonTrainingFootprint {
    pub fn browser_budget(&self, requested_batch_size: usize) -> WorkloadTrainingBudget {
        WorkloadTrainingBudget {
            max_window_secs: 30,
            max_checkpoint_bytes: self.estimated_checkpoint_bytes,
            max_shard_bytes: self.estimated_shard_bytes,
            requires_webgpu: true,
            max_batch_size: Some(requested_batch_size.max(1) as u32),
            precision: Some(burn_p2p::Precision::Fp16),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonNativeTargetDecision {
    pub requested_target: DragonNativeTarget,
    pub effective_target: DragonNativeTarget,
    pub can_train: bool,
    pub trainer_memory_budget_bytes: Option<u64>,
    pub downgrade_reason: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonNativeCapabilityAssessment {
    pub experiment_kind: crate::config::DragonExperimentKind,
    pub backend_label: String,
    pub model_config: BDHConfig,
    pub batch_size: usize,
    pub block_size: usize,
    pub footprint: DragonTrainingFootprint,
    pub target_decision: DragonNativeTargetDecision,
}

#[cfg(feature = "native")]
impl DragonNativeTargetDecision {
    pub fn burn_target(&self) -> BurnTarget {
        match self.effective_target {
            DragonNativeTarget::Auto | DragonNativeTarget::Trainer => BurnTarget::Trainer,
            DragonNativeTarget::Validator => BurnTarget::Validator,
            DragonNativeTarget::Reducer => BurnTarget::Reducer,
        }
    }
}

#[cfg(feature = "wasm-ui")]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DragonBrowserHostCapabilityProbe {
    pub navigator_gpu_exposed: bool,
    pub worker_gpu_exposed: bool,
    pub dedicated_worker_exposed: bool,
    pub persistent_storage_exposed: bool,
    pub web_transport_exposed: bool,
    pub web_rtc_exposed: bool,
    pub system_memory_bytes: Option<u64>,
}

#[cfg(feature = "wasm-ui")]
#[derive(Clone, Debug, PartialEq)]
pub struct DragonBrowserCapabilityDecision {
    pub capability: BrowserCapabilityReport,
    pub connect_target: BrowserAppTarget,
    pub can_train: bool,
    pub trainer_memory_budget_bytes: Option<u64>,
    pub training_budget: Option<WorkloadTrainingBudget>,
    pub footprint: Option<DragonTrainingFootprint>,
    pub downgrade_reason: Option<String>,
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn detect_browser_host_capabilities() -> DragonBrowserHostCapabilityProbe {
    let Some(window) = web_sys::window() else {
        return DragonBrowserHostCapabilityProbe::default();
    };
    let window_value = JsValue::from(window.clone());
    let navigator = Reflect::get(&window_value, &JsValue::from_str("navigator")).ok();
    let navigator_ref = navigator.as_ref();
    let has_gpu = navigator_ref
        .and_then(|navigator| Reflect::get(navigator, &JsValue::from_str("gpu")).ok())
        .is_some_and(|value| !value.is_null() && !value.is_undefined());
    let has_worker = Reflect::get(&window_value, &JsValue::from_str("Worker"))
        .ok()
        .is_some_and(|value| !value.is_null() && !value.is_undefined());
    let has_storage_manager = navigator_ref
        .and_then(|navigator| Reflect::get(navigator, &JsValue::from_str("storage")).ok())
        .is_some_and(|value| !value.is_null() && !value.is_undefined());
    let has_web_transport = Reflect::get(&window_value, &JsValue::from_str("WebTransport"))
        .ok()
        .is_some_and(|value| !value.is_null() && !value.is_undefined());
    let has_webrtc = Reflect::get(&window_value, &JsValue::from_str("RTCPeerConnection"))
        .ok()
        .is_some_and(|value| !value.is_null() && !value.is_undefined());
    let system_memory_bytes = navigator_ref
        .and_then(|navigator| Reflect::get(navigator, &JsValue::from_str("deviceMemory")).ok())
        .and_then(|value| value.as_f64())
        .filter(|value| value.is_finite() && *value > 0.0)
        .map(|gib| (gib * GIB as f64) as u64);

    DragonBrowserHostCapabilityProbe {
        navigator_gpu_exposed: has_gpu,
        worker_gpu_exposed: has_gpu && has_worker,
        dedicated_worker_exposed: has_worker,
        persistent_storage_exposed: has_storage_manager,
        web_transport_exposed: has_web_transport,
        web_rtc_exposed: has_webrtc,
        system_memory_bytes,
    }
}

pub fn estimate_language_training_footprint(
    model_config: &BDHConfig,
    batch_size: usize,
    block_size: usize,
    backend_class: DragonCapabilityClass,
) -> DragonTrainingFootprint {
    let embed = model_config.n_embd as u64;
    let latent_total = model_config.latent_total() as u64;
    let latent_per_head = model_config.latent_per_head() as u64;
    let layers = model_config.n_layer.max(1) as u64;
    let heads = model_config.n_head.max(1) as u64;
    let vocab = model_config.vocab_size.max(1) as u64;
    let batch = batch_size.max(1) as u64;
    let block = block_size.max(1) as u64;
    let tokens = batch * block;

    let embedding_params = 2 * vocab * embed;
    let residual_params = 2 * embed * embed;
    let projection_params = 4 * embed * embed + 6 * embed * latent_total;
    let sequence_params = match model_config.sequence_kernel.memory_system {
        SequenceMemorySystem::LinearAttention => 2 * heads * latent_per_head * embed,
        SequenceMemorySystem::Rwkv8StateSpace => 3 * embed * embed,
        SequenceMemorySystem::Mamba1SelectiveScan => 4 * embed * embed,
        SequenceMemorySystem::Mamba2StateSpaceDuality => 5 * embed * embed,
        SequenceMemorySystem::Mamba3StateSpaceDuality => {
            6 * embed * embed + 2 * embed * latent_total
        }
    };
    let parameter_count: u64 =
        embedding_params + layers * (projection_params + residual_params + sequence_params);
    let parameter_bytes = parameter_count.saturating_mul(4);

    let optimizer_state_bytes = match backend_class {
        DragonCapabilityClass::NativeCpu | DragonCapabilityClass::BrowserCpu => {
            parameter_bytes.saturating_mul(5).saturating_div(1)
        }
        DragonCapabilityClass::NativeWgpu
        | DragonCapabilityClass::NativeCuda
        | DragonCapabilityClass::BrowserWgpu => parameter_bytes.saturating_mul(9).saturating_div(2),
    };

    let activation_width = match model_config.sequence_kernel.memory_system {
        SequenceMemorySystem::LinearAttention => 8 * embed + 4 * latent_total,
        SequenceMemorySystem::Rwkv8StateSpace => 10 * embed + 4 * latent_total,
        SequenceMemorySystem::Mamba1SelectiveScan => 12 * embed + 6 * latent_total,
        SequenceMemorySystem::Mamba2StateSpaceDuality => 13 * embed + 7 * latent_total,
        SequenceMemorySystem::Mamba3StateSpaceDuality => 14 * embed + 8 * latent_total,
    };
    let executor_multiplier = match model_config.sequence_kernel.executor {
        SequenceTrainingExecutor::Reference => 2,
        SequenceTrainingExecutor::DenseScoreShortContext => 1,
    };
    let activation_bytes = tokens
        .saturating_mul(layers)
        .saturating_mul(activation_width)
        .saturating_mul(4)
        .saturating_mul(executor_multiplier);

    let training_bytes = parameter_bytes
        .saturating_add(optimizer_state_bytes)
        .saturating_add(activation_bytes);
    let checkpoint_bytes = (parameter_bytes.saturating_mul(5) / 4).clamp(16 * MIB, 512 * MIB);
    let shard_bytes = ((tokens.saturating_mul(16)).saturating_mul(8)).clamp(4 * MIB, 64 * MIB);

    let per_token_work = layers
        .saturating_mul(
            4 * embed * embed
                + 6 * embed * latent_total
                + match model_config.sequence_kernel.memory_system {
                    SequenceMemorySystem::LinearAttention => 2 * heads * block * latent_per_head,
                    SequenceMemorySystem::Rwkv8StateSpace => 3 * embed * embed,
                    SequenceMemorySystem::Mamba1SelectiveScan => 4 * embed * embed,
                    SequenceMemorySystem::Mamba2StateSpaceDuality => 5 * embed * embed,
                    SequenceMemorySystem::Mamba3StateSpaceDuality => {
                        6 * embed * embed + 2 * embed * latent_total
                    }
                },
        )
        .max(1);
    let backend_work_rate: f64 = match backend_class {
        DragonCapabilityClass::NativeCpu => 5.0e9,
        DragonCapabilityClass::BrowserCpu => 1.5e9,
        DragonCapabilityClass::BrowserWgpu => 1.1e11,
        DragonCapabilityClass::NativeWgpu => 1.5e11,
        DragonCapabilityClass::NativeCuda => 2.0e11,
    };
    let estimated_tokens_per_second =
        (backend_work_rate / per_token_work as f64).clamp(1.0, 500_000.0);

    DragonTrainingFootprint {
        estimated_parameter_bytes: parameter_bytes,
        estimated_optimizer_state_bytes: optimizer_state_bytes,
        estimated_activation_bytes: activation_bytes,
        estimated_training_bytes: training_bytes,
        estimated_checkpoint_bytes: checkpoint_bytes,
        estimated_shard_bytes: shard_bytes,
        estimated_tokens_per_second,
    }
}

pub fn decide_native_target(
    requested_target: DragonNativeTarget,
    policy: &DragonCapabilityPolicy,
    backend_class: DragonCapabilityClass,
    footprint: &DragonTrainingFootprint,
) -> DragonNativeTargetDecision {
    if matches!(
        requested_target,
        DragonNativeTarget::Validator | DragonNativeTarget::Reducer
    ) {
        return DragonNativeTargetDecision {
            requested_target,
            effective_target: requested_target,
            can_train: false,
            trainer_memory_budget_bytes: None,
            downgrade_reason: None,
        };
    }

    let trainer_memory_budget_bytes = policy.memory_budget_bytes(backend_class);
    let can_train = trainer_memory_budget_bytes
        .map(|budget| footprint.estimated_training_bytes <= budget)
        .unwrap_or(true);
    let effective_target = if can_train || !policy.allow_native_validator_fallback {
        DragonNativeTarget::Trainer
    } else {
        DragonNativeTarget::Validator
    };
    let downgrade_reason = (!can_train && policy.allow_native_validator_fallback).then(|| {
        format!(
            "estimated training footprint {} MiB exceeds {:?} MiB budget for {:?}; downgrading to validator",
            footprint.estimated_training_bytes / MIB,
            trainer_memory_budget_bytes.map(|value| value / MIB),
            backend_class
        )
    });

    DragonNativeTargetDecision {
        requested_target,
        effective_target,
        can_train,
        trainer_memory_budget_bytes,
        downgrade_reason,
    }
}

#[cfg(feature = "wasm-ui")]
pub fn decide_browser_capability(
    config: Option<&DragonBrowserTrainingConfig>,
    host: &DragonBrowserHostCapabilityProbe,
) -> DragonBrowserCapabilityDecision {
    let mut capability = BrowserCapabilityReport::default();
    capability.navigator_gpu_exposed = host.navigator_gpu_exposed;
    capability.worker_gpu_exposed = host.worker_gpu_exposed;
    capability.dedicated_worker = if host.dedicated_worker_exposed {
        BrowserWorkerSupport::DedicatedWorker
    } else {
        BrowserWorkerSupport::Unavailable("dedicated worker unavailable".into())
    };
    capability.persistent_storage_exposed = host.persistent_storage_exposed;
    capability.web_transport_exposed = host.web_transport_exposed;
    capability.web_rtc_exposed = host.web_rtc_exposed;

    let Some(config) = config else {
        capability.gpu_support = if host.navigator_gpu_exposed && host.worker_gpu_exposed {
            BrowserGpuSupport::Available
        } else {
            BrowserGpuSupport::Unavailable("webgpu unavailable".into())
        };
        capability.recommended_role = if host.dedicated_worker_exposed {
            BrowserRuntimeRole::BrowserVerifier
        } else {
            BrowserRuntimeRole::BrowserObserver
        };
        return DragonBrowserCapabilityDecision {
            capability,
            connect_target: if host.dedicated_worker_exposed {
                BrowserAppTarget::Validate
            } else {
                BrowserAppTarget::Observe
            },
            can_train: false,
            trainer_memory_budget_bytes: None,
            training_budget: None,
            footprint: None,
            downgrade_reason: None,
        };
    };

    let footprint = estimate_language_training_footprint(
        &config.model_config,
        config.batch_size,
        config.block_size,
        match config.execution_backend {
            crate::config::DragonBrowserExecutionBackend::Cpu => DragonCapabilityClass::BrowserCpu,
            crate::config::DragonBrowserExecutionBackend::Auto
            | crate::config::DragonBrowserExecutionBackend::Wgpu => {
                DragonCapabilityClass::BrowserWgpu
            }
        },
    );
    let budget = footprint.browser_budget(config.batch_size);
    capability.max_training_window_secs = budget.max_window_secs;
    capability.max_checkpoint_bytes = budget.max_checkpoint_bytes;
    capability.max_shard_bytes = budget.max_shard_bytes;

    let trainer_memory_budget_bytes = config
        .capability_policy
        .memory_budget_bytes(DragonCapabilityClass::BrowserWgpu)
        .or(host.system_memory_bytes.map(|bytes| bytes / 2));
    let gpu_ready = host.navigator_gpu_exposed && host.worker_gpu_exposed;
    let worker_ready = host.dedicated_worker_exposed;
    let can_train = gpu_ready
        && worker_ready
        && trainer_memory_budget_bytes
            .map(|budget| footprint.estimated_training_bytes <= budget)
            .unwrap_or(false);

    if can_train {
        capability.gpu_support = BrowserGpuSupport::Available;
        capability.recommended_role = BrowserRuntimeRole::BrowserTrainerWgpu;
        return DragonBrowserCapabilityDecision {
            capability,
            connect_target: BrowserAppTarget::Train,
            can_train: true,
            trainer_memory_budget_bytes,
            training_budget: Some(budget),
            footprint: Some(footprint),
            downgrade_reason: None,
        };
    }

    capability.gpu_support = if gpu_ready {
        BrowserGpuSupport::Available
    } else {
        BrowserGpuSupport::Unavailable("webgpu unavailable".into())
    };
    capability.recommended_role =
        if worker_ready && config.capability_policy.allow_browser_verifier_fallback {
            BrowserRuntimeRole::BrowserVerifier
        } else {
            BrowserRuntimeRole::BrowserObserver
        };

    let connect_target = match capability.recommended_role {
        BrowserRuntimeRole::BrowserVerifier => BrowserAppTarget::Validate,
        BrowserRuntimeRole::BrowserTrainerWgpu => BrowserAppTarget::Train,
        _ => BrowserAppTarget::Observe,
    };
    let downgrade_reason = Some(if !gpu_ready {
        "webgpu unavailable; downgrading browser peer to verifier/observer".into()
    } else if !worker_ready {
        "dedicated worker unavailable; downgrading browser peer to observer".into()
    } else {
        format!(
            "estimated training footprint {} MiB exceeds {:?} MiB browser trainer budget; downgrading to verifier",
            footprint.estimated_training_bytes / MIB,
            trainer_memory_budget_bytes.map(|value| value / MIB)
        )
    });

    DragonBrowserCapabilityDecision {
        capability,
        connect_target,
        can_train: false,
        trainer_memory_budget_bytes,
        training_budget: None,
        footprint: Some(footprint),
        downgrade_reason,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn estimated_training_footprint_scales_with_model_size() {
        let tiny = BDHConfig {
            n_layer: 2,
            n_embd: 32,
            n_head: 4,
            vocab_size: 256,
            ..BDHConfig::default()
        };
        let larger = BDHConfig {
            n_layer: 6,
            n_embd: 128,
            n_head: 8,
            vocab_size: 4096,
            ..BDHConfig::default()
        };

        let tiny_fp =
            estimate_language_training_footprint(&tiny, 2, 64, DragonCapabilityClass::NativeWgpu);
        let larger_fp = estimate_language_training_footprint(
            &larger,
            4,
            128,
            DragonCapabilityClass::NativeWgpu,
        );

        assert!(larger_fp.estimated_training_bytes > tiny_fp.estimated_training_bytes);
        assert!(larger_fp.estimated_tokens_per_second < tiny_fp.estimated_tokens_per_second);
    }
}
