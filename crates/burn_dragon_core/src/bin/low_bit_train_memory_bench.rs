use burn::nn::DropoutConfig;
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{Tensor, TensorData};
use burn_autodiff::Autodiff;
use burn_cubecl::CubeBackend;
use burn_cubecl::cubecl::Runtime;
use burn_dragon_core::{
    BlockPattern1d, LowBitActivationFormat, LowBitKernelRuntimeKind, LowBitProjectionPlan,
    LowBitSavedActivationConfig, LowBitSavedActivationMode, LowBitWeightFormat,
    PackedLowBitProjectionArtifacts, lowrank_residual_step_next,
};
use burn_dragon_kernel::api::projection::LowrankGradInputExecutor;
use burn_wgpu::{RuntimeOptions, WgpuRuntime, graphics};

#[cfg(feature = "cuda")]
use burn_cubecl::cubecl::cuda::CudaRuntime;

type WgpuInnerBackend = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuAutodiffBackend = Autodiff<WgpuInnerBackend>;

#[cfg(feature = "cuda")]
type CudaInnerBackend = CubeBackend<CudaRuntime, f32, i32, u8>;
#[cfg(feature = "cuda")]
type CudaAutodiffBackend = Autodiff<CudaInnerBackend>;

#[derive(Clone, Copy)]
struct MemorySnapshot {
    reserved: u64,
    in_use: u64,
}

fn deterministic_values(len: usize, offset: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| {
            (((idx as f32) * 0.173) + offset).sin() * 0.5
                + (((idx as f32) * 0.117) + offset).cos() * 0.25
        })
        .collect()
}

fn init_wgpu(device: &<WgpuAutodiffBackend as BackendTrait>::Device) {
    static INIT: std::sync::Once = std::sync::Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

fn memory_snapshot_wgpu(device: &<WgpuAutodiffBackend as BackendTrait>::Device) -> MemorySnapshot {
    let usage = <WgpuRuntime as Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

#[cfg(feature = "cuda")]
fn memory_snapshot_cuda(device: &<CudaAutodiffBackend as BackendTrait>::Device) -> MemorySnapshot {
    let usage = <CudaRuntime as Runtime>::client(device).memory_usage();
    MemorySnapshot {
        reserved: usage.bytes_reserved,
        in_use: usage.bytes_in_use,
    }
}

fn low_bit_plan() -> LowBitProjectionPlan {
    LowBitProjectionPlan {
        x_weight_format: Some(LowBitWeightFormat::Int8),
        x_activation_format: Some(LowBitActivationFormat::Int8),
        y_weight_format: Some(LowBitWeightFormat::Int8),
        y_activation_format: Some(LowBitActivationFormat::Int8),
        residual_weight_format: Some(LowBitWeightFormat::Int8),
        residual_activation_format: Some(LowBitActivationFormat::Int8),
    }
}

fn mode_config(mode: LowBitSavedActivationMode) -> LowBitSavedActivationConfig {
    LowBitSavedActivationConfig {
        mode,
        format: LowBitActivationFormat::Int8,
    }
}

fn run_wgpu_case(mode: LowBitSavedActivationMode) {
    let device = <WgpuAutodiffBackend as BackendTrait>::Device::default();
    init_wgpu(&device);

    let batch = 4usize;
    let time = 128usize;
    let embd = 256usize;
    let heads = 4usize;
    let latent = embd;

    let current = Tensor::<WgpuAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(batch * 1 * time * embd, 0.2),
            [batch, 1, time, embd],
        ),
        &device,
    )
    .require_grad();
    let encoder = Tensor::<WgpuAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(1 * heads * embd * latent, 0.7),
            [1, heads, embd, latent],
        ),
        &device,
    )
    .require_grad();
    let encoder_v = Tensor::<WgpuAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(1 * heads * embd * latent, 1.1),
            [1, heads, embd, latent],
        ),
        &device,
    )
    .require_grad();
    let decoder = Tensor::<WgpuAutodiffBackend, 2>::from_data(
        TensorData::new(
            deterministic_values(heads * latent * embd, 1.6),
            [heads * latent, embd],
        ),
        &device,
    )
    .require_grad();
    let dropout = DropoutConfig::new(0.0).init();

    WgpuInnerBackend::memory_cleanup(&device);
    let before = memory_snapshot_wgpu(&device);
    let output = lowrank_residual_step_next(
        current.clone(),
        encoder.clone(),
        encoder_v.clone(),
        decoder.clone(),
        &dropout,
        false,
        false,
        0.0,
        false,
        low_bit_plan(),
        mode_config(mode),
        PackedLowBitProjectionArtifacts {
            runtime: LowBitKernelRuntimeKind::PackedNativeTrainingForward,
            ..Default::default()
        },
        &BlockPattern1d::dense(latent),
        LowrankGradInputExecutor::Auto,
        None,
        |query, current| query + current,
        |values| values.clamp_min(0.0),
        |values| values,
    );
    let after_forward = memory_snapshot_wgpu(&device);
    let grads = output.sum().backward();
    let _ = encoder.grad(&grads);
    let _ = decoder.grad(&grads);
    let after_backward = memory_snapshot_wgpu(&device);

    println!(
        "train_memory backend=wgpu mode={} reserved_before={} in_use_before={} reserved_after_forward={} in_use_after_forward={} reserved_after_backward={} in_use_after_backward={}",
        mode.as_str(),
        before.reserved,
        before.in_use,
        after_forward.reserved,
        after_forward.in_use,
        after_backward.reserved,
        after_backward.in_use,
    );
}

#[cfg(feature = "cuda")]
fn run_cuda_case(mode: LowBitSavedActivationMode) {
    let device = <CudaAutodiffBackend as BackendTrait>::Device::default();

    let batch = 4usize;
    let time = 128usize;
    let embd = 256usize;
    let heads = 4usize;
    let latent = embd;

    let current = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(batch * 1 * time * embd, 0.2),
            [batch, 1, time, embd],
        ),
        &device,
    )
    .require_grad();
    let encoder = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(1 * heads * embd * latent, 0.7),
            [1, heads, embd, latent],
        ),
        &device,
    )
    .require_grad();
    let encoder_v = Tensor::<CudaAutodiffBackend, 4>::from_data(
        TensorData::new(
            deterministic_values(1 * heads * embd * latent, 1.1),
            [1, heads, embd, latent],
        ),
        &device,
    )
    .require_grad();
    let decoder = Tensor::<CudaAutodiffBackend, 2>::from_data(
        TensorData::new(
            deterministic_values(heads * latent * embd, 1.6),
            [heads * latent, embd],
        ),
        &device,
    )
    .require_grad();
    let dropout = DropoutConfig::new(0.0).init();

    CudaInnerBackend::memory_cleanup(&device);
    let before = memory_snapshot_cuda(&device);
    let output = lowrank_residual_step_next(
        current.clone(),
        encoder.clone(),
        encoder_v.clone(),
        decoder.clone(),
        &dropout,
        false,
        false,
        0.0,
        false,
        low_bit_plan(),
        mode_config(mode),
        PackedLowBitProjectionArtifacts {
            runtime: LowBitKernelRuntimeKind::PackedNativeTrainingForward,
            ..Default::default()
        },
        &BlockPattern1d::dense(latent),
        LowrankGradInputExecutor::Auto,
        None,
        |query, current| query + current,
        |values| values.clamp_min(0.0),
        |values| values,
    );
    let after_forward = memory_snapshot_cuda(&device);
    let grads = output.sum().backward();
    let _ = encoder.grad(&grads);
    let _ = decoder.grad(&grads);
    let after_backward = memory_snapshot_cuda(&device);

    println!(
        "train_memory backend=cuda mode={} reserved_before={} in_use_before={} reserved_after_forward={} in_use_after_forward={} reserved_after_backward={} in_use_after_backward={}",
        mode.as_str(),
        before.reserved,
        before.in_use,
        after_forward.reserved,
        after_forward.in_use,
        after_backward.reserved,
        after_backward.in_use,
    );
}

fn main() {
    let backend = std::env::args()
        .skip(1)
        .find_map(|arg| arg.strip_prefix("--backend=").map(str::to_string))
        .unwrap_or_else(|| "wgpu".to_string());

    let modes = [
        LowBitSavedActivationMode::Disabled,
        LowBitSavedActivationMode::QuantizedCacheExp,
        LowBitSavedActivationMode::QuantizedCacheRecomputeExp,
    ];

    match backend.as_str() {
        "wgpu" => {
            for mode in modes {
                run_wgpu_case(mode);
            }
        }
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                for mode in modes {
                    run_cuda_case(mode);
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                eprintln!("low_bit_train_memory_bench backend=cuda requires --features cuda");
                std::process::exit(1);
            }
        }
        other => {
            eprintln!("unsupported backend: {other}");
            std::process::exit(2);
        }
    }
}
