use anyhow::Result;
use burn::backend::Autodiff;
use burn_ndarray::NdArray;

use crate::capability::DragonNativeCapabilityAssessment;
use crate::config::DragonExperimentKind;
use crate::config::DragonNativeAuthBundle;
use crate::config::DragonNativePeerConfig;
use crate::experiments::climbmix::prepare_climbmix_peer_for_backend;
use crate::experiments::common::{PreparedNativePeer, assess_native_peer_for_backend};
use crate::experiments::nca::prepare_nca_peer_for_backend;
pub use crate::native_runtime::{ManagedRunningNativePeer, spawn_prepared_native_peer};

pub type NativeCpuBackend = Autodiff<NdArray<f32>>;

#[cfg(feature = "wgpu")]
pub type NativeWgpuBackend = Autodiff<burn_wgpu::Wgpu<f32>>;

#[cfg(feature = "cuda")]
pub type NativeCudaBackend = Autodiff<burn_cuda::Cuda<f32>>;

pub fn assess_native_peer(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
) -> Result<DragonNativeCapabilityAssessment> {
    assess_native_peer_for_backend(native, experiment_kind, backend_label)
}

pub fn prepare_nca_native_cpu(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeCpuBackend>> {
    prepare_nca_peer_for_backend::<NativeCpuBackend>(native, "cpu", Default::default(), auth_bundle)
}

pub fn prepare_climbmix_native_cpu(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeCpuBackend>> {
    prepare_climbmix_peer_for_backend::<NativeCpuBackend>(
        native,
        "cpu",
        Default::default(),
        auth_bundle,
    )
}

#[cfg(feature = "wgpu")]
pub fn prepare_nca_native_wgpu(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeWgpuBackend>> {
    prepare_nca_peer_for_backend::<NativeWgpuBackend>(
        native,
        "wgpu",
        Default::default(),
        auth_bundle,
    )
}

#[cfg(feature = "wgpu")]
pub fn prepare_climbmix_native_wgpu(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeWgpuBackend>> {
    prepare_climbmix_peer_for_backend::<NativeWgpuBackend>(
        native,
        "wgpu",
        Default::default(),
        auth_bundle,
    )
}

#[cfg(feature = "cuda")]
pub fn prepare_nca_native_cuda(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeCudaBackend>> {
    prepare_nca_peer_for_backend::<NativeCudaBackend>(
        native,
        "cuda",
        Default::default(),
        auth_bundle,
    )
}

#[cfg(feature = "cuda")]
pub fn prepare_climbmix_native_cuda(
    native: &DragonNativePeerConfig,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<NativeCudaBackend>> {
    prepare_climbmix_peer_for_backend::<NativeCudaBackend>(
        native,
        "cuda",
        Default::default(),
        auth_bundle,
    )
}
