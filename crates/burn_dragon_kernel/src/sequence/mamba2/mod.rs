pub mod backward;
pub mod bench;
pub mod forward;
pub mod rmsnorm_runtime;
pub mod ssd_runtime;

/// Mamba-2 SSD currently ships as a tensorized CUDA/WGPU path. WGPU inference can route the SSD
/// recurrence core through a custom Cube forward kernel. On CUDA, the default training path uses
/// the custom analytic backward wrapper with the fused SSD recurrence core and fused conv /
/// RMSNorm shell.
pub const STATUS: &str = "tensorized_default_wgpu_custom_ssd_forward_and_default_cuda_custom_analytic_backward_with_fused_ssd_and_shell";
pub const FORWARD_ACCELERATION_AVAILABLE: bool = true;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = true;
pub const CUDA_DEFAULT_TRAIN_PATH: &str = "custom_analytic_backward_wrapper";
pub const CUDA_FUSED_ANALYTIC_BACKWARD_AVAILABLE: bool = true;
pub const UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
pub const UPSTREAM_TARGET_KIND: &str = "mamba2_state_space_duality";
