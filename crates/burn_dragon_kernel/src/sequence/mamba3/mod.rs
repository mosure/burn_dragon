pub mod backward;
pub mod backward_runtime;
pub mod bc_runtime;
pub mod forward;
pub mod forward_runtime;
pub mod preprocess_runtime;
pub mod rotary_runtime;

/// Mamba-3 currently follows the upstream SISO formulation exactly at the tensorized level.
/// The best current WGPU training path is direct graph with the fused preprocess and long-chunk
/// state-update boundaries enabled. The custom analytical wrapper remains available, but it is
/// not the default on WGPU because it is slower on real training shapes. CUDA keeps the custom
/// wrapper as the stronger experimental path.
pub const STATUS: &str = "wgpu_direct_graph_preprocess_state_update_cuda_custom_wrapper";
pub const FORWARD_ACCELERATION_AVAILABLE: bool = true;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = true;
pub const UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
pub const UPSTREAM_TARGET_KIND: &str = "mamba3_state_space_duality";
