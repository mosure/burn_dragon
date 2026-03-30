pub mod backward;
pub mod bench;
pub mod forward;
mod runtime;
#[cfg(test)]
mod tests;

/// The local RWKV-8 path follows the BDH-aligned decayed normalized recurrence contract while
/// taking the public RWKV-v8 CUDA sources as the closest upstream implementation reference. The
/// analytical wrapper is now the default training route, while the tensorized direct-graph path
/// remains available as an opt-out baseline.
pub const STATUS: &str = "analytic_wrapper_default_with_runtime_forward_backward_shell_and_tensorized_direct_graph_baseline";
pub const FORWARD_ACCELERATION_AVAILABLE: bool = forward::AVAILABLE;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = backward::AVAILABLE;
pub const UPSTREAM_MODEL_REPO: &str = "https://github.com/BlinkDL/RWKV-LM";
pub const UPSTREAM_KERNEL_REPO: &str = "https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v8/cuda";
pub const UPSTREAM_TARGET_KIND: &str = "public_rwkv_v8_cuda_reference";
