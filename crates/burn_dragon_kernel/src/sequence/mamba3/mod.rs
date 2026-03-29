pub mod backward;
pub mod bc_runtime;
pub mod forward;
pub mod rotary_runtime;

/// Mamba-3 currently follows the upstream SISO formulation exactly at the tensorized level.
/// The current CUDA training path defaults to the custom analytical backward wrapper over the
/// chunked tensorized SISO recurrence. Forward/state parity is tested against the direct graph
/// path.
pub const STATUS: &str = "tensorized_siso_custom_backward_path";
pub const FORWARD_ACCELERATION_AVAILABLE: bool = true;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = true;
pub const UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
pub const UPSTREAM_TARGET_KIND: &str = "mamba3_state_space_duality";
