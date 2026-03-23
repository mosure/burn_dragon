pub mod bench;
pub mod conv;
pub mod selective_scan_backward;
pub mod selective_scan_forward;

/// Mamba selective state-space kernels are planned to match the pinned upstream Mamba-1 reference
/// and its selective-scan implementation. The current forward path is tensorized and exact on
/// tested fixtures, but it is not yet a true fused selective-scan kernel.
pub const STATUS: &str = "default_tensorized_direct_path";
pub const FORWARD_ACCELERATION_AVAILABLE: bool =
    conv::AVAILABLE && selective_scan_forward::AVAILABLE;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = selective_scan_backward::AVAILABLE;
pub const UPSTREAM_REPO: &str = "https://github.com/state-spaces/mamba";
pub const UPSTREAM_COMMIT: &str = "c5afbdf";
pub const UPSTREAM_TARGET_KIND: &str = "mamba1_selective_scan";
