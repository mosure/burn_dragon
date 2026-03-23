pub mod backward;
pub mod bench;
pub mod forward;

/// The repo currently exposes an `rwkv8`-named experimental family, but the clearest public
/// upstream fused CUDA anchor today still lives in the official RWKV CUDA repo's RWKV-7 fast
/// fused path. Keep the naming and parity claims honest until a public RWKV-8 fused target is
/// actually pinned and ported.
pub const STATUS: &str = "experimental_tensorized_train_wrapper";
pub const FORWARD_ACCELERATION_AVAILABLE: bool = forward::AVAILABLE;
pub const BACKWARD_ACCELERATION_AVAILABLE: bool = backward::AVAILABLE;
pub const UPSTREAM_MODEL_REPO: &str = "https://github.com/BlinkDL/RWKV-LM";
pub const UPSTREAM_KERNEL_REPO: &str =
    "https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused";
pub const UPSTREAM_TARGET_KIND: &str = "public_rwkv7_fast_fused";
