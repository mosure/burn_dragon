#![recursion_limit = "256"]

mod recurrent;

pub use recurrent::{
    RecurrentAttentionOutput, RecurrentProfileSnapshot, recurrent_profile_reset,
    recurrent_profile_snapshot, supports_backend as supports_recurrent_backend,
    try_fused_recurrent_attention_wgpu,
};
