#![recursion_limit = "256"]

mod fusion_compat;
mod local_grid_rho;
mod recurrent;
mod sparse_graph_rho;
mod vision_rho;

pub mod kernels {
    pub mod graph {
        pub use crate::sparse_graph_rho::{
            SparseGraphCsr, SparseGraphRhoAttentionError, SparseGraphRhoAttentionOutput,
            fused_sparse_graph_rho_attention_wgpu, supports_sparse_graph_rho_backend,
            try_fused_sparse_graph_rho_attention_wgpu,
        };
    }

    pub mod recurrent {
        pub use crate::recurrent::{
            RecurrentAttentionOutput, RecurrentProfileSnapshot, recurrent_profile_reset,
            recurrent_profile_snapshot, supports_backend as supports_recurrent_backend,
            try_fused_recurrent_attention_wgpu,
        };
    }

    pub mod spatial {
        pub use crate::local_grid_rho::{
            LocalGridNeighborhood, LocalGridRhoAttentionOutput, LocalGridShape2d,
            supports_local_grid_rho_backend, try_fused_local_grid_rho_attention_wgpu,
            try_fused_local_grid_rho_attention_wgpu_head_decay,
        };
        pub use crate::vision_rho::{
            VisionRhoAttentionOutput, supports_vision_rho_backend,
            try_fused_vision_rho_attention_wgpu,
        };
    }
}

pub use local_grid_rho::{
    LocalGridNeighborhood, LocalGridRhoAttentionOutput, LocalGridShape2d,
    supports_local_grid_rho_backend, try_fused_local_grid_rho_attention_wgpu,
    try_fused_local_grid_rho_attention_wgpu_head_decay,
};
pub use recurrent::{
    RecurrentAttentionOutput, RecurrentProfileSnapshot, recurrent_profile_reset,
    recurrent_profile_snapshot, supports_backend as supports_recurrent_backend,
    try_fused_recurrent_attention_wgpu,
};
pub use sparse_graph_rho::{
    SparseGraphCsr, SparseGraphRhoAttentionError, SparseGraphRhoAttentionOutput,
    fused_sparse_graph_rho_attention_wgpu, supports_sparse_graph_rho_backend,
    try_fused_sparse_graph_rho_attention_wgpu,
};
pub use vision_rho::{
    VisionRhoAttentionOutput, supports_vision_rho_backend, try_fused_vision_rho_attention_wgpu,
};
