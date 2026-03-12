#![recursion_limit = "256"]

//! Fused Dragon execution kernels and compiled-plan helpers.
//!
//! Preferred library-facing surface:
//! - [`api::recurrent`]
//! - [`api::spatial`]
//! - [`api::graph`]
//! - [`api::expert`] for lower-level kernel-plan access

mod fusion_compat;
mod local_grid_rho;
mod profiling;
mod recurrent;
mod sparse_graph_rho;
mod structured_pyramid_rho;
mod vision_rho;

pub mod api {
    //! Curated public surface for the fused execution layer.
    //!
    //! This mirrors the active kernel families instead of exposing the entire file/module layout.

    pub use crate::kernels::{graph, recurrent, spatial};

    pub mod expert {
        //! Lower-level fused-kernel surface for advanced callers.

        pub use crate::kernels;
    }
}

pub mod kernels {
    pub mod graph {
        pub use crate::sparse_graph_rho::{
            SparseGraphCsr, SparseGraphRhoAttentionError, SparseGraphRhoAttentionOutput,
            SparseGraphRhoProfileSnapshot, fused_sparse_graph_rho_attention_wgpu,
            sparse_graph_rho_profile_reset, sparse_graph_rho_profile_snapshot,
            supports_sparse_graph_rho_backend, try_fused_sparse_graph_rho_attention_wgpu,
        };
    }

    pub mod recurrent {
        pub use crate::recurrent::{
            CompiledRecurrentAttentionPlan, RecurrentAttentionOutput, RecurrentProfileSnapshot,
            recurrent_profile_reset, recurrent_profile_snapshot,
            supports_backend as supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
            try_fused_recurrent_attention_wgpu_with_plan,
        };
    }

    pub mod spatial {
        pub use crate::local_grid_rho::{
            CompiledLocalGridRhoPlan, LocalGridNeighborhood, LocalGridRhoAttentionOutput,
            LocalGridRhoPlanSpec, LocalGridRhoProfileSnapshot, LocalGridShape2d,
            local_grid_rho_profile_reset, local_grid_rho_profile_snapshot,
            supports_local_grid_rho_backend,
            try_fused_local_grid_rho_attention_wgpu,
            try_fused_local_grid_rho_attention_wgpu_head_decay,
            try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan,
        };
        pub use crate::structured_pyramid_rho::{
            CompiledStructuredPyramidRhoPlan, StructuredPyramidProfileSnapshot,
            StructuredPyramidRhoStepInput, StructuredPyramidRhoStepOutput,
            StructuredPyramidShape, reference_structured_pyramid_rho_step,
            structured_pyramid_profile_reset, structured_pyramid_profile_snapshot,
            supports_structured_pyramid_rho_backend, try_fused_structured_pyramid_rho_step_wgpu,
            try_fused_structured_pyramid_rho_step_wgpu_with_plan,
        };
        pub use crate::vision_rho::{
            VisionRhoAttentionOutput, supports_vision_rho_backend,
            try_fused_vision_rho_attention_wgpu,
        };
    }
}
