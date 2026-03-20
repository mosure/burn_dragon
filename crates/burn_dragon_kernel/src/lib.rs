#![recursion_limit = "256"]

//! Fused Dragon execution kernels and compiled-plan helpers.
//!
//! Preferred library-facing surface:
//! - [`api::recurrent`]
//! - [`api::spatial`]
//! - [`api::graph`]
//! - [`api::expert`] for lower-level kernel-plan access

mod dense_attention;
mod dense_causal_attention;
mod dense_scores;
mod fusion_compat;
mod local_grid_rho;
mod profiling;
mod recurrent;
mod relu_lowrank;
mod sequence;
mod sparse_graph_rho;
mod structured_pyramid_rho;
mod vision_rho;

pub mod api {
    //! Curated public surface for the fused execution layer.
    //!
    //! This mirrors the active kernel families instead of exposing the entire file/module layout.

    pub use crate::kernels::{attention, graph, projection, recurrent, spatial};

    pub mod expert {
        //! Lower-level fused-kernel surface for advanced callers.

        pub use crate::kernels;
    }
}

/// Namespaced fused-kernel families exposed by this crate.
pub mod kernels {
    /// Sequence-kernel family namespace used by the language line.
    pub mod sequence {
        pub use crate::sequence::{linear, mamba, rwkv8};
    }

    /// Dense attention and score kernels used by language/vision recurrent executors.
    pub mod attention {
        pub use crate::dense_attention::{
            CompiledDenseAttentionPlan, supports_dense_attention_backend,
            try_fused_dense_row_l1_attention_wgpu, try_fused_dense_row_l1_attention_wgpu_with_plan,
        };
        pub use crate::dense_causal_attention::{
            CompiledDenseCausalAttentionPlan, supports_dense_causal_attention_backend,
            try_fused_dense_causal_attention_wgpu, try_fused_dense_causal_attention_wgpu_with_plan,
        };
        pub use crate::dense_scores::{
            CompiledDenseScoresPlan, supports_dense_scores_backend,
            try_fused_dense_row_l1_scores_wgpu, try_fused_dense_row_l1_scores_wgpu_with_plan,
        };
    }

    /// Sparse graph-routing kernels over recurrent `rho` state.
    pub mod graph {
        pub use crate::sparse_graph_rho::{
            SparseGraphCsr, SparseGraphRhoAttentionError, SparseGraphRhoAttentionOutput,
            SparseGraphRhoProfileSnapshot, fused_sparse_graph_rho_attention_wgpu,
            sparse_graph_rho_profile_reset, sparse_graph_rho_profile_snapshot,
            supports_sparse_graph_rho_backend, try_fused_sparse_graph_rho_attention_wgpu,
        };
    }

    /// Core recurrent attention kernels.
    pub mod recurrent {
        pub use crate::recurrent::{
            CompiledRecurrentAttentionPlan, RecurrentAttentionOutput, RecurrentProfileSnapshot,
            recurrent_profile_reset, recurrent_profile_snapshot,
            supports_backend as supports_recurrent_backend, try_fused_recurrent_attention_wgpu,
            try_fused_recurrent_attention_wgpu_with_plan,
        };
    }

    /// Fused low-rank projection kernels used in recurrent x/y projection paths.
    pub mod projection {
        pub use crate::relu_lowrank::{
            LowrankGradInputExecutor, LowrankProjectionProfileSnapshot,
            relu_lowrank_forward_profile_reset, relu_lowrank_forward_profile_snapshot,
            relu_lowrank_grad_input_profile_reset, relu_lowrank_grad_input_profile_snapshot,
            relu_lowrank_grad_weight_profile_reset, relu_lowrank_grad_weight_profile_snapshot,
            supports_relu_lowrank_projection_backend, try_fused_relu_lowrank_projection_wgpu,
            try_fused_relu_lowrank_projection_wgpu_with_executor,
        };
    }

    /// Spatial/topological recurrent kernels for local-grid, pyramid, and vision-rho paths.
    pub mod spatial {
        pub use crate::local_grid_rho::{
            CompiledLocalGridRhoPlan, LocalGridNeighborhood, LocalGridRhoAttentionOutput,
            LocalGridRhoPlanSpec, LocalGridRhoProfileSnapshot, LocalGridShape2d,
            local_grid_rho_profile_reset, local_grid_rho_profile_snapshot,
            supports_local_grid_rho_backend, try_fused_local_grid_rho_attention_wgpu,
            try_fused_local_grid_rho_attention_wgpu_head_decay,
            try_fused_local_grid_rho_attention_wgpu_head_decay_with_plan,
        };
        pub use crate::structured_pyramid_rho::{
            CompiledStructuredPyramidRhoPlan, CompiledStructuredPyramidSplitPlan,
            StructuredPyramidBankMode, StructuredPyramidCoarseOnlyNoPatchStepInput,
            StructuredPyramidCoarseOnlyStepInput, StructuredPyramidCoarseOnlyStepOutput,
            StructuredPyramidProfileSnapshot, StructuredPyramidRhoStepInput,
            StructuredPyramidRhoStepOutput, StructuredPyramidShape, StructuredPyramidSplitPlanSpec,
            StructuredPyramidSplitRhoStepInput, reference_structured_pyramid_rho_step,
            structured_pyramid_profile_reset, structured_pyramid_profile_snapshot,
            supports_structured_pyramid_rho_backend,
            try_fused_structured_pyramid_coarse_only_no_patch_step_wgpu_with_plan,
            try_fused_structured_pyramid_coarse_only_step_wgpu_with_plan,
            try_fused_structured_pyramid_rho_step_wgpu,
            try_fused_structured_pyramid_rho_step_wgpu_with_plan,
            try_fused_structured_pyramid_split_step_wgpu_with_plan,
        };
        pub use crate::vision_rho::{
            VisionRhoAttentionOutput, supports_vision_rho_backend,
            try_fused_vision_rho_attention_wgpu,
        };
    }
}
