use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;

use crate::local_grid_rho::{
    LocalGridNeighborhood, LocalGridRhoAttentionOutput, LocalGridShape2d,
    supports_local_grid_rho_backend, try_fused_local_grid_rho_attention_wgpu,
};

pub type VisionRhoAttentionOutput<B> = LocalGridRhoAttentionOutput<B>;

pub fn supports_vision_rho_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    supports_local_grid_rho_backend::<B>()
}

#[allow(clippy::too_many_arguments)]
pub fn try_fused_vision_rho_attention_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: Option<&BurnTensor<B, 5>>,
    grid_height: usize,
    grid_width: usize,
    local_radius: usize,
    local_diagonals: bool,
    local_self: bool,
    decay: f32,
) -> Option<VisionRhoAttentionOutput<B>>
where
    B::FloatTensorPrimitive: 'static,
{
    let neighborhood = if local_diagonals {
        LocalGridNeighborhood::moore(local_radius)
    } else {
        LocalGridNeighborhood::von_neumann(local_radius)
    }
    .with_self_edges(local_self);

    try_fused_local_grid_rho_attention_wgpu(
        query,
        value,
        rho,
        LocalGridShape2d::new(grid_height, grid_width),
        neighborhood,
        decay,
    )
}
