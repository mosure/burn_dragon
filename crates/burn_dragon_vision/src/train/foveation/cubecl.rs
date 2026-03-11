use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;

use crate::train::saccade::{SaccadeLaplacianImages, SaccadeMipLevel};

pub(crate) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    false
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn try_foveated_patch_cubecl<B: BackendTrait>(
    _levels: &[SaccadeMipLevel<B>],
    _base_grid: &BurnTensor<B, 4>,
    _center_x: &BurnTensor<B, 3>,
    _center_y: &BurnTensor<B, 3>,
    _sigma_px: &BurnTensor<B, 3>,
    _radius_px: &BurnTensor<B, 3>,
    _lod_sigma: &BurnTensor<B, 3>,
    _laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    _grid_sample_max_bytes: u64,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}
