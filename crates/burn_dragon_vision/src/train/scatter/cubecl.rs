use burn::tensor::Tensor as BurnTensor;
use burn::tensor::backend::Backend as BackendTrait;

pub(crate) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    false
}

pub(crate) fn try_weighted_sum_tokens_cubecl<B: BackendTrait>(
    _weights: &BurnTensor<B, 3>,
    _tokens: &BurnTensor<B, 3>,
) -> Option<BurnTensor<B, 3>>
where
    B::FloatTensorPrimitive: 'static,
{
    None
}
