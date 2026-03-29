use super::*;
use burn::tensor::backend::AutodiffBackend;

fn dense_causal_backward_impl<B: BackendTrait>(
    grad_output: BurnTensor<B, 4>,
    query: BurnTensor<B, 4>,
    value: BurnTensor<B, 4>,
    decay: BurnTensor<B, 1>,
) -> (BurnTensor<B, 4>, BurnTensor<B, 4>, BurnTensor<B, 1>)
where
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = query.shape().dims::<4>();
    let value_heads = value.shape().dims::<4>()[1];
    let value_dim = value.shape().dims::<4>()[3];

    let value_per_head = if value_heads == heads {
        value.clone()
    } else {
        value.clone().repeat_dim(1, heads)
    };

    let pos_row = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, time, 1]);
    let pos_col = BurnTensor::<B, 1, Int>::arange(0..time as i64, &query.device())
        .float()
        .reshape([1, 1, 1, time]);
    let gap = (pos_row - pos_col).tril(-1);
    let decay_matrix = decay
        .clone()
        .reshape([1, heads, 1, 1])
        .repeat_dim(2, time)
        .repeat_dim(3, time)
        .powf(gap.clone());

    let raw_scores = query.clone().matmul(query.clone().swap_dims(2, 3)).tril(-1);
    let scores = raw_scores * decay_matrix.clone();

    let batch_heads = batch * heads;
    let query_flat = query.clone().reshape([batch_heads, time, latent]);
    let value_flat = value_per_head
        .clone()
        .reshape([batch_heads, time, value_dim]);
    let grad_output_flat = grad_output.clone().reshape([batch_heads, time, value_dim]);

    let grad_value_heads = scores
        .clone()
        .swap_dims(2, 3)
        .reshape([batch_heads, time, time])
        .matmul(grad_output_flat.clone())
        .reshape([batch, heads, time, value_dim]);
    let grad_value = if value_heads == heads {
        grad_value_heads
    } else {
        grad_value_heads
            .sum_dim(1)
            .reshape([batch, 1, time, value_dim])
    };

    let grad_scores = grad_output_flat
        .matmul(value_flat.swap_dims(1, 2))
        .reshape([batch, heads, time, time])
        .tril(-1);
    let grad_raw_scores = grad_scores.clone() * decay_matrix;
    let grad_query = (grad_raw_scores.clone() + grad_raw_scores.swap_dims(2, 3))
        .reshape([batch_heads, time, time])
        .matmul(query_flat)
        .reshape([batch, heads, time, latent]);

    let safe_decay = decay.clone().add_scalar(1.0e-12).reshape([1, heads, 1, 1]);
    let grad_decay = ((grad_scores * gap) * scores)
        .div(safe_decay)
        .sum_dim(0)
        .sum_dim(2)
        .sum_dim(3)
        .reshape([heads]);

    (grad_query, grad_value, grad_decay)
}

fn dense_causal_attention_backward_impl<B: BackendTrait>(
    ops: Ops<DenseCausalAttentionBackwardState<B::FloatTensorPrimitive>, 3>,
    grads: &mut Gradients,
) where
    B::FloatTensorPrimitive: 'static,
{
    let grad_output =
        BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(grads.consume::<B>(&ops.node)));
    let DenseCausalAttentionBackwardState {
        query,
        value,
        decay,
    } = ops.state;
    let query = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(query));
    let value = BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(value));
    let decay = BurnTensor::<B, 1>::from_primitive(TensorPrimitive::Float(decay));

    let (grad_query, grad_value, grad_decay) =
        dense_causal_backward_impl(grad_output, query, value, decay);

    if let Some(parent) = &ops.parents[0] {
        grads.register::<B>(parent.id, grad_query.into_primitive().tensor());
    }
    if let Some(parent) = &ops.parents[1] {
        grads.register::<B>(parent.id, grad_value.into_primitive().tensor());
    }
    if let Some(parent) = &ops.parents[2] {
        grads.register::<B>(parent.id, grad_decay.into_primitive().tensor());
    }
}

impl Backward<WgpuCubeBackend, 3> for FusedDenseCausalAttentionBackward<WgpuCubeBackend> {
    type State = DenseCausalAttentionBackwardState<CubeTensor<WgpuRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        dense_causal_attention_backward_impl::<WgpuCubeBackend>(ops, grads);
    }
}

#[cfg(feature = "cuda")]
impl Backward<CudaCubeBackend, 3> for FusedDenseCausalAttentionBackward<CudaCubeBackend> {
    type State = DenseCausalAttentionBackwardState<CubeTensor<CudaRuntime>>;

    fn backward(
        self,
        ops: Ops<Self::State, 3>,
        grads: &mut Gradients,
        _checkpointer: &mut Checkpointer,
    ) {
        dense_causal_attention_backward_impl::<CudaCubeBackend>(ops, grads);
    }
}

fn dense_causal_attention_autodiff_custom_wgpu<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let decay_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: WgpuCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let decay_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let context = forward_runtime::dense_causal_attention_runtime::<WgpuRuntime>(
        query_inner.clone(),
        value_inner.clone(),
        decay_inner.clone(),
        meta_inner,
    );

    let output = match FusedDenseCausalAttentionBackward::<WgpuCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            DenseCausalAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                decay: decay_inner,
            },
            context,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context),
    };

    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(output)?,
    )))
}

#[cfg(feature = "cuda")]
fn dense_causal_attention_autodiff_custom_cuda<B: BackendTrait>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let query_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(query.clone().into_primitive().tensor())?;
    let value_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(value.clone().into_primitive().tensor())?;
    let decay_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(decay.clone().into_primitive().tensor())?;
    let meta_ad: CudaCubeAutodiffTensor =
        try_cast_primitive::<B, _>(meta.clone().into_primitive().tensor())?;

    let query_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(query_ad.clone());
    let value_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(value_ad.clone());
    let decay_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(decay_ad.clone());
    let meta_inner = <CudaCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);

    let context = forward_runtime::dense_causal_attention_runtime::<CudaRuntime>(
        query_inner.clone(),
        value_inner.clone(),
        decay_inner.clone(),
        meta_inner,
    );

    let output = match FusedDenseCausalAttentionBackward::<CudaCubeBackend>(PhantomData)
        .prepare::<NoCheckpointing>([
            query_ad.node.clone(),
            value_ad.node.clone(),
            decay_ad.node.clone(),
        ])
        .compute_bound()
        .stateful()
    {
        OpsKind::Tracked(prep) => prep.finish(
            DenseCausalAttentionBackwardState {
                query: query_inner,
                value: value_inner,
                decay: decay_inner,
            },
            context,
        ),
        OpsKind::UnTracked(prep) => prep.finish(context),
    };

    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        try_cast_backend::<B, _>(output)?,
    )))
}

pub(super) fn dense_causal_attention_autodiff_custom<B: BackendTrait, R: CubeRuntime + 'static>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        return dense_causal_attention_autodiff_custom_wgpu(query, value, decay, meta);
    }
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        return dense_causal_attention_autodiff_custom_cuda(query, value, decay, meta);
    }
    None
}
