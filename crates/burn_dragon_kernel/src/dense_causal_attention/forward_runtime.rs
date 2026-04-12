use super::*;

pub(super) fn try_fusion_path_runtime<B, BT, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R>>>() {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let fusion_query: FusionTensor<FusionCubeRuntime<R>> = try_cast_primitive::<B, _>(prim_query)?;
    let fusion_client = fusion_query.client.clone();
    let query = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let value = resolve_fusion_tensor_runtime::<B, BT, R, 4>(value)?;
    let decay = resolve_fusion_tensor_runtime::<B, BT, R, 1>(decay)?;
    let meta = resolve_fusion_tensor_runtime::<B, BT, R, 1>(meta)?;

    let output = dense_causal_attention_runtime::<R>(query, value, decay, meta);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_prim = try_cast_backend::<B, _>(output_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn try_direct_path_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let prim_query = query.clone().into_primitive().tensor();
    let query: CubeTensor<R> = try_cast_primitive::<B, _>(prim_query)?;
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value: CubeTensor<R> = try_cast_primitive::<B, _>(prim_value)?;
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_decay = decay.clone().into_primitive().tensor();
    let decay: CubeTensor<R> = try_cast_primitive::<B, _>(prim_decay)?;
    if decay.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta: CubeTensor<R> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let output = dense_causal_attention_runtime::<R>(query, value, decay, meta);
    let output_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn try_direct_path_autodiff_cube_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    super::dense_causal_attention_autodiff_custom::<B, R>(query, value, decay, meta)
}

pub(super) fn try_fusion_path_autodiff_runtime<B, BT, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let prim_query = query.clone().into_primitive().tensor();
        let query_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_query)?;
        let fusion_query: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query_ad.clone());
        let fusion_client = fusion_query.client.clone();
        let query = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_query);
        if query.dtype != DType::F32 {
            return None;
        }

        let prim_value = value.clone().into_primitive().tensor();
        let value_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_value)?;
        let fusion_value: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(value_ad.clone());
        let value = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_value);
        if value.dtype != DType::F32 {
            return None;
        }

        let prim_decay = decay.clone().into_primitive().tensor();
        let decay_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_decay)?;
        let fusion_decay: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(decay_ad);
        let decay = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_decay);
        if decay.dtype != DType::F32 {
            return None;
        }

        let prim_meta = meta.clone().into_primitive().tensor();
        let meta_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_meta)?;
        let fusion_meta: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(meta_ad);
        let meta = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_meta);
        if meta.dtype != DType::F32 {
            return None;
        }

        let output = dense_causal_attention_runtime::<WgpuRuntime>(query, value, decay, meta);
        let output_fusion = register_fusion_float_tensor(&fusion_client, output);
        let output_ad = wrap_fusion_autodiff_inner::<B, BT, WgpuRuntime>(output_fusion)?;
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        let prim_query = query.clone().into_primitive().tensor();
        let query_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_query)?;
        let fusion_query: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(query_ad.clone());
        let fusion_client = fusion_query.client.clone();
        let query = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_query);
        if query.dtype != DType::F32 {
            return None;
        }

        let prim_value = value.clone().into_primitive().tensor();
        let value_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_value)?;
        let fusion_value: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(value_ad.clone());
        let value = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_value);
        if value.dtype != DType::F32 {
            return None;
        }

        let prim_decay = decay.clone().into_primitive().tensor();
        let decay_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_decay)?;
        let fusion_decay: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(decay_ad);
        let decay = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_decay);
        if decay.dtype != DType::F32 {
            return None;
        }

        let prim_meta = meta.clone().into_primitive().tensor();
        let meta_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_meta)?;
        let fusion_meta: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(meta_ad);
        let meta = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_meta);
        if meta.dtype != DType::F32 {
            return None;
        }

        let output = dense_causal_attention_runtime::<CudaRuntime>(query, value, decay, meta);
        let output_fusion = register_fusion_float_tensor(&fusion_client, output);
        let output_ad = wrap_fusion_autodiff_inner::<B, BT, CudaRuntime>(output_fusion)?;
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    if !matches_autodiff_fusion_type::<B, BT, R>() {
        return None;
    }

    let prim_query = query.clone().into_primitive().tensor();
    let query_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_query)?;
    let fusion_query: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(query_ad.clone())?;
    let fusion_client = fusion_query.client.clone();
    let query = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_query);
    if query.dtype != DType::F32 {
        return None;
    }

    let prim_value = value.clone().into_primitive().tensor();
    let value_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_value)?;
    let fusion_value: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(value_ad.clone())?;
    let value = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_value);
    if value.dtype != DType::F32 {
        return None;
    }

    let prim_decay = decay.clone().into_primitive().tensor();
    let decay_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_decay)?;
    let fusion_decay: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(decay_ad)?;
    let decay = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_decay);
    if decay.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_meta)?;
    let fusion_meta: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(meta_ad)?;
    let meta = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_meta);
    if meta.dtype != DType::F32 {
        return None;
    }

    let output = dense_causal_attention_runtime::<R>(query, value, decay, meta);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_ad = wrap_fusion_autodiff_inner::<B, BT, R>(output_fusion)?;
    let output_prim = try_cast_backend::<B, _>(output_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(crate) fn dense_causal_attention_runtime<R: CubeRuntime + 'static>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            return dense_causal_attention_cube_runtime::<R>(query, value, decay, meta);
        }
    }
    dense_causal_attention_wgsl_runtime::<R>(query, value, decay, meta)
}

pub(super) fn dense_causal_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let value_dim = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, value_dim]),
    );

    let workgroups_x = div_ceil_u32(value_dim as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(workgroups_x, time as u32, (batch * heads) as u32);
    let kernel = SourceKernel::new(
        DenseCausalAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        decay.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);

    output
}

#[cfg(feature = "cuda")]
fn dense_causal_attention_cube_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> CubeTensor<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let value_dim = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, value_dim]),
    );

    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(value_dim as u32, WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );

    let _ = dense_causal_attention_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        query.clone().into_tensor_arg(),
        value.clone().into_tensor_arg(),
        decay.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        meta.clone().into_tensor_arg(),
        MAX_FUSED_TIME,
    );

    output
}
