use super::*;

pub(super) fn try_direct_path_runtime<B: BackendTrait, R: CubeRuntime + 'static>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    sparse_mask: Option<&BurnTensor<B, 4>>,
    meta: &BurnTensor<B, 1>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim_input = input.clone().into_primitive().tensor();
    let input: CubeTensor<R> = try_cast_primitive::<B, _>(prim_input)?;
    if input.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight: CubeTensor<R> = try_cast_primitive::<B, _>(prim_weight)?;
    if weight.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta: CubeTensor<R> = try_cast_primitive::<B, _>(prim_meta)?;
    if meta.dtype != DType::F32 {
        return None;
    }

    let sparse_mask = sparse_mask
        .map(|mask| {
            let prim_mask = mask.clone().into_primitive().tensor();
            let mask: CubeTensor<R> = try_cast_primitive::<B, _>(prim_mask)?;
            (mask.dtype == DType::F32).then_some(mask)
        })
        .flatten();

    let output = relu_lowrank_runtime::<R>(input, weight, shape, meta, sparse_mask);
    let output_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn try_fusion_path_runtime<B, BT, R>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    sparse_mask: Option<&BurnTensor<B, 4>>,
    meta: &BurnTensor<B, 1>,
    shape: LowrankProjectionShape,
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

    let prim_input = input.clone().into_primitive().tensor();
    let fusion_input: FusionTensor<FusionCubeRuntime<R>> = try_cast_primitive::<B, _>(prim_input)?;
    let fusion_client = fusion_input.client.clone();
    let input = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_input);
    if input.dtype != DType::F32 {
        return None;
    }

    let weight = resolve_fusion_tensor_runtime::<B, BT, R, 4>(weight)?;
    let meta = resolve_fusion_tensor_runtime::<B, BT, R, 1>(meta)?;
    let sparse_mask = sparse_mask.and_then(resolve_fusion_tensor_runtime::<B, BT, R, 4>);

    let output = relu_lowrank_runtime::<R>(input, weight, shape, meta, sparse_mask);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_prim = try_cast_backend::<B, _>(output_fusion)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn try_direct_path_autodiff_cube_runtime<B: BackendTrait, R: CubeRuntime + 'static>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    threshold: f32,
    sparse_mask: Option<&BurnTensor<B, 4>>,
    meta: &BurnTensor<B, 1>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let prim_input = input.clone().into_primitive().tensor();
        let input_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_input)?;
        let prim_weight = weight.clone().into_primitive().tensor();
        let weight_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_weight)?;
        let prim_meta = meta.clone().into_primitive().tensor();
        let meta_ad: WgpuCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_meta)?;
        let meta_inner: CubeTensor<WgpuRuntime> =
            <WgpuCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);
        if meta_inner.dtype != DType::F32 {
            return None;
        }
        let mask_ad = sparse_mask.and_then(|mask| {
            let prim_mask = mask.clone().into_primitive().tensor();
            try_cast_primitive::<B, WgpuCubeAutodiffTensor>(prim_mask)
        });
        let output_ad =
            fused_relu_lowrank_autodiff_wgpu(input_ad, weight_ad, mask_ad, shape, meta_inner);
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        let prim_input = input.clone().into_primitive().tensor();
        let input_ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_input)?;
        let prim_weight = weight.clone().into_primitive().tensor();
        let weight_ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_weight)?;
        let prim_meta = meta.clone().into_primitive().tensor();
        let meta_ad: CudaCubeAutodiffTensor = try_cast_primitive::<B, _>(prim_meta)?;
        let meta_inner: CubeTensor<CudaRuntime> =
            <CudaCubeAutodiffBackend as AutodiffBackend>::inner(meta_ad);
        if meta_inner.dtype != DType::F32 {
            return None;
        }
        let mask_ad = sparse_mask.and_then(|mask| {
            let prim_mask = mask.clone().into_primitive().tensor();
            try_cast_primitive::<B, CudaCubeAutodiffTensor>(prim_mask)
        });
        let output_ad =
            fused_relu_lowrank_autodiff_cuda(input_ad, weight_ad, mask_ad, shape, meta_inner);
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    let prim_input = input.clone().into_primitive().tensor();
    let input_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_input)?;
    let input_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(input_ad.clone())?;
    if input_inner.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_weight)?;
    let weight_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(weight_ad.clone())?;
    if weight_inner.dtype != DType::F32 {
        return None;
    }

    let prim_meta = meta.clone().into_primitive().tensor();
    let meta_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_meta)?;
    let meta_inner: CubeTensor<R> = extract_autodiff_inner::<B, R>(meta_ad)?;
    if meta_inner.dtype != DType::F32 {
        return None;
    }

    let mask_ad = sparse_mask.and_then(|mask| {
        let prim_mask = mask.clone().into_primitive().tensor();
        try_cast_primitive::<B, B::FloatTensorPrimitive>(prim_mask)
    });
    let mask_inner = mask_ad
        .clone()
        .map(extract_autodiff_inner::<B, R>)
        .flatten();
    if mask_inner
        .as_ref()
        .is_some_and(|mask| mask.dtype != DType::F32)
    {
        return None;
    }

    let _ = threshold;
    let output =
        relu_lowrank_runtime::<R>(input_inner, weight_inner, shape, meta_inner, mask_inner);
    let output_ad = wrap_autodiff_inner::<B, R>(output)?;
    let output_prim = try_cast_backend::<B, _>(output_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn try_fusion_path_autodiff_runtime<B, BT, R>(
    input: &BurnTensor<B, 4>,
    weight: &BurnTensor<B, 4>,
    threshold: f32,
    sparse_mask: Option<&BurnTensor<B, 4>>,
    meta: &BurnTensor<B, 1>,
    shape: LowrankProjectionShape,
) -> Option<BurnTensor<B, 4>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if TypeId::of::<R>() == TypeId::of::<WgpuRuntime>() {
        let prim_input = input.clone().into_primitive().tensor();
        let input_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_input)?;
        let fusion_input: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(input_ad.clone());
        let fusion_client = fusion_input.client.clone();
        let input = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_input);
        if input.dtype != DType::F32 {
            return None;
        }

        let prim_weight = weight.clone().into_primitive().tensor();
        let weight_ad: WgpuFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_weight)?;
        let fusion_weight: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
            <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(weight_ad.clone());
        let weight = fusion_client
            .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_weight);
        if weight.dtype != DType::F32 {
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

        let mask_ad = sparse_mask.and_then(|mask| {
            let prim_mask = mask.clone().into_primitive().tensor();
            try_cast_primitive::<B, WgpuFusionAutodiffTensor<BT>>(prim_mask)
        });
        let mask = mask_ad.clone().map(|mask| {
            let fusion_mask: FusionTensor<FusionCubeRuntime<WgpuRuntime>> =
                <WgpuFusionAutodiffBackend<BT> as AutodiffBackend>::inner(mask);
            fusion_client
                .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_mask)
        });
        if mask.as_ref().is_some_and(|mask| mask.dtype != DType::F32) {
            return None;
        }

        let _ = threshold;
        let output = relu_lowrank_runtime::<WgpuRuntime>(input, weight, shape, meta, mask);
        let output_fusion = register_fusion_float_tensor(&fusion_client, output);
        let output_ad = fused_relu_lowrank_autodiff_fusion_wgpu::<BT>(
            input_ad,
            weight_ad,
            mask_ad,
            shape,
            output_fusion,
        );
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        let prim_input = input.clone().into_primitive().tensor();
        let input_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_input)?;
        let fusion_input: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(input_ad.clone());
        let fusion_client = fusion_input.client.clone();
        let input = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_input);
        if input.dtype != DType::F32 {
            return None;
        }

        let prim_weight = weight.clone().into_primitive().tensor();
        let weight_ad: CudaFusionAutodiffTensor<BT> = try_cast_primitive::<B, _>(prim_weight)?;
        let fusion_weight: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
            <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(weight_ad.clone());
        let weight = fusion_client
            .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_weight);
        if weight.dtype != DType::F32 {
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

        let mask_ad = sparse_mask.and_then(|mask| {
            let prim_mask = mask.clone().into_primitive().tensor();
            try_cast_primitive::<B, CudaFusionAutodiffTensor<BT>>(prim_mask)
        });
        let mask = mask_ad.clone().map(|mask| {
            let fusion_mask: FusionTensor<FusionCubeRuntime<CudaRuntime>> =
                <CudaFusionAutodiffBackend<BT> as AutodiffBackend>::inner(mask);
            fusion_client
                .resolve_tensor_float::<CubeBackend<CudaRuntime, f32, i32, BT>>(fusion_mask)
        });
        if mask.as_ref().is_some_and(|mask| mask.dtype != DType::F32) {
            return None;
        }

        let _ = threshold;
        let output = relu_lowrank_runtime::<CudaRuntime>(input, weight, shape, meta, mask);
        let output_fusion = register_fusion_float_tensor(&fusion_client, output);
        let output_ad = fused_relu_lowrank_autodiff_fusion_cuda::<BT>(
            input_ad,
            weight_ad,
            mask_ad,
            shape,
            output_fusion,
        );
        let output_prim = try_cast_backend::<B, _>(output_ad)?;
        return Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
            output_prim,
        )));
    }

    if !matches_autodiff_fusion_type::<B, BT, R>() {
        return None;
    }

    let prim_input = input.clone().into_primitive().tensor();
    let input_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_input)?;
    let fusion_input: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(input_ad.clone())?;
    let fusion_client = fusion_input.client.clone();
    let input = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_input);
    if input.dtype != DType::F32 {
        return None;
    }

    let prim_weight = weight.clone().into_primitive().tensor();
    let weight_ad: B::FloatTensorPrimitive = try_cast_primitive::<B, _>(prim_weight)?;
    let fusion_weight: FusionTensor<FusionCubeRuntime<R>> =
        extract_fusion_autodiff_inner::<B, BT, R>(weight_ad.clone())?;
    let weight = fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_weight);
    if weight.dtype != DType::F32 {
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

    let mask_ad = sparse_mask.and_then(|mask| {
        let prim_mask = mask.clone().into_primitive().tensor();
        try_cast_primitive::<B, B::FloatTensorPrimitive>(prim_mask)
    });
    let mask = mask_ad.clone().and_then(|mask| {
        let fusion_mask: FusionTensor<FusionCubeRuntime<R>> =
            extract_fusion_autodiff_inner::<B, BT, R>(mask)?;
        Some(fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_mask))
    });
    if mask.as_ref().is_some_and(|mask| mask.dtype != DType::F32) {
        return None;
    }

    let _ = threshold;
    let output = relu_lowrank_runtime::<R>(input, weight, shape, meta, mask);
    let output_fusion = register_fusion_float_tensor(&fusion_client, output);
    let output_ad = wrap_fusion_autodiff_inner::<B, BT, R>(output_fusion)?;
    let output_prim = try_cast_backend::<B, _>(output_ad)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(
        output_prim,
    )))
}

pub(super) fn relu_lowrank_runtime<R: CubeRuntime + 'static>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
    sparse_mask: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    #[cfg(feature = "cuda")]
    {
        if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
            return relu_lowrank_cube_runtime::<R>(input, weight, shape, meta, sparse_mask);
        }
    }
    relu_lowrank_wgsl_runtime::<R>(input, weight, shape, meta, sparse_mask)
}

pub(super) fn relu_lowrank_wgsl_runtime<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
    sparse_mask: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let meta = into_contiguous(meta);
    let sparse_mask = sparse_mask.map(into_contiguous);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([shape.batch, shape.heads, shape.time, shape.latent]),
    );
    let mask = sparse_mask
        .unwrap_or_else(|| empty_device::<R, f32>(client.clone(), device, Shape::new([1])));

    let workgroups_x = div_ceil_u32(shape.latent as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(
        workgroups_x,
        shape.time as u32,
        (shape.batch * shape.heads) as u32,
    );
    let kernel = SourceKernel::new(ReluLowrankKernel, CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1));
    let bindings = KernelArguments::new().with_buffers(vec![
        input.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
        mask.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    if let Some(start) = total_start {
        profile_record(&RELU_LOWRANK_FORWARD_PROFILE, |state| {
            state.calls = state.calls.saturating_add(1);
            state.launches = state.launches.saturating_add(1);
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
        });
    }
    output
}

#[cfg(feature = "cuda")]
fn relu_lowrank_cube_runtime<R: CubeRuntime>(
    input: CubeTensor<R>,
    weight: CubeTensor<R>,
    shape: LowrankProjectionShape,
    meta: CubeTensor<R>,
    sparse_mask: Option<CubeTensor<R>>,
) -> CubeTensor<R> {
    let prof_enabled = profile_enabled();
    let total_start = prof_enabled.then(Instant::now);
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let meta = into_contiguous(meta);
    let sparse_mask = sparse_mask.map(into_contiguous);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<R, f32>(
        client.clone(),
        device.clone(),
        Shape::new([shape.batch, shape.heads, shape.time, shape.latent]),
    );
    let mask = sparse_mask
        .unwrap_or_else(|| empty_device::<R, f32>(client.clone(), device, Shape::new([1])));

    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(shape.latent as u32, WORKGROUP_SIZE_X),
        shape.time as u32,
        (shape.batch * shape.heads) as u32,
    );

    let _ = relu_lowrank_cube_kernel::launch::<R>(
        &client,
        cube_count,
        cube_dim,
        input.clone().into_tensor_arg(),
        weight.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        meta.clone().into_tensor_arg(),
        mask.clone().into_tensor_arg(),
    );

    if let Some(start) = total_start {
        profile_record(&RELU_LOWRANK_FORWARD_PROFILE, |state| {
            state.calls = state.calls.saturating_add(1);
            state.launches = state.launches.saturating_add(1);
            state.total_ns = state.total_ns.saturating_add(start.elapsed().as_nanos());
        });
    }

    output
}
