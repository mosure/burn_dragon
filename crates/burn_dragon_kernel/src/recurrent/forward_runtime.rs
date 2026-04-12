use super::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RecurrentForwardRuntimeKind {
    Wgsl,
    Exact,
    Tiled,
}

pub(super) fn try_fusion_path_runtime<B, BT, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
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
    let rho = resolve_fusion_tensor_runtime::<B, BT, R, 4>(rho)?;
    let decay = resolve_fusion_tensor_runtime::<B, BT, R, 1>(decay)?;
    let meta = resolve_fusion_tensor_runtime::<B, BT, R, 1>(meta)?;

    let output = recurrent_attention_runtime::<R>(query, value, rho, decay, meta, false);

    let context_fusion = register_fusion_float_tensor(&fusion_client, output.context);
    let rho_fusion = register_fusion_float_tensor(&fusion_client, output.rho);

    let context_prim = try_cast_backend::<B, _>(context_fusion)?;
    let rho_prim = try_cast_backend::<B, _>(rho_fusion)?;

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

pub(super) fn try_direct_path_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
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

    let prim_rho = rho.clone().into_primitive().tensor();
    let rho: CubeTensor<R> = try_cast_primitive::<B, _>(prim_rho)?;
    if rho.dtype != DType::F32 {
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

    let output = recurrent_attention_runtime::<R>(query, value, rho, decay, meta, false);

    let context_prim = try_cast_backend::<B, _>(output.context)?;
    let rho_prim = try_cast_backend::<B, _>(output.rho)?;

    Some(RecurrentAttentionOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
    })
}

pub(super) fn try_direct_path_runtime_with_state_history<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionCapturedOutput<B>>
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

    let prim_rho = rho.clone().into_primitive().tensor();
    let rho: CubeTensor<R> = try_cast_primitive::<B, _>(prim_rho)?;
    if rho.dtype != DType::F32 {
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

    let output = recurrent_attention_runtime::<R>(query, value, rho, decay, meta, true);
    let state_history = output.state_history?;

    let context_prim = try_cast_backend::<B, _>(output.context)?;
    let rho_prim = try_cast_backend::<B, _>(output.rho)?;
    let state_history_prim = try_cast_backend::<B, _>(state_history)?;

    Some(RecurrentAttentionCapturedOutput {
        context: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(context_prim)),
        rho: BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(rho_prim)),
        state_history: BurnTensor::<B, 5>::from_primitive(TensorPrimitive::Float(
            state_history_prim,
        )),
    })
}

pub(super) fn try_direct_path_autodiff_cube_runtime<B, R>(
    query: &BurnTensor<B, 4>,
    value: &BurnTensor<B, 4>,
    rho: &BurnTensor<B, 4>,
    decay: &BurnTensor<B, 1>,
    meta: &BurnTensor<B, 1>,
) -> Option<RecurrentAttentionOutput<B>>
where
    B: BackendTrait,
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    super::try_direct_path_autodiff_cube_runtime_custom::<B, R>(query, value, rho, decay, meta)
}

fn recurrent_attention_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
    capture_state_history: bool,
) -> RecurrentRuntimeOutput<R> {
    let time = query.meta.shape.dims::<4>()[2];
    match select_recurrent_forward_runtime::<R>(capture_state_history, time) {
        RecurrentForwardRuntimeKind::Wgsl => {
            recurrent_attention_wgsl_runtime::<R>(query, value, rho, decay, meta)
        }
        RecurrentForwardRuntimeKind::Exact => recurrent_attention_cube_exact_runtime::<R>(
            query,
            value,
            rho,
            decay,
            meta,
            capture_state_history,
        ),
        RecurrentForwardRuntimeKind::Tiled => recurrent_attention_cube_tiled_runtime::<R>(
            query,
            value,
            rho,
            decay,
            meta,
            capture_state_history,
        ),
    }
}

fn select_recurrent_forward_runtime<R: CubeRuntime>(
    capture_state_history: bool,
    time: usize,
) -> RecurrentForwardRuntimeKind {
    #[cfg(feature = "cuda")]
    if TypeId::of::<R>() == TypeId::of::<CudaRuntime>() {
        if capture_state_history {
            return RecurrentForwardRuntimeKind::Exact;
        }
        return match std::env::var("BURN_DRAGON_CUDA_RECURRENT_RUNTIME")
            .ok()
            .as_deref()
        {
            Some("tiled") => RecurrentForwardRuntimeKind::Tiled,
            Some("exact") => RecurrentForwardRuntimeKind::Exact,
            _ if use_cuda_tiled_recurrent_experimental() => RecurrentForwardRuntimeKind::Tiled,
            _ => RecurrentForwardRuntimeKind::Exact,
        };
    }

    if capture_state_history {
        return match std::env::var("BURN_DRAGON_WGPU_RECURRENT_HISTORY_RUNTIME")
            .ok()
            .as_deref()
        {
            Some("exact") => RecurrentForwardRuntimeKind::Exact,
            Some("tiled") => RecurrentForwardRuntimeKind::Tiled,
            Some("auto") | None if time >= 128 => RecurrentForwardRuntimeKind::Tiled,
            Some("auto") | None => RecurrentForwardRuntimeKind::Exact,
            _ => RecurrentForwardRuntimeKind::Exact,
        };
    }

    match std::env::var("BURN_DRAGON_WGPU_RECURRENT_RUNTIME")
        .ok()
        .as_deref()
    {
        Some("wgsl") => RecurrentForwardRuntimeKind::Wgsl,
        Some("exact") => RecurrentForwardRuntimeKind::Exact,
        Some("tiled") => RecurrentForwardRuntimeKind::Tiled,
        Some("auto") | None if time >= 128 => RecurrentForwardRuntimeKind::Exact,
        Some("auto") | None => RecurrentForwardRuntimeKind::Wgsl,
        _ => RecurrentForwardRuntimeKind::Wgsl,
    }
}

pub(super) fn recurrent_attention_wgsl_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
) -> RecurrentRuntimeOutput<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho = into_contiguous(rho);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, time, _latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, embd]),
    );

    let workgroups_x = div_ceil_u32(embd as u32, WORKGROUP_SIZE_X);
    let count = CubeCount::Static(workgroups_x, heads as u32, batch as u32);

    let kernel = SourceKernel::new(
        RecurrentAttentionKernel,
        CubeDim::new_3d(WORKGROUP_SIZE_X, 1, 1),
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        query.handle.clone().binding(),
        value.handle.clone().binding(),
        rho.handle.clone().binding(),
        decay.handle.clone().binding(),
        context.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);

    let dispatch_start = profile_enabled().then(Instant::now);
    client.launch(Box::new(kernel), count, bindings);
    if let Some(start) = dispatch_start {
        let dispatch_ns = start.elapsed().as_nanos();
        profile_record(&RECURRENT_PROFILE, |state| {
            state.launches = state.launches.saturating_add(1);
            state.dispatch_ns = state.dispatch_ns.saturating_add(dispatch_ns);
        });
    }

    RecurrentRuntimeOutput {
        context,
        rho,
        state_history: None,
    }
}

fn recurrent_attention_cube_exact_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
    capture_state_history: bool,
) -> RecurrentRuntimeOutput<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho = into_contiguous(rho);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, _time, _latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, query.meta.shape.dims::<4>()[2], embd]),
    );
    let state_history = capture_state_history.then(|| {
        empty_device::<R, f32>(
            client.clone(),
            query.device.clone(),
            Shape::new([
                batch,
                heads,
                query.meta.shape.dims::<4>()[2],
                query.meta.shape.dims::<4>()[3],
                embd,
            ]),
        )
    });

    let cube_dim = CubeDim::new_1d(RECURRENT_TILED_WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, RECURRENT_TILED_WORKGROUP_SIZE_X),
        heads as u32,
        batch as u32,
    );

    if let Some(history) = state_history.clone() {
        let _ = recurrent_attention_cube_exact_history_kernel::launch::<R>(
            &client,
            cube_count,
            cube_dim,
            query.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            rho.clone().into_tensor_arg(),
            decay.clone().into_tensor_arg(),
            context.clone().into_tensor_arg(),
            history.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
        );
        RecurrentRuntimeOutput {
            context,
            rho,
            state_history: Some(history),
        }
    } else {
        let _ = recurrent_attention_cube_exact_kernel::launch::<R>(
            &client,
            cube_count,
            cube_dim,
            query.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            rho.clone().into_tensor_arg(),
            decay.clone().into_tensor_arg(),
            context.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
        );

        RecurrentRuntimeOutput {
            context,
            rho,
            state_history: None,
        }
    }
}

fn recurrent_attention_cube_tiled_runtime<R: CubeRuntime>(
    query: CubeTensor<R>,
    value: CubeTensor<R>,
    rho: CubeTensor<R>,
    decay: CubeTensor<R>,
    meta: CubeTensor<R>,
    capture_state_history: bool,
) -> RecurrentRuntimeOutput<R> {
    let query = into_contiguous(query);
    let value = into_contiguous(value);
    let rho = into_contiguous(rho);
    let decay = into_contiguous(decay);
    let meta = into_contiguous(meta);

    let [batch, heads, _time, _latent] = query.meta.shape.dims::<4>();
    let embd = value.meta.shape.dims::<4>()[3];

    let client = query.client.clone();
    let device = query.device.clone();
    let context = empty_device::<R, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, query.meta.shape.dims::<4>()[2], embd]),
    );
    let state_history = capture_state_history.then(|| {
        empty_device::<R, f32>(
            client.clone(),
            query.device.clone(),
            Shape::new([
                batch,
                heads,
                query.meta.shape.dims::<4>()[2],
                query.meta.shape.dims::<4>()[3],
                embd,
            ]),
        )
    });

    let cube_dim = CubeDim::new_1d(WORKGROUP_SIZE_X);
    let cube_count = CubeCount::Static(
        div_ceil_u32(embd as u32, WORKGROUP_SIZE_X),
        heads as u32,
        batch as u32,
    );

    if let Some(history) = state_history.clone() {
        let _ = recurrent_attention_cube_tiled_history_kernel::launch::<R>(
            &client,
            cube_count,
            cube_dim,
            query.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            rho.clone().into_tensor_arg(),
            decay.clone().into_tensor_arg(),
            context.clone().into_tensor_arg(),
            history.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
            RECURRENT_TILED_WORKGROUP_SIZE_X as usize,
        );
        RecurrentRuntimeOutput {
            context,
            rho,
            state_history: Some(history),
        }
    } else {
        let _ = recurrent_attention_cube_tiled_kernel::launch::<R>(
            &client,
            cube_count,
            cube_dim,
            query.clone().into_tensor_arg(),
            value.clone().into_tensor_arg(),
            rho.clone().into_tensor_arg(),
            decay.clone().into_tensor_arg(),
            context.clone().into_tensor_arg(),
            meta.clone().into_tensor_arg(),
            RECURRENT_TILED_WORKGROUP_SIZE_X as usize,
        );

        RecurrentRuntimeOutput {
            context,
            rho,
            state_history: None,
        }
    }
}

#[cfg(feature = "cuda")]
fn use_cuda_tiled_recurrent_experimental() -> bool {
    std::env::var("BURN_DRAGON_CUDA_TILED_RECURRENT_EXPERIMENTAL")
        .ok()
        .as_deref()
        == Some("1")
}
