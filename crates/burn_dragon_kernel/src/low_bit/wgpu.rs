use super::*;

const PACKED_DOT_WGPU_WORKGROUP_SIZE_X: u32 = 64;
const LOW_BIT_PACKED_LOWRANK_WGSL_SHADER: &str = include_str!("../low_bit_packed_lowrank.wgsl");
const LOW_BIT_PACKED_LOWRANK_SCALE_PTR_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_lowrank_scale_ptr.wgsl");
const LOW_BIT_PACKED_LOWRANK_PREPACKED_SCALE_PTR_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_lowrank_prepacked_scale_ptr.wgsl");
const LOW_BIT_PACKED_LOWRANK_FROM_F32_SCALE_PTR_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_lowrank_from_f32_scale_ptr.wgsl");
const LOW_BIT_PACKED_DECODER_TAIL_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_decoder_tail.wgsl");
const LOW_BIT_PACKED_DECODER_TAIL_SCALE_PTR_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_decoder_tail_scale_ptr.wgsl");
const LOW_BIT_PACKED_DECODER_TAIL_PREPACKED_SCALE_PTR_WGSL_SHADER: &str =
    include_str!("../low_bit_packed_decoder_tail_prepacked_scale_ptr.wgsl");

struct PackedDotLowrankProjectionKernel;

impl KernelSource for PackedDotLowrankProjectionKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_LOWRANK_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotDecoderTailKernel;

impl KernelSource for PackedDotDecoderTailKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_DECODER_TAIL_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotLowrankProjectionScalePtrKernel;

impl KernelSource for PackedDotLowrankProjectionScalePtrKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_LOWRANK_SCALE_PTR_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotDecoderTailScalePtrKernel;

impl KernelSource for PackedDotDecoderTailScalePtrKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_DECODER_TAIL_SCALE_PTR_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotLowrankProjectionPrepackedScalePtrKernel;

impl KernelSource for PackedDotLowrankProjectionPrepackedScalePtrKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_LOWRANK_PREPACKED_SCALE_PTR_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotLowrankProjectionFromF32ScalePtrKernel;

impl KernelSource for PackedDotLowrankProjectionFromF32ScalePtrKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_LOWRANK_FROM_F32_SCALE_PTR_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

struct PackedDotDecoderTailPrepackedScalePtrKernel;

impl KernelSource for PackedDotDecoderTailPrepackedScalePtrKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(LOW_BIT_PACKED_DECODER_TAIL_PREPACKED_SCALE_PTR_WGSL_SHADER)
    }

    fn id(&self) -> burn_cubecl::cubecl::prelude::KernelId {
        KernelId::new::<Self>()
    }
}

#[derive(Clone, Copy, Default)]
struct WgpuPackedDotDeviceSupport {
    lowrank: Option<bool>,
    decoder_tail: Option<bool>,
}

static WGPU_PACKED_DOT_SUPPORT: OnceLock<Mutex<StdHashMap<String, WgpuPackedDotDeviceSupport>>> =
    OnceLock::new();

pub fn try_wgpu_packed_dot_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    diagnose_wgpu_packed_dot_lowrank_projection(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
    .ok()
}

pub fn try_wgpu_packed_dot_lowrank_projection_device_scale<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_lowrank_projection_wgpu_device_scale_impl(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
    .ok()
}

pub fn try_wgpu_packed_dot_lowrank_projection_prepacked_input_device_scale<B: BackendTrait>(
    input_packed: &BurnTensor<B, 4, Int>,
    weight_packed: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_lowrank_projection_prepacked_wgpu_device_scale_impl(
        input_packed,
        weight_packed,
        activation_scale,
        weight_scale,
        latent_out,
    )
    .ok()
}

pub fn try_wgpu_packed_dot_lowrank_projection_from_f32_device_scale<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight_packed: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
    qmax: i32,
    positive_only: bool,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    match try_packed_dot_lowrank_projection_from_f32_wgpu_device_scale_impl(
        input,
        weight_packed,
        activation_scale,
        weight_scale,
        latent_out,
        qmax,
        positive_only,
    ) {
        Ok(output) => Some(output),
        Err(err) => {
            if std::env::var_os("BDH_DEBUG_WGPU_LOWRANK_FROM_F32").is_some() {
                eprintln!("[low-bit][wgpu][lowrank-from-f32] {err}");
            }
            None
        }
    }
}

pub fn cached_wgpu_packed_dot_lowrank_support<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
) -> Option<bool> {
    let device_key = format!("{:?}", input.device());
    cached_wgpu_packed_dot_support(&device_key, "lowrank")
}

pub fn cached_wgpu_packed_dot_decoder_tail_support<B: BackendTrait>(
    y_neuron: &BurnTensor<B, 4>,
) -> Option<bool> {
    let device_key = format!("{:?}", y_neuron.device());
    cached_wgpu_packed_dot_support(&device_key, "decoder_tail")
}

pub fn try_cube_fused_packed_lowrank_projection_wgpu<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_lowrank_projection::<B, WgpuRuntime>(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
}

pub fn try_wgpu_packed_dot_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    diagnose_wgpu_packed_dot_decoder_tail(y_codes, weight_codes, activation_scale, weight_scale)
        .ok()
}

pub fn try_wgpu_packed_dot_decoder_tail_device_scale<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_decoder_tail_wgpu_device_scale_impl(
        y_codes,
        weight_codes,
        activation_scale,
        weight_scale,
    )
    .ok()
}

pub fn try_wgpu_packed_dot_decoder_tail_prepacked_input_device_scale<B: BackendTrait>(
    y_packed: &BurnTensor<B, 4, Int>,
    weight_packed: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_decoder_tail_prepacked_wgpu_device_scale_impl(
        y_packed,
        weight_packed,
        activation_scale,
        weight_scale,
    )
    .ok()
}

pub fn try_wgpu_quantize_pack_activation_i8x4<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    try_quantize_pack_activation_i8x4_wgpu_impl(input, activation_scale, qmax, positive_only).ok()
}

pub fn try_wgpu_quantize_activation_codes_i32<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Option<BurnTensor<B, 4, Int>>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    try_quantize_activation_codes_i32_wgpu_impl(input, activation_scale, qmax, positive_only).ok()
}

pub fn diagnose_wgpu_quantize_pack_activation_i8x4<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Result<BurnTensor<B, 4, Int>, String>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    try_quantize_pack_activation_i8x4_wgpu_impl(input, activation_scale, qmax, positive_only)
}

pub fn diagnose_wgpu_packed_dot_lowrank_projection<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_lowrank_projection_wgpu_impl(
        input_codes,
        weight_codes,
        activation_scale,
        weight_scale,
        latent_out,
    )
}

pub fn diagnose_wgpu_packed_dot_decoder_tail<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_packed_dot_decoder_tail_wgpu_impl(y_codes, weight_codes, activation_scale, weight_scale)
}

pub fn try_cube_fused_packed_decoder_tail_wgpu<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Option<BurnTensor<B, 4>>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    try_direct_packed_decoder_tail::<B, WgpuRuntime>(
        y_codes,
        weight_codes,
        activation_scale,
        weight_scale,
    )
}

fn create_lowrank_packed_dot_meta_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            artifact_latent as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot low-rank meta tensor")
}

fn wgpu_packed_dot_support_cache() -> &'static Mutex<StdHashMap<String, WgpuPackedDotDeviceSupport>>
{
    WGPU_PACKED_DOT_SUPPORT.get_or_init(|| Mutex::new(StdHashMap::new()))
}

fn cached_wgpu_packed_dot_support(device_key: &str, kind: &'static str) -> Option<bool> {
    let cache = wgpu_packed_dot_support_cache().lock().ok()?;
    let state = cache.get(device_key)?;
    match kind {
        "lowrank" => state.lowrank,
        "decoder_tail" => state.decoder_tail,
        _ => None,
    }
}

fn record_wgpu_packed_dot_support(device_key: &str, kind: &'static str, supported: bool) {
    if let Ok(mut cache) = wgpu_packed_dot_support_cache().lock() {
        let state = cache.entry(device_key.to_owned()).or_default();
        match kind {
            "lowrank" => state.lowrank = Some(supported),
            "decoder_tail" => state.decoder_tail = Some(supported),
            _ => {}
        }
    }
}

fn should_cache_wgpu_packed_dot_failure(error: &str) -> bool {
    error.contains("launch failed")
        || error.contains("Validation")
        || error.contains("packed_4x8_integer_dot_product")
        || error.contains("dot4I8Packed")
        || error.contains("requires extension")
}

fn create_lowrank_packed_dot_meta_wgpu_device_scale(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            latent_out as f32,
            artifact_latent as f32,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot low-rank device-scale meta tensor")
}

pub(super) fn packed_lowrank_projection_packed_dot_wgsl_runtime(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let meta = create_lowrank_packed_dot_meta_wgpu(
        &input.device,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_out]),
    );
    let kernel = SourceKernel::new(
        PackedDotLowrankProjectionKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(latent_out as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        input.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn packed_lowrank_projection_packed_dot_wgsl_runtime_device_scale(
    input: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    latent_out: usize,
    artifact_latent: usize,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let weight = into_contiguous(weight);
    let activation_scale = into_contiguous(activation_scale);
    let meta = create_lowrank_packed_dot_meta_wgpu_device_scale(
        &input.device,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_out]),
    );
    let kernel = SourceKernel::new(
        PackedDotLowrankProjectionScalePtrKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(latent_out as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        input.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        activation_scale.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn create_lowrank_packed_dot_meta_wgpu_prepacked_device_scale(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    pack_len: usize,
    latent_out: usize,
    artifact_latent: usize,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            pack_len as f32,
            latent_out as f32,
            artifact_latent as f32,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot low-rank prepacked device-scale meta tensor")
}

fn packed_lowrank_projection_prepacked_packed_dot_wgsl_runtime_device_scale(
    input_packed: CubeTensor<WgpuRuntime>,
    weight_packed: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    pack_len: usize,
    latent_out: usize,
    artifact_latent: usize,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input_packed = into_contiguous(input_packed);
    let weight_packed = into_contiguous(weight_packed);
    let activation_scale = into_contiguous(activation_scale);
    let meta = create_lowrank_packed_dot_meta_wgpu_prepacked_device_scale(
        &input_packed.device,
        batch,
        input_heads,
        heads,
        time,
        pack_len,
        latent_out,
        artifact_latent,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = input_packed.client.clone();
    let device = input_packed.device.clone();
    let output = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_out]),
    );
    let kernel = SourceKernel::new(
        PackedDotLowrankProjectionPrepackedScalePtrKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(latent_out as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        input_packed.handle.clone().binding(),
        weight_packed.handle.clone().binding(),
        output.handle.clone().binding(),
        activation_scale.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn create_lowrank_packed_dot_meta_wgpu_from_f32_device_scale(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    pack_len: usize,
    latent_out: usize,
    artifact_latent: usize,
    qmax: i32,
    positive_only: bool,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            input_heads as f32,
            heads as f32,
            time as f32,
            embd as f32,
            pack_len as f32,
            latent_out as f32,
            artifact_latent as f32,
            qmax as f32,
            if positive_only { 1.0 } else { 0.0 },
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot low-rank from-f32 device-scale meta tensor")
}

fn packed_lowrank_projection_from_f32_packed_dot_wgsl_runtime_device_scale(
    input: CubeTensor<WgpuRuntime>,
    weight_packed: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    batch: usize,
    input_heads: usize,
    heads: usize,
    time: usize,
    embd: usize,
    pack_len: usize,
    latent_out: usize,
    artifact_latent: usize,
    qmax: i32,
    positive_only: bool,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let weight_packed = into_contiguous(weight_packed);
    let activation_scale = into_contiguous(activation_scale);
    let meta = create_lowrank_packed_dot_meta_wgpu_from_f32_device_scale(
        &input.device,
        batch,
        input_heads,
        heads,
        time,
        embd,
        pack_len,
        latent_out,
        artifact_latent,
        qmax,
        positive_only,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<WgpuRuntime, f32>(
        client.clone(),
        device,
        Shape::new([batch, heads, time, latent_out]),
    );
    let kernel = SourceKernel::new(
        PackedDotLowrankProjectionFromF32ScalePtrKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(latent_out as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        (batch * heads) as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        input.handle.clone().binding(),
        weight_packed.handle.clone().binding(),
        output.handle.clone().binding(),
        activation_scale.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn create_decoder_tail_packed_dot_meta_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            artifact_latent_per_head as f32,
            dim as f32,
            activation_scale,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot decoder-tail meta tensor")
}

fn create_decoder_tail_packed_dot_meta_wgpu_device_scale(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent as f32,
            artifact_latent_per_head as f32,
            dim as f32,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot decoder-tail device-scale meta tensor")
}

pub(super) fn packed_decoder_tail_packed_dot_wgsl_runtime(
    y: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let y = into_contiguous(y);
    let weight = into_contiguous(weight);
    let meta = create_decoder_tail_packed_dot_meta_wgpu(
        &y.device,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = y.client.clone();
    let device = y.device.clone();
    let output =
        empty_device::<WgpuRuntime, f32>(client.clone(), device, Shape::new([batch, 1, time, dim]));
    let kernel = SourceKernel::new(
        PackedDotDecoderTailKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(dim as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        batch as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        y.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn packed_decoder_tail_packed_dot_wgsl_runtime_device_scale(
    y: CubeTensor<WgpuRuntime>,
    weight: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    batch: usize,
    heads: usize,
    time: usize,
    latent: usize,
    artifact_latent_per_head: usize,
    dim: usize,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let y = into_contiguous(y);
    let weight = into_contiguous(weight);
    let activation_scale = into_contiguous(activation_scale);
    let meta = create_decoder_tail_packed_dot_meta_wgpu_device_scale(
        &y.device,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = y.client.clone();
    let device = y.device.clone();
    let output =
        empty_device::<WgpuRuntime, f32>(client.clone(), device, Shape::new([batch, 1, time, dim]));
    let kernel = SourceKernel::new(
        PackedDotDecoderTailScalePtrKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(dim as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        batch as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        y.handle.clone().binding(),
        weight.handle.clone().binding(),
        output.handle.clone().binding(),
        activation_scale.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn create_decoder_tail_packed_dot_meta_wgpu_prepacked_device_scale(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    batch: usize,
    heads: usize,
    time: usize,
    latent_pack: usize,
    dim: usize,
    weight_scale: f32,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            batch as f32,
            heads as f32,
            time as f32,
            latent_pack as f32,
            dim as f32,
            weight_scale,
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu packed-dot decoder-tail prepacked device-scale meta tensor")
}

fn packed_decoder_tail_prepacked_packed_dot_wgsl_runtime_device_scale(
    y_packed: CubeTensor<WgpuRuntime>,
    weight_packed: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    batch: usize,
    heads: usize,
    time: usize,
    latent_pack: usize,
    dim: usize,
    weight_scale: f32,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let y_packed = into_contiguous(y_packed);
    let weight_packed = into_contiguous(weight_packed);
    let activation_scale = into_contiguous(activation_scale);
    let meta = create_decoder_tail_packed_dot_meta_wgpu_prepacked_device_scale(
        &y_packed.device,
        batch,
        heads,
        time,
        latent_pack,
        dim,
        weight_scale,
    );
    let meta = into_contiguous(meta);

    let client = y_packed.client.clone();
    let device = y_packed.device.clone();
    let output =
        empty_device::<WgpuRuntime, f32>(client.clone(), device, Shape::new([batch, 1, time, dim]));
    let kernel = SourceKernel::new(
        PackedDotDecoderTailPrepackedScalePtrKernel,
        CubeDim::new_3d(PACKED_DOT_WGPU_WORKGROUP_SIZE_X, 1, 1),
    );
    let count = CubeCount::Static(
        div_ceil_u32(dim as u32, PACKED_DOT_WGPU_WORKGROUP_SIZE_X),
        time as u32,
        batch as u32,
    );
    let bindings = KernelArguments::new().with_buffers(vec![
        y_packed.handle.clone().binding(),
        weight_packed.handle.clone().binding(),
        output.handle.clone().binding(),
        activation_scale.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.launch(Box::new(kernel), count, bindings);
    Ok(output)
}

fn create_quantize_pack_i8x4_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    outer: usize,
    inner: usize,
    pack_len: usize,
    qmax: i32,
    positive_only: bool,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            outer as f32,
            inner as f32,
            pack_len as f32,
            qmax as f32,
            if positive_only { 1.0 } else { 0.0 },
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu quantize-pack params tensor")
}

fn create_quantize_codes_i32_params_wgpu(
    device: &<WgpuCubeBackend as BackendTrait>::Device,
    total: usize,
    qmax: i32,
    positive_only: bool,
) -> CubeTensor<WgpuRuntime> {
    let params = Tensor::<WgpuCubeBackend, 1>::from_data(
        [
            total as f32,
            qmax as f32,
            if positive_only { 1.0 } else { 0.0 },
        ],
        device,
    );
    try_cast_float_primitive::<WgpuCubeBackend, _>(params.into_primitive().tensor())
        .expect("wgpu quantize-codes params tensor")
}

fn quantize_pack_i8x4_cube_runtime(
    input: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    outer: usize,
    inner: usize,
    pack_len: usize,
    qmax: i32,
    positive_only: bool,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let activation_scale = into_contiguous(activation_scale);
    let params = create_quantize_pack_i8x4_params_wgpu(
        &input.device,
        outer,
        inner,
        pack_len,
        qmax,
        positive_only,
    );
    let params = into_contiguous(params);

    let client = input.client.clone();
    let device = input.device.clone();
    let output =
        empty_device::<WgpuRuntime, i32>(client.clone(), device, Shape::new([outer, pack_len]));
    let cube_dim_x = cube_workgroup_size_x::<WgpuRuntime>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(
        div_ceil_u32((outer * pack_len) as u32, cube_dim_x),
        1u32,
        1u32,
    );
    let _ = quantize_pack_i8x4_cube_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        input.clone().into_tensor_arg(),
        activation_scale.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    Ok(output)
}

fn quantize_codes_i32_cube_runtime(
    input: CubeTensor<WgpuRuntime>,
    activation_scale: CubeTensor<WgpuRuntime>,
    shape: [usize; 4],
    qmax: i32,
    positive_only: bool,
) -> Result<CubeTensor<WgpuRuntime>, String> {
    let input = into_contiguous(input);
    let activation_scale = into_contiguous(activation_scale);
    let total = shape.into_iter().product::<usize>();
    let params = create_quantize_codes_i32_params_wgpu(&input.device, total, qmax, positive_only);
    let params = into_contiguous(params);
    let groups = total.div_ceil(4);

    let client = input.client.clone();
    let device = input.device.clone();
    let output = empty_device::<WgpuRuntime, i32>(client.clone(), device, Shape::new(shape));
    let cube_dim_x = cube_workgroup_size_x::<WgpuRuntime>();
    let cube_dim = CubeDim::new_1d(cube_dim_x);
    let cube_count = CubeCount::Static(div_ceil_u32(groups as u32, cube_dim_x), 1u32, 1u32);
    let _ = quantize_codes_i32_cube_kernel::launch::<WgpuRuntime>(
        &client,
        cube_count,
        cube_dim,
        input.clone().into_tensor_arg(),
        activation_scale.clone().into_tensor_arg(),
        output.clone().into_tensor_arg(),
        params.clone().into_tensor_arg(),
    );
    Ok(output)
}

fn try_packed_dot_lowrank_projection_wgpu_impl<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: f32,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, weight_embd, artifact_latent] = weight_codes.shape().dims::<3>();
    if weight_embd != embd
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return Err(format!(
            "wgpu packed-dot lowrank shape mismatch: input_heads={input_heads} heads={heads} embd={embd} weight_embd={weight_embd} latent_out={latent_out} artifact_latent={artifact_latent}"
        ));
    }

    let input = resolve_wgpu_int_tensor_read::<B, 4>(input_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 3>(weight_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 {
        return Err(format!(
            "wgpu packed-dot lowrank dtype mismatch: input={:?} weight={:?}",
            input.dtype, weight.dtype
        ));
    }
    let device_key = format!("{:?}", input.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "lowrank"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot lowrank cached unsupported for device {device_key}"
        ));
    }

    let output_origin = resolve_wgpu_float_output_origin_from_device::<B>(&input_codes.device())
        .ok_or_else(|| {
            format!(
                "wgpu packed-dot lowrank output-origin resolve failed for backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let output = match packed_lowrank_projection_packed_dot_wgsl_runtime(
        input,
        weight,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        activation_scale,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "lowrank", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "lowrank", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_lowrank_projection_wgpu_device_scale_impl<B: BackendTrait>(
    input_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input_codes.shape().dims::<4>();
    let [heads, weight_embd, artifact_latent] = weight_codes.shape().dims::<3>();
    if weight_embd != embd
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return Err(format!(
            "wgpu packed-dot lowrank device-scale shape mismatch: input_heads={input_heads} heads={heads} embd={embd} weight_embd={weight_embd} latent_out={latent_out} artifact_latent={artifact_latent}"
        ));
    }

    let input = resolve_wgpu_int_tensor_read::<B, 4>(input_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank device-scale cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 3>(weight_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank device-scale cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank device-scale cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu packed-dot lowrank device-scale dtype mismatch: input={:?} weight={:?} scale={:?}",
            input.dtype, weight.dtype, scale.dtype
        ));
    }
    let device_key = format!("{:?}", input.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "lowrank"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot lowrank device-scale cached unsupported for device {device_key}"
        ));
    }

    let output_origin = resolve_wgpu_float_output_origin_from_device::<B>(&input_codes.device())
        .ok_or_else(|| {
            format!(
                "wgpu packed-dot lowrank device-scale output-origin resolve failed for backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let output = match packed_lowrank_projection_packed_dot_wgsl_runtime_device_scale(
        input,
        weight,
        scale,
        batch,
        input_heads,
        heads,
        time,
        embd,
        latent_out,
        artifact_latent,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "lowrank", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "lowrank", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank device-scale float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_lowrank_projection_prepacked_wgpu_device_scale_impl<B: BackendTrait>(
    input_packed: &BurnTensor<B, 4, Int>,
    weight_packed: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, pack_len] = input_packed.shape().dims::<4>();
    let [heads, weight_pack_len, artifact_latent] = weight_packed.shape().dims::<3>();
    if weight_pack_len != pack_len
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return Err(format!(
            "wgpu packed-dot lowrank prepacked device-scale shape mismatch: input_heads={input_heads} heads={heads} pack_len={pack_len} weight_pack_len={weight_pack_len} latent_out={latent_out} artifact_latent={artifact_latent}"
        ));
    }

    let input = resolve_wgpu_int_tensor_read::<B, 4>(input_packed).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank prepacked device-scale cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 3>(weight_packed).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank prepacked device-scale cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank prepacked device-scale cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu packed-dot lowrank prepacked device-scale dtype mismatch: input={:?} weight={:?} scale={:?}",
            input.dtype, weight.dtype, scale.dtype
        ));
    }
    let device_key = format!("{:?}", input.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "lowrank"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot lowrank prepacked device-scale cached unsupported for device {device_key}"
        ));
    }

    let output_origin =
        resolve_wgpu_float_output_origin_from_device::<B>(&input_packed.device()).ok_or_else(
            || {
                format!(
                    "wgpu packed-dot lowrank prepacked device-scale output-origin resolve failed for backend {}",
                    core::any::type_name::<B>()
                )
            },
        )?;
    let output = match packed_lowrank_projection_prepacked_packed_dot_wgsl_runtime_device_scale(
        input,
        weight,
        scale,
        batch,
        input_heads,
        heads,
        time,
        pack_len,
        latent_out,
        artifact_latent,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "lowrank", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "lowrank", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank prepacked device-scale float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_lowrank_projection_from_f32_wgpu_device_scale_impl<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    weight_packed: &BurnTensor<B, 3, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
    latent_out: usize,
    qmax: i32,
    positive_only: bool,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, input_heads, time, embd] = input.shape().dims::<4>();
    let pack_len = embd.div_ceil(4);
    let [heads, weight_pack_len, artifact_latent] = weight_packed.shape().dims::<3>();
    if weight_pack_len != pack_len
        || !(input_heads == 1 || input_heads == heads)
        || latent_out > artifact_latent
    {
        return Err(format!(
            "wgpu packed-dot lowrank from-f32 device-scale shape mismatch: input_heads={input_heads} heads={heads} embd={embd} pack_len={pack_len} weight_pack_len={weight_pack_len} latent_out={latent_out} artifact_latent={artifact_latent}"
        ));
    }

    let (input, output_origin) = resolve_wgpu_float_tensor_with_output_origin::<B, 4>(input)
        .ok_or_else(|| {
            format!(
                "wgpu packed-dot lowrank from-f32 device-scale cast failed for input backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 3>(weight_packed).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank from-f32 device-scale cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank from-f32 device-scale cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input.dtype != DType::F32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu packed-dot lowrank from-f32 device-scale dtype mismatch: input={:?} weight={:?} scale={:?}",
            input.dtype, weight.dtype, scale.dtype
        ));
    }
    let device_key = format!("{:?}", input.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "lowrank"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot lowrank from-f32 device-scale cached unsupported for device {device_key}"
        ));
    }

    let output = match packed_lowrank_projection_from_f32_packed_dot_wgsl_runtime_device_scale(
        input,
        weight,
        scale,
        batch,
        input_heads,
        heads,
        time,
        embd,
        pack_len,
        latent_out,
        artifact_latent,
        qmax,
        positive_only,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "lowrank", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "lowrank", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot lowrank from-f32 device-scale float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_decoder_tail_wgpu_impl<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: f32,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [artifact_latent_total, dim] = weight_codes.shape().dims::<2>();
    if artifact_latent_total % heads != 0 {
        return Err(format!(
            "wgpu packed-dot decoder-tail shape mismatch: artifact_latent_total={artifact_latent_total} heads={heads}"
        ));
    }
    let artifact_latent_per_head = artifact_latent_total / heads;
    if latent > artifact_latent_per_head {
        return Err(format!(
            "wgpu packed-dot decoder-tail latent mismatch: latent={latent} artifact_latent_per_head={artifact_latent_per_head}"
        ));
    }

    let y = resolve_wgpu_int_tensor_read::<B, 4>(y_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 2>(weight_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 {
        return Err(format!(
            "wgpu packed-dot decoder-tail dtype mismatch: y={:?} weight={:?}",
            y.dtype, weight.dtype
        ));
    }
    let device_key = format!("{:?}", y.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "decoder_tail"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot decoder-tail cached unsupported for device {device_key}"
        ));
    }

    let output_origin = resolve_wgpu_float_output_origin_from_device::<B>(&y_codes.device())
        .ok_or_else(|| {
            format!(
                "wgpu packed-dot decoder-tail output-origin resolve failed for backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let output = match packed_decoder_tail_packed_dot_wgsl_runtime(
        y,
        weight,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        activation_scale,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "decoder_tail", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "decoder_tail", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_decoder_tail_wgpu_device_scale_impl<B: BackendTrait>(
    y_codes: &BurnTensor<B, 4, Int>,
    weight_codes: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent] = y_codes.shape().dims::<4>();
    let [artifact_latent_total, dim] = weight_codes.shape().dims::<2>();
    if artifact_latent_total % heads != 0 {
        return Err(format!(
            "wgpu packed-dot decoder-tail device-scale shape mismatch: artifact_latent_total={artifact_latent_total} heads={heads}"
        ));
    }
    let artifact_latent_per_head = artifact_latent_total / heads;
    if latent > artifact_latent_per_head {
        return Err(format!(
            "wgpu packed-dot decoder-tail device-scale latent mismatch: latent={latent} artifact_latent_per_head={artifact_latent_per_head}"
        ));
    }

    let y = resolve_wgpu_int_tensor_read::<B, 4>(y_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail device-scale cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 2>(weight_codes).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail device-scale cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail device-scale cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu packed-dot decoder-tail device-scale dtype mismatch: y={:?} weight={:?} scale={:?}",
            y.dtype, weight.dtype, scale.dtype
        ));
    }
    let device_key = format!("{:?}", y.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "decoder_tail"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot decoder-tail device-scale cached unsupported for device {device_key}"
        ));
    }

    let output_origin =
        resolve_wgpu_float_output_origin_from_device::<B>(&y_codes.device()).ok_or_else(|| {
            format!(
                "wgpu packed-dot decoder-tail device-scale output-origin resolve failed for backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let output = match packed_decoder_tail_packed_dot_wgsl_runtime_device_scale(
        y,
        weight,
        scale,
        batch,
        heads,
        time,
        latent,
        artifact_latent_per_head,
        dim,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "decoder_tail", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "decoder_tail", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail device-scale float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_packed_dot_decoder_tail_prepacked_wgpu_device_scale_impl<B: BackendTrait>(
    y_packed: &BurnTensor<B, 4, Int>,
    weight_packed: &BurnTensor<B, 2, Int>,
    activation_scale: &BurnTensor<B, 1>,
    weight_scale: f32,
) -> Result<BurnTensor<B, 4>, String>
where
    B::IntTensorPrimitive: 'static,
    B::FloatTensorPrimitive: 'static,
{
    let [batch, heads, time, latent_pack] = y_packed.shape().dims::<4>();
    let [packed_latent_total, dim] = weight_packed.shape().dims::<2>();
    if packed_latent_total % heads != 0 {
        return Err(format!(
            "wgpu packed-dot decoder-tail prepacked device-scale shape mismatch: packed_latent_total={packed_latent_total} heads={heads}"
        ));
    }
    let weight_pack_len = packed_latent_total / heads;
    if weight_pack_len != latent_pack {
        return Err(format!(
            "wgpu packed-dot decoder-tail prepacked device-scale latent mismatch: latent_pack={latent_pack} weight_pack_len={weight_pack_len}"
        ));
    }

    let y = resolve_wgpu_int_tensor_read::<B, 4>(y_packed).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail prepacked device-scale cast failed for input backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let weight = resolve_wgpu_int_tensor_read::<B, 2>(weight_packed).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail prepacked device-scale cast failed for weight backend {}",
            core::any::type_name::<B>()
        )
    })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail prepacked device-scale cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if y.dtype != DType::I32 || weight.dtype != DType::I32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu packed-dot decoder-tail prepacked device-scale dtype mismatch: y={:?} weight={:?} scale={:?}",
            y.dtype, weight.dtype, scale.dtype
        ));
    }
    let device_key = format!("{:?}", y.device);
    if matches!(
        cached_wgpu_packed_dot_support(&device_key, "decoder_tail"),
        Some(false)
    ) {
        return Err(format!(
            "wgpu packed-dot decoder-tail prepacked device-scale cached unsupported for device {device_key}"
        ));
    }

    let output_origin =
        resolve_wgpu_float_output_origin_from_device::<B>(&y_packed.device()).ok_or_else(|| {
            format!(
                "wgpu packed-dot decoder-tail prepacked device-scale output-origin resolve failed for backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let output = match packed_decoder_tail_prepacked_packed_dot_wgsl_runtime_device_scale(
        y,
        weight,
        scale,
        batch,
        heads,
        time,
        latent_pack,
        dim,
        weight_scale,
    ) {
        Ok(output) => {
            record_wgpu_packed_dot_support(&device_key, "decoder_tail", true);
            output
        }
        Err(err) => {
            if should_cache_wgpu_packed_dot_failure(&err) {
                record_wgpu_packed_dot_support(&device_key, "decoder_tail", false);
            }
            return Err(err);
        }
    };
    wrap_wgpu_float_output_for_backend::<B, 4>(output, output_origin).ok_or_else(|| {
        format!(
            "wgpu packed-dot decoder-tail prepacked device-scale float cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_quantize_pack_activation_i8x4_wgpu_impl<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Result<BurnTensor<B, 4, Int>, String>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let [batch, heads, time, inner] = input.shape().dims::<4>();
    let outer = batch * heads * time;
    let pack_len = inner.div_ceil(4);
    let (input_tensor, output_origin) = resolve_wgpu_float_tensor_with_output_origin::<B, 4>(input)
        .ok_or_else(|| {
            format!(
                "wgpu quantize-pack cast failed for input backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu quantize-pack cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input_tensor.dtype != DType::F32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu quantize-pack dtype mismatch: input={:?} scale={:?}",
            input_tensor.dtype, scale.dtype
        ));
    }
    let output = quantize_pack_i8x4_cube_runtime(
        input_tensor,
        scale,
        outer,
        inner,
        pack_len,
        qmax,
        positive_only,
    )?;
    wrap_wgpu_int_output_for_backend::<B, 4>(
        output,
        int_output_origin_from_float_origin(&output_origin),
    )
    .map(|tensor| tensor.reshape([batch, heads, time, pack_len]))
    .ok_or_else(|| {
        format!(
            "wgpu quantize-pack int cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}

fn try_quantize_activation_codes_i32_wgpu_impl<B: BackendTrait>(
    input: &BurnTensor<B, 4>,
    activation_scale: &BurnTensor<B, 1>,
    qmax: i32,
    positive_only: bool,
) -> Result<BurnTensor<B, 4, Int>, String>
where
    B::FloatTensorPrimitive: 'static,
    B::IntTensorPrimitive: 'static,
{
    let shape = input.shape().dims::<4>();
    let (input_tensor, output_origin) = resolve_wgpu_float_tensor_with_output_origin::<B, 4>(input)
        .ok_or_else(|| {
            format!(
                "wgpu quantize-codes cast failed for input backend {}",
                core::any::type_name::<B>()
            )
        })?;
    let scale = resolve_wgpu_float_tensor_read::<B, 1>(activation_scale).ok_or_else(|| {
        format!(
            "wgpu quantize-codes cast failed for scale backend {}",
            core::any::type_name::<B>()
        )
    })?;
    if input_tensor.dtype != DType::F32 || scale.dtype != DType::F32 {
        return Err(format!(
            "wgpu quantize-codes dtype mismatch: input={:?} scale={:?}",
            input_tensor.dtype, scale.dtype
        ));
    }
    let output = quantize_codes_i32_cube_runtime(input_tensor, scale, shape, qmax, positive_only)?;
    wrap_wgpu_int_output_for_backend::<B, 4>(
        output,
        int_output_origin_from_float_origin(&output_origin),
    )
    .ok_or_else(|| {
        format!(
            "wgpu quantize-codes int cast failed for backend {}",
            core::any::type_name::<B>()
        )
    })
}
