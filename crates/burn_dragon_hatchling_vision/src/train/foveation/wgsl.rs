#![allow(clippy::too_many_arguments)]

use std::any::{Any, TypeId};

use burn::tensor::{DType, Shape, TensorData, TensorPrimitive};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::Tensor as BurnTensor;
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::empty_device;
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::FusionTensor;
use burn_fusion::stream::StreamId;
use burn_wgpu::{KernelSource, SourceKernel, SourceTemplate, WgpuRuntime};
use cubecl::prelude::*;
use cubecl_runtime::server::Bindings;

use crate::FOVEATION_BUFFER_SHADER;

use crate::train::saccade::{SaccadeLaplacianImages, SaccadeMipLevel};
use burn_dragon_hatchling_core::config::VisionFoveaWarpMode;

const MAX_LEVELS: usize = 8;
const META_HEADER_LEN: usize = 10;
const META_LEN: usize = META_HEADER_LEN + (MAX_LEVELS * 6) + 3;
const PARAM_STRIDE: usize = 5;
const WORKGROUP_SIZE: u32 = 8;

pub(crate) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    matches_type::<
        B::FloatTensorPrimitive,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>,
    >() || matches_type::<
        B::FloatTensorPrimitive,
        FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
    >() || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
}

pub(crate) fn try_foveated_patch_wgsl<B: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
    base_grid: &BurnTensor<B, 4>,
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    warp_mode: VisionFoveaWarpMode,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if levels.is_empty() {
        return None;
    }
    if !supports_backend::<B>() {
        return None;
    }
    let [_, patch_h, patch_w, _] = base_grid.shape().dims::<4>();
    if patch_h == 0 || patch_w == 0 {
        return None;
    }

    let use_laplacian = laplacian_images.is_some();
    let level_count = levels.len();
    if level_count > MAX_LEVELS {
        return None;
    }
    if use_laplacian {
        let residual_count = laplacian_images
            .as_ref()
            .map(|laplacian| laplacian.residuals.len())
            .unwrap_or(0);
        if residual_count + 1 > MAX_LEVELS {
            return None;
        }
    }

    if let Some(result) =
        try_foveated_patch_wgsl_fusion::<B, u32>(
            levels,
            center_x,
            center_y,
            sigma_px,
            radius_px,
            lod_sigma,
            laplacian_images,
            patch_h,
            patch_w,
            warp_mode,
        )
    {
        return Some(result);
    }
    if let Some(result) =
        try_foveated_patch_wgsl_fusion::<B, u8>(
            levels,
            center_x,
            center_y,
            sigma_px,
            radius_px,
            lod_sigma,
            laplacian_images,
            patch_h,
            patch_w,
            warp_mode,
        )
    {
        return Some(result);
    }
    try_foveated_patch_wgsl_direct::<B>(
        levels,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        laplacian_images,
        patch_h,
        patch_w,
        warp_mode,
    )
}

fn try_foveated_patch_wgsl_fusion<B: BackendTrait, BT: BoolElement>(
    levels: &[SaccadeMipLevel<B>],
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    patch_h: usize,
    patch_w: usize,
    warp_mode: VisionFoveaWarpMode,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>>>() {
        return None;
    }

    let use_laplacian = laplacian_images.is_some();
    let gaussian_buffer = build_gaussian_buffer(levels, use_laplacian)?;
    let (residual_buffer, residual_meta) =
        build_residual_buffer(laplacian_images, &gaussian_buffer.device())?;
    let params = build_params(center_x, center_y, sigma_px, radius_px, lod_sigma);
    let meta = build_meta(
        levels,
        laplacian_images,
        &gaussian_buffer,
        &residual_meta,
        patch_h,
        patch_w,
        warp_mode,
    );

    let prim_gaussian = gaussian_buffer.clone().into_primitive().tensor();
    let fusion_gaussian: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim_gaussian)?;
    let fusion_client = fusion_gaussian.client.clone();
    let gaussian = fusion_client
        .resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion_gaussian);
    if gaussian.dtype != DType::F32 {
        return None;
    }

    let residual = resolve_fusion_tensor::<B, BT, 1>(&residual_buffer)?;
    let params = resolve_fusion_tensor::<B, BT, 1>(&params)?;
    let meta = resolve_fusion_tensor::<B, BT, 1>(&meta)?;

    let [batch, channels, _, _] = levels.first()?.image.shape().dims::<4>();
    let output = foveated_patch_wgsl_runtime::<WgpuRuntime>(
        gaussian,
        residual,
        params,
        meta,
        patch_h,
        patch_w,
        batch,
        channels,
    );
    let shape = output.shape.clone();
    let dtype = output.dtype;
    let handle = output.into();
    let fusion_out = fusion_client.register_tensor(handle, shape, StreamId::current(), dtype);
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn try_foveated_patch_wgsl_direct<B: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    laplacian_images: Option<&SaccadeLaplacianImages<B>>,
    patch_h: usize,
    patch_w: usize,
    warp_mode: VisionFoveaWarpMode,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>() {
        return None;
    }

    let use_laplacian = laplacian_images.is_some();
    let gaussian = build_gaussian_buffer(levels, use_laplacian)?;
    let (residual, residual_meta) =
        build_residual_buffer(laplacian_images, &gaussian.device())?;
    let params = build_params(center_x, center_y, sigma_px, radius_px, lod_sigma);
    let meta = build_meta(
        levels,
        laplacian_images,
        &gaussian,
        &residual_meta,
        patch_h,
        patch_w,
        warp_mode,
    );

    let gaussian = resolve_direct_tensor::<B, 1>(&gaussian)?;
    let residual = resolve_direct_tensor::<B, 1>(&residual)?;
    let params = resolve_direct_tensor::<B, 1>(&params)?;
    let meta = resolve_direct_tensor::<B, 1>(&meta)?;

    let [batch, channels, _, _] = levels.first()?.image.shape().dims::<4>();
    let output = foveated_patch_wgsl_runtime::<WgpuRuntime>(
        gaussian,
        residual,
        params,
        meta,
        patch_h,
        patch_w,
        batch,
        channels,
    );
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn build_params<B: BackendTrait>(
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
) -> BurnTensor<B, 1> {
    let [batch, _, _] = center_x.shape().dims::<3>();
    BurnTensor::cat(
        vec![
            center_x.clone(),
            center_y.clone(),
            sigma_px.clone(),
            radius_px.clone(),
            lod_sigma.clone(),
        ],
        2,
    )
    .reshape([batch * PARAM_STRIDE])
}

fn build_gaussian_buffer<B: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
    use_laplacian: bool,
) -> Option<BurnTensor<B, 1>>
where
    B::FloatTensorPrimitive: 'static,
{
    if use_laplacian {
        let device = levels.first()?.image.device();
        return Some(BurnTensor::<B, 1>::zeros([1], &device));
    }
    let (buffer, _) = pack_levels(levels.iter().map(|level| &level.image).collect())?;
    Some(buffer)
}

#[derive(Clone, Copy)]
struct ResidualMeta {
    offsets: [u32; MAX_LEVELS],
    widths: [u32; MAX_LEVELS],
    heights: [u32; MAX_LEVELS],
    coarse_offset: u32,
    coarse_width: u32,
    coarse_height: u32,
    residual_count: usize,
}

fn build_residual_buffer<B: BackendTrait>(
    laplacian: Option<&SaccadeLaplacianImages<B>>,
    device: &B::Device,
) -> Option<(BurnTensor<B, 1>, ResidualMeta)>
where
    B::FloatTensorPrimitive: 'static,
{
    if let Some(laplacian) = laplacian {
        let residuals: Vec<&BurnTensor<B, 4>> = laplacian.residuals.iter().collect();
        let (buffer, meta) = pack_residuals(residuals, &laplacian.coarse)?;
        return Some((buffer, meta));
    }

    let buffer = BurnTensor::<B, 1>::zeros([1], device);
    let meta = ResidualMeta {
        offsets: [0u32; MAX_LEVELS],
        widths: [0u32; MAX_LEVELS],
        heights: [0u32; MAX_LEVELS],
        coarse_offset: 0,
        coarse_width: 1,
        coarse_height: 1,
        residual_count: 0,
    };
    Some((buffer, meta))
}

fn build_meta<B: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
    laplacian: Option<&SaccadeLaplacianImages<B>>,
    gaussian: &BurnTensor<B, 1>,
    residual_meta: &ResidualMeta,
    patch_h: usize,
    patch_w: usize,
    warp_mode: VisionFoveaWarpMode,
) -> BurnTensor<B, 1> {
    let device = gaussian.device();
    let [batch, channels, _, _] = levels
        .first()
        .map(|level| level.image.shape().dims::<4>())
        .unwrap_or([0, 0, 0, 0]);
    let level_count = levels.len().min(MAX_LEVELS);
    let mode = if laplacian.is_some() { 1u32 } else { 0u32 };

    let base_source = if let Some(laplacian) = laplacian {
        laplacian.residuals.first().unwrap_or(&laplacian.coarse)
    } else {
        &levels[0].image
    };
    let [_, _, base_h, base_w] = base_source.shape().dims::<4>();

    let mut meta = vec![0.0f32; META_LEN];
    meta[0] = patch_w as f32;
    meta[1] = patch_h as f32;
    meta[2] = channels as f32;
    meta[3] = level_count as f32;
    meta[4] = residual_meta.residual_count as f32;
    meta[5] = mode as f32;
    meta[6] = match warp_mode {
        VisionFoveaWarpMode::Warped => 0.0,
        VisionFoveaWarpMode::Patched => 1.0,
    };
    meta[7] = base_w as f32;
    meta[8] = base_h as f32;
    meta[9] = batch as f32;

    let (gauss_offsets, gauss_widths, gauss_heights) =
        pack_level_meta(levels.iter().map(|level| &level.image).collect(), level_count);
    write_meta_array(&mut meta, 10, &gauss_offsets);
    write_meta_array(&mut meta, 10 + MAX_LEVELS, &gauss_widths);
    write_meta_array(&mut meta, 10 + MAX_LEVELS * 2, &gauss_heights);

    write_meta_array(&mut meta, 10 + MAX_LEVELS * 3, &residual_meta.offsets);
    write_meta_array(&mut meta, 10 + MAX_LEVELS * 4, &residual_meta.widths);
    write_meta_array(&mut meta, 10 + MAX_LEVELS * 5, &residual_meta.heights);

    let coarse_base = 10 + MAX_LEVELS * 6;
    meta[coarse_base] = residual_meta.coarse_offset as f32;
    meta[coarse_base + 1] = residual_meta.coarse_width as f32;
    meta[coarse_base + 2] = residual_meta.coarse_height as f32;

    BurnTensor::<B, 1>::from_data(TensorData::new(meta, [META_LEN]), &device)
}

fn pack_level_meta<B: BackendTrait>(
    levels: Vec<&BurnTensor<B, 4>>,
    level_count: usize,
) -> ([u32; MAX_LEVELS], [u32; MAX_LEVELS], [u32; MAX_LEVELS]) {
    let mut offsets = [0u32; MAX_LEVELS];
    let mut widths = [0u32; MAX_LEVELS];
    let mut heights = [0u32; MAX_LEVELS];
    let mut offset = 0u32;
    for (idx, level) in levels.into_iter().take(level_count).enumerate() {
        let [batch, channels, height, width] = level.shape().dims::<4>();
        offsets[idx] = offset;
        widths[idx] = width as u32;
        heights[idx] = height as u32;
        let elems = batch * channels * height * width;
        offset = offset.saturating_add(elems as u32);
    }
    (offsets, widths, heights)
}

fn pack_levels<B: BackendTrait>(
    levels: Vec<&BurnTensor<B, 4>>,
) -> Option<(BurnTensor<B, 1>, usize)>
where
    B::FloatTensorPrimitive: 'static,
{
    let first = levels.first()?;
    let [batch, channels, _, _] = first.shape().dims::<4>();
    let mut flats = Vec::with_capacity(levels.len());
    for level in levels {
        let [b, c, h, w] = level.shape().dims::<4>();
        if b != batch || c != channels {
            return None;
        }
        let elems = b * c * h * w;
        flats.push(level.clone().reshape([elems]));
    }
    let merged = if flats.len() == 1 {
        flats[0].clone()
    } else {
        BurnTensor::cat(flats, 0)
    };
    Some((merged, batch))
}

fn pack_residuals<B: BackendTrait>(
    residuals: Vec<&BurnTensor<B, 4>>,
    coarse: &BurnTensor<B, 4>,
) -> Option<(BurnTensor<B, 1>, ResidualMeta)>
where
    B::FloatTensorPrimitive: 'static,
{
    let mut offsets = [0u32; MAX_LEVELS];
    let mut widths = [0u32; MAX_LEVELS];
    let mut heights = [0u32; MAX_LEVELS];
    let mut offset = 0u32;
    let residual_count = residuals.len().min(MAX_LEVELS - 1);

    let mut flats = Vec::with_capacity(residual_count + 1);
    for (idx, level) in residuals.into_iter().take(residual_count).enumerate() {
        let [batch, channels, height, width] = level.shape().dims::<4>();
        offsets[idx] = offset;
        widths[idx] = width as u32;
        heights[idx] = height as u32;
        let elems = batch * channels * height * width;
        flats.push(level.clone().reshape([elems]));
        offset = offset.saturating_add(elems as u32);
    }

    let [batch, channels, coarse_h, coarse_w] = coarse.shape().dims::<4>();
    let coarse_elems = batch * channels * coarse_h * coarse_w;
    let coarse_offset = offset;
    flats.push(coarse.clone().reshape([coarse_elems]));
    let merged = if flats.len() == 1 {
        flats[0].clone()
    } else {
        BurnTensor::cat(flats, 0)
    };

    let meta = ResidualMeta {
        offsets,
        widths,
        heights,
        coarse_offset,
        coarse_width: coarse_w as u32,
        coarse_height: coarse_h as u32,
        residual_count,
    };
    Some((merged, meta))
}

fn write_meta_array(meta: &mut [f32], start: usize, data: &[u32; MAX_LEVELS]) {
    for (idx, value) in data.iter().enumerate() {
        meta[start + idx] = *value as f32;
    }
}

fn resolve_fusion_tensor<B: BackendTrait, BT: BoolElement, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<WgpuRuntime, BT>> =
        try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<WgpuRuntime, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn resolve_direct_tensor<B: BackendTrait, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<WgpuRuntime>>
where
    B::FloatTensorPrimitive: 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let cube: CubeTensor<WgpuRuntime> = try_cast_primitive::<B, _>(prim)?;
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn foveated_patch_wgsl_runtime<R: CubeRuntime>(
    gaussian: CubeTensor<R>,
    residual: CubeTensor<R>,
    params: CubeTensor<R>,
    meta: CubeTensor<R>,
    patch_h: usize,
    patch_w: usize,
    batch: usize,
    channels: usize,
) -> CubeTensor<R> {
    let gaussian = into_contiguous(gaussian);
    let residual = into_contiguous(residual);
    let params = into_contiguous(params);
    let meta = into_contiguous(meta);

    let client = gaussian.client.clone();
    let device = gaussian.device.clone();
    let shape = Shape::new([batch, channels, patch_h, patch_w]);
    let output = empty_device::<R, f32>(client.clone(), device, shape);

    let workgroups_x = div_ceil_u32(patch_w as u32, WORKGROUP_SIZE);
    let workgroups_y = div_ceil_u32(patch_h as u32, WORKGROUP_SIZE);
    let count = CubeCount::Static(workgroups_x, workgroups_y, batch as u32);

    let kernel = SourceKernel::new(FoveationBufferKernel, CubeDim::new(WORKGROUP_SIZE, WORKGROUP_SIZE, 1));
    let bindings = Bindings::new().with_buffers(vec![
        gaussian.handle.clone().binding(),
        residual.handle.clone().binding(),
        output.handle.clone().binding(),
        params.handle.clone().binding(),
        meta.handle.clone().binding(),
    ]);
    client.execute(Box::new(kernel), count, bindings);
    output
}

fn div_ceil_u32(value: u32, divisor: u32) -> u32 {
    if value == 0 {
        return 0;
    }
    (value + divisor - 1) / divisor
}

#[derive(Clone)]
struct FoveationBufferKernel;

impl KernelSource for FoveationBufferKernel {
    fn source(&self) -> SourceTemplate {
        SourceTemplate::new(FOVEATION_BUFFER_SHADER)
    }

    fn id(&self) -> KernelId {
        KernelId::new::<Self>()
    }
}

fn matches_type<A: 'static, B: 'static>() -> bool {
    TypeId::of::<A>() == TypeId::of::<B>()
}

fn try_cast_primitive<B: BackendTrait, T: 'static>(
    value: B::FloatTensorPrimitive,
) -> Option<T>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed.downcast::<T>().ok().map(|boxed| *boxed)
}

fn try_cast_backend<B: BackendTrait, T: 'static>(value: T) -> Option<B::FloatTensorPrimitive>
where
    B::FloatTensorPrimitive: 'static,
{
    let boxed: Box<dyn Any> = Box::new(value);
    boxed
        .downcast::<B::FloatTensorPrimitive>()
        .ok()
        .map(|boxed| *boxed)
}
