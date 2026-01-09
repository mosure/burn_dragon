use std::any::{Any, TypeId};

use burn::tensor::{Tensor as BurnTensor, TensorPrimitive};
use burn::tensor::backend::Backend as BackendTrait;
use burn::tensor::{DType, Shape, TensorData};
use burn_cubecl::{BoolElement, CubeBackend, CubeRuntime};
use burn_cubecl::fusion::FusionCubeRuntime;
use burn_cubecl::kernel::into_contiguous;
use burn_cubecl::ops::numeric::{empty_device, zeros_device};
use burn_cubecl::tensor::CubeTensor;
use burn_fusion::FusionTensor;
use burn_fusion::stream::StreamId;
use burn_wgpu::WgpuRuntime;
#[cfg(feature = "cuda")]
use cubecl::cuda::CudaRuntime;
use cubecl::{calculate_cube_count_elemwise, prelude::*};

use super::{SaccadeLaplacianImages, SaccadeMipLevel};

const SUBSAMPLE_AXIS: u32 = super::SACCADE_FOVEA_SUBSAMPLES as u32;
const SUBSAMPLES: u32 = SUBSAMPLE_AXIS * SUBSAMPLE_AXIS;
const EPS: f32 = super::SACCADE_EPS;
const LOD_WINDOW: f32 = super::SACCADE_FOVEA_LOD_WINDOW;
const SQRT2: f32 = super::SACCADE_FOVEA_SQRT2;
const SQRT_PI_OVER_2: f32 = super::SACCADE_FOVEA_SQRT_PI_OVER_2;
const ERF_A: f32 = super::SACCADE_FOVEA_ERF_A;
const PI: f32 = super::SACCADE_FOVEA_PI;
const LN_2: f32 = super::SACCADE_LN_2;
const MAX_LAPLACIAN_LEVELS: usize = 8;
const MAX_LAPLACIAN_RESIDUALS: usize = MAX_LAPLACIAN_LEVELS - 1;

pub(super) fn supports_backend<B: BackendTrait>() -> bool
where
    B::FloatTensorPrimitive: 'static,
{
    #[cfg(feature = "cuda")]
    {
        return matches_type::<
            B::FloatTensorPrimitive,
            FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>,
        >() || matches_type::<
            B::FloatTensorPrimitive,
            FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
        >() || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<CudaRuntime, u32>>,
            >()
            || matches_type::<
                B::FloatTensorPrimitive,
                FusionTensor<FusionCubeRuntime<CudaRuntime, u8>>,
            >()
            || matches_type::<B::FloatTensorPrimitive, CubeTensor<CudaRuntime>>();
    }
    #[cfg(not(feature = "cuda"))]
    {
        return matches_type::<
            B::FloatTensorPrimitive,
            FusionTensor<FusionCubeRuntime<WgpuRuntime, u32>>,
        >() || matches_type::<
            B::FloatTensorPrimitive,
            FusionTensor<FusionCubeRuntime<WgpuRuntime, u8>>,
        >() || matches_type::<B::FloatTensorPrimitive, CubeTensor<WgpuRuntime>>();
    }
}

pub(super) fn try_foveated_patch_cubecl<B: BackendTrait>(
    levels: &[SaccadeMipLevel<B>],
    base_grid: &BurnTensor<B, 4>,
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    laplacian_images: Option<&SaccadeLaplacianImages<B>>,
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

    let level_count = levels.len().min(MAX_LAPLACIAN_LEVELS);
    if let Some(laplacian) = laplacian_images {
        if levels.len() <= MAX_LAPLACIAN_LEVELS {
            if let Some(result) =
                try_foveated_patch_cubecl_laplacian_fused_runtime::<B, u32, WgpuRuntime>(
                    laplacian,
                    level_count,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    patch_h,
                    patch_w,
                )
            {
                return Some(result);
            }
            if let Some(result) =
                try_foveated_patch_cubecl_laplacian_fused_runtime::<B, u8, WgpuRuntime>(
                    laplacian,
                    level_count,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    patch_h,
                    patch_w,
                )
            {
                return Some(result);
            }
            #[cfg(feature = "cuda")]
            if let Some(result) =
                try_foveated_patch_cubecl_laplacian_fused_runtime::<B, u32, CudaRuntime>(
                    laplacian,
                    level_count,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    patch_h,
                    patch_w,
                )
            {
                return Some(result);
            }
            #[cfg(feature = "cuda")]
            if let Some(result) =
                try_foveated_patch_cubecl_laplacian_fused_runtime::<B, u8, CudaRuntime>(
                    laplacian,
                    level_count,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    patch_h,
                    patch_w,
                )
            {
                return Some(result);
            }
            #[cfg(feature = "cuda")]
            {
                if let Some(result) =
                    try_foveated_patch_cubecl_laplacian_direct_runtime::<B, CudaRuntime>(
                        laplacian,
                        level_count,
                        center_x,
                        center_y,
                        sigma_px,
                        radius_px,
                        lod_sigma,
                        patch_h,
                        patch_w,
                    )
                {
                    return Some(result);
                }
            }
            if let Some(result) =
                try_foveated_patch_cubecl_laplacian_direct_runtime::<B, WgpuRuntime>(
                    laplacian,
                    level_count,
                    center_x,
                    center_y,
                    sigma_px,
                    radius_px,
                    lod_sigma,
                    patch_h,
                    patch_w,
                )
            {
                return Some(result);
            }
        }
    }

    let level_images = if laplacian_images.is_some() && levels.len() > MAX_LAPLACIAN_LEVELS {
        let laplacian = laplacian_images.expect("checked");
        let mut recon = Vec::with_capacity(levels.len());
        let mut current = laplacian.coarse.clone();
        recon.push(current.clone());
        for residual in laplacian.residuals.iter().rev() {
            let [batch, _, res_h, res_w] = residual.shape().dims::<4>();
            let [_, _, cur_h, cur_w] = current.shape().dims::<4>();
            let device = current.device();
            let grid = super::build_image_grid::<B>(res_h, res_w, cur_h, cur_w, &device);
            let grid = if grid.shape().dims::<4>()[0] == batch {
                grid
            } else {
                grid.repeat_dim(0, batch)
            };
            let upsampled = super::grid_sample_2d_bilinear::<B>(current, grid);
            current = upsampled + residual.clone();
            recon.push(current.clone());
        }
        recon.reverse();
        recon
    } else {
        levels.iter().map(|level| level.image.clone()).collect()
    };

    if let Some(result) = try_foveated_patch_cubecl_fused_runtime::<B, u32, WgpuRuntime>(
        &level_images,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        patch_h,
        patch_w,
    ) {
        return Some(result);
    }
    if let Some(result) = try_foveated_patch_cubecl_fused_runtime::<B, u8, WgpuRuntime>(
        &level_images,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        patch_h,
        patch_w,
    ) {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    if let Some(result) = try_foveated_patch_cubecl_fused_runtime::<B, u32, CudaRuntime>(
        &level_images,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        patch_h,
        patch_w,
    ) {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    if let Some(result) = try_foveated_patch_cubecl_fused_runtime::<B, u8, CudaRuntime>(
        &level_images,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        patch_h,
        patch_w,
    ) {
        return Some(result);
    }
    #[cfg(feature = "cuda")]
    {
        if let Some(result) = try_foveated_patch_cubecl_direct_runtime::<B, CudaRuntime>(
            &level_images,
            center_x,
            center_y,
            sigma_px,
            radius_px,
            lod_sigma,
            patch_h,
            patch_w,
        ) {
            return Some(result);
        }
    }
    try_foveated_patch_cubecl_direct_runtime::<B, WgpuRuntime>(
        &level_images,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        patch_h,
        patch_w,
    )
}

fn try_foveated_patch_cubecl_fused_runtime<B: BackendTrait, BT: BoolElement, R: CubeRuntime>(
    level_images: &[BurnTensor<B, 4>],
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    patch_h: usize,
    patch_w: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R, BT>>>() {
        return None;
    }
    let first = level_images.first()?;
    let prim_first = first.clone().into_primitive().tensor();
    let fusion_first: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim_first)?;
    let fusion_client = fusion_first.client.clone();
    let mut cube_levels = Vec::with_capacity(level_images.len());
    let first_level =
        fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_first);
    if first_level.dtype != DType::F32 {
        return None;
    }
    cube_levels.push(first_level);
    for level in level_images.iter().skip(1) {
        let prim = level.clone().into_primitive().tensor();
        let fusion: FusionTensor<FusionCubeRuntime<R, BT>> =
            try_cast_primitive::<B, _>(prim)?;
        let client = fusion.client.clone();
        let cube = client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion);
        if cube.dtype != DType::F32 {
            return None;
        }
        cube_levels.push(cube);
    }

    let device = center_x.device();
    let [_, _, base_h, base_w] = level_images.first()?.shape().dims::<4>();
    let base_dims = BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![base_w as f32, base_h as f32], [2]),
        &device,
    );
    let center_x = resolve_fusion_tensor::<B, BT, R, 3>(center_x)?;
    let center_y = resolve_fusion_tensor::<B, BT, R, 3>(center_y)?;
    let sigma_px = resolve_fusion_tensor::<B, BT, R, 3>(sigma_px)?;
    let radius_px = resolve_fusion_tensor::<B, BT, R, 3>(radius_px)?;
    let lod_sigma = resolve_fusion_tensor::<B, BT, R, 3>(lod_sigma)?;
    let base_dims = resolve_fusion_tensor::<B, BT, R, 1>(&base_dims)?;

    let output = foveated_patch_cubecl_runtime::<R>(
        cube_levels,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        base_dims,
        patch_h,
        patch_w,
    );
    let shape = output.shape.clone();
    let dtype = output.dtype;
    let handle = output.into();
    let fusion_out = fusion_client.register_tensor(handle, shape, StreamId::current(), dtype);
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn try_foveated_patch_cubecl_direct_runtime<B: BackendTrait, R: CubeRuntime>(
    level_images: &[BurnTensor<B, 4>],
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    patch_h: usize,
    patch_w: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<R>>() {
        return None;
    }
    let mut cube_levels = Vec::with_capacity(level_images.len());
    for level in level_images {
        let prim = level.clone().into_primitive().tensor();
        let cube: CubeTensor<R> = try_cast_primitive::<B, _>(prim)?;
        if cube.dtype != DType::F32 {
            return None;
        }
        cube_levels.push(cube);
    }
    let device = center_x.device();
    let [_, _, base_h, base_w] = level_images.first()?.shape().dims::<4>();
    let base_dims = BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![base_w as f32, base_h as f32], [2]),
        &device,
    );
    let center_x = resolve_direct_tensor::<B, R, 3>(center_x)?;
    let center_y = resolve_direct_tensor::<B, R, 3>(center_y)?;
    let sigma_px = resolve_direct_tensor::<B, R, 3>(sigma_px)?;
    let radius_px = resolve_direct_tensor::<B, R, 3>(radius_px)?;
    let lod_sigma = resolve_direct_tensor::<B, R, 3>(lod_sigma)?;
    let base_dims = resolve_direct_tensor::<B, R, 1>(&base_dims)?;

    let output = foveated_patch_cubecl_runtime::<R>(
        cube_levels,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        base_dims,
        patch_h,
        patch_w,
    );
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn try_foveated_patch_cubecl_laplacian_fused_runtime<
    B: BackendTrait,
    BT: BoolElement,
    R: CubeRuntime,
>(
    laplacian: &SaccadeLaplacianImages<B>,
    level_count: usize,
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    patch_h: usize,
    patch_w: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, FusionTensor<FusionCubeRuntime<R, BT>>>() {
        return None;
    }
    let level_count = level_count.max(1).min(MAX_LAPLACIAN_LEVELS);
    let residual_count = level_count.saturating_sub(1).min(laplacian.residuals.len());
    let prim_coarse = laplacian.coarse.clone().into_primitive().tensor();
    let fusion_coarse: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim_coarse)?;
    let fusion_client = fusion_coarse.client.clone();
    let coarse =
        fusion_client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion_coarse);
    if coarse.dtype != DType::F32 {
        return None;
    }

    let device = center_x.device();
    let base_source = laplacian
        .residuals
        .first()
        .unwrap_or(&laplacian.coarse);
    let [_, _, base_h, base_w] = base_source.shape().dims::<4>();
    let base_dims = BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![base_w as f32, base_h as f32], [2]),
        &device,
    );

    let center_x = resolve_fusion_tensor::<B, BT, R, 3>(center_x)?;
    let center_y = resolve_fusion_tensor::<B, BT, R, 3>(center_y)?;
    let sigma_px = resolve_fusion_tensor::<B, BT, R, 3>(sigma_px)?;
    let radius_px = resolve_fusion_tensor::<B, BT, R, 3>(radius_px)?;
    let lod_sigma = resolve_fusion_tensor::<B, BT, R, 3>(lod_sigma)?;
    let base_dims = resolve_fusion_tensor::<B, BT, R, 1>(&base_dims)?;

    let mut cube_residuals = Vec::with_capacity(residual_count);
    for residual in laplacian.residuals.iter().take(residual_count) {
        let cube = resolve_fusion_tensor::<B, BT, R, 4>(residual)?;
        cube_residuals.push(cube);
    }

    let output = foveated_patch_cubecl_laplacian_runtime::<R>(
        cube_residuals,
        coarse,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        base_dims,
        patch_h,
        patch_w,
        level_count as u32,
    );
    let shape = output.shape.clone();
    let dtype = output.dtype;
    let handle = output.into();
    let fusion_out = fusion_client.register_tensor(handle, shape, StreamId::current(), dtype);
    let out_prim = try_cast_backend::<B, _>(fusion_out)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn try_foveated_patch_cubecl_laplacian_direct_runtime<B: BackendTrait, R: CubeRuntime>(
    laplacian: &SaccadeLaplacianImages<B>,
    level_count: usize,
    center_x: &BurnTensor<B, 3>,
    center_y: &BurnTensor<B, 3>,
    sigma_px: &BurnTensor<B, 3>,
    radius_px: &BurnTensor<B, 3>,
    lod_sigma: &BurnTensor<B, 3>,
    patch_h: usize,
    patch_w: usize,
) -> Option<BurnTensor<B, 4>>
where
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    if !matches_type::<B::FloatTensorPrimitive, CubeTensor<R>>() {
        return None;
    }
    let level_count = level_count.max(1).min(MAX_LAPLACIAN_LEVELS);
    let residual_count = level_count.saturating_sub(1).min(laplacian.residuals.len());
    let coarse = resolve_direct_tensor::<B, R, 4>(&laplacian.coarse)?;

    let device = center_x.device();
    let base_source = laplacian
        .residuals
        .first()
        .unwrap_or(&laplacian.coarse);
    let [_, _, base_h, base_w] = base_source.shape().dims::<4>();
    let base_dims = BurnTensor::<B, 1>::from_data(
        TensorData::new(vec![base_w as f32, base_h as f32], [2]),
        &device,
    );

    let center_x = resolve_direct_tensor::<B, R, 3>(center_x)?;
    let center_y = resolve_direct_tensor::<B, R, 3>(center_y)?;
    let sigma_px = resolve_direct_tensor::<B, R, 3>(sigma_px)?;
    let radius_px = resolve_direct_tensor::<B, R, 3>(radius_px)?;
    let lod_sigma = resolve_direct_tensor::<B, R, 3>(lod_sigma)?;
    let base_dims = resolve_direct_tensor::<B, R, 1>(&base_dims)?;

    let mut cube_residuals = Vec::with_capacity(residual_count);
    for residual in laplacian.residuals.iter().take(residual_count) {
        let cube = resolve_direct_tensor::<B, R, 4>(residual)?;
        cube_residuals.push(cube);
    }

    let output = foveated_patch_cubecl_laplacian_runtime::<R>(
        cube_residuals,
        coarse,
        center_x,
        center_y,
        sigma_px,
        radius_px,
        lod_sigma,
        base_dims,
        patch_h,
        patch_w,
        level_count as u32,
    );
    let out_prim = try_cast_backend::<B, _>(output)?;
    Some(BurnTensor::<B, 4>::from_primitive(TensorPrimitive::Float(out_prim)))
}

fn resolve_fusion_tensor<B: BackendTrait, BT: BoolElement, R: CubeRuntime, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<R>>
where
    B::FloatTensorPrimitive: 'static,
    BT: BoolElement + 'static,
    R: CubeRuntime + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let fusion: FusionTensor<FusionCubeRuntime<R, BT>> =
        try_cast_primitive::<B, _>(prim)?;
    let client = fusion.client.clone();
    let cube = client.resolve_tensor_float::<CubeBackend<R, f32, i32, BT>>(fusion);
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn resolve_direct_tensor<B: BackendTrait, R: CubeRuntime, const D: usize>(
    tensor: &BurnTensor<B, D>,
) -> Option<CubeTensor<R>>
where
    B::FloatTensorPrimitive: 'static,
    R: CubeRuntime + 'static,
{
    let prim = tensor.clone().into_primitive().tensor();
    let cube: CubeTensor<R> = try_cast_primitive::<B, _>(prim)?;
    if cube.dtype != DType::F32 {
        return None;
    }
    Some(cube)
}

fn foveated_patch_cubecl_runtime<R: CubeRuntime>(
    level_images: Vec<CubeTensor<R>>,
    center_x: CubeTensor<R>,
    center_y: CubeTensor<R>,
    sigma_px: CubeTensor<R>,
    radius_px: CubeTensor<R>,
    lod_sigma: CubeTensor<R>,
    base_dims: CubeTensor<R>,
    patch_h: usize,
    patch_w: usize,
) -> CubeTensor<R> {
    let levels: Vec<_> = level_images.into_iter().map(into_contiguous).collect();
    let center_x = into_contiguous(center_x);
    let center_y = into_contiguous(center_y);
    let sigma_px = into_contiguous(sigma_px);
    let radius_px = into_contiguous(radius_px);
    let lod_sigma = into_contiguous(lod_sigma);
    let base_dims = into_contiguous(base_dims);

    let first = levels.first().expect("levels not empty");
    let [batch, channels, _, _] = first.shape.dims::<4>();
    let subsamples = SUBSAMPLES as usize;
    let sbatch = batch * subsamples;
    let color_shape = Shape::new([sbatch, channels, patch_h, patch_w]);
    let weight_shape = Shape::new([sbatch, patch_h, patch_w]);

    let client = first.client.clone();
    let device = first.device.clone();

    let color_accum = zeros_device::<R, f32>(client.clone(), device.clone(), color_shape);
    let weight_sum = zeros_device::<R, f32>(client.clone(), device.clone(), weight_shape);

    let cube_dim = CubeDim::default();
    let elem_count = color_accum.shape.num_elements();
    let cube_count = calculate_cube_count_elemwise(elem_count, cube_dim);
    let level_count = levels.len() as u32;
    for (level_idx, level) in levels.into_iter().enumerate() {
        foveated_accumulate_kernel::launch::<R>(
            &client,
            cube_count.clone(),
            cube_dim,
            level.as_tensor_arg::<f32>(1),
            color_accum.as_tensor_arg::<f32>(1),
            weight_sum.as_tensor_arg::<f32>(1),
            center_x.as_tensor_arg::<f32>(1),
            center_y.as_tensor_arg::<f32>(1),
            sigma_px.as_tensor_arg::<f32>(1),
            radius_px.as_tensor_arg::<f32>(1),
            lod_sigma.as_tensor_arg::<f32>(1),
            base_dims.as_tensor_arg::<f32>(1),
            ScalarArg::new(level_idx as u32),
            ScalarArg::new(level_count),
        );
    }

    let output_shape = Shape::new([batch, channels, patch_h, patch_w]);
    let output = empty_device::<R, f32>(client.clone(), device, output_shape);
    let out_elems = output.shape.num_elements();
    let out_cube_count = calculate_cube_count_elemwise(out_elems, cube_dim);
    foveated_finalize_kernel::launch::<R>(
        &client,
        out_cube_count,
        cube_dim,
        color_accum.as_tensor_arg::<f32>(1),
        weight_sum.as_tensor_arg::<f32>(1),
        output.as_tensor_arg::<f32>(1),
    );
    output
}

fn foveated_patch_cubecl_laplacian_runtime<R: CubeRuntime>(
    residuals: Vec<CubeTensor<R>>,
    coarse: CubeTensor<R>,
    center_x: CubeTensor<R>,
    center_y: CubeTensor<R>,
    sigma_px: CubeTensor<R>,
    radius_px: CubeTensor<R>,
    lod_sigma: CubeTensor<R>,
    base_dims: CubeTensor<R>,
    patch_h: usize,
    patch_w: usize,
    level_count: u32,
) -> CubeTensor<R> {
    let residuals: Vec<_> = residuals.into_iter().map(into_contiguous).collect();
    let coarse = into_contiguous(coarse);
    let center_x = into_contiguous(center_x);
    let center_y = into_contiguous(center_y);
    let sigma_px = into_contiguous(sigma_px);
    let radius_px = into_contiguous(radius_px);
    let lod_sigma = into_contiguous(lod_sigma);
    let base_dims = into_contiguous(base_dims);

    let [batch, channels, _, _] = coarse.shape.dims::<4>();
    let subsamples = SUBSAMPLES as usize;
    let sbatch = batch * subsamples;
    let color_shape = Shape::new([sbatch, channels, patch_h, patch_w]);
    let weight_shape = Shape::new([sbatch, patch_h, patch_w]);

    let client = coarse.client.clone();
    let device = coarse.device.clone();

    let color_accum = zeros_device::<R, f32>(client.clone(), device.clone(), color_shape);
    let weight_sum = zeros_device::<R, f32>(client.clone(), device.clone(), weight_shape.clone());
    let prefix_weight =
        zeros_device::<R, f32>(client.clone(), device.clone(), weight_shape);

    let cube_dim = CubeDim::default();
    let elem_count = color_accum.shape.num_elements();
    let cube_count = calculate_cube_count_elemwise(elem_count, cube_dim);
    let level_count = if level_count == 0 {
        1u32
    } else if level_count > MAX_LAPLACIAN_LEVELS as u32 {
        MAX_LAPLACIAN_LEVELS as u32
    } else {
        level_count
    };

    let zero_shape = Shape::new([batch.max(1), channels.max(1), 1, 1]);
    let zero = zeros_device::<R, f32>(client.clone(), device.clone(), zero_shape);

    for level_idx in 0..level_count {
        foveated_laplacian_weight_kernel::launch::<R>(
            &client,
            cube_count.clone(),
            cube_dim,
            weight_sum.as_tensor_arg::<f32>(1),
            prefix_weight.as_tensor_arg::<f32>(1),
            sigma_px.as_tensor_arg::<f32>(1),
            radius_px.as_tensor_arg::<f32>(1),
            lod_sigma.as_tensor_arg::<f32>(1),
            ScalarArg::new(level_idx),
            ScalarArg::new(level_count),
        );

        let residual = residuals
            .get(level_idx as usize)
            .unwrap_or(&zero);
        foveated_laplacian_residual_kernel::launch::<R>(
            &client,
            cube_count.clone(),
            cube_dim,
            residual.as_tensor_arg::<f32>(1),
            color_accum.as_tensor_arg::<f32>(1),
            prefix_weight.as_tensor_arg::<f32>(1),
            center_x.as_tensor_arg::<f32>(1),
            center_y.as_tensor_arg::<f32>(1),
            sigma_px.as_tensor_arg::<f32>(1),
            radius_px.as_tensor_arg::<f32>(1),
            base_dims.as_tensor_arg::<f32>(1),
        );
    }

    foveated_laplacian_coarse_kernel::launch::<R>(
        &client,
        cube_count.clone(),
        cube_dim,
        coarse.as_tensor_arg::<f32>(1),
        color_accum.as_tensor_arg::<f32>(1),
        weight_sum.as_tensor_arg::<f32>(1),
        center_x.as_tensor_arg::<f32>(1),
        center_y.as_tensor_arg::<f32>(1),
        sigma_px.as_tensor_arg::<f32>(1),
        radius_px.as_tensor_arg::<f32>(1),
        base_dims.as_tensor_arg::<f32>(1),
    );

    let output_shape = Shape::new([batch, channels, patch_h, patch_w]);
    let output = empty_device::<R, f32>(client.clone(), device, output_shape);
    let out_elems = output.shape.num_elements();
    let out_cube_count = calculate_cube_count_elemwise(out_elems, cube_dim);
    foveated_finalize_kernel::launch::<R>(
        &client,
        out_cube_count,
        cube_dim,
        color_accum.as_tensor_arg::<f32>(1),
        weight_sum.as_tensor_arg::<f32>(1),
        output.as_tensor_arg::<f32>(1),
    );
    output
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

#[cube(launch)]
fn foveated_accumulate_kernel(
    input: &Tensor<f32>,
    output: &mut Tensor<f32>,
    weight_sum: &mut Tensor<f32>,
    center_x: &Tensor<f32>,
    center_y: &Tensor<f32>,
    sigma_px: &Tensor<f32>,
    radius_px: &Tensor<f32>,
    lod_sigma: &Tensor<f32>,
    base_dims: &Tensor<f32>,
    level_idx: u32,
    level_count: u32,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let channels = output.shape(1);
    if out_w == 0 || out_h == 0 || channels == 0 {
        terminate!();
    }

    let x = ABSOLUTE_POS % out_w;
    let pos = ABSOLUTE_POS / out_w;
    let y = pos % out_h;
    let pos = pos / out_h;
    let c = pos % channels;
    let sb = pos / channels;

    let subsamples = SUBSAMPLES;
    let batch = output.shape(0) / subsamples;
    if batch == 0 {
        terminate!();
    }
    let subsample = sb / batch;
    let batch_idx = sb - subsample * batch;

    let half = f32::cast_from(out_h) * 0.5f32;
    let half_safe = max_f32(half, 1.0f32);
    let pixel_du = 1.0f32 / half_safe;

    let sx = subsample % SUBSAMPLE_AXIS;
    let sy = subsample / SUBSAMPLE_AXIS;
    let jitter_x = (f32::cast_from(sx) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;
    let jitter_y = (f32::cast_from(sy) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;

    let x_base = (f32::cast_from(x) + 0.5f32 - half) / half_safe;
    let y_base = (f32::cast_from(y) + 0.5f32 - half) / half_safe;
    let ux = x_base + jitter_x / half_safe;
    let uy = y_base + jitter_y / half_safe;

    let cx_idx = batch_idx * center_x.stride(0);
    let cy_idx = batch_idx * center_y.stride(0);
    let sigma_idx = batch_idx * sigma_px.stride(0);
    let radius_idx = batch_idx * radius_px.stride(0);
    let lod_idx = batch_idx * lod_sigma.stride(0);
    let cx = center_x[cx_idx];
    let cy = center_y[cy_idx];
    let sigma = max_f32(sigma_px[sigma_idx], EPS);
    let radius = max_f32(radius_px[radius_idx], EPS);
    let lod_sigma = max_f32(lod_sigma[lod_idx], EPS);

    let k = radius / sigma;
    let u_max = min_f32(erf_approx(k / SQRT2), 0.999f32);
    let u_scaled_x = clamp_f32(ux, -1.0f32, 1.0f32) * u_max;
    let u_scaled_y = clamp_f32(uy, -1.0f32, 1.0f32) * u_max;

    let erf_inv_x = erfinv_approx(u_scaled_x);
    let erf_inv_y = erfinv_approx(u_scaled_y);
    let dx = sigma * SQRT2 * erf_inv_x;
    let dy = sigma * SQRT2 * erf_inv_y;
    let dx_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_x * erf_inv_x);
    let dy_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_y * erf_inv_y);

    let local_scale = max_f32(Abs::abs(dx_deriv), Abs::abs(dy_deriv)) * pixel_du;
    let img_x = cx + dx;
    let img_y = cy + dy;

    let width = input.shape(3);
    let height = input.shape(2);
    let base_width = base_dims[0u32];
    let base_height = base_dims[1u32];
    if width == 0 || height == 0 || base_width <= 0.0f32 || base_height <= 0.0f32 {
        terminate!();
    }
    let fx = img_x / base_width;
    let fy = img_y / base_height;

    let sigma_sq = sigma * sigma;
    let dist = Sqrt::sqrt(((dx * dx) / sigma_sq) + ((dy * dy) / sigma_sq));
    let lod_dist = if dist <= 1.0f32 {
        0.0f32.into()
    } else {
        Log::log(max_f32(dist, 1.0f32)) / LN_2
    };
    let lod_scale = if local_scale <= 1.0f32 {
        0.0f32.into()
    } else {
        Log::log(max_f32(local_scale, 1.0f32)) / LN_2
    };
    let max_level = if level_count > 0 {
        f32::cast_from(level_count - 1)
    } else {
        0.0f32.into()
    };
    let lod = clamp_f32(max_f32(lod_dist, lod_scale), 0.0f32, max_level);

    let level_f = f32::cast_from(level_idx);
    let diff = (level_f - lod) / lod_sigma;
    let weight = if Abs::abs(lod - level_f) > LOD_WINDOW {
        0.0f32.into()
    } else {
        Exp::exp(-0.5f32 * diff * diff)
    };

    let sample = sample_bilinear(input, batch_idx, c, fx, fy);
    let out_idx = sb * output.stride(0)
        + c * output.stride(1)
        + y * output.stride(2)
        + x * output.stride(3);
    output[out_idx] = output[out_idx] + sample * weight;

    if c == 0 {
        let w_idx = sb * weight_sum.stride(0)
            + y * weight_sum.stride(1)
            + x * weight_sum.stride(2);
        weight_sum[w_idx] = weight_sum[w_idx] + weight;
    }
}

#[cube(launch)]
fn foveated_laplacian_weight_kernel(
    weight_sum: &mut Tensor<f32>,
    prefix_weight: &mut Tensor<f32>,
    sigma_px: &Tensor<f32>,
    radius_px: &Tensor<f32>,
    lod_sigma: &Tensor<f32>,
    level_idx: u32,
    level_count: u32,
) {
    if ABSOLUTE_POS >= weight_sum.len() {
        terminate!();
    }
    let out_w = weight_sum.shape(2);
    let out_h = weight_sum.shape(1);
    if out_w == 0 || out_h == 0 {
        terminate!();
    }

    let x = ABSOLUTE_POS % out_w;
    let pos = ABSOLUTE_POS / out_w;
    let y = pos % out_h;
    let sb = pos / out_h;

    let subsamples = SUBSAMPLES;
    let batch = weight_sum.shape(0) / subsamples;
    if batch == 0 {
        terminate!();
    }
    let subsample = sb / batch;
    let batch_idx = sb - subsample * batch;

    let half = f32::cast_from(out_h) * 0.5f32;
    let half_safe = max_f32(half, 1.0f32);
    let pixel_du = 1.0f32 / half_safe;

    let sx = subsample % SUBSAMPLE_AXIS;
    let sy = subsample / SUBSAMPLE_AXIS;
    let jitter_x = (f32::cast_from(sx) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;
    let jitter_y = (f32::cast_from(sy) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;

    let x_base = (f32::cast_from(x) + 0.5f32 - half) / half_safe;
    let y_base = (f32::cast_from(y) + 0.5f32 - half) / half_safe;
    let ux = x_base + jitter_x / half_safe;
    let uy = y_base + jitter_y / half_safe;

    let sigma_idx = batch_idx * sigma_px.stride(0);
    let radius_idx = batch_idx * radius_px.stride(0);
    let lod_idx = batch_idx * lod_sigma.stride(0);
    let sigma = max_f32(sigma_px[sigma_idx], EPS);
    let radius = max_f32(radius_px[radius_idx], EPS);
    let lod_sigma = max_f32(lod_sigma[lod_idx], EPS);

    let k = radius / sigma;
    let u_max = min_f32(erf_approx(k / SQRT2), 0.999f32);
    let u_scaled_x = clamp_f32(ux, -1.0f32, 1.0f32) * u_max;
    let u_scaled_y = clamp_f32(uy, -1.0f32, 1.0f32) * u_max;

    let erf_inv_x = erfinv_approx(u_scaled_x);
    let erf_inv_y = erfinv_approx(u_scaled_y);
    let dx = sigma * SQRT2 * erf_inv_x;
    let dy = sigma * SQRT2 * erf_inv_y;
    let dx_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_x * erf_inv_x);
    let dy_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_y * erf_inv_y);

    let local_scale = max_f32(Abs::abs(dx_deriv), Abs::abs(dy_deriv)) * pixel_du;

    let sigma_sq = sigma * sigma;
    let dist = Sqrt::sqrt(((dx * dx) / sigma_sq) + ((dy * dy) / sigma_sq));
    let lod_dist = if dist <= 1.0f32 {
        0.0f32.into()
    } else {
        Log::log(max_f32(dist, 1.0f32)) / LN_2
    };
    let lod_scale = if local_scale <= 1.0f32 {
        0.0f32.into()
    } else {
        Log::log(max_f32(local_scale, 1.0f32)) / LN_2
    };
    let max_level = if level_count > 0 {
        f32::cast_from(level_count - 1)
    } else {
        0.0f32.into()
    };
    let lod = clamp_f32(max_f32(lod_dist, lod_scale), 0.0f32, max_level);

    let level_f = f32::cast_from(level_idx);
    let diff = (level_f - lod) / lod_sigma;
    let weight = if Abs::abs(lod - level_f) > LOD_WINDOW {
        0.0f32.into()
    } else {
        Exp::exp(-0.5f32 * diff * diff)
    };

    let w_idx = sb * weight_sum.stride(0)
        + y * weight_sum.stride(1)
        + x * weight_sum.stride(2);
    let prefix = prefix_weight[w_idx] + weight;
    prefix_weight[w_idx] = prefix;
    weight_sum[w_idx] = weight_sum[w_idx] + weight;
}

#[cube(launch)]
fn foveated_laplacian_residual_kernel(
    residual: &Tensor<f32>,
    output: &mut Tensor<f32>,
    prefix_weight: &Tensor<f32>,
    center_x: &Tensor<f32>,
    center_y: &Tensor<f32>,
    sigma_px: &Tensor<f32>,
    radius_px: &Tensor<f32>,
    base_dims: &Tensor<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let channels = output.shape(1);
    if out_w == 0 || out_h == 0 || channels == 0 {
        terminate!();
    }

    let x = ABSOLUTE_POS % out_w;
    let pos = ABSOLUTE_POS / out_w;
    let y = pos % out_h;
    let pos = pos / out_h;
    let c = pos % channels;
    let sb = pos / channels;

    let subsamples = SUBSAMPLES;
    let batch = output.shape(0) / subsamples;
    if batch == 0 {
        terminate!();
    }
    let subsample = sb / batch;
    let batch_idx = sb - subsample * batch;

    let half = f32::cast_from(out_h) * 0.5f32;
    let half_safe = max_f32(half, 1.0f32);
    let pixel_du = 1.0f32 / half_safe;

    let sx = subsample % SUBSAMPLE_AXIS;
    let sy = subsample / SUBSAMPLE_AXIS;
    let jitter_x = (f32::cast_from(sx) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;
    let jitter_y = (f32::cast_from(sy) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;

    let x_base = (f32::cast_from(x) + 0.5f32 - half) / half_safe;
    let y_base = (f32::cast_from(y) + 0.5f32 - half) / half_safe;
    let ux = x_base + jitter_x / half_safe;
    let uy = y_base + jitter_y / half_safe;

    let cx_idx = batch_idx * center_x.stride(0);
    let cy_idx = batch_idx * center_y.stride(0);
    let sigma_idx = batch_idx * sigma_px.stride(0);
    let radius_idx = batch_idx * radius_px.stride(0);
    let cx = center_x[cx_idx];
    let cy = center_y[cy_idx];
    let sigma = max_f32(sigma_px[sigma_idx], EPS);
    let radius = max_f32(radius_px[radius_idx], EPS);

    let k = radius / sigma;
    let u_max = min_f32(erf_approx(k / SQRT2), 0.999f32);
    let u_scaled_x = clamp_f32(ux, -1.0f32, 1.0f32) * u_max;
    let u_scaled_y = clamp_f32(uy, -1.0f32, 1.0f32) * u_max;

    let erf_inv_x = erfinv_approx(u_scaled_x);
    let erf_inv_y = erfinv_approx(u_scaled_y);
    let dx = sigma * SQRT2 * erf_inv_x;
    let dy = sigma * SQRT2 * erf_inv_y;
    let dx_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_x * erf_inv_x);
    let dy_deriv =
        sigma * SQRT2 * u_max * SQRT_PI_OVER_2 * Exp::exp(erf_inv_y * erf_inv_y);

    let _local_scale = max_f32(Abs::abs(dx_deriv), Abs::abs(dy_deriv)) * pixel_du;
    let img_x = cx + dx;
    let img_y = cy + dy;

    let base_width = base_dims[0u32];
    let base_height = base_dims[1u32];
    if base_width <= 0.0f32 || base_height <= 0.0f32 {
        terminate!();
    }
    let fx = img_x / base_width;
    let fy = img_y / base_height;

    let sample = sample_bilinear(residual, batch_idx, c, fx, fy);
    let w_idx = sb * prefix_weight.stride(0)
        + y * prefix_weight.stride(1)
        + x * prefix_weight.stride(2);
    let weight = prefix_weight[w_idx];
    let out_idx = sb * output.stride(0)
        + c * output.stride(1)
        + y * output.stride(2)
        + x * output.stride(3);
    output[out_idx] = output[out_idx] + sample * weight;
}

#[cube(launch)]
fn foveated_laplacian_coarse_kernel(
    coarse: &Tensor<f32>,
    output: &mut Tensor<f32>,
    weight_sum: &Tensor<f32>,
    center_x: &Tensor<f32>,
    center_y: &Tensor<f32>,
    sigma_px: &Tensor<f32>,
    radius_px: &Tensor<f32>,
    base_dims: &Tensor<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let channels = output.shape(1);
    if out_w == 0 || out_h == 0 || channels == 0 {
        terminate!();
    }

    let x = ABSOLUTE_POS % out_w;
    let pos = ABSOLUTE_POS / out_w;
    let y = pos % out_h;
    let pos = pos / out_h;
    let c = pos % channels;
    let sb = pos / channels;

    let subsamples = SUBSAMPLES;
    let batch = output.shape(0) / subsamples;
    if batch == 0 {
        terminate!();
    }
    let subsample = sb / batch;
    let batch_idx = sb - subsample * batch;

    let half = f32::cast_from(out_h) * 0.5f32;
    let half_safe = max_f32(half, 1.0f32);

    let sx = subsample % SUBSAMPLE_AXIS;
    let sy = subsample / SUBSAMPLE_AXIS;
    let jitter_x = (f32::cast_from(sx) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;
    let jitter_y = (f32::cast_from(sy) + 0.5f32) / f32::cast_from(SUBSAMPLE_AXIS) - 0.5f32;

    let x_base = (f32::cast_from(x) + 0.5f32 - half) / half_safe;
    let y_base = (f32::cast_from(y) + 0.5f32 - half) / half_safe;
    let ux = x_base + jitter_x / half_safe;
    let uy = y_base + jitter_y / half_safe;

    let cx_idx = batch_idx * center_x.stride(0);
    let cy_idx = batch_idx * center_y.stride(0);
    let sigma_idx = batch_idx * sigma_px.stride(0);
    let radius_idx = batch_idx * radius_px.stride(0);
    let cx = center_x[cx_idx];
    let cy = center_y[cy_idx];
    let sigma = max_f32(sigma_px[sigma_idx], EPS);
    let radius = max_f32(radius_px[radius_idx], EPS);

    let k = radius / sigma;
    let u_max = min_f32(erf_approx(k / SQRT2), 0.999f32);
    let u_scaled_x = clamp_f32(ux, -1.0f32, 1.0f32) * u_max;
    let u_scaled_y = clamp_f32(uy, -1.0f32, 1.0f32) * u_max;

    let erf_inv_x = erfinv_approx(u_scaled_x);
    let erf_inv_y = erfinv_approx(u_scaled_y);
    let dx = sigma * SQRT2 * erf_inv_x;
    let dy = sigma * SQRT2 * erf_inv_y;
    let img_x = cx + dx;
    let img_y = cy + dy;

    let base_width = base_dims[0u32];
    let base_height = base_dims[1u32];
    if base_width <= 0.0f32 || base_height <= 0.0f32 {
        terminate!();
    }
    let fx = img_x / base_width;
    let fy = img_y / base_height;

    let sample = sample_bilinear(coarse, batch_idx, c, fx, fy);
    let w_idx = sb * weight_sum.stride(0)
        + y * weight_sum.stride(1)
        + x * weight_sum.stride(2);
    let weight = weight_sum[w_idx];
    let out_idx = sb * output.stride(0)
        + c * output.stride(1)
        + y * output.stride(2)
        + x * output.stride(3);
    output[out_idx] = output[out_idx] + sample * weight;
}

#[cube(launch)]
fn foveated_finalize_kernel(
    color_accum: &Tensor<f32>,
    weight_sum: &Tensor<f32>,
    output: &mut Tensor<f32>,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }
    let out_w = output.shape(3);
    let out_h = output.shape(2);
    let channels = output.shape(1);
    if out_w == 0 || out_h == 0 || channels == 0 {
        terminate!();
    }

    let x = ABSOLUTE_POS % out_w;
    let pos = ABSOLUTE_POS / out_w;
    let y = pos % out_h;
    let pos = pos / out_h;
    let c = pos % channels;
    let b = pos / channels;

    let batch = output.shape(0);
    let mut sum = 0.0f32;
    let mut subsample = 0u32;
    while subsample < SUBSAMPLES {
        let sb = subsample * batch + b;
        let color_idx = sb * color_accum.stride(0)
            + c * color_accum.stride(1)
            + y * color_accum.stride(2)
            + x * color_accum.stride(3);
        let w_idx = sb * weight_sum.stride(0)
            + y * weight_sum.stride(1)
            + x * weight_sum.stride(2);
        let weight = if weight_sum[w_idx] < EPS {
            EPS.into()
        } else {
            weight_sum[w_idx]
        };
        sum += color_accum[color_idx] / weight;
        subsample += 1u32;
    }
    let out_idx = b * output.stride(0)
        + c * output.stride(1)
        + y * output.stride(2)
        + x * output.stride(3);
    output[out_idx] = sum / f32::cast_from(SUBSAMPLES);
}

#[cube]
fn erf_approx(x: f32) -> f32 {
    let sign = if x >= 0.0f32 {
        f32::new(1.0f32)
    } else {
        f32::new(-1.0f32)
    };
    let ax = Abs::abs(x);
    let t = 1.0f32 / (1.0f32 + 0.3275911f32 * ax);
    let a1 = 0.254829592f32;
    let a2 = -0.284496736f32;
    let a3 = 1.421413741f32;
    let a4 = -1.453152027f32;
    let a5 = 1.061405429f32;
    let y = 1.0f32
        - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t
            * Exp::exp(-ax * ax);
    sign * y
}

#[cube]
fn erfinv_approx(x: f32) -> f32 {
    let sign = if x >= 0.0f32 {
        f32::new(1.0f32)
    } else {
        f32::new(-1.0f32)
    };
    let xx = clamp_f32(x, -0.999f32, 0.999f32);
    let ln = Log::log(1.0f32 - xx * xx);
    let term = 2.0f32 / (PI * ERF_A) + ln * 0.5f32;
    let inside = max_f32(term * term - ln / ERF_A, 0.0f32);
    let result = max_f32(Sqrt::sqrt(inside) - term, 0.0f32);
    sign * Sqrt::sqrt(result)
}

#[cube]
fn sample_bilinear(
    input: &Tensor<f32>,
    batch_idx: u32,
    channel: u32,
    fx: f32,
    fy: f32,
) -> f32 {
    let width = input.shape(3);
    let height = input.shape(2);
    let mut out = 0.0f32;
    if width > 0 && height > 0 {
        let fx = clamp_f32(fx, 0.0f32, 1.0f32);
        let fy = clamp_f32(fy, 0.0f32, 1.0f32);
        let x = fx * f32::cast_from(width) - 0.5f32;
        let y = fy * f32::cast_from(height) - 0.5f32;
        let x0 = Floor::floor(x);
        let y0 = Floor::floor(y);
        let x1 = x0 + 1.0f32;
        let y1 = y0 + 1.0f32;
        let tx = x - x0;
        let ty = y - y0;

        let max_x = f32::cast_from(width - 1);
        let max_y = f32::cast_from(height - 1);
        let x0i = clamp_f32(x0, 0.0f32, max_x) as u32;
        let y0i = clamp_f32(y0, 0.0f32, max_y) as u32;
        let x1i = clamp_f32(x1, 0.0f32, max_x) as u32;
        let y1i = clamp_f32(y1, 0.0f32, max_y) as u32;

        let base = batch_idx * input.stride(0) + channel * input.stride(1);
        let stride_y = input.stride(2);
        let stride_x = input.stride(3);
        let c00 = input[base + y0i * stride_y + x0i * stride_x];
        let c10 = input[base + y0i * stride_y + x1i * stride_x];
        let c01 = input[base + y1i * stride_y + x0i * stride_x];
        let c11 = input[base + y1i * stride_y + x1i * stride_x];

        let a = lerp(c00, c10, tx);
        let b = lerp(c01, c11, tx);
        out = lerp(a, b, ty);
    }
    out
}

#[cube]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[cube]
fn clamp_f32(value: f32, lo: f32, hi: f32) -> f32 {
    if value < lo {
        lo
    } else if value > hi {
        hi
    } else {
        value
    }
}

#[cube]
fn max_f32(a: f32, b: f32) -> f32 {
    if a > b { a } else { b }
}

#[cube]
fn min_f32(a: f32, b: f32) -> f32 {
    if a < b { a } else { b }
}
