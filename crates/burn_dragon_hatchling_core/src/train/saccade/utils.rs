use crate::train::prelude::*;

pub(crate) fn gaussian_downsample_kernel<B: BackendTrait>(channels: usize, device: &B::Device) -> Tensor<B, 4> {
    let weights = [1.0_f32, 4.0, 6.0, 4.0, 1.0];
    let mut kernel = vec![0.0_f32; channels * 5 * 5];
    for c in 0..channels {
        let base = c * 25;
        for ky in 0..5 {
            for kx in 0..5 {
                kernel[base + ky * 5 + kx] = (weights[ky] * weights[kx]) / 256.0;
            }
        }
    }
    Tensor::<B, 4>::from_data(TensorData::new(kernel, [channels, 1, 5, 5]), device)
}

pub(crate) fn replicate_pad2d<B: BackendTrait>(images: Tensor<B, 4>, pad: usize) -> Tensor<B, 4> {
    if pad == 0 {
        return images;
    }
    let [_, _, height, width] = images.shape().dims::<4>();
    if height == 0 || width == 0 {
        return images;
    }
    let top = images
        .clone()
        .slice_dim(2, 0..1)
        .repeat_dim(2, pad);
    let bottom = images
        .clone()
        .slice_dim(2, height - 1..height)
        .repeat_dim(2, pad);
    let padded_v = Tensor::cat(vec![top, images, bottom], 2);
    let left = padded_v
        .clone()
        .slice_dim(3, 0..1)
        .repeat_dim(3, pad);
    let right = padded_v
        .clone()
        .slice_dim(3, width - 1..width)
        .repeat_dim(3, pad);
    Tensor::cat(vec![left, padded_v, right], 3)
}

pub(crate) fn downsample_image<B: BackendTrait>(images: Tensor<B, 4>) -> Option<Tensor<B, 4>> {
    let [_batch, channels, height, width] = images.shape().dims::<4>();
    if channels == 0 || height < 2 || width < 2 {
        return None;
    }
    let even_h = height - (height % 2);
    let even_w = width - (width % 2);
    if even_h == 0 || even_w == 0 {
        return None;
    }
    let device = images.device();
    let images = images
        .slice_dim(2, 0..even_h)
        .slice_dim(3, 0..even_w);
    let padded = replicate_pad2d(images, 2);
    let kernel = gaussian_downsample_kernel::<B>(channels, &device);
    let options = ConvOptions::new([2, 2], [0, 0], [1, 1], channels.max(1));
    Some(conv2d(padded, kernel, None, options))
}

pub(crate) fn should_fix_grid<B: BackendTrait>() -> bool
where
    B::Device: 'static,
{
#[cfg(any(feature = "train", feature = "cli"))]
    {
        if TypeId::of::<B::Device>() == TypeId::of::<NdArrayDevice>() {
            return false;
        }
    }
    true
}

pub(crate) fn fix_grid_for_burn<B: BackendTrait>(
    grid: Tensor<B, 4>,
    height_in: usize,
    width_in: usize,
) -> Tensor<B, 4> {
    if !should_fix_grid::<B>() {
        return grid;
    }
    if width_in <= 1 || height_in <= 1 {
        return grid;
    }
    let x_half = (width_in - 1) as f32 * 0.5;
    let y_half = (height_in - 1) as f32 * 0.5;
    if (x_half - y_half).abs() <= f32::EPSILON {
        return grid;
    }
    // burn-tensor's default grid_sample scales y by x_half; compensate for all backends.
    let scale = y_half / x_half;
    let grid_x = grid.clone().slice_dim(3, 0..1);
    let grid_y = grid.slice_dim(3, 1..2).mul_scalar(scale);
    Tensor::cat(vec![grid_x, grid_y], 3)
}

pub(crate) fn grid_from_fx_fy<B: BackendTrait>(
    fx: &Tensor<B, 3>,
    fy: &Tensor<B, 3>,
    level_w: usize,
    level_h: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let grid_shape = fx.shape().dims::<3>();
    let grid_x = if level_w > 1 {
        fx.clone()
            .mul_scalar(level_w as f32)
            .sub_scalar(0.5)
            .mul_scalar(2.0 / (level_w - 1) as f32)
            .add_scalar(-1.0)
    } else {
        Tensor::<B, 3>::zeros(grid_shape, device)
    };
    let grid_y = if level_h > 1 {
        fy.clone()
            .mul_scalar(level_h as f32)
            .sub_scalar(0.5)
            .mul_scalar(2.0 / (level_h - 1) as f32)
            .add_scalar(-1.0)
    } else {
        Tensor::<B, 3>::zeros(grid_shape, device)
    };
    Tensor::cat(
        vec![grid_x.unsqueeze_dim::<4>(3), grid_y.unsqueeze_dim::<4>(3)],
        3,
    )
}

pub(crate) fn grid_sample_2d_bilinear<B: BackendTrait>(
    tensor: Tensor<B, 4>,
    grid: Tensor<B, 4>,
) -> Tensor<B, 4> {
    let [_, _, height_in, width_in] = tensor.shape().dims::<4>();
    let grid = fix_grid_for_burn::<B>(grid, height_in, width_in);
    tensor.grid_sample_2d(grid, InterpolateMode::Bilinear)
}

pub(crate) fn build_foveated_base_grid<B: BackendTrait>(
    patch_size: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let patch = patch_size.max(1);
    let half = patch as f32 * 0.5;
    let mut coords = Vec::with_capacity(patch * patch * 2);
    for y in 0..patch {
        for x in 0..patch {
            let ux = (x as f32 + 0.5 - half) / half;
            let uy = (y as f32 + 0.5 - half) / half;
            coords.push(ux);
            coords.push(uy);
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(coords, [patch * patch * 2]), device)
        .reshape([patch, patch, 2])
        .unsqueeze_dim::<4>(0)
}

pub(crate) fn build_fovea_jitter<B: BackendTrait>(full_patch_h: usize, device: &B::Device) -> FoveaJitter<B> {
    let subsamples = SACCADE_FOVEA_SUBSAMPLES * SACCADE_FOVEA_SUBSAMPLES;
    let full_half = full_patch_h as f32 * 0.5;
    let scale = if full_half > 0.0 {
        1.0 / full_half
    } else {
        0.0
    };
    let mut jitter_values = Vec::with_capacity(subsamples * 2);
    let mut sequential = Vec::with_capacity(subsamples);
    for sy in 0..SACCADE_FOVEA_SUBSAMPLES {
        for sx in 0..SACCADE_FOVEA_SUBSAMPLES {
            let jitter_x = (sx as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
            let jitter_y = (sy as f32 + 0.5) / SACCADE_FOVEA_SUBSAMPLES as f32 - 0.5;
            let jitter_x = jitter_x * scale;
            let jitter_y = jitter_y * scale;
            jitter_values.push(jitter_x);
            jitter_values.push(jitter_y);
            sequential.push(Tensor::<B, 4>::from_data(
                TensorData::new(vec![jitter_x, jitter_y], [1, 1, 1, 2]),
                device,
            ));
        }
    }
    let batched = Tensor::<B, 5>::from_data(
        TensorData::new(jitter_values, [subsamples, 1, 1, 1, 2]),
        device,
    );
    FoveaJitter { batched, sequential }
}

pub(crate) fn build_image_grid<B: BackendTrait>(
    out_height: usize,
    out_width: usize,
    in_height: usize,
    in_width: usize,
    device: &B::Device,
) -> Tensor<B, 4> {
    let out_height = out_height.max(1);
    let out_width = out_width.max(1);
    let in_height = in_height.max(1);
    let in_width = in_width.max(1);
    let mut coords = Vec::with_capacity(out_height * out_width * 2);
    let scale_x = if in_width > 1 {
        2.0 / (in_width as f32 - 1.0)
    } else {
        0.0
    };
    let scale_y = if in_height > 1 {
        2.0 / (in_height as f32 - 1.0)
    } else {
        0.0
    };
    let denom_w = out_width as f32;
    let denom_h = out_height as f32;
    for y in 0..out_height {
        let fy = (y as f32 + 0.5) / denom_h;
        let gy = if in_height > 1 {
            (fy * in_height as f32 - 0.5) * scale_y - 1.0
        } else {
            0.0
        };
        for x in 0..out_width {
            let fx = (x as f32 + 0.5) / denom_w;
            let gx = if in_width > 1 {
                (fx * in_width as f32 - 0.5) * scale_x - 1.0
            } else {
                0.0
            };
            coords.push(gx);
            coords.push(gy);
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(coords, [out_height * out_width * 2]), device)
        .reshape([out_height, out_width, 2])
        .unsqueeze_dim::<4>(0)
}

pub(crate) fn build_level_coords<B: BackendTrait>(grid: PatchGrid, device: &B::Device) -> Tensor<B, 2> {
    let mut coords = Vec::with_capacity(grid.height * grid.width * 2);
    let inv_w = 1.0 / (grid.width.max(1) as f32);
    let inv_h = 1.0 / (grid.height.max(1) as f32);
    for y in 0..grid.height {
        let cy = (y as f32 + 0.5) * inv_h;
        for x in 0..grid.width {
            let cx = (x as f32 + 0.5) * inv_w;
            coords.push(cx);
            coords.push(cy);
        }
    }
    Tensor::<B, 1>::from_data(
        TensorData::new(coords, [grid.height * grid.width * 2]),
        device,
    )
    .reshape([grid.height * grid.width, 2])
}

pub(crate) fn saccade_eye_color(eye: usize) -> [f32; 3] {
    const PALETTE: [[f32; 3]; 6] = [
        [0.95, 0.25, 0.25],
        [0.25, 0.65, 0.95],
        [0.25, 0.85, 0.4],
        [0.95, 0.75, 0.25],
        [0.75, 0.35, 0.95],
        [0.9, 0.9, 0.2],
    ];
    PALETTE[eye % PALETTE.len()]
}

pub(crate) fn saccade_circle_overlay<B: BackendTrait>(
    images: Tensor<B, 4>,
    mean: Tensor<B, 2>,
    sigma: Tensor<B, 2>,
    color: [f32; 3],
) -> Option<Tensor<B, 4>> {
    let outer = sigma
        .clone()
        .mul_scalar(SACCADE_RING_OUTER_SCALE)
        .max_pair(sigma.clone())
        .clamp_max(1.0 - SACCADE_EPS);
    let images = saccade_ring_overlay(
        images,
        mean.clone(),
        outer,
        color,
        SACCADE_RING_OUTER_INTENSITY,
    )?;
    saccade_ring_overlay(images, mean, sigma, SACCADE_RING_INNER_COLOR, 1.0)
}

pub(crate) fn saccade_ring_overlay<B: BackendTrait>(
    images: Tensor<B, 4>,
    mean: Tensor<B, 2>,
    radius: Tensor<B, 2>,
    color: [f32; 3],
    intensity_scale: f32,
) -> Option<Tensor<B, 4>> {
    let device = images.device();
    let [batch, channels, height, width] = images.shape().dims::<4>();
    if batch == 0 || channels < 3 || height == 0 || width == 0 {
        return None;
    }
    let x_coords = Tensor::<B, 1>::from_data(
        TensorData::new(
            (0..width)
                .map(|x| (x as f32 + 0.5) / width as f32)
                .collect::<Vec<_>>(),
            [width],
        ),
        &device,
    )
    .reshape([1, 1, 1, width]);
    let y_coords = Tensor::<B, 1>::from_data(
        TensorData::new(
            (0..height)
                .map(|y| (y as f32 + 0.5) / height as f32)
                .collect::<Vec<_>>(),
            [height],
        ),
        &device,
    )
    .reshape([1, 1, height, 1]);

    let cx = mean
        .clone()
        .slice_dim(1, 0..1)
        .reshape([batch, 1, 1, 1]);
    let cy = mean
        .slice_dim(1, 1..2)
        .reshape([batch, 1, 1, 1]);
    let radius = radius.reshape([batch, 1, 1, 1]);

    let dx = x_coords - cx;
    let dy = y_coords - cy;
    let dist = (dx.powf_scalar(2.0) + dy.powf_scalar(2.0)).sqrt();
    let ring = dist.sub(radius).abs();
    let ring_mask = activation::relu(
        ring.mul_scalar(-1.0).add_scalar(SACCADE_RING_WIDTH),
    )
    .div_scalar(SACCADE_RING_WIDTH.max(SACCADE_EPS));
    let color_tensor = Tensor::<B, 1>::from_data(
        TensorData::new(vec![color[0], color[1], color[2]], [3]),
        &device,
    )
    .reshape([1, 3, 1, 1]);
    let ring_rgb = ring_mask
        .clone()
        .repeat_dim(1, 3)
        .mul(color_tensor)
        .mul_scalar(SACCADE_RING_INTENSITY * intensity_scale);
    let inv_mask = ring_mask.mul_scalar(-1.0).add_scalar(1.0);
    let overlay = images.mul(inv_mask) + ring_rgb;
    Some(overlay)
}

pub(crate) fn saccade_patch_views<B: BackendTrait>(
    patches: Vec<Tensor<B, 4>>,
    target_height: usize,
) -> Option<Vec<Tensor<B, 4>>> {
    if patches.is_empty() || target_height == 0 {
        return None;
    }
    let mut views = Vec::with_capacity(patches.len());
    for patch in patches {
        let view = pad_view_height_centered(patch, target_height);
        views.push(view);
    }
    Some(views)
}

pub(crate) fn pad_view_width<B: BackendTrait>(view: Tensor<B, 4>, target_width: usize) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_width <= width {
        return view;
    }
    let pad = target_width - width;
    if pad == 0 {
        return view;
    }
    let device = view.device();
    let padding = Tensor::<B, 4>::zeros([batch, channels, height, pad], &device);
    Tensor::cat(vec![view, padding], 3)
}

pub(crate) fn pad_view_width_centered<B: BackendTrait>(
    view: Tensor<B, 4>,
    target_width: usize,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_width <= width {
        return view;
    }
    let pad = target_width - width;
    if pad == 0 {
        return view;
    }
    let left = pad / 2;
    let right = pad - left;
    let device = view.device();
    let padding_left = Tensor::<B, 4>::zeros([batch, channels, height, left], &device);
    let padding_right = Tensor::<B, 4>::zeros([batch, channels, height, right], &device);
    Tensor::cat(vec![padding_left, view, padding_right], 3)
}

pub(crate) fn view_separator_like<B: BackendTrait>(like: &Tensor<B, 4>, width: usize) -> Tensor<B, 4> {
    let [batch, channels, height, _] = like.shape().dims::<4>();
    if width == 0 || batch == 0 || channels == 0 || height == 0 {
        return Tensor::<B, 4>::zeros([batch.max(1), channels.max(1), height.max(1), width.max(1)], &like.device());
    }
    Tensor::<B, 4>::zeros([batch, channels, height, width], &like.device())
}

pub(crate) fn pad_view_height_centered<B: BackendTrait>(
    view: Tensor<B, 4>,
    target_height: usize,
) -> Tensor<B, 4> {
    let [batch, channels, height, width] = view.shape().dims::<4>();
    if target_height <= height {
        return view;
    }
    let pad = target_height - height;
    if pad == 0 {
        return view;
    }
    let top = pad / 2;
    let bottom = pad - top;
    let device = view.device();
    let padding_top = Tensor::<B, 4>::zeros([batch, channels, top, width], &device);
    let padding_bottom = Tensor::<B, 4>::zeros([batch, channels, bottom, width], &device);
    Tensor::cat(vec![padding_top, view, padding_bottom], 2)
}


