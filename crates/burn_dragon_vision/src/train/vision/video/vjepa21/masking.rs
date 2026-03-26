use crate::train::prelude::*;

#[derive(Debug, Clone)]
pub(super) struct Vjepa21MaskBatch<B: BackendTrait> {
    pub(super) visible: Tensor<B, 3>,
    pub(super) target: Tensor<B, 3>,
    pub(super) context_distance: Tensor<B, 3>,
    pub(super) visible_ratio: Tensor<B, 1>,
}

pub(super) fn sample_mask_batch<B: BackendTrait>(
    batch_size: usize,
    clip_len: usize,
    patch_count: usize,
    config: &VisionVideoVjepa21Config,
    device: &B::Device,
) -> Vjepa21MaskBatch<B> {
    let grid_h = (patch_count as f64).sqrt() as usize;
    let grid_h = grid_h.max(1);
    let grid_w = patch_count.div_ceil(grid_h).max(1);
    let total_tokens = clip_len * patch_count;
    let mut rng = thread_rng();
    let mut visible_data = vec![0.0_f32; batch_size * total_tokens];
    let mut target_data = vec![0.0_f32; batch_size * total_tokens];
    let mut distance_data = vec![1.0_f32; batch_size * total_tokens];
    let max_context_duration = ((clip_len as f32) * config.mask.max_context_frames_ratio)
        .floor()
        .max(1.0) as usize;
    let max_context_duration = max_context_duration.clamp(1, clip_len.max(1));

    for batch_idx in 0..batch_size {
        let mut visible = vec![true; total_tokens];
        for _ in 0..config.mask.num_blocks.max(1) {
            let temporal_scale = rng.gen_range(
                config
                    .mask
                    .temporal_scale_min
                    .min(config.mask.temporal_scale_max)
                    ..=config
                        .mask
                        .temporal_scale_min
                        .max(config.mask.temporal_scale_max),
            );
            let temporal_span = ((clip_len as f32) * temporal_scale).round() as usize;
            let temporal_span = temporal_span.clamp(1, clip_len.max(1));

            let spatial_scale = rng.gen_range(
                config
                    .mask
                    .spatial_scale_min
                    .min(config.mask.spatial_scale_max)
                    ..=config
                        .mask
                        .spatial_scale_min
                        .max(config.mask.spatial_scale_max),
            );
            let spatial_keep = ((patch_count as f32) * spatial_scale).round() as usize;
            let spatial_keep = spatial_keep.clamp(1, patch_count.max(1));

            let aspect_ratio = rng.gen_range(
                config
                    .mask
                    .aspect_ratio_min
                    .min(config.mask.aspect_ratio_max)
                    ..=config
                        .mask
                        .aspect_ratio_min
                        .max(config.mask.aspect_ratio_max),
            );
            let mut block_h = ((spatial_keep as f32 * aspect_ratio).sqrt().round() as usize)
                .clamp(1, grid_h.max(1));
            let mut block_w = ((spatial_keep as f32 / aspect_ratio).sqrt().round() as usize)
                .clamp(1, grid_w.max(1));
            if block_h * block_w > patch_count {
                block_h = block_h.min(grid_h.max(1));
                block_w = block_w.min(grid_w.max(1));
            }
            let top = if grid_h > block_h {
                rng.gen_range(0..=grid_h - block_h)
            } else {
                0
            };
            let left = if grid_w > block_w {
                rng.gen_range(0..=grid_w - block_w)
            } else {
                0
            };
            let start_t = if clip_len > temporal_span {
                rng.gen_range(0..=clip_len - temporal_span)
            } else {
                0
            };

            for t in start_t..(start_t + temporal_span).min(clip_len) {
                for y in top..(top + block_h).min(grid_h) {
                    for x in left..(left + block_w).min(grid_w) {
                        let token_idx = y * grid_w + x;
                        if token_idx < patch_count {
                            visible[t * patch_count + token_idx] = false;
                        }
                    }
                }
            }
        }
        for t in max_context_duration..clip_len {
            for token_idx in 0..patch_count {
                visible[t * patch_count + token_idx] = false;
            }
        }
        if visible.iter().all(|flag| !*flag) {
            visible[0] = true;
        }
        let masked_positions = visible
            .iter()
            .enumerate()
            .filter_map(|(idx, visible)| (!*visible).then_some(idx))
            .collect::<Vec<_>>();
        if masked_positions.is_empty() {
            visible[0] = false;
        }
        let masked_positions = visible
            .iter()
            .enumerate()
            .filter_map(|(idx, visible)| (!*visible).then_some(idx))
            .collect::<Vec<_>>();
        let mut visible_count = 0usize;
        for token_idx in 0..total_tokens {
            let row = batch_idx * total_tokens + token_idx;
            let is_visible = visible[token_idx];
            visible_data[row] = if is_visible { 1.0 } else { 0.0 };
            target_data[row] = if is_visible { 0.0 } else { 1.0 };
            if is_visible {
                visible_count += 1;
                let t = token_idx / patch_count;
                let local = token_idx % patch_count;
                let y = local / grid_w;
                let x = local % grid_w;
                let mut min_distance = f32::MAX;
                for &masked in &masked_positions {
                    let masked_t = masked / patch_count;
                    let masked_local = masked % patch_count;
                    let masked_y = masked_local / grid_w;
                    let masked_x = masked_local % grid_w;
                    let dt = masked_t as f32 - t as f32;
                    let dy = masked_y as f32 - y as f32;
                    let dx = masked_x as f32 - x as f32;
                    let distance = (dt * dt + dy * dy + dx * dx).sqrt().max(1.0);
                    min_distance = min_distance.min(distance);
                }
                let offset = if config.loss.offset_context_loss {
                    (grid_w.max(grid_h) / 16).max(1) as f32
                } else {
                    1.0
                };
                distance_data[row] = (min_distance / offset).sqrt().max(1.0);
            } else {
                distance_data[row] = 1.0;
            }
        }
        let visible_ratio = visible_count as f32 / total_tokens.max(1) as f32;
        distance_data[batch_idx * total_tokens] = distance_data[batch_idx * total_tokens].max(1.0);
        target_data[batch_idx * total_tokens] = target_data[batch_idx * total_tokens].max(0.0);
        visible_data[batch_idx * total_tokens] = visible_data[batch_idx * total_tokens].max(0.0);
        let _ = visible_ratio;
    }

    let visible = Tensor::<B, 3>::from_data(
        TensorData::new(visible_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let target = Tensor::<B, 3>::from_data(
        TensorData::new(target_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let context_distance = Tensor::<B, 3>::from_data(
        TensorData::new(distance_data, [batch_size, clip_len, patch_count]),
        device,
    );
    let visible_ratio = visible
        .clone()
        .sum()
        .div_scalar((batch_size * total_tokens).max(1) as f32)
        .reshape([1]);

    Vjepa21MaskBatch {
        visible,
        target,
        context_distance,
        visible_ratio,
    }
}
