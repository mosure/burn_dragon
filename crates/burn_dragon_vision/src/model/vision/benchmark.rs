use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, activation};

use burn_dragon_core::kernel::relu_lowrank;
use burn_dragon_core::{
    LowBitProjectionPlan, LowBitSavedActivationConfig, PackedLowBitProjectionArtifacts,
    lowrank_residual_step,
};

use crate::{
    VisionDistillConfig, VisionDistillationLossConfig, VisionDragonOutput,
    vision_distillation_loss_terms,
};

use super::{VisionAttentionMode, VisionBackboneKind, VisionDragon, VisionLatentActivation};

const BENCH_ROW_NORM_EPS: f32 = 1e-6;

/// Stable benchmark adapter for decomposing the dense recurrent step.
///
/// The long-term benchmark surface should depend on this adapter rather than on ad hoc helper
/// methods attached directly to `VisionDragon`.
pub struct VisionDenseBenchAdapter<'a, B: Backend> {
    model: &'a VisionDragon<B>,
}

#[derive(Clone, Debug)]
pub struct VisionDenseAttentionBenchAdapter {
    attention_mode: VisionAttentionMode,
    slopes: Option<Vec<f32>>,
}

impl VisionDenseAttentionBenchAdapter {
    pub fn new(attention_mode: VisionAttentionMode, slopes: Option<Vec<f32>>) -> Self {
        Self {
            attention_mode,
            slopes,
        }
    }

    pub fn from_model<B: Backend>(model: &VisionDragon<B>) -> Self {
        let slopes = if model.use_alibi {
            model.alibi_slopes.as_ref().map(|slopes| {
                slopes
                    .clone()
                    .to_data()
                    .convert::<f32>()
                    .into_vec::<f32>()
                    .expect("alibi slopes vec")
            })
        } else {
            None
        };
        Self::new(model.attention_mode, slopes)
    }

    pub fn qk_scores<B: Backend>(&self, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        let [batch, heads, time, latent_dim] = query.shape().dims::<4>();
        let query_scaled = query.clone().div_scalar(scale);
        let query_scaled_flat = query_scaled.reshape([batch * heads, time, latent_dim]);
        let key_flat = query
            .reshape([batch * heads, time, latent_dim])
            .swap_dims(1, 2);
        query_scaled_flat
            .matmul(key_flat)
            .reshape([batch, heads, time, time])
    }

    pub fn qk_scores_direct_4d<B: Backend>(&self, query: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        query
            .clone()
            .div_scalar(scale)
            .matmul(query.swap_dims(2, 3))
    }

    pub fn apply_alibi_and_norm<B: Backend>(&self, mut scores: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_, heads, time, _] = scores.shape().dims::<4>();
        if let Some(slopes) = self.slopes.as_ref() {
            let device = scores.device();
            let slopes =
                Tensor::<B, 1>::from_data(slopes.as_slice(), &device).reshape([1, heads, 1, 1]);
            let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, time, 1]);
            let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, 1, time]);
            scores = scores + slopes * (pos_col - pos_row);
        }
        match self.attention_mode {
            VisionAttentionMode::Softmax => activation::softmax(scores, 3),
            VisionAttentionMode::RowL1 => {
                let denom = scores
                    .clone()
                    .abs()
                    .sum_dim(3)
                    .add_scalar(BENCH_ROW_NORM_EPS);
                scores / denom
            }
        }
    }

    pub fn expand_shared_value_for_heads<B: Backend>(
        &self,
        value: Tensor<B, 4>,
        heads: usize,
    ) -> Tensor<B, 3> {
        let [batch, _, value_time, value_dim] = value.shape().dims::<4>();
        value
            .reshape([batch, 1, value_time, value_dim])
            .repeat_dim(1, heads)
            .reshape([batch * heads, value_time, value_dim])
    }

    pub fn repeated_value_attention<B: Backend>(
        &self,
        scores: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, _] = scores.shape().dims::<4>();
        let value_flat = self.expand_shared_value_for_heads(value, heads);
        let value_dim = value_flat.shape().dims::<3>()[2];
        scores
            .reshape([batch * heads, time, time])
            .matmul(value_flat)
            .reshape([batch, heads, time, value_dim])
    }

    pub fn shared_value_batchmatmul<B: Backend>(
        &self,
        scores: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let [batch, heads, time, _] = scores.shape().dims::<4>();
        let [_, _, value_time, value_dim] = value.shape().dims::<4>();
        scores
            .swap_dims(1, 2)
            .reshape([batch, time * heads, time])
            .matmul(value.reshape([batch, value_time, value_dim]))
            .reshape([batch, time, heads, value_dim])
            .swap_dims(1, 2)
    }

    pub fn full_attention_current<B: Backend>(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let scores = self.qk_scores(query);
        let scores = self.apply_alibi_and_norm(scores);
        self.repeated_value_attention(scores, value)
    }

    pub fn full_attention_blockwise_row_l1<B: Backend>(
        &self,
        query: Tensor<B, 4>,
        value: Tensor<B, 4>,
        block_size: usize,
    ) -> Tensor<B, 4> {
        assert!(
            matches!(self.attention_mode, VisionAttentionMode::RowL1),
            "full_attention_blockwise_row_l1 only supports row_l1 attention"
        );
        let device = query.device();
        let [batch, heads, time, _] = query.shape().dims::<4>();
        let latent = query.shape().dims::<4>()[3] as f32;
        let value_dim = value.shape().dims::<4>()[3];
        let block_size = block_size.max(1);
        let total_blocks = time.div_ceil(block_size);
        let q = query.div_scalar(latent.sqrt().max(1.0));
        let v = value.repeat_dim(1, heads);
        let slopes = self.slopes.as_ref().map(|slopes| {
            Tensor::<B, 1>::from_data(slopes.as_slice(), &device).reshape([1, heads, 1, 1])
        });
        let positions = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
            .float()
            .reshape([1, 1, time, 1]);

        let mut outputs = Vec::with_capacity(total_blocks);
        for row in 0..total_blocks {
            let row_start = row * block_size;
            let row_end = usize::min(row_start + block_size, time);
            let row_len = row_end - row_start;
            let row_range = row_start..row_end;
            let q_block = q.clone().slice_dim(2, row_range.clone());
            let pos_row = positions
                .clone()
                .slice_dim(2, row_range)
                .reshape([1, 1, row_len, 1]);
            let mut block_acc = Tensor::<B, 4>::zeros([batch, heads, row_len, value_dim], &device);
            let mut row_norm = Tensor::<B, 4>::zeros([batch, heads, row_len, 1], &device);
            for col in 0..total_blocks {
                let col_start = col * block_size;
                let col_end = usize::min(col_start + block_size, time);
                let col_len = col_end - col_start;
                let col_range = col_start..col_end;
                let k_block = q.clone().slice_dim(2, col_range.clone());
                let mut scores = q_block.clone().matmul(k_block.swap_dims(2, 3));
                if let Some(slopes) = slopes.as_ref() {
                    let pos_col = positions
                        .clone()
                        .slice_dim(2, col_range.clone())
                        .reshape([1, 1, 1, col_len]);
                    scores = scores + slopes.clone() * (pos_col - pos_row.clone());
                }
                let v_block = v.clone().slice_dim(2, col_range);
                row_norm = row_norm + scores.clone().abs().sum_dim(3);
                block_acc = block_acc + scores.matmul(v_block);
            }
            outputs.push(block_acc / row_norm.add_scalar(BENCH_ROW_NORM_EPS));
        }
        Tensor::cat(outputs, 2)
    }
}

impl<'a, B: Backend> VisionDenseBenchAdapter<'a, B> {
    pub fn new(model: &'a VisionDragon<B>) -> Self {
        assert_eq!(
            model.backbone_kind,
            VisionBackboneKind::Dense,
            "VisionDenseBenchAdapter only supports the dense backbone",
        );
        Self { model }
    }

    pub fn x_projection(&self, current: Tensor<B, 4>) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        self.x_projection_with_encoder(current, self.x_projection_encoder())
    }

    pub fn x_projection_reference(&self, current: Tensor<B, 4>) -> Tensor<B, 4> {
        self.x_projection_reference_with_encoder(current, self.x_projection_encoder())
    }

    pub fn x_projection_encoder(&self) -> Tensor<B, 4> {
        let encoder_raw = self.model.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        encoder_raw.reshape([1, heads, embd_enc, latent])
    }

    pub fn x_projection_with_encoder(
        &self,
        current: Tensor<B, 4>,
        encoder: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let fused = self.model.kernel.enabled
            && matches!(self.model.latent_activation, VisionLatentActivation::Relu);
        let fused = fused && self.model.kernel.projection_executor.use_x();
        let latent = encoder.shape().dims::<4>()[3];
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        if fused {
            relu_lowrank::fused_forward_with_executor(
                current,
                encoder,
                None,
                self.model.kernel.relu_threshold,
                latent_pattern,
                sparse_mask,
                self.model.kernel.lowrank_grad_input_executor,
            )
        } else {
            self.x_projection_reference_with_encoder(current, encoder)
        }
    }

    pub fn x_projection_reference_with_encoder(
        &self,
        current: Tensor<B, 4>,
        encoder: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let latent = encoder.shape().dims::<4>()[3];
        let sparse_mask = if latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        relu_lowrank::reference_forward(
            current,
            encoder,
            None,
            self.model.kernel.relu_threshold,
            latent_pattern,
            sparse_mask,
        )
    }

    pub fn attention_context(&self, x_neuron: Tensor<B, 4>, current: Tensor<B, 4>) -> Tensor<B, 4> {
        self.model
            .apply_token_norm(self.model.full_attention(x_neuron, current))
    }

    pub fn y_projection(&self, attn: Tensor<B, 4>) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        self.y_projection_with_encoder(attn, self.y_projection_encoder())
    }

    pub fn y_projection_reference(&self, attn: Tensor<B, 4>) -> Tensor<B, 4> {
        self.y_projection_reference_with_encoder(attn, self.y_projection_encoder())
    }

    pub fn y_projection_encoder(&self) -> Tensor<B, 4> {
        let encoder_v_raw = self.model.encoder_v.val();
        let [heads, embd, latent] = encoder_v_raw.shape().dims::<3>();
        encoder_v_raw.reshape([1, heads, embd, latent])
    }

    pub fn y_projection_with_encoder(
        &self,
        attn: Tensor<B, 4>,
        encoder_v: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let fused = self.model.kernel.enabled
            && matches!(self.model.latent_activation, VisionLatentActivation::Relu);
        let fused = fused && self.model.kernel.projection_executor.use_y();
        let latent = encoder_v.shape().dims::<4>()[3];
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let sparse_mask = if fused && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &attn.device()))
        } else {
            None
        };
        if fused {
            relu_lowrank::fused_forward_with_executor(
                attn,
                encoder_v,
                None,
                self.model.kernel.relu_threshold,
                latent_pattern,
                sparse_mask,
                self.model.kernel.lowrank_grad_input_executor,
            )
        } else {
            self.y_projection_reference_with_encoder(attn, encoder_v)
        }
    }

    pub fn y_projection_reference_with_encoder(
        &self,
        attn: Tensor<B, 4>,
        encoder_v: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let latent = encoder_v.shape().dims::<4>()[3];
        let sparse_mask = if latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &attn.device()))
        } else {
            None
        };
        relu_lowrank::reference_forward(
            attn,
            encoder_v,
            None,
            self.model.kernel.relu_threshold,
            latent_pattern,
            sparse_mask,
        )
    }

    pub fn y_path_from_attention(
        &self,
        current: Tensor<B, 4>,
        x_neuron: Tensor<B, 4>,
        attn: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        self.tail(current, x_neuron, self.y_projection(attn))
    }

    pub fn y_path_reference_from_attention(
        &self,
        current: Tensor<B, 4>,
        x_neuron: Tensor<B, 4>,
        attn: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        self.tail(current, x_neuron, self.y_projection_reference(attn))
    }

    pub fn y_path_with_encoder_from_attention(
        &self,
        current: Tensor<B, 4>,
        x_neuron: Tensor<B, 4>,
        attn: Tensor<B, 4>,
        encoder_v: Tensor<B, 4>,
    ) -> Tensor<B, 4>
    where
        B::FloatTensorPrimitive: 'static,
    {
        self.tail(
            current,
            x_neuron,
            self.y_projection_with_encoder(attn, encoder_v),
        )
    }

    pub fn tail(
        &self,
        current: Tensor<B, 4>,
        x_neuron: Tensor<B, 4>,
        y_gate: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        let y_neuron = self.model.dropout.forward(x_neuron * y_gate);
        let [batch, heads, time, latent] = y_neuron.shape().dims::<4>();
        let mlp_out = if heads == 1 {
            y_neuron
                .reshape([batch * time, latent])
                .matmul(self.model.decoder.val())
                .reshape([batch, 1, time, self.model.embed_dim])
        } else {
            let decoder = self
                .model
                .decoder
                .val()
                .reshape([heads, latent, self.model.embed_dim]);
            y_neuron
                .swap_dims(0, 1)
                .reshape([heads, batch * time, latent])
                .matmul(decoder)
                .sum_dim(0)
                .reshape([batch, 1, time, self.model.embed_dim])
        };
        let mlp_out = self.model.apply_token_norm(mlp_out);
        self.model.apply_token_norm(current + mlp_out)
    }

    pub fn step(&self, current: Tensor<B, 4>) -> Tensor<B, 4> {
        let encoder_raw = self.model.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);

        let encoder_v_raw = self.model.encoder_v.val();
        let [heads_v, embd_v, latent_v] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads_v, embd_v, latent_v]);

        let fused = self.model.kernel.enabled
            && matches!(self.model.latent_activation, VisionLatentActivation::Relu);
        let fused_x = fused && self.model.kernel.projection_executor.use_x();
        let fused_y = fused && self.model.kernel.projection_executor.use_y();
        let apply_threshold = matches!(self.model.latent_activation, VisionLatentActivation::Relu);
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let sparse_mask = if (fused_x || fused_y) && latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };

        lowrank_residual_step(
            current,
            encoder,
            encoder_v,
            self.model.decoder.val(),
            &self.model.dropout,
            fused_x,
            fused_y,
            self.model.kernel.relu_threshold,
            apply_threshold,
            LowBitProjectionPlan::default(),
            LowBitSavedActivationConfig::default(),
            PackedLowBitProjectionArtifacts::default(),
            latent_pattern,
            self.model.kernel.lowrank_grad_input_executor,
            sparse_mask,
            |query, value| self.model.full_attention(query, value),
            |values| self.model.apply_latent_activation(values),
            |values| self.model.apply_token_norm(values),
        )
        .next
    }

    pub fn x_projection_wgpu_kernel(&self, current: Tensor<B, 4>) -> Option<Tensor<B, 4>>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let encoder_raw = self.model.encoder.val();
        let [heads, embd_enc, latent] = encoder_raw.shape().dims::<3>();
        let encoder = encoder_raw.reshape([1, heads, embd_enc, latent]);
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let sparse_mask = if latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &current.device()))
        } else {
            None
        };
        relu_lowrank::try_wgpu_fused_forward_with_executor(
            &current,
            &encoder,
            None,
            self.model.kernel.relu_threshold,
            sparse_mask.as_ref(),
            self.model.kernel.lowrank_grad_input_executor,
        )
    }

    pub fn y_projection_wgpu_kernel(&self, attn: Tensor<B, 4>) -> Option<Tensor<B, 4>>
    where
        B::FloatTensorPrimitive: 'static,
    {
        let encoder_v_raw = self.model.encoder_v.val();
        let [heads, embd, latent] = encoder_v_raw.shape().dims::<3>();
        let encoder_v = encoder_v_raw.reshape([1, heads, embd, latent]);
        let latent_pattern = &self.model.kernel.block_sparse.latent;
        let sparse_mask = if latent_pattern.is_sparse() {
            Some(latent_pattern.mask::<B>(latent, &attn.device()))
        } else {
            None
        };
        relu_lowrank::try_wgpu_fused_forward_with_executor(
            &attn,
            &encoder_v,
            None,
            self.model.kernel.relu_threshold,
            sparse_mask.as_ref(),
            self.model.kernel.lowrank_grad_input_executor,
        )
    }
}

/// Benchmark adapter for repeated-vs-scheduled rollout distill evaluation.
pub struct VisionRolloutScheduleBenchAdapter<'a, B: Backend> {
    model: &'a VisionDragon<B>,
}

impl<'a, B: Backend> VisionRolloutScheduleBenchAdapter<'a, B> {
    pub fn new(model: &'a VisionDragon<B>) -> Self {
        Self { model }
    }

    pub fn select_trajectory_indices(total: usize, max: usize) -> Vec<usize> {
        if total == 0 || max == 0 {
            return Vec::new();
        }
        if max >= total {
            return (0..total).collect();
        }
        if max == 1 {
            return vec![total - 1];
        }
        let last = (total - 1) as f32;
        let denom = (max - 1) as f32;
        let mut indices = Vec::with_capacity(max);
        for i in 0..max {
            let idx = ((i as f32) * last / denom).round() as usize;
            indices.push(idx.min(total - 1));
        }
        indices.sort_unstable();
        indices.dedup();
        indices
    }

    pub fn rollout_supervision_steps(
        total_steps: usize,
        frames: usize,
        stride: usize,
    ) -> Vec<usize> {
        if total_steps == 0 {
            return Vec::new();
        }
        let stride = stride.max(1);
        let candidates = (1..=total_steps)
            .filter(|step| *step == total_steps || *step == 1 || ((*step - 1) % stride) == 0)
            .collect::<Vec<_>>();
        let mut steps = if candidates.len() <= frames.max(1) {
            candidates
        } else {
            Self::select_trajectory_indices(candidates.len(), frames.max(1))
                .into_iter()
                .map(|index| candidates[index])
                .collect::<Vec<_>>()
        };
        steps.push(total_steps);
        steps.sort_unstable();
        steps.dedup();
        steps
    }

    pub fn default_distill_steps(
        distill: &VisionDistillConfig,
        rollout_steps: usize,
    ) -> Vec<usize> {
        let steps = Self::rollout_supervision_steps(
            rollout_steps,
            distill.rollout_supervision_frames,
            distill.rollout_supervision_stride,
        );
        if steps.len() >= 2 {
            steps
        } else {
            Self::normalize_steps(&[1, 2, 4, rollout_steps], rollout_steps)
        }
    }

    pub fn normalize_steps(steps: &[usize], max_step: usize) -> Vec<usize> {
        let mut steps = steps
            .iter()
            .map(|step| (*step).max(1).min(max_step.max(1)))
            .collect::<Vec<_>>();
        steps.sort_unstable();
        steps.dedup();
        steps
    }

    pub fn repeated_distill_loss(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        schedule: &[(usize, usize)],
        loss: &VisionDistillationLossConfig,
    ) -> Tensor<B, 1> {
        let mut total = None;
        for (step, backprop_steps) in schedule.iter().copied() {
            let output =
                self.model
                    .forward_images_steps_rollout(images.clone(), step, backprop_steps);
            total = Some(match total {
                Some(acc) => {
                    acc + self.distill_total(
                        output,
                        teacher_patch.clone(),
                        teacher_cls.clone(),
                        loss,
                    )
                }
                None => {
                    self.distill_total(output, teacher_patch.clone(), teacher_cls.clone(), loss)
                }
            });
        }
        total.expect("at least one repeated rollout step")
    }

    pub fn scheduled_distill_loss(
        &self,
        images: Tensor<B, 4>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        schedule: &[(usize, usize)],
        loss: &VisionDistillationLossConfig,
    ) -> Tensor<B, 1> {
        let outputs = self
            .model
            .forward_images_steps_rollout_schedule(images, schedule);
        let mut total = None;
        for (_step, output) in outputs {
            total = Some(match total {
                Some(acc) => {
                    acc + self.distill_total(
                        output,
                        teacher_patch.clone(),
                        teacher_cls.clone(),
                        loss,
                    )
                }
                None => {
                    self.distill_total(output, teacher_patch.clone(), teacher_cls.clone(), loss)
                }
            });
        }
        total.expect("at least one scheduled rollout step")
    }

    fn distill_total(
        &self,
        output: VisionDragonOutput<B>,
        teacher_patch: Tensor<B, 3>,
        teacher_cls: Tensor<B, 2>,
        loss: &VisionDistillationLossConfig,
    ) -> Tensor<B, 1> {
        vision_distillation_loss_terms(
            output.patch_tokens,
            teacher_patch,
            output.cls_token,
            teacher_cls,
            loss,
        )
        .total
    }
}
