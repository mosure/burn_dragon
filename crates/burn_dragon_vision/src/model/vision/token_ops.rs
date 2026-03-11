use super::*;

impl<B: Backend> VisionDragon<B> {
    pub(super) fn apply_token_norm<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        match &self.token_norm {
            Some(norm) => norm.forward(tokens),
            None => tokens,
        }
    }

    pub(super) fn apply_latent_activation<const D: usize>(
        &self,
        values: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match self.latent_activation {
            VisionLatentActivation::Relu => activation::relu(values),
            VisionLatentActivation::Gelu => activation::gelu(values),
            VisionLatentActivation::Identity => values,
        }
    }

    pub(super) fn sync_cls_tokens_multi(&self, tokens: Tensor<B, 4>) -> Tensor<B, 4> {
        if !(self.use_cls_token && self.cls_sync_alpha > 0.0) {
            return tokens;
        }
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if streams <= 1 || time == 0 {
            return tokens;
        }
        let alpha = self.cls_sync_alpha.clamp(0.0, 1.0);
        let cls = tokens.clone().slice_dim(2, 0..1);
        let shared = cls
            .clone()
            .sum_dim(1)
            .mul_scalar(1.0 / streams as f32)
            .reshape([batch, 1, 1, dim])
            .repeat_dim(1, streams);
        let blended = cls.mul_scalar(1.0 - alpha) + shared.mul_scalar(alpha);
        if time <= 1 {
            return blended;
        }
        let rest = tokens.slice_dim(2, 1..time);
        Tensor::cat(vec![blended, rest], 2)
    }

    pub(super) fn full_attention(&self, query: Tensor<B, 4>, value: Tensor<B, 4>) -> Tensor<B, 4> {
        let latent = query.shape().dims::<4>()[3] as f32;
        let scale = latent.sqrt().max(1.0);
        let k = query.clone();
        let query_scaled = query.clone().div_scalar(scale);
        let mut scores = query_scaled.matmul(k.swap_dims(2, 3));
        if self.use_alibi
            && let Some(slopes) = self.alibi_slopes.as_ref()
        {
            let device = query.device();
            let [_, heads, time, _] = query.shape().dims::<4>();
            let slopes = slopes.clone().reshape([1, heads, 1, 1]);
            let pos_row = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, time, 1]);
            let pos_col = Tensor::<B, 1, Int>::arange(0..time as i64, &device)
                .float()
                .reshape([1, 1, 1, time]);
            let alibi = slopes * (pos_col - pos_row);
            scores = scores + alibi;
        }
        match self.attention_mode {
            VisionAttentionMode::Softmax => {
                scores = activation::softmax(scores, 3);
            }
            VisionAttentionMode::RowL1 => {
                let denom = scores.clone().abs().sum_dim(3).add_scalar(ROW_NORM_EPS);
                scores = scores / denom;
            }
        }
        let value = value.repeat_dim(1, self.n_head);
        scores.matmul(value)
    }

    pub(super) fn prepend_cls(&self, tokens: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _time, dim] = tokens.shape().dims::<3>();
        let cls = self
            .cls_token
            .as_ref()
            .expect("cls token enabled")
            .val()
            .reshape([1, 1, dim])
            .repeat_dim(0, batch);
        let cls = if let Some(cls_pos) = &self.cls_pos {
            cls + cls_pos.val().reshape([1, 1, dim])
        } else {
            cls
        };
        Tensor::cat(vec![cls, tokens], 1)
    }

    pub(super) fn split_output(&self, tokens: Tensor<B, 3>) -> VisionDragonOutput<B> {
        let [batch, time, dim] = tokens.shape().dims::<3>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens.clone().slice_dim(1, 0..1).reshape([batch, dim]);
            let patch_tokens = tokens.slice_dim(1, 1..time);
            VisionDragonOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(1).reshape([batch, dim]);
            VisionDragonOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }

    pub(super) fn split_output_multi(&self, tokens: Tensor<B, 4>) -> VisionDragonMultiOutput<B> {
        let [batch, streams, time, dim] = tokens.shape().dims::<4>();
        if self.use_cls_token && time > 0 {
            let cls_token = tokens
                .clone()
                .slice_dim(2, 0..1)
                .reshape([batch, streams, dim]);
            let patch_tokens = tokens.slice_dim(2, 1..time);
            VisionDragonMultiOutput {
                patch_tokens,
                cls_token,
            }
        } else {
            let cls_token = tokens.clone().mean_dim(2).reshape([batch, streams, dim]);
            VisionDragonMultiOutput {
                patch_tokens: tokens,
                cls_token,
            }
        }
    }
}
