use crate::VisionDragonOutput;
use crate::train::prelude::*;
use burn_dragon_core::{ModelState, StructuredTopologyState};

const VIDEO_FRAME_ENCODE_CHUNK: usize = 512;

#[derive(Module, Debug)]
pub(crate) struct VisionVideoPredictor<B: BackendTrait> {
    norm: DragonNorm<B>,
    hidden: Option<Linear<B>>,
    out: Linear<B>,
}

impl<B: BackendTrait> VisionVideoPredictor<B> {
    pub(crate) fn new(
        embed_dim: usize,
        hidden_dim: usize,
        projection_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        let norm = DragonNorm::new(norm_config, embed_dim.max(1), device);
        let hidden = if hidden_dim > 0 {
            Some(LinearConfig::new(embed_dim.max(1), hidden_dim).init(device))
        } else {
            None
        };
        let out_in = if hidden.is_some() {
            hidden_dim.max(1)
        } else {
            embed_dim.max(1)
        };
        let out = LinearConfig::new(out_in, projection_dim.max(1)).init(device);
        Self { norm, hidden, out }
    }

    pub(crate) fn forward<const D: usize>(&self, tokens: Tensor<B, D>) -> Tensor<B, D> {
        let tokens = self.norm.forward(tokens);
        let tokens = if let Some(hidden) = &self.hidden {
            activation::gelu(hidden.forward(tokens))
        } else {
            tokens
        };
        self.out.forward(tokens)
    }
}

#[derive(Module, Debug)]
pub(crate) struct VisionVideoObservationMerger<B: BackendTrait> {
    prior_norm: DragonNorm<B>,
    obs_norm: DragonNorm<B>,
    gate_prior: Linear<B>,
    gate_obs: Linear<B>,
}

impl<B: BackendTrait> VisionVideoObservationMerger<B> {
    pub(crate) fn new(
        embed_dim: usize,
        norm_config: &DragonNormConfig,
        device: &B::Device,
    ) -> Self {
        Self {
            prior_norm: DragonNorm::new(norm_config, embed_dim.max(1), device),
            obs_norm: DragonNorm::new(norm_config, embed_dim.max(1), device),
            gate_prior: LinearConfig::new(embed_dim.max(1), embed_dim.max(1)).init(device),
            gate_obs: LinearConfig::new(embed_dim.max(1), embed_dim.max(1)).init(device),
        }
    }

    pub(crate) fn forward(&self, prior: Tensor<B, 3>, observation: Tensor<B, 3>) -> Tensor<B, 3> {
        let prior_norm = self.prior_norm.forward(prior.clone());
        let obs_norm = self.obs_norm.forward(observation.clone());
        let gate = activation::sigmoid(
            self.gate_prior.forward(prior_norm) + self.gate_obs.forward(obs_norm),
        );
        prior.clone() + gate * (observation - prior)
    }
}

pub(crate) struct VisionVideoForward<B: BackendTrait> {
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) predicted_proj: Tensor<B, 3>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) target_proj: Tensor<B, 3>,
    pub(crate) predicted_proj_all: Tensor<B, 3>,
    pub(crate) target_proj_all: Tensor<B, 3>,
    pub(crate) observation_proj: Tensor<B, 3>,
    pub(crate) observation_target_proj: Tensor<B, 3>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) context_posterior_patch_tokens: Tensor<B, 4>,
    #[allow(dead_code)]
    pub(crate) context_posterior_cls_embed: Tensor<B, 3>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) context_summary: Tensor<B, 2>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) cls_embed: Tensor<B, 3>,
    pub(crate) frame_patch_tokens: Tensor<B, 4>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) future_hidden_all: Tensor<B, 3>,
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) future_cls_embed: Tensor<B, 3>,
    pub(crate) future_patch_tokens: Tensor<B, 4>,
    pub(crate) context_structured_state: Option<StructuredTopologyState<B>>,
    pub(crate) probe_logits: Option<Tensor<B, 2>>,
    pub(crate) probe_labels: Option<Tensor<B, 1, Int>>,
    pub(crate) clip_frames: Tensor<B, 5>,
    pub(crate) context_len: usize,
    pub(crate) target_len: usize,
    pub(crate) future_len_all: usize,
}

pub(crate) struct VisionVideoContextForward<B: BackendTrait> {
    pub(crate) posterior_patch_tokens: Tensor<B, 4>,
    pub(crate) posterior_cls_embed: Tensor<B, 3>,
    pub(crate) patch_tokens: Tensor<B, 4>,
    pub(crate) cls_embed: Tensor<B, 3>,
    pub(crate) hidden: Tensor<B, 3>,
    pub(crate) temporal_state: ModelState<B>,
    pub(crate) structured_state: Option<StructuredTopologyState<B>>,
}

pub(crate) fn encode_clip_frames_with_model<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_frames: Tensor<B, 5>,
    steps: usize,
    backprop_steps: usize,
    embed_dim: usize,
    projection_dim: usize,
) -> (Tensor<B, 3>, Tensor<B, 4>, Tensor<B, 3>) {
    let [batch_size, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
    let flat_count = batch_size * clip_len;
    let flat_frames = clip_frames.reshape([flat_count, channels, height, width]);
    let chunk_size = VIDEO_FRAME_ENCODE_CHUNK.max(1);
    let mut patch_tokens_chunks = Vec::new();
    let mut cls_embed_chunks = Vec::new();
    let mut cls_proj_chunks = Vec::new();

    for chunk_start in (0..flat_count).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(flat_count);
        let chunk_len = chunk_end.saturating_sub(chunk_start);
        if chunk_len == 0 {
            continue;
        }
        let frames_chunk = flat_frames.clone().slice_dim(0, chunk_start..chunk_end);
        let patch = model.patch_embed(frames_chunk);
        let encoded = model.forward_tokens_embed_steps_rollout(patch.tokens, steps, backprop_steps);
        let cls_embed_chunk = encoded.cls_token;
        let cls_proj_chunk = model
            .project_tokens(cls_embed_chunk.clone().unsqueeze_dim::<3>(1))
            .reshape([chunk_len, projection_dim.max(1)]);
        patch_tokens_chunks.push(encoded.patch_tokens);
        cls_embed_chunks.push(cls_embed_chunk);
        cls_proj_chunks.push(cls_proj_chunk);
    }

    let patch_tokens_flat = if patch_tokens_chunks.len() == 1 {
        patch_tokens_chunks
            .pop()
            .expect("missing patch token chunk")
    } else {
        Tensor::cat(patch_tokens_chunks, 0)
    };
    let cls_embed_flat = if cls_embed_chunks.len() == 1 {
        cls_embed_chunks.pop().expect("missing cls chunk")
    } else {
        Tensor::cat(cls_embed_chunks, 0)
    };
    let cls_proj_flat = if cls_proj_chunks.len() == 1 {
        cls_proj_chunks.pop().expect("missing cls projection chunk")
    } else {
        Tensor::cat(cls_proj_chunks, 0)
    };
    let cls_embed = cls_embed_flat
        .clone()
        .reshape([batch_size, clip_len, embed_dim.max(1)]);
    let [_, token_count, patch_dim] = patch_tokens_flat.shape().dims::<3>();
    let frame_patch_tokens =
        patch_tokens_flat.reshape([batch_size, clip_len, token_count, patch_dim]);
    let cls_proj = cls_proj_flat.reshape([batch_size, clip_len, projection_dim.max(1)]);
    (cls_embed, frame_patch_tokens, cls_proj)
}

fn project_clip_frames_with_model_only<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_frames: Tensor<B, 5>,
    steps: usize,
    projection_dim: usize,
) -> Tensor<B, 3> {
    let [batch_size, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
    let flat_count = batch_size * clip_len;
    let flat_frames = clip_frames.reshape([flat_count, channels, height, width]);
    let chunk_size = VIDEO_FRAME_ENCODE_CHUNK.max(1);
    let mut cls_proj_chunks = Vec::new();

    for chunk_start in (0..flat_count).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(flat_count);
        let chunk_len = chunk_end.saturating_sub(chunk_start);
        if chunk_len == 0 {
            continue;
        }
        let frames_chunk = flat_frames.clone().slice_dim(0, chunk_start..chunk_end);
        let patch = model.patch_embed(frames_chunk);
        let encoded = model.forward_tokens_embed_steps_rollout(patch.tokens, steps, steps);
        let cls_proj_chunk = model
            .project_tokens(encoded.cls_token.unsqueeze_dim::<3>(1))
            .reshape([chunk_len, projection_dim.max(1)]);
        cls_proj_chunks.push(cls_proj_chunk);
    }

    let cls_proj_flat = if cls_proj_chunks.len() == 1 {
        cls_proj_chunks.pop().expect("missing cls projection chunk")
    } else {
        Tensor::cat(cls_proj_chunks, 0)
    };
    cls_proj_flat.reshape([batch_size, clip_len, projection_dim.max(1)])
}

pub(crate) fn embed_clip_frames_raw_with_model<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_frames: Tensor<B, 5>,
) -> Tensor<B, 4> {
    let [batch_size, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
    let flat_frames = clip_frames.reshape([batch_size * clip_len, channels, height, width]);
    let patch = model.patch_embed(flat_frames);
    let [_, token_count, dim] = patch.tokens.shape().dims::<3>();
    patch
        .tokens
        .reshape([batch_size, clip_len, token_count, dim])
}

pub(crate) fn project_clip_frames_with_model<B: BackendTrait>(
    model: &VisionDragon<B>,
    clip_frames: Tensor<B, 5>,
    steps: usize,
    projection_dim: usize,
) -> Tensor<B, 3> {
    project_clip_frames_with_model_only(model, clip_frames, steps, projection_dim)
}

pub(crate) fn split_clip_observation_and_target_projections<B: BackendTrait>(
    model: &VisionDragon<B>,
    teacher_model: Option<&VisionDragon<B>>,
    clip_frames: Tensor<B, 5>,
    context_len: usize,
    future_len_all: usize,
    steps: usize,
    projection_dim: usize,
) -> (Tensor<B, 4>, Tensor<B, 3>, Tensor<B, 3>) {
    let context_clip_frames = clip_frames.clone().slice_dim(1, 0..context_len);
    let context_observation_patch_tokens =
        embed_clip_frames_raw_with_model(model, context_clip_frames);
    let projected = if let Some(teacher) = teacher_model {
        project_clip_frames_with_model(
            teacher,
            clip_frames,
            steps,
            projection_dim,
        )
        .detach()
    } else {
        project_clip_frames_with_model(
            model,
            clip_frames,
            steps,
            projection_dim,
        )
        .detach()
    };
    let context_target_proj = projected.clone().slice_dim(1, 0..context_len);
    let target_proj_all = projected.slice_dim(1, context_len..context_len + future_len_all);
    (
        context_observation_patch_tokens,
        context_target_proj,
        target_proj_all,
    )
}

pub(crate) fn split_clip_observation_and_target_projections_train<B: BackendTrait>(
    model: &VisionDragon<B>,
    teacher_model: Option<&VisionDragon<B>>,
    clip_frames: Tensor<B, 5>,
    context_len: usize,
    future_len_all: usize,
    steps: usize,
    projection_dim: usize,
    include_context_target_proj: bool,
) -> (Tensor<B, 4>, Option<Tensor<B, 3>>, Tensor<B, 3>) {
    if include_context_target_proj {
        let (
            context_observation_patch_tokens,
            context_target_proj,
            target_proj_all,
        ) = split_clip_observation_and_target_projections(
            model,
            teacher_model,
            clip_frames,
            context_len,
            future_len_all,
            steps,
            projection_dim,
        );
        return (
            context_observation_patch_tokens,
            Some(context_target_proj),
            target_proj_all,
        );
    }

    let context_clip_frames = clip_frames.clone().slice_dim(1, 0..context_len);
    let context_observation_patch_tokens =
        embed_clip_frames_raw_with_model(model, context_clip_frames);
    let target_clip_frames = clip_frames
        .slice_dim(1, context_len..context_len + future_len_all);
    let target_proj_all = if let Some(teacher) = teacher_model {
        project_clip_frames_with_model(teacher, target_clip_frames, steps, projection_dim).detach()
    } else {
        project_clip_frames_with_model(model, target_clip_frames, steps, projection_dim).detach()
    };
    (context_observation_patch_tokens, None, target_proj_all)
}

pub(crate) fn repeat_last_future_query<B: BackendTrait>(
    future_queries: Tensor<B, 2>,
    future_len: usize,
) -> Tensor<B, 2> {
    let [base_len, embed_dim] = future_queries.shape().dims::<2>();
    let future_len = future_len.max(1);
    if future_len <= base_len {
        return future_queries.slice_dim(0, 0..future_len);
    }
    let last = future_queries
        .clone()
        .slice_dim(0, (base_len.saturating_sub(1))..base_len)
        .reshape([1, embed_dim])
        .repeat_dim(0, future_len - base_len);
    Tensor::cat(vec![future_queries, last], 0)
}

pub(crate) type VisionVideoRolloutOutput<B> = VisionDragonOutput<B>;
