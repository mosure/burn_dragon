use crate::config::{AutoGazeConfig, ConnectorConfig, GazeModelConfig, VisionModelConfig};
use crate::{FixationPoint, FixationSet, FrameFixationTrace};
use anyhow::{Context, Result, bail};
use burn::module::{Ignored, Module, Param};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{
    Embedding, EmbeddingConfig, LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig3d,
};
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::module::interpolate;
use burn::tensor::ops::{InterpolateMode, InterpolateOptions, PadMode};
use burn::tensor::{Int, Tensor, TensorData};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::path::Path;

#[derive(Module, Debug)]
pub struct Conv3dBlockForStreaming<B: Backend> {
    pub conv3d: Conv3d<B>,
    #[module(ignore)]
    temporal_patch_size: usize,
}

impl<B: Backend> Conv3dBlockForStreaming<B> {
    pub fn new(
        hidden_dim: usize,
        temporal_patch_size: usize,
        spatial_kernel_size: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            conv3d: Conv3dConfig::new(
                [hidden_dim.max(1), hidden_dim.max(1)],
                [
                    temporal_patch_size.max(1),
                    spatial_kernel_size.max(1),
                    spatial_kernel_size.max(1),
                ],
            )
            .with_padding(PaddingConfig3d::Explicit(
                0,
                spatial_kernel_size.saturating_sub(1) / 2,
                spatial_kernel_size.saturating_sub(1) / 2,
            ))
            .init(device),
            temporal_patch_size: temporal_patch_size.max(1),
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Tensor<B, 5>>,
    ) -> (Tensor<B, 5>, Tensor<B, 5>) {
        let x = if use_cache {
            if let Some(past) = past_conv_values {
                Tensor::cat(vec![past, x], 2)
            } else {
                x.pad(
                    [
                        (0, 0),
                        (0, 0),
                        (self.temporal_patch_size.saturating_sub(1), 0),
                        (0, 0),
                        (0, 0),
                    ],
                    PadMode::Constant(0.0),
                )
            }
        } else {
            x.pad(
                [
                    (0, 0),
                    (0, 0),
                    (self.temporal_patch_size.saturating_sub(1), 0),
                    (0, 0),
                    (0, 0),
                ],
                PadMode::Constant(0.0),
            )
        };
        let time = x.shape().dims::<5>()[2];
        let keep = self.temporal_patch_size.saturating_sub(1);
        let new_past_conv_values = if keep == 0 {
            Tensor::<B, 5>::zeros(
                [
                    x.shape().dims::<5>()[0],
                    x.shape().dims::<5>()[1],
                    0,
                    x.shape().dims::<5>()[3],
                    x.shape().dims::<5>()[4],
                ],
                &x.device(),
            )
        } else {
            x.clone().slice_dim(2, time.saturating_sub(keep)..time)
        };

        let x = self.conv3d.forward(x);
        (activation::relu(x), new_past_conv_values)
    }
}

#[derive(Module, Debug)]
pub struct ShallowVideoConvNet<B: Backend> {
    pub temporal_conv: Conv3d<B>,
    pub norm: LayerNorm<B>,
    pub blocks: Vec<Conv3dBlockForStreaming<B>>,
    pub out_proj: Conv3d<B>,
    #[module(ignore)]
    temporal_patch_size: usize,
}

impl<B: Backend> ShallowVideoConvNet<B> {
    pub fn new(config: &VisionModelConfig, device: &B::Device) -> Self {
        let hidden_dim = config.hidden_dim.max(1);
        let out_dim = config.out_dim.max(1);
        let temporal_patch_size = config.temporal_patch_size.max(1);
        let temporal_conv = Conv3dConfig::new(
            [3, hidden_dim],
            [
                temporal_patch_size,
                config.kernel_size.max(1),
                config.kernel_size.max(1),
            ],
        )
        .with_stride([
            temporal_patch_size,
            config.kernel_size.max(1),
            config.kernel_size.max(1),
        ])
        .init(device);
        let norm = LayerNormConfig::new(hidden_dim).init(device);
        let blocks = (0..config.depth.max(1))
            .map(|_| {
                Conv3dBlockForStreaming::new(
                    hidden_dim,
                    config.trunk_temporal_kernel_size.max(1),
                    config.trunk_spatial_kernel_size.max(1),
                    device,
                )
            })
            .collect();
        let out_proj = Conv3dConfig::new([hidden_dim, out_dim], [1, 1, 1]).init(device);
        Self {
            temporal_conv,
            norm,
            blocks,
            out_proj,
            temporal_patch_size,
        }
    }

    pub fn forward(
        &self,
        x: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Vec<Tensor<B, 5>>>,
    ) -> (Tensor<B, 5>, Vec<Tensor<B, 5>>) {
        let mut x = x.permute([0, 2, 1, 3, 4]);
        x = self.temporal_conv.forward(x);
        let [batch, channels, time, height, width] = x.shape().dims::<5>();
        let x_flat = x
            .permute([0, 2, 1, 3, 4])
            .reshape([batch * time, channels, height * width])
            .swap_dims(1, 2);
        let x_flat = self.norm.forward(x_flat);
        let mut x = x_flat
            .swap_dims(1, 2)
            .reshape([batch, time, channels, height, width])
            .permute([0, 2, 1, 3, 4]);

        let mut new_past = Vec::with_capacity(self.blocks.len());
        for (index, block) in self.blocks.iter().enumerate() {
            let past = past_conv_values
                .as_ref()
                .and_then(|values| values.get(index))
                .cloned();
            let (next_x, next_past) = block.forward(x, use_cache, past);
            x = next_x;
            new_past.push(next_past);
        }
        let x = self.out_proj.forward(x);
        (x, new_past)
    }
}

#[derive(Module, Debug)]
pub struct Connector<B: Backend> {
    pub pos_embed: Param<Tensor<B, 2>>,
}

impl<B: Backend> Connector<B> {
    pub fn new(config: &ConnectorConfig, device: &B::Device) -> Self {
        Self {
            pos_embed: Param::from_tensor(Tensor::<B, 2>::random(
                [config.num_tokens.max(1), config.hidden_dim.max(1)],
                burn::tensor::Distribution::Normal(0.0, 1.0),
                device,
            )),
        }
    }

    pub fn forward(&self, x: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch, time, tokens, dim] = x.shape().dims::<4>();
        let pos = self.pos_embed.val().reshape([1, 1, tokens, dim]);
        x + pos.repeat_dim(0, batch).repeat_dim(1, time)
    }
}

#[derive(Clone, Debug)]
pub struct AutoGazeGenerateOutput {
    pub gazing_pos: Vec<Vec<i64>>,
    pub num_gazing_each_frame: Vec<usize>,
    pub if_padded_gazing: Vec<Vec<bool>>,
    pub confidences: Vec<Vec<f32>>,
}

#[derive(Debug)]
pub struct AutoGazeCausalLmOutput<B: Backend> {
    pub logits: Tensor<B, 3>,
    pub task_loss_prediction: Tensor<B, 3>,
    pub hidden_states: Tensor<B, 3>,
}

#[derive(Module, Debug)]
pub struct LlamaRmsNorm<B: Backend> {
    pub weight: Param<Tensor<B, 1>>,
    #[module(ignore)]
    eps: f32,
}

impl<B: Backend> LlamaRmsNorm<B> {
    pub fn new(width: usize, eps: f32, device: &B::Device) -> Self {
        Self {
            weight: Param::from_tensor(Tensor::<B, 1>::ones([width.max(1)], device)),
            eps: eps.max(1.0e-8),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let [_, _, width] = x.shape().dims::<3>();
        let (var, mean) = x.clone().var_mean_bias(2);
        let rms = var.add(mean.powf_scalar(2.0)).add_scalar(self.eps).sqrt();
        let weight = self.weight.val().reshape([1, 1, width]);
        x.div(rms).mul(weight)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    #[module(ignore)]
    num_heads: usize,
    #[module(ignore)]
    num_key_value_heads: usize,
    #[module(ignore)]
    head_dim: usize,
    #[module(ignore)]
    inv_freq: Tensor<B, 1>,
}

impl<B: Backend> AutoGazeLlamaAttention<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        let num_heads = config.num_attention_heads.max(1);
        let num_key_value_heads = config.num_key_value_heads.max(1);
        let head_dim = config.head_dim.max(1);
        let inv_freq = llama3_inv_freq(config, device);
        Self {
            q_proj: LinearConfig::new(config.hidden_size.max(1), num_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            k_proj: LinearConfig::new(config.hidden_size.max(1), num_key_value_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            v_proj: LinearConfig::new(config.hidden_size.max(1), num_key_value_heads * head_dim)
                .with_bias(config.attention_bias)
                .init(device),
            o_proj: LinearConfig::new(num_heads * head_dim, config.hidden_size.max(1))
                .with_bias(config.attention_bias)
                .init(device),
            num_heads,
            num_key_value_heads,
            head_dim,
            inv_freq,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let [batch, seq, _] = hidden_states.shape().dims::<3>();
        let q = split_heads(
            self.q_proj.forward(hidden_states.clone()),
            self.num_heads,
            self.head_dim,
        );
        let k = split_heads(
            self.k_proj.forward(hidden_states.clone()),
            self.num_key_value_heads,
            self.head_dim,
        );
        let v = split_heads(
            self.v_proj.forward(hidden_states),
            self.num_key_value_heads,
            self.head_dim,
        );

        let q = self.apply_rope(q, position_ids.clone());
        let mut k = self.apply_rope(k, position_ids);
        let mut v = v;
        if self.num_key_value_heads != self.num_heads {
            let repeat = (self.num_heads / self.num_key_value_heads.max(1)).max(1);
            k = k.repeat_dim(1, repeat);
            v = v.repeat_dim(1, repeat);
        }

        let scores = q
            .matmul(k.swap_dims(2, 3))
            .div_scalar((self.head_dim as f32).sqrt().max(1.0));
        let bias = causal_attention_bias(batch, seq, attention_mask, &scores.device());
        let attn = activation::softmax(scores + bias, 3);
        let out = attn.matmul(v);
        self.o_proj.forward(merge_heads(out))
    }

    fn apply_rope(&self, x: Tensor<B, 4>, position_ids: Tensor<B, 2, Int>) -> Tensor<B, 4> {
        let [batch, heads, seq, dim] = x.shape().dims::<4>();
        let half = dim / 2;
        let pos = position_ids.float().reshape([batch, 1, seq, 1]);
        let inv = self.inv_freq.clone().reshape([1, 1, 1, half]);
        let freqs = pos.mul(inv);
        let phases = Tensor::cat(vec![freqs.clone(), freqs], 3);
        let cos = phases.clone().cos().repeat_dim(1, heads);
        let sin = phases.sin().repeat_dim(1, heads);
        x.clone().mul(cos) + rotate_half(x).mul(sin)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaMlp<B: Backend> {
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> AutoGazeLlamaMlp<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            gate_proj: LinearConfig::new(
                config.hidden_size.max(1),
                config.intermediate_size.max(1),
            )
            .with_bias(config.mlp_bias)
            .init(device),
            up_proj: LinearConfig::new(config.hidden_size.max(1), config.intermediate_size.max(1))
                .with_bias(config.mlp_bias)
                .init(device),
            down_proj: LinearConfig::new(
                config.intermediate_size.max(1),
                config.hidden_size.max(1),
            )
            .with_bias(config.mlp_bias)
            .init(device),
        }
    }

    pub fn forward(&self, hidden_states: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = activation::silu(self.gate_proj.forward(hidden_states.clone()));
        let up = self.up_proj.forward(hidden_states);
        self.down_proj.forward(gate.mul(up))
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaDecoderLayer<B: Backend> {
    pub self_attn: AutoGazeLlamaAttention<B>,
    pub mlp: AutoGazeLlamaMlp<B>,
    pub input_layernorm: LlamaRmsNorm<B>,
    pub post_attention_layernorm: LlamaRmsNorm<B>,
}

impl<B: Backend> AutoGazeLlamaDecoderLayer<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            self_attn: AutoGazeLlamaAttention::new(config, device),
            mlp: AutoGazeLlamaMlp::new(config, device),
            input_layernorm: LlamaRmsNorm::new(
                config.hidden_size.max(1),
                config.rms_norm_eps,
                device,
            ),
            post_attention_layernorm: LlamaRmsNorm::new(
                config.hidden_size.max(1),
                config.rms_norm_eps,
                device,
            ),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let attn = self.self_attn.forward(
            self.input_layernorm.forward(hidden_states.clone()),
            attention_mask.clone(),
            position_ids.clone(),
        );
        let hidden_states = hidden_states + attn;
        let mlp = self
            .mlp
            .forward(self.post_attention_layernorm.forward(hidden_states.clone()));
        hidden_states + mlp
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaModel<B: Backend> {
    pub embed_tokens: Embedding<B>,
    pub layers: Vec<AutoGazeLlamaDecoderLayer<B>>,
    pub norm: LlamaRmsNorm<B>,
}

impl<B: Backend> AutoGazeLlamaModel<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        let layers = (0..config.num_hidden_layers.max(1))
            .map(|_| AutoGazeLlamaDecoderLayer::new(config, device))
            .collect();
        Self {
            embed_tokens: EmbeddingConfig::new(config.vocab_size.max(1), config.hidden_size.max(1))
                .init(device),
            layers,
            norm: LlamaRmsNorm::new(config.hidden_size.max(1), config.rms_norm_eps, device),
        }
    }

    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> Tensor<B, 3> {
        let mut hidden_states = inputs_embeds;
        for layer in self.layers.iter() {
            hidden_states =
                layer.forward(hidden_states, attention_mask.clone(), position_ids.clone());
        }
        self.norm.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeLlamaForCausalLmMultiTokenPred<B: Backend> {
    pub model: AutoGazeLlamaModel<B>,
    pub lm_head: Linear<B>,
    pub task_loss_prediction_head: Linear<B>,
    #[module(ignore)]
    vocab_size: usize,
    #[module(ignore)]
    num_multi_token_pred: usize,
}

impl<B: Backend> AutoGazeLlamaForCausalLmMultiTokenPred<B> {
    pub fn new(config: &crate::config::GazeDecoderConfig, device: &B::Device) -> Self {
        Self {
            model: AutoGazeLlamaModel::new(config, device),
            lm_head: LinearConfig::new(
                config.hidden_size.max(1),
                config.vocab_size.max(1) * config.num_multi_token_pred.max(1),
            )
            .with_bias(false)
            .init(device),
            task_loss_prediction_head: LinearConfig::new(
                config.hidden_size.max(1),
                config.num_multi_token_pred.max(1),
            )
            .with_bias(false)
            .init(device),
            vocab_size: config.vocab_size.max(1),
            num_multi_token_pred: config.num_multi_token_pred.max(1),
        }
    }

    pub fn forward(
        &self,
        inputs_embeds: Tensor<B, 3>,
        attention_mask: Option<Tensor<B, 2, Int>>,
        position_ids: Tensor<B, 2, Int>,
    ) -> AutoGazeCausalLmOutput<B> {
        let hidden_states = self
            .model
            .forward(inputs_embeds, attention_mask, position_ids);
        let logits = self.lm_head.forward(hidden_states.clone());
        let task_loss_prediction = self
            .task_loss_prediction_head
            .forward(hidden_states.clone());
        AutoGazeCausalLmOutput {
            logits,
            task_loss_prediction,
            hidden_states,
        }
    }
}

#[derive(Module, Debug)]
pub struct AutoGazeGazingModel<B: Backend> {
    pub vision_model: ShallowVideoConvNet<B>,
    pub connector: Connector<B>,
    pub gaze_decoder: AutoGazeLlamaForCausalLmMultiTokenPred<B>,
    #[module(ignore)]
    input_img_size: usize,
    #[module(ignore)]
    num_vision_tokens_each_frame: usize,
    #[module(ignore)]
    frame_sampling_rate: usize,
    #[module(ignore)]
    num_multi_token_pred: usize,
    #[module(ignore)]
    eos_token_id: i64,
}

impl<B: Backend> AutoGazeGazingModel<B> {
    pub fn new(config: &GazeModelConfig, device: &B::Device) -> Self {
        Self {
            vision_model: ShallowVideoConvNet::new(&config.vision_model_config, device),
            connector: Connector::new(&config.connector_config, device),
            gaze_decoder: AutoGazeLlamaForCausalLmMultiTokenPred::new(
                &config.gaze_decoder_config,
                device,
            ),
            input_img_size: config.input_img_size.max(1),
            num_vision_tokens_each_frame: config.num_vision_tokens_each_frame.max(1),
            frame_sampling_rate: config.vision_model_config.temporal_patch_size.max(1),
            num_multi_token_pred: config.gaze_decoder_config.num_multi_token_pred.max(1),
            eos_token_id: config.gaze_decoder_config.eos_token_id,
        }
    }

    pub fn embed_video(
        &self,
        video: Tensor<B, 5>,
        use_cache: bool,
        past_conv_values: Option<Vec<Tensor<B, 5>>>,
    ) -> (Tensor<B, 4>, Vec<Tensor<B, 5>>) {
        let [_batch, _time, channels, height, width] = video.shape().dims::<5>();
        let device = video.device();
        let video = adapt_video_channels(video, channels, &device);
        let video = if height != width {
            panic!("AutoGaze Burn port currently expects square frames");
        } else {
            video
        };
        let (vision_features, new_past) =
            self.vision_model
                .forward(video, use_cache, past_conv_values);
        let vision_features = vision_features.swap_dims(1, 2).permute([0, 1, 3, 4, 2]);
        let [batch, time, height, width, dim] = vision_features.shape().dims::<5>();
        let vision_features = vision_features.reshape([batch, time, height * width, dim]);
        (self.connector.forward(vision_features), new_past)
    }

    pub fn generate(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
    ) -> AutoGazeGenerateOutput {
        let video = self.resize_video(video);
        let (video_embeds, _) = self.embed_video(video, false, None);
        let [batch, frames, vision_tokens, dim] = video_embeds.shape().dims::<4>();
        let device = video_embeds.device();

        let mut gazing_pos = vec![Vec::<i64>::new(); batch];
        let mut if_padded_gazing = vec![Vec::<bool>::new(); batch];
        let mut confidences = vec![Vec::<f32>::new(); batch];
        let mut num_gazing_each_frame = Vec::with_capacity(frames);

        let mut prefix_embeds: Option<Tensor<B, 3>> = None;
        let mut prefix_attention_mask = vec![vec![]; batch];

        for frame_idx in 0..frames {
            let frame_embed = video_embeds
                .clone()
                .slice_dim(1, frame_idx..(frame_idx + 1))
                .reshape([batch, vision_tokens, dim]);
            let mut sequence_embeds = if let Some(prefix) = prefix_embeds.take() {
                Tensor::cat(vec![prefix, frame_embed.clone()], 1)
            } else {
                frame_embed.clone()
            };
            for row in prefix_attention_mask.iter_mut() {
                row.extend(std::iter::repeat_n(1, vision_tokens));
            }

            let mut frame_tokens = vec![Vec::<i64>::new(); batch];
            let mut frame_padded = vec![Vec::<bool>::new(); batch];
            let mut frame_confidences = vec![Vec::<f32>::new(); batch];
            let mut finished = vec![false; batch];
            let max_tokens = max_gaze_tokens_each_frame.max(1);

            while frame_tokens.iter().map(Vec::len).max().unwrap_or(0) < max_tokens
                && finished.iter().any(|done| !done)
            {
                let seq_len = sequence_embeds.shape().dims::<3>()[1];
                let attention_mask =
                    attention_mask_tensor::<B>(&prefix_attention_mask, seq_len, &device);
                let position_ids =
                    position_ids_from_attention_mask::<B>(&prefix_attention_mask, seq_len, &device);
                let outputs = self.gaze_decoder.forward(
                    sequence_embeds.clone(),
                    Some(attention_mask),
                    position_ids,
                );
                let last_logits = outputs
                    .logits
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([
                        batch,
                        self.num_multi_token_pred,
                        self.gaze_decoder.vocab_size,
                    ]);
                let last_task = outputs
                    .task_loss_prediction
                    .slice_dim(1, seq_len.saturating_sub(1)..seq_len)
                    .reshape([batch, self.num_multi_token_pred]);
                let (next_tokens, next_valid, next_confidences) = greedy_select_multi_tokens(
                    last_logits,
                    last_task,
                    &frame_tokens,
                    &finished,
                    self.eos_token_id,
                    max_tokens,
                );

                let new_tokens = next_tokens.first().map(Vec::len).unwrap_or(0);
                if new_tokens == 0 {
                    break;
                }
                let flat_tokens: Vec<i64> = next_tokens
                    .iter()
                    .flat_map(|tokens| tokens.iter().copied())
                    .collect();
                let token_tensor = Tensor::<B, 2, Int>::from_data(
                    TensorData::new(flat_tokens, [batch, new_tokens]),
                    &device,
                );
                let token_embeds = self.gaze_decoder.model.embed_tokens.forward(token_tensor);
                sequence_embeds = Tensor::cat(vec![sequence_embeds, token_embeds], 1);

                for batch_idx in 0..batch {
                    for local_idx in 0..new_tokens {
                        let token = next_tokens[batch_idx][local_idx];
                        let valid = next_valid[batch_idx][local_idx];
                        let confidence = next_confidences[batch_idx][local_idx];
                        frame_tokens[batch_idx].push(token);
                        frame_padded[batch_idx].push(!valid);
                        frame_confidences[batch_idx].push(confidence);
                        prefix_attention_mask[batch_idx].push(if valid { 1 } else { 0 });
                        if !valid {
                            finished[batch_idx] = true;
                        }
                    }
                }
            }

            let frame_count = frame_tokens.first().map(Vec::len).unwrap_or(0);
            num_gazing_each_frame.push(frame_count);
            let frame_offset = (frame_idx * self.num_vision_tokens_each_frame) as i64;
            for batch_idx in 0..batch {
                gazing_pos[batch_idx].extend(
                    frame_tokens[batch_idx]
                        .iter()
                        .map(|token| token + frame_offset),
                );
                if_padded_gazing[batch_idx].extend(frame_padded[batch_idx].iter().copied());
                confidences[batch_idx].extend(frame_confidences[batch_idx].iter().copied());
            }
            prefix_embeds = Some(sequence_embeds);
        }

        AutoGazeGenerateOutput {
            gazing_pos,
            num_gazing_each_frame,
            if_padded_gazing,
            confidences,
        }
    }

    fn resize_video(&self, video: Tensor<B, 5>) -> Tensor<B, 5> {
        let [batch, time, channels, height, width] = video.shape().dims::<5>();
        let device = video.device();
        let video = adapt_video_channels(video, channels, &device);
        if height == self.input_img_size && width == self.input_img_size {
            return video;
        }
        let video = video.reshape([batch * time, 3, height, width]);
        let video = interpolate(
            video,
            [self.input_img_size, self.input_img_size],
            InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
        );
        video.reshape([batch, time, 3, self.input_img_size, self.input_img_size])
    }
}

#[derive(Module, Debug)]
pub struct NativeAutoGazeModel<B: Backend> {
    pub gazing_model: AutoGazeGazingModel<B>,
    #[module(ignore)]
    pub config: Ignored<AutoGazeConfig>,
}

impl<B: Backend> NativeAutoGazeModel<B> {
    pub fn new(config: &AutoGazeConfig, device: &B::Device) -> Self {
        Self {
            gazing_model: AutoGazeGazingModel::new(&config.gaze_model_config, device),
            config: Ignored(config.clone()),
        }
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let dir = dir.as_ref();
        let config = AutoGazeConfig::from_json_file(dir.join("config.json"))?;
        let mut model = Self::new(&config, device);
        let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(false)
            .validate(true);
        let result = model
            .load_from(&mut store)
            .with_context(|| format!("load AutoGaze weights from {}", dir.display()))?;
        if !result.errors.is_empty() {
            bail!("failed to apply AutoGaze weights: {:?}", result.errors);
        }
        Ok(model)
    }

    pub fn generate(
        &self,
        video: Tensor<B, 5>,
        max_gaze_tokens_each_frame: usize,
    ) -> AutoGazeGenerateOutput {
        self.gazing_model
            .generate(video, max_gaze_tokens_each_frame)
    }

    pub fn default_max_gaze_tokens_each_frame(&self) -> usize {
        self.gazing_model.num_multi_token_pred.max(1)
    }

    pub fn trace_video(
        &self,
        video: Tensor<B, 5>,
        k: usize,
        max_gaze_tokens_each_frame: usize,
    ) -> Vec<FrameFixationTrace> {
        let generated = self.generate(video, max_gaze_tokens_each_frame.max(k.max(1)));
        generated_to_traces(&generated, &self.config.0, k)
    }

    pub fn trace_clip_from_frames(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace {
        let device = self.gazing_model.connector.pos_embed.val().device();
        let clip = Tensor::<B, 5>::from_data(
            TensorData::new(
                frames.to_vec(),
                [
                    1,
                    clip_len.max(1),
                    channels.max(1),
                    height.max(1),
                    width.max(1),
                ],
            ),
            &device,
        );
        self.trace_video(
            clip,
            k,
            self.gazing_model.num_multi_token_pred.max(k.max(1)),
        )
        .into_iter()
        .next()
        .unwrap_or_else(|| FrameFixationTrace::new(vec![]))
    }
}

fn adapt_video_channels<B: Backend>(
    video: Tensor<B, 5>,
    channels: usize,
    device: &B::Device,
) -> Tensor<B, 5> {
    match channels {
        3 => video,
        1 => video.repeat_dim(2, 3),
        other => {
            let [batch, time, _, height, width] = video.shape().dims::<5>();
            let data = vec![0.0f32; batch * time * 3 * height * width];
            let mut out = Tensor::<B, 5>::from_data(
                TensorData::new(data, [batch, time, 3, height, width]),
                device,
            );
            let keep = other.min(3);
            out = out.slice_assign(
                [0..batch, 0..time, 0..keep, 0..height, 0..width],
                video.slice_dim(2, 0..keep),
            );
            out
        }
    }
}

impl<B: Backend> crate::AutoGazeTeacher for NativeAutoGazeModel<B> {
    fn trace_clip(
        &self,
        frames: &[f32],
        clip_len: usize,
        channels: usize,
        height: usize,
        width: usize,
        k: usize,
    ) -> FrameFixationTrace {
        self.trace_clip_from_frames(frames, clip_len, channels, height, width, k)
    }
}

fn split_heads<B: Backend>(tokens: Tensor<B, 3>, heads: usize, head_dim: usize) -> Tensor<B, 4> {
    let [batch, seq, _] = tokens.shape().dims::<3>();
    tokens
        .reshape([batch, seq, heads.max(1), head_dim.max(1)])
        .swap_dims(1, 2)
}

fn merge_heads<B: Backend>(tokens: Tensor<B, 4>) -> Tensor<B, 3> {
    let [batch, heads, seq, head_dim] = tokens.shape().dims::<4>();
    tokens
        .swap_dims(1, 2)
        .reshape([batch, seq, heads * head_dim])
}

fn rotate_half<B: Backend>(x: Tensor<B, 4>) -> Tensor<B, 4> {
    let dim = x.shape().dims::<4>()[3];
    let half = dim / 2;
    let x1 = x.clone().slice_dim(3, 0..half);
    let x2 = x.slice_dim(3, half..dim);
    Tensor::cat(vec![x2.mul_scalar(-1.0), x1], 3)
}

fn causal_attention_bias<B: Backend>(
    batch: usize,
    seq_len: usize,
    attention_mask: Option<Tensor<B, 2, Int>>,
    device: &B::Device,
) -> Tensor<B, 4> {
    let q_pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).reshape([1, 1, seq_len, 1]);
    let k_pos = Tensor::<B, 1, Int>::arange(0..seq_len as i64, device).reshape([1, 1, 1, seq_len]);
    let causal = k_pos.lower_equal(q_pos).float();
    let mut bias = causal.sub_scalar(1.0).abs().mul_scalar(-1.0e9);
    if let Some(mask) = attention_mask {
        let key_valid = mask.float().reshape([batch.max(1), 1, 1, seq_len]);
        let key_bias = key_valid.sub_scalar(1.0).abs().mul_scalar(-1.0e9);
        bias = bias + key_bias;
    }
    bias
}

fn llama3_inv_freq<B: Backend>(
    config: &crate::config::GazeDecoderConfig,
    device: &B::Device,
) -> Tensor<B, 1> {
    let head_dim = config.head_dim.max(1);
    let half = (head_dim / 2).max(1);
    let base = config.rope_theta.max(1.0);
    let mut inv_freq = Vec::with_capacity(half);
    for idx in 0..half {
        let dim_index = (idx * 2) as f32;
        inv_freq.push(1.0 / base.powf(dim_index / head_dim as f32));
    }

    if let Some(rope_scaling) = config.rope_scaling.as_ref() {
        let rope_type = rope_scaling
            .get("rope_type")
            .and_then(|value| value.as_str())
            .unwrap_or("default");
        if rope_type == "llama3" {
            let factor = rope_scaling
                .get("factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32;
            let low_freq_factor = rope_scaling
                .get("low_freq_factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(1.0) as f32;
            let high_freq_factor = rope_scaling
                .get("high_freq_factor")
                .and_then(|value| value.as_f64())
                .unwrap_or(4.0) as f32;
            let original_max_position_embeddings = rope_scaling
                .get("original_max_position_embeddings")
                .and_then(|value| value.as_f64())
                .unwrap_or(config.max_position_embeddings as f64)
                as f32;

            let low_freq_wavelen = original_max_position_embeddings / low_freq_factor.max(1.0);
            let high_freq_wavelen =
                original_max_position_embeddings / high_freq_factor.max(low_freq_factor + 1.0e-6);
            for value in inv_freq.iter_mut() {
                let wavelen = 2.0 * std::f32::consts::PI / (*value).max(1.0e-12);
                let scaled = if wavelen > low_freq_wavelen {
                    *value / factor.max(1.0)
                } else {
                    *value
                };
                if wavelen >= high_freq_wavelen && wavelen <= low_freq_wavelen {
                    let smooth_factor = (original_max_position_embeddings / wavelen
                        - low_freq_factor)
                        / (high_freq_factor - low_freq_factor).max(1.0e-6);
                    *value =
                        (1.0 - smooth_factor) * scaled / factor.max(1.0) + smooth_factor * scaled;
                } else {
                    *value = scaled;
                }
            }
        }
    }

    Tensor::<B, 1>::from_data(TensorData::new(inv_freq, [half]), device)
}

fn attention_mask_tensor<B: Backend>(
    mask_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = mask_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in mask_rows {
        values.extend(row.iter().copied().take(seq_len));
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, seq_len]), device)
}

fn position_ids_from_attention_mask<B: Backend>(
    mask_rows: &[Vec<i64>],
    seq_len: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let batch = mask_rows.len().max(1);
    let mut values = Vec::with_capacity(batch * seq_len);
    for row in mask_rows {
        let mut position = 0i64;
        for mask in row.iter().copied().take(seq_len) {
            values.push(position);
            if mask != 0 {
                position += 1;
            }
        }
    }
    Tensor::<B, 2, Int>::from_data(TensorData::new(values, [batch, seq_len]), device)
}

fn greedy_select_multi_tokens<B: Backend>(
    logits: Tensor<B, 3>,
    _task_loss: Tensor<B, 2>,
    prior_tokens: &[Vec<i64>],
    finished: &[bool],
    eos_token_id: i64,
    max_tokens: usize,
) -> (Vec<Vec<i64>>, Vec<Vec<bool>>, Vec<Vec<f32>>) {
    let [batch, num_multi, vocab] = logits.shape().dims::<3>();
    let scores = logits
        .into_data()
        .to_vec::<f32>()
        .expect("convert logits to f32 vec");
    let mut per_batch_tokens = vec![Vec::new(); batch];
    let mut per_batch_valid = vec![Vec::new(); batch];
    let mut per_batch_confidences = vec![Vec::new(); batch];

    for batch_idx in 0..batch {
        let mut disallowed = prior_tokens[batch_idx].clone();
        for multi_idx in 0..num_multi {
            if finished.get(batch_idx).copied().unwrap_or(false)
                || prior_tokens[batch_idx].len() + per_batch_tokens[batch_idx].len() >= max_tokens
            {
                break;
            }

            let base = (batch_idx * num_multi + multi_idx) * vocab;
            let mut best_index = None;
            let mut best_score = f32::NEG_INFINITY;
            let mut exp_sum = 0.0f32;
            for vocab_idx in 0..vocab {
                if vocab_idx as i64 == eos_token_id || disallowed.contains(&(vocab_idx as i64)) {
                    continue;
                }
                let score = scores[base + vocab_idx];
                if score.is_finite() {
                    exp_sum += score.exp();
                    if score > best_score {
                        best_score = score;
                        best_index = Some(vocab_idx as i64);
                    }
                }
            }

            if let Some(token) = best_index {
                disallowed.push(token);
                per_batch_tokens[batch_idx].push(token);
                per_batch_valid[batch_idx].push(true);
                per_batch_confidences[batch_idx].push(if exp_sum > 0.0 {
                    best_score.exp() / exp_sum
                } else {
                    1.0
                });
            } else {
                break;
            }
        }
    }

    let padded_len = per_batch_tokens.iter().map(Vec::len).max().unwrap_or(0);
    let mut out_tokens = Vec::with_capacity(batch);
    let mut out_valid = Vec::with_capacity(batch);
    let mut out_confidences = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let mut tokens = per_batch_tokens[batch_idx].clone();
        let mut valid = per_batch_valid[batch_idx].clone();
        let mut confidences = per_batch_confidences[batch_idx].clone();
        while tokens.len() < padded_len {
            tokens.push(eos_token_id);
            valid.push(false);
            confidences.push(0.0);
        }
        out_tokens.push(tokens);
        out_valid.push(valid);
        out_confidences.push(confidences);
    }

    (out_tokens, out_valid, out_confidences)
}

fn generated_to_traces(
    generated: &AutoGazeGenerateOutput,
    config: &AutoGazeConfig,
    k: usize,
) -> Vec<FrameFixationTrace> {
    let tokens_each_scale = tokens_each_scale(config);
    let grids: Vec<usize> = tokens_each_scale
        .iter()
        .map(|tokens| (*tokens as f64).sqrt().round() as usize)
        .collect();
    let scales = config.scale_values();
    let mut traces = Vec::with_capacity(generated.gazing_pos.len());
    for batch_idx in 0..generated.gazing_pos.len() {
        let mut cursor = 0usize;
        let mut frames = Vec::with_capacity(generated.num_gazing_each_frame.len());
        for (frame_idx, frame_len) in generated.num_gazing_each_frame.iter().copied().enumerate() {
            let mut points = Vec::new();
            let mut stop_probability = 0.0f32;
            for local_idx in 0..frame_len {
                let global_idx = cursor + local_idx;
                let token = generated.gazing_pos[batch_idx][global_idx]
                    - (frame_idx * config.num_vision_tokens_each_frame) as i64;
                let padded = generated.if_padded_gazing[batch_idx][global_idx];
                if padded {
                    stop_probability = 1.0;
                    continue;
                }
                if let Some(point) = token_to_fixation_point(
                    token.max(0) as usize,
                    &tokens_each_scale,
                    &grids,
                    &scales,
                    generated.confidences[batch_idx][global_idx],
                ) {
                    points.push(point);
                }
            }
            cursor += frame_len;
            frames.push(FixationSet::new(points, stop_probability, k));
        }
        traces.push(FrameFixationTrace::new(frames));
    }
    traces
}

fn token_to_fixation_point(
    token: usize,
    tokens_each_scale: &[usize],
    grids: &[usize],
    scales: &[usize],
    confidence: f32,
) -> Option<FixationPoint> {
    let mut offset = 0usize;
    let max_scale = (*scales.iter().max().unwrap_or(&224)).max(1) as f32;
    for (scale_idx, token_count) in tokens_each_scale.iter().copied().enumerate() {
        if token < offset + token_count {
            let local = token - offset;
            let grid = grids.get(scale_idx).copied().unwrap_or(1).max(1);
            let row = local / grid;
            let col = local % grid;
            let x = (col as f32 + 0.5) / grid as f32;
            let y = (row as f32 + 0.5) / grid as f32;
            let patch =
                16.0 * (max_scale / scales.get(scale_idx).copied().unwrap_or(224).max(1) as f32);
            let scale = (patch / max_scale).clamp(0.01, 1.0);
            return Some(FixationPoint::new(x, y, scale, confidence));
        }
        offset += token_count;
    }
    None
}

fn tokens_each_scale(config: &AutoGazeConfig) -> Vec<usize> {
    let scales = config.scale_values();
    if scales.is_empty() {
        return vec![config.num_vision_tokens_each_frame.max(1)];
    }
    let sum_sq: usize = scales.iter().map(|scale| scale * scale).sum();
    let mut counts = Vec::with_capacity(scales.len());
    let mut assigned = 0usize;
    for (index, scale) in scales.iter().copied().enumerate() {
        if index + 1 == scales.len() {
            counts.push(config.num_vision_tokens_each_frame.saturating_sub(assigned));
        } else {
            let count = ((scale * scale) as f64 / sum_sq.max(1) as f64
                * config.num_vision_tokens_each_frame as f64)
                .floor() as usize;
            counts.push(count);
            assigned += count;
        }
    }
    counts
}
