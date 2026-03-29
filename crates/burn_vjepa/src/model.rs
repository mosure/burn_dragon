use crate::config::Vjepa2Config;
use anyhow::{Context, Result, bail};
use burn::module::{Ignored, Module, Param};
use burn::nn::conv::{Conv3d, Conv3dConfig};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::ElementConversion;
use burn::tensor::activation;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_store::{ModuleSnapshot, PyTorchToBurnAdapter, SafetensorsStore};
use std::path::Path;

#[derive(Debug)]
pub struct Vjepa2PredictorOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub target_hidden_state: Tensor<B, 3>,
}

#[derive(Debug)]
pub struct Vjepa2ModelOutput<B: Backend> {
    pub last_hidden_state: Tensor<B, 3>,
    pub masked_hidden_state: Tensor<B, 3>,
    pub predictor_output: Option<Vjepa2PredictorOutput<B>>,
}

#[derive(Module, Debug)]
pub struct Vjepa2PatchEmbeddings3d<B: Backend> {
    pub proj: Conv3d<B>,
    #[module(ignore)]
    patch_size: usize,
    #[module(ignore)]
    tubelet_size: usize,
    #[module(ignore)]
    hidden_size: usize,
}

impl<B: Backend> Vjepa2PatchEmbeddings3d<B> {
    pub fn new(config: &Vjepa2Config, device: &B::Device) -> Self {
        Self {
            proj: Conv3dConfig::new(
                [config.in_chans.max(1), config.hidden_size.max(1)],
                [
                    config.tubelet_size.max(1),
                    config.patch_size.max(1),
                    config.patch_size.max(1),
                ],
            )
            .with_stride([
                config.tubelet_size.max(1),
                config.patch_size.max(1),
                config.patch_size.max(1),
            ])
            .init(device),
            patch_size: config.patch_size.max(1),
            tubelet_size: config.tubelet_size.max(1),
            hidden_size: config.hidden_size.max(1),
        }
    }

    pub fn forward(&self, pixel_values_videos: Tensor<B, 5>) -> Tensor<B, 3> {
        let x = self.proj.forward(pixel_values_videos);
        let [batch, channels, depth, height, width] = x.shape().dims::<5>();
        x.reshape([batch, channels, depth * height * width])
            .swap_dims(1, 2)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Embeddings<B: Backend> {
    pub patch_embeddings: Vjepa2PatchEmbeddings3d<B>,
    #[module(ignore)]
    tubelet_size: usize,
}

impl<B: Backend> Vjepa2Embeddings<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        Self {
            patch_embeddings: Vjepa2PatchEmbeddings3d::new(&config, device),
            tubelet_size: config.tubelet_size.max(1),
        }
    }

    pub fn forward(&self, pixel_values_videos: Tensor<B, 5>) -> Tensor<B, 3> {
        let num_frames = pixel_values_videos.shape().dims::<5>()[1];
        let mut pixel_values_videos = pixel_values_videos.permute([0, 2, 1, 3, 4]);
        if num_frames < self.tubelet_size.max(1) {
            pixel_values_videos = pixel_values_videos.repeat_dim(2, self.tubelet_size.max(1));
        }
        self.patch_embeddings.forward(pixel_values_videos)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2RopeAttention<B: Backend> {
    pub query: Linear<B>,
    pub key: Linear<B>,
    pub value: Linear<B>,
    pub proj: Linear<B>,
    #[module(ignore)]
    hidden_size: usize,
    #[module(ignore)]
    num_attention_heads: usize,
    #[module(ignore)]
    attention_head_size: usize,
    #[module(ignore)]
    all_head_size: usize,
    #[module(ignore)]
    grid_size: usize,
    #[module(ignore)]
    d_dim: usize,
    #[module(ignore)]
    h_dim: usize,
    #[module(ignore)]
    w_dim: usize,
    #[module(ignore)]
    scaling: f32,
    #[module(ignore)]
    omega_cache: Tensor<B, 1>,
}

impl<B: Backend> Vjepa2RopeAttention<B> {
    pub fn new(
        config: &Vjepa2Config,
        hidden_size: usize,
        num_attention_heads: usize,
        device: &B::Device,
    ) -> Self {
        let hidden_size = hidden_size.max(1);
        let num_attention_heads = num_attention_heads.max(1);
        let attention_head_size = hidden_size / num_attention_heads;
        let all_head_size = num_attention_heads * attention_head_size;
        let grid_size = config.grid_size().max(1);
        let d_dim = 2 * ((attention_head_size / 3) / 2);
        let h_dim = 2 * ((attention_head_size / 3) / 2);
        let w_dim = 2 * ((attention_head_size / 3) / 2);
        let half = (attention_head_size / 2).max(1);
        let mut omega = Tensor::<B, 1, Int>::arange(0..half as i64, device).float();
        omega = omega.div_scalar(attention_head_size.max(2) as f32 / 2.0);
        omega = omega.mul_scalar(-10000.0f32.ln()).exp();

        Self {
            query: LinearConfig::new(hidden_size, all_head_size)
                .with_bias(config.qkv_bias)
                .init(device),
            key: LinearConfig::new(hidden_size, all_head_size)
                .with_bias(config.qkv_bias)
                .init(device),
            value: LinearConfig::new(hidden_size, all_head_size)
                .with_bias(config.qkv_bias)
                .init(device),
            proj: LinearConfig::new(hidden_size, hidden_size).init(device),
            hidden_size,
            num_attention_heads,
            attention_head_size,
            all_head_size,
            grid_size,
            d_dim,
            h_dim,
            w_dim,
            scaling: (attention_head_size as f32).powf(-0.5),
            omega_cache: omega,
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let [batch_size, seq_length, _] = hidden_states.shape().dims::<3>();
        let query_layer = self
            .query
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);
        let key_layer = self
            .key
            .forward(hidden_states.clone())
            .reshape([
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);
        let value_layer = self
            .value
            .forward(hidden_states)
            .reshape([
                batch_size,
                seq_length,
                self.num_attention_heads,
                self.attention_head_size,
            ])
            .swap_dims(1, 2);

        let pos_ids = self.get_position_ids(seq_length, position_mask, &query_layer.device());
        let key_layer = self.apply_rotary_embeddings(key_layer, &pos_ids);
        let query_layer = self.apply_rotary_embeddings(query_layer, &pos_ids);

        let attn_weights = activation::softmax(
            query_layer
                .matmul(key_layer.swap_dims(2, 3))
                .mul_scalar(self.scaling),
            3,
        );
        let context_layer = attn_weights.matmul(value_layer).swap_dims(1, 2).reshape([
            batch_size,
            seq_length,
            self.all_head_size,
        ]);
        self.proj.forward(context_layer)
    }

    fn get_position_ids(
        &self,
        token_size: usize,
        masks: Option<Tensor<B, 2, Int>>,
        device: &B::Device,
    ) -> (Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>) {
        let ids = if let Some(masks) = masks {
            masks
                .unsqueeze_dim::<3>(1)
                .repeat_dim(1, self.num_attention_heads)
                .float()
        } else {
            Tensor::<B, 1, Int>::arange(0..token_size as i64, device)
                .unsqueeze_dim::<2>(0)
                .unsqueeze_dim::<3>(0)
                .repeat_dim(1, self.num_attention_heads)
                .float()
        };
        let tokens_per_frame = (self.grid_size * self.grid_size) as f32;
        let tokens_per_row = self.grid_size as f32;
        let frame_ids = ids.clone().div_scalar(tokens_per_frame).floor();
        let local_ids = ids.clone() - frame_ids.clone().mul_scalar(tokens_per_frame);
        let height_ids = local_ids.clone().div_scalar(tokens_per_row).floor();
        let width_ids = local_ids - height_ids.clone().mul_scalar(tokens_per_row);
        (frame_ids, height_ids, width_ids)
    }

    fn apply_rotary_embeddings(
        &self,
        qk: Tensor<B, 4>,
        pos_ids: &(Tensor<B, 3>, Tensor<B, 3>, Tensor<B, 3>),
    ) -> Tensor<B, 4> {
        let (d_mask, h_mask, w_mask) = pos_ids;
        let mut start = 0usize;
        let mut parts = Vec::new();
        if self.d_dim > 0 {
            parts.push(self.rotate_queries_or_keys(
                qk.clone().slice_dim(3, start..start + self.d_dim),
                d_mask.clone(),
            ));
            start += self.d_dim;
        }
        if self.h_dim > 0 {
            parts.push(self.rotate_queries_or_keys(
                qk.clone().slice_dim(3, start..start + self.h_dim),
                h_mask.clone(),
            ));
            start += self.h_dim;
        }
        if self.w_dim > 0 {
            parts.push(self.rotate_queries_or_keys(
                qk.clone().slice_dim(3, start..start + self.w_dim),
                w_mask.clone(),
            ));
            start += self.w_dim;
        }
        if start < self.attention_head_size {
            parts.push(qk.slice_dim(3, start..self.attention_head_size));
        }
        Tensor::cat(parts, 3)
    }

    fn rotate_queries_or_keys(&self, x: Tensor<B, 4>, pos: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch, heads, tokens, dim] = x.shape().dims::<4>();
        if dim == 0 {
            return x;
        }
        let half = dim / 2;
        let freq = pos.unsqueeze_dim::<4>(3)
            * self
                .omega_cache
                .clone()
                .slice_dim(0, 0..half)
                .reshape([1, 1, 1, half.max(1)]);
        let emb_sin = freq.clone().sin().repeat_dim(3, 2);
        let emb_cos = freq.cos().repeat_dim(3, 2);

        let y = x.clone().reshape([batch, heads, tokens, half.max(1), 2]);
        let y1 = y
            .clone()
            .slice_dim(4, 0..1)
            .reshape([batch, heads, tokens, half.max(1)]);
        let y2 = y
            .slice_dim(4, 1..2)
            .reshape([batch, heads, tokens, half.max(1)]);
        let rotated = Tensor::cat(
            vec![
                y2.mul_scalar(-1.0).unsqueeze_dim::<5>(4),
                y1.unsqueeze_dim::<5>(4),
            ],
            4,
        )
        .reshape([batch, heads, tokens, half.max(1) * 2]);

        x.mul(emb_cos) + rotated.mul(emb_sin)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Mlp<B: Backend> {
    pub fc1: Linear<B>,
    pub fc2: Linear<B>,
    #[module(ignore)]
    hidden_act: Ignored<String>,
}

impl<B: Backend> Vjepa2Mlp<B> {
    pub fn new(
        config: &Vjepa2Config,
        hidden_size: usize,
        mlp_ratio: f32,
        device: &B::Device,
    ) -> Self {
        let hidden_features = ((hidden_size as f32) * mlp_ratio.max(1.0)).round() as usize;
        Self {
            fc1: LinearConfig::new(hidden_size, hidden_features.max(1)).init(device),
            fc2: LinearConfig::new(hidden_features.max(1), hidden_size).init(device),
            hidden_act: Ignored(config.hidden_act.clone()),
        }
    }

    pub fn forward(&self, hidden_state: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden_state = self.fc1.forward(hidden_state);
        let hidden_state = match self.hidden_act.0.as_str() {
            "relu" => activation::relu(hidden_state),
            "gelu" | "gelu_new" => activation::gelu(hidden_state),
            other => panic!("unsupported V-JEPA2 activation {other}"),
        };
        self.fc2.forward(hidden_state)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Layer<B: Backend> {
    pub norm1: LayerNorm<B>,
    pub attention: Vjepa2RopeAttention<B>,
    pub norm2: LayerNorm<B>,
    pub mlp: Vjepa2Mlp<B>,
}

impl<B: Backend> Vjepa2Layer<B> {
    pub fn new(
        config: &Vjepa2Config,
        hidden_size: usize,
        num_attention_heads: usize,
        mlp_ratio: f32,
        device: &B::Device,
    ) -> Self {
        Self {
            norm1: LayerNormConfig::new(hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            attention: Vjepa2RopeAttention::new(config, hidden_size, num_attention_heads, device),
            norm2: LayerNormConfig::new(hidden_size)
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            mlp: Vjepa2Mlp::new(config, hidden_size, mlp_ratio, device),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        position_mask: Option<Tensor<B, 2, Int>>,
    ) -> Tensor<B, 3> {
        let residual = hidden_states.clone();
        let hidden_states = self.norm1.forward(hidden_states);
        let attention_output = self.attention.forward(hidden_states, position_mask.clone());
        let hidden_states = attention_output + residual;
        let residual = hidden_states.clone();
        let hidden_states = self.norm2.forward(hidden_states);
        residual + self.mlp.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Encoder<B: Backend> {
    pub embeddings: Vjepa2Embeddings<B>,
    pub layer: Vec<Vjepa2Layer<B>>,
    pub layernorm: LayerNorm<B>,
}

impl<B: Backend> Vjepa2Encoder<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        let layer = (0..config.num_hidden_layers.max(1))
            .map(|_| {
                Vjepa2Layer::new(
                    &config,
                    config.hidden_size.max(1),
                    config.num_attention_heads.max(1),
                    config.mlp_ratio.max(1.0),
                    device,
                )
            })
            .collect();
        Self {
            embeddings: Vjepa2Embeddings::new(config.clone(), device),
            layer,
            layernorm: LayerNormConfig::new(config.hidden_size.max(1))
                .with_epsilon(config.layer_norm_eps)
                .init(device),
        }
    }

    pub fn forward(&self, pixel_values_videos: Tensor<B, 5>) -> Tensor<B, 3> {
        let mut hidden_states = self.embeddings.forward(pixel_values_videos);
        for layer_module in self.layer.iter() {
            hidden_states = layer_module.forward(hidden_states, None);
        }
        self.layernorm.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2PredictorEmbeddings<B: Backend> {
    pub predictor_embeddings: Linear<B>,
    pub mask_tokens: Param<Tensor<B, 4>>,
    #[module(ignore)]
    pred_hidden_size: usize,
    #[module(ignore)]
    pred_num_mask_tokens: usize,
}

impl<B: Backend> Vjepa2PredictorEmbeddings<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        let mask_tokens = if config.pred_zero_init_mask_tokens {
            Param::from_tensor(Tensor::<B, 4>::zeros(
                [
                    config.pred_num_mask_tokens.max(1),
                    1,
                    1,
                    config.pred_hidden_size.max(1),
                ],
                device,
            ))
        } else {
            Param::from_tensor(Tensor::<B, 4>::random(
                [
                    config.pred_num_mask_tokens.max(1),
                    1,
                    1,
                    config.pred_hidden_size.max(1),
                ],
                burn::tensor::Distribution::Normal(0.0, 0.02),
                device,
            ))
        };
        Self {
            predictor_embeddings: LinearConfig::new(
                config.hidden_size.max(1),
                config.pred_hidden_size.max(1),
            )
            .init(device),
            mask_tokens,
            pred_hidden_size: config.pred_hidden_size.max(1),
            pred_num_mask_tokens: config.pred_num_mask_tokens.max(1),
        }
    }

    pub fn forward(
        &self,
        hidden_states: Tensor<B, 3>,
        context_mask: &[Tensor<B, 2, Int>],
        target_mask: &[Tensor<B, 2, Int>],
        mask_index: usize,
    ) -> (Tensor<B, 3>, Tensor<B, 2, Int>) {
        let batch = hidden_states.shape().dims::<3>()[0];
        let context = self.predictor_embeddings.forward(hidden_states);
        let mask_index = mask_index % self.pred_num_mask_tokens.max(1);
        let max_patch_num = target_mask
            .first()
            .expect("target mask")
            .clone()
            .max()
            .into_scalar()
            .elem::<i64>() as usize
            + 1;
        let target = self
            .mask_tokens
            .val()
            .slice_dim(0, mask_index..mask_index + 1)
            .reshape([1, 1, self.pred_hidden_size.max(1)])
            .repeat_dim(0, batch)
            .repeat_dim(1, max_patch_num.max(1));
        let target = apply_masks(target, target_mask);
        let context = context.repeat_dim(0, context_mask.len().max(1));
        let embeddings = Tensor::cat(vec![context, target], 1);
        let cm = Tensor::cat(context_mask.iter().cloned().collect(), 0);
        let tm = Tensor::cat(target_mask.iter().cloned().collect(), 0);
        let masks = Tensor::cat(vec![cm, tm], 1);
        (embeddings, masks)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Predictor<B: Backend> {
    pub embeddings: Vjepa2PredictorEmbeddings<B>,
    pub layer: Vec<Vjepa2Layer<B>>,
    pub layernorm: LayerNorm<B>,
    pub proj: Linear<B>,
}

impl<B: Backend> Vjepa2Predictor<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        let layer = (0..config.pred_num_hidden_layers.max(1))
            .map(|_| {
                Vjepa2Layer::new(
                    &config,
                    config.pred_hidden_size.max(1),
                    config.pred_num_attention_heads.max(1),
                    config.pred_mlp_ratio.max(1.0),
                    device,
                )
            })
            .collect();
        Self {
            embeddings: Vjepa2PredictorEmbeddings::new(config.clone(), device),
            layer,
            layernorm: LayerNormConfig::new(config.pred_hidden_size.max(1))
                .with_epsilon(config.layer_norm_eps)
                .init(device),
            proj: LinearConfig::new(config.pred_hidden_size.max(1), config.hidden_size.max(1))
                .init(device),
        }
    }

    pub fn forward(
        &self,
        encoder_hidden_states: Tensor<B, 3>,
        context_mask: &[Tensor<B, 2, Int>],
        target_mask: &[Tensor<B, 2, Int>],
    ) -> Tensor<B, 3> {
        let encoder_hidden_states = apply_masks(encoder_hidden_states, context_mask);
        let context_tokens = encoder_hidden_states.shape().dims::<3>()[1];
        let (mut hidden_states, position_masks) =
            self.embeddings
                .forward(encoder_hidden_states, context_mask, target_mask, 1);
        let argsort = position_masks.clone().argsort(1);
        hidden_states = sort_tokens(hidden_states, argsort.clone());
        let position_masks = position_masks.gather(1, argsort.clone());
        for layer_module in self.layer.iter() {
            hidden_states = layer_module.forward(hidden_states, Some(position_masks.clone()));
        }
        hidden_states = self.layernorm.forward(hidden_states);
        hidden_states = unsort_tokens(hidden_states, argsort);
        let total_tokens = hidden_states.shape().dims::<3>()[1];
        hidden_states = hidden_states.slice_dim(1, context_tokens..total_tokens);
        self.proj.forward(hidden_states)
    }
}

#[derive(Module, Debug)]
pub struct Vjepa2Model<B: Backend> {
    pub encoder: Vjepa2Encoder<B>,
    pub predictor: Vjepa2Predictor<B>,
    #[module(ignore)]
    pub config: Ignored<Vjepa2Config>,
}

impl<B: Backend> Vjepa2Model<B> {
    pub fn new(config: Vjepa2Config, device: &B::Device) -> Self {
        Self {
            encoder: Vjepa2Encoder::new(config.clone(), device),
            predictor: Vjepa2Predictor::new(config.clone(), device),
            config: Ignored(config),
        }
    }

    pub fn from_hf_dir(dir: impl AsRef<Path>, device: &B::Device) -> Result<Self> {
        let dir = dir.as_ref();
        let config = Vjepa2Config::from_json_file(dir.join("config.json"))?;
        let mut model = Self::new(config, device);
        let mut store = SafetensorsStore::from_file(dir.join("model.safetensors"))
            .with_from_adapter(PyTorchToBurnAdapter)
            .allow_partial(false)
            .validate(true);
        let result = model
            .load_from(&mut store)
            .with_context(|| format!("load V-JEPA2 weights from {}", dir.display()))?;
        if !result.errors.is_empty() {
            bail!("failed to apply V-JEPA2 weights: {:?}", result.errors);
        }
        Ok(model)
    }

    pub fn get_vision_features(&self, pixel_values_videos: Tensor<B, 5>) -> Tensor<B, 3> {
        self.encoder.forward(pixel_values_videos)
    }

    pub fn forward(
        &self,
        pixel_values_videos: Tensor<B, 5>,
        context_mask: Option<Vec<Tensor<B, 2, Int>>>,
        target_mask: Option<Vec<Tensor<B, 2, Int>>>,
        skip_predictor: bool,
    ) -> Vjepa2ModelOutput<B> {
        let sequence_output = self.encoder.forward(pixel_values_videos.clone());
        let batch = pixel_values_videos.shape().dims::<5>()[0];
        let tokens = sequence_output.shape().dims::<3>()[1];
        let device = sequence_output.device();
        let context_mask = context_mask.unwrap_or_else(|| {
            vec![
                Tensor::<B, 1, Int>::arange(0..tokens as i64, &device)
                    .unsqueeze_dim::<2>(0)
                    .repeat_dim(0, batch),
            ]
        });
        let target_mask = target_mask.unwrap_or_else(|| {
            vec![
                Tensor::<B, 1, Int>::arange(0..tokens as i64, &device)
                    .unsqueeze_dim::<2>(0)
                    .repeat_dim(0, batch),
            ]
        });
        let masked_hidden_state = apply_masks(sequence_output.clone(), &context_mask);
        let predictor_output = if skip_predictor {
            None
        } else {
            let predicted =
                self.predictor
                    .forward(sequence_output.clone(), &context_mask, &target_mask);
            Some(Vjepa2PredictorOutput {
                last_hidden_state: predicted,
                target_hidden_state: apply_masks(sequence_output.clone(), &target_mask),
            })
        };
        Vjepa2ModelOutput {
            last_hidden_state: sequence_output,
            masked_hidden_state,
            predictor_output,
        }
    }
}

fn apply_masks<B: Backend>(tensor: Tensor<B, 3>, masks: &[Tensor<B, 2, Int>]) -> Tensor<B, 3> {
    let feature_dim = tensor.shape().dims::<3>()[2];
    let mut all = Vec::with_capacity(masks.len());
    for mask in masks.iter() {
        let mask_keep = mask
            .clone()
            .unsqueeze_dim::<3>(2)
            .repeat_dim(2, feature_dim.max(1));
        all.push(tensor.clone().gather(1, mask_keep));
    }
    Tensor::cat(all, 0)
}

fn sort_tokens<B: Backend>(
    hidden_states: Tensor<B, 3>,
    argsort: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let hidden = hidden_states.shape().dims::<3>()[2];
    let index = argsort.unsqueeze_dim::<3>(2).repeat_dim(2, hidden.max(1));
    hidden_states.gather(1, index)
}

fn unsort_tokens<B: Backend>(
    hidden_states: Tensor<B, 3>,
    argsort: Tensor<B, 2, Int>,
) -> Tensor<B, 3> {
    let reverse = argsort.argsort(1);
    let hidden = hidden_states.shape().dims::<3>()[2];
    let index = reverse.unsqueeze_dim::<3>(2).repeat_dim(2, hidden.max(1));
    hidden_states.gather(1, index)
}
