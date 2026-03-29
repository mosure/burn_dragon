use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Vjepa2Config {
    pub patch_size: usize,
    pub crop_size: usize,
    pub frames_per_clip: usize,
    pub tubelet_size: usize,
    pub hidden_size: usize,
    pub in_chans: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub drop_path_rate: f32,
    pub mlp_ratio: f32,
    pub layer_norm_eps: f64,
    pub qkv_bias: bool,
    pub attention_probs_dropout_prob: f64,
    pub hidden_act: String,
    pub initializer_range: f32,
    pub attention_dropout: f64,
    pub num_pooler_layers: usize,
    pub pred_hidden_size: usize,
    pub pred_num_attention_heads: usize,
    pub pred_num_hidden_layers: usize,
    pub pred_num_mask_tokens: usize,
    pub pred_zero_init_mask_tokens: bool,
    pub pred_mlp_ratio: f32,
    #[serde(default = "default_attn_implementation")]
    pub attn_implementation: String,
}

fn default_attn_implementation() -> String {
    "eager".to_string()
}

impl Default for Vjepa2Config {
    fn default() -> Self {
        Self {
            patch_size: 16,
            crop_size: 256,
            frames_per_clip: 64,
            tubelet_size: 2,
            hidden_size: 1024,
            in_chans: 3,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            drop_path_rate: 0.0,
            mlp_ratio: 4.0,
            layer_norm_eps: 1.0e-6,
            qkv_bias: true,
            attention_probs_dropout_prob: 0.0,
            hidden_act: "gelu".to_string(),
            initializer_range: 0.02,
            attention_dropout: 0.0,
            num_pooler_layers: 3,
            pred_hidden_size: 384,
            pred_num_attention_heads: 12,
            pred_num_hidden_layers: 12,
            pred_num_mask_tokens: 10,
            pred_zero_init_mask_tokens: true,
            pred_mlp_ratio: 4.0,
            attn_implementation: default_attn_implementation(),
        }
    }
}

impl Vjepa2Config {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            fs::read_to_string(path).with_context(|| format!("read config {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse config {}", path.display()))
    }

    pub fn grid_size(&self) -> usize {
        self.crop_size.max(1) / self.patch_size.max(1)
    }

    pub fn grid_depth(&self) -> usize {
        self.frames_per_clip.max(1) / self.tubelet_size.max(1)
    }

    pub fn num_patches(&self) -> usize {
        self.grid_depth() * self.grid_size() * self.grid_size()
    }
}
