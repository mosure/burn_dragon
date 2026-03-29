use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct AutoGazeConfig {
    pub model_type: String,
    pub attn_mode: String,
    pub scales: String,
    pub num_vision_tokens_each_frame: usize,
    pub max_num_frames: usize,
    pub use_flash_attn: bool,
    pub has_task_loss_requirement_during_training: bool,
    pub has_task_loss_requirement_during_inference: bool,
    pub gazing_ratio_config: serde_json::Value,
    pub gazing_ratio_each_frame_config: serde_json::Value,
    pub task_loss_requirement_config: serde_json::Value,
    pub gaze_model_config: GazeModelConfig,
}

impl Default for AutoGazeConfig {
    fn default() -> Self {
        Self {
            model_type: "autogaze".to_string(),
            attn_mode: "sdpa".to_string(),
            scales: "224".to_string(),
            num_vision_tokens_each_frame: 196,
            max_num_frames: 16,
            use_flash_attn: false,
            has_task_loss_requirement_during_training: false,
            has_task_loss_requirement_during_inference: false,
            gazing_ratio_config: serde_json::json!({}),
            gazing_ratio_each_frame_config: serde_json::json!({}),
            task_loss_requirement_config: serde_json::json!({}),
            gaze_model_config: GazeModelConfig::default(),
        }
    }
}

impl AutoGazeConfig {
    pub fn from_json_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = fs::read(path)
            .with_context(|| format!("read AutoGaze config from {}", path.display()))?;
        serde_json::from_slice(&bytes)
            .with_context(|| format!("parse AutoGaze config {}", path.display()))
    }

    pub fn scale_values(&self) -> Vec<usize> {
        self.scales
            .split('+')
            .filter_map(|part| part.trim().parse::<usize>().ok())
            .collect()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeModelConfig {
    pub input_img_size: usize,
    pub num_vision_tokens_each_frame: usize,
    pub attn_mode: String,
    pub vision_model_config: VisionModelConfig,
    pub connector_config: ConnectorConfig,
    pub gaze_decoder_config: GazeDecoderConfig,
}

impl Default for GazeModelConfig {
    fn default() -> Self {
        Self {
            input_img_size: 224,
            num_vision_tokens_each_frame: 196,
            attn_mode: "sdpa".to_string(),
            vision_model_config: VisionModelConfig::default(),
            connector_config: ConnectorConfig::default(),
            gaze_decoder_config: GazeDecoderConfig::default(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionModelConfig {
    pub hidden_dim: usize,
    pub out_dim: usize,
    pub depth: usize,
    pub kernel_size: usize,
    pub temporal_patch_size: usize,
    pub trunk_temporal_kernel_size: usize,
    pub trunk_spatial_kernel_size: usize,
}

impl Default for VisionModelConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 192,
            out_dim: 192,
            depth: 1,
            kernel_size: 16,
            temporal_patch_size: 1,
            trunk_temporal_kernel_size: 3,
            trunk_spatial_kernel_size: 3,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct ConnectorConfig {
    pub hidden_dim: usize,
    pub num_tokens: usize,
}

impl Default for ConnectorConfig {
    fn default() -> Self {
        Self {
            hidden_dim: 192,
            num_tokens: 196,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeDecoderConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    pub rope_theta: f32,
    pub rope_scaling: Option<serde_json::Value>,
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub mlp_bias: bool,
    pub head_dim: usize,
    pub attn_mode: String,
    pub num_multi_token_pred: usize,
}

impl Default for GazeDecoderConfig {
    fn default() -> Self {
        Self {
            model_type: "llama".to_string(),
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1.0e-6,
            use_cache: true,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            attention_dropout: 0.0,
            mlp_bias: false,
            head_dim: 128,
            attn_mode: "sdpa".to_string(),
            num_multi_token_pred: 1,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_cached_autogaze_config_shape() {
        let path = Path::new(
            "/home/mosure/.cache/huggingface/hub/models--nvidia--AutoGaze/snapshots/5100fae739ec1bf3f875914fa1b703846a18943a/config.json",
        );
        let config = AutoGazeConfig::from_json_file(path).expect("parse autogaze config");
        assert_eq!(config.model_type, "autogaze");
        assert_eq!(config.gaze_model_config.vision_model_config.hidden_dim, 192);
        assert_eq!(
            config.gaze_model_config.gaze_decoder_config.hidden_size,
            192
        );
        assert_eq!(
            config
                .gaze_model_config
                .gaze_decoder_config
                .num_hidden_layers,
            4
        );
        assert_eq!(config.scale_values(), vec![32, 64, 112, 224]);
    }
}
