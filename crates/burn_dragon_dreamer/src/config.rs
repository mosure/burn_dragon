use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DreamerLatentBackend {
    Pooled,
    TransformerBaseline,
    BdhPooled,
    BdhChallenger,
}

impl DreamerLatentBackend {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pooled => "pooled",
            Self::TransformerBaseline => "transformer_baseline",
            Self::BdhPooled => "bdh_pooled",
            Self::BdhChallenger => "bdh_challenger",
        }
    }

    pub fn uses_slot_tokenizer(self) -> bool {
        matches!(self, Self::TransformerBaseline | Self::BdhChallenger)
    }

    pub fn uses_bdh_core(self) -> bool {
        matches!(self, Self::BdhPooled | Self::BdhChallenger)
    }

    pub fn is_transformer_baseline(self) -> bool {
        matches!(self, Self::TransformerBaseline)
    }

    pub fn is_bdh_challenger(self) -> bool {
        matches!(self, Self::BdhChallenger)
    }
}

impl Default for DreamerLatentBackend {
    fn default() -> Self {
        Self::Pooled
    }
}

impl DreamerConfig {
    pub fn moving_mnist_transformer_baseline() -> Self {
        Self {
            latent_backend: DreamerLatentBackend::TransformerBaseline,
            transformer_layers: 4,
            transformer_heads: 4,
            slot_grid_size: 7,
            latent_dim: 160,
            peripheral_dim: 128,
            fovea_dim: 128,
            tokenizer_loss_weight: 0.9,
            tokenizer_slot_align_weight: 0.45,
            recon_loss_weight: 0.4,
            recon_current_weight: 1.4,
            recon_future_weight: 2.4,
            recon_motion_weight: 1.1,
            tokenizer_pretrain_steps: 24,
            ..Self::default()
        }
    }

    pub fn moving_mnist_bdh_challenger() -> Self {
        Self {
            latent_backend: DreamerLatentBackend::BdhChallenger,
            use_bdh_posterior: true,
            bdh_layers: 2,
            bdh_heads: 4,
            slot_grid_size: 7,
            latent_dim: 160,
            peripheral_dim: 128,
            fovea_dim: 128,
            transformer_layers: 3,
            transformer_heads: 4,
            tokenizer_loss_weight: 1.0,
            tokenizer_slot_align_weight: 0.5,
            recon_loss_weight: 0.45,
            recon_current_weight: 1.4,
            recon_future_weight: 2.5,
            recon_motion_weight: 1.15,
            tokenizer_pretrain_steps: 28,
            ..Self::default()
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct DreamerConfig {
    pub channels: usize,
    pub frame_size: usize,
    pub crop_size: usize,
    pub k_fovea: usize,
    pub latent_backend: DreamerLatentBackend,
    pub use_bdh_posterior: bool,
    pub bdh_layers: usize,
    pub bdh_heads: usize,
    pub bdh_mlp_internal_dim_multiplier: usize,
    pub peripheral_dim: usize,
    pub fovea_dim: usize,
    pub latent_dim: usize,
    pub transformer_layers: usize,
    pub transformer_heads: usize,
    pub slot_grid_size: usize,
    pub teacher_dim: usize,
    pub crop_teacher_dim: usize,
    pub current_loss_weight: f32,
    pub future_loss_weight: f32,
    pub prior_loss_weight: f32,
    pub gaze_loss_weight: f32,
    pub query_loss_weight: f32,
    pub recon_loss_weight: f32,
    pub tokenizer_loss_weight: f32,
    pub tokenizer_slot_align_weight: f32,
    pub recon_current_weight: f32,
    pub recon_future_weight: f32,
    pub recon_edge_weight: f32,
    pub recon_motion_weight: f32,
    pub tokenizer_pretrain_steps: usize,
    pub warmup_fraction: f32,
    pub tokenizer_scale_start: f32,
    pub tokenizer_scale_end: f32,
    pub dynamics_scale_start: f32,
    pub dynamics_scale_end: f32,
    pub recon_scale_start: f32,
    pub recon_scale_end: f32,
    pub gaze_scale_start: f32,
    pub gaze_scale_end: f32,
}

impl Default for DreamerConfig {
    fn default() -> Self {
        Self {
            channels: 1,
            frame_size: 28,
            crop_size: 12,
            k_fovea: 1,
            latent_backend: DreamerLatentBackend::Pooled,
            use_bdh_posterior: false,
            bdh_layers: 2,
            bdh_heads: 4,
            bdh_mlp_internal_dim_multiplier: 4,
            peripheral_dim: 96,
            fovea_dim: 96,
            latent_dim: 128,
            transformer_layers: 2,
            transformer_heads: 4,
            slot_grid_size: 7,
            teacher_dim: 64,
            crop_teacher_dim: 32,
            current_loss_weight: 1.0,
            future_loss_weight: 1.0,
            prior_loss_weight: 0.2,
            gaze_loss_weight: 0.5,
            query_loss_weight: 0.2,
            recon_loss_weight: 0.2,
            tokenizer_loss_weight: 0.5,
            tokenizer_slot_align_weight: 0.25,
            recon_current_weight: 1.0,
            recon_future_weight: 2.0,
            recon_edge_weight: 0.5,
            recon_motion_weight: 0.75,
            tokenizer_pretrain_steps: 0,
            warmup_fraction: 0.33,
            tokenizer_scale_start: 2.5,
            tokenizer_scale_end: 1.0,
            dynamics_scale_start: 0.15,
            dynamics_scale_end: 1.0,
            recon_scale_start: 1.5,
            recon_scale_end: 1.0,
            gaze_scale_start: 1.25,
            gaze_scale_end: 1.0,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct MovingMnistDreamerTrainConfig {
    pub model: DreamerConfig,
    pub steps: usize,
    pub batch_size: usize,
    pub context_len: usize,
    pub target_len: usize,
    pub learning_rate: f64,
    pub weight_decay: f32,
    pub log_every: usize,
    pub validate_every: usize,
    pub valid_batches: usize,
    pub min_velocity: f32,
    pub max_velocity: f32,
    pub train_seed: u64,
    pub val_seed: u64,
    pub autogaze_trace_store: Option<PathBuf>,
    pub autogaze_train_trace_store: Option<PathBuf>,
    pub autogaze_val_trace_store: Option<PathBuf>,
    pub tokenizer_checkpoint: Option<PathBuf>,
    pub freeze_tokenizer: bool,
    pub vjepa_checkpoint: Option<PathBuf>,
    pub vjepa_config_paths: Vec<PathBuf>,
    pub vjepa_feature_store: Option<PathBuf>,
    pub crop_teacher_feature_store: Option<PathBuf>,
    pub allow_teacher_fallbacks: bool,
    pub run_root: Option<PathBuf>,
    pub artifact_enabled: bool,
    pub artifact_dir: Option<PathBuf>,
    pub artifact_every: usize,
    pub artifact_samples: usize,
    pub artifact_future_steps: usize,
}

impl Default for MovingMnistDreamerTrainConfig {
    fn default() -> Self {
        Self {
            model: DreamerConfig::default(),
            steps: 120,
            batch_size: 8,
            context_len: 4,
            target_len: 2,
            learning_rate: 3.0e-4,
            weight_decay: 1.0e-4,
            log_every: 20,
            validate_every: 40,
            valid_batches: 8,
            min_velocity: 1.0,
            max_velocity: 3.0,
            train_seed: 13,
            val_seed: 29,
            autogaze_trace_store: None,
            autogaze_train_trace_store: None,
            autogaze_val_trace_store: None,
            tokenizer_checkpoint: None,
            freeze_tokenizer: false,
            vjepa_checkpoint: None,
            vjepa_config_paths: Vec::new(),
            vjepa_feature_store: None,
            crop_teacher_feature_store: None,
            allow_teacher_fallbacks: false,
            run_root: Some(PathBuf::from("runs/burn_dragon_dreamer/moving_mnist")),
            artifact_enabled: true,
            artifact_dir: None,
            artifact_every: 0,
            artifact_samples: 1,
            artifact_future_steps: 8,
        }
    }
}

impl MovingMnistDreamerTrainConfig {
    pub fn moving_mnist_transformer_baseline() -> Self {
        Self {
            model: DreamerConfig::moving_mnist_transformer_baseline(),
            steps: 160,
            batch_size: 8,
            context_len: 4,
            target_len: 4,
            artifact_future_steps: 10,
            ..Self::default()
        }
    }

    pub fn moving_mnist_bdh_challenger() -> Self {
        Self {
            model: DreamerConfig::moving_mnist_bdh_challenger(),
            steps: 160,
            batch_size: 8,
            context_len: 4,
            target_len: 4,
            artifact_future_steps: 10,
            ..Self::default()
        }
    }
}
