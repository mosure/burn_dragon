#![cfg(feature = "train")]

use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::module::Module;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::record::{BinFileRecorder, FullPrecisionSettings, Recorder};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_dragon_checkpoint::write_json_snapshot;
use burn_dragon_core::api::recurrent::BDH;
use burn_dragon_language::api::checkpoint::{
    load_language_core_from_checkpoint, load_tokenizer_for_checkpoint,
};
use burn_dragon_language::api::inference::CharVocab;
use burn_dragon_train::api::config::WgpuRuntimeConfig;
use burn_dragon_train::api::expert::train::pipeline::{create_run_dir, write_latest_run};
use burn_dragon_train::api::runtime::cleanup_device_memory;
use burn_dragon_vision::api::checkpoint::load_vision_encoder_from_checkpoint;
use burn_dragon_vision::api::model::VisionDragon;
use serde::{Deserialize, Serialize};

use crate::checkpoint::write_training_snapshot as write_model_snapshot;
use crate::config::VlJepaDragonConfig;
use crate::config_io::{load_merged_config, load_merged_value};
use crate::ema::{init_momentum_teacher, sync_optional_teacher_from_student};
use crate::train::{
    ImagenetteVisionLanguageDataset, JsonlVideoLanguageDataset, JsonlVisionLanguageDataset,
    MnistVideoLanguageDataset, MnistVisionLanguageDataset, MovingMnistVideoLanguageDataset,
    MovingMnistVideoLanguageDatasetConfig, TargetTextBankBatch, VideoLanguageJsonlRecord,
    VisionLanguageJsonlRecord, collate_video_language_segments, collate_vision_language_segments,
    multimodal_eval_step, multimodal_train_step_with_frozen_cores, multimodal_video_eval_step,
    multimodal_video_train_step_with_frozen_cores,
};
use burn_dragon_stream::StreamDataset;

const TRAINING_CONFIG_SNAPSHOT_FILE_NAME: &str = "multimodal_training_config.json";
const TOKENIZER_SNAPSHOT_FILE_NAME: &str = "multimodal_tokenizer.json";
const MULTIMODAL_DEVICE_CLEANUP_EVERY_STEPS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultimodalTaskKind {
    #[default]
    ImageText,
    VideoText,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MultimodalRuntimeConfig {
    ImageText(MultimodalTrainingConfig),
    VideoText(MultimodalVideoTrainingConfig),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultimodalImageTextSource {
    #[default]
    Jsonl,
    MnistLabelText,
    ImagenetteLabelText,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum MultimodalVideoTextSource {
    #[default]
    Jsonl,
    MnistLabelText,
    MovingMnistLabelText,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalMnistLabelTextConfig {
    #[serde(default)]
    pub max_train_records: Option<usize>,
    #[serde(default)]
    pub max_validation_records: Option<usize>,
    #[serde(default = "default_digit_query_q_text")]
    pub query_q_text: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalImagenetteLabelTextConfig {
    #[serde(default = "default_imagenette_root")]
    pub root: PathBuf,
    #[serde(default = "default_imagenette_train_dir")]
    pub train_dir: String,
    #[serde(default = "default_imagenette_val_dir")]
    pub validation_dir: String,
    #[serde(default)]
    pub max_train_records: Option<usize>,
    #[serde(default)]
    pub max_validation_records: Option<usize>,
    #[serde(default = "default_imagenette_query_q_text")]
    pub query_q_text: String,
}

impl Default for MultimodalImagenetteLabelTextConfig {
    fn default() -> Self {
        Self {
            root: default_imagenette_root(),
            train_dir: default_imagenette_train_dir(),
            validation_dir: default_imagenette_val_dir(),
            max_train_records: None,
            max_validation_records: Some(512),
            query_q_text: default_imagenette_query_q_text(),
        }
    }
}

impl Default for MultimodalMnistLabelTextConfig {
    fn default() -> Self {
        Self {
            max_train_records: None,
            max_validation_records: Some(256),
            query_q_text: default_digit_query_q_text(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalMovingMnistLabelTextConfig {
    #[serde(default)]
    pub max_train_records: Option<usize>,
    #[serde(default)]
    pub max_validation_records: Option<usize>,
    #[serde(default = "default_digit_query_q_text")]
    pub query_q_text: String,
    #[serde(default = "default_moving_mnist_digit_size")]
    pub digit_size: usize,
    #[serde(default = "default_moving_mnist_in_channels")]
    pub in_channels: usize,
    #[serde(default = "default_moving_mnist_frame_stride")]
    pub frame_stride: usize,
    #[serde(default = "default_moving_mnist_min_velocity")]
    pub min_velocity: f32,
    #[serde(default = "default_moving_mnist_max_velocity")]
    pub max_velocity: f32,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

impl Default for MultimodalMovingMnistLabelTextConfig {
    fn default() -> Self {
        Self {
            max_train_records: None,
            max_validation_records: Some(256),
            query_q_text: default_digit_query_q_text(),
            digit_size: default_moving_mnist_digit_size(),
            in_channels: default_moving_mnist_in_channels(),
            frame_stride: default_moving_mnist_frame_stride(),
            min_velocity: default_moving_mnist_min_velocity(),
            max_velocity: default_moving_mnist_max_velocity(),
            seed: default_seed(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalImageTextDataConfig {
    #[serde(default)]
    pub source: MultimodalImageTextSource,
    pub manifest: PathBuf,
    #[serde(default)]
    pub validation_manifest: Option<PathBuf>,
    pub image_size: usize,
    #[serde(default = "default_include_unknown_char")]
    pub include_unknown_char: bool,
    #[serde(default = "default_normalize_mean")]
    pub normalize_mean: [f32; 3],
    #[serde(default = "default_normalize_std")]
    pub normalize_std: [f32; 3],
    #[serde(default)]
    pub mnist: MultimodalMnistLabelTextConfig,
    #[serde(default)]
    pub imagenette: MultimodalImagenetteLabelTextConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalVideoTextDataConfig {
    #[serde(default)]
    pub source: MultimodalVideoTextSource,
    pub manifest: PathBuf,
    #[serde(default)]
    pub validation_manifest: Option<PathBuf>,
    pub image_size: usize,
    #[serde(default = "default_video_clip_frames")]
    pub clip_frames: usize,
    #[serde(default = "default_requested_horizons")]
    pub requested_horizons: Vec<usize>,
    #[serde(default = "default_include_unknown_char")]
    pub include_unknown_char: bool,
    #[serde(default = "default_normalize_mean")]
    pub normalize_mean: [f32; 3],
    #[serde(default = "default_normalize_std")]
    pub normalize_std: [f32; 3],
    #[serde(default)]
    pub mnist: MultimodalMnistLabelTextConfig,
    #[serde(default)]
    pub moving_mnist: MultimodalMovingMnistLabelTextConfig,
}

impl Default for MultimodalVideoTextDataConfig {
    fn default() -> Self {
        Self {
            source: MultimodalVideoTextSource::Jsonl,
            manifest: PathBuf::from("data/multimodal/video_text.jsonl"),
            validation_manifest: None,
            image_size: 32,
            clip_frames: default_video_clip_frames(),
            requested_horizons: default_requested_horizons(),
            include_unknown_char: true,
            normalize_mean: default_normalize_mean(),
            normalize_std: default_normalize_std(),
            mnist: MultimodalMnistLabelTextConfig::default(),
            moving_mnist: MultimodalMovingMnistLabelTextConfig::default(),
        }
    }
}

impl Default for MultimodalImageTextDataConfig {
    fn default() -> Self {
        Self {
            source: MultimodalImageTextSource::Jsonl,
            manifest: PathBuf::from("data/multimodal/image_text.jsonl"),
            validation_manifest: None,
            image_size: 32,
            include_unknown_char: true,
            normalize_mean: default_normalize_mean(),
            normalize_std: default_normalize_std(),
            mnist: MultimodalMnistLabelTextConfig::default(),
            imagenette: MultimodalImagenetteLabelTextConfig::default(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalPretrainedVisionEncoderConfig {
    pub checkpoint: PathBuf,
    #[serde(default)]
    pub epoch: Option<usize>,
    #[serde(default)]
    pub config_paths: Vec<PathBuf>,
    #[serde(default)]
    pub freeze: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalPretrainedTextCoreConfig {
    pub checkpoint: PathBuf,
    #[serde(default)]
    pub epoch: Option<usize>,
    #[serde(default)]
    pub config_paths: Vec<PathBuf>,
    #[serde(default = "default_language_backend_name")]
    pub backend_name: String,
    #[serde(default = "default_true")]
    pub use_pretrained_tokenizer: bool,
    #[serde(default = "default_true")]
    pub initialize_fusion: bool,
    #[serde(default)]
    pub freeze_query: bool,
    #[serde(default = "default_true")]
    pub freeze_target: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MultimodalPretrainedInitConfig {
    #[serde(default)]
    pub vision_x_encoder: Option<MultimodalPretrainedVisionEncoderConfig>,
    #[serde(default)]
    pub text_core: Option<MultimodalPretrainedTextCoreConfig>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalTrainingLoopConfig {
    #[serde(default = "default_run_root")]
    pub run_root: PathBuf,
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f64,
    #[serde(default)]
    pub max_steps_per_epoch: Option<usize>,
    #[serde(default)]
    pub max_validation_steps_per_epoch: Option<usize>,
    #[serde(default = "default_checkpoint_every_epochs")]
    pub checkpoint_every_epochs: usize,
    #[serde(default = "default_artifact_every_epochs")]
    pub artifact_every_epochs: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_weight_decay")]
    pub weight_decay: f32,
}

impl Default for MultimodalTrainingLoopConfig {
    fn default() -> Self {
        Self {
            run_root: default_run_root(),
            epochs: 1,
            batch_size: 1,
            learning_rate: 1.0e-3,
            max_steps_per_epoch: None,
            max_validation_steps_per_epoch: None,
            checkpoint_every_epochs: default_checkpoint_every_epochs(),
            artifact_every_epochs: default_artifact_every_epochs(),
            seed: default_seed(),
            weight_decay: default_weight_decay(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MultimodalTrainingConfig {
    #[serde(default)]
    pub task: MultimodalTaskKind,
    #[serde(default)]
    pub model: VlJepaDragonConfig,
    #[serde(default)]
    pub data: MultimodalImageTextDataConfig,
    #[serde(default)]
    pub pretrained: MultimodalPretrainedInitConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    #[serde(default)]
    pub training: MultimodalTrainingLoopConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct MultimodalVideoTrainingConfig {
    #[serde(default = "default_video_text_task")]
    pub task: MultimodalTaskKind,
    #[serde(default)]
    pub model: VlJepaDragonConfig,
    #[serde(default)]
    pub data: MultimodalVideoTextDataConfig,
    #[serde(default)]
    pub pretrained: MultimodalPretrainedInitConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    #[serde(default)]
    pub training: MultimodalTrainingLoopConfig,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalEpochArtifact {
    pub epoch: usize,
    pub steps: usize,
    pub mean_total_loss: f32,
    pub mean_diagonal_similarity: f32,
    pub mean_top1_accuracy: f32,
    pub validation_steps: usize,
    pub validation_mean_total_loss: Option<f32>,
    pub validation_mean_diagonal_similarity: Option<f32>,
    pub validation_mean_top1_accuracy: Option<f32>,
    pub validation_refine_curve: Option<Vec<RefineProbeMetric>>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MultimodalTrainingReport {
    pub run_dir: PathBuf,
    pub run_name: String,
    pub checkpoint_paths: Vec<PathBuf>,
    pub artifact_paths: Vec<PathBuf>,
    pub epochs: Vec<MultimodalEpochArtifact>,
    pub tokenizer_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RefineProbeMetric {
    pub refine_steps: usize,
    pub mean_total_loss: f32,
    pub mean_top1_accuracy: f32,
}

struct ImageTextDatasetBundle {
    train: Vec<crate::train::VisionLanguageCpuSegment>,
    validation: Option<Vec<crate::train::VisionLanguageCpuSegment>>,
    vocab: CharVocab,
    supports_batched_contrastive: bool,
    target_bank_texts: Option<Vec<String>>,
}

struct VideoTextDatasetBundle {
    train: Vec<crate::train::VideoLanguageCpuSegment>,
    validation: Option<Vec<crate::train::VideoLanguageCpuSegment>>,
    vocab: CharVocab,
    supports_batched_contrastive: bool,
    target_bank_texts: Option<Vec<String>>,
}

struct PreparedTargetBank<B: BackendTrait> {
    tokens: burn::tensor::Tensor<B, 2, burn::tensor::Int>,
    mask: Option<burn::tensor::Tensor<B, 2, burn::tensor::Bool>>,
    token_sequences: Vec<Vec<i64>>,
}

struct ResolvedMultimodalInit<B: BackendTrait> {
    model_config: VlJepaDragonConfig,
    pretrained_vocab: Option<CharVocab>,
    vision_x_encoder: Option<VisionDragon<B>>,
    text_core: Option<BDH<B>>,
    fusion_core: Option<BDH<B>>,
    freeze_vision_x_encoder: bool,
    freeze_query_q_encoder: bool,
    freeze_target_y_encoder: bool,
}

pub fn load_multimodal_training_runtime_config(
    config_paths: &[PathBuf],
) -> Result<MultimodalTrainingConfig> {
    load_merged_config(config_paths)
}

pub fn load_multimodal_runtime_config(config_paths: &[PathBuf]) -> Result<MultimodalRuntimeConfig> {
    #[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
    struct TaskProbe {
        #[serde(default)]
        task: MultimodalTaskKind,
    }

    let merged = load_merged_value(config_paths, TaskProbe::default())?;
    let task = serde_json::from_value::<TaskProbe>(merged.clone())
        .context("failed to inspect multimodal task kind")?
        .task;
    match task {
        MultimodalTaskKind::ImageText => serde_json::from_value(merged)
            .map(MultimodalRuntimeConfig::ImageText)
            .context("failed to deserialize image-text multimodal runtime config"),
        MultimodalTaskKind::VideoText => serde_json::from_value(merged)
            .map(MultimodalRuntimeConfig::VideoText)
            .context("failed to deserialize video-text multimodal runtime config"),
    }
}

pub fn load_multimodal_video_training_runtime_config(
    config_paths: &[PathBuf],
) -> Result<MultimodalVideoTrainingConfig> {
    load_merged_config(config_paths)
}

pub fn write_runtime_snapshot(
    run_dir: &Path,
    config: &MultimodalTrainingConfig,
) -> Result<PathBuf> {
    write_runtime_snapshot_json(run_dir, config)
}

pub fn write_video_runtime_snapshot(
    run_dir: &Path,
    config: &MultimodalVideoTrainingConfig,
) -> Result<PathBuf> {
    write_runtime_snapshot_json(run_dir, config)
}

fn write_runtime_snapshot_json<T: Serialize>(run_dir: &Path, config: &T) -> Result<PathBuf> {
    write_json_snapshot(run_dir, TRAINING_CONFIG_SNAPSHOT_FILE_NAME, config)?;
    Ok(training_runtime_snapshot_path(run_dir))
}

pub fn training_runtime_snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TRAINING_CONFIG_SNAPSHOT_FILE_NAME)
}

pub fn tokenizer_snapshot_path(run_dir: &Path) -> PathBuf {
    run_dir.join(TOKENIZER_SNAPSHOT_FILE_NAME)
}

pub fn artifact_dir(run_dir: &Path) -> PathBuf {
    run_dir.join("artifacts")
}

fn maybe_cleanup_multimodal_device<B: BackendTrait>(device: &B::Device, step_count: usize) {
    if step_count == 0 {
        return;
    }
    if MULTIMODAL_DEVICE_CLEANUP_EVERY_STEPS > 1
        && !step_count.is_multiple_of(MULTIMODAL_DEVICE_CLEANUP_EVERY_STEPS)
    {
        return;
    }
    let _ = cleanup_device_memory::<B>(device, false);
}

fn resolve_pretrained_char_vocab(
    pretrained: &MultimodalPretrainedTextCoreConfig,
) -> Result<CharVocab> {
    let checkpoint = pretrained.checkpoint.clone();
    let tokenizer = load_tokenizer_for_checkpoint(
        &pretrained.config_paths,
        Some(&checkpoint),
        &pretrained.backend_name,
    )?;
    tokenizer
        .as_ref()
        .as_any()
        .downcast_ref::<CharVocab>()
        .cloned()
        .ok_or_else(|| {
            anyhow!("multimodal pretrained text core currently requires a char tokenizer")
        })
}

fn resolve_image_pretrained_init<B: AutodiffBackend>(
    config: &MultimodalTrainingConfig,
    device: &B::Device,
) -> Result<ResolvedMultimodalInit<B>> {
    let mut model_config = config.model.clone();
    let default_model = VlJepaDragonConfig::default();
    let mut pretrained_vocab = None;
    let mut vision_x_encoder = None;
    let mut text_core = None;
    let mut fusion_core = None;
    let mut freeze_vision_x_encoder = false;
    let mut freeze_query_q_encoder = false;
    let mut freeze_target_y_encoder = false;

    if let Some(pretrained) = config.pretrained.text_core.as_ref() {
        let checkpoint = pretrained.checkpoint.clone();
        let language_config =
            burn_dragon_language::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                Some(&checkpoint),
                &pretrained.backend_name,
            )?;
        let source_config = burn_dragon_language::api::inference::build_model_config(
            &language_config.model,
            language_config.training.block_size,
        );
        model_config.query_text = source_config.clone();
        model_config.target_text = source_config;
        if pretrained.initialize_fusion {
            model_config.fusion = burn_dragon_language::api::inference::build_model_config(
                &language_config.model,
                language_config.training.block_size,
            );
        }
        if model_config.target_dim == default_model.target_dim {
            model_config.target_dim = model_config.target_text.n_embd;
        }
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.query_text.n_embd;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        let loaded_text = load_language_core_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            &pretrained.backend_name,
            device,
        )?;
        text_core = Some(loaded_text.clone());
        if pretrained.initialize_fusion {
            fusion_core = Some(loaded_text);
        }
        if pretrained.use_pretrained_tokenizer {
            pretrained_vocab = Some(resolve_pretrained_char_vocab(pretrained)?);
        }
        freeze_query_q_encoder = pretrained.freeze_query;
        freeze_target_y_encoder = pretrained.freeze_target;
    }

    if let Some(pretrained) = config.pretrained.vision_x_encoder.as_ref() {
        let vision_training =
            burn_dragon_vision::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                &pretrained.checkpoint,
            )?;
        model_config.vision = vision_training.vision.build();
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.vision.embed_dim;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        vision_x_encoder = Some(load_vision_encoder_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            device,
        )?);
        freeze_vision_x_encoder = pretrained.freeze;
    }

    Ok(ResolvedMultimodalInit {
        model_config,
        pretrained_vocab,
        vision_x_encoder,
        text_core,
        fusion_core,
        freeze_vision_x_encoder,
        freeze_query_q_encoder,
        freeze_target_y_encoder,
    })
}

fn resolve_video_pretrained_init<B: AutodiffBackend>(
    config: &MultimodalVideoTrainingConfig,
    device: &B::Device,
) -> Result<ResolvedMultimodalInit<B>> {
    let mut model_config = config.model.clone();
    let default_model = VlJepaDragonConfig::default();
    let mut pretrained_vocab = None;
    let mut vision_x_encoder = None;
    let mut text_core = None;
    let mut fusion_core = None;
    let mut freeze_vision_x_encoder = false;
    let mut freeze_query_q_encoder = false;
    let mut freeze_target_y_encoder = false;

    if let Some(pretrained) = config.pretrained.text_core.as_ref() {
        let checkpoint = pretrained.checkpoint.clone();
        let language_config =
            burn_dragon_language::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                Some(&checkpoint),
                &pretrained.backend_name,
            )?;
        let source_config = burn_dragon_language::api::inference::build_model_config(
            &language_config.model,
            language_config.training.block_size,
        );
        model_config.query_text = source_config.clone();
        model_config.target_text = source_config;
        if pretrained.initialize_fusion {
            model_config.fusion = burn_dragon_language::api::inference::build_model_config(
                &language_config.model,
                language_config.training.block_size,
            );
        }
        if model_config.target_dim == default_model.target_dim {
            model_config.target_dim = model_config.target_text.n_embd;
        }
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.query_text.n_embd;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        let loaded_text = load_language_core_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            &pretrained.backend_name,
            device,
        )?;
        text_core = Some(loaded_text.clone());
        if pretrained.initialize_fusion {
            fusion_core = Some(loaded_text);
        }
        if pretrained.use_pretrained_tokenizer {
            pretrained_vocab = Some(resolve_pretrained_char_vocab(pretrained)?);
        }
        freeze_query_q_encoder = pretrained.freeze_query;
        freeze_target_y_encoder = pretrained.freeze_target;
    }

    if let Some(pretrained) = config.pretrained.vision_x_encoder.as_ref() {
        let vision_training =
            burn_dragon_vision::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                &pretrained.checkpoint,
            )?;
        model_config.vision = vision_training.vision.build();
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.vision.embed_dim;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        vision_x_encoder = Some(load_vision_encoder_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            device,
        )?);
        freeze_vision_x_encoder = pretrained.freeze;
    }

    Ok(ResolvedMultimodalInit {
        model_config,
        pretrained_vocab,
        vision_x_encoder,
        text_core,
        fusion_core,
        freeze_vision_x_encoder,
        freeze_query_q_encoder,
        freeze_target_y_encoder,
    })
}

pub fn train_backend<B, Init>(
    config: &MultimodalTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let run_root = match backend_name {
        "cpu" => config.training.run_root.join("cpu"),
        "cuda" => config.training.run_root.join("cuda"),
        "wgpu" | "wgpu-nofusion" | "wgpu-fused-core" => config.training.run_root.join("wgpu"),
        _ => config.training.run_root.clone(),
    };
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    let mut report = run_image_text_training_backend::<B, _>(config, &run_dir, init)?;
    report.run_name = run_name;
    Ok(report)
}

pub fn train_video_backend<B, Init>(
    config: &MultimodalVideoTrainingConfig,
    backend_name: &str,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    let run_root = match backend_name {
        "cpu" => config.training.run_root.join("cpu"),
        "cuda" => config.training.run_root.join("cuda"),
        "wgpu" | "wgpu-nofusion" | "wgpu-fused-core" => config.training.run_root.join("wgpu"),
        _ => config.training.run_root.clone(),
    };
    let (run_dir, run_name) = create_run_dir(&run_root)?;
    write_latest_run(&run_root, &run_name)?;
    let mut report = run_video_text_training_backend::<B, _>(config, &run_dir, init)?;
    report.run_name = run_name;
    Ok(report)
}

pub fn run_image_text_training_backend<B, Init>(
    config: &MultimodalTrainingConfig,
    run_dir: &Path,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    fs::create_dir_all(artifact_dir(run_dir)).with_context(|| {
        format!(
            "failed to create artifact directory {}",
            artifact_dir(run_dir).display()
        )
    })?;
    fs::create_dir_all(run_dir.join("checkpoint")).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            run_dir.join("checkpoint").display()
        )
    })?;

    write_runtime_snapshot(run_dir, config)?;
    let device = B::Device::default();
    init(&device);
    B::seed(&device, config.training.seed);
    let resolved_init = resolve_image_pretrained_init::<B>(config, &device)?;
    write_model_snapshot(run_dir, &resolved_init.model_config)?;
    let bundle = load_image_text_dataset_bundle(config, resolved_init.pretrained_vocab.as_ref())?;
    if !bundle.supports_batched_contrastive && config.training.batch_size != 1 {
        return Err(anyhow!(
            "multimodal image-text source {:?} requires training.batch_size = 1 because stream resets/detaches are explicit per segment",
            config.data.source
        ));
    }
    let vocab = bundle.vocab;
    let vocab_path = tokenizer_snapshot_path(run_dir);
    vocab.save(&vocab_path)?;
    let prepared_target_bank =
        prepare_target_bank::<B>(bundle.target_bank_texts.as_deref(), &vocab, &device);
    let mut model =
        crate::model::VlJepaDragon::<B>::new(resolved_init.model_config.clone(), &device);
    if let Some(vision_x_encoder) = resolved_init.vision_x_encoder {
        model.vision_x_encoder.replace_encoder(vision_x_encoder);
        model.vision_x_encoder.set_force_projection(true);
    }
    if let Some(text_core) = resolved_init.text_core {
        model.query_q_encoder.replace_encoder(text_core.clone());
        model.target_y_encoder.replace_encoder(text_core);
        model.query_q_encoder.set_force_projection(true);
        model.target_y_encoder.set_force_projection(true);
    }
    if let Some(fusion_core) = resolved_init.fusion_core {
        model.replace_fusion_core(fusion_core);
    }
    model.set_frozen_modalities(
        resolved_init.freeze_vision_x_encoder,
        resolved_init.freeze_query_q_encoder,
        resolved_init.freeze_target_y_encoder,
    );
    let frozen_cores = model.frozen_core_set();
    let mut target_teacher = init_momentum_teacher::<B, _>(
        &model.target_y_encoder,
        &resolved_init.model_config.target_teacher,
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, crate::model::VlJepaDragon<B>>();

    let mut checkpoint_paths = Vec::new();
    let mut artifact_paths = Vec::new();
    let mut epochs = Vec::new();

    for epoch in 0..config.training.epochs {
        let mut loss_sum = 0.0_f32;
        let mut diagonal_similarity_sum = 0.0_f32;
        let mut top1_accuracy_sum = 0.0_f32;
        let mut steps = 0_usize;
        if bundle.supports_batched_contrastive {
            let batch_indices =
                if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                    contiguous_batch_indices(
                        bundle.train.len(),
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                    )
                } else {
                    unique_target_batches(
                        &bundle.train,
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                        |segment| &segment.payload.target_y_tokens,
                    )
                };
            for batch_indices in batch_indices {
                let segments = batch_indices
                    .into_iter()
                    .map(|index| bundle.train[index].clone())
                    .collect::<Vec<_>>();
                let collated = collate_vision_language_segments::<B>(&segments, &device);
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    segments
                        .iter()
                        .map(|segment| segment.payload.target_y_tokens.as_slice()),
                    &device,
                );
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated batch missing stream metadata"))?;
                let step = multimodal_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    model.init_state(),
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = match target_bank {
                    Some(target_bank) => labeled_top1_accuracy(
                        step.loss.similarities.clone(),
                        target_bank.target_indices,
                    ),
                    None => bidirectional_top1_accuracy(step.loss.similarities.clone()),
                };
                let loss = step.loss;
                let forward = step.forward;
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        } else {
            let mut state = model.init_state();
            for range in batch_ranges(bundle.train.len(), 1, config.training.max_steps_per_epoch) {
                let segment = bundle.train[range.start].clone();
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    [segment.payload.target_y_tokens.as_slice()],
                    &device,
                );
                let collated =
                    collate_vision_language_segments::<B>(std::slice::from_ref(&segment), &device);
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated batch missing stream metadata"))?;
                let step = multimodal_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    state,
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = target_bank
                    .map(|target_bank| {
                        labeled_top1_accuracy(
                            step.loss.similarities.clone(),
                            target_bank.target_indices,
                        )
                    })
                    .unwrap_or_else(|| bidirectional_top1_accuracy(step.loss.similarities.clone()));
                let loss = step.loss;
                let forward = step.forward;
                let next_state = forward.state.detach();
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );
                state = next_state;

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        }

        let validation = if let Some(validation) = bundle.validation.as_ref() {
            Some(run_image_text_validation_epoch::<B>(
                &model,
                target_teacher.as_ref(),
                prepared_target_bank.as_ref(),
                validation,
                &device,
                config,
                &resolved_init.model_config,
                bundle.supports_batched_contrastive,
                true,
            )?)
        } else {
            None
        };

        let epoch_index = epoch + 1;
        let artifact = MultimodalEpochArtifact {
            epoch: epoch_index,
            steps,
            mean_total_loss: loss_sum / steps.max(1) as f32,
            mean_diagonal_similarity: diagonal_similarity_sum / steps.max(1) as f32,
            mean_top1_accuracy: top1_accuracy_sum / steps.max(1) as f32,
            validation_steps: validation
                .as_ref()
                .map(|metrics| metrics.steps)
                .unwrap_or(0),
            validation_mean_total_loss: validation.as_ref().map(|metrics| metrics.mean_total_loss),
            validation_mean_diagonal_similarity: validation
                .as_ref()
                .map(|metrics| metrics.mean_diagonal_similarity),
            validation_mean_top1_accuracy: validation
                .as_ref()
                .map(|metrics| metrics.mean_top1_accuracy),
            validation_refine_curve: validation
                .as_ref()
                .and_then(|metrics| metrics.refine_curve.clone()),
        };
        epochs.push(artifact.clone());

        if config.training.artifact_every_epochs > 0
            && epoch_index % config.training.artifact_every_epochs == 0
        {
            let artifact_path = artifact_dir(run_dir).join(format!("epoch-{epoch_index}.json"));
            let payload =
                serde_json::to_string_pretty(&artifact).context("serialize epoch artifact")?;
            fs::write(&artifact_path, payload)
                .with_context(|| format!("failed to write {}", artifact_path.display()))?;
            artifact_paths.push(artifact_path);
        }

        if config.training.checkpoint_every_epochs > 0
            && epoch_index % config.training.checkpoint_every_epochs == 0
        {
            let checkpoint_base = run_dir.join("checkpoint").join(format!("model-{epoch}"));
            BinFileRecorder::<FullPrecisionSettings>::new()
                .record(model.clone().into_record(), checkpoint_base.clone())
                .with_context(|| {
                    format!("failed to write checkpoint {}", checkpoint_base.display())
                })?;
            checkpoint_paths.push(checkpoint_base.with_extension("bin"));
        }
    }

    Ok(MultimodalTrainingReport {
        run_dir: run_dir.to_path_buf(),
        run_name: run_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "multimodal-run".to_string()),
        checkpoint_paths,
        artifact_paths,
        epochs,
        tokenizer_path: vocab_path,
    })
}

pub fn run_video_text_training_backend<B, Init>(
    config: &MultimodalVideoTrainingConfig,
    run_dir: &Path,
    init: Init,
) -> Result<MultimodalTrainingReport>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Default + Clone,
    Init: Fn(&B::Device),
{
    fs::create_dir_all(run_dir)
        .with_context(|| format!("failed to create run directory {}", run_dir.display()))?;
    fs::create_dir_all(artifact_dir(run_dir)).with_context(|| {
        format!(
            "failed to create artifact directory {}",
            artifact_dir(run_dir).display()
        )
    })?;
    fs::create_dir_all(run_dir.join("checkpoint")).with_context(|| {
        format!(
            "failed to create checkpoint directory {}",
            run_dir.join("checkpoint").display()
        )
    })?;

    write_video_runtime_snapshot(run_dir, config)?;
    let device = B::Device::default();
    init(&device);
    B::seed(&device, config.training.seed);
    let resolved_init = resolve_video_pretrained_init::<B>(config, &device)?;
    write_model_snapshot(run_dir, &resolved_init.model_config)?;
    let bundle = load_video_text_dataset_bundle(config, resolved_init.pretrained_vocab.as_ref())?;
    if !bundle.supports_batched_contrastive && config.training.batch_size != 1 {
        return Err(anyhow!(
            "multimodal video-text source {:?} requires training.batch_size = 1 because stream resets/detaches are explicit per segment",
            config.data.source
        ));
    }
    let vocab = bundle.vocab;
    let vocab_path = tokenizer_snapshot_path(run_dir);
    vocab.save(&vocab_path)?;
    let prepared_target_bank =
        prepare_target_bank::<B>(bundle.target_bank_texts.as_deref(), &vocab, &device);
    let mut model =
        crate::model::VlJepaDragon::<B>::new(resolved_init.model_config.clone(), &device);
    if let Some(vision_x_encoder) = resolved_init.vision_x_encoder {
        model.vision_x_encoder.replace_encoder(vision_x_encoder);
        model.vision_x_encoder.set_force_projection(true);
    }
    if let Some(text_core) = resolved_init.text_core {
        model.query_q_encoder.replace_encoder(text_core.clone());
        model.target_y_encoder.replace_encoder(text_core);
        model.query_q_encoder.set_force_projection(true);
        model.target_y_encoder.set_force_projection(true);
    }
    if let Some(fusion_core) = resolved_init.fusion_core {
        model.replace_fusion_core(fusion_core);
    }
    model.set_frozen_modalities(
        resolved_init.freeze_vision_x_encoder,
        resolved_init.freeze_query_q_encoder,
        resolved_init.freeze_target_y_encoder,
    );
    let frozen_cores = model.frozen_core_set();
    let mut target_teacher = init_momentum_teacher::<B, _>(
        &model.target_y_encoder,
        &resolved_init.model_config.target_teacher,
    );
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.training.weight_decay)
        .init::<B, crate::model::VlJepaDragon<B>>();

    let mut checkpoint_paths = Vec::new();
    let mut artifact_paths = Vec::new();
    let mut epochs = Vec::new();

    for epoch in 0..config.training.epochs {
        let mut loss_sum = 0.0_f32;
        let mut diagonal_similarity_sum = 0.0_f32;
        let mut top1_accuracy_sum = 0.0_f32;
        let mut steps = 0_usize;
        if bundle.supports_batched_contrastive {
            let batch_indices =
                if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                    contiguous_batch_indices(
                        bundle.train.len(),
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                    )
                } else {
                    unique_target_batches(
                        &bundle.train,
                        config.training.batch_size,
                        config.training.max_steps_per_epoch,
                        |segment| &segment.payload.target_y_tokens,
                    )
                };
            for batch_indices in batch_indices {
                let segments = batch_indices
                    .into_iter()
                    .map(|index| bundle.train[index].clone())
                    .collect::<Vec<_>>();
                let collated = collate_video_language_segments::<B>(&segments, &device);
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    segments
                        .iter()
                        .map(|segment| segment.payload.target_y_tokens.as_slice()),
                    &device,
                );
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated video batch missing stream metadata"))?;
                let step = multimodal_video_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    model.init_state(),
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = match target_bank {
                    Some(target_bank) => labeled_top1_accuracy(
                        step.loss.similarities.clone(),
                        target_bank.target_indices,
                    ),
                    None => bidirectional_top1_accuracy(step.loss.similarities.clone()),
                };
                let loss = step.loss;
                let forward = step.forward;
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        } else {
            let mut state = model.init_state();
            for range in batch_ranges(bundle.train.len(), 1, config.training.max_steps_per_epoch) {
                let segment = bundle.train[range.start].clone();
                let target_bank = target_bank_batch_for_targets::<B, _>(
                    prepared_target_bank.as_ref(),
                    [segment.payload.target_y_tokens.as_slice()],
                    &device,
                );
                let collated =
                    collate_video_language_segments::<B>(std::slice::from_ref(&segment), &device);
                let stream = collated
                    .stream
                    .first()
                    .copied()
                    .ok_or_else(|| anyhow!("collated video batch missing stream metadata"))?;
                let step = multimodal_video_train_step_with_frozen_cores(
                    &model,
                    Some(&frozen_cores),
                    target_teacher.as_ref(),
                    target_bank.as_ref(),
                    collated.payload,
                    &stream,
                    state,
                    &resolved_init.model_config,
                );
                let loss_scalar = scalar_from_tensor(step.loss.total.clone());
                let diagonal_similarity = diagonal_similarity_mean(step.loss.similarities.clone());
                let top1_accuracy = target_bank
                    .map(|target_bank| {
                        labeled_top1_accuracy(
                            step.loss.similarities.clone(),
                            target_bank.target_indices,
                        )
                    })
                    .unwrap_or_else(|| bidirectional_top1_accuracy(step.loss.similarities.clone()));
                let loss = step.loss;
                let forward = step.forward;
                let next_state = forward.state.detach();
                drop(forward);
                let grads = GradientsParams::from_grads(loss.total.backward(), &model);
                model = optimizer.step(config.training.learning_rate, model, grads);
                target_teacher = sync_optional_teacher_from_student::<B, _>(
                    target_teacher,
                    &model.target_y_encoder,
                    &resolved_init.model_config.target_teacher,
                );
                state = next_state;

                loss_sum += loss_scalar;
                diagonal_similarity_sum += diagonal_similarity;
                top1_accuracy_sum += top1_accuracy;
                steps = steps.saturating_add(1);
                maybe_cleanup_multimodal_device::<B>(&device, steps);
            }
        }

        let validation = if let Some(validation) = bundle.validation.as_ref() {
            Some(run_video_text_validation_epoch::<B>(
                &model,
                target_teacher.as_ref(),
                prepared_target_bank.as_ref(),
                validation,
                &device,
                config,
                &resolved_init.model_config,
                bundle.supports_batched_contrastive,
                true,
            )?)
        } else {
            None
        };

        let epoch_index = epoch + 1;
        let artifact = MultimodalEpochArtifact {
            epoch: epoch_index,
            steps,
            mean_total_loss: loss_sum / steps.max(1) as f32,
            mean_diagonal_similarity: diagonal_similarity_sum / steps.max(1) as f32,
            mean_top1_accuracy: top1_accuracy_sum / steps.max(1) as f32,
            validation_steps: validation
                .as_ref()
                .map(|metrics| metrics.steps)
                .unwrap_or(0),
            validation_mean_total_loss: validation.as_ref().map(|metrics| metrics.mean_total_loss),
            validation_mean_diagonal_similarity: validation
                .as_ref()
                .map(|metrics| metrics.mean_diagonal_similarity),
            validation_mean_top1_accuracy: validation
                .as_ref()
                .map(|metrics| metrics.mean_top1_accuracy),
            validation_refine_curve: validation
                .as_ref()
                .and_then(|metrics| metrics.refine_curve.clone()),
        };
        epochs.push(artifact.clone());

        if config.training.artifact_every_epochs > 0
            && epoch_index % config.training.artifact_every_epochs == 0
        {
            let artifact_path = artifact_dir(run_dir).join(format!("epoch-{epoch_index}.json"));
            let payload =
                serde_json::to_string_pretty(&artifact).context("serialize epoch artifact")?;
            fs::write(&artifact_path, payload)
                .with_context(|| format!("failed to write {}", artifact_path.display()))?;
            artifact_paths.push(artifact_path);
        }

        if config.training.checkpoint_every_epochs > 0
            && epoch_index % config.training.checkpoint_every_epochs == 0
        {
            let checkpoint_base = run_dir.join("checkpoint").join(format!("model-{epoch}"));
            BinFileRecorder::<FullPrecisionSettings>::new()
                .record(model.clone().into_record(), checkpoint_base.clone())
                .with_context(|| {
                    format!("failed to write checkpoint {}", checkpoint_base.display())
                })?;
            checkpoint_paths.push(checkpoint_base.with_extension("bin"));
        }
    }

    Ok(MultimodalTrainingReport {
        run_dir: run_dir.to_path_buf(),
        run_name: run_dir
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| "multimodal-run".to_string()),
        checkpoint_paths,
        artifact_paths,
        epochs,
        tokenizer_path: vocab_path,
    })
}

#[derive(Clone)]
struct EpochMetrics {
    steps: usize,
    mean_total_loss: f32,
    mean_diagonal_similarity: f32,
    mean_top1_accuracy: f32,
    refine_curve: Option<Vec<RefineProbeMetric>>,
}

fn load_image_text_dataset_bundle(
    config: &MultimodalTrainingConfig,
    pretrained_vocab: Option<&CharVocab>,
) -> Result<ImageTextDatasetBundle> {
    match config.data.source {
        MultimodalImageTextSource::Jsonl => {
            let vocab = pretrained_vocab
                .cloned()
                .unwrap_or(build_vocab_from_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                    config.data.include_unknown_char,
                )?);
            let train = dataset_to_vec(JsonlVisionLanguageDataset::from_jsonl(
                &config.data.manifest,
                config.data.image_size,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = config
                .data
                .validation_manifest
                .as_ref()
                .map(|manifest| {
                    JsonlVisionLanguageDataset::from_jsonl(
                        manifest,
                        config.data.image_size,
                        &vocab,
                        config.data.normalize_mean,
                        config.data.normalize_std,
                    )
                    .map(dataset_to_vec)
                })
                .transpose()?;
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: false,
                target_bank_texts: Some(unique_target_texts_from_vision_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                )?),
            })
        }
        MultimodalImageTextSource::MnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char)
                    .context("failed to build character vocabulary from MNIST label-text dataset")?
            });
            let train = dataset_to_vec(MnistVisionLanguageDataset::from_mnist(
                true,
                config.data.image_size,
                config.data.mnist.max_train_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(MnistVisionLanguageDataset::from_mnist(
                false,
                config.data.image_size,
                config.data.mnist.max_validation_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?));
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
        MultimodalImageTextSource::ImagenetteLabelText => {
            let label_texts = imagenette_target_texts();
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.imagenette.query_q_text.as_str()]
                    .into_iter()
                    .chain(label_texts.iter().map(String::as_str))
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from Imagenette label-text dataset",
                )?
            });
            let train = dataset_to_vec(ImagenetteVisionLanguageDataset::from_imagenette(
                &config.data.imagenette.root,
                &config.data.imagenette.train_dir,
                config.data.image_size,
                config.data.imagenette.max_train_records,
                &config.data.imagenette.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(
                ImagenetteVisionLanguageDataset::from_imagenette(
                    &config.data.imagenette.root,
                    &config.data.imagenette.validation_dir,
                    config.data.image_size,
                    config.data.imagenette.max_validation_records,
                    &config.data.imagenette.query_q_text,
                    &vocab,
                    config.data.normalize_mean,
                    config.data.normalize_std,
                )?,
            ));
            Ok(ImageTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(label_texts),
            })
        }
    }
}

fn load_video_text_dataset_bundle(
    config: &MultimodalVideoTrainingConfig,
    pretrained_vocab: Option<&CharVocab>,
) -> Result<VideoTextDatasetBundle> {
    match config.data.source {
        MultimodalVideoTextSource::Jsonl => {
            let vocab = pretrained_vocab
                .cloned()
                .unwrap_or(build_video_vocab_from_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                    config.data.include_unknown_char,
                )?);
            let train = dataset_to_vec(JsonlVideoLanguageDataset::from_jsonl(
                &config.data.manifest,
                config.data.image_size,
                config.data.clip_frames,
                config.model.tbptt.target_alignment_policy,
                &config.data.requested_horizons,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = config
                .data
                .validation_manifest
                .as_ref()
                .map(|manifest| {
                    JsonlVideoLanguageDataset::from_jsonl(
                        manifest,
                        config.data.image_size,
                        config.data.clip_frames,
                        config.model.tbptt.target_alignment_policy,
                        &config.data.requested_horizons,
                        &vocab,
                        config.data.normalize_mean,
                        config.data.normalize_std,
                    )
                    .map(dataset_to_vec)
                })
                .transpose()?;
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: false,
                target_bank_texts: Some(unique_target_texts_from_video_manifest(
                    &config.data.manifest,
                    config.training.max_steps_per_epoch,
                )?),
            })
        }
        MultimodalVideoTextSource::MnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from MNIST video label-text dataset",
                )?
            });
            let train = dataset_to_vec(MnistVideoLanguageDataset::from_mnist(
                true,
                config.data.image_size,
                config.data.clip_frames,
                config.data.mnist.max_train_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?);
            let validation = Some(dataset_to_vec(MnistVideoLanguageDataset::from_mnist(
                false,
                config.data.image_size,
                config.data.clip_frames,
                config.data.mnist.max_validation_records,
                &config.data.mnist.query_q_text,
                &vocab,
                config.data.normalize_mean,
                config.data.normalize_std,
            )?));
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
        MultimodalVideoTextSource::MovingMnistLabelText => {
            let vocab = pretrained_vocab.cloned().unwrap_or({
                let texts = [config.data.moving_mnist.query_q_text.as_str()]
                    .into_iter()
                    .chain([
                        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
                        "nine",
                    ])
                    .collect::<Vec<_>>();
                CharVocab::fit(texts.into_iter(), config.data.include_unknown_char).context(
                    "failed to build character vocabulary from moving-MNIST label-text dataset",
                )?
            });
            let train = dataset_to_vec(MovingMnistVideoLanguageDataset::from_moving_mnist(
                MovingMnistVideoLanguageDatasetConfig {
                    train_split: true,
                    frame_size: config.data.image_size,
                    clip_frames: config.data.clip_frames,
                    requested_horizons: &config.data.requested_horizons,
                    max_records: config.data.moving_mnist.max_train_records,
                    digit_size: config.data.moving_mnist.digit_size,
                    in_channels: config.data.moving_mnist.in_channels,
                    frame_stride: config.data.moving_mnist.frame_stride,
                    min_velocity: config.data.moving_mnist.min_velocity,
                    max_velocity: config.data.moving_mnist.max_velocity,
                    seed: config.data.moving_mnist.seed,
                    query_q_text: &config.data.moving_mnist.query_q_text,
                    vocab: &vocab,
                    normalize_mean: config.data.normalize_mean,
                    normalize_std: config.data.normalize_std,
                },
            )?);
            let validation = Some(dataset_to_vec(
                MovingMnistVideoLanguageDataset::from_moving_mnist(
                    MovingMnistVideoLanguageDatasetConfig {
                        train_split: false,
                        frame_size: config.data.image_size,
                        clip_frames: config.data.clip_frames,
                        requested_horizons: &config.data.requested_horizons,
                        max_records: config.data.moving_mnist.max_validation_records,
                        digit_size: config.data.moving_mnist.digit_size,
                        in_channels: config.data.moving_mnist.in_channels,
                        frame_stride: config.data.moving_mnist.frame_stride,
                        min_velocity: config.data.moving_mnist.min_velocity,
                        max_velocity: config.data.moving_mnist.max_velocity,
                        seed: config.data.moving_mnist.seed ^ 0xA5A5_A5A5_A5A5_A5A5,
                        query_q_text: &config.data.moving_mnist.query_q_text,
                        vocab: &vocab,
                        normalize_mean: config.data.normalize_mean,
                        normalize_std: config.data.normalize_std,
                    },
                )?,
            ));
            Ok(VideoTextDatasetBundle {
                train,
                validation,
                vocab,
                supports_batched_contrastive: true,
                target_bank_texts: Some(digit_target_texts()),
            })
        }
    }
}

fn dataset_to_vec<D>(dataset: D) -> Vec<D::Item>
where
    D: StreamDataset,
{
    (0..dataset.len())
        .filter_map(|index| dataset.get(index))
        .collect()
}

fn batch_ranges(
    total: usize,
    batch_size: usize,
    max_steps: Option<usize>,
) -> Vec<std::ops::Range<usize>> {
    let batch_size = batch_size.max(1);
    let step_limit = max_steps.unwrap_or(usize::MAX);
    let mut ranges = Vec::new();
    let mut start = 0;
    while start < total && ranges.len() < step_limit {
        let end = (start + batch_size).min(total);
        ranges.push(start..end);
        start = end;
    }
    ranges
}

fn unique_target_batches<T, F>(
    items: &[T],
    batch_size: usize,
    max_steps: Option<usize>,
    key_fn: F,
) -> Vec<Vec<usize>>
where
    F: Fn(&T) -> &[i64],
{
    let batch_size = batch_size.max(1);
    let step_limit = max_steps.unwrap_or(usize::MAX);
    let mut groups: Vec<(Vec<i64>, VecDeque<usize>)> = Vec::new();
    for (index, item) in items.iter().enumerate() {
        let key = key_fn(item).to_vec();
        if let Some((_, indices)) = groups.iter_mut().find(|(existing, _)| *existing == key) {
            indices.push_back(index);
        } else {
            let mut indices = VecDeque::new();
            indices.push_back(index);
            groups.push((key, indices));
        }
    }

    let mut batches = Vec::new();
    while groups.iter().any(|(_, indices)| !indices.is_empty()) && batches.len() < step_limit {
        let mut batch = Vec::with_capacity(batch_size);
        for (_, indices) in groups.iter_mut() {
            if batch.len() >= batch_size {
                break;
            }
            if let Some(index) = indices.pop_front() {
                batch.push(index);
            }
        }
        if batch.is_empty() {
            break;
        }
        batches.push(batch);
    }
    batches
}

fn prepare_target_bank<B: BackendTrait>(
    target_bank_texts: Option<&[String]>,
    vocab: &CharVocab,
    device: &B::Device,
) -> Option<PreparedTargetBank<B>> {
    let texts = target_bank_texts?;
    if texts.is_empty() {
        return None;
    }
    let token_sequences = texts
        .iter()
        .map(|text| {
            vocab
                .encode(text, true, true)
                .into_iter()
                .map(i64::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let max_len = token_sequences
        .iter()
        .map(Vec::len)
        .max()
        .unwrap_or(1)
        .max(1);
    let mut token_data = vec![0_i64; token_sequences.len() * max_len];
    let mut mask_data = vec![0_i64; token_sequences.len() * max_len];
    for (row, tokens) in token_sequences.iter().enumerate() {
        for (col, token) in tokens.iter().copied().enumerate() {
            token_data[row * max_len + col] = token;
            mask_data[row * max_len + col] = 1;
        }
    }
    Some(PreparedTargetBank {
        tokens: burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(token_data, [token_sequences.len(), max_len]),
            device,
        ),
        mask: Some(
            burn::tensor::Tensor::<B, 2, burn::tensor::Int>::from_data(
                burn::tensor::TensorData::new(mask_data, [token_sequences.len(), max_len]),
                device,
            )
            .greater_elem(0),
        ),
        token_sequences,
    })
}

fn target_bank_batch_for_targets<B: BackendTrait, I>(
    bank: Option<&PreparedTargetBank<B>>,
    targets: I,
    device: &B::Device,
) -> Option<TargetTextBankBatch<B>>
where
    I: IntoIterator,
    I::Item: AsRef<[i64]>,
{
    let bank = bank?;
    let mut indices = Vec::new();
    for target in targets {
        let target = target.as_ref();
        let index = bank
            .token_sequences
            .iter()
            .position(|candidate| candidate.as_slice() == target)?;
        indices.push(index as i64);
    }
    Some(TargetTextBankBatch {
        tokens: bank.tokens.clone(),
        mask: bank.mask.clone(),
        target_indices: burn::tensor::Tensor::<B, 1, burn::tensor::Int>::from_data(
            burn::tensor::TensorData::new(indices.clone(), [indices.len()]),
            device,
        ),
    })
}

fn contiguous_batch_indices(
    total: usize,
    batch_size: usize,
    max_steps: Option<usize>,
) -> Vec<Vec<usize>> {
    batch_ranges(total, batch_size, max_steps)
        .into_iter()
        .map(|range| range.collect())
        .collect()
}

fn refine_probe_steps(max_steps: usize) -> Vec<usize> {
    if max_steps == 0 {
        return vec![0];
    }
    let mut steps = vec![0];
    let mut current = 1usize;
    while current < max_steps {
        steps.push(current);
        current = current.saturating_mul(2);
    }
    if steps.last().copied() != Some(max_steps) {
        steps.push(max_steps);
    }
    steps
}

#[allow(clippy::too_many_arguments)]
fn run_image_text_validation_refine_curve<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VisionLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
) -> Result<Vec<RefineProbeMetric>> {
    let max_refine_steps = model_config
        .eval_fusion_refine_steps
        .unwrap_or(model_config.fusion_refine_steps);
    let mut curve = Vec::new();
    for refine_steps in refine_probe_steps(max_refine_steps) {
        let mut eval_config = model_config.clone();
        eval_config.eval_fusion_refine_steps = Some(refine_steps);
        let metrics = run_image_text_validation_epoch(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            &eval_config,
            batched,
            false,
        )?;
        curve.push(RefineProbeMetric {
            refine_steps,
            mean_total_loss: metrics.mean_total_loss,
            mean_top1_accuracy: metrics.mean_top1_accuracy,
        });
    }
    Ok(curve)
}

#[allow(clippy::too_many_arguments)]
fn run_video_text_validation_refine_curve<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VideoLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalVideoTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
) -> Result<Vec<RefineProbeMetric>> {
    let max_refine_steps = model_config
        .eval_fusion_refine_steps
        .unwrap_or(model_config.fusion_refine_steps);
    let mut curve = Vec::new();
    for refine_steps in refine_probe_steps(max_refine_steps) {
        let mut eval_config = model_config.clone();
        eval_config.eval_fusion_refine_steps = Some(refine_steps);
        let metrics = run_video_text_validation_epoch(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            &eval_config,
            batched,
            false,
        )?;
        curve.push(RefineProbeMetric {
            refine_steps,
            mean_total_loss: metrics.mean_total_loss,
            mean_top1_accuracy: metrics.mean_top1_accuracy,
        });
    }
    Ok(curve)
}

#[allow(clippy::too_many_arguments)]
fn run_image_text_validation_epoch<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VisionLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
    collect_refine_curve: bool,
) -> Result<EpochMetrics> {
    let mut loss_sum = 0.0_f32;
    let mut diagonal_similarity_sum = 0.0_f32;
    let mut top1_accuracy_sum = 0.0_f32;
    let mut steps = 0_usize;
    if batched {
        let batch_indices =
            if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                contiguous_batch_indices(
                    validation.len(),
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                )
            } else {
                unique_target_batches(
                    validation,
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                    |segment| &segment.payload.target_y_tokens,
                )
            };
        for batch_indices in batch_indices {
            let segments = batch_indices
                .into_iter()
                .map(|index| validation[index].clone())
                .collect::<Vec<_>>();
            let collated = collate_vision_language_segments::<B>(&segments, device);
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                segments
                    .iter()
                    .map(|segment| segment.payload.target_y_tokens.as_slice()),
                device,
            );
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation image-text batch missing stream metadata")
                })?;
            let step = multimodal_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                model.init_state(),
                model_config,
            );
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    } else {
        let mut state = model.init_state();
        for range in batch_ranges(
            validation.len(),
            1,
            config.training.max_validation_steps_per_epoch,
        ) {
            let segment = validation[range.start].clone();
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                [segment.payload.target_y_tokens.as_slice()],
                device,
            );
            let collated = collate_vision_language_segments::<B>(&[segment], device);
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation image-text batch missing stream metadata")
                })?;
            let step = multimodal_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                state,
                model_config,
            );
            state = step.forward.state.detach();
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    }
    let mean_total_loss = loss_sum / steps.max(1) as f32;
    let mean_diagonal_similarity = diagonal_similarity_sum / steps.max(1) as f32;
    let mean_top1_accuracy = top1_accuracy_sum / steps.max(1) as f32;
    let refine_curve = if collect_refine_curve {
        Some(run_image_text_validation_refine_curve(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            model_config,
            batched,
        )?)
    } else {
        None
    };
    Ok(EpochMetrics {
        steps,
        mean_total_loss,
        mean_diagonal_similarity,
        mean_top1_accuracy,
        refine_curve,
    })
}

#[allow(clippy::too_many_arguments)]
fn run_video_text_validation_epoch<B: AutodiffBackend>(
    model: &crate::model::VlJepaDragon<B>,
    target_teacher: Option<&crate::adapters::TargetTextDragonEncoderAdapter<B>>,
    prepared_target_bank: Option<&PreparedTargetBank<B>>,
    validation: &[crate::train::VideoLanguageCpuSegment],
    device: &B::Device,
    config: &MultimodalVideoTrainingConfig,
    model_config: &VlJepaDragonConfig,
    batched: bool,
    collect_refine_curve: bool,
) -> Result<EpochMetrics> {
    let mut loss_sum = 0.0_f32;
    let mut diagonal_similarity_sum = 0.0_f32;
    let mut top1_accuracy_sum = 0.0_f32;
    let mut steps = 0_usize;
    if batched {
        let batch_indices =
            if prepared_target_bank.is_some() && config.model.target_bank_loss_weight > 0.0 {
                contiguous_batch_indices(
                    validation.len(),
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                )
            } else {
                unique_target_batches(
                    validation,
                    config.training.batch_size,
                    config.training.max_validation_steps_per_epoch,
                    |segment| &segment.payload.target_y_tokens,
                )
            };
        for batch_indices in batch_indices {
            let segments = batch_indices
                .into_iter()
                .map(|index| validation[index].clone())
                .collect::<Vec<_>>();
            let collated = collate_video_language_segments::<B>(&segments, device);
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                segments
                    .iter()
                    .map(|segment| segment.payload.target_y_tokens.as_slice()),
                device,
            );
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation video-text batch missing stream metadata")
                })?;
            let step = multimodal_video_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                model.init_state(),
                model_config,
            );
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    } else {
        let mut state = model.init_state();
        for range in batch_ranges(
            validation.len(),
            1,
            config.training.max_validation_steps_per_epoch,
        ) {
            let segment = validation[range.start].clone();
            let target_bank = target_bank_batch_for_targets::<B, _>(
                prepared_target_bank,
                [segment.payload.target_y_tokens.as_slice()],
                device,
            );
            let collated = collate_video_language_segments::<B>(&[segment], device);
            let stream =
                collated.stream.first().copied().ok_or_else(|| {
                    anyhow!("validation video-text batch missing stream metadata")
                })?;
            let step = multimodal_video_eval_step(
                model,
                target_teacher,
                target_bank.as_ref(),
                collated.payload,
                &stream,
                state,
                model_config,
            );
            state = step.forward.state.detach();
            loss_sum += scalar_from_tensor(step.loss.total);
            diagonal_similarity_sum += diagonal_similarity_mean(step.loss.similarities.clone());
            top1_accuracy_sum += match target_bank {
                Some(target_bank) => {
                    labeled_top1_accuracy(step.loss.similarities, target_bank.target_indices)
                }
                None => bidirectional_top1_accuracy(step.loss.similarities),
            };
            steps = steps.saturating_add(1);
        }
    }
    let mean_total_loss = loss_sum / steps.max(1) as f32;
    let mean_diagonal_similarity = diagonal_similarity_sum / steps.max(1) as f32;
    let mean_top1_accuracy = top1_accuracy_sum / steps.max(1) as f32;
    let refine_curve = if collect_refine_curve {
        Some(run_video_text_validation_refine_curve(
            model,
            target_teacher,
            prepared_target_bank,
            validation,
            device,
            config,
            model_config,
            batched,
        )?)
    } else {
        None
    };
    Ok(EpochMetrics {
        steps,
        mean_total_loss,
        mean_diagonal_similarity,
        mean_top1_accuracy,
        refine_curve,
    })
}

fn build_vocab_from_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
    include_unknown_char: bool,
) -> Result<CharVocab> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VisionLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        texts.push(record.query_q_text);
        texts.push(record.target_y_text);
    }
    CharVocab::fit(texts.iter().map(|text| text.as_str()), include_unknown_char)
        .context("failed to build character vocabulary from multimodal manifest")
}

fn build_video_vocab_from_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
    include_unknown_char: bool,
) -> Result<CharVocab> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VideoLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        texts.push(record.query_q_text);
        texts.push(record.target_y_text);
    }
    CharVocab::fit(texts.iter().map(|text| text.as_str()), include_unknown_char)
        .context("failed to build character vocabulary from multimodal video manifest")
}

fn unique_target_texts_from_vision_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VisionLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        if !texts.iter().any(|text| text == &record.target_y_text) {
            texts.push(record.target_y_text);
        }
    }
    Ok(texts)
}

fn unique_target_texts_from_video_manifest(
    manifest: &Path,
    max_steps_per_epoch: Option<usize>,
) -> Result<Vec<String>> {
    let mut texts = Vec::new();
    for (index, line) in fs::read_to_string(manifest)
        .with_context(|| format!("failed to read {}", manifest.display()))?
        .lines()
        .enumerate()
    {
        if let Some(limit) = max_steps_per_epoch
            && index >= limit
        {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let record: VideoLanguageJsonlRecord = serde_json::from_str(line).with_context(|| {
            format!("failed to parse record {index} from {}", manifest.display())
        })?;
        if !texts.iter().any(|text| text == &record.target_y_text) {
            texts.push(record.target_y_text);
        }
    }
    Ok(texts)
}

fn digit_target_texts() -> Vec<String> {
    [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn imagenette_target_texts() -> Vec<String> {
    [
        "tench",
        "english springer",
        "cassette player",
        "chainsaw",
        "church",
        "french horn",
        "garbage truck",
        "gas pump",
        "golf ball",
        "parachute",
    ]
    .into_iter()
    .map(str::to_string)
    .collect()
}

fn scalar_from_tensor<B: BackendTrait>(tensor: burn::tensor::Tensor<B, 1>) -> f32 {
    tensor
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss vec")[0]
}

fn diagonal_similarity_mean<B: BackendTrait>(similarities: burn::tensor::Tensor<B, 2>) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let diag = rows.min(cols).max(1);
    let mut sum = 0.0_f32;
    for index in 0..diag {
        sum += values[index * cols + index];
    }
    sum / diag as f32
}

fn bidirectional_top1_accuracy<B: BackendTrait>(similarities: burn::tensor::Tensor<B, 2>) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let diag = rows.min(cols).max(1);
    let mut row_hits = 0.0_f32;
    for row in 0..diag {
        let row_slice = &values[row * cols..(row + 1) * cols];
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in row_slice.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        if best_index == row {
            row_hits += 1.0;
        }
    }
    let mut col_hits = 0.0_f32;
    for col in 0..diag {
        let mut best_index = 0;
        let mut best_value = f32::NEG_INFINITY;
        for row in 0..rows {
            let value = values[row * cols + col];
            if value > best_value {
                best_value = value;
                best_index = row;
            }
        }
        if best_index == col {
            col_hits += 1.0;
        }
    }
    (row_hits / diag as f32 + col_hits / diag as f32) / 2.0
}

fn labeled_top1_accuracy<B: BackendTrait>(
    similarities: burn::tensor::Tensor<B, 2>,
    target_indices: burn::tensor::Tensor<B, 1, burn::tensor::Int>,
) -> f32 {
    let [rows, cols] = similarities.shape().dims::<2>();
    let values = similarities
        .into_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("similarity vec");
    let labels = target_indices
        .into_data()
        .convert::<i64>()
        .into_vec::<i64>()
        .expect("label vec");
    let mut hits = 0.0_f32;
    for row in 0..rows {
        let row_slice = &values[row * cols..(row + 1) * cols];
        let mut best_index = 0_usize;
        let mut best_value = f32::NEG_INFINITY;
        for (index, value) in row_slice.iter().copied().enumerate() {
            if value > best_value {
                best_value = value;
                best_index = index;
            }
        }
        if labels.get(row).copied() == Some(best_index as i64) {
            hits += 1.0;
        }
    }
    hits / rows.max(1) as f32
}

fn default_include_unknown_char() -> bool {
    true
}

fn default_video_text_task() -> MultimodalTaskKind {
    MultimodalTaskKind::VideoText
}

fn default_true() -> bool {
    true
}

fn default_language_backend_name() -> String {
    "wgpu-fused-core".to_string()
}

fn default_digit_query_q_text() -> String {
    "which digit?".to_string()
}

fn default_imagenette_root() -> PathBuf {
    PathBuf::from("data/imagenette2-160")
}

fn default_imagenette_train_dir() -> String {
    "train".to_string()
}

fn default_imagenette_val_dir() -> String {
    "val".to_string()
}

fn default_imagenette_query_q_text() -> String {
    "which object category is shown?".to_string()
}

fn default_normalize_mean() -> [f32; 3] {
    [0.0, 0.0, 0.0]
}

fn default_normalize_std() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

fn default_checkpoint_every_epochs() -> usize {
    1
}

fn default_artifact_every_epochs() -> usize {
    1
}

fn default_seed() -> u64 {
    1337
}

fn default_weight_decay() -> f32 {
    0.0
}

fn default_run_root() -> PathBuf {
    PathBuf::from("runs/multimodal")
}

fn default_video_clip_frames() -> usize {
    2
}

fn default_requested_horizons() -> Vec<usize> {
    vec![1]
}

fn default_moving_mnist_digit_size() -> usize {
    14
}

fn default_moving_mnist_in_channels() -> usize {
    3
}

fn default_moving_mnist_frame_stride() -> usize {
    1
}

fn default_moving_mnist_min_velocity() -> f32 {
    1.0
}

fn default_moving_mnist_max_velocity() -> f32 {
    2.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_autodiff::Autodiff;
    use burn_dragon_stream::TargetAlignmentPolicy;
    use burn_ndarray::NdArray;
    use image::RgbImage;
    use tempfile::tempdir;

    type Backend = Autodiff<NdArray<f32>>;

    #[test]
    fn runs_image_text_training_and_writes_checkpoints_and_artifacts() {
        let dir = tempdir().expect("tempdir");
        let image_path = dir.path().join("sample.png");
        let mut image = RgbImage::new(8, 8);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 8) as u8, (y * 8) as u8, 127]);
        }
        image.save(&image_path).expect("save image");

        let manifest = dir.path().join("dataset.jsonl");
        fs::write(
            &manifest,
            [
                serde_json::to_string(&VisionLanguageJsonlRecord {
                    image_path: PathBuf::from("sample.png"),
                    query_q_text: "what".to_string(),
                    target_y_text: "color".to_string(),
                    source_id: 1,
                    episode_id: 1,
                    segment_id: 0,
                    step_index: 0,
                    absolute_time: 0,
                    boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
                })
                .expect("record 0"),
                serde_json::to_string(&VisionLanguageJsonlRecord {
                    image_path: PathBuf::from("sample.png"),
                    query_q_text: "what".to_string(),
                    target_y_text: "shape".to_string(),
                    source_id: 1,
                    episode_id: 1,
                    segment_id: 1,
                    step_index: 1,
                    absolute_time: 1,
                    boundary: burn_dragon_stream::StreamBoundary::Continue,
                })
                .expect("record 1"),
            ]
            .join("\n"),
        )
        .expect("write manifest");

        let mut config = MultimodalTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(2);

        let run_dir = dir.path().join("run");
        let report = run_image_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run multimodal training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
        assert!(run_dir.join("multimodal_vl_jepa_config.json").is_file());
    }

    #[test]
    fn train_backend_creates_named_run_under_backend_root() {
        let dir = tempdir().expect("tempdir");
        let image_path = dir.path().join("sample.png");
        let mut image = RgbImage::new(8, 8);
        for (x, y, pixel) in image.enumerate_pixels_mut() {
            *pixel = image::Rgb([(x * 8) as u8, (y * 8) as u8, 127]);
        }
        image.save(&image_path).expect("save image");

        let manifest = dir.path().join("dataset.jsonl");
        fs::write(
            &manifest,
            serde_json::to_string(&VisionLanguageJsonlRecord {
                image_path: PathBuf::from("sample.png"),
                query_q_text: "what".to_string(),
                target_y_text: "color".to_string(),
                source_id: 1,
                episode_id: 1,
                segment_id: 0,
                step_index: 0,
                absolute_time: 0,
                boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
            })
            .expect("record"),
        )
        .expect("write manifest");

        let mut config = MultimodalTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(1);
        config.training.run_root = dir.path().join("runs");

        let report = train_backend::<Backend, _>(&config, "cpu", |_| {}).expect("train backend");
        assert!(
            report
                .run_dir
                .starts_with(dir.path().join("runs").join("cpu"))
        );
        assert!(!report.run_name.is_empty());
        assert!(report.run_dir.join("checkpoint").is_dir());
    }

    #[test]
    fn runs_video_text_training_and_writes_checkpoints_and_artifacts() {
        let dir = tempdir().expect("tempdir");
        let mut records = Vec::new();
        for frame in 0..6 {
            let image_path = dir.path().join(format!("frame-{frame}.png"));
            let mut image = RgbImage::new(8, 8);
            for (x, y, pixel) in image.enumerate_pixels_mut() {
                *pixel = image::Rgb([(x * 8 + frame) as u8, (y * 8) as u8, 127]);
            }
            image.save(&image_path).expect("save image");
        }
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-0.png"),
            query_q_text: "look".into(),
            target_y_text: "zero".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 0,
            step_index: 0,
            absolute_time: 0,
            boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-1.png"),
            query_q_text: "look".into(),
            target_y_text: "one".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 1,
            step_index: 1,
            absolute_time: 1,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-2.png"),
            query_q_text: "look".into(),
            target_y_text: "two".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 2,
            step_index: 2,
            absolute_time: 2,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-3.png"),
            query_q_text: "look".into(),
            target_y_text: "three".into(),
            source_id: 1,
            episode_id: 1,
            segment_id: 3,
            step_index: 3,
            absolute_time: 3,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-4.png"),
            query_q_text: "new".into(),
            target_y_text: "four".into(),
            source_id: 1,
            episode_id: 2,
            segment_id: 0,
            step_index: 0,
            absolute_time: 4,
            boundary: burn_dragon_stream::StreamBoundary::ResetEpisode,
        });
        records.push(VideoLanguageJsonlRecord {
            frame_path: PathBuf::from("frame-5.png"),
            query_q_text: "new".into(),
            target_y_text: "five".into(),
            source_id: 1,
            episode_id: 2,
            segment_id: 1,
            step_index: 1,
            absolute_time: 5,
            boundary: burn_dragon_stream::StreamBoundary::Continue,
        });

        let manifest = dir.path().join("video.jsonl");
        fs::write(
            &manifest,
            records
                .iter()
                .map(|record| serde_json::to_string(record).expect("record json"))
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .expect("write manifest");

        let mut config = MultimodalVideoTrainingConfig::default();
        config.model.vision.image_size = 8;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.model.tbptt.target_alignment_policy = TargetAlignmentPolicy::VariableFuture;
        config.data.manifest = manifest;
        config.data.image_size = 8;
        config.data.clip_frames = 2;
        config.data.requested_horizons = vec![1, 2];
        config.training.epochs = 1;
        config.training.max_steps_per_epoch = Some(2);

        let run_dir = dir.path().join("video-run");
        let report = run_video_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run multimodal video training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "video-run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
    }

    #[test]
    fn runs_mnist_video_text_training_and_writes_checkpoints_and_artifacts() {
        type Backend = Autodiff<NdArray<f32>>;
        let dir = tempdir().expect("tempdir");
        let mut config = MultimodalVideoTrainingConfig::default();
        config.model.vision.image_size = 32;
        config.model.vision.patch_size = 4;
        config.model.vision.embed_dim = 16;
        config.model.vision.projection_dim = 16;
        config.model.vision.projection_hidden_dim = 16;
        config.model.vision.steps = 1;
        config.model.query_text.n_layer = 2;
        config.model.query_text.n_embd = 16;
        config.model.query_text.n_head = 2;
        config.model.target_text.n_layer = 2;
        config.model.target_text.n_embd = 16;
        config.model.target_text.n_head = 2;
        config.model.fusion.n_layer = 2;
        config.model.fusion.n_embd = 16;
        config.model.fusion.n_head = 2;
        config.model.fusion_dim = 16;
        config.model.target_dim = 16;
        config.data.source = MultimodalVideoTextSource::MnistLabelText;
        config.data.clip_frames = 2;
        config.data.mnist.max_train_records = Some(16);
        config.data.mnist.max_validation_records = Some(8);
        config.training.epochs = 1;
        config.training.batch_size = 4;
        config.training.max_steps_per_epoch = Some(2);
        config.training.max_validation_steps_per_epoch = Some(1);

        let run_dir = dir.path().join("mnist-video-run");
        let report = run_video_text_training_backend::<Backend, _>(&config, &run_dir, |_| {})
            .expect("run MNIST video-text training");

        assert_eq!(report.epochs.len(), 1);
        assert!(!report.checkpoint_paths.is_empty());
        assert!(!report.artifact_paths.is_empty());
        assert!(report.tokenizer_path.is_file());
        assert_eq!(report.run_name, "mnist-video-run");
        assert!(training_runtime_snapshot_path(&run_dir).is_file());
    }

    #[test]
    fn unique_target_batches_distribute_duplicate_targets_across_batches() {
        let items = vec![
            vec![1_i64],
            vec![2_i64],
            vec![1_i64],
            vec![2_i64],
            vec![3_i64],
            vec![3_i64],
        ];
        let batches = unique_target_batches(&items, 3, None, |tokens| tokens);
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0], vec![0, 1, 4]);
        assert_eq!(batches[1], vec![2, 3, 5]);
    }
}
