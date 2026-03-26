#![cfg(feature = "train")]

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use burn_dragon_checkpoint::write_json_snapshot;
use burn_dragon_train::api::config::{RunLayoutConfig, WgpuRuntimeConfig};
use burn_dragon_train::api::expert::train::pipeline::{
    resolve_backend_partition_run_root, resolve_run_root_for_config_paths,
};
use serde::{Deserialize, Serialize};

use crate::config::VlJepaDragonConfig;
use crate::config_io::{load_merged_config, load_merged_value};

const TRAINING_CONFIG_SNAPSHOT_FILE_NAME: &str = "multimodal_training_config.json";
const TOKENIZER_SNAPSHOT_FILE_NAME: &str = "multimodal_tokenizer.json";

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
    pub run_layout: RunLayoutConfig,
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
    pub run_layout: RunLayoutConfig,
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

pub(crate) fn resolve_multimodal_backend_run_root(
    run_layout: &RunLayoutConfig,
    legacy_run_root: &Path,
    backend_name: &str,
) -> PathBuf {
    let base_run_root = if run_layout != &RunLayoutConfig::default() {
        resolve_run_root_for_config_paths("multimodal", run_layout, &[])
    } else {
        legacy_run_root.to_path_buf()
    };
    resolve_backend_partition_run_root(&base_run_root, backend_name)
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
    let probe = serde_json::from_value::<TaskProbe>(merged.clone())
        .context("failed to inspect multimodal task kind")?;
    match probe.task {
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
    "what object is shown?".to_string()
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
