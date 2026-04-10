use std::path::PathBuf;

use burn_dragon_core::{LanguageModuleLrScaleTarget, SequenceKernelConfig};

use super::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct DatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(default)]
    pub validation: Option<ValidationDatasetConfig>,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ValidationDatasetConfig {
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    #[serde(default)]
    pub train_split_ratio: Option<f32>,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatasetSourceConfig {
    Shakespeare {
        #[serde(default)]
        url: Option<String>,
    },
    LocalText {
        path: PathBuf,
    },
    HuggingFace(HuggingFaceDatasetConfig),
    DeepMath {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    TinyChat {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    WebscaleRl {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    PoetryFoundation {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    #[serde(alias = "openwebtext_gpt2")]
    OpenWebTextGpt2 {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    NemotronClimbMix {
        #[serde(default)]
        revision: Option<String>,
        #[serde(default)]
        max_records: Option<usize>,
    },
    UniversalityManifest {
        manifest: PathBuf,
    },
    UniversalityNca {
        config: PathBuf,
    },
}

impl Default for DatasetSourceConfig {
    fn default() -> Self {
        Self::Shakespeare { url: None }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct HuggingFaceDatasetConfig {
    pub repo_id: String,
    #[serde(default)]
    pub token: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub format: HuggingFaceRecordFormat,
    #[serde(default = "default_hf_train_files")]
    pub train_files: Vec<String>,
    #[serde(default)]
    pub auto_discover_train_files: bool,
    #[serde(default)]
    pub validation_files: Vec<String>,
    #[serde(default = "default_hf_text_fields")]
    pub text_fields: Vec<String>,
    #[serde(default)]
    pub sequence_field: Option<String>,
    #[serde(default = "default_hf_field_separator")]
    pub field_separator: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HuggingFaceRecordFormat {
    #[default]
    Jsonl,
    Text,
    Parquet,
    Csv,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct InitTransferConfig {
    #[serde(default)]
    pub interface_checkpoint_path: Option<PathBuf>,
    #[serde(default)]
    pub interface_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub preserve_interface_input_embedding: bool,
    #[serde(default)]
    pub preserve_interface_output_head: bool,
    #[serde(default)]
    pub interface_output_head_blend_alpha: Option<f32>,
    #[serde(default)]
    pub backbone_blend_alpha: Option<f32>,
    #[serde(default)]
    pub decoder_blend_alpha: Option<f32>,
    #[serde(default)]
    pub norm_blend_alpha: Option<f32>,
    #[serde(default)]
    pub backbone_grad_scale: Option<f32>,
    #[serde(default)]
    pub backbone_grad_scale_steps: Option<usize>,
    #[serde(default)]
    pub fresh_top_layers: Option<usize>,
    #[serde(default)]
    pub preserve_fresh_decoder: bool,
    #[serde(default)]
    pub preserve_fresh_norm: bool,
    #[serde(default)]
    pub match_fresh_rms: bool,
}

impl Default for InitTransferConfig {
    fn default() -> Self {
        Self {
            interface_checkpoint_path: None,
            interface_checkpoint_epoch: None,
            preserve_interface_input_embedding: false,
            preserve_interface_output_head: false,
            interface_output_head_blend_alpha: None,
            backbone_blend_alpha: None,
            decoder_blend_alpha: None,
            norm_blend_alpha: None,
            backbone_grad_scale: None,
            backbone_grad_scale_steps: None,
            fresh_top_layers: None,
            preserve_fresh_decoder: false,
            preserve_fresh_norm: false,
            match_fresh_rms: false,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModuleLrScaleScheduleConfig {
    pub final_scale: f32,
    #[serde(default)]
    pub start_fraction: f32,
    #[serde(default = "default_module_lr_scale_schedule_end_fraction")]
    pub end_fraction: f32,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModuleLrScaleEntry {
    pub target: LanguageModuleLrScaleTarget,
    pub scale: f32,
    #[serde(default)]
    pub schedule: Option<ModuleLrScaleScheduleConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContinualBackpropTarget {
    #[default]
    SharedLowrankLatents,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum ContinualBackpropLrCoupling {
    #[default]
    None,
    GlobalRatio,
    TargetGroupRatio,
}

fn default_continual_backprop_utility_decay() -> f32 {
    0.99
}

fn default_continual_backprop_replacement_rate() -> f32 {
    1.0e-4
}

fn default_continual_backprop_maturity_steps() -> usize {
    100
}

fn default_continual_backprop_sample_interval_steps() -> usize {
    8
}

fn default_continual_backprop_replace_interval_steps() -> usize {
    64
}

fn default_continual_backprop_utility_epsilon() -> f32 {
    1.0e-6
}

fn default_continual_backprop_lr_coupling_power() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct ContinualBackpropConfig {
    pub enabled: bool,
    pub target: ContinualBackpropTarget,
    #[serde(default = "default_continual_backprop_utility_decay")]
    pub utility_decay: f32,
    #[serde(default = "default_continual_backprop_replacement_rate")]
    pub replacement_rate: f32,
    #[serde(default = "default_continual_backprop_maturity_steps")]
    pub maturity_steps: usize,
    #[serde(default = "default_continual_backprop_sample_interval_steps")]
    pub sample_interval_steps: usize,
    #[serde(default = "default_continual_backprop_replace_interval_steps")]
    pub replace_interval_steps: usize,
    #[serde(default = "default_continual_backprop_utility_epsilon")]
    pub utility_epsilon: f32,
    #[serde(default)]
    pub lr_coupling: ContinualBackpropLrCoupling,
    #[serde(default = "default_continual_backprop_lr_coupling_power")]
    pub lr_coupling_power: f32,
}

impl Default for ContinualBackpropConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            target: ContinualBackpropTarget::default(),
            utility_decay: default_continual_backprop_utility_decay(),
            replacement_rate: default_continual_backprop_replacement_rate(),
            maturity_steps: default_continual_backprop_maturity_steps(),
            sample_interval_steps: default_continual_backprop_sample_interval_steps(),
            replace_interval_steps: default_continual_backprop_replace_interval_steps(),
            utility_epsilon: default_continual_backprop_utility_epsilon(),
            lr_coupling: ContinualBackpropLrCoupling::default(),
            lr_coupling_power: default_continual_backprop_lr_coupling_power(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    #[serde(default)]
    pub tbptt_chunk_size: Option<usize>,
    #[serde(default)]
    pub tbptt_persist_across_steps: bool,
    #[serde(default)]
    pub min_logical_block_size: Option<usize>,
    pub batch_size: usize,
    #[serde(default = "default_training_seed")]
    pub seed: u64,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub target_effective_batch_size: Option<usize>,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    #[serde(default = "default_checkpoint_interval_iters")]
    pub checkpoint_interval_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub launch_mode: burn_dragon_train::train::pipeline::TrainingLaunchMode,
    #[serde(default)]
    pub resume_run_dir: Option<PathBuf>,
    #[serde(default)]
    pub resume_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub init_checkpoint_path: Option<PathBuf>,
    #[serde(default)]
    pub init_checkpoint_epoch: Option<usize>,
    #[serde(default)]
    pub init_transfer: InitTransferConfig,
    #[serde(default)]
    pub continual_backprop: ContinualBackpropConfig,
    #[serde(default)]
    pub module_lr_scales: Vec<ModuleLrScaleEntry>,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
    #[serde(default)]
    pub sequence_kernel_override: Option<SequenceKernelConfig>,
    #[serde(default)]
    pub gdpo: Option<burn_dragon_train::GdpoConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingConfig {
    pub dataset: DatasetConfig,
    pub training: TrainingHyperparameters,
    pub optimizer: burn_dragon_train::OptimizerConfig,
    #[serde(default)]
    pub parallel: burn_dragon_train::ParallelConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub wgpu: burn_dragon_train::WgpuRuntimeConfig,
    #[serde(default)]
    pub run_layout: burn_dragon_train::RunLayoutConfig,
    #[serde(default)]
    pub model: ModelOverrides,
}

fn default_train_split_ratio() -> f32 {
    0.9
}

fn default_hf_train_files() -> Vec<String> {
    vec!["train.jsonl".to_string()]
}

fn default_hf_text_fields() -> Vec<String> {
    vec!["text".to_string()]
}

fn default_hf_field_separator() -> String {
    "\n".to_string()
}

fn default_context_strategy() -> ContextStrategyConfig {
    ContextStrategyConfig::Infinite
}

fn default_module_lr_scale_schedule_end_fraction() -> f32 {
    1.0
}

fn default_training_seed() -> u64 {
    1337
}

fn default_gradient_accumulation_steps() -> usize {
    1
}

fn default_checkpoint_interval_iters() -> usize {
    2_000
}
