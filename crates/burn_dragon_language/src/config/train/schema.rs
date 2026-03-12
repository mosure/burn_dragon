use std::path::PathBuf;

use super::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct DatasetConfig {
    pub cache_dir: PathBuf,
    #[serde(default = "default_train_split_ratio")]
    pub train_split_ratio: f32,
    #[serde(flatten)]
    pub source: DatasetSourceConfig,
    #[serde(default)]
    pub tokenizer: TokenizerConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DatasetSourceConfig {
    Shakespeare {
        #[serde(default)]
        url: Option<String>,
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
    pub validation_files: Vec<String>,
    #[serde(default = "default_hf_text_fields")]
    pub text_fields: Vec<String>,
    #[serde(default = "default_hf_field_separator")]
    pub field_separator: String,
    #[serde(default)]
    pub template: Option<String>,
    #[serde(default)]
    pub max_records: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum HuggingFaceRecordFormat {
    #[default]
    Jsonl,
    Text,
    Parquet,
    Csv,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingHyperparameters {
    pub block_size: usize,
    pub batch_size: usize,
    #[serde(default = "default_gradient_accumulation_steps")]
    pub gradient_accumulation_steps: usize,
    #[serde(default)]
    pub target_effective_batch_size: Option<usize>,
    #[serde(default)]
    pub epochs: Option<usize>,
    pub max_iters: usize,
    pub log_frequency: usize,
    #[serde(default)]
    pub fast_train: bool,
    #[serde(default = "default_context_strategy")]
    pub context_strategy: ContextStrategyConfig,
    #[serde(default)]
    pub gdpo: Option<burn_dragon_train::GdpoConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingConfig {
    pub dataset: DatasetConfig,
    pub training: TrainingHyperparameters,
    pub optimizer: burn_dragon_train::OptimizerConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub wgpu: burn_dragon_train::WgpuRuntimeConfig,
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

fn default_gradient_accumulation_steps() -> usize {
    1
}
