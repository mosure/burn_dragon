use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use toml::Value;

use crate::tokenizer::TokenizerConfig;

use super::{
    GenerationConfig, GdpoHardGate, ModelOverrides, TrainingHyperparameters, WgpuRuntimeConfig,
};

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
pub struct OptimizerConfig {
    pub learning_rate: f64,
    pub weight_decay: f32,
    #[serde(default)]
    pub lr_schedule: Option<LearningRateScheduleConfig>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_norm: Option<f32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grad_clip_value: Option<f32>,
}

impl OptimizerConfig {
    pub fn validate(&self) -> Result<()> {
        if self.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }
        if let Some(clip) = self.grad_clip_norm {
            if clip <= 0.0 {
                return Err(anyhow!("optimizer.grad_clip_norm must be > 0"));
            }
        }
        if let Some(clip) = self.grad_clip_value {
            if clip <= 0.0 {
                return Err(anyhow!("optimizer.grad_clip_value must be > 0"));
            }
        }
        if self.grad_clip_norm.is_some() && self.grad_clip_value.is_some() {
            return Err(anyhow!(
                "optimizer.grad_clip_norm and optimizer.grad_clip_value are mutually exclusive"
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum LearningRateScheduleConfig {
    Constant {
        #[serde(default)]
        initial_lr: Option<f64>,
    },
    Cosine {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        min_lr: Option<f64>,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Linear {
        #[serde(default)]
        initial_lr: Option<f64>,
        final_lr: f64,
        #[serde(default)]
        num_iters: Option<usize>,
    },
    Exponential {
        #[serde(default)]
        initial_lr: Option<f64>,
        gamma: f64,
    },
    Step {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default = "default_step_gamma")]
        gamma: f64,
        #[serde(default)]
        step_size: Option<usize>,
    },
    Noam {
        #[serde(default)]
        initial_lr: Option<f64>,
        #[serde(default)]
        warmup_steps: Option<usize>,
        #[serde(default)]
        model_size: Option<usize>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrainingConfig {
    pub dataset: DatasetConfig,
    pub training: TrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    pub generation: GenerationConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    #[serde(default)]
    pub model: ModelOverrides,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.block_size == 0 {
            return Err(anyhow!("training.block_size must be > 0"));
        }
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if let Some(epochs) = self.training.epochs {
            if epochs == 0 {
                return Err(anyhow!("training.epochs must be > 0"));
            }
        }
        self.optimizer.validate()?;
        if !(0.0 < self.dataset.train_split_ratio && self.dataset.train_split_ratio <= 1.0) {
            return Err(anyhow!(
                "dataset.train_split_ratio must be in (0, 1] (got {})",
                self.dataset.train_split_ratio
            ));
        }
        if let Some(max_tokens) = self.generation.max_tokens {
            if max_tokens <= 0 {
                return Err(anyhow!("generation.max_tokens must be > 0"));
            }
        }
        if self.generation.temperature <= 0.0 {
            return Err(anyhow!("generation.temperature must be > 0"));
        }
        if let Some(top_k) = self.generation.top_k {
            if top_k == 0 {
                return Err(anyhow!("generation.top_k must be > 0"));
            }
        }

        match &self.dataset.source {
            DatasetSourceConfig::HuggingFace(config) => {
                if config.repo_id.trim().is_empty() {
                    return Err(anyhow!("dataset.repo_id must not be empty"));
                }
                if config.train_files.is_empty() {
                    return Err(anyhow!("dataset.train_files must not be empty"));
                }
                if config.text_fields.is_empty() {
                    return Err(anyhow!("dataset.text_fields must not be empty"));
                }
            }
            DatasetSourceConfig::DeepMath { max_records, .. }
            | DatasetSourceConfig::TinyChat { max_records, .. }
            | DatasetSourceConfig::WebscaleRl { max_records, .. }
            | DatasetSourceConfig::PoetryFoundation { max_records, .. } => {
                if matches!(max_records, Some(0)) {
                    return Err(anyhow!("dataset.max_records must be > 0 when set"));
                }
            }
            DatasetSourceConfig::Shakespeare { .. } => {}
        }

        if let Some(gdpo) = &self.training.gdpo {
            if gdpo.enabled {
                if gdpo.group_size == 0 {
                    return Err(anyhow!("training.gdpo.group_size must be > 0"));
                }
                if gdpo.hard_weight < 0.0 {
                    return Err(anyhow!("training.gdpo.hard_weight must be >= 0"));
                }
                if gdpo.easy_weight < 0.0 {
                    return Err(anyhow!("training.gdpo.easy_weight must be >= 0"));
                }
                if gdpo.policy_weight < 0.0 {
                    return Err(anyhow!("training.gdpo.policy_weight must be >= 0"));
                }
                if gdpo.policy_clip_range < 0.0 {
                    return Err(anyhow!("training.gdpo.policy_clip_range must be >= 0"));
                }
                if let GdpoHardGate::Percentile { quantile } = gdpo.hard_gate {
                    if !(0.0..=1.0).contains(&quantile) {
                        return Err(anyhow!(
                            "training.gdpo.hard_gate.quantile must be in [0, 1] (got {})",
                            quantile
                        ));
                    }
                }
            }
        }

        if let Some(n_layer) = self.model.n_layer {
            if n_layer == 0 {
                return Err(anyhow!("model.n_layer must be > 0 when set"));
            }
        }
        if let Some(n_embd) = self.model.n_embd {
            if n_embd == 0 {
                return Err(anyhow!("model.n_embd must be > 0 when set"));
            }
        }
        if let Some(n_head) = self.model.n_head {
            if n_head == 0 {
                return Err(anyhow!("model.n_head must be > 0 when set"));
            }
        }
        if let Some(multiplier) = self.model.mlp_internal_dim_multiplier {
            if multiplier == 0 {
                return Err(anyhow!("model.mlp_internal_dim_multiplier must be > 0 when set"));
            }
        }
        if let Some(dropout) = self.model.dropout {
            if dropout < 0.0 {
                return Err(anyhow!("model.dropout must be >= 0"));
            }
        }
        if let Some(block_size) = self.model.block_size {
            if block_size == 0 {
                return Err(anyhow!("model.block_size must be > 0 when set"));
            }
        }

        if let Some(schedule) = &self.optimizer.lr_schedule {
            match schedule {
                LearningRateScheduleConfig::Constant { initial_lr }
                | LearningRateScheduleConfig::Cosine { initial_lr, .. }
                | LearningRateScheduleConfig::Linear { initial_lr, .. }
                | LearningRateScheduleConfig::Exponential { initial_lr, .. }
                | LearningRateScheduleConfig::Step { initial_lr, .. }
                | LearningRateScheduleConfig::Noam { initial_lr, .. } => {
                    if matches!(initial_lr.as_ref(), Some(value) if *value <= 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.initial_lr must be > 0"));
                    }
                }
            }

            match schedule {
                LearningRateScheduleConfig::Cosine { min_lr, num_iters, .. } => {
                    if matches!(min_lr.as_ref(), Some(value) if *value < 0.0) {
                        return Err(anyhow!("optimizer.lr_schedule.min_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Linear { final_lr, num_iters, .. } => {
                    if *final_lr < 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.final_lr must be >= 0"));
                    }
                    if matches!(num_iters, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.num_iters must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Exponential { gamma, .. } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Step { gamma, step_size, .. } => {
                    if *gamma <= 0.0 {
                        return Err(anyhow!("optimizer.lr_schedule.gamma must be > 0"));
                    }
                    if matches!(step_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.step_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Noam { warmup_steps, model_size, .. } => {
                    if matches!(warmup_steps, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.warmup_steps must be > 0"));
                    }
                    if matches!(model_size, Some(0)) {
                        return Err(anyhow!("optimizer.lr_schedule.model_size must be > 0"));
                    }
                }
                LearningRateScheduleConfig::Constant { .. } => {}
            }
        }

        Ok(())
    }
}

pub fn load_training_config(paths: &[PathBuf]) -> Result<TrainingConfig> {
    if paths.is_empty() {
        return Err(anyhow!("at least one configuration path is required"));
    }

    let mut iter = paths.iter();
    let first_path = iter
        .next()
        .ok_or_else(|| anyhow!("configuration iterator unexpectedly empty"))?;
    let mut value = load_value(first_path)?;

    for path in iter {
        let overlay = load_value(path)?;
        merge_values(&mut value, overlay);
    }

    value
        .try_into::<TrainingConfig>()
        .map_err(|err| anyhow!(err))
}

fn load_value(path: &Path) -> Result<Value> {
    let content = fs::read_to_string(path)
        .with_context(|| format!("failed to read configuration file {}", path.display()))?;
    let table: toml::value::Table = toml::from_str(&content)
        .with_context(|| format!("failed to parse {} as TOML", path.display()))?;
    Ok(Value::Table(table))
}

fn merge_values(base: &mut Value, overlay: Value) {
    match (base, overlay) {
        (Value::Table(base_table), Value::Table(overlay_table)) => {
            if let Some(Value::String(overlay_type)) = overlay_table.get("type") {
                let type_changed = match base_table.get("type") {
                    Some(Value::String(base_type)) => base_type != overlay_type,
                    Some(_) => true,
                    None => !base_table.is_empty(),
                };
                if type_changed {
                    base_table.clear();
                }
            }
            for (key, overlay_value) in overlay_table {
                match base_table.get_mut(&key) {
                    Some(base_value) => merge_values(base_value, overlay_value),
                    None => {
                        base_table.insert(key, overlay_value);
                    }
                }
            }
        }
        (base_value, overlay_value) => {
            *base_value = overlay_value;
        }
    }
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

fn default_step_gamma() -> f64 {
    0.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ContextStrategyConfig;
    use tempfile::tempdir;

    fn write_config(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let path = dir.join(name);
        let trimmed_lines: Vec<&str> = contents.lines().map(|line| line.trim_start()).collect();
        let mut formatted = trimmed_lines.join("\n");
        if formatted.starts_with('\n') {
            formatted = formatted.trim_start_matches('\n').to_string();
        }
        fs::write(&path, formatted).expect("write config");
        path
    }

    #[test]
    fn load_merges_in_order() {
        let dir = tempdir().expect("tempdir");

        let base_contents = [
            "[dataset]",
            "cache_dir = \"data\"",
            "train_split_ratio = 0.8",
            "type = \"shakespeare\"",
            "",
            "[training]",
            "block_size = 256",
            "batch_size = 16",
            "max_iters = 1000",
            "log_frequency = 50",
            "",
            "[optimizer]",
            "learning_rate = 0.001",
            "weight_decay = 0.05",
            "",
            "[optimizer.lr_schedule]",
            "type = \"cosine\"",
            "min_lr = 0.00005",
            "num_iters = 100",
            "",
            "[generation]",
            "prompt = \"Base prompt\"",
            "max_tokens = 64",
            "temperature = 0.9",
            "top_k = 4",
            "",
            "[model]",
            "n_layer = 6",
            "n_embd = 256",
            "n_head = 4",
            "mlp_internal_dim_multiplier = 4",
            "dropout = 0.1",
            "fused_kernels = false",
            "rotary_embedding = \"alibi\"",
        ]
        .join("\n");
        let base = write_config(dir.path(), "base.toml", &base_contents);

        let override_contents = [
            "[training]",
            "max_iters = 2000",
            "",
            "[optimizer]",
            "learning_rate = 0.0005",
            "",
            "[optimizer.lr_schedule]",
            "type = \"linear\"",
            "final_lr = 0.0002",
            "num_iters = 50",
            "",
            "[model]",
            "n_embd = 320",
            "fused_kernels = true",
            "block_size = 256",
        ]
        .join("\n");
        let override_cfg = write_config(dir.path(), "override.toml", &override_contents);

        let config = load_training_config(&[base, override_cfg]).expect("load config");

        assert_eq!(
            config.training,
            TrainingHyperparameters {
                block_size: 256,
                batch_size: 16,
                epochs: None,
                max_iters: 2000,
                log_frequency: 50,
                fast_train: false,
                context_strategy: ContextStrategyConfig::Infinite,
                gdpo: None,
            }
        );
        assert!((config.optimizer.learning_rate - 0.0005).abs() < f64::EPSILON);
        assert!((config.optimizer.weight_decay - 0.05).abs() < f32::EPSILON);
        assert_eq!(
            config.optimizer.lr_schedule,
            Some(LearningRateScheduleConfig::Linear {
                initial_lr: None,
                final_lr: 0.0002,
                num_iters: Some(50),
            })
        );
        assert_eq!(config.dataset.tokenizer, TokenizerConfig::default());
        assert!((config.dataset.train_split_ratio - 0.8).abs() < f32::EPSILON);
        assert_eq!(
            config.dataset.source,
            DatasetSourceConfig::Shakespeare { url: None }
        );
        assert_eq!(config.generation.max_tokens, Some(64));
        assert_eq!(
            config.training.context_strategy,
            ContextStrategyConfig::Infinite
        );
        assert_eq!(
            config.generation.context_strategy,
            ContextStrategyConfig::Infinite
        );
        assert_eq!(config.model.n_layer, Some(6));
        assert_eq!(config.model.n_embd, Some(320));
        assert_eq!(config.model.n_head, Some(4));
        assert_eq!(config.model.mlp_internal_dim_multiplier, Some(4));
        assert_eq!(config.model.dropout, Some(0.1));
        assert_eq!(config.model.fused_kernels, Some(true));
        assert_eq!(config.model.block_size, Some(256));
        assert_eq!(config.model.rotary_embedding, Some(crate::positional::RotaryEmbedding::Alibi));
    }

    #[test]
    fn schedule_constant_round_trips() {
        let text = r#"
            learning_rate = 0.002
            weight_decay = 0.1

            [lr_schedule]
            type = "constant"
        "#;
        let optimizer: OptimizerConfig = toml::from_str(text).expect("parse optimizer config");
        assert_eq!(
            optimizer.lr_schedule,
            Some(LearningRateScheduleConfig::Constant { initial_lr: None })
        );
    }

    #[test]
    fn huggingface_dataset_config_parses() {
        let text = r#"
            cache_dir = "data"
            train_split_ratio = 0.75
            type = "hugging_face"
            repo_id = "zwhe99/DeepMath-103K"
            revision = "main"
            format = "parquet"
            train_files = [
                "data/train-00000-of-00010.parquet",
                "data/train-00001-of-00010.parquet",
            ]
            validation_files = []
            text_fields = ["question", "final_answer"]
            field_separator = "\n\n"
            template = "{question}\n{final_answer}"
            max_records = 1000
        "#;
        let dataset: DatasetConfig = toml::from_str(text).expect("parse dataset config");
        assert_eq!(dataset.train_split_ratio, 0.75);
        match &dataset.source {
            DatasetSourceConfig::HuggingFace(hf) => {
                assert_eq!(hf.repo_id, "zwhe99/DeepMath-103K");
                assert_eq!(hf.revision.as_deref(), Some("main"));
                assert_eq!(hf.format, HuggingFaceRecordFormat::Parquet);
                assert_eq!(
                    hf.train_files,
                    vec![
                        "data/train-00000-of-00010.parquet".to_string(),
                        "data/train-00001-of-00010.parquet".to_string()
                    ]
                );
                assert!(hf.validation_files.is_empty());
                assert_eq!(hf.text_fields, vec!["question", "final_answer"]);
                assert_eq!(hf.field_separator, "\n\n");
                assert_eq!(hf.template.as_deref(), Some("{question}\n{final_answer}"));
                assert_eq!(hf.max_records, Some(1000));
            }
            other => panic!("unexpected dataset source: {other:?}"),
        }
    }
}
