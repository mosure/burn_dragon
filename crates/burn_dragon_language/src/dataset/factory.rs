use anyhow::{Context, Result};

use crate::config::{
    DatasetConfig, DatasetSourceConfig, HuggingFaceDatasetConfig, HuggingFaceRecordFormat,
    TrainingHyperparameters,
};

use super::{
    Dataset, HuggingFaceDataset, LocalTextDataset, ShakespeareDataset, UniversalityDataset,
};

pub fn build_dataset(
    cfg: &DatasetConfig,
    training: &TrainingHyperparameters,
) -> Result<(Dataset, String)> {
    let dataset = match &cfg.source {
        DatasetSourceConfig::Shakespeare { url } => Dataset::from_shakespeare(
            ShakespeareDataset::new_with_source(
                &cfg.cache_dir,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
                url.as_deref(),
            )
            .with_context(|| "failed to prepare Shakespeare dataset")?,
        ),
        DatasetSourceConfig::LocalText { path } => Dataset::from_local_text(
            LocalTextDataset::new(
                &cfg.cache_dir,
                path,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
            )
            .with_context(|| format!("failed to prepare local text dataset {}", path.display()))?,
        ),
        DatasetSourceConfig::HuggingFace(hf_cfg) => Dataset::from_huggingface(
            HuggingFaceDataset::new(
                &cfg.cache_dir,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
                hf_cfg,
            )
            .with_context(|| {
                format!("failed to prepare Hugging Face dataset {}", hf_cfg.repo_id)
            })?,
        ),
        DatasetSourceConfig::DeepMath {
            revision,
            max_records,
        } => {
            let config = deepmath_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare DeepMath-103K dataset")?,
            )
        }
        DatasetSourceConfig::TinyChat {
            revision,
            max_records,
        } => {
            let config = tinychat_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare TinyChat dataset")?,
            )
        }
        DatasetSourceConfig::WebscaleRl {
            revision,
            max_records,
        } => {
            let config = webscale_rl_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare Webscale-RL dataset")?,
            )
        }
        DatasetSourceConfig::PoetryFoundation {
            revision,
            max_records,
        } => {
            let config = poetry_foundation_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare Poetry Foundation Poems dataset")?,
            )
        }
        DatasetSourceConfig::OpenWebTextGpt2 {
            revision,
            max_records,
        } => {
            let config = openwebtext_gpt2_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare OpenWebText GPT-2 dataset")?,
            )
        }
        DatasetSourceConfig::NemotronClimbMix {
            revision,
            max_records,
        } => {
            let config = nemotron_climbmix_config(revision, *max_records);
            Dataset::from_huggingface(
                HuggingFaceDataset::new(
                    &cfg.cache_dir,
                    training.block_size,
                    training.batch_size,
                    cfg.train_split_ratio,
                    &cfg.tokenizer,
                    &config,
                )
                .with_context(|| "failed to prepare Nemotron-ClimbMix dataset")?,
            )
        }
        DatasetSourceConfig::UniversalityManifest { manifest } => Dataset::from_universality(
            UniversalityDataset::new(
                manifest,
                training.block_size,
                training.batch_size,
                cfg.train_split_ratio,
                &cfg.tokenizer,
            )
            .with_context(|| {
                format!(
                    "failed to prepare universality manifest {}",
                    manifest.display()
                )
            })?,
        ),
        DatasetSourceConfig::UniversalityNca { config } => Dataset::from_universality(
            UniversalityDataset::new_on_the_fly(
                config,
                training.block_size,
                training.batch_size,
                training
                    .min_logical_block_size
                    .map(|value| value.max(training.block_size)),
                &cfg.tokenizer,
            )
            .with_context(|| {
                format!(
                    "failed to prepare on-the-fly universality NCA dataset {}",
                    config.display()
                )
            })?,
        ),
    };

    let description = match &dataset {
        Dataset::Shakespeare(ds) => format!(
            "Prepared Shakespeare dataset with batch_size={}, block_size={}, split_ratio={}",
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
        Dataset::LocalText(ds) => format!(
            "Prepared local text dataset {} with docs={}, batch_size={}, block_size={}, split_ratio={}",
            ds.source_path().display(),
            ds.document_count(),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
        Dataset::HuggingFace(ds) => format!(
            "Prepared Hugging Face dataset {} (rev: {}) with batch_size={}, block_size={}, split_ratio={}",
            ds.repo_id(),
            ds.revision().unwrap_or("main"),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio()
        ),
        Dataset::Universality(ds) => format!(
            "Prepared {} {} from {} with batch_size={}, block_size={}, split_ratio={}{}",
            ds.source_kind_label(),
            ds.dataset_name(),
            ds.source_path().display(),
            ds.batch_size(),
            ds.block_size(),
            ds.train_split_ratio(),
            ds.train_probe_summary().map(|summary| format!(
                ", train_docs={}, val_docs={}, doc_tokens={}, probe_mean_gzip={:.4}, probe_complexity={:.2}, runtime_doc_cache_limit={}",
                summary.sample_count,
                ds.validation_probe_summary()
                    .map(|probe| probe.sample_count)
                    .unwrap_or_default(),
                summary.document_token_count,
                summary.mean_gzip_complexity_ratio,
                summary.mean_complexity_score,
                ds.runtime_document_cache_limit().unwrap_or_default()
            )).unwrap_or_default()
        ),
    };

    Ok((dataset, description))
}

fn deepmath_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    let train_files = (0..10)
        .map(|idx| format!("data/train-{idx:05}-of-00010.parquet"))
        .collect();

    HuggingFaceDatasetConfig {
        repo_id: "zwhe99/DeepMath-103K".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files,
        auto_discover_train_files: false,
        validation_files: Vec::new(),
        text_fields: vec!["question".to_string(), "final_answer".to_string()],
        sequence_field: None,
        field_separator: "\n\n".to_string(),
        template: Some("Question:\n{question}\n\nAnswer:\n{final_answer}".to_string()),
        max_records,
    }
}

fn tinychat_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "starhopp3r/TinyChat".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Text,
        train_files: vec!["tinychat.txt".to_string()],
        auto_discover_train_files: false,
        validation_files: Vec::new(),
        text_fields: vec!["text".to_string()],
        sequence_field: None,
        field_separator: "\n\n".to_string(),
        template: None,
        max_records,
    }
}

fn webscale_rl_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    let mut train_files = Vec::with_capacity(12);
    for idx in 0..12 {
        train_files.push(format!("data/part-{idx}.parquet"));
    }

    HuggingFaceDatasetConfig {
        repo_id: "Salesforce/Webscale-RL".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files,
        auto_discover_train_files: false,
        validation_files: Vec::new(),
        text_fields: vec![
            "pretrain_text".to_string(),
            "question".to_string(),
            "answer".to_string(),
            "domain".to_string(),
            "persona".to_string(),
        ],
        sequence_field: None,
        field_separator: "\n\n".to_string(),
        template: Some(
            "Context:\n{pretrain_text}\n\nDomain: {domain}\nPersona: {persona}\n\nQuestion: \
             {question}\nAnswer: {answer}"
                .to_string(),
        ),
        max_records,
    }
}

fn poetry_foundation_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "suayptalha/Poetry-Foundation-Poems".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Csv,
        train_files: vec!["PoetryFoundationData.csv".to_string()],
        validation_files: Vec::new(),
        text_fields: vec!["Title".to_string(), "Poem".to_string()],
        sequence_field: None,
        field_separator: "\n\n\n".to_string(),
        template: Some("{Title}\n\n\n{Poem}\n\n\n\n\n\n".to_string()),
        max_records,
        auto_discover_train_files: false,
    }
}

fn openwebtext_gpt2_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "chanind/openwebtext-gpt2".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files: Vec::new(),
        auto_discover_train_files: true,
        validation_files: Vec::new(),
        text_fields: Vec::new(),
        sequence_field: Some("input_ids".to_string()),
        field_separator: "\n".to_string(),
        template: None,
        max_records,
    }
}

fn nemotron_climbmix_config(
    revision: &Option<String>,
    max_records: Option<usize>,
) -> HuggingFaceDatasetConfig {
    HuggingFaceDatasetConfig {
        repo_id: "nvidia/Nemotron-ClimbMix".to_string(),
        token: None,
        revision: revision.clone(),
        format: HuggingFaceRecordFormat::Parquet,
        train_files: Vec::new(),
        auto_discover_train_files: true,
        validation_files: Vec::new(),
        text_fields: Vec::new(),
        sequence_field: Some("tokens".to_string()),
        field_separator: "\n".to_string(),
        template: None,
        max_records,
    }
}
