use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::stats::{ComplexityHistogramBin, CorpusStats, SampleStats};

pub const UNIVERSALITY_MANIFEST_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CorpusKind {
    Nca,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SampleSplit {
    Train,
    Validation,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerFamily {
    Gpt2ByteCompatible,
    PatchTokenIds,
    RustBpe,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct UniversalityTokenizerManifest {
    pub family: TokenizerFamily,
    pub vocab_size: usize,
    #[serde(default)]
    pub bos_id: Option<u32>,
    #[serde(default)]
    pub eos_id: Option<u32>,
    #[serde(default)]
    pub frame_special_tokens: bool,
    #[serde(default)]
    pub pad_id: Option<u32>,
    #[serde(default)]
    pub unk_id: Option<u32>,
    #[serde(default)]
    pub tokenizer_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct UniversalityChunkManifest {
    pub file_name: String,
    pub split: SampleSplit,
    pub token_offset: usize,
    pub token_count: usize,
    pub sample_count: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct UniversalityCorpusManifest {
    pub version: u32,
    pub corpus_kind: CorpusKind,
    pub dataset_name: String,
    pub seed: u64,
    pub train_token_count: usize,
    pub val_token_count: usize,
    pub token_count: usize,
    pub chunk_token_capacity: usize,
    pub tokenizer: UniversalityTokenizerManifest,
    pub chunk_dir: PathBuf,
    pub preview_dir: PathBuf,
    pub sample_records_path: PathBuf,
    pub chunks: Vec<UniversalityChunkManifest>,
    pub stats: CorpusStats,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct UniversalitySampleRecord {
    pub sample_index: usize,
    pub split: SampleSplit,
    pub family: String,
    pub complexity_band: String,
    #[serde(default)]
    pub rule_seed: Option<u64>,
    #[serde(default)]
    pub complexity_filter_matched: bool,
    #[serde(default)]
    pub identity_bias: f32,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default)]
    pub step_stride: usize,
    #[serde(default)]
    pub start_step: usize,
    pub token_offset: usize,
    pub token_count: usize,
    pub preview_path: Option<PathBuf>,
    pub serialized_char_count: usize,
    pub stats: SampleStats,
}

pub fn load_manifest(path: &Path) -> Result<UniversalityCorpusManifest> {
    let bytes = fs::read(path).with_context(|| format!("failed to read {}", path.display()))?;
    let manifest: UniversalityCorpusManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("failed to parse {}", path.display()))?;
    Ok(manifest)
}

pub fn write_manifest(path: &Path, manifest: &UniversalityCorpusManifest) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let payload = serde_json::to_vec_pretty(manifest).context("serialize manifest")?;
    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn write_sample_records(path: &Path, samples: &[UniversalitySampleRecord]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut out = String::new();
    for sample in samples {
        out.push_str(&serde_json::to_string(sample).context("serialize sample record")?);
        out.push('\n');
    }
    fs::write(path, out).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn family_counts(samples: &[UniversalitySampleRecord]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for sample in samples {
        *counts.entry(sample.family.clone()).or_insert(0) += 1;
    }
    counts
}

pub fn complexity_histogram(samples: &[UniversalitySampleRecord]) -> Vec<ComplexityHistogramBin> {
    let scores = samples
        .iter()
        .map(|sample| sample.stats.complexity_score)
        .collect::<Vec<_>>();
    crate::stats::build_complexity_histogram(&scores)
}
