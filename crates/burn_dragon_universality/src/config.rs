use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NcaComplexityBand {
    Simple,
    Medium,
    Complex,
}

impl Default for NcaComplexityBand {
    fn default() -> Self {
        Self::Medium
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NcaFamilyKind {
    NeuralStochastic,
    LifeLikeBinary,
    Cyclic,
    NeuralTotalistic,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct UsizeRangeConfig {
    pub min: usize,
    pub max: usize,
}

impl UsizeRangeConfig {
    pub fn validate(&self, label: &str) -> Result<()> {
        if self.min > self.max {
            return Err(anyhow!("{label}.min must be <= {label}.max"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct FloatRangeConfig {
    pub min: f32,
    pub max: f32,
}

impl FloatRangeConfig {
    pub fn validate(&self, label: &str) -> Result<()> {
        if !self.min.is_finite() || !self.max.is_finite() {
            return Err(anyhow!("{label} bounds must be finite"));
        }
        if self.min > self.max {
            return Err(anyhow!("{label}.min must be <= {label}.max"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum NcaComplexityMetric {
    #[default]
    Gzip,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NcaRuleFilterConfig {
    #[serde(default)]
    pub metric: NcaComplexityMetric,
    #[serde(default)]
    pub threshold: Option<f32>,
    #[serde(default)]
    pub upper_bound: Option<f32>,
    #[serde(default = "default_rule_filter_max_attempts")]
    pub max_attempts: usize,
    #[serde(default)]
    pub scoring_examples: Option<usize>,
}

impl Default for NcaRuleFilterConfig {
    fn default() -> Self {
        Self {
            metric: NcaComplexityMetric::Gzip,
            threshold: None,
            upper_bound: None,
            max_attempts: default_rule_filter_max_attempts(),
            scoring_examples: None,
        }
    }
}

impl NcaRuleFilterConfig {
    pub fn validate(&self, label: &str) -> Result<()> {
        if let Some(threshold) = self.threshold {
            if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
                return Err(anyhow!("{label}.threshold must be within [0, 1]"));
            }
        }
        if let Some(upper_bound) = self.upper_bound {
            if !upper_bound.is_finite() || !(0.0..=1.0).contains(&upper_bound) {
                return Err(anyhow!("{label}.upper_bound must be within [0, 1]"));
            }
        }
        if matches!(
            (self.threshold, self.upper_bound),
            (Some(lower), Some(upper)) if lower > upper
        ) {
            return Err(anyhow!("{label}.threshold must be <= {label}.upper_bound"));
        }
        if self.max_attempts == 0 {
            return Err(anyhow!("{label}.max_attempts must be > 0"));
        }
        if matches!(self.scoring_examples, Some(0)) {
            return Err(anyhow!("{label}.scoring_examples must be > 0"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NcaFamilyConfig {
    pub kind: NcaFamilyKind,
    #[serde(default = "default_weight")]
    pub weight: usize,
    #[serde(default)]
    pub complexity: NcaComplexityBand,
    #[serde(default)]
    pub grid_size: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub steps: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub state_count: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub step_stride: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub start_step: Option<UsizeRangeConfig>,
    #[serde(default)]
    pub identity_bias: Option<FloatRangeConfig>,
    #[serde(default)]
    pub temperature: Option<FloatRangeConfig>,
    #[serde(default)]
    pub rule_filter: Option<NcaRuleFilterConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NcaSerializationConfig {
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,
    #[serde(default = "default_true")]
    pub include_observable_header: bool,
    #[serde(default = "default_preview_samples")]
    pub preview_samples: usize,
}

impl Default for NcaSerializationConfig {
    fn default() -> Self {
        Self {
            patch_size: default_patch_size(),
            include_observable_header: true,
            preview_samples: default_preview_samples(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum NcaTokenizationConfig {
    Gpt2ByteCompatible {
        #[serde(default = "default_gpt2_vocab_size")]
        vocab_size: usize,
        #[serde(default = "default_gpt2_eos_id")]
        eos_id: Option<u32>,
    },
    PatchTokenIds {
        #[serde(default = "default_gpt2_vocab_size")]
        vocab_size: usize,
        #[serde(default = "default_gpt2_eos_id")]
        eos_id: Option<u32>,
        #[serde(default = "default_true")]
        frame_special_tokens: bool,
    },
    RustBpe {
        vocab_path: PathBuf,
        mergeable_vocab_size: usize,
        #[serde(default)]
        bos_id: Option<u32>,
        #[serde(default)]
        eos_id: Option<u32>,
        #[serde(default)]
        pad_id: Option<u32>,
        #[serde(default)]
        unk_id: Option<u32>,
    },
}

impl Default for NcaTokenizationConfig {
    fn default() -> Self {
        Self::PatchTokenIds {
            vocab_size: default_gpt2_vocab_size(),
            eos_id: default_gpt2_eos_id(),
            frame_special_tokens: default_true(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct NcaCorpusConfig {
    pub output_dir: PathBuf,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_name")]
    pub name: String,
    pub train_samples: usize,
    pub validation_samples: usize,
    #[serde(default = "default_chunk_token_capacity")]
    pub chunk_token_capacity: usize,
    #[serde(default)]
    pub serialization: NcaSerializationConfig,
    #[serde(default)]
    pub tokenization: NcaTokenizationConfig,
    #[serde(default = "default_families")]
    pub families: Vec<NcaFamilyConfig>,
}

impl NcaCorpusConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(anyhow!("name must not be empty"));
        }
        if self.train_samples == 0 {
            return Err(anyhow!("train_samples must be > 0"));
        }
        if self.chunk_token_capacity == 0 {
            return Err(anyhow!("chunk_token_capacity must be > 0"));
        }
        if self.serialization.patch_size == 0 {
            return Err(anyhow!("serialization.patch_size must be > 0"));
        }
        if self.serialization.preview_samples == 0 {
            return Err(anyhow!("serialization.preview_samples must be > 0"));
        }
        if self.families.is_empty() {
            return Err(anyhow!("families must not be empty"));
        }
        for (index, family) in self.families.iter().enumerate() {
            if family.weight == 0 {
                return Err(anyhow!("families[{index}].weight must be > 0"));
            }
            if let Some(range) = &family.grid_size {
                range.validate(&format!("families[{index}].grid_size"))?;
                if range.min == 0 || range.max == 0 {
                    return Err(anyhow!("families[{index}].grid_size bounds must be > 0"));
                }
            }
            if let Some(range) = &family.steps {
                range.validate(&format!("families[{index}].steps"))?;
                if range.min == 0 || range.max == 0 {
                    return Err(anyhow!("families[{index}].steps bounds must be > 0"));
                }
            }
            if let Some(range) = &family.state_count {
                range.validate(&format!("families[{index}].state_count"))?;
                if range.min == 0 || range.max == 0 {
                    return Err(anyhow!("families[{index}].state_count bounds must be > 0"));
                }
            }
            if let Some(range) = &family.step_stride {
                range.validate(&format!("families[{index}].step_stride"))?;
                if range.min == 0 || range.max == 0 {
                    return Err(anyhow!("families[{index}].step_stride bounds must be > 0"));
                }
            }
            if let Some(range) = &family.start_step {
                range.validate(&format!("families[{index}].start_step"))?;
            }
            if let Some(range) = &family.identity_bias {
                range.validate(&format!("families[{index}].identity_bias"))?;
            }
            if let Some(range) = &family.temperature {
                range.validate(&format!("families[{index}].temperature"))?;
            }
            if let Some(filter) = &family.rule_filter {
                filter.validate(&format!("families[{index}].rule_filter"))?;
            }
        }
        match &self.tokenization {
            NcaTokenizationConfig::Gpt2ByteCompatible { vocab_size, eos_id } => {
                if *vocab_size < 257 {
                    return Err(anyhow!(
                        "tokenization.vocab_size must be >= 257 for gpt2_byte_compatible"
                    ));
                }
                if matches!(eos_id, Some(id) if *id as usize >= *vocab_size) {
                    return Err(anyhow!(
                        "tokenization.eos_id must be < tokenization.vocab_size"
                    ));
                }
            }
            NcaTokenizationConfig::PatchTokenIds {
                vocab_size,
                eos_id,
                frame_special_tokens,
            } => {
                if *vocab_size < 2 {
                    return Err(anyhow!(
                        "tokenization.vocab_size must be >= 2 for patch_token_ids"
                    ));
                }
                if matches!(eos_id, Some(id) if *id as usize >= *vocab_size) {
                    return Err(anyhow!(
                        "tokenization.eos_id must be < tokenization.vocab_size"
                    ));
                }
                let patch_states = max_patch_state_count(self.families.as_slice());
                let patch_vocab_size = patch_states
                    .checked_pow(
                        (self.serialization.patch_size * self.serialization.patch_size) as u32,
                    )
                    .ok_or_else(|| anyhow!("patch token vocabulary overflow"))?;
                let frame_special_budget = usize::from(*frame_special_tokens) * 2;
                let special_budget = usize::from(eos_id.is_some()) + frame_special_budget;
                if patch_vocab_size.saturating_add(special_budget) > *vocab_size {
                    return Err(anyhow!(
                        "tokenization.vocab_size={} is too small for patch_token_ids (need at least {} states^patch_cells + specials = {})",
                        vocab_size,
                        patch_states,
                        patch_vocab_size + special_budget
                    ));
                }
                if *frame_special_tokens {
                    let frame_start_id = patch_vocab_size as u32;
                    let frame_end_id = frame_start_id
                        .checked_add(1)
                        .ok_or_else(|| anyhow!("patch frame special token overflow"))?;
                    if matches!(eos_id, Some(id) if *id == frame_start_id || *id == frame_end_id) {
                        return Err(anyhow!(
                            "tokenization.eos_id collides with patch frame special token ids"
                        ));
                    }
                }
            }
            NcaTokenizationConfig::RustBpe {
                vocab_path,
                mergeable_vocab_size,
                bos_id,
                eos_id,
                pad_id,
                unk_id,
            } => {
                if *mergeable_vocab_size < 256 {
                    return Err(anyhow!(
                        "tokenization.mergeable_vocab_size must be >= 256 for rust_bpe"
                    ));
                }
                if vocab_path.as_os_str().is_empty() {
                    return Err(anyhow!("tokenization.vocab_path must not be empty"));
                }
                let special_max = [*bos_id, *eos_id, *pad_id, *unk_id]
                    .into_iter()
                    .flatten()
                    .max()
                    .unwrap_or_default() as usize;
                if special_max >= *mergeable_vocab_size && special_max >= usize::MAX - 1 {
                    return Err(anyhow!("tokenization special token ids overflow"));
                }
            }
        }
        Ok(())
    }
}

pub fn load_nca_config(path: &Path) -> Result<NcaCorpusConfig> {
    let contents = fs::read_to_string(path)
        .with_context(|| format!("failed to read NCA config {}", path.display()))?;
    let config: NcaCorpusConfig =
        toml::from_str(&contents).with_context(|| format!("failed to parse {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

fn default_seed() -> u64 {
    1337
}

fn default_name() -> String {
    "nca_universality".to_string()
}

fn default_weight() -> usize {
    1
}

fn default_patch_size() -> usize {
    2
}

fn default_preview_samples() -> usize {
    4
}

fn default_chunk_token_capacity() -> usize {
    1_048_576
}

fn default_gpt2_vocab_size() -> usize {
    50_257
}

fn default_gpt2_eos_id() -> Option<u32> {
    Some(50_256)
}

fn default_true() -> bool {
    true
}

pub fn default_families() -> Vec<NcaFamilyConfig> {
    vec![
        NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Simple,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Simple)),
        },
        NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Medium,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Medium)),
        },
        NcaFamilyConfig {
            kind: NcaFamilyKind::NeuralStochastic,
            weight: 1,
            complexity: NcaComplexityBand::Complex,
            grid_size: Some(UsizeRangeConfig { min: 12, max: 12 }),
            steps: Some(UsizeRangeConfig { min: 10, max: 10 }),
            state_count: Some(UsizeRangeConfig { min: 10, max: 10 }),
            step_stride: Some(UsizeRangeConfig { min: 2, max: 2 }),
            start_step: Some(UsizeRangeConfig { min: 0, max: 0 }),
            identity_bias: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            temperature: Some(FloatRangeConfig { min: 0.0, max: 0.0 }),
            rule_filter: Some(default_rule_filter_for_band(NcaComplexityBand::Complex)),
        },
    ]
}

pub fn default_rule_filter_for_band(band: NcaComplexityBand) -> NcaRuleFilterConfig {
    let (threshold, upper_bound) = match band {
        NcaComplexityBand::Simple => (Some(0.0), Some(0.30)),
        NcaComplexityBand::Medium => (Some(0.30), Some(0.50)),
        NcaComplexityBand::Complex => (Some(0.50), Some(1.0)),
    };
    NcaRuleFilterConfig {
        metric: NcaComplexityMetric::Gzip,
        threshold,
        upper_bound,
        max_attempts: default_rule_filter_max_attempts(),
        scoring_examples: None,
    }
}

fn default_rule_filter_max_attempts() -> usize {
    256
}

fn max_patch_state_count(families: &[NcaFamilyConfig]) -> usize {
    families
        .iter()
        .map(max_state_count_for_family)
        .max()
        .unwrap_or(1)
}

fn max_state_count_for_family(family: &NcaFamilyConfig) -> usize {
    if let Some(range) = family.state_count {
        return range.max.max(1);
    }
    match family.kind {
        NcaFamilyKind::NeuralStochastic => 10,
        NcaFamilyKind::LifeLikeBinary => 2,
        NcaFamilyKind::Cyclic => match family.complexity {
            NcaComplexityBand::Simple => 4,
            NcaComplexityBand::Medium => 6,
            NcaComplexityBand::Complex => 8,
        },
        NcaFamilyKind::NeuralTotalistic => match family.complexity {
            NcaComplexityBand::Simple => 6,
            NcaComplexityBand::Medium => 10,
            NcaComplexityBand::Complex => 16,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn default_config_validates() {
        let dir = tempdir().expect("tempdir");
        let config = NcaCorpusConfig {
            output_dir: dir.path().join("out"),
            seed: 1337,
            name: "demo".to_string(),
            train_samples: 8,
            validation_samples: 2,
            chunk_token_capacity: 1024,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: default_families(),
        };

        config.validate().expect("valid config");
    }
}
