use anyhow::{Context, Result, anyhow};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::sync::Arc;

use crate::config::{NcaCorpusConfig, NcaFamilyConfig, NcaFamilyKind};
use crate::manifest::{SampleSplit, UniversalityTokenizerManifest};
use crate::nca::{compute_sample_stats, generate_sample, serialize_sample};
use crate::stats::{ComplexityHistogramBin, SampleStats, build_complexity_histogram};
use crate::tokenize::CorpusTokenizer;

const TRAIN_SPLIT_TAG: u64 = 0xA8A7_9B1C_D3E4_F501;
const VAL_SPLIT_TAG: u64 = 0x5EED_CAFE_1357_2468;
const DEFAULT_PROBE_SAMPLES: usize = 32;

#[derive(Debug, Clone)]
pub struct RuntimeCorpusSummary {
    pub sample_count: usize,
    pub token_count: usize,
    pub document_token_count: usize,
    pub mean_gzip_complexity_ratio: f32,
    pub min_gzip_complexity_ratio: f32,
    pub max_gzip_complexity_ratio: f32,
    pub mean_complexity_score: f32,
    pub complexity_histogram: Vec<ComplexityHistogramBin>,
}

#[derive(Debug, Clone)]
pub struct RuntimeSampleDocument {
    pub split: SampleSplit,
    pub sample_index: usize,
    pub family: String,
    pub complexity_band: String,
    pub rule_seed: Option<u64>,
    pub complexity_filter_matched: bool,
    pub token_count: usize,
    pub tokens: Vec<u32>,
    pub serialized_preview: String,
    pub stats: SampleStats,
}

#[derive(Clone)]
pub struct OnlineNcaCorpus {
    config: NcaCorpusConfig,
    tokenizer: Arc<CorpusTokenizer>,
    tokenizer_manifest: UniversalityTokenizerManifest,
    document_token_count: usize,
}

impl OnlineNcaCorpus {
    pub fn new(config: NcaCorpusConfig) -> Result<Self> {
        Self::new_with_min_logical_document_tokens(config, None)
    }

    pub fn new_with_min_logical_document_tokens(
        config: NcaCorpusConfig,
        min_logical_document_tokens: Option<usize>,
    ) -> Result<Self> {
        let adjusted_config =
            adapt_config_for_min_logical_document_tokens(config, min_logical_document_tokens)?;
        adjusted_config.validate()?;
        let tokenizer = Arc::new(CorpusTokenizer::from_config(&adjusted_config.tokenization)?);
        let tokenizer_manifest = tokenizer.manifest();
        let document_token_count = fixed_document_token_count(&adjusted_config)?;
        Ok(Self {
            config: adjusted_config,
            tokenizer,
            tokenizer_manifest,
            document_token_count,
        })
    }

    pub fn load(path: &std::path::Path) -> Result<Self> {
        let config = crate::load_nca_config(path)?;
        Self::new(config)
    }

    pub fn load_with_min_logical_document_tokens(
        path: &std::path::Path,
        min_logical_document_tokens: Option<usize>,
    ) -> Result<Self> {
        let config = crate::load_nca_config(path)?;
        Self::new_with_min_logical_document_tokens(config, min_logical_document_tokens)
    }

    pub fn config(&self) -> &NcaCorpusConfig {
        &self.config
    }

    pub fn dataset_name(&self) -> &str {
        &self.config.name
    }

    pub fn tokenizer_manifest(&self) -> &UniversalityTokenizerManifest {
        &self.tokenizer_manifest
    }

    pub fn train_samples(&self) -> usize {
        self.config.train_samples
    }

    pub fn validation_samples(&self) -> usize {
        self.config.validation_samples
    }

    pub fn sample_count(&self, split: SampleSplit) -> usize {
        match split {
            SampleSplit::Train => self.train_samples(),
            SampleSplit::Validation => self.validation_samples(),
        }
    }

    pub fn document_token_count(&self) -> usize {
        self.document_token_count
    }

    pub fn train_token_count(&self) -> usize {
        self.train_samples()
            .saturating_mul(self.document_token_count())
    }

    pub fn val_token_count(&self) -> usize {
        self.validation_samples()
            .saturating_mul(self.document_token_count())
    }

    pub fn total_token_count(&self) -> usize {
        self.train_token_count()
            .saturating_add(self.val_token_count())
    }

    pub fn generate_document(
        &self,
        split: SampleSplit,
        sample_index: usize,
    ) -> Result<RuntimeSampleDocument> {
        self.generate_document_for_epoch(split, 0, sample_index)
    }

    pub fn generate_document_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<RuntimeSampleDocument> {
        let sample = self.generate_raw_sample(split, epoch_index, sample_index)?;
        let tokens = self.encode_tokens_from_sample(&sample)?;
        if tokens.len() != self.document_token_count {
            return Err(anyhow!(
                "on-the-fly NCA document token length drifted (expected={} actual={})",
                self.document_token_count,
                tokens.len()
            ));
        }
        let stats = compute_sample_stats(&sample, &self.config.serialization);
        let serialized_preview = serialize_sample(&sample, &self.config.serialization);
        Ok(RuntimeSampleDocument {
            split,
            sample_index,
            family: family_kind_label(sample.family_kind).to_string(),
            complexity_band: format!("{:?}", sample.complexity_band).to_lowercase(),
            rule_seed: sample.rule_seed,
            complexity_filter_matched: sample.complexity_filter_matched,
            token_count: tokens.len(),
            tokens,
            serialized_preview,
            stats,
        })
    }

    pub fn generate_document_tokens(
        &self,
        split: SampleSplit,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        self.generate_document_tokens_for_epoch(split, 0, sample_index)
    }

    pub fn generate_document_tokens_for_epoch(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<Vec<u32>> {
        let sample = self.generate_raw_sample(split, epoch_index, sample_index)?;
        let tokens = self.encode_tokens_from_sample(&sample)?;
        if tokens.len() != self.document_token_count {
            return Err(anyhow!(
                "on-the-fly NCA document token length drifted (expected={} actual={})",
                self.document_token_count,
                tokens.len()
            ));
        }
        Ok(tokens)
    }

    fn generate_raw_sample(
        &self,
        split: SampleSplit,
        epoch_index: usize,
        sample_index: usize,
    ) -> Result<crate::nca::NcaSample> {
        let sample_count = self.sample_count(split);
        if sample_index >= sample_count {
            return Err(anyhow!(
                "sample_index {} out of range for {:?} split with {} samples",
                sample_index,
                split,
                sample_count
            ));
        }
        let effective_sample_index = match split {
            SampleSplit::Train => epoch_index
                .saturating_mul(sample_count)
                .saturating_add(sample_index),
            SampleSplit::Validation => sample_index,
        };
        let mut rng = StdRng::seed_from_u64(derive_sample_seed(
            self.config.seed,
            split,
            effective_sample_index,
        ));
        let family = choose_family(&self.config, &mut rng);
        Ok(generate_sample(
            family,
            &self.config.serialization,
            &mut rng,
        ))
    }

    fn encode_tokens_from_sample(&self, sample: &crate::nca::NcaSample) -> Result<Vec<u32>> {
        match self.tokenizer.as_ref() {
            CorpusTokenizer::PatchTokenIds { .. } => self
                .tokenizer
                .encode_patch_sample(sample, &self.config.serialization),
            _ => Err(anyhow!(
                "on-the-fly NCA training currently requires patch_token_ids tokenization"
            )),
        }
    }

    pub fn probe_summary(
        &self,
        split: SampleSplit,
        max_samples: usize,
    ) -> Result<RuntimeCorpusSummary> {
        let sample_count = self.sample_count(split);
        let probe_count = sample_count.min(max_samples.max(1));
        let mut gzip_ratios = Vec::with_capacity(probe_count);
        let mut complexity_scores = Vec::with_capacity(probe_count);
        for sample_index in 0..probe_count {
            let sample = self.generate_document(split, sample_index)?;
            gzip_ratios.push(sample.stats.gzip_complexity_ratio);
            complexity_scores.push(sample.stats.complexity_score);
        }
        let histogram = build_complexity_histogram(&complexity_scores);
        let mean_gzip_complexity_ratio = mean(gzip_ratios.iter().copied());
        let min_gzip_complexity_ratio = gzip_ratios
            .iter()
            .copied()
            .reduce(f32::min)
            .unwrap_or_default();
        let max_gzip_complexity_ratio = gzip_ratios
            .iter()
            .copied()
            .reduce(f32::max)
            .unwrap_or_default();
        let mean_complexity_score = mean(complexity_scores.iter().copied());
        Ok(RuntimeCorpusSummary {
            sample_count,
            token_count: sample_count.saturating_mul(self.document_token_count()),
            document_token_count: self.document_token_count(),
            mean_gzip_complexity_ratio,
            min_gzip_complexity_ratio,
            max_gzip_complexity_ratio,
            mean_complexity_score,
            complexity_histogram: histogram,
        })
    }

    pub fn default_probe_summary(&self, split: SampleSplit) -> Result<RuntimeCorpusSummary> {
        self.probe_summary(split, DEFAULT_PROBE_SAMPLES)
    }
}

pub fn fixed_document_token_count(config: &NcaCorpusConfig) -> Result<usize> {
    match &config.tokenization {
        crate::config::NcaTokenizationConfig::PatchTokenIds {
            eos_id,
            frame_special_tokens,
            ..
        } => {
            let mut expected: Option<usize> = None;
            for (index, family) in config.families.iter().enumerate() {
                let grid =
                    fixed_range_value(family.grid_size, &format!("families[{index}].grid_size"))?;
                let steps = fixed_range_value(family.steps, &format!("families[{index}].steps"))?;
                if grid % config.serialization.patch_size != 0 {
                    return Err(anyhow!(
                        "families[{index}].grid_size={} must be divisible by serialization.patch_size={}",
                        grid,
                        config.serialization.patch_size
                    ));
                }
                let patches_per_frame = (grid / config.serialization.patch_size)
                    * (grid / config.serialization.patch_size);
                let frame_token_count =
                    patches_per_frame.saturating_add(usize::from(*frame_special_tokens) * 2);
                let token_count = steps
                    .checked_mul(frame_token_count)
                    .and_then(|value| value.checked_add(usize::from(eos_id.is_some())))
                    .ok_or_else(|| anyhow!("on-the-fly NCA token length overflow"))?;
                if let Some(previous) = expected {
                    if previous != token_count {
                        return Err(anyhow!(
                            "on-the-fly NCA training currently requires fixed document length across families (got {} and {})",
                            previous,
                            token_count
                        ));
                    }
                } else {
                    expected = Some(token_count);
                }
            }
            expected.context("on-the-fly NCA training requires at least one family")
        }
        _ => Err(anyhow!(
            "on-the-fly NCA training currently requires tokenization.type = `patch_token_ids`"
        )),
    }
}

fn adapt_config_for_min_logical_document_tokens(
    mut config: NcaCorpusConfig,
    min_logical_document_tokens: Option<usize>,
) -> Result<NcaCorpusConfig> {
    let Some(min_logical_document_tokens) = min_logical_document_tokens.filter(|value| *value > 0)
    else {
        return Ok(config);
    };

    let base_document_token_count = fixed_document_token_count(&config)?;
    let eos_tokens = match &config.tokenization {
        crate::config::NcaTokenizationConfig::PatchTokenIds { eos_id, .. } => {
            usize::from(eos_id.is_some())
        }
        _ => {
            return Ok(config);
        }
    };
    let base_logical_document_tokens = base_document_token_count.saturating_sub(1);
    if base_logical_document_tokens >= min_logical_document_tokens {
        return Ok(config);
    }

    let desired_document_token_count = min_logical_document_tokens
        .checked_add(1)
        .ok_or_else(|| anyhow!("requested on-the-fly NCA logical document length overflow"))?;
    let desired_payload_tokens = desired_document_token_count
        .checked_sub(eos_tokens)
        .ok_or_else(|| anyhow!("invalid on-the-fly NCA tokenization layout"))?;

    let mut patches_per_frame = Vec::with_capacity(config.families.len());
    let mut payload_alignment = 1usize;
    for (index, family) in config.families.iter().enumerate() {
        let grid = fixed_range_value(family.grid_size, &format!("families[{index}].grid_size"))?;
        if grid % config.serialization.patch_size != 0 {
            return Err(anyhow!(
                "families[{index}].grid_size={} must be divisible by serialization.patch_size={}",
                grid,
                config.serialization.patch_size
            ));
        }
        let frame_tokens = (grid / config.serialization.patch_size)
            * (grid / config.serialization.patch_size)
            + match &config.tokenization {
                crate::config::NcaTokenizationConfig::PatchTokenIds {
                    frame_special_tokens,
                    ..
                } => usize::from(*frame_special_tokens) * 2,
                _ => 0,
            };
        payload_alignment = lcm_usize(payload_alignment, frame_tokens)
            .ok_or_else(|| anyhow!("on-the-fly NCA payload alignment overflow"))?;
        patches_per_frame.push(frame_tokens);
    }

    let target_payload_tokens = desired_payload_tokens
        .div_ceil(payload_alignment)
        .checked_mul(payload_alignment)
        .ok_or_else(|| anyhow!("on-the-fly NCA target payload length overflow"))?;

    for (family, patches) in config
        .families
        .iter_mut()
        .zip(patches_per_frame.into_iter())
    {
        let steps = target_payload_tokens
            .checked_div(patches)
            .ok_or_else(|| anyhow!("invalid per-family on-the-fly NCA step layout"))?;
        family.steps = Some(crate::config::UsizeRangeConfig {
            min: steps,
            max: steps,
        });
    }

    Ok(config)
}

fn fixed_range_value(range: Option<crate::config::UsizeRangeConfig>, label: &str) -> Result<usize> {
    let Some(range) = range else {
        return Err(anyhow!(
            "{label} must be specified explicitly for on-the-fly NCA training"
        ));
    };
    if range.min != range.max {
        return Err(anyhow!(
            "{label} must have min == max for on-the-fly NCA training"
        ));
    }
    Ok(range.min)
}

fn gcd_usize(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn lcm_usize(lhs: usize, rhs: usize) -> Option<usize> {
    if lhs == 0 || rhs == 0 {
        return Some(0);
    }
    let gcd = gcd_usize(lhs, rhs);
    lhs.checked_div(gcd)?.checked_mul(rhs)
}

fn choose_family<'a>(config: &'a NcaCorpusConfig, rng: &mut StdRng) -> &'a NcaFamilyConfig {
    let total_weight = config
        .families
        .iter()
        .map(|family| family.weight)
        .sum::<usize>()
        .max(1);
    let mut cursor = rng.gen_range(0..total_weight);
    for family in &config.families {
        if cursor < family.weight {
            return family;
        }
        cursor -= family.weight;
    }
    &config.families[0]
}

fn derive_sample_seed(base_seed: u64, split: SampleSplit, sample_index: usize) -> u64 {
    let split_tag = match split {
        SampleSplit::Train => TRAIN_SPLIT_TAG,
        SampleSplit::Validation => VAL_SPLIT_TAG,
    };
    let mixed = base_seed ^ split_tag ^ (sample_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    splitmix64(mixed)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn family_kind_label(kind: NcaFamilyKind) -> &'static str {
    match kind {
        NcaFamilyKind::NeuralStochastic => "neural_stochastic",
        NcaFamilyKind::LifeLikeBinary => "life_like_binary",
        NcaFamilyKind::Cyclic => "cyclic",
        NcaFamilyKind::NeuralTotalistic => "neural_totalistic",
    }
}

fn mean(values: impl IntoIterator<Item = f32>) -> f32 {
    let mut total = 0.0f32;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{
        NcaCorpusConfig, NcaSerializationConfig, NcaTokenizationConfig, default_families,
    };

    fn fixed_patch_config() -> NcaCorpusConfig {
        let mut config = NcaCorpusConfig {
            output_dir: "ignored".into(),
            seed: 1337,
            name: "runtime".to_string(),
            train_samples: 8,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: default_families(),
        };
        for family in &mut config.families {
            family.grid_size = Some(crate::config::UsizeRangeConfig { min: 12, max: 12 });
            family.steps = Some(crate::config::UsizeRangeConfig { min: 10, max: 10 });
            family.state_count = Some(crate::config::UsizeRangeConfig { min: 10, max: 10 });
            family.step_stride = Some(crate::config::UsizeRangeConfig { min: 2, max: 2 });
            family.start_step = Some(crate::config::UsizeRangeConfig { min: 0, max: 0 });
            family.identity_bias = Some(crate::config::FloatRangeConfig { min: 0.0, max: 0.0 });
            family.temperature = Some(crate::config::FloatRangeConfig { min: 0.0, max: 0.0 });
        }
        config
    }

    #[test]
    fn online_corpus_reports_fixed_document_token_count() {
        let config = fixed_patch_config();
        let corpus = OnlineNcaCorpus::new(config).expect("runtime corpus");
        assert_eq!(corpus.document_token_count(), 381);
        assert_eq!(corpus.train_token_count(), 8 * 381);
        assert_eq!(corpus.val_token_count(), 4 * 381);
    }

    #[test]
    fn online_corpus_is_deterministic_per_split_and_index() {
        let config = fixed_patch_config();
        let corpus = OnlineNcaCorpus::new(config).expect("runtime corpus");
        let first = corpus
            .generate_document(SampleSplit::Train, 2)
            .expect("first sample");
        let second = corpus
            .generate_document(SampleSplit::Train, 2)
            .expect("second sample");
        let val = corpus
            .generate_document(SampleSplit::Validation, 2)
            .expect("val sample");
        assert_eq!(first.tokens, second.tokens);
        assert_ne!(first.tokens, val.tokens);
    }

    #[test]
    fn online_corpus_train_split_changes_across_epochs() {
        let config = fixed_patch_config();
        let corpus = OnlineNcaCorpus::new(config).expect("runtime corpus");
        let epoch0 = corpus
            .generate_document_for_epoch(SampleSplit::Train, 0, 2)
            .expect("epoch0 sample");
        let epoch1 = corpus
            .generate_document_for_epoch(SampleSplit::Train, 1, 2)
            .expect("epoch1 sample");
        assert_ne!(epoch0.tokens, epoch1.tokens);
    }

    #[test]
    fn online_corpus_validation_split_stays_fixed_across_epochs() {
        let config = fixed_patch_config();
        let corpus = OnlineNcaCorpus::new(config).expect("runtime corpus");
        let epoch0 = corpus
            .generate_document_for_epoch(SampleSplit::Validation, 0, 2)
            .expect("epoch0 validation sample");
        let epoch5 = corpus
            .generate_document_for_epoch(SampleSplit::Validation, 5, 2)
            .expect("epoch5 validation sample");
        assert_eq!(epoch0.tokens, epoch5.tokens);
    }

    #[test]
    fn online_corpus_epoch_stream_is_deterministic_across_instances() {
        let config = fixed_patch_config();
        let corpus_a = OnlineNcaCorpus::new(config.clone()).expect("runtime corpus a");
        let corpus_b = OnlineNcaCorpus::new(config).expect("runtime corpus b");

        let train_a = corpus_a
            .generate_document_for_epoch(SampleSplit::Train, 7, 3)
            .expect("train sample a");
        let train_b = corpus_b
            .generate_document_for_epoch(SampleSplit::Train, 7, 3)
            .expect("train sample b");
        let val_a = corpus_a
            .generate_document_for_epoch(SampleSplit::Validation, 7, 1)
            .expect("val sample a");
        let val_b = corpus_b
            .generate_document_for_epoch(SampleSplit::Validation, 7, 1)
            .expect("val sample b");

        assert_eq!(train_a.tokens, train_b.tokens);
        assert_eq!(val_a.tokens, val_b.tokens);
        assert_ne!(train_a.tokens, val_a.tokens);
    }

    #[test]
    fn online_corpus_can_adapt_document_length_for_large_logical_blocks() {
        let config = fixed_patch_config();
        let corpus = OnlineNcaCorpus::new_with_min_logical_document_tokens(config, Some(4096))
            .expect("runtime corpus");
        assert_eq!(corpus.document_token_count(), 4105);

        let doc = corpus
            .generate_document(SampleSplit::Train, 0)
            .expect("train sample");
        assert_eq!(doc.token_count, 4105);
        assert_eq!(doc.stats.steps, 108);
        assert!(doc.stats.mean_transition_rate.is_finite());
        assert!(doc.stats.mean_transition_rate > 0.0);
        assert!(doc.stats.unique_frames > 1);
    }
}
