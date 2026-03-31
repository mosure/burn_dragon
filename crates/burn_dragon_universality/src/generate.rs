use std::fs;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::config::NcaCorpusConfig;
use crate::manifest::{
    CorpusKind, SampleSplit, UniversalityChunkManifest, UniversalityCorpusManifest,
    UniversalitySampleRecord, complexity_histogram, family_counts, write_manifest,
    write_sample_records,
};
use crate::nca::{compute_sample_stats, generate_sample, serialize_sample};
use crate::stats::CorpusStats;
use crate::tokenize::CorpusTokenizer;

#[derive(Debug, Clone)]
pub struct GeneratedCorpusReport {
    pub manifest_path: PathBuf,
    pub sample_records_path: PathBuf,
    pub preview_dir: PathBuf,
    pub train_samples: usize,
    pub validation_samples: usize,
    pub train_token_count: usize,
    pub val_token_count: usize,
}

pub fn generate_nca_corpus(config: &NcaCorpusConfig) -> Result<GeneratedCorpusReport> {
    config.validate()?;
    fs::create_dir_all(&config.output_dir)
        .with_context(|| format!("failed to create {}", config.output_dir.display()))?;

    let chunk_dir = config.output_dir.join("chunks");
    let preview_dir = config.output_dir.join("preview");
    if chunk_dir.exists() {
        fs::remove_dir_all(&chunk_dir)
            .with_context(|| format!("failed to reset {}", chunk_dir.display()))?;
    }
    if preview_dir.exists() {
        fs::remove_dir_all(&preview_dir)
            .with_context(|| format!("failed to reset {}", preview_dir.display()))?;
    }
    fs::create_dir_all(&chunk_dir)?;
    fs::create_dir_all(&preview_dir)?;

    let tokenizer = CorpusTokenizer::from_config(&config.tokenization)?;
    let mut rng = StdRng::seed_from_u64(config.seed);
    let mut preview_budget = config.serialization.preview_samples;
    let mut sample_records = Vec::with_capacity(config.train_samples + config.validation_samples);

    let mut train_writer =
        ChunkWriter::new(&chunk_dir, SampleSplit::Train, config.chunk_token_capacity);
    for sample_index in 0..config.train_samples {
        let record = generate_sample_record(
            config,
            &tokenizer,
            SampleSplit::Train,
            sample_index,
            &mut rng,
            &preview_dir,
            &mut preview_budget,
            &mut train_writer,
        )?;
        sample_records.push(record);
    }
    let train_token_count = train_writer.total_token_count();
    let train_chunks = train_writer.finish()?;

    let mut val_writer = ChunkWriter::new(
        &chunk_dir,
        SampleSplit::Validation,
        config.chunk_token_capacity,
    );
    for sample_index in 0..config.validation_samples {
        let record = generate_sample_record(
            config,
            &tokenizer,
            SampleSplit::Validation,
            sample_index,
            &mut rng,
            &preview_dir,
            &mut preview_budget,
            &mut val_writer,
        )?;
        sample_records.push(record);
    }
    let val_token_count = val_writer.total_token_count();
    let val_chunks = val_writer.finish()?;

    for record in &mut sample_records {
        if record.split == SampleSplit::Validation {
            record.token_offset += train_token_count;
        }
    }

    let mut chunks = Vec::with_capacity(train_chunks.len() + val_chunks.len());
    let mut running_offset = 0usize;
    for chunk in train_chunks.into_iter().chain(val_chunks.into_iter()) {
        chunks.push(UniversalityChunkManifest {
            file_name: chunk.file_name,
            split: chunk.split,
            token_offset: running_offset,
            token_count: chunk.token_count,
            sample_count: chunk.sample_count,
        });
        running_offset += chunk.token_count;
    }

    let total_token_count = train_token_count + val_token_count;
    let total_samples = sample_records.len();
    let mean_token_count = if total_samples == 0 {
        0.0
    } else {
        sample_records
            .iter()
            .map(|record| record.token_count as f32)
            .sum::<f32>()
            / total_samples as f32
    };
    let mean_entropy_bits = mean(
        sample_records
            .iter()
            .map(|record| record.stats.mean_entropy_bits),
    );
    let mean_transition_rate = mean(
        sample_records
            .iter()
            .map(|record| record.stats.mean_transition_rate),
    );
    let mean_active_ratio = mean(
        sample_records
            .iter()
            .map(|record| record.stats.active_ratio_mean),
    );
    let gzip_ratios = sample_records
        .iter()
        .map(|record| record.stats.gzip_complexity_ratio)
        .collect::<Vec<_>>();
    let complexity_scores = sample_records
        .iter()
        .map(|record| record.stats.complexity_score)
        .collect::<Vec<_>>();
    let family_counts = family_counts(&sample_records);
    let complexity_histogram = complexity_histogram(&sample_records);
    let min_complexity_score = complexity_scores
        .iter()
        .copied()
        .reduce(f32::min)
        .unwrap_or_default();
    let max_complexity_score = complexity_scores
        .iter()
        .copied()
        .reduce(f32::max)
        .unwrap_or_default();
    let mean_complexity_score = mean(complexity_scores.into_iter());
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
    let mean_gzip_complexity_ratio = mean(gzip_ratios.into_iter());

    let stats = CorpusStats {
        total_samples,
        train_samples: config.train_samples,
        validation_samples: config.validation_samples,
        total_token_count,
        mean_token_count,
        mean_entropy_bits,
        mean_transition_rate,
        mean_active_ratio,
        mean_gzip_complexity_ratio,
        min_gzip_complexity_ratio,
        max_gzip_complexity_ratio,
        mean_complexity_score,
        min_complexity_score,
        max_complexity_score,
        family_counts,
        complexity_histogram,
    };

    let sample_records_path = config.output_dir.join("sample_records.jsonl");
    write_sample_records(&sample_records_path, &sample_records)?;
    let manifest_path = config.output_dir.join("manifest.json");
    let manifest = UniversalityCorpusManifest {
        version: crate::manifest::UNIVERSALITY_MANIFEST_VERSION,
        corpus_kind: CorpusKind::Nca,
        dataset_name: config.name.clone(),
        seed: config.seed,
        train_token_count,
        val_token_count,
        token_count: total_token_count,
        chunk_token_capacity: config.chunk_token_capacity,
        tokenizer: tokenizer.manifest(),
        chunk_dir: PathBuf::from("chunks"),
        preview_dir: PathBuf::from("preview"),
        sample_records_path: PathBuf::from("sample_records.jsonl"),
        chunks,
        stats,
    };
    write_manifest(&manifest_path, &manifest)?;

    Ok(GeneratedCorpusReport {
        manifest_path,
        sample_records_path,
        preview_dir,
        train_samples: config.train_samples,
        validation_samples: config.validation_samples,
        train_token_count,
        val_token_count,
    })
}

fn generate_sample_record(
    config: &NcaCorpusConfig,
    tokenizer: &CorpusTokenizer,
    split: SampleSplit,
    sample_index: usize,
    rng: &mut StdRng,
    preview_dir: &Path,
    preview_budget: &mut usize,
    writer: &mut ChunkWriter,
) -> Result<UniversalitySampleRecord> {
    let family = choose_family(config, rng);
    let sample = generate_sample(family, &config.serialization, rng);
    let serialized = serialize_sample(&sample, &config.serialization);
    let token_offset = writer.total_token_count();
    let tokens = match tokenizer {
        CorpusTokenizer::PatchTokenIds { .. } => {
            tokenizer.encode_patch_sample(&sample, &config.serialization)?
        }
        CorpusTokenizer::Gpt2ByteCompatible { .. } | CorpusTokenizer::RustBpe { .. } => {
            tokenizer.encode(&serialized)
        }
    };
    writer.push_document(&tokens)?;
    let stats = compute_sample_stats(&sample, &config.serialization);
    let preview_path = if *preview_budget > 0 {
        *preview_budget -= 1;
        let preview_name = format!(
            "{}_{sample_index:05}.txt",
            match split {
                SampleSplit::Train => "train",
                SampleSplit::Validation => "validation",
            }
        );
        let path = preview_dir.join(&preview_name);
        fs::write(&path, &serialized)
            .with_context(|| format!("failed to write {}", path.display()))?;
        Some(PathBuf::from(preview_name))
    } else {
        None
    };

    Ok(UniversalitySampleRecord {
        sample_index,
        split,
        family: family_kind_label(sample.family_kind).to_string(),
        complexity_band: format!("{:?}", sample.complexity_band).to_lowercase(),
        rule_seed: sample.rule_seed,
        complexity_filter_matched: sample.complexity_filter_matched,
        identity_bias: sample.identity_bias,
        temperature: sample.temperature,
        step_stride: sample.step_stride,
        start_step: sample.start_step,
        token_offset,
        token_count: tokens.len(),
        preview_path,
        serialized_char_count: serialized.len(),
        stats,
    })
}

fn choose_family<'a>(
    config: &'a NcaCorpusConfig,
    rng: &mut StdRng,
) -> &'a crate::config::NcaFamilyConfig {
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

fn family_kind_label(kind: crate::config::NcaFamilyKind) -> &'static str {
    match kind {
        crate::config::NcaFamilyKind::NeuralStochastic => "neural_stochastic",
        crate::config::NcaFamilyKind::LifeLikeBinary => "life_like_binary",
        crate::config::NcaFamilyKind::Cyclic => "cyclic",
        crate::config::NcaFamilyKind::NeuralTotalistic => "neural_totalistic",
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

struct ChunkWriter {
    chunk_dir: PathBuf,
    split: SampleSplit,
    chunk_token_capacity: usize,
    current_writer: Option<BufWriter<fs::File>>,
    current_file_name: Option<String>,
    current_token_count: usize,
    current_sample_count: usize,
    part_index: usize,
    total_token_count: usize,
    chunks: Vec<ChunkPart>,
}

#[derive(Debug)]
struct ChunkPart {
    file_name: String,
    split: SampleSplit,
    token_count: usize,
    sample_count: usize,
}

impl ChunkWriter {
    fn new(chunk_dir: &Path, split: SampleSplit, chunk_token_capacity: usize) -> Self {
        Self {
            chunk_dir: chunk_dir.to_path_buf(),
            split,
            chunk_token_capacity: chunk_token_capacity.max(1),
            current_writer: None,
            current_file_name: None,
            current_token_count: 0,
            current_sample_count: 0,
            part_index: 0,
            total_token_count: 0,
            chunks: Vec::new(),
        }
    }

    fn total_token_count(&self) -> usize {
        self.total_token_count
    }

    fn push_document(&mut self, tokens: &[u32]) -> Result<()> {
        if tokens.is_empty() {
            return Ok(());
        }
        let mut cursor = 0usize;
        let mut counted_for_current_chunk = false;
        while cursor < tokens.len() {
            self.ensure_writer()?;
            if !counted_for_current_chunk {
                self.current_sample_count += 1;
                counted_for_current_chunk = true;
            }
            let remaining = self
                .chunk_token_capacity
                .saturating_sub(self.current_token_count)
                .max(1);
            let copy_len = remaining.min(tokens.len() - cursor);
            if let Some(writer) = &mut self.current_writer {
                for token in &tokens[cursor..cursor + copy_len] {
                    writer
                        .write_all(&token.to_le_bytes())
                        .context("write token chunk")?;
                }
            }
            self.current_token_count += copy_len;
            self.total_token_count += copy_len;
            cursor += copy_len;
            if self.current_token_count >= self.chunk_token_capacity {
                self.finish_current_chunk()?;
                counted_for_current_chunk = false;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> Result<Vec<ChunkPart>> {
        self.finish_current_chunk()?;
        Ok(self.chunks)
    }

    fn ensure_writer(&mut self) -> Result<()> {
        if self.current_writer.is_some() {
            return Ok(());
        }
        fs::create_dir_all(&self.chunk_dir)
            .with_context(|| format!("failed to create {}", self.chunk_dir.display()))?;
        let split_label = match self.split {
            SampleSplit::Train => "train",
            SampleSplit::Validation => "validation",
        };
        let file_name = format!(
            "{split_label}-chunk-{part:05}.u32le",
            part = self.part_index
        );
        let path = self.chunk_dir.join(&file_name);
        self.part_index += 1;
        self.current_writer = Some(BufWriter::new(
            fs::File::create(&path)
                .with_context(|| format!("failed to create {}", path.display()))?,
        ));
        self.current_file_name = Some(file_name);
        Ok(())
    }

    fn finish_current_chunk(&mut self) -> Result<()> {
        if let Some(mut writer) = self.current_writer.take() {
            writer.flush().context("flush token chunk")?;
            if let Some(file_name) = self.current_file_name.take() {
                self.chunks.push(ChunkPart {
                    file_name,
                    split: self.split,
                    token_count: self.current_token_count,
                    sample_count: self.current_sample_count,
                });
            }
            self.current_token_count = 0;
            self.current_sample_count = 0;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{NcaCorpusConfig, NcaSerializationConfig, NcaTokenizationConfig};
    use tempfile::tempdir;

    #[test]
    fn generate_corpus_writes_manifest_and_previews() {
        let dir = tempdir().expect("tempdir");
        let config = NcaCorpusConfig {
            output_dir: dir.path().join("nca"),
            seed: 1337,
            name: "smoke".to_string(),
            train_samples: 4,
            validation_samples: 2,
            chunk_token_capacity: 128,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: crate::config::default_families(),
        };
        let report = generate_nca_corpus(&config).expect("generate corpus");
        assert!(report.manifest_path.is_file());
        assert!(report.sample_records_path.is_file());
        assert!(report.preview_dir.is_dir());
        let manifest = crate::manifest::load_manifest(&report.manifest_path).expect("manifest");
        assert_eq!(manifest.stats.total_samples, 6);
        assert_eq!(manifest.train_token_count, report.train_token_count);
    }
}
