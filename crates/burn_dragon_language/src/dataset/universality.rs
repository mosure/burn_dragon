use std::collections::HashMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use burn::tensor::backend::Backend;
use memmap2::{Mmap, MmapOptions};

use super::DatasetSplit;
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::tokenizer::{SharedTokenizer, TokenizerConfig, TokenizerKind};

const DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT: usize = 8;
const DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT: usize = 64;

#[derive(Clone)]
enum UniversalityStorage {
    Manifest(ManifestStorage),
    OnTheFly(OnTheFlyStorage),
}

#[derive(Clone)]
struct ManifestStorage {
    tokens: Arc<ChunkedTokens>,
    manifest_path: PathBuf,
    preferred_logical_document_tokens: Option<usize>,
}

#[derive(Clone)]
struct OnTheFlyStorage {
    corpus: Arc<burn_dragon_universality::OnlineNcaCorpus>,
    config_path: PathBuf,
    cache_limit: usize,
    cache: Arc<Mutex<DocumentRuntimeCache>>,
    train_probe_summary: burn_dragon_universality::RuntimeCorpusSummary,
    validation_probe_summary: burn_dragon_universality::RuntimeCorpusSummary,
}

#[derive(Clone)]
struct ChunkedTokens {
    chunks: Arc<Vec<ChunkedTokenFile>>,
    cache_limit: usize,
    cache: Arc<Mutex<ChunkRuntimeCache>>,
}

#[derive(Clone)]
struct ChunkedTokenFile {
    path: PathBuf,
    token_offset: usize,
    token_count: usize,
}

#[derive(Default)]
struct ChunkRuntimeCache {
    tick: u64,
    entries: HashMap<usize, CachedChunk>,
}

struct CachedChunk {
    mmap: Arc<Mmap>,
    last_used_tick: u64,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct RuntimeDocumentKey {
    split_tag: u8,
    sample_index: usize,
}

#[derive(Default)]
struct DocumentRuntimeCache {
    tick: u64,
    entries: HashMap<RuntimeDocumentKey, CachedDocument>,
}

struct CachedDocument {
    tokens: Arc<Vec<u32>>,
    last_used_tick: u64,
}

#[derive(Clone)]
pub struct UniversalityDataset {
    storage: UniversalityStorage,
    train_len: usize,
    token_count: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
    dataset_name: String,
}

impl UniversalityDataset {
    pub fn new(
        manifest_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let tokenizer = validate_pretokenized_tokenizer(tokenizer_cfg)?;
        let manifest_path = manifest_path.as_ref().to_path_buf();
        let manifest =
            burn_dragon_universality::load_manifest(&manifest_path).map_err(io::Error::other)?;
        validate_tokenizer_against_manifest(tokenizer.as_ref(), &manifest.tokenizer)?;
        let preferred_logical_document_tokens =
            fixed_manifest_logical_document_tokens(&manifest).map_err(io::Error::other)?;
        if matches!(
            manifest.corpus_kind,
            burn_dragon_universality::CorpusKind::Nca
        ) && matches!(
            preferred_logical_document_tokens,
            Some(logical_document_tokens) if block_size > logical_document_tokens
        ) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "training.block_size={} exceeds prepared NCA logical document length {}; regenerate the manifest with longer single-rule rollouts",
                    block_size,
                    preferred_logical_document_tokens.unwrap_or_default()
                ),
            ));
        }

        let manifest_dir = manifest_path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let chunk_root = manifest_dir.join(&manifest.chunk_dir);
        let mut chunks = Vec::with_capacity(manifest.chunks.len());
        for chunk in &manifest.chunks {
            let path = chunk_root.join(&chunk.file_name);
            if !path.is_file() {
                return Err(io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("universality chunk missing: {}", path.display()),
                ));
            }
            let byte_len = fs::metadata(&path)?.len() as usize;
            let expected_bytes = chunk.token_count.saturating_mul(4);
            if byte_len != expected_bytes {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "universality chunk {} size mismatch (expected={} actual={})",
                        path.display(),
                        expected_bytes,
                        byte_len
                    ),
                ));
            }
            chunks.push(ChunkedTokenFile {
                path,
                token_offset: chunk.token_offset,
                token_count: chunk.token_count,
            });
        }

        Ok(Self {
            storage: UniversalityStorage::Manifest(ManifestStorage {
                tokens: Arc::new(ChunkedTokens {
                    chunks: Arc::new(chunks),
                    cache_limit: runtime_chunk_cache_limit(),
                    cache: Arc::new(Mutex::new(ChunkRuntimeCache::default())),
                }),
                manifest_path,
                preferred_logical_document_tokens,
            }),
            train_len: manifest.train_token_count,
            token_count: manifest.token_count,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer,
            dataset_name: manifest.dataset_name,
        })
    }

    pub fn new_on_the_fly(
        config_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        min_logical_document_tokens: Option<usize>,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let tokenizer = validate_pretokenized_tokenizer(tokenizer_cfg)?;
        let config_path = config_path.as_ref().to_path_buf();
        let target_logical_document_tokens = min_logical_document_tokens
            .unwrap_or(block_size)
            .max(block_size);
        let corpus =
            burn_dragon_universality::OnlineNcaCorpus::load_with_min_logical_document_tokens(
                &config_path,
                Some(target_logical_document_tokens),
            )
            .map_err(io::Error::other)?;
        validate_tokenizer_against_manifest(tokenizer.as_ref(), corpus.tokenizer_manifest())?;
        let document_token_count = corpus.document_token_count();
        let logical_document_tokens = document_token_count.saturating_sub(1);
        if logical_document_tokens == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "on-the-fly NCA corpus must yield at least one input token per document",
            ));
        }
        if block_size > logical_document_tokens {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "training.block_size={} exceeds adapted on-the-fly NCA logical document length {}",
                    block_size, logical_document_tokens
                ),
            ));
        }

        let train_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Train)
            .map_err(io::Error::other)?;
        let validation_probe_summary = corpus
            .default_probe_summary(burn_dragon_universality::SampleSplit::Validation)
            .map_err(io::Error::other)?;

        let train_len = corpus.train_token_count();
        let token_count = corpus.total_token_count();
        let train_split_ratio = if token_count == 0 {
            1.0
        } else {
            train_len as f32 / token_count as f32
        };
        let dataset_name = config_file_display_name(
            config_path
                .file_stem()
                .and_then(|value| value.to_str())
                .unwrap_or("nca"),
        )
        .to_string();

        Ok(Self {
            storage: UniversalityStorage::OnTheFly(OnTheFlyStorage {
                corpus: Arc::new(corpus),
                config_path,
                cache_limit: runtime_document_cache_limit(batch_size),
                cache: Arc::new(Mutex::new(DocumentRuntimeCache::default())),
                train_probe_summary,
                validation_probe_summary,
            }),
            train_len,
            token_count,
            block_size,
            batch_size,
            train_split_ratio,
            tokenizer,
            dataset_name,
        })
    }

    pub fn dataset_name(&self) -> &str {
        &self.dataset_name
    }

    pub fn source_path(&self) -> &Path {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => &storage.manifest_path,
            UniversalityStorage::OnTheFly(storage) => &storage.config_path,
        }
    }

    pub fn source_kind_label(&self) -> &'static str {
        match &self.storage {
            UniversalityStorage::Manifest(_) => "universality manifest",
            UniversalityStorage::OnTheFly(_) => "on-the-fly universality NCA",
        }
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    pub fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn token_count(&self) -> usize {
        self.token_count
    }

    pub fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => storage.tokens.copy_into(start, dst),
            UniversalityStorage::OnTheFly(storage) => storage.copy_into(start, self.train_len, dst),
        }
    }

    pub fn train_len(&self) -> usize {
        self.train_len
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }

    pub fn sample_batch<B: Backend>(
        &self,
        split: DatasetSplit,
        device: &B::Device,
    ) -> SequenceBatch<B> {
        super::scheduler::sample_batch(self, split, device)
    }

    pub fn train_probe_summary(&self) -> Option<&burn_dragon_universality::RuntimeCorpusSummary> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(&storage.train_probe_summary),
        }
    }

    pub fn validation_probe_summary(
        &self,
    ) -> Option<&burn_dragon_universality::RuntimeCorpusSummary> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(&storage.validation_probe_summary),
        }
    }

    pub fn runtime_document_cache_limit(&self) -> Option<usize> {
        match &self.storage {
            UniversalityStorage::Manifest(_) => None,
            UniversalityStorage::OnTheFly(storage) => Some(storage.cache_limit),
        }
    }
}

impl TokenSequenceDataset for UniversalityDataset {
    fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    fn token_count(&self) -> usize {
        self.token_count
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        self.copy_token_range(start, dst);
    }

    fn train_len(&self) -> usize {
        self.train_len
    }

    fn block_size(&self) -> usize {
        self.block_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn train_split_ratio(&self) -> f32 {
        self.train_split_ratio
    }

    fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
        match &self.storage {
            UniversalityStorage::Manifest(storage) => storage.preferred_logical_document_tokens,
            UniversalityStorage::OnTheFly(storage) => {
                Some(storage.corpus.document_token_count().saturating_sub(1))
            }
        }
    }
}

impl OnTheFlyStorage {
    fn copy_into(&self, start: usize, train_len: usize, dst: &mut [u32]) {
        let mut remaining = dst.len();
        let mut written = 0usize;
        let mut cursor = start;
        let document_token_count = self.corpus.document_token_count();

        while remaining > 0 {
            let (split, split_offset, split_sample_count) = if cursor < train_len {
                (
                    burn_dragon_universality::SampleSplit::Train,
                    cursor,
                    self.corpus.train_samples(),
                )
            } else {
                (
                    burn_dragon_universality::SampleSplit::Validation,
                    cursor.saturating_sub(train_len),
                    self.corpus.validation_samples(),
                )
            };

            let sample_index = split_offset / document_token_count;
            if sample_index >= split_sample_count {
                panic!(
                    "on-the-fly NCA token request out of range: split={split:?} sample_index={sample_index} sample_count={split_sample_count} start={start} len={}",
                    dst.len()
                );
            }
            let token_index = split_offset % document_token_count;
            let copy_len = document_token_count
                .saturating_sub(token_index)
                .min(remaining);
            let document_tokens = self.document_tokens(split, sample_index);
            dst[written..written + copy_len]
                .copy_from_slice(&document_tokens[token_index..token_index + copy_len]);
            written += copy_len;
            remaining -= copy_len;
            cursor += copy_len;
        }
    }

    fn document_tokens(
        &self,
        split: burn_dragon_universality::SampleSplit,
        sample_index: usize,
    ) -> Arc<Vec<u32>> {
        let key = RuntimeDocumentKey {
            split_tag: split_tag(split),
            sample_index,
        };
        {
            let mut cache = self
                .cache
                .lock()
                .expect("universality runtime cache poisoned");
            cache.tick = cache.tick.wrapping_add(1);
            let tick = cache.tick;
            if let Some(entry) = cache.entries.get_mut(&key) {
                entry.last_used_tick = tick;
                return Arc::clone(&entry.tokens);
            }
        }

        let generated_tokens = Arc::new(
            self.corpus
                .generate_document_tokens(split, sample_index)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to generate on-the-fly NCA sample split={split:?} sample_index={sample_index}: {error:#}"
                    )
                }),
        );

        let mut cache = self
            .cache
            .lock()
            .expect("universality runtime cache poisoned");
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        if let Some(entry) = cache.entries.get_mut(&key) {
            entry.last_used_tick = tick;
            return Arc::clone(&entry.tokens);
        }
        cache.entries.insert(
            key,
            CachedDocument {
                tokens: Arc::clone(&generated_tokens),
                last_used_tick: tick,
            },
        );
        while cache.entries.len() > self.cache_limit {
            let evict_key = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(key, _)| *key)
                .expect("universality runtime cache should not be empty");
            cache.entries.remove(&evict_key);
        }
        generated_tokens
    }
}

fn split_tag(split: burn_dragon_universality::SampleSplit) -> u8 {
    match split {
        burn_dragon_universality::SampleSplit::Train => 0,
        burn_dragon_universality::SampleSplit::Validation => 1,
    }
}

impl ChunkedTokens {
    fn copy_into(&self, start: usize, dst: &mut [u32]) {
        let mut remaining = dst.len();
        let mut written = 0usize;
        let mut cursor = start;

        while remaining > 0 {
            let chunk_idx = self
                .chunks
                .partition_point(|chunk| chunk.token_offset + chunk.token_count <= cursor)
                .min(self.chunks.len().saturating_sub(1));
            let chunk = &self.chunks[chunk_idx];
            let chunk_data = self.chunk_data(chunk_idx);
            let chunk_tokens = mmap_as_u32_slice(&chunk_data, chunk.token_count);
            let chunk_start = cursor.saturating_sub(chunk.token_offset);
            let copy_len = chunk.token_count.saturating_sub(chunk_start).min(remaining);
            dst[written..written + copy_len]
                .copy_from_slice(&chunk_tokens[chunk_start..chunk_start + copy_len]);
            cursor += copy_len;
            written += copy_len;
            remaining -= copy_len;
        }
    }

    fn chunk_data(&self, chunk_idx: usize) -> Arc<Mmap> {
        let mut cache = self
            .cache
            .lock()
            .expect("universality chunk cache poisoned");
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        if let Some(entry) = cache.entries.get_mut(&chunk_idx) {
            entry.last_used_tick = tick;
            return Arc::clone(&entry.mmap);
        }

        let chunk = &self.chunks[chunk_idx];
        let file = fs::File::open(&chunk.path).unwrap_or_else(|err| {
            panic!(
                "failed to open universality chunk {}: {err}",
                chunk.path.display()
            )
        });
        let mmap = unsafe { MmapOptions::new().map(&file) }.unwrap_or_else(|err| {
            panic!(
                "failed to mmap universality chunk {}: {err}",
                chunk.path.display()
            )
        });
        let len = mmap.len() / 4;
        if mmap.len() % 4 != 0 || len != chunk.token_count {
            panic!(
                "universality chunk {} size mismatch: bytes={} tokens={} expected_tokens={}",
                chunk.path.display(),
                mmap.len(),
                len,
                chunk.token_count
            );
        }
        let mmap = Arc::new(mmap);
        cache.entries.insert(
            chunk_idx,
            CachedChunk {
                mmap: Arc::clone(&mmap),
                last_used_tick: tick,
            },
        );
        while cache.entries.len() > self.cache_limit {
            let evict_key = cache
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used_tick)
                .map(|(idx, _)| *idx)
                .expect("universality chunk cache should not be empty");
            cache.entries.remove(&evict_key);
        }
        mmap
    }
}

fn validate_pretokenized_tokenizer(tokenizer_cfg: &TokenizerConfig) -> io::Result<SharedTokenizer> {
    if !matches!(tokenizer_cfg.kind, TokenizerKind::Pretokenized(_)) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "universality datasets require tokenizer.type = `pretokenized`",
        ));
    }
    tokenizer_cfg
        .fit(std::iter::empty())
        .map_err(io::Error::other)
}

fn validate_tokenizer_against_manifest(
    tokenizer: &dyn crate::tokenizer::Tokenizer,
    manifest: &burn_dragon_universality::UniversalityTokenizerManifest,
) -> io::Result<()> {
    if tokenizer.len() != manifest.vocab_size {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "universality dataset tokenizer vocab mismatch (config={} manifest={})",
                tokenizer.len(),
                manifest.vocab_size
            ),
        ));
    }
    if tokenizer.bos_id() != manifest.bos_id
        || tokenizer.eos_id() != manifest.eos_id
        || tokenizer.pad_id() != manifest.pad_id
        || tokenizer.unk_id() != manifest.unk_id
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality dataset tokenizer special ids do not match manifest",
        ));
    }
    Ok(())
}

fn config_file_display_name(file_stem: &str) -> &str {
    file_stem
}

fn runtime_chunk_cache_limit() -> usize {
    std::env::var("BDH_PREPARED_TOKEN_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT)
}

fn runtime_document_cache_limit(batch_size: usize) -> usize {
    std::env::var("BDH_UNIVERSALITY_RUNTIME_DOCUMENT_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or_else(|| DEFAULT_RUNTIME_DOCUMENT_CACHE_LIMIT.max(batch_size.saturating_mul(4)))
}

fn fixed_manifest_logical_document_tokens(
    manifest: &burn_dragon_universality::UniversalityCorpusManifest,
) -> io::Result<Option<usize>> {
    let train_samples = manifest.stats.train_samples;
    let val_samples = manifest.stats.validation_samples;
    let train_doc_tokens = if train_samples > 0 {
        let per_doc = manifest.train_token_count / train_samples;
        (manifest.train_token_count % train_samples == 0).then_some(per_doc)
    } else {
        None
    };
    let val_doc_tokens = if val_samples > 0 {
        let per_doc = manifest.val_token_count / val_samples;
        (manifest.val_token_count % val_samples == 0).then_some(per_doc)
    } else {
        None
    };
    let document_token_count = match (train_doc_tokens, val_doc_tokens) {
        (Some(train), Some(val)) if train == val => Some(train),
        (Some(train), None) => Some(train),
        (None, Some(val)) => Some(val),
        (None, None) => None,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "universality manifest has inconsistent prepared document lengths across splits",
            ));
        }
    };

    match document_token_count {
        Some(0) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "universality manifest document token count must be > 0",
        )),
        Some(count) => Ok(Some(count.saturating_sub(1))),
        None => Ok(None),
    }
}

fn mmap_as_u32_slice(mmap: &Mmap, len: usize) -> &[u32] {
    debug_assert_eq!(mmap.len(), len * 4);
    #[cfg(not(target_endian = "little"))]
    compile_error!("universality mmap currently assumes little-endian hosts");
    unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, len) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig};
    use burn_dragon_universality::config::NcaCorpusConfig;
    use burn_dragon_universality::{
        NcaSerializationConfig, NcaTokenizationConfig, generate_nca_corpus,
    };
    use tempfile::tempdir;

    fn pretokenized_tokenizer() -> TokenizerConfig {
        TokenizerConfig {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                vocab_size: 50_257,
                bos_id: None,
                eos_id: Some(50_256),
                pad_id: None,
                unk_id: None,
            }),
        }
    }

    fn fixed_runtime_config() -> NcaCorpusConfig {
        let mut config = NcaCorpusConfig {
            output_dir: "ignored".into(),
            seed: 1337,
            name: "runtime".to_string(),
            train_samples: 8,
            validation_samples: 4,
            chunk_token_capacity: 1024,
            serialization: NcaSerializationConfig::default(),
            tokenization: NcaTokenizationConfig::default(),
            families: burn_dragon_universality::config::default_families(),
        };
        for family in &mut config.families {
            family.grid_size =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 12, max: 12 });
            family.steps = Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
            family.state_count =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 10, max: 10 });
            family.step_stride =
                Some(burn_dragon_universality::UsizeRangeConfig { min: 2, max: 2 });
            family.start_step = Some(burn_dragon_universality::UsizeRangeConfig { min: 0, max: 0 });
            family.identity_bias =
                Some(burn_dragon_universality::FloatRangeConfig { min: 0.0, max: 0.0 });
            family.temperature =
                Some(burn_dragon_universality::FloatRangeConfig { min: 0.0, max: 0.0 });
        }
        config
    }

    #[test]
    fn universality_dataset_loads_generated_manifest() {
        let dir = tempdir().expect("tempdir");
        let corpus_dir = dir.path().join("corpus");
        let mut config = fixed_runtime_config();
        config.output_dir = corpus_dir.clone();
        config.train_samples = 4;
        config.validation_samples = 2;
        config.chunk_token_capacity = 128;
        config.name = "dataset".to_string();
        let report = generate_nca_corpus(&config).expect("generate corpus");
        let dataset =
            UniversalityDataset::new(&report.manifest_path, 16, 2, 0.9, &pretokenized_tokenizer())
                .expect("load universality dataset");
        assert_eq!(
            dataset.token_count(),
            report.train_token_count + report.val_token_count
        );
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(380)
        );
        let mut buffer = vec![0u32; 17];
        dataset.copy_token_range(0, &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
    }

    #[test]
    fn nca_manifest_rejects_block_sizes_longer_than_prepared_document() {
        let dir = tempdir().expect("tempdir");
        let corpus_dir = dir.path().join("corpus");
        let mut config = fixed_runtime_config();
        config.output_dir = corpus_dir.clone();
        config.train_samples = 4;
        config.validation_samples = 2;
        config.chunk_token_capacity = 128;
        config.name = "dataset".to_string();
        let report = generate_nca_corpus(&config).expect("generate corpus");
        let error = UniversalityDataset::new(
            &report.manifest_path,
            512,
            2,
            0.9,
            &pretokenized_tokenizer(),
        )
        .expect_err("manifest should reject overlong block size");
        assert!(
            error
                .to_string()
                .contains("exceeds prepared NCA logical document length"),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn on_the_fly_universality_dataset_is_deterministic() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");
        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset");
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(380)
        );

        let mut first = vec![0u32; 32];
        let mut second = vec![0u32; 32];
        dataset.copy_token_range(0, &mut first);
        dataset.copy_token_range(0, &mut second);
        assert_eq!(first, second);
    }

    #[test]
    fn on_the_fly_universality_dataset_spans_documents_without_materializing_corpus() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        let document_token_count =
            burn_dragon_universality::fixed_document_token_count(&config).expect("doc tokens");
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");
        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            32,
            2,
            None,
            &pretokenized_tokenizer(),
        )
        .expect("load on-the-fly dataset");
        let mut buffer = vec![0u32; 48];
        dataset.copy_token_range(document_token_count.saturating_sub(24), &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
        assert_eq!(
            dataset.train_len(),
            config.train_samples * document_token_count
        );
    }

    #[test]
    fn on_the_fly_universality_dataset_adapts_document_length_for_large_block_size() {
        let dir = tempdir().expect("tempdir");
        let config_path = dir.path().join("nca.toml");
        let config = fixed_runtime_config();
        fs::write(&config_path, toml::to_string_pretty(&config).expect("toml"))
            .expect("write config");

        let dataset = UniversalityDataset::new_on_the_fly(
            &config_path,
            4096,
            16,
            Some(4096),
            &pretokenized_tokenizer(),
        )
        .expect("load adapted on-the-fly dataset");

        assert!(dataset.block_size() == 4096);
        assert_eq!(
            dataset.preferred_logical_document_tokens(DatasetSplit::Train),
            Some(4104)
        );
        let mut buffer = vec![0u32; 4097];
        dataset.copy_token_range(0, &mut buffer);
        assert!(buffer.iter().any(|value| *value != 0));
    }
}
