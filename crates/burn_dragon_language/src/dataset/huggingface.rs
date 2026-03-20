use std::collections::HashMap;
use std::fs;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use burn::tensor::backend::Backend;
use csv::ReaderBuilder;
use hf_hub::api::sync::ApiBuilder;
use hf_hub::{Repo, RepoType};
use memmap2::{Mmap, MmapOptions};
use parquet::file::reader::{FileReader, SerializedFileReader};
use parquet::record::{ListAccessor, RowAccessor};
use rayon::prelude::*;
use serde_json::Value;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use super::DatasetSplit;
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::config::{HuggingFaceDatasetConfig, HuggingFaceRecordFormat};
use crate::tokenizer::{SharedTokenizer, TokenizerConfig, TokenizerKind};

const DEFAULT_RECORD_DELIMITER: &str = "\n";
const PREPARED_TOKEN_CACHE_VERSION: u32 = 2;
const DEFAULT_PREPARED_TOKEN_CHUNK_TOKENS: usize = 4 * 1024 * 1024;
const DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT: usize = 8;
const DEFAULT_PREPARED_PARALLEL_FILE_BATCH: usize = 4;

#[derive(serde::Serialize, serde::Deserialize)]
struct PreparedTokenCacheMetadata {
    version: u32,
    tokenizer_len: u32,
    train_token_count: usize,
    val_token_count: usize,
    token_count: usize,
    chunk_token_capacity: usize,
    chunks: Vec<PreparedTokenChunkMetadata>,
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct PreparedTokenChunkMetadata {
    file_name: String,
    token_offset: usize,
    token_count: usize,
}

#[derive(serde::Serialize)]
struct PreparedTokenCacheKey<'a> {
    version: u32,
    repo_id: &'a str,
    revision: Option<&'a str>,
    format: HuggingFaceRecordFormat,
    sequence_field: Option<&'a str>,
    max_records: Option<usize>,
    tokenizer_len: u32,
    train_paths: Vec<String>,
    validation_paths: Vec<String>,
}

#[derive(Clone)]
enum TokenStorage {
    Owned(Arc<Vec<u32>>),
    Chunked(Arc<ChunkedTokens>),
}

#[derive(Clone)]
struct ChunkedTokens {
    chunks: Arc<Vec<ChunkedTokenFile>>,
    len: usize,
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

impl TokenStorage {
    fn len(&self) -> usize {
        match self {
            Self::Owned(tokens) => tokens.len(),
            Self::Chunked(chunked) => chunked.len,
        }
    }

    fn copy_into(&self, start: usize, dst: &mut [u32]) {
        match self {
            Self::Owned(tokens) => {
                let end = start + dst.len();
                dst.copy_from_slice(&tokens[start..end]);
            }
            Self::Chunked(chunked) => chunked.copy_into(start, dst),
        }
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
            let available = chunk.token_count.saturating_sub(chunk_start);
            let copy_len = available.min(remaining);
            let src_end = chunk_start + copy_len;
            dst[written..written + copy_len].copy_from_slice(&chunk_tokens[chunk_start..src_end]);
            cursor += copy_len;
            written += copy_len;
            remaining -= copy_len;
        }
    }

    fn chunk_data(&self, chunk_idx: usize) -> Arc<Mmap> {
        let mut cache = self.cache.lock().expect("chunk cache poisoned");
        cache.tick = cache.tick.wrapping_add(1);
        let tick = cache.tick;
        if let Some(entry) = cache.entries.get_mut(&chunk_idx) {
            entry.last_used_tick = tick;
            return Arc::clone(&entry.mmap);
        }

        let chunk = &self.chunks[chunk_idx];
        let file = fs::File::open(&chunk.path).unwrap_or_else(|err| {
            panic!(
                "failed to open prepared token chunk {}: {err}",
                chunk.path.display()
            )
        });
        let mmap = unsafe { MmapOptions::new().map(&file) }.unwrap_or_else(|err| {
            panic!(
                "failed to mmap prepared token chunk {}: {err}",
                chunk.path.display()
            )
        });
        let len = mmap.len() / 4;
        if mmap.len() % 4 != 0 || len != chunk.token_count {
            panic!(
                "prepared token chunk {} size mismatch: bytes={} tokens={} expected_tokens={}",
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
                .expect("chunk cache should not be empty");
            cache.entries.remove(&evict_key);
        }
        mmap
    }
}

fn mmap_as_u32_slice(mmap: &Mmap, len: usize) -> &[u32] {
    debug_assert_eq!(mmap.len(), len * 4);
    #[cfg(not(target_endian = "little"))]
    compile_error!("prepared token mmap currently assumes little-endian hosts");
    unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const u32, len) }
}

fn prepared_token_chunk_tokens() -> usize {
    std::env::var("BDH_PREPARED_TOKEN_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_PREPARED_TOKEN_CHUNK_TOKENS)
}

fn runtime_chunk_cache_limit() -> usize {
    std::env::var("BDH_PREPARED_TOKEN_CACHE_LIMIT")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_RUNTIME_CHUNK_CACHE_LIMIT)
}

fn prepared_parallel_file_batch() -> usize {
    let available = std::thread::available_parallelism()
        .map(|count| count.get().max(1))
        .unwrap_or(DEFAULT_PREPARED_PARALLEL_FILE_BATCH);
    std::env::var("BDH_PREPARED_TOKEN_PARALLEL_FILES")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .map(|value| value.min(available))
        .unwrap_or_else(|| available.min(DEFAULT_PREPARED_PARALLEL_FILE_BATCH))
}

#[derive(Clone)]
pub struct HuggingFaceDataset {
    tokens: TokenStorage,
    train_len: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
    repo_id: String,
    revision: Option<String>,
}

impl HuggingFaceDataset {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cache_dir: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
        hf_cfg: &HuggingFaceDatasetConfig,
    ) -> io::Result<Self> {
        if !is_pretokenized(hf_cfg) && hf_cfg.text_fields.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "huggingface dataset requires at least one text field",
            ));
        }
        if is_pretokenized(hf_cfg) && !matches!(tokenizer_cfg.kind, TokenizerKind::Pretokenized(_))
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "pretokenized huggingface datasets require tokenizer.type = `pretokenized`",
            ));
        }

        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;
        let hf_cache_dir = cache_dir.join("huggingface");
        fs::create_dir_all(&hf_cache_dir)?;

        let token = hf_cfg
            .token
            .clone()
            .or_else(|| std::env::var("HF_TOKEN").ok())
            .filter(|value| !value.trim().is_empty());

        let mut api_builder = ApiBuilder::new().with_cache_dir(hf_cache_dir.clone());
        if let Some(token) = token {
            api_builder = api_builder.with_token(Some(token));
        }
        let api = api_builder.build().map_err(io::Error::other)?;

        let repo = if let Some(revision) = &hf_cfg.revision {
            Repo::with_revision(hf_cfg.repo_id.clone(), RepoType::Dataset, revision.clone())
        } else {
            Repo::new(hf_cfg.repo_id.clone(), RepoType::Dataset)
        };
        let repo = api.repo(repo);

        let train_files =
            resolve_hf_files(&repo, &hf_cache_dir, &hf_cfg.train_files, hf_cfg, true)?;
        let validation_files = resolve_hf_files(
            &repo,
            &hf_cache_dir,
            &hf_cfg.validation_files,
            hf_cfg,
            false,
        )?;

        let (tokens, mut train_len, val_token_count, tokenizer) = if is_pretokenized(hf_cfg) {
            let tokenizer = tokenizer_cfg
                .fit(std::iter::empty())
                .map_err(io::Error::other)?;
            let tokenizer_len = tokenizer.len() as u32;
            let (tokens, train_len, val_token_count) = if hf_cfg.max_records.is_some() {
                info!(
                    "building bounded pretokenized dataset for {} without prepared cache (max_records={:?})",
                    hf_cfg.repo_id, hf_cfg.max_records
                );
                let (tokens, train_len, val_token_count) = collect_pretokenized_tokens_via_repo(
                    &repo,
                    &hf_cfg.repo_id,
                    &train_files,
                    &validation_files,
                    hf_cfg,
                    tokenizer_len,
                )?;
                (
                    TokenStorage::Owned(Arc::new(tokens)),
                    train_len,
                    val_token_count,
                )
            } else {
                let train_paths = resolve_hf_local_paths(&repo, &train_files)?;
                let validation_paths = resolve_hf_local_paths(&repo, &validation_files)?;
                let cache_key = build_prepared_token_cache_key(
                    hf_cfg,
                    &train_paths,
                    &validation_paths,
                    tokenizer_len,
                )?;
                let cache_paths = prepared_token_cache_paths(cache_dir, &cache_key);

                if let Some(cached) = open_prepared_token_cache(&cache_paths, tokenizer_len)? {
                    info!(
                        "loaded prepared pretokenized cache for {} (tokens={}, train_tokens={}, val_tokens={}, runtime_chunk_cache_limit={})",
                        hf_cfg.repo_id,
                        cached.0.len(),
                        cached.1,
                        cached.2,
                        runtime_chunk_cache_limit()
                    );
                    cached
                } else {
                    info!(
                        "building prepared pretokenized cache for {} from {} train files and {} validation files",
                        hf_cfg.repo_id,
                        train_paths.len(),
                        validation_paths.len()
                    );
                    let built = build_prepared_token_cache_streaming(
                        &cache_paths,
                        &hf_cfg.repo_id,
                        &train_paths,
                        &validation_paths,
                        hf_cfg,
                        tokenizer_len,
                    )?;
                    info!(
                        "wrote prepared pretokenized cache for {} to {}",
                        hf_cfg.repo_id,
                        cache_paths.0.display()
                    );
                    built
                }
            };
            (tokens, train_len, val_token_count, tokenizer)
        } else {
            let mut train_records = Vec::new();
            for file in &train_files {
                if hf_cfg
                    .max_records
                    .is_some_and(|limit| train_records.len() >= limit)
                {
                    break;
                }
                let path = repo
                    .get(file)
                    .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
                collect_records(&path, hf_cfg, hf_cfg.max_records, &mut train_records)?;
            }

            let mut val_records = Vec::new();
            for file in &validation_files {
                let path = repo
                    .get(file)
                    .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
                collect_records(&path, hf_cfg, hf_cfg.max_records, &mut val_records)?;
            }

            if train_records.is_empty() && val_records.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "huggingface dataset contains no records",
                ));
            }

            let tokenizer_path = tokenizer_cfg.storage_path(cache_dir);
            let tokenizer = if let Some(path) = tokenizer_path {
                if path.is_file() {
                    tokenizer_cfg.load(&path).map_err(io::Error::other)?
                } else {
                    let tokenizer = tokenizer_cfg
                        .fit(record_iter(&train_records, &val_records))
                        .map_err(io::Error::other)?;
                    tokenizer_cfg
                        .save(&*tokenizer, &path)
                        .map_err(io::Error::other)?;
                    tokenizer
                }
            } else {
                tokenizer_cfg
                    .fit(record_iter(&train_records, &val_records))
                    .map_err(io::Error::other)?
            };

            for record in record_iter(&train_records, &val_records) {
                tokenizer_cfg
                    .validate_corpus(&*tokenizer, record)
                    .map_err(io::Error::other)?;
            }

            let (tokens, train_len, val_token_count) =
                encode_records(&hf_cfg.repo_id, &tokenizer, train_records, val_records);
            (
                TokenStorage::Owned(Arc::new(tokens)),
                train_len,
                val_token_count,
                tokenizer,
            )
        };

        if tokens.len() <= block_size + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "encoded huggingface dataset smaller than block size",
            ));
        }

        if val_token_count == 0 {
            let split_ratio = train_split_ratio.clamp(0.0, 1.0);
            let mut ratio_len = ((tokens.len() as f32) * split_ratio) as usize;
            let min_len = block_size + 1;
            let max_len = tokens.len().saturating_sub(1);
            if ratio_len < min_len {
                ratio_len = min_len;
            } else if ratio_len > max_len {
                ratio_len = max_len;
            }
            train_len = ratio_len;
        } else if train_len <= block_size {
            train_len = (block_size + 1).min(tokens.len().saturating_sub(1));
        }

        Ok(Self {
            tokens,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: train_split_ratio.clamp(0.0, 1.0),
            tokenizer: tokenizer.clone(),
            repo_id: hf_cfg.repo_id.clone(),
            revision: hf_cfg.revision.clone(),
        })
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
        self.tokens.len()
    }

    pub fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        self.tokens.copy_into(start, dst);
    }

    pub fn train_len(&self) -> usize {
        self.train_len
    }

    pub fn repo_id(&self) -> &str {
        &self.repo_id
    }

    pub fn revision(&self) -> Option<&str> {
        self.revision.as_deref()
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

    pub fn decode(&self, tokens: &[i64]) -> String {
        TokenSequenceDataset::decode(self, tokens)
    }
}

impl TokenSequenceDataset for HuggingFaceDataset {
    fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        self.tokens.copy_into(start, dst);
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
}

fn is_pretokenized(cfg: &HuggingFaceDatasetConfig) -> bool {
    cfg.sequence_field
        .as_ref()
        .is_some_and(|field| !field.trim().is_empty())
}

fn resolve_hf_local_paths(
    repo: &hf_hub::api::sync::ApiRepo,
    files: &[String],
) -> io::Result<Vec<std::path::PathBuf>> {
    files
        .iter()
        .map(|file| {
            repo.get(file)
                .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))
        })
        .collect()
}

fn build_prepared_token_cache_key(
    cfg: &HuggingFaceDatasetConfig,
    train_paths: &[std::path::PathBuf],
    validation_paths: &[std::path::PathBuf],
    tokenizer_len: u32,
) -> io::Result<String> {
    let key = PreparedTokenCacheKey {
        version: PREPARED_TOKEN_CACHE_VERSION,
        repo_id: &cfg.repo_id,
        revision: cfg.revision.as_deref(),
        format: cfg.format,
        sequence_field: cfg.sequence_field.as_deref(),
        max_records: cfg.max_records,
        tokenizer_len,
        train_paths: train_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
        validation_paths: validation_paths
            .iter()
            .map(|path| path.to_string_lossy().into_owned())
            .collect(),
    };
    let payload = serde_json::to_vec(&key).map_err(io::Error::other)?;
    Ok(format!("{:x}", Sha256::digest(payload)))
}

fn prepared_token_cache_paths(
    cache_dir: &Path,
    key: &str,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let prepared_dir = cache_dir.join("prepared");
    (
        prepared_dir.join(format!("{key}.json")),
        prepared_dir.join(format!("{key}.chunks")),
    )
}

fn open_prepared_token_cache(
    paths: &(std::path::PathBuf, std::path::PathBuf),
    tokenizer_len: u32,
) -> io::Result<Option<(TokenStorage, usize, usize)>> {
    let (meta_path, chunk_dir) = paths;
    if !meta_path.is_file() || !chunk_dir.is_dir() {
        return Ok(None);
    }

    let meta_bytes = fs::read(meta_path)?;
    let meta: PreparedTokenCacheMetadata =
        serde_json::from_slice(&meta_bytes).map_err(io::Error::other)?;
    if meta.version != PREPARED_TOKEN_CACHE_VERSION || meta.tokenizer_len != tokenizer_len {
        return Ok(None);
    }

    if meta.chunks.is_empty() {
        return Ok(None);
    }

    let mut chunks = Vec::with_capacity(meta.chunks.len());
    for chunk in &meta.chunks {
        let path = chunk_dir.join(&chunk.file_name);
        if !path.is_file() {
            return Ok(None);
        }
        let bytes = fs::metadata(&path)?.len() as usize;
        let expected = chunk.token_count.saturating_mul(4);
        if bytes != expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "prepared token chunk {} byte length mismatch (expected={} actual={})",
                    path.display(),
                    expected,
                    bytes
                ),
            ));
        }
        chunks.push(ChunkedTokenFile {
            path,
            token_offset: chunk.token_offset,
            token_count: chunk.token_count,
        });
    }
    let storage = TokenStorage::Chunked(Arc::new(ChunkedTokens {
        len: meta.token_count,
        chunks: Arc::new(chunks),
        cache_limit: runtime_chunk_cache_limit(),
        cache: Arc::new(Mutex::new(ChunkRuntimeCache::default())),
    }));
    Ok(Some((
        storage,
        meta.train_token_count,
        meta.val_token_count,
    )))
}

#[cfg(test)]
fn save_prepared_token_cache(
    paths: &(std::path::PathBuf, std::path::PathBuf),
    tokenizer_len: u32,
    tokens: &[u32],
    train_token_count: usize,
    val_token_count: usize,
) -> io::Result<()> {
    let chunk_token_capacity = prepared_token_chunk_tokens().min(tokens.len().max(1));
    save_prepared_token_cache_with_chunk_capacity(
        paths,
        tokenizer_len,
        tokens,
        train_token_count,
        val_token_count,
        chunk_token_capacity,
    )
}

#[cfg(test)]
fn save_prepared_token_cache_with_chunk_capacity(
    paths: &(std::path::PathBuf, std::path::PathBuf),
    tokenizer_len: u32,
    tokens: &[u32],
    train_token_count: usize,
    val_token_count: usize,
    chunk_token_capacity: usize,
) -> io::Result<()> {
    let (meta_path, chunk_dir) = paths;
    if let Some(parent) = meta_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if chunk_dir.exists() {
        fs::remove_dir_all(chunk_dir)?;
    }
    fs::create_dir_all(chunk_dir)?;

    let chunk_token_capacity = chunk_token_capacity.max(1).min(tokens.len().max(1));
    let mut chunks = Vec::new();
    for (chunk_idx, slice) in tokens.chunks(chunk_token_capacity).enumerate() {
        let file_name = format!("chunk-{chunk_idx:05}.u32le");
        let chunk_path = chunk_dir.join(&file_name);
        let mut writer = BufWriter::new(fs::File::create(&chunk_path)?);
        write_token_slice(&mut writer, slice)?;
        writer.flush()?;
        chunks.push(PreparedTokenChunkMetadata {
            file_name,
            token_offset: chunk_idx * chunk_token_capacity,
            token_count: slice.len(),
        });
    }

    let meta = PreparedTokenCacheMetadata {
        version: PREPARED_TOKEN_CACHE_VERSION,
        tokenizer_len,
        train_token_count,
        val_token_count,
        token_count: tokens.len(),
        chunk_token_capacity,
        chunks,
    };

    fs::write(
        meta_path,
        serde_json::to_vec_pretty(&meta).map_err(io::Error::other)?,
    )?;
    Ok(())
}

fn build_prepared_token_cache_streaming(
    paths: &(std::path::PathBuf, std::path::PathBuf),
    repo_id: &str,
    train_paths: &[std::path::PathBuf],
    validation_paths: &[std::path::PathBuf],
    cfg: &HuggingFaceDatasetConfig,
    tokenizer_len: u32,
) -> io::Result<(TokenStorage, usize, usize)> {
    let (meta_path, chunk_dir) = paths;
    if let Some(parent) = meta_path.parent() {
        fs::create_dir_all(parent)?;
    }
    if chunk_dir.exists() {
        fs::remove_dir_all(chunk_dir)?;
    }
    fs::create_dir_all(chunk_dir)?;

    let parallel_files = prepared_parallel_file_batch();
    let chunk_token_capacity = prepared_token_chunk_tokens();
    info!(
        "prepared pretokenized cache builder for {} using parallel_file_batch={parallel_files}, chunk_token_capacity={chunk_token_capacity}, runtime_chunk_cache_limit={}",
        repo_id,
        runtime_chunk_cache_limit()
    );

    let (train_chunks, train_token_count) = stream_pretokenized_paths_to_chunks(
        train_paths,
        cfg,
        tokenizer_len,
        chunk_dir,
        "train",
        parallel_files,
        chunk_token_capacity,
    )?;
    let (val_chunks, val_token_count) = stream_pretokenized_paths_to_chunks(
        validation_paths,
        cfg,
        tokenizer_len,
        chunk_dir,
        "validation",
        parallel_files,
        chunk_token_capacity,
    )?;

    if train_token_count == 0 && val_token_count == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("huggingface dataset {repo_id} contains no token sequences"),
        ));
    }

    let mut chunks = train_chunks;
    chunks.extend(val_chunks);
    let meta = PreparedTokenCacheMetadata {
        version: PREPARED_TOKEN_CACHE_VERSION,
        tokenizer_len,
        train_token_count,
        val_token_count,
        token_count: train_token_count + val_token_count,
        chunk_token_capacity,
        chunks,
    };
    fs::write(
        meta_path,
        serde_json::to_vec_pretty(&meta).map_err(io::Error::other)?,
    )?;

    open_prepared_token_cache(paths, tokenizer_len)?
        .ok_or_else(|| io::Error::other("prepared token cache was not readable after writing"))
}

fn collect_pretokenized_tokens_via_repo(
    repo: &hf_hub::api::sync::ApiRepo,
    repo_id: &str,
    train_files: &[String],
    validation_files: &[String],
    cfg: &HuggingFaceDatasetConfig,
    tokenizer_len: u32,
) -> io::Result<(Vec<u32>, usize, usize)> {
    let mut train_sequences = Vec::new();
    for file in train_files {
        if cfg
            .max_records
            .is_some_and(|limit| train_sequences.len() >= limit)
        {
            break;
        }
        let path = repo
            .get(file)
            .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
        collect_token_sequences(&path, cfg, cfg.max_records, &mut train_sequences)?;
    }

    let mut val_sequences = Vec::new();
    for file in validation_files {
        let path = repo
            .get(file)
            .map_err(|err| io::Error::other(format!("failed to download {file}: {err}")))?;
        collect_token_sequences(&path, cfg, cfg.max_records, &mut val_sequences)?;
    }

    if train_sequences.is_empty() && val_sequences.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("huggingface dataset {repo_id} contains no token sequences"),
        ));
    }

    flatten_token_sequences(repo_id, tokenizer_len, train_sequences, val_sequences)
}

fn stream_pretokenized_paths_to_chunks(
    paths: &[std::path::PathBuf],
    cfg: &HuggingFaceDatasetConfig,
    tokenizer_len: u32,
    chunk_dir: &Path,
    split_label: &str,
    parallel_files: usize,
    chunk_token_capacity: usize,
) -> io::Result<(Vec<PreparedTokenChunkMetadata>, usize)> {
    let mut token_count = 0usize;
    let mut prepared_chunks = Vec::new();
    let mut token_offset = 0usize;

    for (batch_idx, batch) in paths.chunks(parallel_files.max(1)).enumerate() {
        let batch_start = batch_idx * parallel_files.max(1);
        let file_batches = if batch.len() <= 1 {
            batch
                .iter()
                .enumerate()
                .map(|(offset, path)| {
                    write_pretokenized_file_chunks(
                        path,
                        cfg,
                        tokenizer_len,
                        chunk_dir,
                        &format!(
                            "{split_label}-{file_idx:06}",
                            file_idx = batch_start + offset
                        ),
                        chunk_token_capacity,
                    )
                })
                .collect::<Vec<_>>()
        } else {
            batch
                .par_iter()
                .enumerate()
                .map(|(offset, path)| {
                    write_pretokenized_file_chunks(
                        path,
                        cfg,
                        tokenizer_len,
                        chunk_dir,
                        &format!(
                            "{split_label}-{file_idx:06}",
                            file_idx = batch_start + offset
                        ),
                        chunk_token_capacity,
                    )
                })
                .collect::<Vec<_>>()
        };

        let file_batches = file_batches.into_iter().collect::<io::Result<Vec<_>>>()?;
        for file_chunks in file_batches {
            token_count += file_chunks.token_count;
            for chunk in file_chunks.chunks {
                let token_count = chunk.token_count;
                prepared_chunks.push(PreparedTokenChunkMetadata {
                    file_name: chunk.file_name,
                    token_offset,
                    token_count,
                });
                token_offset += token_count;
            }
        }
    }

    Ok((prepared_chunks, token_count))
}

struct FileChunkBuild {
    token_count: usize,
    chunks: Vec<FileChunkPart>,
}

struct FileChunkPart {
    file_name: String,
    token_count: usize,
}

struct ChunkFileWriter {
    chunk_dir: PathBuf,
    file_prefix: String,
    chunk_token_capacity: usize,
    current_part: usize,
    current_token_count: usize,
    token_count: usize,
    chunks: Vec<FileChunkPart>,
    writer: Option<BufWriter<fs::File>>,
}

impl ChunkFileWriter {
    fn new(chunk_dir: &Path, file_prefix: &str, chunk_token_capacity: usize) -> Self {
        Self {
            chunk_dir: chunk_dir.to_path_buf(),
            file_prefix: file_prefix.to_string(),
            chunk_token_capacity: chunk_token_capacity.max(1),
            current_part: 0,
            current_token_count: 0,
            token_count: 0,
            chunks: Vec::new(),
            writer: None,
        }
    }

    fn push_sequence(&mut self, sequence: &[u32]) -> io::Result<()> {
        if sequence.len() < 2 {
            return Ok(());
        }
        let mut cursor = 0usize;
        while cursor < sequence.len() {
            self.ensure_writer()?;
            let remaining_chunk = self
                .chunk_token_capacity
                .saturating_sub(self.current_token_count)
                .max(1);
            let copy_len = remaining_chunk.min(sequence.len() - cursor);
            if let Some(writer) = &mut self.writer {
                write_token_slice(writer, &sequence[cursor..cursor + copy_len])?;
            }
            self.current_token_count += copy_len;
            self.token_count += copy_len;
            cursor += copy_len;
            if self.current_token_count >= self.chunk_token_capacity {
                self.finish_current_chunk()?;
            }
        }
        Ok(())
    }

    fn finish(mut self) -> io::Result<FileChunkBuild> {
        self.finish_current_chunk()?;
        Ok(FileChunkBuild {
            token_count: self.token_count,
            chunks: self.chunks,
        })
    }

    fn ensure_writer(&mut self) -> io::Result<()> {
        if self.writer.is_none() {
            let file_name = format!("{}-{:05}.u32le", self.file_prefix, self.current_part);
            let path = self.chunk_dir.join(file_name);
            self.writer = Some(BufWriter::new(fs::File::create(path)?));
        }
        Ok(())
    }

    fn finish_current_chunk(&mut self) -> io::Result<()> {
        if self.current_token_count == 0 {
            self.writer = None;
            return Ok(());
        }
        if let Some(mut writer) = self.writer.take() {
            writer.flush()?;
        }
        self.chunks.push(FileChunkPart {
            file_name: format!("{}-{:05}.u32le", self.file_prefix, self.current_part),
            token_count: self.current_token_count,
        });
        self.current_part += 1;
        self.current_token_count = 0;
        Ok(())
    }
}

fn write_pretokenized_file_chunks(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    tokenizer_len: u32,
    chunk_dir: &Path,
    file_prefix: &str,
    chunk_token_capacity: usize,
) -> io::Result<FileChunkBuild> {
    let mut writer = ChunkFileWriter::new(chunk_dir, file_prefix, chunk_token_capacity);
    stream_token_sequences(path, cfg, None, |sequence| {
        if let Some(max_id) = sequence.iter().copied().max()
            && max_id >= tokenizer_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "token id {max_id} exceeded tokenizer vocab size {} for {}",
                    tokenizer_len,
                    path.display()
                ),
            ));
        }
        writer.push_sequence(sequence)
    })?;
    writer.finish()
}

fn write_token_slice<W: Write>(writer: &mut W, tokens: &[u32]) -> io::Result<()> {
    for &token in tokens {
        writer.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}

fn resolve_hf_files(
    repo: &hf_hub::api::sync::ApiRepo,
    hf_cache_dir: &Path,
    explicit_files: &[String],
    cfg: &HuggingFaceDatasetConfig,
    prefer_train_split: bool,
) -> io::Result<Vec<String>> {
    if !explicit_files.is_empty() {
        return Ok(explicit_files.to_vec());
    }
    if !prefer_train_split || !cfg.auto_discover_train_files {
        return Ok(Vec::new());
    }

    let mut files = match repo.info() {
        Ok(info) => info
            .siblings
            .into_iter()
            .map(|file| file.rfilename)
            .filter(|path| matches_hf_format(path, cfg.format))
            .collect::<Vec<_>>(),
        Err(err) => {
            warn!(
                "failed to inspect repo {} via hf api ({err}); falling back to local cache discovery",
                cfg.repo_id
            );
            discover_hf_files_from_local_cache(hf_cache_dir, cfg)?
        }
    };

    if prefer_train_split {
        let train_like = files
            .iter()
            .any(|path| path.contains("train") || path.contains("/default/"));
        if train_like {
            files.retain(|path| path.contains("train") || path.contains("/default/"));
        }
    }

    files.sort();
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no {} files discovered for dataset {}",
                hf_format_name(cfg.format),
                cfg.repo_id
            ),
        ));
    }
    Ok(files)
}

fn discover_hf_files_from_local_cache(
    hf_cache_dir: &Path,
    cfg: &HuggingFaceDatasetConfig,
) -> io::Result<Vec<String>> {
    let repo_dir = hf_cache_dir.join(format!("datasets--{}", cfg.repo_id.replace('/', "--")));
    let snapshot_root = resolve_hf_snapshot_root(&repo_dir, cfg.revision.as_deref())?;
    let mut files = Vec::new();
    collect_relative_files(&snapshot_root, &snapshot_root, &mut files)?;
    files.retain(|path| matches_hf_format(path, cfg.format));
    if files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "no cached {} files found for dataset {} under {}",
                hf_format_name(cfg.format),
                cfg.repo_id,
                snapshot_root.display()
            ),
        ));
    }
    Ok(files)
}

fn resolve_hf_snapshot_root(repo_dir: &Path, revision: Option<&str>) -> io::Result<PathBuf> {
    let snapshots_dir = repo_dir.join("snapshots");
    let refs_dir = repo_dir.join("refs");
    let preferred_ref = revision.unwrap_or("main");
    let ref_path = refs_dir.join(preferred_ref);
    if let Ok(snapshot_id) = fs::read_to_string(&ref_path) {
        let snapshot_id = snapshot_id.trim();
        let snapshot_root = snapshots_dir.join(snapshot_id);
        if snapshot_root.is_dir() {
            return Ok(snapshot_root);
        }
    }

    let mut snapshots = fs::read_dir(&snapshots_dir)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().is_dir())
        .map(|entry| entry.path())
        .collect::<Vec<_>>();
    snapshots.sort();
    snapshots
        .pop()
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "no cached hf snapshots found"))
}

fn collect_relative_files(root: &Path, dir: &Path, files: &mut Vec<String>) -> io::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            collect_relative_files(root, &path, files)?;
        } else if path.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(io::Error::other)?
                .to_string_lossy()
                .replace('\\', "/");
            files.push(rel);
        }
    }
    Ok(())
}

fn matches_hf_format(path: &str, format: HuggingFaceRecordFormat) -> bool {
    match format {
        HuggingFaceRecordFormat::Jsonl => path.ends_with(".jsonl") || path.ends_with(".json"),
        HuggingFaceRecordFormat::Text => path.ends_with(".txt"),
        HuggingFaceRecordFormat::Parquet => path.ends_with(".parquet"),
        HuggingFaceRecordFormat::Csv => path.ends_with(".csv"),
    }
}

fn hf_format_name(format: HuggingFaceRecordFormat) -> &'static str {
    match format {
        HuggingFaceRecordFormat::Jsonl => "json/jsonl",
        HuggingFaceRecordFormat::Text => "text",
        HuggingFaceRecordFormat::Parquet => "parquet",
        HuggingFaceRecordFormat::Csv => "csv",
    }
}

fn encode_records(
    repo_id: &str,
    tokenizer: &SharedTokenizer,
    train_records: Vec<String>,
    val_records: Vec<String>,
) -> (Vec<u32>, usize, usize) {
    let mut tokens = Vec::new();
    let mut train_len = 0usize;

    for record in train_records {
        let mut encoded = tokenizer.encode(record.as_str(), false, false);
        if encoded.len() < 2 {
            warn!(
                "skipping short training record from {} ({} tokens)",
                repo_id,
                encoded.len()
            );
            continue;
        }
        train_len += encoded.len();
        tokens.append(&mut encoded);
    }

    let mut val_token_count = 0usize;
    for record in val_records {
        let mut encoded = tokenizer.encode(record.as_str(), false, false);
        if encoded.len() < 2 {
            warn!(
                "skipping short validation record from {} ({} tokens)",
                repo_id,
                encoded.len()
            );
            continue;
        }
        val_token_count += encoded.len();
        tokens.append(&mut encoded);
    }

    (tokens, train_len, val_token_count)
}

fn flatten_token_sequences(
    repo_id: &str,
    tokenizer_len: u32,
    train_sequences: Vec<Vec<u32>>,
    val_sequences: Vec<Vec<u32>>,
) -> io::Result<(Vec<u32>, usize, usize)> {
    let mut tokens = Vec::new();
    let mut train_len = 0usize;
    let mut val_token_count = 0usize;

    for sequence in train_sequences {
        if sequence.len() < 2 {
            warn!(
                "skipping short training token sequence from {} ({} tokens)",
                repo_id,
                sequence.len()
            );
            continue;
        }
        if let Some(max_id) = sequence.iter().copied().max()
            && max_id >= tokenizer_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "token id {max_id} exceeded tokenizer vocab size {} for {}",
                    tokenizer_len, repo_id
                ),
            ));
        }
        train_len += sequence.len();
        tokens.extend(sequence);
    }

    for sequence in val_sequences {
        if sequence.len() < 2 {
            warn!(
                "skipping short validation token sequence from {} ({} tokens)",
                repo_id,
                sequence.len()
            );
            continue;
        }
        if let Some(max_id) = sequence.iter().copied().max()
            && max_id >= tokenizer_len
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "token id {max_id} exceeded tokenizer vocab size {} for {}",
                    tokenizer_len, repo_id
                ),
            ));
        }
        val_token_count += sequence.len();
        tokens.extend(sequence);
    }

    Ok((tokens, train_len, val_token_count))
}

fn collect_token_sequences(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    sequences: &mut Vec<Vec<u32>>,
) -> io::Result<()> {
    stream_token_sequences(path, cfg, max_records, |sequence| {
        sequences.push(sequence.to_vec());
        Ok(())
    })
}

fn stream_token_sequences<F>(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    mut on_sequence: F,
) -> io::Result<()>
where
    F: FnMut(&[u32]) -> io::Result<()>,
{
    match cfg.format {
        HuggingFaceRecordFormat::Jsonl => {
            stream_jsonl_token_sequences(path, cfg, max_records, &mut on_sequence)
        }
        HuggingFaceRecordFormat::Parquet => {
            stream_parquet_token_sequences(path, cfg, max_records, &mut on_sequence)
        }
        HuggingFaceRecordFormat::Text | HuggingFaceRecordFormat::Csv => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "pretokenized datasets do not support {:?} record format",
                cfg.format
            ),
        )),
    }
}

fn stream_jsonl_token_sequences<F>(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    on_sequence: &mut F,
) -> io::Result<()>
where
    F: FnMut(&[u32]) -> io::Result<()>,
{
    let field = cfg
        .sequence_field
        .as_deref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing sequence_field"))?;
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let mut emitted = 0usize;

    for line in reader.lines() {
        if max_records.is_some_and(|limit| emitted >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse JSON record from {}: {err}", path.display()),
            )
        })?;
        let sequence_value = value.get(field).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing `{field}` in dataset record"),
            )
        })?;
        let sequence = extract_token_sequence_from_json(sequence_value)?;
        if !sequence.is_empty() {
            on_sequence(&sequence)?;
            emitted += 1;
        }
    }

    Ok(())
}

fn stream_parquet_token_sequences<F>(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    on_sequence: &mut F,
) -> io::Result<()>
where
    F: FnMut(&[u32]) -> io::Result<()>,
{
    let debug = std::env::var_os("BDH_HF_PRETOKENIZED_DEBUG").is_some();
    let field = cfg
        .sequence_field
        .as_deref()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "missing sequence_field"))?;
    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file).map_err(io::Error::other)?;

    let row_iter = reader.get_row_iter(None).map_err(io::Error::other)?;
    let mut emitted = 0usize;
    for row in row_iter {
        if max_records.is_some_and(|limit| emitted >= limit) {
            break;
        }
        let row = row.map_err(io::Error::other)?;
        let field_index = row
            .get_column_iter()
            .position(|(name, _)| name.as_str() == field)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "missing token field `{field}` in parquet file {}",
                        path.display()
                    ),
                )
            })?;
        if debug && emitted == 0 {
            let columns = row
                .get_column_iter()
                .map(|(name, field)| format!("{name}={field:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            eprintln!(
                "[hf-pretokenized] first row path={} columns={columns}",
                path.display()
            );
        }
        let list = row.get_list(field_index).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "failed to read token list field `{field}` from {}: {err}",
                    path.display()
                ),
            )
        })?;
        let sequence = extract_token_sequence_from_list(list)?;
        if !sequence.is_empty() {
            if debug && emitted < 3 {
                eprintln!(
                    "[hf-pretokenized] path={} push sequence_len={} next_total={}",
                    path.display(),
                    sequence.len(),
                    emitted + 1
                );
            }
            on_sequence(&sequence)?;
            emitted += 1;
        }
    }

    if debug {
        eprintln!(
            "[hf-pretokenized] done path={} collected_total={} limit={max_records:?}",
            path.display(),
            emitted
        );
    }

    Ok(())
}

fn extract_token_sequence_from_json(value: &Value) -> io::Result<Vec<u32>> {
    let array = value.as_array().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "pretokenized dataset field must be a JSON array",
        )
    })?;
    let mut sequence = Vec::with_capacity(array.len());
    for value in array {
        let token = match value {
            Value::Number(number) => number.as_u64().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "token sequence must contain non-negative integers",
                )
            })? as u32,
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unexpected token value {other} in pretokenized dataset"),
                ));
            }
        };
        sequence.push(token);
    }
    Ok(sequence)
}

fn extract_token_sequence_from_list(list: &parquet::record::List) -> io::Result<Vec<u32>> {
    let mut sequence = Vec::with_capacity(list.len());
    for index in 0..list.len() {
        let token = list
            .get_long(index)
            .or_else(|_| list.get_int(index).map(i64::from))
            .map_err(|err| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("pretokenized parquet list must contain integers: {err}"),
                )
            })?;
        if token < 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "token sequences must contain non-negative ids",
            ));
        }
        sequence.push(token as u32);
    }
    Ok(sequence)
}

fn collect_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    match cfg.format {
        HuggingFaceRecordFormat::Jsonl => collect_jsonl_records(path, cfg, max_records, records),
        HuggingFaceRecordFormat::Text => collect_text_records(path, cfg, max_records, records),
        HuggingFaceRecordFormat::Csv => collect_csv_records(path, cfg, max_records, records),
        HuggingFaceRecordFormat::Parquet => {
            collect_parquet_records(path, cfg, max_records, records)
        }
    }
}

fn collect_jsonl_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);

    for line in reader.lines() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(&line).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse JSON record from {}: {err}", path.display()),
            )
        })?;

        match render_hf_record(cfg, extract_fields_from_json(cfg, &value)?)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn collect_text_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = BufReader::new(file);
    let field_name = cfg
        .text_fields
        .first()
        .cloned()
        .unwrap_or_else(|| "text".to_string());

    for line in reader.lines() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let text = line?;
        if text.trim().is_empty() {
            continue;
        }
        let mut fields = HashMap::new();
        fields.insert(field_name.as_str(), text);
        match render_hf_record(cfg, fields)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn collect_csv_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let headers = reader.headers().map_err(|err| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to read CSV headers from {}: {err}", path.display()),
        )
    })?;

    let mut index_map = HashMap::new();
    for field in &cfg.text_fields {
        let idx = headers
            .iter()
            .position(|header| header == field)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("missing field `{}` in csv file {}", field, path.display()),
                )
            })?;
        index_map.insert(field.as_str(), idx);
    }

    for record in reader.records() {
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let record = record.map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to read CSV record from {}: {err}", path.display()),
            )
        })?;
        if record.is_empty() {
            continue;
        }
        let mut field_values = HashMap::new();
        for field in &cfg.text_fields {
            let idx = *index_map.get(field.as_str()).expect("field index missing");
            let value = record.get(idx).unwrap_or("").to_string();
            field_values.insert(field.as_str(), value);
        }
        match render_hf_record(cfg, field_values)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn collect_parquet_records(
    path: &Path,
    cfg: &HuggingFaceDatasetConfig,
    max_records: Option<usize>,
    records: &mut Vec<String>,
) -> io::Result<()> {
    let file = fs::File::open(path)?;
    let reader = SerializedFileReader::new(file).map_err(io::Error::other)?;

    let row_iter = reader.get_row_iter(None).map_err(io::Error::other)?;
    for row in row_iter {
        let row = row.map_err(io::Error::other)?;

        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        let mut field_values = HashMap::new();
        for field in &cfg.text_fields {
            let idx = row
                .get_column_iter()
                .position(|(name, _)| name.as_str() == field)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "missing field `{}` in parquet file {}",
                            field,
                            path.display()
                        ),
                    )
                })?;

            let value = if let Ok(s) = row.get_string(idx) {
                s.clone()
            } else if let Ok(bytes) = row.get_bytes(idx) {
                String::from_utf8_lossy(bytes.data()).to_string()
            } else {
                row.get_column_iter()
                    .nth(idx)
                    .map(|(_, field)| field.to_string())
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!(
                                "unable to render parquet field `{}` in {}",
                                field,
                                path.display()
                            ),
                        )
                    })?
            };

            field_values.insert(field.as_str(), value);
        }

        match render_hf_record(cfg, field_values)? {
            Some(rendered) => records.push(rendered),
            None => continue,
        }
    }

    Ok(())
}

fn extract_fields_from_json<'a>(
    cfg: &'a HuggingFaceDatasetConfig,
    value: &'a Value,
) -> io::Result<HashMap<&'a str, String>> {
    let mut map = HashMap::new();
    for field in &cfg.text_fields {
        let field_value = value.get(field).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("missing `{field}` in dataset record"),
            )
        })?;
        let text = match field_value {
            Value::String(s) => s.clone(),
            other => other.to_string(),
        };
        map.insert(field.as_str(), text);
    }
    Ok(map)
}

fn render_hf_record(
    cfg: &HuggingFaceDatasetConfig,
    fields: HashMap<&str, String>,
) -> io::Result<Option<String>> {
    if fields.is_empty() {
        return Ok(None);
    }

    let rendered = if let Some(template) = &cfg.template {
        render_template(template, &fields)?
    } else {
        let mut ordered = Vec::with_capacity(cfg.text_fields.len());
        for field in &cfg.text_fields {
            let value = fields.get(field.as_str()).cloned().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("template field `{field}` missing from record"),
                )
            })?;
            ordered.push(value);
        }

        let mut joined = if ordered.len() == 1 {
            ordered.into_iter().next().unwrap()
        } else {
            ordered.join(if cfg.field_separator.is_empty() {
                DEFAULT_RECORD_DELIMITER
            } else {
                cfg.field_separator.as_str()
            })
        };
        if !joined.ends_with('\n') {
            joined.push('\n');
        }
        joined
    };

    Ok(Some(rendered))
}

fn render_template(template: &str, fields: &HashMap<&str, String>) -> io::Result<String> {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '{' {
            let mut key = String::new();
            let mut closed = false;
            for next in chars.by_ref() {
                if next == '}' {
                    closed = true;
                    break;
                }
                key.push(next);
            }
            if !closed {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unclosed template placeholder",
                ));
            }
            if key.trim().is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "empty template placeholder {}",
                ));
            }
            let field_key = key.trim();
            let value = fields.get(field_key).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("unknown template placeholder {{{field_key}}}"),
                )
            })?;
            result.push_str(value);
        } else {
            result.push(ch);
        }
    }

    if !result.ends_with('\n') {
        result.push('\n');
    }

    Ok(result)
}

fn record_iter<'a>(train: &'a [String], val: &'a [String]) -> impl Iterator<Item = &'a str> {
    train
        .iter()
        .map(String::as_str)
        .chain(val.iter().map(String::as_str))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    fn storage_to_vec(storage: &TokenStorage) -> Vec<u32> {
        let mut tokens = vec![0u32; storage.len()];
        storage.copy_into(0, &mut tokens);
        tokens
    }

    #[test]
    fn extracts_pretokenized_json_sequences() {
        let value = json!([1, 2, 3, 4]);
        let sequence = extract_token_sequence_from_json(&value).expect("sequence");
        assert_eq!(sequence, vec![1, 2, 3, 4]);
    }

    #[test]
    fn flatten_token_sequences_rejects_ids_outside_vocab() {
        let err = flatten_token_sequences("repo/test", 8, vec![vec![1, 2, 9]], Vec::new())
            .expect_err("token ids outside the tokenizer vocab should be rejected");
        assert!(err.to_string().contains("exceeded tokenizer vocab size"));
    }

    #[test]
    fn render_template_replaces_fields() {
        let mut fields = HashMap::new();
        fields.insert("prompt", "Question".to_string());
        fields.insert("answer", "Answer".to_string());
        let rendered = render_template("{prompt}\n\n{answer}", &fields).expect("render template");
        assert_eq!(rendered, "Question\n\nAnswer\n");
    }

    #[test]
    fn prepared_token_cache_round_trip() {
        let dir = tempdir().expect("tempdir");
        let paths = prepared_token_cache_paths(dir.path(), "cache-key");
        let tokens = vec![1, 2, 3, 4, 5];
        save_prepared_token_cache(&paths, 64, &tokens, 3, 2).expect("save cache");
        let loaded = open_prepared_token_cache(&paths, 64)
            .expect("load cache")
            .expect("cache hit");
        assert_eq!(storage_to_vec(&loaded.0), tokens);
        assert_eq!(loaded.1, 3);
        assert_eq!(loaded.2, 2);
    }

    #[test]
    fn prepared_token_cache_rejects_mismatched_tokenizer_len() {
        let dir = tempdir().expect("tempdir");
        let paths = prepared_token_cache_paths(dir.path(), "cache-key");
        save_prepared_token_cache(&paths, 64, &[1, 2, 3], 3, 0).expect("save cache");
        let loaded = open_prepared_token_cache(&paths, 128).expect("load cache");
        assert!(loaded.is_none());
    }

    #[test]
    fn prepared_token_cache_copy_across_chunk_boundaries() {
        let dir = tempdir().expect("tempdir");
        let paths = prepared_token_cache_paths(dir.path(), "cache-key");
        let tokens = vec![1, 2, 3, 4, 5, 6, 7];
        save_prepared_token_cache_with_chunk_capacity(&paths, 64, &tokens, 5, 2, 2)
            .expect("save cache");
        let loaded = open_prepared_token_cache(&paths, 64)
            .expect("load cache")
            .expect("cache hit");
        let mut window = vec![0u32; 5];
        loaded.0.copy_into(1, &mut window);
        assert_eq!(window, vec![2, 3, 4, 5, 6]);
    }
}
