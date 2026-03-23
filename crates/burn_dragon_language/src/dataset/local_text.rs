use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use burn::tensor::backend::Backend;

use super::DatasetSplit;
use super::scheduler::{SequenceBatch, TokenSequenceDataset};
use crate::tokenizer::{SharedTokenizer, TokenizerConfig};

#[derive(Clone)]
pub struct LocalTextDataset {
    tokens: Vec<u32>,
    train_len: usize,
    block_size: usize,
    batch_size: usize,
    train_split_ratio: f32,
    tokenizer: SharedTokenizer,
    source_path: PathBuf,
    document_count: usize,
    preferred_logical_document_tokens: Option<usize>,
}

impl LocalTextDataset {
    pub fn new(
        cache_dir: impl AsRef<Path>,
        source_path: impl AsRef<Path>,
        block_size: usize,
        batch_size: usize,
        train_split_ratio: f32,
        tokenizer_cfg: &TokenizerConfig,
    ) -> io::Result<Self> {
        let cache_dir = cache_dir.as_ref();
        fs::create_dir_all(cache_dir)?;

        let source_path = source_path.as_ref().to_path_buf();
        let raw = fs::read_to_string(&source_path)?;
        let documents = raw
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        if documents.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "local text dataset {} did not contain any non-empty lines",
                    source_path.display()
                ),
            ));
        }

        let tokenizer_path = tokenizer_cfg.storage_path(cache_dir);
        let tokenizer = if let Some(path) = tokenizer_path {
            if path.is_file() {
                tokenizer_cfg.load(&path).map_err(io::Error::other)?
            } else {
                let tokenizer = tokenizer_cfg
                    .fit(documents.iter().map(String::as_str))
                    .map_err(io::Error::other)?;
                tokenizer_cfg
                    .save(&*tokenizer, &path)
                    .map_err(io::Error::other)?;
                tokenizer
            }
        } else {
            tokenizer_cfg
                .fit(documents.iter().map(String::as_str))
                .map_err(io::Error::other)?
        };

        let mut tokens = Vec::new();
        let mut logical_lengths = Vec::with_capacity(documents.len());
        let mut document_spans = Vec::with_capacity(documents.len());
        for document in &documents {
            tokenizer_cfg
                .validate_corpus(&*tokenizer, document.as_str())
                .map_err(io::Error::other)?;
            let encoded = tokenizer.encode(document, false, true);
            if encoded.len() <= 1 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "local text document from {} encoded to {} tokens; expected at least 2 with EOS",
                        source_path.display(),
                        encoded.len()
                    ),
                ));
            }
            logical_lengths.push(encoded.len() - 1);
            document_spans.push(encoded.len());
            tokens.extend(encoded);
        }

        if tokens.len() <= block_size + 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "encoded local text dataset {} is smaller than block size {}",
                    source_path.display(),
                    block_size
                ),
            ));
        }

        let split_ratio = train_split_ratio.clamp(0.0, 1.0);
        let mut train_document_count = ((documents.len() as f32) * split_ratio) as usize;
        train_document_count = train_document_count.clamp(1, documents.len());
        let minimum_train_tokens = block_size + 1;
        let mut train_len = document_spans
            .iter()
            .take(train_document_count)
            .copied()
            .sum::<usize>();
        while train_len < minimum_train_tokens && train_document_count < documents.len() {
            train_len += document_spans[train_document_count];
            train_document_count += 1;
        }
        if train_len < minimum_train_tokens {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "local text dataset {} cannot satisfy block size {} with {} documents",
                    source_path.display(),
                    block_size,
                    documents.len()
                ),
            ));
        }

        let preferred_logical_document_tokens = logical_lengths
            .first()
            .copied()
            .filter(|length| logical_lengths.iter().all(|candidate| candidate == length));

        Ok(Self {
            tokens,
            train_len,
            block_size,
            batch_size,
            train_split_ratio: split_ratio,
            tokenizer,
            source_path,
            document_count: documents.len(),
            preferred_logical_document_tokens,
        })
    }

    pub fn source_path(&self) -> &Path {
        &self.source_path
    }

    pub fn document_count(&self) -> usize {
        self.document_count
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
        let end = start + dst.len();
        dst.copy_from_slice(&self.tokens[start..end]);
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

    pub fn decode(&self, tokens: &[i64]) -> String {
        TokenSequenceDataset::decode(self, tokens)
    }
}

impl TokenSequenceDataset for LocalTextDataset {
    fn tokenizer(&self) -> SharedTokenizer {
        self.tokenizer.clone()
    }

    fn token_count(&self) -> usize {
        self.tokens.len()
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        let end = start + dst.len();
        dst.copy_from_slice(&self.tokens[start..end]);
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
        self.preferred_logical_document_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::{TokenizerConfig, TokenizerKind};
    use tempfile::tempdir;

    #[test]
    fn local_text_dataset_uses_non_empty_lines_as_documents() {
        let dir = tempdir().expect("tempdir");
        let source = dir.path().join("docs.txt");
        fs::write(&source, "alpha|a=1\n\nbeta_|a=2\n gamma|a=3 \n").expect("write source");

        let dataset = LocalTextDataset::new(
            dir.path().join("cache"),
            &source,
            8,
            2,
            1.0,
            &TokenizerConfig {
                vocab_path: None,
                kind: TokenizerKind::Byte(Default::default()),
            },
        )
        .expect("dataset");

        assert_eq!(dataset.document_count(), 3);
        assert_eq!(dataset.preferred_logical_document_tokens, Some(9));
        assert!(dataset.token_count() > dataset.block_size());
    }
}
