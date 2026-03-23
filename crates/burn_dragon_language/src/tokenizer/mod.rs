pub mod byte;
pub mod char_vocab;
pub mod pretokenized;
pub mod rust_bpe;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Result, anyhow};
use byte::ByteTokenizer;
use char_vocab::CharVocab;
use pretokenized::PretokenizedTokenizer;
use rust_bpe::RustBpeTokenizer;
use serde::{Deserialize, Serialize};

pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32>;
    fn decode(&self, ids: &[u32]) -> String;
    fn decode_with_options(&self, ids: &[u32], stop_at_eos: bool) -> String {
        if stop_at_eos {
            self.decode(ids)
        } else {
            self.decode(ids)
        }
    }
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn bos_id(&self) -> Option<u32>;
    fn eos_id(&self) -> Option<u32>;
    fn pad_id(&self) -> Option<u32>;
    fn unk_id(&self) -> Option<u32>;
    fn as_any(&self) -> &dyn std::any::Any;
}

pub type SharedTokenizer = Arc<dyn Tokenizer>;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TokenizerConfig {
    #[serde(default)]
    pub vocab_path: Option<PathBuf>,
    #[serde(flatten)]
    pub kind: TokenizerKind,
}

impl Default for TokenizerConfig {
    fn default() -> Self {
        Self {
            vocab_path: None,
            kind: TokenizerKind::Char(CharTokenizerConfig::default()),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TokenizerKind {
    Char(CharTokenizerConfig),
    Byte(ByteTokenizerConfig),
    Pretokenized(PretokenizedTokenizerConfig),
    RustBpe(RustBpeTokenizerConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct CharTokenizerConfig {
    #[serde(default = "default_true")]
    pub include_unknown: bool,
}

impl Default for CharTokenizerConfig {
    fn default() -> Self {
        Self {
            include_unknown: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ByteTokenizerConfig {
    #[serde(default = "default_true")]
    pub add_special_tokens: bool,
}

impl Default for ByteTokenizerConfig {
    fn default() -> Self {
        Self {
            add_special_tokens: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PretokenizedTokenizerConfig {
    pub vocab_size: usize,
    #[serde(default)]
    pub bos_id: Option<u32>,
    #[serde(default)]
    pub eos_id: Option<u32>,
    #[serde(default)]
    pub pad_id: Option<u32>,
    #[serde(default)]
    pub unk_id: Option<u32>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct RustBpeTokenizerConfig {
    pub mergeable_vocab_size: usize,
    #[serde(default)]
    pub pattern: Option<String>,
    #[serde(default)]
    pub bos_id: Option<u32>,
    #[serde(default)]
    pub eos_id: Option<u32>,
    #[serde(default)]
    pub pad_id: Option<u32>,
    #[serde(default)]
    pub unk_id: Option<u32>,
}

impl TokenizerConfig {
    pub fn storage_path(&self, cache_dir: &Path) -> Option<PathBuf> {
        match &self.kind {
            TokenizerKind::Char(_) => Some(match &self.vocab_path {
                Some(path) if path.is_absolute() => path.clone(),
                Some(path) => cache_dir.join(path),
                None => cache_dir.join("vocab.json"),
            }),
            TokenizerKind::RustBpe(_) => Some(match &self.vocab_path {
                Some(path) if path.is_absolute() => path.clone(),
                Some(path) => cache_dir.join(path),
                None => cache_dir.join("tokenizer.rustbpe.json"),
            }),
            TokenizerKind::Byte(_) | TokenizerKind::Pretokenized(_) => None,
        }
    }

    pub fn load(&self, path: &Path) -> Result<SharedTokenizer> {
        match &self.kind {
            TokenizerKind::Char(_) => {
                let vocab = CharVocab::load(path)?;
                Ok(Arc::new(vocab) as SharedTokenizer)
            }
            TokenizerKind::Byte(config) => {
                Ok(Arc::new(ByteTokenizer::new(config.add_special_tokens)) as SharedTokenizer)
            }
            TokenizerKind::Pretokenized(config) => Ok(Arc::new(PretokenizedTokenizer::new(
                config.vocab_size,
                config.bos_id,
                config.eos_id,
                config.pad_id,
                config.unk_id,
            )) as SharedTokenizer),
            TokenizerKind::RustBpe(config) => Ok(Arc::new(RustBpeTokenizer::load(
                path,
                config.mergeable_vocab_size,
                config.bos_id,
                config.eos_id,
                config.pad_id,
                config.unk_id,
            )?) as SharedTokenizer),
        }
    }

    pub fn fit<'a, I>(&self, texts: I) -> Result<SharedTokenizer>
    where
        I: Iterator<Item = &'a str>,
    {
        match &self.kind {
            TokenizerKind::Char(config) => {
                let vocab = CharVocab::fit(texts, config.include_unknown)?;
                Ok(Arc::new(vocab) as SharedTokenizer)
            }
            TokenizerKind::Byte(config) => {
                Ok(Arc::new(ByteTokenizer::new(config.add_special_tokens)) as SharedTokenizer)
            }
            TokenizerKind::Pretokenized(config) => Ok(Arc::new(PretokenizedTokenizer::new(
                config.vocab_size,
                config.bos_id,
                config.eos_id,
                config.pad_id,
                config.unk_id,
            )) as SharedTokenizer),
            TokenizerKind::RustBpe(config) => {
                let mut tokenizer = RustBpeTokenizer::new_untrained(
                    config.mergeable_vocab_size,
                    config.pattern.as_deref(),
                    config.bos_id,
                    config.eos_id,
                    config.pad_id,
                    config.unk_id,
                )?;
                tokenizer.train_from_texts(texts)?;
                Ok(Arc::new(tokenizer) as SharedTokenizer)
            }
        }
    }

    pub fn save(&self, tokenizer: &dyn Tokenizer, path: &Path) -> Result<()> {
        match &self.kind {
            TokenizerKind::Char(_) => {
                let vocab = tokenizer
                    .as_any()
                    .downcast_ref::<CharVocab>()
                    .ok_or_else(|| anyhow!("expected char tokenizer"))?;
                vocab.save(path)
            }
            TokenizerKind::RustBpe(_) => {
                let tokenizer = tokenizer
                    .as_any()
                    .downcast_ref::<RustBpeTokenizer>()
                    .ok_or_else(|| anyhow!("expected rust_bpe tokenizer"))?;
                tokenizer.save(path)
            }
            TokenizerKind::Byte(_) | TokenizerKind::Pretokenized(_) => Ok(()),
        }
    }

    pub fn requires_strict_coverage(&self) -> bool {
        matches!(&self.kind, TokenizerKind::Char(config) if !config.include_unknown)
    }

    pub fn validate_corpus(&self, tokenizer: &dyn Tokenizer, text: &str) -> Result<()> {
        match &self.kind {
            TokenizerKind::Char(config) if !config.include_unknown => {
                let vocab = tokenizer
                    .as_any()
                    .downcast_ref::<CharVocab>()
                    .ok_or_else(|| anyhow!("expected char tokenizer"))?;
                for ch in text.chars() {
                    if !vocab.contains(ch) {
                        return Err(anyhow!(
                            "vocabulary missing character {ch:?} found in dataset"
                        ));
                    }
                }
                Ok(())
            }
            TokenizerKind::Byte(_) | TokenizerKind::Pretokenized(_) => Ok(()),
            _ => Ok(()),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            TokenizerKind::Char(_) => "char",
            TokenizerKind::Byte(_) => "byte",
            TokenizerKind::Pretokenized(_) => "pretokenized",
            TokenizerKind::RustBpe(_) => "rust_bpe",
        }
    }
}

fn default_true() -> bool {
    true
}
