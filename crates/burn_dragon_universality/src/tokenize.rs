use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rustbpe::Tokenizer as RustBpeInner;
use serde::Deserialize;

use crate::config::NcaTokenizationConfig;
use crate::manifest::{TokenizerFamily, UniversalityTokenizerManifest};

pub enum CorpusTokenizer {
    Gpt2ByteCompatible {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
    PatchTokenIds {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
    RustBpe {
        inner: RustBpeInner,
        mergeable_vocab_size: usize,
        bos_id: Option<u32>,
        eos_id: Option<u32>,
        pad_id: Option<u32>,
        unk_id: Option<u32>,
        tokenizer_id: String,
    },
}

impl CorpusTokenizer {
    pub fn from_config(config: &NcaTokenizationConfig) -> Result<Self> {
        match config {
            NcaTokenizationConfig::Gpt2ByteCompatible { vocab_size, eos_id } => {
                Ok(Self::Gpt2ByteCompatible {
                    vocab_size: *vocab_size,
                    eos_id: *eos_id,
                })
            }
            NcaTokenizationConfig::PatchTokenIds { vocab_size, eos_id } => {
                Ok(Self::PatchTokenIds {
                    vocab_size: *vocab_size,
                    eos_id: *eos_id,
                })
            }
            NcaTokenizationConfig::RustBpe {
                vocab_path,
                mergeable_vocab_size,
                bos_id,
                eos_id,
                pad_id,
                unk_id,
            } => {
                let payload = fs::read_to_string(vocab_path)
                    .with_context(|| format!("failed to read {}", vocab_path.display()))?;
                let record: RustBpeRecord = serde_json::from_str(&payload)
                    .with_context(|| format!("failed to parse {}", vocab_path.display()))?;
                let tokenizer_id = format!("rust_bpe:{}", vocab_path.display());
                let inner = RustBpeInner::from_merges(
                    record.pattern,
                    record
                        .merges
                        .into_iter()
                        .map(|merge| ((merge.left, merge.right), merge.token_id))
                        .collect::<HashMap<_, _>>(),
                )
                .map_err(|err| anyhow!("failed to load rust_bpe tokenizer: {err}"))?;
                Ok(Self::RustBpe {
                    inner,
                    mergeable_vocab_size: *mergeable_vocab_size,
                    bos_id: *bos_id,
                    eos_id: *eos_id,
                    pad_id: *pad_id,
                    unk_id: *unk_id,
                    tokenizer_id,
                })
            }
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        match self {
            Self::Gpt2ByteCompatible { eos_id, .. } => {
                let mut tokens = text
                    .as_bytes()
                    .iter()
                    .map(|&byte| byte as u32)
                    .collect::<Vec<_>>();
                if let Some(eos_id) = eos_id {
                    tokens.push(*eos_id);
                }
                tokens
            }
            Self::PatchTokenIds { .. } => {
                panic!("patch token ids tokenizer requires encode_patch_tokens, not encode(text)")
            }
            Self::RustBpe {
                inner,
                bos_id,
                eos_id,
                ..
            } => {
                let mut tokens = Vec::new();
                if let Some(bos_id) = bos_id {
                    tokens.push(*bos_id);
                }
                tokens.extend(inner.encode(text));
                if let Some(eos_id) = eos_id {
                    tokens.push(*eos_id);
                }
                tokens
            }
        }
    }

    pub fn encode_patch_tokens(&self, patch_tokens: &[u32]) -> Result<Vec<u32>> {
        match self {
            Self::PatchTokenIds { vocab_size, eos_id } => {
                let mut tokens =
                    Vec::with_capacity(patch_tokens.len() + usize::from(eos_id.is_some()));
                for &token in patch_tokens {
                    if token as usize >= *vocab_size {
                        return Err(anyhow!(
                            "patch token id {} exceeds configured vocab_size {}",
                            token,
                            vocab_size
                        ));
                    }
                    if matches!(eos_id, Some(eos_id) if *eos_id == token) {
                        return Err(anyhow!(
                            "patch token id {} collides with eos_id {}; increase token offset or vocab",
                            token,
                            token
                        ));
                    }
                    tokens.push(token);
                }
                if let Some(eos_id) = eos_id {
                    tokens.push(*eos_id);
                }
                Ok(tokens)
            }
            Self::Gpt2ByteCompatible { .. } | Self::RustBpe { .. } => Err(anyhow!(
                "tokenizer does not support direct patch-token encoding"
            )),
        }
    }

    pub fn manifest(&self) -> UniversalityTokenizerManifest {
        match self {
            Self::Gpt2ByteCompatible { vocab_size, eos_id } => UniversalityTokenizerManifest {
                family: TokenizerFamily::Gpt2ByteCompatible,
                vocab_size: *vocab_size,
                bos_id: None,
                eos_id: *eos_id,
                pad_id: None,
                unk_id: None,
                tokenizer_id: "gpt2_byte_compatible".to_string(),
            },
            Self::PatchTokenIds { vocab_size, eos_id } => UniversalityTokenizerManifest {
                family: TokenizerFamily::PatchTokenIds,
                vocab_size: *vocab_size,
                bos_id: None,
                eos_id: *eos_id,
                pad_id: None,
                unk_id: None,
                tokenizer_id: "patch_token_ids".to_string(),
            },
            Self::RustBpe {
                mergeable_vocab_size,
                bos_id,
                eos_id,
                pad_id,
                unk_id,
                tokenizer_id,
                ..
            } => {
                let special_max = [*bos_id, *eos_id, *pad_id, *unk_id]
                    .into_iter()
                    .flatten()
                    .max()
                    .unwrap_or_default() as usize;
                UniversalityTokenizerManifest {
                    family: TokenizerFamily::RustBpe,
                    vocab_size: (*mergeable_vocab_size).max(special_max.saturating_add(1)),
                    bos_id: *bos_id,
                    eos_id: *eos_id,
                    pad_id: *pad_id,
                    unk_id: *unk_id,
                    tokenizer_id: tokenizer_id.clone(),
                }
            }
        }
    }
}

#[derive(Deserialize)]
struct RustBpeRecord {
    pattern: String,
    merges: Vec<RustBpeMergeRecord>,
}

#[derive(Deserialize)]
struct RustBpeMergeRecord {
    left: u32,
    right: u32,
    token_id: u32,
}

pub fn tokenizer_id_from_path(path: &Path) -> String {
    format!("rust_bpe:{}", path.display())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpt2_byte_compatible_tokenizer_uses_byte_ids() {
        let tokenizer = CorpusTokenizer::from_config(&NcaTokenizationConfig::Gpt2ByteCompatible {
            vocab_size: 50_257,
            eos_id: Some(50_256),
        })
        .expect("tokenizer");
        let ids = tokenizer.encode("A");
        assert_eq!(ids, vec![65, 50_256]);
    }

    #[test]
    fn patch_token_ids_appends_eos() {
        let tokenizer = CorpusTokenizer::from_config(&NcaTokenizationConfig::PatchTokenIds {
            vocab_size: 50_257,
            eos_id: Some(50_256),
        })
        .expect("tokenizer");
        let ids = tokenizer
            .encode_patch_tokens(&[12, 34, 56])
            .expect("patch tokens");
        assert_eq!(ids, vec![12, 34, 56, 50_256]);
    }
}
