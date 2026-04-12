use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use burn_dragon_tokenizer::Tokenizer as RustBpeInner;
use serde::Deserialize;

use crate::config::NcaTokenizationConfig;
use crate::manifest::{TokenizerFamily, UniversalityTokenizerManifest};
use crate::nca::NcaSample;

pub enum CorpusTokenizer {
    Gpt2ByteCompatible {
        vocab_size: usize,
        eos_id: Option<u32>,
    },
    PatchTokenIds {
        vocab_size: usize,
        eos_id: Option<u32>,
        frame_special_tokens: bool,
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
            NcaTokenizationConfig::PatchTokenIds {
                vocab_size,
                eos_id,
                frame_special_tokens,
            } => Ok(Self::PatchTokenIds {
                vocab_size: *vocab_size,
                eos_id: *eos_id,
                frame_special_tokens: *frame_special_tokens,
            }),
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
            Self::PatchTokenIds {
                vocab_size,
                eos_id,
                frame_special_tokens: _,
            } => {
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

    pub fn encode_patch_sample(
        &self,
        sample: &NcaSample,
        serialization: &crate::config::NcaSerializationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            Self::PatchTokenIds {
                vocab_size,
                eos_id,
                frame_special_tokens,
            } => {
                let patch_size = serialization.patch_size.max(1);
                let patches_per_row = sample.width / patch_size;
                let patches_per_col = sample.height / patch_size;
                let patch_cells = patch_size
                    .checked_mul(patch_size)
                    .ok_or_else(|| anyhow!("patch cell count overflow"))?;
                let patch_vocab_size = (sample.state_count as u32)
                    .checked_pow(patch_cells as u32)
                    .ok_or_else(|| anyhow!("patch token vocabulary overflow"))?;
                let frame_token_overhead = usize::from(*frame_special_tokens) * 2;
                let mut tokens = Vec::with_capacity(
                    sample.frames.len()
                        * (patches_per_row * patches_per_col + frame_token_overhead)
                        + usize::from(eos_id.is_some()),
                );
                let frame_start_id = (*frame_special_tokens).then_some(patch_vocab_size);
                let frame_end_id = frame_start_id.and_then(|id| id.checked_add(1));

                for frame in &sample.frames {
                    if let Some(start_id) = frame_start_id {
                        if start_id as usize >= *vocab_size {
                            return Err(anyhow!(
                                "frame start token id {} exceeds configured vocab_size {}",
                                start_id,
                                vocab_size
                            ));
                        }
                        tokens.push(start_id);
                    }
                    for patch_y in 0..patches_per_col {
                        for patch_x in 0..patches_per_row {
                            let mut value = 0u32;
                            for dy in 0..patch_size {
                                for dx in 0..patch_size {
                                    let x = patch_x * patch_size + dx;
                                    let y = patch_y * patch_size + dy;
                                    let cell = frame[y * sample.width + x] as u32;
                                    value = value
                                        .checked_mul(sample.state_count as u32)
                                        .and_then(|acc| acc.checked_add(cell))
                                        .ok_or_else(|| anyhow!("patch token id overflow"))?;
                                }
                            }
                            if value as usize >= *vocab_size {
                                return Err(anyhow!(
                                    "patch token id {} exceeds configured vocab_size {}",
                                    value,
                                    vocab_size
                                ));
                            }
                            tokens.push(value);
                        }
                    }
                    if let Some(end_id) = frame_end_id {
                        if end_id as usize >= *vocab_size {
                            return Err(anyhow!(
                                "frame end token id {} exceeds configured vocab_size {}",
                                end_id,
                                vocab_size
                            ));
                        }
                        tokens.push(end_id);
                    }
                }

                if let Some(eos_id) = eos_id {
                    if Some(*eos_id) == frame_start_id || Some(*eos_id) == frame_end_id {
                        return Err(anyhow!(
                            "patch document eos_id {} collides with frame boundary token ids",
                            eos_id
                        ));
                    }
                    tokens.push(*eos_id);
                }
                Ok(tokens)
            }
            Self::Gpt2ByteCompatible { .. } | Self::RustBpe { .. } => Err(anyhow!(
                "tokenizer does not support direct NCA patch-sample encoding"
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
                frame_special_tokens: false,
                pad_id: None,
                unk_id: None,
                tokenizer_id: "gpt2_byte_compatible".to_string(),
            },
            Self::PatchTokenIds {
                vocab_size,
                eos_id,
                frame_special_tokens,
            } => UniversalityTokenizerManifest {
                family: TokenizerFamily::PatchTokenIds,
                vocab_size: *vocab_size,
                bos_id: None,
                eos_id: *eos_id,
                frame_special_tokens: *frame_special_tokens,
                pad_id: None,
                unk_id: None,
                tokenizer_id: if *frame_special_tokens {
                    "patch_token_ids:framed".to_string()
                } else {
                    "patch_token_ids".to_string()
                },
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
                    frame_special_tokens: false,
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
            frame_special_tokens: true,
        })
        .expect("tokenizer");
        let ids = tokenizer
            .encode_patch_tokens(&[12, 34, 56])
            .expect("patch tokens");
        assert_eq!(ids, vec![12, 34, 56, 50_256]);
    }

    #[test]
    fn patch_sample_encoding_wraps_each_frame_with_boundary_tokens() {
        let tokenizer = CorpusTokenizer::from_config(&NcaTokenizationConfig::PatchTokenIds {
            vocab_size: 50_257,
            eos_id: Some(50_256),
            frame_special_tokens: true,
        })
        .expect("tokenizer");
        let sample = NcaSample {
            family_kind: crate::config::NcaFamilyKind::NeuralStochastic,
            complexity_band: crate::config::NcaComplexityBand::Simple,
            width: 2,
            height: 2,
            state_count: 10,
            frames: vec![vec![1, 2, 3, 4], vec![4, 3, 2, 1]],
            rule_seed: Some(1),
            complexity_filter_matched: true,
            identity_bias: 0.0,
            temperature: 0.0,
            step_stride: 1,
            start_step: 0,
            gzip_complexity_ratio: 0.0,
        };
        let serialization = crate::config::NcaSerializationConfig {
            patch_size: 2,
            include_observable_header: true,
            preview_samples: 1,
        };
        let tokens = tokenizer
            .encode_patch_sample(&sample, &serialization)
            .expect("patch sample tokens");
        assert_eq!(
            tokens,
            vec![10_000, 1_234, 10_001, 10_000, 4_321, 10_001, 50_256]
        );
    }
}
