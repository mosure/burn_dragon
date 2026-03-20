use std::any::Any;
use std::collections::HashMap;
#[cfg(feature = "train")]
use std::fs;
#[cfg(feature = "train")]
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use rustbpe::Tokenizer as RustBpeInner;
use serde::{Deserialize, Serialize};

use super::Tokenizer;

pub struct RustBpeTokenizer {
    inner: RustBpeInner,
    mergeable_vocab_size: usize,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    vocab_size: usize,
}

impl RustBpeTokenizer {
    pub fn new_untrained(
        mergeable_vocab_size: usize,
        pattern: Option<&str>,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        let pattern = pattern.unwrap_or(rustbpe::GPT4_PATTERN);
        let inner = RustBpeInner::new_with_pattern(pattern)
            .map_err(|err| anyhow!("failed to compile rustbpe pattern: {err}"))?;
        Ok(Self::from_inner(
            inner,
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
        ))
    }

    pub fn from_parts(
        mergeable_vocab_size: usize,
        pattern: impl Into<String>,
        merges: HashMap<(u32, u32), u32>,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        let inner = RustBpeInner::from_merges(pattern.into(), merges)
            .map_err(|err| anyhow!("failed to build rustbpe tokenizer from merges: {err}"))?;
        Ok(Self::from_inner(
            inner,
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
        ))
    }

    fn from_inner(
        inner: RustBpeInner,
        mergeable_vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Self {
        let special_max = [bos, eos, pad, unk].into_iter().flatten().max();
        let vocab_size = special_max
            .map(|id| mergeable_vocab_size.max(id as usize + 1))
            .unwrap_or(mergeable_vocab_size)
            .max(1);
        Self {
            inner,
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
            vocab_size,
        }
    }

    #[cfg(feature = "train")]
    pub fn train_from_texts<'a, I>(&mut self, texts: I) -> Result<()>
    where
        I: Iterator<Item = &'a str>,
    {
        let pattern = self.inner.pattern.clone();
        self.inner
            .train_from_texts(
                texts,
                self.mergeable_vocab_size as u32,
                Some(pattern.as_str()),
            )
            .map_err(|err| anyhow!("failed to train rustbpe tokenizer: {err}"))
    }

    #[cfg(feature = "train")]
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        let record = RustBpeRecord::from_tokenizer(self);
        let json =
            serde_json::to_string_pretty(&record).context("failed to serialize rustbpe record")?;
        fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

    #[cfg(feature = "train")]
    pub fn load(
        path: impl AsRef<Path>,
        mergeable_vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        let path = path.as_ref();
        let data = fs::read_to_string(path)
            .with_context(|| format!("failed to read rustbpe vocabulary {}", path.display()))?;
        let record: RustBpeRecord = serde_json::from_str(&data)
            .with_context(|| format!("failed to parse rustbpe vocabulary {}", path.display()))?;
        Self::from_parts(
            mergeable_vocab_size,
            record.pattern,
            record
                .merges
                .into_iter()
                .map(|merge| ((merge.left, merge.right), merge.token_id))
                .collect(),
            bos,
            eos,
            pad,
            unk,
        )
    }
}

impl Tokenizer for RustBpeTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos && let Some(bos) = self.bos {
            tokens.push(bos);
        }
        tokens.extend(self.inner.encode(text));
        if add_eos && let Some(eos) = self.eos {
            tokens.push(eos);
        }
        tokens
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut rendered = String::new();
        let mut segment = Vec::new();
        let flush_segment =
            |segment: &mut Vec<u32>, rendered: &mut String, inner: &RustBpeInner| {
                if segment.is_empty() {
                    return;
                }
                if let Ok(text) = inner.decode_to_string(segment) {
                    rendered.push_str(&text);
                }
                segment.clear();
            };

        for &id in ids {
            if Some(id) == self.pad || Some(id) == self.bos {
                continue;
            }
            if Some(id) == self.eos {
                break;
            }
            if Some(id) == self.unk {
                flush_segment(&mut segment, &mut rendered, &self.inner);
                rendered.push('?');
                continue;
            }
            if (id as usize) < self.mergeable_vocab_size {
                segment.push(id);
            }
        }
        flush_segment(&mut segment, &mut rendered, &self.inner);
        rendered
    }

    fn len(&self) -> usize {
        self.vocab_size
    }

    fn is_empty(&self) -> bool {
        self.vocab_size == 0
    }

    fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    fn pad_id(&self) -> Option<u32> {
        self.pad
    }

    fn unk_id(&self) -> Option<u32> {
        self.unk
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Serialize, Deserialize)]
struct RustBpeRecord {
    pattern: String,
    merges: Vec<RustBpeMergeRecord>,
}

impl RustBpeRecord {
    fn from_tokenizer(tokenizer: &RustBpeTokenizer) -> Self {
        let mut merges = tokenizer
            .inner
            .merges
            .iter()
            .map(|(&(left, right), &token_id)| RustBpeMergeRecord {
                left,
                right,
                token_id,
            })
            .collect::<Vec<_>>();
        merges.sort_by_key(|merge| merge.token_id);
        Self {
            pattern: tokenizer.inner.pattern.clone(),
            merges,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct RustBpeMergeRecord {
    left: u32,
    right: u32,
    token_id: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_bpe_encode_decode_round_trip() {
        let mut tokenizer = RustBpeTokenizer::new_untrained(260, None, None, Some(260), None, None)
            .expect("create tokenizer");
        tokenizer
            .train_from_texts(["hello world", "hello rust"].into_iter())
            .expect("train tokenizer");
        let ids = tokenizer.encode("hello world", false, true);
        assert_eq!(ids.last().copied(), tokenizer.eos_id());
        assert_eq!(tokenizer.decode(&ids), "hello world");
    }

    #[test]
    fn rust_bpe_save_load_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rustbpe.json");
        let mut tokenizer = RustBpeTokenizer::new_untrained(260, None, None, None, None, Some(261))
            .expect("create tokenizer");
        tokenizer
            .train_from_texts(["abc abc", "abc def"].into_iter())
            .expect("train tokenizer");
        tokenizer.save(&path).expect("save tokenizer");

        let loaded = RustBpeTokenizer::load(&path, 260, None, None, None, Some(261)).expect("load");
        let ids = loaded.encode("abc def", false, false);
        assert_eq!(loaded.decode(&ids), "abc def");
    }
}
