use std::any::Any;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use burn_dragon_tokenizer::Tokenizer as RustBpeInner;
use fancy_regex::Regex;
use serde::{Deserialize, Serialize};

use super::Tokenizer;

pub struct RustBpeTokenizer {
    backend: RustBpeBackend,
    mergeable_vocab_size: usize,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
    vocab_size: usize,
}

impl RustBpeTokenizer {
    const GPT2_PATTERN: &str =
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";

    pub fn new_untrained(
        mergeable_vocab_size: usize,
        pattern: Option<&str>,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        let pattern = pattern.unwrap_or(burn_dragon_tokenizer::GPT4_PATTERN);
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
            backend: RustBpeBackend::Native(inner),
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
            vocab_size,
        }
    }

    pub fn train_from_texts<'a, I>(&mut self, texts: I) -> Result<()>
    where
        I: Iterator<Item = &'a str>,
    {
        match &mut self.backend {
            RustBpeBackend::Native(inner) => {
                let pattern = inner.pattern.clone();
                inner
                    .train_from_texts(
                        texts,
                        self.mergeable_vocab_size as u32,
                        Some(pattern.as_str()),
                    )
                    .map_err(|err| anyhow!("failed to train rustbpe tokenizer: {err}"))
            }
            RustBpeBackend::Gpt2ByteLevel(_) => Err(anyhow!(
                "cannot train a HuggingFace byte-level tokenizer through rust_bpe"
            )),
        }
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("failed to create directory {}", parent.display()))?;
        }
        if !matches!(self.backend, RustBpeBackend::Native(_)) {
            return Err(anyhow!(
                "saving HuggingFace byte-level tokenizer snapshots is not supported"
            ));
        }
        let record = RustBpeRecord::from_tokenizer(self);
        let json =
            serde_json::to_string_pretty(&record).context("failed to serialize rustbpe record")?;
        fs::write(path, json).with_context(|| format!("failed to write {}", path.display()))?;
        Ok(())
    }

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
        if let Ok(record) = serde_json::from_str::<RustBpeRecord>(&data) {
            return Self::from_parts(
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
            );
        }

        let record: HuggingFaceTokenizerJsonRecord =
            serde_json::from_str(&data).with_context(|| {
                format!(
                    "failed to parse rustbpe or HuggingFace tokenizer {}",
                    path.display()
                )
            })?;
        Self::from_huggingface_tokenizer_json_record(
            record,
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
        )
    }

    fn from_huggingface_tokenizer_json_record(
        record: HuggingFaceTokenizerJsonRecord,
        mergeable_vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        if record.is_byte_level_bpe() {
            return Self::from_huggingface_byte_level_tokenizer_json_record(
                record,
                mergeable_vocab_size,
                bos,
                eos,
                pad,
                unk,
            );
        }

        let vocab = record.model.vocab;
        let mut merges = HashMap::with_capacity(record.model.merges.len());
        for merge in record.model.merges {
            let (left_token, right_token) = merge.into_pair()?;
            let left = *vocab
                .get(&left_token)
                .ok_or_else(|| anyhow!("missing left merge token {left_token:?} in vocab"))?;
            let right = *vocab
                .get(&right_token)
                .ok_or_else(|| anyhow!("missing right merge token {right_token:?} in vocab"))?;
            let merged_token = format!("{left_token}{right_token}");
            let merged_id = *vocab
                .get(&merged_token)
                .ok_or_else(|| anyhow!("missing merged token {merged_token:?} in vocab"))?;
            merges.insert((left, right), merged_id);
        }

        Self::from_parts(
            mergeable_vocab_size,
            Self::GPT2_PATTERN,
            merges,
            bos,
            eos,
            pad,
            unk,
        )
    }

    fn from_huggingface_byte_level_tokenizer_json_record(
        record: HuggingFaceTokenizerJsonRecord,
        mergeable_vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Result<Self> {
        let backend = Gpt2ByteLevelBpe::from_huggingface_record(&record)?;
        let special_max = [bos, eos, pad, unk].into_iter().flatten().max();
        let vocab_size = special_max
            .map(|id| mergeable_vocab_size.max(id as usize + 1))
            .unwrap_or(mergeable_vocab_size)
            .max(record.model.vocab.len())
            .max(1);
        Ok(Self {
            backend: RustBpeBackend::Gpt2ByteLevel(backend),
            mergeable_vocab_size,
            bos,
            eos,
            pad,
            unk,
            vocab_size,
        })
    }
}

impl Tokenizer for RustBpeTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos && let Some(bos) = self.bos {
            tokens.push(bos);
        }
        match &self.backend {
            RustBpeBackend::Native(inner) => tokens.extend(inner.encode(text)),
            RustBpeBackend::Gpt2ByteLevel(inner) => {
                tokens.extend(inner.encode(text, self.unk));
            }
        }
        if add_eos && let Some(eos) = self.eos {
            tokens.push(eos);
        }
        tokens
    }

    fn decode(&self, ids: &[u32]) -> String {
        self.decode_with_options(ids, true)
    }

    fn decode_with_options(&self, ids: &[u32], stop_at_eos: bool) -> String {
        let mut rendered = String::new();
        let mut segment = Vec::new();
        let flush_segment =
            |segment: &mut Vec<u32>, rendered: &mut String, backend: &RustBpeBackend| {
                if segment.is_empty() {
                    return;
                }
                match backend {
                    RustBpeBackend::Native(inner) => {
                        if let Ok(text) = inner.decode_to_string(segment) {
                            rendered.push_str(&text);
                        }
                    }
                    RustBpeBackend::Gpt2ByteLevel(inner) => {
                        rendered.push_str(&inner.decode(segment));
                    }
                }
                segment.clear();
            };

        for &id in ids {
            if Some(id) == self.pad || Some(id) == self.bos {
                continue;
            }
            if Some(id) == self.eos {
                flush_segment(&mut segment, &mut rendered, &self.backend);
                if stop_at_eos {
                    break;
                }
                continue;
            }
            if Some(id) == self.unk {
                flush_segment(&mut segment, &mut rendered, &self.backend);
                rendered.push('?');
                continue;
            }
            if (id as usize) < self.mergeable_vocab_size {
                segment.push(id);
            }
        }
        flush_segment(&mut segment, &mut rendered, &self.backend);
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

enum RustBpeBackend {
    Native(RustBpeInner),
    Gpt2ByteLevel(Gpt2ByteLevelBpe),
}

struct Gpt2ByteLevelBpe {
    pattern: Regex,
    byte_encoder: [char; 256],
    byte_decoder: HashMap<char, u8>,
    bpe_ranks: HashMap<(String, String), usize>,
    vocab_by_piece: HashMap<String, u32>,
    piece_by_id: Vec<Option<String>>,
}

impl Gpt2ByteLevelBpe {
    fn from_huggingface_record(record: &HuggingFaceTokenizerJsonRecord) -> Result<Self> {
        let (byte_encoder, byte_decoder) = gpt2_byte_mapping();
        let pattern = Regex::new(RustBpeTokenizer::GPT2_PATTERN)
            .map_err(|err| anyhow!("failed to compile GPT-2 regex: {err}"))?;
        let mut piece_by_id = vec![None; record.model.vocab.len()];
        for (piece, &id) in &record.model.vocab {
            let idx = id as usize;
            if idx >= piece_by_id.len() {
                piece_by_id.resize(idx + 1, None);
            }
            piece_by_id[idx] = Some(piece.clone());
        }
        let mut bpe_ranks = HashMap::with_capacity(record.model.merges.len());
        for (rank, merge) in record.model.merges.iter().enumerate() {
            let (left, right) = merge.clone().into_pair()?;
            bpe_ranks.insert((left, right), rank);
        }
        Ok(Self {
            pattern,
            byte_encoder,
            byte_decoder,
            bpe_ranks,
            vocab_by_piece: record.model.vocab.clone(),
            piece_by_id,
        })
    }

    fn encode(&self, text: &str, unk: Option<u32>) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk_match in self.pattern.find_iter(text) {
            let chunk = match chunk_match {
                Ok(mat) => mat.as_str(),
                Err(_) => continue,
            };
            let mapped = self.map_bytes_to_unicode(chunk.as_bytes());
            for piece in self.bpe(&mapped) {
                if let Some(&id) = self.vocab_by_piece.get(&piece) {
                    ids.push(id);
                } else if let Some(unk_id) = unk {
                    ids.push(unk_id);
                }
            }
        }
        ids
    }

    fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(Some(piece)) = self.piece_by_id.get(id as usize) else {
                continue;
            };
            for ch in piece.chars() {
                if let Some(&byte) = self.byte_decoder.get(&ch) {
                    bytes.push(byte);
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    fn bpe(&self, mapped: &str) -> Vec<String> {
        let mut word: Vec<String> = mapped.chars().map(|ch| ch.to_string()).collect();
        if word.len() < 2 {
            return word;
        }

        loop {
            let mut best: Option<(usize, usize, String)> = None;
            for idx in 0..word.len() - 1 {
                let pair = (word[idx].clone(), word[idx + 1].clone());
                if let Some(&rank) = self.bpe_ranks.get(&pair) {
                    let merged = format!("{}{}", pair.0, pair.1);
                    match &best {
                        Some((_, best_rank, _)) if rank >= *best_rank => {}
                        _ => best = Some((idx, rank, merged)),
                    }
                }
            }
            let Some((idx, _, merged)) = best else {
                break;
            };
            word[idx] = merged;
            word.remove(idx + 1);
            if word.len() < 2 {
                break;
            }
        }

        word
    }

    fn map_bytes_to_unicode(&self, bytes: &[u8]) -> String {
        let mut mapped = String::with_capacity(bytes.len());
        for &byte in bytes {
            mapped.push(self.byte_encoder[byte as usize]);
        }
        mapped
    }
}

fn gpt2_byte_mapping() -> ([char; 256], HashMap<char, u8>) {
    let mut bs = Vec::new();
    bs.extend(33u16..=126u16);
    bs.extend(161u16..=172u16);
    bs.extend(174u16..=255u16);

    let mut cs = bs.clone();
    let mut next = 0u16;
    for byte in 0u16..=255u16 {
        if !bs.contains(&byte) {
            bs.push(byte);
            cs.push(256 + next);
            next += 1;
        }
    }

    let mut encoder = ['\0'; 256];
    let mut decoder = HashMap::with_capacity(256);
    for (byte, codepoint) in bs.into_iter().zip(cs.into_iter()) {
        let ch = char::from_u32(codepoint as u32).expect("valid GPT-2 byte mapping codepoint");
        encoder[byte as usize] = ch;
        decoder.insert(ch, byte as u8);
    }
    (encoder, decoder)
}

#[derive(Serialize, Deserialize)]
struct RustBpeRecord {
    pattern: String,
    merges: Vec<RustBpeMergeRecord>,
}

impl RustBpeRecord {
    fn from_tokenizer(tokenizer: &RustBpeTokenizer) -> Self {
        let RustBpeBackend::Native(inner) = &tokenizer.backend else {
            panic!("cannot serialize a non-native rust_bpe tokenizer");
        };
        let mut merges = inner
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
            pattern: inner.pattern.clone(),
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

#[derive(Deserialize)]
struct HuggingFaceTokenizerJsonRecord {
    #[serde(default)]
    pre_tokenizer: Option<HuggingFaceTokenizerComponentRecord>,
    #[serde(default)]
    decoder: Option<HuggingFaceTokenizerComponentRecord>,
    model: HuggingFaceBpeModelRecord,
}

impl HuggingFaceTokenizerJsonRecord {
    fn is_byte_level_bpe(&self) -> bool {
        self.pre_tokenizer
            .as_ref()
            .is_some_and(HuggingFaceTokenizerComponentRecord::is_byte_level)
            || self
                .decoder
                .as_ref()
                .is_some_and(HuggingFaceTokenizerComponentRecord::is_byte_level)
    }
}

#[derive(Deserialize)]
struct HuggingFaceTokenizerComponentRecord {
    #[serde(rename = "type")]
    component_type: String,
}

impl HuggingFaceTokenizerComponentRecord {
    fn is_byte_level(&self) -> bool {
        self.component_type == "ByteLevel"
    }
}

#[derive(Deserialize)]
struct HuggingFaceBpeModelRecord {
    vocab: HashMap<String, u32>,
    merges: Vec<HuggingFaceBpeMergeRecord>,
}

#[derive(Clone, Deserialize)]
#[serde(untagged)]
enum HuggingFaceBpeMergeRecord {
    Pair(Vec<String>),
    String(String),
}

impl HuggingFaceBpeMergeRecord {
    fn into_pair(self) -> Result<(String, String)> {
        match self {
            Self::Pair(parts) => {
                if parts.len() == 2 {
                    Ok((parts[0].clone(), parts[1].clone()))
                } else {
                    Err(anyhow!(
                        "expected merge pair with 2 entries, found {}",
                        parts.len()
                    ))
                }
            }
            Self::String(value) => {
                let mut parts = value.splitn(2, ' ');
                let left = parts
                    .next()
                    .ok_or_else(|| anyhow!("missing left merge token in {value:?}"))?;
                let right = parts
                    .next()
                    .ok_or_else(|| anyhow!("missing right merge token in {value:?}"))?;
                Ok((left.to_string(), right.to_string()))
            }
        }
    }
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

    #[test]
    fn rust_bpe_load_accepts_huggingface_bpe_tokenizer_json() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("tokenizer.json");
        std::fs::write(
            &path,
            r#"{
              "pre_tokenizer": { "type": "ByteLevel" },
              "decoder": { "type": "ByteLevel" },
              "model": {
                "vocab": {
                  "A": 65,
                  "B": 66,
                  "Ġ": 220,
                  "AB": 256
                },
                "merges": ["A B"]
              }
            }"#,
        )
        .expect("write tokenizer");

        let loaded = RustBpeTokenizer::load(&path, 257, None, None, None, None)
            .expect("load hf tokenizer json");
        let ids = loaded.encode("AB", false, false);
        assert_eq!(ids, vec![256]);
        assert_eq!(loaded.decode(&ids), "AB");
        assert_eq!(loaded.encode(" A", false, false), vec![220, 65]);
        assert_eq!(loaded.decode(&[220, 65]), " A");
    }
}
