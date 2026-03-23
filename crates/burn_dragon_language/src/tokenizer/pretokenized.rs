use super::Tokenizer;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PretokenizedTokenizer {
    vocab_size: usize,
    bos: Option<u32>,
    eos: Option<u32>,
    pad: Option<u32>,
    unk: Option<u32>,
}

impl PretokenizedTokenizer {
    pub fn new(
        vocab_size: usize,
        bos: Option<u32>,
        eos: Option<u32>,
        pad: Option<u32>,
        unk: Option<u32>,
    ) -> Self {
        Self {
            vocab_size: vocab_size.max(1),
            bos,
            eos,
            pad,
            unk,
        }
    }

    fn parse_token(text: &str) -> Option<u32> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }
        trimmed.parse::<u32>().ok()
    }
}

impl Tokenizer for PretokenizedTokenizer {
    fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> Vec<u32> {
        let mut tokens = Vec::new();
        if add_bos && let Some(bos) = self.bos {
            tokens.push(bos);
        }

        for chunk in text.split(|ch: char| ch.is_whitespace() || matches!(ch, ',' | '[' | ']')) {
            match Self::parse_token(chunk) {
                Some(token) => tokens.push(token),
                None if chunk.trim().is_empty() => {}
                None => {
                    if let Some(unk) = self.unk {
                        tokens.push(unk);
                    }
                }
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
        let mut rendered = Vec::with_capacity(ids.len());
        for &id in ids {
            if Some(id) == self.pad || Some(id) == self.bos {
                continue;
            }
            if Some(id) == self.eos {
                if stop_at_eos {
                    break;
                }
                continue;
            }
            rendered.push(id.to_string());
        }
        rendered.join(" ")
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

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pretokenized_round_trip_preserves_numeric_ids() {
        let tokenizer = PretokenizedTokenizer::new(16, Some(14), Some(15), Some(13), Some(12));
        let ids = tokenizer.encode("[1, 2, 3]", true, true);
        assert_eq!(ids, vec![14, 1, 2, 3, 15]);
        assert_eq!(tokenizer.decode(&ids), "1 2 3");
    }

    #[test]
    fn pretokenized_unknown_falls_back_to_unk_when_available() {
        let tokenizer = PretokenizedTokenizer::new(16, None, None, None, Some(7));
        let ids = tokenizer.encode("10 nope 11", false, false);
        assert_eq!(ids, vec![10, 7, 11]);
    }
}
