mod factory;
mod huggingface;
mod local_text;
pub mod scheduler;
mod shakespeare;
mod universality;

use crate::tokenizer::SharedTokenizer;

pub use factory::build_dataset;
pub use huggingface::HuggingFaceDataset;
pub use local_text::LocalTextDataset;
pub use scheduler::{
    RandomDataLoader, SequenceBatch, StreamingDataLoader, TokenSequenceDataset,
    sample_batch_with_shape,
};
pub use shakespeare::ShakespeareDataset;
pub use universality::UniversalityDataset;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DatasetSplit {
    Train,
    Val,
}

#[derive(Clone)]
pub enum Dataset {
    Shakespeare(ShakespeareDataset),
    LocalText(LocalTextDataset),
    HuggingFace(HuggingFaceDataset),
    Universality(UniversalityDataset),
}

impl Dataset {
    pub fn from_shakespeare(dataset: ShakespeareDataset) -> Self {
        Self::Shakespeare(dataset)
    }

    pub fn from_huggingface(dataset: HuggingFaceDataset) -> Self {
        Self::HuggingFace(dataset)
    }

    pub fn from_local_text(dataset: LocalTextDataset) -> Self {
        Self::LocalText(dataset)
    }

    pub fn from_universality(dataset: UniversalityDataset) -> Self {
        Self::Universality(dataset)
    }

    pub fn tokenizer(&self) -> SharedTokenizer {
        TokenSequenceDataset::tokenizer(self)
    }

    pub fn train_split_ratio(&self) -> f32 {
        TokenSequenceDataset::train_split_ratio(self)
    }

    pub fn batch_size(&self) -> usize {
        TokenSequenceDataset::batch_size(self)
    }

    pub fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        TokenSequenceDataset::steps_per_epoch(self, split)
    }
}

impl TokenSequenceDataset for Dataset {
    fn tokenizer(&self) -> SharedTokenizer {
        match self {
            Dataset::Shakespeare(dataset) => dataset.tokenizer(),
            Dataset::LocalText(dataset) => dataset.tokenizer(),
            Dataset::HuggingFace(dataset) => dataset.tokenizer(),
            Dataset::Universality(dataset) => dataset.tokenizer(),
        }
    }

    fn token_count(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.token_count(),
            Dataset::LocalText(dataset) => dataset.token_count(),
            Dataset::HuggingFace(dataset) => dataset.token_count(),
            Dataset::Universality(dataset) => dataset.token_count(),
        }
    }

    fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
        match self {
            Dataset::Shakespeare(dataset) => dataset.copy_token_range(start, dst),
            Dataset::LocalText(dataset) => dataset.copy_token_range(start, dst),
            Dataset::HuggingFace(dataset) => dataset.copy_token_range(start, dst),
            Dataset::Universality(dataset) => dataset.copy_token_range(start, dst),
        }
    }

    fn train_len(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.train_len(),
            Dataset::LocalText(dataset) => dataset.train_len(),
            Dataset::HuggingFace(dataset) => dataset.train_len(),
            Dataset::Universality(dataset) => dataset.train_len(),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.block_size(),
            Dataset::LocalText(dataset) => dataset.block_size(),
            Dataset::HuggingFace(dataset) => dataset.block_size(),
            Dataset::Universality(dataset) => dataset.block_size(),
        }
    }

    fn batch_size(&self) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => dataset.batch_size(),
            Dataset::LocalText(dataset) => dataset.batch_size(),
            Dataset::HuggingFace(dataset) => dataset.batch_size(),
            Dataset::Universality(dataset) => dataset.batch_size(),
        }
    }

    fn train_split_ratio(&self) -> f32 {
        match self {
            Dataset::Shakespeare(dataset) => dataset.train_split_ratio(),
            Dataset::LocalText(dataset) => dataset.train_split_ratio(),
            Dataset::HuggingFace(dataset) => dataset.train_split_ratio(),
            Dataset::Universality(dataset) => dataset.train_split_ratio(),
        }
    }

    fn preferred_logical_document_tokens(&self, split: DatasetSplit) -> Option<usize> {
        match self {
            Dataset::Shakespeare(dataset) => dataset.preferred_logical_document_tokens(split),
            Dataset::LocalText(dataset) => dataset.preferred_logical_document_tokens(split),
            Dataset::HuggingFace(dataset) => dataset.preferred_logical_document_tokens(split),
            Dataset::Universality(dataset) => dataset.preferred_logical_document_tokens(split),
        }
    }

    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match self {
            Dataset::Shakespeare(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
            Dataset::LocalText(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
            Dataset::HuggingFace(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
            Dataset::Universality(dataset) => {
                TokenSequenceDataset::split_offset_and_span(dataset, split)
            }
        }
    }

    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        match self {
            Dataset::Shakespeare(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
            Dataset::LocalText(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
            Dataset::Universality(dataset) => TokenSequenceDataset::steps_per_epoch(dataset, split),
        }
    }

    fn decode(&self, tokens: &[i64]) -> String {
        match self {
            Dataset::Shakespeare(dataset) => TokenSequenceDataset::decode(dataset, tokens),
            Dataset::LocalText(dataset) => TokenSequenceDataset::decode(dataset, tokens),
            Dataset::HuggingFace(dataset) => TokenSequenceDataset::decode(dataset, tokens),
            Dataset::Universality(dataset) => TokenSequenceDataset::decode(dataset, tokens),
        }
    }
}

pub type ShakespeareSplit = DatasetSplit;
pub type ShakespeareBatch<B> = SequenceBatch<B>;
pub type ShakespeareRandomDataLoader<B> = RandomDataLoader<B>;
