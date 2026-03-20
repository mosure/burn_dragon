use std::mem::size_of;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;
#[cfg(feature = "train")]
use rayon::prelude::*;

use crate::summary_events::summary_event_mask_tensor;
use crate::tokenizer::SharedTokenizer;

use super::DatasetSplit;

/// Abstraction over text corpora that can be converted into BDH-compatible batches.
pub trait TokenSequenceDataset: Send + Sync {
    /// Return a shared tokenizer handle (cloned per call).
    fn tokenizer(&self) -> SharedTokenizer;

    /// Return the full number of token ids representing the corpus.
    fn token_count(&self) -> usize;

    /// Copy a contiguous token range into `dst`.
    fn copy_token_range(&self, start: usize, dst: &mut [u32]);

    /// Number of tokens reserved for the training split from the start of the corpus.
    fn train_len(&self) -> usize;

    /// Maximum sequence length per sample.
    fn block_size(&self) -> usize;

    /// Number of sequences per batch.
    fn batch_size(&self) -> usize;

    /// Ratio used when determining train/validation split boundaries.
    fn train_split_ratio(&self) -> f32;

    /// Preferred logical document length, excluding the next-token target, when the dataset has
    /// hard semantic document boundaries that should inform TBPTT streaming and random window
    /// sampling.
    fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
        None
    }

    /// Provide the offset and span of the requested split.
    fn split_offset_and_span(&self, split: DatasetSplit) -> (usize, usize) {
        match split {
            DatasetSplit::Train => (0, self.train_len()),
            DatasetSplit::Val => {
                let tokens = self.token_count();
                let train_len = self.train_len();
                let remaining = tokens.saturating_sub(train_len);
                if remaining <= self.block_size() + 1 {
                    (0, train_len)
                } else {
                    (train_len, remaining)
                }
            }
        }
    }

    /// Number of steps per epoch for a given split (defaults derived from token counts).
    fn steps_per_epoch(&self, split: DatasetSplit) -> usize {
        let (_offset, span) = self.split_offset_and_span(split);
        let tokens_per_step = self.block_size() * self.batch_size();
        if tokens_per_step == 0 {
            return 1;
        }
        let steps = span.div_ceil(tokens_per_step);
        steps.max(1)
    }

    /// Decode token ids back into text.
    fn decode(&self, tokens: &[i64]) -> String {
        let ids: Vec<u32> = tokens
            .iter()
            .filter_map(|&tok| (tok >= 0).then_some(tok as u32))
            .collect();
        self.tokenizer().decode(&ids)
    }
}

/// Sample a random batch from any dataset implementing [`TokenSequenceDataset`].
pub fn sample_batch<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    device: &B::Device,
) -> SequenceBatch<B> {
    sample_batch_with_shape::<B, T>(
        dataset,
        split,
        dataset.batch_size(),
        dataset.block_size(),
        None,
        device,
    )
}

/// Sample a random batch with an explicit batch/block shape from any dataset implementing
/// [`TokenSequenceDataset`].
pub fn sample_batch_with_shape<B: Backend, T: TokenSequenceDataset + ?Sized>(
    dataset: &T,
    split: DatasetSplit,
    batch_size: usize,
    block_size: usize,
    summary_event_token_ids: Option<&[u32]>,
    device: &B::Device,
) -> SequenceBatch<B> {
    let prof_enabled = crate::train::profile::enabled();
    let cpu_start = prof_enabled.then(Instant::now);
    let (offset, span) = dataset.split_offset_and_span(split);

    #[cfg(not(feature = "train"))]
    let mut rng = thread_rng();
    let mut inputs = vec![0i64; batch_size * block_size];
    let mut targets = vec![0i64; batch_size * block_size];
    #[cfg(not(feature = "train"))]
    let mut sample = vec![0u32; block_size + 1];

    if let Some(logical_document_tokens) = dataset.preferred_logical_document_tokens(split) {
        let document_span = logical_document_tokens.saturating_add(1);
        let num_documents = (span / document_span).max(1);
        let max_start_in_document = logical_document_tokens
            .saturating_sub(block_size)
            .min(document_span.saturating_sub(block_size + 1));
        #[cfg(feature = "train")]
        {
            inputs
                .par_chunks_mut(block_size)
                .zip(targets.par_chunks_mut(block_size))
                .for_each(|(input_row, target_row)| {
                    let mut rng = thread_rng();
                    let doc_index = if num_documents <= 1 {
                        0
                    } else {
                        rng.gen_range(0..num_documents)
                    };
                    let start_in_document = if max_start_in_document == 0 {
                        0
                    } else {
                        rng.gen_range(0..=max_start_in_document)
                    };
                    let start =
                        offset + doc_index.saturating_mul(document_span) + start_in_document;
                    let mut sample = vec![0u32; block_size + 1];
                    dataset.copy_token_range(start, &mut sample);
                    for t in 0..block_size {
                        input_row[t] = sample[t] as i64;
                        target_row[t] = sample[t + 1] as i64;
                    }
                });
        }
        #[cfg(not(feature = "train"))]
        for batch_idx in 0..batch_size {
            let doc_index = if num_documents <= 1 {
                0
            } else {
                rng.gen_range(0..num_documents)
            };
            let start_in_document = if max_start_in_document == 0 {
                0
            } else {
                rng.gen_range(0..=max_start_in_document)
            };
            let start = offset + doc_index.saturating_mul(document_span) + start_in_document;
            dataset.copy_token_range(start, &mut sample);
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = sample[t] as i64;
                targets[batch_idx * block_size + t] = sample[t + 1] as i64;
            }
        }
    } else {
        #[cfg(feature = "train")]
        {
            inputs
                .par_chunks_mut(block_size)
                .zip(targets.par_chunks_mut(block_size))
                .for_each(|(input_row, target_row)| {
                    let mut rng = thread_rng();
                    let max_start = span.saturating_sub(block_size + 1);
                    let start_offset = if max_start == 0 {
                        0
                    } else {
                        rng.gen_range(0..=max_start)
                    };
                    let start = offset + start_offset;
                    let mut sample = vec![0u32; block_size + 1];
                    dataset.copy_token_range(start, &mut sample);
                    for t in 0..block_size {
                        input_row[t] = sample[t] as i64;
                        target_row[t] = sample[t + 1] as i64;
                    }
                });
        }
        #[cfg(not(feature = "train"))]
        for batch_idx in 0..batch_size {
            let max_start = span.saturating_sub(block_size + 1);
            let start_offset = if max_start == 0 {
                0
            } else {
                rng.gen_range(0..=max_start)
            };
            let start = offset + start_offset;
            dataset.copy_token_range(start, &mut sample);
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = sample[t] as i64;
                targets[batch_idx * block_size + t] = sample[t + 1] as i64;
            }
        }
    }

    let cpu_ns = cpu_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    let tensor_copy_start = prof_enabled.then(Instant::now);
    let summary_event_mask = summary_event_mask_tensor::<B>(
        &inputs,
        batch_size,
        block_size,
        summary_event_token_ids,
        device,
    );
    let inputs_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(inputs, [batch_size, block_size]), device);
    let targets_tensor =
        Tensor::<B, 2, Int>::from_data(TensorData::new(targets, [batch_size, block_size]), device);
    let tensor_copy_ns = tensor_copy_start
        .map(|start| start.elapsed().as_nanos())
        .unwrap_or_default();

    if prof_enabled {
        let values = batch_size.saturating_mul(block_size);
        let copy_bytes = (values.saturating_mul(2).saturating_mul(size_of::<i64>())) as u128;
        crate::train::profile::record_dataloader(cpu_ns, tensor_copy_ns, copy_bytes, 0);
    }

    SequenceBatch::new(inputs_tensor, targets_tensor, summary_event_mask)
}

/// Batched token inputs and targets for language modeling.
#[derive(Clone)]
pub struct SequenceBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
    pub summary_event_mask: Option<Tensor<B, 2, Int>>,
    pub reset_stream_state: bool,
}

impl<B: Backend> SequenceBatch<B> {
    pub fn new(
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
        summary_event_mask: Option<Tensor<B, 2, Int>>,
    ) -> Self {
        Self {
            inputs,
            targets,
            summary_event_mask,
            reset_stream_state: false,
        }
    }

    pub fn with_reset_stream_state(mut self, reset_stream_state: bool) -> Self {
        self.reset_stream_state = reset_stream_state;
        self
    }
}

/// Data loader that produces random sequences from any `TokenSequenceDataset`.
pub struct RandomDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
}

pub struct StreamingDataLoader<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    logical_document_tokens: usize,
    seed: u64,
}

impl<B: Backend> Clone for RandomDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
        }
    }
}

impl<B: Backend> Clone for StreamingDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
        }
    }
}

impl<B: Backend> RandomDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
            summary_event_token_ids: None,
        }
    }

    pub fn with_summary_event_token_ids(
        mut self,
        summary_event_token_ids: Option<Vec<u32>>,
    ) -> Self {
        self.summary_event_token_ids = summary_event_token_ids;
        self
    }
}

fn resolve_stream_logical_document_tokens(
    dataset: &dyn TokenSequenceDataset,
    split: DatasetSplit,
    requested_min_logical_block_size: Option<usize>,
) -> usize {
    let block_size = dataset.block_size().max(1);
    if let Some(document_tokens) = dataset.preferred_logical_document_tokens(split) {
        return document_tokens.max(block_size);
    }
    let (_, span) = dataset.split_offset_and_span(split);
    let max_inputs = span.saturating_sub(1);
    let desired = requested_min_logical_block_size
        .unwrap_or(block_size)
        .max(block_size);
    let rounded_up = desired.div_ceil(block_size).saturating_mul(block_size);
    let max_multiple = (max_inputs / block_size).max(1).saturating_mul(block_size);
    rounded_up.min(max_multiple).max(block_size)
}

fn gcd_usize(mut lhs: usize, mut rhs: usize) -> usize {
    while rhs != 0 {
        let remainder = lhs % rhs;
        lhs = rhs;
        rhs = remainder;
    }
    lhs
}

fn resolve_stream_document_permutation(
    seed: u64,
    epoch_index: usize,
    num_documents: usize,
) -> (usize, usize) {
    if num_documents <= 1 {
        return (0, 1);
    }
    let mixed_seed = seed
        ^ (epoch_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (num_documents as u64).rotate_left(17);
    let mut rng = StdRng::seed_from_u64(mixed_seed);
    let document_start = rng.gen_range(0..num_documents);
    let document_stride = loop {
        let candidate = rng.gen_range(1..num_documents);
        if gcd_usize(candidate, num_documents) == 1 {
            break candidate;
        }
    };
    (document_start, document_stride)
}

impl<B: Backend> StreamingDataLoader<B> {
    pub fn new<T>(
        dataset: Arc<T>,
        split: DatasetSplit,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
        min_logical_block_size: Option<usize>,
        seed: u64,
    ) -> Self
    where
        T: TokenSequenceDataset + 'static,
    {
        let dataset: Arc<dyn TokenSequenceDataset> = dataset;
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));
        let logical_document_tokens =
            resolve_stream_logical_document_tokens(dataset.as_ref(), split, min_logical_block_size);

        Self {
            dataset,
            split,
            device: device.clone(),
            steps_per_epoch,
            total_steps,
            consumed_steps,
            summary_event_token_ids: None,
            logical_document_tokens,
            seed,
        }
    }

    pub fn with_summary_event_token_ids(
        mut self,
        summary_event_token_ids: Option<Vec<u32>>,
    ) -> Self {
        self.summary_event_token_ids = summary_event_token_ids;
        self
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for RandomDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        Box::new(RandomIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);
        let total_steps = self.total_steps.map(|limit| limit.min(steps));
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: steps,
            total_steps,
            consumed_steps,
            summary_event_token_ids: self.summary_event_token_ids.clone(),
        })
    }
}

struct StreamingIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
    logical_document_tokens: usize,
    chunks_per_document: usize,
    next_document_group: usize,
    chunk_index_in_document: usize,
    num_documents: usize,
    document_start: usize,
    document_stride: usize,
}

impl<B: Backend> Iterator for StreamingIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        let prof_enabled = crate::train::profile::enabled();
        let cpu_start = prof_enabled.then(Instant::now);
        let (offset, _span) = self.dataset.split_offset_and_span(self.split);
        let batch_size = self.dataset.batch_size();
        let block_size = self.dataset.block_size();
        let mut inputs = vec![0i64; batch_size * block_size];
        let mut targets = vec![0i64; batch_size * block_size];
        #[cfg(not(feature = "train"))]
        let mut sample = vec![0u32; block_size + 1];
        let document_span = self.logical_document_tokens + 1;
        let reset_stream_state = self.chunk_index_in_document == 0;

        #[cfg(feature = "train")]
        {
            let next_document_group = self.next_document_group;
            let document_start = self.document_start;
            let document_stride = self.document_stride;
            let num_documents = self.num_documents.max(1);
            let chunk_index_in_document = self.chunk_index_in_document;
            inputs
                .par_chunks_mut(block_size)
                .zip(targets.par_chunks_mut(block_size))
                .enumerate()
                .for_each(|(batch_idx, (input_row, target_row))| {
                    let doc_rank = (next_document_group + batch_idx) % num_documents;
                    let doc_idx = (document_start
                        .wrapping_add(doc_rank.wrapping_mul(document_stride)))
                        % num_documents;
                    let doc_start = offset + doc_idx.saturating_mul(document_span);
                    let start = doc_start + chunk_index_in_document.saturating_mul(block_size);
                    let mut sample = vec![0u32; block_size + 1];
                    self.dataset.copy_token_range(start, &mut sample);
                    for t in 0..block_size {
                        input_row[t] = sample[t] as i64;
                        target_row[t] = sample[t + 1] as i64;
                    }
                });
        }
        #[cfg(not(feature = "train"))]
        for batch_idx in 0..batch_size {
            let doc_rank = (self.next_document_group + batch_idx) % self.num_documents.max(1);
            let doc_idx = (self
                .document_start
                .wrapping_add(doc_rank.wrapping_mul(self.document_stride)))
                % self.num_documents.max(1);
            let doc_start = offset + doc_idx.saturating_mul(document_span);
            let start = doc_start + self.chunk_index_in_document.saturating_mul(block_size);
            self.dataset.copy_token_range(start, &mut sample);
            for t in 0..block_size {
                inputs[batch_idx * block_size + t] = sample[t] as i64;
                targets[batch_idx * block_size + t] = sample[t + 1] as i64;
            }
        }

        let cpu_ns = cpu_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        let tensor_copy_start = prof_enabled.then(Instant::now);
        let summary_event_mask = summary_event_mask_tensor::<B>(
            &inputs,
            batch_size,
            block_size,
            self.summary_event_token_ids.as_deref(),
            &self.device,
        );
        let inputs_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(inputs, [batch_size, block_size]),
            &self.device,
        );
        let targets_tensor = Tensor::<B, 2, Int>::from_data(
            TensorData::new(targets, [batch_size, block_size]),
            &self.device,
        );
        let tensor_copy_ns = tensor_copy_start
            .map(|start| start.elapsed().as_nanos())
            .unwrap_or_default();

        if prof_enabled {
            let values = batch_size.saturating_mul(block_size);
            let copy_bytes = (values.saturating_mul(2).saturating_mul(size_of::<i64>())) as u128;
            crate::train::profile::record_dataloader(cpu_ns, tensor_copy_ns, copy_bytes, 0);
        }

        self.chunk_index_in_document += 1;
        if self.chunk_index_in_document >= self.chunks_per_document {
            self.chunk_index_in_document = 0;
            self.next_document_group =
                (self.next_document_group + batch_size) % self.num_documents.max(1);
        }

        Some(
            SequenceBatch::new(inputs_tensor, targets_tensor, summary_event_mask)
                .with_reset_stream_state(reset_stream_state),
        )
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for StreamingIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.step, self.steps_total)
    }
}

impl<B> DataLoader<B, SequenceBatch<B>> for StreamingDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<SequenceBatch<B>> + 'a> {
        let steps_total =
            if let (Some(limit), Some(consumed)) = (self.total_steps, &self.consumed_steps) {
                let used = consumed.load(Ordering::Relaxed);
                if used >= limit {
                    0
                } else {
                    (limit - used).min(self.steps_per_epoch)
                }
            } else {
                self.steps_per_epoch
            };

        let (offset, span) = self.dataset.split_offset_and_span(self.split);
        let _ = offset;
        let block_size = self.dataset.block_size().max(1);
        let logical_document_tokens = self.logical_document_tokens.max(block_size);
        let chunks_per_document = logical_document_tokens.div_ceil(block_size).max(1);
        let document_span = logical_document_tokens + 1;
        let num_documents = (span / document_span).max(1);
        let consumed = self
            .consumed_steps
            .as_ref()
            .map(|counter| counter.load(Ordering::Relaxed))
            .unwrap_or_default();
        let epoch_index = consumed / self.steps_per_epoch.max(1);
        let (document_start, document_stride) =
            resolve_stream_document_permutation(self.seed, epoch_index, num_documents);

        Box::new(StreamingIterator {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            logical_document_tokens,
            chunks_per_document,
            next_document_group: 0,
            chunk_index_in_document: 0,
            num_documents,
            document_start,
            document_stride,
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: device.clone(),
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, SequenceBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);
        let total_steps = self.total_steps.map(|limit| limit.min(steps));
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            split: self.split,
            device: self.device.clone(),
            steps_per_epoch: steps,
            total_steps,
            consumed_steps,
            summary_event_token_ids: self.summary_event_token_ids.clone(),
            logical_document_tokens: self.logical_document_tokens,
            seed: self.seed,
        })
    }
}

#[cfg(test)]
mod streaming_tests {
    use super::*;
    use burn_ndarray::NdArray;

    type TestBackend = NdArray<f32>;

    #[derive(Clone)]
    struct TinyDataset {
        tokens: Arc<Vec<u32>>,
        train_len: usize,
        block_size: usize,
        batch_size: usize,
        tokenizer: SharedTokenizer,
        preferred_logical_document_tokens: Option<usize>,
    }

    impl TokenSequenceDataset for TinyDataset {
        fn tokenizer(&self) -> SharedTokenizer {
            self.tokenizer.clone()
        }

        fn token_count(&self) -> usize {
            self.tokens.len()
        }

        fn copy_token_range(&self, start: usize, dst: &mut [u32]) {
            dst.copy_from_slice(&self.tokens[start..start + dst.len()]);
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
            1.0
        }

        fn preferred_logical_document_tokens(&self, _split: DatasetSplit) -> Option<usize> {
            self.preferred_logical_document_tokens
        }
    }

    fn tiny_pretokenized_tokenizer() -> SharedTokenizer {
        use crate::tokenizer::{PretokenizedTokenizerConfig, TokenizerConfig, TokenizerKind};
        TokenizerConfig {
            vocab_path: None,
            kind: TokenizerKind::Pretokenized(PretokenizedTokenizerConfig {
                vocab_size: 256,
                bos_id: None,
                eos_id: Some(255),
                pad_id: None,
                unk_id: None,
            }),
        }
        .fit(std::iter::empty())
        .expect("tokenizer")
    }

    #[test]
    fn streaming_loader_resets_only_on_new_logical_document() {
        let device = <TestBackend as Backend>::Device::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..65).collect()),
            train_len: 65,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
        });
        let loader = StreamingDataLoader::<TestBackend>::new(
            Arc::clone(&dataset),
            DatasetSplit::Train,
            &device,
            4,
            Some(4),
            Some(8),
            1337,
        );
        let mut iter = loader.iter();
        let first = iter.next().expect("first");
        let second = iter.next().expect("second");
        let third = iter.next().expect("third");
        assert!(first.reset_stream_state);
        assert!(!second.reset_stream_state);
        assert!(third.reset_stream_state);
    }

    #[test]
    fn random_sampling_respects_preferred_logical_document_boundaries() {
        let device = <TestBackend as Backend>::Device::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new(vec![
                100, 101, 102, 103, 104, 105, 106, 107, 255, 200, 201, 202, 203, 204, 205, 206,
                207, 255, 300, 301, 302, 303, 304, 305, 306, 307, 255,
            ]),
            train_len: 27,
            block_size: 4,
            batch_size: 8,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: Some(8),
        });

        for _ in 0..32 {
            let batch = sample_batch_with_shape::<TestBackend, _>(
                dataset.as_ref(),
                DatasetSplit::Train,
                dataset.batch_size,
                dataset.block_size,
                None,
                &device,
            );
            let inputs = batch
                .inputs
                .into_data()
                .to_vec::<i64>()
                .expect("batch inputs");
            let targets = batch
                .targets
                .into_data()
                .to_vec::<i64>()
                .expect("batch targets");
            for row in 0..dataset.batch_size {
                let input_row = &inputs[row * dataset.block_size..(row + 1) * dataset.block_size];
                let target_row = &targets[row * dataset.block_size..(row + 1) * dataset.block_size];
                let bucket = input_row[0] / 100;
                assert!(bucket >= 1 && bucket <= 3);
                assert!(input_row.iter().all(|value| *value / 100 == bucket));
                assert!(
                    target_row
                        .iter()
                        .all(|value| { *value == 255 || *value / 100 == bucket })
                );
            }
        }
    }

    #[test]
    fn streaming_loader_seed_is_stable_but_changes_document_order() {
        let device = <TestBackend as Backend>::Device::default();
        let dataset = Arc::new(TinyDataset {
            tokens: Arc::new((0u32..257).collect()),
            train_len: 257,
            block_size: 4,
            batch_size: 2,
            tokenizer: tiny_pretokenized_tokenizer(),
            preferred_logical_document_tokens: None,
        });
        let batch_inputs = |seed| {
            let loader = StreamingDataLoader::<TestBackend>::new(
                Arc::clone(&dataset),
                DatasetSplit::Train,
                &device,
                4,
                Some(4),
                Some(8),
                seed,
            );
            let batch = loader.iter().next().expect("streaming batch");
            batch
                .inputs
                .to_data()
                .convert::<i64>()
                .into_vec::<i64>()
                .expect("batch tokens")
        };

        let first = batch_inputs(1337);
        let repeated = batch_inputs(1337);
        let different = batch_inputs(7331);

        assert_eq!(first, repeated);
        assert_ne!(first, different);
    }
}

struct RandomIterator<B: Backend> {
    dataset: Arc<dyn TokenSequenceDataset>,
    split: DatasetSplit,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    summary_event_token_ids: Option<Vec<u32>>,
}

impl<B: Backend> Iterator for RandomIterator<B> {
    type Item = SequenceBatch<B>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.step >= self.steps_total {
            return None;
        }
        self.step += 1;

        if let Some(counter) = &self.consumed_steps {
            if let Some(limit) = self.total_steps {
                let previous = counter.fetch_add(1, Ordering::Relaxed);
                if previous >= limit {
                    return None;
                }
            } else {
                counter.fetch_add(1, Ordering::Relaxed);
            }
        }

        Some(sample_batch_with_shape::<B, _>(
            &*self.dataset,
            self.split,
            self.dataset.batch_size(),
            self.dataset.block_size(),
            self.summary_event_token_ids.as_deref(),
            &self.device,
        ))
    }
}

impl<B: Backend> DataLoaderIterator<SequenceBatch<B>> for RandomIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(self.step, self.steps_total)
    }
}
