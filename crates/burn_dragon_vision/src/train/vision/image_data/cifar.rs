use anyhow::{Result, anyhow};
use burn::data::dataloader::{DataLoader, DataLoaderIterator, Progress};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use rand::prelude::*;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

const CIFAR_HEIGHT: usize = 32;
const CIFAR_WIDTH: usize = 32;
const CIFAR_CHANNELS: usize = 3;
const CIFAR_IMAGE_BYTES: usize = CIFAR_HEIGHT * CIFAR_WIDTH * CIFAR_CHANNELS;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CifarType {
    Cifar10,
    Cifar100,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CifarSplit {
    Train,
    Test,
}

pub struct CifarDataset {
    images: Vec<u8>,
    labels: Vec<u8>,
}

impl CifarDataset {
    pub fn new<P: AsRef<Path>>(root: P, cifar_type: CifarType, split: CifarSplit) -> Result<Self> {
        let root = resolve_root(root.as_ref(), cifar_type);
        let (files, record_len, label_offset) = match (cifar_type, split) {
            (CifarType::Cifar10, CifarSplit::Train) => (
                vec![
                    "data_batch_1.bin",
                    "data_batch_2.bin",
                    "data_batch_3.bin",
                    "data_batch_4.bin",
                    "data_batch_5.bin",
                ],
                1 + CIFAR_IMAGE_BYTES,
                0,
            ),
            (CifarType::Cifar10, CifarSplit::Test) => {
                (vec!["test_batch.bin"], 1 + CIFAR_IMAGE_BYTES, 0)
            }
            (CifarType::Cifar100, CifarSplit::Train) => {
                (vec!["train.bin"], 2 + CIFAR_IMAGE_BYTES, 1)
            }
            (CifarType::Cifar100, CifarSplit::Test) => (vec!["test.bin"], 2 + CIFAR_IMAGE_BYTES, 1),
        };

        let mut images = Vec::new();
        let mut labels = Vec::new();
        for file in files {
            let path = root.join(file);
            let (file_images, file_labels) = read_records(&path, record_len, label_offset)?;
            images.extend_from_slice(&file_images);
            labels.extend_from_slice(&file_labels);
        }

        Ok(Self { images, labels })
    }

    pub fn with_max_records(mut self, max_records: Option<usize>) -> Self {
        let Some(limit) = max_records else {
            return self;
        };
        let limit = limit.min(self.labels.len());
        self.labels.truncate(limit);
        self.images
            .truncate(limit.saturating_mul(CIFAR_IMAGE_BYTES));
        self
    }

    pub fn len(&self) -> usize {
        self.labels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    pub fn steps_per_epoch(&self, batch_size: usize) -> usize {
        if batch_size == 0 {
            return 1;
        }
        self.len().div_ceil(batch_size).max(1)
    }

    pub fn sample_batch<B: Backend>(&self, batch_size: usize, device: &B::Device) -> CifarBatch<B> {
        let len = self.len();
        assert!(len > 0, "cifar dataset is empty");
        let mut rng = thread_rng();

        let mut images = vec![0.0f32; batch_size * CIFAR_IMAGE_BYTES];
        let mut labels = vec![0i64; batch_size];

        for (batch_idx, label) in labels.iter_mut().enumerate() {
            let idx = rng.gen_range(0..len);
            let src_offset = idx * CIFAR_IMAGE_BYTES;
            let dst_offset = batch_idx * CIFAR_IMAGE_BYTES;
            for i in 0..CIFAR_IMAGE_BYTES {
                images[dst_offset + i] = self.images[src_offset + i] as f32 / 255.0;
            }
            *label = self.labels[idx] as i64;
        }

        let images_tensor = Tensor::<B, 4>::from_data(
            TensorData::new(
                images,
                [batch_size, CIFAR_CHANNELS, CIFAR_HEIGHT, CIFAR_WIDTH],
            ),
            device,
        );
        let labels_tensor =
            Tensor::<B, 1, Int>::from_data(TensorData::new(labels, [batch_size]), device);
        CifarBatch::new(images_tensor, labels_tensor)
    }
}

#[derive(Clone)]
pub struct CifarBatch<B: Backend> {
    pub images: Tensor<B, 4>,
    pub labels: Tensor<B, 1, Int>,
}

impl<B: Backend> CifarBatch<B> {
    pub fn new(images: Tensor<B, 4>, labels: Tensor<B, 1, Int>) -> Self {
        Self { images, labels }
    }
}

pub struct CifarDataLoader<B: Backend> {
    dataset: Arc<CifarDataset>,
    batch_size: usize,
    steps_per_epoch: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
    device: B::Device,
}

impl<B: Backend> Clone for CifarDataLoader<B> {
    fn clone(&self) -> Self {
        Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: self.device.clone(),
        }
    }
}

impl<B: Backend> CifarDataLoader<B> {
    pub fn new(
        dataset: Arc<CifarDataset>,
        batch_size: usize,
        device: &B::Device,
        steps_per_epoch: usize,
        total_steps: Option<usize>,
    ) -> Self {
        let steps_per_epoch = if steps_per_epoch == 0 {
            dataset.steps_per_epoch(batch_size)
        } else {
            steps_per_epoch
        };
        let steps_per_epoch = steps_per_epoch.max(1);
        let total_steps = total_steps.filter(|value| *value > 0);
        let consumed_steps = total_steps.as_ref().map(|_| Arc::new(AtomicUsize::new(0)));

        Self {
            dataset,
            batch_size,
            steps_per_epoch,
            total_steps,
            consumed_steps,
            device: device.clone(),
        }
    }
}

impl<B> DataLoader<B, CifarBatch<B>> for CifarDataLoader<B>
where
    B: Backend + 'static,
    B::Device: Clone,
{
    fn iter<'a>(&'a self) -> Box<dyn DataLoaderIterator<CifarBatch<B>> + 'a> {
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

        Box::new(CifarIterator {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            device: self.device.clone(),
            steps_total,
            step: 0,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.clone(),
        })
    }

    fn num_items(&self) -> usize {
        self.steps_per_epoch * self.batch_size
    }

    fn to_device(&self, device: &B::Device) -> Arc<dyn DataLoader<B, CifarBatch<B>>> {
        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: self.steps_per_epoch,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: device.clone(),
        })
    }

    fn slice(&self, start: usize, end: usize) -> Arc<dyn DataLoader<B, CifarBatch<B>>> {
        let end = end.min(self.steps_per_epoch);
        let start = start.min(end);
        let steps = (end - start).max(1);

        Arc::new(Self {
            dataset: Arc::clone(&self.dataset),
            batch_size: self.batch_size,
            steps_per_epoch: steps,
            total_steps: self.total_steps,
            consumed_steps: self.consumed_steps.as_ref().map(Arc::clone),
            device: self.device.clone(),
        })
    }
}

struct CifarIterator<B: Backend> {
    dataset: Arc<CifarDataset>,
    batch_size: usize,
    device: B::Device,
    steps_total: usize,
    step: usize,
    total_steps: Option<usize>,
    consumed_steps: Option<Arc<AtomicUsize>>,
}

impl<B: Backend> Iterator for CifarIterator<B> {
    type Item = CifarBatch<B>;

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
        Some(
            self.dataset
                .sample_batch::<B>(self.batch_size, &self.device),
        )
    }
}

impl<B: Backend> DataLoaderIterator<CifarBatch<B>> for CifarIterator<B> {
    fn progress(&self) -> Progress {
        Progress::new(
            self.step * self.batch_size,
            self.steps_total * self.batch_size,
        )
    }
}

fn resolve_root(root: &Path, cifar_type: CifarType) -> PathBuf {
    let subdir = match cifar_type {
        CifarType::Cifar10 => "cifar-10-batches-bin",
        CifarType::Cifar100 => "cifar-100-binary",
    };
    let candidate = root.join(subdir);
    if candidate.is_dir() {
        candidate
    } else {
        root.to_path_buf()
    }
}

fn read_records(path: &Path, record_len: usize, label_offset: usize) -> Result<(Vec<u8>, Vec<u8>)> {
    let bytes =
        fs::read(path).map_err(|err| anyhow!("failed to read {}: {err}", path.display()))?;
    if bytes.len() % record_len != 0 {
        return Err(anyhow!(
            "invalid CIFAR record size in {}: {} bytes (record_len={})",
            path.display(),
            bytes.len(),
            record_len
        ));
    }
    if label_offset >= record_len {
        return Err(anyhow!("label offset out of range for {}", path.display()));
    }

    let records = bytes.len() / record_len;
    let mut labels = Vec::with_capacity(records);
    let mut images = Vec::with_capacity(records * CIFAR_IMAGE_BYTES);
    let image_offset = record_len - CIFAR_IMAGE_BYTES;

    for idx in 0..records {
        let start = idx * record_len;
        labels.push(bytes[start + label_offset]);
        let image_start = start + image_offset;
        images.extend_from_slice(&bytes[image_start..image_start + CIFAR_IMAGE_BYTES]);
    }

    Ok((images, labels))
}
