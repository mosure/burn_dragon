use std::fs;
use std::sync::Arc;

use burn::data::dataloader::split::split_dataloader;
use burn::tensor::backend::Backend as BackendTrait;
use burn_dragon::language::dataset::{RandomDataLoader, ShakespeareDataset, ShakespeareSplit};
use burn_dragon::language::tokenizer::{ByteTokenizerConfig, TokenizerConfig, TokenizerKind};
use burn_ndarray::NdArray;
use tempfile::tempdir;

#[test]
fn dataset_batches_match_expected_shape() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(512);
    fs::write(&file_path, content).expect("write dataset");

    let block_size = 32;
    let batch_size = 4;
    let tokenizer = TokenizerConfig::default();
    let dataset = ShakespeareDataset::new(cache_dir, block_size, batch_size, 0.8, &tokenizer)
        .expect("create dataset");

    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    <Backend as BackendTrait>::seed(&device, 0);

    let batch = dataset.sample_batch::<Backend>(ShakespeareSplit::Train, &device);
    assert_eq!(batch.inputs.shape().dims(), [batch_size, block_size]);
    assert_eq!(batch.targets.shape().dims(), [batch_size, block_size]);
}

#[test]
fn byte_tokenizer_dataset_initializes() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(128);
    fs::write(&file_path, content).expect("write dataset");

    let tokenizer = TokenizerConfig {
        vocab_path: None,
        kind: TokenizerKind::Byte(ByteTokenizerConfig::default()),
    };

    let dataset = ShakespeareDataset::new(cache_dir, 16, 2, 0.75, &tokenizer)
        .expect("create dataset with byte tokenizer");
    let tokenizer = dataset.tokenizer();
    assert!(tokenizer.len() >= 256);
}

#[test]
fn random_dataloader_split_uses_step_counts_for_ddp() {
    let dir = tempdir().expect("tempdir");
    let cache_dir = dir.path();
    let file_path = cache_dir.join("tinyshakespeare.txt");
    let content = b"The quick brown fox jumps over the lazy dog. ".repeat(256);
    fs::write(&file_path, content).expect("write dataset");

    let tokenizer = TokenizerConfig::default();
    let dataset =
        Arc::new(ShakespeareDataset::new(cache_dir, 16, 4, 0.8, &tokenizer).expect("dataset"));

    type Backend = NdArray<f32>;
    let device = <Backend as BackendTrait>::Device::default();
    let loader: Arc<dyn burn::data::dataloader::DataLoader<Backend, _>> =
        Arc::new(RandomDataLoader::<Backend>::new(
            Arc::clone(&dataset),
            ShakespeareSplit::Train,
            &device,
            5,
            Some(1),
        ));

    assert_eq!(
        loader.num_items(),
        5,
        "num_items should be in steps, not examples"
    );

    let splits = split_dataloader(Arc::clone(&loader), &[device, device]);
    assert_eq!(splits.len(), 2);
    assert_eq!(splits[0].num_items(), 2);
    assert_eq!(splits[1].num_items(), 3);

    let left_count = splits[0].iter().count();
    let right_count = splits[1].iter().count();
    assert_eq!(
        left_count, 1,
        "local DDP shard should respect per-shard total_steps"
    );
    assert_eq!(
        right_count, 1,
        "local DDP shard should not share a consumed-step counter"
    );
}
