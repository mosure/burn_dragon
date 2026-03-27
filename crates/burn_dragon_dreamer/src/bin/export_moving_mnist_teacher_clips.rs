use anyhow::{Context, Result};
use burn_dragon_vision::{
    MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VisionNormalize,
};
use safetensors::tensor::{Dtype, View, serialize_to_file};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::PathBuf;

#[derive(Clone)]
struct OwnedTensor {
    shape: Vec<usize>,
    data: Vec<u8>,
    dtype: Dtype,
}

impl View for OwnedTensor {
    fn dtype(&self) -> Dtype {
        self.dtype
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.data)
    }

    fn data_len(&self) -> usize {
        self.data.len()
    }
}

fn tensor_f32(shape: &[usize], values: &[f32]) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 4);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    OwnedTensor {
        shape: shape.to_vec(),
        data,
        dtype: Dtype::F32,
    }
}

fn tensor_i64(shape: &[usize], values: &[i64]) -> OwnedTensor {
    let mut data = Vec::with_capacity(values.len() * 8);
    for value in values {
        data.extend_from_slice(&value.to_le_bytes());
    }
    OwnedTensor {
        shape: shape.to_vec(),
        data,
        dtype: Dtype::I64,
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let output = args.get(1).map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("runs/burn_dragon_dreamer/moving_mnist_teacher_clips.safetensors")
    });
    let split = match args.get(2).map(|value| value.as_str()).unwrap_or("train") {
        "train" | "Train" | "TRAIN" => MovingMnistSplit::Train,
        "val" | "valid" | "validation" | "Val" | "VAL" => MovingMnistSplit::Val,
        other => anyhow::bail!("unknown split {other}; expected train or val"),
    };
    let max_records = args
        .get(3)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(match split {
            MovingMnistSplit::Train => 256,
            MovingMnistSplit::Val => 128,
        });
    let context_len = args
        .get(4)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(4);
    let target_len = args
        .get(5)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(2);
    let extra_future_frames = args
        .get(6)
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(10);

    let normalize = VisionNormalize::new([0.5; 3], [0.5; 3]);
    let seed = match split {
        MovingMnistSplit::Train => 13,
        MovingMnistSplit::Val => 29,
    };
    let dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split,
        frame_size: 28,
        digit_size: 12,
        in_channels: 1,
        context_len,
        target_len,
        extra_future_frames,
        frame_stride: 1,
        max_records: Some(max_records),
        normalize: normalize.clone(),
        min_velocity: 1.0,
        max_velocity: 3.0,
        seed,
    })?;

    let count = dataset.len();
    let clip_len = context_len + target_len + extra_future_frames;
    let mut clips = Vec::with_capacity(count * clip_len * 28 * 28);
    let mut labels = Vec::with_capacity(count);
    for index in 0..count {
        let clip = dataset
            .rendered_clip(index)
            .unwrap_or_else(|| panic!("missing moving mnist clip {index}"));
        clips.extend_from_slice(&clip.frames);
        labels.push(clip.label);
    }

    let mut metadata = HashMap::new();
    metadata.insert(
        "split".to_string(),
        match split {
            MovingMnistSplit::Train => "train".to_string(),
            MovingMnistSplit::Val => "val".to_string(),
        },
    );
    metadata.insert("context_len".to_string(), context_len.to_string());
    metadata.insert("target_len".to_string(), target_len.to_string());
    metadata.insert(
        "extra_future_frames".to_string(),
        extra_future_frames.to_string(),
    );
    metadata.insert("clip_len".to_string(), clip_len.to_string());
    metadata.insert("frame_size".to_string(), "28".to_string());
    metadata.insert("channels".to_string(), "1".to_string());
    metadata.insert("digit_size".to_string(), "12".to_string());
    metadata.insert("normalize_mean".to_string(), "0.5".to_string());
    metadata.insert("normalize_std".to_string(), "0.5".to_string());
    metadata.insert("min_velocity".to_string(), "1.0".to_string());
    metadata.insert("max_velocity".to_string(), "3.0".to_string());
    metadata.insert("seed".to_string(), seed.to_string());

    let tensors = vec![
        (
            "clips".to_string(),
            tensor_f32(&[count, clip_len, 1, 28, 28], &clips),
        ),
        ("labels".to_string(), tensor_i64(&[count], &labels)),
    ];
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }
    serialize_to_file(tensors, Some(metadata), &output)
        .with_context(|| format!("write teacher clip store {}", output.display()))?;
    println!(
        "moving_mnist_teacher_clips wrote={} split={} count={} clip_len={}",
        output.display(),
        match split {
            MovingMnistSplit::Train => "train",
            MovingMnistSplit::Val => "val",
        },
        count,
        clip_len
    );
    Ok(())
}
