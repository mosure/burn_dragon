use anyhow::{Result, anyhow};
use half::f16;
use safetensors::{SafeTensors, tensor::TensorView};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct PrecomputedClipFeatureStore {
    current_features: Vec<f32>,
    future_features: Vec<f32>,
    clip_count: usize,
    context_len: usize,
    target_len: usize,
    feature_dim: usize,
}

impl PrecomputedClipFeatureStore {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        Self::from_safetensors(&tensors)
    }

    pub fn from_safetensors(tensors: &SafeTensors<'_>) -> Result<Self> {
        let current = tensors
            .tensor("current_features")
            .map_err(|_| anyhow!("missing current_features tensor"))?;
        let future = tensors
            .tensor("future_features")
            .map_err(|_| anyhow!("missing future_features tensor"))?;
        let current_shape = current.shape();
        let future_shape = future.shape();
        if current_shape.len() != 3 || future_shape.len() != 3 {
            return Err(anyhow!(
                "feature tensors must have shape [clips, frames, feature_dim]"
            ));
        }
        if current_shape[0] != future_shape[0] || current_shape[2] != future_shape[2] {
            return Err(anyhow!(
                "current_features and future_features must agree on clip count and feature_dim"
            ));
        }
        Ok(Self {
            current_features: tensor_to_f32(&current)?,
            future_features: tensor_to_f32(&future)?,
            clip_count: current_shape[0],
            context_len: current_shape[1],
            target_len: future_shape[1],
            feature_dim: current_shape[2],
        })
    }

    pub fn clip_count(&self) -> usize {
        self.clip_count
    }

    pub fn context_len(&self) -> usize {
        self.context_len
    }

    pub fn target_len(&self) -> usize {
        self.target_len
    }

    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    pub fn current_clip(&self, clip_idx: usize) -> Option<&[f32]> {
        self.slice(&self.current_features, clip_idx, self.context_len)
    }

    pub fn future_clip(&self, clip_idx: usize) -> Option<&[f32]> {
        self.slice(&self.future_features, clip_idx, self.target_len)
    }

    fn slice<'a>(&'a self, source: &'a [f32], clip_idx: usize, frames: usize) -> Option<&'a [f32]> {
        if clip_idx >= self.clip_count {
            return None;
        }
        let clip_span = frames * self.feature_dim;
        let start = clip_idx * clip_span;
        let end = start + clip_span;
        source.get(start..end)
    }
}

fn tensor_to_f32(view: &TensorView<'_>) -> Result<Vec<f32>> {
    match view.dtype() {
        safetensors::Dtype::F32 => Ok(view
            .data()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("f32 chunk")))
            .collect()),
        safetensors::Dtype::F16 => Ok(view
            .data()
            .chunks_exact(2)
            .map(|chunk| {
                let value = u16::from_le_bytes(chunk.try_into().expect("f16 chunk"));
                f16::from_bits(value).to_f32()
            })
            .collect()),
        other => Err(anyhow!(
            "unsupported dtype in clip feature store: {other:?}"
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, View, serialize_to_file};
    use tempfile::NamedTempFile;

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

        fn data(&self) -> std::borrow::Cow<'_, [u8]> {
            std::borrow::Cow::Borrowed(&self.data)
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

    #[test]
    fn loads_clip_feature_store() {
        let temp = NamedTempFile::new().expect("tempfile");
        let mut tensors = Vec::new();
        tensors.push((
            "current_features".to_string(),
            tensor_f32(&[1, 2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        ));
        tensors.push((
            "future_features".to_string(),
            tensor_f32(&[1, 1, 3], &[7.0, 8.0, 9.0]),
        ));
        serialize_to_file(tensors, None, temp.path()).expect("write safetensors");

        let store = PrecomputedClipFeatureStore::from_file(temp.path()).expect("load store");
        assert_eq!(store.clip_count(), 1);
        assert_eq!(store.context_len(), 2);
        assert_eq!(store.target_len(), 1);
        assert_eq!(store.current_clip(0).expect("current")[0], 1.0);
        assert_eq!(store.future_clip(0).expect("future")[2], 9.0);
    }
}
