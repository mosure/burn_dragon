use crate::{FixationPoint, FixationSet, FrameFixationTrace};
use anyhow::{Result, anyhow};
use half::f16;
use safetensors::{SafeTensors, tensor::TensorView};
use std::fs;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct AutoGazeTraceStore {
    traces: Vec<FrameFixationTrace>,
    clip_len: usize,
    k: usize,
    visibility_maps: Option<Vec<f32>>,
    visibility_height: usize,
    visibility_width: usize,
}

impl AutoGazeTraceStore {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        let tensors = SafeTensors::deserialize(&bytes)?;
        Self::from_safetensors(&tensors)
    }

    pub fn from_safetensors(tensors: &SafeTensors<'_>) -> Result<Self> {
        let fixations = tensors
            .tensor("fixations")
            .map_err(|_| anyhow!("missing fixations tensor"))?;
        let scales = tensors
            .tensor("scales")
            .map_err(|_| anyhow!("missing scales tensor"))?;
        let confidences = tensors
            .tensor("confidences")
            .map_err(|_| anyhow!("missing confidences tensor"))?;
        let stops = tensors
            .tensor("stop_probabilities")
            .map_err(|_| anyhow!("missing stop_probabilities tensor"))?;

        let shape = fixations.shape();
        if shape.len() != 4 || *shape.last().unwrap_or(&0) != 2 {
            return Err(anyhow!(
                "fixations tensor must have shape [clips, frames, k, 2]"
            ));
        }
        let clips = shape[0];
        let clip_len = shape[1];
        let k = shape[2];
        let fixation_values = tensor_to_f32(&fixations)?;
        let scale_values = tensor_to_f32(&scales)?;
        let confidence_values = tensor_to_f32(&confidences)?;
        let stop_values = tensor_to_f32(&stops)?;
        let visibility = tensors.tensor("visibility_maps").ok();
        let (visibility_maps, visibility_height, visibility_width) = if let Some(view) = visibility
        {
            let shape = view.shape();
            if shape.len() != 4 || shape[0] != clips || shape[1] != clip_len {
                return Err(anyhow!(
                    "visibility_maps tensor must have shape [clips, frames, height, width]"
                ));
            }
            let values = tensor_to_f32(&view)?;
            (Some(values), shape[2], shape[3])
        } else {
            (None, 0, 0)
        };

        let mut traces = Vec::with_capacity(clips);
        for clip_idx in 0..clips {
            let mut frames = Vec::with_capacity(clip_len);
            for frame_idx in 0..clip_len {
                let mut points = Vec::with_capacity(k);
                for point_idx in 0..k {
                    let base = (((clip_idx * clip_len + frame_idx) * k + point_idx) * 2) as usize;
                    let scalar_idx = ((clip_idx * clip_len + frame_idx) * k + point_idx) as usize;
                    points.push(FixationPoint::new(
                        fixation_values[base],
                        fixation_values[base + 1],
                        scale_values[scalar_idx],
                        confidence_values[scalar_idx],
                    ));
                }
                let stop_idx = clip_idx * clip_len + frame_idx;
                frames.push(FixationSet::new(points, stop_values[stop_idx], k));
            }
            traces.push(FrameFixationTrace::new(frames));
        }

        Ok(Self {
            traces,
            clip_len,
            k,
            visibility_maps,
            visibility_height,
            visibility_width,
        })
    }

    pub fn trace(&self, index: usize) -> Option<&FrameFixationTrace> {
        self.traces.get(index)
    }

    pub fn len(&self) -> usize {
        self.traces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    pub fn clip_len(&self) -> usize {
        self.clip_len
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn visibility_shape(&self) -> Option<(usize, usize)> {
        self.visibility_maps
            .as_ref()
            .map(|_| (self.visibility_height, self.visibility_width))
    }

    pub fn visibility_map(&self, clip_idx: usize, frame_idx: usize) -> Option<&[f32]> {
        let maps = self.visibility_maps.as_ref()?;
        if clip_idx >= self.traces.len() || frame_idx >= self.clip_len {
            return None;
        }
        let frame_area = self.visibility_height * self.visibility_width;
        let base = (clip_idx * self.clip_len + frame_idx) * frame_area;
        maps.get(base..base + frame_area)
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
            "unsupported dtype in safetensors trace store: {other:?}"
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
    fn loads_trace_store_from_safetensors() {
        let temp = NamedTempFile::new().expect("tempfile");
        let mut tensors = Vec::new();
        tensors.push((
            "fixations".to_string(),
            tensor_f32(&[1, 2, 1, 2], &[0.25, 0.75, 0.5, 0.25]),
        ));
        tensors.push(("scales".to_string(), tensor_f32(&[1, 2, 1], &[0.2, 0.3])));
        tensors.push((
            "confidences".to_string(),
            tensor_f32(&[1, 2, 1], &[0.9, 0.8]),
        ));
        tensors.push((
            "stop_probabilities".to_string(),
            tensor_f32(&[1, 2], &[0.1, 0.2]),
        ));
        tensors.push((
            "visibility_maps".to_string(),
            tensor_f32(&[1, 2, 2, 2], &[1.0, 0.0, 0.0, 1.0, 0.2, 0.4, 0.6, 0.8]),
        ));
        serialize_to_file(tensors, None, temp.path()).expect("write safetensors");

        let store = AutoGazeTraceStore::from_file(temp.path()).expect("load store");
        assert_eq!(store.len(), 1);
        assert_eq!(store.clip_len(), 2);
        assert_eq!(store.k(), 1);
        let trace = store.trace(0).expect("trace");
        assert_eq!(trace.frames[0].points[0].x, 0.25);
        assert_eq!(trace.frames[1].points[0].y, 0.25);
        assert_eq!(store.visibility_shape(), Some((2, 2)));
        assert_eq!(
            store.visibility_map(0, 1).expect("visibility"),
            &[0.2, 0.4, 0.6, 0.8]
        );
    }
}
