use crate::MovingMnistDreamerTrainConfig;
use crate::artifacts::SequenceTensor;
use crate::model::extract_crops;
use crate::runtime::{Backend, TrainBackend, runtime_to_train_tensor3, train_to_runtime_tensor5};
use anyhow::{Context, Result};
use burn::tensor::TensorData;
use burn_autogaze::{AutoGazeTraceStore, FrameFixationTrace, NativeAutoGazeModel};
use burn_dragon_vision::{MovingMnistSplit, VideoClipBatch};
use burn_vjepa::{
    CheckpointVisionDragonTeacher, ClipFeatureTeacher, NativeVjepa2Teacher,
    PrecomputedClipFeatureStore,
};
use std::path::PathBuf;

#[derive(Debug)]
pub(crate) enum AutoGazeSource {
    PassiveFullFrame { frame_size: usize },
    Native(NativeAutoGazeModel<Backend>),
    Store(AutoGazeTraceStore),
}

#[derive(Debug)]
pub(crate) enum ClipTeacherSource {
    Native(NativeVjepa2Teacher<Backend>),
    Checkpoint(CheckpointVisionDragonTeacher<Backend>),
    Store(PrecomputedClipFeatureStore),
}

impl AutoGazeSource {
    pub(crate) fn traces_for_batch(
        &self,
        indices: &[usize],
        batch: &VideoClipBatch<TrainBackend>,
        k_fovea: usize,
    ) -> Vec<FrameFixationTrace> {
        match self {
            Self::PassiveFullFrame { .. } => passive_full_frame_traces(
                indices.len(),
                batch.clip_frames.shape().dims::<5>()[1],
                k_fovea,
            ),
            Self::Native(teacher) => teacher.trace_video(
                train_to_runtime_tensor5(batch.clip_frames.clone(), &batch.clip_frames.device()),
                k_fovea,
                teacher
                    .default_max_gaze_tokens_each_frame()
                    .max(k_fovea.max(1)),
            ),
            Self::Store(store) => indices
                .iter()
                .map(|index| {
                    store
                        .trace(*index)
                        .cloned()
                        .unwrap_or_else(|| panic!("missing AutoGaze trace for clip index {index}"))
                })
                .collect(),
        }
    }

    pub(crate) fn visibility_for_batch(
        &self,
        indices: &[usize],
        steps: usize,
    ) -> Option<SequenceTensor> {
        let store = match self {
            Self::PassiveFullFrame { frame_size } => {
                return Some(passive_full_frame_visibility(
                    indices.len(),
                    steps,
                    *frame_size,
                    *frame_size,
                ));
            }
            Self::Native(_) => return None,
            Self::Store(store) => store,
        };
        let (height, width) = store.visibility_shape()?;
        let steps = steps.min(store.clip_len());
        let mut data = Vec::with_capacity(indices.len() * steps * height * width);
        for index in indices {
            for step in 0..steps {
                let map = store.visibility_map(*index, step)?;
                data.extend_from_slice(map);
            }
        }
        Some(SequenceTensor {
            data,
            batch: indices.len(),
            steps,
            channels: 1,
            height,
            width,
        })
    }
}

impl ClipTeacherSource {
    pub(crate) fn feature_dim(&self) -> usize {
        match self {
            Self::Native(teacher) => teacher.feature_dim(),
            Self::Checkpoint(teacher) => teacher.feature_dim(),
            Self::Store(store) => store.feature_dim(),
        }
    }

    pub(crate) fn target_len_limit(&self) -> Option<usize> {
        match self {
            Self::Native(_) => None,
            Self::Checkpoint(_) => None,
            Self::Store(store) => Some(store.target_len()),
        }
    }

    pub(crate) fn encode_clip_from_batch(
        &self,
        indices: &[usize],
        clip_frames: burn::tensor::Tensor<TrainBackend, 5>,
    ) -> burn::tensor::Tensor<TrainBackend, 3> {
        match self {
            Self::Native(teacher) => {
                let device = clip_frames.device();
                let runtime = train_to_runtime_tensor5(clip_frames, &device);
                runtime_to_train_tensor3(teacher.encode_clip(runtime), &device)
            }
            Self::Checkpoint(teacher) => {
                let device = clip_frames.device();
                let runtime = train_to_runtime_tensor5(clip_frames, &device);
                runtime_to_train_tensor3(teacher.encode_clip(runtime), &device)
            }
            Self::Store(store) => {
                teacher_features_from_store(store, indices, &clip_frames.device())
            }
        }
    }
}

pub(crate) fn uses_global_teacher(config: &MovingMnistDreamerTrainConfig) -> bool {
    !config.model.passive_full_frame
        && (config.model.current_loss_weight > 0.0 || config.model.future_loss_weight > 0.0)
}

pub(crate) fn uses_crop_teacher(config: &MovingMnistDreamerTrainConfig) -> bool {
    !config.model.passive_full_frame || config.model.query_loss_weight > 0.0
}

pub(crate) fn zero_teacher_features(
    batch: usize,
    steps: usize,
    dim: usize,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> burn::tensor::Tensor<TrainBackend, 3> {
    burn::tensor::Tensor::<TrainBackend, 3>::zeros([batch, steps, dim.max(1)], device)
}

pub(crate) fn load_autogaze_source(
    config: &MovingMnistDreamerTrainConfig,
    split: MovingMnistSplit,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<AutoGazeSource> {
    if config.model.passive_full_frame {
        return Ok(AutoGazeSource::PassiveFullFrame {
            frame_size: config.model.frame_size,
        });
    }
    if let Some(hf_dir) = config.autogaze_hf_dir.as_ref() {
        let hf_dir = expand_home_path(hf_dir);
        let teacher = NativeAutoGazeModel::<Backend>::from_hf_dir(&hf_dir, device)
            .with_context(|| format!("load native AutoGaze teacher from {}", hf_dir.display()))?;
        return Ok(AutoGazeSource::Native(teacher));
    }
    let path = match split {
        MovingMnistSplit::Train => config
            .autogaze_train_trace_store
            .as_ref()
            .or(config.autogaze_trace_store.as_ref()),
        MovingMnistSplit::Val => config
            .autogaze_val_trace_store
            .as_ref()
            .or(config.autogaze_trace_store.as_ref()),
    };
    if let Some(path) = path {
        let store = AutoGazeTraceStore::from_file(path)
            .with_context(|| format!("load AutoGaze trace store {}", path.display()))?;
        let requested_clip_len = config.context_len + config.target_len;
        if store.clip_len() < requested_clip_len {
            anyhow::bail!(
                "AutoGaze trace store clip_len={} is smaller than requested clip_len={}",
                store.clip_len(),
                requested_clip_len
            );
        }
        if store.k() < config.model.k_fovea.max(1) {
            anyhow::bail!(
                "AutoGaze trace store k={} is smaller than requested k_fovea={}",
                store.k(),
                config.model.k_fovea.max(1)
            );
        }
        return Ok(AutoGazeSource::Store(store));
    }
    anyhow::bail!(
        "missing AutoGaze teacher for split {:?}; set autogaze_hf_dir or autogaze_{}_trace_store or autogaze_trace_store{}",
        split,
        match split {
            MovingMnistSplit::Train => "train",
            MovingMnistSplit::Val => "val",
        },
        if config.allow_teacher_fallbacks {
            " (teacher fallbacks are disabled in the Dreamer training path)"
        } else {
            ""
        }
    )
}

pub(crate) fn load_vjepa_source(
    config: &mut MovingMnistDreamerTrainConfig,
    split: MovingMnistSplit,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<ClipTeacherSource> {
    if let Some(hf_dir) = config.vjepa_hf_dir.as_ref() {
        let hf_dir = expand_home_path(hf_dir);
        let teacher = NativeVjepa2Teacher::from_hf_dir(&hf_dir, device)
            .with_context(|| format!("load native V-JEPA2 teacher {}", hf_dir.display()))?;
        config.model.teacher_dim = teacher.feature_dim();
        return Ok(ClipTeacherSource::Native(teacher));
    }
    if let Some(checkpoint) = config.vjepa_checkpoint.as_ref() {
        let teacher = CheckpointVisionDragonTeacher::from_checkpoint(
            checkpoint,
            &config.vjepa_config_paths,
            1,
            1,
            device,
        )
        .with_context(|| format!("load V-JEPA checkpoint teacher {}", checkpoint.display()))?;
        config.model.teacher_dim = teacher.feature_dim();
        return Ok(ClipTeacherSource::Checkpoint(teacher));
    }
    let path = match split {
        MovingMnistSplit::Train => config
            .vjepa_train_feature_store
            .as_ref()
            .or(config.vjepa_feature_store.as_ref()),
        MovingMnistSplit::Val => config
            .vjepa_val_feature_store
            .as_ref()
            .or(config.vjepa_feature_store.as_ref()),
    };
    if let Some(path) = path {
        let store = PrecomputedClipFeatureStore::from_file(path)
            .with_context(|| format!("load V-JEPA feature store {}", path.display()))?;
        if store.context_len() != config.context_len || store.target_len() != config.target_len {
            anyhow::bail!(
                "V-JEPA feature store shape [{} current, {} future] does not match config [{} current, {} future]",
                store.context_len(),
                store.target_len(),
                config.context_len,
                config.target_len
            );
        }
        config.model.teacher_dim = store.feature_dim();
        return Ok(ClipTeacherSource::Store(store));
    }
    anyhow::bail!(
        "missing V-JEPA teacher source for split {:?}; set vjepa_hf_dir, vjepa_checkpoint, vjepa_{}_feature_store, or vjepa_feature_store{}",
        split,
        match split {
            MovingMnistSplit::Train => "train",
            MovingMnistSplit::Val => "val",
        },
        if config.allow_teacher_fallbacks {
            " (teacher fallbacks are disabled in the Dreamer training path)"
        } else {
            ""
        }
    )
}

pub(crate) fn load_crop_teacher_source(
    config: &mut MovingMnistDreamerTrainConfig,
    split: MovingMnistSplit,
    device: &<Backend as burn::tensor::backend::Backend>::Device,
) -> Result<ClipTeacherSource> {
    if let Some(hf_dir) = config
        .crop_teacher_hf_dir
        .as_ref()
        .or(config.vjepa_hf_dir.as_ref())
    {
        let hf_dir = expand_home_path(hf_dir);
        let teacher = NativeVjepa2Teacher::from_hf_dir(&hf_dir, device)
            .with_context(|| format!("load native crop V-JEPA2 teacher {}", hf_dir.display()))?;
        config.model.crop_teacher_dim = teacher.feature_dim();
        return Ok(ClipTeacherSource::Native(teacher));
    }
    let path = match split {
        MovingMnistSplit::Train => config
            .crop_teacher_train_feature_store
            .as_ref()
            .or(config.crop_teacher_feature_store.as_ref()),
        MovingMnistSplit::Val => config
            .crop_teacher_val_feature_store
            .as_ref()
            .or(config.crop_teacher_feature_store.as_ref()),
    };
    if let Some(path) = path {
        let store = PrecomputedClipFeatureStore::from_file(path)
            .with_context(|| format!("load crop teacher feature store {}", path.display()))?;
        if store.context_len() != config.context_len || store.target_len() != config.target_len {
            anyhow::bail!(
                "crop teacher feature store shape [{} current, {} future] does not match config [{} current, {} future]",
                store.context_len(),
                store.target_len(),
                config.context_len,
                config.target_len
            );
        }
        config.model.crop_teacher_dim = store.feature_dim();
        return Ok(ClipTeacherSource::Store(store));
    }
    if let Some(checkpoint) = config.vjepa_checkpoint.as_ref() {
        let teacher = CheckpointVisionDragonTeacher::from_checkpoint(
            checkpoint,
            &config.vjepa_config_paths,
            1,
            1,
            device,
        )
        .with_context(|| {
            format!(
                "load crop V-JEPA checkpoint teacher {}",
                checkpoint.display()
            )
        })?;
        config.model.crop_teacher_dim = teacher.feature_dim();
        return Ok(ClipTeacherSource::Checkpoint(teacher));
    }
    anyhow::bail!(
        "missing crop teacher source for split {:?}; set crop_teacher_hf_dir, vjepa_hf_dir, crop_teacher_{}_feature_store, crop_teacher_feature_store, or vjepa_checkpoint{}",
        split,
        match split {
            MovingMnistSplit::Train => "train",
            MovingMnistSplit::Val => "val",
        },
        if config.allow_teacher_fallbacks {
            " (teacher fallbacks are disabled in the Dreamer training path)"
        } else {
            ""
        }
    )
}

pub(crate) fn teacher_features_from_store(
    store: &PrecomputedClipFeatureStore,
    indices: &[usize],
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> burn::tensor::Tensor<TrainBackend, 3> {
    let total_steps = store.context_len() + store.target_len();
    let mut data = Vec::with_capacity(indices.len() * total_steps * store.feature_dim());
    for index in indices {
        let current = store
            .current_clip(*index)
            .unwrap_or_else(|| panic!("missing V-JEPA current features for clip index {index}"));
        let future = store
            .future_clip(*index)
            .unwrap_or_else(|| panic!("missing V-JEPA future features for clip index {index}"));
        data.extend_from_slice(current);
        data.extend_from_slice(future);
    }
    burn::tensor::Tensor::<TrainBackend, 3>::from_data(
        TensorData::new(data, [indices.len(), total_steps, store.feature_dim()]),
        device,
    )
}

pub(crate) fn encode_crop_teacher(
    teacher: &ClipTeacherSource,
    indices: &[usize],
    clip_frames: burn::tensor::Tensor<TrainBackend, 5>,
    traces: &[FrameFixationTrace],
    config: &MovingMnistDreamerTrainConfig,
) -> burn::tensor::Tensor<TrainBackend, 3> {
    if let ClipTeacherSource::Store(store) = teacher {
        return teacher_features_from_store(store, indices, &clip_frames.device());
    }
    let device = clip_frames.device();
    let [batch, clip_len, channels, height, width] = clip_frames.shape().dims::<5>();
    let k = config.model.k_fovea.max(1);
    let mut encoded = Vec::with_capacity(clip_len);
    for step in 0..clip_len {
        let frame = clip_frames
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, channels, height, width]);
        let mut weights = Vec::with_capacity(batch * k);
        let stacked = extract_crops(frame, traces, step, config.model.crop_size, k).reshape([
            batch * k,
            1,
            channels,
            config.model.crop_size,
            config.model.crop_size,
        ]);
        for batch_idx in 0..batch {
            let frame_trace = traces[batch_idx]
                .frames
                .get(step)
                .or_else(|| traces[batch_idx].frames.last())
                .expect("trace frame");
            for point in frame_trace.points.iter().take(k) {
                weights.push(point.confidence.max(1.0e-4));
            }
        }
        let encoded_step = match teacher {
            ClipTeacherSource::Native(teacher) => {
                let runtime = train_to_runtime_tensor5(stacked, &device);
                runtime_to_train_tensor3(teacher.encode_clip(runtime), &device)
            }
            ClipTeacherSource::Checkpoint(teacher) => {
                let runtime = train_to_runtime_tensor5(stacked, &device);
                runtime_to_train_tensor3(teacher.encode_clip(runtime), &device)
            }
            ClipTeacherSource::Store(_) => unreachable!("store path returned early"),
        }
        .reshape([batch, k, teacher.feature_dim()]);
        let weight_tensor = burn::tensor::Tensor::<TrainBackend, 3>::from_data(
            burn::tensor::TensorData::new(weights, [batch, k, 1]),
            &device,
        );
        let merged = encoded_step
            .mul(weight_tensor.clone())
            .sum_dim(1)
            .reshape([batch, teacher.feature_dim()]);
        let denom = weight_tensor
            .sum_dim(1)
            .reshape([batch, 1])
            .add_scalar(1.0e-6);
        encoded.push((merged / denom).unsqueeze_dim::<3>(1));
    }
    burn::tensor::Tensor::cat(encoded, 1)
}

fn expand_home_path(path: &std::path::Path) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix("~/") {
        if let Ok(home) = std::env::var("HOME") {
            return PathBuf::from(home).join(rest);
        }
    }
    path.to_path_buf()
}

pub(crate) fn passive_full_frame_traces(
    batch: usize,
    clip_len: usize,
    k_fovea: usize,
) -> Vec<FrameFixationTrace> {
    use burn_autogaze::{FixationPoint, FixationSet};

    let set = FixationSet::new(
        vec![FixationPoint::new(0.5, 0.5, 1.0, 1.0)],
        1.0,
        k_fovea.max(1),
    );
    (0..batch)
        .map(|_| FrameFixationTrace::new(vec![set.clone(); clip_len]))
        .collect()
}

fn passive_full_frame_visibility(
    batch: usize,
    steps: usize,
    height: usize,
    width: usize,
) -> SequenceTensor {
    SequenceTensor {
        data: vec![1.0; batch * steps * height * width],
        batch,
        steps,
        channels: 1,
        height,
        width,
    }
}
