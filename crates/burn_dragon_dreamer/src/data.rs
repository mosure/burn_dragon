use crate::runtime::TrainBackend;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use burn_dragon_vision::{MovingMnistVideoDataset, VideoClipBatch};

#[derive(Clone)]
pub struct CachedSequenceSplit<B: Backend, TTrace: Clone> {
    device: B::Device,
    clip_frames: Tensor<B, 5>,
    actions: Option<Tensor<B, 3>>,
    teacher_features: Tensor<B, 3>,
    crop_teacher_features: Tensor<B, 3>,
    traces: Vec<TTrace>,
}

pub struct CachedSequenceBatch<B: Backend, TTrace: Clone> {
    pub clip_frames: Tensor<B, 5>,
    pub actions: Option<Tensor<B, 3>>,
    pub teacher_features: Tensor<B, 3>,
    pub crop_teacher_features: Tensor<B, 3>,
    pub traces: Vec<TTrace>,
}

pub trait DreamerSequenceDataset {
    fn len(&self) -> usize;

    fn batch_from_indices<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> VideoClipBatch<B>;

    fn action_batch_from_indices<B: Backend>(
        &self,
        _indices: &[usize],
        _device: &B::Device,
    ) -> Option<Tensor<B, 3>> {
        None
    }
}

impl DreamerSequenceDataset for MovingMnistVideoDataset {
    fn len(&self) -> usize {
        Self::len(self)
    }

    fn batch_from_indices<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> VideoClipBatch<B> {
        Self::batch_from_indices::<B>(self, indices, device)
    }

    fn action_batch_from_indices<B: Backend>(
        &self,
        indices: &[usize],
        device: &B::Device,
    ) -> Option<Tensor<B, 3>> {
        Some(Self::action_batch_from_indices::<B>(self, indices, device))
    }
}

pub(crate) fn sample_indices(len: usize, batch_size: usize, seed: u64) -> Vec<usize> {
    let len = len.max(1);
    let batch_size = batch_size.max(1);
    let mut state = seed ^ 0x9E37_79B9_7F4A_7C15;
    let mut indices = Vec::with_capacity(batch_size);
    for offset in 0..batch_size {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15 ^ offset as u64);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        indices.push((z as usize) % len);
    }
    indices
}

impl<B: Backend, TTrace: Clone> CachedSequenceSplit<B, TTrace> {
    pub fn new(
        device: B::Device,
        clip_frames: Vec<f32>,
        clip_frames_shape: [usize; 4],
        actions: Option<Vec<f32>>,
        actions_shape: Option<[usize; 2]>,
        teacher_features: Vec<f32>,
        teacher_features_shape: [usize; 2],
        crop_teacher_features: Vec<f32>,
        crop_teacher_features_shape: [usize; 2],
        traces: Vec<TTrace>,
    ) -> Self {
        Self {
            clip_frames: Tensor::<B, 5>::from_data(
                TensorData::new(
                    clip_frames,
                    [
                        traces.len(),
                        clip_frames_shape[0],
                        clip_frames_shape[1],
                        clip_frames_shape[2],
                        clip_frames_shape[3],
                    ],
                ),
                &device,
            ),
            actions: actions
                .zip(actions_shape)
                .map(|(actions, [action_steps, action_dim])| {
                    Tensor::<B, 3>::from_data(
                        TensorData::new(actions, [traces.len(), action_steps, action_dim]),
                        &device,
                    )
                }),
            teacher_features: Tensor::<B, 3>::from_data(
                TensorData::new(
                    teacher_features,
                    [
                        traces.len(),
                        teacher_features_shape[0],
                        teacher_features_shape[1],
                    ],
                ),
                &device,
            ),
            crop_teacher_features: Tensor::<B, 3>::from_data(
                TensorData::new(
                    crop_teacher_features,
                    [
                        traces.len(),
                        crop_teacher_features_shape[0],
                        crop_teacher_features_shape[1],
                    ],
                ),
                &device,
            ),
            device,
            traces,
        }
    }

    pub fn len(&self) -> usize {
        self.traces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.traces.is_empty()
    }

    pub fn batch(&self, indices: &[usize]) -> CachedSequenceBatch<B, TTrace> {
        CachedSequenceBatch {
            clip_frames: gather_rows_tensor(self.clip_frames.clone(), indices, &self.device),
            actions: self
                .actions
                .as_ref()
                .map(|tensor| gather_rows_tensor(tensor.clone(), indices, &self.device)),
            teacher_features: gather_rows_tensor(
                self.teacher_features.clone(),
                indices,
                &self.device,
            ),
            crop_teacher_features: gather_rows_tensor(
                self.crop_teacher_features.clone(),
                indices,
                &self.device,
            ),
            traces: indices
                .iter()
                .map(|&index| self.traces[index].clone())
                .collect(),
        }
    }
}

pub fn build_cached_sequence_split<D, TTrace, FTrace, FTeacher, FCropTeacher>(
    dataset: &D,
    teacher_cache_batch_size: usize,
    device: &<TrainBackend as Backend>::Device,
    mut traces_for_batch: FTrace,
    mut encode_teacher: FTeacher,
    mut encode_crop_teacher: FCropTeacher,
) -> CachedSequenceSplit<TrainBackend, TTrace>
where
    D: DreamerSequenceDataset,
    TTrace: Clone,
    FTrace: FnMut(&[usize], &VideoClipBatch<TrainBackend>) -> Vec<TTrace>,
    FTeacher: FnMut(&[usize], Tensor<TrainBackend, 5>) -> Tensor<TrainBackend, 3>,
    FCropTeacher: FnMut(&[usize], Tensor<TrainBackend, 5>, &[TTrace]) -> Tensor<TrainBackend, 3>,
{
    let indices: Vec<usize> = (0..dataset.len()).collect();
    let chunk_size = teacher_cache_batch_size.clamp(1, 8);
    let mut clip_frames = Vec::new();
    let mut actions = None::<Vec<f32>>;
    let mut actions_shape = None::<[usize; 2]>;
    let mut teacher_features = Vec::new();
    let mut crop_teacher_features = Vec::new();
    let mut clip_frames_shape = None;
    let mut teacher_features_shape = None;
    let mut crop_teacher_features_shape = None;
    let mut traces = Vec::with_capacity(indices.len());

    for chunk_indices in indices.chunks(chunk_size) {
        let batch = dataset.batch_from_indices::<TrainBackend>(chunk_indices, device);
        let chunk_actions =
            dataset.action_batch_from_indices::<TrainBackend>(chunk_indices, device);
        let chunk_traces = traces_for_batch(chunk_indices, &batch);
        let chunk_clip_frames = batch.clip_frames.detach();
        let chunk_teacher_features =
            encode_teacher(chunk_indices, chunk_clip_frames.clone().detach());
        let chunk_crop_teacher_features = encode_crop_teacher(
            chunk_indices,
            chunk_clip_frames.clone().detach(),
            &chunk_traces,
        );

        traces.extend(chunk_traces);
        let [chunk_batch, frames, channels, height, width] = chunk_clip_frames.shape().dims::<5>();
        if let Some(chunk_actions) = chunk_actions {
            let [action_batch, action_steps, action_dim] = chunk_actions.shape().dims::<3>();
            assert_eq!(action_batch, chunk_indices.len());
            actions_shape.get_or_insert([action_steps, action_dim]);
            actions.get_or_insert_with(Vec::new).extend(
                chunk_actions
                    .to_data()
                    .to_vec::<f32>()
                    .expect("cached actions"),
            );
        }
        let [teacher_batch, teacher_steps, teacher_dim] =
            chunk_teacher_features.shape().dims::<3>();
        let [crop_batch, crop_steps, crop_dim] = chunk_crop_teacher_features.shape().dims::<3>();
        assert_eq!(chunk_batch, chunk_indices.len());
        assert_eq!(teacher_batch, chunk_indices.len());
        assert_eq!(crop_batch, chunk_indices.len());
        clip_frames_shape.get_or_insert([frames, channels, height, width]);
        teacher_features_shape.get_or_insert([teacher_steps, teacher_dim]);
        crop_teacher_features_shape.get_or_insert([crop_steps, crop_dim]);
        clip_frames.extend(
            chunk_clip_frames
                .to_data()
                .to_vec::<f32>()
                .expect("cached clip frames"),
        );
        teacher_features.extend(
            chunk_teacher_features
                .to_data()
                .to_vec::<f32>()
                .expect("cached teacher features"),
        );
        crop_teacher_features.extend(
            chunk_crop_teacher_features
                .to_data()
                .to_vec::<f32>()
                .expect("cached crop teacher features"),
        );
    }

    CachedSequenceSplit::new(
        device.clone(),
        clip_frames,
        clip_frames_shape.expect("cached clip frame shape"),
        actions,
        actions_shape,
        teacher_features,
        teacher_features_shape.expect("cached teacher shape"),
        crop_teacher_features,
        crop_teacher_features_shape.expect("cached crop teacher shape"),
        traces,
    )
}

fn gather_rows_tensor<B: Backend, const D: usize>(
    tensor: Tensor<B, D>,
    indices: &[usize],
    device: &B::Device,
) -> Tensor<B, D> {
    let index = Tensor::<B, 1, Int>::from_data(
        TensorData::new(
            indices.iter().map(|&index| index as i64).collect(),
            [indices.len()],
        ),
        device,
    );
    tensor.select(0, index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn_ndarray::NdArray;

    #[test]
    fn cached_split_batches_requested_rows_in_order() {
        type TestBackend = NdArray<f32>;
        let split = CachedSequenceSplit::<TestBackend, usize>::new(
            Default::default(),
            (0..2 * 3 * 1 * 2 * 2).map(|x| x as f32).collect(),
            [3, 1, 2, 2],
            Some((0..2 * 3 * 2).map(|x| x as f32).collect()),
            Some([3, 2]),
            (0..2 * 3 * 4).map(|x| x as f32).collect(),
            [3, 4],
            (0..2 * 3 * 2).map(|x| x as f32).collect(),
            [3, 2],
            vec![10, 20],
        );
        let batch = split.batch(&[1, 0]);
        assert_eq!(batch.clip_frames.shape().dims::<5>(), [2, 3, 1, 2, 2]);
        assert_eq!(
            batch.actions.expect("cached actions").shape().dims::<3>(),
            [2, 3, 2]
        );
        assert_eq!(batch.teacher_features.shape().dims::<3>(), [2, 3, 4]);
        assert_eq!(batch.crop_teacher_features.shape().dims::<3>(), [2, 3, 2]);
        assert_eq!(batch.traces, vec![20, 10]);
        let clip = batch
            .clip_frames
            .to_data()
            .to_vec::<f32>()
            .expect("clip data");
        assert_eq!(clip[0], 12.0);
        assert_eq!(clip[clip.len() - 1], 11.0);
    }
}
