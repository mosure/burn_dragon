use crate::artifacts::{
    ArtifactMetrics, DreamerArtifactSnapshot, FixationPointArtifact, FixationSequence,
    LatentTensor, SequenceTensor, write_moving_mnist_artifacts,
};
use crate::checkpoint::load_module_checkpoint;
use crate::{
    DragonDreamer, DreamerDebugOutput, DreamerLatentBackend, MovingMnistDreamerTrainConfig,
};
use anyhow::{Context, Result};
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::tensor::TensorData;
use burn_autodiff::Autodiff;
use burn_autogaze::{AutoGazeTraceStore, FrameFixationTrace};
use burn_cuda::{Cuda, CudaDevice};
use burn_dragon_vision::{
    MovingMnistSplit, MovingMnistVideoDataset, MovingMnistVideoDatasetConfig, VisionNormalize,
};
use burn_vjepa::{CheckpointVisionDragonTeacher, ClipFeatureTeacher, PrecomputedClipFeatureStore};
use std::fs;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) type Backend = Cuda<f32>;
pub(crate) type TrainBackend = Autodiff<Backend>;

#[derive(Clone, Debug)]
pub struct MovingMnistDreamerRunSummary {
    pub initial_valid_total: f32,
    pub final_valid_total: f32,
    pub initial_valid_future: f32,
    pub final_valid_future: f32,
    pub best_valid_total: f32,
    pub best_valid_future: f32,
    pub final_train_total: f32,
    pub final_train_future: f32,
    pub final_valid_recon_current: f32,
    pub final_valid_recon_future: f32,
    pub final_valid_future_psnr: f32,
    pub final_valid_future_fg_iou: f32,
    pub final_valid_future_motion_ratio: f32,
    pub final_valid_future_stop_mean: f32,
    pub final_valid_future_stop_std: f32,
    pub final_valid_tokenizer: f32,
    pub final_valid_tokenizer_recon: f32,
    pub final_valid_slot_align: f32,
    pub latent_backend: String,
    pub steps: usize,
    pub run_dir: Option<PathBuf>,
    pub artifact_dir: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) enum AutoGazeSource {
    Store(AutoGazeTraceStore),
}

#[derive(Debug)]
pub(crate) enum ClipTeacherSource {
    Checkpoint(CheckpointVisionDragonTeacher<TrainBackend>),
    Store(PrecomputedClipFeatureStore),
}

#[derive(Clone)]
pub(crate) struct CachedMovingMnistSplit {
    clip_frames: burn::tensor::Tensor<TrainBackend, 5>,
    teacher_features: burn::tensor::Tensor<TrainBackend, 3>,
    crop_teacher_features: burn::tensor::Tensor<TrainBackend, 3>,
    traces: Vec<FrameFixationTrace>,
}

pub(crate) struct CachedMovingMnistBatch {
    pub clip_frames: burn::tensor::Tensor<TrainBackend, 5>,
    pub teacher_features: burn::tensor::Tensor<TrainBackend, 3>,
    pub crop_teacher_features: burn::tensor::Tensor<TrainBackend, 3>,
    pub traces: Vec<FrameFixationTrace>,
}

impl AutoGazeSource {
    pub(crate) fn traces_for_batch(
        &self,
        indices: &[usize],
        _batch: &burn_dragon_vision::VideoClipBatch<TrainBackend>,
        _k_fovea: usize,
    ) -> Vec<FrameFixationTrace> {
        match self {
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

    fn visibility_for_batch(&self, indices: &[usize], steps: usize) -> Option<SequenceTensor> {
        let store = match self {
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
            Self::Checkpoint(teacher) => teacher.feature_dim(),
            Self::Store(store) => store.feature_dim(),
        }
    }

    pub(crate) fn target_len_limit(&self) -> Option<usize> {
        match self {
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
            Self::Checkpoint(teacher) => teacher.encode_clip(clip_frames),
            Self::Store(store) => {
                teacher_features_from_store(store, indices, &clip_frames.device())
            }
        }
    }
}

impl CachedMovingMnistSplit {
    pub(crate) fn build(
        dataset: &MovingMnistVideoDataset,
        global_teacher: &ClipTeacherSource,
        crop_teacher: &ClipTeacherSource,
        teacher: &AutoGazeSource,
        config: &MovingMnistDreamerTrainConfig,
        device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
    ) -> Self {
        let indices: Vec<usize> = (0..dataset.len()).collect();
        let batch = dataset.batch_from_indices::<TrainBackend>(&indices, device);
        let traces = teacher.traces_for_batch(&indices, &batch, config.model.k_fovea);
        let clip_frames = batch.clip_frames.detach();
        let teacher_features =
            global_teacher.encode_clip_from_batch(&indices, clip_frames.clone().detach());
        let crop_teacher_features = encode_crop_teacher(
            crop_teacher,
            &indices,
            clip_frames.clone().detach(),
            &traces,
            config,
        );
        Self {
            clip_frames,
            teacher_features,
            crop_teacher_features,
            traces,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.traces.len()
    }

    pub(crate) fn batch(&self, indices: &[usize]) -> CachedMovingMnistBatch {
        CachedMovingMnistBatch {
            clip_frames: select_rows(self.clip_frames.clone(), indices),
            teacher_features: select_rows(self.teacher_features.clone(), indices),
            crop_teacher_features: select_rows(self.crop_teacher_features.clone(), indices),
            traces: indices
                .iter()
                .map(|&index| self.traces[index].clone())
                .collect(),
        }
    }
}

fn select_rows<const D: usize>(
    tensor: burn::tensor::Tensor<TrainBackend, D>,
    indices: &[usize],
) -> burn::tensor::Tensor<TrainBackend, D> {
    let mut rows = Vec::with_capacity(indices.len());
    for &index in indices {
        rows.push(tensor.clone().slice_dim(0, index..index + 1));
    }
    burn::tensor::Tensor::cat(rows, 0)
}

pub(crate) fn prepare_run_dir(config: &MovingMnistDreamerTrainConfig) -> Result<Option<PathBuf>> {
    let Some(run_root) = config.run_root.as_ref() else {
        return Ok(None);
    };
    fs::create_dir_all(run_root)
        .with_context(|| format!("create run root {}", run_root.display()))?;
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock before unix epoch")?
        .as_millis();
    for attempt in 0..1024usize {
        let candidate = run_root.join(format!("exp_{stamp}_{attempt:03}"));
        if !candidate.exists() {
            fs::create_dir_all(&candidate)
                .with_context(|| format!("create run dir {}", candidate.display()))?;
            return Ok(Some(candidate));
        }
    }
    anyhow::bail!(
        "unable to allocate a unique run directory under {}",
        run_root.display()
    );
}

fn load_saved_tokenizer_config(
    checkpoint_base: &std::path::Path,
) -> Result<Option<MovingMnistDreamerTrainConfig>> {
    let Some(run_dir) = checkpoint_base.parent().and_then(|path| path.parent()) else {
        return Ok(None);
    };
    let config_path = run_dir.join("config.json");
    if !config_path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(&config_path)
        .with_context(|| format!("read tokenizer config {}", config_path.display()))?;
    let config = serde_json::from_slice::<MovingMnistDreamerTrainConfig>(&bytes)
        .with_context(|| format!("parse tokenizer config {}", config_path.display()))?;
    Ok(Some(config))
}

pub(crate) fn load_autogaze_source(
    config: &MovingMnistDreamerTrainConfig,
    split: MovingMnistSplit,
) -> Result<AutoGazeSource> {
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
        "missing AutoGaze teacher traces for split {:?}; set autogaze_{}_trace_store or autogaze_trace_store{}",
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
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Result<ClipTeacherSource> {
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
    if let Some(path) = config.vjepa_feature_store.as_ref() {
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
        "missing V-JEPA teacher source; set vjepa_checkpoint or vjepa_feature_store{}",
        if config.allow_teacher_fallbacks {
            " (teacher fallbacks are disabled in the Dreamer training path)"
        } else {
            ""
        }
    )
}

pub(crate) fn load_crop_teacher_source(
    config: &mut MovingMnistDreamerTrainConfig,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Result<ClipTeacherSource> {
    if let Some(path) = config.crop_teacher_feature_store.as_ref() {
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
        "missing crop teacher source; set crop_teacher_feature_store or vjepa_checkpoint{}",
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

pub fn train_moving_mnist(
    mut config: MovingMnistDreamerTrainConfig,
) -> Result<MovingMnistDreamerRunSummary> {
    let device = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CudaDevice::default))
        .map_err(|_| anyhow::anyhow!("CUDA device initialization failed"))?;
    let run_dir = prepare_run_dir(&config)?;
    let artifact_root = run_dir.as_ref().map(|path| path.join("artifacts"));
    if config.artifact_enabled && config.artifact_dir.is_none() {
        config.artifact_dir = artifact_root;
    }
    if let Some(run_dir) = run_dir.as_ref() {
        fs::write(
            run_dir.join("config.json"),
            serde_json::to_vec_pretty(&config).context("serialize dreamer config")?,
        )
        .with_context(|| format!("write run config {}", run_dir.join("config.json").display()))?;
    }
    let train_dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split: MovingMnistSplit::Train,
        frame_size: config.model.frame_size,
        digit_size: 12,
        in_channels: config.model.channels,
        context_len: config.context_len,
        target_len: config.target_len,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records: Some(256),
        normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
        min_velocity: config.min_velocity,
        max_velocity: config.max_velocity,
        seed: config.train_seed,
    })?;
    let valid_dataset = MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
        split: MovingMnistSplit::Val,
        frame_size: config.model.frame_size,
        digit_size: 12,
        in_channels: config.model.channels,
        context_len: config.context_len,
        target_len: config.target_len,
        extra_future_frames: 0,
        frame_stride: 1,
        max_records: Some(128),
        normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
        min_velocity: config.min_velocity,
        max_velocity: config.max_velocity,
        seed: config.val_seed,
    })?;
    let artifact_target_len = config.artifact_future_steps.max(config.target_len);
    let artifact_dataset =
        MovingMnistVideoDataset::new_from_mnist(MovingMnistVideoDatasetConfig {
            split: MovingMnistSplit::Val,
            frame_size: config.model.frame_size,
            digit_size: 12,
            in_channels: config.model.channels,
            context_len: config.context_len,
            target_len: artifact_target_len,
            extra_future_frames: 0,
            frame_stride: 1,
            max_records: Some(128),
            normalize: VisionNormalize::new([0.5; 3], [0.5; 3]),
            min_velocity: config.min_velocity,
            max_velocity: config.max_velocity,
            seed: config.val_seed,
        })?;

    let train_teacher = load_autogaze_source(&config, MovingMnistSplit::Train)?;
    let valid_teacher = load_autogaze_source(&config, MovingMnistSplit::Val)?;
    let global_teacher = load_vjepa_source(&mut config, &device)?;
    let crop_teacher = load_crop_teacher_source(&mut config, &device)?;
    let train_cache = CachedMovingMnistSplit::build(
        &train_dataset,
        &global_teacher,
        &crop_teacher,
        &train_teacher,
        &config,
        &device,
    );
    let valid_cache = CachedMovingMnistSplit::build(
        &valid_dataset,
        &global_teacher,
        &crop_teacher,
        &valid_teacher,
        &config,
        &device,
    );
    let mut model = DragonDreamer::<TrainBackend>::new(config.model.clone(), &device);
    if let Some(checkpoint) = config.tokenizer_checkpoint.as_ref() {
        if config.model.latent_backend.uses_slot_tokenizer() {
            let mut tokenizer_config = load_saved_tokenizer_config(checkpoint)?
                .map(|saved| saved.model)
                .unwrap_or_else(|| config.model.clone());
            tokenizer_config.latent_backend = DreamerLatentBackend::TransformerBaseline;
            tokenizer_config.use_bdh_posterior = false;
            let tokenizer_model = DragonDreamer::<TrainBackend>::new(tokenizer_config, &device);
            let tokenizer_model = load_module_checkpoint(tokenizer_model, checkpoint, &device)?;
            model.restore_tokenizer_from(&tokenizer_model);
        } else {
            model = load_module_checkpoint(model, checkpoint, &device)?;
        }
    }
    let frozen_tokenizer = if config.freeze_tokenizer {
        Some(model.clone())
    } else {
        None
    };
    let mut optimizer = AdamWConfig::new()
        .with_weight_decay(config.weight_decay)
        .init::<TrainBackend, DragonDreamer<TrainBackend>>();

    let initial_valid =
        evaluate_validation(&model, &valid_cache, &config, config.valid_batches, &device);
    let mut best_valid_total = initial_valid.total;
    let mut best_valid_future = initial_valid.future;
    let mut final_train_total = initial_valid.total;
    let mut final_train_future = initial_valid.future;
    let mut last_artifact_dir = None;
    let tokenizer_pretrain_steps = if config.tokenizer_checkpoint.is_some() {
        0
    } else {
        config.model.tokenizer_pretrain_steps
    };

    for step in 0..config.steps {
        let indices = sample_indices(
            train_cache.len(),
            config.batch_size,
            config
                .train_seed
                .wrapping_add((step as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)),
        );
        let batch = train_cache.batch(&indices);
        let forward = if config.model.latent_backend.uses_slot_tokenizer()
            && step < tokenizer_pretrain_steps
        {
            model.forward_tokenizer_pretrain(
                batch.clip_frames,
                &batch.traces,
                batch.teacher_features,
                batch.crop_teacher_features,
            )
        } else {
            model.forward(
                batch.clip_frames,
                &batch.traces,
                batch.teacher_features,
                batch.crop_teacher_features,
                config.context_len,
                config.target_len,
            )
        };
        let training_total = training_objective(&forward, &config, step, tokenizer_pretrain_steps);
        final_train_total = scalar(&training_total);
        final_train_future = scalar(&forward.future);
        let grads = GradientsParams::from_grads(training_total.backward(), &model);
        model = optimizer.step(config.learning_rate, model, grads);
        if let Some(frozen) = frozen_tokenizer.as_ref() {
            model.restore_tokenizer_from(frozen);
        }

        if (step + 1) % config.log_every.max(1) == 0 {
            println!(
                "step={} train_total={:.5} train_future={:.5}",
                step + 1,
                final_train_total,
                final_train_future
            );
        }

        if (step + 1) % config.validate_every.max(1) == 0 || step + 1 == config.steps {
            let valid =
                evaluate_validation(&model, &valid_cache, &config, config.valid_batches, &device);
            best_valid_total = best_valid_total.min(valid.total);
            best_valid_future = best_valid_future.min(valid.future);
            println!(
                "valid step={} total={:.5} current={:.5} future={:.5} prior={:.5} gaze={:.5} query={:.5} recon={:.5} tok={:.5} tok_recon={:.5} slot_align={:.5} recon_cur={:.5} recon_fut={:.5} edge={:.5} motion={:.5} fut_psnr={:.2} fut_iou={:.3} fut_motion_ratio={:.3} fut_stop_mean={:.3} fut_stop_std={:.3} fut_fix_motion={:.3} ctx_fix_l1={:.3} fut_fix_l1={:.3}",
                step + 1,
                valid.total,
                valid.current,
                valid.future,
                valid.prior,
                valid.gaze,
                valid.query,
                valid.recon,
                valid.tokenizer,
                valid.tokenizer_recon,
                valid.slot_align,
                valid.recon_current,
                valid.recon_future,
                valid.recon_edge,
                valid.recon_motion,
                valid.future_psnr,
                valid.future_fg_iou,
                valid.future_motion_ratio,
                valid.future_stop_mean,
                valid.future_stop_std,
                valid.future_fixation_motion,
                valid.context_fixation_teacher_l1,
                valid.future_fixation_teacher_l1,
            );
            if should_write_artifacts(&config, step + 1) {
                last_artifact_dir = Some(write_validation_artifacts(
                    &model,
                    &global_teacher,
                    &crop_teacher,
                    &valid_teacher,
                    &artifact_dataset,
                    &config,
                    artifact_target_len,
                    step + 1,
                    false,
                    &device,
                )?);
            }
        }
    }

    let final_valid =
        evaluate_validation(&model, &valid_cache, &config, config.valid_batches, &device);
    if config.artifact_enabled && config.artifact_dir.is_some() {
        last_artifact_dir = Some(write_validation_artifacts(
            &model,
            &global_teacher,
            &crop_teacher,
            &valid_teacher,
            &artifact_dataset,
            &config,
            artifact_target_len,
            config.steps.max(1),
            true,
            &device,
        )?);
    }
    Ok(MovingMnistDreamerRunSummary {
        initial_valid_total: initial_valid.total,
        final_valid_total: final_valid.total,
        initial_valid_future: initial_valid.future,
        final_valid_future: final_valid.future,
        best_valid_total,
        best_valid_future,
        final_train_total,
        final_train_future,
        final_valid_recon_current: final_valid.recon_current,
        final_valid_recon_future: final_valid.recon_future,
        final_valid_future_psnr: final_valid.future_psnr,
        final_valid_future_fg_iou: final_valid.future_fg_iou,
        final_valid_future_motion_ratio: final_valid.future_motion_ratio,
        final_valid_future_stop_mean: final_valid.future_stop_mean,
        final_valid_future_stop_std: final_valid.future_stop_std,
        final_valid_tokenizer: final_valid.tokenizer,
        final_valid_tokenizer_recon: final_valid.tokenizer_recon,
        final_valid_slot_align: final_valid.slot_align,
        latent_backend: config.model.latent_backend.as_str().to_string(),
        steps: config.steps,
        run_dir,
        artifact_dir: last_artifact_dir,
    })
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
    let crop = config.model.crop_size;
    let k = config.model.k_fovea.max(1);
    let half = crop / 2;
    let mut encoded = Vec::with_capacity(clip_len);
    for step in 0..clip_len {
        let frame = clip_frames
            .clone()
            .slice_dim(1, step..step + 1)
            .reshape([batch, channels, height, width]);
        let mut patches = Vec::with_capacity(batch * k);
        let mut weights = Vec::with_capacity(batch * k);
        for batch_idx in 0..batch {
            let sample = frame.clone().slice_dim(0, batch_idx..batch_idx + 1);
            let frame_trace = traces[batch_idx]
                .frames
                .get(step)
                .or_else(|| traces[batch_idx].frames.last())
                .expect("trace frame");
            for point in frame_trace.points.iter().take(k) {
                let center_x = (point.x * width as f32).round() as isize;
                let center_y = (point.y * height as f32).round() as isize;
                let x0 = (center_x - half as isize).clamp(0, width.saturating_sub(crop) as isize)
                    as usize;
                let y0 = (center_y - half as isize).clamp(0, height.saturating_sub(crop) as isize)
                    as usize;
                let patch = sample
                    .clone()
                    .slice_dim(2, y0..(y0 + crop).min(height))
                    .slice_dim(3, x0..(x0 + crop).min(width));
                let [_, _, patch_h, patch_w] = patch.shape().dims::<4>();
                let patch = if patch_h == crop && patch_w == crop {
                    patch
                } else {
                    let mut padded = burn::tensor::Tensor::<TrainBackend, 4>::zeros(
                        [1, channels, crop, crop],
                        &device,
                    );
                    padded =
                        padded.slice_assign([0..1, 0..channels, 0..patch_h, 0..patch_w], patch);
                    padded
                };
                patches.push(patch.unsqueeze_dim::<5>(1));
                weights.push(point.confidence.max(1.0e-4));
            }
        }
        let stacked = burn::tensor::Tensor::cat(patches, 0);
        let encoded_step = match teacher {
            ClipTeacherSource::Checkpoint(teacher) => teacher.encode_clip(stacked),
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

fn should_write_artifacts(config: &MovingMnistDreamerTrainConfig, step: usize) -> bool {
    config.artifact_enabled
        && config.artifact_dir.is_some()
        && config.artifact_every > 0
        && step % config.artifact_every.max(1) == 0
}

fn write_validation_artifacts(
    model: &DragonDreamer<TrainBackend>,
    global_teacher: &ClipTeacherSource,
    crop_teacher: &ClipTeacherSource,
    teacher: &AutoGazeSource,
    dataset: &MovingMnistVideoDataset,
    config: &MovingMnistDreamerTrainConfig,
    artifact_target_len: usize,
    step: usize,
    is_final: bool,
    device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> Result<PathBuf> {
    let effective_artifact_target_len = [
        global_teacher.target_len_limit(),
        crop_teacher.target_len_limit(),
    ]
    .into_iter()
    .flatten()
    .fold(artifact_target_len, |limit, teacher_limit| {
        limit.min(teacher_limit.max(1))
    })
    .max(1);
    let sample_count = config.artifact_samples.max(1).min(dataset.len().max(1));
    let indices: Vec<usize> = (0..sample_count).collect();
    let batch = dataset.batch_from_indices::<TrainBackend>(&indices, device);
    let traces = teacher.traces_for_batch(&indices, &batch, config.model.k_fovea);
    let teacher_features =
        global_teacher.encode_clip_from_batch(&indices, batch.clip_frames.clone().detach());
    let crop_teacher_features = encode_crop_teacher(
        crop_teacher,
        &indices,
        batch.clip_frames.clone().detach(),
        &traces,
        config,
    );
    let (forward, debug) = model.forward_with_debug(
        batch.clip_frames,
        &traces,
        teacher_features,
        crop_teacher_features,
        config.context_len,
        effective_artifact_target_len,
    );
    let teacher_visibility =
        teacher.visibility_for_batch(&indices, config.context_len + effective_artifact_target_len);
    let snapshot = build_artifact_snapshot(debug, teacher_visibility, config.model.crop_size)?;
    let metrics = artifact_metrics_from_forward_and_snapshot(
        &forward,
        &snapshot,
        config.model.latent_backend.as_str(),
    );
    let root = config
        .artifact_dir
        .as_ref()
        .expect("artifact dir should be present");
    let output_dir = if is_final {
        root.join("final")
    } else {
        root.join(format!("step_{step:06}"))
    };
    write_moving_mnist_artifacts(
        &output_dir,
        &snapshot,
        &metrics,
        config.model.latent_backend.as_str(),
    )
}

fn build_artifact_snapshot(
    debug: DreamerDebugOutput<TrainBackend>,
    teacher_visibility: Option<SequenceTensor>,
    crop_size: usize,
) -> Result<DreamerArtifactSnapshot> {
    Ok(DreamerArtifactSnapshot {
        current_reference: sequence_from_tensor(debug.context_reference_frames)?,
        current_reconstruction: sequence_from_tensor(debug.context_reconstruction_frames)?,
        future_reference: sequence_from_tensor(debug.future_reference_frames)?,
        future_reconstruction: sequence_from_tensor(debug.future_reconstruction_frames)?,
        context_latents: latent_from_tensor(debug.context_latents)?,
        future_latents: latent_from_tensor(debug.future_latents)?,
        teacher_visibility,
        teacher_fixations: fixations_from_tensor(debug.teacher_fixations)?,
        predicted_fixations: fixations_from_tensor(debug.predicted_fixations)?,
        crop_size,
    })
}

fn artifact_metrics_from_forward_and_snapshot(
    forward: &crate::DreamerForward<TrainBackend>,
    snapshot: &DreamerArtifactSnapshot,
    latent_backend: &str,
) -> ArtifactMetrics {
    let current_stats = sequence_reconstruction_metrics(
        &snapshot.current_reference,
        &snapshot.current_reconstruction,
    );
    let future_stats = sequence_reconstruction_metrics(
        &snapshot.future_reference,
        &snapshot.future_reconstruction,
    );
    let fixation_stats = future_fixation_metrics(
        &snapshot.predicted_fixations,
        snapshot.current_reference.steps,
        snapshot.future_reference.steps,
    );
    let fixation_alignment = fixation_teacher_alignment_metrics(
        &snapshot.teacher_fixations,
        &snapshot.predicted_fixations,
        snapshot.current_reference.steps,
        snapshot.future_reference.steps,
    );
    ArtifactMetrics {
        latent_backend: latent_backend.to_string(),
        total: scalar(&forward.total),
        current: scalar(&forward.current),
        future: scalar(&forward.future),
        prior: scalar(&forward.prior),
        gaze: scalar(&forward.gaze),
        query: scalar(&forward.query),
        recon: scalar(&forward.recon),
        tokenizer: scalar(&forward.tokenizer),
        tokenizer_recon: scalar(&forward.tokenizer_recon),
        slot_align: scalar(&forward.slot_align),
        recon_current: scalar(&forward.recon_current),
        recon_future: scalar(&forward.recon_future),
        recon_edge: scalar(&forward.recon_edge),
        recon_motion: scalar(&forward.recon_motion),
        current_mae: current_stats.mae,
        future_mae: future_stats.mae,
        current_psnr: current_stats.psnr,
        future_psnr: future_stats.psnr,
        current_fg_iou: current_stats.fg_iou,
        future_fg_iou: future_stats.fg_iou,
        future_frame_std: future_stats.frame_std,
        future_latent_std: standard_deviation(&snapshot.future_latents.data),
        future_motion_mse: future_stats.motion_mse,
        future_ref_motion_mse: future_stats.ref_motion_mse,
        future_motion_ratio: future_stats.motion_ratio,
        future_stop_mean: fixation_stats.stop_mean,
        future_stop_std: fixation_stats.stop_std,
        future_fixation_motion: fixation_stats.fixation_motion,
        future_confidence_mean: fixation_stats.confidence_mean,
        context_fixation_teacher_l1: fixation_alignment.context_l1,
        future_fixation_teacher_l1: fixation_alignment.future_l1,
    }
}

fn sequence_from_tensor(tensor: burn::tensor::Tensor<TrainBackend, 5>) -> Result<SequenceTensor> {
    let [batch, steps, channels, height, width] = tensor.shape().dims::<5>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("sequence tensor to host data")?;
    Ok(SequenceTensor {
        data,
        batch,
        steps,
        channels,
        height,
        width,
    })
}

fn latent_from_tensor(tensor: burn::tensor::Tensor<TrainBackend, 3>) -> Result<LatentTensor> {
    let [batch, steps, dim] = tensor.shape().dims::<3>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("latent tensor to host data")?;
    Ok(LatentTensor {
        data,
        batch,
        steps,
        dim,
    })
}

fn fixations_from_tensor(
    tensor: burn::tensor::Tensor<TrainBackend, 3>,
) -> Result<FixationSequence> {
    let [batch, steps, features] = tensor.shape().dims::<3>();
    let data = tensor
        .detach()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .context("fixation tensor to host data")?;
    let k = features.saturating_sub(1) / 4;
    let mut points = Vec::with_capacity(batch);
    let mut stop_probabilities = Vec::with_capacity(batch);
    for batch_idx in 0..batch {
        let mut per_step_points = Vec::with_capacity(steps);
        let mut per_step_stop = Vec::with_capacity(steps);
        for step_idx in 0..steps {
            let base = (batch_idx * steps + step_idx) * features;
            let mut per_fixation = Vec::with_capacity(k);
            for fixation_idx in 0..k {
                let offset = base + fixation_idx * 4;
                per_fixation.push(FixationPointArtifact {
                    x: data[offset].clamp(0.0, 1.0),
                    y: data[offset + 1].clamp(0.0, 1.0),
                    scale: data[offset + 2].clamp(0.01, 1.0),
                    confidence: data[offset + 3].clamp(0.0, 1.0),
                });
            }
            per_step_points.push(per_fixation);
            per_step_stop.push(data[base + k * 4].clamp(0.0, 1.0));
        }
        points.push(per_step_points);
        stop_probabilities.push(per_step_stop);
    }
    Ok(FixationSequence {
        points,
        stop_probabilities,
    })
}

#[derive(Clone, Debug, Default)]
struct LossSnapshot {
    total: f32,
    current: f32,
    future: f32,
    prior: f32,
    gaze: f32,
    query: f32,
    recon: f32,
    tokenizer: f32,
    tokenizer_recon: f32,
    slot_align: f32,
    recon_current: f32,
    recon_future: f32,
    recon_edge: f32,
    recon_motion: f32,
    current_mae: f32,
    future_mae: f32,
    current_psnr: f32,
    future_psnr: f32,
    current_fg_iou: f32,
    future_fg_iou: f32,
    future_frame_std: f32,
    future_latent_std: f32,
    future_motion_mse: f32,
    future_ref_motion_mse: f32,
    future_motion_ratio: f32,
    future_stop_mean: f32,
    future_stop_std: f32,
    future_fixation_motion: f32,
    future_confidence_mean: f32,
    context_fixation_teacher_l1: f32,
    future_fixation_teacher_l1: f32,
}

impl LossSnapshot {
    fn add_metrics(&mut self, metrics: &ArtifactMetrics) {
        self.total += metrics.total;
        self.current += metrics.current;
        self.future += metrics.future;
        self.prior += metrics.prior;
        self.gaze += metrics.gaze;
        self.query += metrics.query;
        self.recon += metrics.recon;
        self.tokenizer += metrics.tokenizer;
        self.tokenizer_recon += metrics.tokenizer_recon;
        self.slot_align += metrics.slot_align;
        self.recon_current += metrics.recon_current;
        self.recon_future += metrics.recon_future;
        self.recon_edge += metrics.recon_edge;
        self.recon_motion += metrics.recon_motion;
        self.current_mae += metrics.current_mae;
        self.future_mae += metrics.future_mae;
        self.current_psnr += metrics.current_psnr;
        self.future_psnr += metrics.future_psnr;
        self.current_fg_iou += metrics.current_fg_iou;
        self.future_fg_iou += metrics.future_fg_iou;
        self.future_frame_std += metrics.future_frame_std;
        self.future_latent_std += metrics.future_latent_std;
        self.future_motion_mse += metrics.future_motion_mse;
        self.future_ref_motion_mse += metrics.future_ref_motion_mse;
        self.future_motion_ratio += metrics.future_motion_ratio;
        self.future_stop_mean += metrics.future_stop_mean;
        self.future_stop_std += metrics.future_stop_std;
        self.future_fixation_motion += metrics.future_fixation_motion;
        self.future_confidence_mean += metrics.future_confidence_mean;
        self.context_fixation_teacher_l1 += metrics.context_fixation_teacher_l1;
        self.future_fixation_teacher_l1 += metrics.future_fixation_teacher_l1;
    }

    fn div_scalar(mut self, denom: f32) -> Self {
        self.total /= denom;
        self.current /= denom;
        self.future /= denom;
        self.prior /= denom;
        self.gaze /= denom;
        self.query /= denom;
        self.recon /= denom;
        self.tokenizer /= denom;
        self.tokenizer_recon /= denom;
        self.slot_align /= denom;
        self.recon_current /= denom;
        self.recon_future /= denom;
        self.recon_edge /= denom;
        self.recon_motion /= denom;
        self.current_mae /= denom;
        self.future_mae /= denom;
        self.current_psnr /= denom;
        self.future_psnr /= denom;
        self.current_fg_iou /= denom;
        self.future_fg_iou /= denom;
        self.future_frame_std /= denom;
        self.future_latent_std /= denom;
        self.future_motion_mse /= denom;
        self.future_ref_motion_mse /= denom;
        self.future_motion_ratio /= denom;
        self.future_stop_mean /= denom;
        self.future_stop_std /= denom;
        self.future_fixation_motion /= denom;
        self.future_confidence_mean /= denom;
        self.context_fixation_teacher_l1 /= denom;
        self.future_fixation_teacher_l1 /= denom;
        self
    }
}

#[derive(Clone, Debug, Default)]
struct SequenceReconstructionMetrics {
    mae: f32,
    psnr: f32,
    fg_iou: f32,
    frame_std: f32,
    motion_mse: f32,
    ref_motion_mse: f32,
    motion_ratio: f32,
}

#[derive(Clone, Debug, Default)]
struct FutureFixationMetrics {
    stop_mean: f32,
    stop_std: f32,
    fixation_motion: f32,
    confidence_mean: f32,
}

#[derive(Clone, Debug, Default)]
struct FixationAlignmentMetrics {
    context_l1: f32,
    future_l1: f32,
}

fn sequence_reconstruction_metrics(
    reference: &SequenceTensor,
    reconstruction: &SequenceTensor,
) -> SequenceReconstructionMetrics {
    let batch = reference.batch.min(reconstruction.batch);
    let steps = reference.steps.min(reconstruction.steps);
    let channels = reference.channels.min(reconstruction.channels);
    let height = reference.height.min(reconstruction.height);
    let width = reference.width.min(reconstruction.width);
    if batch == 0 || steps == 0 || channels == 0 || height == 0 || width == 0 {
        return SequenceReconstructionMetrics::default();
    }

    let mut abs_sum = 0.0f64;
    let mut sq_sum = 0.0f64;
    let mut pixel_count = 0usize;
    let mut intersection = 0usize;
    let mut union = 0usize;
    let mut pred_sum = 0.0f64;
    let mut pred_sq_sum = 0.0f64;
    let mut pred_motion_sq_sum = 0.0f64;
    let mut ref_motion_sq_sum = 0.0f64;
    let mut motion_count = 0usize;

    for batch_idx in 0..batch {
        for step_idx in 0..steps {
            for channel_idx in 0..channels {
                for y in 0..height {
                    for x in 0..width {
                        let ref_value = unit_interval(sequence_value(
                            reference,
                            batch_idx,
                            step_idx,
                            channel_idx,
                            y,
                            x,
                        ));
                        let pred_value = unit_interval(sequence_value(
                            reconstruction,
                            batch_idx,
                            step_idx,
                            channel_idx,
                            y,
                            x,
                        ));
                        let diff = pred_value - ref_value;
                        abs_sum += diff.abs() as f64;
                        sq_sum += (diff * diff) as f64;
                        pred_sum += pred_value as f64;
                        pred_sq_sum += (pred_value * pred_value) as f64;
                        pixel_count += 1;

                        let ref_fg = ref_value > 0.2;
                        let pred_fg = pred_value > 0.2;
                        if ref_fg && pred_fg {
                            intersection += 1;
                        }
                        if ref_fg || pred_fg {
                            union += 1;
                        }

                        if step_idx > 0 {
                            let prev_ref = unit_interval(sequence_value(
                                reference,
                                batch_idx,
                                step_idx - 1,
                                channel_idx,
                                y,
                                x,
                            ));
                            let prev_pred = unit_interval(sequence_value(
                                reconstruction,
                                batch_idx,
                                step_idx - 1,
                                channel_idx,
                                y,
                                x,
                            ));
                            let pred_delta = pred_value - prev_pred;
                            let ref_delta = ref_value - prev_ref;
                            pred_motion_sq_sum += (pred_delta * pred_delta) as f64;
                            ref_motion_sq_sum += (ref_delta * ref_delta) as f64;
                            motion_count += 1;
                        }
                    }
                }
            }
        }
    }

    let pixel_denom = pixel_count.max(1) as f64;
    let mse = (sq_sum / pixel_denom) as f32;
    let pred_mean = pred_sum / pixel_denom;
    let pred_var = (pred_sq_sum / pixel_denom) - pred_mean * pred_mean;
    let motion_denom = motion_count.max(1) as f64;
    let motion_mse = (pred_motion_sq_sum / motion_denom) as f32;
    let ref_motion_mse = (ref_motion_sq_sum / motion_denom) as f32;
    SequenceReconstructionMetrics {
        mae: (abs_sum / pixel_denom) as f32,
        psnr: 10.0 * (1.0f32 / mse.max(1.0e-8)).log10(),
        fg_iou: if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        },
        frame_std: pred_var.max(0.0).sqrt() as f32,
        motion_mse,
        ref_motion_mse,
        motion_ratio: motion_mse / ref_motion_mse.max(1.0e-8),
    }
}

fn future_fixation_metrics(
    fixations: &FixationSequence,
    context_steps: usize,
    future_steps: usize,
) -> FutureFixationMetrics {
    let end = context_steps + future_steps;
    let mut stop_values = Vec::new();
    let mut confidence_sum = 0.0f32;
    let mut confidence_count = 0usize;
    let mut motion_sum = 0.0f32;
    let mut motion_count = 0usize;

    for (batch_idx, steps) in fixations.points.iter().enumerate() {
        let stop_steps = fixations
            .stop_probabilities
            .get(batch_idx)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        for step_idx in context_steps..end.min(steps.len()) {
            if let Some(stop) = stop_steps.get(step_idx) {
                stop_values.push(*stop);
            }
            for point in &steps[step_idx] {
                confidence_sum += point.confidence;
                confidence_count += 1;
            }
            if step_idx > 0 {
                let previous = &steps[step_idx - 1];
                let current = &steps[step_idx];
                for fixation_idx in 0..previous.len().min(current.len()) {
                    let dx = current[fixation_idx].x - previous[fixation_idx].x;
                    let dy = current[fixation_idx].y - previous[fixation_idx].y;
                    motion_sum += (dx * dx + dy * dy).sqrt();
                    motion_count += 1;
                }
            }
        }
    }

    let stop_mean = if stop_values.is_empty() {
        0.0
    } else {
        stop_values.iter().sum::<f32>() / stop_values.len() as f32
    };
    let stop_std = if stop_values.is_empty() {
        0.0
    } else {
        let mean = stop_mean;
        let variance = stop_values
            .iter()
            .map(|value| {
                let diff = *value - mean;
                diff * diff
            })
            .sum::<f32>()
            / stop_values.len() as f32;
        variance.max(0.0).sqrt()
    };

    FutureFixationMetrics {
        stop_mean,
        stop_std,
        fixation_motion: if motion_count == 0 {
            0.0
        } else {
            motion_sum / motion_count as f32
        },
        confidence_mean: if confidence_count == 0 {
            0.0
        } else {
            confidence_sum / confidence_count as f32
        },
    }
}

fn fixation_teacher_alignment_metrics(
    teacher: &FixationSequence,
    predicted: &FixationSequence,
    context_steps: usize,
    future_steps: usize,
) -> FixationAlignmentMetrics {
    let mut context_sum = 0.0f32;
    let mut context_count = 0usize;
    let mut future_sum = 0.0f32;
    let mut future_count = 0usize;
    let end = context_steps + future_steps;
    let batch = teacher.points.len().min(predicted.points.len());
    for batch_idx in 0..batch {
        let teacher_steps = &teacher.points[batch_idx];
        let predicted_steps = &predicted.points[batch_idx];
        let total_steps = teacher_steps.len().min(predicted_steps.len()).min(end);
        for step_idx in 0..total_steps {
            let teacher_points = &teacher_steps[step_idx];
            let predicted_points = &predicted_steps[step_idx];
            for point_idx in 0..teacher_points.len().min(predicted_points.len()) {
                let teacher_point = &teacher_points[point_idx];
                let predicted_point = &predicted_points[point_idx];
                let l1 = (teacher_point.x - predicted_point.x).abs()
                    + (teacher_point.y - predicted_point.y).abs()
                    + (teacher_point.scale - predicted_point.scale).abs()
                    + (teacher_point.confidence - predicted_point.confidence).abs();
                if step_idx < context_steps {
                    context_sum += l1;
                    context_count += 1;
                } else {
                    future_sum += l1;
                    future_count += 1;
                }
            }
        }
    }
    FixationAlignmentMetrics {
        context_l1: if context_count == 0 {
            0.0
        } else {
            context_sum / context_count as f32
        },
        future_l1: if future_count == 0 {
            0.0
        } else {
            future_sum / future_count as f32
        },
    }
}

fn unit_interval(value: f32) -> f32 {
    (value * 0.5 + 0.5).clamp(0.0, 1.0)
}

fn sequence_value(
    sequence: &SequenceTensor,
    batch_idx: usize,
    step_idx: usize,
    channel_idx: usize,
    y: usize,
    x: usize,
) -> f32 {
    let index = ((((batch_idx * sequence.steps + step_idx) * sequence.channels + channel_idx)
        * sequence.height
        + y)
        * sequence.width)
        + x;
    sequence.data[index]
}

fn standard_deviation(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().copied().sum::<f32>() / values.len() as f32;
    let variance = values
        .iter()
        .map(|value| {
            let diff = *value - mean;
            diff * diff
        })
        .sum::<f32>()
        / values.len() as f32;
    variance.max(0.0).sqrt()
}

fn evaluate_validation(
    model: &DragonDreamer<TrainBackend>,
    dataset: &CachedMovingMnistSplit,
    config: &MovingMnistDreamerTrainConfig,
    batches: usize,
    _device: &<TrainBackend as burn::tensor::backend::Backend>::Device,
) -> LossSnapshot {
    let mut totals = LossSnapshot::default();
    let count = batches.max(1);
    for batch_idx in 0..count {
        let indices = sample_indices(
            dataset.len(),
            config.batch_size,
            config
                .val_seed
                .wrapping_add((batch_idx as u64).wrapping_mul(0x517C_C1B7_2722_0A95)),
        );
        let batch = dataset.batch(&indices);
        let (forward, debug) = model.forward_with_debug(
            batch.clip_frames,
            &batch.traces,
            batch.teacher_features,
            batch.crop_teacher_features,
            config.context_len,
            config.target_len,
        );
        let snapshot = build_artifact_snapshot(debug, None, config.model.crop_size)
            .expect("validation debug snapshot");
        let metrics = artifact_metrics_from_forward_and_snapshot(
            &forward,
            &snapshot,
            config.model.latent_backend.as_str(),
        );
        totals.add_metrics(&metrics);
    }
    totals.div_scalar(count as f32)
}

pub(crate) fn scalar(tensor: &burn::tensor::Tensor<TrainBackend, 1>) -> f32 {
    tensor
        .clone()
        .into_data()
        .to_vec::<f32>()
        .expect("scalar tensor")[0]
}

fn training_objective(
    forward: &crate::DreamerForward<TrainBackend>,
    config: &MovingMnistDreamerTrainConfig,
    step: usize,
    tokenizer_pretrain_steps: usize,
) -> burn::tensor::Tensor<TrainBackend, 1> {
    if !config.model.latent_backend.uses_slot_tokenizer() {
        return forward.total.clone();
    }

    if step < tokenizer_pretrain_steps {
        return forward
            .current
            .clone()
            .mul_scalar(config.model.current_loss_weight * 0.25)
            + forward
                .query
                .clone()
                .mul_scalar(config.model.query_loss_weight * 0.5)
            + forward.tokenizer.clone().mul_scalar(
                config.model.tokenizer_loss_weight * config.model.tokenizer_scale_start,
            )
            + forward
                .recon
                .clone()
                .mul_scalar(config.model.recon_loss_weight * config.model.recon_scale_start)
            + forward
                .gaze
                .clone()
                .mul_scalar(config.model.gaze_loss_weight * 0.5 * config.model.gaze_scale_start);
    }

    let total_steps = config.steps.max(1);
    let warmup_steps = ((total_steps as f32 * config.model.warmup_fraction.clamp(0.0, 1.0)).round()
        as usize)
        .max(12)
        .min(total_steps);
    let warmup_progress = if step + 1 >= warmup_steps {
        1.0
    } else {
        (step + 1) as f32 / warmup_steps as f32
    };
    let tokenizer_scale = config.model.tokenizer_scale_start
        + (config.model.tokenizer_scale_end - config.model.tokenizer_scale_start) * warmup_progress;
    let dynamics_scale = config.model.dynamics_scale_start
        + (config.model.dynamics_scale_end - config.model.dynamics_scale_start) * warmup_progress;
    let recon_scale = config.model.recon_scale_start
        + (config.model.recon_scale_end - config.model.recon_scale_start) * warmup_progress;
    let gaze_scale = config.model.gaze_scale_start
        + (config.model.gaze_scale_end - config.model.gaze_scale_start) * warmup_progress;

    forward
        .current
        .clone()
        .mul_scalar(config.model.current_loss_weight * dynamics_scale)
        + forward
            .future
            .clone()
            .mul_scalar(config.model.future_loss_weight * dynamics_scale)
        + forward
            .prior
            .clone()
            .mul_scalar(config.model.prior_loss_weight * dynamics_scale)
        + forward
            .gaze
            .clone()
            .mul_scalar(config.model.gaze_loss_weight * gaze_scale)
        + forward
            .query
            .clone()
            .mul_scalar(config.model.query_loss_weight * dynamics_scale)
        + forward
            .tokenizer
            .clone()
            .mul_scalar(config.model.tokenizer_loss_weight * tokenizer_scale)
        + forward
            .recon
            .clone()
            .mul_scalar(config.model.recon_loss_weight * recon_scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::tensor::{Dtype, View, serialize_to_file};
    use std::borrow::Cow;
    use tempfile::tempdir;

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

    fn write_trace_store(
        path: &std::path::Path,
        clips: usize,
        clip_len: usize,
        k: usize,
        frame_size: usize,
    ) {
        let mut fixations = Vec::with_capacity(clips * clip_len * k * 2);
        let mut scales = Vec::with_capacity(clips * clip_len * k);
        let mut confidences = Vec::with_capacity(clips * clip_len * k);
        let mut stops = Vec::with_capacity(clips * clip_len);
        let mut visibility = vec![0.0f32; clips * clip_len * frame_size * frame_size];

        for clip_idx in 0..clips {
            for frame_idx in 0..clip_len {
                let x = ((frame_idx + 1) as f32 / (clip_len + 1) as f32).clamp(0.15, 0.85);
                let y = ((clip_idx % 7 + frame_idx + 1) as f32 / (clip_len + 7) as f32)
                    .clamp(0.15, 0.85);
                for _ in 0..k {
                    fixations.extend_from_slice(&[x, y]);
                    scales.push(0.22);
                    confidences.push(0.9);
                }
                stops.push(0.1);

                let cx = (x * frame_size as f32).round() as isize;
                let cy = (y * frame_size as f32).round() as isize;
                let base = (clip_idx * clip_len + frame_idx) * frame_size * frame_size;
                for yy in 0..frame_size {
                    for xx in 0..frame_size {
                        let dx = xx as isize - cx;
                        let dy = yy as isize - cy;
                        if dx.abs() <= 3 && dy.abs() <= 3 {
                            visibility[base + yy * frame_size + xx] = 1.0;
                        }
                    }
                }
            }
        }

        let tensors = vec![
            (
                "fixations".to_string(),
                tensor_f32(&[clips, clip_len, k, 2], &fixations),
            ),
            (
                "scales".to_string(),
                tensor_f32(&[clips, clip_len, k], &scales),
            ),
            (
                "confidences".to_string(),
                tensor_f32(&[clips, clip_len, k], &confidences),
            ),
            (
                "stop_probabilities".to_string(),
                tensor_f32(&[clips, clip_len], &stops),
            ),
            (
                "visibility_maps".to_string(),
                tensor_f32(&[clips, clip_len, frame_size, frame_size], &visibility),
            ),
        ];
        serialize_to_file(tensors, None, path).expect("write trace store");
    }

    fn write_feature_store(
        path: &std::path::Path,
        clips: usize,
        context_len: usize,
        target_len: usize,
        feature_dim: usize,
        scale: f32,
    ) {
        let mut current = Vec::with_capacity(clips * context_len * feature_dim);
        let mut future = Vec::with_capacity(clips * target_len * feature_dim);
        for clip_idx in 0..clips {
            for frame_idx in 0..context_len {
                for feat_idx in 0..feature_dim {
                    let value =
                        (((clip_idx + 1) * (frame_idx + 1) * (feat_idx + 3)) as f32 * scale).sin();
                    current.push(value);
                }
            }
            for frame_idx in 0..target_len {
                for feat_idx in 0..feature_dim {
                    let value = (((clip_idx + 2) * (frame_idx + context_len + 1) * (feat_idx + 5))
                        as f32
                        * scale)
                        .cos();
                    future.push(value);
                }
            }
        }
        let tensors = vec![
            (
                "current_features".to_string(),
                tensor_f32(&[clips, context_len, feature_dim], &current),
            ),
            (
                "future_features".to_string(),
                tensor_f32(&[clips, target_len, feature_dim], &future),
            ),
        ];
        serialize_to_file(tensors, None, path).expect("write feature store");
    }

    fn strict_teacher_config(k_fovea: usize) -> (tempfile::TempDir, MovingMnistDreamerTrainConfig) {
        let temp = tempdir().expect("temp dir");
        let train_traces = temp.path().join("train_autogaze.safetensors");
        let val_traces = temp.path().join("val_autogaze.safetensors");
        let global_features = temp.path().join("global_vjepa.safetensors");
        let crop_features = temp.path().join("crop_teacher.safetensors");

        let mut config = MovingMnistDreamerTrainConfig {
            steps: 8,
            batch_size: 2,
            validate_every: 4,
            valid_batches: 1,
            run_root: None,
            artifact_enabled: false,
            ..Default::default()
        };
        config.model.k_fovea = k_fovea.max(1);
        config.artifact_future_steps = config.target_len;
        write_trace_store(
            &train_traces,
            256,
            config.context_len + config.target_len,
            config.model.k_fovea.max(1),
            config.model.frame_size,
        );
        write_trace_store(
            &val_traces,
            128,
            config.context_len + config.target_len,
            config.model.k_fovea.max(1),
            config.model.frame_size,
        );
        write_feature_store(
            &global_features,
            256,
            config.context_len,
            config.target_len,
            config.model.teacher_dim,
            0.007,
        );
        write_feature_store(
            &crop_features,
            256,
            config.context_len,
            config.target_len,
            config.model.crop_teacher_dim,
            0.011,
        );
        config.autogaze_train_trace_store = Some(train_traces);
        config.autogaze_val_trace_store = Some(val_traces);
        config.vjepa_feature_store = Some(global_features);
        config.crop_teacher_feature_store = Some(crop_features);
        (temp, config)
    }

    #[test]
    fn strict_teacher_loaders_require_assets() {
        let config = MovingMnistDreamerTrainConfig::default();
        let err =
            load_autogaze_source(&config, MovingMnistSplit::Train).expect_err("missing store");
        assert!(err.to_string().contains("missing AutoGaze teacher traces"));
    }

    #[test]
    fn moving_mnist_training_smoke_improves_validation() {
        let (_temp, mut config) = strict_teacher_config(1);
        config.steps = 20;
        config.batch_size = 4;
        config.validate_every = 10;
        config.valid_batches = 2;
        let summary = train_moving_mnist(config).expect("training");
        assert!(summary.final_valid_total.is_finite());
        assert!(summary.final_valid_future.is_finite());
        assert!(
            summary.best_valid_total <= summary.initial_valid_total,
            "expected some validation improvement"
        );
    }

    #[test]
    fn moving_mnist_multifovea_smoke_is_finite() {
        let (_temp, config) = strict_teacher_config(2);
        let summary = train_moving_mnist(config).expect("training");
        assert!(summary.final_valid_total.is_finite());
        assert!(summary.final_valid_future.is_finite());
    }

    #[test]
    fn moving_mnist_writes_artifact_snapshots() {
        let (temp, mut config) = strict_teacher_config(1);
        config.steps = 2;
        config.batch_size = 2;
        config.validate_every = 1;
        config.valid_batches = 1;
        config.artifact_enabled = true;
        config.artifact_dir = Some(temp.path().join("dreamer_artifacts"));
        config.artifact_every = 1;
        config.artifact_samples = 1;
        let summary = train_moving_mnist(config).expect("training");
        let artifact_dir = summary.artifact_dir.expect("artifact dir");
        for name in [
            "current_reference.png",
            "current_reconstruction.png",
            "future_reference.png",
            "future_reconstruction.png",
            "current_latent_pca.png",
            "future_latent_pca.png",
            "autogaze_teacher_label_patches.png",
            "fovea_saccade_reads.png",
            "fixation_overlays.png",
            "metrics.json",
        ] {
            assert!(
                artifact_dir.join(name).is_file(),
                "expected artifact {}",
                artifact_dir.join(name).display()
            );
        }
    }
}
