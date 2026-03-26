use crate::train::prelude::*;
use burn::record::Recorder;
use burn_dragon_train::train::metrics::ScalarValue;
use burn_train::metric::{Adaptor, ItemLazy};

use super::models::{DistillTeacherModel, VisionDistillModel};

type PreparedDistillData<B> = (
    Arc<ImageNetDataset>,
    Arc<ImageNetDataset>,
    Option<DistillTeacherModel<B>>,
);

pub(super) struct DistillDatasetRequest<'a, B: BackendTrait> {
    pub config: &'a VisionTrainingConfig,
    pub vision_config: &'a VisionDragonConfig,
    pub normalize: VisionNormalize,
    pub train_aug: ImageNetAugmentations,
    pub val_aug: ImageNetAugmentations,
    pub train_root: &'a Path,
    pub val_root: &'a Path,
    pub student_patch_tokens: usize,
    pub device: &'a B::Device,
}

#[derive(Debug, Clone, Serialize)]
pub struct VisionDistillCheckpointEvalSummary {
    pub backend: String,
    pub checkpoint: PathBuf,
    pub batches: usize,
    pub distill_total: f64,
    pub distill_patch: f64,
    pub distill_cls: f64,
    pub horizons: [usize; VISION_ROLLOUT_HORIZON_COUNT],
    pub distill_total_to_horizon: [f64; VISION_ROLLOUT_HORIZON_COUNT],
    pub distill_patch_to_horizon: [f64; VISION_ROLLOUT_HORIZON_COUNT],
    pub distill_cls_to_horizon: [f64; VISION_ROLLOUT_HORIZON_COUNT],
}

fn required_teacher_patch_path<'a>(path: &'a Option<PathBuf>, field: &str) -> Result<&'a Path> {
    path.as_deref()
        .ok_or_else(|| anyhow!("{field} is required for patch-and-cls teacher targets"))
}

fn teacher_target_patch_tokens(
    target: &VisionTeacherTargetConfig,
    teacher: &VisionTeacherFeatureConfig,
    student_patch_tokens: usize,
) -> Result<Option<usize>> {
    match target.target_kind {
        VisionTeacherTargetKind::PatchAndCls => match target.decoder_mode {
            VisionTeacherDecoderMode::SharedProjection
            | VisionTeacherDecoderMode::DedicatedProjection => {
                if let Some(tokens) = teacher
                    .patch_tokens
                    .filter(|tokens| *tokens != student_patch_tokens)
                {
                    return Err(anyhow!(
                        "teacher target `{}` patch_tokens ({}) must match student tokens ({}) for {:?} patch-and-cls supervision",
                        target.name,
                        tokens,
                        student_patch_tokens,
                        target.decoder_mode
                    ));
                }
                Ok(Some(teacher.patch_tokens.unwrap_or(student_patch_tokens)))
            }
            VisionTeacherDecoderMode::DedicatedSpatialProjection => {
                let patch_tokens = teacher.patch_tokens.ok_or_else(|| {
                        anyhow!(
                            "teacher target `{}` requires teacher.patch_tokens for dedicated spatial patch-and-cls supervision",
                            target.name
                        )
                    })?;
                Ok(Some(patch_tokens))
            }
        },
        VisionTeacherTargetKind::ClsOnly | VisionTeacherTargetKind::GlobalOnly => Ok(None),
    }
}

fn build_auxiliary_teacher_targets(
    targets: &[VisionTeacherTargetConfig],
    train_records: usize,
    val_records: usize,
    student_patch_tokens: usize,
    cache_in_memory: bool,
) -> Result<(Vec<ImageTeacherTargetStore>, Vec<ImageTeacherTargetStore>)> {
    let mut train_targets = Vec::with_capacity(targets.len());
    let mut val_targets = Vec::with_capacity(targets.len());
    for target in targets {
        let teacher = match &target.teacher {
            VisionTeacherConfig::Features(teacher) => teacher,
            VisionTeacherConfig::Model(_) => {
                return Err(anyhow!(
                    "auxiliary teacher target `{}` currently requires precomputed feature files",
                    target.name
                ));
            }
        };
        let patch_tokens = teacher_target_patch_tokens(target, teacher, student_patch_tokens)?;
        let train_store = Arc::new(DinoFeatureStore::new_optional_patch_with_options(
            &teacher.train_cls_path,
            match target.target_kind {
                VisionTeacherTargetKind::PatchAndCls => Some(required_teacher_patch_path(
                    &teacher.train_patch_path,
                    &format!(
                        "mode.teacher_targets[name={}].teacher.train_patch_path",
                        target.name
                    ),
                )?),
                VisionTeacherTargetKind::ClsOnly | VisionTeacherTargetKind::GlobalOnly => None,
            },
            teacher.feature_dim,
            patch_tokens,
            Some(train_records),
            cache_in_memory,
        )?);
        train_targets.push(ImageTeacherTargetStore {
            name: target.name.clone(),
            weight: target.weight,
            target_kind: target.target_kind,
            store: train_store,
        });
        let val_store = Arc::new(DinoFeatureStore::new_optional_patch_with_options(
            &teacher.val_cls_path,
            match target.target_kind {
                VisionTeacherTargetKind::PatchAndCls => Some(required_teacher_patch_path(
                    &teacher.val_patch_path,
                    &format!(
                        "mode.teacher_targets[name={}].teacher.val_patch_path",
                        target.name
                    ),
                )?),
                VisionTeacherTargetKind::ClsOnly | VisionTeacherTargetKind::GlobalOnly => None,
            },
            teacher.feature_dim,
            patch_tokens,
            Some(val_records),
            cache_in_memory,
        )?);
        val_targets.push(ImageTeacherTargetStore {
            name: target.name.clone(),
            weight: target.weight,
            target_kind: target.target_kind,
            store: val_store,
        });
    }
    Ok((train_targets, val_targets))
}

pub(super) fn build_distill_datasets_and_teacher<B: BackendTrait>(
    request: DistillDatasetRequest<'_, B>,
) -> Result<PreparedDistillData<B>> {
    let DistillDatasetRequest {
        config,
        vision_config,
        normalize,
        train_aug,
        val_aug,
        train_root,
        val_root,
        student_patch_tokens,
        device,
    } = request;
    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill,
        other => {
            return Err(anyhow!(
                "distill dataset preparation requires distill mode, got {:?}",
                std::mem::discriminant(other)
            ));
        }
    };

    match &distill.teacher {
        VisionTeacherConfig::Features(teacher) => {
            if teacher.feature_dim != vision_config.projection_dim {
                return Err(anyhow!(
                    "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                    teacher.feature_dim,
                    vision_config.projection_dim
                ));
            }
            if let Some(tokens) = teacher
                .patch_tokens
                .filter(|tokens| *tokens != student_patch_tokens)
            {
                return Err(anyhow!(
                    "teacher.patch_tokens ({}) must match ceil(image_size/patch_size)^2 ({})",
                    tokens,
                    student_patch_tokens
                ));
            }
            let teacher_tokens = teacher.patch_tokens.unwrap_or(student_patch_tokens);
            let train_patch_path = required_teacher_patch_path(
                &teacher.train_patch_path,
                "mode.teacher.train_patch_path",
            )?;
            let val_patch_path = required_teacher_patch_path(
                &teacher.val_patch_path,
                "mode.teacher.val_patch_path",
            )?;

            let mut train_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                root: train_root.to_path_buf(),
                split: ImageNetSplit::Train,
                max_records: config.dataset.max_records,
                augmentations: train_aug,
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views: 1,
                local_views: 0,
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?;
            let train_records = train_dataset.len();
            let train_teacher = Arc::new(DinoFeatureStore::new_with_options(
                &teacher.train_cls_path,
                train_patch_path,
                teacher.feature_dim,
                teacher_tokens,
                Some(train_records),
                config.dataset.cache_teacher_features_in_memory,
            )?);
            let mut val_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                root: val_root.to_path_buf(),
                split: ImageNetSplit::Val,
                max_records: config.dataset.max_records,
                augmentations: val_aug,
                local_augmentations: None,
                normalize,
                teacher: None,
                rac_teacher_latent: None,
                teacher_targets: Vec::new(),
                views: 1,
                local_views: 0,
                min_view_overlap: 0.0,
                view_overlap_attempts: 1,
                cache_decoded: config.dataset.cache_decoded,
                cache_capacity: config.dataset.cache_capacity,
                cache_preprocessed: config.dataset.cache_preprocessed,
            })?;
            let val_records = val_dataset.len();
            let (train_targets, val_targets) = build_auxiliary_teacher_targets(
                distill.auxiliary_teacher_targets(),
                train_records,
                val_records,
                student_patch_tokens,
                config.dataset.cache_teacher_features_in_memory,
            )?;
            train_dataset = train_dataset
                .with_teacher(Arc::clone(&train_teacher))
                .with_teacher_targets(train_targets);
            let train_dataset = Arc::new(train_dataset);
            let val_teacher = Arc::new(DinoFeatureStore::new_with_options(
                &teacher.val_cls_path,
                val_patch_path,
                teacher.feature_dim,
                teacher_tokens,
                Some(val_records),
                config.dataset.cache_teacher_features_in_memory,
            )?);
            val_dataset = val_dataset
                .with_teacher(Arc::clone(&val_teacher))
                .with_teacher_targets(val_targets);
            Ok((train_dataset, Arc::new(val_dataset), None))
        }
        VisionTeacherConfig::Model(teacher) => {
            #[cfg(not(feature = "burn_dino"))]
            {
                let _ = teacher;
                let _ = device;
                Err(anyhow!(
                    "VisionTeacherConfig::Model requires the optional `burn_dino` feature; precomputed teacher features remain supported on Burn 0.21"
                ))
            }

            #[cfg(feature = "burn_dino")]
            {
                let image_size = teacher.image_size.unwrap_or(vision_config.image_size);
                let patch_size = teacher.patch_size.unwrap_or(vision_config.patch_size);
                if patch_size == 0 {
                    return Err(anyhow!("teacher.patch_size must be > 0"));
                }
                if image_size % patch_size != 0 {
                    return Err(anyhow!(
                        "teacher image_size must be divisible by patch_size ({} % {} != 0)",
                        image_size,
                        patch_size
                    ));
                }
                let teacher_grid = image_size.div_ceil(patch_size);
                let teacher_tokens = teacher_grid * teacher_grid;
                if teacher_tokens != student_patch_tokens {
                    return Err(anyhow!(
                        "teacher patch tokens ({}) must match student tokens ({})",
                        teacher_tokens,
                        student_patch_tokens
                    ));
                }
                if let Some(tokens) = teacher
                    .patch_tokens
                    .filter(|tokens| *tokens != teacher_tokens)
                {
                    return Err(anyhow!(
                        "teacher.patch_tokens ({}) must match ceil(image_size/patch_size)^2 ({})",
                        tokens,
                        teacher_tokens
                    ));
                }

                let feature_dim = teacher
                    .feature_dim
                    .unwrap_or_else(|| teacher_variant_dim(teacher.variant));
                if feature_dim != vision_config.projection_dim {
                    return Err(anyhow!(
                        "teacher.feature_dim ({}) must match vision.projection_dim ({})",
                        feature_dim,
                        vision_config.projection_dim
                    ));
                }

                let mut dino_config = build_dino_config(teacher.variant, image_size, patch_size);
                if teacher.register_tokens > 0 {
                    dino_config = dino_config.with_register_tokens(teacher.register_tokens);
                }
                if dino_config.embedding_dimension != feature_dim {
                    return Err(anyhow!(
                        "teacher.feature_dim ({}) must match DINO embedding dim ({})",
                        feature_dim,
                        dino_config.embedding_dimension
                    ));
                }

                let teacher_model =
                    load_model_from_checkpoint::<B>(&dino_config, &teacher.checkpoint_path, device)
                        .map_err(|err| {
                            anyhow!(
                                "failed to load teacher checkpoint {}: {err}",
                                teacher.checkpoint_path.display()
                            )
                        })?;

                let mut train_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                    root: train_root.to_path_buf(),
                    split: ImageNetSplit::Train,
                    max_records: config.dataset.max_records,
                    augmentations: train_aug,
                    local_augmentations: None,
                    normalize,
                    teacher: None,
                    rac_teacher_latent: None,
                    teacher_targets: Vec::new(),
                    views: 1,
                    local_views: 0,
                    min_view_overlap: 0.0,
                    view_overlap_attempts: 1,
                    cache_decoded: config.dataset.cache_decoded,
                    cache_capacity: config.dataset.cache_capacity,
                    cache_preprocessed: config.dataset.cache_preprocessed,
                })?;
                let mut val_dataset = ImageNetDataset::new(ImageNetDatasetConfig {
                    root: val_root.to_path_buf(),
                    split: ImageNetSplit::Val,
                    max_records: config.dataset.max_records,
                    augmentations: val_aug,
                    local_augmentations: None,
                    normalize,
                    teacher: None,
                    rac_teacher_latent: None,
                    teacher_targets: Vec::new(),
                    views: 1,
                    local_views: 0,
                    min_view_overlap: 0.0,
                    view_overlap_attempts: 1,
                    cache_decoded: config.dataset.cache_decoded,
                    cache_capacity: config.dataset.cache_capacity,
                    cache_preprocessed: config.dataset.cache_preprocessed,
                })?;
                let (train_targets, val_targets) = build_auxiliary_teacher_targets(
                    distill.auxiliary_teacher_targets(),
                    train_dataset.len(),
                    val_dataset.len(),
                    student_patch_tokens,
                    config.dataset.cache_teacher_features_in_memory,
                )?;
                train_dataset = train_dataset.with_teacher_targets(train_targets);
                val_dataset = val_dataset.with_teacher_targets(val_targets);

                Ok((
                    Arc::new(train_dataset),
                    Arc::new(val_dataset),
                    Some(teacher_model),
                ))
            }
        }
    }
}

fn checkpoint_base(path: &Path) -> PathBuf {
    match path.extension().and_then(|ext| ext.to_str()) {
        Some("bin") => path.with_extension(""),
        _ => path.to_path_buf(),
    }
}

fn scalar_tensor_value(tensor: Tensor<MetricsBackend, 1>) -> f64 {
    tensor
        .mean()
        .into_data()
        .iter::<f64>()
        .next()
        .unwrap_or(0.0)
}

fn scalar_output_value<I>(output: &VisionOutput<MetricsBackend>) -> f64
where
    VisionOutput<MetricsBackend>: Adaptor<I>,
    I: ScalarValue<MetricsBackend>,
{
    let input = <VisionOutput<MetricsBackend> as Adaptor<I>>::adapt(output);
    scalar_tensor_value(input.value())
}

fn synced_distill_rollout_metrics(
    output: &VisionOutput<MetricsBackend>,
) -> (
    [f64; VISION_ROLLOUT_HORIZON_COUNT],
    [f64; VISION_ROLLOUT_HORIZON_COUNT],
    [f64; VISION_ROLLOUT_HORIZON_COUNT],
) {
    (
        [
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 0>>(output),
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 1>>(output),
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 2>>(output),
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 3>>(output),
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 4>>(output),
            scalar_output_value::<RolloutInvToHorizonInput<MetricsBackend, 5>>(output),
        ],
        [
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 0>>(output),
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 1>>(output),
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 2>>(output),
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 3>>(output),
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 4>>(output),
            scalar_output_value::<RolloutStateNormRatioToHorizonInput<MetricsBackend, 5>>(output),
        ],
        [
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 0>>(output),
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 1>>(output),
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 2>>(output),
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 3>>(output),
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 4>>(output),
            scalar_output_value::<RolloutStateMotionToHorizonInput<MetricsBackend, 5>>(output),
        ],
    )
}

pub fn eval_vision_distill_checkpoint_backend<B, Init>(
    config: &VisionTrainingConfig,
    checkpoint: &Path,
    backend_name: &str,
    init_backend: Init,
) -> Result<VisionDistillCheckpointEvalSummary>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
    Init: Fn(&B::Device),
{
    let device = B::Device::default();
    B::seed(&device, 1337);
    init_backend(&device);

    let training = &config.training;
    if training.batch_size == 0 {
        return Err(anyhow!("vision eval batch_size must be > 0"));
    }

    let vision_config = config.vision.build();
    if vision_config.patch_size == 0 {
        return Err(anyhow!("vision.patch_size must be > 0"));
    }
    if config.augment.image_size != vision_config.image_size {
        return Err(anyhow!(
            "augment.image_size ({}) must match vision.image_size ({})",
            config.augment.image_size,
            vision_config.image_size
        ));
    }

    let distill = match &config.mode {
        VisionTrainingModeConfig::Distill(distill) => distill.clone(),
        _ => {
            return Err(anyhow!(
                "vision distill checkpoint eval requires mode.type = \"distill\""
            ));
        }
    };

    let rollout = resolve_vision_rollout(training, vision_config.steps)?;
    maybe_download_vision_dataset(&config.dataset)?;

    let student_patch_tokens = vision_config
        .image_size
        .div_ceil(vision_config.patch_size)
        .pow(2);
    let normalize =
        VisionNormalize::new(config.augment.normalize_mean, config.augment.normalize_std);
    let train_aug = ImageNetAugmentations::new(
        ImageNetSplit::Train,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let val_aug = ImageNetAugmentations::new(
        ImageNetSplit::Val,
        config.augment.image_size,
        config.augment.resize_short,
        config.augment.min_scale,
        config.augment.max_scale,
        config.augment.min_aspect_ratio,
        config.augment.max_aspect_ratio,
        config.augment.flip_prob,
        config.augment.color_jitter_prob,
        config.augment.brightness,
        config.augment.contrast,
        config.augment.saturation,
        config.augment.hue,
        config.augment.grayscale_prob,
        config.augment.blur_prob,
        config.augment.blur_sigma_min,
        config.augment.blur_sigma_max,
        config.augment.solarize_prob,
        config.augment.solarize_threshold,
    );
    let train_root = config.dataset.imagenet_root.join(&config.dataset.train_dir);
    let val_root = config.dataset.imagenet_root.join(&config.dataset.val_dir);
    let (_, val_dataset, teacher) =
        build_distill_datasets_and_teacher::<B>(DistillDatasetRequest {
            config,
            vision_config: &vision_config,
            normalize,
            train_aug,
            val_aug,
            train_root: &train_root,
            val_root: &val_root,
            student_patch_tokens,
            device: &device,
        })?;

    let valid_device = device.clone();
    let valid_steps = val_dataset.steps_per_epoch(training.batch_size);
    let valid_loader: Arc<dyn DataLoader<ValidBackend<B>, ImageNetBatch<ValidBackend<B>>>> =
        Arc::new(ImageNetDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&val_dataset),
            training.batch_size,
            &valid_device,
            valid_steps,
            None,
            config.dataset.prefetch_batches,
            config.dataset.prefetch_workers,
            config.dataset.prefetch_to_device,
        ));

    let model = VisionDragon::<B>::new(vision_config, &device);
    let mut valid_model =
        VisionDistillModel::new(model, distill, teacher, rollout, &device).valid();
    let checkpoint_base = checkpoint_base(checkpoint);
    let checkpoint_path = checkpoint_base.with_extension("bin");
    if !checkpoint_path.exists() {
        return Err(anyhow!(
            "checkpoint file {} not found",
            checkpoint_path.display()
        ));
    }
    let record = BinFileRecorder::<FullPrecisionSettings>::new()
        .load::<<VisionDistillModel<ValidBackend<B>> as Module<ValidBackend<B>>>::Record>(
            checkpoint_base.clone(),
            &valid_device,
        )
        .with_context(|| format!("failed to load checkpoint {}", checkpoint_path.display()))?;
    valid_model = valid_model.load_record(record);

    let mut batches = 0usize;
    let mut distill_total = 0.0f64;
    let mut distill_patch = 0.0f64;
    let mut distill_cls = 0.0f64;
    let mut distill_total_to_horizon = [0.0f64; VISION_ROLLOUT_HORIZON_COUNT];
    let mut distill_patch_to_horizon = [0.0f64; VISION_ROLLOUT_HORIZON_COUNT];
    let mut distill_cls_to_horizon = [0.0f64; VISION_ROLLOUT_HORIZON_COUNT];

    for batch in valid_loader.iter() {
        let output = ValidStep::step(&valid_model, batch).sync();
        let batch_total = scalar_output_value::<LossValue<MetricsBackend>>(&output);
        let batch_patch = scalar_output_value::<InvLossInput<MetricsBackend>>(&output);
        let batch_cls = scalar_output_value::<ObserveLossInput<MetricsBackend>>(&output);
        let (batch_total_rollout, batch_patch_rollout, batch_cls_rollout) =
            synced_distill_rollout_metrics(&output);

        distill_total += batch_total;
        distill_patch += batch_patch;
        distill_cls += batch_cls;
        for index in 0..VISION_ROLLOUT_HORIZON_COUNT {
            distill_total_to_horizon[index] += batch_total_rollout[index];
            distill_patch_to_horizon[index] += batch_patch_rollout[index];
            distill_cls_to_horizon[index] += batch_cls_rollout[index];
        }
        batches += 1;
    }

    if batches == 0 {
        return Err(anyhow!("validation loader produced zero distill batches"));
    }

    let inv_batches = 1.0 / batches as f64;
    for index in 0..VISION_ROLLOUT_HORIZON_COUNT {
        distill_total_to_horizon[index] *= inv_batches;
        distill_patch_to_horizon[index] *= inv_batches;
        distill_cls_to_horizon[index] *= inv_batches;
    }

    Ok(VisionDistillCheckpointEvalSummary {
        backend: backend_name.to_string(),
        checkpoint: checkpoint_path,
        batches,
        distill_total: distill_total * inv_batches,
        distill_patch: distill_patch * inv_batches,
        distill_cls: distill_cls * inv_batches,
        horizons: VISION_ROLLOUT_HORIZON_CAPS,
        distill_total_to_horizon,
        distill_patch_to_horizon,
        distill_cls_to_horizon,
    })
}
