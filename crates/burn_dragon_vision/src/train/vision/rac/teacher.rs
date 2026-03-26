use crate::train::prelude::*;

pub(crate) fn rac_semantic_teacher_feature_dim(
    config: &VisionRacConfig,
    fallback_dim: usize,
) -> Option<usize> {
    if config.semantic_teacher.weight <= 0.0 {
        return None;
    }
    match &config.semantic_teacher.teacher {
        VisionTeacherConfig::Features(teacher) => Some(teacher.feature_dim.max(1)),
        VisionTeacherConfig::Model(teacher) => {
            Some(teacher.feature_dim.unwrap_or(fallback_dim).max(1))
        }
    }
}

pub(crate) fn build_rac_semantic_teacher_store(
    split: ImageNetSplit,
    config: &VisionRacConfig,
    expected_records: usize,
    cache_in_memory: bool,
) -> Result<Option<Arc<DinoFeatureStore>>> {
    if config.semantic_teacher.weight <= 0.0 {
        return Ok(None);
    }

    let teacher = match &config.semantic_teacher.teacher {
        VisionTeacherConfig::Features(teacher) => teacher,
        VisionTeacherConfig::Model(_) => {
            return Err(anyhow!(
                "RAC semantic teacher currently supports precomputed feature stores only; VisionTeacherConfig::Model is not yet supported"
            ));
        }
    };

    let cls_path = match split {
        ImageNetSplit::Train => teacher.train_cls_path.as_path(),
        ImageNetSplit::Val => teacher.val_cls_path.as_path(),
    };
    let patch_path = match split {
        ImageNetSplit::Train => teacher.train_patch_path.as_deref(),
        ImageNetSplit::Val => teacher.val_patch_path.as_deref(),
    };

    Ok(Some(Arc::new(
        DinoFeatureStore::new_optional_patch_with_options(
            cls_path,
            patch_path,
            teacher.feature_dim.max(1),
            teacher.patch_tokens,
            Some(expected_records),
            cache_in_memory,
        )?,
    )))
}

pub(crate) fn build_rac_teacher_latent_store(
    split: ImageNetSplit,
    config: &VisionRacConfig,
    expected_records: usize,
    cache_in_memory: bool,
) -> Result<Option<Arc<ImageTensorStore>>> {
    let spec = match config.teacher.kind {
        VisionRacTeacherKind::PooledImage => return Ok(None),
        VisionRacTeacherKind::PrecomputedLatent => config
            .teacher
            .precomputed_latent
            .as_ref()
            .ok_or_else(|| anyhow!("missing RAC precomputed latent teacher config"))?,
    };
    let path = match split {
        ImageNetSplit::Train => spec.train_path.as_path(),
        ImageNetSplit::Val => spec.val_path.as_path(),
    };
    Ok(Some(Arc::new(ImageTensorStore::new_with_options(
        path,
        spec.channels.max(1),
        spec.height.max(1),
        spec.width.max(1),
        Some(expected_records),
        cache_in_memory,
    )?)))
}
