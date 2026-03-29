mod artifacts;
mod dataset;
mod run;
mod teacher;
mod validation;

pub use crate::config::{DreamerConfig, DreamerLatentBackend, MovingMnistDreamerTrainConfig};
pub use crate::tokenizer::{MovingMnistTokenizerRunSummary, train_moving_mnist_tokenizer};
pub(crate) use artifacts::{should_write_artifacts, write_validation_artifacts};
pub(crate) use dataset::{
    build_cached_moving_mnist_split, build_moving_mnist_dreamer_datasets,
    build_moving_mnist_tokenizer_datasets,
};
pub use run::train_moving_mnist;
pub(crate) use teacher::{
    AutoGazeSource, ClipTeacherSource, encode_crop_teacher, load_autogaze_source,
    load_crop_teacher_source, load_vjepa_source, passive_full_frame_traces, uses_crop_teacher,
    uses_global_teacher, zero_teacher_features,
};
pub use validation::{MovingMnistDreamerRunSummary, SelectedValidationMetrics};
pub(crate) use validation::{
    artifact_metrics_from_forward_and_snapshot, dynamics_quality_score, evaluate_validation,
    rollout_selection_score, scalar_pack,
};
