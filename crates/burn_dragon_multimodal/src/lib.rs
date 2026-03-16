#![recursion_limit = "256"]

//! Multimodal Dragon composition and protocol-faithful VL-JEPA assembly.

pub mod adapters;
pub mod config;
pub mod data;
pub mod loss;
pub mod model;
pub mod state;

#[cfg(feature = "train")]
pub mod checkpoint;
#[cfg(feature = "train")]
mod config_io;
#[cfg(feature = "train")]
mod ema;
#[cfg(feature = "train")]
pub mod runtime;
#[cfg(feature = "train")]
pub mod train;

pub mod api {
    //! Curated multimodal-facing Dragon API.

    pub mod config {
        pub use crate::config::{
            FusionSlotConfig, MultimodalTbpttConfig, TargetTeacherConfig, TargetTextEncoderKind,
            VlJepaDragonConfig,
        };
    }

    pub mod data {
        pub use crate::data::{
            MultimodalStepMode, VideoLanguageTripletBatch, VideoLanguageTripletSegment,
            VisionLanguageTripletBatch, VisionLanguageTripletSegment,
        };
    }

    pub mod model {
        pub use crate::adapters::{
            TargetTextDragonEncoderAdapter, TargetTextEmbeddingOutput, TargetTextEncoderAdapter,
            TextDragonFusionAdapter, TextFusionAdapter, TextFusionOutput,
            VisionDragonFusionAdapter, VisionFusionAdapter, VisionFusionOutput,
        };
        pub use crate::model::{
            FusionCoreOutput, FusionInputBatch, VlJepaDragon, VlJepaForwardOutput, VlJepaTargets,
        };
        pub use crate::state::{MultimodalDragonState, VisionMultimodalState, detach_model_state};
    }

    pub mod loss {
        pub use crate::loss::{VlJepaLossBreakdown, vl_jepa_bidirectional_info_nce_loss};
    }

    #[cfg(feature = "train")]
    pub mod checkpoint {
        pub use crate::checkpoint::{
            MultimodalBurnpackExportReport, MultimodalRunConfigSnapshot, default_checkpoint_dir,
            export_multimodal_checkpoint_to_burnpack, load_training_config_for_checkpoint,
            load_training_snapshot_from_run_dir, training_snapshot_path, write_training_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod runtime {
        pub use crate::runtime::{
            MultimodalEpochArtifact, MultimodalImageTextDataConfig, MultimodalImageTextSource,
            MultimodalMnistLabelTextConfig, MultimodalMovingMnistLabelTextConfig,
            MultimodalRuntimeConfig, MultimodalTaskKind, MultimodalTrainingConfig,
            MultimodalTrainingLoopConfig, MultimodalTrainingReport, MultimodalVideoTextDataConfig,
            MultimodalVideoTextSource, MultimodalVideoTrainingConfig, artifact_dir,
            load_multimodal_runtime_config, load_multimodal_training_runtime_config,
            load_multimodal_video_training_runtime_config, run_image_text_training_backend,
            run_video_text_training_backend, tokenizer_snapshot_path, train_backend,
            train_video_backend, training_runtime_snapshot_path, write_runtime_snapshot,
            write_video_runtime_snapshot,
        };
    }

    #[cfg(feature = "train")]
    pub mod train {
        pub use crate::train::{
            JsonlVideoLanguageDataset, JsonlVisionLanguageDataset, MnistVisionLanguageDataset,
            MovingMnistVideoLanguageDataset, MultimodalTrainStepOutput, VideoLanguageCpuSample,
            VideoLanguageJsonlRecord, VisionLanguageCpuSample, VisionLanguageJsonlRecord,
            collate_video_language_segments, collate_vision_language_segments,
            multimodal_train_step, multimodal_video_train_step,
        };
    }

    pub mod expert {
        pub use crate::{adapters, config, data, loss, model, state};
        #[cfg(feature = "train")]
        pub use crate::{checkpoint, train};
    }
}

pub use adapters::{
    TargetTextDragonEncoderAdapter, TargetTextEmbeddingOutput, TargetTextEncoderAdapter,
    TextDragonFusionAdapter, TextFusionAdapter, TextFusionOutput, VisionDragonFusionAdapter,
    VisionFusionAdapter, VisionFusionOutput,
};
pub use config::{
    FusionSlotConfig, MultimodalTbpttConfig, TargetTeacherConfig, TargetTextEncoderKind,
    VlJepaDragonConfig,
};
pub use data::{
    MultimodalStepMode, VideoLanguageTripletBatch, VideoLanguageTripletSegment,
    VisionLanguageTripletBatch, VisionLanguageTripletSegment,
};
pub use loss::{VlJepaLossBreakdown, vl_jepa_bidirectional_info_nce_loss};
pub use model::{
    FusionCoreOutput, FusionInputBatch, VlJepaDragon, VlJepaForwardOutput, VlJepaTargets,
};
pub use state::{MultimodalDragonState, VisionMultimodalState, detach_model_state};

#[cfg(feature = "train")]
pub use checkpoint::{
    MultimodalBurnpackExportReport, MultimodalRunConfigSnapshot, default_checkpoint_dir,
    export_multimodal_checkpoint_to_burnpack, load_training_config_for_checkpoint,
    load_training_snapshot_from_run_dir, training_snapshot_path, write_training_snapshot,
};
#[cfg(feature = "train")]
pub use runtime::{
    MultimodalEpochArtifact, MultimodalImageTextDataConfig, MultimodalImageTextSource,
    MultimodalMnistLabelTextConfig, MultimodalMovingMnistLabelTextConfig, MultimodalRuntimeConfig,
    MultimodalTaskKind, MultimodalTrainingConfig, MultimodalTrainingLoopConfig,
    MultimodalTrainingReport, MultimodalVideoTextDataConfig, MultimodalVideoTextSource,
    MultimodalVideoTrainingConfig, artifact_dir, load_multimodal_runtime_config,
    load_multimodal_training_runtime_config, load_multimodal_video_training_runtime_config,
    run_image_text_training_backend, run_video_text_training_backend, tokenizer_snapshot_path,
    train_backend, train_video_backend, training_runtime_snapshot_path, write_runtime_snapshot,
    write_video_runtime_snapshot,
};
#[cfg(feature = "train")]
pub use train::{
    JsonlVideoLanguageDataset, JsonlVisionLanguageDataset, MnistVisionLanguageDataset,
    MovingMnistVideoLanguageDataset, MultimodalTrainStepOutput, VideoLanguageCpuSample,
    VideoLanguageJsonlRecord, VisionLanguageCpuSample, VisionLanguageJsonlRecord,
    collate_video_language_segments, collate_vision_language_segments, multimodal_train_step,
    multimodal_video_train_step,
};
