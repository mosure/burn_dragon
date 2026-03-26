#![cfg(feature = "train")]

use anyhow::{Result, anyhow};
use burn::tensor::backend::{AutodiffBackend, Backend as BackendTrait};
use burn_dragon_core::api::recurrent::BDH;
use burn_dragon_language::api::checkpoint::{
    load_language_core_from_checkpoint, load_tokenizer_for_checkpoint,
};
use burn_dragon_language::api::inference::CharVocab;
use burn_dragon_vision::api::checkpoint::load_vision_encoder_from_checkpoint;
use burn_dragon_vision::api::model::VisionDragon;

use crate::config::VlJepaDragonConfig;
use crate::runtime_config::{
    MultimodalPretrainedTextCoreConfig, MultimodalTrainingConfig, MultimodalVideoTrainingConfig,
};

pub(crate) struct ResolvedMultimodalInit<B: BackendTrait> {
    pub(crate) model_config: VlJepaDragonConfig,
    pub(crate) pretrained_vocab: Option<CharVocab>,
    pub(crate) vision_x_encoder: Option<VisionDragon<B>>,
    pub(crate) text_core: Option<BDH<B>>,
    pub(crate) fusion_core: Option<BDH<B>>,
    pub(crate) freeze_vision_x_encoder: bool,
    pub(crate) freeze_query_q_encoder: bool,
    pub(crate) freeze_target_y_encoder: bool,
}

fn resolve_pretrained_char_vocab(
    pretrained: &MultimodalPretrainedTextCoreConfig,
) -> Result<CharVocab> {
    let checkpoint = pretrained.checkpoint.clone();
    let tokenizer = load_tokenizer_for_checkpoint(
        &pretrained.config_paths,
        Some(&checkpoint),
        &pretrained.backend_name,
    )?;
    tokenizer
        .as_ref()
        .as_any()
        .downcast_ref::<CharVocab>()
        .cloned()
        .ok_or_else(|| {
            anyhow!("multimodal pretrained text core currently requires a char tokenizer")
        })
}

pub(crate) fn resolve_image_pretrained_init<B: AutodiffBackend>(
    config: &MultimodalTrainingConfig,
    device: &B::Device,
) -> Result<ResolvedMultimodalInit<B>> {
    let mut model_config = config.model.clone();
    let default_model = VlJepaDragonConfig::default();
    let mut pretrained_vocab = None;
    let mut vision_x_encoder = None;
    let mut text_core = None;
    let mut fusion_core = None;
    let mut freeze_vision_x_encoder = false;
    let mut freeze_query_q_encoder = false;
    let mut freeze_target_y_encoder = false;

    if let Some(pretrained) = config.pretrained.text_core.as_ref() {
        let checkpoint = pretrained.checkpoint.clone();
        let language_config =
            burn_dragon_language::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                Some(&checkpoint),
                &pretrained.backend_name,
            )?;
        let source_config = burn_dragon_language::api::inference::build_model_config(
            &language_config.model,
            language_config.training.block_size,
        );
        model_config.query_text = source_config.clone();
        model_config.target_text = source_config;
        if pretrained.initialize_fusion {
            model_config.fusion = burn_dragon_language::api::inference::build_model_config(
                &language_config.model,
                language_config.training.block_size,
            );
        }
        if model_config.target_dim == default_model.target_dim {
            model_config.target_dim = model_config.target_text.n_embd;
        }
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.query_text.n_embd;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        let loaded_text = load_language_core_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            &pretrained.backend_name,
            device,
        )?;
        text_core = Some(loaded_text.clone());
        if pretrained.initialize_fusion {
            fusion_core = Some(loaded_text);
        }
        if pretrained.use_pretrained_tokenizer {
            pretrained_vocab = Some(resolve_pretrained_char_vocab(pretrained)?);
        }
        freeze_query_q_encoder = pretrained.freeze_query;
        freeze_target_y_encoder = pretrained.freeze_target;
    }

    if let Some(pretrained) = config.pretrained.vision_x_encoder.as_ref() {
        let vision_training =
            burn_dragon_vision::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                &pretrained.checkpoint,
            )?;
        model_config.vision = vision_training.vision.build();
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.vision.embed_dim;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        vision_x_encoder = Some(load_vision_encoder_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            device,
        )?);
        freeze_vision_x_encoder = pretrained.freeze;
    }

    Ok(ResolvedMultimodalInit {
        model_config,
        pretrained_vocab,
        vision_x_encoder,
        text_core,
        fusion_core,
        freeze_vision_x_encoder,
        freeze_query_q_encoder,
        freeze_target_y_encoder,
    })
}

pub(crate) fn resolve_video_pretrained_init<B: AutodiffBackend>(
    config: &MultimodalVideoTrainingConfig,
    device: &B::Device,
) -> Result<ResolvedMultimodalInit<B>> {
    let mut model_config = config.model.clone();
    let default_model = VlJepaDragonConfig::default();
    let mut pretrained_vocab = None;
    let mut vision_x_encoder = None;
    let mut text_core = None;
    let mut fusion_core = None;
    let mut freeze_vision_x_encoder = false;
    let mut freeze_query_q_encoder = false;
    let mut freeze_target_y_encoder = false;

    if let Some(pretrained) = config.pretrained.text_core.as_ref() {
        let checkpoint = pretrained.checkpoint.clone();
        let language_config =
            burn_dragon_language::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                Some(&checkpoint),
                &pretrained.backend_name,
            )?;
        let source_config = burn_dragon_language::api::inference::build_model_config(
            &language_config.model,
            language_config.training.block_size,
        );
        model_config.query_text = source_config.clone();
        model_config.target_text = source_config;
        if pretrained.initialize_fusion {
            model_config.fusion = burn_dragon_language::api::inference::build_model_config(
                &language_config.model,
                language_config.training.block_size,
            );
        }
        if model_config.target_dim == default_model.target_dim {
            model_config.target_dim = model_config.target_text.n_embd;
        }
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.query_text.n_embd;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        let loaded_text = load_language_core_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            &pretrained.backend_name,
            device,
        )?;
        text_core = Some(loaded_text.clone());
        if pretrained.initialize_fusion {
            fusion_core = Some(loaded_text);
        }
        if pretrained.use_pretrained_tokenizer {
            pretrained_vocab = Some(resolve_pretrained_char_vocab(pretrained)?);
        }
        freeze_query_q_encoder = pretrained.freeze_query;
        freeze_target_y_encoder = pretrained.freeze_target;
    }

    if let Some(pretrained) = config.pretrained.vision_x_encoder.as_ref() {
        let vision_training =
            burn_dragon_vision::api::checkpoint::load_training_config_for_checkpoint(
                &pretrained.config_paths,
                &pretrained.checkpoint,
            )?;
        model_config.vision = vision_training.vision.build();
        if model_config.fusion_dim == default_model.fusion_dim {
            model_config.fusion_dim = model_config.vision.embed_dim;
        }
        if model_config.fusion.n_embd == default_model.fusion.n_embd {
            model_config.fusion.n_embd = model_config.fusion_dim;
        }
        vision_x_encoder = Some(load_vision_encoder_from_checkpoint::<B>(
            &pretrained.checkpoint,
            pretrained.epoch,
            &pretrained.config_paths,
            device,
        )?);
        freeze_vision_x_encoder = pretrained.freeze;
    }

    Ok(ResolvedMultimodalInit {
        model_config,
        pretrained_vocab,
        vision_x_encoder,
        text_core,
        fusion_core,
        freeze_vision_x_encoder,
        freeze_query_q_encoder,
        freeze_target_y_encoder,
    })
}
