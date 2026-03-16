use super::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct VisionTrainingConfig {
    pub dataset: VisionDatasetConfig,
    pub training: VisionTrainingHyperparameters,
    pub optimizer: OptimizerConfig,
    #[serde(default)]
    pub wgpu: WgpuRuntimeConfig,
    pub vision: VisionModelConfig,
    #[serde(default)]
    pub augment: VisionAugmentationConfig,
    #[serde(default)]
    pub mode: VisionTrainingModeConfig,
}

impl VisionTrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.training.batch_size == 0 {
            return Err(anyhow!("training.batch_size must be > 0"));
        }
        if self.training.gradient_accumulation_steps == 0 {
            return Err(anyhow!("training.gradient_accumulation_steps must be > 0"));
        }
        if self.training.max_iters == 0 {
            return Err(anyhow!("training.max_iters must be > 0"));
        }
        if self.training.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        if self.training.batch_repeats == 0 {
            return Err(anyhow!("training.batch_repeats must be > 0"));
        }
        if self.training.trace_train_loss_every == 0 {
            return Err(anyhow!("training.trace_train_loss_every must be > 0"));
        }
        if let Some(epochs) = self.training.epochs
            && epochs == 0
        {
            return Err(anyhow!("training.epochs must be > 0"));
        }
        if matches!(
            self.training.schedule_mode,
            Some(VisionTrainScheduleMode::Epochs)
        ) && self.training.epochs.is_none()
        {
            return Err(anyhow!(
                "training.schedule_mode=epochs requires training.epochs to be set"
            ));
        }
        self.optimizer.validate()?;

        if self.vision.image_size == 0 {
            return Err(anyhow!("vision.image_size must be > 0"));
        }
        if self.vision.patch_size == 0 {
            return Err(anyhow!("vision.patch_size must be > 0"));
        }
        if self.vision.in_channels == 0 {
            return Err(anyhow!("vision.in_channels must be > 0"));
        }
        if self.vision.embed_dim == 0 {
            return Err(anyhow!("vision.embed_dim must be > 0"));
        }
        if self.vision.steps == 0 {
            return Err(anyhow!("vision.steps must be > 0"));
        }
        if self.vision.cross_eye_steps > self.vision.steps {
            return Err(anyhow!(
                "vision.cross_eye_steps ({}) must be <= vision.steps ({})",
                self.vision.cross_eye_steps,
                self.vision.steps
            ));
        }
        if self.vision.n_head == 0 {
            return Err(anyhow!("vision.n_head must be > 0"));
        }
        if self.vision.mlp_internal_dim_multiplier == 0 {
            return Err(anyhow!("vision.mlp_internal_dim_multiplier must be > 0"));
        }
        if self.vision.projection_dim == 0 {
            return Err(anyhow!("vision.projection_dim must be > 0"));
        }
        if self.vision.projection_hidden_dim == 0 {
            return Err(anyhow!("vision.projection_hidden_dim must be > 0"));
        }
        if self.vision.num_eyes == 0 {
            return Err(anyhow!("vision.num_eyes must be > 0"));
        }
        if self.vision.cls_sync_alpha < 0.0 || self.vision.cls_sync_alpha > 1.0 {
            return Err(anyhow!("vision.cls_sync_alpha must be between 0.0 and 1.0"));
        }
        if self.vision.dropout < 0.0 {
            return Err(anyhow!("vision.dropout must be >= 0"));
        }
        if matches!(self.vision.pos_max_height, Some(0)) {
            return Err(anyhow!("vision.pos_max_height must be > 0 when set"));
        }
        if matches!(self.vision.pos_max_width, Some(0)) {
            return Err(anyhow!("vision.pos_max_width must be > 0 when set"));
        }
        if !self.vision.allow_softmax_attention
            && matches!(self.vision.attention_mode, VisionAttentionMode::Softmax)
        {
            return Err(anyhow!(
                "vision.attention_mode=softmax requires vision.allow_softmax_attention=true"
            ));
        }

        load_validate::validate_vision_mhc(&self.vision)?;
        load_validate::validate_vision_trm_graph(&self.vision)?;
        load_validate::validate_vision_rho_stream(&self.vision)?;
        load_validate::validate_vision_rollout(&self.training, self.vision.steps)?;
        load_validate::validate_vision_mode(
            &self.mode,
            &self.vision,
            &self.dataset,
            &self.augment,
        )?;

        if self.optimizer.learning_rate <= 0.0 {
            return Err(anyhow!("optimizer.learning_rate must be > 0"));
        }
        if self.optimizer.weight_decay < 0.0 {
            return Err(anyhow!("optimizer.weight_decay must be >= 0"));
        }

        Ok(())
    }
}
