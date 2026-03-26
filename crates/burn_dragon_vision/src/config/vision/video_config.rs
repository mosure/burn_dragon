use super::*;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoTemporalConfig {
    pub n_layer: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    pub rollout_fast_steps_per_slow_step: usize,
    pub predict_backprop_frames: usize,
    pub mode_embeddings: bool,
    pub refine_passes: usize,
    pub fused: bool,
    pub wgpu_recurrent_kernel: bool,
    pub wgpu_rollout_fused: bool,
    pub latent_block_size: usize,
    pub time_block_size: usize,
}

impl Default for VisionVideoTemporalConfig {
    fn default() -> Self {
        Self {
            n_layer: 2,
            n_head: 4,
            mlp_internal_dim_multiplier: 2,
            rollout_fast_steps_per_slow_step: 4,
            predict_backprop_frames: 0,
            mode_embeddings: true,
            refine_passes: 0,
            fused: true,
            wgpu_recurrent_kernel: true,
            wgpu_rollout_fused: true,
            latent_block_size: 8,
            time_block_size: 8,
        }
    }
}

impl<B: Backend> Module<B> for VisionVideoTemporalConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoTemporalConfig {
    type InnerModule = VisionVideoTemporalConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionVideoTemporalConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("n_layer", &self.n_layer)
            .add("n_head", &self.n_head)
            .add(
                "mlp_internal_dim_multiplier",
                &self.mlp_internal_dim_multiplier,
            )
            .add(
                "rollout_fast_steps_per_slow_step",
                &self.rollout_fast_steps_per_slow_step,
            )
            .add("predict_backprop_frames", &self.predict_backprop_frames)
            .add("mode_embeddings", &self.mode_embeddings)
            .add("refine_passes", &self.refine_passes)
            .add("fused", &self.fused)
            .add("wgpu_recurrent_kernel", &self.wgpu_recurrent_kernel)
            .add("wgpu_rollout_fused", &self.wgpu_rollout_fused)
            .add("latent_block_size", &self.latent_block_size)
            .add("time_block_size", &self.time_block_size)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoTemporalConfig {}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VisionVideoParadigmKind {
    #[default]
    LegacyRollout,
    #[serde(rename = "vjepa_2_1")]
    Vjepa21,
}

impl fmt::Display for VisionVideoParadigmKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LegacyRollout => write!(f, "legacy_rollout"),
            Self::Vjepa21 => write!(f, "vjepa_2_1"),
        }
    }
}

impl ModuleDisplayDefault for VisionVideoParadigmKind {
    fn content(&self, content: Content) -> Option<Content> {
        content.add_formatted(self).optional()
    }
}

impl ModuleDisplay for VisionVideoParadigmKind {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoVjepa21MaskConfig {
    pub spatial_scale_min: f32,
    pub spatial_scale_max: f32,
    pub temporal_scale_min: f32,
    pub temporal_scale_max: f32,
    pub aspect_ratio_min: f32,
    pub aspect_ratio_max: f32,
    pub num_blocks: usize,
    pub max_context_frames_ratio: f32,
    pub full_complement: bool,
}

impl Default for VisionVideoVjepa21MaskConfig {
    fn default() -> Self {
        Self {
            spatial_scale_min: 0.2,
            spatial_scale_max: 0.8,
            temporal_scale_min: 0.5,
            temporal_scale_max: 1.0,
            aspect_ratio_min: 0.3,
            aspect_ratio_max: 3.0,
            num_blocks: 4,
            max_context_frames_ratio: 1.0,
            full_complement: false,
        }
    }
}

impl<B: Backend> Module<B> for VisionVideoVjepa21MaskConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoVjepa21MaskConfig {
    type InnerModule = VisionVideoVjepa21MaskConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionVideoVjepa21MaskConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("spatial_scale_min", &self.spatial_scale_min)
            .add("spatial_scale_max", &self.spatial_scale_max)
            .add("temporal_scale_min", &self.temporal_scale_min)
            .add("temporal_scale_max", &self.temporal_scale_max)
            .add("aspect_ratio_min", &self.aspect_ratio_min)
            .add("aspect_ratio_max", &self.aspect_ratio_max)
            .add("num_blocks", &self.num_blocks)
            .add("max_context_frames_ratio", &self.max_context_frames_ratio)
            .add("full_complement", &self.full_complement)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoVjepa21MaskConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoVjepa21LossConfig {
    pub masked_weight: f32,
    pub context_weight: f32,
    pub predict_all: bool,
    pub weight_distance_loss: bool,
    pub offset_context_loss: bool,
    pub normalize_targets: bool,
    pub loss_exp: f32,
}

impl Default for VisionVideoVjepa21LossConfig {
    fn default() -> Self {
        Self {
            masked_weight: 1.0,
            context_weight: 0.5,
            predict_all: true,
            weight_distance_loss: true,
            offset_context_loss: false,
            normalize_targets: true,
            loss_exp: 1.0,
        }
    }
}

impl<B: Backend> Module<B> for VisionVideoVjepa21LossConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoVjepa21LossConfig {
    type InnerModule = VisionVideoVjepa21LossConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionVideoVjepa21LossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("masked_weight", &self.masked_weight)
            .add("context_weight", &self.context_weight)
            .add("predict_all", &self.predict_all)
            .add("weight_distance_loss", &self.weight_distance_loss)
            .add("offset_context_loss", &self.offset_context_loss)
            .add("normalize_targets", &self.normalize_targets)
            .add("loss_exp", &self.loss_exp)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoVjepa21LossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoVjepa21Config {
    pub clip_frames: usize,
    pub observe_steps: usize,
    pub observe_backprop_steps: usize,
    pub predictor_hidden_dim: usize,
    pub use_mask_token_bias: bool,
    pub checkpoint_depths: Vec<usize>,
    pub teacher_ema: VisionMomentumTeacherConfig,
    pub mask: VisionVideoVjepa21MaskConfig,
    pub loss: VisionVideoVjepa21LossConfig,
}

impl Default for VisionVideoVjepa21Config {
    fn default() -> Self {
        Self {
            clip_frames: 8,
            observe_steps: 1,
            observe_backprop_steps: 1,
            predictor_hidden_dim: 0,
            use_mask_token_bias: true,
            checkpoint_depths: vec![1, 2, 4],
            teacher_ema: VisionMomentumTeacherConfig::default(),
            mask: VisionVideoVjepa21MaskConfig::default(),
            loss: VisionVideoVjepa21LossConfig::default(),
        }
    }
}

impl<B: Backend> Module<B> for VisionVideoVjepa21Config {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoVjepa21Config {
    type InnerModule = VisionVideoVjepa21Config;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionVideoVjepa21Config {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("clip_frames", &self.clip_frames)
            .add("observe_steps", &self.observe_steps)
            .add("observe_backprop_steps", &self.observe_backprop_steps)
            .add("predictor_hidden_dim", &self.predictor_hidden_dim)
            .add("use_mask_token_bias", &self.use_mask_token_bias)
            .add(
                "checkpoint_depths",
                &format!("{:?}", self.checkpoint_depths),
            )
            .add("teacher_ema", &self.teacher_ema)
            .add("mask", &self.mask)
            .add("loss", &self.loss)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoVjepa21Config {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoLejepaLossConfig {
    pub prediction_weight: f32,
    pub observe_weight: f32,
    pub cosine_weight: f32,
    pub probe_weight: f32,
    pub debug_recon_weight: f32,
    pub debug_recon_hidden_dim: usize,
    pub sigreg: VisionLejepaLossConfig,
}

impl Default for VisionVideoLejepaLossConfig {
    fn default() -> Self {
        Self {
            prediction_weight: 1.0,
            observe_weight: 1.0,
            cosine_weight: 0.1,
            probe_weight: 0.25,
            debug_recon_weight: 1.0,
            debug_recon_hidden_dim: 256,
            sigreg: VisionLejepaLossConfig::default(),
        }
    }
}

impl ModuleDisplayDefault for VisionVideoLejepaLossConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("prediction_weight", &self.prediction_weight)
            .add("observe_weight", &self.observe_weight)
            .add("cosine_weight", &self.cosine_weight)
            .add("probe_weight", &self.probe_weight)
            .add("debug_recon_weight", &self.debug_recon_weight)
            .add("debug_recon_hidden_dim", &self.debug_recon_hidden_dim)
            .add("sigreg", &self.sigreg)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoLejepaLossConfig {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(default)]
pub struct VisionVideoLejepaConfig {
    pub paradigm: VisionVideoParadigmKind,
    pub context_frames: usize,
    pub target_frames: usize,
    pub train_target_frames_min: usize,
    pub train_target_frames_max: usize,
    pub train_target_warmup_steps: usize,
    pub frame_stride: usize,
    pub predictor_hidden_dim: usize,
    pub teacher_ema: VisionMomentumTeacherConfig,
    pub temporal: VisionVideoTemporalConfig,
    pub loss: VisionVideoLejepaLossConfig,
    pub artifact_output: VisionArtifactOutputMode,
    pub artifact_fps: u32,
    pub artifact_every: usize,
    pub artifact_max_images: usize,
    pub artifact_future_frames: usize,
    pub artifact_upscale: usize,
    pub artifact_overwrite: bool,
    pub vjepa21: VisionVideoVjepa21Config,
}

impl Default for VisionVideoLejepaConfig {
    fn default() -> Self {
        Self {
            paradigm: VisionVideoParadigmKind::default(),
            context_frames: 4,
            target_frames: 2,
            train_target_frames_min: 0,
            train_target_frames_max: 0,
            train_target_warmup_steps: 0,
            frame_stride: 1,
            predictor_hidden_dim: 0,
            teacher_ema: VisionMomentumTeacherConfig::default(),
            temporal: VisionVideoTemporalConfig::default(),
            loss: VisionVideoLejepaLossConfig::default(),
            artifact_output: VisionArtifactOutputMode::Mp4,
            artifact_fps: 6,
            artifact_every: 0,
            artifact_max_images: 4,
            artifact_future_frames: 0,
            artifact_upscale: 4,
            artifact_overwrite: true,
            vjepa21: VisionVideoVjepa21Config::default(),
        }
    }
}

impl<B: Backend> Module<B> for VisionVideoLejepaConfig {
    type Record = ();

    fn collect_devices(&self, devices: burn::module::Devices<B>) -> burn::module::Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: burn::module::ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: burn::module::ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for VisionVideoLejepaConfig {
    type InnerModule = VisionVideoLejepaConfig;

    fn valid(&self) -> Self::InnerModule {
        self.clone()
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for VisionVideoLejepaConfig {
    fn content(&self, content: Content) -> Option<Content> {
        content
            .add("paradigm", &self.paradigm)
            .add("context_frames", &self.context_frames)
            .add("target_frames", &self.target_frames)
            .add("train_target_frames_min", &self.train_target_frames_min)
            .add("train_target_frames_max", &self.train_target_frames_max)
            .add("train_target_warmup_steps", &self.train_target_warmup_steps)
            .add("frame_stride", &self.frame_stride)
            .add("predictor_hidden_dim", &self.predictor_hidden_dim)
            .add("teacher_ema", &self.teacher_ema)
            .add("temporal", &self.temporal)
            .add("loss", &self.loss)
            .add("artifact_output", &self.artifact_output)
            .add("artifact_fps", &self.artifact_fps)
            .add("artifact_every", &self.artifact_every)
            .add("artifact_max_images", &self.artifact_max_images)
            .add("artifact_future_frames", &self.artifact_future_frames)
            .add("artifact_upscale", &self.artifact_upscale)
            .add("artifact_overwrite", &self.artifact_overwrite)
            .add("vjepa21", &self.vjepa21)
            .optional()
    }
}

impl ModuleDisplay for VisionVideoLejepaConfig {}

impl VisionVideoLejepaConfig {
    pub fn is_vjepa21(&self) -> bool {
        self.paradigm == VisionVideoParadigmKind::Vjepa21
    }

    pub fn effective_train_target_frames_min(&self) -> usize {
        if self.train_target_frames_min == 0 {
            self.target_frames.max(1)
        } else {
            self.train_target_frames_min.max(1)
        }
    }

    pub fn effective_train_target_frames_max(&self) -> usize {
        let min_frames = self.effective_train_target_frames_min();
        let configured_max = if self.train_target_frames_max == 0 {
            self.target_frames
        } else {
            self.train_target_frames_max
        };
        configured_max.max(min_frames)
    }

    pub fn max_supervised_target_frames(&self) -> usize {
        self.target_frames
            .max(self.effective_train_target_frames_max())
            .max(1)
    }
}
