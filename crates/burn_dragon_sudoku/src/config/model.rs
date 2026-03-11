use super::*;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuGridPositional {
    #[default]
    Additive,
    #[serde(rename = "rope_2d")]
    Rope2d,
}

impl SudokuGridPositional {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Additive => "additive",
            Self::Rope2d => "rope_2d",
        }
    }
}

impl std::fmt::Display for SudokuGridPositional {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<B: Backend> Module<B> for SudokuGridPositional {
    type Record = ();

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        devices
    }

    fn fork(self, _device: &B::Device) -> Self {
        self
    }

    fn to_device(self, _device: &B::Device) -> Self {
        self
    }

    fn visit<Visitor: ModuleVisitor<B>>(&self, _visitor: &mut Visitor) {}

    fn map<Mapper: ModuleMapper<B>>(self, _mapper: &mut Mapper) -> Self {
        self
    }

    fn load_record(self, _record: Self::Record) -> Self {
        self
    }

    fn into_record(self) -> Self::Record {}
}

impl<B: AutodiffBackend> AutodiffModule<B> for SudokuGridPositional {
    type InnerModule = SudokuGridPositional;

    fn valid(&self) -> Self::InnerModule {
        *self
    }

    fn from_inner(module: Self::InnerModule) -> Self {
        module
    }
}

impl ModuleDisplayDefault for SudokuGridPositional {
    fn content(&self, content: Content) -> Option<Content> {
        let summary = format!("grid_positional={self}");
        content
            .set_top_level_type("SudokuGridPositional")
            .add_formatted(&summary)
            .optional()
    }
}

impl ModuleDisplay for SudokuGridPositional {}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuCacheMhcConfig {
    #[serde(default)]
    pub enabled: bool,
    #[serde(default = "default_cache_mhc_num_streams")]
    pub num_streams: usize,
    #[serde(default = "default_cache_mhc_num_views")]
    pub num_views: usize,
    #[serde(default = "default_cache_mhc_iters")]
    pub mhc_iters: usize,
    #[serde(default = "default_cache_mhc_tau")]
    pub mhc_tau: f32,
    #[serde(default = "default_cache_mhc_add_branch_out_to_residual")]
    pub add_branch_out_to_residual: bool,
    #[serde(default = "default_cache_mhc_dropout")]
    pub dropout: f64,
}

impl Default for SudokuCacheMhcConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_streams: default_cache_mhc_num_streams(),
            num_views: default_cache_mhc_num_views(),
            mhc_iters: default_cache_mhc_iters(),
            mhc_tau: default_cache_mhc_tau(),
            add_branch_out_to_residual: default_cache_mhc_add_branch_out_to_residual(),
            dropout: default_cache_mhc_dropout(),
        }
    }
}

impl SudokuCacheMhcConfig {
    pub fn to_core(&self) -> ManifoldHyperConnectionsConfig {
        ManifoldHyperConnectionsConfig {
            enabled: self.enabled,
            num_streams: self.num_streams,
            num_views: self.num_views,
            mhc_iters: self.mhc_iters,
            mhc_tau: self.mhc_tau,
            add_branch_out_to_residual: self.add_branch_out_to_residual,
            dropout: self.dropout,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SudokuCacheUpdateMode {
    #[default]
    Overwrite,
    GatedResidual,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct SudokuCacheUpdateConfig {
    #[serde(default)]
    pub mode: SudokuCacheUpdateMode,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuModelConfig {
    pub n_layer: usize,
    pub n_embd: usize,
    pub n_head: usize,
    pub mlp_internal_dim_multiplier: usize,
    #[serde(default = "default_summary_tokens")]
    pub summary_tokens: usize,
    #[serde(default = "default_policy_heads")]
    pub policy_heads: usize,
    #[serde(default)]
    pub policy_head: SudokuPolicyHead,
    #[serde(default = "default_policy_mlp_hidden_mult")]
    pub policy_mlp_hidden_mult: usize,
    #[serde(default)]
    pub rotary_embedding: RotaryEmbedding,
    #[serde(default)]
    pub grid_positional: SudokuGridPositional,
    #[serde(default = "default_grid_rope_theta")]
    pub grid_rope_theta: f32,
    #[serde(default = "default_dropout")]
    pub dropout: f64,
    #[serde(default)]
    pub fused_kernels: bool,
    #[serde(default)]
    pub relu_threshold: f32,
    #[serde(default)]
    pub cache_mhc: SudokuCacheMhcConfig,
    #[serde(default)]
    pub cache_update: SudokuCacheUpdateConfig,
}

impl Default for SudokuModelConfig {
    fn default() -> Self {
        Self {
            n_layer: 6,
            n_embd: 256,
            n_head: 4,
            mlp_internal_dim_multiplier: 4,
            summary_tokens: default_summary_tokens(),
            policy_heads: default_policy_heads(),
            policy_head: SudokuPolicyHead::default(),
            policy_mlp_hidden_mult: default_policy_mlp_hidden_mult(),
            rotary_embedding: RotaryEmbedding::default(),
            grid_positional: SudokuGridPositional::default(),
            grid_rope_theta: default_grid_rope_theta(),
            dropout: default_dropout(),
            fused_kernels: false,
            relu_threshold: 0.0,
            cache_mhc: SudokuCacheMhcConfig::default(),
            cache_update: SudokuCacheUpdateConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SudokuArtifactConfig {
    #[serde(default)]
    pub output: burn_dragon_train::VisionArtifactOutputMode,
    #[serde(default = "default_artifact_fps")]
    pub fps: u32,
    #[serde(default = "default_artifact_samples")]
    pub max_samples: usize,
    #[serde(default = "default_artifact_sample_policy")]
    pub sample_policy: bool,
    pub overwrite: bool,
}

impl Default for SudokuArtifactConfig {
    fn default() -> Self {
        Self {
            output: burn_dragon_train::VisionArtifactOutputMode::Mp4,
            fps: default_artifact_fps(),
            max_samples: default_artifact_samples(),
            sample_policy: default_artifact_sample_policy(),
            overwrite: true,
        }
    }
}
