use serde::{Deserialize, Serialize};

use burn_dragon_core::{
    AttentionResidualConfig, BdhInitializationConfig, BlockAttentionResidualConfig,
    ClockedSlowMemoryConfig, DragonNormConfig, LatentFanoutScheduleConfig, MambaSequenceConfig,
    ManifoldHyperConnectionsConfig, ResidualConnectorKind, RotaryEmbedding, SequenceKernelKind,
    SummaryMemoryConfig, YNeuronRecurrenceConfig,
};

pub mod ttcl;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Default)]
pub struct ModelOverrides {
    pub n_layer: Option<usize>,
    pub n_embd: Option<usize>,
    pub n_head: Option<usize>,
    pub mlp_internal_dim_multiplier: Option<usize>,
    #[serde(alias = "neuron_space_dim")]
    pub latent_total: Option<usize>,
    #[serde(alias = "init")]
    pub initialization: Option<BdhInitializationConfig>,
    pub sequence_kernel: Option<SequenceKernelKind>,
    pub mamba: Option<MambaSequenceConfig>,
    pub residual_connector: Option<ResidualConnectorKind>,
    pub attention_residual: Option<AttentionResidualConfig>,
    pub block_attention_residual: Option<BlockAttentionResidualConfig>,
    pub latent_fanout_schedule: Option<LatentFanoutScheduleConfig>,
    pub relu_threshold: Option<f32>,
    pub dropout: Option<f64>,
    pub normalization: Option<DragonNormConfig>,
    pub fused_kernels: Option<bool>,
    pub block_size: Option<usize>,
    #[serde(alias = "rollout_fast_steps")]
    pub rollout_fast_steps_per_slow_step: Option<usize>,
    pub rotary_embedding: Option<RotaryEmbedding>,
    #[serde(alias = "y_sparse_recurrence")]
    pub y_neuron_recurrence: Option<YNeuronRecurrenceConfig>,
    pub clocked_slow_memory: Option<ClockedSlowMemoryConfig>,
    pub summary_memory: Option<SummaryMemoryConfig>,
    pub mhc: Option<ManifoldHyperConnectionsConfig>,
}

pub use ttcl::{
    AccuracyByObjectCount, CheckpointEvaluationSummary, CheckpointSelectionMode,
    CheckpointSelectionSummary, DeductionEpisodesConfig, GeneratedPermutationTransferData,
    PermutationEpisode, PermutationExample, PermutationRenderConfig, PermutationTaskKind,
    PermutationTransferExperimentConfig, PermutationTransferRunSummary, ProtocolDifficultySummary,
    ProtocolEpisodeMetrics, ProtocolSourceDifficultySummary, ProtocolSummary,
    SourceCheckpointConfig, SourceHoldoutConfig, SupportRewrite, TrackingCorpusConfig,
    TtclProtocolConfig, TtclProtocolMode, TtclTrainingConfig, derive_render_config,
    generate_permutation_transfer_data, load_permutation_transfer_experiment_config,
    render_run_summary_markdown, resolve_ttcl_output_dir, summarize_protocols,
    write_example_corpus, write_generated_transfer_data,
};
