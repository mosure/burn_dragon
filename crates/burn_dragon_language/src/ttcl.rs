#![cfg(feature = "train")]

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};

use crate::ModelOverrides;

fn default_true() -> bool {
    true
}

fn default_seed() -> u64 {
    1337
}

fn default_checkpoint_interval_iters() -> usize {
    1
}

fn default_log_frequency() -> usize {
    1
}

fn default_base_batch_size() -> usize {
    8
}

fn default_adaptation_batch_size() -> usize {
    4
}

fn default_base_learning_rate() -> f64 {
    1e-3
}

fn default_adaptation_learning_rate() -> f64 {
    5e-4
}

fn default_extra_clues() -> usize {
    1
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TtclProtocolMode {
    ZeroShot,
    InContextOnly,
    ResetPerEpisode,
    Continual,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum SupportRewrite {
    #[default]
    None,
    RenameObjects,
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PermutationTaskKind {
    TrackingShuffledObjects,
    LogicalDeduction,
}

impl PermutationTaskKind {
    fn mode_char(self) -> char {
        match self {
            Self::TrackingShuffledObjects => 't',
            Self::LogicalDeduction => 'd',
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct PermutationRenderConfig {
    pub render_object_count: usize,
    pub relation_slots: usize,
    pub operation_slots: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TtclTrainingConfig {
    #[serde(default = "default_base_batch_size")]
    pub base_batch_size: usize,
    #[serde(default = "default_adaptation_batch_size")]
    pub adaptation_batch_size: usize,
    pub base_max_iters: usize,
    pub source_stream_max_iters: usize,
    #[serde(default = "default_base_learning_rate")]
    pub base_learning_rate: f64,
    #[serde(default = "default_adaptation_learning_rate")]
    pub adaptation_learning_rate: f64,
    #[serde(default = "default_checkpoint_interval_iters")]
    pub checkpoint_interval_iters: usize,
    #[serde(default = "default_log_frequency")]
    pub log_frequency: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TrackingCorpusConfig {
    pub object_counts: Vec<usize>,
    pub examples_per_count: usize,
    pub min_swaps: usize,
    pub max_swaps: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct SourceHoldoutConfig {
    pub object_counts: Vec<usize>,
    pub examples_per_count: usize,
    pub min_swaps: usize,
    pub max_swaps: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct DeductionEpisodesConfig {
    pub object_counts: Vec<usize>,
    pub episodes_per_count: usize,
    pub support_pool_size: usize,
    pub query_examples_per_episode: usize,
    #[serde(default = "default_extra_clues")]
    pub extra_clues: usize,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct TtclProtocolConfig {
    pub name: String,
    pub mode: TtclProtocolMode,
    #[serde(default)]
    pub support_examples: usize,
    #[serde(default)]
    pub gradient_steps: usize,
    #[serde(default)]
    pub rewrite: SupportRewrite,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PermutationTransferExperimentConfig {
    pub name: String,
    pub output_dir: PathBuf,
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default = "default_true")]
    pub train_source_stream: bool,
    pub model: ModelOverrides,
    pub training: TtclTrainingConfig,
    pub source_pretrain: TrackingCorpusConfig,
    pub source_stream: TrackingCorpusConfig,
    pub source_holdout: SourceHoldoutConfig,
    pub target: DeductionEpisodesConfig,
    pub protocols: Vec<TtclProtocolConfig>,
}

impl TtclTrainingConfig {
    fn validate(&self) -> Result<()> {
        if self.base_batch_size == 0 {
            return Err(anyhow!("training.base_batch_size must be > 0"));
        }
        if self.adaptation_batch_size == 0 {
            return Err(anyhow!("training.adaptation_batch_size must be > 0"));
        }
        if self.base_max_iters == 0 {
            return Err(anyhow!("training.base_max_iters must be > 0"));
        }
        if self.source_stream_max_iters == 0 {
            return Err(anyhow!("training.source_stream_max_iters must be > 0"));
        }
        if self.base_learning_rate <= 0.0 {
            return Err(anyhow!("training.base_learning_rate must be > 0"));
        }
        if self.adaptation_learning_rate <= 0.0 {
            return Err(anyhow!("training.adaptation_learning_rate must be > 0"));
        }
        if self.checkpoint_interval_iters == 0 {
            return Err(anyhow!("training.checkpoint_interval_iters must be > 0"));
        }
        if self.log_frequency == 0 {
            return Err(anyhow!("training.log_frequency must be > 0"));
        }
        Ok(())
    }
}

impl TrackingCorpusConfig {
    fn validate(&self, label: &str) -> Result<()> {
        validate_object_counts(&self.object_counts, label)?;
        if self.examples_per_count == 0 {
            return Err(anyhow!("{label}.examples_per_count must be > 0"));
        }
        if self.max_swaps == 0 {
            return Err(anyhow!("{label}.max_swaps must be > 0"));
        }
        if self.min_swaps == 0 {
            return Err(anyhow!("{label}.min_swaps must be > 0"));
        }
        if self.min_swaps > self.max_swaps {
            return Err(anyhow!("{label}.min_swaps must be <= {label}.max_swaps"));
        }
        Ok(())
    }
}

impl SourceHoldoutConfig {
    fn validate(&self) -> Result<()> {
        validate_object_counts(&self.object_counts, "source_holdout.object_counts")?;
        if self.examples_per_count == 0 {
            return Err(anyhow!("source_holdout.examples_per_count must be > 0"));
        }
        if self.max_swaps == 0 || self.min_swaps == 0 || self.min_swaps > self.max_swaps {
            return Err(anyhow!(
                "source_holdout requires 0 < min_swaps <= max_swaps"
            ));
        }
        Ok(())
    }
}

impl DeductionEpisodesConfig {
    fn validate(&self) -> Result<()> {
        validate_object_counts(&self.object_counts, "target.object_counts")?;
        if self.episodes_per_count == 0 {
            return Err(anyhow!("target.episodes_per_count must be > 0"));
        }
        if self.support_pool_size == 0 {
            return Err(anyhow!("target.support_pool_size must be > 0"));
        }
        if self.query_examples_per_episode == 0 {
            return Err(anyhow!("target.query_examples_per_episode must be > 0"));
        }
        Ok(())
    }
}

impl TtclProtocolConfig {
    fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(anyhow!("protocol name must not be empty"));
        }
        match self.mode {
            TtclProtocolMode::ZeroShot => {
                if self.support_examples != 0 || self.gradient_steps != 0 {
                    return Err(anyhow!(
                        "protocol `{}` zero_shot must use support_examples=0 and gradient_steps=0",
                        self.name
                    ));
                }
            }
            TtclProtocolMode::InContextOnly => {
                if self.support_examples == 0 {
                    return Err(anyhow!(
                        "protocol `{}` in_context_only requires support_examples > 0",
                        self.name
                    ));
                }
                if self.gradient_steps != 0 {
                    return Err(anyhow!(
                        "protocol `{}` in_context_only requires gradient_steps=0",
                        self.name
                    ));
                }
            }
            TtclProtocolMode::ResetPerEpisode | TtclProtocolMode::Continual => {
                if self.support_examples == 0 {
                    return Err(anyhow!(
                        "protocol `{}` requires support_examples > 0",
                        self.name
                    ));
                }
                if self.gradient_steps == 0 {
                    return Err(anyhow!(
                        "protocol `{}` requires gradient_steps > 0",
                        self.name
                    ));
                }
            }
        }
        Ok(())
    }
}

impl PermutationTransferExperimentConfig {
    pub fn validate(&self) -> Result<()> {
        if self.name.trim().is_empty() {
            return Err(anyhow!("experiment name must not be empty"));
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(anyhow!("output_dir must not be empty"));
        }
        self.training.validate()?;
        self.source_pretrain.validate("source_pretrain")?;
        self.source_stream.validate("source_stream")?;
        self.source_holdout.validate()?;
        self.target.validate()?;
        if self.protocols.is_empty() {
            return Err(anyhow!("at least one protocol is required"));
        }
        for protocol in &self.protocols {
            protocol.validate()?;
            if protocol.support_examples > self.target.support_pool_size {
                return Err(anyhow!(
                    "protocol `{}` support_examples={} exceeds target.support_pool_size={}",
                    protocol.name,
                    protocol.support_examples,
                    self.target.support_pool_size
                ));
            }
        }
        Ok(())
    }
}

fn validate_object_counts(object_counts: &[usize], label: &str) -> Result<()> {
    if object_counts.is_empty() {
        return Err(anyhow!("{label} must not be empty"));
    }
    for count in object_counts {
        if *count < 2 {
            return Err(anyhow!("{label} entries must be >= 2 (got {count})"));
        }
        if *count > 9 {
            return Err(anyhow!(
                "{label} entries must be <= 9 so answers stay single-token digits (got {count})"
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PermutationExample {
    pub task: PermutationTaskKind,
    pub object_count: usize,
    pub render: PermutationRenderConfig,
    pub init_order: String,
    pub relations: Vec<String>,
    pub operations: Vec<String>,
    pub query: char,
    pub answer_rank: usize,
    pub solution_order: String,
}

impl PermutationExample {
    pub fn prompt(&self) -> String {
        format!(
            "perm|m={}|n={}|init={}|rel={}|ops={}|q={}|a=",
            self.task.mode_char(),
            self.object_count,
            self.rendered_init(),
            self.rendered_relations().join(","),
            self.rendered_operations().join(","),
            self.query
        )
    }

    pub fn answer(&self) -> String {
        self.answer_rank.to_string()
    }

    pub fn document(&self) -> String {
        format!("{}{}\n", self.prompt(), self.answer())
    }

    pub fn renamed(&self, rng: &mut StdRng) -> Self {
        let active = active_letters(self.object_count);
        let mut remapped = active.clone();
        remapped.shuffle(rng);
        let renamed = |value: char| {
            active
                .iter()
                .position(|candidate| *candidate == value)
                .map(|index| remapped[index])
                .unwrap_or(value)
        };
        let rename_string = |value: &str| value.chars().map(renamed).collect::<String>();
        Self {
            task: self.task,
            object_count: self.object_count,
            render: self.render,
            init_order: rename_string(&self.init_order),
            relations: self
                .relations
                .iter()
                .map(|item| rename_string(item))
                .collect(),
            operations: self
                .operations
                .iter()
                .map(|item| rename_string(item))
                .collect(),
            query: renamed(self.query),
            answer_rank: self.answer_rank,
            solution_order: rename_string(&self.solution_order),
        }
    }

    fn rendered_init(&self) -> String {
        if self.init_order.is_empty() {
            "_".repeat(self.render.render_object_count)
        } else {
            pad_field(&self.init_order, self.render.render_object_count, '_')
        }
    }

    fn rendered_relations(&self) -> Vec<String> {
        let mut items = self.relations.clone();
        items.resize(self.render.relation_slots, "_<_".to_string());
        items
    }

    fn rendered_operations(&self) -> Vec<String> {
        let mut items = self.operations.clone();
        items.resize(self.render.operation_slots, "__".to_string());
        items
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq, Eq)]
pub struct PermutationEpisode {
    pub episode_id: String,
    pub object_count: usize,
    pub support_pool: Vec<PermutationExample>,
    pub queries: Vec<PermutationExample>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct GeneratedPermutationTransferData {
    pub render: PermutationRenderConfig,
    pub source_pretrain: Vec<PermutationExample>,
    pub source_stream: Vec<PermutationExample>,
    pub source_holdout: Vec<PermutationExample>,
    pub target_episodes: Vec<PermutationEpisode>,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ProtocolEpisodeMetrics {
    pub protocol_name: String,
    pub mode: TtclProtocolMode,
    pub episode_id: String,
    pub episode_index: usize,
    pub object_count: usize,
    pub support_examples: usize,
    pub gradient_steps: usize,
    pub query_accuracy_source_checkpoint: f64,
    pub query_accuracy_before: f64,
    pub query_accuracy_after: f64,
    pub source_accuracy_source_checkpoint: f64,
    pub source_accuracy_before: f64,
    pub source_accuracy_after: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ProtocolSummary {
    pub protocol_name: String,
    pub mode: TtclProtocolMode,
    pub support_examples: usize,
    pub gradient_steps: usize,
    pub episodes: usize,
    pub mean_query_accuracy_before: f64,
    pub mean_query_accuracy_after: f64,
    pub mean_query_delta: f64,
    pub mean_query_delta_vs_source_checkpoint: f64,
    pub mean_source_accuracy_before: f64,
    pub mean_source_accuracy_after: f64,
    pub mean_source_delta: f64,
    pub mean_source_delta_vs_source_checkpoint: f64,
    pub max_source_forgetting_vs_source_checkpoint: f64,
    pub final_query_accuracy_after: f64,
    pub final_query_delta_vs_source_checkpoint: f64,
    pub final_source_accuracy_after: f64,
    pub final_source_delta_vs_source_checkpoint: f64,
    pub final_source_retention_ratio: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ProtocolDifficultySummary {
    pub protocol_name: String,
    pub mode: TtclProtocolMode,
    pub object_count: usize,
    pub episodes: usize,
    pub mean_query_accuracy_source_checkpoint: f64,
    pub mean_query_accuracy_before: f64,
    pub mean_query_accuracy_after: f64,
    pub mean_query_delta: f64,
    pub mean_query_delta_vs_source_checkpoint: f64,
    pub mean_source_accuracy_source_checkpoint: f64,
    pub mean_source_accuracy_before: f64,
    pub mean_source_accuracy_after: f64,
    pub mean_source_delta: f64,
    pub mean_source_delta_vs_source_checkpoint: f64,
    pub max_source_forgetting_vs_source_checkpoint: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct PermutationTransferRunSummary {
    pub experiment_name: String,
    pub output_dir: PathBuf,
    pub seed: u64,
    pub backend: String,
    pub train_source_stream: bool,
    pub render: PermutationRenderConfig,
    pub base_block_size: usize,
    pub source_pretrain_examples: usize,
    pub source_stream_examples: usize,
    pub source_holdout_examples: usize,
    pub target_episodes: usize,
    pub source_pretrain_checkpoint_dir: PathBuf,
    pub source_pretrain_checkpoint_epoch: usize,
    pub source_stream_checkpoint_dir: PathBuf,
    pub source_stream_checkpoint_epoch: usize,
    pub protocol_summaries: Vec<ProtocolSummary>,
    pub difficulty_summaries: Vec<ProtocolDifficultySummary>,
    pub episode_metrics: Vec<ProtocolEpisodeMetrics>,
}

pub fn load_permutation_transfer_experiment_config(
    path: &Path,
) -> Result<PermutationTransferExperimentConfig> {
    let payload =
        fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))?;
    let config: PermutationTransferExperimentConfig =
        toml::from_str(&payload).with_context(|| format!("failed to parse {}", path.display()))?;
    config.validate()?;
    Ok(config)
}

pub fn resolve_ttcl_output_dir(config_path: &Path, output_dir: &Path) -> PathBuf {
    if output_dir.is_absolute() {
        output_dir.to_path_buf()
    } else {
        let _ = config_path;
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(output_dir)
    }
}

pub fn derive_render_config(
    config: &PermutationTransferExperimentConfig,
) -> PermutationRenderConfig {
    let render_object_count = config
        .source_pretrain
        .object_counts
        .iter()
        .chain(config.source_stream.object_counts.iter())
        .chain(config.source_holdout.object_counts.iter())
        .chain(config.target.object_counts.iter())
        .copied()
        .max()
        .unwrap_or(2);
    let operation_slots = config
        .source_pretrain
        .max_swaps
        .max(config.source_stream.max_swaps)
        .max(config.source_holdout.max_swaps);
    let relation_slots = render_object_count
        .saturating_sub(1)
        .saturating_add(config.target.extra_clues);
    PermutationRenderConfig {
        render_object_count,
        relation_slots,
        operation_slots,
    }
}

pub fn generate_permutation_transfer_data(
    config: &PermutationTransferExperimentConfig,
) -> GeneratedPermutationTransferData {
    let render = derive_render_config(config);
    let mut rng = StdRng::seed_from_u64(config.seed);
    let source_pretrain = generate_tracking_corpus(&config.source_pretrain, render, &mut rng);
    let source_stream = generate_tracking_corpus(&config.source_stream, render, &mut rng);
    let source_holdout = generate_tracking_holdout(&config.source_holdout, render, &mut rng);
    let target_episodes = generate_deduction_episodes(&config.target, render, &mut rng);
    GeneratedPermutationTransferData {
        render,
        source_pretrain,
        source_stream,
        source_holdout,
        target_episodes,
    }
}

pub fn write_example_corpus(path: &Path, examples: &[PermutationExample]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let payload = examples
        .iter()
        .map(PermutationExample::document)
        .collect::<String>();
    fs::write(path, payload).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(())
}

pub fn write_generated_transfer_data(
    output_dir: &Path,
    data: &GeneratedPermutationTransferData,
) -> Result<()> {
    fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    write_example_corpus(
        &output_dir.join("source_pretrain.txt"),
        &data.source_pretrain,
    )?;
    write_example_corpus(&output_dir.join("source_stream.txt"), &data.source_stream)?;
    write_example_corpus(&output_dir.join("source_holdout.txt"), &data.source_holdout)?;
    for episode in &data.target_episodes {
        let support_path = output_dir
            .join("episodes")
            .join(&episode.episode_id)
            .join("support_pool.txt");
        let query_path = output_dir
            .join("episodes")
            .join(&episode.episode_id)
            .join("queries.txt");
        write_example_corpus(&support_path, &episode.support_pool)?;
        write_example_corpus(&query_path, &episode.queries)?;
    }
    let manifest_path = output_dir.join("generated_transfer_data.json");
    fs::write(
        &manifest_path,
        serde_json::to_string_pretty(data).context("serialize generated transfer data")?,
    )
    .with_context(|| format!("failed to write {}", manifest_path.display()))?;
    Ok(())
}

pub fn summarize_protocols(
    summary: &mut PermutationTransferRunSummary,
    metrics: Vec<ProtocolEpisodeMetrics>,
    protocols: &[TtclProtocolConfig],
) {
    summary.episode_metrics = metrics;
    summary.episode_metrics.sort_by(|lhs, rhs| {
        lhs.protocol_name
            .cmp(&rhs.protocol_name)
            .then(lhs.episode_index.cmp(&rhs.episode_index))
            .then(lhs.episode_id.cmp(&rhs.episode_id))
    });
    summary.protocol_summaries = protocols
        .iter()
        .map(|protocol| {
            let rows = summary
                .episode_metrics
                .iter()
                .filter(|row| row.protocol_name == protocol.name)
                .collect::<Vec<_>>();
            let last_row = rows.iter().copied().max_by_key(|row| row.episode_index);
            ProtocolSummary {
                protocol_name: protocol.name.clone(),
                mode: protocol.mode,
                support_examples: protocol.support_examples,
                gradient_steps: protocol.gradient_steps,
                episodes: rows.len(),
                mean_query_accuracy_before: mean_metric(&rows, |row| row.query_accuracy_before),
                mean_query_accuracy_after: mean_metric(&rows, |row| row.query_accuracy_after),
                mean_query_delta: mean_metric(&rows, |row| {
                    row.query_accuracy_after - row.query_accuracy_before
                }),
                mean_query_delta_vs_source_checkpoint: mean_metric(&rows, |row| {
                    row.query_accuracy_after - row.query_accuracy_source_checkpoint
                }),
                mean_source_accuracy_before: mean_metric(&rows, |row| row.source_accuracy_before),
                mean_source_accuracy_after: mean_metric(&rows, |row| row.source_accuracy_after),
                mean_source_delta: mean_metric(&rows, |row| {
                    row.source_accuracy_after - row.source_accuracy_before
                }),
                mean_source_delta_vs_source_checkpoint: mean_metric(&rows, |row| {
                    row.source_accuracy_after - row.source_accuracy_source_checkpoint
                }),
                max_source_forgetting_vs_source_checkpoint: rows
                    .iter()
                    .map(|row| row.source_accuracy_after - row.source_accuracy_source_checkpoint)
                    .fold(0.0, |worst, delta| worst.min(delta)),
                final_query_accuracy_after: last_row.map_or(0.0, |row| row.query_accuracy_after),
                final_query_delta_vs_source_checkpoint: last_row.map_or(0.0, |row| {
                    row.query_accuracy_after - row.query_accuracy_source_checkpoint
                }),
                final_source_accuracy_after: last_row.map_or(0.0, |row| row.source_accuracy_after),
                final_source_delta_vs_source_checkpoint: last_row.map_or(0.0, |row| {
                    row.source_accuracy_after - row.source_accuracy_source_checkpoint
                }),
                final_source_retention_ratio: last_row.map_or(0.0, |row| {
                    retention_ratio(
                        row.source_accuracy_after,
                        row.source_accuracy_source_checkpoint,
                    )
                }),
            }
        })
        .collect();
    summary.difficulty_summaries = protocols
        .iter()
        .flat_map(|protocol| {
            let mut object_counts = summary
                .episode_metrics
                .iter()
                .filter(|row| row.protocol_name == protocol.name)
                .map(|row| row.object_count)
                .collect::<Vec<_>>();
            object_counts.sort_unstable();
            object_counts.dedup();
            object_counts
                .into_iter()
                .map(|object_count| {
                    let rows = summary
                        .episode_metrics
                        .iter()
                        .filter(|row| {
                            row.protocol_name == protocol.name && row.object_count == object_count
                        })
                        .collect::<Vec<_>>();
                    ProtocolDifficultySummary {
                        protocol_name: protocol.name.clone(),
                        mode: protocol.mode,
                        object_count,
                        episodes: rows.len(),
                        mean_query_accuracy_source_checkpoint: mean_metric(&rows, |row| {
                            row.query_accuracy_source_checkpoint
                        }),
                        mean_query_accuracy_before: mean_metric(&rows, |row| {
                            row.query_accuracy_before
                        }),
                        mean_query_accuracy_after: mean_metric(&rows, |row| {
                            row.query_accuracy_after
                        }),
                        mean_query_delta: mean_metric(&rows, |row| {
                            row.query_accuracy_after - row.query_accuracy_before
                        }),
                        mean_query_delta_vs_source_checkpoint: mean_metric(&rows, |row| {
                            row.query_accuracy_after - row.query_accuracy_source_checkpoint
                        }),
                        mean_source_accuracy_source_checkpoint: mean_metric(&rows, |row| {
                            row.source_accuracy_source_checkpoint
                        }),
                        mean_source_accuracy_before: mean_metric(&rows, |row| {
                            row.source_accuracy_before
                        }),
                        mean_source_accuracy_after: mean_metric(&rows, |row| {
                            row.source_accuracy_after
                        }),
                        mean_source_delta: mean_metric(&rows, |row| {
                            row.source_accuracy_after - row.source_accuracy_before
                        }),
                        mean_source_delta_vs_source_checkpoint: mean_metric(&rows, |row| {
                            row.source_accuracy_after - row.source_accuracy_source_checkpoint
                        }),
                        max_source_forgetting_vs_source_checkpoint: rows
                            .iter()
                            .map(|row| {
                                row.source_accuracy_after - row.source_accuracy_source_checkpoint
                            })
                            .fold(0.0, |worst, delta| worst.min(delta)),
                    }
                })
                .collect::<Vec<_>>()
        })
        .collect();
}

pub fn render_run_summary_markdown(summary: &PermutationTransferRunSummary) -> String {
    let mut out = String::new();
    out.push_str("# Permutation Transfer TTCL Summary\n\n");
    out.push_str(&format!("Experiment: `{}`\n\n", summary.experiment_name));
    out.push_str("## Setup\n\n");
    out.push_str(&format!(
        "- output_dir: `{}`\n- seed: `{}`\n- backend: `{}`\n- train_source_stream: `{}`\n- render_object_count: `{}`\n- relation_slots: `{}`\n- operation_slots: `{}`\n- block_size: `{}`\n- source_pretrain_examples: `{}`\n- source_stream_examples: `{}`\n- source_holdout_examples: `{}`\n- target_episodes: `{}`\n- source_pretrain_checkpoint: `{}` @ epoch `{}`\n- source_stream_checkpoint: `{}` @ epoch `{}`\n\n",
        summary.output_dir.display(),
        summary.seed,
        summary.backend,
        summary.train_source_stream,
        summary.render.render_object_count,
        summary.render.relation_slots,
        summary.render.operation_slots,
        summary.base_block_size,
        summary.source_pretrain_examples,
        summary.source_stream_examples,
        summary.source_holdout_examples,
        summary.target_episodes,
        summary.source_pretrain_checkpoint_dir.display(),
        summary.source_pretrain_checkpoint_epoch,
        summary.source_stream_checkpoint_dir.display(),
        summary.source_stream_checkpoint_epoch,
    ));
    out.push_str("## Protocols\n\n");
    out.push_str("| Protocol | Mode | Support | Steps | Episodes | Query Before | Query After | Query Delta | Query vs Src Ckpt | Source Before | Source After | Source Delta | Source vs Src Ckpt | Max Source Forget | Final Src Retention |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for protocol in &summary.protocol_summaries {
        out.push_str(&format!(
            "| {} | {:?} | {} | {} | {} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:+.4} | {:.4} |\n",
            protocol.protocol_name,
            protocol.mode,
            protocol.support_examples,
            protocol.gradient_steps,
            protocol.episodes,
            protocol.mean_query_accuracy_before,
            protocol.mean_query_accuracy_after,
            protocol.mean_query_delta,
            protocol.mean_query_delta_vs_source_checkpoint,
            protocol.mean_source_accuracy_before,
            protocol.mean_source_accuracy_after,
            protocol.mean_source_delta,
            protocol.mean_source_delta_vs_source_checkpoint,
            protocol.max_source_forgetting_vs_source_checkpoint,
            protocol.final_source_retention_ratio,
        ));
    }
    out.push_str("\n## Difficulty Breakdown\n\n");
    out.push_str("| Protocol | Objects | Episodes | Query Src Ckpt | Query Before | Query After | Query Delta | Query vs Src Ckpt | Source Src Ckpt | Source Before | Source After | Source Delta | Source vs Src Ckpt | Max Source Forget |\n");
    out.push_str(
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n",
    );
    for row in &summary.difficulty_summaries {
        out.push_str(&format!(
            "| {} | {} | {} | {:.4} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:.4} | {:.4} | {:.4} | {:+.4} | {:+.4} | {:+.4} |\n",
            row.protocol_name,
            row.object_count,
            row.episodes,
            row.mean_query_accuracy_source_checkpoint,
            row.mean_query_accuracy_before,
            row.mean_query_accuracy_after,
            row.mean_query_delta,
            row.mean_query_delta_vs_source_checkpoint,
            row.mean_source_accuracy_source_checkpoint,
            row.mean_source_accuracy_before,
            row.mean_source_accuracy_after,
            row.mean_source_delta,
            row.mean_source_delta_vs_source_checkpoint,
            row.max_source_forgetting_vs_source_checkpoint,
        ));
    }
    out.push_str("\n## Episodes\n\n");
    out.push_str("| Protocol | Ep# | Episode | Objects | Support | Steps | Query Src Ckpt | Query Before | Query After | Source Src Ckpt | Source Before | Source After |\n");
    out.push_str("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n");
    for row in &summary.episode_metrics {
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} | {} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} | {:.4} |\n",
            row.protocol_name,
            row.episode_index,
            row.episode_id,
            row.object_count,
            row.support_examples,
            row.gradient_steps,
            row.query_accuracy_source_checkpoint,
            row.query_accuracy_before,
            row.query_accuracy_after,
            row.source_accuracy_source_checkpoint,
            row.source_accuracy_before,
            row.source_accuracy_after
        ));
    }
    out
}

fn mean_metric(
    rows: &[&ProtocolEpisodeMetrics],
    mapper: impl Fn(&ProtocolEpisodeMetrics) -> f64,
) -> f64 {
    if rows.is_empty() {
        0.0
    } else {
        rows.iter().map(|row| mapper(row)).sum::<f64>() / rows.len() as f64
    }
}

fn retention_ratio(value: f64, anchor: f64) -> f64 {
    if anchor <= 0.0 { 0.0 } else { value / anchor }
}

fn generate_tracking_corpus(
    config: &TrackingCorpusConfig,
    render: PermutationRenderConfig,
    rng: &mut StdRng,
) -> Vec<PermutationExample> {
    let mut examples = Vec::new();
    for object_count in &config.object_counts {
        for _ in 0..config.examples_per_count {
            examples.push(generate_tracking_example(
                *object_count,
                render,
                config.min_swaps,
                config.max_swaps,
                rng,
            ));
        }
    }
    examples
}

fn generate_tracking_holdout(
    config: &SourceHoldoutConfig,
    render: PermutationRenderConfig,
    rng: &mut StdRng,
) -> Vec<PermutationExample> {
    let corpus = TrackingCorpusConfig {
        object_counts: config.object_counts.clone(),
        examples_per_count: config.examples_per_count,
        min_swaps: config.min_swaps,
        max_swaps: config.max_swaps,
    };
    generate_tracking_corpus(&corpus, render, rng)
}

fn generate_deduction_episodes(
    config: &DeductionEpisodesConfig,
    render: PermutationRenderConfig,
    rng: &mut StdRng,
) -> Vec<PermutationEpisode> {
    let mut episodes = Vec::new();
    for object_count in &config.object_counts {
        for episode_index in 0..config.episodes_per_count {
            let support_pool = (0..config.support_pool_size)
                .map(|_| generate_deduction_example(*object_count, render, config.extra_clues, rng))
                .collect::<Vec<_>>();
            let queries = (0..config.query_examples_per_episode)
                .map(|_| generate_deduction_example(*object_count, render, config.extra_clues, rng))
                .collect::<Vec<_>>();
            episodes.push(PermutationEpisode {
                episode_id: format!("logical_deduction_n{}_ep{:02}", object_count, episode_index),
                object_count: *object_count,
                support_pool,
                queries,
            });
        }
    }
    episodes
}

fn active_letters(object_count: usize) -> Vec<char> {
    (0..object_count)
        .map(|index| (b'A' + index as u8) as char)
        .collect()
}

fn pad_field(value: &str, width: usize, fill: char) -> String {
    let mut out = value.to_string();
    while out.len() < width {
        out.push(fill);
    }
    out
}

fn generate_tracking_example(
    object_count: usize,
    render: PermutationRenderConfig,
    min_swaps: usize,
    max_swaps: usize,
    rng: &mut StdRng,
) -> PermutationExample {
    let letters = active_letters(object_count);
    let mut current_order = letters.clone();
    let swap_count =
        rng.gen_range(min_swaps..=max_swaps.min(render.operation_slots).max(min_swaps));
    let mut operations = Vec::with_capacity(swap_count);
    for _ in 0..swap_count {
        let lhs_index = rng.gen_range(0..object_count);
        let mut rhs_index = rng.gen_range(0..object_count);
        while rhs_index == lhs_index {
            rhs_index = rng.gen_range(0..object_count);
        }
        let lhs = letters[lhs_index];
        let rhs = letters[rhs_index];
        operations.push(format!("{lhs}{rhs}"));
        let lhs_pos = current_order
            .iter()
            .position(|candidate| *candidate == lhs)
            .expect("lhs in order");
        let rhs_pos = current_order
            .iter()
            .position(|candidate| *candidate == rhs)
            .expect("rhs in order");
        current_order.swap(lhs_pos, rhs_pos);
    }
    let query = *letters.choose(rng).expect("query letter");
    let answer_rank = current_order
        .iter()
        .position(|candidate| *candidate == query)
        .expect("query in final order")
        + 1;
    PermutationExample {
        task: PermutationTaskKind::TrackingShuffledObjects,
        object_count,
        render,
        init_order: letters.iter().collect(),
        relations: Vec::new(),
        operations,
        query,
        answer_rank,
        solution_order: current_order.iter().collect(),
    }
}

fn generate_deduction_example(
    object_count: usize,
    render: PermutationRenderConfig,
    extra_clues: usize,
    rng: &mut StdRng,
) -> PermutationExample {
    let mut order = active_letters(object_count);
    order.shuffle(rng);
    let mut relations = order
        .windows(2)
        .map(|window| format!("{}<{}", window[0], window[1]))
        .collect::<Vec<_>>();
    let mut extra_candidates = Vec::new();
    for lhs_index in 0..object_count {
        for rhs_index in (lhs_index + 2)..object_count {
            extra_candidates.push(format!("{}<{}", order[lhs_index], order[rhs_index]));
        }
    }
    extra_candidates.shuffle(rng);
    relations.extend(extra_candidates.into_iter().take(extra_clues));
    relations.shuffle(rng);
    relations.truncate(render.relation_slots);
    let query = *order.choose(rng).expect("query letter");
    let answer_rank = order
        .iter()
        .position(|candidate| *candidate == query)
        .expect("query in order")
        + 1;
    PermutationExample {
        task: PermutationTaskKind::LogicalDeduction,
        object_count,
        render,
        init_order: String::new(),
        relations,
        operations: Vec::new(),
        query,
        answer_rank,
        solution_order: order.iter().collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn tiny_config() -> PermutationTransferExperimentConfig {
        PermutationTransferExperimentConfig {
            name: "perm_ttcl_smoke".to_string(),
            output_dir: "artifacts/language/perm_ttcl_smoke".into(),
            seed: 7,
            train_source_stream: true,
            model: ModelOverrides {
                n_layer: Some(2),
                n_embd: Some(32),
                n_head: Some(4),
                mlp_internal_dim_multiplier: Some(2),
                dropout: Some(0.0),
                ..Default::default()
            },
            training: TtclTrainingConfig {
                base_batch_size: 4,
                adaptation_batch_size: 2,
                base_max_iters: 4,
                source_stream_max_iters: 2,
                base_learning_rate: 1e-3,
                adaptation_learning_rate: 5e-4,
                checkpoint_interval_iters: 1,
                log_frequency: 1,
            },
            source_pretrain: TrackingCorpusConfig {
                object_counts: vec![3, 5],
                examples_per_count: 4,
                min_swaps: 2,
                max_swaps: 5,
            },
            source_stream: TrackingCorpusConfig {
                object_counts: vec![5, 7],
                examples_per_count: 3,
                min_swaps: 3,
                max_swaps: 5,
            },
            source_holdout: SourceHoldoutConfig {
                object_counts: vec![3, 5, 7],
                examples_per_count: 2,
                min_swaps: 2,
                max_swaps: 5,
            },
            target: DeductionEpisodesConfig {
                object_counts: vec![3, 5, 7],
                episodes_per_count: 2,
                support_pool_size: 4,
                query_examples_per_episode: 3,
                extra_clues: 1,
            },
            protocols: vec![
                TtclProtocolConfig {
                    name: "zero".to_string(),
                    mode: TtclProtocolMode::ZeroShot,
                    support_examples: 0,
                    gradient_steps: 0,
                    rewrite: SupportRewrite::None,
                },
                TtclProtocolConfig {
                    name: "reset_4x1".to_string(),
                    mode: TtclProtocolMode::ResetPerEpisode,
                    support_examples: 4,
                    gradient_steps: 1,
                    rewrite: SupportRewrite::RenameObjects,
                },
            ],
        }
    }

    #[test]
    fn generator_produces_fixed_width_documents_across_tasks() {
        let config = tiny_config();
        config.validate().expect("valid config");
        let data = generate_permutation_transfer_data(&config);
        let mut lengths = data
            .source_pretrain
            .iter()
            .chain(data.source_stream.iter())
            .chain(data.source_holdout.iter())
            .chain(
                data.target_episodes
                    .iter()
                    .flat_map(|episode| episode.support_pool.iter()),
            )
            .chain(
                data.target_episodes
                    .iter()
                    .flat_map(|episode| episode.queries.iter()),
            )
            .map(|example| example.document().len())
            .collect::<Vec<_>>();
        lengths.sort_unstable();
        lengths.dedup();
        assert_eq!(lengths.len(), 1, "all docs should share one fixed width");
    }

    #[test]
    fn rewrite_preserves_answer_rank() {
        let config = tiny_config();
        let data = generate_permutation_transfer_data(&config);
        let mut rng = StdRng::seed_from_u64(99);
        let original = data.target_episodes[0].support_pool[0].clone();
        let renamed = original.renamed(&mut rng);
        assert_eq!(original.answer_rank, renamed.answer_rank);
        assert_eq!(original.task, renamed.task);
        assert_eq!(original.object_count, renamed.object_count);
    }

    #[test]
    fn generated_transfer_data_writes_corpora_and_manifest() {
        let config = tiny_config();
        let data = generate_permutation_transfer_data(&config);
        let dir = tempdir().expect("tempdir");
        write_generated_transfer_data(dir.path(), &data).expect("write artifacts");
        assert!(dir.path().join("source_pretrain.txt").is_file());
        assert!(dir.path().join("source_stream.txt").is_file());
        assert!(dir.path().join("source_holdout.txt").is_file());
        assert!(
            dir.path()
                .join("episodes")
                .join(&data.target_episodes[0].episode_id)
                .join("support_pool.txt")
                .is_file()
        );
        assert!(dir.path().join("generated_transfer_data.json").is_file());
    }

    #[test]
    fn smoke_config_parses_and_validates() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("config")
            .join("language")
            .join("experiments")
            .join("permutation_state_transfer_ttcl_smoke.toml");
        let config = load_permutation_transfer_experiment_config(&path).expect("load smoke config");
        config.validate().expect("validate smoke config");
        assert_eq!(config.protocols.len(), 7);
    }

    #[test]
    fn summarize_protocols_tracks_source_checkpoint_forgetting() {
        let mut summary = PermutationTransferRunSummary {
            experiment_name: "summary_smoke".to_string(),
            output_dir: PathBuf::from("artifacts/language/summary_smoke"),
            seed: 7,
            backend: "ndarray".to_string(),
            train_source_stream: true,
            render: PermutationRenderConfig {
                render_object_count: 7,
                relation_slots: 8,
                operation_slots: 5,
            },
            base_block_size: 64,
            source_pretrain_examples: 16,
            source_stream_examples: 8,
            source_holdout_examples: 4,
            target_episodes: 2,
            source_pretrain_checkpoint_dir: PathBuf::from("runs/source_pretrain/checkpoint"),
            source_pretrain_checkpoint_epoch: 4,
            source_stream_checkpoint_dir: PathBuf::from("runs/source_stream/checkpoint"),
            source_stream_checkpoint_epoch: 2,
            protocol_summaries: Vec::new(),
            difficulty_summaries: Vec::new(),
            episode_metrics: Vec::new(),
        };
        let protocols = vec![TtclProtocolConfig {
            name: "continual_16x1".to_string(),
            mode: TtclProtocolMode::Continual,
            support_examples: 16,
            gradient_steps: 1,
            rewrite: SupportRewrite::None,
        }];
        summarize_protocols(
            &mut summary,
            vec![
                ProtocolEpisodeMetrics {
                    protocol_name: "continual_16x1".to_string(),
                    mode: TtclProtocolMode::Continual,
                    episode_id: "logical_deduction_n5_ep00".to_string(),
                    episode_index: 0,
                    object_count: 5,
                    support_examples: 16,
                    gradient_steps: 1,
                    query_accuracy_source_checkpoint: 0.25,
                    query_accuracy_before: 0.25,
                    query_accuracy_after: 0.50,
                    source_accuracy_source_checkpoint: 0.80,
                    source_accuracy_before: 0.80,
                    source_accuracy_after: 0.76,
                },
                ProtocolEpisodeMetrics {
                    protocol_name: "continual_16x1".to_string(),
                    mode: TtclProtocolMode::Continual,
                    episode_id: "logical_deduction_n7_ep00".to_string(),
                    episode_index: 1,
                    object_count: 7,
                    support_examples: 16,
                    gradient_steps: 1,
                    query_accuracy_source_checkpoint: 0.10,
                    query_accuracy_before: 0.35,
                    query_accuracy_after: 0.45,
                    source_accuracy_source_checkpoint: 0.80,
                    source_accuracy_before: 0.76,
                    source_accuracy_after: 0.64,
                },
            ],
            &protocols,
        );

        assert_eq!(summary.protocol_summaries.len(), 1);
        let protocol = &summary.protocol_summaries[0];
        assert!((protocol.mean_query_delta_vs_source_checkpoint - 0.30).abs() < 1e-9);
        assert!((protocol.max_source_forgetting_vs_source_checkpoint + 0.16).abs() < 1e-9);
        assert!((protocol.final_source_retention_ratio - 0.80).abs() < 1e-9);
        assert_eq!(summary.difficulty_summaries.len(), 2);
        assert_eq!(summary.episode_metrics[0].episode_index, 0);
        assert_eq!(summary.episode_metrics[1].episode_index, 1);
    }
}
