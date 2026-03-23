use std::collections::HashMap;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use burn::tensor::backend::Backend as BackendTrait;
use burn_autodiff::Autodiff;
use burn_dragon::checkpoint::resolve_checkpoint_base;
use burn_dragon_language::api::train::ttcl::{
    PermutationEpisode, PermutationExample, PermutationTransferRunSummary, TtclProtocolMode,
    derive_render_config, generate_permutation_transfer_data,
    load_permutation_transfer_experiment_config, render_run_summary_markdown,
    resolve_ttcl_output_dir, summarize_protocols, write_example_corpus,
    write_generated_transfer_data,
};
use burn_dragon_language::checkpoint::{RUN_DIR_ENV, RUN_NAME_ENV, RUN_ROOT_ENV};
use burn_dragon_language::train::{prepare_dataset, train_backend};
use burn_dragon_language::{
    DatasetConfig, DatasetSourceConfig, ExperimentBackend, GenerationConfig,
    GenerationOutputFormat, GenerationTokenizerSourceConfig, ModelOverrides, TrainingConfig,
    TrainingHyperparameters, load_language_core_from_checkpoint, load_tokenizer_for_checkpoint,
};
use burn_dragon_train::{OptimizerConfig, WgpuRuntimeConfig};
use burn_ndarray::NdArray;
use burn_wgpu::{CubeBackend, WgpuRuntime};
use clap::{Parser, ValueEnum};
use rand::SeedableRng;

#[cfg(feature = "language-cuda")]
use burn_cuda::Cuda;

type NdTrainBackend = Autodiff<NdArray<f32>>;
type NdInferBackend = NdArray<f32>;
type WgpuNoFusion = CubeBackend<WgpuRuntime, f32, i32, u32>;
type WgpuTrainBackend = Autodiff<WgpuNoFusion>;

#[derive(Debug, Clone)]
struct TrainingArtifact {
    checkpoint_dir: PathBuf,
    checkpoint_epoch: usize,
}

#[derive(Debug, Parser)]
#[command(
    author,
    version,
    about = "Run synthetic test-time continual learning experiments"
)]
struct Args {
    /// Path to the experiment config TOML.
    #[arg(short = 'c', long = "config")]
    config: PathBuf,
    /// Optional seed override for repeatable sweeps from one committed config.
    #[arg(long)]
    seed: Option<u64>,
    /// Optional output directory override for repeatable sweeps from one committed config.
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Backend used for training stages.
    #[arg(long, value_enum, default_value_t = BackendArg::Ndarray)]
    backend: BackendArg,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum BackendArg {
    Ndarray,
    Wgpu,
    WgpuNoFusion,
    Cuda,
}

impl BackendArg {
    fn experiment_backend(self) -> ExperimentBackend {
        match self {
            Self::Ndarray => ExperimentBackend::Ndarray,
            Self::Wgpu => ExperimentBackend::Wgpu,
            Self::WgpuNoFusion => ExperimentBackend::WgpuNoFusion,
            Self::Cuda => ExperimentBackend::Cuda,
        }
    }
}

struct EnvVarGuard {
    keys: Vec<(&'static str, Option<OsString>)>,
}

impl EnvVarGuard {
    fn capture(keys: &[&'static str]) -> Self {
        Self {
            keys: keys
                .iter()
                .map(|key| (*key, std::env::var_os(key)))
                .collect(),
        }
    }

    fn set(key: &str, value: &Path) {
        unsafe {
            std::env::set_var(key, value.as_os_str());
        }
    }

    fn clear(key: &str) {
        unsafe {
            std::env::remove_var(key);
        }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        for (key, value) in &self.keys {
            match value {
                Some(saved) => unsafe { std::env::set_var(key, saved) },
                None => unsafe { std::env::remove_var(key) },
            }
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let mut experiment = load_permutation_transfer_experiment_config(&args.config)?;
    if let Some(seed) = args.seed {
        experiment.seed = seed;
    }
    if let Some(output_dir) = args.output_dir {
        experiment.output_dir = output_dir;
    }
    let output_dir = resolve_ttcl_output_dir(&args.config, &experiment.output_dir);
    let generated = generate_permutation_transfer_data(&experiment);
    let data_dir = output_dir.join("generated");
    write_generated_transfer_data(&data_dir, &generated)?;

    let block_size = generated
        .source_pretrain
        .first()
        .map(|example| example.prompt().len() + example.answer().len())
        .ok_or_else(|| anyhow!("source_pretrain is empty"))?;

    let backend = args.backend.experiment_backend();
    let base_artifact = run_training_stage(
        build_training_config(
            &data_dir.join("source_pretrain.txt"),
            &output_dir.join("cache/source_pretrain"),
            block_size,
            experiment.training.base_batch_size,
            experiment.training.base_max_iters,
            experiment.training.base_learning_rate,
            experiment.training.checkpoint_interval_iters,
            experiment.training.log_frequency,
            experiment.seed,
            &experiment.model,
            None,
        ),
        &output_dir.join("runs/source_pretrain"),
        backend,
    )?;

    let source_artifact = if experiment.train_source_stream {
        run_training_stage(
            build_training_config(
                &data_dir.join("source_stream.txt"),
                &output_dir.join("cache/source_stream"),
                block_size,
                experiment.training.base_batch_size,
                experiment.training.source_stream_max_iters,
                experiment.training.base_learning_rate,
                experiment.training.checkpoint_interval_iters,
                experiment.training.log_frequency,
                experiment.seed.wrapping_add(1),
                &experiment.model,
                Some(&base_artifact),
            ),
            &output_dir.join("runs/source_stream"),
            backend,
        )?
    } else {
        base_artifact.clone()
    };

    let source_checkpoint_source_accuracy =
        evaluate_examples(&source_artifact, &generated.source_holdout, None)
            .context("source checkpoint holdout accuracy")?;
    let source_checkpoint_query_accuracy = generated
        .target_episodes
        .iter()
        .map(|episode| {
            Ok((
                episode.episode_id.clone(),
                evaluate_examples(&source_artifact, &episode.queries, None).with_context(|| {
                    format!(
                        "source checkpoint query accuracy for {}",
                        episode.episode_id
                    )
                })?,
            ))
        })
        .collect::<Result<HashMap<_, _>>>()?;

    let mut summary = PermutationTransferRunSummary {
        experiment_name: experiment.name.clone(),
        output_dir: output_dir.clone(),
        seed: experiment.seed,
        backend: format!("{backend:?}"),
        train_source_stream: experiment.train_source_stream,
        render: derive_render_config(&experiment),
        base_block_size: block_size,
        source_pretrain_examples: generated.source_pretrain.len(),
        source_stream_examples: generated.source_stream.len(),
        source_holdout_examples: generated.source_holdout.len(),
        target_episodes: generated.target_episodes.len(),
        source_pretrain_checkpoint_dir: base_artifact.checkpoint_dir.clone(),
        source_pretrain_checkpoint_epoch: base_artifact.checkpoint_epoch,
        source_stream_checkpoint_dir: source_artifact.checkpoint_dir.clone(),
        source_stream_checkpoint_epoch: source_artifact.checkpoint_epoch,
        protocol_summaries: Vec::new(),
        difficulty_summaries: Vec::new(),
        episode_metrics: Vec::new(),
    };

    let mut all_metrics = Vec::new();
    for protocol in &experiment.protocols {
        let mut rolling_artifact = source_artifact.clone();
        for (episode_index, episode) in generated.target_episodes.iter().enumerate() {
            let start_artifact = match protocol.mode {
                TtclProtocolMode::Continual => &rolling_artifact,
                _ => &source_artifact,
            };
            let query_source_checkpoint = *source_checkpoint_query_accuracy
                .get(&episode.episode_id)
                .ok_or_else(|| {
                    anyhow!(
                        "missing source checkpoint query accuracy for {}",
                        episode.episode_id
                    )
                })?;
            let source_source_checkpoint = source_checkpoint_source_accuracy;
            let (query_before, source_before) =
                if matches!(protocol.mode, TtclProtocolMode::Continual) && episode_index > 0 {
                    (
                        evaluate_examples(start_artifact, &episode.queries, None)
                            .context("query before")?,
                        evaluate_examples(start_artifact, &generated.source_holdout, None)
                            .context("source before")?,
                    )
                } else {
                    (query_source_checkpoint, source_source_checkpoint)
                };

            let support_examples = select_support_examples(
                episode,
                protocol.support_examples,
                protocol.mode,
                protocol.rewrite,
                experiment.seed,
            );

            let (query_after, source_after, next_artifact) = match protocol.mode {
                TtclProtocolMode::ZeroShot => (query_before, source_before, None),
                TtclProtocolMode::InContextOnly => (
                    evaluate_examples(start_artifact, &episode.queries, Some(&support_examples))
                        .context("query after in-context")?,
                    source_before,
                    None,
                ),
                TtclProtocolMode::ResetPerEpisode | TtclProtocolMode::Continual => {
                    let support_dir = output_dir
                        .join("adaptation")
                        .join(&protocol.name)
                        .join(&episode.episode_id);
                    let support_path = support_dir.join("support.txt");
                    write_example_corpus(&support_path, &support_examples)?;
                    let adaptation = run_training_stage(
                        build_training_config(
                            &support_path,
                            &support_dir.join("cache"),
                            block_size,
                            experiment
                                .training
                                .adaptation_batch_size
                                .min(protocol.support_examples.max(1)),
                            protocol.gradient_steps,
                            experiment.training.adaptation_learning_rate,
                            experiment.training.checkpoint_interval_iters,
                            experiment.training.log_frequency,
                            experiment.seed.wrapping_add(17),
                            &experiment.model,
                            Some(start_artifact),
                        ),
                        &output_dir
                            .join("runs")
                            .join(&protocol.name)
                            .join(&episode.episode_id),
                        backend,
                    )?;
                    (
                        evaluate_examples(&adaptation, &episode.queries, None)
                            .context("query after adaptation")?,
                        evaluate_examples(&adaptation, &generated.source_holdout, None)
                            .context("source after adaptation")?,
                        Some(adaptation),
                    )
                }
            };

            if let Some(next_artifact) = next_artifact
                && matches!(protocol.mode, TtclProtocolMode::Continual)
            {
                rolling_artifact = next_artifact;
            }

            all_metrics.push(burn_dragon_language::ProtocolEpisodeMetrics {
                protocol_name: protocol.name.clone(),
                mode: protocol.mode,
                episode_id: episode.episode_id.clone(),
                episode_index,
                object_count: episode.object_count,
                support_examples: protocol.support_examples,
                gradient_steps: protocol.gradient_steps,
                query_accuracy_source_checkpoint: query_source_checkpoint,
                query_accuracy_before: query_before,
                query_accuracy_after: query_after,
                source_accuracy_source_checkpoint: source_source_checkpoint,
                source_accuracy_before: source_before,
                source_accuracy_after: source_after,
            });
        }
    }

    summarize_protocols(&mut summary, all_metrics, &experiment.protocols);

    std::fs::create_dir_all(&output_dir)
        .with_context(|| format!("failed to create {}", output_dir.display()))?;
    let summary_json = output_dir.join("summary.json");
    let summary_md = output_dir.join("summary.md");
    std::fs::write(
        &summary_json,
        serde_json::to_string_pretty(&summary).context("serialize summary")?,
    )
    .with_context(|| format!("failed to write {}", summary_json.display()))?;
    std::fs::write(&summary_md, render_run_summary_markdown(&summary))
        .with_context(|| format!("failed to write {}", summary_md.display()))?;

    println!("output_dir: {}", output_dir.display());
    println!("summary_json: {}", summary_json.display());
    println!("summary_md: {}", summary_md.display());
    Ok(())
}

fn build_training_config(
    source_path: &Path,
    cache_dir: &Path,
    block_size: usize,
    batch_size: usize,
    max_iters: usize,
    learning_rate: f64,
    checkpoint_interval_iters: usize,
    log_frequency: usize,
    seed: u64,
    model: &ModelOverrides,
    init: Option<&TrainingArtifact>,
) -> TrainingConfig {
    TrainingConfig {
        dataset: DatasetConfig {
            cache_dir: cache_dir.to_path_buf(),
            train_split_ratio: 1.0,
            validation: None,
            source: DatasetSourceConfig::LocalText {
                path: source_path.to_path_buf(),
            },
            tokenizer: burn_dragon_language::tokenizer::TokenizerConfig {
                vocab_path: None,
                kind: burn_dragon_language::tokenizer::TokenizerKind::Byte(
                    burn_dragon_language::tokenizer::ByteTokenizerConfig {
                        add_special_tokens: true,
                    },
                ),
            },
        },
        training: TrainingHyperparameters {
            block_size,
            tbptt_chunk_size: None,
            tbptt_persist_across_steps: false,
            min_logical_block_size: None,
            batch_size: batch_size.max(1),
            seed,
            gradient_accumulation_steps: 1,
            target_effective_batch_size: None,
            epochs: None,
            max_iters: max_iters.max(1),
            checkpoint_interval_iters: checkpoint_interval_iters.max(1),
            log_frequency: log_frequency.max(1),
            fast_train: false,
            resume_run_dir: None,
            resume_checkpoint_epoch: None,
            init_checkpoint_path: init.map(|artifact| artifact.checkpoint_dir.clone()),
            init_checkpoint_epoch: init.map(|artifact| artifact.checkpoint_epoch),
            context_strategy: burn_dragon_language::ContextStrategyConfig::Infinite,
            sequence_kernel_override: None,
            gdpo: None,
        },
        optimizer: OptimizerConfig {
            learning_rate,
            weight_decay: 0.0,
            lr_schedule: None,
            grad_clip_norm: None,
            grad_clip_value: None,
        },
        parallel: Default::default(),
        generation: GenerationConfig {
            prompt: String::new(),
            max_tokens: Some(1),
            max_chars: None,
            temperature: 1.0,
            top_k: Some(1),
            context_strategy: burn_dragon_language::ContextStrategyConfig::Infinite,
            prompt_tokenizer: GenerationTokenizerSourceConfig::default(),
            decode_tokenizer: GenerationTokenizerSourceConfig::default(),
            output_format: GenerationOutputFormat::DecodedText,
        },
        wgpu: WgpuRuntimeConfig::default(),
        model: model.clone(),
    }
}

fn run_training_stage(
    config: TrainingConfig,
    run_root: &Path,
    backend: ExperimentBackend,
) -> Result<TrainingArtifact> {
    std::fs::create_dir_all(run_root)
        .with_context(|| format!("failed to create {}", run_root.display()))?;
    let dataset = prepare_dataset(&config.dataset, &config.training).context("prepare dataset")?;
    let _env = EnvVarGuard::capture(&[RUN_ROOT_ENV, RUN_DIR_ENV, RUN_NAME_ENV]);
    EnvVarGuard::set(RUN_ROOT_ENV, run_root);
    EnvVarGuard::clear(RUN_DIR_ENV);
    EnvVarGuard::clear(RUN_NAME_ENV);

    match backend {
        ExperimentBackend::Ndarray => {
            train_backend::<NdTrainBackend, _>(&config, dataset, "ndarray", |_| {})
                .context("train ndarray backend")?;
        }
        ExperimentBackend::Wgpu => {
            let mut config = config;
            config.wgpu.training.fused_core_recurrent = Some(true);
            config.wgpu.training.fused_core_rollout = Some(true);
            let wgpu_config = config.wgpu.clone();
            train_backend::<WgpuTrainBackend, _>(
                &config,
                dataset,
                "wgpu-fused-core",
                move |device| burn_dragon_train::wgpu::init_runtime(device, &wgpu_config),
            )
            .context("train wgpu backend")?;
        }
        ExperimentBackend::WgpuNoFusion => {
            let mut config = config;
            config.wgpu.training.fused_core_recurrent = Some(false);
            config.wgpu.training.fused_core_rollout = Some(false);
            let wgpu_config = config.wgpu.clone();
            train_backend::<WgpuTrainBackend, _>(
                &config,
                dataset,
                "wgpu-nofusion",
                move |device| burn_dragon_train::wgpu::init_runtime(device, &wgpu_config),
            )
            .context("train wgpu-nofusion backend")?;
        }
        ExperimentBackend::Cuda => {
            #[cfg(feature = "language-cuda")]
            {
                type CudaTrainBackend = Autodiff<Cuda<f32, i32>>;
                train_backend::<CudaTrainBackend, _>(&config, dataset, "cuda", |_| {})
                    .context("train cuda backend")?;
            }
            #[cfg(not(feature = "language-cuda"))]
            {
                return Err(anyhow!(
                    "cuda backend requested but burn_dragon_cli was not built with --features language-cuda"
                ));
            }
        }
    }

    let run_dir = burn_dragon_language::checkpoint::resolve_latest_run_dir_in(run_root)
        .ok_or_else(|| anyhow!("no run directory found under {}", run_root.display()))?;
    let checkpoint_dir = run_dir.join("checkpoint");
    let (_, checkpoint_epoch) =
        resolve_checkpoint_base(&checkpoint_dir, None).context("resolve checkpoint epoch")?;
    Ok(TrainingArtifact {
        checkpoint_dir,
        checkpoint_epoch,
    })
}

fn select_support_examples(
    episode: &PermutationEpisode,
    support_examples: usize,
    mode: TtclProtocolMode,
    rewrite: burn_dragon_language::SupportRewrite,
    seed: u64,
) -> Vec<PermutationExample> {
    let take = support_examples.min(episode.support_pool.len());
    let mut support = episode
        .support_pool
        .iter()
        .take(take)
        .cloned()
        .collect::<Vec<_>>();
    if take == 0 || matches!(mode, TtclProtocolMode::ZeroShot) {
        return support;
    }
    if matches!(rewrite, burn_dragon_language::SupportRewrite::RenameObjects) {
        let mut rng = rand::rngs::StdRng::seed_from_u64(
            seed ^ (episode.object_count as u64) ^ (take as u64).rotate_left(7),
        );
        support = support
            .into_iter()
            .map(|example| example.renamed(&mut rng))
            .collect();
    }
    support
}

fn evaluate_examples(
    artifact: &TrainingArtifact,
    examples: &[PermutationExample],
    in_context_support: Option<&[PermutationExample]>,
) -> Result<f64> {
    let device = <NdInferBackend as BackendTrait>::Device::default();
    let checkpoint = artifact.checkpoint_dir.clone();
    let tokenizer = load_tokenizer_for_checkpoint(&[], Some(&checkpoint), "ndarray")
        .context("load tokenizer for evaluation")?;
    let model = load_language_core_from_checkpoint::<NdInferBackend>(
        &checkpoint,
        Some(artifact.checkpoint_epoch),
        &[],
        "ndarray",
        &device,
    )
    .context("load checkpoint for evaluation")?;
    let mut correct = 0usize;
    for example in examples {
        let prompt = compose_prompt(example, in_context_support);
        let prediction = predict_answer(&model, tokenizer.as_ref(), &prompt, &device)?;
        if prediction == example.answer() {
            correct += 1;
        }
    }
    Ok(correct as f64 / examples.len().max(1) as f64)
}

fn compose_prompt(
    example: &PermutationExample,
    in_context_support: Option<&[PermutationExample]>,
) -> String {
    match in_context_support {
        Some(support) if !support.is_empty() => {
            let prefix = support
                .iter()
                .map(PermutationExample::document)
                .collect::<String>();
            format!("{prefix}{}", example.prompt())
        }
        _ => example.prompt(),
    }
}

fn predict_answer<B: BackendTrait>(
    model: &burn_dragon_language::BDH<B>,
    tokenizer: &dyn burn_dragon_language::tokenizer::Tokenizer,
    prompt: &str,
    device: &B::Device,
) -> Result<String> {
    let prompt_tokens = tokenizer
        .encode(prompt, false, false)
        .into_iter()
        .map(i64::from)
        .collect::<Vec<_>>();
    let (mut state, last_logits) =
        burn_dragon_language::prefill_state(model, &prompt_tokens, device).context("prefill")?;
    let (next_token, _) = burn_dragon_language::sample_next_token(
        model,
        &mut state,
        last_logits,
        1.0,
        Some(1),
        device,
    )
    .context("sample next token")?;
    Ok(tokenizer
        .decode(&[next_token as u32])
        .chars()
        .take(1)
        .collect())
}
