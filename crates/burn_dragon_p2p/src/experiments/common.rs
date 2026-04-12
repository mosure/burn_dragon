use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Result, bail};
use burn::data::dataloader::batcher::Batcher;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Int, Tensor, TensorData};
use burn::train::{Learner, LearningComponentsMarker};
use burn_dragon_language::api::checkpoint::apply_init_checkpoint_to_language_core;
use burn_dragon_language::api::inference::build_model_config_with_tokenizer;
use burn_dragon_language::config::ValidationDatasetConfig;
use burn_dragon_language::dataset::{
    Dataset, DatasetSplit, RandomDataLoader, SequenceBatch, StreamingDataLoader,
    TokenSequenceDataset,
};
use burn_dragon_language::summary_event_mask_tensor;
use burn_dragon_language::tokenizer::{SharedTokenizer, Tokenizer};
use burn_dragon_language::train::schedule::{resolve_lr_scheduler, resolve_train_schedule};
use burn_dragon_language::train::steps::LanguageTrainModel;
use burn_dragon_language::train::utils::prepare_datasets;
use burn_dragon_language::train::{
    LanguageOptimizer, resolve_language_optimizer, validate_language_continual_backprop,
};
use burn_dragon_language::{BDH, BDHConfig, DatasetConfig, TrainingConfig};
use burn_dragon_train::train::constants::ValidBackend;
use burn_dragon_train::train::metrics::{
    LanguageModelOutput, LanguageModelTrainItem, LossValue, ScalarValue,
};
use burn_dragon_train::train::pipeline::ResolvedLrScheduler;
use burn_p2p::burn::{
    BurnLearnerDataPipeline, BurnLearnerProject, BurnTrainLoader, BurnValidationLoader,
    BurnWorkloadAdapter, connect, from_learner, from_loaders,
};
use burn_p2p::{
    DatasetViewId, EvalSplit, GeneratedWorkloadInputProvider, LeaseDataPipeline,
    LeaseDataPipelineDescriptor, LeaseDataPipelineKind, MetricReport, MetricValue, NodeBuilder,
    SelectedWorkloadProject, SingleWorkloadProjectFamily,
};
use burn_train::InferenceStep;
use burn_train::metric::{Adaptor, ItemLazy};

use crate::auth::compose_auth_config;
use crate::capability::{
    DragonCapabilityClass, DragonNativeCapabilityAssessment, DragonNativeTargetDecision,
    DragonTrainingFootprint, decide_native_target, estimate_language_training_footprint,
};
use crate::capability_state::{
    apply_native_downgrade_state, clear_native_downgrade, persist_native_downgrade,
};
use crate::config::{
    DragonExistingShardDatasetConfig, DragonExperimentKind, DragonManifestBundle,
    DragonNativeAuthBundle, DragonNativePeerConfig, DragonShardExportConfig, TokenWindowRecord,
};
use crate::manifests::build_manifest_bundle;
use crate::profile::resolve_native_training_profile;

pub type DragonLearningComponents<B> =
    LearningComponentsMarker<B, ResolvedLrScheduler, LanguageTrainModel<B>, LanguageOptimizer<B>>;

pub type DragonProjectFamily<B> = SingleWorkloadProjectFamily<
    BurnWorkloadAdapter<BurnLearnerProject<DragonLearningComponents<B>>>,
>;

pub type DragonNodeBuilder<B> = NodeBuilder<SelectedWorkloadProject<DragonProjectFamily<B>>>;
pub type DragonBurnProject<B> = BurnLearnerProject<DragonLearningComponents<B>>;

#[derive(Clone)]
pub struct PreparedNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    pub project: DragonBurnProject<B>,
    pub builder: DragonNodeBuilder<B>,
    pub manifests: DragonManifestBundle,
    pub config: TrainingConfig,
    pub storage_root: PathBuf,
    pub experiment_kind: DragonExperimentKind,
    pub backend_label: String,
    pub model_config: BDHConfig,
    pub footprint: DragonTrainingFootprint,
    pub target_decision: DragonNativeTargetDecision,
}

impl<B> PreparedNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    pub fn record_runtime_training_failure(&self, reason: &str) -> Result<()> {
        let downgrade_to = match self.target_decision.effective_target {
            crate::config::DragonNativeTarget::Reducer => "reducer",
            crate::config::DragonNativeTarget::Validator => "validator",
            crate::config::DragonNativeTarget::Auto
            | crate::config::DragonNativeTarget::Trainer => "validator",
        };
        let _ = persist_native_downgrade(
            &self.storage_root,
            self.experiment_kind,
            &self.backend_label,
            &self.model_config,
            self.config.training.batch_size,
            self.config.training.block_size,
            &self.footprint,
            self.target_decision.trainer_memory_budget_bytes,
            downgrade_to,
            reason,
            "runtime",
        )?;
        Ok(())
    }

    pub fn clear_runtime_downgrade(&self) -> Result<()> {
        clear_native_downgrade(
            &self.storage_root,
            self.experiment_kind,
            &self.backend_label,
            &self.model_config,
            self.config.training.batch_size,
            self.config.training.block_size,
        )
    }
}

#[derive(Clone, Debug)]
pub struct TokenWindowBatcher {
    summary_event_token_ids: Option<Vec<u32>>,
}

impl TokenWindowBatcher {
    pub fn new(summary_event_token_ids: Option<Vec<u32>>) -> Self {
        Self {
            summary_event_token_ids,
        }
    }
}

#[derive(Clone, Debug)]
struct DragonGeneratedInputDescriptor {
    provider: &'static str,
    metadata: BTreeMap<String, String>,
}

impl GeneratedWorkloadInputProvider for DragonGeneratedInputDescriptor {
    fn provider_id(&self) -> String {
        self.provider.into()
    }

    fn metadata(&self) -> BTreeMap<String, String> {
        self.metadata.clone()
    }
}

fn trim_http_base(base_url: &str) -> String {
    base_url.trim_end_matches('/').to_owned()
}

fn dragon_sharded_input_descriptor(
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    registration: &burn_p2p::DatasetRegistration,
    shard_count: usize,
    http_upstream: Option<&str>,
) -> LeaseDataPipelineDescriptor {
    let mut descriptor = LeaseDataPipelineDescriptor::new(
        format!("dragon-{}-shards", experiment_kind.workload_slug()),
        LeaseDataPipelineKind::ShardedStatic,
    )
    .with_metadata_entry("experiment_kind", experiment_kind.workload_slug())
    .with_metadata_entry("dataset_id", registration.manifest.dataset_id.as_str())
    .with_metadata_entry(
        "dataset_view_id",
        registration.view.dataset_view_id.as_str(),
    )
    .with_metadata_entry("source_uri", registration.manifest.source_uri.clone())
    .with_metadata_entry("format", registration.manifest.format.clone());

    if let Some(base_url) = http_upstream {
        return descriptor
            .with_shard_manifest_http_source(
                format!("{}/fetch-manifest.json", trim_http_base(base_url)),
                Some(shard_count as u64),
            )
            .with_metadata_entry("upstream", "http");
    }

    descriptor = descriptor.with_metadata_entry("upstream", "local");
    match dataset_source {
        burn_dragon_language::DatasetSourceConfig::UniversalityNca { config } => {
            let provider = DragonGeneratedInputDescriptor {
                provider: "burn_dragon_universality_nca",
                metadata: BTreeMap::from([
                    ("config_path".into(), config.display().to_string()),
                    (
                        "experiment_kind".into(),
                        experiment_kind.workload_slug().into(),
                    ),
                ]),
            };
            descriptor.with_generated_input_source(&provider)
        }
        burn_dragon_language::DatasetSourceConfig::UniversalityManifest { manifest } => descriptor
            .with_custom_input_source(
                "universality-manifest",
                BTreeMap::from([
                    ("manifest_path".into(), manifest.display().to_string()),
                    (
                        "experiment_kind".into(),
                        experiment_kind.workload_slug().into(),
                    ),
                ]),
            ),
        burn_dragon_language::DatasetSourceConfig::NemotronClimbMix {
            revision,
            max_records,
        } => {
            let mut metadata = BTreeMap::from([(
                "experiment_kind".into(),
                experiment_kind.workload_slug().into(),
            )]);
            if let Some(revision) = revision {
                metadata.insert("revision".into(), revision.clone());
            }
            if let Some(max_records) = max_records {
                metadata.insert("max_records".into(), max_records.to_string());
            }
            descriptor.with_custom_input_source("nemotron-climbmix", metadata)
        }
        _ => descriptor.with_custom_input_source(
            "burn-dragon-language",
            BTreeMap::from([(
                "experiment_kind".into(),
                experiment_kind.workload_slug().into(),
            )]),
        ),
    }
}

fn dragon_sharded_data_pipeline<B>(
    descriptor: LeaseDataPipelineDescriptor,
    dataset: burn_p2p::burn::BurnShardedDataset<TokenWindowRecord>,
    batcher: TokenWindowBatcher,
    batch_size: usize,
) -> BurnLearnerDataPipeline<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let registration = dataset.registration().clone();
    let microshard_plan = dataset.microshard_plan().clone();
    LeaseDataPipeline::new(
        descriptor,
        move || Ok(registration.clone()),
        move |_registration| Ok(microshard_plan.clone()),
        move |_lease, cached_microshards, device| {
            dataset.load_batches(cached_microshards, batcher.clone(), batch_size, device)
        },
    )
}

impl<B: Backend> Batcher<B, TokenWindowRecord, SequenceBatch<B>> for TokenWindowBatcher {
    fn batch(&self, items: Vec<TokenWindowRecord>, device: &B::Device) -> SequenceBatch<B> {
        let batch_size = items.len().max(1);
        let block_size = items
            .first()
            .map(|item| item.inputs.len())
            .unwrap_or_default()
            .max(1);
        let mut inputs = Vec::with_capacity(batch_size * block_size);
        let mut targets = Vec::with_capacity(batch_size * block_size);
        let mut reset_stream_state = false;
        for item in items {
            reset_stream_state |= item.reset_stream_state;
            inputs.extend(item.inputs);
            targets.extend(item.targets);
        }
        let summary_event_mask = summary_event_mask_tensor::<B>(
            &inputs,
            batch_size,
            block_size,
            self.summary_event_token_ids.as_deref(),
            device,
        );
        SequenceBatch::<B> {
            inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [batch_size, block_size]),
                device,
            ),
            targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, block_size]),
                device,
            ),
            summary_event_mask,
            reset_stream_state,
        }
    }
}

fn dataset_view_id_for_dataset(
    dataset: &Dataset,
    shard_export: Option<&DragonShardExportConfig>,
) -> Result<DatasetViewId> {
    if let Some(shard_export) = shard_export {
        let dataset_name = shard_export
            .dataset_name
            .as_deref()
            .unwrap_or("burn-dragon-p2p-dataset");
        return DatasetViewId::derive(&(
            "burn-dragon-p2p-dataset-view",
            dataset_name,
            dataset.block_size(),
            dataset.batch_size(),
            dataset.token_count(),
        ))
        .map_err(Into::into);
    }

    DatasetViewId::derive(&(
        "burn-dragon-p2p-inline-dataset-view",
        dataset.block_size(),
        dataset.batch_size(),
        dataset.token_count(),
        dataset.train_split_ratio().to_bits(),
    ))
    .map_err(Into::into)
}

fn summary_event_token_ids(dataset: &Arc<Dataset>) -> Option<Vec<u32>> {
    summary_event_token_ids_for_tokenizer(dataset.tokenizer().as_ref())
}

fn summary_event_token_ids_for_tokenizer(tokenizer: &dyn Tokenizer) -> Option<Vec<u32>> {
    let ids = [tokenizer.bos_id(), tokenizer.eos_id()]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    (!ids.is_empty()).then_some(ids)
}

fn validation_dataset_config_for(
    dataset_cfg: &DatasetConfig,
    validation_cfg: &ValidationDatasetConfig,
) -> DatasetConfig {
    DatasetConfig {
        cache_dir: validation_cfg
            .cache_dir
            .clone()
            .unwrap_or_else(|| dataset_cfg.cache_dir.join("validation")),
        train_split_ratio: validation_cfg
            .train_split_ratio
            .unwrap_or(dataset_cfg.train_split_ratio),
        validation: None,
        source: validation_cfg.source.clone(),
        tokenizer: dataset_cfg.tokenizer.clone(),
    }
}

fn load_tokenizer_without_dataset(config: &TrainingConfig) -> Result<SharedTokenizer> {
    let tokenizer_cfg = &config.dataset.tokenizer;
    match tokenizer_cfg.storage_path(&config.dataset.cache_dir) {
        Some(path) if path.is_file() => tokenizer_cfg.load(&path).map_err(Into::into),
        Some(path) => bail!(
            "shard-first p2p setup requires a persisted tokenizer at {}",
            path.display()
        ),
        None => tokenizer_cfg
            .fit(std::iter::empty::<&str>())
            .map_err(Into::into),
    }
}

fn resolve_model_config_for_capability(config: &TrainingConfig) -> Result<BDHConfig> {
    let tokenizer = match load_tokenizer_without_dataset(config) {
        Ok(tokenizer) => tokenizer,
        Err(_) => prepare_datasets(&config.dataset, &config.training)?
            .train
            .tokenizer()
            .clone(),
    };
    build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        tokenizer.as_ref(),
    )
}

fn assess_loaded_native_training_config(
    config: &TrainingConfig,
    requested_target: crate::config::DragonNativeTarget,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    capability_policy: &crate::config::DragonCapabilityPolicy,
) -> Result<DragonNativeCapabilityAssessment> {
    ensure_supported_training_mode(config, experiment_kind)?;
    let model_config = resolve_model_config_for_capability(config)?;
    let footprint = estimate_language_training_footprint(
        &model_config,
        config.training.batch_size,
        config.training.block_size,
        DragonCapabilityClass::from_backend_label(backend_label),
    );
    let target_decision = decide_native_target(
        requested_target,
        capability_policy,
        DragonCapabilityClass::from_backend_label(backend_label),
        &footprint,
    );

    Ok(DragonNativeCapabilityAssessment {
        experiment_kind,
        backend_label: backend_label.to_owned(),
        model_config,
        batch_size: config.training.batch_size,
        block_size: config.training.block_size,
        footprint,
        target_decision,
    })
}

pub fn assess_native_peer_for_backend(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
) -> Result<DragonNativeCapabilityAssessment> {
    let resolved = resolve_native_training_profile(native, experiment_kind, true)?;
    let config = resolved.config;
    let assessment = assess_loaded_native_training_config(
        &config,
        native.target_or_default(),
        experiment_kind,
        backend_label,
        &native.capability_policy,
    )?;
    apply_native_downgrade_state(&native.storage_root, &config, assessment)
}

fn ensure_supported_training_mode(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<()> {
    if !matches!(
        config.parallel.mode,
        burn_dragon_train::ParallelismKind::Single
    ) {
        bail!("burn_dragon_p2p currently requires parallel.mode = \"single\"");
    }
    if config.parallel.pipeline.enabled {
        bail!("burn_dragon_p2p does not support pipeline parallel training");
    }
    match (&config.dataset.source, experiment_kind) {
        (
            burn_dragon_language::DatasetSourceConfig::UniversalityManifest { .. },
            DragonExperimentKind::NcaPrepretraining,
        )
        | (
            burn_dragon_language::DatasetSourceConfig::UniversalityNca { .. },
            DragonExperimentKind::NcaPrepretraining,
        )
        | (
            burn_dragon_language::DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => {}
        (source, DragonExperimentKind::NcaPrepretraining) => {
            bail!(
                "NCA p2p peers require universality datasets, found {:?}",
                source
            )
        }
        (source, DragonExperimentKind::ClimbMixPretraining) => {
            bail!(
                "ClimbMix p2p peers require nemotron_climbmix data, found {:?}",
                source
            )
        }
    }
    Ok(())
}

fn mean_loss_from_valid_output<B: Backend>(output: LanguageModelOutput<B>) -> f64 {
    mean_loss_from_output_ref(&output)
}

fn mean_loss_from_output_ref<B: Backend>(output: &LanguageModelOutput<B>) -> f64 {
    let loss_value: LossValue<B> = output.adapt();
    let values = loss_value
        .value()
        .to_data()
        .convert::<f32>()
        .into_vec::<f32>()
        .expect("loss tensor");
    if values.is_empty() {
        0.0
    } else {
        values.iter().map(|value| *value as f64).sum::<f64>() / values.len() as f64
    }
}

fn mean_loss_from_train_output_ref<B: AutodiffBackend>(output: &LanguageModelTrainItem<B>) -> f64 {
    mean_loss_from_output_ref(&output.clone().sync())
}

fn language_evaluate<B>(
    model: &LanguageTrainModel<ValidBackend<B>>,
    validation_loader: BurnValidationLoader<DragonLearningComponents<B>>,
    _split: EvalSplit,
) -> MetricReport
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let mut total = 0.0;
    let mut count = 0usize;
    for item in validation_loader.iter() {
        total += mean_loss_from_valid_output(model.step(item));
        count += 1;
    }
    MetricReport {
        metrics: std::collections::BTreeMap::from([
            (
                "loss".into(),
                MetricValue::Float(if count == 0 {
                    0.0
                } else {
                    total / count as f64
                }),
            ),
            (
                "evaluation_batches".into(),
                MetricValue::Integer(count as i64),
            ),
        ]),
        captured_at: chrono::Utc::now(),
    }
}

fn build_train_loader<B>(
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    config: &TrainingConfig,
    steps_per_epoch: usize,
    total_steps: usize,
    device: &B::Device,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnTrainLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    if config.training.tbptt_persist_across_steps {
        Arc::new(
            StreamingDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                device,
                steps_per_epoch,
                Some(total_steps),
                config.training.min_logical_block_size,
                config.training.seed,
            )
            .with_summary_event_token_ids(summary_event_token_ids),
        )
    } else {
        Arc::new(
            RandomDataLoader::<B>::new(
                Arc::clone(&datasets.train),
                DatasetSplit::Train,
                device,
                steps_per_epoch,
                Some(total_steps),
            )
            .with_summary_event_token_ids(summary_event_token_ids),
        )
    }
}

fn build_valid_loader<B>(
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    _config: &TrainingConfig,
    device: &<ValidBackend<B> as Backend>::Device,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnValidationLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_steps = datasets.valid.steps_per_epoch(DatasetSplit::Val);
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            Arc::clone(&datasets.valid),
            DatasetSplit::Val,
            device,
            valid_steps,
            None,
        )
        .with_summary_event_token_ids(summary_event_token_ids),
    )
}

fn build_valid_loader_for_dataset<B>(
    dataset: Arc<Dataset>,
    device: &<ValidBackend<B> as Backend>::Device,
    summary_event_token_ids: Option<Vec<u32>>,
) -> BurnValidationLoader<DragonLearningComponents<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let valid_steps = dataset.steps_per_epoch(DatasetSplit::Val);
    Arc::new(
        RandomDataLoader::<ValidBackend<B>>::new(
            dataset,
            DatasetSplit::Val,
            device,
            valid_steps,
            None,
        )
        .with_summary_event_token_ids(summary_event_token_ids),
    )
}

fn window_records_from_dataset(
    dataset: &Dataset,
    split: DatasetSplit,
    max_records: Option<usize>,
) -> Vec<TokenWindowRecord> {
    let (offset, span) = dataset.split_offset_and_span(split);
    let block_size = dataset.block_size();
    if block_size == 0 || span <= block_size {
        return Vec::new();
    }
    let logical_document_tokens = dataset.preferred_logical_document_tokens(split);
    let document_span = logical_document_tokens.map(|tokens| tokens.saturating_add(1));
    let mut records = Vec::new();
    let max_start = span.saturating_sub(block_size + 1);
    let mut local_start = 0usize;
    while local_start <= max_start {
        let start = offset + local_start;
        let mut sample = vec![0_u32; block_size + 1];
        dataset.copy_token_range(start, &mut sample);
        let reset_stream_state =
            document_span.is_some_and(|document_span| local_start % document_span == 0);
        records.push(TokenWindowRecord {
            inputs: sample[..block_size]
                .iter()
                .map(|token| *token as i64)
                .collect(),
            targets: sample[1..].iter().map(|token| *token as i64).collect(),
            reset_stream_state,
        });
        if max_records.is_some_and(|limit| records.len() >= limit) {
            break;
        }
        local_start = local_start.saturating_add(block_size);
    }
    records
}

fn attach_sharded_dataset<B>(
    builder: burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>,
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    datasets: &burn_dragon_language::train::utils::PreparedDatasets,
    shard_export: &DragonShardExportConfig,
    summary_event_token_ids: Option<Vec<u32>>,
) -> Result<burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let records = window_records_from_dataset(
        &datasets.train,
        DatasetSplit::Train,
        shard_export.max_records,
    );
    if records.is_empty() {
        bail!(
            "shard export for {} produced no records",
            shard_export.root.display()
        );
    }
    let dataset_name = shard_export
        .dataset_name
        .clone()
        .unwrap_or_else(|| "burn-dragon-p2p-dataset".into());
    let mut config = burn_p2p::burn::BurnShardedDatasetConfig::new(dataset_name)
        .with_source_uri(shard_export.root.display().to_string())
        .with_view_metadata_entry("dataset_kind", "language-token-windows");
    if let Some(count) = shard_export.microshards {
        config = config.with_microshards(count);
    }
    let sharded =
        burn_p2p::burn::BurnShardedDataset::write_local(&shard_export.root, &records, config)?;
    let sharded = if let Some(base_url) = &shard_export.http_upstream {
        sharded.with_http_upstream(base_url.clone())
    } else {
        sharded
    };
    let descriptor = dragon_sharded_input_descriptor(
        experiment_kind,
        dataset_source,
        sharded.registration(),
        sharded.microshard_plan().microshards.len(),
        shard_export.http_upstream.as_deref(),
    );
    Ok(
        builder.with_data_pipeline(dragon_sharded_data_pipeline::<B>(
            descriptor,
            sharded,
            TokenWindowBatcher::new(summary_event_token_ids),
            datasets.train.batch_size(),
        )),
    )
}

fn attach_existing_sharded_dataset<B>(
    builder: burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>,
    experiment_kind: DragonExperimentKind,
    dataset_source: &burn_dragon_language::DatasetSourceConfig,
    shard_dataset: &DragonExistingShardDatasetConfig,
    batch_size: usize,
    summary_event_token_ids: Option<Vec<u32>>,
) -> Result<burn_p2p::burn::BurnLearnerProjectBuilder<DragonLearningComponents<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let sharded =
        burn_p2p::burn::BurnShardedDataset::<TokenWindowRecord>::read_local(&shard_dataset.root)?;
    let sharded = if let Some(base_url) = &shard_dataset.http_upstream {
        sharded.with_http_upstream(base_url.clone())
    } else {
        sharded.with_local_upstream(shard_dataset.root.display().to_string())
    };
    let descriptor = dragon_sharded_input_descriptor(
        experiment_kind,
        dataset_source,
        sharded.registration(),
        sharded.microshard_plan().microshards.len(),
        shard_dataset.http_upstream.as_deref(),
    );
    Ok(
        builder.with_data_pipeline(dragon_sharded_data_pipeline::<B>(
            descriptor,
            sharded,
            TokenWindowBatcher::new(summary_event_token_ids),
            batch_size,
        )),
    )
}

fn build_language_learner<B>(
    config: &TrainingConfig,
    backend_label: &str,
    model_config: &BDHConfig,
    total_steps: usize,
    scheduler_iters: Option<usize>,
    device: &B::Device,
) -> Result<Learner<DragonLearningComponents<B>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let mut base_model = BDH::<B>::new(model_config.clone(), device);
    let fresh_model = base_model.clone();
    if let Some(checkpoint_path) = &config.training.init_checkpoint_path {
        base_model = apply_init_checkpoint_to_language_core(
            &base_model,
            config,
            checkpoint_path,
            config.training.init_checkpoint_epoch,
            backend_label,
            device,
        )?;
    }
    validate_language_continual_backprop(&config.training, &base_model, 1)?;

    let model = LanguageTrainModel::new(base_model)
        .with_pipeline_plan(None)
        .with_tbptt_chunk_size(config.training.tbptt_chunk_size)
        .with_tbptt_persist_across_steps(config.training.tbptt_persist_across_steps)
        .with_continual_backprop(&config.training.continual_backprop)
        .with_gradient_scale_schedule(&config.training, total_steps);
    let optimizer = resolve_language_optimizer::<B>(
        &config.training,
        &config.optimizer,
        total_steps,
        fresh_model,
    )?;
    let scheduler = resolve_lr_scheduler(
        &config.optimizer,
        total_steps,
        scheduler_iters,
        model_config,
    )?;
    Ok(Learner::new(model, optimizer, scheduler))
}

fn shard_dataset_upstream(
    shard_export: Option<&DragonShardExportConfig>,
    existing_shard_dataset: Option<&DragonExistingShardDatasetConfig>,
) -> Result<Option<burn_p2p::UpstreamAdapter>> {
    if shard_export.is_some() && existing_shard_dataset.is_some() {
        bail!("configure at most one of shard_export or existing_shard_dataset");
    }
    Ok(match (shard_export, existing_shard_dataset) {
        (Some(shard_export), None) => Some(if let Some(base_url) = &shard_export.http_upstream {
            burn_p2p::UpstreamAdapter::Http {
                base_url: base_url.clone(),
            }
        } else {
            burn_p2p::UpstreamAdapter::Local {
                root: shard_export.root.display().to_string(),
            }
        }),
        (None, Some(shard_dataset)) => Some(if let Some(base_url) = &shard_dataset.http_upstream {
            burn_p2p::UpstreamAdapter::Http {
                base_url: base_url.clone(),
            }
        } else {
            burn_p2p::UpstreamAdapter::Local {
                root: shard_dataset.root.display().to_string(),
            }
        }),
        (None, None) => None,
        (Some(_), Some(_)) => unreachable!(),
    })
}

fn ensure_tokenizer_compatible(
    train_tokenizer: &dyn Tokenizer,
    valid_tokenizer: &dyn Tokenizer,
    tokenizer_label: &str,
) -> Result<()> {
    if train_tokenizer.len() != valid_tokenizer.len() {
        bail!(
            "validation dataset tokenizer is incompatible with the training tokenizer: vocab sizes differ (train={}, valid={}, tokenizer={tokenizer_label})",
            train_tokenizer.len(),
            valid_tokenizer.len(),
        );
    }
    if train_tokenizer.bos_id() != valid_tokenizer.bos_id()
        || train_tokenizer.eos_id() != valid_tokenizer.eos_id()
        || train_tokenizer.pad_id() != valid_tokenizer.pad_id()
        || train_tokenizer.unk_id() != valid_tokenizer.unk_id()
    {
        bail!(
            "validation dataset tokenizer is incompatible with the training tokenizer: special token ids differ (tokenizer={tokenizer_label})"
        );
    }
    Ok(())
}

fn prepare_validation_loader_only<B>(
    config: &TrainingConfig,
    device: &<ValidBackend<B> as Backend>::Device,
    base_tokenizer: &dyn Tokenizer,
    summary_event_token_ids: Option<Vec<u32>>,
) -> Result<Option<BurnValidationLoader<DragonLearningComponents<B>>>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let Some(validation_cfg) = &config.dataset.validation else {
        return Ok(None);
    };
    let effective_cfg = validation_dataset_config_for(&config.dataset, validation_cfg);
    let prepared = prepare_datasets(&effective_cfg, &config.training)?;
    ensure_tokenizer_compatible(
        base_tokenizer,
        prepared.valid.tokenizer().as_ref(),
        &config.dataset.tokenizer.kind_name(),
    )?;
    Ok(Some(build_valid_loader_for_dataset::<B>(
        prepared.valid,
        device,
        summary_event_token_ids,
    )))
}

pub fn prepare_language_peer_for_backend<B>(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    device: B::Device,
    auth_bundle: Option<&DragonNativeAuthBundle>,
) -> Result<PreparedNativePeer<B>>
where
    B: AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let resolved = resolve_native_training_profile(native, experiment_kind, true)?;
    let config = resolved.config;
    let capability_assessment = apply_native_downgrade_state(
        &native.storage_root,
        &config,
        assess_loaded_native_training_config(
            &config,
            native.target_or_default(),
            experiment_kind,
            backend_label,
            &native.capability_policy,
        )?,
    )?;
    let use_existing_shards = native.existing_shard_dataset.as_ref();
    let dataset_upstream = shard_dataset_upstream(
        native.shard_export.as_ref(),
        native.existing_shard_dataset.as_ref(),
    )?;
    let model_config = capability_assessment.model_config.clone();
    let footprint = capability_assessment.footprint.clone();
    let target_decision = capability_assessment.target_decision.clone();

    let (project, dataset_view_id) = if let Some(existing_shards) = use_existing_shards {
        let sharded = burn_p2p::burn::BurnShardedDataset::<TokenWindowRecord>::read_local(
            &existing_shards.root,
        )?;
        let total_examples = sharded
            .shard_examples()
            .values()
            .copied()
            .sum::<usize>()
            .max(1);
        let steps_per_epoch = total_examples.div_ceil(config.training.batch_size.max(1));
        let train_schedule = resolve_train_schedule(&config.training, steps_per_epoch)?;
        let total_steps = train_schedule.total_steps.max(1);
        let scheduler_iters = match train_schedule.source {
            burn_dragon_train::train::pipeline::ScheduleSource::Epochs => Some(total_steps),
            burn_dragon_train::train::pipeline::ScheduleSource::MaxIters => None,
        };
        let tokenizer = load_tokenizer_without_dataset(&config)?;
        let summary_event_token_ids = summary_event_token_ids_for_tokenizer(tokenizer.as_ref());
        let learner = build_language_learner::<B>(
            &config,
            backend_label,
            &model_config,
            total_steps,
            scheduler_iters,
            &device,
        )?;
        let backend_label_owned = backend_label.to_owned();
        let estimated_tokens_per_second = footprint.estimated_tokens_per_second;
        let mut builder = from_learner(learner, device.clone())
            .with_benchmark(move |model, _device| {
                let inventory = burn_p2p::burn::inspect_module::<B, _>(model);
                burn_p2p::CapabilityEstimate {
                    preferred_backends: vec![backend_label_owned.clone()],
                    work_units_per_second: estimated_tokens_per_second
                        .max((inventory.parameter_count.max(1) as f64).sqrt()),
                    target_window_seconds: 30,
                }
            })
            .with_step_metrics(|step_index, output, metrics| {
                metrics.insert(
                    "train_steps".into(),
                    MetricValue::Integer((step_index + 1) as i64),
                );
                metrics.insert(
                    "train_loss".into(),
                    MetricValue::Float(mean_loss_from_train_output_ref(output)),
                );
                Ok(())
            });
        if let Some(validation_loader) = prepare_validation_loader_only::<B>(
            &config,
            &device,
            tokenizer.as_ref(),
            summary_event_token_ids.clone(),
        )? {
            let validation_for_eval = validation_loader.clone();
            builder = builder
                .with_validation_loader(validation_loader)
                .with_evaluate(move |model, split| {
                    language_evaluate::<B>(model, validation_for_eval.clone(), split)
                });
        }
        builder = attach_existing_sharded_dataset::<B>(
            builder,
            experiment_kind,
            &config.dataset.source,
            existing_shards,
            config.training.batch_size,
            summary_event_token_ids,
        )?;
        (
            builder.build()?,
            sharded.registration().view.dataset_view_id.clone(),
        )
    } else {
        let datasets = prepare_datasets(&config.dataset, &config.training)?;
        let summary_event_token_ids = summary_event_token_ids(&datasets.train);
        let steps_per_epoch = datasets.train.steps_per_epoch(DatasetSplit::Train);
        let train_schedule = resolve_train_schedule(&config.training, steps_per_epoch)?;
        let total_steps = train_schedule.total_steps.max(1);
        let scheduler_iters = match train_schedule.source {
            burn_dragon_train::train::pipeline::ScheduleSource::Epochs => Some(total_steps),
            burn_dragon_train::train::pipeline::ScheduleSource::MaxIters => None,
        };

        let train_loader = build_train_loader::<B>(
            &datasets,
            &config,
            train_schedule.steps_per_epoch,
            total_steps,
            &device,
            summary_event_token_ids.clone(),
        );
        let valid_device = device.clone();
        let validation_loader = build_valid_loader::<B>(
            &datasets,
            &config,
            &valid_device,
            summary_event_token_ids.clone(),
        );
        let learner = build_language_learner::<B>(
            &config,
            backend_label,
            &model_config,
            total_steps,
            scheduler_iters,
            &device,
        )?;
        let validation_for_eval = validation_loader.clone();
        let backend_label_owned = backend_label.to_owned();
        let estimated_tokens_per_second = footprint.estimated_tokens_per_second;
        let mut builder = from_loaders(learner, device.clone(), train_loader, validation_loader)
            .with_benchmark(move |model, _device| {
                let inventory = burn_p2p::burn::inspect_module::<B, _>(model);
                burn_p2p::CapabilityEstimate {
                    preferred_backends: vec![backend_label_owned.clone()],
                    work_units_per_second: estimated_tokens_per_second
                        .max((inventory.parameter_count.max(1) as f64).sqrt()),
                    target_window_seconds: 30,
                }
            })
            .with_evaluate(move |model, split| {
                language_evaluate::<B>(model, validation_for_eval.clone(), split)
            })
            .with_step_metrics(|step_index, output, metrics| {
                metrics.insert(
                    "train_steps".into(),
                    MetricValue::Integer((step_index + 1) as i64),
                );
                metrics.insert(
                    "train_loss".into(),
                    MetricValue::Float(mean_loss_from_train_output_ref(output)),
                );
                Ok(())
            });

        if let Some(shard_export) = &native.shard_export {
            builder = attach_sharded_dataset::<B>(
                builder,
                experiment_kind,
                &config.dataset.source,
                &datasets,
                shard_export,
                summary_event_token_ids.clone(),
            )?;
        }
        (
            builder.build()?,
            dataset_view_id_for_dataset(&datasets.train, native.shard_export.as_ref())?,
        )
    };

    let git_commit = native.git_commit.as_deref().unwrap_or("unknown");
    let enabled_features = native
        .enabled_features_label
        .as_deref()
        .unwrap_or(backend_label);
    let mut manifest_seed = resolved.manifest_seed;
    let effective_seed_node_urls = native.effective_seed_node_urls();
    if !effective_seed_node_urls.is_empty() {
        manifest_seed.bootstrap_addrs = effective_seed_node_urls;
    }
    let manifests = build_manifest_bundle(
        &manifest_seed,
        experiment_kind,
        backend_label,
        &model_config,
        &resolved.profile,
        dataset_view_id,
        &footprint,
        native.app_semver.clone(),
        git_commit,
        enabled_features,
    )?;

    let auth_available = auth_bundle.is_some()
        || native.auth.as_ref().is_some_and(|auth| {
            auth.local_peer_auth.is_some() && !auth.trust_bundle_endpoints.is_empty()
        });
    if !auth_available {
        bail!("burn_dragon_p2p peers require a GitHub-authenticated auth bundle");
    }

    let mut node_builder = connect(
        target_decision.burn_target(),
        manifests.release_manifest.clone(),
        project.clone(),
        manifests.workload_config.clone(),
    )?;
    node_builder = node_builder
        .with_mainnet(burn_p2p::GenesisSpec {
            network_id: manifests.network_manifest.network_id.clone(),
            protocol_version: semver::Version::new(
                u64::from(manifests.network_manifest.protocol_major),
                0,
                0,
            ),
            display_name: manifests.network_manifest.description.clone(),
            created_at: manifests.network_manifest.created_at,
            metadata: Default::default(),
        })
        .with_storage(native.storage_root.clone())
        .with_identity(native.identity.clone());
    let mut node_builder = node_builder.with_network(manifests.network_manifest.clone())?;
    for peer in native.effective_bootstrap_peers()? {
        node_builder = node_builder.with_bootstrap_peer(peer);
    }
    let auth_config = compose_auth_config(
        native.auth.clone(),
        auth_bundle,
        &manifests.experiment_directory,
    );
    node_builder = node_builder.with_auth(auth_config);
    if let Some(upstream) = dataset_upstream {
        node_builder = node_builder.with_dataset(upstream);
    }

    Ok(PreparedNativePeer {
        project,
        builder: node_builder,
        manifests,
        config,
        storage_root: native.storage_root.clone(),
        experiment_kind,
        backend_label: backend_label.to_owned(),
        model_config,
        footprint,
        target_decision,
    })
}
