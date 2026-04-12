use std::collections::BTreeMap;

use anyhow::{Result, anyhow, bail};
use burn_p2p::ExperimentDirectoryEntry;
use serde::{Deserialize, Serialize};

#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

#[cfg(all(not(feature = "native"), feature = "wasm-peer"))]
use burn_dragon_core::BDHConfig;
#[cfg(feature = "native")]
use burn_dragon_language::api::inference::build_model_config_with_tokenizer;
#[cfg(feature = "native")]
use burn_dragon_language::config::ValidationDatasetConfig;
#[cfg(feature = "native")]
use burn_dragon_language::{BDHConfig, DatasetSourceConfig, TrainingConfig, load_training_config};
#[cfg(feature = "native")]
use burn_dragon_universality::NcaCorpusConfig;
#[cfg(feature = "native")]
use burn_p2p::BrowserEdgeSnapshot;

#[cfg(feature = "native")]
use crate::auth::fetch_edge_snapshot;
use crate::config::{
    DragonBrowserDatasetSplit, DragonBrowserExecutionBackend, DragonBrowserShardSelectionPolicy,
    DragonCapabilityPolicy, DragonExperimentKind, TokenWindowRecord,
};
#[cfg(feature = "wasm-peer")]
use crate::config::{
    DragonBrowserLiveParticipantConfig, DragonBrowserTokenSource, DragonBrowserTrainingConfig,
};
#[cfg(feature = "native")]
use crate::config::{DragonManifestSeed, DragonNativePeerConfig};

pub const DRAGON_PROFILE_VERSION_METADATA_KEY: &str = "dragon_profile_version";
pub const DRAGON_PROFILE_JSON_METADATA_KEY: &str = "dragon_profile_json";
const DRAGON_PROFILE_VERSION: u32 = 1;
#[cfg(feature = "native")]
const DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW: usize = 4;
#[cfg(feature = "native")]
const PORTABLE_NCA_CORPUS_FILE_NAME: &str = "nca-corpus.toml";
#[cfg(feature = "native")]
const PORTABLE_CACHE_DIR_NAME: &str = "__dragon_network_profile_cache__";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonExperimentProfile {
    pub version: u32,
    pub experiment_kind: DragonExperimentKind,
    pub native: DragonNativeExperimentProfile,
    #[serde(default)]
    pub browser: Option<DragonBrowserExperimentProfile>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonNativeExperimentProfile {
    pub training_toml: String,
    #[serde(default)]
    pub nca_corpus_toml: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonBrowserExperimentProfile {
    pub model_config: BDHConfig,
    #[serde(default)]
    pub execution_backend: DragonBrowserExecutionBackend,
    pub block_size: usize,
    pub learning_rate: f64,
    #[serde(default)]
    pub weight_decay: f32,
    pub batch_size: usize,
    #[serde(default)]
    pub max_train_batches: Option<usize>,
    #[serde(default)]
    pub max_eval_batches: Option<usize>,
    #[serde(default)]
    pub capability_policy: DragonCapabilityPolicy,
    pub train_source: DragonBrowserProfileTokenSource,
    #[serde(default)]
    pub eval_source: Option<DragonBrowserProfileTokenSource>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DragonBrowserProfileTokenSource {
    Inline {
        records: Vec<TokenWindowRecord>,
    },
    HttpJson {
        url: String,
    },
    ShardManifestHttp {
        manifest_url: String,
        #[serde(default)]
        selection: DragonBrowserShardSelectionPolicy,
        #[serde(default)]
        max_shards_per_window: Option<usize>,
    },
    GeneratedNca {
        corpus_toml: String,
        split: DragonBrowserDatasetSplit,
        #[serde(default)]
        max_documents: Option<usize>,
    },
}

#[cfg(feature = "native")]
#[derive(Clone, Debug)]
pub struct ResolvedNativeTrainingProfile {
    pub config: TrainingConfig,
    pub manifest_seed: DragonManifestSeed,
    pub profile: DragonExperimentProfile,
    pub directory_entry: Option<ExperimentDirectoryEntry>,
}

impl DragonExperimentProfile {
    pub fn metadata_entries(&self) -> Result<BTreeMap<String, String>> {
        Ok(BTreeMap::from([
            (
                DRAGON_PROFILE_VERSION_METADATA_KEY.into(),
                self.version.to_string(),
            ),
            (
                DRAGON_PROFILE_JSON_METADATA_KEY.into(),
                serde_json::to_string(self)?,
            ),
        ]))
    }

    pub fn from_entry_metadata(entry: &ExperimentDirectoryEntry) -> Result<Option<Self>> {
        let Some(profile_json) = entry.metadata.get(DRAGON_PROFILE_JSON_METADATA_KEY) else {
            return Ok(None);
        };
        let profile = serde_json::from_str::<Self>(profile_json).map_err(|error| {
            anyhow!(
                "failed to decode Dragon experiment profile for {}: {error}",
                entry.experiment_id.as_str()
            )
        })?;
        if profile.version != DRAGON_PROFILE_VERSION {
            bail!(
                "unsupported Dragon experiment profile version {} for {}",
                profile.version,
                entry.experiment_id.as_str()
            );
        }
        Ok(Some(profile))
    }
}

pub fn find_matching_entry<'a>(
    entries: &'a [ExperimentDirectoryEntry],
    selected_experiment_id: Option<&str>,
    selected_revision_id: Option<&str>,
    experiment_kind: Option<DragonExperimentKind>,
) -> Option<&'a ExperimentDirectoryEntry> {
    let matches_revision = |entry: &&ExperimentDirectoryEntry| {
        selected_revision_id
            .is_none_or(|revision_id| entry.current_revision_id.as_str() == revision_id)
    };

    if let Some(experiment_id) = selected_experiment_id
        && let Some(entry) = entries
            .iter()
            .filter(|entry| entry.experiment_id.as_str() == experiment_id)
            .find(matches_revision)
    {
        return Some(entry);
    }

    if let Some(experiment_kind) = experiment_kind
        && let Some(entry) = entries
            .iter()
            .filter(|entry| {
                entry.metadata.get("experiment_kind").map(String::as_str)
                    == Some(experiment_kind.workload_slug())
            })
            .find(matches_revision)
    {
        return Some(entry);
    }

    entries
        .iter()
        .filter(|entry| {
            entry
                .metadata
                .contains_key(DRAGON_PROFILE_JSON_METADATA_KEY)
        })
        .find(matches_revision)
}

#[cfg(feature = "native")]
fn ensure_portable_native_profile(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<()> {
    match (&config.dataset.source, experiment_kind) {
        (DatasetSourceConfig::UniversalityNca { .. }, DragonExperimentKind::NcaPrepretraining)
        | (
            DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => {}
        _ => bail!(
            "network-published Dragon profiles currently support only universality_nca and nemotron_climb_mix datasets"
        ),
    }

    if config.training.resume_run_dir.is_some() {
        bail!("network-published Dragon profiles do not support training.resume_run_dir");
    }
    if config.training.init_checkpoint_path.is_some() {
        bail!("network-published Dragon profiles do not support training.init_checkpoint_path");
    }
    if config
        .training
        .init_transfer
        .interface_checkpoint_path
        .is_some()
    {
        bail!(
            "network-published Dragon profiles do not support training.init_transfer.interface_checkpoint_path"
        );
    }
    Ok(())
}

#[cfg(feature = "native")]
fn portable_training_template(
    config: &TrainingConfig,
    nca_corpus_toml: Option<&str>,
) -> Result<String> {
    let mut portable = config.clone();
    portable.dataset.cache_dir = PathBuf::from(PORTABLE_CACHE_DIR_NAME);
    if let Some(validation) = portable.dataset.validation.as_mut() {
        validation.cache_dir = None;
    }
    if nca_corpus_toml.is_some() {
        portable.dataset.source = DatasetSourceConfig::UniversalityNca {
            config: PathBuf::from(PORTABLE_NCA_CORPUS_FILE_NAME),
        };
        if let Some(validation) = portable.dataset.validation.as_mut()
            && matches!(
                validation.source,
                DatasetSourceConfig::UniversalityNca { .. }
            )
        {
            validation.source = DatasetSourceConfig::UniversalityNca {
                config: PathBuf::from(PORTABLE_NCA_CORPUS_FILE_NAME),
            };
        }
    }
    toml::to_string(&portable).map_err(Into::into)
}

#[cfg(feature = "native")]
fn browser_profile_from_native_config(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
    model_config: &BDHConfig,
    revision_id: Option<&str>,
    browser_climbmix_manifest_url: Option<&str>,
) -> Result<Option<DragonBrowserExperimentProfile>> {
    match (&config.dataset.source, experiment_kind) {
        (
            DatasetSourceConfig::UniversalityNca {
                config: nca_config_path,
            },
            DragonExperimentKind::NcaPrepretraining,
        ) => {
            let corpus_toml = fs::read_to_string(nca_config_path).map_err(|error| {
                anyhow!(
                    "failed to read portable NCA corpus config {}: {error}",
                    nca_config_path.display()
                )
            })?;
            Ok(Some(DragonBrowserExperimentProfile {
                model_config: model_config.clone(),
                execution_backend: DragonBrowserExecutionBackend::Auto,
                block_size: config.training.block_size,
                learning_rate: config.optimizer.learning_rate,
                weight_decay: config.optimizer.weight_decay,
                batch_size: config.training.batch_size,
                max_train_batches: Some(config.training.max_iters.max(1)),
                max_eval_batches: Some(config.training.max_iters.clamp(1, 8)),
                capability_policy: DragonCapabilityPolicy::default(),
                train_source: DragonBrowserProfileTokenSource::GeneratedNca {
                    corpus_toml: corpus_toml.clone(),
                    split: DragonBrowserDatasetSplit::Train,
                    max_documents: None,
                },
                eval_source: Some(DragonBrowserProfileTokenSource::GeneratedNca {
                    corpus_toml,
                    split: DragonBrowserDatasetSplit::Validation,
                    max_documents: None,
                }),
            }))
        }
        (
            DatasetSourceConfig::NemotronClimbMix { .. },
            DragonExperimentKind::ClimbMixPretraining,
        ) => Ok(Some(DragonBrowserExperimentProfile {
            model_config: model_config.clone(),
            execution_backend: DragonBrowserExecutionBackend::Auto,
            block_size: config.training.block_size,
            learning_rate: config.optimizer.learning_rate,
            weight_decay: config.optimizer.weight_decay,
            batch_size: config.training.batch_size,
            max_train_batches: Some(config.training.max_iters.max(1)),
            max_eval_batches: None,
            capability_policy: DragonCapabilityPolicy::default(),
            train_source: DragonBrowserProfileTokenSource::ShardManifestHttp {
                manifest_url: browser_climbmix_manifest_url
                    .map(str::trim)
                    .filter(|url| !url.is_empty())
                    .map(ToOwned::to_owned)
                    .unwrap_or_else(|| {
                        default_browser_climbmix_manifest_url(experiment_kind, revision_id)
                    }),
                selection: DragonBrowserShardSelectionPolicy::DeterministicPeer,
                max_shards_per_window: Some(DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW),
            },
            eval_source: None,
        })),
        _ => Ok(None),
    }
}

#[cfg(feature = "native")]
fn default_browser_climbmix_manifest_url(
    experiment_kind: DragonExperimentKind,
    revision_id: Option<&str>,
) -> String {
    match revision_id {
        Some(revision_id) if !revision_id.trim().is_empty() => format!(
            "/dragon-datasets/{}/{}/fetch-manifest.json",
            experiment_kind.workload_slug(),
            revision_id.trim()
        ),
        _ => format!(
            "/dragon-datasets/{}/fetch-manifest.json",
            experiment_kind.workload_slug()
        ),
    }
}

#[cfg(feature = "native")]
pub fn build_profile_from_local_config(
    config: &TrainingConfig,
    experiment_kind: DragonExperimentKind,
    revision_id: Option<&str>,
    browser_climbmix_manifest_url: Option<&str>,
) -> Result<DragonExperimentProfile> {
    ensure_portable_native_profile(config, experiment_kind)?;
    let model_config = build_model_config_with_tokenizer(
        &config.model,
        config.training.block_size,
        config
            .dataset
            .tokenizer
            .fit(std::iter::empty::<&str>())?
            .as_ref(),
    )?;
    let nca_corpus_toml = match &config.dataset.source {
        DatasetSourceConfig::UniversalityNca {
            config: nca_config_path,
        } => Some(fs::read_to_string(nca_config_path).map_err(|error| {
            anyhow!(
                "failed to read portable NCA corpus config {}: {error}",
                nca_config_path.display()
            )
        })?),
        _ => None,
    };
    Ok(DragonExperimentProfile {
        version: DRAGON_PROFILE_VERSION,
        experiment_kind,
        native: DragonNativeExperimentProfile {
            training_toml: portable_training_template(config, nca_corpus_toml.as_deref())?,
            nca_corpus_toml,
        },
        browser: browser_profile_from_native_config(
            config,
            experiment_kind,
            &model_config,
            revision_id,
            browser_climbmix_manifest_url,
        )?,
    })
}

#[cfg(feature = "native")]
fn profile_storage_root(storage_root: &Path, entry: &ExperimentDirectoryEntry) -> PathBuf {
    storage_root
        .join("network_profiles")
        .join(entry.study_id.as_str())
        .join(entry.experiment_id.as_str())
        .join(entry.current_revision_id.as_str())
}

#[cfg(feature = "native")]
fn validation_cache_dir_for(cache_dir: &Path, validation: &mut ValidationDatasetConfig) {
    validation.cache_dir = Some(cache_dir.join("validation"));
}

#[cfg(feature = "native")]
pub fn materialize_native_training_config(
    storage_root: &Path,
    entry: &ExperimentDirectoryEntry,
    profile: &DragonExperimentProfile,
) -> Result<TrainingConfig> {
    let mut config =
        toml::from_str::<TrainingConfig>(&profile.native.training_toml).map_err(|error| {
            anyhow!(
                "failed to decode native Dragon training config for {}: {error}",
                entry.experiment_id.as_str()
            )
        })?;
    let profile_root = profile_storage_root(storage_root, entry);
    let cache_dir = profile_root.join("cache");
    fs::create_dir_all(&cache_dir)?;
    config.dataset.cache_dir = cache_dir.clone();
    if let Some(validation) = config.dataset.validation.as_mut() {
        validation_cache_dir_for(&cache_dir, validation);
    }

    if let Some(corpus_toml) = profile.native.nca_corpus_toml.as_ref() {
        let mut corpus = toml::from_str::<NcaCorpusConfig>(corpus_toml).map_err(|error| {
            anyhow!(
                "failed to decode portable NCA corpus config for {}: {error}",
                entry.experiment_id.as_str()
            )
        })?;
        corpus.output_dir = profile_root.join("nca-generated");
        let corpus_path = profile_root.join(PORTABLE_NCA_CORPUS_FILE_NAME);
        fs::write(&corpus_path, toml::to_string(&corpus)?)?;
        config.dataset.source = DatasetSourceConfig::UniversalityNca {
            config: corpus_path.clone(),
        };
        if let Some(validation) = config.dataset.validation.as_mut()
            && matches!(
                validation.source,
                DatasetSourceConfig::UniversalityNca { .. }
            )
        {
            validation.source = DatasetSourceConfig::UniversalityNca {
                config: corpus_path,
            };
        }
    }

    Ok(config)
}

#[cfg(feature = "native")]
fn manifest_seed_from_entry(
    default_seed: &DragonManifestSeed,
    entry: &ExperimentDirectoryEntry,
) -> DragonManifestSeed {
    let mut seed = default_seed.clone();
    seed.network_id = entry.network_id.as_str().to_owned();
    seed.study_id = entry.study_id.as_str().to_owned();
    seed.experiment_id = entry.experiment_id.as_str().to_owned();
    seed.revision_id = entry.current_revision_id.as_str().to_owned();
    seed.display_name = entry.display_name.clone();
    seed
}

#[cfg(feature = "native")]
fn fetch_matching_profile_entry(
    snapshot: &BrowserEdgeSnapshot,
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
) -> Result<Option<(ExperimentDirectoryEntry, DragonExperimentProfile)>> {
    let Some(entry) = find_matching_entry(
        &snapshot.directory.entries,
        Some(&native.manifest.experiment_id),
        Some(&native.manifest.revision_id),
        Some(experiment_kind),
    ) else {
        return Ok(None);
    };
    let Some(profile) = DragonExperimentProfile::from_entry_metadata(entry)? else {
        return Ok(None);
    };
    Ok(Some((entry.clone(), profile)))
}

#[cfg(feature = "native")]
pub fn resolve_native_training_profile(
    native: &DragonNativePeerConfig,
    experiment_kind: DragonExperimentKind,
    use_network_profile: bool,
) -> Result<ResolvedNativeTrainingProfile> {
    let has_local_training = !native.training_config_paths.is_empty();

    if use_network_profile && let Some(edge_base_url) = native.effective_edge_base_url() {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        match runtime.block_on(fetch_edge_snapshot(edge_base_url)) {
            Ok(snapshot) => {
                if let Some((entry, profile)) =
                    fetch_matching_profile_entry(&snapshot, native, experiment_kind)?
                {
                    let config =
                        materialize_native_training_config(&native.storage_root, &entry, &profile)?;
                    return Ok(ResolvedNativeTrainingProfile {
                        config,
                        manifest_seed: manifest_seed_from_entry(&native.manifest, &entry),
                        profile,
                        directory_entry: Some(entry),
                    });
                }
            }
            Err(error) if !has_local_training => return Err(error),
            Err(_) => {}
        }
    }

    if !has_local_training {
        bail!(
            "no network-published Dragon profile was available and native.training_config_paths is empty"
        );
    }

    let config = load_training_config(&native.training_config_paths)?;
    let profile = build_profile_from_local_config(
        &config,
        experiment_kind,
        Some(&native.manifest.revision_id),
        None,
    )?;
    Ok(ResolvedNativeTrainingProfile {
        config,
        manifest_seed: native.manifest.clone(),
        profile,
        directory_entry: None,
    })
}

#[cfg(feature = "wasm-peer")]
fn browser_source_from_profile(
    source: DragonBrowserProfileTokenSource,
) -> Result<DragonBrowserTokenSource> {
    match source {
        DragonBrowserProfileTokenSource::Inline { records } => {
            Ok(DragonBrowserTokenSource::Inline { records })
        }
        DragonBrowserProfileTokenSource::HttpJson { url } => {
            Ok(DragonBrowserTokenSource::HttpJson { url })
        }
        DragonBrowserProfileTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => Ok(DragonBrowserTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        }),
        DragonBrowserProfileTokenSource::GeneratedNca {
            corpus_toml,
            split,
            max_documents,
        } => Ok(DragonBrowserTokenSource::GeneratedNca {
            corpus: toml::from_str(&corpus_toml)?,
            split,
            max_documents,
        }),
    }
}

#[cfg(feature = "wasm-peer")]
pub fn browser_training_config_from_profile(
    entry: &ExperimentDirectoryEntry,
    profile: &DragonExperimentProfile,
) -> Result<Option<DragonBrowserTrainingConfig>> {
    let Some(browser) = profile.browser.clone() else {
        return Ok(None);
    };
    Ok(Some(DragonBrowserTrainingConfig {
        experiment_kind: profile.experiment_kind,
        model_config: browser.model_config,
        execution_backend: browser.execution_backend,
        block_size: browser.block_size,
        learning_rate: browser.learning_rate,
        weight_decay: browser.weight_decay,
        batch_size: browser.batch_size,
        max_train_batches: browser.max_train_batches,
        max_eval_batches: browser.max_eval_batches,
        capability_policy: browser.capability_policy,
        training_lease: None,
        train_source: browser_source_from_profile(browser.train_source)?,
        eval_source: match browser.eval_source {
            Some(source) => Some(browser_source_from_profile(source)?),
            None => None,
        },
        live_participant: Some(DragonBrowserLiveParticipantConfig {
            principal_id: "browser-live-participant".into(),
            study_id: entry.study_id.as_str().to_owned(),
            experiment_id: entry.experiment_id.as_str().to_owned(),
            revision_id: entry.current_revision_id.as_str().to_owned(),
            workload_id: entry.workload_id.as_str().to_owned(),
        }),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    use burn_p2p::{
        ContentId, DatasetViewId, ExperimentId, ExperimentOptInPolicy,
        ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, NetworkId, PeerRole,
        PeerRoleSet, RevisionId, StudyId, WorkloadId,
    };

    fn sample_entry() -> ExperimentDirectoryEntry {
        ExperimentDirectoryEntry {
            network_id: NetworkId::new("dragon-net"),
            study_id: StudyId::new("dragon-study"),
            experiment_id: ExperimentId::new("nca-prepretraining"),
            workload_id: WorkloadId::new("dragon-nca"),
            display_name: "NCA".into(),
            model_schema_hash: ContentId::new("schema"),
            dataset_view_id: DatasetViewId::new("view"),
            resource_requirements: ExperimentResourceRequirements {
                minimum_roles: BTreeSet::from([PeerRole::TrainerGpu]),
                minimum_device_memory_bytes: None,
                minimum_system_memory_bytes: None,
                estimated_download_bytes: 0,
                estimated_window_seconds: 30,
            },
            visibility: ExperimentVisibility::Public,
            opt_in_policy: ExperimentOptInPolicy::Open,
            current_revision_id: RevisionId::new("r1"),
            current_head_id: None,
            allowed_roles: PeerRoleSet::new([PeerRole::TrainerGpu]),
            allowed_scopes: BTreeSet::from([ExperimentScope::Connect]),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn profile_metadata_round_trip_decodes_from_directory_entry() {
        let profile = DragonExperimentProfile {
            version: DRAGON_PROFILE_VERSION,
            experiment_kind: DragonExperimentKind::NcaPrepretraining,
            native: DragonNativeExperimentProfile {
                training_toml: "[training]\nblock_size = 64\nbatch_size = 2\n".into(),
                nca_corpus_toml: Some("seed = 1337\n".into()),
            },
            browser: None,
        };
        let mut entry = sample_entry();
        entry.metadata.extend(
            profile
                .metadata_entries()
                .expect("profile metadata should serialize"),
        );

        let decoded = DragonExperimentProfile::from_entry_metadata(&entry)
            .expect("profile metadata should decode")
            .expect("profile should be present");

        assert_eq!(decoded, profile);
    }

    #[cfg(feature = "native")]
    #[test]
    fn climbmix_profile_builds_browser_shard_manifest_source() {
        let config: TrainingConfig = toml::from_str(
            r#"
[dataset]
cache_dir = "./cache/climbmix-r1"
train_split_ratio = 0.9
type = "nemotron_climb_mix"
max_records = 256

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = 6
n_embd = 96
n_head = 8
latent_total = 192

[training]
block_size = 128
batch_size = 4
max_iters = 32
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.003
weight_decay = 0.0

[generation]
prompt = "1 2 3"
"#,
        )
        .expect("training config");

        let profile = build_profile_from_local_config(
            &config,
            DragonExperimentKind::ClimbMixPretraining,
            Some("climbmix-r1"),
            None,
        )
        .expect("profile");

        match profile.browser.expect("browser profile").train_source {
            DragonBrowserProfileTokenSource::ShardManifestHttp {
                manifest_url,
                selection,
                max_shards_per_window,
            } => {
                assert_eq!(
                    manifest_url,
                    "/dragon-datasets/climbmix-pretraining/climbmix-r1/fetch-manifest.json"
                );
                assert_eq!(
                    selection,
                    DragonBrowserShardSelectionPolicy::DeterministicPeer
                );
                assert_eq!(
                    max_shards_per_window,
                    Some(DEFAULT_BROWSER_CLIMBMIX_MAX_SHARDS_PER_WINDOW)
                );
            }
            other => panic!("expected shard-manifest browser source, got {other:?}"),
        }
    }
}
