use std::collections::BTreeSet;
use std::path::PathBuf;

#[cfg(feature = "wasm-peer")]
use burn_dragon_core::BDHConfig;
#[cfg(feature = "wasm-peer")]
use burn_dragon_universality::NcaCorpusConfig;
#[cfg(feature = "native")]
use burn_p2p::NetworkManifest;
use burn_p2p::{AuthConfig, ExperimentScope, IdentityConfig, PeerRole, PeerRoleSet, SwarmAddress};
use semver::Version;
use serde::{Deserialize, Serialize};
use url::form_urlencoded;

const GIB: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonCapabilityPolicy {
    #[serde(default = "default_native_cpu_memory_budget_bytes")]
    pub native_cpu_memory_budget_bytes: Option<u64>,
    #[serde(default = "default_native_wgpu_memory_budget_bytes")]
    pub native_wgpu_memory_budget_bytes: Option<u64>,
    #[serde(default = "default_native_cuda_memory_budget_bytes")]
    pub native_cuda_memory_budget_bytes: Option<u64>,
    #[serde(default = "default_browser_wgpu_memory_budget_bytes")]
    pub browser_wgpu_memory_budget_bytes: Option<u64>,
    #[serde(default = "default_allow_native_validator_fallback")]
    pub allow_native_validator_fallback: bool,
    #[serde(default = "default_allow_browser_verifier_fallback")]
    pub allow_browser_verifier_fallback: bool,
}

impl Default for DragonCapabilityPolicy {
    fn default() -> Self {
        Self {
            native_cpu_memory_budget_bytes: default_native_cpu_memory_budget_bytes(),
            native_wgpu_memory_budget_bytes: default_native_wgpu_memory_budget_bytes(),
            native_cuda_memory_budget_bytes: default_native_cuda_memory_budget_bytes(),
            browser_wgpu_memory_budget_bytes: default_browser_wgpu_memory_budget_bytes(),
            allow_native_validator_fallback: default_allow_native_validator_fallback(),
            allow_browser_verifier_fallback: default_allow_browser_verifier_fallback(),
        }
    }
}

impl DragonCapabilityPolicy {
    pub fn memory_budget_bytes(
        &self,
        capability_class: crate::capability::DragonCapabilityClass,
    ) -> Option<u64> {
        match capability_class {
            crate::capability::DragonCapabilityClass::NativeCpu => {
                self.native_cpu_memory_budget_bytes
            }
            crate::capability::DragonCapabilityClass::NativeWgpu => {
                self.native_wgpu_memory_budget_bytes
            }
            crate::capability::DragonCapabilityClass::NativeCuda => {
                self.native_cuda_memory_budget_bytes
            }
            crate::capability::DragonCapabilityClass::BrowserCpu => {
                self.native_cpu_memory_budget_bytes
            }
            crate::capability::DragonCapabilityClass::BrowserWgpu => {
                self.browser_wgpu_memory_budget_bytes
            }
        }
    }
}

fn default_native_cpu_memory_budget_bytes() -> Option<u64> {
    Some(8 * GIB)
}

fn default_native_wgpu_memory_budget_bytes() -> Option<u64> {
    Some(4 * GIB)
}

fn default_native_cuda_memory_budget_bytes() -> Option<u64> {
    Some(6 * GIB)
}

fn default_browser_wgpu_memory_budget_bytes() -> Option<u64> {
    Some(2 * GIB)
}

fn default_allow_native_validator_fallback() -> bool {
    true
}

fn default_allow_browser_verifier_fallback() -> bool {
    true
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonExperimentKind {
    NcaPrepretraining,
    ClimbMixPretraining,
}

impl DragonExperimentKind {
    pub fn workload_slug(self) -> &'static str {
        match self {
            Self::NcaPrepretraining => "nca-prepretraining",
            Self::ClimbMixPretraining => "climbmix-pretraining",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            Self::NcaPrepretraining => "NCA pre-pre-training",
            Self::ClimbMixPretraining => "ClimbMix pre-training",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonNativeTarget {
    Auto,
    Trainer,
    Validator,
    Reducer,
}

impl DragonNativeTarget {
    pub fn roles(self, gpu: bool) -> PeerRoleSet {
        match self {
            Self::Auto | Self::Trainer => {
                if gpu {
                    PeerRoleSet::new([PeerRole::TrainerGpu])
                } else {
                    PeerRoleSet::new([PeerRole::TrainerCpu])
                }
            }
            Self::Validator => {
                PeerRoleSet::new([PeerRole::Authority, PeerRole::Validator, PeerRole::Archive])
            }
            Self::Reducer => PeerRoleSet::new([PeerRole::Reducer]),
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonPeerNetworkConfig {
    #[serde(default)]
    pub edge_base_url: Option<String>,
    #[serde(default)]
    pub seed_node_urls: Option<Vec<String>>,
}

impl DragonPeerNetworkConfig {
    pub fn parse_seed_node_list(input: &str) -> Option<Vec<String>> {
        let mut seeds = Vec::new();
        for value in input.split(',') {
            let trimmed = value.trim();
            if !trimmed.is_empty() && !seeds.iter().any(|existing| existing == trimmed) {
                seeds.push(trimmed.to_owned());
            }
        }
        (!seeds.is_empty()).then_some(seeds)
    }

    pub fn edge_base_url(&self) -> Option<&str> {
        self.edge_base_url
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty())
    }

    pub fn seed_node_urls(&self) -> &[String] {
        self.seed_node_urls.as_deref().unwrap_or(&[])
    }

    pub fn normalized(mut self) -> Self {
        self.edge_base_url = self
            .edge_base_url
            .take()
            .map(|url| url.trim().trim_end_matches('/').to_owned())
            .filter(|url| !url.is_empty());
        self.seed_node_urls = self
            .seed_node_urls
            .take()
            .map(|urls| {
                let mut normalized = Vec::new();
                for url in urls {
                    let trimmed = url.trim();
                    if trimmed.is_empty() {
                        continue;
                    }
                    if !normalized.iter().any(|existing| existing == trimmed) {
                        normalized.push(trimmed.to_owned());
                    }
                }
                normalized
            })
            .filter(|urls| !urls.is_empty());
        self
    }

    pub fn with_edge_base_url(mut self, edge_base_url: Option<String>) -> Self {
        self.edge_base_url = edge_base_url;
        self.normalized()
    }

    pub fn with_seed_node_urls(mut self, seed_node_urls: Option<Vec<String>>) -> Self {
        self.seed_node_urls = seed_node_urls;
        self.normalized()
    }

    pub fn merged_with(
        &self,
        edge_base_url: Option<String>,
        seed_node_urls: Option<Vec<String>>,
    ) -> Self {
        let mut merged = self.clone();
        if edge_base_url.is_some() {
            merged.edge_base_url = edge_base_url;
        }
        if seed_node_urls.is_some() {
            merged.seed_node_urls = seed_node_urls;
        }
        merged.normalized()
    }

    pub fn parse_seed_node_query(query: &str) -> Option<Vec<String>> {
        let mut seeds = Vec::new();
        for (key, value) in form_urlencoded::parse(query.trim_start_matches('?').as_bytes()) {
            let key = key.as_ref();
            if !matches!(key, "seed" | "seed_url" | "seed_node_url" | "seed_node") {
                continue;
            }
            if let Some(values) = Self::parse_seed_node_list(value.as_ref()) {
                for parsed in values {
                    if !seeds.iter().any(|existing| existing == &parsed) {
                        seeds.push(parsed);
                    }
                }
            }
        }
        (!seeds.is_empty()).then_some(seeds)
    }

    pub fn parse_edge_base_url_query(query: &str) -> Option<String> {
        for (key, value) in form_urlencoded::parse(query.trim_start_matches('?').as_bytes()) {
            if matches!(key.as_ref(), "edge" | "edge_url" | "edge_base_url") {
                let trimmed = value.trim();
                if !trimmed.is_empty() {
                    return Some(trimmed.trim_end_matches('/').to_owned());
                }
            }
        }
        None
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonManifestSeed {
    pub project_family_id: String,
    pub network_id: String,
    pub study_id: String,
    pub experiment_id: String,
    pub revision_id: String,
    pub display_name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub protocol_major: u16,
    #[serde(default)]
    pub authority_public_keys: Vec<String>,
    #[serde(default)]
    pub bootstrap_addrs: Vec<String>,
}

impl Default for DragonManifestSeed {
    fn default() -> Self {
        Self {
            project_family_id: "burn-dragon-language".into(),
            network_id: "burn-dragon-net".into(),
            study_id: "burn-dragon-study".into(),
            experiment_id: "language-pretraining".into(),
            revision_id: "r1".into(),
            display_name: "burn_dragon language pretraining".into(),
            description: "burn_dragon peer-to-peer language training network".into(),
            protocol_major: 0,
            authority_public_keys: Vec::new(),
            bootstrap_addrs: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonShardExportConfig {
    pub root: PathBuf,
    #[serde(default)]
    pub dataset_name: Option<String>,
    #[serde(default)]
    pub microshards: Option<u32>,
    #[serde(default)]
    pub max_records: Option<usize>,
    #[serde(default)]
    pub http_upstream: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonExistingShardDatasetConfig {
    pub root: PathBuf,
    #[serde(default)]
    pub http_upstream: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonNativePeerConfig {
    pub training_config_paths: Vec<PathBuf>,
    pub storage_root: PathBuf,
    #[serde(default)]
    pub network: DragonPeerNetworkConfig,
    #[serde(default)]
    pub target: Option<DragonNativeTarget>,
    #[serde(default)]
    pub identity: IdentityConfig,
    #[serde(default)]
    pub bootstrap_peers: Vec<SwarmAddress>,
    pub manifest: DragonManifestSeed,
    #[serde(default = "default_app_semver")]
    pub app_semver: Version,
    #[serde(default)]
    pub git_commit: Option<String>,
    #[serde(default)]
    pub enabled_features_label: Option<String>,
    #[serde(default)]
    pub auth: Option<AuthConfig>,
    #[serde(default)]
    pub capability_policy: DragonCapabilityPolicy,
    #[serde(default)]
    pub shard_export: Option<DragonShardExportConfig>,
    #[serde(default)]
    pub existing_shard_dataset: Option<DragonExistingShardDatasetConfig>,
}

fn default_app_semver() -> Version {
    Version::new(0, 5, 0)
}

impl DragonNativePeerConfig {
    pub fn target_or_default(&self) -> DragonNativeTarget {
        self.target.unwrap_or(DragonNativeTarget::Auto)
    }

    pub fn effective_edge_base_url(&self) -> Option<&str> {
        self.network.edge_base_url()
    }

    pub fn effective_seed_node_urls(&self) -> Vec<String> {
        if !self.network.seed_node_urls().is_empty() {
            return self.network.seed_node_urls().to_vec();
        }
        self.manifest.bootstrap_addrs.clone()
    }

    pub fn effective_bootstrap_peers(&self) -> anyhow::Result<Vec<SwarmAddress>> {
        let mut peers = self.bootstrap_peers.clone();
        for url in self.effective_seed_node_urls() {
            let address = SwarmAddress::new(url.clone())
                .map_err(|error| anyhow::anyhow!("invalid seed node url `{url}`: {error}"))?;
            if !peers.iter().any(|existing| existing == &address) {
                peers.push(address);
            }
        }
        Ok(peers)
    }

    pub fn with_network_overrides(
        mut self,
        edge_base_url: Option<String>,
        seed_node_urls: Option<Vec<String>>,
    ) -> Self {
        self.network = self.network.merged_with(edge_base_url, seed_node_urls);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonBrowserAppConfig {
    #[serde(default)]
    pub network: DragonPeerNetworkConfig,
    #[serde(default)]
    pub selected_experiment_id: Option<String>,
    #[serde(default)]
    pub selected_revision_id: Option<String>,
    #[serde(default)]
    pub requested_scopes: BTreeSet<ExperimentScope>,
    #[serde(default)]
    pub require_github_auth: bool,
    #[cfg(feature = "wasm-peer")]
    #[serde(default)]
    pub training: Option<DragonBrowserTrainingConfig>,
}

impl DragonBrowserAppConfig {
    pub fn selected_experiment(&self) -> Option<(String, Option<String>)> {
        self.selected_experiment_id
            .as_ref()
            .map(|experiment_id| (experiment_id.clone(), self.selected_revision_id.clone()))
    }

    pub fn effective_edge_base_url(&self) -> Option<&str> {
        self.network.edge_base_url()
    }

    pub fn effective_seed_node_urls(&self) -> &[String] {
        self.network.seed_node_urls()
    }

    pub fn with_network_overrides(
        mut self,
        edge_base_url: Option<String>,
        seed_node_urls: Option<Vec<String>>,
    ) -> Self {
        self.network = self.network.merged_with(edge_base_url, seed_node_urls);
        self
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonNativeAuthBundle {
    pub auth_config: AuthConfig,
    pub trust_bundle_endpoint: String,
    #[serde(default)]
    pub edge_base_url: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub principal_id: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenWindowRecord {
    pub inputs: Vec<i64>,
    pub targets: Vec<i64>,
    pub reset_stream_state: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonBrowserDatasetSplit {
    Train,
    Validation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonBrowserShardSelectionPolicy {
    Sequential,
    #[default]
    DeterministicPeer,
}

#[cfg(feature = "wasm-peer")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DragonBrowserTokenSource {
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
        corpus: NcaCorpusConfig,
        split: DragonBrowserDatasetSplit,
        #[serde(default)]
        max_documents: Option<usize>,
    },
}

#[cfg(feature = "wasm-peer")]
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DragonBrowserLiveParticipantConfig {
    pub principal_id: String,
    pub study_id: String,
    pub experiment_id: String,
    pub revision_id: String,
    pub workload_id: String,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DragonBrowserExecutionBackend {
    #[default]
    Auto,
    Cpu,
    Wgpu,
}

impl DragonBrowserExecutionBackend {
    pub fn backend_label(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Wgpu => "wgpu",
            Self::Auto => {
                if cfg!(feature = "wgpu") {
                    "wgpu"
                } else {
                    "cpu"
                }
            }
        }
    }
}

#[cfg(feature = "wasm-peer")]
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DragonBrowserTrainingConfig {
    pub experiment_kind: DragonExperimentKind,
    pub model_config: BDHConfig,
    #[serde(default)]
    pub execution_backend: DragonBrowserExecutionBackend,
    #[serde(default = "default_browser_block_size")]
    pub block_size: usize,
    #[serde(default = "default_browser_learning_rate")]
    pub learning_rate: f64,
    #[serde(default)]
    pub weight_decay: f32,
    #[serde(default = "default_browser_batch_size")]
    pub batch_size: usize,
    #[serde(default)]
    pub max_train_batches: Option<usize>,
    #[serde(default)]
    pub max_eval_batches: Option<usize>,
    #[serde(default)]
    pub capability_policy: DragonCapabilityPolicy,
    #[serde(default)]
    pub training_lease: Option<burn_p2p::WorkloadTrainingLease>,
    pub train_source: DragonBrowserTokenSource,
    #[serde(default)]
    pub eval_source: Option<DragonBrowserTokenSource>,
    #[serde(default)]
    pub live_participant: Option<DragonBrowserLiveParticipantConfig>,
}

#[cfg(feature = "wasm-peer")]
fn default_browser_learning_rate() -> f64 {
    1.0e-3
}

#[cfg(feature = "wasm-peer")]
fn default_browser_block_size() -> usize {
    128
}

#[cfg(feature = "wasm-peer")]
fn default_browser_batch_size() -> usize {
    4
}

#[derive(Clone, Debug)]
#[cfg(feature = "native")]
pub struct DragonManifestBundle {
    pub release_manifest: burn_p2p::ClientReleaseManifest,
    pub network_manifest: NetworkManifest,
    pub supported_workload: burn_p2p::SupportedWorkload,
    pub experiment_directory: Vec<burn_p2p::ExperimentDirectoryEntry>,
    pub workload_config: burn_p2p::burn::BurnWorkloadConfig,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_node_list_parser_normalizes_and_deduplicates() {
        assert_eq!(
            DragonPeerNetworkConfig::parse_seed_node_list(
                " /dnsaddr/seed-1 , /dnsaddr/seed-2,/dnsaddr/seed-1 , "
            ),
            Some(vec!["/dnsaddr/seed-1".into(), "/dnsaddr/seed-2".into(),])
        );
        assert_eq!(DragonPeerNetworkConfig::parse_seed_node_list(" , "), None);
    }

    #[test]
    fn peer_network_query_parsing_handles_edge_and_seed_urls() {
        let query = "?edge=https%3A%2F%2Fedge.example%2F&seed=/dnsaddr/seed-1&seed=/dnsaddr/seed-2,/dnsaddr/seed-3";
        assert_eq!(
            DragonPeerNetworkConfig::parse_edge_base_url_query(query).as_deref(),
            Some("https://edge.example")
        );
        assert_eq!(
            DragonPeerNetworkConfig::parse_seed_node_query(query),
            Some(vec![
                "/dnsaddr/seed-1".into(),
                "/dnsaddr/seed-2".into(),
                "/dnsaddr/seed-3".into(),
            ])
        );
    }

    #[test]
    fn native_peer_prefers_explicit_seed_node_urls_over_manifest_defaults() {
        let config = DragonNativePeerConfig {
            training_config_paths: Vec::new(),
            storage_root: PathBuf::from("tmp"),
            network: DragonPeerNetworkConfig::default().with_seed_node_urls(Some(vec![
                "/dnsaddr/runtime-seed/tcp/4001/p2p/12D3KooWRuntime".into(),
            ])),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: DragonManifestSeed {
                bootstrap_addrs: vec![
                    "/dnsaddr/manifest-seed/tcp/4001/p2p/12D3KooWManifest".into(),
                ],
                ..DragonManifestSeed::default()
            },
            app_semver: Version::new(0, 5, 0),
            git_commit: None,
            enabled_features_label: None,
            auth: None,
            capability_policy: DragonCapabilityPolicy::default(),
            shard_export: None,
            existing_shard_dataset: None,
        };

        assert_eq!(
            config.effective_seed_node_urls(),
            vec!["/dnsaddr/runtime-seed/tcp/4001/p2p/12D3KooWRuntime".to_owned()]
        );
    }

    #[test]
    fn native_target_defaults_to_auto_for_safe_downgrade() {
        let config = DragonNativePeerConfig {
            training_config_paths: Vec::new(),
            storage_root: PathBuf::from("tmp"),
            network: DragonPeerNetworkConfig::default(),
            target: None,
            identity: Default::default(),
            bootstrap_peers: Vec::new(),
            manifest: DragonManifestSeed::default(),
            app_semver: Version::new(0, 5, 0),
            git_commit: None,
            enabled_features_label: None,
            auth: None,
            capability_policy: DragonCapabilityPolicy::default(),
            shard_export: None,
            existing_shard_dataset: None,
        };

        assert_eq!(config.target_or_default(), DragonNativeTarget::Auto);
    }
}
