use std::collections::{BTreeMap, BTreeSet};

use burn_dragon_language::BDHConfig;
use burn_p2p::burn::{BurnArtifactConfig, BurnRecordPrecision, BurnWorkloadConfig};
use burn_p2p::{
    BrowserRolePolicy, BrowserVisibilityPolicy, ChunkingScheme, ClientPlatform,
    ClientReleaseManifest, ContentId, DatasetViewId, ExperimentDirectoryEntry,
    ExperimentDirectoryPolicyExt, ExperimentId, ExperimentOptInPolicy,
    ExperimentResourceRequirements, ExperimentScope, ExperimentVisibility, NetworkId,
    NetworkManifest, PeerRole, PeerRoleSet, Precision, ProjectFamilyId, RevisionId,
    RevisionManifest, StudyId, SupportedWorkload, WindowActivation, WindowId, WorkloadId,
};
use chrono::Utc;
use sha2::{Digest, Sha256};

use crate::capability::DragonTrainingFootprint;
use crate::config::{DragonExperimentKind, DragonManifestBundle, DragonManifestSeed};
use crate::profile::DragonExperimentProfile;

fn stable_content_id<T: serde::Serialize>(label: &str, value: &T) -> ContentId {
    let bytes = serde_json::to_vec(value).expect("stable content id json");
    let mut hasher = Sha256::new();
    hasher.update(label.as_bytes());
    hasher.update([0]);
    hasher.update(bytes);
    ContentId::new(format!("{label}-{:x}", hasher.finalize()))
}

fn backend_resource_class(backend_label: &str) -> String {
    if backend_label.eq_ignore_ascii_case("cpu") || backend_label.eq_ignore_ascii_case("ndarray") {
        "cpu".into()
    } else if backend_label.eq_ignore_ascii_case("cuda") {
        "cuda".into()
    } else {
        "wgpu".into()
    }
}

fn trainer_minimum_role(backend_label: &str) -> PeerRole {
    if backend_label.eq_ignore_ascii_case("cpu") || backend_label.eq_ignore_ascii_case("ndarray") {
        PeerRole::TrainerCpu
    } else {
        PeerRole::TrainerGpu
    }
}

fn minimum_device_memory_bytes(
    backend_label: &str,
    footprint: &DragonTrainingFootprint,
) -> Option<u64> {
    match trainer_minimum_role(backend_label) {
        PeerRole::TrainerCpu => None,
        PeerRole::TrainerGpu => Some(footprint.estimated_training_bytes),
        _ => None,
    }
}

fn minimum_system_memory_bytes(backend_label: &str, footprint: &DragonTrainingFootprint) -> u64 {
    let floor = 512 * 1024 * 1024;
    match trainer_minimum_role(backend_label) {
        PeerRole::TrainerCpu => footprint.estimated_training_bytes.max(floor),
        PeerRole::TrainerGpu => footprint
            .estimated_checkpoint_bytes
            .saturating_add(footprint.estimated_shard_bytes)
            .max(floor),
        _ => floor,
    }
}

pub fn build_manifest_bundle(
    seed: &DragonManifestSeed,
    experiment_kind: DragonExperimentKind,
    backend_label: &str,
    model_config: &BDHConfig,
    profile: &DragonExperimentProfile,
    dataset_view_id: DatasetViewId,
    footprint: &DragonTrainingFootprint,
    app_semver: semver::Version,
    git_commit: &str,
    enabled_features_label: &str,
) -> anyhow::Result<DragonManifestBundle> {
    let workload_id = WorkloadId::new(format!(
        "dragon-{}-{backend_label}",
        experiment_kind.workload_slug()
    ));
    let model_program_hash = stable_content_id(
        "dragon-model-program",
        &serde_json::json!({
            "arch": "dragon_bdh",
            "n_embd": model_config.n_embd,
            "n_head": model_config.n_head,
            "n_layer": model_config.n_layer,
            "latent_total": model_config.latent_total(),
            "latent_per_head": model_config.latent_per_head(),
            "sequence_kernel": model_config.sequence_kernel,
            "vocab_size": model_config.vocab_size,
            "backend": backend_label,
        }),
    );
    let checkpoint_format_hash = stable_content_id(
        "dragon-checkpoint-format",
        &serde_json::json!({
            "format": "named_mpk",
            "precision": "full",
        }),
    );
    let revision_family_hash = stable_content_id(
        "dragon-revision-family",
        &serde_json::json!({
            "experiment_kind": experiment_kind,
            "backend": backend_label,
        }),
    );
    let supported_workload = SupportedWorkload {
        workload_id: workload_id.clone(),
        workload_name: format!(
            "burn_dragon {} ({backend_label})",
            experiment_kind.display_name()
        ),
        model_program_hash,
        checkpoint_format_hash: checkpoint_format_hash.clone(),
        supported_revision_family: revision_family_hash,
        resource_class: backend_resource_class(backend_label),
    };
    let release_train_hash = stable_content_id(
        "dragon-release-train",
        &serde_json::json!({
            "project_family_id": seed.project_family_id,
            "backend": backend_label,
            "experiment_kind": experiment_kind,
            "app_semver": app_semver,
        }),
    );
    let target_artifact_id = if backend_label.eq_ignore_ascii_case("cuda") {
        "native-cuda"
    } else if backend_label.eq_ignore_ascii_case("wgpu") {
        "native-wgpu"
    } else {
        "native-cpu"
    };
    let target_platform = ClientPlatform::Native;
    let target_artifact_hash = stable_content_id(
        "dragon-target-artifact",
        &serde_json::json!({
            "target_artifact_id": target_artifact_id,
            "target_platform": target_platform,
            "release_train_hash": release_train_hash,
        }),
    );
    let release_manifest = ClientReleaseManifest {
        project_family_id: ProjectFamilyId::new(&seed.project_family_id),
        release_train_hash: release_train_hash.clone(),
        target_artifact_id: target_artifact_id.into(),
        target_artifact_hash: target_artifact_hash.clone(),
        target_platform,
        app_semver,
        git_commit: git_commit.into(),
        cargo_lock_hash: stable_content_id("dragon-cargo-lock", &"workspace"),
        burn_version_string: "0.21.0-pre.3".into(),
        enabled_features_hash: stable_content_id("dragon-features", &enabled_features_label),
        protocol_major: seed.protocol_major,
        supported_workloads: vec![supported_workload.clone()],
        built_at: Utc::now(),
    };
    let network_manifest = NetworkManifest {
        network_id: NetworkId::new(&seed.network_id),
        project_family_id: release_manifest.project_family_id.clone(),
        protocol_major: seed.protocol_major,
        required_release_train_hash: release_train_hash,
        allowed_target_artifact_hashes: BTreeSet::from([target_artifact_hash]),
        authority_public_keys: seed.authority_public_keys.clone(),
        bootstrap_addrs: seed.bootstrap_addrs.clone(),
        auth_policy_hash: stable_content_id("dragon-auth-policy", &seed.project_family_id),
        created_at: Utc::now(),
        description: seed.description.clone(),
    };
    let experiment_id = ExperimentId::new(&seed.experiment_id);
    let resource_requirements = ExperimentResourceRequirements {
        minimum_roles: BTreeSet::from([trainer_minimum_role(backend_label)]),
        minimum_device_memory_bytes: minimum_device_memory_bytes(backend_label, footprint),
        minimum_system_memory_bytes: Some(minimum_system_memory_bytes(backend_label, footprint)),
        estimated_download_bytes: footprint
            .estimated_checkpoint_bytes
            .saturating_add(footprint.estimated_shard_bytes),
        estimated_window_seconds: 30,
    };
    let allowed_roles = PeerRoleSet::new([
        trainer_minimum_role(backend_label),
        PeerRole::Validator,
        PeerRole::Viewer,
        PeerRole::BrowserObserver,
        PeerRole::BrowserVerifier,
        PeerRole::BrowserTrainerWgpu,
    ]);
    let allowed_scopes = BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Discover,
        ExperimentScope::Train {
            experiment_id: experiment_id.clone(),
        },
        ExperimentScope::Validate {
            experiment_id: experiment_id.clone(),
        },
    ]);
    let mut metadata = BTreeMap::from([
        (
            "experiment_kind".into(),
            experiment_kind.workload_slug().into(),
        ),
        ("backend".into(), backend_label.into()),
        (
            "estimated_training_bytes".into(),
            footprint.estimated_training_bytes.to_string(),
        ),
        (
            "estimated_checkpoint_bytes".into(),
            footprint.estimated_checkpoint_bytes.to_string(),
        ),
        (
            "estimated_shard_bytes".into(),
            footprint.estimated_shard_bytes.to_string(),
        ),
        (
            "estimated_tokens_per_second".into(),
            format!("{:.1}", footprint.estimated_tokens_per_second),
        ),
    ]);
    metadata.extend(profile.metadata_entries()?);

    let mut experiment_directory_entry = ExperimentDirectoryEntry {
        network_id: network_manifest.network_id.clone(),
        study_id: StudyId::new(&seed.study_id),
        experiment_id: experiment_id.clone(),
        workload_id: workload_id.clone(),
        display_name: seed.display_name.clone(),
        model_schema_hash: stable_content_id("dragon-model-schema", &model_config),
        dataset_view_id,
        resource_requirements,
        visibility: ExperimentVisibility::Public,
        opt_in_policy: ExperimentOptInPolicy::Open,
        current_revision_id: RevisionId::new(&seed.revision_id),
        current_head_id: None,
        allowed_roles,
        allowed_scopes,
        metadata,
    };
    experiment_directory_entry.apply_revision_policy(&RevisionManifest {
        experiment_id: experiment_id.clone(),
        revision_id: RevisionId::new(&seed.revision_id),
        workload_id: workload_id.clone(),
        required_release_train_hash: release_manifest.release_train_hash.clone(),
        model_schema_hash: experiment_directory_entry.model_schema_hash.clone(),
        checkpoint_format_hash: checkpoint_format_hash.clone(),
        dataset_view_id: experiment_directory_entry.dataset_view_id.clone(),
        training_config_hash: stable_content_id(
            "dragon-training-config",
            &serde_json::json!({
                "experiment_kind": experiment_kind,
                "backend": backend_label,
                "vocab_size": model_config.vocab_size,
            }),
        ),
        merge_topology_policy_hash: stable_content_id(
            "dragon-merge-topology",
            &serde_json::json!({
                "strategy": "microcohort-reduce-plus-validator-promotion",
                "window_secs": 30,
            }),
        ),
        slot_requirements: experiment_directory_entry.resource_requirements.clone(),
        activation_window: WindowActivation {
            activation_window: WindowId(0),
            grace_windows: 0,
        },
        lag_policy: Default::default(),
        merge_window_miss_policy: Default::default(),
        robustness_policy: None,
        browser_enabled: true,
        browser_role_policy: BrowserRolePolicy {
            observer: true,
            verifier: true,
            trainer_wgpu: true,
            fallback: true,
        },
        max_browser_checkpoint_bytes: Some(footprint.estimated_checkpoint_bytes),
        max_browser_window_secs: Some(30),
        max_browser_shard_bytes: Some(footprint.estimated_shard_bytes),
        requires_webgpu: true,
        max_browser_batch_size: Some(8),
        recommended_browser_precision: Some(Precision::Fp16),
        visibility_policy: BrowserVisibilityPolicy::SwarmEligible,
        description: seed.description.clone(),
    });
    let experiment_directory = vec![experiment_directory_entry];
    let workload_config = BurnWorkloadConfig::new(
        supported_workload.clone(),
        BurnArtifactConfig::named_mpk(BurnRecordPrecision::Full, ChunkingScheme::default()),
    )
    .with_root_ema(BurnWorkloadConfig::standard_root_ema_decay());
    Ok(DragonManifestBundle {
        release_manifest,
        network_manifest,
        supported_workload,
        experiment_directory,
        workload_config,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use semver::Version;

    fn seed() -> DragonManifestSeed {
        DragonManifestSeed {
            project_family_id: "dragon-family".into(),
            network_id: "dragon-net".into(),
            study_id: "dragon-study".into(),
            experiment_id: "dragon-exp".into(),
            revision_id: "r1".into(),
            display_name: "dragon".into(),
            description: "dragon".into(),
            protocol_major: 0,
            authority_public_keys: Vec::new(),
            bootstrap_addrs: Vec::new(),
        }
    }

    #[test]
    fn gpu_manifests_publish_device_memory_requirements() {
        let model_config = BDHConfig::default();
        let footprint = DragonTrainingFootprint {
            estimated_parameter_bytes: 1024,
            estimated_optimizer_state_bytes: 2048,
            estimated_activation_bytes: 4096,
            estimated_training_bytes: 8192,
            estimated_checkpoint_bytes: 4096,
            estimated_shard_bytes: 2048,
            estimated_tokens_per_second: 1234.0,
        };
        let bundle = build_manifest_bundle(
            &seed(),
            DragonExperimentKind::NcaPrepretraining,
            "wgpu",
            &model_config,
            &DragonExperimentProfile {
                version: 1,
                experiment_kind: DragonExperimentKind::NcaPrepretraining,
                native: crate::profile::DragonNativeExperimentProfile {
                    training_toml: String::new(),
                    nca_corpus_toml: None,
                },
                browser: None,
            },
            DatasetViewId::new("dataset-view"),
            &footprint,
            Version::new(0, 5, 0),
            "test",
            "native,wgpu",
        )
        .expect("manifest bundle");
        let requirements = &bundle.experiment_directory[0].resource_requirements;
        assert_eq!(
            requirements.minimum_device_memory_bytes,
            Some(footprint.estimated_training_bytes)
        );
        assert_eq!(
            requirements.minimum_system_memory_bytes,
            Some(
                footprint
                    .estimated_checkpoint_bytes
                    .saturating_add(footprint.estimated_shard_bytes)
                    .max(512 * 1024 * 1024)
            )
        );
        assert_eq!(
            requirements.estimated_download_bytes,
            footprint
                .estimated_checkpoint_bytes
                .saturating_add(footprint.estimated_shard_bytes)
        );
    }
}
