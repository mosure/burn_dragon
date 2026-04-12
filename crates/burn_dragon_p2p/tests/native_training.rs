#![cfg(feature = "native")]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::net::TcpListener;
use std::path::Path;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::thread;
use std::time::Duration;

use burn_dragon_p2p::auth::{
    begin_native_github_login, complete_native_github_login, fetch_edge_snapshot,
};
use burn_dragon_p2p::config::{
    DragonCapabilityPolicy, DragonExistingShardDatasetConfig, DragonManifestSeed,
    DragonNativeAuthBundle, DragonNativePeerConfig, DragonNativeTarget, DragonShardExportConfig,
    TokenWindowRecord,
};
use burn_dragon_p2p::native::{prepare_climbmix_native_cpu, prepare_nca_native_cpu};
use burn_dragon_p2p::profile::{DragonBrowserProfileTokenSource, DragonExperimentProfile};
use burn_p2p::burn::{BurnShardedDataset, BurnShardedDatasetConfig, BurnWorkload};
use burn_p2p::{
    AuthConfig, AuthProvider, BrowserMode, CallbackPayload, ContentId, EdgePeerEnrollmentRequest,
    ExperimentDirectoryEntry, ExperimentScope, HeadDescriptor, LeaseId, LoginRequest, MetricValue,
    MicroShardId, NodeCertificate, NodeCertificateClaims, PeerId, PeerRole, PeerRoleSet,
    PrincipalClaims, PrincipalId, PrincipalSession, ProjectFamilyId, RevocationEpoch, ShardCache,
    WindowCtx, WindowId, WorkloadInputSource, WorkloadTrainingLease,
};
use burn_p2p_browser::{
    BrowserConformanceHarness, BrowserDirectorySnapshot, BrowserEdgeClient, BrowserEdgeMode,
    BrowserEdgePaths, BrowserEdgeSnapshot, BrowserEnrollmentConfig, BrowserLeaderboardSnapshot,
    BrowserLoginProvider, BrowserReceiptSubmissionResponse, BrowserRuntimeConfig,
    BrowserRuntimeRole, BrowserSessionState, BrowserTrainingBudget, BrowserTrainingPlan,
    BrowserTransportSurface, BrowserUiBindings, BrowserValidationPlan, BrowserWorkerCommand,
    BrowserWorkerEvent, BrowserWorkerIdentity, TrustBundleExport,
    browser_conformance_capability_for_role, browser_conformance_directory,
    browser_conformance_session, browser_conformance_transport,
};
use burn_p2p_core::{SignatureAlgorithm, SignatureMetadata};
use chrono::Utc;
use semver::Version;
use tempfile::tempdir;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

#[derive(Clone, Copy)]
struct SmokeModelSpec {
    n_layer: usize,
    n_embd: usize,
    n_head: usize,
    latent_total: usize,
    block_size: usize,
    batch_size: usize,
    max_iters: usize,
}

const SMALL_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 2,
    n_embd: 32,
    n_head: 4,
    latent_total: 64,
    block_size: 64,
    batch_size: 2,
    max_iters: 8,
};

const MEDIUM_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 4,
    n_embd: 64,
    n_head: 4,
    latent_total: 128,
    block_size: 128,
    batch_size: 4,
    max_iters: 24,
};

const LARGE_SPEC: SmokeModelSpec = SmokeModelSpec {
    n_layer: 6,
    n_embd: 96,
    n_head: 8,
    latent_total: 192,
    block_size: 128,
    batch_size: 4,
    max_iters: 32,
};

fn dummy_auth_bundle() -> DragonNativeAuthBundle {
    DragonNativeAuthBundle {
        auth_config: AuthConfig::new(),
        trust_bundle_endpoint: "https://edge.example/trust-bundle".into(),
        edge_base_url: None,
        session_id: None,
        principal_id: None,
    }
}

fn native_manifest_seed() -> DragonManifestSeed {
    DragonManifestSeed {
        project_family_id: "burn-dragon-language".into(),
        network_id: "dragon-p2p-testnet".into(),
        study_id: "dragon-p2p-study".into(),
        experiment_id: "dragon-p2p-exp".into(),
        revision_id: "r1".into(),
        display_name: "dragon p2p smoke".into(),
        description: "dragon p2p smoke network".into(),
        protocol_major: 0,
        authority_public_keys: Vec::new(),
        bootstrap_addrs: Vec::new(),
    }
}

fn write(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write config");
}

fn nca_corpus_config_toml(output_dir: &Path) -> String {
    format!(
        r#"
output_dir = "{}"
seed = 1337
name = "dragon-p2p-nca-smoke"
train_samples = 12
validation_samples = 6
chunk_token_capacity = 1024
"#,
        output_dir.display()
    )
}

fn nca_training_config_toml(
    cache_dir: &Path,
    nca_config_path: &Path,
    spec: SmokeModelSpec,
) -> String {
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "universality_nca"
config = "{}"

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = {}
n_embd = {}
n_head = {}
latent_total = {}

[model.language_head]
type = "nca_factorized_patch"
state_count = 10
patch_size = 2
frame_special_tokens = true
eos_id = 50256

[training]
block_size = {}
batch_size = {}
max_iters = {}
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.001
weight_decay = 0.0

[generation]
prompt = "0 0 0"
"#,
        cache_dir.display(),
        nca_config_path.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        spec.max_iters,
    )
}

fn climbmix_training_config_toml(cache_dir: &Path, spec: SmokeModelSpec) -> String {
    format!(
        r#"
[dataset]
cache_dir = "{}"
train_split_ratio = 0.9
type = "nemotron_climb_mix"
max_records = 64

[dataset.tokenizer]
type = "pretokenized"
vocab_size = 50257
eos_id = 50256

[model]
n_layer = {}
n_embd = {}
n_head = {}
latent_total = {}

[training]
block_size = {}
batch_size = {}
max_iters = {}
checkpoint_interval_iters = 4
log_frequency = 1
seed = 1337

[optimizer]
learning_rate = 0.003
weight_decay = 0.0

[generation]
prompt = "1 2 3"
"#,
        cache_dir.display(),
        spec.n_layer,
        spec.n_embd,
        spec.n_head,
        spec.latent_total,
        spec.block_size,
        spec.batch_size,
        spec.max_iters,
    )
}

fn simple_token_window_records(count: usize, block_size: usize) -> Vec<TokenWindowRecord> {
    (0..count)
        .map(|offset| {
            let base = ((offset * 7) % 1024) as i64;
            let inputs = (0..block_size)
                .map(|index| (base + index as i64) % 50256)
                .collect();
            let targets = (1..=block_size)
                .map(|index| (base + index as i64) % 50256)
                .collect();
            TokenWindowRecord {
                inputs,
                targets,
                reset_stream_state: offset % 4 == 0,
            }
        })
        .collect()
}

fn write_existing_climbmix_shards(root: &Path, count: usize, block_size: usize) {
    let records = simple_token_window_records(count, block_size);
    BurnShardedDataset::write_local(
        root,
        &records,
        BurnShardedDatasetConfig::new("dragon-climbmix-smoke")
            .with_microshards(4)
            .with_view_metadata_entry("experiment_kind", "climbmix-pretraining"),
    )
    .expect("write shard dataset");
}

fn metric_float(stats: &std::collections::BTreeMap<String, MetricValue>, key: &str) -> f64 {
    match stats.get(key).expect("metric") {
        MetricValue::Float(value) => *value,
        other => panic!("expected float metric for {key}, got {other:?}"),
    }
}

fn metric_integer(stats: &std::collections::BTreeMap<String, MetricValue>, key: &str) -> i64 {
    match stats.get(key).expect("metric") {
        MetricValue::Integer(value) => *value,
        other => panic!("expected integer metric for {key}, got {other:?}"),
    }
}

fn log_loss_series(label: &str, losses: &[f64]) {
    let first = losses.first().copied().unwrap_or(f64::NAN);
    let final_loss = losses.last().copied().unwrap_or(f64::NAN);
    let best = losses.iter().copied().fold(f64::INFINITY, f64::min);
    eprintln!("{label}: losses={losses:?} first={first:.4} best={best:.4} final={final_loss:.4}");
}

fn shard_manifest_url(base_url: &str) -> String {
    format!("{}/fetch-manifest.json", base_url.trim_end_matches('/'))
}

#[derive(Clone)]
struct NativeWindowObservation {
    head: HeadDescriptor,
    loss: f64,
}

fn browser_harness_pair(
    entry: &burn_p2p::ExperimentDirectoryEntry,
    release_train_hash: burn_p2p::ContentId,
    target_artifact_hash: burn_p2p::ContentId,
    network_id: burn_p2p::NetworkId,
) -> (BrowserConformanceHarness, BrowserConformanceHarness) {
    let session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-principal"),
        BTreeSet::from([
            ExperimentScope::Connect,
            ExperimentScope::Train {
                experiment_id: entry.experiment_id.clone(),
            },
            ExperimentScope::Validate {
                experiment_id: entry.experiment_id.clone(),
            },
        ]),
    );
    let trainer = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserTrainerWgpu,
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                release_train_hash.clone(),
                "browser-wasm",
                target_artifact_hash.clone(),
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
        browser_conformance_transport(),
        browser_conformance_directory(network_id.clone(), vec![entry.clone()]),
        session.clone(),
    );
    let verifier = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserVerifier,
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                release_train_hash,
                "browser-wasm",
                target_artifact_hash,
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
        browser_conformance_transport(),
        browser_conformance_directory(network_id, vec![entry.clone()]),
        session,
    );
    (trainer, verifier)
}

fn flush_and_ack_receipts(harness: &mut BrowserConformanceHarness) -> usize {
    let flush_events =
        harness
            .runtime
            .apply_command(BrowserWorkerCommand::FlushReceiptOutbox, None, None);
    let receipt_ids = flush_events
        .iter()
        .find_map(|event| match event {
            BrowserWorkerEvent::ReceiptOutboxReady { receipts, .. } => Some(
                receipts
                    .iter()
                    .map(|receipt| receipt.receipt_id.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .unwrap_or_default();
    if receipt_ids.is_empty() {
        assert!(
            harness.pending_receipts().is_empty(),
            "receipt flush emitted no receipts but the outbox is still non-empty"
        );
        return 0;
    }
    let ack_events = harness.runtime.apply_command(
        BrowserWorkerCommand::AcknowledgeSubmittedReceipts {
            receipt_ids: receipt_ids.clone(),
        },
        None,
        None,
    );
    assert!(ack_events.iter().any(|event| matches!(
        event,
        BrowserWorkerEvent::ReceiptsAcknowledged {
            receipt_ids: acknowledged,
            pending_receipts: 0,
        } if *acknowledged == receipt_ids
    )));
    assert!(
        harness.pending_receipts().is_empty(),
        "browser receipt outbox should be empty after acknowledgement"
    );
    receipt_ids.len()
}

fn run_training_windows_with_heads<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    windows: usize,
    head_prefix: &str,
) -> Vec<NativeWindowObservation>
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let project = &prepared.project;
    let device = project.runtime_device();
    let registration = project.dataset_registration().expect("registration");
    let microshard_plan = project
        .microshard_plan(&registration)
        .expect("microshard plan");
    let cache_root = tempdir().expect("cache root");
    let shard_cache = ShardCache::new(cache_root.path());
    let entry = &prepared.manifests.experiment_directory[0];

    let mut model = project.init_model(&device);
    let mut observations = Vec::new();
    let mut global_step = 0u64;
    let mut parent_head_id = None;

    for window_ordinal in 0..windows {
        let lease = burn_p2p::LeasePlanner::default()
            .plan_lease(
                prepared.manifests.network_manifest.network_id.clone(),
                entry.study_id.clone(),
                entry.experiment_id.clone(),
                entry.current_revision_id.clone(),
                &microshard_plan.dataset_view,
                PeerId::new(format!("{head_prefix}-peer-{}", window_ordinal + 1)),
                WindowId((window_ordinal + 1) as u64),
                Utc::now(),
                1,
                &microshard_plan.microshards,
            )
            .expect("lease")
            .lease;
        let cached = shard_cache
            .fetch_lease_microshards(&registration, &microshard_plan, &lease)
            .expect("cached microshards");
        let batches = project.load_batches(&lease, &cached).expect("load batches");
        let mut ctx = WindowCtx {
            device: device.clone(),
            model,
            lease,
            cached_microshards: cached,
            batches,
        };
        let report = project.train_window(&mut ctx).expect("train window");
        let train_steps = metric_integer(&report.stats, "train_steps");
        assert!(train_steps > 0);
        let loss = metric_float(&report.stats, "train_loss");
        assert!(loss.is_finite(), "train loss must be finite");
        global_step += train_steps as u64;
        let head = HeadDescriptor {
            head_id: burn_p2p::HeadId::new(format!("{head_prefix}-head-{}", window_ordinal + 1)),
            study_id: entry.study_id.clone(),
            experiment_id: entry.experiment_id.clone(),
            revision_id: entry.current_revision_id.clone(),
            artifact_id: burn_p2p::ArtifactId::new(format!(
                "{head_prefix}-artifact-{}",
                window_ordinal + 1
            )),
            parent_head_id: parent_head_id.clone(),
            global_step,
            created_at: Utc::now(),
            metrics: report.stats.clone(),
        };
        parent_head_id = Some(head.head_id.clone());
        observations.push(NativeWindowObservation { head, loss });
        model = ctx.model;
    }

    observations
}

fn run_training_windows<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    windows: usize,
) -> Vec<f64>
where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let project = &prepared.project;
    let device = project.runtime_device();
    let registration = project.dataset_registration().expect("registration");
    let microshard_plan = project
        .microshard_plan(&registration)
        .expect("microshard plan");
    let cache_root = tempdir().expect("cache root");
    let shard_cache = ShardCache::new(cache_root.path());
    let entry = &prepared.manifests.experiment_directory[0];

    let mut model = project.init_model(&device);
    let mut losses = Vec::new();
    for window_ordinal in 0..windows {
        let lease = burn_p2p::LeasePlanner::default()
            .plan_lease(
                prepared.manifests.network_manifest.network_id.clone(),
                entry.study_id.clone(),
                entry.experiment_id.clone(),
                entry.current_revision_id.clone(),
                &microshard_plan.dataset_view,
                PeerId::new(format!("peer-{}", window_ordinal + 1)),
                WindowId((window_ordinal + 1) as u64),
                Utc::now(),
                1,
                &microshard_plan.microshards,
            )
            .expect("lease")
            .lease;
        let cached = shard_cache
            .fetch_lease_microshards(&registration, &microshard_plan, &lease)
            .expect("cached microshards");
        let batches = project.load_batches(&lease, &cached).expect("load batches");
        let mut ctx = WindowCtx {
            device: device.clone(),
            model,
            lease,
            cached_microshards: cached,
            batches,
        };
        let report = project.train_window(&mut ctx).expect("train window");
        assert!(metric_integer(&report.stats, "train_steps") > 0);
        let loss = metric_float(&report.stats, "train_loss");
        assert!(loss.is_finite(), "train loss must be finite");
        losses.push(loss);
        model = ctx.model;
    }
    losses
}

#[derive(Default)]
struct MockEdgeState {
    authorized_directory_fetches: usize,
    unauthorized_directory_fetches: usize,
    receipt_submission_batches: usize,
    submitted_receipt_ids: Vec<String>,
    enrolled_peer_ids: BTreeSet<String>,
    sessions: BTreeMap<String, PrincipalSession>,
    pending_logins: BTreeMap<String, PendingLogin>,
}

#[derive(Clone)]
struct PendingLogin {
    requested_scopes: BTreeSet<ExperimentScope>,
    state: String,
}

struct LocalEdgeMock {
    base_url: String,
    state: Arc<Mutex<MockEdgeState>>,
    stop: Arc<AtomicBool>,
    join: Option<thread::JoinHandle<()>>,
}

impl Drop for LocalEdgeMock {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(join) = self.join.take() {
            join.join().expect("edge server thread");
        }
    }
}

fn edge_scopes(entry: &ExperimentDirectoryEntry) -> BTreeSet<ExperimentScope> {
    BTreeSet::from([
        ExperimentScope::Connect,
        ExperimentScope::Train {
            experiment_id: entry.experiment_id.clone(),
        },
        ExperimentScope::Validate {
            experiment_id: entry.experiment_id.clone(),
        },
    ])
}

fn edge_snapshot_for_manifests(
    manifests: &burn_dragon_p2p::config::DragonManifestBundle,
    browser_mode: BrowserMode,
) -> BrowserEdgeSnapshot {
    let mut paths = BrowserEdgePaths::default();
    paths.login_path = "/login/github".into();
    paths.callback_path = "/callback/github".into();

    BrowserEdgeSnapshot {
        network_id: manifests.network_manifest.network_id.clone(),
        edge_mode: BrowserEdgeMode::Peer,
        browser_mode,
        social_mode: burn_p2p::SocialMode::Disabled,
        profile_mode: burn_p2p::ProfileMode::Disabled,
        transports: BrowserTransportSurface {
            webrtc_direct: false,
            webtransport_gateway: true,
            wss_fallback: true,
        },
        paths,
        auth_enabled: true,
        login_providers: vec![BrowserLoginProvider {
            label: "GitHub".into(),
            login_path: "/login/github".into(),
            callback_path: Some("/callback/github".into()),
            device_path: None,
        }],
        required_release_train_hash: Some(manifests.release_manifest.release_train_hash.clone()),
        allowed_target_artifact_hashes: BTreeSet::from([manifests
            .release_manifest
            .target_artifact_hash
            .clone()]),
        directory: BrowserDirectorySnapshot {
            network_id: manifests.network_manifest.network_id.clone(),
            generated_at: Utc::now(),
            entries: manifests.experiment_directory.clone(),
        },
        heads: Vec::new(),
        leaderboard: BrowserLeaderboardSnapshot {
            network_id: manifests.network_manifest.network_id.clone(),
            score_version: "leaderboard_score_v1".into(),
            entries: Vec::new(),
            captured_at: Utc::now(),
        },
        trust_bundle: Some(TrustBundleExport {
            network_id: manifests.network_manifest.network_id.clone(),
            project_family_id: ProjectFamilyId::new(
                manifests.release_manifest.project_family_id.as_str(),
            ),
            required_release_train_hash: manifests.release_manifest.release_train_hash.clone(),
            allowed_target_artifact_hashes: BTreeSet::from([manifests
                .release_manifest
                .target_artifact_hash
                .clone()]),
            minimum_revocation_epoch: RevocationEpoch(0),
            active_issuer_peer_id: PeerId::new("dragon-edge-issuer"),
            issuers: Vec::new(),
            reenrollment: None,
        }),
        captured_at: Utc::now(),
    }
}

fn current_edge_head(entry: &ExperimentDirectoryEntry, label: &str) -> HeadDescriptor {
    HeadDescriptor {
        head_id: burn_p2p::HeadId::new(format!("{label}-edge-head")),
        study_id: entry.study_id.clone(),
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new(format!("{label}-edge-artifact")),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }
}

fn browser_worker_identity(label: &str) -> BrowserWorkerIdentity {
    BrowserWorkerIdentity {
        peer_id: PeerId::new(format!("{label}-browser-peer")),
        peer_public_key_hex: "deadbeef".into(),
        serial: 1,
        client_policy_hash: None,
    }
}

fn json_header() -> Header {
    Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).expect("json header")
}

fn respond_json<T: serde::Serialize>(request: Request, status: u16, value: &T) {
    let payload = serde_json::to_string(value).expect("serialize json response");
    request
        .respond(
            Response::from_string(payload)
                .with_status_code(StatusCode(status))
                .with_header(json_header()),
        )
        .expect("respond json");
}

fn respond_text(request: Request, status: u16, body: &str) {
    request
        .respond(
            Response::from_string(body.to_owned())
                .with_status_code(StatusCode(status))
                .with_header(json_header()),
        )
        .expect("respond text");
}

fn read_json<T: serde::de::DeserializeOwned>(request: &mut Request) -> T {
    let mut body = String::new();
    std::io::Read::read_to_string(request.as_reader(), &mut body).expect("request body");
    serde_json::from_str(&body).expect("decode request json")
}

fn header_value(request: &Request, name: &'static str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|header| header.field.equiv(name))
        .map(|header| header.value.as_str().to_owned())
}

fn principal_from_provider_code(provider_code: Option<String>) -> PrincipalId {
    let suffix = provider_code
        .as_deref()
        .filter(|value| !value.is_empty())
        .unwrap_or("github-user");
    PrincipalId::new(format!("github-{suffix}"))
}

fn node_certificate_for_session(
    snapshot: &BrowserEdgeSnapshot,
    session: &PrincipalSession,
    enrollment: &EdgePeerEnrollmentRequest,
) -> NodeCertificate {
    NodeCertificate::new(
        Version::new(0, 1, 0),
        NodeCertificateClaims {
            network_id: snapshot.network_id.clone(),
            project_family_id: snapshot
                .trust_bundle
                .as_ref()
                .expect("trust bundle")
                .project_family_id
                .clone(),
            release_train_hash: enrollment.release_train_hash.clone(),
            target_artifact_hash: enrollment.target_artifact_hash.clone(),
            peer_id: enrollment.peer_id.clone(),
            peer_public_key_hex: enrollment.peer_public_key_hex.clone(),
            principal_id: session.claims.principal_id.clone(),
            provider: session.claims.provider.clone(),
            granted_roles: PeerRoleSet::new([
                PeerRole::TrainerCpu,
                PeerRole::BrowserTrainerWgpu,
                PeerRole::BrowserVerifier,
                PeerRole::Viewer,
            ]),
            experiment_scopes: enrollment.requested_scopes.clone(),
            client_policy_hash: enrollment.client_policy_hash.clone(),
            auth_policy_snapshot: None,
            not_before: Utc::now(),
            not_after: Utc::now() + chrono::Duration::minutes(30),
            serial: enrollment.serial,
            revocation_epoch: RevocationEpoch(0),
        },
        SignatureMetadata {
            signer: PeerId::new("dragon-edge-issuer"),
            key_id: "dragon-edge-key".into(),
            algorithm: SignatureAlgorithm::Ed25519,
            signed_at: Utc::now(),
            signature_hex: "00".into(),
        },
    )
    .expect("node certificate")
}

fn spawn_local_edge(snapshot: BrowserEdgeSnapshot) -> LocalEdgeMock {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind edge");
    let addr = listener.local_addr().expect("edge local addr");
    let server = Server::from_listener(listener, None).expect("tiny_http server");
    let state = Arc::new(Mutex::new(MockEdgeState::default()));
    let stop = Arc::new(AtomicBool::new(false));
    let state_for_thread = Arc::clone(&state);
    let stop_for_thread = Arc::clone(&stop);
    let snapshot_for_thread = snapshot.clone();

    let join = thread::spawn(move || {
        while !stop_for_thread.load(Ordering::SeqCst) {
            let Some(mut request) = server
                .recv_timeout(Duration::from_millis(100))
                .expect("receive request")
            else {
                continue;
            };

            match (request.method(), request.url()) {
                (&Method::Get, "/portal/snapshot") => {
                    respond_json(request, 200, &snapshot_for_thread);
                }
                (&Method::Get, "/trust") => {
                    respond_json(
                        request,
                        200,
                        snapshot_for_thread
                            .trust_bundle
                            .as_ref()
                            .expect("trust bundle"),
                    );
                }
                (&Method::Get, "/directory") => {
                    let Some(session_id) = header_value(&request, "x-session-id") else {
                        state_for_thread
                            .lock()
                            .expect("state")
                            .unauthorized_directory_fetches += 1;
                        respond_text(request, 401, r#"{"error":"missing session"}"#);
                        continue;
                    };
                    let authorized = state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .contains_key(session_id.as_str());
                    if !authorized {
                        state_for_thread
                            .lock()
                            .expect("state")
                            .unauthorized_directory_fetches += 1;
                        respond_text(request, 401, r#"{"error":"unknown session"}"#);
                        continue;
                    }
                    state_for_thread
                        .lock()
                        .expect("state")
                        .authorized_directory_fetches += 1;
                    respond_json(request, 200, &snapshot_for_thread.directory.entries);
                }
                (&Method::Post, "/login/github") => {
                    let login: LoginRequest = read_json(&mut request);
                    let ordinal = state_for_thread.lock().expect("state").pending_logins.len() + 1;
                    let login_id = ContentId::new(format!("mock-login-{ordinal}"));
                    let state_token = format!("mock-state-{ordinal}");
                    state_for_thread
                        .lock()
                        .expect("state")
                        .pending_logins
                        .insert(
                            login_id.as_str().to_owned(),
                            PendingLogin {
                                requested_scopes: login.requested_scopes,
                                state: state_token.clone(),
                            },
                        );
                    respond_json(
                        request,
                        200,
                        &burn_p2p::LoginStart {
                            login_id,
                            provider: AuthProvider::GitHub,
                            state: state_token,
                            authorize_url: Some("https://github.example/authorize".into()),
                            expires_at: Utc::now() + chrono::Duration::minutes(5),
                        },
                    );
                }
                (&Method::Post, "/callback/github") => {
                    let callback: CallbackPayload = read_json(&mut request);
                    let pending = state_for_thread
                        .lock()
                        .expect("state")
                        .pending_logins
                        .get(callback.login_id.as_str())
                        .cloned()
                        .expect("pending login");
                    assert_eq!(
                        callback.state, pending.state,
                        "callback state must match login"
                    );
                    let principal_id = principal_from_provider_code(callback.provider_code.clone());
                    let session = PrincipalSession {
                        session_id: ContentId::new(format!(
                            "mock-session-{}",
                            callback.login_id.as_str()
                        )),
                        network_id: snapshot_for_thread.network_id.clone(),
                        claims: PrincipalClaims {
                            principal_id,
                            provider: AuthProvider::GitHub,
                            display_name: "dragon github principal".into(),
                            org_memberships: BTreeSet::from(["dragon".into()]),
                            group_memberships: BTreeSet::from(["trainers".into()]),
                            granted_roles: PeerRoleSet::new([
                                PeerRole::TrainerCpu,
                                PeerRole::BrowserTrainerWgpu,
                                PeerRole::BrowserVerifier,
                            ]),
                            granted_scopes: pending.requested_scopes,
                            custom_claims: BTreeMap::new(),
                            issued_at: Utc::now(),
                            expires_at: Utc::now() + chrono::Duration::minutes(30),
                        },
                        issued_at: Utc::now(),
                        expires_at: Utc::now() + chrono::Duration::minutes(30),
                    };
                    state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .insert(session.session_id.as_str().to_owned(), session.clone());
                    respond_json(request, 200, &session);
                }
                (&Method::Post, "/enroll") => {
                    let enrollment: EdgePeerEnrollmentRequest = read_json(&mut request);
                    let session = state_for_thread
                        .lock()
                        .expect("state")
                        .sessions
                        .get(enrollment.session_id.as_str())
                        .cloned()
                        .expect("session for enrollment");
                    let certificate =
                        node_certificate_for_session(&snapshot_for_thread, &session, &enrollment);
                    state_for_thread
                        .lock()
                        .expect("state")
                        .enrolled_peer_ids
                        .insert(enrollment.peer_id.as_str().to_owned());
                    respond_json(request, 200, &certificate);
                }
                (&Method::Post, "/receipts/browser") => {
                    let Some(session_id) = header_value(&request, "x-session-id") else {
                        respond_text(request, 401, r#"{"error":"missing session"}"#);
                        continue;
                    };
                    let mut state = state_for_thread.lock().expect("state");
                    if !state.sessions.contains_key(session_id.as_str()) {
                        respond_text(request, 401, r#"{"error":"unknown session"}"#);
                        continue;
                    }
                    let receipts: Vec<burn_p2p::ContributionReceipt> = read_json(&mut request);
                    state.receipt_submission_batches += 1;
                    state.submitted_receipt_ids.extend(
                        receipts
                            .iter()
                            .map(|receipt| receipt.receipt_id.as_str().to_owned()),
                    );
                    let response = BrowserReceiptSubmissionResponse {
                        accepted_receipt_ids: receipts
                            .iter()
                            .map(|receipt| receipt.receipt_id.clone())
                            .collect(),
                        pending_receipt_count: 0,
                    };
                    drop(state);
                    respond_json(request, 200, &response);
                }
                _ => {
                    respond_text(request, 404, r#"{"error":"not found"}"#);
                }
            }
        }
    });

    LocalEdgeMock {
        base_url: format!("http://{addr}"),
        state,
        stop,
        join: Some(join),
    }
}

fn acknowledge_browser_receipts(
    harness: &mut BrowserConformanceHarness,
    receipt_ids: Vec<burn_p2p::ContributionReceiptId>,
) {
    let ack_events = harness.runtime.apply_command(
        BrowserWorkerCommand::AcknowledgeSubmittedReceipts {
            receipt_ids: receipt_ids.clone(),
        },
        None,
        None,
    );
    assert!(ack_events.iter().any(|event| matches!(
        event,
        BrowserWorkerEvent::ReceiptsAcknowledged {
            receipt_ids: acknowledged,
            pending_receipts: 0,
        } if *acknowledged == receipt_ids
    )));
    assert!(
        harness.pending_receipts().is_empty(),
        "browser receipt outbox should be empty after edge acknowledgement"
    );
}

fn browser_runtime_for_edge(
    edge_base_url: &str,
    network_id: burn_p2p::NetworkId,
    release_train_hash: ContentId,
    target_artifact_hash: ContentId,
    role: BrowserRuntimeRole,
) -> BrowserRuntimeConfig {
    BrowserRuntimeConfig {
        role,
        ..BrowserRuntimeConfig::new(
            edge_base_url,
            network_id,
            release_train_hash,
            "browser-wasm",
            target_artifact_hash,
        )
    }
}

fn run_edge_drill_for_prepared<B>(
    prepared: &burn_dragon_p2p::experiments::common::PreparedNativePeer<B>,
    label: &str,
) where
    B: burn::tensor::backend::AutodiffBackend + Clone + 'static,
    B::Device: Clone,
{
    let entry = prepared.manifests.experiment_directory[0].clone();
    let snapshot = edge_snapshot_for_manifests(&prepared.manifests, BrowserMode::Trainer);
    let edge = spawn_local_edge(snapshot.clone());
    let requested_scopes = edge_scopes(&entry);
    let native_storage = tempdir().expect("native auth storage");

    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("tokio runtime");

    runtime.block_on(async {
        let fetched_snapshot = fetch_edge_snapshot(&edge.base_url)
            .await
            .expect("native fetch edge snapshot");
        assert_eq!(fetched_snapshot.network_id, snapshot.network_id);

        let pending = begin_native_github_login(
            &edge.base_url,
            &prepared.manifests.release_manifest,
            requested_scopes.clone(),
            1800,
            Some(format!("{label}-native")),
            false,
        )
        .await
        .expect("begin native github login");
        assert!(matches!(pending.login.provider, AuthProvider::GitHub));

        let native = complete_native_github_login(
            native_storage.path(),
            &pending,
            "native-provider-code",
            None,
        )
        .await
        .expect("complete native github login");
        assert!(matches!(
            native.session.claims.provider,
            AuthProvider::GitHub
        ));
        assert!(native.auth.auth_config.local_peer_auth.is_some());
        assert_eq!(
            native.auth.trust_bundle_endpoint,
            format!("{}/trust", edge.base_url)
        );

        let browser_boot_client = BrowserEdgeClient::new(
            BrowserUiBindings::new(&edge.base_url),
            BrowserEnrollmentConfig::for_runtime_sync(&snapshot),
        );
        let browser_snapshot = browser_boot_client
            .fetch_browser_edge_snapshot()
            .await
            .expect("browser fetch edge snapshot");
        assert_eq!(browser_snapshot.network_id, snapshot.network_id);

        let browser_client = BrowserEdgeClient::new(
            BrowserUiBindings::from_edge_snapshot(&edge.base_url, &browser_snapshot),
            BrowserEnrollmentConfig::from_edge_snapshot(
                &browser_snapshot,
                "browser-wasm",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                requested_scopes,
                1800,
            )
            .expect("browser enrollment config"),
        );
        let browser_login = browser_client
            .begin_login(Some(format!("{label}-browser")))
            .await
            .expect("begin browser github login");
        assert!(matches!(browser_login.provider, AuthProvider::GitHub));

        let browser_session = browser_client
            .complete_provider_login(&browser_login, "browser-provider-code")
            .await
            .expect("complete browser github login");
        assert!(matches!(
            browser_session.claims.provider,
            AuthProvider::GitHub
        ));

        let worker_identity = browser_worker_identity(label);
        let browser_certificate = browser_client
            .enroll(&browser_client.build_enrollment_request(&browser_session, &worker_identity))
            .await
            .expect("browser enroll");
        let trust_bundle = browser_client
            .fetch_trust_bundle()
            .await
            .expect("browser trust bundle");

        assert!(
            browser_client.fetch_directory(None).await.is_err(),
            "directory fetch without session should be rejected"
        );
        let directory = browser_client
            .fetch_directory(Some(&browser_session.session_id))
            .await
            .expect("authorized directory fetch");
        assert_eq!(directory[0].experiment_id, entry.experiment_id);

        let head = current_edge_head(&entry, label);
        let browser_session_state = BrowserSessionState {
            session: Some(browser_session.clone()),
            certificate: Some(browser_certificate),
            trust_bundle: Some(trust_bundle),
            enrolled_at: Some(Utc::now()),
            reenrollment_required: false,
        };

        let mut trainer = BrowserConformanceHarness::start(
            browser_runtime_for_edge(
                &edge.base_url,
                prepared.manifests.network_manifest.network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                BrowserRuntimeRole::BrowserTrainerWgpu,
            ),
            browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
            browser_conformance_transport(),
            browser_conformance_directory(
                prepared.manifests.network_manifest.network_id.clone(),
                vec![entry.clone()],
            ),
            browser_session_state.clone(),
        );
        trainer.select_experiment(
            entry.experiment_id.clone(),
            Some(entry.current_revision_id.clone()),
        );
        trainer.apply_heads(std::slice::from_ref(&head));
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training against edge-backed session");
        assert!(training.receipt_id.is_some());
        let pending_training_receipts = trainer.pending_receipts();
        assert!(
            !pending_training_receipts.is_empty(),
            "browser training should enqueue at least one receipt"
        );
        let training_submission = browser_client
            .submit_receipts(&browser_session.session_id, &pending_training_receipts)
            .await
            .expect("submit browser training receipts");
        assert_eq!(
            training_submission.accepted_receipt_ids.len(),
            pending_training_receipts.len()
        );
        acknowledge_browser_receipts(&mut trainer, training_submission.accepted_receipt_ids);

        let mut verifier = BrowserConformanceHarness::start(
            browser_runtime_for_edge(
                &edge.base_url,
                prepared.manifests.network_manifest.network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
                BrowserRuntimeRole::BrowserVerifier,
            ),
            browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
            browser_conformance_transport(),
            browser_conformance_directory(
                prepared.manifests.network_manifest.network_id.clone(),
                vec![entry.clone()],
            ),
            browser_session_state,
        );
        verifier.select_experiment(
            entry.experiment_id.clone(),
            Some(entry.current_revision_id.clone()),
        );
        verifier.apply_heads(std::slice::from_ref(&head));
        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation against edge-backed session");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        let pending_validation_receipts = verifier.pending_receipts();
        assert!(
            !pending_validation_receipts.is_empty(),
            "browser validation should enqueue at least one receipt"
        );
        let validation_submission = browser_client
            .submit_receipts(&browser_session.session_id, &pending_validation_receipts)
            .await
            .expect("submit browser validation receipts");
        assert_eq!(
            validation_submission.accepted_receipt_ids.len(),
            pending_validation_receipts.len()
        );
        acknowledge_browser_receipts(&mut verifier, validation_submission.accepted_receipt_ids);
    });

    let state = edge.state.lock().expect("edge state");
    assert_eq!(
        state.enrolled_peer_ids.len(),
        2,
        "native and browser peers should both enroll against the same edge"
    );
    assert_eq!(
        state.authorized_directory_fetches, 1,
        "browser directory fetch should succeed once with a session"
    );
    assert_eq!(
        state.unauthorized_directory_fetches, 1,
        "browser directory fetch without a session should be rejected"
    );
    assert!(
        state.receipt_submission_batches >= 2,
        "browser training and validation receipts should both submit to the edge"
    );
    assert!(
        state.submitted_receipt_ids.len() >= 2,
        "edge should record submitted browser receipts"
    );
}

#[test]
fn nca_native_peer_exports_shards_and_executes_training_windows() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root.clone(),
            dataset_name: Some("dragon-nca-smoke".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    assert_eq!(
        prepared.project.data_pipeline_kind(),
        burn_p2p::LeaseDataPipelineKind::ShardedStatic
    );
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::Generated { descriptor }) => {
            assert_eq!(descriptor.provider, "burn_dragon_universality_nca");
            assert_eq!(
                descriptor
                    .metadata
                    .get("experiment_kind")
                    .map(String::as_str),
                Some("nca-prepretraining")
            );
            assert_eq!(
                descriptor.metadata.get("config_path").map(String::as_str),
                Some(nca_config_path.to_string_lossy().as_ref())
            );
        }
        other => panic!("expected generated input source, got {other:?}"),
    }
    assert!(shard_root.join("fetch-manifest.json").is_file());
    assert!(shard_root.join("burn-sharded-dataset.json").is_file());

    let losses = run_training_windows(&prepared, 3);
    log_loss_series("nca_native_smoke", &losses);
    assert!(losses.last().copied().unwrap_or(f64::INFINITY) <= losses[0] + 0.5);
}

#[test]
fn nca_native_auto_target_downgrades_to_validator_under_tight_budget() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("downgrade".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: DragonCapabilityPolicy {
            native_cpu_memory_budget_bytes: Some(1),
            ..Default::default()
        },
        shard_export: None,
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    assert_eq!(
        prepared.target_decision.requested_target,
        DragonNativeTarget::Auto
    );
    assert_eq!(
        prepared.target_decision.effective_target,
        DragonNativeTarget::Validator
    );
    assert!(!prepared.target_decision.can_train);
    assert!(
        prepared
            .target_decision
            .downgrade_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("downgrading to validator"))
    );
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .resource_requirements
            .minimum_device_memory_bytes,
        None
    );
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .resource_requirements
            .minimum_system_memory_bytes,
        Some(
            prepared
                .footprint
                .estimated_training_bytes
                .max(512 * 1024 * 1024)
        )
    );
    let expected_training_bytes = prepared.footprint.estimated_training_bytes.to_string();
    assert_eq!(
        prepared.manifests.experiment_directory[0]
            .metadata
            .get("estimated_training_bytes")
            .map(String::as_str),
        Some(expected_training_bytes.as_str())
    );
}

#[test]
fn nca_native_persisted_runtime_failure_downgrades_to_validator_on_reprepare() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-downgrade-persisted"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("downgrade-persisted".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("trainer");
    assert!(prepared.target_decision.can_train);
    assert_eq!(
        prepared.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );

    prepared
        .record_runtime_training_failure("out of memory allocating optimizer state")
        .expect("persist runtime downgrade");

    let downgraded =
        prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("reprepare");
    assert_eq!(
        downgraded.target_decision.effective_target,
        DragonNativeTarget::Validator
    );
    assert!(!downgraded.target_decision.can_train);
    assert!(
        downgraded
            .target_decision
            .downgrade_reason
            .as_deref()
            .is_some_and(|reason| reason.contains("persisted trainer failure"))
    );

    downgraded
        .clear_runtime_downgrade()
        .expect("clear persisted downgrade");

    let recovered = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("recovered");
    assert!(recovered.target_decision.can_train);
    assert_eq!(
        recovered.target_decision.effective_target,
        DragonNativeTarget::Trainer
    );
}

#[test]
fn climbmix_native_existing_shards_supports_multi_peer_windows() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 16, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");

    assert_eq!(
        peer_a.manifests.network_manifest.network_id,
        peer_b.manifests.network_manifest.network_id
    );
    assert_eq!(
        peer_a.manifests.supported_workload.workload_id,
        peer_b.manifests.supported_workload.workload_id
    );
    assert_eq!(
        peer_a.manifests.experiment_directory[0].dataset_view_id,
        peer_b.manifests.experiment_directory[0].dataset_view_id
    );

    let losses_a = run_training_windows(&peer_a, 3);
    let losses_b = run_training_windows(&peer_b, 3);
    log_loss_series("climbmix_native_smoke_peer_a", &losses_a);
    log_loss_series("climbmix_native_smoke_peer_b", &losses_b);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
}

#[test]
fn browser_conformance_uses_native_dragon_manifests() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-browser-compat"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("smoke".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-browser-net".into()),
            microshards: Some(2),
            max_records: Some(16),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::Generated { descriptor }) => {
            assert_eq!(descriptor.provider, "burn_dragon_universality_nca");
        }
        other => panic!("expected generated input source, got {other:?}"),
    }
    let entry = prepared.manifests.experiment_directory[0].clone();
    let network_id = prepared.manifests.network_manifest.network_id.clone();
    let session = browser_conformance_session(
        network_id.clone(),
        PrincipalId::new("browser-principal"),
        BTreeSet::from([
            ExperimentScope::Connect,
            ExperimentScope::Train {
                experiment_id: entry.experiment_id.clone(),
            },
            ExperimentScope::Validate {
                experiment_id: entry.experiment_id.clone(),
            },
        ]),
    );
    let mut harness = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserTrainerWgpu,
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                "browser-wasm",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserTrainerWgpu),
        browser_conformance_transport(),
        browser_conformance_directory(network_id.clone(), vec![entry.clone()]),
        session.clone(),
    );
    harness.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    harness.apply_heads(&[HeadDescriptor {
        head_id: burn_p2p::HeadId::new("dragon-head"),
        study_id: entry.study_id.clone(),
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new("dragon-artifact"),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }]);
    let training_lease = WorkloadTrainingLease {
        lease_id: LeaseId::new("dragon-browser-lease"),
        window_id: WindowId(1),
        dataset_view_id: entry.dataset_view_id.clone(),
        assignment_hash: ContentId::new("dragon-browser-assignment"),
        microshards: vec![MicroShardId::new("dragon-browser-shard-a")],
    };

    let training = harness
        .run_training(BrowserTrainingPlan {
            study_id: entry.study_id.clone(),
            experiment_id: entry.experiment_id.clone(),
            revision_id: entry.current_revision_id.clone(),
            workload_id: entry.workload_id.clone(),
            budget: BrowserTrainingBudget::default(),
            lease: Some(training_lease.clone()),
        })
        .expect("training");
    assert_eq!(harness.active_training_lease(), Some(&training_lease));
    let mut verifier = BrowserConformanceHarness::start(
        BrowserRuntimeConfig {
            role: BrowserRuntimeRole::BrowserVerifier,
            ..BrowserRuntimeConfig::new(
                "https://edge.example",
                network_id.clone(),
                prepared
                    .manifests
                    .release_manifest
                    .release_train_hash
                    .clone(),
                "browser-wasm",
                prepared
                    .manifests
                    .release_manifest
                    .target_artifact_hash
                    .clone(),
            )
        },
        browser_conformance_capability_for_role(BrowserRuntimeRole::BrowserVerifier),
        browser_conformance_transport(),
        browser_conformance_directory(network_id, vec![entry.clone()]),
        session,
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.apply_heads(&[HeadDescriptor {
        head_id: burn_p2p::HeadId::new("dragon-head"),
        study_id: entry.study_id,
        experiment_id: entry.experiment_id.clone(),
        revision_id: entry.current_revision_id.clone(),
        artifact_id: burn_p2p::ArtifactId::new("dragon-artifact"),
        parent_head_id: None,
        global_step: 1,
        created_at: Utc::now(),
        metrics: Default::default(),
    }]);

    let validation = verifier
        .run_validation(BrowserValidationPlan {
            head_id: burn_p2p::HeadId::new("dragon-head"),
            max_checkpoint_bytes: 8 * 1024 * 1024,
            sample_budget: 4,
            emit_receipt: true,
        })
        .expect("validation");

    eprintln!(
        "browser_conformance: window_secs={} training_receipt={:?} validation_receipt={:?}",
        training.window_secs, training.receipt_id, validation.emitted_receipt_id
    );
    assert_eq!(training.window_secs, 30);
    assert!(training.receipt_id.is_some());
    assert!(validation.accepted);
}

#[test]
fn climbmix_http_shards_publish_http_input_source_descriptor() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-http-shards");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 16, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let http_upstream = "https://datasets.example/climbmix";
    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-http-climbmix"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("http".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root,
            http_upstream: Some(http_upstream.into()),
        }),
    };

    let prepared = prepare_climbmix_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let profile = DragonExperimentProfile::from_entry_metadata(
        prepared
            .manifests
            .experiment_directory
            .first()
            .expect("directory entry"),
    )
    .expect("profile decode")
    .expect("profile");
    match prepared
        .project
        .data_pipeline_descriptor()
        .input_source
        .as_ref()
    {
        Some(WorkloadInputSource::ShardManifestHttp {
            manifest_url,
            shard_count,
        }) => {
            assert_eq!(manifest_url, &shard_manifest_url(http_upstream));
            assert_eq!(*shard_count, Some(4));
        }
        other => panic!("expected shard-manifest http input source, got {other:?}"),
    }
    match profile.browser.expect("browser profile").train_source {
        DragonBrowserProfileTokenSource::ShardManifestHttp {
            manifest_url,
            selection,
            max_shards_per_window,
        } => {
            assert_eq!(
                manifest_url,
                "/dragon-datasets/climbmix-pretraining/r1/fetch-manifest.json"
            );
            assert_eq!(
                selection,
                burn_dragon_p2p::config::DragonBrowserShardSelectionPolicy::DeterministicPeer
            );
            assert_eq!(max_shards_per_window, Some(4));
        }
        other => panic!("expected browser shard-manifest source, got {other:?}"),
    }
}

#[test]
fn nca_mixed_fleet_browser_and_native_same_net_progresses() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-mixed");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("mixed".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-mixed".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let entry = prepared.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = browser_harness_pair(
        &entry,
        prepared
            .manifests
            .release_manifest
            .release_train_hash
            .clone(),
        prepared
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        prepared.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let native_obs = run_training_windows_with_heads(&prepared, 3, "nca-mixed");
    let native_losses = native_obs.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &native_obs {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert_eq!(training.window_secs, 30);
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert_eq!(validation.checked_chunks, 4);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("nca_mixed_fleet_native", &native_losses);
    assert!(native_losses.iter().all(|loss| loss.is_finite()));
    assert!(native_losses.iter().copied().fold(f64::INFINITY, f64::min) <= native_losses[0] + 0.5);
    assert!(
        (1..=native_obs.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=native_obs.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
fn climbmix_mixed_fleet_browser_and_native_same_net_progresses() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-mixed");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 24, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("mixed".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let entry = peer_a.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = browser_harness_pair(
        &entry,
        peer_a.manifests.release_manifest.release_train_hash.clone(),
        peer_a
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        peer_a.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let obs_a = run_training_windows_with_heads(&peer_a, 2, "climbmix-peer-a");
    let obs_b = run_training_windows_with_heads(&peer_b, 2, "climbmix-peer-b");
    let ordered = [obs_a.as_slice(), obs_b.as_slice()]
        .into_iter()
        .flat_map(|slice| slice.iter())
        .cloned()
        .collect::<Vec<_>>();
    let losses_a = obs_a.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_b = obs_b.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &ordered {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert_eq!(training.window_secs, 30);
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 4,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("climbmix_mixed_fleet_peer_a", &losses_a);
    log_loss_series("climbmix_mixed_fleet_peer_b", &losses_b);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(
        (1..=ordered.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=ordered.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "mixed-fleet scale rung"]
fn nca_mixed_fleet_browser_and_native_same_net_medium() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-mixed-medium");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-cache"),
            &nca_config_path,
            MEDIUM_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-mixed-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("mixed-medium".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-mixed-medium".into()),
            microshards: Some(8),
            max_records: Some(96),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let entry = prepared.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = browser_harness_pair(
        &entry,
        prepared
            .manifests
            .release_manifest
            .release_train_hash
            .clone(),
        prepared
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        prepared.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let native_obs = run_training_windows_with_heads(&prepared, 5, "nca-mixed-medium");
    let native_losses = native_obs.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &native_obs {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));
        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 6,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("nca_mixed_fleet_medium_native", &native_losses);
    assert!(native_losses.iter().all(|loss| loss.is_finite()));
    assert!(
        native_losses.iter().copied().fold(f64::INFINITY, f64::min) <= native_losses[0] - 0.5,
        "mixed-fleet medium NCA should show a material best-window improvement"
    );
    assert!(
        (1..=native_obs.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=native_obs.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "mixed-fleet scale rung"]
fn climbmix_mixed_fleet_browser_and_native_three_peers_medium() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-mixed-medium");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 48, 16);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), MEDIUM_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("mixed-medium".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let entry = peer_a.manifests.experiment_directory[0].clone();
    let (mut trainer, mut verifier) = browser_harness_pair(
        &entry,
        peer_a.manifests.release_manifest.release_train_hash.clone(),
        peer_a
            .manifests
            .release_manifest
            .target_artifact_hash
            .clone(),
        peer_a.manifests.network_manifest.network_id.clone(),
    );
    trainer.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );
    verifier.select_experiment(
        entry.experiment_id.clone(),
        Some(entry.current_revision_id.clone()),
    );

    let obs_a = run_training_windows_with_heads(&peer_a, 2, "climbmix-medium-peer-a");
    let obs_b = run_training_windows_with_heads(&peer_b, 2, "climbmix-medium-peer-b");
    let obs_c = run_training_windows_with_heads(&peer_c, 2, "climbmix-medium-peer-c");
    let ordered = [obs_a.as_slice(), obs_b.as_slice(), obs_c.as_slice()]
        .into_iter()
        .flat_map(|slice| slice.iter())
        .cloned()
        .collect::<Vec<_>>();
    let losses_a = obs_a.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_b = obs_b.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let losses_c = obs_c.iter().map(|obs| obs.loss).collect::<Vec<_>>();
    let mut train_receipts = 0usize;
    let mut verify_receipts = 0usize;

    for obs in &ordered {
        trainer.apply_heads(std::slice::from_ref(&obs.head));
        verifier.apply_heads(std::slice::from_ref(&obs.head));

        let training = trainer
            .run_training(BrowserTrainingPlan {
                study_id: entry.study_id.clone(),
                experiment_id: entry.experiment_id.clone(),
                revision_id: entry.current_revision_id.clone(),
                workload_id: entry.workload_id.clone(),
                budget: BrowserTrainingBudget::default(),
                lease: None,
            })
            .expect("browser training");
        assert!(training.receipt_id.is_some());
        train_receipts += flush_and_ack_receipts(&mut trainer);

        let validation = verifier
            .run_validation(BrowserValidationPlan {
                head_id: obs.head.head_id.clone(),
                max_checkpoint_bytes: 8 * 1024 * 1024,
                sample_budget: 6,
                emit_receipt: true,
            })
            .expect("browser validation");
        assert!(validation.accepted);
        assert!(validation.emitted_receipt_id.is_some());
        verify_receipts += flush_and_ack_receipts(&mut verifier);
    }

    log_loss_series("climbmix_mixed_fleet_medium_peer_a", &losses_a);
    log_loss_series("climbmix_mixed_fleet_medium_peer_b", &losses_b);
    log_loss_series("climbmix_mixed_fleet_medium_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
    assert!(
        (1..=ordered.len()).contains(&train_receipts),
        "browser training receipts should flush at least once and at most once per window"
    );
    assert!(
        (1..=ordered.len()).contains(&verify_receipts),
        "browser validation receipts should flush at least once and at most once per window"
    );
}

#[test]
#[ignore = "scale rung"]
fn nca_native_peer_medium_model_converges_over_more_windows() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-medium");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(
            &root.path().join("nca-cache"),
            &nca_config_path,
            MEDIUM_SPEC,
        ),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-medium"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("scale".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-medium".into()),
            microshards: Some(8),
            max_records: Some(96),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let losses = run_training_windows(&prepared, 6);
    log_loss_series("nca_native_scale", &losses);
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(
        losses.iter().copied().fold(f64::INFINITY, f64::min) <= losses[0] - 0.5,
        "medium NCA rung should show a material best-window improvement"
    );
}

#[test]
#[ignore = "scale rung"]
fn climbmix_native_three_peers_medium_model_stays_consistent() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-medium");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 48, 16);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), MEDIUM_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("scale".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let losses_a = run_training_windows(&peer_a, 4);
    let losses_b = run_training_windows(&peer_b, 4);
    let losses_c = run_training_windows(&peer_c, 4);
    log_loss_series("climbmix_native_scale_peer_a", &losses_a);
    log_loss_series("climbmix_native_scale_peer_b", &losses_b);
    log_loss_series("climbmix_native_scale_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
}

#[test]
#[ignore = "large rung"]
fn nca_native_peer_large_model_converges_over_more_windows() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-large");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, LARGE_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-large"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("large".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-large".into()),
            microshards: Some(8),
            max_records: Some(128),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };

    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    let losses = run_training_windows(&prepared, 8);
    log_loss_series("nca_native_large", &losses);
    assert!(losses.iter().all(|loss| loss.is_finite()));
    assert!(
        losses.iter().copied().fold(f64::INFINITY, f64::min) <= losses[0] - 0.5,
        "large NCA rung should show a material improvement over the initial window"
    );
}

#[test]
#[ignore = "large rung"]
fn climbmix_native_three_peers_large_model_stays_consistent() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-large");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 64, 24);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), LARGE_SPEC),
    );

    let base_native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("large".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root.clone(),
            http_upstream: None,
        }),
    };
    let peer_a =
        prepare_climbmix_native_cpu(&base_native, Some(&dummy_auth_bundle())).expect("peer a");
    let mut peer_b_config = base_native.clone();
    peer_b_config.storage_root = root.path().join("storage-peer-b");
    let peer_b =
        prepare_climbmix_native_cpu(&peer_b_config, Some(&dummy_auth_bundle())).expect("peer b");
    let mut peer_c_config = base_native.clone();
    peer_c_config.storage_root = root.path().join("storage-peer-c");
    let peer_c =
        prepare_climbmix_native_cpu(&peer_c_config, Some(&dummy_auth_bundle())).expect("peer c");

    let losses_a = run_training_windows(&peer_a, 5);
    let losses_b = run_training_windows(&peer_b, 5);
    let losses_c = run_training_windows(&peer_c, 5);
    log_loss_series("climbmix_native_large_peer_a", &losses_a);
    log_loss_series("climbmix_native_large_peer_b", &losses_b);
    log_loss_series("climbmix_native_large_peer_c", &losses_c);
    assert!(losses_a.iter().all(|loss| loss.is_finite()));
    assert!(losses_b.iter().all(|loss| loss.is_finite()));
    assert!(losses_c.iter().all(|loss| loss.is_finite()));
    assert!(losses_a.iter().copied().fold(f64::INFINITY, f64::min) <= losses_a[0] + 0.5);
    assert!(losses_b.iter().copied().fold(f64::INFINITY, f64::min) <= losses_b[0] + 0.5);
    assert!(losses_c.iter().copied().fold(f64::INFINITY, f64::min) <= losses_c[0] + 0.5);
}

#[test]
#[ignore = "edge-backed deployment rung"]
fn nca_edge_drill_native_and_browser_github_auth_and_receipts() {
    let root = tempdir().expect("root");
    let nca_config_path = root.path().join("nca.toml");
    let training_config_path = root.path().join("nca-train.toml");
    let shard_root = root.path().join("nca-shards-edge");
    write(&nca_config_path, &nca_corpus_config_toml(root.path()));
    write(
        &training_config_path,
        &nca_training_config_toml(&root.path().join("nca-cache"), &nca_config_path, SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-native"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("edge-drill".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: Some(DragonShardExportConfig {
            root: shard_root,
            dataset_name: Some("dragon-nca-edge".into()),
            microshards: Some(4),
            max_records: Some(32),
            http_upstream: None,
        }),
        existing_shard_dataset: None,
    };
    let prepared = prepare_nca_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    run_edge_drill_for_prepared(&prepared, "nca-edge");
}

#[test]
#[ignore = "edge-backed deployment rung"]
fn climbmix_edge_drill_native_and_browser_github_auth_and_receipts() {
    let root = tempdir().expect("root");
    let shard_root = root.path().join("climbmix-shards-edge");
    fs::create_dir_all(&shard_root).expect("mkdir shards");
    write_existing_climbmix_shards(&shard_root, 24, 8);
    let training_config_path = root.path().join("climbmix-train.toml");
    write(
        &training_config_path,
        &climbmix_training_config_toml(&root.path().join("climbmix-cache"), SMALL_SPEC),
    );

    let native = DragonNativePeerConfig {
        training_config_paths: vec![training_config_path],
        storage_root: root.path().join("storage-edge-peer-a"),
        network: Default::default(),
        target: None,
        identity: Default::default(),
        bootstrap_peers: Vec::new(),
        manifest: native_manifest_seed(),
        app_semver: semver::Version::new(0, 5, 0),
        git_commit: Some("edge-drill".into()),
        enabled_features_label: Some("native-cpu".into()),
        auth: None,
        capability_policy: Default::default(),
        shard_export: None,
        existing_shard_dataset: Some(DragonExistingShardDatasetConfig {
            root: shard_root,
            http_upstream: None,
        }),
    };
    let prepared = prepare_climbmix_native_cpu(&native, Some(&dummy_auth_bundle())).expect("peer");
    run_edge_drill_for_prepared(&prepared, "climbmix-edge");
}
