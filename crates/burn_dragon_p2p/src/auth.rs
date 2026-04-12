use std::collections::BTreeSet;
#[cfg(feature = "native")]
use std::fs;
#[cfg(feature = "native")]
use std::path::{Path, PathBuf};

#[cfg(feature = "native")]
use anyhow::Context;
use anyhow::{Result, anyhow, bail};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p::ProjectFamilyId;
use burn_p2p::{
    AuthConfig, AuthProvider, BrowserEdgeSnapshot, ClientReleaseManifest, EdgeAuthClient,
    EdgeEnrollmentConfig, ExperimentDirectoryEntry, ExperimentScope, LoginStart, PrincipalSession,
};
#[cfg(feature = "native")]
use burn_p2p::{ContentId, EdgePeerIdentity, create_peer_auth_envelope};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_browser::durability::{load_durable_browser_storage, persist_durable_browser_storage};
#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
use burn_p2p_browser::{
    BrowserEdgeClient, BrowserEnrollmentConfig, BrowserSessionState, BrowserUiBindings,
};
use chrono::Utc;
#[cfg(feature = "native")]
use libp2p_identity::Keypair;
use serde::{Deserialize, Serialize};

use crate::config::DragonNativeAuthBundle;

fn github_provider_for_snapshot(
    snapshot: &BrowserEdgeSnapshot,
) -> Result<&burn_p2p::BrowserLoginProvider> {
    snapshot
        .login_providers
        .iter()
        .find(|provider| {
            provider.label.to_ascii_lowercase().contains("github")
                || provider.login_path.to_ascii_lowercase().contains("github")
        })
        .ok_or_else(|| anyhow!("edge snapshot does not advertise a GitHub login provider"))
}

pub fn native_github_enrollment_config(
    snapshot: &BrowserEdgeSnapshot,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<EdgeEnrollmentConfig> {
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot is missing a trust bundle"))?;
    let provider = github_provider_for_snapshot(snapshot)?;

    if !snapshot.allowed_target_artifact_hashes.is_empty()
        && !snapshot
            .allowed_target_artifact_hashes
            .contains(&release_manifest.target_artifact_hash)
    {
        bail!(
            "release target artifact {} is not approved by the edge",
            release_manifest.target_artifact_hash.as_str()
        );
    }

    Ok(EdgeEnrollmentConfig {
        network_id: snapshot.network_id.clone(),
        project_family_id: trust_bundle.project_family_id.clone(),
        release_train_hash: snapshot
            .required_release_train_hash
            .clone()
            .unwrap_or_else(|| trust_bundle.required_release_train_hash.clone()),
        target_artifact_id: release_manifest.target_artifact_id.clone(),
        target_artifact_hash: release_manifest.target_artifact_hash.clone(),
        login_path: provider.login_path.clone(),
        device_path: provider.device_path.clone(),
        callback_path: provider.callback_path.clone().unwrap_or_default(),
        enroll_path: snapshot.paths.enroll_path.clone(),
        trust_bundle_path: snapshot.paths.trust_bundle_path.clone(),
        requested_scopes,
        session_ttl_secs,
    })
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DragonPendingGitHubLogin {
    pub edge_base_url: String,
    pub enrollment: EdgeEnrollmentConfig,
    pub login: LoginStart,
}

#[derive(Clone, Debug)]
pub struct DragonGitHubSession {
    pub auth: DragonNativeAuthBundle,
    pub session: PrincipalSession,
}

#[cfg(feature = "native")]
fn identity_key_path(storage_root: &Path) -> PathBuf {
    storage_root.join("state").join("identity.key")
}

#[cfg(feature = "native")]
fn load_or_generate_node_keypair(storage_root: &Path) -> Result<Keypair> {
    let path = identity_key_path(storage_root);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    if path.is_file() {
        let bytes =
            fs::read(&path).with_context(|| format!("failed to read {}", path.display()))?;
        return Keypair::from_protobuf_encoding(&bytes)
            .map_err(|error| anyhow!("failed to decode {}: {error}", path.display()));
    }
    let keypair = Keypair::generate_ed25519();
    let bytes = keypair
        .to_protobuf_encoding()
        .map_err(|error| anyhow!("failed to encode identity keypair: {error}"))?;
    fs::write(&path, bytes).with_context(|| format!("failed to write {}", path.display()))?;
    Ok(keypair)
}

#[cfg(feature = "native")]
fn edge_peer_identity_for_storage(
    storage_root: &Path,
    client_policy_hash: Option<ContentId>,
) -> Result<(Keypair, EdgePeerIdentity)> {
    let keypair = load_or_generate_node_keypair(storage_root)?;
    let peer_id = burn_p2p::PeerId::new(
        libp2p_identity::PeerId::from_public_key(&keypair.public()).to_string(),
    );
    let public_key_hex = hex::encode(keypair.public().encode_protobuf());
    let identity = EdgePeerIdentity {
        peer_id,
        peer_public_key_hex: public_key_hex,
        serial: 1,
        client_policy_hash,
    };
    Ok((keypair, identity))
}

#[cfg(feature = "native")]
pub async fn fetch_edge_snapshot(edge_base_url: &str) -> Result<BrowserEdgeSnapshot> {
    reqwest::Client::new()
        .get(format!(
            "{}/portal/snapshot",
            edge_base_url.trim_end_matches('/')
        ))
        .send()
        .await?
        .error_for_status()?
        .json::<BrowserEdgeSnapshot>()
        .await
        .map_err(Into::into)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn fetch_edge_snapshot(edge_base_url: &str) -> Result<BrowserEdgeSnapshot> {
    gloo_net::http::Request::get(&format!(
        "{}/portal/snapshot",
        edge_base_url.trim_end_matches('/')
    ))
    .send()
    .await
    .map_err(|error| anyhow!("failed to fetch edge snapshot: {error}"))?
    .json::<BrowserEdgeSnapshot>()
    .await
    .map_err(|error| anyhow!("failed to decode edge snapshot: {error}"))
}

pub async fn begin_native_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    principal_hint: Option<String>,
    use_device_flow: bool,
) -> Result<DragonPendingGitHubLogin> {
    let snapshot = fetch_edge_snapshot(edge_base_url).await?;
    let enrollment = native_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let client = EdgeAuthClient::new(edge_base_url, enrollment.clone());
    let login = if use_device_flow {
        client.begin_device_login(principal_hint).await?
    } else {
        client.begin_login(principal_hint).await?
    };
    if !matches!(login.provider, AuthProvider::GitHub) {
        bail!("edge login flow did not return a GitHub provider");
    }
    Ok(DragonPendingGitHubLogin {
        edge_base_url: edge_base_url.trim_end_matches('/').to_owned(),
        enrollment,
        login,
    })
}

#[cfg(feature = "native")]
pub async fn complete_native_github_login(
    storage_root: &Path,
    pending: &DragonPendingGitHubLogin,
    provider_code: &str,
    client_manifest_id: Option<ContentId>,
) -> Result<DragonGitHubSession> {
    let client = EdgeAuthClient::new(&pending.edge_base_url, pending.enrollment.clone());
    let session = client
        .complete_provider_login(&pending.login, provider_code.to_owned())
        .await?;
    if !matches!(session.claims.provider, AuthProvider::GitHub) {
        bail!("completed native session is not GitHub-authenticated");
    }
    let (node_keypair, identity) = edge_peer_identity_for_storage(storage_root, None)?;
    let certificate = client
        .enroll(&client.build_enrollment_request(&session, &identity))
        .await?;
    let trust_bundle_endpoint = format!(
        "{}{}",
        pending.edge_base_url.trim_end_matches('/'),
        pending.enrollment.trust_bundle_path
    );
    let peer_auth = create_peer_auth_envelope(
        &node_keypair,
        certificate,
        client_manifest_id,
        pending.enrollment.requested_scopes.clone(),
        ContentId::new(format!(
            "github-auth-{}",
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        )),
        Utc::now(),
    )?;
    Ok(DragonGitHubSession {
        auth: DragonNativeAuthBundle {
            auth_config: AuthConfig::new()
                .with_local_peer_auth(peer_auth)
                .with_trust_bundle_endpoint(trust_bundle_endpoint.clone()),
            trust_bundle_endpoint,
            edge_base_url: Some(pending.edge_base_url.clone()),
            session_id: Some(session.session_id.as_str().to_owned()),
            principal_id: Some(session.claims.principal_id.as_str().to_owned()),
        },
        session,
    })
}

pub fn compose_auth_config(
    base: Option<AuthConfig>,
    github_auth: Option<&DragonNativeAuthBundle>,
    experiment_directory: &[ExperimentDirectoryEntry],
) -> AuthConfig {
    let mut auth = base.unwrap_or_default();
    if let Some(github_auth) = github_auth {
        auth.local_peer_auth = github_auth.auth_config.local_peer_auth.clone();
        auth.trust_bundle_endpoints = github_auth.auth_config.trust_bundle_endpoints.clone();
    }
    auth.experiment_directory = experiment_directory.to_vec();
    auth
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn browser_github_enrollment_config(
    snapshot: &BrowserEdgeSnapshot,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
) -> Result<BrowserEnrollmentConfig> {
    let trust_bundle = snapshot
        .trust_bundle
        .as_ref()
        .ok_or_else(|| anyhow!("edge snapshot is missing a trust bundle"))?;
    let provider = github_provider_for_snapshot(snapshot)?;
    Ok(BrowserEnrollmentConfig {
        network_id: snapshot.network_id.clone(),
        project_family_id: ProjectFamilyId::new(trust_bundle.project_family_id.as_str()),
        release_train_hash: snapshot
            .required_release_train_hash
            .clone()
            .unwrap_or_else(|| trust_bundle.required_release_train_hash.clone()),
        target_artifact_id: release_manifest.target_artifact_id.clone(),
        target_artifact_hash: release_manifest.target_artifact_hash.clone(),
        login_path: provider.login_path.clone(),
        callback_path: provider.callback_path.clone().unwrap_or_default(),
        enroll_path: snapshot.paths.enroll_path.clone(),
        trust_bundle_path: snapshot.paths.trust_bundle_path.clone(),
        requested_scopes,
        session_ttl_secs,
    })
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
const PENDING_GITHUB_LOGIN_KEY: &str = "burn-dragon-p2p.pending-github-login";

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
fn browser_storage() -> Result<web_sys::Storage> {
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .local_storage()
        .map_err(|error| anyhow!("localStorage unavailable: {error:?}"))?
        .ok_or_else(|| anyhow!("localStorage unavailable"))
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn begin_browser_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    principal_hint: Option<String>,
) -> Result<LoginStart> {
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let client = BrowserEdgeClient::new(BrowserUiBindings::new(edge_base_url), enrollment);
    let login = client.begin_login(principal_hint).await?;
    if !matches!(login.provider, AuthProvider::GitHub) {
        bail!("browser login flow did not return a GitHub provider");
    }
    browser_storage()?
        .set_item(PENDING_GITHUB_LOGIN_KEY, &serde_json::to_string(&login)?)
        .map_err(|error| anyhow!("failed to persist pending login: {error:?}"))?;
    Ok(login)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn complete_browser_github_login(
    edge_base_url: &str,
    release_manifest: &ClientReleaseManifest,
    requested_scopes: BTreeSet<ExperimentScope>,
    session_ttl_secs: i64,
    provider_code: &str,
) -> Result<BrowserSessionState> {
    let storage = browser_storage()?;
    let pending = storage
        .get_item(PENDING_GITHUB_LOGIN_KEY)
        .map_err(|error| anyhow!("failed to read pending login: {error:?}"))?
        .ok_or_else(|| anyhow!("missing pending GitHub login state"))?;
    let login: LoginStart = serde_json::from_str(&pending)?;
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    let enrollment = browser_github_enrollment_config(
        &snapshot,
        release_manifest,
        requested_scopes,
        session_ttl_secs,
    )?;
    let client = BrowserEdgeClient::new(BrowserUiBindings::new(edge_base_url), enrollment);
    let session = client
        .complete_provider_login(&login, provider_code.to_owned())
        .await?;
    if !matches!(session.claims.provider, AuthProvider::GitHub) {
        bail!("completed browser session is not GitHub-authenticated");
    }
    let trust_bundle = client.fetch_trust_bundle().await.ok();
    let mut durable = load_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to load durable browser storage: {error}"))?;
    durable.session = BrowserSessionState {
        session: Some(session),
        certificate: None,
        trust_bundle,
        enrolled_at: Some(Utc::now()),
        reenrollment_required: false,
    };
    persist_durable_browser_storage(&snapshot.network_id, &durable)
        .await
        .map_err(|error| anyhow!("failed to persist durable browser storage: {error}"))?;
    let _ = storage.remove_item(PENDING_GITHUB_LOGIN_KEY);
    Ok(durable.session)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub async fn load_browser_session(edge_base_url: &str) -> Result<BrowserSessionState> {
    let snapshot = BrowserEdgeClient::new(
        BrowserUiBindings::new(edge_base_url),
        BrowserEnrollmentConfig::for_runtime_sync(&fetch_edge_snapshot(edge_base_url).await?),
    )
    .fetch_browser_edge_snapshot()
    .await?;
    Ok(load_durable_browser_storage(&snapshot.network_id)
        .await
        .map_err(|error| anyhow!("failed to load durable browser storage: {error}"))?
        .session)
}

#[cfg(all(feature = "wasm-ui", target_arch = "wasm32"))]
pub fn provider_code_from_window_location() -> Option<String> {
    let search = web_sys::window()?.location().search().ok()?;
    let query = search.strip_prefix('?').unwrap_or(&search);
    url::form_urlencoded::parse(query.as_bytes())
        .find_map(|(key, value)| (key == "code").then(|| value.into_owned()))
}
