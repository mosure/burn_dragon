use anyhow::{Result, anyhow, bail};
use burn_p2p::{AuthProvider, ExperimentId, ExperimentScope};
use burn_p2p_app::{
    AuthSessionCard, LifecycleAssignmentStatusCard, RuntimeCapabilityCard, TrainingResultPanel,
    TransportHealthPanel,
};
use burn_p2p_browser::{BrowserAppConnectConfig, BrowserAppController, BrowserSessionState};
use burn_p2p_views::{
    BrowserAppClientView, ContributionIdentityPanel, LifecycleAssignmentStatusView,
    RuntimeCapabilitySummaryView, TrainingResultSummaryView,
};
use dioxus::prelude::*;

use crate::auth::{
    begin_browser_github_login, complete_browser_github_login, fetch_edge_snapshot,
    load_browser_session, provider_code_from_window_location,
};
use crate::capability::{decide_browser_capability, detect_browser_host_capabilities};
use crate::capability_state::apply_browser_downgrade_state;
use crate::config::{DragonBrowserAppConfig, DragonPeerNetworkConfig};
#[cfg(feature = "wasm-peer")]
use crate::profile::{
    DragonExperimentProfile, browser_training_config_from_profile, find_matching_entry,
};
#[cfg(feature = "wasm-peer")]
use crate::wasm::training::{
    DragonBrowserTrainingResult, run_browser_training_with_release_manifest,
};

#[cfg(feature = "wasm-peer")]
pub mod training;

#[cfg(feature = "wasm-peer")]
fn browser_backend_label(config: &crate::config::DragonBrowserTrainingConfig) -> &'static str {
    config.execution_backend.backend_label()
}

fn window_query_string() -> Result<String> {
    Ok(web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .location()
        .search()
        .map_err(|error| anyhow!("failed to inspect browser query params: {error:?}"))?)
}

fn config_with_window_network_overrides(
    config: &DragonBrowserAppConfig,
) -> Result<DragonBrowserAppConfig> {
    let query = window_query_string()?;
    Ok(config.clone().with_network_overrides(
        DragonPeerNetworkConfig::parse_edge_base_url_query(&query),
        DragonPeerNetworkConfig::parse_seed_node_query(&query),
    ))
}

fn resolved_edge_base_url(config: &DragonBrowserAppConfig) -> Result<String> {
    config_with_window_network_overrides(config)?
        .effective_edge_base_url()
        .map(str::to_owned)
        .ok_or_else(|| anyhow!("no edge base URL configured"))
}

fn connect_config(config: &DragonBrowserAppConfig) -> Result<BrowserAppConnectConfig> {
    let config = config_with_window_network_overrides(config)?;
    let capability_decision = match config.training.as_ref() {
        Some(training) => apply_browser_downgrade_state(
            &resolved_edge_base_url(&config)?,
            training,
            browser_backend_label(training),
            decide_browser_capability(Some(training), &detect_browser_host_capabilities()),
        ),
        None => decide_browser_capability(None, &detect_browser_host_capabilities()),
    };
    let connect = BrowserAppConnectConfig::new(
        resolved_edge_base_url(&config)?,
        capability_decision.capability,
        capability_decision.connect_target,
    );
    if let Some((experiment_id, revision_id)) = config.selected_experiment() {
        Ok(connect.with_selection(experiment_id, revision_id))
    } else {
        Ok(connect)
    }
}

#[cfg(feature = "wasm-peer")]
async fn resolve_browser_training_config(
    config: &DragonBrowserAppConfig,
) -> Result<crate::config::DragonBrowserTrainingConfig> {
    if let Some(mut training) = config.training.clone() {
        if training.training_lease.is_none() {
            training.training_lease = active_training_lease(config).await?;
        }
        return Ok(training);
    }

    let edge_base_url = resolved_edge_base_url(config)?;
    let snapshot = fetch_edge_snapshot(&edge_base_url).await?;
    let entry = find_matching_entry(
        &snapshot.directory.entries,
        config.selected_experiment_id.as_deref(),
        config.selected_revision_id.as_deref(),
        None,
    )
    .ok_or_else(|| anyhow!("no Dragon experiment entry was found on the current edge"))?;
    let profile = DragonExperimentProfile::from_entry_metadata(entry)?
        .ok_or_else(|| anyhow!("selected experiment does not publish a Dragon training profile"))?;
    let mut training = browser_training_config_from_profile(entry, &profile)?.ok_or_else(|| {
        anyhow!("selected experiment does not publish a browser training profile")
    })?;
    training.training_lease = active_training_lease(config).await?;
    Ok(training)
}

#[cfg(feature = "wasm-peer")]
async fn active_training_lease(
    config: &DragonBrowserAppConfig,
) -> Result<Option<burn_p2p::WorkloadTrainingLease>> {
    let controller = BrowserAppController::connect_with(connect_config(config)?).await?;
    Ok(controller.active_training_lease().cloned())
}

fn auth_provider_label(provider: &AuthProvider) -> String {
    match provider {
        AuthProvider::GitHub => "GitHub".into(),
        AuthProvider::Oidc { issuer } => format!("OIDC ({issuer})"),
        AuthProvider::OAuth { provider } => format!("OAuth ({provider})"),
        AuthProvider::External { authority } => format!("External ({authority})"),
        AuthProvider::Static { authority } => format!("Static ({authority})"),
    }
}

fn session_identity_panel(
    session: Option<&BrowserSessionState>,
) -> Option<ContributionIdentityPanel> {
    let claims = session?.session.as_ref()?.claims.clone();
    let scoped_experiments = claims
        .granted_scopes
        .into_iter()
        .filter_map(|scope| match scope {
            ExperimentScope::Train { experiment_id }
            | ExperimentScope::Validate { experiment_id }
            | ExperimentScope::Archive { experiment_id } => Some(experiment_id),
            ExperimentScope::Connect
            | ExperimentScope::Discover
            | ExperimentScope::Admin { .. } => None,
        })
        .collect::<std::collections::BTreeSet<ExperimentId>>()
        .into_iter()
        .collect();
    Some(ContributionIdentityPanel {
        principal_id: claims.principal_id.as_str().into(),
        provider_label: auth_provider_label(&claims.provider),
        trust_badges: Vec::new(),
        scoped_experiments,
    })
}

fn runtime_capability_summary(view: &BrowserAppClientView) -> RuntimeCapabilitySummaryView {
    let backend_summary = if view.runtime_detail.is_empty() {
        view.capability_summary.clone()
    } else {
        format!("{} | {}", view.runtime_detail, view.capability_summary)
    };
    RuntimeCapabilitySummaryView {
        preferred_role: view.runtime_label.clone(),
        backend_summary,
        can_train: view.training.train_available || view.training.can_train,
    }
}

fn browser_runtime_role_label(role: &burn_p2p_browser::BrowserRuntimeRole) -> &'static str {
    match role {
        burn_p2p_browser::BrowserRuntimeRole::BrowserTrainerWgpu => "browser_trainer_wgpu",
        burn_p2p_browser::BrowserRuntimeRole::BrowserVerifier => "browser_verifier",
        burn_p2p_browser::BrowserRuntimeRole::BrowserObserver => "browser_observer",
        burn_p2p_browser::BrowserRuntimeRole::BrowserFallback => "browser_fallback",
        burn_p2p_browser::BrowserRuntimeRole::Viewer => "viewer",
    }
}

fn lifecycle_assignment_status(
    view: &BrowserAppClientView,
) -> Option<LifecycleAssignmentStatusView> {
    let experiment = view.selected_experiment.as_ref()?;
    let assignment_status = if view.training.active_assignment.is_some() {
        "active".into()
    } else if experiment.train_available {
        "train-ready".into()
    } else if experiment.validate_available {
        "validate-ready".into()
    } else {
        "viewer-only".into()
    };
    Some(LifecycleAssignmentStatusView {
        experiment_label: experiment.display_name.clone(),
        revision_label: experiment.revision_id.clone(),
        lifecycle_phase: view.default_surface.as_str().into(),
        assignment_status,
    })
}

fn latest_training_result_summary(
    view: &BrowserAppClientView,
) -> Option<TrainingResultSummaryView> {
    Some(TrainingResultSummaryView {
        artifact_id: view.training.last_artifact_id.clone()?,
        receipt_id: view.training.last_receipt_id.clone(),
        window_secs: view.training.last_window_secs.unwrap_or_default(),
    })
}

async fn ensure_github_session(
    config: &DragonBrowserAppConfig,
) -> Result<Option<BrowserSessionState>> {
    let session = load_browser_session(&resolved_edge_base_url(config)?).await?;
    if config.require_github_auth {
        let claims = session
            .session
            .as_ref()
            .ok_or_else(|| anyhow!("GitHub sign-in is required before joining this network"))?;
        if !matches!(claims.claims.provider, AuthProvider::GitHub) {
            bail!("browser session is not GitHub-authenticated");
        }
    }
    Ok(Some(session))
}

pub async fn connect_browser_app(config: &DragonBrowserAppConfig) -> Result<BrowserAppClientView> {
    let _ = ensure_github_session(config).await?;
    let controller = BrowserAppController::connect_with(connect_config(config)?).await?;
    Ok(controller.view())
}

pub async fn refresh_browser_app(config: &DragonBrowserAppConfig) -> Result<BrowserAppClientView> {
    let _ = ensure_github_session(config).await?;
    let mut controller = BrowserAppController::connect_with(connect_config(config)?).await?;
    controller.refresh().await.map_err(Into::into)
}

pub async fn resume_or_complete_browser_auth(
    config: &DragonBrowserAppConfig,
    release_manifest: &burn_p2p::ClientReleaseManifest,
) -> Result<Option<BrowserSessionState>> {
    let edge_base_url = resolved_edge_base_url(config)?;
    if let Some(provider_code) = provider_code_from_window_location() {
        let session = complete_browser_github_login(
            &edge_base_url,
            release_manifest,
            config.requested_scopes.clone(),
            3600,
            &provider_code,
        )
        .await?;
        return Ok(Some(session));
    }
    if config.require_github_auth {
        return load_browser_session(&edge_base_url).await.map(Some);
    }
    Ok(None)
}

pub async fn start_browser_github_auth(
    config: &DragonBrowserAppConfig,
    release_manifest: &burn_p2p::ClientReleaseManifest,
) -> Result<()> {
    let edge_base_url = resolved_edge_base_url(config)?;
    let login = begin_browser_github_login(
        &edge_base_url,
        release_manifest,
        config.requested_scopes.clone(),
        3600,
        None,
    )
    .await?;
    let authorize_url = login
        .authorize_url
        .ok_or_else(|| anyhow!("edge did not return a browser authorize URL"))?;
    web_sys::window()
        .ok_or_else(|| anyhow!("window unavailable"))?
        .location()
        .set_href(&authorize_url)
        .map_err(|error| anyhow!("failed to redirect to GitHub auth: {error:?}"))?;
    Ok(())
}

#[derive(Props, Clone, PartialEq)]
pub struct DragonBrowserAppProps {
    pub config: DragonBrowserAppConfig,
    pub release_manifest: burn_p2p::ClientReleaseManifest,
}

#[component]
pub fn DragonBrowserApp(props: DragonBrowserAppProps) -> Element {
    let initial_config = config_with_window_network_overrides(&props.config)
        .unwrap_or_else(|_| props.config.clone());
    let mut edge_url = use_signal(|| {
        initial_config
            .effective_edge_base_url()
            .unwrap_or_default()
            .to_owned()
    });
    let mut seed_node_urls = use_signal(|| initial_config.effective_seed_node_urls().join(", "));
    let status = use_signal(String::new);
    let current_view = use_signal(|| None::<BrowserAppClientView>);
    let session_state = use_signal(|| None::<BrowserSessionState>);
    #[cfg(feature = "wasm-peer")]
    let local_training = use_signal(|| None::<DragonBrowserTrainingResult>);

    let connect_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let mut status = status;
            let mut current_view = current_view;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Connecting…".into());
                match connect_browser_app(&next_config).await {
                    Ok(view) => {
                        current_view.set(Some(view));
                        let session = match resolved_edge_base_url(&next_config) {
                            Ok(edge_base_url) => load_browser_session(&edge_base_url).await.ok(),
                            Err(_) => None,
                        };
                        session_state.set(session);
                        status.set("Connected".into());
                    }
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let refresh_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let mut status = status;
            let mut current_view = current_view;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Refreshing…".into());
                match refresh_browser_app(&next_config).await {
                    Ok(view) => {
                        current_view.set(Some(view));
                        let session = match resolved_edge_base_url(&next_config) {
                            Ok(edge_base_url) => load_browser_session(&edge_base_url).await.ok(),
                            Err(_) => None,
                        };
                        session_state.set(session);
                        status.set("Refreshed".into());
                    }
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let github_login_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let mut status = status;
            spawn(async move {
                status.set("Starting GitHub sign-in…".into());
                if let Err(error) = start_browser_github_auth(&next_config, &release_manifest).await
                {
                    status.set(error.to_string());
                }
            });
        }
    };

    let complete_callback_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let mut status = status;
            let mut session_state = session_state;
            spawn(async move {
                status.set("Completing GitHub callback…".into());
                match resume_or_complete_browser_auth(&next_config, &release_manifest).await {
                    Ok(Some(session)) => {
                        session_state.set(Some(session));
                        status.set("GitHub session ready".into());
                    }
                    Ok(None) => status.set("No callback code found in URL".into()),
                    Err(error) => status.set(error.to_string()),
                }
            });
        }
    };

    let view = current_view.read().clone();
    let session_panel = session_identity_panel(session_state.read().as_ref());
    let callback_available = provider_code_from_window_location().is_some();
    let auth_required = props.config.require_github_auth;
    let browser_host_capabilities = detect_browser_host_capabilities();
    let browser_capability_decision = match (
        props.config.training.as_ref(),
        resolved_edge_base_url(&initial_config),
    ) {
        (Some(training), Ok(edge_base_url)) => apply_browser_downgrade_state(
            &edge_base_url,
            training,
            browser_backend_label(training),
            decide_browser_capability(Some(training), &browser_host_capabilities),
        ),
        (Some(training), Err(_)) => {
            decide_browser_capability(Some(training), &browser_host_capabilities)
        }
        (None, _) => decide_browser_capability(None, &browser_host_capabilities),
    };
    let browser_can_attempt_dynamic_training = props.config.training.is_some()
        || (browser_host_capabilities.navigator_gpu_exposed
            && browser_host_capabilities.worker_gpu_exposed
            && browser_host_capabilities.dedicated_worker_exposed);
    let capability_budget_label = browser_capability_decision
        .trainer_memory_budget_bytes
        .map(|bytes| format!("{} MiB", bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());
    let capability_footprint_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{} MiB", footprint.estimated_training_bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());
    let capability_tokens_per_second_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{:.1}", footprint.estimated_tokens_per_second))
        .unwrap_or_else(|| "n/a".into());
    let capability_window_label = browser_capability_decision
        .training_budget
        .as_ref()
        .map(|budget| budget.max_window_secs.to_string())
        .unwrap_or_else(|| "n/a".into());
    let capability_checkpoint_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| {
            format!(
                "{} MiB",
                footprint.estimated_checkpoint_bytes / (1024 * 1024)
            )
        })
        .unwrap_or_else(|| "n/a".into());
    let capability_shard_label = browser_capability_decision
        .footprint
        .as_ref()
        .map(|footprint| format!("{} MiB", footprint.estimated_shard_bytes / (1024 * 1024)))
        .unwrap_or_else(|| "n/a".into());

    #[cfg(feature = "wasm-peer")]
    let train_action = {
        let props = props.clone();
        move |_| {
            let mut next_config = props.config.clone();
            next_config = next_config.with_network_overrides(
                Some(edge_url.read().clone()),
                DragonPeerNetworkConfig::parse_seed_node_list(&seed_node_urls.read()),
            );
            let release_manifest = props.release_manifest.clone();
            let mut status = status;
            let mut current_view = current_view;
            let mut local_training = local_training;
            spawn(async move {
                let training = match resolve_browser_training_config(&next_config).await {
                    Ok(training) => training,
                    Err(error) => {
                        status.set(error.to_string());
                        return;
                    }
                };
                status.set("Running browser training…".into());
                let edge_base_url = match resolved_edge_base_url(&next_config) {
                    Ok(edge_base_url) => edge_base_url,
                    Err(error) => {
                        status.set(error.to_string());
                        return;
                    }
                };
                match run_browser_training_with_release_manifest(
                    &edge_base_url,
                    &training,
                    &release_manifest,
                )
                .await
                {
                    Ok(result) => {
                        status.set(format!(
                            "Browser training complete: mean train loss {:.4}",
                            result.train_loss_mean
                        ));
                        local_training.set(Some(result));
                        if let Ok(view) = refresh_browser_app(&next_config).await {
                            current_view.set(Some(view));
                        }
                    }
                    Err(error) => {
                        status.set(error.to_string());
                        if let Ok(view) = refresh_browser_app(&next_config).await {
                            current_view.set(Some(view));
                        }
                    }
                }
            });
        }
    };

    #[cfg(feature = "wasm-peer")]
    let train_button = {
        let has_training_config =
            resolved_edge_base_url(&initial_config).is_ok() && browser_can_attempt_dynamic_training;
        rsx! {
            if has_training_config {
                button { onclick: train_action, "Run Browser Training" }
            }
        }
    };
    #[cfg(not(feature = "wasm-peer"))]
    let train_button = rsx! {};

    #[cfg(feature = "wasm-peer")]
    let local_training_section = if let Some(result) = local_training.read().clone() {
        let eval_loss_label = result
            .eval_loss
            .map(|value| format!("{value:.4}"))
            .unwrap_or_else(|| "n/a".into());
        let tokens_per_second_label = result
            .tokens_per_second
            .map(|value| format!("{value:.1}"))
            .unwrap_or_else(|| "n/a".into());
        let train_batches_label = result.train_batches.to_string();
        rsx! {
            section {
                h2 { "Local Browser Training" }
                p { "Experiment: ", {result.experiment_kind_label} }
                p { "Backend: ", {result.backend} }
                p { "Train loss: ", {format!("{:.4}", result.train_loss_mean)} }
                p { "Eval loss: ", {eval_loss_label} }
                p { "Train batches: ", {train_batches_label} }
                p { "Tokens/sec: ", {tokens_per_second_label} }
                if let Some(live) = result.live_participant {
                    p { "Receipt accepted: ", {live.receipt_submission_accepted.to_string()} }
                    p { "Accepted receipts: ", {live.accepted_receipt_ids.join(", ")} }
                    p { "Runtime state: ", {live.runtime_state.unwrap_or_else(|| "n/a".into())} }
                }
            }
        }
    } else {
        rsx! {}
    };
    #[cfg(not(feature = "wasm-peer"))]
    let local_training_section = rsx! {};

    rsx! {
        div {
            class: "burn-dragon-p2p-app",
            h1 { "burn_dragon p2p" }
            p { "Browser peer for NCA and ClimbMix experiment networks." }
            div {
                label { "Edge URL" }
                input {
                    value: "{edge_url}",
                    oninput: move |event| edge_url.set(event.value()),
                }
            }
            div {
                label { "Seed Node URLs" }
                input {
                    value: "{seed_node_urls}",
                    oninput: move |event| seed_node_urls.set(event.value()),
                }
            }
            div {
                button { onclick: connect_action, "Connect" }
                button { onclick: refresh_action, "Refresh" }
                if auth_required {
                    button { onclick: github_login_action, "GitHub Sign-In" }
                }
                if callback_available {
                    button { onclick: complete_callback_action, "Complete Callback" }
                }
                {train_button}
            }
            p { "{status}" }
            if let Some(reason) = browser_capability_decision.downgrade_reason.clone() {
                p { "Capability policy: {reason}" }
            }
            if props.config.training.is_some() {
                section {
                    h2 { "Local Trainer Capability" }
                    p { "Recommended role: {browser_runtime_role_label(&browser_capability_decision.capability.recommended_role)}" }
                    p { "Estimated training footprint: {capability_footprint_label}" }
                    p { "Trainer memory budget: {capability_budget_label}" }
                    p { "Estimated tokens/sec: {capability_tokens_per_second_label}" }
                    p { "Checkpoint budget: {capability_checkpoint_label}" }
                    p { "Shard budget: {capability_shard_label}" }
                    p { "Window budget secs: {capability_window_label}" }
                }
            }
            section {
                h2 { "Session" }
                AuthSessionCard { session: session_panel }
            }
            {local_training_section}
            if let Some(view) = view {
                section {
                    h2 { "Runtime" }
                    RuntimeCapabilityCard { summary: runtime_capability_summary(&view) }
                }
                if let Some(status) = lifecycle_assignment_status(&view) {
                    section {
                        h2 { "Assignment" }
                        LifecycleAssignmentStatusCard { status: status }
                    }
                }
                section {
                    h2 { "Network Training" }
                    TrainingResultPanel { result: latest_training_result_summary(&view) }
                    p { "Last loss: {view.training.last_loss.clone().unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Throughput: {view.training.throughput_summary.clone().unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Optimizer steps: {view.training.optimizer_steps.map(|v| v.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                    p { "Accepted samples: {view.training.accepted_samples.map(|v| v.to_string()).unwrap_or_else(|| \"n/a\".into())}" }
                }
                section {
                    h2 { "Network" }
                    TransportHealthPanel { network: view.network.clone() }
                }
                section {
                    h2 { "Leaderboard" }
                    ul {
                        for entry in view.viewer.leaderboard_preview {
                            li { "{entry.label}: {entry.score} ({entry.receipts} receipts)" }
                        }
                    }
                }
            }
        }
    }
}
