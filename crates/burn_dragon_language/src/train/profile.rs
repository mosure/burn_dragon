use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainProfileSnapshot {
    pub dataloader_cpu_ns: u128,
    pub dataloader_tensor_copy_ns: u128,
    pub dataloader_host_to_device_copy_bytes: u128,
    pub host_sync_points: u64,
    pub forward_ns: u128,
    pub loss_backward_ns: u128,
    pub embed_probe_ns: u128,
    pub first_layer_forward_probe_ns: u128,
    pub first_layer_probe_ns: u128,
    pub logits_loss_probe_ns: u128,
    pub hidden_logits_loss_probe_ns: u128,
    pub hidden_model_forward_probe_ns: u128,
    pub hidden_model_probe_ns: u128,
    pub detail_probe_steps: u64,
    pub train_steps: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainProfileState {
    dataloader_cpu_ns: u128,
    dataloader_tensor_copy_ns: u128,
    dataloader_host_to_device_copy_bytes: u128,
    host_sync_points: u64,
    forward_ns: u128,
    loss_backward_ns: u128,
    embed_probe_ns: u128,
    first_layer_forward_probe_ns: u128,
    first_layer_probe_ns: u128,
    logits_loss_probe_ns: u128,
    hidden_logits_loss_probe_ns: u128,
    hidden_model_forward_probe_ns: u128,
    hidden_model_probe_ns: u128,
    detail_probe_steps: u64,
    train_steps: u64,
}

static TRAIN_PROFILE: OnceLock<Mutex<TrainProfileState>> = OnceLock::new();

pub fn enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

pub fn detail_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE_DETAIL").is_some()
}

fn state() -> &'static Mutex<TrainProfileState> {
    TRAIN_PROFILE.get_or_init(|| Mutex::new(TrainProfileState::default()))
}

fn record(mutator: impl FnOnce(&mut TrainProfileState)) {
    if let Ok(mut profile) = state().lock() {
        mutator(&mut profile);
    }
}

pub fn reset() {
    if let Ok(mut profile) = state().lock() {
        *profile = TrainProfileState::default();
    }
}

pub fn snapshot() -> TrainProfileSnapshot {
    if let Ok(profile) = state().lock() {
        return TrainProfileSnapshot {
            dataloader_cpu_ns: profile.dataloader_cpu_ns,
            dataloader_tensor_copy_ns: profile.dataloader_tensor_copy_ns,
            dataloader_host_to_device_copy_bytes: profile.dataloader_host_to_device_copy_bytes,
            host_sync_points: profile.host_sync_points,
            forward_ns: profile.forward_ns,
            loss_backward_ns: profile.loss_backward_ns,
            embed_probe_ns: profile.embed_probe_ns,
            first_layer_forward_probe_ns: profile.first_layer_forward_probe_ns,
            first_layer_probe_ns: profile.first_layer_probe_ns,
            logits_loss_probe_ns: profile.logits_loss_probe_ns,
            hidden_logits_loss_probe_ns: profile.hidden_logits_loss_probe_ns,
            hidden_model_forward_probe_ns: profile.hidden_model_forward_probe_ns,
            hidden_model_probe_ns: profile.hidden_model_probe_ns,
            detail_probe_steps: profile.detail_probe_steps,
            train_steps: profile.train_steps,
        };
    }
    TrainProfileSnapshot::default()
}

pub fn record_dataloader(
    cpu_ns: u128,
    tensor_copy_ns: u128,
    host_to_device_copy_bytes: u128,
    host_sync_points: u64,
) {
    record(|profile| {
        profile.dataloader_cpu_ns = profile.dataloader_cpu_ns.saturating_add(cpu_ns);
        profile.dataloader_tensor_copy_ns = profile
            .dataloader_tensor_copy_ns
            .saturating_add(tensor_copy_ns);
        profile.dataloader_host_to_device_copy_bytes = profile
            .dataloader_host_to_device_copy_bytes
            .saturating_add(host_to_device_copy_bytes);
        profile.host_sync_points = profile.host_sync_points.saturating_add(host_sync_points);
    });
}

pub fn record_train_step(forward_ns: u128, loss_backward_ns: u128) {
    record(|profile| {
        profile.forward_ns = profile.forward_ns.saturating_add(forward_ns);
        profile.loss_backward_ns = profile.loss_backward_ns.saturating_add(loss_backward_ns);
        profile.train_steps = profile.train_steps.saturating_add(1);
    });
}

pub fn record_detail_probe(
    embed_probe_ns: u128,
    first_layer_forward_probe_ns: u128,
    first_layer_probe_ns: u128,
    logits_loss_probe_ns: u128,
    hidden_logits_loss_probe_ns: u128,
    hidden_model_forward_probe_ns: u128,
    hidden_model_probe_ns: u128,
) {
    record(|profile| {
        profile.embed_probe_ns = profile.embed_probe_ns.saturating_add(embed_probe_ns);
        profile.first_layer_forward_probe_ns = profile
            .first_layer_forward_probe_ns
            .saturating_add(first_layer_forward_probe_ns);
        profile.first_layer_probe_ns = profile
            .first_layer_probe_ns
            .saturating_add(first_layer_probe_ns);
        profile.logits_loss_probe_ns = profile
            .logits_loss_probe_ns
            .saturating_add(logits_loss_probe_ns);
        profile.hidden_logits_loss_probe_ns = profile
            .hidden_logits_loss_probe_ns
            .saturating_add(hidden_logits_loss_probe_ns);
        profile.hidden_model_forward_probe_ns = profile
            .hidden_model_forward_probe_ns
            .saturating_add(hidden_model_forward_probe_ns);
        profile.hidden_model_probe_ns = profile
            .hidden_model_probe_ns
            .saturating_add(hidden_model_probe_ns);
        profile.detail_probe_steps = profile.detail_probe_steps.saturating_add(1);
    });
}
