use std::sync::{Mutex, OnceLock};

#[derive(Clone, Copy, Debug, Default)]
pub struct TrainProfileSnapshot {
    pub dataloader_cpu_ns: u128,
    pub dataloader_image_load_ns: u128,
    pub dataloader_image_transform_ns: u128,
    pub dataloader_teacher_load_ns: u128,
    pub dataloader_tensor_copy_ns: u128,
    pub dataloader_host_to_device_copy_bytes: u128,
    pub host_sync_points: u64,
    pub forward_ns: u128,
    pub loss_backward_ns: u128,
    pub optimizer_ns: u128,
    pub train_steps: u64,
    pub optimizer_steps: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct TrainProfileState {
    dataloader_cpu_ns: u128,
    dataloader_image_load_ns: u128,
    dataloader_image_transform_ns: u128,
    dataloader_teacher_load_ns: u128,
    dataloader_tensor_copy_ns: u128,
    dataloader_host_to_device_copy_bytes: u128,
    host_sync_points: u64,
    forward_ns: u128,
    loss_backward_ns: u128,
    optimizer_ns: u128,
    train_steps: u64,
    optimizer_steps: u64,
}

static TRAIN_PROFILE: OnceLock<Mutex<TrainProfileState>> = OnceLock::new();

pub fn enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

pub fn sync_timing_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE_SYNC").is_some()
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
            dataloader_image_load_ns: profile.dataloader_image_load_ns,
            dataloader_image_transform_ns: profile.dataloader_image_transform_ns,
            dataloader_teacher_load_ns: profile.dataloader_teacher_load_ns,
            dataloader_tensor_copy_ns: profile.dataloader_tensor_copy_ns,
            dataloader_host_to_device_copy_bytes: profile.dataloader_host_to_device_copy_bytes,
            host_sync_points: profile.host_sync_points,
            forward_ns: profile.forward_ns,
            loss_backward_ns: profile.loss_backward_ns,
            optimizer_ns: profile.optimizer_ns,
            train_steps: profile.train_steps,
            optimizer_steps: profile.optimizer_steps,
        };
    }
    TrainProfileSnapshot::default()
}

pub fn record_dataloader(
    cpu_ns: u128,
    image_load_ns: u128,
    image_transform_ns: u128,
    teacher_load_ns: u128,
    tensor_copy_ns: u128,
    host_to_device_copy_bytes: u128,
    host_sync_points: u64,
) {
    record(|profile| {
        profile.dataloader_cpu_ns = profile.dataloader_cpu_ns.saturating_add(cpu_ns);
        profile.dataloader_image_load_ns = profile
            .dataloader_image_load_ns
            .saturating_add(image_load_ns);
        profile.dataloader_image_transform_ns = profile
            .dataloader_image_transform_ns
            .saturating_add(image_transform_ns);
        profile.dataloader_teacher_load_ns = profile
            .dataloader_teacher_load_ns
            .saturating_add(teacher_load_ns);
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

pub fn record_optimizer_step(optimizer_ns: u128) {
    record(|profile| {
        profile.optimizer_ns = profile.optimizer_ns.saturating_add(optimizer_ns);
        profile.optimizer_steps = profile.optimizer_steps.saturating_add(1);
    });
}
