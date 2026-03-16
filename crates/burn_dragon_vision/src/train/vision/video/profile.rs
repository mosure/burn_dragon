//! Stable video profiling bridge over the intermediate `video_lejepa` implementation.
//!
//! This module intentionally exposes only the profiling surface that current benches and tooling
//! need, while keeping the training model implementation private in `models.rs`.

use std::sync::{LazyLock, Mutex};

#[derive(Clone, Copy, Debug, Default)]
pub struct VisionVideoTrainProfileSnapshot {
    pub train_calls: u64,
    pub split_projection_ns: u64,
    pub context_rollout_ns: u64,
    pub predict_rollout_ns: u64,
    pub loss_heads_ns: u64,
}

static VIDEO_TRAIN_PROFILE: LazyLock<Mutex<VisionVideoTrainProfileSnapshot>> =
    LazyLock::new(|| Mutex::new(VisionVideoTrainProfileSnapshot::default()));

#[inline]
pub(super) fn video_train_profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

#[inline]
pub(super) fn video_train_profile_record(f: impl FnOnce(&mut VisionVideoTrainProfileSnapshot)) {
    if !video_train_profile_enabled() {
        return;
    }
    if let Ok(mut state) = VIDEO_TRAIN_PROFILE.lock() {
        f(&mut state);
    }
}

pub fn video_train_profile_reset() {
    if let Ok(mut state) = VIDEO_TRAIN_PROFILE.lock() {
        *state = VisionVideoTrainProfileSnapshot::default();
    }
}

pub fn video_train_profile_snapshot() -> VisionVideoTrainProfileSnapshot {
    VIDEO_TRAIN_PROFILE
        .lock()
        .map(|state| *state)
        .unwrap_or_default()
}
