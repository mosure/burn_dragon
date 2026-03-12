use std::sync::{Mutex, OnceLock};

#[derive(Debug, Default, Clone, Copy)]
pub struct KernelProfileSnapshot {
    pub calls: u64,
    pub launches: u64,
    pub total_ns: u128,
    pub setup_ns: u128,
    pub copy_ns: u128,
    pub dispatch_ns: u128,
    pub transient_allocations: u64,
    pub metadata_upload_bytes: u64,
    pub metadata_reuse_hits: u64,
    pub metadata_reuse_bytes: u64,
    pub resident_rollout_steps: u64,
}

#[derive(Debug, Default)]
pub(crate) struct KernelProfileState {
    pub(crate) calls: u64,
    pub(crate) launches: u64,
    pub(crate) total_ns: u128,
    pub(crate) setup_ns: u128,
    pub(crate) copy_ns: u128,
    pub(crate) dispatch_ns: u128,
    pub(crate) transient_allocations: u64,
    pub(crate) metadata_upload_bytes: u64,
    pub(crate) metadata_reuse_hits: u64,
    pub(crate) metadata_reuse_bytes: u64,
    pub(crate) resident_rollout_steps: u64,
}

pub(crate) struct KernelProfileSite {
    inner: OnceLock<Mutex<KernelProfileState>>,
}

impl KernelProfileSite {
    pub const fn new() -> Self {
        Self {
            inner: OnceLock::new(),
        }
    }

    fn state(&self) -> &Mutex<KernelProfileState> {
        self.inner
            .get_or_init(|| Mutex::new(KernelProfileState::default()))
    }
}

pub(crate) fn profile_enabled() -> bool {
    std::env::var_os("BDH_STAGE_PROFILE").is_some()
}

pub(crate) fn profile_reset(site: &'static KernelProfileSite) {
    if let Ok(mut state) = site.state().lock() {
        *state = KernelProfileState::default();
    }
}

pub(crate) fn profile_snapshot(site: &'static KernelProfileSite) -> KernelProfileSnapshot {
    if let Ok(state) = site.state().lock() {
        return KernelProfileSnapshot {
            calls: state.calls,
            launches: state.launches,
            total_ns: state.total_ns,
            setup_ns: state.setup_ns,
            copy_ns: state.copy_ns,
            dispatch_ns: state.dispatch_ns,
            transient_allocations: state.transient_allocations,
            metadata_upload_bytes: state.metadata_upload_bytes,
            metadata_reuse_hits: state.metadata_reuse_hits,
            metadata_reuse_bytes: state.metadata_reuse_bytes,
            resident_rollout_steps: state.resident_rollout_steps,
        };
    }
    KernelProfileSnapshot::default()
}

pub(crate) fn profile_record(
    site: &'static KernelProfileSite,
    mutator: impl FnOnce(&mut KernelProfileState),
) {
    if let Ok(mut state) = site.state().lock() {
        mutator(&mut state);
    }
}
