#[cfg(not(target_arch = "wasm32"))]
use std::sync::{Mutex, MutexGuard, Once, OnceLock};

#[cfg(not(target_arch = "wasm32"))]
use burn_wgpu::{RuntimeOptions, graphics};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn init_wgpu_test_runtime(device: &burn_wgpu::WgpuDevice) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn wgpu_test_guard() -> MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .expect("wgpu test lock")
}
