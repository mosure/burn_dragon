#[cfg(not(target_arch = "wasm32"))]
use std::sync::Once;

#[cfg(not(target_arch = "wasm32"))]
use burn_wgpu::{RuntimeOptions, graphics};

#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn init_wgpu_test_runtime(device: &burn_wgpu::WgpuDevice) {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        burn_wgpu::init_setup::<graphics::AutoGraphicsApi>(device, RuntimeOptions::default());
    });
}
