#[cfg(feature = "train")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "train")]
pub(crate) fn device_allocation_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

#[cfg(feature = "train")]
pub(crate) fn pin_stream_zero() {
    // CubeCL stream ids are thread-local by default; forcing stream 0 prevents
    // per-thread allocator fragmentation when training work hops threads.
    unsafe {
        burn_fusion::stream::StreamId::swap(burn_fusion::stream::StreamId { value: 0 });
    }
}
