use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::Result;
use burn::tensor::backend::AutodiffBackend;
use burn_p2p::{
    ControlHandle, NodeTelemetrySnapshot, RunningNode, RuntimeStatus, SelectedWorkloadProject,
    TelemetryHandle,
};

use crate::capability_state::persist_native_downgrade;
use crate::config::DragonNativeTarget;
use crate::experiments::common::{DragonProjectFamily, PreparedNativePeer};

const MONITOR_POLL_INTERVAL: Duration = Duration::from_millis(500);
const DROP_SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

pub struct ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    prepared: Option<PreparedNativePeer<B>>,
    running: Option<RunningNode<SelectedWorkloadProject<DragonProjectFamily<B>>>>,
    stop_flag: Arc<AtomicBool>,
    monitor_thread: Option<JoinHandle<()>>,
}

impl<B> ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn stop_and_join(&mut self, timeout: Option<Duration>) -> Result<()> {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(running) = self.running.take() {
            let _ = running.shutdown();
            match timeout {
                Some(timeout) => {
                    let _ = running.await_termination_timeout(timeout)?;
                }
                None => {
                    let _ = running.await_termination()?;
                }
            }
        }
        if let Some(handle) = self.monitor_thread.take() {
            let _ = handle.join();
        }
        Ok(())
    }

    pub fn prepared(&self) -> &PreparedNativePeer<B> {
        self.prepared
            .as_ref()
            .expect("managed native peer should retain prepared peer")
    }

    pub fn telemetry(&self) -> TelemetryHandle {
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .telemetry()
    }

    pub fn control_handle(&self) -> ControlHandle {
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .control_handle()
    }

    pub fn snapshot(&self) -> NodeTelemetrySnapshot {
        self.telemetry().snapshot()
    }

    pub fn shutdown(&self) -> Result<()> {
        self.stop_flag.store(true, Ordering::SeqCst);
        self.running
            .as_ref()
            .expect("managed native peer should retain running node")
            .shutdown()
    }

    pub fn await_termination(mut self) -> Result<PreparedNativePeer<B>> {
        self.stop_and_join(None)?;
        Ok(self
            .prepared
            .take()
            .expect("managed native peer should retain prepared peer"))
    }

    pub fn await_termination_timeout(mut self, timeout: Duration) -> Result<PreparedNativePeer<B>> {
        self.stop_and_join(Some(timeout))?;
        Ok(self
            .prepared
            .take()
            .expect("managed native peer should retain prepared peer"))
    }
}

impl<B> Drop for ManagedRunningNativePeer<B>
where
    B: AutodiffBackend + Clone + 'static,
{
    fn drop(&mut self) {
        let _ = self.stop_and_join(Some(DROP_SHUTDOWN_TIMEOUT));
    }
}

pub fn spawn_prepared_native_peer<B>(
    prepared: PreparedNativePeer<B>,
) -> Result<ManagedRunningNativePeer<B>>
where
    B: AutodiffBackend + Clone + 'static,
{
    let running = prepared.builder.clone().spawn()?;
    let stop_flag = Arc::new(AtomicBool::new(false));
    let monitor_thread = if prepared.target_decision.can_train {
        let stop_flag_for_thread = Arc::clone(&stop_flag);
        let telemetry = running.telemetry();
        let storage_root = prepared.storage_root.clone();
        let experiment_kind = prepared.experiment_kind;
        let backend_label = prepared.backend_label.clone();
        let model_config = prepared.model_config.clone();
        let batch_size = prepared.config.training.batch_size;
        let block_size = prepared.config.training.block_size;
        let footprint = prepared.footprint.clone();
        let trainer_budget_bytes = prepared.target_decision.trainer_memory_budget_bytes;
        let downgrade_to = downgrade_target(prepared.target_decision.effective_target);
        Some(
            thread::Builder::new()
                .name("dragon-native-capability-monitor".into())
                .spawn(move || {
                    let mut persisted = false;
                    while !stop_flag_for_thread.load(Ordering::SeqCst) {
                        let snapshot = telemetry.snapshot();
                        if let Some(error) = snapshot.last_error.as_deref()
                            && is_probable_training_fit_failure(error)
                        {
                            let _ = persist_native_downgrade(
                                &storage_root,
                                experiment_kind,
                                &backend_label,
                                &model_config,
                                batch_size,
                                block_size,
                                &footprint,
                                trainer_budget_bytes,
                                downgrade_to,
                                error,
                                "runtime-monitor",
                            );
                            persisted = true;
                            break;
                        }
                        if snapshot.status == RuntimeStatus::Failed {
                            break;
                        }
                        thread::sleep(MONITOR_POLL_INTERVAL);
                    }

                    if !persisted {
                        let snapshot = telemetry.snapshot();
                        if let Some(error) = snapshot.last_error.as_deref()
                            && is_probable_training_fit_failure(error)
                        {
                            let _ = persist_native_downgrade(
                                &storage_root,
                                experiment_kind,
                                &backend_label,
                                &model_config,
                                batch_size,
                                block_size,
                                &footprint,
                                trainer_budget_bytes,
                                downgrade_to,
                                error,
                                "runtime-monitor-final",
                            );
                        }
                    }
                })?,
        )
    } else {
        None
    };

    Ok(ManagedRunningNativePeer {
        prepared: Some(prepared),
        running: Some(running),
        stop_flag,
        monitor_thread,
    })
}

fn downgrade_target(target: DragonNativeTarget) -> &'static str {
    match target {
        DragonNativeTarget::Reducer => "reducer",
        DragonNativeTarget::Validator => "validator",
        DragonNativeTarget::Auto | DragonNativeTarget::Trainer => "validator",
    }
}

fn is_probable_training_fit_failure(message: &str) -> bool {
    let message = message.to_ascii_lowercase();
    [
        "out of memory",
        "oom",
        "vram",
        "device lost",
        "failed to allocate",
        "insufficient memory",
        "allocation failed",
        "allocator",
        "cuda error",
        "webgpu",
    ]
    .iter()
    .any(|needle| message.contains(needle))
}

#[cfg(test)]
mod tests {
    use super::is_probable_training_fit_failure;

    #[test]
    fn fit_failure_classifier_catches_memory_signals() {
        assert!(is_probable_training_fit_failure(
            "CUDA error: out of memory while allocating optimizer state"
        ));
        assert!(is_probable_training_fit_failure(
            "webgpu device lost after failed to allocate buffer"
        ));
        assert!(!is_probable_training_fit_failure(
            "authentication failed: peer certificate rejected"
        ));
    }
}
